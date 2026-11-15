import material;
// Outgoing radiance at ray hits the camera cannot see (ray GI, ray reflections,
// probe rays) uses the same analytic surface radiance as forward shading.
import vork_constants;
import frame;
import closure;
import variant;
import openpbr_core;
import bsdf;
import vork_ray_alpha;
import vork_ray_visibility;
import vork_ray_sampling;
import surface;
import surface_lighting;
import vork_sampling;
import hit;
import ray_coverage;
import shadow_coverage;
import emitter_mesh;
import light_connection;
import emitter;
import vork_prelude;

// One shadow ray toward the light's centre, respecting shadowed_by selection.
// Retained volumes use
// spatial coverage, independent of the camera exposure that triggers an update.
float3 hit_shadow<P:IProgram,G:IHitGeometry,V:IVariant=Unspecialized>(G g,RaytracingAccelerationStructure scene,StructuredBuffer<uint> words,uint i,float3 world,float3 normal,float3 position_error,uint index,uint count,bool retained=false) {
    let ray_flags=lights[i].shadow_offsets[REG_LIGHT_RAY_WORD];

    if ((ray_flags&REG_LIGHT_RAY_SHADOWED)==0u) {return 1.;}
    let source=emitter_center_li<Unspecialized>(specialized_light<Unspecialized>(lights[i]),i,world,index);
    return hit_shadow_direction<P,G,V>(g,scene,words,i,world,normal,position_error,source.direction,source.distance,index,count,retained);
}
float3 hit_shadow_direction<P:IProgram,G:IHitGeometry,V:IVariant=Unspecialized>(G g,RaytracingAccelerationStructure scene,StructuredBuffer<uint> words,uint i,float3 world,float3 normal,float3 position_error,float3 direction,float distance,uint index,uint count,bool retained=false) {
    let ray_flags=lights[i].shadow_offsets[REG_LIGHT_RAY_WORD];
    if ((ray_flags&REG_LIGHT_RAY_SHADOWED)==0u) {return 1.;}
    let origin=ray_origin(world,normal,direction,position_error);
    let all_casters=(ray_flags&REG_LIGHT_RAY_SHADOWED_BY_ALL)!=0u;
    let groups=lights[i].shadow_offsets[REG_LIGHT_RAY_LINK_WORD];
    RayDesc ray;ray.Origin=origin;ray.Direction=direction;ray.TMin=0.;ray.TMax=max(distance-length(origin-world),0.);
    let fields=field_shadow_transmittance<P,G,V>(g,words,ray,groups,all_casters,index);
    // Use the same installed opacity/link proof as primary connections. No
    // candidate closure or containment evaluation is needed for opaque casters.
    if((ray_flags&REG_LIGHT_RAY_OPAQUE)!=0u) {
        return fields*select(opaque_shadow_hit(scene,origin,direction,ray.TMax,REG_RAY_SHADOW),0.,1.);
    }
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(scene,RAY_FLAG_FORCE_NON_OPAQUE,REG_RAY_SHADOW,ray);
    var result=ConnectionVisibility(fields,0u);
    while (query.Proceed()) {
        let candidate=RayIntersection(query.CandidateInstanceIndex(),query.CandidateInstanceID(),query.CandidatePrimitiveIndex(),query.CandidateTriangleBarycentrics());
        if (!all_casters && !links_intersect(objects[candidate.instance_custom_data].metadata.x,groups)) {continue;}
#ifdef VORK_NO_CONNECTION_TRANSMISSION
        // Scalar coverage is sufficient once installed closure proofs exclude
        // transmission. Keep half-open ownership for fractional shared edges.
        let inverse=query.CandidateWorldToObject3x4();
        if(!connection_owns_hit<G>(g,candidate,inverse,ray.Direction)) {continue;}
        result.transmittance*=1.-candidate_coverage<P,G>(g,words,candidate,inverse,ray.Direction,select(retained,float2(0.),frame.lens.zw));
#else
        result.transmittance*=shadow_candidate_transmittance<P,G,false,true,V>(g,scene,words,candidate,query.CandidateWorldToObject3x4(),ray,query.CandidateTriangleRayT(),select(retained,float2(0.),frame.lens.zw),index);
#endif
        if (!any(result.transmittance>float3(0.))) {return float3(0.);}
    }
    return result.transmittance*shadow_contained_transmittance<P,G,V>(g,scene,words,ray,all_casters,groups,select(retained,float2(0.),frame.lens.zw),index);
}
// World radiance along a ray that escapes the scene, before exposure (diffuse rays).
float3 world_radiance(float3 direction) {
    if ((frame.environment_links.x&REG_RAY_DIFFUSE)==0u) {return float3(0.);}
    var radiance=frame.radiance.xyz;
    if (frame.environment.x>0.) {
        radiance+=frame.environment.x*frame.environment.y*frame.environment_color.rgb*env_specular.SampleLevel(env_sampler,environment_uv(direction,frame.environment_rotation),0.).rgb;
    }
    return radiance;
}
// Final material semantics shared by GI, reflection and probe hits. Consumers
// supply the shadow policy and apply exposure after this scene-linear result.
#if defined(VORK_SURFACE_SLANG_OUTLINE)
[noinline]
#endif
float3 finalized_hit_radiance<V:IVariant,S:IShadows,P:IProgram>(HitSurface h,uint2 pixel,uint index,uint lobes) {
    float3 emission;
    return finalized_hit_radiance<V,S,P>(h,pixel,index,lobes,emission);
}
#if defined(VORK_SURFACE_SLANG_OUTLINE)
[noinline]
#endif
float3 finalized_hit_radiance<V:IVariant,S:IShadows,P:IProgram>(HitSurface h,uint2 pixel,uint index,uint lobes,out float3 emission,bool indirect_only=false,bool capture_endpoint=false,float2 analytic_source=float2(-1.,1.)) {
    emission=float3(0.);
    if(!indirect_only && !h.s.discarded) {emission=emitter_mesh_radiance<V,P>(h);}
    // Lit may encode reflected response; keep physical source radiance separate.
    h.s.closure.emission_luminance=0.;
    let lit=P.lit && P.lit_material(h.v);
    var first=h.s;
    if (lit && P.lobes) {first=surface_at<V,P>(h.v,h.entering,h.s.view,h.k,Lobe(0u,0.));}
    first.closure.emission_luminance=0.;
    let pre_count=select(lit,max(first.lobes,1u),0u);
    var lighting=float3(0.);var direct=float3(0.);
    var site=hit_site(h.v,h.k,h.face,pixel,index,true);
    // Shader-to-RGB first receives all original lobes, then its returned
    // closure is shaded. A retained loop shares the full lighting graph across
    // both phases without specializing away dynamic material dispatch.
    [loop] for(uint i=0u;i<=pre_count;i++) {
        let final=i==pre_count;
        var surface=first;
        if(final) {
            if(lit) {h.s=surface_response<V>(P.apply_lit(h.v,first,h.k,SurfaceLighting(lighting,first.closure.geometry_opacity,direct)));}
            surface=h.s;
        } else {
            if(i>0u) {surface=surface_at<V,P>(h.v,h.entering,h.s.view,h.k,Lobe(i,0.));}
            surface.closure.emission_luminance=0.;
        }
        float3 scattering;float3 indirect_scattering;
        let mask=select(final,select(surface.is_hair,REG_RAY_SPECULAR,lobes),REG_RAY_DIFFUSE|REG_RAY_SPECULAR);
        var policy=AnalyticRadiance(mask,!final,indirect_only && final,capture_endpoint);
        policy.exact_sources=lit && !final;
        policy.selected_source=int(analytic_source.x);policy.source_probability=analytic_source.y;
        let response=analytic_radiance_at<V,S>(h.v,surface,site,policy,scattering,indirect_scattering);
        if(final) {return response.color*(P.lobes ? surface.weight : 1.)+emission;}
        let color=response.color*surface.weight;let source=response.direct*surface.weight;
        if(i==0u) {lighting=color;direct=source;}
        else {lighting+=color;direct+=source;}
    }
    return emission;

}
// Consumers without an installed capability union retain the full evaluator.
float3 finalized_hit_radiance<S:IShadows,P:IProgram>(HitSurface h,uint2 pixel,uint index,uint lobes,out float3 emission,bool indirect_only=false,bool capture_endpoint=false) {
    return finalized_hit_radiance<Unspecialized,S,P>(h,pixel,index,lobes,emission,indirect_only,capture_endpoint);
}
