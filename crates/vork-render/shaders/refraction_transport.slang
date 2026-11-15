// Interface-by-interface raster continuation. Geometry, occupancy and tint are
// shared with path transport; local receiver lighting uses the forward evaluator.
import vork_constants;
import frame;
import material;
import surface;
import stream;
import variant;
import forward;
import rays;
import vork_ray_alpha;
import hit;
import emitter_mesh;
import ray_coverage;
import hit_lighting;
import openpbr_core;
import dielectric_scene;
import path_boundary;
import camera_config;
import bsdf;
import path_sampler;
#ifndef VORK_REFRACTION_BOUNDARIES
#define VORK_REFRACTION_BOUNDARIES 0
#endif
// Sampled secondary lighting uses physical source connections. Explicit LTC
// keeps main's filtered map approximation at the actual receiver, never the
// camera slot's coverage offsets (which belong to a different surface).
struct RefractionShadows<V:IVariant,P:IProgram> : IShadows {
    static const bool probe_rays=true;
    static bool probe_visible(float3 world,float3 normal,float3 error,float3 probe) {return RasterShadows<V>.probe_visible(world,normal,error,probe);}
    static const bool subsurface=V.subsurface;
    static uint integration_count(uint index,out float3 normalization) {let count=index==0xffffffffu ? sample_count() : 1u;normalization=float3(float(count));return count;}
    static float3 integration_sample(uint tap,inout uint index,inout float3 world) {if(index==0xffffffffu) {index=tap;}return float3(1.);}
    static uint sample_count() {return RasterShadows<V>.sample_count();}
    static float thickness(uint object,float3 world,float3 direction) {return RasterShadows<V>.thickness(object,world,direction);}
    static float3 visibility_direction(float3 world,float3 normal,LightValues record,uint light,float3 direction,float distance,float3 error,uint index,uint count,bool fibre=false) {return hit_shadow_direction<P,ReceiverGeometry>(ReceiverGeometry(ray_alpha,ray_hit_buffers),ray_scene,ray_alpha,light,world,normal,error,direction,distance,index,count);}
}
DielectricHit refraction_boundary<V:IVariant,P:IProgram>(float3 origin,float3 direction,float2 pixel,uint event,uint only_object=0xffffffffu) {
    RayDesc ray;ray.Origin=origin;ray.Direction=direction;ray.TMin=0.;ray.TMax=REG_FP32_MAX;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(ray_scene,RAY_FLAG_FORCE_NON_OPAQUE,event,ray);
    // Two triangles sharing an edge can report the same t. Commit order follows
    // the acceleration structure's traversal, which differs between builds, so a
    // tie keeps the lower (object, primitive): the same surface every frame.
    // Every triangle is a candidate here, so the choice never depends on order.
    DielectricHit h={};h.t=-1.;
    while(query.Proceed()) {
        if(query.CandidateType()!=CANDIDATE_NON_OPAQUE_TRIANGLE) {continue;}
        let candidate=RayIntersection(query.CandidateInstanceIndex(),query.CandidateInstanceID(),query.CandidatePrimitiveIndex(),query.CandidateTriangleBarycentrics());
        if(only_object!=0xffffffffu && candidate.instance_custom_data!=only_object) continue;
        let owner=V.has_solid_transmission && instance_has_solid_owner<ReceiverGeometry>(ReceiverGeometry(ray_alpha,ray_hit_buffers),candidate.instance_index,camera_config.boundaries.xy);
        if(!candidate_accepts<P,ReceiverGeometry>(ReceiverGeometry(ray_alpha,ray_hit_buffers),ray_alpha,candidate,query.CandidateTriangleFrontFace(),query.CandidateWorldToObject3x4(),direction,pixel,owner)) {continue;}
        let t=query.CandidateTriangleRayT();
        if(h.t>=0. && (t>h.t || (t==h.t && (candidate.instance_custom_data>h.object || (candidate.instance_custom_data==h.object && candidate.primitive_index>=h.primitive))))) {continue;}
        query.CommitNonOpaqueTriangleHit();
        h.t=t;h.instance=candidate.instance_index;h.object=candidate.instance_custom_data;h.primitive=candidate.primitive_index;
        h.barycentrics=candidate.barycentrics;h.world_to_object=query.CandidateWorldToObject3x4();
    }
    return merge_field_hits<P,ReceiverGeometry>(ReceiverGeometry(ray_alpha,ray_hit_buffers),h,frame.field_geometry.xy,origin,direction,ray.TMax,event,only_object);
}
float3 refraction_environment(uint object,float3 direction,float roughness,uint event) {
    if(!light_receives(frame.environment_links.x,frame.environment_links.y,objects[object].metadata.x)
        || (frame.environment_links.x&event)==0u) {return float3(0.);}
    var value=frame.radiance.xyz;
    if(frame.environment.x>0.) {
        value+=frame.environment.x*frame.environment.y*frame.environment_color.rgb*env_specular.SampleLevel(env_sampler,environment_uv(direction,frame.environment_rotation),roughness*frame.environment.z).rgb;
    }
    return value*frame.radiance.w;
}
// Camera visibility owns the identity check: transmission and camera rays can
// select different opaque surfaces even when every interface is null.
interface IRefractionReceiver {
    static bool matches(DielectricHit hit,float3 world,float3 origin,float3 direction);
}
// Export replaceable diffuse terms only when an undeflected continuation ends
// on its retained camera receiver. Other hits keep their local estimate.
float3 refraction_transport<V:IVariant,P:IProgram,R:IRefractionReceiver,let N:uint>(Varying primary,Surface surface,Coordinates coordinates,float3 camera_origin,PathOccupancy<N> camera_state,float3 facet,float2 pixel,uint2 seed,uint sample_index,inout float3 receiver_ambient,inout float3 receiver_diffuse) {
    let tables=camera_config.boundaries.xy;
    var direction=-surface.view;
    var occupancy=camera_state;
    var identity=ray_alpha[camera_config.boundaries.z+primary.object];
    if(V.has_solid_transmission && surface.closure.transmission_weight>0. && !surface.closure.geometry_thin_walled) {validate_boundary<V,P>(primary,surface.front_facing,coordinates,surface);}
    var boundary=path_boundary(identity,surface.closure,material_record(primary.object));
    var v=primary;var s=surface;var geometric=facet;
    if(V.has_solid_transmission) reconcile_occupancy<ReceiverGeometry,N>(ReceiverGeometry(ray_alpha,ray_hit_buffers),occupancy,primary.world,identity,tables);
    var throughput=occupancy.attenuation(length(primary.world-camera_origin));var radiance=float3(0.);
    let camera_ior=occupancy.ior();
    var incoming_origin=camera_origin;var incoming_event=REG_RAY_CAMERA;
    DielectricHit current_hit={};current_hit.t=-1.;
    if(V.has_solid_transmission && N>1u) {
        let error=surface_position_error(primary.object,primary.world);
        let interval=error+abs(direction)*boundary_distance_error(error,facet,direction);
        if(boundary_contact_possible<ReceiverGeometry>(ReceiverGeometry(ray_alpha,ray_hit_buffers),identity,primary.world,interval,tables)) current_hit=refraction_boundary<V,P>(camera_origin,direction,pixel,REG_RAY_CAMERA,primary.object);
    }
    var events=0u;var undeflected=true;
    var sampler=path_sampler(seed,sample_index,frame.integrator.z,camera_config.seed);
    for(;;) {
        let c=s.closure;
        // The scene-wide proof removes unreachable solid transactions while
        // retaining the same thin-interface BSDF and sampler sequence.
        let solid=V.has_solid_transmission && !c.geometry_thin_walled && c.transmission_weight>0.;
        let incident=occupancy.current();
        let error=surface_position_error(v.object,v.world);
        BoundaryEvent<N> boundary_changes={};
        if(solid) boundary_changes=boundary_event<V,P,ReceiverGeometry,N>(ReceiverGeometry(ray_alpha,ray_hit_buffers),ray_scene,incoming_origin,direction,current_hit,v.world,error,geometric,boundary,s.front_facing,occupancy,tables,pixel,incoming_event);
        var transmitted=incident;if(solid) transmitted=boundary_changes.owner(occupancy);
        let false_hit=solid && incident.object==transmitted.object && incident.medium==transmitted.medium;
        BsdfSample sample;
        if(false_hit) {sample.direction=direction;sample.weight=float3(1.);sample.pdf=0.;sample.event=REG_RAY_TRANSMISSION;}
        else {
        var relative=s;
        relative.closure=boundary_closure(c,incident,transmitted,solid);
        // The closure is already relative to this incident medium. Its normal
        // faces the incident side, so the forward evaluator must not invert it.
        relative.front_facing=true;
        let site=hit_site(v,coordinates,geometric,seed,sample_index,true);
        float3 scattering;float3 indirect_scattering;
        // Continuation replaces this glossy baseline at every transmitting
        // interface. Omit the estimate instead of computing and subtracting it,
        // including builds without participating media.
        let glossy_ambient=c.transmission_weight<=0. || s.is_hair || (P.lit && P.lit_material(v));
        let lighting=AnalyticRadiance(REG_RAY_DIFFUSE|REG_RAY_SPECULAR,true,false,false,glossy_ambient,P.lit && P.lit_material(v));
        let local=analytic_radiance_at<V,RefractionShadows<V,P>>(v,relative,site,lighting,scattering,indirect_scattering);
        if(c.transmission_weight<=0. || s.is_hair || (P.lit && P.lit_material(v))) {
            if(undeflected && R.matches(current_hit,v.world,camera_origin,-surface.view)) {receiver_ambient+=throughput*local.baseline.diffuse;receiver_diffuse+=throughput*local.diffuse;}
            occupancy.clear();return radiance+throughput*local.color;
        }
        // Replace the environment specular estimate with the traced BSDF
        // continuation. Keep finite-light integration and diffuse ambient.
        radiance+=throughput*(local.color-reflection_bsdf_environment(local.reflection));
        let prepared=transport_prepare<V>(relative.closure,-direction,true,relative.geometric,geometric);
        sample=transport_sample(relative.closure,-direction,path_scatter(sampler).xyz,prepared);
        // Discrete sheet transmission and index-matched transmission preserve
        // direction. Roughness and reflective layers do not deflect that event.
        undeflected=undeflected && sample.event==REG_RAY_TRANSMISSION && sample.pdf==0.
            && (relative.closure.geometry_thin_walled || dielectric_ior(relative.closure)==1.);
        }
        if(!all(isfinite(sample.direction)) || !all(isfinite(sample.weight)) || !isfinite(sample.pdf)) {
            InterlockedMax(shader_errors[0],REG_RADIANCE_NONFINITE);occupancy.clear();return radiance;
        }
        if(sample.event!=REG_RAY_SPECULAR && sample.event!=REG_RAY_TRANSMISSION) {occupancy.clear();return radiance;}
        if((sample.event==REG_RAY_TRANSMISSION)!=(dot(geometric,sample.direction)<0.)) {occupancy.clear();return radiance;}
        let outgoing=sample.direction;
        throughput*=sample.weight;
        // The camera's reflection provider applies only outside occupied media.
        // Interior reflections must walk to an exit, even at the first interface.
        if(events==0u && incident.object==0xffffffffu && sample.event==REG_RAY_SPECULAR && frame.projection.z<REG_REFLECTIONS_RAY) {
            occupancy.clear();return radiance+(frame.projection.z==REG_REFLECTIONS_OFF ? float3(0.)
                : throughput*refraction_environment(v.object,outgoing,0.,REG_RAY_SPECULAR));
        }
        if(solid) boundary_changes.finish(occupancy,sample.event==REG_RAY_TRANSMISSION);
        if(!all(isfinite(throughput)) || !any(throughput>float3(0.))) {occupancy.clear();return radiance;}
        // Match path transport's eta-compensated roulette. There is no maximum
        // boundary count: surviving paths carry the discarded paths' energy.
        if(!false_hit && events++>=REG_PATH_ROULETTE_START) {
            let survive=path_survival(throughput,occupancy.ior()/camera_ior);
            if(path_roulette(sampler)>survive) {occupancy.clear();return radiance;}
            throughput/=survive;
        }
        var origin=surface_ray_origin(v.world,error,geometric,outgoing);
        direction=outgoing;
        let next_event=false_hit ? incoming_event : sample.event;
        for(;;) {
            let hit=refraction_boundary<V,P>(origin,direction,pixel,next_event);
            if(hit.t<0.) {occupancy.clear();return radiance+throughput*refraction_environment(v.object,direction,0.,next_event);}
            sampler.vertex++;let pick=path_scatter(sampler).w;
            var h=dielectric_surface<V,P,ReceiverGeometry>(ReceiverGeometry(ray_alpha,ray_hit_buffers),hit,direction,pixel,Lobe(select(P.lobes,LOBE_PICK,0u),pick));
            if(V.has_solid_transmission) reconcile_occupancy<ReceiverGeometry,N>(ReceiverGeometry(ray_alpha,ray_hit_buffers),occupancy,h.v.world,instance_solid<ReceiverGeometry>(ReceiverGeometry(ray_alpha,ray_hit_buffers),hit.instance,tables),tables);
            throughput*=occupancy.attenuation(hit.t);
            let u=path_opacity(sampler);
            let record=ray_alpha[hit.instance*REG_RAY_SLOT_WORDS+1u];
            let opacity=h.s.closure.geometry_opacity*objects[hit.object].opacity.x;
            if(u>=opacity) {origin=surface_ray_origin(h.v.world,surface_position_error(hit.object,h.v.world),h.face,direction);continue;}
            let solid_hit=V.has_solid_transmission && !h.s.closure.geometry_thin_walled && h.s.closure.transmission_weight>0.;
            incoming_origin=origin;incoming_event=next_event;current_hit=hit;
            boundary=solid_hit ? boundary_descriptor<V,P,ReceiverGeometry>(ReceiverGeometry(ray_alpha,ray_hit_buffers),hit,direction,h,tables,pixel) : path_boundary(0xffffffffu,h.s.closure);
            if(P.lit && P.lit_material(h.v)) {radiance+=throughput*emitter_mesh_radiance<V,P>(h)*frame.radiance.w;h.s=lit_surface<V,RefractionShadows<V,P>,P>(h.v,h.entering,h.k,h.s,hit_site(h.v,h.k,h.face,seed,sample_index,true));}
            if(P.lobes) {throughput*=h.s.weight;}
            v=h.v;s=h.s;coordinates=h.k;geometric=h.face;break;
        }
    }
}
