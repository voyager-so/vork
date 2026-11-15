#ifdef VORK_PATH_CONTEXT
#ifdef VORK_PROPAGATION
// Sample launch solid angle, then follow the same authored segments as beauty.
// Its PDF stays in launch coordinates: no inverse bending map or endpoint
// Jacobian is assumed. Small emitters can have high variance.
// Finite analytic lights keep their own shadow links and shadow-enable flag.
// Evaluate their shape along the authored path; no straight endpoint PDF is used.
float3 segment_analytic<V:IVariant,P:IProgram>(float3 origin,float3 direction,uint light,uint transparent,DielectricSegment optical,float optical_length,inout uint rng VORK_GAUSSIAN_SOURCE_PARAMETER VORK_WORK_PARAMETER) {
    let l=light_values<V>(light);
    let flags=lights[light].shadow_offsets[REG_LIGHT_RAY_WORD];
    let shadowed=(flags&REG_LIGHT_RAY_SHADOWED)!=0u;
    var escaped=false;var previous_distance=0.;var throughput=float3(1.);
    for(uint work=0u;work<REG_PROPAGATION_WORK_LIMIT;work++) {
        float limit;float3 transfer;
        if(!propagation_segment(origin,direction,escaped,previous_distance,limit,transfer)) return float3(0.);
        throughput*=transfer;
        let source=emitter_eval_li<V>(l,origin,direction);
        let found=source.pdf>0. && source.distance<=limit;
        let distance=found ? source.distance : limit;
        throughput*=straight_participating_transmittance(origin,direction,distance,rng VORK_WORK_ARGUMENT)*segment_attenuation(optical,distance);
        if(shadowed) throughput*=straight_shadow_transmittance<V,P>(origin,direction,distance,lights[light].shadow_offsets[REG_LIGHT_RAY_LINK_WORD],(flags&REG_LIGHT_RAY_SHADOWED_BY_ALL)!=0u,transparent,optical VORK_GAUSSIAN_SOURCE_ARGUMENT(launch_point) VORK_WORK_ARGUMENT);
        if(found) return throughput*path_spectrum(source.radiance,true)*connection_arrival(optical_length,distance,optical.ior);
        if(limit==REG_FP32_MAX || !any(throughput>float3(0.))) return float3(0.);
        origin+=direction*limit;optical_length+=limit*optical.ior;
    }
    path_boundary_error(REG_PATH_PROPAGATION_ERROR,3u);return float3(0.);
}
float3 segment_connection<V:IVariant,P:IProgram>(float3 origin,float3 direction,uint event,uint groups,uint transparent,DielectricSegment optical,float optical_length,float surface_weight,inout uint rng VORK_GAUSSIAN_SOURCE_PARAMETER VORK_WORK_PARAMETER) {
    var escaped=false;var previous_distance=0.;var throughput=float3(1.);var radiance=float3(0.);var pending=0.;
    for(uint work=0u;work<REG_PROPAGATION_WORK_LIMIT;work++) {
        if(pending<=0.) {
            float3 transfer;
            if(!propagation_segment(origin,direction,escaped,previous_distance,pending,transfer)) return radiance;
            throughput*=transfer;
        }
        let limit=pending;
        let hit=trace<V,P>(origin,direction,limit,event VORK_GAUSSIAN_SOURCE_ARGUMENT(launch_point) VORK_WORK_ARGUMENT);
        let stop=hit.t>=0. ? hit.t : limit;
        throughput*=straight_participating_transmittance(origin,direction,stop,rng VORK_WORK_ARGUMENT)*segment_attenuation(optical,stop);
        if(!any(throughput>float3(0.))) return radiance;
        if(hit.t<0.) {
            if(limit==REG_FP32_MAX) return radiance+throughput*path_environment(direction,event,groups)*surface_weight*path_arrival(REG_FP32_MAX);
            origin+=direction*limit;optical_length+=limit*optical.ior;pending=0.;continue;
        }
        optical_length+=hit.t*optical.ior;
        float3 face;bool entering;float weight;HitSurface shading;
        let c=path_closure<V,P>(hit,direction,P.lobes ? path_random(rng) : 0.,P.lit ? pixel_hash(rng) : 0u,face,entering,weight,shading);
#ifndef VORK_NO_GAUSSIANS
        launch_point=uint2(shading.v.object,shading.v.primitive);
#endif
        if(c.geometry_opacity>=1. || path_random(rng)<c.geometry_opacity) {
            let emitted=material_covers_side<V>(shading.v,shading.entering) ? path_spectrum(emitter_mesh_radiance<V,P>(shading),true) : float3(0.);
            radiance+=throughput*emitted*surface_weight*path_arrival(optical_length);
            throughput*=shading.s.is_hair ? float3(0.) : transport_null_transmittance<V>(c,-direction,optical,shading.s.geometric,face)*weight;
            if(!any(throughput>float3(0.))) return radiance;
        }
        if(transparent++>=path_settings.limits.z) return radiance;
        origin=origin_offset(hit.object,shading.v.world,face,direction);
        pending=max(0.,pending-hit.t-length(origin-shading.v.world));
    }
    path_boundary_error(REG_PATH_PROPAGATION_ERROR,3u);return float3(0.);
}
float3 segment_direction(float2 u) {
    let z=1.-2.*u.x;let r=sqrt(max(0.,1.-z*z));let phi=REG_TAU*u.y;
    return float3(r*cos(phi),z,r*sin(phi));
}
// The ordinary light shape is only a proposal in launch coordinates. The
// uniform half preserves support when bending moves the source's apparent image.
float segment_light_pdf<V:IVariant>(LightValues light,float3 world,float3 direction) {
    return .5*(1./(4.*REG_PI)+max(0.,emitter_eval_li<V>(light,world,direction).pdf));
}
float3 segment_light_direction<V:IVariant>(LightValues light,float3 world,float3 proposed,float proposed_pdf,inout uint rng,out float pdf) {
    var direction=proposed;
    if(path_random(rng)<.5) {
        if(proposed_pdf<=0.) {pdf=0.;return float3(0.);}
    } else direction=segment_direction(float2(path_random(rng),path_random(rng)));
    pdf=segment_light_pdf<V>(light,world,direction);
    return direction;
}
float3 segment_next_event<V:IVariant,P:IProgram>(HitSurface receiver,Closure c,float3 view,float3 world,float3 error,float3 geometric,DielectricInterface media,uint groups,uint continuation,uint transparent,TransportPreparation prepared,PathSampler sampler,float optical_length,inout uint rng VORK_WORK_PARAMETER) {
    let direction=segment_direction(path_environment(sampler));let pdf=1./(4.*REG_PI);
    if(receiver.s.is_hair) return float3(0.);
    var result=float3(0.);
    for(uint i=0u;i<3u;i++) {
        let event=i==0u ? REG_RAY_DIFFUSE : (i==1u ? REG_RAY_SPECULAR : REG_RAY_TRANSMISSION);
        if((event!=REG_RAY_DIFFUSE || !c.geometry_thin_walled) && (event==REG_RAY_TRANSMISSION)!=(dot(geometric,direction)<0.)) continue;
        let response=transport_eval_mask(c,view,direction,event,prepared);
        if(!any(response>float3(0.))) continue;
        let bsdf_pdf=(continuation&event)!=0u ? transport_pdf_mask(c,view,direction,event,prepared) : 0.;
        let optical=dot(geometric,direction)<0. ? media.transmitted : media.incident;
        result+=response/pdf*segment_connection<V,P>(path_origin(receiver,world,error,direction),direction,event,groups,transparent,optical,optical_length,mis_power(pdf,bsdf_pdf),rng VORK_GAUSSIAN_SOURCE_ARGUMENT((uint2(receiver.v.object,receiver.v.primitive))) VORK_WORK_ARGUMENT);
    }
    return result;
}
#endif

#endif
