// Hit kernels own their query globals; imported raster helpers are eliminated.
#ifdef VORK_HIT_KERNEL
#define RASTER_REGISTER(slot)
#else
#define RASTER_REGISTER(slot) : register(slot)
#endif
// Masked visibility owns fragment storage in group 2. Its query-only
// specialization binds these same retained ray tables in group 4 instead.
#ifndef VORK_RAY_BINDING_GROUP
#define VORK_RAY_BINDING_GROUP 2
#endif
#ifndef VORK_RAY_SEQUENCE_REGISTER
#define VORK_RAY_SEQUENCE_REGISTER t11
#define VORK_RAY_SCENE_REGISTER t12
#define VORK_RAY_ALPHA_REGISTER t13
#ifdef VORK_METAL
#define VORK_RAY_HITS_REGISTER b14
#else
#define VORK_RAY_HITS_REGISTER t14
#endif
#endif
import vork_constants;
import light_connection;
#ifdef VORK_CAMERA_SPLIT
[[vk::binding(14,4)]] public StructuredBuffer<float> receiver_emitter_weights : register(t29);
#endif
import vork_sampling;
[[vk::binding(0,VORK_RAY_BINDING_GROUP)]] public StructuredBuffer<uint> shadow_sequence RASTER_REGISTER(VORK_RAY_SEQUENCE_REGISTER);
import frame;
import vork_ray_sampling;
import vork_ray_alpha;
#ifdef VORK_RAY_BINDINGS
[[vk::binding(1,VORK_RAY_BINDING_GROUP)]] public RaytracingAccelerationStructure ray_scene RASTER_REGISTER(VORK_RAY_SCENE_REGISTER);
[[vk::binding(2,VORK_RAY_BINDING_GROUP)]] public StructuredBuffer<uint> ray_alpha RASTER_REGISTER(VORK_RAY_ALPHA_REGISTER);
#else
[[vk::binding(1,VORK_RAY_BINDING_GROUP)]] public RaytracingAccelerationStructure ray_scene;
[[vk::binding(2,VORK_RAY_BINDING_GROUP)]] public StructuredBuffer<uint> ray_alpha;
#endif
export float3 ray_eye() {return frame.eye.xyz;}
import vork_shadow_ray;
export uint shadow_ray_groups(uint index) {return objects[index].metadata.x;}
export uint shadow_ray_material(uint index) {return objects[index].material_index;}
// One emitter ray for sample `sample_index` (the exposure sample unless given) of
// the frame's ray sequence, whose header holds its length.
public float3 ray_transmission(RaytracingAccelerationStructure scene,RayGeometry geometry,float3 world,float3 normal,LightValues light,uint light_index,uint2 pixel,float3 position_error,bool fibre=false,uint sample_override=0xffffffffu) {
    if ((lights[light_index].shadow_offsets[REG_LIGHT_RAY_WORD]&REG_LIGHT_RAY_SHADOWED)==0u) {return float3(1.);}
    let sample_index=select(sample_override==0xffffffffu,uint(frame.temporal.x),sample_override);
    let parameters=frame.integrator.w>>REG_INTEGRATOR_RAY_SEQUENCE_SHIFT;
    let u=ray_emitter_sample(shadow_sequence,parameters,pixel,light_index,sample_index,shadow_sequence[parameters]);
    return shared_ray_transmission(scene,geometry,frame,light,light_index,lights[light_index].shadow_offsets[REG_LIGHT_RAY_WORD],lights[light_index].shadow_offsets[REG_LIGHT_RAY_LINK_WORD],world,normal,sample_index,position_error,u,fibre);
}
// Use the exact connection selected by the BSDF/source estimator.
public float3 ray_visibility_direction(RaytracingAccelerationStructure scene,RayGeometry geometry,float3 world,float3 normal,uint light,float3 direction,float distance,float3 error,uint index,uint count,bool fibre=false) {
    let flags=lights[light].shadow_offsets[REG_LIGHT_RAY_WORD];
    if((flags&REG_LIGHT_RAY_SHADOWED)==0u) {return float3(1.);}
    LightConnection connection={};connection.direction=direction;connection.distance=distance;
    let receiver=Receiver(world,normal,error,0u,index,count,1u);
    let policy=ConnectionPolicy(lights[light].shadow_offsets[REG_LIGHT_RAY_LINK_WORD],(flags&REG_LIGHT_RAY_SHADOWED_BY_ALL)!=0u,REG_RAY_SHADOW,(flags&REG_LIGHT_RAY_OPAQUE)!=0u);
    return ray_connection_transmittance(scene,geometry,receiver,connection,policy,fibre);
}
// Ordinary receivers consume the same RGB straight-segment transmittance.
public float3 ray_visibility(RaytracingAccelerationStructure scene,RayGeometry geometry,float3 world,float3 normal,LightValues light,uint light_index,uint2 pixel,float3 position_error=float3(0.),uint sample_override=0xffffffffu) {
    return ray_transmission(scene,geometry,world,normal,light,light_index,pixel,position_error,false,sample_override);
}

import hit;
// Like the TLAS: a fixed raster register only where the shadow group binds it
// (ray shadows), so a kernel without it numbers its own resources unshifted.
#if defined(VORK_METAL) && defined(VORK_RAY_BINDINGS)
[[vk::binding(3,VORK_RAY_BINDING_GROUP)]] public ParameterBlock<RayHitBuffersTable> ray_hit_buffers RASTER_REGISTER(VORK_RAY_HITS_REGISTER);
#elif defined(VORK_METAL)
[[vk::binding(3,VORK_RAY_BINDING_GROUP)]] public ParameterBlock<RayHitBuffersTable> ray_hit_buffers;
#elif defined(VORK_RAY_BINDINGS)
[[vk::binding(3,VORK_RAY_BINDING_GROUP)]] public StructuredBuffer<uint> ray_hit_buffers[REG_RAY_HIT_BUFFERS] RASTER_REGISTER(VORK_RAY_HITS_REGISTER);
#else
[[vk::binding(3,VORK_RAY_BINDING_GROUP)]] public StructuredBuffer<uint> ray_hit_buffers[REG_RAY_HIT_BUFFERS];
#endif

public typedef RayGeometry ReceiverGeometry;
#if defined(VORK_PROGRAM_TRACE) && !defined(VORK_HIT_KERNEL)
import geometric_trace;
export TraceHit native_geometric_trace(float3 origin,float3 direction,float minimum,float maximum,uint mask,uint object) {
    return geometric_trace<ReceiverGeometry>(ReceiverGeometry(ray_alpha,ray_hit_buffers),ray_scene,origin,direction,minimum,maximum,mask,object);
}
#endif

#ifdef VORK_PROGRAM_COVERAGE
import variant;
import scene_programs;
import ray_coverage;
import shadow_coverage;
import light_connection;
export float3 shadow_ray_fields(RaytracingAccelerationStructure scene,RayGeometry geometry,RayDesc ray,uint groups,bool all_casters,uint sample_index) {
    return field_shadow_transmittance<ScenePrograms,ReceiverGeometry>(geometry,geometry.words,ray,groups,all_casters,sample_index);
}
export float3 shadow_ray_contained(RaytracingAccelerationStructure scene,RayGeometry geometry,RayDesc ray,bool all_casters,uint groups,uint sample_index) {
    return shadow_contained_transmittance<ScenePrograms,ReceiverGeometry>(geometry,scene,geometry.words,ray,all_casters,groups,frame.lens.zw,sample_index);
}
export float3 shadow_ray_candidate(RaytracingAccelerationStructure scene,RayGeometry geometry,RayIntersection hit,float3x4 world_to_object,RayDesc ray,float distance,uint sample_index,uint sample_count) {
    return shadow_candidate_transmittance<ScenePrograms,ReceiverGeometry>(geometry,scene,geometry.words,hit,world_to_object,ray,distance,frame.lens.zw,sample_index);
}
#ifdef VORK_FIBRE
import hair_lighting;
export float3 shadow_ray_fibre_candidate(RaytracingAccelerationStructure scene,RayGeometry geometry,RayIntersection hit,float3x4 inverse,RayDesc ray,float distance,uint sample_index,uint sample_count) {
    let direction=ray.Direction;
    let surface=hit_surface<Unspecialized,ScenePrograms,ReceiverGeometry>(geometry,hit.instance_index,hit.primitive_index,hit.instance_custom_data,inverse,hit.barycentrics,direction,frame.lens.zw);
    if (surface.s.is_hair) {
        if (distance>ray.TMax) {return float3(1.);}
        return surface.entering ? strand_forward_at(surface.s.hair0,surface.s.hair1,surface.s.strand,surface.s.geometric,-direction) : float3(1.);
    }
    return shadow_candidate_transmittance<ScenePrograms,ReceiverGeometry,false,false>(geometry,scene,geometry.words,hit,inverse,ray,distance,frame.lens.zw,sample_index);
}
#endif
#endif

import subsurface_transport;
public float ray_subsurface_exit(RaytracingAccelerationStructure scene,uint object,float3 world,float3 direction) {return subsurface_exit(scene,object,world,direction);}

#ifdef VORK_PROBES
import surface;
import ray_coverage;
public bool ray_probe_visible(RaytracingAccelerationStructure scene,RayGeometry geometry,float3 world,float3 normal,float3 position_error,float3 probe) {
#ifdef VORK_PROGRAM_COVERAGE
    return probe_connection_visible<ScenePrograms,ReceiverGeometry>(geometry,scene,geometry.words,world,normal,position_error,probe);
#else
    return probe_connection_visible<NoPrograms,ReceiverGeometry>(geometry,scene,geometry.words,world,normal,position_error,probe);
#endif
}
#endif
