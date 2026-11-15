// Fragment-list storage and masked receiver acceptance, shared by table and
// generated program visibility entries. This module declares no shader entries.
import frame;
import vork_constants;
import stream;
import material;
import surface;
import decoupled_slots;
import transparent_opacity;
import implicit_scene;
import camera_ray;
import implicit_camera;
import hit;
import variant;
#ifdef VORK_SOFTWARE_RASTER
import software_surface;
#endif

// Shared by table and authored visibility: globals/materials occupy slots 0..10.
#ifdef VORK_METAL
[[vk::binding(0,2)]] public RWStructuredBuffer<uint> fragments : register(u11);
#else
[[vk::binding(0,2)]] public RWStructuredBuffer<uint> fragments;
#endif
public uint field_descriptor(uint object) {
    return fragment_draw_table[(fragment_object_draws[object]&0x7fffffffu)*REG_RAY_SLOT_WORDS+1u];
}
// Exit faces retain their projected silhouette with synthetic near-plane
// depth (stream.slang), including when the camera is inside the domain.
public FieldHit camera_field(uint object,uint sample,float4 position,bool front,uint mirrored) {
    if(front!=(mirrored!=0u)) discard;
    float3 origin,direction;
    let size=uint2(fragments[3],fragments[4]);
    let offset=asfloat(fragment_camera_samples[sample*2u+3u].zw);
    camera_ray(sample_position(uint2(position.xy),offset),size,camera_sample_lens(fragment_camera_samples,sample),origin,direction);
    let hit=field_intersection<FragmentGeometry>(FragmentGeometry(fragment_draw_table,fragment_hit_buffers),field_descriptor(object),object,origin,direction,camera_near_distance(origin,direction),camera_far_distance(origin,direction));
    if(!hit.found) discard;
    return hit;
}
public float field_depth(uint object,FieldHit hit) {
    let world=mul(float4(hit.local,1.),object_model(object));
    let clip=camera_world_clip(world.xyz,frame);
    let depth=clip.z/clip.w;
    if(depth<0. || depth>1.) discard;
    return depth;
}
public struct VisibilityOutput {public uint2 key:SV_Target0;public float depth:SV_Depth;};

public void opaque_depth(uint sample,float4 position) {
    if(fragments[2]==0u) return;
#ifndef VORK_SOFTWARE_RASTER
    if(frame.field_geometry.z==0u) return;
#endif
    let list=FragmentLayout(fragments[2],fragments[3],fragments[4]);
    InterlockedMax(fragments[list.depth(sample,uint2(position.xy),max(frame.sampling.w,1u))],~asuint(position.z));
}
public VisibilityOutput masked_visibility<P:IProgram>(Varying v, bool front, uint primitive) {
    let descriptor=field_descriptor(v.object);
    if(descriptor!=0xffffffffu) {
        let mirrored=dot(cross(object_model(v.object)[0].xyz,object_model(v.object)[1].xyz),object_model(v.object)[2].xyz)<0.;
        if(front!=mirrored) discard;
        float3 origin,direction;let size=uint2(fragments[3],fragments[4]);
        let offset=asfloat(fragment_camera_samples[v.sample*2u+3u].zw);
        camera_ray(sample_position(uint2(v.position.xy),offset),size,camera_sample_lens(fragment_camera_samples,v.sample),origin,direction);
        let c=visibility_coverage(v.sample,frame.sampling.w,frame.temporal.xy);
        FieldHit hit;HitSurface point;uint crossing;
        if(!accepted_camera_field<Unspecialized,P>(descriptor,v.object,origin,direction,size,v.position,c.x,c.y,true,false,hit,point,crossing)) discard;
        opaque_depth(v.sample,point.v.position);
        return VisibilityOutput(visibility_key(v.object,~crossing),point.v.position.z);
    }
#ifdef VORK_SOFTWARE_RASTER
    v=software_visibility_surface(v.object,primitive,v.position,v.sample);
#endif
    v.primitive=primitive;
    let c=visibility_coverage(v.sample,frame.sampling.w,frame.temporal.xy);
    if (P.coverage ? !program_covers<P>(v,front,c.x,c.y) : !masked_covers(v,front,c.x,c.y)) {discard;}
    opaque_depth(v.sample,v.position);return VisibilityOutput(visibility_key(v.object,primitive),v.position.z);
}
