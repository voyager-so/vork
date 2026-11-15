// Geometry-derived finite neighbourhood for physical fibres. Four cumulative
// crossing counts span the first/last light-facing intersections.
import vork_constants;
import vork_prelude;
#ifdef VORK_CAMERA_SPLIT
import camera_config;
#endif
struct FibreLight {float4x4 matrices[6];float4 tiles[6];float4 clip[6];};
#ifndef VORK_FIBRE_BINDING
#define VORK_FIBRE_BINDING 0
#endif
[[vk::binding(VORK_FIBRE_BINDING,2)]] StructuredBuffer<FibreLight> fibre_lights : register(t15);
[[vk::binding(VORK_FIBRE_BINDING+1,2)]] Texture2D<float> fibre_near : register(t4);
[[vk::binding(VORK_FIBRE_BINDING+2,2)]] Texture2D<float> fibre_far : register(t5);
[[vk::binding(VORK_FIBRE_BINDING+3,2)]] Texture2D<float4> fibre_counts : register(t6);
[[vk::binding(VORK_FIBRE_BINDING+4,2)]] Texture2D<uint2> fibre_ids : register(t7);
uint fibre_face(float3 delta) {
    let a=abs(delta);
    if (a.x>=a.y && a.x>=a.z) {return delta.x>=0. ? 0u:1u;}
    if (a.y>=a.z) {return delta.y>=0. ? 2u:3u;}
    return delta.z>=0. ? 4u:5u;
}
float fibre_distance(float z,float4 clip) {return lerp(clip.x,clip.y,z);}
uint fibre_light_count() {
#ifdef VORK_CAMERA_SPLIT
    uint count=camera_config.fibre_light_count;
#else
    uint count,stride;fibre_lights.GetDimensions(count,stride);
#endif
    return count;
}
struct FibreCoordinate {int2 pixel;float first;float last;float depth;bool valid;};
FibreCoordinate fibre_coordinate(uint light,float3 world,float3 source) {
    FibreCoordinate c={};
    uint count=fibre_light_count();
    if(light>=count) {return c;}
    let l=fibre_lights[light];let face=l.clip[0].z==0. ? 0u:fibre_face(world-source);
    let p=mul(float4(world,1.),l.matrices[face]);
    if (p.w<=0.) {return c;}
    let ndc=p.xyz/p.w;let uv=ndc.xy*float2(.5,-.5)+.5;let tile=l.tiles[face];
    if (any(uv<float2(0.)) || any(uv>float2(1.)) || ndc.z<0.) {return c;}
    // Receivers beyond the capture far plane still see the captured fibres.
    // Only the geometry bounds need clipping; the depth metric extrapolates.
    c.pixel=int2(tile.xy+clamp(uv*tile.z,float2(0.),float2(tile.z-1.)));
    let near_z=fibre_near.Load(int3(c.pixel,0));
    if (near_z>=1.) {return c;}
    c.first=fibre_distance(near_z,l.clip[face]);
    c.last=fibre_distance(fibre_far.Load(int3(c.pixel,0)),l.clip[face]);
    c.depth=l.clip[0].z==0. ? fibre_distance(ndc.z,l.clip[face]):max(abs(world-source).x,max(abs(world-source).y,abs(world-source).z));c.valid=true;return c;
}
// Scalar counts feed local returns; ray crossings evaluate forward extinction.
float fibre_neighbours(uint light,float3 world,float3 source,uint object,uint strand) {
    let c=fibre_coordinate(light,world,source);
    if (!c.valid) {return 0.;}
    let counts=fibre_counts.Load(int3(c.pixel,0));let total=counts.w;
    if (all(fibre_ids.Load(int3(c.pixel,0))==uint2(object,strand))) {return max(total-1.,0.);}
    if (c.depth>=c.last) {return 0.;}
    if (c.depth<=c.first) {return max(total-1.,0.);}
    let at=saturate((c.depth-c.first)/max(c.last-c.first,1e-8))*float(REG_FIBRE_LAYERS);
    let layer=min(uint(at),REG_FIBRE_LAYERS-1u);let fraction=at-float(layer);
    let before=layer==0u ? 0.:counts[layer-1u];
    let prefix=lerp(before,counts[layer],fraction);
    // Remove the receiver's crossing from the interpolated depth kernel.
    return max(total-prefix-(1.-fraction),0.);
}
