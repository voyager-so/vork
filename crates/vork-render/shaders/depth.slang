import vork_shadow_coverage;
// The depth prepass entries: position-only opaque depth, single-sided winding and
// alpha-tested coverage. The forward pass tests depth Equal against all of them.
import vork_constants;
import frame;
import stream;
import point_stream;
import material;
import surface;
import variant;
#ifdef VORK_SOFTWARE_RASTER
import software_surface;
#define NOMINAL_PRIMITIVE , uint primitive : SV_PrimitiveID
#define NOMINAL_SAMPLE , float2 sample_position : SV_VulkanSamplePosition
#define NOMINAL_DEPTH : SV_Depth
#define REPAIR_NOMINAL v=software_surface(v.object,primitive,v.position,floor(v.position.xy)+sample_position,float2(0.));
#else
#define NOMINAL_PRIMITIVE
#define NOMINAL_SAMPLE
#define NOMINAL_DEPTH
#define REPAIR_NOMINAL
#endif

[shader("vertex")]
Varying main_vs([[vk::location(0)]] VertexIn v, uint object : SV_VulkanInstanceID, [[vk::location(REG_CAMERA_INSTANCE_LOCATION)]] uint mapped : CAMERA_INSTANCE) {let id=camera_object(object,mapped);return vertex(point_vertex(v,id),id);}
[shader("vertex")]
float4 opaque_depth_vs([[vk::location(0)]] float3 p : POSITION, uint vertex_index : SV_VertexID, uint object : SV_VulkanInstanceID, [[vk::location(REG_CAMERA_INSTANCE_LOCATION)]] uint mapped : CAMERA_INSTANCE) : SV_Position {
    let id=camera_object(object,mapped);return opaque_depth(point_position(p,id,vertex_index),id);
}
[shader("vertex")]
OpaqueDepth opaque_single_depth_vs([[vk::location(0)]] float3 p : POSITION, uint vertex_index : SV_VertexID, uint object : SV_VulkanInstanceID, [[vk::location(REG_CAMERA_INSTANCE_LOCATION)]] uint mapped : CAMERA_INSTANCE) {
    let id=camera_object(object,mapped);return opaque_single_depth(point_position(p,id,vertex_index),id);
}
[shader("fragment")]
void opaque_single_depth_fs(OpaqueDepth v, bool front : SV_IsFrontFace) {
    if (front==(v.mirrored!=0u)) {discard;}
}
// Fibre capture vertices retain the material varying interface.
[shader("vertex")]
Varying shadow_vs([[vk::location(0)]] VertexIn v, uint object : SV_VulkanInstanceID) {return shadow_vertex(point_vertex(v,object),object);}
// Alpha-tested depth: cutout materials, then dithered ones, whose hashed
// coverage needs the primitive (forward_variant::FEATURE_DITHERED).
[shader("fragment")]
#ifdef VORK_SOFTWARE_RASTER
float
#else
void
#endif
depth_fs(Varying v, bool front : SV_IsFrontFace NOMINAL_PRIMITIVE NOMINAL_SAMPLE) NOMINAL_DEPTH {
    REPAIR_NOMINAL
    if (!masked_covers(v,front,uint(frame.temporal.x),uint(frame.temporal.y))) {discard;}
#ifdef VORK_SOFTWARE_RASTER
    return v.position.z;
#endif
}
[shader("fragment")]
#ifdef VORK_SOFTWARE_RASTER
float
#else
void
#endif
hashed_depth_fs(Varying v, bool front : SV_IsFrontFace, uint primitive : SV_PrimitiveID NOMINAL_SAMPLE) NOMINAL_DEPTH {
    REPAIR_NOMINAL
    v.primitive=primitive;
    if (!masked_covers(v,front,uint(frame.temporal.x),uint(frame.temporal.y))) {discard;}
#ifdef VORK_SOFTWARE_RASTER
    return v.position.z;
#endif
}

#ifdef VORK_SOFTWARE_RASTER
struct SoftwareDepth {
    float4 position:SV_Position;
    [[vk::location(0)]] nointerpolation uint object:OBJECT;
    [[vk::location(1)]] nointerpolation uint mirrored:MIRRORED;
};
[shader("vertex")]
SoftwareDepth software_depth_vs([[vk::location(0)]] float3 p:POSITION,uint vertex_index:SV_VertexID,uint direct:SV_VulkanInstanceID,[[vk::location(REG_CAMERA_INSTANCE_LOCATION)]] uint mapped:CAMERA_INSTANCE) {
    let object=camera_object(direct,mapped);let d=opaque_single_depth(point_position(p,object,vertex_index),object);
    return SoftwareDepth(d.position,object,d.mirrored);
}
[shader("fragment")]
float software_depth_fs(SoftwareDepth v,uint primitive:SV_PrimitiveID,float2 sample_position:SV_VulkanSamplePosition):SV_Depth {
    return software_surface(v.object,primitive,v.position,floor(v.position.xy)+sample_position,float2(0.)).position.z;
}
[shader("fragment")]
float software_single_depth_fs(SoftwareDepth v,uint primitive:SV_PrimitiveID,float2 sample_position:SV_VulkanSamplePosition,bool front:SV_IsFrontFace):SV_Depth {
    if(front==(v.mirrored!=0u)) discard;
    return software_surface(v.object,primitive,v.position,floor(v.position.xy)+sample_position,float2(0.)).position.z;
}
#endif
