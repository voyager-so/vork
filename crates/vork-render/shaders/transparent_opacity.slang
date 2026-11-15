// A raster zero is only a candidate: retained shading reconstructs the point
// and its analytic footprint, which can select a different opacity mip.
import vork_constants;
import frame;
import stream;
import material;
import material_layout;
import surface;
import variant;
import hit;
import camera_ray;
#ifdef VORK_METAL
// These resources share the fragments pipeline layout with field controls.
// Imported ray helpers must not change their Metal slots.
[[vk::binding(0,3)]] ParameterBlock<RayHitBuffersTable> fragment_hit_buffers : register(b12);
[[vk::binding(1,3)]] StructuredBuffer<uint> fragment_draw_table : register(t13);
[[vk::binding(2,3)]] StructuredBuffer<uint> fragment_object_draws : register(t14);
[[vk::binding(3,3)]] StructuredBuffer<uint4> fragment_camera_samples : register(t15);
#else
[[vk::binding(0,3)]] StructuredBuffer<uint> fragment_hit_buffers[REG_RAY_HIT_BUFFERS];
[[vk::binding(1,3)]] StructuredBuffer<uint> fragment_draw_table;
[[vk::binding(2,3)]] StructuredBuffer<uint> fragment_object_draws;
[[vk::binding(3,3)]] StructuredBuffer<uint4> fragment_camera_samples;
#endif
typedef RayGeometry FragmentGeometry;
// Solid exit suppression uses same-object front geometry even when a front
// texel is transparent. Keep that geometry unless the whole material is zero,
// or constant inputs prove it cannot require a solid exit.
bool transparent_rejection_safe(uint material, out bool zero_factor) {
    zero_factor=!material_alpha_bound<Unspecialized>(material) && material_constant_alpha(material)==0.;
    return zero_factor ||
        ((material_byte<Unspecialized>(material,MATERIAL_THIN)&15u)==0u && material_value(material,MATERIAL_THIN_WORD)>.5) ||
        ((material_byte<Unspecialized>(material,MATERIAL_TRANSMISSION)&15u)==0u && material_value(material,MATERIAL_TRANSMISSION_WORD)==0.);
}
float transparent_sample_opacity(Varying v, uint primitive, uint2 size, out bool varying) {
    let st=surface_triangle<FragmentGeometry>(FragmentGeometry(fragment_draw_table,fragment_hit_buffers),fragment_object_draws[v.object]&0x7fffffffu,primitive,v.object);
    varying=material_varying_opacity<Unspecialized>(material_record(v.object),
        float3(st.t.c[0].COLOR_0.a,st.t.c[1].COLOR_0.a,st.t.c[2].COLOR_0.a));
    let offset=asfloat(fragment_camera_samples[v.sample*2u+3u].zw);
    let position=sample_position(uint2(v.position.xy),offset);
    float3 origin,direction;
    camera_ray(position,size,camera_sample_lens(fragment_camera_samples,v.sample),origin,direction);
    let barycentrics=triangle_barycentrics(origin,direction,st.p0,st.p1,st.p2);
    let point=triangle_point(st.t,primitive,barycentrics,direction,float2(2.)/float2(size));
    let k=point.k;
    return material_opacity<Unspecialized>(point.v,material_uv(point.v,k.uv_dx,k.uv_dy,k.uv1_dx,k.uv1_dy));
}
