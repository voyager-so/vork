// Derived current-pose area/emission guides, using the existing hit geometry.
import frame;
import vork_constants;
import hit_geometry;
import emitter_mesh_proposal;
[[vk::binding(11,0)]] StructuredBuffer<uint> source_words : register(t6);
[[vk::binding(12,0)]] RWStructuredBuffer<float> source_weights : register(u7);
import material_layout;
[[vk::binding(14,0)]] StructuredBuffer<MaterialRecord> source_materials : register(t9);
struct AreaStep {uint4 value;uint4 emission;}; // support base, first node, count, dispatch width
[[vk::binding(13,0)]] ConstantBuffer<AreaStep> area_step : register(b8);
// wgpu Metal binding arrays are one argument buffer of resource handles.
#ifdef VORK_METAL
[[vk::binding(0,3)]] ParameterBlock<RayHitBuffersTable> source_hit_buffers : register(b10);

#else
[[vk::binding(0,3)]] StructuredBuffer<uint> source_hit_buffers[REG_RAY_HIT_BUFFERS] : register(t10);

#endif
typedef RayGeometry AreaGeometry;
uint area_index(uint3 group,uint lane) {return (group.y*area_step.value.w+group.x)*REG_EMITTER_WEIGHT_GROUP+lane;}
[shader("compute")][numthreads(REG_EMITTER_WEIGHT_GROUP,1,1)]
void areas(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    let i=area_index(group,lane);if(i>=area_step.value.z) {return;}
    let base=area_step.value.x;let count=source_words[base+1u];var area=0.;
    if(i<count) {
        let source=emitter_mesh_triangle<AreaGeometry>(AreaGeometry(source_words,source_hit_buffers),base,i);
        if (objects[source.object].metadata.y!=0u) {area=hit_emitter<AreaGeometry>(AreaGeometry(source_words,source_hit_buffers),source.instance,source.primitive,source.object,(float3x4)object_normal(source.object),float2(0.)).area;}
    }
    // Builtin factors come from this frame's animated material records. For
    // arbitrary programs the current world area remains a complete proposal;
    // never infer zero support from a point evaluation or initial input values.
    if(i<count && area>0.) {
        let source=emitter_mesh_triangle<AreaGeometry>(AreaGeometry(source_words,source_hit_buffers),base,i);
        let record=source_materials[objects[source.object].material_index];
        if(record.tag.x==0u) {
            let c=area_step.emission.x;let l=area_step.emission.y;
            let color=float3(record.values[c/4u][c%4u],record.values[(c+1u)/4u][(c+1u)%4u],record.values[(c+2u)/4u][(c+2u)%4u]);
            let power=dot(max(color,float3(0.)),float3(.2126,.7152,.0722))*max(record.values[l/4u][l%4u],0.);
            let weight=area*power;
            // Overflowed guides fall back to area; the actual estimator retains
            // full radiance and divides by the probability actually selected.
            if(isfinite(weight)) {area=weight;}
        }
    }
    source_weights[area_step.value.y+i]=area;
}
[shader("compute")][numthreads(REG_EMITTER_WEIGHT_GROUP,1,1)]
void reduce(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    let i=area_index(group,lane);if(i>=area_step.value.z) {return;}
    let node=area_step.value.y+i;
    // Every sibling covers the same padded leaf count. Store mean mass to
    // avoid overflowing the sum of individually finite source guides.
    source_weights[node]=.5*source_weights[2u*node]+.5*source_weights[2u*node+1u];
}
