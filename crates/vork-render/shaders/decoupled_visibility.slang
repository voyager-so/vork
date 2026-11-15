// Decoupled visibility passes (passes/decoupled.rs): one sample's triangle key
// per pixel into its batch's key layer; decoupled_merge.slang records each key's
// covered samples in the pixel's slots and decoupled.slang shades each slot.
import vork_constants;
import frame;
import stream;
import point_stream;
import material;
import surface;
import decoupled_slots;
import variant;

import decoupled_fragments;
import transparent_opacity;
import implicit_scene;
import camera_ray;
#ifdef VORK_SOFTWARE_RASTER
import software_surface;
#endif

VisibilityOutput visible(Visibility v,uint primitive,bool front,bool single) {
    var depth=v.position.z;
    if(field_descriptor(v.object)!=0xffffffffu) {
        let hit=camera_field(v.object,v.sample,v.position,front,v.mirrored);
        depth=field_depth(v.object,hit);primitive=0xffffffffu;
    } else {
        if(single && front==(v.mirrored!=0u)) discard;
#ifdef VORK_SOFTWARE_RASTER
        depth=software_visibility_surface(v.object,primitive,v.position,v.sample).position.z;
#endif
    }
    var position=v.position;position.z=depth;
    opaque_depth(v.sample,position);
    return VisibilityOutput(visibility_key(v.object,primitive),depth);
}

// One visibility pass's constant vertex stream (pipelines.rs): its sample's
// clip-space offset bits (sampling::clip_offset), and the sample, for hashed
// alpha and transparent fragments.
#define VISIBILITY_PASS [[vk::location(REG_VISIBILITY_PASS_LOCATION)]] uint4 pass : VISIBILITY_PASS, [[vk::location(REG_VISIBILITY_LENS_LOCATION)]] float2 disk : VISIBILITY_LENS
float4 offset(float4 position, uint4 pass, float2 disk) {
    position.xy+=asfloat(pass.xy)*position.w;
    // Exact thin-lens translation: post.w is the sampled CoC gain in pixels.
    if (frame.post.w>0.) {
        position.xy+=aperture_point(disk,frame.aperture)*frame.lens.zw*(frame.post.w*(position.w-frame.lens.y));
    }
    return position;
}

struct Visibility {
    float4 position : SV_Position;
    [[vk::location(0)]] nointerpolation uint object : OBJECT;
    [[vk::location(1)]] nointerpolation uint mirrored : MIRRORED;
    // The pass's sample, for volume depth.
    [[vk::location(2)]] nointerpolation uint sample : VISIBILITY_SAMPLE;
};
[shader("vertex")]
Visibility vs([[vk::location(0)]] float3 p : POSITION, uint vertex_index : SV_VertexID, uint direct : SV_VulkanInstanceID, [[vk::location(REG_CAMERA_INSTANCE_LOCATION)]] uint mapped : CAMERA_INSTANCE, VISIBILITY_PASS) {
    let object=camera_object(direct,mapped);
    let single=opaque_single_depth(point_position(p,object,vertex_index),object);
    Visibility o;
    // A culled object's clip position stays outside the frustum after the sub-pixel shift.
    o.position=offset(single.position,pass,disk);
    o.object=object;o.mirrored=single.mirrored;o.sample=pass.z;
    return o;
}
[shader("fragment")]
VisibilityOutput fs(Visibility v, uint primitive : SV_PrimitiveID,bool front:SV_IsFrontFace) {return visible(v,primitive,front,false);}
// Transparent draws' fragments in front of the pass's opaque depth list their
// key and covered samples per pixel, which the merge folds into the slots
// (decoupled_merge.slang) and the shade blends (decoupled.slang); the layout is
// decoupled_slots.slang FragmentLayout. A key that an earlier pass of the batch
// listed at the pixel gains this pass's sample; otherwise it starts a record in
// this pass's list. Earlier passes' lists are complete (one triangle covers a
// pixel once per pass, so this pass adds no record a fragment of it looks for).
// Explicit opaque depth keeps hidden fragments out before storage writes.
void record_transparent(Varying v,uint primitive) {
    let k=visibility_key(v.object,primitive);let pixel=uint2(v.position.xy);
    let list=FragmentLayout(fragments[2],fragments[3],fragments[4]);
    let bit=1u<<(v.sample%32u);let pass=v.sample%REG_MERGE_BATCH;
    for (uint layer=0u;layer<pass;layer++) {
        for (var e=fragments[list.head(layer,pixel)];e!=0u;e=fragments[list.record(e-1u)]) {
            let at=list.record(e-1u);
            if (fragments[at+1u]==k.x && fragments[at+2u]==k.y) {uint ignored;InterlockedOr(fragments[at+3u],bit,ignored);return;}
        }
    }
    uint n;InterlockedAdd(fragments[0],1u,n);
    if (n>=list.capacity) {InterlockedMax(fragments[1],n+1u);return;}
    let at=list.record(n);
    fragments[at+1u]=k.x;fragments[at+2u]=k.y;fragments[at+3u]=bit;
    uint next;InterlockedExchange(fragments[list.head(pass,pixel)],n+1u,next);
    fragments[at]=next;
}
void triangle_fragments(Varying v,uint primitive) {
    let list=FragmentLayout(fragments[2],fragments[3],fragments[4]);
    // Program selection is uniform per primitive. The cheap masked-path zero
    // prefilter computes derivatives before any opacity or list divergence.
    if (primitive!=0xffffffffu && material_records[material_record(v.object)].tag.x==0u) {
        let opacity=material_opacity<Unspecialized>(v,material_uv(v));
        bool zero_factor;
        if (opacity==0. && transparent_rejection_safe(material_record(v.object),zero_factor)) {
            bool varying;
            // Varying zero coverage affects sample-evaluation policy.
            // Retain its ordinary key so depth and identity ties use one rule.
            if (transparent_sample_opacity(v,primitive,uint2(list.width,list.height),varying)==0.
                && !varying && zero_factor) {return;}
        }
    }
    record_transparent(v,primitive);
}
struct TransparentDepth {float depth:SV_Depth;};
[shader("fragment")]
TransparentDepth transparent_fragments_fs(Varying v, uint primitive : SV_PrimitiveID,bool front:SV_IsFrontFace) {
    var depth=v.position.z;
    let descriptor=field_descriptor(v.object);
    if(descriptor!=0xffffffffu) {
        var hit=camera_field(v.object,v.sample,v.position,front,uint(dot(cross(object_model(v.object)[0].xyz,object_model(v.object)[1].xyz),object_model(v.object)[2].xyz)<0.));
        float3 origin,direction;let size=uint2(fragments[3],fragments[4]);
        let offset=asfloat(fragment_camera_samples[v.sample*2u+3u].zw);
        camera_ray(sample_position(uint2(v.position.xy),offset),size,camera_sample_lens(fragment_camera_samples,v.sample),origin,direction);
        let list=FragmentLayout(fragments[2],fragments[3],fragments[4]);
        let stored=fragments[list.depth(v.sample,uint2(v.position.xy),max(frame.sampling.w,1u))];
        depth=field_depth(v.object,hit);
        for(uint crossing=0u;hit.found;crossing++) {
            let z=field_depth(v.object,hit);
            if(stored!=0u && z>asfloat(~stored)) break;
            record_transparent(v,~crossing);
            hit=field_intersection<FragmentGeometry>(FragmentGeometry(fragment_draw_table,fragment_hit_buffers),descriptor,v.object,origin,direction,hit.t+REG_FP32_GAMMA7*max(1.,abs(hit.t)),camera_far_distance(origin,direction));
        }
        return TransparentDepth(depth);
    }
#ifdef VORK_SOFTWARE_RASTER
    v=software_visibility_surface(v.object,primitive,v.position,v.sample);depth=v.position.z;
#endif
    let list=FragmentLayout(fragments[2],fragments[3],fragments[4]);
#ifdef VORK_SOFTWARE_RASTER
    // Storage writes require explicit rejection; late depth testing alone
    // cannot undo a fragment-list insertion.
    if(true) {
#else
    if(frame.field_geometry.z!=0u) {
#endif
        let stored=fragments[list.depth(v.sample,uint2(v.position.xy),max(frame.sampling.w,1u))];
        if(stored!=0u && depth>asfloat(~stored)) discard;
    }
    triangle_fragments(v,primitive);
    return TransparentDepth(depth);
}
// Single-sided draws keep the depth prepass's winding rule (depth.slang).
[shader("fragment")]
VisibilityOutput single_fs(Visibility v, bool front : SV_IsFrontFace, uint primitive : SV_PrimitiveID) {return visible(v,primitive,front,true);}
// Alpha-tested draws: the masked depth prepass's coverage rule, with hashed
// alpha decided per sample of the exposure (strata times visibility samples).
[shader("vertex")]
Varying masked_vs([[vk::location(0)]] VertexIn v, uint direct : SV_VulkanInstanceID, [[vk::location(REG_CAMERA_INSTANCE_LOCATION)]] uint mapped : CAMERA_INSTANCE, VISIBILITY_PASS) {
    let id=camera_object(direct,mapped);var o=vertex(point_vertex(v,id),id);
    o.position=offset(o.position,pass,disk);
    o.sample=pass.z;
    return o;
}
[shader("fragment")]
VisibilityOutput masked_fs(Varying v, bool front : SV_IsFrontFace, uint primitive : SV_PrimitiveID) {
    return masked_visibility<NoPrograms>(v,front,primitive);
}

// Triangle-only visibility uses the raster depth attachment before list writes.
[shader("fragment")]
[earlydepthstencil]
void transparent_triangles_fs(Varying v, uint primitive : SV_PrimitiveID, bool front : SV_IsFrontFace) {
    triangle_fragments(v,primitive);
}
