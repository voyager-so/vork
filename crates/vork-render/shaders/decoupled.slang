// Decoupled shading. A scene-time stratum's spatial samples are visibility passes
// (decoupled_visibility.slang) that write one triangle key per pixel;
// A single opaque sample is already one slot; otherwise decoupled_merge.slang
// groups the keys and records their covered samples in the pixel's slots;
// each slot is then shaded once, at the centroid of its samples, through the
// ray-hit surface path.
import vork_constants;
import camera_config;
import fibre_transport;
#define VORK_CAMERA_AOVS camera_config.aovs
#define VORK_CAMERA_DIFFUSION camera_config.diffusion

import frame;
import stream;
import variant;
import material;
import emitter;
import material_layout;
import surface;
#if !defined(VORK_CAMERA_CLASSIFY) && (!defined(VORK_COMPOSE_STORED) || defined(VORK_CAMERA_FUSED))
#define VORK_CAMERA_SHADES
#endif
#ifdef VORK_CAMERA_SHADES
import hit_lighting;
import ray_coverage;
import vork_ray_alpha;
import vork_ray_sampling;
import surface_lighting;
#endif
import vork_sampling;
import camera_ray;
import implicit_scene;
#ifdef VORK_CAMERA_ENVIRONMENT
import camera_environment;
#endif

#if defined(VORK_CAMERA_CLASSIFY)
import hit_geometry;
#else
import hit;
#endif
#ifdef VORK_CAMERA_SHADES
import forward;
import shadows;
import rays;
#endif

import decoupled_slots;
import reflection_bsdf;
[[vk::binding(13,4)]] RWStructuredBuffer<uint> nonfinite : register(u28);
import output_diagnostics;

#ifdef VORK_CAMERA_SHADES
import transmission;
import transmission_mesh;
#endif
import subsurface_capture;
import reflection_receiver;
#if defined(VORK_COMPOSE_STORED)
import forward;
#endif

struct CameraPolicy : IVariant {
    // Static-only policy: avoid Slang 2026.21 synthesizing a zero initializer.
    __init() {}
    static const bool layered = true;
    static const bool metal = true;
#ifdef VORK_RAY_BINDINGS
    static const bool ray_shadows = camera_config.ray_shadows!=0u;
    static const bool subsurface = (camera_config.flags&32u)!=0u;
#else
    static const bool ray_shadows = false;
    static const bool subsurface = false;
#endif
    static const bool specialized = false;
    static const uint textures_lo = 0xffffffffu;
    static const uint textures_hi = 0xffffffffu;
#ifdef VORK_CAMERA_OPAQUE
    static const bool has_transmission = false;
    static const bool has_solid_transmission = false;
#else
    static const bool has_transmission = true;
    static const bool has_solid_transmission = (camera_config.flags&8u)!=0u;
#endif
    static const bool has_coat = true;
    static const bool has_fuzz = true;
    static const bool has_anisotropy = true;
    static const bool has_coat_anisotropy = true;
    static const bool has_film = true;
    static const bool has_tint = true;
    static const bool has_mask = true;
    static const bool has_punctual = true;
    static const bool has_finite = true;
    static const bool has_rectangle = true;
    static const bool has_ellipse = true;
    static const bool has_shadow_rays = true;
    static const bool has_emitter_profiles = true;
    static const bool has_camera_receivers = (camera_config.flags&1u)!=0u;
    static const bool has_reflections = (camera_config.flags&2u)!=0u;
    static const bool has_indirect_controls = (camera_config.flags&4u)!=0u;
    static const bool has_lobes = true;
    static const bool has_lit = true;
    static const uint light_types = 15u;
    static const bool has_implicit_geometry = true;
    static const bool has_weighted_ior = true;
    static const bool has_gltf_tint = true;
    static const bool has_null_connection = true;
    static const bool has_camera_interior = true;
    static const bool has_subsurface = true;
    static const bool has_hair = true;
    static const bool installed = false;
};

#define VORK_CAMERA_SINGLE (camera_config.samples==1u)
#define VORK_CAMERA_TRANSPARENT camera_config.transparent
#define single_sample VORK_CAMERA_SINGLE
#define VORK_CAMERA_DIRECT (camera_config.direct!=0u)
#define direct_keys VORK_CAMERA_DIRECT
// A single opaque table slot has an implicit pixel-indexed work list.
bool direct_evaluations() {return direct_keys && camera_config.materials==2u;}
// Constant-opacity opaque table slots share a compact result list per band.
bool slot_evaluations() {return !direct_keys && (camera_config.flags&REG_CAMERA_IMPLICIT_SLOTS)!=0u;}
static uint2 pixel_key;
static uint2 composition_pixel=uint2(0u);
uint visibility_samples() {return single_sample ? 1u : frame.sampling.w;}

// Shading: one thread per pixel, one evaluation per slot. Groups 0-2 are the
// forward pass's (scene globals, material table, shadow reads), group 3 the hit
// geometry and retained evaluation tables, group 4 the colour, world and receiver images.
#ifdef VORK_METAL
// Metal binds a resource array through an argument buffer (hit_geometry.slang).
[[vk::binding(0,3)]] ParameterBlock<RayHitBuffersTable> camera_hit_buffers : register(b19);
#else
[[vk::binding(0,3)]] StructuredBuffer<uint> camera_hit_buffers[REG_RAY_HIT_BUFFERS] : register(t19);
#endif
[[vk::binding(0,4)]] [[vk::image_format("rgba32f")]] RWTexture2D<float4> hdr : register(u8);
[[vk::binding(1,4)]] Texture2DArray<uint4> environment_source : register(t9);
[[vk::binding(2,4)]] SamplerState environment_sampler : register(s1);
#ifdef VORK_RASTER_MSAA
#define AMBIENT_FORMAT "rgba32f"
#else
#define AMBIENT_FORMAT "rgba16f"
#endif
[[vk::binding(3,4)]] [[vk::image_format(AMBIENT_FORMAT)]] WTexture2D<float4> ambient : register(u10);
[[vk::binding(4,4)]] [[vk::image_format("rgba32f")]] WTexture2D<float4> receivers : register(u11);
// AO normals retain the receiver object in w for its geometry uncertainty bound.
[[vk::binding(5,4)]] [[vk::image_format("rgba32f")]] WTexture2D<float4> ao_normals : register(u12);
// Coverage- and transmittance-weighted FP32 RGB, allocated only for indirect replacement.
[[vk::binding(6,4)]] RWStructuredBuffer<float> diffuse_response : register(u25);
// First record: sequence length. Then two uint4 words per visibility pass,
// the second holding its unit-disk lens point.
[[vk::binding(7,4)]] StructuredBuffer<uint4> camera_samples : register(t26);
float2 sample_lens(uint sample) {
    return camera_sample_lens(camera_samples,sample);
}
static float2 shading_lens=float2(0.);
static float2 shading_mean_lens;

// Screen refraction (passes/decoupled.rs Refraction): the opaque pass writes each
// pixel's opaque image, unweighted by what transparency transmits, and its
// receiver (normal and NDC depth, negative for background); the composite pass
// refracts through that image's pyramid.
[[vk::binding(8,4)]] Texture2D<float4> refraction_source : register(t13);
[[vk::binding(9,4)]] SamplerState refraction_sampler : register(s2);
[[vk::binding(10,4)]] Texture2DArray<uint2> visibility_keys : register(t14);
[[vk::binding(12,4)]] [[vk::image_format("rgba16f")]] WTexture2D<float4> refraction_output : register(u15);
static float4 surface_aovs[VORK_AOV_SLOTS];
static float4 pixel_aovs[VORK_AOV_SLOTS];
struct EvaluatedSlot {
    Response response; Diffusion diffusion;
    float3 world; float3 geometric; float2 barycentrics;
    // Exact camera parameter for a field hit; triangle slots keep zero.
    float ray_distance;
    float4 alpha; float4 aovs[VORK_AOV_SLOTS];
};

#if defined(VORK_COMPOSE_STORED)
void add_aovs<let AOVS:int>(float weight) {
    if (VORK_AOV_CHANNELS!=0u) for (uint a=0u;a<VORK_CAMERA_AOVS;a++) {pixel_aovs[a]+=surface_aovs[a]*weight;}
}

#endif
struct ReflectionReceiver { float weight; uint2 key; float3 world; float3 normal; float roughness; };
// Per World-link class: the transmitted reflective coverage and the receiver transmitting most of it.
struct ReflectionControls { float samples[REG_REFLECTION_CLASSES]; ReflectionReceiver receivers[REG_REFLECTION_CLASSES]; };
// The receiver's slot key (object, primitive), point, shading normal and triangle plane normal (facing the camera).
struct AoReceiver { uint samples; uint2 key; float3 world; float3 normal; float3 plane; };
bool before(uint2 a, uint2 b) {return a.x<b.x || (a.x==b.x && a.y<b.y);}
// Equal coverage chooses the nearest represented surface, independently of
// packed object/primitive addresses. Addresses change under harmless relabeling.
bool receiver_before(float3 point,float3 normal,float3 other,float3 other_normal) {
    let depth=dot(point-frame.eye.xyz,float3(frame.view[0].z,frame.view[1].z,frame.view[2].z));
    let other_depth=dot(other-frame.eye.xyz,float3(frame.view[0].z,frame.view[1].z,frame.view[2].z));
    if(depth!=other_depth) {return depth>other_depth;}
    if(normal.x!=other_normal.x) {return normal.x<other_normal.x;}
    if(normal.y!=other_normal.y) {return normal.y<other_normal.y;}
    return normal.z<other_normal.z;
}
typedef RayGeometry CameraGeometry;

void camera_ray(float2 position, uint2 size, out float3 origin, out float3 direction) {
    camera_ray(position,size,shading_lens,origin,direction);
}
// Grouped slots retain their masks; a direct key’s presence is its sole coverage bit.
// Keeping this data-dependent avoids llvmpipe empty-mask descriptor loads; a
// literal 1 reintroduced the crash even with the uniform dense walk.
uint coverage_word(uint at,uint word) {if (direct_keys) {return uint(pixel_key.x!=0u);}return slots[at+REG_VISIBILITY_KEY_WORDS+word];}

// The slot being shaded: its key word (SlotShadows reads its coverage mask)
// and covered sample count, its pixel, and its point on the triangle's plane.
static uint shading_key;
static uint shading_masks;
static uint shading_count;
static float3 shading_footprint;
static uint2 shading_pixel;
// The slot's point in pixels and in the world, and its world displacement on the
// triangle's plane per pixel along +x and +NDC y (hit_coordinates' derivatives).
static float2 shading_position;
static float3 shading_point;
static float3 shading_dx;
static float3 shading_dy;
static uint2 shading_size;
static float3 shading_face;
// Lazy aperture reach: shared by every sun proof at this shading point.
// The one covered sample being shaded at its own point, or ~0 for the slot's centroid.
static uint shading_sample=0xffffffffu;
uint exposure_sample(uint sample) {return uint(frame.temporal.x)*frame.sampling.w+sample;}
uint lighting_sample() {return shading_sample==0xffffffffu ? 0xffffffffu : exposure_sample(shading_sample);}

// Read the same film offsets used by visibility and the host's moment tables.
// The second record retains lens xy and film offset zw in pixel units.
float2 visibility_lattice(uint index) {
    return .5+asfloat(camera_samples[index*2u+3u].zw);
}
#ifdef VORK_CAMERA_ENVIRONMENT
export float4 camera_environment_image(uint address,float2 uv) {
    return environment_image_sample(environment_images,uv,
        uint2(material_words[address],material_words[address+1u]),
        uint3(material_words[address+2u],material_words[address+3u],material_words[address+4u]),
        material_words[address+5u]);
}
#endif
// The camera background seen through image position `position` with lens offset `lens`.
float4 camera_background_at(float2 position, uint2 size, float2 lens) {
    var background=float4(frame.background.rgb,1.-frame.flags.z);
    // Environment-background selection also admits declared sun discs. A
    // constant World already lives in frame.background and has no texture.
    if (frame.environment.w>0. && frame.environment.x>0.) {
        float3 origin;float3 direction;camera_ray(position,size,lens,origin,direction);
#ifdef VORK_CAMERA_ENVIRONMENT
        background.rgb=primary_environment(frame.medium.z,mul(float4(origin,0.),frame.environment_rotation).xyz,normalize(mul(float4(direction,0.),frame.environment_rotation).xyz),frame.scene.x)*frame.environment.x*frame.environment_color.rgb;
#else
        let uv=environment_uv(direction,frame.environment_rotation);
        background.rgb=environment_radiance(environment_source,uv,uint(frame.environment_color.w))*frame.environment.x*frame.environment_color.rgb;
#endif
    }
    if(frame.environment.w>0.) {float3 origin;float3 direction;camera_ray(position,size,lens,origin,direction);background.rgb+=camera_distant_lights(frame,lights,direction);}
    return background;
}
// Where covered sample `index`'s camera ray meets the slot's plane, relative to
// the slot's point (pixel y runs down, NDC y up). The derivatives map screen to
// plane affinely for an orthographic camera; a pinhole's map is projective, so
// the linear displacement divides by 1 - dz/z0 (view depths of the
// displacement and of the slot's point), the exact intersection.
#ifdef VORK_CAMERA_SHADES
float3 sample_offset(uint index) {
    if (frame.post.w>0.) {
        float3 origin;float3 direction;
        camera_ray(sample_position(shading_pixel,visibility_lattice(index)-0.5),shading_size,sample_lens(index),origin,direction);
        return origin+direction*(dot(shading_point-origin,shading_face)/dot(direction,shading_face))-shading_point;
    }
    let d=sample_position(shading_pixel,visibility_lattice(index)-0.5)-shading_position;
    let linear=shading_dx*d.x-shading_dy*d.y;
    if (frame.flags.x>.5) {return linear;}
    let forward=-float3(frame.view[0].z,frame.view[1].z,frame.view[2].z);
    // A covered sample's ray meets the plane in front of the camera: the divisor is positive.
    return linear/max(1.-dot(linear,forward)/dot(shading_point-frame.eye.xyz,forward),1e-3);
}

#endif
#ifdef VORK_CAMERA_SHADES
// Triangle shadows of a slot: one emitter ray per
// covered sample at that sample's receiver point, using RasterShadows.
// Previs instead evaluates one central light connection at the slot centroid.
struct SlotShadows<V:IVariant> : IShadows {
    static const bool probe_rays=RasterShadows<V>.probe_rays;
    static bool probe_visible(float3 world,float3 normal,float3 position_error,float3 probe) {return RasterShadows<V>.probe_visible(world,normal,position_error,probe);}
    static const bool subsurface=RasterShadows<V>.subsurface;
    static uint integration_count(uint index,out float3 normalization) {
        normalization=float3(1.);
        if(shading_sample!=0xffffffffu || (frame.integrator.w&REG_INTEGRATOR_PREVIS_CENTER)!=0u) {return 1u;}
        normalization=float3(0.);
        for(uint word=0u;word<shading_masks;word++) {
            var bits=coverage_word(shading_key,word);
            while(bits!=0u) {let i=word*32u+firstbitlow(bits);bits&=bits-1u;normalization+=float3(1.);}
        }
        normalization=select(normalization>float3(0.),normalization,float3(1.));
        return visibility_samples();
    }
    static float3 integration_sample(uint tap,inout uint index,inout float3 world) {
        if(shading_sample!=0xffffffffu) {return float3(1.);}
        if((frame.integrator.w&REG_INTEGRATOR_PREVIS_CENTER)!=0u) {index=exposure_sample(0u);return float3(1.);}
        if((coverage_word(shading_key,tap/32u)&(1u<<(tap%32u)))==0u) {return float3(0.);}
        index=exposure_sample(tap);world+=sample_offset(tap);
        return float3(1.);
    }
    static uint sample_count() {return RasterShadows<V>.sample_count();}
    static float3 visibility_direction(float3 world,float3 normal,LightValues record,uint light,float3 direction,float distance,float3 position_error,uint index,uint count,bool fibre=false) {return RasterShadows<V>.visibility_direction(world,normal,record,light,direction,distance,position_error,index,count,fibre);}
    static float thickness(uint object,float3 world,float3 direction) {return RasterShadows<V>.thickness(object,world,direction);}
};
#endif
// A slot's key: its object and the object's draw. A transparent draw's slot
// (object_draws' top bit) blends over the opaque ones.
uint2 slot_key(uint at) {if (direct_keys) {return pixel_key;}return uint2(slots[at],slots[at+1u]);}
uint slot_object(uint at) {return slot_key(at).x-1u;}
uint slot_draw(uint at) {return object_draws[slot_object(at)]&0x7fffffffu;}
bool transparent_slot(uint at) {return object_draws[slot_object(at)]>=0x80000000u;}
// A slot's triangle, loaded once for all its evaluations: its key's hit record
// and world corners.
SurfaceTriangle slot_triangle(uint at) {
    return surface_triangle<CameraGeometry>(CameraGeometry(draw_table,camera_hit_buffers),slot_draw(at),slot_key(at).y,slot_object(at));
}
// The slot's surface where the camera ray through `position` (pixels) meets its
// triangle; the slot's shading state follows that point.
#ifndef VORK_CAMERA_CLASSIFY
HitSurface slot_surface<V:IVariant,P:IProgram,let decode_table:bool=true>(SurfaceTriangle st, uint at, uint2 pixel, float2 position, uint2 size, out float2 b, out float ray_distance) {
    ray_distance=0.;
    shading_lens=float2(0.);
    if (frame.post.w>0.) {
        if (shading_sample==0xffffffffu) {shading_lens=shading_mean_lens;} else {shading_lens=sample_lens(shading_sample);}
    }
    float3 origin;float3 direction;camera_ray(position,size,origin,direction);
    if(field_primitive(st.primitive)) {
        let root=field_crossing<CameraGeometry>(CameraGeometry(draw_table,camera_hit_buffers),draw_table[slot_draw(at)*REG_RAY_SLOT_WORDS+1u],st.t.object,st.primitive,origin,direction,camera_near_distance(origin,direction),REG_FP32_MAX);
        b=float2(0.);ray_distance=root.t;
        var h=field_surface<V,P>(st.t.object,root,direction,float2(2.)/float2(size),Lobe(0u,0.));
        h.v.position.xy=position;
        shading_key=at;shading_pixel=pixel;shading_position=position;shading_point=h.v.world;shading_dx=h.k.position_dx;shading_dy=h.k.position_dy;
        shading_size=size;shading_face=h.face;surface_aovs=h.s.aovs;
        return h;
    }
    b=triangle_barycentrics(origin,direction,st.p0,st.p1,st.p2);
    let footprint=shading_sample==0xffffffffu ? shading_footprint : float3(0.);
    var h=triangle_surface<V,P,decode_table>(st.t,st.primitive,b,direction,float2(2.)/float2(size),Lobe(0u,0.),footprint);
    // Its camera pixel position, as SV_Position's xy (thin-walled screen refraction reads it).
    h.v.position.xy=position;
    shading_key=at;shading_pixel=pixel;shading_position=position;shading_point=h.v.world;shading_dx=h.k.position_dx;shading_dy=h.k.position_dy;
    shading_size=size;shading_face=h.face;
    surface_aovs=h.s.aovs;
    return h;
}
#endif
#if defined(VORK_COMPOSE_STORED)
Response slot_radiance<V:IVariant>(uint at, out float3 world, out float3 geometric, out Diffusion d, out float2 b) {
    let e=read_evaluation<V>(at,shading_sample);world=e.world;geometric=e.geometric;d=e.diffusion;b=e.barycentrics;
    surface_aovs=e.aovs;
    return e.response;
}
// Clear records that have no eligible reflection in this composition.
void clear_slot_reflection<V:IVariant>(uint at,SlotLayout layout) {
    if (V.has_reflections && frame.projection.z==REG_REFLECTIONS_RAY) {reflection_records[5u][layout.reflection(at,direct_keys)]=packHalf2x16(float2(0.,0.));}
}
// Keep each reflection lobe with its coverage mask. Post tracing reads the
// record only for indices this triangle actually covered. The lobe's weight
// carries the slot's mean transmittance (exact where it is uniform over the slot).
bool slot_reflection<V:IVariant>(uint at,SlotLayout layout,float3 world,float3 geometric,Response response,uint object,float transmittance) {
    if (!V.has_reflections || frame.projection.z!=REG_REFLECTIONS_RAY) {return false;}
    let base=layout.reflection(at,direct_keys);
    let payload=reflection_payload(response.reflected*transmittance,response.roughness,object);
    reflection_records[5u][base]=packHalf2x16(payload);
    if (!surface_reflection_traceable(payload)) {return false;}
    // Match raster's shading-side hemisphere convention; the plane is unchanged.
    if (dot(geometric,response.normal)<0.) {geometric=-geometric;}
    reflection_records[0u][base]=asuint(geometric.x);reflection_records[1u][base]=asuint(geometric.y);reflection_records[2u][base]=asuint(geometric.z);
    reflection_records[3u][base]=asuint(dot(geometric,world-frame.eye.xyz));
    reflection_records[4u][base]=packHalf2x16(encode_octahedron(response.normal));
    reflection_bsdf_set_environment(response.reflection,reflection_bsdf_environment(response.reflection)*transmittance);
    // The last vector shares the two otherwise unused BSDF components.
    // Every live component still makes exactly the same half-float round trip.
    for(uint i=0u;i<REG_REFLECTION_BSDF_VECTORS-1u;i++) {
        var value=response.reflection.data[i];if(i==8u) {value.xyz*=transmittance;}
        if(i==9u) {value.w=response.reflection.data[11].z;}
        if(i==10u) {value.w=response.reflection.data[11].w;}
        reflection_records[6u+i*2u][base]=packHalf2x16(value.xy);
        reflection_records[7u+i*2u][base]=packHalf2x16(value.zw);
    }
    reflection_records[REG_VISIBILITY_REFLECTION_WORDS-1u][base]=packHalf2x16(response.reflection.data[11].xy);
    return true;
}
#endif
// Moments shared by ordinary coverage and the transmittance-weight walk.
struct SlotMoments {
    float2 sum;float3 second;
    __init() {sum=float2(0.);second=float3(0.);}
    [mutating] void add(float2 offset) {sum+=offset;second+=float3(offset*offset,offset.x*offset.y);}
    float3 footprint(uint count) {
        let mean=sum/float(count);
        var covariance=second/float(count)-float3(mean*mean,mean.x*mean.y);
        covariance.xy=max(covariance.xy,float2(0.));
        // Cholesky of the sample-centre covariance. The composite texture
        // filter adds this support to each original one-pixel LOD footprint.
        let xx=sqrt(covariance.x);let yx=xx>0. ? covariance.z/xx : 0.;
        return float3(xx,yx,sqrt(max(covariance.y-yx*yx,0.)));
    }
};
// Preserve the occupied-slot dependency even for one sample: llvmpipe can
// issue invalid descriptor loads when an empty execution mask folds to a literal 1.
uint single_coverage(uint at,out float2 centroid,out float2 lens) {
    centroid=visibility_lattice(0u)-.5;lens=sample_lens(0u);shading_footprint=float3(0.);
    return coverage_word(at,0u);
}
// A slot's covered samples and the sum of their lattice offsets.
#ifdef VORK_COMPOSE_STORED
uint retained_coverage(uint at, SlotLayout layout, out float2 centroid, out float2 lens) {
    if(direct_evaluations()) {return single_coverage(at,centroid,lens);}
#ifdef VORK_CAMERA_FUSED
    return slot_coverage(at,layout,centroid,lens);
#else
    // Classification already computed these moments for every occupied slot.
    let record=layout.record(at,direct_keys)*REG_CAMERA_EVALUATION_METADATA_WORDS;
    centroid=asfloat(uint2(evaluation_metadata[record+3u],evaluation_metadata[record+4u]));
    lens=asfloat(uint2(evaluation_metadata[record+5u],evaluation_metadata[record+6u]));
    shading_footprint=asfloat(uint3(evaluation_metadata[record+7u],evaluation_metadata[record+8u],evaluation_metadata[record+9u]));
    return evaluation_metadata[record+2u];
#endif
}
#endif
uint slot_coverage(uint at, SlotLayout layout, out float2 centroid, out float2 lens) {
    // Circular/anamorphic aperture shaping is linear: its covered lens sum
    // shares the coverage lookup. Polygonal apertures retain the exact walk.
    let lookup_lens=frame.post.w<=0. || frame.aperture.x<3.;
    if(lookup_lens) {
        var covered=0u;
        for(uint word=0u;word<layout.masks;word++) {covered+=countbits(coverage_word(at,word));}
        if(covered==visibility_samples()) {
            let first=camera_samples[0].y;
            let a=asfloat(camera_samples[first]);let b=asfloat(camera_samples[first+1u]);
            var moments=SlotMoments();moments.sum=a.xy;moments.second=float3(a.zw,b.x);
            centroid=moments.sum;lens=frame.post.w>0. ? aperture_point(b.yz,frame.aperture) : float2(0.);shading_footprint=moments.footprint(covered);
            return covered;
        }
    }
    lens=float2(0.);var count=0u;var moments=SlotMoments();
    let lookup=camera_samples[0].z;
    if(lookup_lens && lookup!=0u) {
        for(uint word=0u;word<layout.masks;word++) {
            let bits=coverage_word(at,word);count+=countbits(bits);
            for(uint byte=0u;byte<4u;byte++) {
                let mask=(bits>>(byte*8u))&255u;
                if(mask==0u) {continue;}
                let first=lookup+((word*4u+byte)*256u+mask)*2u;
                let a=asfloat(camera_samples[first]);let b=asfloat(camera_samples[first+1u]);
                moments.sum+=a.xy;moments.second+=float3(a.zw,b.x);
                if(frame.post.w>0.) {lens+=aperture_point(b.yz,frame.aperture);}
            }
        }
        centroid=moments.sum;shading_footprint=moments.footprint(count);return count;
    }
    for (uint word=0u;word<layout.masks;word++) {
        var bits=coverage_word(at,word);
        count+=countbits(bits);
        while (bits!=0u) {
            let i=word*32u+firstbitlow(bits);bits&=bits-1u;
            let offset=visibility_lattice(i)-0.5;
            moments.add(offset);
            if (frame.post.w>0.) {lens+=sample_lens(i);}
        }
    }
    centroid=moments.sum;shading_footprint=moments.footprint(count);
    return count;
}
void retain_slot_coverage(uint metadata,uint count,float2 centroid,float2 lens) {
    evaluation_metadata[metadata+2u]=count;evaluation_metadata[metadata+3u]=asuint(centroid.x);evaluation_metadata[metadata+4u]=asuint(centroid.y);
    evaluation_metadata[metadata+5u]=asuint(lens.x);evaluation_metadata[metadata+6u]=asuint(lens.y);
    evaluation_metadata[metadata+7u]=asuint(shading_footprint.x);evaluation_metadata[metadata+8u]=asuint(shading_footprint.y);evaluation_metadata[metadata+9u]=asuint(shading_footprint.z);
}
// Metadata names the first retained evaluation and whether the slot is sampled.
uint evaluation_index(uint at,uint sample) {
    if(direct_evaluations()) {uint w,h;hdr.GetDimensions(w,h);return composition_pixel.y*w+composition_pixel.x;}
    uint w,h;hdr.GetDimensions(w,h);let layout=SlotLayout(w*h,visibility_samples(),camera_config.packet_pixels,camera_config.packet_first_pixel);
    if(slot_evaluations()) {return evaluation_metadata[layout.record(at,direct_keys)*REG_CAMERA_EVALUATION_METADATA_WORDS+1u];}
    let record=layout.record(at,direct_keys)*REG_CAMERA_EVALUATION_METADATA_WORDS;let base=evaluation_metadata[record];
    if((evaluation_metadata[record+1u]&1u)==0u) {return base;}
    var rank=0u;for(uint word=0u;word<sample/32u;word++) {rank+=countbits(coverage_word(at,word));}
    return base+rank+countbits(coverage_word(at,sample/32u)&((1u<<(sample%32u))-1u));
}
// A missing retained evaluation is an allocation placeholder, not a receiver.
bool evaluation_available(uint at,uint sample) {
#ifdef VORK_CAMERA_FUSED
    return true;
#else
    return evaluation_index(at,sample)<evaluation_allocator[1];
#endif
}
EvaluatedSlot read_evaluation<V:IVariant>(uint at,uint sample) {
#ifdef VORK_CAMERA_FUSED
    return fused_evaluation<V>(at);
#else
    let index=evaluation_index(at,sample);if(index>=evaluation_allocator[1]) {return (EvaluatedSlot)0;}
    return evaluation_read<V>(index);
#endif
}
// Authored programs and textured glTF tables shade each covered sample.
// Other table materials retain their prefiltered slot-centroid evaluation.
#if defined(VORK_COMPOSE_STORED)
bool per_sample(uint at) {
#ifdef VORK_CAMERA_FUSED
    return false;
#else
    if(direct_evaluations() || slot_evaluations()) {return false;}
    uint w,h;hdr.GetDimensions(w,h);return (evaluation_metadata[SlotLayout(w*h,visibility_samples(),camera_config.packet_pixels,camera_config.packet_first_pixel).record(at,direct_keys)*REG_CAMERA_EVALUATION_METADATA_WORDS+1u]&1u)!=0u;
#endif
}
// Colour and reflection records must reject the same fully suppressed slots.
bool composited_transparent(uint at,SlotLayout layout) {
    if(!transparent_slot(at)) {return false;}
#ifndef VORK_CAMERA_FUSED
    if((evaluation_metadata[layout.record(at,direct_keys)*REG_CAMERA_EVALUATION_METADATA_WORDS+1u]&2u)!=0u) {return false;}
#endif
    return true;
}

#endif

bool varying_opacity<V:IVariant>(SurfaceTriangle st) {
    return field_primitive(st.primitive) || material_varying_opacity<V>(material_record(st.t.object),
        float3(st.t.c[0].COLOR_0.a,st.t.c[1].COLOR_0.a,st.t.c[2].COLOR_0.a));
}
#ifndef VORK_CAMERA_CLASSIFY
bool per_sample_opacity<V:IVariant,P:IProgram>(SurfaceTriangle st) {
    return (!single_sample && P.surface && material_records[material_record(st.t.object)].tag.x!=0u) || varying_opacity<V>(st);
}
#endif
// A pass over a pixel's slots: its dense slots, then its overflow list (a pixel
// lists overflow only once its dense slots are full).
interface ISlotVisitor {[mutating] void visit(uint at);}
void visit_slots<F:ISlotVisitor>(SlotLayout layout, uint index, inout F f) {
    // Keep the dense trip count uniform, including lanes whose pixels are empty.
    // Nested early returns can lower descriptor loads under an empty execution mask.
    let dense=direct_keys ? 1u : REG_VISIBILITY_SLOTS;var live=true;
    for (uint s=0u;s<dense;s++) {
        let at=layout.dense(index,s);
        if (live) {live=slot_key(at).x!=0u;}
        if (live) {f.visit(at);}
    }
    if (!direct_keys && live) {
        for (var e=slots[layout.head(index)];e!=0u;e=slots[layout.entry(e-1u)]) {f.visit(layout.entry(e-1u)+1u);}
    }
}

// What each visibility sample transmits through the pixel's transparent slots
// to the opaque surface or background behind them: all of it in a scene without
// transparency, else per visibility sample; `active` once a
// transparent slot covered a sample, so other pixels count samples instead.
interface ITransmittance {
    static const bool transparent;
    __init();
    bool active();
    [mutating] float at(uint i);
    [mutating] void scale(uint i, float f);
}
// Whether the scene has subsurface diffusion: the shade then captures its inputs.
interface IDiffusionCapture {static const bool enabled;}
struct CameraDiffusion : IDiffusionCapture {static const bool enabled=VORK_CAMERA_DIFFUSION!=0u;};
// Opaque radiance and screen controls retain their covered-sample weighting.
// Diffusion captures each slot independently, preserving distinct objects and surfaces.
#if defined(VORK_COMPOSE_STORED)
struct OpaqueShade<V:IVariant,P:IProgram,R:ITransmittance,C:IDiffusionCapture,let AOVS:int> : ISlotVisitor {
    SlotLayout layout;uint2 pixel;uint2 size;R t;
    float3 color;float3 unweighted;uint covered;float2 offsets;float transmitted;float2 transmitted_offsets;float2 transmitted_lens;float3 ambient;float3 diffuse;AoReceiver receiver;ReflectionControls reflection;
    // Transmitted and geometric coverage share one member to avoid the pinned
    // Slang compiler's generic-struct member lookup crash.
    float2 diffusion_coverages;float3 diffusion_world;Diffusion diffusion;
    #ifdef VORK_RECEIVER_DIFFUSION
    float receiver_coverage;float3 receiver_world;Diffusion receiver_diffusion;
    #endif
    __init(SlotLayout layout, uint2 pixel, uint2 size, R t) {
        this.layout=layout;this.pixel=pixel;this.size=size;this.t=t;
        color=float3(0.);unweighted=float3(0.);covered=0u;offsets=float2(0.);transmitted=0.;transmitted_offsets=float2(0.);transmitted_lens=float2(0.);ambient=float3(0.);diffuse=float3(0.);receiver=AoReceiver(0u,uint2(0u),float3(0.),float3(0.),float3(0.));
        for (uint c=0u;c<REG_REFLECTION_CLASSES;c++) {reflection.samples[c]=0.;reflection.receivers[c]=ReflectionReceiver(0.,uint2(0u),float3(0.),float3(0.),0.);}
        reset_diffusion();
    }
    [mutating] void reset_diffusion() {
        diffusion_coverages.x=0.;diffusion_coverages.y=0.;diffusion_world=float3(0.);diffusion=(Diffusion)0;
        #ifdef VORK_RECEIVER_DIFFUSION
        if (receiver_diffuses()) {receiver_coverage=0.;receiver_world=float3(0.);receiver_diffusion=(Diffusion)0;}
        #endif
    }
    #ifdef VORK_RECEIVER_DIFFUSION
    uint receiver_record_offset() {
        // Rust supplies the uint-word base of the second capture region.
        return camera_config.receiver_record_offset;
    }
    bool receiver_diffuses() {return C.enabled && camera_config.screen!=0u;}
    #endif
    // Both captures use the same sums; only their coverage and transport differ.
    void accumulate_diffusion(float weight, float3 transport, float3 world, Diffusion d,
        inout float coverage, inout float3 position, inout Diffusion sum) {
        coverage+=weight;position+=weight*world;
        sum.light_albedo+=transport*d.light_albedo;sum.ambient_albedo+=transport*d.ambient_albedo;
        sum.radius+=weight*d.radius;sum.albedo+=transport*d.albedo;
        sum.light+=weight*d.light;sum.ambient+=weight*d.ambient;
    }
    [mutating] void diffuses(float coverage, float weight, float3 transport, float3 world, Diffusion d) {
        if (!C.enabled || all(d.radius<=0.)) {return;}
        diffusion_coverages.y+=coverage;
        #ifdef VORK_RECEIVER_DIFFUSION
        // Screen receivers retain geometric coverage independently of front glass
        // and camera-path medium attenuation, just like the opaque source image.
        if (receiver_diffuses()) {
            accumulate_diffusion(coverage,float3(coverage),world,d,receiver_coverage,receiver_world,receiver_diffusion);
        }
        #endif
        if (weight<=0.) {return;}
        accumulate_diffusion(weight,transport,world,d,diffusion_coverages.x,diffusion_world,diffusion);
    }
    void clear_diffusion(uint at) {
        if (C.enabled) {
            clear_diffusion_record(subsurface_records,layout.diffusion(at));
            #ifdef VORK_RECEIVER_DIFFUSION
            if (receiver_diffuses()) {clear_diffusion_record(subsurface_records,receiver_record_offset()+layout.diffusion(at));}
            #endif
        }
    }
    void capture_diffusion(uint at) {
        if (!C.enabled) {return;}
        let record=layout.diffusion(at);
        #ifdef VORK_RECEIVER_DIFFUSION
        if (receiver_diffuses()) {capture_record(receiver_record_offset()+record,receiver_coverage,receiver_coverage,receiver_world,receiver_diffusion);}
        #endif
        capture_record(record,diffusion_coverages.x,diffusion_coverages.y,diffusion_world,diffusion);
    }
    void capture_record(uint record, float coverage, float geometric_coverage, float3 world, Diffusion d) {
        if (coverage<=0.) {clear_diffusion_record(subsurface_records,record);return;}
        let n=max(coverage,1e-30);
        let light=resolve_diffusion(d.light_albedo,d.albedo,d.light,n);
        let ambient=resolve_diffusion(d.ambient_albedo,d.albedo,d.ambient,n);
        let radius=d.radius/n;let albedo=d.albedo/visibility_samples();
        let position=mul(float4(world/n,1.),frame.view).xyz;
        store_diffusion(subsurface_records,record,DiffusionReceiver(light,ambient,radius,albedo,position,coverage/float(visibility_samples()),geometric_coverage/float(visibility_samples())));
    }
    // Transparent surfaces retain their own receiver, weighted by exactly the
    // incoming coverage transmittance and opacity used by front composition.
    [mutating] void transparent_reflection(uint at) {
        clear_slot_reflection<V>(at,layout);reset_diffusion();
        if (camera_config.lit==0u || !composited_transparent(at,layout)) {capture_diffusion(at);return;}
        let reflection_enabled=V.has_reflections && frame.projection.z==REG_REFLECTIONS_RAY;
        if (!C.enabled && !reflection_enabled) {return;}
        var count=0u;var weight=0.;var normal=float3(0.);var world=float3(0.);var geometric=float3(0.);
        var lobe=(Response)0;var environment=float3(0.);var transport=float3(0.);
        for (uint word=0u;word<layout.masks;word++) {
            var bits=coverage_word(at,word);
            while(bits!=0u) {
                let i=word*32u+firstbitlow(bits);bits&=bits-1u;count++;
                let sample=per_sample(at) ? i : 0xffffffffu;
                if(!evaluation_available(at,sample)) {continue;}
                let e=read_evaluation<V>(at,sample);
                if(e.alpha.a<=0.) {continue;}
                let w=transparent_incoming<V>(layout,pixel,size,at,i,e.world,e.geometric)*e.alpha.a;
                if(w<=0.) {continue;}
                let vw=float3(1.)*w;
                // Diffuse substrate and reflection share composition's incoming weight.
                if (C.enabled && any(e.diffusion.radius>0.)) {
                    diffusion_coverages.y+=e.alpha.a;
                    accumulate_diffusion(w,vw,e.world,e.diffusion,diffusion_coverages.x,diffusion_world,diffusion);
                }
                if (!reflection_enabled) {continue;}
                // Keep an evaluated point on the receiver plane; summing world
                // positions can drift outside the ray origin uncertainty bound.
                weight+=w;world=e.world;geometric=e.geometric;normal+=e.response.normal*w;
                environment+=reflection_bsdf_environment(e.response.reflection)*vw;
                transport+=e.response.reflection.data[8].xyz*vw;
                lobe.reflection=e.response.reflection;lobe.reflected+=e.response.reflected*vw;
                lobe.roughness+=e.response.roughness*w;
            }
        }
        capture_diffusion(at);
        // Empty evaluations and cancelling normals do not define a receiver.
        if(!reflection_enabled || weight<=0.) {return;}
        if(!all(isfinite(normal))) {InterlockedMax(shader_errors[0],REG_RADIANCE_NONFINITE);return;}
        if(!(dot(normal,normal)>0.)) {return;}
        reflection_bsdf_set_environment(lobe.reflection,environment/weight);
        lobe.reflection.data[8].xyz=transport/weight;
        lobe.reflected/=weight;lobe.roughness/=weight;lobe.normal=normalize(normal);
        reflects(at,count,weight,world,geometric,lobe,slot_triangle(at));
    }
    [mutating] void visit(uint at) {
        // Incomplete views publish empty evaluation records. They are not
        // surface receivers: normalizing their zero normals would turn a
        // recoverable capacity shortage into a fatal radiance diagnostic.
        // Keep downstream guides empty until the drained reservation replay.
        if((shader_errors[3] | shader_errors[REG_CAMERA_VISIBILITY_OVERFLOW_WORD] | shader_errors[REG_CAMERA_FRAGMENT_OVERFLOW_WORD])!=0u) {
            clear_slot_reflection<V>(at,layout);clear_diffusion(at);return;
        }
        if (transparent_slot(at)) {transparent_reflection(at);return;}
        reset_diffusion();
        float2 centroid;var lens=float2(0.);var count=0u;var weight=0.;var uniform=true;var seen=0.;
        if (t.active()) {
            var moments=SlotMoments();
            for (uint word=0u;word<layout.masks;word++) {
                var bits=coverage_word(at,word);
                while (bits!=0u) {
                    let i=word*32u+firstbitlow(bits);bits&=bits-1u;
                    let offset=visibility_lattice(i)-0.5;
                    uniform=uniform && (count==0u || t.at(i)==seen);seen=t.at(i);
                    count++;moments.add(offset);weight+=t.at(i);transmitted_offsets+=t.at(i)*offset;
                    if (frame.post.w>0.) {let point=sample_lens(i);lens+=point;if (frame.environment.w>0.) {transmitted_lens+=t.at(i)*point;}}
                }
            }
            centroid=moments.sum;shading_footprint=moments.footprint(count);
        } else {
            count=retained_coverage(at,layout,centroid,lens);weight=float(count);if (frame.environment.w>0.) {transmitted_lens+=lens;}
        }
        if(count==0u) {clear_slot_reflection<V>(at,layout);clear_diffusion(at);return;}
        shading_mean_lens=lens/float(count);
        covered+=count;offsets+=centroid;transmitted+=weight;shading_count=count;
        float3 world;float3 geometric;Diffusion d;float2 barycentrics;
        // The slot shades once at its centroid (for all `count` samples, weight
        // `weight`), or at each covered sample: samples seeing different
        // transmittance through a transparent edge weight their own stochastic
        // terms (shadow taps), as a program material's samples do. The AO receiver is the evaluations'
        // mean (points on the triangle's plane), the reflection lobe their
        // transmittance-weighted mean, so the record carries sum(t*r)/count.
        let shared=!per_sample(at) && uniform;
        var normal=float3(0.);var mean=float2(0.);var lobe=(Response)0;
        if (shared) {
            shading_sample=single_sample ? 0u : 0xffffffffu;
            if(!evaluation_available(at,shading_sample)) {clear_slot_reflection<V>(at,layout);clear_diffusion(at);return;}
            let response=slot_radiance<V>(at,world,geometric,d,barycentrics);
            add(response,float(count),weight,float3(weight),world,d,barycentrics,normal,mean,lobe);
        } else {
            for (uint word=0u;word<layout.masks;word++) {
                var bits=coverage_word(at,word);
                while (bits!=0u) {
                    let i=word*32u+firstbitlow(bits);bits&=bits-1u;
                    shading_sample=i;
                    if(!evaluation_available(at,shading_sample)) {clear_slot_reflection<V>(at,layout);clear_diffusion(at);return;}
                    let response=slot_radiance<V>(at,world,geometric,d,barycentrics);
                    add(response,1.,t.at(i),float3(t.at(i)),world,d,barycentrics,normal,mean,lobe);
                }
            }
        }
        capture_diffusion(at);
        if(!V.has_camera_receivers && !C.enabled && !V.has_reflections && !V.has_indirect_controls) {
            // Keep visibility metadata without reconstructing unused controls.
            let key=uint2(slot_object(at),slot_key(at).y);
            if(count>receiver.samples || (count==receiver.samples && before(key,receiver.key))) {
                receiver=AoReceiver(count,key,float3(0.),float3(0.,0.,1.),float3(0.,0.,1.));
            }
            return;
        }
        // Empty evaluations and cancelling normals do not define a receiver.
        // Valid opaque receivers retain geometric coverage even when camera
        // transmittance is zero: screen refraction uses their unweighted image.
        if(!all(isfinite(normal))) {InterlockedMax(shader_errors[0],REG_RADIANCE_NONFINITE);clear_slot_reflection<V>(at,layout);return;}
        if(!(dot(normal,normal)>0.)) {clear_slot_reflection<V>(at,layout);return;}
        let receiver_normal=normalize(normal);
        let st=slot_triangle(at);
        let n=float(count);let lobe_weight=max(weight,1e-20);
        // Average in triangle coordinates: a long world-position sum can drift
        // off the plane, outside the geometry uncertainty used by AO/GI origins.
        // Reconstruct once so that bound covers the retained receiver too.
        var point=world;
        if (!shared && !field_primitive(st.primitive)) {
            var b=max(mean/n,float2(0.));if (b.x+b.y>1.) {b/=b.x+b.y;}
            point=hit_point(st.t,float3(1.-b.x-b.y,b));
        }
        if (count>receiver.samples || (count==receiver.samples && receiver_before(point,receiver_normal,receiver.world,receiver.normal))) {receiver=AoReceiver(count,uint2(st.t.object,st.primitive),point,receiver_normal,geometric);}
        reflection_bsdf_set_environment(lobe.reflection,reflection_bsdf_environment(lobe.reflection)/lobe_weight);
        lobe.reflected/=lobe_weight;lobe.roughness/=lobe_weight;lobe.normal=receiver_normal;
        reflects(at,count,weight,point,geometric,lobe,st);
    }
    // One evaluation standing for `n` covered samples of transmitted weight `w`.
    [mutating] void add(Response response, float n, float w, float3 vw, float3 world, Diffusion d, float2 barycentrics, inout float3 normal, inout float2 mean, inout Response lobe) {
        add_aovs<AOVS>(w);
        color+=response.color*vw;unweighted+=response.color*n;ambient+=response.baseline.diffuse*vw;diffuse+=response.diffuse*vw;normal+=response.normal*n;mean+=barycentrics*n;
        let environment=reflection_bsdf_environment(lobe.reflection)+reflection_bsdf_environment(response.reflection)*vw;
        lobe.reflection=response.reflection;reflection_bsdf_set_environment(lobe.reflection,environment);lobe.reflection.data[8].xyz*=vw/max(w,1e-20);lobe.reflected+=response.reflected*vw;lobe.roughness+=response.roughness*w;
        diffuses(n,w,vw,world,d);
    }
    void write_reflection_guides(uint index) {
        for (uint c=0u;c<REG_REFLECTION_CLASSES;c++) {
            var point=float4(0.,0.,0.,1.);var guide_normal=float3(0.,0.,1.);var roughness=0.;
            let r=reflection.receivers[c];
            if (r.weight>0.) {let clip=mul(float4(r.world,1.),frame.vp);point=float4(r.world,clip.z/clip.w);guide_normal=r.normal;roughness=r.roughness;}
            let at=reflection_guide_at(index,c);
            reflection_guides[at]=asuint(point.x);reflection_guides[at+1u]=asuint(point.y);reflection_guides[at+2u]=asuint(point.z);reflection_guides[at+3u]=asuint(point.w);
            reflection_guides[at+4u]=packHalf2x16(encode_octahedron(guide_normal));reflection_guides[at+5u]=asuint(reflection.samples[c]/float(visibility_samples()));
            reflection_guides[at+6u]=asuint(round(roughness*REG_REFLECTION_PACK_LEVELS)/REG_REFLECTION_PACK_LEVELS);
        }
    }
    // A slot's reflection record and its class's coverage and receiver.
    [mutating] void reflects(uint at, uint count, float weight, float3 world, float3 geometric, Response lobe, SurfaceTriangle st) {
        if (!slot_reflection<V>(at,layout,world,geometric,lobe,st.t.object,weight/float(count))) {return;}
        let c=select(reflection_world_specular(st.t.object),0u,1u);reflection.samples[c]+=weight;
        let r=reflection.receivers[c];let key=uint2(st.t.object,st.primitive);
        if (weight>r.weight || (weight==r.weight && (receiver_before(world,lobe.normal,r.world,r.normal) || (all(world==r.world) && all(lobe.normal==r.normal) && lobe.roughness<r.roughness)))) {reflection.receivers[c]=ReflectionReceiver(weight,key,world,lobe.normal,lobe.roughness);}
    }
};

#endif
// A constant, fully covered dielectric continuation owns every later interface.
// Classification and evaluation must use the same material proof.
bool full_refraction_cover(uint object) {
    let material=material_record(object);
    return material_records[material].tag.x==0u && objects[object].opacity.x>=1.
        && material_constant_alpha(material)>=1.
        && material_value(material,MATERIAL_TRANSMISSION_WORD)>0.
        && !material_alpha_bound<Unspecialized>(material)
        && (material_byte<Unspecialized>(material,MATERIAL_TRANSMISSION)&15u)==0u;
}
#ifdef VORK_CAMERA_SHADES
#ifndef VORK_REFRACTION_BOUNDARIES
#define VORK_REFRACTION_BOUNDARIES 0
#endif
#ifdef VORK_RAY_REFRACTION
#if VORK_REFRACTION_BOUNDARIES > 0
import refraction_transport;
import dielectric_scene;
import path_boundary;
// Null transport can only borrow GI from the opaque camera samples it
// actually reaches. Visibility masks can select another receiver on the same
// ray; direction alone does not establish ownership of the diffuse correction.
struct SlotRefractionReceiver : IRefractionReceiver {
    static bool matches(DielectricHit hit,float3 world,float3 origin,float3 direction) {
        if(hit.t<0.) return false;
        let size=shading_size;
        let layout=SlotLayout(size.x*size.y,visibility_samples(),camera_config.packet_pixels,camera_config.packet_first_pixel);
        let index=shading_pixel.y*size.x+shading_pixel.x;
        for(var at=layout.first(slots,index);at!=0xffffffffu;at=layout.next(slots,index,at)) {
            if(transparent_slot(at) || slot_object(at)!=hit.object) continue;
            if(!hit.field && slot_key(at).y!=hit.primitive) continue;
            var covered=true;
            if(shading_sample!=0xffffffffu) {
                covered=(coverage_word(at,shading_sample/32u)&(1u<<(shading_sample%32u)))!=0u;
            } else {
                // A shared centroid shade may export only when all of the
                // camera samples it represents have this same receiver.
                for(var word=0u;word<layout.masks;word++)
                    covered=covered && (coverage_word(shading_key,word)&~coverage_word(at,word))==0u;
            }
            if(!covered) continue;
            if(hit.field) {
                // Field visibility keys encode camera-root ordinals, whereas
                // secondary queries return roots relative to their own origin.
                let point=origin+direction*ray_plane(origin,direction,slot_plane(at));
                let error=surface_position_error(hit.object,world)+surface_position_error(hit.object,point);
                if(!all(abs(point-world)<=error)) continue;
            }
            return true;
        }
        return false;
    }
};
// The cache is indexed by camera ray, never by a material or an optical
// descriptor. Each exposure/view refreshes it with its current lens and pose.
// Raster ray lines originate on the eye plane. Transport begins at the
// projected near plane, as path does, including each sampled lens origin.
void refraction_camera_ray(float2 position,uint2 size,float2 lens,out float3 origin,out float3 direction) {
    camera_ray(position,size,lens,origin,direction);
    let forward=-float3(frame.view[0].z,frame.view[1].z,frame.view[2].z);
    let distance=(frame.params.x-dot(origin-frame.eye.xyz,forward))/dot(direction,forward);
    origin+=direction*distance;
}
uint refraction_occupancy_at(uint2 pixel,uint width,uint sample) {
    return camera_config.boundaries.w+(((pixel.y-camera_config.band)*width+pixel.x)*visibility_samples()+sample)*(1u+9u*VORK_REFRACTION_BOUNDARIES);
}
PathOccupancy<VORK_REFRACTION_BOUNDARIES> read_refraction_occupancy<V:IVariant,P:IProgram>(uint2 pixel,uint width,uint sample,float3 origin,float2 footprint) {
    PathOccupancy<VORK_REFRACTION_BOUNDARIES> state={};
    // Transport entries retain the complete reachable material union. Thin-only
    // scenes have no owner state, even though their transport ABI reserves a slot.
    if(!V.has_solid_transmission) return state;
    let at=refraction_occupancy_at(pixel,width,sample);state.count=evaluation_metadata[at];
    // Capacities below the inline limit cannot spill: append() reports an
    // overflow instead. Do not compile unreachable containment recovery into
    // every evaluator for these geometrically bounded scenes.
    if(VORK_REFRACTION_BOUNDARIES>=REG_PATH_REGISTER_BOUNDARIES && state.count>VORK_REFRACTION_BOUNDARIES) {
        return camera_occupancy<V,P,ReceiverGeometry,VORK_REFRACTION_BOUNDARIES>(ReceiverGeometry(ray_alpha,ray_hit_buffers),ray_scene,origin,float3(1.,0.,0.),camera_config.boundaries.xy,footprint,frame.field_geometry.x,frame.field_geometry.y);
    }
    for(var i=0u;i<state.count;i++) {
        let k=at+1u+i*9u;PathBoundary b={};
        b.object=evaluation_metadata[k];b.medium=evaluation_metadata[k+1u];b.ior=asfloat(evaluation_metadata[k+2u]);
        b.color=asfloat(uint3(evaluation_metadata[k+3u],evaluation_metadata[k+4u],evaluation_metadata[k+5u]));b.depth=asfloat(evaluation_metadata[k+6u]);
#ifndef VORK_UNIT_BOUNDARY_IOR
        b.reference_ior=asfloat(evaluation_metadata[k+7u]);b.specular_weight=asfloat(evaluation_metadata[k+8u]);
#endif
        state.set(i,b);
    }
    return state;
}
// A full-coverage scene-transport slot already owns every later interface.
// Reject covered slots before material evaluation, rather than tracing the same
// suffix again. Unknown/program/textured coverage keeps the general compositor.
struct RefractionFrontCover : ISlotVisitor {
    SlotPlane rear;uint2 pixel;uint2 size;uint word;uint candidates;uint bits;
    __init(SurfaceTriangle rear,uint2 pixel,uint2 size,uint word,uint candidates) {
        this.rear=slot_plane(rear);this.pixel=pixel;this.size=size;this.word=word;this.candidates=candidates;bits=0u;
    }
    [mutating] void visit(uint at) {
        if(!transparent_slot(at) || (slot_object(at)==rear.object && slot_key(at).y==rear.primitive)) {return;}
        if(!full_refraction_cover(slot_object(at))) return;
        var shared=coverage_word(at,word)&candidates&~bits;let front=slot_plane(at);
        // Field keys retain the root ordinal on each covered camera ray.
        // The complement reverses unsigned order; no traversal is needed.
#if !defined(VORK_NO_IMPLICIT) && !defined(VORK_CAMERA_FUSED)
        if(front.object==rear.object && field_primitive(front.primitive) && field_primitive(rear.primitive)) {
            if(front.primitive>rear.primitive) bits|=shared;
            return;
        }
#endif
        while(shared!=0u) {
            let bit=firstbitlow(shared);shared&=shared-1u;let sample=word*32u+bit;
            float3 origin,direction;camera_ray(sample_position(pixel,visibility_lattice(sample)-.5),size,sample_lens(sample),origin,direction);
            if(ray_plane(origin,direction,front)<ray_plane(origin,direction,rear)) {bits|=1u<<bit;}
        }
    }
};
struct RefractionCameraCoverage : ISlotVisitor {
    uint word;uint bits;
    __init(uint word) {this.word=word;bits=0u;}
    [mutating] void visit(uint at) {
        if(transparent_slot(at)) {bits|=coverage_word(at,word);}
    }
};
void prepare_refraction_occupancy<V:IVariant,P:IProgram>(uint3 gid) {
    uint w,h;hdr.GetDimensions(w,h);let pixel=uint2(gid.x,gid.y+camera_config.band);
    if(pixel.x>=w || pixel.y>=h) {return;}
    let layout=SlotLayout(w*h,visibility_samples(),camera_config.packet_pixels,camera_config.packet_first_pixel);
    for(var sample=0u;sample<visibility_samples();sample++) {
        let at=refraction_occupancy_at(pixel,w,sample);
        // Preserve the installed metadata layout and its zero-count contract.
        if(!V.has_solid_transmission) {evaluation_metadata[at]=0u;continue;}
        var coverage=RefractionCameraCoverage(sample/32u);visit_slots(layout,pixel.y*w+pixel.x,coverage);
        if((coverage.bits&(1u<<(sample%32u)))==0u) {evaluation_metadata[at]=0u;continue;}
        float3 origin,direction;
        refraction_camera_ray(sample_position(pixel,visibility_lattice(sample)-.5),uint2(w,h),sample_lens(sample),origin,direction);
        var state=camera_occupancy<V,P,ReceiverGeometry,VORK_REFRACTION_BOUNDARIES>(ReceiverGeometry(ray_alpha,ray_hit_buffers),ray_scene,origin,float3(1.,0.,0.),camera_config.boundaries.xy,float2(2.)/float2(w,h),frame.field_geometry.x,frame.field_geometry.y);
        evaluation_metadata[at]=state.count;
        for(var i=0u;i<min(state.count,VORK_REFRACTION_BOUNDARIES);i++) {
            let k=at+1u+i*9u;let b=state.at(i);
            evaluation_metadata[k]=b.object;evaluation_metadata[k+1u]=b.medium;evaluation_metadata[k+2u]=asuint(b.ior);
            evaluation_metadata[k+3u]=asuint(b.color.x);evaluation_metadata[k+4u]=asuint(b.color.y);evaluation_metadata[k+5u]=asuint(b.color.z);evaluation_metadata[k+6u]=asuint(b.depth);
#ifndef VORK_UNIT_BOUNDARY_IOR
            evaluation_metadata[k+7u]=asuint(b.reference_ior);evaluation_metadata[k+8u]=asuint(b.specular_weight);
#else
            evaluation_metadata[k+7u]=asuint(b.ior);evaluation_metadata[k+8u]=asuint(1.);
#endif
        }
        state.clear();
    }
}

#endif
#endif
// Interface transport also returns, in receiver_ambient/receiver_diffuse, the
// GI-replaceable terms of receivers reached across null interfaces
// (refraction_transport), normalized like the colour.
float3 slot_transmitted<V:IVariant,P:IProgram>(Varying v,Surface s,Coordinates k,uint2 size,bool interface_transport,inout float3 receiver_ambient,inout float3 receiver_diffuse) {
    // Receiver visibility and mesh boundary crossings vary across a pixel.
    // Keep material evaluation and front lighting at the shading point, but
    // trace transport from every covered camera sample with its own view.
    if (!interface_transport && (!V.has_transmission || s.closure.transmission_weight<=0.)) {return float3(0.);}
    let integrate=(camera_config.flags&16u)!=0u || ((frame.environment_links.w&6u)==0u && shading_count>1u);
    let saved_sample=shading_sample;
    let per_sample=(interface_transport || (V.has_solid_transmission && !s.closure.geometry_thin_walled))
        && integrate && saved_sample==0xffffffffu;
    var color=float3(0.);var null_ambient=float3(0.);var null_diffuse=float3(0.);
    for(uint word=0u;word<select(per_sample,shading_masks,1u);word++) {
        var bits=per_sample ? slots[shading_key+REG_VISIBILITY_KEY_WORDS+word] : 1u;
        while(bits!=0u) {
            let i=word*32u+firstbitlow(bits);bits&=bits-1u;
            if(per_sample ? i%camera_config.transport_parts!=camera_config.transport_part : camera_config.transport_part!=0u) {continue;}
            var sample_v=v;var sample_s=s;
            if (per_sample) {
                shading_sample=i;
                sample_v.world=v.world+sample_offset(i);
                let position=sample_position(shading_pixel,visibility_lattice(i)-0.5);
                float3 origin;float3 direction;camera_ray(position,size,sample_lens(i),origin,direction);
                sample_v.position.xy=position;sample_s.view=-direction;
            }
#if defined(VORK_RAY_REFRACTION) && VORK_REFRACTION_BOUNDARIES > 0
            if(interface_transport) {
                float3 camera_origin;float3 camera_direction;
                refraction_camera_ray(sample_v.position.xy,size,shading_sample==0xffffffffu ? shading_mean_lens : sample_lens(shading_sample),camera_origin,camera_direction);
                color+=refraction_transport<V,P,SlotRefractionReceiver,VORK_REFRACTION_BOUNDARIES>(sample_v,sample_s,k,camera_origin,read_refraction_occupancy<V,P>(shading_pixel,size.x,shading_sample==0xffffffffu ? 0u : shading_sample,camera_origin,float2(2.)/float2(size)),shading_face,float2(2.)/float2(size),shading_pixel,lighting_sample(),null_ambient,null_diffuse);
            } else
#endif
            {color+=transmission_radiance<V,SlotTransmission,P>(sample_v,sample_s,shading_face);}
        }
    }
    shading_sample=saved_sample;
    let normalization=float(select(per_sample,shading_count,1u));
    receiver_ambient+=null_ambient/normalization;receiver_diffuse+=null_diffuse/normalization;
    return color/normalization;
}
// One transport policy for ray receivers, retained mesh exits and the opaque
// screen capture. Only the composite can read the capture written by Opaque.
struct SlotTransmission : ITransmission {
    static uint2 size() {uint w,h;hdr.GetDimensions(w,h);return uint2(w,h);}
    static float4 receiver(int2 pixel) {
        return camera_config.screen!=0u && camera_config.lit!=0u ? refraction_receivers[pixel.y*int(size().x)+pixel.x] : float4(0.,0.,1.,-1.);
    }
    static float4 color(float2 uv,float lod) {
        return camera_config.screen!=0u && camera_config.lit!=0u ? refraction_source.SampleLevel(refraction_sampler,uv,lod) : float4(0.);
    }
    static float3 background(float2 uv) {return camera_background_at(uv*float2(size()),size(),float2(0.)).rgb;}
    static SolidExit solid_exit(uint object,float3 origin,float3 direction,float eta) {
        return mesh_solid_exit<CameraGeometry>(CameraGeometry(draw_table,camera_hit_buffers),object_draws[object]&0x7fffffffu,object,origin,direction,eta);
    }
    static void exhausted(uint object) {InterlockedMax(shader_errors[2],object+1u);}
};
#endif
// Stored transparency already includes coverage/discard and solid-rear decisions.
#if defined(VORK_COMPOSE_STORED)
float4 slot_transparency<V:IVariant>(uint at,uint sample,out float3 receiver_ambient,out float3 receiver_diffuse) {
    let e=read_evaluation<V>(at,sample);
    surface_aovs=e.aovs;receiver_ambient=e.response.baseline.diffuse;receiver_diffuse=e.response.diffuse;
    return e.alpha;
}
#endif
// The samples of a solid's rear slot (`rear`, mask word `word`) where a front
// face of the same object is nearer on that sample's camera ray: the solid's
// front fragment integrates its exit, so a rear fragment never adds over it. Every boundary
// of the mesh in front along a sample's ray is rasterised, so the pixel's slots
// hold them, unless the near plane clips a front face the rear one keeps (a
// camera grazing into glass).
// Coverage ordering retains a plane, never the full material vertex payload.
struct SlotPlane {uint object;uint primitive;float3 point;float3 normal;};
SlotPlane slot_plane(SurfaceTriangle st) {
    if(field_primitive(st.primitive)) return SlotPlane(st.t.object,st.primitive,float3(0.),float3(0.));
    let model=(float3x3)object_model(st.t.object);
    let geometric=exit_face(st.t.c[0].POSITION,st.t.c[1].POSITION,st.t.c[2].POSITION,model);
    // Orient the robust plane normal to the winding side for solid exit suppression.
    let normal=dot(geometric,hit_face(st.t))<0. ? -geometric : geometric;
    return SlotPlane(st.t.object,st.primitive,st.p0,normal);
}
SlotPlane slot_plane(uint at) {return slot_plane(slot_triangle(at));}
float ray_plane(float3 origin,float3 direction,SlotPlane st) {
    if(field_primitive(st.primitive)) {
        let draw=object_draws[st.object]&0x7fffffffu;
        let root=field_crossing<CameraGeometry>(CameraGeometry(draw_table,camera_hit_buffers),draw_table[draw*REG_RAY_SLOT_WORDS+1u],st.object,st.primitive,origin,direction,camera_near_distance(origin,direction),REG_FP32_MAX);
        return root.found ? root.t : REG_FP32_MAX;
    }
#ifdef VORK_SOFTWARE_RASTER
    // The bounded point can leave the camera ray at a quantised raster edge.
    // Order by camera-axis depth, exactly as the raster depth attachment does.
    let triangle=surface_triangle<CameraGeometry>(CameraGeometry(draw_table,camera_hit_buffers),object_draws[st.object]&0x7fffffffu,st.primitive,st.object);
    let b=triangle_barycentrics(origin,direction,triangle.p0,triangle.p1,triangle.p2);
    let point=hit_point(triangle.t,float3(1.-b.x-b.y,b));
    let forward=-float3(frame.view[0].z,frame.view[1].z,frame.view[2].z);
    if(camera_needs_relative_clip(frame)) {
        let model=object_model(st.object);
        let local=triangle.t.c[0].POSITION*(1.-b.x-b.y)+triangle.t.c[1].POSITION*b.x+triangle.t.c[2].POSITION*b.y;
        let relative=mul(float4(local,0.),model).xyz+(model[3].xyz-frame.eye.xyz);
        return dot(relative-(origin-frame.eye.xyz),forward)/dot(direction,forward);
    }
    return dot(point-origin,forward)/dot(direction,forward);
#else
    return dot(st.point-origin,st.normal)/dot(direction,st.normal);
#endif
}
bool ray_front(float3 origin,float3 direction,SlotPlane st) {
    if(!field_primitive(st.primitive)) return dot(st.normal,direction)<0.;
    let draw=object_draws[st.object]&0x7fffffffu;
    let hit=field_crossing<CameraGeometry>(CameraGeometry(draw_table,camera_hit_buffers),draw_table[draw*REG_RAY_SLOT_WORDS+1u],st.object,st.primitive,origin,direction,camera_near_distance(origin,direction),REG_FP32_MAX);
    return hit.found && dot(hit.normal,direction)<0.;
}
float ray_plane(float3 origin,float3 direction,SurfaceTriangle st) {return ray_plane(origin,direction,slot_plane(st));}
struct RayOrder {float depth;uint2 key;};
RayOrder ray_order(float3 origin,float3 direction,SlotPlane plane,uint2 key) {
    return RayOrder(ray_plane(origin,direction,plane),key);
}
bool before(RayOrder a,RayOrder b) {
    // Parallel or degenerate planes have no sortable intersection.
    return isfinite(a.depth) && isfinite(b.depth) &&
        (a.depth<b.depth || (a.depth==b.depth && before(a.key,b.key)));
}
struct FrontCover : ISlotVisitor {
    SlotPlane rear;uint2 pixel;uint2 size;uint word;uint candidates;uint bits;
    __init(SurfaceTriangle rear, uint2 pixel, uint2 size, uint word, uint candidates) {
        this.rear=slot_plane(rear);this.pixel=pixel;this.size=size;this.word=word;this.candidates=candidates;bits=0u;
    }
    [mutating] void visit(uint at) {
        if (slot_object(at)!=rear.object || slot_key(at).y==rear.primitive) {return;}
        var shared=coverage_word(at,word)&candidates&~bits;
        if (shared==0u) {return;}
        let st=slot_plane(at);
#if !defined(VORK_NO_IMPLICIT) && !defined(VORK_CAMERA_FUSED)
        let ordered_field=field_primitive(st.primitive) && field_primitive(rear.primitive);
        if(ordered_field && st.primitive<=rear.primitive) return;
#else
        let ordered_field=false;
#endif
        // With a pinhole camera, plane ordering is linear over the pixel ray
        // cone while neither plane crosses its horizon. Prove the whole cone
        // at its four corners before walking individual covered samples.
#ifndef VORK_SOFTWARE_RASTER
        if(frame.post.w<=0. && !field_primitive(st.primitive) && !field_primitive(rear.primitive)) {
            var all_hidden=true;var all_behind=true;var all_back=true;var rear_side=0.;var front_side=0.;var stable=true;
            for(uint corner=0u;corner<4u;corner++) {
                let offset=float2(float(corner&1u),float(corner>>1u))-.5;
                float3 origin,direction;camera_ray(sample_position(pixel,offset),size,origin,direction);
                let front_dot=dot(st.normal,direction);let rear_dot=dot(rear.normal,direction);
                if(corner==0u) {rear_side=rear_dot;front_side=front_dot;}
                stable=stable && abs(front_dot)>1e-6 && abs(rear_dot)>1e-6 && rear_dot*rear_side>0. && front_dot*front_side>0.;
                let delta=ray_plane(origin,direction,st)-ray_plane(origin,direction,rear);
                all_hidden=all_hidden && front_dot<0. && delta< -1e-6;
                all_behind=all_behind && delta>1e-6;
                all_back=all_back && front_dot>0.;
            }
            if(stable && all_hidden) {bits|=shared;return;}
            if(stable && (all_behind || all_back)) {return;}
        }
#endif
        // Exit suppression requires strictly positive interior thickness;
        // coplanar boundaries do not enclose an interior, regardless of identity.
        while (shared!=0u) {
            let bit=firstbitlow(shared);shared&=shared-1u;
            float3 origin;float3 direction;camera_ray(sample_position(pixel,visibility_lattice(word*32u+bit)-0.5),size,sample_lens(word*32u+bit),origin,direction);
            if (ray_front(origin,direction,st) && (ordered_field || ray_plane(origin,direction,st)<ray_plane(origin,direction,rear))) {bits|=1u<<bit;}
        }
    }
};

struct EarlierFieldCrossings : ISlotVisitor {
    uint object;uint primitive;uint word;uint bits;
    __init(uint object,uint primitive,uint word) {this.object=object;this.primitive=primitive;this.word=word;bits=0u;}
    [mutating] void visit(uint at) {
        if(slot_object(at)==object && field_primitive(slot_key(at).y) && slot_key(at).y>primitive) bits|=coverage_word(at,word);
    }
};
#if defined(VORK_CAMERA_CLASSIFY)
// Coverage-only policy, decided before any material evaluation or transmittance walk.
// Most slots need only their material record. Read triangle attributes only when
// vertex alpha can vary; opaque table slots never reconstruct geometry here.
bool varying_slot_opacity(uint at) {
    if(field_primitive(slot_key(at).y)) return true;
    let material=material_record(slot_object(at));
    if(material_alpha_bound<Unspecialized>(material)) {return true;}
    if((material_records[material].tag.z&REG_FLAG_VERTEX_ALPHA)==0u) {return false;}
    return varying_opacity<Unspecialized>(slot_triangle(at));
}
// A constant front opacity only changes a rear slot's sample result when
// that rear covers both transmitted and uncovered samples of the front.
// Mask relations establish this per slot without expanding a pixel wholesale.
struct SlotTransmittanceSampling : ISlotVisitor {
    SlotLayout layout;uint rear;SlotPlane rear_plane;uint2 pixel;uint2 size;bool sampled;
    __init(SlotLayout layout,uint rear,uint2 pixel,uint2 size) {
        this.layout=layout;this.rear=rear;rear_plane=slot_plane(rear);
        this.pixel=pixel;this.size=size;sampled=false;
    }
    [mutating] void visit(uint at) {
        if(sampled || at==rear || !transparent_slot(at)) {return;}
        let object=slot_object(at);let material=material_record(object);
        let varying=material_sampled(material) || varying_slot_opacity(at);
        if(!varying && (objects[object].opacity.x<=0. || material_constant_alpha(material)<=0.)) {return;}
        var covered=false;var uncovered=false;
        let front_plane=slot_plane(at);
        for(uint word=0u;word<layout.masks;word++) {
            let back=coverage_word(rear,word);let front=coverage_word(at,word);
            uncovered=uncovered || (back&~front)!=0u;
            var shared=back&front;
            // Collection depth-tests transparency against opaque geometry, and
            // composition places every retained transparent sample over opaque.
            // Equal-depth opaque/transparent ties therefore ignore slot identity.
            if(!transparent_slot(rear)) {covered=covered || shared!=0u;continue;}
#if !defined(VORK_NO_IMPLICIT) && !defined(VORK_CAMERA_FUSED)
            if(front_plane.object==rear_plane.object && field_primitive(front_plane.primitive) && field_primitive(rear_plane.primitive)) {
                let nearer=front_plane.primitive>rear_plane.primitive;
                covered=covered || (shared!=0u && nearer);
                uncovered=uncovered || (shared!=0u && !nearer);
                continue;
            }
#endif
            while(shared!=0u) {
                let i=word*32u+firstbitlow(shared);shared&=shared-1u;
                float3 origin,direction;
                camera_ray(sample_position(pixel,visibility_lattice(i)-.5),size,sample_lens(i),origin,direction);
                let front_order=ray_order(origin,direction,front_plane,slot_key(at));
                let rear_order=ray_order(origin,direction,rear_plane,slot_key(rear));
                let nearer=before(front_order,rear_order);
                covered=covered || nearer;uncovered=uncovered || !nearer;
            }
        }
        sampled=covered && (uncovered || varying);
    }
};
uint evaluation_bin(uint at) {
    let tag=material_records[material_record(slot_object(at))].tag.x;
    return tag!=0u ? tag : (!transparent_slot(at) ? camera_config.materials-1u : 0u);
}
uint table_index(uint bin) {return bin==camera_config.materials-1u ? 0u : (bin==0u ? 1u : 0xffffffffu);}
uint table_bin(uint table) {return table==0u ? camera_config.materials-1u : 0u;}
// The fixed 8x8 classify group reserves each table bin as one coherent tile.
// Keep opaque (0) and transparent (1) work separate; authored-program bins
// retain their sparse atomic updates.
groupshared uint table_offsets[128];
groupshared uint table_base[2];
static uint2 table_count;
static uint2 table_cursor;
struct CountTableEvaluations : ISlotVisitor {
    SlotLayout layout;
    __init(SlotLayout layout) {this.layout=layout;}
    [mutating] void visit(uint at) {
        let bin=evaluation_bin(at);
        let table=table_index(bin);
        if(table>=2u) {return;}
        let record=layout.record(at,direct_keys)*REG_CAMERA_EVALUATION_METADATA_WORDS;
        let policy=evaluation_metadata[record+1u];
        if((policy&2u)==0u) {table_count[table]+=(policy&1u)!=0u ? evaluation_metadata[record+2u] : 1u;}
    }
};
struct ClassifyEvaluation : ISlotVisitor {
    SlotLayout layout; uint pixel; bool scatter;
    __init(SlotLayout layout,uint pixel,bool scatter) {this.layout=layout;this.pixel=pixel;this.scatter=scatter;}
    [mutating] void visit(uint at) {
        let material=material_record(slot_object(at));let bin=evaluation_bin(at);let record=layout.record(at,direct_keys)*REG_CAMERA_EVALUATION_METADATA_WORDS;
        if(scatter) {
            if((evaluation_metadata[record+1u]&2u)!=0u) {return;}
            let per=(evaluation_metadata[record+1u]&1u)!=0u;
            let count=per ? evaluation_metadata[record+2u] : 1u;
            uint offset;
            let table=table_index(bin);
            if(table<2u) {offset=table_cursor[table];table_cursor[table]+=count;}
            else {InterlockedAdd(evaluation_bins[bin*6u+5u],count,offset);}
            let base=evaluation_bins[bin*6u+4u]+offset;
            evaluation_metadata[record]=base;
            var rank=0u;
            for(uint word=0u;word<(per ? layout.masks : 1u);word++) {
                var bits=per ? coverage_word(at,word) : 1u;
                while(bits!=0u) {
                    let sample=per ? word*32u+firstbitlow(bits) : 0xffffffffu;bits&=bits-1u;
                    let entry=camera_config.materials*6u+(evaluation_bins[bin*6u+4u]+offset+rank)*4u;
                    if(base+rank<evaluation_allocator[1]) {
                        evaluation_bins[entry]=pixel;evaluation_bins[entry+1u]=at;
                        evaluation_bins[entry+2u]=sample;evaluation_bins[entry+3u]=base+rank;
                    }
                    rank++;
                }
            }
            return;
        }
        float2 centroid,lens;let count=slot_coverage(at,layout,centroid,lens);
        let pixel_xy=uint2(pixel%layout_width(),pixel/layout_width());
        let size=uint2(layout_width(),layout.pixels/layout_width());
        // Programs and varying opacity preserve their existing sample evaluation.
        var per=material_sampled(material) || varying_slot_opacity(at);
        if(!per && !single_sample && VORK_CAMERA_TRANSPARENT!=0) {
            var front=SlotTransmittanceSampling(layout,at,pixel_xy,size);visit_slots(layout,pixel,front);per=front.sampled;
        }
        // Programs already evaluate each sample. A table rear can share a zero
        // result when its constant solid-transmission closure is hidden everywhere.
        var hidden_all=false;
        // Every sample covered by an earlier root is already owned by the
        // same scene continuation used in evaluate_slot. This proof needs no
        // root normal, so apply it before allocating per-sample rear work.
#if !defined(VORK_NO_IMPLICIT) && !defined(VORK_CAMERA_FUSED)
        if(camera_config.lit!=0u && (camera_config.flags&16u)!=0u && transparent_slot(at)
            && field_primitive(slot_key(at).y) && full_refraction_cover(slot_object(at))) {
            hidden_all=true;
            for(uint word=0u;word<layout.masks;word++) {
                var earlier=EarlierFieldCrossings(slot_object(at),slot_key(at).y,word);
                visit_slots(layout,pixel,earlier);
                hidden_all=hidden_all && (coverage_word(at,word)&~earlier.bits)==0u;
            }
        }
#endif
        if(!hidden_all && (camera_config.flags&8u)!=0u && transparent_slot(at) && material_records[material].tag.x==0u) {
            let wall_mapped=(material_byte<Unspecialized>(material,MATERIAL_THIN)&15u)!=0u;
            let transmission_mapped=(material_byte<Unspecialized>(material,MATERIAL_TRANSMISSION)&15u)!=0u;
            if(wall_mapped || transmission_mapped) {per=true;}
            else if(material_value(material,MATERIAL_THIN_WORD)<=.5 && material_value(material,MATERIAL_TRANSMISSION_WORD)>0.) {
                let st=slot_triangle(at);float3 origin,direction;
                camera_ray(sample_position(pixel_xy,centroid/float(count)),size,origin,direction);
                if(!ray_front(origin,direction,slot_plane(st))) {
                    uint hidden=0u;
                    for(uint word=0u;word<layout.masks;word++) {
                        var front=FrontCover(st,pixel_xy,size,word,coverage_word(at,word));
                        visit_slots(layout,pixel,front);hidden+=countbits(front.bits);
                    }
                    hidden_all=hidden==count;per=per || hidden!=0u;
                }
            }
        }
        // Refraction at a transmitting silhouette, and screen refraction's
        // receivers anywhere, change across covered samples even with uniform
        // opacity. Sample that front itself; its constant opacity does not
        // expand unrelated slots behind it.
        if(transparent_slot(at) && (count<visibility_samples() || camera_config.screen!=0u) && material_records[material].tag.x==0u) {
            per=per || material_value(material,MATERIAL_TRANSMISSION_WORD)>0.;
        }
        // A shared glass slot traces every covered ray in slot_transmitted.
        // With media, expand those rays into separate evaluation invocations so
        // refraction cannot multiply the medium interval work by pixel coverage.

        per=per && !single_sample && !hidden_all;
        let evaluations=hidden_all ? 0u : (per ? count : 1u);
        let table=table_index(bin);
        if(table<2u) {table_count[table]+=evaluations;}
        else if(evaluations!=0u) {uint ignored;InterlockedAdd(evaluation_bins[bin*6u+3u],evaluations,ignored);}
        evaluation_metadata[record+1u]=uint(per)|select(hidden_all,2u,0u);
        retain_slot_coverage(record,count,centroid,lens);
    }
};
uint layout_width() {uint w,h;hdr.GetDimensions(w,h);return w;}
// Each table bin reserves coherent work once per 8x8 tile.
uint2 reserve_table_evaluations(uint lane,uint2 count,uint word,uint tables) {
    table_offsets[lane]=count.x;
    if(tables==2u) {table_offsets[64u+lane]=count.y;}
    GroupMemoryBarrierWithGroupSync();
    if(lane<tables) {
        let first=lane*64u;var total=0u;
        for(uint i=0u;i<64u;i++) {let n=table_offsets[first+i];table_offsets[first+i]=total;total+=n;}
        table_base[lane]=0u;
        if(total!=0u) {InterlockedAdd(evaluation_bins[table_bin(lane)*6u+word],total,table_base[lane]);}
    }
    GroupMemoryBarrierWithGroupSync();
    return uint2(table_base[0]+table_offsets[lane],tables==2u ? table_base[1]+table_offsets[64u+lane] : 0u);
}
struct IndexEvaluationSlots : ISlotVisitor {
    SlotLayout layout;uint pixel;bool scatter;uint cursor;uint count;
    __init(SlotLayout layout,uint pixel,bool scatter,uint cursor) {
        this.layout=layout;this.pixel=pixel;this.scatter=scatter;this.cursor=cursor;count=0u;
    }
    [mutating] void visit(uint at) {
        count++;
        if(scatter) {
            let record=layout.record(at,direct_keys);
            if(record>=layout.pixels*REG_VISIBILITY_SLOTS) {
                evaluation_metadata[record*REG_CAMERA_EVALUATION_METADATA_WORDS]=pixel;
            }
            evaluation_metadata[record*REG_CAMERA_EVALUATION_METADATA_WORDS+1u]=cursor-camera_config.materials*6u;
            evaluation_bins[cursor++]=record;
        }
    }
};
// Append only occupied records, retaining compact result indices and overflow owners.
// Only the opaque table bin is populated, so its prefix base offset is always zero.
// Pixel stages run per shading band (passes/decoupled.rs); its rows start at camera_config.band.
void index_evaluation_slots(uint3 band_gid,uint lane) {
    let gid=uint3(band_gid.x,band_gid.y+camera_config.band,band_gid.z);
    uint w,h;hdr.GetDimensions(w,h);let valid=all(gid.xy<uint2(w,h));
    let pixel=gid.y*w+gid.x;let layout=SlotLayout(w*h,visibility_samples(),camera_config.packet_pixels,camera_config.packet_first_pixel);
    var counts=IndexEvaluationSlots(layout,pixel,false,0u);
    if(valid) {visit_slots(layout,pixel,counts);}
    let offsets=reserve_table_evaluations(lane,uint2(counts.count,0u),3u,1u);
    if(valid) {
        var scatter=IndexEvaluationSlots(layout,pixel,true,camera_config.materials*6u+offsets.x);
        visit_slots(layout,pixel,scatter);
    }
}
void classify_evaluations(uint3 band_gid,uint lane,bool scatter) {
    let gid=uint3(band_gid.x,band_gid.y+camera_config.band,band_gid.z);
    uint w,h;hdr.GetDimensions(w,h);let valid=gid.x<w && gid.y<h;
    let pixel=gid.xy;composition_pixel=pixel;let index=pixel.y*w+pixel.x;let layout=SlotLayout(w*h,visibility_samples(),camera_config.packet_pixels,camera_config.packet_first_pixel);
    table_count=uint2(0u);
    if(valid) {
        if(direct_keys) {pixel_key=visibility_keys.Load(int4(pixel,0,0));}
        if(scatter) {var counts=CountTableEvaluations(layout);visit_slots(layout,index,counts);}
        else {
            var visitor=ClassifyEvaluation(layout,index,false);visit_slots(layout,index,visitor);
        }
    }
    let offsets=reserve_table_evaluations(lane,table_count,scatter ? 5u : 3u,2u);
    if(scatter && valid) {
        table_cursor=offsets;
        var visitor=ClassifyEvaluation(layout,index,true);visit_slots(layout,index,visitor);
    }
}

void prefix_evaluations() {
    var offset=0u;
    for(uint m=0u;m<camera_config.materials;m++) {
        let count=evaluation_bins[m*6u+3u];evaluation_bins[m*6u]=min((count+63u)/64u,65535u);
        evaluation_bins[m*6u+1u]=max(((count+63u)/64u+65534u)/65535u,1u);evaluation_bins[m*6u+2u]=1u;
        evaluation_bins[m*6u+4u]=offset;offset+=count;evaluation_bins[m*6u+5u]=0u;
    }
    evaluation_allocator[0]=offset;
    if(offset>evaluation_allocator[1]) {InterlockedMax(shader_errors[3],offset);}
}
#endif
#ifdef VORK_CAMERA_SHADES
// evaluate_slot delivers its slot at each exit, so a stored evaluation's early
// exits end the thread. Returning the slot to one shared store instead keeps a
// convergence barrier live across the whole surface evaluation, and NVIDIA then
// spills barrier registers inside the lighting loops.
// A transparent slot's response.baseline.diffuse and response.diffuse carry only
// its null-interface receivers' GI-replaceable terms (slot_transmitted).
interface IEvaluationSink {[mutating] void deliver(EvaluatedSlot slot);[mutating] void accumulate(float3 transport,float3 receiver_ambient,float3 receiver_diffuse);}
#ifdef VORK_PROBES
uint indirect_work_stride() {return (camera_config.flags&REG_CAMERA_PROBE_LIT)!=0u ? 24u : (VORK_CAMERA_DIFFUSION!=0u ? 16u : 8u);}
#endif
struct StoredEvaluation<V:IVariant> : IEvaluationSink {
    uint output;uint at;uint pixel;uint sample;float2 centroid;
    __init(uint output,uint at,uint pixel,uint sample,float2 centroid) {this.output=output;this.at=at;this.pixel=pixel;this.sample=sample;this.centroid=centroid;}
    [mutating] void accumulate(float3 transport,float3 receiver_ambient,float3 receiver_diffuse) {
        var e=evaluation_read<V>(output);
        e.alpha.rgb+=transport;e.response.baseline.diffuse+=receiver_ambient;e.response.diffuse+=receiver_diffuse;
        evaluation_write<V>(output,e);
    }
    [mutating] void deliver(EvaluatedSlot slot) {
        evaluation_write<V>(output,slot);
#ifdef VORK_PROBES
        let consumes=slot.response.missing_lit || any(slot.response.missing_diffuse!=float3(0.)) || (VORK_CAMERA_DIFFUSION!=0u && any(slot.diffusion.radius>float3(0.)) && any(slot.diffusion.missing!=float3(0.)));
        if((camera_config.flags&REG_CAMERA_PROBE_WORK)!=0u && consumes && camera_config.lit!=0u) {
            // Centroids grow upward and individual samples downward in the
            // same reservation. Each evaluation queues at most once, so the
            // two live ranges cannot overlap while results fit their store.
            // A late authored hook is per record too, so keep its original
            // mixed-queue dispatch bound rather than batching more hook calls.
            let sampled=(sample!=0xffffffffu || shading_count==1u) && (camera_config.flags&REG_CAMERA_PROBE_LIT)==0u;
            uint request;InterlockedAdd(evaluation_allocator[sampled ? 4u : 2u],1u,request);
            if(request<evaluation_allocator[1]) {
                let record=sampled ? evaluation_allocator[1]-1u-request : request;
                let first=REG_CAMERA_INDIRECT_HEADER_WORDS+record*indirect_work_stride();
                evaluation_allocator[first]=at;evaluation_allocator[first+1u]=pixel;
                evaluation_allocator[first+2u]=sample;evaluation_allocator[first+3u]=output;
                evaluation_allocator[first+4u]=asuint(slot.response.missing_diffuse.x);
                evaluation_allocator[first+5u]=asuint(slot.response.missing_diffuse.y);
                evaluation_allocator[first+6u]=asuint(slot.response.missing_diffuse.z);
                evaluation_allocator[first+7u]=uint(slot.response.missing_lit);
                if((camera_config.flags&REG_CAMERA_PROBE_LIT)!=0u) {
                    for(uint channel=0u;channel<3u;channel++) {evaluation_allocator[first+16u+channel]=0u;}
                    evaluation_allocator[first+20u]=asuint(centroid.x);evaluation_allocator[first+21u]=asuint(centroid.y);
                    evaluation_allocator[first+22u]=shading_count;
                }
                if(VORK_CAMERA_DIFFUSION!=0u) {
                    for(uint channel=0u;channel<3u;channel++) {
                        evaluation_allocator[first+8u+channel]=asuint(slot.diffusion.missing[channel]);
                        evaluation_allocator[first+12u+channel]=asuint(slot.diffusion.missing_albedo[channel]);
                    }
                }
            } else {InterlockedMax(shader_errors[3],request+1u);}
        }
#endif
    }
};
struct ReturnedEvaluation : IEvaluationSink {
    EvaluatedSlot slot;
    __init() {slot=(EvaluatedSlot)0;}
    [mutating] void deliver(EvaluatedSlot slot) {this.slot=slot;}
    [mutating] void accumulate(float3 transport,float3 receiver_ambient,float3 receiver_diffuse) {
        this.slot.alpha.rgb+=transport;this.slot.response.baseline.diffuse+=receiver_ambient;this.slot.response.diffuse+=receiver_diffuse;
    }
};
void evaluate_slot<S:IEvaluationSink,V:IVariant,P:IProgram,let decode_table:bool,Q:IProgram>(uint at,uint2 pixel,uint2 size,float2 centroid,uint sample,inout S sink) {
    // Capacity feedback has already proved this view incomplete. Keep valid empty
    // records for downstream consumers, but do not trace lighting or transport
    // for an exposure that collection must reject. Both stored and fused paths
    // use this producer; the CPU still receives the original exact demand.
    let later=camera_config.transport_part!=0u;
    if(later && !transparent_slot(at)) {return;}
    if((shader_errors[3] | shader_errors[REG_CAMERA_VISIBILITY_OVERFLOW_WORD] | shader_errors[REG_CAMERA_FRAGMENT_OVERFLOW_WORD])!=0u) {
        if(!later) {sink.deliver((EvaluatedSlot)0);}
        return;
    }
    let lit=camera_config.lit!=0u;
    let layout=SlotLayout(size.x*size.y,visibility_samples(),camera_config.packet_pixels,camera_config.packet_first_pixel);let index=pixel.y*size.x+pixel.x;
    shading_sample=single_sample ? 0u : sample;
    let position=sample_position(pixel,sample==0xffffffffu ? centroid/float(shading_count) : visibility_lattice(sample)-0.5);
    let st=slot_triangle(at);let transparent=VORK_CAMERA_TRANSPARENT!=0 && transparent_slot(at);
    var hidden=false;
#if defined(VORK_RAY_REFRACTION) && VORK_REFRACTION_BOUNDARIES > 0
    if(transparent && lit && (camera_config.flags&16u)!=0u) {
        var covered=true;
        for(var word=0u;word<layout.masks;word++) {
            let candidates=sample==0xffffffffu ? coverage_word(at,word) : (word==sample/32u ? 1u<<(sample%32u) : 0u);
            var front=RefractionFrontCover(st,pixel,size,word,candidates);visit_slots(layout,index,front);
            covered=covered && (candidates&~front.bits)==0u;
        }
        if(covered) {if(!later) {sink.deliver((EvaluatedSlot)0);}return;}
    }
#endif
    // The first field crossing has no earlier same-object boundary that can
    // suppress it. Its facing is evaluated by slot_surface below.
#if !defined(VORK_NO_IMPLICIT) && !defined(VORK_CAMERA_FUSED)
    let first_field=st.primitive==0xffffffffu;
#else
    let first_field=false;
#endif
    if(transparent && V.has_solid_transmission && !first_field && (sample!=0xffffffffu || single_sample)) {
        float3 origin,direction;camera_ray(sample_position(pixel,centroid/float(shading_count)),size,origin,direction);
        if(!ray_front(origin,direction,slot_plane(st))) {
            let covered_sample=single_sample ? 0u : sample;
            var front=FrontCover(st,pixel,size,covered_sample/32u,coverage_word(at,covered_sample/32u));
            visit_slots(layout,index,front);hidden=((front.bits>>(covered_sample%32u))&1u)!=0u;
        }
    }
    var result=(EvaluatedSlot)0;
    float2 barycentrics;float ray_distance;
    let hsurf=slot_surface<V,P,decode_table>(st,at,pixel,position,size,barycentrics,ray_distance);
    let opacity=hsurf.s.closure.geometry_opacity*objects[st.t.object].opacity.x;
    let replaced=V.has_lit && P.lit && P.lit_material(hsurf.v);
    if(transparent && (hsurf.s.discarded || opacity<=0.)) {if(!later) {sink.deliver(result);}return;}
#if defined(VORK_RAY_REFRACTION) && VORK_REFRACTION_BOUNDARIES > 0
    let interface_transport=transparent && !replaced && V.has_transmission && (camera_config.flags&16u)!=0u;
#else
    let interface_transport=false;
#endif
    var transport=float3(0.);var receiver_ambient=float3(0.);var receiver_diffuse=float3(0.);
    if(V.has_transmission && transparent && !replaced) {
        for(uint i=0u;i<select(V.has_lobes && P.lobes,hsurf.s.lobes,1u);i++) {
            var lobe=hsurf.s;
            if(i>0u) {lobe=surface_at<V,P,decode_table>(hsurf.v,hsurf.entering,hsurf.s.view,hsurf.k,Lobe(i,0.));}
            if(hidden && V.has_solid_transmission && lobe.closure.transmission_weight>0. && !lobe.closure.geometry_thin_walled) {if(!later) {sink.deliver(result);}return;}
            if(lit) {
                let weight=select(V.has_lobes && P.lobes,lobe.weight,1.);
                var lobe_ambient=float3(0.);var lobe_diffuse=float3(0.);
                transport+=slot_transmitted<V,Q>(hsurf.v,lobe,hsurf.k,size,interface_transport,lobe_ambient,lobe_diffuse)*weight;
                receiver_ambient+=lobe_ambient*weight;receiver_diffuse+=lobe_diffuse*weight;
            }
        }
    }
    if(later) {
        if(transparent) {sink.accumulate(transport,receiver_ambient,receiver_diffuse);}
        return;
    }
    // The only camera surface+lighting site. Coverage expansion is outside this kernel.
    if((!transparent || lit) && !interface_transport) {
        result.response=surface_radiance<V,SlotShadows<V>,P,decode_table>(hsurf.v,hsurf.entering,hsurf.k,hsurf.s,hit_site(hsurf.v,hsurf.k,hsurf.face,pixel,lighting_sample(),false,true),result.diffusion,transparent && !replaced);
    }
    result.world=hsurf.v.world;result.geometric=hsurf.face;result.barycentrics=barycentrics;result.ray_distance=ray_distance;
    if(transparent) {
        let alpha=select(material_records[material_record(st.t.object)].extra.y>0.,opacity,objects[st.t.object].opacity.x);
        result.alpha=float4(result.response.color+transport,alpha);
        result.response.baseline.diffuse=receiver_ambient;result.response.diffuse=receiver_diffuse;
    }
    result.aovs=surface_aovs;
    sink.deliver(result);
}
EvaluatedSlot fused_evaluation<V:IVariant>(uint at) {
    uint w,h;hdr.GetDimensions(w,h);
    float2 centroid;shading_count=single_coverage(at,centroid,shading_mean_lens);
    shading_masks=SlotLayout(w*h,visibility_samples(),camera_config.packet_pixels,camera_config.packet_first_pixel).masks;shading_lens=float2(0.);
    var returned=ReturnedEvaluation();
    evaluate_slot<ReturnedEvaluation,V,NoPrograms,true,NoPrograms>(at,composition_pixel,uint2(w,h),centroid,0u,returned);
    return returned.slot;
}
void evaluate_material<V:IVariant,P:IProgram,let decode_table:bool=true,Q:IProgram=P>(uint3 gid) {
    let lit=camera_config.lit!=0u;
    let direct=direct_evaluations();let indexed_slots=slot_evaluations();
    if(!direct && evaluation_allocator[0]>evaluation_allocator[1]) {return;}
    let material=camera_config.material;
    gid.x+=gid.y*65535u*64u;
    uint w,h;hdr.GetDimensions(w,h);let size=uint2(w,h);
    // Keep neighbouring lanes on an 8x8 camera tile, as in the visibility and
    // composition passes. A 64-pixel scanline crosses many more primitives and
    // material branches, even though both orders issue one item per pixel.
    if(direct) {
        let tile=gid.x/64u;let lane=gid.x%64u;let columns=(w+7u)/8u;
        let p=uint2((tile%columns)*8u+lane%8u,(tile/columns)*8u+lane/8u);
        if(any(p>=size)) {return;}
        gid.x=p.y*w+p.x;
    }
    let dense=(camera_config.packet_pixels==0u ? w*h : camera_config.packet_pixels)*REG_VISIBILITY_SLOTS;
    let limit=direct ? w*h : evaluation_bins[material*6u+3u];
    if(gid.x>=limit) {return;}
    let record=indexed_slots ? evaluation_bins[camera_config.materials*6u+gid.x] : 0u;
    var entry=0u;
    if(!direct && !indexed_slots) {entry=camera_config.materials*6u+(evaluation_bins[material*6u+4u]+gid.x)*4u;}
    let index=indexed_slots ? (record<dense ? record/REG_VISIBILITY_SLOTS+(camera_config.packet_pixels==0u ? 0u : camera_config.band*w) : evaluation_metadata[record*REG_CAMERA_EVALUATION_METADATA_WORDS]) : (direct ? gid.x : evaluation_bins[entry]);
    let pixel=uint2(index%w,index/w);
    if(direct_keys) {pixel_key=visibility_keys.Load(int4(pixel,0,0));}
    if(direct && pixel_key.x==0u) {return;}
    let layout=SlotLayout(w*h,visibility_samples(),camera_config.packet_pixels,camera_config.packet_first_pixel);
    let at=indexed_slots ? layout.address(record) : (direct ? layout.dense(index,0u) : evaluation_bins[entry+1u]);
    // Composite consumes transparent slots only; authored bins can mix both kinds.
    if(camera_config.screen!=0u && lit && !transparent_slot(at)) {return;}
    let sample=indexed_slots ? (single_sample ? 0u : 0xffffffffu) : (direct ? 0u : evaluation_bins[entry+2u]);
    let output=indexed_slots ? gid.x : (direct ? index : evaluation_bins[entry+3u]);
    if(output>=evaluation_allocator[1]) {return;}
    shading_masks=layout.masks;shading_lens=float2(0.);
    let metadata=(indexed_slots ? record : layout.record(at,direct_keys))*REG_CAMERA_EVALUATION_METADATA_WORDS;
    float2 centroid;
    if(indexed_slots) {
        float2 lens;shading_count=slot_coverage(at,layout,centroid,lens);
        shading_mean_lens=lens/float(shading_count);
        retain_slot_coverage(metadata,shading_count,centroid,lens);
    } else {
        shading_count=direct ? 1u : evaluation_metadata[metadata+2u];
        centroid=direct ? visibility_lattice(0u)-.5 : asfloat(uint2(evaluation_metadata[metadata+3u],evaluation_metadata[metadata+4u]));
        shading_mean_lens=direct ? sample_lens(0u) : asfloat(uint2(evaluation_metadata[metadata+5u],evaluation_metadata[metadata+6u]))/float(shading_count);
        shading_footprint=direct ? float3(0.) : asfloat(uint3(evaluation_metadata[metadata+7u],evaluation_metadata[metadata+8u],evaluation_metadata[metadata+9u]));
    }

    var stored=StoredEvaluation<V>(output,at,index,sample,centroid);
    evaluate_slot<StoredEvaluation<V>,V,P,decode_table,Q>(at,pixel,size,centroid,sample,stored);
}

#ifdef VORK_PROBES
// One explicit diffuse connection. The only continuation is retained irradiance
// at the secondary hit; unsupported secondary support terminates at depth zero.
// This function never starts another camera or diffuse quadrature.
float3 indirect_connection<V:IVariant,P:IProgram>(EvaluatedSlot e,uint2 pixel,uint sample,uint object) {
    let count=visibility_samples()*uint(max(frame.temporal.y,1.));
    let index=uint(frame.temporal.x)*visibility_samples()+sample;
    let disk=ray_concentric_disk(sample_2d(pixel,index,count,REG_SAMPLING_STREAM_RAY_GI));
    var direction=mul(float3(disk,sqrt(max(1.-dot(disk,disk),0.))),ray_basis(e.response.normal));
    var direction_weight=1.;
    // The concentric map sends the sample square's edge to the horizon (z=0 for
    // u.x=0, which the sequence produces). A tangent direction is still in the
    // cosine-weighted hemisphere; only directions below the surface are void.
    if(dot(direction,e.geometric)<0.) {return float3(0.);}
    let error=surface_position_error(object,e.world);
    let origin=ray_origin(e.world,e.geometric,direction,error);
    RayDesc ray;ray.Origin=origin;ray.Direction=direction;ray.TMin=0.;ray.TMax=REG_FP32_MAX;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(ray_scene,RAY_FLAG_FORCE_NON_OPAQUE,REG_RAY_GI_OCCLUDER,ray);
    while(query.Proceed()) {
        let hit=RayIntersection(query.CandidateInstanceIndex(),query.CandidateInstanceID(),query.CandidatePrimitiveIndex(),query.CandidateTriangleBarycentrics());
        if(diffuse_candidate_accepts<P,ReceiverGeometry>(ReceiverGeometry(ray_alpha,ray_hit_buffers),ray_alpha,hit,query.CandidateTriangleFrontFace(),query.CandidateWorldToObject3x4(),direction,float2(0.))) {query.CommitNonOpaqueTriangleHit();}
    }
    if(query.CommittedStatus()==COMMITTED_NOTHING) {
        var radiance=world_radiance(direction);
        return radiance*direction_weight;
    }
    let hit=RayIntersection(query.CommittedInstanceIndex(),query.CommittedInstanceID(),query.CommittedPrimitiveIndex(),query.CommittedTriangleBarycentrics());
    if(ray_alpha[hit.instance_index*REG_RAY_SLOT_WORDS+2u]==REG_RAY_NO_ALPHA) {
        return float3(0.);
    }
    let u=select(P.lobes,float(screen_ray_hash(pixel,index,REG_SAMPLING_STREAM_RAY_GI_LOBE))*REG_RADICAL_INVERSE_SCALE,0.);
    let h=hit_surface<V,P,ReceiverGeometry>(ReceiverGeometry(ray_alpha,ray_hit_buffers),hit.instance_index,hit.primitive_index,hit.instance_custom_data,query.CommittedWorldToObject3x4(),hit.barycentrics,direction,float2(0.),Lobe(select(P.lobes,LOBE_PICK,0u),u));
    var radiance=finalized_hit_radiance<V,IndirectShadows<P>,P>(h,pixel,index,REG_RAY_DIFFUSE);
    return radiance*direction_weight;
}
struct IndirectShadows<P:IProgram> : IShadows {
    static const bool probe_rays=true;
    static bool probe_visible(float3 world,float3 normal,float3 error,float3 probe) {return probe_connection_visible<P,ReceiverGeometry>(ReceiverGeometry(ray_alpha,ray_hit_buffers),ray_scene,ray_alpha,world,normal,error,probe);}
    static uint integration_count(uint index,out float3 normalization) {let count=index==0xffffffffu ? sample_count() : 1u;normalization=float3(float(count));return count;}
    static float3 integration_sample(uint tap,inout uint index,inout float3 world) {if(index==0xffffffffu) {index=tap;}return float3(1.);}
    static uint sample_count() {return visibility_samples()*uint(max(frame.temporal.y,1.));}
    static float3 visibility_direction(float3 world,float3 normal,LightValues record,uint light,float3 direction,float distance,float3 error,uint index,uint count,bool fibre=false) {return hit_shadow_direction<P,ReceiverGeometry>(ReceiverGeometry(ray_alpha,ray_hit_buffers),ray_scene,ray_alpha,light,world,normal,error,direction,distance,index,count);}
    static const bool subsurface=false;
    static float thickness(uint object,float3 world,float3 direction) {return 0.;}
};
void indirect_material<V:IVariant,P:IProgram>(uint3 gid) {
    let capacity=evaluation_allocator[1];
    if(evaluation_allocator[2]>capacity || evaluation_allocator[4]>capacity-min(evaluation_allocator[2],capacity)) {return;}
    let sampled=(camera_config.flags&REG_CAMERA_INDIRECT_SAMPLED)!=0u;
    let request=camera_config.receiver_record_offset+gid.x;
    if(request>=min(evaluation_allocator[sampled ? 4u : 2u],capacity)) {return;}
    let record=sampled ? capacity-1u-request : request;
    let first=REG_CAMERA_INDIRECT_HEADER_WORDS+record*indirect_work_stride();let at=evaluation_allocator[first];let pixel_index=evaluation_allocator[first+1u];
    let sample=evaluation_allocator[first+2u];let output=evaluation_allocator[first+3u];
    uint w,h;hdr.GetDimensions(w,h);let pixel=uint2(pixel_index%w,pixel_index/w);
    if(direct_keys) {pixel_key=visibility_keys.Load(int4(pixel,0,0));}
    var e=evaluation_read<V>(output);
    let response=asfloat(uint3(evaluation_allocator[first+4u],evaluation_allocator[first+5u],evaluation_allocator[first+6u]));
    var scattering_response=float3(0.);
    if(VORK_CAMERA_DIFFUSION!=0u) {scattering_response=asfloat(uint3(evaluation_allocator[first+8u],evaluation_allocator[first+9u],evaluation_allocator[first+10u]));}
    let deferred_lit=P.lit && evaluation_allocator[first+7u]!=0u;
    if(!deferred_lit && all(response==float3(0.)) && all(scattering_response==float3(0.))) {return;}
    let ray_batch=32u;
    let ray_first=camera_config.material*ray_batch;
    let word=ray_first/32u;
    var bits=0u;var count=1u;
    if(sample!=0xffffffffu) {if(sample/32u==word) {bits=1u<<(sample%32u);}}
    else {
        bits=coverage_word(at,word);
        let layout=SlotLayout(w*h,visibility_samples(),camera_config.packet_pixels,camera_config.packet_first_pixel);
        count=evaluation_metadata[layout.record(at,direct_keys)*REG_CAMERA_EVALUATION_METADATA_WORDS+2u];
    }
    bits&=(0xffffffffu>>(32u-ray_batch))<<(ray_first%32u);
    if(bits!=0u) {uint ignored;InterlockedAdd(evaluation_allocator[3],countbits(bits),ignored);}
    var radiance=float3(0.);
    // Exactly the actual covered samples, at most one word per native pass.
    while(bits!=0u) {
        let bit=firstbitlow(bits);bits&=bits-1u;
        radiance+=indirect_connection<V,P>(e,pixel,word*32u+bit,slot_object(at));
    }
    if(P.lit && deferred_lit) {
        var incident=asfloat(uint3(evaluation_allocator[first+16u],evaluation_allocator[first+17u],evaluation_allocator[first+18u]));
        incident+=radiance/float(max(count,1u));
        for(uint channel=0u;channel<3u;channel++) {evaluation_allocator[first+16u+channel]=asuint(incident[channel]);}
        // A nonlinear hook sees the completed integral exactly once. Earlier
        // mask passes only accumulate incident light; they never run a hook.
        if(ray_first+ray_batch>=visibility_samples()) {
            let layout=SlotLayout(w*h,visibility_samples(),camera_config.packet_pixels,camera_config.packet_first_pixel);let metadata=layout.record(at,direct_keys)*REG_CAMERA_EVALUATION_METADATA_WORDS;
            shading_masks=layout.masks;shading_count=evaluation_allocator[first+22u];shading_lens=float2(0.);
            shading_mean_lens=sample==0xffffffffu ? asfloat(uint2(evaluation_metadata[metadata+5u],evaluation_metadata[metadata+6u]))/float(max(shading_count,1u)) : float2(0.);
            shading_footprint=sample==0xffffffffu ? asfloat(uint3(evaluation_metadata[metadata+7u],evaluation_metadata[metadata+8u],evaluation_metadata[metadata+9u])) : float3(0.);
            let centroid=asfloat(uint2(evaluation_allocator[first+20u],evaluation_allocator[first+21u]));
            camera_indirect_override=true;camera_indirect_irradiance=incident;
            var returned=ReturnedEvaluation();
            evaluate_slot<ReturnedEvaluation,V,P,false,P>(at,pixel,uint2(w,h),centroid,sample,returned);
            camera_indirect_override=false;evaluation_write<V>(output,returned.slot);
        }
        return;
    }
    let added=response*radiance*(frame.radiance.w/float(max(count,1u)));
    e.response.color+=added;
    if(VORK_CAMERA_TRANSPARENT!=0 && transparent_slot(at)) {e.alpha.xyz+=added;} else {e.response.baseline.diffuse+=added;}
    if(VORK_CAMERA_DIFFUSION!=0u) {
        let irradiance=radiance*(frame.radiance.w/float(max(count,1u)));
        let scattering=scattering_response*irradiance;
        let albedo=asfloat(uint3(evaluation_allocator[first+12u],evaluation_allocator[first+13u],evaluation_allocator[first+14u]))*irradiance;
        e.diffusion.light+=scattering;e.diffusion.ambient+=scattering;
        e.diffusion.light_albedo+=albedo;e.diffusion.ambient_albedo+=albedo;
    }
    evaluation_write<V>(output,e);
}

#endif
#endif

// Exact front-to-back order on each camera sample's ray. Slot identity breaks
// equal-depth ties deterministically.
#if defined(VORK_COMPOSE_STORED)
struct SlotCoverage : ISlotVisitor {
    uint word;uint bits;bool transparent;
    __init(uint word,bool transparent) {this.word=word;this.transparent=transparent;bits=0u;}
    [mutating] void visit(uint at) {
        if(transparent_slot(at)==transparent) {bits|=coverage_word(at,word);}
    }
};
struct NextTransparent<V:IVariant> : ISlotVisitor {
    SlotLayout layout;uint sample;float3 origin;float3 direction;
    bool started;RayOrder previous;RayOrder order;uint best;
    __init(SlotLayout layout,uint sample,float3 origin,float3 direction) {
        this.layout=layout;this.sample=sample;this.origin=origin;this.direction=direction;
        started=false;previous=RayOrder(0.,uint2(0u));order=previous;best=0u;
    }
    [mutating] void visit(uint at) {
        if(!composited_transparent(at,layout) || (coverage_word(at,sample/32u)&(1u<<(sample%32u)))==0u) {return;}
        // Hidden scene-continuation suffixes and rejected opacity already
        // have exact zero records. They cannot affect composition order.
#if !defined(VORK_NO_IMPLICIT) && !defined(VORK_CAMERA_FUSED)
        let e=read_evaluation<V>(at,per_sample(at) ? sample : 0xffffffffu);
        if(e.alpha.a<=0.) return;
        // Field coverage is evaluated at each camera sample. Retain that exact
        // root parameter instead of walking the same immutable publication again.
        var candidate=RayOrder(e.ray_distance,slot_key(at));
        if(!field_primitive(candidate.key.y)) candidate=ray_order(origin,direction,slot_plane(at),candidate.key);
#else
        let candidate=ray_order(origin,direction,slot_plane(at),slot_key(at));
#endif
        if(!isfinite(candidate.depth) || (started && !before(previous,candidate))) {return;}
        if(best==0u || before(candidate,order)) {best=at+1u;order=candidate;}
    }
};
// Scalar coverage transmittance is the product of all nearer retained alphas.
// The evaluated world point and geometric normal retain each receiver plane;
// no second sorting walk or mesh reconstruction is needed for this product.
struct IncomingTransmittance<V:IVariant> : ISlotVisitor {
    SlotLayout layout;uint sample;float3 origin;float3 direction;RayOrder target;float transmission;
    __init(SlotLayout layout,uint sample,float3 origin,float3 direction,RayOrder target) {
        this.layout=layout;this.sample=sample;this.origin=origin;this.direction=direction;this.target=target;transmission=1.;
    }
    [mutating] void visit(uint at) {
        if(!composited_transparent(at,layout) || (coverage_word(at,sample/32u)&(1u<<(sample%32u)))==0u) {return;}
        let e=read_evaluation<V>(at,per_sample(at) ? sample : 0xffffffffu);
        if(e.alpha.a<=0.) {return;}
        let depth=dot(e.world-origin,e.geometric)/dot(direction,e.geometric);
        if(before(RayOrder(depth,slot_key(at)),target)) {transmission*=1.-e.alpha.a;}
    }
};
float transparent_incoming<V:IVariant>(SlotLayout layout,uint2 pixel,uint2 size,uint target,uint sample,float3 world,float3 geometric) {
    float3 origin,direction;
    camera_ray(sample_position(pixel,visibility_lattice(sample)-.5),size,sample_lens(sample),origin,direction);
    let depth=dot(world-origin,geometric)/dot(direction,geometric);
    var incoming=IncomingTransmittance<V>(layout,sample,origin,direction,RayOrder(depth,slot_key(target)));
    visit_slots(layout,pixel.y*size.x+pixel.x,incoming);return incoming.transmission;
}
// `transported_ambient`/`transported_diffuse` gain the transparent slots' null-interface receiver terms
// with the same weights as their colour, for the pixel's GI correction.
void blend_transparent<V:IVariant,P:IProgram,R:ITransmittance,let AOVS:int>(SlotLayout layout,uint index,uint2 pixel,uint2 size,inout R t,inout float4 front,bool lit,inout float3 transported_ambient,inout float3 transported_diffuse) {
    for(uint word=0u;word<layout.masks;word++) {
        var covered=SlotCoverage(word,true);visit_slots(layout,index,covered);
        var bits=covered.bits;
        while(bits!=0u) {
            let i=word*32u+firstbitlow(bits);bits&=bits-1u;
            float3 origin,direction;
            camera_ray(sample_position(pixel,visibility_lattice(i)-.5),size,sample_lens(i),origin,direction);
            var walk=NextTransparent<V>(layout,i,origin,direction);
            for(visit_slots(layout,index,walk);walk.best!=0u;walk.best=0u,visit_slots(layout,index,walk)) {
                let at=walk.best-1u;walk.previous=walk.order;walk.started=true;
                float3 receiver_ambient,receiver_diffuse;
                let c=slot_transparency<V>(at,per_sample(at) ? i : 0xffffffffu,receiver_ambient,receiver_diffuse);
                if(c.a<=0.) {continue;}
                if(lit) {add_aovs<AOVS>(t.at(i)*c.a);}
                front+=t.at(i)*float4(c.rgb*float3(1.)*c.a,c.a);
                transported_ambient+=t.at(i)*c.a*receiver_ambient;transported_diffuse+=t.at(i)*c.a*receiver_diffuse;t.scale(i,1.-c.a);
#if !defined(VORK_NO_IMPLICIT) && !defined(VORK_CAMERA_FUSED)
                // An exactly extinguished camera sample has no later colour
                // or AOV contribution. Do not sort another scalar-field suffix.
                if(t.at(i)==0.) break;
#endif
            }
        }
    }
}

// The camera background along the ray through lattice offset `offset`: the
// world image, or its constant (black for transparent film or a world hidden
// from camera rays, values.rs), alpha 0 on transparent film.
float4 camera_background(uint2 pixel, uint2 size, float2 offset) {
    return camera_background_at(sample_position(pixel,offset),size,shading_lens);
}
// With screen refraction the shade runs twice: this opaque pass, whose transparent
// slots give only opacity and which writes the refraction image and receivers,
// then composite_pixel.
void shade_pixel<V:IVariant,P:IProgram,R:ITransmittance,C:IDiffusionCapture,let AOVS:int>(uint3 band_gid) {
    let gid=uint3(band_gid.x,band_gid.y+camera_config.band,band_gid.z);
    let screen=camera_config.screen!=0u && camera_config.lit==0u;
    uint w,h;hdr.GetDimensions(w,h);let size=uint2(w,h);
    if (gid.x>=w || gid.y>=h) {return;}
    let layout=SlotLayout(w*h,visibility_samples(),camera_config.packet_pixels,camera_config.packet_first_pixel);

    shading_lens=float2(0.);shading_masks=layout.masks;
    if (VORK_AOV_CHANNELS!=0u) for (uint a=0u;a<VORK_CAMERA_AOVS;a++) {pixel_aovs[a]=float4(0.);}
    if (direct_keys) {pixel_key=visibility_keys.Load(int4(gid.xy,0,0));}
    let pixel=gid.xy;composition_pixel=pixel;
    let samples=visibility_samples();
    let index=pixel.y*w+pixel.x;
    var t=R();var front=float4(0.);var transported_ambient=float3(0.);var transported_diffuse=float3(0.);
    if (R.transparent) {blend_transparent<V,P,R,AOVS>(layout,index,pixel,size,t,front,!screen,transported_ambient,transported_diffuse);}
    var opaque=OpaqueShade<V,P,R,C,AOVS>(layout,pixel,size,t);
    visit_slots(layout,index,opaque);
    // Uncovered samples see the camera background along the ray through their
    // centroid (weighted by what they transmit). A pixel's lattice offsets sum
    // to zero, so the uncovered offsets sum to -offsets.
    if(opaque.covered>samples) {InterlockedMax(shader_errors[0],REG_CAMERA_COVERAGE_INVALID);}
    let uncovered=samples-opaque.covered;
    var left=float(uncovered);var moment=-opaque.offsets;
    if (t.active()) {
        left=-opaque.transmitted;moment=-opaque.transmitted_offsets;
        for (uint i=0u;i<samples;i++) {left+=t.at(i);moment+=t.at(i)*(visibility_lattice(i)-0.5);}
    }
    var background=float4(0.);
#ifdef VORK_CAMERA_ENVIRONMENT
    // A directional environment program is evaluated along each uncovered
    // sample's own camera ray (film offset and lens), weighted by what it transmits.
    var background_seen=float3(0.);
    if(frame.environment.w<=0.) {
        if(left>0.) background=camera_background(pixel,size,moment/left);
        if(uncovered>0u) background_seen=frame.background.rgb;
    } else if(uncovered>0u) {
        for(uint word=0u;word<layout.masks;word++) {
            var coverage=SlotCoverage(word,false);visit_slots(layout,index,coverage);
            var bits=~coverage.bits;
            while(bits!=0u) {
                let i=word*32u+firstbitlow(bits);bits&=bits-1u;
                if(i>=samples) break;
                shading_lens=frame.post.w>0. ? sample_lens(i) : float2(0.);
                let value=camera_background(pixel,size,visibility_lattice(i)-.5);
                background_seen+=value.rgb/float(uncovered);
                if(left>0.) background+=value*(t.at(i)/left);
            }
        }
    }
#else
    if (left>0.) {
        shading_lens=-opaque.transmitted_lens;
        if (frame.post.w>0.) {for (uint i=0u;i<camera_samples[0].x;i++) {shading_lens+=t.at(i)*sample_lens(i);}}
        shading_lens/=left;
        background=camera_background(pixel,size,moment/left);
    }
#endif
    // The resolve subtracts the same exposure-weighted ambient that colour
    // contains, so seen through transparency, as much as it transmits. AO's
    // receiver is the opaque majority, background included: a sliver of a thin
    // occluder is no full-pixel surface in the occluder field.
    // With reflections the receivers also carry the pixel's source geometry
    // for reflection reconstruction: an AO-inactive surface's depth is offset
    // by one, background is 2.
    let receiver=opaque.receiver;
    let reflections=V.has_reflections && frame.projection.z==REG_REFLECTIONS_RAY;
    // AO, GI and reflections read the receivers (resolve.slang hemisphere, passes/reflections.rs).
    let write_receivers=V.has_camera_receivers && (frame.ao.y>0. || frame.flags.w!=0. || reflections);
    var metadata=select(front.a>0. || receiver.samples!=samples,1.,0.);var position=float4(0.,0.,0.,select(reflections,2.,1.));var normal=float3(0.,0.,1.);
    if (receiver.samples>0u && (receiver.samples>uncovered || reflections)) {
        // Mixed slot/background coverage is also composited; accepted ray hits need conditional radiance.
        metadata=surface_encode_visibility(metadata,objects[receiver.key.x].metadata.y);
        let clip=mul(float4(receiver.world,1.),frame.vp);
        position=float4(receiver.world,clip.z/clip.w+select(receiver.samples>uncovered,0.,1.));normal=receiver.normal;
    }
    ambient.Store(pixel,float4((opaque.ambient+transported_ambient)/float(samples),metadata));
    if (write_receivers) {receivers.Store(pixel,position);ao_normals.Store(pixel,float4(normal,asfloat(receiver.key.x)));}
    if (V.has_indirect_controls && frame.flags.w!=0.) {
        // The signed indirect correction must use the shade's actual response.
        // Quantizing this multiplier changes energy even for a constant World.
        let response=(opaque.diffuse+transported_diffuse)/float(samples);
        diffuse_response[index*3u]=response.x;diffuse_response[index*3u+1u]=response.y;diffuse_response[index*3u+2u]=response.z;
    }
    if (reflections) {opaque.write_reflection_guides(index);}
    if (VORK_AOV_CHANNELS!=0u) for (uint a=0u;a<VORK_CAMERA_AOVS;a++) {aov_values[index*VORK_CAMERA_AOVS+a]=pixel_aovs[a]/float(samples);}
    var sky=background*left;

    let shade_output=(front+float4(opaque.color,opaque.transmitted)+sky)/float(samples);
    if (!all(isfinite(shade_output))) {InterlockedMax(shader_errors[0],REG_RADIANCE_NONFINITE);}
    check_nonfinite(nonfinite,REG_NONFINITE_SHADE,shade_output);
    hdr[pixel]=shade_output;
    if (screen) {
        // Without transparency in the pixel the uncovered samples' background is the one already looked up.
        // Alpha is the opaque coverage, so glass that does not receive the world can remove the background.
#ifdef VORK_CAMERA_ENVIRONMENT
        let seen=background_seen;
#else
        let seen=uncovered==0u ? float3(0.) : t.active() ? camera_background(pixel,size,-opaque.offsets/float(uncovered)).rgb : background.rgb;
#endif
        refraction_output.Store(pixel,float4(opaque.unweighted+seen*float(uncovered),float(opaque.covered))/float(samples));
        // transmission.slang unprojects the receiver's depth at the pixel centre and
        // intersects its plane: store the triangle plane's normal and depth there.
        // Any opaque coverage makes the pixel a receiver (the capture weights it by
        // coverage); a majority rule drops sub-pixel receivers, refracting thin lines as dashes.
        var refraction=float4(0.,0.,1.,-1.);
        if (receiver.samples>0u) {
            float3 origin;float3 direction;camera_ray(sample_position(pixel,float2(0.)),size,float2(0.),origin,direction);
            let facing=dot(direction,receiver.plane);
            let centre=facing!=0. ? origin+direction*(dot(receiver.world-origin,receiver.plane)/facing) : receiver.world;
            let clip=mul(float4(centre,1.),frame.vp);refraction=float4(receiver.plane,clip.z/clip.w);
        }
        refraction_receivers[index]=refraction;
    }
}
// Screen refraction's composite pass: the pixel's transparent slots, lit and
// refracting the opaque pass's image, over its colour; with their AOVs.
void composite_pixel<V:IVariant,P:IProgram,R:ITransmittance,let AOVS:int>(uint3 band_gid) {
    let gid=uint3(band_gid.x,band_gid.y+camera_config.band,band_gid.z);
    uint w,h;hdr.GetDimensions(w,h);let size=uint2(w,h);
    if (gid.x>=w || gid.y>=h) {return;}
    let layout=SlotLayout(w*h,visibility_samples(),camera_config.packet_pixels,camera_config.packet_first_pixel);

    shading_lens=float2(0.);shading_masks=layout.masks;
    if (VORK_AOV_CHANNELS!=0u) for (uint a=0u;a<VORK_CAMERA_AOVS;a++) {pixel_aovs[a]=float4(0.);}
    if (direct_keys) {pixel_key=visibility_keys.Load(int4(gid.xy,0,0));}
    let pixel=gid.xy;composition_pixel=pixel;let index=pixel.y*size.x+pixel.x;let samples=float(visibility_samples());
    // Screen refraction stores the pixel's GI terms in the opaque pass; this
    // composite has no interface-transport receivers to add.
    var t=R();var front=float4(0.);var transported_ambient=float3(0.);var transported_diffuse=float3(0.);
    blend_transparent<V,P,R,AOVS>(layout,index,pixel,size,t,front,true,transported_ambient,transported_diffuse);
    if (!t.active()) {return;}
    if (VORK_AOV_CHANNELS!=0u) for (uint a=0u;a<VORK_CAMERA_AOVS;a++) {aov_values[index*VORK_CAMERA_AOVS+a]+=pixel_aovs[a]/samples;}
    // Screen refraction evaluates glass after the opaque source is captured.
    // Rebuild reflection records from those final evaluations before tracing.
    if (V.has_reflections && frame.projection.z==REG_REFLECTIONS_RAY) {
        var reflected=OpaqueShade<V,P,R,CameraDiffusion,AOVS>(layout,pixel,size,t);
        visit_slots(layout,index,reflected);reflected.write_reflection_guides(index);
    }
    let shade_output=hdr[pixel]+float4(front.rgb,0.)/samples;
    if (!all(isfinite(shade_output))) {InterlockedMax(shader_errors[0],REG_RADIANCE_NONFINITE);}
    check_nonfinite(nonfinite,REG_NONFINITE_SHADE,shade_output);
    hdr[pixel]=shade_output;
}

#endif


// EVALUATION_STORAGE
struct CameraTransmittance : ITransmittance {
    static const bool transparent=camera_config.transparent!=0u;
    uint first;bool covered;
    __init() {first=0u;covered=false;}
    bool active() {return covered;}
    [mutating] float at(uint i) {return covered ? asfloat(evaluation_metadata[first+i]) : 1.;}
    [mutating] void scale(uint i,float f) {
        if(!covered) {
            uint w,h;hdr.GetDimensions(w,h);
            first=camera_config.transmittance_offset+(composition_pixel.y*w+composition_pixel.x)*visibility_samples();
            for(uint j=0u;j<visibility_samples();j++) {evaluation_metadata[first+j]=asuint(1.);}
            covered=true;
        }
        evaluation_metadata[first+i]=asuint(asfloat(evaluation_metadata[first+i])*f);
    }
};
