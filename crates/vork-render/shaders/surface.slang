// Vertex stream, varyings and the evaluated OpenPBR surface.
import vork_constants;
import std.normal_filter;
import frame;
import closure;
import dielectric_interface;
import reflection_bsdf;
import stream;
import hit_geometry;
import material_layout;
import material;
import openpbr_inputs;
import variant;
import vork_shadow_coverage;
#ifndef VORK_ATTRIBUTES
#define VORK_ATTRIBUTES 0
#endif
// Differential geometry shared by raster fragments and ray hits. Library
// coordinate helpers derive their sockets from this and the scene transforms.
struct Coordinates {
#if VORK_ATTRIBUTES > 0
    float4 attribute_dx[VORK_ATTRIBUTES];float4 attribute_dy[VORK_ATTRIBUTES];
#endif
    float3 position_dx; float3 position_dy;
    float3 material_filter;
    float4 shadow_offset;
    float2 uv_dx; float2 uv_dy; float2 uv1_dx; float2 uv1_dy;
    float3 rest_dx; float3 rest_dy;
    float3 normal_dx; float3 normal_dy;
    float3 incoming; float3 incoming_dx; float3 incoming_dy;
};
// Attribute storage is invocation-local scratch: evaluating another surface (for
// example a shadow candidate) replaces it. Bind the receiver at each stage entry.
MaterialUv surface_material_context(Varying v, Coordinates k) {
    let m=material_uv(v,k.uv_dx,k.uv_dy,k.uv1_dx,k.uv1_dy,k.material_filter);
#if VORK_ATTRIBUTES > 0
    for(uint a=0u;a<VORK_ATTRIBUTES;a++) { material_attribute_dx[a]=k.attribute_dx[a];material_attribute_dy[a]=k.attribute_dy[a]; }
#endif
    return m;
}
// TEXCOORD_1, or zero for draws without one (tangent w carries the flag).
float2 varying_uv1(Varying v) {return select(abs(v.TANGENT.w)>=2.,float2(0.),v.TEXCOORD_1);}
Coordinates raster_coordinates(inout Varying v, bool object_front, float3 incident) {
    // The derivative cross has screen orientation; recover the object's winding side.
    let face=cross(ddx(v.world),ddy(v.world));
    let winding=face*select(dot(face,incident)>=0.,1.,-1.)*select(object_front,1.,-1.);
    v.NORMAL=winding_normal(v.NORMAL,winding);
    let geometric=surface_geometric(v,object_front);
    Coordinates k;k.material_filter=float3(0.);k.shadow_offset=float4(0.);
    let uv1=varying_uv1(v);
    k.position_dx=ddx(v.world);k.position_dy=ddy(v.world);
    k.uv_dx=ddx(v.TEXCOORD_0);k.uv_dy=ddy(v.TEXCOORD_0);k.uv1_dx=ddx(uv1);k.uv1_dy=ddy(uv1);
    k.rest_dx=ddx(v.rest);k.rest_dy=ddy(v.rest);
    k.normal_dx=ddx(geometric);k.normal_dy=ddy(geometric);
    k.incoming=incident;
    k.incoming_dx=ddx(incident);k.incoming_dy=ddy(incident);
#if VORK_ATTRIBUTES > 0
    k.attribute_dx[0]=ddx(v.attr0);k.attribute_dy[0]=ddy(v.attr0);
#endif
#if VORK_ATTRIBUTES > 1
    k.attribute_dx[1]=ddx(v.attr1);k.attribute_dy[1]=ddy(v.attr1);
#endif
#if VORK_ATTRIBUTES > 2
    k.attribute_dx[2]=ddx(v.attr2);k.attribute_dy[2]=ddy(v.attr2);
#endif
#if VORK_ATTRIBUTES > 3
    k.attribute_dx[3]=ddx(v.attr3);k.attribute_dy[3]=ddy(v.attr3);
#endif
#if VORK_ATTRIBUTES > 4
    k.attribute_dx[4]=ddx(v.attr4);k.attribute_dy[4]=ddy(v.attr4);
#endif
#if VORK_ATTRIBUTES > 5
    k.attribute_dx[5]=ddx(v.attr5);k.attribute_dy[5]=ddy(v.attr5);
#endif
#if VORK_ATTRIBUTES > 6
    k.attribute_dx[6]=ddx(v.attr6);k.attribute_dy[6]=ddy(v.attr6);
#endif
#if VORK_ATTRIBUTES > 7
    k.attribute_dx[7]=ddx(v.attr7);k.attribute_dy[7]=ddy(v.attr7);
#endif
#if VORK_ATTRIBUTES > 8
#ifdef VORK_STAGE_SCALARS
    [unroll] for(uint i=8u;i<VORK_ATTRIBUTES;i++) {let extra=attribute_extra(v,i-8u);k.attribute_dx[i]=ddx(extra);k.attribute_dy[i]=ddy(extra);}
#else
    [unroll] for(uint i=8u;i<VORK_ATTRIBUTES;i++) {k.attribute_dx[i]=ddx(v.attr_extra[i-8u]);k.attribute_dy[i]=ddy(v.attr_extra[i-8u]);}
#endif
#endif
    return k;
}
// The face normal spanned by the footprint, on the shading normal's side.
float3 coordinates_true_normal(Coordinates k, float3 geometric) {
    let face=cross(k.position_dx,k.position_dy);let length_=length(face);
    return length_>0. ? face/length_*select(dot(face,geometric)>=0.,1.,-1.) : geometric;
}
// One original-LOD tap, without the composite filter. Normalize each before averaging.
float3 normal_map_sample(Tap tap,float3 factors) {
    let n=material_sample_one(tap).xyz*2.-1.;
    return n*factors;
}
// Shared normal/tangent map frame, with fallbacks kept in structured flow.
// `tangent` retains the derived vector for tangent maps' degenerate-frame fallback.
bool surface_tangent_frame(Varying v, float3 normal, Tap t, Coordinates k, out float3 tangent, out float3 axis, out float handedness) {
    tangent=v.TANGENT.xyz;handedness=v.TANGENT.w;var valid=true;
    if (t.basis || abs(v.TANGENT.w)>=2. || dot(tangent,tangent)<1e-12) {
        let determinant=t.dx.x*t.dy.y-t.dx.y*t.dy.x;
        if (abs(determinant)<1e-12) {valid=false;}
        else {
            tangent=(k.position_dx*t.dy.y-k.position_dy*t.dx.y)/determinant;
            let bitangent=(k.position_dy*t.dx.x-k.position_dx*t.dy.x)/determinant;
            // UV derivatives already encode the orientation of both texture axes.
            // Use B = cross(N, T) * w, as for authored glTF tangents.
            // Measure against the original normal, so facing a back face (or
            // mapping the normal before a tangent map) cannot change the sign.
            handedness=sign(dot(cross(v.NORMAL,tangent),bitangent));
        }
    }
    let projected=tangent-normal*dot(normal,tangent);
    valid=valid && !(dot(projected,projected)<1e-12);
    axis=valid ? normalize(projected) : float3(0.);
    return valid;
}
// Constant vectors use the same tangent-space convention as decoded maps.
float3 material_vector_frame(Varying v, float3 normal, float3 local) {
    let projected=v.TANGENT.xyz-normal*dot(normal,v.TANGENT.xyz);
    let fallback=abs(normal.x)<.9 ? cross(float3(1.,0.,0.),normal) : cross(float3(0.,1.,0.),normal);
    let axis=normalize(dot(projected,projected)>1e-12 ? projected : fallback);
    let handedness=v.TANGENT.w==0. ? 1. : v.TANGENT.w;
    return axis*local.x+cross(normal,axis)*handedness*local.y+normal*local.z;
}
// Tangent-space normal mapping.
// Normal textures are linear XYZ; the map scale applies only to tangent X/Y.
float3 surface_normal<V:IVariant,let input:uint,let word:uint>(Varying v, float3 geometric, MaterialUv m, Coordinates k, out float2 normal_variance) {
    normal_variance=float2(0.);
    let r=material_record(v.object);let b=material_byte<V>(r,input);
    // Keep fallback lanes in structured flow: early returns before the bindless
    // taps make llvmpipe dereference a null descriptor under an empty slot mask.
    var result=geometric;
    if ((b&0x80000000u)!=0u && (b&REG_MATERIAL_TAP_INDEX_MASK)==0u) {
        let local=material_constant_color(r,word);
        let world=material_vector_frame(v,geometric,local);
        if (dot(world,world)>0.) {result=normalize(world);}
    }
    if ((b&REG_MATERIAL_TAP_INDEX_MASK)!=0u) {
        let t=material_tap(r,b&REG_MATERIAL_TAP_INDEX_MASK,m);
        float3 tangent;float3 axis;float handedness;
        if (surface_tangent_frame(v,geometric,t,k,tangent,axis,handedness)) {
            let strength=material_value(r,word);
            var factors=float3(strength,strength,1.);
            if ((b&0x80000000u)!=0u) {factors=material_constant_color(r,word);}
            float3 local;
            if (material_filtered(t)) {
                let n0=normal_map_unit(normal_map_sample(material_filter_tap(t,0u),factors));
                let n1=normal_map_unit(normal_map_sample(material_filter_tap(t,1u),factors));
                let n2=normal_map_unit(normal_map_sample(material_filter_tap(t,2u),factors));
                let n3=normal_map_unit(normal_map_sample(material_filter_tap(t,3u),factors));
                local=filtered_normal(n0,n1,n2,n3,normal_variance);
            } else {local=normal_map_sample(t,factors);}
            let world=axis*local.x+cross(geometric,axis)*handedness*local.y+geometric*local.z;
            result=dot(world,world)>0. ? normalize(world) : geometric;
            // Zero scalar strength is the exact unbound frame. Renormalizing
            // geometric * map.z can perturb it by an ULP and change path events.
            if (strength==0. && (b&0x80000000u)==0u) {
                result=geometric;normal_variance=float2(0.);
            }
        }
    }
    return result;
}
// Tangent maps share raster/ray footprints and the normal map's UV-frame rule.
float3 surface_tangent<V:IVariant,uint input,uint word>(Varying v, float3 normal, MaterialUv m, Coordinates k) {
    var result=v.TANGENT.xyz;
    if (V.has_anisotropy) {
        let r=material_record(v.object);let b=material_byte<V>(r,input);
        if ((b&0x80000000u)!=0u && (b&REG_MATERIAL_TAP_INDEX_MASK)==0u) {
            result=material_vector_frame(v,normal,material_constant_color(r,word));
        }
        if ((b&REG_MATERIAL_TAP_INDEX_MASK)!=0u) {
            let tap=material_tap(r,b&REG_MATERIAL_TAP_INDEX_MASK,m);
            float3 tangent;float3 axis;float handedness;
            let valid=surface_tangent_frame(v,normal,tap,k,tangent,axis,handedness);
            result=tangent;
            if (valid) {
                let sample_=material_sample(tap).xyz*2.-1.;
                let strength=material_value(r,word);
                var factors=float3(strength,strength,1.);
                if ((b&0x80000000u)!=0u) {factors=material_constant_color(r,word);}
                let local=sample_*factors;
                result=axis*local.x+cross(normal,axis)*handedness*local.y+normal*local.z;
            }
        }
    }
    return result;
}
bool material_front(Varying v, bool front) {
    let m=object_model(v.object);
    return front != (dot(cross(m[0].xyz,m[1].xyz),m[2].xyz)<0.);
}
// Whether this face is drawn: double-sided materials (tag z bit 1) cover both.
bool material_covers<V:IVariant>(Varying v, bool front) {return material_covers_side<V>(v,material_front(v,front));}
bool material_covers_side<V:IVariant>(uint object, bool object_front) {
    return (material_records[material_record(object)].tag.z&REG_FLAG_DOUBLE_SIDED)!=0u || object_front;
}
bool material_covers_side<V:IVariant>(Varying v, bool object_front) {
    return material_covers_side<V>(v.object,object_front);
}
// `discarded`: WGSL's discard demotes the invocation to a helper, which keeps
// running with valid derivatives. Slang's SPIR-V discard is OpKill, so surface()
// only records it and the entry discards after its last derivative consumer (opaque forward runs
// depth-Equal behind the prepass, so discarded fragments rarely get here).
// A Blend program's surface has `lobes` independently shaded closures; `closure`
// is one of them, contributing `weight` times its radiance (vork_lobe).
// `normal` is the shading normal programs receive (before the program).
struct Surface { bool front_facing; Closure closure; float3 geometric; float3 view; bool discarded; float4 aovs[VORK_AOV_SLOTS]; float4 hair0; float4 hair1; bool is_hair; float3 strand; float weight; uint lobes; float3 normal; };
// Raster reflection and transmission see the same incident-medium-relative
// interface. Path supplies its active incident and transmitted owners.
Surface transmission_interface<V:IVariant>(Surface s,float3 facet) {
    if (V.has_solid_transmission && !s.front_facing && !s.closure.geometry_thin_walled && s.closure.transmission_weight>0.) {
        let eta=dielectric_ior(s.closure);
        let normal=solid_exit_normal(s.closure.geometry_normal,facet,-s.view,eta);
        s.closure=exiting_closure(s.closure,normal);
    }
    return s;
}
// What a material's lit stage receives (Blender's Shader to RGB): the shared
// evaluation's colour of its surface (surface_lighting.slang lit_terms), its coverage and
// the part from scene lights. Scene-linear, before exposure.
struct SurfaceLighting { float3 color; float alpha; float3 direct; };
// Which lobe of a mixed surface to evaluate: lobe `index`, or with LOBE_PICK one
// picked at random by `u` in [0,1), weighted to stay unbiased.
static const uint LOBE_PICK = 0xffffffffu;
struct Lobe { uint index; float u; };
// Material programs replace the closure of program records (programs.rs generates
// the scene's dispatcher); built-in scenes apply none. `lobes`: some program
// returns a Blend, so consumers shade or pick lobes. `lit`: some program has a
// lit stage, which `apply_lit` runs for materials where `lit_material`; with
// `lit_reflects` false every lit stage returns an emission-only closure.
interface IProgram {
    // Whether any material carries a program: a scene without one compiles none of the program paths.
    static const bool surface;
    static const bool coverage;
    static const bool lobes;
    static const bool lit;
    static const bool lit_reflects;
    static Surface apply(Varying v, Surface s, Coordinates k, Lobe lobe);
    static bool lit_material(Varying v);
    static Surface apply_lit(Varying v, Surface s, Coordinates k, SurfaceLighting lighting);
}
struct NoPrograms : IProgram {
    static const bool surface = false;
    static const bool coverage = false;
    static const bool lobes = false;
    static const bool lit = false;
    static const bool lit_reflects = false;
    static Surface apply(Varying v, Surface s, Coordinates k, Lobe lobe) {return s;}
    static bool lit_material(Varying v) {return false;}
    static Surface apply_lit(Varying v, Surface s, Coordinates k, SurfaceLighting lighting) {return s;}
};
Surface surface_at<V:IVariant, P:IProgram,let decode_table:bool=true>(Varying v, bool object_front, float3 view, Coordinates k) {return surface_at<V,P,decode_table>(v,object_front,view,k,Lobe(0u,0.));}
// Installed output proofs also constrain programs, whose packed closures can
// otherwise hide constant-zero response fields from native driver optimization.
Surface surface_response<V:IVariant>(Surface s) {
    if (!V.has_coat) {s.closure.coat_weight=0.;}
    if (!V.has_fuzz) {s.closure.fuzz_weight=0.;}
    if (!V.has_tint) {s.closure.specular_color=float3(1.);}
    if (!V.has_anisotropy) {s.closure.specular_roughness_anisotropy=0.;}
    if (!V.has_coat_anisotropy) {s.closure.coat_roughness_anisotropy=0.;}
    if (!V.has_film) {s.closure.thin_film_weight=0.;}
    if (!V.has_weighted_ior) {s.closure.specular_weight=1.;}
    return s;
}
// Every proof of a key that is the union of all installed materials
// (materials.rs installed_features): the response proofs above and the lobes no
// installed closure has. A kernel with such a key states them where it receives
// a surface across a boundary the compiler does not see through; the response
// code it then inlines folds on the stated values, closures it builds itself
// included. A raster key is one material's and may omit transport on purpose
// (opaque evaluation), and no raster or hit key carries the scene unions, so
// any other key states only its response proofs here.
Surface installed_surface<V:IVariant>(Surface s) {
    s=surface_response<V>(s);
    if (!V.installed) {return s;}
    if (!V.metal) {s.closure.base_metalness=0.;}
    if (!V.has_transmission) {s.closure.transmission_weight=0.;}
    if (!V.has_subsurface) {s.closure.subsurface_weight=0.;}
    if (!V.has_hair) {s.is_hair=false;}
    return s;
}
// One surface evaluation for raster fragments and ray hits: `object_front` is
// the side of the object's own winding, `view` points toward the viewer.
Surface surface_at<V:IVariant, P:IProgram,let decode_table:bool=true>(Varying v, bool object_front, float3 view, Coordinates k, Lobe lobe) {
    let m=surface_material_context(v,k);
    let record=material_record(v.object);
    // Program records contain OpenPBR defaults, not table texture sockets
    // (materials::Pending::program). Only the authored bin skips this decoder;
    // table receivers and secondary hits retain it through the default argument.
    let opacity=decode_table ? material_opacity<V>(v,m) : select((material_records[record].tag.z&REG_FLAG_VERTEX_ALPHA)!=0u,material_value(record,MATERIAL_OPACITY_WORD)*v.COLOR_0.a,material_value(record,MATERIAL_OPACITY_WORD));
    let discarded=!material_covers_side<V>(v,object_front);
    let geometric=surface_geometric(v,object_front);
    var normal_variance=float2(0.);var n=geometric;var tangent=v.TANGENT.xyz;
    if(decode_table) {n=surface_normal<V,MATERIAL_NORMAL,MATERIAL_NORMAL_WORD>(v,geometric,m,k,normal_variance);tangent=surface_tangent<V,MATERIAL_TANGENT,MATERIAL_TANGENT_WORD>(v,n,m,k);}
    var c=closure_defaults(n,tangent);
    if(decode_table) {c=material_closure<V>(record,m,n,tangent);}
    if (any(normal_variance>float2(0.))) {
        c.specular_roughness=filtered_ggx_roughness(c.specular_roughness,normal_variance);
    }
    // OpenPBR: the coat's normal defaults to the unperturbed normal, not the base normal map.
    c.geometry_coat_normal=geometric;c.geometry_coat_tangent=v.TANGENT.xyz;
    if (decode_table && V.has_coat) {
        float2 coat_variance;
        c.geometry_coat_normal=surface_normal<V,MATERIAL_COAT_NORMAL,MATERIAL_COAT_NORMAL_WORD>(v,geometric,m,k,coat_variance);
        c.geometry_coat_tangent=surface_tangent<V,MATERIAL_COAT_TANGENT,MATERIAL_COAT_TANGENT_WORD>(v,c.geometry_coat_normal,m,k);
        if (any(coat_variance>float2(0.))) {
            c.coat_roughness=filtered_ggx_roughness(c.coat_roughness,coat_variance);
        }
    }
    c.geometry_opacity=opacity;
    Surface s={};s.front_facing=object_front;s.closure=c;s.geometric=geometric;s.view=view;s.discarded=discarded;
    s.strand=v.TANGENT.xyz;s.weight=1.;s.lobes=1u;s.normal=n;
    var evaluated=surface_response<V>(P.apply(v,s,k,lobe));
    // Keep this runtime record operation independent of optional closure state.
    // Even default-white COLOR_0 can interpolate to a value slightly off one.
    if ((material_records[record].tag.z&REG_FLAG_VERTEX_COLOR)!=0u) {
        evaluated.closure.base_color*=v.COLOR_0.rgb;
#ifndef VORK_NO_GLTF_TINT
        if (V.has_gltf_tint) { evaluated.closure.gltf_material=true; }
#endif
    }
#ifndef VORK_NO_GLTF_TINT
    if (V.has_gltf_tint && (material_records[record].tag.z&REG_FLAG_GLTF_UNLIT)!=0u) { evaluated.closure.emission_color=evaluated.closure.base_color;evaluated.closure.emission_luminance=1.; }
#endif
    return evaluated;
}
float3 surface_geometric(Varying v, bool object_front) {return finite_unit_normal(v.NORMAL,float3(0.,0.,1.))*select(object_front,1.,-1.);}
Surface surface<V:IVariant, P:IProgram>(Varying v, bool front) {return surface<V,P>(v,front,0u);}
Surface surface<V:IVariant, P:IProgram>(Varying v, bool front, uint lobe) {Coordinates k;return surface<V,P>(v,front,lobe,k);}
// `k`: the fragment's coordinates, for a later lit stage.
Surface surface<V:IVariant, P:IProgram>(Varying v, bool front, uint lobe, out Coordinates k) {
    return surface<V,P>(v,front,lobe,k,uint(frame.temporal.x),uint(frame.temporal.y));
}
// With hashed alpha decided for coverage sample `index` of `count`.
Surface surface<V:IVariant, P:IProgram>(Varying v, bool front, uint lobe, out Coordinates k, uint index, uint count) {
    let object_front=material_front(v,front);
    // Face side and the instance/material record are primitive-uniform, so
    // rejecting this whole derivative quad cannot remove a surviving helper.
    // Keep varying alpha/opacity rejection deferred until after shading.
    if (!material_covers_side<V>(v,object_front)) {discard;}
    let view=camera_view(v.world,frame);
    // Derivatives first: everything after the record read is non-uniform.
    k=raster_coordinates(v,object_front,view);
    let s=surface_at<V,P>(v,object_front,view,k,Lobe(lobe,0.));
    if (V.has_mask) {return raster_coverage(v,s,index,count);}
    return s;
}
// The alpha-tested camera coverage rule (masked depth and visibility): sidedness,
// evaluated opacity, and hashed alpha for sample `index` of `count`.
bool masked_covers(Varying v, bool front, uint index, uint count) {
    let opacity=material_opacity<Unspecialized>(v,material_uv(v));
    if (!material_covers<Unspecialized>(v,front) || opacity<=0.) {return false;}
    return material_records[material_record(v.object)].extra.y!=2. || raster_accepts(v,opacity,index,count);
}
// The same rule for a program's coverage (its surface stage).
bool program_covers<P:IProgram>(Varying v, bool front, uint index, uint count) {
    Coordinates k;return !surface<Unspecialized,P>(v,front,0u,k,index,count).discarded;
}
// Hashed alpha for sample `index` of `count`.
bool raster_accepts(Varying v, float opacity, uint index, uint count) {
    // Coverage belongs to a triangle-instance, independently of authored
    // segmentation IDs. Hash each sample separately: reusing one stratified
    // dimension across layers correlates their transmittance.
    let pixel=uint2(uint(v.position.x)^pixel_hash(v.object),uint(v.position.y)^pixel_hash(v.primitive));
    return coverage_accepts(candidate_scalar_opacity(material_record(v.object),opacity),pixel,index,count,REG_SAMPLING_STREAM_RASTER_ALPHA);
}
Surface raster_coverage(Varying v, Surface s, uint index, uint count) {
    s.discarded |= s.closure.geometry_opacity<=0.;
    if (material_records[material_record(v.object)].extra.y==2.) {
        s.discarded |= !raster_accepts(v,s.closure.geometry_opacity,index,count);
        s.closure.geometry_opacity=1.;
    }
    return s;
}
float program_shadow_opacity<P:IProgram>(Varying v, bool front) {
    let object_front=material_front(v,front);
    let view=camera_view(v.world,frame);
    let k=raster_coordinates(v,object_front,view);
    let s=surface_at<Unspecialized,P>(v,object_front,view,k);
    return candidate_scalar_opacity(material_record(v.object),s.closure.geometry_opacity);
}

// Retained lighting results shared by evaluation and composition.
__exported import radiance_terms;
struct Response { RadianceTerms radiance; ReplaceableBaselines baseline; ReflectionBsdf reflection; float3 color; float3 reflected; float3 diffuse; float3 missing_diffuse; float3 missing_scattering; bool missing_lit; float roughness; float3 normal; float3 direct; };
struct Diffusion {
    float3 light;float3 ambient;float3 radius;float3 albedo;float3 light_albedo;float3 ambient_albedo;
    // Deferred incoming-light coefficients; only the work record retains them.
    float3 missing;float3 missing_albedo;

};
