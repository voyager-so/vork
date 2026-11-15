#ifdef VORK_FIBRE
import fibre_transport;
#endif
import hair_lighting;
__exported import surface_lighting;
// Raster control stages and the raster shadow policy the decoupled shade lights with.
import vork_constants;
import openpbr_core;
import frame;
import closure;
import material_layout;
import material;
import stream;
import surface;
import shadows;
import rays;
import vork_shadow_visibility;
import bsdf;
import subsurface_transport;
import variant;
#ifdef VORK_PROBES
import probes;
#endif
#ifdef VORK_IMPLICIT_CONTROLS
import implicit_controls;
#endif
#ifdef VORK_SOFTWARE_RASTER
import software_surface;
#endif
// Control targets, attachment order.
struct Output {
#if defined(VORK_IMPLICIT_CONTROLS) || defined(VORK_SOFTWARE_RASTER)
    float depth : SV_Depth;
#endif
    uint4 id : SV_Target0; float4 normal : SV_Target1; float4 ambient : SV_Target2; float4 motion : SV_Target3;
#ifdef VORK_DENOISE
    float4 albedo : SV_Target4;
#endif
};
#ifdef VORK_FIBRE
#endif
#ifdef VORK_PROBES
#endif
struct RasterShadows<V:IVariant> : IShadows {
    static const bool probe_rays=true;
    static bool probe_visible(float3 world,float3 normal,float3 position_error,float3 probe) {
#if defined(VORK_PROBES) && defined(VORK_RAY_BINDINGS)
        return ray_probe_visible(ray_scene,ReceiverGeometry(ray_alpha,ray_hit_buffers),world,normal,position_error,probe);
#else
        return true;
#endif
    }
    static const bool subsurface=V.subsurface;
    static uint integration_count(uint index,out float3 normalization) {let count=index==0xffffffffu ? sample_count() : 1u;normalization=float3(float(count));return count;}
    static float3 integration_sample(uint tap,inout uint index,inout float3 world) {if(index==0xffffffffu) {index=tap;}return float3(1.);}
    static uint sample_count() {return uint(frame.temporal.y)*max(frame.sampling.w,1u);}
    static float3 visibility_direction(float3 world,float3 normal,LightValues record,uint light,float3 direction,float distance,float3 position_error,uint index,uint count,bool fibre=false) {
#ifdef VORK_RAY_BINDINGS
        if(V.ray_shadows && V.has_shadow_rays) {return ray_visibility_direction(ray_scene,ReceiverGeometry(ray_alpha,ray_hit_buffers),world,normal,light,direction,distance,position_error,index,count,fibre);}
#endif
        return float3(1.);
    }
    static float thickness(uint object,float3 world,float3 direction) {
        if (V.subsurface) {return ray_subsurface_exit(ray_scene,object,world,direction);}
        return -1.;
    }
};

// Controls' response. No light, shadow or thickness reads: the controls pass
// writes the closest surface's normal, ID and motion (the shade owns thickness).
Response surface_terms<V:IVariant>(Varying v, Surface s) {
    let l=layers<V>(s.closure,s.view);float3 scattering;
    return surface_terms_at<V,RasterShadows<V>>(v,s,l,specular_response<V>(l.base,s.view),material_uv(v),scattering);
}
import vork_visibility;
// Packed specular response shared by raster controls and decoupled slot controls.
bool reflection_world_specular(uint object) {
    return light_receives(frame.environment_links.x,frame.environment_links.y,objects[object].metadata.x) && (frame.environment_links.x&REG_RAY_SPECULAR)!=0u;
}
float reflection_tag(float value,uint object) {
    let world_specular=reflection_world_specular(object);
    return select(world_specular,value,-value-REG_WORLD_SPECULAR_OFFSET);
}
float reflection_roughness_payload(float roughness) {return REG_REFLECTION_PACK_BASE*round(roughness*REG_REFLECTION_PACK_LEVELS);}
float2 reflection_payload(float3 reflected,float roughness,uint object) {
    let packed=round(sqrt(clamp(reflected,float3(0.),float3(1.)))*REG_REFLECTION_PACK_LEVELS);
    let rg=packed.x+REG_REFLECTION_PACK_BASE*packed.y;
    return float2(reflection_tag(rg,object),
        surface_encode_visibility(packed.z+reflection_roughness_payload(roughness),objects[object].metadata.y));
}

// The control targets of one surface.
Output pack_response(Varying v, Response r) {
    let payload=reflection_payload(r.reflected,r.roughness,v.object);
    Output o={};
#if defined(VORK_IMPLICIT_CONTROLS) || defined(VORK_SOFTWARE_RASTER)
    o.depth=v.position.z;
#endif
    // A device-depth slope over one pixel distinguishes continuous creases
    // from separated surfaces without a distance-dependent metre threshold.
    o.id=uint4(uint(objects[v.object].opacity.y),v.object,v.primitive,asuint(abs(ddx(v.position.z))+abs(ddy(v.position.z))));o.normal=float4(r.normal,surface_encode_visibility(0.,objects[v.object].metadata.y));o.ambient=float4(r.baseline.diffuse,payload.y);
    if (motion_required()) {
        let current_clip=mul(float4(v.world,1.),frame.vp);
        o.motion.xy=(current_clip.xy/current_clip.w-v.previous_clip.xy/v.previous_clip.w)*float2(.5,-.5);
        o.motion.zw=(v.next_clip.xy/v.next_clip.w-current_clip.xy/current_clip.w)*float2(.5,-.5);
    }
#ifdef VORK_DENOISE
    // Unlit reflected response, including diffuse and specular lobes. The
    // controls' normal is the matching nominal surface's shading normal.
    o.albedo=float4(clamp(r.diffuse+r.reflected,float3(0.),float3(1.)),1.);
#endif
    return o;
}

// A mixed surface (Blend programs) shades each further lobe on its own and sums
// the weighted terms before packing. Screen passes replace the reflection of the
// lobe that reflects most, as for coat and base. Every lobe runs at every pixel,
// keeping derivatives in uniform control flow.
// Evaluate every lobe at one call site, retaining the strongest reflective controls.
Response surface_lobes<V:IVariant, P:IProgram>(Varying v, bool front, Surface first) {
    Response sum={};var best=0.;
    // The first evaluated surface also supplies controls for an empty Blend.
    for (uint i=0u;i<max(first.lobes,1u);i++) {
        var s=first;
        if (i>0u) {s=surface<V,P>(v,front,i);}
        let r=weighted(surface_terms<V>(v,s),s.weight);
        if (i==0u) {sum=r;best=reflects(r);}
        else {
            sum.color+=r.color;sum.direct+=r.direct;
            sum.radiance.direct+=r.radiance.direct;sum.radiance.emission+=r.radiance.emission;
            sum.radiance.diffuse_indirect+=r.radiance.diffuse_indirect;sum.radiance.glossy_indirect+=r.radiance.glossy_indirect;
            sum.baseline.diffuse+=r.baseline.diffuse;sum.baseline.glossy+=r.baseline.glossy;sum.reflected+=r.reflected;sum.diffuse+=r.diffuse;
            // Keep the retained BSDF and its weight with the selected reflective controls.
            if (reflects(r)>best) {best=reflects(r);sum.roughness=r.roughness;sum.normal=r.normal;sum.reflection=r.reflection;}
        }
    }
    return sum;
}
Output surface_controls<V:IVariant, P:IProgram>(Varying v, bool front, Surface s) {
    Response r;
    if (V.has_lobes && P.lobes) {r=surface_lobes<V,P>(v,front,s);}
    else {r=surface_terms<V>(v,s);}
    // The same payloads as the forward pass's lit result, without lighting it.
    if (V.has_lit && P.lit && P.lit_material(v)) {r=unreplaced(r,r);}
    var output=pack_response(v,r);
#ifdef VORK_DENOISE
    output.albedo.a=1.;
#endif
    return output;
}
// Opaque controls share the forward surface response and depth-equal coverage.
Output controls<V:IVariant, P:IProgram>(Varying v, bool front) {
#ifdef VORK_IMPLICIT_CONTROLS
    Surface accepted;
    if(implicit_control_point<V,P>(v,front,false,accepted)) return surface_controls<V,P>(v,front,accepted);
#endif
    let s=surface<V,P>(v,front);
    let o=surface_controls<V,P>(v,front,s);
    if (s.discarded) {discard;}
    return o;
}
// Controls describe the closest nonzero-opacity surface, including transmissive glass.
Output transparent_controls<V:IVariant, P:IProgram>(Varying v, bool front) {
#ifdef VORK_IMPLICIT_CONTROLS
    Surface accepted;
    if(implicit_control_point<V,P>(v,front,true,accepted)) return surface_controls<V,P>(v,front,accepted);
#endif
    var s=surface<V,P>(v,front);
    if (!V.has_mask) {s=raster_coverage(v,s,uint(frame.temporal.x),uint(frame.temporal.y)); }
    let o=surface_controls<V,P>(v,front,s);
    if (s.discarded || s.closure.geometry_opacity*objects[v.object].opacity.x<=0.) {discard;}
    return o;
}
