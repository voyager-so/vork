import vork_constants;
import openpbr_core;
import dielectric_interface;
import closure;
import variant;
import frame;
__exported import bsdf_microfacet;
__exported import reflection_bsdf;
__exported import bsdf_film;
__exported import bsdf_energy;
// Device lighting prepares view-dependent energy once. Resource-free reflection
// consumers receive the same prepared factor in their retained closure record.
float3 transport_metal_compensation(Closure c,float3 view) {
    if(c.base_metalness<=0. || c.specular_roughness==0.) return float3(0.);
    let alpha=microfacet_alpha(c);
    let roughness=c.specular_roughness_anisotropy>0. ? sqrt(sqrt(alpha.x*alpha.y)) : c.specular_roughness;
    let moments=brdf_moments(float2(max(dot(c.geometry_normal,view),.0001),c.specular_roughness)).xy;
    return transport_metal_compensation(c,view,moments,brdf_average_albedo(roughness));
}
// Closure (generated, exactly OpenPBR 1.1.1) is prepended to this shader.
// Lighting consumes values only; texture and material-record reads stay outside the light loop.
// Specular reflectance from Fresnel moments m = (∫(1-f)g, ∫f·g, ∫μ(1-μ)^6·g), f=(1-μ)^5,
// μ = v·h: split-sum LUT entries for environments and finite emitters.
// Dielectric: Schlick, F0 from the modulated IOR, tinted by specular_color. Metal (OpenPBR 1.1.1 §metal): specular_weight × F82-tint Fresnel
// (Kutz et al. 2021) with F0 = base_weight·base_color and F82 tint = specular_color:
// F(μ) = Schlick(F0,μ) − b·μ(1−μ)^6, b = Schlick(F0,μ̄)(1−tint)/(μ̄(1−μ̄)^6), μ̄ = 1/7.
// Material-only Fresnel terms are prepared once, outside the emitter loop.
float3 substrate_color(Closure c) {return c.base_weight*substrate_albedo(c.base_color,c.subsurface_color,c.subsurface_weight,c.subsurface_scatter_anisotropy,c.geometry_thin_walled);}
struct SpecularResponse { float3 dielectric; float3 tint; float3 metal; float3 correction; float weight; float metalness; float3 directional_albedo; float3 metal_compensation; };
bool internal_reflection_quadrature<V:IVariant>(Closure c) {
    return V.has_solid_transmission && c.transmission_weight>0. && dielectric_ior(c)<1.;
}
bool authored_reflection_quadrature<V:IVariant>(Closure c) {
    return V.has_anisotropy && c.specular_roughness_anisotropy>0.
        || V.has_film && c.thin_film_weight>0.;
}
bool reflection_quadrature<V:IVariant>(Closure c) {
    return internal_reflection_quadrature<V>(c) || authored_reflection_quadrature<V>(c)
        || (c.geometry_thin_walled && c.transmission_weight>0.);
}
// Isotropic internal GGX uses one world estimate or balanced emitter pair per
// camera sample. Complex authored lobes retain their existing local quadrature
// pending separate variance validation.
uint reflection_quadrature_samples<V:IVariant>(Closure c) {
    return authored_reflection_quadrature<V>(c) ? REG_SURFACE_QUADRATURE_SAMPLES : 1u;
}
SpecularResponse specular_response<V:IVariant>(Closure c, float3 view) {
    let ior=dielectric_ior(c);
    let f0=float3(fresnel(1.,ior));
    var response=SpecularResponse(f0,c.specular_color,float3(0.),float3(0.),c.specular_weight,0.,float3(-1.),float3(0.));
    if (ior<1. && dot(refract(-view,c.geometry_normal,1./ior),refract(-view,c.geometry_normal,1./ior))==0.) {
        // At total internal reflection the dielectric reflects all incident energy;
        // Schlick's external-interface F0 cannot describe this boundary.
        response.dielectric=float3(1.);
    }
    if (V.metal && c.base_metalness>0.) {
        response.metalness=c.base_metalness;
        response.metal_compensation=transport_metal_compensation(c,view);
        response.metal=c.base_weight*c.base_color;
        if (V.has_tint) {response.correction=metal_edge_correction(response.metal,c.specular_color);}
    }
    if (internal_reflection_quadrature<V>(c) || (c.base_metalness<1. && c.transmission_weight<1. && c.base_weight>0. && reflection_quadrature<V>(c))) {
        response.directional_albedo=surface_directional_albedo(c,view);
    }
    return response;
}
float3 specular_reflectance(SpecularResponse r,float3 m,bool directional_dielectric) {
    // A matched dielectric interface has no reflection at any angle. Schlick's
    // unit grazing term applies only when the interface has nonzero contrast.
    let grazing=any(r.dielectric!=float3(0.)) ? m.y : 0.;
    let dielectric=directional_dielectric ? r.directional_albedo : (r.dielectric*m.x+float3(grazing))*r.tint;
    if (r.metalness<=0.) { return dielectric; }
    let metal=(1.+r.metal_compensation)*r.weight*max(r.metal*m.x+float3(m.y)-r.correction*m.z,float3(0.));
    return lerp(dielectric,metal,r.metalness);
}
// Exact std/oracle GGX and Lambert response for zero-size emitters. The moment
// response above is used only by approximate LTC/environment integrations.
// View-dependent substrate response, shared by every direction at a vertex.
bool diffuse_quadrature(Closure c) {
    return c.specular_roughness_anisotropy>0. || c.thin_film_weight>0. || (c.geometry_thin_walled && c.transmission_weight>0.);
}
float3 direct_diffuse(Closure c, float3 view, float3 directional_albedo=float3(-1.),float reflection_scale=1.) {
    var diffuse=float3(0.);
    if (c.base_weight>0. && c.base_metalness<1. && c.transmission_weight<1.) {
        var albedo=directional_albedo;
        if (any(albedo<float3(0.))) {
            albedo=diffuse_quadrature(c)
                ? surface_directional_albedo(c,view)
                : c.specular_color*dielectric_albedo(max(dot(c.geometry_normal,view),.0001),c.specular_roughness,dielectric_ior(c));
        }
        diffuse=diffuse_brdf(substrate_color(c),1.,c.base_metalness,c.transmission_weight,reflection_scale*albedo);
    }
    return diffuse;
}
float3 thin_diffuse(Closure c,float3 view,float3 directional_albedo=float3(-1.),float reflection_scale=1.) {
    if (!c.geometry_thin_walled || c.subsurface_weight<=0.) {return float3(0.);}
    var sheet=c;
    sheet.base_color=substrate_albedo(c.base_color,c.subsurface_color,c.subsurface_weight,c.subsurface_scatter_anisotropy,true,true);
    sheet.subsurface_weight=0.;
    return direct_diffuse(sheet,view,directional_albedo,reflection_scale);
}
float thin_diffuse_probability(Closure c) {
    let substrate=(1.-c.base_metalness)*(1.-c.transmission_weight);
    let reflected=dot(substrate_color(c),REG_LUMINANCE)*substrate+c.fuzz_weight;
    let transmitted=c.geometry_thin_walled ? c.base_weight*dot(c.subsurface_color,REG_LUMINANCE)*c.subsurface_weight*.5*(1.+c.subsurface_scatter_anisotropy)*substrate : 0.;
    return transmitted/max(reflected+transmitted,1e-20);
}
float3 direct_brdf_prepared(Closure c, float3 view, float3 light, uint mask, float3 diffuse, float3 metal_compensation,float reflection_scale=1.) {
    let n=c.geometry_normal;
    let nl=max(dot(n,light),0.);let nv=max(dot(n,view),.0001);
    if (nl<=0.) {return float3(0.);}

    // The lobe mask is dynamic at shared hit-lighting call sites. Do not
    // evaluate Fresnel/GGX only to select it away for diffuse-only transport.
    var specular=float3(0.);
    if((mask&REG_RAY_SPECULAR)!=0u) {
        let h=normalize(view+light);
        let nh=max(dot(n,h),0.);let vh=max(dot(view,h),0.);
        let f=compensated_surface_reflection(c,view,vh,metal_compensation);
        specular=c.specular_roughness_anisotropy>0.
            ? f*microfacet_distribution(c,h)*microfacet_visibility(c,view,light)
            : ggx_reflection(f,nv,nl,nh,c.specular_roughness);
    }
    return (select((mask&REG_RAY_DIFFUSE)!=0u,diffuse,float3(0.))+select((mask&REG_RAY_SPECULAR)!=0u,reflection_scale*specular,float3(0.)))*nl;
}
float2 specular_quadrature_point(uint i, float2 rotation, uint samples) {
    return frac(rotation+float2((float(i)+.5)/float(samples),float(reversebits(i)>>8u)*(1./16777216.)));
}

// OpenPBR's view-dependent substrate attenuation. Keep isotropic/no-film LUTs
// for the original closure; integrate new responses once before raster lights.
float3 direct_brdf(Closure c, float3 view, float3 light, uint mask, float3 directional_albedo=float3(-1.),float reflection_scale=1.) {
    if (dot(c.geometry_normal,light)<0.) {
        return (mask&REG_RAY_DIFFUSE)!=0u ? thin_diffuse(c,view,directional_albedo,reflection_scale)*(-dot(c.geometry_normal,light)) : float3(0.);
    }
    return direct_brdf_prepared(c,view,light,mask,(mask&REG_RAY_DIFFUSE)!=0u ? direct_diffuse(c,view,directional_albedo,reflection_scale) : float3(0.),(mask&REG_RAY_SPECULAR)!=0u ? transport_metal_compensation(c,view) : float3(0.),reflection_scale);
}

float3 interface_reflection<V:IVariant>(Closure c,float3 view) {
    let cosine=max(dot(c.geometry_normal,view),0.);
    if (V.has_film && c.thin_film_weight>0. && c.thin_film_thickness>0. && dielectric_ior(c)==1. && c.specular_roughness>0.) {
        return surface_directional_albedo(c,view,c.geometry_thin_walled);
    }
    let f=film_reflection(c,cosine,float3(fresnel(cosine,dielectric_ior(c))));
    return c.geometry_thin_walled ? 2.*f/(1.+f) : f;
}
