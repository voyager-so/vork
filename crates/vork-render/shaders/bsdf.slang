// OpenPBR response for the raster integrator: Fresnel, GGX, coat and fuzz.
import vork_constants;
import openpbr_core;
import frame;
import closure;
import dielectric_interface;
import vork_prelude;
import variant;
__exported import bsdf_response;
// LTC atlas coordinates for live coat and sheen response.
float2 ltc_uv(float2 coordinate, float size=REG_LTC_SIZE) {
    return (clamp(coordinate,float2(0.),float2(1.))*(size-1.)+.5)/size;
}
// World → local frame with X in the plane of the view and normal (LTC fits).
float3x3 ltc_basis(float3 n, float3 view) {
    let fallback=normalize(cross(n,select(abs(n.y)>.99,float3(0.,0.,1.),float3(0.,1.,0.))));
    // Project with the true cosine over |n|²: an interpolated normal a few ulps
    // short of unit length otherwise leaves a residual along n itself. Within
    // rounding of the normal (sin² < 1e-10) the view picks no tangent; the
    // isotropic fit at normal incidence takes any.
    let projected=view-n*(dot(view,n)/dot(n,n));
    let tangent=select(dot(projected,projected)>1e-10*dot(view,view),normalize(projected),fallback);
    return transpose(float3x3(tangent,cross(n,tangent),n));
}
// The quadrature's rotations of this exposure sample: its visibility samples
// (frame.sampling.w) of the render's, as the per-sample estimator rotated it.
// Centroid shading integrates the visibility budget. A receiver already traced
// per sample supplies its exposure-wide index and evaluates just that rotation.
uint quadrature_rotations(uint sample) {return sample==0xffffffffu ? max(frame.sampling.w,1u) : 1u;}
float2 quadrature_rotation(uint j,uint sample,uint count) {
    let rotations=max(frame.sampling.w,1u);
    let index=sample==0xffffffffu ? uint(frame.temporal.x)*rotations+j : sample;
    return sample_2d(uint2(0u),index,count,REG_SAMPLING_STREAM_SURFACE_QUADRATURE);
}
// OpenPBR layering (formulas from the specification's MaterialX reference graph,
// open_pbr_surface.mtlx): fuzz over coat over the base substrate. Evaluated once per
// shaded sample; zero-weight layers leave `base` as the material's own closure.
struct Layers { Closure base; float3 scale; float3 transmittance; float throughput; float coat_f0; float coat_fresnel; float fuzz_albedo; float3x3 fuzz_basis; float3x3 fuzz_matrix; bool coat_albedo; };
float coat_directional_albedo<V:IVariant=Unspecialized>(Closure c,float3 view) {
    if(!V.has_coat_anisotropy || c.coat_roughness_anisotropy==0.) return dielectric_albedo(max(dot(c.geometry_coat_normal,view),.0001),c.coat_roughness,c.coat_ior);
    let coat=transport_coat_closure(c);
    return surface_directional_albedo(coat,view,false).x;
}
// The coat and fuzz layers are compiled in only for variants whose materials have them
// (V.layered); dielectric variants compile without the F82 metal path (V.metal).
Layers layers<V:IVariant>(Closure c, float3 view, float coat_albedo=-1.) {
    let identity=float3x3(float3(1.,0.,0.),float3(0.,1.,0.),float3(0.,0.,1.));
    var l=Layers(c,float3(1.),float3(1.),1.,0.,0.,0.,identity,identity,coat_albedo>=0.);
    if (V.layered && V.has_coat && c.coat_weight>0.) {
        // Independent std GGX coat: keep the base roughness and IOR unchanged.
        l.coat_f0=fresnel(1.,c.coat_ior);
        let n=c.geometry_coat_normal;let nv=max(dot(n,view),.0001);
        l.transmittance=float3(coat_transmission(c.coat_weight,nv,c.coat_ior));
        // Reserve the coat's integrated reflected energy before the substrate.
        // The retained independent method attenuates both external directions.
        l.scale=coat_albedo>=0. ? float3(1.-coat_albedo) : l.transmittance;
        let coat_roughness=specular_roughness(c.coat_roughness);
        let uv=ltc_uv(float2(coat_roughness,sqrt(1.-min(nv,1.))));
        let amplitude=ltc_amplitude.SampleLevel(env_sampler,uv,0.);
        l.coat_fresnel=l.coat_f0*amplitude.x+(1.-l.coat_f0)*amplitude.y;
    }
    if (V.layered && V.has_fuzz && c.fuzz_weight>0.) {
        let n=c.geometry_normal;let nv=clamp(dot(n,view),.0001,1.);
        l.fuzz_albedo=fuzz_albedo(nv,c.fuzz_roughness);
        l.fuzz_basis=ltc_basis(n,view);
        l.fuzz_matrix=fuzz_ltc(nv,c.fuzz_roughness);
        // Zero A marks an absent fit; keep its transform nonsingular.
        if (l.fuzz_matrix[0].x<=0.) {
            l.fuzz_albedo=0.;l.fuzz_matrix=identity;
        }
        l.throughput=1.-c.fuzz_weight*l.fuzz_albedo;
    }
    return l;
}
// OpenPBR emitted radiance, shared by visible surfaces and light samples.
float3 surface_emission(Closure c,Layers l) {
    return c.emission_color*c.emission_luminance*l.transmittance*l.throughput;
}
// Zeltner, Burley & Chiang 2022 LTC fit data, shared with Cycles (see
// src/ltc/README.md). Three 32² planes in the amplitude texture's blue channel.
float fuzz_coefficient(float nv, float roughness, float2 tile) {
    let uv=(ltc_uv(float2(nv,roughness),REG_LTC_SIZE*.5)+tile)*.5;
    return ltc_amplitude.SampleLevel(env_sampler,uv,0.).z;
}
float fuzz_albedo(float nv, float roughness) {
    return fuzz_coefficient(nv,roughness,float2(0.,1.));
}
float3x3 fuzz_ltc(float nv, float roughness) {
    let a=fuzz_coefficient(nv,roughness,float2(0.,0.));
    let b=fuzz_coefficient(nv,roughness,float2(1.,0.));
    return float3x3(float3(a,0.,0.),float3(0.,a,0.),float3(b,0.,1.));
}
// Coat (dielectric GGX reflection) and fuzz for one zero-size emitter.
float3 layer_direct<V:IVariant>(Layers l, Closure c, float3 view, float3 light, uint mask) {
    // Coat is a specular lobe; fuzz (sheen) is diffuse, as in Blender light linking.
    var result=float3(0.);
    if (V.has_coat && c.coat_weight>0. && (mask&REG_RAY_SPECULAR)!=0u) {
        let n=c.geometry_coat_normal;let h=normalize(view+light);
        let nl=max(dot(n,light),0.);let nv=max(dot(n,view),.0001);
        if (nl>0.) {
            if(V.has_coat_anisotropy && c.coat_roughness_anisotropy>0.) {
                let coat=transport_coat_closure(c);
                result+=l.throughput*c.coat_weight*fresnel(max(dot(view,h),0.),c.coat_ior)
                    *microfacet_distribution(coat,h)*microfacet_visibility(coat,view,light)*nl;
            } else result+=l.throughput*coat_brdf(c.coat_weight,c.coat_roughness,c.coat_ior,nv,nl,max(dot(n,h),0.),max(dot(view,h),0.))*nl;
        }
    }
    if (V.has_fuzz && c.fuzz_weight>0. && (mask&REG_RAY_DIFFUSE)!=0u) {
        // The sheen LTC: D(ω) = D_o(M⁻¹ω)·|M⁻¹|/‖M⁻¹ω‖³ for a unit ω.
        let o=mul(mul(light,l.fuzz_basis),l.fuzz_matrix);let len2=dot(o,o);
        result+=c.fuzz_weight*c.fuzz_color*l.fuzz_albedo*max(o.z,0.)/REG_PI*pow(l.fuzz_matrix[0].x/len2,2.);
    }
    return result;
}
// Fuzz over (coat over base): base·scale·throughput + coat·throughput + fuzz,
// where `layer` already holds coat·throughput + fuzz.
float3 layered(Layers l, float3 base, float3 layer) {
    return l.throughput*l.scale*base+layer;
}

// Transport interface shared by ray integrators. Values include the projected
// cosine; PDFs are solid-angle densities. Mixture sampling always evaluates the
// selected lobe with its selection probability, preserving per-event visibility.
struct BsdfSample { float3 direction; float3 weight; float pdf; uint event; };
bool transport_tir(Closure c, float3 view) {
    if (c.geometry_thin_walled) {return false;}
    let d=refract(-view,c.geometry_normal,1./dielectric_ior(c));
    return dot(d,d)==0.;
}
// The three base probabilities are conditional on not selecting the coat.
// Keep their interface Fresnel branching separate from the independent layer.
float3 transport_probabilities(Closure c, float3 view, out float coat_probability) {
    var diffuse=max(dot(c.base_color,REG_LUMINANCE)*c.base_weight*(1.-c.base_metalness)*(1.-c.transmission_weight)+c.fuzz_weight,0.);
    if (c.subsurface_weight>0.) {
        diffuse=max(dot(c.base_weight*lerp(c.base_color,c.subsurface_color,c.subsurface_weight),REG_LUMINANCE)*(1.-c.base_metalness)*(1.-c.transmission_weight)+c.fuzz_weight,0.);
    }
    let tir=c.specular_roughness==0. && transport_tir(c,view);
    let transmission=max(c.transmission_weight*(1.-c.base_metalness),0.);
    // Fresnel-weight transmitting surfaces without removing rough-interface
    // support across the macro-normal critical angle (microfacets may transmit).
    var f=fresnel(max(dot(c.geometry_normal,view),0.),dielectric_ior(c));
    if (c.thin_film_weight>0.) {f=dot(film_dielectric(max(dot(c.geometry_normal,view),0.),dielectric_ior(c),c.thin_film_ior,c.thin_film_thickness),REG_LUMINANCE)*c.thin_film_weight+f*(1.-c.thin_film_weight);}
    var reflected=f;
    if (c.geometry_thin_walled) {reflected=lerp(f,sheet_reflectance(f),c.transmission_weight);f=sheet_reflectance(f);}
    if (c.specular_roughness>0.) {f=clamp(f,.05,.95);}
    // Reserve samples in proportion to reflected energy for opaque dielectrics
    // too. Treating specular_weight=1 as unit reflectance wastes most paths on
    // the ~4% interface of ordinary paint/plastic. Metals retain full support
    // for the colored F82 lobe. The existing rough-transmission support floor
    // below is separate from this reflection-energy estimate.
    let dielectric=reflected*max(dot(c.specular_color,REG_LUMINANCE),0.);
    let coat=c.coat_weight>0. ? c.coat_weight*fresnel(max(dot(c.geometry_coat_normal,view),0.),c.coat_ior) : 0.;
    let reflection=max(lerp(dielectric,c.specular_weight,c.base_metalness),0.)+select(tir,transmission,0.);
    let refractive=select(tir,0.,transmission*(1.-f));
    let total=diffuse+reflection+refractive;
    coat_probability=coat/max(total+coat,1e-20);
    return select(total>0.,float3(diffuse,reflection,refractive)/max(total,1e-20),float3(0.));
}
float3 transport_probabilities(Closure c, float3 view) {
    float coat_probability;return transport_probabilities(c,view,coat_probability);
}
// Retain direction-independent transport work across sampling and both NEE
// strategies. Directional response and joint event PDFs stay with each query.
struct TransportPreparation { float3 probabilities; float coat_probability; float3 diffuse; float coat_albedo; float3 metal_compensation; float3 smooth_normal; float bump_alpha_squared; float3 interface_normal; };
// Rotate an invalid mapped interface normal toward the incident face until
// its mirror ray is above that face (Cycles' geometric reflection criterion).
// In the face-normal plane R.z=Iz*cos(2 theta)+Ix*sin(2 theta). Solve the
// half-angle directly, avoiding an iterative blend or a material roughness floor.
float3 transport_valid_interface_normal(float3 normal,float3 geometric,float3 view) {
    let iz=dot(view,geometric);if (iz<=0.) {return normal;}
    // Reflection is invariant under n -> -n, but the visible-normal density
    // and refraction require the incident-facing orientation of that plane.
    if (dot(normal,geometric)<0.) {normal=-normal;}
    let threshold=min(.9*iz,.01);
    if (dot(normal,view)>0. && dot(reflect(-view,normal),geometric)>=threshold) {return normal;}
    let tangent=normal-geometric*dot(normal,geometric);let tangent_squared=dot(tangent,tangent);
    if (tangent_squared<=0.) {return geometric;}
    let axis=tangent*rsqrt(tangent_squared);let ix=dot(view,axis);let length_squared=ix*ix+iz*iz;
    let cosine_squared=clamp(.5*(1.+(iz*threshold-ix*sqrt(max(length_squared-threshold*threshold,0.)))/length_squared),0.,1.);
    return sqrt(1.-cosine_squared)*axis+sqrt(cosine_squared)*geometric;
}
Closure transport_interface_closure(Closure c, TransportPreparation prepared) {
    c.geometry_normal=prepared.interface_normal;return c;
}
TransportPreparation transport_prepare<V:IVariant=Unspecialized>(Closure c, float3 view, bool coat_albedo, float3 smooth_normal, float3 geometric) {
    // Directional coat reflectance is view-local, just like the diffuse budget.
    let coat=coat_albedo ? (V.has_coat && c.coat_weight>0. ? c.coat_weight*coat_directional_albedo<V>(c,view) : 0.) : -1.;
    var base_c=c;base_c.geometry_normal=transport_valid_interface_normal(c.geometry_normal,geometric,view);
    float coat_probability;let probabilities=transport_probabilities(base_c,view,coat_probability);
    let deviation=abs(dot(smooth_normal,c.geometry_normal));
    // Estevez, Lecocq & Stein (Ray Tracing Gems, ch. 12), Eq. 2.
    let bump_alpha_squared=clamp((1.-deviation*deviation)/max(8.*deviation*deviation,1e-20),0.,1.);
    return TransportPreparation(probabilities,coat_probability,direct_diffuse(base_c,view),coat,transport_metal_compensation(base_c,view),smooth_normal,bump_alpha_squared,base_c.geometry_normal);
}
TransportPreparation transport_prepare<V:IVariant=Unspecialized>(Closure c, float3 view, bool coat_albedo=false) {
    // Closure-only queries have no separate face: retain the authored interface.
    return transport_prepare<V>(c,view,coat_albedo,c.geometry_normal,c.geometry_normal);
}
// Bump shadowing changes the diffuse response, not its cosine proposal/PDF.
// Use the original smooth normal, independently of the coat's authored normal.
float transport_bump_shadow(Closure c, float3 direction, TransportPreparation prepared) {
    let cosine=dot(prepared.smooth_normal,direction);
    // Prevent diffuse/fuzz energy leaking across disagreeing smooth/mapped hemispheres.
    if (cosine*dot(prepared.smooth_normal,c.geometry_normal)*dot(c.geometry_normal,direction)<0.) {return 0.;}
    // Unmapped surfaces have no bump shadowing, including the grazing limit.
    if (prepared.bump_alpha_squared==0.) {return 1.;}
    let mu=abs(cosine);
    return mu>0. ? 2.*mu/(mu+sqrt(prepared.bump_alpha_squared+(1.-prepared.bump_alpha_squared)*mu*mu)) : 0.;
}
// A rough interface shares one visible microfacet between Fresnel branches.
// Choosing the event before the microfacet wastes most grazing/exiting samples
// on total internal reflection and gives the remaining paths large weights.
bool transport_shared_interface(Closure c) {
    return transport_rough_transmission(c) && c.transmission_weight>0. && c.base_metalness<1.;
}
float transport_interface_reflection(Closure c,float3 view,float3 h) {
    var f=fresnel(max(dot(view,h),0.),dielectric_ior(c));
    if (c.thin_film_weight>0.) {f=lerp(f,dot(film_dielectric(max(dot(view,h),0.),dielectric_ior(c),c.thin_film_ior,c.thin_film_thickness),REG_LUMINANCE),c.thin_film_weight);}
    if (c.geometry_thin_walled) {f=sheet_reflectance(f);}
    let reflection=max(lerp(f*max(dot(c.specular_color,REG_LUMINANCE),0.),c.specular_weight,c.base_metalness),0.);
    let transmission=max(c.transmission_weight*(1.-c.base_metalness)*(1.-f),0.);
    return reflection+transmission>0. ? reflection/(reflection+transmission) : 0.;
}
float transport_transmission_probability(Closure c,float3 view,float3 direction,float3 probabilities) {
    if (!transport_shared_interface(c)) {return probabilities.z;}
    var h=normalize(select(c.geometry_thin_walled,view-direction,view+dielectric_ior(c)*direction));
    if (dot(c.geometry_normal,h)<0.) {h=-h;}
    return (1.-probabilities.x)*(1.-transport_interface_reflection(c,view,h));
}
// Joint event/direction density. Path visibility and light contribution are
// event-specific, so MIS must compare the selected event's density rather
// than the sum of overlapping diffuse and glossy sampling distributions.
// The coat has its own normal, tangent and GGX distribution. Reuse the
// visible-normal machinery with a reflection-only closure, independent of base.
Closure transport_coat_closure(Closure c) {
    // Initialize an independent reflection-only interface. No substrate optics
    // (thin sheet, transmission, film, metal or weighted IOR) belong to the coat.
    Closure coat={};
    coat.geometry_normal=c.geometry_coat_normal;coat.geometry_tangent=c.geometry_coat_tangent;
    coat.specular_roughness=c.coat_roughness;coat.specular_roughness_anisotropy=c.coat_roughness_anisotropy;
    coat.specular_weight=1.;coat.specular_color=float3(1.);coat.specular_ior=c.coat_ior;
    return coat;
}
float transport_pdf_mask(Closure c, float3 view, float3 direction, uint mask, TransportPreparation prepared) {
    let base=transport_interface_closure(c,prepared);let n=base.geometry_normal;let nl=dot(n,direction);let p=prepared.probabilities;
    let q=prepared.coat_probability;
    var pdf=(1.-q)*select((mask&REG_RAY_DIFFUSE)!=0u,p.x*(dot(c.geometry_normal,direction)>0. ? 1.-thin_diffuse_probability(c) : thin_diffuse_probability(c))*abs(dot(c.geometry_normal,direction))/REG_PI,0.);
    if ((mask&REG_RAY_SPECULAR)!=0u && q>0. && c.coat_roughness>0. && dot(c.geometry_coat_normal,view)>0. && dot(c.geometry_coat_normal,direction)>0.) {
        let coat=transport_coat_closure(c);let h=normalize(view+direction);
        pdf+=q*microfacet_distribution(coat,h)*transport_visible_scale(coat,view)/4.;
    }
    if (nl<=0.) {return pdf+(1.-q)*select((mask&REG_RAY_TRANSMISSION)!=0u,transport_transmission_probability(base,view,direction,p)*transport_transmission(base,view,direction,layers<Unspecialized>(c,view,prepared.coat_albedo)).pdf,0.);}
    let h=normalize(view+direction);
    let d=microfacet_distribution(base,h);
    let reflection=transport_shared_interface(base) ? (1.-p.x)*transport_interface_reflection(base,view,h) : p.y;
    return pdf+(1.-q)*select((mask&REG_RAY_SPECULAR)!=0u,reflection*(d*transport_visible_scale(base,view)/4.),0.);
}
float transport_pdf_mask(Closure c, float3 view, float3 direction, uint mask) {
    return transport_pdf_mask(c,view,direction,mask,transport_prepare(c,view));
}
float transport_pdf(Closure c, float3 view, float3 direction) {
    return transport_pdf_mask(c,view,direction,REG_RAY_DIFFUSE|REG_RAY_SPECULAR|REG_RAY_TRANSMISSION);
}
// Smooth and equal-IOR transmission retain the zero-density convention.
bool transport_rough_transmission(Closure c) {
    return c.specular_roughness>0. && dielectric_ior(c)!=1.;
}
struct TransmissionResponse {float3 value;float pdf;};
TransmissionResponse transport_transmission(Closure c, float3 view, float3 direction, Layers l) {
    TransmissionResponse result={};
    if (!transport_rough_transmission(c)) {return result;}
    let n=c.geometry_normal;
    var h=normalize(select(c.geometry_thin_walled,view-direction,view+dielectric_ior(c)*direction));
    if (dot(n,h)<0.) {h=-h;}
    var f=0.;
    if (c.geometry_thin_walled) {f=ggx_sheet_transmission(dot(n,view),dot(n,direction),dot(n,h),dot(view,h),c.specular_roughness,dielectric_ior(c),result.pdf);}
    else {f=ggx_transmission(dot(n,view),dot(n,direction),dot(n,h),dot(view,h),dot(direction,h),c.specular_roughness,dielectric_ior(c),result.pdf);}
    var transmission_scale=float3(1.);
    if (c.specular_roughness_anisotropy>0.) {
        let a=specular_alpha(c.specular_roughness);
        let original_d=distribution(max(dot(n,h),0.),a);
        let ratio=microfacet_distribution(c,h)/max(original_d,1e-30);
        result.pdf*=ratio;
        f*=ratio*microfacet_visibility(c,view,direction)/max(smith_visibility(max(dot(n,view),.0001),abs(dot(n,direction)),a,1e-20),1e-30);
    }
    if (c.thin_film_weight>0.) {
        let base_f=fresnel(abs(dot(view,h)),dielectric_ior(c));
        var film_f=lerp(float3(base_f),film_dielectric(abs(dot(view,h)),dielectric_ior(c),c.thin_film_ior,c.thin_film_thickness),c.thin_film_weight);
        var original_f=base_f;
        if (c.geometry_thin_walled) {film_f=2.*film_f/(1.+film_f);original_f=sheet_reflectance(base_f);}
        transmission_scale=(1.-film_f)/max(1.-original_f,1e-20);
    }
    if (result.pdf>0.) {
        result.pdf*=transport_visible_scale(c,view)*abs(dot(view,h))/dot(n,h);
    }
    let tint=transmission_scale*surface_transmission_tint(c);
    let base=l.throughput*tint*c.transmission_weight*(1.-c.base_metalness)*f*abs(dot(n,direction));
    result.value=l.coat_albedo ? l.scale*base : compose_coat(base,float3(0.),c.coat_weight,dot(c.geometry_coat_normal,view),dot(c.geometry_coat_normal,direction),c.coat_ior);
    return result;
}
TransmissionResponse transport_transmission(Closure c, float3 view, float3 direction) {
    return transport_transmission(c,view,direction,layers<Unspecialized>(c,view));
}
float3 transport_eval_mask(Closure c, float3 view, float3 direction, uint mask, TransportPreparation prepared) {
    let l=layers<Unspecialized>(c,view,prepared.coat_albedo);
    let base_c=transport_interface_closure(l.base,prepared);
    let ni=dot(base_c.geometry_normal,direction);
    let transmitted=ni<=0. && (mask&REG_RAY_TRANSMISSION)!=0u ? transport_transmission(base_c,view,direction,l).value : float3(0.);
    // A delta coat has mass only in continuation, never in an NEE density.
    var layer_c=c;if (c.coat_roughness==0. || dot(c.geometry_coat_normal,view)<=0.) {layer_c.coat_weight=0.;}
    let bump=transport_bump_shadow(c,direction,prepared);
    var base=direct_brdf_prepared(base_c,view,direction,mask&REG_RAY_SPECULAR,float3(0.),prepared.metal_compensation);
    if ((mask&REG_RAY_DIFFUSE)!=0u) {base+=dot(c.geometry_normal,direction)>0. ? prepared.diffuse*bump*dot(c.geometry_normal,direction) : thin_diffuse(c,view)*transport_bump_shadow(c,-direction,prepared)*(-dot(c.geometry_normal,direction));}
    let diffuse_mask=dot(c.geometry_normal,direction)>0. ? mask&REG_RAY_DIFFUSE : 0u;
    let layer=layer_direct<Unspecialized>(l,layer_c,view,direction,mask&REG_RAY_SPECULAR)
        +bump*layer_direct<Unspecialized>(l,layer_c,view,direction,diffuse_mask);
    return transmitted+(l.coat_albedo ? layered(l,base,layer) : compose_coat(l.throughput*base,layer,c.coat_weight,
        dot(c.geometry_coat_normal,view),dot(c.geometry_coat_normal,direction),c.coat_ior));
}
float3 transport_eval_mask(Closure c, float3 view, float3 direction, uint mask) {
    return transport_eval_mask(c,view,direction,mask,transport_prepare(c,view));
}
float3 transport_eval(Closure c, float3 view, float3 direction) {
    return transport_eval_mask(c,view,direction,REG_RAY_DIFFUSE|REG_RAY_SPECULAR|REG_RAY_TRANSMISSION);
}
// Discrete straight-through sheet energy, shared by continuation and light connections.
float3 transport_delta_transmittance(Closure c,float3 view,TransportPreparation prepared) {
    let base=transport_interface_closure(c,prepared);
    let n=prepared.interface_normal;
    var interface_fresnel=fresnel(max(dot(n,view),0.),dielectric_ior(base));
    if (base.geometry_thin_walled) {interface_fresnel=sheet_reflectance(interface_fresnel);}
    var transmitted=float3(1.-interface_fresnel);
    if (c.thin_film_weight>0. && c.thin_film_thickness>0.) {
        var film_f=film_dielectric(max(dot(n,view),0.),dielectric_ior(base),c.thin_film_ior,c.thin_film_thickness);
        if (base.geometry_thin_walled) {film_f=2.*film_f/(1.+film_f);}
        transmitted=lerp(transmitted,1.-film_f,c.thin_film_weight);
        // Equal exterior/substrate IOR is a direction delta even when the
        // reflecting film is rough. Reserve its integrated reflected energy.
        if (dielectric_ior(base)==1. && base.specular_roughness>0.) {transmitted=1.-surface_directional_albedo(base,view,base.geometry_thin_walled);}
    }
    let l=layers<Unspecialized>(c,view,prepared.coat_albedo);
    return surface_transmission_tint(base)
        *base.transmission_weight*(1.-base.base_metalness)*transmitted*l.scale*l.throughput;
}
// Final's straight light connection transports the directional transmission
// integral, not the null-event mass. A rough sheet has no null delta, but its
// GGX BTDF still carries light. Keep path NEE on transport_null_transmittance:
// it has a competing sampled continuation and must not count this approximation.
float3 transport_shadow_transmittance<V:IVariant=Unspecialized>(Closure c,float3 view,float3 smooth_normal,float3 geometric,float2 uv) {
    if(c.transmission_weight<=0. || c.base_metalness>=1.) return float3(0.);
    let prepared=transport_prepare<V>(c,view,true,smooth_normal,geometric);
    if(!c.geometry_thin_walled || !transport_rough_transmission(c))
        return transport_delta_transmittance(c,view,prepared);
    let base=transport_interface_closure(c,prepared);
    let l=layers<Unspecialized>(c,view,prepared.coat_albedo);
    // One complete-support VNDF draw per crossing. The connection supplies an
    // independent exposure-varying sample; rejected directions contribute zero.
    let h=transport_visible_normal(base,view,uv);
    let direction=-reflect(-view,h);
    if(dot(base.geometry_normal,direction)>=0. || dot(view,h)<=0.) return float3(0.);
    let response=transport_transmission(base,view,direction,l);
    return response.pdf>0. ? response.value/response.pdf : float3(0.);
}

// A nonabsorbing matched solid has the same straight delta as a sheet.
// This predicate consumes relative interface IOR, after optical preparation.
// Absorbing solids need an ordered segment/occupancy connection, not an
// all-candidate product of interface weights.
bool transport_null_interface(Closure c) {
    if(c.transmission_weight<=0.) return false;
    if(c.geometry_thin_walled) return !transport_rough_transmission(c);
    return dielectric_ior(c)==1. && (c.transmission_depth<=0. || all(c.transmission_color==float3(1.)));
}
bool transport_null_boundary(Closure c,float incident_ior) {
    // A straight shadow connection does not carry an ordered interior set.
    return transport_null_interface(c) && (c.geometry_thin_walled || incident_ior==1.);
}
// A null event preserves the preceding scattering vertex and its source MIS.
// It cannot also own a new finite-source endpoint at this interface.
bool transport_null_event(Closure c,float incident_ior,BsdfSample sample) {
    return transport_null_boundary(c,incident_ior) && sample.event==REG_RAY_TRANSMISSION && sample.pdf==0.;
}
float3 transport_null_transmittance<V:IVariant=Unspecialized>(Closure c,float3 view,DielectricSegment incident,float3 smooth_normal,float3 geometric) {
    let transmitted=dielectric_material_ior(c,incident,vork_path_wavelength);
    c=dielectric_interface(c,incident.ior,c.specular_ior,transmitted/incident.ior);
    if(!transport_null_boundary(c,incident.ior)) return float3(0.);
    return transport_delta_transmittance(c,view,transport_prepare<V>(c,view,true,smooth_normal,geometric));
}

BsdfSample transport_sample(Closure c, float3 view, float3 random, TransportPreparation prepared) {
    BsdfSample result={};let p=prepared.probabilities;let n=prepared.interface_normal;
    let q=prepared.coat_probability;
    let coat_selected=random.z<q;
    if (coat_selected) {
        let coat=transport_coat_closure(c);
        if (dot(coat.geometry_normal,view)<=0.) {return result;}
        result.event=REG_RAY_SPECULAR;
        if (c.coat_roughness==0.) {
            result.direction=reflect(-view,coat.geometry_normal);
            result.weight=float3(layers<Unspecialized>(c,view,prepared.coat_albedo).throughput*c.coat_weight*fresnel(dot(coat.geometry_normal,view),c.coat_ior)/q);
            result.pdf=0.;return result;
        }
        result.direction=reflect(-view,transport_visible_normal(coat,view,random.xy));
        if (dot(coat.geometry_normal,result.direction)<=0.) {return BsdfSample();}
    } else {
        let base=transport_interface_closure(c,prepared);
        random.z=(random.z-q)/max(1.-q,1e-20);
        if (random.z<p.x) {
            let radius=sqrt(random.x);let phi=REG_TAU*random.y;
            let back=thin_diffuse_probability(c);
            let selected=random.z/max(p.x,1e-20);
            let normal=selected<back ? -c.geometry_normal : c.geometry_normal;
            result.direction=basis_direction(normal,float3(radius*cos(phi),radius*sin(phi),sqrt(1.-random.x)));
            result.event=REG_RAY_DIFFUSE;
        } else if (transport_shared_interface(base)) {
            let h=transport_visible_normal(base,view,random.xy);
            let reflection=transport_interface_reflection(base,view,h);
            if ((random.z-p.x)/(1.-p.x)<reflection) {
                result.direction=reflect(-view,h);result.event=REG_RAY_SPECULAR;
            } else {
                result.direction=select(base.geometry_thin_walled,-reflect(-view,h),refract(-view,h,1./dielectric_ior(base)));
                result.event=REG_RAY_TRANSMISSION;
                if (dot(result.direction,result.direction)==0. || dot(n,result.direction)>=0.) {return BsdfSample();}
            }
        } else if (random.z<p.x+p.y) {
            let h=transport_visible_normal(base,view,random.xy);
            result.direction=reflect(-view,h);result.event=REG_RAY_SPECULAR;
        } else if (p.z>0.) {
            let eta=1./dielectric_ior(base);
            if (transport_rough_transmission(base)) {
                let h=transport_visible_normal(base,view,random.xy);
                if (dot(view,h)<=0.) {return result;}
                result.direction=select(base.geometry_thin_walled,-reflect(-view,h),refract(-view,h,eta));result.event=REG_RAY_TRANSMISSION;
                if (dot(result.direction,result.direction)==0. || dot(n,result.direction)>=0.) {return result;}
                let response=transport_transmission(base,view,result.direction,layers<Unspecialized>(c,view,prepared.coat_albedo));
                result.pdf=(1.-q)*p.z*response.pdf;
                if (result.pdf>0.) {result.weight=response.value/result.pdf;}
                return result;
            }
            var direction=refract(-view,n,eta);
            if (base.geometry_thin_walled) {direction=-view;}
            // TIR is assigned to the reflection distribution above, never added
            // again as a second transmission-selected reflection event.
            result.direction=direction;result.event=REG_RAY_TRANSMISSION;
            result.weight=transport_delta_transmittance(c,view,prepared)/((1.-q)*p.z);
            if (!base.geometry_thin_walled) {result.weight*=eta*eta;}
            result.pdf=0.;return result;
        } else {return result;}
    }
    if (!coat_selected && result.event==REG_RAY_SPECULAR && dot(n,result.direction)<=0.) {return BsdfSample();}
    result.pdf=transport_pdf_mask(c,view,result.direction,result.event,prepared);
    if (result.pdf>0.) {result.weight=transport_eval_mask(c,view,result.direction,result.event,prepared)/result.pdf;}
    return result;
}

BsdfSample transport_sample(Closure c, float3 view, float3 random) {
    return transport_sample(c,view,random,transport_prepare(c,view));
}

// Path albedo composition spends coat reflection once; the independent
// method retains the incoming/outgoing Fresnel product shared with std.
float3 layered_direct<V:IVariant>(Layers l, Closure c, float3 base, float3 view, float3 light, uint mask) {
    let coat=layer_direct<V>(l,c,view,light,mask);
    return l.coat_albedo ? layered(l,base,coat) : compose_coat(l.throughput*base,coat,c.coat_weight,
        dot(c.geometry_coat_normal,view),dot(c.geometry_coat_normal,light),c.coat_ior);
}
