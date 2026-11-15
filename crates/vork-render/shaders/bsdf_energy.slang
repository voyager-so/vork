// View-dependent scattering energy belongs to the closure, independently of
// the integrator or the light proposal that consumes it.
import vork_constants;
import openpbr_core;
import dielectric_interface;
import closure;
import bsdf_microfacet;
import bsdf_film;
float3 surface_directional_albedo(Closure c, float3 view, bool sheet=false, bool unit_fresnel=false) {
    var dielectric=c;dielectric.base_metalness=0.;
    let frame_=microfacet_frame(c);let a=microfacet_alpha(c);
    let v=mul(view,frame_);let no=max(v.z,.0001);
    let stretched=normalize(float3(a*v.xy,no));
    let t=dot(stretched.xy,stretched.xy)>0. ? normalize(float3(-stretched.y,stretched.x,0.)) : float3(1.,0.,0.);
    let b=cross(stretched,t);
    let g1=2.*no/(no+length(float3(a*v.xy,no)));
    var result=float3(0.);
    // Retain the complete ordered quadrature without expanding its Fresnel/film
    // graph once per sample in native device compilation.
    [loop] for (uint i=0u;i<REG_SURFACE_ALBEDO_SAMPLES;i++) {
        let u=(float(i)+.5)/float(REG_SURFACE_ALBEDO_SAMPLES);
        let phi=REG_TAU*float(reversebits(i)>>8u)*(1./16777216.);
        let x=sqrt(u)*cos(phi);var y=sqrt(u)*sin(phi);
        y=lerp(sqrt(max(1.-x*x,0.)),y,.5*(1.+stretched.z));
        let nh=x*t+y*b+sqrt(max(1.-x*x-y*y,0.))*stretched;
        let h=normalize(float3(a*nh.xy,max(nh.z,0.)));
        let light=reflect(-v,h);
        if (light.z>0.) {
            let world_light=mul(light,transpose(frame_));
            let cosine=max(dot(v,h),0.);
            let f=c.specular_color*(unit_fresnel ? 1. : fresnel(cosine,dielectric_ior(c)));
            var reflected=unit_fresnel ? f : surface_reflection(dielectric,cosine);
            if (sheet) {
                var interface_f=float3(fresnel(cosine,dielectric_ior(c)));
                if (c.thin_film_weight>0. && c.thin_film_thickness>0.) {interface_f=lerp(interface_f,film_dielectric(cosine,dielectric_ior(c),c.thin_film_ior,c.thin_film_thickness),c.thin_film_weight);}
                reflected=c.specular_color*(2.*interface_f/(1.+interface_f));
            }
            result+=reflected*(4.*no*light.z*microfacet_visibility(c,view,world_light))/g1;
        }
    }
    return clamp(result/float(REG_SURFACE_ALBEDO_SAMPLES),float3(0.),float3(1.));
}

// Scaled-primary-lobe multiscattering: sum the repeated Fresnel/escape
// geometric series (Kulla & Conty 2017), as in Cycles Principled GGX.
float3 transport_metal_fresnel_average(Closure c) {
    let f0=c.base_weight*c.base_color;let b=metal_edge_correction(f0,c.specular_color);
    if (any(b>float3(0.))) {
        // F82 clamps each angular value. Integrating the signed polynomial and
        // then clamping its mean loses positive grazing reflection in dark tints.
        var average=float3(0.);
        for (uint i=0u;i<REG_BRDF_ALBEDO_SIZE;i++) {
            let mu=(float(i)+.5)/float(REG_BRDF_ALBEDO_SIZE);
            average+=2.*mu*reflection_fresnel(c.base_color,c.base_weight,1.,1.,c.specular_color,mu,c.specular_ior)/float(REG_BRDF_ALBEDO_SIZE);
        }
        return clamp(average,float3(0.),float3(1.));
    }
    // 2∫ F82(mu)*mu dmu: Schlick and tint moments are 1/21 and 1/126.
    return clamp(f0+(1.-f0)/21.-b/126.,float3(0.),float3(1.));
}
float3 transport_metal_compensation(Closure c,float3 view,float2 moments,float average_energy) {
    if (c.base_metalness<=0. || c.specular_roughness==0.) {return float3(0.);}
    var energy=0.;
    if (c.specular_roughness_anisotropy==0.) {
        energy=moments.x+moments.y;
    } else {
        var unit=c;unit.base_metalness=0.;unit.specular_weight=1.;unit.specular_color=float3(1.);unit.thin_film_weight=0.;
        energy=surface_directional_albedo(unit,view,false,true).x;
    }
    let missing=1./max(energy,1e-6)-1.;
    // Film-covered GGX uses the underlying F82 average.
    let average_fresnel=transport_metal_fresnel_average(c);
    return missing*average_fresnel*average_energy/max(1.-average_fresnel*(1.-average_energy),float3(1e-6));
}
// Compensate only the metallic component before outer-layer composition.
float3 compensated_surface_reflection(Closure c,float3 view,float cosine,float3 compensation) {
    var f=surface_reflection(c,cosine);
    if(any(compensation>float3(0.))) {
        if(c.base_metalness==1.) {f+=compensation*f;}
        else {
            var metal=c;metal.base_metalness=1.;
            let conductor=reflection_fresnel(c.base_color,c.base_weight,1.,c.specular_weight,c.specular_color,cosine,c.specular_ior);
            f+=c.base_metalness*compensation*film_reflection(metal,cosine,conductor);
        }
    }
    return f;
}
