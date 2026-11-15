// Material reflection state retained from camera shading. No lighting resources:
// screen traversal and hardware hits consume the same BSDF/PDF estimator.
import vork_constants;
import bsdf_microfacet;
import closure;
import openpbr_core;
import bsdf_film;
import bsdf_energy;
// NDF proposal shared by finite-emitter and environment quadrature. Its
// cancelled D term keeps narrow lobes finite without missing a sharp highlight.
float3 specular_quadrature_sample(Closure c, float3 view, float2 u, out float3 direction,float3 compensation) {
    let h=microfacet_sample(c,u);let vh=dot(view,h);let nh=dot(c.geometry_normal,h);
    direction=reflect(-view,h);let nl=dot(c.geometry_normal,direction);
    if (vh<=0. || nl<=0. || nh<=0.) {return float3(0.);}

    return compensated_surface_reflection(c,view,vh,compensation)*microfacet_visibility(c,view,direction)*nl*(4.*vh/nh);
}

struct ReflectionBsdf { float4 data[REG_REFLECTION_BSDF_VECTORS]; };
ReflectionBsdf reflection_bsdf(Closure base,Closure outer,float3 scale,float throughput,float visibility,float3 compensation) {
    ReflectionBsdf r;
    r.data[0]=float4(base.base_color*base.base_weight,0.);
    r.data[1]=float4(base.specular_color,base.specular_weight);
    r.data[2]=float4(base.geometry_normal,base.specular_roughness);
    r.data[3]=float4(base.geometry_tangent,base.specular_roughness_anisotropy);
    r.data[4]=float4(base.base_metalness,base.specular_ior,base.thin_film_weight,base.thin_film_thickness);
    r.data[5]=float4(scale*throughput*visibility,base.thin_film_ior);
    r.data[6]=float4(outer.geometry_coat_normal,outer.coat_roughness);
    r.data[7]=float4(outer.coat_weight,outer.coat_ior,throughput*visibility,0.);
    r.data[10]=float4(compensation,0.);
    r.data[11]=float4(outer.geometry_coat_tangent,outer.coat_roughness_anisotropy);
#ifdef VORK_UNIT_IOR
    let optical=base.specular_ior;
#else
    let optical=base.optical_ior;
#endif
    r.data[9]=float4(base.transmission_weight,base.geometry_thin_walled ? 1. : 0.,optical,0.);
    r.data[8]=float4(1.,1.,1.,0.); // composition transmittance, independent of material visibility
    return r;
}
// Spare components retain the exact shaded environment response to replace.
float3 reflection_bsdf_environment(ReflectionBsdf r) {return float3(r.data[0].w,r.data[7].w,r.data[8].w);}
void reflection_bsdf_set_environment(inout ReflectionBsdf r,float3 value) {r.data[0].w=value.x;r.data[7].w=value.y;r.data[8].w=value.z;}
// One material decode serves both the proposal and the complete evaluator.
Closure reflection_bsdf_closure(ReflectionBsdf r,bool coat) {
    var c=(Closure)0;
    if(coat) {
        c.geometry_normal=normalize(r.data[6].xyz);c.specular_roughness=r.data[6].w;
        c.specular_ior=r.data[7].y;c.specular_color=float3(1.);c.specular_weight=1.;
        c.geometry_tangent=r.data[11].xyz;c.specular_roughness_anisotropy=r.data[11].w;
    } else {
        c.transmission_weight=r.data[9].x;c.geometry_thin_walled=r.data[9].y!=0.;
        c.base_color=r.data[0].xyz;c.base_weight=1.;
        c.specular_color=r.data[1].xyz;c.specular_weight=r.data[1].w;
        c.geometry_normal=normalize(r.data[2].xyz);c.specular_roughness=r.data[2].w;
        c.geometry_tangent=r.data[3].xyz;c.specular_roughness_anisotropy=r.data[3].w;
        c.base_metalness=r.data[4].x;c.specular_ior=r.data[4].y;
#ifdef VORK_UNIT_IOR
        // Shipped and runtime shaders share this prepared-optics record.
        if(r.data[9].z>0.) c.specular_ior=r.data[9].z;
#else
        c.optical_ior=r.data[9].z;
#endif
        c.thin_film_weight=r.data[4].z;c.thin_film_thickness=r.data[4].w;c.thin_film_ior=r.data[5].w;
    }
    return c;
}
// Sampling selects a direction only. Both proposals subsequently evaluate the
// full base/coat BSDF and its full mixture PDF, rather than one selected lobe.
float3 reflection_bsdf_sample(ReflectionBsdf r,float3 view,float2 u) {
    let coat=r.data[7].x>0.;let probability=coat ? .5 : 1.;
    let selected=coat && u.x>=probability;
    u.x=selected ? (u.x-probability)/(1.-probability) : u.x/probability;
    let c=reflection_bsdf_closure(r,selected);
    if(dot(c.geometry_normal,view)<=0.) {return float3(0.);}
    return reflect(-view,transport_visible_normal(c,view,u));
}

// Proposal support includes below-surface reflected directions. A lobe can
// have zero response there while another lobe or transmission still contributes.
float reflection_bsdf_pdf(ReflectionBsdf r,float3 view,float3 light) {
    let half_sum=view+light;
    if(dot(half_sum,half_sum)==0.) {return 0.;}
    let h=normalize(half_sum);let coat=r.data[7].x>0.;var pdf=0.;
    for(uint i=0u;i<(coat ? 2u : 1u);i++) {
        let c=reflection_bsdf_closure(r,i!=0u);
        if(dot(c.geometry_normal,view)<=0. || dot(c.geometry_normal,h)<=0.) {continue;}
        pdf+=microfacet_distribution(c,h)*transport_visible_scale(c,view)*.25*(coat ? .5 : 1.);
    }
    return pdf;
}

float3 reflection_bsdf_evaluate(ReflectionBsdf r,float3 view,float3 light,out float pdf) {
    var value=float3(0.);pdf=reflection_bsdf_pdf(r,view,light);
    let coat=r.data[7].x>0.;
    for(uint i=0u;i<(coat ? 2u : 1u);i++) {
        let c=reflection_bsdf_closure(r,i!=0u);
        var scale=i==0u ? r.data[5].xyz : float3(r.data[7].x*r.data[7].z);
        if(coat && i==0u) {scale*=coat_transmission(r.data[7].x,dot(normalize(r.data[6].xyz),light),r.data[7].y);}
        let nv=dot(c.geometry_normal,view);let nl=dot(c.geometry_normal,light);
        if(nv<=0. || nl<=0.) {continue;}
        let h=normalize(view+light);let vh=dot(view,h);
        let d=microfacet_distribution(c,h);

        value+=compensated_surface_reflection(c,view,vh,i==0u ? r.data[10].xyz : float3(0.))*d*microfacet_visibility(c,view,light)*nl*scale;
    }
    return value*r.data[8].xyz;
}
