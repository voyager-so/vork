// Optical boundary services shared by path, raster and hardware refraction.
// Geometry acquisition and accepted occupancy transitions belong to callers.
import closure;
import openpbr_core;
import vork_constants;
import vork_spectrum;
struct DielectricSegment { float3 color; float depth; float ior; float reference_ior; };
struct DielectricInterface { DielectricSegment incident; DielectricSegment transmitted; };
// Weighted closures retain separate reference and prepared optical ratios.
// Proven unit weights reuse the IOR slot for the prepared ratio, including dispersion.
float dielectric_ior(Closure c) {
#ifdef VORK_UNIT_IOR
    return c.specular_ior;
#else
    return c.optical_ior>0. ? c.optical_ior : modulated_ior(c.specular_ior,c.specular_weight);
#endif
}
float dielectric_bulk_ior(float reference,float weight,float surrounding) {
#ifdef VORK_UNIT_IOR
    return reference;
#else
    if(weight==1.) return reference;
    if(weight==0.) return surrounding;
    return surrounding*modulated_ior(reference/surrounding,weight);
#endif
}
// Modulate the d-line ratio before applying the material's own dispersion.
// Zero weight inherits the surrounding index at the sampled wavelength.
float dielectric_material_ior(Closure c,DielectricSegment incident,float wavelength) {
#ifndef VORK_UNIT_IOR
    if(c.specular_weight==0.) return incident.ior;
#endif
    let bulk=dielectric_bulk_ior(c.specular_ior,c.specular_weight,incident.reference_ior);
    return spectral_ior(bulk,wavelength,c.transmission_weight>0. ? c.transmission_dispersion_scale : 0.,c.transmission_dispersion_abbe_number);
}
// glTF base colour acts at every volume boundary, independently of interior
// attenuation. Native OpenPBR retains its thin/full and solid/sqrt tint rules.
float3 surface_transmission_tint(Closure c) {
#ifndef VORK_NO_GLTF_TINT
    if(c.gltf_material) return c.base_color;
#endif
    return transmission_interface_tint(c.transmission_color,c.transmission_depth,c.geometry_thin_walled);
}
// An escaping ray has no finite interior chord. Positive-depth extinction is
// integrated once per finite segment; depth-zero colour belongs to interfaces.
float3 segment_attenuation(DielectricSegment s,float distance) {
    return s.depth>0. && distance<REG_FP32_MAX ? transmission_attenuation(s.color,s.depth,distance) : float3(1.);
}
Closure dielectric_interface(Closure c,float incident_ior,float transmitted_ior,float optical_ratio) {
#ifdef VORK_UNIT_IOR
    c.specular_ior=optical_ratio;
#else
    c.specular_ior=transmitted_ior/incident_ior;
    c.optical_ior=optical_ratio;
#endif
    c.thin_film_ior/=incident_ior;c.thin_film_thickness*=incident_ior;
    return c;
}
Closure dielectric_interface(Closure c,float incident_ior,float transmitted_ior) {
    return dielectric_interface(c,incident_ior,transmitted_ior,dielectric_bulk_ior(transmitted_ior/incident_ior,c.specular_weight,1.));
}
Closure exiting_closure(Closure c,float3 normal) {
    let optical=dielectric_ior(c);c.geometry_normal=normal;
    return dielectric_interface(c,optical,1.,1./optical);
}
