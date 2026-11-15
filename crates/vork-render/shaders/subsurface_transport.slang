// Finite-layer two-flux (Kubelka–Munk) solution. This is exact for the
// homogeneous two-flux slab, not for arbitrary 3D/random-walk transport.
// R_infinity is the authored subsurface albedo. S = (1-g)/mean-free-path;
// K/S = (1-R_infinity)^2/(2 R_infinity). R+T <= 1, including white slabs.
// Kubelka, JOSA 38 (1948), doi:10.1364/JOSA.38.000448.
import vork_constants;
import closure;
import openpbr_core;
import dielectric_interface;
struct SubsurfaceSlab { float3 reflection; float3 transmission; };
float2 slab_channel(float albedo,float radius,float thickness) {
    let a=clamp(albedo,0.,1.);
    if (radius<=0.) {return float2(a,0.);}
    if (thickness<=0.) {return float2(0.,1.);}
    if (a<=0.) {return float2(0.);}
    let x=thickness/radius;
    if (a==1.) {let t=1./(1.+x);return float2(1.-t,t);}
    let difference=(1.-a)*(1.+a);
    let tau=difference/(2.*a)*x;
    // Divide the hyperbolic solution by exp(tau). Small optical depths
    // use expm1's Taylor series to avoid subtracting nearly equal floats.
    let q=exp(-tau);
    let lost=tau<REG_SUBSURFACE_SERIES_MAX ? 2.*tau*(1.-tau+2.*tau*tau/3.) : 1.-q*q;
    let denominator=difference+a*a*lost;
    return float2(a*lost,difference*q)/denominator;
}
SubsurfaceSlab subsurface_slab(Closure c,float thickness) {
    SubsurfaceSlab s={float3(0.),float3(0.)};
    let radius=c.subsurface_radius*c.subsurface_radius_scale/max(1.-c.subsurface_scatter_anisotropy,.001);
    for (uint i=0u;i<3u;i++) {
        let rt=slab_channel(c.subsurface_color[i],radius[i],thickness);
        s.reflection[i]=rt.x;s.transmission[i]=rt.y;
    }
    return s;
}
bool thickness_active(Closure c) {
    return c.subsurface_weight>0. && !c.geometry_thin_walled && c.base_metalness<1. && c.transmission_weight<1. && any(c.subsurface_radius*c.subsurface_radius_scale>float3(0.));
}
// Diffuse throughput shared by raster light integration and path next events.
float3 subsurface_transmission(Closure c,float3 normal,float3 view,float3 incident,float thickness) {
    let ni=max(-dot(normal,incident),0.);let nv=max(dot(normal,view),0.);
    let entry=1.-fresnel(ni,dielectric_ior(c));let exit=1.-fresnel(nv,dielectric_ior(c));
    return subsurface_slab(c,thickness).transmission*(entry*exit*c.base_weight*c.subsurface_weight*(1.-c.base_metalness)*(1.-c.transmission_weight));
}
// Query only outward crossings of the receiver's own mesh. Do not let another
// object become its thickness, or invent an exit for an open mesh. World rays
// retain metric distance under nonuniform scale and current deformed BLASes.
float subsurface_exit(RaytracingAccelerationStructure scene,uint object,float3 world,float3 direction) {
    RayDesc ray;ray.Origin=world;ray.Direction=direction;ray.TMin=0.;ray.TMax=REG_FP32_MAX;
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(scene,RAY_FLAG_NONE,0xffu,ray);
    while(query.Proceed()) {
        if(query.CandidateType()==CANDIDATE_NON_OPAQUE_TRIANGLE && query.CandidateInstanceID()==object
            && !query.CandidateTriangleFrontFace() && query.CandidateTriangleRayT()>0.) {query.CommitNonOpaqueTriangleHit();}
    }
    return query.CommittedStatus()==COMMITTED_TRIANGLE_HIT ? query.CommittedRayT() : -1.;
}
