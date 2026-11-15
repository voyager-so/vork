// Shared isotropic GGX response terms for raster and std/oracle integrators.
import vork_openpbr_constants;

// All surface evaluators floor GGX alpha after squaring perceptual roughness.
public float specular_alpha(float roughness) {
    return max(roughness*roughness,MIN_ALPHA);
}
// LTC, prefiltered environments and packed ray controls use perceptual roughness.
public float specular_roughness(float roughness) {
    return sqrt(specular_alpha(roughness));
}
// OpenPBR 1.1.1, modulated_ior: weight scales normal reflectivity by
// changing the signed IOR contrast, including interfaces viewed from below.
public float modulated_ior(float eta,float weight) {
    if(weight==1. || eta==1.) return eta;
    let contrast=(eta-1.)/(eta+1.);
    // The next float below one keeps the reconstructed IOR finite at the
    // unit-reflectivity endpoint, within the precision of float Fresnel.
    let limit=asfloat(0x3f7fffffu);
    let epsilon=clamp(sqrt(max(weight,0.))*contrast,-limit,limit);
    return (1.+epsilon)/(1.-epsilon);
}
public float fresnel(float c, float eta) {
    c=clamp(c,0.,1.);
    if (eta==1.) {return 0.;}
    let si2=(1.-c*c)/(eta*eta);if (si2>=1.) {return 1.;}
    let ct=sqrt(max(0.,1.-si2));let rs=(c-eta*ct)/(c+eta*ct);let rp=(eta*c-ct)/(eta*c+ct);return (rs*rs+rp*rp)*.5;
}
// OpenPBR metal edge tint (F82). The same coefficient is also consumed by
// raster's integrated Fresnel moments; white tint has no correction.
public float3 metal_edge_correction(float3 f0, float3 tint) {
    if (all(tint==float3(1.))) {return float3(0.);}
    let mu=1./7.;let edge=f0+(float3(1.)-f0)*pow(1.-mu,5.);
    return edge*(float3(1.)-tint)/(mu*pow(1.-mu,6.));
}
public float3 reflection_fresnel_at_ior(float3 base_color, float base_weight, float metalness,
    float specular_weight, float3 specular_color, float cosine, float eta) {
    // Unit-vector dot products can round outside the physical cosine domain.
    cosine=clamp(cosine,0.,1.);
    let dielectric=float3(fresnel(cosine,eta))*specular_color;
    if (metalness<=0.) {return dielectric;}
    let f0=base_weight*base_color;let f=pow(1.-cosine,5.);
    let metal=max(f0+(float3(1.)-f0)*f-metal_edge_correction(f0,specular_color)*cosine*pow(1.-cosine,6.),float3(0.));
    return lerp(dielectric,specular_weight*metal,metalness);
}
public float3 reflection_fresnel(float3 base_color, float base_weight, float metalness,
    float specular_weight, float3 specular_color, float cosine, float eta) {
    return reflection_fresnel_at_ior(base_color,base_weight,metalness,specular_weight,specular_color,cosine,modulated_ior(eta,specular_weight));
}
// Albedo scaling reserves the integrated reflection, not Fresnel at an
// individual half-vector. This is the nonreciprocal OpenPBR layering rule.
public float3 diffuse_brdf(float3 base_color, float base_weight, float metalness, float transmission, float3 directional_reflectance) {
    return (base_weight*base_color)*(1.-metalness)*(1.-transmission)
        *(float3(1.)-directional_reflectance)/PI;
}
// OpenPBR thin-wall SSS: unit diffuse hemispheres split by anisotropy.
// Radius is absent because the sheet has no interior.
public float3 substrate_albedo(float3 base, float3 subsurface, float weight, float anisotropy, bool thin, bool transmitted=false) {
    if (transmitted) {return thin ? subsurface*weight*.5*(1.+anisotropy) : float3(0.);}
    return lerp(base,subsurface*(thin ? .5*(1.-anisotropy) : 1.),weight);
}
public float coat_transmission(float weight, float cosine, float eta) {
    return 1.-weight*fresnel(abs(cosine),eta);
}
public float3 compose_coat(float3 base, float3 coat, float weight, float no, float ni, float eta) {
    return base*(coat_transmission(weight,no,eta)*coat_transmission(weight,ni,eta))+coat;
}
public float distribution(float nh, float alpha) {
    // Keep the transport form: subtracting one from alpha squared loses the
    // minimum-alpha lobe at nh=1 in float32.
    let x=(1.-nh*nh)+nh*nh*alpha*alpha;return alpha*alpha/(PI*x*x);
}
// Height-correlated Smith G2 / (4 No Ni). The raster denominator floor is
// applied before division so horizon directions remain finite without 0/0.
float smith_denominator(float no, float ni, float alpha) {
    let ci=abs(ni);let a2=alpha*alpha;
    let po=sqrt(a2+(1.-a2)*no*no);let pi=sqrt(a2+(1.-a2)*ci*ci);
    return ci*po+no*pi;
}
public float smith_visibility(float no, float ni, float alpha, float denominator_floor) {
    return .5/max(smith_denominator(no,ni,alpha),denominator_floor);
}
public float g2(float no, float ni, float alpha) {
    return 2.*no*abs(ni)/smith_denominator(no,ni,alpha);
}

// Heitz 2018 GGX visible-normal sampling in a local orthonormal frame.
// Stretch/project/unstretch retains narrow lobes without subtracting alpha²
// from one, and supports isotropic and anisotropic callers alike.
public float3 ggx_visible_normal(float3 view,float2 alpha,float2 u) {
    let v=normalize(float3(alpha*view.xy,view.z));
    let length_squared=dot(v.xy,v.xy);
    let t=length_squared>0. ? float3(-v.y,v.x,0.)/sqrt(length_squared) : float3(1.,0.,0.);
    let b=cross(v,t);let radius=sqrt(u.x);let phi=2.*PI*u.y;
    let x=radius*cos(phi);
    let y=lerp(sqrt(max(1.-x*x,0.)),radius*sin(phi),.5*(1.+v.z));
    let projected=x*t+y*b+sqrt(max(1.-x*x-y*y,0.))*v;
    return normalize(float3(alpha*projected.xy,max(projected.z,0.)));
}
// G2/G1 with the grazing cosine cancelled. Positive no/ni make the
// denominator positive; an absolute visibility floor would change energy.
public float ggx_visible_weight(float no,float ni,float alpha) {
    if (ni<=0.) {return 0.;}
    let a2=alpha*alpha;
    return ni*(no+sqrt(a2+(1.-a2)*no*no))/smith_denominator(no,ni,alpha);
}
// One bounded unit-Fresnel reflection sample, with its Fresnel cosine.
public float ggx_visible_reflection(float no,float alpha,float2 u,out float vh) {
    let view=float3(sqrt(max(1.-no*no,0.)),0.,no);
    let h=ggx_visible_normal(view,float2(alpha),u);let light=reflect(-view,h);
    vh=clamp(dot(view,h),0.,1.);
    return ggx_visible_weight(no,light.z,alpha);
}
// Bounded visible-normal quadrature of exact-Fresnel directional albedo.
public float dielectric_directional_albedo(float no, float roughness, float eta, uint samples) {
    if (eta==1.) {return 0.;}
    if (roughness==0.) {return select(eta<=0.,1.,fresnel(no,eta));}
    let a=specular_alpha(roughness);
    var result=0.;
    for (uint i=0u;i<samples;i++) {
        let u=float2((float(i)+.5)/float(samples),float(reversebits(i)>>8u)*UINT_SCALE);
        float vh;let weight=ggx_visible_reflection(no,a,u,vh);
        result+=select(eta<=0.,1.,fresnel(vh,eta))*weight;
    }
    return result/float(samples);
}

// BRDF values (without the incident cosine); raster and transport share these
// terms and apply their own lobe visibility masks and sampling PDFs afterwards.
public float3 ggx_reflection(float3 reflectance, float no, float ni, float nh, float roughness) {
    let a=specular_alpha(roughness);
    return reflectance*distribution(nh,a)*g2(no,ni,a)/(4.*no*ni);
}
public float3 coat_brdf(float weight, float roughness, float eta, float no, float ni, float nh, float oh) {
    return ggx_reflection(float3(weight*fresnel(oh,eta)),no,ni,nh,roughness);
}

// Walter et al. 2007 rough dielectric transmission. eta = transmitted / incident
// IOR; directions point away from the interface. Returns radiance-mode BTDF
// without the incident cosine, and the NDF-sampled solid-angle PDF (no Fresnel
// branch probability). The eta² Jacobian cancels the radiance eta⁻² factor.
public float ggx_transmission(float no, float ni, float nh, float oh, float ih,
    float roughness, float eta, out float pdf) {
    pdf=0.;
    if (no<=0. || ni>=0. || nh<=0. || oh<=0. || ih>=0.) {return 0.;}
    let denominator=oh+eta*ih;
    if (denominator==0.) {return 0.;}
    let a=specular_alpha(roughness);let d=distribution(nh,a);
    pdf=d*nh*abs(eta*eta*ih/(denominator*denominator));
    return (1.-fresnel(oh,eta))*d*g2(no,ni,a)*abs(oh*ih/(no*ni*denominator*denominator));
}
// Cycles Principled splits a solid's depth-zero colour between its two
// refracting interfaces. A zero-thickness sheet carries the full authored tint;
// a positive-depth solid instead absorbs over its interior segments.
public float3 transmission_interface_tint(float3 color, float depth, bool thin) {
    if (thin) {return color;}
    return depth<=0. ? sqrt(max(color,float3(0.))) : float3(1.);
}
// OpenPBR depth zero denotes interface tint, positive depth an interior medium.
public float3 transmission_attenuation(float3 color, float depth, float distance) {
    if (distance<=0.) {return float3(1.);}
    if (depth<=0.) {return color;}
    // Preserve exact black/clear channels and every positive authored depth.
    // A safe positive base avoids pow(0,x); clear channels also avoid 1^inf.
    let safe=select(color>float3(0.),color,float3(1.));
    // Keep pow's exponent finite when the optical distance overflows FP32.
    let optical_distance=min(distance/depth,3.402823466e38);
    let attenuation=pow(safe,float3(optical_distance));
    return select(color<=float3(0.),float3(0.),select(color==float3(1.),float3(1.),attenuation));
}

// Sum the coincident front/back interface series for a zero-thickness sheet.
public float sheet_reflectance(float f) {return 2.*f/(1.+f);}
public float ggx_sheet_transmission(float no, float ni, float nh, float oh,
    float roughness, float eta, out float pdf) {
    pdf=0.;
    if (no<=0. || ni>=0. || nh<=0. || oh<=0.) {return 0.;}
    let a=specular_alpha(roughness);let d=distribution(nh,a);
    pdf=d*nh/(4.*oh);
    return (1.-sheet_reflectance(fresnel(oh,eta)))*d*g2(no,ni,a)/(4.*no*abs(ni));
}
