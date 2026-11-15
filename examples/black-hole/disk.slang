// Emissive, partially transmitting annulus: real scene triangles.
struct Inputs {
    float energy=3.;float temperature_scale=13500.;float outer_radius=24.;float gravitational_time=.02;
};
uint hash(uint x) {x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
float random(uint x) {return float(hash(x)&0xffffffu)/16777216.;}
// Planck B_lambda integrated against analytic CIE 1931 matching functions.
// Relative radiometry: B_lambda(550 nm, 6000 K)=1. No RGB colour ramp.
float gaussian(float wavelength,float centre,float left,float right) {
    let x=(wavelength-centre)*(wavelength<centre ? left : right);
    return exp(-.5*x*x);
}
float3 blackbody(float temperature) {
    var xyz=float3(0.);
    for(uint i=0u;i<21u;i++) {
        let w=380.+20.*float(i);
        let x=1.056*gaussian(w,599.8,.0264,.0323)+.362*gaussian(w,442.,.0624,.0374)-.065*gaussian(w,501.1,.049,.0382);
        let y=.821*gaussian(w,568.8,.0213,.0247)+.286*gaussian(w,530.9,.0613,.0322);
        let z=1.217*gaussian(w,437.,.0845,.0278)+.681*gaussian(w,459.,.0385,.0725);
        let planck=pow(550./w,5.)*(exp(14387769./(550.*6000.))-1.)/
            (exp(min(14387769./(w*max(temperature,100.)),80.))-1.);
        xyz+=float3(x,y,z)*planck*(20./106.8569);
    }
    return max(float3(dot(float3(3.2406,-1.5372,-.4986),xyz),
        dot(float3(-.9689,1.8758,.0415),xyz),dot(float3(.0557,-.204,1.057),xyz)),0.);
}
float disk_flux(float r,float outer) {
    if(r<=6. || r>=outer) return 0.;
    return pow(6./r,3.)*(1.-sqrt(6./r))*(1.-smoothstep(outer-2.,outer,r));
}
float noise(float3 p) {
    let cell=int3(floor(p));let f=frac(p);let w=f*f*(3.-2.*f);
    var value=0.;
    for(int k=0;k<2;k++) for(int j=0;j<2;j++) for(int i=0;i<2;i++) {
        let c=cell+int3(i,j,k);
        let id=hash(uint(c.x))^hash(uint(c.y)+0x1234567u)^hash(uint(c.z)+0x7654321u);
        let weight=lerp(1.-w.x,w.x,float(i))*lerp(1.-w.y,w.y,float(j))*lerp(1.-w.z,w.z,float(k));
        value+=weight*random(id);
    }
    return value;
}
float orbital_phase(float r,float azimuth,float time,float gravitational_time) {
    return azimuth-time/(gravitational_time*pow(r,1.5));
}
float turbulent_field(float r,float phase,float seed) {
    let p=float3(3.*cos(phase),3.*sin(phase),2.*log(r/6.))+seed*float3(71.3,17.7,31.2);
    // Anisotropic density cascades: narrow radial filaments with orbital shear,
    // rather than isotropic low-frequency painted eddies.
    let fine=float3(14.*cos(phase),14.*sin(phase),8.*log(r/6.))+seed*31.2;
    return .30*noise(p)+.25*noise(fine)+.25*noise(fine*2.13)+.20*noise(fine*4.61);
}
float structure(float r,float azimuth,float time,float gravitational_time) {
    // Finite-lived eddies renew smoothly; without renewal differential rotation
    // winds a frozen initial field into arbitrarily fine concentric stripes.
    // Each of the two overlapping fields still advects at exact local Omega.
    let lifetime=100.*gravitational_time;
    let epoch=floor(time/lifetime);let age=time-epoch*lifetime;
    let old=turbulent_field(r,orbital_phase(r,azimuth,age+lifetime,gravitational_time),epoch-1.);
    let fresh=turbulent_field(r,orbital_phase(r,azimuth,age,gravitational_time),epoch);
    return exp(2.8*(lerp(old,fresh,smoothstep(0.,lifetime,age))-.5));
}

float3 disk_radiance(float3 point,float velocity_cosine,float time,Inputs inputs) {
    let rho=length(point);let r=rho*pow(1.+.5/rho,2.);
    let flux=disk_flux(r,inputs.outer_radius);if(flux<=0.) return float3(0.);
    let density=structure(r,atan2(point.z,point.x),time,inputs.gravitational_time);
    let speed=rsqrt(r-2.);
    let doppler=sqrt(1.-speed*speed)/(1.-speed*velocity_cosine);
    let temperature=inputs.temperature_scale*pow(flux*density,.25);
    // Thermal Doppler transfer is evaluated spectrally before RGB conversion.
    // Gravity's bolometric transfer belongs to the propagation field, once.
    return inputs.energy*blackbody(temperature*doppler);
}
Closure surface(ShadingContext ctx,Inputs inputs) {
    // Radius and turbulence belong to the annulus; physical angles use the world metric.
    let point=world_to_object(ctx,ctx.POSITION);
    let velocity=object_direction_to_world(ctx,float3(-point.z,0.,point.x));
    let normal=object_normal_to_world(ctx,float3(0.,1.,0.));
    let rho=length(point);let r=rho*pow(1.+.5/rho,2.);
    let density=structure(r,atan2(point.z,point.x),ctx.time,inputs.gravitational_time);
    let tau=12.*pow(6./r,3.)*density*(1.-smoothstep(inputs.outer_radius-2.,inputs.outer_radius,r));
    Closure c=Closure();c.base_weight=0.;c.specular_weight=0.;c.geometry_thin_walled=true;
    c.geometry_opacity=1.-exp(-tau/max(abs(dot(ctx.incoming,normal)),.03));
    c.emission_color=disk_radiance(point,dot(ctx.incoming,velocity),ctx.time,inputs);c.emission_luminance=1.;
    return c;
}
