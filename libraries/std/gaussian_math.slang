// Gaussian math shared by native ray traversal and ordinary material programs.
// KHR_gaussian_splatting attributes; Moenne-Loccoz et al. (2024), Sun et al. (2025).
public float3 gaussian_rotate(float4 q,float3 v) {
    let t=2.*cross(q.xyz,v);return v+q.w*t+cross(q.xyz,t);
}
public float3 gaussian_unrotate(float4 q,float3 v) {return gaussian_rotate(float4(-q.xyz,q.w),v);}
public struct GaussianHit {public bool found;public float t;public float opacity;};
public GaussianHit gaussian_hit(float3 position,float3 scale,float4 rotation,float opacity,
    float3 origin,float3 direction,float minimum,float maximum,float radius) {
    GaussianHit h={};
    let o=gaussian_unrotate(rotation,origin-position);
    let d=gaussian_unrotate(rotation,direction);
    // Transform the ray, not a covariance quadratic form. Normalizing the
    // scaled direction before its dot products avoids squaring its condition
    // number for very flat splats.
    let scaled=d/scale;
    let magnitude=max(abs(scaled.x),max(abs(scaled.y),abs(scaled.z)));
    let unit=scaled/magnitude;
    let length_squared=dot(unit,unit);
    h.t=-dot(o/(scale*magnitude),unit)/length_squared;
    if(!isfinite(h.t) || h.t<minimum || h.t>maximum) return h;
    // |(o/S) x (d/S)| / |d/S|, with cross products formed before
    // anisotropic scaling. Avoid subtracting two huge scaled vectors.
    let perpendicular=cross(o,d)/((scale.yzx*magnitude)*scale.zxy);
    let squared=dot(perpendicular,perpendicular)/length_squared;
    h.found=isfinite(squared) && squared<=radius*radius;
    h.opacity=h.found ? opacity*exp(-.5*squared) : 0.;
    return h;
}
public float3 gaussian_normal(float3 scale,float4 rotation) {
    let axis=scale.x<=scale.y && scale.x<=scale.z ? float3(1.,0.,0.)
        : (scale.y<=scale.z ? float3(0.,1.,0.) : float3(0.,0.,1.));
    return gaussian_rotate(rotation,axis);
}
// Real SH, Condon-Shortley phase, ascending degree then m. Coefficients
// remain in the capture frame. Direction points from the viewer to the splat.
public float3 gaussian_radiance(float3 coefficients[16],float3 direction) {
    let d=normalize(direction);let x=d.x;let y=d.y;let z=d.z;
    let xx=x*x;let yy=y*y;let zz=z*z;
    var c=coefficients[0]*0.2820947917738781+float3(.5);
    c+=-0.4886025119029199*y*coefficients[1]+0.4886025119029199*z*coefficients[2]-0.4886025119029199*x*coefficients[3];
    c+=1.092548430592079*x*y*coefficients[4]-1.092548430592079*y*z*coefficients[5]
        +0.3153915652525200*(2.*zz-xx-yy)*coefficients[6]-1.092548430592079*x*z*coefficients[7]
        +0.5462742152960395*(xx-yy)*coefficients[8];
    c+=-0.5900435899266435*y*(3.*xx-yy)*coefficients[9]+2.890611442640554*x*y*z*coefficients[10]
        -0.4570457994644657*y*(4.*zz-xx-yy)*coefficients[11]+0.3731763325901154*z*(2.*zz-3.*xx-3.*yy)*coefficients[12]
        -0.4570457994644657*x*(4.*zz-xx-yy)*coefficients[13]+1.445305721320277*z*(xx-yy)*coefficients[14]
        -0.5900435899266435*x*(xx-3.*yy)*coefficients[15];
    return max(c,float3(0.));
}
