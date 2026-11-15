// Analytic CIE 1931 matching-function fit (Wyman, Sloan, Shirley 2013).
// Integrate Planck's law for blackbody; wavelength takes nanometres.
float color_gaussian(float wavelength,float center,float left,float right) {
    float t=(wavelength-center)*(wavelength<center?left:right);return exp(-0.5*t*t);
}
float3 color_xyz(float nm) {
    return float3(1.056*color_gaussian(nm,599.8,0.0264,0.0323)+0.362*color_gaussian(nm,442.,0.0624,0.0374)-0.065*color_gaussian(nm,501.1,0.049,0.0382),
        0.821*color_gaussian(nm,568.8,0.0213,0.0247)+0.286*color_gaussian(nm,530.9,0.0613,0.0322),
        1.217*color_gaussian(nm,437.,0.0845,0.0278)+0.681*color_gaussian(nm,459.,0.0385,0.0725));
}
float3 color_xyz_rgb(float3 xyz) {
    return float3(dot(xyz,float3(3.2406,-1.5372,-0.4986)),dot(xyz,float3(-0.9689,1.8758,0.0415)),dot(xyz,float3(0.0557,-0.204,1.057)));
}
public float3 wavelength(float nm=500.) {
    if(nm<380. || nm>780.) return float3(0.);
    return max(color_xyz_rgb(color_xyz(nm))*0.4,0.);
}
public float3 blackbody(float temperature=1500.) {
    float3 xyz=0.; float t=max(temperature,1.);
    for(int i=0;i<81;i++) {
        float nm=380.+5.*i, micrometers=nm*0.001;
        float power=1./(pow(micrometers,5.)*(exp(min(14387.76877/(micrometers*t),80.))-1.));
        xyz+=color_xyz(nm)*power*((i==0 || i==80)?0.5:1.);
    }
    return xyz.y>0.?max(color_xyz_rgb(xyz/xyz.y),0.):float3(0.);
}

// Scene-linear colour utilities, independently implemented from Blender's manual
// and standard RGB/HSV/HSL definitions. No Blender source is incorporated.
public float3 color_from_hsv(float3 hsv) {
    float3 rgb=saturate(abs(frac(hsv.x+float3(0.,2./3.,1./3.))*6.-3.)-1.);
    return hsv.z*lerp(float3(1.),rgb,hsv.y);
}
public float3 color_to_hsv(float3 rgb) {
    float hi=max(rgb.x,max(rgb.y,rgb.z)), lo=min(rgb.x,min(rgb.y,rgb.z)), delta=hi-lo;
    float h=0.;
    if(delta>0.) {
        if(hi==rgb.x) h=(rgb.y-rgb.z)/delta;
        else if(hi==rgb.y) h=2.+(rgb.z-rgb.x)/delta;
        else h=4.+(rgb.x-rgb.y)/delta;
        h=frac(h/6.);
    }
    return float3(h,hi==0.?0.:delta/hi,hi);
}
float3 color_to_hsl(float3 rgb) {
    float3 hsv=color_to_hsv(rgb);
    float lo=min(rgb.x,min(rgb.y,rgb.z)), l=(hsv.z+lo)*0.5;
    float denominator=1.-abs(2.*l-1.);
    return float3(hsv.x,denominator==0.?0.:(hsv.z-lo)/denominator,l);
}
float3 color_from_hsl(float3 hsl) {
    float c=(1.-abs(2.*hsl.z-1.))*hsl.y;
    return color_from_hsv(float3(hsl.x,1.,1.))*c+(hsl.z-c*0.5);
}
// Library rule: compare enums as integers, `int(value) == int(Enum.Member)` (or `!=`).
// Slang's enum `==` costs front-end time in every program that loads the module, and
// the integer form emits the same code. Enum fields, arguments, defaults and switches stay enums.
public enum RampInterpolation { Constant, Linear, Ease, Cardinal, B_Spline };
public enum RampColorMode { RGB, HSV, HSL };
public enum HueInterpolation { Near, Far, CW, CCW };
public struct RampStop { public float position; public float4 color; };
float ramp_hue(float a,float b,HueInterpolation mode) {
    float d=b-a;
    if(int(mode) == int(HueInterpolation.Near)) { if(d>0.5) b-=1.; if(d< -0.5) b+=1.; }
    if(int(mode) == int(HueInterpolation.Far)) { if(d>0. && d<0.5) b-=1.; if(d<=0. && d> -0.5) b+=1.; }
    if(int(mode) == int(HueInterpolation.CW) && d>0.) b-=1.;
    if(int(mode) == int(HueInterpolation.CCW) && d<0.) b+=1.;
    return b;
}
// Stops are ordered by position. Empty ramps return transparent black.
public float4 color_ramp<let N:int>(float factor,RampStop stops[N],
    RampInterpolation interpolation=RampInterpolation.Linear,RampColorMode mode=RampColorMode.RGB,
    HueInterpolation hue=HueInterpolation.Near) {
    if(N==0) return float4(0.);
    if(N==1 || factor<stops[0].position) return stops[0].color;
    int i=0;
    for(int j=1;j<N;j++) if(factor>=stops[j].position) i=j;
    if(i==N-1 || int(interpolation) == int(RampInterpolation.Constant)) return stops[i].color;
    float width=stops[i+1].position-stops[i].position;
    float t=width>0.?(factor-stops[i].position)/width:0.;
    if(int(interpolation) == int(RampInterpolation.Ease)) t=t*t*(3.-2.*t);
    float4 a=stops[i].color,b=stops[i+1].color;
    if(int(mode) != int(RampColorMode.RGB)) {
        a.xyz=int(mode) == int(RampColorMode.HSV)?color_to_hsv(a.xyz):color_to_hsl(a.xyz);
        b.xyz=int(mode) == int(RampColorMode.HSV)?color_to_hsv(b.xyz):color_to_hsl(b.xyz);
        b.x=ramp_hue(a.x,b.x,hue);
        float4 result=lerp(a,b,t);
        result.xyz=int(mode) == int(RampColorMode.HSV)?color_from_hsv(result.xyz):color_from_hsl(result.xyz);
        return result;
    }
    if(int(interpolation) == int(RampInterpolation.Cardinal) || int(interpolation) == int(RampInterpolation.B_Spline)) {
        float4 p=stops[max(i-1,0)].color,q=stops[min(i+2,N-1)].color;
        float t2=t*t,t3=t2*t;
        if(int(interpolation) == int(RampInterpolation.B_Spline))
            return ((1.-3.*t+3.*t2-t3)*p+(4.-6.*t2+3.*t3)*a+(1.+3.*t+3.*t2-3.*t3)*b+t3*q)/6.;
        // Cardinal tension 0.71, as specified by the Blender ramp.
        float4 m0=0.71*(b-p),m1=0.71*(q-a);
        return (2.*t3-3.*t2+1.)*a+(t3-2.*t2+t)*m0+(-2.*t3+3.*t2)*b+(t3-t2)*m1;
    }
    return lerp(a,b,t);
}

// Blender Mix Color blend modes and Float Curve, independently implemented from
// Blender's manual. No Blender source is incorporated.
public enum BlendMode { Mix, Darken, Multiply, Burn, Lighten, Screen, Dodge, Add,
    Overlay, Soft_Light, Linear_Light, Difference, Exclusion, Subtract, Divide, Hue, Saturation, Color, Value };
public float3 mix(float3 a,float3 b,float factor=0.5,BlendMode mode=BlendMode.Mix,
                  bool clamp_factor=true,bool clamp_result=false) {
    float f=clamp_factor?saturate(factor):factor;
    if(int(mode) == int(BlendMode.Burn) || int(mode) == int(BlendMode.Dodge)) {
        float3 result;
        for(int i=0;i<3;i++) {
            float denominator=int(mode) == int(BlendMode.Burn)?1.-f+f*b[i]:1.-f*b[i];
            result[i]=int(mode) == int(BlendMode.Burn)?(denominator<=0.?0.:1.-(1.-a[i])/denominator):
                (denominator<=0.?1.:a[i]/denominator);
        }
        return saturate(result);
    }
    float3 c=b;
    switch(mode) {
        case BlendMode.Darken: c=min(a,b); break;
        case BlendMode.Multiply: c=a*b; break;
        case BlendMode.Lighten: c=max(a,b); break;
        case BlendMode.Screen: c=1.-(1.-a)*(1.-b); break;
        case BlendMode.Add: c=a+b; break;
        case BlendMode.Overlay:
            for(int i=0;i<3;i++) c[i]=a[i]<0.5?2.*a[i]*b[i]:1.-2.*(1.-a[i])*(1.-b[i]); break;
        case BlendMode.Soft_Light: c=(1.-2.*b)*a*a+2.*b*a; break;
        case BlendMode.Linear_Light: c=a+2.*b-1.; break;
        case BlendMode.Difference: c=abs(a-b); break;
        case BlendMode.Exclusion: c=a+b-2.*a*b; break;
        case BlendMode.Subtract: c=a-b; break;
        case BlendMode.Divide: for(int i=0;i<3;i++) c[i]=b[i]==0.?a[i]:a[i]/b[i]; break;
        case BlendMode.Hue: case BlendMode.Saturation: case BlendMode.Color: case BlendMode.Value: {
            float3 ah=color_to_hsv(a),bh=color_to_hsv(b);
            if(int(mode) == int(BlendMode.Hue) && bh.y!=0.) ah.x=bh.x;
            if(int(mode) == int(BlendMode.Saturation)) ah.y=bh.y;
            if(int(mode) == int(BlendMode.Color)) { ah.x=bh.x; ah.y=bh.y; }
            if(int(mode) == int(BlendMode.Value)) ah.z=bh.z;
            c=color_from_hsv(ah); break;
        }
        default: break;
    }
    c=lerp(a,c,f);
    return clamp_result?saturate(c):c;
}
// A curve is a sorted sequence of knots with explicit cubic Bezier handles.
// Handles permit authored automatic, aligned and vector handle modes without
// making a shader solve authoring constraints for every shaded sample.
public struct CurvePoint { public float2 position; public float2 left; public float2 right; };
float2 curve_bezier(float2 a,float2 b,float2 c,float2 d,float t) {
    float u=1.-t;return u*u*u*a+3.*u*u*t*b+3.*u*t*t*c+t*t*t*d;
}
public float float_curve<let N:int>(float value,CurvePoint points[N],float factor=1.,bool extrapolate=true) {
    if(N==0) return value;
    if(N==1) return lerp(value,points[0].position.y,factor);
    float result=0.;
    if(value<points[0].position.x || value>points[N-1].position.x) {
        int end=value<points[0].position.x?0:N-1;
        float2 handle=end==0?points[end].left:points[end].right;
        float2 delta=handle-points[end].position;
        result=points[end].position.y;
        if(extrapolate && delta.x!=0.) result+=(value-points[end].position.x)*delta.y/delta.x;
    } else {
        int i=0;for(int j=1;j<N-1;j++) if(value>=points[j].position.x) i=j;
        float2 a=points[i].position,d=points[i+1].position;
        float2 b=points[i].right,c=points[i+1].left;
        // Monotone X handles; bisection avoids unstable cubic roots.
        b.x=clamp(b.x,a.x,d.x);c.x=clamp(c.x,a.x,d.x);
        float lo=0.,hi=1.;
        for(int iteration=0;iteration<24;iteration++) {
            float t=(lo+hi)*0.5;
            if(curve_bezier(a,b,c,d,t).x<value) lo=t;else hi=t;
        }
        result=curve_bezier(a,b,c,d,(lo+hi)*0.5).y;
    }
    return lerp(value,result,factor);
}
