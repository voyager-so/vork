// Blender socket helpers independently expressed over vork's public context.
// Callers pass evaluated source socket values; Blender owns authoring defaults.
import vork;
import std.openpbr_core;

// Diffuse BSDF: pinned EEVEE uses Lambert and ignores Roughness.
public Closure Diffuse_BSDF(float3 Color, float Roughness, float3 Normal) {
    Closure c={};c.base_color=Color;c.specular_weight=0.;c.geometry_normal=Normal;
    return c;
}
// Glass BSDF: EEVEE uses its GGX reflection/refraction response.
public Closure Glass_BSDF(float3 Color,float Roughness,float IOR,float3 Normal,
                          float Thin_Film_Thickness,float Thin_Film_IOR) {
    Closure c={};c.base_weight=0.;c.specular_color=Color;
    c.specular_roughness=Roughness;c.specular_ior=IOR;
    c.transmission_weight=1.;c.transmission_color=Color;c.geometry_normal=Normal;
    c.thin_film_weight=Thin_Film_Thickness>0. ? 1. : 0.;
    c.thin_film_thickness=Thin_Film_Thickness*.001;c.thin_film_ior=Thin_Film_IOR;
    return c;
}
public Closure Transparent_BSDF(float3 Color) {
    // An index-matched smooth sheet is a coloured null event: no reflection,
    // refraction, or volume interval; each crossing multiplies throughput by Color.
    Closure c={};c.base_weight=0.;c.specular_weight=0.;
    c.specular_roughness=0.;c.specular_ior=1.;
    c.transmission_weight=1.;c.transmission_color=Color;c.geometry_thin_walled=true;
    return c;
}
// Blender's Fresnel node uses geometric front/back orientation for the IOR ratio.
public float Fresnel(float IOR,float3 Normal,ShadingContext ctx) {
    float eta=max(IOR,1.e-5);
    return fresnel(abs(dot(normalize(Normal),normalize(ctx.incoming))),ctx.front_facing?eta:1./eta);
}
// Blender Math node manual and RNA enum, independently implemented:
// https://docs.blender.org/manual/en/5.2/modeling/geometry_nodes/utilities/math/math.html
// https://docs.blender.org/api/5.2/bpy_types_enum_items/node_math_items.html
public enum MathOperation { ADD, SUBTRACT, MULTIPLY, DIVIDE, MULTIPLY_ADD,
    MINIMUM, MAXIMUM, LESS_THAN, GREATER_THAN, ABSOLUTE, FLOOR, CEIL,
    POWER, FRACT, TRUNC, SINE, COSINE, TANGENT, ARCTANGENT, SQRT, SIGN,
    RADIANS, DEGREES, EXPONENT, ROUND, MODULO, FLOORED_MODULO, COMPARE,
    LOGARITHM, INVERSE_SQRT, SMOOTH_MIN, SMOOTH_MAX, WRAP, SNAP, PINGPONG,
    ARCSINE, ARCCOSINE, ARCTAN2, SINH, COSH, TANH };
// RNA MODULO is fmod, whose remainder is computed from the exact binary
// operands. A rounded float quotient (including GPU OpFRem lowering) can lose
// an entire period. Binary long division keeps the 24-bit significands exact.
float math_remainder(float a,float b) {
    let sign_bit=asuint(a)&0x80000000u;
    var x=asuint(a)&0x7fffffffu;var y=asuint(b)&0x7fffffffu;
    if(y==0u) return 0.;
    if(x>=0x7f800000u || y>0x7f800000u) return asfloat(0x7fc00000u);
    if(x<y) return a;
    var ex=int(x>>23u);var ey=int(y>>23u);
    x&=0x7fffffu;y&=0x7fffffu;
    if(ex==0) {ex=1;while(x<0x800000u) {x<<=1u;ex--;}} else x|=0x800000u;
    if(ey==0) {ey=1;while(y<0x800000u) {y<<=1u;ey--;}} else y|=0x800000u;
    while(ex>ey) {
        if(x>=y) x-=y;
        x<<=1u;ex--;
    }
    if(x>=y) x-=y;
    if(x==0u) return asfloat(sign_bit);
    while(x<0x800000u) {x<<=1u;ey--;}
    let bits=ey>0 ? (x&0x7fffffu)|(uint(ey)<<23u) : x>>uint(1-ey);
    return asfloat(sign_bit|bits);
}

// The documented floor step rounds its product before subtracting it. Metal
// drops Slang's local precise qualifier; retain this rounding boundary there
// too, instead of allowing a fused multiply/subtract to leave a whole-period
// residual for large quotients.
[noinline] float math_rounded_product(float a,float b) { return a*b; }

// Duplicate Blender Value labels receive positional suffixes in legal identifiers.
// Fold authored constant operations before the native driver sees the full dispatcher.
[ForceInline] public float Math(MathOperation Operation,float Value,float Value_001,float Value_002,bool Clamp) {
    float a=Value,b=Value_001,c=Value_002,r=0.;
    switch(Operation) {
        case MathOperation.ADD:r=a+b;break;
        case MathOperation.SUBTRACT:r=a-b;break;
        case MathOperation.MULTIPLY:r=a*b;break;
        case MathOperation.DIVIDE:r=b==0.?0.:a/b;break;
        case MathOperation.MULTIPLY_ADD:r=a*b+c;break;
        case MathOperation.MINIMUM:r=min(a,b);break;
        case MathOperation.MAXIMUM:r=max(a,b);break;
        case MathOperation.LESS_THAN:r=a<b?1.:0.;break;
        case MathOperation.GREATER_THAN:r=a>b?1.:0.;break;
        case MathOperation.ABSOLUTE:r=abs(a);break;
        case MathOperation.FLOOR:r=floor(a);break;
        case MathOperation.CEIL:r=ceil(a);break;
        case MathOperation.POWER:
            if(b==0.) r=1.;
            else if(a==0.) r=0.;
            else if(a>0.) r=pow(a,b);
            else if(b==trunc(b)) r=pow(-a,b)*(frac(abs(b)*.5)==0.?1.:-1.);
            break;
        case MathOperation.FRACT:r=frac(a);break;
        case MathOperation.TRUNC:r=trunc(a);break;
        case MathOperation.SINE:r=sin(a);break;
        case MathOperation.COSINE:r=cos(a);break;
        case MathOperation.TANGENT:r=tan(a);break;
        case MathOperation.ARCTANGENT:r=atan(a);break;
        case MathOperation.SQRT:r=a>0.?sqrt(a):0.;break;
        case MathOperation.SIGN:r=sign(a);break;
        case MathOperation.RADIANS:r=a*0.017453292519943295;break;
        case MathOperation.DEGREES:r=a*57.29577951308232;break;
        case MathOperation.EXPONENT:r=exp(a);break;
        case MathOperation.ROUND:r=floor(a+.5);break;
        case MathOperation.MODULO:r=b==0.?0.:math_remainder(a,b);break;
        case MathOperation.FLOORED_MODULO: { precise float value=b==0.?0.:a-math_rounded_product(floor(a/b),b);r=value;break; }
        case MathOperation.COMPARE:r=abs(a-b)<=max(c,1.1920928955078125e-7)?1.:0.;break;
        case MathOperation.LOGARITHM:r=a>0. && b>0. && b!=1.?log(a)/log(b):0.;break;
        case MathOperation.INVERSE_SQRT:r=a>0.?rsqrt(a):0.;break;
        case MathOperation.SMOOTH_MIN:
        case MathOperation.SMOOTH_MAX: {
            let h=c>0.?max(c-abs(a-b),0.)/c:0.;
            let smooth=h*h*h*c/6.;
            r=int(Operation) == int(MathOperation.SMOOTH_MIN)?min(a,b)-smooth:max(a,b)+smooth;
            break;
        }
        case MathOperation.WRAP: {
            let range=b-c;
            precise float value=range!=0.?a-math_rounded_product(range,floor((a-c)/range)):c;r=value;
            break;
        }
        case MathOperation.SNAP:r=b==0.?0.:floor(a/b)*b;break;
        case MathOperation.PINGPONG:
            r=b==0.?0.:abs(frac((a-b)/(2.*b))*2.*b-b);break;
        case MathOperation.ARCSINE:r=asin(clamp(a,-1.,1.));break;
        case MathOperation.ARCCOSINE:r=acos(clamp(a,-1.,1.));break;
        case MathOperation.ARCTAN2:r=a==0. && b==0.?0.:atan2(a,b);break;
        case MathOperation.SINH:r=sinh(a);break;
        case MathOperation.COSH:r=cosh(a);break;
        case MathOperation.TANH:r=tanh(a);break;
    }
    return Clamp?saturate(r):r;
}

// Tangent-space Normal Map over the imported active-render-UV tangent frame.
public float3 Normal_Map(float Strength,float3 Color,ShadingContext ctx) {
    float3 n=2.*Color-1.;
    float k=max(Strength,0.);
    float3 mapped=float3(n.xy*k,1.+saturate(k)*(n.z-1.));
    float3 normal=normalize(ctx.NORMAL);
    float3 tangent=normalize(ctx.TANGENT-normal*dot(ctx.TANGENT,normal));
    float3 world=mapped.x*tangent+mapped.y*ctx.tangent_sign*cross(normal,tangent)+mapped.z*normal;
    return dot(world,world)>0.?normalize(world):normal;
}
public float Layer_Weight_Fresnel(float Blend,float3 Normal,ShadingContext ctx) {
    return Fresnel(1./max(1.-Blend,1.e-5),Normal,ctx);
}
