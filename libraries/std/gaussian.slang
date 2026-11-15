import vork;
import std.gaussian_math;
public struct Inputs {
    Attribute<float3> sh_0_0;
    Attribute<float3> sh_1_0; Attribute<float3> sh_1_1; Attribute<float3> sh_1_2;
    Attribute<float3> sh_2_0; Attribute<float3> sh_2_1; Attribute<float3> sh_2_2; Attribute<float3> sh_2_3; Attribute<float3> sh_2_4;
    Attribute<float3> sh_3_0; Attribute<float3> sh_3_1; Attribute<float3> sh_3_2; Attribute<float3> sh_3_3;
    Attribute<float3> sh_3_4; Attribute<float3> sh_3_5; Attribute<float3> sh_3_6;
};
public Closure surface(ShadingContext ctx,Inputs i) {
    float3 coefficients[16]={read_attribute(i.sh_0_0,ctx).value,read_attribute(i.sh_1_0,ctx).value,read_attribute(i.sh_1_1,ctx).value,read_attribute(i.sh_1_2,ctx).value,
        read_attribute(i.sh_2_0,ctx).value,read_attribute(i.sh_2_1,ctx).value,read_attribute(i.sh_2_2,ctx).value,read_attribute(i.sh_2_3,ctx).value,read_attribute(i.sh_2_4,ctx).value,
        read_attribute(i.sh_3_0,ctx).value,read_attribute(i.sh_3_1,ctx).value,read_attribute(i.sh_3_2,ctx).value,read_attribute(i.sh_3_3,ctx).value,read_attribute(i.sh_3_4,ctx).value,read_attribute(i.sh_3_5,ctx).value,read_attribute(i.sh_3_6,ctx).value};
    let direction=world_to_object(ctx,ctx.POSITION-ctx.incoming)-world_to_object(ctx,ctx.POSITION);
    let color=gaussian_radiance(coefficients,direction);
    // Captures are fitted to sRGB imagery. Blender evaluates their radiance in
    // the capture space, then decodes to the renderer's scene-linear space.
    let linear=select(color<=float3(.04045),color/12.92,pow((color+.055)/1.055,float3(2.4)));
    Closure c;c.base_weight=0.;c.specular_weight=0.;
    c.emission_color=linear;c.emission_luminance=1.;
    return c;
}
