// Blender 5.2.1 AgX sRGB and looks, using the original core half-float samples.
// See agx-manifest.json and scripts/color/bake-agx-looks.py.
import vork;
// Blender's look names: None, Very High Contrast, ..., Very Low Contrast.
public enum AgxLook {
    None = 0, Punchy = 1, Greyscale = 2, Very_High_Contrast = 3,
    High_Contrast = 4, Medium_High_Contrast = 5, Base_Contrast = 6,
    Medium_Low_Contrast = 7, Low_Contrast = 8, Very_Low_Contrast = 9
};
import std.view;
public struct Inputs {
    float exposure = 0.;
    [InputRange(0., 3.402823466e38, false, true)] float gamma = 1.;

    AgxLook look = AgxLook::None;
    // Unbound: the library's bake above.
    Image lut;
    // Blender's AgX view for a Display P3 or Rec.2100 destination, in the layout of device-views-manifest.json.
    // Unbound: the library's bake for that display.
    Image destination_lut;
};
import blender.display_lut;
import std.agx_destination;
public ViewResult view(ViewContext ctx, Inputs inputs) {
    ctx.color = view_linear_rec709(ctx) * exp2(inputs.exposure);
    // A destination the view is baked for takes that bake's code values: AgX's own rendering for the display.
    // The looks and an authored `lut` exist for sRGB only; a render to another display refuses them.
    // Enums are compared as integers: Slang's enum equality is slow to compile.
    if (int(inputs.look) == int(AgxLook::None) && agx_destination_baked(ctx, inputs.destination_lut))
        return agx_destination(ctx, inputs.destination_lut, inputs.gamma);
    return finish_view(display_lut(ctx,inputs.lut,int(inputs.look),10.),ctx,inputs.gamma);
}
