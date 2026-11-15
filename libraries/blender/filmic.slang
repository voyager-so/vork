// Blender 5.2.1's Filmic view (display sRGB) and its seven looks, as a display
// material: composition.view = {"program": "blender/filmic", "inputs": {"look": ...}}.
// `lut` is Blender's OCIO transform sampled per look (libraries/blender/filmic.exr,
// baked by scripts/color/bake-filmic.py; filmic-manifest.json pins the OCIO sources).
// Sample (r,g,b) of look L sits at pixel (65 b + r, 65 L + g); each axis is exact
// black, then 2^(-12.47393 + 25 i/64) for i = 1..64. Samples are display-encoded.
import vork;
// Blender's look names: None, Very High Contrast, ..., Very Low Contrast.
public enum FilmicLook {
    None = 0, Very_High_Contrast = 1, High_Contrast = 2, Medium_High_Contrast = 3,
    Medium_Contrast = 4, Medium_Low_Contrast = 5, Low_Contrast = 6, Very_Low_Contrast = 7
};
import std.view;
public struct Inputs {
    float exposure = 0.;
    [InputRange(0., 3.402823466e38, false, true)] float gamma = 1.;

    FilmicLook look = FilmicLook::None;
    // Unbound: the library's bake above.
    Image lut;
};
import blender.display_lut;
public ViewResult view(ViewContext ctx, Inputs inputs) {
    ctx.color = view_linear_rec709(ctx) * exp2(inputs.exposure);
    return finish_view(display_lut(ctx,inputs.lut,int(inputs.look),8.),ctx,inputs.gamma);
}
