import vork;
import std.view;
public struct Inputs {
    float exposure = 0.;
    [InputRange(0., 3.402823466e38, false, true)] float gamma = 1.;
};
// Linear Rec.709 as Display P3 code values: P3 primaries, then the sRGB transfer.
float3 standard_display_p3(float3 c) {
    return view_encode_srgb(float3(dot(c, float3(.82246197, .17753803, 0.)),
                                   dot(c, float3(.03319420, .96680580, 0.)),
                                   dot(c, float3(.01708263, .07239744, .91051993))));
}
// Linear Rec.709 as Rec.2100 code values: Rec.2020 primaries with unit white at the destination's reference
// white, limited to its peak, then ST 2084 or HLG.
float3 standard_rec2100(float3 c, ViewContext ctx) {
    float3 nits = clamp(float3(dot(c, float3(.62740390, .32928304, .04331307)),
                               dot(c, float3(.06909729, .91954040, .01136232)),
                               dot(c, float3(.01639144, .08801331, .89559525))) * ctx.reference_white,
                        float3(0.), float3(ctx.peak_luminance));
    if (uint(ctx.display) == uint(DisplayTarget::Rec2100Pq)) return view_encode_pq(nits);
    return view_encode_hlg(nits, ctx.peak_luminance);
}
public ViewResult view(ViewContext ctx, Inputs inputs) {
    float3 c = view_linear_rec709(ctx) * exp2(inputs.exposure);
    float3 mapped = select(c > float3(.0031308), 1.055 * pow(max(c, float3(0.)), float3(1. / 2.4)) - .055, 12.92 * c);
    if (uint(ctx.display) == uint(DisplayTarget::Srgb)) return finish_view(mapped, ctx, inputs.gamma);
    // Standard encodes for a wider display itself. Its creative gamma acts on the sRGB code values, as it
    // does for an sRGB destination, so the grade does not depend on the display.
    if (inputs.gamma != 1.) c = view_decode_srgb(pow(max(mapped, float3(0.)), float3(1. / inputs.gamma)));
    if (uint(ctx.display) == uint(DisplayTarget::DisplayP3)) return view_from_codes(standard_display_p3(c), ctx.display);
    return view_from_codes(standard_rec2100(c, ctx), ctx.display);
}
