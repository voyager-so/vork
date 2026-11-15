// std/agx's own rendering for a wide-gamut or HDR destination: Blender 5.2.1's AgX view for that display, as
// the view's original OCIO cube. See device-views-manifest.json and scripts/color/bake-device-views.py.
// Sample (r,g,b) sits at pixel (57 b + r, g); each axis is log2 of linear FilmLight E-Gamut, from -12.47393 to
// 12.5260688117 over the 57 samples. The Display P3 cube holds gamma 2.4 codes; the HDR cube ("AgX - HDR 1000
// nits") holds Rec.2100 HLG for a 1000 cd/m² display.
import vork;
import std.view;
// Whether std/agx has its own bake for this destination. The library binds `cube` only for a display it is
// baked for; an sRGB destination has the view's sRGB rendering and never reads it.
public bool agx_destination_baked(ViewContext ctx, Image cube) {
    return uint(ctx.display) != uint(DisplayTarget::Srgb) && vork_slot(cube) != 0xffffffffu;
}
float3 agx_cube_sample(Image cube, float3 at) {
    return vork_view_texel(cube, int2(int(57. * at.b + at.r), int(at.g))).rgb;
}
// The cube at scene-linear Rec.709 `rgb`, interpolated over tetrahedra as OpenColorIO evaluates it.
float3 agx_cube(Image cube, float3 rgb) {
    float n = 57.;
    float low = -12.47393;
    float high = 12.5260688117;
    float3 e = float3(dot(rgb, float3(.5593711138, .3047833443, .1358455569)),
                      dot(rgb, float3(.0762207061, .7879717946, .1358074695)),
                      dot(rgb, float3(.0655267090, .1645467579, .7699264884)));
    float3 t = clamp((log2(max(e, float3(exp2(low)))) - low) / (high - low) * (n - 1.), float3(0.), float3(n - 1.));
    float3 corner = min(floor(t), float3(n - 2.));
    float3 f = t - corner;
    // The tetrahedron follows the fractions in descending order: from the cell's low corner, one step along
    // the axis of the largest, then the middle, then the smallest. Equal fractions weight their shared
    // vertex zero, so their order does not matter.
    bool rg = f.r >= f.g;
    bool rb = f.r >= f.b;
    bool gb = f.g >= f.b;
    float3 largest = float3(rg && rb ? 1. : 0., !rg && gb ? 1. : 0., !rb && !gb ? 1. : 0.);
    float3 smallest = float3(!rg && !rb ? 1. : 0., rg && !gb ? 1. : 0., rb && gb ? 1. : 0.);
    float upper = dot(f, largest);
    float lower = dot(f, smallest);
    float middle = dot(f, float3(1.) - largest - smallest);
    return (1. - upper) * agx_cube_sample(cube, corner)
        + (upper - middle) * agx_cube_sample(cube, corner + largest)
        + (middle - lower) * agx_cube_sample(cube, corner + float3(1.) - smallest)
        + lower * agx_cube_sample(cube, corner + float3(1.));
}
// The bake's code values for the destination at scene-linear Rec.709 `ctx.color`, with the view's creative
// gamma on them.
public ViewResult agx_destination(ViewContext ctx, Image cube, float gamma) {
    if (uint(ctx.display) == uint(DisplayTarget::DisplayP3)) {
        // The cube's gamma 2.4 codes take the display's sRGB transfer.
        float3 codes = view_encode_srgb(pow(max(agx_cube(cube, ctx.color), float3(0.)), float3(2.4)));
        if (gamma != 1.) codes = pow(max(codes, float3(0.)), float3(1. / gamma));
        return view_from_codes(codes, ctx.display);
    }
    // The published view puts unit scene white near 148 cd/m²: its reference white is 100 cd/m² per display
    // unit. With the reference white at 203 cd/m², unit scene white lands on it: the manifest's 203 recipe,
    // a scene exposure of 0.3738 stops ahead of the same view.
    float3 hlg = agx_cube(cube, ctx.color * (ctx.reference_white == 203. ? 1.2957664132118243 : 1.));
    // The creative gamma acts on the HLG signal for either Rec.2100 display, so both show the same grade,
    // and a signal within one stays within the display's peak.
    if (gamma != 1.) hlg = pow(max(hlg, float3(0.)), float3(1. / gamma));
    if (uint(ctx.display) == uint(DisplayTarget::Rec2100Hlg)) return view_from_codes(hlg, ctx.display);
    // ST 2084 from the HLG signal: the inverse OETF, then the 1000 cd/m² OOTF (system gamma 1.2) with
    // OpenColorIO's floor on the surround luminance.
    float3 scene = select(hlg <= float3(.5), hlg * hlg / 3., (exp((hlg - .55991073) / .17883277) + .28466892) / 12.);
    float y = max(abs(dot(scene, float3(.2627, .6780, .0593))), pow(1e-4, 1. / 1.2) / 3.);
    return view_from_codes(view_encode_pq(1000. * scene * pow(y, .2)), ctx.display);
}
