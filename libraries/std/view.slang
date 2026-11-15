import vork;
// IEC sRGB transfer, extended by the linear segment below zero.
public float3 view_decode_srgb(float3 c) {
    return select(c > float3(.04045), pow(max((c + .055) / 1.055, float3(0.)), float3(2.4)), c / 12.92);
}
// The inverse: display-linear light to sRGB-transfer code values, as Display P3 also uses.
public float3 view_encode_srgb(float3 c) {
    return select(c > float3(.0031308), 1.055 * pow(max(c, float3(0.)), float3(1. / 2.4)) - .055, 12.92 * c);
}
// SMPTE ST 2084 (PQ): display light in cd/m² to code values.
public float3 view_encode_pq(float3 nits) {
    float3 l = pow(max(nits, float3(0.)) / 10000., float3(.1593017578125));
    return pow((.8359375 + 18.8515625 * l) / (1. + 18.6875 * l), float3(78.84375));
}
// BT.2100 HLG for a display of `peak` cd/m²: Rec.2020 display light in cd/m² to the signal, through the inverse
// OOTF with the system gamma for that peak (the extended formula outside 400 to 2000 cd/m²) and the OETF. The
// surround gain is bounded as OpenColorIO bounds it.
public float3 view_encode_hlg(float3 nits, float peak) {
    float relative = peak / 1000.;
    float gamma = peak >= 400. && peak <= 2000. ? 1.2 + .42 * .30102999566 * log2(relative) : 1.2 * pow(1.111, log2(relative));
    float3 light = max(nits, float3(0.)) / peak;
    float3 scene = light * pow(max(dot(light, float3(.2627, .6780, .0593)), 1e-4 / pow(3., gamma)), (1. - gamma) / gamma);
    return select(scene <= float3(1. / 12.), sqrt(3. * scene), .17883277 * log(max(12. * scene - .28466892, float3(.000001))) + .55991073);
}
// Linear Rec.709 to display-linear XYZ D65, with Y measured in cd/m².
public float3 view_xyz(float3 rgb, ViewContext ctx) {
    return float3(dot(rgb,float3(.4123908,.35758434,.18048079)),
                  dot(rgb,float3(.21263901,.71516868,.07219232)),
                  dot(rgb,float3(.01933082,.11919478,.95053215))) * ctx.reference_white;
}
// Finish a view that holds its result as sRGB code values, as the bundled views do: their creative gamma
// acts on those codes. For an sRGB destination the codes are the result as they are; for any other
// destination the result is their absolute XYZ, which the engine converts.
public ViewResult finish_view(float3 srgb, ViewContext ctx, float gamma) {
    if (gamma != 1.) srgb = pow(max(srgb, float3(0.)), float3(1. / gamma));
    // Compared as integers: Slang's enum equality costs every view program about 140 ms per compiler pass.
    if (uint(ctx.display) == uint(DisplayTarget::Srgb)) return view_from_codes(srgb, DisplayTarget::Srgb);
    return view_from_xyz(view_xyz(view_decode_srgb(srgb), ctx));
}

// Decode the composition's encoding before applying a view's linear operations: its transfer, then its
// primaries to Rec.709 (the pinned matrices of `composition.color_space`; ACEScg's white adapted to D65).
// Colour outside Rec.709's gamut keeps its signed coordinates.
public float3 view_linear_rec709(ViewContext ctx) {
    float3 c = ctx.color;
    uint space = uint(ctx.color_space);
    if (space == uint(WorkingColorSpace::Srgb)) return view_decode_srgb(c);
    if (space == uint(WorkingColorSpace::DisplayP3)) {
        c = view_decode_srgb(c);
        return float3(dot(c, float3(1.22494018, -.22494021, 0.)),
                      dot(c, float3(-.04205693, 1.04205692, 0.)),
                      dot(c, float3(-.01963755, -.07863605, 1.09827363)));
    }
    if (space == uint(WorkingColorSpace::LinearRec2020))
        return float3(dot(c, float3(1.66049087, -.58764118, -.07284992)),
                      dot(c, float3(-.12455040, 1.13289988, -.00834931)),
                      dot(c, float3(-.01815077, -.10057894, 1.11872971)));
    if (space == uint(WorkingColorSpace::ACEScg))
        return float3(dot(c, float3(1.70505095, -.62179214, -.08325893)),
                      dot(c, float3(-.13025640, 1.14080477, -.01054827)),
                      dot(c, float3(-.02400335, -.12896900, 1.15297234)));
    return c;
}
