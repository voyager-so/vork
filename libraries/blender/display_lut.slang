import vork;
public float3 display_lut(ViewContext ctx,Image lut,int look,float looks) {
    float n = 65.;
    float step = (12.5260688117 - (-12.47393)) / (n - 1.);
    float first = exp2((-12.47393) + step);
    float3 c = max(ctx.color, float3(0.));
    // Entry zero is exact black; its interval interpolates in linear light.
    float3 logarithmic = (log2(max(c, float3(1e-30))) - (-12.47393)) / step;
    float3 index = clamp(select(c < float3(first), c / first, logarithmic), float3(0.), float3(n - 1.));
    // Trilinear: bilinear red/green within the two blue tiles, then blue.
    float tile = min(floor(index.b), n - 2.);
    float2 size = float2(n * n, n * looks);
    float2 uv = (float2(tile * n + index.r, float(look) * n + index.g) + .5) / size;
    float3 a = sample(lut, uv, ctx).rgb;
    float3 b = sample(lut, uv + float2(n / size.x, 0.), ctx).rgb;
    return lerp(a, b, index.b - tile);
}
