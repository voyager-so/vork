//! The pixel node set's kernels (motion-model.md §12.5 "Pixels", stage 5). The executor handles the structural
//! nodes (`convert`, `channels`, the temporal fold); everything else is here.
//!
//! Every node works on premultiplied linear light (the image plan converts at typed edges). Lengths are frame
//! pixels, multiplied by the render scale. Each node states its output domain: a blur grows its input by 3σ, a
//! transform maps its input's bounds, a merge covers what its operator can paint.

use super::exec::{Args, Executor};
use super::image::{ColorTag, Domain, Image};
use super::pass::Kernel;
use super::PixelError;

type R<T> = Result<T, PixelError>;

/// The largest Gaussian σ (in taps) one blur pass runs: 3·σ = 300 taps. A larger σ runs as two passes (see
/// [`blur`]): a dense one at σ = step, then one whose taps are `step` pixels apart.
pub const MAX_SIGMA: f64 = 100.0;

macro_rules! kernel {
    ($name:ident, $body:expr) => {
        pub fn $name() -> Kernel<'static> {
            Kernel {
                name: stringify!($name),
                body: $body,
            }
        }
    };
}

/// One Gaussian pass along P(0).xy (a unit step in pixels): σ = P(0).z, taps ±P(0).w. Axis-aligned steps land on
/// texel centres, so `bilinear` reads exact texels there.
const BLUR_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let dir = P(0u).xy;
    let sigma = P(0u).z;
    let n = i32(P(0u).w);
    if (sigma <= 0.0) { return bilinear(0u, p); }
    let k = -0.5 / (sigma * sigma);
    var acc = bilinear(0u, p);
    var wsum = 1.0;
    for (var i = 1; i <= 320; i = i + 1) {
        if (i > n) { break; }
        let fi = f32(i);
        let w = exp(k * fi * fi);
        acc = acc + (bilinear(0u, p + dir * fi) + bilinear(0u, p - dir * fi)) * w;
        wsum = wsum + 2.0 * w;
    }
    return acc / wsum;
}
"#;
kernel!(blur_pass, BLUR_BODY);

const DEFOCUS_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let dir = vec2<i32>(P(0u).xy);
    let n = u32(P(0u).w);
    let centre = vec2<i32>(floor(p));
    var acc = px(0u, centre);
    for (var tap = 1u; tap <= 156u; tap = tap + 1u) {
        if (tap > n) { break; }
        let weight = P(1u + (tap - 1u) / 4u)[(tap - 1u) % 4u];
        let offset = dir * i32(tap);
        acc = acc + (px(0u, centre + offset) + px(0u, centre - offset)) * weight;
    }
    return acc / P(0u).z;
}
"#;
kernel!(defocus_pass, DEFOCUS_BODY);

/// Glow's bright pass: colour above P(0).x luminance (soft knee P(0).y), times the tint P(1).
const BRIGHT_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let c = here(0u, p);
    let l = dot(unpremul(c), vec3<f32>(0.2126, 0.7152, 0.0722));
    let t = P(0u).x;
    let knee = max(P(0u).y, 1e-4);
    let m = smoothstep(t - knee, t + knee, l);
    return c * m * P(1u);
}
"#;
kernel!(bright, BRIGHT_BODY);

/// `a + b · P(0)` (glow's add-back; premultiplied, so the glow brightens and spreads alpha).
const ADD_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let a = here(0u, p);
    let b = here(1u, p) * P(0u).x;
    return vec4<f32>(a.rgb + b.rgb, min(a.a + b.a * (1.0 - a.a), 1.0));
}
"#;
kernel!(add, ADD_BODY);

/// Displacement: by input 1's red/green around 0.5 (P(0).w = 1), or by seeded fBm value noise. P(0).xy = amount,
/// P(1) = frequency, octaves, seed, evolve.
const DISPLACE_BODY: &str = r#"
fn hash3(q: vec3<f32>) -> f32 {
    var h = dot(q, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453123);
}
fn vnoise(x: vec3<f32>) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let a = mix(mix(hash3(i), hash3(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
                mix(hash3(i + vec3<f32>(0.0, 1.0, 0.0)), hash3(i + vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y);
    let b = mix(mix(hash3(i + vec3<f32>(0.0, 0.0, 1.0)), hash3(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
                mix(hash3(i + vec3<f32>(0.0, 1.0, 1.0)), hash3(i + vec3<f32>(1.0, 1.0, 1.0)), u.x), u.y);
    return mix(a, b, u.z) * 2.0 - 1.0;
}
fn fbm(x: vec3<f32>, octaves: u32) -> f32 {
    var s = 0.0;
    var amp = 0.5;
    var q = x;
    var norm = 0.0;
    for (var o = 0u; o < 8u; o = o + 1u) {
        if (o >= octaves) { break; }
        s = s + amp * vnoise(q);
        norm = norm + amp;
        amp = amp * 0.5;
        q = q * 2.03;
    }
    return s / max(norm, 1e-6);
}
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let amount = P(0u).xy;
    var d = vec2<f32>(0.0);
    if (P(0u).w > 0.5) {
        let m = here(1u, p);
        d = (unpremul(m).rg - vec2<f32>(0.5)) * 2.0 * m.a;
    } else {
        let f = P(1u).x;
        let oct = u32(P(1u).y);
        let seed = P(1u).z;
        let z = P(1u).w + seed * 17.13;
        let q = vec3<f32>(p * f, z);
        d = vec2<f32>(fbm(q, oct), fbm(q + vec3<f32>(31.4, 47.2, seed * 3.7 + 11.0), oct));
    }
    return bilinear(0u, p - d * amount);
}
"#;
kernel!(displace, DISPLACE_BODY);

/// Grade (in linear light): P(0) = exposure, contrast, pivot, saturation; P(1).rgb = lift, P(2).rgb = gamma,
/// P(3).rgb = gain, P(4).rgb = offset, P(5).rgb = tint, P(6).x = hue (degrees).
const GRADE_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let c = here(0u, p);
    if (c.a <= 0.0) { return vec4<f32>(0.0); }
    var rgb = unpremul(c);
    rgb = rgb * exp2(P(0u).x);
    rgb = (rgb - vec3<f32>(P(0u).z)) * P(0u).y + vec3<f32>(P(0u).z);
    rgb = P(3u).rgb * (rgb + P(1u).rgb * (vec3<f32>(1.0) - rgb));
    let g = max(P(2u).rgb, vec3<f32>(1e-4));
    rgb = sign(rgb) * pow(abs(rgb), vec3<f32>(1.0) / g);
    rgb = rgb + P(4u).rgb;
    // Hue: rotate about the grey axis (Rodrigues).
    let h = radians(P(6u).x);
    if (h != 0.0) {
        let k = vec3<f32>(0.57735027);
        let cs = cos(h);
        rgb = rgb * cs + cross(k, rgb) * sin(h) + k * dot(k, rgb) * (1.0 - cs);
    }
    let l = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    rgb = mix(vec3<f32>(l), rgb, P(0u).w);
    rgb = rgb * P(5u).rgb;
    return vec4<f32>(rgb * c.a, c.a);
}
"#;
kernel!(grade, GRADE_BODY);

const INVERT_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let c = here(0u, p);
    return vec4<f32>(vec3<f32>(c.a) - c.rgb, c.a);
}
"#;
kernel!(invert, INVERT_BODY);

const CHROMA_MATTE_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let c = here(0u, p);
    let rgb = unpremul(c);
    let channel = u32(P(0u).x);
    let other = max(rgb[(channel + 1u) % 3u], rgb[(channel + 2u) % 3u]);
    let excess = max(rgb[channel] - other, 0.0);
    let keep = 1.0 - clamp(excess * P(0u).y, 0.0, 1.0);
    return vec4<f32>(keep * c.a, 0.0, 0.0, c.a);
}
"#;
kernel!(chroma_matte, CHROMA_MATTE_BODY);

const CHROMA_KEY_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let c = here(0u, p);
    if (c.a <= 0.0) { return vec4<f32>(0.0); }
    let matte = here(1u, p);
    let alpha = c.a * clamp(matte.r / max(matte.a, 1e-6), 0.0, 1.0);
    var rgb = unpremul(c);
    if (P(0u).z > 0.5) {
        let channel = u32(P(0u).x);
        rgb[channel] = min(rgb[channel], max(rgb[(channel + 1u) % 3u], rgb[(channel + 2u) % 3u]));
    }
    return vec4<f32>(rgb * alpha, alpha);
}
"#;
kernel!(chroma_key, CHROMA_KEY_BODY);

/// Matte: input 0 by input 1's alpha (P(0).x = 0) or luminance (1), inverted when P(0).y = 1.
const MATTE_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let s = here(0u, p);
    let m = here(1u, p);
    var k = m.a;
    if (P(0u).x > 0.5) { k = dot(m.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)); }
    if (P(0u).y > 0.5) { k = 1.0 - k; }
    return s * clamp(k, 0.0, 1.0);
}
"#;
kernel!(matte, MATTE_BODY);

/// Merge `a` (input 0) onto `b` (input 1): P(0).x = operator, P(0).y = mix.
const MERGE_BODY: &str = r#"
fn blend(m: u32, cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    switch m {
        case 7u: { return cb * cs; }
        case 8u: { return cb + cs - cb * cs; }
        case 9u: { return select(1.0 - 2.0 * (1.0 - cb) * (1.0 - cs), 2.0 * cb * cs, cb <= vec3<f32>(0.5)); }
        case 10u: { return min(cb, cs); }
        case 11u: { return max(cb, cs); }
        default: { return abs(cb - cs); }
    }
}
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let a = here(0u, p);
    let b = here(1u, p);
    let m = u32(P(0u).x);
    var r = vec4<f32>(0.0);
    switch m {
        case 0u: { r = a + b * (1.0 - a.a); }            // over
        case 1u: { r = b + a * (1.0 - b.a); }            // under
        case 2u: { r = a * b.a; }                        // in
        case 3u: { r = a * (1.0 - b.a); }                // out
        case 4u: { r = a * b.a + b * (1.0 - a.a); }      // atop
        case 5u: { r = a * (1.0 - b.a) + b * (1.0 - a.a); } // xor
        case 6u: { r = a + b; r.a = min(r.a, 1.0); } // plus
        case 12u: { r = max(a, b); }
        case 13u: { r = min(a, b); }
        default: {
            // A blend mode (W3C Compositing): cs(1 − αb) + cb(1 − αs) + αs·αb·B(Cb, Cs).
            let cs = unpremul(a);
            let cb = unpremul(b);
            let rgb = a.rgb * (1.0 - b.a) + b.rgb * (1.0 - a.a) + a.a * b.a * blend(m, cb, cs);
            r = vec4<f32>(rgb, a.a + b.a * (1.0 - a.a));
        }
    }
    return mix(b, r, P(0u).y);
}
"#;
kernel!(merge, MERGE_BODY);

/// Warp about a point (moda-scene `envelope::Deform`, inverted to resample): P(0).xy = centre (output px), P(0).z =
/// radius, P(0).w = amount; P(1).x = kind (0 bulge, 1 twist). Bulge moves a point at distance r to
/// r · (1 + a · (1 − (r / R)²)²): each output pixel finds its source radius by Newton's method (the map is monotonic in
/// r for a in [−0.95, 1.2]). Twist turns by amount · (1 − r / R)² degrees, so its inverse turns back.
const WARP_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let c = P(0u).xy;
    let big_r = P(0u).z;
    let a = P(0u).w;
    let d = p - c;
    let r1 = length(d);
    if (big_r <= 0.0 || r1 >= big_r || r1 == 0.0) { return bilinear(0u, p); }
    if (P(1u).x > 0.5) {
        let x = r1 / big_r;
        let th = -radians(a * (1.0 - x) * (1.0 - x));
        let cs = cos(th);
        let sn = sin(th);
        return bilinear(0u, c + vec2<f32>(d.x * cs - d.y * sn, d.x * sn + d.y * cs));
    }
    // Solve r · (1 + a (1 − (r/R)²)²) = r1 for r in [0, R].
    var r = r1;
    for (var i = 0; i < 12; i = i + 1) {
        let x = r / big_r;
        let k = 1.0 - x * x;
        let f = r * (1.0 + a * k * k) - r1;
        let df = 1.0 + a * k * (1.0 - 5.0 * x * x);
        r = clamp(r - f / max(df, 1e-4), 0.0, big_r);
    }
    return bilinear(0u, c + d * (r / r1));
}
"#;
kernel!(warp, WARP_BODY);
/// Threshold: a soft step of alpha (P(0).z = 0) or luminance (1) at level P(0).x over softness P(0).y, inverted when
/// P(0).w = 1. Alpha: the step is the new alpha, over the straight colour. Luma: the step scales the pixel (a luma key).
const THRESHOLD_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let c = here(0u, p);
    let rgb = unpremul(c);
    var v = c.a;
    if (P(0u).z > 0.5) { v = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722)) * c.a; }
    let lo = P(0u).x - P(0u).y * 0.5;
    let hi = P(0u).x + P(0u).y * 0.5;
    var k = select(step(P(0u).x, v), clamp((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0), P(0u).y > 0.0);
    k = k * k * (3.0 - 2.0 * k);
    if (P(0u).w > 0.5) { k = 1.0 - k; }
    if (P(0u).z > 0.5) { return c * k; }
    return vec4<f32>(rgb * k, k);
}
"#;
kernel!(threshold, THRESHOLD_BODY);
/// Light rays (radial light scattering): each output pixel averages the bright part of input 0 along the segment
/// towards the light at P(0).xy, `P(0).z` of the way there in P(0).w samples, sample i weighted P(1).x^i. Bright part:
/// luminance above P(1).y (soft knee 0.1). The rays, times P(1).z and the tint P(2), are added over the source
/// (P(1).w = 0) or returned alone (1). With occluders (P(3).x = 1), input 1's alpha blocks the light: sample i is
/// weighted by the transmittance Π_{j ≤ i} (1 − occluder alpha at sample j), the light that gets through the occluders
/// between it and the pixel (sample i itself included). Without, the sum is the one it always was.
const RAYS_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let light = P(0u).xy;
    let n = max(i32(P(0u).w), 1);
    let step_v = (light - p) * P(0u).z / f32(n);
    let decay = P(1u).x;
    let t = P(1u).y;
    var q = p;
    var w = 1.0;
    var wsum = 0.0;
    var acc = vec4<f32>(0.0);
    let occluded = P(3u).x > 0.5;
    var through = 1.0;
    for (var i = 0; i < 512; i = i + 1) {
        if (i >= n) { break; }
        q = q + step_v;
        let s = bilinear(0u, q);
        let l = dot(unpremul(s), vec3<f32>(0.2126, 0.7152, 0.0722));
        if (occluded) {
            through = through * (1.0 - clamp(bilinear(1u, q).a, 0.0, 1.0));
            acc = acc + s * smoothstep(t - 0.1, t + 0.1, l) * w * through;
        } else {
            acc = acc + s * smoothstep(t - 0.1, t + 0.1, l) * w;
        }
        wsum = wsum + w;
        w = w * decay;
    }
    let rays = acc / max(wsum, 1e-6) * P(1u).z * P(2u);
    if (P(1u).w > 0.5) { return vec4<f32>(rays.rgb, min(rays.a, 1.0)); }
    let c = here(0u, p);
    return vec4<f32>(c.rgb + rays.rgb, min(c.a + rays.a * (1.0 - c.a), 1.0));
}
"#;
kernel!(rays, RAYS_BODY);

/// Caustics: a web of light, the edges of 3D cells (F2 − F1 of a seeded Worley field in (x / scale, y / scale,
/// evolve)), domain-warped by value noise, two octaves. P(0) = 1 / scale, evolve, seed, sharpness; P(1) = warp,
/// intensity, masked by input 0's alpha (1) or not (0); P(2) = the straight tint. Light as coverage: premultiplied
/// `tint · v`, alpha `v`, clamped to 1.
const CAUSTICS_BODY: &str = r#"
fn h3(q: vec3<f32>) -> vec3<f32> {
    let r = vec3<f32>(dot(q, vec3<f32>(127.1, 311.7, 74.7)), dot(q, vec3<f32>(269.5, 183.3, 246.1)),
                      dot(q, vec3<f32>(113.5, 271.9, 124.6)));
    return fract(sin(r) * 43758.5453123);
}
fn vn(x: vec3<f32>) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let a = mix(mix(h3(i).x, h3(i + vec3<f32>(1.0, 0.0, 0.0)).x, u.x),
                mix(h3(i + vec3<f32>(0.0, 1.0, 0.0)).x, h3(i + vec3<f32>(1.0, 1.0, 0.0)).x, u.x), u.y);
    let b = mix(mix(h3(i + vec3<f32>(0.0, 0.0, 1.0)).x, h3(i + vec3<f32>(1.0, 0.0, 1.0)).x, u.x),
                mix(h3(i + vec3<f32>(0.0, 1.0, 1.0)).x, h3(i + vec3<f32>(1.0, 1.0, 1.0)).x, u.x), u.y);
    return mix(a, b, u.z) * 2.0 - 1.0;
}
fn web(x: vec3<f32>, seed: f32) -> f32 {
    let i = floor(x);
    let f = fract(x);
    var f1 = 8.0;
    var f2 = 8.0;
    for (var zz = -1; zz <= 1; zz = zz + 1) {
        for (var yy = -1; yy <= 1; yy = yy + 1) {
            for (var xx = -1; xx <= 1; xx = xx + 1) {
                let g = vec3<f32>(f32(xx), f32(yy), f32(zz));
                let r = g + h3(i + g + vec3<f32>(seed * 7.13, seed * 3.71, seed * 1.17)) - f;
                let d = dot(r, r);
                if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) { f2 = d; }
            }
        }
    }
    return sqrt(f2) - sqrt(f1);
}
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let inv = P(0u).x;
    let z = P(0u).y;
    let seed = P(0u).z;
    let sharp = max(P(0u).w, 0.01);
    var x = vec3<f32>(p * inv, z);
    let warp = P(1u).x;
    x = x + vec3<f32>(vn(x * 0.7 + vec3<f32>(3.1, 0.0, 0.0)), vn(x * 0.7 + vec3<f32>(0.0, 5.3, 0.0)), 0.0) * warp;
    let e1 = web(x, seed);
    let e2 = web(x * 2.0 + vec3<f32>(17.0, 9.0, 0.5 * z), seed + 1.0);
    let v = pow(clamp(1.0 - e1 * 2.0, 0.0, 1.0), sharp) + 0.5 * pow(clamp(1.0 - e2 * 2.0, 0.0, 1.0), sharp);
    var k = clamp(v * P(1u).y, 0.0, 1.0);
    if (P(1u).z > 0.5) { k = k * here(0u, p).a; }
    return vec4<f32>(P(2u).rgb * k, k);
}
"#;
kernel!(caustics, CAUSTICS_BODY);

/// The surface the surface nodes read (`normals`, `lighting`, `reflect`): input 0 as a height map, `P(0).x · h` output
/// px high, h its alpha (P(0).y = 0) or its luminance times alpha (1). The normal at an output pixel is from the Sobel
/// estimate of h's gradient: normalize(−P(0).x · ∂h/∂x, −P(0).x · ∂h/∂y, 1), x right, y down, z toward the viewer.
macro_rules! surface_wgsl {
    () => {
        r#"
fn surf_h(q: vec2<i32>) -> f32 {
    let c = px(0u, q);
    if (P(0u).y > 0.5) { return dot(unpremul(c), vec3<f32>(0.2126, 0.7152, 0.0722)) * c.a; }
    return c.a;
}
fn surf_normal(p: vec2<f32>) -> vec3<f32> {
    let q = vec2<i32>(floor(p));
    let a = surf_h(q + vec2<i32>(-1, -1));
    let b = surf_h(q + vec2<i32>(0, -1));
    let c = surf_h(q + vec2<i32>(1, -1));
    let d = surf_h(q + vec2<i32>(-1, 0));
    let f = surf_h(q + vec2<i32>(1, 0));
    let g = surf_h(q + vec2<i32>(-1, 1));
    let h = surf_h(q + vec2<i32>(0, 1));
    let i = surf_h(q + vec2<i32>(1, 1));
    let gx = (c + 2.0 * f + i) - (a + 2.0 * d + g);
    let gy = (g + 2.0 * h + i) - (a + 2.0 * b + c);
    let s = P(0u).x / 8.0;
    return normalize(vec3<f32>(-s * gx, -s * gy, 1.0));
}
"#
    };
}

/// Normals: the surface's normal (see `surface_wgsl`) as data, `N · 0.5 + 0.5` in rgb, alpha 1.
const NORMALS_BODY: &str = concat!(
    surface_wgsl!(),
    r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(surf_normal(p) * 0.5 + vec3<f32>(0.5), 1.0);
}
"#
);
kernel!(normals, NORMALS_BODY);

/// Light on the surface (see `surface_wgsl`), as SVG's feDiffuseLighting / feSpecularLighting compute it. P(1).xyz =
/// the unit vector toward a distant light (P(1).w = 0), or a point light's position in output px (1; its z above the
/// height map's 0). P(2) = the light's straight colour times the strength (w). P(3).x = the specular exponent, P(3).y
/// = 1 for specular. Diffuse: `colour · max(N·L, 0)`, alpha 1. Specular: `colour · max(N·H, 0)^exponent` with
/// H = normalize(L + (0, 0, 1)) (the eye far above), alpha its largest channel (clamped to 1).
const LIGHTING_BODY: &str = concat!(
    surface_wgsl!(),
    r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let n = surf_normal(p);
    var l = P(1u).xyz;
    if (P(1u).w > 0.5) {
        let z = P(0u).x * surf_h(vec2<i32>(floor(p)));
        let v = P(1u).xyz - vec3<f32>(p, z);
        l = v / max(length(v), 1e-6);
    }
    let col = P(2u).rgb * P(2u).w;
    if (P(3u).y > 0.5) {
        let hv = normalize(l + vec3<f32>(0.0, 0.0, 1.0));
        let c = col * pow(max(dot(n, hv), 0.0), P(3u).x);
        return vec4<f32>(c, clamp(max(c.r, max(c.g, c.b)), 0.0, 1.0));
    }
    return vec4<f32>(col * max(dot(n, l), 0.0), 1.0);
}
"#
);
kernel!(lighting, LIGHTING_BODY);

/// Reflect: input 1 (the environment) read by the surface's normal (see `surface_wgsl`). P(0).z = 0: at the point of
/// the environment's area P(1) (x, y, w, h, output px) at `(0.5, 0.5) + N.xy / 2` of it (a sphere map: a flat
/// surface reads its centre, a slope toward its edge). 1: at the pixel itself moved by `N.xy · P(0).w` output px.
const REFLECT_BODY: &str = concat!(
    surface_wgsl!(),
    r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let n = surf_normal(p);
    if (P(0u).z > 0.5) { return bilinear(1u, p + n.xy * P(0u).w); }
    let b = P(1u);
    let uv = vec2<f32>(0.5) + 0.5 * n.xy;
    return bilinear(1u, b.xy + vec2<f32>(0.5) + uv * max(b.zw - vec2<f32>(1.0), vec2<f32>(0.0)));
}
"#
);
kernel!(reflect, REFLECT_BODY);

/// Affine resample: P(0).xy, P(1).xy, P(2).xy = the inverse matrix's columns (x, y, translation); P(3).x = opacity.
const TRANSFORM_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let q = P(0u).xy * p.x + P(1u).xy * p.y + P(2u).xy;
    return bilinear(0u, q) * P(3u).x;
}
"#;
kernel!(transform, TRANSFORM_BODY);

/// Transitions from input 0 to input 1: P(0) = progress, kind, angle (rad), softness (px); P(1).xy = frame size.
const TRANSITION_BODY: &str = r#"
const PI: f32 = 3.14159265;
fn th(q: vec3<f32>) -> f32 {
    return fract(sin(dot(q, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453123);
}
fn tnoise(x: vec3<f32>) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let a = mix(mix(th(i), th(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
                mix(th(i + vec3<f32>(0.0, 1.0, 0.0)), th(i + vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y);
    let b = mix(mix(th(i + vec3<f32>(0.0, 0.0, 1.0)), th(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
                mix(th(i + vec3<f32>(0.0, 1.0, 1.0)), th(i + vec3<f32>(1.0, 1.0, 1.0)), u.x), u.y);
    return mix(a, b, u.z);
}
fn over(b: vec4<f32>, a: vec4<f32>) -> vec4<f32> { return b + a * (1.0 - b.a); }
// Input i at p held inside the frame (a displaced sample never reads past the frame's edge).
fn held(i: u32, p: vec2<f32>) -> vec4<f32> { return bilinear(i, clamp(p, vec2<f32>(0.5), P(1u).xy - vec2<f32>(0.5))); }
fn in_frame(q: vec2<f32>) -> bool { return all(q >= vec2<f32>(0.0)) && all(q <= P(1u).xy); }
// The outgoing image at q with the incoming one `shift` beyond it (a push): each read inside its own frame, so where
// they meet at a fractional offset there is no half-transparent seam.
fn pushed(q: vec2<f32>, shift: vec2<f32>) -> vec4<f32> {
    if (in_frame(q)) { return held(0u, q); }
    if (in_frame(q + shift)) { return held(1u, q + shift); }
    return vec4<f32>(0.0);
}
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let t = clamp(P(0u).x, 0.0, 1.0);
    // At the ends every kind is exactly its input: no soft edge left over, no block average, nothing pulled in from
    // past the frame's edge.
    if (t <= 0.0) { return here(0u, p); }
    if (t >= 1.0) { return here(1u, p); }
    let kind = u32(P(0u).y);
    let dir = vec2<f32>(cos(P(0u).z), sin(P(0u).z));
    let size = P(1u).xy;
    let amount = P(1u).z;
    let seed = P(1u).w;
    let soft = max(P(0u).w, 1e-3);
    let c = size * 0.5;
    // The frame's extent along the direction.
    let span = abs(dir.x) * size.x + abs(dir.y) * size.y;
    switch kind {
        case 1u: { // wipe: the edge crosses the frame along dir
            let q = dot(p - c, dir) + span * 0.5; // 0 → span
            let edge = t * (span + soft) - soft * 0.5;
            let k = clamp((edge - q) / soft + 0.5, 0.0, 1.0);
            return mix(here(0u, p), here(1u, p), k);
        }
        case 2u: { // push: both move along dir
            return pushed(p - dir * span * t, dir * span);
        }
        case 3u: { // slide: the incoming image slides over
            return over(bilinear(1u, p - dir * span * (t - 1.0)), here(0u, p));
        }
        case 4u: { // zoom: the outgoing image flies at the viewer and fades; the incoming one settles from 125 %
            // A cross-mix, so two opaque scenes stay opaque (fading both and laying one over the other let the
            // background through: 75 % alpha at the middle).
            let a = bilinear(0u, c + (p - c) / (1.0 + t));
            let b = bilinear(1u, c + (p - c) / (1.25 - 0.25 * t));
            return mix(a, b, t);
        }
        case 5u: { // iris: a circle opens from the centre
            let r = t * (length(c) + soft) - soft * 0.5;
            let k = clamp((r - length(p - c)) / soft + 0.5, 0.0, 1.0);
            return mix(here(0u, p), here(1u, p), k);
        }
        case 6u: { // clock: a hand sweeps round from angle
            let d = p - c;
            let a = fract((atan2(d.y, d.x) - P(0u).z) / (2.0 * PI));
            // The distance to the hand along the arc, in pixels, softened like a wipe's edge.
            let reach = t * 1.02 - a;
            let k = select(clamp(reach * 2.0 * PI * length(d) / soft + 0.5, 0.0, 1.0), 1.0, t >= 1.0);
            return mix(here(0u, p), here(1u, p), k);
        }
        case 7u: { // crosswarp: the outgoing image shrinks into the incoming one along x
            let u = p / size;
            let x = smoothstep(0.0, 1.0, t * 2.0 + u.x - 1.0);
            let a = bilinear(0u, ((u - 0.5) * (1.0 - x) + 0.5) * size);
            let b = bilinear(1u, ((u - 0.5) * x + 0.5) * size);
            return mix(a, b, x);
        }
        case 8u: { // ripple: rings spread from the centre while it cross-fades
            let d = p - c;
            let l = max(length(d), 1e-3);
            let w = sin(l / 18.0 - t * 30.0) * amount * sin(PI * t);
            let q = p + d / l * w;
            return mix(held(0u, q), held(1u, q), t);
        }
        case 9u: { // pixelize: blocks grow to amount px at the middle, then shrink on the incoming image
            let b = max(1.0, amount * (1.0 - abs(t * 2.0 - 1.0)));
            let i = select(0u, 1u, t >= 0.5);
            // One-pixel blocks are the image itself (a mean over the block would soften it).
            if (b <= 1.0) { return here(i, p); }
            let o = floor(p / b) * b;
            // The block's mean colour (a 4 × 4 grid in it), so thin shapes still tint their blocks.
            var acc = vec4<f32>(0.0);
            for (var y = 0; y < 4; y = y + 1) {
                for (var x = 0; x < 4; x = x + 1) {
                    acc = acc + held(i, o + (vec2<f32>(f32(x), f32(y)) + 0.5) * b / 4.0);
                }
            }
            return acc / 16.0;
        }
        case 10u: { // glitch: rows jump sideways and split their channels; rows cut over at random
            let row = floor(p.y / 14.0);
            let step = floor(t * 24.0);
            let h = th(vec3<f32>(row, step, seed));
            let off = (h - 0.5) * amount * sin(PI * t) * 2.0;
            let k = th(vec3<f32>(row, seed, 3.7));
            let i = select(0u, 1u, k < t * 1.2 - 0.1 || t >= 1.0);
            let r = held(i, p + vec2<f32>(off + amount * 0.15 * sin(PI * t), 0.0));
            let g = held(i, p + vec2<f32>(off, 0.0));
            let bl = held(i, p + vec2<f32>(off - amount * 0.15 * sin(PI * t), 0.0));
            return vec4<f32>(r.r, g.g, bl.b, max(max(r.a, g.a), bl.a));
        }
        case 11u: { // noise: a seeded noise field dissolves into the incoming image
            let n = tnoise(vec3<f32>(p / max(amount, 1.0), seed));
            let w = 0.08;
            let k = clamp((t * (1.0 + w) - n) / w, 0.0, 1.0);
            return mix(here(0u, p), here(1u, p), k);
        }
        case 12u: { // whip: a fast push, smeared along its direction
            let off = dir * span * smoothstep(0.0, 1.0, t);
            let smear = dir * span * 0.18 * sin(PI * t);
            // Average the taps that land on one of the two images: a tap smeared past both (at the frame's edge)
            // would otherwise let the background through.
            var acc = vec4<f32>(0.0);
            var taps = 0.0;
            for (var i = 0; i < 16; i = i + 1) {
                let q = p - off - (f32(i) / 15.0 - 0.5) * smear;
                if (in_frame(q) || in_frame(q + dir * span)) {
                    acc = acc + pushed(q, dir * span);
                    taps = taps + 1.0;
                }
            }
            return acc / max(taps, 1.0);
        }
        case 13u: { // dip: through a colour
            let col = P(2u);
            if (t < 0.5) { return mix(here(0u, p), col, t * 2.0); }
            return mix(col, here(1u, p), t * 2.0 - 1.0);
        }
        case 14u: { // cube: the frame turns like a cube's face, the incoming face coming round from the right
            let u = p.x / size.x;
            let edge = 1.0 - t;
            let persp = 0.25 * sin(PI * t);
            if (u < edge) {
                let ua = u / max(edge, 1e-4);
                let sc = 1.0 - persp * ua;
                // The turning faces darken (their colour), they don't turn translucent.
                let s = bilinear(0u, vec2<f32>(ua * size.x, c.y + (p.y - c.y) / sc));
                return vec4<f32>(s.rgb * (1.0 - 0.3 * t), s.a);
            }
            let ub = (u - edge) / max(t, 1e-4);
            let sc = 1.0 - persp * (1.0 - ub);
            let s = bilinear(1u, vec2<f32>(ub * size.x, c.y + (p.y - c.y) / sc));
            return vec4<f32>(s.rgb * (0.7 + 0.3 * t), s.a);
        }
        default: { return mix(here(0u, p), here(1u, p), t); }
    }
}
"#;
kernel!(transition, TRANSITION_BODY);

/// The `transition` node's kinds, in the kernel's order (moda-motion's `nodes::TRANSITIONS` lists the same).
pub const TRANSITION_KINDS: &[&str] = &[
    "dissolve", "wipe", "push", "slide", "zoom", "iris", "clock", "crosswarp", "ripple", "pixelize", "glitch", "noise",
    "whip", "dip", "cube",
];

/// One Gaussian pass of σ `sigma` output pixels along the unit direction `dir`.
///
/// Up to [`MAX_SIGMA`] it is one dense pass. Beyond it, with `step = ⌈σ / MAX_SIGMA⌉`: a dense pass at σ = step
/// smooths away what a `step`-pixel grid would alias, then a pass with taps `step` pixels apart adds the rest
/// (σ² = step² + rest²). Every pass has at most 300 taps (two per axis up to σ = 10,000), and the result stays a
/// Gaussian.
fn gaussian(exec: &Executor, src: &Image, dir: (f32, f32), sigma: f64) -> R<Image> {
    gaussian_impl(exec, src, dir, sigma, false)
}

fn gaussian_impl(exec: &Executor, src: &Image, dir: (f32, f32), sigma: f64, defocus: bool) -> R<Image> {
    if !(sigma > 0.0) {
        return Ok(src.clone());
    }
    let step = (sigma / MAX_SIGMA).ceil().max(1.0);
    let (img, rest) = if step > 1.0 {
        (gaussian_impl(exec, src, dir, step, defocus)?, (sigma * sigma - step * step).max(0.0).sqrt())
    } else {
        (src.clone(), sigma)
    };
    let s = rest / step; // σ in taps
    let n = (3.0 * s).ceil();
    let reach = n * step + 1.0;
    let d = img.domain.grow(
        (reach * dir.0.abs() as f64).ceil() as u32,
        (reach * dir.1.abs() as f64).ceil() as u32,
    );
    let step = step as f32;
    if defocus && n <= 156.0 {
        let sigma = s as f32;
        let coefficient = -0.5 / (sigma * sigma);
        let mut params = vec![[0.0; 4]; 1 + (n as usize).div_ceil(4)];
        let mut sum = 1.0f32;
        for tap in 1..=n as usize {
            let distance = tap as f32;
            let weight = (coefficient * distance * distance).exp();
            params[1 + (tap - 1) / 4][(tap - 1) % 4] = weight;
            sum += 2.0 * weight;
        }
        params[0] = [dir.0 * step, dir.1 * step, sum, n as f32];
        return exec.run(defocus_pass(), &[&img], &params, d, img.tag());
    }
    exec.run(blur_pass(), &[&img], &[[dir.0 * step, dir.1 * step, s as f32, n as f32]], d, img.tag())
}

/// Axis-aligned depth of field with frame-uniform Gaussian weights prepared once per pass.
/// Uses the same taps, domains, accumulation order and large-radius decomposition as [`blur`].
pub fn defocus(exec: &Executor, src: &Image, sigma: f64) -> R<Image> {
    let horizontal = gaussian_impl(exec, src, (1.0, 0.0), sigma, true)?;
    gaussian_impl(exec, &horizontal, (0.0, 1.0), sigma, true)
}

/// A Gaussian blur of `src` by σ = (sx, sy) output pixels; `angle` (degrees) makes it one directional pass.
pub fn blur(exec: &Executor, src: &Image, sx: f64, sy: f64, angle: Option<f64>) -> R<Image> {
    if let Some(deg) = angle {
        let r = deg.to_radians();
        return gaussian(exec, src, (r.cos() as f32, r.sin() as f32), sx);
    }
    let img = gaussian(exec, src, (1.0, 0.0), sx)?;
    gaussian(exec, &img, (0.0, 1.0), sy)
}

const MERGE_MODES: &[&str] = &[
    "over", "under", "in", "out", "atop", "xor", "plus", "multiply", "screen", "overlay", "darken", "lighten",
    "difference", "max", "min",
];

fn merge_code(mode: &str) -> Option<f32> {
    // Blend modes use their own codes in the kernel's `blend()` (7–11 and the default, difference = 14).
    Some(match mode {
        "over" => 0.0,
        "under" => 1.0,
        "in" => 2.0,
        "out" => 3.0,
        "atop" => 4.0,
        "xor" => 5.0,
        "plus" => 6.0,
        "multiply" => 7.0,
        "screen" => 8.0,
        "overlay" => 9.0,
        "darken" => 10.0,
        "lighten" => 11.0,
        "max" => 12.0,
        "min" => 13.0,
        "difference" => 14.0,
        _ => return None,
    })
}

/// A 2×3 affine `[a, b, c, d, e, f]` (x' = a·x + c·y + e, y' = b·x + d·y + f).
type Affine = [f64; 6];

fn invert_affine(m: &Affine) -> Option<Affine> {
    let det = m[0] * m[3] - m[1] * m[2];
    if det.abs() < 1e-12 || !det.is_finite() {
        return None;
    }
    Some([
        m[3] / det,
        -m[1] / det,
        -m[2] / det,
        m[0] / det,
        (m[2] * m[5] - m[3] * m[4]) / det,
        (m[1] * m[4] - m[0] * m[5]) / det,
    ])
}

fn map_domain(m: &Affine, d: &Domain) -> Domain {
    let pts = [
        (d.x as f64, d.y as f64),
        (d.right() as f64, d.y as f64),
        (d.x as f64, d.bottom() as f64),
        (d.right() as f64, d.bottom() as f64),
    ];
    let mut b = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (x, y) in pts {
        let (u, v) = (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]);
        b = (b.0.min(u), b.1.min(v), b.2.max(u), b.3.max(v));
    }
    Domain::covering(b.0, b.1, b.2, b.3)
}

/// Run pixel node `ty` over its named inputs (`input(name)`), lengths scaled by `scale`.
pub fn run<'i>(
    exec: &Executor,
    args: &Args<'_>,
    ty: &str,
    input: &dyn Fn(&str) -> Result<&'i Image, PixelError>,
    scale: f64,
) -> R<Image> {
    let node = args.node;
    match ty {
        "blur" => {
            let src = input("src")?;
            let (rx, ry) = args.vec2("radius", (8.0, 8.0))?;
            let angle = match args.params.get("angle") {
                None => None,
                Some(_) => Some(args.num("angle", 0.0)?),
            };
            blur(exec, src, rx.max(0.0) * scale, ry.max(0.0) * scale, angle)
        }
        "glow" => {
            let src = input("src")?;
            let radius = args.num("radius", 12.0)?.max(0.0) * scale;
            let intensity = args.num("intensity", 1.0)? as f32;
            let threshold = args.num("threshold", 0.0)? as f32;
            let tint = args.color("color")?.unwrap_or([1.0; 4]);
            let bright = exec.run(
                bright(),
                &[src],
                &[[threshold, 0.1, 0.0, 0.0], [tint[0], tint[1], tint[2], 1.0]],
                src.domain,
                src.tag(),
            )?;
            let soft = blur(exec, &bright, radius, radius, None)?;
            let d = src.domain.union(&soft.domain);
            exec.run(add(), &[src, &soft], &[[intensity, 0.0, 0.0, 0.0]], d, src.tag())
        }
        "displace" => {
            let src = input("src")?;
            let (ax, ay) = args.vec2("amount", (10.0, 10.0))?;
            let (ax, ay) = (ax * scale, ay * scale);
            // The plan passes `map` as an input when the node has one.
            let map = input("map").ok();
            let freq = args.num("frequency", 0.02)? / scale.max(1e-6);
            let oct = args.num("octaves", 2.0)?.round().clamp(1.0, 8.0);
            let seed = args.num("seed", 0.0)?;
            let evolve = args.num("evolve", 0.0)?;
            let d = src.domain.grow(ax.abs().ceil() as u32 + 1, ay.abs().ceil() as u32 + 1);
            let params = [
                [ax as f32, ay as f32, 0.0, if map.is_some() { 1.0 } else { 0.0 }],
                [freq as f32, oct as f32, seed as f32, evolve as f32],
            ];
            match map {
                Some(m) => exec.run(displace(), &[src, m], &params, d, src.tag()),
                None => exec.run(displace(), &[src], &params, d, src.tag()),
            }
        }
        "grade" => {
            let src = input("src")?;
            let hue = args.num("hue", 0.0)? as f32;
            let rgb4 = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
            let params = [
                [
                    args.num("exposure", 0.0)? as f32,
                    args.num("contrast", 1.0)? as f32,
                    args.num("pivot", 0.18)? as f32,
                    args.num("saturation", 1.0)? as f32,
                ],
                rgb4(args.rgb("lift", 0.0)?),
                rgb4(args.rgb("gamma", 1.0)?),
                rgb4(args.rgb("gain", 1.0)?),
                rgb4(args.rgb("offset", 0.0)?),
                rgb4(args.rgb("tint", 1.0)?),
                [hue, 0.0, 0.0, 0.0],
            ];
            exec.run(grade(), &[src], &params, src.domain, src.tag())
        }
        "sprites" => super::sprites::run(exec, args, input, scale),
        "threshold" => {
            let src = input("src")?;
            let level = args.num("level", 0.5)? as f32;
            let soft = args.num("softness", 0.0)?.max(0.0) as f32;
            let luma = match args.string("channel", "alpha")? {
                "alpha" => 0.0,
                "luma" => 1.0,
                other => return Err(PixelError::Invalid(format!("`{node}.channel`: `{other}` is not alpha or luma"))),
            };
            let invert = if args.bool("invert", false)? { 1.0 } else { 0.0 };
            exec.run(threshold(), &[src], &[[level, soft, luma, invert]], src.domain, src.tag())
        }
        "rays" => {
            let src = input("src")?;
            let (w, h) = exec.frame_size();
            let (lx, ly) = match args.params.get("center") {
                None => (w as f64 / 2.0, h as f64 / 2.0),
                Some(_) => {
                    let (x, y) = args.vec2("center", (0.0, 0.0))?;
                    (x * scale, y * scale)
                }
            };
            let length = args.num("length", 0.5)?.clamp(0.0, 1.0) as f32;
            let samples = args.num("samples", 64.0)?.round().clamp(1.0, 512.0) as f32;
            let decay = args.num("decay", 0.97)?.clamp(0.0, 1.0) as f32;
            let threshold = args.num("threshold", 0.0)? as f32;
            let intensity = args.num("intensity", 1.0)?.max(0.0) as f32;
            let tint = args.color("color")?.unwrap_or([1.0; 4]);
            let alone = match args.string("mode", "add")? {
                "add" => 0.0,
                "rays" => 1.0,
                other => return Err(PixelError::Invalid(format!("`{node}.mode`: `{other}` is not add or rays"))),
            };
            // The plan passes `occluders` as an input when the node has one.
            let occluders = input("occluders").ok();
            // Rays reach from the source's pixels away from the light, to the frame's edge.
            let d = src.domain.union(&Domain::frame(w, h));
            let p = [
                [lx as f32, ly as f32, length, samples],
                [decay, threshold, intensity, alone],
                [tint[0], tint[1], tint[2], 1.0],
                [if occluders.is_some() { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
            ];
            match occluders {
                Some(o) => exec.run(rays(), &[src, o], &p, d, src.tag()),
                None => exec.run(rays(), &[src], &p, d, src.tag()),
            }
        }
        "caustics" => {
            let (w, h) = exec.frame_size();
            let src = input("src").ok();
            let cell = args.num("scale", 80.0)?.max(1e-3) * scale;
            let tint = args.color("color")?.unwrap_or([1.0; 4]);
            let p = [
                [
                    (1.0 / cell) as f32,
                    args.num("evolve", 0.0)? as f32,
                    args.num("seed", 0.0)? as f32,
                    args.num("sharpness", 4.0)?.max(0.01) as f32,
                ],
                [
                    args.num("warp", 0.4)?.max(0.0) as f32,
                    args.num("intensity", 1.0)?.max(0.0) as f32,
                    if src.is_some() { 1.0 } else { 0.0 },
                    0.0,
                ],
                [tint[0], tint[1], tint[2], 1.0],
            ];
            match src {
                Some(s) => exec.run(caustics(), &[s], &p, s.domain, ColorTag::LinearSrgb),
                None => exec.run(caustics(), &[], &p, Domain::frame(w, h), ColorTag::LinearSrgb),
            }
        }
        "normals" | "lighting" | "reflect" => {
            let src = input("src")?;
            let height = (args.num("height", 10.0)? * scale) as f32;
            let luma = match args.string("channel", "alpha")? {
                "alpha" => 0.0,
                "luma" => 1.0,
                other => return Err(PixelError::Invalid(format!("`{node}.channel`: `{other}` is not alpha or luma"))),
            };
            // The surface's normals reach one pixel past its height map's edge.
            let d = src.domain.grow(1, 1);
            match ty {
                "normals" => exec.run(normals(), &[src], &[[height, luma, 0.0, 0.0]], d, src.tag()),
                "lighting" => {
                    let specular = match args.string("mode", "diffuse")? {
                        "diffuse" => 0.0,
                        "specular" => 1.0,
                        other => return Err(PixelError::Invalid(format!("`{node}.mode`: `{other}` is not diffuse or specular"))),
                    };
                    let to_light = match args.params.get("at") {
                        // A point light: (x, y) frame px, `z` px above the surface's 0.
                        Some(_) => {
                            let (x, y) = args.vec2("at", (0.0, 0.0))?;
                            let z = args.num("z", 100.0)?;
                            [(x * scale) as f32, (y * scale) as f32, (z * scale) as f32, 1.0]
                        }
                        // A distant light, as SVG's feDistantLight: azimuth clockwise from +x (y down), elevation up
                        // from the surface's plane.
                        None => {
                            let az = args.num("azimuth", 225.0)?.to_radians();
                            let el = args.num("elevation", 45.0)?.to_radians();
                            [(az.cos() * el.cos()) as f32, (az.sin() * el.cos()) as f32, el.sin() as f32, 0.0]
                        }
                    };
                    let col = args.color("color")?.unwrap_or([1.0; 4]);
                    let strength = args.num("strength", 1.0)?.max(0.0) as f32;
                    let shininess = args.num("shininess", 20.0)?.clamp(1.0, 128.0) as f32;
                    let p = [[height, luma, 0.0, 0.0], to_light, [col[0], col[1], col[2], strength], [shininess, specular, 0.0, 0.0]];
                    exec.run(lighting(), &[src], &p, d, src.tag())
                }
                _ => {
                    let env = input("env")?;
                    let offset = match args.string("mode", "sphere")? {
                        "sphere" => 0.0,
                        "offset" => 1.0,
                        other => return Err(PixelError::Invalid(format!("`{node}.mode`: `{other}` is not sphere or offset"))),
                    };
                    let distance = (args.num("distance", 20.0)? * scale) as f32;
                    let b = env.domain;
                    let p = [[height, luma, offset, distance], [b.x as f32, b.y as f32, b.w as f32, b.h as f32]];
                    exec.run(reflect(), &[src, env], &p, d, env.tag())
                }
            }
        }
        "warp" => {
            let src = input("src")?;
            let (w, h) = exec.frame_size();
            let (cx, cy) = match args.params.get("center") {
                None => (w as f64 / 2.0, h as f64 / 2.0),
                Some(_) => {
                    let (x, y) = args.vec2("center", (0.0, 0.0))?;
                    (x * scale, y * scale)
                }
            };
            let radius = args.num("radius", 100.0)?.max(0.0) * scale;
            let kind = match args.string("kind", "bulge")? {
                "bulge" => 0.0,
                "twist" => 1.0,
                other => return Err(PixelError::Invalid(format!("`{node}.kind`: `{other}` is not bulge or twist"))),
            };
            let amount = args.num("amount", 0.5)?;
            // The bulge's range where it never folds (moda-scene `envelope::BULGE_RANGE`).
            let amount = if kind == 0.0 { amount.clamp(-0.95, 1.2) } else { amount };
            // A bulge can push content out to its radius: the domain covers the disc.
            let disc = Domain::covering(cx - radius, cy - radius, cx + radius, cy + radius);
            let d = src.domain.union(&disc);
            exec.run(warp(), &[src], &[[cx as f32, cy as f32, radius as f32, amount as f32], [kind, 0.0, 0.0, 0.0]], d, src.tag())
        }
        "invert" => {
            let src = input("src")?;
            exec.run(invert(), &[src], &[], src.domain, src.tag())
        }
        "chroma_key" => {
            let src = input("src")?;
            let colour = args.color("screen_colour")?.unwrap_or([0.0, 1.0, 0.0, 1.0]);
            let gain = args.num("screen_gain", 100.0)?.clamp(0.0, 1000.0) / 100.0;
            let softness = args.num("screen_softness", 0.0)?.clamp(0.0, 100.0) * scale;
            let despill = if args.bool("despill", true)? { 1.0 } else { 0.0 };
            let channel = if colour[2] > colour[1] { 2 } else { 1 };
            let excess = colour[channel] - colour[(channel + 1) % 3].max(colour[(channel + 2) % 3]);
            if gain == 0.0 || excess <= 1e-6 {
                return Ok(src.clone());
            }
            let uniforms = [[channel as f32, (gain as f32 / excess), despill, 0.0]];
            let matte = exec.run(chroma_matte(), &[src], &uniforms, src.domain, src.tag())?;
            let matte = blur(exec, &matte, softness, softness, None)?;
            exec.run(chroma_key(), &[src, &matte], &uniforms, src.domain, src.tag())
        }
        "matte" => {
            let src = input("src")?;
            let m = input("matte")?;
            let mode = args.string("mode", "alpha")?;
            let (luma, inv) = match mode {
                "alpha" => (0.0, 0.0),
                "luma" => (1.0, 0.0),
                "alpha_inverted" => (0.0, 1.0),
                "luma_inverted" => (1.0, 1.0),
                other => return Err(PixelError::Invalid(format!("`{node}.mode`: no matte mode `{other}`"))),
            };
            let d = if inv > 0.5 { src.domain } else { src.domain.intersect(&m.domain) };
            if d.is_empty() {
                return exec.solid(Domain::new(0, 0, 1, 1), [0.0; 4], src.tag());
            }
            exec.run(matte(), &[src, m], &[[luma, inv, 0.0, 0.0]], d, src.tag())
        }
        "merge" => {
            let a = input("a")?;
            let b = input("b")?;
            let mode = args.string("mode", "over")?;
            let code = merge_code(mode).ok_or_else(|| {
                PixelError::Invalid(format!("`{node}.mode`: `{mode}` is not one of {}", MERGE_MODES.join(", ")))
            })?;
            let mix = args.num("mix", 1.0)?.clamp(0.0, 1.0) as f32;
            let d = match mode {
                "in" => a.domain.intersect(&b.domain),
                "out" => a.domain,
                "atop" => b.domain,
                _ => a.domain.union(&b.domain),
            };
            let d = if mix < 1.0 { d.union(&b.domain) } else { d };
            if d.is_empty() {
                return exec.solid(Domain::new(0, 0, 1, 1), [0.0; 4], a.tag());
            }
            exec.run(merge(), &[a, b], &[[code, mix, 0.0, 0.0]], d, a.tag())
        }
        "transform" => {
            let src = input("src")?;
            let (tx, ty_) = args.vec2("position", (0.0, 0.0))?;
            let (sx, sy) = args.vec2("scale", (1.0, 1.0))?;
            let rot = args.num("rotation", 0.0)?.to_radians();
            let limit_center = exec_frame_center(exec);
            let (ax, ay) = match args.params.get("anchor") {
                None => limit_center,
                Some(_) => {
                    let (x, y) = args.vec2("anchor", (0.0, 0.0))?;
                    (x * scale, y * scale)
                }
            };
            let opacity = args.num("opacity", 1.0)? as f32;
            let (c, s) = (rot.cos(), rot.sin());
            // p' = anchor + position + R·S·(p − anchor)
            let m: Affine = [
                c * sx,
                s * sx,
                -s * sy,
                c * sy,
                0.0,
                0.0,
            ];
            let (ox, oy) = (m[0] * ax + m[2] * ay, m[1] * ax + m[3] * ay);
            let m = [m[0], m[1], m[2], m[3], ax + tx * scale - ox, ay + ty_ * scale - oy];
            // A zero scale (a scale-in from 0, a flip through 0) squashes the image to nothing: transparent.
            let Some(inv) = invert_affine(&m) else {
                return exec.solid(Domain::new(0, 0, 1, 1), [0.0; 4], src.tag());
            };
            let d = map_domain(&m, &src.domain);
            let p = [
                [inv[0] as f32, inv[1] as f32, 0.0, 0.0],
                [inv[2] as f32, inv[3] as f32, 0.0, 0.0],
                [inv[4] as f32, inv[5] as f32, 0.0, 0.0],
                [opacity, 0.0, 0.0, 0.0],
            ];
            exec.run(transform(), &[src], &p, d, src.tag())
        }
        "transition" => {
            let from = input("from")?;
            let to = input("to")?;
            let t = args.num("progress", 0.0)? as f32;
            let name = args.string("kind", "dissolve")?;
            let kind = TRANSITION_KINDS.iter().position(|k| *k == name).ok_or_else(|| {
                PixelError::Invalid(format!("`{node}.kind`: no transition `{name}` ({})", TRANSITION_KINDS.join(", ")))
            })? as f32;
            // Kind defaults: clock starts at 12 o'clock; `amount` is a ripple's, a pixel block's or a glitch's size.
            let angle = args.num("angle", if name == "clock" { -90.0 } else { 0.0 })?.to_radians() as f32;
            let soft = (args.num("softness", if matches!(name, "iris" | "clock") { 4.0 } else { 40.0 })? * scale) as f32;
            let default_amount = match name { "ripple" => 24.0, "pixelize" => 48.0, "glitch" => 60.0, "noise" => 80.0, _ => 0.0 };
            let amount = (args.num("amount", default_amount)? * scale) as f32;
            let seed = args.num("seed", 0.0)? as f32;
            let col = args.color("color")?.unwrap_or([0.0, 0.0, 0.0, 1.0]);
            let col = [col[0] * col[3], col[1] * col[3], col[2] * col[3], col[3]];
            let (w, h) = exec.frame_size();
            let d = from.domain.union(&to.domain);
            // Dip draws its colour, and cube and crosswarp move pixels in from anywhere: over the whole frame.
            let d = if matches!(name, "dip" | "cube" | "crosswarp") { d.union(&Domain::frame(w, h)) } else { d };
            exec.run(
                transition(),
                &[from, to],
                &[[t, kind, angle, soft], [w as f32, h as f32, amount, seed], col],
                d,
                to.tag(),
            )
        }
        other => Err(PixelError::Invalid(format!("`{node}`: `{other}` isn't a pixel node"))),
    }
}

fn exec_frame_center(exec: &Executor) -> (f64, f64) {
    let (w, h) = exec.frame_size();
    (w as f64 / 2.0, h as f64 / 2.0)
}

// ------------------------------------------------------------------------------------------------------------
// Shader nodes

/// A WGSL shader node's code: an agent shader (moda-wgsl, `fn shade(p: vec2<f32>) -> vec4<f32>` returning
/// premultiplied linear light) as a frame pass. `p` is in frame px whatever the render scale; `src(p)` /
/// `input(i, p)` read the node's image inputs; the parameters are the schema's (`moda_wgsl::parse_schema`).
#[derive(Clone, Debug, PartialEq)]
pub struct ShaderCode {
    /// The code object's key (`b3:…`): mixed into the seed.
    pub hash: String,
    pub source: String,
    pub schema: moda_wgsl::Schema,
    /// Image inputs, in order (`src` first when present; a transition's `from`, `to`).
    pub inputs: Vec<String>,
    /// [`moda_wgsl::Placement::Pass`] (a `shader` node), or [`moda_wgsl::Placement::Transition`] (a
    /// `transition#<hash>` node: `from_scene(p)`, `to_scene(p)`, `progress()` from its `progress` parameter).
    pub placement: moda_wgsl::Placement,
}

/// What a shader node made, beyond its image: the first non-finite pixel (frame px) and the time it took (ms).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShaderReport {
    pub nan: Option<(f64, f64)>,
    pub ms: f64,
}

/// A parameter as moda-wgsl's JSON value.
fn param_json(p: &super::exec::Param) -> Option<serde_json::Value> {
    use super::exec::Param;
    Some(match p {
        Param::Num(v) => serde_json::json!(v),
        Param::Vec2(x, y) => serde_json::json!([x, y]),
        Param::Color(c) => serde_json::json!([c[0], c[1], c[2], c[3]]),
        Param::Bool(b) => serde_json::json!(b),
        // Text and a particle list have no shader parameter type: the schema's default applies.
        Param::Str(_) | Param::Floats(_) => return None,
    })
}

/// A colour value (`[r, g, b, a]`, linear, straight) from JSON: a list of numbers, or a CSS hex string (sRGB).
fn json_color(v: &serde_json::Value) -> Option<[f32; 4]> {
    if let Some(a) = v.as_array() {
        let n: Vec<f32> = a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
        return match n.len() {
            3 => Some([n[0], n[1], n[2], 1.0]),
            4 => Some([n[0], n[1], n[2], n[3]]),
            _ => None,
        };
    }
    let h = v.as_str()?.strip_prefix('#')?;
    let byte = |i: usize| u8::from_str_radix(h.get(i..i + 2)?, 16).ok();
    let (r, g, b, a) = match h.len() {
        3 | 4 => {
            let d = |i: usize| u8::from_str_radix(&h[i..i + 1], 16).ok().map(|v| v * 17);
            (d(0)?, d(1)?, d(2)?, if h.len() == 4 { d(3)? } else { 255 })
        }
        6 => (byte(0)?, byte(2)?, byte(4)?, 255),
        8 => (byte(0)?, byte(2)?, byte(4)?, byte(6)?),
        _ => return None,
    };
    let lin = |c: u8| {
        let e = c as f32 / 255.0;
        if e <= 0.04045 { e / 12.92 } else { ((e + 0.055) / 1.055).powf(2.4) }
    };
    Some([lin(r), lin(g), lin(b), a as f32 / 255.0])
}

/// The time budget for an agent pass over `w × h` output px: `per_1080p_ms` per 1920×1080 px, never less than
/// that.
pub fn shader_budget_ms(per_1080p_ms: f64, w: u32, h: u32) -> f64 {
    per_1080p_ms * (w as f64 * h as f64 / (1920.0 * 1080.0)).max(1.0)
}

/// Run a WGSL shader node: its declared parameters from `args` (`seed` mixed with the code hash), its image inputs
/// from `input`, over the union of its inputs' domains (or the frame, for a generator).
#[allow(clippy::too_many_arguments)]
pub fn shader<'i>(
    exec: &Executor,
    args: &Args<'_>,
    code: &ShaderCode,
    input: &dyn Fn(&str) -> Result<&'i Image, PixelError>,
    scale: f64,
    time_s: f64,
    budget_per_1080p_ms: f64,
) -> R<(Image, ShaderReport)> {
    if code.inputs.len() > super::pass::MAX_INPUTS {
        return Err(PixelError::Invalid(format!("a shader takes at most {} image inputs", super::pass::MAX_INPUTS)));
    }
    let module = moda_wgsl::module(&code.source, &code.schema, code.placement);
    let (w, h) = exec.frame_size();
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let (fw, fh) = (w as f64 / scale, h as f64 / scale);
    let seed = args.num("seed", 0.0)?.max(0.0) as u32;
    let host = moda_wgsl::HostSlots {
        frame_size: [fw as f32, fh as f32],
        time: time_s as f32,
        px_scale: scale as f32,
        seed: moda_wgsl::mix_seed(seed, &code.hash),
        progress: args.num("progress", 0.0)?.clamp(0.0, 1.0) as f32,
        bounds: [0.0, 0.0, fw as f32, fh as f32],
        to_out: [scale, 0.0, 0.0, scale, 0.0, 0.0],
    };
    let mut params: Vec<[f32; 4]> = host
        .encode()
        .ok_or_else(|| PixelError::Invalid("a shader at a zero render scale".into()))?
        .to_vec();
    params.extend(moda_wgsl::encode_params(
        &code.schema,
        &|n| args.params.get(n).and_then(param_json),
        &json_color,
    ));
    let ins: Vec<&Image> = code.inputs.iter().map(|n| input(n)).collect::<R<_>>()?;
    let domain = if ins.is_empty() {
        Domain::frame(w, h)
    } else if code.placement == moda_wgsl::Placement::Transition {
        // A transition covers the frame (either scene may show anywhere), like the built-in kinds.
        ins.iter().fold(Domain::frame(w, h), |d, i| d.union(&i.domain))
    } else {
        ins.iter().fold(Domain::EMPTY, |d, i| d.union(&i.domain))
    };
    let budget = shader_budget_ms(budget_per_1080p_ms, domain.w, domain.h);
    let (img, nan, ms) = exec.run_agent(args.node, &module.wgsl, &ins, &params, domain, ColorTag::LinearSrgb, budget)?;
    let nan = nan.map(|(x, y)| (x as f64 / scale, y as f64 / scale));
    Ok((img, ShaderReport { nan, ms }))
}

fn intern(s: String) -> &'static str {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SET: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut set = SET.get_or_init(Default::default).lock().expect("intern lock");
    if let Some(s) = set.get(s.as_str()) {
        return s;
    }
    let leaked: &'static str = Box::leak(s.into_boxed_str());
    set.insert(leaked);
    leaked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_helpers() {
        let m: Affine = [2.0, 0.0, 0.0, 3.0, 10.0, -5.0];
        let i = invert_affine(&m).unwrap();
        assert_eq!(i, [0.5, 0.0, 0.0, 1.0 / 3.0, -5.0, 5.0 / 3.0]);
        assert_eq!(map_domain(&m, &Domain::new(0, 0, 4, 2)), Domain::new(10, -5, 8, 6));
        assert!(invert_affine(&[0.0; 6]).is_none());
    }

    #[test]
    fn colors_parse_from_hex_and_lists() {
        assert_eq!(json_color(&serde_json::json!("#ffffff")), Some([1.0, 1.0, 1.0, 1.0]));
        assert_eq!(json_color(&serde_json::json!("#0000")), Some([0.0, 0.0, 0.0, 0.0]));
        assert_eq!(json_color(&serde_json::json!([0.5, 0.25, 0.0])), Some([0.5, 0.25, 0.0, 1.0]));
        assert_eq!(json_color(&serde_json::json!("red")), None);
        assert_eq!(shader_budget_ms(50.0, 100, 100), 50.0);
        assert_eq!(shader_budget_ms(50.0, 3840, 2160), 200.0);
    }
}
