// std v1: the agent-shader prelude (docs/design/effects-code-first.md §4.3). Every agent shader is compiled with it;
// its version is part of the shader's identity (STD_VERSION). Shaders call these as `std::name(…)` (compiled with
// the `::` as two underscores: the same length, so line and column numbers in errors are the shader's own).
//
// Determinism: the only randomness is the integer hash below (lowbias32), seeded explicitly. Integer maths is exact
// on every GPU, so the same seed gives the same pattern everywhere; no `fract(sin(…))`.
//
// Colours are linear light. Shaders return premultiplied RGBA; the colour helpers here take straight colours.

// ---- Hashes ---------------------------------------------------------------------------------------------------

// lowbias32 (Chris Wellons): a well-mixed 32-bit permutation, bit-exact on every device.
fn std__hash(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}

// The top 24 bits of a hash as a number in [0, 1).
fn std__unit(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

// A hash of an integer cell and a seed.
fn std__hash_cell(c: vec2<i32>, seed: u32) -> u32 {
    return std__hash(bitcast<u32>(c.x) ^ std__hash(bitcast<u32>(c.y) ^ std__hash(seed)));
}

// A number in [0, 1) for the integer cell containing p.
fn std__hash21(p: vec2<f32>, seed: u32) -> f32 {
    return std__unit(std__hash_cell(vec2<i32>(floor(p)), seed));
}

// Two numbers in [0, 1) for the integer cell containing p.
fn std__hash22(p: vec2<f32>, seed: u32) -> vec2<f32> {
    let h = std__hash_cell(vec2<i32>(floor(p)), seed);
    return vec2<f32>(std__unit(h), std__unit(std__hash(h ^ 0x9e3779b9u)));
}

// A number in [0, 1) for the integer cell containing p (3D: x, y and, say, a frame number).
fn std__hash31(p: vec3<f32>, seed: u32) -> f32 {
    let c = vec3<i32>(floor(p));
    return std__unit(std__hash(bitcast<u32>(c.z) ^ std__hash_cell(c.xy, seed)));
}

// ---- Noise ----------------------------------------------------------------------------------------------------
// p is in noise cells: divide a pixel position by the feature size you want (p / 40.0 for ~40 px features).

// Value noise in [0, 1]: hashed values at cell corners, quintic interpolation.
fn std__value(p: vec2<f32>, seed: u32) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = std__unit(std__hash_cell(i, seed));
    let b = std__unit(std__hash_cell(i + vec2<i32>(1, 0), seed));
    let c = std__unit(std__hash_cell(i + vec2<i32>(0, 1), seed));
    let d = std__unit(std__hash_cell(i + vec2<i32>(1, 1), seed));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// A unit gradient for a cell: one of 16 directions, picked by hash.
fn std__grad(c: vec2<i32>, seed: u32) -> vec2<f32> {
    let a = f32(std__hash_cell(c, seed) & 15u) * 0.39269908; // 2π / 16
    return vec2<f32>(cos(a), sin(a));
}

// 2D simplex noise in about [-1, 1].
fn std__simplex(p: vec2<f32>, seed: u32) -> f32 {
    let k1 = 0.36602540378; // (√3 − 1) / 2
    let k2 = 0.21132486540; // (3 − √3) / 6
    let s = floor(p + (p.x + p.y) * k1);
    let x0 = p - s + (s.x + s.y) * k2;
    let o = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), x0.x > x0.y);
    let x1 = x0 - o + k2;
    let x2 = x0 - 1.0 + 2.0 * k2;
    let i = vec2<i32>(s);
    let h = max(vec3<f32>(0.5) - vec3<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2)), vec3<f32>(0.0));
    let n = vec3<f32>(
        dot(std__grad(i, seed), x0),
        dot(std__grad(i + vec2<i32>(o), seed), x1),
        dot(std__grad(i + vec2<i32>(1, 1), seed), x2),
    );
    let h4 = h * h * h * h;
    return 70.0 * dot(h4, n);
}

// Worley (cellular) noise: (F1, F2), the distances to the nearest and second-nearest feature point, in cells.
fn std__worley(p: vec2<f32>, seed: u32) -> vec2<f32> {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    var d = vec2<f32>(8.0, 8.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = i + vec2<i32>(x, y);
            let h = std__hash_cell(c, seed);
            let pt = vec2<f32>(f32(x), f32(y)) + vec2<f32>(std__unit(h), std__unit(std__hash(h ^ 0x9e3779b9u)));
            let r = length(pt - f);
            if (r < d.x) {
                d = vec2<f32>(r, d.x);
            } else if (r < d.y) {
                d.y = r;
            }
        }
    }
    return d;
}

// Fractal sum of simplex noise: `octaves` (1–12) layers, each at twice the frequency and half the amplitude,
// normalised to about [-1, 1]. Each octave has its own seed.
fn std__fbm(p: vec2<f32>, octaves: u32, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var norm = 0.0;
    var q = p;
    let n = clamp(octaves, 1u, 12u);
    for (var o = 0u; o < 12u; o++) {
        if (o >= n) {
            break;
        }
        sum += amp * std__simplex(q, std__hash(seed + o));
        norm += amp;
        amp *= 0.5;
        q = q * 2.0;
    }
    return sum / norm;
}

// The curl of simplex noise at p: a divergence-free flow direction (not normalised).
fn std__curl(p: vec2<f32>, seed: u32) -> vec2<f32> {
    let e = 0.01;
    let dx = std__simplex(p + vec2<f32>(e, 0.0), seed) - std__simplex(p - vec2<f32>(e, 0.0), seed);
    let dy = std__simplex(p + vec2<f32>(0.0, e), seed) - std__simplex(p - vec2<f32>(0.0, e), seed);
    return vec2<f32>(dy, -dx) / (2.0 * e);
}

// ---- 2D signed distance -----------------------------------------------------------------------------------------
// Negative inside, positive outside, in the units of p.

fn std__sd_circle(p: vec2<f32>, r: f32) -> f32 {
    return length(p) - r;
}

// A box centred on the origin with half-size `half`.
fn std__sd_box(p: vec2<f32>, half: vec2<f32>) -> f32 {
    let d = abs(p) - half;
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}

fn std__sd_round_box(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    return std__sd_box(p, half - vec2<f32>(r)) - r;
}

// The segment from a to b.
fn std__sd_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0);
    return length(pa - ba * h);
}

// A ring of radius r and thickness w.
fn std__sd_ring(p: vec2<f32>, r: f32, w: f32) -> f32 {
    return abs(length(p) - r) - w * 0.5;
}

fn std__op_union(a: f32, b: f32) -> f32 {
    return min(a, b);
}

fn std__op_subtract(a: f32, b: f32) -> f32 {
    return max(a, -b);
}

fn std__op_intersect(a: f32, b: f32) -> f32 {
    return max(a, b);
}

// A union blended over distance k.
fn std__op_smooth_union(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Antialiased coverage (0–1) of a distance in pixels: 1 inside, 0 outside, a one-pixel ramp at the edge.
fn std__fill(d: f32) -> f32 {
    return clamp(0.5 - d, 0.0, 1.0);
}

// ---- Geometry ------------------------------------------------------------------------------------------------

// p rotated by `angle` radians about the origin.
fn std__rotate(p: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

// ---- Colour --------------------------------------------------------------------------------------------------

// Linear sRGB → OKLab (Björn Ottosson).
fn std__oklab(c: vec3<f32>) -> vec3<f32> {
    let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    let l_ = sign(l) * pow(abs(l), 1.0 / 3.0);
    let m_ = sign(m) * pow(abs(m), 1.0 / 3.0);
    let s_ = sign(s) * pow(abs(s), 1.0 / 3.0);
    return vec3<f32>(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    );
}

// OKLab → linear sRGB.
fn std__from_oklab(c: vec3<f32>) -> vec3<f32> {
    let l_ = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
    let m_ = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
    let s_ = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    return vec3<f32>(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    );
}

// Straight colours a → b at t, interpolated in OKLab (alpha linearly).
fn std__mix_oklab(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    let m = mix(std__oklab(a.rgb), std__oklab(b.rgb), t);
    return vec4<f32>(std__from_oklab(m), mix(a.a, b.a, t));
}

// A straight colour → premultiplied.
fn std__premul(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(c.rgb * c.a, c.a);
}

// A premultiplied colour → straight.
fn std__unpremul(c: vec4<f32>) -> vec4<f32> {
    if (c.a <= 0.0) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(c.rgb / c.a, c.a);
}

// Relative luminance of a linear colour.
fn std__luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// ---- Merge (the motion graph's merge operators, on premultiplied colours) ----------------------------------------

const std__OVER: u32 = 0u;
const std__UNDER: u32 = 1u;
const std__IN: u32 = 2u;
const std__OUT: u32 = 3u;
const std__ATOP: u32 = 4u;
const std__XOR: u32 = 5u;
const std__PLUS: u32 = 6u;
const std__MULTIPLY: u32 = 7u;
const std__SCREEN: u32 = 8u;
const std__OVERLAY: u32 = 9u;
const std__DARKEN: u32 = 10u;
const std__LIGHTEN: u32 = 11u;
const std__MAX: u32 = 12u;
const std__MIN: u32 = 13u;
const std__DIFFERENCE: u32 = 14u;

fn std__blend_fn(m: u32, cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    switch m {
        case 7u: { return cb * cs; }
        case 8u: { return cb + cs - cb * cs; }
        case 9u: { return select(1.0 - 2.0 * (1.0 - cb) * (1.0 - cs), 2.0 * cb * cs, cb <= vec3<f32>(0.5)); }
        case 10u: { return min(cb, cs); }
        case 11u: { return max(cb, cs); }
        default: { return abs(cb - cs); }
    }
}

// `a` (on top) merged onto `b` with operator `m` (std::OVER … std::DIFFERENCE).
fn std__merge(a: vec4<f32>, b: vec4<f32>, m: u32) -> vec4<f32> {
    switch m {
        case 0u: { return a + b * (1.0 - a.a); }
        case 1u: { return b + a * (1.0 - b.a); }
        case 2u: { return a * b.a; }
        case 3u: { return a * (1.0 - b.a); }
        case 4u: { return a * b.a + b * (1.0 - a.a); }
        case 5u: { return a * (1.0 - b.a) + b * (1.0 - a.a); }
        case 6u: { let r = a + b; return vec4<f32>(r.rgb, min(r.a, 1.0)); }
        case 12u: { return max(a, b); }
        case 13u: { return min(a, b); }
        default: {
            let cs = std__unpremul(a).rgb;
            let cb = std__unpremul(b).rgb;
            let rgb = a.rgb * (1.0 - b.a) + b.rgb * (1.0 - a.a) + a.a * b.a * std__blend_fn(m, cb, cs);
            return vec4<f32>(rgb, a.a + b.a * (1.0 - a.a));
        }
    }
}

// ---- Debugging -------------------------------------------------------------------------------------------------

// A number as an opaque false colour: below 0 blue, 0–1 dark → yellow, above 1 red.
fn std__debug(v: f32) -> vec4<f32> {
    if (v < 0.0) {
        return vec4<f32>(0.0, 0.1, min(-v, 1.0), 1.0);
    }
    if (v > 1.0) {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(v, v * v, 0.15 * (1.0 - v), 1.0);
}
