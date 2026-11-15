//! The pixel graph's built-in kernels (WGSL bodies for [`super::pass`]) and their parameter packing.

use super::color::Conversion;
use super::pass::Kernel;

/// Transfers shared by every kernel that decodes or encodes (mirrored for negative values, like the CPU
/// reference in [`super::color`]).
pub const TRANSFERS: &str = r#"
const PQ_M1: f32 = 0.1593017578125;
const PQ_M2: f32 = 78.84375;
const PQ_C1: f32 = 0.8359375;
const PQ_C2: f32 = 18.8515625;
const PQ_C3: f32 = 18.6875;
const PQ_WHITE: f32 = 49.26108374384236; // 10000 / 203
const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 0.28466892;
const HLG_C: f32 = 0.55991073;
const HLG_WHITE: f32 = 0.2649565; // HLG inverse OETF of 0.75

fn decode1(id: u32, v: f32) -> f32 {
    let s = sign(v);
    let c = abs(v);
    switch id {
        case 1u: {
            if (c <= 0.04045) { return s * c / 12.92; }
            return s * pow((c + 0.055) / 1.055, 2.4);
        }
        case 2u: {
            let p = pow(min(c, 1.0), 1.0 / PQ_M2);
            return s * pow(max(p - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * p), 1.0 / PQ_M1) * PQ_WHITE;
        }
        case 3u: {
            if (c <= 0.5) { return s * c * c / 3.0 / HLG_WHITE; }
            return s * (exp((c - HLG_C) / HLG_A) + HLG_B) / 12.0 / HLG_WHITE;
        }
        default: { return v; }
    }
}

fn encode1(id: u32, v: f32) -> f32 {
    let s = sign(v);
    let l = abs(v);
    switch id {
        case 1u: {
            if (l <= 0.0031308) { return s * l * 12.92; }
            return s * (1.055 * pow(l, 1.0 / 2.4) - 0.055);
        }
        case 2u: {
            let p = pow(l / PQ_WHITE, PQ_M1);
            return s * pow((PQ_C1 + PQ_C2 * p) / (1.0 + PQ_C3 * p), PQ_M2);
        }
        case 3u: {
            let e = l * HLG_WHITE;
            if (e <= 1.0 / 12.0) { return s * sqrt(3.0 * e); }
            return s * (HLG_A * log(12.0 * e - HLG_B) + HLG_C);
        }
        default: { return v; }
    }
}

fn decode3(id: u32, c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(decode1(id, c.r), decode1(id, c.g), decode1(id, c.b));
}

fn encode3(id: u32, c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(encode1(id, c.r), encode1(id, c.g), encode1(id, c.b));
}
"#;

/// `convert`: P(0).xy = decode / encode transfer ids, P(1..=3).xyz = the matrix's rows. Premultiplied in and out.
pub const CONVERT_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let c = here(0u, p);
    if (c.a <= 0.0) { return vec4<f32>(0.0, 0.0, 0.0, max(c.a, 0.0)); }
    let lin = decode3(u32(P(0u).x), c.rgb / c.a);
    let m = vec3<f32>(dot(P(1u).xyz, lin), dot(P(2u).xyz, lin), dot(P(3u).xyz, lin));
    return vec4<f32>(encode3(u32(P(0u).y), m) * c.a, c.a);
}
"#;

fn with_transfers(body: &str) -> String {
    format!("{TRANSFERS}\n{body}")
}

/// Kernel sources are built once per process.
macro_rules! kernel_source {
    ($name:ident, $body:expr) => {
        pub fn $name() -> Kernel<'static> {
            static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
            Kernel {
                name: stringify!($name),
                body: SRC.get_or_init(|| with_transfers($body)).as_str(),
            }
        }
    };
}

kernel_source!(convert, CONVERT_BODY);

/// `solid`: P(0), premultiplied, everywhere.
pub const SOLID_BODY: &str = "fn kernel(p: vec2<f32>) -> vec4<f32> { return P(0u); }";
kernel_source!(solid, SOLID_BODY);

/// A magenta checker (8 output px squares): an agent shader that failed in a preview.
pub const MAGENTA_BODY: &str = "fn kernel(p: vec2<f32>) -> vec4<f32> {
    let k = (i32(floor(p.x / 8.0)) + i32(floor(p.y / 8.0))) & 1;
    return select(vec4<f32>(1.0, 0.0, 1.0, 1.0), vec4<f32>(0.35, 0.0, 0.35, 1.0), k == 1);
}";
kernel_source!(magenta, MAGENTA_BODY);

/// `copy`: input 0 re-cut to the output domain.
pub const COPY_BODY: &str = "fn kernel(p: vec2<f32>) -> vec4<f32> { return here(0u, p); }";
kernel_source!(copy, COPY_BODY);

/// `flatten`: input 0 over the premultiplied background P(0).
pub const FLATTEN_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let c = here(0u, p);
    return c + P(0u) * (1.0 - c.a);
}
"#;
kernel_source!(flatten, FLATTEN_BODY);

/// `channels`: P(0) picks each output channel: 0–3 = r g b a, 4 = 0, 5 = 1, 6 = Rec.709 luma. Works on
/// straight colour, and premultiplies the result by its new alpha.
pub const CHANNELS_BODY: &str = r#"
fn pick(c: vec4<f32>, k: f32) -> f32 {
    let i = u32(k);
    switch i {
        case 0u: { return c.r; }
        case 1u: { return c.g; }
        case 2u: { return c.b; }
        case 3u: { return c.a; }
        case 4u: { return 0.0; }
        case 5u: { return 1.0; }
        default: { return dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)); }
    }
}
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let pm = here(0u, p);
    let c = vec4<f32>(unpremul(pm), pm.a);
    let s = P(0u);
    let a = pick(c, s.w);
    return vec4<f32>(pick(c, s.x) * a, pick(c, s.y) * a, pick(c, s.z) * a, a);
}
"#;
kernel_source!(channels, CHANNELS_BODY);

/// `accumulate`: P(0).x = mode (0 add, 1 over: earlier inputs on top, 2 max), P(0).y = input count, P(1) = weights.
pub const ACCUMULATE_BODY: &str = r#"
fn kernel(p: vec2<f32>) -> vec4<f32> {
    let mode = u32(P(0u).x);
    let n = u32(P(0u).y);
    let w = P(1u);
    var c = here(0u, p) * w.x;
    for (var i = 1u; i < 4u; i = i + 1u) {
        if (i >= n) { break; }
        let x = here(i, p) * w[i];
        switch mode {
            case 0u: { c = c + x; }
            case 1u: { c = c + x * (1.0 - c.a); }
            default: { c = max(c, x); }
        }
    }
    return c;
}
"#;
kernel_source!(accumulate, ACCUMULATE_BODY);

/// `project`: input 0 seen through a plane homography (2.5D layers, motion-model.md §12.6). P(0..=2).xyz = the
/// rows of the inverse homography, output frame pixel → input pixel; its third component is 1 / depth, so a pixel
/// whose ray meets the plane behind the camera (≤ 0) stays transparent. Four rotated-grid samples per pixel (the
/// host captures at twice the projection's density or more).
pub const PROJECT_BODY: &str = r#"
fn tap(q: vec2<f32>) -> vec4<f32> {
    let h = vec3<f32>(dot(P(0u).xyz, vec3<f32>(q, 1.0)), dot(P(1u).xyz, vec3<f32>(q, 1.0)), dot(P(2u).xyz, vec3<f32>(q, 1.0)));
    if (h.z <= 0.0) { return vec4<f32>(0.0); }
    return bilinear(0u, h.xy / h.z);
}
fn kernel(p: vec2<f32>) -> vec4<f32> {
    // Within a quarter pixel of the centre: with a capture at twice the output's density, each tap's bilinear
    // footprint stays inside the pixel, so a plane at rest resamples to exactly its flat pixels.
    return 0.25 * (tap(p + vec2<f32>(-0.0833, -0.25)) + tap(p + vec2<f32>(0.25, -0.0833))
        + tap(p + vec2<f32>(0.0833, 0.25)) + tap(p + vec2<f32>(-0.25, 0.0833)));
}
"#;
kernel_source!(project, PROJECT_BODY);

/// The `convert` kernel's parameter slots.
pub fn convert_params(c: &Conversion) -> Vec<[f32; 4]> {
    let row = |i: usize| [c.matrix[i][0] as f32, c.matrix[i][1] as f32, c.matrix[i][2] as f32, 0.0];
    vec![[c.decode.id() as f32, c.encode.id() as f32, 0.0, 0.0], row(0), row(1), row(2)]
}

/// Whether a conversion changes anything (same tag → no pass).
pub fn is_identity(c: &Conversion) -> bool {
    c.decode == c.encode && c.matrix == super::color::IDENTITY
}

/// Convert an image's colour group to `to` (one pass; none when the tags already agree). Other groups pass through.
pub fn convert_image(
    passes: &super::Passes,
    pool: &super::TexturePool,
    img: &super::Image,
    to: super::ColorTag,
) -> Result<super::Image, super::PixelError> {
    let color = img.color();
    if color.tag == to {
        return Ok(img.clone());
    }
    let c = super::color::conversion(color.tag, to).ok_or_else(|| {
        super::PixelError::Invalid(format!("can't convert {} to {} (data is never converted)", color.tag.name(), to.name()))
    })?;
    let mut out = color.clone();
    out.tag = to;
    if !is_identity(&c) {
        out.texture = passes.run(
            pool,
            convert(),
            &[super::PassInput { texture: &color.texture, domain: img.domain }],
            &convert_params(&c),
            img.domain,
            color.storage(),
        )?;
    }
    Ok(img.clone().with_group(out))
}
