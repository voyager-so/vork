//! Agent-written WGSL shaders (docs/design/effects-code-first.md §4.3).
//!
//! An agent writes one function, `fn shade(p: vec2<f32>) -> vec4<f32>`, returning premultiplied linear RGBA at `p`.
//! The same source runs in four placements ([`Placement`]); each gives `p` its own space and its own accessors:
//!
//! | Placement | `p` | Accessors beyond the common ones |
//! |---|---|---|
//! | [`Placement::Pass`] (motion graph `shader` node) | frame px | `src(p)`, `input(i, p)`: the node's image inputs |
//! | [`Placement::Fill`] (a shape's or text's paint) | layer px (its box's top-left is 0, 0) | none: the renderer clips and antialiases the result to the shape or the glyphs, as for any fill |
//! | [`Placement::Effect`] (a layer effect) | layer px | `src(p)`: the layer as drawn (its effects before this one applied). `backdrop(p)` (what's behind it) is planned, not built |
//! | [`Placement::Transition`] | frame px | `from_scene(p)`, `to_scene(p)`, `progress()` (0–1) |
//!
//! Common accessors: `frame_size()`, `time()` (seconds), `px_scale()` (output pixels per `p` unit), `uv(p)` (0–1
//! over `bounds()`), `bounds()` (x, y, w, h of the placement in `p` units), `seed() -> u32`, every declared parameter
//! as a function (`amount()`, `tint()`), and the std prelude (`std::fbm`, `std::worley`, `std::sd_box`,
//! `std::mix_oklab`, `std::merge` …; [`std_source`]).
//!
//! `p` never depends on the render scale, so a half-size preview and a full-size export draw the same picture.
//!
//! The module a placement compiles to is [`module`] (a fill on text: [`fill_text_module`]); [`check`] runs naga and
//! the rules (determinism, loop bounds, no bindings) on exactly those modules, without a GPU, and reports errors at
//! the line and column of the agent's source.
//!
//! Hosts: a frame pass, layer effect and transition run as a pixel pass ([`PASS_PRELUDE`]: parameters in header
//! slots from [`RESERVED_SLOTS`]); a fill is the procedural-fill pipeline's `shadeMain` (the instanced shape module
//! and the vector-text module, `src/hosts/`, copied from moda-shaders' production modules and pinned to them by a
//! test), with parameters in the instance's `shaderParams[2..16]` then `colorData[0..16]`.

mod check;
mod loops;

pub use check::{check, Diag, Report};

/// The prelude's version: part of every shader's identity (a cache key and a document's recorded behaviour).
pub const STD_VERSION: u32 = 1;

/// The std prelude, as compiled (`std__name`).
pub const STD_WGSL: &str = include_str!("std.wgsl");

/// The std prelude as agents call it (`std::name`): what `api shader.std` prints.
pub fn std_source() -> String {
    STD_WGSL.replace("std__", "std::")
}

/// Image inputs a pass reads.
pub const MAX_INPUTS: usize = 4;
/// `vec4<f32>` parameter slots in a pass's header.
pub const PARAM_SLOTS: usize = 40;
/// Slots an agent shader's host fills before its declared parameters (see [`HostSlots`]).
pub const RESERVED_SLOTS: usize = 7;
/// Slots an agent shader's declared parameters may take (a `colors` parameter takes one per colour plus one).
pub const MAX_PARAM_SLOTS: usize = 30;
/// Longest shader source.
pub const MAX_SOURCE_BYTES: usize = 64 * 1024;
/// Most colours in one `colors` parameter.
pub const MAX_COLORS: usize = 8;
/// Longest loop an agent shader may write (iterations of one loop, from its constant bound).
pub const MAX_LOOP_ITERATIONS: u64 = 1024;

/// The WGSL every pixel kernel is compiled with (moda-engine's pixel passes): the header (output origin and size,
/// input rects, parameter slots), four input textures, `P(k)`, `px(i, q)`, `bilinear(i, p)`, `here(i, p)`,
/// `unpremul(c)` and the fullscreen vertex shader. Kernels add `fn kernel(p) -> vec4<f32>` and [`PASS_FRAGMENT`].
pub const PASS_PRELUDE: &str = r#"
struct Header {
    out_origin: vec2<f32>,
    out_size: vec2<f32>,
    // Each input's domain: x, y, w, h in output pixels (w = 0: no input).
    rects: array<vec4<f32>, 4>,
    params: array<vec4<f32>, 40>,
};
@group(0) @binding(0) var<uniform> H: Header;
@group(0) @binding(1) var t0: texture_2d<f32>;
@group(0) @binding(2) var t1: texture_2d<f32>;
@group(0) @binding(3) var t2: texture_2d<f32>;
@group(0) @binding(4) var t3: texture_2d<f32>;

fn P(k: u32) -> vec4<f32> { return H.params[k]; }

fn load_raw(i: u32, c: vec2<i32>) -> vec4<f32> {
    switch i {
        case 0u: { return textureLoad(t0, c, 0); }
        case 1u: { return textureLoad(t1, c, 0); }
        case 2u: { return textureLoad(t2, c, 0); }
        default: { return textureLoad(t3, c, 0); }
    }
}

// Input i's texel at integer output pixel q; transparent outside its domain.
fn px(i: u32, q: vec2<i32>) -> vec4<f32> {
    let r = H.rects[i];
    let c = q - vec2<i32>(r.xy);
    if (r.z <= 0.0 || c.x < 0 || c.y < 0 || c.x >= i32(r.z) || c.y >= i32(r.w)) {
        return vec4<f32>(0.0);
    }
    return load_raw(i, c);
}

// Input i at output position p (pixel centres at +0.5), bilinear, transparent outside its domain.
fn bilinear(i: u32, p: vec2<f32>) -> vec4<f32> {
    let s = p - vec2<f32>(0.5);
    let f = floor(s);
    let w = s - f;
    let q = vec2<i32>(f);
    // On a texel centre (an axis-aligned blur's taps, a whole-pixel offset) the weights are 0: its texel, read once.
    if (w.x == 0.0 && w.y == 0.0) {
        return px(i, q);
    }
    let a = mix(px(i, q), px(i, q + vec2<i32>(1, 0)), w.x);
    let b = mix(px(i, q + vec2<i32>(0, 1)), px(i, q + vec2<i32>(1, 1)), w.x);
    return mix(a, b, w.y);
}

// Input i at the output pixel's own position.
fn here(i: u32, p: vec2<f32>) -> vec4<f32> {
    return px(i, vec2<i32>(floor(p)));
}

fn unpremul(c: vec4<f32>) -> vec3<f32> {
    if (c.a <= 0.0) { return vec3<f32>(0.0); }
    return c.rgb / c.a;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}
"#;

/// The fragment entry of a built-in kernel.
pub const PASS_FRAGMENT: &str = r#"
@fragment
fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
    return kernel(H.out_origin + fc.xy);
}
"#;

/// What an agent shader's host adds to the pass ABI: a storage buffer where the first non-finite result is recorded
/// (`[count, x, y, 0]`, output pixels), and the guard that draws it as a magenta checker instead.
const AGENT_HOST: &str = r#"
@group(1) @binding(0) var<storage, read_write> moda_nan: array<atomic<u32>, 4>;

fn moda_guard(c: vec4<f32>, q: vec2<f32>) -> vec4<f32> {
    let bad = (bitcast<vec4<u32>>(c) & vec4<u32>(0x7f800000u)) == vec4<u32>(0x7f800000u);
    if (any(bad)) {
        if (atomicAdd(&moda_nan[0], 1u) == 0u) {
            atomicStore(&moda_nan[1], u32(max(q.x, 0.0)));
            atomicStore(&moda_nan[2], u32(max(q.y, 0.0)));
        }
        let k = (i32(floor(q.x / 8.0)) + i32(floor(q.y / 8.0))) & 1;
        return select(vec4<f32>(1.0, 0.0, 1.0, 1.0), vec4<f32>(0.35, 0.0, 0.35, 1.0), k == 1);
    }
    return c;
}

// p (the placement's space) → output pixels, and back: slots 5–6 and 3–4.
fn moda_to_out(p: vec2<f32>) -> vec2<f32> {
    let m = P(5u);
    let t = P(6u).xy;
    return vec2<f32>(m.x * p.x + m.z * p.y + t.x, m.y * p.x + m.w * p.y + t.y);
}
fn moda_from_out(q: vec2<f32>) -> vec2<f32> {
    let m = P(3u);
    let t = P(4u).xy;
    return vec2<f32>(m.x * q.x + m.z * q.y + t.x, m.y * q.x + m.w * q.y + t.y);
}

fn frame_size() -> vec2<f32> { return P(0u).xy; }
fn time() -> f32 { return P(0u).z; }
fn px_scale() -> f32 { return P(0u).w; }
fn seed() -> u32 { return bitcast<u32>(P(1u).x); }
fn bounds() -> vec4<f32> { return P(2u); }
fn uv(p: vec2<f32>) -> vec2<f32> { return (p - P(2u).xy) / max(P(2u).zw, vec2<f32>(1e-6)); }
fn moda_param(k: u32) -> vec4<f32> { return P(7u + k); }
"#;

/// Where a shader runs; see the crate docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    Pass,
    Fill,
    Effect,
    Transition,
}

impl Placement {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "pass" => Self::Pass,
            "fill" => Self::Fill,
            "effect" => Self::Effect,
            "transition" => Self::Transition,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fill => "fill",
            Self::Effect => "effect",
            Self::Transition => "transition",
        }
    }

    /// The placement's own accessors, and its fragment entry.
    fn accessors(self) -> (&'static str, &'static str) {
        match self {
            Self::Pass => (
                "fn input(i: u32, p: vec2<f32>) -> vec4<f32> { return bilinear(i, moda_to_out(p)); }\n\
                 fn src(p: vec2<f32>) -> vec4<f32> { return bilinear(0u, moda_to_out(p)); }\n",
                "@fragment\nfn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {\n\
                 \x20   let q = H.out_origin + fc.xy;\n    return moda_guard(shade(moda_from_out(q)), q);\n}\n",
            ),
            Self::Fill => ("", ""),
            // A layer capture is the vector renderer's 8-bit, sRGB-encoded premultiplied colour: decoded to linear
            // for the shader, its result encoded back.
            Self::Effect => (
                "fn moda_decode(c: vec4<f32>) -> vec4<f32> {\n\
                 \x20   if (c.a <= 0.0) { return vec4<f32>(0.0); }\n\
                 \x20   let e = clamp(c.rgb / c.a, vec3<f32>(0.0), vec3<f32>(1.0));\n\
                 \x20   let l = select(pow((e + 0.055) / 1.055, vec3<f32>(2.4)), e / 12.92, e <= vec3<f32>(0.04045));\n\
                 \x20   return vec4<f32>(l * c.a, c.a);\n}\n\
                 fn moda_encode(c: vec4<f32>) -> vec4<f32> {\n\
                 \x20   let a = clamp(c.a, 0.0, 1.0);\n\
                 \x20   if (a <= 0.0) { return vec4<f32>(0.0); }\n\
                 \x20   let l = clamp(c.rgb / c.a, vec3<f32>(0.0), vec3<f32>(1.0));\n\
                 \x20   let e = select(1.055 * pow(l, vec3<f32>(1.0 / 2.4)) - 0.055, l * 12.92, l <= vec3<f32>(0.0031308));\n\
                 \x20   return vec4<f32>(e * a, a);\n}\n\
                 fn src(p: vec2<f32>) -> vec4<f32> { return moda_decode(bilinear(0u, moda_to_out(p))); }\n",
                "@fragment\nfn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {\n\
                 \x20   let q = H.out_origin + fc.xy;\n    return moda_encode(moda_guard(shade(moda_from_out(q)), q));\n}\n",
            ),
            Self::Transition => (
                "fn progress() -> f32 { return P(1u).y; }\n\
                 fn from_scene(p: vec2<f32>) -> vec4<f32> { return bilinear(0u, moda_to_out(p)); }\n\
                 fn to_scene(p: vec2<f32>) -> vec4<f32> { return bilinear(1u, moda_to_out(p)); }\n\
                 fn src(p: vec2<f32>) -> vec4<f32> { return bilinear(0u, moda_to_out(p)); }\n",
                "@fragment\nfn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {\n\
                 \x20   let q = H.out_origin + fc.xy;\n    return moda_guard(shade(moda_from_out(q)), q);\n}\n",
            ),
        }
    }
}

/// A declared parameter's type.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamType {
    Number,
    Vec2,
    Color,
    Bool,
    /// Up to [`MAX_COLORS`] colours: `name(i)`, `name_count()`, and `std::palette(t)` over the first such parameter.
    Colors(usize),
}

impl ParamType {
    /// Header slots the parameter takes.
    pub fn slots(&self) -> usize {
        match self {
            Self::Colors(n) => n + 1,
            _ => 1,
        }
    }
}

/// A parameter a shader declares in its schema, in schema order.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: ParamType,
    pub default: serde_json::Value,
}

/// A shader's declared interface: its parameters (header slots from [`RESERVED_SLOTS`] on, in order) and its
/// image inputs (`type: "image"` entries; a frame pass's node wires them by name, `src` first).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Schema {
    pub params: Vec<Param>,
    pub images: Vec<String>,
}

impl Schema {
    /// The first parameter slot of each parameter (0-based; a pass's header puts slot k at `RESERVED_SLOTS + k`).
    pub fn slots(&self) -> Vec<usize> {
        let mut at = 0;
        self.params
            .iter()
            .map(|p| {
                let s = at;
                at += p.ty.slots();
                s
            })
            .collect()
    }

    pub fn slot_count(&self) -> usize {
        self.params.iter().map(|p| p.ty.slots()).sum()
    }
}

/// WGSL words a parameter can't be named (keywords, reserved words that matter here, and the ABI's own names).
const RESERVED_NAMES: &[&str] = &[
    "alias", "break", "case", "const", "const_assert", "continue", "continuing", "default", "diagnostic", "discard",
    "else", "enable", "false", "fn", "for", "if", "let", "loop", "override", "requires", "return", "struct", "switch",
    "true", "var", "while", "bool", "f32", "f16", "i32", "u32", "vec2", "vec3", "vec4", "mat2x2", "mat3x3", "mat4x4",
    "array", "atomic", "ptr", "sampler", "texture_2d", "P", "H", "px", "bilinear", "here", "unpremul", "load_raw",
    "vs", "fs", "kernel", "shade", "src", "input", "coverage", "backdrop", "progress", "from_scene", "to_scene",
    "frame_size", "time", "px_scale", "seed", "bounds", "uv", "std",
    // The fill hosts' own names.
    "premultiplyAlpha", "srgbToLinear", "linearToSrgb", "luminance", "desaturate", "applyOpacity", "blendNormal",
    "frame", "frameUniforms", "paths", "curvePoints", "clipMask", "clipMaskSampler", "clipMaskTexture",
    "clipMaskCoverage", "vertexMain", "fragmentMain", "shadeMain", "shadeStroke", "pageClip", "kineticGlyphLocal",
    "vectorWindingAlongRay", "vectorGlyphCoverage", "CustomShaderData", "VertexOutput", "VertexInput",
    "VectorShadeVSOut", "ClipMaskUniforms", "INVERSE_SQRT_2", "AA_PIXEL_EXPANSION", "MIN_BLUR_RADIUS",
    "MIN_GRADIENT_STOP_DIFF", "STROKE_DEPTH_OFFSET",
];

fn valid_name(n: &str) -> bool {
    let mut cs = n.chars();
    cs.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !n.starts_with("__")
        && !n.starts_with("moda_")
        && !n.starts_with("std__")
        && !RESERVED_NAMES.contains(&n)
}

/// Parse a code object's `schema` (`{ name: { type, default, … } }`): `number` (the default), `vec2`, `color`,
/// `bool`, `colors` (a list; its length is the default's, or `count`) and `image` (an input). Errors are
/// `shader_invalid`.
pub fn parse_schema(schema: &serde_json::Value) -> Result<Schema, Diag> {
    let mut out = Schema::default();
    let Some(map) = schema.as_object() else {
        if schema.is_null() {
            return Ok(out);
        }
        return Err(Diag::invalid("the schema must be an object of parameters: { amount: { type: 'number', default: 1 } }"));
    };
    for (name, d) in map {
        if !valid_name(name) {
            return Err(Diag::invalid(format!(
                "parameter `{name}` can't be a shader parameter name (a WGSL identifier that isn't a keyword or one of the shader's own functions)"
            )));
        }
        let ty = d.get("type").and_then(|t| t.as_str()).unwrap_or("number");
        let default = d.get("default").cloned().unwrap_or(serde_json::Value::Null);
        let ty = match ty {
            "number" => ParamType::Number,
            "vec2" => ParamType::Vec2,
            "color" => ParamType::Color,
            "bool" => ParamType::Bool,
            "colors" => {
                let n = d
                    .get("count")
                    .and_then(|c| c.as_u64())
                    .map(|c| c as usize)
                    .or_else(|| default.as_array().map(|a| a.len()))
                    .unwrap_or(0);
                if n == 0 || n > MAX_COLORS {
                    return Err(Diag::invalid(format!(
                        "parameter `{name}` (colors): 1–{MAX_COLORS} colours (give a default list or `count`)"
                    )));
                }
                ParamType::Colors(n)
            }
            "image" => {
                out.images.push(name.clone());
                continue;
            }
            other => {
                return Err(Diag::invalid(format!(
                    "parameter `{name}`: `{other}` isn't a shader type (number, vec2, color, bool, colors, image)"
                )))
            }
        };
        out.params.push(Param { name: name.clone(), ty, default });
    }
    if out.slot_count() > MAX_PARAM_SLOTS {
        return Err(Diag::invalid(format!(
            "the parameters take {} slots; a shader has {MAX_PARAM_SLOTS} (one per parameter, a colors list one per colour plus one)",
            out.slot_count()
        )));
    }
    if out.images.len() > MAX_INPUTS {
        return Err(Diag::invalid(format!("a shader reads at most {MAX_INPUTS} images")));
    }
    Ok(out)
}

/// A shader compiled for a placement: the whole module, and where the agent's source sits in it.
#[derive(Clone, Debug)]
pub struct Module {
    pub wgsl: String,
    /// Byte range of the agent's source in `wgsl`.
    pub user: std::ops::Range<usize>,
}

/// Every agent call `std::name` → `std__name` (the same length: positions don't move).
fn spell_std(source: &str) -> String {
    source.replace("std::", "std__")
}

/// The parameter accessors (`amount()`, `tint()`, a colors list's `name(i)` / `name_count()` and `std::palette`),
/// over the host's `moda_param(k)` (parameter slot k, 0-based).
fn param_accessors(schema: &Schema) -> String {
    let mut out = String::new();
    let mut palette = false;
    for (p, slot) in schema.params.iter().zip(schema.slots()) {
        let n = &p.name;
        match &p.ty {
            ParamType::Number => out.push_str(&format!("fn {n}() -> f32 {{ return moda_param({slot}u).x; }}\n")),
            ParamType::Vec2 => out.push_str(&format!("fn {n}() -> vec2<f32> {{ return moda_param({slot}u).xy; }}\n")),
            ParamType::Color => out.push_str(&format!("fn {n}() -> vec4<f32> {{ return moda_param({slot}u); }}\n")),
            ParamType::Bool => out.push_str(&format!("fn {n}() -> bool {{ return moda_param({slot}u).x > 0.5; }}\n")),
            ParamType::Colors(count) => {
                // Slot `slot` holds the count; the colours (premultiplied) follow.
                out.push_str(&format!(
                    "fn {n}_count() -> u32 {{ return u32(moda_param({slot}u).x); }}\n\
                     fn {n}(i: u32) -> vec4<f32> {{ return moda_param({slot}u + 1u + min(i, {last}u)); }}\n",
                    last = count - 1
                ));
                if !palette {
                    palette = true;
                    // OKLab-interpolated stops, evenly spaced over 0–1 (straight colours in, premultiplied out).
                    out.push_str(&format!(
                        "fn std__palette(t: f32) -> vec4<f32> {{\n\
                         \x20   let n = {n}_count();\n\
                         \x20   if (n <= 1u) {{ return {n}(0u); }}\n\
                         \x20   let x = clamp(t, 0.0, 1.0) * f32(n - 1u);\n\
                         \x20   let i = min(u32(floor(x)), n - 2u);\n\
                         \x20   let a = std__unpremul({n}(i));\n\
                         \x20   let b = std__unpremul({n}(i + 1u));\n\
                         \x20   return std__premul(std__mix_oklab(a, b, x - f32(i)));\n}}\n"
                    ));
                }
            }
        }
    }
    out
}

/// The procedural-fill host's accessors and `shadeMain` adapter (shape and text modules alike): `p` is the
/// instance's local position from its bounds' top-left; parameters are `shaderParams[2..16]` then
/// `colorData[0..16]`; `shaderParams[0]` = (time, seed bits, px_scale, 0), `shaderParams[1]` = frame size.
const FILL_ADAPTER_HEAD: &str = r#"
var<private> moda_path: CustomShaderData;

fn moda_param(k: u32) -> vec4<f32> {
    if (k < 14u) {
        return moda_path.shaderParams[k + 2u];
    }
    return moda_path.colorData[min(k - 14u, 15u)];
}
fn frame_size() -> vec2<f32> { return moda_path.shaderParams[1].xy; }
fn time() -> f32 { return moda_path.shaderParams[0].x; }
fn px_scale() -> f32 { return moda_path.shaderParams[0].z; }
fn seed() -> u32 { return bitcast<u32>(moda_path.shaderParams[0].y); }
fn bounds() -> vec4<f32> { return vec4<f32>(0.0, 0.0, moda_path.bounds.zw); }
fn uv(p: vec2<f32>) -> vec2<f32> { return p / max(moda_path.bounds.zw, vec2<f32>(1e-6)); }
fn src(p: vec2<f32>) -> vec4<f32> { return vec4<f32>(0.0); }
"#;

/// The fill's `shadeMain`: the agent's premultiplied linear result → the straight, sRGB-encoded colour the fill
/// pipeline expects (the vector renderer blends encoded colours, as every other fill); a non-finite result is a
/// magenta checker (8 px squares in world space).
const FILL_ADAPTER_TAIL: &str = r#"
fn moda_encode(l: vec3<f32>) -> vec3<f32> {
    let c = clamp(l, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3<f32>(0.0031308));
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
    moda_path = path;
    let c = shade(input.localPos - path.bounds.xy);
    let bad = (bitcast<vec4<u32>>(c) & vec4<u32>(0x7f800000u)) == vec4<u32>(0x7f800000u);
    if (any(bad)) {
        let k = (i32(floor(input.worldPos.x / 8.0)) + i32(floor(input.worldPos.y / 8.0))) & 1;
        return select(vec4<f32>(1.0, 0.0, 1.0, 1.0), vec4<f32>(0.35, 0.0, 0.35, 1.0), k == 1);
    }
    if (c.a <= 0.0) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(moda_encode(c.rgb / c.a), min(c.a, 1.0));
}
"#;

/// The production procedural-fill module around a shader body (instanced shapes).
pub const FILL_HOST_HEAD: &str = include_str!("hosts/fill-head.wgsl");
pub const FILL_HOST_TAIL: &str = include_str!("hosts/fill-tail.wgsl");
/// The production vector-text fill module around a shader body (glyph outlines, analytic coverage).
pub const TEXT_HOST_HEAD: &str = include_str!("hosts/text-head.wgsl");
pub const TEXT_HOST_TAIL: &str = include_str!("hosts/text-tail.wgsl");

fn assemble(head: &str, accessors: &str, schema: &Schema, source: &str, tail: &[&str]) -> Module {
    let mut wgsl = String::with_capacity(head.len() + STD_WGSL.len() + source.len() + 8192);
    wgsl.push_str(head);
    wgsl.push_str(STD_WGSL);
    wgsl.push('\n');
    wgsl.push_str(accessors);
    wgsl.push_str(&param_accessors(schema));
    wgsl.push_str("// ---- shader ----\n");
    let start = wgsl.len();
    wgsl.push_str(&spell_std(source));
    let end = wgsl.len();
    wgsl.push_str("\n// ---- entry ----\n");
    for t in tail {
        wgsl.push_str(t);
    }
    Module { wgsl, user: start..end }
}

/// The whole WGSL module for `source` in `placement` (host ABI, std prelude, accessors, the source, the entry); for
/// a fill, the instanced shape module ([`fill_text_module`] is the text one). Doesn't validate: [`check`] does.
pub fn module(source: &str, schema: &Schema, placement: Placement) -> Module {
    if placement == Placement::Fill {
        return assemble(FILL_HOST_HEAD, FILL_ADAPTER_HEAD, schema, source, &[FILL_ADAPTER_TAIL, "\n", FILL_HOST_TAIL]);
    }
    let (acc, entry) = placement.accessors();
    let head = format!("{PASS_PRELUDE}{AGENT_HOST}");
    assemble(&head, acc, schema, source, &[entry])
}

/// A fill's module for text (the vector-text fill pipeline).
pub fn fill_text_module(source: &str, schema: &Schema) -> Module {
    assemble(TEXT_HOST_HEAD, FILL_ADAPTER_HEAD, schema, source, &[FILL_ADAPTER_TAIL, "\n", TEXT_HOST_TAIL])
}

/// A fill's parameters and host values as the instance's `shaderParams` (16) and `colorData` (16) arrays.
pub fn encode_fill(host: &HostSlots, params: &[[f32; 4]]) -> ([[f32; 4]; 16], [[f32; 4]; 16]) {
    let mut sp = [[0.0f32; 4]; 16];
    let mut cd = [[0.0f32; 4]; 16];
    sp[0] = [host.time, f32::from_bits(host.seed), host.px_scale, 0.0];
    sp[1] = [host.frame_size[0], host.frame_size[1], 0.0, 0.0];
    for (k, v) in params.iter().enumerate().take(MAX_PARAM_SLOTS) {
        if k < 14 {
            sp[k + 2] = *v;
        } else {
            cd[k - 14] = *v;
        }
    }
    (sp, cd)
}

/// The header slots an agent shader's host fills, from 0 to [`RESERVED_SLOTS`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HostSlots {
    /// Frame size in `p` units (for a fill or effect: the frame's, not the layer's).
    pub frame_size: [f32; 2],
    pub time: f32,
    /// Output pixels per `p` unit.
    pub px_scale: f32,
    pub seed: u32,
    /// A transition's progress, 0–1.
    pub progress: f32,
    /// The placement's bounds in `p` units: x, y, w, h.
    pub bounds: [f32; 4],
    /// `p` → output pixels: `[a, b, c, d, e, f]` (x' = a·x + c·y + e, y' = b·x + d·y + f).
    pub to_out: [f64; 6],
}

impl HostSlots {
    /// The reserved slots, in header order. `None` when `to_out` can't be inverted.
    pub fn encode(&self) -> Option<[[f32; 4]; RESERVED_SLOTS]> {
        let m = self.to_out;
        let det = m[0] * m[3] - m[1] * m[2];
        if det.abs() < 1e-12 || !det.is_finite() {
            return None;
        }
        let inv = [
            m[3] / det,
            -m[1] / det,
            -m[2] / det,
            m[0] / det,
            (m[2] * m[5] - m[3] * m[4]) / det,
            (m[1] * m[4] - m[0] * m[5]) / det,
        ];
        let f = |v: f64| v as f32;
        Some([
            [self.frame_size[0], self.frame_size[1], self.time, self.px_scale],
            [f32::from_bits(self.seed), self.progress, 0.0, 0.0],
            self.bounds,
            [f(inv[0]), f(inv[1]), f(inv[2]), f(inv[3])],
            [f(inv[4]), f(inv[5]), 0.0, 0.0],
            [f(m[0]), f(m[1]), f(m[2]), f(m[3])],
            [f(m[4]), f(m[5]), 0.0, 0.0],
        ])
    }
}

/// A shader's seed: the node's `seed` mixed with the code's hash (never a node id), so two placements of one shader
/// with one seed draw the same, and two shaders with one seed don't.
pub fn mix_seed(seed: u32, code_hash: &str) -> u32 {
    // FNV-1a over the hash text, then lowbias32 with the seed: stable on every platform.
    let mut h: u32 = 0x811c_9dc5;
    for b in code_hash.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    lowbias32(h ^ lowbias32(seed))
}

fn lowbias32(x: u32) -> u32 {
    let mut h = x;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    h
}

/// Header slots for parameter values (straight colours become premultiplied), in schema order after the reserved
/// slots. `value(name)` gives a parameter's value as JSON (number, `[x, y]`, `[r, g, b, a]` linear 0–1 straight,
/// bool, or a list of colours); missing values fall back to the schema's default, then 0.
pub fn encode_params(
    schema: &Schema,
    value: &dyn Fn(&str) -> Option<serde_json::Value>,
    color: &dyn Fn(&serde_json::Value) -> Option<[f32; 4]>,
) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(schema.slot_count());
    let num = |v: &serde_json::Value| v.as_f64().map(|n| n as f32).or_else(|| v.as_bool().map(|b| b as u8 as f32));
    let premul = |c: [f32; 4]| [c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]];
    for p in &schema.params {
        let v = value(&p.name).filter(|v| !v.is_null()).unwrap_or_else(|| p.default.clone());
        match &p.ty {
            ParamType::Number | ParamType::Bool => out.push([num(&v).unwrap_or(0.0), 0.0, 0.0, 0.0]),
            ParamType::Vec2 => {
                let a = v.as_array();
                let g = |i: usize| a.and_then(|a| a.get(i)).and_then(num).unwrap_or(0.0);
                out.push([g(0), g(1), 0.0, 0.0]);
            }
            ParamType::Color => out.push(premul(color(&v).unwrap_or([0.0, 0.0, 0.0, 1.0]))),
            ParamType::Colors(n) => {
                let list: Vec<[f32; 4]> =
                    v.as_array().map(|a| a.iter().filter_map(color).take(*n).collect()).unwrap_or_default();
                out.push([list.len().max(1) as f32, 0.0, 0.0, 0.0]);
                for i in 0..*n {
                    out.push(premul(list.get(i).or(list.last()).copied().unwrap_or([0.0, 0.0, 0.0, 1.0])));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
