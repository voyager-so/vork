//! Paint stacks (E3.1, docs/roadmap/design-editor/data-model-audit.md §3):
//! the typed `Paint` / `Fill` / `Stroke` a node's `fills` / `strokes` (and a
//! rich-text node's `textFills`, a page's `fills`) hold, and the helpers the
//! renderers read them through.
//!
//! The mirror of `packages/editor/src/core/scene/paint.ts` (the authority for
//! the JSON shape). The paint's `type` is the only routing rule: what is
//! stored is what draws, and a node without a paint draws nothing (there is no
//! default fill or stroke colour). Layers run bottom to top. The engine draws
//! every visible layer with its opacity and blend mode (E3.1 P6,
//! `moda-engine/src/paint/layers.rs`), each as a clone of the node carrying
//! that one layer, so the per-paint readers here ([`top_fill`] /
//! [`top_stroke`]) see the layer they draw.
//!
//! `tests/flat_paint_keys.rs` fails when any file in the renderer crates names
//! one of the flat keys this model replaced.

use serde::{Deserialize, Serialize};
use crate::value::{json, Map, Value};

use crate::types::{RawNode, RawPage};

// ---------------------------------------------------------------------------
// Types (serde shape = the TS JSON shape: camelCase keys, absent = omitted)
// ---------------------------------------------------------------------------

/// A point in the node's 0-100 relative bounds space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PaintPoint {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PaintSize {
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PaintRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// One gradient stop. `color` is a `ColorValue`: a CSS string or a variable
/// reference object (resolved before rendering), so it stays raw JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientStop {
    pub offset: f64,
    pub color: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolidPaint {
    pub color: Value,
}

/// The colour space a gradient's stops blend in (CSS Color 4's `in <space>`). Absent = `srgb`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientInterpolation {
    /// Straight RGBA in gamma-encoded sRGB: the original ramp.
    #[serde(rename = "srgb")]
    Srgb,
    /// Linear-light sRGB, premultiplied.
    #[serde(rename = "srgb-linear")]
    SrgbLinear,
    /// OKLab, premultiplied: perceptually even, no grey dip between complements.
    #[serde(rename = "oklab")]
    Oklab,
    /// OKLCh along the shorter hue arc, premultiplied: keeps chroma through the middle.
    #[serde(rename = "oklch")]
    Oklch,
}

/// How a linear, radial, conic or diamond gradient renders its stops (Illustrator audit §11; the TS twin is
/// `core/scene/gradientInterpolation.ts`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct GradientOptions {
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "lenient_bool")]
    pub repeat: Option<bool>,
    /// An unknown space reads as absent (the sRGB ramp) rather than failing the whole paint.
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "lenient_interpolation")]
    pub interpolation: Option<GradientInterpolation>,
    /// Add ±1 LSB noise so long, low-contrast ramps don't band in 8-bit output. Absent = off.
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "lenient_bool")]
    pub dither: Option<bool>,
}

fn lenient_interpolation<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<GradientInterpolation>, D::Error> {
    Ok(crate::value::from_value(Value::deserialize(d)?).ok())
}

fn lenient_bool<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<bool>, D::Error> {
    Ok(Value::deserialize(d)?.as_bool())
}

impl GradientOptions {
    /// The shaders' `flags` (`wgsl/include/gradient-space.wgsl`): space code + 8 × dither + 16 × repeat.
    pub fn shader_flags(&self) -> f32 {
        let space = match self.interpolation {
            None | Some(GradientInterpolation::Srgb) => 0.0,
            Some(GradientInterpolation::SrgbLinear) => 1.0,
            Some(GradientInterpolation::Oklab) => 2.0,
            Some(GradientInterpolation::Oklch) => 3.0,
        };
        space + if self.dither == Some(true) { 8.0 } else { 0.0 }
            + if self.repeat == Some(true) { 16.0 } else { 0.0 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearGradientPaint {
    pub start: PaintPoint,
    pub end: PaintPoint,
    pub stops: Vec<GradientStop>,
    #[serde(flatten)]
    pub options: GradientOptions,
}

/// Radial (and, with the same fields, diamond) gradient placement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RadialGradientPaint {
    pub center: PaintPoint,
    pub radius_x: f64,
    pub radius_y: f64,
    /// Degrees.
    pub rotation: f64,
    pub stops: Vec<GradientStop>,
    #[serde(flatten)]
    pub options: GradientOptions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConicGradientPaint {
    pub center: PaintPoint,
    /// Degrees; 0 points up, clockwise.
    pub angle: f64,
    pub stops: Vec<GradientStop>,
    #[serde(flatten)]
    pub options: GradientOptions,
}

/// One colour point of a freeform gradient (Illustrator's Freeform
/// gradient, E8.18). `x`/`y` are in the node's 0-100 bounds space; `spread`
/// (0..100, % of half the gradient box's diagonal) is the radius the colour
/// holds solid over; points sharing a `line` join, in array order, into a
/// line (Illustrator's Lines mode).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FreeformGradientPoint {
    pub x: f64,
    pub y: f64,
    pub color: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spread: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<f64>,
}

/// Illustrator's Freeform gradient (fills only): colour points blended by
/// inverse-distance weighting (`moda-engine` `paint/gradient.rs`, the TS
/// twin `core/scene/freeformGradient.ts`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FreeformGradientPaint {
    pub points: Vec<FreeformGradientPoint>,
}

/// How many freeform points the renderer draws (`MAX_FREEFORM_POINTS` in
/// paint.ts).
pub const MAX_FREEFORM_POINTS: usize = 16;

/// Mirroring of a Fill / Fit image (`ImagePlacement.flip`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PaintFlip {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<bool>,
}

/// Where an image or video sits in the shape: Figma's image scale mode, a
/// LIVE mode resolved against the shape's current box ([`ImagePlacement::resolve`],
/// the TS twin `core/scene/imagePlacement.ts`). `fit`: `cover` (default) /
/// `contain` re-fit on every size change; `fill` (CSS `object-fit: fill`)
/// stretches the image on each axis onto the box; `crop` keeps `offset` / `scale`
/// (px at the box size `box`) relative to the box; `tile` keeps its px.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ImagePlacement {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub align: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flip: Option<PaintFlip>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<PaintPoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<PaintPoint>,
    #[serde(rename = "box", skip_serializing_if = "Option::is_none")]
    pub box_size: Option<PaintSize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rotation: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub natural_size: Option<PaintSize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_bounds: Option<PaintRect>,
}

/// The pixel transform an image fill draws with (`path-pattern.wgsl`): a
/// shape-local point `p` samples the image at `R(p − offset) / scale`
/// natural px, `R` the rotation; a negative scale mirrors that axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedImagePlacement {
    pub offset: (f64, f64),
    pub scale: (f64, f64),
    /// Degrees.
    pub rotation: f64,
    /// `repeat` | `repeat-x` | `repeat-y` | `no-repeat`.
    pub repeat: &'static str,
}

/// The box a boxed node's image fill sizes to: `[0, 0, width, height]`, or
/// `None` for a node without both (a path: callers pass its geometry bounds).
pub fn node_image_box(node: &RawNode) -> Option<[f64; 4]> {
    match (node.f64("width"), node.f64("height")) {
        (Some(w), Some(h)) if w.is_finite() && h.is_finite() => Some([0.0, 0.0, w, h]),
        _ => None,
    }
}

impl ImagePlacement {
    /// The stored scale mode: `cover` (absent), `contain`, `fill` (stretch), `crop` or `tile`.
    pub fn mode(&self) -> &str {
        match self.fit.as_deref() {
            Some(m @ ("contain" | "fill" | "crop" | "tile")) => m,
            _ => "cover",
        }
    }

    /// Resolve against the shape's box `[x, y, w, h]` (node-local). `natural`
    /// is the decoded image size, used when the placement stores none; Fill and
    /// Fit without any natural size draw at scale 1 from the box origin.
    pub fn resolve(&self, image_box: [f64; 4], natural: Option<(f64, f64)>) -> ResolvedImagePlacement {
        let fin = |v: Option<f64>, d: f64| v.filter(|v| v.is_finite()).unwrap_or(d);
        let or1 = |v: Option<f64>| v.filter(|v| v.is_finite() && *v != 0.0).unwrap_or(1.0);
        let pos = |v: f64| v.is_finite() && v > 0.0;
        let [bx, by, bw, bh] = image_box;
        let (bx, by) = (fin(Some(bx), 0.0), fin(Some(by), 0.0));
        let (bw, bh) = (fin(Some(bw), 0.0).max(1e-6), fin(Some(bh), 0.0).max(1e-6));
        let rotation = fin(self.rotation, 0.0);
        match self.mode() {
            "tile" => {
                let repeat = match self.repeat.as_deref() {
                    Some("repeat-x") => "repeat-x",
                    Some("repeat-y") => "repeat-y",
                    _ => "repeat",
                };
                ResolvedImagePlacement {
                    offset: (
                        fin(self.offset.map(|o| o.x), 0.0),
                        fin(self.offset.map(|o| o.y), 0.0),
                    ),
                    scale: (or1(self.scale.map(|s| s.x)), or1(self.scale.map(|s| s.y))),
                    rotation,
                    repeat,
                }
            }
            "crop" => {
                let kx = self
                    .box_size
                    .filter(|b| pos(b.width))
                    .map(|b| bw / b.width)
                    .unwrap_or(1.0);
                let ky = self
                    .box_size
                    .filter(|b| pos(b.height))
                    .map(|b| bh / b.height)
                    .unwrap_or(1.0);
                ResolvedImagePlacement {
                    offset: (
                        fin(self.offset.map(|o| o.x), 0.0) * kx,
                        fin(self.offset.map(|o| o.y), 0.0) * ky,
                    ),
                    scale: (
                        or1(self.scale.map(|s| s.x)) * kx,
                        or1(self.scale.map(|s| s.y)) * ky,
                    ),
                    rotation,
                    repeat: "no-repeat",
                }
            }
            mode => {
                let natural = self
                    .natural_size
                    .map(|n| (n.width, n.height))
                    .filter(|(w, h)| pos(*w) && pos(*h))
                    .or(natural.filter(|(w, h)| pos(*w) && pos(*h)));
                let Some((nw, nh)) = natural else {
                    return ResolvedImagePlacement {
                        offset: (bx, by),
                        scale: (1.0, 1.0),
                        rotation,
                        repeat: "no-repeat",
                    };
                };
                // The content (auto-trimmed padding) is what Fill / Fit size to.
                let (cx, cy, cw, ch) = match self.content_bounds {
                    Some(c) if pos(c.width) && pos(c.height) => {
                        (fin(Some(c.x), 0.0), fin(Some(c.y), 0.0), c.width, c.height)
                    }
                    _ => (0.0, 0.0, nw, nh),
                };
                let r = rotation.to_radians();
                let fx = self.flip.and_then(|f| f.x).unwrap_or(false);
                let fy = self.flip.and_then(|f| f.y).unwrap_or(false);
                let sgn = |f: bool| if f { -1.0 } else { 1.0 };
                if mode == "fill" {
                    // Stretch (CSS `object-fit: fill`): each image axis scaled onto the box
                    // side it lies closest to (a quarter turn swaps them), centred in the box.
                    let swapped = r.sin().abs() > r.cos().abs();
                    let kx = if swapped { bh } else { bw } / cw;
                    let ky = if swapped { bw } else { bh } / ch;
                    let q = ((cx + cw / 2.0) * kx * sgn(fx), (cy + ch / 2.0) * ky * sgn(fy));
                    let (c, s) = (r.cos(), r.sin());
                    let d = (q.0 * c + q.1 * s, -q.0 * s + q.1 * c);
                    return ResolvedImagePlacement {
                        offset: (bx + bw / 2.0 - d.0, by + bh / 2.0 - d.1),
                        scale: (kx * sgn(fx), ky * sgn(fy)),
                        rotation,
                        repeat: "no-repeat",
                    };
                }
                let (ac, as_) = (r.cos().abs(), r.sin().abs());
                let foot_w = cw * ac + ch * as_;
                let foot_h = cw * as_ + ch * ac;
                let (sx, sy) = (bw / foot_w, bh / foot_h);
                let k = if mode == "contain" { sx.min(sy) } else { sx.max(sy) };
                let mut parts = self
                    .align
                    .as_deref()
                    .unwrap_or("center middle")
                    .split_whitespace();
                let ah = match parts.next() {
                    Some("left") => 0.0,
                    Some("right") => 1.0,
                    _ => 0.5,
                };
                let av = match parts.next() {
                    Some("top") => 0.0,
                    Some("bottom") => 1.0,
                    _ => 0.5,
                };
                let centre = (
                    bx + (bw - foot_w * k) * ah + foot_w * k / 2.0,
                    by + (bh - foot_h * k) * av + foot_h * k / 2.0,
                );
                let q = ((cx + cw / 2.0) * k * sgn(fx), (cy + ch / 2.0) * k * sgn(fy));
                // Sample space → shape space: rotate by −rotation.
                let (c, s) = (r.cos(), r.sin());
                let d = (q.0 * c + q.1 * s, -q.0 * s + q.1 * c);
                ResolvedImagePlacement {
                    offset: (centre.0 - d.0, centre.1 - d.1),
                    scale: (k * sgn(fx), k * sgn(fy)),
                    rotation,
                    repeat: "no-repeat",
                }
            }
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ImageOverlay {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blend_mode: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ImageRecolor {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smoothness: Option<f64>,
}

/// Hue/Saturation with Colorize enabled, in degrees and percent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ImageColorize {
    pub enabled: bool,
    pub hue: f64,
    pub saturation: f64,
    pub lightness: f64,
}

/// Colour treatments of an image or video: Figma's seven image adjustments
/// (E7.18; each −1..1, absent or 0 = unchanged; the formulas are
/// `moda-engine`'s `paint::pattern::image_adjust`), plus Colorize, palette
/// recolour and overlay.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ImageAdjust {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub colorize: Option<ImageColorize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exposure: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contrast: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub saturation: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tint: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub highlights: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadows: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overlay: Option<ImageOverlay>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recolor: Option<ImageRecolor>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImagePaint {
    /// Absent: the renderer draws its white placeholder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub src: Option<String>,
    #[serde(default)]
    pub placement: ImagePlacement,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adjust: Option<ImageAdjust>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct VideoPlayback {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trim_start_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trim_end_ms: Option<f64>,
    #[serde(rename = "loop", skip_serializing_if = "Option::is_none")]
    pub loop_: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub muted: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume: Option<f64>,
    /// Finite dB number, or the JSON-safe silence token "-inf". Separate field: no existing stored unit changes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume_db: Option<crate::value::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoPaint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub src: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub poster_src: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub poster_time_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub natural_size: Option<PaintSize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub has_audio: Option<bool>,
    #[serde(default)]
    pub playback: VideoPlayback,
    #[serde(default)]
    pub placement: ImagePlacement,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adjust: Option<ImageAdjust>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShaderPaint {
    pub shader_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Map>,
}

/// Where a vector pattern's tile space sits in the node (E8.5, `moda_scene::pattern`): node-local =
/// `T(x, y) · R(rotation) · S(scale)` · pattern space, or any affine as `matrix` (`patterns.ts`
/// `PatternTransform`), which takes precedence when it is usable (six finite numbers, a non-zero determinant). It is
/// the paint's own transform, independent of the node's (Illustrator's "transform patterns only"): the tiles move
/// with the node, and this moves them inside it.
#[derive(Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PatternTransform {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<f64>,
    /// Uniform; absent, zero or not finite = 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
    /// Degrees, clockwise (the node rotation's sense).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rotation: Option<f64>,
    /// Any affine `[a, b, c, d, e, f]` (x' = a·x + c·y + e, y' = b·x + d·y + f): a non-uniform scale, shear or
    /// reflection of the tiles. Used in place of the fields above when usable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matrix: Option<[f64; 6]>,
}

// Debug as before `matrix` existed when it is absent: plan digests print paints (existing_documents_unchanged).
impl std::fmt::Debug for PatternTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("PatternTransform");
        d.field("x", &self.x)
            .field("y", &self.y)
            .field("scale", &self.scale)
            .field("rotation", &self.rotation);
        if let Some(m) = &self.matrix {
            d.field("matrix", m);
        }
        d.finish()
    }
}

impl PatternTransform {
    /// The `matrix` when it is usable: six finite numbers and a non-zero determinant (`isUsablePatternMatrix`).
    pub fn usable_matrix(&self) -> Option<crate::types::Affine> {
        let m = self.matrix?;
        if !m.iter().all(|v| v.is_finite()) || m[0] * m[3] - m[1] * m[2] == 0.0 {
            return None;
        }
        Some(crate::types::Affine {
            a: m[0],
            b: m[1],
            c: m[2],
            d: m[3],
            e: m[4],
            f: m[5],
        })
    }

    /// Pattern space → node-local.
    pub fn affine(&self) -> crate::types::Affine {
        if let Some(m) = self.usable_matrix() {
            return m;
        }
        let fin = |v: Option<f64>, d: f64| v.filter(|v| v.is_finite()).unwrap_or(d);
        let scale = self.scale.filter(|s| s.is_finite() && *s != 0.0).unwrap_or(1.0);
        crate::transform::from_trs(crate::types::Transform2D {
            x: fin(self.x, 0.0),
            y: fin(self.y, 0.0),
            rotation: fin(self.rotation, 0.0),
            scale_x: scale,
            scale_y: scale,
        })
    }
}

/// A vector pattern (Illustrator's pattern swatch, E8.5): the document pattern `patternId` names
/// (`canvas.patterns`, `moda_scene::pattern`), tiled over the layer's region and clipped to it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PatternPaint {
    pub pattern_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<PatternTransform>,
}

impl PatternPaint {
    /// Pattern space → node-local (identity without a transform).
    pub fn affine(&self) -> crate::types::Affine {
        self.transform.unwrap_or_default().affine()
    }
}

/// What one fill or stroke layer paints. Tagged by `type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Paint {
    Solid(SolidPaint),
    Linear(LinearGradientPaint),
    Radial(RadialGradientPaint),
    Conic(ConicGradientPaint),
    /// Figma's diamond gradient: placed like a radial, ramped by the L1
    /// distance (the gradient WGSL's fill type 4).
    Diamond(RadialGradientPaint),
    /// Illustrator's Freeform gradient (E8.18): fills only.
    Freeform(FreeformGradientPaint),
    Image(ImagePaint),
    Video(VideoPaint),
    Shader(ShaderPaint),
    /// A vector pattern from the document's pattern table (E8.5).
    Pattern(PatternPaint),
    /// Illustrator's gradient mesh (fills only; `crate::mesh_gradient`).
    Mesh(crate::mesh_gradient::MeshGradientPaint),
}

impl Paint {
    /// A stop gradient's interpolation space and dither (default for any other paint).
    pub fn gradient_options(&self) -> GradientOptions {
        match self {
            Paint::Linear(l) => l.options,
            Paint::Radial(r) | Paint::Diamond(r) => r.options,
            Paint::Conic(c) => c.options,
            _ => GradientOptions::default(),
        }
    }
}

/// One fill layer (bottom to top in a node's `fills`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Fill {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blend_mode: Option<String>,
    /// Prints over the inks under it in a CMYK export (`paint.ts` `PaintLayerBase.overprint`); nothing here draws it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overprint: Option<bool>,
    pub paint: Paint,
    /// Effects scoped to this fill alone (Illustrator's Appearance, E8.2; `crate::appearance`): geometry
    /// effects reshape just this layer's outline, drawing effects (blur, glow, shadows) filter just its pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effects: Option<Vec<Value>>,
}

/// One stroke layer, with its own weight, alignment, caps, joins, dashes,
/// width/brush objects and arrowheads (Illustrator's model). The brush and
/// width objects are kept as stored JSON; `stroke_model` parses and
/// normalises them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Stroke {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blend_mode: Option<String>,
    /// Prints over the inks under it in a CMYK export (`paint.ts` `PaintLayerBase.overprint`); nothing here draws it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overprint: Option<bool>,
    pub paint: Paint,
    pub width: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cap: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub join: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub miter_limit: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dash: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dash_offset: Option<f64>,
    /// How the dash pattern is laid along the path: absent / `exact` = as
    /// authored from each sub-path's start; `corners` = Illustrator's "Align
    /// dashes to corners and path ends, adjusting lengths to fit".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dash_alignment: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub taper: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wave: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brush: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_arrow: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_arrow: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arrow_scale: Option<f64>,
    /// How a linear or radial gradient lies on the stroke (Illustrator's
    /// Stroke options, E8.18): absent / `within` = over the node's box, as a
    /// fill; `along` = its stops run along the path; `across` = from the
    /// stroke's left edge to its right.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradient_mode: Option<String>,
    /// Per-side weights on a rectangle or frame (E6.2, Figma's individual stroke weights): each side draws its own
    /// weight, placed by `align`, and `width` is the widest side. Other node types ignore it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sides: Option<StrokeSides>,
    /// Effects scoped to this stroke alone (see [`Fill::effects`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effects: Option<Vec<Value>>,
}

/// A stroke's per-side weights (px, ≥ 0; 0 draws nothing on that side).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct StrokeSides {
    #[serde(default)]
    pub top: f64,
    #[serde(default)]
    pub right: f64,
    #[serde(default)]
    pub bottom: f64,
    #[serde(default)]
    pub left: f64,
}

impl StrokeSides {
    /// `[top, right, bottom, left]`, non-finite or negative weights as 0.
    pub fn weights(&self) -> [f64; 4] {
        [self.top, self.right, self.bottom, self.left].map(|w| if w.is_finite() { w.max(0.0) } else { 0.0 })
    }
}

/// How a stroke's gradient lies on it ([`Stroke::gradient_mode`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrokeGradientMode {
    Within,
    Along,
    Across,
}

impl Stroke {
    /// The stroke's gradient mode (`within` for absent or unknown).
    pub fn gradient_mode(&self) -> StrokeGradientMode {
        match self.gradient_mode.as_deref() {
            Some("along") => StrokeGradientMode::Along,
            Some("across") => StrokeGradientMode::Across,
            _ => StrokeGradientMode::Within,
        }
    }

    /// A solid stroke of `width`, no settings.
    pub fn solid(id: &str, color: Value, width: f64) -> Stroke {
        Stroke {
            id: id.to_string(),
            visible: None,
            opacity: None,
            blend_mode: None,
            overprint: None,
            paint: Paint::Solid(SolidPaint { color }),
            width,
            align: None,
            cap: None,
            join: None,
            miter_limit: None,
            dash: None,
            dash_offset: None,
            dash_alignment: None,
            profile: None,
            taper: None,
            wave: None,
            dynamic: None,
            brush: None,
            start_arrow: None,
            end_arrow: None,
            arrow_scale: None,
            gradient_mode: None,
            sides: None,
            effects: None,
        }
    }

    /// The stroke's geometry (everything but the paint and layer props).
    pub fn style(&self) -> StrokeStyle {
        StrokeStyle {
            width: Some(self.width),
            align: self.align.clone(),
            cap: self.cap.clone(),
            join: self.join.clone(),
            miter_limit: self.miter_limit,
            dash: zero_dashes_as_dots(self.dash.as_deref(), self.cap.as_deref(), self.width),
            dash_offset: self.dash_offset,
            dash_alignment: self.dash_alignment.clone(),
            profile: self.profile.clone().filter(|v| !v.is_null()),
            taper: self.taper.clone().filter(|v| !v.is_null()),
            wave: self.wave.clone().filter(|v| !v.is_null()),
            dynamic: self.dynamic.clone().filter(|v| !v.is_null()),
            brush: self.brush.clone().filter(|v| !v.is_null()),
            start_arrow: self.start_arrow.clone(),
            end_arrow: self.end_arrow.clone(),
            arrow_scale: self.arrow_scale,
            sides: self.sides,
        }
    }
}

/// A dash's drawn length (px) for a zero-length dash with a round or square cap.
const ZERO_DASH_PX: f64 = 0.01;

/// A zero-length dash with a round or square cap is a dot, as in SVG and PDF (`dash="0 20"` is a dotted line). The
/// dash walk drops pieces of 0.001 px or less, so such a dash would draw nothing; it is drawn as a 0.01 px piece taken
/// from the gap after it, which keeps the pattern's period. A butt cap draws nothing for it, and the pattern is
/// unchanged. Stored dashes are in units of half the stroke width (`scale_dash_pattern`).
fn zero_dashes_as_dots(dash: Option<&[f64]>, cap: Option<&str>, width: f64) -> Option<Vec<f64>> {
    let d = dash?;
    let capped = matches!(cap, Some("round") | Some("square"));
    let unit = width / 2.0;
    let is_zero = |v: f64| (0.0..=0.001).contains(&(v * unit));
    if !capped || width.is_nan() || width <= 0.0 || !d.iter().all(|v| v.is_finite()) || !d.iter().any(|v| is_zero(*v)) {
        return Some(d.to_vec());
    }
    // Odd patterns repeat twice (SVG); doubled here so every entry keeps its on / off role.
    let mut out = d.to_vec();
    if out.len() % 2 == 1 {
        out.extend_from_slice(d);
    }
    let eps = ZERO_DASH_PX / unit;
    for i in (0..out.len()).step_by(2) {
        if is_zero(out[i]) && out[i + 1] > 2.0 * eps {
            out[i + 1] -= eps - out[i];
            out[i] = eps;
        }
    }
    Some(out)
}

impl Fill {
    /// A solid fill layer.
    pub fn solid(id: &str, color: Value) -> Fill {
        Fill {
            id: id.to_string(),
            visible: None,
            opacity: None,
            blend_mode: None,
            overprint: None,
            paint: Paint::Solid(SolidPaint { color }),
            effects: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

/// Node types with fills (rich text's are its box background).
pub const FILL_NODE_TYPES: &[&str] = &[
    "rectangle",
    "container",
    "section",
    "ellipse",
    "path",
    "boolean",
    "star",
    "polygon",
    "richtext",
];

/// Node types with strokes.
pub const STROKE_NODE_TYPES: &[&str] = &[
    "rectangle",
    "container",
    "section",
    "ellipse",
    "path",
    "boolean",
    "star",
    "polygon",
    "richtext",
    "line",
    "table",
    "generated",
];

/// Parse a stored layer list, skipping any entry that is not a well-formed
/// layer (a hand-edited or truncated document never panics the renderer).
fn parse_layers<T: for<'de> Deserialize<'de>>(v: Option<&Value>) -> Vec<T> {
    match v {
        Some(Value::Array(items)) => items.iter().filter_map(|i| T::deserialize(i).ok()).collect(),
        _ => Vec::new(),
    }
}

/// A fill stack from its JSON value (`fills`).
pub fn parse_fills(v: Option<&Value>) -> Vec<Fill> {
    parse_layers(v)
}

/// A stroke stack from its JSON value (`strokes`).
pub fn parse_strokes(v: Option<&Value>) -> Vec<Stroke> {
    parse_layers(v)
}

/// A node's fill stack, bottom to top (`[]` for node types without fills).
pub fn node_fills(node: &RawNode) -> Vec<Fill> {
    node_fills_shared(node).to_vec()
}

/// [`node_fills`], shared: parsed once per node (`RawNode` keeps it until its fields change).
pub fn node_fills_shared(node: &RawNode) -> std::sync::Arc<[Fill]> {
    node.fills_memo(|| {
        if !FILL_NODE_TYPES.contains(&node.node_type()) {
            return Vec::new();
        }
        parse_fills(node.json_field("fills"))
    })
}

/// A node's stroke stack, bottom to top.
pub fn node_strokes(node: &RawNode) -> Vec<Stroke> {
    node_strokes_shared(node).to_vec()
}

/// [`node_strokes`], shared: parsed once per node (`RawNode` keeps it until its fields change).
pub fn node_strokes_shared(node: &RawNode) -> std::sync::Arc<[Stroke]> {
    node.strokes_memo(|| {
        if !STROKE_NODE_TYPES.contains(&node.node_type()) {
            return Vec::new();
        }
        parse_strokes(node.json_field("strokes"))
    })
}

/// A rich-text node's glyph-colour override: the paints of its visible
/// `textFills` layers, bottom to top (`[]` = the run colours render).
pub fn node_text_fills(node: &RawNode) -> Vec<Paint> {
    if node.node_type() != "richtext" {
        return Vec::new();
    }
    parse_fills(node.json_field("textFills"))
        .into_iter()
        .filter(|f| shown(f.visible, f.opacity))
        .map(|f| f.paint)
        .collect()
}

/// A page's background stack.
pub fn page_fills(page: &RawPage) -> Vec<Fill> {
    parse_fills(page.0.get("fills"))
}

/// Whether a layer shows: switched on (`visible` absent or true) and not at opacity 0 (the TS `layerShows`).
pub fn shown(visible: Option<bool>, opacity: Option<f64>) -> bool {
    if opacity == Some(0.0) {
        return false;
    }
    visible != Some(false)
}

/// The top visible layer of a fill stack.
pub fn top_of_fills(fills: Vec<Fill>) -> Option<Fill> {
    fills.into_iter().rev().find(|f| shown(f.visible, f.opacity))
}

/// The node's top visible fill (what the renderers draw until P6).
pub fn top_fill(node: &RawNode) -> Option<Fill> {
    node_fills_shared(node).iter().rev().find(|f| shown(f.visible, f.opacity)).cloned()
}

/// The node's top visible stroke.
pub fn top_stroke(node: &RawNode) -> Option<Stroke> {
    with_top_stroke(node, |s| s.cloned())
}

/// `f` of the node's top visible stroke (by reference: no copy of it).
pub fn with_top_stroke<T>(node: &RawNode, f: impl FnOnce(Option<&Stroke>) -> T) -> T {
    let all = node_strokes_shared(node);
    f(all.iter().rev().find(|s| shown(s.visible, s.opacity)))
}

/// The top visible fill's PAINT object as stored (borrowed JSON; the same
/// layer [`top_fill`] picks), for readers that return references into the
/// node (the shader-effect predicates).
pub fn top_fill_paint_json(node: &RawNode) -> Option<&Map> {
    if !FILL_NODE_TYPES.contains(&node.node_type()) {
        return None;
    }
    let Some(Value::Array(fills)) = node.json_field("fills") else {
        return None;
    };
    fills
        .iter()
        .rev()
        .filter(|f| Fill::deserialize(*f).is_ok())
        .find(|f| {
            f.get("visible").and_then(Value::as_bool) != Some(false)
                && f.get("opacity").and_then(Value::as_f64) != Some(0.0)
        })
        .and_then(|f| f.get("paint"))
        .and_then(Value::as_object)
}

/// The stroke that carries the node's width model and brush: the top-most
/// visible stroke with a width profile or a brush, else the top visible
/// stroke (the Width tool and the brush tools act on the same one; TS twin in
/// the editor's stroke helpers).
pub fn model_stroke(node: &RawNode) -> Option<Stroke> {
    with_model_stroke(node, |s| s.cloned())
}

/// `f` of the node's model stroke ([`model_stroke`], by reference: no copy of it).
pub fn with_model_stroke<T>(node: &RawNode, f: impl FnOnce(Option<&Stroke>) -> T) -> T {
    let all = node_strokes_shared(node);
    let has = |v: &Option<Value>| v.as_ref().is_some_and(|v| !v.is_null());
    let mut shown_strokes = all.iter().filter(|s| shown(s.visible, s.opacity));
    let mut last = None;
    let mut model = None;
    for s in shown_strokes.by_ref() {
        if has(&s.profile) || has(&s.brush) {
            model = Some(s);
        }
        last = Some(s);
    }
    f(model.or(last))
}

/// The top visible stroke of a node-shaped JSON object (a line spec, a
/// transient's source map).
pub fn top_stroke_of(map: &Map) -> Option<Stroke> {
    parse_strokes(map.get("strokes"))
        .into_iter()
        .rev()
        .find(|s| shown(s.visible, s.opacity))
}

/// The rich-text node's glyph override paint (the last `textFills` entry).
pub fn top_text_fill(node: &RawNode) -> Option<Paint> {
    node_text_fills(node).pop()
}

/// The page's top visible background layer.
pub fn top_page_fill(page: &RawPage) -> Option<Fill> {
    top_of_fills(page_fills(page))
}

/// The container body gate (`frame.rs`), as `(has_fill, has_stroke)`: a
/// visible fill, and a visible stroke with a positive width, whatever their
/// paint types.
pub fn container_body_gate(node: &RawNode) -> (bool, bool) {
    (
        top_fill(node).is_some(),
        with_top_stroke(node, |s| s.is_some_and(|s| s.width > 0.0)),
    )
}

/// The stops of a gradient paint (`None` for solid, image, video and shader).
pub fn paint_stops(paint: &Paint) -> Option<&[GradientStop]> {
    match paint {
        Paint::Linear(p) => Some(&p.stops),
        Paint::Radial(p) | Paint::Diamond(p) => Some(&p.stops),
        Paint::Conic(p) => Some(&p.stops),
        _ => None,
    }
}

/// Every colour a paint's stops or freeform points hold (`None` for solid,
/// image, video and shader).
pub fn paint_stop_colors(paint: &Paint) -> Option<Vec<&Value>> {
    match paint {
        Paint::Freeform(f) => Some(f.points.iter().map(|p| &p.color).collect()),
        _ => paint_stops(paint).map(|stops| stops.iter().map(|s| &s.color).collect()),
    }
}

/// The CSS colour string of a solid paint (a resolved colour; an unresolved
/// variable reference is not a string).
pub fn solid_color(paint: &Paint) -> Option<&str> {
    match paint {
        Paint::Solid(s) => s.color.as_str(),
        _ => None,
    }
}

/// The colour a flat-colour consumer (text strokes, arrowheads without a
/// gradient pipeline) uses for a paint: a solid's colour, a gradient's first
/// stop. `None` for images, videos and shaders.
pub fn representative_color(paint: &Paint) -> Option<&str> {
    match paint {
        Paint::Solid(s) => s.color.as_str(),
        Paint::Freeform(f) => f.points.first().and_then(|p| p.color.as_str()),
        _ => paint_stops(paint)
            .and_then(|s| s.first())
            .and_then(|s| s.color.as_str()),
    }
}

/// Stops → the interleaved `[offset, color, …]` array a stop parser takes.
pub fn stops_to_flat(stops: &[GradientStop]) -> Vec<Value> {
    stops
        .iter()
        .flat_map(|s| [Value::from(s.offset), s.color.clone()])
        .collect()
}

/// Every image or video source a fill stack paints: image `src`, video `src`
/// and poster. The asset collectors walk these.
pub fn paint_sources(fills: &[Fill]) -> Vec<&str> {
    let mut out = Vec::new();
    for f in fills {
        match &f.paint {
            // A paint layer's surface is not an asset (the engine's surface registry fills it).
            Paint::Image(p) => out.extend(
                p.src
                    .as_deref()
                    .filter(|s| !s.starts_with(crate::normalize::PAINT_SURFACE_SRC_PREFIX)),
            ),
            Paint::Video(p) => {
                out.extend(p.src.as_deref());
                out.extend(p.poster_src.as_deref());
            }
            _ => {}
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Stroke geometry
// ---------------------------------------------------------------------------

/// The top stroke's GEOMETRY, owned (the paint and layer props left out). A
/// node without a stroke has the default (every field `None`): no stroke band.
/// `dash` and the brush objects stay as stored so every lane keeps its own
/// parse.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StrokeStyle {
    pub width: Option<f64>,
    pub align: Option<String>,
    pub cap: Option<String>,
    pub join: Option<String>,
    pub miter_limit: Option<f64>,
    pub dash: Option<Vec<f64>>,
    pub dash_offset: Option<f64>,
    /// [`Stroke::dash_alignment`].
    pub dash_alignment: Option<String>,
    pub profile: Option<Value>,
    pub taper: Option<Value>,
    pub wave: Option<Value>,
    pub dynamic: Option<Value>,
    pub brush: Option<Value>,
    pub start_arrow: Option<String>,
    pub end_arrow: Option<String>,
    pub arrow_scale: Option<f64>,
    pub sides: Option<StrokeSides>,
}

impl StrokeStyle {
    /// `width > 0`.
    pub fn has_width(&self) -> bool {
        self.width.unwrap_or(0.0) > 0.0
    }

    /// Dashes aligned to corners and path ends ([`Stroke::dash_alignment`] `corners`).
    pub fn dash_aligned(&self) -> bool {
        self.dash_alignment.as_deref() == Some("corners")
    }

    /// The dash pattern as the JSON array the dash parsers take.
    pub fn dash_value(&self) -> Option<Value> {
        self.dash
            .as_ref()
            .map(|d| Value::Array(d.iter().map(|x| Value::from(*x)).collect()))
    }
}

/// The node's top stroke geometry (default without a stroke).
pub fn node_stroke_style(node: &RawNode) -> StrokeStyle {
    with_top_stroke(node, |s| s.map(Stroke::style).unwrap_or_default())
}

/// The MODEL stroke's geometry ([`model_stroke`]: the stroke that carries the
/// width model and brush), default without a stroke.
pub fn model_stroke_style(node: &RawNode) -> StrokeStyle {
    with_model_stroke(node, |s| s.map(Stroke::style).unwrap_or_default())
}

/// The top stroke geometry of a node-shaped JSON object.
pub fn stroke_style_of(map: &Map) -> StrokeStyle {
    top_stroke_of(map).map(|s| s.style()).unwrap_or_default()
}

/// The top stroke geometry of anything node-shaped, through a field getter
/// (a line spec parsed from a JSON object).
pub fn stroke_style_from<'a>(get: impl Fn(&str) -> Option<&'a Value>) -> StrokeStyle {
    parse_strokes(get("strokes"))
        .into_iter()
        .rev()
        .find(|s| shown(s.visible, s.opacity))
        .map(|s| s.style())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Writers for transient nodes (a line outlined as a path, a page background
// rectangle, a generated graphic's placeholder, brush artwork)
// ---------------------------------------------------------------------------

/// Set a node's fill stack (`[]` removes the key).
pub fn set_fills(node: &mut Map, fills: Vec<Fill>) {
    if fills.is_empty() {
        node.remove("fills");
    } else {
        node.insert("fills".into(), json!(fills));
    }
}

/// Set a node's stroke stack (`[]` removes the key).
pub fn set_strokes(node: &mut Map, strokes: Vec<Stroke>) {
    if strokes.is_empty() {
        node.remove("strokes");
    } else {
        node.insert("strokes".into(), json!(strokes));
    }
}

/// One solid fill.
pub fn set_solid_fill(node: &mut Map, color: Value) {
    set_fills(node, vec![Fill::solid("fill-0", color)]);
}

/// One fill with `paint`.
pub fn set_fill_paint(node: &mut Map, paint: Paint) {
    set_fills(
        node,
        vec![Fill {
            id: "fill-0".into(),
            visible: None,
            opacity: None,
            blend_mode: None,
            overprint: None,
            paint,
            effects: None,
        }],
    );
}

/// One solid stroke of `width` with optional caps/joins.
pub fn set_solid_stroke(
    node: &mut Map,
    color: Value,
    width: f64,
    cap: Option<&str>,
    join: Option<&str>,
) {
    let mut s = Stroke::solid("stroke-0", color, width);
    s.cap = cap.map(str::to_string);
    s.join = join.map(str::to_string);
    set_strokes(node, vec![s]);
}

/// A transient path's stroke: `stroke`'s paint and layer props at `width`,
/// its width model (profile / taper / wave) and caps/joins (a line body
/// outlined as a path).
pub fn set_width_model(node: &mut Map, stroke: &Stroke, width: f64) {
    let mut s = stroke.clone();
    s.width = width;
    s.align = None;
    s.miter_limit = None;
    s.dash = None;
    s.dash_offset = None;
    s.dash_alignment = None;
    s.dynamic = None;
    s.brush = None;
    s.start_arrow = None;
    s.end_arrow = None;
    s.arrow_scale = None;
    s.sides = None;
    set_strokes(node, vec![s]);
}

/// A no-repeat image fill at `scale` over its `natural` size, at the origin
/// (a generated graphic's cached asset).
pub fn set_image_fill(
    node: &mut Map,
    src: &str,
    scale: (f64, f64),
    natural: (f64, f64),
) {
    set_fill_paint(
        node,
        Paint::Image(ImagePaint {
            src: Some(src.to_string()),
            // Stretched over the node's box: a Crop relative to that box.
            placement: ImagePlacement {
                fit: Some("crop".into()),
                offset: Some(PaintPoint { x: 0.0, y: 0.0 }),
                scale: Some(PaintPoint {
                    x: scale.0,
                    y: scale.1,
                }),
                box_size: Some(PaintSize {
                    width: natural.0 * scale.0,
                    height: natural.1 * scale.1,
                }),
                natural_size: Some(PaintSize {
                    width: natural.0,
                    height: natural.1,
                }),
                ..Default::default()
            },
            adjust: None,
        }),
    );
}

/// Remove every stroke (a placed brush's own body: the artwork replaces the
/// stroke band).
pub fn clear_strokes(node: &mut Map) {
    node.remove("strokes");
}

/// Remove the brush from every stroke (a refused placement draws the plain
/// stroke).
pub fn clear_stroke_brush(node: &mut Map) {
    if let Some(Value::Array(strokes)) = node.get_mut("strokes") {
        for s in strokes.iter_mut() {
            if let Some(o) = s.as_object_mut() {
                o.remove("brush");
            }
        }
    }
}

/// Whether a node-shaped JSON object strokes: a visible stroke with a positive
/// width whose paint is not the transparent solid.
pub fn map_strokes_visibly(map: &Map) -> bool {
    top_stroke_of(map).is_some_and(|s| {
        s.width > 0.0 && !matches!(&s.paint, Paint::Solid(p) if p.color == "transparent")
    })
}

/// Multiply every stroke width of a node-shaped JSON object by `k` (a placed
/// brush copy at a scale); a non-finite result becomes 0.
pub fn scale_stroke_widths(map: &mut Map, k: f64) {
    if let Some(Value::Array(strokes)) = map.get_mut("strokes") {
        for s in strokes.iter_mut() {
            if let Some(o) = s.as_object_mut() {
                let w = o.get("width").and_then(Value::as_f64).unwrap_or(0.0) * k;
                o.insert("width".into(), json!(if w.is_finite() { w } else { 0.0 }));
            }
        }
    }
}

/// Remove the width model and brush (`profile`, `taper`, `wave`, `dynamic`,
/// `brush`) from every stroke of a node-shaped JSON object: a placed copy of
/// brush artwork strokes plainly.
pub fn strip_stroke_models(map: &mut Map) {
    if let Some(Value::Array(strokes)) = map.get_mut("strokes") {
        for s in strokes.iter_mut() {
            if let Some(o) = s.as_object_mut() {
                for k in ["profile", "taper", "wave", "dynamic", "brush"] {
                    o.remove(k);
                }
            }
        }
    }
}

/// The paint fields of the transient rectangle a page background expands to:
/// the page's `fills`, as stored (literal colours).
pub fn page_background_fields(page: &RawPage) -> Map {
    let mut m = Map::new();
    if let Some(fills) = page.0.get("fills") {
        m.insert("fills".into(), fills.clone());
    }
    m
}

// ---------------------------------------------------------------------------
// Colour walking (variable resolution, brush colourisation)
// ---------------------------------------------------------------------------

/// Which colour of a node's paint a [`map_paint_colors`] /
/// [`visit_paint_colors`] callback sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintColorSlot {
    /// A solid fill's colour (`fills[layer]`).
    Fill { layer: usize },
    /// A solid stroke's colour.
    Stroke { layer: usize },
    /// A solid glyph override (`textFills[layer]`).
    TextFill { layer: usize },
    /// An image or video fill's overlay colour.
    ImageOverlay { layer: usize },
    /// Stop `index` of a fill gradient (a freeform gradient's point `index`).
    FillStop { layer: usize, index: usize },
    /// Stop `index` of a stroke gradient.
    StrokeStop { layer: usize, index: usize },
    /// Stop `index` of a glyph-override gradient.
    TextFillStop { layer: usize, index: usize },
}

impl PaintColorSlot {
    /// A gradient stop (rather than a solid or overlay colour).
    pub fn is_stop(self) -> bool {
        matches!(
            self,
            PaintColorSlot::FillStop { .. }
                | PaintColorSlot::StrokeStop { .. }
                | PaintColorSlot::TextFillStop { .. }
        )
    }

    /// A stroke colour (solid or stop).
    pub fn is_stroke(self) -> bool {
        matches!(
            self,
            PaintColorSlot::Stroke { .. } | PaintColorSlot::StrokeStop { .. }
        )
    }
}

/// What a [`map_paint_colors`] callback does with one colour.
#[derive(Debug, Clone, PartialEq)]
pub enum ColorEdit {
    Keep,
    Set(Value),
    /// Remove a solid layer's colour (the layer then paints nothing; the
    /// renderer skips a paint that does not parse); a stop becomes `null`.
    Remove,
}

#[derive(Clone, Copy)]
enum List {
    Fills,
    Strokes,
    TextFills,
}

fn slot_for(list: List, layer: usize, stop: Option<usize>) -> PaintColorSlot {
    match (list, stop) {
        (List::Fills, None) => PaintColorSlot::Fill { layer },
        (List::Strokes, None) => PaintColorSlot::Stroke { layer },
        (List::TextFills, None) => PaintColorSlot::TextFill { layer },
        (List::Fills, Some(index)) => PaintColorSlot::FillStop { layer, index },
        (List::Strokes, Some(index)) => PaintColorSlot::StrokeStop { layer, index },
        (List::TextFills, Some(index)) => PaintColorSlot::TextFillStop { layer, index },
    }
}

/// Walk one paint object's colours in place.
fn walk_paint(
    paint: &mut Map,
    list: List,
    layer: usize,
    f: &mut dyn FnMut(PaintColorSlot, &mut Option<Value>),
) {
    let ty = paint
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    match ty.as_str() {
        "solid" => {
            let mut v = paint.remove("color");
            if v.is_some() {
                f(slot_for(list, layer, None), &mut v);
            }
            if let Some(v) = v {
                paint.insert("color".into(), v);
            }
        }
        "linear" | "radial" | "conic" | "diamond" | "freeform" => {
            // A freeform gradient's points walk as its stops (their colours).
            let key = if ty == "freeform" { "points" } else { "stops" };
            if let Some(Value::Array(stops)) = paint.get_mut(key) {
                for (index, stop) in stops.iter_mut().enumerate() {
                    let Some(stop) = stop.as_object_mut() else {
                        continue;
                    };
                    let mut v = Some(stop.remove("color").unwrap_or(Value::Null));
                    f(slot_for(list, layer, Some(index)), &mut v);
                    stop.insert("color".into(), v.unwrap_or(Value::Null));
                }
            }
        }
        "image" | "video" => {
            let overlay = paint
                .get_mut("adjust")
                .and_then(Value::as_object_mut)
                .and_then(|a| a.get_mut("overlay"))
                .and_then(Value::as_object_mut);
            if let Some(overlay) = overlay {
                let mut v = overlay.remove("color");
                if v.is_some() {
                    f(PaintColorSlot::ImageOverlay { layer }, &mut v);
                }
                if let Some(v) = v {
                    overlay.insert("color".into(), v);
                }
            }
        }
        _ => {}
    }
}

fn walk_colors(node: &mut Map, f: &mut dyn FnMut(PaintColorSlot, &mut Option<Value>)) {
    for (key, list) in [
        ("fills", List::Fills),
        ("strokes", List::Strokes),
        ("textFills", List::TextFills),
    ] {
        let Some(Value::Array(layers)) = node.get_mut(key) else {
            continue;
        };
        for (layer, entry) in layers.iter_mut().enumerate() {
            if let Some(paint) = entry.get_mut("paint").and_then(Value::as_object_mut) {
                walk_paint(paint, list, layer, f);
            }
        }
    }
}

/// Visit every stored colour of a node's paint (fills, strokes, rich-text
/// glyph overrides; solids, gradient stops, image overlays), in stack order.
pub fn visit_paint_colors(node: &Map, f: &mut dyn FnMut(PaintColorSlot, &Value)) {
    let mut copy = node.clone();
    walk_colors(&mut copy, &mut |slot, v| {
        if let Some(v) = v {
            f(slot, v);
        }
    });
}

/// Rewrite the stored colours of a node's paint in place, in
/// [`visit_paint_colors`] order. Brush colourisation goes through this.
pub fn map_paint_colors(
    node: &mut Map,
    f: &mut dyn FnMut(PaintColorSlot, &Value) -> ColorEdit,
) {
    walk_colors(node, &mut |slot, v| {
        let Some(current) = v.as_ref() else {
            return;
        };
        match f(slot, current) {
            ColorEdit::Keep => {}
            ColorEdit::Set(next) => *v = Some(next),
            ColorEdit::Remove => {
                *v = if slot.is_stop() {
                    Some(Value::Null)
                } else {
                    None
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_mode_stretches_the_image_onto_the_box() {
        // CSS `object-fit: fill`: a 100 × 50 image in a 300 × 300 box, each axis scaled to the box.
        let placement: ImagePlacement = crate::value::from_value(json!({"fit": "fill", "naturalSize": {"width": 100, "height": 50}})).unwrap();
        assert_eq!(placement.mode(), "fill");
        let r = placement.resolve([0.0, 0.0, 300.0, 300.0], None);
        assert_eq!(r.scale, (3.0, 6.0));
        assert_eq!(r.offset, (0.0, 0.0));
        assert_eq!(r.repeat, "no-repeat");
        // Mirrored on x: the image runs from the box's right edge towards −x.
        let flipped: ImagePlacement =
            crate::value::from_value(json!({"fit": "fill", "flip": {"x": true}, "naturalSize": {"width": 100, "height": 50}})).unwrap();
        let f = flipped.resolve([0.0, 0.0, 300.0, 300.0], None);
        assert_eq!(f.scale, (-3.0, 6.0));
        assert_eq!(f.offset, (300.0, 0.0));
    }

    #[test]
    fn per_side_weights_round_trip_on_the_stroke() {
        let strokes = json!([{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 4,
            "sides": {"top": 4, "bottom": 2}}]);
        let parsed = parse_strokes(Some(&strokes));
        let sides = parsed[0].sides.expect("sides");
        assert_eq!(sides.weights(), [4.0, 0.0, 2.0, 0.0]);
        assert_eq!(parsed[0].style().sides, Some(sides));
        let back = crate::value::to_value(&parsed[0]).unwrap();
        assert_eq!(back["sides"], json!({"top": 4.0, "right": 0.0, "bottom": 2.0, "left": 0.0}));
        assert!(crate::value::to_value(Stroke::solid("s", json!("#000"), 1.0)).unwrap().get("sides").is_none());
    }

    fn raw(v: Value) -> RawNode {
        RawNode::from_value(v).unwrap()
    }

    /// Numbers as f64 (serde writes a parsed `0` back as `0.0`).
    fn as_floats(v: &Value) -> Value {
        match v {
            Value::Number(n) => json!(n.as_f64().unwrap()),
            Value::Array(items) => Value::Array(items.iter().map(as_floats).collect()),
            // A shader's params and the width-model objects are kept verbatim (untyped bags).
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            if k == "params" || k == "taper" {
                                v.clone()
                            } else {
                                as_floats(v)
                            },
                        )
                    })
                    .collect(),
            ),
            other => other.clone(),
        }
    }

    #[test]
    fn stacks_round_trip_through_serde() {
        let fills = json!([
            {"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}},
            {"id": "fill-1", "visible": false, "opacity": 0.5, "blendMode": "multiply",
             "paint": {"type": "linear", "start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 0},
                       "stops": [{"offset": 0, "color": "#fff"}, {"offset": 1, "color": {"type": "variable", "variableId": "v"}}]}},
            {"id": "fill-2", "paint": {"type": "image", "src": "a.png",
             "placement": {"fit": "cover", "offset": {"x": 1, "y": 2}, "repeat": "no-repeat"},
             "adjust": {"exposure": 0.2, "overlay": {"enabled": true, "color": "#000", "opacity": 0.3}}}},
            {"id": "fill-3", "paint": {"type": "video", "src": "v.mp4", "posterSrc": "p.png",
             "playback": {"loop": true, "rate": 2}, "placement": {}}},
            {"id": "fill-4", "paint": {"type": "shader", "shaderId": "mesh", "params": {"a": 1}}},
            {"id": "fill-5", "paint": {"type": "diamond", "center": {"x": 50, "y": 50}, "radiusX": 10,
             "radiusY": 20, "rotation": 0, "stops": []}},
            {"id": "fill-6", "paint": {"type": "conic", "center": {"x": 50, "y": 50}, "angle": 90, "stops": []}},
            {"id": "fill-7", "paint": {"type": "freeform", "points": [
                {"x": 10, "y": 20, "color": "#f00", "spread": 30},
                {"x": 40, "y": 50, "color": {"type": "variable", "variableId": "v"}, "line": 1},
                {"x": 80, "y": 50, "color": "#00f", "line": 1}]}}
        ]);
        let parsed = parse_fills(Some(&fills));
        assert_eq!(parsed.len(), 8);
        assert_eq!(crate::value::to_value(&parsed).unwrap(), as_floats(&fills));
        let strokes = json!([{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 2,
            "cap": "round", "dash": [4, 2], "taper": {"startLength": 10}, "startArrow": "triangle",
            "arrowScale": 1.5, "gradientMode": "along"}]);
        let parsed = parse_strokes(Some(&strokes));
        assert_eq!(crate::value::to_value(&parsed).unwrap(), as_floats(&strokes));
        assert_eq!(parsed[0].gradient_mode(), StrokeGradientMode::Along);
        let mut s = parsed[0].clone();
        s.gradient_mode = Some("sideways".into());
        assert_eq!(s.gradient_mode(), StrokeGradientMode::Within, "unknown modes read as within");
    }

    #[test]
    fn parsed_fills_and_effects_follow_every_edit() {
        let solid = |c: &str| json!([{"id": "f", "paint": {"type": "solid", "color": c}}]);
        let shadow = |blur: f64| json!([{"id": "e", "type": "drop-shadow", "color": "#000000", "opacity": 0.3, "blur": blur, "offset": {"x": 0, "y": 4}}]);
        let color = |n: &RawNode| match top_fill(n).map(|f| f.paint) {
            Some(Paint::Solid(s)) => s.color,
            other => panic!("{other:?}"),
        };
        let blur = |n: &RawNode| crate::effect::drop_shadow(n).map(|s| s.blur);
        let mut n = raw(json!({"id": "n", "type": "rectangle", "fills": solid("#ff0000"), "effects": shadow(4.0)}));
        assert_eq!(color(&n), "#ff0000");
        assert_eq!(blur(&n), Some(4.0));
        let before = n.clone();
        n.set("fills", solid("#00ff00"));
        assert_eq!(color(&n), "#00ff00", "set");
        assert_eq!(color(&before), "#ff0000", "a clone keeps its own fills");
        *n.get_mut("effects").unwrap() = shadow(9.0);
        assert_eq!(blur(&n), Some(9.0), "get_mut");
        n.fields_mut().insert("fills".into(), solid("#0000ff"));
        assert_eq!(color(&n), "#0000ff", "fields_mut");
        n.remove("effects");
        assert_eq!(blur(&n), None, "remove");
        let moved = n.with_fields({
            let mut m = n.fields().clone();
            m.insert("fills".into(), solid("#ffffff"));
            m
        });
        assert_eq!(color(&moved), "#ffffff", "with_fields");
        n.set("type", json!("line"));
        assert!(node_fills(&n).is_empty(), "a type change re-gates the fills");
    }

    #[test]
    fn malformed_layers_are_skipped_and_types_gate() {
        let n = raw(json!({"id": "n", "type": "rectangle", "fills": [
            {"id": "a", "paint": {"type": "nope"}},
            {"paint": {"type": "solid", "color": "#fff"}},
            {"id": "b", "paint": {"type": "solid", "color": "#0f0"}}
        ]}));
        assert_eq!(node_fills(&n).len(), 1);
        let g = raw(json!({"id": "g", "type": "group", "fills": [{"id": "b", "paint": {"type": "solid", "color": "#0f0"}}]}));
        assert!(node_fills(&g).is_empty());
        let l = raw(json!({"id": "l", "type": "line", "fills": [{"id": "b", "paint": {"type": "solid", "color": "#0f0"}}]}));
        assert!(node_fills(&l).is_empty(), "lines have no fills");
    }

    #[test]
    fn freeform_point_colours_walk_as_stops() {
        let mut node = json!({"id": "n", "type": "path", "fills": [{"id": "f", "paint": {"type": "freeform",
            "points": [{"x": 0, "y": 0, "color": "#111111"}, {"x": 100, "y": 0, "color": "#222222", "line": 1}]}}]});
        let map = node.as_object_mut().unwrap();
        let mut seen = Vec::new();
        visit_paint_colors(map, &mut |slot, v| seen.push((slot, v.clone())));
        assert_eq!(
            seen,
            vec![
                (
                    PaintColorSlot::FillStop { layer: 0, index: 0 },
                    json!("#111111")
                ),
                (
                    PaintColorSlot::FillStop { layer: 0, index: 1 },
                    json!("#222222")
                ),
            ]
        );
        map_paint_colors(map, &mut |slot, _| match slot {
            PaintColorSlot::FillStop { index: 1, .. } => ColorEdit::Set(json!("#abcdef")),
            _ => ColorEdit::Keep,
        });
        assert_eq!(
            node["fills"][0]["paint"]["points"][1]["color"],
            json!("#abcdef")
        );
        assert_eq!(node["fills"][0]["paint"]["points"][1]["line"], json!(1));
        let raw = RawNode::from_value(node).unwrap();
        let fill = top_fill(&raw).unwrap();
        assert_eq!(representative_color(&fill.paint), Some("#111111"));
        assert_eq!(paint_stop_colors(&fill.paint).unwrap().len(), 2);
    }

    #[test]
    fn the_top_visible_layer_draws() {
        let n = raw(json!({"id": "n", "type": "ellipse",
            "fills": [
                {"id": "a", "paint": {"type": "solid", "color": "#111"}},
                {"id": "b", "paint": {"type": "solid", "color": "#222"}},
                {"id": "c", "visible": false, "paint": {"type": "solid", "color": "#333"}}
            ],
            "strokes": [
                {"id": "s", "paint": {"type": "solid", "color": "#444"}, "width": 3, "align": "inside"}
            ]}));
        assert_eq!(top_fill(&n).unwrap().id, "b");
        let style = node_stroke_style(&n);
        assert_eq!(style.width, Some(3.0));
        assert_eq!(style.align.as_deref(), Some("inside"));
        assert_eq!(container_body_gate(&n), (true, true));
        let bare = raw(json!({"id": "n", "type": "container"}));
        assert_eq!(container_body_gate(&bare), (false, false));
        assert!(!node_stroke_style(&bare).has_width());
    }

    #[test]
    fn a_container_with_only_a_gradient_stroke_draws_it() {
        let n = raw(json!({"id": "n", "type": "container", "strokes": [
            {"id": "s", "width": 4, "paint": {"type": "linear", "start": {"x": 0, "y": 0},
             "end": {"x": 100, "y": 0}, "stops": [{"offset": 0, "color": "#f00"}, {"offset": 1, "color": "#00f"}]}}
        ]}));
        assert_eq!(container_body_gate(&n), (false, true));
    }

    #[test]
    fn map_paint_colors_walks_every_layer() {
        let mut n = json!({"id": "n", "type": "rectangle",
            "fills": [
                {"id": "a", "paint": {"type": "solid", "color": "red"}},
                {"id": "b", "paint": {"type": "radial", "center": {"x": 50, "y": 50}, "radiusX": 50, "radiusY": 50,
                 "rotation": 0, "stops": [{"offset": 0, "color": "red"}, {"offset": 1, "color": "blue"}]}},
                {"id": "c", "paint": {"type": "image", "placement": {}, "adjust": {"overlay": {"color": "red"}}}}
            ],
            "strokes": [{"id": "s", "width": 1, "paint": {"type": "solid", "color": "red"}}]
        })
        .as_object()
        .cloned()
        .unwrap();
        let mut seen = Vec::new();
        visit_paint_colors(&n, &mut |slot, _| seen.push(slot));
        assert_eq!(
            seen,
            vec![
                PaintColorSlot::Fill { layer: 0 },
                PaintColorSlot::FillStop { layer: 1, index: 0 },
                PaintColorSlot::FillStop { layer: 1, index: 1 },
                PaintColorSlot::ImageOverlay { layer: 2 },
                PaintColorSlot::Stroke { layer: 0 },
            ]
        );
        map_paint_colors(&mut n, &mut |slot, v| {
            if v == "red" && !slot.is_stroke() {
                ColorEdit::Set(json!("#ff0000"))
            } else if v == "blue" {
                ColorEdit::Remove
            } else {
                ColorEdit::Keep
            }
        });
        assert_eq!(n["fills"][0]["paint"]["color"], "#ff0000");
        assert_eq!(n["fills"][1]["paint"]["stops"][0]["color"], "#ff0000");
        assert_eq!(n["fills"][1]["paint"]["stops"][1]["color"], Value::Null);
        assert_eq!(n["fills"][2]["paint"]["adjust"]["overlay"]["color"], "#ff0000");
        assert_eq!(n["strokes"][0]["paint"]["color"], "red");
    }


    /// `matrix` takes precedence over x / y / scale / rotation when usable (`patterns.ts` `patternMatrix`); a
    /// singular or non-finite one falls back to them; and a transform without one prints as before it existed.
    #[test]
    fn a_pattern_transform_matrix_takes_precedence() {
        let t: PatternTransform = crate::value::from_value(json!({"x": 5, "scale": 3, "matrix": [2, 0, 0.5, 1, 7, -3]})).unwrap();
        let a = t.affine();
        assert_eq!((a.a, a.b, a.c, a.d, a.e, a.f), (2.0, 0.0, 0.5, 1.0, 7.0, -3.0));
        // A reflection is usable (negative determinant).
        let r: PatternTransform = crate::value::from_value(json!({"matrix": [-1, 0, 0, 1, 10, 0]})).unwrap();
        assert_eq!(r.affine().a, -1.0);
        // Singular: the plain fields.
        let s: PatternTransform = crate::value::from_value(json!({"x": 5, "scale": 3, "matrix": [1, 2, 2, 4, 0, 0]})).unwrap();
        let a = s.affine();
        assert_eq!((a.a, a.d, a.e), (3.0, 3.0, 5.0));
        assert!(s.usable_matrix().is_none());
        // Round trip, and the Debug form of a matrix-free transform is unchanged.
        let back = crate::value::to_value(t).unwrap();
        assert_eq!(back["matrix"], json!([2.0, 0.0, 0.5, 1.0, 7.0, -3.0]));
        let plain: PatternTransform = crate::value::from_value(json!({"scale": 2})).unwrap();
        assert!(crate::value::to_value(plain).unwrap().get("matrix").is_none());
        assert_eq!(format!("{plain:?}"), "PatternTransform { x: None, y: None, scale: Some(2.0), rotation: None }");
        let paint: PatternPaint = crate::value::from_value(json!({"patternId": "p", "transform": {"matrix": [1, 0, -1, 1, 0, 0]}})).unwrap();
        assert_eq!(paint.affine().c, -1.0);
    }
}

#[cfg(test)]
mod gradient_options_tests {
    use super::*;

    #[test]
    fn gradient_options_parse_flatten_and_pack_flags() {
        let p: Paint = crate::value::from_value(json!({
            "type": "linear", "start": {"x": 0, "y": 50}, "end": {"x": 100, "y": 50},
            "stops": [{"offset": 0, "color": "#f00"}, {"offset": 1, "color": "#00f"}],
            "interpolation": "oklch", "dither": true
        }))
        .unwrap();
        let o = p.gradient_options();
        assert_eq!(o.interpolation, Some(GradientInterpolation::Oklch));
        assert_eq!(o.shader_flags(), 11.0);
        // Round trip keeps the fields beside the stops.
        let v = crate::value::to_value(&p).unwrap();
        assert_eq!(v["interpolation"], "oklch");
        assert_eq!(v["dither"], true);
        // Absent = the original ramp; an unknown space or a non-boolean dither is ignored, not an error.
        let q: Paint = crate::value::from_value(json!({
            "type": "radial", "center": {"x": 50, "y": 50}, "radiusX": 50, "radiusY": 50, "rotation": 0,
            "stops": [], "interpolation": "hsl", "dither": "yes"
        }))
        .unwrap();
        assert_eq!(q.gradient_options(), GradientOptions::default());
        assert_eq!(q.gradient_options().shader_flags(), 0.0);
        for (name, code) in [("srgb", 0.0), ("srgb-linear", 1.0), ("oklab", 2.0), ("oklch", 3.0)] {
            let c: Paint = crate::value::from_value(json!({
                "type": "conic", "center": {"x": 50, "y": 50}, "angle": 0, "stops": [], "interpolation": name
            }))
            .unwrap();
            assert_eq!(c.gradient_options().shader_flags(), code, "{name}");
        }
    }
}
