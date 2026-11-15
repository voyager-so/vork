//! Effect stacks (E-M.7, docs/roadmap/design-editor/data-model-audit.md
//! §2.2): the typed `Effect` a node's `effects` list holds, and the helpers
//! the renderers read it through.
//!
//! The mirror of `packages/editor/src/core/scene/effect.ts` (the authority
//! for the JSON shape). One list replaced the flat drop shadow
//! (`shadow*`), the flat inner shadow (`innerShadow*`), `backgroundBlur`,
//! `layerBlur`, `shaderEffects[]` and the rich-text `effects[]` (whose
//! `enabled` flag is now `visible`).
//!
//! Every entry has an `id` and `visible` (absent = visible; only a strict
//! `false` hides it). Order is back to front. A node draws every visible
//! drop shadow, inner shadow, glow, echo and extrude, and every visible
//! backdrop effect, stacked in list order ([`backdrops_json`]; the filter
//! kinds' params are `crate::backdrop_kinds`), but ONE layer blur: the first
//! visible one ([`layer_blur`]).
//!
//! `tests/flat_paint_keys.rs` fails when any file in the renderer crates
//! names one of the flat keys this model replaced.

use serde::{Deserialize, Serialize};
use crate::value::{Map, Value};

use crate::types::RawNode;

/// An offset in canvas px.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct EffectOffset {
    #[serde(default)]
    pub x: f64,
    #[serde(default)]
    pub y: f64,
}

impl EffectOffset {
    pub fn is_zero(&self) -> bool {
        self.x == 0.0 && self.y == 0.0
    }
}

/// A drop shadow. `color` is a `ColorValue` (a CSS string, or a variable
/// reference resolved before rendering), so it stays raw JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DropShadow {
    pub color: Value,
    #[serde(default)]
    pub blur: f64,
    #[serde(default)]
    pub offset: EffectOffset,
    /// Grows (negative: shrinks) the silhouette before the blur. Rectangles,
    /// ellipses and frames only, as in Figma (the SDF lane); the capture
    /// lane ignores it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spread: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_behind_node: Option<bool>,
}

impl DropShadow {
    /// `opacity ?? 1`.
    pub fn opacity(&self) -> f64 {
        self.opacity.unwrap_or(1.0)
    }

    /// `spread ?? 0`.
    pub fn spread(&self) -> f64 {
        self.spread.unwrap_or(0.0)
    }

    /// `showBehindNode === true`.
    pub fn shows_behind_node(&self) -> bool {
        self.show_behind_node == Some(true)
    }
}

/// CSS `filter: drop-shadow()`, a layer filter (`filter-shadow`): the layer's
/// own rendered picture casts it. The picture's alpha, moved by `offset` and
/// blurred, is tinted with `color` and drawn under the picture, so
/// translucent paint casts a proportionally weaker shadow, a group's shadow
/// follows its children's filters, masks and opacity, and the shadow shows
/// through translucent paint. A `drop-shadow` is Figma's instead: cast from
/// the layer's shape with every fill and stroke opaque.
///
/// It runs first among the layer filters, so a layer blur blurs the shadow
/// with the picture (CSS `drop-shadow() blur()`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FilterShadow {
    pub color: Value,
    /// The blur radius, node-local px: a Gaussian of σ = blur / 2, as every
    /// blur here ([`BLUR_REACH_PER_RADIUS`]). CSS gives the function σ itself.
    #[serde(default)]
    pub blur: f64,
    /// Canvas px, as a drop shadow's.
    #[serde(default)]
    pub offset: EffectOffset,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
}

impl FilterShadow {
    /// `opacity ?? 1`.
    pub fn opacity(&self) -> f64 {
        self.opacity.unwrap_or(1.0)
    }
}

/// An inner (inset) shadow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InnerShadow {
    pub color: Value,
    #[serde(default)]
    pub blur: f64,
    #[serde(default)]
    pub offset: EffectOffset,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spread: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
}

impl InnerShadow {
    /// `opacity ?? 1`.
    pub fn opacity(&self) -> f64 {
        self.opacity.unwrap_or(1.0)
    }

    /// `spread ?? 0`.
    pub fn spread(&self) -> f64 {
        self.spread.unwrap_or(0.0)
    }
}

/// Figma's progressive blur (`ProgressiveBlur` in TS): the radius ramps
/// from `start_radius` at `start` to the blur's `radius` at `end`, along the
/// start→end line, constant beyond either point. The points are fractions of
/// the node's bounds (0..1).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressiveBlur {
    #[serde(default)]
    pub start_radius: f64,
    #[serde(default)]
    pub start: EffectOffset,
    #[serde(default = "progressive_end_default")]
    pub end: EffectOffset,
}

fn progressive_end_default() -> EffectOffset {
    EffectOffset { x: 0.0, y: 1.0 }
}

/// How far a blur of radius R reaches, in units of R: a blur's radius follows
/// Figma's convention, a Gaussian of σ = R / 2 (Figma exports a layer blur R as
/// CSS `filter: blur(R/2)`, and a shadow's blur B is CSS `box-shadow`'s, also
/// σ = B / 2), and the renderers size their kernels by the 3σ reach.
/// Layer blur, background blur (uniform and progressive) and glass frost use it.
pub const BLUR_REACH_PER_RADIUS: f64 = 1.5;

/// A blur (layer blur, background blur): uniform, or progressive when
/// `progressive` is set (`radius` is then the END radius). σ = radius / 2
/// ([`BLUR_REACH_PER_RADIUS`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RadiusBlur {
    #[serde(default)]
    pub radius: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progressive: Option<ProgressiveBlur>,
}

impl RadiusBlur {
    /// The larger of the two end radii (what sizes a capture's padding).
    pub fn max_radius(&self) -> f64 {
        match self.progressive {
            Some(p) => self.radius.max(p.start_radius),
            None => self.radius,
        }
    }

    /// Whether the blur changes any pixel (`blurRadiusIsActive`).
    pub fn is_active(&self) -> bool {
        self.max_radius() > 0.0
    }
}

/// Figma's noise (`NoiseEffect` in TS): cells of `size` output px over the
/// node's own pixels, each painting at a ramp of its random value, more of
/// them as `density` rises (`layer-filter-noise.wgsl`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Noise {
    /// `mono` | `duo` | `multi`.
    pub mode: String,
    pub color: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secondary_color: Option<Value>,
    #[serde(default)]
    pub size: f64,
    #[serde(default)]
    pub density: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
}

impl Noise {
    /// `opacity ?? 1`.
    pub fn opacity(&self) -> f64 {
        self.opacity.unwrap_or(1.0)
    }

    /// Whether it paints any cell (`noiseIsActive`).
    pub fn is_active(&self) -> bool {
        if !(self.size > 0.0) || self.opacity() <= 0.0 {
            return false;
        }
        self.density > 0.0
            || (self.mode == "duo" && self.secondary_color.as_ref().is_some_and(color_truthy))
    }
}

/// Figma's texture (`TextureEffect` in TS): a grain that displaces the
/// node's own pixels by up to `radius` canvas px.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Texture {
    #[serde(default)]
    pub size: f64,
    #[serde(default)]
    pub radius: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_to_shape: Option<bool>,
}

impl Texture {
    /// Whether it moves any pixel (`textureIsActive`).
    pub fn is_active(&self) -> bool {
        self.radius > 0.0 && self.size > 0.0
    }

    /// `clipToShape === true`.
    pub fn clips_to_shape(&self) -> bool {
        self.clip_to_shape == Some(true)
    }
}

/// Illustrator's Feather (Stylize): the layer's edges fade to transparent
/// INSIDE its outline over `radius` canvas px; its interior stays opaque. The
/// layer filter blurs the layer's hard alpha `min(1, 127·a)` (σ = radius / 2)
/// and scales every pixel by `clamp(2·blur − 1, 0, 1)`
/// (`layer-filter-shadow.wgsl` `fs_feather`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Feather {
    #[serde(default)]
    pub radius: f64,
}

impl Feather {
    /// Whether it changes any pixel (`featherIsActive`).
    pub fn is_active(&self) -> bool {
        self.radius > 0.0
    }
}

/// Liquid glass (a backdrop effect).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Glass {
    #[serde(default)]
    pub frost: f64,
    #[serde(default)]
    pub refraction: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refraction_style: Option<String>,
    #[serde(default)]
    pub depth: f64,
    #[serde(default)]
    pub light_angle: f64,
    #[serde(default)]
    pub light_intensity: f64,
    #[serde(default)]
    pub dispersion: f64,
    #[serde(default)]
    pub splay: f64,
}

/// Pixelate (a backdrop effect).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Pixelate {
    #[serde(default)]
    pub pixel_size: f64,
}

/// Dither (a backdrop effect).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Dither {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    #[serde(default)]
    pub brightness: f64,
    #[serde(default)]
    pub contrast: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pixel_size: Option<f64>,
}

/// Roughen (Illustrator's Effect ▸ Distort & Transform ▸ Roughen): anchors
/// added along the path at `detail` per inch, every anchor moved by a seeded
/// random offset of at most `size`. Geometry: [`crate::live_effects`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Roughen {
    /// Largest displacement: a percentage of the art's larger side
    /// (`sizeMode` relative, the default) or canvas px (absolute).
    #[serde(default)]
    pub size: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_mode: Option<String>,
    /// Anchors added per inch (72 px) of path.
    #[serde(default)]
    pub detail: f64,
    /// `smooth` (default) or `corner`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub points: Option<String>,
    /// Random seed; the same seed always gives the same shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<f64>,
}

/// Zig Zag: `ridges` extra anchors per segment, moved alternately either side
/// of the path by `size`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ZigZag {
    #[serde(default)]
    pub size: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_mode: Option<String>,
    #[serde(default)]
    pub ridges: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub points: Option<String>,
}

/// Pucker & Bloat: `amount` in % (−200..200). Negative puckers (anchors out,
/// segments in), positive bloats (anchors in, segments out).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PuckerBloat {
    #[serde(default)]
    pub amount: f64,
}

/// Twist: rotates the art about its centre, `angle` degrees at the centre
/// falling to 0 at the corners of its bounds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Twist {
    #[serde(default)]
    pub angle: f64,
}

/// Transform (with copies): scale / move / rotate / reflect about a reference
/// point of the art's bounds; `copies` > 0 keeps the original and adds that
/// many copies, each transformed once more than the last.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TransformEffect {
    /// Percent; absent = 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_x: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_y: Option<f64>,
    /// Canvas px.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub move_x: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub move_y: Option<f64>,
    /// Degrees, clockwise on screen (as a node's `rotation`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotate: Option<f64>,
    /// Reflect across the horizontal axis (y flips), as Illustrator's Reflect X.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reflect_x: Option<bool>,
    /// Reflect across the vertical axis (x flips), as Illustrator's Reflect Y.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reflect_y: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub copies: Option<f64>,
    /// One of the 9 reference points (`top-left` … `bottom-right`); absent = `center`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// Illustrator's Random: each copy's scale, move and rotation are drawn
    /// between none and the value, from `seed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub random: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<f64>,
}

/// Scribble (Stylize): the fill (and the stroke's band) redrawn as one
/// back-and-forth stroke of hatch lines. Geometry: `live_effects_distort`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScribbleEffect {
    /// Hatch direction, degrees counter-clockwise from horizontal (Illustrator's dialog).
    #[serde(default)]
    pub angle: f64,
    /// How far the strokes run past the edge (negative: stop inside it), px.
    #[serde(default)]
    pub path_overlap: f64,
    #[serde(default)]
    pub path_overlap_variation: f64,
    #[serde(default)]
    pub stroke_width: f64,
    /// 0..100 %: how far the turns loop.
    #[serde(default)]
    pub curviness: f64,
    #[serde(default)]
    pub curviness_variation: f64,
    /// Distance between hatch lines, px.
    #[serde(default)]
    pub spacing: f64,
    #[serde(default)]
    pub spacing_variation: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<f64>,
    /// The preset the settings came from (a label; the numbers are what draw).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
}

/// Tweak (Distort & Transform): anchors and / or handles moved by seeded random
/// offsets up to `horizontal` / `vertical` (percent of the art's width /
/// height, or px when `sizeMode` is absolute).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TweakEffect {
    #[serde(default)]
    pub horizontal: f64,
    #[serde(default)]
    pub vertical: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_mode: Option<String>,
    /// Modify anchor points; absent = true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchors: Option<bool>,
    /// Modify "In" control points (the handle before each anchor); absent = true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_handles: Option<bool>,
    /// Modify "Out" control points (the handle after each anchor); absent = true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_handles: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<f64>,
}

/// Free Distort (Distort & Transform): the art's bounds moved onto four
/// corners, given in the bounds' unit space (0,0 = top-left, 1,1 =
/// bottom-right), in the order top-left, top-right, bottom-right, bottom-left.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FreeDistortEffect {
    #[serde(default)]
    pub corners: Vec<[f64; 2]>,
}

/// Convert to Shape: the art replaced by a rectangle, rounded rectangle or
/// ellipse over its bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConvertToShapeEffect {
    /// `rectangle` (default), `rounded-rectangle` or `ellipse`.
    #[serde(default)]
    pub shape: String,
    /// `relative` (default: the bounds plus `extraWidth` / `extraHeight` each
    /// side) or `absolute` (`width` × `height` about the centre).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_width: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_height: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corner_radius: Option<f64>,
}

/// Round Corners (Stylize): every corner rounded by `radius` canvas px.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RoundCornersEffect {
    #[serde(default)]
    pub radius: f64,
}

/// Offset Path (Path): grown (positive) or shrunk by `offset` canvas px.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OffsetPathEffect {
    #[serde(default)]
    pub offset: f64,
    /// `miter` (default), `round` or `bevel`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub join: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub miter_limit: Option<f64>,
}

/// Warp: one of Illustrator's 15 envelope presets over the art's bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WarpEffect {
    /// `arc`, `arc-lower`, `arc-upper`, `arch`, `bulge`, `shell-lower`,
    /// `shell-upper`, `flag`, `wave`, `fish`, `rise`, `fisheye`, `inflate`,
    /// `squeeze`, `twist`.
    #[serde(default)]
    pub style: String,
    /// Percent, −100..100.
    #[serde(default)]
    pub bend: f64,
    /// Horizontal / vertical distortion, percent, −100..100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub horizontal: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertical: Option<f64>,
    /// `horizontal` (default) or `vertical`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orientation: Option<String>,
}

/// What an effect draws. Tagged by `type` (kebab-case, as in TS).
///
/// `Shader`, `Glow`, `InnerGlow`, `Echo` and `Extrude` keep their params as stored JSON:
/// their readers (`shader_effects`, `moda-text`) parse them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum EffectKind {
    DropShadow(DropShadow),
    InnerShadow(InnerShadow),
    LayerBlur(RadiusBlur),
    Noise(Noise),
    Texture(Texture),
    Feather(Feather),
    FilterShadow(FilterShadow),
    BackgroundBlur(RadiusBlur),
    Glass(Glass),
    Pixelate(Pixelate),
    Dither(Dither),
    // The backdrop filter kinds (live-nodes stage 7b): their params are schema-driven
    // (`crate::backdrop_kinds`), so they keep the stored JSON like the shader and text effects.
    Grain(Map),
    Vignette(Map),
    Glaze(Map),
    Displace(Map),
    Grade(Map),
    // The adjustment kinds (adjustment-layers.md): schema-driven too (`crate::backdrop_kinds::ADJUSTMENT_KINDS`).
    Levels(Map),
    Curves(Map),
    BrightnessContrast(Map),
    HueSaturation(Map),
    ColorBalance(Map),
    Vibrance(Map),
    Invert(Map),
    Desaturate(Map),
    GradientMap(Map),
    ChannelMixer(Map),
    Shader(Map),
    Glow(Map),
    InnerGlow(Map),
    Echo(Map),
    Extrude(Map),
    // Live geometry effects (E8.1): they reshape the outline before any fill or
    // stroke draws. See `crate::live_effects`.
    Roughen(Roughen),
    ZigZag(ZigZag),
    PuckerBloat(PuckerBloat),
    Twist(Twist),
    Transform(TransformEffect),
    RoundCorners(RoundCornersEffect),
    OffsetPath(OffsetPathEffect),
    OutlineStroke(Map),
    Warp(WarpEffect),
    Scribble(ScribbleEffect),
    Tweak(TweakEffect),
    FreeDistort(FreeDistortEffect),
    ConvertToShape(ConvertToShapeEffect),
}

/// One entry of a node's `effects`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Effect {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
    #[serde(flatten)]
    pub kind: EffectKind,
}

impl Effect {
    /// Only `visible: false` hides an effect.
    pub fn is_visible(&self) -> bool {
        self.visible != Some(false)
    }

    /// The stored `type` tag.
    pub fn type_name(&self) -> &'static str {
        match self.kind {
            EffectKind::DropShadow(_) => "drop-shadow",
            EffectKind::InnerShadow(_) => "inner-shadow",
            EffectKind::LayerBlur(_) => "layer-blur",
            EffectKind::Noise(_) => "noise",
            EffectKind::Texture(_) => "texture",
            EffectKind::Feather(_) => "feather",
            EffectKind::FilterShadow(_) => "filter-shadow",
            EffectKind::BackgroundBlur(_) => "background-blur",
            EffectKind::Glass(_) => "glass",
            EffectKind::Pixelate(_) => "pixelate",
            EffectKind::Dither(_) => "dither",
            EffectKind::Grain(_) => "grain",
            EffectKind::Vignette(_) => "vignette",
            EffectKind::Glaze(_) => "glaze",
            EffectKind::Displace(_) => "displace",
            EffectKind::Grade(_) => "grade",
            EffectKind::Levels(_) => "levels",
            EffectKind::Curves(_) => "curves",
            EffectKind::BrightnessContrast(_) => "brightness-contrast",
            EffectKind::HueSaturation(_) => "hue-saturation",
            EffectKind::ColorBalance(_) => "color-balance",
            EffectKind::Vibrance(_) => "vibrance",
            EffectKind::Invert(_) => "invert",
            EffectKind::Desaturate(_) => "desaturate",
            EffectKind::GradientMap(_) => "gradient-map",
            EffectKind::ChannelMixer(_) => "channel-mixer",
            EffectKind::Shader(_) => "shader",
            EffectKind::Glow(_) => "glow",
            EffectKind::InnerGlow(_) => "inner-glow",
            EffectKind::Echo(_) => "echo",
            EffectKind::Extrude(_) => "extrude",
            EffectKind::Roughen(_) => "roughen",
            EffectKind::ZigZag(_) => "zig-zag",
            EffectKind::PuckerBloat(_) => "pucker-bloat",
            EffectKind::Twist(_) => "twist",
            EffectKind::Transform(_) => "transform",
            EffectKind::RoundCorners(_) => "round-corners",
            EffectKind::OffsetPath(_) => "offset-path",
            EffectKind::OutlineStroke(_) => "outline-stroke",
            EffectKind::Warp(_) => "warp",
            EffectKind::Scribble(_) => "scribble",
            EffectKind::Tweak(_) => "tweak",
            EffectKind::FreeDistort(_) => "free-distort",
            EffectKind::ConvertToShape(_) => "convert-to-shape",
        }
    }

    pub fn is_backdrop(&self) -> bool {
        matches!(
            self.kind,
            EffectKind::BackgroundBlur(_)
                | EffectKind::Glass(_)
                | EffectKind::Pixelate(_)
                | EffectKind::Dither(_)
                | EffectKind::Grain(_)
                | EffectKind::Vignette(_)
                | EffectKind::Glaze(_)
                | EffectKind::Displace(_)
                | EffectKind::Grade(_)
                | EffectKind::Levels(_)
                | EffectKind::Curves(_)
                | EffectKind::BrightnessContrast(_)
                | EffectKind::HueSaturation(_)
                | EffectKind::ColorBalance(_)
                | EffectKind::Vibrance(_)
                | EffectKind::Invert(_)
                | EffectKind::Desaturate(_)
                | EffectKind::GradientMap(_)
                | EffectKind::ChannelMixer(_)
        )
    }

    /// A live geometry effect (reshapes the outline; `crate::live_effects`).
    pub fn is_geometry(&self) -> bool {
        GEOMETRY_EFFECT_TYPES.contains(&self.type_name())
    }
}

/// The live geometry effect types (E8.1), in the TS `GEOMETRY_EFFECT_TYPES` order.
pub const GEOMETRY_EFFECT_TYPES: &[&str] = &[
    "roughen",
    "zig-zag",
    "pucker-bloat",
    "twist",
    "transform",
    "round-corners",
    "offset-path",
    "outline-stroke",
    "warp",
    "scribble",
    "tweak",
    "free-distort",
    "convert-to-shape",
];

/// The backdrop effect types: the four original kinds, the filter kinds
/// ([`crate::backdrop_kinds::FILTER_KINDS`]), then the adjustment kinds ([`crate::backdrop_kinds::ADJUSTMENT_KINDS`]).
pub const BACKDROP_EFFECT_TYPES: &[&str] = &[
    "background-blur",
    "glass",
    "pixelate",
    "dither",
    "grain",
    "vignette",
    "glaze",
    "displace",
    "grade",
    "levels",
    "curves",
    "brightness-contrast",
    "hue-saturation",
    "color-balance",
    "vibrance",
    "invert",
    "desaturate",
    "gradient-map",
    "channel-mixer",
];

/// Whether a raw JSON entry is visible (`isEffectVisible`): only a strict
/// `false` hides it.
pub fn is_visible_json(effect: &Value) -> bool {
    effect.get("visible") != Some(&Value::Bool(false))
}

/// An effect list from its JSON value, skipping any entry that is not a
/// well-formed effect (a hand-edited or truncated document never panics the
/// renderer).
pub fn parse_effects(v: Option<&Value>) -> Vec<Effect> {
    match v {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|i| Effect::deserialize(i).ok())
            .collect(),
        _ => Vec::new(),
    }
}

/// A node's effect list, back to front.
pub fn node_effects(node: &RawNode) -> Vec<Effect> {
    node_effects_shared(node).to_vec()
}

/// [`node_effects`], shared: parsed once per node (`RawNode` keeps it until its fields change).
pub fn node_effects_shared(node: &RawNode) -> std::sync::Arc<[Effect]> {
    node.effects_memo(|| parse_effects(node.json_field("effects")))
}

/// The raw JSON entries of the node's `effects` (visible or not), for readers
/// that return references into the node.
pub fn node_effects_json(node: &RawNode) -> &[Value] {
    match node.json_field("effects") {
        Some(Value::Array(items)) => items,
        _ => &[],
    }
}

/// The raw JSON of the node's visible effects of `type_name`, in list order.
pub fn visible_effects_json<'a>(
    node: &'a RawNode,
    type_name: &'a str,
) -> impl Iterator<Item = &'a Map> + 'a {
    node_effects_json(node)
        .iter()
        .filter(move |e| {
            e.get("type").and_then(Value::as_str) == Some(type_name) && is_visible_json(e)
        })
        .filter_map(Value::as_object)
}

/// Whether a drop shadow draws: a colour, and a blur, a spread or an offset
/// (`drawsDropShadow`).
pub fn drop_shadow_draws(s: &DropShadow) -> bool {
    color_truthy(&s.color) && (s.blur > 0.0 || s.spread() != 0.0 || !s.offset.is_zero())
}

/// Whether an inner shadow draws: a colour, not fully transparent, and a
/// blur, a spread or an offset (`innerShadowDraws`).
pub fn inner_shadow_draws(s: &InnerShadow) -> bool {
    color_truthy(&s.color)
        && s.opacity != Some(0.0)
        && (s.blur > 0.0 || s.spread.unwrap_or(0.0) != 0.0 || !s.offset.is_zero())
}

/// JS truthiness of a colour value (a non-empty string or a variable
/// reference object).
fn color_truthy(v: &Value) -> bool {
    match v {
        Value::String(s) => !s.is_empty(),
        Value::Object(_) => true,
        _ => false,
    }
}

/// The node's drop shadows that draw, back to front.
pub fn drop_shadows(node: &RawNode) -> Vec<DropShadow> {
    node_effects_shared(node)
        .iter()
        .filter(|e| e.is_visible())
        .filter_map(|e| match &e.kind {
            EffectKind::DropShadow(s) if drop_shadow_draws(s) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

/// The node's inner shadows that draw, back to front.
pub fn inner_shadows(node: &RawNode) -> Vec<InnerShadow> {
    node_effects_shared(node)
        .iter()
        .filter(|e| e.is_visible())
        .filter_map(|e| match &e.kind {
            EffectKind::InnerShadow(s) if inner_shadow_draws(s) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

/// The first drop shadow that draws (single-shadow lanes).
pub fn drop_shadow(node: &RawNode) -> Option<DropShadow> {
    drop_shadows(node).into_iter().next()
}

/// The first inner shadow that draws (single-shadow lanes).
pub fn inner_shadow(node: &RawNode) -> Option<InnerShadow> {
    inner_shadows(node).into_iter().next()
}

/// The backdrop effect the node draws: its first visible background blur,
/// glass, pixelate or dither.
pub fn backdrop(node: &RawNode) -> Option<Effect> {
    node_effects_shared(node)
        .iter()
        .find(|e| e.is_backdrop() && e.is_visible())
        .cloned()
}

/// The raw JSON of every visible backdrop entry, in list order (back to front):
/// a node's backdrops stack, each applied to what the one before it left.
pub fn backdrops_json(node: &RawNode) -> Vec<&Map> {
    node_effects_json(node)
        .iter()
        .filter(|e| {
            e.get("type")
                .and_then(Value::as_str)
                .is_some_and(|t| BACKDROP_EFFECT_TYPES.contains(&t))
                && is_visible_json(e)
        })
        .filter_map(Value::as_object)
        .collect()
}

/// The raw JSON of the node's first visible backdrop entry (see [`backdrop`]).
pub fn backdrop_json(node: &RawNode) -> Option<&Map> {
    node_effects_json(node)
        .iter()
        .find(|e| {
            e.get("type")
                .and_then(Value::as_str)
                .is_some_and(|t| BACKDROP_EFFECT_TYPES.contains(&t))
                && is_visible_json(e)
        })
        .and_then(Value::as_object)
}

/// The layer blur the node draws: its first visible one.
pub fn layer_blur(node: &RawNode) -> Option<RadiusBlur> {
    node_effects_shared(node)
        .iter()
        .filter(|e| e.is_visible())
        .find_map(|e| match &e.kind {
            EffectKind::LayerBlur(b) => Some(b.clone()),
            _ => None,
        })
}

/// Whether a backdrop effect changes any pixel (`backdropIsActive`).
pub fn backdrop_is_active(e: &Effect) -> bool {
    if !e.is_visible() {
        return false;
    }
    match &e.kind {
        EffectKind::BackgroundBlur(b) => b.is_active(),
        EffectKind::Pixelate(p) => p.pixel_size > 0.0,
        EffectKind::Dither(d) => d.pixel_size.unwrap_or(1.0) > 0.0,
        EffectKind::Glass(g) => {
            g.frost > 0.0
                || (g.refraction > 0.0 && g.depth > 0.0)
                || g.light_intensity > 0.0
                || (g.dispersion > 0.0 && g.depth > 0.0)
        }
        EffectKind::Grain(m)
        | EffectKind::Vignette(m)
        | EffectKind::Glaze(m)
        | EffectKind::Displace(m)
        | EffectKind::Grade(m)
        | EffectKind::Levels(m)
        | EffectKind::Curves(m)
        | EffectKind::BrightnessContrast(m)
        | EffectKind::HueSaturation(m)
        | EffectKind::ColorBalance(m)
        | EffectKind::Vibrance(m)
        | EffectKind::Invert(m)
        | EffectKind::Desaturate(m)
        | EffectKind::GradientMap(m)
        | EffectKind::ChannelMixer(m)
        => crate::backdrop_kinds::kind(e.type_name()).is_some_and(|k| k.is_active(m)),
        _ => false,
    }
}

/// Whether the node has a layer blur that changes any pixel.
pub fn has_active_layer_blur(node: &RawNode) -> bool {
    layer_blur(node).is_some_and(|b| b.is_active())
}

/// The node's filter shadows that draw, back to front: every visible one with
/// a colour and an opacity. One with no blur and no offset draws too: it sits
/// exactly under the picture and shows through translucent paint.
pub fn filter_shadows(node: &RawNode) -> Vec<FilterShadow> {
    node_effects_shared(node)
        .iter()
        .filter(|e| e.is_visible())
        .filter_map(|e| match &e.kind {
            EffectKind::FilterShadow(s) if color_truthy(&s.color) && s.opacity() > 0.0 => Some(s.clone()),
            _ => None,
        })
        .collect()
}

/// The node's noise effects that paint, back to front (every one draws).
pub fn noise_effects(node: &RawNode) -> Vec<Noise> {
    node_effects_shared(node)
        .iter()
        .filter(|e| e.is_visible())
        .filter_map(|e| match &e.kind {
            EffectKind::Noise(n) if n.is_active() => Some(n.clone()),
            _ => None,
        })
        .collect()
}

/// The texture the node draws: its first visible one, if it moves pixels.
pub fn texture(node: &RawNode) -> Option<Texture> {
    node_effects_shared(node)
        .iter()
        .filter(|e| e.is_visible())
        .find_map(|e| match &e.kind {
            EffectKind::Texture(t) => Some(t.clone()),
            _ => None,
        })
        .filter(Texture::is_active)
}

/// The feather the node draws: its first visible one, if it changes pixels.
pub fn feather(node: &RawNode) -> Option<Feather> {
    node_effects_shared(node)
        .iter()
        .filter(|e| e.is_visible())
        .find_map(|e| match &e.kind {
            EffectKind::Feather(f) => Some(f.clone()),
            _ => None,
        })
        .filter(Feather::is_active)
}

/// Whether the node carries a layer filter that changes pixels: an active
/// layer blur (uniform or progressive), noise, texture, feather or filter
/// shadow (`hasLayerFilter`).
pub fn has_active_layer_filter(node: &RawNode) -> bool {
    has_active_layer_blur(node)
        || !filter_shadows(node).is_empty()
        || !noise_effects(node).is_empty()
        || texture(node).is_some()
        || feather(node).is_some()
        || !agent_shader_effects(node).is_empty()
}

/// An agent shader as a layer effect (docs/design/effects-code-first.md §4.3): a visible `shader` entry of
/// `effects` whose `shaderId` is a code key (`b3:…`) naming a WGSL code object in the document's code table. It runs
/// over the layer as drawn (after its blur, noise, texture and feather), in list order.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentShaderEffect {
    /// The code key.
    pub code: String,
    /// Values for the code's schema parameters (and `seed`).
    pub params: Map,
    /// How far (layer px) the effect may draw outside the layer: the capture is padded by it.
    pub pad: f64,
}

/// The node's agent shader effects, in list order.
pub fn agent_shader_effects(node: &RawNode) -> Vec<AgentShaderEffect> {
    node_effects_json(node)
        .iter()
        .filter(|e| e.get("type").and_then(Value::as_str) == Some("shader") && is_visible_json(e))
        .filter(|e| e.get("mode").and_then(Value::as_str) != Some(crate::shader_effects::SOURCE_MASK_SHADER_EFFECT_MODE))
        .filter_map(|e| {
            let code = e.get("shaderId").and_then(Value::as_str).filter(|s| s.starts_with("b3:"))?;
            Some(AgentShaderEffect {
                code: code.to_string(),
                params: e.get("params").and_then(Value::as_object).cloned().unwrap_or_default(),
                pad: e.get("pad").and_then(Value::as_f64).filter(|v| v.is_finite()).unwrap_or(0.0).clamp(0.0, 4096.0),
            })
        })
        .collect()
}

/// The raw JSON of the node's visible glow / echo / extrude entries, back to
/// front (what the text effect passes draw).
pub fn text_effects_json(node: &RawNode) -> Vec<&Map> {
    node_effects_json(node)
        .iter()
        .filter(|e| {
            matches!(
                e.get("type").and_then(Value::as_str),
                Some("glow" | "echo" | "extrude")
            ) && is_visible_json(e)
        })
        .filter_map(Value::as_object)
        .collect()
}

/// Write a node's effect list (`effects` is removed when empty).
pub fn set_effects(node: &mut Map, effects: Vec<Effect>) {
    if effects.is_empty() {
        node.remove("effects");
    } else {
        node.insert(
            "effects".into(),
            crate::value::to_value(effects).expect("effects serialize"),
        );
    }
}

/// Remove every effect from a node.
pub fn clear_effects(node: &mut Map) {
    node.remove("effects");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    fn raw(v: Value) -> RawNode {
        RawNode::new(v.as_object().unwrap().clone())
    }

    #[test]
    fn parses_every_type_and_round_trips() {
        let v = json!([
            {"id": "d", "type": "drop-shadow", "color": "#000", "blur": 4, "offset": {"x": 1, "y": 2}, "spread": -4, "opacity": 0.5, "showBehindNode": true},
            {"id": "i", "type": "inner-shadow", "color": "#111", "blur": 2, "offset": {"x": 0, "y": 1}, "spread": 3},
            {"id": "l", "type": "layer-blur", "radius": 6, "visible": false},
            {"id": "lp", "type": "layer-blur", "radius": 6, "progressive": {"startRadius": 1, "start": {"x": 0, "y": 0}, "end": {"x": 1, "y": 1}}},
            {"id": "n", "type": "noise", "mode": "duo", "color": "#000", "secondaryColor": "#fff", "size": 2, "density": 0.4, "opacity": 0.5},
            {"id": "tx", "type": "texture", "size": 4, "radius": 3, "clipToShape": true},
            {"id": "b", "type": "background-blur", "radius": 8},
            {"id": "g", "type": "glass", "frost": 1, "refraction": 2, "depth": 3, "lightAngle": 4, "lightIntensity": 5, "dispersion": 6, "splay": 7},
            {"id": "p", "type": "pixelate", "pixelSize": 9},
            {"id": "t", "type": "dither", "brightness": 0, "contrast": 100, "pixelSize": 2},
            {"id": "s", "type": "shader", "shaderId": "x", "mode": "post-process", "source": {"type": "self-asset"}, "params": {}},
            {"id": "gl", "type": "glow", "color": "#fff", "radiusPx": 12},
            {"id": "ig", "type": "inner-glow", "color": "#fff", "radiusPx": 8, "source": "center", "mode": "normal"},
            {"id": "e", "type": "echo", "count": 3, "offsetX": 1, "offsetY": 1},
            {"id": "x", "type": "extrude", "depthPx": 4, "color": "#444"},
            {"id": "f", "type": "feather", "radius": 10},
            {"id": "bad", "type": "nope"}
        ]);
        let effects = parse_effects(Some(&v));
        assert_eq!(effects.len(), 16);
        assert_eq!(effects[15].type_name(), "feather");
        assert_eq!(effects[12].type_name(), "inner-glow");
        let back = crate::value::to_value(&effects).unwrap();
        for (a, b) in back.as_array().unwrap().iter().zip(v.as_array().unwrap()) {
            assert_eq!(a.get("id"), b.get("id"));
            assert_eq!(a.get("type"), b.get("type"));
            assert_eq!(a.get("visible"), b.get("visible"));
        }
        assert_eq!(back[0]["showBehindNode"], json!(true));
        assert_eq!(back[0]["spread"], json!(-4.0));
        assert_eq!(back[3]["progressive"]["startRadius"], json!(1.0));
        assert_eq!(back[4]["secondaryColor"], json!("#fff"));
        assert_eq!(back[5]["clipToShape"], json!(true));
        assert_eq!(back[10]["shaderId"], json!("x"));
        assert_eq!(back[12]["source"], json!("center"));
        assert_eq!(effects[9].type_name(), "dither");
        assert!(
            back[2].get("progressive").is_none(),
            "a uniform blur writes no progressive key"
        );
    }

    #[test]
    fn readers_pick_visible_drawing_entries() {
        let n = raw(json!({"id": "n", "type": "rectangle", "effects": [
            {"id": "a", "type": "drop-shadow", "color": "#000", "blur": 0, "offset": {"x": 0, "y": 0}},
            {"id": "b", "type": "drop-shadow", "color": "#000", "blur": 3, "offset": {"x": 0, "y": 0}, "visible": false},
            {"id": "c", "type": "drop-shadow", "color": "#f00", "blur": 3, "offset": {"x": 0, "y": 0}},
            {"id": "d", "type": "drop-shadow", "color": "#0f0", "blur": 0, "offset": {"x": 2, "y": 0}},
            {"id": "e", "type": "glass", "frost": 0, "refraction": 0, "depth": 0, "lightAngle": 0, "lightIntensity": 0, "dispersion": 0, "splay": 0, "visible": false},
            {"id": "f", "type": "background-blur", "radius": 4},
            {"id": "g", "type": "layer-blur", "radius": 0}
        ]}));
        let shadows = drop_shadows(&n);
        assert_eq!(shadows.len(), 2);
        assert_eq!(shadows[0].color, json!("#f00"));
        assert_eq!(backdrop(&n).unwrap().id, "f");
        assert_eq!(backdrop_json(&n).unwrap().get("id"), Some(&json!("f")));
        assert!(layer_blur(&n).is_some());
        assert!(!has_active_layer_blur(&n));
    }

    #[test]
    fn layer_filters_read_active_entries() {
        let n = |effects: Value| raw(json!({"id": "n", "type": "rectangle", "effects": effects}));
        // A progressive blur with a zero end radius still blurs at its start.
        assert!(has_active_layer_filter(&n(json!([
            {"id": "l", "type": "layer-blur", "radius": 0, "progressive": {"startRadius": 4, "start": {"x": 0, "y": 0}, "end": {"x": 0, "y": 1}}}
        ]))));
        assert!(!has_active_layer_filter(&n(
            json!([{"id": "l", "type": "layer-blur", "radius": 0}])
        )));
        // Noise: density 0 is inert unless duo paints its off-cells.
        let noise = |mode: &str, density: f64| json!({"id": "x", "type": "noise", "mode": mode, "color": "#000", "secondaryColor": "#fff", "size": 1, "density": density});
        assert!(!has_active_layer_filter(&n(json!([noise("mono", 0.0)]))));
        assert!(has_active_layer_filter(&n(json!([noise("duo", 0.0)]))));
        assert_eq!(
            noise_effects(&n(json!([noise("mono", 0.5), noise("multi", 0.2)]))).len(),
            2
        );
        // Texture: the first visible one, only when it moves pixels.
        assert!(texture(&n(
            json!([{"id": "t", "type": "texture", "size": 4, "radius": 0}])
        ))
        .is_none());
        let t = texture(&n(json!([
            {"id": "t0", "type": "texture", "size": 4, "radius": 9, "visible": false},
            {"id": "t1", "type": "texture", "size": 4, "radius": 2}
        ])))
        .unwrap();
        assert_eq!((t.radius, t.clips_to_shape()), (2.0, false));
    }

    #[test]
    fn filter_shadows_are_layer_filters_and_not_drop_shadows() {
        let n = |effects: Value| raw(json!({"id": "n", "type": "path", "effects": effects}));
        let node = n(json!([
            {"id": "a", "type": "filter-shadow", "color": "#000", "blur": 8, "offset": {"x": 2, "y": 4}},
            {"id": "b", "type": "filter-shadow", "color": "#000", "blur": 8, "visible": false},
            {"id": "c", "type": "filter-shadow", "color": "", "blur": 8},
            {"id": "d", "type": "filter-shadow", "color": "#000", "opacity": 0}
        ]));
        let shadows = filter_shadows(&node);
        assert_eq!(shadows.len(), 1);
        assert_eq!((shadows[0].blur, shadows[0].offset, shadows[0].opacity()), (8.0, EffectOffset { x: 2.0, y: 4.0 }, 1.0));
        assert!(has_active_layer_filter(&node));
        // Figma's lanes read `drop-shadow` only.
        assert!(drop_shadows(&node).is_empty());
        // No blur and no offset still draws: the shadow shows through translucent paint.
        let under = n(json!([{"id": "a", "type": "filter-shadow", "color": "#000"}]));
        assert_eq!(filter_shadows(&under).len(), 1);
        assert!(has_active_layer_filter(&under));
        assert_eq!(node_effects(&under)[0].type_name(), "filter-shadow");
    }

    #[test]
    fn feather_reads_the_first_visible_active_entry() {
        let n = |effects: Value| raw(json!({"id": "n", "type": "rectangle", "effects": effects}));
        // Radius 0 is stored but inert; a hidden one is skipped.
        assert!(feather(&n(json!([{"id": "f", "type": "feather", "radius": 0}]))).is_none());
        assert!(!has_active_layer_filter(&n(json!([{"id": "f", "type": "feather", "radius": 0}]))));
        let f = feather(&n(json!([
            {"id": "f0", "type": "feather", "radius": 20, "visible": false},
            {"id": "f1", "type": "feather", "radius": 6}
        ])))
        .unwrap();
        assert_eq!(f.radius, 6.0);
        assert!(has_active_layer_filter(&n(json!([{"id": "f", "type": "feather", "radius": 6}]))));
    }

    #[test]
    fn missing_params_take_their_defaults() {
        let n = raw(json!({"id": "n", "type": "rectangle", "effects": [
            {"id": "d", "type": "drop-shadow", "color": "#000", "blur": 2},
            {"id": "i", "type": "inner-shadow", "color": "#000", "blur": 2}
        ]}));
        let d = drop_shadow(&n).unwrap();
        assert_eq!(
            (d.opacity(), d.offset, d.shows_behind_node(), d.spread()),
            (1.0, EffectOffset::default(), false, 0.0)
        );
        let i = inner_shadow(&n).unwrap();
        assert_eq!((i.opacity(), i.spread()), (1.0, 0.0));
    }
}
