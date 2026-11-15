//! The text op surface shared by the text lanes: the node spec every lane reads (`TextNodeSpec`), the ordered op
//! stream the engine draws (`TextOp` / `TextNodeOps`: decoration and emoji draws), and the gradient-stop parser.
//! The glyph lanes live in `crate::vector`.

use crate::runs::{MsdfRun, RunKind};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Output types (re-exported through moda-engine's seams.rs — see SEAMS.md
// change log entry for M3a Gate 2)
// ---------------------------------------------------------------------------

/// Which pipeline draws an op (modules from the extraction manifest).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextOpPipeline {
    /// `text-decoration` module (bullets, decoration rects).
    Solid,
    /// `text-emoji` module; `emoji` indexes `TextNodeOps::emojis`.
    Emoji { emoji: u32 },
}

/// One draw in node-local space. Vertex layout `[x, y, u, v]` (arrayStride
/// 16, the layout every text pipeline shares), uint32 triangle indices.
#[derive(Debug, Clone, PartialEq)]
pub struct TextOp {
    pub pipeline: TextOpPipeline,
    /// Straight rgba for the `color` uniform (ignored by the emoji shader).
    pub color: [f32; 4],
    /// params.y — node opacity (shadow pass: the shadow's opacity REPLACES it).
    pub opacity: f32,
    /// params.z — blur px (shadow pass).
    pub blur_px: f32,
    /// params.w — stroke px (stroke pass).
    pub stroke_px: f32,
    pub verts: Vec<f32>,
    pub indices: Vec<u32>,
}

/// `msdf-text-gradient` group(0) bindings 5-8, packed exactly like
/// `MsdfTextRenderer.drawRunsWithSharedLayout` (lines 988-1030).
#[derive(Debug, Clone, PartialEq)]
pub struct TextGradientUniforms {
    /// binding 5: [x, y, width, height] local space.
    pub text_bounds: [f32; 4],
    /// binding 6: linear [startX, startY, endX, endY] · radial
    /// [centerX, centerY, radiusX, radiusY] (0-100 percentages).
    pub params: [f32; 4],
    /// binding 7: [numStops, fillType (1 linear | 2 radial), rotation, flags (`GradientSpec::flags`)].
    pub config: [f32; 4],
    /// binding 8: 8 × [offset, r, g, floor(b·255) + a·0.99999]; unused slots
    /// stay ZERO (text gradients have no 10000-offset sentinel — that's the
    /// path-gradient packer).
    pub stops: [[f32; 4]; 8],
}

/// Legacy per-glyph quad info (the seam's `TextQuad` payload) — main fill
/// pass only; kept for the Gate-1 `--dump-text-quads` numeric surface.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphQuadOut {
    pub plane: [f32; 4],
    pub uv: [f32; 4],
    pub color: [f32; 4],
    pub atlas: u32,
}

#[derive(Debug, Clone, Default)]
pub struct TextNodeOps {
    pub ops: Vec<TextOp>,
    /// Atlas page keys `"<family>-<weight>-<style>/<page>"`, deduped in
    /// first-use order. `TextOpPipeline::Msdf*.atlas` indexes this.
    pub atlases: Vec<String>,
    /// Emoji cluster strings, deduped in first-use order.
    pub emojis: Vec<String>,
    pub gradient: Option<TextGradientUniforms>,
    pub quads: Vec<GlyphQuadOut>,
}

/// Gradient fill spec (already resolved from node fields by the caller).
#[derive(Debug, Clone, PartialEq)]
pub struct GradientSpec {
    /// 1 = linear, 2 = radial (the shader's fillType encoding).
    pub fill_type: u32,
    pub start: (f64, f64),
    pub end: (f64, f64),
    /// radians (radial only; node degrees × π/180).
    pub rotation: f64,
    pub stops: Vec<(f64, [f64; 4])>,
    /// Interpolation space + 8 × dither (`moda_scene::paint::GradientOptions::shader_flags`); 0 = the sRGB ramp.
    pub flags: f32,
}

/// Node fields the walk consumes (extracted by `moda_engine::text_draw`;
/// coercions applied by the caller per `renderWithMsdf`):
/// `width ?? 400`, `height ?? 200`, `opacity ?? 1`,
/// `paragraphSpacing || 0` max 0, `verticalAlign || 'top'`.
#[derive(Debug, Clone, Default)]
pub struct TextNodeSpec {
    pub node_id: String,
    /// htmlContent with CSS variables already resolved.
    pub html: std::sync::Arc<str>,
    pub width: Option<f64>,
    pub height: Option<f64>,
    pub paragraph_spacing: f64,
    /// Internal line-box convention; does not change shaping or baseline metrics.
    pub line_height_policy: crate::types::LineHeightPolicy,
    pub vertical_align: Option<String>,
    /// E6.3 "Truncate text": the node's line and height limits, `None` when
    /// truncation is off (`crate::truncation`). Vector lane only.
    pub truncation: Option<crate::truncation::TextTruncation>,
    pub opacity: f64,
    /// Resolved CSS color string (variable refs already substituted).
    pub stroke_color: Option<String>,
    pub stroke_width: Option<f64>,
    /// fillPriority 'color' + fill → solid override of every run color.
    pub fill_override: Option<String>,
    pub gradient: Option<GradientSpec>,
    /// fillPriority 'shader' + a text-fill-ELIGIBLE fillShaderId (the caller
    /// applies `supportsMsdfTextFill` and packs the instance): main-pass
    /// glyph segments emit `TextOpPipeline::MsdfShaderFill`. Ineligible ids
    /// leave this false and per-run colors apply (the production fallback,
    /// `WebGPUTextSystem.ts:599-621`).
    pub shader_fill: bool,
    /// The node's text effects from `effects` (drop shadows, glow, echo,
    /// extrude), pre-resolved exactly like `WebGPUTextSystem.resolveNodeEffects`
    /// (WebGPUTextSystem.ts:551-585): hidden entries, non-drawing drop
    /// shadows and other types already dropped, colors resolved to plain CSS
    /// strings (glow/extrude default '#000000' when unresolvable), node array
    /// order kept (back-to-front). The vector lane renders them all, in order
    /// (`vector::ops::resolve_effect_passes`); the MSDF plan draws every drop
    /// shadow ([`TextNodeSpec::drop_shadows`]), back to front, before the
    /// stroke and fill passes.
    pub effects: Vec<TextEffectSpec>,
    /// KineticText per-glyph deltas, one resolved {@link
    /// crate::vector::ops::GlyphXform} per code point of the node's decoded source
    /// text ({@link crate::parser::parse_rich_text_with_source}). Empty (the
    /// default) = no per-glyph animation: every glyph record carries identity and
    /// the vector lane stays byte-exact. Presence forces the vector engine. Read
    /// from the node's `unitTransforms` array (animation-mutated by the Rust v3
    /// evaluator) in `text_draw.rs`; the vector lane maps each glyph → its source
    /// code point and indexes this. The MSDF lane ignores it (no per-glyph).
    pub unit_transforms: std::sync::Arc<Vec<crate::vector::ops::GlyphXform>>,
    /// KineticText units: per code point (as `unit_transforms`), the index of the first code point of the unit
    /// (word, line) it belongs to. A unit's glyphs scale and turn about the unit's centre, as one piece; a code
    /// point past the end, or naming itself, is a unit of its own (its glyph turns about its own centre). Empty
    /// (the default): every code point is its own unit. From `unitTransforms[i].unit`.
    pub unit_groups: std::sync::Arc<Vec<u32>>,
    /// KineticText fill: per code point (as `unit_transforms`), how much of the glyph's fill draws, 0…1, on top of
    /// its opacity. The text's stroke and effect passes keep the glyph's own opacity, so 0 leaves a stroked glyph's
    /// outline. Empty (the default): full fill. From `unitTransforms[i].fill`.
    pub unit_fill: std::sync::Arc<Vec<f32>>,
    /// The draw's paints: the colour (straight rgba) of each range that names a paint (`--nf-paint`,
    /// `types::InlineStyle::paint`), in place of the range's `color`. Paint is the draw's, so text that differs
    /// only in its colours is one layout. A range naming no paint, or one past the end, draws in its `color`.
    /// `fill_override`, a gradient and a shader fill come before it.
    pub run_colors: std::sync::Arc<Vec<[f32; 4]>>,
    /// Draw only these canonical code points, retaining the full layout and unit pivots.
    pub unit_range: Option<std::ops::Range<usize>>,
    /// The node's `distort` warp as the local-px 3×3 homography,
    /// row-major. Rendered by the vector lane's DrawUniforms rows only; the
    /// MSDF and Degraded plans ignore it (their warp is out of scope — the
    /// accepted fallback gap). Does not affect layout — the warp applies at
    /// draw time in the vertex shaders.
    pub distort: Option<[f32; 9]>,
    /// F7.2 type on a path: the node's text runs along this path (`crate::text_path`) instead of in its box.
    /// Vector lane only; the MSDF and degraded plans draw it straight.
    pub text_path: Option<crate::text_path::TextPathSpec>,
    /// E7.20 auto-width text (`resizeMode: 'auto-width'`): lines break only at
    /// the text's own line breaks, never at the box edge. The box width still
    /// places the lines (alignment), as the editor keeps it at the widest line.
    pub auto_width: bool,
    /// Area type (`crate::area`): the text fits its shape, thread and wrap objects. Vector lane; the degraded
    /// tier draws it as box text.
    pub area: Option<crate::area::AreaSpec>,
}

impl TextNodeSpec {
    /// The width lines wrap at: the box width (`width ?? 400`), or unbounded for
    /// auto-width text.
    pub fn wrap_width(&self) -> f64 {
        if self.auto_width {
            f64::INFINITY
        } else {
            self.width.unwrap_or(400.0).max(0.0)
        }
    }

    /// Vertical text: the length lines wrap at, the box height (`height ?? 200`), or unbounded for auto-width text.
    pub fn inline_wrap_width(&self) -> f64 {
        if self.auto_width {
            f64::INFINITY
        } else {
            self.height.unwrap_or(200.0).max(0.0)
        }
    }
}

impl TextNodeSpec {
    /// The drop shadows of [`TextNodeSpec::effects`], back to front: the MSDF
    /// lane draws one shadow pass per entry (`MsdfTextRenderer.ts:792-828`).
    pub fn drop_shadows(&self) -> impl Iterator<Item = &TextDropShadow> {
        self.effects.iter().filter_map(|e| match e {
            TextEffectSpec::DropShadow(s) => Some(s),
            _ => None,
        })
    }
}

/// A resolved `drop-shadow` effect on text: the glyphs re-drawn in `color`,
/// offset and blurred, behind the text.
#[derive(Debug, Clone, PartialEq)]
pub struct TextDropShadow {
    /// Resolved CSS color (non-empty).
    pub color: String,
    pub offset_x: f64,
    pub offset_y: f64,
    /// Blur radius (× viewZoom — 1 on the export path).
    pub blur: f64,
    /// REPLACES the node opacity (the effect's `opacity`, absent = 1).
    pub opacity: f64,
}

/// One resolved text effect — `ResolvedTextEffect`
/// (vectorTextEffects.ts:51-63): the effect params minus id/visible.
/// Numeric params are `Option<f64>`: `None` mirrors a JS `undefined` /
/// non-number value, which the pass resolver's `Number.isFinite` guards
/// reject (or default, for the optional params).
#[derive(Debug, Clone, PartialEq)]
pub enum TextEffectSpec {
    DropShadow(TextDropShadow),
    Glow {
        /// Resolved CSS color (`resolveCssColor(effect.color) ?? '#000000'`).
        color: String,
        radius_px: Option<f64>,
        intensity: Option<f64>,
        opacity: Option<f64>,
    },
    Echo {
        count: Option<f64>,
        offset_x: Option<f64>,
        offset_y: Option<f64>,
        scale_step: Option<f64>,
        color_start: Option<String>,
        color_end: Option<String>,
        opacity_step: Option<f64>,
    },
    Extrude {
        depth_px: Option<f64>,
        angle_deg: Option<f64>,
        /// Resolved CSS color (`resolveCssColor(effect.color) ?? '#000000'`).
        color: String,
    },
}

// ---------------------------------------------------------------------------
// parseGradientStops port (`WebGPUGradientSystem.ts:420-477`)
// ---------------------------------------------------------------------------

const MAX_GRADIENT_STOPS: usize = 8;

/// Strict color parse used by gradient stops: `colorToNormalizedRgba`
/// DIRECTLY (hex / integer-rgb only — named colors and hsl() fall to the
/// white default; colorHelpers.ts:377-399).
fn color_to_normalized_rgba_strict(color: &str) -> Option<[f64; 4]> {
    crate::color::strict_color_to_normalized_rgba(color)
}

/// `WebGPUGradientSystem.parseGradientStops` over the raw
/// `[offset, color, offset, color, …]` node array.
pub fn parse_gradient_stops(color_stops: Option<&Vec<Value>>) -> Vec<(f64, [f64; 4])> {
    let mut stops: Vec<(f64, [f64; 4])> = Vec::new();
    let Some(arr) = color_stops else {
        return vec![(0.0, [1.0, 1.0, 1.0, 1.0])];
    };
    let mut i = 0;
    while i < arr.len() && stops.len() < MAX_GRADIENT_STOPS {
        let offset = arr[i].as_f64().unwrap_or(f64::NAN);
        let color_value = arr.get(i + 1);
        // Math.max(0, Math.min(1, offset)) — NaN propagates to NaN in JS;
        // fixtures always carry numbers, treat NaN as 0 like JS Math.min/max
        // chain would yield NaN → clamp deterministically to 0 here and note.
        let validated_offset = if offset.is_nan() {
            0.0
        } else {
            offset.clamp(0.0, 1.0)
        };
        let color = match color_value {
            Some(Value::String(s)) => {
                color_to_normalized_rgba_strict(s).unwrap_or([1.0, 1.0, 1.0, 1.0])
            }
            _ => [1.0, 1.0, 1.0, 1.0],
        };
        let clamped = [
            color[0].clamp(0.0, 1.0),
            color[1].clamp(0.0, 1.0),
            color[2].clamp(0.0, 1.0),
            color[3].clamp(0.0, 1.0),
        ];
        stops.push((validated_offset, clamped));
        i += 2;
    }
    if stops.is_empty() {
        stops.push((0.0, [1.0, 1.0, 1.0, 1.0]));
    }
    // stable sort by offset
    stops.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Positions are kept: two stops at one position are a hard stop (the text shader steps there).
    stops
}

// ---------------------------------------------------------------------------
// The recursive draw walk (drawRunsWithSharedLayout)
// ---------------------------------------------------------------------------

/// Emoji runs in `convertMsdfRunsToParagraphs` allocation order (paragraph
/// groups sorted numerically, runs in order) — `emoji_<n>` n = position here.
pub(crate) fn emoji_runs_in_layout_order<'a>(runs: &'a [MsdfRun]) -> Vec<&'a MsdfRun> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<usize, Vec<&MsdfRun>> = BTreeMap::new();
    for r in runs {
        groups.entry(r.paragraph_index).or_default().push(r);
    }
    let mut out = Vec::new();
    for (_, rs) in groups {
        for r in rs {
            if r.kind == RunKind::Emoji {
                out.push(r);
            }
        }
    }
    out
}

/// `generateBulletDisc` (`msdfLayoutUtils.ts:87-106`): clamped ≥3 segments,
/// fan triangulation, verts as [x, y, 0, 0].
pub fn generate_bullet_disc(
    center_x: f64,
    center_y: f64,
    radius: f64,
    segments: u32,
) -> (Vec<f32>, Vec<u32>) {
    let n = segments.max(3);
    let mut verts: Vec<f32> = Vec::with_capacity(n as usize * 4);
    let mut indices: Vec<u32> = Vec::new();
    for i in 0..n {
        let theta = (i as f64 / n as f64) * std::f64::consts::PI * 2.0;
        let px = center_x + theta.cos() * radius;
        let py = center_y + theta.sin() * radius;
        verts.extend_from_slice(&[px as f32, py as f32, 0.0, 0.0]);
    }
    for i in 1..(n - 1) {
        indices.extend_from_slice(&[0, i, i + 1]);
    }
    (verts, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_stop_parsing_matches_production() {
        let raw = vec![
            serde_json::json!(0),
            serde_json::json!("#ff006e"),
            serde_json::json!(0.5),
            serde_json::json!("#ffbe0b"),
            serde_json::json!(1),
            serde_json::json!("#3a86ff"),
        ];
        let stops = parse_gradient_stops(Some(&raw));
        assert_eq!(stops.len(), 3);
        assert_eq!(stops[1].0, 0.5);
        assert_eq!(stops[2].1, [58.0 / 255.0, 134.0 / 255.0, 1.0, 1.0]);
        // named colors do NOT parse on this path → white
        let raw2 = vec![serde_json::json!(0), serde_json::json!("red")];
        assert_eq!(parse_gradient_stops(Some(&raw2))[0].1, [1.0, 1.0, 1.0, 1.0]);
    }
}
