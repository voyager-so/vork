//! Vector-engine node ops — the plan side of `--text-engine vector`.
//!
//! Rust mirror of the browser's vector text draw orchestration:
//! `WebGPUTextSystem.renderWithVector` (WebGPUTextSystem.ts:633-762) →
//! `VectorTextRenderer.drawRuns` → `buildLayout`/`walkLines` →
//! `drawRunsAnalytic` (VectorTextRenderer.ts:352-658) with the Phase-1
//! stroke/shadow effect passes (`vectorTextEffects.ts`). Output is a flat,
//! ordered op stream (`VectorNodeOps`) the engine draws verbatim — draw
//! order == blend order, one shared node depth, exactly like the MSDF lane's
//! `TextNodeOps`.
//!
//! Only the ANALYTIC fill is ported (`?text=vector`, the engine the flag
//! ships). The tess A/B baseline (`?text=vector-tess`) is not a CLI mode; its
//! pipeline ships in the manifest (`vector-text-tess`) because the analytic
//! path draws its solid primitives (UL bullets, decoration
//! rects, the shadow pass's primitive re-draw) through that same pipeline
//! (`VectorTextRenderer.drawPrimitiveTriangles`).
//!
//! Layout numbers are pinned by Gate 1 (`vector_oracle_parity`); the glyph
//! curves by `vector_curve_parity`. This module's added surface is the
//! instance-record assembly (AA pad, effect pads/offsets, color bucketing)
//! and the op ordering, all cited line-by-line below.

use std::collections::HashMap;

use crate::catalog::Catalog;
use crate::color::{
    css_color_to_normalized_rgba, normalize_run_color, rgba_normalized_array_to_css,
};
use crate::decorations::{compute_strikethrough_rects, compute_underline_rects, DecorationRect};
use crate::line_metrics::compute_baseline_for_line_with_fallback;
use crate::ops::{TextGradientUniforms, TextNodeSpec};
use crate::paragraph_layout::layout_paragraphs_to_lines_with_policy;
use crate::positioning::{
    advance_pen_y_for_paragraph_break, compute_bullet_center_x, compute_bullet_metrics,
    compute_line_start_x, compute_list_indentation, compute_segment_positions_justified,
    justify_space_px, measure_line_width, JustifiedMeasurer,
};
use crate::runs::{convert_msdf_runs_to_paragraphs, extract_msdf_runs_by_paragraph};
use crate::vector::cjk::{js_whitespace, segment_text_for_wrap_with};
use crate::vector::measure::{
    place_glyphs, style_font_key, style_to_variant, VectorFontStore, VectorMeasurer,
};
use crate::vector::node_layout::{
    bullet_x_height_px, resolve_vector_runs_with, select_text_engine_with, Canvas2dFontMetricsProvider,
    EmojiRuns, RunResolution, VectorMetricsProvider,
};
use crate::vector::otjs::ShapeError;

/// Manifest-pinned constants (`vector_text` section).
/// `ANALYTIC_AA_PAD_DEVICE_PX` (VectorTextRenderer.ts:155).
pub const ANALYTIC_AA_PAD_DEVICE_PX: f64 = 2.0;
/// `EFFECT_PAD_MAX_EM` (vectorTextEffects.ts:271).
pub const EFFECT_PAD_MAX_EM: f64 = 4.0;
/// `DEFAULT_EXTRUDE_ANGLE_DEG` (vectorTextEffects.ts:43).
pub const DEFAULT_EXTRUDE_ANGLE_DEG: f64 = 45.0;
/// `ECHO_COUNT_MAX` (core/scene/nodeEffects.ts:22).
pub const ECHO_COUNT_MAX: f64 = 50.0;
/// `ECHO_SCALE_MAX` (vectorTextEffects.ts:117).
pub const ECHO_SCALE_MAX: f64 = 16.0;
/// `BULLET_DISC_SEGMENTS` (VectorTextRenderer.ts:165).
const BULLET_DISC_SEGMENTS: u32 = 16;

/// Resolved per-glyph transform (KineticText). Geometry-INDEPENDENT:
/// `lin` is the 2×2 linear part (rotation·scale) row-major as `[a, c, b, d]`
/// (row0 = (a, c), row1 = (b, d)); `off` is the px offset; `opacity` is 0..1. The
/// renderer applies it ABOUT EACH GLYPH'S EM-BOX CENTER in the vertex shader —
/// the center is per-instance and geometry-dependent (and differs for each
/// effect-pass copy, which is pen/scale-shifted), so only this center-independent
/// part is stored on the record. Matrix-capable by construction: `lin` is a
/// matrix (not baked offset/rotate/scale shader ops), so a future Z axis slots in
/// as an added row/column without restructuring the pipeline (design invariant 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphXform {
    pub lin: [f32; 4],
    pub off: [f32; 2],
    pub opacity: f32,
}

/// Narrow an `f64` to `f32`, substituting `fallback` if the result isn't finite
/// (overflow to ±inf, or a NaN that survived an upstream `clamp`). Keeps the GPU
/// instance stream finite on hostile/malformed input. See `GlyphXform::resolve`.
fn finite_f32(v: f64, fallback: f32) -> f32 {
    let n = v as f32;
    if n.is_finite() {
        n
    } else {
        fallback
    }
}

impl GlyphXform {
    /// Identity — no transform, full opacity. The value every glyph carries when
    /// the node has no `unitTransforms`; the vertex shader fast-paths an identity
    /// instance to the original `pen + em*fontSizePx` math (bit-exact), keeping
    /// static text byte-identical.
    pub const IDENTITY: GlyphXform = GlyphXform {
        lin: [1.0, 0.0, 0.0, 1.0],
        off: [0.0, 0.0],
        opacity: 1.0,
    };

    /// Resolve a `GlyphTransform` (offsetX/offsetY/scale/rotation/opacity) into the
    /// stored center-independent form. Rotation is degrees, scale uniform; both
    /// pivot about the glyph center (applied in the shader). Mirrors the TS
    /// `GlyphTransform` field semantics (core/scene/nodes.ts).
    pub fn resolve(
        offset_x: f64,
        offset_y: f64,
        scale: f64,
        rotation_deg: f64,
        opacity: f64,
    ) -> GlyphXform {
        let rot = rotation_deg.to_radians();
        let (sin, cos) = rot.sin_cos();
        // Guard the f64→f32 narrowing per component: a finite-but-huge imported
        // value (e.g. 1e100) overflows to ±inf in f32, and a NaN slips through
        // `clamp` (NaN.clamp == NaN) — either would put non-finite data into the
        // GPU instance stream / Tess walls and render undefined. The evaluator
        // already drops non-finite *f64*, but it can't see the narrowing overflow;
        // a hostile/malformed scene reaching `extract_unit_transforms` directly can.
        // Non-finite → the identity component for that field (so an absurd glyph
        // degrades to no-op, never inf). Finite values pass through unchanged, so
        // legitimate transforms — and the identity fast-path — are byte-unaffected.
        // Row-major 2×2 = R·S, y-down (positive degrees → clockwise on screen):
        // row0 = (s·cos, -s·sin), row1 = (s·sin, s·cos).
        GlyphXform {
            lin: [
                finite_f32(scale * cos, 1.0),
                finite_f32(scale * -sin, 0.0),
                finite_f32(scale * sin, 0.0),
                finite_f32(scale * cos, 1.0),
            ],
            off: [finite_f32(offset_x, 0.0), finite_f32(offset_y, 0.0)],
            opacity: finite_f32(opacity.clamp(0.0, 1.0), 1.0),
        }
    }

    /// `self ∘ inner`: `inner` applied first, both about the same em-box centre (`lin·(p − C) + C + off`), so the
    /// result is `lin = L·Lᵢ`, `off = L·oᵢ + o`. E8.22 composes a run's character scale and rotation under a
    /// KineticText transform this way.
    pub fn after(self, inner: GlyphXform) -> GlyphXform {
        let [a, c, b, d] = self.lin;
        let [ia, ic, ib, id] = inner.lin;
        GlyphXform {
            lin: [a * ia + c * ib, a * ic + c * id, b * ia + d * ib, b * ic + d * id],
            off: [
                a * inner.off[0] + c * inner.off[1] + self.off[0],
                b * inner.off[0] + d * inner.off[1] + self.off[1],
            ],
            opacity: self.opacity * inner.opacity,
        }
    }

    /// True when this is exactly the identity transform (full opacity, no linear
    /// part, no offset). Lets the non-kinetic path stay on byte-exact records and
    /// lets extrude skip the per-glyph outline transform entirely.
    pub fn is_identity(&self) -> bool {
        *self == GlyphXform::IDENTITY
    }
}

/// Desktop fork: the resolved `font_style` of a run that asked for italic in a
/// family with no italic face. It selects the upright face (anything but
/// "italic" does) and marks the run's glyphs for a synthetic oblique.
pub const SYNTHETIC_OBLIQUE: &str = "synthetic-oblique";

/// Horizontal shear of a synthetic oblique: x' = x - k·y (y-down), Blink's
/// `SkFont::setSkewX(-1/4)`.
const SYNTHETIC_OBLIQUE_SKEW: f32 = 0.25;

impl GlyphXform {
    /// `self` applied after a synthetic-oblique shear about the glyph's
    /// baseline. The shader pivots `lin` about the em-box center C, so pivoting
    /// the shear at the baseline instead needs an x offset of -k·C.y (px, y-down).
    fn with_synthetic_oblique(self, em_box: [f32; 4], font_size_px: f32) -> GlyphXform {
        let k = SYNTHETIC_OBLIQUE_SKEW;
        let cy = (em_box[1] + em_box[3]) * 0.5 * font_size_px;
        // S = [1, -k; 0, 1], offset (-k·cy, 0); result = self ∘ S.
        let [a, c, b, d] = self.lin;
        let (sx, sy) = (-k * cy, 0.0);
        GlyphXform {
            lin: [a, a * -k + c, b, b * -k + d],
            off: [a * sx + c * sy + self.off[0], b * sx + d * sy + self.off[1]],
            opacity: self.opacity,
        }
    }
}

/// The KineticText unit (the index of its first code point) code point `cp` belongs to.
fn unit_group(spec: &TextNodeSpec, cp: usize) -> u32 {
    spec.unit_groups.get(cp).copied().unwrap_or(cp as u32)
}

/// The centre of each multi-glyph KineticText unit's ink at rest (node-local px), keyed by the unit's first code
/// point: the union of its glyphs' outline bounds, as the layout placed them.
fn unit_pivots(spec: &TextNodeSpec, segments: &[PlacedSegment]) -> HashMap<u32, (f64, f64)> {
    let mut boxes: HashMap<u32, ([f64; 4], usize)> = HashMap::new();
    for seg in segments {
        for g in &seg.placed {
            let cluster = seg.glyphs.get(g.glyph_idx).map_or(g.glyph_idx, |shaped| shaped.cluster);
            if g.curve_count == 0 {
                continue;
            }
            let size = seg.font_size_px;
            let y = seg.baseline_y - g.y_offset_px - seg.character.baseline_shift;
            let b = [
                g.pen_x + g.bounds[0] as f64 * size,
                y + g.bounds[1] as f64 * size,
                g.pen_x + g.bounds[2] as f64 * size,
                y + g.bounds[3] as f64 * size,
            ];
            let Some(cp) = seg.source.get(cluster) else { continue };
            let e = boxes
                .entry(unit_group(spec, cp))
                .or_insert((
                    [
                        f64::INFINITY,
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        f64::NEG_INFINITY,
                    ],
                    0,
                ));
            e.0 = [
                e.0[0].min(b[0]),
                e.0[1].min(b[1]),
                e.0[2].max(b[2]),
                e.0[3].max(b[3]),
            ];
            e.1 += 1;
        }
    }
    boxes
        .into_iter()
        .filter(|(_, (_, n))| *n > 1)
        .map(|(g, (b, _))| (g, ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)))
        .collect()
}

/// Upper bound on the per-node `unitTransforms` array the renderer honors. A
/// malformed/hostile *imported scene* reaches the renderer's
/// `extract_unit_transforms` directly, so the extraction path applies this cap
/// to bound per-frame allocation.
pub const UNIT_TRANSFORMS_MAX_LENGTH: usize = 10_000;

/// How many `unitTransforms` entries the renderer honors for a node of `len` entries and HTML `html`: all of them up
/// to [`UNIT_TRANSFORMS_MAX_LENGTH`], and past it as many as the text has code points (the evaluator writes one per
/// code point, so a text longer than the cap still moves to its last character; a hostile array stays bounded by the
/// text it indexes).
pub fn unit_transforms_cap(len: usize, html: &str) -> usize {
    if len <= UNIT_TRANSFORMS_MAX_LENGTH {
        return len;
    }
    len.min(UNIT_TRANSFORMS_MAX_LENGTH.max(crate::parser::source_code_point_count(html)))
}

/// One analytic glyph instance (the row `packAnalyticInstances` consumes). The
/// base 9 fields (36 bytes) are followed by the KineticText per-glyph transform:
/// `glyph_lin` (row-major 2×2), `glyph_off` (px), `glyph_opacity`.
/// For a node with no `unitTransforms` every record carries {@link GlyphXform::IDENTITY}
/// and the shader fast-paths it to the original math (byte-exact static text).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct VectorAnalyticRecord {
    pub pen_x: f32,
    pub baseline_y: f32,
    pub font_size_px: f32,
    /// Padded glyph bounds in em units [minX, minY, maxX, maxY].
    pub em_box: [f32; 4],
    pub first_curve: u32,
    pub curve_count: u32,
    /// Per-glyph linear part (rotation·scale), row-major `[a, c, b, d]`.
    pub glyph_lin: [f32; 4],
    /// Per-glyph px offset.
    pub glyph_off: [f32; 2],
    /// Per-glyph opacity (0..1); multiplies coverage alpha, inherited by effects.
    pub glyph_opacity: f32,
}

// ABI tripwire (mirrors the 464-byte assert on `GradientInstance` in
// paint/gradient.rs): this record IS the GPU analytic-instance row — 16 floats /
// 64 bytes, hand-packed field-by-field in `moda-engine`'s `text_draw.rs`, packed
// again in TS `packAnalyticInstances`, and consumed by the `vector_text` WGSL
// vertex layouts (`glyph_lin`@36, `glyph_off`@52, `glyph_opacity`@60). A field or
// type change here that desyncs those four edit sites trips this at compile time.
const _: () = assert!(core::mem::size_of::<VectorAnalyticRecord>() == 64);

/// Which extracted pipeline draws an op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorOpPipeline {
    /// `vector-text-tess` — solid local-space triangles (UL bullets,
    /// decoration rects, shadow-pass primitive re-draw).
    Tess,
    /// `vector-text-analytic` — instanced winding-coverage glyph quads
    /// (solid fill + the stroke/shadow distance-to-curve passes).
    Analytic,
    /// `vector-text-analytic-gradient`.
    AnalyticGradient,
    /// `<id>.vector-text-fill` (module/instance ride the engine batch).
    AnalyticShaderFill,
}

/// One draw in node-local space (depth/model/clip ride the engine batch).
#[derive(Debug, Clone, PartialEq)]
pub struct VectorOp {
    pub pipeline: VectorOpPipeline,
    /// DrawUniforms.color (straight rgba).
    pub color: [f32; 4],
    /// params.y — node opacity. Drop shadow: its opacity REPLACES it;
    /// structured effects MULTIPLY it (`writeDrawUniforms`,
    /// VectorTextRenderer.ts:1320 — may exceed 1 for glow intensity).
    pub opacity: f32,
    /// params.z — blur px, LOCAL canvas px (shadow / glow passes).
    pub blur_px: f32,
    /// params.w — stroke px, LOCAL canvas px (stroke pass). Intentionally
    /// differs from MSDF's screen-px widths (quirk ledger).
    pub stroke_px: f32,
    /// Glow only (`Analytic` pipeline): render through the additive
    /// (emitted-light) blend variant — `createVectorAnalyticAdditivePipeline`
    /// (vectorTextShaders.ts:886-906, TEXT_ADDITIVE_BLEND :92-95).
    pub additive: bool,
    /// Tess: flat `[x0,y0, x1,y1, ...]` triangle list.
    pub tris: Vec<f32>,
    /// Analytic*: per-glyph instance records.
    pub records: Vec<VectorAnalyticRecord>,
    /// `AnalyticGradient` only: this op's own gradient — a text range's
    /// `--nf-text-fill` (`crate::run_fill`). `None` = the node's
    /// [`VectorNodeOps::gradient`].
    pub gradient: Option<TextGradientUniforms>,
    /// The paint `color` is: `TextNodeSpec::run_colors[paint]`. A draw of the same layout with other paints differs
    /// from this one in the `color` of its ops that name a paint and in nothing else, so recolouring a kept batch is
    /// writing those colours. Several ops may name one paint: ranges that name it and do not follow one another. `None`: the colour is not a paint's (an effect's or the stroke's, a node-level fill, or
    /// a range that names no paint the draw supplies).
    pub paint: Option<u32>,
}

/// Everything the engine needs to draw one vector text node.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VectorNodeOps {
    pub ops: Vec<VectorOp>,
    /// The layout's curve buffer, shared by every draw of the layout ([`VectorLayout`]): 6 floats per quadratic
    /// segment, em units,
    /// y-down, pen origin — the `GlyphCurveStore` payload, deduped per
    /// (fontKey, glyphIndex) in first-use (visit) order. `first_curve`
    /// indexes are relative to THIS buffer.
    pub curve_points: std::sync::Arc<Vec<f32>>,
    /// Gradient uniforms (vector binding 3; same 44-float layout the MSDF
    /// lane splits over bindings 5-8 — `packTextGradientUniformFloats`).
    pub gradient: Option<TextGradientUniforms>,
    /// Colour emoji clusters, drawn as textured quads (the engine's `text-emoji` image
    /// pipeline) above the glyph passes. Their layout slot is 1 em + letter spacing, like
    /// the MSDF lane's placeholder; the bitmap comes from the colour emoji font at draw
    /// time (`degraded::DegradedEngine::emoji_cell_rgba`).
    pub emojis: Vec<VectorEmoji>,
}

/// One emoji cluster placed in node-local px.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorEmoji {
    pub cluster: String,
    /// The image's corners, top-left, top-right, bottom-right, bottom-left, as `[x, y]`
    /// pairs: an axis-aligned cell of [`EMOJI_CELL_WIDTH_EM`] × [`EMOJI_CELL_HEIGHT_EM`]
    /// em with its top [`EMOJI_CELL_ASCENT_EM`] above the baseline, turned onto the path
    /// for type on a path.
    pub corners: [f32; 8],
    /// Node opacity (the emoji is not tinted by the run colour or fill).
    pub opacity: f32,
    /// An effect pass's copy (drop shadow, glow, stroke, echo) instead of the emoji
    /// itself: the cell's silhouette, tinted and shaped as the pass shapes glyphs, over
    /// `corners` grown by [`EmojiEffect::pad_texels`]. Its opacity is baked in; `opacity`
    /// is then 1.
    pub effect: Option<EmojiEffect>,
}

/// Texels per em of an emoji effect texture (half the cell's: the passes are soft).
pub const EMOJI_EFFECT_TEXELS_PER_EM: f64 = 64.0;

/// One effect pass over an emoji cell, in the effect texture's texels
/// ([`EMOJI_EFFECT_TEXELS_PER_EM`] per em of the copy's size), quantised so a drag that
/// barely changes a blur reuses the texture.
#[derive(Debug, Clone, PartialEq)]
pub struct EmojiEffect {
    /// Straight RGB and alpha of the pass colour.
    pub color: [f32; 4],
    /// The pass opacity (node opacity × the effect's; a glow's intensity may exceed 1).
    pub gain: f32,
    /// The analytic shader's `blur px`: the silhouette edge softens over ± this.
    pub blur_texels: f32,
    /// The analytic shader's `stroke px`: an outside band this wide.
    pub stroke_texels: f32,
    /// How far the texture reaches past the cell on every side.
    pub pad_texels: u32,
    /// Glow: emitted light (added to what is below) rather than painted over it.
    pub additive: bool,
}

impl EmojiEffect {
    /// The key a batch's emoji list carries for this copy of `cluster` (the engine
    /// resolves its texture from it, [`EmojiEffect::parse_key`]).
    pub fn key(&self, cluster: &str) -> String {
        let c = self.color;
        format!(
            "\u{1}fx {} {} {} {} {} {} {} {} {}\u{1}{cluster}",
            c[0], c[1], c[2], c[3], self.gain, self.blur_texels, self.stroke_texels, self.pad_texels,
            u8::from(self.additive)
        )
    }

    /// Inverse of [`EmojiEffect::key`]: the effect (if any) and the cluster.
    pub fn parse_key(key: &str) -> (Option<EmojiEffect>, &str) {
        let Some(rest) = key.strip_prefix("\u{1}fx ") else {
            return (None, key);
        };
        let Some((params, cluster)) = rest.split_once('\u{1}') else {
            return (None, key);
        };
        let v: Vec<f32> = params.split(' ').filter_map(|s| s.parse().ok()).collect();
        if v.len() != 9 {
            return (None, key);
        }
        let effect = EmojiEffect {
            color: [v[0], v[1], v[2], v[3]],
            gain: v[4],
            blur_texels: v[5],
            stroke_texels: v[6],
            pad_texels: v[7] as u32,
            additive: v[8] != 0.0,
        };
        (Some(effect), cluster)
    }
}

/// The emoji image cell, in em of the run's font size. One em wide (the layout slot), and
/// taller than it so a colour-font glyph's descent below the baseline is not clipped.
pub const EMOJI_CELL_WIDTH_EM: f64 = 1.0;
pub const EMOJI_CELL_HEIGHT_EM: f64 = 1.25;
/// How far the cell's top sits above the baseline.
pub const EMOJI_CELL_ASCENT_EM: f64 = 1.0;

/// Vector routing outcome for one node.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)] // mirrors ops::TextEngineDecision
pub enum VectorEngineDecision {
    Vector(VectorNodeOps),
    /// The browser would not draw this node through the vector path —
    /// `selectTextEngine` chose another engine, run resolution failed, or
    /// the vector path threw (`renderWithVector`'s catch → `renderWithMsdf`).
    /// The caller re-plans through the MSDF lane.
    Fallback {
        reason: String,
        /// A LOW-CARDINALITY class for the decline, safe to group and
        /// alert on, where `reason` is free text (it interpolates family names,
        /// variants and shaping-error messages). One of `select-text-engine:*`,
        /// `complex-script`, `no-runs`, `emoji-run`, `unresolved-variant`,
        /// `coverage`, `vector-error`, or `font-unavailable:<miss class>`.
        cause: String,
        /// The font families implicated in this decline, for the
        /// resource report's `font_fallback_class_nodes[].families`. That field
        /// was emitted EMPTY on every record, so the family survived only inside
        /// the free-text `reason` and had to be recovered by regex (and was lost
        /// entirely once the record hit the log truncation cap). Empty for
        /// declines with no font context (`no-runs`, `complex-script`, a
        /// mid-layout throw).
        families: Vec<String>,
    },
}

/// `colorKey` (VectorTextRenderer.ts:1259-1261): toFixed(4) bucket key.
fn color_key(c: [f32; 4]) -> String {
    format!("{:.4},{:.4},{:.4},{:.4}", c[0], c[1], c[2], c[3])
}

/// The colour a run whose CSS `color` is `css` draws in (straight rgba): the run-colour quantization and the
/// segment colour of the draw. A caller supplying a range's colour with the draw (`TextNodeSpec::run_colors`) gets the
/// colour the same `color` in the HTML would draw.
pub fn run_color(css: &str) -> [f32; 4] {
    css_rgba_or_black(&rgba_normalized_array_to_css(normalize_run_color(css)))
}

fn css_rgba_or_black(css: &str) -> [f32; 4] {
    let c = css_color_to_normalized_rgba(css).unwrap_or([0.0, 0.0, 0.0, 1.0]);
    [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32]
}

/// `appendIndexedTrianglesXY` (vectorLayout.ts:128-139): de-index 4-float
/// `[x, y, u, v]` vertices into a flat xy triangle list.
fn append_indexed_triangles_xy(
    out: &mut Vec<f32>,
    verts: &[f32],
    indices: &[u32],
    dx: f32,
    dy: f32,
) {
    for &i in indices {
        let base = i as usize * 4;
        out.push(verts[base] + dx);
        out.push(verts[base + 1] + dy);
    }
}

/// Vertical text: turn a line's geometry (stride-4 vertices, x and y first) into its column.
fn map_xy_into_column(verts: &mut [f32], column: &crate::vertical::ColumnFrame) {
    for v in verts.chunks_exact_mut(4) {
        let (x, y) = column.map_point(v[0] as f64, v[1] as f64);
        v[0] = x as f32;
        v[1] = y as f32;
    }
}

/// `translateTriangles` (VectorTextRenderer.ts:1274-1281).
fn translate_triangles(tris: &[f32], dx: f32, dy: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(tris.len());
    let mut i = 0;
    while i < tris.len() {
        out.push(tris[i] + dx);
        out.push(tris[i + 1] + dy);
        i += 2;
    }
    out
}

/// The ordered draws of one kind (glyph records, or a primitive's triangles), each with its colour.
///
/// A range that names no paint goes to its colour's bucket, in the order colours first come (JS Map iteration
/// order). A range that names a paint is drawn where it comes: it joins the entry before it when that entry names
/// the same paint with the same colour, and never an earlier one, so ranges draw in text order however their paints
/// and colours repeat, and no entry is searched for.
struct Buckets<T> {
    /// Each entry's colour key (empty for an entry that names a paint), colour and draws.
    entries: Vec<(String, [f32; 4], Vec<T>)>,
    /// Each entry's paint (`InlineStyle::paint`): a draw that supplies paints colours the entry with it; an entry's
    /// own colour is what it draws in when the draw supplies none. A paint may be several entries'.
    paints: Vec<Option<u32>>,
    /// The entry of each colour key.
    by_color: HashMap<String, usize>,
}
impl<T> Buckets<T> {
    fn new() -> Self {
        Buckets {
            entries: Vec::new(),
            paints: Vec::new(),
            by_color: HashMap::new(),
        }
    }
    /// The bucket of `color`, for a range that names no paint.
    fn get(&mut self, color: [f32; 4]) -> &mut Vec<T> {
        let key = color_key(color);
        let i = match self.by_color.get(&key) {
            Some(&i) => i,
            None => {
                self.by_color.insert(key.clone(), self.entries.len());
                self.entries.push((key, color, Vec::new()));
                self.paints.push(None);
                self.entries.len() - 1
            }
        };
        &mut self.entries[i].2
    }
    /// The entry a range of `color` that names `paint` draws in: the last one when it names that paint with that
    /// colour, else a new one after it. Without a paint, [`Self::get`].
    fn get_painted(&mut self, color: [f32; 4], paint: Option<u32>) -> &mut Vec<T> {
        let Some(paint) = paint else {
            return self.get(color);
        };
        let joins = self.paints.last() == Some(&Some(paint)) && self.entries.last().is_some_and(|e| e.1 == color);
        if !joins {
            self.entries.push((String::new(), color, Vec::new()));
            self.paints.push(Some(paint));
        }
        &mut self.entries.last_mut().unwrap().2
    }
}

/// Pass kind — `VectorTextEffectPass.kind` (vectorTextEffects.ts:82).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffectPassKind {
    Shadow,
    Stroke,
    Glow,
    Echo,
    Extrude,
}

/// `VectorTextEffectPass` (vectorTextEffects.ts:81-103): one effect render
/// pass over the node's placed glyph instances (or, for extrude, its wall
/// geometry).
struct EffectPass {
    kind: EffectPassKind,
    color: [f32; 4],
    pen_offset_x: f64,
    pen_offset_y: f64,
    extra_pad_local_px: f64,
    blur_px: f64,
    stroke_px: f64,
    /// Legacy shadow only: REPLACES node opacity (MSDF parity).
    opacity_override: Option<f64>,
    /// Structured effects: MULTIPLIES node opacity (may exceed 1 — glow
    /// intensity).
    opacity_factor: Option<f64>,
    /// Glow: additive (emitted-light) blend variant.
    additive: bool,
    /// Echo pseudo-perspective: uniform per-pass glyph scale about the text
    /// box center. Default 1.
    scale: f64,
    /// Extrude only: (depthPx, angleDeg) — swept wall geometry instead of
    /// instanced glyph re-draws.
    extrude: Option<(f64, f64)>,
}

impl EffectPass {
    fn base(kind: EffectPassKind, color: [f32; 4]) -> EffectPass {
        EffectPass {
            kind,
            color,
            pen_offset_x: 0.0,
            pen_offset_y: 0.0,
            extra_pad_local_px: 0.0,
            blur_px: 0.0,
            stroke_px: 0.0,
            opacity_override: None,
            opacity_factor: None,
            additive: false,
            scale: 1.0,
            extrude: None,
        }
    }
}

/// `clamp01` (vectorTextEffects.ts:105-107).
fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

/// `lerpRgba` (vectorTextEffects.ts:109-111).
fn lerp_rgba(a: [f32; 4], b: [f32; 4], t: f64) -> [f32; 4] {
    let t = t as f32;
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// `Number.isFinite(x ?? d)` over an `Option<f64>` (JS `undefined ?? d`).
fn finite_or(v: Option<f64>, default: f64) -> Option<f64> {
    let x = v.unwrap_or(default);
    x.is_finite().then_some(x)
}

/// `appendEchoPasses` (vectorTextEffects.ts:126-160): expand one echo effect
/// into its per-copy passes, FARTHEST COPY FIRST.
fn append_echo_passes(
    count: Option<f64>,
    offset_x: f64,
    offset_y: f64,
    scale_step: Option<f64>,
    color_start: Option<&str>,
    color_end: Option<&str>,
    opacity_step: Option<f64>,
    passes: &mut Vec<EffectPass>,
) {
    // Non-finite counts render nothing; count < 1 renders nothing too.
    let Some(count) = count.filter(|c| c.is_finite()) else {
        return;
    };
    let count = count.round().min(ECHO_COUNT_MAX);
    if count < 1.0 {
        return;
    }
    let count = count as i64;
    let color_start_rgba = css_color_to_normalized_rgba(color_start.unwrap_or("#000000"))
        .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32])
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    // `effect.colorEnd ? (parse ?? colorStart) : colorStart` — empty string
    // is falsy in JS.
    let color_end_rgba = match color_end.filter(|s| !s.is_empty()) {
        Some(s) => css_color_to_normalized_rgba(s)
            .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32])
            .unwrap_or(color_start_rgba),
        None => color_start_rgba,
    };
    // Non-finite steps fall back to 1; scaleStep floors at 0, opacityStep
    // clamps to [0,1] (vectorTextEffects.ts:140-141).
    let scale_step = finite_or(scale_step, 1.0)
        .map(|s| s.max(0.0))
        .unwrap_or(1.0);
    let opacity_step = finite_or(opacity_step, 1.0).map(clamp01).unwrap_or(1.0);
    let mut k = count;
    while k >= 1 {
        let t = if count > 1 {
            (k - 1) as f64 / (count - 1) as f64
        } else {
            0.0
        };
        // Cumulative-scale floor/cap (vectorTextEffects.ts:145-147).
        let scale = scale_step.powi(k as i32).clamp(1e-3, ECHO_SCALE_MAX);
        passes.push(EffectPass {
            pen_offset_x: offset_x * k as f64,
            pen_offset_y: offset_y * k as f64,
            opacity_factor: Some(opacity_step.powi(k as i32)),
            scale,
            ..EffectPass::base(
                EffectPassKind::Echo,
                lerp_rgba(color_start_rgba, color_end_rgba, t),
            )
        });
        k -= 1;
    }
}

/// The effect passes of the text effect list, back to front (the `for effect
/// of args.effects` loop, vectorTextEffects.ts:187-232). `with_shadows`
/// includes the drop shadows (in list order); without them this is the
/// structured-effects set the engine's routing predicate
/// (`structured_effects_active`) tests, so both share one set of activity
/// rules.
fn effect_passes(effects: &[crate::ops::TextEffectSpec], with_shadows: bool) -> Vec<EffectPass> {
    use crate::ops::TextEffectSpec as E;
    let mut passes = Vec::new();
    for effect in effects {
        match effect {
            E::DropShadow(shadow) => {
                if !with_shadows {
                    continue;
                }
                let blur_px = shadow.blur.max(0.0);
                passes.push(EffectPass {
                    pen_offset_x: shadow.offset_x,
                    pen_offset_y: shadow.offset_y,
                    extra_pad_local_px: blur_px,
                    blur_px,
                    opacity_override: Some(shadow.opacity),
                    ..EffectPass::base(EffectPassKind::Shadow, css_rgba_or_black(&shadow.color))
                });
            }
            E::Glow {
                color,
                radius_px,
                intensity,
                opacity,
            } => {
                // Finite guards mirror vectorTextEffects.ts:189-199: radius 0
                // = no halo (render NOTHING, not an additive fill duplicate).
                let Some(radius_px) = radius_px.filter(|r| r.is_finite()) else {
                    continue;
                };
                let radius_px = radius_px.max(0.0);
                if radius_px <= 0.0 {
                    continue;
                }
                let opacity_factor =
                    clamp01(opacity.unwrap_or(1.0)) * intensity.unwrap_or(1.0).max(0.0);
                if !opacity_factor.is_finite() || opacity_factor <= 0.0 {
                    continue;
                }
                passes.push(EffectPass {
                    extra_pad_local_px: radius_px,
                    blur_px: radius_px,
                    opacity_factor: Some(opacity_factor),
                    additive: true,
                    ..EffectPass::base(EffectPassKind::Glow, css_rgba_or_black(color))
                });
            }
            E::Echo {
                count,
                offset_x,
                offset_y,
                scale_step,
                color_start,
                color_end,
                opacity_step,
            } => {
                let (Some(ox), Some(oy)) = (
                    offset_x.filter(|v| v.is_finite()),
                    offset_y.filter(|v| v.is_finite()),
                ) else {
                    continue;
                };
                append_echo_passes(
                    *count,
                    ox,
                    oy,
                    *scale_step,
                    color_start.as_deref(),
                    color_end.as_deref(),
                    *opacity_step,
                    &mut passes,
                );
            }
            E::Extrude {
                depth_px,
                angle_deg,
                color,
            } => {
                let Some(depth_px) = depth_px.filter(|d| d.is_finite()) else {
                    continue;
                };
                let depth_px = depth_px.max(0.0);
                if depth_px <= 0.0 {
                    continue;
                }
                let angle = angle_deg
                    .filter(|a| a.is_finite())
                    .unwrap_or(DEFAULT_EXTRUDE_ANGLE_DEG);
                passes.push(EffectPass {
                    extrude: Some((depth_px, angle)),
                    ..EffectPass::base(EffectPassKind::Extrude, css_rgba_or_black(color))
                });
            }
        }
    }
    passes
}

/// Engine routing predicate (task #142, Q-124): does this node carry
/// structured effects (glow / echo / extrude) that would actually emit a
/// render pass? Used by the msdf/hybrid lanes to route the node through the
/// vector arm — the only engine that renders them, in the browser and
/// natively. A drop shadow does not route (the MSDF lane draws it too), and
/// pass-inert entries (radius-0 glow, count-0 echo, depth-0 extrude,
/// malformed params) keep the node on its byte-identical default lane.
pub fn structured_effects_active(spec: &TextNodeSpec) -> bool {
    !effect_passes(&spec.effects, false).is_empty()
}

/// `resolveVectorTextEffectPasses` (vectorTextEffects.ts:168-250): the effect
/// list in node array order (drop shadows, glow, echo expanding farthest copy
/// first, extrude) → legacy stroke. Submission order is back-to-front.
fn resolve_effect_passes(spec: &TextNodeSpec) -> Vec<EffectPass> {
    let mut passes = effect_passes(&spec.effects, true);
    let stroke_width = spec.stroke_width.unwrap_or(0.0);
    if spec
        .stroke_color
        .as_deref()
        .filter(|s| !s.is_empty())
        .is_some()
        && stroke_width > 0.0
    {
        passes.push(EffectPass {
            extra_pad_local_px: stroke_width,
            stroke_px: stroke_width,
            ..EffectPass::base(
                EffectPassKind::Stroke,
                css_rgba_or_black(spec.stroke_color.as_deref().unwrap()),
            )
        });
    }
    passes
}

/// Which engine's vertical-placement semantics the vector lane uses for
/// `verticalAlign: middle|bottom` (FIX r2, task #128).
///
/// The two browser engines genuinely disagree here:
///
/// - the VECTOR engine offsets by the laid-out `totalHeightPx` — a bare
///   `Σ lineHeightPx` (`paragraphLayout.ts:80`, `drawRuns`:364-367);
/// - the CANVAS2D pipeline measures content height with a PER-PARAGRAPH
///   last-line content-box overflow term (`intrinsicSize.ts:70`:
///   `Math.max(0, lastContentBox - lastLineHeightPx)`) and positions the
///   raster quad by that measured height (`WebGPURichTextSystem` →
///   `measureRichTextContentSize`; ported in `degraded.rs`).
///
/// Whenever a font's ascent+descent exceeds the line height (negative
/// leading — e.g. Onest's 1.275 em content box under the default 1.2
/// line-height), the two place middle-aligned text `overflow/2` apart
/// (2.81 px at 74.94 px — the dominant term of the IB53 hybrid
/// regressions). `--text-engine vector` keeps `VectorParity` (browser
/// `?text=vector` truth, pinned by `vector_oracle_parity` /
/// goldens-vector); the HYBRID arm uses `Canvas2dParity` because its
/// comparison target is the default-config golden, where fallback-class
/// nodes are Canvas2D (quirk Q-120).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorPlacement {
    /// `drawRuns`:364-367 — offset from the laid-out `Σ lineHeightPx`.
    VectorParity,
    /// Canvas2D-pipeline placement: `Σ lineHeightPx` plus each paragraph's
    /// last-line content-box overflow (`intrinsicSize.ts:35-71`).
    Canvas2dParity,
}

/// The Canvas2D pipeline's measured content height over an ALREADY laid-out
/// vector layout — `measureParagraphsIntrinsicSize`'s height aggregation
/// (`intrinsicSize.ts:35-79`) replayed on the vector lines: per paragraph,
/// `Σ lineHeightPx` plus `max(0, lastContentBox − lastLineHeightPx)`, with
/// paragraph spacing between paragraphs. Content boxes come from the SAME
/// metrics provider the vector walk uses for baselines (TTF
/// `chooseBaselineMetrics`) — for chain-rewritten coverage chunks this is
/// the fallback family's metrics where the browser's intrinsic measure used
/// the styled (primary) family's; recorded deviation, see Q-120.
pub(crate) fn canvas2d_placement_height(
    layout: &crate::paragraph_layout::ParagraphLayoutResult,
    paragraph_spacing_px: f64,
    provider: &dyn crate::line_metrics::FontMetricsProvider,
) -> f64 {
    let mut total = 0.0f64;
    for (p, para) in layout.paragraphs.iter().enumerate() {
        if p > 0 {
            total += paragraph_spacing_px.max(0.0);
        }
        let mut last_line_height = 0.0f64;
        let mut last_content_box = 0.0f64;
        for line in &para.lines {
            total += line.line_height_px;
            let mut max_ascent = 0.0f64;
            let mut max_descent = 0.0f64;
            for seg in &line.segments {
                let (a, d) = provider.metrics(&seg.style);
                if a > max_ascent {
                    max_ascent = a;
                }
                if d > max_descent {
                    max_descent = d;
                }
            }
            last_line_height = line.line_height_px;
            last_content_box = max_ascent + max_descent;
        }
        if !para.lines.is_empty() {
            total += (last_content_box - last_line_height).max(0.0);
        }
    }
    total
}

/// Vertical placement offset for `verticalAlign: middle|bottom` — the pen
/// origin is `v_offset` on both arms.
///
/// - `VectorParity` (drawRuns:364-367): offset within the text box from
///   the laid-out `Σ lineHeightPx`.
/// - `Canvas2dParity` (Q-120 + Q-125): the Canvas2D pipeline aligns the
///   raster QUAD against the full node box —
///   `(node.height − textureLogicalHeight) / 2` for middle and the full
///   difference for bottom (`transforms.ts:96-100`) — where
///   `textureLogicalHeight` is the raster's CEILED logical height,
///   `ceil(intrinsic content height)`
///   (`measureRichTextContentSize`, canvas2dRichTextRenderer.ts:443), and
///   the intrinsic aggregation runs over the Chromium-rounded canvas
///   metrics the caller passes as `provider`.
pub(crate) fn placement_v_offset(
    placement: VectorPlacement,
    vertical_align: &str,
    box_h: f64,
    layout: &crate::paragraph_layout::ParagraphLayoutResult,
    paragraph_spacing_px: f64,
    provider: &dyn crate::line_metrics::FontMetricsProvider,
) -> f64 {
    let (span_h, placement_height) = match placement {
        VectorPlacement::VectorParity => (box_h, layout.total_height_px),
        VectorPlacement::Canvas2dParity => {
            let intrinsic = canvas2d_placement_height(layout, paragraph_spacing_px, provider);
            (box_h, intrinsic.ceil())
        }
    };
    match vertical_align {
        "middle" => (span_h - placement_height) / 2.0,
        "bottom" => span_h - placement_height,
        _ => 0.0,
    }
}

/// One line of a [`VectorLayout`]: its top and height (px, in the node), and the source code points of each of
/// its segments in order.
#[derive(Debug, Clone)]
pub struct LaidLine {
    pub top: f64,
    pub height: f64,
    pub source: Vec<crate::types::SourceIndices>,
}

/// One glyph of a placed segment: where its pen is and which curves of the layout's buffer draw it.
#[derive(Debug, Clone, Copy)]
struct PlacedGlyph {
    gid: u32,
    pen_x: f64,
    /// Offset from the baseline, y-up px (`place_glyphs`).
    y_offset_px: f64,
    /// Its index among the segment's shaped glyphs.
    glyph_idx: usize,
    first_curve: u32,
    curve_count: u32,
    /// Unpadded curve bounds in em: min x, min y, max x, max y.
    bounds: [f32; 4],
}

/// A line's segment as the line walk places it (`walkLines`' `PlacedSegment`).
struct PlacedSegment {
    /// The paint its range names (`InlineStyle::paint`): `TextNodeSpec::run_colors` colours it.
    paint: Option<u32>,
    /// Each glyph at its pen with its curves in the layout's buffer, in the order the draw visits them.
    placed: Vec<PlacedGlyph>,
    font_key: String,
    family: String,
    weight: i64,
    style: &'static str,
    glyphs: Vec<crate::vector::measure::ShapedGlyph>,
    font_size_px: f64,
    letter_spacing_px: f64,
    start_x: f64,
    baseline_y: f64,
    color: [f32; 4],
    /// Original source index of each shaped character, retained through layout.
    source: crate::types::SourceIndices,
    /// Italic requested, no italic face: shear the upright glyphs.
    synthetic_oblique: bool,
    /// E6.3: the run's axis settings, so the draw loads the instance it shaped with.
    variations: Vec<crate::vector::varfont::Variation>,
    /// F7.2: how far the glyphs move along their own down direction so the chosen line (Align to Path) sits
    /// on the path.
    path_shift: f64,
    /// The run's gradient (`--nf-text-fill`), drawn when no node-level fill overrides the runs.
    fill: Option<crate::ops::GradientSpec>,
    /// E8.22: baseline shift, scale and rotation (the advances already carry scale and kerning).
    character: crate::character::CharacterStyle,
    /// Vertical text: the line's column, how this piece draws, its line's baseline and its ascender / descender
    /// (px) — each glyph is turned into the column (`crate::vertical::place`).
    column: Option<(crate::vertical::ColumnFrame, crate::vertical::GlyphOrient, f64, f64, f64)>,
}

/// An emoji's image slot as the line walk places it.
struct PlacedEmoji {
    cluster: String,
    x: f64,
    baseline_y: f64,
    font_size_px: f64,
    path_shift: f64,
    /// The emoji's first code point: KineticText's index into `unitTransforms` (its unit moves it like a glyph).
    code_point: Option<usize>,
    /// Vertical text: the line's column (the emoji stands upright on its centre line).
    column: Option<crate::vertical::ColumnFrame>,
}

/// One text node laid out on the vector lane ([`accept_vector_layout`]): immutable, and the one layout its box,
/// lines, ink boxes and every draw read. It depends on what lays text out and on nothing of where a node draws, so
/// nodes with the same text, fonts and box share one. It holds its runs' colours and gradients as the HTML gives
/// them; a draw that supplies its own colours (`TextNodeSpec::run_colors`) shares it across colours too.
pub struct VectorLayout {
    /// The node's box (px): the given size, or for a fitted box its widest line.
    box_w: f64,
    box_h: f64,
    /// The first paragraph's alignment (where text starts along a path).
    first_align: &'static str,
    total_height_px: f64,
    lines: Vec<LaidLine>,
    segments: Vec<PlacedSegment>,
    emojis: Vec<PlacedEmoji>,
    bullet_primitives: Buckets<f32>,
    decoration_primitives: Buckets<f32>,
    all_primitive_tris: Vec<f32>,
    /// F7.2: the path the text runs along, with the text's sizes along it (ascending) and its width.
    path: Option<(crate::text_path::TextPathSpec, crate::text_path::PathMeasure)>,
    path_sizes: Vec<(f64, f64)>,
    path_text_width: f64,
    /// Every placed glyph's curves: 6 floats per quadratic segment, em units, y-down, pen origin (the
    /// `GlyphCurveStore` payload), deduped per (font instance, glyph) in the order the draw visits them.
    curve_points: std::sync::Arc<Vec<f32>>,
}

impl VectorLayout {
    /// The box's width (px): the node's, or for a fitted box the widest line.
    pub fn box_width(&self) -> f64 {
        self.box_w
    }

    /// The laid-out text's height (px).
    pub fn height(&self) -> f64 {
        if self.total_height_px.is_finite() {
            return self.total_height_px;
        }
        self.lines.iter().map(|line| line.top + line.height).fold(0.0, f64::max)
    }

    /// The lines this layout draws, top to bottom.
    pub fn lines(&self) -> &[LaidLine] {
        &self.lines
    }

    /// Each source code point's ink box at rest (node px: left, top, right, bottom), of `count` code points: the
    /// union of the outline bounds of the glyphs placed for it (a ligature's glyph counts for its first code
    /// point). A segment with a glyph its font lacks has no boxes.
    pub fn code_point_boxes(&self, count: usize) -> Vec<Option<[f64; 4]>> {
        let mut boxes: Vec<Option<[f64; 4]>> = vec![None; count];
        for seg in &self.segments {
            if seg.glyphs.is_empty() || seg.glyphs.iter().any(|g| g.glyph == 0) {
                continue;
            }
            let size = seg.font_size_px;
            for g in &seg.placed {
                let cp = seg.glyphs.get(g.glyph_idx).and_then(|shaped| seg.source.get(shaped.cluster));
                let Some(cp) = cp.filter(|&cp| cp < count) else { continue };
                if g.curve_count == 0 {
                    continue;
                }
                let y = seg.baseline_y - g.y_offset_px - seg.character.baseline_shift;
                let b = [
                    g.pen_x + f64::from(g.bounds[0]) * size,
                    y + f64::from(g.bounds[1]) * size,
                    g.pen_x + f64::from(g.bounds[2]) * size,
                    y + f64::from(g.bounds[3]) * size,
                ];
                boxes[cp] = Some(match boxes[cp] {
                    None => b,
                    Some(a) => [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])],
                });
            }
        }
        boxes
    }

    /// Place every segment's glyphs and read their outlines into the layout's curve buffer, in the order the
    /// draw visits them (`GlyphCurveStore.ensureGlyph`): kerning and outline errors unwind to the node's catch.
    fn outline(&mut self, store: &VectorFontStore, catalog: &Catalog) -> Result<(), ShapeError> {
        let mut curve_points: Vec<f32> = Vec::new();
        let mut curve_ranges: HashMap<(String, u32), (u32, u32, [f32; 4])> = HashMap::new();
        for seg in &mut self.segments {
            let instance = crate::vector::measure::Instance {
                size_px: Some(seg.font_size_px),
                variations: &seg.variations,
            };
            // A glyph's curves are the instance's: a span's `font-variation-settings` (or another size of an
            // optical-size face) is the same variant with other outlines, so it can't share the variant's entries.
            let curve_font_key = format!(
                "{}{}",
                seg.font_key,
                store.instance_suffix(catalog, &seg.family, seg.weight, seg.style, instance)
            );
            let font = store
                .font_for_variant(catalog, &seg.family, seg.weight, seg.style, instance)
                .expect("font resolved in walk");
            // placeGlyphs (kerning + letter spacing) over the shaped glyphs.
            let mut pens: Vec<(u32, f64, f64, usize)> = Vec::new();
            place_glyphs(
                &font,
                &seg.glyphs,
                seg.font_size_px,
                seg.start_x,
                |g, pen_x, y_offset_px, idx| pens.push((g, pen_x, y_offset_px, idx)),
                seg.letter_spacing_px,
            );
            let mut placed = Vec::with_capacity(pens.len());
            for (gid, pen_x, y_offset_px, glyph_idx) in pens {
                let key = (curve_font_key.clone(), gid);
                let (first_curve, curve_count, bounds) = match curve_ranges.get(&key) {
                    Some(r) => *r,
                    None => {
                        let q = font.glyph_quadratics(gid)?;
                        let first = (curve_points.len() / 6) as u32;
                        curve_points.extend_from_slice(&q.points);
                        let entry = (first, q.curve_count, [q.min_x, q.min_y, q.max_x, q.max_y]);
                        curve_ranges.insert(key, entry);
                        OUTLINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        entry
                    }
                };
                placed.push(PlacedGlyph { gid, pen_x, y_offset_px, glyph_idx, first_curve, curve_count, bounds });
            }
            seg.placed = placed;
        }
        self.curve_points = std::sync::Arc::new(curve_points);
        Ok(())
    }
}

/// A glyph instance retained for the effect passes (`PlacedGlyphInstance`,
/// vectorTextEffects.ts:105-112).
struct PlacedGlyphInstance {
    pen_x: f64,
    baseline_y: f64,
    font_size_px: f64,
    base_pad_em: f64,
    em_bounds: [f32; 4], // unpadded curve-store bounds
    first_curve: u32,
    curve_count: u32,
    /// KineticText per-glyph transform — inherited by every effect
    /// pass so glow/echo/shadow/stroke track the glyph; identity for static text.
    xform: GlyphXform,
}

/// Why the vector lane does not draw a node: the browser would not draw it through the vector path
/// (`selectTextEngine` chose another engine, run resolution failed, or the vector path threw: `renderWithVector`'s
/// catch → `renderWithMsdf`).
#[derive(Debug, Clone)]
pub struct VectorDecline {
    pub reason: String,
    /// A LOW-CARDINALITY class for the decline ([`VectorEngineDecision::Fallback`]).
    pub cause: String,
    /// The font families implicated in this decline.
    pub families: Vec<String>,
    /// The text laid out and a glyph's outline could not be read.
    pub outlines: bool,
}

static LAYOUTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static OUTLINES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many layouts this process has accepted ([`accept_vector_layout`]) and how many glyph outlines they emitted
/// into their curve buffers.
pub fn layout_counts() -> [u64; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    [LAYOUTS.load(Relaxed), OUTLINES.load(Relaxed)]
}

/// Lay one node out on the vector lane, once: the routing gates, the resolved runs, the wrapped lines, every glyph
/// at its pen and every outline. A node's box, lines, ink boxes and draws ([`VectorLayout::ops`]) all read the
/// result. `spec` is read for what lays text out (its HTML, box, spacing, line-height policy, vertical alignment,
/// truncation, path and area); a draw may replace the runs' colours (`TextNodeSpec::run_colors`); `fit_width` makes the box as wide as the widest line (see [`VectorLayout::box_width`]).
pub fn accept_vector_layout(
    spec: &TextNodeSpec,
    store: &VectorFontStore,
    catalog: &Catalog,
    placement: VectorPlacement,
    fit_width: bool,
) -> Result<VectorLayout, VectorDecline> {
    let routed = |reason: String, cause: String, families: Vec<String>| VectorDecline {
        reason,
        cause,
        families,
        outlines: false,
    };
    // selectTextEngine with the vector flag ON (WebGPUTextSystem.ts:339-370).
    let engine_choice = select_text_engine_with(&spec.html, catalog, EmojiRuns::Draw);
    if engine_choice != "vector" {
        return Err(routed(
            format!("select-text-engine:{engine_choice}"),
            format!("select-text-engine:{engine_choice}"),
            // `select_text_engine` returns non-vector precisely when NO run's family
            // is an msdf family — i.e. "the document's fonts aren't in our catalog
            // at all", which makes the family names the single most useful datum
            // here; reporting `[]` threw away the answer. This DOES
            // re-parse the html (a third parse on this path), which is acceptable
            // only because the branch is reached exactly when the node has no
            // catalog family — i.e. little or no shapeable text to re-parse. Don't
            // copy the pattern onto a hot path.
            crate::runs::extraction_rejected_families(&spec.html, catalog),
        ));
    }

    // Complex-script / RTL / bidi gate: the otjs vector shaper maps code points
    // 1:1 with no bidi, joining, reordering or GPOS mark positioning (it even
    // hard-throws on Arabic), so it renders these scripts WRONG or blank.
    // Decline the whole node so it routes to the cosmic-text lane
    // (`crate::degraded`, rustybuzz + `unicode-bidi`), which shapes them
    // correctly. Latin/Greek/Cyrillic/CJK/Kana/Hangul are excluded and keep the
    // crisp GPU-outline path. See `crate::complex_script`.
    if crate::complex_script::text_needs_complex_shaping(&spec.html) {
        return Err(routed("complex-script".to_string(), "complex-script".to_string(), Vec::new()));
    }
    // Likewise a right-to-left paragraph (`crate::paragraph_direction`) whose base level reorders its text (a final
    // '!' on the left): the cosmic-text lane reorders it; this lane draws in logical order.
    if crate::paragraph_direction::html_needs_rtl_reorder(&spec.html) {
        return Err(routed("complex-script".to_string(), "rtl-paragraph".to_string(), Vec::new()));
    }

    // renderWithVector:654-655 — zero extracted runs → false (MSDF fallback).
    let runs = extract_msdf_runs_by_paragraph(&spec.html, catalog);
    if runs.is_empty() {
        return Err(routed("no-runs".to_string(), "no-runs".to_string(), Vec::new()));
    }

    // resolveVectorRuns (VectorTextRenderer.ts:1098-1133).
    let mut resolved = match resolve_vector_runs_with(&runs, store, catalog, EmojiRuns::Draw) {
        RunResolution::Resolved(r) => r,
        RunResolution::Fallback {
            reason,
            family,
            diag,
        } => {
            // `reason[:family]` is the historical shape (the taxonomy in the
            // resource report prefix-matches it); this appends the variant +
            // differentiated miss class in parentheses so a log line no longer
            // needs a canvas join to say WHICH weight/style missed and WHY.
            let mut reason = match &family {
                Some(f) => format!("{reason}:{f}"),
                None => reason.to_string(),
            };
            if let Some(d) = &diag.detail {
                reason.push_str(&format!(" ({d})"));
            }
            // Families from `diag`, NOT from `family`: the latter is the oracle dump's
            // contract and is `None` for `coverage`, which is how coverage
            // declines kept reporting `families: []`.
            return Err(routed(reason, diag.cause, diag.families));
        }
    };
    // Run-color quantization — the shared extract→convert pipeline's color
    // handling, applied exactly where the MSDF lane applies it (ops.rs:362).
    // (Run text is already sanitized inside resolve_vector_runs.)
    for r in &mut resolved {
        r.color = rgba_normalized_array_to_css(normalize_run_color(&r.color));
    }
    // Vertical text (`crate::vertical`): each run cut into upright, sideways and combined pieces. Not along a path or
    // in area type, which lay out horizontally.
    let writing_mode = crate::vertical::runs_writing_mode(&resolved);
    if writing_mode.is_vertical() && spec.text_path.is_none() && spec.area.is_none() {
        resolved = crate::vertical::split_runs(&resolved, writing_mode);
    }

    // renderWithVector's catch (WebGPUTextSystem.ts:758-761): ANY vector-
    // path throw → MSDF fallback for the node.
    let thrown = |e: ShapeError, outlines: bool| VectorDecline {
        reason: format!("vector-error: {}", e.message()),
        cause: "vector-error".to_string(),
        families: Vec::new(),
        outlines,
    };
    let mut layout = place_vector_layout(spec, &resolved, store, catalog, placement, fit_width)
        .map_err(|e| thrown(e, false))?;
    layout.outline(store, catalog).map_err(|e| thrown(e, true))?;
    LAYOUTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(layout)
}

/// Build the ordered vector op stream for one node, or the routing fallback: its layout
/// ([`accept_vector_layout`]) and that layout's draws ([`VectorLayout::ops`]).
///
/// `analytic_pad_scale` mirrors `VectorDrawArgs.analyticPadScale`
/// (WebGPUTextSystem.ts:712-715): min real local→screen axis scale ×
/// viewZoom × DPR — on the export path, min(model sx, sy) × export scale.
pub fn build_vector_node_ops(
    spec: &TextNodeSpec,
    store: &VectorFontStore,
    catalog: &Catalog,
    analytic_pad_scale: f64,
    placement: VectorPlacement,
) -> VectorEngineDecision {
    let layout = match accept_vector_layout(spec, store, catalog, placement, false) {
        Ok(layout) => layout,
        Err(VectorDecline { reason, cause, families, .. }) => {
            return VectorEngineDecision::Fallback { reason, cause, families }
        }
    };
    match layout.ops(spec, store, catalog, analytic_pad_scale) {
        Ok(ops) => VectorEngineDecision::Vector(ops),
        // renderWithVector's catch (WebGPUTextSystem.ts:758-761): ANY vector-
        // path throw → MSDF fallback for the node.
        Err(e) => VectorEngineDecision::Fallback {
            reason: format!("vector-error: {}", e.message()),
            cause: "vector-error".to_string(),
            families: Vec::new(),
        },
    }
}

/// The widest line of wrapped text placed in a box `inline_w` wide: per line, from its leftmost segment's start to
/// its rightmost segment's end, as the line walk positions them (alignment, indents, list offset, justification).
fn widest_line(
    layout: &crate::paragraph_layout::ParagraphLayoutResult,
    inline_w: f64,
    measurer: &VectorMeasurer<'_>,
) -> f64 {
    let mut widest: f64 = 0.0;
    for para in &layout.paragraphs {
        let is_list_paragraph = para.list_kind.is_some();
        let indent = compute_list_indentation(
            para.list_kind.as_deref(),
            para.list_level,
            para.font_size,
            None,
            None,
        );
        for line in &para.lines {
            let total_width = measure_line_width(line, measurer);
            let (container_x, container_w) = crate::positioning::indented_container(line, 0.0, inline_w);
            let line_start_x = compute_line_start_x(
                crate::paragraph_direction::line_start_align(para.align, para.rtl, line),
                total_width,
                container_x,
                container_w,
                is_list_paragraph,
            );
            let justify_px = justify_space_px(para.align, line, total_width, container_w - indent.left_offset_px);
            let positions =
                compute_segment_positions_justified(line, line_start_x, indent.left_offset_px, measurer, justify_px);
            let (mut left, mut right) = (f64::INFINITY, f64::NEG_INFINITY);
            for pos in &positions {
                if !(pos.start_x.is_finite() && pos.width.is_finite()) {
                    continue;
                }
                left = left.min(pos.start_x);
                right = right.max(pos.start_x + pos.width);
            }
            if right > left {
                widest = widest.max(right - left);
            }
        }
    }
    widest
}

#[allow(clippy::too_many_lines)]
fn place_vector_layout(
    spec: &TextNodeSpec,
    resolved: &[crate::runs::MsdfRun],
    store: &VectorFontStore,
    catalog: &Catalog,
    placement: VectorPlacement,
    fit_width: bool,
) -> Result<VectorLayout, ShapeError> {
    // renderWithVector:657-659 box coercions.
    let box_w = spec.width.unwrap_or(400.0).max(0.0);
    let box_h = spec.height.unwrap_or(200.0).max(0.0);
    let paragraph_spacing_px = spec.paragraph_spacing.max(0.0);
    // Vertical text: lines are laid out across the node's height (the inline size) and stacked across its width,
    // then turned into columns (`crate::vertical::ColumnFrame`).
    let writing_mode = crate::vertical::runs_writing_mode(resolved);
    let vertical = writing_mode.is_vertical() && resolved.iter().any(|r| r.vertical.glyph.is_some());

    // buildLayout (VectorTextRenderer.ts:1073-1087).
    let paragraphs = convert_msdf_runs_to_paragraphs(resolved);
    // Q-121: the measurer carries the placement parity — `Canvas2dParity`
    // (hybrid arm) measures letter-spaced runs with Canvas2D semantics, so
    // wrap, alignment, segment positions and decoration widths all agree
    // with the golden's engine.
    let measurer = VectorMeasurer::with_placement(store, catalog, placement);
    let segmenter = |word: &str, rules| segment_text_for_wrap_with(word, rules);
    // F7.2 type on a path: the text is one run along the path, so paragraphs don't wrap.
    let on_path = spec.text_path.as_ref().and_then(|p| {
        crate::text_path::PathMeasure::new(&p.points, p.closed, p.flip).map(|m| (p, m))
    });
    // Metrics: the vector engine's own lane keeps the float TTF metrics
    // (makeMetricsProvider); the hybrid arm's `Canvas2dParity` uses the
    // golden engine's `getCanvasFontMetrics` truth — Chromium-rounded
    // integer ascent/descent (Q-125) — for baselines AND the intrinsic
    // placement height.
    let vector_metrics = VectorMetricsProvider { store, catalog };
    let canvas2d_metrics = Canvas2dFontMetricsProvider(VectorMetricsProvider { store, catalog });
    let metrics_provider: &dyn crate::line_metrics::FontMetricsProvider = match placement {
        VectorPlacement::VectorParity => &vector_metrics,
        VectorPlacement::Canvas2dParity => &canvas2d_metrics,
    };
    // Area type fits each line into its shape, thread and wrap objects (`crate::area`); not along a path.
    let area_spec = spec.area.as_ref().filter(|_| on_path.is_none());
    let (layout, area) = match area_spec {
        Some(a) => {
            let (l, al) = crate::area::layout_area(
                &paragraphs,
                a,
                paragraph_spacing_px,
                &measurer,
                &metrics_provider,
                Some(&segmenter),
            );
            (l, Some(al))
        }
        None => (
            layout_paragraphs_to_lines_with_policy(
                &paragraphs,
                if on_path.is_some() {
                    1.0e7
                } else if vertical {
                    spec.inline_wrap_width()
                } else {
                    spec.wrap_width()
                },
                paragraph_spacing_px,
                &measurer,
                Some(&segmenter),
                spec.line_height_policy,
            ),
            None,
        ),
    };
    if let Some(e) = measurer.take_poison() {
        return Err(e);
    }
    // A box fitted to its text is as wide as the widest line laid out in the box it was given, so alignment places
    // the lines within that width. The lines themselves are the ones just wrapped.
    let box_w = if fit_width {
        let fitted = widest_line(&layout, if vertical { box_h } else { box_w }, &measurer);
        if let Some(e) = measurer.take_poison() {
            return Err(e);
        }
        fitted
    } else {
        box_w
    };
    let (inline_w, block_h) = if vertical { (box_h, box_w) } else { (box_w, box_h) };
    let box_w_node = box_w;
    // E6.3: hide the lines that overflow and end the last one in an ellipsis (area type hides its overflow).
    let mut layout = layout;
    if let (Some(t), None, None) = (&spec.truncation, &on_path, &area) {
        crate::truncation::truncate_layout(&mut layout, t, inline_w, paragraph_spacing_px, &measurer);
    }

    // drawRuns:364-367 — vertical offset from the LAID-OUT height (the vector
    // engine's divergence from MSDF's legacy height walk, quirk Q-107).
    // Under `Canvas2dParity` (the hybrid arm, Q-120 + Q-125) the offset is
    // the Canvas2D pipeline's raster-quad alignment instead — see
    // `placement_v_offset`.
    let vertical_align = match spec.vertical_align.as_deref() {
        Some(s) if !s.is_empty() => s,
        _ => "top",
    };
    let v_offset = if on_path.is_some() || area.is_some() {
        0.0
    } else {
        placement_v_offset(
            placement,
            vertical_align,
            block_h,
            &layout,
            paragraph_spacing_px,
            metrics_provider,
        )
    };

    // ── walkLines (VectorTextRenderer.ts:387-494) ──
    let mut segments: Vec<PlacedSegment> = Vec::new();
    let mut lines: Vec<LaidLine> = Vec::new();

    // Emoji: each layout placeholder takes the next emoji run in `convertMsdfRunsToParagraphs`
    // order (`emoji_<n>`); the cursor advances on every placeholder, drawn or not, so a line
    // another frame draws can't shift later emoji onto the wrong cluster.
    let emoji_map = crate::ops::emoji_runs_in_layout_order(resolved);
    let mut next_emoji: usize = 0;
    let mut placed_emojis: Vec<PlacedEmoji> = Vec::new();
    let count_placeholders = |line: &crate::types::Line| -> usize {
        line.segments
            .iter()
            .map(|s| s.text.chars().filter(|&c| c == crate::runs::EMOJI_PLACEHOLDER).count())
            .sum()
    };

    let mut bullet_primitives: Buckets<f32> = Buckets::new();
    let mut decoration_primitives: Buckets<f32> = Buckets::new();
    let mut all_primitive_tris: Vec<f32> = Vec::new();

    let x = 0.0;
    let mut pen_y = v_offset;
    // F7.2: along a path every line follows the one before it, a space apart; `path_cursor` is where the next starts.
    let mut path_cursor = 0.0f64;
    let mut path_text_width = 0.0f64;
    let mut path_sizes: Vec<(f64, f64)> = Vec::new();
    for (p, para) in layout.paragraphs.iter().enumerate() {
        pen_y = advance_pen_y_for_paragraph_break(pen_y, p == 0, paragraph_spacing_px);

        let is_list_paragraph = para.list_kind.is_some();
        let bullet_style = para.list_style.filter(|s| s.is_bullet());
        let indent = compute_list_indentation(
            para.list_kind.as_deref(),
            para.list_level,
            para.font_size,
            None,
            None,
        );

        for (line_index, line) in para.lines.iter().enumerate() {
            // Area type: the line's slot in this frame; lines another frame draws (or the overflow) are skipped.
            let slot = match &area {
                Some(a) => match a.visible(p, line_index) {
                    Some(s) => Some(*s),
                    None => {
                        next_emoji += count_placeholders(line);
                        continue;
                    }
                },
                None => None,
            };
            let (container_x, box_w) = slot.map(|s| (s.x, s.width)).unwrap_or((x, inline_w));
            // The paragraph's indents (`crate::paragraph_format`); along a path the text runs from the start.
            let (container_x, box_w) = if on_path.is_some() {
                (container_x, box_w)
            } else {
                crate::positioning::indented_container(line, container_x, box_w)
            };
            let total_width = measure_line_width(line, &measurer);
            let line_start_x = if on_path.is_some() {
                path_cursor
            } else {
                compute_line_start_x(
                    crate::paragraph_direction::line_start_align(para.align, para.rtl, line),
                    total_width,
                    container_x,
                    box_w,
                    is_list_paragraph,
                )
            };
            let list_offset = if on_path.is_some() {
                0.0
            } else {
                indent.left_offset_px
            };
            // Justified text widens its spaces (not along a path).
            let justify_px = if on_path.is_some() {
                0.0
            } else {
                justify_space_px(para.align, line, total_width, box_w - list_offset)
            };
            let positions = compute_segment_positions_justified(
                line,
                line_start_x,
                list_offset,
                &measurer,
                justify_px,
            );
            let baseline = compute_baseline_for_line_with_fallback(
                line,
                line.line_height_px,
                para.font_size,
                &metrics_provider,
            );
            let baseline_y = if on_path.is_some() {
                0.0
            } else if let Some(s) = &slot {
                s.top() + baseline.baseline_from_top_px
            } else {
                pen_y + baseline.baseline_from_top_px
            };
            if on_path.is_some() {
                // The text's sizes along the path: each word and space at its own, the gap to the next line at the
                // paragraph's (curvature spacing, `text_path::PathSizes`).
                path_sizes.extend(
                    line.segments
                        .iter()
                        .zip(positions.iter())
                        .map(|(seg, pos)| (pos.start_x, seg.style.font_size)),
                );
                path_sizes.push((path_cursor + total_width, para.font_size));
                path_text_width = path_cursor + total_width;
                path_cursor += total_width + 0.3 * para.font_size;
            }
            // Vertical text: this line's column (area type and paths are horizontal).
            let column = (vertical && on_path.is_none() && slot.is_none()).then_some(crate::vertical::ColumnFrame {
                mode: writing_mode,
                block_size: box_w_node,
                line_top: pen_y,
                line_height: line.line_height_px,
            });

            // List bullet on the first visual line (walkLines:432-453): a disc, a ring or a square (E7.16).
            if let Some(bullet_style) = bullet_style.filter(|_| line_index == 0 && on_path.is_none()) {
                let style = line.segments.first().map(|s| &s.style);
                // `style?.color || para.paragraphColor || 'rgba(0,0,0,1)'`
                let color_css = style
                    .map(|s| s.color.clone())
                    .filter(|c| !c.is_empty())
                    .or_else(|| para.paragraph_color.clone().filter(|c| !c.is_empty()))
                    .unwrap_or_else(|| "rgba(0,0,0,1)".to_string());
                let color = css_rgba_or_black(&color_css);
                // The bullet takes its paint with the first segment's colour.
                let paint = style.filter(|s| !s.color.is_empty()).and_then(|s| s.paint);
                let m = compute_bullet_metrics(
                    para.font_size,
                    bullet_x_height_px(store, catalog, style, para.font_size),
                    Some(baseline.ascent),
                );
                let center_y =
                    baseline_y + m.center_y_offset_from_baseline + m.vertical_adjustment_px;
                let text_start_x = line_start_x + indent.left_offset_px;
                let center_x = compute_bullet_center_x(
                    text_start_x,
                    indent.marker_box_width_px,
                    indent.marker_gap_px,
                    indent.left_offset_px,
                    m.horizontal_adjustment_px,
                );
                let (mut verts, indices) = crate::list_markers::bullet_geometry(
                    bullet_style,
                    center_x,
                    center_y,
                    m.radius_px,
                    BULLET_DISC_SEGMENTS,
                );
                if let Some(c) = &column {
                    map_xy_into_column(&mut verts, c);
                }
                append_indexed_triangles_xy(
                    bullet_primitives.get_painted(color, paint),
                    &verts,
                    &indices,
                    0.0,
                    0.0,
                );
                append_indexed_triangles_xy(&mut all_primitive_tris, &verts, &indices, 0.0, 0.0);
            }

            // E7.16: a numbered item's counter (`3.`, `b.`, `iv.`), right-aligned in the hanging indent of its first
            // line, drawn like a segment of its own before the line's text.
            let marker = if line_index == 0 && on_path.is_none() {
                crate::list_markers::counter_marker_segment(
                    para,
                    line_start_x + indent.left_offset_px,
                    indent.marker_gap_px,
                    &measurer,
                )
            } else {
                None
            };
            // Tab leaders (`crate::tab_stops`): each copy of the leader string drawn like a segment of its own, on the
            // paragraph's grid.
            let leaders = crate::tab_stops::leader_segments(line, &positions);
            let draw_segments = marker
                .iter()
                .map(|(seg, x)| (seg, *x, true))
                .chain(
                    line.segments
                        .iter()
                        .zip(positions.iter())
                        .map(|(seg, pos)| (seg, pos.start_x, false)),
                )
                .chain(leaders.iter().map(|(seg, x)| (seg, *x, true)));

            // Glyph segments (whitespace-only segments position but draw nothing).
            for (seg, seg_start_x, is_marker) in draw_segments {
                if seg.text.contains(crate::runs::EMOJI_PLACEHOLDER) {
                    // An emoji segment: each placeholder is an image slot at its prefix's width
                    // (the MSDF walk's rule for coalesced placeholders); no glyphs to shape.
                    // The same shift a glyph segment of this size takes, so the emoji keeps the
                    // text's baseline on the path.
                    let path_shift = match &on_path {
                        Some((tp, _)) => {
                            let (ascent, descent) = metrics_provider.metrics(&seg.style);
                            match tp.align {
                                crate::text_path::TextPathAlign::Baseline => 0.0,
                                crate::text_path::TextPathAlign::Ascender => ascent,
                                crate::text_path::TextPathAlign::Descender => -descent,
                                crate::text_path::TextPathAlign::Center => (ascent - descent) / 2.0,
                            }
                        }
                        None => 0.0,
                    };
                    let mut prefix = String::new();
                    for (source_offset, ch) in seg.text.chars().enumerate() {
                        if ch != crate::runs::EMOJI_PLACEHOLDER {
                            prefix.push(ch);
                            continue;
                        }
                        let id = next_emoji;
                        next_emoji += 1;
                        let Some(run) = emoji_map.get(id) else {
                            prefix.push(ch);
                            continue;
                        };
                        let code_point = if !is_marker { seg.style.source.get(source_offset) } else { None };
                        placed_emojis.push(PlacedEmoji {
                            cluster: run.text.clone(),
                            x: seg_start_x + crate::measure::TextMeasurer::measure(&measurer, &prefix, &seg.style),
                            baseline_y,
                            font_size_px: seg.style.font_size,
                            path_shift,
                            code_point,
                            column,
                        });
                        prefix.push(ch);
                    }
                    continue;
                }
                if seg.text.is_empty() || seg.text.chars().all(js_whitespace) {
                    continue;
                }
                let (family, weight, st) = style_to_variant(&seg.style);
                let size = crate::vector::measure::Instance::of(&seg.style);
                let Some(font) = store.font_for_variant(catalog, &family, weight, st, size) else {
                    continue; // unreachable: the pre-pass resolved every font
                };
                // Q-121: the hybrid arm (`Canvas2dParity`) places letter-
                // spaced text the way the Canvas2D fallback raster draws it —
                // one fillText per char (unkerned, no cross-char GSUB).
                // Q-122 (Phase 2): ls==0 hybrid segments shape with HarfBuzz
                // semantics (rustybuzz — extension-lookup kerning + default
                // GSUB + per-segment bidi), the golden Canvas2D engine's own
                // shaper; an unavailable HB face falls back to the opentype
                // path like the browser shaper's null return. The vector
                // lane keeps the browser vector engine's dispatch: otjs, with
                // the HarfBuzz reroute for fonts otjs cannot shape.
                let glyphs = crate::vector::measure::shape_segment(
                    store, catalog, &family, weight, st, size, &font, &seg.text, &seg.style, placement,
                )?;
                let source = if !is_marker { seg.style.source.clone() } else { Default::default() };
                let path_shift = match &on_path {
                    Some((tp, _)) => {
                        let (ascent, descent) = metrics_provider.metrics(&seg.style);
                        match tp.align {
                            crate::text_path::TextPathAlign::Baseline => 0.0,
                            crate::text_path::TextPathAlign::Ascender => ascent,
                            crate::text_path::TextPathAlign::Descender => -descent,
                            crate::text_path::TextPathAlign::Center => (ascent - descent) / 2.0,
                        }
                    }
                    None => 0.0,
                };
                segments.push(PlacedSegment {
                    paint: seg.style.paint,
                    placed: Vec::new(),
                    path_shift,
                    variations: seg.style.font_variations.clone(),
                    font_key: style_font_key(&seg.style),
                    family: family.clone(),
                    weight,
                    style: st,
                    glyphs,
                    font_size_px: seg.style.font_size,
                    letter_spacing_px: seg.style.letter_spacing,
                    start_x: seg_start_x,
                    baseline_y,
                    color: css_rgba_or_black(&seg.style.color),
                    source,
                    synthetic_oblique: seg.style.font_style == SYNTHETIC_OBLIQUE,
                    fill: seg.style.fill.clone(),
                    character: seg.style.character,
                    column: column.map(|c| {
                        let (ascent, descent) = metrics_provider.metrics(&seg.style);
                        (
                            c,
                            seg.style.vertical.glyph.unwrap_or(crate::vertical::GlyphOrient::Sideways),
                            baseline_y,
                            ascent,
                            descent,
                        )
                    }),
                });
                // Ruby (`crate::ruby`): the annotation, a segment of its own over the base (right of it in a column).
                if seg.style.ruby.text.is_some() && on_path.is_none() && !is_marker {
                    let mut bare = seg.style.clone();
                    bare.ruby = Default::default();
                    let unpadded = crate::vector::measure::shape_segment(
                        store, catalog, &family, weight, st, size, &font, &seg.text, &bare, placement,
                    )?;
                    if let Some(plan) = crate::ruby::plan(&font, &unpadded, &seg.style) {
                        let upm = if font.units_per_em != 0 { font.units_per_em as f64 } else { 1000.0 };
                        let base_scale = seg.style.font_size / upm;
                        let rstyle = crate::ruby::annotation_style(&seg.style, column.is_some());
                        let rsize = crate::vector::measure::Instance::of(&rstyle);
                        if let Some(rfont) = store.font_for_variant(catalog, &family, weight, st, rsize) {
                            let plain = seg.style.ruby.plain();
                            let mut rglyphs = crate::vector::measure::shape_segment(
                                store, catalog, &family, weight, st, rsize, &rfont, &plain, &rstyle, placement,
                            )?;
                            let rupm = if rfont.units_per_em != 0 { rfont.units_per_em as f64 } else { 1000.0 };
                            let rscale = rstyle.font_size / rupm;
                            for g in rglyphs.iter_mut() {
                                let x_px = plan.ruby_x.get(g.cluster).copied().unwrap_or(0.0) * base_scale;
                                g.x_offset = x_px / rscale.max(1e-12);
                                g.x_advance = 0.0;
                                g.y_offset = 0.0;
                            }
                            let (base_ascent, _) = metrics_provider.metrics(&seg.style);
                            let (r_ascent, r_descent) = metrics_provider.metrics(&rstyle);
                            let ruby_baseline = match column {
                                Some(_) => baseline_y,
                                None => crate::ruby::baseline_above(
                                    baseline_y - seg.style.character.baseline_shift,
                                    base_ascent,
                                    r_descent,
                                ),
                            };
                            segments.push(PlacedSegment {
                    paint: seg.style.paint,
                    placed: Vec::new(),
                                path_shift: 0.0,
                                variations: rstyle.font_variations.clone(),
                                font_key: style_font_key(&rstyle),
                                family: family.clone(),
                                weight,
                                style: st,
                                glyphs: rglyphs,
                                font_size_px: rstyle.font_size,
                                letter_spacing_px: 0.0,
                                start_x: seg_start_x,
                                baseline_y: ruby_baseline,
                                color: css_rgba_or_black(&seg.style.color),
                                source: Default::default(),
                                synthetic_oblique: seg.style.font_style == SYNTHETIC_OBLIQUE,
                                fill: seg.style.fill.clone(),
                                character: crate::character::CharacterStyle::DEFAULT,
                                column: column.map(|c| {
                                    (
                                        c.shifted(crate::ruby::column_offset(seg.style.font_size, rstyle.font_size)),
                                        crate::vertical::GlyphOrient::Upright,
                                        baseline_y,
                                        r_ascent,
                                        r_descent,
                                    )
                                }),
                            });
                        }
                    }
                }
            }

            // Underline / strikethrough rects (walkLines:476-487): per-rect
            // append into per-color buckets, run colors NEVER overridden. (Not along a path.)
            let offset_x = line_start_x + indent.left_offset_px;
            let rects: Vec<DecorationRect> = if on_path.is_some() {
                Vec::new()
            } else {
                let justified = JustifiedMeasurer {
                    inner: &measurer,
                    space_px: justify_px,
                };
                compute_underline_rects(line, baseline_y, &justified)
                    .into_iter()
                    .chain(compute_strikethrough_rects(line, baseline_y, &justified))
                    .collect()
            };
            for rect in rects {
                let color_css = if rect.color.is_empty() {
                    "rgba(0,0,0,1)".to_string()
                } else {
                    rect.color.clone()
                };
                let color = css_rgba_or_black(&color_css);
                let (mut verts, idx) = crate::decorations::build_decoration_geometry(
                    offset_x,
                    std::slice::from_ref(&rect),
                );
                if let Some(c) = &column {
                    map_xy_into_column(&mut verts, c);
                }
                append_indexed_triangles_xy(
                    decoration_primitives.get_painted(color, rect.paint.filter(|_| !rect.color.is_empty())),
                    &verts,
                    &idx,
                    0.0,
                    0.0,
                );
                append_indexed_triangles_xy(&mut all_primitive_tris, &verts, &idx, 0.0, 0.0);
            }

            lines.push(LaidLine {
                top: slot.map(|s| s.top()).unwrap_or(pen_y),
                height: line.line_height_px,
                source: line.segments.iter().map(|s| s.style.source.clone()).collect(),
            });
            pen_y += line.line_height_px;
        }
    }
    if let Some(e) = measurer.take_poison() {
        return Err(e);
    }
    path_sizes.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(VectorLayout {
        box_w,
        box_h,
        first_align: layout.paragraphs.first().map(|p| p.align).unwrap_or("left"),
        total_height_px: area
            .as_ref()
            .map_or(layout.total_height_px, crate::vector::node_layout::area_content_bottom),
        lines,
        segments,
        emojis: placed_emojis,
        bullet_primitives,
        decoration_primitives,
        all_primitive_tris,
        path: on_path.map(|(p, m)| (p.clone(), m)),
        path_sizes,
        path_text_width,
        curve_points: Default::default(),
    })
}

impl VectorLayout {
/// The node's ordered draws from its layout: `spec` gives what a draw adds to the layout (opacity, fill override,
/// gradient, the ranges' paints, stroke, effects, KineticText unit transforms, fills and range), `analytic_pad_scale` the glyph quads'
/// antialiasing pad. The layout fields of `spec` are not read: they are this layout.
#[allow(clippy::too_many_lines)]
pub fn ops(
    &self,
    spec: &TextNodeSpec,
    store: &VectorFontStore,
    catalog: &Catalog,
    analytic_pad_scale: f64,
) -> Result<VectorNodeOps, ShapeError> {
    let VectorLayout {
        first_align,
        segments,
        emojis: placed_emojis,
        bullet_primitives,
        decoration_primitives,
        all_primitive_tris,
        path,
        path_sizes,
        curve_points,
        ..
    } = self;
    let (box_w, box_h, path_text_width) = (self.box_w, self.box_h, self.path_text_width);
    let on_path: Option<(&crate::text_path::TextPathSpec, &crate::text_path::PathMeasure)> =
        path.as_ref().map(|(p, m)| (p, m));
    let kinetic = !spec.unit_transforms.is_empty();
    let effect_passes = resolve_effect_passes(spec);
    // F7.2: where the text starts along the path (start offset and the first paragraph's alignment).
    let path_start = on_path.as_ref().map(|(tp, m)| {
        let align = *first_align;
        crate::text_path::start_distance(m, tp.start_offset, align, path_text_width, &path_sizes)
    });
    // Curvature spacing: each glyph's advance maps to a distance along the path (text_path::PathSpacing).
    let mut path_spacing = on_path
        .as_ref()
        .zip(path_start)
        .map(|((_, m), start)| crate::text_path::PathSpacing::new(m, start, &path_sizes));

    let mut path_ink_shift = 0.0f64;
    // KineticText units on a path: each unit's pivot distance along the path, and the advance → distance maps its
    // pivots and its moved glyphs read (their own, so the walk's ascending queries don't restart from the start).
    let mut path_unit_s: HashMap<u32, f64> = HashMap::new();
    let mut pivot_spacing = path_spacing.clone();
    let mut moved_spacing = path_spacing.clone();

    // ── drawRunsAnalytic (VectorTextRenderer.ts:509-658) ──
    let has_gradient = spec
        .gradient
        .as_ref()
        .map(|g| !g.stops.is_empty())
        .unwrap_or(false);
    // `wantsShaderFill = !hasGradient && !!args.shaderFill` (drawRuns:355-356);
    // the engine only sets spec.shader_fill for ELIGIBLE ids (the production
    // null-pipeline fallback → per-run solid colors).
    let wants_shader_fill = !has_gradient && spec.shader_fill;
    let bucket_override: Option<[f32; 4]> = if has_gradient || wants_shader_fill {
        Some([1.0, 1.0, 1.0, 1.0]) // WHITE_RGBA — collapses to one bucket
    } else {
        spec.fill_override
            .as_deref()
            .and_then(css_color_to_normalized_rgba)
            .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32])
    };

    let mut buckets: Buckets<VectorAnalyticRecord> = Buckets::new();
    // Runs with their own gradient (`--nf-text-fill`), one op per distinct
    // gradient, in first-use order. Node-level fills override them.
    let mut fill_buckets: Vec<(crate::ops::GradientSpec, Vec<VectorAnalyticRecord>)> = Vec::new();
    // Effect passes re-emit every placed instance with a pass-specific pen
    // offset and quad padding; extrude passes sweep glyph outlines instead
    // (drawRunsAnalytic:664-669 — each collected only when active).
    let mut effect_glyphs: Vec<PlacedGlyphInstance> = Vec::new();
    let collect_effects = effect_passes
        .iter()
        .any(|p| p.kind != EffectPassKind::Extrude);
    let mut extrude_glyphs: Vec<crate::vector::extrude::ExtrudePlacedGlyph> = Vec::new();
    let collect_extrude = effect_passes
        .iter()
        .any(|p| p.kind == EffectPassKind::Extrude);

    // KineticText units: a word's (a line's) glyphs scale and turn about the centre of the unit's ink at rest, as one
    // piece, not each about its own. The unit's box, from every glyph of it in every segment; keyed by the unit's
    // first code point (`spec.unit_groups`). Without groups (per-character units) nothing changes.
    // On a path the pivot is the same box, read along the path: the unit's ink centre in the text's advance maps to a
    // distance along the path (`path_unit_s`, per unit), and its glyphs gather toward that distance (see below).
    let unit_pivots = if kinetic && !spec.unit_groups.is_empty() {
        unit_pivots(spec, segments)
    } else {
        HashMap::new()
    };

    for seg in segments {
        // AA pad (drawRunsAnalytic:533-539).
        let pad_em = (ANALYTIC_AA_PAD_DEVICE_PX
            / (seg.font_size_px.max(1e-3) * analytic_pad_scale.max(1e-3)))
        .min(1.0);
        // The paint the draw supplies for this range, when nothing at node level fills the text.
        let paint = seg.paint.filter(|p| bucket_override.is_none() && spec.run_colors.get(*p as usize).is_some());
        let color = bucket_override
            .unwrap_or_else(|| paint.and_then(|p| spec.run_colors.get(p as usize)).copied().unwrap_or(seg.color));

        // placeGlyphs (kerning + letter spacing) over the shaped glyphs; the
        // curve ensure happens in visit order (store append order matches the
        // browser's). Kerning/outline errors both unwind to the node catch.
        let instance = crate::vector::measure::Instance {
            size_px: Some(seg.font_size_px),
            variations: &seg.variations,
        };
        let font = store
            .font_for_variant(catalog, &seg.family, seg.weight, seg.style, instance)
            .expect("font resolved in walk");
        let run_fill = seg
            .fill
            .as_ref()
            .filter(|g| bucket_override.is_none() && !g.stops.is_empty());
        let records = match run_fill {
            Some(g) => {
                let i = match fill_buckets.iter().position(|(f, _)| f == g) {
                    Some(i) => i,
                    None => {
                        fill_buckets.push((g.clone(), Vec::new()));
                        fill_buckets.len() - 1
                    }
                };
                &mut fill_buckets[i].1
            }
            // A range whose paint the draw supplies draws where it comes (`Buckets::get_painted`): the ops follow the
            // text's ranges in text order (A, B, A draws A, B, A, where buckets by colour drew A, A, B), and a draw
            // with other paints has the same ops.
            None => buckets.get_painted(color, paint),
        };
        // Vertical text, a combined piece: each glyph's horizontal offset in the piece and the piece's width, px.
        let (combined, combined_width) = match seg.column {
            Some((_, crate::vertical::GlyphOrient::Combined, ..)) => {
                let upm = if font.units_per_em != 0 { font.units_per_em as f64 } else { 1000.0 };
                let scale = seg.font_size_px / upm * seg.character.horizontal_scale;
                let mut offsets = Vec::with_capacity(seg.glyphs.len());
                let mut acc = 0.0;
                for g in &seg.glyphs {
                    offsets.push(acc);
                    acc += font.advance_width(g.glyph) * scale;
                }
                (offsets, acc)
            }
            _ => (Vec::new(), 0.0),
        };
        // Each glyph's pen and curves are the layout's ([`VectorLayout::outline`]). Q-122: GPOS y-offsets (HB mark
        // positioning on the hybrid arm) ride each glyph's baseline, y-up px from `place_glyphs`, folded into the
        // record's y-down baseline below. The otjs lanes always emit 0, so `baseline_y - 0.0` keeps
        // `--text-engine vector` bit-exact.
        'glyphs: for &PlacedGlyph { gid, pen_x, y_offset_px, glyph_idx, first_curve, curve_count, bounds } in &seg.placed {
            // E8.22: a baseline shift raises (positive) or lowers the glyphs; the line keeps its baseline.
            let glyph_baseline_y = seg.baseline_y - y_offset_px - seg.character.baseline_shift;
            // KineticText: map glyph → its source code point (shaper
            // cluster, ligature → first code point) → unitTransforms[index].
            // Identity for static text → byte-exact records + shader fast-path.
            let code_point = if kinetic {
                let cluster = seg.glyphs.get(glyph_idx).map_or(glyph_idx, |g| g.cluster);
                seg.source.get(cluster)
            } else { None };
            let xform = code_point.and_then(|cp| spec.unit_transforms.get(cp)).copied().unwrap_or(GlyphXform::IDENTITY);
            if spec.unit_range.as_ref().is_some_and(|range|
                code_point.is_none_or(|cp| !range.contains(&cp))) {
                continue 'glyphs;
            }
            if curve_count == 0 {
                continue; // outline-less glyph (space) — no instance
            }
            let em_box = [
                (bounds[0] as f64 - pad_em) as f32,
                (bounds[1] as f64 - pad_em) as f32,
                (bounds[2] as f64 + pad_em) as f32,
                (bounds[3] as f64 + pad_em) as f32,
            ];
            // KineticText units: the transform turns about the unit's centre W, not the glyph's C (the shader's pivot):
            // L(p − W) + W + o = L(p − C) + C + o + (L − I)(C − W).
            // On a path the unit transform is applied as the glyph is placed on the path instead (`path_unit`).
            let kin = xform;
            let unit = code_point
                .map(|cp| unit_group(spec, cp))
                .and_then(|g| unit_pivots.get(&g).map(|&w| (g, w)))
                .filter(|_| !kin.is_identity());
            let path_unit = unit.filter(|_| on_path.is_some());
            let xform = match unit.filter(|_| on_path.is_none()).map(|(_, w)| w) {
                Some((wx, wy)) => {
                    let size = seg.font_size_px;
                    let cx = pen_x + (em_box[0] as f64 + em_box[2] as f64) * 0.5 * size;
                    let cy = glyph_baseline_y + (em_box[1] as f64 + em_box[3] as f64) * 0.5 * size;
                    let (dx, dy) = (cx - wx, cy - wy);
                    let [a, c, b, d] = xform.lin.map(f64::from);
                    GlyphXform {
                        off: [
                            (xform.off[0] as f64 + (a - 1.0) * dx + c * dy) as f32,
                            (xform.off[1] as f64 + b * dx + (d - 1.0) * dy) as f32,
                        ],
                        ..xform
                    }
                }
                _ => xform,
            };
            // E8.22: the run's scale and rotation shape the outline first; a KineticText transform acts on top.
            let shape = |xform: GlyphXform| {
                let xform = if seg.character.transforms_glyphs() {
                    xform.after(seg.character.glyph_xform(em_box, seg.font_size_px as f32))
                } else {
                    xform
                };
                if seg.synthetic_oblique {
                    xform.with_synthetic_oblique(em_box, seg.font_size_px as f32)
                } else {
                    xform
                }
            };
            let xform = shape(xform);
            // F7.2: move the glyph's advance centre onto the path and turn it to the tangent; a glyph past the end
            // of an open path is not drawn (Illustrator's overflow).
            let xform = 'path: {
                match (&on_path, path_start, path_spacing.as_mut()) {
                (Some((_, measure)), Some(_start), Some(spacing)) => {
                    let size = seg.font_size_px;
                    let centre_x = pen_x
                        + (bounds[0] as f64 + bounds[2] as f64) * 0.5 * size * seg.character.horizontal_scale;
                    let mut along = spacing.distance(centre_x) + path_ink_shift;
                    // Check ink edges, not just the centre: a last glyph straddling
                    // the seam would collide with the first even without a second lap.
                    let half_width = (bounds[2] - bounds[0]) as f64 * 0.5 * size * seg.character.horizontal_scale.abs();
                    // Curvature spacing can pull the first ink edge behind the
                    // start bracket. Move the run forward enough to keep that
                    // first glyph, and account for the shift at the end too.
                    if measure.is_closed() && along < half_width {
                        path_ink_shift += half_width - along;
                        along = half_width;
                    }
                    if !measure.fits_closed_glyph(along, half_width) {
                        continue 'glyphs;
                    }
                    let Some((point, tangent)) = measure.at(along) else {
                        continue 'glyphs;
                    };
                    let c = (
                        (em_box[0] as f64 + em_box[2] as f64) * 0.5 * size,
                        (em_box[1] as f64 + em_box[3] as f64) * 0.5 * size,
                    );
                    let Some((group, (wx, wy))) = path_unit else {
                        let (lin, off) = crate::text_path::place_on_path(
                            (pen_x, glyph_baseline_y),
                            c,
                            xform.lin,
                            xform.off,
                            (centre_x, glyph_baseline_y),
                            point,
                            tangent,
                            seg.path_shift,
                        );
                        break 'path GlyphXform { lin, off, ..xform };
                    };
                    // KineticText unit on a path (AE's Anchor Point Grouping by word or line): the unit transform
                    // L (= R(θ)·S) and move o act on the unit as one piece about its pivot W. S gathers the glyph's
                    // centre toward W in the text's frame — its advance part along the path (the glyph is re-placed
                    // at the distance of its moved advance centre, turned to the tangent there), its cross part
                    // along the normal — and shapes the glyph about its own centre; θ then turns the placed glyph
                    // about W's place on the path, and o moves it in W's tangent frame. On a straight path this is
                    // the straight text's L(p − W) + W + o.
                    let [a, cc, b, d] = kin.lin.map(f64::from);
                    let theta = (b - cc).atan2(a + d);
                    let (ct, st) = (theta.cos(), theta.sin());
                    let s_lin = [ct * a + st * b, ct * cc + st * d, ct * b - st * a, ct * d - st * cc];
                    let (dx, dy) = (
                        pen_x + c.0 - wx,
                        glyph_baseline_y + c.1 - wy,
                    );
                    let (sdx, sdy) = (s_lin[0] * dx + s_lin[1] * dy, s_lin[2] * dx + s_lin[3] * dy);
                    let local = GlyphXform {
                        lin: s_lin.map(|v| v as f32),
                        off: [0.0, (sdy - dy) as f32],
                        opacity: kin.opacity,
                    }
                    .after(shape(GlyphXform::IDENTITY));
                    let Some(moved) = moved_spacing.as_mut() else {
                        continue 'glyphs;
                    };
                    let along = moved.distance(centre_x + sdx - dx) + path_ink_shift;
                    let Some((point, tangent)) = measure.at(along) else {
                        continue 'glyphs;
                    };
                    let (lin, off) = crate::text_path::place_on_path(
                        (pen_x, glyph_baseline_y),
                        c,
                        local.lin,
                        local.off,
                        (centre_x, glyph_baseline_y),
                        point,
                        tangent,
                        seg.path_shift,
                    );
                    let pivot_s = match path_unit_s.get(&group) {
                        Some(&s) => s,
                        None => {
                            let s = pivot_spacing.as_mut().map_or(along, |p| p.distance(wx)) + path_ink_shift;
                            path_unit_s.insert(group, s);
                            s
                        }
                    };
                    let ((px, py), (pc, ps)) = measure.at(pivot_s).unwrap_or((point, tangent));
                    let normal = seg.path_shift + wy - glyph_baseline_y;
                    let pivot = (px - ps * normal, py + pc * normal);
                    let [l0, l1, l2, l3] = lin.map(f64::from);
                    let (lin, centre) = if theta == 0.0 {
                        (lin, (pen_x + c.0 + off[0] as f64, glyph_baseline_y + c.1 + off[1] as f64))
                    } else {
                        let (vx, vy) = (
                            pen_x + c.0 + off[0] as f64 - pivot.0,
                            glyph_baseline_y + c.1 + off[1] as f64 - pivot.1,
                        );
                        (
                            [
                                (ct * l0 - st * l2) as f32,
                                (ct * l1 - st * l3) as f32,
                                (st * l0 + ct * l2) as f32,
                                (st * l1 + ct * l3) as f32,
                            ],
                            (pivot.0 + ct * vx - st * vy, pivot.1 + st * vx + ct * vy),
                        )
                    };
                    let (ox, oy) = (kin.off[0] as f64, kin.off[1] as f64);
                    let off = [
                        (centre.0 + pc * ox - ps * oy - pen_x - c.0) as f32,
                        (centre.1 + ps * ox + pc * oy - glyph_baseline_y - c.1) as f32,
                    ];
                    GlyphXform { lin, off, ..xform }
                }
                _ => xform,
                }
            };
            // Vertical text: the glyph turned into its line's column, after its own transforms.
            let xform = match seg.column {
                Some((frame, orient, line_baseline_y, ascent_px, descent_px)) if on_path.is_none() => {
                    let size = seg.font_size_px;
                    let upm = if font.units_per_em != 0 { font.units_per_em as f64 } else { 1000.0 };
                    let (combined_offset_px, combined_width_px) = if orient == crate::vertical::GlyphOrient::Combined {
                        (
                            combined.get(glyph_idx).copied().unwrap_or(0.0),
                            combined.last().map_or(0.0, |_| combined_width),
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    let place = crate::vertical::GlyphPlace {
                        orient,
                        pen_x,
                        glyph_baseline_y,
                        line_baseline_y,
                        font_size_px: size,
                        units_per_em: upm,
                        h_advance: font.advance_width(gid) * seg.character.horizontal_scale,
                        v_origin: font
                            .vertical_metrics()
                            .map_or(ascent_px * upm / size.max(1e-9), |m| m.origin(gid)),
                        ascent_px,
                        descent_px,
                        combined_offset_px,
                        combined_width_px,
                    };
                    let (lin, p1) = crate::vertical::place(&frame, &place);
                    let c = (
                        pen_x + (em_box[0] as f64 + em_box[2] as f64) * 0.5 * size,
                        glyph_baseline_y + (em_box[1] as f64 + em_box[3] as f64) * 0.5 * size,
                    );
                    let (l, off) = crate::vertical::compose(
                        xform.lin.map(f64::from),
                        (xform.off[0] as f64, xform.off[1] as f64),
                        c,
                        (pen_x, glyph_baseline_y),
                        lin,
                        p1,
                    );
                    GlyphXform {
                        lin: l.map(|v| v as f32),
                        off: [off.0 as f32, off.1 as f32],
                        ..xform
                    }
                }
                _ => xform,
            };
            // KineticText fill: the fill record alone takes the code point's fill amount (the effect copies below
            // keep the glyph's opacity, so a stroke stays).
            let fill = code_point
                .and_then(|cp| spec.unit_fill.get(cp))
                .map_or(1.0, |f| if f.is_finite() { f.clamp(0.0, 1.0) } else { 1.0 });
            records.push(VectorAnalyticRecord {
                pen_x: pen_x as f32,
                baseline_y: glyph_baseline_y as f32,
                font_size_px: seg.font_size_px as f32,
                em_box,
                first_curve,
                curve_count,
                glyph_lin: xform.lin,
                glyph_off: xform.off,
                glyph_opacity: xform.opacity * fill,
            });
            if collect_effects {
                effect_glyphs.push(PlacedGlyphInstance {
                    pen_x,
                    baseline_y: glyph_baseline_y,
                    font_size_px: seg.font_size_px,
                    base_pad_em: pad_em,
                    em_bounds: bounds,
                    first_curve,
                    curve_count,
                    xform,
                });
            }
            if collect_extrude {
                extrude_glyphs.push(crate::vector::extrude::ExtrudePlacedGlyph {
                    font: font.clone(),
                    gid,
                    pen_x,
                    baseline_y: glyph_baseline_y,
                    font_size_px: seg.font_size_px,
                    em_bounds: bounds,
                    xform,
                });
            }
        }
    }

    let total_instances: usize = buckets.entries.iter().map(|(_, _, r)| r.len()).sum::<usize>()
        + fill_buckets.iter().map(|(_, r)| r.len()).sum::<usize>();
    let has_primitives = !all_primitive_tris.is_empty();
    let mut out = VectorNodeOps::default();

    // Emoji cells in node-local px; along a path, the cell turns onto the tangent at its
    // advance centre like a glyph does (`text_path::place_on_path`). Each effect pass
    // (drop shadow, glow, echo, stroke; extrude sweeps outlines only) re-emits every
    // cell as a tinted silhouette copy first, back to front, so the emoji carry the
    // text's effects; the emoji themselves come last, over them.
    struct EmojiFrame {
        cluster: String,
        font_size_px: f64,
        /// The cell rect, before the path turn: x0, y0, x1, y1.
        rect: [f64; 4],
        /// Along a path: (target, anchor, cos, sin) — q ↦ target + R(q − anchor).
        turn: Option<((f64, f64), (f64, f64), f64, f64)>,
        /// KineticText: its unit's transform, about `pivot` (the unit's centre, or the cell's), before the path turn.
        xform: GlyphXform,
        pivot: (f64, f64),
    }
    let mut frames: Vec<EmojiFrame> = Vec::new();
    for e in placed_emojis {
        // Its code point addresses a unit only in KineticText.
        let code_point = e.code_point.filter(|_| kinetic);
        if spec.unit_range.as_ref().is_some_and(|range|
            code_point.is_none_or(|cp| !range.contains(&cp))) {
            continue;
        }
        let fs = e.font_size_px;
        let y0 = e.baseline_y - EMOJI_CELL_ASCENT_EM * fs;
        let rect = match &e.column {
            // Vertical text: upright on the column's centre line, its em down the column.
            Some(c) => {
                let x0 = c.centre_x() - 0.5 * EMOJI_CELL_WIDTH_EM * fs;
                let top = e.x + 0.5 * (fs - EMOJI_CELL_HEIGHT_EM * fs);
                [x0, top, x0 + EMOJI_CELL_WIDTH_EM * fs, top + EMOJI_CELL_HEIGHT_EM * fs]
            }
            None => [
                e.x,
                y0,
                e.x + EMOJI_CELL_WIDTH_EM * fs,
                y0 + EMOJI_CELL_HEIGHT_EM * fs,
            ],
        };
        let turn = match (&on_path, path_start, path_spacing.as_mut()) {
            (Some((_, measure)), Some(_start), Some(spacing)) => {
                let centre_x = e.x + 0.5 * EMOJI_CELL_WIDTH_EM * fs;
                let mut along = spacing.distance(centre_x) + path_ink_shift;
                let half_width = 0.5 * EMOJI_CELL_WIDTH_EM * fs;
                if measure.is_closed() && along < half_width {
                    path_ink_shift += half_width - along;
                    along = half_width;
                }
                if !measure.fits_closed_glyph(along, 0.5 * EMOJI_CELL_WIDTH_EM * fs) {
                    continue;
                }
                let Some((point, (cos, sin))) = measure.at(along) else {
                    continue;
                };
                let target = (point.0 - sin * e.path_shift, point.1 + cos * e.path_shift);
                Some((target, (centre_x, e.baseline_y), cos, sin))
            }
            _ => None,
        };
        let xform = code_point
            .and_then(|cp| spec.unit_transforms.get(cp))
            .copied()
            .unwrap_or(GlyphXform::IDENTITY);
        let pivot = code_point
            .and_then(|cp| unit_pivots.get(&unit_group(spec, cp)))
            .copied()
            .unwrap_or(((rect[0] + rect[2]) * 0.5, (rect[1] + rect[3]) * 0.5));
        frames.push(EmojiFrame {
            cluster: e.cluster.clone(),
            font_size_px: fs,
            rect,
            turn,
            xform,
            pivot,
        });
    }
    // One copy's corners: the rect grown by `pad` px, turned onto the path, scaled about
    // `origin` by `scale` (echo), then moved by `offset`.
    let corners_of = |f: &EmojiFrame, pad: f64, scale: f64, origin: (f64, f64), offset: (f64, f64)| {
        let [x0, y0, x1, y1] = f.rect;
        let local = [(x0 - pad, y0 - pad), (x1 + pad, y0 - pad), (x1 + pad, y1 + pad), (x0 - pad, y1 + pad)];
        let mut corners = [0f32; 8];
        for (i, (qx, qy)) in local.into_iter().enumerate() {
            // Its unit's transform, as a glyph's: L(q − W) + W + o.
            let (qx, qy) = if f.xform.is_identity() {
                (qx, qy)
            } else {
                let [a, c, b, d] = f.xform.lin.map(f64::from);
                let (vx, vy) = (qx - f.pivot.0, qy - f.pivot.1);
                (
                    f.pivot.0 + a * vx + c * vy + f.xform.off[0] as f64,
                    f.pivot.1 + b * vx + d * vy + f.xform.off[1] as f64,
                )
            };
            let (px, py) = match f.turn {
                Some((target, anchor, cos, sin)) => {
                    let v = (qx - anchor.0, qy - anchor.1);
                    (target.0 + cos * v.0 - sin * v.1, target.1 + sin * v.0 + cos * v.1)
                }
                None => (qx, qy),
            };
            let (px, py) = (origin.0 + (px - origin.0) * scale, origin.1 + (py - origin.1) * scale);
            corners[i * 2] = (px + offset.0) as f32;
            corners[i * 2 + 1] = (py + offset.1) as f32;
        }
        corners
    };
    let box_centre = (box_w / 2.0, box_h / 2.0);
    for pass in effect_passes.iter().filter(|p| p.extrude.is_none()) {
        let gain = pass
            .opacity_override
            .unwrap_or(spec.opacity * pass.opacity_factor.unwrap_or(1.0)) as f32;
        for f in &frames {
            // The copy's own size (an echo copy shrinks), and its effect in that size's texels.
            let size = (f.font_size_px * pass.scale).max(1e-3);
            let texels_per_px = EMOJI_EFFECT_TEXELS_PER_EM / size;
            let quantise = |px: f64| ((px * texels_per_px * 2.0).round() / 2.0) as f32;
            let blur_texels = quantise(pass.blur_px);
            let stroke_texels = quantise(pass.stroke_px);
            let reach = (blur_texels + stroke_texels) as f64 + 1.0;
            let pad_texels = reach.min(EFFECT_PAD_MAX_EM * EMOJI_EFFECT_TEXELS_PER_EM).ceil() as u32;
            let pad_px = pad_texels as f64 / texels_per_px / pass.scale.max(1e-3);
            out.emojis.push(VectorEmoji {
                cluster: f.cluster.clone(),
                corners: corners_of(f, pad_px, pass.scale, box_centre, (pass.pen_offset_x, pass.pen_offset_y)),
                opacity: 1.0,
                effect: Some(EmojiEffect {
                    color: pass.color,
                    gain: gain * f.xform.opacity,
                    blur_texels,
                    stroke_texels,
                    pad_texels,
                    additive: pass.additive,
                }),
            });
        }
    }
    for f in &frames {
        out.emojis.push(VectorEmoji {
            cluster: f.cluster.clone(),
            corners: corners_of(f, 0.0, 1.0, box_centre, (0.0, 0.0)),
            opacity: spec.opacity as f32 * f.xform.opacity,
            effect: None,
        });
    }

    if total_instances == 0 && !has_primitives {
        // "nothing visible, but handled (don't fall back)" (drawRunsAnalytic:582)
        return Ok(out);
    }

    let opacity = spec.opacity as f32;

    // Gradient uniforms (packTextGradientUniformFloats, vectorTextShaders.ts:54-81).
    // textBounds = {x:0, y:0, width:boxW, height:boxH}
    // (resolveTextNodeFill, WebGPUTextSystem.ts:573/593).
    if has_gradient {
        let g = spec.gradient.as_ref().unwrap();
        out.gradient = Some(crate::run_fill::gradient_uniforms(g, box_w, box_h));
    }

    // 1. Effects behind the fill; retain the legacy stroke to paint over it.
    //    (drawRunsAnalytic:745-823): the effect list (drop shadows, glow,
    //    echo, extrude, in order) → legacy stroke. Echo copies scale about the text box center
    //    (drawRunsAnalytic:750).
    let scale_origin = (box_w / 2.0, box_h / 2.0);
    let mut stroke_ops = Vec::new();
    for pass in &effect_passes {
        let ops = if pass.kind == EffectPassKind::Stroke { &mut stroke_ops } else { &mut out.ops };
        // `writeDrawUniforms` u[21] (VectorTextRenderer.ts:1320):
        // override REPLACES node opacity; factor MULTIPLIES it.
        let pass_opacity =
            pass.opacity_override
                .unwrap_or(spec.opacity * pass.opacity_factor.unwrap_or(1.0)) as f32;
        // Extrude draws swept wall geometry through the solid-primitive
        // pipeline; its crisp front face is the node's own fill, drawn on
        // top below (drawRunsAnalytic:752-761).
        if let Some((depth_px, angle_deg)) = pass.extrude {
            if !extrude_glyphs.is_empty() {
                let wall_tris = crate::vector::extrude::build_extrude_wall_triangles(
                    &extrude_glyphs,
                    depth_px,
                    angle_deg,
                )?;
                if wall_tris.len() >= 6 {
                    ops.push(VectorOp {
                        pipeline: VectorOpPipeline::Tess,
                        color: pass.color,
                        opacity: pass_opacity,
                        blur_px: 0.0,
                        stroke_px: 0.0,
                        additive: false,
                        tris: wall_tris,
                        records: Vec::new(),
                        gradient: None,
                        paint: None,
                    });
                }
            }
            continue;
        }
        // The shadow silhouette includes the solid primitives; structured
        // effects (glow/echo) apply to glyph outlines only and the stroke
        // pass skips primitives (drawRunsAnalytic:762-777).
        if pass.kind == EffectPassKind::Shadow && !all_primitive_tris.is_empty() {
            ops.push(VectorOp {
                pipeline: VectorOpPipeline::Tess,
                color: pass.color,
                opacity: pass_opacity,
                blur_px: 0.0,
                stroke_px: 0.0,
                additive: false,
                tris: translate_triangles(
                    &all_primitive_tris,
                    pass.pen_offset_x as f32,
                    pass.pen_offset_y as f32,
                ),
                records: Vec::new(),
                gradient: None,
                paint: None,
            });
        }
        if !effect_glyphs.is_empty() && !curve_points.is_empty() {
            // buildEffectInstanceRecords (vectorTextEffects.ts:280-311):
            // pens shifted by the pass offset, quads padded to the effect's
            // reach; a pass `scale` (echo) shrinks each instance's font size
            // and pulls its pen toward the text box center.
            let scale = pass.scale;
            let (ox, oy) = scale_origin;
            let records: Vec<VectorAnalyticRecord> = effect_glyphs
                .iter()
                .map(|g| {
                    let font_size_px = if scale == 1.0 {
                        g.font_size_px
                    } else {
                        (g.font_size_px * scale).max(1e-3)
                    };
                    let pen_x = if scale == 1.0 {
                        g.pen_x
                    } else {
                        ox + (g.pen_x - ox) * scale
                    };
                    let baseline_y = if scale == 1.0 {
                        g.baseline_y
                    } else {
                        oy + (g.baseline_y - oy) * scale
                    };
                    let extra_pad_em = pass.extra_pad_local_px / font_size_px.max(1e-3);
                    // Re-derive the base AA pad for the scaled size
                    // (vectorTextEffects.ts:294-296).
                    let base_pad_em = if scale == 1.0 {
                        g.base_pad_em
                    } else {
                        (g.base_pad_em / scale.max(1e-3)).min(1.0)
                    };
                    let pad_em = (base_pad_em + extra_pad_em).min(EFFECT_PAD_MAX_EM);
                    VectorAnalyticRecord {
                        pen_x: (pen_x + pass.pen_offset_x) as f32,
                        baseline_y: (baseline_y + pass.pen_offset_y) as f32,
                        font_size_px: font_size_px as f32,
                        em_box: [
                            (g.em_bounds[0] as f64 - pad_em) as f32,
                            (g.em_bounds[1] as f64 - pad_em) as f32,
                            (g.em_bounds[2] as f64 + pad_em) as f32,
                            (g.em_bounds[3] as f64 + pad_em) as f32,
                        ],
                        first_curve: g.first_curve,
                        curve_count: g.curve_count,
                        // Effects parity: the pass re-emits each glyph
                        // carrying its SAME per-glyph transform + opacity, so the
                        // glow/echo/shadow/stroke copy tracks the glyph (the shader
                        // recomputes the pivot from this record's pass-shifted
                        // pen/emBox). A glyph at opacity 0 → 0 on every pass.
                        // Passed verbatim by design: `glyph_lin` (scale/rotation)
                        // composes multiplicatively with the pass `scale` in the
                        // shader, but the absolute px `glyph_off` is intentionally
                        // NOT pre-scaled — a scaled echo copy still tracks the
                        // glyph's full screen-space kinetic displacement rather
                        // than a shrunken fraction of it.
                        glyph_lin: g.xform.lin,
                        glyph_off: g.xform.off,
                        glyph_opacity: g.xform.opacity,
                    }
                })
                .collect();
            ops.push(VectorOp {
                pipeline: VectorOpPipeline::Analytic,
                color: pass.color,
                opacity: pass_opacity,
                blur_px: pass.blur_px as f32,
                stroke_px: pass.stroke_px as f32,
                additive: pass.additive,
                tris: Vec::new(),
                records,
                gradient: None,
                paint: None,
            });
        }
    }

    // 2. Bullets behind the glyph fill (:629-632).
    let painted = |color: &[f32; 4], paint: &Option<u32>| -> [f32; 4] {
        paint.and_then(|p| spec.run_colors.get(p as usize)).copied().unwrap_or(*color)
    };
    // The paint an op's colour is: one the draw supplies.
    let supplied = |paint: &Option<u32>| paint.filter(|p| spec.run_colors.get(*p as usize).is_some());
    for ((_, color, tris), paint) in bullet_primitives.entries.iter().zip(&bullet_primitives.paints) {
        let color = &painted(color, paint);
        if tris.len() < 6 {
            continue;
        }
        out.ops.push(VectorOp {
            pipeline: VectorOpPipeline::Tess,
            color: *color,
            opacity,
            blur_px: 0.0,
            stroke_px: 0.0,
            additive: false,
            tris: tris.clone(),
            records: Vec::new(),
            gradient: None,
            paint: supplied(paint),
        });
    }

    // 3. Glyph fill buckets (:634-650).
    if !curve_points.is_empty() && total_instances > 0 {
        let fill_pipeline = if wants_shader_fill {
            VectorOpPipeline::AnalyticShaderFill
        } else if has_gradient {
            VectorOpPipeline::AnalyticGradient
        } else {
            VectorOpPipeline::Analytic
        };
        for ((_, color, records), paint) in buckets.entries.iter().zip(&buckets.paints) {
            if records.is_empty() {
                continue;
            }
            out.ops.push(VectorOp {
                pipeline: fill_pipeline,
                // Gradient draws write WHITE draw-uniform color (the shader's
                // empty-gradient fallback, drawAnalyticGradientBucket:725).
                color: *color,
                opacity,
                blur_px: 0.0,
                stroke_px: 0.0,
                additive: false,
                tris: Vec::new(),
                records: records.clone(),
                gradient: None,
                paint: *paint,
            });
        }
        // Text ranges with their own gradient, over the same box as a node gradient.
        for (g, records) in &fill_buckets {
            if records.is_empty() {
                continue;
            }
            out.ops.push(VectorOp {
                pipeline: VectorOpPipeline::AnalyticGradient,
                color: [1.0, 1.0, 1.0, 1.0],
                opacity,
                blur_px: 0.0,
                stroke_px: 0.0,
                additive: false,
                tris: Vec::new(),
                records: records.clone(),
                gradient: Some(crate::run_fill::gradient_uniforms(g, box_w, box_h)),
                paint: None,
            });
        }
    }

    out.ops.extend(stroke_ops);

    // 4. Decorations over the glyph fill (:652-655).
    for ((_, color, tris), paint) in decoration_primitives.entries.iter().zip(&decoration_primitives.paints) {
        let color = &painted(color, paint);
        if tris.len() < 6 {
            continue;
        }
        out.ops.push(VectorOp {
            pipeline: VectorOpPipeline::Tess,
            color: *color,
            opacity,
            blur_px: 0.0,
            stroke_px: 0.0,
            additive: false,
            tris: tris.clone(),
            records: Vec::new(),
            gradient: None,
            paint: supplied(paint),
        });
    }

    if total_instances > 0 {
        out.curve_points = curve_points.clone();
    }
    Ok(out)
}
}

#[cfg(test)]
#[cfg_attr(not(vork_private_tests), allow(dead_code, unused_imports))]
mod tests {
    use super::*;
    use crate::test_fixtures::store_and_catalog;
    use crate::vector::measure::{shape, shape_vector};

    fn spec(html: &str) -> TextNodeSpec {
        TextNodeSpec {
            node_id: "t".into(),
            html: html.into(),
            width: Some(1000.0),
            height: Some(400.0),
            opacity: 1.0,
            ..Default::default()
        }
    }

    /// A black drop shadow offset (4, 4), opacity 0.3.
    fn shadow(blur: f64) -> crate::ops::TextEffectSpec {
        crate::ops::TextEffectSpec::DropShadow(crate::ops::TextDropShadow {
            color: "rgba(0,0,0,1)".into(),
            offset_x: 4.0,
            offset_y: 4.0,
            blur,
            opacity: 0.3,
        })
    }

    /// Run the success-path `build_vector_node_ops` call and assert the node
    /// routed to the vector lane, returning its ops. The `message` is the exact
    /// panic literal the inlined `VectorEngineDecision::Vector` destructure used.
    /// Runs A, B, A in two colours, each naming its own paint: one op a run, in text order, each naming its paint; and
    /// the same ops whatever the colours, so a draw with other paints differs in its ops' colours alone.
    #[test]
    fn glyph_ops_follow_the_runs_paints_in_text_order_whatever_their_colours() {
        let (store, catalog) = store_and_catalog();
        let run = |paint: u32, text: &str| {
            format!("<span style=\"font-family: Inter; font-size: 32px; --nf-paint: {paint}\">{text}</span>")
        };
        let mut s = spec(&format!("<p>{}{}{}</p>", run(0, "A"), run(1, "B"), run(2, "C")));
        let layout = accept_vector_layout(&s, &store, &catalog, VectorPlacement::VectorParity, false)
            .expect("the vector lane lays this text out");
        let (red, blue, green) = ([1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.5], [0.0, 1.0, 0.0, 0.5]);
        let mut draw = |colors: [[f32; 4]; 3]| {
            s.run_colors = std::sync::Arc::new(colors.to_vec());
            layout.ops(&s, &store, &catalog, 1.0).expect("the layout draws")
        };
        let first = draw([red, blue, red]);
        assert_eq!(first.ops.iter().map(|op| op.paint).collect::<Vec<_>>(), [Some(0), Some(1), Some(2)]);
        assert_eq!(first.ops.iter().map(|op| op.color).collect::<Vec<_>>(), [red, blue, red]);
        let pens: Vec<f32> = first.ops.iter().map(|op| op.records[0].pen_x).collect();
        assert!(pens[0] < pens[1] && pens[1] < pens[2], "text order: {pens:?}");
        for colors in [[green, green, green], [red, green, blue], [[0.0; 4]; 3]] {
            let mut again = draw(colors);
            assert_eq!(again.ops.iter().map(|op| op.color).collect::<Vec<_>>(), colors);
            for (op, was) in again.ops.iter_mut().zip(&first.ops) {
                op.color = was.color;
            }
            assert_eq!(again, first, "only the ops' colours follow the paints");
        }
        // A range whose paint the draw does not supply draws in its HTML colour, bucketed by that colour.
        s.run_colors = Default::default();
        let unpainted = layout.ops(&s, &store, &catalog, 1.0).expect("the layout draws");
        assert!(unpainted.ops.iter().all(|op| op.paint.is_none()));
        assert_eq!(unpainted.ops.len(), 1, "three runs of the HTML's one colour are one op");
    }

    /// Ranges naming paints 0, 1, 0: three ops in text order, the first and last naming paint 0. The last range
    /// does not join the first one's op and draw under the middle one.
    #[test]
    fn a_repeated_paint_draws_each_range_where_it_comes() {
        let (store, catalog) = store_and_catalog();
        let run = |paint: u32, text: &str| {
            format!("<span style=\"font-family: Inter; font-size: 32px; letter-spacing: -12px; --nf-paint: {paint}\">{text}</span>")
        };
        let mut s = spec(&format!("<p>{}{}{}</p>", run(0, "A"), run(1, "B"), run(0, "C")));
        let layout = accept_vector_layout(&s, &store, &catalog, VectorPlacement::VectorParity, false)
            .expect("the vector lane lays this text out");
        let (red, blue, green): ([f32; 4], [f32; 4], [f32; 4]) = ([1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.5], [0.0, 1.0, 0.0, 0.5]);
        s.run_colors = std::sync::Arc::new(vec![red, blue]);
        let first = layout.ops(&s, &store, &catalog, 1.0).expect("the layout draws");
        assert_eq!(first.ops.iter().map(|op| op.paint).collect::<Vec<_>>(), [Some(0), Some(1), Some(0)]);
        assert_eq!(first.ops.iter().map(|op| op.color).collect::<Vec<_>>(), [red, blue, red]);
        let pens: Vec<f32> = first.ops.iter().map(|op| op.records[0].pen_x).collect();
        assert!(pens[0] < pens[1] && pens[1] < pens[2], "text order: {pens:?}");
        // Both paints in one colour: the same three ops, not one.
        s.run_colors = std::sync::Arc::new(vec![green, green]);
        let mut again = layout.ops(&s, &store, &catalog, 1.0).expect("the layout draws");
        assert!(again.ops.iter().all(|op| op.color == green));
        for (op, was) in again.ops.iter_mut().zip(&first.ops) {
            op.color = was.color;
        }
        assert_eq!(again, first, "only the ops' colours follow the paints");
    }

    /// Two underlines whose runs name one paint but have their own colours in the HTML: a draw that supplies no
    /// colour for that paint draws each in its own colour, and one that supplies it draws both in it.
    #[test]
    fn a_primitive_keeps_its_own_colour_when_the_draw_supplies_none_for_its_paint() {
        let (store, catalog) = store_and_catalog();
        let line = |color: &str, text: &str| {
            format!("<p><span style=\"font-family: Inter; font-size: 32px; color: {color}; --nf-paint: 0; text-decoration: underline\">{text}</span></p>")
        };
        let mut s = spec(&format!("{}{}", line("rgb(255, 0, 0)", "A"), line("rgb(0, 0, 255)", "B")));
        let layout = accept_vector_layout(&s, &store, &catalog, VectorPlacement::VectorParity, false)
            .expect("the vector lane lays this text out");
        let underlines = |ops: &VectorNodeOps| -> Vec<([f32; 4], Option<u32>)> {
            ops.ops.iter().filter(|op| op.pipeline == VectorOpPipeline::Tess).map(|op| (op.color, op.paint)).collect()
        };
        let (red, blue, green): ([f32; 4], [f32; 4], [f32; 4]) = ([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        let unsupplied = layout.ops(&s, &store, &catalog, 1.0).expect("the layout draws");
        assert_eq!(underlines(&unsupplied), [(red, None), (blue, None)]);
        s.run_colors = std::sync::Arc::new(vec![green]);
        let supplied = layout.ops(&s, &store, &catalog, 1.0).expect("the layout draws");
        assert_eq!(underlines(&supplied), [(green, Some(0)), (green, Some(0))]);
    }

    fn expected_vector(
        spec: &TextNodeSpec,
        store: &VectorFontStore,
        catalog: &Catalog,
        placement: VectorPlacement,
        message: &str,
    ) -> crate::vector::VectorNodeOps {
        let decision = build_vector_node_ops(spec, store, catalog, 1.0, placement);
        let VectorEngineDecision::Vector(ops) = decision else {
            panic!("{message}");
        };
        ops
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn vertical_alignment_places_glyphs_in_both_renderers() {
        let (store, catalog) = store_and_catalog();
        for placement in [VectorPlacement::VectorParity, VectorPlacement::Canvas2dParity] {
            for lines in [1, 2] {
                let content = if lines == 1 { "Hi" } else { "Hi<br>Hi" };
                let mut node = spec(&format!(
                    "<p style=\"font-family: Inter; font-size: 20px; line-height: 2\">{content}</p>"
                ));
                node.height = Some(200.0);
                let top = expected_vector(&node, &store, &catalog, placement, "text must render");
                let top_records: Vec<_> = top.ops.iter().flat_map(|op| &op.records).collect();
                assert!(!top_records.is_empty());
                for (alignment, fraction) in [("middle", 0.5), ("bottom", 1.0)] {
                    node.vertical_align = Some(alignment.into());
                    let aligned = expected_vector(&node, &store, &catalog, placement, "text must render");
                    let aligned_records: Vec<_> = aligned.ops.iter().flat_map(|op| &op.records).collect();
                    assert_eq!(aligned_records.len(), top_records.len());
                    let offset = (200.0 - f64::from(lines) * 40.0) * fraction;
                    for (original, actual) in top_records.iter().zip(aligned_records) {
                        assert_eq!(actual.pen_x, original.pen_x);
                        assert!((f64::from(actual.baseline_y - original.baseline_y) - offset).abs() < 1e-4);
                    }
                }
            }
        }
    }

    /// KineticText: a reveal preset forces an underlined node onto the
    /// vector lane (it now carries `unit_transforms`). The underline decoration
    /// MUST still be emitted there — a `Tess` op alongside the analytic glyph op,
    /// a non-degenerate quad below the baseline — independent of per-glyph opacity.
    /// Pins the data contract that the preset port + lane-routing keep decorations
    /// on the kinetic vector lane (prior decoration coverage was only the isolated
    /// `compute_underline_rects` unit tests + an MSDF-lane op test; nothing
    /// exercised the vector lane). NOTE: this asserts the op STREAM, not pixels —
    /// the GPU rasterization of the decoration quad is out of scope for a CPU test.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn kinetic_underline_emits_decoration_on_vector_lane() {
        let (store, catalog) = store_and_catalog();
        let mut s = spec(
            "<p><span style=\"font-family: Inter; font-size: 32px; text-decoration: underline\">AB</span></p>",
        );
        // A reveal mid-flight: glyph A visible, glyph B hidden → the node is kinetic.
        s.unit_transforms = (vec![
            GlyphXform::resolve(0.0, 0.0, 1.0, 0.0, 1.0),
            GlyphXform::resolve(0.0, 0.0, 1.0, 0.0, 0.0),
        ]).into();
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "a kinetic node must route to the vector lane",
        );
        let glyph = ops
            .ops
            .iter()
            .find(|o| o.pipeline == VectorOpPipeline::Analytic)
            .expect("analytic glyph op");
        let deco = ops
            .ops
            .iter()
            .find(|o| o.pipeline == VectorOpPipeline::Tess)
            .expect("underline decoration must be emitted on the kinetic vector lane");
        // The per-glyph reveal reached the glyph records.
        assert_eq!(glyph.records.len(), 2);
        assert_eq!(glyph.records[0].glyph_opacity, 1.0);
        assert_eq!(glyph.records[1].glyph_opacity, 0.0);
        // The decoration is a non-degenerate quad (≥2 triangles), opaque, and uses
        // the node opacity — not accidentally alpha-0 or empty.
        assert!(
            deco.tris.len() >= 12,
            "underline quad tris: {}",
            deco.tris.len()
        );
        assert_eq!(deco.color[3], 1.0);
        assert_eq!(deco.opacity, 1.0);
        // It sits BELOW the glyph baseline (underlineY = baseline + 0.10·fontSize),
        // so it's positioned with the text, not off in dead space.
        let baseline = glyph.records[0].baseline_y;
        let min_y = deco
            .tris
            .iter()
            .skip(1)
            .step_by(2)
            .copied()
            .fold(f32::INFINITY, f32::min);
        assert!(
            min_y > baseline,
            "underline y={min_y} must sit below baseline y={baseline}"
        );
    }

    /// E7.20: auto-width text never wraps at the box edge; the same node as
    /// fixed-width text wraps into several lines.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn auto_width_text_does_not_wrap_at_the_box_edge() {
        let (store, catalog) = store_and_catalog();
        let baselines = |auto_width: bool| {
            let mut s = spec(
                "<p><span style=\"font-family: Inter; font-size: 32px\">Hello wide world</span></p>",
            );
            s.width = Some(60.0);
            s.auto_width = auto_width;
            let ops = expected_vector(&s, &store, &catalog, VectorPlacement::VectorParity, "vector");
            let mut ys: Vec<f32> = ops
                .ops
                .iter()
                .filter(|o| o.pipeline == VectorOpPipeline::Analytic)
                .flat_map(|o| o.records.iter().map(|r| r.baseline_y))
                .collect();
            ys.sort_by(f32::total_cmp);
            ys.dedup();
            ys.len()
        };
        assert!(baselines(false) > 1, "fixed width wraps");
        assert_eq!(baselines(true), 1, "auto width stays on one line");
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn plain_inter_text_routes_vector_and_emits_instances() {
        let (store, catalog) = store_and_catalog();
        let s = spec("<p><span style=\"font-family: Inter; font-size: 32px\">Hello AV</span></p>");
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector decision",
        );
        assert_eq!(ops.ops.len(), 1);
        let op = &ops.ops[0];
        assert_eq!(op.pipeline, VectorOpPipeline::Analytic);
        // "Hello AV" = 8 codepoints, 7 drawn glyphs ('l' has curves; space skipped)
        assert_eq!(op.records.len(), 7);
        assert!(!ops.curve_points.is_empty());
        assert_eq!(ops.curve_points.len() % 6, 0);
        // dedup: 'l' appears twice but its curves are stored once
        let unique_first: std::collections::HashSet<u32> =
            op.records.iter().map(|r| r.first_curve).collect();
        assert_eq!(unique_first.len(), 6); // H e l o A V
                                           // AA pad applied: em box exceeds raw hull
        let r = &op.records[0];
        assert!(r.em_box[2] > r.em_box[0]);
    }

    /// The vector lane reroutes otjs-incompatible fonts through
    /// HarfBuzz instead of declining whole-node. Lato's GSUB liga lives in
    /// type-7 extension lookups (the exact `lookupType: 7` otjs throw that
    /// used to fall back to MSDF) — the node now SERVES vector, and the
    /// drawn glyphs must be HB-shaped (ffi ligated), so a "fix" that merely
    /// swallows the throw into an unshaped cmap walk fails this pin.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn lato_otjs_throw_reroutes_to_harfbuzz_and_serves_vector() {
        let (store, catalog) = store_and_catalog();
        let s = spec("<p><span style=\"font-family: Lato; font-size: 24px\">Office</span></p>");
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector decision (ENG-3341 reroute)",
        );
        // "Office" ligates (ffi) — fewer drawn glyphs than code points.
        assert!(
            ops.ops[0].records.len() < 6,
            "ffi must ligate via HarfBuzz: {} glyphs",
            ops.ops[0].records.len()
        );
        // The otjs walk itself still throws (port parity, Q-105d) — only the
        // engine dispatch changed.
        let font = store
            .font_for_variant(&catalog, "Lato", 400, "normal", None)
            .unwrap();
        assert!(crate::vector::measure::shape(&font, "Office").is_err());
    }

    /// Extension-lookup kerning reaches the VECTOR lane's pens too
    /// (Lato kerns ONLY through type-9 extension lookups, invisible to otjs)
    /// — proof the reroute shapes through HarfBuzz, not a throw-swallowing
    /// cmap walk.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn lato_reroute_kerns_extension_lookups_on_vector_lane() {
        let (store, catalog) = store_and_catalog();
        let s = spec("<p><span style=\"font-family: Lato; font-size: 100px\">AV</span></p>");
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector decision",
        );
        let font = store
            .font_for_variant(&catalog, "Lato", 400, "normal", None)
            .unwrap();
        let scale = 100.0 / font.units_per_em as f64;
        let bare = font.advance_width(font.char_to_glyph_index('A')) * scale;
        let recs = &ops.ops[0].records;
        assert_eq!(recs.len(), 2);
        let pen_delta = (recs[1].pen_x - recs[0].pen_x) as f64;
        assert!(
            pen_delta < bare - 1.0,
            "A→V must kern through the extension lookups: Δ{pen_delta} vs bare {bare}"
        );
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn effects_emit_shadow_then_fill_then_stroke() {
        let (store, catalog) = store_and_catalog();
        let mut s = spec("<p><span style=\"font-family: Inter; font-size: 32px\">Fx</span></p>");
        s.effects = vec![shadow(6.0)];
        s.stroke_color = Some("#ff0000".into());
        s.stroke_width = Some(3.0);
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector",
        );
        // Structured shadow, fill, then legacy stroke (SVG paint order).
        assert_eq!(ops.ops.len(), 3);
        assert_eq!(ops.ops[0].blur_px, 6.0);
        assert_eq!(ops.ops[0].opacity, 0.3);
        // shadow pens offset by 4px
        assert_eq!(
            ops.ops[0].records[0].pen_x,
            ops.ops[1].records[0].pen_x + 4.0
        );
        assert_eq!(ops.ops[2].stroke_px, 3.0);
        assert_eq!(ops.ops[2].opacity, 1.0);
        // stroke pads the quad outward beyond the fill pass
        assert!(ops.ops[2].records[0].em_box[0] < ops.ops[1].records[0].em_box[0]);
        assert_eq!(ops.ops[1].blur_px, 0.0);
    }

    /// Canvas2dParity expected baseline for a single-style spec — the
    /// browser Canvas2D raster pipeline's math (Q-120 + Q-125): per-line
    /// baseline from Chromium-ROUNDED canvas metrics, middle/bottom offset
    /// from the CEILED raster logical height against the full node box.
    /// `lines` is the per-line (line_height_px) list; returns the FIRST
    /// line's baseline.
    fn c2d_expected_first_baseline(
        store: &VectorFontStore,
        catalog: &Catalog,
        family: &str,
        font_px: f64,
        line_height_px: f64,
        n_lines: usize,
        node_h: f64,
        vertical_align: &str,
    ) -> f64 {
        let m = store
            .metrics_for_variant(catalog, family, 400, "normal", None)
            .unwrap();
        let af = (m.ascent / m.units_per_em) * font_px;
        let df = (m.descent.abs() / m.units_per_em) * font_px;
        let (ai, di) = ((af + 0.5).floor(), (df + 0.5).floor());
        let intrinsic = line_height_px * n_lines as f64 + (ai + di - line_height_px).max(0.0);
        let t = intrinsic.ceil();
        let v_offset = match vertical_align {
            "middle" => (node_h - t) / 2.0,
            "bottom" => node_h - t,
            _ => 0.0,
        };
        v_offset + (line_height_px - (ai + di)) / 2.0 + ai
    }

    /// Q-120 (FIX r2) + Q-125: the hybrid arm's `Canvas2dParity` middle
    /// placement is the Canvas2D raster pipeline's — intrinsic height incl.
    /// the last-line content-box overflow (`intrinsicSize.ts:70`), measured
    /// with Chromium-rounded canvas metrics, CEILED to the raster's logical
    /// height (`measureRichTextContentSize`), quad-aligned against the full
    /// node box (`transforms.ts:96-100`). Formula pin against the TTF
    /// metrics the provider itself serves; the pixel truth is the browser
    /// golden + IB53 measurement.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn canvas2d_parity_middle_align_uses_ceiled_raster_height_and_rounded_metrics() {
        let (store, catalog) = store_and_catalog();
        // line-height 1.0 < Inter's 1.21 em content box → overflow.
        let html =
            "<p><span style=\"font-family: Inter; font-size: 100px; line-height: 1\">Hg</span></p>";
        let mut s = spec(html);
        s.vertical_align = Some("middle".into());
        let base = |placement| {
            let ops = expected_vector(&s, &store, &catalog, placement, "expected vector decision");
            ops.ops[0].records[0].baseline_y as f64
        };
        let vector_y = base(VectorPlacement::VectorParity);
        let c2d_y = base(VectorPlacement::Canvas2dParity);
        let expected = c2d_expected_first_baseline(
            &store, &catalog, "Inter", 100.0, 100.0, 1, 400.0, "middle",
        );
        // records carry f32 — pin at 1e-3 px (the oracle gates' tolerance).
        assert!(
            (c2d_y - expected).abs() < 1e-3,
            "c2d baseline {c2d_y} != browser-pipeline formula {expected}"
        );
        // The overflow lift direction is preserved: c2d sits ABOVE the
        // vector engine's bare Σ lineHeightPx placement.
        assert!(vector_y - c2d_y > 0.5, "lift lost: {vector_y} vs {c2d_y}");
    }

    /// Q-125: even for TOP alignment the golden's Canvas2D raster computes
    /// per-line baselines from Chromium's integer-rounded
    /// `fontBoundingBox{Ascent,Descent}` — so `Canvas2dParity` shifts the
    /// baseline by exactly the metric-rounding delta while horizontal
    /// placement stays byte-identical.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn canvas2d_parity_top_align_baseline_uses_rounded_canvas_metrics() {
        let (store, catalog) = store_and_catalog();
        let s = spec("<p><span style=\"font-family: Inter; font-size: 32px\">Hello</span></p>");
        let v = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector decisions",
        );
        let c = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::Canvas2dParity,
            "expected vector decisions",
        );
        let m = store
            .metrics_for_variant(&catalog, "Inter", 400, "normal", None)
            .unwrap();
        let af = (m.ascent / m.units_per_em) * 32.0;
        let df = (m.descent.abs() / m.units_per_em) * 32.0;
        let (ai, di) = ((af + 0.5).floor(), (df + 0.5).floor());
        // The pin only bites when the metrics are actually fractional.
        assert!(
            (ai - af).abs() > 0.05 || (di - df).abs() > 0.05,
            "fixture metrics must be fractional: {af}/{df}"
        );
        assert_eq!(v.ops.len(), 1);
        assert_eq!(c.ops.len(), 1);
        assert_eq!(v.ops[0].records.len(), 5);
        assert_eq!(c.ops[0].records.len(), 5);
        // baselineY = (lineH − (A+D))/2 + A ⇒ Δ = (Ai−Di)/2 − (Af−Df)/2.
        let expected_delta = (ai - di) / 2.0 - (af - df) / 2.0;
        for (rv, rc) in v.ops[0].records.iter().zip(c.ops[0].records.iter()) {
            assert_eq!(rv.pen_x, rc.pen_x, "horizontal placement must not move");
            let delta = rc.baseline_y as f64 - rv.baseline_y as f64;
            assert!(
                (delta - expected_delta).abs() < 1e-3,
                "baseline delta {delta} != rounding delta {expected_delta}"
            );
        }
    }

    /// Q-125: the middle/bottom offset uses the CEILED raster logical
    /// height. Box-text lines are whole pixels now (Figma rounds each line:
    /// 32px × 1.7 = 54.4 → 54), so the line is 54 and the ceil leaves it be;
    /// the baseline still pins the middle offset (no overflow, rounded
    /// content box fits the line).
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn canvas2d_parity_middle_offset_ceils_raster_height() {
        let (store, catalog) = store_and_catalog();
        let html = "<p><span style=\"font-family: Inter; font-size: 32px; line-height: 1.7\">Hello</span></p>";
        let mut s = spec(html);
        s.vertical_align = Some("middle".into());
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::Canvas2dParity,
            "expected vector decision",
        );
        let line_h: f64 = crate::types::round_line_height_px(32.0 * 1.7);
        assert_eq!(line_h, 54.0, "the line is rounded to whole pixels");
        let expected = c2d_expected_first_baseline(
            &store, &catalog, "Inter", 32.0, line_h, 1, 400.0, "middle",
        );
        let got = ops.ops[0].records[0].baseline_y as f64;
        assert!(
            (got - expected).abs() < 1e-3,
            "ceil pin: baseline {got} != {expected}"
        );
    }

    /// Multi-paragraph: the overflow term accumulates PER PARAGRAPH
    /// (`intrinsicSize.ts` adds it inside the paragraph loop), not just for
    /// the final line of the node — pinned through the full browser-pipeline
    /// formula (rounded metrics + per-paragraph overflow + ceil).
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn canvas2d_parity_overflow_accumulates_per_paragraph() {
        let (store, catalog) = store_and_catalog();
        let para =
            "<p><span style=\"font-family: Inter; font-size: 100px; line-height: 1\">Hg</span></p>";
        let mut one = spec(para);
        one.vertical_align = Some("bottom".into());
        let mut two = spec(&format!("{para}{para}"));
        two.vertical_align = Some("bottom".into());
        let last_baseline = |s: &TextNodeSpec, placement| {
            let ops = expected_vector(s, &store, &catalog, placement, "expected vector decision");
            ops.ops[0].records.last().unwrap().baseline_y as f64
        };
        let m = store
            .metrics_for_variant(&catalog, "Inter", 400, "normal", None)
            .unwrap();
        let af = (m.ascent / m.units_per_em) * 100.0;
        let df = (m.descent.abs() / m.units_per_em) * 100.0;
        let (ai, di) = ((af + 0.5).floor(), (df + 0.5).floor());
        let overflow = (ai + di - 100.0).max(0.0);
        assert!(overflow > 0.5, "fixture font must overflow: {overflow}");
        // bottom v_offset = H − ceil(Σ lineH + n·overflow); last baseline =
        // v_offset + (n−1)·lineH + leading/2 + Ai.
        let expect = |n: f64| {
            let t = (100.0 * n + overflow * n).ceil();
            (400.0 - t) + (n - 1.0) * 100.0 + (100.0 - (ai + di)) / 2.0 + ai
        };
        let one_y = last_baseline(&one, VectorPlacement::Canvas2dParity);
        let two_y = last_baseline(&two, VectorPlacement::Canvas2dParity);
        assert!(
            (one_y - expect(1.0)).abs() < 1e-3,
            "one-para bottom baseline {one_y} != {}",
            expect(1.0)
        );
        assert!(
            (two_y - expect(2.0)).abs() < 1e-3,
            "two-para bottom baseline {two_y} != {}",
            expect(2.0)
        );
    }

    /// Scope pin: the reroute fires ONLY for fonts the otjs walk
    /// cannot shape OR that carry required GSUB features it skips — Inter
    /// (direct liga/kern, no rvrn/rclt) keeps the opentype.js path byte-exact
    /// through the dispatch.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn shape_vector_keeps_otjs_path_for_compatible_fonts() {
        let (store, catalog) = store_and_catalog();
        let font = store
            .font_for_variant(&catalog, "Inter", 400, "normal", None)
            .unwrap();
        assert!(!store.font_needs_harfbuzz(&catalog, "Inter", 400, "normal", None));
        let via_dispatch = shape_vector(
            &store,
            &catalog,
            "Inter",
            400,
            "normal",
            None,
            &font,
            "Office AV",
        )
        .unwrap();
        let via_otjs = shape(&font, "Office AV").unwrap();
        assert_eq!(via_dispatch, via_otjs);
    }

    /// Regression: Fraunces carries the REQUIRED `rvrn` feature (which
    /// remaps e.g. italic `l`→`l.alt`) that the otjs walk never applies. Even
    /// though `string_to_glyphs` does NOT throw on it, the vector lane must
    /// reroute it through HarfBuzz so the canvas glyphs match the browser/DOM
    /// editor. (Before this fix it stayed on the otjs path and rendered the
    /// wrong default-master glyph.)
    #[test]
    #[ignore = "needs unported Moda fixture full Fraunces Regular with rvrn (spacing subset omits rvrn)"]
    fn shape_vector_reroutes_rvrn_fonts_through_harfbuzz() {
        let (store, catalog) = crate::test_fixtures::oracle_store_and_catalog();
        let font = store
            .font_for_variant(&catalog, "Fraunces", 400, "normal", None)
            .unwrap();
        // The otjs walk parses and shapes Fraunces without throwing ...
        assert!(shape(&font, "circle").is_ok());
        // ... but `rvrn` makes the otjs glyphs disagree with the browser, so the
        // vector lane must reroute to HarfBuzz.
        assert!(store.font_needs_harfbuzz(&catalog, "Fraunces", 400, "normal", None));
    }

    /// Q-122 (Phase 2): the hybrid arm's ls==0 lane shapes through HarfBuzz
    /// — a Lato node (GSUB liga in type-7 extension lookups, the exact otjs
    /// `lookupType: 7` throw) SERVES vector under `Canvas2dParity`
    /// (eligibility is HB's, like the golden's Canvas2D engine). Now
    /// the `VectorParity` lane also serves it (font-capability
    /// reroute — pinned separately above), so BOTH placements render.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn canvas2d_parity_hb_serves_where_otjs_threw() {
        let (store, catalog) = store_and_catalog();
        let s = spec("<p><span style=\"font-family: Lato; font-size: 24px\">Office</span></p>");
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::Canvas2dParity,
            "hybrid arm must serve via HB",
        );
        // "Office" ligates (ffi) — fewer drawn glyphs than code points.
        assert!(ops.ops[0].records.len() < 6, "{}", ops.ops[0].records.len());
    }

    /// Q-122: extension-lookup kerning reaches the hybrid arm's pens — the
    /// A→V advance shrinks vs the otjs-blind bare advance (Lato kerns ONLY
    /// through type-9 extension lookups).
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn canvas2d_parity_hb_kerns_extension_lookups() {
        let (store, catalog) = store_and_catalog();
        let s = spec("<p><span style=\"font-family: Lato; font-size: 100px\">AV</span></p>");
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::Canvas2dParity,
            "expected vector decision",
        );
        let font = store
            .font_for_variant(&catalog, "Lato", 400, "normal", None)
            .unwrap();
        let scale = 100.0 / font.units_per_em as f64;
        let bare = font.advance_width(font.char_to_glyph_index('A')) * scale;
        let recs = &ops.ops[0].records;
        assert_eq!(recs.len(), 2);
        let pen_delta = (recs[1].pen_x - recs[0].pen_x) as f64;
        assert!(
            pen_delta < bare - 1.0,
            "extension kern must shrink the A→V pen: {pen_delta} vs bare {bare}"
        );
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn complex_script_declines_to_fallback() {
        // The otjs vector shaper cannot do bidi / joining / reordering / GPOS
        // marks, so any complex-script run declines the whole node (→ MSDF →
        // cosmic-text, which shapes it correctly). Simple scripts (Latin/CJK/
        // Cyrillic) must NOT hit this decline — they keep the crisp GPU path.
        let (store, catalog) = store_and_catalog();
        let complex = [
            "<p><span style=\"font-family: Inter; font-size: 32px\">مرحبا</span></p>", // Arabic
            "<p><span style=\"font-family: Inter; font-size: 32px\">שלום</span></p>",  // Hebrew
            "<p><span style=\"font-family: Inter; font-size: 32px\">नमस्ते</span></p>",  // Devanagari
            "<p><span style=\"font-family: Inter; font-size: 32px\">สวัสดี</span></p>",  // Thai
            "<p><span style=\"font-family: Inter; font-size: 32px\">வணக்கம்</span></p>", // Tamil
        ];
        for html in complex {
            let d = build_vector_node_ops(
                &spec(html),
                &store,
                &catalog,
                1.0,
                VectorPlacement::VectorParity,
            );
            match d {
                VectorEngineDecision::Fallback { reason, .. } => {
                    assert_eq!(reason, "complex-script", "{html:?}")
                }
                VectorEngineDecision::Vector(_) => {
                    panic!("complex script must decline, not render via otjs: {html:?}")
                }
            }
        }
        // Latin must NOT trip the complex-script decline.
        let latin = build_vector_node_ops(
            &spec(
                "<p><span style=\"font-family: Inter; font-size: 32px\">Hello world 123</span></p>",
            ),
            &store,
            &catalog,
            1.0,
            VectorPlacement::VectorParity,
        );
        assert!(
            !matches!(&latin, VectorEngineDecision::Fallback { reason, .. } if reason == "complex-script"),
            "Latin must keep the vector path, got {latin:?}"
        );
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn glow_emits_additive_pass_behind_fill() {
        // vectorTextEffects.ts:188-210: glow = additive analytic pass, blur =
        // radius, quad padded by radius, opacityFactor = clamp01(opacity) ×
        // max(0, intensity) MULTIPLYING node opacity.
        let (store, catalog) = store_and_catalog();
        let mut s = spec("<p><span style=\"font-family: Inter; font-size: 32px\">Fx</span></p>");
        s.opacity = 0.5;
        s.effects = vec![crate::ops::TextEffectSpec::Glow {
            color: "#00ff00".into(),
            radius_px: Some(12.0),
            intensity: Some(2.0),
            opacity: Some(0.8),
        }];
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector",
        );
        assert_eq!(ops.ops.len(), 2); // glow pass, fill
        let glow = &ops.ops[0];
        assert_eq!(glow.pipeline, VectorOpPipeline::Analytic);
        assert!(glow.additive);
        assert_eq!(glow.blur_px, 12.0);
        assert_eq!(glow.color, [0.0, 1.0, 0.0, 1.0]);
        // node 0.5 × (clamp01(0.8) × 2.0) = 0.8
        assert!((glow.opacity - 0.8).abs() < 1e-6);
        // glow pens are NOT offset; quads pad outward beyond the fill pass
        let fill = &ops.ops[1];
        assert!(!fill.additive);
        assert_eq!(glow.records[0].pen_x, fill.records[0].pen_x);
        assert!(glow.records[0].em_box[0] < fill.records[0].em_box[0]);

        // radius 0 / non-finite radius = pass-inert: no glow op, and the
        // routing predicate stays false (vectorTextEffects.ts:193-197).
        let mut s0 = s.clone();
        s0.effects = vec![crate::ops::TextEffectSpec::Glow {
            color: "#00ff00".into(),
            radius_px: Some(0.0),
            intensity: None,
            opacity: None,
        }];
        assert!(!structured_effects_active(&s0));
        let ops0 = expected_vector(
            &s0,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector",
        );
        assert_eq!(ops0.ops.len(), 1); // fill only
        assert!(structured_effects_active(&s));
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn echo_expands_farthest_copy_first_with_scale_and_lerp() {
        // appendEchoPasses (vectorTextEffects.ts:126-160): copy k at
        // k × offset, scaleStep^k about the box center, colorStart→colorEnd
        // lerp, opacityStep^k × node opacity; farthest copy first.
        let (store, catalog) = store_and_catalog();
        let mut s = spec("<p><span style=\"font-family: Inter; font-size: 32px\">Fx</span></p>");
        s.effects = vec![crate::ops::TextEffectSpec::Echo {
            count: Some(3.0),
            offset_x: Some(6.0),
            offset_y: Some(4.0),
            scale_step: Some(0.9),
            color_start: Some("#ff0000".into()),
            color_end: Some("#0000ff".into()),
            opacity_step: Some(0.8),
        }];
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector",
        );
        assert_eq!(ops.ops.len(), 4); // 3 echo copies + fill
        let fill = &ops.ops[3];
        // Farthest copy (k=3) first: t=1 → colorEnd; nearest (k=1) last: t=0
        // → colorStart.
        assert_eq!(ops.ops[0].color, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(ops.ops[2].color, [1.0, 0.0, 0.0, 1.0]);
        for (i, k) in [(0usize, 3i32), (1, 2), (2, 1)] {
            let op = &ops.ops[i];
            assert_eq!(op.pipeline, VectorOpPipeline::Analytic);
            assert!(!op.additive);
            assert!(
                (op.opacity as f64 - 0.8f64.powi(k)).abs() < 1e-6,
                "copy {k}"
            );
            // Pen offset k × (6, 4) plus the scale pull toward the box
            // center: penX' = ox + (pen - ox)·s^k + 6k.
            let scale = 0.9f64.powi(k);
            let ox = 1000.0 / 2.0; // width 1000
            let base_pen = fill.records[0].pen_x as f64;
            let expect_x = ox + (base_pen - ox) * scale + 6.0 * k as f64;
            assert!(
                (op.records[0].pen_x as f64 - expect_x).abs() < 1e-3,
                "copy {k} pen {} != {expect_x}",
                op.records[0].pen_x
            );
            // Shrunken copies scale the instance font size.
            assert!(
                (op.records[0].font_size_px as f64 - 32.0 * scale).abs() < 1e-3,
                "copy {k}"
            );
        }
        // count < 1 / non-finite offsets render nothing.
        let mut s0 = s.clone();
        s0.effects = vec![crate::ops::TextEffectSpec::Echo {
            count: Some(0.0),
            offset_x: Some(6.0),
            offset_y: Some(4.0),
            scale_step: None,
            color_start: None,
            color_end: None,
            opacity_step: None,
        }];
        assert!(!structured_effects_active(&s0));
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn extrude_emits_swept_wall_behind_fill() {
        // vectorTextExtrude.ts: a swept side wall (Tess pipeline) behind the
        // crisp analytic front face; depth in local px at the node-local
        // y-down angle (0° = +x).
        let (store, catalog) = store_and_catalog();
        let mut s = spec("<p><span style=\"font-family: Inter; font-size: 32px\">Fx</span></p>");
        s.effects = vec![crate::ops::TextEffectSpec::Extrude {
            depth_px: Some(10.0),
            angle_deg: Some(0.0),
            color: "#444444".into(),
        }];
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector",
        );
        assert_eq!(ops.ops.len(), 2); // wall, fill
        let wall = &ops.ops[0];
        assert_eq!(wall.pipeline, VectorOpPipeline::Tess);
        assert!(!wall.tris.is_empty());
        assert_eq!(wall.tris.len() % 12, 0); // two triangles (12 floats) per segment
        assert_eq!(wall.opacity, 1.0); // node opacity (no override/factor)
                                       // Each quad's swept corners sit depth px along +x: triangle 1 =
                                       // [a, b, b+d]; verify x2 - x1 == 10 within f32 for the first quad.
        let t = &wall.tris;
        assert!((t[4] - t[2] - 10.0).abs() < 1e-3);
        assert!((t[5] - t[3]).abs() < 1e-3); // dy = 0 at angle 0

        // depth 0 is pass-inert.
        let mut s0 = s.clone();
        s0.effects = vec![crate::ops::TextEffectSpec::Extrude {
            depth_px: Some(0.0),
            angle_deg: None,
            color: "#444444".into(),
        }];
        assert!(!structured_effects_active(&s0));
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn pass_order_shadow_effects_stroke_fill() {
        // resolveVectorTextEffectPasses order contract (vectorTextEffects.ts
        // module doc): effects[] (array order: drop shadow first here; echo
        // expands) → fill → legacy stroke.
        let (store, catalog) = store_and_catalog();
        let mut s = spec("<p><span style=\"font-family: Inter; font-size: 32px\">Fx</span></p>");
        s.stroke_color = Some("#ff0000".into());
        s.stroke_width = Some(3.0);
        s.effects = vec![
            shadow(0.0),
            crate::ops::TextEffectSpec::Glow {
                color: "#00ffcc".into(),
                radius_px: Some(8.0),
                intensity: None,
                opacity: None,
            },
            crate::ops::TextEffectSpec::Echo {
                count: Some(2.0),
                offset_x: Some(5.0),
                offset_y: Some(5.0),
                scale_step: None,
                color_start: None,
                color_end: None,
                opacity_step: None,
            },
            crate::ops::TextEffectSpec::Extrude {
                depth_px: Some(6.0),
                angle_deg: None,
                color: "#222222".into(),
            },
        ];
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector",
        );
        // shadow, glow, echo k=2, echo k=1, extrude wall, fill, stroke
        assert_eq!(ops.ops.len(), 7);
        assert_eq!(ops.ops[0].blur_px, 0.0); // shadow (no blur set)
        assert_eq!(ops.ops[0].opacity, 0.3);
        assert!(ops.ops[1].additive); // glow
        assert_eq!(
            ops.ops[2].records[0].pen_x - ops.ops[5].records[0].pen_x,
            10.0
        ); // echo k=2
        assert_eq!(
            ops.ops[3].records[0].pen_x - ops.ops[5].records[0].pen_x,
            5.0
        ); // echo k=1
        assert_eq!(ops.ops[4].pipeline, VectorOpPipeline::Tess); // extrude wall
        assert_eq!(ops.ops[6].stroke_px, 3.0); // stroke
        assert_eq!(ops.ops[5].blur_px, 0.0); // fill
        assert_eq!(ops.ops[5].stroke_px, 0.0);
    }

    /// Pen x of every placed fill glyph, in draw order.
    fn fill_pens(ops: &crate::vector::VectorNodeOps) -> Vec<f32> {
        ops.ops
            .iter()
            .filter(|o| o.pipeline == VectorOpPipeline::Analytic)
            .flat_map(|o| o.records.iter().map(|r| r.pen_x))
            .collect()
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn emoji_stays_in_the_vector_lane_as_a_one_em_image_cell() {
        // A text layer with an emoji used to decline the whole node (`emoji-run`) to the
        // degraded raster. It now stays vector: the text keeps its outlines and spacing,
        // the emoji takes one em of the line as an image cell.
        let (store, catalog) = store_and_catalog();
        let node = |text: &str| {
            let s = spec(&format!(
                "<p><span style=\"font-family: Inter; font-size: 40px\">{text}</span></p>"
            ));
            expected_vector(&s, &store, &catalog, VectorPlacement::VectorParity, text)
        };
        let with_emoji = node("Launch 🚀 today");
        let one_space = node("Launch today");
        let two_spaces = node("Launch  today");

        assert_eq!(with_emoji.emojis.len(), 1);
        let e = &with_emoji.emojis[0];
        assert_eq!(e.cluster, "🚀");

        let pens = fill_pens(&with_emoji);
        let plain = fill_pens(&one_space);
        let spaced = fill_pens(&two_spaces);
        assert_eq!(pens.len(), 11, "every letter drawn: {pens:?}");
        // "Launch" is placed exactly as without the emoji.
        assert_eq!(pens[..6], plain[..6]);
        // "today" moves right by exactly one space and one em: the emoji's slot.
        let space = spaced[6] - plain[6];
        assert!(space > 5.0, "space advance {space}");
        for i in 6..11 {
            assert!((pens[i] - (plain[i] + space + 40.0)).abs() < 1e-3, "glyph {i}: {pens:?}");
        }
        // The cell starts after "Launch " and is one em wide, its top one em above the
        // baseline and its bottom a quarter em below.
        let x0 = e.corners[0];
        assert!((x0 + 40.0 + space - pens[6]).abs() < 1e-3, "cell x {x0}");
        assert!((e.corners[2] - e.corners[0] - 40.0).abs() < 1e-4);
        let baseline = with_emoji
            .ops
            .iter()
            .find(|o| o.pipeline == VectorOpPipeline::Analytic)
            .unwrap()
            .records[0]
            .baseline_y;
        assert!((e.corners[1] - (baseline - 40.0)).abs() < 1e-3);
        assert!((e.corners[5] - (baseline + 10.0)).abs() < 1e-3);
        assert_eq!(e.opacity, 1.0);
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn emoji_cells_move_and_fade_with_their_kinetic_unit() {
        // `text.units` used to leave emoji where they rest at full opacity (visible before their turn); the cell now
        // takes the transform of its first code point, about its own centre.
        let (store, catalog) = store_and_catalog();
        let text = "Go 🚀 x";
        let base = spec(&format!("<p><span style=\"font-family: Inter; font-size: 40px\">{text}</span></p>"));
        let rest = expected_vector(&base, &store, &catalog, VectorPlacement::VectorParity, text);
        let mut moved = base.clone();
        // G o ␠ 🚀 ␠ x: the rocket (code point 3) rises 30 px, doubles and fades to 0.25; the rest stay.
        moved.unit_transforms = (vec![GlyphXform::IDENTITY; 6]).into();
        std::sync::Arc::make_mut(&mut moved.unit_transforms)[3] = GlyphXform::resolve(0.0, -30.0, 2.0, 0.0, 0.25);
        let moved = expected_vector(&moved, &store, &catalog, VectorPlacement::VectorParity, text);
        let (a, b) = (&rest.emojis[0], &moved.emojis[0]);
        let centre = |c: &[f32; 8]| ((c[0] + c[4]) / 2.0, (c[1] + c[5]) / 2.0);
        let (ca, cb) = (centre(&a.corners), centre(&b.corners));
        assert!((cb.0 - ca.0).abs() < 1e-3 && (cb.1 - (ca.1 - 30.0)).abs() < 1e-3, "{ca:?} → {cb:?}");
        assert!(((b.corners[2] - b.corners[0]) - 2.0 * (a.corners[2] - a.corners[0])).abs() < 1e-3, "twice as wide");
        assert_eq!((a.opacity, b.opacity), (1.0, 0.25));
        assert_eq!(fill_pens(&rest), fill_pens(&moved), "the letters stay");
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn emoji_clusters_map_to_their_placeholders_across_lines() {
        // Placeholders take the emoji runs in order, wrapped lines included.
        let (store, catalog) = store_and_catalog();
        let mut s = spec(
            "<p><span style=\"font-family: Inter; font-size: 20px\">a 👋🏽 b</span></p>\
             <p><span style=\"font-family: Inter; font-size: 20px\">c 👨‍👩‍👧 d 🇯🇵</span></p>",
        );
        s.width = Some(60.0);
        let ops = expected_vector(&s, &store, &catalog, VectorPlacement::VectorParity, "vector");
        let clusters: Vec<&str> = ops.emojis.iter().map(|e| e.cluster.as_str()).collect();
        assert_eq!(clusters, vec!["👋🏽", "👨‍👩‍👧", "🇯🇵"]);
        // Later paragraphs sit lower.
        assert!(ops.emojis[1].corners[1] > ops.emojis[0].corners[1]);
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn emoji_carry_the_text_effects() {
        // A drop shadow and a text stroke re-emit every emoji as a tinted silhouette copy
        // (shadow first, then stroke, back to front), under the emoji itself.
        let (store, catalog) = store_and_catalog();
        let mut s = spec("<p><span style=\"font-family: Inter; font-size: 40px\">Launch 🚀</span></p>");
        s.effects = vec![shadow(6.0)];
        s.stroke_color = Some("#ff0000".into());
        s.stroke_width = Some(4.0);
        let ops = expected_vector(&s, &store, &catalog, VectorPlacement::VectorParity, "vector");
        assert_eq!(ops.emojis.len(), 3, "{:?}", ops.emojis);
        let (sh, st, fill) = (&ops.emojis[0], &ops.emojis[1], &ops.emojis[2]);
        assert!(fill.effect.is_none());
        let sh_fx = sh.effect.as_ref().expect("shadow copy");
        let st_fx = st.effect.as_ref().expect("stroke copy");
        // 40 px → 64 texels per em: 1.6 texels per px.
        assert_eq!((sh_fx.blur_texels, sh_fx.stroke_texels), (9.5, 0.0));
        assert_eq!((st_fx.blur_texels, st_fx.stroke_texels), (0.0, 6.5));
        assert!((sh_fx.gain - 0.3).abs() < 1e-6 && sh_fx.color[..3] == [0.0, 0.0, 0.0]);
        assert_eq!(st_fx.color, [1.0, 0.0, 0.0, 1.0]);
        assert!(!sh_fx.additive);
        // The shadow copy sits at the shadow offset, grown by its pad on every side.
        let pad = sh_fx.pad_texels as f32 / 1.6;
        assert!((sh.corners[0] - (fill.corners[0] + 4.0 - pad)).abs() < 1e-3, "{:?} {:?}", sh.corners, fill.corners);
        assert!((sh.corners[1] - (fill.corners[1] + 4.0 - pad)).abs() < 1e-3);
        assert!((sh.corners[4] - (fill.corners[4] + 4.0 + pad)).abs() < 1e-3);
        // The stroke copy stays in place.
        let pad = st_fx.pad_texels as f32 / 1.6;
        assert!((st.corners[0] - (fill.corners[0] - pad)).abs() < 1e-3);
        // A glow's copy is emitted light.
        let mut g = spec("<p><span style=\"font-family: Inter; font-size: 56px\">Glow ✨</span></p>");
        g.effects = vec![crate::ops::TextEffectSpec::Glow {
            color: "#ff00aa".into(),
            radius_px: Some(12.0),
            intensity: None,
            opacity: None,
        }];
        let glow = expected_vector(&g, &store, &catalog, VectorPlacement::VectorParity, "vector");
        assert_eq!(glow.emojis.len(), 2);
        assert!(glow.emojis[0].effect.as_ref().is_some_and(|fx| fx.additive && fx.blur_texels > 10.0));
        // Keys round-trip.
        let key = sh_fx.key("🚀");
        assert_eq!(EmojiEffect::parse_key(&key), (Some(sh_fx.clone()), "🚀"));
        assert_eq!(EmojiEffect::parse_key("🚀"), (None, "🚀"));
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn emoji_on_a_path_turns_with_the_tangent() {
        let (store, catalog) = store_and_catalog();
        let mut s = spec("<p><span style=\"font-family: Inter; font-size: 40px\">ab 🚀 cd</span></p>");
        // A vertical line, top to bottom: the text runs downward, turned 90°.
        s.text_path = Some(crate::text_path::TextPathSpec {
            points: vec![(0.0, 0.0), (0.0, 1000.0)],
            closed: false,
            flip: false,
            start_offset: 0.0,
            align: crate::text_path::TextPathAlign::Baseline,
        });
        let ops = expected_vector(&s, &store, &catalog, VectorPlacement::VectorParity, "vector");
        assert_eq!(ops.emojis.len(), 1, "type on a path keeps the emoji");
        let c = ops.emojis[0].corners;
        // Top-left → top-right runs along the path (down the page), one em long.
        assert!((c[2] - c[0]).abs() < 1e-3 && (c[3] - c[1] - 40.0).abs() < 1e-3, "{c:?}");
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn emoji_decline_reports_every_affected_family_deduped() {
        // Where emoji still decline (the oracle dump, the text-layer extractor), one
        // emoji declines the WHOLE node, so every family in it loses the primary
        // engine — not just the first. Inter appears twice to pin the dedup.
        let (store, catalog) = store_and_catalog();
        let runs = extract_msdf_runs_by_paragraph(
            "<p><span style=\"font-family: Inter\">hi </span>\
             <span style=\"font-family: Lato\">there</span>\
             <span style=\"font-family: Inter\"> 😀</span></p>",
            &catalog,
        );
        let RunResolution::Fallback { reason, diag, .. } =
            crate::vector::node_layout::resolve_vector_runs(&runs, &store, &catalog)
        else {
            panic!("expected fallback");
        };
        let families = diag.families;
        assert_eq!(reason, "emoji-run");
        assert_eq!(
            families,
            vec!["Inter".to_string(), "Lato".to_string()],
            "every affected family, first-seen order, deduped"
        );
    }

    /// A synthetic oblique shears about the baseline even though the shader
    /// pivots `lin` about the em-box center: a point on the baseline stays put.
    #[test]
    fn synthetic_oblique_pivots_at_the_baseline() {
        let em_box = [-0.05f32, -0.8, 0.6, 0.25];
        let size = 40.0f32;
        let x = GlyphXform::IDENTITY.with_synthetic_oblique(em_box, size);
        assert_eq!(x.lin, [1.0, -0.25, 0.0, 1.0]);
        let (cx, cy) = (
            (em_box[0] + em_box[2]) * 0.5 * size,
            (em_box[1] + em_box[3]) * 0.5 * size,
        );
        // shader: p' = C + lin·(p − C) + off
        let apply = |px: f32, py: f32| {
            let (dx, dy) = (px - cx, py - cy);
            (
                cx + x.lin[0] * dx + x.lin[1] * dy + x.off[0],
                cy + x.lin[2] * dx + x.lin[3] * dy + x.off[1],
            )
        };
        let (bx, by) = apply(10.0, 0.0);
        assert!((bx - 10.0).abs() < 1e-4 && by.abs() < 1e-4);
        let (tx, _) = apply(10.0, -20.0); // 20 px above the baseline leans right by 5
        assert!((tx - 15.0).abs() < 1e-4);
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn emoji_only_text_stays_in_the_vector_lane() {
        // Emoji-only text used to route to the degraded tier before the vector lane saw it
        // (`select-text-engine:canvas2d`). It is vector text now: one image cell per
        // cluster, adjacent ones one em apart, inside the node's box.
        let (store, catalog) = store_and_catalog();
        let s = spec("<p><span style=\"font-family: Inter; font-size: 40px\">🚀🚀 ✨</span></p>");
        let ops = expected_vector(&s, &store, &catalog, VectorPlacement::VectorParity, "emoji-only is vector");
        let clusters: Vec<&str> = ops.emojis.iter().map(|e| e.cluster.as_str()).collect();
        assert_eq!(clusters, vec!["🚀", "🚀", "✨"]);
        let c = |i: usize| ops.emojis[i].corners;
        assert!((c(1)[0] - c(0)[0] - 40.0).abs() < 1e-3, "adjacent emoji one em apart");
        assert!(c(2)[0] > c(1)[2], "the space separates the third");
        // The baseline sits where a text line of this size puts it (the cell's top em is
        // mostly transparent headroom above the glyph).
        assert!(c(0)[1] > -4.0 && c(0)[5] < 60.0, "cell near the box: {:?}", c(0));
    }

    /// Native Chrome Canvas 2D keeps A/V kerning with nonzero `letterSpacing`
    /// and includes tracking after each character, including the space.
    /// The old oracle modeled Moda's manual per-character helper instead.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn canvas2d_parity_letter_spacing_keeps_kerning_and_tracks_word_gaps() {
        let (store, catalog) = store_and_catalog();
        let ls = 4.0;
        let size = 100.0;
        let html = format!(
            "<p><span style=\"font-family: Fraunces; font-size: {size}px; \
             letter-spacing: {ls}px\">AV AV</span></p>"
        );
        let s = spec(&html);
        let records = |placement| {
            let ops = expected_vector(&s, &store, &catalog, placement, "expected vector decision");
            assert_eq!(ops.ops.len(), 1);
            ops.ops[0]
                .records
                .iter()
                .map(|r| r.pen_x as f64)
                .collect::<Vec<_>>()
        };
        let v = records(VectorPlacement::VectorParity);
        let c = records(VectorPlacement::Canvas2dParity);
        assert_eq!(v.len(), 4); // A V A V (space draws nothing)
        assert_eq!(c.len(), 4);

        let font = store
            .font_for_variant(&catalog, "Fraunces", 400, "normal", None)
            .expect("Fraunces vendored");
        let scale = size / font.units_per_em as f64;
        let gid = |ch: char| font.char_to_glyph_index(ch);
        let kern_av = font.get_kerning_value(gid('A'), gid('V')).unwrap() * scale;
        assert!(kern_av < -1.0, "AV must kern in this pin (got {kern_av})");
        let adv = |ch: char| font.advance_width(gid(ch)) * scale;

        // Chrome 141, Fraunces Regular at 100px and letterSpacing=4px:
        // A=71.5, V=70.35, space=22.6, kern(AV)=-4.5 (unhinted).
        // AV=145.35, space=26.6, second word=171.95, total=317.3.
        let word = adv('A') + kern_av + ls + adv('V') + ls;
        for pens in [&v, &c] {
            assert!(((pens[1] - pens[0]) - (adv('A') + kern_av + ls)).abs() < 1e-3);
            assert!((pens[2] - (word + adv(' ') + ls)).abs() < 1e-3);
            assert!(((pens[3] - pens[2]) - (adv('A') + kern_av + ls)).abs() < 1e-3);
        }
        assert_eq!(v, c);
    }

    /// Q-121: with letterSpacing == 0 the Canvas2D pipeline is a plain
    /// whole-string `fillText` — the letter-spacing lane must NOT fire and
    /// `Canvas2dParity` glyph x-placement stays identical to `VectorParity`
    /// (top-aligned so the Q-120 vertical term is zero too).
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn canvas2d_parity_zero_letter_spacing_keeps_vector_shaping() {
        let (store, catalog) = store_and_catalog();
        let s =
            spec("<p><span style=\"font-family: Fraunces; font-size: 100px\">AV To AV</span></p>");
        let pens = |placement| {
            let ops = expected_vector(&s, &store, &catalog, placement, "expected vector decision");
            ops.ops[0]
                .records
                .iter()
                .map(|r| (r.pen_x, r.baseline_y))
                .collect::<Vec<_>>()
        };
        let v = pens(VectorPlacement::VectorParity);
        let c = pens(VectorPlacement::Canvas2dParity);
        // HB and otjs agree glyph-for-glyph horizontally (the Fraunces
        // direct-kern pin); baselines differ by exactly the Q-125 canvas
        // metric rounding, which is placement parity, not shaping.
        assert_eq!(v.len(), c.len());
        let m = store
            .metrics_for_variant(&catalog, "Fraunces", 400, "normal", None)
            .unwrap();
        let af = (m.ascent / m.units_per_em) * 100.0;
        let df = (m.descent.abs() / m.units_per_em) * 100.0;
        let (ai, di) = ((af + 0.5).floor(), (df + 0.5).floor());
        let expected_delta = ((ai - di) / 2.0 - (af - df) / 2.0) as f32;
        for ((vx, vy), (cx, cy)) in v.iter().zip(c.iter()) {
            assert_eq!(vx, cx, "ls==0 HB shaping must keep otjs pen_x");
            assert!(
                (cy - vy - expected_delta).abs() < 1e-3,
                "baseline delta {} != Q-125 rounding {expected_delta}",
                cy - vy
            );
        }
    }

    /// F7.3: `font-feature-settings` on a run reaches the drawn glyphs on both
    /// placements — "liga" 0 un-ligates fi in Lato, and in Inter,
    /// "tnum" widens the proportional 1 so the pens move.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn font_feature_settings_reach_the_drawn_glyphs() {
        let (store, catalog) = store_and_catalog();
        let html = |family: &str, text: &str, ff: &str| {
            format!(
                "<p><span style=\"font-family: {family}; font-size: 40px{ff}\">{text}</span></p>"
            )
        };
        for placement in [
            VectorPlacement::VectorParity,
            VectorPlacement::Canvas2dParity,
        ] {
            let records = |family: &str, text: &str, ff: &str| {
                let ops = expected_vector(
                    &spec(&html(family, text, ff)),
                    &store,
                    &catalog,
                    placement,
                    "expected vector decision",
                );
                ops.ops[0]
                    .records
                    .iter()
                    .map(|r| r.pen_x)
                    .collect::<Vec<_>>()
            };
            let plain = records("Lato", "fifi", "");
            let unligated = records("Lato", "fifi", "; font-feature-settings: 'liga' 0");
            assert_eq!(plain.len(), 2, "{placement:?}: fi ligates by default");
            assert_eq!(unligated.len(), 4, "{placement:?}: liga 0 draws f and i");
            let plain = records("Inter", "1111", "");
            let tabular = records("Inter", "1111", "; font-feature-settings: 'tnum' 1");
            assert_eq!(tabular.len(), plain.len());
            let last = |v: &[f32]| v[v.len() - 1];
            assert!(
                last(&tabular) > last(&plain) + 1.0,
                "{placement:?}: tabular 1s are wider ({} vs {})",
                last(&tabular),
                last(&plain)
            );
        }
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn text_transform_draws_the_transformed_glyphs() {
        // E6.3 text case: `text-transform` draws exactly what the literal
        // transformed text draws (same glyphs, same pens, same wraps), and the
        // untransformed text differs.
        let (store, catalog) = store_and_catalog();
        let draw = |css: &str, text: &str| {
            let html = format!(
                "<p><span style=\"font-family: Inter; font-size: 40px{css}\">{text}</span></p>"
            );
            let ops = expected_vector(
                &spec(&html),
                &store,
                &catalog,
                VectorPlacement::VectorParity,
                "expected vector decision",
            );
            ops.ops[0]
                .records
                .iter()
                .map(|r| (r.pen_x, r.baseline_y, r.em_box))
                .collect::<Vec<_>>()
        };
        let cases = [
            ("uppercase", "type case", "TYPE CASE"),
            ("lowercase", "TYPE Case", "type case"),
            ("capitalize", "type case", "Type Case"),
        ];
        for (transform, typed, literal) in cases {
            let transformed = draw(&format!("; text-transform: {transform}"), typed);
            assert_eq!(transformed, draw("", literal), "{transform}");
            assert_ne!(
                transformed,
                draw("", typed),
                "{transform} changes the glyphs"
            );
        }
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn text_wrap_balance_moves_glyphs_between_lines() {
        // E6.3: two lines either way; balance moves words from the first to
        // the second so their glyph counts even out, drawn with real advances.
        let (store, catalog) = store_and_catalog();
        let per_line = |wrap: &str| {
            let html = format!(
                "<p style=\"text-wrap: {wrap}\"><span style=\"font-family: Inter; font-size: 40px\">\
                 Balanced lines read better in a headline</span></p>"
            );
            let mut s = spec(&html);
            s.width = Some(700.0);
            let ops = expected_vector(
                &s,
                &store,
                &catalog,
                VectorPlacement::VectorParity,
                "expected vector decision",
            );
            let mut counts: Vec<(i64, usize)> = Vec::new();
            for r in &ops.ops[0].records {
                let y = r.baseline_y.round() as i64;
                match counts.last_mut() {
                    Some((ly, n)) if *ly == y => *n += 1,
                    _ => counts.push((y, 1)),
                }
            }
            counts.into_iter().map(|(_, n)| n).collect::<Vec<_>>()
        };
        let auto = per_line("auto");
        let balanced = per_line("balance");
        assert_eq!(auto.len(), 2, "{auto:?}");
        assert_eq!(balanced.len(), 2, "{balanced:?}");
        assert_eq!(auto.iter().sum::<usize>(), balanced.iter().sum::<usize>());
        assert!(
            balanced[0].abs_diff(balanced[1]) < auto[0].abs_diff(auto[1]),
            "auto {auto:?} vs balanced {balanced:?}"
        );
    }

    /// Where the shader puts a record's em point `(ex, ey)`: `pen + lin·(em·size − C) + C + off`.
    fn drawn(r: &VectorAnalyticRecord, ex: f32, ey: f32) -> (f32, f32) {
        let size = r.font_size_px;
        let cx = (r.em_box[0] + r.em_box[2]) * 0.5 * size;
        let cy = (r.em_box[1] + r.em_box[3]) * 0.5 * size;
        let [a, c, b, d] = r.glyph_lin;
        let (px, py) = (ex * size - cx, ey * size - cy);
        (
            r.pen_x + a * px + c * py + cx + r.glyph_off[0],
            r.baseline_y + b * px + d * py + cy + r.glyph_off[1],
        )
    }

    fn path_spec(html: &str, points: Vec<(f64, f64)>, closed: bool) -> TextNodeSpec {
        let mut s = spec(html);
        s.text_path = Some(crate::text_path::TextPathSpec {
            points,
            closed,
            ..Default::default()
        });
        s
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn text_on_a_path_sits_on_the_path_and_turns_with_it() {
        // F7.2: on a vertical path running down, every glyph's baseline midpoint lands on x = 100 and its x axis
        // points down (lin = rotation by 90°); glyph centres advance down the path in reading order.
        let (store, catalog) = store_and_catalog();
        let html = "<p><span style=\"font-family: Inter; font-size: 20px\">Path</span></p>";
        let ops = expected_vector(
            &path_spec(html, vec![(100.0, 0.0), (100.0, 400.0)], false),
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "vector",
        );
        let recs = &ops.ops[0].records;
        assert_eq!(recs.len(), 4);
        let mut last_y = -1.0f32;
        for r in recs {
            let mid_x = (r.em_box[0] + r.em_box[2]) * 0.5; // em units; bounds centre ± AA pad is symmetric
            let on = drawn(r, mid_x, 0.0);
            assert!((on.0 - 100.0).abs() < 0.05, "{on:?}");
            assert!(on.1 > last_y, "glyphs run down the path: {on:?}");
            last_y = on.1;
            let [a, c, b, d] = r.glyph_lin;
            assert!(
                a.abs() < 1e-5
                    && (b - 1.0).abs() < 1e-5
                    && (c + 1.0).abs() < 1e-5
                    && d.abs() < 1e-5
            );
        }
        // The first glyph starts at the path's start: its centre is half an advance down.
        let first = drawn(&recs[0], (recs[0].em_box[0] + recs[0].em_box[2]) * 0.5, 0.0);
        assert!(first.1 > 0.0 && first.1 < 20.0, "{first:?}");
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn closed_ring_clips_full_glyphs_at_the_seam_with_offsets_and_alignment() {
        let (store, catalog) = store_and_catalog();
        let ring: Vec<_> = (0..128).map(|i| {
            let t = i as f64 * std::f64::consts::TAU / 128.0;
            (100.0 + 40.0 * t.cos(), 100.0 + 40.0 * t.sin())
        }).collect();
        // Overset on purpose; no alignment may place it on an earlier lap.
        for align in ["left", "center", "right"] {
            for offset in [0.0, 0.75] {
                let html = format!("<p style=\"text-align:{align}\"><span style=\"font-family:Inter;font-size:20px\">MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM</span></p>");
                let mut s = path_spec(&html, ring.clone(), true);
                s.text_path.as_mut().unwrap().start_offset = offset;
                let measure = crate::text_path::PathMeasure::new(&ring, true, false).unwrap();
                let ops = expected_vector(&s, &store, &catalog, VectorPlacement::VectorParity, "vector");
                let records: Vec<_> = ops.ops.iter().flat_map(|o| &o.records).collect();
                assert!(!records.is_empty());
                assert!(records.len() < 36, "overset glyphs must be hidden");
                let mut previous = -1.0;
                for r in records {
                    let centre = drawn(r, (r.em_box[0] + r.em_box[2]) * 0.5, 0.0);
                    let angle = (centre.1 as f64 - 100.0).atan2(centre.0 as f64 - 100.0).rem_euclid(std::f64::consts::TAU);
                    let along = angle * 40.0;
                    let half = (r.em_box[2] - r.em_box[0]) as f64 * r.font_size_px as f64 * 0.5;
                    assert!(along >= offset * measure.length() - 0.5);
                    assert!(along > previous, "text never wraps back onto itself");
                    assert!(along + half < measure.length() + 0.5, "whole glyph fits before the seam");
                    previous = along;
                }
            }
        }
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn text_on_a_path_flips_offsets_and_overflows() {
        let (store, catalog) = store_and_catalog();
        let html = "<p><span style=\"font-family: Lato; font-size: 20px\">Overflowing text</span></p>";
        let line = vec![(0.0, 100.0), (200.0, 100.0)];
        let centres = |s: &TextNodeSpec| -> Vec<(f32, f32)> {
            let ops = expected_vector(s, &store, &catalog, VectorPlacement::VectorParity, "vector");
            ops.ops
                .first()
                .map(|o| {
                    o.records
                        .iter()
                        .map(|r| drawn(r, (r.em_box[0] + r.em_box[2]) * 0.5, 0.0))
                        .collect()
                })
                .unwrap_or_default()
        };
        let plain = centres(&path_spec(html, line.clone(), false));
        // Flip: the text runs right to left along the same line, upside down (lin = −identity).
        let mut flipped = path_spec(html, line.clone(), false);
        flipped.text_path.as_mut().unwrap().flip = true;
        let f = centres(&flipped);
        assert_eq!(f.len(), plain.len());
        assert!(f[0].0 > f[1].0, "{f:?}");
        // Start offset 50%: the text starts at x = 100, and what passes the end (x = 200) is hidden.
        let mut half = path_spec(html, line.clone(), false);
        half.text_path.as_mut().unwrap().start_offset = 0.5;
        let h = centres(&half);
        assert!(
            h.len() < plain.len(),
            "overflow hidden: {} vs {}",
            h.len(),
            plain.len()
        );
        assert!(h.iter().all(|c| c.0 >= 100.0 && c.0 <= 200.0), "{h:?}");
        // Closed-path geometry is circular, but text that fits stays in the first lap.
        let square = vec![(0.0, 0.0), (60.0, 0.0), (60.0, 60.0), (0.0, 60.0)];
        let wrapped = centres(&path_spec(html, square, true));
        assert_eq!(wrapped.len(), plain.len());
        // Align to path: ascender moves the glyphs below the path (the ascender line on it).
        let mut asc = path_spec(html, line, false);
        asc.text_path.as_mut().unwrap().align = crate::text_path::TextPathAlign::Ascender;
        let a = centres(&asc);
        assert!(a.iter().zip(&plain).all(|(a, p)| a.1 > p.1 + 10.0), "{a:?}");
    }

    /// A ring of radius `r` about (`c`, `c`) from its leftmost point, clockwise on screen over the top (Type on a
    /// Path's circle), as four cubics, flattened.
    fn ring(c: f64, r: f64) -> Vec<(f64, f64)> {
        use crate::text_path::PathCmd::{Cubic, Move};
        let k = 0.552_284_749_8 * r;
        crate::text_path::flatten(&[
            Move(c - r, c),
            Cubic(c - r, c - k, c - k, c - r, c, c - r),
            Cubic(c + k, c - r, c + r, c - k, c + r, c),
            Cubic(c + r, c + k, c + k, c + r, c, c + r),
            Cubic(c - k, c + r, c - r, c + k, c - r, c),
        ])
    }

    /// Each drawn glyph's anchor (its outline-box centre on the baseline): its distance into the straight text, and
    /// where the draw put it.
    fn path_anchors(ops: &crate::vector::VectorNodeOps) -> Vec<(f64, (f64, f64), f32)> {
        ops.ops
            .iter()
            .flat_map(|o| &o.records)
            .map(|r| {
                let mid = (r.em_box[0] + r.em_box[2]) * 0.5;
                let at = drawn(r, mid, 0.0);
                (r.pen_x as f64 + mid as f64 * r.font_size_px as f64, (at.0 as f64, at.1 as f64), r.font_size_px)
            })
            .collect()
    }

    /// No drift along a long string: 200 characters round a 200 px circle (closed, and flipped onto its inside) put
    /// every glyph's anchor where the analytic curvature spacing does — advance ÷ (1 + κ·c·size), κ = ±1/200 — on the
    /// circle, the last glyph as close as the first.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn a_long_string_round_a_circle_does_not_drift() {
        use crate::text_path::{curve_spacing_ratio, PathMeasure};
        let (store, catalog) = store_and_catalog();
        let (c, r, size) = (300.0, 200.0, 11.0);
        let text: String = "Type on a path keeps every glyph on its mark. ".chars().cycle().take(200).collect();
        let html = format!("<p><span style=\"font-family: Inter; font-size: {size}px\">{text}</span></p>");
        let pts = ring(c, r);
        for flip in [false, true] {
            let mut s = path_spec(&html, pts.clone(), true);
            s.text_path.as_mut().unwrap().flip = flip;
            let ops = expected_vector(&s, &store, &catalog, VectorPlacement::VectorParity, "vector");
            let anchors = path_anchors(&ops);
            let glyphs = text.chars().filter(|ch| !ch.is_whitespace()).count();
            assert_eq!(anchors.len(), glyphs, "flip {flip}: the whole string fits the lap");
            let m = PathMeasure::new(&pts, true, flip).unwrap();
            let ratio = curve_spacing_ratio(if flip { -1.0 / r } else { 1.0 / r }, size);
            let (mut worst_model, mut worst_circle) = (0.0f64, 0.0f64);
            for &(a, (x, y), _) in &anchors {
                let along = a / ratio;
                let (p, _) = m.at(along).unwrap();
                worst_model = worst_model.max((p.0 - x).hypot(p.1 - y));
                // The true circle: from 9 o'clock, clockwise over the top (anticlockwise when flipped).
                let turn = if flip { -along / r } else { along / r };
                let (ex, ey) = (c + r * (std::f64::consts::PI + turn).cos(), c + r * (std::f64::consts::PI + turn).sin());
                worst_circle = worst_circle.max((ex - x).hypot(ey - y));
            }
            eprintln!("flip {flip}: {} glyphs, worst {worst_model:.4} px from the analytic spacing, {worst_circle:.4} px from the circle", anchors.len());
            assert!(worst_model < 0.05, "flip {flip}: a glyph {worst_model:.3} px from the analytic spacing");
            assert!(worst_circle < 0.3, "flip {flip}: a glyph {worst_circle:.3} px off the circle's arc length");
        }
    }

    /// Text whose size changes partway: each glyph is spaced at its own size, so the 12 px run follows the 40 px run
    /// without a jump. Before, every glyph re-integrated the whole run at its own size: the first 12 px glyph sat
    /// ~6 px further along than the 40 px run ended (a gap; an overlap the other way round).
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn mixed_sizes_on_a_path_space_each_glyph_at_its_own_size() {
        use crate::text_path::{curve_spacing_ratio, PathMeasure};
        let (store, catalog) = store_and_catalog();
        let (c, r) = (300.0, 200.0);
        let pts = ring(c, r);
        let m = PathMeasure::new(&pts, true, false).unwrap();
        for (first, second) in [(40.0, 12.0), (12.0, 40.0)] {
            let html = format!(
                "<p><span style=\"font-family: Inter; font-size: {first}px\">MMMMMMMM </span>\
                 <span style=\"font-family: Inter; font-size: {second}px\">bbbbbbbbbbbb</span></p>"
            );
            let ops = expected_vector(&path_spec(&html, pts.clone(), true), &store, &catalog, VectorPlacement::VectorParity, "vector");
            let anchors = path_anchors(&ops);
            assert_eq!(anchors.len(), 20);
            // Where the second run starts in the straight text: its first glyph's pen.
            let boundary = ops
                .ops
                .iter()
                .flat_map(|o| &o.records)
                .filter(|rec| rec.font_size_px == second as f32)
                .map(|rec| rec.pen_x as f64)
                .fold(f64::INFINITY, f64::min);
            let (r1, r2) = (curve_spacing_ratio(1.0 / r, first), curve_spacing_ratio(1.0 / r, second));
            let mut worst = 0.0f64;
            for &(a, (x, y), _) in &anchors {
                let along = if a <= boundary { a / r1 } else { boundary / r1 + (a - boundary) / r2 };
                let (p, _) = m.at(along).unwrap();
                worst = worst.max((p.0 - x).hypot(p.1 - y));
            }
            eprintln!("{first} px then {second} px: worst {worst:.4} px");
            assert!(worst < 0.1, "{first} px then {second} px: a glyph {worst:.3} px from its piecewise spacing");
        }
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn numbered_list_draws_its_counters_in_the_hanging_indent() {
        // E7.16: `<ol>` items draw `1.` / `2.` (then `a.` a level down) right-aligned before the text, which starts
        // at the numbered hanging indent (max(24, 1.5em)) on every line; a plain paragraph draws no counter.
        let (store, catalog) = store_and_catalog();
        let p = |t: &str| {
            format!("<p><span style=\"font-family: Inter; font-size: 20px\">{t}</span></p>")
        };
        let html = format!(
            "<ol><li>{}</li><li>{}<ol><li>{}</li></ol></li></ol>",
            p("One"),
            p("Two"),
            p("Nested")
        );
        let ops = expected_vector(
            &spec(&html),
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "vector",
        );
        let records = &ops.ops[0].records;
        // Baselines: three lines, each with its counter glyphs before the text.
        let mut lines: Vec<(f32, Vec<f32>)> = Vec::new();
        for r in records {
            match lines.last_mut() {
                Some((y, xs)) if (*y - r.baseline_y).abs() < 0.5 => xs.push(r.pen_x),
                _ => lines.push((r.baseline_y, vec![r.pen_x])),
            }
        }
        assert_eq!(lines.len(), 3, "{lines:?}");
        // `1.` + `One` = 5 glyphs; `2.` + `Two` = 5; `a.` + `Nested` = 8.
        let counts: Vec<usize> = lines.iter().map(|(_, xs)| xs.len()).collect();
        assert_eq!(counts, vec![5, 5, 8]);
        // Level 0 text starts at 30px (1.5em of 20px); the counters sit left of it, inside the box.
        let text_x = lines[0].1[2];
        assert!((text_x - 30.0).abs() < 0.5, "text at {text_x}");
        assert!(lines[0].1[0] >= 0.0 && lines[0].1[1] < 30.0);
        // Level 1: indented by 1.5em more.
        let nested_text_x = lines[2].1[2];
        assert!(
            (nested_text_x - 60.0).abs() < 0.5,
            "nested text at {nested_text_x}"
        );
        // No bullet geometry for a numbered list.
        assert!(ops.ops.iter().all(|o| o.tris.is_empty()));
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn bullet_levels_draw_disc_ring_and_square() {
        // E7.16: nested bullets cycle disc → circle (a ring: more triangles, a hole) → square (two triangles).
        let (store, catalog) = store_and_catalog();
        let p = |t: &str| {
            format!("<p><span style=\"font-family: Inter; font-size: 20px\">{t}</span></p>")
        };
        let html = format!(
            "<ul><li>{}<ul><li>{}<ul><li>{}</li></ul></li></ul></li></ul>",
            p("Disc"),
            p("Circle"),
            p("Square")
        );
        let ops = expected_vector(
            &spec(&html),
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "vector",
        );
        let tris: Vec<f32> = ops
            .ops
            .iter()
            .flat_map(|o| o.tris.iter().copied())
            .collect();
        // disc: 16-gon fan = 14 triangles; ring: 32; square: 2.
        assert_eq!(tris.len() / 6, 14 + 32 + 2);
        // An explicit style wins over the level default.
        let html = format!(
            "<ul style=\"list-style-type: square\"><li>{}</li></ul>",
            p("Square")
        );
        let ops = expected_vector(
            &spec(&html),
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "vector",
        );
        let n: usize = ops.ops.iter().map(|o| o.tris.len() / 6).sum();
        assert_eq!(n, 2);
    }

    #[test]
    #[ignore = "needs unported Moda fixture fonts-variable/InterVariable.ttf"]
    fn font_variation_settings_instance_the_drawn_face() {
        // E6.3: a variable face drawn at the run's axis values. `wght` 700 set
        // on the Regular variant draws what the Bold named instance draws, and
        // differs from 400; explicit `opsz` beats the automatic one; an axis
        // the face lacks changes nothing.
        let (store, catalog) = crate::test_fixtures::variable_store_and_catalog();
        let draw = |css: &str| {
            let html = format!(
                "<p><span style=\"font-family: Inter Variable; font-size: 40px; {css}\">Variable type</span></p>"
            );
            let ops = expected_vector(
                &spec(&html),
                &store,
                &catalog,
                VectorPlacement::VectorParity,
                "expected vector decision",
            );
            ops.ops[0]
                .records
                .iter()
                .map(|r| (r.pen_x, r.em_box))
                .collect::<Vec<_>>()
        };
        let regular = draw("");
        let bold = draw("font-weight: 700");
        let wght700 = draw("font-variation-settings: 'wght' 700");
        assert_ne!(bold, regular, "the Bold instance is another face");
        assert_eq!(wght700, bold, "wght 700 on Regular = the Bold instance");
        let last = |v: &[(f32, [f32; 4])]| v[v.len() - 1].0;
        assert!(last(&wght700) > last(&regular) + 5.0, "bolder is wider");
        // The run's value wins over the variant's pinned coordinate.
        assert_eq!(
            draw("font-weight: 700; font-variation-settings: 'wght' 400"),
            regular
        );
        // At 40px the automatic opsz is the axis max (32); an explicit 14 differs.
        assert_ne!(draw("font-variation-settings: 'opsz' 14"), regular);
        assert_eq!(draw("font-variation-settings: 'opsz' 32"), regular);
        assert_eq!(draw("font-variation-settings: 'XXXX' 5"), regular);
    }

    #[test]
    #[ignore = "needs unported Moda fixture fonts-variable/InterVariable.ttf"]
    fn a_span_at_other_axis_values_draws_its_own_outlines_of_a_shared_glyph() {
        // The same glyph at two instances in one text: the span's `wght` 900 "e" and the run's 400 "e" are one glyph
        // id of one variant. The curve store kept one entry per (variant, glyph), so whichever drew first gave its
        // outline to both (a bold span bled onto the regular letters after it, or vanished).
        let (store, catalog) = crate::test_fixtures::variable_store_and_catalog();
        let html = "<p><span style=\"font-family: Inter Variable; font-size: 40px\">ee</span>\
            <span style=\"font-family: Inter Variable; font-size: 40px; font-variation-settings: 'wght' 900\">ee</span>\
            <span style=\"font-family: Inter Variable; font-size: 40px\">ee</span></p>";
        let ops = expected_vector(
            &spec(html),
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector decision",
        );
        let records: Vec<_> = ops.ops.iter().flat_map(|o| o.records.iter()).collect();
        assert_eq!(records.len(), 6);
        let curves = |i: usize| (records[i].first_curve, records[i].em_box);
        assert_eq!(curves(0), curves(1));
        assert_eq!(
            curves(0),
            curves(4),
            "the regular e after the span is the regular e"
        );
        assert_eq!(curves(2), curves(3));
        assert_ne!(curves(2), curves(0), "the span's e is the wght 900 outline");
        assert!(
            records[2].em_box[2] > records[0].em_box[2],
            "bolder is wider"
        );
    }

    // ── KineticText per-glyph transform + opacity (Slice B) ──
    // Plan-level proof (GPU-independent): the renderer attaches the right
    // per-glyph transform/opacity to every analytic record (fill AND effect
    // passes), maps glyphs → canonical code points correctly, and keeps static
    // text byte-exact (identity). The shader applies these per-instance fields
    // (vector-text-analytic.wgsl `kineticGlyphLocal` + `glyphOpacity`).

    #[test]
    fn glyph_xform_resolve_builds_matrix() {
        let id = GlyphXform::resolve(0.0, 0.0, 1.0, 0.0, 1.0);
        assert_eq!(id, GlyphXform::IDENTITY);
        assert!(id.is_identity());
        // Uniform scale 2 → diagonal linear part; offset + opacity passthrough.
        let s = GlyphXform::resolve(5.0, -3.0, 2.0, 0.0, 0.5);
        assert_eq!(s.lin, [2.0, 0.0, 0.0, 2.0]);
        assert_eq!(s.off, [5.0, -3.0]);
        assert_eq!(s.opacity, 0.5);
        // 90° rotation, row-major [cos, -sin, sin, cos] = [0, -1, 1, 0].
        let r = GlyphXform::resolve(0.0, 0.0, 1.0, 90.0, 1.0);
        assert!(r.lin[0].abs() < 1e-6 && (r.lin[1] + 1.0).abs() < 1e-6);
        assert!((r.lin[2] - 1.0).abs() < 1e-6 && r.lin[3].abs() < 1e-6);
        // Opacity clamps to [0, 1].
        assert_eq!(GlyphXform::resolve(0.0, 0.0, 1.0, 0.0, 5.0).opacity, 1.0);
        assert_eq!(GlyphXform::resolve(0.0, 0.0, 1.0, 0.0, -1.0).opacity, 0.0);
    }

    #[test]
    fn glyph_xform_resolve_sanitizes_nonfinite_outputs() {
        // A finite-but-huge f64 (1e300 * cos) overflows to ±inf in f32; a NaN
        // survives `clamp`. resolve must never emit non-finite GPU data — each
        // overflowing field falls back to its identity component.
        let huge = GlyphXform::resolve(1e300, 1e300, 1e300, 0.0, 1.0);
        assert!(
            huge.lin.iter().all(|v| v.is_finite())
                && huge.off.iter().all(|v| v.is_finite())
                && huge.opacity.is_finite(),
            "non-finite leaked: {huge:?}"
        );
        // Huge offset → identity offset (0), not inf.
        assert_eq!(huge.off, [0.0, 0.0]);
        // NaN opacity (NaN.clamp == NaN) → identity opacity (1.0), still finite.
        let nan = GlyphXform::resolve(0.0, 0.0, 1.0, 0.0, f64::NAN);
        assert_eq!(nan.opacity, 1.0);
        // NaN rotation → non-finite lin → identity linear part.
        let nan_rot = GlyphXform::resolve(0.0, 0.0, 1.0, f64::NAN, 1.0);
        assert!(nan_rot.lin.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn static_text_records_are_identity() {
        let (store, catalog) = store_and_catalog();
        let s = spec("<p><span style=\"font-family: Inter; font-size: 32px\">Hello</span></p>");
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector",
        );
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].records.len(), 5);
        // No unitTransforms → every record is identity → shader fast-path → byte-exact.
        for op in &ops.ops {
            for r in &op.records {
                assert_eq!(r.glyph_lin, [1.0, 0.0, 0.0, 1.0]);
                assert_eq!(r.glyph_off, [0.0, 0.0]);
                assert_eq!(r.glyph_opacity, 1.0);
            }
        }
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn unit_transforms_apply_per_glyph_on_fill() {
        let (store, catalog) = store_and_catalog();
        let mut s = spec("<p><span style=\"font-family: Inter; font-size: 32px\">AB</span></p>");
        // canonical "AB": idx0 'A' rises 10px; idx1 'B' scales 2× at half opacity.
        s.unit_transforms = (vec![
            GlyphXform::resolve(0.0, 10.0, 1.0, 0.0, 1.0),
            GlyphXform::resolve(0.0, 0.0, 2.0, 0.0, 0.5),
        ]).into();
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector",
        );
        let recs = &ops.ops[0].records; // single fill bucket
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].glyph_off, [0.0, 10.0]);
        assert_eq!(recs[0].glyph_opacity, 1.0);
        assert_eq!(recs[1].glyph_lin, [2.0, 0.0, 0.0, 2.0]);
        assert_eq!(recs[1].glyph_opacity, 0.5);
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn a_unit_of_several_glyphs_scales_and_turns_about_its_own_centre() {
        let (store, catalog) = store_and_catalog();
        let mut s = spec("<p><span style=\"font-family: Inter; font-size: 32px\">AB CD</span></p>");
        // "AB" is one unit (both code points name 0) scaled 2× and turned 30°; "CD" stays; the space is its own.
        let turn = GlyphXform::resolve(0.0, 0.0, 2.0, 30.0, 1.0);
        s.unit_transforms = (vec![
            turn,
            turn,
            GlyphXform::IDENTITY,
            GlyphXform::IDENTITY,
            GlyphXform::IDENTITY,
        ]).into();
        s.unit_groups = (vec![0, 0, 2, 3, 3]).into();
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector",
        );
        let recs = &ops.ops[0].records;
        assert_eq!(recs.len(), 4);
        // Each glyph's em-box centre C (the shader's pivot), and the unit's centre W (its glyphs' boxes).
        let centre = |r: &VectorAnalyticRecord| {
            let k = r.font_size_px as f64;
            (
                r.pen_x as f64 + (r.em_box[0] + r.em_box[2]) as f64 * 0.5 * k,
                r.baseline_y as f64 + (r.em_box[1] + r.em_box[3]) as f64 * 0.5 * k,
            )
        };
        let edge = |r: &VectorAnalyticRecord, i: usize| {
            let k = r.font_size_px as f64;
            if i % 2 == 0 {
                r.pen_x as f64 + r.em_box[i] as f64 * k
            } else {
                r.baseline_y as f64 + r.em_box[i] as f64 * k
            }
        };
        let w = (
            (edge(&recs[0], 0).min(edge(&recs[1], 0)) + edge(&recs[0], 2).max(edge(&recs[1], 2)))
                * 0.5,
            (edge(&recs[0], 1).min(edge(&recs[1], 1)) + edge(&recs[0], 3).max(edge(&recs[1], 3)))
                * 0.5,
        );
        // L(p − W) + W: the glyph's centre lands at W + L(C − W), which the shader reaches as C + off.
        let [a, c, b, d] = turn.lin.map(f64::from);
        for r in &recs[..2] {
            let (cx, cy) = centre(r);
            let want = (
                w.0 + a * (cx - w.0) + c * (cy - w.1),
                w.1 + b * (cx - w.0) + d * (cy - w.1),
            );
            let got = (cx + r.glyph_off[0] as f64, cy + r.glyph_off[1] as f64);
            assert!(
                (got.0 - want.0).abs() < 0.05 && (got.1 - want.1).abs() < 0.05,
                "{got:?} vs {want:?}"
            );
            assert_eq!(r.glyph_lin, turn.lin);
        }
        // The resting word is untouched.
        assert!(recs[2..]
            .iter()
            .all(|r| r.glyph_off == [0.0, 0.0] && r.glyph_lin == [1.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn a_word_unit_on_a_path_gathers_along_the_path_about_its_centre() {
        // #1717: a word unit on type on a path scales as one piece (its glyphs move toward the word's centre along
        // the path), not each glyph about itself at its resting place (the word spreading out as it shrinks).
        let (store, catalog) = store_and_catalog();
        let points = crate::text_path::flatten(&[
            crate::text_path::PathCmd::Move(0.0, 150.0),
            crate::text_path::PathCmd::Quad(350.0, -100.0, 700.0, 150.0),
        ]);
        let placed = |transforms: Vec<GlyphXform>, groups: Vec<u32>| {
            let mut s = spec("<p><span style=\"font-family: Inter; font-size: 48px\">curve</span></p>");
            s.text_path = Some(crate::text_path::TextPathSpec {
                points: points.clone(),
                closed: false,
                flip: false,
                start_offset: 0.3,
                align: crate::text_path::TextPathAlign::Baseline,
            });
            s.unit_transforms = (transforms).into();
            s.unit_groups = (groups).into();
            let ops = expected_vector(&s, &store, &catalog, VectorPlacement::VectorParity, "vector");
            let recs = &ops.ops[0].records;
            assert_eq!(recs.len(), 5);
            // Where the shader draws each glyph's em-box centre: C + off.
            recs.iter()
                .map(|r| {
                    let k = r.font_size_px as f64;
                    (
                        r.pen_x as f64 + (r.em_box[0] + r.em_box[2]) as f64 * 0.5 * k + r.glyph_off[0] as f64,
                        r.baseline_y as f64 + (r.em_box[1] + r.em_box[3]) as f64 * 0.5 * k + r.glyph_off[1] as f64,
                        r.glyph_lin,
                    )
                })
                .collect::<Vec<_>>()
        };
        let span = |g: &[(f64, f64, [f32; 4])]| (g[4].0 - g[0].0).hypot(g[4].1 - g[0].1);
        let half = GlyphXform::resolve(0.0, 0.0, 0.5, 0.0, 1.0);
        let rest = placed(Vec::new(), Vec::new());
        let word = placed(vec![half; 5], vec![0; 5]);
        let chars = placed(vec![half; 5], Vec::new());
        let ratio = span(&word) / span(&rest);
        assert!((ratio - 0.5).abs() < 0.05, "the word gathers to about half its span: {ratio}");
        // Each glyph is still drawn at half size.
        for g in &word {
            let det = g.2[0] as f64 * g.2[3] as f64 - g.2[1] as f64 * g.2[2] as f64;
            assert!((det - 0.25).abs() < 1e-3, "{det}");
        }
        // About its centre: the middle glyph barely moves.
        assert!((word[2].0 - rest[2].0).hypot(word[2].1 - rest[2].1) < 3.0);
        // Per-character units keep their places on the path, each shrinking about itself.
        for (c, r) in chars.iter().zip(&rest) {
            assert!((c.0 - r.0).abs() < 1e-3 && (c.1 - r.1).abs() < 1e-3);
        }
        // A word turned 90 degrees stands across the path: its glyphs line up along the normal at its centre.
        let turned = placed(vec![GlyphXform::resolve(0.0, 0.0, 1.0, 90.0, 1.0); 5], vec![0; 5]);
        let along = (rest[4].0 - rest[0].0, rest[4].1 - rest[0].1);
        let across = (turned[4].0 - turned[0].0, turned[4].1 - turned[0].1);
        let cos = (along.0 * across.0 + along.1 * across.1) / (along.0.hypot(along.1) * across.0.hypot(across.1));
        assert!(cos.abs() < 0.2, "{cos}");
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn opacity_zero_glyph_contributes_zero_on_fill_and_every_effect_pass() {
        let (store, catalog) = store_and_catalog();
        let mut s = spec("<p><span style=\"font-family: Inter; font-size: 64px\">AB</span></p>");
        // Glow (an effect pass that re-emits glyph instances) + a glyph at opacity 0.
        s.effects = vec![crate::ops::TextEffectSpec::Glow {
            color: "#ffffff".into(),
            radius_px: Some(8.0),
            intensity: Some(1.0),
            opacity: Some(1.0),
        }];
        s.unit_transforms = (vec![
            GlyphXform::IDENTITY,
            GlyphXform::resolve(0.0, 0.0, 1.0, 0.0, 0.0),
        ]).into();
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector",
        );
        // Glow pass + fill pass both carry analytic glyph records. The opacity-0
        // glyph ('B', record index 1) must read opacity 0 on EVERY pass, so it
        // emits zero coverage everywhere; 'A' stays full.
        let mut analytic_passes = 0;
        for op in &ops.ops {
            if op.records.is_empty() {
                continue;
            }
            assert_eq!(op.records.len(), 2);
            assert_eq!(op.records[0].glyph_opacity, 1.0);
            assert_eq!(
                op.records[1].glyph_opacity, 0.0,
                "opacity-0 glyph must be 0 on every pass"
            );
            analytic_passes += 1;
        }
        assert!(
            analytic_passes >= 2,
            "expected a glow pass AND a fill pass, got {analytic_passes}"
        );
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn unit_index_maps_across_words_skipping_space() {
        let (store, catalog) = store_and_catalog();
        let mut s = spec("<p><span style=\"font-family: Inter; font-size: 32px\">AB CD</span></p>");
        // canonical "AB CD" (5 code points): 0=A 1=B 2=space 3=C 4=D. Tag each
        // code point with offsetY = index*10 so we can read back the mapping.
        s.unit_transforms = ((0..5)
            .map(|i| GlyphXform::resolve(0.0, i as f64 * 10.0, 1.0, 0.0, 1.0))
            .collect::<Vec<_>>()).into();
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector",
        );
        // Drawn glyphs A,B,C,D (the space has no outline → no record). C/D must map
        // to canonical indices 3/4 — proof the inter-word space is accounted for,
        // not collapsed into a shifted index.
        let recs = &ops.ops[0].records;
        assert_eq!(recs.len(), 4);
        assert_eq!(recs[0].glyph_off[1], 0.0); // A → idx 0
        assert_eq!(recs[1].glyph_off[1], 10.0); // B → idx 1
        assert_eq!(recs[2].glyph_off[1], 30.0); // C → idx 3 (idx 2 is the space)
        assert_eq!(recs[3].glyph_off[1], 40.0); // D → idx 4
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn canvas2d_hb_clusters_are_codepoints_not_bytes() {
        // The Canvas2D HarfBuzz fallback (reached via the Hybrid VectorParity
        // decline) must ALSO report code-point clusters, not raw rustybuzz BYTE
        // offsets, including its per-run bidi rebasing (`run.start + cluster`).
        // "aéb": é is 2 UTF-8 bytes, so 'b' is byte 3 but code point 2.
        let (store, catalog) = store_and_catalog();
        let font = store
            .font_for_variant(&catalog, "Lato", 400, "normal", None)
            .unwrap();
        let hb = store
            .hb_face_for_variant(&catalog, "Lato", 400, "normal", None)
            .expect("Lato HB face");
        let glyphs = crate::vector::hb::shape_canvas2d_hb(&hb, &font, "aéb").unwrap();
        let clusters: Vec<usize> = glyphs.iter().map(|g| g.cluster).collect();
        assert_eq!(
            clusters,
            vec![0, 1, 2],
            "Canvas2D HB clusters must be code-point indices, not byte offsets"
        );
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn unit_index_otjs_maps_each_glyph_to_source_codepoint() {
        let (store, catalog) = store_and_catalog();
        // Inter shapes "office" through the otjs path one-glyph-per-code-point
        // (no ligature in the vendored subset). Each glyph's cluster must be its
        // SOURCE CODE-POINT index — the contract `string_to_glyphs_clustered`
        // guarantees (here it coincides 0..5; the ligature shift is covered below).
        let mut s =
            spec("<p><span style=\"font-family: Inter; font-size: 64px\">office</span></p>");
        s.unit_transforms = ((0..6)
            .map(|i| GlyphXform::resolve(0.0, i as f64 * 100.0, 1.0, 0.0, 1.0))
            .collect::<Vec<_>>()).into();
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector",
        );
        let offs: Vec<f32> = ops.ops[0].records.iter().map(|r| r.glyph_off[1]).collect();
        assert_eq!(offs, vec![0.0, 100.0, 200.0, 300.0, 400.0, 500.0]);
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn vector_hb_clusters_are_codepoints_not_bytes() {
        // The vector HarfBuzz reroute must report CODE-POINT clusters, not the
        // raw rustybuzz BYTE offsets — otherwise a kinetic node with non-ASCII
        // text on an HB-rerouted font (Lato) mis-maps every glyph after a
        // multi-byte code point. "aéb": é is 2 UTF-8 bytes, so 'b' is at byte 3
        // but code point 2.
        let (store, catalog) = store_and_catalog();
        let font = store
            .font_for_variant(&catalog, "Lato", 400, "normal", None)
            .unwrap();
        let glyphs = crate::vector::measure::shape_vector(
            &store, &catalog, "Lato", 400, "normal", None, &font, "aéb",
        )
        .unwrap();
        let clusters: Vec<usize> = glyphs.iter().map(|g| g.cluster).collect();
        assert_eq!(
            clusters,
            vec![0, 1, 2],
            "HB clusters must be code-point indices, not byte offsets"
        );
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn unit_index_maps_ligature_to_first_codepoint() {
        let (store, catalog) = store_and_catalog();
        // Lato reroutes to HarfBuzz on the vector lane (its liga is otjs-incompatible)
        // and DOES ligate "office" → fewer glyphs. The ligature survivor
        // takes its first code point's transform, and EVERY glyph after it keeps its
        // true code-point index (c → 4, e → 5) — NOT shifted by the deleted
        // component count. This is the cross-lane "ligature → first code point"
        // contract; a shift bug would map the tail to 2/3 here.
        let mut s = spec("<p><span style=\"font-family: Lato; font-size: 64px\">office</span></p>");
        s.unit_transforms = ((0..6)
            .map(|i| GlyphXform::resolve(0.0, i as f64 * 100.0, 1.0, 0.0, 1.0))
            .collect::<Vec<_>>()).into();
        let ops = expected_vector(
            &s,
            &store,
            &catalog,
            VectorPlacement::VectorParity,
            "expected vector",
        );
        let recs = &ops.ops[0].records;
        let offs: Vec<f32> = recs.iter().map(|r| r.glyph_off[1]).collect();
        assert!(
            recs.len() < 6,
            "Lato 'office' must ligate (got {} glyphs)",
            recs.len()
        );
        // The trailing glyphs keep their true code points regardless of how many
        // components ligated: c → 4, e → 5 (never 2/3).
        assert_eq!(
            *offs.last().unwrap(),
            500.0,
            "e must map to code point 5: {offs:?}"
        );
        assert_eq!(
            offs[offs.len() - 2],
            400.0,
            "c must map to code point 4: {offs:?}"
        );
        assert_eq!(offs[0], 0.0, "o must map to code point 0: {offs:?}");
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn extrude_skips_zero_opacity_glyph() {
        let (store, catalog) = store_and_catalog();
        let wall_tris = |units: Vec<GlyphXform>| -> usize {
            let mut s =
                spec("<p><span style=\"font-family: Inter; font-size: 64px\">AB</span></p>");
            s.effects = vec![crate::ops::TextEffectSpec::Extrude {
                depth_px: Some(10.0),
                angle_deg: Some(45.0),
                color: "#000000".into(),
            }];
            s.unit_transforms = (units).into();
            let ops = expected_vector(
                &s,
                &store,
                &catalog,
                VectorPlacement::VectorParity,
                "expected vector",
            );
            ops.ops
                .iter()
                .find(|o| o.pipeline == VectorOpPipeline::Tess && !o.tris.is_empty())
                .map_or(0, |o| o.tris.len())
        };
        let full = wall_tris(vec![GlyphXform::IDENTITY, GlyphXform::IDENTITY]);
        let one_zero = wall_tris(vec![
            GlyphXform::IDENTITY,
            GlyphXform::resolve(0.0, 0.0, 1.0, 0.0, 0.0),
        ]);
        assert!(full > 0, "both glyphs should sweep walls");
        assert!(
            one_zero > 0 && one_zero < full,
            "the opacity-0 glyph's extrude walls must be skipped: {one_zero} vs {full}"
        );
        // Documented limitation (nodes.ts GlyphTransform): extrude honors opacity
        // as BINARY visibility — a partially-faded glyph (0.5) still emits its FULL
        // wall geometry (the Tess pipeline has no per-vertex alpha), identical to
        // the fully-opaque case. Pin it so the contract and behavior stay aligned.
        let one_half = wall_tris(vec![
            GlyphXform::IDENTITY,
            GlyphXform::resolve(0.0, 0.0, 1.0, 0.0, 0.5),
        ]);
        assert_eq!(
            one_half, full,
            "partial opacity must still emit full extrude walls (binary visibility)"
        );
    }
}
