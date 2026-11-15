//! pattern.rs — pattern (image) fill instance packing (M3b).
//!
//! Ports the pattern branch of `WebGPUPathRenderer` instance packing
//! (`renderers/webgpu/rendering/systems/shapes/WebGPUPathRenderer.ts:1232-1505`),
//! `encodePatternRepeatMode` (`:194-209`), `selectPatternTexture` (`:220-231`),
//! and the perspective metadata builder
//! (`renderers/webgpu/utils/shapeToPathNodeUtils.ts:227-247`). The WGSL
//! consumer is the extracted `wgsl/path-pattern.wgsl` `PathPatternData`
//! struct (108 floats / 432 bytes) — the image adjustment / colorChange / overlay /
//! inverse-homography math lives in that shader; this module only packs data.
//!
//! COPY THE BUGS:
//! - `fillPatternScale?.x || 1` — JS falsy coercion: scale 0 renders as 1,
//! - overlay color parse failure → default RED [1,0,0,1],
//! - colorChange source/target colors are packed from the node arrays even
//!   when the feature is disabled (only the count is gated),
//! - placeholder fills force repeat 0 and disable the adjustments/overlay/
//!   colorChange/perspective (Q-22),
//! - perspective rows pack the flag in row0.w (1 = enabled); disabled rows
//!   are the IDENTITY rows [1,0,0,0]/[0,1,0,0]/[0,0,1,0].

use bytemuck::{Pod, Zeroable};
use moda_scene::paint::{ImageAdjust, ImagePlacement, Paint};
use moda_scene::types::RawNode;
use moda_scene::value::Value;

use super::homography::{default_quad, Point2D, ProjectiveTransform, Quad};
use super::image_source::PatternTextureState;
use crate::color::{parse_color, Rgba};

/// `BLEND_MODE_DISABLED` (`renderers/webgpu/constants/blendModes.ts:73`).
pub const BLEND_MODE_DISABLED: f32 = -1.0;
/// `SHAPE_DEFAULTS.pattern.defaultOverlayOpacity` (`core/config/shapeDefaults.ts:168`).
pub const DEFAULT_OVERLAY_OPACITY: f64 = 1.0;
/// `PATH_RENDERER_DEFAULTS.DEFAULT_PATTERN_BOUNDS` (`WebGPUConstants.ts:165`).
pub const DEFAULT_PATTERN_BOUNDS: (f32, f32) = (200.0, 200.0);

// ---------------------------------------------------------------------------
// GPU instance ABI — `PathPatternData` (wgsl/path-pattern.wgsl)
// ---------------------------------------------------------------------------

/// The 432-byte `PathPatternData` storage-buffer instance (linear layout —
/// every vec2/vec4/vec3 lands on its natural WGSL alignment, no implicit
/// padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PatternInstance {
    /// floats 0..16.
    pub model_matrix: [f32; 16],
    /// floats 16..20 — white tint; `.a` = node opacity.
    pub fill_color: [f32; 4],
    /// floats 20..24 — solid stroke color (a × opacity); zeroed when a
    /// gradient-stroke overlay owns the stroke.
    pub stroke_color: [f32; 4],
    /// floats 24..28 — [vertexOffset, vertexCount, strokeWidth ‖ 1, depth].
    pub path_info: [f32; 4],
    /// floats 28..32 — [repeatMode, scaleX, scaleY, rotation(rad)].
    pub pattern_data: [f32; 4],
    /// floats 32..34 — bound texture dimensions in px.
    pub texture_dims: [f32; 2],
    /// floats 34..36 — `fillPatternOffset` (node-local units).
    pub pattern_offset: [f32; 2],
    /// floats 36..40 — [0, 0, width, height] shape bounds for UV.
    pub bounds: [f32; 4],
    /// floats 40..44.
    pub page_clip_bounds: [f32; 4],
    /// floats 44..48 — image adjustments (E7.18, [`image_adjust`]):
    /// [exposure, contrast, saturation, temperature], each −1..1.
    pub adjust_a: [f32; 4],
    /// floats 48..52.
    pub overlay_color: [f32; 4],
    /// floats 52..54 — [opacity, blendMode (−1 = disabled)].
    pub overlay_settings: [f32; 2],
    /// floats 54..56 — image adjustments [tint, highlights].
    pub adjust_b: [f32; 2],
    /// floats 56..72.
    pub color_change_source: [[f32; 4]; 4],
    /// floats 72..88.
    pub color_change_target: [[f32; 4]; 4],
    /// floats 88..91 — [count, tolerance, smoothness] (vec3, 16-aligned).
    pub color_change_params: [f32; 3],
    /// float 91 — image adjustment: shadows.
    pub adjust_shadows: f32,
    /// floats 92..96 — inverse homography row 0; .w = enabled flag.
    pub perspective_row0: [f32; 4],
    /// floats 96..100.
    pub perspective_row1: [f32; 4],
    /// floats 100..104.
    pub perspective_row2: [f32; 4],
    /// [enabled, hue turns, saturation fraction, lightness fraction].
    pub colorize: [f32; 4],
}

// Keep the diagnostic plan representation stable for existing paints. The appended
// GPU ABI slot is zero when Colorize is disabled; include it only when it affects
// rendering, so existing-document digests still detect changes to active paint.
impl std::fmt::Debug for PatternInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("PatternInstance");
        d.field("model_matrix", &self.model_matrix);
        d.field("fill_color", &self.fill_color);
        d.field("stroke_color", &self.stroke_color);
        d.field("path_info", &self.path_info);
        d.field("pattern_data", &self.pattern_data);
        d.field("texture_dims", &self.texture_dims);
        d.field("pattern_offset", &self.pattern_offset);
        d.field("bounds", &self.bounds);
        d.field("page_clip_bounds", &self.page_clip_bounds);
        d.field("adjust_a", &self.adjust_a);
        d.field("overlay_color", &self.overlay_color);
        d.field("overlay_settings", &self.overlay_settings);
        d.field("adjust_b", &self.adjust_b);
        d.field("color_change_source", &self.color_change_source);
        d.field("color_change_target", &self.color_change_target);
        d.field("color_change_params", &self.color_change_params);
        d.field("adjust_shadows", &self.adjust_shadows);
        d.field("perspective_row0", &self.perspective_row0);
        d.field("perspective_row1", &self.perspective_row1);
        d.field("perspective_row2", &self.perspective_row2);
        if self.colorize != [0.0; 4] {
            d.field("colorize", &self.colorize);
        }
        d.finish()
    }
}

const _: () = assert!(core::mem::size_of::<PatternInstance>() == 432);

// ---------------------------------------------------------------------------
// Field helpers
// ---------------------------------------------------------------------------

/// `encodePatternRepeatMode` (`WebGPUPathRenderer.ts:194-209`).
pub fn encode_pattern_repeat_mode(repeat: Option<&str>, force_repeat: bool) -> f32 {
    if force_repeat {
        return 0.0;
    }
    match repeat {
        Some("repeat-x") => 0.25,
        Some("repeat-y") => 0.5,
        Some("no-repeat") => 0.75,
        _ => 0.0,
    }
}

/// `getBlendModeValue` (`renderers/webgpu/constants/blendModes.ts:49-61`).
fn blend_mode_value(mode: Option<&str>) -> f32 {
    match mode {
        Some("multiply") => 1.0,
        Some("screen") => 2.0,
        Some("overlay") => 3.0,
        _ => 0.0,
    }
}

/// Figma's image adjustments (E7.18) on one straight sRGB-encoded colour, each
/// amount −1..1 (0 = unchanged): the reference `wgsl/path-pattern.wgsl`'s
/// `applyImageAdjust` implements (in f32) and the pixel tests check against.
///
/// 1. Exposure and white balance, in linear light: × 2^(2·exposure) (±2
///    stops); temperature warms (red × 1 + 0.2t, blue × 1 − 0.2t); tint
///    moves toward magenta (green × 1 − 0.2·tint).
/// 2. Contrast about mid grey: (c − ½)(1 + contrast) + ½.
/// 3. Highlights and shadows by luma L (Rec. 709): + ¼(highlights ·
///    smoothstep(½, 1, L) + shadows · (1 − smoothstep(0, ½, L))).
/// 4. Saturation: mix(L, c, 1 + saturation).
///
/// Every step clamps to 0..1.
pub fn image_adjust(rgb: [f64; 3], a: &ImageAdjustAmounts) -> [f64; 3] {
    fn to_lin(c: f64) -> f64 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }
    fn to_srgb(l: f64) -> f64 {
        if l <= 0.0031308 {
            l * 12.92
        } else {
            1.055 * l.powf(1.0 / 2.4) - 0.055
        }
    }
    fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
    let luma = |c: [f64; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let gain = 2f64.powf(2.0 * a.exposure);
    let wb = [
        1.0 + 0.2 * a.temperature,
        1.0 - 0.2 * a.tint,
        1.0 - 0.2 * a.temperature,
    ];
    let mut c = [0.0; 3];
    for k in 0..3 {
        c[k] = to_srgb((to_lin(rgb[k]) * gain * wb[k]).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        c[k] = ((c[k] - 0.5) * (1.0 + a.contrast) + 0.5).clamp(0.0, 1.0);
    }
    let l = luma(c);
    let tone = 0.25
        * (a.highlights * smoothstep(0.5, 1.0, l) + a.shadows * (1.0 - smoothstep(0.0, 0.5, l)));
    for v in &mut c {
        *v = (*v + tone).clamp(0.0, 1.0);
    }
    let l = luma(c);
    c.map(|v| (l + (v - l) * (1.0 + a.saturation)).clamp(0.0, 1.0))
}

/// The seven amounts of [`image_adjust`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ImageAdjustAmounts {
    pub exposure: f64,
    pub contrast: f64,
    pub saturation: f64,
    pub temperature: f64,
    pub tint: f64,
    pub highlights: f64,
    pub shadows: f64,
}

/// `fillPatternScale?.x || 1` (JS `||`: 0/NaN → 1).
fn one_if_falsy(v: f64) -> f64 {
    if v == 0.0 || v.is_nan() {
        1.0
    } else {
        v
    }
}

/// The placement and colour treatments of the node's top image or video fill
/// (`moda_scene::paint::top_fill`). Anything else packs the defaults.
pub fn image_fill_parts(node: &RawNode) -> (ImagePlacement, ImageAdjust) {
    match moda_scene::paint::top_fill(node).map(|f| f.paint) {
        Some(Paint::Image(p)) => (p.placement, p.adjust.unwrap_or_default()),
        Some(Paint::Video(p)) => (p.placement, p.adjust.unwrap_or_default()),
        _ => Default::default(),
    }
}

// ---------------------------------------------------------------------------
// Pattern-warp view (UV inverse mapping + warped-quad geometry metadata).
// Fed exclusively by the canonical `distort` reader below via
// `NodeDistort::as_pattern_warp`.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PerspectivePattern {
    pub inverse_homography: ProjectiveTransform,
    pub source_width: f64,
    pub source_height: f64,
    /// The warped destination quad (the image of the node's w×h box).
    pub corners: Quad,
}

// ---------------------------------------------------------------------------
// Distort metadata (the canonical `distort` field — `getNodeDistort`
// + `getNodeDistortLocalTransform`, `core/scene/distortGeometry.ts`).
// ---------------------------------------------------------------------------

/// Node types that can carry a `distort` warp
/// (`DISTORT_CAPABLE_NODE_TYPES`, `core/scene/distortGeometry.ts`).
pub const DISTORT_CAPABLE_TYPES: [&str; 5] =
    ["rectangle", "ellipse", "star", "polygon", "richtext"];

/// Render metadata for one node's `distort` warp.
#[derive(Debug, Clone)]
pub struct NodeDistort {
    /// Unwarped local px → warped local px (geometry pre-warps through this).
    pub local: ProjectiveTransform,
    /// Warped local px → source local px (the pattern shader's UV inverse map).
    pub inverse_homography: ProjectiveTransform,
    pub source_width: f64,
    pub source_height: f64,
    /// The warped corners of the node's w×h box in local px (TL/TR/BR/BL).
    pub corners: Quad,
}

impl NodeDistort {
    /// View this warp as the legacy pattern-metadata shape so the existing
    /// `PatternInstanceContext.perspective` packing (inverse-homography UV
    /// rows) serves both the legacy corner-pin path and the distort path
    /// without a second ABI.
    pub fn as_pattern_warp(&self) -> PerspectivePattern {
        PerspectivePattern {
            inverse_homography: self.inverse_homography,
            source_width: self.source_width,
            source_height: self.source_height,
            corners: self.corners,
        }
    }
}

/// Port of `getNodeDistort` + `getNodeDistortLocalTransform`
/// (`core/scene/distortGeometry.ts`): the stored form is
/// `{ type: 'matrix', matrix: [9 numbers], source? }` — a normalized
/// unit-square homography (`source` is advisory authoring state; the matrix is
/// render truth and the only thing read here). Applies to ANY fill: geometry
/// warps on the CPU, pattern fills additionally inverse-map UVs.
pub fn node_distort(node: &RawNode) -> Option<NodeDistort> {
    if !DISTORT_CAPABLE_TYPES.contains(&node.node_type()) {
        return None;
    }
    let distort = node.object("distort")?;
    if distort.get("type").and_then(Value::as_str) != Some("matrix") {
        return None;
    }
    // `coerceDistortMatrix`: exactly 9 finite numbers.
    let raw = distort.get("matrix")?.as_array()?;
    if raw.len() != 9 {
        return None;
    }
    let mut matrix = [0.0f64; 9];
    for (i, v) in raw.iter().enumerate() {
        matrix[i] = v.as_f64().filter(|f| f.is_finite())?;
    }
    // Validate the STORED matrix as-is — no pre-validation re-normalization —
    // mirroring `getNodeDistort` (readers tolerate a positively-scaled
    // representative; writers normalize).
    if !super::homography::is_valid_distort_matrix(&matrix) {
        return None;
    }
    // Post-validation h33 normalization (renderer-side hardening, not a TS
    // acceptance change): homographies are scale-invariant, but the GPU lanes
    // are not — the pattern shader gates on an ABSOLUTE denominator epsilon
    // and the vector-text rows are f32 — so a positively-scaled valid
    // representative could discard fragments or overflow while the f64
    // geometry warp sails through. Validation guarantees `matrix[8] > EPSILON`
    // (the (0,0) corner denominator). Stored matrices already have h33 == 1,
    // so the common path skips the division entirely (bit-exact).
    let h33 = matrix[8];
    if h33 != 1.0 {
        for entry in &mut matrix {
            *entry /= h33;
        }
    }
    // An (approximately) IDENTITY matrix is "no distortion", not a warp to
    // render: without this gate a visually no-op persisted `distort` — a
    // reset-to-zero gesture, an animation at its rest pose, an agent writing
    // identity instead of removing the field — would still flatten curves to
    // polylines and disable the SDF fast paths, drop shadows, and background
    // blur. Declining here (the single reader every consumer gates on) keeps
    // identity-distort nodes byte-identical to distort-free ones. Epsilon
    // mirrors `areDistortMatricesApproximatelyEqual`'s default (1e-9): a
    // sub-1e-9 lean is far below any renderable difference. The TS
    // bounds/hit-test wiring (node-distort plan PR 3) must mirror this gate.
    if super::homography::are_distort_matrices_approximately_equal(
        &matrix,
        &super::homography::IDENTITY_DISTORT_MATRIX,
        1e-9,
    ) {
        return None;
    }
    let width = node.f64("width").unwrap_or(0.0);
    let height = node.f64("height").unwrap_or(0.0);
    // `getNodeDistortLocalTransform` re-guards dims; NaN is rejected by the
    // `!(w > 0)` shape of the check on both sides.
    let local = super::homography::unit_distort_local_transform(&matrix, width, height)?;
    let source = default_quad(width, height);
    let mut corners = [Point2D { x: 0.0, y: 0.0 }; 4];
    for (i, c) in source.iter().enumerate() {
        corners[i] = super::homography::apply_projective_to_point(*c, &local)?;
    }
    // No local-px convexity re-check: the TS render path gates only on the
    // unit-space validation above. Affine scaling preserves convexity, so the
    // check would be redundant — and its cross-product epsilon is
    // scale-dependent (a sub-pixel node could fail here while TS warps it).
    // (Conversely, this reader ALSO requires the inversion + corner
    // projections below to succeed before any warp applies, which the TS
    // reader does not. The gap is constructible only by a hand-authored
    // stored matrix with a huge positive h33 scale — every writer in the
    // codebase h33-normalizes before validating, so no produced document
    // can reach it.)
    let inverse = super::homography::invert_projective(&local)?;
    Some(NodeDistort {
        local,
        inverse_homography: inverse,
        source_width: width,
        source_height: height,
        corners,
    })
}

// ---------------------------------------------------------------------------
// Instance packing
// ---------------------------------------------------------------------------

/// Inputs the frame walk provides (everything else comes off the node).
pub struct PatternInstanceContext<'a> {
    pub node: &'a RawNode,
    pub model_matrix: [f32; 16],
    pub depth: f32,
    pub page_clip_bounds: [f32; 4],
    /// Tessellated bounds size (maxX−minX, maxY−minY) — pattern UV `bounds`
    /// uses [0, 0, w, h] (`WebGPUPathRenderer.ts:1437` comment "Shape bounds
    /// for UV calculation"); perspective patterns use source dims instead.
    pub tess_bounds_size: Option<(f32, f32)>,
    pub texture_state: PatternTextureState,
    pub texture_dims: (u32, u32),
    pub perspective: Option<&'a PerspectivePattern>,
    /// `hasVisibleGradientStroke(node)` — zero the solid stroke color so the
    /// gradient overlay instance owns the stroke.
    pub gradient_stroke_overlay: bool,
    /// The box the image fill sizes to, node-local `[x, y, w, h]`: the node's
    /// `width` × `height`, or a path's geometric bounds.
    pub image_box: [f64; 4],
    /// M3c (interface-change note): the texture's uploaded/natural ratio —
    /// `fillPatternScale` divides by this so SVG/small-raster upscaled
    /// textures keep their canvas-unit size
    /// (`WebGPUPathRenderer.ts:1338-1345`). (1, 1) for unscaled textures.
    pub texture_scale_factor: (f64, f64),
}

/// Pack one `PathPatternData` instance
/// (`WebGPUPathRenderer.ts:1326-1474` pattern branch).
/// `path_info[0..2]` (vertexOffset/vertexCount) are fixed up at upload.
pub fn build_pattern_instance(ctx: &PatternInstanceContext) -> PatternInstance {
    let node = ctx.node;
    let opacity = node.f64("opacity").unwrap_or(1.0) as f32;
    let is_placeholder = ctx.texture_state == PatternTextureState::Loading
        || ctx.texture_state == PatternTextureState::Failed;

    let (placement, adjust) = image_fill_parts(node);

    // Stroke: solid stroke color, or zeroed for a gradient-stroke overlay.
    let stroke = if ctx.gradient_stroke_overlay {
        Rgba {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        }
    } else {
        super::gradient::solid_stroke_color(
            super::gradient::top_stroke(node).as_ref().map(|s| &s.paint),
        )
    };
    let stroke_width = super::gradient::instance_stroke_width(node);

    // Bounds for UV: perspective source dims > tessellated bounds > 200×200.
    let (width, height) = if let Some(p) = ctx.perspective {
        (p.source_width as f32, p.source_height as f32)
    } else {
        ctx.tess_bounds_size.unwrap_or(DEFAULT_PATTERN_BOUNDS)
    };

    // The scale mode resolved against the box (Fill / Fit re-fit live; Crop
    // stretches with the box). The natural size falls back to the decoded
    // texture's (uploaded dims ÷ upscale factor).
    let natural = (!is_placeholder && ctx.texture_dims.0 > 1 && ctx.texture_dims.1 > 1).then(|| {
        (
            ctx.texture_dims.0 as f64 / ctx.texture_scale_factor.0,
            ctx.texture_dims.1 as f64 / ctx.texture_scale_factor.1,
        )
    });
    let resolved = placement.resolve(ctx.image_box, natural);

    // Pattern scale ÷ texture scale factor (M3c: SVG/small-raster textures
    // upload upscaled; compensate exactly like the production renderer —
    // `WebGPUPathRenderer.ts:1328-1345`).
    let scale_x = one_if_falsy(resolved.scale.0) / ctx.texture_scale_factor.0;
    let scale_y = one_if_falsy(resolved.scale.1) / ctx.texture_scale_factor.1;

    // Image adjustments (E7.18). A placeholder draws unadjusted.
    let adj = |v: Option<f64>| {
        if is_placeholder {
            0.0
        } else {
            v.filter(|v| v.is_finite()).unwrap_or(0.0).clamp(-1.0, 1.0) as f32
        }
    };
    let adjust_a = [
        adj(adjust.exposure),
        adj(adjust.contrast),
        adj(adjust.saturation),
        adj(adjust.temperature),
    ];
    let adjust_b = [adj(adjust.tint), adj(adjust.highlights)];
    let adjust_shadows = adj(adjust.shadows);

    // Overlay.
    let overlay = adjust.overlay.unwrap_or_default();
    let overlay_color = parse_color(overlay.color.as_ref().and_then(Value::as_str))
        .map(|c| [c.r as f32, c.g as f32, c.b as f32, c.a as f32])
        .unwrap_or([1.0, 0.0, 0.0, 1.0]);
    let overlay_enabled = !is_placeholder && overlay.enabled == Some(true);
    let overlay_settings = [
        overlay.opacity.unwrap_or(DEFAULT_OVERLAY_OPACITY) as f32,
        if overlay_enabled {
            blend_mode_value(overlay.blend_mode.as_deref())
        } else {
            BLEND_MODE_DISABLED
        },
    ];

    // Color change (palette recolor). Colors pack regardless of `enabled`.
    let recolor = adjust.recolor.unwrap_or_default();
    let extracted = recolor.from.as_deref();
    let replacements = recolor.to.as_deref();
    let extracted_len = extracted.map(|a| a.len()).unwrap_or(0);
    let replacements_len = replacements.map(|a| a.len()).unwrap_or(0);
    let enabled = !is_placeholder && recolor.enabled == Some(true);
    let color_count = if enabled {
        extracted_len.min(replacements_len).min(4)
    } else {
        0
    };
    let mut source_colors = [[0.0f32, 0.0, 0.0, 1.0]; 4];
    let mut target_colors = [[0.0f32, 0.0, 0.0, 1.0]; 4];
    for i in 0..4 {
        if let Some(s) = extracted
            .and_then(|a| a.get(i))
            .and_then(|s| parse_color(Some(s)))
        {
            source_colors[i] = [s.r as f32, s.g as f32, s.b as f32, s.a as f32];
        }
        if let Some(t) = replacements
            .and_then(|a| a.get(i))
            .and_then(|s| parse_color(Some(s)))
        {
            target_colors[i] = [t.r as f32, t.g as f32, t.b as f32, t.a as f32];
        }
    }
    let color_change_params = [
        color_count as f32,
        recolor.tolerance.unwrap_or(0.3).clamp(0.0, 1.0) as f32,
        recolor.smoothness.unwrap_or(0.6).clamp(0.0, 1.0) as f32,
    ];

    // Perspective rows (disabled for placeholders so the shader never
    // discards on a near-zero denominator).
    let h = if is_placeholder {
        None
    } else {
        ctx.perspective.map(|p| p.inverse_homography)
    };
    let (row0, row1, row2) = match h {
        Some(h) => (
            [h.h11 as f32, h.h12 as f32, h.h13 as f32, 1.0],
            [h.h21 as f32, h.h22 as f32, h.h23 as f32, 0.0],
            [h.h31 as f32, h.h32 as f32, h.h33 as f32, 0.0],
        ),
        None => (
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ),
    };

    let pattern_offset = resolved.offset;

    PatternInstance {
        model_matrix: ctx.model_matrix,
        fill_color: [1.0, 1.0, 1.0, opacity],
        stroke_color: [
            stroke.r as f32,
            stroke.g as f32,
            stroke.b as f32,
            stroke.a as f32 * opacity,
        ],
        path_info: [0.0, 0.0, stroke_width, ctx.depth],
        pattern_data: [
            encode_pattern_repeat_mode(Some(resolved.repeat), is_placeholder),
            scale_x as f32,
            scale_y as f32,
            resolved.rotation.to_radians() as f32,
        ],
        texture_dims: [ctx.texture_dims.0 as f32, ctx.texture_dims.1 as f32],
        pattern_offset: [pattern_offset.0 as f32, pattern_offset.1 as f32],
        bounds: [0.0, 0.0, width, height],
        page_clip_bounds: ctx.page_clip_bounds,
        adjust_a,
        overlay_color,
        overlay_settings,
        adjust_b,
        color_change_source: source_colors,
        color_change_target: target_colors,
        color_change_params,
        adjust_shadows,
        perspective_row0: row0,
        perspective_row1: row1,
        perspective_row2: row2,
        colorize: adjust.colorize.filter(|c| c.enabled && !is_placeholder).map_or([0.0; 4], |c| {
            let finite = |v: f64| if v.is_finite() { v } else { 0.0 };
            [1.0, (finite(c.hue).clamp(0.0, 360.0) / 360.0) as f32,
             (finite(c.saturation).clamp(0.0, 100.0) / 100.0) as f32,
             (finite(c.lightness).clamp(-100.0, 100.0) / 100.0) as f32]
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    fn raw(v: moda_scene::value::Value) -> RawNode {
        RawNode::from_value(v).unwrap()
    }

    fn ctx<'a>(
        node: &'a RawNode,
        state: PatternTextureState,
        perspective: Option<&'a PerspectivePattern>,
    ) -> PatternInstanceContext<'a> {
        PatternInstanceContext {
            node,
            model_matrix: [0.0; 16],
            depth: 0.5,
            page_clip_bounds: [0.0, 0.0, 100.0, 100.0],
            tess_bounds_size: Some((260.0, 200.0)),
            texture_state: state,
            texture_dims: (16, 16),
            perspective,
            gradient_stroke_overlay: false,
            image_box: [0.0, 0.0, 260.0, 200.0],
            texture_scale_factor: (1.0, 1.0),
        }
    }

    #[test]
    fn plan_debug_preserves_disabled_colorize_and_records_active_colorize() {
        let mut instance = PatternInstance::zeroed();
        let before = format!("{instance:?}");
        assert!(!before.contains("colorize"));
        instance.colorize = [1.0, 0.25, 0.5, 0.0];
        assert!(format!("{instance:?}").contains("colorize: [1.0, 0.25, 0.5, 0.0]"));
        instance.colorize = [0.0; 4];
        assert_eq!(format!("{instance:?}"), before);
    }

    #[test]
    fn repeat_mode_encoding() {
        assert_eq!(encode_pattern_repeat_mode(Some("repeat"), false), 0.0);
        assert_eq!(encode_pattern_repeat_mode(Some("repeat-x"), false), 0.25);
        assert_eq!(encode_pattern_repeat_mode(Some("repeat-y"), false), 0.5);
        assert_eq!(encode_pattern_repeat_mode(Some("no-repeat"), false), 0.75);
        assert_eq!(encode_pattern_repeat_mode(None, false), 0.0);
        // Placeholder force-tiles even no-repeat.
        assert_eq!(encode_pattern_repeat_mode(Some("no-repeat"), true), 0.0);
    }

    #[test]
    fn image_adjustments_pack_clamped_and_placeholders_drop_them() {
        let node = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "placement": {}, "adjust": {
                "exposure": 0.5, "contrast": -0.25, "saturation": 2.0, "temperature": 0.1,
                "tint": -0.2, "highlights": 0.3, "shadows": -0.4}}}]}));
        let inst = build_pattern_instance(&ctx(&node, PatternTextureState::Loaded, None));
        assert_eq!(inst.adjust_a, [0.5, -0.25, 1.0, 0.1]);
        assert_eq!(inst.adjust_b, [-0.2, 0.3]);
        assert_eq!(inst.adjust_shadows, -0.4);

        // Placeholders draw unadjusted.
        let inst = build_pattern_instance(&ctx(&node, PatternTextureState::Failed, None));
        assert_eq!(inst.adjust_a, [0.0; 4]);
        assert_eq!(inst.adjust_b, [0.0; 2]);
        assert_eq!(inst.adjust_shadows, 0.0);
        assert_eq!(inst.pattern_data[0], 0.0); // force-tiled
        assert_eq!(inst.overlay_settings[1], BLEND_MODE_DISABLED);
    }

    #[test]
    fn image_adjust_reference_formula() {
        let none = ImageAdjustAmounts::default();
        let c = [0.2, 0.5, 0.8];
        let out = image_adjust(c, &none);
        for k in 0..3 {
            assert!((out[k] - c[k]).abs() < 1e-12, "identity");
        }
        // Exposure +0.5 is one stop: linear × 2.
        let e = image_adjust(
            [0.5; 3],
            &ImageAdjustAmounts {
                exposure: 0.5,
                ..none
            },
        );
        assert!((e[0] - 0.686).abs() < 1e-3, "{e:?}");
        // Saturation −1 is grey; contrast −1 is flat mid grey.
        let g = image_adjust(
            c,
            &ImageAdjustAmounts {
                saturation: -1.0,
                ..none
            },
        );
        assert!((g[0] - g[2]).abs() < 1e-12);
        let flat = image_adjust(
            c,
            &ImageAdjustAmounts {
                contrast: -1.0,
                ..none
            },
        );
        assert_eq!(flat, [0.5; 3]);
        // Warm: more red, less blue.
        let w = image_adjust(
            [0.5; 3],
            &ImageAdjustAmounts {
                temperature: 1.0,
                ..none
            },
        );
        assert!(w[0] > 0.5 && w[2] < 0.5 && (w[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn overlay_and_color_change_packing() {
        let node = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "placement": {}, "adjust": {"overlay": {"enabled": true, "color": "#06a77d", "opacity": 0.55, "blendMode": "multiply"}, "recolor": {"enabled": true, "from": ["#dc3232", "#325adc"], "to": ["#06a77d", "#ffbe0b"], "tolerance": 0.3, "smoothness": 0.6}}}}]
        }));
        let inst = build_pattern_instance(&ctx(&node, PatternTextureState::Loaded, None));
        assert_eq!(inst.overlay_settings, [0.55, 1.0]);
        assert!((inst.overlay_color[1] - 167.0 / 255.0).abs() < 1e-6);
        assert_eq!(inst.color_change_params[0], 2.0);
        assert!((inst.color_change_source[0][0] - 220.0 / 255.0).abs() < 1e-6);
        assert!((inst.color_change_target[1][0] - 1.0).abs() < 1e-6);
        // Unused slots stay [0,0,0,1].
        assert_eq!(inst.color_change_source[2], [0.0, 0.0, 0.0, 1.0]);

        // Disabled colorChange still packs colors, but count = 0.
        let node2 = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "placement": {}, "adjust": {"recolor": {"from": ["#dc3232"], "to": ["#06a77d"]}}}}]
        }));
        let inst2 = build_pattern_instance(&ctx(&node2, PatternTextureState::Loaded, None));
        assert_eq!(inst2.color_change_params[0], 0.0);
        assert!((inst2.color_change_source[0][0] - 220.0 / 255.0).abs() < 1e-6);
    }

    /// Figma's scale modes are live: a Fill image re-covers whatever box the
    /// shape has (the QA B4 empty band), a Fit one centres, a Crop stretches
    /// with the box it was placed in.
    #[test]
    fn scale_modes_resolve_against_the_box() {
        let fill = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "placement": {"fit": "cover", "naturalSize": {"width": 100, "height": 100}}}}]}));
        let inst = build_pattern_instance(&ctx(&fill, PatternTextureState::Loaded, None));
        // 260 × 200 box: cover scale 2.6, centred vertically.
        assert!((inst.pattern_data[1] - 2.6).abs() < 1e-5);
        assert!((inst.pattern_offset[0]).abs() < 1e-4);
        assert!((inst.pattern_offset[1] - (200.0 - 260.0) / 2.0).abs() < 1e-4);
        assert_eq!(inst.pattern_data[0], 0.75, "placed once");

        let fit = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "placement": {"fit": "contain", "naturalSize": {"width": 100, "height": 100}}}}]}));
        let inst = build_pattern_instance(&ctx(&fit, PatternTextureState::Loaded, None));
        assert!((inst.pattern_data[1] - 2.0).abs() < 1e-5);
        assert!((inst.pattern_offset[0] - 30.0).abs() < 1e-4);

        let crop = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "placement": {"fit": "crop",
                "offset": {"x": -10, "y": -20}, "scale": {"x": 1.5, "y": 1.5}, "box": {"width": 130, "height": 100}}}}]}));
        let inst = build_pattern_instance(&ctx(&crop, PatternTextureState::Loaded, None));
        // The 130 × 100 box doubled to 260 × 200: the crop doubles with it.
        assert!((inst.pattern_data[1] - 3.0).abs() < 1e-5);
        assert!((inst.pattern_offset[0] + 20.0).abs() < 1e-4);
        assert!((inst.pattern_offset[1] + 40.0).abs() < 1e-4);

        // CSS `object-fit: fill`: stretched on each axis onto the 260 × 200 box, from its origin.
        let stretch = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "placement": {"fit": "fill", "naturalSize": {"width": 100, "height": 100}}}}]}));
        let inst = build_pattern_instance(&ctx(&stretch, PatternTextureState::Loaded, None));
        assert!((inst.pattern_data[1] - 2.6).abs() < 1e-5);
        assert!((inst.pattern_data[2] - 2.0).abs() < 1e-5);
        assert!(inst.pattern_offset[0].abs() < 1e-4);
        assert!(inst.pattern_offset[1].abs() < 1e-4);
    }

    #[test]
    fn scale_falsy_coercion_and_defaults() {
        let node = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "placement": {"fit": "tile", "scale": {"x": 0, "y": 2.5}, "rotation": 30}}}]}));
        let inst = build_pattern_instance(&ctx(&node, PatternTextureState::Loaded, None));
        assert_eq!(inst.pattern_data[1], 1.0); // 0 → 1 (JS ||)
        assert_eq!(inst.pattern_data[2], 2.5);
        assert!((inst.pattern_data[3] - 30.0f32.to_radians()).abs() < 1e-7);
        // Overlay default opacity 1, disabled (−1).
        assert_eq!(inst.overlay_settings, [1.0, BLEND_MODE_DISABLED]);
        // Default red overlay color when unset.
        assert_eq!(inst.overlay_color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn pattern_warp_rows_pack_and_disable() {
        // Warp rows pack from the distort pattern-warp view (h33 = 1, flag on).
        let node = raw(json!({"id": "n", "type": "rectangle",
            "width": 480, "height": 320,
            "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "data:image/png;base64,xx", "placement": {}}}],
            "distort": fixture_distort()}));
        let p = node_distort(&node).unwrap().as_pattern_warp();
        let inst = build_pattern_instance(&ctx(&node, PatternTextureState::Loaded, Some(&p)));
        assert_eq!(inst.perspective_row0[3], 1.0); // enabled flag
        assert_eq!(inst.bounds, [0.0, 0.0, 480.0, 320.0]); // source dims

        // No warp → identity rows, flag 0, tess bounds.
        let flat = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "data:x", "placement": {}}}]}));
        let inst = build_pattern_instance(&ctx(&flat, PatternTextureState::Loaded, None));
        assert_eq!(inst.perspective_row0, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(inst.bounds, [0.0, 0.0, 260.0, 200.0]);
    }

    /// The patterns/distort-pattern fixture quad as a normalized unit-square matrix:
    /// distortMatrixFromLocalCorners([(60,30),(440,0),(480,320),(0,290)], 480, 320).
    fn fixture_distort() -> moda_scene::value::Value {
        json!({"type": "matrix", "matrix": [
            0.6364829396325459, -0.125, 0.125,
            -0.09375, 0.6364829396325459, 0.09375,
            -0.16929133858267717, -0.19422572178477693, 1.0
        ]})
    }

    #[test]
    fn node_distort_gates_corners_and_inverse() {
        let node = raw(json!({"id": "n", "type": "rectangle",
            "width": 480, "height": 320,
            "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3366ff"}}],
            "distort": fixture_distort()}));
        let d = node_distort(&node).expect("valid matrix");
        assert_eq!(d.source_width, 480.0);
        // Derived corners land on the equivalent corner-pin quad.
        let expected = [(60.0, 30.0), (440.0, 0.0), (480.0, 320.0), (0.0, 290.0)];
        for (c, (ex, ey)) in d.corners.iter().zip(expected.iter()) {
            assert!((c.x - ex).abs() < 1e-6 && (c.y - ey).abs() < 1e-6, "{c:?}");
        }
        // The inverse homography maps warped corners back to the source rect.
        let back = crate::paint::homography::apply_projective_to_point(
            d.corners[2],
            &d.inverse_homography,
        )
        .unwrap();
        assert!((back.x - 480.0).abs() < 1e-6 && (back.y - 320.0).abs() < 1e-6);

        // The pattern-warp view packs into the SAME instance rows as legacy
        // corner-pin metadata (enabled flag + source-dim UV bounds).
        let pattern_node = raw(json!({"id": "n", "type": "rectangle",
            "width": 480, "height": 320,
            "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "data:image/png;base64,xx", "placement": {}}}],
            "distort": fixture_distort()}));
        let dp = node_distort(&pattern_node).unwrap().as_pattern_warp();
        let inst =
            build_pattern_instance(&ctx(&pattern_node, PatternTextureState::Loaded, Some(&dp)));
        assert_eq!(inst.perspective_row0[3], 1.0); // enabled flag
        assert_eq!(inst.bounds, [0.0, 0.0, 480.0, 320.0]); // source dims

        // Distort is NOT pattern-gated: a color fill still warps geometry.
        let color_fill = raw(json!({"id": "n", "type": "rectangle",
            "width": 480, "height": 320, "distort": fixture_distort()}));
        assert!(node_distort(&color_fill).is_some());

        // A corner-pin-shaped payload under `distort` is out-of-schema and
        // rejected (the legacy field converts via the load-time migration).
        let corner_pin = raw(json!({"id": "n", "type": "rectangle",
            "width": 480, "height": 320,
            "distort": {"type": "corner-pin", "corners": [
                {"x": 60, "y": 30}, {"x": 440, "y": 0},
                {"x": 480, "y": 320}, {"x": 0, "y": 290}]}}));
        assert!(node_distort(&corner_pin).is_none());

        // Unsupported node types never warp.
        let line = raw(json!({"id": "n", "type": "line",
            "width": 480, "height": 320,
            "distort": fixture_distort()}));
        assert!(node_distort(&line).is_none());

        // Malformed / invalid matrices are rejected (TS parity: coerce + validate).
        for bad in [
            json!({"type": "matrix", "matrix": [1, 0, 0, 0, 1, 0, 0, 0]}), // 8 entries
            json!({"type": "matrix", "matrix": [1, 0, 0, 0, true, 0, 0, 0, 1]}), // bool entry
            json!({"type": "matrix", "matrix": [1, 0, 0, 2, 0, 0, 0, 0, 1]}), // singular
            json!({"type": "matrix", "matrix": [1, 0, 0, 0, 1, 0, -1.5, 0, 1]}), // sign flip
        ] {
            let node = raw(json!({"id": "n", "type": "rectangle",
                "width": 480, "height": 320, "distort": bad}));
            assert!(node_distort(&node).is_none(), "must reject invalid distort");
        }

        // Degenerate dims are rejected even with a valid matrix.
        let zero_w = raw(json!({"id": "n", "type": "rectangle",
            "width": 0, "height": 320, "distort": fixture_distort()}));
        assert!(node_distort(&zero_w).is_none());

        // An advisory `source` rides along and is ignored by the renderer.
        let with_source = raw(json!({"id": "n", "type": "rectangle",
            "width": 480, "height": 320,
            "distort": {"type": "matrix",
                "matrix": [1, 0, 0, 0, 1, 0, 0.3, 0.1, 1],
                "source": {"kind": "skew", "skewXDeg": 15, "skewYDeg": 0}}}));
        assert!(node_distort(&with_source).is_some());
    }

    #[test]
    fn identity_distort_is_no_distortion() {
        // A persisted IDENTITY matrix is a visually no-op field — it must NOT
        // count as a warp, or it would flatten curves and disable the SDF fast
        // paths, drop shadows, and background blur (all gated on
        // `node_distort(..).is_some()`). Every consumer shares this reader, so
        // one assertion covers them all.
        let identity = raw(json!({"id": "n", "type": "rectangle",
            "width": 480, "height": 320,
            "distort": {"type": "matrix", "matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1]}}));
        assert!(node_distort(&identity).is_none());

        // Near-identity within the 1e-9 epsilon (a rest-pose animation frame,
        // a reset gesture's float noise): also no distortion.
        let near = raw(json!({"id": "n", "type": "rectangle",
            "width": 480, "height": 320,
            "distort": {"type": "matrix",
                "matrix": [1.0000000001, 0, 0, 0, 1, 0, 0, 0, 1]}}));
        assert!(node_distort(&near).is_none());

        // A positively-scaled identity representative normalizes to identity
        // first, then declines.
        let scaled = raw(json!({"id": "n", "type": "rectangle",
            "width": 480, "height": 320,
            "distort": {"type": "matrix", "matrix": [2, 0, 0, 0, 2, 0, 0, 0, 2]}}));
        assert!(node_distort(&scaled).is_none());

        // ...but a real (if small) lean beyond the epsilon still warps.
        let lean = raw(json!({"id": "n", "type": "rectangle",
            "width": 480, "height": 320,
            "distort": {"type": "matrix", "matrix": [1, 0, 0, 0, 1, 0, 1e-6, 0, 1]}}));
        assert!(node_distort(&lean).is_some());
    }
}
