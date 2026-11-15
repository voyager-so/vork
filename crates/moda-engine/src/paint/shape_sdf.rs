//! shape_sdf.rs — analytic SDF fast path for solid/gradient rect & ellipse
//!
//! Port of the WebGPU SDF fast path: SDF-eligible solid/gradient rect & ellipse
//! bodies render through the `shape-solid` / `shape-gradient` WGSL modules
//! (extracted byte-identical from the production assemblers) instead of being
//! tessellated, so they stay crisp at any zoom. Anything failing the eligibility
//! predicate falls through to the existing tessellation path unchanged.
//!
//! Ports (originally line for line, now extended for SDF strokes — see below):
//! - `WebGPUShapeSystem.classifySolidShapeBody` (`:1924-1986`),
//! - `WebGPUShapeSystem.buildSolidShapeInstance` (`:1988-2048`),
//! - `WebGPUShapeSystem.tryEnqueueGradientRectangle/Ellipse` + `enqueueGradientShape`
//!   eligibility/phase (`:2164-2313`),
//! - `WebGPUShadowUtils.computeInnerShadowInstanceFields` (`WebGPUShadowUtils.ts:155-184`).
//!
//! This intentionally DIVERGES from the TS classifiers above for one case: a
//! solid, non-dashed stroke on a curved body (corner-radius rect/container or
//! ellipse) is accepted here and rendered as the analytic SDF stroke ring,
//! whereas the TS `classifySolidShapeBody` / gradient enqueue keep stroked bodies
//! on tessellation. The WASM engine is becoming the sole renderer, so this
//! native-only routing is the documented contract, not a porting bug.
//!
//! The `mask` (active clip-mask context) and `disableOpaqueFastPath`
//! (blur-segmented multi-pass) disqualifiers live in the caller (`frame.rs`
//! `build_body_item`), which owns that context; everything node-intrinsic is
//! here so it is unit-testable against the WebGPU truth table.

use bytemuck::{Pod, Zeroable};
use moda_scene::paint::{self as scene_paint, Paint, Stroke};
use moda_scene::types::RawNode;
use moda_scene::value::Value;

use crate::color::{color_to_normalized_rgb, color_to_normalized_rgba};
use crate::paint::gradient::{self, MAX_GRADIENT_STOPS};

/// `shapeType` discriminator in the `dimensions.w` slot (matches the WGSL
/// `sdShape` dispatch: `< 0.5` → rounded box, else ellipse).
pub const SHAPE_RECT: f32 = 0.0;
pub const SHAPE_ELLIPSE: f32 = 1.0;

// ---------------------------------------------------------------------------
// GPU instance ABI
// ---------------------------------------------------------------------------

/// `RectData` (`wgsl/shape-solid.wgsl`) — 48 floats / 192 bytes. Field order
/// and offsets mirror the WGSL struct exactly: every `vec4` lands on a 16-byte
/// boundary, so the flat `[f32; N]` layout matches std430 with no padding.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct ShapeSolidSdfInstance {
    /// floats 0..16 — column-major model matrix.
    pub model_matrix: [f32; 16],
    /// floats 16..20 — fill rgba (straight; a = fill α × opacity for the
    /// transparent phase, or fill α for the opaque phase — both ≤ 1).
    pub color: [f32; 4],
    /// floats 20..24 — [width, height, depth, shapeType].
    pub dimensions: [f32; 4],
    /// floats 24..28 — corner radii [tl, tr, br, bl] (rect) or
    /// [radiusX, radiusY, 0, 0] (ellipse).
    pub corner_radii: [f32; 4],
    /// floats 28..32 — page/effective clip rect [x, y, w, h] in canvas space.
    pub page_clip_bounds: [f32; 4],
    /// floats 32..36 — inner-shadow rgb + combined alpha (0 ⇒ no inner shadow).
    pub inner_shadow_color: [f32; 4],
    /// floats 36..40 — inner-shadow [offsetX, offsetY, blur, spread] in
    /// shape-local pixels.
    pub inner_shadow_params: [f32; 4],
    /// floats 40..44 — stroke rgb (straight) + combined alpha (stroke α ×
    /// opacity); a = 0 ⇒ no stroke (SDF stroke ring).
    pub stroke_color: [f32; 4],
    /// floats 44..48 — [width, alignCode, 0, 0]; alignCode 0=center, 1=inside,
    /// 2=outside (shape-local pixels).
    pub stroke_params: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<ShapeSolidSdfInstance>() == 192);

/// `GradientShapeData` (`wgsl/shape-gradient.wgsl`) — 88 floats / 352 bytes.
/// The two `vec2`s (`gradient_start`/`gradient_end`) pair to fill one 16-byte
/// slot, so the flat layout matches std430 exactly (same reasoning as
/// `GradientInstance`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct ShapeGradientSdfInstance {
    /// floats 0..16 — column-major model matrix.
    pub model_matrix: [f32; 16],
    /// floats 16..20 — [1, 1, 1, opacity]; the gradient color comes from the
    /// stops, scaled by this `.a` in the shader.
    pub color: [f32; 4],
    /// floats 20..24 — [width, height, depth, shapeType].
    pub dimensions: [f32; 4],
    /// floats 24..28 — corner radii (rect) or [radiusX, radiusY, 0, 0] (ellipse).
    pub corner_radii: [f32; 4],
    /// floats 28..30 — linear start point OR radial center (% 0-100).
    pub gradient_start: [f32; 2],
    /// floats 30..32 — linear end point OR radial radiusX/radiusY (%).
    pub gradient_end: [f32; 2],
    /// floats 32..36 — [fillType (1 linear, 2 radial), numStops, rotation, 0].
    pub gradient_meta: [f32; 4],
    /// floats 36..68 — packed fill stops.
    pub gradient_stops: [[f32; 4]; MAX_GRADIENT_STOPS],
    /// floats 68..72 — page/effective clip rect [x, y, w, h] in canvas space.
    pub page_clip_bounds: [f32; 4],
    /// floats 72..76 — inner-shadow rgb + combined alpha (0 ⇒ none).
    pub inner_shadow_color: [f32; 4],
    /// floats 76..80 — inner-shadow [offsetX, offsetY, blur, spread] shape-local.
    pub inner_shadow_params: [f32; 4],
    /// floats 80..84 — stroke rgb (straight) + combined alpha (stroke α ×
    /// opacity); a = 0 ⇒ no stroke (SDF stroke ring).
    pub stroke_color: [f32; 4],
    /// floats 84..88 — [width, alignCode, 0, 0]; alignCode 0=center, 1=inside,
    /// 2=outside (shape-local pixels).
    pub stroke_params: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<ShapeGradientSdfInstance>() == 352);

// ---------------------------------------------------------------------------
// Eligibility classifiers (node-intrinsic; mask/disable gating is the caller's)
// ---------------------------------------------------------------------------

/// The two SDF-eligible body shapes (matches the WGSL `shapeType` codes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeType {
    /// rectangle / container.
    Rect,
    /// full-sweep ellipse / circle.
    Ellipse,
}

impl ShapeType {
    fn code(self) -> f32 {
        match self {
            ShapeType::Rect => SHAPE_RECT,
            ShapeType::Ellipse => SHAPE_ELLIPSE,
        }
    }

    /// rect/container → `Rect`, ellipse → `Ellipse`; anything else is not an
    /// SDF body shape.
    pub fn from_node_type(node_type: &str) -> Option<ShapeType> {
        match node_type {
            "rectangle" | "container" => Some(ShapeType::Rect),
            "ellipse" => Some(ShapeType::Ellipse),
            _ => None,
        }
    }
}

/// SDF stroke-ring fields (curved-body solid stroke). Instance-ready:
/// the alpha and align are already resolved so the instance builders just copy
/// them into `stroke_color` / `stroke_params`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeFields {
    /// Straight stroke rgb + combined alpha (stroke color α × node opacity);
    /// copied into the instance `stroke_color`. `a == 0` ⇒ effectively no stroke.
    pub color: [f32; 4],
    /// `[width, alignCode, 0, 0]` — alignCode 0=center, 1=inside, 2=outside.
    pub params: [f32; 4],
}

/// Result of classifying a solid rect/ellipse body for the SDF fast path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolidShapeClass {
    /// Opaque phase (depth-write) iff no drop shadow, fully opaque fill ×
    /// opacity, AND (no stroke OR opaque stroke). Otherwise the transparent
    /// SDF phase.
    pub is_opaque: bool,
    /// Drop shadow present — does not disqualify, only forces transparent phase
    /// (the drop-shadow quad still renders via the existing shadow draw).
    pub has_shadow: bool,
    /// Straight fill rgb (alpha tracked separately via `body_alpha`/`opacity`).
    pub fill_rgb: (f32, f32, f32),
    /// `node.opacity ?? 1`.
    pub opacity: f32,
    /// `fillRgba.a × opacity` — the body alpha used in the transparent phase.
    pub body_alpha: f32,
    /// SDF stroke ring (curved bodies only): `Some` for a qualifying solid,
    /// non-dashed stroke; `None` for an unstroked body.
    pub stroke: Option<StrokeFields>,
}

/// Whether an ellipse is the SDF-representable full-sweep, no-inner-radius case
/// (`sweepAngle === undefined || |sweepAngle| ≥ 360` AND `innerRadiusRatio == 0`).
fn ellipse_is_full_sweep(node: &RawNode) -> bool {
    let inner_radius_ratio = node.f64("innerRadiusRatio").unwrap_or(0.0);
    let is_full_sweep = match node.f64("sweepAngle") {
        None => true,
        Some(s) => s.abs() >= 360.0,
    };
    is_full_sweep && inner_radius_ratio <= 0.0
}

/// `node.blendMode && node.blendMode !== 'normal'` — only normal/undefined pass.
fn blend_mode_is_normal(node: &RawNode) -> bool {
    match node.str("blendMode") {
        None => true,
        Some(b) => b == "normal",
    }
}

/// The top stroke when it is visible: a stroke paint (`moda_scene::paint::
/// top_stroke`) with `strokeWidth > 0`. A visible stroke that is NOT an
/// SDF-eligible solid stroke ring (`solid_stroke_ring_fields`) disqualifies the
/// body; an eligible one is rendered as the SDF stroke ring instead.
///
/// A `strokeWidth > 0` stroke whose color resolves to alpha 0 still counts as
/// "visible" here on purpose. The ring carries alpha 0 (so the shader paints
/// nothing — `strokeColor.a <= 0`), but it must stay CLASSIFIED rather than
/// collapsing to `None`: `drop_shadow_instance` expands the drop-shadow
/// silhouette by the PAINTED stroke width (`painted_stroke_width` — any paint
/// route counts, alpha 0 included) regardless of alpha, so a `None` stroke
/// would let `stroke_preserves_shadow_cover` skip the knockout while the shadow
/// is still expanded — leaking a half-stroke halo around the fill. Keeping it
/// classified makes `stroke_preserves_shadow_cover` return false (zero-alpha is
/// not opaque), so the knockout stays on, matching the tessellated path. See
/// `alpha_zero_stroke_keeps_shadow_knockout`.
fn visible_stroke(node: &RawNode) -> Option<Stroke> {
    gradient::top_stroke(node).filter(|s| s.width > 0.0)
}

/// A curved SDF body: an ellipse (already full-sweep/no-inner-radius gated by
/// the caller) or a rect/container with a non-zero corner radius. Sharp rects
/// gain nothing from an SDF stroke ring (straight strokes are already crisp) so
/// they keep tessellating.
fn is_curved_shape(node: &RawNode, shape_type: ShapeType) -> bool {
    match shape_type {
        ShapeType::Ellipse => true,
        ShapeType::Rect => has_nonzero_corner_radius(node),
    }
}

/// `cornerRadius` is a scalar > 0, or an array with any element > 0. Mirrors the
/// ticket's curved-rect gate on the RAW radius (pre-clamp): a degenerate clamp
/// to 0 only yields a 0-area shape, which is invisible either way.
fn has_nonzero_corner_radius(node: &RawNode) -> bool {
    let is_positive = |v: &Value| v.as_f64().map(|f| f > 0.0).unwrap_or(false);
    match node.json_field("cornerRadius") {
        Some(Value::Array(a)) => a.iter().any(is_positive),
        Some(v) => is_positive(v),
        None => false,
    }
}

/// The stroke's dash is a non-empty array (`Array.isArray(dash) && length`).
/// A dashed stroke is out of scope for the SDF ring, so it disqualifies. A
/// non-array `dash` (the WEB-NEXTJS-13T string regression) is treated as no
/// dash, matching the tessellator's `Array.isArray` guard.
fn stroke_has_dash(node: &RawNode) -> bool {
    scene_paint::node_stroke_style(node)
        .dash
        .is_some_and(|a| !a.is_empty())
}

/// Trim paths reshape the stroke into arc-length windows, which the SDF ring
/// cannot express — a trimmed stroke disqualifies the ring and falls the body
/// back to tessellation (where trim is implemented), exactly like dash.
fn stroke_has_trim(node: &RawNode) -> bool {
    let start = node.f64("trimPathStart").unwrap_or(0.0);
    let end = node.f64("trimPathEnd").unwrap_or(1.0);
    let offset = node.f64("trimPathOffset").unwrap_or(0.0);
    !crate::tess::is_trim_identity(start, end, offset)
}

/// Build the SDF stroke-ring fields for a node's visible stroke, or `None` if
/// the stroke is not SDF-eligible — falling the whole body back to
/// tessellation. Eligible = curved shape + solid stroke paint + no dash + a
/// parseable stroke color. `opacity` is `node.opacity ?? 1`; the stroke alpha
/// is `strokeColor.a × opacity`, the same convention as the tessellated body's
/// stroke.
fn solid_stroke_ring_fields(
    node: &RawNode,
    stroke: &Stroke,
    shape_type: ShapeType,
    opacity: f32,
) -> Option<StrokeFields> {
    if !is_curved_shape(node, shape_type) {
        return None;
    }
    // Solid stroke only: any other paint is out of scope.
    let color = scene_paint::solid_color(&stroke.paint)?;
    if stroke_has_dash(node) {
        return None;
    }
    if stroke_has_trim(node) {
        return None;
    }
    // hex / rgb() only (matches the fill convention). A named/unparseable stroke
    // color declines the ring so the body tessellates (where it renders black).
    let c = color_to_normalized_rgba(color)?;
    let width = stroke.width as f32;
    let align_code = match stroke.align.as_deref() {
        Some("inside") => 1.0,
        Some("outside") => 2.0,
        _ => 0.0, // center (default)
    };
    Some(StrokeFields {
        color: [c.r as f32, c.g as f32, c.b as f32, c.a as f32 * opacity],
        params: [width, align_code, 0.0, 0.0],
    })
}

/// Whether the opaque body fully occludes its OWN drop shadow, so the SDF
/// shadow knockout-skip (`opaque_sdf_covers_shadow`) stays correct.
///
/// The drop-shadow silhouette is always expanded by HALF the stroke width on
/// every side (center-stroke assumption, `drop_shadow_instance`), and the
/// knockout-skip relies on the painted body covering that expanded band. The
/// body's outward paint reach decides whether it does:
/// - no stroke → silhouette is unexpanded, body covers it exactly → TRUE.
/// - opaque CENTER stroke → paint reaches +half_stroke = the expanded band → TRUE.
/// - opaque OUTSIDE stroke → paint reaches +stroke_width ⊇ the expanded band → TRUE.
/// FALSE for everything else:
/// - any INSIDE stroke → paint stops at the fill edge (+0); the outer half-stroke
///   band is never painted, so skipping the knockout would leak the expanded drop
///   shadow as a halo. Keeping the knockout matches the tessellated path (which
///   never sets the flag), so SDF and non-SDF inside strokes render identically.
/// - a translucent CENTER/OUTSIDE stroke → its band sits beyond the fill but does
///   not fully occlude, so the shadow must still be carved (CSS box-shadow).
pub fn stroke_preserves_shadow_cover(stroke: Option<StrokeFields>) -> bool {
    stroke.map_or(true, |s| {
        // alignCode: 0=center, 1=inside, 2=outside (decoded like the shader).
        let inside = s.params[1] > 0.5 && s.params[1] < 1.5;
        !inside && s.color[3] >= 1.0
    })
}

/// The node carries a valid `distort` matrix: its geometry is
/// pre-warped through the local homography, so the body is never an SDF box
/// and the SDF fast paths must decline and route through tessellation.
fn has_perspective(node: &RawNode) -> bool {
    crate::paint::pattern::node_distort(node).is_some()
}

/// The SDF instance bakes ONE inner shadow (`inner_shadow_fields`): a node
/// with several declines the lane, so its body tessellates and every inner
/// shadow draws as its own quad over it (`effects::inner_shadow_instances`).
fn bakes_too_many_inner_shadows(node: &RawNode) -> bool {
    crate::effects::inner_shadow_quad_count(node) > 1
}

/// `hasRenderableSourceMaskShaderEffect(node)` — a renderable source-mask shader
/// effect routes the node through the custom-shader pipeline, so the SDF body
/// fast path must decline.
fn has_source_mask_shader_effect(node: &RawNode) -> bool {
    moda_scene::shader_effects::renderable_source_mask_effect(node).is_some()
}

/// `classifySolidShapeBody` (`WebGPUShapeSystem.ts:1924-1986`). Returns `None`
/// for an ineligible body (the caller falls back to tessellation). Node-intrinsic
/// only: the active-mask and `disableOpaqueFastPath` checks are the caller's.
pub fn classify_solid_shape_body(node: &RawNode, shape_type: ShapeType) -> Option<SolidShapeClass> {
    // Ellipses must be full-sweep with no inner radius.
    if shape_type == ShapeType::Ellipse && !ellipse_is_full_sweep(node) {
        return None;
    }
    if has_source_mask_shader_effect(node) || bakes_too_many_inner_shadows(node) {
        return None;
    }
    if !blend_mode_is_normal(node) {
        return None;
    }
    // The SDF solid path is a constant-color body: the top fill is a solid (or
    // there is none: a transparent body, the stroke-only look); a gradient,
    // image, video or shader fill routes elsewhere. The same paint the
    // tessellated body draws.
    let fill_paint = gradient::fill_paint(node);
    let fill_color = match &fill_paint {
        None => None,
        Some(Paint::Solid(s)) => Some(s.color.as_str()?.to_string()),
        Some(_) => return None,
    };
    // A gradient stroke needs the path pipeline (the SDF stroke ring is solid
    // only) — out of scope, decline.
    if gradient::has_visible_gradient_stroke(node) {
        return None;
    }
    let visible = visible_stroke(node);
    let opacity = node.f64("opacity").unwrap_or(1.0) as f32;
    // Stroke ring: a visible solid stroke on a CURVED body is rendered
    // as the analytic SDF stroke ring; a stroke that is not ring-eligible (sharp
    // rect, dashed, or a gradient) disqualifies the whole body so it
    // tessellates (no regression — straight strokes are already crisp).
    let stroke = match &visible {
        Some(s) => Some(solid_stroke_ring_fields(node, s, shape_type, opacity)?),
        None => None,
    };
    // Smoothed (squircle) and styled (inverted / chamfer) corners have no SDF: tessellate the path.
    if shape_type == ShapeType::Rect
        && crate::shape_to_path::rect_has_path_corners(node.json_field("cornerRadius"), node.json_field("cornerSmoothing"), node.json_field("cornerStyle"))
    {
        return None;
    }
    if has_perspective(node) {
        return None;
    }
    // No fill is a transparent body (the stroke-only look); a colour that
    // doesn't parse declines (the tessellated body draws nothing for it).
    let fill = match fill_color {
        None => crate::color::Rgba {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        },
        Some(c) => color_to_normalized_rgba(&c)?,
    };

    let has_shadow = crate::effects::has_drop_shadow_quad(node);
    let fill_a = fill.a as f32;
    let body_alpha = fill_a * opacity;
    // Opaque phase requires an opaque stroke too — a translucent stroke (or one
    // sitting on a translucent body) z-interleaves in the transparent phase.
    // `color[3]` is stroke α × opacity, but `is_opaque` already gates `opacity
    // >= 1`, so this is decisive only when opacity == 1 (⇔ raw stroke α >= 1).
    let stroke_opaque = stroke.map_or(true, |s| s.color[3] >= 1.0);
    let is_opaque = !has_shadow && fill_a >= 1.0 && opacity >= 1.0 && stroke_opaque;
    Some(SolidShapeClass {
        is_opaque,
        has_shadow,
        fill_rgb: (fill.r as f32, fill.g as f32, fill.b as f32),
        opacity,
        body_alpha,
        stroke,
    })
}

/// Result of classifying a gradient rect/ellipse body for the SDF fast path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientShapeClass {
    /// 1 = linear, 2 = radial (the only two SDF-eligible gradient fills).
    pub fill_type: u8,
    /// `(opacity < 1) || gradientHasTransparency(node)` — the FILL's
    /// transparency only. It supplies the fill term of `opaque_sdf_covers_shadow`;
    /// the stroke's separate effect on shadow cover is decided by
    /// `stroke_preserves_shadow_cover`.
    pub is_transparent: bool,
    /// Drop shadow present — forces the transparent phase.
    pub has_shadow: bool,
    /// SDF stroke ring (curved bodies only): `Some` for a qualifying solid,
    /// non-dashed stroke; `None` for an unstroked body.
    pub stroke: Option<StrokeFields>,
}

/// The SDF lane's linear or radial fill: the top fill, when it is one.
fn sdf_gradient_paint(node: &RawNode) -> Option<Paint> {
    gradient::fill_paint(node).filter(|p| matches!(p, Paint::Linear(_) | Paint::Radial(_)))
}

/// `tryEnqueueGradientRectangle/Ellipse` + `isTransparentGradientShape`
/// (`WebGPUShapeSystem.ts:2164-2217`). Node-intrinsic eligibility only.
pub fn classify_gradient_shape(
    node: &RawNode,
    shape_type: ShapeType,
) -> Option<GradientShapeClass> {
    if has_source_mask_shader_effect(node) || bakes_too_many_inner_shadows(node) {
        return None;
    }
    let Some(paint) = sdf_gradient_paint(node) else {
        return None;
    };
    let fill_type = match paint {
        Paint::Linear(_) => 1u8,
        Paint::Diamond(_) => 4u8,
        _ => 2u8,
    };
    if shape_type == ShapeType::Ellipse && !ellipse_is_full_sweep(node) {
        return None;
    }
    if !blend_mode_is_normal(node) {
        return None;
    }
    // A gradient stroke is out of scope (solid stroke ring only).
    if gradient::has_visible_gradient_stroke(node) {
        return None;
    }
    let visible = visible_stroke(node);
    let opacity = node.f64("opacity").unwrap_or(1.0);

    // Stroke ring: a visible solid stroke on a CURVED body rides along
    // with the gradient fill; a non-ring-eligible stroke disqualifies the body.
    let stroke = match &visible {
        Some(s) => Some(solid_stroke_ring_fields(
            node,
            s,
            shape_type,
            opacity as f32,
        )?),
        None => None,
    };
    // Smoothed (squircle) and styled (inverted / chamfer) corners have no SDF: tessellate the path.
    if shape_type == ShapeType::Rect
        && crate::shape_to_path::rect_has_path_corners(node.json_field("cornerRadius"), node.json_field("cornerSmoothing"), node.json_field("cornerStyle"))
    {
        return None;
    }
    if has_perspective(node) {
        return None;
    }

    let is_transparent = opacity < 1.0 || gradient::gradient_has_transparency(node);
    let has_shadow = crate::effects::has_drop_shadow_quad(node);
    Some(GradientShapeClass {
        fill_type,
        is_transparent,
        has_shadow,
        stroke,
    })
}

// ---------------------------------------------------------------------------
// Instance builders
// ---------------------------------------------------------------------------

/// `clampCornerRadii` (`WebGPUShapeSystem.ts:1904-1915`): each radius clamped to
/// `[0, min(width/2, height/2)]`.
fn corner_radii(node: &RawNode, shape_type: ShapeType, width: f64, height: f64) -> [f32; 4] {
    if shape_type == ShapeType::Ellipse {
        return [(width / 2.0) as f32, (height / 2.0) as f32, 0.0, 0.0];
    }
    let corner = node.json_field("cornerRadius");
    let read = |i: usize| -> f64 {
        match corner {
            Some(Value::Array(a)) => a.get(i).and_then(Value::as_f64).unwrap_or(0.0),
            Some(v) => v.as_f64().unwrap_or(0.0),
            None => 0.0,
        }
    };
    let limit = (width * 0.5).min(height * 0.5);
    let clamp = |v: f64| v.max(0.0).min(limit) as f32;
    [
        clamp(read(0)),
        clamp(read(1)),
        clamp(read(2)),
        clamp(read(3)),
    ]
}

/// `computeInnerShadowInstanceFields` (`WebGPUShadowUtils.ts:155-184`): the
/// per-instance inner-shadow rgb+alpha and shape-local [offsetX, offsetY, blur,
/// spread]. Returns the disabled fields (alpha 0) when there is no inner shadow.
fn inner_shadow_fields(node: &RawNode, model_matrix: &[f32; 16]) -> ([f32; 4], [f32; 4]) {
    // `extractInnerShadowParameters` gate (`hasInnerShadowEffect`): the
    // node's one `inner-shadow` entry that draws (`effect::inner_shadow`; a
    // node with more never reaches the SDF lane, `bakes_too_many_inner_shadows`).
    // An unfilled frame's inner shadow is its content's (a subtree capture,
    // `layer_blur::shadow::is_subtree_capture_node`), not its box's.
    if crate::layer_blur::shadow::is_subtree_capture_node(node) {
        return ([0.0; 4], [0.0; 4]);
    }
    let Some(shadow) = moda_scene::effect::inner_shadow(node) else {
        return ([0.0; 4], [0.0; 4]);
    };
    // `blur` and `offset` are always stored; a missing `spread` is 0 and a
    // missing `opacity` 1 (effect.ts).
    let blur = shadow.blur;
    let spread = shadow.spread();
    let (ox, oy) = (shadow.offset.x, shadow.offset.y);
    let opacity = shadow.opacity().clamp(0.0, 1.0);
    // colorToNormalizedRgb (hex / rgb() only) ?? black.
    let (r, g, b) = shadow
        .color
        .as_str()
        .and_then(color_to_normalized_rgb)
        .unwrap_or((0.0, 0.0, 0.0));

    // Invert the model matrix's linear 2x2 (columns 0 and 1) to map the canvas-
    // space offset into shape-local space; scale blur/spread by sqrt|det|.
    let a = model_matrix[0] as f64;
    let bb = model_matrix[1] as f64;
    let c = model_matrix[4] as f64;
    let d = model_matrix[5] as f64;
    let det = a * d - bb * c;
    let inv_det = if det.abs() > 1e-6 { 1.0 / det } else { 0.0 };
    let local_offset_x = (d * ox - c * oy) * inv_det;
    let local_offset_y = (-bb * ox + a * oy) * inv_det;
    // `Math.sqrt(Math.abs(det)) || 1` — JS `||` substitutes 1 only for a FALSY
    // result (0 or NaN), NOT a small-but-nonzero scale. We copy that verbatim
    // (no threshold clamp): a near-singular matrix scales the shape to ~zero
    // area, so a tiny `linear_scale` blowing the inset shadow up is an
    // invisible body either way — matching WebGPU exactly.
    let linear_scale = {
        let s = det.abs().sqrt();
        if s == 0.0 || s.is_nan() {
            1.0
        } else {
            s
        }
    };

    (
        [r as f32, g as f32, b as f32, opacity as f32],
        [
            local_offset_x as f32,
            local_offset_y as f32,
            (blur / linear_scale) as f32,
            (spread / linear_scale) as f32,
        ],
    )
}

/// `buildSolidShapeInstance` (`WebGPUShapeSystem.ts:1988-2048`). `alpha` is the
/// `opacity` (opaque phase) or `body_alpha` (transparent phase). `page_clip` is
/// the effective (mask-intersected) clip; `model_matrix`/`depth` come from the
/// caller's z + canvas affine.
pub fn build_solid_shape_instance(
    node: &RawNode,
    shape_type: ShapeType,
    fill_rgb: (f32, f32, f32),
    alpha: f32,
    stroke: Option<StrokeFields>,
    model_matrix: [f32; 16],
    depth: f32,
    page_clip: [f32; 4],
) -> ShapeSolidSdfInstance {
    let width = node.f64("width").unwrap_or(0.0);
    let height = node.f64("height").unwrap_or(0.0);
    let (inner_color, inner_params) = inner_shadow_fields(node, &model_matrix);
    let (stroke_color, stroke_params) = stroke_instance_fields(stroke);
    ShapeSolidSdfInstance {
        model_matrix,
        color: [fill_rgb.0, fill_rgb.1, fill_rgb.2, alpha],
        dimensions: [width as f32, height as f32, depth, shape_type.code()],
        corner_radii: corner_radii(node, shape_type, width, height),
        page_clip_bounds: page_clip,
        inner_shadow_color: inner_color,
        inner_shadow_params: inner_params,
        stroke_color,
        stroke_params,
    }
}

/// `enqueueGradientShape` instance packing (`WebGPUShapeSystem.ts:2256-2306`).
pub fn build_gradient_shape_instance(
    node: &RawNode,
    shape_type: ShapeType,
    fill_type: u8,
    stroke: Option<StrokeFields>,
    model_matrix: [f32; 16],
    depth: f32,
    page_clip: [f32; 4],
) -> ShapeGradientSdfInstance {
    let opacity = node.f64("opacity").unwrap_or(1.0) as f32;
    let width = node.f64("width").unwrap_or(0.0);
    let height = node.f64("height").unwrap_or(0.0);
    let paint = sdf_gradient_paint(node);
    let (gradient_start, gradient_end, rotation, num_stops, stops) = match &paint {
        Some(p) => gradient::shape_sdf_gradient_params(p),
        // Unreachable: the classifier admitted a linear or radial fill.
        None => ([0.0; 2], [0.0; 2], 0.0, 0, gradient::EMPTY_GRADIENT_STOPS),
    };
    let (inner_color, inner_params) = inner_shadow_fields(node, &model_matrix);
    let (stroke_color, stroke_params) = stroke_instance_fields(stroke);
    ShapeGradientSdfInstance {
        model_matrix,
        color: [1.0, 1.0, 1.0, opacity],
        dimensions: [width as f32, height as f32, depth, shape_type.code()],
        corner_radii: corner_radii(node, shape_type, width, height),
        gradient_start,
        gradient_end,
        gradient_meta: [
            fill_type as f32,
            num_stops as f32,
            rotation,
            paint.as_ref().map_or(0.0, |p| p.gradient_options().shader_flags()),
        ],
        gradient_stops: stops,
        page_clip_bounds: page_clip,
        inner_shadow_color: inner_color,
        inner_shadow_params: inner_params,
        stroke_color,
        stroke_params,
    }
}

/// Resolve the instance `stroke_color` / `stroke_params` from a classification's
/// stroke fields. `None` ⇒ the disabled pair (`stroke_color.a == 0`), which the
/// shaders read as "no stroke" (the quad does not grow and the fill renders
/// byte-identically to the pre-SDF-stroke path).
fn stroke_instance_fields(stroke: Option<StrokeFields>) -> ([f32; 4], [f32; 4]) {
    match stroke {
        Some(s) => (s.color, s.params),
        None => ([0.0; 4], [0.0; 4]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    fn raw(v: Value) -> RawNode {
        RawNode::from_value(v).unwrap()
    }

    fn solid_rect(extra: Value) -> RawNode {
        let mut m = json!({
            "id": "n", "type": "rectangle", "width": 100.0, "height": 80.0,
            "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}]
        });
        if let Value::Object(e) = extra {
            for (k, v) in e {
                m.as_object_mut().unwrap().insert(k, v);
            }
        }
        raw(m)
    }

    #[test]
    fn solid_rect_is_eligible_and_opaque() {
        let c = classify_solid_shape_body(&solid_rect(json!({})), ShapeType::Rect).unwrap();
        assert!(c.is_opaque);
        assert!(!c.has_shadow);
        assert_eq!(c.opacity, 1.0);
        assert_eq!(c.body_alpha, 1.0);
    }

    #[test]
    fn distort_declines_the_sdf_fast_path() {
        // A valid distort pre-warps the geometry, so the body must
        // tessellate — the axis-aligned SDF box cannot represent it.
        let warped = solid_rect(json!({
            "distort": {"type": "matrix", "matrix": [1, 0, 0, 0, 1, 0, 0.3, 0.1, 1]}
        }));
        assert!(classify_solid_shape_body(&warped, ShapeType::Rect).is_none());
        // An INVALID distort keeps the pre-existing fast path (renders
        // unwarped everywhere, consistently).
        let invalid = solid_rect(json!({
            "distort": {"type": "matrix", "matrix": [1, 0, 0, 2, 0, 0, 0, 0, 1]}
        }));
        assert!(classify_solid_shape_body(&invalid, ShapeType::Rect).is_some());
    }

    #[test]
    fn full_ellipse_eligible_partial_and_donut_rejected() {
        let full =
            raw(json!({"id":"e","type":"ellipse","width":100.0,"height":100.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}]}));
        assert!(classify_solid_shape_body(&full, ShapeType::Ellipse).is_some());

        // |sweepAngle| >= 360 still counts as full.
        let full360 = raw(
            json!({"id":"e","type":"ellipse","width":100.0,"height":100.0,
            "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}],"sweepAngle":360.0}),
        );
        assert!(classify_solid_shape_body(&full360, ShapeType::Ellipse).is_some());

        let wedge = raw(
            json!({"id":"e","type":"ellipse","width":100.0,"height":100.0,
            "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}],"sweepAngle":120.0}),
        );
        assert!(classify_solid_shape_body(&wedge, ShapeType::Ellipse).is_none());

        let donut = raw(
            json!({"id":"e","type":"ellipse","width":100.0,"height":100.0,
            "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}],"innerRadiusRatio":0.5}),
        );
        assert!(classify_solid_shape_body(&donut, ShapeType::Ellipse).is_none());
    }

    /// A rect WITH a non-zero corner radius (the SDF stroke ring needs a curved
    /// body). `solid_rect` stays sharp so it keeps exercising the reject path.
    fn rounded_rect(extra: Value) -> RawNode {
        let mut m = json!({
            "id": "n", "type": "rectangle", "width": 100.0, "height": 80.0,
            "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}], "cornerRadius": 16.0
        });
        if let Value::Object(e) = extra {
            for (k, v) in e {
                m.as_object_mut().unwrap().insert(k, v);
            }
        }
        raw(m)
    }

    #[test]
    fn stroke_pattern_conic_shader_blend_reject() {
        // Visible solid stroke on a SHARP rect still disqualifies (the SDF
        // stroke ring is curved-only; sharp strokes stay on tessellation).
        assert!(classify_solid_shape_body(
            &solid_rect(json!({"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#000"},"width":4.0}]})),
            ShapeType::Rect
        )
        .is_none());
        // A zero-width stroke does NOT disqualify.
        assert!(classify_solid_shape_body(
            &solid_rect(json!({"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#000"},"width":0.0}]})),
            ShapeType::Rect
        )
        .is_some());
        // Non-solid fill paints (video draws its poster through the pattern
        // texture path).
        let stops = json!([{"offset": 0, "color": "#ff0000"}, {"offset": 1, "color": "#0000ff"}]);
        for paint in [
            json!({"type": "image", "src": "a.png"}),
            json!({"type": "video", "src": "a.mp4"}),
            json!({"type": "linear", "start": {"x": 0, "y": 50}, "end": {"x": 100, "y": 50}, "stops": stops}),
            json!({"type": "radial", "center": {"x": 50, "y": 50}, "radiusX": 50, "radiusY": 50, "rotation": 0,
                   "stops": stops}),
            json!({"type": "conic", "center": {"x": 50, "y": 50}, "angle": 0, "stops": stops}),
            json!({"type": "shader", "shaderId": "plasma"}),
        ] {
            let ty = paint["type"].clone();
            assert!(
                classify_solid_shape_body(
                    &solid_rect(json!({"fills": [{"id": "fill-0", "paint": paint}]})),
                    ShapeType::Rect
                )
                .is_none(),
                "a {ty} fill should reject the SOLID path"
            );
        }
        // Non-normal blend mode.
        assert!(classify_solid_shape_body(
            &solid_rect(json!({"blendMode":"multiply"})),
            ShapeType::Rect
        )
        .is_none());
        // 'normal' is fine.
        assert!(classify_solid_shape_body(
            &solid_rect(json!({"blendMode":"normal"})),
            ShapeType::Rect
        )
        .is_some());
    }

    #[test]
    fn gradient_stroke_and_unparseable_fill_reject() {
        // A visible gradient stroke disqualifies the solid SDF body.
        let grad_stroke = solid_rect(json!({"strokes":[{"id":"stroke-0","width":4.0,"paint":{"type":"linear","start":{"x":0,"y":0},"end":{"x":100,"y":0},"stops":[{"offset":0,"color":"#ff0000"},{"offset":1,"color":"#00ff00"}]}}]}));
        assert!(classify_solid_shape_body(&grad_stroke, ShapeType::Rect).is_none());

        // An unparseable fill colour → ineligible (the tessellated body draws nothing).
        let named =
            raw(json!({"id":"n","type":"rectangle","width":10.0,"height":10.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "red"}}]}));
        assert!(classify_solid_shape_body(&named, ShapeType::Rect).is_none());
        // No fill is a transparent body ([Q5]: no grey default fill; this used
        // to decline so tessellation could draw the grey): eligible, alpha 0.
        let nofill = raw(json!({"id":"n","type":"rectangle","width":10.0,"height":10.0}));
        let c = classify_solid_shape_body(&nofill, ShapeType::Rect).expect("transparent body");
        assert_eq!(c.body_alpha, 0.0);
        assert!(!c.is_opaque);
    }

    #[test]
    fn translucent_and_shadowed_go_transparent() {
        // Translucent fill → transparent phase.
        let t = classify_solid_shape_body(
            &solid_rect(json!({"fills":[{"id":"fill-0","paint":{"type":"solid","color":"rgba(58,134,255,0.5)"}}]})),
            ShapeType::Rect,
        )
        .unwrap();
        assert!(!t.is_opaque);
        assert!((t.body_alpha - 0.5).abs() < 1e-6);

        // Opacity < 1 → transparent phase.
        let o = classify_solid_shape_body(&solid_rect(json!({"opacity":0.5})), ShapeType::Rect)
            .unwrap();
        assert!(!o.is_opaque);

        // Opaque fill + drop shadow → eligible but transparent phase.
        let s = classify_solid_shape_body(
            &solid_rect(json!({"effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000","blur":12.0}]})),
            ShapeType::Rect,
        )
        .unwrap();
        assert!(s.has_shadow);
        assert!(!s.is_opaque);
    }

    #[test]
    fn gradient_classifier_rect_and_ellipse() {
        let lin = raw(json!({
            "id":"g","type":"rectangle","width":100.0,"height":80.0,
            "fills": [{"id": "fill-0", "paint": {"type": "linear", "start": {"x": 0, "y": 50}, "end": {"x": 100, "y": 50}, "stops": [{"offset": 0, "color": "#ff006e"}, {"offset": 1, "color": "#3a86ff"}]}}]
        }));
        let c = classify_gradient_shape(&lin, ShapeType::Rect).unwrap();
        assert_eq!(c.fill_type, 1);
        assert!(!c.is_transparent);

        let rad = raw(json!({
            "id":"g","type":"ellipse","width":100.0,"height":100.0,
            "fills": [{"id": "fill-0", "paint": {"type": "radial", "center": {"x": 50, "y": 50}, "radiusX": 50, "radiusY": 50, "rotation": 0, "stops": [{"offset": 0, "color": "#06a77d"}, {"offset": 1, "color": "#1d2733"}]}}]
        }));
        let c = classify_gradient_shape(&rad, ShapeType::Ellipse).unwrap();
        assert_eq!(c.fill_type, 2);

        // A partial-sweep gradient ellipse is rejected.
        let wedge = raw(json!({
            "id":"g","type":"ellipse","width":100.0,"height":100.0,"sweepAngle":90.0,
            "fills": [{"id": "fill-0", "paint": {"type": "radial", "center": {"x": 50, "y": 50}, "radiusX": 50, "radiusY": 50, "rotation": 0, "stops": [{"offset": 0, "color": "#06a77d"}, {"offset": 1, "color": "#1d2733"}]}}]
        }));
        assert!(classify_gradient_shape(&wedge, ShapeType::Ellipse).is_none());

        // A solid-fill rect is not a gradient body.
        assert!(classify_gradient_shape(&solid_rect(json!({})), ShapeType::Rect).is_none());

        // A transparent gradient stop forces the transparent phase.
        let alpha = raw(json!({
            "id":"g","type":"rectangle","width":100.0,"height":80.0,
            "fills": [{"id": "fill-0", "paint": {"type": "linear", "start": {"x": 0, "y": 50}, "end": {"x": 100, "y": 50}, "stops": [{"offset": 0, "color": "#ff006e"}, {"offset": 1, "color": "rgba(58,134,255,0)"}]}}]
        }));
        assert!(
            classify_gradient_shape(&alpha, ShapeType::Rect)
                .unwrap()
                .is_transparent
        );
    }

    #[test]
    fn solid_instance_corner_radii_and_dims() {
        let node = solid_rect(json!({"cornerRadius":[48, 0, 60, 0]}));
        let inst = build_solid_shape_instance(
            &node,
            ShapeType::Rect,
            (0.2, 0.5, 1.0),
            1.0,
            None,
            identity(),
            0.5,
            [0.0, 0.0, 1280.0, 800.0],
        );
        // width 100, height 80 → limit = min(50,40) = 40; 48 and 60 clamp to 40.
        assert_eq!(inst.corner_radii, [40.0, 0.0, 40.0, 0.0]);
        assert_eq!(inst.dimensions, [100.0, 80.0, 0.5, SHAPE_RECT]);
        assert_eq!(inst.color, [0.2, 0.5, 1.0, 1.0]);
        assert_eq!(inst.page_clip_bounds, [0.0, 0.0, 1280.0, 800.0]);
        // No inner shadow → disabled fields.
        assert_eq!(inst.inner_shadow_color, [0.0; 4]);
    }

    #[test]
    fn ellipse_instance_radii_are_half_dims() {
        let node =
            raw(json!({"id":"e","type":"ellipse","width":200.0,"height":120.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}]}));
        let inst = build_solid_shape_instance(
            &node,
            ShapeType::Ellipse,
            (1.0, 1.0, 1.0),
            1.0,
            None,
            identity(),
            0.0,
            [0.0; 4],
        );
        assert_eq!(inst.corner_radii, [100.0, 60.0, 0.0, 0.0]);
        assert_eq!(inst.dimensions[3], SHAPE_ELLIPSE);
    }

    #[test]
    fn inner_shadow_baked_into_instance() {
        let node = solid_rect(json!({"effects":[{"id":"inner-shadow-0","type":"inner-shadow",
            "color":"#000000","blur":8.0,"offset":{"x":4.0,"y":6.0},"opacity":0.4,"spread":2.0}]}));
        let inst = build_solid_shape_instance(
            &node,
            ShapeType::Rect,
            (1.0, 1.0, 1.0),
            1.0,
            None,
            identity(),
            0.0,
            [0.0; 4],
        );
        // Identity model matrix: det = 1, so offsets pass through and blur/spread
        // are unscaled.
        assert_eq!(inst.inner_shadow_color, [0.0, 0.0, 0.0, 0.4]);
        assert_eq!(inst.inner_shadow_params, [4.0, 6.0, 8.0, 2.0]);

        // Opacity 0 disables the inner shadow.
        let off = solid_rect(json!({"effects":[{"id":"inner-shadow-0","type":"inner-shadow",
            "color":"#000000","blur":8.0,"offset":{"x":0.0,"y":0.0},"opacity":0.0}]}));
        let inst = build_solid_shape_instance(
            &off,
            ShapeType::Rect,
            (1.0, 1.0, 1.0),
            1.0,
            None,
            identity(),
            0.0,
            [0.0; 4],
        );
        assert_eq!(inst.inner_shadow_color, [0.0; 4]);
    }

    #[test]
    fn gradient_instance_meta_and_color() {
        let node = raw(json!({
            "id":"g","type":"rectangle","width":100.0,"height":80.0,"opacity":0.5,
            "fills": [{"id": "fill-0", "paint": {"type": "linear", "start": {"x": 10, "y": 50}, "end": {"x": 90, "y": 50}, "stops": [{"offset": 0, "color": "#ff006e"}, {"offset": 1, "color": "#3a86ff"}]}}]
        }));
        let inst = build_gradient_shape_instance(
            &node,
            ShapeType::Rect,
            1,
            None,
            identity(),
            0.25,
            [0.0; 4],
        );
        assert_eq!(inst.color, [1.0, 1.0, 1.0, 0.5]);
        assert_eq!(inst.gradient_meta[0], 1.0); // linear
        assert_eq!(inst.gradient_meta[1], 2.0); // 2 stops
        assert_eq!(inst.gradient_start, [10.0, 50.0]);
        assert_eq!(inst.gradient_end, [90.0, 50.0]);
        assert_eq!(inst.dimensions[2], 0.25); // depth
    }

    // -----------------------------------------------------------------------
    // SDF stroke ring
    // -----------------------------------------------------------------------

    #[test]
    fn smoothed_corners_decline_the_sdf() {
        // cornerSmoothing > 0 on a rounded rect → tessellate the squircle path.
        let smooth = rounded_rect(json!({"cornerSmoothing": 0.6}));
        assert!(classify_solid_shape_body(&smooth, ShapeType::Rect).is_none());
        let grad = rounded_rect(json!({"cornerSmoothing": 0.6, "fills": [{"id": "fill-0", "paint": {"type": "linear",
            "start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 0},
            "stops": [{"offset": 0, "color": "#ff0000"}, {"offset": 1, "color": "#00ff00"}]}}]}));
        assert!(classify_gradient_shape(&grad, ShapeType::Rect).is_none());
        // Smoothing 0, or smoothing on a sharp rect, keeps the fast path.
        assert!(classify_solid_shape_body(&rounded_rect(json!({"cornerSmoothing": 0.0})), ShapeType::Rect).is_some());
        assert!(classify_solid_shape_body(&solid_rect(json!({"cornerSmoothing": 0.6})), ShapeType::Rect).is_some());
    }

    #[test]
    fn solid_stroke_on_curved_body_accepted() {
        // Rounded rect + center stroke → accepted, stroke fields packed.
        let c = classify_solid_shape_body(
            &rounded_rect(json!({"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#ff0000"},"width":6.0}]})),
            ShapeType::Rect,
        )
        .unwrap();
        let s = c.stroke.expect("rounded rect stroke ring");
        assert_eq!(s.color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(s.params, [6.0, 0.0, 0.0, 0.0]); // center
        assert!(c.is_opaque); // opaque fill + opaque stroke + no shadow

        // Ellipse + stroke → accepted (ellipses are curved by definition).
        let e = classify_solid_shape_body(
            &raw(
                json!({"id":"e","type":"ellipse","width":100.0,"height":100.0,
                "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}],"strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 3}]}),
            ),
            ShapeType::Ellipse,
        )
        .unwrap();
        assert!(e.stroke.is_some());

        // A non-zero element in a per-corner array also counts as curved.
        let arr = classify_solid_shape_body(
            &raw(
                json!({"id":"n","type":"rectangle","width":100.0,"height":80.0,
                "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}],"strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 2}],"cornerRadius":[0,0,0,8]}),
            ),
            ShapeType::Rect,
        )
        .unwrap();
        assert!(arr.stroke.is_some());
    }

    #[test]
    fn gradient_route_without_stroke_string_declines_sdf_so_knockout_stays_on() {
        // A gradient stroke is a paint route (the silhouette expands by its width)
        // but not a solid SDF ring, so the SDF body would carry `stroke = None` and
        // set opaque_sdf_covers_shadow — painting the expanded band as a shadow ring.
        // The body must tessellate instead (build_body_item's lane, knockout on).
        let node = rounded_rect(json!({
            "strokes": [{"id": "stroke-0", "width": 16.0, "paint": {"type": "linear",
                "start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 0},
                "stops": [{"offset": 0, "color": "#ff0000"}, {"offset": 1, "color": "#00ff00"}]}}],
            "effects": [{"id": "drop-shadow-0", "type": "drop-shadow", "color": "#000", "blur": 12.0}]
        }));
        assert!(classify_solid_shape_body(&node, ShapeType::Rect).is_none());
        // With a real solid stroke the body classifies as before (ring + knockout gate).
        let solid = rounded_rect(json!({
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": 16.0}],
            "effects": [{"id": "drop-shadow-0", "type": "drop-shadow", "color": "#000", "blur": 12.0}]
        }));
        assert!(classify_solid_shape_body(&solid, ShapeType::Rect).is_some());
        // And no stroke at all still classifies with stroke = None: nothing expands.
        let none = rounded_rect(json!({"effects": [{"id": "drop-shadow-0", "type": "drop-shadow", "color": "#000", "blur": 12.0}]}));
        let c = classify_solid_shape_body(&none, ShapeType::Rect).unwrap();
        assert!(c.stroke.is_none());
    }

    #[test]
    fn alpha_zero_stroke_keeps_shadow_knockout() {
        // A curved body with a non-zero-width but fully transparent stroke
        // (rgba(...,0)) stays on the SDF fast path: the ring resolves to alpha 0 (the
        // shader paints nothing). Crucially the stroke stays CLASSIFIED (Some, not
        // None) so the drop-shadow knockout is preserved: `drop_shadow_instance`
        // expands the silhouette by the PAINTED stroke width (an alpha-0 stroke string
        // is still a paint route, `painted_stroke_width`) regardless of alpha, so a
        // None stroke would skip the knockout and leak the expanded shadow as a halo.
        let c = classify_solid_shape_body(
            &rounded_rect(json!({"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"rgba(0,0,0,0)"},"width":6.0}]})),
            ShapeType::Rect,
        )
        .expect("transparent-stroked curved body stays SDF-eligible");
        let s = c
            .stroke
            .expect("alpha-0 stroke stays classified (ring drawn as nothing)");
        assert_eq!(s.color[3], 0.0);
        assert!(
            !stroke_preserves_shadow_cover(c.stroke),
            "an invisible stroke with raw width must keep the knockout (no halo)"
        );

        // Sanity: the same body with an OPAQUE stroke carries a visible ring.
        let opaque = classify_solid_shape_body(
            &rounded_rect(json!({"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"rgba(0,0,0,1)"},"width":6.0}]})),
            ShapeType::Rect,
        )
        .unwrap();
        assert_eq!(opaque.stroke.unwrap().color[3], 1.0);
    }

    #[test]
    fn stroke_align_codes() {
        for (align, code) in [("center", 0.0), ("inside", 1.0), ("outside", 2.0)] {
            let c = classify_solid_shape_body(
                &rounded_rect(json!({"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#000"},"width":4.0,"align":align}]})),
                ShapeType::Rect,
            )
            .unwrap();
            assert_eq!(c.stroke.unwrap().params[1], code, "align {align}");
        }
        // Missing strokeAlign defaults to center (0).
        let def = classify_solid_shape_body(
            &rounded_rect(json!({"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#000"},"width":4.0}]})),
            ShapeType::Rect,
        )
        .unwrap();
        assert_eq!(def.stroke.unwrap().params[1], 0.0);
    }

    #[test]
    fn stroke_rejects_sharp_dashed_pattern_gradient() {
        // Sharp rect + stroke → reject (curved-only).
        assert!(classify_solid_shape_body(
            &solid_rect(json!({"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#000"},"width":4.0}]})),
            ShapeType::Rect
        )
        .is_none());
        // cornerRadius 0 is still sharp.
        assert!(classify_solid_shape_body(
            &solid_rect(json!({"cornerRadius":0.0,"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#000"},"width":4.0}]})),
            ShapeType::Rect
        )
        .is_none());
        // Dashed stroke → reject even on a curved body.
        assert!(classify_solid_shape_body(
            &rounded_rect(json!({"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#000"},"width":4.0,"dash":[6.0,4.0]}]})),
            ShapeType::Rect
        )
        .is_none());
        // Empty dash array → NOT dashed, accepted.
        assert!(classify_solid_shape_body(
            &rounded_rect(json!({"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#000"},"width":4.0,"dash":[]}]})),
            ShapeType::Rect
        )
        .unwrap()
        .stroke
        .is_some());
        // Gradient stroke → reject (out of scope).
        assert!(classify_solid_shape_body(
            &rounded_rect(json!({"strokes":[{"id":"stroke-0","width":4.0,"paint":{"type":"linear","start":{"x":0,"y":0},"end":{"x":100,"y":0},"stops":[{"offset":0,"color":"#ff0000"},{"offset":1,"color":"#00ff00"}]}}]})),
            ShapeType::Rect
        )
        .is_none());
        // Pattern fill + stroke → reject (pattern fill is out of scope).
        assert!(classify_solid_shape_body(
            &rounded_rect(json!({"fills":[{"id":"fill-0","paint":{"type":"image","src":"a.png"}}],
                "strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#000"},"width":4.0}]})),
            ShapeType::Rect
        )
        .is_none());
    }

    #[test]
    fn stroke_phase_opaque_vs_translucent() {
        // Opaque stroke on opaque fill → opaque phase.
        let op = classify_solid_shape_body(
            &rounded_rect(json!({"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#000"},"width":4.0}]})),
            ShapeType::Rect,
        )
        .unwrap();
        assert!(op.is_opaque);

        // Translucent stroke (alpha < 1) forces the transparent phase even with
        // an opaque fill.
        let tr = classify_solid_shape_body(
            &rounded_rect(json!({"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"rgba(0,0,0,0.5)"},"width":4.0}]})),
            ShapeType::Rect,
        )
        .unwrap();
        assert!(!tr.is_opaque);
        assert_eq!(tr.stroke.unwrap().color[3], 0.5);
        // …but the fill is still opaque, so body_alpha (drives covers_shadow) is 1.
        assert!(tr.body_alpha >= 0.999);

        // Opacity < 1 scales the stroke alpha (stroke α × opacity).
        let dim = classify_solid_shape_body(
            &rounded_rect(json!({"opacity":0.5,"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#000"},"width":4.0}]})),
            ShapeType::Rect,
        )
        .unwrap();
        assert!(!dim.is_opaque);
        assert_eq!(dim.stroke.unwrap().color[3], 0.5);
    }

    #[test]
    fn gradient_fill_with_solid_stroke() {
        let lin = raw(json!({
            "id":"g","type":"rectangle","width":100.0,"height":80.0,"cornerRadius":20.0,
            "fills": [{"id": "fill-0", "paint": {"type": "linear", "start": {"x": 0, "y": 50}, "end": {"x": 100, "y": 50}, "stops": [{"offset": 0, "color": "#ff006e"}, {"offset": 1, "color": "#3a86ff"}]}}],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#222222"}, "width": 5, "align": "outside"}]
        }));
        let c = classify_gradient_shape(&lin, ShapeType::Rect).unwrap();
        assert_eq!(c.fill_type, 1);
        let s = c.stroke.expect("gradient body stroke ring");
        assert_eq!(s.params, [5.0, 2.0, 0.0, 0.0]); // outside

        // A gradient body on a SHARP rect with a stroke → reject (curved-only).
        let sharp = raw(json!({
            "id":"g","type":"rectangle","width":100.0,"height":80.0,
            "fills": [{"id": "fill-0", "paint": {"type": "linear", "start": {"x": 0, "y": 50}, "end": {"x": 100, "y": 50}, "stops": [{"offset": 0, "color": "#ff006e"}, {"offset": 1, "color": "#3a86ff"}]}}],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#222222"}, "width": 5}]
        }));
        assert!(classify_gradient_shape(&sharp, ShapeType::Rect).is_none());
    }

    #[test]
    fn stroke_packed_into_instance() {
        // Solid: stroke_color/stroke_params land at floats 40..44 / 44..48.
        let node =
            rounded_rect(json!({"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#ff0000"},"width":6.0,"align":"inside"}]}));
        let stroke = classify_solid_shape_body(&node, ShapeType::Rect)
            .unwrap()
            .stroke;
        let inst = build_solid_shape_instance(
            &node,
            ShapeType::Rect,
            (0.2, 0.5, 1.0),
            1.0,
            stroke,
            identity(),
            0.5,
            [0.0; 4],
        );
        assert_eq!(inst.stroke_color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(inst.stroke_params, [6.0, 1.0, 0.0, 0.0]);
        let floats: [f32; 48] = bytemuck::cast(inst);
        assert_eq!(&floats[40..44], &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(&floats[44..48], &[6.0, 1.0, 0.0, 0.0]);

        // No stroke → disabled fields (byte-zero, shader reads "no stroke").
        let plain = build_solid_shape_instance(
            &solid_rect(json!({})),
            ShapeType::Rect,
            (1.0, 1.0, 1.0),
            1.0,
            None,
            identity(),
            0.0,
            [0.0; 4],
        );
        assert_eq!(plain.stroke_color, [0.0; 4]);
        assert_eq!(plain.stroke_params, [0.0; 4]);

        // Gradient: stroke_color/stroke_params land at floats 80..84 / 84..88.
        let gnode = raw(json!({
            "id":"g","type":"rectangle","width":100.0,"height":80.0,"cornerRadius":20.0,
            "fills": [{"id": "fill-0", "paint": {"type": "linear", "start": {"x": 0, "y": 50}, "end": {"x": 100, "y": 50}, "stops": [{"offset": 0, "color": "#ff006e"}, {"offset": 1, "color": "#3a86ff"}]}}],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#00ff00"}, "width": 4, "align": "outside"}]
        }));
        let gstroke = classify_gradient_shape(&gnode, ShapeType::Rect)
            .unwrap()
            .stroke;
        let ginst = build_gradient_shape_instance(
            &gnode,
            ShapeType::Rect,
            1,
            gstroke,
            identity(),
            0.0,
            [0.0; 4],
        );
        assert_eq!(ginst.stroke_color, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(ginst.stroke_params, [4.0, 2.0, 0.0, 0.0]);
        let gfloats: [f32; 88] = bytemuck::cast(ginst);
        assert_eq!(&gfloats[80..84], &[0.0, 1.0, 0.0, 1.0]);
        assert_eq!(&gfloats[84..88], &[4.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn stroke_shadow_cover_only_for_opaque_center_or_outside() {
        let mk = |align: f32, alpha: f32| {
            Some(StrokeFields {
                color: [0.0, 0.0, 0.0, alpha],
                params: [8.0, align, 0.0, 0.0],
            })
        };
        // No stroke → an opaque fill still covers its (unexpanded) shadow.
        assert!(stroke_preserves_shadow_cover(None));

        // Opaque center/outside stroke reaches the expanded shadow band → covers.
        assert!(stroke_preserves_shadow_cover(mk(0.0, 1.0))); // center opaque
        assert!(stroke_preserves_shadow_cover(mk(2.0, 1.0))); // outside opaque

        // INSIDE stroke never paints the outer half-stroke band, so it must keep the
        // knockout (else the silhouette-expanded drop shadow leaks as a halo) — even
        // when the stroke itself is opaque. This is the blocker case.
        assert!(!stroke_preserves_shadow_cover(mk(1.0, 1.0))); // inside opaque
        assert!(!stroke_preserves_shadow_cover(mk(1.0, 0.4))); // inside translucent

        // Translucent center/outside stroke does NOT fully occlude (band beyond fill).
        assert!(!stroke_preserves_shadow_cover(mk(0.0, 0.4))); // center translucent
        assert!(!stroke_preserves_shadow_cover(mk(2.0, 0.4))); // outside translucent
    }

    fn identity() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    }
}
