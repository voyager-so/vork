//! Backdrop orchestration port (plan side): eligibility, target-pixel param
//! mapping with production clamps, sampling-region math, and coverage
//! geometry — producing `seams::BackdropSplit` items for the executor.
//!
//! Normative sources:
//! - `core/scene/nodes.ts:1938-1962` (`hasActiveBackgroundBlur`)
//! - `renderers/webgpu/utils/blurEligibility.ts:19-62` (`isNodeBlurEligible`)
//! - `renderers/webgpu/rendering/pipeline/WebGPURenderOrchestrator.ts:73-110,
//!   1165-1561` (`getShapeBackdropBlurInfo`, normalize helpers,
//!   `buildCoverageVerticesTargetPx`, `getNodeLocalExtent`)
//! - `renderers/webgpu/utils/projectionUtils.ts:9-46`
//! - `renderers/webgpu/rendering/systems/effects/WebGPUBackdropBlurRenderer.ts:
//!   460-605, 2191-2223` (sample-region intersect/expand/quantize)
//!
//! Clamps (px space, render-semantics §4.7): blur ≤ 200, pixelate ≤ 400,
//! dither cell ≤ 256. `linear` / `pixelate-linear` are stored-but-inert
//! (quirks Q-12/Q-31) — they never produce a split.

use moda_scene::types::RawNode;
use moda_scene::Scene;
use moda_scene::value::Value;

use crate::seams::{BackdropOp, BackdropSplit, BlurRamp, CoverageSource, INFINITE_CLIP};
use crate::tess::{self, moda_engine_path_types as pt};

/// The zoom-scaled blur reach (3σ) never exceeds this many target pixels.
const MAX_BLUR_RADIUS_PX: f64 = 300.0;
/// A blur radius (σ = R / 2, Figma's convention) as the kernels' 3σ reach.
const REACH: f64 = moda_scene::effect::BLUR_REACH_PER_RADIUS;
const MAX_PIXELATE_SIZE_PX: f64 = 400.0;
const MAX_DITHER_PIXEL_SIZE_PX: f64 = 256.0;
/// `device.limits?.maxTextureDimension2D ?? 8192` fallback
/// (BackdropBlurRenderer:580).
const MAX_TEXTURE_DIM: f64 = 8192.0;

const BACKDROP_CAPABLE: [&str; 6] = [
    "rectangle",
    "container",
    "ellipse",
    "star",
    "polygon",
    "path",
];

/// JS `Math.min` / `Math.max`: NaN-propagating (Rust's f64::min/max return
/// the non-NaN operand — that would silently "fix" malformed params the
/// production renderer rejects downstream via its finiteness guards).
fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}
fn js_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Rect {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) w: f64,
    pub(crate) h: f64,
}

impl Rect {
    pub(crate) fn finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.w.is_finite() && self.h.is_finite()
    }
}

impl Rect {
    pub(crate) fn intersect(&self, other: &Rect) -> Option<Rect> {
        intersect(self, other)
    }
}

/// `intersectRect` (BackdropBlurRenderer:2207-2223) — NaN-rejecting.
fn intersect(a: &Rect, b: &Rect) -> Option<Rect> {
    if !a.finite() || !b.finite() {
        return None;
    }
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    let right = (a.x + a.w).min(b.x + b.w);
    let bottom = (a.y + a.h).min(b.y + b.h);
    if right <= x || bottom <= y {
        return None;
    }
    Some(Rect {
        x,
        y,
        w: right - x,
        h: bottom - y,
    })
}

/// Page-level context for one export plan (export config: zoom 1, camera at
/// page origin, target = ceil(page × scale)).
#[derive(Clone, Copy)]
pub struct PlanCtx<'a> {
    pub scene: &'a Scene,
    pub width_px: u32,
    pub height_px: u32,
    /// Canvas-space page clip from the plan builder ([x,y,w,h] or
    /// `INFINITE_CLIP`).
    pub page_clip: [f32; 4],
    /// f32 view-projection (same matrix the executor uploads).
    vp: [f32; 16],
    /// The target starts transparent (a transparent export): multiply blends through the split too
    /// (`BlendMode::split_id_over`).
    pub transparent_target: bool,
    /// Rect / ellipse bodies keep the analytic SDF path under this context (a page whose only splits are paint
    /// stacks'). Backdrop effects and node blend modes turn it off, as before.
    pub keep_shape_sdf: bool,
    /// The shader clock (seconds): agent shader layer effects' `time()` (`layer_blur::plan`).
    pub shader_time_s: f64,
}

impl<'a> PlanCtx<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scene: &'a Scene,
        width_px: u32,
        height_px: u32,
        page_x: f64,
        page_y: f64,
        page_w: f64,
        page_h: f64,
        page_clip: [f32; 4],
    ) -> PlanCtx<'a> {
        PlanCtx {
            scene,
            width_px,
            height_px,
            page_clip,
            vp: crate::frame::view_projection(page_x, page_y, 1.0, page_w, page_h),
            transparent_target: false,
            keep_shape_sdf: false,
            shader_time_s: 0.0,
        }
    }

    /// A backdrop context that projects to the CAMERA frame's
    /// target (viewport × dpr under the editor view-projection) instead of the
    /// per-page export target. Used when a backdrop page is composited inside the
    /// combined live-viewport camera frame: the split's sampling region, node
    /// rect, and coverage geometry are then in camera target pixels (scaling with
    /// zoom), so the effect lands crisp at the on-screen position instead of being
    /// computed for a page-local raster that the snapshot path would upscale.
    /// `page_clip` stays the page's canvas-space bounds (a backdrop node is still
    /// clipped to its page); only the projection target differs from `new`.
    pub fn new_camera(
        scene: &'a Scene,
        width_px: u32,
        height_px: u32,
        page_clip: [f32; 4],
        vp: [f32; 16],
    ) -> PlanCtx<'a> {
        PlanCtx {
            scene,
            width_px,
            height_px,
            page_clip,
            vp,
            transparent_target: false,
            keep_shape_sdf: false,
            shader_time_s: 0.0,
        }
    }

    /// The same context with rect / ellipse bodies on the SDF path (`keep_shape_sdf`).
    pub fn keeping_shape_sdf(mut self, keep: bool) -> PlanCtx<'a> {
        self.keep_shape_sdf = keep;
        self
    }

    /// The same context at shader clock `t` (seconds).
    pub fn at_shader_time(mut self, t: f64) -> PlanCtx<'a> {
        self.shader_time_s = t;
        self
    }

    /// The same context over a target that starts transparent (`transparent_target`).
    pub fn over_transparent(mut self, transparent: bool) -> PlanCtx<'a> {
        self.transparent_target = transparent;
        self
    }

    /// `projectPointToTargetPixels` (projectionUtils.ts:9-25). The production
    /// matrix is a Float32Array; arithmetic runs in f64 on f32 values —
    /// mirrored exactly.
    pub(crate) fn project_point(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let m = &self.vp;
        let clip_x = m[0] as f64 * x + m[4] as f64 * y + m[12] as f64;
        let clip_y = m[1] as f64 * x + m[5] as f64 * y + m[13] as f64;
        let clip_w = m[3] as f64 * x + m[7] as f64 * y + m[15] as f64;
        if clip_w.abs() < 1e-6 {
            return None;
        }
        let ndc_x = clip_x / clip_w;
        let ndc_y = clip_y / clip_w;
        Some((
            (ndc_x * 0.5 + 0.5) * self.width_px as f64,
            (1.0 - (ndc_y * 0.5 + 0.5)) * self.height_px as f64,
        ))
    }

    /// `projectRectToTargetPixels` — AABB of the four projected corners.
    pub(crate) fn project_rect(&self, r: &Rect) -> Option<Rect> {
        let corners = [
            self.project_point(r.x, r.y)?,
            self.project_point(r.x + r.w, r.y)?,
            self.project_point(r.x, r.y + r.h)?,
            self.project_point(r.x + r.w, r.y + r.h)?,
        ];
        let min_x = corners.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
        let max_x = corners
            .iter()
            .map(|c| c.0)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y = corners.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
        let max_y = corners
            .iter()
            .map(|c| c.1)
            .fold(f64::NEG_INFINITY, f64::max);
        Some(Rect {
            x: min_x,
            y: min_y,
            w: max_x - min_x,
            h: max_y - min_y,
        })
    }

    /// `computeCanvasTargetScale` (WebGPURenderOrchestrator.ts:83-95).
    pub(crate) fn canvas_target_scale(&self) -> f64 {
        let (Some(o), Some(xa), Some(ya)) = (
            self.project_point(0.0, 0.0),
            self.project_point(1.0, 0.0),
            self.project_point(0.0, 1.0),
        ) else {
            return 1.0;
        };
        let xs = (xa.0 - o.0).hypot(xa.1 - o.1);
        let ys = (ya.0 - o.0).hypot(ya.1 - o.1);
        let s = xs.min(ys);
        if s.is_finite() && s > 0.0 {
            s
        } else {
            1.0
        }
    }
}

// ---------------------------------------------------------------------------
// Predicates
// ---------------------------------------------------------------------------

/// The node's backdrop effects: every visible backdrop entry of `effects`, in
/// list order (`effect::backdrops_json`; a hidden entry renders as if absent).
/// They stack: each plans its own split, sampling what the one before it
/// composited. Read as raw JSON: the activity test and the param clamps copy
/// the TS `??` / NaN quirks of missing params.
fn blur_objs(node: &RawNode) -> Vec<&moda_scene::value::Map> {
    moda_scene::effect::backdrops_json(node)
}

fn obj_f64(o: &moda_scene::value::Map, key: &str) -> Option<f64> {
    o.get(key).and_then(Value::as_f64)
}

/// A background blur's progressive ramp (`ProgressiveBlur`), when it has one.
fn progressive(o: &moda_scene::value::Map) -> Option<moda_scene::effect::ProgressiveBlur> {
    o.get("progressive")
        .and_then(|p| moda_scene::value::from_value(p.clone()).ok())
}

/// The ramp's start radius (canvas px), or NaN without a ramp.
fn progressive_start_radius(o: &moda_scene::value::Map) -> f64 {
    progressive(o).map_or(f64::NAN, |p| p.start_radius)
}

/// `hasActiveBackgroundBlur` (`core/scene/nodes.ts:1938-1962`). The cheap
/// node-level predicate that also drives the page-level MSAA-1 forcing
/// (quirk Q-02) — a clamped-to-nothing radius still counts as active.
pub fn has_active_background_blur(node: &RawNode) -> bool {
    BACKDROP_CAPABLE.contains(&node.node_type()) && blur_objs(node).into_iter().any(backdrop_entry_is_active)
}

/// One backdrop entry's share of [`has_active_background_blur`].
fn backdrop_entry_is_active(blur: &moda_scene::value::Map) -> bool {
    let Some(ty) = blur.get("type").and_then(Value::as_str) else {
        return false;
    };
    if moda_scene::backdrop_kinds::kind(ty).is_some() {
        return moda_scene::backdrop_kinds::filter_is_active(blur);
    }
    match ty {
        "background-blur" => {
            obj_f64(blur, "radius").unwrap_or(f64::NAN) > 0.0
                || progressive_start_radius(blur) > 0.0
        }
        "pixelate" => obj_f64(blur, "pixelSize").unwrap_or(f64::NAN) > 0.0,
        "dither" => obj_f64(blur, "pixelSize").unwrap_or(1.0) > 0.0,
        // glass (the TS default arm)
        _ => {
            let f = |k: &str| obj_f64(blur, k).unwrap_or(f64::NAN);
            f("frost") > 0.0
                || (f("refraction") > 0.0 && f("depth") > 0.0)
                || f("lightIntensity") > 0.0
                || (f("dispersion") > 0.0 && f("depth") > 0.0)
        }
    }
}

/// Does any node on the page carry an active backdrop effect? Drives the
/// sample_count-1 / opaque-white forcing (`WebGPUExportCoordinator.ts:786-795`,
/// `WebGPURenderer.ts:2548-2583`).
pub fn page_has_active_backdrop(scene: &Scene, page_id: &str) -> bool {
    scene
        .flattened_z_order_for_page(page_id)
        .iter()
        .filter_map(|id| scene.effective_node(id))
        .any(has_active_background_blur)
}

/// A layer as an export warning names it: `"Name" (id)`, or the id alone.
fn layer_label(node: &RawNode) -> String {
    match node.json_field("name").and_then(Value::as_str).filter(|n| !n.is_empty()) {
        Some(name) => format!("\"{name}\" ({})", node.id()),
        None => node.id().to_string(),
    }
}

/// Code of the [`note_backdrop_aliases_frame`] warning.
pub const BACKDROP_ALIASES_FRAME: &str = "backdrop_aliases_frame";
/// Code of the backdrop blur clamp warning ([`backdrop_split_for`]).
pub const BLUR_RADIUS_CLAMPED: &str = "blur_radius_clamped";

/// Quirk Q-02 made visible: a surface (page, frame, the canvas root) that holds
/// an active backdrop effect renders at sample_count 1, so every edge on it
/// loses MSAA. When the caller asked for MSAA (`sample_count > 1`), record a
/// `backdrop_aliases_frame` render-quality event naming the first such layer
/// in `z_order` (the resource report's `render_warnings`, an export warning).
pub(crate) fn note_backdrop_aliases_frame(scene: &Scene, z_order: &[String], sample_count: u32) {
    if sample_count <= 1 {
        return;
    }
    let Some(node) = z_order
        .iter()
        .filter_map(|id| scene.effective_node(id))
        .find(|n| has_active_background_blur(n))
    else {
        return;
    };
    let layer = layer_label(node);
    moda_assets::report::record(moda_assets::report::RuntimeEvent::RenderQuality {
        code: BACKDROP_ALIASES_FRAME.to_string(),
        node_id: node.id().to_string(),
        message: format!(
            "layer {layer} holds a backdrop effect, so the whole frame it is in renders without anti-aliasing (no MSAA): \
             every edge in that frame is aliased"
        ),
    });
}

/// A backdrop blur whose 3σ reach (`radius × REACH × scale`) passes
/// [`MAX_BLUR_RADIUS_PX`] draws at the clamp: record a `blur_radius_clamped`
/// render-quality event with the asked radius and the one it draws as at this
/// scale (the clamp is in target px, so that depends on the export scale).
fn note_blur_radius_clamped(node: &RawNode, what: &str, radius: f64, effective_scale: f64) {
    if !(radius * REACH * effective_scale > MAX_BLUR_RADIUS_PX) {
        return;
    }
    let effective = MAX_BLUR_RADIUS_PX / (REACH * effective_scale);
    let layer = layer_label(node);
    let round = |v: f64| (v * 10.0).round() / 10.0;
    moda_assets::report::record(moda_assets::report::RuntimeEvent::RenderQuality {
        code: BLUR_RADIUS_CLAMPED.to_string(),
        node_id: node.id().to_string(),
        message: format!(
            "layer {layer}: {what} {} draws as {} at this scale (a blur's reach is capped at {MAX_BLUR_RADIUS_PX} output px); \
             any larger radius looks the same",
            round(radius),
            round(effective),
        ),
    });
}

/// The camera frame also renders freeform canvas-root (off-page)
/// nodes, so a backdrop-blur node on the pasteboard must split the pass too.
/// Mirror of `page_has_active_backdrop` over the canvas-root z-order.
pub fn canvas_root_has_active_backdrop(scene: &Scene) -> bool {
    scene
        .flattened_z_order_for_canvas()
        .iter()
        .filter_map(|id| scene.effective_node(id))
        .any(has_active_background_blur)
}

/// `--isolate` × backdrop: whether an isolated render of `z_order` (a page's
/// or the canvas root's flattened z-order) draws a backdrop effect that would
/// sample the wrong thing. An isolated render draws only the isolated
/// subtrees, so a backdrop node samples only what its subtree drew behind it.
/// That is right when the subtree is a clipping frame the node sits inside (a
/// frame export: Figma's semantics, the frame's own content) and wrong
/// otherwise — isolating the blurred node itself, or a group around it, would
/// make a plate that claims the element while blurring nothing that sits
/// behind it on the canvas. A backdrop outside every isolated subtree doesn't
/// render at all, so it is no conflict. Empty `isolate`: never a conflict.
pub fn isolated_backdrop_conflict(
    scene: &Scene,
    z_order: &[String],
    isolate: &std::collections::HashSet<&str>,
) -> bool {
    if isolate.is_empty() {
        return false;
    }
    z_order.iter().any(|id| {
        let Some(node) = scene.effective_node(id) else {
            return false;
        };
        if !has_active_background_blur(node) || !scene.is_node_isolated(id, isolate) {
            return false;
        }
        // The outermost isolated id on the node's chain is the subtree that renders.
        let root = std::iter::once(id.clone())
            .chain(scene.transform_parent_chain(id))
            .filter(|a| isolate.contains(a.as_str()))
            .last();
        match root {
            Some(root) if &root != id => !scene
                .effective_node(&root)
                .is_some_and(|r| r.node_type() == "container" && r.truthy("clipContent")),
            _ => true,
        }
    })
}

/// A clip-mask group (`clipMaskNodeId`) among the node's ancestors. Backdrop
/// blur declines under one; a layer capture's composite takes the group's mask
/// instead (`frame::capture_composite_mask`). A clipping CONTAINER (every Design board
/// clips) is not a mask owner here: its clip is a (rounded) rect, which the
/// backdrop composite applies itself (`clipping_containers_clip`) and layer
/// blur checks against its capture.
pub(crate) fn has_ancestor_clip_mask_group(scene: &Scene, node_id: &str) -> bool {
    scene.transform_parent_chain(node_id).iter().any(|pid| {
        scene
            .effective_node(pid)
            .map(|p| p.node_type() == "group" && p.truthy("clipMaskNodeId"))
            .unwrap_or(false)
    })
}

/// One clipping container's clip in canvas space: its box and per-corner radii
/// [tl, tr, br, bl] (canvas px).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ContainerClip {
    pub(crate) rect: Rect,
    pub(crate) radii: [f64; 4],
}

/// The clips of every clipping container (`clipContent`) above the node,
/// innermost first, or `None` when one of them can't be expressed as an
/// axis-aligned (rounded) rect — a rotated or sheared container — or has no
/// box. Radii are the container's corner radii through its canvas scale,
/// clamped to its half-extent (a smoothed corner is read as its circular
/// radius, which is close).
pub(crate) fn clipping_containers_clip(scene: &Scene, node_id: &str) -> Option<Vec<ContainerClip>> {
    const EPS: f64 = 1e-6;
    let mut out = Vec::new();
    for pid in scene.transform_parent_chain(node_id) {
        let Some(p) = scene.effective_node(&pid) else {
            continue;
        };
        if p.node_type() != "container" || !p.truthy("clipContent") {
            continue;
        }
        let a = scene.node_canvas_affine(&pid)?;
        if a.b.abs() > EPS || a.c.abs() > EPS {
            return None;
        }
        let rect = node_canvas_aabb(scene, p)?;
        if !rect.finite() {
            return None;
        }
        let limit = (rect.w.min(rect.h) * 0.5).max(0.0);
        let s = a.a.abs().min(a.d.abs());
        let radii = corner_radii(p).map(|r| (r.max(0.0) * s).min(limit));
        out.push(ContainerClip { rect, radii });
    }
    Some(out)
}

/// Whether the canvas-space `region` stays clear of every rounded corner of
/// `clip` (so clipping to its plain box is exact there).
pub(crate) fn region_clear_of_corners(region: &Rect, clip: &ContainerClip) -> bool {
    let c = &clip.rect;
    let [tl, tr, br, bl] = clip.radii;
    let square = |x: f64, y: f64, s: f64| Rect { x, y, w: s, h: s };
    [
        square(c.x, c.y, tl),
        square(c.x + c.w - tr, c.y, tr),
        square(c.x + c.w - br, c.y + c.h - br, br),
        square(c.x, c.y + c.h - bl, bl),
    ]
    .iter()
    .all(|sq| sq.w <= 0.0 || intersect(sq, region).is_none())
}

/// `isNodeBlurEligible` (`blurEligibility.ts:19-62`).
fn is_node_blur_eligible(scene: &Scene, node: &RawNode) -> bool {
    if !has_active_background_blur(node) {
        return false;
    }
    if node.node_type() == "ellipse" {
        if node.f64("startAngle").map(|v| v != 0.0).unwrap_or(false) {
            return false;
        }
        if node.f64("sweepAngle").map(|v| v != 360.0).unwrap_or(false) {
            return false;
        }
        if node
            .f64("innerRadiusRatio")
            .map(|v| v > 0.0)
            .unwrap_or(false)
        {
            return false;
        }
    }
    if node
        .str("blendMode")
        .map(|b| !b.is_empty() && b != "normal")
        .unwrap_or(false)
    {
        return false;
    }
    if node.f64("opacity").unwrap_or(1.0) < 1.0 {
        return false;
    }
    // A distort-warped node declines background blur entirely — the
    // blur coverage silhouette (CoverageSource::Rect/Ellipse half-extents,
    // build_coverage_vertices_target_px) is built from UNWARPED local
    // geometry, so keeping it would cut an unwarped blur hole behind a warped
    // panel. Same decline posture as the SDF silhouette shadow (shadow.rs);
    // warping the coverage silhouette is a polish follow-up. NOTE the
    // page-level flags (`page_has_active_backdrop` → MSAA-1 / opaque-export
    // forcing, Q-02) deliberately still fire: they run the cheap TS-parity
    // `hasActiveBackgroundBlur` predicate, exactly as they do for every other
    // ineligibility reason here (opacity < 1, ancestor mask, sheared
    // transform) — do NOT make them distort-aware, that would diverge from
    // the TS function they mirror.
    if crate::paint::node_distort(node).is_some() {
        return false;
    }
    // A clip-mask group declines (masking the composite is P3). Clipping
    // containers are applied by `compute_backdrop_split`: their (rounded)
    // box clips the sampled region and the composite.
    if has_ancestor_clip_mask_group(scene, node.id()) {
        return false;
    }
    let Some(affine) = scene.node_canvas_affine(node.id()) else {
        return false;
    };
    // Reject sheared absolute transforms (blurEligibility.ts:41-60).
    let (a, b, c, d) = (affine.a, affine.b, affine.c, affine.d);
    let col1 = a * a + b * b;
    let col2 = c * c + d * d;
    let denom = col1.max(col2);
    if denom > 1e-12 && (a * c + b * d).abs() / denom > 1e-4 {
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// `boundsFromPath` (`utils/geometryHelpers.ts:692-852`): move/line + cubic &
/// quadratic extrema; an arc through the cubic Béziers it is drawn with.
fn bounds_from_cmds(cmds: &[pt::PathCmd]) -> Option<Rect> {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut cur = (0.0f64, 0.0f64);
    let upd = |x: f64, y: f64, mn_x: &mut f64, mn_y: &mut f64, mx_x: &mut f64, mx_y: &mut f64| {
        *mn_x = mn_x.min(x);
        *mn_y = mn_y.min(y);
        *mx_x = mx_x.max(x);
        *mx_y = mx_y.max(y);
    };

    fn cubic_extrema(p0: f64, p1: f64, p2: f64, p3: f64) -> Vec<f64> {
        let a = -3.0 * p0 + 9.0 * p1 - 9.0 * p2 + 3.0 * p3;
        let b = 6.0 * p0 - 12.0 * p1 + 6.0 * p2;
        let c = -3.0 * p0 + 3.0 * p1;
        let mut out = Vec::new();
        if a.abs() < 1e-10 {
            if b.abs() > 1e-10 {
                let t = -c / b;
                if (0.0..=1.0).contains(&t) {
                    out.push(t);
                }
            }
        } else {
            let disc = b * b - 4.0 * a * c;
            if disc >= 0.0 {
                let s = disc.sqrt();
                for t in [(-b + s) / (2.0 * a), (-b - s) / (2.0 * a)] {
                    if (0.0..=1.0).contains(&t) {
                        out.push(t);
                    }
                }
            }
        }
        out
    }
    fn cubic_eval(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
        let mt = 1.0 - t;
        mt * mt * mt * p0 + 3.0 * mt * mt * t * p1 + 3.0 * mt * t * t * p2 + t * t * t * p3
    }
    fn quad_extrema(p0: f64, p1: f64, p2: f64) -> Vec<f64> {
        let denom = p0 - 2.0 * p1 + p2;
        if denom.abs() < 1e-10 {
            return Vec::new();
        }
        let t = (p0 - p1) / denom;
        if (0.0..=1.0).contains(&t) {
            vec![t]
        } else {
            Vec::new()
        }
    }
    fn quad_eval(p0: f64, p1: f64, p2: f64, t: f64) -> f64 {
        let mt = 1.0 - t;
        mt * mt * p0 + 2.0 * mt * t * p1 + t * t * p2
    }

    for c in cmds {
        match c {
            pt::PathCmd::Move { x, y } => {
                cur = (*x, *y);
                upd(cur.0, cur.1, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
            }
            pt::PathCmd::Line { x, y } => {
                upd(*x, *y, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                cur = (*x, *y);
            }
            pt::PathCmd::Curve { x, y, cp1, cp2 } => {
                upd(cur.0, cur.1, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                upd(*x, *y, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                for t in cubic_extrema(cur.0, cp1.0, cp2.0, *x) {
                    let ex = cubic_eval(cur.0, cp1.0, cp2.0, *x, t);
                    let ey = cubic_eval(cur.1, cp1.1, cp2.1, *y, t);
                    upd(ex, ey, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                }
                for t in cubic_extrema(cur.1, cp1.1, cp2.1, *y) {
                    let ex = cubic_eval(cur.0, cp1.0, cp2.0, *x, t);
                    let ey = cubic_eval(cur.1, cp1.1, cp2.1, *y, t);
                    upd(ex, ey, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                }
                cur = (*x, *y);
            }
            pt::PathCmd::Quadratic { x, y, cp } => {
                upd(cur.0, cur.1, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                upd(*x, *y, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                for t in quad_extrema(cur.0, cp.0, *x) {
                    let ex = quad_eval(cur.0, cp.0, *x, t);
                    let ey = quad_eval(cur.1, cp.1, *y, t);
                    upd(ex, ey, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                }
                for t in quad_extrema(cur.1, cp.1, *y) {
                    let ex = quad_eval(cur.0, cp.0, *x, t);
                    let ey = quad_eval(cur.1, cp.1, *y, t);
                    upd(ex, ey, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                }
                cur = (*x, *y);
            }
            arc @ pt::PathCmd::Arc { x, y, .. } => {
                // The arc as the Béziers it is drawn with (`tess::path_bounds`), so a bulge past its endpoints is in
                // the bounds: a layer blur's capture taken over the endpoint hull cut the node's own body.
                let from = pt::PathCmd::Move { x: cur.0, y: cur.1 };
                if let Some([x0, y0, x1, y1]) = tess::path_bounds(&[from, arc.clone()]) {
                    upd(x0, y0, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                    upd(x1, y1, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                }
                upd(*x, *y, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                cur = (*x, *y);
            }
        }
    }
    if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
        return None;
    }
    Some(Rect {
        x: min_x,
        y: min_y,
        w: max_x - min_x,
        h: max_y - min_y,
    })
}

/// `getNodeLocalExtent` (WebGPURenderOrchestrator.ts:1451-1465) and the local
/// rect the canvas AABB is taken over.
pub(crate) fn node_local_rect(node: &RawNode) -> Option<Rect> {
    if node.node_type() == "path" {
        let cmds = node
            .points()
            .map(pt::cmds_from_path_points)
            .unwrap_or_default();
        let b = bounds_from_cmds(&cmds)?;
        if b.w <= 0.0 || b.h <= 0.0 {
            return None;
        }
        Some(b)
    } else {
        Some(Rect {
            x: 0.0,
            y: 0.0,
            w: node.f64("width").unwrap_or(0.0),
            h: node.f64("height").unwrap_or(0.0),
        })
    }
}

/// Canvas-space rotated AABB (`scene.getNodeBoundsCanvas` semantics: the
/// canvas affine applied to the local rect corners). NOTE: no
/// distort branch here on purpose — `is_node_blur_eligible` declines any
/// distort-warped node (the blur coverage silhouette is unwarped), and BOTH
/// callers sit behind a distort gate (this backdrop gate, and the layer-blur
/// eligibility in `layer_blur::plan::is_node_layer_blur_eligible` —
/// keep any new caller gated the same way). When the polish follow-up warps
/// the coverage silhouette, re-add the warped-corner AABB (map the warped
/// local corners through the affine, THEN take the AABB — see
/// effects/mask.rs `node_canvas_aabb` for the exact shape).
pub(crate) fn node_canvas_aabb(scene: &Scene, node: &RawNode) -> Option<Rect> {
    let a = scene.node_canvas_affine(node.id())?;
    let local = node_local_rect(node)?;
    let apply = |x: f64, y: f64| (a.a * x + a.c * y + a.e, a.b * x + a.d * y + a.f);
    let corners = [
        apply(local.x, local.y),
        apply(local.x + local.w, local.y),
        apply(local.x, local.y + local.h),
        apply(local.x + local.w, local.y + local.h),
    ];
    let min_x = corners.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
    let max_x = corners
        .iter()
        .map(|c| c.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = corners.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
    let max_y = corners
        .iter()
        .map(|c| c.1)
        .fold(f64::NEG_INFINITY, f64::max);
    Some(Rect {
        x: min_x,
        y: min_y,
        w: max_x - min_x,
        h: max_y - min_y,
    })
}

/// `getRoundedRectCornerRadii` (`core/scene/nodes.ts:1819-1828`):
/// scalar-or-array `cornerRadius` → [tl, tr, br, bl].
fn corner_radii(node: &RawNode) -> [f64; 4] {
    match node.json_field("cornerRadius") {
        Some(Value::Array(a)) => {
            let g = |i: usize| a.get(i).and_then(Value::as_f64).unwrap_or(0.0);
            [g(0), g(1), g(2), g(3)]
        }
        Some(v) => {
            let r = v.as_f64().unwrap_or(0.0);
            [r, r, r, r]
        }
        None => [0.0; 4],
    }
}

/// `normalizePixelatePixelSizeForTarget` (WebGPURenderOrchestrator.ts:96-102).
fn normalize_pixelate(raw: f64) -> Option<f64> {
    if !raw.is_finite() {
        return None;
    }
    if raw < 1.0 {
        if raw >= 1.0 - 1e-3 {
            Some(1.0)
        } else {
            None
        }
    } else {
        Some(raw)
    }
}

/// `normalizeDitherPixelSizeForTarget` (WebGPURenderOrchestrator.ts:104-110).
fn normalize_dither(raw: f64) -> Option<f64> {
    if !raw.is_finite() || raw <= 0.0 {
        return None;
    }
    if raw >= 1.0 - 1e-3 {
        Some(raw.max(1.0))
    } else {
        Some(raw)
    }
}

/// `ditherPatternToRendererMode` (WebGPURenderOrchestrator.ts:73-81).
fn dither_pattern_mode(pattern: Option<&str>) -> u32 {
    match pattern {
        Some("bayer-4x4") => 1,
        Some("floyd-steinberg") => 2,
        Some("atkinson") => 3,
        _ => 0,
    }
}

/// The sampled region in whole target texels: `sample` rounded outward, but
/// never past an edge of `clip` (the page, target and clipping frames). The
/// blur samples the region clamp-to-edge, so a texel row past the clip is
/// spread over the whole kernel: f32 projection noise (a frame edge at
/// 240.0000036) rounded outward took in the row beyond the frame, which in a
/// transparent frame export is (0,0,0,0), and a large radius then faded the
/// blur toward that corner and let the sharp content show through (#1679). A
/// clip edge within `EPS` of a texel boundary snaps to it; a fractional one
/// rounds inward, so the region holds only texels the clip fully covers.
fn quantize_sample(sample: &Rect, clip: &Rect) -> Rect {
    const EPS: f64 = 1e-3;
    let outward = Rect {
        x: sample.x.floor(),
        y: sample.y.floor(),
        w: sample.w.ceil().max(1.0),
        h: sample.h.ceil().max(1.0),
    };
    let x0 = outward.x.max((clip.x - EPS).ceil());
    let y0 = outward.y.max((clip.y - EPS).ceil());
    let x1 = (outward.x + outward.w).min((clip.x + clip.w + EPS).floor());
    let y1 = (outward.y + outward.h).min((clip.y + clip.h + EPS).floor());
    if x1 - x0 >= 1.0 && y1 - y0 >= 1.0 {
        Rect {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
        }
    } else {
        // A clip thinner than a texel: keep the outward region.
        outward
    }
}

/// `computeSampleExpansion` (BackdropBlurRenderer:2191-2205).
fn sample_expansion(op: &BackdropOp) -> f64 {
    match op {
        BackdropOp::Glass {
            blur_px,
            refraction_px,
            dispersion_px,
            ..
        } => js_max(
            1.0,
            (*blur_px as f64 + *refraction_px as f64 + *dispersion_px as f64).ceil() + 1.0,
        ),
        // Displace reads up to its largest offset past the node; the others read the pixel they write.
        BackdropOp::Filter(f) if f.reach_px > 0.0 => js_max(1.0, (f.reach_px as f64).ceil() + 1.0),
        BackdropOp::Filter(_)
        | BackdropOp::Adjust(_)
        | BackdropOp::Pixelate { .. }
        | BackdropOp::Dither { .. }
        | BackdropOp::BlendBegin
        | BackdropOp::GroupBegin
        | BackdropOp::BlendEnd { .. }
        | BackdropOp::StackEnd { .. } => 0.0,
        BackdropOp::Blur {
            radius_px, ramp, ..
        } => {
            let widest = ramp.map_or(*radius_px, |r| radius_px.max(r.start_radius_px));
            js_max(1.0, (widest as f64).ceil())
        }
    }
}

/// A colour param as straight sRGB + alpha (the engine's colour parser; an
/// unparseable one is transparent, so it paints nothing).
fn filter_color(s: &str) -> [f32; 4] {
    crate::color::parse_color(Some(s))
        .map(|c| [c.r as f32, c.g as f32, c.b as f32, c.a as f32])
        .unwrap_or([0.0; 4])
}

/// A backdrop filter kind's op (`moda_scene::backdrop_kinds`, the schema's
/// defaults and clamps) with its lengths in target px: `scale` is the canvas →
/// target scale, `half_px` the node's local half-size in target px. `None`
/// when the entry is inert. The uniform layout is `backdrop-filter.wgsl`'s,
/// from `header` on.
fn filter_op(
    kind_name: &str,
    entry: &moda_scene::value::Map,
    scale: f64,
    half_px: [f64; 2],
) -> Option<BackdropOp> {
    use crate::seams::{BackdropFilter, BACKDROP_FILTER_UNIFORM_FLOATS};
    let kind = moda_scene::backdrop_kinds::kind(kind_name)?;
    if !kind.is_active(entry) {
        return None;
    }
    if moda_scene::backdrop_kinds::is_adjustment(kind_name) {
        return adjust_op(kind, entry);
    }
    let n = |name: &str| kind.number(entry, name);
    let mut u = [0f32; BACKDROP_FILTER_UNIFORM_FLOATS];
    // header 0..4, p0 4..8, p1 8..12, p2 12..16 (the clip box, `backdrop_split_for`), p3 / p4 (the rounded
    // clip, the executor), colors 24..56, stops 56..64.
    const P0: usize = 4;
    const P1: usize = 8;
    const COLORS: usize = 24;
    const STOPS: usize = 56;
    let mut reach_px = 0.0f32;
    let id = match kind_name {
        "grain" => {
            u[P0] = n("amount") as f32;
            u[P0 + 1] = (n("size") * scale) as f32;
            u[P0 + 2] = if kind.text(entry, "mode") == "color" { 1.0 } else { 0.0 };
            u[P0 + 3] = n("seed") as f32;
            0
        }
        "vignette" => {
            u[P0] = n("amount") as f32;
            u[P0 + 1] = n("midpoint") as f32;
            u[P0 + 2] = n("roundness") as f32;
            u[P0 + 3] = n("feather") as f32;
            u[COLORS..COLORS + 4].copy_from_slice(&filter_color(kind.text(entry, "color")));
            1
        }
        "glaze" => {
            u[1] = crate::seams::BlendMode::from_node_value(Some(kind.text(entry, "mode"))).exact_split_id() as f32;
            u[P0] = n("amount") as f32;
            // A stored gradient (up to 8 stops) paints across the node's box; otherwise the colour.
            let gradient = entry.get("gradient").and_then(Value::as_object);
            let stops: Vec<(f64, [f32; 4])> = gradient
                .and_then(|g| g.get("stops"))
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|st| {
                            let offset = st.get("offset").and_then(Value::as_f64).filter(|v| v.is_finite())?;
                            let color = st.get("color").and_then(Value::as_str)?;
                            Some((offset.clamp(0.0, 1.0), filter_color(color)))
                        })
                        .take(8)
                        .collect()
                })
                .unwrap_or_default();
            let radial = gradient.and_then(|g| g.get("type")).and_then(Value::as_str) == Some("radial");
            if stops.len() >= 2 {
                u[2] = stops.len() as f32;
                u[3] = if radial { 2.0 } else { 1.0 };
                // Offsets never decrease (CSS: a stop before the previous one sits on it).
                let mut last = 0.0f64;
                for (i, (offset, color)) in stops.iter().enumerate() {
                    last = if i == 0 { *offset } else { offset.max(last) };
                    u[STOPS + i] = last as f32;
                    u[COLORS + i * 4..COLORS + i * 4 + 4].copy_from_slice(color);
                }
                // CSS linear-gradient: 0deg points up, clockwise; the gradient line
                // is |W sin θ| + |H cos θ| long, through the centre.
                let angle = gradient
                    .and_then(|g| g.get("angle"))
                    .and_then(Value::as_f64)
                    .filter(|v| v.is_finite())
                    .unwrap_or(180.0)
                    .to_radians();
                let (sin, cos) = angle.sin_cos();
                let len = (2.0 * half_px[0] * sin).abs() + (2.0 * half_px[1] * cos).abs();
                u[P1] = sin as f32;
                u[P1 + 1] = -cos as f32;
                u[P1 + 2] = if len > 1e-6 { (1.0 / len) as f32 } else { 0.0 };
            } else {
                u[2] = 1.0;
                u[COLORS..COLORS + 4].copy_from_slice(&filter_color(kind.text(entry, "color")));
            }
            2
        }
        "displace" => {
            let amount_px = n("amount") * scale;
            u[P0] = amount_px as f32;
            u[P0 + 1] = (n("scale") * scale) as f32;
            u[P0 + 2] = n("seed") as f32;
            reach_px = amount_px as f32;
            3
        }
        "grade" => {
            // White balance, exposure and lift / gain folded into one per-channel
            // slope and an offset (linear light): c' = gain · (wb · 2^ev · c + lift · (1 − wb · 2^ev · c)) + offset.
            // With the rest at their defaults, exposure / offset / gamma are Photoshop's Exposure (`paint.adjust`).
            let (t, tint) = (n("temperature"), n("tint"));
            let wb = [1.0 + 0.25 * t, 1.0 - 0.25 * tint, 1.0 - 0.25 * t];
            let (lift, gain) = (n("lift"), n("gain"));
            let k = n("exposure").exp2() * gain * (1.0 - lift);
            for (i, w) in wb.iter().enumerate() {
                u[P0 + i] = (w * k) as f32;
            }
            u[P0 + 3] = (gain * lift + n("offset")) as f32;
            u[P1] = (1.0 / n("gamma")) as f32;
            u[P1 + 1] = n("saturation") as f32;
            4
        }
        _ => return None,
    };
    u[0] = id as f32;
    Some(BackdropOp::Filter(Box::new(BackdropFilter { kind: id, uniforms: u, reach_px })))
}

/// moda-paint's parameter quantisation (1/65536; NaN → 0), so the tables built here equal `paint.adjust`'s.
fn q(v: f64) -> f64 {
    if v.is_finite() {
        (v * 65536.0).round() / 65536.0
    } else {
        0.0
    }
}

/// An adjustment kind's op (adjustment-layers.md §3): the `backdrop-adjust.wgsl` uniforms and lookup texture,
/// built as moda-paint's `Point::new` builds its kernels (`native/crates/moda-paint/src/filter/point.rs`), from the
/// same moda-color tables, so the separable kinds are byte-identical to `paint.adjust`.
fn adjust_op(kind: &moda_scene::backdrop_kinds::Kind, entry: &moda_scene::value::Map) -> Option<BackdropOp> {
    use crate::seams::{BackdropAdjust, BACKDROP_ADJUST_UNIFORM_FLOATS};
    use moda_color::tables;
    let n = |name: &str| q(kind.number(entry, name));
    let mut u = [0f32; BACKDROP_ADJUST_UNIFORM_FLOATS];
    // header 0..4, p0 4..8, b0..b2 8..20, m0..m2 20..32.
    const P0: usize = 4;
    const B0: usize = 8;
    const M0: usize = 20;
    let mut lut = vec![0u8; 256 * 2 * 4];
    let mut set_tables = |t: [tables::Table; 3]| {
        for i in 0..256 {
            lut[i * 4] = t[0][i];
            lut[i * 4 + 1] = t[1][i];
            lut[i * 4 + 2] = t[2][i];
        }
    };
    let channel = |t: tables::Table| -> [tables::Table; 3] {
        let id = tables::identity();
        match kind.param("channel").map(|_| kind.text(entry, "channel")) {
            Some("red") => [t, id, id],
            Some("green") => [id, t, id],
            Some("blue") => [id, id, t],
            _ => [t, t, t],
        }
    };
    let id: u32 = match kind.name.as_str() {
        "levels" => {
            set_tables(channel(tables::levels_table(n("inBlack"), n("inWhite"), n("gamma"), n("outBlack"), n("outWhite"))));
            0
        }
        "curves" => {
            let pts: Vec<(f64, f64)> = kind.points(entry, "points").into_iter().map(|(x, y)| (q(x), q(y))).collect();
            set_tables(channel(tables::curve_table(&pts).ok()?));
            0
        }
        "brightness-contrast" => {
            let t = tables::brightness_contrast_table(n("brightness"), n("contrast"));
            set_tables([t, t, t]);
            0
        }
        "invert" => {
            let t: tables::Table = std::array::from_fn(|i| 255 - i as u8);
            set_tables([t, t, t]);
            0
        }
        "hue-saturation" => {
            let colorize = kind.boolean(entry, "colorize");
            let hue = kind.number(entry, "hue");
            let sat = kind.number(entry, "saturation");
            u[1] = if colorize { 1.0 } else { 0.0 };
            u[P0] = (if colorize { q(hue).clamp(0.0, 360.0) } else { q(hue).clamp(-180.0, 180.0) } / 360.0) as f32;
            u[P0 + 1] = (if colorize { q(sat).clamp(0.0, 100.0) } else { q(sat).clamp(-100.0, 100.0) } / 100.0) as f32;
            u[P0 + 2] = (n("lightness") / 100.0) as f32;
            1
        }
        "color-balance" => {
            for (r, name) in ["shadows", "midtones", "highlights"].iter().enumerate() {
                for (k, v) in kind.numbers(entry, name).into_iter().enumerate().take(3) {
                    u[B0 + r * 4 + k] = (q(v) / 100.0) as f32;
                }
            }
            u[1] = if kind.boolean(entry, "preserveLuminosity") { 1.0 } else { 0.0 };
            2
        }
        "vibrance" => {
            u[P0] = (n("vibrance") / 100.0) as f32;
            u[P0 + 1] = (n("saturation") / 100.0) as f32;
            3
        }
        "desaturate" => {
            u[2] = match kind.text(entry, "method") {
                "average" => 1.0,
                "lightness" => 2.0,
                _ => 0.0,
            };
            4
        }
        "gradient-map" => {
            let stops: Vec<(f64, [u8; 3])> = kind
                .stops(entry, "stops")
                .into_iter()
                .map(|(at, css)| {
                    let b = |v: f64| (v * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8;
                    let rgb = crate::color::parse_color(Some(&css)).map_or([0, 0, 0], |c| [b(c.r), b(c.g), b(c.b)]);
                    (q(at), rgb)
                })
                .collect();
            let row = tables::gradient_row(&stops, kind.boolean(entry, "reverse")).ok()?;
            for (i, c) in row.iter().enumerate() {
                lut[(256 + i) * 4..(256 + i) * 4 + 3].copy_from_slice(c);
            }
            5
        }
        "channel-mixer" => {
            let row = |name: &str| -> [f32; 4] {
                let r = kind.numbers(entry, name);
                [
                    (q(r[0]).clamp(-2.0, 2.0) * 65536.0).round() as f32,
                    (q(r[1]).clamp(-2.0, 2.0) * 65536.0).round() as f32,
                    (q(r[2]).clamp(-2.0, 2.0) * 65536.0).round() as f32,
                    (q(r[3]).clamp(-1.0, 1.0) * 255.0 * 65536.0).round() as f32,
                ]
            };
            let red = row("red");
            let rows = if kind.boolean(entry, "monochrome") { [red; 3] } else { [red, row("green"), row("blue")] };
            for (k, r) in rows.iter().enumerate() {
                u[M0 + k * 4..M0 + k * 4 + 4].copy_from_slice(r);
            }
            6
        }
        _ => return None,
    };
    u[0] = id as f32;
    Some(BackdropOp::Adjust(Box::new(BackdropAdjust { uniforms: u, lut: std::sync::Arc::new(lut) })))
}

// ---------------------------------------------------------------------------
// The split builder
// ---------------------------------------------------------------------------

/// The node's backdrop splits, one per stacked backdrop entry that draws, in
/// list order (empty: render the node plainly). A node with one backdrop gets
/// exactly the split it always did.
pub fn compute_backdrop_splits(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    dropped: &mut Vec<String>,
) -> Vec<BackdropSplit> {
    if !is_node_blur_eligible(scene, node) {
        return Vec::new();
    }
    blur_objs(node)
        .into_iter()
        .filter_map(|blur| backdrop_split_for(scene, node, ctx, dropped, blur))
        .collect()
}

/// The node's first backdrop split (see [`compute_backdrop_splits`]).
pub fn compute_backdrop_split(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    dropped: &mut Vec<String>,
) -> Option<BackdropSplit> {
    compute_backdrop_splits(scene, node, ctx, dropped).into_iter().next()
}

/// Port of `getShapeBackdropBlurInfo` + the region math at the head of
/// `compositeShapeBackdropBlur`, for one backdrop entry `blur` of an eligible
/// node. `None` means "no effect from this entry" (every production bail
/// returns null / false).
fn backdrop_split_for(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    dropped: &mut Vec<String>,
    blur: &moda_scene::value::Map,
) -> Option<BackdropSplit> {
    let blur_type = blur.get("type").and_then(Value::as_str).unwrap_or("");

    let node_bounds = node_canvas_aabb(scene, node)?;
    let node_rect = ctx.project_rect(&node_bounds)?;
    if !node_rect.finite() || !(node_rect.w > 0.0) || !(node_rect.h > 0.0) {
        return None;
    }

    // Effective clip = page clip (ancestor mask owners are rejected by
    // eligibility, so `computeEffectivePageClip` degenerates to the page
    // rect on this path).
    let clip_rect = if ctx.page_clip == INFINITE_CLIP {
        None
    } else {
        let r = Rect {
            x: ctx.page_clip[0] as f64,
            y: ctx.page_clip[1] as f64,
            w: ctx.page_clip[2] as f64,
            h: ctx.page_clip[3] as f64,
        };
        let p = ctx.project_rect(&r)?;
        if p.finite() {
            Some(p)
        } else {
            None
        }
    };

    // Rotation + mirrored-frame sign from the canvas affine
    // (WebGPURenderOrchestrator.ts:1253-1262).
    let affine = scene.node_canvas_affine(node.id())?;
    let mut rotation_rad = 0.0f64;
    if affine.a * affine.a + affine.b * affine.b > 1e-12 {
        rotation_rad = affine.b.atan2(affine.a);
    }
    let local_frame_y_sign: f64 = if affine.a * affine.d - affine.b * affine.c < 0.0 {
        -1.0
    } else {
        1.0
    };

    let local = node_local_rect(node)?;
    let (local_w, local_h) = (local.w, local.h);

    // Pixel-space axis scales from the projected affine axes
    // (WebGPURenderOrchestrator.ts:1275-1299).
    let center_x = node_bounds.x + node_bounds.w * 0.5;
    let center_y = node_bounds.y + node_bounds.h * 0.5;
    let center_px = ctx.project_point(center_x, center_y)?;
    let axis_x_px = ctx.project_point(center_x + affine.a, center_y + affine.b)?;
    let axis_y_px = ctx.project_point(center_x + affine.c, center_y + affine.d)?;
    let scale_x = (axis_x_px.0 - center_px.0).hypot(axis_x_px.1 - center_px.1);
    let scale_y = (axis_y_px.0 - center_px.0).hypot(axis_y_px.1 - center_px.1);

    let effective_scale = scale_x.abs().min(scale_y.abs()).max(0.0001);
    let canvas_target_scale = ctx.canvas_target_scale().max(0.0001);
    let local_half_w = local_w * 0.5 * scale_x.abs();
    let local_half_h = local_h * 0.5 * scale_y.abs();

    // ---- effect mapping with production clamps -----------------------------
    let mut op: BackdropOp = match blur_type {
        "background-blur" => {
            let radius = obj_f64(blur, "radius").unwrap_or(f64::NAN);
            let blur_radius_px = js_min(radius * REACH * effective_scale, MAX_BLUR_RADIUS_PX);
            note_blur_radius_clamped(node, "background blur radius", radius, effective_scale);
            if let Some(p) = progressive(blur) {
                note_blur_radius_clamped(node, "background blur start radius", p.start_radius, effective_scale);
            }
            // Progressive: the ramp's points are fractions of the node's local
            // bounds, through its canvas affine, projected to target px.
            let ramp = progressive(blur).and_then(|p| {
                let at = |f: moda_scene::effect::EffectOffset| {
                    let (lx, ly) = (local.x + f.x * local.w, local.y + f.y * local.h);
                    ctx.project_point(
                        affine.a * lx + affine.c * ly + affine.e,
                        affine.b * lx + affine.d * ly + affine.f,
                    )
                    .map(|(x, y)| [x as f32, y as f32])
                };
                Some(BlurRamp {
                    start_radius_px: js_min(
                        p.start_radius.max(0.0) * REACH * effective_scale,
                        MAX_BLUR_RADIUS_PX,
                    ) as f32,
                    start_px: at(p.start)?,
                    end_px: at(p.end)?,
                })
            });
            let widest = ramp.map_or(blur_radius_px, |r| {
                js_max(blur_radius_px.max(0.0), r.start_radius_px as f64)
            });
            if !(widest > 0.5) {
                return None;
            }
            BackdropOp::Blur {
                radius_px: blur_radius_px.max(0.0) as f32,
                frost: 0.0,
                ramp,
            }
        }
        "pixelate" => {
            let raw = obj_f64(blur, "pixelSize").unwrap_or(1.0) * canvas_target_scale;
            if !raw.is_finite() {
                return None;
            }
            let normalized = normalize_pixelate(raw)?;
            BackdropOp::Pixelate {
                cell_px: js_min(MAX_PIXELATE_SIZE_PX, normalized) as f32,
            }
        }
        "dither" => {
            let raw = obj_f64(blur, "pixelSize").unwrap_or(1.0) * canvas_target_scale;
            let normalized = normalize_dither(raw)?;
            let brightness = obj_f64(blur, "brightness")
                .filter(|v| v.is_finite())
                .unwrap_or(0.0)
                .clamp(-100.0, 100.0);
            let contrast = obj_f64(blur, "contrast")
                .filter(|v| v.is_finite())
                .unwrap_or(100.0)
                .clamp(0.0, 200.0);
            BackdropOp::Dither {
                cell_px: js_min(MAX_DITHER_PIXEL_SIZE_PX, normalized) as f32,
                pattern: dither_pattern_mode(blur.get("pattern").and_then(Value::as_str)),
                brightness: brightness as f32,
                contrast: contrast as f32,
            }
        }
        kind if moda_scene::backdrop_kinds::kind(kind).is_some() => filter_op(
            kind,
            blur,
            canvas_target_scale,
            [local_half_w, local_half_h],
        )?,
        // glass (default arm, WebGPURenderOrchestrator.ts:1366-1400)
        _ => {
            let half_min_dimension = local_half_w.min(local_half_h);
            let frost = obj_f64(blur, "frost").unwrap_or(f64::NAN);
            note_blur_radius_clamped(node, "glass frost", frost, effective_scale);
            let blur_radius_px = js_min(js_max(0.0, frost * REACH * effective_scale), MAX_BLUR_RADIUS_PX);
            let depth = js_max(0.0, obj_f64(blur, "depth").unwrap_or(f64::NAN));
            let depth_px = (depth / 100.0) * js_max(1.0, half_min_dimension);
            let refraction = js_max(0.0, obj_f64(blur, "refraction").unwrap_or(f64::NAN));
            let refraction_px_raw = (refraction / 100.0) * js_max(2.0, half_min_dimension * 0.35);
            let refraction_px = if depth_px > 1e-4 {
                refraction_px_raw
            } else {
                0.0
            };
            let light_angle_rad =
                obj_f64(blur, "lightAngle").unwrap_or(0.0) * std::f64::consts::PI / 180.0;
            let pct = |v: Option<f64>| (v.unwrap_or(0.0) / 100.0).clamp(0.0, 1.0);
            let light_intensity = pct(obj_f64(blur, "lightIntensity"));
            let dispersion_px = if depth_px > 1e-4 {
                pct(obj_f64(blur, "dispersion")) * js_min(16.0, js_max(2.0, depth_px * 0.25))
            } else {
                0.0
            };
            let splay = pct(obj_f64(blur, "splay"));
            // Figma's specular rim is a crisp line about 1.5 px wide at the edge.
            let rim_width_px = 1.5 * effective_scale;
            if !(blur_radius_px > 0.5
                || refraction_px > 0.25
                || light_intensity > 0.001
                || dispersion_px > 0.001)
            {
                return None;
            }
            let style = blur.get("refractionStyle").and_then(Value::as_str);
            let refraction_style = match style {
                Some("lens") => 1,
                Some("swirl") => 2,
                _ => 0,
            };
            BackdropOp::Glass {
                blur_px: blur_radius_px as f32,
                depth_px: depth_px as f32,
                refraction_px: refraction_px as f32,
                dispersion_px: dispersion_px as f32,
                rim_width_px: rim_width_px as f32,
                light_angle_rad: light_angle_rad as f32,
                light_splay: splay as f32,
                light_intensity: light_intensity as f32,
                refraction_style,
            }
        }
    };

    // ---- sampling region (compositeShapeBackdropBlur:525-604) ---------------
    let target = Rect {
        x: 0.0,
        y: 0.0,
        w: ctx.width_px as f64,
        h: ctx.height_px as f64,
    };
    let node_rect_clipped = intersect(&node_rect, &target)?;
    let mut effective_clip = match &clip_rect {
        Some(c) => intersect(c, &target)?,
        None => target,
    };
    // Clipping containers (Figma: an effect inside a frame samples, and
    // draws, only within the frame): each box narrows the sampled region and
    // the composite scissor; a rounded corner the visible part reaches is
    // applied by the composite's clip SDF (one per split — a node that reaches
    // the rounded corners of two nested frames renders plainly).
    let container_clips: Vec<(Rect, [f64; 4])> = clipping_containers_clip(scene, node.id())?
        .iter()
        .map(|c| {
            let s = ctx.canvas_target_scale();
            Some((ctx.project_rect(&c.rect)?, c.radii.map(|r| r * s)))
        })
        .collect::<Option<_>>()?;
    for (r, _) in &container_clips {
        effective_clip = intersect(&effective_clip, r)?;
    }
    let visible = intersect(&node_rect_clipped, &effective_clip)?;
    // A filter kind that samples away from the pixel it writes (displace) stays inside this box, so it never
    // reads what lies outside a clipping frame (`quantize_sample` keeps the region inside it too).
    if let BackdropOp::Filter(f) = &mut op {
        f.uniforms[12..16].copy_from_slice(&[
            effective_clip.x as f32,
            effective_clip.y as f32,
            effective_clip.w as f32,
            effective_clip.h as f32,
        ]);
    }
    let mut rounded = container_clips.iter().filter(|(rect, radii)| {
        !region_clear_of_corners(
            &visible,
            &ContainerClip {
                rect: *rect,
                radii: *radii,
            },
        )
    });
    let clip_rrect_px = rounded.next().map(|(r, radii)| {
        (
            [r.x as f32, r.y as f32, r.w as f32, r.h as f32],
            radii.map(|v| v as f32),
        )
    });
    if rounded.next().is_some() {
        return None;
    }
    let expansion = sample_expansion(&op);
    let sample = intersect(
        &Rect {
            x: visible.x - expansion,
            y: visible.y - expansion,
            w: visible.w + expansion * 2.0,
            h: visible.h + expansion * 2.0,
        },
        &effective_clip,
    )?;
    if !(sample.w > 0.0 && sample.h > 0.0) {
        return None;
    }
    let q = quantize_sample(&sample, &effective_clip);
    if !q.finite() || q.w < 1.0 || q.h < 1.0 || q.w > MAX_TEXTURE_DIM || q.h > MAX_TEXTURE_DIM {
        return None;
    }

    // ---- coverage source ----------------------------------------------------
    let node_type = node.node_type();
    let smoothed_rect = (node_type == "rectangle" || node_type == "container")
        && crate::shape_to_path::rect_has_path_corners(node.json_field("cornerRadius"), node.json_field("cornerSmoothing"), node.json_field("cornerStyle"));
    let coverage = if (node_type == "rectangle" || node_type == "container") && !smoothed_rect {
        // Corner radii × effectiveScale, clamped to the local half-extent
        // (WebGPURenderOrchestrator.ts:1410-1419).
        let limit = local_half_w.min(local_half_h);
        let radii = corner_radii(node);
        let clamp_radius = |r: f64| (r * effective_scale).max(0.0).min(limit);
        CoverageSource::Rect {
            size: [(local_half_w * 2.0) as f32, (local_half_h * 2.0) as f32],
            radii: [
                clamp_radius(radii[0]) as f32,
                clamp_radius(radii[1]) as f32,
                clamp_radius(radii[2]) as f32,
                clamp_radius(radii[3]) as f32,
            ],
        }
    } else if node_type == "ellipse" {
        CoverageSource::Ellipse {
            size: [(local_half_w * 2.0) as f32, (local_half_h * 2.0) as f32],
        }
    } else {
        // star / polygon / path / smoothed-corner rect → tessellated fill
        // silhouette in target px (`buildCoverageVerticesTargetPx`); the
        // shaders' rounded-box SDF has no squircle.
        let verts = build_coverage_vertices_target_px(scene, node, ctx, dropped)?;
        if verts.len() < 6 {
            return None;
        }
        CoverageSource::Geometry(crate::seams::TessellatedGeometry {
            vertex_count: (verts.len() / 2) as u32,
            vertex_types: vec![0.0; verts.len() / 2],
            bounds: [0.0; 4],
            positions: verts,
            fill_dropped: None,
            stroke_dropped: None,
            stroke_uv: Vec::new(),
        })
    };

    Some(BackdropSplit {
        node_id: node.id().to_string(),
        op,
        region_px: [q.x as u32, q.y as u32, q.w as u32, q.h as u32],
        coverage,
        node_rect_px: [
            node_rect.x as f32,
            node_rect.y as f32,
            node_rect.w as f32,
            node_rect.h as f32,
        ],
        visible_rect_px: [
            visible.x as f32,
            visible.y as f32,
            visible.w as f32,
            visible.h as f32,
        ],
        rotation_rad: rotation_rad as f32,
        local_frame_y_sign: local_frame_y_sign as f32,
        local_half_size_px: [local_half_w as f32, local_half_h as f32],
        clip_rrect_px,
    })
}

/// `buildCoverageVerticesTargetPx` (WebGPURenderOrchestrator.ts:1476-1551):
/// tessellate the fill (no stroke, shared caps), then map local → canvas →
/// target px as one affine recovered from three reference points.
fn build_coverage_vertices_target_px(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    dropped: &mut Vec<String>,
) -> Option<Vec<f32>> {
    let (cmds, closed, fill_rule) = match node.node_type() {
        "rectangle" | "container" => (
            crate::shape_to_path::rect_outline_cmds(
                node.f64("width").unwrap_or(0.0),
                node.f64("height").unwrap_or(0.0),
                node.json_field("cornerRadius"),
                node.json_field("cornerSmoothing"),
                node.json_field("cornerStyle"),
            ),
            true,
            pt::FillRule::NonZero,
        ),
        "star" => (
            tess::shape_paths::star_to_path_cmds(
                node.f64("width").unwrap_or(0.0),
                node.f64("height").unwrap_or(0.0),
                node.f64("numPoints").unwrap_or(5.0),
                Some(node.f64("innerRadiusRatio").unwrap_or(0.5)),
                node.f64("cornerRadius").unwrap_or(0.0),
            ),
            true,
            pt::FillRule::NonZero,
        ),
        "polygon" => (
            tess::shape_paths::polygon_to_path_cmds(
                node.f64("width").unwrap_or(0.0),
                node.f64("height").unwrap_or(0.0),
                node.f64("sides").unwrap_or(3.0),
                node.f64("cornerRadius").unwrap_or(0.0),
            ),
            true,
            pt::FillRule::NonZero,
        ),
        "path" => (
            tess::path_cmds_for_node(node),
            node.bool("closed").unwrap_or(false),
            pt::fill_rule_from_str(node.str("fillRule")),
        ),
        _ => return None,
    };
    if cmds.is_empty() {
        return None;
    }

    // Fill-only tessellation (production passes strokeWidth undefined).
    let geo = tess::tessellate_combined(
        &cmds,
        0.0,
        closed,
        pt::LineCap::Butt,
        pt::LineJoin::Miter,
        fill_rule,
        pt::StrokeAlign::Center,
    );
    // Backdrop coverage paints independently of the body's fill alpha:
    // transparent fills intentionally expose only the backdrop effect.
    if let Some(why) = &geo.fill_dropped {
        dropped.push(format!(
            "{} ({}): backdrop coverage fill dropped: {why}",
            node.id(),
            node.node_type()
        ));
        return None;
    }
    if geo.positions.len() < 6 {
        return None;
    }

    let affine = scene.node_canvas_affine(node.id())?;
    let project_local = |lx: f64, ly: f64| -> Option<(f64, f64)> {
        let cx = affine.a * lx + affine.c * ly + affine.e;
        let cy = affine.b * lx + affine.d * ly + affine.f;
        ctx.project_point(cx, cy)
    };
    let p0 = project_local(0.0, 0.0)?;
    let px = project_local(1.0, 0.0)?;
    let py = project_local(0.0, 1.0)?;
    let (ax, bx) = (px.0 - p0.0, px.1 - p0.1);
    let (ay, by) = (py.0 - p0.0, py.1 - p0.1);

    let mut out = Vec::with_capacity(geo.positions.len());
    let mut i = 0;
    while i + 1 < geo.positions.len() {
        let lx = geo.positions[i] as f64;
        let ly = geo.positions[i + 1] as f64;
        out.push((p0.0 + ax * lx + ay * ly) as f32);
        out.push((p0.1 + bx * lx + by * ly) as f32);
        i += 2;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene_with(nodes: moda_scene::value::Value) -> Scene {
        let ids: Vec<moda_scene::value::Value> = nodes
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["id"].clone())
            .collect();
        let doc = moda_scene::value::json!({
            "nodes": nodes,
            "pages": [{ "id": "p1", "width": 1280, "height": 800, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                        "nodeIds": ids }],
            "config": {},
        });
        Scene::load(&doc).unwrap()
    }

    /// Region assertions tolerate ±1 px: floor/ceil quantization sits on top
    /// of the f32 VP projection (1/640 is not exactly representable), so a
    /// coordinate can land at 99.99999 — the production renderer has the
    /// identical behavior, but hand-predicting which side it lands on is
    /// not worth it for unit tests (the SSIM gate covers exactness).
    fn assert_region_close(got: [u32; 4], want: [u32; 4]) {
        for i in 0..4 {
            let d = (got[i] as i64 - want[i] as i64).abs();
            assert!(d <= 1, "region {got:?} vs {want:?} (component {i})");
        }
    }

    fn ctx<'a>(scene: &'a Scene) -> PlanCtx<'a> {
        let (px, py) = scene.page_position(0);
        PlanCtx::new(
            scene,
            1280,
            800,
            px,
            py,
            1280.0,
            800.0,
            [px as f32, py as f32, 1280.0, 800.0],
        )
    }

    #[test]
    fn hidden_backdrop_is_inactive() {
        // design-editor E1.9: `visible: false` keeps the config but renders nothing.
        let scene = scene_with(moda_scene::value::json!([
            { "id": "n1", "type": "rectangle", "x": 10, "y": 10, "width": 100, "height": 100,
              "effects": [{ "id": "backdrop-0", "type": "background-blur", "radius": 24, "visible": false }] },
        ]));
        let n = scene.effective_node("n1").unwrap();
        assert!(!has_active_background_blur(n));
        assert!(compute_backdrop_split(&scene, n, &ctx(&scene), &mut Vec::new()).is_none());
        assert!(!page_has_active_backdrop(&scene, "p1"));

        let shown = scene_with(moda_scene::value::json!([
            { "id": "n1", "type": "rectangle", "x": 10, "y": 10, "width": 100, "height": 100,
              "effects": [{ "id": "backdrop-0", "type": "background-blur", "radius": 24, "visible": true }] },
        ]));
        assert!(has_active_background_blur(
            shown.effective_node("n1").unwrap()
        ));
    }

    #[test]
    fn uniform_blur_params_and_region() {
        let scene = scene_with(moda_scene::value::json!([
            { "id": "r", "type": "rectangle", "x": 100, "y": 100, "width": 200, "height": 150,
              "cornerRadius": 24,
              "effects": [{ "id": "backdrop-0", "type": "background-blur", "radius": 24 }] },
        ]));
        assert!(page_has_active_backdrop(&scene, "p1"));
        let n = scene.effective_node("r").unwrap();
        let s = compute_backdrop_split(&scene, n, &ctx(&scene), &mut Vec::new()).expect("split");
        match s.op {
            // Radius 24 (σ = 12) reaches 3σ = 36.
            BackdropOp::Blur { radius_px, .. } => {
                assert!((radius_px - 36.0).abs() < 1e-4)
            }
            ref o => panic!("wrong op {o:?}"),
        }
        // expansion = ceil(36) = 36 → region = node rect expanded 36, clamped.
        assert_region_close(s.region_px, [64, 64, 272, 222]);
        for (got, want) in s.node_rect_px.iter().zip([100.0, 100.0, 200.0, 150.0]) {
            assert!((got - want).abs() < 1e-3, "node_rect {:?}", s.node_rect_px);
        }
        for (got, want) in s.local_half_size_px.iter().zip([100.0, 75.0]) {
            assert!((got - want).abs() < 1e-3, "half {:?}", s.local_half_size_px);
        }
        match s.coverage {
            CoverageSource::Rect { radii, .. } => {
                for r in radii {
                    assert!((r - 24.0).abs() < 1e-3, "radii {radii:?}");
                }
            }
            ref c => panic!("wrong coverage {c:?}"),
        }
    }

    #[test]
    fn uniform_reach_clamps_at_300() {
        let scene = scene_with(moda_scene::value::json!([
            { "id": "r", "type": "rectangle", "x": 0, "y": 0, "width": 400, "height": 400,
              "effects": [{ "id": "backdrop-0", "type": "background-blur", "radius": 1000 }] },
        ]));
        let n = scene.effective_node("r").unwrap();
        let s = compute_backdrop_split(&scene, n, &ctx(&scene), &mut Vec::new()).unwrap();
        match s.op {
            BackdropOp::Blur { radius_px, .. } => assert_eq!(radius_px, 300.0),
            ref o => panic!("{o:?}"),
        }
    }

    #[test]
    fn pixelate_and_dither_clamps() {
        let scene = scene_with(moda_scene::value::json!([
            { "id": "p", "type": "rectangle", "x": 0, "y": 0, "width": 100, "height": 100,
              "effects": [{ "id": "backdrop-0", "type": "pixelate", "pixelSize": 900 }] },
            { "id": "d", "type": "rectangle", "x": 200, "y": 0, "width": 100, "height": 100,
              "effects": [{ "id": "backdrop-0", "type": "dither", "pixelSize": 900, "pattern": "atkinson",
                                   "brightness": 250, "contrast": 500 }] },
        ]));
        let p = compute_backdrop_split(
            &scene,
            scene.effective_node("p").unwrap(),
            &ctx(&scene),
            &mut Vec::new(),
        )
        .unwrap();
        match p.op {
            BackdropOp::Pixelate { cell_px } => assert_eq!(cell_px, 400.0),
            ref o => panic!("{o:?}"),
        }
        let d = compute_backdrop_split(
            &scene,
            scene.effective_node("d").unwrap(),
            &ctx(&scene),
            &mut Vec::new(),
        )
        .unwrap();
        match d.op {
            BackdropOp::Dither {
                cell_px,
                pattern,
                brightness,
                contrast,
            } => {
                assert_eq!(cell_px, 256.0);
                assert_eq!(pattern, 3);
                assert_eq!(brightness, 100.0);
                assert_eq!(contrast, 200.0);
            }
            ref o => panic!("{o:?}"),
        }
        // Pixelate / dither sample with no outward expansion.
        assert_region_close(p.region_px, [0, 0, 100, 100]);
    }

    #[test]
    fn glass_param_mapping() {
        // bg-glass-default: 400x300 at (440,250), frost 30, refraction 60,
        // depth 40, lightIntensity 40.
        let scene = scene_with(moda_scene::value::json!([
            { "id": "g", "type": "rectangle", "x": 440, "y": 250, "width": 400, "height": 300,
              "cornerRadius": 32,
              "effects": [{ "id": "backdrop-0", "type": "glass", "frost": 30, "refraction": 60,
                "refractionStyle": "barrel", "depth": 40, "lightAngle": 0,
                "lightIntensity": 40, "dispersion": 0, "splay": 0 }] },
        ]));
        let s = compute_backdrop_split(
            &scene,
            scene.effective_node("g").unwrap(),
            &ctx(&scene),
            &mut Vec::new(),
        )
        .unwrap();
        match s.op {
            BackdropOp::Glass {
                blur_px,
                depth_px,
                refraction_px,
                dispersion_px,
                rim_width_px,
                light_intensity,
                refraction_style,
                ..
            } => {
                // Frost 30 is a blur radius: σ = 15, reach 45.
                assert!((blur_px - 45.0).abs() < 1e-4);
                // halfMin = 150 → depth = 0.4 × 150 = 60
                assert!((depth_px - 60.0).abs() < 1e-4);
                // refraction = 0.6 × max(2, 150·0.35=52.5) = 31.5
                assert!((refraction_px - 31.5).abs() < 1e-4);
                assert_eq!(dispersion_px, 0.0);
                assert!((rim_width_px - 1.5).abs() < 1e-4);
                assert!((light_intensity - 0.4).abs() < 1e-6);
                assert_eq!(refraction_style, 0);
            }
            ref o => panic!("{o:?}"),
        }
        // expansion = ceil(45 + 31.5 + 0) + 1 = 78
        assert_region_close(s.region_px, [362, 172, 556, 456]);
    }

    #[test]
    fn eligibility_rejections() {
        let scene = scene_with(moda_scene::value::json!([
            { "id": "op", "type": "rectangle", "x": 0, "y": 0, "width": 100, "height": 100,
              "opacity": 0.5, "effects": [{ "id": "backdrop-0", "type": "background-blur", "radius": 24 }] },
            { "id": "bm", "type": "rectangle", "x": 0, "y": 0, "width": 100, "height": 100,
              "blendMode": "multiply", "effects": [{ "id": "backdrop-0", "type": "background-blur", "radius": 24 }] },
            { "id": "we", "type": "ellipse", "x": 0, "y": 0, "width": 100, "height": 100,
              "sweepAngle": 180, "effects": [{ "id": "backdrop-0", "type": "background-blur", "radius": 24 }] },
        ]));
        for id in ["op", "bm", "we"] {
            let n = scene.effective_node(id).unwrap();
            assert!(
                compute_backdrop_split(&scene, n, &ctx(&scene), &mut Vec::new()).is_none(),
                "{id} must be rejected"
            );
        }
    }

    #[test]
    fn star_coverage_vertices_in_target_px() {
        let scene = scene_with(moda_scene::value::json!([
            { "id": "s", "type": "star", "x": 100, "y": 100, "width": 200, "height": 200,
              "numPoints": 5, "innerRadiusRatio": 0.5,
              "effects": [{ "id": "backdrop-0", "type": "background-blur", "radius": 16 }] },
        ]));
        let s = compute_backdrop_split(
            &scene,
            scene.effective_node("s").unwrap(),
            &ctx(&scene),
            &mut Vec::new(),
        )
        .unwrap();
        match &s.coverage {
            CoverageSource::Geometry(g) => {
                assert!(g.positions.len() >= 6);
                // All coverage verts inside the projected node rect (top
                // vertex at x=200, y=100 in target px).
                for pair in g.positions.chunks(2) {
                    assert!(pair[0] >= 99.0 && pair[0] <= 301.0, "x {}", pair[0]);
                    assert!(pair[1] >= 99.0 && pair[1] <= 301.0, "y {}", pair[1]);
                }
            }
            c => panic!("wrong coverage {c:?}"),
        }
    }
}
