//! Variable-width / brushed strokes drawn as filled outlines (brush plan P8,
//! docs/roadmap/brushes.md §5).
//!
//! A path whose stroke has a width profile, taper, wave, dynamic wobble or a
//! calligraphic brush ([`needs_outline`], the twin of TS `needsStrokeOutline`)
//! is not stroked by the uniform stroker: moda-scene's `stroke_outline`
//! returns the stroke's outline, this module flattens it and lyon fills it
//! (nonzero), and the triangles ride the combined stream as STROKE vertices —
//! so stroke paint (solid / gradient), opacity and blend apply unchanged. Trim
//! and dash become arc-length windows on the outline ([`arc_windows`]), so
//! widths stay continuous across dashes. One implementation for the WASM
//! canvas and native moda-render: canvas = export.
//!
//! Resolution: the outline's fit and flattening tolerance follow the render
//! scale in power-of-two bands ([`band_for_scale`]); the per-node tess cache
//! keys on the band ([`outline_cache_salt`]).
//!
//! Limits: the outline fill takes the stroke vertex budget (100k expanded
//! vertices) and lyon's sweep-cost guard (2M). An outline the guard refuses
//! as one fill is cut into arc-length chunks and filled in batches, then
//! coarsened; only if all of that fails is it drawn as a uniform stroke with
//! a `skipped` warning — never a page refusal ([`tessellate_outlined`]).

use std::cell::Cell;

use moda_scene::stroke_outline::{
    stroke_outline, stroke_outline_windows, subpath_spans, ArcWindow, Outline, OutlineOptions,
    PathPoint,
};
use moda_scene::types::RawNode;
use moda_scene::value::{json, Value};

use super::dash::{
    aligned_dash_intervals, normalize_dash_pattern, scale_dash_pattern, seed_dash_phase,
    BASE_DASH_STROKE_WIDTH,
};
use super::moda_engine_path_types::{FillRule, LineCap, LineJoin, PathCmd, StrokeAlign};
use super::tessellator::{body_fill, combine_fill_and_outline, fill_outline_banded};
use super::{is_trim_identity, tessellate_combined_with_dash_trim, TrimParams};
use crate::seams::TessellatedGeometry;

/// Outline fit + flattening tolerance at band 0 (1 px per canvas unit), in
/// canvas units — i.e. a fifth of a device pixel at every band.
pub const OUTLINE_TOLERANCE_PX: f64 = 0.2;
/// Zoom bands: `round(log2(px per unit))`, clamped. Below 1/16 the outline
/// stops getting coarser (tolerance ≤ 3.2 units); above 512 it stops getting
/// finer (≤ 0.0004 units). 512 device px per unit is the editor's top zoom,
/// 25600%, on a 2× display, so curves stay within a fifth of a device pixel
/// across the whole zoom range.
pub const MIN_BAND: i32 = -4;
pub const MAX_BAND: i32 = 9;
/// Most arc-length windows (dashes) one stroke may produce; the rest of a
/// denser dash pattern is dropped (a 20k-dash stroke is sub-pixel noise).
pub const MAX_WINDOWS: usize = 20_000;

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|x| x != 0.0 && !x.is_nan()),
        Some(Value::String(s)) => !s.is_empty(),
        Some(_) => true,
    }
}

/// `needsStrokeOutline` (core/geometry/strokeOutline.ts): a path whose stroke
/// has a profile / taper / wave, a dynamic wobble with wiggle > 0, or names a
/// calligraphic brush in the document's brush table; a line with a profile /
/// taper / wave (brushes.md §4.9: lines carry the width model only).
pub fn needs_outline<'a>(node: &RawNode, brush: impl Fn(&str) -> Option<&'a Value>) -> bool {
    let style = moda_scene::paint::model_stroke_style(node);
    let width_model =
        truthy(style.profile.as_ref()) || truthy(style.taper.as_ref()) || truthy(style.wave.as_ref());
    match node.node_type() {
        "line" => return width_model,
        "path" => {}
        _ => return false,
    }
    if width_model {
        return true;
    }
    if style
        .dynamic
        .as_ref()
        .and_then(|d| d.get("wiggle"))
        .and_then(Value::as_f64)
        .is_some_and(|w| w > 0.0)
    {
        return true;
    }
    brush_id(node)
        .as_deref()
        .and_then(brush)
        .is_some_and(|b| b.get("type").and_then(Value::as_str) == Some("calligraphic"))
}

/// The model stroke's `brush.brushId`, when set.
pub fn brush_id(node: &RawNode) -> Option<String> {
    moda_scene::paint::with_model_stroke(node, |s| {
        s?.brush
            .as_ref()
            .and_then(|b| b.get("brushId"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    })
}

/// [`needs_outline`] against a scene's brush table.
pub fn scene_needs_outline(scene: &moda_scene::Scene, node: &RawNode) -> bool {
    needs_outline(node, |id| scene.doc().brush(id))
}

/// True when any node in the scene draws an outlined stroke (gates the
/// zoom-band term of the camera plan keys, so scenes without variable strokes
/// keep a zoom-invariant camera plan).
pub fn scene_has_outlines(scene: &moda_scene::Scene) -> bool {
    // Resolved / transient nodes too (variables, expansions): what the plan draws.
    scene
        .node_order()
        .iter()
        .filter_map(|id| scene.effective_node(id))
        .chain(
            scene
                .transient_by_id()
                .keys()
                .filter_map(|id| scene.effective_node(id)),
        )
        .any(|n| node_has_outline(scene, n))
}

/// The per-node term of [`scene_has_outlines`].
pub fn node_has_outline(scene: &moda_scene::Scene, n: &RawNode) -> bool {
    // A group's appearance (E8.2) strokes its merged outline like a path's.
    if let Some(record) = scene.group_appearance(n.id()) {
        if moda_scene::paint::node_strokes(record)
            .iter()
            .any(|s| s.brush.is_some() || s.profile.is_some() || s.taper.is_some() || s.wave.is_some() || s.dynamic.is_some())
        {
            return true;
        }
    }
    scene_needs_outline(scene, n)
        || super::placed::placed_brush(scene, n)
            .is_some_and(|(k, _)| super::placed::band_dependent(k))
}

/// True when any node strokes or fills curves (ellipses, curved or rounded
/// paths, rounded shapes, connectors, curved or rounded line routes). Curve
/// flattening follows the zoom band
/// (`tessellator::flatness_tolerance`), so such scenes key camera plans on the
/// band too (quantised: one plan per band, `MIN_BAND..=MAX_BAND`);
/// straight-only scenes stay zoom-invariant.
pub fn scene_has_curves(scene: &moda_scene::Scene) -> bool {
    scene
        .node_order()
        .iter()
        .filter_map(|id| scene.effective_node(id))
        .chain(
            scene
                .transient_by_id()
                .keys()
                .filter_map(|id| scene.effective_node(id)),
        )
        .any(node_has_curves)
}

/// Whether `n`'s stroke draws round ends: round caps or joins, or a circle arrowhead. Their arcs flatten
/// per zoom band (`tessellator::round_arc_segments`), so a straight line or polyline with them is curved.
pub fn node_has_round_ends(n: &RawNode) -> bool {
    let style = moda_scene::paint::node_stroke_style(n);
    let stroked = n.node_type() == "line" || style.width.is_some_and(|w| w.is_finite() && w > 0.0);
    stroked
        && (style.cap.as_deref() == Some("round")
            || style.join.as_deref() == Some("round")
            || style.start_arrow.as_deref() == Some("circle")
            || style.end_arrow.as_deref() == Some("circle"))
}

/// The per-node term of [`scene_has_curves`].
pub fn node_has_curves(n: &RawNode) -> bool {
    // Illustrator's Appearance (E8.2): a layer's own geometry effects, and a group's merged outline, may curve
    // whatever the node's own outline is; counted as curved (the plan keys on the band).
    if crate::paint::layers::has_entry_effects(n) || moda_scene::appearance::group_has_appearance(n) {
        return true;
    }
    // A vector pattern paint (E8.5): its tile art's curves flatten per band like the node's own (counted whatever the
    // art is: reading the pattern's shapes here would cost a lookup per node per build).
    if crate::pattern_paint::enabled() && !moda_scene::pattern::node_pattern_ids(n).is_empty() {
        return true;
    }
    let curved = |n: &RawNode| -> bool {
        if node_has_round_ends(n) {
            return true;
        }
        // A line's body is curved on a curved route or a rounded elbow
        // (`line_to_path`); its stroke defaults on, so no stroke check.
        if n.node_type() == "line" {
            let routed = n.array("waypoints").is_some_and(|w| !w.is_empty());
            return routed
                && match n.str("routing") {
                    Some("curved") => true,
                    Some("elbowed") => n
                        .f64("cornerRadius")
                        .is_some_and(|r| r.is_finite() && r > 0.0),
                    _ => false,
                };
        }
        // Stroked or filled: both flatten per band.
        let stroked = moda_scene::paint::node_stroke_style(n)
            .width
            .is_some_and(|w| w.is_finite() && w > 0.0);
        if !stroked && !crate::paint::fill_paints(n) {
            return false;
        }
        let rounded = n
            .f64("cornerRadius")
            .is_some_and(|r| r.is_finite() && r > 0.0);
        match n.node_type() {
            "ellipse" | "connector" => true,
            "path" => {
                rounded
                    || n.points().is_some_and(|pts| {
                        pts.iter()
                            .any(|p| p.kind().is_some_and(|t| t != "move" && t != "line"))
                    })
            }
            "rectangle" | "container" | "star" | "polygon" => rounded,
            _ => false,
        }
    };
    curved(n)
}

/// The camera scale a plan build enters: the real one for scenes with
/// outlines or curves (both refit per band, above and below 1×), none
/// otherwise (straight-only scenes keep one zoom-invariant plan).
pub fn camera_scale_for(has_outlines: bool, has_curves: bool, px_per_unit: f64) -> Option<f64> {
    (has_outlines || has_curves).then_some(px_per_unit)
}

/// Zoom band for `px_per_unit` device pixels per canvas unit.
pub fn band_for_scale(px_per_unit: f64) -> i32 {
    if !(px_per_unit.is_finite() && px_per_unit > 0.0) {
        return 0;
    }
    (px_per_unit.log2().round() as i32).clamp(MIN_BAND, MAX_BAND)
}

/// Fit / flattening tolerance (canvas units) at `band`.
pub fn tolerance_for_band(band: i32) -> f64 {
    OUTLINE_TOLERANCE_PX / 2f64.powi(band.clamp(MIN_BAND, MAX_BAND))
}

thread_local! {
    /// Camera frames: device px per canvas unit (zoom × dpr), set for the
    /// duration of a camera plan build. Export / page plans leave it unset.
    static CAMERA_SCALE: Cell<Option<f64>> = const { Cell::new(None) };
    /// Page plans: `RenderOptions::scale`, set by the coverage plan builder
    /// when no camera scale is active.
    static PLAN_SCALE: Cell<Option<f64>> = const { Cell::new(None) };
    /// Camera frames: the visible canvas rect `[x0, y0, x1, y1]`, for
    /// viewport-first outline refinement.
    static CAMERA_VIEW: Cell<Option<[f64; 4]>> = const { Cell::new(None) };
}

/// Restores the previous camera view on drop.
pub(crate) struct ViewGuard(Option<[f64; 4]>);

impl Drop for ViewGuard {
    fn drop(&mut self) {
        CAMERA_VIEW.with(|c| c.set(self.0));
    }
}

/// Enter a camera build whose viewport shows `view` (canvas units).
pub(crate) fn enter_view(view: Option<[f64; 4]>) -> ViewGuard {
    ViewGuard(CAMERA_VIEW.with(|c| c.replace(view)))
}

/// The current camera view, if any.
pub fn camera_view() -> Option<[f64; 4]> {
    CAMERA_VIEW.with(Cell::get)
}

/// How much of `bbox` (canvas) the viewport `view` shows, in canvas units² —
/// 0 when off-screen.
pub fn visible_area(bbox: [f64; 4], view: [f64; 4]) -> f64 {
    let w = bbox[2].min(view[2]) - bbox[0].max(view[0]);
    let h = bbox[3].min(view[3]) - bbox[1].max(view[1]);
    if w > 0.0 && h > 0.0 {
        w * h
    } else {
        0.0
    }
}

/// `view` grown by half its size on every side: strokes that far off-screen
/// are refined ahead of a pan reaching them.
pub fn near_view(view: [f64; 4]) -> [f64; 4] {
    let (dx, dy) = (0.5 * (view[2] - view[0]), 0.5 * (view[3] - view[1]));
    [view[0] - dx, view[1] - dy, view[2] + dx, view[3] + dy]
}

/// Canvas AABB of node-local `bounds` ([x, y, w, h]) under `affine`, grown
/// by `pad` (canvas units).
pub fn canvas_bbox(bounds: [f32; 4], affine: &moda_scene::types::Affine, pad: f64) -> [f64; 4] {
    let (x0, y0) = (bounds[0] as f64, bounds[1] as f64);
    let (x1, y1) = (x0 + bounds[2] as f64, y0 + bounds[3] as f64);
    let mut r = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
        let cx = affine.a * x + affine.c * y + affine.e;
        let cy = affine.b * x + affine.d * y + affine.f;
        r = [r[0].min(cx), r[1].min(cy), r[2].max(cx), r[3].max(cy)];
    }
    [r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad]
}

/// Node-local bounds `[x, y, w, h]` of path commands' points (control points
/// included), for a cheap reach estimate before outlining.
pub fn cmds_bounds(cmds: &[PathCmd]) -> [f32; 4] {
    let mut r = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut add = |x: f64, y: f64| {
        if x.is_finite() && y.is_finite() {
            r = [r[0].min(x), r[1].min(y), r[2].max(x), r[3].max(y)];
        }
    };
    for c in cmds {
        match *c {
            PathCmd::Move { x, y } | PathCmd::Line { x, y } => add(x, y),
            PathCmd::Curve { x, y, cp1, cp2 } => {
                add(x, y);
                add(cp1.0, cp1.1);
                add(cp2.0, cp2.1);
            }
            PathCmd::Quadratic { x, y, cp } => {
                add(x, y);
                add(cp.0, cp.1);
            }
            _ => {}
        }
    }
    if !r[0].is_finite() {
        return [0.0; 4];
    }
    [
        r[0] as f32,
        r[1] as f32,
        (r[2] - r[0]) as f32,
        (r[3] - r[1]) as f32,
    ]
}

thread_local! {
    /// The node being planned draws inside a layer capture (a subtree capture's group or frame): its geometry
    /// reaches the screen through the capture's composite, whatever its own box.
    static IN_CAPTURE: Cell<bool> = const { Cell::new(false) };
}

/// Restores the previous capture flag on drop.
pub(crate) struct CaptureGuard(bool);

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        IN_CAPTURE.with(|c| c.set(self.0));
    }
}

/// Plan a node inside (`true`) or outside a layer capture.
pub(crate) fn enter_capture(inside: bool) -> CaptureGuard {
    CaptureGuard(IN_CAPTURE.with(|c| c.replace(inside)))
}

/// A refinement's canvas box, for viewport-first refinement: `bbox`, or the whole canvas inside a layer capture.
/// A capture composites (and blurs) the whole captured subtree over its region, so a stale body inside it changes
/// on-screen pixels even when its own box is off screen: it refines as if on screen, and keeps the frame refining
/// until it has.
pub fn refine_bbox(bbox: [f64; 4]) -> [f64; 4] {
    if IN_CAPTURE.with(Cell::get) {
        [f64::NEG_INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY]
    } else {
        bbox
    }
}

/// Restores the previous scale on drop.
pub(crate) struct ScaleGuard {
    camera: bool,
    prev: Option<f64>,
}

impl Drop for ScaleGuard {
    fn drop(&mut self) {
        if self.camera {
            CAMERA_SCALE.with(|c| c.set(self.prev));
        } else {
            PLAN_SCALE.with(|c| c.set(self.prev));
        }
    }
}

/// Enter a camera plan build at `zoom × dpr` (only when the scene has
/// outlines; `None` keeps the camera plan zoom-invariant).
pub(crate) fn enter_camera(px_per_unit: Option<f64>) -> ScaleGuard {
    let prev = CAMERA_SCALE.with(|c| c.replace(px_per_unit));
    ScaleGuard { camera: true, prev }
}

/// Enter a page plan build at `scale` (the export pixel ratio).
pub(crate) fn enter_plan(scale: f64) -> ScaleGuard {
    let prev = PLAN_SCALE.with(|c| c.replace(Some(scale)));
    ScaleGuard {
        camera: false,
        prev,
    }
}

/// The band the current build draws outlines and curves at: the view's band
/// (camera or export scale), raised by the scale of the node being
/// tessellated ([`enter_node_scale`], [`node_band`]).
pub fn current_band() -> i32 {
    node_band(view_band(), NODE_SCALE.with(Cell::get))
}

/// The band of the view alone (camera `zoom × dpr`, or the export pixel
/// ratio), without the node's own scale.
pub fn view_band() -> i32 {
    let s = CAMERA_SCALE
        .with(Cell::get)
        .or_else(|| PLAN_SCALE.with(Cell::get))
        .unwrap_or(1.0);
    band_for_scale(s)
}

/// Highest band a node's own scale takes it to: 2^16 device px per node-local
/// unit. Outline fitting stays capped at [`MAX_BAND`] ([`tolerance_for_band`]
/// clamps); curve flattening and round caps / joins follow the band up to
/// here.
pub const MAX_NODE_BAND: i32 = 16;

thread_local! {
    /// The largest scale of the node being tessellated: its canvas affine's
    /// linear part (its own scale × its parents', animated values included),
    /// as [`affine_max_scale`] measures it. 1 outside a node.
    static NODE_SCALE: Cell<f64> = const { Cell::new(1.0) };
}

/// Restores the previous node scale on drop.
pub(crate) struct NodeScaleGuard(f64);

impl Drop for NodeScaleGuard {
    fn drop(&mut self) {
        NODE_SCALE.with(|c| c.set(self.0));
    }
}

/// Tessellate the next node at `scale` canvas units per node-local unit
/// ([`affine_max_scale`] of its canvas affine): its curves flatten for the
/// size they reach the screen at, not their source size.
pub(crate) fn enter_node_scale(scale: f64) -> NodeScaleGuard {
    NodeScaleGuard(NODE_SCALE.with(|c| c.replace(scale)))
}

/// The largest singular value of `affine`'s linear part `[a c; b d]`: how far
/// a node-local unit can stretch on the canvas in any direction (the larger
/// axis of a non-uniform scale, skew included). 1 for a degenerate or
/// non-finite affine.
pub fn affine_max_scale(affine: &moda_scene::types::Affine) -> f64 {
    let (a, b, c, d) = (affine.a, affine.b, affine.c, affine.d);
    let sum = 0.5 * (a * a + b * b + c * c + d * d);
    let diff = 0.5 * (a * a + b * b - c * c - d * d);
    let cross = a * c + b * d;
    let s = (sum + diff.hypot(cross)).sqrt();
    if s.is_finite() && s > 0.0 {
        s
    } else {
        1.0
    }
}

/// The band a node draws its curves at: the view's band raised by the node's
/// own scale (`node_scale` canvas units per node-local unit) in the same
/// power-of-two steps, `round(view_band + log2(node_scale))`, up to
/// [`MAX_NODE_BAND`]. A small shape scaled up flattens for its on-screen
/// size (within a device pixel), and a scale change within the same octave
/// keeps its band and so its cached tessellation. A scale below 1 never lowers
/// the view's band.
pub fn node_band(view_band: i32, node_scale: f64) -> i32 {
    if !(node_scale.is_finite() && node_scale > 1.0) {
        return view_band;
    }
    let band = (f64::from(view_band) + node_scale.log2()).round();
    (band as i32).clamp(view_band, MAX_NODE_BAND.max(view_band))
}

/// The camera band folded into plan keys (`None` outside a camera build or
/// for a scene without outlines).
pub fn camera_key_band() -> Option<i32> {
    CAMERA_SCALE.with(Cell::get).map(band_for_scale)
}

/// Salt for the per-node tess cache key of an outlined node: the zoom band
/// and the brush definition's content hash (`SceneDocument::brush_hash`), so
/// a brush-table edit re-outlines its strokes.
pub fn outline_cache_salt(band: i32, brush_hash: Option<u64>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    b"stroke-outline".hash(&mut h);
    band.hash(&mut h);
    brush_hash.hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------------------
// Trim + dash → arc-length windows
// ---------------------------------------------------------------------------

const TRIM_EPS: f64 = 1e-4;
const MIN_SUBPATH: f64 = 0.001;

/// A visible run of one subpath: `[start, end]` in its arc length (`end` may
/// pass the seam of a closed subpath), `trimmed` when it came from trim.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Run {
    subpath: usize,
    start: f64,
    end: f64,
    len: f64,
    closed: bool,
    trimmed: bool,
}

/// Trim windows, mirroring `apply_trim_to_path` (tessellator.rs) on the
/// outliner's own subpath lengths.
fn trim_runs(spans: &[(usize, f64, bool)], closed: bool, t: TrimParams) -> Vec<Run> {
    let mut out = Vec::new();
    if spans.is_empty() {
        return out;
    }
    let start = t.start.clamp(0.0, 1.0);
    let end0 = t.end.clamp(0.0, 1.0);
    let off = t.offset.rem_euclid(1.0);
    let end = if end0 < start && closed {
        end0 + 1.0
    } else {
        end0
    };
    let push =
        |out: &mut Vec<Run>, (i, len, sclosed): (usize, f64, bool), a: f64, b: f64, wrap: bool| {
            if len <= TRIM_EPS || b - a <= TRIM_EPS {
                return;
            }
            let run = |s: f64, e: f64| Run {
                subpath: i,
                start: s,
                end: e,
                len,
                closed: sclosed,
                trimmed: true,
            };
            if b - a >= len - TRIM_EPS {
                out.push(run(0.0, len));
                return;
            }
            let s = a.rem_euclid(len);
            let e = s + (b - a);
            if e <= len + TRIM_EPS {
                out.push(run(s, e.min(len)));
            } else if wrap {
                out.push(run(s, e));
            } else {
                out.push(run(s, len));
                out.push(run(0.0, e - len));
            }
        };
    if t.individual && spans.len() > 1 {
        let total: f64 = spans.iter().map(|s| s.1).sum();
        if total <= TRIM_EPS {
            return out;
        }
        let a = (start + off) * total;
        let b = (end + off) * total;
        if b - a <= TRIM_EPS {
            return out;
        }
        let mut cursor = 0.0;
        for &sp in spans {
            let seg_start = cursor;
            cursor += sp.1;
            for shift in [-total, 0.0, total] {
                let lo = (a + shift).max(seg_start);
                let hi = (b + shift).min(seg_start + sp.1);
                if hi - lo > TRIM_EPS {
                    push(&mut out, sp, lo - seg_start, hi - seg_start, false);
                }
            }
        }
        return out;
    }
    for &sp in spans {
        push(
            &mut out,
            sp,
            (start + off) * sp.1,
            (end + off) * sp.1,
            closed,
        );
    }
    out
}

/// A trim along an axis (`trimPathBy: 'x' | 'y'`, `super::trim_axis`) as runs on the outliner's subpaths: the
/// windows measured on the path's own segments, scaled to the outliner's lengths. `None` for a trim by length, or
/// when the two don't split the path into the same subpaths (the trim is then by length).
fn axis_trim_runs(
    all_spans: &[moda_scene::stroke_outline::SubpathSpan],
    node: &RawNode,
    closed: bool,
    t: TrimParams,
) -> Option<Vec<Run>> {
    t.by.axis()?;
    let per = super::axis_trim_arc_windows(&super::path_cmds_for_node(node), closed, t)?;
    if per.len() != all_spans.len() {
        return None;
    }
    let mut out = Vec::new();
    for (i, (span, (len, windows))) in all_spans.iter().zip(per).enumerate() {
        if span.length <= MIN_SUBPATH || len <= 0.0 {
            continue;
        }
        let k = span.length / len;
        for (a, b) in windows {
            if (b - a) * k > TRIM_EPS {
                out.push(Run {
                    subpath: i,
                    start: a * k,
                    end: b * k,
                    len: span.length,
                    closed: span.closed,
                    trimmed: true,
                });
            }
        }
    }
    Some(out)
}

/// Dash intervals over `[0, len]`, mirroring the arc-length dash walk
/// (`walk_path_by_arc_length[_phased]`): pieces shorter than 0.001 are
/// dropped, the pattern runs continuously through vertices.
fn dash_intervals(len: f64, pattern: &[f64], offset: f64, budget: &mut usize) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = Vec::new();
    let total: f64 = pattern.iter().sum();
    if pattern.is_empty() || !(total > 0.0) || len < 0.001 {
        return out;
    }
    let (mut idx, mut on, mut rem) = if offset == 0.0 {
        (0, true, pattern[0])
    } else {
        seed_dash_phase(pattern, offset)
    };
    let mut pos = 0.0;
    let mut guard = 0usize;
    while len - pos >= 0.001 {
        guard += 1;
        if guard > 4 * MAX_WINDOWS + 64 || *budget == 0 {
            break;
        }
        let step = rem.min(len - pos).max(0.0);
        if on && step > 0.001 {
            match out.last_mut() {
                Some(last) if (last.1 - pos).abs() <= 1e-9 => last.1 = pos + step,
                _ => {
                    out.push((pos, pos + step));
                    *budget -= 1;
                }
            }
        }
        pos += step;
        rem -= step;
        if rem < 0.001 {
            idx = (idx + 1) % pattern.len();
            rem = pattern[idx];
            on = !on;
        }
    }
    out
}

/// Trim and dash as arc-length windows on the outliner's subpaths. `None` =
/// no trim and no dash (outline the whole stroke). The flag: the dash pattern
/// was cut at [`MAX_WINDOWS`].
pub fn arc_windows(
    node: &RawNode,
    trim: Option<TrimParams>,
    dash: Option<&[f64]>,
    dash_offset: f64,
    stroke_width: f64,
) -> (Option<Vec<ArcWindow>>, bool) {
    let trim = trim.filter(|t| !is_trim_identity(t.start, t.end, t.offset));
    let dash = dash.filter(|d| !d.is_empty());
    if trim.is_none() && dash.is_none() {
        return (None, false);
    }
    arc_windows_on(
        &subpath_spans(node),
        node,
        trim,
        dash,
        dash_offset,
        stroke_width,
    )
}

/// [`arc_windows`] on already-measured subpaths (`subpath_spans(node)`), for
/// callers that also need the lengths.
pub fn arc_windows_on(
    all_spans: &[moda_scene::stroke_outline::SubpathSpan],
    node: &RawNode,
    trim: Option<TrimParams>,
    dash: Option<&[f64]>,
    dash_offset: f64,
    stroke_width: f64,
) -> (Option<Vec<ArcWindow>>, bool) {
    let trim = trim.filter(|t| !is_trim_identity(t.start, t.end, t.offset));
    let dash = dash.filter(|d| !d.is_empty());
    if trim.is_none() && dash.is_none() {
        return (None, false);
    }
    let closed = node.bool("closed").unwrap_or(false);
    let spans: Vec<(usize, f64, bool)> = all_spans
        .iter()
        .enumerate()
        .filter(|(_, s)| s.length > MIN_SUBPATH)
        .map(|(i, s)| (i, s.length, s.closed))
        .collect();
    let runs: Vec<Run> = match trim {
        Some(t) => axis_trim_runs(all_spans, node, closed, t).unwrap_or_else(|| trim_runs(&spans, closed, t)),
        None => spans
            .iter()
            .map(|&(i, len, c)| Run {
                subpath: i,
                start: 0.0,
                end: len,
                len,
                closed: c,
                trimmed: false,
            })
            .collect(),
    };
    let extent = |r: &Run| r.trimmed.then_some((r.start, r.end));
    let Some(d) = dash else {
        return (
            Some(
                runs.iter()
                    .map(|r| ArcWindow {
                        subpath: r.subpath,
                        start: r.start,
                        end: r.end,
                        taper_extent: extent(r),
                    })
                    .collect(),
            ),
            false,
        );
    };
    let pattern = normalize_dash_pattern(scale_dash_pattern(d, stroke_width));
    let offset = dash_offset * stroke_width / BASE_DASH_STROKE_WIDTH;
    // Aligned dashes ([`super::with_dash_alignment`]): each untrimmed subpath's pattern is fitted between its
    // corners, measured on the path's own segments and scaled to the outliner's length. Only when both split the
    // path into the same subpaths.
    let corners: Option<Vec<(f64, Vec<f64>)>> = super::dash_alignment_now()
        .then(|| super::subpath_corners(&super::path_cmds_for_node(node), closed))
        .filter(|c| c.len() == all_spans.len());
    let mut budget = MAX_WINDOWS;
    let mut out = Vec::new();
    for r in &runs {
        let len = r.end - r.start;
        let aligned = corners.as_ref().filter(|_| !r.trimmed).map(|c| {
            let (total, at) = &c[r.subpath];
            let k = if *total > 0.0 { len / total } else { 1.0 };
            let at: Vec<f64> = at.iter().map(|x| x * k).collect();
            let iv = aligned_dash_intervals(len, &at, r.closed, &pattern);
            let iv: Vec<(f64, f64)> = iv.into_iter().take(budget).collect();
            budget -= iv.len();
            iv
        });
        let mut iv = match aligned {
            Some(iv) => iv,
            None => dash_intervals(len, &pattern, offset, &mut budget),
        };
        // An untrimmed closed subpath: the dash through the seam is one piece,
        // and a dash that never turns off is the closed ring.
        if !r.trimmed && r.closed && !iv.is_empty() {
            let first_on = iv[0].0 <= 1e-9;
            let last_on = (iv[iv.len() - 1].1 - len).abs() <= 1e-6;
            if first_on && last_on {
                if iv.len() == 1 {
                    iv[0] = (0.0, len);
                } else {
                    let tail = iv.pop().unwrap_or_default();
                    iv[0] = (tail.0, len + iv[0].1);
                }
            }
        }
        for (a, b) in iv {
            out.push(ArcWindow {
                subpath: r.subpath,
                start: r.start + a,
                end: r.start + b,
                taper_extent: extent(r),
            });
        }
    }
    (Some(out), budget == 0)
}

/// The wasm `stroke_outline` export: [`moda_scene::stroke_outline_json`] plus the node's trim
/// (`trimPathStart` / `trimPathEnd` / `trimPathOffset` / `trimPathMode` / `trimPathBy` in the request), resolved to arc-length
/// windows exactly as the renderer resolves them ([`arc_windows`]): a trimmed stroke outlines only its visible
/// run, with taper evaluated over that run (Outline Stroke on a write-on frame). A dashed stroke (the request
/// stroke's `dash` / `dashOffset`) outlines as its separate dash pieces, the same windows the canvas draws.
pub fn stroke_outline_trimmed_json(request_json: &str) -> String {
    moda_scene::stroke_outline_json_with(request_json, &|node, request| {
        let trim = super::trim_params_from_fields(
            |k| request.get(k).and_then(Value::as_f64),
            request.get("trimPathMode").and_then(Value::as_str),
            request.get("trimPathBy").and_then(Value::as_str),
        );
        let style = moda_scene::paint::model_stroke_style(node);
        let dash: Option<Vec<f64>> = style.dash.clone().filter(|d| !d.is_empty());
        if trim.is_none() && dash.is_none() {
            return None;
        }
        let offset = style.dash_offset.filter(|v| v.is_finite()).unwrap_or(0.0);
        let width = style.width.unwrap_or(1.0);
        // A trim that shows nothing is an empty window list (→ "empty"), not "no trim".
        Some(
            super::with_dash_alignment(style.dash_aligned(), || {
                arc_windows(node, trim, dash.as_deref(), offset, width)
            })
            .0
            .unwrap_or_default(),
        )
    })
}

// ---------------------------------------------------------------------------
// Outline → contours → fill
// ---------------------------------------------------------------------------

/// Flatten the outline's move / line / curve points into closed contours.
fn flatten(points: &[PathPoint], tol: f64) -> Vec<Vec<[f64; 2]>> {
    let mut out: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut cur = [0.0, 0.0];
    for p in points {
        match *p {
            PathPoint::Move { x, y } => {
                cur = [x, y];
                out.push(vec![cur]);
            }
            PathPoint::Line { x, y } => {
                cur = [x, y];
                if let Some(c) = out.last_mut() {
                    c.push(cur);
                }
            }
            PathPoint::Curve { x, y, cp1, cp2 } => {
                let (p0, p1, p2, p3) = (cur, [cp1.0, cp1.1], [cp2.0, cp2.1], [x, y]);
                // Uniform subdivision bound: error ≤ 3/4 · max|second difference| / n².
                let dd = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
                    (a[0] - 2.0 * b[0] + c[0]).hypot(a[1] - 2.0 * b[1] + c[1])
                };
                let l = dd(p0, p1, p2).max(dd(p1, p2, p3));
                let n = ((0.75 * l / tol.max(1e-6)).sqrt().ceil() as usize).clamp(1, 256);
                if let Some(c) = out.last_mut() {
                    for i in 1..=n {
                        let t = i as f64 / n as f64;
                        let u = 1.0 - t;
                        let (a, b, cc, d) =
                            (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
                        c.push([
                            a * p0[0] + b * p1[0] + cc * p2[0] + d * p3[0],
                            a * p0[1] + b * p1[1] + cc * p2[1] + d * p3[1],
                        ]);
                    }
                }
                cur = p3;
            }
            // The outliner emits only move / line / curve.
            _ => {}
        }
    }
    out.retain(|c| c.len() >= 3);
    out
}

/// The outline's contours at `tol` (empty = nothing to draw).
fn outline_contours(
    node: &RawNode,
    brush: Option<&Value>,
    windows: Option<&[ArcWindow]>,
    tol: f64,
) -> Result<Vec<Vec<[f64; 2]>>, String> {
    let opts = OutlineOptions {
        tolerance: tol,
        ..OutlineOptions::default()
    };
    let r = match windows {
        Some(ws) => stroke_outline_windows(node, brush, &opts, ws),
        None => stroke_outline(node, brush, &opts),
    };
    match r {
        Ok(Outline::Empty) => Ok(Vec::new()),
        Ok(Outline::Shape { points, .. }) => Ok(flatten(&points, tol)),
        Err(e) => Err(format!("{}: {}", e.code, e.message)),
    }
}

/// The outline of `node`'s top visible stroke drawn along `cmds` (node-local) at `width`, as closed contours to fill
/// nonzero (the 3D extruder's stroke parts, `solid::outline`). `cap` / `join` override the stroke's own (a line's
/// parts carry theirs); `width_model` keeps its profile / taper / wave (a body), else the width is uniform (an
/// arrowhead caret). Centred: the caller clips an aligned stroke. Dash, brush and wobble are not applied.
pub(crate) fn stroke_contours(
    node: &RawNode,
    cmds: &[PathCmd],
    closed: bool,
    width: f64,
    cap: Option<&str>,
    join: Option<&str>,
    width_model: bool,
    tol: f64,
) -> Vec<Vec<[f64; 2]>> {
    let Some(mut stroke) = moda_scene::paint::top_stroke(node) else { return Vec::new() };
    if !(width.is_finite() && width > 0.0) || cmds.is_empty() {
        return Vec::new();
    }
    if let Some(c) = cap {
        stroke.cap = Some(c.to_string());
    }
    if let Some(j) = join {
        stroke.join = Some(j.to_string());
    }
    if !width_model {
        stroke.profile = None;
        stroke.taper = None;
        stroke.wave = None;
    }
    let mut v = json!({"id": node.id(), "type": "path", "x": 0, "y": 0, "points": cmds_points(cmds), "closed": closed});
    let Some(m) = v.as_object_mut() else { return Vec::new() };
    moda_scene::paint::set_width_model(m, &stroke, width);
    let Some(path) = RawNode::from_value(v) else { return Vec::new() };
    outline_contours(&path, None, None, tol).unwrap_or_default()
}

/// A line's body as the path the outliner measures and outlines: its drawn
/// commands (routing, corner rounding and the arrowhead set-back already applied
/// by `line_to_path`), open, with the line's width model (§4.9). Markers are
/// fractions of this body's length.
pub fn line_body_as_path(node: &RawNode, cmds: &[PathCmd], stroke_width: f64) -> RawNode {
    let mut v = json!({"id": node.id(), "type": "path", "x": 0, "y": 0, "points": cmds_points(cmds),
        "closed": false});
    if let (Some(m), Some(stroke)) = (v.as_object_mut(), moda_scene::paint::top_stroke(node)) {
        moda_scene::paint::set_width_model(m, &stroke, stroke_width);
    }

    RawNode::from_value(v).unwrap_or_else(|| node.clone())
}

/// Mark `spec` (a line needing an outline) as outlined, with the arrowhead
/// set-back's cap allowance: how far a round / square cap reaches past each
/// arrowed end, from the width model at the body's FINAL length (brushes.md
/// §4.9). A cap spans both sides, so it reaches their mean; an end tapered to
/// a point reaches nothing. No outline is built: the body is measured once
/// (butt set-back) and the widths come from the outliner's own width
/// functions (`open_end_half_widths`). The final length depends on the reach
/// (a percent taper, a px-wavelength wave's phase), so it is settled by a
/// few damped fixed-point steps.
pub fn outlined_line_spec(node: &RawNode, spec: &mut super::line_to_path::LineSpec) {
    use super::line_to_path as ltp;
    spec.body_outlined = true;
    spec.outlined_cap_reach = [0.0, 0.0];
    let capped = matches!(spec.line_cap.as_deref(), Some("round" | "square"));
    let arrowed = [spec.start_arrow.is_some(), spec.end_arrow.is_some()];
    if !capped || !(arrowed[0] || arrowed[1]) {
        return;
    }
    let mut probe = spec.clone();
    probe.dash = None;
    let body = ltp::convert_line_body(&probe);
    if body.stroke_width <= 0.0 {
        return;
    }
    let butt_len = cmds_length(&body.cmds);
    let path = line_body_as_path(node, &body.cmds, body.stroke_width);
    let mut reach = [0.0_f64; 2];
    for step in 0..10 {
        let len = butt_len
            - (0..2)
                .filter(|&i| arrowed[i])
                .map(|i| reach[i])
                .sum::<f64>();
        let Some(ends) = moda_scene::stroke_outline::open_end_half_widths(&path, len.max(1e-3))
        else {
            return;
        };
        for i in 0..2 {
            let target = if arrowed[i] {
                0.5 * (ends[i].0 + ends[i].1).max(0.0)
            } else {
                0.0
            };
            // The first step takes the reach at the butt length; later ones are damped (a short
            // wavelength can make the plain iteration overshoot).
            reach[i] = if step == 0 {
                target
            } else {
                0.5 * (reach[i] + target)
            };
        }
    }
    spec.outlined_cap_reach = reach;
}

/// Arc length of `cmds` (all subpaths), as the outliner measures it.
fn cmds_length(cmds: &[PathCmd]) -> f64 {
    let v = json!({"id": "len", "type": "path", "x": 0, "y": 0, "points": cmds_points(cmds), "closed": false});
    RawNode::from_value(v)
        .map(|n| subpath_spans(&n).iter().map(|s| s.length).sum())
        .unwrap_or(0.0)
}

/// A line's label gap as an arc-length interval on its un-gapped body (`full`), read off the
/// gapped body (`gapped`: the run before the gap, the run after it, or one of them when the gap
/// reaches an end), so it holds for every routing. `None` when nothing is carved.
pub fn line_gap_interval(full: &[PathCmd], gapped: &[PathCmd]) -> Option<(f64, f64)> {
    let lengths = |cmds: &[PathCmd]| -> Vec<f64> {
        let v = json!({"id": "gap", "type": "path", "x": 0, "y": 0, "points": cmds_points(cmds), "closed": false});
        RawNode::from_value(v)
            .map(|n| subpath_spans(&n).iter().map(|s| s.length).collect())
            .unwrap_or_default()
    };
    let total: f64 = lengths(full).iter().sum();
    let runs = lengths(gapped);
    let first = |cmds: &[PathCmd]| match cmds.first() {
        Some(PathCmd::Move { x, y }) => Some((*x, *y)),
        _ => None,
    };
    let (a, b) = match runs.as_slice() {
        [] => (0.0, total),
        [only]
            if first(full)
                .zip(first(gapped))
                .is_some_and(|(p, q)| (p.0 - q.0).abs() + (p.1 - q.1).abs() < 1e-6) =>
        {
            (*only, total)
        }
        [only] => (0.0, total - only),
        [before, .., after] => (*before, total - after),
    };
    (b - a > MIN_SUBPATH).then_some((a.max(0.0), b.min(total)))
}

/// `windows` (or, for `None`, every subpath whole) with the arc-length interval `cut` removed
/// from subpath 0 (a line body is one open subpath). Taper extents are kept, so the stroke
/// tapers over the whole body, not per piece.
fn cut_windows(node: &RawNode, windows: Option<Vec<ArcWindow>>, cut: (f64, f64)) -> Vec<ArcWindow> {
    let windows = windows.unwrap_or_else(|| {
        subpath_spans(node)
            .iter()
            .enumerate()
            .filter(|(_, s)| s.length > MIN_SUBPATH)
            .map(|(i, s)| ArcWindow {
                subpath: i,
                start: 0.0,
                end: s.length,
                taper_extent: None,
            })
            .collect()
    });
    let (a, b) = cut;
    let mut out = Vec::with_capacity(windows.len() + 1);
    for w in windows {
        if w.subpath != 0 || w.end <= a || w.start >= b {
            out.push(w);
            continue;
        }
        if a - w.start > MIN_SUBPATH {
            out.push(ArcWindow { end: a, ..w });
        }
        if w.end - b > MIN_SUBPATH {
            out.push(ArcWindow { start: b, ..w });
        }
    }
    out
}

fn cmds_points(cmds: &[PathCmd]) -> Vec<Value> {
    cmds
        .iter()
        .map(|c| match *c {
            PathCmd::Move { x, y } => json!({"type": "move", "x": x, "y": y}),
            PathCmd::Line { x, y } => json!({"type": "line", "x": x, "y": y}),
            PathCmd::Curve { x, y, cp1, cp2 } => json!({"type": "curve", "x": x, "y": y,
                "cp1": {"x": cp1.0, "y": cp1.1}, "cp2": {"x": cp2.0, "y": cp2.1}}),
            PathCmd::Quadratic { x, y, cp } => json!({"type": "quadratic", "x": x, "y": y, "cp": {"x": cp.0, "y": cp.1}}),
            PathCmd::Arc { x, y, rx, ry, x_axis_rotation, large_arc, sweep } => json!({"type": "arc", "x": x, "y": y,
                "rx": rx, "ry": ry, "xAxisRotation": x_axis_rotation, "largeArc": large_arc, "sweep": sweep}),
        })
        .collect()
}

/// Everything the body build knows about the path besides the node.
pub struct OutlineBody<'a> {
    pub cmds: &'a [PathCmd],
    pub closed: bool,
    pub fill_rule: FillRule,
    pub stroke_width: f64,
    pub trim: Option<TrimParams>,
    pub dash: Option<&'a [f64]>,
    pub dash_offset: f64,
    pub line_cap: LineCap,
    pub line_join: LineJoin,
    pub stroke_align: StrokeAlign,
    /// An arc-length interval cut out of the outline after trim and dash (a
    /// line's label gap, [`line_gap_interval`]).
    pub cut: Option<(f64, f64)>,
}

/// Tessellate a path body whose stroke is an outline: the body fill (as
/// today) + the outline fill as stroke vertices. The second value is a
/// `skipped`-class note whenever the outline was not filled whole at its
/// tolerance (cells, coarsened, or the uniform fallback).
///
/// Ladder: (1) the outline filled nonzero — whole, or, when lyon's
/// sweep-cost guard refuses it, bisected into clipped cells that tile the
/// same coverage exactly (no double blending, no interior caps); (2) the
/// same at 2× / 4× the tolerance (fewer vertices) if the stroke vertex budget
/// is exceeded; (3) the uniform stroke. Never a page refusal.
pub fn tessellate_outlined(
    node: &RawNode,
    brush: Option<&Value>,
    body: &OutlineBody<'_>,
    band: i32,
) -> (TessellatedGeometry, Option<String>) {
    let fill = body_fill(body.cmds, body.closed, body.fill_rule);
    if body.stroke_width <= 0.0 {
        return (combine_fill_and_outline(fill, None), None);
    }
    let (windows, truncated) = arc_windows(
        node,
        body.trim,
        body.dash,
        body.dash_offset,
        body.stroke_width,
    );
    let windows = match body.cut {
        Some(cut) => Some(cut_windows(node, windows, cut)),
        None => windows,
    };
    let mut notes: Vec<String> = Vec::new();
    if truncated {
        notes.push(format!("dash pattern cut at {MAX_WINDOWS} dashes"));
    }
    if windows.as_ref().is_some_and(Vec::is_empty) {
        return (combine_fill_and_outline(fill, None), join_notes(notes));
    }
    let base_tol = tolerance_for_band(band);
    let mut why = String::new();
    for mul in [1.0, 2.0, 4.0] {
        let tol = base_tol * mul;
        let contours = match outline_contours(node, brush, windows.as_deref(), tol) {
            Ok(c) => c,
            Err(e) => {
                why = e;
                break;
            }
        };
        if contours.is_empty() {
            return (combine_fill_and_outline(fill, None), join_notes(notes));
        }
        match fill_outline_banded(&contours) {
            Ok((tris, b, cells)) => {
                if cells > 1 {
                    notes.push(format!(
                        "stroke outline too complex for one fill; filled in {cells} cells"
                    ));
                }
                if mul > 1.0 {
                    notes.push(format!(
                        "stroke outline coarsened to {tol:.3}-unit tolerance ({why})"
                    ));
                }
                return (
                    combine_fill_and_outline(fill, Some((tris, b))),
                    join_notes(notes),
                );
            }
            Err(e) => why = e,
        }
    }
    // The uniform stroke (same trim / dash).
    let g = tessellate_combined_with_dash_trim(
        body.cmds,
        body.stroke_width,
        body.closed,
        body.trim,
        body.dash,
        body.dash_offset,
        body.line_cap,
        body.line_join,
        body.fill_rule,
        body.stroke_align,
    );
    notes.push(format!(
        "stroke outline not drawable ({why}); drew a uniform stroke"
    ));
    (g, join_notes(notes))
}

/// Trim parameters of a node (`None` when absent or identity) — the one
/// derivation the body builder and [`tessellate_outlined_node`] share.
pub fn trim_params_from_node(node: &RawNode) -> Option<TrimParams> {
    super::trim_params_from_fields(|k| node.f64(k), node.str("trimPathMode"), node.str("trimPathBy"))
}

/// [`tessellate_outlined`] for a path node from its own fields — what the
/// body builder draws for it at `band` (same derivation of path commands,
/// fill rule, caps, joins, trim and dash). Pure and GPU-free: the WASM
/// `outline_tessellation` export runs it in a worker.
pub fn tessellate_outlined_node(
    node: &RawNode,
    brush: Option<&Value>,
    band: i32,
) -> (TessellatedGeometry, Option<String>) {
    use super::moda_engine_path_types as pt;
    let cmds = super::path_cmds_for_node(node);
    let style = moda_scene::paint::node_stroke_style(node);
    let dash = style.dash.clone();
    let body = OutlineBody {
        cmds: &cmds,
        closed: node.bool("closed").unwrap_or(false),
        fill_rule: pt::fill_rule_from_str(node.str("fillRule")),
        stroke_width: crate::paint::painted_stroke_width(node),
        trim: trim_params_from_node(node),
        dash: dash.as_deref().filter(|d| !d.is_empty()),
        dash_offset: style.dash_offset.unwrap_or(0.0),
        line_cap: pt::line_cap_from_str(style.cap.as_deref()),
        line_join: pt::line_join_from_str(style.join.as_deref()),
        stroke_align: pt::stroke_align_from_str(style.align.as_deref()),
        cut: None,
    };
    super::with_dash_alignment(style.dash_aligned(), || {
        tessellate_outlined(node, brush, &body, band)
    })
}

fn join_notes(notes: Vec<String>) -> Option<String> {
    (!notes.is_empty()).then(|| notes.join("; "))
}

#[cfg(test)]
mod tests {
    use super::super::tessellator::fill_outline_contours;
    use super::*;
    use moda_scene::value::json;

    fn node(v: Value) -> RawNode {
        let mut v = v;
        v["type"] = json!("path");
        v["id"] = json!("p");
        RawNode::from_value(v).unwrap()
    }

    fn line(len: f64) -> Value {
        json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": len, "y": 0}])
    }

    /// `{"strokes": [a black stroke of `width` with `extra`'s fields]}`.
    fn stroke_of(width: f64, extra: Value) -> Value {
        let mut layer = json!({"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": width});
        for (k, v) in extra.as_object().unwrap() {
            layer[k] = v.clone();
        }
        json!({"strokes": [layer]})
    }

    /// `base` with `extra`'s fields merged in.
    fn with(mut base: Value, extra: Value) -> Value {
        for (k, v) in extra.as_object().unwrap() {
            base[k] = v.clone();
        }
        base
    }

    #[test]
    fn lines_outline_their_width_model_only() {
        // brushes.md §4.9: lines carry the profile, taper and wave; not brushes or the wobble.
        let none = |_: &str| -> Option<&Value> { None };
        // `extra` is the line's stroke layer fields (a black 10 px stroke).
        let line_node = |extra: Value| {
            let v = with(json!({"id": "l", "type": "line", "points": [0, 0, 200, 0]}), stroke_of(10.0, extra));
            RawNode::from_value(v).unwrap()
        };
        assert!(!needs_outline(&line_node(json!({})), none));
        assert!(needs_outline(
            &line_node(json!({"profile": {"preset": "eye"}})),
            none
        ));
        assert!(needs_outline(
            &line_node(json!({"taper": {"startLength": 20}})),
            none
        ));
        assert!(needs_outline(
            &line_node(json!({"wave": {"amount": 0.2}})),
            none
        ));
        let cal_def = json!({"id": "c", "type": "calligraphic"});
        let cal = |id: &str| (id == "c").then_some(&cal_def);
        assert!(!needs_outline(
            &line_node(json!({"brush": {"brushId": "c"}})),
            cal
        ));
        assert!(!needs_outline(
            &line_node(json!({"dynamic": {"wiggle": 3}})),
            none
        ));

        // The body is outlined as a path over the line's drawn commands: an eye profile is thin at the
        // ends and full in the middle.
        let eye = line_node(json!({"profile": {"preset": "eye"}}));
        let cmds = [
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Line { x: 200.0, y: 0.0 },
        ];
        let body = line_body_as_path(&eye, &cmds, 10.0);
        assert_eq!(body.node_type(), "path");
        let contours = outline_contours(&body, None, None, 0.1).unwrap();
        let half_at = |x0: f64| {
            contours
                .iter()
                .flatten()
                .filter(|p| (p[0] - x0).abs() < 3.0)
                .map(|p| p[1].abs())
                .fold(0.0_f64, f64::max)
        };
        assert!(
            (half_at(100.0) - 5.0).abs() < 0.3,
            "full width in the middle: {}",
            half_at(100.0)
        );
        assert!(half_at(4.0) < 2.0, "thin at the start: {}", half_at(4.0));
    }

    #[test]
    fn labelled_line_outlines_whole_then_cuts_the_gap() {
        use crate::tess::line_to_path::{line_parts_with_gap, LineSpec, TextGap};
        // An eye-profiled line with a label gap over 40%..60%: one eye over the whole line with the
        // gap cut out, not an eye per run (#132 review).
        let v = with(
            json!({"id": "l", "type": "line", "points": [0, 0, 200, 0]}),
            stroke_of(10.0, json!({"profile": {"preset": "eye"}})),
        );
        let node = RawNode::from_value(v.clone()).unwrap();
        let spec = LineSpec::from_json(|k| v.get(k));
        let gap = TextGap {
            start_t: 0.4,
            end_t: 0.6,
        };
        let full = &line_parts_with_gap(&spec, None)[0].cmds;
        let gapped = &line_parts_with_gap(&spec, Some(gap))[0].cmds;
        let (a, b) = line_gap_interval(full, gapped).unwrap();
        assert!(
            (a - 80.0).abs() < 1e-6 && (b - 120.0).abs() < 1e-6,
            "{a}..{b}"
        );

        let body = line_body_as_path(&node, full, 10.0);
        let windows = cut_windows(&body, None, (a, b));
        assert_eq!(windows.len(), 2);
        let contours = outline_contours(&body, None, Some(&windows), 0.1).unwrap();
        let pts: Vec<[f64; 2]> = contours.into_iter().flatten().collect();
        assert!(
            pts.iter()
                .all(|p| p[0] <= 80.0 + 1e-6 || p[0] >= 120.0 - 1e-6),
            "nothing drawn in the gap"
        );
        let half_at = |x0: f64| {
            pts.iter()
                .filter(|p| (p[0] - x0).abs() < 2.0)
                .map(|p| p[1].abs())
                .fold(0.0_f64, f64::max)
        };
        // Beside the gap the eye is near its fullest; a per-run eye would pinch to its ends there.
        assert!(
            half_at(79.0) > 4.0,
            "full before the gap: {}",
            half_at(79.0)
        );
        assert!(
            half_at(121.0) > 4.0,
            "full after the gap: {}",
            half_at(121.0)
        );
        assert!(
            half_at(3.0) < 2.0,
            "thin at the line's start: {}",
            half_at(3.0)
        );

        // The gap at an end: one run left.
        let at_end = &line_parts_with_gap(
            &spec,
            Some(TextGap {
                start_t: 0.8,
                end_t: 1.0,
            }),
        )[0]
        .cmds;
        let (a, b) = line_gap_interval(full, at_end).unwrap();
        assert!(
            (a - 160.0).abs() < 1e-6 && (b - 200.0).abs() < 1e-6,
            "{a}..{b}"
        );
        let at_start = &line_parts_with_gap(
            &spec,
            Some(TextGap {
                start_t: 0.0,
                end_t: 0.3,
            }),
        )[0]
        .cmds;
        let (a, b) = line_gap_interval(full, at_start).unwrap();
        assert!(a.abs() < 1e-6 && (b - 60.0).abs() < 1e-6, "{a}..{b}");
        // No gap carved: nothing to cut.
        assert!(line_gap_interval(full, full).is_none());
    }

    #[test]
    fn outlined_line_body_meets_its_arrowhead() {
        use crate::tess::line_to_path::{line_parts_with_gap, LinePartKind, LineSpec};
        // A round-capped line sets its body back by the cap so the cap reaches the head; an outlined
        // body (which may taper to a point) is set back only by the head (#132 review).
        let v = with(
            json!({"id": "l", "type": "line", "points": [0, 0, 200, 0], "pointerLength": 30, "pointerWidth": 60}),
            stroke_of(40.0, json!({"cap": "round", "endArrow": "triangle"})),
        );
        let mut spec = LineSpec::from_json(|k| v.get(k));
        let body_end = |spec: &LineSpec| {
            let parts = line_parts_with_gap(spec, None);
            assert_eq!(parts[0].kind, LinePartKind::Body);
            match parts[0].cmds.last() {
                Some(PathCmd::Line { x, .. }) => *x,
                other => panic!("{other:?}"),
            }
        };
        assert!(
            (body_end(&spec) - 150.0).abs() < 1e-6,
            "{}",
            body_end(&spec)
        );
        spec.body_outlined = true;
        assert!(
            (body_end(&spec) - 170.0).abs() < 1e-6,
            "{}",
            body_end(&spec)
        );

        // Measured on the outline: a full-width round end reaches its whole half-width into the head's
        // base, as a plain line does, so it is set back by it (no bulge past the head); a body tapered
        // to a point at that end has no cap to allow for.
        // `extra`: stroke layer fields.
        let measured = |extra: Value| {
            let mut w = v.clone();
            for (k, x) in extra.as_object().unwrap() {
                w["strokes"][0][k] = x.clone();
            }
            let node = RawNode::from_value(w.clone()).unwrap();
            let mut spec = LineSpec::from_json(|k| w.get(k));
            outlined_line_spec(&node, &mut spec);
            (spec.outlined_cap_reach, body_end(&spec))
        };
        let (reach, end) = measured(json!({"profile": {"markers": [
            {"id": "a", "t": 0, "left": 1, "right": 1}, {"id": "b", "t": 1, "left": 1, "right": 1}]}}));
        assert!(
            (reach[1] - 20.0).abs() < 0.5 && (end - 150.0).abs() < 0.5,
            "full end: reach {reach:?}, body to {end}"
        );
        let taper = |end_width: f64| {
            json!({"taper": {"units": "percent", "startLength": 0, "endLength": 50,
            "startWidth": 0, "endWidth": end_width, "startEase": 0, "endEase": 0}})
        };
        let (reach, end) = measured(taper(0.0));
        assert!(
            reach[1] < 0.5 && (end - 170.0).abs() < 0.5,
            "tapered end: reach {reach:?}, body to {end}"
        );
        let (reach, end) = measured(taper(0.5));
        assert!(
            (reach[1] - 10.0).abs() < 0.5 && (end - 160.0).abs() < 0.5,
            "half-width end: reach {reach:?}, body to {end}"
        );
        // Asymmetric sides: the cap spans both, so it reaches their mean (30 left, 10 right → 20), not the wider.
        let (reach, end) = measured(json!({"profile": {"markers": [
            {"id": "a", "t": 0, "left": 1.5, "right": 0.5}, {"id": "b", "t": 1, "left": 1.5, "right": 0.5}]}}));
        assert!(
            (reach[1] - 20.0).abs() < 0.5 && (end - 150.0).abs() < 0.5,
            "asymmetric end: reach {reach:?}, body to {end}"
        );
        // A px-wavelength wave: its phase at the end depends on the body's final length, which depends on the
        // reach. The reach must match the wave at the end where the body actually stops.
        let (reach, end) = measured(
            json!({"wave": {"amount": 0.5, "units": "px", "wavelength": 37, "phase": 0}}),
        );
        let at_end = 20.0 * (1.0 + 0.5 * (std::f64::consts::TAU * end / 37.0).sin());
        assert!(
            (reach[1] - at_end).abs() < 0.3,
            "wave end: reach {reach:?} vs {at_end} at x {end}"
        );
        assert!(
            (end - (170.0 - reach[1])).abs() < 1e-6,
            "set back by head + reach: {end}"
        );
    }

    #[test]
    fn predicate_matches_ts() {
        let none = |_: &str| -> Option<&Value> { None };
        assert!(!needs_outline(&node(json!({"points": line(10.0)})), none));
        assert!(needs_outline(
            &node(stroke_of(1.0, json!({"profile": {"preset": "wedge"}}))),
            none
        ));
        assert!(needs_outline(&node(stroke_of(1.0, json!({"taper": {}}))), none));
        assert!(needs_outline(
            &node(stroke_of(1.0, json!({"wave": {"amount": 0}}))),
            none
        ));
        assert!(!needs_outline(&node(stroke_of(1.0, json!({"profile": null}))), none));
        assert!(!needs_outline(
            &node(stroke_of(1.0, json!({"dynamic": {"wiggle": 0}}))),
            none
        ));
        assert!(needs_outline(
            &node(stroke_of(1.0, json!({"dynamic": {"wiggle": 2}}))),
            none
        ));
        let cal_def = json!({"id": "c", "type": "calligraphic"});
        let art_def = json!({"id": "a", "type": "art"});
        let cal = |id: &str| (id == "c").then_some(&cal_def);
        let art = |id: &str| (id == "a").then_some(&art_def);
        assert!(needs_outline(
            &node(stroke_of(1.0, json!({"brush": {"brushId": "c"}}))),
            cal
        ));
        assert!(!needs_outline(
            &node(stroke_of(1.0, json!({"brush": {"brushId": "x"}}))),
            cal
        ));
        assert!(!needs_outline(
            &node(stroke_of(1.0, json!({"brush": {"brushId": "a"}}))),
            art
        ));
        let mut rect = node(stroke_of(1.0, json!({"profile": {"preset": "wedge"}})));
        rect.set("type", json!("rectangle"));
        assert!(!needs_outline(&rect, none));
    }

    #[test]
    fn bands() {
        assert_eq!(band_for_scale(1.0), 0);
        assert_eq!(band_for_scale(2.0), 1);
        assert_eq!(band_for_scale(1.3), 0);
        assert_eq!(band_for_scale(1.5), 1);
        assert_eq!(band_for_scale(1e-9), MIN_BAND);
        assert_eq!(band_for_scale(1e9), MAX_BAND);
        assert_eq!(band_for_scale(f64::NAN), 0);
        assert!((tolerance_for_band(1) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn node_scale_raises_the_band() {
        // A node at its source size keeps the view's band.
        assert_eq!(node_band(0, 1.0), 0);
        assert_eq!(node_band(-2, 1.0), -2);
        // ×100 is 6.64 octaves: band 7 at 1×, 8 on a 2× display or export.
        assert_eq!(node_band(0, 100.0), 7);
        assert_eq!(node_band(1, 100.0), 8);
        // Zoomed out, a scaled-up node still gets its own octaves.
        assert_eq!(node_band(-3, 100.0), 4);
        // Within half an octave: the same band, so no re-tessellation.
        assert_eq!(node_band(0, 1.3), 0);
        assert_eq!(node_band(0, 1.5), 1);
        assert_eq!(node_band(3, 1.41), 3);
        // A scale below 1 (or a degenerate one) never lowers the view's band.
        assert_eq!(node_band(2, 0.1), 2);
        assert_eq!(node_band(2, 0.0), 2);
        assert_eq!(node_band(2, f64::NAN), 2);
        assert_eq!(node_band(2, f64::INFINITY), 2);
        // Capped.
        assert_eq!(node_band(0, 1e12), MAX_NODE_BAND);
        assert_eq!(node_band(MAX_NODE_BAND + 2, 4.0), MAX_NODE_BAND + 2);
        // The flattening tolerance at the node's band (`tessellator::flatness_tolerance`: 0.5 / 2^band node-local
        // units) is at most one device pixel on screen, at any view and node scale under the cap.
        for view in [0.1, 0.25, 0.7, 1.0, 1.4, 2.0, 3.0, 8.0] {
            for scale in [1.0, 1.2, 1.9, 3.0, 37.0, 100.0, 250.0, 1000.0] {
                let band = node_band(band_for_scale(view), scale);
                let device_px = 0.5 / 2f64.powi(band) * scale * view;
                assert!(device_px <= 1.0 + 1e-9, "view {view} scale {scale}: {device_px} px");
            }
        }
    }

    #[test]
    fn affine_max_scale_is_the_largest_singular_value() {
        use moda_scene::types::Affine;
        let aff = |a: f64, b: f64, c: f64, d: f64| Affine { a, b, c, d, e: 7.0, f: -3.0 };
        let near = |x: f64, y: f64| assert!((x - y).abs() < 1e-9, "{x} vs {y}");
        near(affine_max_scale(&aff(1.0, 0.0, 0.0, 1.0)), 1.0);
        near(affine_max_scale(&aff(100.0, 0.0, 0.0, 100.0)), 100.0);
        // Non-uniform: the larger axis.
        near(affine_max_scale(&aff(3.0, 0.0, 0.0, 40.0)), 40.0);
        near(affine_max_scale(&aff(-50.0, 0.0, 0.0, 2.0)), 50.0);
        // Rotation doesn't change it.
        let (s, c) = 0.7f64.sin_cos();
        near(affine_max_scale(&aff(5.0 * c, 5.0 * s, -2.0 * s, 2.0 * c)), 5.0);
        // A shear [1 k; 0 1]: σ = (k + √(k² + 4)) / 2.
        let k = 3.0;
        near(affine_max_scale(&aff(1.0, 0.0, k, 1.0)), (k + (k * k + 4.0f64).sqrt()) / 2.0);
        // Degenerate / non-finite.
        near(affine_max_scale(&aff(0.0, 0.0, 0.0, 0.0)), 1.0);
        near(affine_max_scale(&aff(f64::NAN, 0.0, 0.0, 1.0)), 1.0);
    }

    #[test]
    fn node_scale_enters_the_current_band() {
        let _p = enter_plan(2.0);
        assert_eq!(current_band(), 1);
        {
            let _n = enter_node_scale(100.0);
            assert_eq!(current_band(), 8);
            assert_eq!(view_band(), 1);
            // Camera plan keys stay on the camera alone.
            let _c = enter_camera(Some(1.0));
            assert_eq!(current_band(), 7);
            assert_eq!(camera_key_band(), Some(0));
        }
        assert_eq!(current_band(), 1);
    }

    #[test]
    fn camera_scale_overrides_plan_scale() {
        let _p = enter_plan(4.0);
        assert_eq!(current_band(), 2);
        assert_eq!(camera_key_band(), None);
        {
            let _c = enter_camera(Some(0.5));
            assert_eq!(current_band(), -1);
            assert_eq!(camera_key_band(), Some(-1));
        }
        assert_eq!(current_band(), 2);
    }

    fn windows(v: Value, trim: Option<TrimParams>, dash: Option<&[f64]>) -> Vec<(f64, f64, bool)> {
        arc_windows(&node(v), trim, dash, 0.0, 2.0)
            .0
            .unwrap()
            .iter()
            .map(|w| (w.start, w.end, w.taper_extent.is_some()))
            .collect()
    }

    fn trim(start: f64, end: f64, offset: f64) -> Option<TrimParams> {
        Some(TrimParams {
            start,
            end,
            offset,
            individual: false,
            by: super::super::TrimBy::Length,
        })
    }

    #[test]
    fn dashed_json_outlines_the_dash_pieces() {
        let run = |req: moda_scene::value::Value| -> moda_scene::value::Value {
            moda_scene::value::from_str(&stroke_outline_trimmed_json(&req.to_string())).unwrap()
        };
        // A 100 px line, width 2, dash 10 / 10 in Moda's half-weight units (10 px / 10 px): 5 pieces, the last
        // ending at 90.
        let base = with(
            json!({"points": line(100.0)}),
            stroke_of(2.0, json!({"cap": "butt", "dash": [10, 10]})),
        );
        let out = run(base.clone());
        assert_eq!(out["status"], "ok");
        assert_eq!(out["subpathCount"], 5, "{out}");
        assert!(
            (out["bounds"]["maxX"].as_f64().unwrap() - 90.0).abs() < 1e-3,
            "{out}"
        );
        // The dash offset shifts the pieces like the canvas: half a dash at the start, 6 pieces.
        let mut shifted = base.clone();
        shifted["strokes"][0]["dashOffset"] = json!(5);
        let out = run(shifted);
        assert_eq!(out["subpathCount"], 6, "{out}");
        assert!(
            (out["points"][1]["x"].as_f64().unwrap() - 5.0).abs() < 1e-6,
            "{out}"
        );
        // Dashes and trim together: the dashes within the visible half.
        let mut both = base.clone();
        both["trimPathEnd"] = json!(0.5);
        let out = run(both);
        assert!(
            out["bounds"]["maxX"].as_f64().unwrap() <= 50.0 + 1e-3,
            "{out}"
        );
        // No dash, no trim: the continuous outline.
        let mut plain = base;
        plain["strokes"][0]["dash"] = json!([]);
        assert_eq!(run(plain)["subpathCount"], 1);
    }

    #[test]
    fn trimmed_json_outlines_only_the_visible_run() {
        let bounds = |req: moda_scene::value::Value| -> moda_scene::value::Value {
            let out: moda_scene::value::Value =
                moda_scene::value::from_str(&stroke_outline_trimmed_json(&req.to_string())).unwrap();
            out
        };
        let base = with(json!({"points": line(100.0)}), stroke_of(10.0, json!({"cap": "butt"})));
        let whole = bounds(base.clone());
        assert_eq!(whole["status"], "ok");
        assert!((whole["bounds"]["maxX"].as_f64().unwrap() - 100.0).abs() < 1e-6);
        let mut half = base.clone();
        half["trimPathEnd"] = json!(0.5);
        let half = bounds(half);
        assert_eq!(half["status"], "ok");
        assert!(
            (half["bounds"]["maxX"].as_f64().unwrap() - 50.0).abs() < 1e-3,
            "{half}"
        );
        let mut none = base.clone();
        none["trimPathStart"] = json!(0.6);
        none["trimPathEnd"] = json!(0.6);
        assert_eq!(bounds(none)["status"], "empty");
        // Identity trim = the untrimmed outline.
        let mut identity = base;
        identity["trimPathOffset"] = json!(0.3);
        assert_eq!(bounds(identity), whole);
    }

    #[test]
    fn trim_by_x_outlines_the_run_up_to_the_window_edge() {
        let spike = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 50, "y": -1000}, {"type": "line", "x": 100, "y": 0}, {"type": "line", "x": 300, "y": 0}]);
        let mut req = with(json!({"points": spike}), stroke_of(10.0, json!({"cap": "butt"})));
        req["trimPathEnd"] = json!(0.5);
        req["trimPathBy"] = json!("x");
        let out: moda_scene::value::Value = moda_scene::value::from_str(&stroke_outline_trimmed_json(&req.to_string())).unwrap();
        assert_eq!(out["status"], "ok");
        assert!((out["bounds"]["maxX"].as_f64().unwrap() - 150.0).abs() < 1e-3, "{out}");
        // The arc-length windows on the node itself: the spike's two sides, then 50 of the flat run.
        let v = with(json!({"points": req["points"].clone()}), stroke_of(2.0, json!({})));
        let side = (50.0f64).hypot(1000.0);
        let w = windows(v, Some(TrimParams { by: super::super::TrimBy::X, ..trim(0.0, 0.5, 0.0).unwrap() }), None);
        assert_eq!(w.len(), 1);
        assert!(w[0].0.abs() < 1e-9 && (w[0].1 - (2.0 * side + 50.0)).abs() < 1e-6, "{w:?}");
    }

    #[test]
    fn trim_and_dash_windows() {
        let v = with(json!({"points": line(100.0)}), stroke_of(2.0, json!({})));
        assert!(arc_windows(&node(v.clone()), None, None, 0.0, 2.0)
            .0
            .is_none());
        assert_eq!(
            windows(v.clone(), trim(0.25, 0.75, 0.0), None),
            vec![(25.0, 75.0, true)]
        );
        // Open path: an offset window past the end wraps as two runs.
        assert_eq!(
            windows(v.clone(), trim(0.0, 0.5, 0.75), None),
            vec![(75.0, 100.0, true), (0.0, 25.0, true)]
        );
        // Dash [10, 10] at width 2 = 10 on, 10 off.
        let d = [10.0, 10.0];
        let w = windows(v.clone(), None, Some(&d));
        assert_eq!(w.len(), 5);
        assert_eq!(w[0], (0.0, 10.0, false));
        assert_eq!(w[4], (80.0, 90.0, false));
        // Dash inside a trim restarts at the run and carries its extent.
        let w = windows(v, trim(0.05, 0.5, 0.0), Some(&d));
        assert_eq!(w[0], (5.0, 15.0, true));
    }

    #[test]
    fn closed_dash_stitches_through_the_seam() {
        let sq = json!({"points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
                                   {"type": "line", "x": 100, "y": 100}, {"type": "line", "x": 0, "y": 100}],
                        "closed": true, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": 2}]});
        // 30 on / 20 off over 400: last dash 350–380? no: 0-30,50-80,…,350-380 → no stitch.
        let w = windows(sq.clone(), None, Some(&[30.0, 20.0]));
        assert_eq!(w.len(), 8);
        // 30 on / 10 off over 400: dashes end exactly at 400 → the last joins the first.
        let w = windows(sq.clone(), None, Some(&[30.0, 10.0]));
        assert_eq!(w.len(), 10);
        let w = windows(sq.clone(), None, Some(&[50.0, 30.0, 20.0, 0.0]));
        assert!(w.iter().all(|(a, b, _)| b > a));
        // Never turns off → the ring.
        let w = windows(sq, None, Some(&[1000.0, 1.0]));
        assert_eq!(w, vec![(0.0, 400.0, false)]);
    }

    #[test]
    fn aligned_dash_windows_turn_the_corners_like_the_canvas() {
        // The outliner (width profiles, brushes) takes the same aligned windows as the uniform stroke: one window
        // centred on each corner, the start corner's through the seam as one window.
        let sq = json!({"points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
                                   {"type": "line", "x": 100, "y": 100}, {"type": "line", "x": 0, "y": 100}],
                        "closed": true, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": 2}]});
        let w = super::super::with_dash_alignment(true, || windows(sq.clone(), None, Some(&[20.0, 10.0])));
        assert_eq!(w.len(), 12, "{w:?}");
        let half = 1000.0 / 90.0;
        for c in [100.0, 200.0, 300.0, 400.0] {
            assert!(
                w.iter().any(|(a, b, _)| (c - a - half).abs() < 1e-6 && (b - c - half).abs() < 1e-6),
                "a window centred on {c}: {w:?}"
            );
        }
        // Exact dashes differ.
        assert_ne!(windows(sq, None, Some(&[20.0, 10.0])), w);
    }

    #[test]
    fn outlined_body_has_stroke_vertices_and_falls_back() {
        let v = with(json!({"points": line(100.0)}), stroke_of(10.0, json!({"profile": {"preset": "eye"}})));
        let n = node(v);
        let cmds = super::super::path_cmds_for_node(&n);
        let body = OutlineBody {
            cmds: &cmds,
            closed: false,
            fill_rule: FillRule::NonZero,
            stroke_width: 10.0,
            trim: None,
            dash: None,
            dash_offset: 0.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            stroke_align: StrokeAlign::Center,
            cut: None,
        };
        let (g, note) = tessellate_outlined(&n, None, &body, 0);
        assert!(note.is_none());
        assert!(g.vertex_count > 0);
        assert!(
            g.vertex_types.iter().all(|t| *t == 1.0),
            "a line has no fill area"
        );
        // Eye: widest (10) in the middle, pointed ends.
        let ys: Vec<f32> = g.positions.chunks(2).map(|p| p[1]).collect();
        let max = ys.iter().cloned().fold(f32::MIN, f32::max);
        assert!((max - 5.0).abs() < 0.1, "{max}");
    }

    #[test]
    fn dense_outline_never_refuses() {
        // A long zig-zag scribble: 3,000 anchors crossing itself, wide and
        // wobbly — far past the sweep budget as one fill.
        // (Hatching: 3,000 anchors, ~40k units of stroke over a 600-unit
        // square, every pass crossing the previous ones.)
        let mut pts = vec![json!({"type": "move", "x": 0, "y": 0})];
        for i in 1..3000 {
            let t = i as f64;
            let x = 300.0 + 280.0 * (t * 0.05).sin() * (t * 0.0021).cos();
            let y = 300.0 + 280.0 * (t * 0.043).cos() * (t * 0.0017).sin();
            pts.push(json!({"type": "line", "x": x, "y": y}));
        }
        let v = with(
            json!({"points": pts}),
            stroke_of(12.0, json!({"profile": {"preset": "taper"}, "join": "round",
                "dynamic": {"frequency": 4, "wiggle": 3, "smoothen": 0.2}})),
        );
        let n = node(v);
        let cmds = super::super::path_cmds_for_node(&n);
        let body = OutlineBody {
            cmds: &cmds,
            closed: false,
            fill_rule: FillRule::NonZero,
            stroke_width: 12.0,
            trim: None,
            dash: None,
            dash_offset: 0.0,
            line_cap: LineCap::Round,
            line_join: LineJoin::Round,
            stroke_align: StrokeAlign::Center,
            cut: None,
        };
        let t0 = std::time::Instant::now();
        let (g, note) = tessellate_outlined(&n, None, &body, 0);
        eprintln!(
            "dense outline: {} vertices in {:?}, note {note:?}",
            g.vertex_count,
            t0.elapsed()
        );
        assert!(g.vertex_count > 0, "never an empty stroke: {note:?}");
        // One fill is over the sweep budget; the cell fill draws it, and says so.
        let whole = outline_contours(&n, None, None, tolerance_for_band(0)).unwrap();
        let one = fill_outline_contours(&whole);
        assert!(one.is_err(), "the test must exercise the cell path");
        let note = note.expect("a note when the fill is split");
        assert!(
            note.contains("cells") && !note.contains("uniform"),
            "{note}"
        );
    }
}

#[cfg(test)]
mod bench {
    use super::super::tessellator::fill_outline_contours;
    use super::*;
    use moda_scene::value::json;

    /// `cargo test --release -p moda-engine --lib tess::outline::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn outline_cost_breakdown() {
        let mut s: u32 = 42;
        let mut r = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s as f64 / 4294967296.0
        };
        let mut nodes = Vec::new();
        for i in 0..400 {
            let mut pts = vec![json!({"type": "move", "x": 5, "y": 40.0 + r() * 20.0})];
            for k in 1..=5 {
                let px = 5.0 + k as f64 * 15.0;
                let py = 10.0 + r() * 60.0;
                pts.push(json!({"type": "curve", "x": px, "y": py,
                    "cp1": {"x": px - 10.0, "y": py - 20.0 + r() * 40.0},
                    "cp2": {"x": px - 5.0, "y": py - 20.0 + r() * 40.0}}));
            }
            let mut v = json!({"type": "path", "id": format!("s{i}"), "points": pts,
                               "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "cap": "round", "join": "round"}]});
            match i % 4 {
                0 => v["strokes"][0]["profile"] = json!({"preset": "eye"}),
                1 => {
                    v["strokes"][0]["taper"] = json!({"units": "percent", "startLength": 20, "endLength": 40,
                        "startWidth": 0, "endWidth": 0, "startEase": 0, "endEase": 0.5});
                    v["strokes"][0]["wave"] =
                        json!({"amount": 0.3, "units": "px", "wavelength": 18, "phase": 0});
                }
                2 => v["strokes"][0]["brush"] = json!({"brushId": "nib"}),
                _ => {
                    v["strokes"][0]["dynamic"] = json!({"frequency": 4, "wiggle": 2, "smoothen": 0.4});
                    v["strokes"][0]["profile"] = json!({"preset": "eye"});
                }
            }
            nodes.push(RawNode::from_value(v).unwrap());
        }
        let nib = json!({"id": "nib", "type": "calligraphic", "angle": {"value": 40},
                         "roundness": {"value": 0.2}, "size": {"value": 1}});
        for band in [-2, 1, 3, 99] {
            // band 99: tolerance 0 = polylines, no curve fitting.
            let tol = if band == 99 {
                0.0
            } else {
                tolerance_for_band(band)
            };
            let (mut t_solve, mut t_flat, mut t_fill, mut verts) = (0.0, 0.0, 0.0, 0usize);
            for n in &nodes {
                let brush = (brush_id(n).is_some()).then_some(&nib);
                let t0 = std::time::Instant::now();
                let o = stroke_outline(
                    n,
                    brush,
                    &OutlineOptions {
                        tolerance: tol,
                        ..OutlineOptions::default()
                    },
                )
                .unwrap();
                let t1 = std::time::Instant::now();
                let Outline::Shape { points, .. } = o else {
                    continue;
                };
                let c = flatten(&points, tol.max(0.05));
                let t2 = std::time::Instant::now();
                let (tris, _) = fill_outline_contours(&c).unwrap();
                let t3 = std::time::Instant::now();
                verts += tris.len() / 2;
                t_solve += (t1 - t0).as_secs_f64();
                t_flat += (t2 - t1).as_secs_f64();
                t_fill += (t3 - t2).as_secs_f64();
            }
            let n = nodes.len() as f64;
            eprintln!(
                "band {band}: per stroke solve {:.3} ms, flatten {:.3} ms, fill {:.3} ms, {:.0} vertices",
                1e3 * t_solve / n,
                1e3 * t_flat / n,
                1e3 * t_fill / n,
                verts as f64 / n
            );
        }
    }
}
