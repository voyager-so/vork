//! M2 — the faithful port of
//! `renderers/webgpu/geometry/WebGPUPathTessellator.ts` (line citations are to
//! that file) for flattening, strokes and dashes, gated by the vertex oracle
//! (`fixtures/oracle/*/tessellation.json`, `tools/oracle/diff_vertices.py`,
//! STROKE vertices only).
//!
//! FILLS ARE NOT A PORT. The TS ear clipper (and its containment-tree hole
//! bridging) assumed a simple polygon, so a self-intersecting outline lost
//! points under nonzero and filled solid under evenodd. `tessellate_path`
//! feeds the same flattened contours to lyon's sweep-line fill tessellator,
//! which applies the fill rule to the actual winding. Fill vertices therefore
//! differ from the TS stream and are excluded from the oracle; the fill's
//! OUTLINE vertices are still the shared flattening, so a filled shape's edge
//! lands on the same pixels either way.
//!
//! COPY THE BUGS (each pinned by tests or the oracle):
//! - Combined bounds always spread the fill bounds even when the fill is
//!   empty (`{minX:0,…}`), so stroke-only paths get bounds that include the
//!   origin (:177-183). Quirk Q-49.
//! - When a subpath exceeds MAX_PATH_SEGMENTS, the curve/quadratic/arc cases
//!   `break` out of the *switch*, skipping the command WITHOUT updating the
//!   current point (:954-957, :983-986, :1019-1023). Quirk Q-50.
//! - Dashed strokes are dashed per sub-path with one running phase (a dash
//!   runs through a vertex and takes the join there), a closed sub-path's
//!   wrapping dash is stitched through the seam, and a dash that never turns
//!   off strokes as the closed contour. Stroke alignment still only reaches
//!   that last case: an open dash piece has no interior, so it stays centered.
//! - Open subpaths are filled as if closed (:478-497). Quirk Q-51.

use super::arc::{arc_to_beziers, ArcParams};
use super::dash::{
    aligned_dash_intervals, cubic_arc_length, dash_pieces, extract_path_segments_by_arc_length,
    intervals_as_walk_pattern, normalize_dash_pattern, quadratic_arc_length, scale_dash_pattern,
    segment_corner_positions, walk_path_by_arc_length, walk_path_by_arc_length_phased,
    PathSegment, Pt, Seg, BASE_DASH_STROKE_WIDTH,
};
use super::moda_engine_path_types::*;
use super::trim_axis::{self, TrimBy};
use crate::seams::TessellatedGeometry;

// MIRRORED: `_FLATNESS_TOLERANCE_PT` in backend/app/services/pdf_import/extract.py
// converts this into page points to predict the edge count this produces. Move that
// one with this one; a finer threshold here means more edges than the importer
// expects, and it stops refusing nodes this will refuse. The importer's estimate
// covers FILLS (always this threshold) and 1× strokes. Two stroke-only departures
// it does not model: wide curved strokes also split at the turn limit
// (`Flat::stroke`, a few more edges at 1×), and above 1× strokes flatten finer
// per zoom band. Neither can refuse a node the importer keeps: when a stroke's
// flattening would hit a cap, `stroke_within_caps` gives bands back and, as a
// last resort, flattens exactly as before both (this threshold, no turn limit).
// Below 1× everything flattens coarser (fewer edges than the estimate).
const FLATNESS_THRESHOLD: f64 = 0.5; // :55 — canvas units at 1×

/// How finely curves are flattened. `dist`: the control points' largest
/// distance from the chord. `turn`: for strokes, the largest turn of a
/// flattened piece (its control polygon) — a wide stroke magnifies a chord's
/// angle into a visible facet on its outer edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Flat {
    pub dist: f64,
    pub turn: Option<f64>,
}

impl Flat {
    /// The pre-band flattening: 0.5 units at every scale, no turn limit —
    /// what main drew; the last resort of `stroke_within_caps` and
    /// `fill_within_caps`.
    pub(crate) fn legacy() -> Flat {
        Flat {
            dist: FLATNESS_THRESHOLD,
            turn: None,
        }
    }

    /// The fill flattening: 0.5 units at 1× (the studio-golden flattening),
    /// halved per zoom band above and doubled per band below, like strokes —
    /// so curved fills (placed art-brush artwork included) keep their curves
    /// at a 2× export or when zoomed in, and cost less zoomed out. No turn
    /// limit (a fill has no offset).
    pub(crate) fn fill() -> Flat {
        if LEGACY.with(std::cell::Cell::get) {
            return Flat::legacy();
        }
        Flat {
            dist: flatness_tolerance(),
            turn: None,
        }
    }

    /// The stroke flattening for a stroke of half-width `half`: the outer
    /// offset of a chord turning θ sags `half·(1 − cos θ/2)`, kept within the
    /// same tolerance. Under `stroke_within_caps`' last resort it is the
    /// pre-band flattening (0.5 units, no turn limit), which draws whatever
    /// main drew.
    pub(crate) fn stroke(half: f64) -> Flat {
        if LEGACY.with(std::cell::Cell::get) {
            return Flat::legacy();
        }
        let dist = flatness_tolerance();
        let turn = (half > dist).then(|| 2.0 * (1.0 - dist / half).clamp(-1.0, 1.0).acos());
        Flat { dist, turn }
    }
}

/// Chord tolerance (canvas units) for the current render scale (F4.1): half a
/// device pixel at every zoom band, i.e. `0.5 / 2^band` units. Band 0 (1×) is
/// the studio-golden 0.5 units, unchanged; a 2× export or a 4× canvas zoom
/// flattens to a quarter / eighth of a unit, and a 1/4× zoom-out to 2 units,
/// so curves are not over-flattened when they are small on screen. The scale
/// is `zoom × dpr` for the canvas and the export pixel ratio for exports, in
/// the zoom bands the outlined strokes use (`outline::current_band`, clamped
/// to `MIN_BAND..=MAX_BAND`: 8 units at 1/16× and below, 1/1024 unit at 512×
/// and above). The camera plan keys and the per-node tess caches fold the band
/// in.
///
/// Only bands above 1× give bands back when a cap is hit (`COARSEN`): below
/// 1× the flattening is already coarser than the 1× one.
pub(crate) fn flatness_tolerance() -> f64 {
    FLATNESS_THRESHOLD / 2f64.powi(flatten_band())
}

/// The zoom band curves flatten at: the render band below 1×, and above it
/// the render band less the bands `stroke_within_caps` / `fill_path` gave
/// back (never below 1×).
fn flatten_band() -> i32 {
    let band = super::outline::current_band();
    if band < 0 {
        band
    } else {
        (band - COARSEN.with(std::cell::Cell::get)).max(0)
    }
}

thread_local! {
    /// Zoom bands given back for the stroke being tessellated: a stroke whose
    /// band flattening would hit a cap (`MAX_PATH_SEGMENTS`,
    /// `MAX_STROKE_VERTICES`) is redone coarser, band by band, down to the 1×
    /// flattening — never truncated or dropped because of the zoom.
    static COARSEN: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
    /// Set when flattening or stroking hit a cap during the current attempt.
    static CAP_HIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The last resort: strokes flatten as before zoom bands and the turn
    /// limit (0.5 units, no turn limit).
    static LEGACY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Attempts the last `stroke_within_caps` made (tests / diagnostics).
    static ATTEMPTS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

fn note_cap_hit() {
    CAP_HIT.with(|c| c.set(true));
}

/// Run `stroke` at the finest flattening that fits the caps, never worse
/// than before zoom bands:
/// - The band's flattening (with the turn limit) is tried first, unless an
///   up-front estimate of its flattened size (`estimate_flattened`) already
///   exceeds `MAX_PATH_SEGMENTS` / `MAX_STROKE_VERTICES`; then it starts at
///   the first coarser band the estimate fits, so a huge stroke at 64× costs
///   one or two attempts, not seven.
/// - An attempt that still hits a cap is redone one band coarser, down to 1×.
/// - The last resort is the pre-band flattening (0.5 units, no turn limit),
///   whose result is accepted only if its outline is complete.
///
/// Straight-only paths run once (flattening does not apply).
fn stroke_within_caps(
    points: &[PathCmd],
    reach: f64,
    on_fraction: f64,
    stroke: impl Fn() -> (Vec<f64>, [f64; 4]),
) -> Result<(Vec<f64>, [f64; 4]), String> {
    ATTEMPTS.with(|a| a.set(0));
    let attempt = |c: i32, legacy: bool| {
        COARSEN.with(|k| k.set(c));
        LEGACY.with(|k| k.set(legacy));
        CAP_HIT.with(|k| k.set(false));
        ATTEMPTS.with(|a| a.set(a.get() + 1));
        let out = stroke();
        let fits = !CAP_HIT.with(std::cell::Cell::get) && out.0.len() / 2 <= MAX_STROKE_VERTICES;
        (out, fits)
    };
    let prev = (
        COARSEN.with(std::cell::Cell::get),
        LEGACY.with(std::cell::Cell::get),
    );
    let restore = || {
        COARSEN.with(|k| k.set(prev.0));
        LEGACY.with(|k| k.set(prev.1));
    };
    if !has_curve_cmds(points) {
        let (out, _) = attempt(0, false);
        let complete = !CAP_HIT.with(std::cell::Cell::get);
        restore();
        return if complete { Ok(out) } else { Err(INCOMPLETE_PATH.into()) };
    }
    let band = super::outline::current_band().max(0);
    // Up-front estimate: the first level whose flattening fits.
    let fits_estimate = |c: i32| {
        COARSEN.with(|k| k.set(c));
        LEGACY.with(|k| k.set(false));
        let (worst_subpath, total) = estimate_flattened(points, Flat::stroke(reach));
        worst_subpath <= MAX_PATH_SEGMENTS
            && (total as f64) * EST_VERTS_PER_SEGMENT * on_fraction <= MAX_STROKE_VERTICES as f64
    };
    let start = (0..=band).find(|&c| fits_estimate(c));
    if let Some(start) = start {
        for c in start..=band {
            let (out, fits) = attempt(c, false);
            if fits {
                restore();
                return Ok(out);
            }
        }
    }
    let (out, _) = attempt(band, true);
    let complete = !CAP_HIT.with(std::cell::Cell::get);
    restore();
    if complete { Ok(out) } else { Err(INCOMPLETE_PATH.into()) }
}

/// Stroke vertices per flattened segment, for the up-front estimate: two
/// triangles per piece plus a miter / bevel join (round joins and caps add
/// more, which the retry covers).
const EST_VERTS_PER_SEGMENT: f64 = 9.0;

/// Estimated flattened size of `points` at `flat`: (the largest subpath's
/// segment count, the total). Mirrors `tessellate_cubic_bezier`'s midpoint
/// recursion: control-point deviation shrinks ~4× per level, the control
/// polygon's turn ~2× per level.
fn estimate_flattened(points: &[PathCmd], flat: Flat) -> (usize, usize) {
    let cubic = |p0: P, p1: P, p2: P, p3: P| -> usize {
        let dev = cubic_deviation(p0, p1, p2, p3);
        let mut depth = 0u32;
        if dev > flat.dist {
            depth = (dev / flat.dist).log(4.0).ceil().max(0.0) as u32;
        }
        if let Some(t) = flat.turn {
            let turn = control_turn(p0, p1, p2, p3);
            if turn > t {
                depth = depth.max((turn / t).log2().ceil().max(0.0) as u32);
            }
        }
        1usize << depth.min(MAX_RECURSION_DEPTH)
    };
    let (mut worst, mut total, mut here) = (0usize, 0usize, 0usize);
    let mut cur: P = [0.0, 0.0];
    for p in points {
        let n = match p {
            PathCmd::Move { x, y } => {
                worst = worst.max(here);
                here = 0;
                cur = [*x, *y];
                continue;
            }
            PathCmd::Line { .. } => 1,
            PathCmd::Curve { x, y, cp1, cp2 } => {
                cubic(cur, [cp1.0, cp1.1], [cp2.0, cp2.1], [*x, *y])
            }
            PathCmd::Quadratic { x, y, cp } => {
                let c1 = [
                    cur[0] + (2.0 / 3.0) * (cp.0 - cur[0]),
                    cur[1] + (2.0 / 3.0) * (cp.1 - cur[1]),
                ];
                let c2 = [
                    *x + (2.0 / 3.0) * (cp.0 - *x),
                    *y + (2.0 / 3.0) * (cp.1 - *y),
                ];
                cubic(cur, c1, c2, [*x, *y])
            }
            PathCmd::Arc {
                x,
                y,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            } => {
                let arc = ArcParams {
                    rx: *rx,
                    ry: *ry,
                    x_axis_rotation: *x_axis_rotation,
                    large_arc: *large_arc,
                    sweep: *sweep,
                    x: *x,
                    y: *y,
                };
                let beziers = arc_to_beziers(cur[0], cur[1], &arc);
                let mut b0 = cur;
                let mut n = beziers.is_empty() as usize;
                for b in &beziers {
                    n += cubic(
                        b0,
                        [b.cp1[0], b.cp1[1]],
                        [b.cp2[0], b.cp2[1]],
                        [b.end[0], b.end[1]],
                    );
                    b0 = [b.end[0], b.end[1]];
                }
                n
            }
        };
        here += n;
        total += n;
        if let PathCmd::Line { x, y }
        | PathCmd::Curve { x, y, .. }
        | PathCmd::Quadratic { x, y, .. }
        | PathCmd::Arc { x, y, .. } = p
        {
            cur = [*x, *y];
        }
    }
    (worst.max(here), total)
}

/// The control points' largest distance from the chord (`is_curve_flat`'s
/// measure).
fn cubic_deviation(p0: P, p1: P, p2: P, p3: P) -> f64 {
    let (dx, dy) = (p3[0] - p0[0], p3[1] - p0[1]);
    let d2 = dx * dx + dy * dy;
    let dist = |p: P| {
        if d2 == 0.0 {
            return (p[0] - p0[0]).hypot(p[1] - p0[1]);
        }
        let t = (((p[0] - p0[0]) * dx + (p[1] - p0[1]) * dy) / d2).clamp(0.0, 1.0);
        (p[0] - (p0[0] + t * dx)).hypot(p[1] - (p0[1] + t * dy))
    };
    dist(p1).max(dist(p2))
}

fn has_curve_cmds(points: &[PathCmd]) -> bool {
    points
        .iter()
        .any(|p| !matches!(p, PathCmd::Move { .. } | PathCmd::Line { .. }))
}
const POLY_EPS: f64 = 1e-4; // :56
/// Cap on the vertices a fill may emit — both the UNIQUE vertices lyon
/// creates and the EXPANDED stream the renderer consumes (three references
/// per triangle). Enforced inside lyon's sweep by `BoundedFillBuilder`, so a
/// pathological outline aborts the moment it crosses either bound instead of
/// after an unbounded sweep, and every overflow is a `fill_dropped` refusal
/// that keeps the stroke. Public so the guardrail suite can assert against it.
///
/// It bounds memory (4 MB of f32 positions), not work: the sweep's work is
/// bounded by `MAX_FILL_SWEEP_COST` and the cells. It was 50,000, the TS
/// renderer's cap, and ordinary compound art ran past that: 16 fern fronds as
/// one path (~2,500 points, overlapping leaflets) need 52,000 and drew nothing
/// (#2151); 26 fronds need 115,000 and fill in ~30 ms.
pub const MAX_FILL_VERTICES: usize = 500_000; // :59

// Conservative pre-sweep envelope for ONE lyon sweep. Each overlapping pair may
// intersect and each resulting event may scan the active set. Charge
// active-width squared when an edge enters; a sweep over it never starts. It
// bounds a sweep, not a fill: a fill over it is split into cells that each fit
// (`fill_cells`), which is what lets a line of text or a 600-path union — many
// edges sharing one vertical range — fill at all. What still refuses a fill is
// the vertex cap (dense self-crossings make a vertex per crossing) or a cell
// that is over this bound at the deepest split.
//
// `_MAX_FILL_SWEEP_COST` in backend/app/services/pdf_import/extract.py used to
// mirror this as a WHOLE-fill limit, so the PDF importer would not build a node
// the engine refused. It is now stricter than the engine (it refuses nodes that
// fill in cells) and can be retired; a refused fill also no longer fails the
// page (the CLI draws the rest and reports it).
const MAX_FILL_SWEEP_COST: usize = 2_000_000;
// The flattening target for a stroke's curves: `stroke_within_caps` coarsens a curved stroke band by band until it
// fits. Not a drop cap (that is `MAX_STROKE_STREAM_VERTICES`).
const MAX_STROKE_VERTICES: usize = 100_000; // :60
/// The most vertices one path's stroke stream may hold. The TS stroker's cap was `MAX_STROKE_VERTICES` and it
/// emptied the stroke past it with no reason: a contour field of a few thousand polylines in one `<path d>` (round
/// joins and caps cost ~30 vertices per vertex, bevels ~12) drew nothing at all. The GPU upload splits a stream
/// over the device's buffer limit into chunks (`tests/vertex_chunks.rs`), so this only bounds memory: 4M vertices
/// is 32 MB of positions. Past it the stroke is refused with a `stroke_dropped` reason, never silently.
pub const MAX_STROKE_STREAM_VERTICES: usize = 4_000_000;
const MAX_TOTAL_VERTICES: usize = MAX_FILL_VERTICES + MAX_STROKE_STREAM_VERTICES; // :61
                                           // MIRRORED as `MAX_RESULT_POINTS` in moda-scene's `path_bool/solve.rs`, where
                                           // the boolean solver refuses to EMIT a path this one would silently truncate
                                           // (see the slice below). `cap_parity_tests` at the bottom of this file fails if
                                           // they drift.
const MAX_PATH_SEGMENTS: usize = 10_000; // :62
                                         // MIRRORED as `_MAX_FLATTEN_DEPTH` in backend/app/services/pdf_import/extract.py.
const MAX_RECURSION_DEPTH: u32 = 12; // :1093
const MITER_LIMIT: f64 = 4.0; // :1830 — the default (`miterLimit` absent)

thread_local! {
    /// The node's `miterLimit` while its geometry is built ([`with_miter_limit`]).
    static MITER_LIMIT_NOW: std::cell::Cell<f64> = const { std::cell::Cell::new(MITER_LIMIT) };
}

/// Run `f` with the stroker's miter limit set to `limit` (a node's
/// `miterLimit`, already clamped by `RawNode::miter_limit`), restoring the
/// default after.
pub fn with_miter_limit<T>(limit: f64, f: impl FnOnce() -> T) -> T {
    /// Restores the previous limit on drop, so a panic inside `f` (caught
    /// further up) cannot leak this node's limit into the next tessellation.
    struct Restore(f64);
    impl Drop for Restore {
        fn drop(&mut self) {
            MITER_LIMIT_NOW.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(MITER_LIMIT_NOW.with(|c| c.replace(limit)));
    f()
}

thread_local! {
    /// The stroke's dash alignment while its geometry is built ([`with_dash_alignment`]).
    static DASH_ALIGN_NOW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with dashes aligned to corners and path ends (`true`: a stroke's
/// `dashAlignment: "corners"`, Illustrator's "Align dashes to corners and path
/// ends, adjusting lengths to fit") or laid out exactly (`false`, the
/// default), restoring the previous setting after. Like [`with_miter_limit`],
/// a per-node setting read by every dash walk under `f`: the uniform stroke's
/// ([`apply_dash_to_path`]) and the outliner's dash windows
/// (`outline::arc_windows_on`). The renderer sets it per body item
/// (`frame.rs` `build_body_item`), the outliner's own entry points
/// (`tessellate_outlined_node`, the `stroke_outline` export) from the node.
/// Lines keep exact dashes.
pub fn with_dash_alignment<T>(aligned: bool, f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DASH_ALIGN_NOW.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(DASH_ALIGN_NOW.with(|c| c.replace(aligned)));
    f()
}

/// Whether dashes are aligned to corners right now ([`with_dash_alignment`]).
pub fn dash_alignment_now() -> bool {
    DASH_ALIGN_NOW.with(std::cell::Cell::get)
}

/// Corner positions per sub-path of `points` (see
/// [`segment_corner_positions`]), with each sub-path's measured length, in
/// sub-path order — what the outliner needs to align its dash windows.
pub fn subpath_corners(points: &[PathCmd], closed: bool) -> Vec<(f64, Vec<f64>)> {
    split_subpaths(points)
        .into_iter()
        .map(|sp| {
            let segs = path_points_to_segments(sp, closed);
            let total: f64 = segs.iter().map(|s| s.arc_length).sum();
            (total, segment_corner_positions(&segs, closed))
        })
        .collect()
}

const EPS: f64 = 1e-6;

type P = Pt;

/// Oracle-precision result: f32 vertex stream (the GPU/oracle contract is a
/// Float32Array on both sides) + f64 bounds (the TS bounds object stays f64).
pub struct OracleGeometry {
    pub positions: Vec<f32>,
    pub vertex_types: Vec<f32>,
    /// [minX, minY, maxX, maxY] in f64 — zeros when empty (copy the bug).
    pub bounds: [f64; 4],
    /// Why the fill was dropped (fail-closed), if it was. See
    /// [`TessellatedGeometry::fill_dropped`].
    pub fill_dropped: Option<String>,
    /// Why the stroke was dropped, if it was. See [`TessellatedGeometry::stroke_dropped`].
    pub stroke_dropped: Option<String>,
}

/// Frozen M1 seam signature (SEAMS.md §1) — no dash. Delegates.
pub fn tessellate_combined(
    points: &[PathCmd],
    stroke_width: f64,
    closed: bool,
    line_cap: LineCap,
    line_join: LineJoin,
    fill_rule: FillRule,
    stroke_align: StrokeAlign,
) -> TessellatedGeometry {
    tessellate_combined_with_dash(
        points,
        stroke_width,
        closed,
        None,
        line_cap,
        line_join,
        fill_rule,
        stroke_align,
    )
}

/// `tessellateCombined` (:83-191) with the full parameter surface.
#[allow(clippy::too_many_arguments)]
pub fn tessellate_combined_with_dash(
    points: &[PathCmd],
    stroke_width: f64,
    closed: bool,
    dash: Option<&[f64]>,
    line_cap: LineCap,
    line_join: LineJoin,
    fill_rule: FillRule,
    stroke_align: StrokeAlign,
) -> TessellatedGeometry {
    tessellate_combined_with_dash_trim(
        points,
        stroke_width,
        closed,
        None,
        dash,
        0.0,
        line_cap,
        line_join,
        fill_rule,
        stroke_align,
    )
}

/// `tessellateCombined` with an additional stroke-only trim-path pass applied
/// before dashing. When `trim` is `None` (or resolves to identity) this is
/// byte-identical to [`tessellate_combined_with_dash`]. See
/// `docs/roadmap/canvas/stroke-path-animation.md`.
#[allow(clippy::too_many_arguments)]
pub fn tessellate_combined_cached(
    points: &[PathCmd],
    stroke_width: f64,
    closed: bool,
    trim: Option<TrimParams>,
    dash: Option<&[f64]>,
    dash_offset: f64,
    line_cap: LineCap,
    line_join: LineJoin,
    fill_rule: FillRule,
    stroke_align: StrokeAlign,
) -> std::sync::Arc<TessellatedGeometry> {
    use std::collections::{HashMap, VecDeque};
    use std::sync::Arc;

    struct Entry {
        geometry: Arc<TessellatedGeometry>,
        cap_hit: bool,
        attempts: u32,
        vertex_peak: usize,
        cell_count: usize,
    }

    #[derive(Default)]
    struct Cache {
        entries: HashMap<Vec<u64>, Entry, rustc_hash::FxBuildHasher>,
        order: VecDeque<(Vec<u64>, usize)>,
        bytes: usize,
    }

    thread_local! {
        static CACHE: std::cell::RefCell<Cache> = std::cell::RefCell::new(Cache::default());
    }

    if CAP_HIT.with(std::cell::Cell::get) {
        return Arc::new(tessellate_combined_with_dash_trim(points, stroke_width, closed,
            trim, dash, dash_offset, line_cap, line_join, fill_rule, stroke_align));
    }

    let mut key = vec![
        stroke_width.to_bits(), closed as u64, dash_offset.to_bits(), line_cap as u64,
        line_join as u64, fill_rule as u64, stroke_align as u64,
        super::outline::current_band() as u64,
        MITER_LIMIT_NOW.with(std::cell::Cell::get).to_bits(),
        DASH_ALIGN_NOW.with(std::cell::Cell::get) as u64,
        SWEEP_BUDGET.with(std::cell::Cell::get) as u64,
        FILL_VERTEX_CAP.with(std::cell::Cell::get) as u64,
        COARSEN.with(std::cell::Cell::get) as u64,
        LEGACY.with(std::cell::Cell::get) as u64,
        trim.is_some() as u64,
    ];
    if let Some(trim) = trim {
        key.extend([trim.start.to_bits(), trim.end.to_bits(), trim.offset.to_bits(),
            trim.individual as u64, trim.by as u64]);
    }
    key.push(dash.is_some() as u64);
    if let Some(dash) = dash {
        key.push(dash.len() as u64);
        key.extend(dash.iter().map(|value| value.to_bits()));
    }
    key.push(points.len() as u64);
    for point in points {
        match point {
            PathCmd::Move { x, y } => key.extend([0, x.to_bits(), y.to_bits()]),
            PathCmd::Line { x, y } => key.extend([1, x.to_bits(), y.to_bits()]),
            PathCmd::Curve { x, y, cp1, cp2 } => key.extend([2, x.to_bits(), y.to_bits(),
                cp1.0.to_bits(), cp1.1.to_bits(), cp2.0.to_bits(), cp2.1.to_bits()]),
            PathCmd::Quadratic { x, y, cp } => key.extend([3, x.to_bits(), y.to_bits(),
                cp.0.to_bits(), cp.1.to_bits()]),
            PathCmd::Arc { x, y, rx, ry, x_axis_rotation, large_arc, sweep } => key.extend([
                4, x.to_bits(), y.to_bits(), rx.to_bits(), ry.to_bits(), x_axis_rotation.to_bits(),
                *large_arc as u64, *sweep as u64]),
        }
    }
    if let Some(geometry) = CACHE.with(|cache| {
        let cache = cache.borrow();
        cache.entries.get(&key).map(|entry| {
            CAP_HIT.with(|value| value.set(entry.cap_hit));
            ATTEMPTS.with(|value| value.set(entry.attempts));
            FILL_VERTEX_PEAK.with(|value| value.set(entry.vertex_peak));
            FILL_CELL_COUNT.with(|value| value.set(entry.cell_count));
            entry.geometry.clone()
        })
    }) {
        return geometry;
    }
    let geometry = Arc::new(tessellate_combined_with_dash_trim(points, stroke_width, closed,
        trim, dash, dash_offset, line_cap, line_join, fill_rule, stroke_align));
    let bytes = 2 * key.capacity() * std::mem::size_of::<u64>()
        + geometry.positions.capacity() * std::mem::size_of::<f32>()
        + geometry.vertex_types.capacity() * std::mem::size_of::<f32>()
        + geometry.stroke_uv.capacity() * std::mem::size_of::<f32>()
        + geometry.fill_dropped.as_ref().map_or(0, String::capacity)
        + geometry.stroke_dropped.as_ref().map_or(0, String::capacity)
        + 256;
    const BUDGET: usize = 64 << 20;
    if bytes <= BUDGET {
        CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            while cache.bytes + bytes > BUDGET {
                let (old, bytes) = cache.order.pop_front().unwrap();
                cache.entries.remove(&old);
                cache.bytes -= bytes;
            }
            cache.bytes += bytes;
            cache.order.push_back((key.clone(), bytes));
            cache.entries.insert(key, Entry {
                geometry: geometry.clone(),
                cap_hit: CAP_HIT.with(std::cell::Cell::get),
                attempts: ATTEMPTS.with(std::cell::Cell::get),
                vertex_peak: FILL_VERTEX_PEAK.with(std::cell::Cell::get),
                cell_count: FILL_CELL_COUNT.with(std::cell::Cell::get),
            });
        });
    }
    geometry
}

#[allow(clippy::too_many_arguments)]
pub fn tessellate_combined_with_dash_trim(
    points: &[PathCmd],
    stroke_width: f64,
    closed: bool,
    trim: Option<TrimParams>,
    dash: Option<&[f64]>,
    dash_offset: f64,
    line_cap: LineCap,
    line_join: LineJoin,
    fill_rule: FillRule,
    stroke_align: StrokeAlign,
) -> TessellatedGeometry {
    let o = tessellate_combined_oracle(
        points,
        stroke_width,
        closed,
        trim,
        dash,
        dash_offset,
        line_cap,
        line_join,
        fill_rule,
        stroke_align,
    );
    let vertex_count = (o.positions.len() / 2) as u32;
    TessellatedGeometry {
        positions: o.positions,
        vertex_types: o.vertex_types,
        vertex_count,
        bounds: [
            o.bounds[0] as f32,
            o.bounds[1] as f32,
            (o.bounds[2] - o.bounds[0]) as f32,
            (o.bounds[3] - o.bounds[1]) as f32,
        ],
        fill_dropped: o.fill_dropped,
        stroke_dropped: o.stroke_dropped,
        stroke_uv: Vec::new(),
    }
}

#[cfg(test)]
mod exact_cache_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn complete_inputs_reuse_only_identical_geometry() {
        let points = vec![PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Curve { x: 20.0, y: 5.0, cp1: (3.0, 17.0), cp2: (12.0, -4.0) },
            PathCmd::Line { x: 1.0, y: 24.0 }];
        let trim = TrimParams { start: 0.1, end: 0.9, offset: 0.0, individual: false, by: TrimBy::Length };
        let mut previous = Vec::new();
        for variant in 0..16 {
            let mut points = points.clone();
            if variant == 1 {
                points[0] = PathCmd::Move { x: -0.0, y: 0.0 };
            }
            let width = if variant == 2 { 3.0 } else { 2.0 };
            let closed = variant != 3;
            let trim = match variant {
                4 => Some(trim),
                5 => Some(TrimParams { by: TrimBy::X, ..trim }),
                6 => Some(TrimParams { individual: true, ..trim }),
                _ => None,
            };
            let dash = (variant == 7 || variant == 8).then_some([2.0, 1.0]);
            let offset = if variant == 8 { 0.5 } else { 0.0 };
            let cap = if variant == 9 { LineCap::Round } else { LineCap::Butt };
            let join = if variant == 10 { LineJoin::Bevel } else { LineJoin::Miter };
            let rule = if variant == 11 { FillRule::EvenOdd } else { FillRule::NonZero };
            let align = if variant == 12 { StrokeAlign::Outside } else { StrokeAlign::Center };
            let _scale = super::super::outline::enter_plan(if variant == 13 { 4.0 } else { 1.0 });
            with_miter_limit(if variant == 14 { 1.0 } else { 4.0 }, || {
                with_dash_alignment(variant == 15, || {
                    let cached = tessellate_combined_cached(&points, width, closed, trim,
                        dash.as_ref().map(|dash| dash.as_slice()), offset, cap, join, rule, align);
                    let hit = tessellate_combined_cached(&points, width, closed, trim,
                        dash.as_ref().map(|dash| dash.as_slice()), offset, cap, join, rule, align);
                    assert!(Arc::ptr_eq(&cached, &hit));
                    assert!(previous.iter().all(|old| !Arc::ptr_eq(old, &cached)));
                    previous.push(cached.clone());
                    let fresh = tessellate_combined_with_dash_trim(&points, width, closed, trim,
                        dash.as_ref().map(|dash| dash.as_slice()), offset, cap, join, rule, align);
                    assert_eq!(*cached, fresh, "variant {variant}");
                });
            });
        }
    }

    #[test]
    fn fill_refusal_and_work_budgets_are_keyed() {
        let points = [PathCmd::Move { x: 0.0, y: 0.0 }, PathCmd::Line { x: 10.0, y: 0.0 },
            PathCmd::Line { x: 10.0, y: 10.0 }, PathCmd::Line { x: 0.0, y: 10.0 }];
        for cap in [1, MAX_FILL_VERTICES] {
            for budget in [1, MAX_FILL_SWEEP_COST] {
                with_fill_vertex_cap(cap, || with_fill_sweep_budget(budget, || {
                    let cached = tessellate_combined_cached(&points, 0.0, true, None, None,
                        0.0, LineCap::Butt, LineJoin::Miter, FillRule::NonZero, StrokeAlign::Center);
                    let fresh = tessellate_combined_with_dash_trim(&points, 0.0, true, None, None,
                        0.0, LineCap::Butt, LineJoin::Miter, FillRule::NonZero, StrokeAlign::Center);
                    assert_eq!(*cached, fresh);
                }));
            }
        }
    }

    #[test]
    fn hits_restore_tessellator_cap_and_diagnostic_state() {
        let points = [PathCmd::Move { x: 0.0, y: 0.0 }, PathCmd::Line { x: 11.0, y: 0.0 },
            PathCmd::Line { x: 11.0, y: 11.0 }];
        let draw = || tessellate_combined_cached(&points, 1.0, true, None, None,
            0.0, LineCap::Butt, LineJoin::Miter, FillRule::NonZero, StrokeAlign::Center);
        CAP_HIT.with(|value| value.set(false));
        let first = draw();
        let expected = (CAP_HIT.with(std::cell::Cell::get), ATTEMPTS.with(std::cell::Cell::get),
            last_fill_vertex_peak(), last_fill_cell_count());
        ATTEMPTS.with(|value| value.set(999));
        FILL_VERTEX_PEAK.with(|value| value.set(999));
        FILL_CELL_COUNT.with(|value| value.set(999));
        let hit = draw();
        assert!(Arc::ptr_eq(&first, &hit));
        assert_eq!((CAP_HIT.with(std::cell::Cell::get), ATTEMPTS.with(std::cell::Cell::get),
            last_fill_vertex_peak(), last_fill_cell_count()), expected);
        CAP_HIT.with(|value| value.set(true));
        let uncached = draw();
        assert!(!Arc::ptr_eq(&first, &uncached));
        assert_eq!(*first, *uncached);
    }
}

/// Full-precision variant used by the oracle dump (Moda's `examples/dump_tess.rs`; export-example-absent: Moda's example, not ported).
#[allow(clippy::too_many_arguments)]
pub fn tessellate_combined_oracle(
    points: &[PathCmd],
    stroke_width: f64,
    closed: bool,
    trim: Option<TrimParams>,
    dash: Option<&[f64]>,
    dash_offset: f64,
    line_cap: LineCap,
    line_join: LineJoin,
    fill_rule: FillRule,
    stroke_align: StrokeAlign,
) -> OracleGeometry {
    if points.is_empty() {
        return OracleGeometry {
            positions: Vec::new(),
            vertex_types: Vec::new(),
            bounds: [0.0; 4],
            fill_dropped: None,
            stroke_dropped: None,
        };
    }
    // The TS tessellator sliced the command list to its first MAX_PATH_SEGMENTS (:98-101), so every subpath past
    // command 10,000 of a big `<path d>` (a contour field of ~50k points) was dropped with no warning. Nothing is
    // sliced here: MAX_PATH_SEGMENTS bounds each subpath's flattening (`subpath_to_line_segments`, reported through
    // `note_cap_hit`), and the fill and stroke streams have their own caps, each refused with a reason.

    // Fill (:104-112).
    let PathFill {
        tris: fill_tris,
        bounds: fill_bounds,
        dropped: fill_dropped,
    } = fill_path(points, closed, fill_rule);
    // The TS post-hoc cap (:104-112) emptied the WHOLE item — fill and stroke —
    // with no reason. `BoundedFillBuilder` now bounds the expanded stream
    // inside the sweep, so an over-cap fill is a `fill_dropped` refusal with
    // an empty stream and the check is unreachable.
    debug_assert!(
        fill_tris.len() / 2 <= MAX_FILL_VERTICES,
        "BoundedFillBuilder let {} expanded fill vertices through",
        fill_tris.len() / 2
    );

    // Stroke (:114-150). A dashed stroke is split into pieces first: open pieces
    // (caps at their ends, joins at interior vertices) and closed pieces (a dash
    // that never turned off around a closed sub-path, stroked as the contour).
    let mut stroke: Option<(Vec<f64>, [f64; 4])> = None;
    let mut stroke_dropped = None;
    if stroke_width > 0.0 {
        // Trim (stroke-only) runs BEFORE dash: it reshapes the stroke input into
        // open sub-path windows, while the fill above stays computed from the
        // untrimmed `points`. Trimmed output is always open — so, like dashing, it
        // is never closed/joined/aligned (a trimmed closed shape's stroke falls
        // back to centered; accepted, see the roadmap doc). Identity trim resolves
        // to `None`, keeping the no-trim path byte-identical.
        let trimmed: Option<Vec<PathCmd>> = trim.and_then(|t| {
            if is_trim_identity(t.start, t.end, t.offset) {
                None
            } else {
                Some(apply_trim_to_path(points, closed, t))
            }
        });
        let (trim_points, trim_closed): (&[PathCmd], bool) = match &trimmed {
            Some(tp) => (tp.as_slice(), false),
            None => (points, closed),
        };
        // The dash's drawn fraction scales the estimated stroke vertices.
        let on_fraction = match dash {
            Some(d) if !d.is_empty() => {
                let n = normalize_dash_pattern(scale_dash_pattern(d, stroke_width));
                let total: f64 = n.iter().sum();
                let on: f64 = n.iter().step_by(2).sum();
                if total > 0.0 {
                    (on / total).clamp(0.0, 1.0)
                } else {
                    1.0
                }
            }
            _ => 1.0,
        };
        // The stroke's outer offset, as `generate_stroke_geometry` flattens for.
        let reach = match stroke_align {
            StrokeAlign::Center => stroke_width / 2.0,
            _ => stroke_width,
        };
        let dashed_run = dash.is_some_and(|d| !d.is_empty());
        // An inside / outside stroke that is dashed or trimmed is cut into open
        // pieces, which have no inside: Figma and Illustrator draw such a
        // stroke centred at twice the width, caps included, and keep the half
        // inside (outside) the shape. The pieces are stroked that way here and
        // clipped to the shape's fill region.
        let clip_aligned = closed
            && stroke_align != StrokeAlign::Center
            && (dashed_run || trimmed.is_some());
        let (piece_width, piece_align) = if clip_aligned {
            (stroke_width * 2.0, StrokeAlign::Center)
        } else {
            (stroke_width, stroke_align)
        };
        let dash_error = std::cell::RefCell::new(None);
        let s = stroke_within_caps(trim_points, reach, on_fraction, || {
            let pieces = match dash {
                Some(d) if !d.is_empty() => {
                    // The pattern stays scaled by the stroke's own width.
                    let dashed = match apply_dash_to_path(trim_points, d, stroke_width, trim_closed, dash_offset) {
                        Ok(dashed) => dashed,
                        Err(reason) => {
                            *dash_error.borrow_mut() = Some(reason);
                            return (Vec::new(), [0.0; 4]);
                        }
                    };
                    let open = generate_stroke_geometry(
                        &dashed.open,
                        piece_width,
                        false,
                        line_cap,
                        line_join,
                        piece_align,
                    );
                    let closed = generate_stroke_geometry(
                        &dashed.closed,
                        piece_width,
                        true,
                        line_cap,
                        line_join,
                        piece_align,
                    );
                    concat_strokes(open, closed)
                }
                _ => generate_stroke_geometry(
                    trim_points,
                    piece_width,
                    trim_closed,
                    line_cap,
                    line_join,
                    piece_align,
                ),
            };
            if clip_aligned {
                clip_stroke_to_shape(pieces, points, fill_rule, stroke_align)
            } else {
                pieces
            }
        });
        stroke = filter_stroke(s, &mut stroke_dropped);
        if let Some(reason) = dash_error.into_inner() {
            stroke = None;
            stroke_dropped = Some(reason);
        }
    }

    combine(fill_tris, fill_bounds, fill_dropped, stroke, stroke_dropped)
}

/// The part of a centred stroke's triangles `tris` inside the shape `points`
/// (filled under `fill_rule`) for an inside stroke, or outside it for an
/// outside one, re-tessellated. Used for aligned strokes that dashing or
/// trimming cut into open pieces (see `tessellate_combined_oracle`). A clip
/// that fails keeps the centred stroke.
fn clip_stroke_to_shape(
    stroke: (Vec<f64>, [f64; 4]),
    points: &[PathCmd],
    fill_rule: FillRule,
    align: StrokeAlign,
) -> (Vec<f64>, [f64; 4]) {
    use i_overlay::core::fill_rule::FillRule as OverlayFill;
    use i_overlay::core::overlay_rule::OverlayRule;
    use i_overlay::float::simplify::SimplifyShape;
    use i_overlay::float::single::SingleFloatOverlay;

    let (tris, _) = &stroke;
    if tris.is_empty() {
        return stroke;
    }
    // Every triangle wound the same way, so their nonzero union is the stroke.
    let soup: Vec<Vec<[f64; 2]>> = tris
        .chunks_exact(6)
        .filter_map(|t| {
            let (a, b, c) = ([t[0], t[1]], [t[2], t[3]], [t[4], t[5]]);
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
            if !area.is_finite() || area.abs() < 1e-12 {
                None
            } else if area > 0.0 {
                Some(vec![a, b, c])
            } else {
                Some(vec![a, c, b])
            }
        })
        .collect();
    let band: Vec<Vec<[f64; 2]>> = soup
        .simplify_shape(OverlayFill::NonZero)
        .into_iter()
        .flatten()
        .filter(|c| c.len() >= 3)
        .collect();
    let shape = split_into_subpaths(points)
        .iter()
        .filter(|sp| sp.len() >= 2)
        .map(|sp| subpath_to_line_segments(sp, true, Flat::fill()).map(|s| clean_closed_contour(&s)))
        .collect::<Result<Vec<_>, _>>();
    let shape: Vec<_> = match shape {
        Ok(shape) => shape.into_iter().filter(|c| c.len() >= 3).collect(),
        // The retry owner sees CAP_HIT and retries the entire stroke.
        Err(_) => return (Vec::new(), [0.0; 4]),
    };
    if band.is_empty() || shape.is_empty() {
        return stroke;
    }
    let region: Vec<Vec<[f64; 2]>> = shape
        .simplify_shape(match fill_rule {
            FillRule::EvenOdd => OverlayFill::EvenOdd,
            FillRule::NonZero => OverlayFill::NonZero,
        })
        .into_iter()
        .flatten()
        .filter(|c| c.len() >= 3)
        .collect();
    let rule = if align == StrokeAlign::Outside {
        OverlayRule::Difference
    } else {
        OverlayRule::Intersect
    };
    let clipped: Vec<Vec<P>> = band
        .overlay(&region, rule, OverlayFill::EvenOdd)
        .into_iter()
        .flatten()
        .filter(|c| c.len() >= 3)
        .collect();
    if clipped.is_empty() {
        return (Vec::new(), [0.0; 4]);
    }
    match fill_cells(&clipped, FillRule::EvenOdd, fill_vertex_cap()) {
        Ok((t, _)) if !t.is_empty() => {
            let b = tris_bounds(&t);
            (t, b)
        }
        Ok(_) => (Vec::new(), [0.0; 4]),
        Err(_) => stroke,
    }
}

/// How a vector network's stroke is drawn (`tessellate_network`).
#[derive(Debug, Clone, Copy)]
pub struct NetworkStrokeStyle<'a> {
    pub width: f64,
    pub dash: Option<&'a [f64]>,
    pub dash_offset: f64,
    /// The stroke's cap: trimmed pieces and open arrow carets take it.
    pub line_cap: LineCap,
    pub line_join: LineJoin,
    pub miter_limit: f64,
    pub trim: Option<TrimParams>,
    /// The arrowheads' length and width (`path_arrows::path_arrow_size`).
    pub arrow_size: f64,
}

/// A vector network's body (E7.9, `tess::network`): every region filled on
/// its own under its own rule, then every stroke piece (a dash runs along each
/// piece; each open end with an arrow cap cut back under its arrowhead), the
/// aligned region loops, the separately drawn caps and the junction wedges,
/// and the arrowheads. A trim windows the pieces in order (see
/// `tess::network`). One geometry, fill vertices first, like
/// [`tessellate_combined_with_dash_trim`].
pub fn tessellate_network(
    fills: &[(Vec<PathCmd>, FillRule)],
    stroke: &super::network::NetworkStroke,
    style: &NetworkStrokeStyle,
) -> TessellatedGeometry {
    let stroke_width = style.width;
    let line_join = style.line_join;
    let mut fill_tris: Vec<f64> = Vec::new();
    let mut fill_bounds = [0.0; 4];
    let mut fill_dropped = None;
    let mut first = true;
    for (cmds, rule) in fills {
        let f = fill_path(cmds, true, *rule);
        if f.dropped.is_some() && fill_dropped.is_none() {
            fill_dropped = f.dropped.clone();
        }
        if f.tris.is_empty() {
            continue;
        }
        fill_bounds = if first {
            f.bounds
        } else {
            union_bounds(fill_bounds, f.bounds)
        };
        first = false;
        fill_tris.extend(f.tris);
    }
    let mut stroke_out = None;
    let mut stroke_dropped = None;
    let has_stroke = !stroke.pieces.is_empty() || !stroke.aligned.is_empty();
    if stroke_width > 0.0 && has_stroke {
        // Each piece's body (cut back under its arrowheads) and its heads, per end.
        struct Body {
            cmds: Vec<PathCmd>,
            closed: bool,
            cap: LineCap,
            heads: [Option<super::path_arrows::PathArrows>; 2],
        }
        let bodies: Vec<Body> = stroke
            .pieces
            .iter()
            .map(|p| {
                let [a0, a1] = p.arrows;
                if p.closed || (a0.is_none() && a1.is_none()) {
                    return Body {
                        cmds: p.cmds.clone(),
                        closed: p.closed,
                        cap: p.cap,
                        heads: [None, None],
                    };
                }
                let run = |s, e| {
                    super::path_arrows::run_arrows(
                        &p.cmds,
                        s,
                        e,
                        style.arrow_size,
                        Some("butt"),
                        stroke_width,
                    )
                };
                let body = run(a0, a1)
                    .map(|a| a.body)
                    .unwrap_or_else(|| p.cmds.clone());
                Body {
                    cmds: body,
                    closed: false,
                    cap: p.cap,
                    heads: [
                        a0.and_then(|s| run(Some(s), None)),
                        a1.and_then(|e| run(None, Some(e))),
                    ],
                }
            })
            .collect();
        let mut all: Vec<PathCmd> = bodies.iter().flat_map(|b| b.cmds.clone()).collect();
        for l in &stroke.aligned {
            all.extend(l.iter().cloned());
        }
        let dash = style.dash.filter(|d| !d.is_empty());
        let on_fraction = match dash {
            Some(d) => {
                let n = normalize_dash_pattern(scale_dash_pattern(d, stroke_width));
                let total: f64 = n.iter().sum();
                let on: f64 = n.iter().step_by(2).sum();
                if total > 0.0 {
                    (on / total).clamp(0.0, 1.0)
                } else {
                    1.0
                }
            }
            None => 1.0,
        };
        let dash_error = std::cell::RefCell::new(None);
        let stroke_run = |cmds: &[PathCmd], closed: bool, cap: LineCap| match dash {
            Some(d) => {
                let dashed = match apply_dash_to_path(cmds, d, stroke_width, closed, style.dash_offset) {
                    Ok(dashed) => dashed,
                    Err(reason) => {
                        *dash_error.borrow_mut() = Some(reason);
                        return (Vec::new(), [0.0; 4]);
                    }
                };
                concat_strokes(
                    generate_stroke_geometry(
                        &dashed.open,
                        stroke_width,
                        false,
                        cap,
                        line_join,
                        StrokeAlign::Center,
                    ),
                    generate_stroke_geometry(
                        &dashed.closed,
                        stroke_width,
                        true,
                        cap,
                        line_join,
                        StrokeAlign::Center,
                    ),
                )
            }
            None => generate_stroke_geometry(
                cmds,
                stroke_width,
                closed,
                cap,
                line_join,
                StrokeAlign::Center,
            ),
        };
        let heads_geometry = |h: &super::path_arrows::PathArrows| {
            let mut acc: (Vec<f64>, [f64; 4]) = (Vec::new(), [0.0; 4]);
            if !h.filled.is_empty() {
                let f = fill_path(&h.filled, true, FillRule::NonZero);
                if !f.tris.is_empty() {
                    acc = concat_strokes(acc, (f.tris, f.bounds));
                }
            }
            if !h.open.is_empty() {
                acc = concat_strokes(
                    acc,
                    generate_stroke_geometry(
                        &h.open,
                        stroke_width,
                        false,
                        style.line_cap,
                        line_join,
                        StrokeAlign::Center,
                    ),
                );
            }
            acc
        };
        let s = stroke_within_caps(&all, stroke_width / 2.0, on_fraction, || {
            let mut acc: (Vec<f64>, [f64; 4]) = (Vec::new(), [0.0; 4]);
            if let Some(t) = style.trim {
                // The window runs along the pieces in order; what survives is
                // stroked open with the stroke's cap.
                let trimmed = apply_trim_to_path(&all, false, t);
                acc = concat_strokes(acc, stroke_run(&trimmed, false, style.line_cap));
                let ends: Vec<[f64; 2]> = split_subpaths(&trimmed)
                    .into_iter()
                    .flat_map(|sp| {
                        let pt = |c: &PathCmd| cmd_end(c);
                        [sp.first().map(pt), sp.last().map(pt)]
                    })
                    .flatten()
                    .collect();
                let visible = |p: Option<[f64; 2]>| {
                    p.is_some_and(|p| ends.iter().any(|e| (e[0] - p[0]).hypot(e[1] - p[1]) < 1e-6))
                };
                for b in &bodies {
                    let at = [b.cmds.first().map(cmd_end), b.cmds.last().map(cmd_end)];
                    for k in 0..2 {
                        if let Some(h) = &b.heads[k] {
                            if visible(at[k]) {
                                acc = concat_strokes(acc, heads_geometry(h));
                            }
                        }
                    }
                }
                return acc;
            }
            for b in &bodies {
                if !b.cmds.is_empty() {
                    acc = concat_strokes(acc, stroke_run(&b.cmds, b.closed, b.cap));
                }
                for h in b.heads.iter().flatten() {
                    acc = concat_strokes(acc, heads_geometry(h));
                }
            }
            // Inside / outside: each region's loops, aligned (undashed only).
            for l in &stroke.aligned {
                acc = concat_strokes(
                    acc,
                    generate_stroke_geometry(
                        l,
                        stroke_width,
                        true,
                        style.line_cap,
                        line_join,
                        stroke.align,
                    ),
                );
            }
            // Caps and junctions sit on the vertices, so a dash does not cut them.
            let extra = super::network::cap_and_junction_tris(
                stroke,
                stroke_width / 2.0,
                style.miter_limit,
                flatness_tolerance(),
            );
            if !extra.is_empty() {
                let pts: Vec<P> = extra.chunks(2).map(|c| [c[0], c[1]]).collect();
                let b = calc_bounds(&pts);
                acc = concat_strokes(acc, (extra, b));
            }
            acc
        });
        stroke_out = filter_stroke(s, &mut stroke_dropped);
        if let Some(reason) = dash_error.into_inner() {
            stroke_out = None;
            stroke_dropped = Some(reason);
        }
    }
    let o = combine(fill_tris, fill_bounds, fill_dropped, stroke_out, stroke_dropped);
    let vertex_count = (o.positions.len() / 2) as u32;
    TessellatedGeometry {
        positions: o.positions,
        vertex_types: o.vertex_types,
        vertex_count,
        bounds: [
            o.bounds[0] as f32,
            o.bounds[1] as f32,
            (o.bounds[2] - o.bounds[0]) as f32,
            (o.bounds[3] - o.bounds[1]) as f32,
        ],
        fill_dropped: o.fill_dropped,
        stroke_dropped: o.stroke_dropped,
        // Filled in by `stroke_param::with_stroke_uv` when a path gradient needs it.
        stroke_uv: Vec::new(),
    }
}

/// A command's end point.
fn cmd_end(c: &PathCmd) -> [f64; 2] {
    match c {
        PathCmd::Move { x, y }
        | PathCmd::Line { x, y }
        | PathCmd::Curve { x, y, .. }
        | PathCmd::Quadratic { x, y, .. }
        | PathCmd::Arc { x, y, .. } => [*x, *y],
    }
}

/// Append `b`'s stroke triangles after `a`'s. An empty stroke reports zero bounds,
/// which must not be unioned in (it would drag the box to the origin).
fn concat_strokes(a: (Vec<f64>, [f64; 4]), b: (Vec<f64>, [f64; 4])) -> (Vec<f64>, [f64; 4]) {
    if b.0.is_empty() {
        return a;
    }
    if a.0.is_empty() {
        return b;
    }
    let (mut tris, bounds) = a;
    tris.extend(b.0);
    (tris, union_bounds(bounds, b.1))
}

/// An empty stroke is `None`; one over [`MAX_STROKE_STREAM_VERTICES`] too, with the reason in `dropped`.
fn filter_stroke(
    s: Result<(Vec<f64>, [f64; 4]), String>,
    dropped: &mut Option<String>,
) -> Option<(Vec<f64>, [f64; 4])> {
    let s = match s {
        Ok(s) => s,
        Err(why) => { *dropped = Some(why); return None; }
    };
    let n = s.0.len() / 2;
    if n > MAX_STROKE_STREAM_VERTICES {
        *dropped = Some(format!(
            "the stroke needs {n} vertices, over the {MAX_STROKE_STREAM_VERTICES} cap (split the path into several)"
        ));
        return None;
    }
    (n > 0).then_some(s)
}

/// Combine fill + stroke streams (:152-190), including the bounds quirk.
fn combine(
    fill_tris: Vec<f64>,
    fill_bounds: [f64; 4],
    fill_dropped: Option<String>,
    stroke: Option<(Vec<f64>, [f64; 4])>,
    stroke_dropped: Option<String>,
) -> OracleGeometry {
    let fill_count = fill_tris.len() / 2;
    let stroke_count = stroke.as_ref().map(|s| s.0.len() / 2).unwrap_or(0);
    let total = fill_count + stroke_count;
    // The TS total cap (:152-190) emptied the item with no reason. It is
    // unreachable here: the fill is bounded to MAX_FILL_VERTICES inside the
    // sweep and `filter_stroke` drops a stroke over MAX_STROKE_STREAM_VERTICES
    // (with a reason), and the two caps sum to MAX_TOTAL_VERTICES.
    debug_assert!(
        total <= MAX_TOTAL_VERTICES,
        "fill ({fill_count}) + stroke ({stroke_count}) vertices exceed the total cap"
    );

    let mut positions: Vec<f32> = Vec::with_capacity(total * 2);
    positions.extend(fill_tris.iter().map(|v| *v as f32));
    let mut vertex_types = vec![0.0f32; fill_count];
    // QUIRK (:177-183): bounds start from the fill bounds even when the fill
    // is empty ({0,0,0,0}); stroke bounds are min/maxed in. A stroke-only
    // path therefore reports bounds that include the origin (Q-49).
    let mut bounds = fill_bounds;
    if let Some((stroke_tris, stroke_bounds)) = stroke {
        if !stroke_tris.is_empty() {
            positions.extend(stroke_tris.iter().map(|v| *v as f32));
            vertex_types.extend(std::iter::repeat(1.0f32).take(stroke_count));
        }
        bounds = [
            bounds[0].min(stroke_bounds[0]),
            bounds[1].min(stroke_bounds[1]),
            bounds[2].max(stroke_bounds[2]),
            bounds[3].max(stroke_bounds[3]),
        ];
    }

    OracleGeometry {
        positions,
        vertex_types,
        bounds,
        fill_dropped,
        stroke_dropped,
    }
}

// ---------------------------------------------------------------------------
// Variable-width outlines (plan P8): the stroke is a filled outline
// ---------------------------------------------------------------------------

/// The body FILL of a path whose stroke is drawn as an outline — exactly the
/// fill [`tessellate_combined_oracle`] computes.
pub(crate) fn body_fill(points: &[PathCmd], closed: bool, fill_rule: FillRule) -> PathFill {
    if points.is_empty() {
        return PathFill::EMPTY;
    }
    fill_path(points, closed, fill_rule)
}

/// Fill a stroke outline's (already flattened, closed) contours under the
/// nonzero rule. The outline is the STROKE: it takes the stroke vertex budget
/// (`MAX_STROKE_VERTICES`), not the fill's, and is never truncated by
/// `MAX_PATH_SEGMENTS` (that cap is on the input path). Err = the sweep was
/// refused (caller decimates or falls back; never a page refusal).
#[cfg(test)]
pub(crate) fn fill_outline_contours(contours: &[Vec<P>]) -> Result<(Vec<f64>, [f64; 4]), String> {
    fill_outline_banded(contours).map(|(tris, b, _)| (tris, b))
}

/// [`fill_outline_contours`] with the number of cells the fill took (1 =
/// one sweep). See [`fill_cells`].
pub(crate) fn fill_outline_banded(
    contours: &[Vec<P>],
) -> Result<(Vec<f64>, [f64; 4], usize), String> {
    let contours: Vec<Vec<P>> = contours
        .iter()
        .map(|c| clean_closed_contour(c))
        .filter(|c| c.len() >= 3)
        .collect();
    let (tris, cells) = fill_cells(&contours, FillRule::NonZero, MAX_STROKE_VERTICES)?;
    let b = if tris.is_empty() {
        [0.0; 4]
    } else {
        tris_bounds(&tris)
    };
    Ok((tris, b, cells))
}

fn tris_bounds(tris: &[f64]) -> [f64; 4] {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for v in tris.chunks_exact(2) {
        b[0] = b[0].min(v[0]);
        b[1] = b[1].min(v[1]);
        b[2] = b[2].max(v[0]);
        b[3] = b[3].max(v[1]);
    }
    b
}

/// Deepest bisection of one fill (cells as small as 1/256 of the outline's
/// box on each side).
const MAX_BAND_DEPTH: u32 = 16;

/// Most regions one fill may visit while bisecting (every split counts, so
/// this bounds the clipping and pre-sweep estimates as well as the sweeps).
/// A line of text takes about a dozen; a 600-circle union a few dozen.
const MAX_FILL_REGIONS: usize = 4096;

/// Fill closed, flattened `contours` under `fill_rule`, emitting at most
/// `cap` vertices, and return the triangle stream and the number of cells it
/// took (1 = one sweep).
///
/// lyon's sweep does work per event proportional to the edges crossing the
/// sweep line, so its cost grows with edges × crowding, and it is unbounded on
/// outlines that cross themselves densely. [`check_fill_sweep_cost`] bounds
/// ONE sweep. A fill over that bound is not refused: its box is bisected
/// along its longer side and each half is the outline CLIPPED to it
/// (Sutherland–Hodgman against axis-aligned half-planes, which keeps the
/// winding number of every point inside the half, so nonzero and evenodd both
/// resolve exactly as they would whole), recursively until each cell fits.
/// Splitting a line of text across its width cuts both the edges and the
/// crowding in each cell, so the total work tracks the edge count.
///
/// Cells tile the box without overlap and meet on axis-aligned cut lines,
/// where both sides get bit-identical crossing vertices ([`clip_half_plane`])
/// at an exactly representable coordinate, so the union covers the same
/// samples as one sweep's triangles: no seam, no double blending under a
/// translucent paint. The only geometric change is that an edge crossing a
/// cut is split at a point rounded to f32 — the rounding lyon already applies
/// to every vertex.
///
/// Refusals: a cell still over the sweep bound at [`MAX_BAND_DEPTH`], more
/// than [`MAX_FILL_REGIONS`] regions, or more than `cap` vertices in total
/// (dense self-crossings produce a vertex per crossing, so this is what stops
/// a pathological star). Every refusal is a `fill_dropped`, never a panic.
pub(crate) fn fill_cells(
    contours: &[Vec<P>],
    fill_rule: FillRule,
    cap: usize,
) -> Result<(Vec<f64>, usize), String> {
    FILL_VERTEX_PEAK.with(|c| c.set(0));
    FILL_CELL_COUNT.with(|c| c.set(0));
    if contours.is_empty() {
        return Ok((Vec::new(), 1));
    }
    // lyon's event queue sorts on partial_cmp: a NaN would panic inside the
    // sweep rather than surface as an Err, so refuse non-finite input here —
    // judged after the f32 cast lyon sees, which turns an absurd f64 into inf.
    if contours.iter().any(|c| {
        c.iter()
            .any(|p| !(p[0] as f32).is_finite() || !(p[1] as f32).is_finite())
    }) {
        return Err("non-finite coordinate".to_string());
    }
    let mut rect = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in contours.iter().flatten() {
        rect = [
            rect[0].min(p[0]),
            rect[1].min(p[1]),
            rect[2].max(p[0]),
            rect[3].max(p[1]),
        ];
    }
    let mut cells = CellFill {
        rule: fill_rule,
        cap,
        tris: Vec::new(),
        cells: 0,
        regions: 0,
    };
    let outcome = cells.fill(contours, rect, 0);
    FILL_CELL_COUNT.with(|c| c.set(cells.cells));
    match outcome {
        Ok(()) => Ok((cells.tris, cells.cells.max(1))),
        Err(FillRefusal::Cap) => Err(format!("fill exceeds {cap} vertices")),
        Err(FillRefusal::Complexity(why) | FillRefusal::Failed(why)) => Err(why),
    }
}

/// Why one sweep was refused.
#[derive(Debug)]
enum FillRefusal {
    /// Over the pre-sweep work bound; a smaller cell may fit.
    Complexity(String),
    /// The vertex budget ran out; final.
    Cap,
    /// lyon failed or panicked; final.
    Failed(String),
}

/// State of one [`fill_cells`] call: the stream so far, shared by all cells.
struct CellFill {
    rule: FillRule,
    cap: usize,
    tris: Vec<f64>,
    cells: usize,
    regions: usize,
}

impl CellFill {
    /// Fill `contours` (already clipped to `rect` = [x0, y0, x1, y1]),
    /// bisecting while one sweep would be over the work bound.
    fn fill(&mut self, contours: &[Vec<P>], rect: [f64; 4], depth: u32) -> Result<(), FillRefusal> {
        if contours.is_empty() {
            return Ok(());
        }
        self.regions += 1;
        if self.regions > MAX_FILL_REGIONS {
            return Err(FillRefusal::Complexity(format!(
                "fill exceeds {MAX_FILL_REGIONS} cells"
            )));
        }
        // The stream holds x,y pairs: `tris.len() / 2` expanded vertices.
        let budget = self.cap.saturating_sub(self.tris.len() / 2);
        if budget < 3 {
            return Err(FillRefusal::Cap);
        }
        let axis = if rect[2] - rect[0] >= rect[3] - rect[1] {
            0
        } else {
            1
        };
        // An f32 cut, so the crossing vertices on it are exact in lyon's
        // precision on both sides.
        let mid = (0.5 * (rect[axis] + rect[axis + 2])) as f32 as f64;
        let can_split = depth < MAX_BAND_DEPTH && mid > rect[axis] && mid < rect[axis + 2];
        match check_fill_sweep_cost(contours, budget) {
            Ok(()) => {
                let (t, sound) = sound_sweep(contours, self.rule, budget)?;
                // An unsound triangulation (see `sound_sweep`) is re-cut:
                // different cells give lyon different vertex configurations.
                // Where the cell cannot be cut, it ships as lyon made it —
                // what one sweep of the whole outline would have drawn.
                if sound || !can_split {
                    self.tris.extend(t);
                    self.cells += 1;
                    return Ok(());
                }
            }
            Err(over) if !can_split => return Err(over),
            Err(_) => {}
        }
        let (mut lo, mut hi) = (rect, rect);
        lo[axis + 2] = mid;
        hi[axis] = mid;
        let too_deep = || {
            FillRefusal::Complexity(format!(
                "fill exceeds {MAX_FILL_SWEEP_COST} sweep complexity budget (winds too deep to split)"
            ))
        };
        let a = clip_contours(contours, axis, rect[axis], mid).ok_or_else(too_deep)?;
        self.fill(&a, lo, depth + 1)?;
        drop(a);
        let b = clip_contours(contours, axis, mid, rect[axis + 2]).ok_or_else(too_deep)?;
        self.fill(&b, hi, depth + 1)
    }
}

/// Clip closed contours to the slab `lo <= p[axis] <= hi`: closed contours
/// whose winding number equals the input's at every point strictly inside.
///
/// Sutherland–Hodgman closes each clipped contour with BRIDGES along the slab
/// sides — one per excursion outside — and a cell inherits every bridge of
/// its parent that overlaps it. Bisecting further never removes them, so
/// along a cut through a crowded outline the bridges pile up and a deep cell
/// can cost as much to sweep as its parent. Only their sum matters: winding is
/// additive, and collinear edges of opposite direction cancel everywhere off
/// their line. So every edge on a side (bridge or original) is replaced by
/// that side's NET coverage — usually nothing, or the side itself once where
/// the outline surrounds the cell — split at every vertex another edge ends
/// on, and the edges are re-linked into closed loops. Coverage and fill rule
/// are exactly those of the Sutherland–Hodgman contours.
///
/// A side carries the outline's winding number there as that many copies.
/// `None` when the copies outnumber the real edges (plus a margin): only an
/// outline wound thousands of times over itself does that, and it is refused.
fn clip_contours(contours: &[Vec<P>], axis: usize, lo: f64, hi: f64) -> Option<Vec<Vec<P>>> {
    let o = 1 - axis;
    let mut edges: Vec<(P, P)> = Vec::new();
    // Per side: (from, to) along the side's own coordinate.
    let mut sides: [Vec<(f64, f64)>; 2] = [Vec::new(), Vec::new()];
    for c in contours {
        let (a, b) = c
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| {
                (a.min(p[axis]), b.max(p[axis]))
            });
        if !(b > lo && a < hi) {
            continue;
        }
        let c = clip_half_plane(&clip_half_plane(c, axis, lo, true), axis, hi, false);
        for i in 0..c.len() {
            let (p, q) = (c[i], c[(i + 1) % c.len()]);
            if p == q {
                continue;
            }
            if p[axis] == lo && q[axis] == lo {
                sides[0].push((p[o], q[o]));
            } else if p[axis] == hi && q[axis] == hi {
                sides[1].push((p[o], q[o]));
            } else {
                edges.push((p, q));
            }
        }
    }
    let mut copies_left = 2 * edges.len() + 1024;
    for (side, at) in [lo, hi].into_iter().enumerate() {
        // (coordinate, change in net coverage); 0 marks a vertex to split at.
        let mut marks: Vec<(f64, i32)> = Vec::with_capacity(sides[side].len() * 2);
        for &(from, to) in &sides[side] {
            marks.push((from, 1));
            marks.push((to, -1));
        }
        if marks.is_empty() {
            continue;
        }
        for (p, q) in &edges {
            for v in [p, q] {
                if v[axis] == at {
                    marks.push((v[o], 0));
                }
            }
        }
        marks.sort_by(|a, b| a.0.total_cmp(&b.0));
        let point = |t: f64| {
            let mut r = [0.0; 2];
            r[axis] = at;
            r[o] = t;
            r
        };
        let mut net = 0i32;
        let mut i = 0;
        while i < marks.len() {
            let t = marks[i].0;
            while i < marks.len() && marks[i].0 == t {
                net += marks[i].1;
                i += 1;
            }
            if net != 0 && i < marks.len() {
                let (a, b) = (point(t), point(marks[i].0));
                let e = if net > 0 { (a, b) } else { (b, a) };
                let k = net.unsigned_abs() as usize;
                copies_left = copies_left.checked_sub(k)?;
                edges.extend(std::iter::repeat_n(e, k));
            }
        }
    }
    Some(link_loops(&edges))
}

/// Re-link directed edges whose every vertex has as many edges in as out (a
/// closed edge set) into closed contours. A walk from any unused edge can
/// only stop where it started, so each walk is one contour.
fn link_loops(edges: &[(P, P)]) -> Vec<Vec<P>> {
    let key = |p: &P| ((p[0] + 0.0).to_bits(), (p[1] + 0.0).to_bits());
    let mut out: std::collections::HashMap<(u64, u64), Vec<usize>> =
        std::collections::HashMap::with_capacity(edges.len());
    for (i, (p, _)) in edges.iter().enumerate().rev() {
        out.entry(key(p)).or_default().push(i);
    }
    let mut used = vec![false; edges.len()];
    let mut loops = Vec::new();
    for first in 0..edges.len() {
        if used[first] {
            continue;
        }
        used[first] = true;
        let start = key(&edges[first].0);
        let mut contour = vec![edges[first].0];
        let mut at = edges[first].1;
        while key(&at) != start {
            let next = out.get_mut(&key(&at)).and_then(|list| {
                while let Some(e) = list.pop() {
                    if !used[e] {
                        return Some(e);
                    }
                }
                None
            });
            // Unreachable for a closed edge set; close the contour anyway.
            let Some(e) = next else { break };
            used[e] = true;
            contour.push(at);
            at = edges[e].1;
        }
        if contour.len() >= 3 {
            loops.push(contour);
        }
    }
    loops
}

/// Sutherland–Hodgman against `p[axis] >= at` (`keep_above`) or `<= at`. The
/// crossing point is computed from the edge's endpoints in a canonical order,
/// so the two cells sharing a cut line get bit-identical vertices there.
fn clip_half_plane(poly: &[P], axis: usize, at: f64, keep_above: bool) -> Vec<P> {
    let o = 1 - axis;
    let inside = |p: &P| {
        if keep_above {
            p[axis] >= at
        } else {
            p[axis] <= at
        }
    };
    let cross = |a: &P, b: &P| -> P {
        let (p, q) = if (a[axis], a[o]) <= (b[axis], b[o]) {
            (a, b)
        } else {
            (b, a)
        };
        let t = (at - p[axis]) / (q[axis] - p[axis]);
        let mut r = [0.0; 2];
        r[axis] = at;
        r[o] = p[o] + (q[o] - p[o]) * t;
        r
    };
    let n = poly.len();
    let mut out: Vec<P> = Vec::with_capacity(n + 4);
    for i in 0..n {
        let cur = &poly[i];
        let prev = &poly[(i + n - 1) % n];
        match (inside(prev), inside(cur)) {
            (true, true) => out.push(*cur),
            (true, false) => out.push(cross(prev, cur)),
            (false, true) => {
                out.push(cross(prev, cur));
                out.push(*cur);
            }
            (false, false) => {}
        }
    }
    out
}

/// Combine a body fill with an outline fill emitted as STROKE vertices, so
/// stroke paint (solid / gradient), opacity and blend apply unchanged.
pub(crate) fn combine_fill_and_outline(
    fill: PathFill,
    outline: Option<(Vec<f64>, [f64; 4])>,
) -> crate::seams::TessellatedGeometry {
    let o = combine(
        fill.tris,
        fill.bounds,
        fill.dropped,
        outline.filter(|(t, _)| !t.is_empty()),
        None,
    );
    let vertex_count = (o.positions.len() / 2) as u32;
    crate::seams::TessellatedGeometry {
        positions: o.positions,
        vertex_types: o.vertex_types,
        vertex_count,
        bounds: [
            o.bounds[0] as f32,
            o.bounds[1] as f32,
            (o.bounds[2] - o.bounds[0]) as f32,
            (o.bounds[3] - o.bounds[1]) as f32,
        ],
        fill_dropped: o.fill_dropped,
        stroke_dropped: o.stroke_dropped,
        stroke_uv: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Dash (:205-432)
// ---------------------------------------------------------------------------

/// A dashed stroke's input, split by how it is stroked: `open` pieces (caps at
/// both ends, joins at interior vertices) and `closed` pieces (whole contours).
#[derive(Debug, Default, PartialEq)]
pub struct DashedStroke {
    pub open: Vec<PathCmd>,
    pub closed: Vec<PathCmd>,
}

/// `applyDashToPath` (:218-242). Each sub-path is dashed on its own so a dash
/// never bridges the gap to the next one (a compound path's inner contour, a
/// trimmed shape's wrap runs, a connector body split around a text-gap label).
fn apply_dash_to_path(
    points: &[PathCmd],
    dash_pattern: &[f64],
    stroke_width: f64,
    closed: bool,
    dash_offset: f64,
) -> Result<DashedStroke, String> {
    let normalized = normalize_dash_pattern(scale_dash_pattern(dash_pattern, stroke_width));
    // `dashOffset` (marching ants) is authored in the same units as the dash
    // array and scaled the same way, so one full loop = the authored pattern
    // sum.
    let scaled_offset = dash_offset * stroke_width / BASE_DASH_STROKE_WIDTH;
    let aligned = dash_alignment_now();
    let mut out = DashedStroke::default();
    // Bound work, including gaps and runs too short to emit vertices. The stroke vertex
    // cap cannot bound those: tiny positive lengths can take billions of walk steps.
    const MAX_DASH_STEPS: f64 = 1_000_000.0;
    let period: f64 = normalized.iter().sum();
    let mut steps = 0.0;
    for sp in split_subpaths(points) {
        let raw = path_points_to_segments(sp, closed);
        let total: f64 = raw.iter().map(|s| s.arc_length).sum();
        if total < 0.001 { continue; } // The arc-length walker emits nothing for these subpaths.
        if period > 0.0 {
            // Each segment can start/end in a partial period; a phase adds at most one.
            steps += ((total / period).ceil() + raw.len() as f64 + 1.0) * normalized.len() as f64;
            if !steps.is_finite() || steps > MAX_DASH_STEPS {
                return Err(format!("stroke-dasharray expansion exceeds the {MAX_DASH_STEPS} step budget"));
            }
        }
        // Aligned dashes: the pattern is re-fitted per span between corners, as an explicit on/off run over the
        // sub-path (the offset does not apply). Corners come from the unflattened segments.
        let aligned_pattern = aligned.then(|| {
            let corners = segment_corner_positions(&raw, closed);
            intervals_as_walk_pattern(&aligned_dash_intervals(total, &corners, closed, &normalized), total)
        });
        // Dashes follow the curve: the walk runs over the path flattened like
        // the stroke (each curve's pieces keep its measured arc length, so dash
        // positions are unchanged), and emits every vertex inside a dash —
        // not a chord between the dash's ends.
        // The last resort (`stroke_within_caps`) dashes exactly as before
        // curve-following dashes: chords between dash ends, far fewer vertices.
        let segments = if LEGACY.with(std::cell::Cell::get) {
            raw
        } else {
            flatten_path_segments(
                &raw,
                Flat::stroke(stroke_width / 2.0),
            )
        };
        let (Some(first), Some(last)) = (segments.first(), segments.last()) else {
            continue;
        };
        let walked = if let Some(p) = &aligned_pattern {
            walk_path_by_arc_length(&segments, p)
        } else if dash_offset == 0.0 {
            walk_path_by_arc_length(&segments, &normalized)
        } else {
            walk_path_by_arc_length_phased(&segments, &normalized, scaled_offset)
        };
        for piece in dash_pieces(&walked, closed, seg_start(&first.seg), seg_end(&last.seg)) {
            let target = if piece.closed {
                &mut out.closed
            } else {
                &mut out.open
            };
            let mut pts = piece.points.into_iter();
            if let Some(p) = pts.next() {
                target.push(PathCmd::Move { x: p[0], y: p[1] });
            }
            target.extend(pts.map(|p| PathCmd::Line { x: p[0], y: p[1] }));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Trim paths (stroke-only) — docs/roadmap/canvas/stroke-path-animation.md
// ---------------------------------------------------------------------------

/// Trim parameters as fractions of arc length (or, with `by` an axis, of the
/// path's extent along it: `trim_axis.rs`). `individual` selects the
/// concatenate-and-slide multi-subpath mode (`trimPathMode: 'individual'`);
/// the default `false` is `'simultaneous'`.
#[derive(Debug, Clone, Copy)]
pub struct TrimParams {
    pub start: f64,
    pub end: f64,
    pub offset: f64,
    pub individual: bool,
    /// `trimPathBy`: arc length (the default), or the x or y axis.
    pub by: TrimBy,
}

const TRIM_EPS: f64 = 1e-4;

/// True when trim leaves the path unchanged (full path visible), so the caller
/// can skip the pass and stay byte-identical to the untrimmed render.
///
/// A full `[0,1]` window shows the entire path for ANY offset (rotating a full
/// loop is a no-op), so offset is intentionally ignored here — this keeps an
/// untrimmed-but-offset closed shape from degrading to an open (unjoined) stroke.
pub fn is_trim_identity(start: f64, end: f64, _offset: f64) -> bool {
    start <= TRIM_EPS && end >= 1.0 - TRIM_EPS
}

/// A node's trim from its fields (`trimPathStart` / `trimPathEnd` / `trimPathOffset` / `trimPathMode` /
/// `trimPathBy`, read through `num` / `mode` / `by`), or `None` when it shows the whole path. The one reading shared
/// by the renderer (`frame.rs`) and the `stroke_outline` export (`outline.rs`). Non-finite values read as their
/// defaults.
pub fn trim_params_from_fields(
    num: impl Fn(&str) -> Option<f64>,
    mode: Option<&str>,
    by: Option<&str>,
) -> Option<TrimParams> {
    let field = |k: &str, default: f64| num(k).filter(|v| v.is_finite()).unwrap_or(default);
    let (start, end, offset) = (
        field("trimPathStart", 0.0),
        field("trimPathEnd", 1.0),
        field("trimPathOffset", 0.0),
    );
    if is_trim_identity(start, end, offset) {
        return None;
    }
    Some(TrimParams {
        start,
        end,
        offset,
        individual: mode == Some("individual"),
        by: TrimBy::parse(by),
    })
}

fn seg_start(seg: &Seg) -> Pt {
    match seg {
        Seg::Line { p0, .. } | Seg::Quadratic { p0, .. } | Seg::Cubic { p0, .. } => *p0,
    }
}

fn seg_end(seg: &Seg) -> Pt {
    match seg {
        Seg::Line { p1, .. } | Seg::Quadratic { p1, .. } | Seg::Cubic { p1, .. } => *p1,
    }
}

/// Emit a curve-preserving segment run as an open sub-path (`Move` + one command
/// per segment) onto `out`.
fn segments_to_cmds(segs: &[PathSegment], out: &mut Vec<PathCmd>) {
    if segs.is_empty() {
        return;
    }
    let p0 = seg_start(&segs[0].seg);
    out.push(PathCmd::Move { x: p0[0], y: p0[1] });
    for s in segs {
        match &s.seg {
            Seg::Line { p1, .. } => out.push(PathCmd::Line { x: p1[0], y: p1[1] }),
            Seg::Quadratic { p1, cp, .. } => out.push(PathCmd::Quadratic {
                x: p1[0],
                y: p1[1],
                cp: (cp[0], cp[1]),
            }),
            Seg::Cubic { p1, cp1, cp2, .. } => out.push(PathCmd::Curve {
                x: p1[0],
                y: p1[1],
                cp1: (cp1[0], cp1[1]),
                cp2: (cp2[0], cp2[1]),
            }),
        }
    }
}

/// Split a command stream into sub-paths at each `Move` (each slice keeps its
/// leading `Move`).
fn split_subpaths(points: &[PathCmd]) -> Vec<&[PathCmd]> {
    let mut out: Vec<&[PathCmd]> = Vec::new();
    let mut start = 0usize;
    for (i, c) in points.iter().enumerate() {
        if i > start && matches!(c, PathCmd::Move { .. }) {
            out.push(&points[start..i]);
            start = i;
        }
    }
    if start < points.len() {
        out.push(&points[start..]);
    }
    out
}

/// Keep the arc-length window [a, b] of one sub-path as curve-preserving run(s).
///
/// The window start is normalized modulo `len`, so an offset that pushes the
/// window past the end WRAPS rather than clamping to empty (this is what lets a
/// trimmed-and-offset OPEN line "march" and reappear at the start instead of
/// silently vanishing). When the window wraps the seam:
/// - closed sub-path: the two halves meet at the seam point → concatenated into
///   ONE continuous run (continuous dash phase, no double caps at a
///   visually-continuous point);
/// - open sub-path: the two halves are disconnected (no closing seam) → returned
///   as TWO separate runs so nothing is drawn across the open path's endpoints.
///
/// Returns 0, 1, or 2 runs. (`b <= a`, i.e. an empty/inverted window, yields 0.)
fn window_runs(
    segs: &[PathSegment],
    len: f64,
    closed: bool,
    a: f64,
    b: f64,
) -> Vec<Vec<PathSegment>> {
    let mut out: Vec<Vec<PathSegment>> = Vec::new();
    if len <= TRIM_EPS || b - a <= TRIM_EPS {
        return out;
    }
    // Window covers the whole (or more than the whole) path → keep everything.
    if b - a >= len - TRIM_EPS {
        out.push(segs.to_vec());
        return out;
    }
    let s = a.rem_euclid(len); // normalize start into [0, len) — wraps negatives too
    let e = s + (b - a);
    if e <= len + TRIM_EPS {
        let r = extract_path_segments_by_arc_length(segs, s, e.min(len));
        if !r.is_empty() {
            out.push(r);
        }
        return out;
    }
    // Window wraps past the seam.
    let first = extract_path_segments_by_arc_length(segs, s, len);
    let second = extract_path_segments_by_arc_length(segs, 0.0, e - len);
    if closed {
        // Contiguous at the seam → one continuous run.
        let mut merged = first;
        merged.extend(second);
        if !merged.is_empty() {
            out.push(merged);
        }
    } else {
        // Disconnected across the open path's endpoints → two separate runs.
        if !first.is_empty() {
            out.push(first);
        }
        if !second.is_empty() {
            out.push(second);
        }
    }
    out
}

/// Apply trim to a path, returning new OPEN sub-paths. The result feeds the
/// stroke pipeline only (fill is tessellated from the untrimmed `points`).
fn apply_trim_to_path(points: &[PathCmd], closed: bool, t: TrimParams) -> Vec<PathCmd> {
    // Per-subpath curve-preserving segments + arc lengths.
    let mut subs: Vec<(Vec<PathSegment>, f64)> = Vec::new();
    for sp in split_subpaths(points) {
        let segs = path_points_to_segments(sp, closed);
        let len: f64 = segs.iter().map(|s| s.arc_length).sum();
        if len > 0.001 && !segs.is_empty() {
            subs.push((segs, len));
        }
    }
    if subs.is_empty() {
        return Vec::new();
    }
    if let Some(cut) = axis_trim_runs(&subs, closed, t) {
        let mut out = Vec::new();
        for (segs, runs) in cut {
            for run in runs {
                segments_to_cmds(&trim_axis::run_segments(segs, &run), &mut out);
            }
        }
        return out;
    }

    // Clamp start/end to [0,1] — an animatable keyframe/ease can overshoot the
    // authored range (AE clamps trim likewise). Offset is intentionally NOT
    // clamped (it wraps). NaN falls through to empty windows downstream.
    let start = t.start.clamp(0.0, 1.0);
    let end0 = t.end.clamp(0.0, 1.0);
    // Normalize offset into one turn so an offset animated past ±1 still reaches
    // the window (open paths would otherwise push it off the path; the individual
    // ±total shift loop relies on this too).
    let off = t.offset.rem_euclid(1.0);

    let mut out: Vec<PathCmd> = Vec::new();

    if t.individual && subs.len() > 1 {
        let total: f64 = subs.iter().map(|(_, l)| *l).sum();
        if total <= TRIM_EPS {
            return out;
        }
        // `start > end` wraps across the concatenation when it is loop-like (any
        // sub-path closed); otherwise it selects nothing.
        let end = if end0 < start && closed {
            end0 + 1.0
        } else {
            end0
        };
        let a = (start + off) * total;
        let b = (end + off) * total;
        if b - a <= TRIM_EPS {
            return out;
        }
        let mut cursor = 0.0f64;
        for (segs, len) in &subs {
            let seg_start = cursor;
            cursor += *len;
            // Intersect the (possibly wrapping) global window with this sub-path's
            // span; ±total shifts resolve the seam. Overlap at window≈total is
            // harmless overdraw of identical geometry.
            for shift in [-total, 0.0, total] {
                let lo = (a + shift).max(seg_start);
                let hi = (b + shift).min(seg_start + *len);
                if hi - lo > TRIM_EPS {
                    // Windows are pre-bounded to [0,len] here, so no seam wrap; still
                    // iterate for the uniform return type.
                    for run in window_runs(segs, *len, false, lo - seg_start, hi - seg_start) {
                        segments_to_cmds(&run, &mut out);
                    }
                }
            }
        }
        return out;
    }

    // Simultaneous (default): each sub-path trims to [start,end] of its own length.
    for (segs, len) in &subs {
        // `start > end` wraps across the seam for closed sub-paths; open sub-paths
        // have no seam, so the window stays empty.
        let end = if end0 < start && closed {
            end0 + 1.0
        } else {
            end0
        };
        let a = (start + off) * *len;
        let b = (end + off) * *len;
        for run in window_runs(segs, *len, closed, a, b) {
            segments_to_cmds(&run, &mut out);
        }
    }
    out
}

/// A trim along an axis (`trim_axis.rs`) on measured subpaths: each subpath's kept runs, or `None` when the trim is
/// by length (or the path has no extent on the axis, which trims by length too).
fn axis_trim_runs<'a>(
    subs: &'a [(Vec<PathSegment>, f64)],
    closed: bool,
    t: TrimParams,
) -> Option<Vec<(&'a [PathSegment], Vec<Vec<trim_axis::Piece>>)>> {
    let axis = t.by.axis()?;
    let ext = trim_axis::extent(subs.iter().map(|(s, _)| s.as_slice()), axis)?;
    let (lo, hi) = trim_axis::window(ext, t.start, t.end)?;
    Some(
        subs.iter()
            .map(|(segs, _)| (segs.as_slice(), trim_axis::runs(segs, closed, axis, lo, hi)))
            .collect(),
    )
}

/// A trim along an axis as arc-length windows, for the lanes that cut by arc length (the stroke outliner, placed
/// brushes): per subpath of `points` (every one, in order, those with no length included), its length and its kept
/// windows `(start, end)` (`end` may pass a closed subpath's length through its seam). `None` when the trim is by
/// length or the path has no extent on the axis.
pub fn axis_trim_arc_windows(
    points: &[PathCmd],
    closed: bool,
    t: TrimParams,
) -> Option<Vec<(f64, Vec<(f64, f64)>)>> {
    t.by.axis()?;
    let subs: Vec<(Vec<PathSegment>, f64)> = split_subpaths(points)
        .into_iter()
        .map(|sp| {
            let segs = path_points_to_segments(sp, closed);
            let len = segs.iter().map(|s| s.arc_length).sum();
            (segs, len)
        })
        .collect();
    let cut = axis_trim_runs(&subs, closed, t)?;
    Some(
        cut.into_iter()
            .zip(&subs)
            .map(|((segs, runs), (_, len))| {
                (*len, runs.iter().map(|r| trim_axis::run_arc_window(segs, r)).collect())
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Open path ends (E6.1 arrowheads on open paths)
// ---------------------------------------------------------------------------

/// An open path end: the anchor and the unit tangent pointing OUT of the path there (away from the path's body).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathEnd {
    pub point: Pt,
    pub out: Pt,
}

/// The first and last drawn segments of an open path's first and last sub-paths (sub-paths with no length are
/// skipped), or `None` when nothing has length.
fn end_segments(points: &[PathCmd]) -> Option<(PathSegment, PathSegment)> {
    let subs: Vec<Vec<PathSegment>> = split_subpaths(points)
        .into_iter()
        .map(|sp| path_points_to_segments(sp, false))
        .filter(|s| !s.is_empty())
        .collect();
    Some((subs.first()?.first()?.clone(), subs.last()?.last()?.clone()))
}

fn unit(v: Pt) -> Option<Pt> {
    let l = v[0].hypot(v[1]);
    (l > 1e-9 && l.is_finite()).then(|| [v[0] / l, v[1] / l])
}

/// The first point of `candidates` away from `p` (a retracted handle sits on its anchor): the direction the
/// path leaves `p` in.
fn tangent_from(p: Pt, candidates: &[Pt]) -> Option<Pt> {
    candidates.iter().find_map(|q| {
        unit([q[0] - p[0], q[1] - p[1]]).filter(|_| (q[0] - p[0]).hypot(q[1] - p[1]) > 1e-6)
    })
}

/// The start and end of an open path, each with its outward end tangent (E6.1): where a stroke's `startArrow`
/// / `endArrow` sit. The start is the first sub-path's first anchor, the end the last sub-path's last anchor.
pub fn open_path_ends(points: &[PathCmd]) -> Option<(PathEnd, PathEnd)> {
    let (first, last) = end_segments(points)?;
    let start = match first.seg {
        Seg::Line { p0, p1 } => (p0, tangent_from(p0, &[p1])?),
        Seg::Quadratic { p0, p1, cp } => (p0, tangent_from(p0, &[cp, p1])?),
        Seg::Cubic { p0, p1, cp1, cp2 } => (p0, tangent_from(p0, &[cp1, cp2, p1])?),
    };
    let end = match last.seg {
        Seg::Line { p0, p1 } => (p1, tangent_from(p1, &[p0])?),
        Seg::Quadratic { p0, p1, cp } => (p1, tangent_from(p1, &[cp, p0])?),
        Seg::Cubic { p0, p1, cp1, cp2 } => (p1, tangent_from(p1, &[cp2, cp1, p0])?),
    };
    // The tangents found point INTO the path; the arrowheads point out.
    Some((
        PathEnd {
            point: start.0,
            out: [-start.1[0], -start.1[1]],
        },
        PathEnd {
            point: end.0,
            out: [-end.1[0], -end.1[1]],
        },
    ))
}

/// An open path with `start` px of arc length cut off its first sub-path's start and `end` px off its last
/// sub-path's end (curves kept): the stroke under a stroke's arrowheads. A sub-path cut to nothing is dropped.
pub fn cut_open_path_ends(points: &[PathCmd], start: f64, end: f64) -> Vec<PathCmd> {
    let subs: Vec<Vec<PathSegment>> = split_subpaths(points)
        .into_iter()
        .map(|sp| path_points_to_segments(sp, false))
        .filter(|s| !s.is_empty())
        .collect();
    let n = subs.len();
    let mut out = Vec::new();
    for (i, segs) in subs.iter().enumerate() {
        let len: f64 = segs.iter().map(|s| s.arc_length).sum();
        let a = if i == 0 { start.max(0.0) } else { 0.0 };
        let b = len - if i + 1 == n { end.max(0.0) } else { 0.0 };
        if a <= 0.0 && b >= len {
            segments_to_cmds(segs, &mut out);
        } else if b - a > TRIM_EPS {
            segments_to_cmds(&extract_path_segments_by_arc_length(segs, a, b), &mut out);
        }
    }
    out
}

/// Curved dash-walk segments → line pieces at `flat`, each curve's pieces
/// scaled to sum to its measured arc length.
fn flatten_path_segments(segs: &[PathSegment], flat: Flat) -> Vec<PathSegment> {
    // A memory bound only: dash GAPS flatten too, so the piece count says
    // nothing about the stroke's vertices (sparse dashes on a long curve have
    // many pieces and few vertices). Past it, `stroke_within_caps` retries
    // coarser; its last resort (the pre-band flattening) is never cut.
    const MAX_PIECES: usize = 1_000_000;
    let capped = !LEGACY.with(std::cell::Cell::get);
    let mut out = Vec::with_capacity(segs.len());
    for s in segs {
        if capped && out.len() > MAX_PIECES {
            note_cap_hit();
            break;
        }
        let pts = match s.seg {
            Seg::Line { .. } => {
                out.push(s.clone());
                continue;
            }
            Seg::Quadratic { p0, p1, cp } => {
                let c1 = [
                    p0[0] + (2.0 / 3.0) * (cp[0] - p0[0]),
                    p0[1] + (2.0 / 3.0) * (cp[1] - p0[1]),
                ];
                let c2 = [
                    p1[0] + (2.0 / 3.0) * (cp[0] - p1[0]),
                    p1[1] + (2.0 / 3.0) * (cp[1] - p1[1]),
                ];
                tessellate_cubic_bezier(p0, c1, c2, p1, 0, flat)
            }
            Seg::Cubic { p0, p1, cp1, cp2 } => tessellate_cubic_bezier(p0, cp1, cp2, p1, 0, flat),
        };
        let chords: Vec<f64> = pts
            .windows(2)
            .map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]))
            .collect();
        let total: f64 = chords.iter().sum();
        let k = if total > 0.0 {
            s.arc_length / total
        } else {
            0.0
        };
        for (w, c) in pts.windows(2).zip(&chords) {
            out.push(PathSegment {
                seg: Seg::Line { p0: w[0], p1: w[1] },
                arc_length: c * k,
            });
        }
    }
    out
}

/// `pathPointsToSegments` (:251-395).
fn path_points_to_segments(points: &[PathCmd], closed: bool) -> Vec<PathSegment> {
    let mut segments: Vec<PathSegment> = Vec::new();
    let mut current_pos: P = [0.0, 0.0];
    let mut subpath_start: P = [0.0, 0.0];
    let mut subpath_has_segments = false;

    for point in points {
        match point {
            PathCmd::Move { x, y } => {
                if closed && subpath_has_segments {
                    let closing_length = (subpath_start[0] - current_pos[0])
                        .hypot(subpath_start[1] - current_pos[1]);
                    if closing_length > 0.001 {
                        segments.push(PathSegment {
                            seg: Seg::Line {
                                p0: current_pos,
                                p1: subpath_start,
                            },
                            arc_length: closing_length,
                        });
                    }
                }
                current_pos = [*x, *y];
                subpath_start = current_pos;
                subpath_has_segments = false;
            }
            PathCmd::Line { x, y } => {
                let end_pos = [*x, *y];
                let length = (end_pos[0] - current_pos[0]).hypot(end_pos[1] - current_pos[1]);
                if length > 0.001 {
                    segments.push(PathSegment {
                        seg: Seg::Line {
                            p0: current_pos,
                            p1: end_pos,
                        },
                        arc_length: length,
                    });
                    subpath_has_segments = true;
                }
                current_pos = end_pos;
            }
            PathCmd::Curve { x, y, cp1, cp2 } => {
                let end_pos = [*x, *y];
                let cp1p = [cp1.0, cp1.1];
                let cp2p = [cp2.0, cp2.1];
                let arc_length = cubic_arc_length(current_pos, cp1p, cp2p, end_pos);
                if arc_length > 0.001 {
                    segments.push(PathSegment {
                        seg: Seg::Cubic {
                            p0: current_pos,
                            p1: end_pos,
                            cp1: cp1p,
                            cp2: cp2p,
                        },
                        arc_length,
                    });
                    subpath_has_segments = true;
                }
                current_pos = end_pos;
            }
            PathCmd::Quadratic { x, y, cp } => {
                let end_pos = [*x, *y];
                let cpp = [cp.0, cp.1];
                let arc_length = quadratic_arc_length(current_pos, cpp, end_pos);
                if arc_length > 0.001 {
                    segments.push(PathSegment {
                        seg: Seg::Quadratic {
                            p0: current_pos,
                            p1: end_pos,
                            cp: cpp,
                        },
                        arc_length,
                    });
                    subpath_has_segments = true;
                }
                current_pos = end_pos;
            }
            PathCmd::Arc {
                x,
                y,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            } => {
                let arc = ArcParams {
                    rx: *rx,
                    ry: *ry,
                    x_axis_rotation: *x_axis_rotation,
                    large_arc: *large_arc,
                    sweep: *sweep,
                    x: *x,
                    y: *y,
                };
                let beziers = arc_to_beziers(current_pos[0], current_pos[1], &arc);
                if beziers.is_empty() {
                    let end_pos = [*x, *y];
                    let length = (end_pos[0] - current_pos[0]).hypot(end_pos[1] - current_pos[1]);
                    if length > 0.001 {
                        segments.push(PathSegment {
                            seg: Seg::Line {
                                p0: current_pos,
                                p1: end_pos,
                            },
                            arc_length: length,
                        });
                        subpath_has_segments = true;
                    }
                    current_pos = end_pos;
                } else {
                    let mut bezier_start = current_pos;
                    for b in &beziers {
                        let cp1p = [b.cp1[0], b.cp1[1]];
                        let cp2p = [b.cp2[0], b.cp2[1]];
                        let end_pos = [b.end[0], b.end[1]];
                        let arc_length = cubic_arc_length(bezier_start, cp1p, cp2p, end_pos);
                        if arc_length > 0.001 {
                            segments.push(PathSegment {
                                seg: Seg::Cubic {
                                    p0: bezier_start,
                                    p1: end_pos,
                                    cp1: cp1p,
                                    cp2: cp2p,
                                },
                                arc_length,
                            });
                            subpath_has_segments = true;
                        }
                        bezier_start = end_pos;
                    }
                    current_pos = [*x, *y];
                }
            }
        }
    }

    if closed && subpath_has_segments {
        let closing_length =
            (subpath_start[0] - current_pos[0]).hypot(subpath_start[1] - current_pos[1]);
        if closing_length > 0.001 {
            segments.push(PathSegment {
                seg: Seg::Line {
                    p0: current_pos,
                    p1: subpath_start,
                },
                arc_length: closing_length,
            });
        }
    }

    segments
}

// ---------------------------------------------------------------------------
// Fill
// ---------------------------------------------------------------------------

/// Flat triangle xy stream of a path's fill. The path is flattened with the
/// same De Casteljau / arc code the stroke uses (so the two agree on every
/// edge), then handed to lyon's sweep-line fill tessellator, which resolves
/// self-intersections and holes from the fill rule alone. Public for the
/// holes / fill-rule suites.
pub fn tessellate_path(points: &[PathCmd], closed: bool, fill_rule: FillRule) -> Vec<f64> {
    fill_path(points, closed, fill_rule).tris
}

/// A path's fill: the triangle stream, its bounds, and — when the fill was
/// refused (fail-closed) — the reason. The plan builder files it in
/// `PlanWarnings.dropped` (a fill that would have painted, or a clip mask:
/// the page ships without it and names it) or `PlanWarnings.skipped` (a fill whose
/// resolved paint is invisible — alpha-0 solid, a line body — where the
/// empty stream degrades only bounds). A legitimately empty fill (too few
/// points, zero area) is not a drop and carries no reason.
#[derive(Clone)]
pub(crate) struct PathFill {
    tris: Vec<f64>,
    bounds: [f64; 4],
    dropped: Option<String>,
}

impl PathFill {
    const EMPTY: PathFill = PathFill {
        tris: Vec::new(),
        bounds: [0.0; 4],
        dropped: None,
    };

    fn dropped(why: String) -> PathFill {
        PathFill {
            dropped: Some(why),
            ..PathFill::EMPTY
        }
    }
}

/// [`fill_path_at`] at the finest flattening that fits the caps, never worse
/// than before zoom bands: the band's flattening first (or, when the up-front
/// estimate already exceeds `MAX_PATH_SEGMENTS`, the first coarser band that
/// fits), one band coarser whenever the fill is truncated or dropped over its
/// vertex cap, and the pre-band 0.5-unit flattening as the last resort.
/// Every attempt must produce the complete outline.
fn fill_path(points: &[PathCmd], closed: bool, fill_rule: FillRule) -> PathFill {
    // A cap already hit by the caller's attempt: the fill may only add to it, which the memo doesn't model.
    if CAP_HIT.with(std::cell::Cell::get) {
        return fill_path_uncached(points, closed, fill_rule);
    }
    let inputs = FillInputs::now(closed, fill_rule);
    if let Some(hit) = FILL_MEMO.with(|m| m.borrow().get(points, &inputs)) {
        return hit;
    }
    let out = fill_path_uncached(points, closed, fill_rule);
    FILL_MEMO.with(|m| m.borrow_mut().put(points, inputs, &out));
    out
}

/// Everything besides the path that [`fill_path_uncached`] reads: its own arguments and the thread's flattening
/// band, coarsening, legacy flag, sweep budget and vertex cap.
#[derive(Clone, Copy, PartialEq)]
struct FillInputs {
    closed: bool,
    fill_rule: FillRule,
    band: i32,
    coarsen: i32,
    legacy: bool,
    sweep_budget: usize,
    vertex_cap: usize,
}

impl FillInputs {
    fn now(closed: bool, fill_rule: FillRule) -> FillInputs {
        FillInputs {
            closed,
            fill_rule,
            band: super::outline::current_band(),
            coarsen: COARSEN.with(std::cell::Cell::get),
            legacy: LEGACY.with(std::cell::Cell::get),
            sweep_budget: SWEEP_BUDGET.with(std::cell::Cell::get),
            vertex_cap: fill_vertex_cap(),
        }
    }
}

/// One remembered fill: its inputs, the result and what the fill left in the thread's cells (`CAP_HIT`, which the
/// caller reads, and the fill-statistics test hooks).
struct FillMemoEntry {
    points: Vec<PathCmd>,
    inputs: FillInputs,
    out: PathFill,
    cap_hit: bool,
    vertex_peak: usize,
    cell_count: usize,
}

/// The fills of this thread's recent paths, by an exact match of the path and every input: a motion export
/// re-plans the same outlines frame after frame (a 3D plane's layers at a camera scale that changes every frame,
/// in the same zoom band), and the sweep is most of a plan. Bounded by [`FILL_MEMO_FLOATS`]: emptied when full.
#[derive(Default)]
struct FillMemo {
    entries: std::collections::HashMap<u64, Vec<FillMemoEntry>>,
    floats: usize,
}

/// Triangle-stream and path floats the fill memo holds at most (64 MB).
const FILL_MEMO_FLOATS: usize = 8 << 20;

fn fill_memo_hash(points: &[PathCmd], inputs: &FillInputs) -> u64 {
    use std::hash::Hasher;
    let mut h = rustc_hash::FxHasher::default();
    h.write_usize(points.len());
    let mut f = |v: f64| h.write_u64(v.to_bits());
    for p in points {
        match *p {
            PathCmd::Move { x, y } => [0.0, x, y].into_iter().for_each(&mut f),
            PathCmd::Line { x, y } => [1.0, x, y].into_iter().for_each(&mut f),
            PathCmd::Curve { x, y, cp1, cp2 } => [2.0, x, y, cp1.0, cp1.1, cp2.0, cp2.1].into_iter().for_each(&mut f),
            PathCmd::Quadratic { x, y, cp } => [3.0, x, y, cp.0, cp.1].into_iter().for_each(&mut f),
            PathCmd::Arc { x, y, rx, ry, x_axis_rotation, large_arc, sweep } => {
                [4.0, x, y, rx, ry, x_axis_rotation, f64::from(u8::from(large_arc)), f64::from(u8::from(sweep))]
                    .into_iter()
                    .for_each(&mut f)
            }
        }
    }
    h.write_u8(u8::from(inputs.closed));
    h.write_u8(inputs.fill_rule as u8);
    h.write_i32(inputs.band);
    h.finish()
}

impl FillMemo {
    fn get(&self, points: &[PathCmd], inputs: &FillInputs) -> Option<PathFill> {
        let e = self
            .entries
            .get(&fill_memo_hash(points, inputs))?
            .iter()
            .find(|e| e.inputs == *inputs && e.points == points)?;
        CAP_HIT.with(|c| c.set(e.cap_hit));
        FILL_VERTEX_PEAK.with(|c| c.set(e.vertex_peak));
        FILL_CELL_COUNT.with(|c| c.set(e.cell_count));
        Some(e.out.clone())
    }

    fn put(&mut self, points: &[PathCmd], inputs: FillInputs, out: &PathFill) {
        let floats = out.tris.len() + points.len() * 8;
        if floats > FILL_MEMO_FLOATS / 4 {
            return;
        }
        if self.floats + floats > FILL_MEMO_FLOATS {
            self.entries.clear();
            self.floats = 0;
        }
        self.floats += floats;
        self.entries.entry(fill_memo_hash(points, &inputs)).or_default().push(FillMemoEntry {
            points: points.to_vec(),
            inputs,
            out: out.clone(),
            cap_hit: CAP_HIT.with(std::cell::Cell::get),
            vertex_peak: FILL_VERTEX_PEAK.with(std::cell::Cell::get),
            cell_count: FILL_CELL_COUNT.with(std::cell::Cell::get),
        });
    }
}

thread_local! {
    static FILL_MEMO: std::cell::RefCell<FillMemo> = std::cell::RefCell::new(FillMemo::default());
}

fn fill_path_uncached(points: &[PathCmd], closed: bool, fill_rule: FillRule) -> PathFill {
    let prev = (
        COARSEN.with(std::cell::Cell::get),
        LEGACY.with(std::cell::Cell::get),
    );
    let set = |c: i32, legacy: bool| {
        COARSEN.with(|k| k.set(c));
        LEGACY.with(|k| k.set(legacy));
    };
    let band = super::outline::current_band().max(0);
    let out = if band == 0 || !has_curve_cmds(points) {
        set(0, false);
        fill_path_at(points, closed, fill_rule)
    } else {
        let start = (0..=band).find(|&c| {
            set(c, false);
            estimate_flattened(points, Flat::fill()).0 <= MAX_PATH_SEGMENTS
        });
        let mut done = None;
        if let Some(start) = start {
            for c in start..=band {
                set(c, false);
                CAP_HIT.with(|k| k.set(false));
                let f = fill_path_at(points, closed, fill_rule);
                if !CAP_HIT.with(std::cell::Cell::get) && f.dropped.is_none() {
                    done = Some(f);
                    break;
                }
            }
        }
        done.unwrap_or_else(|| {
            set(band, true);
            fill_path_at(points, closed, fill_rule)
        })
    };
    set(prev.0, prev.1);
    out
}

fn fill_path_at(points: &[PathCmd], closed: bool, fill_rule: FillRule) -> PathFill {
    if points.is_empty() {
        return PathFill::EMPTY;
    }
    // Every subpath is a closed contour to the fill (SVG semantics: a fill
    // implicitly closes an open subpath — Q-51 keeps open line bodies filled).
    // `subpath_to_line_segments` keeps the `closed` distinction only for the
    // strict-equality closing point, which the contour cleanup folds away.
    // Zoomed out (below 1×), a fill's outline also drops vertices the half-pixel tolerance can't see: a path of
    // hundreds of points a few pixels across drew every one of them (a 20k-layer vector document at fit was ~1 GB
    // of vertices, past the 256 MB buffer limit). Curves flatten to half the tolerance and the outline then
    // simplifies to the other half, so the fill stays within the band's half device pixel of the path.
    let simplify = !LEGACY.with(std::cell::Cell::get) && flatten_band() < 0;
    let flat = if simplify {
        Flat {
            dist: flatness_tolerance() * 0.5,
            turn: None,
        }
    } else {
        Flat::fill()
    };
    let mut contours = Vec::new();
    for sp in split_into_subpaths(points).iter().filter(|sp| sp.len() >= 2) {
        let segments = match subpath_to_line_segments(sp, closed, flat) {
            Ok(segments) => segments,
            Err(why) => return PathFill::dropped(why.into()),
        };
        let c = clean_closed_contour(&segments);
        let c = if simplify { simplify_closed_contour(c, flatness_tolerance() * 0.5) } else { c };
        if c.len() >= 3 { contours.push(c); }
    }
    if contours.is_empty() {
        return PathFill::EMPTY;
    }
    match fill_cells(&contours, fill_rule, fill_vertex_cap()) {
        Ok((tris, _)) if tris.is_empty() => PathFill::EMPTY,
        Ok((tris, _)) => PathFill {
            bounds: tris_bounds(&tris),
            tris,
            dropped: None,
        },
        Err(why) => PathFill::dropped(why),
    }
}

/// Runs lyon over already-flattened contours. lyon works in f32; the vertices
/// it emits for input points are those points cast to f32, which is exactly
/// what `combine` casts the stream to, so the fill's outline stays bit-identical
/// to what the ear clipper emitted for the same contour. Intersection points on
/// self-intersecting outlines are the only vertices lyon invents.
///
/// A panic in the sweep is converted to an `Err`: lyon keeps release-mode
/// asserts in its intersection handling, and on the server a panic is a failed
/// export. (`panic = "abort"` builds — wasm — cannot unwind; the non-finite
/// refusal in [`fill_cells`] covers the input class that is known to panic.)
///
/// One sweep, bounded by `cap` vertices; the caller has already checked the
/// sweep's cost ([`check_fill_sweep_cost`]). Fills go through [`fill_cells`].
fn lyon_sweep(
    contours: &[Vec<P>],
    fill_rule: FillRule,
    cap: usize,
    horizontal: bool,
) -> Result<(Vec<f64>, bool), FillRefusal> {
    use lyon_tessellation::math::point;
    use lyon_tessellation::path::PathEvent;
    use lyon_tessellation::{FillOptions, FillRule as LyonFillRule, FillTessellator};

    let rule = match fill_rule {
        FillRule::NonZero => LyonFillRule::NonZero,
        FillRule::EvenOdd => LyonFillRule::EvenOdd,
    };
    let options = FillOptions::DEFAULT
        .with_fill_rule(rule)
        .with_sweep_orientation(if horizontal {
            lyon_tessellation::Orientation::Horizontal
        } else {
            lyon_tessellation::Orientation::Vertical
        });
    let events = contours.iter().flat_map(|c| {
        let pt = |p: &P| point(p[0] as f32, p[1] as f32);
        let first = pt(&c[0]);
        let last = pt(&c[c.len() - 1]);
        std::iter::once(PathEvent::Begin { at: first })
            .chain(c.windows(2).map(move |w| PathEvent::Line {
                from: pt(&w[0]),
                to: pt(&w[1]),
            }))
            .chain(std::iter::once(PathEvent::End {
                last,
                first,
                close: true,
            }))
    });

    // One tessellator per thread: it recycles its event queue between calls,
    // which is most of the per-call cost on the small polygons (rounded rects,
    // stars) the editor re-tessellates during a drag. A failed, capped or
    // panicked sweep leaves it in an unknown state AND holding whatever the
    // sweep grew its queues to, so it is replaced, never reused.
    thread_local! {
        static TESSELLATOR: std::cell::RefCell<FillTessellator> =
            std::cell::RefCell::new(FillTessellator::new());
    }
    let mut builder = BoundedFillBuilder::new(cap);
    let outcome = TESSELLATOR.with(|cell| {
        let mut tess = cell.borrow_mut();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tess.tessellate(events, &options, &mut builder)
        }));
        if !matches!(outcome, Ok(Ok(()))) || builder.over_cap {
            *tess = FillTessellator::new();
        }
        outcome
    });
    FILL_VERTEX_PEAK.with(|c| c.set(c.get().max(builder.peak_vertices)));
    use lyon_tessellation::geometry_builder::GeometryBuilderError;
    use lyon_tessellation::TessellationError;
    match outcome {
        // A sweep that crossed the cap is refused whether lyon aborted at the
        // next vertex or finished first (the last triangles of an outline can
        // arrive after its last vertex).
        Ok(Ok(())) if !builder.over_cap => {}
        Ok(Ok(()))
        | Ok(Err(TessellationError::GeometryBuilder(GeometryBuilderError::TooManyVertices))) => {
            return Err(FillRefusal::Cap)
        }
        Ok(Err(e)) => return Err(FillRefusal::Failed(format!("lyon: {e:?}"))),
        Err(_) => return Err(FillRefusal::Failed("lyon panicked".to_string())),
    }

    let mut tris = Vec::with_capacity(builder.indices.len() * 2);
    for i in &builder.indices {
        let v = builder.vertices[*i as usize];
        tris.push(v.x as f64);
        tris.push(v.y as f64);
    }
    // lyon invents vertices only at crossings. A crossing outline's vertices
    // are placed to ~0.01 units, close enough to a neighbour to fail the
    // check spuriously, and re-cutting a dense crossing mesh would multiply
    // the work; the check is for crossing-free cells, where it is exact.
    let input_vertices: usize = contours.iter().map(Vec::len).sum();
    let sound = builder.vertices.len() > input_vertices
        || !has_vertex_inside_a_triangle(&builder.vertices, &builder.indices);
    Ok((tris, sound))
}

/// One sweep, checked. lyon 1.0.22 mis-triangulates some simple polygons: a
/// y-monotone piece whose bottom is a horizontal edge, with a reflex vertex
/// on one chain, gets a triangle across the reflex notch — paint OUTSIDE the
/// outline (e.g. `[0,0] [6,-7] [11,-7] [1.3,4] [12,18] [7,18] [0,9]`, see
/// `lyon_notch_bug_is_caught`). Such a triangle has an input vertex strictly
/// inside it, which no valid triangulation has, so that is the check. An
/// unsound vertical sweep is retried as a horizontal one (the configuration
/// rotates away); the bool is whether the returned stream passed.
fn sound_sweep(
    contours: &[Vec<P>],
    fill_rule: FillRule,
    cap: usize,
) -> Result<(Vec<f64>, bool), FillRefusal> {
    let vertical = lyon_sweep(contours, fill_rule, cap, false)?;
    if vertical.1 {
        return Ok(vertical);
    }
    match lyon_sweep(contours, fill_rule, cap, true) {
        Ok(horizontal) if horizontal.1 => Ok(horizontal),
        _ => Ok(vertical),
    }
}

/// Whether any vertex lies strictly inside a triangle of the stream (see
/// [`sound_sweep`]). Vertices are bucketed in a uniform grid, so each
/// triangle tests only the vertices under its box; the total work is capped
/// (a pathological layout of long thin triangles reports "sound" rather than
/// stall the fill).
fn has_vertex_inside_a_triangle(
    vertices: &[lyon_tessellation::math::Point],
    indices: &[u32],
) -> bool {
    let n = vertices.len();
    if n < 4 || indices.len() < 3 {
        return false;
    }
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for v in vertices {
        x0 = x0.min(v.x);
        y0 = y0.min(v.y);
        x1 = x1.max(v.x);
        y1 = y1.max(v.y);
    }
    let (w, h) = (
        (x1 - x0).max(f32::MIN_POSITIVE),
        (y1 - y0).max(f32::MIN_POSITIVE),
    );
    // About one vertex per grid cell, in the box's aspect ratio.
    let gx = ((n as f32 * w / h).sqrt().ceil() as usize).clamp(1, 1024);
    let gy = (n.div_ceil(gx)).clamp(1, 1024);
    let col = |x: f32| (((x - x0) / w * gx as f32) as usize).min(gx - 1);
    let row = |y: f32| (((y - y0) / h * gy as f32) as usize).min(gy - 1);
    // Counting sort into cells (CSR).
    let mut start = vec![0u32; gx * gy + 1];
    for v in vertices {
        start[row(v.y) * gx + col(v.x) + 1] += 1;
    }
    for i in 0..gx * gy {
        start[i + 1] += start[i];
    }
    let mut fill = start.clone();
    let mut cell_vertices = vec![0u32; n];
    for (i, v) in vertices.iter().enumerate() {
        let c = row(v.y) * gx + col(v.x);
        cell_vertices[fill[c] as usize] = i as u32;
        fill[c] += 1;
    }
    let mut work_left = 64 * (n + indices.len()) + 100_000;
    for t in indices.chunks_exact(3) {
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| vertices[i as usize]);
        let (ax, ay, bx, by, cx, cy) = (
            a.x as f64, a.y as f64, b.x as f64, b.y as f64, c.x as f64, c.y as f64,
        );
        let area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        if area2 == 0.0 {
            continue;
        }
        // Strictly inside: every barycentric coordinate above 1e-6.
        let eps = 1e-6 * area2.abs();
        let s = area2.signum();
        let (tx0, tx1) = (a.x.min(b.x).min(c.x), a.x.max(b.x).max(c.x));
        let (ty0, ty1) = (a.y.min(b.y).min(c.y), a.y.max(b.y).max(c.y));
        for r in row(ty0)..=row(ty1) {
            for q in col(tx0)..=col(tx1) {
                let cell = r * gx + q;
                if work_left == 0 {
                    return false;
                }
                work_left -= 1;
                for &vi in &cell_vertices[start[cell] as usize..start[cell + 1] as usize] {
                    if work_left == 0 {
                        return false;
                    }
                    work_left -= 1;
                    if t.contains(&vi) {
                        continue;
                    }
                    let (px, py) = (
                        vertices[vi as usize].x as f64,
                        vertices[vi as usize].y as f64,
                    );
                    let d0 = s * ((bx - ax) * (py - ay) - (by - ay) * (px - ax));
                    let d1 = s * ((cx - bx) * (py - by) - (cy - by) * (px - bx));
                    let d2 = s * ((ax - cx) * (py - cy) - (ay - cy) * (px - cx));
                    if d0 > eps && d1 > eps && d2 > eps {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Bound ONE sweep's input size and a conservative active-edge work estimate
/// using only sorted vertical intervals (O(n log n), no pairwise intersection
/// walk). Use f32 coordinates as lyon does, and process starts before ends at
/// ties: horizontal/coincident edges must be charged too. A refusal here is
/// [`FillRefusal::Complexity`]: [`fill_cells`] splits the region and asks
/// again per cell.
fn check_fill_sweep_cost(contours: &[Vec<P>], cap: usize) -> Result<(), FillRefusal> {
    let count: usize = contours.iter().map(Vec::len).sum();
    if count > cap {
        return Err(FillRefusal::Complexity(format!(
            "fill exceeds {cap} input edges"
        )));
    }
    let budget = SWEEP_BUDGET.with(std::cell::Cell::get);
    // Few edges are cheap however they sit: even if every pair crossed (n²/2
    // crossings, each event scanning at most n edges) the sweep does about
    // n³/2 work. Without this, a cell whose edges all share one vertex (a
    // sunburst's hub) or one span would be refused at any split depth.
    if (count as f64).powi(3) <= 8.0 * budget as f64 {
        return Ok(());
    }
    // lyon merges coincident edges into one active edge (summing their
    // winding), so an edge repeated — a cell side that carries the outline's
    // winding number k as k copies — is charged once.
    let mut segments: Vec<[u32; 4]> = Vec::with_capacity(count);
    for contour in contours {
        for i in 0..contour.len() {
            let (p, q) = (contour[i], contour[(i + 1) % contour.len()]);
            let a = [p[1] as f32, p[0] as f32];
            let b = [q[1] as f32, q[0] as f32];
            let (a, b) = if a <= b { (a, b) } else { (b, a) };
            segments.push([
                (a[0] + 0.0).to_bits(),
                (a[1] + 0.0).to_bits(),
                (b[0] + 0.0).to_bits(),
                (b[1] + 0.0).to_bits(),
            ]);
        }
    }
    segments.sort_unstable();
    segments.dedup();
    let mut events = Vec::with_capacity(segments.len() * 2);
    for s in &segments {
        let (a, b) = (f32::from_bits(s[0]), f32::from_bits(s[2]));
        events.push((a, false));
        events.push((b, true));
    }
    events.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut active = 0usize;
    let mut cost = 0usize;
    for (_, end) in events {
        if end {
            active -= 1;
        } else {
            active += 1;
            cost = cost.saturating_add(active.saturating_mul(active));
            if cost > budget {
                return Err(FillRefusal::Complexity(format!(
                    "fill exceeds {budget} sweep complexity budget"
                )));
            }
        }
    }
    Ok(())
}

thread_local! {
    /// The per-sweep work bound in force ([`MAX_FILL_SWEEP_COST`] outside
    /// [`with_fill_sweep_budget`]).
    static SWEEP_BUDGET: std::cell::Cell<usize> = const { std::cell::Cell::new(MAX_FILL_SWEEP_COST) };
}

/// Run `f` with the per-sweep work bound set to `budget`: `usize::MAX` fills
/// every outline in one sweep (the reference the cell tests compare against),
/// a small value forces many cells. Test and bench hook; never set by a
/// render lane.
#[doc(hidden)]
pub fn with_fill_sweep_budget<T>(budget: usize, f: impl FnOnce() -> T) -> T {
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            SWEEP_BUDGET.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(SWEEP_BUDGET.with(|c| c.replace(budget)));
    f()
}

thread_local! {
    /// The fill vertex cap in force ([`MAX_FILL_VERTICES`] outside
    /// [`with_fill_vertex_cap`]).
    static FILL_VERTEX_CAP: std::cell::Cell<usize> = const { std::cell::Cell::new(MAX_FILL_VERTICES) };
}

fn fill_vertex_cap() -> usize {
    FILL_VERTEX_CAP.with(std::cell::Cell::get)
}

/// Run `f` with a path fill's vertex cap set to `cap` (at most
/// [`MAX_FILL_VERTICES`]), so the refusal suites can trip it with outlines of
/// test size. Test hook; never set by a render lane.
#[doc(hidden)]
pub fn with_fill_vertex_cap<T>(cap: usize, f: impl FnOnce() -> T) -> T {
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            FILL_VERTEX_CAP.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(FILL_VERTEX_CAP.with(|c| c.replace(cap.min(MAX_FILL_VERTICES))));
    f()
}

thread_local! {
    static FILL_VERTEX_PEAK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FILL_CELL_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Unique vertices the largest sweep of the most recent fill on this thread
/// emitted before lyon returned (peak, so a capped sweep reports at most the
/// cap). Test hook for the guardrail suite; never read by the renderer.
#[doc(hidden)]
pub fn last_fill_vertex_peak() -> usize {
    FILL_VERTEX_PEAK.with(|c| c.get())
}

/// Sweeps (cells) the most recent fill on this thread ran: 1 for an outline
/// one sweep handles, more when [`fill_cells`] split it. Test hook.
#[doc(hidden)]
pub fn last_fill_cell_count() -> usize {
    FILL_CELL_COUNT.with(|c| c.get())
}

/// lyon geometry sink that refuses the sweep the moment it crosses `cap` on
/// either count: the UNIQUE vertices (`add_fill_vertex`) or the EXPANDED
/// triangle stream (`add_triangle`, three references per triangle — the
/// count the renderer consumes and the TS cap measured). lyon emits both as
/// its sweep processes events, so a self-crossing outline whose intersections
/// multiply (a 10,000-segment star polygon crosses itself tens of millions of
/// times) is stopped inside the sweep; a large SIMPLE outline (a 20,000-point
/// circle) trips the expanded bound first. `add_triangle` cannot return an
/// error, so it records `over_cap`, stops collecting, and the next
/// `add_fill_vertex` returns `GeometryBuilderError::TooManyVertices` (lyon then
/// calls `abort_geometry`); the caller also reads `over_cap` for a sweep that
/// finished before another vertex arrived. Either way the fill is a refusal.
/// This bounds output storage. `check_fill_sweep_cost` separately refuses
/// dense outlines before entering lyon, whose internal work has no budget API.
struct BoundedFillBuilder {
    vertices: Vec<lyon_tessellation::math::Point>,
    indices: Vec<u32>,
    cap: usize,
    peak_vertices: usize,
    /// The expanded stream crossed `cap`; the fill is refused.
    over_cap: bool,
}

impl BoundedFillBuilder {
    fn new(cap: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(64),
            indices: Vec::with_capacity(128),
            cap,
            peak_vertices: 0,
            over_cap: false,
        }
    }
}

impl lyon_tessellation::geometry_builder::GeometryBuilder for BoundedFillBuilder {
    fn begin_geometry(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    fn add_triangle(
        &mut self,
        a: lyon_tessellation::VertexId,
        b: lyon_tessellation::VertexId,
        c: lyon_tessellation::VertexId,
    ) {
        if self.over_cap || self.indices.len() + 3 > self.cap {
            self.over_cap = true;
            return;
        }
        self.indices.push(a.0);
        self.indices.push(b.0);
        self.indices.push(c.0);
    }

    fn abort_geometry(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }
}

impl lyon_tessellation::geometry_builder::FillGeometryBuilder for BoundedFillBuilder {
    fn add_fill_vertex(
        &mut self,
        vertex: lyon_tessellation::FillVertex,
    ) -> Result<
        lyon_tessellation::VertexId,
        lyon_tessellation::geometry_builder::GeometryBuilderError,
    > {
        let id = self.vertices.len();
        if self.over_cap || id >= self.cap {
            return Err(lyon_tessellation::geometry_builder::GeometryBuilderError::TooManyVertices);
        }
        self.vertices.push(vertex.position());
        self.peak_vertices = self.peak_vertices.max(self.vertices.len());
        Ok(lyon_tessellation::VertexId(id as u32))
    }
}

/// `splitIntoSubpaths` (:895-918).
fn split_into_subpaths(points: &[PathCmd]) -> Vec<Vec<PathCmd>> {
    let mut out: Vec<Vec<PathCmd>> = Vec::new();
    let mut cur: Vec<PathCmd> = Vec::new();
    for p in points {
        if matches!(p, PathCmd::Move { .. }) {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        }
        cur.push(p.clone());
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

const INCOMPLETE_PATH: &str = "path flattening exceeds its segment cap (split the path into several)";

/// Flatten one complete subpath, or refuse it. Never expose a partial contour.
fn subpath_to_line_segments(points: &[PathCmd], closed: bool, flat: Flat) -> Result<Vec<P>, &'static str> {
    let mut segs: Vec<P> = Vec::new();
    let mut cur: P = [0.0, 0.0];
    let mut first: P = [0.0, 0.0];
    let mut has_first = false;

    for p in points {
        match p {
            PathCmd::Move { x, y } => {
                cur = [*x, *y];
                if !has_first {
                    first = cur;
                    has_first = true;
                }
                segs.push(cur);
            }
            PathCmd::Line { x, y } => {
                segs.push([*x, *y]);
                cur = [*x, *y];
            }
            PathCmd::Curve { x, y, cp1, cp2 } => {
                if segs.len() > MAX_PATH_SEGMENTS {
                    note_cap_hit();
                    return Err(INCOMPLETE_PATH);
                }
                let pts =
                    tessellate_cubic_bezier(cur, [cp1.0, cp1.1], [cp2.0, cp2.1], [*x, *y], 0, flat);
                for q in pts.iter().skip(1) {
                    if segs.len() >= MAX_PATH_SEGMENTS {
                        note_cap_hit();
                        return Err(INCOMPLETE_PATH);
                    }
                    segs.push(*q);
                }
                cur = [*x, *y];
            }
            PathCmd::Quadratic { x, y, cp } => {
                if segs.len() > MAX_PATH_SEGMENTS {
                    note_cap_hit();
                    return Err(INCOMPLETE_PATH);
                }
                // q→c elevation (:997-1000).
                let cp1 = [
                    cur[0] + (2.0 / 3.0) * (cp.0 - cur[0]),
                    cur[1] + (2.0 / 3.0) * (cp.1 - cur[1]),
                ];
                let cp2 = [
                    *x + (2.0 / 3.0) * (cp.0 - *x),
                    *y + (2.0 / 3.0) * (cp.1 - *y),
                ];
                let pts = tessellate_cubic_bezier(cur, cp1, cp2, [*x, *y], 0, flat);
                for q in pts.iter().skip(1) {
                    if segs.len() >= MAX_PATH_SEGMENTS {
                        note_cap_hit();
                        return Err(INCOMPLETE_PATH);
                    }
                    segs.push(*q);
                }
                cur = [*x, *y];
            }
            PathCmd::Arc {
                x,
                y,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            } => {
                if segs.len() > MAX_PATH_SEGMENTS {
                    note_cap_hit();
                    return Err(INCOMPLETE_PATH);
                }
                let arc = ArcParams {
                    rx: *rx,
                    ry: *ry,
                    x_axis_rotation: *x_axis_rotation,
                    large_arc: *large_arc,
                    sweep: *sweep,
                    x: *x,
                    y: *y,
                };
                let beziers = arc_to_beziers(cur[0], cur[1], &arc);
                if beziers.is_empty() {
                    // Degenerate arc → line to endpoint (:1031).
                    segs.push([*x, *y]);
                } else {
                    let mut bx = cur[0];
                    let mut by = cur[1];
                    for b in &beziers {
                        if segs.len() > MAX_PATH_SEGMENTS {
                            note_cap_hit();
                            return Err(INCOMPLETE_PATH);
                        }
                        let pts = tessellate_cubic_bezier(
                            [bx, by],
                            [b.cp1[0], b.cp1[1]],
                            [b.cp2[0], b.cp2[1]],
                            [b.end[0], b.end[1]],
                            0,
                            flat,
                        );
                        for q in pts.iter().skip(1) {
                            if segs.len() >= MAX_PATH_SEGMENTS {
                                note_cap_hit();
                                return Err(INCOMPLETE_PATH);
                            }
                            segs.push(*q);
                        }
                        bx = b.end[0];
                        by = b.end[1];
                    }
                }
                cur = [*x, *y];
            }
        }
    }

    // Close the subpath if needed (:1066-1074) — strict coordinate equality.
    if closed && segs.len() > 2 && has_first {
        let last = segs[segs.len() - 1];
        if last[0] != first[0] || last[1] != first[1] {
            segs.push(first);
        }
    }
    Ok(segs)
}

/// `tessellateCubicBezier` (:1082-1119) — recursive De Casteljau midpoint
/// subdivision to `flat` (chord distance, and for strokes the piece's turn),
/// depth ≤ 12.
pub(crate) fn tessellate_cubic_bezier(
    p0: P,
    p1: P,
    p2: P,
    p3: P,
    depth: u32,
    flat: Flat,
) -> Vec<P> {
    let mut out = Vec::with_capacity(2);
    out.push(p0);
    append_cubic_endpoints(p0, p1, p2, p3, depth, flat, &mut out);
    out
}

// The left-first subdivision emits one endpoint per leaf. Shared subdivision points
// are already the preceding leaf's endpoint, so each right branch omits its start.
fn append_cubic_endpoints(
    p0: P,
    p1: P,
    p2: P,
    p3: P,
    depth: u32,
    flat: Flat,
    out: &mut Vec<P>,
) {
    if depth >= MAX_RECURSION_DEPTH {
        out.push(p3);
        return;
    }
    if is_curve_flat(p0, p1, p2, p3, flat.dist)
        && flat.turn.is_none_or(|t| control_turn(p0, p1, p2, p3) <= t)
    {
        out.push(p3);
        return;
    }
    let p01 = mid(p0, p1);
    let p12 = mid(p1, p2);
    let p23 = mid(p2, p3);
    let p012 = mid(p01, p12);
    let p123 = mid(p12, p23);
    let p0123 = mid(p012, p123);
    append_cubic_endpoints(p0, p01, p012, p0123, depth + 1, flat, out);
    append_cubic_endpoints(p0123, p123, p23, p3, depth + 1, flat, out);
}

/// Total turn of a cubic's control polygon (an upper bound on its tangent's turn).
fn control_turn(p0: P, p1: P, p2: P, p3: P) -> f64 {
    let dirs: Vec<P> = [
        [p1[0] - p0[0], p1[1] - p0[1]],
        [p2[0] - p1[0], p2[1] - p1[1]],
        [p3[0] - p2[0], p3[1] - p2[1]],
    ]
    .into_iter()
    .filter(|d| d[0].hypot(d[1]) > 1e-12)
    .collect();
    dirs.windows(2)
        .map(|w| {
            (w[0][0] * w[1][1] - w[0][1] * w[1][0])
                .atan2(w[0][0] * w[1][0] + w[0][1] * w[1][1])
                .abs()
        })
        .sum()
}

fn mid(a: P, b: P) -> P {
    [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
}

/// `isCurveFlat` (:1124-1153).
fn is_curve_flat(p0: P, p1: P, p2: P, p3: P, tolerance: f64) -> bool {
    let dx = p3[0] - p0[0];
    let dy = p3[1] - p0[1];
    let d2 = dx * dx + dy * dy;
    if d2 == 0.0 {
        return true;
    }
    let dist = |p: P| -> f64 {
        let t = ((p[0] - p0[0]) * dx + (p[1] - p0[1]) * dy) / d2;
        let ex = p[0] - (p0[0] + t * dx);
        let ey = p[1] - (p0[1] + t * dy);
        (ex * ex + ey * ey).sqrt()
    };
    dist(p1).max(dist(p2)) <= tolerance
}

// --- contour utilities (:665-890) ---

fn points_almost_equal(a: P, b: P) -> bool {
    (a[0] - b[0]).abs() <= POLY_EPS && (a[1] - b[1]).abs() <= POLY_EPS
}

/// `cleanClosedContour` (:720-743).
fn clean_closed_contour(points: &[P]) -> Vec<P> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut p: Vec<P> = points.to_vec();
    if p.len() > 2 && points_almost_equal(p[0], p[p.len() - 1]) {
        p.pop();
    }
    let mut deduped: Vec<P> = Vec::with_capacity(p.len());
    for pt in p {
        if deduped
            .last()
            .map(|prev| !points_almost_equal(*prev, pt))
            .unwrap_or(true)
        {
            deduped.push(pt);
        }
    }
    if deduped.len() > 2 && points_almost_equal(deduped[0], deduped[deduped.len() - 1]) {
        deduped.pop();
    }
    remove_nearly_collinear(deduped)
}

/// A closed contour with the vertices within `tol` of the outline kept dropped (Douglas–Peucker, split at the
/// first vertex and the one farthest from it): every dropped vertex, and so every point of the original outline,
/// stays within `tol` of the simplified one. Contours that would collapse below a triangle keep their points.
fn simplify_closed_contour(points: Vec<P>, tol: f64) -> Vec<P> {
    let n = points.len();
    if n <= 4 {
        return points;
    }
    let d2 = |p: P, q: P| (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
    let far = (1..n).max_by(|&a, &b| d2(points[0], points[a]).total_cmp(&d2(points[0], points[b]))).unwrap_or(n / 2);
    // Distance from `p` to the segment a–b.
    let seg_dist = |p: P, a: P, b: P| -> f64 {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len2 = dx * dx + dy * dy;
        if len2 <= f64::EPSILON {
            return d2(p, a).sqrt();
        }
        let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0);
        d2(p, [a[0] + t * dx, a[1] + t * dy]).sqrt()
    };
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[far] = true;
    // Spans (a, b) of indices around the ring, b possibly past n (wrapping).
    let mut stack: Vec<(usize, usize)> = vec![(0, far), (far, n)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (pa, pb) = (points[a % n], points[b % n]);
        let (mut worst, mut at) = (0.0f64, a);
        for i in a + 1..b {
            let d = seg_dist(points[i % n], pa, pb);
            if d > worst {
                worst = d;
                at = i;
            }
        }
        if worst > tol {
            keep[at % n] = true;
            stack.push((a, at));
            stack.push((at, b));
        }
    }
    let out: Vec<P> = points.iter().zip(&keep).filter(|(_, k)| **k).map(|(p, _)| *p).collect();
    if out.len() >= 3 {
        out
    } else {
        points
    }
}

/// A polyline with the vertices within `tol` of it dropped (Douglas–Peucker; the first and last points stay, so a
/// closed ring that repeats its first point keeps its closure).
fn simplify_polyline(points: Vec<P>, tol: f64) -> Vec<P> {
    let n = points.len();
    if n <= 3 {
        return points;
    }
    let seg_dist = |p: P, a: P, b: P| -> f64 {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len2 = dx * dx + dy * dy;
        let t = if len2 <= f64::EPSILON {
            0.0
        } else {
            (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
        };
        ((p[0] - a[0] - t * dx).powi(2) + (p[1] - a[1] - t * dy).powi(2)).sqrt()
    };
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (mut worst, mut at) = (0.0f64, a);
        for i in a + 1..b {
            let d = seg_dist(points[i], points[a], points[b]);
            if d > worst {
                worst = d;
                at = i;
            }
        }
        if worst > tol {
            keep[at] = true;
            stack.push((a, at));
            stack.push((at, b));
        }
    }
    let out: Vec<P> = points.iter().zip(&keep).filter(|(_, k)| **k).map(|(p, _)| *p).collect();
    // A ring that would collapse keeps its points.
    if out.len() >= 3 || points[0] != points[n - 1] {
        out
    } else {
        points
    }
}

/// `removeNearlyCollinear` (:745-766).
fn remove_nearly_collinear(points: Vec<P>) -> Vec<P> {
    if points.len() < 4 {
        return points;
    }
    let mut result = points;
    let mut i = 0usize;
    while result.len() >= 4 && i < result.len() {
        let n = result.len();
        let prev = result[(i + n - 1) % n];
        let curr = result[i];
        let next = result[(i + 1) % n];
        let area2 =
            (curr[0] - prev[0]) * (next[1] - prev[1]) - (curr[1] - prev[1]) * (next[0] - prev[0]);
        if area2.abs() <= POLY_EPS {
            result.remove(i);
            continue;
        }
        i += 1;
    }
    result
}

/// `contourArea` (:768-775) — signed shoelace / 2.
fn contour_area(points: &[P]) -> f64 {
    let mut area = 0.0;
    for i in 0..points.len() {
        let j = (i + 1) % points.len();
        area += points[i][0] * points[j][1] - points[j][0] * points[i][1];
    }
    area / 2.0
}

fn union_bounds(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

fn calc_bounds(points: &[P]) -> [f64; 4] {
    if points.is_empty() {
        return [0.0; 4];
    }
    let mut b = [points[0][0], points[0][1], points[0][0], points[0][1]];
    for p in points {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    b
}

// ---------------------------------------------------------------------------
// Stroke (:1393-1911)
// ---------------------------------------------------------------------------

/// `generateStrokeGeometry` (:1393-1595). Returns (triangle stream, bounds).
fn generate_stroke_geometry(
    points: &[PathCmd],
    stroke_width: f64,
    closed: bool,
    line_cap: LineCap,
    line_join: LineJoin,
    stroke_align: StrokeAlign,
) -> (Vec<f64>, [f64; 4]) {
    if points.is_empty() || stroke_width <= 0.0 {
        return (Vec::new(), [0.0; 4]);
    }
    let mut tris: Vec<f64> = Vec::new();
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    let subpaths = split_into_subpaths(points);

    for sp in &subpaths {
        if sp.len() < 2 {
            continue;
        }
        // The flattening must hold at the stroke's outer offset: half the
        // width when centred, the whole width when inside / outside aligned.
        let reach = match stroke_align {
            StrokeAlign::Center => stroke_width / 2.0,
            _ => stroke_width,
        };
        // Zoomed out, a stroke no wider than the tolerance (no turn limit) drops the centreline vertices the half
        // pixel can't see, as fills do (`fill_path_at`): curves flatten to half the tolerance, the polyline
        // simplifies to the other half.
        let simplify = !LEGACY.with(std::cell::Cell::get)
            && flatten_band() < 0
            && Flat::stroke(reach).turn.is_none();
        let segments = if simplify {
            let half_tol = flatness_tolerance() * 0.5;
            simplify_polyline(
                match subpath_to_line_segments(sp, closed, Flat { dist: half_tol, turn: None }) {
                    Ok(s) => s,
                    Err(_) => return (Vec::new(), [0.0; 4]),
                },
                half_tol,
            )
        } else {
            match subpath_to_line_segments(sp, closed, Flat::stroke(reach)) {
                Ok(s) => s,
                Err(_) => return (Vec::new(), [0.0; 4]),
            }
        };
        if segments.len() < 2 {
            continue;
        }
        let mut effective: Vec<P> = segments.clone();
        let mut is_closed_loop = false;
        if closed && segments.len() > 2 {
            let first = segments[0];
            let last = segments[segments.len() - 1];
            if first[0] == last[0] && first[1] == last[1] {
                effective.pop();
                is_closed_loop = true;
            }
        }

        // Align as asymmetric side offsets (:1444-1476); closed contours only.
        let half = stroke_width / 2.0;
        let mut left_off = half;
        let mut right_off = half;
        if is_closed_loop && stroke_align != StrokeAlign::Center {
            let exterior_is_left = contour_area(&effective) < 0.0;
            let band_on_left = if stroke_align == StrokeAlign::Outside {
                exterior_is_left
            } else {
                !exterior_is_left
            };
            left_off = if band_on_left { stroke_width } else { 0.0 };
            right_off = if band_on_left { 0.0 } else { stroke_width };
        }

        let (left, right) =
            generate_stroke_offsets(&effective, left_off, right_off, is_closed_loop, line_join);
        if left.len() < 2 || right.len() < 2 {
            continue;
        }
        let mut left = left;
        let mut right = right;
        if !is_closed_loop && line_cap == LineCap::Square {
            apply_square_caps(&mut left, &mut right, &effective, half);
        }

        let segment_count = left.len();
        for i in 0..segment_count {
            let next_i = if i + 1 >= segment_count {
                if is_closed_loop {
                    0
                } else {
                    break;
                }
            } else {
                i + 1
            };
            tris.extend_from_slice(&[
                left[i][0],
                left[i][1],
                right[i][0],
                right[i][1],
                left[next_i][0],
                left[next_i][1],
            ]);
            tris.extend_from_slice(&[
                right[i][0],
                right[i][1],
                right[next_i][0],
                right[next_i][1],
                left[next_i][0],
                left[next_i][1],
            ]);
        }

        let mut bounds_points: Vec<P> = left.clone();
        bounds_points.extend(right.iter().cloned());

        // Round caps (:1520-1569) — 10-segment fans at 1×, finer as the zoom band needs (`round_arc_segments`).
        if !is_closed_loop && line_cap == LineCap::Round {
            let cap_segments = 10usize;
            if let Some(start_t) = endpoint_tangent(&effective, true) {
                let center = effective[0];
                let outward = [-start_t[0], -start_t[1]];
                let arc = round_cap_arc(center, left[0], right[0], outward, half, cap_segments);
                if arc.len() >= 2 {
                    bounds_points.extend(arc.iter().cloned());
                    for i in 0..arc.len() - 1 {
                        tris.extend_from_slice(&[
                            center[0],
                            center[1],
                            arc[i][0],
                            arc[i][1],
                            arc[i + 1][0],
                            arc[i + 1][1],
                        ]);
                    }
                }
            }
            if let Some(end_t) = endpoint_tangent(&effective, false) {
                let center = effective[effective.len() - 1];
                let last = left.len() - 1;
                let arc = round_cap_arc(center, left[last], right[last], end_t, half, cap_segments);
                if arc.len() >= 2 {
                    bounds_points.extend(arc.iter().cloned());
                    for i in 0..arc.len() - 1 {
                        tris.extend_from_slice(&[
                            center[0],
                            center[1],
                            arc[i][0],
                            arc[i][1],
                            arc[i + 1][0],
                            arc[i + 1][1],
                        ]);
                    }
                }
            }
        }
        bounds = union_bounds(bounds, calc_bounds(&bounds_points));
    }

    if tris.is_empty() {
        return (Vec::new(), [0.0; 4]);
    }
    (tris, bounds)
}

/// `getEndpointTangent` (:1614-1639).
fn endpoint_tangent(segments: &[P], start: bool) -> Option<P> {
    if segments.len() < 2 {
        return None;
    }
    if start {
        let p0 = segments[0];
        for s in segments.iter().skip(1) {
            let dx = s[0] - p0[0];
            let dy = s[1] - p0[1];
            let len = dx.hypot(dy);
            if len > EPS {
                return Some([dx / len, dy / len]);
            }
        }
        None
    } else {
        let pn = segments[segments.len() - 1];
        for s in segments.iter().rev().skip(1) {
            let dx = pn[0] - s[0];
            let dy = pn[1] - s[1];
            let len = dx.hypot(dy);
            if len > EPS {
                return Some([dx / len, dy / len]);
            }
        }
        None
    }
}

/// `applySquareCaps` (:1641-1669).
fn apply_square_caps(left: &mut [P], right: &mut [P], segments: &[P], offset: f64) {
    if left.is_empty() || right.is_empty() {
        return;
    }
    if let Some(t) = endpoint_tangent(segments, true) {
        let (sx, sy) = (-t[0] * offset, -t[1] * offset);
        left[0][0] += sx;
        left[0][1] += sy;
        right[0][0] += sx;
        right[0][1] += sy;
    }
    if let Some(t) = endpoint_tangent(segments, false) {
        let (ex, ey) = (t[0] * offset, t[1] * offset);
        let li = left.len() - 1;
        left[li][0] += ex;
        left[li][1] += ey;
        let ri = right.len() - 1;
        right[ri][0] += ex;
        right[ri][1] += ey;
    }
}

/// `generateRoundCapArcPoints` (:1671-1701).
fn round_cap_arc(center: P, a: P, b: P, outward: P, radius: f64, segments: usize) -> Vec<P> {
    let start_angle = (a[1] - center[1]).atan2(a[0] - center[0]);
    let end_angle = (b[1] - center[1]).atan2(b[0] - center[0]);
    let mut delta = normalize_angle(end_angle - start_angle);
    if delta.abs() < EPS {
        return Vec::new();
    }
    let mid_angle = start_angle + delta * 0.5;
    let mid_dot = mid_angle.cos() * outward[0] + mid_angle.sin() * outward[1];
    if mid_dot < 0.0 {
        let two_pi = std::f64::consts::PI * 2.0;
        delta = if delta > 0.0 {
            delta - two_pi
        } else {
            delta + two_pi
        };
    }
    let segments = round_arc_segments(radius, delta, segments);
    let mut pts = Vec::with_capacity(segments + 1);
    for i in 0..=segments {
        let t = i as f64 / segments as f64;
        let angle = start_angle + delta * t;
        pts.push([
            center[0] + angle.cos() * radius,
            center[1] + angle.sin() * radius,
        ]);
    }
    pts
}

/// `normalizeAngle` (:1600-1606).
fn normalize_angle(angle: f64) -> f64 {
    let two_pi = std::f64::consts::PI * 2.0;
    let mut a = angle;
    while a <= -std::f64::consts::PI {
        a += two_pi;
    }
    while a > std::f64::consts::PI {
        a -= two_pi;
    }
    a
}

/// `getRoundJoinSegments` (:1608-1612) — ~22.5°/segment, 4..16.
fn round_join_segments(delta: f64) -> usize {
    let abs = delta.abs();
    let ceil = (abs / (std::f64::consts::PI / 8.0)).ceil();
    (ceil.max(4.0).min(16.0)) as usize
}

/// Most segments one round cap or join takes (a 2048-unit radius at the finest band stays within it).
const MAX_ARC_SEGMENTS: usize = 4096;

/// Segments for a round cap or join of `radius` turning `sweep` radians: at least `legacy` (the TS
/// port's fixed count, what 1× thin strokes draw), and enough that each chord sags at most a quarter
/// of the band's flatness tolerance — a quarter of a device pixel or less anywhere in the zoom band,
/// so a cap or join stays round at any zoom or export scale.
fn round_arc_segments(radius: f64, sweep: f64, legacy: usize) -> usize {
    if LEGACY.with(std::cell::Cell::get) || !(radius > 0.0) || !sweep.is_finite() {
        return legacy;
    }
    let tol = flatness_tolerance() * 0.25;
    if radius <= tol {
        return legacy;
    }
    let step = 2.0 * (1.0 - tol / radius).acos();
    let n = (sweep.abs() / step).ceil();
    if !n.is_finite() {
        return legacy;
    }
    (n as usize).clamp(legacy, MAX_ARC_SEGMENTS.max(legacy))
}

/// `padPoints` (:1703-1709).
fn pad_points(points: &mut Vec<P>, target: usize) {
    if points.is_empty() {
        return;
    }
    let last = *points.last().unwrap();
    while points.len() < target {
        points.push(last);
    }
}

/// `generateStrokeOffsets` (:1711-1818).
fn generate_stroke_offsets(
    segments: &[P],
    left_offset: f64,
    right_offset: f64,
    closed: bool,
    line_join: LineJoin,
) -> (Vec<P>, Vec<P>) {
    let mut left: Vec<P> = Vec::new();
    let mut right: Vec<P> = Vec::new();
    if segments.is_empty() {
        return (left, right);
    }
    let n = segments.len();
    for i in 0..n {
        let curr = segments[i];
        if !closed && (i == 0 || i == n - 1) {
            match endpoint_tangent(segments, i == 0) {
                Some(t) => {
                    let (nx, ny) = (-t[1], t[0]);
                    left.push([curr[0] + nx * left_offset, curr[1] + ny * left_offset]);
                    right.push([curr[0] - nx * right_offset, curr[1] - ny * right_offset]);
                }
                None => {
                    left.push(curr);
                    right.push(curr);
                }
            }
            continue;
        }

        let mut prev = segments[(i + n - 1) % n];
        let mut next = segments[(i + 1) % n];
        let mut dx1 = curr[0] - prev[0];
        let mut dy1 = curr[1] - prev[1];
        let mut len1 = dx1.hypot(dy1);
        let mut dx2 = next[0] - curr[0];
        let mut dy2 = next[1] - curr[1];
        let mut len2 = dx2.hypot(dy2);

        // Zero-length lookback/lookahead (:1757-1779) — `(i - j + n) % n`
        // with j in 2..=n, stopping when the index wraps to i.
        if len1 <= EPS {
            // JS `(i - j + n) % n` for 2 <= j <= n: i + n >= j always holds.
            let mut j = 2usize;
            while j <= n && len1 <= EPS {
                let lb = (i + n - j) % n;
                if lb == i {
                    break;
                }
                prev = segments[lb];
                dx1 = curr[0] - prev[0];
                dy1 = curr[1] - prev[1];
                len1 = dx1.hypot(dy1);
                j += 1;
            }
        }
        if len2 <= EPS {
            let mut j = 2usize;
            while j <= n && len2 <= EPS {
                let la = (i + j) % n;
                if la == i {
                    break;
                }
                next = segments[la];
                dx2 = next[0] - curr[0];
                dy2 = next[1] - curr[1];
                len2 = dx2.hypot(dy2);
                j += 1;
            }
        }
        if len1 <= EPS && len2 <= EPS {
            left.push(curr);
            right.push(curr);
            continue;
        }

        let dir1 = if len1 > EPS {
            [dx1 / len1, dy1 / len1]
        } else {
            [dx2 / len2, dy2 / len2]
        };
        let dir2 = if len2 > EPS {
            [dx2 / len2, dy2 / len2]
        } else {
            dir1
        };
        let cross = dir1[0] * dir2[1] - dir1[1] * dir2[0];

        // Collapsed-side shortcut (:1795-1807).
        let mut left_join = if left_offset <= 0.0 {
            vec![curr]
        } else {
            compute_join_points(
                curr,
                dir1,
                dir2,
                left_offset,
                true,
                line_join,
                cross,
                len1.min(len2),
            )
        };
        let mut right_join = if right_offset <= 0.0 {
            vec![curr]
        } else {
            compute_join_points(
                curr,
                dir1,
                dir2,
                right_offset,
                false,
                line_join,
                cross,
                len1.min(len2),
            )
        };
        let target = left_join.len().max(right_join.len());
        pad_points(&mut left_join, target);
        pad_points(&mut right_join, target);
        left.extend(left_join);
        right.extend(right_join);
    }
    (left, right)
}

/// `computeJoinPoints` (:1820-1910).
#[allow(clippy::too_many_arguments)]
fn compute_join_points(
    curr: P,
    dir1: P,
    dir2: P,
    offset: f64,
    left_side: bool,
    line_join: LineJoin,
    cross: f64,
    // The shorter adjacent leg: how far along a leg the inner crossing may lie.
    leg_room: f64,
) -> Vec<P> {
    let m = if left_side { 1.0 } else { -1.0 };
    let n1 = [-dir1[1] * m, dir1[0] * m];
    let n2 = [-dir2[1] * m, dir2[0] * m];
    let p1: P = [curr[0] + n1[0] * offset, curr[1] + n1[1] * offset];
    let p2: P = [curr[0] + n2[0] * offset, curr[1] + n2[1] * offset];

    let is_convex = if left_side { cross < -EPS } else { cross > EPS };

    let mut mx = n1[0] + n2[0];
    let mut my = n1[1] + n2[1];
    let mlen = mx.hypot(my);
    if mlen <= EPS {
        return vec![p1];
    }
    mx /= mlen;
    my /= mlen;

    let dot = dir1[0] * dir2[0] + dir1[1] * dir2[1];
    let denom = ((1.0 + dot) / 2.0).max(0.0).sqrt();
    let raw_miter = if denom > EPS {
        offset / denom
    } else {
        f64::INFINITY
    };
    let max_miter = offset * MITER_LIMIT_NOW.with(std::cell::Cell::get);
    let exceeds = raw_miter > max_miter;
    let clamped = raw_miter.min(max_miter);
    let miter_point: P = [curr[0] + mx * clamped, curr[1] + my * clamped];

    // Inner side of a turn: the two legs' offsets cross at the UNCLAMPED
    // miter distance (the miter limit only concerns the outer tip). Use that
    // crossing while it lies on both legs; past a leg's end, route through the
    // vertex (the pivot) — the legs' own quads then cover the corner. The old
    // clamp to 4 × offset left a notch at sharp inner corners.
    if !is_convex {
        let along = (raw_miter * raw_miter - offset * offset).max(0.0).sqrt();
        if raw_miter.is_finite() && along <= leg_room + EPS {
            return vec![[curr[0] + mx * raw_miter, curr[1] + my * raw_miter]];
        }
        // Known limit: past a short leg the legs' own quads overlap near the
        // vertex (a 20° apex with 12-unit legs at width 14 double-covers ~60 %
        // of the join area) — as the old clamped inner point did. Opaque
        // strokes are unaffected; a translucent one can darken there.
        return vec![p1, curr, p2];
    }

    match line_join {
        LineJoin::Miter => {
            if exceeds {
                vec![p1, p2]
            } else {
                vec![miter_point]
            }
        }
        LineJoin::Bevel => vec![p1, p2],
        LineJoin::Round => {
            let start_angle = n1[1].atan2(n1[0]);
            let end_angle = n2[1].atan2(n2[0]);
            let delta = normalize_angle(end_angle - start_angle);
            if delta.abs() < EPS {
                return vec![p1];
            }
            let segs = round_arc_segments(offset.abs(), delta, round_join_segments(delta));
            let mut pts = Vec::with_capacity(segs + 1);
            for i in 0..=segs {
                let t = i as f64 / segs as f64;
                let angle = start_angle + delta * t;
                pts.push([
                    curr[0] + angle.cos() * offset,
                    curr[1] + angle.sin() * offset,
                ]);
            }
            pts
        }
    }
}

#[cfg(test)]
mod trim_tests {
    use super::*;

    fn line(x: f64, y: f64) -> PathCmd {
        PathCmd::Line { x, y }
    }
    fn mv(x: f64, y: f64) -> PathCmd {
        PathCmd::Move { x, y }
    }
    fn xy(c: &PathCmd) -> (f64, f64) {
        match c {
            PathCmd::Move { x, y } | PathCmd::Line { x, y } => (*x, *y),
            PathCmd::Quadratic { x, y, .. } | PathCmd::Curve { x, y, .. } => (*x, *y),
            PathCmd::Arc { x, y, .. } => (*x, *y),
        }
    }
    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 0.01, "expected {b}, got {a}");
    }
    fn t(start: f64, end: f64, offset: f64, individual: bool) -> TrimParams {
        TrimParams {
            start,
            end,
            offset,
            individual,
            by: TrimBy::Length,
        }
    }
    fn count_moves(cmds: &[PathCmd]) -> usize {
        cmds.iter()
            .filter(|c| matches!(c, PathCmd::Move { .. }))
            .count()
    }

    #[test]
    fn identity_guard() {
        assert!(is_trim_identity(0.0, 1.0, 0.0));
        // A full [0,1] window is identity for ANY offset (rotating a full loop is
        // a no-op) — this prevents an offset-only closed shape degrading to open.
        assert!(is_trim_identity(0.0, 1.0, 2.0));
        assert!(is_trim_identity(0.0, 1.0, 0.25));
        assert!(!is_trim_identity(0.0, 0.5, 0.0));
        assert!(!is_trim_identity(0.1, 1.0, 0.0));
    }

    #[test]
    fn open_line_first_half() {
        let pts = vec![mv(0.0, 0.0), line(100.0, 0.0)];
        let out = apply_trim_to_path(&pts, false, t(0.0, 0.5, 0.0, false));
        assert_eq!(out.len(), 2);
        approx(xy(&out[0]).0, 0.0);
        approx(xy(&out[1]).0, 50.0);
    }

    #[test]
    fn open_line_middle() {
        let pts = vec![mv(0.0, 0.0), line(100.0, 0.0)];
        let out = apply_trim_to_path(&pts, false, t(0.25, 0.75, 0.0, false));
        approx(xy(&out[0]).0, 25.0);
        approx(xy(&out[1]).0, 75.0);
    }

    #[test]
    fn open_line_start_after_end_is_empty() {
        let pts = vec![mv(0.0, 0.0), line(100.0, 0.0)];
        let out = apply_trim_to_path(&pts, false, t(0.75, 0.25, 0.0, false));
        assert!(out.is_empty());
    }

    #[test]
    fn open_line_offset_wraps_instead_of_vanishing() {
        // A partial window on an OPEN line, offset past the end, must WRAP (march
        // back to the start) rather than clamp to empty and vanish (review MB-1).
        // [0.6,1.0] + 0.5 → [1.1,1.5] mod 1 = [0.1,0.5] → visible run at x∈[10,50].
        let pts = vec![mv(0.0, 0.0), line(100.0, 0.0)];
        let out = apply_trim_to_path(&pts, false, t(0.6, 1.0, 0.5, false));
        assert!(
            !out.is_empty(),
            "offset must not make an open trimmed line vanish"
        );
        approx(xy(&out[0]).0, 10.0);
        approx(xy(&out[1]).0, 50.0);
    }

    #[test]
    fn open_line_offset_shifts_window() {
        let pts = vec![mv(0.0, 0.0), line(100.0, 0.0)];
        let out = apply_trim_to_path(&pts, false, t(0.0, 0.5, 0.25, false));
        approx(xy(&out[0]).0, 25.0);
        approx(xy(&out[1]).0, 75.0);
    }

    #[test]
    fn open_line_simultaneous_offset_over_one_normalizes() {
        // Offset 1.25 must behave like 0.25 (rem_euclid), not push the window off
        // the open path and vanish (Gemini review, high).
        let pts = vec![mv(0.0, 0.0), line(100.0, 0.0)];
        let base = apply_trim_to_path(&pts, false, t(0.0, 0.5, 0.25, false));
        let wrapped = apply_trim_to_path(&pts, false, t(0.0, 0.5, 1.25, false));
        assert!(
            !wrapped.is_empty(),
            "offset 1.25 must not empty an open path"
        );
        approx(xy(&wrapped[0]).0, xy(&base[0]).0);
        approx(xy(&wrapped[1]).0, xy(&base[1]).0);
    }

    #[test]
    fn closed_square_wraps_across_seam() {
        // Perimeter 400; window [0.75, 0.25] wraps the seam. The two halves are
        // contiguous at the seam, so they emit as ONE continuous sub-path (single
        // Move) — no dash phase reset / double caps at a visually-continuous point.
        let pts = vec![
            mv(0.0, 0.0),
            line(100.0, 0.0),
            line(100.0, 100.0),
            line(0.0, 100.0),
        ];
        let out = apply_trim_to_path(&pts, true, t(0.75, 0.25, 0.0, false));
        assert_eq!(
            count_moves(&out),
            1,
            "wrapping window is one continuous sub-path"
        );
    }

    #[test]
    fn quadratic_segment_is_preserved_not_flattened() {
        let pts = vec![
            mv(0.0, 0.0),
            PathCmd::Quadratic {
                x: 100.0,
                y: 0.0,
                cp: (50.0, 50.0),
            },
        ];
        let out = apply_trim_to_path(&pts, false, t(0.0, 0.5, 0.0, false));
        assert!(
            matches!(out.get(1), Some(PathCmd::Quadratic { .. })),
            "trim must keep curves"
        );
    }

    #[test]
    fn multi_subpath_simultaneous_trims_each() {
        let pts = vec![
            mv(0.0, 0.0),
            line(100.0, 0.0),
            mv(0.0, 50.0),
            line(100.0, 50.0),
        ];
        let out = apply_trim_to_path(&pts, false, t(0.0, 0.5, 0.0, false));
        assert_eq!(count_moves(&out), 2, "both sub-paths reveal together");
        // each kept run ends at x=50
        approx(xy(&out[1]).0, 50.0);
        approx(xy(&out[3]).0, 50.0);
    }

    #[test]
    fn multi_subpath_individual_reveals_first_then_second() {
        let pts = vec![
            mv(0.0, 0.0),
            line(100.0, 0.0),
            mv(0.0, 50.0),
            line(100.0, 50.0),
        ];
        // Global window [0, 0.5] of total(200) = [0,100] → only the first line.
        let out = apply_trim_to_path(&pts, false, t(0.0, 0.5, 0.0, true));
        assert_eq!(count_moves(&out), 1, "only the first sub-path is revealed");
        approx(xy(&out[0]).1, 0.0);
        approx(xy(&out[1]).0, 100.0);
    }

    #[test]
    fn multi_subpath_individual_offset_normalizes_modulo_one() {
        // An out-of-range offset (3.0) must behave like its fractional part (0.0),
        // not shift the window off the concatenation (regression, review pass 1).
        let pts = vec![
            mv(0.0, 0.0),
            line(100.0, 0.0),
            mv(0.0, 50.0),
            line(100.0, 50.0),
        ];
        let base = apply_trim_to_path(&pts, false, t(0.0, 0.5, 0.0, true));
        let wrapped = apply_trim_to_path(&pts, false, t(0.0, 0.5, 3.0, true));
        assert_eq!(count_moves(&wrapped), count_moves(&base));
        assert!(!wrapped.is_empty(), "offset 3.0 must not drop all geometry");
    }

    #[test]
    fn aligned_dashes_turn_every_corner_of_a_closed_square() {
        // A 100 square, width 2 (the pattern unscaled), 20 on / 10 off. Exact: the dashes fall where the pattern
        // takes them, so some corners sit in gaps. Aligned: every corner is an interior vertex of one dash, and the
        // start corner's dash runs through the seam.
        let sq = vec![mv(0.0, 0.0), line(100.0, 0.0), line(100.0, 100.0), line(0.0, 100.0)];
        let corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let pieces = |cmds: &[PathCmd]| -> Vec<Vec<(f64, f64)>> {
            let mut out: Vec<Vec<(f64, f64)>> = Vec::new();
            for c in cmds {
                match c {
                    PathCmd::Move { .. } => out.push(vec![xy(c)]),
                    _ => out.last_mut().expect("a Move first").push(xy(c)),
                }
            }
            out
        };
        let near = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6;
        let interior = |ps: &[Vec<(f64, f64)>], c: (f64, f64)| {
            ps.iter().any(|p| p.len() >= 3 && p[1..p.len() - 1].iter().any(|v| near(*v, c)))
        };
        let exact = pieces(&apply_dash_to_path(&sq, &[20.0, 10.0], 2.0, true, 0.0).unwrap().open);
        assert!(!corners.iter().all(|c| interior(&exact, *c)), "exact dashes miss a corner");
        let aligned = with_dash_alignment(true, || apply_dash_to_path(&sq, &[20.0, 10.0], 2.0, true, 0.0).unwrap());
        assert!(aligned.closed.is_empty());
        let aligned = pieces(&aligned.open);
        for c in corners {
            assert!(interior(&aligned, c), "no dash turns {c:?}: {aligned:?}");
        }
        // 3 repeats per side: 4 corner dashes + 2 more per side.
        assert_eq!(aligned.len(), 12, "{aligned:?}");
        // Each corner dash has equal arms (10 × 100/90).
        let arm = 1000.0 / 90.0;
        let seam = aligned.iter().find(|p| p.iter().any(|v| near(*v, (0.0, 0.0)))).expect("the seam dash");
        assert!(near(seam[0], (0.0, arm)) && near(seam[seam.len() - 1], (arm, 0.0)), "{seam:?}");
        // The setting is per call: it is off again afterwards.
        assert!(!dash_alignment_now());
    }

    #[test]
    fn aligned_dashes_on_an_open_path_are_centred_on_its_ends_and_corners() {
        // An open L (50 + 50), 10 on / 10 off: each arm fits 2.5 repeats → rounded to 3 (squeezed by 50/60).
        let l = vec![mv(0.0, 0.0), line(50.0, 0.0), line(50.0, 50.0)];
        let d = with_dash_alignment(true, || apply_dash_to_path(&l, &[10.0, 10.0], 2.0, false, 0.0).unwrap()).open;
        let k = 50.0 / 60.0;
        let first = xy(&d[1]);
        assert!(xy(&d[0]) == (0.0, 0.0) && (first.0 - 5.0 * k).abs() < 1e-6, "half a dash from the start: {d:?}");
        let last = xy(&d[d.len() - 1]);
        assert!((last.0 - 50.0).abs() < 1e-6 && (last.1 - 50.0).abs() < 1e-6, "ends on the end: {d:?}");
        assert!(d.iter().any(|c| matches!(c, PathCmd::Line { x, y } if (*x - 50.0).abs() < 1e-9 && y.abs() < 1e-9)), "the corner is inside a dash");
    }

    #[test]
    fn trim_then_dash_confines_dashes_to_the_trimmed_window() {
        // Trim [0,0.5] of a 100px line, then dash the survivor — dashes must stay
        // within [0,50] (proves trim runs BEFORE dash, per the pipeline order).
        let pts = vec![mv(0.0, 0.0), line(100.0, 0.0)];
        let trimmed = apply_trim_to_path(&pts, false, t(0.0, 0.5, 0.0, false));
        let dashed = apply_dash_to_path(&trimmed, &[10.0, 10.0], 2.0, false, 0.0).unwrap().open;
        assert!(
            !dashed.is_empty(),
            "trimmed+dashed line should still draw dashes"
        );
        for c in &dashed {
            assert!(
                xy(c).0 <= 50.0 + 0.01,
                "dash escaped the trimmed window at x={}",
                xy(c).0
            );
        }
    }

    #[test]
    fn dash_restart_per_subpath_does_not_bridge_the_gap() {
        // Two sub-paths with a spatial gap (x 20..100), like a connector body split
        // around a text label. A dash long enough to span the boundary must NOT be
        // carried across (which would draw a line through the gap).
        let pts = vec![
            mv(0.0, 0.0),
            line(20.0, 0.0),
            mv(100.0, 0.0),
            line(120.0, 0.0),
        ];
        let dash = [30.0, 10.0]; // dash 30 spans the 20-long first sub-path
                                 // BASE_DASH_STROKE_WIDTH = 2, so stroke_width 2 keeps the pattern unscaled.
        let out = apply_dash_to_path(&pts, &dash, 2.0, false, 0.0).unwrap().open;
        assert_eq!(count_moves(&out), 2, "each sub-path starts its own dash");
        // No emitted vertex lands in the gap (x strictly between 20 and 100).
        for c in &out {
            let x = xy(c).0;
            assert!(
                !(x > 20.0 + 0.01 && x < 100.0 - 0.01),
                "dash bridged the gap at x={x}"
            );
        }
    }

    #[test]
    fn trim_by_x_ends_at_the_window_edge_whatever_the_segment_lengths() {
        // A tall spike then a flat run: by length, half the stroke is still on the spike's way up; by x it has reached
        // x = 150 on the flat run.
        let pts = vec![mv(0.0, 0.0), line(50.0, -1000.0), line(100.0, 0.0), line(300.0, 0.0)];
        let by_x = TrimParams { by: TrimBy::X, ..t(0.0, 0.5, 0.0, false) };
        let out = apply_trim_to_path(&pts, false, by_x);
        assert_eq!(count_moves(&out), 1);
        approx(xy(out.last().unwrap()).0, 150.0);
        approx(xy(out.last().unwrap()).1, 0.0);
        let by_len = apply_trim_to_path(&pts, false, t(0.0, 0.5, 0.0, false));
        assert!(xy(by_len.last().unwrap()).0 < 100.0, "by length it is still on the spike");
        // Start and end both cut: the window [75, 150] keeps the spike's way down from x = 75.
        let mid = apply_trim_to_path(&pts, false, TrimParams { by: TrimBy::X, ..t(0.25, 0.5, 0.0, false) });
        approx(xy(&mid[0]).0, 75.0);
        approx(xy(&mid[0]).1, -500.0);
        // By y the window is the path's height: the top quarter is the spike's tip.
        let tip = apply_trim_to_path(&pts, false, TrimParams { by: TrimBy::Y, ..t(0.0, 0.25, 0.0, false) });
        assert_eq!(count_moves(&tip), 1);
        approx(xy(&tip[0]).1, -1000.0 + 250.0);
        // Offset doesn't move an axis window.
        let offset = apply_trim_to_path(&pts, false, TrimParams { by: TrimBy::X, ..t(0.0, 0.5, 0.4, false) });
        assert_eq!(offset, out);
    }

    #[test]
    fn trim_clamps_overshoot_to_unit_range() {
        // A keyframe/ease can overshoot [0,1] (e.g. elastic → end 1.5, start -0.5);
        // the tessellator clamps so it behaves like the clamped values.
        let pts = vec![
            mv(0.0, 0.0),
            line(100.0, 0.0),
            line(100.0, 100.0),
            line(0.0, 100.0),
        ];
        let overshoot = apply_trim_to_path(&pts, true, t(-0.5, 1.5, 0.0, false));
        let clamped = apply_trim_to_path(&pts, true, t(0.0, 1.0, 0.0, false));
        assert_eq!(
            overshoot.len(),
            clamped.len(),
            "out-of-range trim must clamp to [0,1]"
        );
    }
}

#[cfg(test)]
mod stroke_cap_tests {
    use super::*;

    #[test]
    fn incomplete_curves_are_refused_at_every_zoom() {
        for band in [0, 3] {
            let _scale = super::super::outline::enter_plan(2.0f64.powi(band));
            for curve in [
                PathCmd::Curve { x: 100.0, y: 100.0, cp1: (100.0, 30.0), cp2: (100.0, 70.0) },
                PathCmd::Quadratic { x: 100.0, y: 100.0, cp: (100.0, 50.0) },
                PathCmd::Arc { x: 100.0, y: 100.0, rx: 50.0, ry: 50.0, x_axis_rotation: 0.0, large_arc: false, sweep: true },
            ] {
                let mut points = vec![PathCmd::Move { x: 0.0, y: 0.0 }];
                points.extend((1..=MAX_PATH_SEGMENTS).map(|i| PathCmd::Line { x: 100.0 * i as f64 / MAX_PATH_SEGMENTS as f64, y: 0.0 }));
                points.push(curve);
                points.push(PathCmd::Line { x: 0.0, y: 100.0 });
                CAP_HIT.with(|c| c.set(false));
                let fill = fill_path(&points, true, FillRule::NonZero);
                assert!(fill.dropped.is_some(), "band {band}: partial fill accepted");
                assert!(fill.tris.is_empty());
                CAP_HIT.with(|c| c.set(false));
                let geometry = tessellate_combined(&points, 1.0, true, LineCap::Butt, LineJoin::Miter, FillRule::NonZero, StrokeAlign::Center);
                assert!(geometry.fill_dropped.is_some());
                assert!(geometry.stroke_dropped.is_some(), "band {band}: partial stroke accepted");
            }
        }
    }

    #[test]
    fn long_complete_line_contours_are_preserved() {
        let mut points = vec![PathCmd::Move { x: 0.0, y: 0.0 }];
        points.extend((1..=MAX_PATH_SEGMENTS).map(|i| PathCmd::Line { x: 100.0 * i as f64 / MAX_PATH_SEGMENTS as f64, y: 0.0 }));
        points.extend([PathCmd::Line { x: 100.0, y: 100.0 }, PathCmd::Line { x: 0.0, y: 100.0 }]);
        let fill = fill_path(&points, true, FillRule::NonZero);
        assert!(fill.dropped.is_none());
        assert!(!fill.tris.is_empty());
    }

    /// A stroke over the stream cap is refused with a reason (reported as a plan warning), not emptied silently;
    /// one at the cap is kept whole.
    #[test]
    fn a_stroke_over_the_stream_cap_is_refused_with_a_reason() {
        let at_cap = (vec![0.0; MAX_STROKE_STREAM_VERTICES * 2], [0.0; 4]);
        let mut why = None;
        assert!(filter_stroke(Ok(at_cap), &mut why).is_some());
        assert!(why.is_none());
        let over = (vec![0.0; (MAX_STROKE_STREAM_VERTICES + 3) * 2], [0.0; 4]);
        assert!(filter_stroke(Ok(over), &mut why).is_none());
        let why = why.expect("a refused stroke carries its reason");
        assert!(why.contains("vertices"), "{why}");
        let o = combine(Vec::new(), [0.0; 4], None, None, Some(why));
        assert!(o.stroke_dropped.is_some());
    }
}

#[cfg(test)]
mod cap_parity_tests {
    /// The boolean solver refuses a result longer than this cap. The
    /// tessellator used to slice a path to its first `MAX_PATH_SEGMENTS`
    /// commands, silently; it no longer does (the cap bounds one subpath's
    /// flattening), but the two stay in step so a boolean result never holds a
    /// subpath the flattening would cut. moda-engine depends on moda-scene, so
    /// the assertion lives on this side.
    #[test]
    fn boolean_result_cap_matches_the_path_segment_cap() {
        assert_eq!(
            moda_scene::path_bool::MAX_RESULT_POINTS,
            super::MAX_PATH_SEGMENTS,
            "path_bool::MAX_RESULT_POINTS and tess MAX_PATH_SEGMENTS must move together"
        );
    }
}

#[cfg(test)]
mod banded_fill_tests {
    use super::*;

    fn area(tris: &[f64]) -> f64 {
        tris.chunks_exact(6)
            .map(|t| ((t[2] - t[0]) * (t[5] - t[1]) - (t[4] - t[0]) * (t[3] - t[1])).abs() * 0.5)
            .sum()
    }

    /// One sweep, whatever it costs: the reference cells are compared with.
    fn whole(contours: &[Vec<P>], rule: FillRule) -> Vec<f64> {
        let (tris, cells) = with_fill_sweep_budget(usize::MAX, || {
            fill_cells(contours, rule, MAX_STROKE_VERTICES)
        })
        .unwrap();
        assert_eq!(cells, 1);
        tris
    }

    /// Winding number of `p` around the contours (exact, f64).
    fn winding(contours: &[Vec<P>], p: P) -> i32 {
        let mut w = 0;
        for c in contours {
            for i in 0..c.len() {
                let (a, b) = (c[i], c[(i + 1) % c.len()]);
                let side = (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1]);
                if a[1] <= p[1] && b[1] > p[1] && side > 0.0 {
                    w += 1;
                } else if a[1] > p[1] && b[1] <= p[1] && side < 0.0 {
                    w -= 1;
                }
            }
        }
        w
    }

    /// Distance from `p` to the nearest contour edge.
    fn edge_distance(contours: &[Vec<P>], p: P) -> f64 {
        let mut d = f64::INFINITY;
        for c in contours {
            for i in 0..c.len() {
                let (a, b) = (c[i], c[(i + 1) % c.len()]);
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                let l2 = dx * dx + dy * dy;
                let t = if l2 > 0.0 {
                    (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                d = d.min((p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy));
            }
        }
        d
    }

    /// How many triangles of the stream contain `p`.
    fn coverage(tris: &[f64], p: P) -> usize {
        tris.chunks_exact(6)
            .filter(|t| {
                let s = |ax: f64, ay: f64, bx: f64, by: f64| {
                    (bx - ax) * (p[1] - ay) - (p[0] - ax) * (by - ay)
                };
                let (d0, d1, d2) = (
                    s(t[0], t[1], t[2], t[3]),
                    s(t[2], t[3], t[4], t[5]),
                    s(t[4], t[5], t[0], t[1]),
                );
                (d0 > 0.0 && d1 > 0.0 && d2 > 0.0) || (d0 < 0.0 && d1 < 0.0 && d2 < 0.0)
            })
            .count()
    }

    /// Deterministic self-crossing polygons (plus a hole-like second
    /// contour): forced into many cells, the stream covers every sample
    /// exactly as the fill rule says — once inside, never outside, never
    /// twice — and its area equals one sweep's.
    #[test]
    fn cells_resolve_both_fill_rules_exactly() {
        let mut seed = 0x2545_f491_u64;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for case in 0..12 {
            let n = 30 + case * 5;
            let mut contours = vec![(0..n)
                .map(|_| [rnd() * 200.0, rnd() * 120.0])
                .collect::<Vec<P>>()];
            contours.push(
                (0..8)
                    .map(|k| {
                        let a = k as f64 * std::f64::consts::TAU / 8.0;
                        [100.0 + 30.0 * a.cos(), 60.0 - 30.0 * a.sin()]
                    })
                    .collect(),
            );
            for rule in [FillRule::NonZero, FillRule::EvenOdd] {
                let reference = whole(&contours, rule);
                let (tris, cells) = with_fill_sweep_budget(1_000, || {
                    fill_cells(&contours, rule, MAX_STROKE_VERTICES)
                })
                .unwrap();
                assert!(cells > 1, "case {case}: forced budget must split ({cells})");
                let (a, b) = (area(&reference), area(&tris));
                assert!(
                    (a - b).abs() <= 1e-4 * a.max(1.0),
                    "case {case} {rule:?}: one sweep {a} vs {cells} cells {b}"
                );
                for i in 0..80 {
                    for j in 0..48 {
                        let p = [i as f64 * 2.5 + 0.617, j as f64 * 2.5 + 0.389];
                        if edge_distance(&contours, p) < 1e-3 {
                            continue;
                        }
                        let w = winding(&contours, p);
                        let inside = match rule {
                            FillRule::NonZero => w != 0,
                            FillRule::EvenOdd => w % 2 != 0,
                        };
                        assert_eq!(
                            coverage(&tris, p),
                            inside as usize,
                            "case {case} {rule:?} at {p:?} (winding {w})"
                        );
                    }
                }
            }
        }
    }

    /// A dense scribble is over one sweep's bound; it now bisects instead of
    /// being refused.
    #[test]
    fn dense_scribble_bisects_instead_of_refusing() {
        let mut c = Vec::new();
        for i in 0..6000 {
            let t = i as f64;
            c.push([
                300.0 + 280.0 * (t * 0.05).sin(),
                300.0 + 20.0 * (t * 0.3).cos() + (t * 0.001),
            ]);
        }
        let contours = vec![c];
        assert!(matches!(
            check_fill_sweep_cost(&contours, MAX_STROKE_VERTICES),
            Err(FillRefusal::Complexity(_))
        ));
        let (_, _, cells) = fill_outline_banded(&contours).unwrap();
        assert!(cells > 1);
    }

    /// A lyon 1.0.22 mis-triangulation (see `sound_sweep`): a chevron whose
    /// bottom edge is horizontal gets a triangle across its notch. The check
    /// flags it, and the fill that ships covers exactly the polygon.
    #[test]
    fn lyon_notch_bug_is_caught() {
        let chevron: Vec<P> = vec![
            [0.0, 0.0],
            [6.0, -7.0],
            [11.0, -7.0],
            [1.3, 4.0],
            [12.0, 18.0],
            [7.0, 18.0],
            [0.0, 9.0],
        ];
        let polygon = {
            let mut a = 0.0;
            for i in 0..chevron.len() {
                let (p, q) = (chevron[i], chevron[(i + 1) % chevron.len()]);
                a += p[0] * q[1] - q[0] * p[1];
            }
            a.abs() * 0.5
        };
        let contours = vec![chevron];
        for rule in [FillRule::NonZero, FillRule::EvenOdd] {
            let (raw, sound) = lyon_sweep(&contours, rule, MAX_FILL_VERTICES, false).unwrap();
            assert!(!sound, "the check flags lyon's triangulation");
            assert!(area(&raw) > polygon + 1.0, "lyon paints outside the notch");
            let (tris, _) = fill_cells(&contours, rule, MAX_FILL_VERTICES).unwrap();
            let a = area(&tris);
            assert!((a - polygon).abs() < 1e-3, "{rule:?}: {a} vs {polygon}");
        }
    }

    /// The vertex cap is shared by all cells: a field of pentagrams (each
    /// crossing itself five times) splits into many cells that each fit, and
    /// together they run out the cap — the fill is refused with its reason.
    #[test]
    fn the_vertex_cap_spans_every_cell() {
        let stars: Vec<Vec<P>> = (0..80 * 60)
            .map(|k| {
                let (cx, cy) = ((k % 80) as f64 * 10.0, (k / 80) as f64 * 10.0);
                (0..5)
                    .map(|i| {
                        let a = (i * 2 % 5) as f64 * std::f64::consts::TAU / 5.0;
                        [cx + 4.0 * a.cos(), cy + 4.0 * a.sin()]
                    })
                    .collect()
            })
            .collect();
        let cap = 50_000;
        let err = fill_cells(&stars, FillRule::NonZero, cap).unwrap_err();
        assert_eq!(err, format!("fill exceeds {cap} vertices"));
        assert!(last_fill_vertex_peak() <= cap);
    }
}

#[cfg(test)]
mod band_cap_tests {
    use super::*;

    fn fill_at(cmds: &[PathCmd], scale: f64) -> OracleGeometry {
        let _g = super::super::outline::enter_plan(scale);
        tessellate_combined_oracle(
            cmds,
            0.0,
            true,
            None,
            None,
            0.0,
            LineCap::Butt,
            LineJoin::Miter,
            FillRule::NonZero,
            StrokeAlign::Center,
        )
    }

    /// Curved fills refine per band above 1× (art-brush artwork is a fill),
    /// coarsen per band below it, and are exactly the pre-band flattening at 1×.
    #[test]
    fn curved_fills_refine_per_band() {
        let blob = [
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Curve {
                x: 200.0,
                y: 0.0,
                cp1: (0.0, 150.0),
                cp2: (200.0, 150.0),
            },
            PathCmd::Curve {
                x: 0.0,
                y: 0.0,
                cp1: (200.0, -150.0),
                cp2: (0.0, -150.0),
            },
        ];
        let n = |scale: f64| fill_at(&blob, scale).positions.len();
        assert!(n(4.0) > n(1.0), "{} vs {}", n(4.0), n(1.0));
        assert!(
            n(0.25) < n(1.0),
            "coarser below 1×: {} vs {}",
            n(0.25),
            n(1.0)
        );
        let legacy = {
            LEGACY.with(|k| k.set(true));
            let g = fill_at(&blob, 1.0).positions;
            LEGACY.with(|k| k.set(false));
            g
        };
        assert_eq!(
            fill_at(&blob, 1.0).positions,
            legacy,
            "1× = the pre-band flattening"
        );
    }

    /// A fill whose band flattening would exceed the fill caps falls back to a
    /// coarser band — never dropped or truncated because of the zoom.
    #[test]
    fn huge_curved_fills_fall_back_instead_of_dropping() {
        // A tall strip with 1000 shallow waves down its left edge (vertical, so
        // the sweep's active set stays small; no self-overlap): ~4000 segments
        // at 1× (as on main), over MAX_PATH_SEGMENTS at 64×.
        let mut c = vec![PathCmd::Move { x: 0.0, y: 0.0 }];
        for i in 0..1000 {
            let y0 = i as f64 * 10.0;
            c.push(PathCmd::Curve {
                x: 0.0,
                y: y0 + 10.0,
                cp1: (5.0, y0 + 3.0),
                cp2: (-5.0, y0 + 7.0),
            });
        }
        c.push(PathCmd::Line {
            x: 200.0,
            y: 10_000.0,
        });
        c.push(PathCmd::Line { x: 200.0, y: 0.0 });
        let one = fill_at(&c, 1.0);
        assert!(
            one.fill_dropped.is_none() && !one.positions.is_empty(),
            "{:?}",
            one.fill_dropped
        );
        let g = fill_at(&c, 64.0);
        assert!(g.fill_dropped.is_none(), "{:?}", g.fill_dropped);
        assert!(g.positions.len() >= one.positions.len());
    }

    /// `n` S-curves 10 units apart, 60 tall.
    fn s_curves(n: usize) -> Vec<PathCmd> {
        let mut v = vec![PathCmd::Move { x: 0.0, y: 0.0 }];
        for i in 0..n {
            let x0 = i as f64 * 10.0;
            v.push(PathCmd::Curve {
                x: x0 + 10.0,
                y: 0.0,
                cp1: (x0 + 2.0, 60.0),
                cp2: (x0 + 8.0, -60.0),
            });
        }
        v
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        cmds: &[PathCmd],
        scale: f64,
        width: f64,
        closed: bool,
        dash: Option<&[f64]>,
        cap: LineCap,
        join: LineJoin,
        align: StrokeAlign,
    ) -> (usize, f64, u32) {
        let _g = super::super::outline::enter_plan(scale);
        let g = tessellate_combined_oracle(
            cmds,
            width,
            closed,
            None,
            dash,
            0.0,
            cap,
            join,
            FillRule::NonZero,
            align,
        );
        let (n, x) = stroke_extent(&g);
        (n, x, ATTEMPTS.with(std::cell::Cell::get))
    }

    /// Review repro 1: a wide curved stroke main draws must never be dropped —
    /// the last resort is the pre-band flattening without the turn limit.
    #[test]
    fn wide_curved_strokes_main_draws_are_never_dropped() {
        let c = s_curves(400);
        let dash = [4.0, 2.0];
        for scale in [1.0, 4.0, 64.0] {
            let (n, x, _) = run(
                &c,
                scale,
                10.0,
                false,
                Some(&dash),
                LineCap::Round,
                LineJoin::Round,
                StrokeAlign::Center,
            );
            assert!(
                n > 0 && x > 3990.0,
                "dashed at {scale}×: {n} vertices to x {x}"
            );
            let (n, x, _) = run(
                &c,
                scale,
                10.0,
                true,
                None,
                LineCap::Round,
                LineJoin::Round,
                StrokeAlign::Inside,
            );
            assert!(
                n > 0 && x > 3990.0,
                "inside, closed at {scale}×: {n} vertices to x {x}"
            );
        }
    }

    /// Review repro 2: dash gaps are not stroke vertices — sparse dashes on a
    /// long curved path run to its end at 1×.
    #[test]
    fn sparse_dashes_on_long_curves_reach_the_end() {
        let c = s_curves(3000);
        let dash = [1.0, 200.0];
        let (n, x, _) = run(
            &c,
            1.0,
            2.0,
            false,
            Some(&dash),
            LineCap::Butt,
            LineJoin::Miter,
            StrokeAlign::Center,
        );
        assert!(n > 0 && x > 29_000.0, "{n} vertices, last dash at x {x}");
    }

    /// Review repro 3: a stroke far over the caps at 64× starts at the band its
    /// estimated size fits, instead of retrying every band in one frame.
    #[test]
    fn capped_strokes_jump_to_a_band_that_fits() {
        let c = s_curves(400);
        let (n, x, attempts) = run(
            &c,
            64.0,
            10.0,
            false,
            None,
            LineCap::Butt,
            LineJoin::Miter,
            StrokeAlign::Center,
        );
        assert!(n > 0 && x > 3995.0);
        assert!(attempts <= 2, "{attempts} attempts");
    }

    /// The largest sag (device px at `scale`) of the chords through the stroke vertices lying on the circle of
    /// `radius` about `c`: the flattening of a round cap or join there.
    fn arc_sag_px(g: &OracleGeometry, c: (f64, f64), radius: f64, scale: f64) -> (usize, f64) {
        let mut angles: Vec<f64> = g
            .positions
            .chunks(2)
            .zip(&g.vertex_types)
            .filter(|(_, t)| **t == 1.0)
            .map(|(p, _)| (p[0] as f64 - c.0, p[1] as f64 - c.1))
            .filter(|(dx, dy)| (dx.hypot(*dy) - radius).abs() < 1e-3)
            .map(|(dx, dy)| dy.atan2(dx))
            .collect();
        angles.sort_by(f64::total_cmp);
        angles.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
        if angles.len() < 2 {
            return (angles.len(), f64::INFINITY);
        }
        let mut gaps: Vec<f64> = angles.windows(2).map(|w| w[1] - w[0]).collect();
        gaps.push(angles[0] + std::f64::consts::TAU - angles[angles.len() - 1]);
        gaps.sort_by(f64::total_cmp);
        gaps.pop(); // the side the arc doesn't cover
        let widest = gaps.last().copied().unwrap_or(0.0);
        (angles.len(), radius * (1.0 - (widest / 2.0).cos()) * scale)
    }

    /// Round caps, round joins and round dash caps stay round at any zoom: at 1×, 16× and 64× each arc's
    /// chords sag at most a quarter of a device pixel (they were fixed 10-segment fans and 22.5° joins,
    /// faceted from about 8× up). 1× thin strokes keep the TS port's counts.
    #[test]
    fn round_caps_and_joins_stay_round_at_64x() {
        let open = |pts: &[(f64, f64)]| -> Vec<PathCmd> {
            pts.iter()
                .enumerate()
                .map(|(i, &(x, y))| if i == 0 { PathCmd::Move { x, y } } else { PathCmd::Line { x, y } })
                .collect()
        };
        let at = |cmds: &[PathCmd], scale: f64, cap: LineCap, join: LineJoin, dash: Option<&[f64]>| {
            let _g = super::super::outline::enter_plan(scale);
            tessellate_combined_oracle(
                cmds,
                16.0,
                false,
                None,
                dash,
                0.0,
                cap,
                join,
                FillRule::NonZero,
                StrokeAlign::Center,
            )
        };
        let line = open(&[(-60.0, 12.0), (12.0, 12.0)]);
        let vee = open(&[(-30.0, 40.0), (12.0, 10.0), (54.0, 40.0)]);
        // Dashes from x = 0 (the pattern scales with the width: this first dash is 32 units long): its start
        // cap is centred there.
        let dashed = open(&[(0.0, 12.0), (100.0, 12.0)]);
        for scale in [1.0, 16.0, 64.0] {
            let cap = at(&line, scale, LineCap::Round, LineJoin::Miter, None);
            let join = at(&vee, scale, LineCap::Butt, LineJoin::Round, None);
            let dash = at(&dashed, scale, LineCap::Round, LineJoin::Miter, Some(&[4.0, 400.0]));
            for (what, g, c) in [
                ("cap", &cap, (12.0, 12.0)),
                ("join", &join, (12.0, 10.0)),
                ("dash cap", &dash, (0.0, 12.0)),
            ] {
                let (n, sag) = arc_sag_px(g, c, 8.0, scale);
                assert!(n >= 5, "{what} at {scale}×: only {n} vertices on the arc");
                if scale > 1.0 {
                    assert!(sag <= 0.25, "{what} at {scale}×: chords sag {sag:.3} px");
                }
            }
        }
        // 1×: the TS port's fan (11 points on a cap of this size).
        assert_eq!(arc_sag_px(&at(&line, 1.0, LineCap::Round, LineJoin::Miter, None), (12.0, 12.0), 8.0, 1.0).0, 11);
    }

    /// 400 tall S-waves, 10 units apart: fine at 1×, far over the segment
    /// and vertex caps at the 64× band's flattening.
    fn waves() -> Vec<PathCmd> {
        let mut v = vec![PathCmd::Move { x: 0.0, y: 0.0 }];
        for i in 0..400 {
            let x0 = i as f64 * 10.0;
            v.push(PathCmd::Curve {
                x: x0 + 10.0,
                y: 0.0,
                cp1: (x0 + 2.0, 60.0),
                cp2: (x0 + 8.0, -60.0),
            });
        }
        v
    }

    fn stroke_at(scale: f64, dash: Option<&[f64]>) -> OracleGeometry {
        let _g = super::super::outline::enter_plan(scale);
        tessellate_combined_oracle(
            &waves(),
            10.0,
            false,
            None,
            dash,
            0.0,
            LineCap::Butt,
            LineJoin::Miter,
            FillRule::NonZero,
            StrokeAlign::Center,
        )
    }

    fn stroke_extent(g: &OracleGeometry) -> (usize, f64) {
        let mut n = 0;
        let mut max_x = f64::NEG_INFINITY;
        for (i, t) in g.vertex_types.iter().enumerate() {
            if *t == 1.0 {
                n += 1;
                max_x = max_x.max(g.positions[i * 2] as f64);
            }
        }
        (n, max_x)
    }

    #[test]
    fn a_huge_curved_stroke_at_64x_falls_back_coarser_instead_of_truncating() {
        let (n1, x1) = stroke_extent(&stroke_at(1.0, None));
        let (n64, x64) = stroke_extent(&stroke_at(64.0, None));
        assert!(n1 > 0 && x1 > 3995.0, "1×: {n1} vertices to x {x1}");
        assert!(n64 > 0, "64× dropped the stroke");
        assert!(x64 > 3995.0, "64× truncated the path at x {x64}");
        assert!(n64 <= MAX_STROKE_VERTICES);
        // It still refines as far as the caps allow.
        assert!(n64 >= n1, "{n64} < {n1}");
    }

    #[test]
    fn a_huge_dashed_curved_stroke_at_64x_still_draws() {
        let dash = [4.0, 2.0];
        let (n1, x1) = stroke_extent(&stroke_at(1.0, Some(&dash)));
        let (n64, x64) = stroke_extent(&stroke_at(64.0, Some(&dash)));
        assert!(n1 > 0 && x1 > 3990.0, "1×: {n1} to {x1}");
        assert!(n64 > 0, "64× dashed stroke vanished");
        assert!(x64 > 3990.0, "64× dashes stop at x {x64}");
    }

    #[test]
    fn small_curves_still_refine_per_band() {
        let c = [
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Curve {
                x: 100.0,
                y: 0.0,
                cp1: (0.0, 80.0),
                cp2: (100.0, 80.0),
            },
        ];
        let n = |scale: f64| {
            let _g = super::super::outline::enter_plan(scale);
            let g = tessellate_combined_oracle(
                &c,
                4.0,
                false,
                None,
                None,
                0.0,
                LineCap::Butt,
                LineJoin::Miter,
                FillRule::NonZero,
                StrokeAlign::Center,
            );
            stroke_extent(&g).0
        };
        assert!(n(8.0) > n(1.0), "{} vs {}", n(8.0), n(1.0));
    }
}

#[cfg(test)]
mod join_tests {
    use super::*;

    #[test]
    fn miter_limit_is_restored_after_a_panic() {
        let r = std::panic::catch_unwind(|| with_miter_limit(40.0, || panic!("boom")));
        assert!(r.is_err());
        assert_eq!(MITER_LIMIT_NOW.with(std::cell::Cell::get), MITER_LIMIT);
    }

    fn v(apex_deg: f64) -> Vec<PathCmd> {
        let h = 110.0;
        let dx = h * (apex_deg.to_radians() / 2.0).tan();
        vec![
            PathCmd::Move {
                x: 120.0 - dx,
                y: 140.0,
            },
            PathCmd::Line { x: 120.0, y: 30.0 },
            PathCmd::Line {
                x: 120.0 + dx,
                y: 140.0,
            },
        ]
    }

    fn top(limit: f64) -> f64 {
        let g = with_miter_limit(limit, || {
            tessellate_combined(
                &v(20.0),
                14.0,
                false,
                LineCap::Butt,
                LineJoin::Miter,
                FillRule::NonZero,
                StrokeAlign::Center,
            )
        });
        let mut min_y = f64::INFINITY;
        for (i, t) in g.vertex_types.iter().enumerate() {
            if *t == 1.0 {
                min_y = min_y.min(g.positions[i * 2 + 1] as f64);
            }
        }
        min_y
    }

    #[test]
    fn miter_limit_is_per_node() {
        // Apex 20°: miter ratio 1/sin(10°) = 5.76 — bevelled at the default 4,
        // mitered at 10 (the tip reaches 7 × 5.76 above the vertex).
        let bevel = top(4.0);
        let miter = top(10.0);
        assert!(
            bevel > 20.0,
            "bevelled: nothing far above the vertex ({bevel})"
        );
        assert!(
            (miter - (30.0 - 7.0 / (10f64).to_radians().sin())).abs() < 0.01,
            "miter tip {miter}"
        );
    }

    #[test]
    fn sharp_inner_corner_fills_to_the_offset_crossing() {
        // The inner offsets of a 20° V (half-width 7) cross 7 / sin 10° below
        // the apex; the stroke must fill every point above that on the axis.
        let g = tessellate_combined(
            &v(20.0),
            14.0,
            false,
            LineCap::Butt,
            LineJoin::Miter,
            FillRule::NonZero,
            StrokeAlign::Center,
        );
        let crossing = 30.0 + 7.0 / (10f64).to_radians().sin();
        let inside = |x: f64, y: f64| -> bool {
            g.positions
                .chunks(6)
                .zip(g.vertex_types.chunks(3))
                .any(|(t, ty)| {
                    if ty[0] != 1.0 {
                        return false;
                    }
                    let (a, b, c) = (
                        [t[0] as f64, t[1] as f64],
                        [t[2] as f64, t[3] as f64],
                        [t[4] as f64, t[5] as f64],
                    );
                    let s = |p: [f64; 2], q: [f64; 2]| {
                        (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0])
                    };
                    let (d1, d2, d3) = (s(a, b), s(b, c), s(c, a));
                    !((d1 < 0.0 || d2 < 0.0 || d3 < 0.0) && (d1 > 0.0 || d2 > 0.0 || d3 > 0.0))
                })
        };
        for k in 1..20 {
            let y = 30.0 + (crossing - 30.0) * k as f64 / 20.0 - 0.01;
            assert!(
                inside(120.0, y),
                "axis point at y={y} (crossing {crossing}) is not filled"
            );
        }
        assert!(
            !inside(120.0, crossing + 1.0),
            "below the crossing is the V's inside"
        );
    }
}

/// F4.1: zoom-adaptive flattening. The flattened curve stays within half a
/// device pixel of the true curve at every band, and its size follows the
/// scale (about √8 more segments per 8× zoom), clamped at the band limits.
#[cfg(test)]
mod zoom_adaptive_flattening_tests {
    use super::super::outline::{enter_plan, MAX_BAND, MIN_BAND};
    use super::*;

    const KAPPA: f64 = 0.552_284_749_830_793_6;

    /// A circle of radius `r` as four cubics, and an S-bend of the same size
    /// (curvature that changes along the curve).
    fn shapes(r: f64) -> Vec<Vec<PathCmd>> {
        let k = r * KAPPA;
        let circle = vec![
            PathCmd::Move { x: r, y: 0.0 },
            PathCmd::Curve {
                x: 0.0,
                y: r,
                cp1: (r, k),
                cp2: (k, r),
            },
            PathCmd::Curve {
                x: -r,
                y: 0.0,
                cp1: (-k, r),
                cp2: (-r, k),
            },
            PathCmd::Curve {
                x: 0.0,
                y: -r,
                cp1: (-r, -k),
                cp2: (-k, -r),
            },
            PathCmd::Curve {
                x: r,
                y: 0.0,
                cp1: (k, -r),
                cp2: (r, -k),
            },
        ];
        let s_bend = vec![
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Curve {
                x: 2.0 * r,
                y: 0.0,
                cp1: (0.3 * r, -1.6 * r),
                cp2: (1.7 * r, 1.6 * r),
            },
        ];
        vec![circle, s_bend]
    }

    fn cubic_at(p0: P, p1: P, p2: P, p3: P, t: f64) -> P {
        let u = 1.0 - t;
        let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
        [
            a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
            a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
        ]
    }

    fn dist_to_segment(p: P, a: P, b: P) -> f64 {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len2 = dx * dx + dy * dy;
        let t = if len2 > 0.0 {
            (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy)
    }

    /// Largest distance (canvas units) from the true curve to the flattened
    /// polyline, over dense samples of every cubic. The polyline follows the
    /// curve in order, so each sample is measured against the pieces around
    /// the last one it matched (linear cost).
    fn max_error(cmds: &[PathCmd], poly: &[P]) -> f64 {
        let mut samples: Vec<P> = Vec::new();
        let mut cur: P = [0.0, 0.0];
        for c in cmds {
            match *c {
                PathCmd::Move { x, y } => cur = [x, y],
                PathCmd::Curve { x, y, cp1, cp2 } => {
                    let (p1, p2, p3) = ([cp1.0, cp1.1], [cp2.0, cp2.1], [x, y]);
                    for i in 0..=4000 {
                        samples.push(cubic_at(cur, p1, p2, p3, i as f64 / 4000.0));
                    }
                    cur = p3;
                }
                _ => unreachable!("test shapes are cubics"),
            }
        }
        let mut seg = 0usize;
        let mut worst = 0.0f64;
        for s in samples {
            let d = |i: usize| dist_to_segment(s, poly[i], poly[i + 1]);
            while seg + 2 < poly.len() && d(seg + 1) <= d(seg) {
                seg += 1;
            }
            worst = worst.max(d(seg));
        }
        worst
    }

    fn flatten_at(cmds: &[PathCmd], scale: f64, flat: impl Fn() -> Flat) -> Vec<P> {
        let _g = enter_plan(scale);
        subpath_to_line_segments(cmds, false, flat()).expect("complete test outline")
    }

    #[test]
    fn error_stays_under_half_a_device_pixel_at_1x_8x_64x_and_512x() {
        // 512 device px per canvas unit is the editor's top zoom (25600%) on a 2× display.
        for r in [20.0, 100.0, 1000.0] {
            for (i, cmds) in shapes(r).iter().enumerate() {
                for scale in [1.0, 8.0, 64.0, 512.0] {
                    for (kind, poly) in [
                        ("fill", flatten_at(cmds, scale, Flat::fill)),
                        ("stroke", flatten_at(cmds, scale, || Flat::stroke(r * 0.1))),
                    ] {
                        let px = max_error(cmds, &poly) * scale;
                        assert!(
                            px <= 0.5 + 1e-6,
                            "r {r} shape {i} {kind} at {scale}×: {px:.4} device px ({} points)",
                            poly.len()
                        );
                    }
                }
                // The check has teeth: the fixed 1× chord seen at 64× is
                // several pixels off (the facets F4.1 removes).
                let fixed = flatten_at(cmds, 1.0, Flat::legacy);
                let px = max_error(cmds, &fixed) * 64.0;
                assert!(
                    px > 2.0,
                    "r {r} shape {i}: 1× chord at 64× is only {px:.3} px off"
                );
            }
        }
    }

    #[test]
    fn zoomed_out_error_stays_under_half_a_device_pixel() {
        // Below 1× the tolerance grows with the band (fewer pieces), and the
        // error in device pixels stays within the same half pixel.
        for cmds in shapes(400.0) {
            for scale in [0.5, 0.25, 1.0 / 16.0] {
                let poly = flatten_at(&cmds, scale, Flat::fill);
                let px = max_error(&cmds, &poly) * scale;
                assert!(px <= 0.5 + 1e-6, "{scale}×: {px:.4} device px");
            }
        }
    }

    /// A dense outline (300 line points around a wobbly circle, like a traced vector icon) zoomed out: its fill and
    /// its hairline stroke drop the points the half pixel can't see, and the simplified outline stays within half a
    /// device pixel of every original point.
    #[test]
    fn zoomed_out_dense_outlines_drop_invisible_vertices() {
        let n = 300;
        let pts: Vec<P> = (0..n)
            .map(|i| {
                let a = i as f64 / n as f64 * std::f64::consts::TAU;
                let r = 40.0 + 0.6 * (a * 23.0).sin();
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        let cmds: Vec<PathCmd> = pts
            .iter()
            .enumerate()
            .map(|(i, p)| match i {
                0 => PathCmd::Move { x: p[0], y: p[1] },
                _ => PathCmd::Line { x: p[0], y: p[1] },
            })
            .collect();
        let draw = |scale: f64, stroke: f64| {
            let _g = enter_plan(scale);
            tessellate_combined_oracle(
                &cmds,
                stroke,
                true,
                None,
                None,
                0.0,
                LineCap::Butt,
                LineJoin::Miter,
                FillRule::NonZero,
                StrokeAlign::Center,
            )
            .positions
            .len()
        };
        for stroke in [0.0, 1.0] {
            let (at1, at8th) = (draw(1.0, stroke), draw(0.125, stroke));
            assert!(at8th * 4 < at1, "stroke {stroke}: {at8th} positions at 1/8× vs {at1} at 1×");
        }
        for scale in [0.5, 0.125, 1.0 / 16.0] {
            let _g = enter_plan(scale);
            let tol = flatness_tolerance() * 0.5;
            let ring = simplify_closed_contour(pts.clone(), tol);
            let worst = pts
                .iter()
                .map(|p| {
                    (0..ring.len())
                        .map(|i| dist_to_segment(*p, ring[i], ring[(i + 1) % ring.len()]))
                        .fold(f64::INFINITY, f64::min)
                })
                .fold(0.0, f64::max);
            assert!(worst * scale <= 0.25 + 1e-9, "{scale}×: {:.3} device px", worst * scale);
            assert!(ring.len() < n / 3, "{scale}×: {} of {n} points kept", ring.len());
            let line = simplify_polyline(pts.clone(), tol);
            assert_eq!((line[0], line[line.len() - 1]), (pts[0], pts[n - 1]), "the ends stay");
        }
    }

    #[test]
    fn vertex_counts_grow_sensibly_with_zoom() {
        let circle = &shapes(100.0)[0];
        let n = |scale: f64| flatten_at(circle, scale, Flat::fill).len();
        let (n1, n8, n64) = (n(1.0), n(8.0), n(64.0));
        // The flattening error falls with the square of the piece length, so
        // 8× the scale needs about √8 ≈ 2.8× the pieces (binary subdivision
        // lands on 2× or 4×), never the 8× a linear refinement would cost.
        for (lo, hi, what) in [(n1, n8, "1× → 8×"), (n8, n64, "8× → 64×")] {
            let ratio = (hi - 1) as f64 / (lo - 1) as f64;
            assert!(
                (1.9..=4.1).contains(&ratio),
                "{what}: {lo} → {hi} points (×{ratio:.2})"
            );
        }
        // Zooming out needs fewer pieces, and the bands are clamped: nothing
        // changes past 64× or below 1/16×.
        assert!(n(0.25) < n1, "{} vs {n1}", n(0.25));
        let top = 2f64.powi(MAX_BAND);
        let bottom = 2f64.powi(MIN_BAND);
        assert_eq!(n(top * 4.0), n(top), "clamped above {top}×");
        assert_eq!(n(bottom / 4.0), n(bottom), "clamped below {bottom}×");
        // The whole fill (triangles) follows.
        let tris = |scale: f64| {
            let _g = enter_plan(scale);
            tessellate_combined_oracle(
                circle,
                0.0,
                true,
                None,
                None,
                0.0,
                LineCap::Butt,
                LineJoin::Miter,
                FillRule::NonZero,
                StrokeAlign::Center,
            )
            .positions
            .len()
        };
        assert!(
            tris(0.25) < tris(1.0) && tris(1.0) < tris(8.0) && tris(8.0) < tris(64.0),
            "{} {} {} {}",
            tris(0.25),
            tris(1.0),
            tris(8.0),
            tris(64.0)
        );
    }

    #[test]
    fn the_1x_flattening_is_unchanged() {
        // Band 0 is the studio-golden 0.5-unit flattening (what the oracle
        // fixtures and goldens pin), and so is every scale that rounds to it.
        for cmds in shapes(100.0) {
            let legacy = flatten_at(&cmds, 1.0, Flat::legacy);
            for scale in [1.0, 1.3, 0.75] {
                assert_eq!(flatten_at(&cmds, scale, Flat::fill), legacy, "{scale}×");
            }
        }
    }
}
