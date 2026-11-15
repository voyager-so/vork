//! Offset Path (Illustrator's Object ▸ Path ▸ Offset Path…, shared-foundation F4.2).
//!
//! A path grown (`distance > 0`) or shrunk (`distance < 0`) by a distance, with
//! miter / round / bevel joins and a miter limit.
//!
//! 1. Curves are flattened to polylines within a third of `tolerance` (and at
//!    most 4° of turn per chord) by Simplify's sampler.
//! 2. `i_overlay` (MIT/Apache, integer arithmetic) resolves the input under its
//!    fill rule and offsets it: joins and the clean-up of self-intersections,
//!    swallowed holes and inverted loops in one robust pass. Miter joins are
//!    offset as bevels plus the miter wedges within the limit
//!    (`miter_wedges`), so a miter over the limit is bevelled as in
//!    Illustrator, not clipped as i_overlay's own miter join does. An open path has no inside, so its offset is the closed outline
//!    around it (a stroke of width `2·|distance|`), as Illustrator makes it,
//!    with butt ends (round ends with round joins).
//! 3. The polygons are refitted to curves within the rest of the tolerance by
//!    Simplify (corners kept), so the result stays within `tolerance` of the
//!    exact offset.
//!
//! Why not the stroke outliner plus the flo_curves boolean solver (the first
//! version): the outline band of a curved path offset by more than its
//! curvature radius (cusps, trimmed loops) makes flo_curves' ray-collision sort
//! inconsistent, which panics. That hit about 30% of random curved shapes and
//! some realistic blobs at large offsets, and the wasm build is
//! `panic = "abort"`. `kurbo` was evaluated too: it has stroke expansion and
//! offset-curve fitting but no booleans, so the clean-up would still need a
//! solver. `i_overlay`'s integer engine has no such failure mode;
//! `offset_fuzz_no_panic` and `large_offsets_on_realistic_shapes_never_panic`
//! pin that.
//!
//! Contract (`core/geometry/pathOps.ts` on the TS side):
//!
//! ```jsonc
//! // request: the path in ONE coordinate space (the caller rebases)
//! { "points": PathPoint[], "closed": bool, "fillRule"?: "nonzero"|"evenodd",
//!   "distance": number,               // path units; negative insets a closed path
//!   "join"?: "miter"|"round"|"bevel", // default miter
//!   "miterLimit"?: 4,                 // 1–500, ratio as for strokes
//!   "tolerance"?: 0.1 }               // max distance of the result from the exact offset
//! // response (the boolean envelope)
//! { "status": "ok", "points", "fillRule": "evenodd", "closed": true, "subpathCount", "bounds" }
//! { "status": "empty" }                // an inset that removed the whole shape
//! { "status": "error", "code", "message" }
//! ```

use i_overlay::core::fill_rule::FillRule as OverlayFill;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::simplify::SimplifyShape;
use i_overlay::float::single::SingleFloatOverlay;
use i_overlay::mesh::float::outline::offset::OutlineOffset;
use i_overlay::mesh::float::stroke::offset::StrokeOffset;
use i_overlay::mesh::float::style::{LineCap, LineJoin, OutlineStyle, StrokeStyle};
use crate::value::{json, Value};

use super::simplify::{flatten, simplify_path, split_subpaths, subpath_cubics, SimplifyOptions};
use super::vec2::V2;
use crate::blend::BlendPathPoint;
use crate::path_bool::{BooleanError, BooleanOutcome, Bounds, FillRule};
use crate::types::Affine;

/// Largest |distance| accepted (path units).
pub const MAX_OFFSET: f64 = 100_000.0;
/// Default tolerance: the result is within this of the exact offset (path units).
pub const DEFAULT_OFFSET_TOLERANCE: f64 = 0.1;
/// Illustrator's default miter limit for Offset Path.
pub const DEFAULT_OFFSET_MITER_LIMIT: f64 = 4.0;
/// Refits finer than this run on a copy scaled up to it (Simplify's own floors are px).
const REFIT_SCALE_TOLERANCE: f64 = 0.1;
/// The result stays within this fraction of |distance| of the exact offset (and within `tolerance`).
pub const RELATIVE_OFFSET_TOLERANCE: f64 = 0.02;
/// Largest `points` array accepted.
pub const MAX_OFFSET_POINTS: usize = 100_000;
/// Largest flattened polygon (vertices) handed to the offsetter.
const MAX_POLYGON_VERTICES: usize = 400_000;
/// Coordinates beyond this are refused (the integer engine scales to the bounds).
const MAX_COORD: f64 = 1e7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffsetJoin {
    Miter,
    Round,
    Bevel,
}

impl OffsetJoin {
    pub fn from_token(s: &str) -> Option<OffsetJoin> {
        Some(match s {
            "miter" => OffsetJoin::Miter,
            "round" => OffsetJoin::Round,
            "bevel" => OffsetJoin::Bevel,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct OffsetRequest {
    pub points: Vec<BlendPathPoint>,
    pub closed: bool,
    pub fill_rule: FillRule,
    pub distance: f64,
    pub join: OffsetJoin,
    pub miter_limit: f64,
    pub tolerance: f64,
}

impl OffsetRequest {
    pub fn new(points: Vec<BlendPathPoint>, closed: bool, distance: f64) -> OffsetRequest {
        OffsetRequest {
            points,
            closed,
            fill_rule: FillRule::NonZero,
            distance,
            join: OffsetJoin::Miter,
            miter_limit: DEFAULT_OFFSET_MITER_LIMIT,
            tolerance: DEFAULT_OFFSET_TOLERANCE,
        }
    }
}

fn err(code: &str, message: &str) -> BooleanError {
    BooleanError {
        code: code.to_string(),
        message: message.to_string(),
        operand_index: None,
    }
}

type Contour = Vec<[f64; 2]>;

/// Every subpath as a polyline (curves within `flat`), repeated points dropped.
fn polylines(points: &[BlendPathPoint], closed: bool, flat: f64) -> Vec<Contour> {
    let mut out = Vec::new();
    for sub in split_subpaths(points) {
        let mut poly: Vec<V2> = flatten(&subpath_cubics(&sub, false), flat);
        poly.dedup_by(|b, a| a.dist(*b) <= 1e-9);
        if closed && poly.len() > 1 && poly[0].dist(poly[poly.len() - 1]) <= 1e-9 {
            poly.pop();
        }
        if poly.len() >= 2 {
            out.push(poly.iter().map(|p| [p.x, p.y]).collect());
        }
    }
    out
}

/// The larger side of the box around the path's points (anchors and handles; 0 for none or a non-finite one).
fn path_extent(points: &[BlendPathPoint]) -> f64 {
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for sub in split_subpaths(points) {
        for c in subpath_cubics(&sub, false) {
            for p in c.0 {
                lo = [lo[0].min(p.x), lo[1].min(p.y)];
                hi = [hi[0].max(p.x), hi[1].max(p.y)];
            }
        }
    }
    let e = (hi[0] - lo[0]).max(hi[1] - lo[1]);
    if e.is_finite() {
        e
    } else {
        0.0
    }
}

fn in_range(p: &[f64; 2]) -> bool {
    p[0].is_finite() && p[1].is_finite() && p[0].abs() <= MAX_COORD && p[1].abs() <= MAX_COORD
}

/// Offset a path. Pure and total: every input comes back as an outcome or a
/// typed error.
pub fn offset_path(request: &OffsetRequest) -> Result<BooleanOutcome, BooleanError> {
    let d = request.distance;
    if !d.is_finite() || d.abs() > MAX_OFFSET {
        return Err(err(
            "invalid_request",
            "distance must be a finite number within ±100000",
        ));
    }
    if request.points.len() > MAX_OFFSET_POINTS {
        return Err(err("too_many_points", "at most 100000 path points"));
    }
    let requested = if request.tolerance.is_finite() && request.tolerance > 0.0 {
        request.tolerance.clamp(1e-3, 10.0)
    } else {
        DEFAULT_OFFSET_TOLERANCE
    };
    // The result's error must stay small beside the distance: a fixed 0.1 moved the edges of a small offset (a 1 px
    // square grown by 0.05 came back 0.04 off, a 0.1 px one grown by 0.005 four times the distance off). Not below
    // a millionth of the path's extent, so a large path offset by a hair doesn't flatten to millions of vertices.
    let extent = path_extent(&request.points);
    let tolerance = if d.abs() < 1e-9 {
        requested
    } else {
        requested.min(
            (d.abs() * RELATIVE_OFFSET_TOLERANCE)
                .max(extent * 1e-6)
                .max(1e-9),
        )
    };
    let miter_limit = if request.miter_limit.is_finite() {
        request.miter_limit.clamp(1.0, 500.0)
    } else {
        DEFAULT_OFFSET_MITER_LIMIT
    };
    let contours = polylines(&request.points, request.closed, tolerance / 3.0);
    if contours.iter().any(|c| !c.iter().all(in_range)) {
        return Err(err(
            "non_finite_coordinate",
            "the path has a non-finite or out-of-range coordinate",
        ));
    }
    if contours.iter().map(Vec::len).sum::<usize>() > MAX_POLYGON_VERTICES {
        return Err(err(
            "too_complex",
            "the path flattens to too many vertices (raise the tolerance)",
        ));
    }
    if contours.is_empty() {
        return Ok(BooleanOutcome::Empty);
    }

    // Round joins and caps: chord steps whose sagitta stays within a third of the tolerance.
    let arc_step = (8.0 * (tolerance / 3.0) / d.abs().max(1e-9)).sqrt();
    // Miter joins are offset as bevels, and every corner whose miter stays within `limit × distance` gets its
    // miter wedge back (`miter_wedges`). i_overlay's own miter join CLIPS a longer miter at the limit (a flat
    // cut far out at the tip) where Illustrator's Offset Path and Moda's stroke joins BEVEL it.
    let join = match request.join {
        OffsetJoin::Round => LineJoin::Round(arc_step),
        OffsetJoin::Bevel | OffsetJoin::Miter => LineJoin::Bevel,
    };
    let miter = request.join == OffsetJoin::Miter;
    let shapes = if request.closed {
        let fill = match request.fill_rule {
            FillRule::NonZero => OverlayFill::NonZero,
            FillRule::EvenOdd => OverlayFill::EvenOdd,
        };
        let resolved = contours.simplify_shape(fill);
        if d.abs() < 1e-9 || resolved.is_empty() {
            resolved
        } else {
            let bevelled = resolved.outline(&OutlineStyle::new(d).line_join(join));
            let wedges = if miter {
                let rings: Vec<(&Contour, bool)> = resolved
                    .iter()
                    .flat_map(|shape| shape.iter().enumerate().map(|(i, c)| (c, i > 0)))
                    .collect();
                miter_wedges(&rings, true, d, miter_limit)
            } else {
                Vec::new()
            };
            if wedges.is_empty() {
                bevelled
            } else {
                // Growing: the wedges add to the offset; shrinking: they are part of what is cut away.
                let rule = if d > 0.0 {
                    OverlayRule::Union
                } else {
                    OverlayRule::Difference
                };
                bevelled.overlay(&wedges, rule, OverlayFill::NonZero)
            }
        }
    } else {
        if d.abs() < 1e-9 {
            return Ok(BooleanOutcome::Empty);
        }
        let cap = if request.join == OffsetJoin::Round {
            LineCap::Round(arc_step)
        } else {
            LineCap::Butt
        };
        let style = StrokeStyle::new(2.0 * d.abs())
            .line_join(join)
            .start_cap(cap.clone())
            .end_cap(cap);
        let stroked = contours.stroke(style, false);
        let wedges = if miter {
            // An open path's outline is a stroke: a wedge on whichever side each corner opens.
            let rings: Vec<(&Contour, bool)> = contours
                .iter()
                .flat_map(|c| [(c, false), (c, true)])
                .collect();
            miter_wedges(&rings, false, d.abs(), miter_limit)
        } else {
            Vec::new()
        };
        if wedges.is_empty() {
            stroked
        } else {
            stroked.overlay(&wedges, OverlayRule::Union, OverlayFill::NonZero)
        }
    };

    let mut polygon: Vec<BlendPathPoint> = Vec::new();
    for contour in shapes.iter().flatten() {
        let contour = &drop_straight_vertices(contour, tolerance * 0.25);
        if contour.len() < 3 {
            continue;
        }
        if !contour.iter().all(in_range) {
            return Err(err(
                "out_of_range",
                "the offset exceeds the supported coordinate range; reduce the distance or rebase the path",
            ));
        }
        polygon.push(BlendPathPoint::Move {
            x: contour[0][0],
            y: contour[0][1],
        });
        polygon.extend(
            contour[1..]
                .iter()
                .map(|p| BlendPathPoint::Line { x: p[0], y: p[1] }),
        );
    }
    if polygon.is_empty() {
        return Ok(BooleanOutcome::Empty);
    }
    Ok(refit(polygon, tolerance * 2.0 / 3.0))
}

/// A closed ring without the vertices it runs (almost) straight through: within `within` of the line between its
/// neighbours, going on. Where a merged miter wedge meets the bevelled outline, its ends are left on the straight
/// edges (a few thousandths off them, from the integer overlay) and would come back as extra anchors.
fn drop_straight_vertices(ring: &Contour, within: f64) -> Contour {
    // Near-duplicates first (the two ends of a seam a hair apart make a zig-zag no straight test sees).
    let mut out: Contour = Vec::with_capacity(ring.len());
    for &p in ring {
        if out
            .last()
            .is_none_or(|q: &[f64; 2]| (p[0] - q[0]).hypot(p[1] - q[1]) > within)
        {
            out.push(p);
        }
    }
    while out.len() > 3
        && (out[0][0] - out[out.len() - 1][0]).hypot(out[0][1] - out[out.len() - 1][1]) <= within
    {
        out.pop();
    }
    // Judged against the original neighbours, and never two in a row: a removed vertex is then within `within` of
    // the edge that replaces it (removing greedily, each against neighbours already shortened, let the error add up
    // along a curve's many short chords).
    let n = out.len();
    if n <= 3 {
        return out;
    }
    let straight = |i: usize| {
        let (p, v, q) = (out[(i + n - 1) % n], out[i], out[(i + 1) % n]);
        let (ux, uy, wx, wy) = (v[0] - p[0], v[1] - p[1], q[0] - v[0], q[1] - v[1]);
        let (cx, cy) = (q[0] - p[0], q[1] - p[1]);
        let chord = cx.hypot(cy);
        let (lu, lw) = (ux.hypot(uy), wx.hypot(wy));
        // Off the chord p → q by at most `within`, between p and q, and turning by well under a degree (a seam
        // left on a straight edge, not one of a curve's samples).
        chord > 0.0
            && ((v[0] - p[0]) * cy - (v[1] - p[1]) * cx).abs() <= within * chord
            && (ux * wy - uy * wx).abs() <= 1e-3 * lu * lw
            && ux * wx + uy * wy > 0.0
            && ux * cx + uy * cy > 0.0
            && wx * cx + wy * cy > 0.0
    };
    let mut drop = vec![false; n];
    for i in 0..n {
        drop[i] = straight(i) && !(i > 0 && drop[i - 1]) && !(i == n - 1 && drop[0]);
    }
    let kept: Contour = (0..n).filter(|&i| !drop[i]).map(|i| out[i]).collect();
    if kept.len() >= 3 {
        kept
    } else {
        out
    }
}

/// The miter wedges a bevelled offset lacks: for every corner of `rings` where the two offset edges part, the
/// triangle between their ends and the miter point, when the miter is at most `limit × |d|` from the corner
/// (the stroke miter limit: the ratio is 1 / cos(turn / 2)). Corners turning less than 5° are left bevelled,
/// as i_overlay leaves them. Each wedge is returned as the kite (corner, a, miter point, b): it covers the bevel
/// triangle too, which the bevelled outline already holds (an outset) or already lacks (an inset), so the overlay
/// merges it instead of leaving it beside the bevel.
///
/// `rings`: each ring with a flag. Closed rings (`closed`): the flag says it is a hole, and the ring's inside is
/// read from its orientation (an outer ring's inside lies on the side its area's sign gives, a hole's on the
/// other). Open rings (`!closed`): the same polyline is listed twice, the flag picking the side (false = left),
/// with `d` = the half width. Every wedge is wound like a positive-area ring.
fn miter_wedges(rings: &[(&Contour, bool)], closed: bool, d: f64, limit: f64) -> Vec<Contour> {
    let min_turn_cos = (5.0_f64).to_radians().cos();
    let mut out = Vec::new();
    for &(ring, flag) in rings {
        let n = ring.len();
        if n < 3 {
            continue;
        }
        let area: f64 = (0..n)
            .map(|i| {
                let (p, q) = (ring[i], ring[(i + 1) % n]);
                p[0] * q[1] - q[0] * p[1]
            })
            .sum();
        // Offset side: `left` = the offset lies left of travel.
        let left = if closed { (area > 0.0) == flag } else { !flag };
        let normal = |u: [f64; 2]| if left { [-u[1], u[0]] } else { [u[1], -u[0]] };
        let unit = |a: [f64; 2], b: [f64; 2]| {
            let (x, y) = (b[0] - a[0], b[1] - a[1]);
            let l = (x * x + y * y).sqrt();
            (l > 1e-12).then(|| [x / l, y / l])
        };
        let (first, last) = if closed { (0, n) } else { (1, n - 1) };
        for i in first..last {
            let (prev, v, next) = (ring[(i + n - 1) % n], ring[i], ring[(i + 1) % n]);
            let (Some(u1), Some(u2)) = (unit(prev, v), unit(v, next)) else {
                continue;
            };
            let (n1, n2) = (normal(u1), normal(u2));
            let cos = n1[0] * n2[0] + n1[1] * n2[1];
            if cos > min_turn_cos || cos <= -1.0 + 1e-9 {
                continue;
            }
            let a = [v[0] + d * n1[0], v[1] + d * n1[1]];
            let b = [v[0] + d * n2[0], v[1] + d * n2[1]];
            // Only where the offset edges part (b lies ahead of a along the first edge).
            if (b[0] - a[0]) * u1[0] + (b[1] - a[1]) * u1[1] <= 1e-12 {
                continue;
            }
            // Miter ratio 1 / cos(turn / 2) = sqrt(2 / (1 + cos)).
            if (2.0 / (1.0 + cos)).sqrt() > limit + 1e-9 {
                continue;
            }
            let k = d / (1.0 + cos);
            let peak = [v[0] + k * (n1[0] + n2[0]), v[1] + k * (n1[1] + n2[1])];
            // The wedge reaches back to the corner itself (a kite), so it overlaps the bevelled outline rather than
            // only meeting it along the bevel: met edge to edge, the union kept it as its own subpath, and the
            // offset's stroke drew the bevel line across every mitred corner.
            let mut kite = vec![v, a, peak, b];
            let signed: f64 = (0..4)
                .map(|i| {
                    let (p, q) = (kite[i], kite[(i + 1) % 4]);
                    p[0] * q[1] - q[0] * p[1]
                })
                .sum();
            if signed < 0.0 {
                kite.reverse();
            }
            out.push(kite);
        }
    }
    out
}

/// The polygonal result refitted to curves within `tolerance` (corners kept;
/// a contour that can't be fitted within it stays polygonal).
fn refit(points: Vec<BlendPathPoint>, tolerance: f64) -> BooleanOutcome {
    // Simplify works in px: it holds its tolerance to at least 0.01 and samples at least every 0.5. A finer
    // tolerance (a small offset) is fitted in a copy scaled up until it is REFIT_SCALE_TOLERANCE, then scaled back.
    let k = if tolerance < REFIT_SCALE_TOLERANCE {
        REFIT_SCALE_TOLERANCE / tolerance
    } else {
        1.0
    };
    let (ox, oy) = points
        .iter()
        .map(|p| match *p {
            BlendPathPoint::Move { x, y }
            | BlendPathPoint::Line { x, y }
            | BlendPathPoint::Curve { x, y, .. }
            | BlendPathPoint::Quadratic { x, y, .. }
            | BlendPathPoint::Arc { x, y, .. } => (x, y),
        })
        .fold((f64::INFINITY, f64::INFINITY), |(x, y), (px, py)| {
            (x.min(px), y.min(py))
        });
    let (ox, oy) = if ox.is_finite() && oy.is_finite() {
        (ox, oy)
    } else {
        (0.0, 0.0)
    };
    let up = Affine {
        a: k,
        b: 0.0,
        c: 0.0,
        d: k,
        e: -ox * k,
        f: -oy * k,
    };
    let down = Affine {
        a: 1.0 / k,
        b: 0.0,
        c: 0.0,
        d: 1.0 / k,
        e: ox,
        f: oy,
    };
    let scaled: Vec<BlendPathPoint> = if k > 1.0 {
        points.iter().map(|p| p.apply_affine(up)).collect()
    } else {
        points
    };
    let fitted = simplify_path(
        &scaled,
        true,
        &SimplifyOptions {
            tolerance: tolerance * k,
            ..Default::default()
        },
    );
    let points: Vec<BlendPathPoint> = if k > 1.0 {
        fitted.points.iter().map(|p| p.apply_affine(down)).collect()
    } else {
        fitted.points
    };
    let mut bounds = Bounds {
        min_x: f64::INFINITY,
        min_y: f64::INFINITY,
        max_x: f64::NEG_INFINITY,
        max_y: f64::NEG_INFINITY,
    };
    let subs = split_subpaths(&points);
    for sub in &subs {
        for c in subpath_cubics(sub, true) {
            for t in 0..=16 {
                let p = c.eval(t as f64 / 16.0);
                bounds.include(p.x, p.y);
            }
        }
    }
    BooleanOutcome::Shape {
        subpath_count: subs.len(),
        points,
        bounds,
    }
}

fn error_json(e: &BooleanError) -> String {
    json!({"status": "error", "code": e.code, "message": e.message}).to_string()
}

/// JSON in / JSON out; always an envelope, never panics. See the module docs.
pub fn offset_path_json(request_json: &str) -> String {
    if request_json.len() > 64 * 1024 * 1024 {
        return error_json(&err("request_too_large", "request exceeds 64 MiB"));
    }
    let value: Value = match crate::value::from_str(request_json) {
        Ok(v) => v,
        Err(e) => {
            return error_json(&err(
                "invalid_json",
                &format!("request is not valid JSON: {e}"),
            ))
        }
    };
    let Some(obj) = value.as_object() else {
        return error_json(&err("invalid_request", "request must be an object"));
    };
    let Some(raw) = obj.get("points").and_then(Value::as_array) else {
        return error_json(&err("invalid_request", "points must be an array"));
    };
    if raw.len() > MAX_OFFSET_POINTS {
        return error_json(&err("too_many_points", "at most 100000 path points"));
    }
    let mut points = Vec::with_capacity(raw.len());
    for p in raw {
        match BlendPathPoint::from_json(p) {
            Some(p) => points.push(p),
            None => {
                return error_json(&err(
                    "invalid_request",
                    "points contains a malformed PathPoint",
                ))
            }
        }
    }
    let Some(distance) = obj.get("distance").and_then(Value::as_f64) else {
        return error_json(&err("invalid_request", "distance must be a number"));
    };
    let fill_rule = match obj.get("fillRule").and_then(Value::as_str) {
        None => FillRule::NonZero,
        Some(s) => match FillRule::from_token(s) {
            Some(f) => f,
            None => {
                return error_json(&err(
                    "invalid_request",
                    "fillRule must be nonzero or evenodd",
                ))
            }
        },
    };
    let join = match obj.get("join").and_then(Value::as_str) {
        None => OffsetJoin::Miter,
        Some(s) => match OffsetJoin::from_token(s) {
            Some(j) => j,
            None => {
                return error_json(&err(
                    "invalid_request",
                    "join must be miter, round or bevel",
                ))
            }
        },
    };
    let request = OffsetRequest {
        points,
        closed: obj.get("closed").and_then(Value::as_bool).unwrap_or(false),
        fill_rule,
        distance,
        join,
        miter_limit: obj
            .get("miterLimit") // an offset request field, not a node key
            .and_then(Value::as_f64)
            .unwrap_or(DEFAULT_OFFSET_MITER_LIMIT),
        tolerance: obj
            .get("tolerance")
            .and_then(Value::as_f64)
            .unwrap_or(DEFAULT_OFFSET_TOLERANCE),
    };
    match offset_path(&request) {
        Ok(BooleanOutcome::Shape {
            points,
            subpath_count,
            bounds,
        }) => json!({
            "status": "ok",
            "points": points.iter().map(BlendPathPoint::to_json).collect::<Vec<_>>(),
            "fillRule": "evenodd",
            "closed": true,
            "subpathCount": subpath_count,
            "bounds": {
                "minX": bounds.min_x,
                "minY": bounds.min_y,
                "maxX": bounds.max_x,
                "maxY": bounds.max_y,
            },
        })
        .to_string(),
        Ok(BooleanOutcome::Empty) => json!({"status": "empty"}).to_string(),
        Err(e) => error_json(&e),
    }
}

#[cfg(test)]
#[path = "offset_path_tests.rs"]
mod tests;
