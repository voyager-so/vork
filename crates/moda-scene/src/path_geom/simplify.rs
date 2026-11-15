//! Simplify (Illustrator's Object ▸ Path ▸ Simplify…, shared-foundation F4.2).
//!
//! Each subpath is sampled densely (adaptive flattening, then points at most
//! `max(tolerance, 0.5)` apart), refitted with the fewest cubics that stay
//! within the tolerance (corners kept; see "Curves" below), and the fit is then
//! MEASURED: the largest distance between the fit and the original, both ways
//! (every original sample to the fit and every fit sample to the original). A
//! fit that strays past `tolerance` is retried at ¾ of the tolerance (up to
//! four times); a subpath that still strays, or that would not come out with
//! fewer anchors, is kept as it was. So the result is never further than
//! `tolerance` from the input and never has more anchors.
//!
//! `straightLines` is Illustrator's "Convert to Straight Lines": the subpath
//! becomes a polygon (Ramer–Douglas–Peucker on the samples, within
//! `tolerance`, corners always kept).
//!
//! Contract (`core/geometry/pathOps.ts`):
//!
//! ```jsonc
//! // request
//! { "points": PathPoint[], "closed": bool,
//!   "tolerance"?: 1,         // max distance from the original, path units
//!   "cornerAngle"?: 35,      // turns sharper than this (degrees) stay corners
//!   "straightLines"?: false }
//! // response
//! { "status": "ok", "points": PathPoint[], "simplified": bool,
//!   "anchorsBefore", "anchorsAfter", "maxDeviation",
//!   "subpaths": [{ "simplified": bool, "sampleS"?: number[], "sampleT"?: number[] }] }
//! { "status": "empty" }
//! { "status": "error", "code", "message" }
//! ```
//!
//! `sampleS[i]` is the arc length along the ORIGINAL subpath of dense sample
//! `i` and `sampleT[i]` the arc-length fraction of the fitted subpath where it
//! landed (curve fits only), so callers can carry pen ink recorded along the
//! old path over to the new one.

use crate::value::{json, Value};

use super::arc::{arc_to_beziers, ArcParams};
use super::fit::{
    chord_params, detect_corners, fit_points, generate, reparameterize, shared_arc_len, FitOptions,
    FitSeg, MAX_COORD,
};
use super::vec2::{v2, Cubic, V2};
use crate::blend::BlendPathPoint;

/// Largest `points` array accepted.
pub const MAX_SIMPLIFY_POINTS: usize = 100_000;
/// Dense samples per subpath, at most (the spacing grows to fit).
const MAX_SAMPLES_PER_SUBPATH: usize = 20_000;
/// Dense samples over all subpaths, at most (the spacing grows to fit, as for one subpath). The fit costs about
/// 5 µs a sample, so this bounds a call at about 2 s: 25,000 long subpaths at the per-subpath cap took 40 min.
const MAX_SAMPLES_TOTAL: usize = 400_000;
/// Default corner angle (degrees of turn), as the Path menu has always used.
pub const DEFAULT_CORNER_ANGLE: f64 = 35.0;

#[derive(Debug, Clone, Copy)]
pub struct SimplifyOptions {
    pub tolerance: f64,
    pub corner_angle: f64,
    pub straight_lines: bool,
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        SimplifyOptions {
            tolerance: 1.0,
            corner_angle: DEFAULT_CORNER_ANGLE,
            straight_lines: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SubpathResult {
    pub simplified: bool,
    pub sample_s: Option<Vec<f64>>,
    pub sample_t: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Default)]
pub struct SimplifyOutput {
    pub points: Vec<BlendPathPoint>,
    pub simplified: bool,
    pub anchors_before: usize,
    pub anchors_after: usize,
    /// Largest distance between the result and the original (0 when unchanged).
    pub max_deviation: f64,
    pub subpaths: Vec<SubpathResult>,
}

// ---------------------------------------------------------------------------
// Segments and sampling
// ---------------------------------------------------------------------------

/// Split at every `Move`; each subpath keeps its points (the first is its start).
pub(crate) fn split_subpaths(points: &[BlendPathPoint]) -> Vec<Vec<BlendPathPoint>> {
    let mut out: Vec<Vec<BlendPathPoint>> = Vec::new();
    for p in points {
        if matches!(p, BlendPathPoint::Move { .. }) || out.is_empty() {
            out.push(Vec::new());
        }
        if let Some(last) = out.last_mut() {
            last.push(p.clone());
        }
    }
    out.retain(|s| !s.is_empty());
    out
}

fn anchor(p: &BlendPathPoint) -> V2 {
    match p {
        BlendPathPoint::Move { x, y }
        | BlendPathPoint::Line { x, y }
        | BlendPathPoint::Curve { x, y, .. }
        | BlendPathPoint::Quadratic { x, y, .. }
        | BlendPathPoint::Arc { x, y, .. } => v2(*x, *y),
    }
}

/// The subpath as cubics (lines as degenerate cubics), closing segment included when `closed`.
pub(crate) fn subpath_cubics(sub: &[BlendPathPoint], closed: bool) -> Vec<Cubic> {
    let mut out = Vec::new();
    let Some(first) = sub.first() else {
        return out;
    };
    let start = anchor(first);
    let mut cur = start;
    for p in &sub[1..] {
        let end = anchor(p);
        match p {
            BlendPathPoint::Move { .. } | BlendPathPoint::Line { .. } => {
                out.push(Cubic::line(cur, end))
            }
            BlendPathPoint::Curve { cp1, cp2, .. } => {
                out.push(Cubic([cur, v2(cp1.0, cp1.1), v2(cp2.0, cp2.1), end]))
            }
            BlendPathPoint::Quadratic { cp, .. } => {
                let c = v2(cp.0, cp.1);
                out.push(Cubic([
                    cur,
                    cur + (c - cur) * (2.0 / 3.0),
                    end + (c - end) * (2.0 / 3.0),
                    end,
                ]));
            }
            BlendPathPoint::Arc {
                x,
                y,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            } => {
                let arcs = arc_to_beziers(
                    cur.x,
                    cur.y,
                    &ArcParams {
                        rx: *rx,
                        ry: *ry,
                        x_axis_rotation: *x_axis_rotation,
                        large_arc: *large_arc,
                        sweep: *sweep,
                        x: *x,
                        y: *y,
                    },
                );
                if arcs.is_empty() {
                    out.push(Cubic::line(cur, end));
                } else {
                    let mut a = cur;
                    for b in arcs {
                        let e = v2(b.end[0], b.end[1]);
                        out.push(Cubic([
                            a,
                            v2(b.cp1[0], b.cp1[1]),
                            v2(b.cp2[0], b.cp2[1]),
                            e,
                        ]));
                        a = e;
                    }
                }
            }
        }
        cur = end;
    }
    if closed && cur.dist(start) > 1e-9 {
        out.push(Cubic::line(cur, start));
    }
    out
}

/// Flatten cubics into a polyline whose chords stay within `flat` of the curve.
pub(crate) fn flatten(cubics: &[Cubic], flat: f64) -> Vec<V2> {
    let mut out: Vec<V2> = Vec::new();
    for c in cubics {
        if out.is_empty() {
            out.push(c.0[0]);
        }
        if c.is_chord(1e-9, 1e-9) {
            out.push(c.0[3]);
            continue;
        }
        let [p0, p1, p2, p3] = c.0;
        // Max second difference bounds the chord error: n ≥ sqrt(3·dd / (4·flat)).
        let dd = (p0 - p1 * 2.0 + p2).len().max((p1 - p2 * 2.0 + p3).len()) * 6.0;
        let by_error = (dd / (8.0 * flat.max(1e-6))).sqrt().ceil();
        // And at most ~4° of turn per chord (the control polygon bounds the turn).
        let turn = |a: V2, b: V2| match (a.unit(), b.unit()) {
            (Some(a), Some(b)) => a.cross(b).atan2(a.dot(b)).abs(),
            _ => 0.0,
        };
        let total_turn = turn(p1 - p0, p2 - p1) + turn(p2 - p1, p3 - p2);
        let by_turn = (total_turn / 4f64.to_radians()).ceil();
        let n = (by_error.max(by_turn) as usize).clamp(1, 512);
        let mut parameters: Vec<f64> = (1..=n).map(|i| i as f64 / n as f64).collect();
        // Only collinear reversals need extra samples: keep the established
        // sampling of other curves unchanged.
        if c.is_chord(1e-9, f64::INFINITY) {
            parameters.extend(c.extrema(|p| p.x));
            parameters.extend(c.extrema(|p| p.y));
        }
        parameters.sort_by(f64::total_cmp);
        parameters.dedup();
        for t in parameters {
            out.push(c.eval(t));
        }
    }
    out
}

/// Points at most `step` apart along `poly` (every vertex kept), with arc lengths.
fn densify(poly: &[V2], step: f64) -> (Vec<V2>, Vec<f64>) {
    let mut pts = Vec::with_capacity(poly.len());
    let mut s = Vec::with_capacity(poly.len());
    let mut acc = 0.0;
    for (i, p) in poly.iter().enumerate() {
        if i == 0 {
            pts.push(*p);
            s.push(0.0);
            continue;
        }
        let a = poly[i - 1];
        let len = a.dist(*p);
        if len <= 1e-12 {
            continue;
        }
        let parts = ((len / step).ceil() as usize).max(1);
        for k in 1..=parts {
            let f = k as f64 / parts as f64;
            pts.push(a.lerp(*p, f));
            s.push(acc + len * f);
        }
        acc += len;
    }
    (pts, s)
}

// ---------------------------------------------------------------------------
// Distance: a uniform grid over polyline segments
// ---------------------------------------------------------------------------

struct SegGrid {
    x0: f64,
    y0: f64,
    cell: f64,
    w: usize,
    h: usize,
    cells: Vec<Vec<u32>>,
    segs: Vec<(V2, V2)>,
}

impl SegGrid {
    fn new(poly: &[V2], closed: bool, cell: f64) -> SegGrid {
        let mut segs: Vec<(V2, V2)> = poly.windows(2).map(|w| (w[0], w[1])).collect();
        if closed && poly.len() > 2 {
            segs.push((poly[poly.len() - 1], poly[0]));
        }
        if segs.is_empty() && !poly.is_empty() {
            segs.push((poly[0], poly[0]));
        }
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for p in poly {
            x0 = x0.min(p.x);
            y0 = y0.min(p.y);
            x1 = x1.max(p.x);
            y1 = y1.max(p.y);
        }
        // Keep the grid at most ~512² cells.
        let span = (x1 - x0).max(y1 - y0).max(1e-9);
        let cell = cell.max(span / 512.0).max(1e-9);
        let w = ((x1 - x0) / cell).floor() as usize + 1;
        let h = ((y1 - y0) / cell).floor() as usize + 1;
        let mut cells = vec![Vec::new(); w * h];
        for (k, (a, b)) in segs.iter().enumerate() {
            let ci =
                |v: f64, o: f64, n: usize| (((v - o) / cell).floor().max(0.0) as usize).min(n - 1);
            let (i0, i1) = (ci(a.x.min(b.x), x0, w), ci(a.x.max(b.x), x0, w));
            let (j0, j1) = (ci(a.y.min(b.y), y0, h), ci(a.y.max(b.y), y0, h));
            for j in j0..=j1 {
                for i in i0..=i1 {
                    cells[j * w + i].push(k as u32);
                }
            }
        }
        SegGrid {
            x0,
            y0,
            cell,
            w,
            h,
            cells,
            segs,
        }
    }

    /// Distance from `p` to the polyline, or `cap` when nothing is within `cap`.
    fn dist(&self, p: V2, cap: f64) -> f64 {
        let r = (cap / self.cell).ceil() as isize;
        let ci = ((p.x - self.x0) / self.cell).floor() as isize;
        let cj = ((p.y - self.y0) / self.cell).floor() as isize;
        let mut best = cap;
        for j in (cj - r).max(0)..=(cj + r).min(self.h as isize - 1) {
            for i in (ci - r).max(0)..=(ci + r).min(self.w as isize - 1) {
                for &k in &self.cells[j as usize * self.w + i as usize] {
                    let (a, b) = self.segs[k as usize];
                    let d = b - a;
                    let l2 = d.len2();
                    let t = if l2 > 0.0 {
                        ((p - a).dot(d) / l2).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    best = best.min(p.dist(a + d * t));
                }
            }
        }
        best
    }
}

/// Two-way distance between two polylines, capped at `cap`.
fn hausdorff(a: &[V2], a_closed: bool, b: &[V2], b_closed: bool, cap: f64) -> f64 {
    let ga = SegGrid::new(a, a_closed, cap);
    let gb = SegGrid::new(b, b_closed, cap);
    let mut worst: f64 = 0.0;
    for p in a {
        worst = worst.max(gb.dist(*p, cap));
        if worst >= cap {
            return cap;
        }
    }
    for p in b {
        worst = worst.max(ga.dist(*p, cap));
        if worst >= cap {
            return cap;
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// Straight lines: Ramer–Douglas–Peucker with forced corners
// ---------------------------------------------------------------------------

fn seg_dist(p: V2, a: V2, b: V2) -> f64 {
    let d = b - a;
    let l2 = d.len2();
    let t = if l2 > 0.0 {
        ((p - a).dot(d) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    p.dist(a + d * t)
}

fn rdp(pts: &[V2], tol: f64, keep: &mut [bool]) {
    // Explicit stack: (start, end).
    let mut stack = vec![(0usize, pts.len() - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (mut worst, mut at) = (0.0, a);
        for (i, p) in pts.iter().enumerate().take(b).skip(a + 1) {
            let d = seg_dist(*p, pts[a], pts[b]);
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
}

/// Polygon through a subset of `pts` within `tol`; `corners` are always kept.
fn straight(pts: &[V2], closed: bool, tol: f64, corners: &[usize]) -> Vec<V2> {
    let n = pts.len();
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    for &c in corners {
        if c < n {
            keep[c] = true;
        }
    }
    if closed {
        // Also split at the point farthest from the start, so a loop is two runs.
        let far = (0..n)
            .max_by(|&i, &j| pts[0].dist(pts[i]).total_cmp(&pts[0].dist(pts[j])))
            .unwrap_or(0);
        keep[far] = true;
    }
    let fixed: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
    for w in fixed.windows(2) {
        let (a, b) = (w[0], w[1]);
        let mut sub = vec![false; b - a + 1];
        rdp(&pts[a..=b], tol, &mut sub);
        for (k, s) in sub.iter().enumerate() {
            if *s {
                keep[a + k] = true;
            }
        }
    }
    if closed {
        // The closing run: last kept → wraps to 0.
        let last = (0..n).rev().find(|&i| keep[i]).unwrap_or(0);
        if last > 0 && last < n - 1 {
            let mut ring: Vec<V2> = pts[last..].to_vec();
            ring.push(pts[0]);
            let mut sub = vec![false; ring.len()];
            rdp(&ring, tol, &mut sub);
            for (k, s) in sub.iter().enumerate().take(ring.len() - 1) {
                if *s {
                    keep[last + k] = true;
                }
            }
        }
    }
    let mut out: Vec<V2> = (0..n).filter(|&i| keep[i]).map(|i| pts[i]).collect();
    if closed && out.len() > 1 && out[0].dist(out[out.len() - 1]) <= 1e-9 {
        out.pop();
    }
    out
}

// ---------------------------------------------------------------------------
// One subpath
// ---------------------------------------------------------------------------

struct SubOut {
    points: Vec<BlendPathPoint>,
    result: SubpathResult,
    deviation: f64,
}

fn keep_as_is(sub: &[BlendPathPoint]) -> SubOut {
    SubOut {
        points: sub.to_vec(),
        result: SubpathResult::default(),
        deviation: 0.0,
    }
}

/// A subpath's length along its dense flattening (what the sample spacing divides).
fn subpath_length(sub: &[BlendPathPoint], closed: bool, tol: f64) -> f64 {
    let cubics = subpath_cubics(sub, closed);
    let length: f64 = flatten(&cubics, tol * 0.02)
        .windows(2)
        .map(|w| w[0].dist(w[1]))
        .sum();
    if length.is_finite() {
        length
    } else {
        0.0
    }
}

/// `min_step`: the sample spacing the whole path's budget allows ([`MAX_SAMPLES_TOTAL`]).
fn simplify_subpath(
    sub: &[BlendPathPoint],
    closed: bool,
    opts: &SimplifyOptions,
    min_step: f64,
) -> SubOut {
    let tol = opts.tolerance;
    let cubics = subpath_cubics(sub, closed);
    if cubics.is_empty() || sub.len() < 3 {
        return keep_as_is(sub);
    }
    let original = flatten(&cubics, tol * 0.02);
    if original
        .iter()
        .any(|p| !p.is_finite() || p.x.abs() > MAX_COORD || p.y.abs() > MAX_COORD)
    {
        return keep_as_is(sub);
    }
    let length: f64 = original.windows(2).map(|w| w[0].dist(w[1])).sum();
    let step = tol
        .max(0.5)
        .max(length / MAX_SAMPLES_PER_SUBPATH as f64)
        .max(min_step);
    let (mut samples, mut sample_s) = densify(&original, step);
    if closed && samples.len() > 2 && samples[0].dist(samples[samples.len() - 1]) <= 1e-9 {
        samples.pop();
        sample_s.pop();
    }
    if samples.len() < 3 {
        return keep_as_is(sub);
    }
    // The measure: the original (densely flattened) against the candidate.
    let measure = |cand: &[BlendPathPoint]| -> f64 {
        let cand_cubics = subpath_cubics(cand, closed);
        let cand_poly = flatten(&cand_cubics, tol * 0.02);
        hausdorff(&original, closed, &cand_poly, closed, tol * 4.0)
    };

    if opts.straight_lines {
        let fit = fit_points(
            &samples,
            &FitOptions {
                tolerance: tol,
                corner_angle: opts.corner_angle,
                closed,
            },
        );
        // Corners of the fit mark where the polygon must turn; RDP adds the rest.
        let poly = straight(&samples, closed, tol * 0.95, &fit.corners);
        if poly.len() < 2 {
            return keep_as_is(sub);
        }
        let mut pts = vec![BlendPathPoint::Move {
            x: poly[0].x,
            y: poly[0].y,
        }];
        pts.extend(
            poly[1..]
                .iter()
                .map(|p| BlendPathPoint::Line { x: p.x, y: p.y }),
        );
        let dev = measure(&pts);
        if pts.len() >= sub.len() || dev > tol {
            return keep_as_is(sub);
        }
        return SubOut {
            points: pts,
            result: SubpathResult {
                simplified: true,
                sample_s: None,
                sample_t: None,
            },
            deviation: dev,
        };
    }

    // Corners are measured over at least two samples' spacing, so a polyline's
    // own small turns at a fine tolerance don't read as corners. Tangents are
    // read over a few of the input's chords (a polyline's vertices are where
    // it bends) and at least four samples.
    let mut chords: Vec<f64> = cubics
        .iter()
        .filter(|c| c.is_chord(1e-9, 1e-9))
        .map(|c| c.0[0].dist(c.0[3]))
        .collect();
    chords.sort_by(f64::total_cmp);
    let chord = chords.get(chords.len() / 2).copied().unwrap_or(0.0);
    let reach = Reach {
        corner: (2.0 * tol).max(2.0 * step),
        tangent: (4.0 * step).max(1.5 * chord),
    };
    let mut try_tol = tol;
    for _ in 0..4 {
        let fit = fit_curves(&samples, closed, try_tol, opts.corner_angle, reach);
        if fit.points.len() < 2 || fit.points.len() >= sub.len() {
            return keep_as_is(sub);
        }
        let dev = measure(&fit.points);
        if dev <= tol {
            return SubOut {
                points: straighten(fit.points),
                result: SubpathResult {
                    simplified: true,
                    sample_s: Some(sample_s),
                    sample_t: Some(fit.sample_t),
                },
                deviation: dev,
            };
        }
        try_tol *= 0.75;
    }
    keep_as_is(sub)
}

// ---------------------------------------------------------------------------
// Curves: the fewest cubics within the tolerance
// ---------------------------------------------------------------------------
//
// The Pencil's fitter (Schneider) splits at the worst sample whenever a cubic
// misses, measures the miss along the curve's parameter (which overstates the
// real distance) and gives up on a cubic after a few reparameterisations, so a
// tight tolerance costs it many anchors. Simplify fits instead:
//
// - Greedily, the LONGEST run of samples one cubic covers from each anchor
//   (exponential then binary search on the end sample), so anchors land as far
//   apart as the tolerance allows.
// - Each trial cubic is refined until its reparameterisation stops helping and
//   measured by DISTANCE: every sample to the curve, and the curve between two
//   samples to their chord.
// - The tangent at an anchor is the data's: the slope of a least-squares
//   quadratic through the samples around it, over a few of the input's chords
//   (a polyline's own vertices would otherwise tilt it). The next cubic leaves
//   along it (G1).
// - Then every anchor's tangent is re-aimed (golden section over its angle) to
//   keep its two cubics closest, and two neighbouring cubics that now refit as
//   one are merged; a few rounds.
// - Corners (turns sharper than the corner angle) split the subpath into runs
//   whose ends have their own tangents.
//
// On Illustrator's parity cases (docs/qa/illustrator-parity.md) this reaches
// its automatic Simplify's 0.13–0.19 px with as few or fewer anchors.

/// Reparameterisation passes per trial cubic, at most.
const TRIAL_ITERATIONS: usize = 16;
/// Golden-section steps when polishing an anchor's tangent.
const POLISH_STEPS: usize = 14;
/// How far (radians) either way of its tangent an anchor is re-aimed.
const POLISH_SPAN: f64 = 0.3;

/// One trial cubic over `pts` and where each sample landed on it.
struct Trial {
    cubic: Cubic,
    u: Vec<f64>,
    err: f64,
}

/// The largest distance between `pts` and `c` (samples at parameters `u`):
/// every sample to the curve, and the curve halfway between two samples to
/// their chord. A parameter that runs backwards (the cubic loops) is a miss.
fn trial_error(c: &Cubic, pts: &[V2], u: &[f64]) -> f64 {
    let mut worst: f64 = 0.0;
    for i in 0..pts.len() {
        worst = worst.max(c.eval(u[i]).dist(pts[i]));
        if i + 1 < pts.len() {
            if u[i + 1] < u[i] - 1e-9 {
                return f64::INFINITY;
            }
            let mid = c.eval(0.5 * (u[i] + u[i + 1]));
            worst = worst.max(seg_dist(mid, pts[i], pts[i + 1]));
        }
    }
    worst
}

/// The best cubic through `pts` (ends fixed) leaving along `t1`, arriving back
/// along `t2` (pointing from the last sample into the curve). Refinement stops
/// once the error is `done` or less (a yes / no question asks for `tol`), stops
/// improving, or is still far past `tol` after a few passes.
fn trial(pts: &[V2], t1: V2, t2: V2, tol: f64, done: f64) -> Trial {
    let poly_len: f64 = pts.windows(2).map(|w| w[0].dist(w[1])).sum();
    let mut u = chord_params(pts);
    let mut best = Trial {
        cubic: Cubic::line(pts[0], pts[pts.len() - 1]),
        u: u.clone(),
        err: f64::INFINITY,
    };
    let mut prev = f64::INFINITY;
    for k in 0..TRIAL_ITERATIONS {
        let c = generate(pts, &u, t1, t2, poly_len);
        reparameterize(&c, pts, &mut u);
        reparameterize(&c, pts, &mut u);
        let err = trial_error(&c, pts, &u);
        if err < best.err {
            best = Trial {
                cubic: c,
                u: u.clone(),
                err,
            };
        }
        // Well inside, no longer improving, or hopeless (a few passes in and still far out).
        if err <= done || err > 0.98 * prev || (k >= 2 && err > 4.0 * tol) {
            break;
        }
        prev = err;
    }
    best
}

/// Unit tangent of the data at sample `i` of `pts`, pointing forward: the
/// slope at `i` of a least-squares quadratic (in arc length) through the
/// samples within `reach` either side, clipped to the run. A polyline's
/// chords turn at its vertices; the quadratic reads the curve they trace.
fn data_tangent(pts: &[V2], i: usize, reach: f64) -> V2 {
    let n = pts.len();
    let fallback = || {
        let a = pts[i.saturating_sub(1)];
        let b = pts[(i + 1).min(n - 1)];
        (b - a)
            .unit()
            .or((pts[n - 1] - pts[0]).unit())
            .unwrap_or(v2(1.0, 0.0))
    };
    // Samples within `reach` of i along the run, with signed arc length.
    let mut near: Vec<(f64, V2)> = vec![(0.0, pts[i])];
    let mut s = 0.0;
    for k in (0..i).rev() {
        s -= pts[k].dist(pts[k + 1]);
        if -s > reach {
            break;
        }
        near.push((s, pts[k]));
    }
    s = 0.0;
    for k in i + 1..n {
        s += pts[k].dist(pts[k - 1]);
        if s > reach {
            break;
        }
        near.push((s, pts[k]));
    }
    if near.len() < 3 {
        return fallback();
    }
    // Normal equations for p(s) = c0 + c1 s + c2 s² (per coordinate).
    let (mut m, mut rx, mut ry) = ([[0.0f64; 3]; 3], [0.0f64; 3], [0.0f64; 3]);
    for (s, p) in &near {
        let b = [1.0, *s, s * s];
        for r in 0..3 {
            for c in 0..3 {
                m[r][c] += b[r] * b[c];
            }
            rx[r] += b[r] * p.x;
            ry[r] += b[r] * p.y;
        }
    }
    // Cramer's rule for the linear coefficient.
    let det3 = |a: [[f64; 3]; 3]| {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let d = det3(m);
    if d.abs() < 1e-18 {
        return fallback();
    }
    let with_col = |rhs: [f64; 3]| {
        let mut a = m;
        for r in 0..3 {
            a[r][1] = rhs[r];
        }
        det3(a) / d
    };
    v2(with_col(rx), with_col(ry))
        .unit()
        .unwrap_or_else(fallback)
}

fn rotate(t: V2, a: f64) -> V2 {
    let (s, c) = a.sin_cos();
    v2(t.x * c - t.y * s, t.x * s + t.y * c)
}

/// Golden-section search of `f` over `[lo, hi]`; the best (x, f(x)) seen.
fn golden<T>(lo: f64, hi: f64, steps: usize, mut f: impl FnMut(f64) -> (f64, T)) -> (f64, T) {
    const R: f64 = 0.618_033_988_749_895;
    let (mut a, mut b) = (lo, hi);
    let mut x1 = b - R * (b - a);
    let mut x2 = a + R * (b - a);
    let (mut f1, mut v1) = f(x1);
    let (mut f2, mut v2) = f(x2);
    for _ in 0..steps {
        if f1 <= f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            v2 = v1;
            x1 = b - R * (b - a);
            (f1, v1) = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            v1 = v2;
            x2 = a + R * (b - a);
            (f2, v2) = f(x2);
        }
    }
    if f1 <= f2 {
        (f1, v1)
    } else {
        (f2, v2)
    }
}

/// One cubic from sample `a` (leaving along `ta`) to sample `b` within `tol`,
/// arriving along the data's tangent at `b` (`fixed_end` pins it: a closed
/// loop's seam). Returns the trial and the tangent it arrives along.
fn reach_to(
    pts: &[V2],
    a: usize,
    b: usize,
    ta: V2,
    fixed_end: Option<V2>,
    tol: f64,
    reach: f64,
) -> Option<(Trial, V2)> {
    let tb = fixed_end.unwrap_or_else(|| -data_tangent(pts, b, reach));
    let t = trial(&pts[a..=b], ta, tb, tol, tol);
    (t.err <= tol).then_some((t, tb))
}

/// Fit the run `pts` (its first and last samples are anchors) with as few
/// cubics as reach within `tol`. `start` is the tangent leaving the first
/// sample (`None`: from the data); `end` pins the tangent arriving at the last
/// (pointing back into the curve). Each cubic comes with the parameters of
/// its samples, first included, last excluded.
fn fit_run_min(
    pts: &[V2],
    start: Option<V2>,
    end: Option<V2>,
    tol: f64,
    reach: f64,
) -> Vec<(Cubic, Vec<f64>)> {
    let n = pts.len();
    let mut out: Vec<(usize, usize, Cubic, Vec<f64>)> = Vec::new();
    if n < 2 {
        return Vec::new();
    }
    let mut a = 0;
    let mut ta = start.unwrap_or_else(|| data_tangent(pts, 0, reach));
    while a < n - 1 {
        let last = n - 1;
        let probe = |b: usize, ta: V2| {
            let fixed = if b == last { end } else { None };
            reach_to(pts, a, b, ta, fixed, tol, reach)
        };
        // The whole rest at once?
        let mut best = probe(last, ta).map(|f| (last, f));
        if best.is_none() {
            // Exponential, then binary search for the furthest end that fits.
            let mut lo = a + 1;
            let mut lo_fit: Option<(Trial, V2)> = None;
            let mut hi = last;
            let mut span = 8;
            loop {
                let b = (a + span).min(last);
                if b >= hi {
                    break;
                }
                match probe(b, ta) {
                    Some(f) => {
                        lo = b;
                        lo_fit = Some(f);
                        span *= 2;
                    }
                    None => {
                        hi = b;
                        break;
                    }
                }
            }
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                match probe(mid, ta) {
                    Some(f) => {
                        lo = mid;
                        lo_fit = Some(f);
                    }
                    None => hi = mid,
                }
            }
            best = Some(match lo_fit {
                Some(f) => (lo, f),
                None => {
                    // Two neighbouring samples: a straight step (the tangent is
                    // the step's own direction, so the next cubic stays G1).
                    let d = (pts[lo] - pts[a]).unit().unwrap_or(ta);
                    let c = Cubic::line(pts[a], pts[lo]);
                    (
                        lo,
                        (
                            Trial {
                                cubic: c,
                                u: vec![0.0, 1.0],
                                err: 0.0,
                            },
                            -d,
                        ),
                    )
                }
            });
        }
        let Some((b, (t, tb))) = best else { break };
        out.push((a, b, t.cubic, t.u));
        a = b;
        ta = -tb;
    }
    for _ in 0..POLISH_ROUNDS {
        polish_tangents(pts, &mut out, start.is_none(), end.is_none(), tol);
        let before = out.len();
        merge_neighbours(pts, &mut out, tol);
        if out.len() == before {
            break;
        }
    }
    out.into_iter()
        .map(|(_, _, c, mut u)| {
            u.pop();
            (c, u)
        })
        .collect()
}

/// The unit tangent a cubic leaves `c.0[0]` along (`start`) or arrives back
/// from at `c.0[3]` (pointing into the curve).
fn end_tangent_of(c: &Cubic, start: bool) -> Option<V2> {
    let [p0, p1, p2, p3] = c.0;
    if start {
        (p1 - p0).unit().or((p2 - p0).unit()).or((p3 - p0).unit())
    } else {
        (p2 - p3).unit().or((p1 - p3).unit()).or((p0 - p3).unit())
    }
}

/// Rounds of tangent polishing and merging after the greedy pass.
const POLISH_ROUNDS: usize = 3;

/// Re-aim every anchor's tangent (golden section over its angle) at the one
/// that keeps the worse of its two cubics closest; a run's free ends (a corner
/// or an open end) are aimed for their one cubic. The greedy pass fixed each
/// tangent looking one cubic ahead, and a better-aimed pair often merges.
fn polish_tangents(
    pts: &[V2],
    segs: &mut [(usize, usize, Cubic, Vec<f64>)],
    free_start: bool,
    free_end: bool,
    tol: f64,
) {
    let m = segs.len();
    for j in 0..=m {
        // Anchor j: the end of segs[j - 1] and the start of segs[j].
        let left = j.checked_sub(1);
        let right = (j < m).then_some(j);
        if (left.is_none() && !free_start) || (right.is_none() && !free_end) {
            continue;
        }
        let base = match (left, right) {
            (_, Some(r)) => end_tangent_of(&segs[r].2, true),
            (Some(l), None) => end_tangent_of(&segs[l].2, false).map(|t| -t),
            (None, None) => None,
        };
        let Some(base) = base else { continue };
        let fit_left = |t: V2| {
            left.and_then(|l| {
                let (a, b, c, _) = &segs[l];
                end_tangent_of(c, true).map(|ta| trial(&pts[*a..=*b], ta, -t, tol, 0.25 * tol))
            })
        };
        let fit_right = |t: V2| {
            right.and_then(|r| {
                let (a, b, c, _) = &segs[r];
                end_tangent_of(c, false).map(|tb| trial(&pts[*a..=*b], t, tb, tol, 0.25 * tol))
            })
        };
        let current = {
            let l = fit_left(base).map_or(0.0, |t| t.err);
            let r = fit_right(base).map_or(0.0, |t| t.err);
            l.max(r)
        };
        let (err, found) = golden(-POLISH_SPAN, POLISH_SPAN, POLISH_STEPS, |ang| {
            let t = rotate(base, ang);
            let (l, r) = (fit_left(t), fit_right(t));
            let e = l
                .as_ref()
                .map_or(0.0, |x| x.err)
                .max(r.as_ref().map_or(0.0, |x| x.err));
            (e, (l, r))
        });
        if err < current {
            let (l, r) = found;
            if let (Some(i), Some(t)) = (left, l) {
                segs[i].2 = t.cubic;
                segs[i].3 = t.u;
            }
            if let (Some(i), Some(t)) = (right, r) {
                segs[i].2 = t.cubic;
                segs[i].3 = t.u;
            }
        }
    }
}

/// Greedy leaves anchors the search could not look past; drop every one whose
/// two cubics refit as one (the outer tangents kept) within `tol`.
fn merge_neighbours(pts: &[V2], segs: &mut Vec<(usize, usize, Cubic, Vec<f64>)>, tol: f64) {
    let mut k = 0;
    while k + 1 < segs.len() {
        let (a, _, ca, _) = &segs[k];
        let (_, b, cb, _) = &segs[k + 1];
        let (a, b) = (*a, *b);
        let merged = end_tangent_of(ca, true)
            .zip(end_tangent_of(cb, false))
            .map(|(ta, tb)| trial(&pts[a..=b], ta, tb, tol, tol))
            .filter(|t| t.err <= tol);
        match merged {
            Some(t) => {
                segs[k] = (a, b, t.cubic, t.u);
                segs.remove(k + 1);
                k = k.saturating_sub(1);
            }
            None => k += 1,
        }
    }
}

/// How far (path units) corners and the data's tangents are measured over.
#[derive(Debug, Clone, Copy)]
struct Reach {
    corner: f64,
    tangent: f64,
}

/// The fitted subpath: path points, and for every sample the arc-length
/// fraction of the fitted path where it landed.
struct CurveFit {
    points: Vec<BlendPathPoint>,
    sample_t: Vec<f64>,
}

/// Fit `samples` (a closed loop's last sample is not repeated) with the
/// fewest cubics within `tol`, corners kept.
fn fit_curves(samples: &[V2], closed: bool, tol: f64, corner_angle: f64, reach: Reach) -> CurveFit {
    let n = samples.len();
    let corners = detect_corners(samples, closed, reach.corner, corner_angle.to_radians());
    // Runs of sample indices, anchors at both ends.
    let mut segs: Vec<(Cubic, Vec<f64>, Vec<usize>)> = Vec::new();
    let mut push_run = |idx: &[usize], start: Option<V2>, end: Option<V2>| {
        let pts: Vec<V2> = idx.iter().map(|&i| samples[i]).collect();
        let mut k = 0;
        for (c, u) in fit_run_min(&pts, start, end, tol, reach.tangent) {
            let m = u.len();
            segs.push((c, u, idx[k..k + m].to_vec()));
            k += m;
        }
    };
    if closed {
        // The loop still starts at sample 0 (where the path started). When that
        // is not a corner, the runs meeting there share one tangent.
        let seam = (!corners[0]).then(|| {
            let ring: Vec<V2> = (0..=2 * n).map(|k| samples[k % n]).collect();
            data_tangent(&ring, n, reach.tangent)
        });
        let order: Vec<usize> = (0..=n).map(|k| k % n).collect();
        let mut a = 0;
        for k in 1..=n {
            if k == n || corners[k] {
                let start = if a == 0 { seam } else { None };
                let end = if k == n { seam.map(|t| -t) } else { None };
                push_run(&order[a..=k], start, end);
                a = k;
            }
        }
    } else {
        let mut a = 0;
        for (k, &corner) in corners.iter().enumerate().skip(1) {
            if corner || k == n - 1 {
                let idx: Vec<usize> = (a..=k).collect();
                push_run(&idx, None, None);
                a = k;
            }
        }
    }

    let fit_segs: Vec<FitSeg> = segs.iter().map(|(c, _, _)| FitSeg::Cubic(*c)).collect();
    let lens: Vec<f64> = fit_segs.iter().map(|s| shared_arc_len(s, 1.0)).collect();
    let total: f64 = lens.iter().sum();
    let mut sample_t = vec![0.0; n];
    let mut acc = 0.0;
    for (k, (_, u, idx)) in segs.iter().enumerate() {
        for (&i, &x) in idx.iter().zip(u) {
            if total > 0.0 {
                sample_t[i] = ((acc + shared_arc_len(&fit_segs[k], x)) / total).clamp(0.0, 1.0);
            }
        }
        acc += lens[k];
    }
    if !closed && n > 0 {
        sample_t[n - 1] = 1.0;
    }
    let mut points = Vec::with_capacity(segs.len() + 1);
    if let Some(first) = fit_segs.first() {
        let p = first.start();
        points.push(BlendPathPoint::Move { x: p.x, y: p.y });
    }
    points.extend(fit_segs.iter().map(FitSeg::to_path_point));
    CurveFit { points, sample_t }
}

/// Curves whose handles lie on their chord (the fitter's straight runs) as lines.
fn straighten(points: Vec<BlendPathPoint>) -> Vec<BlendPathPoint> {
    let mut cur = v2(0.0, 0.0);
    points
        .into_iter()
        .map(|p| {
            let end = anchor(&p);
            let out = match p {
                BlendPathPoint::Curve { x, y, cp1, cp2 } => {
                    let d = end - cur;
                    let l2 = d.len2();
                    let on_chord = |c: (f64, f64)| {
                        let q = v2(c.0, c.1) - cur;
                        let t = if l2 > 0.0 { q.dot(d) / l2 } else { -1.0 };
                        (0.0..=1.0).contains(&t) && q.cross(d).abs() <= 1e-7 * l2 + 1e-12
                    };
                    if on_chord(cp1) && on_chord(cp2) {
                        BlendPathPoint::Line { x, y }
                    } else {
                        BlendPathPoint::Curve { x, y, cp1, cp2 }
                    }
                }
                other => other,
            };
            cur = end;
            out
        })
        .collect()
}

/// Simplify a path: every subpath refitted within `tolerance`, or kept.
pub fn simplify_path(
    points: &[BlendPathPoint],
    closed: bool,
    opts: &SimplifyOptions,
) -> SimplifyOutput {
    simplify_path_budgeted(points, closed, opts, MAX_SAMPLES_TOTAL)
}

/// [`simplify_path`] with `max_samples` dense samples over all subpaths.
pub(crate) fn simplify_path_budgeted(
    points: &[BlendPathPoint],
    closed: bool,
    opts: &SimplifyOptions,
    max_samples: usize,
) -> SimplifyOutput {
    let opts = SimplifyOptions {
        tolerance: if opts.tolerance.is_finite() {
            opts.tolerance.clamp(0.01, 1000.0)
        } else {
            1.0
        },
        corner_angle: if opts.corner_angle.is_finite() {
            opts.corner_angle.clamp(1.0, 179.0)
        } else {
            DEFAULT_CORNER_ANGLE
        },
        straight_lines: opts.straight_lines,
    };
    let subs = split_subpaths(points);
    let mut out = SimplifyOutput {
        anchors_before: points.len(),
        ..Default::default()
    };
    let total: f64 = subs
        .iter()
        .filter(|sub| sub.len() >= 3)
        .map(|sub| subpath_length(sub, closed, opts.tolerance))
        .sum();
    let min_step = total / max_samples.max(1) as f64;
    for sub in &subs {
        let r = simplify_subpath(sub, closed, &opts, min_step);
        out.simplified |= r.result.simplified;
        out.max_deviation = out.max_deviation.max(r.deviation);
        out.points.extend(r.points);
        out.subpaths.push(r.result);
    }
    out.anchors_after = out.points.len();
    out
}

fn error(code: &str, message: &str) -> String {
    json!({"status": "error", "code": code, "message": message}).to_string()
}

/// JSON in / JSON out; always an envelope, never panics. See the module docs.
pub fn simplify_path_json(request_json: &str) -> String {
    if request_json.len() > 64 * 1024 * 1024 {
        return error("request_too_large", "request exceeds 64 MiB");
    }
    let value: Value = match crate::value::from_str(request_json) {
        Ok(v) => v,
        Err(e) => return error("invalid_json", &format!("request is not valid JSON: {e}")),
    };
    let Some(obj) = value.as_object() else {
        return error("invalid_request", "request must be an object");
    };
    let Some(raw) = obj.get("points").and_then(Value::as_array) else {
        return error("invalid_request", "points must be an array");
    };
    if raw.len() > MAX_SIMPLIFY_POINTS {
        return error("too_many_points", "at most 100000 path points");
    }
    let mut points = Vec::with_capacity(raw.len());
    for p in raw {
        match BlendPathPoint::from_json(p) {
            Some(p) => points.push(p),
            None => return error("invalid_request", "points contains a malformed PathPoint"),
        }
    }
    if points.is_empty() {
        return json!({"status": "empty"}).to_string();
    }
    let d = SimplifyOptions::default();
    let opts = SimplifyOptions {
        tolerance: obj
            .get("tolerance")
            .and_then(Value::as_f64)
            .unwrap_or(d.tolerance),
        corner_angle: obj
            .get("cornerAngle")
            .and_then(Value::as_f64)
            .unwrap_or(d.corner_angle),
        straight_lines: obj
            .get("straightLines")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    };
    let closed = obj.get("closed").and_then(Value::as_bool).unwrap_or(false);
    let out = simplify_path(&points, closed, &opts);
    let subpaths: Vec<Value> = out
        .subpaths
        .iter()
        .map(|s| {
            let mut v = json!({ "simplified": s.simplified });
            if let (Some(ss), Some(st)) = (&s.sample_s, &s.sample_t) {
                v["sampleS"] = json!(ss);
                v["sampleT"] = json!(st);
            }
            v
        })
        .collect();
    json!({
        "status": "ok",
        "points": out.points.iter().map(BlendPathPoint::to_json).collect::<Vec<_>>(),
        "simplified": out.simplified,
        "anchorsBefore": out.anchors_before,
        "anchorsAfter": out.anchors_after,
        "maxDeviation": out.max_deviation,
        "subpaths": subpaths,
    })
    .to_string()
}

#[cfg(test)]
#[path = "simplify_tests.rs"]
mod tests;
