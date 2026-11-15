//! Curve fitting: sampled points → cubic Bézier path (brush plan §5, P5).
//!
//! Schneider's algorithm ("An Algorithm for Automatically Fitting Digitized
//! Curves", Graphics Gems, 1990): chord-length parameterisation, least-squares
//! handle lengths along fixed end tangents, Newton–Raphson reparameterisation,
//! and splitting at the worst point with a shared (G1) tangent. Two callers:
//!
//! - [`fit_points`] — Pencil / Paintbrush input and Simplify. Adds corner
//!   detection (sharp turns become cusps, not rounded blobs) and returns, for
//!   every input sample, the arc-length fraction of the fitted path it landed
//!   on, so ink channels (`InkChannels.t`, brushes.md §4.5) recorded against the
//!   raw samples can be re-expressed over the fitted curve.
//! - [`fit_polyline`] — the stroke outliner fits its dense offset polylines back
//!   to cubics between forced breaks (joins, caps, trimmed loops).
//!
//! Everything here is total: non-finite input is dropped, degenerate runs fall
//! back to lines, recursion is an explicit stack with a hard budget. The wasm
//! build is `panic = "abort"`, so "no panics on any input" is load-bearing
//! (see `fit_fuzz_no_panic` in the tests).

use crate::value::{json, Value};

use super::vec2::{v2, Cubic, V2};
use crate::blend::BlendPathPoint;

/// Largest input `fit_points_json` accepts.
pub const MAX_FIT_POINTS: usize = 100_000;
/// Hard cap on emitted segments per fit (a runaway split can never allocate
/// more than this; runs past it are emitted as lines).
const MAX_SEGMENTS: usize = 50_000;
const REPARAM_ITERATIONS: usize = 4;
/// Coordinates beyond this are treated as garbage (dropped) so products stay finite.
pub const MAX_COORD: f64 = 1e9;

/// One fitted segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FitSeg {
    Line(V2, V2),
    Cubic(Cubic),
}

impl FitSeg {
    pub fn start(&self) -> V2 {
        match self {
            FitSeg::Line(a, _) => *a,
            FitSeg::Cubic(c) => c.0[0],
        }
    }
    pub fn end(&self) -> V2 {
        match self {
            FitSeg::Line(_, b) => *b,
            FitSeg::Cubic(c) => c.0[3],
        }
    }
    pub fn as_cubic(&self) -> Cubic {
        match self {
            FitSeg::Line(a, b) => Cubic::line(*a, *b),
            FitSeg::Cubic(c) => *c,
        }
    }
    pub fn arc_len_to(&self, u: f64) -> f64 {
        match self {
            FitSeg::Line(a, b) => a.dist(*b) * u,
            FitSeg::Cubic(c) => c.arc_len_to(u),
        }
    }
    pub fn to_path_point(&self) -> BlendPathPoint {
        match self {
            FitSeg::Line(_, b) => BlendPathPoint::Line { x: b.x, y: b.y },
            FitSeg::Cubic(c) => BlendPathPoint::Curve {
                x: c.0[3].x,
                y: c.0[3].y,
                cp1: (c.0[1].x, c.0[1].y),
                cp2: (c.0[2].x, c.0[2].y),
            },
        }
    }
}

/// Where each fitted input point landed: segment index and parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitParam {
    pub seg: usize,
    pub u: f64,
}

// ---------------------------------------------------------------------------
// Core Schneider fit of one run between two fixed end tangents
// ---------------------------------------------------------------------------

pub(crate) fn chord_params(pts: &[V2]) -> Vec<f64> {
    let mut u = Vec::with_capacity(pts.len());
    u.push(0.0);
    let mut acc = 0.0;
    for w in pts.windows(2) {
        acc += w[0].dist(w[1]);
        u.push(acc);
    }
    if acc > 0.0 {
        for x in u.iter_mut() {
            *x /= acc;
        }
    } else {
        let n = (pts.len() - 1).max(1) as f64;
        for (i, x) in u.iter_mut().enumerate() {
            *x = i as f64 / n;
        }
    }
    u
}

/// Least-squares handle lengths along `t1` (out of p0) and `t2` (back into p3).
pub(crate) fn generate(pts: &[V2], u: &[f64], t1: V2, t2: V2, max_alpha: f64) -> Cubic {
    let p0 = pts[0];
    let p3 = pts[pts.len() - 1];
    let (mut c00, mut c01, mut c11, mut x0, mut x1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (p, &t) in pts.iter().zip(u) {
        let mt = 1.0 - t;
        let b0 = mt * mt * mt;
        let b1 = 3.0 * mt * mt * t;
        let b2 = 3.0 * mt * t * t;
        let b3 = t * t * t;
        let a0 = t1 * b1;
        let a1 = t2 * b2;
        c00 += a0.dot(a0);
        c01 += a0.dot(a1);
        c11 += a1.dot(a1);
        let tmp = *p - (p0 * (b0 + b1) + p3 * (b2 + b3));
        x0 += a0.dot(tmp);
        x1 += a1.dot(tmp);
    }
    let det = c00 * c11 - c01 * c01;
    let seg_len = p0.dist(p3);
    let eps = 1e-6 * seg_len.max(1e-9);
    let (mut al, mut ar) = if det.abs() > 1e-12 {
        ((x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det)
    } else {
        (0.0, 0.0)
    };
    if !(al.is_finite() && ar.is_finite()) || al < eps || ar < eps {
        // Wu/Barsky heuristic: fall back to a third of the chord.
        al = seg_len / 3.0;
        ar = seg_len / 3.0;
    }
    al = al.min(max_alpha);
    ar = ar.min(max_alpha);
    Cubic([p0, p0 + t1 * al, p3 + t2 * ar, p3])
}

fn max_error(c: &Cubic, pts: &[V2], u: &[f64]) -> (f64, usize) {
    let mut worst = 0.0;
    let mut at = pts.len() / 2;
    for i in 1..pts.len().saturating_sub(1) {
        let d = (c.eval(u[i]) - pts[i]).len2();
        if d >= worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

pub(crate) fn reparameterize(c: &Cubic, pts: &[V2], u: &mut [f64]) {
    for (p, t) in pts.iter().zip(u.iter_mut()) {
        let q = c.eval(*t) - *p;
        let d1 = c.d1(*t);
        let d2 = c.d2(*t);
        let den = d1.dot(d1) + q.dot(d2);
        if den.abs() > 1e-12 {
            let next = *t - q.dot(d1) / den;
            if next.is_finite() {
                *t = next.clamp(0.0, 1.0);
            }
        }
    }
}

/// Fit `pts` (≥ 2 points, first/last fixed) with end tangents `t1`, `t2`
/// (`t2` points from the last point back into the curve). Appends to `out`, and
/// to `params` the (segment, u) of every point except the run's last.
fn fit_run(
    pts: &[V2],
    t1: V2,
    t2: V2,
    tol2: f64,
    out: &mut Vec<FitSeg>,
    params: &mut Vec<FitParam>,
) {
    // Explicit stack of (first, last, t1, t2); pushed right-then-left so runs
    // are emitted in order.
    let mut stack: Vec<(usize, usize, V2, V2)> = vec![(0, pts.len() - 1, t1, t2)];
    while let Some((a, b, ta, tb)) = stack.pop() {
        let run = &pts[a..=b];
        let emit_line = |out: &mut Vec<FitSeg>, params: &mut Vec<FitParam>| {
            let seg = out.len();
            let u = chord_params(run);
            out.push(FitSeg::Line(run[0], run[run.len() - 1]));
            for &x in &u[..u.len() - 1] {
                params.push(FitParam { seg, u: x });
            }
        };
        if out.len() >= MAX_SEGMENTS {
            emit_line(out, params);
            continue;
        }
        let chord = run[0].dist(run[run.len() - 1]);
        if run.len() == 2 {
            let d = run[1] - run[0];
            let straight = chord < 1e-12
                || (d.cross(ta).abs() <= 1e-9 * chord && d.cross(tb).abs() <= 1e-9 * chord);
            if straight {
                emit_line(out, params);
            } else {
                let seg = out.len();
                let al = chord / 3.0;
                out.push(FitSeg::Cubic(Cubic([
                    run[0],
                    run[0] + ta * al,
                    run[1] + tb * al,
                    run[1],
                ])));
                params.push(FitParam { seg, u: 0.0 });
            }
            continue;
        }
        let poly_len: f64 = run.windows(2).map(|w| w[0].dist(w[1])).sum();
        if poly_len < 1e-12 {
            emit_line(out, params);
            continue;
        }
        let mut u = chord_params(run);
        let mut c = generate(run, &u, ta, tb, poly_len);
        let (mut err, mut split) = max_error(&c, run, &u);
        if err > tol2 && err < 64.0 * tol2 {
            for _ in 0..REPARAM_ITERATIONS {
                reparameterize(&c, run, &mut u);
                c = generate(run, &u, ta, tb, poly_len);
                let (e, s) = max_error(&c, run, &u);
                err = e;
                split = s;
                if err <= tol2 {
                    break;
                }
            }
        }
        if err <= tol2 {
            let seg = out.len();
            out.push(FitSeg::Cubic(c));
            for &x in &u[..u.len() - 1] {
                params.push(FitParam { seg, u: x });
            }
            continue;
        }
        let split = split.clamp(1, run.len() - 2);
        let center = (run[split - 1] - run[split + 1])
            .unit()
            .or((run[split] - run[split + 1]).unit())
            .or((run[split - 1] - run[split]).unit())
            .unwrap_or(-ta);
        let s = a + split;
        stack.push((s, b, -center, tb));
        stack.push((a, s, ta, center));
    }
}

/// Direction from `pts[i]` towards the point `reach` away along the run in
/// direction `step` (±1), falling back to the adjacent point.
fn end_tangent(pts: &[V2], i: usize, forward: bool, reach: f64) -> Option<V2> {
    let n = pts.len();
    let mut best = None;
    let mut j = i;
    loop {
        j = if forward {
            if j + 1 >= n {
                break;
            }
            j + 1
        } else {
            if j == 0 {
                break;
            }
            j - 1
        };
        let d = pts[j] - pts[i];
        if let Some(u) = d.unit() {
            best = Some(u);
            if d.len() >= reach {
                break;
            }
        }
    }
    best
}

/// Fit a run of points between two breaks, choosing end tangents from the data.
fn fit_open_run(
    pts: &[V2],
    reach: f64,
    tol2: f64,
    out: &mut Vec<FitSeg>,
    params: &mut Vec<FitParam>,
) {
    let n = pts.len();
    if n < 2 {
        return;
    }
    let chord_dir = (pts[n - 1] - pts[0]).unit().unwrap_or(v2(1.0, 0.0));
    let t1 = end_tangent(pts, 0, true, reach).unwrap_or(chord_dir);
    let t2 = end_tangent(pts, n - 1, false, reach).unwrap_or(-chord_dir);
    fit_run(pts, t1, t2, tol2, out, params);
}

// ---------------------------------------------------------------------------
// fit_polyline — the outliner's entry point
// ---------------------------------------------------------------------------

/// Fit a polyline back to cubics. `breaks[i]` forces a corner at point `i`
/// (the curve passes through it with independent tangents on each side).
/// `closed` treats the polyline as a loop (the last point connects back to
/// the first; do not repeat it). `tolerance <= 0` returns the polyline as
/// lines. Consecutive duplicate points are ignored.
pub fn fit_polyline(points: &[V2], breaks: &[bool], closed: bool, tolerance: f64) -> Vec<FitSeg> {
    let mut pts: Vec<V2> = Vec::with_capacity(points.len() + 1);
    let mut brk: Vec<bool> = Vec::with_capacity(points.len() + 1);
    for (i, p) in points.iter().enumerate() {
        if !p.is_finite() {
            continue;
        }
        let b = breaks.get(i).copied().unwrap_or(false);
        if let Some(last) = pts.last() {
            if last.dist(*p) <= 1e-9 {
                let k = brk.len() - 1;
                brk[k] |= b;
                continue;
            }
        }
        pts.push(*p);
        brk.push(b);
    }
    if closed && pts.len() > 1 && pts[0].dist(pts[pts.len() - 1]) <= 1e-9 {
        let b = brk.pop().unwrap_or(false);
        pts.pop();
        brk[0] |= b;
    }
    let mut out = Vec::new();
    if pts.len() < 2 {
        return out;
    }
    if tolerance <= 0.0 || !tolerance.is_finite() {
        for w in pts.windows(2) {
            out.push(FitSeg::Line(w[0], w[1]));
        }
        if closed {
            out.push(FitSeg::Line(pts[pts.len() - 1], pts[0]));
        }
        return out;
    }
    // A long chord between much shorter ones is a straight run of the path (the outliner samples curves densely
    // and lines only at their ends): it stays a line, its own segment, instead of a cubic that bows off it to take
    // in the curve beside it (the outlined bell's flat bottom bowed 0.04 px).
    let n = pts.len();
    let chord = |i: usize| pts[i].dist(pts[(i + 1) % n]);
    let chords = if closed { n } else { n - 1 };
    for i in 0..chords {
        let len = chord(i);
        let before = if i > 0 {
            chord(i - 1)
        } else if closed {
            chord(n - 1)
        } else {
            0.0
        };
        let after = if i + 1 < chords {
            chord(i + 1)
        } else if closed {
            chord(0)
        } else {
            0.0
        };
        if len > 8.0 * tolerance && len >= 4.0 * before.max(after) {
            brk[i] = true;
            brk[(i + 1) % n] = true;
        }
    }
    // Likewise a run of collinear chords (a line sampled along its length). A ring is scanned from a chord that
    // turns from the one before it, so no run straddles the seam.
    let dir = |i: usize| (pts[(i + 1) % n] - pts[i]).unit();
    let same = |a: Option<V2>, b: Option<V2>| match (a, b) {
        (Some(a), Some(b)) => a.dot(b) > 0.0 && a.cross(b).abs() < 1e-6,
        _ => false,
    };
    let first = if closed {
        (0..n)
            .find(|&i| !same(dir((i + n - 1) % n), dir(i)))
            .unwrap_or(0)
    } else {
        0
    };
    // A run one chord short of a forced break takes that chord in when the break sits a hair off its line (a closed
    // stroke's seam is offset along its own, very slightly kinked, bisector): else the chord is a line of its own
    // (the lucide heart's seam was two anchors 0.25 apart). Only at a break: absorbing any near-collinear chord
    // moves where a line meets the curve after it (the bell gained a point).
    let forced = brk.clone();
    let near_line = |a: V2, b: V2, q: V2| match (b - a).unit() {
        Some(u) => (q - a).cross(u).abs() <= 0.01 * tolerance,
        None => false,
    };
    // Chords `i` and `i + 1` (scan order) are collinear: the absorbed chord must belong to no run of its own.
    let paired = |i: usize| i + 1 < chords && same(dir((first + i) % n), dir((first + i + 1) % n));
    let mut j = 0;
    while j < chords {
        let s = (first + j) % n;
        let d0 = dir(s);
        let mut k = j + 1;
        while k < chords && same(d0, dir((first + k) % n)) {
            k += 1;
        }
        let e = (first + k) % n;
        if k - j >= 2 && pts[s].dist(pts[e]) > 8.0 * tolerance {
            let (mut a, mut b) = (s, e);
            if k < chords && !forced[e] && !paired(k) {
                let e2 = (first + k + 1) % n;
                if forced[e2] && near_line(pts[s], pts[e2], pts[e]) {
                    b = e2;
                }
            }
            if j > 0 && !forced[s] && (j < 2 || !paired(j - 2)) {
                let s2 = (first + j - 1) % n;
                if forced[s2] && near_line(pts[s2], pts[b], pts[s]) {
                    a = s2;
                }
            }
            brk[a] = true;
            brk[b] = true;
        }
        j = k;
    }
    // The greedy fit spends the whole tolerance at the samples; keep a margin for the curve between them.
    let tol2 = (0.9 * tolerance) * (0.9 * tolerance);
    let mut scratch = Vec::new();
    if closed {
        // Rotate so the loop starts at a break; a loop with none is split at
        // point 0 with a shared tangent.
        let n = pts.len();
        let start = brk.iter().position(|&b| b);
        let mut ring: Vec<V2> = Vec::with_capacity(n + 1);
        let mut ring_brk: Vec<bool> = Vec::with_capacity(n + 1);
        let s = start.unwrap_or(0);
        for k in 0..=n {
            ring.push(pts[(s + k) % n]);
            ring_brk.push(brk[(s + k) % n]);
        }
        if start.is_none() {
            let t = (pts[1] - pts[n - 1]).unit().unwrap_or(v2(1.0, 0.0));
            fit_run_greedy(&ring, t, -t, tol2, &mut out);
            return out;
        }
        fit_runs_between_breaks(&ring, &ring_brk, tol2, &mut out, &mut scratch);
        return out;
    }
    brk[0] = true;
    let last = brk.len() - 1;
    brk[last] = true;
    fit_runs_between_breaks(&pts, &brk, tol2, &mut out, &mut scratch);
    out
}

/// One cubic through `run` (≥ 3 points) with end tangents `ta`, `tb`, if one lands within `tol2` (squared) of every
/// point: least squares plus Newton reparameterisation, as [`fit_run`] tries before it splits.
fn fit_one(run: &[V2], ta: V2, tb: V2, tol2: f64) -> Option<Cubic> {
    let poly_len: f64 = run.windows(2).map(|w| w[0].dist(w[1])).sum();
    if poly_len < 1e-12 {
        return None;
    }
    let mut u = chord_params(run);
    let mut c = generate(run, &u, ta, tb, poly_len);
    let (mut err, _) = max_error(&c, run, &u);
    // Worth reparameterising when the miss is small against the tolerance, or against the run itself: on a long run
    // chord-length parameters alone put a good cubic's points a few px off where they belong (a stroked r400
    // circle's quarter measured 2.1 px off before Newton, 0.09 after), which is no sign the shape doesn't fit.
    if err > tol2 && (err < 64.0 * tol2 || err < 1e-4 * poly_len * poly_len) {
        for _ in 0..REPARAM_ITERATIONS {
            reparameterize(&c, run, &mut u);
            c = generate(run, &u, ta, tb, poly_len);
            err = max_error(&c, run, &u).0;
            if err <= tol2 {
                break;
            }
        }
    }
    if err > tol2 {
        return None;
    }
    // Between samples too: a long cubic can bulge off a long straight chord (samples are sparse there) that the
    // sample points alone don't see.
    let bulge = run.windows(2).zip(u.windows(2)).any(|(p, t)| {
        p[0].dist(p[1]) > 4.0 * tol2.sqrt()
            && (c.eval(0.5 * (t[0] + t[1])) - p[0].lerp(p[1], 0.5)).len2() > tol2
    });
    (!bulge).then_some(c)
}

/// The curve's forward direction at interior point `i`: the central difference, else a one-sided one.
fn through_tangent(pts: &[V2], i: usize) -> V2 {
    (pts[i + 1] - pts[i - 1])
        .unit()
        .or((pts[i + 1] - pts[i]).unit())
        .or((pts[i] - pts[i - 1]).unit())
        .unwrap_or(v2(1.0, 0.0))
}

/// Most a stroke outline's cubic turns: a quarter circle and a little. A cubic can span a half circle within a
/// loose tolerance, but its apex then sits off the true extreme (a blob's round cap came out 0.2 short of its
/// end); Illustrator's outlines place an anchor every quarter turn.
const MAX_PIECE_TURN: f64 = 100.0 * std::f64::consts::PI / 180.0;

/// [`fit_run`] for the stroke outliner: as few cubics as fit. Where Schneider's split at the worst point halves a
/// run unevenly (a stroked circle's half became arcs of 22°, 21°, 113° and 24°), this takes the longest prefix one
/// cubic fits (galloping then binary search on its end; at most [`MAX_PIECE_TURN`]), carries on from there with a
/// G1 tangent, and then, when the same number of pieces of equal length fit too, uses those instead (a circle's
/// quarters, a round cap's two halves meeting at its apex).
fn fit_run_greedy(pts: &[V2], t1: V2, t2: V2, tol2: f64, out: &mut Vec<FitSeg>) {
    let last = pts.len() - 1;
    if last < 2 {
        let mut scratch = Vec::new();
        fit_run(pts, t1, t2, tol2, out, &mut scratch);
        return;
    }
    // Net turning so far (radians, signed: jitter where a side collapses cancels out) at each point, and arc length.
    let mut turn = vec![0.0; pts.len()];
    let mut arc = vec![0.0; pts.len()];
    // Chords shorter than half the tolerance don't steer (a side collapsing onto its evolute jitters there).
    let min_chord = 0.5 * tol2.sqrt();
    let mut dir: Option<V2> = None;
    for i in 1..=last {
        let d = pts[i] - pts[i - 1];
        arc[i] = arc[i - 1] + d.len();
        turn[i] = turn[i - 1];
        if d.len() >= min_chord {
            let d1 = d.unit().unwrap_or(v2(1.0, 0.0));
            if let Some(d0) = dir {
                // The turn into chord i happens at point i - 1.
                turn[i - 1] += d0.cross(d1).atan2(d0.dot(d1));
                turn[i] = turn[i - 1];
            }
            dir = Some(d1);
        }
    }
    let start_tangent = |a: usize| if a == 0 { t1 } else { through_tangent(pts, a) };
    let forward = |b: usize| {
        if b == last {
            -t2
        } else {
            through_tangent(pts, b)
        }
    };
    let piece = |a: usize, b: usize| -> Option<Cubic> {
        let (ta, t) = (start_tangent(a), forward(b));
        if b - a < 2 {
            let chord = pts[a].dist(pts[b]);
            return Some(Cubic([
                pts[a],
                pts[a] + ta * (chord / 3.0),
                pts[b] - t * (chord / 3.0),
                pts[b],
            ]));
        }
        if (turn[b] - turn[a]).abs() > MAX_PIECE_TURN {
            return None;
        }
        fit_one(&pts[a..=b], ta, -t, tol2)
    };
    let mut ends: Vec<usize> = Vec::new();
    let mut a = 0;
    while a < last {
        if ends.len() + out.len() >= MAX_SEGMENTS {
            ends.push(last);
            break;
        }
        // Gallop out from `a` (each probe costs its length, so a long path of short pieces stays near-linear),
        // then bisect between the last end that fits (`lo`) and the first that doesn't (`hi`).
        let mut lo = a + 1;
        let mut hi = None;
        let mut step = 2;
        loop {
            let b = (a + step).min(last);
            if piece(a, b).is_some() {
                lo = b;
                if b == last {
                    break;
                }
                step *= 2;
            } else {
                hi = Some(b);
                break;
            }
        }
        if let Some(mut hi) = hi {
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if piece(a, mid).is_some() {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
        }
        ends.push(lo);
        a = lo;
    }
    // The same count at equal arc lengths, when every piece still fits.
    let k = ends.len();
    if k > 1 && k < 64 {
        let mut even: Vec<usize> = (1..k)
            .map(|j| {
                let target = arc[last] * j as f64 / k as f64;
                let i = arc.partition_point(|&s| s < target).clamp(1, last - 1);
                // A near tie (a symmetric cap puts the target midway) goes to the earlier sample, so rounding noise
                // (native vs WASM maths) can't pick differently.
                if (arc[i - 1] - target).abs() <= (arc[i] - target).abs() + 1e-9 * arc[last] {
                    i - 1
                } else {
                    i
                }
            })
            .collect();
        even.push(last);
        even.dedup();
        let mut prev = 0;
        let ok = even.len() == k
            && even.iter().all(|&e| {
                let fits = e > prev && piece(prev, e).is_some();
                prev = e;
                fits
            });
        if ok {
            ends = even;
        }
    }
    let mut prev = 0;
    for e in ends {
        match piece(prev, e) {
            Some(c) => out.push(straightened(c, 0.01 * tol2.sqrt())),
            None => {
                let mut scratch = Vec::new();
                fit_run(
                    &pts[prev..=e],
                    start_tangent(prev),
                    -forward(e),
                    tol2,
                    out,
                    &mut scratch,
                );
            }
        }
        prev = e;
    }
}

/// A cubic whose handles lie on its chord (within `eps`: a straight run) as the line it draws.
fn straightened(c: Cubic, eps: f64) -> FitSeg {
    let [p0, p1, p2, p3] = c.0;
    let d = p3 - p0;
    let len = d.len();
    let on = |q: V2| {
        let t = (q - p0).dot(d) / (len * len);
        (q - p0).cross(d).abs() <= eps * len && (-1e-9..=1.0 + 1e-9).contains(&t)
    };
    if len > 0.0 && on(p1) && on(p2) {
        FitSeg::Line(p0, p3)
    } else {
        FitSeg::Cubic(c)
    }
}

/// The tangent at `p0` of the parabola through `p0`, `p1`, `p2` (chord-length parameters), pointing towards `p1`:
/// on a sampled arc it is off by O(θ²) where the chord `p0 → p1` is off by half the sample turn, which kept a
/// run's cubic from meeting a tight tolerance (a stroked circle's quarter split in two).
fn parabola_tangent(p0: V2, p1: V2, p2: V2) -> Option<V2> {
    let chord = (p1 - p0).unit()?;
    let (s1, s2) = (p0.dist(p1), p1.dist(p2));
    if s2 <= 1e-12 {
        return Some(chord);
    }
    let d = (p1 - p0) * ((s1 + s2) / (s1 * s2)) - (p2 - p0) * (s1 / (s2 * (s1 + s2)));
    // The estimate only refines the chord: a turn past ~30° between samples is a corner the run shouldn't bend
    // through, so the chord stands.
    d.unit().filter(|t| t.dot(chord) > 0.87).or(Some(chord))
}

fn fit_runs_between_breaks(
    pts: &[V2],
    brk: &[bool],
    tol2: f64,
    out: &mut Vec<FitSeg>,
    params: &mut Vec<FitParam>,
) {
    let mut a = 0;
    for i in 1..pts.len() {
        if brk[i] || i == pts.len() - 1 {
            let run = &pts[a..=i];
            let n = run.len();
            if n >= 3 {
                let chord = (run[n - 1] - run[0]).unit().unwrap_or(v2(1.0, 0.0));
                let t1 = parabola_tangent(run[0], run[1], run[2]).unwrap_or(chord);
                let t2 = parabola_tangent(run[n - 1], run[n - 2], run[n - 3]).unwrap_or(-chord);
                fit_run_greedy(run, t1, t2, tol2, out);
            } else {
                fit_open_run(run, 0.0, tol2, out, params);
            }
            a = i;
        }
    }
}

// ---------------------------------------------------------------------------
// fit_points — Pencil / Paintbrush / Simplify
// ---------------------------------------------------------------------------

/// Options for [`fit_points`].
#[derive(Debug, Clone, Copy)]
pub struct FitOptions {
    /// Maximum distance (path units) between an input sample and the fitted
    /// curve. Illustrator's Pencil "Fidelity" spans ~0.5 (accurate) to ~20
    /// (smooth) px; this is that number.
    pub tolerance: f64,
    /// A turn sharper than this (degrees), measured over a neighbourhood of
    /// about `2 × tolerance`, becomes a corner.
    pub corner_angle: f64,
    /// Fit a closed loop (first and last samples join).
    pub closed: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            tolerance: 1.0,
            corner_angle: 70.0,
            closed: false,
        }
    }
}

/// Result of [`fit_points`].
#[derive(Debug, Clone)]
pub struct FitOutput {
    /// Path points: one `Move`, then `Line`/`Curve`. A closed fit's last
    /// segment ends on the first point; the caller sets `closed`.
    pub points: Vec<BlendPathPoint>,
    pub closed: bool,
    /// For every INPUT sample (same length and order as the input, including
    /// dropped duplicates / non-finite samples, which inherit a neighbour's
    /// value): the arc-length fraction 0..1 of the fitted path where that sample
    /// landed. Non-decreasing for open fits. Use it to re-express ink channels
    /// recorded against the samples over the fitted curve.
    pub sample_t: Vec<f64>,
    /// Indices (into the input) of the samples that became corners.
    pub corners: Vec<usize>,
}

/// Longest walk (in samples) any corner test takes from one point, so the
/// detector is linear in the input whatever its spacing.
const CORNER_WALK: isize = 64;

/// Corners: the turn between the points `reach` away on either side (at least
/// one sample) exceeds `min_turn`, and either both chords at the sample are
/// long (≥ `2 × reach`, i.e. 4 × tolerance: a sparse polygon — agents send 3–6
/// point shapes and expect corners) or the turn is CONCENTRATED at the sample;
/// then it must be the sharpest within `reach` (non-maximum suppression).
///
/// Concentration tells a vertex from a sparsely sampled arc: at a vertex the
/// legs are straight, so the turn measured twice as far out (at least two
/// samples) is about the same; along an arc it grows with the stencil. So a
/// fast pointer stroke (10–30 px between samples) keeps round arc tops, while
/// a V drawn with sparse samples is still a corner.
pub(crate) fn detect_corners(pts: &[V2], closed: bool, reach: f64, min_turn: f64) -> Vec<bool> {
    let n = pts.len();
    let at = |i: isize| -> V2 { pts[i.rem_euclid(n as isize) as usize] };
    let walk = CORNER_WALK.min(n as isize);
    // The point at least `r` away and at least `min_steps` samples away.
    let reach_from = |i: usize, dir: isize, r: f64, min_steps: isize| -> Option<V2> {
        let p = pts[i];
        let mut found = None;
        for k in 1..=walk {
            let j = i as isize + dir * k;
            if !closed && (j < 0 || j >= n as isize) {
                break;
            }
            if closed && j.rem_euclid(n as isize) as usize == i {
                break;
            }
            let q = at(j);
            found = Some((q, k));
            if k >= min_steps && q.dist(p) >= r {
                break;
            }
        }
        found.filter(|(_, k)| *k >= min_steps).map(|(q, _)| q)
    };
    let turn_at = |i: usize, r: f64, min_steps: isize| -> Option<f64> {
        let p = pts[i];
        let (b, f) = (
            reach_from(i, -1, r, min_steps)?,
            reach_from(i, 1, r, min_steps)?,
        );
        let (d0, d1) = ((p - b).unit()?, (f - p).unit()?);
        Some(d0.cross(d1).atan2(d0.dot(d1)).abs())
    };
    let narrow: Vec<f64> = (0..n)
        .map(|i| {
            if !closed && (i == 0 || i == n - 1) {
                0.0
            } else {
                turn_at(i, reach, 1).unwrap_or(0.0)
            }
        })
        .collect();
    let mut turn = vec![0.0f64; n];
    for i in 0..n {
        let t = narrow[i];
        if t < min_turn {
            turn[i] = t;
            continue;
        }
        // Sparse polygon: a big turn between two long straight chords.
        let chord = |d: isize| -> f64 {
            let j = i as isize + d;
            if !closed && (j < 0 || j >= n as isize) {
                0.0
            } else {
                pts[i].dist(at(j))
            }
        };
        if chord(-1) >= 2.0 * reach && chord(1) >= 2.0 * reach {
            turn[i] = t;
            continue;
        }
        let concentrated = match turn_at(i, 2.0 * reach, 2) {
            // Straight legs: the wider stencil turns about the same.
            Some(wide) => wide <= 1.35 * t + 1e-9,
            // Too near an end for the wide stencil: compare with the
            // neighbours' turns instead (a sampled arc turns at each).
            None => {
                let nb = |d: isize| -> f64 {
                    let j = i as isize + d;
                    if !closed && (j < 0 || j >= n as isize) {
                        0.0
                    } else {
                        narrow[j.rem_euclid(n as isize) as usize]
                    }
                };
                t >= 2.0 * nb(-1).max(nb(1))
            }
        };
        if concentrated {
            turn[i] = t;
        }
    }
    let mut corner = vec![false; n];
    for i in 0..n {
        if turn[i] < min_turn {
            continue;
        }
        // Keep only the sharpest sample within `reach` (ties → first).
        let mut is_max = true;
        for dir in [-1isize, 1] {
            let mut acc = 0.0;
            let mut j = i as isize;
            for _ in 0..walk {
                let k = j + dir;
                if !closed && (k < 0 || k >= n as isize) {
                    break;
                }
                acc += at(j).dist(at(k));
                j = k;
                let jj = k.rem_euclid(n as isize) as usize;
                if jj == i || acc > reach {
                    break;
                }
                if turn[jj] > turn[i] || (turn[jj] == turn[i] && jj < i) {
                    is_max = false;
                    break;
                }
            }
        }
        corner[i] = is_max;
    }
    corner
}

/// Arc length of `seg` over `[0, u]` with `path_length::cubic_arc_length`
/// (the prefix of a cubic is itself a cubic: de Casteljau split at `u`).
pub(crate) fn shared_arc_len(seg: &FitSeg, u: f64) -> f64 {
    use crate::path_length::cubic_arc_length;
    let u = u.clamp(0.0, 1.0);
    match seg {
        FitSeg::Line(a, b) => a.dist(*b) * u,
        FitSeg::Cubic(c) => {
            let [p0, p1, p2, p3] = c.0;
            let (a, b, cc) = (p0.lerp(p1, u), p1.lerp(p2, u), p2.lerp(p3, u));
            let (d, e) = (a.lerp(b, u), b.lerp(cc, u));
            let f = d.lerp(e, u);
            let arr = |v: V2| [v.x, v.y];
            if u >= 1.0 {
                cubic_arc_length(arr(p0), arr(p1), arr(p2), arr(p3))
            } else {
                cubic_arc_length(arr(p0), arr(a), arr(d), arr(f))
            }
        }
    }
}

/// Fit sampled input points (Pencil / Paintbrush / Simplify) to a cubic path.
pub fn fit_points(input: &[V2], opts: &FitOptions) -> FitOutput {
    let tolerance = if opts.tolerance.is_finite() {
        opts.tolerance.clamp(0.01, 1000.0)
    } else {
        1.0
    };
    let corner_angle = if opts.corner_angle.is_finite() {
        opts.corner_angle.clamp(1.0, 179.0)
    } else {
        70.0
    };
    // Scrub: finite, deduplicated. `origin[k]` = input index of kept point k.
    let mut pts: Vec<V2> = Vec::new();
    let mut kept_of_input: Vec<Option<usize>> = Vec::with_capacity(input.len());
    for p in input.iter().take(MAX_FIT_POINTS) {
        if !(p.is_finite() && p.x.abs() <= MAX_COORD && p.y.abs() <= MAX_COORD) {
            kept_of_input.push(None);
            continue;
        }
        if let Some(last) = pts.last() {
            if last.dist(*p) <= 1e-9 * (1.0 + p.x.abs().max(p.y.abs())) {
                kept_of_input.push(Some(pts.len() - 1));
                continue;
            }
        }
        pts.push(*p);
        kept_of_input.push(Some(pts.len() - 1));
    }
    kept_of_input.resize(input.len(), None);
    let mut closed = opts.closed;
    // A closed fit whose last sample repeats the first.
    let mut wraps_to_first = false;
    if closed
        && pts.len() > 2
        && pts[0].dist(pts[pts.len() - 1]) <= 1e-9 * (1.0 + pts[0].x.abs().max(pts[0].y.abs()))
    {
        pts.pop();
        wraps_to_first = true;
    }
    if pts.len() < 3 {
        closed = false;
    }

    let finish =
        |points: Vec<BlendPathPoint>, kept_t: Vec<f64>, corners: Vec<usize>, closed: bool| {
            let mut sample_t = Vec::with_capacity(input.len());
            let mut last = 0.0;
            for k in &kept_of_input {
                let t = match k {
                    Some(k) => kept_t.get(*k).copied().unwrap_or(last),
                    None => last,
                };
                let t = if closed { t } else { t.max(last) };
                sample_t.push(t);
                last = t;
            }
            // Leading dropped samples take the first kept value.
            if let Some(first) = kept_of_input.iter().position(Option::is_some) {
                let v = sample_t[first];
                for t in sample_t.iter_mut().take(first) {
                    *t = v;
                }
            }
            if wraps_to_first && closed {
                if let Some(t) = sample_t.last_mut() {
                    *t = 1.0;
                }
            }
            FitOutput {
                points,
                closed,
                sample_t,
                corners,
            }
        };

    if pts.is_empty() {
        return finish(Vec::new(), Vec::new(), Vec::new(), false);
    }
    if pts.len() == 1 {
        return finish(
            vec![BlendPathPoint::Move {
                x: pts[0].x,
                y: pts[0].y,
            }],
            vec![0.0],
            Vec::new(),
            false,
        );
    }

    let reach = 2.0 * tolerance;
    let corner_flags = detect_corners(&pts, closed, reach, corner_angle.to_radians());
    let n = pts.len();
    let tol2 = tolerance * tolerance;
    let mut segs: Vec<FitSeg> = Vec::new();
    let mut params: Vec<FitParam> = Vec::new();
    // `order[k]` = kept-point index of the k-th point in fitting order.
    let order: Vec<usize>;
    if closed {
        // The loop starts where the user started (input sample 0). When that
        // is not a corner, the runs meeting there share its central tangent.
        order = (0..=n).map(|k| k % n).collect();
        let ring: Vec<V2> = order.iter().map(|&i| pts[i]).collect();
        let seam = (!corner_flags[0]).then(|| (pts[1] - pts[n - 1]).unit().unwrap_or(v2(1.0, 0.0)));
        let mut a = 0;
        for k in 1..ring.len() {
            if (k < n && corner_flags[k]) || k == ring.len() - 1 {
                let run = &ring[a..=k];
                let chord = (run[run.len() - 1] - run[0]).unit().unwrap_or(v2(1.0, 0.0));
                let t1 = match seam {
                    Some(t) if a == 0 => t,
                    _ => end_tangent(run, 0, true, reach).unwrap_or(chord),
                };
                let t2 = match seam {
                    Some(t) if k == ring.len() - 1 => -t,
                    _ => end_tangent(run, run.len() - 1, false, reach).unwrap_or(-chord),
                };
                fit_run(run, t1, t2, tol2, &mut segs, &mut params);
                a = k;
            }
        }
    } else {
        order = (0..n).collect();
        let mut a = 0;
        for k in 1..n {
            if corner_flags[k] || k == n - 1 {
                fit_open_run(&pts[a..=k], reach, tol2, &mut segs, &mut params);
                a = k;
            }
        }
    }
    // The final point of the fit order sits at the end of the last segment.
    params.push(FitParam {
        seg: segs.len().saturating_sub(1),
        u: 1.0,
    });

    // Arc-length fractions, measured with the ONE arc-length implementation
    // markers use (`path_length::cubic_arc_length`, what `anchor_fractions`
    // sums), so a sample's `t` and an anchor's `t` agree on the fitted path.
    let seg_len: Vec<f64> = segs.iter().map(|s| shared_arc_len(s, 1.0)).collect();
    let mut cum = Vec::with_capacity(segs.len() + 1);
    let mut acc = 0.0;
    cum.push(0.0);
    for l in &seg_len {
        acc += l;
        cum.push(acc);
    }
    let total = acc;
    let mut kept_t = vec![0.0; n];
    for (k, p) in params.iter().enumerate() {
        let Some(&idx) = order.get(k) else { break };
        let t = if total > 0.0 && p.seg < segs.len() {
            let within = if p.u >= 1.0 {
                seg_len[p.seg]
            } else {
                shared_arc_len(&segs[p.seg], p.u)
            };
            ((cum[p.seg] + within) / total).clamp(0.0, 1.0)
        } else {
            0.0
        };
        // Closed: the loop's closing point (k == n) is the start again.
        if closed && k == n {
            break;
        }
        kept_t[idx] = t;
    }

    let mut points = Vec::with_capacity(segs.len() + 1);
    if let Some(first) = segs.first() {
        let p = first.start();
        points.push(BlendPathPoint::Move { x: p.x, y: p.y });
    }
    for s in &segs {
        points.push(s.to_path_point());
    }
    // Input indices of corners (first input sample mapped to each kept corner).
    let mut corners = Vec::new();
    for (k, &c) in corner_flags.iter().enumerate() {
        if c {
            if let Some(i) = kept_of_input.iter().position(|x| *x == Some(k)) {
                corners.push(i);
            }
        }
    }
    finish(points, kept_t, corners, closed)
}

// ---------------------------------------------------------------------------
// JSON surface (wasm export `fit_points`)
// ---------------------------------------------------------------------------

fn parse_xy(v: &Value) -> Option<V2> {
    if let Some(a) = v.as_array() {
        return Some(v2(a.first()?.as_f64()?, a.get(1)?.as_f64()?));
    }
    let o = v.as_object()?;
    Some(v2(o.get("x")?.as_f64()?, o.get("y")?.as_f64()?))
}

fn error(code: &str, message: &str) -> String {
    json!({"status": "error", "code": code, "message": message}).to_string()
}

/// JSON in / JSON out; always an envelope, never panics.
///
/// ```jsonc
/// // request
/// { "points": [{"x","y"}] | [[x, y]], "tolerance"?: 1, "cornerAngle"?: 70, "closed"?: false }
/// // response
/// { "status": "ok", "points": PathPoint[], "closed": bool, "sampleT": number[], "corners": number[] }
/// { "status": "empty" }                       // no finite input point
/// { "status": "error", "code": "invalid_json" | "invalid_request" | "too_many_points", "message" }
/// ```
pub fn fit_points_json(request_json: &str) -> String {
    if request_json.len() > 32 * 1024 * 1024 {
        return error("request_too_large", "request exceeds 32 MiB");
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
    if raw.len() > MAX_FIT_POINTS {
        return error("too_many_points", "at most 100000 points");
    }
    let input: Vec<V2> = raw
        .iter()
        .map(|p| parse_xy(p).unwrap_or(v2(f64::NAN, f64::NAN)))
        .collect();
    let d = FitOptions::default();
    let opts = FitOptions {
        tolerance: obj
            .get("tolerance")
            .and_then(Value::as_f64)
            .unwrap_or(d.tolerance),
        corner_angle: obj
            .get("cornerAngle")
            .and_then(Value::as_f64)
            .unwrap_or(d.corner_angle),
        closed: obj.get("closed").and_then(Value::as_bool).unwrap_or(false),
    };
    let out = fit_points(&input, &opts);
    if out.points.is_empty() {
        return json!({"status": "empty"}).to_string();
    }
    json!({
        "status": "ok",
        "points": out.points.iter().map(BlendPathPoint::to_json).collect::<Vec<_>>(),
        "closed": out.closed,
        "sampleT": out.sample_t,
        "corners": out.corners,
    })
    .to_string()
}

#[cfg(test)]
#[path = "fit_tests.rs"]
pub(crate) mod tests;
