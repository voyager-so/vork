//! Path points → sampled centerline (per subpath), with arc lengths, tangents
//! and corners. The outliner offsets these samples; nothing downstream looks at
//! the original segments again.

use crate::blend::BlendPathPoint;
use crate::path_geom::arc::{arc_to_beziers, ArcParams};
use crate::path_geom::fit::MAX_COORD;
use crate::path_geom::round_corners::{round_path_corners_with, PathCorners, PointCorner};
use crate::path_geom::vec2::{v2, Cubic, V2};
use crate::path_length::{anchor_fractions, AnchorFractions};

/// Below this a segment is a point.
pub(crate) const EPS_LEN: f64 = 1e-9;
/// A joint turning less than this (radians, ≈10°) is offset along the bisector
/// instead of getting join geometry: within 0.4 % of the half-width of any join
/// style, and it keeps dense polylines (SVG imports, unfitted input) from
/// turning every vertex into a join + fitting break.
const JOINT_CORNER: f64 = 0.17;
/// An interior sample turning more than this (a cusp inside one cubic) is a corner.
const CUSP_CORNER: f64 = 0.8;
/// Largest turn between consecutive samples on a curve (≈6°; the fit back to
/// cubics interpolates between these exact samples).
const MAX_TURN_STEP: f64 = 0.1;

#[derive(Debug, Clone, Copy)]
pub(crate) enum Seg {
    Line(V2, V2),
    Cubic(Cubic),
}

impl Seg {
    fn start(&self) -> V2 {
        match self {
            Seg::Line(a, _) => *a,
            Seg::Cubic(c) => c.0[0],
        }
    }
    fn end(&self) -> V2 {
        match self {
            Seg::Line(_, b) => *b,
            Seg::Cubic(c) => c.0[3],
        }
    }
    fn eval(&self, u: f64) -> V2 {
        match self {
            Seg::Line(a, b) => a.lerp(*b, u),
            Seg::Cubic(c) => c.eval(u),
        }
    }
    fn tangent(&self, u: f64) -> Option<V2> {
        match self {
            Seg::Line(a, b) => (*b - *a).unit(),
            Seg::Cubic(c) => c.tangent(u),
        }
    }
    pub(crate) fn length(&self) -> f64 {
        match self {
            Seg::Line(a, b) => a.dist(*b),
            Seg::Cubic(c) => c.arc_len(),
        }
    }
    /// Upper bound on how much the tangent turns along the segment.
    fn turn_bound(&self) -> f64 {
        match self {
            Seg::Line(..) => 0.0,
            Seg::Cubic(c) => {
                let edges: Vec<V2> = [c.0[1] - c.0[0], c.0[2] - c.0[1], c.0[3] - c.0[2]]
                    .into_iter()
                    .filter_map(V2::unit)
                    .collect();
                edges
                    .windows(2)
                    .map(|w| w[0].cross(w[1]).atan2(w[0].dot(w[1])).abs())
                    .sum()
            }
        }
    }
}

/// One centerline sample.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sample {
    pub p: V2,
    /// Arc length from the start of the whole path (all subpaths, points order).
    pub s: f64,
    /// Arc length from the start of this subpath.
    pub s_local: f64,
    /// Unit tangent arriving at / leaving the sample (equal unless `corner`).
    pub t_in: V2,
    pub t_out: V2,
    pub corner: bool,
    /// Second of a coincident pair at a width discontinuity: evaluate widths
    /// from the outgoing side.
    pub after: bool,
    /// Force a break in the fitted outline here (width kink or step).
    pub brk: bool,
    /// A corner INSIDE a curve (a cusp), not a path vertex: it gets round
    /// geometry whatever the node's `lineJoin`.
    pub cusp: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct Subpath {
    pub samples: Vec<Sample>,
    /// Ring: the last sample connects back to the first (not repeated).
    pub closed: bool,
    pub length: f64,
    /// Index of the subpath in points order (seeds per-subpath randomness).
    pub index: usize,
}

/// A parsed subpath before sampling.
#[derive(Debug, Clone)]
pub(crate) struct SegPath {
    pub segs: Vec<Seg>,
    pub closed: bool,
    /// Start point (used for a subpath with no segments of length).
    pub start: V2,
    /// The subpath had at least one drawing command (a lone move draws nothing).
    pub has_commands: bool,
    /// Per segment: the arc length the sampler uses. Its own length, except on
    /// unrounded paths, where segments are rescaled so every anchor sits
    /// exactly at `path_length::anchor_fractions` (the one arc length markers
    /// are measured with).
    pub lens: Vec<f64>,
    /// Per segment: the input point it ends on (`usize::MAX` = closing segment).
    src: Vec<usize>,
}

fn finite_pt(x: f64, y: f64) -> Option<V2> {
    (x.is_finite() && y.is_finite() && x.abs() <= MAX_COORD && y.abs() <= MAX_COORD)
        .then(|| v2(x, y))
}

fn point_ok(p: &BlendPathPoint) -> bool {
    let xy = |x: f64, y: f64| finite_pt(x, y).is_some();
    match p {
        BlendPathPoint::Move { x, y } | BlendPathPoint::Line { x, y } => xy(*x, *y),
        BlendPathPoint::Curve { x, y, cp1, cp2 } => {
            xy(*x, *y) && xy(cp1.0, cp1.1) && xy(cp2.0, cp2.1)
        }
        BlendPathPoint::Quadratic { x, y, cp } => xy(*x, *y) && xy(cp.0, cp.1),
        BlendPathPoint::Arc {
            x,
            y,
            rx,
            ry,
            x_axis_rotation,
            ..
        } => xy(*x, *y) && xy(*rx, *ry) && x_axis_rotation.is_finite(),
    }
}

/// Points → subpaths of lines and cubics.
pub(crate) fn segment_paths(points: &[Option<BlendPathPoint>], closed: bool) -> Vec<SegPath> {
    let mut out: Vec<SegPath> = Vec::new();
    let mut cur: Option<V2> = None;
    let mut start = V2::default();
    let close = |out: &mut Vec<SegPath>, cur: Option<V2>, start: V2| {
        if let (Some(sp), Some(c)) = (out.last_mut(), cur) {
            if closed && c.dist(start) > EPS_LEN {
                sp.segs.push(Seg::Line(c, start));
                sp.src.push(usize::MAX);
            }
            sp.closed = closed;
        }
    };
    for (index, p) in points.iter().enumerate() {
        let Some(p) = p.as_ref().filter(|p| point_ok(p)) else {
            continue;
        };
        let begin = |out: &mut Vec<SegPath>, at: V2| {
            out.push(SegPath {
                segs: Vec::new(),
                closed: false,
                start: at,
                has_commands: false,
                lens: Vec::new(),
                src: Vec::new(),
            })
        };
        match p {
            BlendPathPoint::Move { x, y } => {
                close(&mut out, cur, start);
                start = v2(*x, *y);
                cur = Some(start);
                begin(&mut out, start);
                continue;
            }
            _ => {
                if cur.is_none() {
                    // A path that doesn't start with a move starts at its first point.
                    let (x, y) = match p {
                        BlendPathPoint::Line { x, y }
                        | BlendPathPoint::Curve { x, y, .. }
                        | BlendPathPoint::Quadratic { x, y, .. }
                        | BlendPathPoint::Arc { x, y, .. } => (*x, *y),
                        BlendPathPoint::Move { .. } => unreachable!(),
                    };
                    start = v2(x, y);
                    cur = Some(start);
                    begin(&mut out, start);
                    continue;
                }
            }
        }
        let a = cur.unwrap_or_default();
        let mut new_segs: Vec<Seg> = Vec::new();
        let end = match p {
            BlendPathPoint::Line { x, y } => {
                let b = v2(*x, *y);
                new_segs.push(Seg::Line(a, b));
                b
            }
            BlendPathPoint::Curve { x, y, cp1, cp2 } => {
                let b = v2(*x, *y);
                new_segs.push(Seg::Cubic(Cubic([
                    a,
                    v2(cp1.0, cp1.1),
                    v2(cp2.0, cp2.1),
                    b,
                ])));
                b
            }
            BlendPathPoint::Quadratic { x, y, cp } => {
                let b = v2(*x, *y);
                let q = v2(cp.0, cp.1);
                new_segs.push(Seg::Cubic(Cubic([
                    a,
                    a.lerp(q, 2.0 / 3.0),
                    b.lerp(q, 2.0 / 3.0),
                    b,
                ])));
                b
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
                let b = v2(*x, *y);
                let beziers = arc_to_beziers(
                    a.x,
                    a.y,
                    &ArcParams {
                        rx: *rx,
                        ry: *ry,
                        x_axis_rotation: *x_axis_rotation,
                        large_arc: *large_arc,
                        sweep: *sweep,
                        x: b.x,
                        y: b.y,
                    },
                );
                if beziers.is_empty() {
                    new_segs.push(Seg::Line(a, b));
                } else {
                    let mut from = a;
                    for bz in beziers {
                        let to = v2(bz.end[0], bz.end[1]);
                        new_segs.push(Seg::Cubic(Cubic([
                            from,
                            v2(bz.cp1[0], bz.cp1[1]),
                            v2(bz.cp2[0], bz.cp2[1]),
                            to,
                        ])));
                        from = to;
                    }
                }
                b
            }
            BlendPathPoint::Move { .. } => unreachable!(),
        };
        let sp = out.last_mut().expect("a subpath was begun");
        sp.has_commands = true;
        for s in new_segs {
            if s.start().is_finite() && s.end().is_finite() {
                sp.segs.push(s);
                sp.src.push(index);
            }
        }
        cur = Some(end);
    }
    close(&mut out, cur, start);
    // Drop segments of no length.
    for sp in out.iter_mut() {
        let keep: Vec<bool> = sp
            .segs
            .iter()
            .map(|s| s.length() > EPS_LEN && s.length().is_finite())
            .collect();
        let mut k = keep.iter();
        sp.segs.retain(|_| *k.next().unwrap_or(&false));
        let mut k = keep.iter();
        sp.src.retain(|_| *k.next().unwrap_or(&false));
        sp.lens = sp.segs.iter().map(Seg::length).collect();
    }
    out
}

/// Parse `PathNode.points` JSON (+ `cornerRadius`, like the engine's
/// `path_cmds_for_node`) into subpaths, and each input point's arc-length
/// fraction from `path_length::anchor_fractions` (what anchored width markers
/// bind to, measured on the UNROUNDED path).
///
/// On an unrounded path every segment's sampling length is rescaled to its
/// share of the anchor fractions, so a sample on anchor `k` has exactly
/// `anchor_t[k]` (the rescale is within the two measurements' tolerance). A
/// rounded path (`cornerRadius`) has different segments; it keeps its own
/// lengths.
/// `corners`: the node's radius and style; each point's own settings are read from `raw_points` here, in step with
/// the points kept.
pub(crate) fn node_segments(
    raw_points: &[crate::value::Value],
    closed: bool,
    corners: &PathCorners,
) -> (Vec<SegPath>, Vec<Option<f64>>, bool) {
    let fractions = anchor_fractions(raw_points, closed);
    let periodic = fractions.is_periodic(closed);
    let anchor_t: Vec<Option<f64>> = fractions
        .anchor_t
        .iter()
        .map(|t| t.is_finite().then_some(*t))
        .collect();
    let parsed: Vec<Option<BlendPathPoint>> =
        raw_points.iter().map(BlendPathPoint::from_json).collect();
    let mut corners = PathCorners {
        radius: corners.radius,
        style: corners.style,
        points: Vec::new(),
    };
    let own: Vec<PointCorner> = raw_points
        .iter()
        .zip(&parsed)
        .filter(|(_, p)| p.as_ref().is_some_and(point_ok))
        .map(|(v, _)| PointCorner::from_json(v))
        .collect();
    if own.iter().any(|c| *c != PointCorner::default()) {
        corners.points = own;
    }
    if corners.rounds_anything() {
        let valid: Vec<BlendPathPoint> = parsed.into_iter().flatten().filter(point_ok).collect();
        let rounded: Vec<Option<BlendPathPoint>> = round_path_corners_with(&valid, closed, &corners)
            .into_iter()
            .map(Some)
            .collect();
        return (segment_paths(&rounded, closed), anchor_t, periodic);
    }
    let mut paths = segment_paths(&parsed, closed);
    let own_total: f64 = paths.iter().flat_map(|sp| sp.lens.iter()).sum();
    if own_total > 0.0 && fractions.total_length > 0.0 {
        for sp in paths.iter_mut() {
            rescale_to_anchors(sp, &fractions, own_total);
        }
    }
    (paths, anchor_t, periodic)
}

/// Give each run of segments ending on one input point the length its anchor
/// fractions span (split among the run by own length).
fn rescale_to_anchors(sp: &mut SegPath, fractions: &AnchorFractions, own_total: f64) {
    let n = sp.segs.len();
    let anchor = |i: usize| fractions.anchor_t.get(i).copied().filter(|t| t.is_finite());
    let mut i = 0;
    while i < n {
        let src = sp.src[i];
        let mut j = i;
        while j < n && sp.src[j] == src {
            j += 1;
        }
        // Fraction at the run's start: the previous run's end anchor, or the
        // subpath start.
        let t0 = if i > 0 {
            anchor(sp.src[i - 1])
        } else {
            fractions
                .subpaths
                .iter()
                .find(|e| e.start_index <= src && src <= e.end_index)
                .map(|e| e.start)
        };
        let t1 = if src == usize::MAX {
            let prev = if i > 0 { sp.src[i - 1] } else { usize::MAX };
            fractions
                .subpaths
                .iter()
                .find(|e| e.start_index <= prev && prev <= e.end_index)
                .map(|e| e.end)
        } else {
            anchor(src)
        };
        if let (Some(t0), Some(t1)) = (t0, t1) {
            let own: f64 = sp.lens[i..j].iter().sum();
            let span = (t1 - t0).max(0.0) * own_total;
            if own > 0.0 && span > 0.0 && (span / own - 1.0).abs() < 0.01 {
                for k in i..j {
                    sp.lens[k] *= span / own;
                }
            }
        }
        i = j;
    }
}

/// Sampling density knobs, chosen by the width model.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Density {
    /// Largest arc-length step between samples.
    pub max_step: f64,
    /// Most samples this subpath may produce (before width events); the turn
    /// and length limits coarsen to fit.
    pub budget: usize,
}

/// Samples a segment would like: arc-length steps and ≤ `MAX_TURN_STEP` of
/// turn per step (an upper bound — the adaptive pass usually needs fewer).
pub(crate) fn segment_estimate(seg: &Seg, len: f64, max_step: f64) -> f64 {
    (len / max_step.max(1e-9))
        .ceil()
        .max((seg.turn_bound() / MAX_TURN_STEP).ceil())
        .max(1.0)
}

/// Estimated samples for a whole subpath.
pub(crate) fn subpath_estimate(sp: &SegPath, max_step: f64) -> f64 {
    sp.segs
        .iter()
        .zip(&sp.lens)
        .map(|(s, l)| segment_estimate(s, *l, max_step))
        .sum()
}

/// Parameters of a cubic's samples: arc-length steps of at most `max_step`,
/// then refined (bisection passes) until no step turns more than
/// `MAX_TURN_STEP`, within `budget` intervals. Returns interior + end params
/// (excludes 0).
fn cubic_params(c: &Cubic, len: f64, max_step: f64, budget: usize) -> Vec<f64> {
    let n0 = ((len / max_step).ceil() as usize).clamp(1, budget.max(1));
    // Arc-length-uniform start via a chord table.
    let m = 4 * n0;
    let mut table: Vec<(f64, f64)> = Vec::with_capacity(m + 1);
    let mut prev = c.0[0];
    let mut acc = 0.0;
    table.push((0.0, 0.0));
    for j in 1..=m {
        let u = j as f64 / m as f64;
        let q = c.eval(u);
        acc += prev.dist(q);
        prev = q;
        table.push((u, acc));
    }
    let chord_total = acc.max(1e-300);
    let mut us: Vec<f64> = Vec::with_capacity(n0 + 1);
    us.push(0.0);
    let mut j = 0;
    for i in 1..n0 {
        let target = chord_total * i as f64 / n0 as f64;
        while j + 1 < table.len() && table[j + 1].1 < target {
            j += 1;
        }
        let (u0, a0) = table[j];
        let (u1, a1) = table[(j + 1).min(m)];
        let f = if a1 > a0 {
            ((target - a0) / (a1 - a0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        us.push(u0 + (u1 - u0) * f);
    }
    us.push(1.0);
    // Refine by turn: split every interval whose end tangents differ by more
    // than the bound, pass by pass (so a tight bend gets evenly dense), until
    // none do or the budget is spent. A true cusp stops at the width floor.
    let tangent = |u: f64| c.tangent(u).unwrap_or(v2(1.0, 0.0));
    let mut tans: Vec<V2> = us.iter().map(|u| tangent(*u)).collect();
    for _ in 0..40 {
        if us.len() > budget {
            break;
        }
        let mut next_u = Vec::with_capacity(us.len() * 2);
        let mut next_t = Vec::with_capacity(us.len() * 2);
        let mut split = false;
        for k in 0..us.len() {
            next_u.push(us[k]);
            next_t.push(tans[k]);
            if k + 1 == us.len() {
                break;
            }
            let (a, b) = (tans[k], tans[k + 1]);
            let turn = a.cross(b).atan2(a.dot(b)).abs();
            let width = us[k + 1] - us[k];
            if turn > MAX_TURN_STEP && width > 1e-7 && next_u.len() + (us.len() - k) <= budget + 1 {
                let mid = 0.5 * (us[k] + us[k + 1]);
                next_u.push(mid);
                next_t.push(tangent(mid));
                split = true;
            }
        }
        us = next_u;
        tans = next_t;
        if !split {
            break;
        }
    }
    us.remove(0);
    us
}

#[derive(Clone, Copy)]
struct Raw {
    p: V2,
    s: f64,
    seg: usize,
    u: f64,
    /// End of segment `seg`, start of `seg + 1`.
    joint: bool,
}

/// Sample one subpath. `s0` = arc length of the path before this subpath.
pub(crate) fn sample_subpath(
    sp: &SegPath,
    index: usize,
    s0: f64,
    density: Density,
    events: &[(f64, bool)],
) -> Subpath {
    let mut raws: Vec<Raw> = Vec::new();
    let mut s = s0;
    let max_step = density.max_step.max(1e-6);
    // Share the budget by each segment's estimate.
    let est: Vec<f64> = sp
        .segs
        .iter()
        .zip(&sp.lens)
        .map(|(g, l)| segment_estimate(g, *l, max_step))
        .collect();
    let est_total: f64 = est.iter().sum::<f64>().max(1.0);
    let spare = density.budget.saturating_sub(sp.segs.len()) as f64;
    for (k, seg) in sp.segs.iter().enumerate() {
        let len = sp.lens[k];
        let seg_budget = (1.0 + spare * est[k] / est_total).floor() as usize;
        let n = ((len / max_step).ceil() as usize).clamp(1, seg_budget.max(1));
        if k == 0 {
            raws.push(Raw {
                p: seg.start(),
                s,
                seg: k,
                u: 0.0,
                joint: false,
            });
        } else if let Some(last) = raws.last_mut() {
            last.joint = true;
        }
        match seg {
            Seg::Line(a, b) => {
                for i in 1..=n {
                    let u = i as f64 / n as f64;
                    raws.push(Raw {
                        p: a.lerp(*b, u),
                        s: s + len * u,
                        seg: k,
                        u,
                        joint: false,
                    });
                }
            }
            Seg::Cubic(c) => {
                let params = cubic_params(c, len, max_step, seg_budget.max(1));
                let count = params.len();
                let mut last_p = c.0[0];
                let mut last_s = s;
                for u in params {
                    let q = c.eval(u);
                    last_s += last_p.dist(q);
                    last_p = q;
                    raws.push(Raw {
                        p: q,
                        s: last_s,
                        seg: k,
                        u,
                        joint: false,
                    });
                }
                // Keep the measured length as the segment's length: spread the
                // tiny chord/length discrepancy over the samples.
                let drift = (s + len) - last_s;
                if drift.abs() > 0.0 && count > 0 {
                    let base = raws.len() - count;
                    for (i, r) in raws[base..].iter_mut().enumerate() {
                        r.s += drift * (i + 1) as f64 / count as f64;
                    }
                }
            }
        }
        s += len;
    }
    // Width events: insert samples (pairs at discontinuities).
    let mut is_after: Vec<bool> = vec![false; raws.len()];
    let mut is_brk: Vec<bool> = vec![false; raws.len()];
    if !events.is_empty() && raws.len() >= 2 {
        let mut merged: Vec<Raw> = Vec::with_capacity(raws.len() + events.len() * 2);
        let mut after = Vec::with_capacity(merged.capacity());
        let mut brk = Vec::with_capacity(merged.capacity());
        let mut e = 0;
        let s_end = raws[raws.len() - 1].s;
        let seg_of = |k: usize| &sp.segs[k.min(sp.segs.len() - 1)];
        for i in 0..raws.len() {
            // Events strictly before raws[i] (and after raws[i-1]).
            while e < events.len() && events[e].0 < raws[i].s - 1e-9 {
                let (es, disc) = events[e];
                e += 1;
                if i == 0 || es <= raws[i - 1].s + 1e-9 {
                    continue;
                }
                let (a, b) = (&raws[i - 1], &raws[i]);
                let seg = b.seg;
                let ua = if a.seg == seg { a.u } else { 0.0 };
                let f = ((es - a.s) / (b.s - a.s)).clamp(0.0, 1.0);
                let u = ua + (b.u - ua) * f;
                let p = seg_of(seg).eval(u);
                merged.push(Raw {
                    p,
                    s: es,
                    seg,
                    u,
                    joint: false,
                });
                after.push(false);
                brk.push(true);
                if disc {
                    merged.push(Raw {
                        p,
                        s: es,
                        seg,
                        u,
                        joint: false,
                    });
                    after.push(true);
                    brk.push(true);
                }
            }
            let r = &raws[i];
            let mut on_this = false;
            let mut disc_here = false;
            while e < events.len() && (events[e].0 - r.s).abs() <= 1e-9 {
                on_this = true;
                disc_here |= events[e].1;
                e += 1;
            }
            merged.push(*r);
            after.push(false);
            brk.push(on_this);
            if disc_here && r.s < s_end - 1e-9 && r.s > raws[0].s + 1e-9 {
                merged.push(*r);
                after.push(true);
                brk.push(true);
            } else if disc_here && r.s <= raws[0].s + 1e-9 {
                // A discontinuity at the very start: the outgoing value wins.
                let k = after.len() - 1;
                after[k] = true;
            }
        }
        raws = merged;
        is_after = after;
        is_brk = brk;
    }

    // Tangents + corners.
    let n = raws.len();
    let mut samples: Vec<Sample> = Vec::with_capacity(n);
    let s_start = raws.first().map(|r| r.s).unwrap_or(s0);
    for (i, r) in raws.iter().enumerate() {
        let seg = &sp.segs[r.seg.min(sp.segs.len().saturating_sub(1))];
        let t = seg.tangent(r.u).unwrap_or(v2(1.0, 0.0));
        samples.push(Sample {
            p: r.p,
            s: r.s,
            s_local: r.s - s_start,
            t_in: t,
            t_out: t,
            corner: false,
            after: is_after[i],
            brk: is_brk[i],
            cusp: false,
        });
    }
    let length = s - s0;
    // Joints between segments. The joint sample belongs to the NEXT segment
    // (tangent at u=0 is its t_out); t_in comes from the previous segment's end.
    let apply_joint = |samples: &mut [Sample], at: usize, t_in: V2, t_out: V2| {
        let turn = t_in.cross(t_out).atan2(t_in.dot(t_out)).abs();
        let smp = &mut samples[at];
        if turn > JOINT_CORNER {
            smp.corner = true;
            smp.t_in = t_in;
            smp.t_out = t_out;
        } else {
            let t = (t_in + t_out).unit().unwrap_or(t_out);
            smp.t_in = t;
            smp.t_out = t;
        }
    };
    for (i, r) in raws.iter().enumerate() {
        if !r.joint || r.seg + 1 >= sp.segs.len() {
            continue;
        }
        let t_in = sp.segs[r.seg].tangent(1.0).unwrap_or(v2(1.0, 0.0));
        let t_out = sp.segs[r.seg + 1].tangent(0.0).unwrap_or(t_in);
        apply_joint(&mut samples, i, t_in, t_out);
    }
    if sp.closed && n >= 2 && !sp.segs.is_empty() {
        // Seam: the last raw repeats the first point. Keep it as the seam's
        // INCOMING sample (its width is the profile at the end, t = 1) and
        // the first as the outgoing one, so a width step at the seam is a
        // coincident pair like any discontinuity (`offset.rs` routes a step
        // at a corner through the pivot). Both break the fit, and neither can
        // become the rotated ring start.
        let t_in = sp.segs[sp.segs.len() - 1]
            .tangent(1.0)
            .unwrap_or(v2(1.0, 0.0));
        let t_out = sp.segs[0].tangent(0.0).unwrap_or(t_in);
        let last = samples.len() - 1;
        samples[last].p = samples[0].p;
        samples[last].brk = true;
        samples[0].brk = true;
        samples[0].after = true;
        apply_joint(&mut samples, 0, t_in, t_out);
        apply_joint(&mut samples, last, t_in, t_out);
    }
    // Interior cusps: a sample whose chords (to the nearest distinct
    // neighbours) turn sharply. After the turn-adaptive sampling this is a
    // real cusp (or a budget-starved bend): round geometry, not `lineJoin`.
    let m = samples.len();
    let pos: Vec<V2> = samples.iter().map(|x| x.p).collect();
    let distinct = |i: usize, dir: isize| -> Option<usize> {
        let mut j = i as isize;
        for _ in 0..m {
            j += dir;
            if j < 0 || j >= m as isize {
                if !sp.closed {
                    return None;
                }
                j = j.rem_euclid(m as isize);
            }
            let jj = j as usize;
            if jj == i {
                return None;
            }
            if pos[jj].dist(pos[i]) > EPS_LEN {
                return Some(jj);
            }
        }
        None
    };
    for i in 0..m {
        if samples[i].corner {
            continue;
        }
        if let (Some(a), Some(b)) = (distinct(i, -1), distinct(i, 1)) {
            let (Some(d0), Some(d1)) = ((pos[i] - pos[a]).unit(), (pos[b] - pos[i]).unit()) else {
                continue;
            };
            let turn = d0.cross(d1).atan2(d0.dot(d1)).abs();
            if turn > CUSP_CORNER {
                samples[i].corner = true;
                samples[i].cusp = true;
                samples[i].t_in = d0;
                samples[i].t_out = d1;
            }
        }
    }
    Subpath {
        samples,
        closed: sp.closed,
        length,
        index,
    }
}
