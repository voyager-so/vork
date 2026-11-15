//! Path plumbing shared by the brush placers: `PathPoint` lists ↔ subpaths of
//! cubics, de Casteljau splits, bounds.

use crate::blend::BlendPathPoint;
use crate::path_geom::arc::{arc_to_beziers, ArcParams};
use crate::path_geom::vec2::{v2, Cubic, V2};

/// Coordinates beyond this are treated as garbage (the subpath is dropped).
pub const MAX_COORD: f64 = 1e9;

/// One subpath as a chain of cubics (lines are degree-elevated).
#[derive(Debug, Clone, PartialEq)]
pub struct Subpath {
    pub segs: Vec<Cubic>,
    pub closed: bool,
}

fn finite(p: V2) -> bool {
    p.is_finite() && p.x.abs() <= MAX_COORD && p.y.abs() <= MAX_COORD
}

/// Split a `PathPoint` list into subpaths of cubics, the way the renderers
/// read it: `move` starts a subpath (the pen starts at (0, 0)), quadratics are
/// elevated, arcs go through `arc_to_beziers` (degenerate → line), and every
/// subpath of a closed path is closed (a closing line is added when the last
/// anchor is not on the first). Subpaths with non-finite or huge coordinates
/// are dropped; subpaths with no segment are dropped.
pub fn subpaths_from_points(points: &[BlendPathPoint], closed: bool) -> Vec<Subpath> {
    let mut out: Vec<Subpath> = Vec::new();
    let mut segs: Vec<Cubic> = Vec::new();
    let mut ok = true;
    let mut start = v2(0.0, 0.0);
    let mut cur = start;
    let mut flush = |segs: &mut Vec<Cubic>, ok: bool, start: V2, cur: V2| {
        if ok && !segs.is_empty() {
            let mut s = std::mem::take(segs);
            if closed && cur.dist(start) > 1e-9 {
                s.push(Cubic::line(cur, start));
            }
            out.push(Subpath { segs: s, closed });
        }
        segs.clear();
    };
    for p in points {
        match *p {
            BlendPathPoint::Move { x, y } => {
                flush(&mut segs, ok, start, cur);
                start = v2(x, y);
                cur = start;
                ok = finite(start);
            }
            BlendPathPoint::Line { x, y } => {
                let e = v2(x, y);
                ok &= finite(e);
                segs.push(Cubic::line(cur, e));
                cur = e;
            }
            BlendPathPoint::Curve { x, y, cp1, cp2 } => {
                let (c1, c2, e) = (v2(cp1.0, cp1.1), v2(cp2.0, cp2.1), v2(x, y));
                ok &= finite(c1) && finite(c2) && finite(e);
                segs.push(Cubic([cur, c1, c2, e]));
                cur = e;
            }
            BlendPathPoint::Quadratic { x, y, cp } => {
                let (q, e) = (v2(cp.0, cp.1), v2(x, y));
                ok &= finite(q) && finite(e);
                segs.push(Cubic([
                    cur,
                    cur + (q - cur) * (2.0 / 3.0),
                    e + (q - e) * (2.0 / 3.0),
                    e,
                ]));
                cur = e;
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
                let e = v2(x, y);
                ok &= finite(e) && rx.is_finite() && ry.is_finite() && x_axis_rotation.is_finite();
                if !ok {
                    cur = e;
                    continue;
                }
                let arc = ArcParams {
                    rx,
                    ry,
                    x_axis_rotation,
                    large_arc,
                    sweep,
                    x,
                    y,
                };
                let beziers = arc_to_beziers(cur.x, cur.y, &arc);
                if beziers.is_empty() {
                    segs.push(Cubic::line(cur, e));
                } else {
                    let mut from = cur;
                    for b in beziers {
                        let to = v2(b.end[0], b.end[1]);
                        segs.push(Cubic([
                            from,
                            v2(b.cp1[0], b.cp1[1]),
                            v2(b.cp2[0], b.cp2[1]),
                            to,
                        ]));
                        from = to;
                    }
                }
                cur = e;
            }
        }
    }
    flush(&mut segs, ok, start, cur);
    out
}

/// Euclidean length via `sqrt(x² + y²)`: `f64::hypot` is a slow software
/// routine in WASM, and the warp's hot loop measures millions of vectors.
#[inline]
pub fn norm(v: V2) -> f64 {
    (v.x * v.x + v.y * v.y).sqrt()
}

/// Split at `t` (de Casteljau).
pub fn split(c: &Cubic, t: f64) -> (Cubic, Cubic) {
    let [p0, p1, p2, p3] = c.0;
    let a = p0.lerp(p1, t);
    let b = p1.lerp(p2, t);
    let cc = p2.lerp(p3, t);
    let d = a.lerp(b, t);
    let e = b.lerp(cc, t);
    let f = d.lerp(e, t);
    (Cubic([p0, a, d, f]), Cubic([f, e, cc, p3]))
}

/// The piece of `c` over `[t0, t1]`.
pub fn sub(c: &Cubic, t0: f64, t1: f64) -> Cubic {
    if t0 <= 0.0 && t1 >= 1.0 {
        return *c;
    }
    let (_, right) = split(c, t0.clamp(0.0, 1.0));
    if t0 >= 1.0 {
        return right;
    }
    let u = ((t1 - t0) / (1.0 - t0)).clamp(0.0, 1.0);
    split(&right, u).0
}

/// Flatness: largest distance of the handles from the chord.
pub fn flatness(c: &Cubic) -> f64 {
    let [p0, p1, p2, p3] = c.0;
    let d = p3 - p0;
    let l = d.len();
    if l < 1e-12 {
        return p1.dist(p0).max(p2.dist(p0));
    }
    ((p1 - p0).cross(d).abs() / l).max((p2 - p0).cross(d).abs() / l)
}

/// Axis-aligned bounds of control points `[minx, miny, maxx, maxy]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: V2,
    pub max: V2,
}

impl Default for Bounds {
    fn default() -> Self {
        Bounds::empty()
    }
}

impl Bounds {
    pub fn empty() -> Bounds {
        Bounds {
            min: v2(f64::INFINITY, f64::INFINITY),
            max: v2(f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }
    pub fn add(&mut self, p: V2) {
        self.min = v2(self.min.x.min(p.x), self.min.y.min(p.y));
        self.max = v2(self.max.x.max(p.x), self.max.y.max(p.y));
    }
    pub fn is_empty(&self) -> bool {
        !(self.min.x <= self.max.x && self.min.y <= self.max.y)
    }
    pub fn union(&mut self, o: &Bounds) {
        if !o.is_empty() {
            self.add(o.min);
            self.add(o.max);
        }
    }
    pub fn width(&self) -> f64 {
        self.max.x - self.min.x
    }
    pub fn height(&self) -> f64 {
        self.max.y - self.min.y
    }
}

/// Tight-ish bounds of a cubic chain (curve sampled at 16 steps per segment).
pub fn curve_bounds(segs: &[Cubic]) -> Bounds {
    let mut b = Bounds::empty();
    for c in segs {
        for i in 0..=16 {
            b.add(c.eval(i as f64 / 16.0));
        }
    }
    b
}

/// Subpaths → `PathPoint`s (move + curves; closed subpaths end on their start,
/// with no extra closing segment — the renderer closes them).
pub fn to_points(subpaths: &[Subpath]) -> Vec<BlendPathPoint> {
    let mut out = Vec::new();
    for sp in subpaths {
        let Some(first) = sp.segs.first() else {
            continue;
        };
        out.push(BlendPathPoint::Move {
            x: first.0[0].x,
            y: first.0[0].y,
        });
        for c in &sp.segs {
            out.push(BlendPathPoint::Curve {
                x: c.0[3].x,
                y: c.0[3].y,
                cp1: (c.0[1].x, c.0[1].y),
                cp2: (c.0[2].x, c.0[2].y),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subpaths_close_and_drop_garbage() {
        let pts = vec![
            BlendPathPoint::Move { x: 0.0, y: 0.0 },
            BlendPathPoint::Line { x: 10.0, y: 0.0 },
            BlendPathPoint::Line { x: 10.0, y: 10.0 },
            BlendPathPoint::Move {
                x: f64::NAN,
                y: 0.0,
            },
            BlendPathPoint::Line { x: 1.0, y: 1.0 },
            BlendPathPoint::Move { x: 20.0, y: 0.0 },
            BlendPathPoint::Quadratic {
                x: 30.0,
                y: 0.0,
                cp: (25.0, 5.0),
            },
        ];
        let s = subpaths_from_points(&pts, true);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].segs.len(), 3, "closing line added");
        assert_eq!(s[1].segs.len(), 2);
        let open = subpaths_from_points(&pts, false);
        assert_eq!(open[0].segs.len(), 2);
    }

    #[test]
    fn sub_matches_eval() {
        let c = Cubic([v2(0.0, 0.0), v2(3.0, 9.0), v2(7.0, -4.0), v2(10.0, 2.0)]);
        let s = sub(&c, 0.2, 0.7);
        for i in 0..=4 {
            let u = i as f64 / 4.0;
            assert!(s.eval(u).dist(c.eval(0.2 + 0.5 * u)) < 1e-9);
        }
    }
}

/// Parameter of `c` at arc length `target` from its start (bisection on the
/// Gauss–Legendre arc length; clamped to [0, 1]).
pub fn t_at_length(c: &Cubic, target: f64) -> f64 {
    // A degree-elevated line (handles at its thirds) is uniformly parameterised.
    let [p0, p1, p2, p3] = c.0;
    if p1.dist(p0.lerp(p3, 1.0 / 3.0)) < 1e-9 && p2.dist(p0.lerp(p3, 2.0 / 3.0)) < 1e-9 {
        let len = p0.dist(p3);
        return if len > 0.0 {
            (target / len).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
    let total = c.arc_len();
    if total.is_nan() || total <= 0.0 || target <= 0.0 {
        return 0.0;
    }
    if target >= total {
        return 1.0;
    }
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..24 {
        let mid = (lo + hi) / 2.0;
        if c.arc_len_to(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Round every corner of a subpath with a circular fillet, per corner: the
/// fillet would trim `radius · tan(turn / 2)` from each neighbour, but never
/// more than `max_trim` nor half a neighbour, so at acute corners the fillet
/// gets tighter instead of eating into the path. Turns under 0.05° and
/// near-reversals (cusps) are left alone. Output is cubic-only; a closed
/// subpath's seam corner is rounded too (its fillet is the last segment).
pub fn fillet_corners(sp: &Subpath, radius: f64, max_trim: f64) -> Subpath {
    fillet_corners_mapped(sp, radius, max_trim).0
}

/// [`fillet_corners`], plus the arc-length correspondence between the input
/// and the filleted output: `(input s, output s)` knots at every piece
/// boundary (monotone; piecewise-linear between them — exact on the kept
/// pieces, proportional across each fillet). [`ArcMap`] reads them.
pub fn fillet_corners_mapped(sp: &Subpath, radius: f64, max_trim: f64) -> (Subpath, ArcMap) {
    let segs: Vec<Cubic> = sp
        .segs
        .iter()
        .copied()
        .filter(|c| c.0.iter().any(|p| p.dist(c.0[0]) > 1e-12))
        .collect();
    let n = segs.len();
    let lens: Vec<f64> = segs.iter().map(|c| c.arc_len()).collect();
    if n == 0 || radius.is_nan() || radius <= 0.0 || max_trim.is_nan() || max_trim <= 0.0 {
        let total: f64 = lens.iter().sum();
        return (sp.clone(), ArcMap::identity(total));
    }
    // Trim lengths at each segment's start / end, and the turn at each joint.
    let mut trim_start = vec![0.0; n];
    let mut trim_end = vec![0.0; n];
    let mut rounded = vec![false; n]; // joint j = start of segs[j]
    let first = if sp.closed { 0 } else { 1 };
    for j in first..n {
        let i = (j + n - 1) % n;
        if i == j {
            continue;
        }
        let (Some(t1), Some(t2)) = (segs[i].tangent(1.0), segs[j].tangent(0.0)) else {
            continue;
        };
        let turn = t1.cross(t2).atan2(t1.dot(t2)).abs();
        if !(1e-3..=std::f64::consts::PI - 1e-3).contains(&turn) {
            continue;
        }
        let d = (radius * (turn / 2.0).tan())
            .min(max_trim)
            .min(lens[i] / 2.0)
            .min(lens[j] / 2.0);
        if d.is_nan() || d <= 1e-9 {
            continue;
        }
        trim_end[i] = d;
        trim_start[j] = d;
        rounded[j] = true;
    }
    // A segment whose two trims meet (both corners took half of it, or rounding
    // pushed them together) vanishes: its trims are scaled to its length and the
    // two fillets join at the meeting point, rather than the segment being kept
    // untrimmed (overshooting arcs) or cut to nothing (a break in the path).
    for i in 0..n {
        let sum = trim_start[i] + trim_end[i];
        if sum > lens[i] && sum > 0.0 {
            let f = lens[i] / sum;
            trim_start[i] *= f;
            trim_end[i] *= f;
        }
    }
    // Each segment after trimming: its kept piece, or (when it vanished) the
    // point where its fillets meet with the segment's tangent there.
    enum Kept {
        Piece(Cubic),
        Point(V2, V2),
    }
    let kept: Vec<Kept> = (0..n)
        .map(|i| {
            let t0 = t_at_length(&segs[i], trim_start[i]);
            let t1 = t_at_length(&segs[i], lens[i] - trim_end[i]);
            if lens[i] - trim_start[i] - trim_end[i] > 1e-6 * lens[i].max(1.0) && t1 > t0 + 1e-12 {
                Kept::Piece(sub(&segs[i], t0, t1))
            } else {
                let t = (t0 + t1) / 2.0;
                let tan = segs[i].tangent(t).unwrap_or_else(|| {
                    (segs[i].0[3] - segs[i].0[0]).unit().unwrap_or(v2(1.0, 0.0))
                });
                Kept::Point(segs[i].eval(t), tan)
            }
        })
        .collect();
    let end_of = |k: &Kept| match k {
        Kept::Piece(c) => c.tangent(1.0).map(|t| (c.0[3], t)),
        Kept::Point(p, t) => Some((*p, *t)),
    };
    let start_of = |k: &Kept| match k {
        Kept::Piece(c) => c.tangent(0.0).map(|t| (c.0[0], t)),
        Kept::Point(p, t) => Some((*p, *t)),
    };
    let arc = |a: &Kept, b: &Kept| -> Option<Cubic> {
        let ((p1, t1), (p2, t2)) = (end_of(a)?, start_of(b)?);
        let theta = t1.cross(t2).atan2(t1.dot(t2)).abs();
        let chord = p1.dist(p2);
        if chord < 1e-12 {
            return None;
        }
        if theta < 1e-6 {
            return Some(Cubic::line(p1, p2));
        }
        // Circular arc of turn θ through a chord c: r = c / (2 sin(θ/2)).
        let r = chord / (2.0 * (theta / 2.0).sin());
        let h = 4.0 / 3.0 * (theta / 4.0).tan() * r;
        Some(Cubic([p1, p1 + t1 * h, p2 - t2 * h, p2]))
    };
    let mut out = Vec::with_capacity(2 * n);
    // Input arc length at each segment start, and the output length so far.
    let seg_start: Vec<f64> = lens
        .iter()
        .scan(0.0, |acc, l| {
            let s = *acc;
            *acc += l;
            Some(s)
        })
        .collect();
    let total_in: f64 = lens.iter().sum();
    let mut knots: Vec<(f64, f64)> = Vec::with_capacity(2 * n + 2);
    let mut s_out = 0.0;
    // A closed subpath whose seam is rounded starts mid-fillet: the output
    // starts at the first segment's kept start (input `trim_start[0]`).
    let s0_in = if sp.closed && rounded[0] {
        trim_start[0]
    } else {
        0.0
    };
    knots.push((s0_in, 0.0));
    for j in 0..n {
        if j > 0 && rounded[j] {
            if let Some(a) = arc(&kept[j - 1], &kept[j]) {
                s_out += a.arc_len();
                out.push(a);
            }
            knots.push((seg_start[j] + trim_start[j], s_out));
        }
        if let Kept::Piece(c) = &kept[j] {
            s_out += c.arc_len();
            out.push(*c);
        }
        knots.push((seg_start[j] + lens[j] - trim_end[j], s_out));
    }
    if sp.closed && rounded[0] {
        if let Some(a) = arc(&kept[n - 1], &kept[0]) {
            s_out += a.arc_len();
            out.push(a);
        }
        // The seam fillet ends back at the first kept start, one lap on.
        knots.push((total_in + s0_in, s_out));
    }
    knots.dedup_by(|b, a| (b.0 - a.0).abs() < 1e-12 && (b.1 - a.1).abs() < 1e-12);
    (
        Subpath {
            segs: out,
            closed: sp.closed,
        },
        ArcMap {
            knots,
            total_in,
            closed: sp.closed,
        },
    )
}

/// Arc-length correspondence input ↔ filleted output (from
/// [`fillet_corners_mapped`]), piecewise-linear through its knots.
#[derive(Debug, Clone)]
pub struct ArcMap {
    knots: Vec<(f64, f64)>,
    total_in: f64,
    closed: bool,
}

impl ArcMap {
    pub fn identity(total: f64) -> ArcMap {
        ArcMap {
            knots: vec![(0.0, 0.0), (total, total)],
            total_in: total,
            closed: false,
        }
    }

    fn interp(pairs: &[(f64, f64)], x: f64, from_in: bool) -> f64 {
        let key = |p: &(f64, f64)| if from_in { p.0 } else { p.1 };
        let val = |p: &(f64, f64)| if from_in { p.1 } else { p.0 };
        let n = pairs.len();
        if n == 0 {
            return x;
        }
        if n == 1 {
            return val(&pairs[0]) + (x - key(&pairs[0]));
        }
        let i = pairs.partition_point(|p| key(p) <= x).clamp(1, n - 1);
        let (a, b) = (pairs[i - 1], pairs[i]);
        let span = key(&b) - key(&a);
        if span.abs() < 1e-15 {
            return val(&a);
        }
        val(&a) + (x - key(&a)) / span * (val(&b) - val(&a))
    }

    /// Output (filleted) arc length at input arc length `s` (a closed
    /// subpath's input may start mid-seam: `s` is taken modulo one lap, from
    /// the output's start).
    pub fn to_out(&self, s: f64) -> f64 {
        let s = if self.closed && self.total_in > 0.0 {
            let s0 = self.knots.first().map_or(0.0, |k| k.0);
            s0 + (s - s0).rem_euclid(self.total_in)
        } else {
            s
        };
        Self::interp(&self.knots, s, true)
    }

    /// Input arc length at output arc length `s` (in `[0, total_in)` for a
    /// closed subpath).
    pub fn to_in(&self, s: f64) -> f64 {
        let v = Self::interp(&self.knots, s, false);
        if self.closed && self.total_in > 0.0 {
            v.rem_euclid(self.total_in)
        } else {
            v
        }
    }

    /// Output length the knots span.
    pub fn total_out(&self) -> f64 {
        self.knots.last().map_or(0.0, |k| k.1)
    }
}

#[cfg(test)]
mod fillet_tests {
    use super::*;
    use crate::path_geom::vec2::v2;

    fn poly(pts: &[(f64, f64)], closed: bool) -> Subpath {
        let p: Vec<V2> = pts.iter().map(|&(x, y)| v2(x, y)).collect();
        let mut segs: Vec<Cubic> = p.windows(2).map(|w| Cubic::line(w[0], w[1])).collect();
        if closed {
            segs.push(Cubic::line(*p.last().unwrap(), p[0]));
        }
        Subpath { segs, closed }
    }

    #[test]
    fn right_angle_gets_a_circular_fillet() {
        let f = fillet_corners(
            &poly(&[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)], false),
            10.0,
            1e9,
        );
        assert_eq!(f.segs.len(), 3);
        assert!(f.segs[0].0[3].dist(v2(90.0, 0.0)) < 1e-6);
        assert!(f.segs[2].0[0].dist(v2(100.0, 10.0)) < 1e-6);
        // Arc midpoint on the circle centred (90, 10), radius 10.
        let m = f.segs[1].eval(0.5);
        assert!((m.dist(v2(90.0, 10.0)) - 10.0).abs() < 0.01);
        for w in f.segs.windows(2) {
            assert!(w[0].0[3].dist(w[1].0[0]) < 1e-9);
        }
    }

    #[test]
    fn acute_corners_are_capped() {
        // A 15° spike: tan(82.5°) · 10 ≈ 76 px uncapped.
        let a = 15f64.to_radians();
        let f = fillet_corners(
            &poly(
                &[
                    (0.0, 0.0),
                    (200.0, 0.0),
                    (200.0 - 200.0 * a.cos(), 200.0 * a.sin()),
                ],
                false,
            ),
            10.0,
            10.0,
        );
        assert!(f.segs[0].0[3].dist(v2(190.0, 0.0)) < 1e-6);
        let len: f64 = f.segs.iter().map(|c| c.arc_len()).sum();
        assert!(len > 400.0 - 20.0, "{len}");
    }

    fn continuous(f: &Subpath) {
        for w in f.segs.windows(2) {
            assert!(w[0].0[3].dist(w[1].0[0]) < 1e-9, "break at {:?}", w[0].0[3]);
        }
    }

    #[test]
    fn short_middle_segment_joins_its_fillets() {
        // Staircase with a 40 px riser (the #70 review case) and a 10 px one:
        // both corners want more than half the riser, so it vanishes and the
        // two fillets meet on it. No break, and the path stays near the corners.
        for rise in [40.0, 10.0] {
            let f = fillet_corners(
                &poly(
                    &[(0.0, 0.0), (100.0, 0.0), (100.0, rise), (200.0, rise)],
                    false,
                ),
                20.0,
                1e9,
            );
            continuous(&f);
            assert!(f.segs.first().unwrap().0[0].dist(v2(0.0, 0.0)) < 1e-9);
            assert!(f.segs.last().unwrap().0[3].dist(v2(200.0, rise)) < 1e-9);
            // Two lines + two arcs (the riser dropped).
            assert_eq!(f.segs.len(), 4, "{rise}");
            let mid = f.segs[1].0[3];
            assert!(mid.dist(v2(100.0, rise / 2.0)) < 1e-6, "{mid:?}");
        }
    }

    #[test]
    fn trims_of_exactly_half_each() {
        // 90° corners 2·d apart: each trims d = 10, the 20 px middle vanishes exactly.
        let f = fillet_corners(
            &poly(
                &[(0.0, 0.0), (100.0, 0.0), (100.0, 20.0), (0.0, 20.0)],
                false,
            ),
            10.0,
            1e9,
        );
        continuous(&f);
        assert_eq!(f.segs.len(), 4);
    }

    #[test]
    fn arc_map_follows_the_fillets() {
        // 90° corner at input s = 100, fillet radius 10: trims 10 each side.
        let (f, m) = fillet_corners_mapped(
            &poly(&[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)], false),
            10.0,
            1e9,
        );
        let arc = f.segs[1].arc_len();
        assert!((m.to_out(90.0) - 90.0).abs() < 1e-9);
        assert!((m.to_out(110.0) - (90.0 + arc)).abs() < 1e-9);
        assert!(
            (m.to_out(150.0) - (90.0 + arc + 40.0)).abs() < 1e-9,
            "kept pieces are exact"
        );
        assert!((m.to_in(m.to_out(150.0)) - 150.0).abs() < 1e-9);
        assert!((m.total_out() - (180.0 + arc)).abs() < 1e-9);
        // Closed square: the seam fillet makes the output start 5 px in.
        let (_, m) = fillet_corners_mapped(
            &poly(&[(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)], true),
            5.0,
            1e9,
        );
        assert!(m.to_out(5.0).abs() < 1e-9);
        assert!((m.to_in(m.to_out(25.0)) - 25.0).abs() < 1e-9);
        assert!(
            (m.to_in(m.to_out(2.0)) - 2.0).abs() < 1e-9,
            "inside the seam fillet, across the lap"
        );
    }

    #[test]
    fn closed_seam_and_cusps() {
        let f = fillet_corners(
            &poly(&[(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)], true),
            5.0,
            1e9,
        );
        assert_eq!(f.segs.len(), 8);
        assert!(f.segs.last().unwrap().0[3].dist(f.segs[0].0[0]) < 1e-9);
        let cusp = fillet_corners(
            &poly(&[(0.0, 0.0), (50.0, 0.0), (0.0, 0.0)], false),
            5.0,
            1e9,
        );
        assert_eq!(cusp.segs.len(), 2);
    }
}
