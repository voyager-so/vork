//! Blend shape geometry: a source's outline as cubic contours, the point correspondence between two outlines, and
//! interpolation between corresponded outlines.
//!
//! Every segment is a cubic (lines, quadratics and arcs are converted exactly or, for arcs, to within the usual
//! quarter-circle cubic error), so two outlines correspond segment for segment once they have the same number of
//! segments per contour:
//!
//! 1. **Contours pair by index.** A contour with no partner blends to (or from) a point at its place relative to its
//!    shape's centre, so it grows or shrinks in.
//! 2. **Start point and direction are aligned.** Both contours are sampled at even arc length; for closed contours
//!    every start offset in both directions is tried and the one whose samples lie closest to the partner's (each
//!    relative to its contour's centre) wins, and the contour is re-started there (split mid-segment if need be).
//!    Open contours are reversed when that brings the ends closer. Illustrator leaves this to the user's choice of
//!    start points; choosing the closest correspondence keeps redrawn or rotated copies of a shape from twisting.
//! 3. **Anchors are matched by arc length.** Each contour is split at the other's anchor positions (as fractions of
//!    its length), so both have an anchor at every position either had and segment `i` covers the same share of
//!    both outlines. A square's corners meet a circle's points at 45°, not its anchors at 0°, so a square-to-circle
//!    blend is a run of rounded squares rather than a twist (where Illustrator pairs anchors by index).
//!
//! Interpolation is then a lerp of every anchor and control point. A segment stays a line when it is a line in
//! both outlines, so a blend between polygons expands to polygons.

use super::points::{arc_endpoint_to_center, BlendPathPoint};
use crate::value::{json, Value};

pub type Pt = (f64, f64);

fn lerp_pt(a: Pt, b: Pt, t: f64) -> Pt {
    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
}

fn dist(a: Pt, b: Pt) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// One cubic segment. `line` marks a straight segment (control points on the chord at thirds).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Seg {
    pub p0: Pt,
    pub c1: Pt,
    pub c2: Pt,
    pub p3: Pt,
    pub line: bool,
}

impl Seg {
    pub fn line(a: Pt, b: Pt) -> Seg {
        Seg {
            p0: a,
            c1: lerp_pt(a, b, 1.0 / 3.0),
            c2: lerp_pt(a, b, 2.0 / 3.0),
            p3: b,
            line: true,
        }
    }

    pub fn curve(p0: Pt, c1: Pt, c2: Pt, p3: Pt) -> Seg {
        Seg {
            p0,
            c1,
            c2,
            p3,
            line: false,
        }
    }

    pub fn point_at(&self, t: f64) -> Pt {
        let mt = 1.0 - t;
        let (a, b, c, d) = (mt * mt * mt, 3.0 * mt * mt * t, 3.0 * mt * t * t, t * t * t);
        (
            a * self.p0.0 + b * self.c1.0 + c * self.c2.0 + d * self.p3.0,
            a * self.p0.1 + b * self.c1.1 + c * self.c2.1 + d * self.p3.1,
        )
    }

    /// de Casteljau split at `t`.
    pub fn split(&self, t: f64) -> (Seg, Seg) {
        let p01 = lerp_pt(self.p0, self.c1, t);
        let p12 = lerp_pt(self.c1, self.c2, t);
        let p23 = lerp_pt(self.c2, self.p3, t);
        let a = lerp_pt(p01, p12, t);
        let b = lerp_pt(p12, p23, t);
        let m = lerp_pt(a, b, t);
        (
            Seg {
                p0: self.p0,
                c1: p01,
                c2: a,
                p3: m,
                line: self.line,
            },
            Seg {
                p0: m,
                c1: b,
                c2: p23,
                p3: self.p3,
                line: self.line,
            },
        )
    }

    pub fn reversed(&self) -> Seg {
        Seg {
            p0: self.p3,
            c1: self.c2,
            c2: self.c1,
            p3: self.p0,
            line: self.line,
        }
    }

    /// Length estimate: the mean of the chord and the control polygon (exact for lines, close for flat cubics).
    pub fn approx_len(&self) -> f64 {
        let chord = dist(self.p0, self.p3);
        let poly = dist(self.p0, self.c1) + dist(self.c1, self.c2) + dist(self.c2, self.p3);
        (chord + poly) / 2.0
    }

    fn map(&self, f: &impl Fn(Pt) -> Pt) -> Seg {
        Seg {
            p0: f(self.p0),
            c1: f(self.c1),
            c2: f(self.c2),
            p3: f(self.p3),
            line: self.line,
        }
    }

    fn lerp(&self, other: &Seg, t: f64) -> Seg {
        Seg {
            p0: lerp_pt(self.p0, other.p0, t),
            c1: lerp_pt(self.c1, other.c1, t),
            c2: lerp_pt(self.c2, other.c2, t),
            p3: lerp_pt(self.p3, other.p3, t),
            line: self.line && other.line,
        }
    }
}

/// One subpath.
#[derive(Debug, Clone, PartialEq)]
pub struct Contour {
    pub segs: Vec<Seg>,
    pub closed: bool,
}

impl Contour {
    /// The mean of the anchors (a stable centre for alignment; the bbox centre would jump as segments split).
    fn anchor_centroid(&self) -> Pt {
        if self.segs.is_empty() {
            return (0.0, 0.0);
        }
        let n = self.segs.len() as f64;
        let (sx, sy) = self
            .segs
            .iter()
            .fold((0.0, 0.0), |acc, s| (acc.0 + s.p0.0, acc.1 + s.p0.1));
        (sx / n, sy / n)
    }

    fn len(&self) -> f64 {
        self.segs.iter().map(Seg::approx_len).sum()
    }

    /// Split the longest segments in half until there are `n` (the first of equally long ones; a segment with a
    /// NaN length only when every one has). A heap keeps it O(k log n) for k splits: a contour of 100k anchors
    /// blended with a square needs about that many.
    fn subdivide_to(&mut self, n: usize) {
        if self.segs.len() >= n {
            return;
        }
        use std::cmp::{Ordering, Reverse};
        use std::collections::BinaryHeap;
        /// A piece of original segment `i` starting at parameter `t0` of it; pieces sort in outline order by
        /// `(i, t0)`, and halving a piece halves its parameter range.
        struct Piece {
            len: f64,
            i: usize,
            t0: f64,
            t1: f64,
            seg: Seg,
        }
        impl Piece {
            fn new(seg: Seg, i: usize, t0: f64, t1: f64) -> Piece {
                let l = seg.approx_len();
                // The linear scan compared with `>` from -1, so NaN never won over a real length.
                let len = if l.is_nan() { -1.0 } else { l };
                Piece {
                    len,
                    i,
                    t0,
                    t1,
                    seg,
                }
            }
            fn key(&self) -> (f64, Reverse<(usize, f64)>) {
                (self.len, Reverse((self.i, self.t0)))
            }
        }
        impl PartialEq for Piece {
            fn eq(&self, o: &Self) -> bool {
                self.cmp(o) == Ordering::Equal
            }
        }
        impl Eq for Piece {}
        impl PartialOrd for Piece {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Piece {
            fn cmp(&self, o: &Self) -> Ordering {
                let (a, Reverse((ai, at))) = self.key();
                let (b, Reverse((bi, bt))) = o.key();
                a.total_cmp(&b)
                    .then_with(|| bi.cmp(&ai))
                    .then_with(|| bt.total_cmp(&at))
            }
        }
        let mut heap: BinaryHeap<Piece> = self
            .segs
            .iter()
            .enumerate()
            .map(|(i, s)| Piece::new(*s, i, 0.0, 1.0))
            .collect();
        while heap.len() < n {
            let p = heap.pop().expect("a contour being subdivided has segments");
            let (a, b) = p.seg.split(0.5);
            let mid = (p.t0 + p.t1) / 2.0;
            heap.push(Piece::new(a, p.i, p.t0, mid));
            heap.push(Piece::new(b, p.i, mid, p.t1));
        }
        let mut pieces = heap.into_vec();
        pieces.sort_by(|a, b| a.i.cmp(&b.i).then(a.t0.total_cmp(&b.t0)));
        self.segs = pieces.into_iter().map(|p| p.seg).collect();
    }

    fn reversed(&self) -> Contour {
        Contour {
            segs: self.segs.iter().rev().map(Seg::reversed).collect(),
            closed: self.closed,
        }
    }

    fn rotated(&self, shift: usize) -> Contour {
        let n = self.segs.len();
        Contour {
            segs: (0..n).map(|i| self.segs[(i + shift) % n]).collect(),
            closed: self.closed,
        }
    }

    /// A contour of `n` zero-length segments at `p` (the partner of a contour the other outline doesn't have).
    fn point(p: Pt, n: usize, closed: bool, line: bool) -> Contour {
        Contour {
            segs: vec![
                Seg {
                    p0: p,
                    c1: p,
                    c2: p,
                    p3: p,
                    line,
                };
                n.max(1)
            ],
            closed,
        }
    }
}

/// An outline: one or more contours in one coordinate space.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Shape {
    pub contours: Vec<Contour>,
}

/// An axis-aligned box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl Bounds {
    pub fn center(&self) -> Pt {
        (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )
    }
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }
}

/// The cubics of an SVG arc from `p0` (one per quarter turn or less); a degenerate arc is a line.
#[allow(clippy::too_many_arguments)]
fn arc_to_segs(
    p0: Pt,
    rx: f64,
    ry: f64,
    rotation_deg: f64,
    large_arc: bool,
    sweep: bool,
    p3: Pt,
    out: &mut Vec<Seg>,
) {
    let Some((cx, cy, rx, ry, theta1, dtheta, phi)) = arc_endpoint_to_center(
        p0.0,
        p0.1,
        rx,
        ry,
        rotation_deg,
        large_arc,
        sweep,
        p3.0,
        p3.1,
    ) else {
        out.push(Seg::line(p0, p3));
        return;
    };
    let pieces = (dtheta.abs() / std::f64::consts::FRAC_PI_2).ceil().max(1.0) as usize;
    let delta = dtheta / pieces as f64;
    let k = 4.0 / 3.0 * (delta / 4.0).tan();
    let (cp, sp) = (phi.cos(), phi.sin());
    let at = |th: f64| -> Pt {
        let (c, s) = (th.cos(), th.sin());
        (
            cx + rx * c * cp - ry * s * sp,
            cy + rx * c * sp + ry * s * cp,
        )
    };
    let deriv = |th: f64| -> Pt {
        let (c, s) = (th.cos(), th.sin());
        (-rx * s * cp - ry * c * sp, -rx * s * sp + ry * c * cp)
    };
    let mut start = p0;
    for i in 0..pieces {
        let a = theta1 + delta * i as f64;
        let b = a + delta;
        let end = if i + 1 == pieces { p3 } else { at(b) };
        let (da, db) = (deriv(a), deriv(b));
        out.push(Seg::curve(
            start,
            (start.0 + k * da.0, start.1 + k * da.1),
            (end.0 - k * db.0, end.1 - k * db.1),
            end,
        ));
        start = end;
    }
}

impl Shape {
    /// Parse PathNode points. `closed` closes every contour (a PathNode's `closed` applies to all its subpaths).
    /// Zero-length segments are dropped: they carry no outline and would skew the correspondence.
    pub fn from_points(points: &[BlendPathPoint], closed: bool) -> Shape {
        let mut contours: Vec<Contour> = Vec::new();
        let mut segs: Vec<Seg> = Vec::new();
        let mut start: Option<Pt> = None;
        let mut cur: Option<Pt> = None;
        let mut flush = |segs: &mut Vec<Seg>, start: Option<Pt>| {
            if segs.is_empty() {
                return;
            }
            if closed {
                if let (Some(s), Some(last)) = (start, segs.last().map(|g| g.p3)) {
                    if dist(s, last) > 1e-9 {
                        segs.push(Seg::line(last, s));
                    }
                }
            }
            contours.push(Contour {
                segs: std::mem::take(segs),
                closed,
            });
        };
        for p in points {
            let (x, y) = match p {
                BlendPathPoint::Move { x, y }
                | BlendPathPoint::Line { x, y }
                | BlendPathPoint::Curve { x, y, .. }
                | BlendPathPoint::Quadratic { x, y, .. }
                | BlendPathPoint::Arc { x, y, .. } => (*x, *y),
            };
            let end = (x, y);
            let Some(p0) = cur.filter(|_| !matches!(p, BlendPathPoint::Move { .. })) else {
                flush(&mut segs, start);
                start = Some(end);
                cur = Some(end);
                continue;
            };
            match p {
                BlendPathPoint::Move { .. } => unreachable!(),
                BlendPathPoint::Line { .. } => segs.push(Seg::line(p0, end)),
                BlendPathPoint::Curve { cp1, cp2, .. } => {
                    segs.push(Seg::curve(p0, *cp1, *cp2, end))
                }
                BlendPathPoint::Quadratic { cp, .. } => segs.push(Seg::curve(
                    p0,
                    lerp_pt(p0, *cp, 2.0 / 3.0),
                    lerp_pt(end, *cp, 2.0 / 3.0),
                    end,
                )),
                BlendPathPoint::Arc {
                    rx,
                    ry,
                    x_axis_rotation,
                    large_arc,
                    sweep,
                    ..
                } => arc_to_segs(
                    p0,
                    *rx,
                    *ry,
                    *x_axis_rotation,
                    *large_arc,
                    *sweep,
                    end,
                    &mut segs,
                ),
            }
            cur = Some(end);
        }
        flush(&mut segs, start);
        for c in &mut contours {
            c.segs.retain(|s| s.approx_len() > 1e-9);
        }
        contours.retain(|c| !c.segs.is_empty());
        Shape { contours }
    }

    pub fn is_empty(&self) -> bool {
        self.contours.is_empty()
    }

    pub fn is_closed(&self) -> bool {
        !self.contours.is_empty() && self.contours.iter().all(|c| c.closed)
    }

    /// Bounds of the curve itself (each segment sampled), not of its control points.
    pub fn bounds(&self) -> Option<Bounds> {
        let mut b: Option<Bounds> = None;
        let mut add = |p: Pt| {
            b = Some(match b {
                None => Bounds {
                    min_x: p.0,
                    min_y: p.1,
                    max_x: p.0,
                    max_y: p.1,
                },
                Some(b) => Bounds {
                    min_x: b.min_x.min(p.0),
                    min_y: b.min_y.min(p.1),
                    max_x: b.max_x.max(p.0),
                    max_y: b.max_y.max(p.1),
                },
            })
        };
        for c in &self.contours {
            for s in &c.segs {
                if s.line {
                    add(s.p0);
                    add(s.p3);
                } else {
                    for i in 0..=16 {
                        add(s.point_at(i as f64 / 16.0));
                    }
                }
            }
        }
        b
    }

    pub fn center(&self) -> Option<Pt> {
        self.bounds().map(|b| b.center())
    }

    pub fn map(&self, f: impl Fn(Pt) -> Pt) -> Shape {
        Shape {
            contours: self
                .contours
                .iter()
                .map(|c| Contour {
                    segs: c.segs.iter().map(|s| s.map(&f)).collect(),
                    closed: c.closed,
                })
                .collect(),
        }
    }

    pub fn translated(&self, dx: f64, dy: f64) -> Shape {
        self.map(|p| (p.0 + dx, p.1 + dy))
    }

    /// Rotated by `angle` radians about the origin.
    pub fn rotated(&self, angle: f64) -> Shape {
        if angle == 0.0 {
            return self.clone();
        }
        let (c, s) = (angle.cos(), angle.sin());
        self.map(|p| (p.0 * c - p.1 * s, p.0 * s + p.1 * c))
    }

    /// Lerp two outlines that already correspond ([`correspond`]'s output).
    pub fn lerp(&self, other: &Shape, t: f64) -> Shape {
        Shape {
            contours: self
                .contours
                .iter()
                .zip(&other.contours)
                .map(|(a, b)| Contour {
                    segs: a
                        .segs
                        .iter()
                        .zip(&b.segs)
                        .map(|(x, y)| x.lerp(y, t))
                        .collect(),
                    closed: a.closed && b.closed,
                })
                .collect(),
        }
    }

    /// PathNode points. A closed contour's final line back to its start is left to `closed`.
    pub fn to_points(&self) -> Vec<Value> {
        let mut out = Vec::new();
        for c in &self.contours {
            let Some(first) = c.segs.first() else {
                continue;
            };
            out.push(json!({"type": "move", "x": first.p0.0, "y": first.p0.1}));
            let n = c.segs.len();
            for (i, s) in c.segs.iter().enumerate() {
                if c.closed && i + 1 == n && s.line && n > 1 {
                    break;
                }
                if s.line {
                    out.push(json!({"type": "line", "x": s.p3.0, "y": s.p3.1}));
                } else {
                    out.push(json!({
                        "type": "curve", "x": s.p3.0, "y": s.p3.1,
                        "cp1": {"x": s.c1.0, "y": s.c1.1},
                        "cp2": {"x": s.c2.0, "y": s.c2.1},
                    }));
                }
            }
        }
        out
    }

    /// Sample the outline as a polyline (for spines): `per_curve` points per curved segment.
    pub fn polyline(&self, per_curve: usize) -> Vec<Pt> {
        let mut out: Vec<Pt> = Vec::new();
        // A spine is one path: its first contour.
        let Some(c) = self.contours.first() else {
            return out;
        };
        for s in &c.segs {
            if out.is_empty() {
                out.push(s.p0);
            }
            if s.line {
                out.push(s.p3);
            } else {
                for i in 1..=per_curve {
                    out.push(s.point_at(i as f64 / per_curve as f64));
                }
            }
        }
        out
    }
}

/// Samples per segment for arc-length tables.
const ARC_SAMPLES: usize = 16;
/// Samples per contour for choosing the alignment of closed contours.
const ALIGN_SAMPLES: usize = 64;
/// Arc-length fractions closer than this are one anchor position.
const FRACTION_EPS: f64 = 1e-3;

/// Cumulative arc length of a segment at `t = i / ARC_SAMPLES`.
fn seg_table(s: &Seg) -> [f64; ARC_SAMPLES + 1] {
    let mut out = [0.0; ARC_SAMPLES + 1];
    if s.line {
        let l = dist(s.p0, s.p3);
        for (i, v) in out.iter_mut().enumerate() {
            *v = l * i as f64 / ARC_SAMPLES as f64;
        }
        return out;
    }
    let mut prev = s.p0;
    for i in 1..=ARC_SAMPLES {
        let p = s.point_at(i as f64 / ARC_SAMPLES as f64);
        out[i] = out[i - 1] + dist(prev, p);
        prev = p;
    }
    out
}

/// A contour's arc-length parametrisation: each segment's table and start length, and the total.
struct ArcParam {
    tables: Vec<[f64; ARC_SAMPLES + 1]>,
    starts: Vec<f64>,
    total: f64,
}

impl ArcParam {
    fn new(c: &Contour) -> ArcParam {
        let tables: Vec<_> = c.segs.iter().map(seg_table).collect();
        let mut starts = Vec::with_capacity(tables.len());
        let mut total = 0.0;
        for t in &tables {
            starts.push(total);
            total += t[ARC_SAMPLES];
        }
        ArcParam {
            tables,
            starts,
            total,
        }
    }

    /// Anchor positions as fractions of the length (segment starts after the first).
    fn anchor_fractions(&self) -> Vec<f64> {
        if self.total <= 0.0 {
            return Vec::new();
        }
        self.starts.iter().skip(1).map(|s| s / self.total).collect()
    }

    /// (segment, t) at arc-length fraction `f`.
    fn locate(&self, f: f64) -> (usize, f64) {
        let target = f.clamp(0.0, 1.0) * self.total;
        let i = self
            .starts
            .partition_point(|&s| s <= target)
            .saturating_sub(1)
            .min(self.tables.len() - 1);
        let table = &self.tables[i];
        let local = target - self.starts[i];
        let j = table.partition_point(|&v| v < local).clamp(1, ARC_SAMPLES);
        let (a, b) = (table[j - 1], table[j]);
        let frac = if b > a {
            ((local - a) / (b - a)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (i, ((j - 1) as f64 + frac) / ARC_SAMPLES as f64)
    }
}

/// `n` points at even arc-length fractions (the first at the start).
fn arc_samples(c: &Contour, n: usize) -> Vec<Pt> {
    let param = ArcParam::new(c);
    (0..n)
        .map(|k| {
            let (i, t) = param.locate(k as f64 / n as f64);
            c.segs[i].point_at(t)
        })
        .collect()
}

fn centroid(pts: &[Pt]) -> Pt {
    let n = pts.len().max(1) as f64;
    let (x, y) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    (x / n, y / n)
}

/// Split `c` at arc-length fractions `fs` (ascending), except where it already has an anchor within
/// [`FRACTION_EPS`] (or another split already landed).
fn split_at_fractions(c: &Contour, fs: &[f64]) -> Contour {
    let param = ArcParam::new(c);
    if param.total <= 0.0 {
        return c.clone();
    }
    let own = param.anchor_fractions();
    // Per segment, the local t values to split at.
    let mut cuts: Vec<Vec<f64>> = vec![Vec::new(); c.segs.len()];
    // Both lists ascend (`fs` is the other contour's anchor fractions), so the nearest taken fraction is a
    // binary search away in `own` and the last one in `split`: linear in the anchors, where scanning every
    // taken fraction per cut was quadratic (a 16k-anchor blend took seconds).
    let mut split: Vec<f64> = Vec::new();
    let near = |sorted: &[f64], f: f64| {
        let k = sorted.partition_point(|&g| g < f);
        [k.checked_sub(1), Some(k)]
            .into_iter()
            .flatten()
            .filter_map(|j| sorted.get(j))
            .any(|g| (g - f).abs() < FRACTION_EPS)
    };
    for &f in fs {
        if f <= FRACTION_EPS
            || f >= 1.0 - FRACTION_EPS
            || near(&own, f)
            || split
                .iter()
                .rev()
                .take_while(|&&g| g >= f - FRACTION_EPS)
                .any(|g| (g - f).abs() < FRACTION_EPS)
        {
            continue;
        }
        split.push(f);
        let (i, t) = param.locate(f);
        if t > 1e-6 && t < 1.0 - 1e-6 {
            cuts[i].push(t);
        }
    }
    let mut segs = Vec::with_capacity(c.segs.len() + fs.len());
    for (seg, mut ts) in c.segs.iter().zip(cuts) {
        ts.sort_by(f64::total_cmp);
        let mut rest = *seg;
        let mut done = 0.0;
        for t in ts {
            // Re-map t into what is left of the segment.
            let local = (t - done) / (1.0 - done);
            let (a, b) = rest.split(local);
            segs.push(a);
            rest = b;
            done = t;
        }
        segs.push(rest);
    }
    Contour {
        segs,
        closed: c.closed,
    }
}

/// `c` (closed) re-started at arc-length fraction `f`: an existing anchor within [`FRACTION_EPS`] is used as is.
fn restart_at(c: &Contour, f: f64) -> Contour {
    let param = ArcParam::new(c);
    if param.total <= 0.0 || f <= FRACTION_EPS || f >= 1.0 - FRACTION_EPS {
        return c.clone();
    }
    let own = param.anchor_fractions();
    if let Some(k) = own.iter().position(|g| (g - f).abs() < FRACTION_EPS) {
        return c.rotated(k + 1);
    }
    let (i, t) = param.locate(f);
    let (a, b) = c.segs[i].split(t);
    let mut segs = vec![b];
    segs.extend_from_slice(&c.segs[i + 1..]);
    segs.extend_from_slice(&c.segs[..i]);
    segs.push(a);
    Contour {
        segs,
        closed: c.closed,
    }
}

/// Sum of squared distances between two equal-length sample lists, each relative to its centroid.
fn sample_cost(a: &[Pt], b: &[Pt]) -> f64 {
    let (ca, cb) = (centroid(a), centroid(b));
    a.iter()
        .zip(b)
        .map(|(p, q)| {
            let dx = (p.0 - ca.0) - (q.0 - cb.0);
            let dy = (p.1 - ca.1) - (q.1 - cb.1);
            dx * dx + dy * dy
        })
        .sum()
}

/// Orient and re-start `b` to line up with `a` by arc length: for closed contours, the direction and start that
/// minimise the distance between evenly spaced samples; for open ones, the direction that brings the ends closer.
fn align(a: &Contour, b: &Contour) -> Contour {
    if a.closed && b.closed {
        let sa = arc_samples(a, ALIGN_SAMPLES);
        let mut best = (f64::INFINITY, b.clone(), 0usize);
        for cand in [b.clone(), b.reversed()] {
            let sb = arc_samples(&cand, ALIGN_SAMPLES);
            for shift in 0..ALIGN_SAMPLES {
                let rotated: Vec<Pt> = (0..ALIGN_SAMPLES)
                    .map(|i| sb[(i + shift) % ALIGN_SAMPLES])
                    .collect();
                let cost = sample_cost(&sa, &rotated);
                if cost < best.0 - 1e-9 {
                    best = (cost, cand.clone(), shift);
                }
            }
        }
        restart_at(&best.1, best.2 as f64 / ALIGN_SAMPLES as f64)
    } else {
        let ends = |c: &Contour| -> Vec<Pt> {
            let mut v = arc_samples(c, 2);
            v.push(c.segs.last().map_or((0.0, 0.0), |s| s.p3));
            v
        };
        let r = b.reversed();
        if sample_cost(&ends(a), &ends(&r)) < sample_cost(&ends(a), &ends(b)) - 1e-9 {
            r
        } else {
            b.clone()
        }
    }
}

/// Make two contours correspond: aligned ([`align`]), then each split at the other's anchor positions by arc-length
/// fraction, so segment `i` covers the same share of the outline in both.
fn correspond_contours(a: &Contour, b: &Contour) -> (Contour, Contour) {
    let b = align(a, b);
    let (fa, fb) = (
        ArcParam::new(a).anchor_fractions(),
        ArcParam::new(&b).anchor_fractions(),
    );
    let (mut x, mut y) = (split_at_fractions(a, &fb), split_at_fractions(&b, &fa));
    // Anchors that nearly coincide can leave the counts one apart; even them out.
    let n = x.segs.len().max(y.segs.len());
    x.subdivide_to(n);
    y.subdivide_to(n);
    (x, y)
}

/// Make two outlines correspond (see the module docs): the results have the same contours with the same segment
/// counts, lined up, so [`Shape::lerp`] morphs one into the other. A closed and an open outline blend as open ones
/// (the closed one opens at its start).
pub fn correspond(a: &Shape, b: &Shape) -> (Shape, Shape) {
    let closed = a.is_closed() && b.is_closed();
    let open = |s: &Shape| Shape {
        contours: s
            .contours
            .iter()
            .map(|c| Contour {
                segs: c.segs.clone(),
                closed,
            })
            .collect(),
    };
    let (a, b) = (open(a), open(b));
    let ca = a.center().unwrap_or((0.0, 0.0));
    let cb = b.center().unwrap_or((0.0, 0.0));
    let n = a.contours.len().max(b.contours.len());
    let mut out_a = Shape::default();
    let mut out_b = Shape::default();
    for i in 0..n {
        let (x, y) = match (a.contours.get(i), b.contours.get(i)) {
            (Some(x), Some(y)) => correspond_contours(x, y),
            (Some(x), None) => {
                // Shrinks to a point at its place relative to the centre.
                let p = x.anchor_centroid();
                let at = (cb.0 + p.0 - ca.0, cb.1 + p.1 - ca.1);
                let line = x.segs.iter().all(|s| s.line);
                (x.clone(), Contour::point(at, x.segs.len(), closed, line))
            }
            (None, Some(y)) => {
                let p = y.anchor_centroid();
                let at = (ca.0 + p.0 - cb.0, ca.1 + p.1 - cb.1);
                let line = y.segs.iter().all(|s| s.line);
                (Contour::point(at, y.segs.len(), closed, line), y.clone())
            }
            (None, None) => unreachable!(),
        };
        out_a.contours.push(x);
        out_b.contours.push(y);
    }
    (out_a, out_b)
}

/// Total outline length (used by tests and the smooth-colour step count).
pub fn shape_len(s: &Shape) -> f64 {
    s.contours.iter().map(Contour::len).sum()
}
