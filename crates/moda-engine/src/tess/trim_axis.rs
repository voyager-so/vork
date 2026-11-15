//! Trim along an axis (`trimPathBy: 'x' | 'y'`): the stroke is kept where the path's x (or y) lies in the window
//! `[min + start × (max − min), min + end × (max − min)]` of the whole path's extent on that axis, instead of the arc-length
//! window of `trimPathBy: 'length'` (the default).
//!
//! On a chart line whose points run left to right, `trimPathEnd = t` draws the line up to x = lerp(min x, max x, t):
//! the end moves at a constant speed along the x axis, whatever the segments' lengths, and the end point lies exactly
//! on the segment it cuts (a straight segment is cut at its exact parameter, a curve at the parameter where its x equals
//! the window's edge, found by bisection on a monotone piece).
//!
//! `trimPathOffset` and `trimPathMode` do not apply (every subpath is cut by the one shared window). A path with no
//! extent on the axis (a vertical line trimmed by x) is trimmed by length instead.

use super::dash::{
    cubic_arc_length, quadratic_arc_length, subdivide_cubic, subdivide_quadratic, PathSegment, Pt, Seg,
};

/// Which measure trim windows are fractions of (`trimPathBy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrimBy {
    /// Arc length (`'length'`, the default).
    #[default]
    Length,
    /// The x coordinate across the whole path's width (`'x'`).
    X,
    /// The y coordinate across the whole path's height (`'y'`).
    Y,
}

impl TrimBy {
    /// `trimPathBy`'s stored value; anything else (absent included) is `Length`.
    pub fn parse(s: Option<&str>) -> Self {
        match s {
            Some("x") => TrimBy::X,
            Some("y") => TrimBy::Y,
            _ => TrimBy::Length,
        }
    }

    /// The coordinate index this measure cuts along, or `None` for arc length.
    pub fn axis(self) -> Option<usize> {
        match self {
            TrimBy::Length => None,
            TrimBy::X => Some(0),
            TrimBy::Y => Some(1),
        }
    }
}

/// One kept piece: segment `seg` of the subpath between parameters `t0 < t1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Piece {
    pub seg: usize,
    pub t0: f64,
    pub t1: f64,
}

/// An extent below this (in path units) has no axis to trim along.
const MIN_EXTENT: f64 = 1e-6;
/// Pieces shorter than this in parameter are dropped (a window edge grazing a segment end).
const MIN_PIECE: f64 = 1e-9;

fn coord(seg: &Seg, axis: usize, t: f64) -> f64 {
    let u = 1.0 - t;
    match seg {
        Seg::Line { p0, p1 } => p0[axis] + (p1[axis] - p0[axis]) * t,
        Seg::Quadratic { p0, p1, cp } => u * u * p0[axis] + 2.0 * u * t * cp[axis] + t * t * p1[axis],
        Seg::Cubic { p0, p1, cp1, cp2 } => {
            u * u * u * p0[axis] + 3.0 * u * u * t * cp1[axis] + 3.0 * u * t * t * cp2[axis] + t * t * t * p1[axis]
        }
    }
}

/// The parameters in (0, 1) where the segment's coordinate turns (its derivative's roots).
fn turning_points(seg: &Seg, axis: usize) -> Vec<f64> {
    let mut ts = Vec::new();
    match seg {
        Seg::Line { .. } => {}
        Seg::Quadratic { p0, p1, cp } => {
            let d = p0[axis] - 2.0 * cp[axis] + p1[axis];
            if d.abs() > 1e-12 {
                ts.push((p0[axis] - cp[axis]) / d);
            }
        }
        Seg::Cubic { p0, p1, cp1, cp2 } => {
            // The derivative / 3: a t² + b t + c.
            let a = -p0[axis] + 3.0 * cp1[axis] - 3.0 * cp2[axis] + p1[axis];
            let b = 2.0 * (p0[axis] - 2.0 * cp1[axis] + cp2[axis]);
            let c = cp1[axis] - p0[axis];
            if a.abs() < 1e-12 {
                if b.abs() > 1e-12 {
                    ts.push(-c / b);
                }
            } else {
                let disc = b * b - 4.0 * a * c;
                if disc >= 0.0 {
                    let r = disc.sqrt();
                    ts.push((-b + r) / (2.0 * a));
                    ts.push((-b - r) / (2.0 * a));
                }
            }
        }
    }
    ts.retain(|t| *t > 0.0 && *t < 1.0);
    ts.sort_by(f64::total_cmp);
    ts
}

/// Where the segment's coordinate equals `v`: every parameter in [0, 1]. Straight segments are solved exactly; curves
/// by bisection on each monotone piece (between turning points), to 1e-12.
fn crossings(seg: &Seg, axis: usize, v: f64) -> Vec<f64> {
    if let Seg::Line { p0, p1 } = seg {
        let d = p1[axis] - p0[axis];
        if d == 0.0 {
            return Vec::new();
        }
        let t = (v - p0[axis]) / d;
        return if (0.0..=1.0).contains(&t) { vec![t] } else { Vec::new() };
    }
    let mut knots = vec![0.0];
    knots.extend(turning_points(seg, axis));
    knots.push(1.0);
    let mut out = Vec::new();
    for w in knots.windows(2) {
        let (mut a, mut b) = (w[0], w[1]);
        let (fa, fb) = (coord(seg, axis, a) - v, coord(seg, axis, b) - v);
        if fa == 0.0 {
            out.push(a);
            continue;
        }
        if fb == 0.0 || fa.signum() == fb.signum() {
            continue;
        }
        let rising = fb > fa;
        for _ in 0..60 {
            let m = 0.5 * (a + b);
            if (coord(seg, axis, m) - v > 0.0) == rising {
                b = m;
            } else {
                a = m;
            }
            if b - a < 1e-12 {
                break;
            }
        }
        out.push(0.5 * (a + b));
    }
    if coord(seg, axis, 1.0) == v {
        out.push(1.0);
    }
    out
}

/// The whole path's extent on `axis` (every subpath's end points and curve turning points), or `None` for none.
pub fn extent<'a>(subpaths: impl IntoIterator<Item = &'a [PathSegment]>, axis: usize) -> Option<(f64, f64)> {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for segs in subpaths {
        for s in segs {
            let mut add = |c: f64| {
                if c.is_finite() {
                    lo = lo.min(c);
                    hi = hi.max(c);
                }
            };
            add(coord(&s.seg, axis, 0.0));
            add(coord(&s.seg, axis, 1.0));
            for t in turning_points(&s.seg, axis) {
                add(coord(&s.seg, axis, t));
            }
        }
    }
    lo.is_finite().then_some((lo, hi))
}

/// The coordinate window `[lo, hi]` a trim's `start` / `end` (fractions, clamped to 0…1) select of `ext`, or `None`
/// when the extent is too small to trim along (the caller trims by length).
pub fn window(ext: (f64, f64), start: f64, end: f64) -> Option<(f64, f64)> {
    let w = ext.1 - ext.0;
    if !(w > MIN_EXTENT) {
        return None;
    }
    let (s, e) = (start.clamp(0.0, 1.0), end.clamp(0.0, 1.0));
    Some((ext.0 + s * w, ext.0 + e * w))
}

/// The pieces of one subpath whose coordinate lies in `[lo, hi]`, grouped into runs that join end to start (a run is
/// drawn as one open stroke). On a closed subpath a run through the seam is one run.
pub fn runs(segs: &[PathSegment], closed: bool, axis: usize, lo: f64, hi: f64) -> Vec<Vec<Piece>> {
    let mut out: Vec<Vec<Piece>> = Vec::new();
    if !(hi > lo) {
        return out;
    }
    for (i, s) in segs.iter().enumerate() {
        let mut cuts = vec![0.0, 1.0];
        cuts.extend(crossings(&s.seg, axis, lo));
        cuts.extend(crossings(&s.seg, axis, hi));
        cuts.sort_by(f64::total_cmp);
        cuts.dedup_by(|a, b| (*a - *b).abs() < MIN_PIECE);
        for w in cuts.windows(2) {
            let (t0, t1) = (w[0], w[1]);
            if t1 - t0 < MIN_PIECE {
                continue;
            }
            let c = coord(&s.seg, axis, 0.5 * (t0 + t1));
            if c < lo || c > hi {
                continue;
            }
            let piece = Piece { seg: i, t0, t1 };
            let joins = out.last().and_then(|r| r.last()).is_some_and(|p| {
                (p.seg == i && (p.t1 - t0).abs() < MIN_PIECE) || (p.seg + 1 == i && p.t1 >= 1.0 - MIN_PIECE && t0 <= MIN_PIECE)
            });
            if joins {
                let run = out.last_mut().expect("a run");
                let last = run.last_mut().expect("a piece");
                if last.seg == i {
                    last.t1 = t1;
                } else {
                    run.push(piece);
                }
            } else {
                out.push(vec![piece]);
            }
        }
    }
    // A closed subpath: the last run ending at the seam and the first starting there are one run.
    if closed && out.len() > 1 {
        let starts = out[0][0].seg == 0 && out[0][0].t0 <= MIN_PIECE;
        let last = out.last().and_then(|r| r.last()).copied();
        if starts && last.is_some_and(|p| p.seg + 1 == segs.len() && p.t1 >= 1.0 - MIN_PIECE) {
            let head = out.remove(0);
            out.last_mut().expect("a run").extend(head);
        }
    }
    out
}

/// Segment `s` between parameters `t0` and `t1`, as a segment of the same kind (a straight one cut at its exact
/// parameter).
pub fn piece_segment(s: &PathSegment, t0: f64, t1: f64) -> PathSegment {
    let lerp = |a: Pt, b: Pt, t: f64| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    match &s.seg {
        Seg::Line { p0, p1 } => {
            let (a, b) = (lerp(*p0, *p1, t0), lerp(*p0, *p1, t1));
            PathSegment {
                seg: Seg::Line { p0: a, p1: b },
                arc_length: s.arc_length * (t1 - t0),
            }
        }
        Seg::Quadratic { p0, p1, cp } => {
            let (a, c, b) = subdivide_quadratic(*p0, *cp, *p1, t0, t1);
            PathSegment {
                seg: Seg::Quadratic { p0: a, p1: b, cp: c },
                arc_length: quadratic_arc_length(a, c, b),
            }
        }
        Seg::Cubic { p0, p1, cp1, cp2 } => {
            let (a, c1, c2, b) = subdivide_cubic(*p0, *cp1, *cp2, *p1, t0, t1);
            PathSegment {
                seg: Seg::Cubic { p0: a, p1: b, cp1: c1, cp2: c2 },
                arc_length: cubic_arc_length(a, c1, c2, b),
            }
        }
    }
}

/// A run as segments.
pub fn run_segments(segs: &[PathSegment], run: &[Piece]) -> Vec<PathSegment> {
    run.iter().map(|p| piece_segment(&segs[p.seg], p.t0, p.t1)).collect()
}

/// A run as an arc-length window `[start, end]` on its subpath (`end` may pass the length of a closed subpath when the
/// run goes through the seam), for the lanes that cut strokes by arc length (the outliner, placed brushes).
pub fn run_arc_window(segs: &[PathSegment], run: &[Piece]) -> (f64, f64) {
    let before: Vec<f64> = segs
        .iter()
        .scan(0.0, |acc, s| {
            let at = *acc;
            *acc += s.arc_length;
            Some(at)
        })
        .collect();
    let total: f64 = segs.iter().map(|s| s.arc_length).sum();
    let at = |seg: usize, t: f64| before[seg] + if t <= 0.0 { 0.0 } else { piece_segment(&segs[seg], 0.0, t).arc_length };
    let (first, last) = (run[0], run[run.len() - 1]);
    let start = at(first.seg, first.t0);
    let mut end = at(last.seg, last.t1);
    if end < start {
        end += total;
    }
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(p0: Pt, p1: Pt) -> PathSegment {
        PathSegment {
            seg: Seg::Line { p0, p1 },
            arc_length: (p1[0] - p0[0]).hypot(p1[1] - p0[1]),
        }
    }

    /// A polyline through `pts`.
    fn poly(pts: &[Pt]) -> Vec<PathSegment> {
        pts.windows(2).map(|w| line(w[0], w[1])).collect()
    }

    /// The end point of the last kept piece.
    fn end_of(segs: &[PathSegment], runs: &[Vec<Piece>]) -> Pt {
        let s = run_segments(segs, runs.last().unwrap());
        match s.last().unwrap().seg {
            Seg::Line { p1, .. } | Seg::Quadratic { p1, .. } | Seg::Cubic { p1, .. } => p1,
        }
    }

    #[test]
    fn a_line_with_very_uneven_segments_reaches_the_expected_x_at_each_fraction() {
        // Segment lengths ≈ 1000, 1, 1000.5 and 10: arc-length trim would sit on the first spike for half the time.
        let segs = poly(&[[0.0, 0.0], [10.0, 1000.0], [11.0, 1000.0], [20.0, 0.0], [30.0, 0.0]]);
        let ext = extent([segs.as_slice()], 0).unwrap();
        assert_eq!(ext, (0.0, 30.0));
        for (t, x, y) in [(0.25, 7.5, 750.0), (1.0 / 3.0, 10.0, 1000.0), (0.35, 10.5, 1000.0), (0.5, 15.0, 1000.0 - 1000.0 * 4.0 / 9.0), (0.8, 24.0, 0.0)] {
            let (lo, hi) = window(ext, 0.0, t).unwrap();
            let r = runs(&segs, false, 0, lo, hi);
            assert_eq!(r.len(), 1, "one run at t = {t}");
            let p = end_of(&segs, &r);
            assert!((p[0] - x).abs() < 1e-9, "t = {t}: x = {} (want {x})", p[0]);
            // The end point is exactly on the segment it cuts.
            assert!((p[1] - y).abs() < 1e-6, "t = {t}: y = {} (want {y})", p[1]);
        }
    }

    #[test]
    fn a_curve_is_cut_where_its_x_is_the_window_edge() {
        // An S-shaped cubic from (0, 0) to (100, 0) with x monotone.
        let segs = vec![PathSegment {
            seg: Seg::Cubic { p0: [0.0, 0.0], cp1: [80.0, -60.0], cp2: [20.0, 60.0], p1: [100.0, 0.0] },
            arc_length: cubic_arc_length([0.0, 0.0], [80.0, -60.0], [20.0, 60.0], [100.0, 0.0]),
        }];
        let r = runs(&segs, false, 0, 0.0, 37.0);
        assert_eq!(r.len(), 1);
        let p = end_of(&segs, &r);
        assert!((p[0] - 37.0).abs() < 1e-9, "{p:?}");
    }

    #[test]
    fn a_path_that_turns_back_keeps_every_part_inside_the_window() {
        // Out to x = 100 and back to x = 0 below: a window [0, 50] keeps two runs.
        let segs = poly(&[[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]);
        let r = runs(&segs, false, 0, 0.0, 50.0);
        assert_eq!(r.len(), 2, "{r:?}");
        let second = run_segments(&segs, &r[1]);
        match second[0].seg {
            Seg::Line { p0, p1 } => {
                assert!((p0[0] - 50.0).abs() < 1e-9 && (p1[0] - 0.0).abs() < 1e-9, "{p0:?} {p1:?}");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn a_closed_run_through_the_seam_is_one_run() {
        // A square from its top-left corner: the window x ≤ 10 keeps the left side, which the seam splits.
        let segs = poly(&[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0], [0.0, 0.0]]);
        let r = runs(&segs, true, 0, 0.0, 10.0);
        assert_eq!(r.len(), 1, "{r:?}");
        let (a, b) = run_arc_window(&segs, &r[0]);
        assert!((a - 290.0).abs() < 1e-9 && (b - 410.0).abs() < 1e-9, "{a} {b}");
    }

    #[test]
    fn arc_windows_measure_the_kept_run() {
        let segs = poly(&[[0.0, 0.0], [10.0, 1000.0], [20.0, 0.0]]);
        let r = runs(&segs, false, 0, 0.0, 5.0);
        let (a, b) = run_arc_window(&segs, &r[0]);
        assert_eq!(a, 0.0);
        assert!((b - segs[0].arc_length / 2.0).abs() < 1e-9);
    }

    #[test]
    fn no_extent_means_no_axis_window() {
        let segs = poly(&[[5.0, 0.0], [5.0, 100.0]]);
        assert_eq!(window(extent([segs.as_slice()], 0).unwrap(), 0.0, 0.5), None);
        assert!(window(extent([segs.as_slice()], 1).unwrap(), 0.0, 0.5).is_some());
        assert_eq!(TrimBy::parse(Some("x")), TrimBy::X);
        assert_eq!(TrimBy::parse(Some("length")), TrimBy::Length);
        assert_eq!(TrimBy::parse(None), TrimBy::Length);
    }
}
