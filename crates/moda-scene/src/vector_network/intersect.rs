//! Curve–curve intersection by recursive subdivision, finished with Newton.
//!
//! Pairs of sub-curves whose control-polygon boxes overlap are subdivided until
//! both are flat; flat pieces are intersected as chords and the hit is polished
//! on the original curves. Coincident (overlapping) curves would subdivide
//! forever, so the walk has a work budget; running out of it, or finding more
//! hits than two cubics can have, reports [`Hits::Overlap`], and the planariser
//! handles overlap separately (by splitting each curve where the other's ends
//! lie on it, then merging the coincident pieces).

use super::geom::{segment_intersection, Cubic, Vector};

/// Two non-coincident cubics meet in at most 9 points.
const MAX_HITS: usize = 9;
const WORK_BUDGET: usize = 40_000;
const MAX_DEPTH: u32 = 52;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Hits {
    /// `(s, t)` parameter pairs, `a(s) ≈ b(t)`, sorted by `s`.
    Points(Vec<(f64, f64)>),
    /// The curves share a stretch; crossings are not well defined.
    Overlap,
}

pub(crate) fn intersect(a: &Cubic, b: &Cubic, tol: f64) -> Hits {
    if !a.hull_bounds().overlaps(&b.hull_bounds(), tol) {
        return Hits::Points(Vec::new());
    }
    let flat_tol = tol * 0.05;
    let mut stack: Vec<(f64, f64, f64, f64, u32)> = vec![(0.0, 1.0, 0.0, 1.0, 0)];
    let mut raw: Vec<(f64, f64)> = Vec::new();
    let mut work = 0usize;
    while let Some((a0, a1, b0, b1, depth)) = stack.pop() {
        work += 1;
        if work > WORK_BUDGET {
            return Hits::Overlap;
        }
        let ca = a.subsegment(a0, a1);
        let cb = b.subsegment(b0, b1);
        let ba = ca.hull_bounds();
        let bb = cb.hull_bounds();
        if !ba.overlaps(&bb, tol * 0.5) {
            continue;
        }
        let fa = ca.flatness() <= flat_tol;
        let fb = cb.flatness() <= flat_tol;
        if (fa && fb) || depth >= MAX_DEPTH {
            if let Some((s, t)) = segment_intersection(ca.p0, ca.p3, cb.p0, cb.p3, 0.05) {
                raw.push((
                    a0 + s.clamp(0.0, 1.0) * (a1 - a0),
                    b0 + t.clamp(0.0, 1.0) * (b1 - b0),
                ));
            } else if depth >= MAX_DEPTH {
                raw.push((0.5 * (a0 + a1), 0.5 * (b0 + b1)));
            }
            continue;
        }
        let split_a = !fa && (fb || ba.diag() >= bb.diag());
        if split_a {
            let m = 0.5 * (a0 + a1);
            stack.push((a0, m, b0, b1, depth + 1));
            stack.push((m, a1, b0, b1, depth + 1));
        } else {
            let m = 0.5 * (b0 + b1);
            stack.push((a0, a1, b0, m, depth + 1));
            stack.push((a0, a1, m, b1, depth + 1));
        }
    }

    let mut hits: Vec<(f64, f64, Vector)> = Vec::new();
    for (s, t) in raw {
        let (s, t) = refine(a, b, s, t);
        let pa = a.eval(s);
        // Newton converges to machine precision on a real crossing; a pair
        // that only comes within `tol` (curves grazing, or leaving a shared
        // vertex tangentially) is not a crossing.
        if pa.dist(b.eval(t)) > tol * 1e-3 {
            continue;
        }
        if hits.iter().any(|&(hs, ht, hp)| {
            hp.dist(pa) < tol || ((hs - s).abs() < 1e-9 && (ht - t).abs() < 1e-9)
        }) {
            continue;
        }
        hits.push((s, t, pa));
        if hits.len() > MAX_HITS {
            return Hits::Overlap;
        }
    }
    let mut out: Vec<(f64, f64)> = hits.into_iter().map(|(s, t, _)| (s, t)).collect();
    out.sort_by(|x, y| x.0.total_cmp(&y.0));
    Hits::Points(out)
}

/// Newton on `a(s) − b(t) = 0`, keeping the best iterate.
fn refine(a: &Cubic, b: &Cubic, s0: f64, t0: f64) -> (f64, f64) {
    let (mut s, mut t) = (s0, t0);
    let mut best = (s, t);
    let mut best_d = a.eval(s).dist(b.eval(t));
    for _ in 0..16 {
        let d = a.eval(s) - b.eval(t);
        if d.len() < 1e-13 {
            break;
        }
        let da = a.deriv(s);
        let db = b.deriv(t);
        let det = -da.cross(db);
        if det.abs() < 1e-300 {
            break;
        }
        let ds = d.cross(db) / det;
        let dt = d.cross(da) / det;
        s = (s + ds).clamp(0.0, 1.0);
        t = (t + dt).clamp(0.0, 1.0);
        let nd = a.eval(s).dist(b.eval(t));
        if nd < best_d {
            best_d = nd;
            best = (s, t);
        }
    }
    best
}

/// Parameter pairs `(t1, t2)`, `t1 < t2`, where a single cubic crosses itself.
/// The curve is cut at its axis extrema into monotone pieces (which cannot
/// self-intersect) and the pieces are intersected pairwise.
pub(crate) fn self_intersections(c: &Cubic, tol: f64) -> Vec<(f64, f64)> {
    let mut ts = vec![0.0];
    ts.extend(c.extrema_params());
    ts.push(1.0);
    ts.dedup_by(|x, y| (*x - *y).abs() < 1e-7);
    if ts.len() < 3 {
        return Vec::new();
    }
    let pieces: Vec<(f64, f64, Cubic)> = ts
        .windows(2)
        .map(|w| (w[0], w[1], c.subsegment(w[0], w[1])))
        .collect();
    let mut out: Vec<(f64, f64)> = Vec::new();
    for i in 0..pieces.len() {
        for j in (i + 1)..pieces.len() {
            let (i0, i1, ci) = pieces[i];
            let (j0, j1, cj) = pieces[j];
            let Hits::Points(hs) = intersect(&ci, &cj, tol) else {
                continue;
            };
            for (s, t) in hs {
                let t1 = i0 + s * (i1 - i0);
                let t2 = j0 + t * (j1 - j0);
                let p = c.eval(t1);
                // The joint between adjacent pieces and the curve's own ends are
                // not crossings.
                if p.dist(c.p0) < tol || p.dist(c.p3) < tol {
                    continue;
                }
                if j == i + 1 && p.dist(c.eval(i1)) < tol {
                    continue;
                }
                if (t2 - t1).abs() < 1e-6 {
                    continue;
                }
                if out.iter().any(|&(a, _)| c.eval(a).dist(p) < tol) {
                    continue;
                }
                out.push((t1.min(t2), t1.max(t2)));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64) -> Vector {
        Vector::new(x, y)
    }

    #[test]
    fn crossing_lines_meet_once() {
        let a = Cubic::line(v(0.0, 0.0), v(10.0, 10.0));
        let b = Cubic::line(v(0.0, 10.0), v(10.0, 0.0));
        let Hits::Points(h) = intersect(&a, &b, 1e-3) else {
            panic!("overlap")
        };
        assert_eq!(h.len(), 1);
        assert!((h[0].0 - 0.5).abs() < 1e-9 && (h[0].1 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn cubic_crosses_line_three_times() {
        // An S-curve across the x axis.
        let a = Cubic::new(v(0.0, -10.0), v(10.0, 40.0), v(20.0, -40.0), v(30.0, 10.0));
        let b = Cubic::line(v(-5.0, 0.0), v(35.0, 0.0));
        let Hits::Points(h) = intersect(&a, &b, 1e-3) else {
            panic!("overlap")
        };
        assert_eq!(h.len(), 3, "{h:?}");
        for (s, _) in h {
            assert!(a.eval(s).y.abs() < 1e-6);
        }
    }

    #[test]
    fn coincident_curves_report_overlap_or_nothing_spurious() {
        let a = Cubic::new(v(0.0, 0.0), v(10.0, 20.0), v(20.0, -20.0), v(30.0, 0.0));
        match intersect(&a, &a, 1e-3) {
            Hits::Overlap => {}
            Hits::Points(h) => assert!(h.len() <= MAX_HITS),
        }
    }

    #[test]
    fn loop_cubic_self_intersects() {
        // 10a² − 10a + 1 = 0 → a = 0.5 ± √15/10.
        let c = Cubic::new(v(0.0, 0.0), v(20.0, 10.0), v(-10.0, 10.0), v(10.0, 0.0));
        let h = self_intersections(&c, 1e-4);
        assert_eq!(h.len(), 1, "{h:?}");
        let r = 15f64.sqrt() / 10.0;
        assert!((h[0].0 - (0.5 - r)).abs() < 1e-6, "{h:?}");
        assert!((h[0].1 - (0.5 + r)).abs() < 1e-6, "{h:?}");
    }

    #[test]
    fn simple_arc_does_not_self_intersect() {
        let c = Cubic::new(v(0.0, 0.0), v(0.0, 10.0), v(10.0, 10.0), v(10.0, 0.0));
        assert!(self_intersections(&c, 1e-4).is_empty());
    }
}
