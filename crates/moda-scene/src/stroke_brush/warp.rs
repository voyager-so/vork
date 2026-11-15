//! Nonlinear warp of cubic paths, output kept cubic (brushes.md §5: "warp is
//! nonlinear: subdivide adaptively and refit to cubics within a tolerance").
//!
//! Each source cubic `c` is first cut where the caller says the map changes
//! character (`cuts`: spine knots, guide boundaries), then approximated piece
//! by piece with the Hermite cubic of its image: ends `W(c(0))`, `W(c(1))`,
//! end derivatives `J_W · c'` (central finite differences of `W` in the
//! direction of the handle). The approximation is checked against the true
//! image at seven interior parameters; a piece off by more than `tolerance`
//! is split in half and retried, up to `MAX_DEPTH`. Neighbouring pieces share
//! their end derivative, so the output is G1 wherever the map is smooth.
//!
//! A shared work budget (`WarpBudget`) bounds the pieces examined on any
//! input; once it runs out the warp stops and flags `exhausted`, and the
//! placement fails with a clear `too_complex` error instead of stalling.

use super::geom::{norm, split, sub};
use crate::path_geom::vec2::{Cubic, V2};

const MAX_DEPTH: u32 = 14;

/// (source piece, image + derivative at its start, at its end, depth).
type Pending = (Cubic, (V2, V2), (V2, V2), u32);
const CHECK_T: [f64; 7] = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875];

/// Shared work limit across one placement: pieces examined (each costs about
/// a dozen map evaluations). `exhausted` is set when it ran out.
#[derive(Debug, Clone, Copy)]
pub struct WarpBudget {
    pub remaining: usize,
    pub exhausted: bool,
}

impl WarpBudget {
    pub fn new(pieces: usize) -> WarpBudget {
        WarpBudget {
            remaining: pieces,
            exhausted: false,
        }
    }
}

pub struct Warper<'a> {
    pub map: &'a dyn Fn(V2) -> V2,
    pub tolerance: f64,
    /// Finite-difference step, in source units.
    pub step: f64,
}

impl Warper<'_> {
    fn derivative(&self, p: V2, dir: V2) -> V2 {
        let l = norm(dir);
        if !l.is_finite() || l <= 1e-12 {
            return V2::default();
        }
        let h = self.step.min((l * 1e-3).max(1e-12));
        let u = dir * (1.0 / l);
        let d = ((self.map)(p + u * h) - (self.map)(p - u * h)) * (l / (2.0 * h));
        if d.is_finite() {
            d
        } else {
            V2::default()
        }
    }

    /// One-sided derivative from inside the piece (`forward` from its start,
    /// backward from its end), so a piece that ends on a kink of the map
    /// takes the tangent of its own side.
    fn one_sided(&self, p: V2, img: V2, dir: V2, forward: bool) -> V2 {
        let l = norm(dir);
        if !l.is_finite() || l <= 1e-12 {
            return V2::default();
        }
        // Never step past a short piece's other end.
        let h = self.step.min((l * 1e-3).max(1e-12));
        let u = dir * (1.0 / l);
        let d = if forward {
            ((self.map)(p + u * h) - img) * (l / h)
        } else {
            (img - (self.map)(p - u * h)) * (l / h)
        };
        if d.is_finite() {
            d
        } else {
            V2::default()
        }
    }

    /// Image position and derivative at an end (`t` = 0 or 1) of `c`,
    /// differenced from inside the piece.
    fn end_data(&self, c: &Cubic, t: f64) -> (V2, V2) {
        let p = c.eval(t);
        let img = (self.map)(p);
        (img, self.one_sided(p, img, c.d1(t), t < 0.5))
    }

    /// Warp one source cubic, cut at the sorted parameters `cuts` (in (0, 1)),
    /// appending the image to `out`.
    pub fn warp(&self, c: &Cubic, cuts: &[f64], out: &mut Vec<Cubic>, budget: &mut WarpBudget) {
        let mut t0 = 0.0;
        for &t1 in cuts.iter().chain(std::iter::once(&1.0)) {
            if t1 <= t0 + 1e-9 {
                continue;
            }
            let piece = sub(c, t0, t1);
            self.warp_piece(&piece, out, budget);
            t0 = t1;
        }
    }

    /// Adaptive Hermite fit. Split points share their image position and
    /// derivative between the two halves (computed once).
    fn warp_piece(&self, c: &Cubic, out: &mut Vec<Cubic>, budget: &mut WarpBudget) {
        // (source piece, image start, derivative at start, image end, derivative at end, depth);
        // derivatives are w.r.t. the ORIGINAL piece's parameter scaled to this piece.
        let a = self.end_data(c, 0.0);
        let d = self.end_data(c, 1.0);
        let mut stack: Vec<Pending> = vec![(*c, a, d, 0)];
        while let Some((piece, (p0, d0), (p3, d3), depth)) = stack.pop() {
            if budget.remaining == 0 {
                budget.exhausted = true;
                return;
            }
            budget.remaining -= 1;
            let h = Cubic([p0, p0 + d0 * (1.0 / 3.0), p3 - d3 * (1.0 / 3.0), p3]);
            let mut mid: Option<V2> = None;
            let good = depth >= MAX_DEPTH
                || CHECK_T.iter().all(|&t| {
                    let img = (self.map)(piece.eval(t));
                    if t == 0.5 {
                        mid = Some(img);
                    }
                    let e = norm(img - h.eval(t));
                    e <= self.tolerance || !e.is_finite()
                });
            if good {
                if h.0.iter().all(|p| p.is_finite()) {
                    out.push(h);
                }
                continue;
            }
            let (l, r) = split(&piece, 0.5);
            let m = r.0[0];
            let pm = mid.unwrap_or_else(|| (self.map)(m));
            // Derivatives of the halves' own parameters: half the parent's.
            let dm = self.derivative(m, r.d1(0.0));
            stack.push((r, (pm, dm), (p3, d3 * 0.5), depth + 1));
            stack.push((l, (p0, d0 * 0.5), (pm, dm), depth + 1));
        }
    }
}

/// Parameters in (0, 1) where `coord(c(t)) == value` for each `value` in the
/// sorted `values` (monotone intervals + bisection). `coord` picks the coordinate (e.g. the along axis).
pub fn crossings(c: &Cubic, coord: impl Fn(V2) -> f64, values: &[f64], max: usize) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let lo = c.0.iter().map(|p| coord(*p)).fold(f64::INFINITY, f64::min);
    let hi =
        c.0.iter()
            .map(|p| coord(*p))
            .fold(f64::NEG_INFINITY, f64::max);
    let a = values.partition_point(|v| *v <= lo);
    let b = values.partition_point(|v| *v < hi);
    if a >= b {
        return Vec::new();
    }
    let mut parameters = vec![0.0];
    parameters.extend(c.extrema(&coord));
    parameters.push(1.0);
    let samples: Vec<f64> = parameters.iter().map(|&t| coord(c.eval(t))).collect();
    let mut out = Vec::new();
    for &v in &values[a..b] {
        for i in 0..parameters.len() - 1 {
            let (f0, f1) = (samples[i] - v, samples[i + 1] - v);
            if f0 == 0.0 && i > 0 {
                out.push(parameters[i]);
                continue;
            }
            if (f0 < 0.0) != (f1 < 0.0) && f1 != 0.0 {
                let (mut x0, mut x1) = (parameters[i], parameters[i + 1]);
                let mut g0 = f0;
                for _ in 0..40 {
                    let xm = (x0 + x1) / 2.0;
                    let gm = coord(c.eval(xm)) - v;
                    if (gm < 0.0) == (g0 < 0.0) {
                        x0 = xm;
                        g0 = gm;
                    } else {
                        x1 = xm;
                    }
                }
                out.push((x0 + x1) / 2.0);
            }
        }
        if out.len() >= max {
            break;
        }
    }
    out.retain(|t| *t > 1e-9 && *t < 1.0 - 1e-9);
    out.sort_by(|p, q| p.total_cmp(q));
    out.dedup_by(|p, q| (*p - *q).abs() < 1e-9);
    out.truncate(max);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path_geom::vec2::v2;

    #[test]
    fn crossings_isolate_three_clustered_roots_and_linear_coordinates() {
        for shift in [0.0, 0.2, 0.8] {
            let roots = [shift + 0.01, shift + 0.02, shift + 0.03];
            let [r, s, t] = roots;
            let (a, b, d) = (-(r+s+t), r*s+r*t+s*t, -r*s*t);
            // Power coefficients t³ + a·t² + b·t + d to Bezier controls.
            let values = [d, d+b/3.0, d+2.0*b/3.0+a/3.0, d+b+a+1.0];
            let c = Cubic(values.map(|x| v2(x, 0.0)));
            let got = crossings(&c, |p| 3.0*p.x + 2.0*p.y, &[0.0], 16);
            assert_eq!(got.len(), 3, "{got:?}");
            for (want, got) in roots.into_iter().zip(got) {
                assert!((want-got).abs() < 1e-8, "{want} != {got}");
            }
        }
    }

    #[test]
    fn crossings_find_excursions_between_uniform_samples() {
        for sign in [-1.0, 1.0] {
            let c = Cubic([
                v2(9.0 * sign, 0.0),
                v2(-247.0 * sign, 1.0),
                v2(3593.0 * sign, 2.0),
                v2(11529.0 * sign, 3.0),
            ]);
            let roots = crossings(&c, |p| p.x, &[0.0], 16);
            assert_eq!(roots.len(), 2, "{roots:?}");
            assert!((roots[0] - 1.0 / 64.0).abs() < 1e-9);
            assert!((roots[1] - 3.0 / 64.0).abs() < 1e-9);
        }
    }

    #[test]
    fn affine_map_is_exact_in_one_piece() {
        let map = |p: V2| v2(2.0 * p.x + 1.0, -p.y + 3.0 * p.x);
        let w = Warper {
            map: &map,
            tolerance: 1e-6,
            step: 1e-3,
        };
        let c = Cubic([v2(0.0, 0.0), v2(1.0, 4.0), v2(3.0, -2.0), v2(5.0, 1.0)]);
        let mut out = Vec::new();
        let mut budget = WarpBudget::new(1000);
        w.warp(&c, &[], &mut out, &mut budget);
        assert_eq!(out.len(), 1);
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            assert!(out[0].eval(t).dist(map(c.eval(t))) < 1e-6);
        }
    }

    #[test]
    fn polar_map_stays_within_tolerance() {
        // Wrap a line around a circle of radius 100.
        let map = |p: V2| {
            let a = p.x / 100.0;
            v2((100.0 + p.y) * a.cos(), (100.0 + p.y) * a.sin())
        };
        let w = Warper {
            map: &map,
            tolerance: 0.05,
            step: 1e-3,
        };
        let c = Cubic::line(v2(0.0, 10.0), v2(600.0, 10.0));
        let mut out = Vec::new();
        let mut budget = WarpBudget::new(10_000);
        w.warp(&c, &[0.5], &mut out, &mut budget);
        assert!(out.len() > 4 && out.len() < 64, "{}", out.len());
        for seg in &out {
            for i in 0..=8 {
                let q = seg.eval(i as f64 / 8.0);
                assert!((q.len() - 110.0).abs() < 0.06, "{}", q.len());
            }
        }
        // Continuity.
        for pair in out.windows(2) {
            assert!(pair[0].0[3].dist(pair[1].0[0]) < 1e-9);
        }
    }

    #[test]
    fn budget_bounds_output() {
        let map = |p: V2| v2(p.x, (p.x * 50.0).sin() * 30.0 + p.y);
        let w = Warper {
            map: &map,
            tolerance: 1e-6,
            step: 1e-4,
        };
        let c = Cubic::line(v2(0.0, 0.0), v2(100.0, 0.0));
        let mut out = Vec::new();
        let mut budget = WarpBudget::new(50);
        w.warp(&c, &[], &mut out, &mut budget);
        assert!(out.len() <= 50);
        assert!(
            budget.exhausted,
            "an unreachable tolerance runs the budget out"
        );
        let mut budget = WarpBudget::new(10_000);
        w.warp(
            &Cubic::line(v2(0.0, 0.0), v2(0.001, 0.0)),
            &[],
            &mut out,
            &mut budget,
        );
        assert!(!budget.exhausted);
    }

    #[test]
    fn crossings_find_all_roots() {
        let c = Cubic([v2(0.0, 0.0), v2(10.0, 0.0), v2(-5.0, 0.0), v2(5.0, 0.0)]);
        let r = crossings(&c, |p| p.x, &[1.0, 2.0], 16);
        for t in &r {
            let x = c.eval(*t).x;
            assert!((x - 1.0).abs() < 1e-9 || (x - 2.0).abs() < 1e-9);
        }
        assert_eq!(r.len(), 4, "{r:?}");
    }
}
