//! The stroke's centerline as an arc-length–parameterised "spine": position,
//! unit tangent and signed curvature at any distance `s` along one subpath.
//!
//! Built by flattening each cubic adaptively (flatness AND turning angle, so
//! tight fillets get enough samples) and keeping the TRUE tangent and
//! curvature from the curve's derivatives at every sample; evaluation lerps
//! between samples (tangents renormalised). Beyond either end the spine is
//! extended along the end tangent, so finite differences near the ends stay
//! meaningful. `knots` are the distances of the segment joins, where the
//! spine's curvature may jump: warping splits artwork there.

use super::geom::{flatness, norm, split, Subpath};
use crate::path_geom::vec2::{v2, Cubic, V2};

/// Most samples one spine may hold.
pub const MAX_SPINE_SAMPLES: usize = 200_000;
const MAX_DEPTH: u32 = 16;
/// Largest tangent turn between two samples (radians).
const MAX_TURN: f64 = 0.12;

#[derive(Debug, Clone)]
pub struct Spine {
    s: Vec<f64>,
    p: Vec<V2>,
    t: Vec<V2>,
    k: Vec<f64>,
    pub length: f64,
    pub knots: Vec<f64>,
    /// Distances where the tangent jumps (corners left unrounded).
    pub kinks: Vec<f64>,
    pub closed: bool,
    /// Flattening hit `MAX_SPINE_SAMPLES`: the spine is incomplete.
    pub truncated: bool,
}

/// One evaluation.
#[derive(Debug, Clone, Copy)]
pub struct SpinePoint {
    pub pos: V2,
    /// Unit tangent (direction of travel).
    pub tan: V2,
    /// Signed curvature: positive turns right on the y-down screen (towards
    /// the right normal).
    pub curvature: f64,
}

trait IntoUnit {
    fn into_unit(self) -> Option<V2>;
}

impl IntoUnit for V2 {
    /// `V2::unit` without `hypot`.
    fn into_unit(self) -> Option<V2> {
        let l = norm(self);
        (l > 1e-12 && l.is_finite()).then(|| v2(self.x / l, self.y / l))
    }
}

impl SpinePoint {
    /// Right of the direction of travel on a y-down canvas: heading +x, right is +y.
    pub fn right(&self) -> V2 {
        v2(-self.tan.y, self.tan.x)
    }
}

fn curvature(c: &Cubic, t: f64) -> f64 {
    let d1 = c.d1(t);
    let l = norm(d1);
    if l < 1e-9 {
        return 0.0;
    }
    let k = d1.cross(c.d2(t)) / (l * l * l);
    if k.is_finite() {
        k
    } else {
        0.0
    }
}

struct Builder {
    s: Vec<f64>,
    p: Vec<V2>,
    t: Vec<V2>,
    k: Vec<f64>,
    last_tan: V2,
}

impl Builder {
    fn push(&mut self, pos: V2, tan: V2, k: f64) {
        let s = match self.p.last() {
            Some(prev) => self.s.last().copied().unwrap_or(0.0) + norm(pos - *prev),
            None => 0.0,
        };
        self.s.push(s);
        self.p.push(pos);
        self.t.push(tan);
        self.k.push(k);
        self.last_tan = tan;
    }

    fn tangent(&self, c: &Cubic, t: f64) -> V2 {
        c.tangent(t).unwrap_or(self.last_tan)
    }

    /// Emit samples of `c` over (t0, t1], recursing until flat and gently turning.
    fn flatten(&mut self, c: &Cubic, tol: f64) {
        // Explicit stack of (piece, t0, t1, depth) to stay recursion-free.
        let mut stack: Vec<(Cubic, f64, f64, u32)> = vec![(*c, 0.0, 1.0, 0)];
        while let Some((piece, t0, t1, depth)) = stack.pop() {
            if self.p.len() >= MAX_SPINE_SAMPLES {
                return;
            }
            let turn = match (piece.tangent(0.0), piece.tangent(1.0)) {
                (Some(a), Some(b)) => a.cross(b).atan2(a.dot(b)).abs(),
                _ => 0.0,
            };
            if depth >= MAX_DEPTH || (flatness(&piece) <= tol && turn <= MAX_TURN) {
                let tan = self.tangent(c, t1);
                self.push(c.eval(t1), tan, curvature(c, t1));
                continue;
            }
            let (a, b) = split(&piece, 0.5);
            let tm = (t0 + t1) / 2.0;
            // Push the right half first so the left is processed first.
            stack.push((b, tm, t1, depth + 1));
            stack.push((a, t0, tm, depth + 1));
        }
    }
}

impl Spine {
    /// Build from one subpath. `None` when it has (almost) no length.
    pub fn new(sp: &Subpath, tol: f64) -> Option<Spine> {
        let tol = tol.max(1e-4);
        let first = sp.segs.first()?;
        let mut b = Builder {
            s: Vec::new(),
            p: Vec::new(),
            t: Vec::new(),
            k: Vec::new(),
            last_tan: first.tangent(0.0).unwrap_or(v2(1.0, 0.0)),
        };
        let start_tan = sp
            .segs
            .iter()
            .find_map(|c| {
                c.tangent(0.0)
                    .filter(|_| c.0[0].dist(c.0[3]) > 1e-12 || flatness(c) > 1e-12)
            })
            .unwrap_or(v2(1.0, 0.0));
        b.last_tan = start_tan;
        b.push(first.0[0], start_tan, curvature(first, 0.0));
        let mut knots = Vec::new();
        let mut kinks = Vec::new();
        for c in &sp.segs {
            let degenerate = c.0.iter().all(|q| q.dist(c.0[0]) < 1e-12);
            if degenerate {
                continue;
            }
            // Restart the segment with its own start tangent (a corner keeps
            // two samples at the same s: incoming, then outgoing).
            let tan0 = b.tangent(c, 0.0);
            if b.last_tan.dot(tan0) < 1.0 - 1e-12 {
                if b.last_tan.dot(tan0) < (0.01f64).cos() {
                    kinks.push(b.s.last().copied().unwrap_or(0.0));
                }
                b.push(c.0[0], tan0, curvature(c, 0.0));
            }
            b.flatten(c, tol);
            knots.push(b.s.last().copied().unwrap_or(0.0));
        }
        knots.pop();
        let truncated = b.p.len() >= MAX_SPINE_SAMPLES;
        let length = b.s.last().copied().unwrap_or(0.0);
        if !length.is_finite() || length <= 1e-9 {
            return None;
        }
        Some(Spine {
            s: b.s,
            p: b.p,
            t: b.t,
            k: b.k,
            length,
            knots,
            kinks,
            closed: sp.closed,
            truncated,
        })
    }

    /// Illustrator's placement polyline (scatter brushes): each cubic cut into
    /// [`chord_count`] chords of equal parameter step, walked by chord length,
    /// with each chord's own direction as the tangent (so copies on one chord
    /// share a rotation, and sit on the chord rather than the curve). Measured
    /// against Illustrator 30.8 on 13 cubics (docs/qa/illustrator-parity.md,
    /// "Scatter placement"). Curvature is 0 everywhere; corners and knots as in
    /// [`Spine::new`].
    pub fn chords(sp: &Subpath) -> Option<Spine> {
        if sp.segs.is_empty() {
            return None;
        }
        let mut b = Builder {
            s: Vec::new(),
            p: Vec::new(),
            t: Vec::new(),
            k: Vec::new(),
            last_tan: v2(1.0, 0.0),
        };
        let mut knots = Vec::new();
        let mut kinks = Vec::new();
        let mut started = false;
        for c in &sp.segs {
            let n = chord_count(c);
            let mut prev = c.eval(0.0);
            for i in 1..=n {
                if b.p.len() + 2 > MAX_SPINE_SAMPLES {
                    break;
                }
                let q = if i == n { c.0[3] } else { c.eval(i as f64 / n as f64) };
                let Some(tan) = (q - prev).into_unit() else {
                    continue;
                };
                if !started {
                    b.push(prev, tan, 0.0);
                    started = true;
                } else if b.last_tan.dot(tan) < 1.0 - 1e-12 {
                    if i == 1 && b.last_tan.dot(tan) < (0.01f64).cos() {
                        kinks.push(b.s.last().copied().unwrap_or(0.0));
                    }
                    // A new chord direction: a second sample at the same s.
                    b.push(prev, tan, 0.0);
                }
                b.push(q, tan, 0.0);
                prev = q;
            }
            if started {
                knots.push(b.s.last().copied().unwrap_or(0.0));
            }
        }
        knots.pop();
        let truncated = b.p.len() + 2 > MAX_SPINE_SAMPLES;
        let length = b.s.last().copied().unwrap_or(0.0);
        if !started || !length.is_finite() || length <= 1e-9 {
            return None;
        }
        Some(Spine {
            s: b.s,
            p: b.p,
            t: b.t,
            k: b.k,
            length,
            knots,
            kinks,
            closed: sp.closed,
            truncated,
        })
    }

    pub fn sample_count(&self) -> usize {
        self.p.len()
    }

    /// Position, tangent and curvature at distance `s` (extended linearly
    /// beyond the ends).
    pub fn eval(&self, s: f64) -> SpinePoint {
        let n = self.s.len();
        if s <= 0.0 || n == 1 {
            let tan = self.t[0];
            return SpinePoint {
                pos: self.p[0] + tan * s.min(0.0),
                tan,
                curvature: if s < 0.0 { 0.0 } else { self.k[0] },
            };
        }
        if s >= self.length {
            let tan = self.t[n - 1];
            return SpinePoint {
                pos: self.p[n - 1] + tan * (s - self.length),
                tan,
                curvature: if s > self.length { 0.0 } else { self.k[n - 1] },
            };
        }
        // First sample with s_i > s; the bracket is [i-1, i].
        let i = self.s.partition_point(|x| *x <= s).clamp(1, n - 1);
        let (s0, s1) = (self.s[i - 1], self.s[i]);
        let f = if s1 > s0 { (s - s0) / (s1 - s0) } else { 1.0 };
        let tan = (self.t[i - 1] * (1.0 - f) + self.t[i] * f)
            .into_unit()
            .unwrap_or(self.t[i]);
        SpinePoint {
            pos: self.p[i - 1].lerp(self.p[i], f),
            tan,
            curvature: self.k[i - 1] + (self.k[i] - self.k[i - 1]) * f,
        }
    }
}

/// How many equal-parameter chords Illustrator cuts a cubic into when it
/// places scatter copies: `1 + floor(0.64 · √d)`, where `d` is the largest
/// coordinate of the control polygon's second differences (`P0 − 2P1 + P2`,
/// `P1 − 2P2 + P3`, max-norm, in document units). A straight segment is one
/// chord. Fitted to 13 cubics in Illustrator 30.8 (all exact; the factor lies
/// in 0.636–0.645), capped for safety.
pub fn chord_count(c: &Cubic) -> usize {
    let [p0, p1, p2, p3] = c.0;
    let d1 = p0 - p1 * 2.0 + p2;
    let d2 = p1 - p2 * 2.0 + p3;
    let d = d1.x.abs().max(d1.y.abs()).max(d2.x.abs()).max(d2.y.abs());
    if !d.is_finite() {
        return 1;
    }
    (1.0 + (0.64 * d.sqrt()).floor()).clamp(1.0, 4096.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blend::BlendPathPoint;
    use crate::stroke_brush::geom::subpaths_from_points;

    fn circle(r: f64) -> Subpath {
        let k = r * 0.5522847498;
        let c = |x: f64, y: f64, a: (f64, f64), b: (f64, f64)| BlendPathPoint::Curve {
            x,
            y,
            cp1: a,
            cp2: b,
        };
        let pts = vec![
            BlendPathPoint::Move { x: r, y: 0.0 },
            c(0.0, r, (r, k), (k, r)),
            c(-r, 0.0, (-k, r), (-r, k)),
            c(0.0, -r, (-r, -k), (-k, -r)),
            c(r, 0.0, (k, -r), (r, -k)),
        ];
        subpaths_from_points(&pts, true).remove(0)
    }

    #[test]
    fn circle_length_tangent_curvature() {
        let sp = Spine::new(&circle(50.0), 0.01).unwrap();
        let expect = 2.0 * std::f64::consts::PI * 50.0;
        assert!((sp.length - expect).abs() < 0.05, "{}", sp.length);
        assert_eq!(sp.knots.len(), 3);
        assert!(sp.kinks.is_empty());
        for i in 0..20 {
            let q = sp.eval(sp.length * i as f64 / 20.0);
            assert!((q.pos.len() - 50.0).abs() < 0.02);
            assert!(q.tan.dot(q.pos).abs() < 0.5, "tangent ⟂ radius");
            // Counter-clockwise on a y-down screen (x → y) turns right: +1/r.
            assert!((q.curvature - 0.02).abs() < 0.0005, "{}", q.curvature);
        }
    }

    #[test]
    fn corners_keep_both_tangents_and_extend() {
        let pts = vec![
            BlendPathPoint::Move { x: 0.0, y: 0.0 },
            BlendPathPoint::Line { x: 10.0, y: 0.0 },
            BlendPathPoint::Line { x: 10.0, y: 10.0 },
        ];
        let sp = Spine::new(&subpaths_from_points(&pts, false)[0], 0.01).unwrap();
        assert_eq!(sp.length, 20.0);
        assert_eq!(sp.knots, vec![10.0]);
        assert_eq!(sp.kinks, vec![10.0]);
        assert!(sp.eval(5.0).tan.dist(v2(1.0, 0.0)) < 1e-12);
        assert!(sp.eval(15.0).tan.dist(v2(0.0, 1.0)) < 1e-12);
        assert!(sp.eval(-2.0).pos.dist(v2(-2.0, 0.0)) < 1e-12);
        assert!(sp.eval(23.0).pos.dist(v2(10.0, 13.0)) < 1e-12);
    }

    #[test]
    fn degenerate_spine_is_none() {
        let pts = vec![
            BlendPathPoint::Move { x: 1.0, y: 1.0 },
            BlendPathPoint::Line { x: 1.0, y: 1.0 },
        ];
        assert!(Spine::new(&subpaths_from_points(&pts, false)[0], 0.1).is_none());
    }

    fn cubic(p: [(f64, f64); 4]) -> Cubic {
        Cubic([
            v2(p[0].0, p[0].1),
            v2(p[1].0, p[1].1),
            v2(p[2].0, p[2].1),
            v2(p[3].0, p[3].1),
        ])
    }

    #[test]
    fn chord_counts_match_illustrator() {
        // (control points, Illustrator's chord count read from copy rotations).
        let k = 0.5523;
        let arc = |r: f64| {
            [
                (30.0, 30.0),
                (30.0 + k * r, 30.0),
                (30.0 + r, 30.0 + r - k * r),
                (30.0 + r, 30.0 + r),
            ]
        };
        let cases: Vec<([(f64, f64); 4], usize)> = vec![
            ([(30.0, 90.0), (80.0, -10.0), (160.0, 190.0), (210.0, 90.0)], 12),
            ([(30.0, 40.0), (260.0, 40.0), (260.0, 120.0), (30.0, 120.0)], 10),
            ([(20.0, 180.0), (40.0, 20.0), (120.0, 20.0), (280.0, 150.0)], 9),
            ([(20.0, 40.0), (100.0, 30.0), (200.0, 50.0), (300.0, 40.0)], 4),
            ([(20.0, 40.0), (100.0, 20.0), (200.0, 60.0), (300.0, 40.0)], 5),
            (arc(25.0), 3),
            (arc(50.0), 4),
            (arc(100.0), 5),
            (arc(200.0), 7),
        ];
        for (p, n) in cases {
            assert_eq!(chord_count(&cubic(p)), n, "{p:?}");
        }
        let line = cubic([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]);
        assert_eq!(chord_count(&line), 1);
    }

    #[test]
    fn chord_spine_walks_chords_with_their_directions() {
        let sp = Subpath {
            segs: vec![cubic([(30.0, 40.0), (260.0, 40.0), (260.0, 120.0), (30.0, 120.0)])],
            closed: false,
        };
        let spine = Spine::chords(&sp).unwrap();
        let c = sp.segs[0];
        // Ten chords: the first runs from B(0) to B(0.1), at 2.1° below +x.
        let q = spine.eval(1.0);
        let chord = (c.eval(0.1) - c.eval(0.0)).into_unit().unwrap();
        assert!(q.tan.dist(chord) < 1e-9);
        assert!((q.tan.y.atan2(q.tan.x).to_degrees() - 2.1).abs() < 0.05);
        // Positions lie on the chords; the length is the polyline's.
        let poly: f64 = (0..10)
            .map(|i| c.eval(i as f64 / 10.0).dist(c.eval((i + 1) as f64 / 10.0)))
            .sum();
        assert!((spine.length - poly).abs() < 1e-9);
        assert!(spine.eval(spine.length).pos.dist(c.0[3]) < 1e-9);
    }
}
