//! Point and cubic Bézier arithmetic for the vector network.
//!
//! Every segment of a network is handled as a cubic: a straight segment is the
//! cubic whose inner control points sit at 1/3 and 2/3 of the chord, so it is
//! linearly parametrised and flat, and one code path serves both.

use std::ops::{Add, Mul, Neg, Sub};

use serde::{Deserialize, Serialize};

/// A 2D point or vector. Serialises as `{ "x", "y" }`, which is also Figma's
/// plugin-API `Vector` (the shape of `tangentStart` / `tangentEnd`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Vector {
    pub x: f64,
    pub y: f64,
}

impl Vector {
    pub const ZERO: Vector = Vector { x: 0.0, y: 0.0 };

    pub const fn new(x: f64, y: f64) -> Vector {
        Vector { x, y }
    }
    pub fn dot(self, o: Vector) -> f64 {
        self.x * o.x + self.y * o.y
    }
    /// z of the 3D cross product; positive when `o` is counter-clockwise of
    /// `self` in a y-up frame.
    pub fn cross(self, o: Vector) -> f64 {
        self.x * o.y - self.y * o.x
    }
    pub fn len(self) -> f64 {
        self.x.hypot(self.y)
    }
    pub fn dist(self, o: Vector) -> f64 {
        (self - o).len()
    }
    pub fn lerp(self, o: Vector, t: f64) -> Vector {
        Vector::new(self.x + (o.x - self.x) * t, self.y + (o.y - self.y) * t)
    }
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
    pub fn is_zero(self) -> bool {
        self.x == 0.0 && self.y == 0.0
    }
    pub fn angle(self) -> f64 {
        self.y.atan2(self.x)
    }
}

impl Add for Vector {
    type Output = Vector;
    fn add(self, o: Vector) -> Vector {
        Vector::new(self.x + o.x, self.y + o.y)
    }
}
impl Sub for Vector {
    type Output = Vector;
    fn sub(self, o: Vector) -> Vector {
        Vector::new(self.x - o.x, self.y - o.y)
    }
}
impl Mul<f64> for Vector {
    type Output = Vector;
    fn mul(self, s: f64) -> Vector {
        Vector::new(self.x * s, self.y * s)
    }
}
impl Neg for Vector {
    type Output = Vector;
    fn neg(self) -> Vector {
        Vector::new(-self.x, -self.y)
    }
}

/// Axis-aligned box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub min: Vector,
    pub max: Vector,
}

impl Rect {
    pub fn from_points(points: &[Vector]) -> Rect {
        let mut r = Rect {
            min: Vector::new(f64::INFINITY, f64::INFINITY),
            max: Vector::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
        };
        for p in points {
            r.include(*p);
        }
        r
    }
    pub fn include(&mut self, p: Vector) {
        self.min.x = self.min.x.min(p.x);
        self.min.y = self.min.y.min(p.y);
        self.max.x = self.max.x.max(p.x);
        self.max.y = self.max.y.max(p.y);
    }
    pub fn union(self, o: Rect) -> Rect {
        let mut r = self;
        r.include(o.min);
        r.include(o.max);
        r
    }
    pub fn overlaps(&self, o: &Rect, pad: f64) -> bool {
        self.min.x <= o.max.x + pad
            && o.min.x <= self.max.x + pad
            && self.min.y <= o.max.y + pad
            && o.min.y <= self.max.y + pad
    }
    pub fn contains(&self, p: Vector, pad: f64) -> bool {
        p.x >= self.min.x - pad
            && p.x <= self.max.x + pad
            && p.y >= self.min.y - pad
            && p.y <= self.max.y + pad
    }
    pub fn diag(&self) -> f64 {
        (self.max - self.min).len()
    }
    pub fn center(&self) -> Vector {
        self.min.lerp(self.max, 0.5)
    }
}

/// A cubic Bézier `p0 → p3` with control points `p1`, `p2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cubic {
    pub p0: Vector,
    pub p1: Vector,
    pub p2: Vector,
    pub p3: Vector,
}

impl Cubic {
    pub fn new(p0: Vector, p1: Vector, p2: Vector, p3: Vector) -> Cubic {
        Cubic { p0, p1, p2, p3 }
    }

    /// The straight segment `a → b`, linearly parametrised.
    pub fn line(a: Vector, b: Vector) -> Cubic {
        Cubic::new(a, a.lerp(b, 1.0 / 3.0), a.lerp(b, 2.0 / 3.0), b)
    }

    pub fn eval(&self, t: f64) -> Vector {
        let mt = 1.0 - t;
        let a = mt * mt * mt;
        let b = 3.0 * mt * mt * t;
        let c = 3.0 * mt * t * t;
        let d = t * t * t;
        Vector::new(
            a * self.p0.x + b * self.p1.x + c * self.p2.x + d * self.p3.x,
            a * self.p0.y + b * self.p1.y + c * self.p2.y + d * self.p3.y,
        )
    }

    pub fn deriv(&self, t: f64) -> Vector {
        let mt = 1.0 - t;
        let d0 = self.p1 - self.p0;
        let d1 = self.p2 - self.p1;
        let d2 = self.p3 - self.p2;
        (d0 * (mt * mt) + d1 * (2.0 * mt * t) + d2 * (t * t)) * 3.0
    }

    pub fn second_deriv(&self, t: f64) -> Vector {
        let a = self.p2 - self.p1 * 2.0 + self.p0;
        let b = self.p3 - self.p2 * 2.0 + self.p1;
        (a * (1.0 - t) + b * t) * 6.0
    }

    /// de Casteljau split at `t`.
    pub fn split(&self, t: f64) -> (Cubic, Cubic) {
        let p01 = self.p0.lerp(self.p1, t);
        let p12 = self.p1.lerp(self.p2, t);
        let p23 = self.p2.lerp(self.p3, t);
        let p012 = p01.lerp(p12, t);
        let p123 = p12.lerp(p23, t);
        let m = p012.lerp(p123, t);
        (
            Cubic::new(self.p0, p01, p012, m),
            Cubic::new(m, p123, p23, self.p3),
        )
    }

    /// The piece of the curve over `[t0, t1]` (`t0 < t1`).
    pub fn subsegment(&self, t0: f64, t1: f64) -> Cubic {
        if t0 <= 0.0 {
            if t1 >= 1.0 {
                return *self;
            }
            return self.split(t1).0;
        }
        let (_, right) = self.split(t0);
        if t1 >= 1.0 {
            return right;
        }
        let span = 1.0 - t0;
        if span <= 0.0 {
            return Cubic::new(self.p3, self.p3, self.p3, self.p3);
        }
        right.split(((t1 - t0) / span).clamp(0.0, 1.0)).0
    }

    pub fn reversed(&self) -> Cubic {
        Cubic::new(self.p3, self.p2, self.p1, self.p0)
    }

    /// Bounds of the control polygon (contains the curve).
    pub fn hull_bounds(&self) -> Rect {
        Rect::from_points(&[self.p0, self.p1, self.p2, self.p3])
    }

    /// `∫ (x·y' − y·x') / 2 dt` over `[0, 1]`: this curve's term of the
    /// shoelace (Green's theorem) area of a closed chain. The integrand is a
    /// degree-5 polynomial, so 3-point Gauss–Legendre is exact.
    pub fn signed_area(&self) -> f64 {
        const W: [f64; 3] = [5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0];
        let h = 0.5 * (0.6f64).sqrt();
        let ts = [0.5 - h, 0.5, 0.5 + h];
        let mut acc = 0.0;
        for (w, t) in W.iter().zip(ts) {
            let p = self.eval(t);
            let d = self.deriv(t);
            acc += w * (p.x * d.y - p.y * d.x);
        }
        acc * 0.5
    }

    /// Largest distance of an inner control point from the chord.
    pub fn flatness(&self) -> f64 {
        let chord = self.p3 - self.p0;
        let len = chord.len();
        if len < 1e-300 {
            return self.p1.dist(self.p0).max(self.p2.dist(self.p0));
        }
        let d1 = (self.p1 - self.p0).cross(chord).abs() / len;
        let d2 = (self.p2 - self.p0).cross(chord).abs() / len;
        // Control points beyond the chord's ends also count (a curve that
        // doubles back along its chord is not flat).
        let beyond = |p: Vector| {
            let s = (p - self.p0).dot(chord) / (len * len);
            if s < 0.0 {
                -s * len
            } else if s > 1.0 {
                (s - 1.0) * len
            } else {
                0.0
            }
        };
        d1.max(d2).max(beyond(self.p1)).max(beyond(self.p2))
    }

    /// Append a polyline approximation (within about `tol`) of the curve,
    /// excluding `p0`, to `out`.
    pub fn flatten_into(&self, tol: f64, out: &mut Vec<Vector>) {
        let dd1 = self.p2 - self.p1 * 2.0 + self.p0;
        let dd2 = self.p3 - self.p2 * 2.0 + self.p1;
        let m = dd1.len().max(dd2.len());
        let n = if m <= 0.0 {
            1
        } else {
            ((0.75 * m / tol.max(1e-9)).sqrt().ceil() as usize).clamp(1, 512)
        };
        for i in 1..=n {
            out.push(self.eval(i as f64 / n as f64));
        }
    }

    /// Parameters in `(0, 1)` where `x'` or `y'` vanishes; between them the
    /// curve is monotone in both axes, so it cannot cross itself.
    pub fn extrema_params(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for axis in 0..2 {
            let g = |v: Vector| if axis == 0 { v.x } else { v.y };
            let (a, b, c, d) = (g(self.p0), g(self.p1), g(self.p2), g(self.p3));
            // B'(t)/3 = (b−a)(1−t)² + 2(c−b)(1−t)t + (d−c)t²
            let qa = (b - a) - 2.0 * (c - b) + (d - c);
            let qb = 2.0 * ((c - b) - (b - a));
            let qc = b - a;
            for r in quadratic_roots(qa, qb, qc) {
                if r > 1e-9 && r < 1.0 - 1e-9 {
                    out.push(r);
                }
            }
        }
        out.sort_by(f64::total_cmp);
        out.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        out
    }

    /// Direction the curve leaves `p0` in: the first non-vanishing derivative.
    pub fn start_direction(&self) -> Vector {
        for p in [self.p1, self.p2, self.p3] {
            let d = p - self.p0;
            if d.len() > 1e-12 {
                return d;
            }
        }
        Vector::new(1.0, 0.0)
    }

    /// Nearest parameter to `p` and the distance there.
    pub fn closest_param(&self, p: Vector) -> (f64, f64) {
        const N: usize = 32;
        let mut best_t = 0.0;
        let mut best_d = f64::INFINITY;
        for i in 0..=N {
            let t = i as f64 / N as f64;
            let d = self.eval(t).dist(p);
            if d < best_d {
                best_d = d;
                best_t = t;
            }
        }
        // Newton on f(t) = (B(t) − p)·B'(t).
        let mut t = best_t;
        for _ in 0..12 {
            let b = self.eval(t) - p;
            let d1 = self.deriv(t);
            let d2 = self.second_deriv(t);
            let f = b.dot(d1);
            let fp = d1.dot(d1) + b.dot(d2);
            if fp.abs() < 1e-300 {
                break;
            }
            let next = (t - f / fp).clamp(0.0, 1.0);
            let d = self.eval(next).dist(p);
            if d < best_d {
                best_d = d;
                best_t = next;
            }
            if (next - t).abs() < 1e-14 {
                break;
            }
            t = next;
        }
        (best_t, best_d)
    }

    /// The first point, walking from `p0`, at distance `r` from `p0` (or `p3`
    /// when the curve never gets that far).
    pub fn point_at_radius(&self, r: f64) -> Vector {
        const N: usize = 64;
        let mut prev = 0.0;
        for i in 1..=N {
            let t = i as f64 / N as f64;
            if self.eval(t).dist(self.p0) >= r {
                let (mut lo, mut hi) = (prev, t);
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    if self.eval(mid).dist(self.p0) >= r {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                return self.eval(hi);
            }
            prev = t;
        }
        self.p3
    }

    pub fn is_finite(&self) -> bool {
        self.p0.is_finite() && self.p1.is_finite() && self.p2.is_finite() && self.p3.is_finite()
    }
}

/// Real roots of `a·t² + b·t + c` (a linear or constant equation when `a` ≈ 0).
pub fn quadratic_roots(a: f64, b: f64, c: f64) -> Vec<f64> {
    let scale = a.abs().max(b.abs()).max(c.abs());
    if scale == 0.0 {
        return Vec::new();
    }
    if a.abs() <= 1e-12 * scale {
        if b.abs() <= 1e-12 * scale {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return Vec::new();
    }
    let sq = disc.sqrt();
    // Numerically stable form.
    let q = -0.5 * (b + b.signum() * sq);
    let mut out = Vec::new();
    if q != 0.0 {
        out.push(c / q);
        out.push(q / a);
    } else {
        out.push(0.0);
    }
    out
}

/// Intersection parameters `(s, t)` of the segments `p→p2` and `q→q2`, allowing
/// a small overshoot `slack` past either end. Parallel segments return `None`.
pub fn segment_intersection(
    p: Vector,
    p2: Vector,
    q: Vector,
    q2: Vector,
    slack: f64,
) -> Option<(f64, f64)> {
    let r = p2 - p;
    let s = q2 - q;
    let denom = r.cross(s);
    if denom.abs() <= 1e-12 * r.len() * s.len() || denom == 0.0 {
        return None;
    }
    let qp = q - p;
    let t = qp.cross(s) / denom;
    let u = qp.cross(r) / denom;
    if t < -slack || t > 1.0 + slack || u < -slack || u > 1.0 + slack {
        return None;
    }
    Some((t, u))
}

/// Winding number of `p` with respect to the closed polygon `poly`.
pub fn winding_number(poly: &[Vector], p: Vector) -> i32 {
    let n = poly.len();
    if n < 2 {
        return 0;
    }
    let mut w = 0;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        if a.y <= p.y {
            if b.y > p.y && (b - a).cross(p - a) > 0.0 {
                w += 1;
            }
        } else if b.y <= p.y && (b - a).cross(p - a) < 0.0 {
            w -= 1;
        }
    }
    w
}
