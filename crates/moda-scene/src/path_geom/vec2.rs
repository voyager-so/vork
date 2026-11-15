//! Tiny 2D vector + cubic helpers shared by curve fitting (`fit.rs`) and the
//! stroke outliner (`stroke_outline/`). Plain `f64` math, no allocation.

use std::ops::{Add, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct V2 {
    pub x: f64,
    pub y: f64,
}

pub const fn v2(x: f64, y: f64) -> V2 {
    V2 { x, y }
}

impl V2 {
    pub fn dot(self, o: V2) -> f64 {
        self.x * o.x + self.y * o.y
    }
    /// z of the 3D cross product.
    pub fn cross(self, o: V2) -> f64 {
        self.x * o.y - self.y * o.x
    }
    pub fn len(self) -> f64 {
        self.x.hypot(self.y)
    }
    pub fn len2(self) -> f64 {
        self.x * self.x + self.y * self.y
    }
    pub fn dist(self, o: V2) -> f64 {
        (self - o).len()
    }
    /// Unit vector, or `None` for a (near-)zero vector.
    pub fn unit(self) -> Option<V2> {
        let l = self.len();
        (l > 1e-12 && l.is_finite()).then(|| v2(self.x / l, self.y / l))
    }
    pub fn lerp(self, o: V2, t: f64) -> V2 {
        v2(self.x + (o.x - self.x) * t, self.y + (o.y - self.y) * t)
    }
    /// Left of the direction of travel on a y-down canvas: heading +x, left is -y.
    pub fn left_normal(self) -> V2 {
        v2(self.y, -self.x)
    }
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

impl Add for V2 {
    type Output = V2;
    fn add(self, o: V2) -> V2 {
        v2(self.x + o.x, self.y + o.y)
    }
}
impl Sub for V2 {
    type Output = V2;
    fn sub(self, o: V2) -> V2 {
        v2(self.x - o.x, self.y - o.y)
    }
}
impl Mul<f64> for V2 {
    type Output = V2;
    fn mul(self, k: f64) -> V2 {
        v2(self.x * k, self.y * k)
    }
}
impl Neg for V2 {
    type Output = V2;
    fn neg(self) -> V2 {
        v2(-self.x, -self.y)
    }
}

/// A cubic Bézier `[p0, c1, c2, p3]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cubic(pub [V2; 4]);

impl Cubic {
    /// Conservative finite-chord test, shared by conversion and flattening.
    pub fn is_chord(&self, distance_epsilon: f64, parameter_epsilon: f64) -> bool {
        let [p0, p1, p2, p3] = self.0;
        let d = p3 - p0;
        let length = d.len();
        if length <= 1e-12 {
            return p1.dist(p0) <= distance_epsilon && p2.dist(p0) <= distance_epsilon;
        }
        [p1, p2].into_iter().all(|p| {
            let v = p - p0;
            let t = v.dot(d) / (length * length);
            v.cross(d).abs() / length <= distance_epsilon
                && (-parameter_epsilon..=1.0 + parameter_epsilon).contains(&t)
        })
    }

    /// Interior derivative roots of a linear coordinate projection. Partitioning
    /// here makes each interval monotone, including collinear reversals.
    pub fn extrema(&self, coord: impl Fn(V2) -> f64) -> Vec<f64> {
        let [p0, p1, p2, p3] = self.0.map(coord);
        let (d0, d1, d2) = (p1 - p0, p2 - p1, p3 - p2);
        let (mut a, mut b, mut c) = (d0 - 2.0 * d1 + d2, 2.0 * (d1 - d0), d0);
        let scale = a.abs().max(b.abs()).max(c.abs());
        if scale == 0.0 || !scale.is_finite() {
            return Vec::new();
        }
        a /= scale;
        b /= scale;
        c /= scale;
        let mut roots = Vec::new();
        if a == 0.0 {
            if b != 0.0 {
                roots.push(-c / b);
            }
        } else {
            let disc = b * b - 4.0 * a * c;
            if disc >= 0.0 {
                let q = -0.5 * (b + disc.sqrt().copysign(b));
                if q == 0.0 {
                    roots.push(-b / (2.0 * a));
                } else {
                    roots.extend([q / a, c / q]);
                }
            }
        }
        roots.retain(|t| t.is_finite() && *t > 0.0 && *t < 1.0);
        roots.sort_by(f64::total_cmp);
        roots.dedup();
        roots
    }

    pub fn line(a: V2, b: V2) -> Cubic {
        Cubic([a, a.lerp(b, 1.0 / 3.0), a.lerp(b, 2.0 / 3.0), b])
    }

    pub fn eval(&self, t: f64) -> V2 {
        let [p0, p1, p2, p3] = self.0;
        let mt = 1.0 - t;
        p0 * (mt * mt * mt) + p1 * (3.0 * mt * mt * t) + p2 * (3.0 * mt * t * t) + p3 * (t * t * t)
    }

    /// First derivative.
    pub fn d1(&self, t: f64) -> V2 {
        let [p0, p1, p2, p3] = self.0;
        let mt = 1.0 - t;
        (p1 - p0) * (3.0 * mt * mt) + (p2 - p1) * (6.0 * mt * t) + (p3 - p2) * (3.0 * t * t)
    }

    /// Second derivative.
    pub fn d2(&self, t: f64) -> V2 {
        let [p0, p1, p2, p3] = self.0;
        (p2 - p1 * 2.0 + p0) * (6.0 * (1.0 - t)) + (p3 - p2 * 2.0 + p1) * (6.0 * t)
    }

    /// Arc length over `[0, t]`, 8-point Gauss–Legendre (plenty for the
    /// sub-pixel bookkeeping it is used for; not an exact integral at cusps).
    pub fn arc_len_to(&self, t: f64) -> f64 {
        const X: [f64; 4] = [
            0.183_434_642_495_649_8,
            0.525_532_409_916_329,
            0.796_666_477_413_626_7,
            0.960_289_856_497_536_3,
        ];
        const W: [f64; 4] = [
            0.362_683_783_378_362,
            0.313_706_645_877_887_3,
            0.222_381_034_453_374_5,
            0.101_228_536_290_376_3,
        ];
        let half = t * 0.5;
        let mut sum = 0.0;
        for i in 0..4 {
            sum += W[i] * (self.d1(half * (1.0 + X[i])).len() + self.d1(half * (1.0 - X[i])).len());
        }
        sum * half
    }

    pub fn arc_len(&self) -> f64 {
        self.arc_len_to(1.0)
    }

    /// Unit tangent at `t`, falling back through the control polygon when the
    /// derivative vanishes (coincident handles, cusps).
    pub fn tangent(&self, t: f64) -> Option<V2> {
        if let Some(u) = self.d1(t).unit() {
            return Some(u);
        }
        let [p0, p1, p2, p3] = self.0;
        if t < 0.5 {
            (p2 - p0).unit().or((p3 - p0).unit())
        } else {
            (p3 - p1).unit().or((p3 - p0).unit())
        }
    }
}
