//! A blend's spine: the path the steps are placed along, as an arc-length-parametrised polyline.
//!
//! Without a replaced spine a blend's spine is the polyline through its objects' centres (Illustrator's straight
//! default spine). A replaced spine (Object ▸ Blend ▸ Replace Spine) is any path; it is sampled finely enough that
//! steps placed by arc length sit on the curve to well under a pixel.

use super::geometry::{Pt, Shape};

/// Samples per curved spine segment.
const SPINE_SAMPLES_PER_CURVE: usize = 64;
/// A polyline turn below this (radians, about 12°) is a sample of a smooth curve, not a corner.
const SMOOTH_TURN: f64 = 0.2;

#[derive(Debug, Clone)]
pub struct Spine {
    pts: Vec<Pt>,
    /// Cumulative arc length at each point (`cum[0] == 0`).
    cum: Vec<f64>,
}

impl Spine {
    /// A spine through `pts`, or `None` when it has no length.
    pub fn from_polyline(pts: &[Pt]) -> Option<Spine> {
        let mut out: Vec<Pt> = Vec::with_capacity(pts.len());
        for &p in pts {
            if !(p.0.is_finite() && p.1.is_finite()) {
                continue;
            }
            if out
                .last()
                .is_some_and(|q: &Pt| (q.0 - p.0).hypot(q.1 - p.1) < 1e-9)
            {
                continue;
            }
            out.push(p);
        }
        if out.len() < 2 {
            return None;
        }
        let mut cum = Vec::with_capacity(out.len());
        cum.push(0.0);
        for i in 1..out.len() {
            let d = (out[i].0 - out[i - 1].0).hypot(out[i].1 - out[i - 1].1);
            cum.push(cum[i - 1] + d);
        }
        Some(Spine { pts: out, cum })
    }

    /// A spine along a path's first contour.
    pub fn from_shape(shape: &Shape) -> Option<Spine> {
        Spine::from_polyline(&shape.polyline(SPINE_SAMPLES_PER_CURVE))
    }

    pub fn length(&self) -> f64 {
        *self.cum.last().unwrap_or(&0.0)
    }

    /// The segment index containing arc length `s` (`after`: at a vertex, the segment starting there).
    fn segment(&self, s: f64, after: bool) -> usize {
        let last = self.pts.len() - 2;
        // First index whose cumulative length exceeds (or reaches, when !after) s.
        let idx = if after {
            self.cum.partition_point(|&c| c <= s)
        } else {
            self.cum.partition_point(|&c| c < s)
        };
        idx.saturating_sub(1).min(last)
    }

    /// The point at arc length `s` (clamped to the spine).
    pub fn point_at(&self, s: f64) -> Pt {
        let s = s.clamp(0.0, self.length());
        let i = self.segment(s, true);
        let (a, b) = (self.pts[i], self.pts[i + 1]);
        let span = self.cum[i + 1] - self.cum[i];
        let t = if span > 0.0 {
            (s - self.cum[i]) / span
        } else {
            0.0
        };
        (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
    }

    fn segment_dir(&self, i: usize) -> f64 {
        let (a, b) = (self.pts[i], self.pts[i + 1]);
        (b.1 - a.1).atan2(b.0 - a.0)
    }

    /// The direction at vertex `i` seen from segment `seg` (`i` is one of its ends). A vertex where the polyline turns
    /// by less than [`SMOOTH_TURN`] is a sample of a smooth curve and takes the mean of its two segments' directions,
    /// so directions along a sampled curve are exact at the samples; a sharper turn is a corner, and each side keeps
    /// its own segment's direction.
    fn vertex_dir(&self, i: usize, seg: usize) -> f64 {
        let own = self.segment_dir(seg);
        let other = if i == seg {
            // The segment's start: the one before it.
            if i == 0 {
                return own;
            }
            self.segment_dir(i - 1)
        } else {
            if i + 1 >= self.pts.len() {
                return own;
            }
            self.segment_dir(i)
        };
        let other = unwrap_angle(own, other);
        if (other - own).abs() < SMOOTH_TURN {
            (own + other) / 2.0
        } else {
            own
        }
    }

    /// Direction of travel at arc length `s`, radians: interpolated along each segment between its ends' directions
    /// (see [`Spine::vertex_dir`]). `after` picks the outgoing segment at a vertex.
    pub fn tangent_at(&self, s: f64, after: bool) -> f64 {
        let s = s.clamp(0.0, self.length());
        let i = self.segment(s, after);
        let span = self.cum[i + 1] - self.cum[i];
        let f = if span > 0.0 {
            ((s - self.cum[i]) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let start = self.vertex_dir(i, i);
        let end = unwrap_angle(start, self.vertex_dir(i + 1, i));
        start + (end - start) * f
    }

    /// Arc length of the point on the spine nearest `p`.
    pub fn project(&self, p: Pt) -> f64 {
        let mut best = (f64::INFINITY, 0.0);
        for i in 0..self.pts.len() - 1 {
            let (a, b) = (self.pts[i], self.pts[i + 1]);
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            let len2 = dx * dx + dy * dy;
            let t = if len2 > 0.0 {
                (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let q = (a.0 + dx * t, a.1 + dy * t);
            let d = (q.0 - p.0).hypot(q.1 - p.1);
            if d < best.0 - 1e-9 {
                best = (d, self.cum[i] + (self.cum[i + 1] - self.cum[i]) * t);
            }
        }
        best.1
    }
}

/// `b` moved by whole turns to within half a turn of `a` (so lerping angles takes the short way).
pub fn unwrap_angle(a: f64, b: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    let mut d = (b - a) % TAU;
    if d > PI {
        d -= TAU;
    } else if d < -PI {
        d += TAU;
    }
    a + d
}
