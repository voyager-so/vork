//! Corner smoothing ("squircle" corners, Figma's 0–100% `cornerSmoothing`).
//!
//! Construction from Figma's "Desperately seeking squircles" (Daniel Furse,
//! 2018): a smoothed corner of radius `R` and smoothing `ξ` ∈ [0, 1]
//!
//! - consumes `p = (1 + ξ)·R` of each adjacent edge (a plain rounded corner
//!   consumes `R`);
//! - keeps a circular arc of radius `R` in the middle whose measure is
//!   `90°·(1 − ξ)` (a quarter circle at ξ = 0, nothing at ξ = 1);
//! - joins each edge to that arc with one cubic Bézier `P1 P2 P3 P4`: `P1 P2
//!   P3` lie on the edge (zero curvature where it leaves the edge), `P3 P4`
//!   is tangent to the arc at `P4`, and the edge interval `P1..P3` is split
//!   two thirds to `a = |P1 P2|` and one third to `b = |P2 P3|`.
//!
//! With `c`, `d` the along-edge / across-edge offsets of `P4` from `P3`:
//!
//! ```text
//! arcMeasure = 90° (1 − ξ)
//! L          = sin(arcMeasure / 2) · R · √2      (chord leg of the arc)
//! α          = (90° − arcMeasure) / 2
//! |P3 P4|    = R · tan(α / 2)
//! β          = 45° · ξ
//! c          = |P3 P4| · cos β,   d = c · tan β
//! b          = (p − L − c − d) / 3,   a = 2b
//! ```
//!
//! Room: when `p` does not fit the corner's budget (the edge length the corner
//! may use, see [`corner_budgets`]), the smoothing is reduced first (the
//! article: fit the largest rounding there is room for), so `p` never exceeds
//! the budget and a pill or circle stays a pill or circle.
//!
//! Per-corner radii: radii are first normalized the way every rounded rect is
//! (`normalizeRoundedRectRadii`, CSS-style proportional scaling), then each
//! corner's budget is shared out along its two edges largest radius first,
//! proportionally to the neighbouring radius (a zero-radius neighbour leaves
//! the whole edge). At ξ = 0 the path is exactly the plain rounded rect.
//!
//! Everything is in node-local space (origin top-left, y down); the path runs
//! clockwise on screen from the top edge, as `rectangleToPathPoints` does.

/// A path segment in node-local space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Seg {
    Move(f64, f64),
    Line(f64, f64),
    Cubic {
        cp1: (f64, f64),
        cp2: (f64, f64),
        to: (f64, f64),
    },
}

/// Distances of one smoothed corner (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CornerParams {
    pub radius: f64,
    /// Smoothing actually applied (≤ the requested one when room is short).
    pub smoothing: f64,
    pub p: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    /// Chord leg `L`: the arc spans `(L, L)` along/across the edge.
    pub arc_section_length: f64,
    /// Arc measure in degrees, `90·(1 − smoothing)`.
    pub arc_measure_deg: f64,
}

/// Clamp a stored `cornerSmoothing` value to [0, 1] (NaN → 0).
pub fn clamp_smoothing(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Figma's per-corner construction for radius `radius`, smoothing `smoothing`
/// and a budget of `budget` along each adjacent edge.
pub fn corner_params(radius: f64, smoothing: f64, budget: f64) -> CornerParams {
    let r = radius.max(0.0);
    let mut s = clamp_smoothing(smoothing);
    let mut p = (1.0 + s) * r;
    if r > 0.0 {
        let max_s = (budget / r - 1.0).max(0.0);
        s = s.min(max_s);
        p = p.min(budget.max(r));
    }
    let arc_measure = 90.0 * (1.0 - s);
    let arc_section_length = (arc_measure / 2.0).to_radians().sin() * r * std::f64::consts::SQRT_2;
    let alpha = (90.0 - arc_measure) / 2.0;
    let p3_to_p4 = r * (alpha / 2.0).to_radians().tan();
    let beta = 45.0 * s;
    let c = p3_to_p4 * beta.to_radians().cos();
    let d = c * beta.to_radians().tan();
    let b = ((p - arc_section_length - c - d) / 3.0).max(0.0);
    let a = 2.0 * b;
    CornerParams {
        radius: r,
        smoothing: s,
        p,
        a,
        b,
        c,
        d,
        arc_section_length,
        arc_measure_deg: arc_measure,
    }
}

/// `normalizeRoundedRectRadii` (roundedRectRadii.ts:32-62) on expanded radii
/// `[tl, tr, br, bl]`: clamp each to half the short side, then scale all down
/// together so no edge's two radii overlap.
pub fn normalize_radii(width: f64, height: f64, radii: [f64; 4]) -> [f64; 4] {
    let w = width.abs();
    let h = height.abs();
    if w <= 0.0 || h <= 0.0 {
        return [0.0; 4];
    }
    let max_r = (w / 2.0).min(h / 2.0);
    let [tl, tr, br, bl] = radii.map(|r| {
        if r.is_finite() {
            r.clamp(0.0, max_r)
        } else {
            0.0
        }
    });
    let fit = |limit: f64, sum: f64| {
        if sum <= 0.0 {
            1.0
        } else {
            (limit / sum).min(1.0)
        }
    };
    let scale = fit(w, tl + tr)
        .min(fit(h, tr + br))
        .min(fit(w, br + bl))
        .min(fit(h, bl + tl));
    if scale >= 1.0 {
        [tl, tr, br, bl]
    } else {
        [tl * scale, tr * scale, br * scale, bl * scale]
    }
}

/// Per-corner edge budgets for normalized radii `[tl, tr, br, bl]`: largest
/// radius first, each corner gets, on both of its edges, either what the
/// already-served neighbour left or its radius-proportional share, and keeps
/// the smaller of the two edges. Uniform radii give `min(w, h) / 2`.
pub fn corner_budgets(width: f64, height: f64, radii: [f64; 4]) -> [f64; 4] {
    let (w, h) = (width.abs(), height.abs());
    // Neighbours of each corner and the length of the shared edge.
    // tl: top (tr, w), left (bl, h); tr: top (tl, w), right (br, h); ...
    let adj: [[(usize, f64); 2]; 4] = [
        [(1, w), (3, h)],
        [(0, w), (2, h)],
        [(1, h), (3, w)],
        [(2, w), (0, h)],
    ];
    let mut order = [0usize, 1, 2, 3];
    // Stable sort, descending radius (ties keep tl, tr, br, bl order).
    order.sort_by(|&i, &j| {
        radii[j]
            .partial_cmp(&radii[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut budget = [-1.0f64; 4];
    for &i in &order {
        let r = radii[i];
        let b = adj[i]
            .iter()
            .map(|&(n, side)| {
                let rn = radii[n];
                if r <= 0.0 && rn <= 0.0 {
                    0.0
                } else if budget[n] >= 0.0 {
                    side - budget[n]
                } else {
                    r / (r + rn) * side
                }
            })
            .fold(f64::INFINITY, f64::min);
        budget[i] = b.max(0.0);
    }
    budget
}

/// Whether `(radii, smoothing)` needs the smoothed construction at all.
pub fn is_smoothed(radii: [f64; 4], smoothing: f64) -> bool {
    clamp_smoothing(smoothing) > 0.0 && radii.iter().any(|r| *r > 0.0)
}

/// The smoothed rounded-rect outline, closed by returning to the start point
/// (no close command, like `rectangleToPathPoints`). `radii` are the raw
/// expanded `[tl, tr, br, bl]`; they are normalized here. At smoothing 0 the
/// result is the plain KAPPA rounded rect.
pub fn smoothed_rect_path(width: f64, height: f64, radii: [f64; 4], smoothing: f64) -> Vec<Seg> {
    let (w, h) = (width, height);
    let radii = normalize_radii(w, h, radii);
    let s = clamp_smoothing(smoothing);
    let budgets = corner_budgets(w, h, radii);
    // Corners clockwise from top-right: (corner point, incoming dir u, outgoing dir v).
    // Index into radii: tr=1, br=2, bl=3, tl=0.
    let corners: [(usize, (f64, f64), (f64, f64), (f64, f64)); 4] = [
        (1, (w, 0.0), (1.0, 0.0), (0.0, 1.0)),
        (2, (w, h), (0.0, 1.0), (-1.0, 0.0)),
        (3, (0.0, h), (-1.0, 0.0), (0.0, -1.0)),
        (0, (0.0, 0.0), (0.0, -1.0), (1.0, 0.0)),
    ];
    let params: Vec<CornerParams> = corners
        .iter()
        .map(|&(i, ..)| corner_params(radii[i], s, budgets[i]))
        .collect();

    let mut out = Vec::with_capacity(20);
    // Start where the top-right corner leaves the top edge.
    let (_, k0, u0, _) = corners[0];
    let start = (k0.0 - params[0].p * u0.0, k0.1 - params[0].p * u0.1);
    out.push(Seg::Move(start.0, start.1));
    let mut cur = start;
    for (ci, &(_, k, u, v)) in corners.iter().enumerate() {
        let cp = params[ci];
        let at = |along: f64, across: f64| {
            (
                k.0 - cp.p * u.0 + along * u.0 + across * v.0,
                k.1 - cp.p * u.1 + along * u.1 + across * v.1,
            )
        };
        // Edge run up to this corner's start.
        let corner_start = at(0.0, 0.0);
        if ci > 0 && dist(cur, corner_start) > 1e-9 {
            out.push(Seg::Line(corner_start.0, corner_start.1));
        }
        if cp.radius <= 0.0 {
            out.push(Seg::Line(k.0, k.1));
            cur = k;
            continue;
        }
        if cp.smoothing <= 0.0 {
            // Plain quarter circle, exactly as rectangleToPathPoints draws it.
            let kr = cp.radius * KAPPA;
            let to = (k.0 + cp.radius * v.0, k.1 + cp.radius * v.1);
            out.push(Seg::Cubic {
                cp1: at(kr, 0.0),
                cp2: at(cp.p, cp.radius - kr),
                to,
            });
            cur = to;
            continue;
        }
        let (a, b, c, d, l) = (cp.a, cp.b, cp.c, cp.d, cp.arc_section_length);
        // Edge → arc.
        let e1 = at(a + b + c, d);
        out.push(Seg::Cubic {
            cp1: at(a, 0.0),
            cp2: at(a + b, 0.0),
            to: e1,
        });
        // Arc of radius R, measure 90°(1−ξ), from e1 to e2.
        let e2 = at(a + b + c + l, d + l);
        if l > 1e-12 {
            let theta = cp.arc_measure_deg.to_radians();
            let kk = 4.0 / 3.0 * (theta / 4.0).tan() * cp.radius;
            let center = (
                k.0 - cp.radius * u.0 + cp.radius * v.0,
                k.1 - cp.radius * u.1 + cp.radius * v.1,
            );
            let t1 = arc_tangent(center, e1, (u.0 + v.0, u.1 + v.1));
            let t2 = arc_tangent(center, e2, (u.0 + v.0, u.1 + v.1));
            out.push(Seg::Cubic {
                cp1: (e1.0 + kk * t1.0, e1.1 + kk * t1.1),
                cp2: (e2.0 - kk * t2.0, e2.1 - kk * t2.1),
                to: e2,
            });
        }
        // Arc → next edge (mirror of the first cubic).
        let end = at(cp.p, cp.p);
        out.push(Seg::Cubic {
            cp1: (e2.0 + d * u.0 + c * v.0, e2.1 + d * u.1 + c * v.1),
            cp2: (
                e2.0 + d * u.0 + (b + c) * v.0,
                e2.1 + d * u.1 + (b + c) * v.1,
            ),
            to: end,
        });
        cur = end;
    }
    if dist(cur, start) > 1e-9 {
        out.push(Seg::Line(start.0, start.1));
    }
    out
}

/// `KAPPA` (`utils/path/constants.ts:9`), the plain quarter-circle handle.
const KAPPA: f64 = 0.5522847498;

fn dist(p: (f64, f64), q: (f64, f64)) -> f64 {
    ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt()
}

/// Unit tangent of the circle around `center` at `pt`, oriented along `dir`.
fn arc_tangent(center: (f64, f64), pt: (f64, f64), dir: (f64, f64)) -> (f64, f64) {
    let (rx, ry) = (pt.0 - center.0, pt.1 - center.1);
    let len = (rx * rx + ry * ry).sqrt().max(1e-12);
    let t = (-ry / len, rx / len);
    if t.0 * dir.0 + t.1 * dir.1 >= 0.0 {
        t
    } else {
        (-t.0, -t.1)
    }
}

/// Node-local path points in the scene's JSON `points` shape
/// (`{type: move|line|curve, x, y, cp1, cp2}`).
pub fn smoothed_rect_json_points(
    width: f64,
    height: f64,
    radii: [f64; 4],
    smoothing: f64,
) -> Vec<crate::value::Value> {
    use crate::value::json;
    smoothed_rect_path(width, height, radii, smoothing)
        .into_iter()
        .map(|s| match s {
            Seg::Move(x, y) => json!({"type": "move", "x": x, "y": y}),
            Seg::Line(x, y) => json!({"type": "line", "x": x, "y": y}),
            Seg::Cubic { cp1, cp2, to } => json!({
                "type": "curve", "x": to.0, "y": to.1,
                "cp1": {"x": cp1.0, "y": cp1.1},
                "cp2": {"x": cp2.0, "y": cp2.1},
            }),
        })
        .collect()
}

/// Flatten a segment list (Move + Line/Cubic) into a polygon, `steps` per cubic.
pub fn flatten(segs: &[Seg], steps: usize) -> Vec<(f64, f64)> {
    let mut pts = Vec::new();
    let mut cur = (0.0, 0.0);
    for s in segs {
        match *s {
            Seg::Move(x, y) | Seg::Line(x, y) => {
                cur = (x, y);
                pts.push(cur);
            }
            Seg::Cubic { cp1, cp2, to } => {
                for i in 1..=steps {
                    let t = i as f64 / steps as f64;
                    pts.push(cubic_at(cur, cp1, cp2, to, t));
                }
                cur = to;
            }
        }
    }
    pts
}

pub fn cubic_at(
    p0: (f64, f64),
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
    t: f64,
) -> (f64, f64) {
    let mt = 1.0 - t;
    let (a, b, c, d) = (mt * mt * mt, 3.0 * mt * mt * t, 3.0 * mt * t * t, t * t * t);
    (
        a * p0.0 + b * p1.0 + c * p2.0 + d * p3.0,
        a * p0.1 + b * p1.1 + c * p2.1 + d * p3.1,
    )
}

/// Even-odd point-in-polygon on the flattened outline (hit-testing helper).
pub fn contains_point(
    width: f64,
    height: f64,
    radii: [f64; 4],
    smoothing: f64,
    x: f64,
    y: f64,
) -> bool {
    let poly = flatten(&smoothed_rect_path(width, height, radii, smoothing), 16);
    let mut inside = false;
    let n = poly.len();
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
#[path = "corner_smoothing_tests.rs"]
mod tests;
