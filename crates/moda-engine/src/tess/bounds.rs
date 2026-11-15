//! A path's geometric bounds: the tight box of its outline (curve and arc
//! extrema, not control points), with no stroke. The box gradient percentages
//! span (`paint::gradient::gradient_box`), as in Illustrator.

use super::arc::{arc_to_beziers, ArcParams};
use super::moda_engine_path_types::PathCmd;

/// Where one coordinate of the cubic with control values `a b c d` turns,
/// t in (0, 1).
fn cubic_extrema(a: f64, b: f64, c: f64, d: f64) -> impl Iterator<Item = f64> {
    // The derivative / 3: A t² + B t + C.
    let qa = -a + 3.0 * b - 3.0 * c + d;
    let qb = 2.0 * (a - 2.0 * b + c);
    let qc = b - a;
    let mut roots = [f64::NAN; 2];
    if qa.abs() < 1e-12 {
        if qb.abs() > 1e-12 {
            roots[0] = -qc / qb;
        }
    } else {
        let disc = qb * qb - 4.0 * qa * qc;
        if disc >= 0.0 {
            let s = disc.sqrt();
            roots = [(-qb + s) / (2.0 * qa), (-qb - s) / (2.0 * qa)];
        }
    }
    roots.into_iter().filter(|t| *t > 0.0 && *t < 1.0)
}

fn cubic_at(a: f64, b: f64, c: f64, d: f64, t: f64) -> f64 {
    let u = 1.0 - t;
    u * u * u * a + 3.0 * u * u * t * b + 3.0 * u * t * t * c + t * t * t * d
}

/// `[min_x, min_y, max_x, max_y]` of the path's outline, or `None` for an
/// empty or non-finite path.
pub fn path_bounds(cmds: &[PathCmd]) -> Option<[f64; 4]> {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut add = |x: f64, y: f64| {
        if x.is_finite() && y.is_finite() {
            b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
        }
    };
    type P = (f64, f64);
    fn cubic(add: &mut impl FnMut(f64, f64), p0: P, p1: P, p2: P, p3: P) {
        add(p3.0, p3.1);
        for t in cubic_extrema(p0.0, p1.0, p2.0, p3.0).chain(cubic_extrema(p0.1, p1.1, p2.1, p3.1))
        {
            add(
                cubic_at(p0.0, p1.0, p2.0, p3.0, t),
                cubic_at(p0.1, p1.1, p2.1, p3.1, t),
            );
        }
    }
    let mut cur = (0.0, 0.0);
    for c in cmds {
        match *c {
            PathCmd::Move { x, y } | PathCmd::Line { x, y } => {
                add(x, y);
                cur = (x, y);
            }
            PathCmd::Curve { x, y, cp1, cp2 } => {
                add(cur.0, cur.1);
                cubic(&mut add, cur, cp1, cp2, (x, y));
                cur = (x, y);
            }
            PathCmd::Quadratic { x, y, cp } => {
                add(cur.0, cur.1);
                let c1 = (
                    cur.0 + (cp.0 - cur.0) * 2.0 / 3.0,
                    cur.1 + (cp.1 - cur.1) * 2.0 / 3.0,
                );
                let c2 = (x + (cp.0 - x) * 2.0 / 3.0, y + (cp.1 - y) * 2.0 / 3.0);
                cubic(&mut add, cur, c1, c2, (x, y));
                cur = (x, y);
            }
            PathCmd::Arc {
                x,
                y,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            } => {
                add(cur.0, cur.1);
                let arcs = arc_to_beziers(
                    cur.0,
                    cur.1,
                    &ArcParams {
                        rx,
                        ry,
                        x_axis_rotation,
                        large_arc,
                        sweep,
                        x,
                        y,
                    },
                );
                let mut a = cur;
                for z in &arcs {
                    let e = (z.end[0], z.end[1]);
                    cubic(&mut add, a, (z.cp1[0], z.cp1[1]), (z.cp2[0], z.cp2[1]), e);
                    a = e;
                }
                add(x, y);
                cur = (x, y);
            }
        }
    }
    (b[0].is_finite() && b[1].is_finite()).then_some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curve_bounds_are_the_extrema_not_the_handles() {
        // A quarter-circle-ish bump: handles reach y = -40, the curve only -30.
        let cmds = [
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Curve {
                x: 100.0,
                y: 0.0,
                cp1: (0.0, -40.0),
                cp2: (100.0, -40.0),
            },
        ];
        let b = path_bounds(&cmds).unwrap();
        assert_eq!([b[0], b[2], b[3]], [0.0, 100.0, 0.0]);
        assert!((b[1] + 30.0).abs() < 1e-9, "{b:?}");
    }

    #[test]
    fn arc_bounds_include_the_bulge() {
        // A half circle of radius 50 from (0, 0) to (100, 0).
        let cmds = [
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Arc {
                x: 100.0,
                y: 0.0,
                rx: 50.0,
                ry: 50.0,
                x_axis_rotation: 0.0,
                large_arc: false,
                sweep: false,
            },
        ];
        let b = path_bounds(&cmds).unwrap();
        assert!((b[0]).abs() < 1e-6 && (b[2] - 100.0).abs() < 1e-6, "{b:?}");
        assert!((b[3] - b[1] - 50.0).abs() < 1e-6, "{b:?}");
    }

    #[test]
    fn empty_path_has_no_bounds() {
        assert_eq!(path_bounds(&[]), None);
    }
}
