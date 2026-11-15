//! Port of `utils/markup/elements/charts/curves.ts` — the PathRecorder plus
//! the d3-shape curve generators it wires up. Each curve is an
//! operation-for-operation port of `node_modules/d3-shape/src/curve/*.js`
//! (linear, monotoneX for 'smooth', stepBefore/stepAfter, basis, cardinal
//! tension 0, catmullRom alpha 0.5) so emitted control points are
//! bit-identical to the production output.
//!
//! The recorder's `closePath` is a no-op (curves.ts:78-81); `_line` is
//! `undefined` in the TS (generateCurvePoints never calls areaStart), which
//! makes the first point a `move` — modeled here as `line: None`.

use crate::value::{json, Value};

/// PathPoint sum type (only the variants chart curves emit).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathPt {
    Move {
        x: f64,
        y: f64,
    },
    Line {
        x: f64,
        y: f64,
    },
    Curve {
        x: f64,
        y: f64,
        cp1: (f64, f64),
        cp2: (f64, f64),
    },
}

impl PathPt {
    pub fn x(&self) -> f64 {
        match *self {
            PathPt::Move { x, .. } | PathPt::Line { x, .. } | PathPt::Curve { x, .. } => x,
        }
    }
    pub fn y(&self) -> f64 {
        match *self {
            PathPt::Move { y, .. } | PathPt::Line { y, .. } | PathPt::Curve { y, .. } => y,
        }
    }
    pub fn to_value(&self) -> Value {
        match *self {
            PathPt::Move { x, y } => json!({"type": "move", "x": x, "y": y}),
            PathPt::Line { x, y } => json!({"type": "line", "x": x, "y": y}),
            PathPt::Curve { x, y, cp1, cp2 } => json!({
                "type": "curve", "x": x, "y": y,
                "cp1": {"x": cp1.0, "y": cp1.1},
                "cp2": {"x": cp2.0, "y": cp2.1},
            }),
        }
    }
}

pub fn path_pts_to_value(pts: &[PathPt]) -> Value {
    Value::Array(pts.iter().map(PathPt::to_value).collect())
}

struct Recorder {
    points: Vec<PathPt>,
}

impl Recorder {
    fn move_to(&mut self, x: f64, y: f64) {
        self.points.push(PathPt::Move { x, y });
    }
    fn line_to(&mut self, x: f64, y: f64) {
        self.points.push(PathPt::Line { x, y });
    }
    fn bezier_curve_to(&mut self, cp1x: f64, cp1y: f64, cp2x: f64, cp2y: f64, x: f64, y: f64) {
        self.points.push(PathPt::Curve {
            x,
            y,
            cp1: (cp1x, cp1y),
            cp2: (cp2x, cp2y),
        });
    }
    // closePath: intentional no-op (curves.ts:78-81)
}

/// `generateCurvePoints(points, curve)` (curves.ts:103-117).
pub fn generate_curve_points(points: &[(f64, f64)], curve: &str) -> Vec<PathPt> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut rec = Recorder { points: Vec::new() };
    match curve {
        "smooth" => monotone_x(&mut rec, points),
        "step" | "step-after" => step(&mut rec, points, 1.0),
        "step-before" => step(&mut rec, points, 0.0),
        "basis" => basis(&mut rec, points),
        "cardinal" => cardinal(&mut rec, points, 0.0),
        "catmull-rom" => catmull_rom(&mut rec, points, 0.5),
        _ => linear(&mut rec, points), // 'linear' + unknown → curveLinear
    }
    rec.points
}

// ---- linear.js ------------------------------------------------------------

fn linear(rec: &mut Recorder, pts: &[(f64, f64)]) {
    let mut point = 0u8;
    for &(x, y) in pts {
        match point {
            0 => {
                point = 1;
                rec.move_to(x, y);
            }
            1 => {
                point = 2;
                rec.line_to(x, y);
            }
            _ => rec.line_to(x, y),
        }
    }
    // lineEnd: closePath no-op
}

// ---- step.js (t = 1 stepAfter, t = 0 stepBefore) ---------------------------

fn step(rec: &mut Recorder, pts: &[(f64, f64)], t: f64) {
    let mut point = 0u8;
    let (mut px, mut py) = (f64::NAN, f64::NAN);
    for &(x, y) in pts {
        match point {
            0 => {
                point = 1;
                rec.move_to(x, y);
            }
            _ => {
                if point == 1 {
                    point = 2;
                }
                if t <= 0.0 {
                    rec.line_to(px, y);
                    rec.line_to(x, y);
                } else {
                    let x1 = px * (1.0 - t) + x * t;
                    rec.line_to(x1, py);
                    rec.line_to(x1, y);
                }
            }
        }
        px = x;
        py = y;
    }
    // lineEnd: 0 < t < 1 never holds for our two presets; closePath no-op
    let _ = py;
}

// ---- basis.js ---------------------------------------------------------------

fn basis_point(rec: &mut Recorder, x0: f64, y0: f64, x1: f64, y1: f64, x: f64, y: f64) {
    rec.bezier_curve_to(
        (2.0 * x0 + x1) / 3.0,
        (2.0 * y0 + y1) / 3.0,
        (x0 + 2.0 * x1) / 3.0,
        (y0 + 2.0 * y1) / 3.0,
        (x0 + 4.0 * x1 + x) / 6.0,
        (y0 + 4.0 * y1 + y) / 6.0,
    );
}

fn basis(rec: &mut Recorder, pts: &[(f64, f64)]) {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    let mut point = 0u8;
    for &(x, y) in pts {
        match point {
            0 => {
                point = 1;
                rec.move_to(x, y);
            }
            1 => point = 2,
            2 => {
                point = 3;
                rec.line_to((5.0 * x0 + x1) / 6.0, (5.0 * y0 + y1) / 6.0);
                basis_point(rec, x0, y0, x1, y1, x, y);
            }
            _ => basis_point(rec, x0, y0, x1, y1, x, y),
        }
        x0 = x1;
        x1 = x;
        y0 = y1;
        y1 = y;
    }
    // lineEnd
    match point {
        3 => {
            basis_point(rec, x0, y0, x1, y1, x1, y1);
            rec.line_to(x1, y1);
        }
        2 => rec.line_to(x1, y1),
        _ => {}
    }
}

// ---- cardinal.js (k = (1 - tension) / 6) ------------------------------------

struct CardinalState {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    k: f64,
}

fn cardinal_point(rec: &mut Recorder, s: &CardinalState, x: f64, y: f64) {
    rec.bezier_curve_to(
        s.x1 + s.k * (s.x2 - s.x0),
        s.y1 + s.k * (s.y2 - s.y0),
        s.x2 + s.k * (s.x1 - x),
        s.y2 + s.k * (s.y1 - y),
        s.x2,
        s.y2,
    );
}

fn cardinal(rec: &mut Recorder, pts: &[(f64, f64)], tension: f64) {
    let mut s = CardinalState {
        x0: f64::NAN,
        y0: f64::NAN,
        x1: f64::NAN,
        y1: f64::NAN,
        x2: f64::NAN,
        y2: f64::NAN,
        k: (1.0 - tension) / 6.0,
    };
    let mut point = 0u8;
    for &(x, y) in pts {
        match point {
            0 => {
                point = 1;
                rec.move_to(x, y);
            }
            1 => {
                point = 2;
                s.x1 = x;
                s.y1 = y;
            }
            2 => {
                point = 3;
                cardinal_point(rec, &s, x, y);
            }
            _ => cardinal_point(rec, &s, x, y),
        }
        s.x0 = s.x1;
        s.x1 = s.x2;
        s.x2 = x;
        s.y0 = s.y1;
        s.y1 = s.y2;
        s.y2 = y;
    }
    // lineEnd
    match point {
        2 => rec.line_to(s.x2, s.y2),
        3 => cardinal_point(rec, &s, s.x1, s.y1),
        _ => {}
    }
}

// ---- catmullRom.js (alpha = 0.5) ---------------------------------------------

const EPSILON: f64 = 1e-12; // d3-shape/src/math.js

struct CatmullState {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    l01_a: f64,
    l12_a: f64,
    l23_a: f64,
    l01_2a: f64,
    l12_2a: f64,
    l23_2a: f64,
    alpha: f64,
}

fn catmull_point(rec: &mut Recorder, s: &CatmullState, x: f64, y: f64) {
    let mut x1 = s.x1;
    let mut y1 = s.y1;
    let mut x2 = s.x2;
    let mut y2 = s.y2;

    if s.l01_a > EPSILON {
        let a = 2.0 * s.l01_2a + 3.0 * s.l01_a * s.l12_a + s.l12_2a;
        let n = 3.0 * s.l01_a * (s.l01_a + s.l12_a);
        x1 = (x1 * a - s.x0 * s.l12_2a + s.x2 * s.l01_2a) / n;
        y1 = (y1 * a - s.y0 * s.l12_2a + s.y2 * s.l01_2a) / n;
    }
    if s.l23_a > EPSILON {
        let b = 2.0 * s.l23_2a + 3.0 * s.l23_a * s.l12_a + s.l12_2a;
        let m = 3.0 * s.l23_a * (s.l23_a + s.l12_a);
        x2 = (x2 * b + s.x1 * s.l23_2a - x * s.l12_2a) / m;
        y2 = (y2 * b + s.y1 * s.l23_2a - y * s.l12_2a) / m;
    }
    rec.bezier_curve_to(x1, y1, x2, y2, s.x2, s.y2);
}

fn catmull_rom(rec: &mut Recorder, pts: &[(f64, f64)], alpha: f64) {
    let mut s = CatmullState {
        x0: f64::NAN,
        y0: f64::NAN,
        x1: f64::NAN,
        y1: f64::NAN,
        x2: f64::NAN,
        y2: f64::NAN,
        l01_a: 0.0,
        l12_a: 0.0,
        l23_a: 0.0,
        l01_2a: 0.0,
        l12_2a: 0.0,
        l23_2a: 0.0,
        alpha,
    };
    let mut point = 0u8;
    for &(x, y) in pts {
        if point != 0 {
            let x23 = s.x2 - x;
            let y23 = s.y2 - y;
            s.l23_2a = (x23 * x23 + y23 * y23).powf(s.alpha);
            s.l23_a = s.l23_2a.sqrt();
        }
        match point {
            0 => {
                point = 1;
                rec.move_to(x, y);
            }
            1 => point = 2,
            2 => {
                point = 3;
                catmull_point(rec, &s, x, y);
            }
            _ => catmull_point(rec, &s, x, y),
        }
        s.l01_a = s.l12_a;
        s.l12_a = s.l23_a;
        s.l01_2a = s.l12_2a;
        s.l12_2a = s.l23_2a;
        s.x0 = s.x1;
        s.x1 = s.x2;
        s.x2 = x;
        s.y0 = s.y1;
        s.y1 = s.y2;
        s.y2 = y;
    }
    // lineEnd
    match point {
        2 => rec.line_to(s.x2, s.y2),
        3 => {
            // `this.point(this._x2, this._y2)` — re-enters the full point()
            // method including the l23 recompute (x23 = y23 = 0 → l23 = 0).
            let (x, y) = (s.x2, s.y2);
            let x23 = s.x2 - x;
            let y23 = s.y2 - y;
            s.l23_2a = (x23 * x23 + y23 * y23).powf(s.alpha);
            s.l23_a = s.l23_2a.sqrt();
            catmull_point(rec, &s, x, y);
        }
        _ => {}
    }
}

// ---- monotone.js (MonotoneX) -------------------------------------------------

fn js_sign(x: f64) -> f64 {
    if x < 0.0 {
        -1.0
    } else {
        1.0
    }
}

struct MonotoneState {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    t0: f64,
}

/// `slope3(that, x2, y2)` — note the JS `(h0 || h1 < 0 && -0)` quirk: when
/// `h0 === 0` the divisor becomes `-0` if `h1 < 0` (→ ±∞ slope) else `+0`.
fn slope3(s: &MonotoneState, x2: f64, y2: f64) -> f64 {
    let h0 = s.x1 - s.x0;
    let h1 = x2 - s.x1;
    let d0 = if h0 != 0.0 {
        h0
    } else if h1 < 0.0 {
        -0.0
    } else {
        0.0
    };
    let d1 = if h1 != 0.0 {
        h1
    } else if h0 < 0.0 {
        -0.0
    } else {
        0.0
    };
    let s0 = (s.y1 - s.y0) / d0;
    let s1 = (y2 - s.y1) / d1;
    let p = (s0 * h1 + s1 * h0) / (h0 + h1);
    let v = (js_sign(s0) + js_sign(s1)) * f64::min(f64::min(s0.abs(), s1.abs()), 0.5 * p.abs());
    // `|| 0` — NaN (and ±0) coerce to 0
    if v.is_nan() {
        0.0
    } else {
        v
    }
}

fn slope2(s: &MonotoneState, t: f64) -> f64 {
    let h = s.x1 - s.x0;
    if h != 0.0 {
        (3.0 * (s.y1 - s.y0) / h - t) / 2.0
    } else {
        t
    }
}

fn monotone_point(rec: &mut Recorder, s: &MonotoneState, t0: f64, t1: f64) {
    let dx = (s.x1 - s.x0) / 3.0;
    rec.bezier_curve_to(
        s.x0 + dx,
        s.y0 + dx * t0,
        s.x1 - dx,
        s.y1 - dx * t1,
        s.x1,
        s.y1,
    );
}

fn monotone_x(rec: &mut Recorder, pts: &[(f64, f64)]) {
    let mut s = MonotoneState {
        x0: f64::NAN,
        y0: f64::NAN,
        x1: f64::NAN,
        y1: f64::NAN,
        t0: f64::NAN,
    };
    let mut point = 0u8;
    for &(x, y) in pts {
        let mut t1 = f64::NAN;
        if x == s.x1 && y == s.y1 {
            continue; // ignore coincident points
        }
        match point {
            0 => {
                point = 1;
                rec.move_to(x, y);
            }
            1 => point = 2,
            2 => {
                point = 3;
                t1 = slope3(&s, x, y);
                monotone_point(rec, &s, slope2(&s, t1), t1);
            }
            _ => {
                t1 = slope3(&s, x, y);
                monotone_point(rec, &s, s.t0, t1);
            }
        }
        s.x0 = s.x1;
        s.x1 = x;
        s.y0 = s.y1;
        s.y1 = y;
        s.t0 = t1;
    }
    // lineEnd
    match point {
        2 => rec.line_to(s.x1, s.y1),
        3 => {
            let t = slope2(&s, s.t0);
            monotone_point(rec, &s, s.t0, t);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_curve() {
        let pts = generate_curve_points(&[(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)], "linear");
        assert_eq!(
            pts,
            vec![
                PathPt::Move { x: 0.0, y: 0.0 },
                PathPt::Line { x: 10.0, y: 5.0 },
                PathPt::Line { x: 20.0, y: 0.0 },
            ]
        );
    }

    #[test]
    fn monotone_matches_d3() {
        // node dump (exact f64s): M0,0
        //   C3.3333333333333335,2.5,6.666666666666666,5,10,5
        //   C13.333333333333334,5,16.666666666666668,2.5,20,0
        let pts = generate_curve_points(&[(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)], "smooth");
        assert_eq!(
            pts,
            vec![
                PathPt::Move { x: 0.0, y: 0.0 },
                PathPt::Curve {
                    x: 10.0,
                    y: 5.0,
                    cp1: (3.3333333333333335, 2.5),
                    cp2: (6.666666666666666, 5.0),
                },
                PathPt::Curve {
                    x: 20.0,
                    y: 0.0,
                    cp1: (13.333333333333334, 5.0),
                    cp2: (16.666666666666668, 2.5),
                },
            ]
        );
    }

    #[test]
    fn catmull_rom_matches_d3() {
        // node dump (exact f64s) for alpha 0.5 over 4 points
        let pts = generate_curve_points(
            &[(0.0, 0.0), (10.0, 5.0), (20.0, 0.0), (30.0, 4.0)],
            "catmull-rom",
        );
        assert_eq!(pts.len(), 4);
        match &pts[2] {
            PathPt::Curve { cp1, cp2, .. } => {
                assert_eq!(cp1.0, 13.333333333333334);
                assert_eq!(cp1.1, 5.0);
                assert_eq!(cp2.0, 16.63494568573216);
                assert_eq!(cp2.1, 0.1399680168151673);
            }
            other => panic!("expected curve, got {other:?}"),
        }
        match &pts[3] {
            PathPt::Curve { cp1, .. } => {
                assert_eq!(cp1.0, 23.30277563796894);
                assert_eq!(cp1.1, -0.13737756150677305);
            }
            other => panic!("expected curve, got {other:?}"),
        }
    }

    #[test]
    fn cardinal_matches_d3() {
        let pts = generate_curve_points(
            &[(0.0, 0.0), (10.0, 5.0), (20.0, 0.0), (30.0, 4.0)],
            "cardinal",
        );
        match &pts[2] {
            PathPt::Curve { cp1, cp2, .. } => {
                assert_eq!(cp1.0, 13.333333333333332);
                assert_eq!(cp2.1, 0.16666666666666666);
            }
            other => panic!("expected curve, got {other:?}"),
        }
    }

    #[test]
    fn step_after_matches_d3() {
        // d3.curveStepAfter: M0,0 L10,0 L10,5 L20,5 L20,0
        let pts = generate_curve_points(&[(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)], "step");
        assert_eq!(
            pts,
            vec![
                PathPt::Move { x: 0.0, y: 0.0 },
                PathPt::Line { x: 10.0, y: 0.0 },
                PathPt::Line { x: 10.0, y: 5.0 },
                PathPt::Line { x: 20.0, y: 5.0 },
                PathPt::Line { x: 20.0, y: 0.0 },
            ]
        );
    }

    #[test]
    fn basis_matches_d3() {
        // node dump (exact f64s): M0,0 L1.6666666666666667,0.8333333333333334
        //   C…,10,3.3333333333333335 C…,18.333333333333332,0.8333333333333334 L20,0
        let pts = generate_curve_points(&[(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)], "basis");
        assert_eq!(pts.len(), 5);
        assert_eq!(
            pts[1],
            PathPt::Line {
                x: 1.6666666666666667,
                y: 0.8333333333333334
            }
        );
        assert_eq!(pts[2].x(), 10.0);
        assert_eq!(pts[2].y(), 3.3333333333333335);
        assert_eq!(pts[3].x(), 18.333333333333332);
        assert_eq!(pts[4], PathPt::Line { x: 20.0, y: 0.0 });
    }

    #[test]
    fn single_point() {
        let pts = generate_curve_points(&[(3.0, 4.0)], "smooth");
        assert_eq!(pts, vec![PathPt::Move { x: 3.0, y: 4.0 }]);
    }
}
