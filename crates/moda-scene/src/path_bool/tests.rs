//! Unit coverage for the boolean path solver.
//!
//! Every behavioural case is tiny — no operand larger than a few dozen cubics.
//! The guard cases in `every_error_code_is_reachable_through_the_envelope` are
//! deliberately not: most are sized just past a cap and cost nothing because
//! the cap refuses them before the solver runs, but the 60x60 grid that pins
//! `MAX_RESULT_POINTS` has to produce a real oversized result (14_400 points)
//! and so genuinely solves. It costs ~1.4 s of the module's ~1.8 s in a debug
//! build, which is the price of proving that cap fires on geometry rather than
//! on a number.

use crate::value::{json, Value};

use super::convert::{conversion_count, flatten_subpath, points_to_subpaths};
use super::solve::{
    check_intermediate, check_output_finite, check_pairwise_work, check_request_bytes,
    check_result_points, solve_on, Lane, MAX_INTERMEDIATE_CUBICS, MAX_OPERANDS, MAX_PAIRWISE_WORK,
    MAX_POINTS_TOTAL, MAX_REQUEST_BYTES, MAX_RESULT_POINTS, MAX_TOTAL_CUBICS,
};
use super::*;

// ---------------------------------------------------------------------------
// Shape helpers (shared with prop_tests)
// ---------------------------------------------------------------------------

pub(super) const KAPPA: f64 = 0.552_284_749_830_793_6;

/// Axis-aligned rectangle, clockwise in a y-down space.
pub(super) fn rect(x: f64, y: f64, w: f64, h: f64) -> Vec<PathPoint> {
    vec![
        PathPoint::Move { x, y },
        PathPoint::Line { x: x + w, y },
        PathPoint::Line { x: x + w, y: y + h },
        PathPoint::Line { x, y: y + h },
    ]
}

/// The same rectangle wound the other way.
pub(super) fn rect_ccw(x: f64, y: f64, w: f64, h: f64) -> Vec<PathPoint> {
    vec![
        PathPoint::Move { x, y },
        PathPoint::Line { x, y: y + h },
        PathPoint::Line { x: x + w, y: y + h },
        PathPoint::Line { x: x + w, y },
    ]
}

/// Ellipse as four kappa cubics (what `shapeToPath` emits for an ellipse node).
pub(super) fn ellipse(cx: f64, cy: f64, rx: f64, ry: f64) -> Vec<PathPoint> {
    let kx = rx * KAPPA;
    let ky = ry * KAPPA;
    vec![
        PathPoint::Move { x: cx + rx, y: cy },
        PathPoint::Curve {
            x: cx,
            y: cy + ry,
            cp1: (cx + rx, cy + ky),
            cp2: (cx + kx, cy + ry),
        },
        PathPoint::Curve {
            x: cx - rx,
            y: cy,
            cp1: (cx - kx, cy + ry),
            cp2: (cx - rx, cy + ky),
        },
        PathPoint::Curve {
            x: cx,
            y: cy - ry,
            cp1: (cx - rx, cy - ky),
            cp2: (cx - kx, cy - ry),
        },
        PathPoint::Curve {
            x: cx + rx,
            y: cy,
            cp1: (cx + kx, cy - ry),
            cp2: (cx + rx, cy - ky),
        },
    ]
}

/// Rounded rectangle using SVG arc corners — exercises the arc → cubic lane.
pub(super) fn rounded_rect(x: f64, y: f64, w: f64, h: f64, r: f64) -> Vec<PathPoint> {
    let arc = |px: f64, py: f64| PathPoint::Arc {
        x: px,
        y: py,
        rx: r,
        ry: r,
        x_axis_rotation: 0.0,
        large_arc: false,
        sweep: true,
    };
    vec![
        PathPoint::Move { x: x + r, y },
        PathPoint::Line { x: x + w - r, y },
        arc(x + w, y + r),
        PathPoint::Line {
            x: x + w,
            y: y + h - r,
        },
        arc(x + w - r, y + h),
        PathPoint::Line { x: x + r, y: y + h },
        arc(x, y + h - r),
        PathPoint::Line { x, y: y + r },
        arc(x + r, y),
    ]
}

// ---------------------------------------------------------------------------
// Measurement helpers
// ---------------------------------------------------------------------------

pub(super) fn contours(points: &[PathPoint]) -> Vec<Vec<(f64, f64)>> {
    points_to_subpaths(points, 1e-6)
        .iter()
        .map(|p| flatten_subpath(p, 24))
        .collect()
}

fn poly_area(c: &[(f64, f64)]) -> f64 {
    let mut total = 0.0;
    for i in 0..c.len() {
        let (x0, y0) = c[i];
        let (x1, y1) = c[(i + 1) % c.len()];
        total += x0 * y1 - x1 * y0;
    }
    total / 2.0
}

fn point_in_poly(c: &[(f64, f64)], p: (f64, f64)) -> bool {
    let mut inside = false;
    let mut j = c.len() - 1;
    for i in 0..c.len() {
        let (xi, yi) = c[i];
        let (xj, yj) = c[j];
        if (yi > p.1) != (yj > p.1) && p.0 < (xj - xi) * (p.1 - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// A point strictly inside a simple contour: the angle bisector at its
/// lowest-then-leftmost vertex (always convex) points into the interior, so
/// stepping a hair along it lands inside — unlike a vertex, which can sit
/// exactly on a neighbouring contour's shared edge and make containment a
/// coin flip.
fn interior_probe(c: &[(f64, f64)]) -> (f64, f64) {
    let n = c.len();
    let mut k = 0;
    for i in 1..n {
        if c[i].1 < c[k].1 || (c[i].1 == c[k].1 && c[i].0 < c[k].0) {
            k = i;
        }
    }
    let v = c[k];
    let prev = c[(k + n - 1) % n];
    let next = c[(k + 1) % n];
    let unit = |p: (f64, f64)| {
        let (dx, dy) = (p.0 - v.0, p.1 - v.1);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-12 {
            (0.0, 0.0)
        } else {
            (dx / len, dy / len)
        }
    };
    let (ax, ay) = unit(prev);
    let (bx, by) = unit(next);
    let step = 1e-4;
    (v.0 + (ax + bx) * step, v.1 + (ay + by) * step)
}

/// Even-odd filled area of a multi-subpath result: |area| of each contour,
/// signed by its nesting depth. flo_curves does not guarantee a consistent
/// hole winding, so depth is measured rather than read off the orientation.
pub(super) fn even_odd_area(points: &[PathPoint]) -> f64 {
    let cs = contours(points);
    let mut total = 0.0;
    for (i, c) in cs.iter().enumerate() {
        if c.len() < 3 {
            continue;
        }
        let probe = interior_probe(c);
        let depth = cs
            .iter()
            .enumerate()
            .filter(|(j, other)| *j != i && other.len() >= 3 && point_in_poly(other, probe))
            .count();
        let sign = if depth % 2 == 0 { 1.0 } else { -1.0 };
        total += sign * poly_area(c).abs();
    }
    total
}

pub(super) fn subpath_areas(points: &[PathPoint]) -> Vec<f64> {
    contours(points)
        .iter()
        .map(|c| poly_area(c).abs())
        .collect()
}

pub(super) fn operand(points: Vec<PathPoint>, fill_rule: FillRule) -> BooleanOperand {
    BooleanOperand { points, fill_rule }
}

pub(super) fn run(op: BooleanOp, operands: Vec<BooleanOperand>) -> BooleanOutcome {
    match boolean_paths(&BooleanRequest {
        op,
        operands,
        accuracy: None,
    }) {
        Ok(outcome) => outcome,
        // `solver_failed` is called out separately because it is not a refusal
        // at all: it is `catch_unwind` in `boolean_paths` converting a panic
        // inside flo_curves into an envelope. Every loop that reaches the solver
        // through this helper therefore fails hard on a panic, which on wasm
        // (`panic = "abort"`, no `catch_unwind`) would abort the shared engine.
        Err(e) if e.code == "solver_failed" => {
            panic!("flo_curves panicked on this input: {}", e.message)
        }
        Err(e) => panic!("unexpected solver error: {} ({})", e.code, e.message),
    }
}

pub(super) fn shape(outcome: BooleanOutcome) -> (Vec<PathPoint>, usize, Bounds) {
    match outcome {
        BooleanOutcome::Shape {
            points,
            subpath_count,
            bounds,
        } => (points, subpath_count, bounds),
        BooleanOutcome::Empty => panic!("expected a shape, got empty"),
    }
}

fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
}

// ---------------------------------------------------------------------------
// Sanity: the analytic area integral used by the degeneracy filter
// ---------------------------------------------------------------------------

#[test]
fn analytic_area_matches_known_shapes() {
    use super::convert::subpath_signed_area;
    let square = points_to_subpaths(&rect(0.0, 0.0, 2.0, 3.0), 1e-6);
    assert_eq!(square.len(), 1);
    assert_close(
        subpath_signed_area(&square[0]).abs(),
        6.0,
        1e-9,
        "square area",
    );

    let circle = points_to_subpaths(&ellipse(0.0, 0.0, 5.0, 5.0), 1e-6);
    // Four kappa cubics approximate π·25 to ~2.7e-4 relative.
    assert_close(
        subpath_signed_area(&circle[0]).abs(),
        std::f64::consts::PI * 25.0,
        0.05,
        "circle area",
    );
}

// ---------------------------------------------------------------------------
// Core ops
// ---------------------------------------------------------------------------

#[test]
fn union_of_overlapping_rects_is_one_subpath_of_lines() {
    let out = run(
        BooleanOp::Union,
        vec![
            operand(rect(0.0, 0.0, 10.0, 10.0), FillRule::NonZero),
            operand(rect(5.0, 5.0, 10.0, 10.0), FillRule::NonZero),
        ],
    );
    let (points, count, bounds) = shape(out);
    assert_eq!(count, 1);
    assert!(
        points
            .iter()
            .all(|p| matches!(p, PathPoint::Move { .. } | PathPoint::Line { .. })),
        "axis-aligned union must come back as lines, got {points:?}"
    );
    assert_close(even_odd_area(&points), 175.0, 0.05, "union area");
    assert_close(bounds.min_x, 0.0, 1e-6, "minX");
    assert_close(bounds.max_x, 15.0, 1e-6, "maxX");
    assert_close(bounds.max_y, 15.0, 1e-6, "maxY");
}

#[test]
fn subtract_of_overlapping_rects_is_an_l() {
    let out = run(
        BooleanOp::Subtract,
        vec![
            operand(rect(0.0, 0.0, 10.0, 10.0), FillRule::NonZero),
            operand(rect(5.0, 5.0, 10.0, 10.0), FillRule::NonZero),
        ],
    );
    let (points, count, bounds) = shape(out);
    assert_eq!(count, 1);
    assert_close(even_odd_area(&points), 75.0, 0.05, "L area");
    assert_close(bounds.max_x, 10.0, 1e-6, "maxX");
}

#[test]
fn intersect_of_overlapping_rects_is_the_overlap() {
    let out = run(
        BooleanOp::Intersect,
        vec![
            operand(rect(0.0, 0.0, 10.0, 10.0), FillRule::NonZero),
            operand(rect(5.0, 5.0, 10.0, 10.0), FillRule::NonZero),
        ],
    );
    let (points, count, bounds) = shape(out);
    assert_eq!(count, 1);
    assert_close(even_odd_area(&points), 25.0, 0.05, "overlap area");
    assert_close(bounds.min_x, 5.0, 1e-6, "minX");
    assert_close(bounds.max_x, 10.0, 1e-6, "maxX");
}

#[test]
fn exclude_of_overlapping_rects_is_two_subpaths() {
    let out = run(
        BooleanOp::Exclude,
        vec![
            operand(rect(0.0, 0.0, 10.0, 10.0), FillRule::NonZero),
            operand(rect(5.0, 5.0, 10.0, 10.0), FillRule::NonZero),
        ],
    );
    let (points, count, _) = shape(out);
    assert_eq!(count, 2);
    assert_close(even_odd_area(&points), 150.0, 0.05, "xor area");
}

#[test]
fn rect_minus_centred_circle_is_a_donut() {
    let r = 3.0;
    let out = run(
        BooleanOp::Subtract,
        vec![
            operand(rect(0.0, 0.0, 20.0, 20.0), FillRule::NonZero),
            operand(ellipse(10.0, 10.0, r, r), FillRule::NonZero),
        ],
    );
    let (points, count, bounds) = shape(out);
    assert_eq!(count, 2, "donut must have an outer and an inner subpath");
    let mut areas = subpath_areas(&points);
    areas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let expected_hole = std::f64::consts::PI * r * r;
    assert_close(areas[0], expected_hole, 0.2, "inner subpath area ≈ πr²");
    assert_close(areas[1], 400.0, 0.1, "outer subpath area");
    assert_close(
        even_odd_area(&points),
        400.0 - expected_hole,
        0.3,
        "donut filled area",
    );
    assert_close(bounds.min_x, 0.0, 1e-6, "minX");
    assert_close(bounds.max_y, 20.0, 1e-6, "maxY");
}

#[test]
fn disjoint_intersect_is_empty() {
    let out = run(
        BooleanOp::Intersect,
        vec![
            operand(rect(0.0, 0.0, 10.0, 10.0), FillRule::NonZero),
            operand(rect(100.0, 100.0, 10.0, 10.0), FillRule::NonZero),
        ],
    );
    assert!(matches!(out, BooleanOutcome::Empty));
}

#[test]
fn identical_shapes_subtract_and_exclude_to_empty() {
    for op in [BooleanOp::Subtract, BooleanOp::Exclude] {
        let out = run(
            op,
            vec![
                operand(rect(2.0, 2.0, 8.0, 8.0), FillRule::NonZero),
                operand(rect(2.0, 2.0, 8.0, 8.0), FillRule::NonZero),
            ],
        );
        assert!(
            matches!(out, BooleanOutcome::Empty),
            "{op:?} of identical shapes"
        );
    }
}

#[test]
fn nested_subtract_makes_a_ring_and_intersect_keeps_the_inner() {
    let outer = rect(0.0, 0.0, 20.0, 20.0);
    let inner = rect(5.0, 5.0, 10.0, 10.0);
    let (points, count, _) = shape(run(
        BooleanOp::Subtract,
        vec![
            operand(outer.clone(), FillRule::NonZero),
            operand(inner.clone(), FillRule::NonZero),
        ],
    ));
    assert_eq!(count, 2);
    assert_close(even_odd_area(&points), 300.0, 0.05, "ring area");

    let (points, _, _) = shape(run(
        BooleanOp::Intersect,
        vec![
            operand(outer, FillRule::NonZero),
            operand(inner, FillRule::NonZero),
        ],
    ));
    assert_close(even_odd_area(&points), 100.0, 0.05, "nested intersect");
}

#[test]
fn nonzero_same_wound_overlapping_subpaths_merge() {
    // One operand, two overlapping same-wound subpaths: non-zero fills the
    // union, so the overlap must not punch a hole.
    let mut points = rect(0.0, 0.0, 10.0, 10.0);
    points.extend(rect(5.0, 5.0, 10.0, 10.0));
    let (result, count, _) = shape(run(
        BooleanOp::Union,
        vec![
            operand(points, FillRule::NonZero),
            operand(rect(100.0, 100.0, 1.0, 1.0), FillRule::NonZero),
        ],
    ));
    assert_eq!(count, 2, "the far-away square stays its own subpath");
    assert_close(even_odd_area(&result), 175.0 + 1.0, 0.05, "merged area");
}

#[test]
fn evenodd_nested_subpaths_stay_a_donut() {
    let mut points = rect(0.0, 0.0, 20.0, 20.0);
    points.extend(rect(5.0, 5.0, 10.0, 10.0));
    let (result, count, _) = shape(run(
        BooleanOp::Union,
        vec![
            operand(points, FillRule::EvenOdd),
            operand(rect(100.0, 100.0, 1.0, 1.0), FillRule::NonZero),
        ],
    ));
    assert_eq!(count, 3, "outer + hole + the far-away square");
    assert_close(
        even_odd_area(&result),
        300.0 + 1.0,
        0.05,
        "donut area preserved",
    );
}

/// Non-zero is *signed* winding, so an overlap between oppositely-wound
/// subpaths winds to zero and is a hole — which is what lyon paints for the
/// same point list under `FillRule::NonZero`. Two 10x10 squares overlapping in
/// a 5x5 corner leave 75 + 75.
#[test]
fn opposite_wound_nonzero_subpaths_cancel_in_the_overlap() {
    let mut points = rect(0.0, 0.0, 10.0, 10.0);
    points.extend(rect_ccw(5.0, 5.0, 10.0, 10.0));
    let (result, count, _) = shape(run(
        BooleanOp::Union,
        vec![
            operand(points, FillRule::NonZero),
            // A speck inside the part that survives, so the union adds nothing.
            operand(rect(0.0, 0.0, 1.0, 1.0), FillRule::NonZero),
        ],
    ));
    assert_eq!(
        count, 2,
        "each square keeps the part the other did not cancel"
    );
    assert_close(even_odd_area(&result), 150.0, 0.05, "winding != 0 area");
}

/// The same two rectangles nested rather than overlapping: the classic hole
/// encoding, and the one Figma paste and SVG import both produce.
#[test]
fn opposite_wound_nonzero_nesting_is_a_hole() {
    let mut points = rect(0.0, 0.0, 20.0, 20.0);
    points.extend(rect_ccw(5.0, 5.0, 10.0, 10.0));
    let (result, count, bounds) = shape(run(
        BooleanOp::Union,
        vec![
            operand(points, FillRule::NonZero),
            operand(rect(0.0, 0.0, 1.0, 1.0), FillRule::NonZero),
        ],
    ));
    assert_eq!(count, 2, "outer ring + the hole");
    assert_close(even_odd_area(&result), 300.0, 0.05, "400 - 100");
    assert_close(bounds.max_x, 20.0, 1e-6, "maxX");
}

/// Same winding, nested: non-zero winds 1 then 2, never 0, so it is solid.
/// This is the case that separates "signed winding" from "even-odd", and the
/// one `path_remove_interior_points` happened to get right.
#[test]
fn same_wound_nonzero_nesting_stays_solid() {
    let mut points = rect(0.0, 0.0, 20.0, 20.0);
    points.extend(rect(5.0, 5.0, 10.0, 10.0));
    let (result, count, _) = shape(run(
        BooleanOp::Union,
        vec![
            operand(points, FillRule::NonZero),
            operand(rect(0.0, 0.0, 1.0, 1.0), FillRule::NonZero),
        ],
    ));
    assert_eq!(count, 1, "no hole: the inner loop winds the same way");
    assert_close(even_odd_area(&result), 400.0, 0.05, "solid square");
}

/// Alternating triple nesting: windings 1, 0, 1 — so the innermost loop is
/// filled *again*. This is what rules out computing the region as
/// "outermost minus the ring inside it": that arithmetic returns 900 - 400 +
/// nothing, or 500, depending on how it is written, and neither is 600.
#[test]
fn alternating_triple_nesting_refills_the_innermost_loop() {
    let mut points = rect(0.0, 0.0, 30.0, 30.0);
    points.extend(rect_ccw(5.0, 5.0, 20.0, 20.0));
    points.extend(rect(10.0, 10.0, 10.0, 10.0));
    let (result, count, _) = shape(run(
        BooleanOp::Union,
        vec![
            operand(points, FillRule::NonZero),
            operand(rect(0.0, 0.0, 1.0, 1.0), FillRule::NonZero),
        ],
    ));
    assert_eq!(count, 3, "outer ring, hole, and the refilled centre");
    assert_close(
        even_odd_area(&result),
        900.0 - 400.0 + 100.0,
        0.05,
        "winding 1 / 0 / 1",
    );
}

/// Even-odd is parity and ignores direction entirely, so the same three loops
/// alternate regardless of how they are wound.
#[test]
fn evenodd_ignores_winding_direction() {
    let mut wound = rect(0.0, 0.0, 30.0, 30.0);
    wound.extend(rect_ccw(5.0, 5.0, 20.0, 20.0));
    wound.extend(rect(10.0, 10.0, 10.0, 10.0));
    let mut same = rect(0.0, 0.0, 30.0, 30.0);
    same.extend(rect(5.0, 5.0, 20.0, 20.0));
    same.extend(rect(10.0, 10.0, 10.0, 10.0));
    for points in [wound, same] {
        let (result, count, _) = shape(run(
            BooleanOp::Union,
            vec![
                operand(points, FillRule::EvenOdd),
                operand(rect(0.0, 0.0, 1.0, 1.0), FillRule::NonZero),
            ],
        ));
        assert_eq!(count, 3, "even-odd alternates either way");
        assert_close(
            even_odd_area(&result),
            900.0 - 400.0 + 100.0,
            0.05,
            "even-odd parity",
        );
    }
}

#[test]
fn three_ary_subtract_equals_a_minus_union_of_the_rest() {
    let a = rect(0.0, 0.0, 30.0, 10.0);
    let b = rect(5.0, 0.0, 5.0, 10.0);
    let c = rect(20.0, 0.0, 5.0, 10.0);
    let (points, count, _) = shape(run(
        BooleanOp::Subtract,
        vec![
            operand(a, FillRule::NonZero),
            operand(b, FillRule::NonZero),
            operand(c, FillRule::NonZero),
        ],
    ));
    assert_eq!(count, 3, "two bites split the bar into three pieces");
    assert_close(
        even_odd_area(&points),
        300.0 - 50.0 - 50.0,
        0.05,
        "3-ary subtract",
    );
}

#[test]
fn rounded_rect_arcs_are_converted_and_solved() {
    let (points, count, bounds) = shape(run(
        BooleanOp::Union,
        vec![
            operand(rounded_rect(0.0, 0.0, 20.0, 20.0, 4.0), FillRule::NonZero),
            operand(rect(10.0, 10.0, 20.0, 20.0), FillRule::NonZero),
        ],
    ));
    assert_eq!(count, 1);
    // Corner slack of a radius-4 round: r²(4 − π) over the four corners. The
    // 10×10 overlap loses one quarter of that slack (the rounded corner at
    // (20,20) sits inside it), so it is not a full 100.
    let corner_slack = 16.0 * (4.0 - std::f64::consts::PI);
    let rounded_area = 400.0 - corner_slack;
    let overlap = 100.0 - corner_slack / 4.0;
    assert_close(
        even_odd_area(&points),
        rounded_area + 400.0 - overlap,
        0.5,
        "rounded ∪ rect",
    );
    assert_close(bounds.max_x, 30.0, 1e-6, "maxX");
}

#[test]
fn bounds_survive_a_large_translation() {
    let offset = 1.0e5;
    let (_, _, bounds) = shape(run(
        BooleanOp::Union,
        vec![
            operand(rect(offset, offset, 10.0, 10.0), FillRule::NonZero),
            operand(
                rect(offset + 5.0, offset + 5.0, 10.0, 10.0),
                FillRule::NonZero,
            ),
        ],
    ));
    assert_close(bounds.min_x, offset, 1e-6, "minX after translation");
    assert_close(bounds.max_x, offset + 15.0, 1e-6, "maxX after translation");
    assert_close(bounds.min_y, offset, 1e-6, "minY after translation");
    assert_close(bounds.max_y, offset + 15.0, 1e-6, "maxY after translation");
}

#[test]
fn quadratic_segments_are_degree_elevated() {
    // A quadratic "roof" over a rect: union must keep the bump.
    let tri = vec![
        PathPoint::Move { x: 0.0, y: 10.0 },
        PathPoint::Quadratic {
            x: 10.0,
            y: 10.0,
            cp: (5.0, -10.0),
        },
    ];
    let (points, _, bounds) = shape(run(
        BooleanOp::Union,
        vec![
            operand(tri, FillRule::NonZero),
            operand(rect(0.0, 10.0, 10.0, 5.0), FillRule::NonZero),
        ],
    ));
    // Quadratic apex sits at y = 10 + (−10 − 10)/2 = 0.
    assert_close(bounds.min_y, 0.0, 1e-6, "quadratic apex");
    assert!(points.iter().any(|p| matches!(p, PathPoint::Curve { .. })));
}

/// `arc_to_beziers` rebuilds its endpoint through cos/sin, so the last cubic
/// lands a few ulps away from the arc's declared end while the path cursor
/// moves there exactly. The converter must pin that tip, or the hairline gap
/// reads to flo_curves as a self-intersection.
#[test]
fn arc_expansion_is_exactly_continuous() {
    // An arc-heavy "ellipse": four quarter arcs with an odd rotation, so the
    // reconstruction error is not conveniently zero.
    let arc = |x: f64, y: f64| PathPoint::Arc {
        x,
        y,
        rx: 37.0,
        ry: 23.5,
        x_axis_rotation: 31.7,
        large_arc: false,
        sweep: true,
    };
    let points = vec![
        PathPoint::Move { x: 10.0, y: 20.0 },
        arc(47.3, 41.9),
        arc(23.1, 77.4),
        arc(-11.7, 55.2),
        arc(10.0, 20.0),
    ];
    let subpaths = points_to_subpaths(&points, 1e-6);
    assert_eq!(subpaths.len(), 1);
    let (start, segments) = &subpaths[0];
    assert!(segments.len() > 4, "arcs must expand to several cubics");

    // Every declared arc endpoint must appear verbatim among the segment ends.
    for (x, y) in [(47.3, 41.9), (23.1, 77.4), (-11.7, 55.2), (10.0, 20.0)] {
        assert!(
            segments.iter().any(|(_, _, e)| e.0 == x && e.1 == y),
            "arc endpoint ({x}, {y}) was not reproduced exactly"
        );
    }
    // The loop closes on the start anchor exactly, so no closing chord was
    // synthesized to paper over drift.
    let last_end = segments.last().expect("segments").2;
    assert_eq!((last_end.0, last_end.1), (start.0, start.1));
}

/// A single cubic whose start and end coincide still bounds area (a teardrop),
/// so the converter must keep it rather than reject it for having one segment.
#[test]
fn single_segment_teardrop_is_a_real_subpath() {
    let teardrop = vec![
        PathPoint::Move { x: 0.0, y: 0.0 },
        PathPoint::Curve {
            x: 0.0,
            y: 0.0,
            cp1: (40.0, -30.0),
            cp2: (40.0, 30.0),
        },
    ];
    let subpaths = points_to_subpaths(&teardrop, 1e-6);
    assert_eq!(subpaths.len(), 1, "a one-segment closed loop must survive");
    assert_eq!(subpaths[0].1.len(), 1);

    let (points, count, bounds) = shape(run(
        BooleanOp::Union,
        vec![
            operand(teardrop.clone(), FillRule::NonZero),
            operand(teardrop, FillRule::NonZero),
        ],
    ));
    assert_eq!(count, 1);
    assert!(
        even_odd_area(&points) > 100.0,
        "teardrop ∪ teardrop must keep positive area, got {}",
        even_odd_area(&points)
    );
    assert_close(bounds.min_x, 0.0, 1e-6, "minX");
    assert!(bounds.max_x > 20.0, "teardrop reaches out to its handles");
}

/// A single segment with no enclosed area still vanishes — the area filter, not
/// the segment count, is what rejects degenerate subpaths.
#[test]
fn zero_area_single_segment_is_still_dropped() {
    let spike = vec![
        PathPoint::Move { x: 0.0, y: 0.0 },
        PathPoint::Curve {
            x: 0.0,
            y: 0.0,
            cp1: (30.0, 0.0),
            cp2: (30.0, 0.0),
        },
    ];
    assert!(points_to_subpaths(&spike, 0.01).is_empty());
}

// ---------------------------------------------------------------------------
// JSON envelope + error codes
// ---------------------------------------------------------------------------

fn request_json(op: &str, operands: Value) -> String {
    json!({ "op": op, "operands": operands }).to_string()
}

/// `n` line points around a circle, as request JSON. One cubic per point (the
/// implicit closing segment replaces the `Move`), so the operand's cubic count
/// is exactly `n` — which is what the work bounds are stated in.
fn polygon_points_json(n: usize, radius: f64) -> Value {
    Value::Array(
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64 * std::f64::consts::TAU;
                json!({ "type": "line", "x": radius * t.cos(), "y": radius * t.sin() })
            })
            .collect(),
    )
}

/// `bars` parallel stripes as one operand, as request JSON: vertical when
/// `vertical`, horizontal otherwise. Two perpendicular sets intersect in
/// `bars²` little squares — the cheapest way to make small, legal operands
/// (4 points and 4 cubics per bar) produce an enormous result.
fn bars_json(bars: usize, vertical: bool) -> Value {
    let span = 2.0 * bars as f64;
    Value::Array(
        (0..bars)
            .map(|i| {
                let o = 2.0 * i as f64;
                let r = if vertical {
                    rect(o, 0.0, 1.0, span)
                } else {
                    rect(0.0, o, span, 1.0)
                };
                points_json(&r)
            })
            .flat_map(|v| v.as_array().expect("points").clone())
            .collect(),
    )
}

fn points_json(points: &[PathPoint]) -> Value {
    Value::Array(points.iter().map(PathPoint::to_json).collect())
}

fn solve_json(op: &str, operands: Value) -> Value {
    crate::value::from_str(&boolean_paths_json(&request_json(op, operands)))
        .expect("envelope is valid JSON")
}

#[test]
fn json_round_trip_produces_an_ok_envelope() {
    let response = solve_json(
        "union",
        json!([
            { "points": points_json(&rect(0.0, 0.0, 10.0, 10.0)), "fillRule": "nonzero" },
            { "points": points_json(&rect(5.0, 5.0, 10.0, 10.0)), "fillRule": "nonzero" },
        ]),
    );
    assert_eq!(response["status"], "ok");
    assert_eq!(response["fillRule"], "evenodd");
    assert_eq!(response["closed"], true);
    assert_eq!(response["subpathCount"], 1);
    assert_eq!(response["bounds"]["minX"], 0.0);
    assert_eq!(response["bounds"]["maxX"], 15.0);
    let points = response["points"].as_array().expect("points array");
    assert!(points.len() >= 3);
    assert_eq!(points[0]["type"], "move");
    // The envelope's points parse straight back into the internal shape.
    let parsed: Vec<PathPoint> = points
        .iter()
        .map(|p| PathPoint::from_json(p).expect("point"))
        .collect();
    assert_close(even_odd_area(&parsed), 175.0, 0.05, "round-tripped area");
}

#[test]
fn json_empty_envelope_has_no_points() {
    let response = solve_json(
        "intersect",
        json!([
            { "points": points_json(&rect(0.0, 0.0, 1.0, 1.0)), "fillRule": "nonzero" },
            { "points": points_json(&rect(50.0, 50.0, 1.0, 1.0)), "fillRule": "nonzero" },
        ]),
    );
    assert_eq!(response["status"], "empty");
    assert!(response.get("points").is_none());
}

#[test]
fn json_fill_rule_defaults_to_nonzero() {
    let response = solve_json(
        "union",
        json!([
            { "points": points_json(&rect(0.0, 0.0, 10.0, 10.0)) },
            { "points": points_json(&rect(5.0, 5.0, 10.0, 10.0)) },
        ]),
    );
    assert_eq!(response["status"], "ok");
}

#[test]
fn every_error_code_is_reachable_through_the_envelope() {
    let square = points_json(&rect(0.0, 0.0, 10.0, 10.0));

    // invalid_json
    let r: Value = crate::value::from_str(&boolean_paths_json("{not json")).unwrap();
    assert_eq!(r["code"], "invalid_json");
    assert_eq!(r["status"], "error");

    // invalid_request — bad op, bad operands, bad point, bad fill rule
    for body in [
        json!({ "op": "xor", "operands": [] }),
        json!({ "op": "union", "operands": {} }),
        json!({ "op": "union", "operands": [{ "points": [{ "type": "wat" }] }, { "points": [] }] }),
        json!({ "op": "union", "operands": [{ "points": [], "fillRule": "winding" }, { "points": [] }] }),
    ] {
        let r: Value = crate::value::from_str(&boolean_paths_json(&body.to_string())).unwrap();
        assert_eq!(r["code"], "invalid_request", "for {body}");
    }
    // operandIndex is reported when the fault is attributable to an operand.
    let attributed =
        json!({ "op": "union", "operands": [{ "points": [] }, { "points": [{ "type": "wat" }] }] })
            .to_string();
    let r: Value = crate::value::from_str(&boolean_paths_json(&attributed)).unwrap();
    assert_eq!(r["operandIndex"], 1);

    // too_few_operands
    let r = solve_json("union", json!([{ "points": square }]));
    assert_eq!(r["code"], "too_few_operands");

    // too_many_operands
    let many: Vec<Value> = (0..=MAX_OPERANDS)
        .map(|i| json!({ "points": points_json(&rect(i as f64, 0.0, 1.0, 1.0)) }))
        .collect();
    let r = solve_json("union", Value::Array(many));
    assert_eq!(r["code"], "too_many_operands");

    // too_many_points
    let long: Vec<Value> = (0..10_001)
        .map(|i| json!({ "type": "line", "x": i as f64, "y": 0.0 }))
        .collect();
    let r = solve_json(
        "union",
        json!([
            { "points": Value::Array(long) },
            { "points": points_json(&rect(0.0, 0.0, 1.0, 1.0)) },
        ]),
    );
    assert_eq!(r["code"], "too_many_points");
    assert_eq!(r["operandIndex"], 0);

    // non_finite_coordinate (out of range and NaN-shaped input)
    let r = solve_json(
        "union",
        json!([
            { "points": points_json(&rect(0.0, 0.0, 1.0, 1.0)) },
            { "points": points_json(&rect(1.0e12, 0.0, 1.0, 1.0)) },
        ]),
    );
    assert_eq!(r["code"], "non_finite_coordinate");
    assert_eq!(r["operandIndex"], 1);

    // too_complex — more cubics than the total budget, caught before any solve.
    // Four 9_000-point rings: each inside MAX_POINTS_PER_OPERAND, together
    // inside MAX_POINTS_TOTAL (40_000), and 36_000 cubics over 32_000.
    let zig = polygon_points_json(9_000, 100.0);
    let r = solve_json("union", Value::Array(vec![json!({ "points": zig }); 4]));
    assert_eq!(r["code"], "too_complex");
    assert!(
        r["message"]
            .as_str()
            .unwrap_or_default()
            .contains("36000 cubic segments"),
        "refused by the cubic budget: {r}"
    );
    assert_eq!(
        MAX_TOTAL_CUBICS, 32_000,
        "the fixture above is sized on this"
    );

    // too_complex — the pairwise work bound, on the exact lane. Both operands
    // sit well inside the exact lane's cubic allowance (2_100 cubics each,
    // 4_200 of 8_000), so that allowance cannot be what refuses them; what does
    // is the work bound at the first flo_curves entry the union lane makes,
    // per-operand normalization, which collides an operand against itself:
    // 2_100² = 4.41e6 curve pairs against a 4e6 ceiling. Before this bound
    // existed the same request was handed straight to flo_curves.
    //
    // 10.1 — the assertion has to be one a downstream refusal cannot satisfy.
    // Elapsed time is not that assertion here: flo_curves prunes by bounding
    // box, so an unguarded run of this convex fixture returns in under a second
    // and any wall-clock bound loose enough for CI would pass. `operandIndex`
    // is: only a pre-solve, per-operand gate can attribute a `too_complex` to
    // operand 0, and no cap downstream of normalization carries an operand at
    // all. Deleting the comparison inside `check_pairwise_work` turns this
    // assertion red (it reports an `ok` outcome).
    let ring = polygon_points_json(2_100, 100.0);
    let exact = solve_on(
        &parse_request(&request_json(
            "union",
            json!([{ "points": ring.clone() }, { "points": ring.clone() }]),
        ))
        .expect("parse"),
        Lane::Exact,
    )
    .expect_err("2_100² curve pairs must be refused on the exact lane");
    assert_eq!(exact.code, "too_complex");
    assert_eq!(
        exact.operand_index,
        Some(0),
        "the refusal must come from the per-operand normalization gate"
    );
    // The shipped lane does not fail the request over it: the polygon lane
    // answers the group flo_curves may not take (one ring, the union of two
    // copies of itself).
    let r = solve_json(
        "union",
        json!([{ "points": ring.clone() }, { "points": ring }]),
    );
    assert_eq!(r["status"], "ok", "the polygon lane answers: {r}");
    assert_eq!(r["subpathCount"], 1);

    // The two budget gates as pure functions. `check_pairwise_work` is the
    // single comparison every flo_curves entry routes through (normalization
    // and every fold step), so it is pinned at its boundary here as well as
    // end-to-end above. `MAX_INTERMEDIATE_CUBICS` (50_000) sits far above the
    // 8_000-cubic input budget and is only reachable through explosive
    // intersection growth, which is not cheap to construct, so it is pinned as
    // a function rather than through a fixture.
    assert!(check_pairwise_work(MAX_PAIRWISE_WORK, "at the limit").is_ok());
    assert_eq!(
        check_pairwise_work(MAX_PAIRWISE_WORK + 1.0, "over the limit")
            .expect_err("over MAX_PAIRWISE_WORK must be refused")
            .code,
        "too_complex"
    );
    assert!(check_intermediate(MAX_INTERMEDIATE_CUBICS).is_ok());
    assert_eq!(
        check_intermediate(MAX_INTERMEDIATE_CUBICS + 1)
            .expect_err("over MAX_INTERMEDIATE_CUBICS must be refused")
            .code,
        "too_complex"
    );

    // too_many_points — the request-byte budget, the only guard that runs
    // before `serde_json` builds a `Value` tree. Without it a 10M-point operand
    // is materialized in full (~500 B of `Value` per point, over a gigabyte)
    // and only then refused; on wasm that allocation aborts the shared engine
    // instance rather than this one call.
    //
    // 10.1 — the resource is "serde never ran", and the parse counter is what
    // pins it. The error code alone cannot: the filler below is not valid JSON,
    // so an ungated run still returns an error envelope (`invalid_json`) — and
    // has already paid for the parse by the time it does.
    let oversized = format!(
        "{{\"op\":\"union\",\"operands\":[{}]}}",
        "x".repeat(MAX_REQUEST_BYTES)
    );
    let parses_before = parse_count();
    let r: Value = crate::value::from_str(&boolean_paths_json(&oversized)).unwrap();
    assert_eq!(r["code"], "too_many_points", "over-budget request bytes");
    assert_eq!(
        parse_count(),
        parses_before,
        "the byte gate must refuse before serde builds a Value tree"
    );
    assert!(check_request_bytes(MAX_REQUEST_BYTES).is_ok());
    assert_eq!(
        check_request_bytes(MAX_REQUEST_BYTES + 1)
            .expect_err("over MAX_REQUEST_BYTES must be refused")
            .code,
        "too_many_points"
    );

    // too_many_points — the cross-operand total. Five operands of 8_200 points
    // each: every one is inside MAX_POINTS_PER_OPERAND (10_000) and the request
    // is inside the byte budget, so neither of those caps can be what refuses
    // it. 41_000 points are over MAX_POINTS_TOTAL (40_000).
    //
    // 10.1 — the resource is the conversion that never happens: without this
    // cap, 2_000 × 10_000 points would expand to as many as 80M cubics and
    // *then* meet MAX_TOTAL_CUBICS. The conversion counter is what a
    // downstream `too_complex` cannot satisfy.
    let bulk: Vec<Value> = (0..5)
        .map(|_| json!({ "points": polygon_points_json(8_200, 100.0) }))
        .collect();
    let conversions_before = conversion_count();
    let r = solve_json("union", Value::Array(bulk));
    assert_eq!(r["code"], "too_many_points", "5 × 8_200 points in total");
    assert!(
        r.get("operandIndex").is_none(),
        "a cross-operand budget is not attributable to one operand"
    );
    assert_eq!(
        conversion_count(),
        conversions_before,
        "the point budget must refuse before points_to_subpaths expands anything"
    );
    assert_eq!(
        MAX_POINTS_TOTAL, 40_000,
        "the fixture above is sized on this"
    );

    // too_complex — the result-point cap. MAX_RESULT_POINTS is the same
    // constant as MAX_POINTS_PER_OPERAND and mirrors the tessellator's
    // MAX_PATH_SEGMENTS, which answers an over-cap path by silently slicing it
    // to the first 10_000 commands. The cubic budgets do not cover this: at
    // MAX_INTERMEDIATE_CUBICS (50_000) an `ok` envelope could carry ~50_000
    // points, five times what the renderer will draw.
    //
    // Two perpendicular 60-bar grids reach it from operands that are inside
    // every other guard — 240 points and 240 cubics each, 57_600 pairwise work
    // against a 4e6 ceiling — and intersect into 3_600 squares. 10.1: the
    // message carries the count, so the assertion below pins the size of the
    // result that was refused rather than just the fact of a refusal.
    let r = solve_json(
        "intersect",
        json!([
            { "points": bars_json(60, true) },
            { "points": bars_json(60, false) },
        ]),
    );
    assert_eq!(r["code"], "too_complex", "60 × 60 grid intersection");
    let message = r["message"].as_str().unwrap_or_default().to_string();
    let produced: usize = message
        .split_whitespace()
        .nth(2)
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("expected a point count in {message:?}"));
    assert!(
        produced > MAX_RESULT_POINTS,
        "refused a result of {produced} points, which is not over the cap"
    );

    assert!(check_result_points(MAX_RESULT_POINTS).is_ok());
    assert_eq!(
        check_result_points(MAX_RESULT_POINTS + 1)
            .expect_err("over MAX_RESULT_POINTS must be refused")
            .code,
        "too_complex"
    );

    // solver_failed — reachable two ways. A real panic inside flo_curves cannot
    // be provoked from a fixture, so the envelope shaping is pinned directly.
    let shaped = envelope(Err(BooleanError {
        code: "solver_failed".to_string(),
        message: "boom".to_string(),
        operand_index: None,
    }));
    assert_eq!(shaped["status"], "error");
    assert_eq!(shaped["code"], "solver_failed");
    assert!(shaped.get("operandIndex").is_none());

    // The other way is the output gate. A non-finite coordinate cannot be
    // injected through the request — `validate` refuses it on the way in — so
    // the scan is pinned as the pure function `solve` calls on the way out. It
    // has to cover control points and arc parameters, not just anchors:
    // `json!` encodes any non-finite f64 as `null`, so one missed field ships
    // as `{"type":"curve","cp1":[null,null], ...}` inside a `status: "ok"`
    // envelope and lands in a saved path node (6.3b).
    assert!(check_output_finite(&rect(0.0, 0.0, 10.0, 10.0)).is_ok());
    let nan = f64::NAN;
    let poisoned: [(&str, PathPoint); 6] = [
        ("anchor x", PathPoint::Line { x: nan, y: 0.0 }),
        ("anchor y", PathPoint::Move { x: 0.0, y: nan }),
        (
            "cp1",
            PathPoint::Curve {
                x: 1.0,
                y: 1.0,
                cp1: (nan, 0.0),
                cp2: (1.0, 1.0),
            },
        ),
        (
            "cp2",
            PathPoint::Curve {
                x: 1.0,
                y: 1.0,
                cp1: (0.0, 0.0),
                cp2: (1.0, nan),
            },
        ),
        (
            "quadratic control",
            PathPoint::Quadratic {
                x: 1.0,
                y: 1.0,
                cp: (nan, 1.0),
            },
        ),
        (
            "arc radius",
            PathPoint::Arc {
                x: 1.0,
                y: 1.0,
                rx: nan,
                ry: 2.0,
                x_axis_rotation: 0.0,
                large_arc: false,
                sweep: true,
            },
        ),
    ];
    for (what, point) in poisoned {
        let mut points = rect(0.0, 0.0, 10.0, 10.0);
        points.push(point);
        let Err(e) = check_output_finite(&points) else {
            panic!("non-finite {what} must be refused");
        };
        assert_eq!(e.code, "solver_failed", "non-finite {what}");
        // And the envelope a caller sees is an error, not an `ok` carrying
        // nulls — which is the whole point of catching it before serialization.
        let shaped = envelope(Err(e));
        assert_eq!(shaped["status"], "error", "non-finite {what}");
        assert!(shaped.get("points").is_none(), "non-finite {what}");
    }
    // Infinity is the other non-finite case, and `f64::min`/`max` do propagate
    // it, so it would otherwise pass a bounds sanity check unnoticed.
    assert_eq!(
        check_output_finite(&[PathPoint::Line {
            x: f64::INFINITY,
            y: 0.0
        }])
        .expect_err("an infinite coordinate must be refused")
        .code,
        "solver_failed"
    );
}

#[test]
fn accuracy_is_clamped_rather_than_rejected() {
    for accuracy in [-5.0, 0.0, 1e-9, 1000.0] {
        let body = json!({
            "op": "union",
            "accuracy": accuracy,
            "operands": [
                { "points": points_json(&rect(0.0, 0.0, 10.0, 10.0)) },
                { "points": points_json(&rect(5.0, 5.0, 10.0, 10.0)) },
            ],
        });
        let r: Value = crate::value::from_str(&boolean_paths_json(&body.to_string())).unwrap();
        assert_eq!(r["status"], "ok", "accuracy {accuracy}");
    }
}

#[test]
fn degenerate_operands_do_not_panic_and_report_empty_or_shape() {
    let zero = rect(0.0, 0.0, 0.0, 0.0);
    let out = run(
        BooleanOp::Union,
        vec![
            operand(zero, FillRule::NonZero),
            operand(rect(0.0, 0.0, 10.0, 10.0), FillRule::NonZero),
        ],
    );
    let (points, count, _) = shape(out);
    assert_eq!(count, 1);
    assert_close(
        even_odd_area(&points),
        100.0,
        0.05,
        "zero-size operand ignored",
    );
}

#[test]
fn both_operands_degenerate_is_empty() {
    let out = run(
        BooleanOp::Union,
        vec![
            operand(rect(0.0, 0.0, 0.0, 0.0), FillRule::NonZero),
            operand(vec![PathPoint::Move { x: 3.0, y: 3.0 }], FillRule::NonZero),
        ],
    );
    assert!(matches!(out, BooleanOutcome::Empty));
}

/// A balanced figure-8 (a bowtie, or a curved "8"): its lobes wind opposite ways, so its net signed area is 0, but
/// both lobes are filled (non-zero and even-odd alike). It was dropped as degenerate, so it vanished from every
/// boolean (union kept only the other operand; subtract said the shapes don't overlap).
#[test]
fn a_balanced_figure_eight_is_not_dropped_as_degenerate() {
    let bowtie = vec![
        PathPoint::Move { x: 10.0, y: 10.0 },
        PathPoint::Line { x: 150.0, y: 150.0 },
        PathPoint::Line { x: 150.0, y: 10.0 },
        PathPoint::Line { x: 10.0, y: 150.0 },
    ];
    // Two 70 × 140 triangles (4900 each) meeting at (80, 80); the circle there has half its area inside them.
    let circle = ellipse(80.0, 80.0, 20.0, 20.0);
    let half_circle = std::f64::consts::PI * 400.0 / 2.0;
    for fill in [FillRule::NonZero, FillRule::EvenOdd] {
        let (union, _, _) = shape(run(
            BooleanOp::Union,
            vec![
                operand(bowtie.clone(), fill),
                operand(circle.clone(), FillRule::NonZero),
            ],
        ));
        assert_close(
            even_odd_area(&union),
            9800.0 + half_circle,
            15.0,
            "bowtie ∪ circle",
        );
        let (cut, _, _) = shape(run(
            BooleanOp::Subtract,
            vec![
                operand(bowtie.clone(), fill),
                operand(circle.clone(), FillRule::NonZero),
            ],
        ));
        assert_close(
            even_odd_area(&cut),
            9800.0 - half_circle,
            15.0,
            "bowtie − circle",
        );
    }
    // A curved 8.
    let eight = vec![
        PathPoint::Move { x: 10.0, y: 80.0 },
        PathPoint::Curve {
            x: 150.0,
            y: 80.0,
            cp1: (10.0, -20.0),
            cp2: (150.0, 180.0),
        },
        PathPoint::Curve {
            x: 10.0,
            y: 80.0,
            cp1: (150.0, -20.0),
            cp2: (10.0, 180.0),
        },
    ];
    let (union, _, _) = shape(run(
        BooleanOp::Union,
        vec![
            operand(eight, FillRule::NonZero),
            operand(rect(300.0, 0.0, 10.0, 10.0), FillRule::NonZero),
        ],
    ));
    assert!(
        even_odd_area(&union) > 2000.0,
        "the 8 survives: {}",
        even_odd_area(&union)
    );
    // A spur (out and back along one line) still encloses nothing and is still dropped.
    let spur = vec![
        PathPoint::Move { x: 0.0, y: 0.0 },
        PathPoint::Line { x: 100.0, y: 0.0 },
        PathPoint::Line { x: 0.0, y: 0.0 },
    ];
    let (only, _, _) = shape(run(
        BooleanOp::Union,
        vec![
            operand(spur, FillRule::NonZero),
            operand(rect(300.0, 0.0, 10.0, 10.0), FillRule::NonZero),
        ],
    ));
    assert_close(even_odd_area(&only), 100.0, 0.5, "the spur adds nothing");
}

/// Balanced figure-8s (a random lobe and its mirror image, wound the other way: net area 0) now reach flo_curves.
/// Every boolean and Pathfinder op on one and a random shape answers without a solver failure (a panic, which the
/// wasm build would abort on; natively `catch_unwind` turns it into `solver_failed`).
#[test]
fn balanced_figure_eights_never_fail_the_solver() {
    use super::panic_fuzz_tests::{Rng, BOOLEAN_OPS, PATHFINDER_OPS};
    for case in 0..60u64 {
        let mut rng = Rng::new(0xF18E ^ case.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let (cx, cy) = (rng.f(80.0, 120.0), rng.f(80.0, 120.0));
        let lobe: Vec<(f64, f64)> = (0..2 + rng.below(4))
            .map(|_| (cx + rng.f(5.0, 60.0), cy + rng.f(-50.0, 50.0)))
            .collect();
        let curved = rng.below(2) == 0;
        let mut eight = vec![PathPoint::Move { x: cx, y: cy }];
        for side in [1.0, -1.0] {
            for &(x, y) in lobe.iter().chain(std::iter::once(&(cx, cy))) {
                let x = cx + side * (x - cx);
                if curved {
                    eight.push(PathPoint::Curve {
                        x,
                        y,
                        cp1: (x + rng.f(-20.0, 20.0), y + rng.f(-20.0, 20.0)),
                        cp2: (x, y),
                    });
                } else {
                    eight.push(PathPoint::Line { x, y });
                }
            }
        }
        let other = if rng.below(2) == 0 {
            ellipse(
                rng.f(40.0, 160.0),
                rng.f(40.0, 160.0),
                rng.f(5.0, 60.0),
                rng.f(5.0, 60.0),
            )
        } else {
            rect(
                rng.f(20.0, 140.0),
                rng.f(20.0, 140.0),
                rng.f(5.0, 80.0),
                rng.f(5.0, 80.0),
            )
        };
        for op in BOOLEAN_OPS {
            for operands in [
                vec![
                    operand(eight.clone(), FillRule::NonZero),
                    operand(other.clone(), FillRule::NonZero),
                ],
                vec![
                    operand(other.clone(), FillRule::NonZero),
                    operand(eight.clone(), FillRule::EvenOdd),
                ],
            ] {
                if let Err(e) = boolean_paths(&BooleanRequest {
                    op,
                    operands,
                    accuracy: None,
                }) {
                    assert_ne!(e.code, "solver_failed", "case {case} {op:?}: {}", e.message);
                }
            }
        }
        for op in PATHFINDER_OPS {
            let outcome = pathfinder_paths(&PathfinderRequest {
                op,
                operands: vec![
                    operand(eight.clone(), FillRule::NonZero),
                    operand(other.clone(), FillRule::NonZero),
                ],
                paint: vec![
                    OperandPaint {
                        painted: true,
                        key: Some("a".into())
                    };
                    2
                ],
                accuracy: None,
            });
            if let Err(e) = outcome {
                assert_ne!(e.code, "solver_failed", "case {case} {op:?}: {}", e.message);
            }
        }
    }
}
