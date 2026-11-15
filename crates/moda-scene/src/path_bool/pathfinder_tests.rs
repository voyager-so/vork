//! Pathfinder ops on overlapping squares and circles: piece counts, areas and
//! which operand each piece takes its paint from.

use crate::value::{json, Value};

use super::tests::{ellipse, even_odd_area, rect};
use super::*;

/// Operand `i` of the request, painted, with its own paint key.
fn filled(points: Vec<PathPoint>) -> (BooleanOperand, OperandPaint) {
    (
        BooleanOperand {
            points,
            fill_rule: FillRule::NonZero,
        },
        OperandPaint {
            painted: true,
            key: None,
        },
    )
}

fn keyed(points: Vec<PathPoint>, key: &str) -> (BooleanOperand, OperandPaint) {
    let (o, mut p) = filled(points);
    p.key = Some(key.to_string());
    (o, p)
}

fn unfilled(points: Vec<PathPoint>) -> (BooleanOperand, OperandPaint) {
    let (o, mut p) = filled(points);
    p.painted = false;
    (o, p)
}

fn run(op: PathfinderOp, operands: Vec<(BooleanOperand, OperandPaint)>) -> Vec<PathfinderPiece> {
    let (operands, paint) = operands.into_iter().unzip();
    match pathfinder_paths(&PathfinderRequest {
        op,
        operands,
        paint,
        accuracy: None,
    }) {
        Ok(PathfinderOutcome::Pieces(pieces)) => pieces,
        Ok(PathfinderOutcome::Empty) => Vec::new(),
        Err(e) => panic!("{op:?} failed: {} {}", e.code, e.message),
    }
}

/// (source, area) per piece, sorted, for order-independent comparison.
fn summary(pieces: &[PathfinderPiece]) -> Vec<(usize, f64)> {
    let mut out: Vec<(usize, f64)> = pieces
        .iter()
        .map(|p| {
            assert!(p.closed);
            (p.source, even_odd_area(&p.points))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    out
}

fn assert_summary(pieces: &[PathfinderPiece], expected: &[(usize, f64)], tol: f64) {
    let got = summary(pieces);
    assert_eq!(
        got.len(),
        expected.len(),
        "piece count: got {got:?}, expected {expected:?}"
    );
    for ((gs, ga), (es, ea)) in got.iter().zip(expected) {
        assert_eq!(gs, es, "source: got {got:?}, expected {expected:?}");
        assert!(
            (ga - ea).abs() <= tol * ea.max(1.0),
            "area: got {got:?}, expected {expected:?}"
        );
    }
}

/// Polyline length of an outline piece (square edges are all straight).
fn length(points: &[PathPoint], closed: bool) -> f64 {
    let xy: Vec<(f64, f64)> = points
        .iter()
        .map(|p| match p {
            PathPoint::Move { x, y } | PathPoint::Line { x, y } => (*x, *y),
            PathPoint::Curve { x, y, .. } | PathPoint::Quadratic { x, y, .. } => (*x, *y),
            PathPoint::Arc { x, y, .. } => (*x, *y),
        })
        .collect();
    let mut total = 0.0;
    for w in xy.windows(2) {
        total += ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt();
    }
    if closed {
        let (a, b) = (xy[xy.len() - 1], xy[0]);
        total += ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
    }
    total
}

/// Back square A (0,0 100×100) under front square B (50,50 100×100).
fn squares() -> (Vec<PathPoint>, Vec<PathPoint>) {
    (rect(0.0, 0.0, 100.0, 100.0), rect(50.0, 50.0, 100.0, 100.0))
}

/// Two r=50 circles 50 apart: each 7853.98, the lens between them 3070.95.
const CIRCLE: f64 = std::f64::consts::PI * 2500.0;
fn lens() -> f64 {
    let (r, d): (f64, f64) = (50.0, 50.0);
    2.0 * r * r * (d / (2.0 * r)).acos() - (d / 2.0) * (4.0 * r * r - d * d).sqrt()
}
fn circles() -> (Vec<PathPoint>, Vec<PathPoint>) {
    (
        ellipse(50.0, 50.0, 50.0, 50.0),
        ellipse(100.0, 50.0, 50.0, 50.0),
    )
}

// ---------------------------------------------------------------------------
// Divide
// ---------------------------------------------------------------------------

#[test]
fn divide_squares_gives_three_faces_painted_by_the_topmost_cover() {
    let (a, b) = squares();
    let pieces = run(PathfinderOp::Divide, vec![filled(a), filled(b)]);
    assert_summary(&pieces, &[(0, 7500.0), (1, 2500.0), (1, 7500.0)], 1e-6);
}

#[test]
fn divide_circles_gives_two_crescents_and_a_lens() {
    let (a, b) = circles();
    let pieces = run(PathfinderOp::Divide, vec![filled(a), filled(b)]);
    let crescent = CIRCLE - lens();
    assert_summary(&pieces, &[(0, crescent), (1, lens()), (1, crescent)], 0.01);
}

#[test]
fn divide_splits_disconnected_faces_into_separate_pieces() {
    // A bar across the whole circle cuts it in two and is itself cut in three.
    let circle = ellipse(50.0, 50.0, 50.0, 50.0);
    let bar = rect(-10.0, 40.0, 120.0, 20.0);
    let pieces = run(PathfinderOp::Divide, vec![filled(circle), filled(bar)]);
    let summary = summary(&pieces);
    assert_eq!(
        summary.iter().filter(|(s, _)| *s == 0).count(),
        2,
        "{summary:?}"
    );
    assert_eq!(
        summary.iter().filter(|(s, _)| *s == 1).count(),
        3,
        "{summary:?}"
    );
    let total: f64 = summary.iter().map(|(_, a)| a).sum();
    let union = CIRCLE + 2400.0 - {
        // The bar's part inside the circle: the circle's band |y-50| <= 10.
        let r: f64 = 50.0;
        let h: f64 = 10.0;
        2.0 * (h * (r * r - h * h).sqrt() + r * r * (h / r).asin())
    };
    assert!((total - union).abs() < 0.01 * union, "{total} vs {union}");
}

#[test]
fn divide_keeps_a_hole_with_its_outer_ring() {
    let big = rect(0.0, 0.0, 100.0, 100.0);
    let small = rect(25.0, 25.0, 50.0, 50.0);
    let pieces = run(PathfinderOp::Divide, vec![filled(big), filled(small)]);
    assert_summary(&pieces, &[(0, 7500.0), (1, 2500.0)], 1e-6);
    let frame = pieces.iter().find(|p| p.source == 0).unwrap();
    assert_eq!(frame.subpath_count, 2);
}

#[test]
fn divide_paints_a_face_from_the_topmost_filled_cover() {
    // An unfilled front square only cuts: the overlap wears the back fill,
    // and the front-only face (covered by nothing filled) is removed, as
    // Illustrator's Divide removes unpainted artwork.
    let (a, b) = squares();
    let pieces = run(PathfinderOp::Divide, vec![filled(a), unfilled(b)]);
    assert_summary(&pieces, &[(0, 2500.0), (0, 7500.0)], 1e-6);
}

#[test]
fn divide_removes_faces_only_unfilled_shapes_cover() {
    // Illustrator 30 (pf-divide-unfilled-middle): an unfilled stroked circle
    // between two filled squares cuts them, and its part outside both is gone.
    let (a, b) = squares();
    let c = rect(-30.0, 20.0, 60.0, 60.0); // half inside A, half over nothing
    let pieces = run(
        PathfinderOp::Divide,
        vec![filled(a), unfilled(c), filled(b)],
    );
    assert!(
        pieces.iter().all(|p| p.source != 1),
        "{:?}",
        summary(&pieces)
    );
    // C cuts A (four pieces: A's rest, A ∩ C, the overlap with B, B's rest) and covers nothing more.
    assert_summary(
        &pieces,
        &[(0, 1800.0), (0, 5700.0), (2, 2500.0), (2, 7500.0)],
        1e-6,
    );
    // Nothing filled at all: nothing to keep.
    let (a, b) = squares();
    assert!(run(PathfinderOp::Divide, vec![unfilled(a), unfilled(b)]).is_empty());
}

// ---------------------------------------------------------------------------
// Trim / Merge
// ---------------------------------------------------------------------------

#[test]
fn trim_removes_hidden_parts_and_keeps_objects_whole() {
    let (a, b) = squares();
    let pieces = run(PathfinderOp::Trim, vec![filled(a), filled(b)]);
    assert_summary(&pieces, &[(0, 7500.0), (1, 10000.0)], 1e-6);

    let (a, b) = circles();
    let pieces = run(PathfinderOp::Trim, vec![filled(a), filled(b)]);
    assert_summary(&pieces, &[(0, CIRCLE - lens()), (1, CIRCLE)], 0.01);
}

#[test]
fn trim_does_not_merge_same_colour() {
    let (a, b) = squares();
    let pieces = run(PathfinderOp::Trim, vec![keyed(a, "red"), keyed(b, "red")]);
    assert_eq!(pieces.len(), 2);
}

#[test]
fn trim_drops_unfilled_operands_and_lets_them_cut_nothing() {
    let (a, b) = squares();
    let pieces = run(PathfinderOp::Trim, vec![filled(a), unfilled(b)]);
    assert_summary(&pieces, &[(0, 10000.0)], 1e-6);
}

#[test]
fn merge_unites_same_colour_and_trims_the_rest() {
    let (a, b) = squares();
    let pieces = run(PathfinderOp::Merge, vec![keyed(a, "red"), keyed(b, "red")]);
    assert_summary(&pieces, &[(1, 17500.0)], 1e-6);

    let (a, b) = squares();
    let pieces = run(PathfinderOp::Merge, vec![keyed(a, "red"), keyed(b, "blue")]);
    assert_summary(&pieces, &[(0, 7500.0), (1, 10000.0)], 1e-6);

    let (a, b) = circles();
    let pieces = run(PathfinderOp::Merge, vec![keyed(a, "red"), keyed(b, "red")]);
    assert_summary(&pieces, &[(1, 2.0 * CIRCLE - lens())], 0.01);
}

#[test]
fn merge_keeps_same_colour_objects_that_do_not_touch_apart() {
    let a = rect(0.0, 0.0, 40.0, 40.0);
    let b = rect(100.0, 0.0, 40.0, 40.0);
    let pieces = run(PathfinderOp::Merge, vec![keyed(a, "red"), keyed(b, "red")]);
    // One merged object, two separate faces, both painted from the front one.
    assert_summary(&pieces, &[(1, 1600.0), (1, 1600.0)], 1e-6);
}

#[test]
fn merge_through_a_different_colour_between() {
    // Red, blue, red stacked: the blue in the middle splits the back red, and
    // the front red joins the parts it touches.
    let back = rect(0.0, 0.0, 100.0, 100.0);
    let mid = rect(40.0, -10.0, 20.0, 120.0);
    let front = rect(0.0, 40.0, 30.0, 20.0);
    let pieces = run(
        PathfinderOp::Merge,
        vec![keyed(back, "red"), keyed(mid, "blue"), keyed(front, "red")],
    );
    // Reds: left part (40×100, front lies inside it) and right part (40×100),
    // not touching each other → two red faces. Blue: whole 20×120.
    assert_summary(&pieces, &[(1, 2400.0), (2, 4000.0), (2, 4000.0)], 1e-6);
}

// ---------------------------------------------------------------------------
// Crop
// ---------------------------------------------------------------------------

#[test]
fn crop_keeps_what_lies_inside_the_frontmost() {
    let (a, b) = squares();
    let pieces = run(PathfinderOp::Crop, vec![filled(a), filled(b)]);
    assert_summary(&pieces, &[(0, 2500.0)], 1e-6);

    let (a, b) = circles();
    let pieces = run(PathfinderOp::Crop, vec![filled(a), filled(b)]);
    assert_summary(&pieces, &[(0, lens())], 0.01);
}

#[test]
fn crop_divides_the_back_objects_into_their_visible_faces() {
    // Two back squares overlapping each other, cropped by a front square that
    // covers part of both.
    let a = rect(0.0, 0.0, 60.0, 60.0);
    let b = rect(40.0, 0.0, 60.0, 60.0);
    let clip = rect(20.0, 20.0, 60.0, 20.0);
    let pieces = run(PathfinderOp::Crop, vec![filled(a), filled(b), filled(clip)]);
    // a is visible over x∈[20,40], b over x∈[40,80], within y∈[20,40].
    assert_summary(&pieces, &[(0, 400.0), (1, 800.0)], 1e-6);
}

#[test]
fn crop_with_no_overlap_is_empty() {
    let a = rect(0.0, 0.0, 10.0, 10.0);
    let b = rect(50.0, 50.0, 10.0, 10.0);
    assert!(run(PathfinderOp::Crop, vec![filled(a), filled(b)]).is_empty());
}

// ---------------------------------------------------------------------------
// Outline
// ---------------------------------------------------------------------------

#[test]
fn outline_emits_each_edge_once_coloured_by_the_frontmost_face() {
    let (a, b) = squares();
    let pieces = run(PathfinderOp::Outline, vec![filled(a), filled(b)]);
    let mut by_source = [0.0f64; 2];
    for p in &pieces {
        by_source[p.source] += length(&p.points, p.closed);
    }
    // A's outline outside B is 300 long and wears A; the two edges of A inside
    // B (100) and B's whole outline (400) wear B.
    assert!((by_source[0] - 300.0).abs() < 1e-6, "{by_source:?}");
    assert!((by_source[1] - 500.0).abs() < 1e-6, "{by_source:?}");
    assert!(pieces.iter().any(|p| !p.closed));
}

#[test]
fn outline_of_disjoint_shapes_is_their_closed_rings() {
    let a = rect(0.0, 0.0, 10.0, 10.0);
    let b = rect(50.0, 50.0, 10.0, 10.0);
    let pieces = run(PathfinderOp::Outline, vec![filled(a), filled(b)]);
    assert_eq!(pieces.len(), 2);
    assert!(pieces.iter().all(|p| p.closed));
}

#[test]
fn outline_skips_unfilled_faces() {
    let (a, b) = squares();
    let pieces = run(PathfinderOp::Outline, vec![filled(a), unfilled(b)]);
    let total: f64 = pieces.iter().map(|p| length(&p.points, p.closed)).sum();
    // Only A's two faces are filled: A's outline (400) plus B's edges inside A (100).
    assert!((total - 500.0).abs() < 1e-6, "{total}");
    assert!(pieces.iter().all(|p| p.source == 0));
}

// ---------------------------------------------------------------------------
// Minus Back
// ---------------------------------------------------------------------------

#[test]
fn minus_back_subtracts_everything_behind_the_frontmost() {
    let (a, b) = squares();
    let pieces = run(PathfinderOp::MinusBack, vec![filled(a), filled(b)]);
    assert_summary(&pieces, &[(1, 7500.0)], 1e-6);

    let (a, b) = circles();
    let pieces = run(PathfinderOp::MinusBack, vec![filled(a), filled(b)]);
    assert_summary(&pieces, &[(1, CIRCLE - lens())], 0.01);
}

#[test]
fn minus_back_stays_one_object_when_it_splits() {
    let bar = rect(-10.0, 40.0, 120.0, 20.0);
    let circle = ellipse(50.0, 50.0, 50.0, 50.0);
    let pieces = run(PathfinderOp::MinusBack, vec![filled(bar), filled(circle)]);
    assert_eq!(pieces.len(), 1);
    assert_eq!(pieces[0].subpath_count, 2);
}

// ---------------------------------------------------------------------------
// Envelope
// ---------------------------------------------------------------------------

fn call(request: Value) -> Value {
    crate::value::from_str(&pathfinder_paths_json(&request.to_string())).unwrap()
}

fn json_points(points: &[PathPoint]) -> Value {
    Value::Array(points.iter().map(PathPoint::to_json).collect())
}

#[test]
fn json_envelope_round_trip() {
    let (a, b) = squares();
    let out = call(json!({
        "op": "divide",
        "operands": [
            { "points": json_points(&a), "fillRule": "nonzero" },
            { "points": json_points(&b), "fillRule": "nonzero", "painted": true, "paintKey": "#f00" },
        ],
    }));
    assert_eq!(out["status"], "ok");
    let pieces = out["pieces"].as_array().unwrap();
    assert_eq!(pieces.len(), 3);
    for p in pieces {
        assert_eq!(p["closed"], true);
        assert_eq!(p["fillRule"], "evenodd");
        assert!(p["source"].as_u64().unwrap() <= 1);
        assert!(p["bounds"]["maxX"].as_f64().unwrap() > p["bounds"]["minX"].as_f64().unwrap());
    }
}

#[test]
fn json_errors_are_typed() {
    let (a, b) = squares();
    assert_eq!(
        call(json!({ "op": "union", "operands": [] }))["code"],
        "invalid_request"
    );
    let bad = call(json!({
        "op": "trim",
        "operands": [
            { "points": json_points(&a) },
            { "points": json_points(&b), "painted": "yes" },
        ],
    }));
    assert_eq!(bad["code"], "invalid_request");
    assert_eq!(bad["operandIndex"], 1);
    let one = call(json!({ "op": "divide", "operands": [{ "points": json_points(&a) }] }));
    assert_eq!(one["code"], "too_few_operands");
    assert_eq!(call(json!("nope"))["code"], "invalid_request");
    assert_eq!(pathfinder_paths_json("{"), pathfinder_paths_json("{"));
}

#[test]
fn many_overlapping_operands_stay_within_budget() {
    // A 6×6 lattice of overlapping circles: the arrangement has hundreds of
    // faces, all of which must come back and account for the union's area.
    let mut operands = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            operands.push(filled(ellipse(
                i as f64 * 30.0,
                j as f64 * 30.0,
                25.0,
                25.0,
            )));
        }
    }
    let pieces = run(PathfinderOp::Divide, operands);
    assert!(pieces.len() > 100, "{}", pieces.len());
}

/// A piece bounded by one curve and its chord (a "D"), or by a single curve
/// (a teardrop), is still a piece. Emitting it used to drop the closing line
/// and leave two points, which reads as "no area", so divide lost it.
#[test]
fn divide_keeps_pieces_of_one_or_two_segments() {
    let d = vec![
        PathPoint::Move { x: 0.0, y: 0.0 },
        PathPoint::Curve {
            x: 0.0,
            y: 20.0,
            cp1: (15.0, 0.0),
            cp2: (15.0, 20.0),
        },
    ];
    let teardrop = vec![
        PathPoint::Move { x: 40.0, y: 10.0 },
        PathPoint::Curve {
            x: 40.0,
            y: 10.0,
            cp1: (70.0, -10.0),
            cp2: (70.0, 30.0),
        },
    ];
    let pieces = run(
        PathfinderOp::Divide,
        vec![filled(d.clone()), filled(teardrop.clone())],
    );
    assert_eq!(pieces.len(), 2, "{pieces:?}");
    for piece in &pieces {
        assert!(piece.points.len() >= 3, "{:?}", piece.points);
    }
    let (want_d, want_teardrop) = (even_odd_area(&d), even_odd_area(&teardrop));
    assert!(want_d > 100.0 && want_teardrop > 100.0);
    assert_summary(&pieces, &[(0, want_d), (1, want_teardrop)], 0.01);
}
