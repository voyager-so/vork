//! Property + no-panic coverage.
//!
//! "No panic" is asserted as `code != "solver_failed"` on every error envelope,
//! not as the absence of a test failure: on native the solver runs under
//! `catch_unwind`, so a panic comes back as an ordinary envelope. The property
//! tests reach the solver through `tests::run`, which panics on *any* error
//! envelope and names `solver_failed` explicitly.
//!
//! The PRNG is a fixed-seed xorshift so a failure is reproducible from the seed
//! printed in the assertion message; nothing here depends on wall-clock time or
//! on the host. Case counts are sized to keep the whole module well under a
//! second in debug.

use crate::value::Value;

use super::tests::{ellipse, even_odd_area, operand, rect, rect_ccw, rounded_rect, run, shape};
use super::*;

/// xorshift64*; deterministic and tiny.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn f(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

fn rotate(points: &[PathPoint], cx: f64, cy: f64, degrees: f64) -> Vec<PathPoint> {
    let r = degrees.to_radians();
    let (c, s) = (r.cos(), r.sin());
    // Rotation about (cx, cy): translate → rotate → translate back.
    let m = crate::types::Affine {
        a: c,
        b: s,
        c: -s,
        d: c,
        e: cx - c * cx + s * cy,
        f: cy - s * cx - c * cy,
    };
    points.iter().map(|p| p.apply_affine(m)).collect()
}

/// A random closed shape from the operand vocabulary the editor allows.
fn random_shape(rng: &mut Rng) -> Vec<PathPoint> {
    let x = rng.f(-40.0, 40.0);
    let y = rng.f(-40.0, 40.0);
    let w = rng.f(6.0, 40.0);
    let h = rng.f(6.0, 40.0);
    let base = match rng.below(3) {
        0 => rect(x, y, w, h),
        1 => ellipse(x + w / 2.0, y + h / 2.0, w / 2.0, h / 2.0),
        _ => rounded_rect(x, y, w, h, (w.min(h) / 2.0).min(rng.f(1.0, 8.0))),
    };
    // Occasionally rotate, so the solver sees non-axis-aligned intersections.
    if rng.below(4) == 0 {
        rotate(&base, x + w / 2.0, y + h / 2.0, rng.f(5.0, 85.0))
    } else {
        base
    }
}

fn area_of(op: BooleanOp, a: &[PathPoint], b: &[PathPoint]) -> f64 {
    match run(
        op,
        vec![
            operand(a.to_vec(), FillRule::NonZero),
            operand(b.to_vec(), FillRule::NonZero),
        ],
    ) {
        BooleanOutcome::Empty => 0.0,
        outcome => even_odd_area(&shape(outcome).0),
    }
}

fn self_area(points: &[PathPoint]) -> f64 {
    // Union with a far-away speck, then subtract the speck's area: the cheapest
    // way to get the solver's own idea of an operand's area.
    let speck = rect(1.0e5, 1.0e5, 1.0, 1.0);
    area_of(BooleanOp::Union, points, &speck) - 1.0
}

/// Tolerance scaled to the shapes involved: flattening + the 0.01 accuracy
/// snap both contribute an absolute, not relative, error.
fn tol(scale: f64) -> f64 {
    (scale.abs() * 2.0e-3).max(0.5)
}

#[test]
fn area_identities_hold_over_random_pairs() {
    let mut rng = Rng::new(0x5EED_B001_1234_ABCD);
    let mut checked = 0;
    for case in 0..150 {
        let a = random_shape(&mut rng);
        let b = random_shape(&mut rng);
        let area_a = self_area(&a);
        let area_b = self_area(&b);
        if area_a <= 1.0 || area_b <= 1.0 {
            continue;
        }
        let union = area_of(BooleanOp::Union, &a, &b);
        let inter = area_of(BooleanOp::Intersect, &a, &b);
        let xor = area_of(BooleanOp::Exclude, &a, &b);
        let sub = area_of(BooleanOp::Subtract, &a, &b);
        let scale = area_a + area_b;
        let t = tol(scale);

        assert!(
            (union + inter - (area_a + area_b)).abs() <= t,
            "case {case}: |A∪B| + |A∩B| = {union} + {inter} != {area_a} + {area_b}"
        );
        assert!(
            (xor - (area_a + area_b - 2.0 * inter)).abs() <= t,
            "case {case}: |A⊕B| = {xor} != {area_a} + {area_b} − 2·{inter}"
        );
        assert!(
            (sub - (area_a - inter)).abs() <= t,
            "case {case}: |A−B| = {sub} != {area_a} − {inter}"
        );
        checked += 1;
    }
    assert!(
        checked > 120,
        "expected most random pairs to be usable, got {checked}"
    );
}

#[test]
fn union_and_intersect_commute() {
    let mut rng = Rng::new(0x00C0_FFEE_D00D_0001);
    for case in 0..60 {
        let a = random_shape(&mut rng);
        let b = random_shape(&mut rng);
        for op in [BooleanOp::Union, BooleanOp::Intersect, BooleanOp::Exclude] {
            let ab = area_of(op, &a, &b);
            let ba = area_of(op, &b, &a);
            assert!(
                (ab - ba).abs() <= tol(ab.max(ba)),
                "case {case} {op:?}: {ab} vs {ba}"
            );
        }
    }
}

#[test]
fn self_operations_are_idempotent_or_empty() {
    let mut rng = Rng::new(0x1357_9BDF_2468_ACE0);
    for case in 0..40 {
        let a = random_shape(&mut rng);
        let area_a = self_area(&a);
        if area_a <= 1.0 {
            continue;
        }
        let union = area_of(BooleanOp::Union, &a, &a);
        assert!(
            (union - area_a).abs() <= tol(area_a),
            "case {case}: A∪A = {union} != {area_a}"
        );
        let inter = area_of(BooleanOp::Intersect, &a, &a);
        assert!(
            (inter - area_a).abs() <= tol(area_a),
            "case {case}: A∩A = {inter} != {area_a}"
        );
        assert_eq!(
            area_of(BooleanOp::Subtract, &a, &a),
            0.0,
            "case {case}: A−A must be empty"
        );
        assert_eq!(
            area_of(BooleanOp::Exclude, &a, &a),
            0.0,
            "case {case}: A⊕A must be empty"
        );
    }
}

/// Direction is meaningful *within* one non-zero operand — an inner loop wound
/// against its outer is a hole — but a single-subpath operand fills the same
/// region either way, and that is what the editor hands over for a simple
/// shape. The four ops must not read anything into its winding.
#[test]
fn winding_direction_does_not_change_a_single_subpath_operand() {
    let a = rect(0.0, 0.0, 12.0, 9.0);
    let b = rect(6.0, 4.0, 12.0, 9.0);
    let b_ccw = rect_ccw(6.0, 4.0, 12.0, 9.0);
    for op in [
        BooleanOp::Union,
        BooleanOp::Intersect,
        BooleanOp::Subtract,
        BooleanOp::Exclude,
    ] {
        let cw = area_of(op, &a, &b);
        let ccw = area_of(op, &a, &b_ccw);
        assert!((cw - ccw).abs() <= 0.05, "{op:?}: {cw} vs {ccw}");
    }
}

#[test]
fn small_perturbations_move_the_area_only_a_little() {
    let mut rng = Rng::new(0x0BAD_F00D_0000_0007);
    for case in 0..40 {
        let a = random_shape(&mut rng);
        let b = random_shape(&mut rng);
        let base = area_of(BooleanOp::Union, &a, &b);
        if base <= 1.0 {
            continue;
        }
        let dx = rng.f(-0.05, 0.05);
        let dy = rng.f(-0.05, 0.05);
        let m = crate::types::Affine {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: dx,
            f: dy,
        };
        let nudged: Vec<PathPoint> = b.iter().map(|p| p.apply_affine(m)).collect();
        let moved = area_of(BooleanOp::Union, &a, &nudged);
        // A 0.05-unit nudge can only change the union by ~perimeter × 0.05.
        assert!(
            (moved - base).abs() <= tol(base).max(base.sqrt() * 0.6),
            "case {case}: union jumped {base} → {moved} on a {dx:.3},{dy:.3} nudge"
        );
    }
}

// ---------------------------------------------------------------------------
// No-panic fuzz
// ---------------------------------------------------------------------------

/// Deliberately nasty operands: zero size, shared edges, duplicate anchors,
/// single points, self-touching bow-ties, huge coordinates.
fn adversarial_shape(rng: &mut Rng) -> Vec<PathPoint> {
    match rng.below(10) {
        0 => rect(0.0, 0.0, 0.0, 0.0),
        1 => vec![PathPoint::Move {
            x: rng.f(-5.0, 5.0),
            y: rng.f(-5.0, 5.0),
        }],
        2 => {
            // Duplicate anchors everywhere.
            let mut pts = rect(0.0, 0.0, 10.0, 10.0);
            let dup: Vec<PathPoint> = pts.clone();
            for p in dup {
                pts.push(p.clone());
                pts.push(p);
            }
            pts
        }
        3 => {
            // Bow-tie: self-intersecting quad.
            vec![
                PathPoint::Move { x: 0.0, y: 0.0 },
                PathPoint::Line { x: 10.0, y: 10.0 },
                PathPoint::Line { x: 10.0, y: 0.0 },
                PathPoint::Line { x: 0.0, y: 10.0 },
            ]
        }
        4 => rect(0.0, 0.0, 10.0, 0.0),
        5 => ellipse(0.0, 0.0, 0.0, 5.0),
        6 => rect(rng.f(-1.0e6, 1.0e6), rng.f(-1.0e6, 1.0e6), 1.0e5, 1.0e5),
        7 => {
            // Shares an entire edge with the canonical unit rect.
            rect(10.0, 0.0, 10.0, 10.0)
        }
        8 => vec![
            PathPoint::Move { x: 0.0, y: 0.0 },
            PathPoint::Arc {
                x: 0.0,
                y: 0.0,
                rx: 0.0,
                ry: 0.0,
                x_axis_rotation: rng.f(-720.0, 720.0),
                large_arc: rng.below(2) == 0,
                sweep: rng.below(2) == 0,
            },
            PathPoint::Line { x: 5.0, y: 5.0 },
        ],
        _ => random_shape(rng),
    }
}

#[test]
fn fuzz_never_panics_and_always_returns_a_valid_envelope() {
    let mut rng = Rng::new(0xF0FF_1CE0_DEAD_BEEF);
    let ops = ["union", "subtract", "intersect", "exclude"];
    for case in 0..300 {
        let op = ops[case % ops.len()];
        let a = adversarial_shape(&mut rng);
        let b = adversarial_shape(&mut rng);
        let body = crate::value::json!({
            "op": op,
            "operands": [
                { "points": a.iter().map(PathPoint::to_json).collect::<Vec<_>>(),
                  "fillRule": if rng.below(2) == 0 { "nonzero" } else { "evenodd" } },
                { "points": b.iter().map(PathPoint::to_json).collect::<Vec<_>>(),
                  "fillRule": if rng.below(2) == 0 { "nonzero" } else { "evenodd" } },
            ],
        })
        .to_string();
        let response = boolean_paths_json(&body);
        let parsed: Value = crate::value::from_str(&response)
            .unwrap_or_else(|e| panic!("case {case} ({op}): invalid envelope {e}: {response}"));
        let status = parsed["status"].as_str().unwrap_or("");
        assert!(
            matches!(status, "ok" | "empty" | "error"),
            "case {case} ({op}): unexpected status in {response}"
        );
        if status == "ok" {
            assert_eq!(parsed["fillRule"], "evenodd", "case {case}");
            assert_eq!(parsed["closed"], true, "case {case}");
            let points = parsed["points"].as_array().expect("points");
            assert!(
                points.len() >= 3,
                "case {case}: ok with {} points",
                points.len()
            );
            for p in points {
                assert!(
                    PathPoint::from_json(p).is_some(),
                    "case {case}: bad point {p}"
                );
                // Production enforces this too, since `solve` gained
                // `check_output_finite`: a non-finite coordinate is now a
                // `solver_failed` envelope rather than an `ok` one carrying
                // `null` (`json!` encodes a non-finite f64 as `null`). Kept
                // here as the end-to-end witness — this asserts on the emitted
                // JSON, which is what a caller actually persists.
                assert!(
                    p["x"].as_f64().is_some_and(f64::is_finite)
                        && p["y"].as_f64().is_some_and(f64::is_finite),
                    "case {case}: non-finite output coordinate {p}"
                );
            }
            assert!(
                parsed["subpathCount"].as_u64().is_some_and(|n| n >= 1),
                "case {case}: bad subpathCount"
            );
            for key in ["minX", "minY", "maxX", "maxY"] {
                assert!(
                    parsed["bounds"][key].as_f64().is_some_and(f64::is_finite),
                    "case {case}: bounds.{key} not finite"
                );
            }
        } else if status == "error" {
            let code = parsed["code"].as_str();
            assert!(code.is_some(), "case {case}: error without a code");
            // 10.1 — the resource this fuzz exists to pin is "flo_curves did not
            // panic", and on native a panic does NOT surface as a failure: it is
            // caught by the `catch_unwind` in `boolean_paths` and returned as a
            // perfectly well-shaped `solver_failed` envelope. Accepting any
            // coded error would let all 300 cases panic and stay green — and
            // wasm, where the release profile is `panic = "abort"`, has no
            // `catch_unwind` to convert anything: the same panic takes down the
            // shared engine instance. A refusal from one of the module's own
            // caps is expected; a panic is not.
            assert_ne!(
                code,
                Some("solver_failed"),
                "case {case} ({op}): flo_curves panicked on {body}"
            );
        }
    }
}
