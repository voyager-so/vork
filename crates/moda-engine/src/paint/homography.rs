//! homography.rs — port of `core/scene/projectiveTransform.ts` (M3b).
//!
//! Convention (TS header comment):
//!   x' = (h11*x + h12*y + h13) / (h31*x + h32*y + h33)
//!   y' = (h21*x + h22*y + h23) / (h31*x + h32*y + h33)
//!
//! The renderer-side pieces of the port: quad helpers (`getDefaultQuad`,
//! `isValidConvexQuad`), the DLT (`cornersToHomography` with its pivoting
//! Gauss-Jordan solver — parity-pinned by the tests below), point application,
//! composition/inversion, and the stored-`distort`-matrix validation /
//! local-transform helpers (`DistortMatrix` in
//! `core/scene/projectiveTransform.ts`). All math in f64 like the JS.

/// `EPSILON = 1e-10` (`core/scene/projectiveTransform.ts:17`).
const EPSILON: f64 = 1e-10;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

pub type Quad = [Point2D; 4];

/// 3x3 homography (`ProjectiveTransform`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectiveTransform {
    pub h11: f64,
    pub h12: f64,
    pub h13: f64,
    pub h21: f64,
    pub h22: f64,
    pub h23: f64,
    pub h31: f64,
    pub h32: f64,
    pub h33: f64,
}

/// `getDefaultQuad` (`projectiveTransform.ts:33-40`).
pub fn default_quad(width: f64, height: f64) -> Quad {
    [
        Point2D { x: 0.0, y: 0.0 },
        Point2D { x: width, y: 0.0 },
        Point2D {
            x: width,
            y: height,
        },
        Point2D { x: 0.0, y: height },
    ]
}

fn is_finite_point(p: &Point2D) -> bool {
    p.x.is_finite() && p.y.is_finite()
}

/// `cross(a, b, c)` (`projectiveTransform.ts:68-74`).
fn cross(a: &Point2D, b: &Point2D, c: &Point2D) -> f64 {
    let ab_x = b.x - a.x;
    let ab_y = b.y - a.y;
    let bc_x = c.x - b.x;
    let bc_y = c.y - b.y;
    ab_x * bc_y - ab_y * bc_x
}

/// `signedArea` (`projectiveTransform.ts:76-84`).
fn signed_area(points: &Quad) -> f64 {
    let mut area = 0.0;
    for i in 0..points.len() {
        let current = &points[i];
        let next = &points[(i + 1) % points.len()];
        area += current.x * next.y - next.x * current.y;
    }
    area / 2.0
}

/// `isValidConvexQuad` (`projectiveTransform.ts:85-103`).
pub fn is_valid_convex_quad(points: &Quad) -> bool {
    if !points.iter().all(is_finite_point) {
        return false;
    }
    if signed_area(points).abs() < EPSILON {
        return false;
    }
    let mut sign = 0.0f64;
    for i in 0..points.len() {
        let turn = cross(
            &points[i],
            &points[(i + 1) % points.len()],
            &points[(i + 2) % points.len()],
        );
        if !turn.is_finite() || turn.abs() < EPSILON {
            return false;
        }
        let next_sign = turn.signum();
        if sign == 0.0 {
            sign = next_sign;
        } else if next_sign != sign {
            return false;
        }
    }
    true
}

/// `solveLinearSystem` (`projectiveTransform.ts:139-178`): Gauss-Jordan with
/// partial pivoting, exact port (including the < EPSILON pivot/factor skips).
fn solve_linear_system(matrix: &[[f64; 8]; 8], vector: &[f64; 8]) -> Option<[f64; 8]> {
    let size = 8usize;
    let mut augmented = [[0.0f64; 9]; 8];
    for (i, row) in matrix.iter().enumerate() {
        augmented[i][..8].copy_from_slice(row);
        augmented[i][8] = vector[i];
    }

    for col in 0..size {
        let mut pivot_row = col;
        let mut pivot_abs = augmented[col][col].abs();
        for row in (col + 1)..size {
            let candidate_abs = augmented[row][col].abs();
            if candidate_abs > pivot_abs {
                pivot_abs = candidate_abs;
                pivot_row = row;
            }
        }
        if !pivot_abs.is_finite() || pivot_abs < EPSILON {
            return None;
        }
        if pivot_row != col {
            augmented.swap(col, pivot_row);
        }
        let pivot = augmented[col][col];
        for entry in col..=size {
            augmented[col][entry] /= pivot;
        }
        for row in 0..size {
            if row == col {
                continue;
            }
            let factor = augmented[row][col];
            if factor.abs() < EPSILON {
                continue;
            }
            for entry in col..=size {
                augmented[row][entry] -= factor * augmented[col][entry];
            }
        }
    }

    let mut solution = [0.0f64; 8];
    for i in 0..size {
        solution[i] = augmented[i][8];
        if !solution[i].is_finite() {
            return None;
        }
    }
    Some(solution)
}

/// `cornersToHomography(source, destination)` (`projectiveTransform.ts:181-204`).
/// Maps `source[i]` → `destination[i]`; `h33` fixed to 1.
pub fn corners_to_homography(source: &Quad, destination: &Quad) -> Option<ProjectiveTransform> {
    if !is_valid_convex_quad(source) || !is_valid_convex_quad(destination) {
        return None;
    }

    let mut matrix = [[0.0f64; 8]; 8];
    let mut vector = [0.0f64; 8];
    for i in 0..4 {
        let Point2D { x, y } = source[i];
        let u = destination[i].x;
        let v = destination[i].y;
        matrix[i * 2] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y];
        vector[i * 2] = u;
        matrix[i * 2 + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y];
        vector[i * 2 + 1] = v;
    }

    let s = solve_linear_system(&matrix, &vector)?;
    Some(ProjectiveTransform {
        h11: s[0],
        h12: s[1],
        h13: s[2],
        h21: s[3],
        h22: s[4],
        h23: s[5],
        h31: s[6],
        h32: s[7],
        h33: 1.0,
    })
}

/// `applyProjectiveToPoint` (`projectiveTransform.ts`).
pub fn apply_projective_to_point(p: Point2D, t: &ProjectiveTransform) -> Option<Point2D> {
    if !is_finite_point(&p) {
        return None;
    }
    let denominator = t.h31 * p.x + t.h32 * p.y + t.h33;
    if !denominator.is_finite() || denominator.abs() < EPSILON {
        return None;
    }
    let x = (t.h11 * p.x + t.h12 * p.y + t.h13) / denominator;
    let y = (t.h21 * p.x + t.h22 * p.y + t.h23) / denominator;
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    Some(Point2D { x, y })
}

/// `composeProjective(a, b)` — apply `b` first, then `a` (`projectiveTransform.ts`).
pub fn compose_projective(a: &ProjectiveTransform, b: &ProjectiveTransform) -> ProjectiveTransform {
    ProjectiveTransform {
        h11: a.h11 * b.h11 + a.h12 * b.h21 + a.h13 * b.h31,
        h12: a.h11 * b.h12 + a.h12 * b.h22 + a.h13 * b.h32,
        h13: a.h11 * b.h13 + a.h12 * b.h23 + a.h13 * b.h33,
        h21: a.h21 * b.h11 + a.h22 * b.h21 + a.h23 * b.h31,
        h22: a.h21 * b.h12 + a.h22 * b.h22 + a.h23 * b.h32,
        h23: a.h21 * b.h13 + a.h22 * b.h23 + a.h23 * b.h33,
        h31: a.h31 * b.h11 + a.h32 * b.h21 + a.h33 * b.h31,
        h32: a.h31 * b.h12 + a.h32 * b.h22 + a.h33 * b.h32,
        h33: a.h31 * b.h13 + a.h32 * b.h23 + a.h33 * b.h33,
    }
}

/// `invertProjective` (`projectiveTransform.ts`): adjugate / determinant.
pub fn invert_projective(t: &ProjectiveTransform) -> Option<ProjectiveTransform> {
    let c11 = t.h22 * t.h33 - t.h23 * t.h32;
    let c12 = -(t.h21 * t.h33 - t.h23 * t.h31);
    let c13 = t.h21 * t.h32 - t.h22 * t.h31;
    let c21 = -(t.h12 * t.h33 - t.h13 * t.h32);
    let c22 = t.h11 * t.h33 - t.h13 * t.h31;
    let c23 = -(t.h11 * t.h32 - t.h12 * t.h31);
    let c31 = t.h12 * t.h23 - t.h13 * t.h22;
    let c32 = -(t.h11 * t.h23 - t.h13 * t.h21);
    let c33 = t.h11 * t.h22 - t.h12 * t.h21;

    let determinant = t.h11 * c11 + t.h12 * c12 + t.h13 * c13;
    if !determinant.is_finite() || determinant.abs() < EPSILON {
        return None;
    }
    let inv_det = 1.0 / determinant;
    let inverse = ProjectiveTransform {
        h11: c11 * inv_det,
        h12: c21 * inv_det,
        h13: c31 * inv_det,
        h21: c12 * inv_det,
        h22: c22 * inv_det,
        h23: c32 * inv_det,
        h31: c13 * inv_det,
        h32: c23 * inv_det,
        h33: c33 * inv_det,
    };
    let all_finite = [
        inverse.h11,
        inverse.h12,
        inverse.h13,
        inverse.h21,
        inverse.h22,
        inverse.h23,
        inverse.h31,
        inverse.h32,
        inverse.h33,
    ]
    .iter()
    .all(|v| v.is_finite());
    all_finite.then_some(inverse)
}

/// Stored node distortion: row-major 3×3 homography over the node's NORMALIZED
/// unit square (`DistortMatrix` in `projectiveTransform.ts`).
pub type DistortMatrix = [f64; 9];

/// `IDENTITY_DISTORT_MATRIX` (`projectiveTransform.ts`).
pub const IDENTITY_DISTORT_MATRIX: DistortMatrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// `areDistortMatricesApproximatelyEqual` (`projectiveTransform.ts`, default
/// epsilon 1e-9).
pub fn are_distort_matrices_approximately_equal(
    a: &DistortMatrix,
    b: &DistortMatrix,
    epsilon: f64,
) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() <= epsilon)
}

/// `matrixToProjective` (`projectiveTransform.ts`).
pub fn matrix_to_projective(m: &DistortMatrix) -> ProjectiveTransform {
    ProjectiveTransform {
        h11: m[0],
        h12: m[1],
        h13: m[2],
        h21: m[3],
        h22: m[4],
        h23: m[5],
        h31: m[6],
        h32: m[7],
        h33: m[8],
    }
}

/// `isValidDistortMatrix` (`projectiveTransform.ts`): finite, invertible (the
/// full `invertProjective` gate — determinant AND every inverse entry finite,
/// exactly like the TS), positive projective denominator at every unit-square
/// corner, and a non-degenerate convex image of the unit square. A mirror /
/// reflection is INTENTIONALLY accepted (free-distort may legitimately
/// reflect); gestures that must not reflect enforce that in their own baker.
pub fn is_valid_distort_matrix(m: &DistortMatrix) -> bool {
    if !m.iter().all(|v| v.is_finite()) {
        return false;
    }
    let t = matrix_to_projective(m);
    if invert_projective(&t).is_none() {
        return false;
    }
    let unit = default_quad(1.0, 1.0);
    let mut corners = [Point2D { x: 0.0, y: 0.0 }; 4];
    for (i, corner) in unit.iter().enumerate() {
        let denominator = t.h31 * corner.x + t.h32 * corner.y + t.h33;
        if !denominator.is_finite() || denominator <= EPSILON {
            return false;
        }
        match apply_projective_to_point(*corner, &t) {
            Some(p) => corners[i] = p,
            None => return false,
        }
    }
    is_valid_convex_quad(&corners)
}

/// `getUnitDistortLocalTransform` (`projectiveTransform.ts`): the node-local-px
/// homography scale(w, h) ∘ H ∘ scale(1/w, 1/h) for a stored normalized matrix.
/// Requires finite positive dims exactly like the TS (`Infinity > 0` is true in
/// Rust too, so the finite checks are load-bearing).
pub fn unit_distort_local_transform(
    m: &DistortMatrix,
    width: f64,
    height: f64,
) -> Option<ProjectiveTransform> {
    if !(width > 0.0) || !(height > 0.0) || !width.is_finite() || !height.is_finite() {
        return None;
    }
    let scale_up = ProjectiveTransform {
        h11: width,
        h12: 0.0,
        h13: 0.0,
        h21: 0.0,
        h22: height,
        h23: 0.0,
        h31: 0.0,
        h32: 0.0,
        h33: 1.0,
    };
    let scale_down = ProjectiveTransform {
        h11: 1.0 / width,
        h12: 0.0,
        h13: 0.0,
        h21: 0.0,
        h22: 1.0 / height,
        h23: 0.0,
        h31: 0.0,
        h32: 0.0,
        h33: 1.0,
    };
    Some(compose_projective(
        &scale_up,
        &compose_projective(&matrix_to_projective(m), &scale_down),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(p: Point2D, t: &ProjectiveTransform) -> Option<Point2D> {
        apply_projective_to_point(p, t)
    }

    #[test]
    fn identity_homography_for_identical_quads() {
        let q = default_quad(100.0, 50.0);
        let h = corners_to_homography(&q, &q).unwrap();
        assert!((h.h11 - 1.0).abs() < 1e-9 && (h.h22 - 1.0).abs() < 1e-9);
        assert!(h.h12.abs() < 1e-9 && h.h31.abs() < 1e-12);
    }

    #[test]
    fn corner_pin_round_trip() {
        // The patterns/distort-pattern fixture quad (pp-corner-pin) mapped back to
        // the 480x320 source rect: dest corner i must land on source corner i.
        let dest: Quad = [
            Point2D { x: 60.0, y: 30.0 },
            Point2D { x: 440.0, y: 0.0 },
            Point2D { x: 480.0, y: 320.0 },
            Point2D { x: 0.0, y: 290.0 },
        ];
        let source = default_quad(480.0, 320.0);
        // inverseHomography = cornersToHomography(destination, source)
        // (`shapeToPathNodeUtils.ts:236`).
        let inv = corners_to_homography(&dest, &source).unwrap();
        for (d, s) in dest.iter().zip(source.iter()) {
            let mapped = apply(*d, &inv).unwrap();
            assert!(
                (mapped.x - s.x).abs() < 1e-6 && (mapped.y - s.y).abs() < 1e-6,
                "corner {d:?} → {mapped:?}, want {s:?}"
            );
        }
    }

    #[test]
    fn degenerate_quad_rejected() {
        let degenerate: Quad = [
            Point2D { x: 0.0, y: 0.0 },
            Point2D { x: 100.0, y: 0.0 },
            Point2D { x: 200.0, y: 0.0 },
            Point2D { x: 300.0, y: 0.0 },
        ];
        assert!(!is_valid_convex_quad(&degenerate));
        assert!(corners_to_homography(&degenerate, &default_quad(10.0, 10.0)).is_none());
        // Concave quad rejected too.
        let concave: Quad = [
            Point2D { x: 0.0, y: 0.0 },
            Point2D { x: 100.0, y: 0.0 },
            Point2D { x: 20.0, y: 20.0 },
            Point2D { x: 0.0, y: 100.0 },
        ];
        assert!(!is_valid_convex_quad(&concave));
    }

    // Distort-matrix parity — pins the SAME acceptance set as the TS
    // suite (tests/vitest/scene/projectiveTransform.test.ts, "distort matrix
    // helpers"): a matrix the TS reader warps must warp natively and vice versa.

    const IDENTITY_MATRIX: DistortMatrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    #[test]
    fn distort_matrix_validation_matches_ts_acceptance_set() {
        // Identity, projective lean, reflection: accepted.
        assert!(is_valid_distort_matrix(&IDENTITY_MATRIX));
        assert!(is_valid_distort_matrix(&[
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3, 0.1, 1.0
        ]));
        assert!(is_valid_distort_matrix(&[
            -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0
        ]));
        // A positively-scaled (non-h33-normalized) representative of the same
        // warp is tolerated — readers tolerate, writers normalize.
        assert!(is_valid_distort_matrix(&[
            2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.6, 0.2, 2.0
        ]));
        // Singular: rejected.
        assert!(!is_valid_distort_matrix(&[
            1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0
        ]));
        // Denominator changes sign over the unit square: rejected.
        assert!(!is_valid_distort_matrix(&[
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.5, 0.0, 1.0
        ]));
        // Collapses the unit square to a degenerate quad: rejected.
        assert!(!is_valid_distort_matrix(&[
            0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0
        ]));
        // Non-finite entries: rejected.
        assert!(!is_valid_distort_matrix(&[
            1.0,
            0.0,
            0.0,
            0.0,
            f64::NAN,
            0.0,
            0.0,
            0.0,
            1.0
        ]));
    }

    #[test]
    fn unit_distort_local_transform_guards_dims_like_ts() {
        // `Infinity > 0.0` is true in Rust too — the finite guard is load-bearing
        // (getUnitDistortLocalTransform, projectiveTransform.ts).
        for (w, h) in [
            (f64::INFINITY, 80.0),
            (100.0, f64::INFINITY),
            (f64::NAN, 80.0),
            (0.0, 80.0),
            (-100.0, 80.0),
        ] {
            assert!(
                unit_distort_local_transform(&IDENTITY_MATRIX, w, h).is_none(),
                "dims ({w}, {h}) must be rejected"
            );
        }
    }

    #[test]
    fn unit_distort_local_transform_is_resize_invariant() {
        // A normalized matrix applied over two reference boxes maps the SAME
        // relative point to proportionally scaled outputs (decision #2: the warp
        // scales with the box).
        let m: DistortMatrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3, 0.1, 1.0];
        let small = unit_distort_local_transform(&m, 100.0, 80.0).unwrap();
        let doubled = unit_distort_local_transform(&m, 200.0, 160.0).unwrap();
        for (x, y) in [(0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (40.0, 30.0)] {
            let a = apply(Point2D { x, y }, &small).unwrap();
            let b = apply(
                Point2D {
                    x: x * 2.0,
                    y: y * 2.0,
                },
                &doubled,
            )
            .unwrap();
            assert!(
                (b.x - a.x * 2.0).abs() < 1e-9 && (b.y - a.y * 2.0).abs() < 1e-9,
                "({x}, {y}): {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn invert_projective_round_trips_and_rejects_singular() {
        let m: DistortMatrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3, 0.1, 1.0];
        let local = unit_distort_local_transform(&m, 480.0, 320.0).unwrap();
        let inv = invert_projective(&local).unwrap();
        for (x, y) in [(0.0, 0.0), (480.0, 0.0), (480.0, 320.0), (123.0, 45.0)] {
            let p = Point2D { x, y };
            let there = apply(p, &local).unwrap();
            let back = apply(there, &inv).unwrap();
            assert!(
                (back.x - x).abs() < 1e-6 && (back.y - y).abs() < 1e-6,
                "({x}, {y}) → {there:?} → {back:?}"
            );
        }
        let singular = matrix_to_projective(&[1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(invert_projective(&singular).is_none());
    }
}
