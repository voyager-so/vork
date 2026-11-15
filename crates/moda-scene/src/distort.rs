//! Shared distort homography math — the moda-scene mirror of the pure TS
//! bakers/validators in `core/scene/projectiveTransform.ts` (node-distort).
//!
//! The stored-distort validity check (`display_list.rs`) lives here.
//!
//! Every function mirrors its TS twin op-for-op (same order of float
//! operations, same epsilons, same accept/reject gates) so the two engines
//! agree for bit-identical parsed inputs. (moda-engine's `paint/homography.rs` carries the READER
//! half — validation/consumption of a stored matrix — for the render lane.)

/// `EPSILON` (projectiveTransform.ts:17).
pub(crate) const QUAD_EPSILON: f64 = 1e-10;

pub(crate) const IDENTITY_DISTORT: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// The TL/TR/BR/BL unit square (`UNIT_QUAD`, projectiveTransform.ts:44).
pub(crate) const UNIT_QUAD: [(f64, f64); 4] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];

fn cross(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    let ab = (b.0 - a.0, b.1 - a.1);
    let bc = (c.0 - b.0, c.1 - b.1);
    ab.0 * bc.1 - ab.1 * bc.0
}

fn signed_area(q: &[(f64, f64); 4]) -> f64 {
    let mut area = 0.0;
    for i in 0..4 {
        let cur = q[i];
        let next = q[(i + 1) % 4];
        area += cur.0 * next.1 - next.0 * cur.1;
    }
    area / 2.0
}

/// `isValidConvexQuad` (projectiveTransform.ts:103).
pub(crate) fn is_valid_convex_quad(q: &[(f64, f64); 4]) -> bool {
    if q.iter().any(|p| !p.0.is_finite() || !p.1.is_finite()) {
        return false;
    }
    if signed_area(q).abs() < QUAD_EPSILON {
        return false;
    }
    let mut sign = 0.0_f64;
    for i in 0..4 {
        let turn = cross(q[i], q[(i + 1) % 4], q[(i + 2) % 4]);
        if !turn.is_finite() || turn.abs() < QUAD_EPSILON {
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

/// `normalizeDistortMatrix` (projectiveTransform.ts): rescale so h33 == 1
/// (`+ 0.0` canonicalizes -0 for clean storage).
pub(crate) fn normalize_distort_matrix(m: &[f64; 9]) -> Option<[f64; 9]> {
    let h33 = m[8];
    if !h33.is_finite() || h33.abs() < QUAD_EPSILON {
        return None;
    }
    let mut out = [0.0f64; 9];
    for (i, v) in m.iter().enumerate() {
        out[i] = v / h33 + 0.0;
        if !out[i].is_finite() {
            return None;
        }
    }
    Some(out)
}

/// `isValidDistortMatrix` (projectiveTransform.ts): finite, invertible (the
/// full `invertProjective` gate — determinant AND every inverse entry finite,
/// mirroring moda-engine's `paint::homography::is_valid_distort_matrix`),
/// positive projective denominator at every unit corner, and the image of the
/// unit square is a non-degenerate convex quad.
pub(crate) fn is_valid_distort_matrix(m: &[f64; 9]) -> bool {
    if !m.iter().all(|v| v.is_finite()) {
        return false;
    }
    // Cofactor expansion in the exact TS `invertProjective` op order.
    let c11 = m[4] * m[8] - m[5] * m[7];
    let c12 = -(m[3] * m[8] - m[5] * m[6]);
    let c13 = m[3] * m[7] - m[4] * m[6];
    let c21 = -(m[1] * m[8] - m[2] * m[7]);
    let c22 = m[0] * m[8] - m[2] * m[6];
    let c23 = -(m[0] * m[7] - m[1] * m[6]);
    let c31 = m[1] * m[5] - m[2] * m[4];
    let c32 = -(m[0] * m[5] - m[2] * m[3]);
    let c33 = m[0] * m[4] - m[1] * m[3];
    let det = m[0] * c11 + m[1] * c12 + m[2] * c13;
    if !det.is_finite() || det.abs() < QUAD_EPSILON {
        return false;
    }
    let inv_det = 1.0 / det;
    // TS additionally requires every inverse entry to be finite; a matrix with
    // ~1e150+ entries can overflow a cofactor while keeping det finite.
    if ![c11, c12, c13, c21, c22, c23, c31, c32, c33]
        .iter()
        .all(|c| (c * inv_det).is_finite())
    {
        return false;
    }
    let mut image = [(0.0, 0.0); 4];
    for (i, (x, y)) in UNIT_QUAD.iter().enumerate() {
        let denominator = m[6] * x + m[7] * y + m[8];
        if !denominator.is_finite() || denominator <= QUAD_EPSILON {
            return false;
        }
        let px = (m[0] * x + m[1] * y + m[2]) / denominator;
        let py = (m[3] * x + m[4] * y + m[5]) / denominator;
        if !px.is_finite() || !py.is_finite() {
            return false;
        }
        image[i] = (px, py);
    }
    is_valid_convex_quad(&image)
}
