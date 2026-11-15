//! Rectangle → path-point conversion for blend geometry.
//!
//! Port of `utils/rectangleToPath.ts` + `utils/shapeTransformHelpers.ts`
//! (`applyShapeTransform`). Lives in moda-scene (not moda-engine) because
//! blend expansion is here and moda-engine already depends on moda-scene.

use crate::transform;
use crate::types::{Affine, Transform2D};
use crate::value::{json, Value};

/// `KAPPA` (`utils/path/constants.ts:9`).
const KAPPA: f64 = 0.5522847498;

#[derive(Debug, Clone, Copy)]
enum LocalPathPoint {
    Move(f64, f64),
    Line(f64, f64),
    Curve {
        x: f64,
        y: f64,
        cp1: (f64, f64),
        cp2: (f64, f64),
    },
}

/// `normalizeRoundedRectRadii` (roundedRectRadii.ts:32-62) for a scalar radius.
fn normalize_rounded_rect_radii_scalar(width: f64, height: f64, radius: f64) -> [f64; 4] {
    let w = width.abs();
    let h = height.abs();
    if w <= 0.0 || h <= 0.0 || radius <= 0.0 {
        return [0.0; 4];
    }
    let max_r = (w / 2.0).min(h / 2.0);
    let v = radius.clamp(0.0, max_r);
    let (tl, tr, br, bl) = (v, v, v, v);
    let sum_to_scale = |limit: f64, sum: f64| -> f64 {
        if sum <= 0.0 {
            1.0
        } else {
            (limit / sum).min(1.0)
        }
    };
    let scale = sum_to_scale(w, tl + tr)
        .min(sum_to_scale(h, tr + br))
        .min(sum_to_scale(w, br + bl))
        .min(sum_to_scale(h, bl + tl));
    if scale >= 1.0 {
        [tl, tr, br, bl]
    } else {
        [tl * scale, tr * scale, br * scale, bl * scale]
    }
}

/// `rectangleToPathPoints` local geometry (`rectangleToPath.ts:32-219`).
fn rectangle_local_path_points(width: f64, height: f64, corner_radius: f64) -> Vec<LocalPathPoint> {
    let radii = normalize_rounded_rect_radii_scalar(width, height, corner_radius);
    let [tl, tr, br, bl] = radii;
    let mut out = Vec::new();

    if radii.iter().all(|r| *r == 0.0) {
        out.push(LocalPathPoint::Move(width, 0.0));
        out.push(LocalPathPoint::Line(width, height));
        out.push(LocalPathPoint::Line(0.0, height));
        out.push(LocalPathPoint::Line(0.0, 0.0));
        out.push(LocalPathPoint::Line(width, 0.0));
        return out;
    }

    let cp_tl = tl * KAPPA;
    let cp_tr = tr * KAPPA;
    let cp_br = br * KAPPA;
    let cp_bl = bl * KAPPA;

    let top_edge = (width - tl - tr).max(0.0);
    let right_edge = (height - tr - br).max(0.0);
    let bottom_edge = (width - bl - br).max(0.0);

    out.push(LocalPathPoint::Move(width - tr, 0.0));
    out.push(LocalPathPoint::Curve {
        x: width,
        y: tr,
        cp1: (width - tr + cp_tr, 0.0),
        cp2: (width, tr - cp_tr),
    });
    if right_edge > 0.0 {
        out.push(LocalPathPoint::Line(width, height - br));
    }
    out.push(LocalPathPoint::Curve {
        x: width - br,
        y: height,
        cp1: (width, height - br + cp_br),
        cp2: (width - br + cp_br, height),
    });
    if bottom_edge > 0.0 {
        out.push(LocalPathPoint::Line(bl, height));
    }
    out.push(LocalPathPoint::Curve {
        x: 0.0,
        y: height - bl,
        cp1: (bl - cp_bl, height),
        cp2: (0.0, height - bl + cp_bl),
    });
    if (height - tl - bl).max(0.0) > 0.0 {
        out.push(LocalPathPoint::Line(0.0, tl));
    }
    out.push(LocalPathPoint::Curve {
        x: tl,
        y: 0.0,
        cp1: (0.0, tl - cp_tl),
        cp2: (tl - cp_tl, 0.0),
    });
    if top_edge > 0.0 {
        out.push(LocalPathPoint::Line(width - tr, 0.0));
    }
    out
}

fn apply_affine_to_local_path(pts: &[LocalPathPoint], m: Affine) -> Vec<LocalPathPoint> {
    let ap = |x: f64, y: f64| transform::apply_point((x, y), m);
    pts.iter()
        .map(|p| match p {
            LocalPathPoint::Move(x, y) => {
                let (x, y) = ap(*x, *y);
                LocalPathPoint::Move(x, y)
            }
            LocalPathPoint::Line(x, y) => {
                let (x, y) = ap(*x, *y);
                LocalPathPoint::Line(x, y)
            }
            LocalPathPoint::Curve { x, y, cp1, cp2 } => {
                let (x, y) = ap(*x, *y);
                LocalPathPoint::Curve {
                    x,
                    y,
                    cp1: ap(cp1.0, cp1.1),
                    cp2: ap(cp2.0, cp2.1),
                }
            }
        })
        .collect()
}

fn local_path_to_json(pts: &[LocalPathPoint]) -> Vec<Value> {
    pts.iter()
        .map(|p| match p {
            LocalPathPoint::Move(x, y) => json!({"type": "move", "x": x, "y": y}),
            LocalPathPoint::Line(x, y) => json!({"type": "line", "x": x, "y": y}),
            LocalPathPoint::Curve { x, y, cp1, cp2 } => json!({
                "type": "curve", "x": x, "y": y,
                "cp1": {"x": cp1.0, "y": cp1.1},
                "cp2": {"x": cp2.0, "y": cp2.1},
            }),
        })
        .collect()
}

/// Absolute canvas path points for an interpolated blend rectangle.
pub fn rectangle_to_path_json_points(
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    corner_radius: f64,
    rotation_degrees: f64,
) -> Vec<Value> {
    let local = rectangle_local_path_points(width, height, corner_radius);
    let m = transform::from_trs(Transform2D {
        x,
        y,
        rotation: rotation_degrees,
        scale_x: 1.0,
        scale_y: 1.0,
    });
    local_path_to_json(&apply_affine_to_local_path(&local, m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounded_rectangle_emits_curve_points() {
        let pts = rectangle_to_path_json_points(10.0, 20.0, 100.0, 80.0, 12.0, 0.0);
        assert!(pts
            .iter()
            .any(|p| p.get("type").and_then(Value::as_str) == Some("curve")));
    }

    #[test]
    fn sharp_rectangle_has_no_curves() {
        let pts = rectangle_to_path_json_points(0.0, 0.0, 50.0, 40.0, 0.0, 0.0);
        assert!(!pts
            .iter()
            .any(|p| p.get("type").and_then(Value::as_str) == Some("curve")));
    }
}
