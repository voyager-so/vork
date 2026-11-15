//! Path points: the PathNode JSON `points` shape, and flattening to a polyline.

use crate::transform;
use crate::value::{json, Value};

// Path points (PathNode JSON `points` ↔ internal repr) + flatten/resample
// ---------------------------------------------------------------------------

/// PathNode point (`core/scene/nodes.ts` PathPoint), JSON-backed.
#[derive(Debug, Clone)]
pub enum BlendPathPoint {
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
    Quadratic {
        x: f64,
        y: f64,
        cp: (f64, f64),
    },
    Arc {
        x: f64,
        y: f64,
        rx: f64,
        ry: f64,
        x_axis_rotation: f64,
        large_arc: bool,
        sweep: bool,
    },
}

impl BlendPathPoint {
    pub fn from_json(v: &Value) -> Option<BlendPathPoint> {
        Self::from_ref(crate::path_points::PathPointRef::Json(v))
    }

    /// A node point ([`crate::types::RawNode::points`]); exactly [`Self::from_json`] of its JSON.
    pub fn from_ref(p: crate::path_points::PathPointRef<'_>) -> Option<BlendPathPoint> {
        if !p.is_object() {
            return None;
        }
        let x = p.f64("x")?;
        let y = p.f64("y")?;
        // A control point: `None` unless it is an object with numeric `x` and `y` (`c.get("x")?.as_f64()?`).
        let cp = |k: &str| -> Option<(f64, f64)> {
            match p.xy(k)? {
                (Some(x), Some(y)) => Some((x, y)),
                _ => None,
            }
        };
        Some(match p.kind()? {
            "move" => BlendPathPoint::Move { x, y },
            "line" => BlendPathPoint::Line { x, y },
            "curve" => BlendPathPoint::Curve {
                x,
                y,
                cp1: cp("cp1")?,
                cp2: cp("cp2")?,
            },
            "quadratic" => BlendPathPoint::Quadratic { x, y, cp: cp("cp")? },
            "arc" => BlendPathPoint::Arc {
                x,
                y,
                rx: p.f64("rx").unwrap_or(0.0),
                ry: p.f64("ry").unwrap_or(0.0),
                x_axis_rotation: p.f64("xAxisRotation").unwrap_or(0.0),
                large_arc: p.bool("largeArc").unwrap_or(false),
                sweep: p.bool("sweep").unwrap_or(false),
            },
            _ => return None,
        })
    }

    pub fn to_json(&self) -> Value {
        match self {
            BlendPathPoint::Move { x, y } => json!({"type": "move", "x": x, "y": y}),
            BlendPathPoint::Line { x, y } => json!({"type": "line", "x": x, "y": y}),
            BlendPathPoint::Curve { x, y, cp1, cp2 } => json!({
                "type": "curve", "x": x, "y": y,
                "cp1": {"x": cp1.0, "y": cp1.1},
                "cp2": {"x": cp2.0, "y": cp2.1},
            }),
            BlendPathPoint::Quadratic { x, y, cp } => json!({
                "type": "quadratic", "x": x, "y": y,
                "cp": {"x": cp.0, "y": cp.1},
            }),
            BlendPathPoint::Arc {
                x,
                y,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            } => json!({
                "type": "arc", "x": x, "y": y, "rx": rx, "ry": ry,
                "xAxisRotation": x_axis_rotation, "largeArc": large_arc, "sweep": sweep,
            }),
        }
    }

    /// Apply an affine to anchors AND control points
    /// (`utils/pathTransformHelpers.transformPathPointsWithAffine`).
    pub fn apply_affine(&self, m: crate::types::Affine) -> BlendPathPoint {
        let ap = |x: f64, y: f64| transform::apply_point((x, y), m);
        match self {
            BlendPathPoint::Move { x, y } => {
                let (x, y) = ap(*x, *y);
                BlendPathPoint::Move { x, y }
            }
            BlendPathPoint::Line { x, y } => {
                let (x, y) = ap(*x, *y);
                BlendPathPoint::Line { x, y }
            }
            BlendPathPoint::Curve { x, y, cp1, cp2 } => {
                let (x, y) = ap(*x, *y);
                BlendPathPoint::Curve {
                    x,
                    y,
                    cp1: ap(cp1.0, cp1.1),
                    cp2: ap(cp2.0, cp2.1),
                }
            }
            BlendPathPoint::Quadratic { x, y, cp } => {
                let (x, y) = ap(*x, *y);
                BlendPathPoint::Quadratic {
                    x,
                    y,
                    cp: ap(cp.0, cp.1),
                }
            }
            BlendPathPoint::Arc {
                x,
                y,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            } => {
                // Production transformPathPoints translates arc anchors only
                // (radii untouched) — blends never carry arcs in practice.
                let (x, y) = ap(*x, *y);
                BlendPathPoint::Arc {
                    x,
                    y,
                    rx: *rx,
                    ry: *ry,
                    x_axis_rotation: *x_axis_rotation,
                    large_arc: *large_arc,
                    sweep: *sweep,
                }
            }
        }
    }

    /// Translate by (dx, dy) — anchors and control points
    /// (blendUtils.ts:993-1017 absolute-point mapping).
    pub fn offset(&self, dx: f64, dy: f64) -> BlendPathPoint {
        self.apply_affine(transform::translate(dx, dy))
    }
}

/// `arcEndpointToCenter` (`core/svg/arcUtils.ts:36-148`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn arc_endpoint_to_center(
    x1: f64,
    y1: f64,
    rx_in: f64,
    ry_in: f64,
    x_axis_rotation: f64,
    large_arc: bool,
    sweep: bool,
    x2: f64,
    y2: f64,
) -> Option<(f64, f64, f64, f64, f64, f64, f64)> {
    let phi = x_axis_rotation * std::f64::consts::PI / 180.0;
    if x1 == x2 && y1 == y2 {
        return None;
    }
    if rx_in == 0.0 || ry_in == 0.0 {
        return None;
    }
    let mut rx = rx_in.abs();
    let mut ry = ry_in.abs();

    let (cos_phi, sin_phi) = (phi.cos(), phi.sin());
    let dx = (x1 - x2) / 2.0;
    let dy = (y1 - y2) / 2.0;
    let x1p = cos_phi * dx + sin_phi * dy;
    let y1p = -sin_phi * dx + cos_phi * dy;

    let lambda = x1p * x1p / (rx * rx) + y1p * y1p / (ry * ry);
    if lambda > 1.0 {
        let s = lambda.sqrt();
        rx *= s;
        ry *= s;
    }
    let rx_sq = rx * rx;
    let ry_sq = ry * ry;
    let mut sq = (rx_sq * ry_sq - rx_sq * y1p * y1p - ry_sq * x1p * x1p)
        / (rx_sq * y1p * y1p + ry_sq * x1p * x1p);
    sq = sq.max(0.0);
    let coef = if large_arc != sweep { 1.0 } else { -1.0 } * sq.sqrt();
    let cxp = coef * (rx * y1p / ry);
    let cyp = coef * (-(ry * x1p) / rx);
    let cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0;
    let cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0;

    let ux = (x1p - cxp) / rx;
    let uy = (y1p - cyp) / ry;
    let vx = (-x1p - cxp) / rx;
    let vy = (-y1p - cyp) / ry;
    let theta1 = uy.atan2(ux);
    let n2 = ((ux * ux + uy * uy) * (vx * vx + vy * vy)).sqrt();
    let dot = ux * vx + uy * vy;
    let cross = ux * vy - uy * vx;
    let mut dtheta = if cross >= 0.0 { 1.0 } else { -1.0 } * (dot / n2).clamp(-1.0, 1.0).acos();
    if !sweep && dtheta > 0.0 {
        dtheta -= 2.0 * std::f64::consts::PI;
    } else if sweep && dtheta < 0.0 {
        dtheta += 2.0 * std::f64::consts::PI;
    }
    Some((cx, cy, rx, ry, theta1, dtheta, phi))
}

const DEFAULT_CURVE_SEGMENTS: usize = 16;

/// `flattenPathToPolyline` (`core/utils/pathSampling.ts:11-103`).
pub fn flatten_path_to_polyline(points: &[BlendPathPoint], closed: bool) -> Vec<(f64, f64)> {
    let mut result: Vec<(f64, f64)> = Vec::new();
    let mut current: Option<(f64, f64)> = None;
    let n = DEFAULT_CURVE_SEGMENTS as f64;

    for p in points {
        match p {
            BlendPathPoint::Move { x, y } => {
                current = Some((*x, *y));
                if result.is_empty() {
                    result.push((*x, *y));
                }
            }
            BlendPathPoint::Line { x, y } => match current {
                None => {
                    current = Some((*x, *y));
                    result.push((*x, *y));
                }
                Some(_) => {
                    result.push((*x, *y));
                    current = Some((*x, *y));
                }
            },
            BlendPathPoint::Curve { x, y, cp1, cp2 } => match current {
                None => {
                    current = Some((*x, *y));
                    result.push((*x, *y));
                }
                Some(c) => {
                    for s in 1..=DEFAULT_CURVE_SEGMENTS {
                        let t = s as f64 / n;
                        let it = 1.0 - t;
                        let px = it * it * it * c.0
                            + 3.0 * it * it * t * cp1.0
                            + 3.0 * it * t * t * cp2.0
                            + t * t * t * x;
                        let py = it * it * it * c.1
                            + 3.0 * it * it * t * cp1.1
                            + 3.0 * it * t * t * cp2.1
                            + t * t * t * y;
                        result.push((px, py));
                    }
                    current = Some((*x, *y));
                }
            },
            BlendPathPoint::Quadratic { x, y, cp } => match current {
                None => {
                    current = Some((*x, *y));
                    result.push((*x, *y));
                }
                Some(c) => {
                    for s in 1..=DEFAULT_CURVE_SEGMENTS {
                        let t = s as f64 / n;
                        let it = 1.0 - t;
                        let px = it * it * c.0 + 2.0 * it * t * cp.0 + t * t * x;
                        let py = it * it * c.1 + 2.0 * it * t * cp.1 + t * t * y;
                        result.push((px, py));
                    }
                    current = Some((*x, *y));
                }
            },
            BlendPathPoint::Arc {
                x,
                y,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            } => match current {
                None => {
                    current = Some((*x, *y));
                    result.push((*x, *y));
                }
                Some(c) => {
                    // tessellateArc (arcUtils.ts:149-189), 8 segs/quadrant.
                    match arc_endpoint_to_center(
                        c.0,
                        c.1,
                        *rx,
                        *ry,
                        *x_axis_rotation,
                        *large_arc,
                        *sweep,
                        *x,
                        *y,
                    ) {
                        None => result.push((*x, *y)),
                        Some((cx, cy, rx, ry, theta1, dtheta, phi)) => {
                            let num_segments = ((dtheta.abs() / std::f64::consts::FRAC_PI_2) * 8.0)
                                .ceil()
                                .max(1.0) as usize;
                            let (cos_phi, sin_phi) = (phi.cos(), phi.sin());
                            for i in 1..=num_segments {
                                let theta = theta1 + dtheta * i as f64 / num_segments as f64;
                                let px = rx * theta.cos();
                                let py = ry * theta.sin();
                                result.push((
                                    cos_phi * px - sin_phi * py + cx,
                                    sin_phi * px + cos_phi * py + cy,
                                ));
                            }
                        }
                    }
                    current = Some((*x, *y));
                }
            },
        }
    }

    if closed && result.len() > 1 {
        let first = result[0];
        let last = result[result.len() - 1];
        if first != last {
            result.push(first);
        }
    }
    result
}
