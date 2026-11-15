//! M2 — faithful port of `core/svg/arcUtils.ts` (W3C SVG endpoint→center
//! parameterization + arc→cubic conversion). Line citations are to that file.
//! COPY THE BUGS: degenerate conditions, clamps, and segment counts must match
//! the TS exactly — the vertex oracle diffs every flattened coordinate.

/// `ArcCenterParams` (arcUtils.ts:15-23).
pub struct ArcCenterParams {
    pub cx: f64,
    pub cy: f64,
    pub rx: f64,
    pub ry: f64,
    pub theta1: f64,
    pub dtheta: f64,
    pub phi: f64,
}

/// Arc command parameters as stored on a path point.
#[derive(Debug, Clone, Copy)]
pub struct ArcParams {
    pub rx: f64,
    pub ry: f64,
    pub x_axis_rotation: f64,
    pub large_arc: bool,
    pub sweep: bool,
    pub x: f64,
    pub y: f64,
}

/// `arcEndpointToCenter` (arcUtils.ts:36-138). Returns None for degenerate
/// arcs (identical endpoints, zero radius).
pub fn arc_endpoint_to_center(x1: f64, y1: f64, arc: &ArcParams) -> Option<ArcCenterParams> {
    let mut rx = arc.rx;
    let mut ry = arc.ry;
    let (x2, y2) = (arc.x, arc.y);

    let phi = arc.x_axis_rotation * std::f64::consts::PI / 180.0;

    // Identical endpoints → skip (strict equality, :44).
    if x1 == x2 && y1 == y2 {
        return None;
    }
    // Zero radius → line (:49).
    if rx == 0.0 || ry == 0.0 {
        return None;
    }

    rx = rx.abs();
    ry = ry.abs();

    let cos_phi = phi.cos();
    let sin_phi = phi.sin();

    let dx = (x1 - x2) / 2.0;
    let dy = (y1 - y2) / 2.0;

    let x1p = cos_phi * dx + sin_phi * dy;
    let y1p = -sin_phi * dx + cos_phi * dy;

    let x1p_sq = x1p * x1p;
    let y1p_sq = y1p * y1p;
    let rx_sq = rx * rx;
    let ry_sq = ry * ry;

    // Radii correction (:75-81).
    let lambda = x1p_sq / rx_sq + y1p_sq / ry_sq;
    if lambda > 1.0 {
        let sqrt_lambda = lambda.sqrt();
        rx = sqrt_lambda * rx;
        ry = sqrt_lambda * ry;
    }

    let rx_sq_new = rx * rx;
    let ry_sq_new = ry * ry;

    let mut sq = (rx_sq_new * ry_sq_new - rx_sq_new * y1p_sq - ry_sq_new * x1p_sq)
        / (rx_sq_new * y1p_sq + ry_sq_new * x1p_sq);
    sq = sq.max(0.0);

    let coef = (if arc.large_arc != arc.sweep {
        1.0
    } else {
        -1.0
    }) * sq.sqrt();

    let cxp = coef * ((rx * y1p) / ry);
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
    let mut dtheta = (if cross >= 0.0 { 1.0 } else { -1.0 }) * (dot / n2).clamp(-1.0, 1.0).acos();

    if !arc.sweep && dtheta > 0.0 {
        dtheta -= 2.0 * std::f64::consts::PI;
    } else if arc.sweep && dtheta < 0.0 {
        dtheta += 2.0 * std::f64::consts::PI;
    }

    Some(ArcCenterParams {
        cx,
        cy,
        rx,
        ry,
        theta1,
        dtheta,
        phi,
    })
}

/// One cubic from `arcToBeziers` output: `{cp1, cp2, end}`.
pub struct ArcBezier {
    pub cp1: [f64; 2],
    pub cp2: [f64; 2],
    pub end: [f64; 2],
}

/// `arcToBeziers` (arcUtils.ts:202-283). Empty Vec = degenerate (caller
/// treats as a line to the endpoint).
pub fn arc_to_beziers(x1: f64, y1: f64, arc: &ArcParams) -> Vec<ArcBezier> {
    let Some(center) = arc_endpoint_to_center(x1, y1, arc) else {
        return Vec::new();
    };
    let ArcCenterParams {
        cx,
        cy,
        rx,
        ry,
        theta1,
        dtheta,
        phi,
    } = center;

    // ≤90° per cubic (:221).
    let num_curves = (dtheta.abs() / (std::f64::consts::PI / 2.0)).ceil();
    if num_curves == 0.0 {
        return Vec::new();
    }
    let num_curves_i = num_curves as usize;
    let segment_angle = dtheta / num_curves;

    let cos_phi = phi.cos();
    let sin_phi = phi.sin();
    let transform = |px: f64, py: f64| -> [f64; 2] {
        [
            cos_phi * px - sin_phi * py + cx,
            sin_phi * px + cos_phi * py + cy,
        ]
    };

    let mut curves = Vec::with_capacity(num_curves_i);
    let mut current_theta = theta1;

    for _ in 0..num_curves_i {
        let start_theta = current_theta;
        let end_theta = current_theta + segment_angle;

        // (4/3)·tan(θ/4) handle (:251).
        let alpha = (4.0 / 3.0) * (segment_angle / 4.0).tan();

        let cos_start = start_theta.cos();
        let sin_start = start_theta.sin();
        let cos_end = end_theta.cos();
        let sin_end = end_theta.sin();

        let p1x = rx * cos_start;
        let p1y = ry * sin_start;
        let p2x = rx * cos_end;
        let p2y = ry * sin_end;

        let cp1x = p1x + alpha * (-rx * sin_start);
        let cp1y = p1y + alpha * (ry * cos_start);
        let cp2x = p2x - alpha * (-rx * sin_end);
        let cp2y = p2y - alpha * (ry * cos_end);

        curves.push(ArcBezier {
            cp1: transform(cp1x, cp1y),
            cp2: transform(cp2x, cp2y),
            end: transform(p2x, p2y),
        });

        current_theta = end_theta;
    }

    curves
}
