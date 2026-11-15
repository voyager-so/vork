//! A shape node's outline as path points, node-local (origin top-left): the
//! rectangle, ellipse, star and polygon converters.
//!
//! Moved here from `moda-engine` (`shape_to_path.rs`, `tess/shape_paths.rs`,
//! which now delegate) so that moda-scene can read any shape's outline: the
//! live geometry effects (`live_effects.rs`) resolve a shape to a path at scene
//! resolution time, before any render lane sees it. The math is unchanged, op
//! for op, so every lane that drew these shapes before draws them the same.
//!
//! Ports:
//! - `utils/rectangleToPath.ts` (`rectangleToPathPoints`, KAPPA corners) and
//!   `utils/roundedRectRadii.ts` (`normalizeRoundedRectRadii`: CSS-like
//!   proportional scaling, the export / tessellation behaviour; the per-corner
//!   clamp without pairwise scaling belongs to the SDF fast path).
//! - `utils/ellipseToPath.ts` (`ellipseDimensionsToPathPoints`: quadrant
//!   beziers, wedges, rings, annular sectors).
//! - `utils/starToPath.ts` / `utils/polygonToPath.ts` with the vertices of
//!   `utils/geometryHelpers.ts` and `roundClosedVertexRing`
//!   (`utils/path/closedShapePath.ts`).

use crate::blend::BlendPathPoint;
use crate::path_geom::round_corners::round_path_corners;
use crate::types::RawNode;
use crate::value::Value;

/// `KAPPA` (`utils/path/constants.ts:9`).
pub const KAPPA: f64 = 0.5522847498;

/// Below this the fillets are hairlines that read as rendering noise rather than
/// as a rounded corner, so the shape stays sharp. Mirrors
/// `MIN_VISIBLE_CORNER_RADIUS` in `utils/path/closedShapePath.ts`.
pub const MIN_VISIBLE_CORNER_RADIUS: f64 = 0.1;

/// Most star points / polygon sides drawn. A deliberate divergence from the TS
/// for absurd inputs: an unclamped `numPoints` of 1e20 overflowed the capacity
/// (a panic = WASM `unreachable`, demoting the renderer) and 1e7 took minutes.
pub const MAX_SHAPE_VERTICES: usize = 4096;

type P = [f64; 2];

fn mv(x: f64, y: f64) -> BlendPathPoint {
    BlendPathPoint::Move { x, y }
}

fn ln(x: f64, y: f64) -> BlendPathPoint {
    BlendPathPoint::Line { x, y }
}

// --- rectangle ---

/// `expandCornerRadii` (roundedRectRadii.ts:14-25). Note the quirky shorthand:
/// `[tl, tr, br] → [tl, tr, br, tr]` and `bl` defaults to `tr`.
pub fn expand_corner_radii(corner_radius: Option<&Value>) -> [f64; 4] {
    match corner_radius {
        Some(Value::Number(n)) => {
            let v = n.as_f64().unwrap_or(0.0);
            [v, v, v, v]
        }
        Some(Value::Array(arr)) => {
            let g = |i: usize| arr.get(i).and_then(Value::as_f64);
            let tl = g(0).unwrap_or(0.0);
            let tr = g(1).unwrap_or(tl);
            let br = g(2).unwrap_or(tl);
            let bl = g(3).unwrap_or(tr);
            [tl, tr, br, bl]
        }
        _ => [0.0; 4],
    }
}

/// `normalizeRoundedRectRadii` (roundedRectRadii.ts:32-62).
pub fn normalize_rounded_rect_radii(
    width: f64,
    height: f64,
    corner_radius: Option<&Value>,
) -> [f64; 4] {
    let w = width.abs();
    let h = height.abs();
    if w <= 0.0 || h <= 0.0 {
        return [0.0; 4];
    }
    let expanded = expand_corner_radii(corner_radius);
    let max_r = (w / 2.0).min(h / 2.0);
    let clamped: Vec<f64> = expanded.iter().map(|r| r.clamp(0.0, max_r)).collect();
    let (tl, tr, br, bl) = (clamped[0], clamped[1], clamped[2], clamped[3]);
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

/// `rectangleToPathPoints` (rectangleToPath.ts) with x = y = 0, rotation 0.
/// A rectangle's outline with `cornerStyle` (live corners: `crate::rect_corners`) or
/// `cornerSmoothing` (E7.11): styled corners, else Figma's smoothed corners
/// (`crate::corner_smoothing`) when smoothing is on and a corner is rounded, else
/// [`rectangle_outline`]. The live geometry effects reshape what the body draws.
pub fn smoothed_or_plain_rectangle_outline(
    width: f64,
    height: f64,
    corner_radius: Option<&Value>,
    corner_smoothing: Option<&Value>,
    corner_style: Option<&Value>,
) -> Vec<BlendPathPoint> {
    use crate::corner_smoothing::{clamp_smoothing, is_smoothed, smoothed_rect_path, Seg};
    let radii = expand_corner_radii(corner_radius);
    let smoothing = clamp_smoothing(corner_smoothing.and_then(Value::as_f64).unwrap_or(0.0));
    let segs = if crate::rect_corners::is_styled_rect(corner_radius, corner_style) {
        crate::rect_corners::styled_rect_path(
            width,
            height,
            corner_radius,
            crate::rect_corners::rect_corner_style(corner_style),
        )
    } else if is_smoothed(radii, smoothing) {
        smoothed_rect_path(width, height, radii, smoothing)
    } else {
        return rectangle_outline(width, height, corner_radius);
    };
    segs.into_iter()
        .map(|s| match s {
            Seg::Move(x, y) => BlendPathPoint::Move { x, y },
            Seg::Line(x, y) => BlendPathPoint::Line { x, y },
            Seg::Cubic { cp1, cp2, to } => BlendPathPoint::Curve { x: to.0, y: to.1, cp1, cp2 },
        })
        .collect()
}

pub fn rectangle_outline(
    width: f64,
    height: f64,
    corner_radius: Option<&Value>,
) -> Vec<BlendPathPoint> {
    let radii = normalize_rounded_rect_radii(width, height, corner_radius);
    let [tl, tr, br, bl] = radii;
    let mut out = Vec::new();

    if radii.iter().all(|r| *r == 0.0) {
        // Simple rectangle: M top-right, L around, L back to start.
        out.push(mv(width, 0.0));
        out.push(ln(width, height));
        out.push(ln(0.0, height));
        out.push(ln(0.0, 0.0));
        out.push(ln(width, 0.0));
        return out;
    }

    let cp_tl = tl * KAPPA;
    let cp_tr = tr * KAPPA;
    let cp_br = br * KAPPA;
    let cp_bl = bl * KAPPA;

    let top_edge = (width - tl - tr).max(0.0);
    let right_edge = (height - tr - br).max(0.0);
    let bottom_edge = (width - bl - br).max(0.0);
    let left_edge = (height - tl - bl).max(0.0);

    // Start at top edge, right of top-right corner.
    out.push(mv(width - tr, 0.0));
    // Top-right corner.
    out.push(BlendPathPoint::Curve {
        cp1: (width - tr + cp_tr, 0.0),
        cp2: (width, tr - cp_tr),
        x: width,
        y: tr,
    });
    if right_edge > 0.0 {
        out.push(ln(width, height - br));
    }
    // Bottom-right corner.
    out.push(BlendPathPoint::Curve {
        cp1: (width, height - br + cp_br),
        cp2: (width - br + cp_br, height),
        x: width - br,
        y: height,
    });
    if bottom_edge > 0.0 {
        out.push(ln(bl, height));
    }
    // Bottom-left corner.
    out.push(BlendPathPoint::Curve {
        cp1: (bl - cp_bl, height),
        cp2: (0.0, height - bl + cp_bl),
        x: 0.0,
        y: height - bl,
    });
    if left_edge > 0.0 {
        out.push(ln(0.0, tl));
    }
    // Top-left corner.
    out.push(BlendPathPoint::Curve {
        cp1: (0.0, tl - cp_tl),
        cp2: (tl - cp_tl, 0.0),
        x: tl,
        y: 0.0,
    });
    if top_edge > 0.0 {
        out.push(ln(width - tr, 0.0));
    }
    out
}

// --- ellipse (ellipseToPath.ts) ---

const MAX_SWEEP_DEGREES: f64 = 360.0;

fn normalize_sweep_angle(sweep: Option<f64>) -> f64 {
    match sweep {
        Some(v) if v.is_finite() => {
            if v.abs() >= MAX_SWEEP_DEGREES {
                MAX_SWEEP_DEGREES * if v < 0.0 { -1.0 } else { 1.0 }
            } else {
                v
            }
        }
        _ => MAX_SWEEP_DEGREES,
    }
}

fn point_on_ellipse(rx: f64, ry: f64, angle_deg: f64) -> (f64, f64) {
    let a = angle_deg.to_radians();
    (rx * a.cos(), ry * a.sin())
}

fn arc_segment_to_bezier(rx: f64, ry: f64, start: f64, end: f64) -> BlendPathPoint {
    let s = start.to_radians();
    let e = end.to_radians();
    let delta = e - s;
    let k = (4.0 / 3.0) * (delta / 4.0).tan();
    let (cs, ss) = (s.cos(), s.sin());
    let (ce, se) = (e.cos(), e.sin());
    let (x0, y0) = (rx * cs, ry * ss);
    let (x1, y1) = (rx * ce, ry * se);
    BlendPathPoint::Curve {
        cp1: (x0 + -rx * ss * k, y0 + ry * cs * k),
        cp2: (x1 + rx * se * k, y1 + -ry * ce * k),
        x: x1,
        y: y1,
    }
}

fn build_arc_segments(rx: f64, ry: f64, start: f64, sweep: f64) -> Vec<BlendPathPoint> {
    if sweep == 0.0 {
        return Vec::new();
    }
    let num = (sweep.abs() / 90.0).ceil() as usize;
    let seg = sweep / num as f64;
    (0..num)
        .map(|i| {
            arc_segment_to_bezier(rx, ry, start + seg * i as f64, start + seg * (i + 1) as f64)
        })
        .collect()
}

fn build_arc_path(rx: f64, ry: f64, start: f64, sweep: f64) -> Vec<BlendPathPoint> {
    let (sx, sy) = point_on_ellipse(rx, ry, start);
    let mut out = vec![mv(sx, sy)];
    out.extend(build_arc_segments(rx, ry, start, sweep));
    out
}

/// `ellipseDimensionsToPathPoints` (ellipseToPath.ts:212-227): node-local
/// (top-left origin), center (w/2, h/2), radii (w/2, h/2), rotation 0.
pub fn ellipse_outline(
    width: f64,
    height: f64,
    start_angle: Option<f64>,
    sweep_angle: Option<f64>,
    inner_radius_ratio: Option<f64>,
) -> Vec<BlendPathPoint> {
    let rx = width / 2.0;
    let ry = height / 2.0;
    let start = start_angle.filter(|v| v.is_finite()).unwrap_or(0.0);
    let sweep = normalize_sweep_angle(sweep_angle);
    let inner = inner_radius_ratio
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(0.0);

    let has_inner = inner > 0.0;
    let full = sweep.abs() >= MAX_SWEEP_DEGREES;

    let local: Vec<BlendPathPoint> = if full && !has_inner {
        build_arc_path(rx, ry, start, sweep)
    } else if full && has_inner {
        // Ring: outer contour + opposite-wound inner contour.
        let mut p = build_arc_path(rx, ry, start, sweep);
        p.extend(build_arc_path(rx * inner, ry * inner, start, -sweep));
        p
    } else if !has_inner {
        // Wedge.
        let (sx, sy) = point_on_ellipse(rx, ry, start);
        let mut p = vec![mv(0.0, 0.0), ln(sx, sy)];
        p.extend(build_arc_segments(rx, ry, start, sweep));
        p.push(ln(0.0, 0.0));
        p
    } else {
        // Annular sector.
        let irx = rx * inner;
        let iry = ry * inner;
        let end = start + sweep;
        let (osx, osy) = point_on_ellipse(rx, ry, start);
        let (iex, iey) = point_on_ellipse(irx, iry, end);
        let mut p = vec![mv(osx, osy)];
        p.extend(build_arc_segments(rx, ry, start, sweep));
        p.push(ln(iex, iey));
        p.extend(build_arc_segments(irx, iry, end, -sweep));
        p.push(ln(osx, osy));
        p
    };

    // applyShapeTransform({x: w/2, y: h/2, rotationDegrees: 0}): a pure
    // translation (cos 0 = 1 exactly).
    translate(local, rx, ry)
}

// --- star / polygon ---

/// `getEllipseVertexAtAngle` (geometryHelpers.ts:39-42): rotate (1,0) by
/// `angleDegrees`, then scale by (rx, ry).
pub fn ellipse_vertex_at_angle(angle_degrees: f64, rx: f64, ry: f64) -> P {
    if angle_degrees == 0.0 {
        return [rx, 0.0];
    }
    let rad = angle_degrees * std::f64::consts::PI / 180.0;
    let (cos, sin) = (rad.cos(), rad.sin());
    [rx * cos, ry * sin]
}

/// `getStretchedStarVertices` (geometryHelpers.ts:105-123). Outer/inner
/// alternate, starting at 12 o'clock (angle −90°), clockwise.
fn stretched_star_vertices(
    num_points: f64,
    outer_rx: f64,
    outer_ry: f64,
    inner_rx: f64,
    inner_ry: f64,
) -> Vec<P> {
    if !num_points.is_finite() {
        return Vec::new(); // JS Math.max(3, NaN) = NaN → zero loop iterations
    }
    let n = num_points.floor().clamp(3.0, MAX_SHAPE_VERTICES as f64) as usize;
    let angle_step = 180.0 / n as f64;
    let mut vertices = Vec::with_capacity(n * 2);
    for i in 0..n * 2 {
        let is_outer = i % 2 == 0;
        let rx = if is_outer { outer_rx } else { inner_rx };
        let ry = if is_outer { outer_ry } else { inner_ry };
        let angle = i as f64 * angle_step - 90.0;
        vertices.push(ellipse_vertex_at_angle(angle, rx, ry));
    }
    vertices
}

/// `getStretchedPolygonVertices` (geometryHelpers.ts:85-94).
fn stretched_polygon_vertices(sides: f64, rx: f64, ry: f64) -> Vec<P> {
    if !sides.is_finite() {
        return Vec::new();
    }
    let n = sides.floor().clamp(3.0, MAX_SHAPE_VERTICES as f64) as usize;
    let angle_step = 360.0 / n as f64;
    let mut vertices = Vec::with_capacity(n);
    for i in 0..n {
        let angle = i as f64 * angle_step - 90.0;
        vertices.push(ellipse_vertex_at_angle(angle, rx, ry));
    }
    vertices
}

/// `createSimpleStarPath` / `createSimplePolygonPath`: M, L…, closing L back
/// to the first vertex.
fn simple_ring(vertices: &[P]) -> Vec<BlendPathPoint> {
    let mut out = Vec::with_capacity(vertices.len() + 1);
    out.push(mv(vertices[0][0], vertices[0][1]));
    for v in vertices.iter().skip(1) {
        out.push(ln(v[0], v[1]));
    }
    out.push(ln(vertices[0][0], vertices[0][1]));
    out
}

/// `roundClosedVertexRing` (`utils/path/closedShapePath.ts`): round every corner
/// of a closed vertex ring through the shared rounder. Radii below
/// `MIN_VISIBLE_CORNER_RADIUS` stay sharp.
pub fn rounded_ring(vertices: &[P], corner_radius: f64) -> Vec<BlendPathPoint> {
    if !(corner_radius >= MIN_VISIBLE_CORNER_RADIUS) {
        return simple_ring(vertices);
    }
    let ring: Vec<BlendPathPoint> = vertices
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if i == 0 {
                mv(v[0], v[1])
            } else {
                ln(v[0], v[1])
            }
        })
        .collect();
    let rounded = round_path_corners(&ring, true, corner_radius);
    // The rounder hands back its input verbatim when no vertex actually turns (a
    // ring collapsed to a point), and that array is not closed. No fillet was
    // emitted means no `Curve`, so use that as the signal and emit the sharp ring.
    if !rounded
        .iter()
        .any(|p| matches!(p, BlendPathPoint::Curve { .. }))
    {
        return simple_ring(vertices);
    }
    rounded
}

/// `convertStarToPathNode` geometry (`starToPathPoints` with x = w/2,
/// y = h/2, rotation 0). `inner_radius_ratio` non-finite → 0.5 (the TS
/// `Number.isFinite` guard); clamped [0,1]. `corner_radius` is the TS
/// `star.cornerRadius || 0` value.
pub fn star_outline(
    width: f64,
    height: f64,
    num_points: f64,
    inner_radius_ratio: Option<f64>,
    corner_radius: f64,
) -> Vec<BlendPathPoint> {
    let ratio = match inner_radius_ratio {
        Some(v) if v.is_finite() => v.max(0.0).min(1.0),
        _ => 0.5,
    };
    let rx = width / 2.0;
    let ry = height / 2.0;
    let vertices = stretched_star_vertices(num_points, rx, ry, rx * ratio, ry * ratio);
    if vertices.is_empty() {
        return Vec::new();
    }
    translate(rounded_ring(&vertices, corner_radius), rx, ry)
}

/// `convertPolygonToPathNode` geometry (`polygonToPathPoints` with center
/// (w/2, h/2), rotation 0). Empty vertices (non-finite sides) → empty path.
pub fn polygon_outline(
    width: f64,
    height: f64,
    sides: f64,
    corner_radius: f64,
) -> Vec<BlendPathPoint> {
    let rx = width / 2.0;
    let ry = height / 2.0;
    let vertices = stretched_polygon_vertices(sides, rx, ry);
    if vertices.is_empty() {
        return Vec::new();
    }
    translate(rounded_ring(&vertices, corner_radius), rx, ry)
}

/// Translate every anchor and handle (`applyShapeTransform` with rotation 0 is an
/// exact translation). Arcs keep their radii.
pub fn translate(points: Vec<BlendPathPoint>, dx: f64, dy: f64) -> Vec<BlendPathPoint> {
    points
        .into_iter()
        .map(|c| match c {
            BlendPathPoint::Move { x, y } => mv(x + dx, y + dy),
            BlendPathPoint::Line { x, y } => ln(x + dx, y + dy),
            BlendPathPoint::Curve { x, y, cp1, cp2 } => BlendPathPoint::Curve {
                x: x + dx,
                y: y + dy,
                cp1: (cp1.0 + dx, cp1.1 + dy),
                cp2: (cp2.0 + dx, cp2.1 + dy),
            },
            BlendPathPoint::Quadratic { x, y, cp } => BlendPathPoint::Quadratic {
                x: x + dx,
                y: y + dy,
                cp: (cp.0 + dx, cp.1 + dy),
            },
            BlendPathPoint::Arc {
                x,
                y,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            } => BlendPathPoint::Arc {
                x: x + dx,
                y: y + dy,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            },
        })
        .collect()
}

/// A shape node's body outline, node-local, with its `closed` flag: what the
/// engine's body lane converts it to (`build_body_item`). `path` nodes return
/// their authored `points` (corner rounding included), `None` for any other
/// type (lines, text, groups, containers, …).
pub fn node_outline(node: &RawNode) -> Option<(Vec<BlendPathPoint>, bool)> {
    let w = node.f64("width").unwrap_or(0.0);
    let h = node.f64("height").unwrap_or(0.0);
    // `x.cornerRadius || 0` (JS falsy: 0/NaN/missing → 0).
    let cr = || {
        node.f64("cornerRadius")
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
    };
    Some(match node.node_type() {
        "rectangle" => (
            smoothed_or_plain_rectangle_outline(
                w,
                h,
                node.json_field("cornerRadius"),
                node.json_field("cornerSmoothing"),
                node.json_field("cornerStyle"),
            ),
            true,
        ),
        "ellipse" => (
            ellipse_outline(
                w,
                h,
                node.f64("startAngle"),
                node.f64("sweepAngle"),
                node.f64("innerRadiusRatio"),
            ),
            true,
        ),
        "star" => (
            star_outline(
                w,
                h,
                node.f64("numPoints").unwrap_or(f64::NAN),
                node.f64("innerRadiusRatio"),
                cr(),
            ),
            true,
        ),
        "polygon" => (
            polygon_outline(w, h, node.f64("sides").unwrap_or(f64::NAN), cr()),
            true,
        ),
        "path" => {
            let points: Vec<BlendPathPoint> = node
                .points()
                .map(|a| a.iter().filter_map(BlendPathPoint::from_ref).collect())
                .unwrap_or_default();
            let closed = node.bool("closed").unwrap_or(false);
            let corners = crate::path_geom::round_corners::PathCorners::read(node);
            let points = if corners.rounds_anything() {
                crate::path_geom::round_corners::round_path_corners_with(&points, closed, &corners)
            } else {
                points
            };
            (points, closed)
        }
        _ => return None,
    })
}
