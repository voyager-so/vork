//! M4b — connector-label geometry: text position / rotation / offsets along
//! a line, arc-length samplers, and the text-gap calculator.
//!
//! Faithful ports of (line citations to each file):
//! - `core/scene/connectorTextHelpers.ts`: `hasConnectorText` (:71-76),
//!   `buildLineSegments` (:97-121), `getTotalPathLength` (:126-128),
//!   `getCenterSegmentMidpointT` (:138-180), `getConnectorTextPosition`
//!   (:259-393) + straight/segmented samplers (:398-488),
//!   `applyTextOffsets` (:498-519), `getLineLength` (:533-550),
//!   `getConnectorTextPositionT` (:575-578) + `remapTextPositionForArrows`
//!   (:589-617), `MIN_LINE_LENGTH_FOR_TEXT`/`isLineLongEnoughForText`
//!   (:622-629), `calculateTextRotation` (:772-795),
//!   `getConnectorTextAnchorLocalX` (:1306-1310),
//!   `CONNECTOR_TEXT_WIDTH/HEIGHT` (:841-842)
//! - `utils/lineToPath.ts`: `getCurvedLineArcLength` (:2010-2016) — now a THIN
//!   DELEGATOR to `getElbowedLineArcLength` (:2028-2037) at
//!   `CURVED_LINE_CORNER_RADIUS`, `sampleElbowedLineAtArcLengthT` (:2049-2106),
//!   `convertPolylineDistanceToElbowedRenderedT` (:2119-2156),
//!   `sampleCurvedLineAtArcLengthT` (:2292-2300) — now delegates to the
//!   elbowed sampler, `computeCurvedLineBodyEndpoints` (:2327-2379) — now trims
//!   along the waypoint polyline direction (start→waypoints[0],
//!   waypoints[last]→end), not the deleted cubic control points cp1/cp2
//! - `renderers/webgpu/utils/WebGPULineToPath.ts`: `calculateLineTextGap`
//!   (:139-228)
//!
//! Coordinate spaces copy production exactly: position sampling runs on
//! CANVAS-ABSOLUTE coordinates (`getAbsoluteLineCoordinates` — node canvas
//! affine applied to endpoints/waypoints), EXCEPT the rounded-elbow sampler
//! which samples in LINE-LOCAL space and pushes the result through the
//! canvas affine (`getConnectorTextPosition` :335-386). The gap calculator
//! runs on the raw LINE-LOCAL `line.points` (production passes `line`
//! directly), which is identical for the unscaled/unrotated common case and
//! copies any divergence for transformed lines.

use super::dash::{
    build_elbowed_path_segments, find_t_at_arc_length_quadratic, quadratic_at, Pt, Seg,
};
use super::line_to_path::{
    arrow_trim_distance, LineSpec, Routing, TextGap, CURVED_LINE_CORNER_RADIUS,
};

/// `MIN_LINE_LENGTH_FOR_TEXT` (:622).
pub const MIN_LINE_LENGTH_FOR_TEXT: f64 = 40.0;
/// `CONNECTOR_TEXT_WIDTH` / `CONNECTOR_TEXT_HEIGHT` (:841-842).
pub const CONNECTOR_TEXT_WIDTH: f64 = 300.0;
pub const CONNECTOR_TEXT_HEIGHT: f64 = 50.0;

/// `ConnectorTextPosition` (:42-51) — canvas position + tangent angle (rad).
#[derive(Debug, Clone, Copy)]
pub struct ConnectorTextPosition {
    pub x: f64,
    pub y: f64,
    pub angle: f64,
}

struct LineSegment {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    length: f64,
}

/// `buildLineSegments` (:97-121).
fn build_line_segments(start: Pt, waypoints: &[Pt], end: Pt) -> Vec<LineSegment> {
    let mut pts: Vec<Pt> = Vec::with_capacity(waypoints.len() + 2);
    pts.push(start);
    pts.extend_from_slice(waypoints);
    pts.push(end);

    let mut segments = Vec::with_capacity(pts.len() - 1);
    for i in 0..pts.len() - 1 {
        let p1 = pts[i];
        let p2 = pts[i + 1];
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        segments.push(LineSegment {
            x1: p1[0],
            y1: p1[1],
            x2: p2[0],
            y2: p2[1],
            length: (dx * dx + dy * dy).sqrt(),
        });
    }
    segments
}

fn total_path_length(segments: &[LineSegment]) -> f64 {
    segments.iter().map(|s| s.length).sum()
}

fn arrow_trim_for_endpoint(spec: &LineSpec, start: bool) -> f64 {
    let shape = if start {
        spec.start_arrow
    } else {
        spec.end_arrow
    };
    match shape {
        // `arrowTrimDistance(shape, pointerLength ?? 10, pointerWidth ?? 10,
        //  { lineCap, strokeWidth })` (:26-33).
        Some(s) => arrow_trim_distance(
            s,
            spec.pointer_length.unwrap_or(10.0),
            spec.pointer_width.unwrap_or(10.0),
            spec.line_cap.as_deref(),
            spec.stroke_width.unwrap_or(0.0),
        ),
        None => 0.0,
    }
}

/// `getCurvedLineArcLength` (lineToPath.ts:2010-2016) — now a THIN DELEGATOR.
/// Curved lines render as the elbowed rounded-corner path, so the visible
/// length is the rendered elbowed arc length at `CURVED_LINE_CORNER_RADIUS`.
/// With empty waypoints the elbowed builder yields a single `[start, end]`
/// line segment, so this still returns the straight start→end distance.
pub fn get_curved_line_arc_length(start: Pt, end: Pt, waypoints: &[Pt]) -> f64 {
    get_elbowed_line_arc_length(start, end, waypoints, CURVED_LINE_CORNER_RADIUS)
}

/// `getElbowedLineArcLength` (:2230-2239).
pub fn get_elbowed_line_arc_length(
    start: Pt,
    end: Pt,
    waypoints: &[Pt],
    corner_radius: f64,
) -> f64 {
    let mut all_points: Vec<Pt> = Vec::with_capacity(waypoints.len() + 2);
    all_points.push(start);
    all_points.extend_from_slice(waypoints);
    all_points.push(end);
    build_elbowed_path_segments(&all_points, corner_radius)
        .iter()
        .map(|s| s.arc_length)
        .sum()
}

/// `getLineLength` (:533-550) — LINE-LOCAL `line.points` space, exactly like
/// production (which passes the node's own points).
pub fn get_line_length(spec: &LineSpec) -> f64 {
    if spec.points.len() < 4 {
        return 0.0;
    }
    let start = [spec.points[0], spec.points[1]];
    let end = [spec.points[2], spec.points[3]];
    if spec.routing == Routing::Curved {
        return get_curved_line_arc_length(start, end, &spec.waypoints);
    }
    if spec.routing == Routing::Elbowed
        && spec.corner_radius.unwrap_or(0.0) > 0.0
        && !spec.waypoints.is_empty()
    {
        return get_elbowed_line_arc_length(
            start,
            end,
            &spec.waypoints,
            spec.corner_radius.unwrap_or(0.0),
        );
    }
    total_path_length(&build_line_segments(start, &spec.waypoints, end))
}

/// `isLineLongEnoughForText` (:627-629).
pub fn is_line_long_enough_for_text(spec: &LineSpec) -> bool {
    get_line_length(spec) >= MIN_LINE_LENGTH_FOR_TEXT
}

/// `getCenterSegmentMidpointT` (:138-180).
fn get_center_segment_midpoint_t(spec: &LineSpec) -> f64 {
    if spec.routing != Routing::Elbowed || spec.waypoints.is_empty() {
        return 0.5;
    }
    let start = [spec.points[0], spec.points[1]];
    let end = [spec.points[2], spec.points[3]];
    let segments = build_line_segments(start, &spec.waypoints, end);
    if segments.is_empty() {
        return 0.5;
    }
    let center_segment_index = segments.len() / 2;
    let total_length = total_path_length(&segments);
    if total_length == 0.0 {
        return 0.5;
    }
    let mut distance_to_segment_start = 0.0;
    for seg in segments.iter().take(center_segment_index) {
        distance_to_segment_start += seg.length;
    }
    let polyline_distance_to_midpoint =
        distance_to_segment_start + segments[center_segment_index].length / 2.0;

    if spec.corner_radius.unwrap_or(0.0) > 0.0 {
        return convert_polyline_distance_to_elbowed_rendered_t(
            start,
            end,
            &spec.waypoints,
            spec.corner_radius.unwrap_or(0.0),
            polyline_distance_to_midpoint,
        );
    }
    polyline_distance_to_midpoint / total_length
}

/// `remapTextPositionForArrows` (:589-617).
fn remap_text_position_for_arrows(spec: &LineSpec, t: f64) -> f64 {
    let total_length = get_line_length(spec);
    if total_length < 0.001 {
        return t;
    }
    let start_arrow = spec.start_arrow.is_some();
    let end_arrow = spec.end_arrow.is_some();
    if !start_arrow && !end_arrow {
        return t;
    }
    let start_trim = arrow_trim_for_endpoint(spec, true);
    let end_trim = arrow_trim_for_endpoint(spec, false);
    let start_offset = if start_arrow {
        start_trim / total_length
    } else {
        0.0
    };
    let end_offset = if end_arrow {
        end_trim / total_length
    } else {
        0.0
    };

    let effective_start = start_offset;
    let effective_end = 1.0 - end_offset;
    let effective_range = effective_end - effective_start;
    if effective_range <= 0.001 {
        return (effective_start + effective_end) / 2.0;
    }
    effective_start + t * effective_range
}

/// `getConnectorTextPositionT` (:575-578). `text_position` is the node's
/// `textPosition` field (None ⇔ undefined).
pub fn get_connector_text_position_t(spec: &LineSpec, text_position: Option<f64>) -> f64 {
    let raw_t = text_position.unwrap_or_else(|| get_center_segment_midpoint_t(spec));
    remap_text_position_for_arrows(spec, raw_t)
}

/// `convertPolylineDistanceToElbowedRenderedT` (:2321-2358).
pub fn convert_polyline_distance_to_elbowed_rendered_t(
    start: Pt,
    end: Pt,
    waypoints: &[Pt],
    corner_radius: f64,
    polyline_dist: f64,
) -> f64 {
    let mut all_points: Vec<Pt> = Vec::with_capacity(waypoints.len() + 2);
    all_points.push(start);
    all_points.extend_from_slice(waypoints);
    all_points.push(end);
    let segments = build_elbowed_path_segments(&all_points, corner_radius);
    let rendered_total: f64 = segments.iter().map(|s| s.arc_length).sum();
    if rendered_total < 0.001 || segments.is_empty() {
        return 0.0;
    }
    if polyline_dist <= 0.0 {
        return 0.0;
    }

    let mut polyline_cursor = 0.0;
    let mut rendered_cursor = 0.0;
    for seg in &segments {
        let rendered_len = seg.arc_length;
        let polyline_len = match &seg.seg {
            Seg::Quadratic { p0, p1, cp } => {
                (cp[0] - p0[0]).hypot(cp[1] - p0[1]) + (p1[0] - cp[0]).hypot(p1[1] - cp[1])
            }
            _ => rendered_len,
        };

        if polyline_dist <= polyline_cursor + polyline_len {
            let offset = (polyline_dist - polyline_cursor).max(0.0);
            let frac = if polyline_len > 0.001 {
                offset / polyline_len
            } else {
                0.0
            };
            return (rendered_cursor + frac * rendered_len) / rendered_total;
        }
        polyline_cursor += polyline_len;
        rendered_cursor += rendered_len;
    }
    1.0
}

/// `sampleElbowedLineAtArcLengthT` (:2251-2308). (`segmentIndex` dropped —
/// no native consumer.)
pub fn sample_elbowed_line_at_arc_length_t(
    start: Pt,
    end: Pt,
    waypoints: &[Pt],
    corner_radius: f64,
    t: f64,
) -> ConnectorTextPosition {
    let clamped_t = t.clamp(0.0, 1.0);
    let mut all_points: Vec<Pt> = Vec::with_capacity(waypoints.len() + 2);
    all_points.push(start);
    all_points.extend_from_slice(waypoints);
    all_points.push(end);
    let segments = build_elbowed_path_segments(&all_points, corner_radius);
    let total: f64 = segments.iter().map(|s| s.arc_length).sum();

    if total < 0.001 || segments.is_empty() {
        return ConnectorTextPosition {
            x: start[0],
            y: start[1],
            angle: (end[1] - start[1]).atan2(end[0] - start[0]),
        };
    }

    let target = clamped_t * total;
    let mut cursor = 0.0;
    let n = segments.len();
    for (i, seg) in segments.iter().enumerate() {
        let seg_len = seg.arc_length;
        let seg_end = cursor + seg_len;
        if target <= seg_end + 1e-6 || i == n - 1 {
            let offset = (target - cursor).max(0.0);
            match &seg.seg {
                Seg::Line { p0, p1 } => {
                    let frac = if seg_len > 0.001 {
                        (offset / seg_len).min(1.0)
                    } else {
                        0.0
                    };
                    return ConnectorTextPosition {
                        x: p0[0] + frac * (p1[0] - p0[0]),
                        y: p0[1] + frac * (p1[1] - p0[1]),
                        angle: (p1[1] - p0[1]).atan2(p1[0] - p0[0]),
                    };
                }
                Seg::Quadratic { p0, p1, cp } => {
                    let param_t = find_t_at_arc_length_quadratic(*p0, *cp, *p1, offset, seg_len);
                    let pt = quadratic_at(*p0, *cp, *p1, param_t);
                    let mt = 1.0 - param_t;
                    let dx = 2.0 * mt * (cp[0] - p0[0]) + 2.0 * param_t * (p1[0] - cp[0]);
                    let dy = 2.0 * mt * (cp[1] - p0[1]) + 2.0 * param_t * (p1[1] - cp[1]);
                    let angle = if dx.hypot(dy) > 1e-6 {
                        dy.atan2(dx)
                    } else {
                        0.0
                    };
                    return ConnectorTextPosition {
                        x: pt[0],
                        y: pt[1],
                        angle,
                    };
                }
                Seg::Cubic { .. } => {} // build_elbowed_path_segments never emits cubics
            }
        }
        cursor = seg_end;
    }

    let last = segments.last().unwrap();
    let (p0, p1) = match &last.seg {
        Seg::Line { p0, p1 } => (*p0, *p1),
        Seg::Quadratic { p0, p1, .. } => (*p0, *p1),
        Seg::Cubic { p0, p1, .. } => (*p0, *p1),
    };
    ConnectorTextPosition {
        x: p1[0],
        y: p1[1],
        angle: (p1[1] - p0[1]).atan2(p1[0] - p0[0]),
    }
}

/// `sampleCurvedLineAtArcLengthT` (lineToPath.ts:2292-2300) — now a THIN
/// DELEGATOR. Curved lines render as the elbowed rounded-corner path, so we
/// delegate to `sample_elbowed_line_at_arc_length_t` at
/// `CURVED_LINE_CORNER_RADIUS`. (The elbowed sampler already drops the TS
/// `segmentIndex` field, which the curved `{ x, y, angle }` shape never had.)
pub fn sample_curved_line_at_arc_length_t(
    start: Pt,
    end: Pt,
    waypoints: &[Pt],
    t: f64,
) -> ConnectorTextPosition {
    sample_elbowed_line_at_arc_length_t(start, end, waypoints, CURVED_LINE_CORNER_RADIUS, t)
}

/// `computeCurvedLineBodyEndpoints` (lineToPath.ts:2327-2379). Curved lines now
/// follow the waypoint polyline (with rounded corners), so the first/last
/// segment directions are start→waypoints[0] and waypoints[last]→end — the same
/// tangents the renderer trims along — NOT the cubic control points cp1/cp2.
pub fn compute_curved_line_body_endpoints(
    start: Pt,
    end: Pt,
    waypoints: &[Pt],
    has_start_arrow: bool,
    has_end_arrow: bool,
    start_trim: f64,
    end_trim: f64,
) -> (Pt, Pt) {
    if waypoints.is_empty() || (!has_start_arrow && !has_end_arrow) {
        return (start, end);
    }
    let first_wp = waypoints[0];
    let last_wp = waypoints[waypoints.len() - 1];

    let mut adjusted_start = start;
    if has_start_arrow {
        let dx = first_wp[0] - start[0];
        let dy = first_wp[1] - start[1];
        let len = dx.hypot(dy);
        if len > 0.001 {
            adjusted_start = [
                start[0] + (dx / len) * start_trim,
                start[1] + (dy / len) * start_trim,
            ];
        }
    }
    let mut adjusted_end = end;
    if has_end_arrow {
        let dx = end[0] - last_wp[0];
        let dy = end[1] - last_wp[1];
        let len = dx.hypot(dy);
        if len > 0.001 {
            adjusted_end = [
                end[0] - (dx / len) * end_trim,
                end[1] - (dy / len) * end_trim,
            ];
        }
    }
    (adjusted_start, adjusted_end)
}

/// Canvas-absolute line geometry the position sampler runs on. The caller
/// (text_draw) computes these by applying the node canvas affine to the
/// endpoints and waypoints (`getAbsoluteLineCoordinates`, :190-248) and
/// supplies the LINE-LOCAL geometry too for the rounded-elbow branch.
pub struct AbsLineGeometry {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub waypoints: Vec<Pt>,
    /// `affineFromTRS`-shaped canvas affine [a, b, c, d, e, f] (column pairs
    /// like `core/scene/affineTransform.ts`): canvas = [a c e; b d f]·local.
    pub canvas_affine: Option<[f64; 6]>,
}

fn apply_affine_pt(p: Pt, m: &[f64; 6]) -> Pt {
    [
        m[0] * p[0] + m[2] * p[1] + m[4],
        m[1] * p[0] + m[3] * p[1] + m[5],
    ]
}

fn apply_affine_vec(v: Pt, m: &[f64; 6]) -> Pt {
    [m[0] * v[0] + m[2] * v[1], m[1] * v[0] + m[3] * v[1]]
}

/// `getStraightLineTextPosition` (:398-403).
fn straight_line_text_position(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    t: f64,
) -> ConnectorTextPosition {
    ConnectorTextPosition {
        x: x1 + (x2 - x1) * t,
        y: y1 + (y2 - y1) * t,
        angle: (y2 - y1).atan2(x2 - x1),
    }
}

/// `getSegmentAngle` (:464-488).
fn segment_angle(segments: &[LineSegment], index: usize) -> f64 {
    let seg = &segments[index];
    if seg.length > 0.0 {
        return (seg.y2 - seg.y1).atan2(seg.x2 - seg.x1);
    }
    if index < segments.len() - 1 {
        let next = &segments[index + 1];
        if next.length > 0.0 {
            return (next.y2 - next.y1).atan2(next.x2 - next.x1);
        }
    }
    if index > 0 {
        let prev = &segments[index - 1];
        if prev.length > 0.0 {
            return (prev.y2 - prev.y1).atan2(prev.x2 - prev.x1);
        }
    }
    0.0
}

/// `getSegmentedLineTextPosition` (:408-459).
fn segmented_line_text_position(segments: &[LineSegment], t: f64) -> ConnectorTextPosition {
    let total_length = total_path_length(segments);
    if total_length == 0.0 {
        let seg = segments.first();
        return ConnectorTextPosition {
            x: seg.map(|s| s.x1).unwrap_or(0.0),
            y: seg.map(|s| s.y1).unwrap_or(0.0),
            angle: 0.0,
        };
    }

    let target_distance = total_length * t;
    let mut accumulated = 0.0;
    for (i, seg) in segments.iter().enumerate() {
        if accumulated + seg.length >= target_distance || i == segments.len() - 1 {
            if seg.length == 0.0 {
                return ConnectorTextPosition {
                    x: seg.x1,
                    y: seg.y1,
                    angle: segment_angle(segments, i),
                };
            }
            let segment_t = (target_distance - accumulated) / seg.length;
            let clamped = segment_t.clamp(0.0, 1.0);
            return ConnectorTextPosition {
                x: seg.x1 + (seg.x2 - seg.x1) * clamped,
                y: seg.y1 + (seg.y2 - seg.y1) * clamped,
                angle: (seg.y2 - seg.y1).atan2(seg.x2 - seg.x1),
            };
        }
        accumulated += seg.length;
    }

    let last = segments.last().unwrap();
    ConnectorTextPosition {
        x: last.x2,
        y: last.y2,
        angle: (last.y2 - last.y1).atan2(last.x2 - last.x1),
    }
}

/// `getConnectorTextPosition` (:259-393). `abs` carries the canvas-absolute
/// endpoints/waypoints; `spec` carries the LINE-LOCAL geometry + styling for
/// the rounded-elbow local-space sampling and the curved-arrow body
/// adjustment.
pub fn get_connector_text_position(
    spec: &LineSpec,
    abs: &AbsLineGeometry,
    t: f64,
) -> ConnectorTextPosition {
    let clamped_t = t.clamp(0.0, 1.0);
    let (x1, y1, x2, y2) = (abs.x1, abs.y1, abs.x2, abs.y2);
    let waypoints = &abs.waypoints;

    if spec.routing == Routing::Straight || waypoints.is_empty() {
        return straight_line_text_position(x1, y1, x2, y2, clamped_t);
    }

    if spec.routing == Routing::Curved {
        let start = [x1, y1];
        let end = [x2, y2];
        let start_arrow = spec.start_arrow.is_some();
        let end_arrow = spec.end_arrow.is_some();

        if !start_arrow && !end_arrow {
            return sample_curved_line_at_arc_length_t(start, end, waypoints, clamped_t);
        }

        let start_trim = arrow_trim_for_endpoint(spec, true);
        let end_trim = arrow_trim_for_endpoint(spec, false);
        let (adjusted_start, adjusted_end) = compute_curved_line_body_endpoints(
            start,
            end,
            waypoints,
            start_arrow,
            end_arrow,
            start_trim,
            end_trim,
        );

        let full_arc_length = get_curved_line_arc_length(start, end, waypoints);
        let body_arc_length = get_curved_line_arc_length(adjusted_start, adjusted_end, waypoints);
        if body_arc_length < 0.001 || full_arc_length < 0.001 {
            return ConnectorTextPosition {
                x: adjusted_start[0],
                y: adjusted_start[1],
                angle: 0.0,
            };
        }

        let start_arrow_arc = if start_arrow { start_trim } else { 0.0 };
        let target_full_arc = clamped_t * full_arc_length;
        let body_frac = ((target_full_arc - start_arrow_arc) / body_arc_length).clamp(0.0, 1.0);

        return sample_curved_line_at_arc_length_t(
            adjusted_start,
            adjusted_end,
            waypoints,
            body_frac,
        );
    }

    // Elbowed + rounded corners: sample in LINE-LOCAL space, then push the
    // result (and tangent) through the canvas affine (:350-386).
    if spec.routing == Routing::Elbowed
        && spec.corner_radius.unwrap_or(0.0) > 0.0
        && !waypoints.is_empty()
    {
        let local_start = [spec.points[0], spec.points[1]];
        let local_end = [spec.points[2], spec.points[3]];
        let local_sample = sample_elbowed_line_at_arc_length_t(
            local_start,
            local_end,
            &spec.waypoints,
            spec.corner_radius.unwrap_or(0.0),
            clamped_t,
        );
        if let Some(m) = &abs.canvas_affine {
            let canvas_point = apply_affine_pt([local_sample.x, local_sample.y], m);
            let local_tangent = [local_sample.angle.cos(), local_sample.angle.sin()];
            let canvas_tangent = apply_affine_vec(local_tangent, m);
            let canvas_angle = if canvas_tangent[0].hypot(canvas_tangent[1]) > 1e-9 {
                canvas_tangent[1].atan2(canvas_tangent[0])
            } else {
                local_sample.angle
            };
            return ConnectorTextPosition {
                x: canvas_point[0],
                y: canvas_point[1],
                angle: canvas_angle,
            };
        }
        return local_sample;
    }

    // Elbowed without corner radius: walk the canvas-space polyline.
    let segments = build_line_segments([x1, y1], waypoints, [x2, y2]);
    segmented_line_text_position(&segments, clamped_t)
}

/// `applyTextOffsets` (:498-519): offsetX is perpendicular (-90° of the
/// tangent so positive moves "up"), offsetY is parallel.
pub fn apply_text_offsets(pos: &ConnectorTextPosition, offset_x: f64, offset_y: f64) -> (f64, f64) {
    // affineFromTRS(rotation deg) of the angle, applied to (1,0) and (0,-1)
    // — equivalent to direct trig on the angle.
    let (sin, cos) = pos.angle.sin_cos();
    let parallel = [cos, sin];
    let perpendicular = [sin, -cos];
    (
        pos.x + perpendicular[0] * offset_x + parallel[0] * offset_y,
        pos.y + perpendicular[1] * offset_x + parallel[1] * offset_y,
    )
}

/// `calculateTextRotation` (:772-795). `text_rotation` is the raw node
/// string; unrecognized/missing values mean 'none'.
pub fn calculate_text_rotation(angle: f64, text_rotation: Option<&str>) -> f64 {
    let mode = text_rotation.unwrap_or("none");
    if mode != "follow" && mode != "perpendicular" {
        return 0.0;
    }
    let mut rotation = angle;
    if mode == "perpendicular" {
        rotation = angle + std::f64::consts::FRAC_PI_2;
    }
    if rotation > std::f64::consts::FRAC_PI_2 {
        rotation -= std::f64::consts::PI;
    } else if rotation < -std::f64::consts::FRAC_PI_2 {
        rotation += std::f64::consts::PI;
    }
    rotation
}

/// `getConnectorTextAnchorLocalX` (:1306-1310).
pub fn connector_text_anchor_local_x(align: &str, width: f64) -> f64 {
    match align {
        "left" => 0.0,
        "right" => width,
        _ => width / 2.0,
    }
}

/// `calculateLineTextGap` (`WebGPULineToPath.ts:139-228`). `line_angle` is
/// the tangent at the text position; `text_position_t` is the
/// already-arrow-remapped t.
pub fn calculate_line_text_gap(
    spec: &LineSpec,
    text_width: f64,
    text_height: f64,
    line_angle: Option<f64>,
    text_position_t: Option<f64>,
    has_text: bool,
    text_position_field: Option<f64>,
    text_padding: Option<f64>,
    text_rotation: Option<&str>,
) -> Option<TextGap> {
    if !has_text || !is_line_long_enough_for_text(spec) {
        return None;
    }

    let line_length = get_line_length(spec);
    if line_length < 0.001 {
        return None;
    }

    let t = text_position_t.or(text_position_field).unwrap_or(0.5);
    let padding = text_padding.unwrap_or(4.0);
    let extra_padding = 3.0;
    let half_width = text_width / 2.0;
    let half_height = text_height / 2.0;

    let rotation = text_rotation.unwrap_or("none");
    let gap_half_length = if rotation == "follow" {
        half_width + padding + extra_padding
    } else if rotation == "perpendicular" {
        half_height + padding + extra_padding
    } else {
        // Horizontal text: where does a line at this angle exit the text box?
        let angle = match line_angle {
            Some(a) => a,
            None => {
                let (x1, y1, x2, y2) = (
                    spec.points[0],
                    spec.points[1],
                    spec.points[2],
                    spec.points[3],
                );
                (y2 - y1).atan2(x2 - x1)
            }
        };
        // applyAffineToVector((1,0), affineFromTRS(angle)) == (cos, sin).
        let cos_a = angle.cos().abs();
        let sin_a = angle.sin().abs();
        let epsilon = 0.001;
        let exit_distance = if cos_a < epsilon {
            half_height / sin_a
        } else if sin_a < epsilon {
            half_width / cos_a
        } else {
            (half_width / cos_a).min(half_height / sin_a)
        };
        exit_distance + padding + extra_padding
    };

    let gap_length_t = (gap_half_length * 2.0) / line_length;
    if gap_length_t >= 0.9 {
        return None;
    }

    Some(TextGap {
        start_t: (t - gap_length_t / 2.0).max(0.0),
        end_t: (t + gap_length_t / 2.0).min(1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight_spec() -> LineSpec {
        LineSpec {
            points: vec![0.0, 0.0, 100.0, 0.0],
            waypoints: vec![],
            routing: Routing::Straight,
            stroke_width: Some(2.0),
            line_cap: None,
            line_join: None,
            dash: None,
            start_arrow: None,
            end_arrow: None,
            start_arrow_defined: false,
            end_arrow_defined: false,
            pointer_length: None,
            pointer_width: None,
            corner_radius: None,
            body_outlined: false,
            outlined_cap_reach: [0.0, 0.0],
        }
    }

    #[test]
    fn straight_length_and_gap() {
        let spec = straight_spec();
        assert_eq!(get_line_length(&spec), 100.0);
        // Horizontal text on a horizontal line: exit through left/right →
        // halfWidth/|cos| = 20; gap half = 20 + 4 + 3 = 27; gapLengthT=0.54.
        let gap = calculate_line_text_gap(
            &spec,
            40.0,
            16.0,
            Some(0.0),
            Some(0.5),
            true,
            None,
            None,
            None,
        )
        .unwrap();
        assert!((gap.start_t - (0.5 - 0.27)).abs() < 1e-12);
        assert!((gap.end_t - (0.5 + 0.27)).abs() < 1e-12);
    }

    #[test]
    fn gap_none_when_too_short_or_too_wide() {
        let mut spec = straight_spec();
        spec.points = vec![0.0, 0.0, 30.0, 0.0]; // < MIN_LINE_LENGTH_FOR_TEXT
        assert!(
            calculate_line_text_gap(&spec, 40.0, 16.0, None, None, true, None, None, None)
                .is_none()
        );
        // >= 90% of the line → no gap.
        let spec2 = straight_spec();
        assert!(calculate_line_text_gap(
            &spec2,
            200.0,
            16.0,
            Some(0.0),
            Some(0.5),
            true,
            None,
            None,
            None
        )
        .is_none());
    }

    #[test]
    fn perpendicular_rotation_flips_to_readable() {
        // Line at 57.3° → perpendicular 147.3° → flipped to -32.7°.
        let angle = 57.3_f64.to_radians();
        let rot = calculate_text_rotation(angle, Some("perpendicular"));
        assert!((rot - (angle + std::f64::consts::FRAC_PI_2 - std::f64::consts::PI)).abs() < 1e-12);
        assert_eq!(calculate_text_rotation(angle, None), 0.0);
        assert_eq!(calculate_text_rotation(angle, Some("none")), 0.0);
    }

    #[test]
    fn elbowed_rendered_arc_length_shorter_than_polyline() {
        // L-shape 100 + 100 with cornerRadius 12: the rendered arc replaces
        // 2r of polyline with a quadratic arc (shorter).
        let len = get_elbowed_line_arc_length([0.0, 0.0], [100.0, 100.0], &[[100.0, 0.0]], 12.0);
        assert!(len < 200.0);
        assert!(len > 195.0); // arc ≈ 18.85 replacing 24 → ~194.85..199
    }

    #[test]
    fn center_segment_midpoint_elbowed() {
        // 3-segment elbow: center segment is index 1; its midpoint in
        // polyline space.
        let spec = LineSpec {
            points: vec![120.0, 140.0, 700.0, 420.0],
            waypoints: vec![[420.0, 140.0], [420.0, 420.0]],
            routing: Routing::Elbowed,
            stroke_width: Some(4.0),
            line_cap: None,
            line_join: None,
            dash: None,
            start_arrow: None,
            end_arrow: None,
            start_arrow_defined: false,
            end_arrow_defined: false,
            pointer_length: Some(16.0),
            pointer_width: Some(16.0),
            corner_radius: None,
            body_outlined: false,
            outlined_cap_reach: [0.0, 0.0],
        };
        // polyline lengths: 300 + 280 + 280 = 860; to mid of segment 1:
        // 300 + 140 = 440 → t = 440/860.
        let t = get_center_segment_midpoint_t(&spec);
        assert!((t - 440.0 / 860.0).abs() < 1e-12);
    }
}
