//! M2 — line → path conversion: body + arrowheads + dash routing.
//!
//! Faithful ports of (line citations to each file):
//! - `utils/lineToPath.ts`: `lineToPathPoints` (:68-106),
//!   `dashedLineToPathPoints` (:1004-1102), `lineWithArrowsToPathPoints`
//!   (:1109-1243), `elbowedLineToPathPoints` (:1489-1557) +
//!   `dashedElbowedLineToPathPoints` (:1572-1601) + `buildRoundedCorner`
//!   (:1872-1917) + `elbowedLineWithArrowsToPathPoints` (:1673-1857),
//!   `curvedLineToPathPoints` (:1999-2001) — now a THIN DELEGATOR to the
//!   elbowed builder with `CURVED_LINE_CORNER_RADIUS` (:1980), so the curved
//!   stroke hugs the avoiding waypoints (line segments + quadratic corner
//!   arcs) instead of a single cubic Bezier that could cut through a shape.
//!   The old cubic helpers (`generateSmoothCurve`, `computeCurveControlPoints`,
//!   `dashedCurvedLineToPathPoints`, …) were deleted on both sides.
//! - `renderers/webgpu/utils/WebGPULineToPath.ts`: `convertLineToPathNode`
//!   (:239-550), `convertLineBodyToPathNode` (:910-1137),
//!   `convertArrowHeadsToPathNode` (:728-857), `computeEndpointTangent`
//!   (:560-598), `generateFilledArrowShapePoints` (:620-692),
//!   `generateOpenArrowStrokePoints` (:698-716)
//! - `core/scene/arrowheads.ts`: `arrowTrimDistance` (:60-86)
//!
//! TEXT GAPS (M4b): carved via `line_parts_with_gap` /
//! `*_with_text_gap_to_path_points` (`utils/lineToPath.ts` :113-153,
//! :1615-1661, :2085-2116, :2720-2779) with measurement supplied by
//! text_draw (production measures with Canvas2D — `WebGPUShapeSystem.ts:
//! 896-915`). The no-gap `line_parts` keeps the frozen M2 behavior — the
//! tessellation oracle's bodies stay UN-gapped
//! (`textGap: "omitted-browser-measure"`).

use super::dash::{
    build_elbowed_path_segments, normalize_dash_pattern, scale_dash_pattern,
    walk_path_by_arc_length, PathSegment, Pt, Seg, WalkCmd,
};
use super::moda_engine_path_types::PathCmd;
use moda_scene::value::Value;

/// `BASE_DASH_STROKE_WIDTH` default used when strokeWidth is undefined in the
/// dash-capable generators (lineToPath.ts:69, :1498, :1993).
const BASE_DASH_STROKE_WIDTH: f64 = 2.0;
/// Cubic arcs in a circle arrowhead (the TS port's `CIRCLE_TESSELLATION_SEGMENTS`, WebGPULineToPath.ts:553,
/// was a fixed 32-gon).
const CIRCLE_ARCS: usize = 8;
/// `CURVED_LINE_CORNER_RADIUS` (lineToPath.ts:1980). Curved lines reuse the
/// elbowed rounded-corner builder so the stroke HUGS the orthogonal avoiding
/// waypoints instead of cutting a single cubic Bezier through a connected
/// shape. The radius is capped at the auto-route clearance (`MIN_EXIT_DISTANCE`
/// in `lineRouting.ts`, 20) so every corner fillet stays within
/// `disk(waypoint, 20)` and can never cross a connected shape. Exposed
/// `pub(crate)` so `connector_text.rs` shares the SAME source of truth when it
/// delegates the curved arc-length/sampling/projection helpers to elbowed.
pub(crate) const CURVED_LINE_CORNER_RADIUS: f64 = 20.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrowShape {
    Triangle,
    Open,
    ReverseTriangle,
    Diamond,
    Circle,
}

impl ArrowShape {
    pub fn from_str(s: &str) -> Option<ArrowShape> {
        match s {
            "triangle" => Some(ArrowShape::Triangle),
            "open" => Some(ArrowShape::Open),
            "reverse-triangle" => Some(ArrowShape::ReverseTriangle),
            "diamond" => Some(ArrowShape::Diamond),
            "circle" => Some(ArrowShape::Circle),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Routing {
    Straight,
    Elbowed,
    Curved,
}

/// The line-node fields the conversion consumes (parsed once from JSON).
#[derive(Debug, Clone)]
pub struct LineSpec {
    /// [x1, y1, x2, y2]; fewer than 4 entries = invalid (empty path node).
    pub points: Vec<f64>,
    pub waypoints: Vec<Pt>,
    pub routing: Routing,
    pub stroke_width: Option<f64>,
    pub line_cap: Option<String>,
    pub line_join: Option<String>,
    pub dash: Option<Vec<f64>>,
    pub start_arrow: Option<ArrowShape>,
    pub end_arrow: Option<ArrowShape>,
    /// Raw `startArrowShape` key present with ANY value, including explicit
    /// JSON `null` — TS `line.startArrowShape !== undefined`
    /// (WebGPULineToPath.ts:273-276). Real API envelopes serialize unset
    /// arrow shapes as explicit `null`, which production treats as a
    /// "defined" non-triangle/non-open arrowhead: `convertLineToPathNode`
    /// flips into filled-silhouette mode (stroke stripped, path closed) and,
    /// since no arrow geometry is generated for a falsy shape, the line
    /// paints ZERO pixels in the browser. Copy-the-bugs parity (WF5 v3
    /// b284e190; quirk Q-91).
    pub start_arrow_defined: bool,
    /// Raw `endArrowShape` key present with any value — see
    /// `start_arrow_defined`.
    pub end_arrow_defined: bool,
    pub pointer_length: Option<f64>,
    pub pointer_width: Option<f64>,
    pub corner_radius: Option<f64>,
    /// The body is drawn as a width-model outline (brushes.md §4.9), not a
    /// uniform stroke: the arrowhead set-back allows for the cap by the
    /// outline's own half-width at that end (`outlined_cap_reach`), not half
    /// the weight, so a body tapering to a point meets its head and a full
    /// round end stops at it. Not JSON fields; the renderer sets them.
    pub body_outlined: bool,
    /// Half-width of the outlined body at its start and end (round / square
    /// caps reach this far past the end); 0 = no allowance.
    pub outlined_cap_reach: [f64; 2],
}

impl LineSpec {
    /// Build from a line node's JSON fields (use with `RawNode::get`).
    pub fn from_json<'a>(get: impl Fn(&str) -> Option<&'a Value>) -> LineSpec {
        let points: Vec<f64> = get("points")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default();
        let waypoints: Vec<Pt> = get("waypoints")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|w| {
                        let o = w.as_object()?;
                        Some([o.get("x")?.as_f64()?, o.get("y")?.as_f64()?])
                    })
                    .collect()
            })
            .unwrap_or_default();
        let routing = match get("routing").and_then(Value::as_str) {
            Some("elbowed") => Routing::Elbowed,
            Some("curved") => Routing::Curved,
            _ => Routing::Straight,
        };
        // The line's top stroke: its width, caps, joins, dash and arrowheads
        // (a line without a stroke has none of them and draws nothing).
        let style = moda_scene::paint::stroke_style_from(|k| get(k));
        let start_arrow = style.start_arrow.as_deref().and_then(ArrowShape::from_str);
        let end_arrow = style.end_arrow.as_deref().and_then(ArrowShape::from_str);
        // `arrowScale` multiplies the line's pointer size (default 10).
        let scale = style.arrow_scale.filter(|s| s.is_finite() && *s > 0.0);
        let pointer = |key: &str| {
            let v = get(key).and_then(Value::as_f64);
            match scale {
                Some(s) => Some(v.unwrap_or(10.0) * s),
                None => v,
            }
        };
        LineSpec {
            points,
            waypoints,
            routing,
            stroke_width: style.width,
            line_cap: style.cap,
            line_join: style.join,
            dash: style.dash,
            start_arrow,
            end_arrow,
            start_arrow_defined: start_arrow.is_some(),
            end_arrow_defined: end_arrow.is_some(),
            pointer_length: pointer("pointerLength"),
            pointer_width: pointer("pointerWidth"),

            corner_radius: get("cornerRadius").and_then(Value::as_f64),
            body_outlined: false,
            outlined_cap_reach: [0.0, 0.0],
        }
    }

    fn valid(&self) -> bool {
        self.points.len() >= 4 // hasValidLinePoints (WebGPULineToPath.ts:43-45)
    }

    pub fn has_arrows(&self) -> bool {
        self.start_arrow.is_some() || self.end_arrow.is_some()
    }

    fn xy(&self) -> (f64, f64, f64, f64) {
        (
            self.points[0],
            self.points[1],
            self.points[2],
            self.points[3],
        )
    }
}

/// One renderable part of a line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinePartKind {
    /// The (stroked) line body — fill 'transparent', stroke = line stroke.
    Body,
    /// Filled arrowheads — closed path, fill = line stroke color, no stroke.
    ArrowFill,
    /// Open-caret arrowheads — stroked path mirroring the line stroke.
    ArrowOpen,
}

#[derive(Debug, Clone)]
pub struct LinePart {
    pub kind: LinePartKind,
    pub cmds: Vec<PathCmd>,
    pub closed: bool,
    /// Stroke width the tessellator should use (0 = no stroke).
    pub stroke_width: f64,
    pub line_cap: Option<String>,
    pub line_join: Option<String>,
}

/// `TextGap` (`utils/lineToPath.ts:25-28`) — arc-length fractions of the
/// rendered line ([0,1]) to carve out for a connector text label.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextGap {
    pub start_t: f64,
    pub end_t: f64,
}

impl TextGap {
    pub fn active(&self) -> bool {
        // Every TS gap arm guards `textGap.startT < textGap.endT`.
        self.start_t < self.end_t
    }
}

/// `arrowTrimDistance` (arrowheads.ts:60-86).
pub fn arrow_trim_distance(
    shape: ArrowShape,
    pointer_length: f64,
    pointer_width: f64,
    line_cap: Option<&str>,
    stroke_width: f64,
) -> f64 {
    let mut base = match shape {
        ArrowShape::Triangle | ArrowShape::ReverseTriangle | ArrowShape::Open => pointer_length,
        ArrowShape::Diamond => pointer_length / 2.0,
        ArrowShape::Circle => pointer_length.min(pointer_width) / 2.0,
    };
    let cap = line_cap.unwrap_or("butt");
    if (cap == "round" || cap == "square") && stroke_width > 0.0 {
        base += stroke_width / 2.0;
    }
    base
}

/// Mirror of `WebGPUShapeSystem.renderLine` / dump-tessellation
/// `itemsForNode('line')`: with arrowheads → stroked body +
/// `convertArrowHeadsToPathNode` parts; without → single
/// `convertLineToPathNode` part. Body part is always FIRST.
pub fn line_parts(spec: &LineSpec) -> Vec<LinePart> {
    line_parts_with_gap(spec, None)
}

/// M4b: `line_parts` with connector-text gap carving
/// (`WebGPUShapeSystem.renderLine` :918-983 — the gap is carved into the
/// body/integrated path; arrowheads are never gapped). The tessellation
/// oracle's line bodies are UN-gapped (`textGap: "omitted-browser-measure"`)
/// — oracle callers must keep passing `None` (i.e. use `line_parts`).
pub fn line_parts_with_gap(spec: &LineSpec, text_gap: Option<TextGap>) -> Vec<LinePart> {
    if spec.has_arrows() {
        let mut parts = vec![convert_line_body_with_gap(spec, text_gap)];
        parts.extend(convert_arrow_heads(spec));
        parts
    } else {
        vec![convert_line_to_path_with_gap(spec, text_gap)]
    }
}

// ---------------------------------------------------------------------------
// convertLineToPathNode (no-arrow primary path + full silhouette fallbacks)
// ---------------------------------------------------------------------------

/// `convertLineToPathNode` (WebGPULineToPath.ts:239-550), without text gaps.
pub fn convert_line_to_path(spec: &LineSpec) -> LinePart {
    convert_line_to_path_with_gap(spec, None)
}

/// `convertLineToPathNode` with a pre-computed text gap (the
/// `options.textGap` arm — :294-298; the auto-calc-from-measurements arm is
/// browser-only and never taken natively: the caller always measures first).
pub fn convert_line_to_path_with_gap(spec: &LineSpec, text_gap: Option<TextGap>) -> LinePart {
    if !spec.valid() {
        return LinePart {
            kind: LinePartKind::Body,
            cmds: Vec::new(),
            closed: false,
            stroke_width: 0.0,
            line_cap: None,
            line_join: None,
        };
    }

    let integrate_start = spec.start_arrow == Some(ArrowShape::Triangle);
    let integrate_end = spec.end_arrow == Some(ArrowShape::Triangle);
    let has_integrated = integrate_start || integrate_end;

    // TS `!== undefined` (WebGPULineToPath.ts:273-276): an explicit JSON
    // null (or unparseable string) arrow shape still counts as "defined",
    // forcing filled-silhouette mode with no arrow geometry → the body
    // paints nothing. Key-presence, NOT parsed-shape presence (Q-91).
    let needs_extra_start = spec.start_arrow_defined
        && spec.start_arrow != Some(ArrowShape::Triangle)
        && spec.start_arrow != Some(ArrowShape::Open);
    let needs_extra_end = spec.end_arrow_defined
        && spec.end_arrow != Some(ArrowShape::Triangle)
        && spec.end_arrow != Some(ArrowShape::Open);
    let has_extra_filled = needs_extra_start || needs_extra_end;

    let p_len = spec.pointer_length.unwrap_or(10.0);
    let p_wid = spec.pointer_width.unwrap_or(10.0);
    let sw = spec.stroke_width.unwrap_or(0.0);

    let (mut start_trim, mut end_trim) = (0.0f64, 0.0f64);
    if has_extra_filled {
        if needs_extra_start {
            if let Some(s) = spec.start_arrow {
                start_trim = arrow_trim_distance(s, p_len, p_wid, spec.line_cap.as_deref(), sw);
            }
        }
        if needs_extra_end {
            if let Some(s) = spec.end_arrow {
                end_trim = arrow_trim_distance(s, p_len, p_wid, spec.line_cap.as_deref(), sw);
            }
        }
    }

    let (x1, y1, x2, y2) = spec.xy();
    let mut points: Vec<PathCmd>;

    if spec.routing == Routing::Curved && !spec.waypoints.is_empty() {
        // Curved lines now follow the waypoint polyline with rounded corners
        // (same as elbowed), so trim the endpoints along the first/last
        // waypoint segment direction — NOT a Bezier tangent
        // (WebGPULineToPath.ts:302-334).
        let mut adj = [x1, y1, x2, y2];
        if start_trim > 0.0 || end_trim > 0.0 {
            if start_trim > 0.0 {
                let wp0 = spec.waypoints[0];
                let dx = wp0[0] - x1;
                let dy = wp0[1] - y1;
                let len = (dx * dx + dy * dy).sqrt();
                if len > 0.001 {
                    adj[0] = x1 + (dx / len) * start_trim;
                    adj[1] = y1 + (dy / len) * start_trim;
                }
            }
            if end_trim > 0.0 {
                let last = spec.waypoints[spec.waypoints.len() - 1];
                let dx = x2 - last[0];
                let dy = y2 - last[1];
                let len = (dx * dx + dy * dy).sqrt();
                if len > 0.001 {
                    adj[2] = x2 - (dx / len) * end_trim;
                    adj[3] = y2 - (dy / len) * end_trim;
                }
            }
        }
        points = curved_line_to_path_points(
            adj,
            &spec.waypoints,
            sw_or_base(spec),
            integrate_start,
            integrate_end,
            p_len,
            p_wid,
            spec.dash.as_deref(),
            text_gap,
        );
    } else if spec.routing == Routing::Elbowed && !spec.waypoints.is_empty() {
        let mut adj = [x1, y1, x2, y2];
        if (start_trim > 0.0 || end_trim > 0.0) && !spec.waypoints.is_empty() {
            if start_trim > 0.0 {
                let wp0 = spec.waypoints[0];
                let dx = wp0[0] - x1;
                let dy = wp0[1] - y1;
                let len = (dx * dx + dy * dy).sqrt();
                if len > 0.001 {
                    adj[0] = x1 + (dx / len) * start_trim;
                    adj[1] = y1 + (dy / len) * start_trim;
                }
            }
            if end_trim > 0.0 {
                let last = spec.waypoints[spec.waypoints.len() - 1];
                let dx = x2 - last[0];
                let dy = y2 - last[1];
                let len = (dx * dx + dy * dy).sqrt();
                if len > 0.001 {
                    adj[2] = x2 - (dx / len) * end_trim;
                    adj[3] = y2 - (dy / len) * end_trim;
                }
            }
        }
        points = elbowed_line_to_path_points(
            adj,
            &spec.waypoints,
            spec.corner_radius.unwrap_or(0.0),
            sw_or_base(spec),
            integrate_start,
            integrate_end,
            p_len,
            p_wid,
            spec.dash.as_deref(),
            text_gap,
        );
    } else {
        let mut adj = [x1, y1, x2, y2];
        if start_trim > 0.0 || end_trim > 0.0 {
            let dx = x2 - x1;
            let dy = y2 - y1;
            let len = (dx * dx + dy * dy).sqrt();
            if len > 0.001 {
                let ux = dx / len;
                let uy = dy / len;
                if start_trim > 0.0 {
                    adj[0] = x1 + ux * start_trim;
                    adj[1] = y1 + uy * start_trim;
                }
                if end_trim > 0.0 {
                    adj[2] = x2 - ux * end_trim;
                    adj[3] = y2 - uy * end_trim;
                }
            }
        }
        points = line_to_path_points(
            adj,
            sw_or_base_arrow(spec, has_integrated),
            integrate_start,
            integrate_end,
            p_len,
            p_wid,
            spec.dash.as_deref(),
            text_gap,
        );
    }

    // Extra filled arrow subpaths (:418-463).
    if needs_extra_start {
        if let Some(t) = endpoint_tangent(spec, true) {
            if let Some(shape) = spec.start_arrow {
                if shape != ArrowShape::Open {
                    if let Some(extra) = filled_arrow_shape_points(
                        shape,
                        spec.points[0],
                        spec.points[1],
                        -t.unit_x,
                        -t.unit_y,
                        -t.perp_x,
                        -t.perp_y,
                        p_len,
                        p_wid,
                    ) {
                        points.extend(extra);
                    }
                }
            }
        }
    }
    if needs_extra_end {
        if let Some(t) = endpoint_tangent(spec, false) {
            if let Some(shape) = spec.end_arrow {
                if shape != ArrowShape::Open {
                    if let Some(extra) = filled_arrow_shape_points(
                        shape,
                        spec.points[2],
                        spec.points[3],
                        t.unit_x,
                        t.unit_y,
                        t.perp_x,
                        t.perp_y,
                        p_len,
                        p_wid,
                    ) {
                        points.extend(extra);
                    }
                }
            }
        }
    }

    let has_filled_shape = has_integrated || has_extra_filled;

    LinePart {
        kind: LinePartKind::Body,
        cmds: points,
        closed: has_filled_shape,
        // filled-shape mode: fill = stroke color, no stroke (:493-496)
        stroke_width: if has_filled_shape {
            0.0
        } else {
            spec.stroke_width.unwrap_or(0.0)
        },
        line_cap: if has_filled_shape {
            None
        } else {
            spec.line_cap.clone()
        },
        line_join: if has_filled_shape {
            None
        } else {
            spec.line_join.clone()
        },
    }
}

/// strokeWidth default for the dash-capable generators (`?? BASE_DASH…`).
fn sw_or_base(spec: &LineSpec) -> f64 {
    spec.stroke_width.unwrap_or(BASE_DASH_STROKE_WIDTH)
}

/// `lineToPathPoints` defaults strokeWidth to BASE_DASH_STROKE_WIDTH (:69)
/// but `lineWithArrowsToPathPoints` defaults it to 1 (:1112).
fn sw_or_base_arrow(spec: &LineSpec, integrated: bool) -> f64 {
    match spec.stroke_width {
        Some(w) => w,
        None if integrated => 1.0,
        None => BASE_DASH_STROKE_WIDTH,
    }
}

// ---------------------------------------------------------------------------
// convertLineBodyToPathNode (:910-1137), no text gap
// ---------------------------------------------------------------------------

pub fn convert_line_body(spec: &LineSpec) -> LinePart {
    convert_line_body_with_gap(spec, None)
}

/// `convertLineBodyToPathNode` with a pre-computed text gap (:913-944):
/// when arrows shorten the body, the gap t-values (fractions of the FULL
/// line) are remapped into the shortened body's t-space
/// (`remapTextGapForShortenedLine`, :864-900).
pub fn convert_line_body_with_gap(spec: &LineSpec, text_gap: Option<TextGap>) -> LinePart {
    if !spec.valid() {
        return LinePart {
            kind: LinePartKind::Body,
            cmds: Vec::new(),
            closed: false,
            stroke_width: 0.0,
            line_cap: None,
            line_join: None,
        };
    }

    let has_start = spec.start_arrow.is_some();
    let has_end = spec.end_arrow.is_some();
    let p_len = spec.pointer_length.unwrap_or(10.0);
    let p_wid = spec.pointer_width.unwrap_or(10.0);
    let sw = spec.stroke_width.unwrap_or(0.0);

    // An outlined body's cap reaches only as far as the outline's half-width
    // at that end (none where it tapers to a point), so it is set back as if
    // butt-capped plus that reach.
    let set_back = |shape: ArrowShape, end: usize| {
        if spec.body_outlined {
            let capped = matches!(spec.line_cap.as_deref(), Some("round" | "square"));
            arrow_trim_distance(shape, p_len, p_wid, None, sw)
                + if capped {
                    spec.outlined_cap_reach[end].max(0.0)
                } else {
                    0.0
                }
        } else {
            arrow_trim_distance(shape, p_len, p_wid, spec.line_cap.as_deref(), sw)
        }
    };
    let start_trim = spec.start_arrow.map(|s| set_back(s, 0)).unwrap_or(0.0);
    let end_trim = spec.end_arrow.map(|s| set_back(s, 1)).unwrap_or(0.0);

    // `if (textGap && (hasStartArrow || hasEndArrow))` (:941-944).
    let text_gap = match text_gap {
        Some(gap) if has_start || has_end => {
            let full_length = super::connector_text::get_line_length(spec);
            Some(remap_text_gap_for_shortened_line(
                gap,
                full_length,
                has_start,
                has_end,
                start_trim,
                end_trim,
            ))
        }
        other => other,
    };

    let (x1, y1, x2, y2) = spec.xy();
    let points: Vec<PathCmd>;

    if spec.routing == Routing::Curved && !spec.waypoints.is_empty() {
        // Curved lines now follow the waypoint polyline with rounded corners
        // (same as elbowed), so trim the endpoints along the first/last
        // waypoint segment direction — NOT a Bezier tangent
        // (WebGPULineToPath.ts:302-334).
        let mut adj = [x1, y1, x2, y2];
        if has_start {
            let wp0 = spec.waypoints[0];
            let dx = wp0[0] - x1;
            let dy = wp0[1] - y1;
            let len = (dx * dx + dy * dy).sqrt();
            if len > 0.001 {
                adj[0] = x1 + (dx / len) * start_trim;
                adj[1] = y1 + (dy / len) * start_trim;
            }
        }
        if has_end {
            let last = spec.waypoints[spec.waypoints.len() - 1];
            let dx = x2 - last[0];
            let dy = y2 - last[1];
            let len = (dx * dx + dy * dy).sqrt();
            if len > 0.001 {
                adj[2] = x2 - (dx / len) * end_trim;
                adj[3] = y2 - (dy / len) * end_trim;
            }
        }
        points = curved_line_to_path_points(
            adj,
            &spec.waypoints,
            sw_or_base(spec),
            false,
            false,
            p_len,
            p_wid,
            spec.dash.as_deref(),
            text_gap,
        );
    } else if spec.routing == Routing::Elbowed && !spec.waypoints.is_empty() {
        let mut adj = [x1, y1, x2, y2];
        if has_start && !spec.waypoints.is_empty() {
            let wp0 = spec.waypoints[0];
            let dx = wp0[0] - x1;
            let dy = wp0[1] - y1;
            let len = (dx * dx + dy * dy).sqrt();
            if len > 0.001 {
                adj[0] = x1 + (dx / len) * start_trim;
                adj[1] = y1 + (dy / len) * start_trim;
            }
        }
        if has_end && !spec.waypoints.is_empty() {
            let last = spec.waypoints[spec.waypoints.len() - 1];
            let dx = x2 - last[0];
            let dy = y2 - last[1];
            let len = (dx * dx + dy * dy).sqrt();
            if len > 0.001 {
                adj[2] = x2 - (dx / len) * end_trim;
                adj[3] = y2 - (dy / len) * end_trim;
            }
        }
        points = elbowed_line_to_path_points(
            adj,
            &spec.waypoints,
            spec.corner_radius.unwrap_or(0.0),
            sw_or_base(spec),
            false,
            false,
            p_len,
            p_wid,
            spec.dash.as_deref(),
            text_gap,
        );
    } else {
        let mut adj = [x1, y1, x2, y2];
        if has_start || has_end {
            let dx = x2 - x1;
            let dy = y2 - y1;
            let len = (dx * dx + dy * dy).sqrt();
            if len > 0.001 {
                let ux = dx / len;
                let uy = dy / len;
                if has_start {
                    adj[0] = x1 + ux * start_trim;
                    adj[1] = y1 + uy * start_trim;
                }
                if has_end {
                    adj[2] = x2 - ux * end_trim;
                    adj[3] = y2 - uy * end_trim;
                }
            }
        }
        points = line_to_path_points(
            adj,
            sw_or_base(spec),
            false,
            false,
            p_len,
            p_wid,
            spec.dash.as_deref(),
            text_gap,
        );
    }

    LinePart {
        kind: LinePartKind::Body,
        cmds: points,
        closed: false,
        stroke_width: spec.stroke_width.unwrap_or(0.0),
        line_cap: spec.line_cap.clone(),
        line_join: spec.line_join.clone(),
    }
}

// ---------------------------------------------------------------------------
// convertArrowHeadsToPathNode (:728-857)
// ---------------------------------------------------------------------------

struct Tangent {
    unit_x: f64,
    unit_y: f64,
    perp_x: f64,
    perp_y: f64,
}

/// `computeEndpointTangent` (:560-598).
fn endpoint_tangent(spec: &LineSpec, start: bool) -> Option<Tangent> {
    let (x1, y1, x2, y2) = spec.xy();
    let has_waypoints = !spec.waypoints.is_empty();

    let (ax, ay, bx, by) = if start {
        if has_waypoints {
            (x1, y1, spec.waypoints[0][0], spec.waypoints[0][1])
        } else {
            (x1, y1, x2, y2)
        }
    } else if has_waypoints {
        let wp = spec.waypoints[spec.waypoints.len() - 1];
        (wp[0], wp[1], x2, y2)
    } else {
        (x1, y1, x2, y2)
    };

    let dx = bx - ax;
    let dy = by - ay;
    let len = dx.hypot(dy);
    if len < 0.001 {
        return None;
    }
    let unit_x = dx / len;
    let unit_y = dy / len;
    Some(Tangent {
        unit_x,
        unit_y,
        perp_x: -unit_y,
        perp_y: unit_x,
    })
}

/// `generateFilledArrowShapePoints` (:620-692). None for `open`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn filled_arrow_shape_points(
    shape: ArrowShape,
    tip_x: f64,
    tip_y: f64,
    unit_x: f64,
    unit_y: f64,
    perp_x: f64,
    perp_y: f64,
    length: f64,
    width: f64,
) -> Option<Vec<PathCmd>> {
    let half_w = width / 2.0;

    match shape {
        ArrowShape::Triangle => Some(vec![
            PathCmd::Move { x: tip_x, y: tip_y },
            PathCmd::Line {
                x: tip_x - unit_x * length - perp_x * half_w,
                y: tip_y - unit_y * length - perp_y * half_w,
            },
            PathCmd::Line {
                x: tip_x - unit_x * length + perp_x * half_w,
                y: tip_y - unit_y * length + perp_y * half_w,
            },
        ]),
        ArrowShape::ReverseTriangle => {
            let inner_x = tip_x - unit_x * length;
            let inner_y = tip_y - unit_y * length;
            Some(vec![
                PathCmd::Move {
                    x: tip_x + perp_x * half_w,
                    y: tip_y + perp_y * half_w,
                },
                PathCmd::Line {
                    x: tip_x - perp_x * half_w,
                    y: tip_y - perp_y * half_w,
                },
                PathCmd::Line {
                    x: inner_x,
                    y: inner_y,
                },
            ])
        }
        ArrowShape::Diamond => {
            let half_l = length / 2.0;
            Some(vec![
                PathCmd::Move {
                    x: tip_x + unit_x * half_l,
                    y: tip_y + unit_y * half_l,
                },
                PathCmd::Line {
                    x: tip_x + perp_x * half_w,
                    y: tip_y + perp_y * half_w,
                },
                PathCmd::Line {
                    x: tip_x - unit_x * half_l,
                    y: tip_y - unit_y * half_l,
                },
                PathCmd::Line {
                    x: tip_x - perp_x * half_w,
                    y: tip_y - perp_y * half_w,
                },
            ])
        }
        ArrowShape::Circle => {
            let radius = length.min(width) / 2.0;
            if radius <= 0.001 {
                return None;
            }
            // Eight cubic arcs (45° each, within 5e-6 of the radius), so the path's flattening makes the
            // disc as round as the zoom band needs (the TS port drew a fixed 32-gon).
            let at = |theta: f64, r: f64| {
                let (c, s) = (theta.cos() * r, theta.sin() * r);
                (tip_x + unit_x * c + perp_x * s, tip_y + unit_y * c + perp_y * s)
            };
            let step = std::f64::consts::TAU / CIRCLE_ARCS as f64;
            // Control arm of a cubic arc of angle `step`: 4/3·tan(step/4).
            let k = 4.0 / 3.0 * (step / 4.0).tan();
            let (x0, y0) = at(0.0, radius);
            let mut pts = Vec::with_capacity(CIRCLE_ARCS + 1);
            pts.push(PathCmd::Move { x: x0, y: y0 });
            for i in 0..CIRCLE_ARCS {
                let (a0, a1) = (i as f64 * step, (i + 1) as f64 * step);
                // Tangent offsets: along the circle at each end, `k·radius` long.
                let off = |theta: f64, sign: f64| {
                    let (c, s) = (-theta.sin() * sign * k * radius, theta.cos() * sign * k * radius);
                    (unit_x * c + perp_x * s, unit_y * c + perp_y * s)
                };
                let (p0, p1) = (at(a0, radius), at(a1, radius));
                let (d0, d1) = (off(a0, 1.0), off(a1, -1.0));
                pts.push(PathCmd::Curve {
                    x: p1.0,
                    y: p1.1,
                    cp1: (p0.0 + d0.0, p0.1 + d0.1),
                    cp2: (p1.0 + d1.0, p1.1 + d1.1),
                });
            }
            Some(pts)
        }
        ArrowShape::Open => None,
    }
}

/// `generateOpenArrowStrokePoints` (:698-716).
#[allow(clippy::too_many_arguments)]
pub(crate) fn open_arrow_stroke_points(
    tip_x: f64,
    tip_y: f64,
    unit_x: f64,
    unit_y: f64,
    perp_x: f64,
    perp_y: f64,
    length: f64,
    width: f64,
) -> Vec<PathCmd> {
    let half_w = width / 2.0;
    let base_x = tip_x - unit_x * length;
    let base_y = tip_y - unit_y * length;
    vec![
        PathCmd::Move {
            x: base_x - perp_x * half_w,
            y: base_y - perp_y * half_w,
        },
        PathCmd::Line { x: tip_x, y: tip_y },
        PathCmd::Line {
            x: base_x + perp_x * half_w,
            y: base_y + perp_y * half_w,
        },
    ]
}

/// `convertArrowHeadsToPathNode` (:728-857) — filled batch first, then the
/// open-caret stroked batch.
pub fn convert_arrow_heads(spec: &LineSpec) -> Vec<LinePart> {
    if !spec.valid() || !spec.has_arrows() {
        return Vec::new();
    }

    let p_len = spec.pointer_length.unwrap_or(10.0);
    let p_wid = spec.pointer_width.unwrap_or(10.0);

    let mut filled: Vec<PathCmd> = Vec::new();
    let mut open_stroke: Vec<PathCmd> = Vec::new();

    if let Some(shape) = spec.start_arrow {
        if let Some(t) = endpoint_tangent(spec, true) {
            // Start tangent points INTO the line — reverse so outward points away.
            let (ux, uy, px, py) = (-t.unit_x, -t.unit_y, -t.perp_x, -t.perp_y);
            let (tip_x, tip_y) = (spec.points[0], spec.points[1]);
            if shape == ArrowShape::Open {
                open_stroke.extend(open_arrow_stroke_points(
                    tip_x, tip_y, ux, uy, px, py, p_len, p_wid,
                ));
            } else if let Some(pts) =
                filled_arrow_shape_points(shape, tip_x, tip_y, ux, uy, px, py, p_len, p_wid)
            {
                filled.extend(pts);
            }
        }
    }

    if let Some(shape) = spec.end_arrow {
        if let Some(t) = endpoint_tangent(spec, false) {
            let (ux, uy, px, py) = (t.unit_x, t.unit_y, t.perp_x, t.perp_y);
            let (tip_x, tip_y) = (spec.points[2], spec.points[3]);
            if shape == ArrowShape::Open {
                open_stroke.extend(open_arrow_stroke_points(
                    tip_x, tip_y, ux, uy, px, py, p_len, p_wid,
                ));
            } else if let Some(pts) =
                filled_arrow_shape_points(shape, tip_x, tip_y, ux, uy, px, py, p_len, p_wid)
            {
                filled.extend(pts);
            }
        }
    }

    let mut parts = Vec::new();
    if !filled.is_empty() {
        parts.push(LinePart {
            kind: LinePartKind::ArrowFill,
            cmds: filled,
            closed: true,
            stroke_width: 0.0,
            line_cap: None,
            line_join: None,
        });
    }
    if !open_stroke.is_empty() {
        parts.push(LinePart {
            kind: LinePartKind::ArrowOpen,
            cmds: open_stroke,
            closed: false,
            stroke_width: spec.stroke_width.unwrap_or(0.0),
            line_cap: spec.line_cap.clone(),
            line_join: spec.line_join.clone(),
        });
    }
    parts
}

// ---------------------------------------------------------------------------
// lineToPath.ts generators (no text gap)
// ---------------------------------------------------------------------------

/// `lineToPathPoints` (:68-106).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn line_to_path_points(
    points: [f64; 4],
    stroke_width: f64,
    pointer_at_beginning: bool,
    pointer_at_ending: bool,
    pointer_length: f64,
    pointer_width: f64,
    dash: Option<&[f64]>,
    text_gap: Option<TextGap>,
) -> Vec<PathCmd> {
    if pointer_at_beginning || pointer_at_ending {
        return line_with_arrows_to_path_points(
            points,
            stroke_width,
            pointer_at_beginning,
            pointer_at_ending,
            pointer_length,
            pointer_width,
        );
    }

    let [x1, y1, x2, y2] = points;

    // `if (textGap && textGap.startT < textGap.endT)` (:80-81) — gap beats
    // the dash branch; the gap generator handles dash itself.
    if let Some(gap) = text_gap.filter(TextGap::active) {
        return line_with_text_gap_to_path_points(x1, y1, x2, y2, gap, dash, stroke_width);
    }

    match dash {
        Some(d) if !d.is_empty() => dashed_line_to_path_points(x1, y1, x2, y2, d, stroke_width),
        _ => vec![
            PathCmd::Move { x: x1, y: y1 },
            PathCmd::Line { x: x2, y: y2 },
        ],
    }
}

/// `lineWithTextGapToPathPoints` (:113-153): two disconnected runs —
/// start→gapStart and gapEnd→end (each dash-walked independently when
/// dashed, so the pattern restarts on each side).
fn line_with_text_gap_to_path_points(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    gap: TextGap,
    dash: Option<&[f64]>,
    stroke_width: f64,
) -> Vec<PathCmd> {
    let mut path_points: Vec<PathCmd> = Vec::new();

    let gap_start_x = x1 + (x2 - x1) * gap.start_t;
    let gap_start_y = y1 + (y2 - y1) * gap.start_t;
    let gap_end_x = x1 + (x2 - x1) * gap.end_t;
    let gap_end_y = y1 + (y2 - y1) * gap.end_t;

    if let Some(d) = dash {
        if !d.is_empty() {
            let mut out =
                dashed_line_to_path_points(x1, y1, gap_start_x, gap_start_y, d, stroke_width);
            out.extend(dashed_line_to_path_points(
                gap_end_x,
                gap_end_y,
                x2,
                y2,
                d,
                stroke_width,
            ));
            return out;
        }
    }

    if gap.start_t > 0.001 {
        path_points.push(PathCmd::Move { x: x1, y: y1 });
        path_points.push(PathCmd::Line {
            x: gap_start_x,
            y: gap_start_y,
        });
    }
    if gap.end_t < 0.999 {
        path_points.push(PathCmd::Move {
            x: gap_end_x,
            y: gap_end_y,
        });
        path_points.push(PathCmd::Line { x: x2, y: y2 });
    }

    path_points
}

/// `remapTextGapForShortenedLine` (`WebGPULineToPath.ts:864-900`).
fn remap_text_gap_for_shortened_line(
    text_gap: TextGap,
    full_length: f64,
    has_start_arrow: bool,
    has_end_arrow: bool,
    start_trim: f64,
    end_trim: f64,
) -> TextGap {
    let mut shortened_length = full_length;
    if has_start_arrow {
        shortened_length -= start_trim;
    }
    if has_end_arrow {
        shortened_length -= end_trim;
    }

    if shortened_length <= 0.001 {
        return TextGap {
            start_t: 0.4,
            end_t: 0.6,
        };
    }

    let start_offset = if has_start_arrow { start_trim } else { 0.0 };

    let full_start_x = text_gap.start_t * full_length;
    let full_end_x = text_gap.end_t * full_length;

    TextGap {
        start_t: ((full_start_x - start_offset) / shortened_length).clamp(0.0, 1.0),
        end_t: ((full_end_x - start_offset) / shortened_length).clamp(0.0, 1.0),
    }
}

/// `dashedLineToPathPoints` (:1004-1102).
fn dashed_line_to_path_points(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    dash_pattern: &[f64],
    stroke_width: f64,
) -> Vec<PathCmd> {
    let mut path_points: Vec<PathCmd> = Vec::new();

    let dx = x2 - x1;
    let dy = y2 - y1;
    let line_length = (dx * dx + dy * dy).sqrt();

    if line_length < 0.001 {
        return vec![
            PathCmd::Move { x: x1, y: y1 },
            PathCmd::Line { x: x2, y: y2 },
        ];
    }

    let unit_x = dx / line_length;
    let unit_y = dy / line_length;

    let scaled = scale_dash_pattern(dash_pattern, stroke_width);
    let normalized = normalize_dash_pattern(scaled);
    let pattern_length: f64 = normalized.iter().sum();

    if pattern_length == 0.0 {
        return vec![
            PathCmd::Move { x: x1, y: y1 },
            PathCmd::Line { x: x2, y: y2 },
        ];
    }

    let mut current_distance = 0.0f64;
    let mut pattern_index = 0usize;
    let mut is_dash = true;

    while current_distance < line_length {
        let segment_length = normalized[pattern_index % normalized.len()];
        let segment_end = (current_distance + segment_length).min(line_length);

        if is_dash && segment_length > 0.0 {
            let start_x = x1 + unit_x * current_distance;
            let start_y = y1 + unit_y * current_distance;
            let end_x = x1 + unit_x * segment_end;
            let end_y = y1 + unit_y * segment_end;

            path_points.push(PathCmd::Move {
                x: start_x,
                y: start_y,
            });
            path_points.push(PathCmd::Line { x: end_x, y: end_y });
        }

        current_distance = segment_end;
        pattern_index += 1;
        is_dash = !is_dash;
    }

    path_points
}

/// `lineWithArrowsToPathPoints` (:1109-1243) — integrated filled silhouette.
fn line_with_arrows_to_path_points(
    points: [f64; 4],
    stroke_width: f64,
    pointer_at_beginning: bool,
    pointer_at_ending: bool,
    pointer_length: f64,
    pointer_width: f64,
) -> Vec<PathCmd> {
    let [x1, y1, x2, y2] = points;
    let half_stroke = stroke_width / 2.0;

    let dx = x2 - x1;
    let dy = y2 - y1;
    let line_length = (dx * dx + dy * dy).sqrt();
    if line_length < 0.001 {
        return vec![PathCmd::Move { x: x1, y: y1 }];
    }

    let unit_x = dx / line_length;
    let unit_y = dy / line_length;
    let perp_x = -unit_y;
    let perp_y = unit_x;

    let body_start_x = x1
        + if pointer_at_beginning {
            unit_x * pointer_length
        } else {
            0.0
        };
    let body_start_y = y1
        + if pointer_at_beginning {
            unit_y * pointer_length
        } else {
            0.0
        };
    let body_end_x = x2
        - if pointer_at_ending {
            unit_x * pointer_length
        } else {
            0.0
        };
    let body_end_y = y2
        - if pointer_at_ending {
            unit_y * pointer_length
        } else {
            0.0
        };

    let mut p: Vec<PathCmd> = Vec::new();
    p.push(PathCmd::Move {
        x: body_start_x - perp_x * half_stroke,
        y: body_start_y - perp_y * half_stroke,
    });

    if pointer_at_beginning {
        p.push(PathCmd::Line {
            x: x1 + unit_x * pointer_length - perp_x * (pointer_width / 2.0),
            y: y1 + unit_y * pointer_length - perp_y * (pointer_width / 2.0),
        });
        p.push(PathCmd::Line { x: x1, y: y1 });
        p.push(PathCmd::Line {
            x: x1 + unit_x * pointer_length + perp_x * (pointer_width / 2.0),
            y: y1 + unit_y * pointer_length + perp_y * (pointer_width / 2.0),
        });
        p.push(PathCmd::Line {
            x: body_start_x + perp_x * half_stroke,
            y: body_start_y + perp_y * half_stroke,
        });
    } else {
        p.push(PathCmd::Line {
            x: body_start_x + perp_x * half_stroke,
            y: body_start_y + perp_y * half_stroke,
        });
    }

    p.push(PathCmd::Line {
        x: body_end_x + perp_x * half_stroke,
        y: body_end_y + perp_y * half_stroke,
    });

    if pointer_at_ending {
        p.push(PathCmd::Line {
            x: x2 - unit_x * pointer_length + perp_x * (pointer_width / 2.0),
            y: y2 - unit_y * pointer_length + perp_y * (pointer_width / 2.0),
        });
        p.push(PathCmd::Line { x: x2, y: y2 });
        p.push(PathCmd::Line {
            x: x2 - unit_x * pointer_length - perp_x * (pointer_width / 2.0),
            y: y2 - unit_y * pointer_length - perp_y * (pointer_width / 2.0),
        });
        p.push(PathCmd::Line {
            x: body_end_x - perp_x * half_stroke,
            y: body_end_y - perp_y * half_stroke,
        });
    } else {
        p.push(PathCmd::Line {
            x: body_end_x - perp_x * half_stroke,
            y: body_end_y - perp_y * half_stroke,
        });
    }

    p
}

/// `elbowedLineToPathPoints` (:1489-1557), no text gap.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn elbowed_line_to_path_points(
    points: [f64; 4],
    waypoints: &[Pt],
    corner_radius: f64,
    stroke_width: f64,
    pointer_at_beginning: bool,
    pointer_at_ending: bool,
    pointer_length: f64,
    pointer_width: f64,
    dash: Option<&[f64]>,
    text_gap: Option<TextGap>,
) -> Vec<PathCmd> {
    let [x1, y1, x2, y2] = points;

    if pointer_at_beginning || pointer_at_ending {
        return elbowed_line_with_arrows_to_path_points(
            points,
            waypoints,
            stroke_width,
            pointer_at_beginning,
            pointer_at_ending,
            pointer_length,
            pointer_width,
        );
    }

    if waypoints.is_empty() {
        // `if (textGap && ...) return lineWithTextGapToPathPoints` (:1509-1510).
        if let Some(gap) = text_gap.filter(TextGap::active) {
            return line_with_text_gap_to_path_points(x1, y1, x2, y2, gap, dash, stroke_width);
        }
        if let Some(d) = dash {
            if !d.is_empty() {
                return dashed_line_to_path_points(x1, y1, x2, y2, d, stroke_width);
            }
        }
        return vec![
            PathCmd::Move { x: x1, y: y1 },
            PathCmd::Line { x: x2, y: y2 },
        ];
    }

    let mut all_points: Vec<Pt> = Vec::with_capacity(waypoints.len() + 2);
    all_points.push([x1, y1]);
    all_points.extend_from_slice(waypoints);
    all_points.push([x2, y2]);

    // Gap beats dash (:1526-1527); the gap carver dash-walks each side.
    if let Some(gap) = text_gap.filter(TextGap::active) {
        return elbowed_line_with_text_gap_to_path_points(
            &all_points,
            gap,
            corner_radius,
            dash,
            stroke_width,
        );
    }

    if let Some(d) = dash {
        if !d.is_empty() {
            return dashed_elbowed_line_to_path_points(&all_points, d, stroke_width, corner_radius);
        }
    }

    let mut path_points: Vec<PathCmd> = Vec::new();
    path_points.push(PathCmd::Move {
        x: all_points[0][0],
        y: all_points[0][1],
    });

    for i in 1..all_points.len() {
        let prev = all_points[i - 1];
        let curr = all_points[i];
        let next = if i < all_points.len() - 1 {
            Some(all_points[i + 1])
        } else {
            None
        };

        if corner_radius > 0.0 && next.is_some() {
            path_points.extend(build_rounded_corner(
                prev,
                curr,
                next.unwrap(),
                corner_radius,
            ));
        } else {
            path_points.push(PathCmd::Line {
                x: curr[0],
                y: curr[1],
            });
        }
    }

    path_points
}

/// `elbowedLineWithTextGapToPathPoints` (:1615-1661): gap start/end are
/// fractions of the RENDERED arc length (line segments + quadratic corner
/// arcs); the rendered path is cut at those distances with De Casteljau
/// splits so the corner radius is preserved up to the gap edge.
fn elbowed_line_with_text_gap_to_path_points(
    all_points: &[Pt],
    gap: TextGap,
    corner_radius: f64,
    dash: Option<&[f64]>,
    stroke_width: f64,
) -> Vec<PathCmd> {
    use super::dash::extract_path_segments_by_arc_length;

    let rendered_segments = build_elbowed_path_segments(all_points, corner_radius);
    let rendered_total: f64 = rendered_segments.iter().map(|s| s.arc_length).sum();

    if rendered_total < 0.001 {
        return Vec::new();
    }

    let gap_start_dist = gap.start_t * rendered_total;
    let gap_end_dist = gap.end_t * rendered_total;

    let before_segments =
        extract_path_segments_by_arc_length(&rendered_segments, 0.0, gap_start_dist);
    let after_segments =
        extract_path_segments_by_arc_length(&rendered_segments, gap_end_dist, rendered_total);

    if let Some(d) = dash {
        if !d.is_empty() {
            let scaled = scale_dash_pattern(d, stroke_width);
            let normalized = normalize_dash_pattern(scaled);
            let pattern_length: f64 = normalized.iter().sum();

            if pattern_length > 0.0 {
                let mut result: Vec<PathCmd> = Vec::new();
                if !before_segments.is_empty() {
                    result.extend(walk_cmds_to_path_cmds(walk_path_by_arc_length(
                        &before_segments,
                        &normalized,
                    )));
                }
                if !after_segments.is_empty() {
                    result.extend(walk_cmds_to_path_cmds(walk_path_by_arc_length(
                        &after_segments,
                        &normalized,
                    )));
                }
                return result;
            }
            // patternLength === 0: fall through to solid rendering below.
        }
    }

    let mut path_points: Vec<PathCmd> = Vec::new();
    path_points.extend(path_segments_to_path_cmds(&before_segments));
    path_points.extend(path_segments_to_path_cmds(&after_segments));
    path_points
}

/// `pathSegmentsToPathPoints` (:981-995): M at the first segment's start,
/// then L/Q/C per segment.
fn path_segments_to_path_cmds(segments: &[PathSegment]) -> Vec<PathCmd> {
    let mut result: Vec<PathCmd> = Vec::new();
    let Some(first) = segments.first() else {
        return result;
    };
    let p0 = match &first.seg {
        Seg::Line { p0, .. } | Seg::Quadratic { p0, .. } | Seg::Cubic { p0, .. } => *p0,
    };
    result.push(PathCmd::Move { x: p0[0], y: p0[1] });
    for seg in segments {
        match &seg.seg {
            Seg::Line { p1, .. } => result.push(PathCmd::Line { x: p1[0], y: p1[1] }),
            Seg::Quadratic { p1, cp, .. } => result.push(PathCmd::Quadratic {
                x: p1[0],
                y: p1[1],
                cp: (cp[0], cp[1]),
            }),
            Seg::Cubic { p1, cp1, cp2, .. } => result.push(PathCmd::Curve {
                x: p1[0],
                y: p1[1],
                cp1: (cp1[0], cp1[1]),
                cp2: (cp2[0], cp2[1]),
            }),
        }
    }
    result
}

/// `dashedElbowedLineToPathPoints` (:1572-1601).
fn dashed_elbowed_line_to_path_points(
    all_points: &[Pt],
    dash_pattern: &[f64],
    stroke_width: f64,
    corner_radius: f64,
) -> Vec<PathCmd> {
    let scaled = scale_dash_pattern(dash_pattern, stroke_width);
    let normalized = normalize_dash_pattern(scaled);
    let pattern_length: f64 = normalized.iter().sum();

    if pattern_length == 0.0 {
        let mut path_points: Vec<PathCmd> = Vec::new();
        path_points.push(PathCmd::Move {
            x: all_points[0][0],
            y: all_points[0][1],
        });
        for p in all_points.iter().skip(1) {
            path_points.push(PathCmd::Line { x: p[0], y: p[1] });
        }
        return path_points;
    }

    let segments = build_elbowed_path_segments(all_points, corner_radius);
    walk_cmds_to_path_cmds(walk_path_by_arc_length(&segments, &normalized))
}

/// `buildRoundedCorner` (:1872-1917).
fn build_rounded_corner(prev: Pt, corner: Pt, next: Pt, radius: f64) -> Vec<PathCmd> {
    let dist_to_prev = (corner[0] - prev[0]).hypot(corner[1] - prev[1]);
    let dist_to_next = (next[0] - corner[0]).hypot(next[1] - corner[1]);

    let max_radius = dist_to_prev.min(dist_to_next) / 2.0;
    let r = radius.min(max_radius);

    if r < 0.001 {
        return vec![PathCmd::Line {
            x: corner[0],
            y: corner[1],
        }];
    }

    let d1x = (corner[0] - prev[0]).signum_js();
    let d1y = (corner[1] - prev[1]).signum_js();
    let d2x = (next[0] - corner[0]).signum_js();
    let d2y = (next[1] - corner[1]).signum_js();

    // Colinear → plain line (:1898-1903).
    if (d1x == d2x && d1y == d2y)
        || (d1x == 0.0 && d2x == 0.0 && d1y == d2y)
        || (d1y == 0.0 && d2y == 0.0 && d1x == d2x)
    {
        return vec![PathCmd::Line {
            x: corner[0],
            y: corner[1],
        }];
    }

    let t1 = [corner[0] - d1x * r, corner[1] - d1y * r];
    let t2 = [corner[0] + d2x * r, corner[1] + d2y * r];

    vec![
        PathCmd::Line { x: t1[0], y: t1[1] },
        PathCmd::Quadratic {
            x: t2[0],
            y: t2[1],
            cp: (corner[0], corner[1]),
        },
    ]
}

/// `elbowedLineWithArrowsToPathPoints` (:1673-1857).
#[allow(clippy::too_many_arguments)]
fn elbowed_line_with_arrows_to_path_points(
    points: [f64; 4],
    waypoints: &[Pt],
    stroke_width: f64,
    pointer_at_beginning: bool,
    pointer_at_ending: bool,
    pointer_length: f64,
    pointer_width: f64,
) -> Vec<PathCmd> {
    let [x1, y1, x2, y2] = points;
    let half_stroke = stroke_width / 2.0;

    let mut all_points: Vec<Pt> = Vec::with_capacity(waypoints.len() + 2);
    all_points.push([x1, y1]);
    all_points.extend_from_slice(waypoints);
    all_points.push([x2, y2]);

    if all_points.len() < 2 {
        return vec![PathCmd::Move { x: x1, y: y1 }];
    }

    struct Perp {
        perp_x: f64,
        perp_y: f64,
        unit_x: f64,
        unit_y: f64,
    }
    let get_perp = |from: Pt, to: Pt| -> Perp {
        let dx = to[0] - from[0];
        let dy = to[1] - from[1];
        let len = dx.hypot(dy);
        if len < 0.001 {
            return Perp {
                perp_x: 0.0,
                perp_y: 0.0,
                unit_x: 0.0,
                unit_y: 0.0,
            };
        }
        let unit_x = dx / len;
        let unit_y = dy / len;
        Perp {
            perp_x: -unit_y,
            perp_y: unit_x,
            unit_x,
            unit_y,
        }
    };

    let mut perps: Vec<Perp> = Vec::with_capacity(all_points.len() - 1);
    for i in 0..all_points.len() - 1 {
        perps.push(get_perp(all_points[i], all_points[i + 1]));
    }

    let mut left_edge: Vec<Pt> = Vec::with_capacity(all_points.len());
    let mut right_edge: Vec<Pt> = Vec::with_capacity(all_points.len());

    for (i, pt) in all_points.iter().enumerate() {
        if i == 0 {
            let mut adjusted = *pt;
            if pointer_at_beginning {
                adjusted = [
                    pt[0] + perps[0].unit_x * pointer_length,
                    pt[1] + perps[0].unit_y * pointer_length,
                ];
            }
            left_edge.push([
                adjusted[0] + perps[0].perp_x * half_stroke,
                adjusted[1] + perps[0].perp_y * half_stroke,
            ]);
            right_edge.push([
                adjusted[0] - perps[0].perp_x * half_stroke,
                adjusted[1] - perps[0].perp_y * half_stroke,
            ]);
        } else if i == all_points.len() - 1 {
            let last_perp = &perps[perps.len() - 1];
            let mut adjusted = *pt;
            if pointer_at_ending {
                adjusted = [
                    pt[0] - last_perp.unit_x * pointer_length,
                    pt[1] - last_perp.unit_y * pointer_length,
                ];
            }
            left_edge.push([
                adjusted[0] + last_perp.perp_x * half_stroke,
                adjusted[1] + last_perp.perp_y * half_stroke,
            ]);
            right_edge.push([
                adjusted[0] - last_perp.perp_x * half_stroke,
                adjusted[1] - last_perp.perp_y * half_stroke,
            ]);
        } else {
            let prev_perp = &perps[i - 1];
            let next_perp = &perps[i];
            left_edge.push([
                pt[0] + prev_perp.perp_x * half_stroke + next_perp.perp_x * half_stroke,
                pt[1] + prev_perp.perp_y * half_stroke + next_perp.perp_y * half_stroke,
            ]);
            right_edge.push([
                pt[0] - prev_perp.perp_x * half_stroke - next_perp.perp_x * half_stroke,
                pt[1] - prev_perp.perp_y * half_stroke - next_perp.perp_y * half_stroke,
            ]);
        }
    }

    let mut p: Vec<PathCmd> = Vec::new();
    p.push(PathCmd::Move {
        x: left_edge[0][0],
        y: left_edge[0][1],
    });
    for e in left_edge.iter().skip(1) {
        p.push(PathCmd::Line { x: e[0], y: e[1] });
    }

    if pointer_at_ending {
        let last_perp = &perps[perps.len() - 1];
        let end_tip = all_points[all_points.len() - 1];
        let arrow_base = [
            end_tip[0] - last_perp.unit_x * pointer_length,
            end_tip[1] - last_perp.unit_y * pointer_length,
        ];
        p.push(PathCmd::Line {
            x: arrow_base[0] + last_perp.perp_x * (pointer_width / 2.0),
            y: arrow_base[1] + last_perp.perp_y * (pointer_width / 2.0),
        });
        p.push(PathCmd::Line {
            x: end_tip[0],
            y: end_tip[1],
        });
        p.push(PathCmd::Line {
            x: arrow_base[0] - last_perp.perp_x * (pointer_width / 2.0),
            y: arrow_base[1] - last_perp.perp_y * (pointer_width / 2.0),
        });
    }

    for e in right_edge.iter().rev() {
        p.push(PathCmd::Line { x: e[0], y: e[1] });
    }

    if pointer_at_beginning {
        let first_perp = &perps[0];
        let start_tip = all_points[0];
        let arrow_base = [
            start_tip[0] + first_perp.unit_x * pointer_length,
            start_tip[1] + first_perp.unit_y * pointer_length,
        ];
        p.push(PathCmd::Line {
            x: arrow_base[0] - first_perp.perp_x * (pointer_width / 2.0),
            y: arrow_base[1] - first_perp.perp_y * (pointer_width / 2.0),
        });
        p.push(PathCmd::Line {
            x: start_tip[0],
            y: start_tip[1],
        });
        p.push(PathCmd::Line {
            x: arrow_base[0] + first_perp.perp_x * (pointer_width / 2.0),
            y: arrow_base[1] + first_perp.perp_y * (pointer_width / 2.0),
        });
    }

    p
}

/// `curvedLineToPathPoints` (lineToPath.ts:1999-2001) — now a THIN DELEGATOR.
///
/// Curved lines follow the same orthogonal avoiding waypoints as elbowed
/// lines, but with generous rounded corners so the stroke reads as a smooth
/// flowing curve while still hugging the shape-avoiding route. We delegate to
/// the elbowed builder with `CURVED_LINE_CORNER_RADIUS`; the rendered path is
/// the waypoint polyline with quadratic-arc fillets at each corner (was a
/// single cubic Bezier that could cut straight through a connected shape).
#[allow(clippy::too_many_arguments)]
fn curved_line_to_path_points(
    points: [f64; 4],
    waypoints: &[Pt],
    stroke_width: f64,
    pointer_at_beginning: bool,
    pointer_at_ending: bool,
    pointer_length: f64,
    pointer_width: f64,
    dash: Option<&[f64]>,
    text_gap: Option<TextGap>,
) -> Vec<PathCmd> {
    elbowed_line_to_path_points(
        points,
        waypoints,
        CURVED_LINE_CORNER_RADIUS,
        stroke_width,
        pointer_at_beginning,
        pointer_at_ending,
        pointer_length,
        pointer_width,
        dash,
        text_gap,
    )
}

fn walk_cmds_to_path_cmds(cmds: Vec<WalkCmd>) -> Vec<PathCmd> {
    cmds.into_iter()
        .map(|c| match c {
            WalkCmd::Move(p) => PathCmd::Move { x: p[0], y: p[1] },
            WalkCmd::Line(p) => PathCmd::Line { x: p[0], y: p[1] },
        })
        .collect()
}

/// M4b: serialize `PathCmd`s as PathNode-style JSON `points` (the array
/// shape `cmds_from_node_points` parses back). Used by the blend-node
/// expansion to feed line-conversion output through the standard path body
/// builder, mirroring how `convertLineToPathNode` returns a PathNode.
pub fn path_cmds_to_json_points(cmds: &[PathCmd]) -> Value {
    use moda_scene::value::json;
    Value::Array(
        cmds.iter()
            .map(|c| match c {
                PathCmd::Move { x, y } => json!({"type": "move", "x": x, "y": y}),
                PathCmd::Line { x, y } => json!({"type": "line", "x": x, "y": y}),
                PathCmd::Curve { x, y, cp1, cp2 } => json!({
                    "type": "curve", "x": x, "y": y,
                    "cp1": {"x": cp1.0, "y": cp1.1},
                    "cp2": {"x": cp2.0, "y": cp2.1},
                }),
                PathCmd::Quadratic { x, y, cp } => json!({
                    "type": "quadratic", "x": x, "y": y,
                    "cp": {"x": cp.0, "y": cp.1},
                }),
                PathCmd::Arc {
                    x,
                    y,
                    rx,
                    ry,
                    x_axis_rotation,
                    large_arc,
                    sweep,
                } => json!({
                    "type": "arc", "x": x, "y": y, "rx": rx, "ry": ry,
                    "xAxisRotation": x_axis_rotation,
                    "largeArc": large_arc, "sweep": sweep,
                }),
            })
            .collect(),
    )
}

/// JS `Math.sign` (±0 stays ±0; NaN stays NaN).
trait SignumJs {
    fn signum_js(self) -> f64;
}
impl SignumJs for f64 {
    fn signum_js(self) -> f64 {
        if self > 0.0 {
            1.0
        } else if self < 0.0 {
            -1.0
        } else {
            self // ±0 / NaN pass through like Math.sign
        }
    }
}

#[cfg(test)]
mod null_arrow_quirk_tests {
    use super::*;
    use moda_scene::value::json;

    fn spec_from(v: moda_scene::value::Value) -> LineSpec {
        let obj = v.as_object().unwrap().clone();
        LineSpec::from_json(|k| obj.get(k))
    }

    /// Q-91 is gone with the paint stacks: the arrowheads live on the stroke
    /// (`startArrow` / `endArrow`), and an explicit `null` arrow is no arrow,
    /// so the line stays a plain stroked line (it used to flip into
    /// filled-silhouette mode and paint nothing).
    #[test]
    fn explicit_null_arrow_keeps_the_stroke() {
        let spec = spec_from(json!({
            "points": [80.0, 488.0, 880.0, 488.0],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#fce8ef"}, "width": 1.5,
                         "cap": "round", "join": "round", "startArrow": null, "endArrow": null, "dash": null}]
        }));
        assert!(!spec.start_arrow_defined && !spec.end_arrow_defined);
        assert_eq!(spec.start_arrow, None);
        let part = convert_line_to_path(&spec);
        assert!(!part.closed);
        assert_eq!(part.stroke_width, 1.5);
    }

    /// Absent keys stay a plain stroked line (synthetic-fixture shape).
    #[test]
    fn absent_arrow_keys_keep_the_stroke() {
        let spec = spec_from(json!({
            "points": [0.0, 0.0, 100.0, 0.0],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": 1.5}]
        }));
        assert!(!spec.start_arrow_defined && !spec.end_arrow_defined);
        let part = convert_line_to_path(&spec);
        assert!(!part.closed);
        assert_eq!(part.stroke_width, 1.5);
    }
}
