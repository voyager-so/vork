//! Stroke-side inputs, results and the JSON envelope shared by the art,
//! scatter and pattern placers.

use crate::value::{json, Map, Value};

use super::colorize::{self, ColorizeCtx};
use super::geom::{to_points, Bounds, Subpath};
use super::PlacedPath;
use crate::path_geom::vec2::Cubic;
use crate::stroke_model::{InkChannels, PreparedWidthProfile};

/// Most stroke-path points accepted (a fitted Pencil stroke has hundreds).
pub const MAX_PATH_POINTS: usize = 20_000;
/// Most shapes a JSON response may carry (the WASM lane: each becomes a node
/// the editor parses). The engine's direct calls are bounded by the placers'
/// own budgets instead (10 000 scatter copies / pattern tiles, 60 000 segments).
pub const MAX_OUTPUT_PATHS: usize = 4_000;
/// Largest JSON response `placement_json` writes.
pub const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
/// Largest request accepted, checked before parsing (20 000 path points and a
/// generous artwork fit well inside).
pub const MAX_REQUEST_BYTES: usize = 4 * 1024 * 1024;
/// Work budget per placement, in warped pieces (≈ a dozen map evaluations
/// each): keeps the worst case near 50 ms in WASM.
pub const MAX_WORK_PIECES: usize = 20_000;
/// Most output segments per placement (scatter copies, pattern tiles).
pub const MAX_OUTPUT_SEGMENTS: usize = 60_000;

/// The stroke a brush is applied to.
#[derive(Debug, Clone)]
pub struct BrushStroke<'a> {
    /// The drawn centerline (after `cornerRadius`), node-local.
    pub subpaths: Vec<Subpath>,
    pub stroke_width: f64,
    /// The node's `stroke` paint (colourisation needs a colour string).
    pub stroke_color: Option<&'a Value>,
    /// Used only with `overrides.widthFrom = 'profile'`.
    pub profile: Option<&'a PreparedWidthProfile>,
    pub ink: Option<&'a InkChannels>,
    pub seed: u32,
    /// Per subpath: its arc length in the outliner's measure (the one trim /
    /// dash windows and `PlacedAt` use). Empty = each subpath's own length.
    pub measured: Vec<f64>,
}

/// Where a placed shape sits along the stroke, for per-frame trim / dash
/// filtering from a cached full placement. Arc lengths are in the stroke
/// outliner's measure (`stroke_outline::subpath_spans`, the one
/// `ArcWindow`s use), from the start of subpath `subpath`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlacedAt {
    pub subpath: usize,
    /// The shape's anchor: a scatter copy's centre, a pattern tile's centre,
    /// the middle of an art brush's (windowed) run.
    pub s: f64,
    /// The arc-length extent it covers: a copy's or tile's length along the
    /// path (on a closed subpath `s_span.1` may exceed the length), or the
    /// art's run.
    pub s_span: (f64, f64),
}

/// Result of a placement.
#[derive(Debug, Clone, Default)]
pub struct Placement {
    pub paths: Vec<PlacedPath>,
    /// Artwork nodes that had no geometry.
    pub skipped: Vec<String>,
    /// Bounds of the output control points.
    pub bounds: Bounds,
    /// Non-fatal notes (e.g. `guides_ignored: …`), passed through to the envelope.
    pub warnings: Vec<String>,
    /// Set when a work budget ran out: the placement is refused (`too_complex`).
    pub too_complex: Option<String>,
}

std::thread_local! {
    /// Shape cap while serving a JSON request (`with_json_cap`); unlimited for
    /// the engine's direct calls.
    static JSON_MAX_PATHS: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

/// Run a placement for the JSON / WASM lane: `emit` stops (and refuses the
/// placement) at `MAX_OUTPUT_PATHS` shapes before copying any more paint.
pub(crate) fn with_json_cap<T>(f: impl FnOnce() -> T) -> T {
    JSON_MAX_PATHS.with(|c| c.set(MAX_OUTPUT_PATHS));
    let out = f();
    JSON_MAX_PATHS.with(|c| c.set(usize::MAX));
    out
}

impl Placement {
    /// Append one placed shape: copies the paint, scales `strokeWidth` by
    /// `stroke_scale`, colourises, and grows the bounds.
    #[allow(clippy::too_many_arguments)]
    pub fn emit(
        &mut self,
        id: String,
        source_id: &str,
        paint: &Map,
        subpaths: Vec<Subpath>,
        stroke_scale: f64,
        ctx: Option<&ColorizeCtx>,
        at: PlacedAt,
    ) {
        if subpaths.is_empty() || self.too_complex.is_some() {
            return;
        }
        let cap = JSON_MAX_PATHS.with(|c| c.get());
        if self.paths.len() >= cap {
            self.too_complex = Some(format!(
                "the brush would place more than {cap} shapes (raise spacing or size)"
            ));
            return;
        }
        let mut paint = paint.clone();
        crate::paint::scale_stroke_widths(&mut paint, stroke_scale);
        if let Some(ctx) = ctx {
            colorize::apply_to_node(&mut paint, ctx);
        }
        for sp in &subpaths {
            for c in &sp.segs {
                for p in c.0 {
                    self.bounds.add(p);
                }
            }
        }
        self.paths.push(PlacedPath {
            id,
            source_id: source_id.to_string(),
            closed: subpaths.iter().all(|s| s.closed),
            points: to_points(&subpaths),
            paint,
            at,
        });
    }
}

pub(crate) fn subpath_length(sp: &Subpath) -> f64 {
    sp.segs
        .iter()
        .map(|c| c.arc_len())
        .filter(|l| l.is_finite())
        .sum()
}

/// The stroke of a path node: its drawn centerline (`cornerRadius` applied),
/// weight, colour, width profile, ink and seed.
pub struct NodeStroke {
    /// Index-aligned with `stroke_outline::subpath_spans` (zero-length
    /// subpaths are kept, empty).
    pub subpaths: Vec<Subpath>,
    /// Per subpath: the outliner's arc length.
    pub measured: Vec<f64>,
    pub stroke_width: f64,
    pub stroke_color: Option<Value>,
    pub profile: Option<PreparedWidthProfile>,
    pub ink: Option<InkChannels>,
    pub seed: u32,
    pub overrides: Option<Map>,
    /// The model stroke's own opacity (bristle strands multiply it in).
    pub stroke_opacity: f64,
}

/// Read the stroke-side inputs from a path node (node-local coordinates).
pub fn node_stroke(node: &Map) -> NodeStroke {
    // The accessors below read only these fields: don't clone the points.
    // Placers only take paths: a request without `type` (the editor's WASM lane sends only the fields the
    // placers read) is a path, so its stroke (weight, colour) is read.
    let mut raw = crate::types::RawNode::new(
        ["type", "strokes", "inkChannels"]
            .iter()
            .filter_map(|k| node.get(*k).map(|v| (k.to_string(), v.clone())))
            .collect(),
    );
    raw.fields_mut()
        .entry("type".to_string())
        .or_insert_with(|| Value::String("path".into()));
    let points_json: &[Value] = node
        .get("points")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let closed = node.get("closed").and_then(Value::as_bool).unwrap_or(false);
    let corners = crate::path_geom::round_corners::PathCorners::of_json(node);
    // The outliner's segmentation (cornerRadius, closing segments, invalid
    // points), so subpath indices and arc lengths match `subpath_spans` and
    // the trim / dash windows the engine derives from it.
    let points_in = &points_json[..points_json.len().min(MAX_PATH_POINTS)];
    let (seg_paths, _, _) =
        crate::stroke_outline::centerline::node_segments(points_in, closed, &corners);
    let subpaths: Vec<Subpath> = seg_paths
        .iter()
        .map(|sp| Subpath {
            segs: sp
                .segs
                .iter()
                .map(|s| match s {
                    crate::stroke_outline::centerline::Seg::Line(a, b) => Cubic::line(*a, *b),
                    crate::stroke_outline::centerline::Seg::Cubic(c) => *c,
                })
                .collect(),
            closed: sp.closed,
        })
        .collect();
    let measured: Vec<f64> = seg_paths.iter().map(|sp| sp.lens.iter().sum()).collect();
    let profile = raw.stroke_profile().map(|p| {
        let fr = crate::path_length::anchor_fractions(points_json, closed);
        PreparedWidthProfile::new_with(&p, Some(&fr.as_options()), fr.is_periodic(closed))
    });
    let brush = raw.stroke_brush();
    // The model stroke (`paint::model_stroke`): its width, and the colour the
    // brush colourises against (a solid's colour, a gradient's first stop).
    let stroke = crate::paint::model_stroke(&raw);
    NodeStroke {
        subpaths,
        measured,
        stroke_width: stroke
            .as_ref()
            .map(|s| s.width)
            .filter(|v| v.is_finite())
            .unwrap_or(1.0),
        stroke_color: stroke
            .as_ref()
            .and_then(|s| crate::paint::representative_color(&s.paint))
            .map(|c| Value::String(c.to_string())),

        profile,
        ink: raw.ink_channels(),
        seed: brush.as_ref().and_then(|b| b.seed).unwrap_or(0),
        overrides: brush.and_then(|b| b.overrides),
        stroke_opacity: stroke
            .as_ref()
            .and_then(|s| s.opacity)
            .filter(|o| o.is_finite())
            .unwrap_or(1.0)
            .clamp(0.0, 1.0),
    }
}

pub(crate) fn error(code: &str, message: &str) -> String {
    json!({"status": "error", "code": code, "message": message}).to_string()
}

/// A validated request: `{ node, brush, tolerance? }`.
pub(crate) struct Request {
    pub node: Map,
    pub brush: Value,
    pub tolerance: Option<f64>,
    /// `windows: [{subpath, start, end}]` (art brushes: trim / dash reveal).
    pub windows: Option<Vec<crate::stroke_outline::ArcWindow>>,
}

/// Validate the common request shape; `Err` is a ready error envelope.
pub(crate) fn parse_request(request_json: &str) -> Result<Request, String> {
    if request_json.len() > MAX_REQUEST_BYTES {
        return Err(error(
            "request_too_large",
            &format!("request exceeds {} MiB", MAX_REQUEST_BYTES >> 20),
        ));
    }
    let value: Value = crate::value::from_str(request_json)
        .map_err(|e| error("invalid_json", &format!("request is not valid JSON: {e}")))?;
    request_from_value(value)
}

/// [`parse_request`] for an already-parsed request.
pub(crate) fn request_from_value(value: Value) -> Result<Request, String> {
    let Value::Object(mut obj) = value else {
        return Err(error("invalid_request", "request must be an object"));
    };
    let Some(Value::Object(node)) = obj.remove("node") else {
        return Err(error("invalid_request", "node must be an object"));
    };
    let Some(points) = node.get("points").and_then(Value::as_array) else {
        return Err(error("invalid_request", "node.points must be an array"));
    };
    if points.len() > MAX_PATH_POINTS {
        return Err(error(
            "too_many_points",
            &format!("at most {MAX_PATH_POINTS} path points"),
        ));
    }
    let Some(brush) = obj.remove("brush") else {
        return Err(error("invalid_brush", "brush is required"));
    };
    let tolerance = obj.get("tolerance").and_then(Value::as_f64);
    let windows = match obj.get("windows") {
        None | Some(Value::Null) => None,
        Some(Value::Array(list)) => {
            if list.len() > 100_000 {
                return Err(error("invalid_request", "at most 100000 windows"));
            }
            let mut out = Vec::with_capacity(list.len());
            for w in list {
                let f = |k: &str| w.get(k).and_then(Value::as_f64).filter(|v| v.is_finite());
                let (Some(subpath), Some(start), Some(end)) = (f("subpath"), f("start"), f("end"))
                else {
                    return Err(error(
                        "invalid_request",
                        "windows[] must be {subpath, start, end} numbers",
                    ));
                };
                if subpath < 0.0 || subpath.fract() != 0.0 {
                    return Err(error(
                        "invalid_request",
                        "windows[].subpath must be a subpath index",
                    ));
                }
                out.push(crate::stroke_outline::ArcWindow {
                    subpath: subpath as usize,
                    start,
                    end,
                    taper_extent: None,
                });
            }
            Some(out)
        }
        Some(_) => return Err(error("invalid_request", "windows must be an array")),
    };
    Ok(Request {
        node,
        brush,
        tolerance,
        windows,
    })
}

/// The response envelope for a placement.
pub fn placement_json(placed: &Placement) -> String {
    use std::fmt::Write;
    if let Some(reason) = &placed.too_complex {
        return error("too_complex", reason);
    }
    if placed.paths.len() > MAX_OUTPUT_PATHS {
        return error(
            "too_complex",
            &format!(
                "the brush would place {} shapes, more than {MAX_OUTPUT_PATHS} (raise spacing or size)",
                placed.paths.len()
            ),
        );
    }
    let b = &placed.bounds;
    if !placed.paths.is_empty()
        && ![b.min.x, b.min.y, b.max.x, b.max.y]
            .iter()
            .all(|v| v.is_finite())
    {
        return error(
            "too_complex",
            "the placed geometry has non-finite coordinates",
        );
    }
    // Written directly rather than through `crate::value::Value`: building a
    // Value per point (preserve_order maps) cost more than the placement.
    let js = |v: &Value| v.to_string();
    let mut s = String::with_capacity(64 + placed.paths.len() * 256);
    if placed.paths.is_empty() {
        s.push_str(r#"{"status":"empty""#);
    } else {
        s.push_str(r#"{"status":"ok","nodes":["#);
        for (i, p) in placed.paths.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            p.write_node_json(&mut s);
        }
        let b = &placed.bounds;
        s.push_str(r#"],"bounds":{"minX":"#);
        num(&mut s, b.min.x);
        s.push_str(r#","minY":"#);
        num(&mut s, b.min.y);
        s.push_str(r#","maxX":"#);
        num(&mut s, b.max.x);
        s.push_str(r#","maxY":"#);
        num(&mut s, b.max.y);
        s.push('}');
    }
    let _ = write!(s, r#","skipped":{}"#, js(&json!(placed.skipped)));
    if !placed.warnings.is_empty() {
        let _ = write!(s, r#","warnings":{}"#, js(&json!(placed.warnings)));
    }
    s.push('}');
    if s.len() > MAX_RESPONSE_BYTES {
        return error(
            "too_complex",
            &format!(
                "the placement exceeds {} MiB (raise spacing or size, or simplify the artwork)",
                MAX_RESPONSE_BYTES >> 20
            ),
        );
    }
    s
}

/// A finite f64 as a JSON number (shortest round-trip form; non-finite → 0).
pub(crate) fn num(s: &mut String, v: f64) {
    use std::fmt::Write;
    if v.is_finite() {
        let _ = write!(s, "{v}");
    } else {
        s.push('0');
    }
}

/// A brush's `scaleWithWeight` flag (Illustrator's sizing: the stroke weight
/// multiplies the designed size). Absent or anything but `true`: the stroke
/// width is the width of the mark (the default, as in Figma).
pub(crate) fn scale_with_weight(o: &Map) -> bool {
    o.get("scaleWithWeight").and_then(Value::as_bool) == Some(true)
}

/// The factor from stroke width to the brush's scale: 1 under
/// `scaleWithWeight`, else `1 / designed` (the design's width across the
/// path), so the mark is as wide as the stroke.
pub(crate) fn weight_unit(scale_with_weight: bool, designed: f64) -> f64 {
    if scale_with_weight || !designed.is_finite() || designed <= 1e-9 {
        1.0
    } else {
        1.0 / designed
    }
}
