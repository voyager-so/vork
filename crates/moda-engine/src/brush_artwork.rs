//! Brush artwork geometry (brushes.md §5, plan P17–P19): the node → path
//! callback `moda_scene::stroke_brush` uses to turn art / scatter / pattern
//! brush artwork leaves into paths. It lives here because the full shape
//! conversions do (`shape_to_path`, `tess::shape_paths`); moda-scene's own
//! `default_geometry` covers only paths, uniform-radius rectangles and full
//! ellipses. The WASM `stroke_brush` export passes this; the engine's
//! brushed-stroke rendering should pass it too, so canvas and export agree.
//!
//! Handled: `path` (with `cornerRadius`), `rectangle` (per-corner radii),
//! `ellipse` (arcs and rings), `polygon`, `star`. Anything else (text, images,
//! lines, frames) returns `None` and is reported in the placement's `skipped`.

use moda_scene::blend::BlendPathPoint;
use moda_scene::value::{Map, Value};

use crate::shape_to_path;
use crate::tess;

/// Node-local `(points, closed)` for one artwork leaf, or `None`.
pub fn brush_artwork_geometry(node: &Map) -> Option<(Vec<BlendPathPoint>, bool)> {
    let f = |k: &str| {
        node.get(k)
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
    };
    let kind = node.get("type").and_then(Value::as_str)?;
    if kind == "path" {
        return moda_scene::stroke_brush::artwork::default_geometry(node);
    }
    // Shapes need their box (as `default_geometry` requires): skip, don't guess.
    let (w, h) = (f("width")?, f("height")?);
    let cr = f("cornerRadius").unwrap_or(0.0);
    let cmds = match kind {
        "rectangle" => shape_to_path::rect_outline_cmds(w, h, node.get("cornerRadius"), node.get("cornerSmoothing"), node.get("cornerStyle")),
        "ellipse" => shape_to_path::ellipse_to_path_cmds(
            w,
            h,
            f("startAngle"),
            f("sweepAngle"),
            f("innerRadiusRatio"),
        ),
        "polygon" => {
            tess::shape_paths::polygon_to_path_cmds(w, h, f("sides").unwrap_or(f64::NAN), cr)
        }
        "star" => tess::shape_paths::star_to_path_cmds(
            w,
            h,
            f("numPoints").unwrap_or(f64::NAN),
            f("innerRadiusRatio"),
            cr,
        ),
        _ => return None,
    };
    let json = tess::line_to_path::path_cmds_to_json_points(&cmds);
    let points = json
        .as_array()?
        .iter()
        .filter_map(BlendPathPoint::from_json)
        .collect();
    Some((points, true))
}

/// `moda_scene::stroke_brush::stroke_brush_json` with this geometry: the WASM
/// `stroke_brush` export's implementation.
///
/// A request with `"visible": true` places only what the canvas draws: the
/// node's trim (request-level `trimPathStart` / `trimPathEnd` /
/// `trimPathOffset` / `trimPathMode` / `trimPathBy`, as the `stroke_outline` request takes
/// them) and its stroke's dash, resolved to arc-length windows exactly as the
/// renderer resolves them (`tess::outline::arc_windows_on`, the node's dash
/// alignment). Art and pattern brushes clip their artwork to the windows in
/// the placer; scatter keeps the copies whose anchor lies in a window
/// (`tess::placed::in_windows`) — the engine's rules, so export and Expand
/// Appearance show what the canvas shows.
pub fn stroke_brush_json(request_json: &str) -> String {
    let geometry = &brush_artwork_geometry;
    let place = |json: &str| moda_scene::stroke_brush::stroke_brush_json_with(json, geometry);
    let Ok(Value::Object(mut req)) = moda_scene::value::from_str::<Value>(request_json) else {
        return place(request_json);
    };
    if req.remove("visible") != Some(Value::Bool(true)) {
        return place(request_json);
    }
    let trim = tess::trim_params_from_fields(
        |k| req.get(k).and_then(Value::as_f64),
        req.get("trimPathMode").and_then(Value::as_str),
        req.get("trimPathBy").and_then(Value::as_str),
    );
    for k in ["trimPathStart", "trimPathEnd", "trimPathOffset", "trimPathMode", "trimPathBy"] {
        req.remove(k);
    }
    let kind = req
        .get("brush")
        .and_then(|b| b.get("type"))
        .and_then(Value::as_str)
        .and_then(tess::placed::PlacedKind::from_brush_type);
    let node = match req.get("node") {
        Some(Value::Object(m)) => {
            let mut m = m.clone();
            m.entry("type").or_insert_with(|| Value::from("path"));
            moda_scene::types::RawNode::new(m)
        }
        _ => return place(&Value::Object(req).to_string()),
    };
    let style = moda_scene::paint::node_stroke_style(&node);
    let dash = style.dash.clone().filter(|d| !d.is_empty());
    let (Some(kind), true) = (kind, trim.is_some() || dash.is_some()) else {
        return place(&Value::Object(req).to_string());
    };
    let spans = moda_scene::stroke_outline::subpath_spans(&node);
    let (windows, _) = tess::with_dash_alignment(style.dash_aligned(), || {
        tess::outline::arc_windows_on(
            &spans,
            &node,
            trim,
            dash.as_deref(),
            style.dash_offset.unwrap_or(0.0),
            crate::paint::painted_stroke_width(&node),
        )
    });
    let Some(windows) = windows else {
        return place(&Value::Object(req).to_string());
    };
    if windows.is_empty() {
        return r#"{"status":"empty"}"#.to_string();
    }
    if tess::placed::windows_in_placer(kind) {
        req.insert(
            "windows".into(),
            Value::Array(
                windows
                    .iter()
                    .map(|w| moda_scene::value::json!({"subpath": w.subpath, "start": w.start, "end": w.end}))
                    .collect(),
            ),
        );
        return place(&Value::Object(req).to_string());
    }
    // Copies shown or hidden whole: filter the full placement.
    let full = place(&Value::Object(req).to_string());
    let Ok(Value::Object(mut out)) = moda_scene::value::from_str::<Value>(&full) else {
        return full;
    };
    if out.get("status").and_then(Value::as_str) != Some("ok") {
        return full;
    }
    let nodes = match out.remove("nodes") {
        Some(Value::Array(nodes)) => nodes,
        _ => return full,
    };
    let kept: Vec<Value> = nodes
        .into_iter()
        .filter(|n| {
            let (Some(subpath), Some(s)) = (
                n.get("subpath").and_then(Value::as_u64),
                n.get("s").and_then(Value::as_f64),
            ) else {
                return true;
            };
            let at = moda_scene::stroke_brush::common::PlacedAt {
                subpath: subpath as usize,
                s,
                s_span: (s, s),
            };
            tess::placed::in_windows(&at, &windows, &spans)
        })
        .collect();
    if kept.is_empty() {
        let mut empty = moda_scene::value::Map::new();
        empty.insert("status".into(), Value::from("empty"));
        for k in ["skipped", "warnings"] {
            if let Some(v) = out.remove(k) {
                empty.insert(k.into(), v);
            }
        }
        return Value::Object(empty).to_string();
    }
    // Bounds of the kept shapes' control points (as the placer reports them).
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    let mut grow = |p: &Value| {
        if let (Some(x), Some(y)) = (p.get("x").and_then(Value::as_f64), p.get("y").and_then(Value::as_f64)) {
            b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
        }
    };
    for n in &kept {
        for p in n.get("points").and_then(Value::as_array).into_iter().flatten() {
            grow(p);
            if let Some(c) = p.get("cp1") {
                grow(c);
            }
            if let Some(c) = p.get("cp2") {
                grow(c);
            }
        }
    }
    out.insert("nodes".into(), Value::Array(kept));
    out.insert(
        "bounds".into(),
        moda_scene::value::json!({"minX": b[0], "minY": b[1], "maxX": b[2], "maxY": b[3]}),
    );
    Value::Object(out).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    fn geom(v: Value) -> Option<(Vec<BlendPathPoint>, bool)> {
        brush_artwork_geometry(v.as_object().unwrap())
    }

    #[test]
    fn shapes_have_geometry() {
        for n in [
            json!({"type": "rectangle", "width": 10, "height": 5, "cornerRadius": [1, 2, 3, 0]}),
            json!({"type": "ellipse", "width": 10, "height": 5, "innerRadiusRatio": 0.5}),
            json!({"type": "polygon", "width": 10, "height": 10, "sides": 6}),
            json!({"type": "star", "width": 10, "height": 10, "numPoints": 5}),
            json!({"type": "path", "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 3, "y": 1}]}),
        ] {
            let (pts, _) = geom(n.clone()).unwrap_or_else(|| panic!("{n}"));
            assert!(pts.len() >= 2, "{n}");
        }
        assert!(geom(json!({"type": "text"})).is_none());
        assert!(
            geom(json!({"type": "rectangle", "width": 10})).is_none(),
            "no height: skipped"
        );
        assert!(geom(json!({"type": "star", "numPoints": 5})).is_none());
    }

    #[test]
    fn absurd_vertex_counts_do_not_stall_or_abort() {
        // Regression (#62 review): 1e20 overflowed, 1e7 × 20 polygons took minutes.
        let many: Vec<Value> = (0..4)
            .map(|i| {
                json!({"id": format!("p{i}"), "type": if i % 2 == 0 { "polygon" } else { "star" },
                "width": 10, "height": 10, "sides": 1e20, "numPoints": 1e9})
            })
            .chain([
                json!({"id": "nan", "type": "polygon", "width": 10, "height": 10, "sides": null}),
            ])
            .collect();
        let req = json!({
            "node": {"points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 20, "y": 0}]},
            "brush": {"id": "s", "type": "scatter", "artwork": {"nodes": many}},
        });
        // No abort, and the clamp keeps each shape to ≤ 4096 vertices.
        let v: Value = moda_scene::value::from_str(&stroke_brush_json(&req.to_string())).unwrap();
        assert!(v["status"] == "ok" || v["status"] == "error", "{v}");
        let (pts, _) = brush_artwork_geometry(many[0].as_object().unwrap()).unwrap();
        assert!(pts.len() <= tess::shape_paths::MAX_SHAPE_VERTICES + 2);
        assert!(
            brush_artwork_geometry(many[4].as_object().unwrap()).map_or(true, |g| g.0.is_empty())
        );
    }

    fn line_request(brush: Value, extra: Value, stroke: Value) -> Value {
        let mut req = json!({
            "node": {"points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0}],
                     "strokes": [stroke]},
            "brush": brush,
        });
        for (k, v) in extra.as_object().unwrap() {
            req[k] = v.clone();
        }
        req
    }

    fn square_artwork() -> Value {
        json!({"nodes": [{"id": "sq", "type": "rectangle", "width": 10, "height": 10,
            "fills": [{"id": "f", "paint": {"type": "solid", "color": "#f00"}}]}]})
    }

    fn placed(req: &Value) -> Value {
        moda_scene::value::from_str(&stroke_brush_json(&req.to_string())).unwrap()
    }

    fn max_x(v: &Value) -> f64 {
        v["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|n| n["points"].as_array().unwrap().iter())
            .map(|p| p["x"].as_f64().unwrap())
            .fold(f64::NEG_INFINITY, f64::max)
    }

    #[test]
    fn a_request_without_node_type_keeps_the_strokes_width_and_colour() {
        // The editor sends only the placer's node fields (no `type`): the stroke must still size and colourise.
        let brush = json!({"id": "a", "type": "art", "artwork": square_artwork(), "colorize": {"method": "tints"}});
        let stroke = json!({"id": "s0", "width": 4, "paint": {"type": "solid", "color": "#00ff00"}});
        let req = line_request(brush, json!({}), stroke);
        assert!(req["node"].get("type").is_none());
        let v = placed(&req);
        assert_eq!(v["status"], "ok", "{v}");
        assert_ne!(v["nodes"][0]["fills"][0]["paint"]["color"], "#f00", "tinted with the stroke colour");
        let h = v["bounds"]["maxY"].as_f64().unwrap() - v["bounds"]["minY"].as_f64().unwrap();
        // Sized by the stroke width (no scaleWithWeight): the 10 px art is 4 px across.
        assert!((h - 4.0).abs() < 1e-6, "10 px art at width 4: {h}");
    }

    #[test]
    fn visible_scatter_keeps_whole_copies_inside_the_trim() {
        let stroke = json!({"id": "s0", "width": 1, "paint": {"type": "solid", "color": "#000"}});
        let brush = json!({"id": "s", "type": "scatter", "artwork": square_artwork()});
        let full = placed(&line_request(brush.clone(), json!({}), stroke.clone()));
        let all = full["nodes"].as_array().unwrap().len();
        // Without `visible`, trim fields are ignored (the whole stroke).
        let ignored = placed(&line_request(brush.clone(), json!({"trimPathEnd": 0.5}), stroke.clone()));
        assert_eq!(ignored["nodes"].as_array().unwrap().len(), all);
        let half = placed(&line_request(brush, json!({"visible": true, "trimPathEnd": 0.5}), stroke));
        let kept = half["nodes"].as_array().unwrap();
        assert!(kept.len() < all && !kept.is_empty(), "{} of {all}", kept.len());
        for n in kept {
            assert!(n["s"].as_f64().unwrap() < 50.0 + 1e-9, "{n}");
        }
        // Whole copies: the last one is not clipped at x = 50, and the bounds follow the kept copies.
        assert!(half["bounds"]["maxX"].as_f64().unwrap() <= 60.0);
        assert!((half["bounds"]["maxX"].as_f64().unwrap() - max_x(&half)).abs() < 1e-6);
    }

    #[test]
    fn visible_art_clips_to_the_dash_windows_and_empty_trim_is_empty() {
        let brush = json!({"id": "a", "type": "art", "artwork": square_artwork()});
        let dashed = json!({"id": "s0", "width": 2, "dash": [20, 30], "paint": {"type": "solid", "color": "#000"}});
        let v = placed(&line_request(brush.clone(), json!({"visible": true}), dashed));
        assert_eq!(v["status"], "ok", "{v}");
        // Dashes scale with width / 2: at width 2, [0, 20) and [50, 70) on 100 px; nothing past x = 70.
        assert!(max_x(&v) <= 70.0 + 1e-6, "{}", max_x(&v));
        assert!(v["nodes"].as_array().unwrap().len() >= 2);
        let plain = json!({"id": "s0", "width": 10, "paint": {"type": "solid", "color": "#000"}});
        let none = placed(&line_request(
            brush,
            json!({"visible": true, "trimPathStart": 0.4, "trimPathEnd": 0.4}),
            plain,
        ));
        assert_eq!(none["status"], "empty", "{none}");
    }

    #[test]
    fn star_artwork_places_along_a_stroke() {
        let req = json!({
            "node": {"points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0}]},
            "brush": {"id": "s", "type": "scatter", "scaleWithWeight": true, "artwork": {"nodes": [
                {"id": "st", "type": "star", "width": 10, "height": 10, "numPoints": 5, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]}]}},
        });
        let v: Value = moda_scene::value::from_str(&stroke_brush_json(&req.to_string())).unwrap();
        assert_eq!(v["status"], "ok");
        // A 5-point star in a 10 × 10 box is ~9.5 wide: 11 copies touch on 100 px.
        assert_eq!(v["nodes"].as_array().unwrap().len(), 11);
    }
}
