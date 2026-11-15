//! Art brush tests: mapping conventions, fits, flips, overlap, widths,
//! colourisation, envelopes, determinism, fuzz and a perf probe.

use crate::value::{json, Value};

use super::art::*;
use super::artwork::default_geometry;
use super::common::PlacedAt;
use super::raster::polygons;
use super::PlacedPath;
use crate::path_geom::vec2::{v2, V2};

pub fn rect_art(w: f64, h: f64) -> Value {
    json!([{"id": "r", "type": "rectangle", "x": 0, "y": 0, "width": w, "height": h, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}]}])
}

/// A triangle pointing right (+x): tip at (100, 10).
pub fn arrow_art() -> Value {
    json!([{"id": "a", "type": "path", "x": 0, "y": 0, "closed": true, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#d03a2f"}}],
        "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 10},
                   {"type": "line", "x": 0, "y": 20}]}])
}

pub fn brush(artwork: Value, extra: Value) -> Value {
    let mut b = json!({"scaleWithWeight": true, "id": "b", "name": "B", "type": "art", "artwork": {"nodes": artwork},
        "scale": 1, "fit": "stretch", "direction": "right", "overlap": "none",
        "colorize": {"method": "none"}});
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

/// A path node with one stroke layer. `extra` sets node fields (`closed`,
/// `cornerRadius`, `inkChannels`) and otherwise the stroke layer's own fields
/// (`width`, `paint`, `profile`, `brush`, …).
pub fn line_node(points: Value, extra: Value) -> Value {
    let mut n = json!({"id": "n", "type": "path", "x": 0, "y": 0, "points": points,
        "closed": false, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#2a9d4a"}, "width": 0.5}]});
    for (k, v) in extra.as_object().unwrap() {
        if ["closed", "cornerRadius", "inkChannels"].contains(&k.as_str()) {
            n[k] = v.clone();
        } else {
            n["strokes"][0][k] = v.clone();
        }
    }
    n
}

pub fn straight(len: f64) -> Value {
    json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": len, "y": 0}])
}

pub fn place(node: &Value, brush: &Value) -> Placement {
    let b = parse_art_brush(brush).expect("art brush");
    art_brush_for_node(
        node.as_object().unwrap(),
        &b,
        &default_geometry,
        &ArtOptions::default(),
    )
}

fn all_points(paths: &[PlacedPath]) -> Vec<V2> {
    paths
        .iter()
        .flat_map(|p| polygons(&p.points).concat())
        .collect()
}

fn bbox(pts: &[V2]) -> (V2, V2) {
    let mut lo = v2(f64::INFINITY, f64::INFINITY);
    let mut hi = v2(f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in pts {
        lo = v2(lo.x.min(p.x), lo.y.min(p.y));
        hi = v2(hi.x.max(p.x), hi.y.max(p.y));
    }
    (lo, hi)
}

fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// Point of the placed art furthest along +x (for the arrow's tip).
fn extreme(paths: &[PlacedPath], key: impl Fn(V2) -> f64) -> V2 {
    all_points(paths)
        .into_iter()
        .max_by(|a, b| key(*a).total_cmp(&key(*b)))
        .unwrap()
}

#[test]
fn stretch_fills_length_and_stroke_width() {
    let p = place(
        &line_node(straight(300.0), json!({})),
        &brush(rect_art(100.0, 20.0), json!({})),
    );
    assert_eq!(p.paths.len(), 1);
    let (lo, hi) = bbox(&all_points(&p.paths));
    assert!(
        near(lo.x, 0.0, 1e-6) && near(hi.x, 300.0, 1e-6),
        "{lo:?} {hi:?}"
    );
    assert!(
        near(lo.y, -5.0, 1e-6) && near(hi.y, 5.0, 1e-6),
        "{lo:?} {hi:?}"
    );
    assert_eq!(p.paths[0].paint["fills"][0]["paint"]["color"], "#000000");
    assert_eq!(p.paths[0].id, "r#0");
}

#[test]
fn scale_multiplies_thickness() {
    let p = place(
        &line_node(straight(300.0), json!({})),
        &brush(rect_art(100.0, 20.0), json!({"scale": 2.5})),
    );
    let (lo, hi) = bbox(&all_points(&p.paths));
    assert!(near(hi.y - lo.y, 25.0, 1e-6));
}

#[test]
fn directions_and_flips() {
    let node = line_node(straight(300.0), json!({}));
    // right: tip at the path end; art top (y=0 side) on the left (−y).
    let p = place(&node, &brush(arrow_art(), json!({})));
    assert!(near(extreme(&p.paths, |q| q.x).x, 300.0, 1e-6));
    let tip = extreme(&p.paths, |q| q.x);
    assert!(near(tip.y, 0.0, 1e-6));
    // left: turned 180° → tip at the start.
    let p = place(&node, &brush(arrow_art(), json!({"direction": "left"})));
    let tip = extreme(&p.paths, |q| -q.x);
    assert!(near(tip.x, 0.0, 1e-6) && near(tip.y, 0.0, 1e-6), "{tip:?}");
    // flipAlong mirrors end to end.
    let p = place(&node, &brush(arrow_art(), json!({"flipAlong": true})));
    let tip = extreme(&p.paths, |q| -q.x);
    assert!(near(tip.x, 0.0, 1e-6));
    // Override beats the brush.
    let n2 = line_node(
        straight(300.0),
        json!({"brush": {"brushId": "b", "overrides": {"flipAlong": false}}}),
    );
    let p = place(&n2, &brush(arrow_art(), json!({"flipAlong": true})));
    assert!(near(extreme(&p.paths, |q| q.x).x, 300.0, 1e-6));
}

#[test]
fn up_and_down_run_the_art_vertically() {
    // A triangle pointing up (tip at (10, 0)), 20 wide and 100 tall.
    let up_arrow = json!([{"id": "u", "type": "path", "closed": true, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}],
        "points": [{"type": "move", "x": 0, "y": 100}, {"type": "line", "x": 10, "y": 0},
                   {"type": "line", "x": 20, "y": 100}]}]);
    let node = line_node(straight(300.0), json!({}));
    let p = place(&node, &brush(up_arrow.clone(), json!({"direction": "up"})));
    let tip = extreme(&p.paths, |q| q.x);
    assert!(
        near(tip.x, 300.0, 1e-6) && near(tip.y, 0.0, 1e-6),
        "{tip:?}"
    );
    let (lo, hi) = bbox(&all_points(&p.paths));
    assert!(near(lo.y, -5.0, 1e-6) && near(hi.y, 5.0, 1e-6));
    let p = place(&node, &brush(up_arrow, json!({"direction": "down"})));
    let tip = extreme(&p.paths, |q| -q.x);
    assert!(near(tip.x, 0.0, 1e-6), "{tip:?}");
}

#[test]
fn flip_across_swaps_sides() {
    // Art only in its top half → the path's left side (−y for a +x path).
    let top = json!([{"id": "t", "type": "rectangle", "x": 0, "y": 0, "width": 100, "height": 10, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}]},
                     {"id": "g", "type": "rectangle", "x": 0, "y": 10, "width": 100, "height": 10}]);
    let node = line_node(straight(300.0), json!({}));
    let p = place(&node, &brush(top.clone(), json!({})));
    let (lo, hi) = bbox(&all_points(&p.paths[..1]));
    assert!(near(lo.y, -5.0, 1e-6) && near(hi.y, 0.0, 1e-6));
    let p = place(&node, &brush(top, json!({"flipAcross": true})));
    let (lo, hi) = bbox(&all_points(&p.paths[..1]));
    assert!(near(lo.y, 0.0, 1e-6) && near(hi.y, 5.0, 1e-6));
}

#[test]
fn guides_protect_the_ends() {
    // Marker squares at the art's ends: [0, 10] and [90, 100] along.
    let art = json!([
        {"id": "s", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 20, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}]},
        {"id": "m", "type": "rectangle", "x": 10, "y": 5, "width": 80, "height": 10, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#888"}}]},
        {"id": "e", "type": "rectangle", "x": 90, "y": 0, "width": 10, "height": 20, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}]},
    ]);
    // strokeWidth 20 = art height → k = 1: ends keep their 10 px length.
    let node = line_node(straight(1000.0), json!({"width": 1}));
    let b = brush(
        art.clone(),
        json!({"fit": "guides", "guides": {"start": 0.1, "end": 0.9}}),
    );
    let p = place(&node, &b);
    let (lo, hi) = bbox(&all_points(&p.paths[..1]));
    assert!(
        near(lo.x, 0.0, 1e-6) && near(hi.x, 10.0, 1e-6),
        "{lo:?} {hi:?}"
    );
    let (lo, hi) = bbox(&all_points(&p.paths[2..3]));
    assert!(
        near(lo.x, 990.0, 1e-6) && near(hi.x, 1000.0, 1e-6),
        "{lo:?} {hi:?}"
    );
    // Stretch for comparison: the end squares scale with the path.
    let p = place(&node, &brush(art.clone(), json!({"fit": "stretch"})));
    let (_, hi) = bbox(&all_points(&p.paths[..1]));
    assert!(near(hi.x, 100.0, 1e-6));
    // A path shorter than both caps: they shrink together.
    let short = line_node(straight(10.0), json!({"width": 1}));
    let p = place(&short, &b);
    let (_, hi) = bbox(&all_points(&p.paths[..1]));
    assert!(near(hi.x, 5.0, 1e-6), "{hi:?}");
    // Guides without guide positions fit like stretch.
    let p = place(&node, &brush(art, json!({"fit": "guides"})));
    let (_, hi) = bbox(&all_points(&p.paths[..1]));
    assert!(near(hi.x, 100.0, 1e-6));
}

#[test]
fn proportional_keeps_aspect() {
    let node = line_node(straight(300.0), json!({"width": 1}));
    let p = place(
        &node,
        &brush(rect_art(100.0, 20.0), json!({"fit": "proportional"})),
    );
    let (lo, hi) = bbox(&all_points(&p.paths));
    assert!(near(hi.x - lo.x, 300.0, 1e-6) && near(hi.y - lo.y, 60.0, 1e-6));
    // The stroke weight multiplies the thickness, as in Illustrator (weight 2: twice as thick,
    // same length).
    let node = line_node(straight(300.0), json!({"width": 2}));
    let p = place(
        &node,
        &brush(rect_art(100.0, 20.0), json!({"fit": "proportional"})),
    );
    let (lo, hi) = bbox(&all_points(&p.paths));
    assert!(near(hi.x - lo.x, 300.0, 1e-6) && near(hi.y - lo.y, 120.0, 1e-6));
}

#[test]
fn circle_path_within_tolerance() {
    let r = 100.0;
    let k = r * 0.5522847498;
    let circle = json!([
        {"type": "move", "x": r, "y": 0},
        {"type": "curve", "x": 0, "y": r, "cp1": {"x": r, "y": k}, "cp2": {"x": k, "y": r}},
        {"type": "curve", "x": -r, "y": 0, "cp1": {"x": -k, "y": r}, "cp2": {"x": -r, "y": k}},
        {"type": "curve", "x": 0, "y": -r, "cp1": {"x": -r, "y": -k}, "cp2": {"x": -k, "y": -r}},
        {"type": "curve", "x": r, "y": 0, "cp1": {"x": k, "y": -r}, "cp2": {"x": r, "y": -k}},
    ]);
    let node = line_node(circle, json!({"closed": true, "width": 1}));
    let p = place(&node, &brush(rect_art(100.0, 20.0), json!({})));
    let pts = all_points(&p.paths);
    // The KAPPA circle deviates from a true circle by ~0.027 %.
    for q in &pts {
        let d = q.len();
        let ok = (d - 90.0).abs() < 0.3 || (d - 110.0).abs() < 0.3 || (d > 89.7 && d < 110.3);
        assert!(ok, "{d}");
    }
    let outer = pts.iter().map(|q| q.len()).fold(0.0, f64::max);
    let inner = pts.iter().map(|q| q.len()).fold(f64::INFINITY, f64::min);
    assert!(
        near(outer, 110.0, 0.3) && near(inner, 90.0, 0.3),
        "{inner} {outer}"
    );
    assert!(p.paths[0].closed);
}

fn segments_intersect(a: V2, b: V2, c: V2, d: V2) -> bool {
    let o = |p: V2, q: V2, r: V2| (q - p).cross(r - p);
    let (d1, d2, d3, d4) = (o(c, d, a), o(c, d, b), o(a, b, c), o(a, b, d));
    (d1 > 1e-9 && d2 < -1e-9 || d1 < -1e-9 && d2 > 1e-9)
        && (d3 > 1e-9 && d4 < -1e-9 || d3 < -1e-9 && d4 > 1e-9)
}

fn self_intersections(poly: &[V2]) -> usize {
    let n = poly.len();
    let mut count = 0;
    for i in 0..n {
        for j in i + 2..n {
            if i == 0 && j == n - 1 {
                continue;
            }
            if segments_intersect(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]) {
                count += 1;
            }
        }
    }
    count
}

#[test]
fn overlap_adjust_prevents_corner_folds() {
    let l = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 200, "y": 0},
                   {"type": "line", "x": 200, "y": 200}]);
    let node = line_node(l, json!({"width": 2}));
    let none = place(
        &node,
        &brush(rect_art(100.0, 20.0), json!({"overlap": "none"})),
    );
    let adjust = place(
        &node,
        &brush(rect_art(100.0, 20.0), json!({"overlap": "adjust"})),
    );
    let folds = |p: &Placement| self_intersections(&polygons(&p.paths[0].points)[0]);
    assert!(folds(&none) > 0, "the plain warp folds at the corner");
    assert_eq!(folds(&adjust), 0, "adjust leaves a simple outline");
    // Adjust still covers the corner's outside and reaches both ends.
    let (lo, hi) = bbox(&all_points(&adjust.paths));
    assert!(near(lo.x, 0.0, 1e-6) && near(hi.y, 200.0, 1e-6));
    assert!(hi.x > 215.0 && lo.y < -15.0, "{lo:?} {hi:?}");
}

#[test]
fn adjust_compresses_the_inside_of_tight_curves() {
    // A 30 px radius semicircle with a 100 px thick art: the inner side would
    // cross the centre without adjustment.
    let r = 30.0;
    let k = r * 0.5522847498;
    let arc = json!([{"type": "move", "x": r, "y": 0},
        {"type": "curve", "x": 0, "y": r, "cp1": {"x": r, "y": k}, "cp2": {"x": k, "y": r}},
        {"type": "curve", "x": -r, "y": 0, "cp1": {"x": -k, "y": r}, "cp2": {"x": -r, "y": k}}]);
    let node = line_node(arc, json!({"width": 5}));
    let p = place(
        &node,
        &brush(rect_art(100.0, 20.0), json!({"overlap": "adjust"})),
    );
    let inner = all_points(&p.paths)
        .iter()
        .map(|q| q.len())
        .fold(f64::INFINITY, f64::min);
    // The inside saturates below the radius of curvature: it never crosses the centre.
    assert!(inner > 0.0 && inner < r, "{inner}");
    let poly = &polygons(&p.paths[0].points)[0];
    for q in poly {
        assert!(q.y > -50.5, "stays on the path's side of the centre: {q:?}");
    }
}

#[test]
fn width_from_profile_shapes_each_side() {
    let node = line_node(
        straight(300.0),
        json!({"profile": {"markers": [
            {"id": "a", "t": 0, "left": 2, "right": 1},
            {"id": "b", "t": 1, "left": 0, "right": 0}]},
            "brush": {"brushId": "b", "overrides": {"widthFrom": "profile"}}}),
    );
    let p = place(&node, &brush(rect_art(100.0, 20.0), json!({})));
    let pts = all_points(&p.paths);
    let at_start: Vec<&V2> = pts.iter().filter(|q| q.x < 1e-6).collect();
    let lo = at_start.iter().map(|q| q.y).fold(f64::INFINITY, f64::min);
    let hi = at_start
        .iter()
        .map(|q| q.y)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(near(lo, -10.0, 1e-6) && near(hi, 5.0, 1e-6), "{lo} {hi}");
    for q in pts.iter().filter(|q| q.x > 300.0 - 1e-6) {
        assert!(q.y.abs() < 1e-6);
    }
    // Without the override the profile is ignored.
    let node2 = line_node(
        straight(300.0),
        json!({"profile": {"preset": "taper"}}),
    );
    let p = place(&node2, &brush(rect_art(100.0, 20.0), json!({})));
    let (lo, hi) = bbox(&all_points(&p.paths));
    assert!(near(hi.y - lo.y, 10.0, 1e-6));
}

#[test]
fn pressure_varies_thickness_along() {
    let node = line_node(
        straight(300.0),
        json!({"inkChannels": {"t": [0, 1], "pressure": [0, 1]}}),
    );
    let b = brush(
        rect_art(100.0, 20.0),
        json!({"scale": {"value": 1, "vary": "pressure", "min": 0.2, "max": 2}}),
    );
    let p = place(&node, &b);
    let pts = all_points(&p.paths);
    let half_at = |x: f64| {
        pts.iter()
            .filter(|q| (q.x - x).abs() < 1e-6)
            .map(|q| q.y.abs())
            .fold(0.0, f64::max)
    };
    assert!(near(half_at(0.0), 1.0, 1e-6), "{}", half_at(0.0));
    assert!(near(half_at(300.0), 10.0, 1e-6), "{}", half_at(300.0));
}

#[test]
fn random_scale_is_seeded() {
    let b = brush(
        rect_art(100.0, 20.0),
        json!({"scale": {"value": 1, "vary": "random", "min": 0.5, "max": 3}}),
    );
    let thick = |seed: u32| {
        let n = line_node(
            straight(300.0),
            json!({"brush": {"brushId": "b", "seed": seed}}),
        );
        let (lo, hi) = bbox(&all_points(&place(&n, &b).paths));
        hi.y - lo.y
    };
    assert_eq!(thick(1), thick(1));
    assert_ne!(thick(1), thick(2));
    for s in 0..20 {
        let t = thick(s);
        assert!((5.0 - 1e-9..=30.0 + 1e-9).contains(&t), "{t}");
    }
}

#[test]
fn colorize_applies_to_output_paints() {
    let art = json!([{"id": "k", "type": "rectangle", "width": 100, "height": 20, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}],
        "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#ffffff"}, "width": 2}]}]);
    let node = line_node(straight(300.0), json!({"paint": {"type": "solid", "color": "#3366cc"}}));
    let p = place(
        &node,
        &brush(art.clone(), json!({"colorize": {"method": "tints"}})),
    );
    assert_eq!(p.paths[0].paint["fills"][0]["paint"]["color"], "#3366cc");
    assert_eq!(p.paths[0].paint["strokes"][0]["paint"]["color"], "#ffffff");
    // Stroke widths scale with the cross scale (10 / 20).
    assert_eq!(p.paths[0].paint["strokes"][0]["width"], json!(1.0));
    // Per-stroke override.
    let n2 = line_node(
        straight(300.0),
        json!({"paint": {"type": "solid", "color": "#3366cc"},
        "brush": {"brushId": "b", "overrides": {"colorize": {"method": "none"}}}}),
    );
    let p = place(&n2, &brush(art, json!({"colorize": {"method": "tints"}})));
    assert_eq!(p.paths[0].paint["fills"][0]["paint"]["color"], "#000000");
}

#[test]
fn compound_paths_get_art_per_subpath() {
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
                     {"type": "move", "x": 0, "y": 50}, {"type": "line", "x": 300, "y": 50}]);
    let p = place(
        &line_node(pts, json!({})),
        &brush(rect_art(100.0, 20.0), json!({})),
    );
    assert_eq!(p.paths.len(), 2);
    assert_eq!(
        (p.paths[0].id.as_str(), p.paths[1].id.as_str()),
        ("r#0", "r#1")
    );
    let (lo, hi) = bbox(&all_points(&p.paths[1..]));
    assert!(near(lo.x, 0.0, 1e-6) && near(hi.x, 300.0, 1e-6) && near(lo.y, 45.0, 1e-6));
}

#[test]
fn output_is_cubic_and_continuous() {
    let wave = json!([{"type": "move", "x": 0, "y": 0},
        {"type": "curve", "x": 150, "y": 0, "cp1": {"x": 50, "y": -80}, "cp2": {"x": 100, "y": 80}},
        {"type": "curve", "x": 300, "y": 0, "cp1": {"x": 200, "y": -80}, "cp2": {"x": 250, "y": 80}}]);
    let p = place(
        &line_node(wave, json!({"width": 0.8})),
        &brush(arrow_art(), json!({})),
    );
    for path in &p.paths {
        assert!(matches!(
            path.points[0],
            crate::blend::BlendPathPoint::Move { .. }
        ));
        assert!(path.points[1..]
            .iter()
            .all(|q| matches!(q, crate::blend::BlendPathPoint::Curve { .. })));
    }
}

#[test]
fn json_envelopes() {
    let ok = art_brush_json(
        &json!({"node": line_node(straight(100.0), json!({})), "brush": brush(rect_art(10.0, 2.0), json!({}))})
            .to_string(),
    );
    let v: Value = crate::value::from_str(&ok).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["nodes"][0]["type"], "path");
    assert_eq!(v["nodes"][0]["sourceId"], "r");
    assert!(v["bounds"]["maxX"].as_f64().unwrap() > 99.0);
    let err = |s: &str| crate::value::from_str::<Value>(&art_brush_json(s)).unwrap()["code"].clone();
    assert_eq!(err("{"), "invalid_json");
    assert_eq!(err("[]"), "invalid_request");
    assert_eq!(err(r#"{"node": {}}"#), "invalid_request");
    assert_eq!(
        err(&json!({"node": {"points": []}, "brush": {"id": "x", "type": "scatter"}}).to_string()),
        "invalid_brush"
    );
    let empty = art_brush_json(
        &json!({"node": line_node(json!([]), json!({})), "brush": brush(rect_art(10.0, 2.0), json!({}))})
            .to_string(),
    );
    assert_eq!(
        crate::value::from_str::<Value>(&empty).unwrap()["status"],
        "empty"
    );
    // Zero-length artwork along the direction.
    let flat = art_brush_json(
        &json!({"node": line_node(straight(100.0), json!({})), "brush": brush(rect_art(0.0, 2.0), json!({}))})
            .to_string(),
    );
    assert_eq!(
        crate::value::from_str::<Value>(&flat).unwrap()["status"],
        "empty"
    );
}

#[test]
fn parse_mirrors_normalize_brush_definition() {
    let b = parse_art_brush(&json!({"id": "x", "type": "art", "artwork": {"nodes": [{"id": "a", "type": "path"}, {"id": 3}]},
        "guides": {"start": 0.8, "end": 0.2}, "fit": "weird", "direction": "sideways", "overlap": "adjust",
        "scale": -2, "flipAlong": 1})).unwrap();
    assert_eq!(b.artwork.len(), 1);
    assert_eq!(b.guides, None, "end ≤ start: ignored");
    assert!(b.guides_ignored);
    assert_eq!(b.fit, ArtFit::Stretch);
    assert_eq!(b.direction, ArtDirection::Right);
    assert_eq!(b.overlap, Overlap::Adjust);
    assert_eq!(b.scale.value, 0.0);
    assert!(!b.flip_along);
    assert_eq!(b.name, "x");
    assert!(parse_art_brush(&json!({"id": "", "type": "art", "artwork": {"nodes": []}})).is_none());
    assert!(parse_art_brush(&json!({"id": "x", "type": "art"})).is_none());
    let o = parse_art_overrides(
        json!({"fit": "add-space", "widthFrom": "profile", "scale": 3}).as_object(),
    );
    assert_eq!(o.fit, None);
    assert!(o.width_from_profile);
    assert_eq!(o.scale.unwrap().value, 3.0);
}

#[test]
fn deterministic_output() {
    let req = json!({"node": line_node(json!([{"type": "move", "x": 0, "y": 0},
        {"type": "curve", "x": 200, "y": 40, "cp1": {"x": 60, "y": -90}, "cp2": {"x": 120, "y": 120}}]),
        json!({"brush": {"brushId": "b", "seed": 99}})),
        "brush": brush(arrow_art(), json!({"scale": {"value": 1, "vary": "random", "min": 0.5, "max": 2},
            "overlap": "adjust", "colorize": {"method": "hue-shift"}}))})
    .to_string();
    assert_eq!(art_brush_json(&req), art_brush_json(&req));
}

/// Tiny deterministic generator for the fuzz.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }
    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[(self.next() * xs.len() as f64) as usize % xs.len()]
    }
}

fn wild(r: &mut Lcg) -> Value {
    match (r.next() * 10.0) as u32 {
        0 => json!(0.0),
        1 => json!(1e12),
        2 => json!(-1e-12),
        _ => json!(r.range(-300.0, 300.0)),
    }
}

fn random_points(r: &mut Lcg, n: usize) -> Value {
    let mut pts = vec![json!({"type": "move", "x": wild(r), "y": wild(r)})];
    for _ in 0..n {
        let t = r.pick(&["line", "curve", "curve", "quadratic", "arc", "move", "line"]);
        pts.push(match t {
            "curve" => json!({"type": "curve", "x": wild(r), "y": wild(r),
                "cp1": {"x": wild(r), "y": wild(r)}, "cp2": {"x": wild(r), "y": wild(r)}}),
            "quadratic" => json!({"type": "quadratic", "x": wild(r), "y": wild(r), "cp": {"x": wild(r), "y": wild(r)}}),
            "arc" => json!({"type": "arc", "x": wild(r), "y": wild(r), "rx": wild(r), "ry": wild(r),
                "xAxisRotation": wild(r), "largeArc": r.next() > 0.5, "sweep": r.next() > 0.5}),
            other => json!({"type": other, "x": wild(r), "y": wild(r)}),
        });
    }
    Value::Array(pts)
}

#[test]
fn fuzz_no_panics_finite_output() {
    let mut r = Lcg(0x5eed);
    for case in 0..300 {
        let n_path = 1 + (r.next() * 12.0) as usize;
        let n_art = 1 + (r.next() * 8.0) as usize;
        let art = json!([
            {"id": "p", "type": "path", "closed": r.next() > 0.3, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#123456"}}], "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#abcdef"}, "width": r.range(0.0, 5.0)}], "points": random_points(&mut r, n_art)},
            {"id": "e", "type": "ellipse", "x": wild(&mut r), "y": wild(&mut r), "width": wild(&mut r), "height": wild(&mut r)},
        ]);
        let guides = json!({"start": r.range(-0.5, 1.5), "end": r.range(-0.5, 1.5)});
        let b = brush(
            art,
            json!({
                "fit": r.pick(&["stretch", "proportional", "guides"]),
                "direction": r.pick(&["left", "right", "up", "down"]),
                "overlap": r.pick(&["adjust", "none"]),
                "flipAlong": r.next() > 0.5, "flipAcross": r.next() > 0.5,
                "guides": guides,
                "scale": {"value": r.range(0.0, 3.0), "vary": r.pick(&["random", "pressure", "tilt"]), "min": 0, "max": 4},
                "colorize": {"method": r.pick(&["none", "tints", "tints-shades", "hue-shift"])},
            }),
        );
        let node = line_node(
            random_points(&mut r, n_path),
            json!({
                "closed": r.next() > 0.7, "width": wild(&mut r), "cornerRadius": r.range(-5.0, 20.0),
                "inkChannels": {"t": [0, 0.5, 1], "pressure": [r.next(), r.next(), r.next()], "tiltX": [10, 40, -30]},
                "profile": {"preset": "eye"},
                "brush": {"brushId": "b", "seed": case, "overrides": {"widthFrom": "profile"}},
            }),
        );
        let req = json!({"node": node, "brush": b, "tolerance": r.range(0.01, 2.0)}).to_string();
        let out: Value = crate::value::from_str(&art_brush_json(&req)).unwrap();
        let status = out["status"].as_str().unwrap();
        // Extreme inputs may refuse with a clear reason instead of stalling.
        let refused = status == "error" && out["code"] == "too_complex";
        assert!(
            status == "ok" || status == "empty" || refused,
            "case {case}: {out}"
        );
        if status == "ok" {
            for n in out["nodes"].as_array().unwrap() {
                for p in n["points"].as_array().unwrap() {
                    for k in ["x", "y"] {
                        assert!(
                            p[k].as_f64().is_some_and(f64::is_finite),
                            "case {case}: {p}"
                        );
                    }
                }
            }
        }
    }
}

/// A 500-anchor wavy path and a 200-anchor artwork. `cargo test --release -p
/// moda-scene --lib stroke_brush::tests_art::perf -- --ignored --nocapture`.
#[test]
#[ignore]
fn perf_500_sample_path_200_anchor_art() {
    let mut pts = vec![json!({"type": "move", "x": 0, "y": 0})];
    for i in 1..500 {
        let x = i as f64 * 4.0;
        pts.push(
            json!({"type": "line", "x": x, "y": (x / 37.0).sin() * 60.0 + (x / 11.0).cos() * 8.0}),
        );
    }
    // A leaf: 200 anchors around an ellipse-ish outline.
    let mut art_pts = Vec::new();
    for i in 0..200 {
        let a = i as f64 / 200.0 * std::f64::consts::TAU;
        let (x, y) = (
            100.0 + 100.0 * a.cos(),
            10.0 * a.sin() * (1.0 + 0.3 * (5.0 * a).sin()),
        );
        art_pts.push(json!({"type": if i == 0 { "move" } else { "line" }, "x": x, "y": y}));
    }
    let art = json!([{"id": "leaf", "type": "path", "closed": true, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#2a9d4a"}}], "points": art_pts}]);
    let node = line_node(Value::Array(pts), json!({"width": 1.2}));
    let b = parse_art_brush(&brush(art, json!({"overlap": "adjust"}))).unwrap();
    let node = node.as_object().unwrap().clone();
    let run = || art_brush_for_node(&node, &b, &default_geometry, &ArtOptions::default());
    let first = run();
    // Best-of and median of 40 runs: the machine is shared, so the minimum is
    // the number to compare against the 3 ms target.
    let mut times: Vec<f64> = (0..40)
        .map(|_| {
            let t0 = std::time::Instant::now();
            std::hint::black_box(run());
            t0.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    times.sort_by(|a, b| a.total_cmp(b));
    let (per, median) = (times[0], times[times.len() / 2]);
    let segs: usize = first.paths.iter().map(|p| p.points.len()).sum();
    println!(
        "art brush: best {per:.3} ms, median {median:.3} ms per placement, {segs} output points"
    );
}

fn dist_to_polyline(p: V2, poly: &[V2]) -> f64 {
    let mut best = f64::INFINITY;
    for w in poly.windows(2) {
        let (a, b) = (w[0], w[1]);
        let d = b - a;
        let l2 = d.len2();
        let t = if l2 > 0.0 {
            ((p - a).dot(d) / l2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        best = best.min(p.dist(a + d * t));
    }
    best
}

#[test]
fn many_anchor_path_stays_within_tolerance() {
    let mut pts = vec![json!({"type": "move", "x": 0, "y": 0})];
    for i in 1..80 {
        let x = i as f64 * 5.0;
        pts.push(
            json!({"type": "line", "x": x, "y": (x / 23.0).sin() * 40.0 + (x / 7.0).cos() * 5.0}),
        );
    }
    let node = line_node(Value::Array(pts), json!({"width": 0.7}));
    for overlap in ["none", "adjust"] {
        let b = parse_art_brush(&brush(arrow_art(), json!({"overlap": overlap}))).unwrap();
        let n = node.as_object().unwrap();
        let tol = 0.25;
        let got = art_brush_for_node(
            n,
            &b,
            &default_geometry,
            &ArtOptions {
                tolerance: tol,
                windows: None,
            },
        );
        let reference = art_brush_for_node(
            n,
            &b,
            &default_geometry,
            &ArtOptions {
                tolerance: 0.002,
                windows: None,
            },
        );
        let fine: Vec<V2> = {
            // Densely sample the reference cubics.
            let mut v = Vec::new();
            let mut cur = V2::default();
            for q in &reference.paths[0].points {
                match q {
                    crate::blend::BlendPathPoint::Move { x, y } => cur = v2(*x, *y),
                    crate::blend::BlendPathPoint::Curve { x, y, cp1, cp2 } => {
                        let c = crate::path_geom::vec2::Cubic([
                            cur,
                            v2(cp1.0, cp1.1),
                            v2(cp2.0, cp2.1),
                            v2(*x, *y),
                        ]);
                        for i in 0..=24 {
                            v.push(c.eval(i as f64 / 24.0));
                        }
                        cur = v2(*x, *y);
                    }
                    _ => {}
                }
            }
            v
        };
        let mut worst: f64 = 0.0;
        for poly in polygons(&got.paths[0].points) {
            for p in poly {
                worst = worst.max(dist_to_polyline(p, &fine));
            }
        }
        assert!(worst <= tol * 1.2, "{overlap}: {worst}");
        assert!(got.paths[0].points.len() < reference.paths[0].points.len());
    }
}

/// Worst cases must refuse (or finish) fast: `cargo test --release -p moda-scene --lib
/// stroke_brush::tests_art::perf_worst -- --ignored --nocapture`. Target < 50 ms each.
#[test]
#[ignore]
fn perf_worst_cases() {
    let wavy = |n: usize, step: f64| {
        let mut pts = vec![json!({"type": "move", "x": 0, "y": 0})];
        for i in 1..n {
            let x = i as f64 * step;
            pts.push(json!({"type": "line", "x": x, "y": (x / 7.0).sin() * 40.0}));
        }
        Value::Array(pts)
    };
    let cases = [
        (
            "art, 20 000-anchor zigzag, tolerance 0.001",
            json!({
            "node": line_node(wavy(20_000, 3.0), json!({"width": 2})),
            "brush": brush(arrow_art(), json!({"overlap": "adjust"})), "tolerance": 0.001}),
        ),
        (
            "art, 20 000 anchors (the cap)",
            json!({
            "node": line_node(wavy(20_000, 1.0), json!({"width": 1})),
            "brush": brush(arrow_art(), json!({}))}),
        ),
        (
            "art, 100 000 anchors (over the cap)",
            json!({
            "node": line_node(wavy(100_000, 1.0), json!({"width": 1})),
            "brush": brush(arrow_art(), json!({}))}),
        ),
        (
            "scatter, tiny copies on a long path",
            json!({
            "node": line_node(straight(1e7), json!({"width": 0.01})),
            "brush": {"id": "s", "type": "scatter", "artwork": {"nodes": arrow_art()}, "spacing": 0}}),
        ),
    ];
    for (name, req) in cases {
        let json = req.to_string();
        let t0 = std::time::Instant::now();
        let out: Value = crate::value::from_str(&super::stroke_brush_json(&json)).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("{name}: {ms:.1} ms → {} {}", out["status"], out["code"]);
    }
}

#[test]
fn adjust_reaches_into_acute_corners() {
    // Spikes of 30°, 15° and a near-hairpin: the art must still reach the
    // corner (the P17 review measured 29 / 56 / 90 px short with a uniform
    // fillet). Half-thickness here is 10 px.
    for interior in [30.0f64, 15.0, 4.0] {
        let a = interior.to_radians();
        let tip = v2(200.0, 0.0);
        let back = v2(200.0 - 200.0 * a.cos(), 200.0 * a.sin());
        let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": tip.x, "y": tip.y},
            {"type": "line", "x": back.x, "y": back.y}]);
        let p = place(
            &line_node(pts, json!({"width": 1})),
            &brush(rect_art(100.0, 20.0), json!({"overlap": "adjust"})),
        );
        let all = all_points(&p.paths);
        let nearest = all
            .iter()
            .map(|q| q.dist(tip))
            .fold(f64::INFINITY, f64::min);
        assert!(
            nearest < 10.0,
            "{interior}°: art stops {nearest} px from the corner"
        );
        // Both legs are still covered to their far ends.
        assert!(all.iter().any(|q| q.dist(v2(0.0, 0.0)) < 10.5));
        assert!(all.iter().any(|q| q.dist(back) < 10.5));
    }
}

#[test]
fn guides_are_in_unturned_art() {
    // Marker squares at the art's LEFT end (x ∈ [0, 10]); guides protect [0, 0.1].
    let art = json!([
        {"id": "s", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 20, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}]},
        {"id": "m", "type": "rectangle", "x": 10, "y": 5, "width": 90, "height": 10, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#888"}}]},
    ]);
    let node = line_node(straight(1000.0), json!({"width": 1}));
    // Direction left runs the art from its right end: the protected square
    // (unturned left end) lands at the path END and keeps its 10 px.
    let b = brush(
        art,
        json!({"fit": "guides", "direction": "left", "guides": {"start": 0.1, "end": 1.0}}),
    );
    let p = place(&node, &b);
    let (lo, hi) = bbox(&all_points(&p.paths[..1]));
    assert!(
        near(lo.x, 990.0, 1e-6) && near(hi.x, 1000.0, 1e-6),
        "{lo:?} {hi:?}"
    );
    // Equal guides: ignored with a warning.
    let bad = brush(
        rect_art(100.0, 20.0),
        json!({"fit": "guides", "guides": {"start": 0.5, "end": 0.5}}),
    );
    let p = place(&node, &bad);
    assert_eq!(p.warnings.len(), 1);
    assert!(p.warnings[0].starts_with("guides_ignored"));
    let v: Value = crate::value::from_str(&super::common::placement_json(&p)).unwrap();
    assert!(v["warnings"][0]
        .as_str()
        .unwrap()
        .starts_with("guides_ignored"));
}

#[test]
fn stroke_weight_multiplies_native_art_size() {
    // Illustrator: at weight 1 the art is its own height; weight and scale multiply.
    for (sw, scale, h) in [(1.0, 1.0, 20.0), (2.0, 1.0, 40.0), (0.5, 3.0, 30.0)] {
        let node = line_node(straight(300.0), json!({"width": sw}));
        let p = place(
            &node,
            &brush(rect_art(100.0, 20.0), json!({"scale": scale})),
        );
        let (lo, hi) = bbox(&all_points(&p.paths));
        assert!(
            near(hi.y - lo.y, h, 1e-6),
            "{sw} × {scale}: {}",
            hi.y - lo.y
        );
    }
}

#[test]
fn art_override_fit_parsing() {
    let n = |fit: &str| {
        line_node(
            straight(1000.0),
            json!({"width": 1,
            "brush": {"brushId": "b", "overrides": {"fit": fit}}}),
        )
    };
    let art = json!([
        {"id": "s", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 20, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}]},
        {"id": "m", "type": "rectangle", "x": 10, "y": 5, "width": 90, "height": 10, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#888"}}]},
    ]);
    let b = brush(
        art,
        json!({"fit": "stretch", "guides": {"start": 0.1, "end": 1.0}}),
    );
    // An art fit overrides the brush's…
    let (_, hi) = bbox(&all_points(&place(&n("guides"), &b).paths[..1]));
    assert!(near(hi.x, 10.0, 1e-6), "{hi:?}");
    // …a pattern fit is ignored (the brush's stretch applies).
    let (_, hi) = bbox(&all_points(&place(&n("add-space"), &b).paths[..1]));
    assert!(near(hi.x, 100.0, 1e-6), "{hi:?}");
}

#[test]
#[ignore]
fn perf_json_end_to_end() {
    let mut pts = vec![json!({"type": "move", "x": 0, "y": 0})];
    for i in 1..500 {
        let x = i as f64 * 4.0;
        pts.push(
            json!({"type": "line", "x": x, "y": (x / 37.0).sin() * 60.0 + (x / 11.0).cos() * 8.0}),
        );
    }
    let mut art_pts = Vec::new();
    for i in 0..200 {
        let a = i as f64 / 200.0 * std::f64::consts::TAU;
        art_pts.push(json!({"type": if i == 0 { "move" } else { "line" }, "x": 100.0 + 100.0 * a.cos(), "y": 10.0 * a.sin() * (1.0 + 0.3 * (5.0 * a).sin())}));
    }
    let req = json!({"node": {"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1.2}]},
        "brush": {"id": "b", "type": "art", "artwork": {"nodes": [{"id": "leaf", "type": "path", "closed": true, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#2a9d4a"}}], "points": art_pts}]}, "overlap": "adjust"}}).to_string();
    let mut best = f64::INFINITY;
    for _ in 0..30 {
        let t = std::time::Instant::now();
        std::hint::black_box(super::stroke_brush_json(&req));
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("json end to end: {best:.2} ms");
}

#[test]
fn staircase_short_riser_has_no_break() {
    // #70 review: staircase (0,0)→(100,0)→(100,40)→(200,40), art 100 × 40,
    // overlap adjust: both corner fillets want more than half the riser.
    for rise in [40.0, 10.0, 20.0] {
        let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
            {"type": "line", "x": 100, "y": rise}, {"type": "line", "x": 200, "y": rise}]);
        let p = place(
            &line_node(pts, json!({"width": 1})),
            &brush(rect_art(100.0, 40.0), json!({"overlap": "adjust"})),
        );
        assert_eq!(p.paths.len(), 1);
        for w in p.paths[0].points.windows(2) {
            if let (
                crate::blend::BlendPathPoint::Curve { x, y, .. },
                crate::blend::BlendPathPoint::Curve { cp1, .. },
            ) = (&w[0], &w[1])
            {
                // Consecutive curves: handles stay near their anchor (no 28 px jumps).
                assert!(v2(*x, *y).dist(v2(cp1.0, cp1.1)) < 60.0, "{rise}");
            }
        }
        let all = all_points(&p.paths);
        assert!(all.iter().any(|q| q.dist(v2(0.0, 0.0)) < 20.5));
        assert!(all.iter().any(|q| q.dist(v2(200.0, rise)) < 20.5));
    }
}

fn place_windows(node: &Value, brush: &Value, windows: &[ArcWindow]) -> Placement {
    let b = parse_art_brush(brush).expect("art brush");
    art_brush_for_node(
        node.as_object().unwrap(),
        &b,
        &default_geometry,
        &ArtOptions {
            tolerance: 0.25,
            windows: Some(windows),
        },
    )
}

fn win(subpath: usize, start: f64, end: f64) -> ArcWindow {
    ArcWindow {
        subpath,
        start,
        end,
        taper_extent: None,
    }
}

#[test]
fn windows_reveal_without_restretching() {
    let node = line_node(straight(300.0), json!({"width": 0.5}));
    let b = brush(arrow_art(), json!({}));
    let full = place(&node, &b);
    let part = place_windows(&node, &b, &[win(0, 100.0, 200.0)]);
    let (lo, hi) = bbox(&all_points(&part.paths));
    assert!(
        near(lo.x, 100.0, 1e-6) && near(hi.x, 200.0, 1e-6),
        "{lo:?} {hi:?}"
    );
    // The arrow tapers to its tip over the whole path: the revealed piece has
    // the full placement's thickness at each x (a reveal, not a re-stretch).
    let thickness = |pts: &[V2], x: f64| {
        pts.iter()
            .filter(|q| (q.x - x).abs() < 0.5)
            .map(|q| q.y.abs())
            .fold(0.0, f64::max)
    };
    let (fp, pp) = (all_points(&full.paths), all_points(&part.paths));
    for x in [110.0, 150.0, 190.0] {
        assert!(
            near(thickness(&fp, x), thickness(&pp, x), 0.3),
            "x {x}: {} vs {}",
            thickness(&fp, x),
            thickness(&pp, x)
        );
    }
    assert_eq!(part.paths[0].id, "a#0.w0");
    assert_eq!(
        part.paths[0].at,
        PlacedAt {
            subpath: 0,
            s: 150.0,
            s_span: (100.0, 200.0)
        }
    );
    // Two windows (dashes): two pieces; none → nothing; the wrong subpath → nothing.
    assert_eq!(
        place_windows(&node, &b, &[win(0, 0.0, 50.0), win(0, 250.0, 300.0)])
            .paths
            .len(),
        2
    );
    assert!(place_windows(&node, &b, &[]).paths.is_empty());
    assert!(place_windows(&node, &b, &[win(1, 0.0, 300.0)])
        .paths
        .is_empty());
    // A window covering everything equals the untrimmed placement.
    let all = place_windows(&node, &b, &[win(0, -10.0, 400.0)]);
    assert_eq!(all.paths[0].points.len(), full.paths[0].points.len());
}

#[test]
fn windows_on_a_curve_stay_inside_the_full_placement() {
    let wave = json!([{"type": "move", "x": 0, "y": 0},
        {"type": "curve", "x": 150, "y": 0, "cp1": {"x": 50, "y": -80}, "cp2": {"x": 100, "y": 80}},
        {"type": "curve", "x": 300, "y": 0, "cp1": {"x": 200, "y": -80}, "cp2": {"x": 250, "y": 80}}]);
    let node = line_node(wave, json!({"width": 1}));
    let b = brush(rect_art(100.0, 20.0), json!({"overlap": "adjust"}));
    let full = place(&node, &b);
    let outline = polygons(&full.paths[0].points);
    let inside = |p: V2| {
        // Nonzero winding of the full outline, or on its boundary.
        let mut wind = 0i32;
        for poly in &outline {
            for i in 0..poly.len() {
                let (a, c) = (poly[i], poly[(i + 1) % poly.len()]);
                if (a.y <= p.y) != (c.y <= p.y) {
                    let x = a.x + (p.y - a.y) / (c.y - a.y) * (c.x - a.x);
                    if x > p.x {
                        wind += if c.y > a.y { 1 } else { -1 };
                    }
                }
            }
        }
        wind != 0
            || outline
                .iter()
                .any(|poly| dist_to_polyline(p, &[poly.as_slice(), &poly[..1]].concat()) < 0.6)
    };
    let part = place_windows(&node, &b, &[win(0, 60.0, 180.0)]);
    let pts = all_points(&part.paths);
    assert!(!pts.is_empty());
    for p in pts {
        assert!(inside(p), "{p:?} outside the full placement");
    }
}

#[test]
fn windows_through_a_closed_seam() {
    let r = 100.0;
    let k = r * 0.5522847498;
    let circle = json!([
        {"type": "move", "x": r, "y": 0},
        {"type": "curve", "x": 0, "y": r, "cp1": {"x": r, "y": k}, "cp2": {"x": k, "y": r}},
        {"type": "curve", "x": -r, "y": 0, "cp1": {"x": -k, "y": r}, "cp2": {"x": -r, "y": k}},
        {"type": "curve", "x": 0, "y": -r, "cp1": {"x": -r, "y": -k}, "cp2": {"x": -k, "y": -r}},
        {"type": "curve", "x": r, "y": 0, "cp1": {"x": k, "y": -r}, "cp2": {"x": r, "y": -k}},
    ]);
    let node = line_node(circle, json!({"closed": true, "width": 1}));
    let len = crate::stroke_brush::common::node_stroke(node.as_object().unwrap()).measured[0];
    let part = place_windows(
        &node,
        &brush(rect_art(100.0, 20.0), json!({})),
        &[win(0, len - 50.0, len + 50.0)],
    );
    // Split at the seam (the art's start and end): two pieces, both within ±50 px of (100, 0).
    assert_eq!(part.paths.len(), 2);
    assert_eq!(
        (part.paths[0].id.as_str(), part.paths[1].id.as_str()),
        ("r#0.w0.0", "r#0.w0.1")
    );
    for p in all_points(&part.paths) {
        let angle = p.y.atan2(p.x).abs();
        assert!(
            angle * r <= 50.0 + 1.0,
            "{p:?} is {} px from the seam",
            angle * r
        );
    }
    let ats: Vec<_> = part.paths.iter().map(|p| p.at.s_span).collect();
    assert!(
        near(ats[0].0, len - 50.0, 1e-6)
            && near(ats[0].1, len, 1e-6)
            && near(ats[1].0, 0.0, 1e-9)
            && near(ats[1].1, 50.0, 1e-6),
        "{ats:?}"
    );
}

#[test]
fn json_windows_and_placement_fields() {
    let req = json!({"node": line_node(straight(300.0), json!({"width": 0.5})), "brush": brush(arrow_art(), json!({})),
        "windows": [{"subpath": 0, "start": 30, "end": 90}]});
    let v: Value = crate::value::from_str(&art_brush_json(&req.to_string())).unwrap();
    assert_eq!(v["nodes"][0]["subpath"], 0);
    assert_eq!(v["nodes"][0]["s"].as_f64(), Some(60.0));
    assert_eq!(v["nodes"][0]["sSpan"][0].as_f64(), Some(30.0));
    assert_eq!(v["nodes"][0]["sSpan"][1].as_f64(), Some(90.0));
    for bad in [
        json!("x"),
        json!([{"subpath": -1, "start": 0, "end": 1}]),
        json!([{"start": 0, "end": 1}]),
    ] {
        let r = json!({"node": line_node(straight(30.0), json!({})), "brush": brush(arrow_art(), json!({})), "windows": bad});
        let v: Value = crate::value::from_str(&art_brush_json(&r.to_string())).unwrap();
        assert_eq!(v["code"], "invalid_request", "{bad}");
    }
}

#[test]
fn window_ends_land_exactly_on_a_filleted_zigzag() {
    // 90° zigzag with overlap adjust (fillet radius ≈ half the art, 10 px): a
    // window's cut ends sit at the exact input arc length on straight runs
    // (the fillets no longer drift the mapping).
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
        {"type": "line", "x": 100, "y": 100}, {"type": "line", "x": 200, "y": 100},
        {"type": "line", "x": 200, "y": 200}]);
    let node = line_node(pts, json!({"width": 1}));
    let b = brush(rect_art(100.0, 20.0), json!({"overlap": "adjust"}));
    // Input s = 150 → (100, 50); s = 250 → (150, 100). Both mid-run.
    let part = place_windows(&node, &b, &[win(0, 150.0, 250.0)]);
    let pts = all_points(&part.paths);
    let outline: Vec<Vec<V2>> = part
        .paths
        .iter()
        .flat_map(|p| polygons(&p.points))
        .collect();
    let near_to = |target: V2| {
        outline
            .iter()
            .map(|poly| dist_to_polyline(target, &[poly.as_slice(), &poly[..1]].concat()))
            .fold(f64::INFINITY, f64::min)
    };
    // The cut edges cross the centerline at the window ends.
    for (target, dir) in [
        (v2(100.0, 50.0), v2(1.0, 0.0)),
        (v2(150.0, 100.0), v2(0.0, 1.0)),
    ] {
        for side in [-9.0, 9.0] {
            let p = target + dir * side;
            assert!(near_to(p) < 0.1, "{p:?}: {}", near_to(p));
        }
    }
    // Nothing before (100, 50) on the riser, nothing past (150, 100) on the next run.
    for q in &pts {
        assert!(!(q.y < 49.0 && q.x > 90.0 && q.x < 111.0), "{q:?}");
        assert!(!(q.x > 151.0 && q.y > 89.0 && q.y < 111.0), "{q:?}");
    }
}

#[test]
fn clipped_strokes_skip_the_cut_edges() {
    // A stroked rectangle revealed in a window: the fill is the closed cut
    // polygon; the stroke is the original outline only, as open runs.
    let art = json!([{"id": "r", "type": "rectangle", "x": 0, "y": 0, "width": 100, "height": 20,
        "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}], "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": 2}]}]);
    let node = line_node(straight(300.0), json!({"width": 1}));
    let part = place_windows(&node, &brush(art, json!({})), &[win(0, 100.0, 200.0)]);
    assert_eq!(part.paths.len(), 2);
    let (fill, stroke) = (&part.paths[0], &part.paths[1]);
    assert_eq!(
        (fill.id.as_str(), stroke.id.as_str()),
        ("r#0.w0", "r#0.w0.stroke")
    );
    assert!(fill.paint.get("strokes").is_none() && fill.closed);
    assert_eq!(stroke.paint.get("fills"), None, "stroke-only part: no fill");
    assert!(!stroke.closed);
    // No stroke segment runs across the path at a cut (x = 100 or 200).
    for poly in polygons(&stroke.points) {
        for w in poly.windows(2) {
            let vertical = (w[0].x - w[1].x).abs() < 1e-6 && (w[0].y - w[1].y).abs() > 1e-3;
            assert!(
                !(vertical && ((w[0].x - 100.0).abs() < 1e-6 || (w[0].x - 200.0).abs() < 1e-6)),
                "{w:?}"
            );
        }
    }
}

#[test]
fn a_ring_window_is_one_piece() {
    let r = 100.0;
    let k = r * 0.5522847498;
    let circle = json!([
        {"type": "move", "x": r, "y": 0},
        {"type": "curve", "x": 0, "y": r, "cp1": {"x": r, "y": k}, "cp2": {"x": k, "y": r}},
        {"type": "curve", "x": -r, "y": 0, "cp1": {"x": -k, "y": r}, "cp2": {"x": -r, "y": k}},
        {"type": "curve", "x": 0, "y": -r, "cp1": {"x": -r, "y": -k}, "cp2": {"x": -k, "y": -r}},
        {"type": "curve", "x": r, "y": 0, "cp1": {"x": k, "y": -r}, "cp2": {"x": r, "y": -k}},
    ]);
    let node = line_node(circle, json!({"closed": true, "width": 1}));
    let len = crate::stroke_brush::common::node_stroke(node.as_object().unwrap()).measured[0];
    let part = place_windows(
        &node,
        &brush(rect_art(100.0, 20.0), json!({})),
        &[win(0, 0.0, len)],
    );
    assert_eq!(part.paths.len(), 1);
    assert!(part.paths[0].closed);
}

#[test]
fn overlap_override_replaces_the_brush_option() {
    let l = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 200, "y": 0},
                   {"type": "line", "x": 200, "y": 200}]);
    let folds = |p: &Placement| self_intersections(&polygons(&p.paths[0].points)[0]);
    let none_brush = brush(rect_art(100.0, 20.0), json!({"overlap": "none"}));
    let adjusted = line_node(
        l.clone(),
        json!({"width": 2, "brush": {"brushId": "b", "overrides": {"overlap": "adjust"}}}),
    );
    assert_eq!(folds(&place(&adjusted, &none_brush)), 0, "per-stroke adjust");
    let adjust_brush = brush(rect_art(100.0, 20.0), json!({"overlap": "adjust"}));
    let plain = line_node(
        l,
        json!({"width": 2, "brush": {"brushId": "b", "overrides": {"overlap": "none"}}}),
    );
    assert!(folds(&place(&plain, &adjust_brush)) > 0, "per-stroke none");
    let o = parse_art_overrides(json!({"overlap": "adjust"}).as_object());
    assert_eq!(o.overlap, Some(Overlap::Adjust));
    assert_eq!(parse_art_overrides(json!({"overlap": "x"}).as_object()).overlap, None);
}
