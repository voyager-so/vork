//! Scatter brush tests.

use crate::value::{json, Value};

use super::artwork::default_geometry;
use super::common::Placement;
use super::raster::polygons;
use super::scatter::*;
use super::tests_art::straight;

/// A path node at stroke weight 1 (scatter copies at their artwork size).
pub fn line_node(points: Value, extra: Value) -> Value {
    let mut e = json!({"width": 1});
    for (k, v) in extra.as_object().unwrap() {
        e[k] = v.clone();
    }
    super::tests_art::line_node(points, e)
}
use crate::path_geom::vec2::{v2, V2};

/// A 10 × 10 square with a marker dot in its top-left quadrant.
pub fn tile_art() -> Value {
    json!([
        {"id": "sq", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3366cc"}}]},
        {"id": "dot", "type": "rectangle", "x": 1, "y": 1, "width": 2, "height": 2, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}]},
    ])
}

pub fn scatter(artwork: Value, extra: Value) -> Value {
    let mut b = json!({"scaleWithWeight": true, "id": "s", "name": "S", "type": "scatter", "artwork": {"nodes": artwork},
        "size": 1, "spacing": 1, "scatter": 0, "rotation": 0, "rotationRelativeTo": "path",
        "colorize": {"method": "none"}});
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

pub fn place(node: &Value, brush: &Value) -> Placement {
    let b = parse_scatter_brush(brush).expect("scatter brush");
    scatter_brush_for_node(node.as_object().unwrap(), &b, &default_geometry)
}

fn squares(p: &Placement) -> Vec<Vec<V2>> {
    p.paths
        .iter()
        .filter(|q| q.source_id == "sq")
        .map(|q| polygons(&q.points).concat())
        .collect()
}

fn centre(poly: &[V2]) -> V2 {
    let (mut lo, mut hi) = (
        v2(f64::INFINITY, f64::INFINITY),
        v2(f64::NEG_INFINITY, f64::NEG_INFINITY),
    );
    for p in poly {
        lo = v2(lo.x.min(p.x), lo.y.min(p.y));
        hi = v2(hi.x.max(p.x), hi.y.max(p.y));
    }
    (lo + hi) * 0.5
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn hairline_artwork_spaces_by_its_stroke() {
    // A 10 px line stroked 2 px wide with butt caps: visually 10 × 2, so
    // spacing 1 steps 10 px — 31 copies centred at 0, 10, …, 300 (not
    // thousands, as zero-height bounds would).
    let art = json!([{"id": "h", "type": "path", "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": 2}],
        "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 10, "y": 0}]}]);
    let p = place(
        &line_node(straight(300.0), json!({})),
        &scatter(art, json!({})),
    );
    assert_eq!(p.paths.len(), 31);
    assert!(p.too_complex.is_none());
}

#[test]
fn copies_touch_at_spacing_one() {
    // strokeWidth 10 = art height → size 1 copies are 10 × 10.
    let p = place(
        &line_node(straight(300.0), json!({})),
        &scatter(tile_art(), json!({})),
    );
    // Illustrator centres the first copy on the path's start.
    let sq = squares(&p);
    assert_eq!(sq.len(), 31);
    for (i, s) in sq.iter().enumerate() {
        let c = centre(s);
        assert!(near(c.x, 10.0 * i as f64) && near(c.y, 0.0), "{i}: {c:?}");
    }
    assert_eq!(p.paths[0].id, "sq#0.0");
    assert_eq!(p.paths[1].id, "dot#0.0");
    let p = place(
        &line_node(straight(300.0), json!({})),
        &scatter(tile_art(), json!({"spacing": 3})),
    );
    assert_eq!(squares(&p).len(), 11);
}

#[test]
fn size_scales_copies_and_stroke_width() {
    let p = place(
        &line_node(straight(300.0), json!({"width": 2})),
        &scatter(tile_art(), json!({"size": 0.5})),
    );
    let s = &squares(&p)[0];
    let c = centre(s);
    assert!(near(c.x, 0.0), "{c:?}");
    let w = s.iter().map(|q| q.x).fold(f64::NEG_INFINITY, f64::max)
        - s.iter().map(|q| q.x).fold(f64::INFINITY, f64::min);
    assert!(near(w, 10.0));
}

#[test]
fn rotation_relative_to_path_and_page() {
    // A downward path (+y): relative to the path, the art's x axis points down.
    let down = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 0, "y": 100}]);
    let p = place(
        &line_node(down.clone(), json!({})),
        &scatter(tile_art(), json!({})),
    );
    // The dot (art top-left) lands at the path's left-back: travel +y, left is +x.
    let dot = centre(&polygons(&p.paths[1].points).concat());
    assert!(dot.x > 0.0 && dot.y < 0.0, "{dot:?}");
    let p = place(
        &line_node(down.clone(), json!({})),
        &scatter(tile_art(), json!({"rotationRelativeTo": "page"})),
    );
    let dot = centre(&polygons(&p.paths[1].points).concat());
    assert!(
        dot.x < 0.0 && dot.y < 0.0,
        "page: unrotated, dot top-left: {dot:?}"
    );
    // 90° counter-clockwise on the page: the art's top-left goes to bottom-left.
    let p = place(
        &line_node(down, json!({})),
        &scatter(
            tile_art(),
            json!({"rotationRelativeTo": "page", "rotation": 90}),
        ),
    );
    let dot = centre(&polygons(&p.paths[1].points).concat());
    assert!(dot.x < 0.0 && dot.y > 0.0, "{dot:?}");
}

#[test]
fn scatter_offsets_to_the_left() {
    let p = place(
        &line_node(straight(100.0), json!({})),
        &scatter(tile_art(), json!({"scatter": 1.5})),
    );
    for s in squares(&p) {
        assert!(near(centre(&s).y, -15.0));
    }
}

#[test]
fn random_draws_are_seeded_per_copy() {
    let b = scatter(
        tile_art(),
        json!({
        "size": {"value": 1, "vary": "random", "min": 0.5, "max": 1.5},
        "scatter": {"value": 0, "vary": "random", "min": -1, "max": 1},
        "rotation": {"value": 0, "vary": "random", "min": -180, "max": 180}}),
    );
    let run = |seed: u32| {
        let n = line_node(
            straight(300.0),
            json!({"brush": {"brushId": "s", "seed": seed}}),
        );
        super::common::placement_json(&place(&n, &b))
    };
    assert_eq!(run(3), run(3));
    assert_ne!(run(3), run(4));
    let n = line_node(
        straight(300.0),
        json!({"brush": {"brushId": "s", "seed": 3}}),
    );
    let p = place(&n, &b);
    let ys: Vec<f64> = squares(&p).iter().map(|s| centre(s).y).collect();
    assert!(
        ys.iter().any(|y| *y > 1.0) && ys.iter().any(|y| *y < -1.0),
        "{ys:?}"
    );
    assert!(ys.iter().all(|y| y.abs() <= 15.0 + 1e-9));
}

#[test]
fn pressure_drives_size_along() {
    let n = line_node(
        straight(300.0),
        json!({"inkChannels": {"t": [0, 1], "pressure": [0, 1]}}),
    );
    let p = place(
        &n,
        &scatter(
            tile_art(),
            json!({"size": {"value": 1, "vary": "pressure", "min": 0.2, "max": 2}}),
        ),
    );
    let sq = squares(&p);
    let width = |s: &Vec<V2>| {
        s.iter().map(|q| q.x).fold(f64::NEG_INFINITY, f64::max)
            - s.iter().map(|q| q.x).fold(f64::INFINITY, f64::min)
    };
    assert!(width(&sq[0]) < 3.0 && width(sq.last().unwrap()) > 15.0);
}

#[test]
fn closed_paths_do_not_double_the_seam() {
    let sqr = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
        {"type": "line", "x": 100, "y": 100}, {"type": "line", "x": 0, "y": 100}]);
    let p = place(
        &line_node(sqr, json!({"closed": true})),
        &scatter(tile_art(), json!({})),
    );
    let sq = squares(&p);
    assert_eq!(sq.len(), 40);
    assert!(centre(&sq[0]).dist(v2(0.0, 0.0)) < 1e-6);
}

#[test]
fn colorize_and_overrides() {
    let n = line_node(
        straight(50.0),
        json!({"paint": {"type": "solid", "color": "#d03a2f"},
        "brush": {"brushId": "s", "overrides": {"scale": 2}}}),
    );
    let p = place(
        &n,
        &scatter(tile_art(), json!({"colorize": {"method": "tints"}})),
    );
    assert_eq!(p.paths[1].paint["fills"][0]["paint"]["color"], "#d03a2f", "black → stroke colour");
    // Size 2 makes 20 px copies, but the step stays the art's 10 px
    // (Illustrator's rule): 6 copies on 50 px, overlapping.
    assert_eq!(squares(&p).len(), 6, "size scales copies, not the step");
}

#[test]
fn zero_size_and_degenerate_inputs_are_bounded() {
    // Far too many copies: refused with a reason, quickly.
    for extra in [json!({"size": 0, "spacing": 0}), json!({"spacing": 0})] {
        let p = place(
            &line_node(straight(1e6), json!({})),
            &scatter(tile_art(), extra),
        );
        assert!(p.too_complex.as_deref().unwrap().contains("copies"));
        let v: Value = crate::value::from_str(&super::common::placement_json(&p)).unwrap();
        assert_eq!(v["code"], "too_complex");
    }
    let p = place(
        &line_node(straight(100.0), json!({})),
        &scatter(json!([{"id": "t", "type": "text"}]), json!({})),
    );
    assert!(p.paths.is_empty());
    assert_eq!(p.skipped, vec!["t".to_string()]);
}

#[test]
fn json_dispatch() {
    let req = json!({"node": line_node(straight(30.0), json!({})), "brush": scatter(tile_art(), json!({}))});
    let v: Value = crate::value::from_str(&super::stroke_brush_json(&req.to_string())).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["nodes"].as_array().unwrap().len(), 8);
    let v: Value = crate::value::from_str(&scatter_brush_json(
        &json!({"node": line_node(straight(30.0), json!({})), "brush": {"id": "x", "type": "art"}})
            .to_string(),
    ))
    .unwrap();
    assert_eq!(v["code"], "invalid_brush");
    // The dispatcher names the offending type instead of blaming the art parser.
    for (brush, needle) in [
        (
            json!({"id": "c", "type": "calligraphic"}),
            "\"calligraphic\"",
        ),
        (json!({"id": "c"}), "none"),
    ] {
        let req = json!({"node": line_node(straight(30.0), json!({})), "brush": brush});
        let v: Value = crate::value::from_str(&super::stroke_brush_json(&req.to_string())).unwrap();
        assert_eq!(v["code"], "invalid_brush");
        assert!(v["message"].as_str().unwrap().contains(needle), "{v}");
    }
}

#[test]
fn fuzz_no_panics() {
    let mut seed = 0x9e37u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for case in 0..50 {
        let mut pts =
            vec![json!({"type": "move", "x": next() * 400.0 - 200.0, "y": next() * 400.0})];
        for _ in 0..(1 + (next() * 6.0) as usize) {
            let v = if next() < 0.03 {
                1e12
            } else {
                next() * 400.0 - 200.0
            };
            pts.push(json!({"type": if next() < 0.5 { "line" } else { "curve" }, "x": v, "y": next() * 300.0,
                "cp1": {"x": next() * 300.0, "y": next() * 300.0}, "cp2": {"x": next() * 300.0, "y": 0}}));
        }
        let dyn_ = |lo: f64, hi: f64, n: f64| {
            let vary = ["random", "pressure", "tilt", "rotation"][(n * 4.0) as usize % 4];
            json!({"value": lo, "vary": vary, "min": lo, "max": hi})
        };
        let b = scatter(
            tile_art(),
            json!({
                "size": dyn_(next() * 2.0, next() * 3.0, next()),
                "spacing": dyn_(next() * 2.0, next() * 3.0, next()),
                "scatter": dyn_(-next() * 3.0, next() * 3.0, next()),
                "rotation": dyn_(-360.0, 360.0, next()),
                "rotationRelativeTo": if next() < 0.5 { "page" } else { "path" },
                "colorize": {"method": "hue-shift"},
            }),
        );
        let n = line_node(
            Value::Array(pts),
            json!({"closed": next() < 0.3, "width": next() * 4.0,
            "inkChannels": {"t": [0, 1], "pressure": [next(), next()], "twist": [0, 300]},
            "brush": {"brushId": "s", "seed": case}}),
        );
        let out: Value = crate::value::from_str(&scatter_brush_json(
            &json!({"node": n, "brush": b}).to_string(),
        ))
        .unwrap();
        let refused = out["status"] == "error" && out["code"] == "too_complex";
        assert!(
            out["status"] == "ok" || out["status"] == "empty" || refused,
            "{case}: {out}"
        );
    }
}

#[test]
#[ignore]
fn perf_scatter_1000_copies() {
    let n = line_node(straight(10_000.0), json!({"width": 1}));
    let mut art = Vec::new();
    for i in 0..10 {
        art.push(json!({"id": format!("e{i}"), "type": "ellipse", "x": i, "y": i, "width": 10, "height": 10, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#123"}}]}));
    }
    let b = parse_scatter_brush(&scatter(
        Value::Array(art),
        json!({"rotation": {"value": 0, "vary": "random", "min": 0, "max": 360}}),
    ))
    .unwrap();
    let node = n.as_object().unwrap().clone();
    let mut count = 0;
    let mut times: Vec<f64> = (0..20)
        .map(|_| {
            let t0 = std::time::Instant::now();
            count = scatter_brush_for_node(&node, &b, &default_geometry)
                .paths
                .len();
            t0.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    times.sort_by(|a, b| a.total_cmp(b));
    println!(
        "scatter: best {:.3} ms, median {:.3} ms per placement, {count} paths",
        times[0],
        times[times.len() / 2]
    );
}

#[test]
fn output_is_capped_in_shapes_and_bytes() {
    // 5 shapes × ~1300 copies: the engine's direct call gets them all (bounded
    // by the copy / segment budgets); the JSON lane refuses over 4000 shapes.
    let art: Vec<Value> = (0..5)
        .map(|i| json!({"id": format!("e{i}"), "type": "ellipse", "x": i, "y": 0, "width": 2, "height": 2, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}]}))
        .collect();
    let p = place(
        &line_node(straight(8000.0), json!({})),
        &scatter(Value::Array(art), json!({})),
    );
    assert!(p.too_complex.is_none(), "{:?}", p.too_complex);
    assert!(p.paths.len() > super::common::MAX_OUTPUT_PATHS);
    let v: Value = crate::value::from_str(&super::common::placement_json(&p)).unwrap();
    assert_eq!(v["code"], "too_complex");
    assert!(v["message"].as_str().unwrap().contains("shapes"));
}

#[test]
fn copies_carry_their_arc_length() {
    let p = place(
        &line_node(straight(100.0), json!({})),
        &scatter(tile_art(), json!({})),
    );
    let sq: Vec<_> = p.paths.iter().filter(|q| q.source_id == "sq").collect();
    for (i, q) in sq.iter().enumerate() {
        let c = 10.0 * i as f64;
        assert!(
            (q.at.s - c).abs() < 1e-6
                && (q.at.s_span.0 - (c - 5.0)).abs() < 1e-6
                && (q.at.s_span.1 - (c + 5.0)).abs() < 1e-6,
            "{i}: {:?}",
            q.at
        );
        assert_eq!(q.at.subpath, 0);
    }
    // A shape and its marker from the same copy share the placement.
    assert_eq!(p.paths[0].at, p.paths[1].at);
    // Rotated copies: the extent is the rotated box on the tangent (a 20 × 10
    // tile turned 90° spans its height, 10, along the path).
    let wide = json!([{"id": "w", "type": "rectangle", "x": 0, "y": 0, "width": 20, "height": 10, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}]}]);
    let p = place(
        &line_node(straight(100.0), json!({})),
        &scatter(wide, json!({"rotation": 90})),
    );
    let at = p.paths[0].at;
    assert!((at.s_span.1 - at.s_span.0 - 10.0).abs() < 1e-6, "{at:?}");
}

#[test]
fn the_json_lane_stops_at_the_shape_cap_before_copying_more() {
    let art: Vec<Value> = (0..5)
        .map(|i| json!({"id": format!("e{i}"), "type": "ellipse", "x": i, "y": 0, "width": 2, "height": 2, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}]}))
        .collect();
    let req = json!({"node": line_node(straight(8000.0), json!({})), "brush": scatter(Value::Array(art), json!({}))});
    let v: Value = crate::value::from_str(&scatter_brush_json(&req.to_string())).unwrap();
    assert_eq!(v["code"], "too_complex");
    // The cap is per call: a direct call right after is unlimited again.
    let small = place(
        &line_node(straight(8000.0), json!({})),
        &scatter(tile_art(), json!({})),
    );
    assert!(small.too_complex.is_none());
}

/// Illustrator's spacing rules (measured on its stock brushes; see
/// docs/roadmap/brushes/illustrator-comparison.md, "Art and scatter brushes").
#[test]
fn spacing_and_scatter_follow_illustrator() {
    let wide = json!([{"id": "sq", "type": "rectangle", "x": 0, "y": 0, "width": 8, "height": 20, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}]}]);
    let xs = |node: Value, brush: Value| -> Vec<f64> {
        squares(&place(&node, &brush))
            .iter()
            .map(|s| (centre(s).x * 1e6).round() / 1e6)
            .collect()
    };
    // The step is the artwork's LARGER dimension (20, its height here) ×
    // spacing × weight, from the path's start.
    let a = xs(
        line_node(straight(100.0), json!({})),
        scatter(wide.clone(), json!({})),
    );
    assert_eq!(a, vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0]);
    let a = xs(
        line_node(straight(100.0), json!({"width": 2})),
        scatter(wide.clone(), json!({"spacing": 0.5})),
    );
    assert_eq!(
        a,
        vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
        "weight 2 × spacing 0.5"
    );
    // Random sizes change the copies, never the step.
    let a = xs(
        line_node(
            straight(100.0),
            json!({"brush": {"brushId": "s", "seed": 7}}),
        ),
        scatter(
            wide.clone(),
            json!({"size": {"value": 1, "vary": "random", "min": 0.5, "max": 2}}),
        ),
    );
    assert_eq!(a, vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0]);
    // The scatter offset is also in the larger dimension × weight, whatever the size.
    let p = place(
        &line_node(straight(10.0), json!({"width": 2})),
        &scatter(wide, json!({"scatter": 0.5, "size": 3})),
    );
    let c = centre(&squares(&p)[0]);
    assert!(near(c.y, -20.0), "0.5 × 20 × weight 2 to the left: {c:?}");
}

#[test]
fn per_stroke_spacing_scatter_and_rotation_overrides() {
    let with = |overrides: Value| {
        let node = line_node(
            straight(300.0),
            json!({"brush": {"brushId": "s", "overrides": overrides}}),
        );
        place(&node, &scatter(tile_art(), json!({})))
    };
    assert_eq!(squares(&with(json!({}))).len(), 31);
    // Spacing 3 → 11 copies.
    assert_eq!(squares(&with(json!({"spacing": 3}))).len(), 11);
    // Scatter 1 (one artwork size to the left of the path, −y on a +x path).
    for s in squares(&with(json!({"scatter": 1}))) {
        assert!(near(centre(&s).y, -10.0), "{:?}", centre(&s));
    }
    // Rotation 90° relative to the page on a +x path: the marker dot (top-left
    // quadrant of the upright square) turns to the bottom-left.
    let p = with(json!({"rotation": 90, "rotationRelativeTo": "page"}));
    let dot = p.paths.iter().find(|q| q.source_id == "dot").unwrap();
    let c = centre(&polygons(&dot.points).concat());
    assert!(c.x < 0.0 && c.y > 0.0, "{c:?}");
    let o = parse_scatter_overrides(
        json!({"spacing": {"value": 2, "vary": "pressure", "min": 1, "max": 3}, "rotationRelativeTo": "page"}).as_object(),
    );
    assert_eq!(o.spacing.unwrap().max, 3.0);
    assert_eq!(o.relative_to, Some(RotationRelativeTo::Page));
    assert!(o.scatter.is_none() && o.rotation.is_none());
}
