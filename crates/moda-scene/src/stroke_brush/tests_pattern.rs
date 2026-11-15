//! Pattern brush tests.

use crate::value::{json, Value};

use super::artwork::default_geometry;
use super::common::Placement;
use super::pattern::*;
use super::raster::polygons;
use super::tests_art::straight;
use super::tests_scatter::line_node;
use crate::path_geom::vec2::{v2, V2};

/// A `w × 10` tile: a body rectangle plus a marker in its top-left corner.
pub fn tile(id: &str, w: f64, fill: &str) -> Value {
    json!({"nodes": [
        {"id": id, "type": "rectangle", "x": 0, "y": 0, "width": w, "height": 10, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": fill}}]},
        {"id": format!("{id}-m"), "type": "rectangle", "x": 0, "y": 0, "width": 2, "height": 2, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}]},
    ]})
}

pub fn pattern(tiles: Value, extra: Value) -> Value {
    let mut b = json!({"scaleWithWeight": true, "id": "p", "name": "P", "type": "pattern", "tiles": tiles, "scale": 1,
        "spacing": 0, "fit": "stretch", "colorize": {"method": "none"}});
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

fn side_only() -> Value {
    json!({"side": tile("s", 10.0, "#3366cc")})
}

pub fn place(node: &Value, brush: &Value) -> Placement {
    let b = parse_pattern_brush(brush).expect("pattern brush");
    pattern_brush_for_node(node.as_object().unwrap(), &b, &default_geometry, 0.1)
}

fn boxes(p: &Placement, source: &str) -> Vec<(V2, V2)> {
    p.paths
        .iter()
        .filter(|q| q.source_id == source)
        .map(|q| {
            let pts = polygons(&q.points).concat();
            let mut lo = v2(f64::INFINITY, f64::INFINITY);
            let mut hi = v2(f64::NEG_INFINITY, f64::NEG_INFINITY);
            for p in pts {
                lo = v2(lo.x.min(p.x), lo.y.min(p.y));
                hi = v2(hi.x.max(p.x), hi.y.max(p.y));
            }
            (lo, hi)
        })
        .collect()
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn stretch_fits_whole_tiles() {
    let p = place(
        &line_node(straight(100.0), json!({})),
        &pattern(side_only(), json!({})),
    );
    let b = boxes(&p, "s");
    assert_eq!(b.len(), 10);
    for (i, (lo, hi)) in b.iter().enumerate() {
        assert!(
            near(lo.x, 10.0 * i as f64) && near(hi.x, 10.0 * (i + 1) as f64),
            "{i}"
        );
        assert!(near(lo.y, -5.0) && near(hi.y, 5.0));
    }
    let p = place(
        &line_node(straight(104.0), json!({})),
        &pattern(side_only(), json!({})),
    );
    let b = boxes(&p, "s");
    assert_eq!(b.len(), 10);
    assert!(near(b[9].1.x, 104.0) && near(b[0].1.x, 10.4));
    assert_eq!(p.paths[0].id, "s#0.0.0");
}

/// Illustrator's convention (matched on stock Gold Leaf): an unpainted rectangle behind a tile is its bounding
/// tile. It sets the tile's size and where its artwork sits across the path, even where nothing is painted.
#[test]
fn an_unpainted_bounding_rectangle_sets_the_tile_size() {
    let side = json!({"nodes": [
        {"id": "box", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 30, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "rgba(0,0,0,0)"}}]},
        {"id": "s", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3366cc"}}]},
    ]});
    let p = place(
        &line_node(straight(100.0), json!({})),
        &pattern(json!({"side": side}), json!({})),
    );
    let b = boxes(&p, "s");
    assert_eq!(b.len(), 10);
    // The 30-tall tile is centred on the path; its painted top third sits on the left (−y) of travel.
    for (lo, hi) in &b {
        assert!(near(lo.y, -15.0) && near(hi.y, -5.0), "{lo:?} {hi:?}");
    }
}

#[test]
fn add_space_and_approximate_keep_tile_size() {
    let p = place(
        &line_node(straight(105.0), json!({})),
        &pattern(side_only(), json!({"fit": "add-space"})),
    );
    let b = boxes(&p, "s");
    assert_eq!(b.len(), 10);
    assert!(
        near(b[0].0.x, 0.25) && near(b[0].1.x, 10.25) && near(b[1].0.x, 10.75),
        "{:?}",
        b[0]
    );
    let p = place(
        &line_node(straight(104.0), json!({})),
        &pattern(side_only(), json!({"fit": "approximate-path"})),
    );
    let b = boxes(&p, "s");
    assert_eq!(b.len(), 10);
    assert!(near(b[0].0.x, 2.0) && near(b[9].1.x, 102.0));
    // Overshoot: 10 tiles on 96 px extend 2 px past both ends.
    let p = place(
        &line_node(straight(96.0), json!({})),
        &pattern(side_only(), json!({"fit": "approximate-path"})),
    );
    let b = boxes(&p, "s");
    assert!(near(b[0].0.x, -2.0) && near(b[9].1.x, 98.0));
}

#[test]
fn spacing_adds_gaps() {
    let p = place(
        &line_node(straight(150.0), json!({})),
        &pattern(side_only(), json!({"spacing": 0.5})),
    );
    let b = boxes(&p, "s");
    assert_eq!(b.len(), 10);
    assert!(near(b[0].0.x, 2.5) && near(b[0].1.x, 12.5) && near(b[1].0.x, 17.5));
}

#[test]
fn start_and_end_tiles_and_flip_along() {
    let tiles = json!({"side": tile("s", 10.0, "#3366cc"), "start": tile("st", 5.0, "#d03a2f"),
        "end": tile("en", 5.0, "#2a9d4a")});
    let p = place(
        &line_node(straight(110.0), json!({})),
        &pattern(tiles.clone(), json!({})),
    );
    let st = boxes(&p, "st")[0];
    let en = boxes(&p, "en")[0];
    assert!(near(st.0.x, 0.0) && near(st.1.x, 5.0) && near(en.0.x, 105.0) && near(en.1.x, 110.0));
    assert_eq!(boxes(&p, "s").len(), 10);
    let p = place(
        &line_node(straight(110.0), json!({})),
        &pattern(tiles, json!({"flipAlong": true})),
    );
    assert!(near(boxes(&p, "en")[0].0.x, 0.0) && near(boxes(&p, "st")[0].1.x, 110.0));
    // Mirrored along: the side tile's marker moves to its right end.
    let m = boxes(&p, "s-m")[0];
    assert!(near(m.1.x, 15.0) && near(m.0.y, -5.0), "{m:?}");
}

#[test]
fn flip_across_mirrors_over_the_path() {
    let p = place(
        &line_node(straight(100.0), json!({})),
        &pattern(side_only(), json!({"flipAcross": true})),
    );
    let m = boxes(&p, "s-m")[0];
    assert!(near(m.1.y, 5.0) && near(m.0.y, 3.0), "{m:?}");
}

fn l_path(turn_right: bool) -> Value {
    let y = if turn_right { 100.0 } else { -100.0 };
    json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
           {"type": "line", "x": 100, "y": y}])
}

#[test]
fn outer_corner_tile_fills_the_corner_square() {
    let tiles = json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": tile("oc", 10.0, "#d03a2f"),
        "innerCorner": tile("ic", 10.0, "#2a9d4a")});
    // Right turn: incoming +x, outgoing +y (down). The tile is drawn as a clockwise
    // frame's top-left corner (incoming leg running up): its top-left goes to the
    // outer miter (105, −5); the incoming side joins its bottom, the outgoing its right.
    let p = place(
        &line_node(l_path(true), json!({})),
        &pattern(tiles.clone(), json!({})),
    );
    let oc = boxes(&p, "oc");
    assert_eq!(oc.len(), 1);
    assert!(
        near(oc[0].0.x, 95.0)
            && near(oc[0].1.x, 105.0)
            && near(oc[0].0.y, -5.0)
            && near(oc[0].1.y, 5.0),
        "{oc:?}"
    );
    let m = boxes(&p, "oc-m")[0];
    assert!(near(m.1.x, 105.0) && near(m.0.y, -5.0), "{m:?}");
    // Legs are cut back by half the thickness: incoming side tiles end at x = 95.
    let s = boxes(&p, "s");
    let max_incoming = s
        .iter()
        .filter(|b| b.1.y <= 5.0 + 1e-9)
        .map(|b| b.1.x)
        .fold(0.0, f64::max);
    assert!(near(max_incoming, 95.0), "{max_incoming}");
    assert!(boxes(&p, "ic").is_empty());
    // Left turn (incoming +x, outgoing −y): the inner tile fills the same
    // square, MIRRORED across the incoming leg (Illustrator): its authored
    // top-left lands at (105, 5), the incoming side on its bottom edge (x = 95)
    // and the outgoing side on its right edge (y = −5).
    let p = place(
        &line_node(l_path(false), json!({})),
        &pattern(tiles, json!({})),
    );
    let ic = boxes(&p, "ic");
    assert_eq!(ic.len(), 1);
    assert!(near(ic[0].0.x, 95.0) && near(ic[0].1.y, 5.0) && near(ic[0].0.y, -5.0));
    let m = boxes(&p, "ic-m")[0];
    assert!(near(m.1.x, 105.0) && near(m.1.y, 5.0), "{m:?}");
}

#[test]
fn a_corner_kind_without_a_tile_stays_empty() {
    // Illustrator: no tile for the corner (and no auto mode) → a bare corner.
    let p = place(
        &line_node(l_path(true), json!({})),
        &pattern(side_only(), json!({})),
    );
    assert!(
        p.paths.iter().any(|q| q.id.starts_with("s#0.1.")),
        "two legs"
    );
    let incoming_max = p
        .paths
        .iter()
        .filter(|q| q.id.starts_with("s#0.0.") && q.source_id == "s")
        .flat_map(|q| polygons(&q.points).concat())
        .map(|q| q.x)
        .fold(0.0, f64::max);
    assert!(near(incoming_max, 95.0), "{incoming_max}");
    // Nothing covers the corner square's outer quadrant.
    for q in p.paths.iter().flat_map(|q| polygons(&q.points).concat()) {
        assert!(!(q.x > 95.5 && q.y < -0.5), "{q:?} in the empty corner");
    }
}

#[test]
fn between_meets_at_the_miter() {
    let tiles = json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": "between"});
    let p = place(
        &line_node(l_path(true), json!({})),
        &pattern(tiles, json!({})),
    );
    let all: Vec<V2> = p
        .paths
        .iter()
        .filter(|q| q.source_id == "s")
        .flat_map(|q| polygons(&q.points).concat())
        .collect();
    // Nothing crosses the miter line through (100, 0) along (1, −1)… from either leg.
    let legs: Vec<&str> = p.paths.iter().map(|q| q.id.as_str()).collect();
    assert!(legs.iter().any(|id| id.starts_with("s#0.1.")));
    for q in p.paths.iter().filter(|q| q.id.starts_with("s#0.0.")) {
        for pt in polygons(&q.points).concat() {
            assert!((pt.x - 100.0) + (pt.y - 0.0) <= 1e-6, "{pt:?}");
        }
    }
    // The outer miter point is reached.
    assert!(all.iter().any(|q| q.dist(v2(105.0, -5.0)) < 1e-6));
}

#[test]
fn sliced_halves_join_at_the_miter() {
    let tiles = json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": "sliced"});
    let p = place(
        &line_node(l_path(true), json!({})),
        &pattern(tiles, json!({})),
    );
    // Both legs clip their corner tile at the miter: incoming pieces stay before it.
    for q in p.paths.iter().filter(|q| q.id.starts_with("s#0.0.")) {
        for pt in polygons(&q.points).concat() {
            assert!((pt.x - 100.0) + pt.y <= 1e-6, "{pt:?}");
        }
    }
    for q in p.paths.iter().filter(|q| q.id.starts_with("s#0.1.")) {
        for pt in polygons(&q.points).concat() {
            assert!((pt.x - 100.0) + pt.y >= -1e-6, "{pt:?}");
        }
    }
    // A half tile at the corner on each leg: 10.5 → rounds to 11 tiles per leg with the extra half.
    let first_leg = p
        .paths
        .iter()
        .filter(|q| q.id.starts_with("s#0.0.") && q.source_id == "s")
        .count();
    assert_eq!(first_leg, 11);
}

#[test]
fn overlap_extends_past_the_corner() {
    let tiles = json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": "overlap"});
    let p = place(
        &line_node(l_path(true), json!({})),
        &pattern(tiles, json!({})),
    );
    let incoming = p
        .paths
        .iter()
        .filter(|q| q.id.starts_with("s#0.0.") && q.source_id == "s");
    let max_x = incoming
        .flat_map(|q| polygons(&q.points).concat())
        .map(|q| q.x)
        .fold(0.0, f64::max);
    assert!(near(max_x, 105.0), "{max_x}");
}

#[test]
fn closed_rectangle_with_corners() {
    let rect = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
        {"type": "line", "x": 100, "y": 60}, {"type": "line", "x": 0, "y": 60}]);
    let tiles =
        json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": tile("oc", 10.0, "#d03a2f")});
    let p = place(
        &line_node(rect, json!({"closed": true})),
        &pattern(tiles, json!({})),
    );
    let oc = boxes(&p, "oc");
    assert_eq!(oc.len(), 4);
    let mut centres: Vec<(i64, i64)> = oc
        .iter()
        .map(|b| {
            (
                ((b.0.x + b.1.x) / 2.0).round() as i64,
                ((b.0.y + b.1.y) / 2.0).round() as i64,
            )
        })
        .collect();
    centres.sort();
    assert_eq!(centres, vec![(0, 0), (0, 60), (100, 0), (100, 60)]);
    // 90 px and 50 px legs after the cuts: 9 + 5 + 9 + 5 tiles.
    assert_eq!(boxes(&p, "s").len(), 28);
}

#[test]
fn closed_loop_without_corners_is_seamless() {
    let r = 50.0;
    let k = r * 0.5522847498;
    let circle = json!([{"type": "move", "x": r, "y": 0},
        {"type": "curve", "x": 0, "y": r, "cp1": {"x": r, "y": k}, "cp2": {"x": k, "y": r}},
        {"type": "curve", "x": -r, "y": 0, "cp1": {"x": -k, "y": r}, "cp2": {"x": -r, "y": k}},
        {"type": "curve", "x": 0, "y": -r, "cp1": {"x": -r, "y": -k}, "cp2": {"x": -k, "y": -r}},
        {"type": "curve", "x": r, "y": 0, "cp1": {"x": k, "y": -r}, "cp2": {"x": r, "y": -k}}]);
    let p = place(
        &line_node(circle, json!({"closed": true})),
        &pattern(side_only(), json!({})),
    );
    let n = boxes(&p, "s").len();
    assert_eq!(n, 31, "round(2π·50 / 10)");
    for q in p.paths.iter().filter(|q| q.source_id == "s") {
        for pt in polygons(&q.points).concat() {
            assert!(pt.len() > 44.9 && pt.len() < 55.1, "{}", pt.len());
        }
    }
}

#[test]
fn scale_colorize_and_dispatch() {
    let n = line_node(
        straight(100.0),
        json!({"paint": {"type": "solid", "color": "#d03a2f"}, "width": 2}),
    );
    let p = place(
        &n,
        &pattern(side_only(), json!({"colorize": {"method": "tints"}})),
    );
    assert_eq!(boxes(&p, "s").len(), 5, "20 px tiles");
    let m = p.paths.iter().find(|q| q.source_id == "s-m").unwrap();
    assert_eq!(m.paint["fills"][0]["paint"]["color"], "#d03a2f");
    let req = json!({"node": line_node(straight(30.0), json!({})), "brush": pattern(side_only(), json!({}))});
    let v: Value = crate::value::from_str(&super::stroke_brush_json(&req.to_string())).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["nodes"].as_array().unwrap().len(), 6);
    let bad = json!({"node": line_node(straight(30.0), json!({})), "brush": {"id": "p", "type": "pattern", "tiles": {}}});
    let v: Value = crate::value::from_str(&super::stroke_brush_json(&bad.to_string())).unwrap();
    assert_eq!(v["code"], "invalid_brush");
}

#[test]
fn parse_mirrors_normalize() {
    let b = parse_pattern_brush(&json!({"id": "p", "type": "pattern", "tiles": {"side": {"nodes": []},
        "outerCorner": "weird", "innerCorner": "between", "start": 5}, "spacing": -3, "fit": "stretch-ish"})).unwrap();
    assert_eq!(b.outer_corner, None);
    assert_eq!(b.inner_corner, Some(CornerTile::Auto(AutoCorner::Between)));
    assert_eq!(b.start, None);
    assert_eq!(b.spacing, 0.0);
    assert_eq!(b.fit, PatternFit::Stretch);
    assert!(parse_pattern_brush(
        &json!({"id": "p", "type": "pattern", "tiles": {"outerCorner": "between"}})
    )
    .is_none());
}

#[test]
fn fuzz_no_panics() {
    let mut seed = 0x1234_5678u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let corners = [
        json!("centered"),
        json!("between"),
        json!("sliced"),
        json!("overlap"),
        json!(null),
        tile("c", 7.0, "#123456"),
    ];
    for case in 0..40 {
        let mut pts = vec![json!({"type": "move", "x": next() * 300.0, "y": next() * 300.0})];
        for _ in 0..(1 + (next() * 8.0) as usize) {
            let t = if next() < 0.6 { "line" } else { "curve" };
            pts.push(json!({"type": t, "x": next() * 300.0, "y": next() * 300.0,
                "cp1": {"x": next() * 300.0, "y": next() * 300.0}, "cp2": {"x": next() * 300.0, "y": next() * 300.0}}));
        }
        let pick = |x: f64| corners[(x * corners.len() as f64) as usize % corners.len()].clone();
        let mut tiles = json!({"side": tile("s", 1.0 + next() * 20.0, "#3366cc")});
        let (o, i) = (pick(next()), pick(next()));
        if !o.is_null() {
            tiles["outerCorner"] = o;
        }
        if !i.is_null() {
            tiles["innerCorner"] = i;
        }
        if next() < 0.5 {
            tiles["start"] = tile("st", 4.0, "#d03a2f");
        }
        if next() < 0.5 {
            tiles["end"] = tile("en", 4.0, "#2a9d4a");
        }
        let fits = ["stretch", "add-space", "approximate-path"];
        let b = pattern(
            tiles,
            json!({"fit": fits[(next() * 3.0) as usize % 3], "spacing": next() * 2.0,
            "flipAlong": next() < 0.5, "flipAcross": next() < 0.5, "scale": next() * 2.0}),
        );
        let n = line_node(
            Value::Array(pts),
            json!({"closed": next() < 0.4, "width": next() * 3.0,
            "cornerRadius": if next() < 0.2 { 10.0 } else { 0.0 }}),
        );
        // Half the cases trimmed / dashed: random windows, some wrapping or past the ends.
        let mut req = json!({"node": n, "brush": b});
        if next() < 0.5 {
            let ws: Vec<Value> = (0..(1 + (next() * 3.0) as usize))
                .map(|_| {
                    let a = next() * 1200.0 - 100.0;
                    json!({"subpath": 0, "start": a, "end": a + next() * 600.0})
                })
                .collect();
            req["windows"] = Value::Array(ws);
        }
        let out: Value = crate::value::from_str(&super::stroke_brush_json(&req.to_string())).unwrap();
        assert!(
            out["status"] == "ok"
                || out["status"] == "empty"
                || (out["status"] == "error" && out["code"] == "too_complex"),
            "{case}: {out}"
        );
        if let Some(nodes) = out["nodes"].as_array() {
            for node in nodes {
                for p in node["points"].as_array().unwrap() {
                    assert!(p["x"].as_f64().is_some_and(f64::is_finite), "{case}");
                }
            }
        }
    }
}

#[test]
#[ignore]
fn perf_pattern() {
    // A 300-anchor zigzag (a corner at every anchor) with corner tiles and sliced corners.
    let mut pts = vec![json!({"type": "move", "x": 0, "y": 0})];
    for i in 1..300 {
        pts.push(
            json!({"type": "line", "x": i as f64 * 20.0, "y": if i % 2 == 0 { 0.0 } else { 30.0 }}),
        );
    }
    let tiles = json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": tile("oc", 10.0, "#d03a2f"), "innerCorner": "sliced"});
    let b = parse_pattern_brush(&pattern(tiles, json!({}))).unwrap();
    let node = line_node(Value::Array(pts), json!({"width": 0.8}))
        .as_object()
        .unwrap()
        .clone();
    let runs = 10;
    let t0 = std::time::Instant::now();
    let mut count = 0;
    for _ in 0..runs {
        count = pattern_brush_for_node(&node, &b, &default_geometry, 0.25)
            .paths
            .len();
    }
    println!(
        "pattern: {:.3} ms per placement, {count} paths",
        t0.elapsed().as_secs_f64() * 1000.0 / runs as f64
    );
}

fn polyline_node(pts: &[(f64, f64)], extra: Value) -> Value {
    let mut v = vec![json!({"type": "move", "x": pts[0].0, "y": pts[0].1})];
    for p in &pts[1..] {
        v.push(json!({"type": "line", "x": p.0, "y": p.1}));
    }
    line_node(Value::Array(v), extra)
}

#[test]
fn huge_tile_counts_refuse_instead_of_overflowing() {
    // #56 review repro: a 1e-7 leg, a 90° between corner, a 1e9 leg, weight 2e-12.
    let n = polyline_node(
        &[(0.0, 0.0), (1e-7, 0.0), (1e-7, 1e9)],
        json!({"width": 2e-12}),
    );
    let tiles = json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": "between", "innerCorner": "between"});
    let t = std::time::Instant::now();
    let p = place(&n, &pattern(tiles, json!({})));
    assert!(t.elapsed().as_secs_f64() < 5.0);
    assert!(
        p.too_complex.is_some() || p.paths.is_empty(),
        "{:?}",
        p.too_complex
    );
}

#[test]
fn corner_and_end_tiles_count_toward_the_budget() {
    // 12 000 corners, each with a corner tile, and almost no room for side tiles.
    let pts: Vec<(f64, f64)> = (0..12_000)
        .map(|i| (i as f64 * 20.0, if i % 2 == 0 { 0.0 } else { 20.0 }))
        .collect();
    let tiles = json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": tile("oc", 10.0, "#d03a2f"), "innerCorner": tile("ic", 10.0, "#2a9d4a")});
    let b = parse_pattern_brush(&pattern(tiles, json!({}))).unwrap();
    let p = pattern_brush_for_node(
        polyline_node(&pts, json!({})).as_object().unwrap(),
        &b,
        &default_geometry,
        0.25,
    );
    assert!(
        p.too_complex
            .as_deref()
            .is_some_and(|m| m.contains("tiles")),
        "{:?}",
        p.too_complex
    );
}

#[test]
fn flip_along_mirrors_corner_tiles_across_their_diagonal() {
    // An asymmetric corner tile: a marker at its top-right.
    let corner = json!({"nodes": [
        {"id": "oc", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#d03a2f"}}]},
        {"id": "oc-m", "type": "rectangle", "x": 8, "y": 0, "width": 2, "height": 2, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}]},
    ]});
    let tiles = json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": corner});
    let n = line_node(l_path(true), json!({}));
    // Tile top-right → the outgoing cut's outer corner (105, 5).
    let m = boxes(&place(&n, &pattern(tiles.clone(), json!({}))), "oc-m")[0];
    assert!(near(m.1.x, 105.0) && near(m.1.y, 5.0), "{m:?}");
    // Flipped along: mirrored across the diagonal that swaps the join edges —
    // top-right → bottom-left, the incoming cut's outer corner (95, −5); the tile
    // still fills the corner square.
    let flipped = place(&n, &pattern(tiles, json!({"flipAlong": true})));
    let oc = boxes(&flipped, "oc")[0];
    assert!(
        near(oc.0.x, 95.0) && near(oc.1.x, 105.0) && near(oc.0.y, -5.0) && near(oc.1.y, 5.0),
        "{oc:?}"
    );
    let m = boxes(&flipped, "oc-m")[0];
    assert!(near(m.0.x, 95.0) && near(m.0.y, -5.0), "{m:?}");
}

#[test]
fn a_short_leg_shares_its_corner_cut() {
    // A 6 px riser between two corner tiles (half-thickness 5): both corners cut
    // 3, on both of their legs, so side tiles meet the corner tiles (no gap).
    let n = polyline_node(
        &[(0.0, 0.0), (100.0, 0.0), (100.0, 6.0), (200.0, 6.0)],
        json!({}),
    );
    let tiles = json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": tile("oc", 10.0, "#d03a2f"), "innerCorner": tile("ic", 10.0, "#2a9d4a")});
    let p = place(&n, &pattern(tiles, json!({})));
    let first_leg_end = p
        .paths
        .iter()
        .filter(|q| q.source_id == "s" && q.id.starts_with("s#0.0."))
        .flat_map(|q| polygons(&q.points).concat())
        .map(|q| q.x)
        .fold(0.0, f64::max);
    let oc = boxes(&p, "oc")[0];
    assert!(
        near(first_leg_end, 97.0) && near(oc.0.x, 97.0),
        "{first_leg_end} vs {oc:?}"
    );
}

#[test]
fn approximate_path_is_seamless_on_closed_loops_and_stays_off_corner_tiles() {
    let r = 50.0;
    let k = r * 0.5522847498;
    let circle = json!([{"type": "move", "x": r, "y": 0},
        {"type": "curve", "x": 0, "y": r, "cp1": {"x": r, "y": k}, "cp2": {"x": k, "y": r}},
        {"type": "curve", "x": -r, "y": 0, "cp1": {"x": -k, "y": r}, "cp2": {"x": -r, "y": k}},
        {"type": "curve", "x": 0, "y": -r, "cp1": {"x": -r, "y": -k}, "cp2": {"x": -k, "y": -r}},
        {"type": "curve", "x": r, "y": 0, "cp1": {"x": k, "y": -r}, "cp2": {"x": r, "y": -k}}]);
    let p = place(
        &line_node(circle, json!({"closed": true})),
        &pattern(
            side_only(),
            json!({"fit": "approximate-path", "spacing": 0.1}),
        ),
    );
    let spans: Vec<(f64, f64)> = p
        .paths
        .iter()
        .filter(|q| q.source_id == "s")
        .map(|q| q.at.s_span)
        .collect();
    let len = 2.0 * std::f64::consts::PI * r;
    // Equal pitch all the way round, the last gap equal to the others.
    let pitch = spans[1].0 - spans[0].0;
    let wrap_gap = spans[0].0 + len - spans.last().unwrap().1;
    let gap = spans[1].0 - spans[0].1;
    assert!(
        (wrap_gap - gap).abs() < 0.05 && (pitch * spans.len() as f64 - len).abs() < 0.1,
        "{wrap_gap} vs {gap}"
    );
    // With a corner tile, approximate-path rounds down instead of overhanging it.
    let tiles =
        json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": tile("oc", 10.0, "#d03a2f")});
    let n = polyline_node(&[(0.0, 0.0), (97.0, 0.0), (97.0, 100.0)], json!({}));
    let p = place(&n, &pattern(tiles, json!({"fit": "approximate-path"})));
    let incoming = p
        .paths
        .iter()
        .filter(|q| q.id.starts_with("s#0.0.") && q.source_id == "s");
    let max_x = incoming
        .flat_map(|q| polygons(&q.points).concat())
        .map(|q| q.x)
        .fold(0.0, f64::max);
    assert!(max_x <= 92.0 + 1e-6, "{max_x}");
}

#[test]
fn hairpins_and_steep_miters_stay_finite() {
    for extra in [
        json!({"outerCorner": tile("oc", 10.0, "#d03a2f"), "innerCorner": tile("ic", 10.0, "#2a9d4a")}),
        json!({"outerCorner": "between", "innerCorner": "sliced"}),
    ] {
        let mut tiles = json!({"side": tile("s", 10.0, "#3366cc")});
        for (k, v) in extra.as_object().unwrap() {
            tiles[k] = v.clone();
        }
        // 170° and 175° turns with short legs.
        let n = polyline_node(
            &[
                (0.0, 0.0),
                (60.0, 0.0),
                (0.0, 5.0),
                (60.0, 7.0),
                (58.0, 30.0),
            ],
            json!({}),
        );
        let p = place(&n, &pattern(tiles, json!({})));
        assert!(p.too_complex.is_none());
        for q in &p.paths {
            for pt in polygons(&q.points).concat() {
                assert!(
                    pt.x.is_finite() && pt.y.is_finite() && pt.x.abs() < 1e4 && pt.y.abs() < 1e4,
                    "{pt:?}"
                );
            }
        }
    }
}

#[test]
fn tiles_carry_their_arc_length() {
    let p = place(
        &line_node(straight(100.0), json!({})),
        &pattern(side_only(), json!({})),
    );
    for (i, q) in p.paths.iter().filter(|q| q.source_id == "s").enumerate() {
        let x = 10.0 * i as f64;
        assert!(
            near(q.at.s_span.0, x) && near(q.at.s_span.1, x + 10.0) && near(q.at.s, x + 5.0),
            "{i}: {:?}",
            q.at
        );
    }
}

#[test]
fn outer_tiles_at_right_turns_inner_at_left_turns() {
    // A clockwise square turns right at every corner (outer tiles); the same
    // square counter-clockwise turns left (inner tiles — none defined here, so
    // its corners stay bare, like Illustrator's Gold Leaf).
    let tiles =
        json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": tile("oc", 10.0, "#d03a2f")});
    let cw = polyline_node(
        &[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        json!({"closed": true}),
    );
    let ccw = polyline_node(
        &[(0.0, 0.0), (0.0, 100.0), (100.0, 100.0), (100.0, 0.0)],
        json!({"closed": true}),
    );
    assert_eq!(
        boxes(&place(&cw, &pattern(tiles.clone(), json!({}))), "oc").len(),
        4
    );
    let p = place(&ccw, &pattern(tiles, json!({})));
    assert!(boxes(&p, "oc").is_empty());
    assert_eq!(
        boxes(&p, "s").len(),
        4 * 9,
        "four legs cut back at the bare corners"
    );
}

#[test]
fn sliced_corners_never_stroke_the_miter_cut() {
    let side = json!({"nodes": [{"id": "s", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10,
        "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3366cc"}}], "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": 1}]}]});
    let tiles = json!({"side": side, "outerCorner": "sliced"});
    let p = place(
        &line_node(l_path(true), json!({})),
        &pattern(tiles, json!({})),
    );
    let strokes: Vec<_> = p
        .paths
        .iter()
        .filter(|q| q.id.ends_with(".stroke"))
        .collect();
    assert!(
        !strokes.is_empty(),
        "the corner tiles' strokes are split out"
    );
    for q in &strokes {
        assert!(!q.closed);
        assert_eq!(q.paint.get("fills"), None, "stroke-only part: no fill");
        // No stroke runs along the miter line x + y = 100 (through the corner).
        for poly in polygons(&q.points) {
            for w in poly.windows(2) {
                let on = |v: V2| ((v.x - 100.0) + v.y).abs() < 1e-3;
                assert!(!(on(w[0]) && on(w[1]) && w[0].dist(w[1]) > 1e-3), "{w:?}");
            }
        }
    }
}

#[test]
fn the_inner_tile_is_the_outer_placement_mirrored_across_the_incoming_leg() {
    // An asymmetric corner tile (marker top-right) as the OUTER tile on a right
    // turn, and as the INNER tile on the same L mirrored across the incoming leg
    // (y → −y, a left turn): the two placements are mirror images (Illustrator,
    // checked against stock Modern on the comparison harness).
    let corner = json!({"nodes": [
        {"id": "k", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#d03a2f"}}]},
        {"id": "k-m", "type": "rectangle", "x": 7, "y": 0, "width": 3, "height": 2, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}]},
    ]});
    let outer = place(
        &line_node(l_path(true), json!({})),
        &pattern(
            json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": corner.clone()}),
            json!({}),
        ),
    );
    let inner = place(
        &line_node(l_path(false), json!({})),
        &pattern(
            json!({"side": tile("s", 10.0, "#3366cc"), "innerCorner": corner}),
            json!({}),
        ),
    );
    let (o, i) = (boxes(&outer, "k-m")[0], boxes(&inner, "k-m")[0]);
    assert!(
        near(o.0.x, i.0.x) && near(o.1.x, i.1.x) && near(o.0.y, -i.1.y) && near(o.1.y, -i.0.y),
        "outer {o:?} vs inner {i:?}"
    );
}

// ── Trim / dash windows: tiles are clipped at window ends (a reveal); corners stay whole.

fn place_windows(node: &Value, brush: &Value, windows: &[super::art::ArcWindow]) -> Placement {
    let b = parse_pattern_brush(brush).expect("pattern brush");
    pattern_brush_for_node_with(
        node.as_object().unwrap(),
        &b,
        &default_geometry,
        &PatternOptions {
            tolerance: 0.1,
            windows: Some(windows),
        },
    )
}

fn win(subpath: usize, start: f64, end: f64) -> super::art::ArcWindow {
    super::art::ArcWindow {
        subpath,
        start,
        end,
        taper_extent: None,
    }
}

#[test]
fn windows_clip_side_tiles_without_relaying_them() {
    let node = line_node(straight(100.0), json!({}));
    let brush = pattern(side_only(), json!({}));
    let p = place_windows(&node, &brush, &[win(0, 25.0, 62.0)]);
    let mut b = boxes(&p, "s");
    b.sort_by(|x, y| x.0.x.total_cmp(&y.0.x));
    // Tiles 20–30 … 60–70 of the full layout, the end ones cut at 25 and 62.
    let xs: Vec<(f64, f64)> = b.iter().map(|(lo, hi)| (lo.x, hi.x)).collect();
    let want = [
        (25.0, 30.0),
        (30.0, 40.0),
        (40.0, 50.0),
        (50.0, 60.0),
        (60.0, 62.0),
    ];
    assert_eq!(xs.len(), want.len(), "{xs:?}");
    for ((a, b), (c, d)) in xs.iter().zip(want) {
        assert!((a - c).abs() < 1e-6 && (b - d).abs() < 1e-6, "{xs:?}");
    }
    assert!(p.paths.iter().all(|q| q.id.contains(".w0")));
    for q in &p.paths {
        assert!(
            q.at.s_span.0 >= 25.0 - 1e-9 && q.at.s_span.1 <= 62.0 + 1e-9,
            "{:?}",
            q.at
        );
    }
    // No windows → nothing; None → the full placement (unchanged ids).
    assert!(place_windows(&node, &brush, &[]).paths.is_empty());
    let full = place(&node, &brush);
    assert_eq!(boxes(&full, "s").len(), 10);
    assert!(full.paths.iter().all(|q| !q.id.contains(".w")));
}

#[test]
fn window_cuts_are_never_stroked() {
    let side = json!({"nodes": [{"id": "s", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10,
        "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3366cc"}}], "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": 1}]}]});
    let p = place_windows(
        &line_node(straight(100.0), json!({})),
        &pattern(json!({"side": side}), json!({})),
        &[win(0, 25.0, 62.0)],
    );
    let strokes: Vec<_> = p
        .paths
        .iter()
        .filter(|q| q.id.ends_with(".stroke"))
        .collect();
    assert_eq!(
        strokes.len(),
        2,
        "the two cut tiles split their strokes out"
    );
    for q in &strokes {
        assert_eq!(q.paint.get("fills"), None, "stroke-only part: no fill");
        for poly in polygons(&q.points) {
            for w in poly.windows(2) {
                for x in [25.0, 62.0] {
                    let on = |v: V2| (v.x - x).abs() < 1e-6;
                    assert!(!(on(w[0]) && on(w[1]) && w[0].dist(w[1]) > 1e-6), "{w:?}");
                }
            }
        }
    }
}

#[test]
fn windows_through_a_closed_seam_keep_corners_whole() {
    // 100 × 60 rectangle, 320 around, first corner at the seam (0, 0).
    let rect = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
        {"type": "line", "x": 100, "y": 60}, {"type": "line", "x": 0, "y": 60}]);
    let node = line_node(rect, json!({"closed": true}));
    let tiles =
        json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": tile("oc", 10.0, "#d03a2f")});
    let brush = pattern(tiles, json!({}));
    // [300, 340] wraps the seam: s 300–320 up the left side, 0–20 along the top.
    let p = place_windows(&node, &brush, &[win(0, 300.0, 340.0)]);
    let oc = boxes(&p, "oc");
    assert_eq!(oc.len(), 1, "only the seam corner");
    let (lo, hi) = oc[0];
    assert!(near((lo.x + hi.x) / 2.0, 0.0) && near((lo.y + hi.y) / 2.0, 0.0));
    let s = boxes(&p, "s");
    assert_eq!(s.len(), 4, "{s:?}");
    for (lo, hi) in &s {
        let top = hi.y <= 5.0 + 1e-6;
        if top {
            assert!(hi.x <= 20.0 + 1e-6, "{lo:?} {hi:?}");
        } else {
            // Up the left side (y falls as s grows): s 300 ↔ y 20.
            assert!(hi.x <= 5.0 + 1e-6 && hi.y <= 20.0 + 1e-6, "{lo:?} {hi:?}");
        }
    }
    // A window covering the ring from its start is the whole stroke.
    let ring = place_windows(&node, &brush, &[win(0, 0.0, 320.0)]);
    let full = place(&node, &brush);
    let ids = |p: &Placement| p.paths.iter().map(|q| q.id.clone()).collect::<Vec<_>>();
    assert_eq!(ids(&ring), ids(&full));
}

#[test]
fn a_tile_across_the_closed_seam_is_one_piece() {
    // Closed rectangle without corner tiles ("between" corners run the legs
    // through): the window [310, 330] is one run across the seam.
    let rect = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
        {"type": "line", "x": 100, "y": 60}, {"type": "line", "x": 0, "y": 60}]);
    let node = line_node(rect, json!({"closed": true}));
    let tiles = json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": "between"});
    let p = place_windows(&node, &pattern(tiles, json!({})), &[win(0, 310.0, 330.0)]);
    let ids: Vec<&str> = p.paths.iter().map(|q| q.id.as_str()).collect();
    let mut uniq = ids.clone();
    uniq.sort();
    uniq.dedup();
    assert_eq!(ids.len(), uniq.len(), "no tile emitted twice: {ids:?}");
    assert!(!p.paths.is_empty());
}

#[test]
fn json_windows_reach_the_pattern_placer() {
    let req = json!({"node": line_node(straight(100.0), json!({})), "brush": pattern(side_only(), json!({})),
        "windows": [{"subpath": 0, "start": 25, "end": 62}]});
    let v: Value = crate::value::from_str(&pattern_brush_json(&req.to_string())).unwrap();
    let nodes = v["nodes"].as_array().unwrap();
    assert!(!nodes.is_empty());
    for n in nodes {
        assert!(n["sSpan"][0].as_f64().unwrap() >= 25.0 - 1e-9);
        assert!(n["sSpan"][1].as_f64().unwrap() <= 62.0 + 1e-9);
    }
}

#[test]
fn wrapping_windows_give_unique_piece_ids() {
    // The review's repro: two long windows on a 320-around rectangle, each wrapping the seam.
    let rect = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
        {"type": "line", "x": 100, "y": 60}, {"type": "line", "x": 0, "y": 60}]);
    let node = line_node(rect, json!({"closed": true}));
    for tiles in [
        json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": tile("oc", 10.0, "#d03a2f")}),
        json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": "between"}),
        json!({"side": tile("s", 13.0, "#3366cc")}),
    ] {
        let brush = pattern(tiles, json!({}));
        for ws in [
            vec![win(0, 5.0, 322.0), win(0, 47.0, 364.0)],
            vec![win(0, 5.0, 322.0)],
            vec![win(0, 300.0, 350.0), win(0, 100.0, 110.0)],
        ] {
            let p = place_windows(&node, &brush, &ws);
            let mut ids: Vec<&str> = p.paths.iter().map(|q| q.id.as_str()).collect();
            let n = ids.len();
            ids.sort();
            ids.dedup();
            assert_eq!(ids.len(), n, "duplicate ids under {ws:?}");
        }
    }
}

#[test]
fn a_ring_window_starting_anywhere_is_the_whole_stroke() {
    let rect = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
        {"type": "line", "x": 100, "y": 60}, {"type": "line", "x": 0, "y": 60}]);
    let node = line_node(rect, json!({"closed": true}));
    let brush = pattern(
        json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": tile("oc", 10.0, "#d03a2f")}),
        json!({}),
    );
    let ids = |p: &Placement| p.paths.iter().map(|q| q.id.clone()).collect::<Vec<_>>();
    let full = ids(&place(&node, &brush));
    for ws in [
        vec![win(0, 37.0, 357.0)],
        vec![win(0, 10.0, 200.0), win(0, 190.0, 330.0)],
    ] {
        assert_eq!(ids(&place_windows(&node, &brush, &ws)), full, "{ws:?}");
    }
}

#[test]
fn window_pieces_count_toward_the_budget() {
    // Dashes much shorter than a tile split every tile into many pieces:
    // 800 tiles × ~20 pieces each goes past MAX_TILES and is refused.
    let node = line_node(straight(8000.0), json!({}));
    let brush = pattern(json!({"side": tile("s", 10.0, "#3366cc")}), json!({}));
    let dashes: Vec<_> = (0..16000)
        .map(|i| win(0, i as f64 * 0.5, i as f64 * 0.5 + 0.25))
        .collect();
    let p = place_windows(&node, &brush, &dashes);
    let why = p.too_complex.as_deref().unwrap_or("");
    assert!(why.contains("tiles"), "{why:?}, {} paths", p.paths.len());
}

/// Side tiles' boxes in path order.
fn sorted_boxes(p: &Placement) -> Vec<(V2, V2)> {
    let mut b = boxes(p, "s");
    b.sort_by(|x, y| x.0.x.total_cmp(&y.0.x));
    b
}

#[test]
fn pen_scale_sizes_each_tile_from_the_ink_at_its_position() {
    // Pressure 0 → 1 along a 300 px line; scale 0.5 → 2 with pressure.
    let node = line_node(
        straight(300.0),
        json!({"inkChannels": {"t": [0, 1], "pressure": [0, 1]}}),
    );
    let scale = json!({"scale": {"value": 1, "vary": "pressure", "min": 0.5, "max": 2}});
    for fit in ["stretch", "add-space", "approximate-path"] {
        let mut extra = scale.clone();
        extra["fit"] = json!(fit);
        let p = place(&node, &pattern(side_only(), extra));
        let b = sorted_boxes(&p);
        assert!(b.len() >= 5, "{fit}: {}", b.len());
        let heights: Vec<f64> = b.iter().map(|(lo, hi)| hi.y - lo.y).collect();
        for w in heights.windows(2) {
            assert!(w[1] > w[0], "{fit}: tiles grow with pressure: {heights:?}");
        }
        assert!(
            heights[0] < 8.0 && *heights.last().unwrap() > 16.0,
            "{fit}: {heights:?}"
        );
        // Each tile's thickness is the scale at its centre (0.5 + 1.5 × pressure).
        for (lo, hi) in &b {
            let cx = (lo.x + hi.x) / 2.0;
            let want = 10.0 * (0.5 + 1.5 * (cx / 300.0).clamp(0.0, 1.0));
            assert!(
                (hi.y - lo.y - want).abs() < 0.06 * want,
                "{fit}: at {cx}: {} vs {want}",
                hi.y - lo.y
            );
        }
        // Tiles never overlap, and stretch fills the line end to end.
        for w in b.windows(2) {
            assert!(w[1].0.x >= w[0].1.x - 1e-6, "{fit}: overlap");
        }
        if fit == "stretch" {
            assert!(near(b[0].0.x, 0.0) && near(b.last().unwrap().1.x, 300.0));
            for w in b.windows(2) {
                assert!(near(w[1].0.x, w[0].1.x), "stretch: contiguous");
            }
        }
    }
    // A fixed scale still lays equal tiles (one scale per stroke).
    let p = place(&node, &pattern(side_only(), json!({"scale": 1.5})));
    let b = sorted_boxes(&p);
    assert_eq!(b.len(), 20);
    assert!(b.iter().all(|(lo, hi)| near(hi.y - lo.y, 15.0)));
    // No pressure channel: the value.
    let bare = line_node(straight(300.0), json!({}));
    let p = place(&bare, &pattern(side_only(), scale.clone()));
    assert!(sorted_boxes(&p)
        .iter()
        .all(|(lo, hi)| near(hi.y - lo.y, 10.0)));
}

#[test]
fn pen_scale_is_floored_so_a_light_touch_stays_bounded() {
    let node = line_node(
        straight(200.0),
        json!({"inkChannels": {"t": [0, 1], "pressure": [0, 0]}}),
    );
    let p = place(
        &node,
        &pattern(
            side_only(),
            json!({"scale": {"value": 1, "vary": "pressure", "min": 0, "max": 2}}),
        ),
    );
    let b = sorted_boxes(&p);
    // 5 % of the largest scale (2): 1 px tiles.
    assert_eq!(b.len(), 200);
    assert!(near(b[0].1.y - b[0].0.y, 1.0));
    assert!(p.too_complex.is_none());
}

#[test]
fn pen_scale_sets_corner_depth_at_the_corner() {
    // Pressure 0 at the start, 1 at the corner (t = 0.5) and after: the corner tile is at scale 2.
    let tiles = json!({"side": tile("s", 10.0, "#3366cc"), "outerCorner": tile("oc", 10.0, "#d03a2f")});
    let node = line_node(
        l_path(true),
        json!({"inkChannels": {"t": [0, 0.5, 1], "pressure": [0, 1, 1]}}),
    );
    let p = place(
        &node,
        &pattern(
            tiles,
            json!({"scale": {"value": 1, "vary": "pressure", "min": 0.5, "max": 2}}),
        ),
    );
    let oc = boxes(&p, "oc");
    assert_eq!(oc.len(), 1);
    assert!(
        near(oc[0].0.x, 90.0)
            && near(oc[0].1.x, 110.0)
            && near(oc[0].0.y, -10.0)
            && near(oc[0].1.y, 10.0),
        "{oc:?}"
    );
}

#[test]
fn width_from_profile_shapes_the_tiles() {
    // Left 2 / right 1 at the start, 0 at the end.
    let profile = json!({"markers": [
        {"id": "a", "t": 0, "left": 2, "right": 1},
        {"id": "b", "t": 1, "left": 0, "right": 0}]});
    let node = line_node(
        straight(100.0),
        json!({"profile": profile.clone(), "brush": {"brushId": "p", "overrides": {"widthFrom": "profile"}}}),
    );
    let p = place(&node, &pattern(side_only(), json!({})));
    let b = sorted_boxes(&p);
    assert_eq!(b.len(), 10, "tile lengths are unchanged");
    // First tile: at x = 0 the left (−y) side is 2 × 5, the right 1 × 5.
    assert!(near(b[0].0.y, -10.0) && near(b[0].1.y, 5.0), "{:?}", b[0]);
    // It thins to nothing at the end.
    let last = b.last().unwrap();
    assert!(last.1.y - last.0.y < 3.1, "{last:?}");
    // Without the override the profile is ignored.
    let plain = line_node(straight(100.0), json!({"profile": profile}));
    let p = place(&plain, &pattern(side_only(), json!({})));
    assert!(sorted_boxes(&p)
        .iter()
        .all(|(lo, hi)| near(lo.y, -5.0) && near(hi.y, 5.0)));
    let o = parse_pattern_overrides(json!({"widthFrom": "profile"}).as_object());
    assert!(o.width_from_profile);
}
