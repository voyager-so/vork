use super::*;
use crate::live_effects::{contours_from_points, Bounds};
use crate::types::RawNode;
use crate::value::json;

fn bx(x: f64, y: f64, w: f64, h: f64) -> Bounds {
    Bounds { min: v2(x, y), max: v2(x + w, y + h) }
}

fn rect_shape(x: f64, y: f64, w: f64, h: f64) -> Shape {
    let s = |x, y| Seg::line(x, y);
    let (a, b, c, d) = (v2(x, y), v2(x + w, y), v2(x + w, y + h), v2(x, y + h));
    vec![Contour { start: a, segs: vec![s(a, b), s(b, c), s(c, d), s(d, a)], closed: true }]
}

/// The distance from `p` to the curve `c` (parameter 0..1): a dense scan, then a local refinement.
fn dist_to(c: &dyn Fn(f64) -> V2, p: V2) -> f64 {
    let n = 2000;
    let mut best = (f64::INFINITY, 0.0);
    for i in 0..=n {
        let t = i as f64 / n as f64;
        let d = c(t).dist(p);
        if d < best.0 {
            best = (d, t);
        }
    }
    let (mut lo, mut hi) = ((best.1 - 1.0 / n as f64).max(0.0), (best.1 + 1.0 / n as f64).min(1.0));
    for _ in 0..100 {
        let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
        if c(a).dist(p) < c(b).dist(p) {
            hi = b;
        } else {
            lo = a;
        }
    }
    c(0.5 * (lo + hi)).dist(p).min(best.0)
}

/// Deterministic sample points over a box (a Halton-like lattice).
fn samples(b: Bounds, n: usize) -> Vec<V2> {
    (0..n)
        .map(|k| {
            let u = (k as f64 * 0.618_033_988_75).fract();
            let v = (k as f64 * 0.754_877_666_2 + 0.1).fract();
            v2(b.min.x + u * b.width(), b.min.y + v * b.height())
        })
        .collect()
}

fn mesh_spec(rows: usize, cols: usize, points: Option<Value>) -> EnvelopeSpec {
    let mut v = json!({"type": "mesh", "rows": rows, "columns": cols});
    if let Some(p) = points {
        v["points"] = p;
    }
    parse_spec(&v).expect("valid mesh")
}

// --- The mesh mapping ---

#[test]
fn identity_mesh_is_exact() {
    let b = bx(13.0, -7.0, 240.0, 90.0);
    for (rows, cols) in [(1, 1), (4, 4), (3, 7), (50, 50)] {
        let map = envelope_map(&mesh_spec(rows, cols, None), b).unwrap();
        for p in samples(b, 500) {
            let q = map.map(p);
            assert!(q.dist(p) < 1e-9, "{rows}x{cols}: {p:?} → {q:?}");
        }
        // Grid lines and the box's corners too.
        for p in [b.min, b.max, v2(b.min.x, b.max.y), v2(b.max.x, b.min.y), b.center()] {
            assert!(map.map(p).dist(p) < 1e-9);
        }
    }
}

#[test]
fn identity_mesh_leaves_curves_where_they_are() {
    let b = bx(0.0, 0.0, 200.0, 100.0);
    let map = envelope_map(&mesh_spec(4, 4, None), b).unwrap();
    // An S curve across several cells.
    let shape: Shape = vec![Contour {
        start: v2(5.0, 50.0),
        segs: vec![Seg::curve(v2(5.0, 50.0), v2(60.0, -40.0), v2(140.0, 140.0), v2(195.0, 50.0))],
        closed: false,
    }];
    let mut budget = MAX_ENVELOPE_SEGMENTS;
    let out = map_shape_exact(&shape, &|p| map.map(p), envelope_tolerance(), map.feature, &mut budget);
    // Every output piece lies on the input curve (within the tolerance).
    let src = &shape[0].segs[0];
    for s in &out[0].segs {
        for k in 0..=8 {
            let p = s.eval(k as f64 / 8.0);
            let best = dist_to(&|t| src.eval(t), p);
            assert!(best < 1e-6, "{best}");
        }
    }
    // An affine map is fitted exactly by one cubic per pre-split piece: no extra splits.
    let n0 = (src.len() / (map.feature / 8.0)).ceil() as usize;
    assert_eq!(out[0].segs.len(), n0);
}

#[test]
fn a_moved_mesh_point_moves_the_map_there_and_nowhere_else_on_the_grid() {
    let b = bx(0.0, 0.0, 300.0, 300.0);
    let mut pts: Vec<Value> = Vec::new();
    for j in 0..=3 {
        for i in 0..=3 {
            let (mut x, mut y) = (i as f64 / 3.0, j as f64 / 3.0);
            if (i, j) == (1, 1) {
                x += 0.1;
                y -= 0.05;
            }
            pts.push(json!({"x": x, "y": y}));
        }
    }
    let map = envelope_map(&mesh_spec(3, 3, Some(Value::Array(pts))), b).unwrap();
    // The moved point lands where it was put.
    let q = map.map(v2(100.0, 100.0));
    assert!(q.dist(v2(130.0, 85.0)) < 1e-9, "{q:?}");
    // The other grid points stay.
    for j in 0..=3 {
        for i in 0..=3 {
            if (i, j) == (1, 1) {
                continue;
            }
            let p = v2(i as f64 * 100.0, j as f64 * 100.0);
            assert!(map.map(p).dist(p) < 1e-9);
        }
    }
    // Cells away from the point don't move.
    let far = v2(250.0, 250.0);
    assert!(map.map(far).dist(far) < 1e-9);
    // The map is continuous across cell edges.
    for t in [0.0, 37.0, 99.999, 100.0, 100.001, 163.0] {
        let a = map.map(v2(100.0 - 1e-7, t));
        let c = map.map(v2(100.0 + 1e-7, t));
        assert!(a.dist(c) < 1e-5);
    }
}

#[test]
fn mesh_handles_bend_the_cell_edges_as_bezier_curves() {
    let b = bx(0.0, 0.0, 100.0, 100.0);
    // One cell; the top-left point's right handle pulled up: the top edge is that cubic.
    let pts = json!([
        {"x": 0, "y": 0, "right": [0.3, -0.3]},
        {"x": 1, "y": 0},
        {"x": 0, "y": 1},
        {"x": 1, "y": 1}
    ]);
    let map = envelope_map(&mesh_spec(1, 1, Some(pts)), b).unwrap();
    let edge = Cubic([v2(0.0, 0.0), v2(30.0, -30.0), v2(200.0 / 3.0, 0.0), v2(100.0, 0.0)]);
    for k in 0..=10 {
        let u = k as f64 / 10.0;
        assert!(map.map(v2(u * 100.0, 0.0)).dist(edge.eval(u)) < 1e-9);
    }
    // The bottom edge is untouched.
    for k in 0..=10 {
        let p = v2(k as f64 * 10.0, 100.0);
        assert!(map.map(p).dist(p) < 1e-9);
    }
}

#[test]
fn mesh_specs_are_validated() {
    assert!(parse_spec(&json!({"type": "mesh", "rows": 0, "columns": 3})).is_none());
    assert!(parse_spec(&json!({"type": "mesh", "rows": 2, "columns": 2, "points": [{"x": 0, "y": 0}]})).is_none());
    assert!(parse_spec(&json!({"type": "mesh", "rows": 51, "columns": 1})).is_none());
    assert!(parse_spec(&json!({"type": "warp", "style": "nope", "bend": 10})).is_none());
    assert!(parse_spec(&json!({"type": "warp", "style": "arc", "bend": 10})).is_some());
    assert!(parse_spec(&json!({"type": "sparkle"})).is_none());
}

// --- Exact subdivision ---

#[test]
fn wave_bends_a_plain_four_point_rectangle() {
    let b = bx(0.0, 0.0, 200.0, 100.0);
    let spec = parse_spec(&json!({"type": "warp", "style": "wave", "bend": 50})).unwrap();
    let map = envelope_map(&spec, b).unwrap();
    let mut budget = MAX_ENVELOPE_SEGMENTS;
    let tol = envelope_tolerance();
    let out = map_shape_exact(&rect_shape(0.0, 0.0, 200.0, 100.0), &|p| map.map(p), tol, map.feature, &mut budget);
    // The top edge ripples: its middle quarter points move by a quarter of the height, opposite ways.
    let top = &out[0];
    let mut ys = Vec::new();
    for s in top.segs.iter() {
        for k in 0..=16 {
            let p = s.eval(k as f64 / 16.0);
            if p.y.abs() < 60.0 && p.x > 1.0 && p.x < 199.0 {
                ys.push(p.y);
            }
        }
    }
    let lo = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(lo < -20.0 && hi > 20.0, "top edge ranges {lo}..{hi}");
    // And the output follows the map within the tolerance everywhere along the edge.
    let edge = Seg::line(v2(0.0, 0.0), v2(200.0, 0.0));
    let mut worst: f64 = 0.0;
    for s in top.segs.iter().take_while(|s| s.end().x < 199.999 && s.end().y.abs() < 60.0) {
        for k in 0..=4 {
            let p = s.eval(k as f64 / 4.0);
            // The true mapped point with the same x (Wave moves points vertically only).
            let truth = map.map(v2(p.x, 0.0));
            worst = worst.max((truth.y - p.y).abs());
        }
    }
    assert!(worst < 10.0 * tol, "{worst}");
    let _ = edge;
}

#[test]
fn subdivision_refines_to_the_tolerance() {
    let b = bx(0.0, 0.0, 100.0, 100.0);
    let spec = parse_spec(&json!({"type": "warp", "style": "twist", "bend": 80})).unwrap();
    let map = envelope_map(&spec, b).unwrap();
    let seg = Seg::line(v2(0.0, 50.0), v2(100.0, 50.0));
    let shape = vec![Contour { start: seg.start(), segs: vec![seg], closed: false }];
    for tol in [0.5, 0.05, 0.001] {
        let mut budget = MAX_ENVELOPE_SEGMENTS;
        let out = map_shape_exact(&shape, &|p| map.map(p), tol, map.feature, &mut budget);
        // Each output piece matches the map at its parameter midpoints.
        let n = out[0].segs.len() as f64;
        let mut worst: f64 = 0.0;
        for (k, s) in out[0].segs.iter().enumerate() {
            let _ = k;
            // Check the piece's midpoint against the nearest true mapped point.
            let p = s.eval(0.5);
            let best = dist_to(&|t| map.map(seg.eval(t)), p);
            worst = worst.max(best);
        }
        assert!(worst <= tol, "tol {tol}: {worst} with {n} pieces");
    }
    // A finer tolerance takes more pieces.
    let count = |tol| {
        let mut budget = MAX_ENVELOPE_SEGMENTS;
        map_shape_exact(&shape, &|p| map.map(p), tol, map.feature, &mut budget)[0].segs.len()
    };
    assert!(count(0.001) > count(0.5));
}

#[test]
fn the_tolerance_follows_the_device_scale() {
    assert!((tolerance_for_scale(1.0) - 0.2).abs() < 1e-12);
    assert!((tolerance_for_scale(4.0) - 0.05).abs() < 1e-12);
    assert!((envelope_tolerance() - 0.2 / 512.0).abs() < 1e-15);
}

#[test]
fn warp_envelope_matches_the_warp_effect_map_except_wave() {
    let b = bx(10.0, 20.0, 200.0, 100.0);
    for style in crate::live_effects::WARP_STYLES {
        let e: WarpEffect = crate::value::from_value(json!({"style": style, "bend": 40, "horizontal": 10})).unwrap();
        let effect = warp_map(&e, b, false).unwrap();
        let spec = parse_spec(&json!({"type": "warp", "style": style, "bend": 40, "horizontal": 10})).unwrap();
        let env = envelope_map(&spec, b).unwrap();
        let same = samples(b, 50).into_iter().all(|p| env.map(p).dist(effect(p)) < 1e-12);
        assert_eq!(same, *style != "wave", "{style}");
    }
}

// --- Top object ---

#[test]
fn a_rectangular_top_object_is_the_identity() {
    let b = bx(0.0, 0.0, 400.0, 200.0);
    let pts = json!([
        {"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 1, "y": 0},
        {"type": "line", "x": 1, "y": 1}, {"type": "line", "x": 0, "y": 1}
    ]);
    let spec = parse_spec(&json!({"type": "top-object", "points": pts, "closed": true})).unwrap();
    let map = envelope_map(&spec, b).unwrap();
    for p in samples(b, 300) {
        assert!(map.map(p).dist(p) < 1e-6, "{p:?} → {:?}", map.map(p));
    }
    // The same ring drawn the other way round gives the same map.
    let rev = json!([
        {"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 0, "y": 1},
        {"type": "line", "x": 1, "y": 1}, {"type": "line", "x": 1, "y": 0}
    ]);
    let spec = parse_spec(&json!({"type": "top-object", "points": rev})).unwrap();
    let map = envelope_map(&spec, b).unwrap();
    for p in samples(b, 100) {
        assert!(map.map(p).dist(p) < 1e-6);
    }
}

#[test]
fn a_top_object_maps_the_box_edges_onto_its_outline() {
    // A circle (four arcs) as the top object.
    let k = 0.552_284_749_8 * 0.5;
    let c = |x: f64, y: f64| (x, y);
    let pts: Vec<BlendPathPoint> = vec![
        BlendPathPoint::Move { x: 0.5, y: 0.0 },
        BlendPathPoint::Curve { x: 1.0, y: 0.5, cp1: c(0.5 + k, 0.0), cp2: c(1.0, 0.5 - k) },
        BlendPathPoint::Curve { x: 0.5, y: 1.0, cp1: c(1.0, 0.5 + k), cp2: c(0.5 + k, 1.0) },
        BlendPathPoint::Curve { x: 0.0, y: 0.5, cp1: c(0.5 - k, 1.0), cp2: c(0.0, 0.5 + k) },
        BlendPathPoint::Curve { x: 0.5, y: 0.0, cp1: c(0.0, 0.5 - k), cp2: c(0.5 - k, 0.0) },
    ];
    let json_pts: Vec<Value> = pts.iter().map(BlendPathPoint::to_json).collect();
    let spec = parse_spec(&json!({"type": "top-object", "points": json_pts})).unwrap();
    let b = bx(0.0, 0.0, 100.0, 100.0);
    let map = envelope_map(&spec, b).unwrap();
    let ring = contours_from_points(&pts, true);
    let on_circle = |p: V2| ((p - v2(50.0, 50.0)).len() - 50.0).abs();
    let _ = ring;
    // Every point of the box's outline lands on the circle; the centre stays inside.
    for k in 0..=40 {
        let t = k as f64 / 40.0 * 100.0;
        for p in [v2(t, 0.0), v2(100.0, t), v2(t, 100.0), v2(0.0, t)] {
            assert!(on_circle(map.map(p)) < 0.2, "{p:?} → {:?}", map.map(p));
        }
    }
    assert!(map.map(v2(50.0, 50.0)).dist(v2(50.0, 50.0)) < 1.0);
}

// --- Image cells ---

#[test]
fn square_to_quad_hits_the_corners() {
    let q = [v2(10.0, 5.0), v2(90.0, 12.0), v2(80.0, 70.0), v2(3.0, 60.0)];
    let m = unit_square_to_quad(q).unwrap();
    for (uv, p) in [((0.0, 0.0), q[0]), ((1.0, 0.0), q[1]), ((1.0, 1.0), q[2]), ((0.0, 1.0), q[3])] {
        assert!(apply_projective(&m, uv.0, uv.1).dist(p) < 1e-9);
    }
}

#[test]
fn image_cells_follow_the_map_within_the_image_tolerance() {
    let b = bx(0.0, 0.0, 200.0, 100.0);
    let spec = parse_spec(&json!({"type": "warp", "style": "flag", "bend": 40})).unwrap();
    let map = envelope_map(&spec, b).unwrap();
    let g = |p: V2| map.map(p);
    let mut budget = MAX_IMAGE_CELLS;
    let cells = image_cells([0.0, 0.0, 200.0, 100.0], &g, IMAGE_TOLERANCE_PX, &mut budget);
    assert!(cells.len() > 4 && cells.len() < MAX_IMAGE_CELLS, "{}", cells.len());
    // The cells tile the source box exactly.
    let area: f64 = cells.iter().map(|c| c.src[2] * c.src[3]).sum();
    assert!((area - 200.0 * 100.0).abs() < 1e-6);
    for c in &cells {
        let m = unit_square_to_quad(c.quad).unwrap();
        for (u, v) in [(0.5, 0.5), (0.3, 0.7), (0.9, 0.1)] {
            let truth = g(v2(c.src[0] + u * c.src[2], c.src[1] + v * c.src[3]));
            assert!(apply_projective(&m, u, v).dist(truth) < IMAGE_TOLERANCE_PX * 2.0);
        }
    }
    // The identity needs no split.
    let id = envelope_map(&mesh_spec(2, 2, None), b).unwrap();
    let mut budget = MAX_IMAGE_CELLS;
    assert_eq!(image_cells([0.0, 0.0, 200.0, 100.0], &|p| id.map(p), IMAGE_TOLERANCE_PX, &mut budget).len(), 2);
}

// --- In a scene ---

fn solid(c: &str) -> Value {
    json!([{"id": "f", "paint": {"type": "solid", "color": c}}])
}

fn doc(envelope: Value) -> Value {
    json!({
        "nodes": [
            {"id": "env", "type": "group", "x": 100, "y": 50, "rotation": 0, "scaleX": 1, "scaleY": 1,
             "children": ["a", "g"], "envelope": envelope},
            {"id": "a", "type": "rectangle", "x": 0, "y": 0, "width": 200, "height": 100, "fills": solid("#f00"),
             "strokes": [{"id": "s", "paint": {"type": "solid", "color": "#000"}, "width": 4}]},
            {"id": "g", "type": "group", "x": 20, "y": 20, "children": ["m", "b"], "clipMaskNodeId": "m", "opacity": 0.5},
            {"id": "m", "type": "ellipse", "x": 0, "y": 0, "width": 60, "height": 60, "fills": solid("#000")},
            {"id": "b", "type": "rectangle", "x": 10, "y": 10, "width": 80, "height": 40, "fills": solid("#00f")},
            {"id": "plain", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10, "fills": solid("#0f0")}
        ],
        "pages": [{"id": "p1", "width": 500, "height": 400, "nodeIds": ["env", "plain"]}]
    })
}

fn wave() -> Value {
    json!({"type": "warp", "style": "wave", "bend": 50})
}

#[test]
fn an_envelope_draws_its_bent_content_in_place_of_its_children() {
    let scene = Scene::load(&doc(wave())).unwrap();
    let z = scene.flattened_z_order_for_page("p1");
    // The envelope, then its transients; never its own children.
    assert!(z.contains(&"env".to_string()));
    for child in ["a", "g", "m", "b"] {
        assert!(!z.contains(&child.to_string()), "{child} drawn: {z:?}");
    }
    assert!(z.iter().any(|id| id == "env~a"), "{z:?}");
    assert!(z.contains(&"plain".to_string()));
    let a = scene.effective_node("env~a").unwrap();
    assert_eq!(a.node_type(), "path");
    assert_eq!(a.str(ENVELOPE_OF_KEY), Some("env"));
    assert_eq!(a.str(ENVELOPE_SOURCE_KEY), Some("a"));
    // Fill, then the stroke's outline filled with its paint.
    let counts = a.array(EFFECT_COPIES_KEY).unwrap();
    assert_eq!(counts.len(), 2);
    assert_eq!(a.array(EFFECT_COPY_FILLS_KEY).unwrap()[1][0]["paint"]["color"], json!("#000"));
    // The transient sits in the envelope's space: its canvas transform is the envelope's.
    let (aff, _) = scene.accumulated_affine("env~a").unwrap();
    assert_eq!((aff.e, aff.f), (100.0, 50.0));
    // The masked group keeps its mask (the warped ellipse) and its opacity.
    let g = scene.effective_node("env~g").unwrap();
    assert_eq!(g.str("clipMaskNodeId"), Some("env~m"));
    assert_eq!(g.f64("opacity"), Some(0.5));
    assert!(scene.envelope_mask_sources.contains(&"env~m".to_string()));
    assert!(crate::display_list::clip_mask_source_ids(&scene).contains("env~m"));
    assert!((scene.ancestor_opacity_product("env~b") - 0.5).abs() < 1e-12);
    // The wave moved the rectangle's top edge.
    let pts = a.points().unwrap();
    let ys: Vec<f64> = pts.json().iter().filter_map(|p| p["y"].as_f64()).collect();
    assert!(ys.iter().any(|y| *y < -10.0), "{ys:?}");
}

#[test]
fn the_envelope_outline_is_its_clip_shape() {
    let scene = Scene::load(&doc(wave())).unwrap();
    let o = scene.envelope_mask_record("env").unwrap();
    assert_eq!(o.node_type(), "path");
    assert!(o.points().unwrap().len() > 8);
    assert_eq!(o.f64("x"), Some(100.0));
}

#[test]
fn a_document_without_envelopes_is_untouched() {
    let mut d = doc(wave());
    d["nodes"][0].as_object_mut().unwrap().remove("envelope");
    let scene = Scene::load(&d).unwrap();
    assert!(scene.envelope_transients.is_empty());
    assert!(scene.transient_by_id.keys().all(|id| id.starts_with("__page_bg")));
    let z = scene.flattened_z_order_for_page("p1");
    assert!(z.contains(&"a".to_string()) && z.contains(&"b".to_string()));
}

#[test]
fn edits_match_a_cold_load() {
    fn same(a: &Scene, b: &Scene) {
        assert_eq!(a.transient_by_id, b.transient_by_id);
        assert_eq!(a.transient_by_parent, b.transient_by_parent);
        assert_eq!(a.transient_parent_of, b.transient_parent_of);
        assert_eq!(a.transient_page_of, b.transient_page_of);
        assert_eq!(a.resolved_by_id, b.resolved_by_id);
        assert_eq!(a.envelope_transients, b.envelope_transients);
        assert_eq!(a.envelope_outline_by_id, b.envelope_outline_by_id);
        assert_eq!(a.envelope_mask_sources, b.envelope_mask_sources);
    }
    let mut d = doc(wave());
    let mut scene = Scene::load(&d).unwrap();
    // A child's property edit re-bends the envelope.
    d["nodes"][1]["width"] = json!(260);
    scene.apply_node_edits(vec![RawNode::from_value(d["nodes"][1].clone()).unwrap()], &[], &[], None).unwrap();
    same(&scene, &Scene::load(&d).unwrap());
    // The envelope's own spec changes.
    d["nodes"][0]["envelope"] = json!({"type": "mesh", "rows": 2, "columns": 3});
    scene.apply_node_edits(vec![RawNode::from_value(d["nodes"][0].clone()).unwrap()], &[], &[], None).unwrap();
    same(&scene, &Scene::load(&d).unwrap());
    // Released: the envelope field goes, the children draw again.
    d["nodes"][0].as_object_mut().unwrap().remove("envelope");
    scene.apply_node_edits(vec![RawNode::from_value(d["nodes"][0].clone()).unwrap()], &[], &[], None).unwrap();
    same(&scene, &Scene::load(&d).unwrap());
    assert!(scene.envelope_transients.is_empty());
    // Made again on a plain group, then the whole envelope deleted.
    d["nodes"][0]["envelope"] = wave();
    scene.apply_node_edits(vec![RawNode::from_value(d["nodes"][0].clone()).unwrap()], &[], &[], None).unwrap();
    same(&scene, &Scene::load(&d).unwrap());
    let ids: Vec<String> = ["env", "a", "g", "m", "b"].iter().map(|s| s.to_string()).collect();
    d["nodes"].as_array_mut().unwrap().retain(|n| !ids.contains(&n["id"].as_str().unwrap().to_string()));
    d["pages"][0]["nodeIds"] = json!(["plain"]);
    scene.apply_node_edits(vec![], &ids, &[("p1".into(), vec!["plain".into()])], None).unwrap();
    same(&scene, &Scene::load(&d).unwrap());
    assert!(scene.transient_by_id.keys().all(|id| id.starts_with("__page_bg")));
}

#[test]
fn nested_envelopes_bend_twice() {
    let mut d = doc(wave());
    d["nodes"].as_array_mut().unwrap().push(json!(
        {"id": "outer", "type": "group", "x": 0, "y": 0, "children": ["env"],
         "envelope": {"type": "warp", "style": "arc", "bend": 30}}
    ));
    d["pages"][0]["nodeIds"] = json!(["outer", "plain"]);
    let scene = Scene::load(&d).unwrap();
    let z = scene.flattened_z_order_for_page("p1");
    assert!(z.iter().any(|id| id == "outer~env~a"), "{z:?}");
    assert!(!z.iter().any(|id| id == "env~a"));
    let inner = scene.effective_node("env~a").unwrap().points().unwrap().json().into_owned();
    let outer = scene.effective_node("outer~env~a").unwrap().points().unwrap().json().into_owned();
    assert_ne!(inner, outer);
    assert_eq!(scene.effective_node("outer~env~a").unwrap().str(ENVELOPE_SOURCE_KEY), Some("a"));
}

#[test]
fn image_fills_become_corner_pinned_cells() {
    let d = json!({
        "nodes": [
            {"id": "env", "type": "group", "x": 0, "y": 0, "children": ["img"],
             "envelope": {"type": "warp", "style": "flag", "bend": 30}},
            {"id": "img", "type": "rectangle", "x": 10, "y": 10, "width": 160, "height": 90,
             "fills": [{"id": "i", "opacity": 0.9, "paint": {"type": "image", "src": "data:image/png;base64,AAAA",
                "placement": {"fit": "cover", "naturalSize": {"width": 320, "height": 180}}}}]}
        ],
        "pages": [{"id": "p1", "width": 300, "height": 200, "nodeIds": ["env"]}]
    });
    let scene = Scene::load(&d).unwrap();
    let group = scene.effective_node("env~img").unwrap();
    assert_eq!(group.node_type(), "group");
    let cells = scene.transient_by_parent.get("env~img~img0").unwrap();
    assert!(cells.len() > 4, "{}", cells.len());
    let c = scene.effective_node(&cells[0]).unwrap();
    assert_eq!(c.node_type(), "rectangle");
    assert_eq!(c.json_field("distort").unwrap()["type"], json!("matrix"));
    let placement = &c.array("fills").unwrap()[0]["paint"]["placement"];
    assert_eq!(placement["fit"], json!("crop"));
    // Cover of 320×180 in 160×90: scale 0.5 at the box's origin; the first cell starts at the box's corner.
    assert!((placement["scale"]["x"].as_f64().unwrap() - 0.5).abs() < 1e-9);
    assert_eq!(c.array("fills").unwrap()[0]["opacity"], json!(0.9));
    // Cells are drawn, in the envelope's space.
    let z = scene.flattened_z_order_for_page("p1");
    assert!(z.contains(&cells[0]));
    let (aff, _) = scene.accumulated_affine(&cells[0]).unwrap();
    assert_eq!((aff.a, aff.d, aff.e, aff.f), (1.0, 1.0, 0.0, 0.0));
}

#[test]
fn envelope_geometry_json_answers_the_tree() {
    let d = doc(wave());
    let nodes: Vec<Value> = d["nodes"].as_array().unwrap().iter().filter(|n| n["id"] != "plain").cloned().collect();
    let out: Value = crate::value::from_str(&envelope_geometry_json(&json!({"envelopeId": "env", "nodes": nodes}).to_string())).unwrap();
    assert_eq!(out["status"], json!("ok"));
    let items = out["items"].as_array().unwrap();
    assert_eq!(items[0]["node"]["id"], json!("env~a"));
    assert_eq!(items[1]["node"]["type"], json!("group"));
    assert_eq!(items[1]["children"].as_array().unwrap().len(), 2);
    // Local to the envelope: the envelope's own x / y are not in the geometry.
    assert!(out["contentBounds"]["minX"].as_f64().unwrap() < 0.0);
    assert!(out["bounds"]["maxY"].as_f64().unwrap() > 100.0);
    let none: Value = crate::value::from_str(&envelope_geometry_json(&json!({"envelopeId": "a", "nodes": nodes}).to_string())).unwrap();
    assert_eq!(none["status"], json!("none"));
}


#[test]
fn deform_bulges_and_twists_about_its_centre_and_never_folds() {
    let spec = parse_spec(&json!({"type":"deform","kind":"bulge","center":{"x":50,"y":50},"radius":40,"amount":0.8})).unwrap();
    let EnvelopeSpec::Deform(d) = spec else { panic!("a deform") };
    // Outside the radius, and the centre itself, stay.
    assert_eq!(d.map(v2(95.0, 50.0)), v2(95.0, 50.0));
    assert_eq!(d.map(v2(50.0, 50.0)), v2(50.0, 50.0));
    // A point 10 px out goes to 10 · (1 + 0.8 · (1 − 1/16)²).
    let k = 1.0 - (10.0f64 / 40.0).powi(2);
    let got = d.map(v2(60.0, 50.0));
    assert!((got.x - (50.0 + 10.0 * (1.0 + 0.8 * k * k))).abs() < 1e-9 && (got.y - 50.0).abs() < 1e-12, "{got:?}");
    // Monotonic along a ray at both ends of the clamped range (a larger amount is clamped to 1.2).
    for a in [1.2, 5.0, -0.95] {
        let d = Deform { amount: a, ..d };
        let mut last = -1.0;
        for i in 0..=400 {
            let r = 40.0 * i as f64 / 400.0;
            let x = d.map(v2(50.0 + r, 50.0)).x - 50.0;
            assert!(x >= last, "amount {a}: folds at r {r}");
            last = x;
        }
    }
    // Twist: 90° at the centre, fading as (1 − r/R)²; distances to the centre are kept.
    let t = Deform { kind: DeformKind::Twist, amount: 90.0, ..d };
    let p = t.map(v2(70.0, 50.0));
    let th = (90.0f64 * 0.25).to_radians();
    assert!((p.x - (50.0 + 20.0 * th.cos())).abs() < 1e-9 && (p.y - (50.0 + 20.0 * th.sin())).abs() < 1e-9, "{p:?}");
    // Refused: an unknown kind, no centre, a radius of 0.
    for bad in [
        json!({"type":"deform","kind":"spin","center":{"x":0,"y":0},"radius":10}),
        json!({"type":"deform","kind":"bulge","radius":10}),
        json!({"type":"deform","kind":"bulge","center":{"x":0,"y":0},"radius":0}),
    ] {
        assert!(parse_spec(&bad).is_none(), "{bad}");
    }
}
