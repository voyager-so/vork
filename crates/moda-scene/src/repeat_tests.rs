use super::*;
use crate::types::RawNode;
use crate::value::json;

fn solid(c: &str) -> Value {
    json!([{"id": "f", "paint": {"type": "solid", "color": c}}])
}

fn doc(repeat: Value) -> Value {
    json!({
        "nodes": [
            {"id": "rep", "type": "group", "x": 100, "y": 50, "rotation": 0, "scaleX": 1, "scaleY": 1,
             "children": ["a", "g"], "repeat": repeat},
            {"id": "a", "type": "rectangle", "x": 0, "y": 0, "width": 40, "height": 20, "fills": solid("#f00")},
            {"id": "g", "type": "group", "x": 10, "y": 5, "children": ["m", "b"], "clipMaskNodeId": "m", "opacity": 0.5},
            {"id": "m", "type": "ellipse", "x": 0, "y": 0, "width": 10, "height": 10, "fills": solid("#000")},
            {"id": "b", "type": "rectangle", "x": 2, "y": 2, "width": 8, "height": 8, "fills": solid("#00f")},
            {"id": "plain", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10, "fills": solid("#0f0")}
        ],
        "pages": [{"id": "p1", "width": 500, "height": 400, "nodeIds": ["rep", "plain"]}]
    })
}

fn close(a: (f64, f64), b: (f64, f64)) -> bool {
    (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9
}

#[test]
fn specs_parse_and_refuse() {
    assert!(parse_spec(&json!({"type": "radial", "count": 12, "radius": 100})).is_some());
    assert!(parse_spec(&json!({"type": "radial", "count": 0, "radius": 100})).is_none());
    assert!(parse_spec(&json!({"type": "radial", "count": 2.5, "radius": 100})).is_none());
    assert!(parse_spec(&json!({"type": "radial", "count": 12, "radius": -1})).is_none());
    assert!(parse_spec(&json!({"type": "radial", "count": 361, "radius": 1})).is_none());
    assert!(parse_spec(&json!({"type": "grid", "rows": 2, "columns": 3})).is_some());
    assert!(parse_spec(&json!({"type": "grid", "rows": 2})).is_none());
    assert!(parse_spec(&json!({"type": "mirror", "axis": {"x": 1, "y": 2, "angle": 90}})).is_some());
    assert!(parse_spec(&json!({"type": "mirror"})).is_none());
    assert!(parse_spec(&json!({"type": "spiral"})).is_none());
    let r = parse_spec(&json!({"type": "radial", "count": 6, "radius": 1, "hidden": [3, 1, 1, -2, 1.5]})).unwrap();
    assert!(matches!(r, RepeatSpec::Radial { ref hidden, start_angle, .. } if *hidden == vec![1, 3] && start_angle == -90.0));
}

#[test]
fn radial_copies_turn_about_a_centre_below_the_source() {
    let spec = parse_spec(&json!({"type": "radial", "count": 4, "radius": 100})).unwrap();
    let b = [0.0, 0.0, 20.0, 20.0];
    // The source's centre is (10, 10); at −90° the centre is 100 below it.
    assert!(close(radial_center(100.0, -90.0, b), (10.0, 110.0)));
    let t = instance_transforms(&spec, Some(b));
    assert_eq!(t.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    // Copy 1 is a quarter turn clockwise: the source's centre lands to the right of the circle's centre.
    assert!(close(transform::apply_point((10.0, 10.0), t[1].1), (110.0, 110.0)));
    assert!(close(transform::apply_point((10.0, 10.0), t[2].1), (10.0, 210.0)));
    assert!(close(transform::apply_point((10.0, 10.0), t[3].1), (-90.0, 110.0)));
    assert_eq!(t[0].1, transform::identity());
    // Reverse overlap stacks them the other way; hidden copies go.
    let spec = parse_spec(&json!({"type": "radial", "count": 4, "radius": 100, "reverseOverlap": true, "hidden": [2]})).unwrap();
    let order: Vec<usize> = instance_transforms(&spec, Some(b)).iter().map(|(i, _)| *i).collect();
    assert_eq!(order, vec![3, 1, 0]);
}

#[test]
fn grid_copies_tile_with_spacing_bricks_and_flips() {
    let b = [0.0, 0.0, 20.0, 10.0];
    let spec = parse_spec(&json!({"type": "grid", "rows": 2, "columns": 3, "spacingX": 5, "spacingY": 2})).unwrap();
    let t = instance_transforms(&spec, Some(b));
    assert_eq!(t.len(), 6);
    assert!(close(transform::apply_point((0.0, 0.0), t[2].1), (50.0, 0.0)));
    assert!(close(transform::apply_point((0.0, 0.0), t[4].1), (25.0, 12.0)));
    // Brick by row: odd rows shift half a tile.
    let spec = parse_spec(&json!({"type": "grid", "rows": 2, "columns": 2, "pattern": "brick-row"})).unwrap();
    let t = instance_transforms(&spec, Some(b));
    assert!(close(transform::apply_point((0.0, 0.0), t[2].1), (10.0, 10.0)));
    // Brick by column with a quarter offset.
    let spec = parse_spec(&json!({"type": "grid", "rows": 1, "columns": 2, "pattern": "brick-column", "offset": 0.25})).unwrap();
    let t = instance_transforms(&spec, Some(b));
    assert!(close(transform::apply_point((0.0, 0.0), t[1].1), (20.0, 2.5)));
    // Flip every other row horizontally: the copy is mirrored about its tile's centre.
    let spec = parse_spec(&json!({"type": "grid", "rows": 2, "columns": 1, "flipRow": "horizontal"})).unwrap();
    let t = instance_transforms(&spec, Some(b));
    assert!(close(transform::apply_point((0.0, 0.0), t[1].1), (20.0, 10.0)));
    assert!(close(transform::apply_point((20.0, 0.0), t[1].1), (0.0, 10.0)));
    let (_, _, _, sx, sy) = trs_of(&t[1].1);
    assert!(sx * sy < 0.0);
}

#[test]
fn mirror_reflects_across_its_axis() {
    let spec = parse_spec(&json!({"type": "mirror", "axis": {"x": 50, "y": 0, "angle": 90}})).unwrap();
    let t = instance_transforms(&spec, None);
    assert_eq!(t.len(), 2);
    assert!(close(transform::apply_point((40.0, 7.0), t[1].1), (60.0, 7.0)));
    // A diagonal axis through the origin swaps x and y.
    let r = reflection(0.0, 0.0, 45.0);
    assert!(close(transform::apply_point((3.0, 1.0), r), (1.0, 3.0)));
}

#[test]
fn trs_rebuilds_every_copy_transform() {
    for a in [
        rotation_about(37.0, 5.0, -3.0),
        reflection(4.0, 9.0, 30.0),
        transform::compose(transform::translate(3.0, 4.0), scale_about(-1.0, -1.0, 2.0, 2.0)),
        transform::compose(transform::translate(3.0, 4.0), scale_about(1.0, -1.0, 2.0, 2.0)),
    ] {
        let (x, y, r, sx, sy) = trs_of(&a);
        let b = transform::from_trs(crate::types::Transform2D { x, y, rotation: r, scale_x: sx, scale_y: sy });
        for p in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (7.0, -2.0)] {
            assert!(close(transform::apply_point(p, a), transform::apply_point(p, b)), "{a:?} vs {b:?}");
        }
    }
}

#[test]
fn a_repeat_draws_its_copies_in_place_of_its_children() {
    let scene = Scene::load(&doc(json!({"type": "grid", "rows": 1, "columns": 3, "spacingX": 10}))).unwrap();
    let z = scene.flattened_z_order_for_page("p1");
    for child in ["a", "g", "m", "b"] {
        assert!(!z.contains(&child.to_string()), "{child} drawn: {z:?}");
    }
    for i in 0..3 {
        assert!(z.contains(&format!("rep~r{i}")), "{z:?}");
        assert!(z.contains(&format!("rep~r{i}~a")), "{z:?}");
        assert!(z.contains(&format!("rep~r{i}~b")), "{z:?}");
    }
    assert!(z.contains(&"plain".to_string()));
    // The source box is (0, 0)–(40, 20): copies step 50 to the right, in the repeat's space.
    assert_eq!(source_box(&scene, "rep"), Some([0.0, 0.0, 40.0, 20.0]));
    let (a0, _) = scene.accumulated_affine("rep~r0~a").unwrap();
    let (a2, _) = scene.accumulated_affine("rep~r2~a").unwrap();
    assert_eq!((a0.e, a0.f), (100.0, 50.0));
    assert_eq!((a2.e, a2.f), (200.0, 50.0));
    let (b2, _) = scene.accumulated_affine("rep~r2~b").unwrap();
    assert_eq!((b2.e, b2.f), (212.0, 57.0));
    // Copies keep their paint, opacity and clip mask (renamed to the copy's own mask).
    let g = scene.effective_node("rep~r1~g").unwrap();
    assert_eq!(g.str("clipMaskNodeId"), Some("rep~r1~m"));
    assert_eq!(g.f64("opacity"), Some(0.5));
    assert!(crate::display_list::clip_mask_source_ids(&scene).contains("rep~r1~m"));
    let a = scene.effective_node("rep~r1~a").unwrap();
    assert_eq!(a.str(REPEAT_OF_KEY), Some("rep"));
    assert_eq!(a.str(REPEAT_SOURCE_KEY), Some("a"));
    assert_eq!(a.json_field("fills"), Some(&solid("#f00")));
    assert!(scene.in_envelope("a") && scene.in_envelope("rep~r1~a") && !scene.in_envelope("plain"));
}

#[test]
fn a_rotated_copy_turns_its_art() {
    let scene = Scene::load(&doc(json!({"type": "radial", "count": 4, "radius": 100}))).unwrap();
    // Copy 1 is a quarter turn about (20, 110) (the source box (0,0)–(40,20) has its centre at (20, 10)).
    let (a, t) = scene.accumulated_affine("rep~r1~a").unwrap();
    assert!((t.rotation - 90.0).abs() < 1e-9);
    // The rectangle's origin (0, 0) → (130, 90) repeat-local → (230, 140) on the page.
    assert!(close((a.e, a.f), (230.0, 140.0)), "{a:?}");
}

#[test]
fn a_mirror_copy_is_reflected() {
    let scene = Scene::load(&doc(json!({"type": "mirror", "axis": {"x": 50, "y": 0, "angle": 90}}))).unwrap();
    let (a, _) = scene.accumulated_affine("rep~r1~a").unwrap();
    // x → 100 − x in the repeat's space: the rectangle's origin lands at 100 + 100.
    assert!(close((a.e, a.f), (200.0, 50.0)), "{a:?}");
    assert!(a.a * a.d - a.b * a.c < 0.0);
}

#[test]
fn nested_live_groups_resolve_inside_out() {
    // A radial repeat of a grid repeat: the outer copies the inner's copies.
    let d = json!({
        "nodes": [
            {"id": "outer", "type": "group", "x": 0, "y": 0, "children": ["inner"], "repeat": {"type": "mirror", "axis": {"x": 100, "y": 0, "angle": 90}}},
            {"id": "inner", "type": "group", "x": 0, "y": 0, "children": ["a"], "repeat": {"type": "grid", "rows": 1, "columns": 2}},
            {"id": "a", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10, "fills": solid("#f00")}
        ],
        "pages": [{"id": "p1", "width": 500, "height": 400, "nodeIds": ["outer"]}]
    });
    let scene = Scene::load(&d).unwrap();
    let z = scene.flattened_z_order_for_page("p1");
    assert!(z.contains(&"outer~r1~inner~r1~a".to_string()), "{z:?}");
    let (a, _) = scene.accumulated_affine("outer~r1~inner~r1~a").unwrap();
    // inner copy 1 at x = 10 (a 10-wide tile), mirrored across x = 100: its origin maps to 190.
    assert!(close((a.e, a.f), (190.0, 0.0)), "{a:?}");
    assert_eq!(scene.effective_node("outer~r1~inner~r1~a").unwrap().str(REPEAT_SOURCE_KEY), Some("a"));
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
        assert_eq!(a.envelope_mask_sources, b.envelope_mask_sources);
    }
    let mut d = doc(json!({"type": "radial", "count": 6, "radius": 80}));
    let mut scene = Scene::load(&d).unwrap();
    // Editing the source moves every copy.
    d["nodes"][1]["width"] = json!(90);
    scene.apply_node_edits(vec![RawNode::from_value(d["nodes"][1].clone()).unwrap()], &[], &[], None).unwrap();
    same(&scene, &Scene::load(&d).unwrap());
    assert_eq!(scene.effective_node("rep~r3~a").unwrap().f64("width"), Some(90.0));
    // The spec changes.
    d["nodes"][0]["repeat"] = json!({"type": "grid", "rows": 2, "columns": 2});
    scene.apply_node_edits(vec![RawNode::from_value(d["nodes"][0].clone()).unwrap()], &[], &[], None).unwrap();
    same(&scene, &Scene::load(&d).unwrap());
    // Released: the children draw again.
    d["nodes"][0].as_object_mut().unwrap().remove("repeat");
    scene.apply_node_edits(vec![RawNode::from_value(d["nodes"][0].clone()).unwrap()], &[], &[], None).unwrap();
    same(&scene, &Scene::load(&d).unwrap());
    assert!(scene.flattened_z_order_for_page("p1").contains(&"a".to_string()));
    // Made again on a plain group.
    d["nodes"][0]["repeat"] = json!({"type": "mirror", "axis": {"x": 0, "y": 0, "angle": 0}});
    scene.apply_node_edits(vec![RawNode::from_value(d["nodes"][0].clone()).unwrap()], &[], &[], None).unwrap();
    same(&scene, &Scene::load(&d).unwrap());
}

#[test]
fn a_huge_repeat_is_capped() {
    let scene = Scene::load(&doc(json!({"type": "grid", "rows": 1000, "columns": 1000}))).unwrap();
    let r = resolve(&scene, "rep");
    assert!(r.transients.len() <= MAX_REPEAT_NODES);
    assert!(r.skipped.iter().any(|s| s.contains("copies")), "{:?}", r.skipped);
}

#[test]
fn a_document_without_repeats_is_untouched() {
    let mut d = doc(json!(null));
    d["nodes"][0].as_object_mut().unwrap().remove("repeat");
    let scene = Scene::load(&d).unwrap();
    assert!(scene.envelope_transients.is_empty());
    assert!(scene.transient_by_id.keys().all(|id| id.starts_with("__page_bg")));
}
