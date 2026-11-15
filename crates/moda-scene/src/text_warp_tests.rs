use super::*;
use crate::live_effects::{split_copies, EFFECT_COPY_FILLS_KEY};
use crate::value::json;

fn square(x: f64, y: f64, s: f64) -> Vec<PathPoint> {
    vec![
        PathPoint::Move { x, y },
        PathPoint::Line { x: x + s, y },
        PathPoint::Line { x: x + s, y: y + s },
        PathPoint::Line { x, y: y + s },
        PathPoint::Line { x, y },
    ]
}

fn solid(c: &str) -> Value {
    json!([{"id": "f", "paint": {"type": "solid", "color": c}}])
}

fn text(effects: Value) -> RawNode {
    RawNode::new(
        json!({"id": "t", "type": "richtext", "x": 5, "y": 6, "width": 100, "height": 20,
            "htmlContent": "<p>ab</p>", "effects": effects, "opacity": 0.5,
            "textFills": [{"id": "tf", "paint": {"type": "solid", "color": "#123456"}}]})
        .as_object()
        .unwrap()
        .clone(),
    )
}

#[test]
fn only_text_with_a_visible_geometry_effect_is_effected() {
    assert!(is_effected_text(&text(
        json!([{"id": "w", "type": "warp", "style": "arc", "bend": 50}])
    )));
    assert!(!is_effected_text(&text(json!([]))));
    assert!(!is_effected_text(&text(
        json!([{"id": "w", "type": "warp", "style": "arc", "bend": 50, "visible": false}])
    )));
    // A drawing effect alone does not reshape the text.
    assert!(!is_effected_text(&text(
        json!([{"id": "s", "type": "drop-shadow", "color": "#000", "offset": {"x": 1, "y": 1}, "radius": 2}])
    )));
}

#[test]
fn one_fill_draws_as_one_path_with_that_fill() {
    let node = text(json!([{"id": "w", "type": "warp", "style": "arc", "bend": 50},
        {"id": "s", "type": "drop-shadow", "color": "#000", "offset": {"x": 1, "y": 1}, "radius": 2}]));
    let r = record(
        &node,
        &[TextOutlineGroup {
            points: [square(0.0, 0.0, 10.0), square(20.0, 0.0, 10.0)].concat(),
            fills: solid("#ff0000"),
        }],
    );
    assert_eq!(r.node_type(), "path");
    assert_eq!(r.json_field("fills"), Some(&solid("#ff0000")));
    assert!(r.json_field("textFills").is_none());
    assert!(r.json_field(EFFECT_COPY_FILLS_KEY).is_none());
    // Placement, opacity and the drawing effects stay; the warp is baked.
    assert_eq!(r.f64("x"), Some(5.0));
    assert_eq!(r.f64("opacity"), Some(0.5));
    let effects = r.array("effects").unwrap();
    assert_eq!(effects.len(), 1);
    assert_eq!(effects[0]["type"], json!("drop-shadow"));
    assert!(split_copies(&r).is_none());
}

#[test]
fn each_fill_group_is_a_copy_with_its_own_fill_and_copies_follow_their_group() {
    // Two runs, then a Transform effect with one copy: four copies, red, red, blue, blue.
    let node = text(json!([
        {"id": "w", "type": "warp", "style": "arc", "bend": 50},
        {"id": "c", "type": "transform", "copies": 1, "moveX": 0, "moveY": 30}
    ]));
    let r = record(
        &node,
        &[
            TextOutlineGroup {
                points: square(0.0, 0.0, 10.0),
                fills: solid("#ff0000"),
            },
            TextOutlineGroup {
                points: square(20.0, 0.0, 10.0),
                fills: solid("#0000ff"),
            },
        ],
    );
    let parts = split_copies(&r).expect("copies");
    let colours: Vec<Value> = parts
        .iter()
        .map(|p| p.json_field("fills").unwrap()[0]["paint"]["color"].clone())
        .collect();
    assert_eq!(
        colours,
        vec![json!("#ff0000"), json!("#ff0000"), json!("#0000ff"), json!("#0000ff")]
    );
    for p in &parts {
        assert!(p.json_field(EFFECT_COPY_FILLS_KEY).is_none());
    }
}

#[test]
fn the_warp_bends_every_group_over_their_shared_bounds() {
    // Two squares side by side under an arc: measured together, the outer edges drop below the inner ones. Warped
    // one by one each square would stay symmetric about its own centre.
    let node = text(json!([{"id": "w", "type": "warp", "style": "arc", "bend": 80}]));
    let r = record(
        &node,
        &[
            TextOutlineGroup {
                points: square(0.0, 0.0, 10.0),
                fills: solid("#ff0000"),
            },
            TextOutlineGroup {
                points: square(90.0, 0.0, 10.0),
                fills: solid("#0000ff"),
            },
        ],
    );
    let parts = split_copies(&r).unwrap();
    let top_at = |p: &RawNode, pick_left: bool| {
        let pts: Vec<(f64, f64)> = p
            .array("points")
            .unwrap()
            .iter()
            .filter_map(PathPoint::from_json)
            .map(|q| match q {
                PathPoint::Move { x, y } | PathPoint::Line { x, y } => (x, y),
                PathPoint::Curve { x, y, .. }
                | PathPoint::Quadratic { x, y, .. }
                | PathPoint::Arc { x, y, .. } => (x, y),
            })
            .collect();
        let x = if pick_left {
            pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min)
        } else {
            pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max)
        };
        pts.iter()
            .filter(|p| (p.0 - x).abs() < 2.0)
            .map(|p| p.1)
            .fold(f64::INFINITY, f64::min)
    };
    // Red: its left (outer) edge is lower (greater y) than its right (inner) one.
    assert!(top_at(&parts[0], true) > top_at(&parts[0], false) + 1.0);
    // Blue: mirrored.
    assert!(top_at(&parts[1], false) > top_at(&parts[1], true) + 1.0);
}

#[test]
fn empty_text_draws_nothing() {
    let node = text(json!([{"id": "w", "type": "warp", "style": "arc", "bend": 50}]));
    let r = record(&node, &[]);
    assert_eq!(r.node_type(), "path");
    assert_eq!(r.array("points").map(Vec::len), Some(0));
}

static FONTS_READY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A stand-in for the engine's outliner: text mentioning "late" waits for its fonts until `FONTS_READY`.
fn test_outliner(_scene: &Scene, node: &RawNode) -> TextOutlines {
    let late = node.str("htmlContent").is_some_and(|h| h.contains("late"));
    if late && !FONTS_READY.load(std::sync::atomic::Ordering::SeqCst) {
        return TextOutlines::Pending;
    }
    TextOutlines::Ready(vec![TextOutlineGroup {
        points: square(0.0, 0.0, 10.0),
        fills: solid("#000000"),
    }])
}

#[test]
fn text_whose_fonts_arrive_late_warps_once_they_do() {
    install_text_outliner(test_outliner);
    let warp = json!([{"id": "w", "type": "warp", "style": "arc", "bend": 50}]);
    let node = |id: &str, html: &str| {
        json!({"id": id, "type": "richtext", "x": 0, "y": 0, "width": 100, "height": 20,
            "htmlContent": html, "effects": warp})
    };
    let mut scene = Scene::load(&json!({
        "nodes": [node("early", "<p>now</p>"), node("t", "<p>late</p>")],
        "pages": [{"id": "p", "width": 200, "height": 200, "nodeIds": ["early", "t"]}]
    })).unwrap();
    assert_eq!(scene.effective_node("early").unwrap().node_type(), "path");
    // Its fonts are not in yet: it draws as plain text for now, and is pending.
    assert_eq!(scene.effective_node("t").unwrap().node_type(), "richtext");
    assert_eq!(scene.pending_text_warps(), vec!["t".to_string()]);
    assert!(scene.retry_pending_text_warps().is_empty());
    FONTS_READY.store(true, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(scene.retry_pending_text_warps(), vec!["t".to_string()]);
    assert_eq!(scene.effective_node("t").unwrap().node_type(), "path");
    assert!(scene.pending_text_warps().is_empty());
}
