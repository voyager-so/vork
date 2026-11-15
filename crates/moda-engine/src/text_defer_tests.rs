//! A live camera frame lays text out within the refine budget and leaves the rest for the next frames
//! (`text_draw::build_richtext_item`, `PlanCaches::text_deferrable`): each next frame (a patch of the retained
//! plan) lays out a budget's worth (at least one layer), until the plan equals a cold build. Exports and frames
//! without a budget never defer.

use super::*;
use crate::frame::{CameraParams, RenderOptions};
use moda_scene::value::json;

fn opts() -> RenderOptions {
    RenderOptions {
        preserve_page_fill: false,
        scale: 1.0,
        transparent: false,
        sample_count: 4,
        shader_time_s: 0.0,
        exclude: Vec::new(),
        isolate: Vec::new(),
        best_effort_text: true,
        text_masks_available: Some(true),
        working: None,
    }
}

fn cam() -> CameraParams {
    CameraParams {
        camera_x: -20.0,
        camera_y: -20.0,
        zoom: 0.5,
        viewport_w: 1600.0,
        viewport_h: 1000.0,
        dpr: 1.0,
        clear_rgb: [0.9, 0.9, 0.9],
    }
}

/// `n` text layers and `n` rectangles in a grid on the canvas root.
fn scene_json(n: usize) -> String {
    let mut nodes = Vec::new();
    let mut roots = Vec::new();
    for i in 0..n {
        let (x, y) = ((i % 10) as f64 * 150.0, (i / 10) as f64 * 60.0);
        nodes.push(json!({"id": format!("r{i}"), "type": "rectangle", "x": x, "y": y, "width": 140.0, "height": 50.0,
            "rotation": 0, "scale": 1, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#dddddd"}}]}));
        nodes.push(json!({"id": format!("t{i}"), "type": "richtext", "x": x + 4.0, "y": y + 4.0, "width": 130.0,
            "height": 20.0, "rotation": 0, "scale": 1,
            "htmlContent": format!("<p><span style=\"font-size: 14px; font-family: Inter; color: #111111\">Layer {i}</span></p>")}));
        roots.push(format!("r{i}"));
        roots.push(format!("t{i}"));
    }
    json!({ "nodes": nodes, "pages": [], "canvasRootNodeIds": roots, "layoutMode": "freeform", "config": {} })
        .to_string()
}

fn frame(
    resident: &ResidentScene,
    caches: &mut PlanCaches,
    budget: Option<f64>,
) -> Arc<crate::seams::CoveragePlan> {
    caches.begin_frame();
    caches.set_outline_refine_budget_ms(budget);
    let plan = build_camera_plan_retained(resident, caches, &[], &cam(), &opts())
        .expect("plan")
        .expect("a plan")
        .0;
    caches.set_outline_refine_budget_ms(None);
    plan
}

fn text_items(plan: &crate::seams::CoveragePlan) -> usize {
    plan.items
        .iter()
        .filter(|i| matches!(i, crate::seams::DrawItem::Text(_)))
        .count()
}

#[test]
fn a_first_frame_defers_text_past_its_budget_and_the_next_frame_draws_it_all() {
    let mut resident = ResidentScene::default();
    resident.update(&scene_json(60)).unwrap();
    let cold = {
        let mut fresh = PlanCaches::default();
        frame(&resident, &mut fresh, None)
    };
    let all_text = text_items(&cold);

    let mut caches = PlanCaches::default();
    // A zero budget: the first layout of the frame runs, every later one is deferred.
    let first = frame(&resident, &mut caches, Some(0.0));
    assert!(caches.text_deferred > 0, "text was deferred");
    assert!(
        text_items(&first) < all_text,
        "the first frame draws fewer text layers"
    );
    assert!(
        caches.outline_refining(),
        "the host is asked for another frame"
    );
    assert_eq!(
        first
            .items
            .iter()
            .filter(|i| !matches!(i, crate::seams::DrawItem::Text(_)))
            .count(),
        cold.items
            .iter()
            .filter(|i| !matches!(i, crate::seams::DrawItem::Text(_)))
            .count(),
        "the shapes are all there"
    );

    // The next frames (patches of the retained plan) lay out what is left, a budget's worth each: with a zero
    // budget, one layer a frame; with room, the rest in one.
    let before = text_items(&first);
    let second = frame(&resident, &mut caches, Some(0.0));
    assert!(text_items(&second) > before, "a patch frame lays out at least one queued layer");
    assert!(caches.outline_refining(), "and queues the rest");
    let mut frames = 0;
    let mut second = second;
    while caches.outline_refining() {
        second = frame(&resident, &mut caches, Some(1000.0));
        frames += 1;
        assert!(frames < 5, "a large budget lays the rest out at once");
    }
    assert_eq!(text_items(&second), all_text);
    let scene = resident.scene().unwrap();
    assert_eq!(
        first_camera_item_divergence(
            scene,
            &[],
            &cam(),
            &opts(),
            resident.hashes().unwrap(),
            &second
        ),
        None,
        "the second frame equals a cold build"
    );
}

/// While the zoom changes, a live frame refines nothing (what it queued waits: its band would change again next
/// frame, and a refit costs the frame its prepared pass); once the zoom holds, refinement resumes.
#[test]
fn refinement_holds_while_the_zoom_changes() {
    let mut resident = ResidentScene::default();
    resident.update(&scene_json(60)).unwrap();
    let mut caches = PlanCaches::default();
    let first = frame(&resident, &mut caches, Some(0.0));
    let queued = text_items(&first);
    assert!(caches.outline_refining(), "text was queued");
    let frame_at = |caches: &mut PlanCaches, zoom: f64| {
        caches.begin_frame();
        caches.set_outline_refine_budget_ms(Some(1000.0));
        let c = CameraParams { zoom, ..cam() };
        let plan = build_camera_plan_retained(&resident, caches, &[], &c, &opts())
            .expect("plan")
            .expect("a plan")
            .0;
        caches.set_outline_refine_budget_ms(None);
        plan
    };
    let zooming = frame_at(&mut caches, 0.51);
    assert_eq!(text_items(&zooming), queued, "a zooming frame refines nothing");
    assert!(caches.outline_refining(), "the queue waits");
    let held = frame_at(&mut caches, 0.51);
    assert!(text_items(&held) > queued, "the zoom held: refinement resumes");
}

#[test]
fn without_a_budget_nothing_is_deferred() {
    let mut resident = ResidentScene::default();
    resident.update(&scene_json(40)).unwrap();
    let mut caches = PlanCaches::default();
    let plan = frame(&resident, &mut caches, None);
    assert_eq!(caches.text_deferred, 0);
    assert!(!caches.outline_refining());
    assert_eq!(
        first_camera_item_divergence(
            resident.scene().unwrap(),
            &[],
            &cam(),
            &opts(),
            resident.hashes().unwrap(),
            &plan
        ),
        None
    );
}

#[test]
fn deferred_text_laid_out_between_frames_serves_the_next_frame_exactly() {
    let mut resident = ResidentScene::default();
    resident.update(&scene_json(60)).unwrap();
    let mut caches = PlanCaches::default();
    let first = frame(&resident, &mut caches, Some(0.0));
    let deferred = caches.text_prewarm_len();
    assert!(deferred > 0, "the first frame deferred text");
    let before = text_items(&first);
    // Between frames: lay it all out (a large budget), then one frame draws it.
    assert_eq!(prewarm_deferred_text(&resident, &mut caches, 1e6), 0);
    assert_eq!(caches.text_prewarmed as usize, deferred);
    let misses = caches.text_misses;
    let mut next = frame(&resident, &mut caches, Some(0.0));
    let mut frames = 1;
    while caches.outline_refining() {
        next = frame(&resident, &mut caches, Some(0.0));
        frames += 1;
        assert!(frames < 5, "the laid-out text converges in a frame or two, not one layer a frame");
    }
    assert!(text_items(&next) > before);
    assert_eq!(caches.text_misses, misses, "every re-planned layer was a cache hit");
    assert_eq!(
        first_camera_item_divergence(
            resident.scene().unwrap(),
            &[],
            &cam(),
            &opts(),
            resident.hashes().unwrap(),
            &next
        ),
        None,
        "equals a cold build"
    );
}

#[test]
fn prewarm_stops_at_its_budget_and_keeps_the_rest() {
    let mut resident = ResidentScene::default();
    resident.update(&scene_json(60)).unwrap();
    let mut caches = PlanCaches::default();
    frame(&resident, &mut caches, Some(0.0));
    let deferred = caches.text_prewarm_len();
    // A zero budget lays out one layer.
    assert_eq!(prewarm_deferred_text(&resident, &mut caches, 0.0), deferred - 1);
}
