//! The Design page cache (`page_cache`): a page shown again reuses its resident scene and retained camera
//! plan, an edit made while it was parked patches correctly after the switch back, and the budget holds.

use super::*;
use crate::frame::{CameraParams, RenderOptions};
use crate::residency::{build_camera_plan_retained, first_camera_item_divergence};
use moda_scene::value::{json, Value};

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
        viewport_w: 1200.0,
        viewport_h: 800.0,
        dpr: 1.0,
        clear_rgb: [0.9, 0.9, 0.9],
    }
}

fn shape(id: &str, i: usize, color: &str) -> Value {
    json!({
        "id": id, "type": if i % 3 == 0 { "ellipse" } else { "rectangle" },
        "x": (i % 20) as f64 * 60.0, "y": (i / 20) as f64 * 50.0, "width": 40.0, "height": 30.0,
        "rotation": 0, "scale": 1,
        "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": color}}],
        "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#222222"}, "width": 2}],
    })
}

/// A page of a Design document as the editor feeds it: `n` stroked shapes (which tessellate) on the canvas
/// root, in rows.
fn page(prefix: &str, n: usize, color: &str) -> Value {
    let nodes: Vec<Value> = (0..n).map(|i| shape(&format!("{prefix}{i}"), i, color)).collect();
    let roots: Vec<String> = (0..n).map(|i| format!("{prefix}{i}")).collect();
    json!({ "nodes": nodes, "pages": [], "canvasRootNodeIds": roots, "layoutMode": "freeform", "config": {} })
}

fn plan(resident: &ResidentScene, caches: &mut PlanCaches) -> std::sync::Arc<crate::seams::CoveragePlan> {
    caches.begin_frame();
    build_camera_plan_retained(resident, caches, &[], &cam(), &opts())
        .expect("plan")
        .expect("a plan")
        .0
}

fn assert_matches_cold(resident: &ResidentScene, got: &crate::seams::CoveragePlan) {
    let scene = resident.scene().expect("resident");
    let hashes = resident.hashes().expect("hashes");
    assert_eq!(
        first_camera_item_divergence(scene, &[], &cam(), &opts(), hashes, got),
        None,
        "the plan equals a cold build of the scene"
    );
}

#[test]
fn a_page_shown_again_reuses_its_scene_and_plan() {
    let (a, b) = (page("a", 200, "#3a86ff").to_string(), page("b", 200, "#ef476f").to_string());
    let mut resident = ResidentScene::default();
    let mut caches = PlanCaches::default();
    let mut pages = PageCache::default();

    resident.update(&a).unwrap();
    let plan_a = plan(&resident, &mut caches);
    let gen_a = resident.generation();
    assert!(pages.park("canvas-a", &mut resident, &mut caches, 1000));
    assert!(resident.scene().is_none(), "parking leaves nothing resident");

    resident.update(&b).unwrap();
    assert_ne!(resident.generation(), gen_a, "generations never repeat across scenes");
    plan(&resident, &mut caches);
    assert!(pages.park("canvas-b", &mut resident, &mut caches, 1000));
    assert_eq!(pages.keys(), vec!["canvas-a", "canvas-b"]);

    // Page A again: no parse, the same generation, the same plan, no tessellation.
    assert!(pages.unpark("canvas-a", &mut resident, &mut caches));
    assert_eq!(resident.generation(), gen_a);
    let misses = caches.tess_misses + caches.text_misses;
    let static_hits = caches.incr_static_hits;
    let again = plan(&resident, &mut caches);
    assert!(std::sync::Arc::ptr_eq(&plan_a, &again), "the parked plan serves the first frame back");
    assert_eq!(caches.tess_misses + caches.text_misses, misses, "nothing re-tessellated");
    assert!(caches.incr_static_hits > static_hits, "served by the retained slot");
    assert_matches_cold(&resident, &again);
    assert_eq!((pages.hits, pages.misses), (1, 0));
    assert!(!pages.unpark("canvas-c", &mut resident, &mut caches), "a page never parked is a miss");
}

#[test]
fn an_edit_made_while_parked_patches_after_the_switch_back() {
    let mut resident = ResidentScene::default();
    let mut caches = PlanCaches::default();
    let mut pages = PageCache::default();
    resident.update(&page("a", 120, "#3a86ff").to_string()).unwrap();
    plan(&resident, &mut caches);
    pages.park("canvas-a", &mut resident, &mut caches, 1000);
    resident.update(&page("b", 120, "#ef476f").to_string()).unwrap();
    plan(&resident, &mut caches);
    pages.park("canvas-b", &mut resident, &mut caches, 1000);
    assert!(pages.unpark("canvas-a", &mut resident, &mut caches));

    // The host's patch of what changed on page A while it was away: a moved, recoloured node and a delete.
    let recolor = json!([{"id": "fill-0", "paint": {"type": "solid", "color": "#06d6a0"}}]);
    let mut moved = shape("a5", 5, "#3a86ff");
    moved["x"] = json!(900.0);
    moved["fills"] = recolor.clone();
    let roots: Vec<String> = (0..120).filter(|i| *i != 7).map(|i| format!("a{i}")).collect();
    let patch = json!({ "upserts": [moved], "removes": ["a7"], "canvasRootNodeIds": roots });
    resident.apply_props(&patch.to_string()).unwrap();
    let patched = plan(&resident, &mut caches);
    assert_matches_cold(&resident, &patched);

    // And the resident scene equals a cold load of the same page.
    let mut doc = page("a", 120, "#3a86ff");
    let nodes = doc["nodes"].as_array_mut().unwrap();
    nodes.retain(|n| n["id"] != "a7");
    for n in nodes.iter_mut().filter(|n| n["id"] == "a5") {
        n["x"] = json!(900.0);
        n["fills"] = recolor.clone();
    }
    doc["canvasRootNodeIds"] = json!(roots);
    assert_eq!(resident.divergence_from_full_load(&doc.to_string()), None);
}

#[test]
fn the_budget_keeps_the_most_recent_pages() {
    let mut resident = ResidentScene::default();
    let mut caches = PlanCaches::default();
    let mut pages = PageCache::default();
    pages.set_budget(2, 10_000_000);
    for k in ["p1", "p2", "p3"] {
        resident.update(&page(k, 10, "#3a86ff").to_string()).unwrap();
        plan(&resident, &mut caches);
        pages.park(k, &mut resident, &mut caches, 1000);
    }
    assert_eq!(pages.keys(), vec!["p2", "p3"], "the page count evicts the least recently parked");
    assert_eq!(pages.evictions, 1);

    // Bytes: a page over the budget on its own is not kept; the total never passes the budget.
    resident.update(&page("p4", 10, "#3a86ff").to_string()).unwrap();
    plan(&resident, &mut caches);
    let state = caches.take_scene_state();
    let plan_bytes = state.plan_bytes();
    assert!(plan_bytes > 0, "a retained plan has a size");
    caches.restore_scene_state(state);
    let each = 1000 + plan_bytes;
    pages.set_budget(8, 3 * each);
    assert!(!pages.park("big", &mut resident, &mut caches, 10_000_000), "over the budget alone");
    assert_eq!(pages.refused, 1);
    for k in ["p5", "p6", "p7", "p8"] {
        resident.update(&page(k, 10, "#3a86ff").to_string()).unwrap();
        plan(&resident, &mut caches);
        pages.park(k, &mut resident, &mut caches, 1000);
        assert!(pages.bytes() <= 3 * each, "within the byte budget");
    }
    assert_eq!(pages.keys(), vec!["p6", "p7", "p8"]);

    pages.set_budget(0, 0);
    assert!(pages.is_empty(), "a zero budget turns the cache off");
}

#[test]
fn a_switch_never_evicts_the_page_it_shows() {
    let mut resident = ResidentScene::default();
    let mut caches = PlanCaches::default();
    let mut pages = PageCache::default();
    // Room for two pages: A and B parked, C on screen.
    for k in ["A", "B"] {
        resident.update(&page(k, 30, "#3a86ff").to_string()).unwrap();
        plan(&resident, &mut caches);
        pages.park(k, &mut resident, &mut caches, 1000);
    }
    let one = pages.bytes() / 2;
    pages.set_budget(2, 2 * one + one / 2);
    resident.update(&page("C", 30, "#3a86ff").to_string()).unwrap();
    plan(&resident, &mut caches);

    // C -> A: parking C would evict the least recent (A) if A were still parked; A is taken out first.
    let (parked, shown, bytes) = pages.switch(Some("C"), Some("A"), &mut resident, &mut caches, 1000);
    assert!(parked && shown, "C parked, A shown");
    assert_eq!(bytes, 1000, "the shown scene's resident bytes come back with it");
    assert_eq!(pages.keys(), vec!["B", "C"]);
    assert!(resident.scene().is_some_and(|s| s.node("A0").is_some()), "A is resident");
    let shown_plan = plan(&resident, &mut caches);
    assert_matches_cold(&resident, &shown_plan);

    // A page that is not parked: the outgoing one parks, nothing is resident (the host full-feeds).
    let (parked, shown, _) = pages.switch(Some("A"), Some("D"), &mut resident, &mut caches, 1000);
    assert!(parked && !shown);
    assert!(resident.scene().is_none());
    assert_eq!(pages.keys(), vec!["C", "A"], "the budget evicted B, the least recently parked");
}
