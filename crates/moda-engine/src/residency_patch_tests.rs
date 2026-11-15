//! The scene-sync patch lane (`upserts` / `removes` / `pageNodeIds` in a
//! `set_node_props` patch): every patched resident scene must equal a cold load
//! of the same document (`divergence_from_full_load`), and its retained plans
//! must equal plans built from that cold load.

use super::*;
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

fn fill(color: &str) -> Value {
    json!([{"id": "fill-0", "paint": {"type": "solid", "color": color}}])
}

fn rect(id: &str, x: f64, y: f64, stroked: bool) -> Value {
    let mut n = json!({
        "id": id, "type": "rectangle", "x": x, "y": y,
        "width": 40.0, "height": 30.0, "rotation": 0, "scale": 1,
        "cornerRadius": 3, "fills": fill("#3a86ff"),
    });
    if stroked {
        // A sharp stroked rect tessellates (a plain rect is an analytic body).
        n["cornerRadius"] = json!(0);
        n["strokes"] =
            json!([{"id": "stroke-0", "paint": {"type": "solid", "color": "#222222"}, "width": 2}]);
    }
    n
}

fn table(id: &str, rows: usize) -> Value {
    let cells: Vec<Value> = (0..rows)
        .map(|r| {
            json!((0..2)
                .map(|c| json!({"id": format!("{id}-c{r}-{c}"), "textContent": format!("<p>R{r}C{c}</p>")}))
                .collect::<Vec<_>>())
        })
        .collect();
    json!({
        "id": id, "type": "table", "x": 400.0, "y": 300.0,
        "width": 120.0, "height": 24.0 * rows as f64, "rotation": 0, "scale": 1,
        "rows": (0..rows).map(|_| json!({"size": 24.0})).collect::<Vec<_>>(),
        "columns": [{"size": 60.0}, {"size": 60.0}],
        "cells": cells, "rowStrokeConfig": "all", "columnStrokeConfig": "all",
        "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#c9c9c9"}, "width": 1}],
    })
}

/// The document as the editor holds it; each edit is applied to it and, as a
/// patch, to the resident scene.
struct Doc(Value);

impl Doc {
    fn new(pages: usize, per_page: usize) -> Doc {
        let mut nodes = Vec::new();
        let mut page_list = Vec::new();
        for p in 0..pages {
            let ids: Vec<String> = (0..per_page).map(|i| format!("p{p}r{i}")).collect();
            for (i, id) in ids.iter().enumerate() {
                nodes.push(rect(
                    id,
                    (i * 37 % 700) as f64,
                    (i * 23 % 500) as f64,
                    i % 2 == 0,
                ));
            }
            page_list.push(
                json!({"id": format!("page-{p}"), "name": "p", "width": 800.0, "height": 600.0,
                "fills": fill("#ffffff"), "nodeIds": ids}),
            );
        }
        Doc(
            json!({"nodes": nodes, "pages": page_list, "config": {"renderPageClipping": true},
            "canvasRootNodeIds": [], "layoutMode": "vertical", "variables": []}),
        )
    }
    fn json(&self) -> String {
        self.0.to_string()
    }
    fn nodes(&mut self) -> &mut Vec<Value> {
        self.0["nodes"].as_array_mut().unwrap()
    }
    fn node(&self, id: &str) -> Option<Value> {
        self.0["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["id"] == id)
            .cloned()
    }
    fn page_ids(&self, p: usize) -> Vec<String> {
        moda_scene::value::from_value(self.0["pages"][p]["nodeIds"].clone()).unwrap()
    }
    fn upsert(&mut self, node: Value) {
        let id = node["id"].clone();
        match self.nodes().iter().position(|n| n["id"] == id) {
            Some(i) => self.nodes()[i] = node,
            None => self.nodes().push(node),
        }
    }
    fn remove(&mut self, id: &str) {
        self.nodes().retain(|n| n["id"] != id);
    }
    fn set_page(&mut self, p: usize, ids: &[String]) {
        self.0["pages"][p]["nodeIds"] = json!(ids);
    }
}

/// One edit: applied to the document model, then sent as a patch.
#[derive(Default)]
struct Edit {
    upserts: Vec<Value>,
    removes: Vec<String>,
    pages: Vec<(usize, Vec<String>)>,
}

impl Edit {
    fn apply(&self, doc: &mut Doc, resident: &mut ResidentScene) -> anyhow::Result<()> {
        for n in &self.upserts {
            doc.upsert(n.clone());
        }
        for id in &self.removes {
            doc.remove(id);
        }
        let mut page_node_ids = moda_scene::value::Map::new();
        for (p, ids) in &self.pages {
            doc.set_page(*p, ids);
            page_node_ids.insert(format!("page-{p}").into(), json!(ids));
        }
        let patch =
            json!({"upserts": self.upserts, "removes": self.removes, "pageNodeIds": page_node_ids});
        resident.apply_props(&patch.to_string())
    }
}

fn check(resident: &ResidentScene, caches: &mut PlanCaches, doc: &Doc, label: &str) {
    let json = doc.json();
    assert_eq!(
        resident.divergence_from_full_load(&json),
        None,
        "{label}: resident diverged"
    );
    let fresh = moda_scene::scene::load_scene_json(&json).unwrap();
    for page in 0..fresh.doc().pages.len() {
        caches.begin_frame();
        let (incr, _w) = build_plan_retained(resident, caches, page, &opts()).unwrap();
        let (full, _w) = crate::frame::build_coverage_plan(&fresh, page, &opts()).unwrap();
        assert_eq!(
            plan_items_fingerprint(&incr),
            plan_items_fingerprint(&full),
            "{label}: page {page} plan diverged from a full rebuild"
        );
    }
    // The editor's viewport lane: the camera plan over every page, from the patched
    // resident scene and its retained caches vs from a cold load.
    let pages: Vec<usize> = (0..fresh.doc().pages.len()).collect();
    let cam = crate::frame::CameraParams {
        camera_x: -50.0,
        camera_y: -50.0,
        zoom: 0.5,
        viewport_w: 900.0,
        viewport_h: 900.0,
        dpr: 1.0,
        clear_rgb: [1.0, 1.0, 1.0],
    };
    caches.begin_frame();
    let incr = build_camera_plan_retained(resident, caches, &pages, &cam, &opts())
        .unwrap()
        .map(|(p, _w)| p);
    let mut cold = ResidentScene::default();
    cold.update(&json).unwrap();
    let full = build_camera_plan_retained(&cold, &mut PlanCaches::default(), &pages, &cam, &opts())
        .unwrap()
        .map(|(p, _w)| p);
    assert_eq!(
        incr.map(|p| plan_items_fingerprint(&p)),
        full.map(|p| plan_items_fingerprint(&p)),
        "{label}: camera plan diverged from a full rebuild"
    );
}

fn start(doc: &Doc) -> (ResidentScene, PlanCaches) {
    let mut resident = ResidentScene::default();
    resident.update(&doc.json()).unwrap();
    let mut caches = PlanCaches::default();
    check(&resident, &mut caches, doc, "initial");
    (resident, caches)
}

fn strings(ids: &[&str]) -> Vec<String> {
    ids.iter().map(|s| s.to_string()).collect()
}

/// The first layer on an empty page, then a fill edit on it.
#[test]
fn first_layer_on_an_empty_page_matches_full_rebuild() {
    let mut doc = Doc::new(1, 0);
    let (mut resident, mut caches) = start(&doc);
    check(&resident, &mut caches, &doc, "empty again");
    Edit {
        upserts: vec![rect("first", 10.0, 10.0, false)],
        pages: vec![(0, strings(&["first"]))],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "first layer");
    let mut red = rect("first", 10.0, 10.0, false);
    red["fills"] = fill("#ff0000");
    Edit {
        upserts: vec![red],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "fill");
}

#[test]
fn add_remove_reorder_match_full_rebuild() {
    let mut doc = Doc::new(2, 12);
    let (mut resident, mut caches) = start(&doc);

    // Add a node on top of page 0.
    let mut ids = doc.page_ids(0);
    ids.push("new1".into());
    let e = Edit {
        upserts: vec![rect("new1", 10.0, 10.0, true)],
        pages: vec![(0, ids)],
        ..Edit::default()
    };
    e.apply(&mut doc, &mut resident).unwrap();
    check(&resident, &mut caches, &doc, "add");

    // Send it to the back.
    let mut ids = doc.page_ids(0);
    ids.retain(|i| i != "new1");
    ids.insert(0, "new1".into());
    Edit {
        pages: vec![(0, ids)],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "reorder");

    // Delete two nodes.
    let mut ids = doc.page_ids(0);
    ids.retain(|i| i != "p0r3" && i != "p0r4");
    let e = Edit {
        removes: strings(&["p0r3", "p0r4"]),
        pages: vec![(0, ids)],
        ..Edit::default()
    };
    e.apply(&mut doc, &mut resident).unwrap();
    check(&resident, &mut caches, &doc, "remove");

    // Move a node to the other page.
    let mut a = doc.page_ids(0);
    a.retain(|i| i != "p0r5");
    let mut b = doc.page_ids(1);
    b.push("p0r5".into());
    Edit {
        pages: vec![(0, a), (1, b)],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "move page");

    // Undo the delete: the nodes come back where they were.
    let mut ids = doc.page_ids(0);
    ids.insert(2, "p0r3".into());
    ids.insert(3, "p0r4".into());
    let e = Edit {
        upserts: vec![
            rect("p0r3", 111.0, 69.0, false),
            rect("p0r4", 148.0, 92.0, true),
        ],
        pages: vec![(0, ids)],
        ..Edit::default()
    };
    e.apply(&mut doc, &mut resident).unwrap();
    check(&resident, &mut caches, &doc, "re-add");
}

#[test]
fn group_and_ungroup_match_full_rebuild() {
    let mut doc = Doc::new(1, 8);
    let (mut resident, mut caches) = start(&doc);

    // Group r1 + r2.
    let mut c1 = doc.node("p0r1").unwrap();
    let mut c2 = doc.node("p0r2").unwrap();
    c1["parentId"] = json!("g1");
    c2["parentId"] = json!("g1");
    let group = json!({"id": "g1", "type": "group", "x": 0.0, "y": 0.0, "rotation": 0, "scale": 1,
        "children": ["p0r1", "p0r2"]});
    let ids = strings(&["p0r0", "g1", "p0r3", "p0r4", "p0r5", "p0r6", "p0r7"]);
    let e = Edit {
        upserts: vec![group, c1.clone(), c2.clone()],
        pages: vec![(0, ids)],
        ..Edit::default()
    };
    e.apply(&mut doc, &mut resident).unwrap();
    check(&resident, &mut caches, &doc, "group");

    // Reorder inside the group (the parent's `children` changes).
    let mut g = doc.node("g1").unwrap();
    g["children"] = json!(["p0r2", "p0r1"]);
    Edit {
        upserts: vec![g.clone()],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "reorder in group");

    // Move the group (a parent transform edit).
    g["x"] = json!(25.0);
    Edit {
        upserts: vec![g],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "move group");

    // Ungroup.
    c1.as_object_mut().unwrap().remove("parentId");
    c2.as_object_mut().unwrap().remove("parentId");
    let ids = strings(&[
        "p0r0", "p0r1", "p0r2", "p0r3", "p0r4", "p0r5", "p0r6", "p0r7",
    ]);
    let e = Edit {
        upserts: vec![c1, c2],
        removes: strings(&["g1"]),
        pages: vec![(0, ids)],
    };
    e.apply(&mut doc, &mut resident).unwrap();
    check(&resident, &mut caches, &doc, "ungroup");
}

#[test]
fn table_hosts_follow_edits_and_pages() {
    let mut doc = Doc::new(2, 4);
    let (mut resident, mut caches) = start(&doc);

    let mut ids = doc.page_ids(0);
    ids.push("t1".into());
    Edit {
        upserts: vec![table("t1", 3)],
        pages: vec![(0, ids)],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "add table");

    // A row more: the expansion changes shape.
    Edit {
        upserts: vec![table("t1", 4)],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "table row");

    // To page 1: its transients' page follows.
    let mut a = doc.page_ids(0);
    a.retain(|i| i != "t1");
    let mut b = doc.page_ids(1);
    b.insert(0, "t1".into());
    Edit {
        pages: vec![(0, a), (1, b)],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "table to page 1");

    let mut b = doc.page_ids(1);
    b.retain(|i| i != "t1");
    Edit {
        removes: strings(&["t1"]),
        pages: vec![(1, b)],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "remove table");
    let scene = resident.scene().unwrap();
    assert!(
        scene.transient_by_id().keys().all(|k| !k.starts_with("t1")),
        "table transients left behind"
    );
}

/// A replace carries the whole node, so a key the edit dropped is dropped (a
/// `nodes` prop merge can only add keys), variables resolve as a load resolves
/// them, and a property-only replace still takes the incremental plan patch.
#[test]
fn replace_drops_keys_resolves_variables_and_stays_incremental() {
    let mut doc = Doc::new(1, 10);
    doc.0["variables"] = json!([{"id": "v1", "name": "brand", "type": "color", "valuesByMode": {"default": "#ff0000"}}]);
    let (mut resident, mut caches) = start(&doc);

    let mut n = doc.node("p0r1").unwrap();
    n["fills"] = fill("#00ff00");
    let hits = caches.incr_patch_hits;
    Edit {
        upserts: vec![n.clone()],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "fill");
    assert!(
        caches.incr_patch_hits > hits,
        "a property-only replace must patch the plan incrementally"
    );

    n.as_object_mut().unwrap().remove("cornerRadius");
    Edit {
        upserts: vec![n.clone()],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "dropped key");
    assert!(resident
        .scene()
        .unwrap()
        .node("p0r1")
        .unwrap()
        .json_field("cornerRadius")
        .is_none());

    n["fills"] = json!([{"id": "fill-0", "paint": {"type": "solid", "color": {"type": "variable", "variableId": "v1"}}}]);
    Edit {
        upserts: vec![n],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "variable-bound fill");

    // An unknown type is dropped, as hydration drops it.
    let mut ids = doc.page_ids(0);
    ids.push("odd".into());
    let e = Edit {
        upserts: vec![json!({"id": "odd", "type": "sparkle"})],
        pages: vec![(0, ids)],
        ..Edit::default()
    };
    e.apply(&mut doc, &mut resident).unwrap();
    check(&resident, &mut caches, &doc, "unknown type");
}

/// E-M.7: a replace that adds or removes a layer filter (layer blur, noise,
/// texture) or a backdrop in `effects`
/// reshapes the display list (a capture group, a pass split) and must match a
/// full rebuild; editing an existing drop shadow only repaints and stays incremental.
#[test]
fn effects_replaces_match_full_rebuild() {
    let mut doc = Doc::new(1, 10);
    let (mut resident, mut caches) = start(&doc);
    let mut n = doc.node("p0r2").unwrap();
    let steps = [
        ("layer blur added", json!([{"id": "b", "type": "layer-blur", "radius": 6}])),
        ("layer blur removed", json!([])),
        ("noise added", json!([{"id": "n", "type": "noise", "mode": "mono", "color": "#000000", "size": 1, "density": 0.5}])),
        ("texture added over the noise", json!([
            {"id": "n", "type": "noise", "mode": "mono", "color": "#000000", "size": 1, "density": 0.5},
            {"id": "t", "type": "texture", "size": 4, "radius": 3}
        ])),
        ("layer filters removed", json!([])),
        ("backdrop added", json!([{"id": "g", "type": "background-blur", "radius": 8}])),
        ("backdrop hidden", json!([{"id": "g", "type": "background-blur", "radius": 8, "visible": false}])),
        ("backdrop removed", json!([])),
    ];
    for (label, effects) in steps {
        n["effects"] = effects;
        Edit {
            upserts: vec![n.clone()],
            ..Edit::default()
        }
        .apply(&mut doc, &mut resident)
        .unwrap();
        check(&resident, &mut caches, &doc, label);
    }
    // Adding the first shadow adds a plan item (a shape change); editing it does not.
    n["effects"] = json!([{"id": "d", "type": "drop-shadow", "color": "#000000", "blur": 4, "offset": {"x": 3, "y": 3}}]);
    Edit {
        upserts: vec![n.clone()],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "drop shadow added");
    n["effects"] = json!([{"id": "d", "type": "drop-shadow", "color": "#ff0000", "blur": 9, "offset": {"x": 6, "y": 2}}]);
    let hits = caches.incr_patch_hits;
    Edit {
        upserts: vec![n],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "drop shadow edited");
    assert!(
        caches.incr_patch_hits > hits,
        "a drop shadow replace must patch the plan incrementally"
    );
}

/// An edit that touches one node keeps every other node's cache entries: the
/// re-plan after adding one stroked rect tessellates that rect only.
#[test]
fn an_add_re_tessellates_only_the_new_node() {
    let mut doc = Doc::new(1, 30);
    let (mut resident, mut caches) = start(&doc);
    let mut ids = doc.page_ids(0);
    ids.push("new1".into());
    Edit {
        upserts: vec![rect("new1", 5.0, 5.0, true)],
        pages: vec![(0, ids)],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    let misses = caches.tess_misses;
    caches.begin_frame();
    build_plan_retained(&resident, &mut caches, 0, &opts()).unwrap();
    assert_eq!(
        caches.tess_misses - misses,
        1,
        "only the added node may miss the tessellation cache"
    );
    check(&resident, &mut caches, &doc, "add");
}

/// Every layer of the document moved in one patch (a held arrow key on a select-all): each repeat is a patch of
/// the whole document, and the resident scene and its plans match a full rebuild after each.
#[test]
fn moving_every_layer_patches_and_matches_full_rebuild() {
    let mut doc = Doc::new(1, 400);
    let (mut resident, mut caches) = start(&doc);
    for step in 1..=3 {
        let upserts: Vec<Value> = doc.0["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| {
                let mut n = n.clone();
                n["x"] = json!(n["x"].as_f64().unwrap() + 1.0);
                n
            })
            .collect();
        Edit {
            upserts,
            ..Edit::default()
        }
        .apply(&mut doc, &mut resident)
        .unwrap();
        check(&resident, &mut caches, &doc, &format!("move all, repeat {step}"));
    }
}

#[test]
fn malformed_edits_are_refused_before_mutating() {
    let doc = Doc::new(1, 4);
    let (mut resident, _caches) = start(&doc);
    for bad in [
        json!({"upserts": [{"type": "rectangle"}]}),
        json!({"upserts": {"id": "x"}}),
        // The upserts are moved out of the patch as they parse: a bad one after a good one still mutates nothing.
        json!({"upserts": [rect("p0r1", 5.0, 5.0, false), 7]}),
        json!({"upserts": [rect("p0r1", 5.0, 5.0, false), {"type": "rectangle"}]}),
        json!({"removes": [1]}),
        json!({"pageNodeIds": {"page-0": "p0r1"}}),
        // An unknown page is refused by the scene before it mutates anything.
        json!({"upserts": [rect("zz", 0.0, 0.0, false)], "pageNodeIds": {"nope": ["zz"]}}),
    ] {
        assert!(resident.apply_props(&bad.to_string()).is_err(), "{bad}");
        assert_eq!(
            resident.divergence_from_full_load(&doc.json()),
            None,
            "{bad} mutated the scene"
        );
    }
}

/// A long random edit sequence, checked against a full rebuild after every step.
#[test]
fn random_edit_sequences_match_full_rebuild() {
    let mut doc = Doc::new(2, 10);
    let (mut resident, mut caches) = start(&doc);
    let mut seed: u64 = 0x5eed;
    let mut rnd = move |n: usize| {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as usize) % n.max(1)
    };
    let mut next_id = 0;
    let mut applied = 0;
    for step in 0..200 {
        let p = rnd(2);
        let mut ids = doc.page_ids(p);
        let top_rects: Vec<String> = ids
            .iter()
            .filter(|i| doc.node(i).is_some_and(|n| n["type"] == "rectangle"))
            .cloned()
            .collect();
        let groups: Vec<String> = ids
            .iter()
            .filter(|i| doc.node(i).is_some_and(|n| n["type"] == "group"))
            .cloned()
            .collect();
        let mut e = Edit::default();
        let label = match rnd(7) {
            0 => {
                next_id += 1;
                let id = format!("n{next_id}");
                let at = rnd(ids.len() + 1);
                ids.insert(at, id.clone());
                e.upserts
                    .push(rect(&id, rnd(700) as f64, rnd(500) as f64, rnd(2) == 0));
                e.pages.push((p, ids));
                "add"
            }
            1 if !top_rects.is_empty() => {
                let id = top_rects[rnd(top_rects.len())].clone();
                ids.retain(|i| *i != id);
                e.removes.push(id);
                e.pages.push((p, ids));
                "remove"
            }
            2 if ids.len() > 1 => {
                let (a, b) = (rnd(ids.len()), rnd(ids.len()));
                ids.swap(a, b);
                e.pages.push((p, ids));
                "reorder"
            }
            3 if !top_rects.is_empty() => {
                let mut n = doc.node(&top_rects[rnd(top_rects.len())]).unwrap();
                n["x"] = json!(rnd(700) as f64);
                n["fills"] = fill(["#ff0000", "#00ff00", "#0000ff"][rnd(3)]);
                e.upserts.push(n);
                "restyle"
            }
            4 if top_rects.len() >= 2 => {
                next_id += 1;
                let gid = format!("g{next_id}");
                let (a, b) = (top_rects[0].clone(), top_rects[1].clone());
                for c in [&a, &b] {
                    let mut n = doc.node(c).unwrap();
                    n["parentId"] = json!(gid);
                    e.upserts.push(n);
                }
                let at = ids.iter().position(|i| *i == a).unwrap();
                ids.retain(|i| *i != a && *i != b);
                ids.insert(at.min(ids.len()), gid.clone());
                e.upserts.push(
                    json!({"id": gid, "type": "group", "x": 0.0, "y": 0.0, "rotation": 0,
                    "scale": 1, "children": [a, b]}),
                );
                e.pages.push((p, ids));
                "group"
            }
            5 if !groups.is_empty() => {
                let gid = groups[rnd(groups.len())].clone();
                let children: Vec<String> =
                    moda_scene::value::from_value(doc.node(&gid).unwrap()["children"].clone()).unwrap();
                let at = ids.iter().position(|i| *i == gid).unwrap();
                ids.remove(at);
                for (k, c) in children.iter().enumerate() {
                    let mut n = doc.node(c).unwrap();
                    n.as_object_mut().unwrap().remove("parentId");
                    e.upserts.push(n);
                    ids.insert(at + k, c.clone());
                }
                e.removes.push(gid);
                e.pages.push((p, ids));
                "ungroup"
            }
            6 if !top_rects.is_empty() => {
                let id = top_rects[rnd(top_rects.len())].clone();
                ids.retain(|i| *i != id);
                let mut other = doc.page_ids(1 - p);
                other.push(id);
                e.pages.push((p, ids));
                e.pages.push((1 - p, other));
                "move page"
            }
            _ => continue,
        };
        e.apply(&mut doc, &mut resident).unwrap();
        check(
            &resident,
            &mut caches,
            &doc,
            &format!("step {step} ({label})"),
        );
        applied += 1;
    }
    assert!(applied > 100, "only {applied} edits applied");
}

/// A page-less Design document: its top level is the canvas root, so adding,
/// deleting and reordering top-level layers carries `canvasRootNodeIds` in the
/// patch (instead of a full feed) and must leave the resident scene equal to a
/// cold load, with the same camera plan.
#[test]
fn canvas_root_edits_match_full_load() {
    let ids: Vec<String> = (0..10).map(|i| format!("c{i}")).collect();
    let nodes: Vec<Value> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| rect(id, (i * 50) as f64, 0.0, i % 2 == 0))
        .collect();
    let mut doc = json!({"nodes": nodes, "pages": [], "canvasRootNodeIds": ids, "config": {}, "variables": []});
    let mut resident = ResidentScene::default();
    resident.update(&doc.to_string()).unwrap();
    let cam = crate::frame::CameraParams {
        camera_x: -50.0,
        camera_y: -50.0,
        zoom: 0.5,
        viewport_w: 900.0,
        viewport_h: 900.0,
        dpr: 1.0,
        clear_rgb: [1.0, 1.0, 1.0],
    };
    let mut caches = PlanCaches::default();
    let mut step = |resident: &mut ResidentScene, doc: &Value, patch: Value, label: &str| {
        resident.apply_props(&patch.to_string()).unwrap();
        let json = doc.to_string();
        assert_eq!(resident.divergence_from_full_load(&json), None, "{label}: resident diverged");
        caches.begin_frame();
        let incr = build_camera_plan_retained(resident, &mut caches, &[], &cam, &opts())
            .unwrap()
            .map(|(p, _)| p);
        let mut cold = ResidentScene::default();
        cold.update(&json).unwrap();
        let full = build_camera_plan_retained(&cold, &mut PlanCaches::default(), &[], &cam, &opts())
            .unwrap()
            .map(|(p, _)| p);
        assert_eq!(
            incr.map(|p| plan_items_fingerprint(&p)),
            full.map(|p| plan_items_fingerprint(&p)),
            "{label}: camera plan diverged from a full rebuild"
        );
    };

    // Add a layer on top.
    let added = rect("new", 30.0, 60.0, true);
    doc["nodes"].as_array_mut().unwrap().push(added.clone());
    let mut roots: Vec<String> = moda_scene::value::from_value(doc["canvasRootNodeIds"].clone()).unwrap();
    roots.push("new".into());
    doc["canvasRootNodeIds"] = json!(roots);
    step(&mut resident, &doc, json!({"upserts": [added], "canvasRootNodeIds": roots}), "add");

    // Delete two.
    doc["nodes"]
        .as_array_mut()
        .unwrap()
        .retain(|n| n["id"] != "c2" && n["id"] != "c7");
    roots.retain(|i| i != "c2" && i != "c7");
    doc["canvasRootNodeIds"] = json!(roots);
    step(&mut resident, &doc, json!({"removes": ["c2", "c7"], "canvasRootNodeIds": roots}), "delete");

    // Send the new one to the back.
    roots.retain(|i| i != "new");
    roots.insert(0, "new".into());
    doc["canvasRootNodeIds"] = json!(roots);
    step(&mut resident, &doc, json!({"canvasRootNodeIds": roots}), "reorder");

    // Malformed: refused.
    assert!(resident.apply_props(&json!({"canvasRootNodeIds": "c1"}).to_string()).is_err());
    assert!(resident.apply_props(&json!({"canvasRootNodeIds": [1]}).to_string()).is_err());
}

/// A new, empty Design document: its retained camera plan has no segment. The
/// first layer drawn, then fill edits on it, must reach the plan, and each plan
/// must equal a cold build (the app drew nothing until a rebuild).
#[test]
fn first_layer_on_an_empty_canvas_matches_full_load() {
    let mut doc = json!({"nodes": [], "pages": [], "canvasRootNodeIds": [], "config": {},
        "layoutMode": "freeform", "variables": []});
    let mut resident = ResidentScene::default();
    resident.update(&doc.to_string()).unwrap();
    let cam = crate::frame::CameraParams {
        camera_x: -50.0,
        camera_y: -50.0,
        zoom: 1.0,
        viewport_w: 900.0,
        viewport_h: 900.0,
        dpr: 1.0,
        clear_rgb: [1.0, 1.0, 1.0],
    };
    let mut caches = PlanCaches::default();
    let mut frame = |resident: &ResidentScene, doc: &Value, label: &str| {
        let json = doc.to_string();
        assert_eq!(resident.divergence_from_full_load(&json), None, "{label}: resident diverged");
        caches.begin_frame();
        let incr = build_camera_plan_retained(resident, &mut caches, &[], &cam, &opts())
            .unwrap()
            .map(|(p, _)| p);
        let mut cold = ResidentScene::default();
        cold.update(&json).unwrap();
        let full = build_camera_plan_retained(&cold, &mut PlanCaches::default(), &[], &cam, &opts())
            .unwrap()
            .map(|(p, _)| p);
        assert_eq!(
            incr.map(|p| plan_items_fingerprint(&p)),
            full.map(|p| plan_items_fingerprint(&p)),
            "{label}: camera plan diverged from a full rebuild"
        );
    };
    frame(&resident, &doc, "empty");
    frame(&resident, &doc, "empty again");

    // Draw the first rectangle (the rect tool's sync: an upsert and the new root list).
    let mut first = rect("first", 100.0, 100.0, false);
    first["cornerRadius"] = json!(0);
    first["fills"] = fill("#cccccc");
    doc["nodes"] = json!([first.clone()]);
    doc["canvasRootNodeIds"] = json!(["first"]);
    resident
        .apply_props(&json!({"upserts": [first], "canvasRootNodeIds": ["first"]}).to_string())
        .unwrap();
    frame(&resident, &doc, "first layer");

    // Recolour its fill, twice.
    for color in ["#ff0000", "#00ff00"] {
        doc["nodes"][0]["fills"] = fill(color);
        resident
            .apply_props(&json!({"nodes": {"first": {"fills": fill(color)}}}).to_string())
            .unwrap();
        frame(&resident, &doc, &format!("fill {color}"));
    }

    // Layers that drew nothing when the plan was built: shown, and given a fill.
    let mut hidden = rect("hidden", 300.0, 100.0, false);
    hidden["visible"] = json!(false);
    let mut bare = rect("bare", 500.0, 100.0, false);
    bare["fills"] = json!([]);
    doc["nodes"].as_array_mut().unwrap().extend([hidden.clone(), bare.clone()]);
    doc["canvasRootNodeIds"] = json!(["first", "hidden", "bare"]);
    resident
        .apply_props(
            &json!({"upserts": [hidden, bare], "canvasRootNodeIds": ["first", "hidden", "bare"]}).to_string(),
        )
        .unwrap();
    frame(&resident, &doc, "hidden and bare layers");
    doc["nodes"][1]["visible"] = json!(true);
    resident
        .apply_props(&json!({"nodes": {"hidden": {"visible": true}}}).to_string())
        .unwrap();
    frame(&resident, &doc, "shown");
    doc["nodes"][2]["fills"] = fill("#0000ff");
    resident
        .apply_props(&json!({"nodes": {"bare": {"fills": fill("#0000ff")}}}).to_string())
        .unwrap();
    frame(&resident, &doc, "filled");

    // Delete every layer, then draw again.
    doc["nodes"] = json!([]);
    doc["canvasRootNodeIds"] = json!([]);
    resident
        .apply_props(&json!({"removes": ["first", "hidden", "bare"], "canvasRootNodeIds": []}).to_string())
        .unwrap();
    frame(&resident, &doc, "emptied");
    let again = rect("again", 40.0, 40.0, false);
    doc["nodes"] = json!([again.clone()]);
    doc["canvasRootNodeIds"] = json!(["again"]);
    resident
        .apply_props(&json!({"upserts": [again], "canvasRootNodeIds": ["again"]}).to_string())
        .unwrap();
    frame(&resident, &doc, "drawn again");
}

/// Illustrator's Appearance (E8.2) through the patch lanes: effects on one fill or stroke, the stack order, and a
/// group's fills and strokes on the merged outline of its children, which follow edits to the group and to
/// anything inside it. Every step equals a cold load and a full rebuild.
#[test]
fn appearance_edits_match_full_rebuild() {
    // A group's merged outline reads rectangles through the engine's operand seam.
    crate::install_scene_seams();
    let mut doc = Doc::new(1, 6);
    let (mut resident, mut caches) = start(&doc);
    // A stroke with its own blur (a capture), then under the fill, then with a roughen of its own.
    let mut n = doc.node("p0r0").unwrap();
    n["strokes"][0]["effects"] = json!([{"id": "b", "type": "layer-blur", "radius": 4}]);
    let steps: Vec<(&str, Value)> = vec![
        ("stroke blur added", n.clone()),
        ("stroke under the fill", {
            let mut m = n.clone();
            m["appearanceOrder"] = json!(["stroke-0", "fill-0"]);
            m
        }),
        ("stroke roughened too", {
            let mut m = n.clone();
            m["appearanceOrder"] = json!(["stroke-0", "fill-0"]);
            m["strokes"][0]["effects"] = json!([
                {"id": "r", "type": "roughen", "size": 3, "detail": 10, "points": "corner"},
                {"id": "b", "type": "layer-blur", "radius": 4}
            ]);
            m
        }),
        ("entry effects removed", doc.node("p0r0").unwrap()),
    ];
    for (label, node) in steps {
        Edit {
            upserts: vec![node],
            ..Edit::default()
        }
        .apply(&mut doc, &mut resident)
        .unwrap();
        check(&resident, &mut caches, &doc, label);
    }
    // Group p0r2 and p0r3, then give the group a stroke under its contents.
    let mut a = doc.node("p0r2").unwrap();
    let mut b = doc.node("p0r3").unwrap();
    a["parentId"] = json!("g");
    b["parentId"] = json!("g");
    let mut g = json!({"id": "g", "type": "group", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
        "children": ["p0r2", "p0r3"]});
    let mut ids = doc.page_ids(0);
    ids.retain(|i| i != "p0r2" && i != "p0r3");
    ids.push("g".into());
    Edit {
        upserts: vec![g.clone(), a.clone(), b.clone()],
        pages: vec![(0, ids)],
        ..Edit::default()
    }
    .apply(&mut doc, &mut resident)
    .unwrap();
    check(&resident, &mut caches, &doc, "grouped");
    g["strokes"] = json!([{"id": "stroke-0", "paint": {"type": "solid", "color": "#ff0066"}, "width": 6}]);
    g["fills"] = json!([{"id": "fill-0", "paint": {"type": "solid", "color": "#ffee00"}}]);
    g["appearanceOrder"] = json!(["stroke-0", "fill-0", "contents"]);
    for (label, upserts) in [
        ("group paint added", vec![g.clone()]),
        ("a child moved", {
            a["x"] = json!(a["x"].as_f64().unwrap() + 25.0);
            vec![a.clone()]
        }),
        ("the group stroke glows", {
            g["strokes"][0]["effects"] = json!([{"id": "gl", "type": "glow", "color": "#ff0066", "radiusPx": 8, "intensity": 1, "opacity": 1}]);
            vec![g.clone()]
        }),
        ("the stroke over the contents", {
            g["appearanceOrder"] = json!(["fill-0", "contents", "stroke-0"]);
            vec![g.clone()]
        }),
        ("the group moved", {
            g["x"] = json!(30.0);
            vec![g.clone()]
        }),
        ("group paint removed", {
            let o = g.as_object_mut().unwrap();
            o.remove("fills");
            o.remove("strokes");
            o.remove("appearanceOrder");
            vec![g.clone()]
        }),
    ] {
        Edit {
            upserts,
            ..Edit::default()
        }
        .apply(&mut doc, &mut resident)
        .unwrap();
        check(&resident, &mut caches, &doc, label);
    }
}

/// The property lane refuses an edit inside (or to) a group that paints its appearance: the group's outline is
/// derived from its descendants, so the host heals with a full feed.
#[test]
fn property_patches_inside_a_painting_group_resync() {
    let doc = json!({"nodes": [
            {"id": "g", "type": "group", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
             "children": ["a"], "strokes": [{"id": "s", "paint": {"type": "solid", "color": "#000"}, "width": 4}]},
            {"id": "a", "type": "rectangle", "parentId": "g", "x": 10, "y": 10, "width": 40, "height": 30,
             "rotation": 0, "scale": 1, "fills": fill("#3a86ff")},
            {"id": "c", "type": "rectangle", "x": 200, "y": 10, "width": 40, "height": 30,
             "rotation": 0, "scale": 1, "fills": fill("#3a86ff")}
        ],
        "pages": [{"id": "page-0", "name": "p", "width": 800.0, "height": 600.0, "fills": fill("#ffffff"),
                   "nodeIds": ["g", "c"]}],
        "config": {"renderPageClipping": true}, "canvasRootNodeIds": [], "layoutMode": "vertical", "variables": []});
    let mut resident = ResidentScene::default();
    resident.update(&doc.to_string()).unwrap();
    for (id, props) in [("a", json!({"x": 30})), ("g", json!({"x": 5})), ("g", json!({"opacity": 0.5}))] {
        let err = resident
            .apply_props(&json!({"nodes": {id: props}}).to_string())
            .expect_err("a painting group's subtree re-feeds");
        assert!(err.to_string().contains("group's appearance"), "{err}");
    }
    // A node outside it patches as before.
    resident
        .apply_props(&json!({"nodes": {"c": {"x": 220}}}).to_string())
        .unwrap();
}

/// A flipbook (motion-model.md §12.10): the playhead patches `flipbook` on the group (the overlay's `current`), and
/// the resident scene then draws the drawing a cold load of the patched document draws. An onion-skinned flipbook
/// is copied from its drawings when the scene is made, so edits to it or inside it, and turning the skin on, resync.
#[test]
fn flipbook_drawing_patches_match_a_full_load_and_onion_skins_resync() {
    let doc_with = |current: u64, onion: Option<Value>| {
        let mut fb = json!({"id": "fb", "type": "group", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
                            "flipbook": {"mode": "loop", "current": current}, "children": ["d0", "d1"]});
        if let Some(o) = onion {
            fb["onionSkin"] = o;
        }
        let mut d0 = rect("d0", 10.0, 10.0, true);
        d0["parentId"] = json!("fb");
        let mut d1 = rect("d1", 100.0, 10.0, false);
        d1["parentId"] = json!("fb");
        Doc(json!({"nodes": [fb, d0, d1],
            "pages": [{"id": "page-0", "name": "p", "width": 800.0, "height": 600.0, "fills": fill("#ffffff"),
                       "nodeIds": ["fb"]}],
            "config": {"renderPageClipping": true}, "canvasRootNodeIds": [], "layoutMode": "vertical", "variables": []}))
    };
    let doc = doc_with(0, None);
    let (mut resident, mut caches) = start(&doc);
    resident
        .apply_props(&json!({"nodes": {"fb": {"flipbook": {"mode": "loop", "current": 1}}}}).to_string())
        .unwrap();
    check(&resident, &mut caches, &doc_with(1, None), "current 0 → 1");
    let z = resident.scene().unwrap().flattened_z_order_for_ids(&strings(&["fb"]));
    assert_eq!(z, strings(&["fb", "d1"]));

    // Turning the onion skin on, and any edit under an onion-skinned flipbook, re-feeds.
    let err = resident
        .apply_props(&json!({"nodes": {"fb": {"onionSkin": {"before": 1, "after": 1}}}}).to_string())
        .expect_err("the skin is made with the scene");
    assert!(err.to_string().contains("onion"), "{err}");
    let onion = doc_with(1, Some(json!({"before": 1, "after": 0})));
    let (mut resident, _caches) = start(&onion);
    for (id, props) in [("fb", json!({"flipbook": {"mode": "loop", "current": 0}})), ("d0", json!({"x": 30}))] {
        let err = resident
            .apply_props(&json!({"nodes": {id: props}}).to_string())
            .expect_err("an onion-skinned flipbook re-feeds");
        assert!(err.to_string().contains("onion"), "{err}");
    }
}
