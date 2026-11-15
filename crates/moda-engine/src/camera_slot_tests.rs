//! Fuzz oracle for the retained camera plan (`camera_slot`): random edit
//! sequences — property edits through both patch lanes, adds, deletes,
//! reorders, reparents, hides, blend-mode and effect edits — interleaved with
//! camera moves, each frame compared with a from-scratch build
//! ([`super::plan_divergence`]). Fixed seeds; thousands of operations.

use super::*;
use crate::frame::{CameraParams, RenderOptions};
use crate::residency::{build_camera_plan_retained, PlanCaches, ResidentScene};
use moda_scene::value::{json, Map, Value};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn f(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (self.next() % 10_000) as f64 / 10_000.0 * (hi - lo)
    }
    fn chance(&mut self, p: f64) -> bool {
        self.f(0.0, 1.0) < p
    }
}

fn solid(c: &str) -> Value {
    json!([{ "id": "fill-0", "paint": { "type": "solid", "color": c } }])
}

const COLORS: [&str; 6] = [
    "#3a86ff", "#fb5607", "#8338ec", "#06d6a0", "#ef476f", "#ffbe0b",
];

fn html(t: &str) -> String {
    format!("<p><span style=\"font-size: 12px; font-family: Inter; color: #111111\">{t}</span></p>")
}

/// The document the fuzzer edits: nodes by id, the top-level lists. Build one with `..Default::default()` for the
/// fields a test doesn't set, so a field added by one PR doesn't break another PR's test that builds a `Doc`.
struct Doc {
    nodes: Map,
    /// Canvas-root list (freeform) and each page's list.
    roots: Vec<String>,
    pages: Vec<(String, Vec<String>)>,
    layout: &'static str,
    next_id: usize,
    /// Blend objects and mask nodes (`with_dependents`), and how often an edit
    /// picks one of them.
    sources: Vec<String>,
    source_bias: f64,
    /// The chance a frame's shader clock advances (0: the editor's still clock).
    clock_moves: f64,
    /// Whether leaves are made clipped to the layer below and edits toggle it (off: the RNG stream is the one from
    /// before clip-to-below existed, which tests pinned to a scenario keep).
    clip_below: bool,
    /// Plan as the live editor does with motion captures on (`crate::motion_capture`): the camera lane's motion
    /// plans, and motion captures requested on the frames whose camera moved.
    motion: bool,
}

impl Default for Doc {
    fn default() -> Doc {
        Doc {
            nodes: Map::new(),
            roots: Vec::new(),
            pages: Vec::new(),
            layout: "freeform",
            next_id: 0,
            sources: Vec::new(),
            source_bias: 0.0,
            clock_moves: 0.0,
            clip_below: true,
            motion: false,
        }
    }
}

impl Doc {
    fn to_json(&self) -> String {
        let pages: Vec<Value> = self
            .pages
            .iter()
            .enumerate()
            .map(|(i, (id, ids))| {
                json!({ "id": id, "name": id, "width": 1200, "height": 900,
                        "fills": solid("#f4f4f5"), "nodeIds": ids,
                        "freeformX": i as f64 * 1400.0, "freeformY": 0 })
            })
            .collect();
        json!({
            "nodes": self.nodes.values().cloned().collect::<Vec<_>>(),
            "pages": pages,
            "canvasRootNodeIds": self.roots,
            "layoutMode": self.layout,
            "variables": [], "brushes": [], "config": {},
            // E8.5: the pattern the patterned leaves paint with (brick tiles with a dot).
            "patterns": [{ "id": "bricks", "name": "Bricks",
                "tile": { "type": "brick-row", "width": 14, "height": 8, "brickOffset": 0.5 },
                "artwork": { "nodes": [
                    { "id": "b", "type": "rectangle", "x": 1, "y": 1, "width": 12, "height": 6,
                      "fills": [{ "id": "f", "paint": { "type": "solid", "color": "#c0392b" } }] },
                    { "id": "d", "type": "ellipse", "x": 5, "y": 2, "width": 4, "height": 4,
                      "fills": [{ "id": "f", "paint": { "type": "solid", "color": "#2d3436" } }] }] } }],
        })
        .to_string()
    }

    fn fresh_id(&mut self) -> String {
        self.next_id += 1;
        format!("n{}", self.next_id)
    }

    /// A random leaf at local (x, y).
    fn leaf(&mut self, rng: &mut Rng, x: f64, y: f64) -> Value {
        let id = self.fresh_id();
        let c = COLORS[rng.below(COLORS.len())];
        let (w, h) = (rng.f(10.0, 80.0), rng.f(10.0, 60.0));
        let mut n = match rng.below(7) {
            0 => {
                json!({ "type": "ellipse", "fills": solid(c), "strokes": [{ "id": "stroke-0", "paint": { "type": "solid", "color": "#111111" }, "width": 1.5 }] })
            }
            1 => json!({ "type": "richtext", "htmlContent": html(&format!("t{}", self.next_id)) }),
            2 => {
                json!({ "type": "star", "numPoints": 5, "innerRadiusRatio": 0.45, "fills": solid(c),
                         "effects": [{ "id": "background-blur-0", "type": "background-blur", "radius": 8 }] })
            }
            3 => json!({ "type": "ellipse", "fills": solid(c),
                         "effects": [{ "id": "layer-blur-0", "type": "layer-blur", "radius": 4 }] }),
            4 => json!({ "type": "rectangle", "fills": solid(c), "cornerRadius": 4,
                         "effects": [{ "id": "drop-shadow-0", "type": "drop-shadow", "color": "#000000", "blur": 8, "offset": { "x": 0, "y": 4 }, "opacity": 0.2 }] }),
            5 => json!({ "type": "path", "closed": true, "fills": solid(c),
                         "points": [{ "type": "move", "x": 8, "y": 0 }, { "type": "line", "x": 16, "y": 16 }, { "type": "line", "x": 0, "y": 16 }] }),
            // A raster paint layer (spike): one pattern item over its surface texture.
            _ if rng.chance(0.25) => json!({ "type": "paint", "surface": { "manifest": "", "density": 2,
                         "tileSize": 256, "format": "rgba8-srgb" } }),
            // An image fill (a pattern body: its pattern instance bakes the depth too).
            _ if rng.chance(0.25) => json!({ "type": "rectangle", "fills": [{ "id": "fill-0", "paint": { "type": "image",
                         "src": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==" } }] }),
            // A vector pattern fill over a solid, and a pattern stroke (E8.5): tile copies under a clip mask.
            _ if rng.chance(0.3) => json!({ "type": "rectangle", "fills": [
                    { "id": "fill-0", "paint": { "type": "solid", "color": c } },
                    { "id": "fill-1", "paint": { "type": "pattern", "patternId": "bricks",
                      "transform": { "rotation": rng.f(-30.0, 30.0), "scale": rng.f(0.8, 1.6) } } }],
                "strokes": [{ "id": "stroke-0", "width": 3, "paint": { "type": "pattern", "patternId": "bricks" } }] }),
            _ => json!({ "type": "rectangle", "fills": solid(c) }),
        };
        let o = n.as_object_mut().expect("object");
        for (k, v) in [
            ("id", json!(id)),
            ("name", json!(id)),
            ("x", json!(x)),
            ("y", json!(y)),
            ("width", json!(w)),
            ("height", json!(h)),
            ("rotation", json!(0)),
            ("scale", json!(1)),
        ] {
            o.insert(k.into(), v);
        }
        if rng.chance(0.1) {
            o.insert(
                "blendMode".into(),
                json!(["multiply", "overlay", "screen"][rng.below(3)]),
            );
        }
        // Clip to the layer below (its base is whatever sibling ends up under it).
        if self.clip_below && rng.chance(0.15) {
            o.insert("clipToBelow".into(), json!(true));
        }
        n
    }

    /// A container (clipping or not) or group with a few leaves; returns the
    /// parent id. Every node is added to `nodes`.
    fn subtree(&mut self, rng: &mut Rng, x: f64, y: f64, depth: usize) -> String {
        let id = self.fresh_id();
        let group = rng.chance(0.4);
        let mut kids = Vec::new();
        for k in 0..(2 + rng.below(4)) {
            let (cx, cy) = (10.0 + k as f64 * 30.0, 10.0 + rng.f(0.0, 40.0));
            let kid = if depth > 0 && rng.chance(0.25) {
                self.subtree(rng, cx, cy, depth - 1)
            } else {
                let n = self.leaf(rng, cx, cy);
                let kid = n["id"].as_str().expect("id").to_string();
                self.nodes.insert(kid.clone().into(), n);
                kid
            };
            kids.push(json!(kid));
        }
        let node = if group {
            json!({ "id": id, "name": id, "type": "group", "x": x, "y": y, "rotation": 0,
                    "scaleX": 1, "scaleY": 1, "children": kids })
        } else {
            json!({ "id": id, "name": id, "type": "container", "x": x, "y": y, "rotation": 0,
                    "scale": 1, "width": 200, "height": 120, "children": kids,
                    "clipContent": rng.chance(0.6), "cornerRadius": 12,
                    "fills": if rng.chance(0.7) { solid("#ffffff") } else { json!([]) } })
        };
        self.nodes.insert(id.clone().into(), node);
        id
    }

    /// A blendable shape (a blend's object, a clip mask) at local (x, y); added to `nodes`.
    fn shape(&mut self, rng: &mut Rng, x: f64, y: f64) -> String {
        let id = self.fresh_id();
        let c = COLORS[rng.below(COLORS.len())];
        let (w, h) = (rng.f(20.0, 120.0), rng.f(20.0, 90.0));
        let mut n = match rng.below(4) {
            0 => json!({ "type": "ellipse", "fills": solid(c) }),
            1 => json!({ "type": "star", "numPoints": 5, "innerRadiusRatio": 0.5, "fills": solid(c) }),
            2 => json!({ "type": "path", "closed": true, "fills": solid(c),
                         "points": [{ "type": "move", "x": 10, "y": 0 }, { "type": "line", "x": 30, "y": 25 }, { "type": "line", "x": 0, "y": 30 }] }),
            _ => json!({ "type": "rectangle", "fills": solid(c), "cornerRadius": rng.f(0.0, 10.0) }),
        };
        let o = n.as_object_mut().expect("object");
        for (k, v) in [
            ("id", json!(id)),
            ("name", json!(id)),
            ("x", json!(x)),
            ("y", json!(y)),
            ("width", json!(w)),
            ("height", json!(h)),
            ("rotation", json!(0)),
            ("scale", json!(1)),
        ] {
            o.insert(k.into(), v);
        }
        if rng.chance(0.3) {
            o.insert(
                "strokes".into(),
                json!([{ "id": "stroke-0", "paint": { "type": "solid", "color": "#111111" }, "width": 2 }]),
            );
        }
        self.nodes.insert(id.clone().into(), n);
        id
    }

    /// A clip-mask group: a mask shape (its bottom child, sometimes an alpha or
    /// luminance mask), then content — leaves, a clipping container, sometimes
    /// another mask group. Returns the group id.
    fn mask_group(&mut self, rng: &mut Rng, x: f64, y: f64, depth: usize) -> String {
        let (mx, my) = (rng.f(0.0, 30.0), rng.f(0.0, 30.0));
        let mask = self.shape(rng, mx, my);
        if rng.chance(0.3) {
            let t = ["vector", "alpha", "luminance"][rng.below(3)];
            self.nodes.get_mut(&mask).unwrap().as_object_mut().unwrap().insert("maskType".into(), json!(t));
        }
        self.sources.push(mask.clone());
        let mut kids = vec![json!(mask)];
        for k in 0..(1 + rng.below(3)) {
            let (cx, cy) = (k as f64 * 25.0, rng.f(0.0, 40.0));
            let kid = if depth > 0 && rng.chance(0.2) {
                self.mask_group(rng, cx, cy, depth - 1)
            } else if rng.chance(0.2) {
                self.subtree(rng, cx, cy, 0)
            } else {
                let n = self.leaf(rng, cx, cy);
                let kid = n["id"].as_str().expect("id").to_string();
                self.nodes.insert(kid.clone().into(), n);
                kid
            };
            kids.push(json!(kid));
        }
        let id = self.fresh_id();
        self.nodes.insert(
            id.clone().into(),
            json!({ "id": id, "name": id, "type": "group", "x": x, "y": y, "rotation": 0,
                    "scaleX": 1, "scaleY": 1, "children": kids, "clipMaskNodeId": mask }),
        );
        id
    }

    /// A blend of two or three objects (one sometimes inside a group), above
    /// them. Returns the top-level ids, bottom first.
    fn blend(&mut self, rng: &mut Rng, x: f64, y: f64) -> Vec<String> {
        let mut tops = Vec::new();
        let mut objects = Vec::new();
        for k in 0..(2 + usize::from(rng.chance(0.3))) {
            let (ox, oy) = (x + k as f64 * rng.f(60.0, 200.0), y + rng.f(-60.0, 60.0));
            let obj = self.shape(rng, ox, oy);
            self.sources.push(obj.clone());
            objects.push(obj.clone());
            if rng.chance(0.2) {
                let g = self.fresh_id();
                self.nodes.insert(
                    g.clone().into(),
                    json!({ "id": g, "name": g, "type": "group", "x": 0, "y": 0, "rotation": 0,
                            "scaleX": 1, "scaleY": 1, "children": [obj] }),
                );
                tops.push(g);
            } else {
                tops.push(obj);
            }
        }
        let id = self.fresh_id();
        let spacing = if rng.chance(0.2) { "smooth" } else { "steps" };
        self.nodes.insert(
            id.clone().into(),
            json!({ "id": id, "name": id, "type": "blend", "sourceIds": objects,
                    "spacing": spacing, "steps": 1 + rng.below(5) }),
        );
        tops.push(id);
        tops
    }

    fn parent_of(&self, id: &str) -> Option<String> {
        self.nodes.iter().find_map(|(pid, n)| {
            n.get("children")
                .and_then(Value::as_array)
                .is_some_and(|c| c.iter().any(|k| k.as_str() == Some(id)))
                .then(|| pid.to_string())
        })
    }

    fn subtree_ids(&self, id: &str, out: &mut Vec<String>) {
        out.push(id.to_string());
        if let Some(kids) = self
            .nodes
            .get(id)
            .and_then(|n| n.get("children"))
            .and_then(Value::as_array)
        {
            for k in kids.iter().filter_map(Value::as_str) {
                self.subtree_ids(k, out);
            }
        }
    }

    /// The list holding top-level `id`: the canvas root (None) or a page index.
    fn list_of(&self, id: &str) -> Option<Option<usize>> {
        if self.roots.iter().any(|r| r == id) {
            return Some(None);
        }
        self.pages
            .iter()
            .position(|(_, ids)| ids.iter().any(|r| r == id))
            .map(Some)
    }

    fn list_mut(&mut self, which: Option<usize>) -> &mut Vec<String> {
        match which {
            None => &mut self.roots,
            Some(i) => &mut self.pages[i].1,
        }
    }

    fn list_patch(&self, which: Option<usize>, patch: &mut Map) {
        match which {
            None => {
                patch.insert("canvasRootNodeIds".into(), json!(self.roots));
            }
            Some(i) => {
                let entry = patch
                    .entry("pageNodeIds")
                    .or_insert_with(|| Value::Object(Map::new()));
                entry
                    .as_object_mut()
                    .expect("object")
                    .insert(self.pages[i].0.clone().into(), json!(self.pages[i].1));
            }
        }
    }
}

fn editor_opts() -> RenderOptions {
    RenderOptions {
        preserve_page_fill: false,
        scale: 2.0,
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

fn cam(x: f64, y: f64, zoom: f64) -> CameraParams {
    CameraParams {
        camera_x: x,
        camera_y: y,
        zoom,
        viewport_w: 800.0,
        viewport_h: 600.0,
        dpr: 2.0,
        clear_rgb: [0.93, 0.93, 0.93],
    }
}

/// One random edit as a patch (applied to `doc` too). `None` = no edit this
/// step (a camera-only frame).
fn random_edit(doc: &mut Doc, rng: &mut Rng) -> Option<Value> {
    let ids: Vec<String> = doc.nodes.keys().map(|k| k.to_string()).collect();
    if ids.is_empty() {
        return None;
    }
    // A blend object or a mask (or, as often, its parent: a leaf inside a source
    // group, a group moved with a source inside).
    let live: Vec<String> = if doc.source_bias > 0.0 {
        doc.sources.iter().filter(|s| doc.nodes.contains_key(*s)).cloned().collect()
    } else {
        Vec::new()
    };
    let id = if !live.is_empty() && rng.chance(doc.source_bias) {
        let s = live[rng.below(live.len())].clone();
        match doc.parent_of(&s) {
            Some(p) if rng.chance(0.25) => p,
            _ => {
                // Geometry the other edits don't reach: an empty box (a mask that
                // builds no mask), a rotation.
                if rng.chance(0.05) {
                    let o = doc.nodes.get_mut(&s).expect("node").as_object_mut().expect("o");
                    o.insert("width".into(), json!(0.0));
                    return Some(json!({ "nodes": { s.clone(): { "width": 0.0 } } }));
                }
                if rng.chance(0.2) {
                    let r = rng.f(-40.0, 40.0);
                    let o = doc.nodes.get_mut(&s).expect("node").as_object_mut().expect("o");
                    o.insert("rotation".into(), json!(r));
                    return Some(json!({ "nodes": { s.clone(): { "rotation": r } } }));
                }
                s
            }
        }
    } else {
        ids[rng.below(ids.len())].clone()
    };
    let is_parent = doc.nodes[&id].get("children").is_some();
    let mut patch = Map::new();
    let mut op = rng.below(17);
    if let Ok(only) = std::env::var("FUZZ_OPS") {
        let allowed: Vec<usize> = only.split(',').filter_map(|x| x.parse().ok()).collect();
        if !allowed.is_empty() && !allowed.contains(&op) {
            op = allowed[op % allowed.len()];
        }
    }
    let nodes_lane = |k: &str, v: Value, doc: &mut Doc, id: &str| {
        doc.nodes
            .get_mut(id)
            .expect("node")
            .as_object_mut()
            .expect("o")
            .insert(k.into(), v.clone());
        json!({ "nodes": { id: { k: v } } })
    };
    let upsert = |doc: &Doc, ids: &[&str]| -> Value {
        json!(ids
            .iter()
            .map(|i| doc.nodes[*i].clone())
            .collect::<Vec<_>>())
    };
    // A paint layer: a committed stroke is a manifest edit; a live one is a pixel upload with no edit at all.
    if doc.nodes[&id]["type"] == "paint" && rng.chance(0.5) {
        if rng.chance(0.5) {
            let tile = vec![(rng.next() & 0xff) as u8; crate::surface::TILE_BYTES];
            let (w, h) = (doc.nodes[&id]["width"].as_f64().unwrap_or(1.0), doc.nodes[&id]["height"].as_f64().unwrap_or(1.0));
            let (wp, hp) = moda_scene::normalize::paint_surface_px(w, h, 2.0);
            crate::surface::upload_tiles(&id, wp, hp, &[(0, 0, Some(&tile))]);
            return None;
        }
        let manifest = format!("{:064x}", rng.next());
        return Some(nodes_lane(
            "surface",
            json!({ "manifest": manifest, "density": 2, "tileSize": 256, "format": "rgba8-srgb" }),
            doc,
            &id,
        ));
    }
    match op {
        // Moves (a group's move reaches its subtree), both lanes.
        0 | 1 | 2 => {
            let x = doc.nodes[&id]["x"].as_f64().unwrap_or(0.0) + rng.f(-40.0, 40.0);
            if op == 2 {
                doc.nodes
                    .get_mut(&id)
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .insert("x".into(), json!(x));
                patch.insert("upserts".into(), upsert(doc, &[&id]));
            } else {
                return Some(nodes_lane("x", json!(x), doc, &id));
            }
        }
        3 if !is_parent => {
            let c = COLORS[rng.below(COLORS.len())];
            return Some(nodes_lane("fills", solid(c), doc, &id));
        }
        4 if !is_parent => {
            let w = rng.f(8.0, 90.0);
            return Some(nodes_lane("width", json!(w), doc, &id));
        }
        5 if doc.nodes[&id]["type"] == "richtext" => {
            let t = format!("edited {}", rng.below(1000));
            return Some(nodes_lane("htmlContent", json!(html(&t)), doc, &id));
        }
        6 => {
            let o = rng.f(0.2, 1.0);
            return Some(nodes_lane("opacity", json!(o), doc, &id));
        }
        // Clip to the layer below on / off: the clipped layers above it change base.
        7 if doc.clip_below && rng.chance(0.4) => {
            let on = !doc.nodes[&id]
                .get("clipToBelow")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            return Some(nodes_lane("clipToBelow", json!(on), doc, &id));
        }
        // Hide / show.
        7 => {
            let hidden = !doc.nodes[&id]
                .get("hidden")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            return Some(nodes_lane("hidden", json!(hidden), doc, &id));
        }
        // Add a leaf or a subtree: top level (anywhere in its list) or into a parent.
        8 | 9 => {
            let (x, y) = (rng.f(0.0, 1100.0), rng.f(0.0, 800.0));
            let new_id = if rng.chance(0.3) {
                doc.subtree(rng, x, y, 1)
            } else {
                let n = doc.leaf(rng, x, y);
                let nid = n["id"].as_str().unwrap().to_string();
                doc.nodes.insert(nid.clone().into(), n);
                nid
            };
            let mut added = Vec::new();
            doc.subtree_ids(&new_id, &mut added);
            let parents: Vec<String> = ids
                .iter()
                .filter(|i| doc.nodes[*i].get("children").is_some())
                .cloned()
                .collect();
            if !parents.is_empty() && rng.chance(0.3) {
                let p = parents[rng.below(parents.len())].clone();
                let kids = doc.nodes.get_mut(&p).unwrap()["children"]
                    .as_array_mut()
                    .unwrap();
                let at = rng.below(kids.len() + 1);
                kids.insert(at, json!(new_id));
                added.push(p);
            } else {
                let which = if doc.pages.is_empty() || (doc.layout == "freeform" && rng.chance(0.5))
                {
                    None
                } else {
                    Some(rng.below(doc.pages.len()))
                };
                let list = doc.list_mut(which);
                let at = if rng.chance(0.5) {
                    list.len()
                } else {
                    rng.below(list.len() + 1)
                };
                list.insert(at, new_id.clone());
                doc.list_patch(which, &mut patch);
            }
            let refs: Vec<&str> = added.iter().map(String::as_str).collect();
            patch.insert("upserts".into(), upsert(doc, &refs));
        }
        // Delete a node (and its subtree).
        10 | 11 => {
            let mut gone = Vec::new();
            doc.subtree_ids(&id, &mut gone);
            match doc.parent_of(&id) {
                Some(p) => {
                    doc.nodes.get_mut(&p).unwrap()["children"]
                        .as_array_mut()
                        .unwrap()
                        .retain(|k| k.as_str() != Some(&id));
                    patch.insert("upserts".into(), upsert(doc, &[&p]));
                }
                None => {
                    let which = doc.list_of(&id)?;
                    doc.list_mut(which).retain(|r| r != &id);
                    doc.list_patch(which, &mut patch);
                }
            }
            for g in &gone {
                doc.nodes.remove(g);
            }
            patch.insert("removes".into(), json!(gone));
        }
        // Reorder a top-level list (a swap or a move), or a parent's children.
        12 | 13 => {
            if let Some(p) = doc.parent_of(&id).filter(|_| rng.chance(0.4)) {
                let kids = doc.nodes.get_mut(&p).unwrap()["children"]
                    .as_array_mut()
                    .unwrap();
                if kids.len() > 1 {
                    let (a, b) = (rng.below(kids.len()), rng.below(kids.len()));
                    kids.swap(a, b);
                }
                patch.insert("upserts".into(), upsert(doc, &[&p]));
            } else {
                let which = doc.list_of(&id).unwrap_or(None);
                let list = doc.list_mut(which);
                if list.len() > 1 {
                    let a = rng.below(list.len());
                    let item = list.remove(a);
                    let b = if rng.chance(0.5) {
                        a.min(list.len())
                    } else {
                        rng.below(list.len() + 1)
                    };
                    list.insert(b, item);
                }
                doc.list_patch(which, &mut patch);
            }
        }
        // Reparent a top-level node into a parent.
        14 => {
            let which = doc.list_of(&id)?;
            let parents: Vec<String> = ids
                .iter()
                .filter(|i| doc.nodes[*i].get("children").is_some() && **i != id)
                .filter(|i| {
                    let mut sub = Vec::new();
                    doc.subtree_ids(&id, &mut sub);
                    !sub.contains(i)
                })
                .cloned()
                .collect();
            if parents.is_empty() {
                return None;
            }
            let p = parents[rng.below(parents.len())].clone();
            doc.list_mut(which).retain(|r| r != &id);
            doc.list_patch(which, &mut patch);
            doc.nodes.get_mut(&p).unwrap()["children"]
                .as_array_mut()
                .unwrap()
                .push(json!(id));
            patch.insert("upserts".into(), upsert(doc, &[&p, &id]));
        }
        // Move a top-level node to another list (page to page, page to the
        // canvas root), at a random position.
        16 => {
            let from = doc.list_of(&id)?;
            let lists: Vec<Option<usize>> = (doc.layout == "freeform")
                .then_some(None)
                .into_iter()
                .chain((0..doc.pages.len()).map(Some))
                .filter(|l| *l != from)
                .collect();
            if lists.is_empty() {
                return None;
            }
            let to = lists[rng.below(lists.len())];
            doc.list_mut(from).retain(|r| r != &id);
            let list = doc.list_mut(to);
            let at = rng.below(list.len() + 1);
            list.insert(at, id.clone());
            doc.list_patch(from, &mut patch);
            doc.list_patch(to, &mut patch);
        }
        // An effect or blend-mode edit.
        _ => {
            if is_parent {
                return None;
            }
            let v = match rng.below(3) {
                0 => json!("normal"),
                1 => json!("multiply"),
                _ => json!("overlay"),
            };
            return Some(nodes_lane("blendMode", v, doc, &id));
        }
    }
    if patch.is_empty() {
        None
    } else {
        Some(Value::Object(patch))
    }
}

thread_local! {
    /// The last `fuzz` run's fallback reasons, with counts.
    static LAST_REASONS: std::cell::RefCell<HashMap<&'static str, u64>> = Default::default();
    /// The refine clock of a `fuzz` run with motion plans (`fuzz_clock`).
    static FUZZ_CLOCK_MS: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    /// The last `fuzz` run's frames planned with motion captures.
    static LAST_MOTION_FRAMES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn fuzz_clock() -> f64 {
    FUZZ_CLOCK_MS.with(|c| c.get())
}

/// Run `steps` random frames over `doc`, checking every served plan.
fn fuzz(mut doc: Doc, seed: u64, steps: usize, visible: &[usize]) -> (u64, u64, u64) {
    let mut rng = Rng(seed);
    let mut resident = ResidentScene::default();
    resident.update(&doc.to_json()).expect("scene");
    let mut caches = PlanCaches::default();
    caches.set_cull_min_nodes(Some(0));
    if doc.motion {
        caches.motion_plans = true;
        // Frames a second apart on the refine clock: only a frame whose zoom changed is mid-zoom.
        caches.set_outline_clock(fuzz_clock);
    }
    let mut opts = editor_opts();
    let (mut cx, mut cy, mut zoom) = (0.0, 0.0, 0.6);
    let mut reasons: HashMap<&'static str, u64> = HashMap::new();
    let mut motion_frames = 0u64;
    for step in 0..steps {
        FUZZ_CLOCK_MS.with(|c| c.set(step as f64 * 1000.0));
        // Camera: mostly still, sometimes a small pan or zoom, rarely a jump.
        let cam_roll = rng.below(10);
        let cam_roll = if std::env::var("FUZZ_NO_CAMERA").is_ok() {
            9
        } else {
            cam_roll
        };
        let before = (cx, cy, zoom);
        match cam_roll {
            0 => {
                cx += rng.f(-60.0, 60.0);
                cy += rng.f(-40.0, 40.0);
            }
            // Down to 0.08: paint leaves are then minified far enough to sample deep mip levels.
            1 => zoom = (zoom * rng.f(0.85, 1.2)).clamp(0.08, 4.0),
            2 if rng.chance(0.2) => {
                cx = rng.f(-600.0, 2400.0);
                cy = rng.f(-400.0, 1200.0);
            }
            _ => {}
        }
        if doc.motion {
            // The host asks for motion captures while the camera moves; a still frame ends the motion.
            caches.set_motion_captures((cx, cy, zoom) != before);
        }
        let edit = random_edit(&mut doc, &mut rng);
        if std::env::var("FUZZ_TRACE").is_ok() {
            eprintln!("step {step}: {:?}", edit.as_ref().map(|e| e.to_string().chars().take(300).collect::<String>()));
        }
        if let Some(p) = &edit {
            if let Err(e) = resident.apply_props(&p.to_string()) {
                // The host heals a refused patch with a full feed.
                let _ = e;
                resident.update(&doc.to_json()).expect("scene");
            }
        }
        // The patched resident scene must be the document (the scene lane's own oracle).
        if step % 50 == 0 {
            if let Some(d) = resident.divergence_from_full_load(&doc.to_json()) {
                panic!("step {step}: resident scene diverged: {d}");
            }
        }
        let fallbacks = caches.incr_fallbacks;
        let (hits0, static0) = (caches.incr_patch_hits, caches.incr_static_hits);
        caches.begin_frame();
        if doc.clock_moves > 0.0 && rng.chance(doc.clock_moves) {
            opts.shader_time_s += 0.25;
        }
        let c = cam(cx, cy, zoom);
        let (plan, _w) = build_camera_plan_retained(&resident, &mut caches, visible, &c, &opts)
            .expect("plan")
            .expect("some");
        if caches.incr_fallbacks != fallbacks {
            *reasons
                .entry(caches.incr_last_fallback.unwrap_or("?"))
                .or_insert(0) += 1;
        }
        let scene = resident.scene().expect("scene");
        let hashes = resident.hashes().expect("hashes");
        let cull = caches.camera_cull;
        let capture = caches.capture_view;
        // A moving frame's reference is built as the frame was: under its motion units, at the clock the motion
        // froze. A still frame's enters nothing (it must match a build with motion captures off).
        let units = caches.motion_units();
        if units.is_some() {
            motion_frames += 1;
        }
        let ref_opts = match caches.motion_clock() {
            Some(t) => RenderOptions { shader_time_s: t, ..opts.clone() },
            None => opts.clone(),
        };
        // A mid-zoom frame whose curves keep a band behind the camera's says it is not exact (`motion_plan`; the
        // host renders again): no from-scratch build reproduces it.
        let divergence = if caches.motion_plan {
            None
        } else {
            let _motion = crate::motion_capture::enter(units);
            plan_divergence(scene, visible, &c, &ref_opts, hashes, cull.as_ref(), capture, &plan)
        };
        if let Some(d) = divergence {
            let lane = if caches.incr_patch_hits != hits0 {
                "patched"
            } else if caches.incr_static_hits != static0 {
                "static"
            } else {
                "rebuilt"
            };
            let first_id = d
                .split("node <")
                .nth(1)
                .and_then(|r| r.split('>').next())
                .unwrap_or("")
                .to_string();
            let where_ = caches
                .camera_slot_ref()
                .map(|s| s.describe(&first_id))
                .unwrap_or_else(|| "no slot".into());
            panic!(
                "step {step} (seed {seed}, {lane}, last fallback {:?}; {where_}; cam {:?}): the served plan diverged after {:?}: {d}",
                caches.incr_last_fallback,
                (cx, cy, zoom),
                edit.map(|e| e.to_string().chars().take(600).collect::<String>())
            );
        }
    }
    eprintln!(
        "fuzz seed {seed}: {} patched, {} static, {} fallbacks, {motion_frames} motion frames {:?}",
        caches.incr_patch_hits, caches.incr_static_hits, caches.incr_fallbacks, reasons
    );
    LAST_REASONS.with(|r| *r.borrow_mut() = reasons);
    LAST_MOTION_FRAMES.with(|m| m.set(motion_frames));
    (
        caches.incr_patch_hits,
        caches.incr_static_hits,
        caches.incr_fallbacks,
    )
}

fn canvas_doc(rng: &mut Rng, top: usize) -> Doc {
    let mut doc = Doc::default();
    for _ in 0..top {
        let (x, y) = (rng.f(0.0, 1400.0), rng.f(0.0, 900.0));
        let id = if rng.chance(0.35) {
            doc.subtree(rng, x, y, 1)
        } else {
            let n = doc.leaf(rng, x, y);
            let id = n["id"].as_str().unwrap().to_string();
            doc.nodes.insert(id.clone().into(), n);
            id
        };
        doc.roots.push(id);
    }
    doc
}

/// Top-level entries for a document with blends and clip masks: mask groups,
/// blends (their objects then the blend node), leaves and subtrees.
fn dep_tops(doc: &mut Doc, rng: &mut Rng, top: usize) -> Vec<String> {
    let mut tops = Vec::new();
    while tops.len() < top {
        let (x, y) = (rng.f(0.0, 1100.0), rng.f(0.0, 800.0));
        match rng.below(10) {
            0..=2 => tops.push(doc.mask_group(rng, x, y, 1)),
            3 | 4 => tops.extend(doc.blend(rng, x, y)),
            5 => tops.push(doc.subtree(rng, x, y, 1)),
            _ => {
                let n = doc.leaf(rng, x, y);
                let id = n["id"].as_str().unwrap().to_string();
                doc.nodes.insert(id.clone().into(), n);
                tops.push(id);
            }
        }
    }
    tops
}

/// A freeform document of blends and clip masks, whose edits pick a blend
/// object or a mask (or its parent) half the time.
fn dep_canvas_doc(rng: &mut Rng, top: usize) -> Doc {
    let mut doc = canvas_doc(rng, 0);
    doc.source_bias = 0.5;
    doc.roots = dep_tops(&mut doc, rng, top);
    doc
}

/// Blend-object and mask edits take the patch lane (the consumer re-plans with
/// them) and match a full rebuild.
#[test]
fn blend_and_mask_source_edits_match_a_full_rebuild() {
    let mut rng = Rng(0x5eed_0b1e);
    let doc = dep_canvas_doc(&mut rng, 30);
    let (patched, _, fallbacks) = fuzz(doc, 0xB1E4D, 1200, &[]);
    assert!(patched > 400, "the slot served {patched} patches");
    assert!(fallbacks < patched / 2, "{fallbacks} fallbacks vs {patched} patches");

    let mut rng = Rng(0x5eed_3a5c);
    let mut doc = canvas_doc(&mut rng, 0);
    doc.layout = "vertical";
    doc.source_bias = 0.5;
    for p in 0..2 {
        let ids = dep_tops(&mut doc, &mut rng, 16);
        doc.pages.push((format!("page-{p}"), ids));
    }
    let (patched, _, _) = fuzz(doc, 0x3A5C, 1000, &[0, 1]);
    assert!(patched > 300, "the slot served {patched} patches");
}

/// A mask's move, a blend object's recolour and a leaf's move inside a
/// clip-mask group each patch (no rebuild) and match a full rebuild.
#[test]
fn a_mask_or_blend_object_edit_patches() {
    let mut rng = Rng(7);
    let mut doc = canvas_doc(&mut rng, 0);
    let g = doc.mask_group(&mut rng, 100.0, 100.0, 0);
    let mask = doc.sources[0].clone();
    let content = doc.nodes[&g]["children"][1].as_str().unwrap().to_string();
    let blend_tops = doc.blend(&mut rng, 400.0, 300.0);
    let object = doc.sources[1].clone();
    doc.roots = vec![g];
    doc.roots.extend(blend_tops);
    let mut resident = ResidentScene::default();
    resident.update(&doc.to_json()).expect("scene");
    let mut caches = PlanCaches::default();
    caches.set_cull_min_nodes(Some(0));
    let opts = editor_opts();
    let c = cam(0.0, 0.0, 0.8);
    let edits = [
        json!({ "nodes": { mask.clone(): { "x": 42.0 } } }),
        json!({ "nodes": { mask.clone(): { "width": 33.0 } } }),
        json!({ "nodes": { content.clone(): { "x": 5.0 } } }),
        json!({ "nodes": { object.clone(): { "fills": solid("#123456") } } }),
        json!({ "nodes": { object.clone(): { "x": 380.0 } } }),
    ];
    for step in 0..=edits.len() {
        if step > 0 {
            resident.apply_props(&edits[step - 1].to_string()).expect("patch");
        }
        let (rebuilds, patches) = (caches.camera_rebuilds, caches.incr_patch_hits);
        caches.begin_frame();
        let (plan, _) = build_camera_plan_retained(&resident, &mut caches, &[], &c, &opts)
            .expect("plan")
            .expect("some");
        if step > 0 {
            assert_eq!(caches.camera_rebuilds, rebuilds, "edit {step} rebuilt: {:?}", caches.incr_last_fallback);
            assert_eq!(caches.incr_patch_hits, patches + 1, "edit {step} patched");
        }
        let scene = resident.scene().expect("scene");
        let hashes = resident.hashes().expect("hashes");
        if let Some(d) = plan_divergence(scene, &[], &c, &opts, hashes, caches.camera_cull.as_ref(), caches.capture_view, &plan) {
            panic!("edit {step}: the served plan diverged: {d}");
        }
    }
}

#[test]
fn rendered_mask_source_edits_match_a_full_rebuild() {
    let document = json!({
        "layoutMode": "freeform",
        "pages": [],
        "canvasRootNodeIds": ["start", "end", "clip", "other"],
        "nodes": [
            {"id": "clip", "type": "group", "children": ["mask", "content"], "clipMaskNodeId": "mask"},
            {"id": "mask", "type": "group", "children": ["line", "blend"], "opacity": 0.5},
            {"id": "line", "type": "line", "points": [0, 20, 100, 20],
                "strokes": [{"id": "stroke", "width": 4, "paint": {"type": "solid", "color": "#000000"}}]},
            {"id": "blend", "type": "blend", "sourceIds": ["start", "end"], "spacing": "steps", "steps": 3},
            {"id": "start", "type": "rectangle", "x": 10, "y": 30, "width": 10, "height": 40, "fills": solid("#000000")},
            {"id": "end", "type": "rectangle", "x": 80, "y": 30, "width": 10, "height": 40, "fills": solid("#000000")},
            {"id": "content", "type": "rectangle", "width": 100, "height": 100, "fills": solid("#ff0000")},
            {"id": "other", "type": "rectangle", "x": 200, "width": 30, "height": 30, "fills": solid("#00ff00")}
        ]
    });
    let mut resident = ResidentScene::default();
    resident.update(&document.to_string()).expect("scene");
    let mut caches = PlanCaches::default();
    caches.set_cull_min_nodes(Some(0));
    let mut opts = editor_opts();
    let camera = cam(0.0, 0.0, 0.8);
    let edits = [
        json!({"nodes": {"other": {"x": 210}}}),
        json!({"nodes": {"line": {"strokes": [{"id": "stroke", "width": 12, "paint": {"type": "solid", "color": "#000000"}}]}}}),
        json!({"nodes": {"start": {"fills": solid("rgba(0,0,0,0.25)")}}}),
        json!({"nodes": {"end": {"x": 70}}}),
        json!({"nodes": {"mask": {"opacity": 0.75}}}),
        json!({"nodes": {"clip": {"x": 15}}}),
    ];
    for step in 0..=edits.len() {
        if step > 0 {
            resident.apply_props(&edits[step - 1].to_string()).expect("patch");
        }
        opts.shader_time_s = step as f64 * 0.25;
        caches.begin_frame();
        let (plan, _) = build_camera_plan_retained(&resident, &mut caches, &[], &camera, &opts)
            .expect("plan").expect("some");
        assert!(plan.masks.iter().any(|mask| mask.mode == 3));
        let divergence = plan_divergence(resident.scene().expect("scene"), &[], &camera, &opts,
            resident.hashes().expect("hashes"), caches.camera_cull.as_ref(), caches.capture_view, &plan);
        assert!(divergence.is_none(), "edit {step}: {divergence:?}");
    }
    assert!(caches.incr_patch_hits > 0);
}

/// A full-size procedural background under a noise effect (art-generative's): its shader body sits in a layer
/// capture, whose clock the present can't re-thread.
fn shader_background(doc: &mut Doc, x: f64, y: f64) -> String {
    let id = doc.fresh_id();
    doc.nodes.insert(id.clone().into(), json!({ "id": id, "name": id, "type": "rectangle", "x": x, "y": y,
        "width": 900, "height": 700, "rotation": 0, "scale": 1,
        "fills": [{ "id": "fill-0", "paint": { "type": "shader", "shaderId": "swirl-shape-gradient",
            "params": { "colors": ["#0d1326", "#15294a", "#3b1f45"], "speed": 0.75, "time": 0 } } }],
        "effects": [{ "type": "noise", "mode": "mono", "color": "#ffffff", "size": 1, "density": 0.5,
            "opacity": 0.1, "id": "noise-0" }] }));
    id
}

/// A plan whose shader clock the present can't re-thread (a procedural fill in a layer capture) is retained while
/// the clock holds still: edits patch it, and every frame matches a full rebuild. When the clock moves it drops,
/// and it is not re-seeded while the clock keeps moving.
#[test]
fn a_captured_shader_background_keeps_the_slot_while_the_clock_holds() {
    let mut rng = Rng(0x5eed_c10c);
    let mut doc = canvas_doc(&mut rng, 0);
    // Its expectations are about the shader background's capture; a clip toggle on it takes it out of one.
    doc.clip_below = false;
    let bg = shader_background(&mut doc, 0.0, 0.0);
    doc.roots.push(bg);
    let tops = dep_tops(&mut doc, &mut rng, 25);
    doc.roots.extend(tops);
    let (patched, _, _) = fuzz(doc, 0xC10C, 600, &[]);
    assert!(patched > 250, "the slot served {patched} patches");
    let reasons = LAST_REASONS.with(|r| r.borrow().clone());
    assert!(
        !reasons.keys().any(|k| k.contains("re-thread")),
        "a still clock seeds the slot: {reasons:?}"
    );

    // A clock that moves on some frames: each move rebuilds, exactly.
    let mut rng = Rng(0x5eed_c10d);
    let mut doc = canvas_doc(&mut rng, 0);
    doc.clip_below = false;
    let bg = shader_background(&mut doc, 100.0, 50.0);
    let (a, b) = (dep_tops(&mut doc, &mut rng, 10), dep_tops(&mut doc, &mut rng, 10));
    doc.roots.extend(a);
    doc.roots.push(bg);
    doc.roots.extend(b);
    doc.clock_moves = 0.2;
    let (patched, _, _) = fuzz(doc, 0xC10D, 500, &[]);
    assert!(patched > 100, "the slot served {patched} patches");
    let reasons = LAST_REASONS.with(|r| r.borrow().clone());
    assert!(reasons.contains_key("shader clock moved"), "{reasons:?}");
}

/// A large procedural fill (a motion-capture unit on its own, `crate::motion_capture::MIN_SHADER_AREA`).
fn big_shader(doc: &mut Doc, rng: &mut Rng, x: f64, y: f64) -> String {
    let id = doc.fresh_id();
    let side = rng.f(300.0, 600.0);
    let shader = ["mesh-gradient", "swirl-shape-gradient"][rng.below(2)];
    doc.nodes.insert(id.clone().into(), json!({ "id": id, "name": id, "type": "rectangle", "x": x, "y": y,
        "width": side, "height": side * 0.8, "rotation": 0, "scale": 1,
        "fills": [{ "id": "fill-0", "paint": { "type": "shader", "shaderId": shader, "params": {} } }] }));
    id
}

/// A document with motion-capture units among the fuzzer's usual layers: large shader fills on their own and
/// stacked in plain groups (one unit per group), and groups with a blend mode.
fn motion_doc(rng: &mut Rng, top: usize) -> Doc {
    let mut doc = canvas_doc(rng, top);
    for _ in 0..3 {
        let (x, y) = (rng.f(-200.0, 1400.0), rng.f(-200.0, 900.0));
        let id = big_shader(&mut doc, rng, x, y);
        let at = rng.below(doc.roots.len() + 1);
        doc.roots.insert(at, id);
    }
    for _ in 0..2 {
        let (x, y) = (rng.f(0.0, 1200.0), rng.f(0.0, 800.0));
        let kids: Vec<Value> = (0..2).map(|k| json!(big_shader(&mut doc, rng, k as f64 * 80.0, 0.0))).collect();
        let g = doc.fresh_id();
        doc.nodes.insert(g.clone().into(), json!({ "id": g, "name": g, "type": "group", "x": x, "y": y, "rotation": 0,
            "scaleX": 1, "scaleY": 1, "children": kids }));
        let at = rng.below(doc.roots.len() + 1);
        doc.roots.insert(at, g);
    }
    for mode in ["screen", "multiply", "overlay"] {
        let (x, y) = (rng.f(0.0, 1200.0), rng.f(0.0, 800.0));
        let g = doc.subtree(rng, x, y, 1);
        let n = doc.nodes.get_mut(&g).unwrap().as_object_mut().unwrap();
        n.insert("type".into(), json!("group"));
        n.insert("blendMode".into(), json!(mode));
        let at = rng.below(doc.roots.len() + 1);
        doc.roots.insert(at, g);
    }
    doc.motion = true;
    doc
}

/// Motion captures on (the live editor's default): the camera's moving frames plan their units as captures in a
/// scene state of their own, the still frames plan with the state still frames had, and edits land in both (a still
/// frame after a motion patches every edit the motion's frames saw at once). Each moving frame matches a
/// from-scratch build under the same units and frozen clock; each still frame matches one with motion captures off.
#[test]
fn motion_capture_toggles_match_a_full_rebuild() {
    let mut rng = Rng(0x5eed_4d0c);
    let doc = motion_doc(&mut rng, 30);
    let (patched, _, _) = fuzz(doc, 0x4D0C, 1200, &[]);
    assert!(patched > 300, "the slot served {patched} patches");
    let moving = LAST_MOTION_FRAMES.with(|m| m.get());
    assert!(moving > 100, "{moving} frames planned with motion captures");

    // With a clock that moves: a moving frame holds the motion's first clock.
    let mut rng = Rng(0x5eed_4d0d);
    let mut doc = motion_doc(&mut rng, 20);
    doc.clock_moves = 0.2;
    fuzz(doc, 0x4D0D, 800, &[]);
    let moving = LAST_MOTION_FRAMES.with(|m| m.get());
    assert!(moving > 50, "{moving} frames planned with motion captures");
}

/// A top-level insert before a frame and an edit of a text inside it, in one patch (or in patches one frame sees
/// together, as the still frame after a motion does): the insert and the frame's whole-root re-plan both start at
/// the frame's root. Applied one after the other, the second re-planned the inserted node's slot: the frame's first
/// child was drawn twice. Found by `motion_capture_toggles_match_a_full_rebuild`.
#[test]
fn an_insert_before_a_root_and_an_edit_inside_it_patch_together() {
    let doc = json!({
        "nodes": [
            { "id": "f", "type": "container", "x": 421.82, "y": 444.96, "children": ["card", "inner"] },
            { "id": "card", "type": "container", "x": 623.078, "children": [], "fills": solid("#ffffff") },
            { "id": "inner", "type": "container", "children": ["t"] },
            { "id": "t", "type": "richtext" },
        ],
        "pages": [], "canvasRootNodeIds": ["f"], "layoutMode": "freeform",
        "variables": [], "brushes": [], "config": {},
    });
    let mut resident = ResidentScene::default();
    resident.update(&doc.to_string()).expect("scene");
    let mut caches = PlanCaches::default();
    caches.set_cull_min_nodes(Some(0));
    let opts = editor_opts();
    let c = cam(1078.0, 485.0, 3.5);
    let _ = build_camera_plan_retained(&resident, &mut caches, &[], &c, &opts).expect("plan").expect("some");
    let patch = json!({
        "canvasRootNodeIds": ["x", "f"],
        "upserts": [{ "id": "x", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10 }],
        "nodes": { "t": { "x": 71.0 } },
    });
    resident.apply_props(&patch.to_string()).expect("patch");
    caches.begin_frame();
    let patches = caches.incr_patch_hits;
    let (plan, _) = build_camera_plan_retained(&resident, &mut caches, &[], &c, &opts).expect("plan").expect("some");
    assert_eq!(caches.incr_patch_hits, patches + 1, "patched: {:?}", caches.incr_last_fallback);
    let scene = resident.scene().expect("scene");
    let hashes = resident.hashes().expect("hashes");
    if let Some(d) = plan_divergence(scene, &[], &c, &opts, hashes, caches.camera_cull.as_ref(), caches.capture_view, &plan) {
        panic!("the served plan diverged: {d}");
    }
}

#[test]
fn canvas_root_edits_match_a_full_rebuild() {
    let mut rng = Rng(0x5eed_0001);
    let doc = canvas_doc(&mut rng, 40);
    let (patched, _, fallbacks) = fuzz(doc, 0xC0FFEE, 1200, &[]);
    assert!(patched > 400, "the slot served {patched} patches");
    assert!(
        fallbacks < patched,
        "{fallbacks} fallbacks vs {patched} patches"
    );
}

#[test]
fn page_edits_match_a_full_rebuild() {
    let mut rng = Rng(0x5eed_0002);
    let mut doc = canvas_doc(&mut rng, 0);
    doc.layout = "vertical";
    for p in 0..2 {
        let mut ids = Vec::new();
        for _ in 0..20 {
            let (x, y) = (rng.f(0.0, 1100.0), rng.f(0.0, 800.0));
            let id = if rng.chance(0.35) {
                doc.subtree(&mut rng, x, y, 1)
            } else {
                let n = doc.leaf(&mut rng, x, y);
                let id = n["id"].as_str().unwrap().to_string();
                doc.nodes.insert(id.clone().into(), n);
                id
            };
            ids.push(id);
        }
        doc.pages.push((format!("page-{p}"), ids));
    }
    let patterns_before = crate::pattern_paint::layers_planned();
    let (patched, _, _) = fuzz(doc, 0xBEEF, 1000, &[0, 1]);
    assert!(patched > 300, "the slot served {patched} patches");
    assert!(crate::pattern_paint::layers_planned() > patterns_before, "patterned leaves were planned");
}

#[test]
fn freeform_pages_and_canvas_root_match_a_full_rebuild() {
    let mut rng = Rng(0x5eed_0003);
    let mut doc = canvas_doc(&mut rng, 15);
    let mut ids = Vec::new();
    for _ in 0..15 {
        let (x, y) = (rng.f(0.0, 1100.0), rng.f(0.0, 800.0));
        let n = doc.leaf(&mut rng, x, y);
        let id = n["id"].as_str().unwrap().to_string();
        doc.nodes.insert(id.clone().into(), n);
        ids.push(id);
    }
    doc.pages.push(("page-a".into(), ids));
    let (patched, _, _) = fuzz(doc, 0xFACE, 1000, &[0]);
    assert!(patched > 200, "the slot served {patched} patches");
}

/// More seeds over every document shape (slow: `--ignored`; `FUZZ_SEEDS`
/// sets how many).
#[test]
#[ignore]
fn many_seeds_match_a_full_rebuild() {
    let seeds: u64 = std::env::var("FUZZ_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    for k in 0..seeds {
        let mut rng = Rng(0xA11CE + k);
        let doc = canvas_doc(&mut rng, 30 + (k as usize % 3) * 20);
        fuzz(doc, 0x1000 + k, 1500, &[]);
        let mut rng = Rng(0xB0B + k);
        let mut doc = canvas_doc(&mut rng, 10);
        doc.layout = if k % 2 == 0 { "freeform" } else { "vertical" };
        if doc.layout != "freeform" {
            doc.roots.clear();
        }
        for p in 0..3 {
            let mut ids = Vec::new();
            for _ in 0..12 {
                let (x, y) = (rng.f(0.0, 1100.0), rng.f(0.0, 800.0));
                let id = if rng.chance(0.35) {
                    doc.subtree(&mut rng, x, y, 2)
                } else {
                    let n = doc.leaf(&mut rng, x, y);
                    let id = n["id"].as_str().unwrap().to_string();
                    doc.nodes.insert(id.clone().into(), n);
                    id
                };
                ids.push(id);
            }
            doc.pages.push((format!("page-{p}"), ids));
        }
        fuzz(doc, 0x2000 + k, 1500, &[0, 1, 2]);
        // Blends and clip masks, on the canvas and on pages.
        let mut rng = Rng(0xD3B5 + k);
        let doc = dep_canvas_doc(&mut rng, 25 + (k as usize % 3) * 15);
        fuzz(doc, 0x3000 + k, 1500, &[]);
        let mut rng = Rng(0xD3B6 + k);
        let mut doc = canvas_doc(&mut rng, 0);
        doc.layout = if k % 2 == 0 { "freeform" } else { "vertical" };
        doc.source_bias = 0.5;
        if doc.layout == "freeform" {
            doc.roots = dep_tops(&mut doc, &mut rng, 8);
        }
        for p in 0..3 {
            let ids = dep_tops(&mut doc, &mut rng, 10);
            doc.pages.push((format!("page-{p}"), ids));
        }
        fuzz(doc, 0x4000 + k, 1500, &[0, 1, 2]);
    }
}

/// A pan over a split whose units hold a procedural fill (a screen-blended group with a shader-filled rectangle,
/// as a poster's glow): every frame re-plans the split units, and the retained slot keeps serving (the fill's
/// clock is re-threaded at present, as a seed with it allows). The slot used to be dropped after every such patch,
/// so every other frame rebuilt the whole plan. Every frame is compared with a from-scratch build.
#[test]
fn a_pan_over_a_split_with_a_procedural_fill_keeps_the_slot() {
    let shader = |id: &str, x: f64, y: f64| {
        json!({ "id": id, "name": id, "type": "rectangle", "x": x, "y": y, "width": 220, "height": 160,
            "rotation": 0, "scale": 1,
            "fills": [{ "id": "fill-0", "paint": { "type": "shader", "shaderId": "mesh-gradient", "params": {} } }] })
    };
    let mut nodes = Map::new();
    let mut roots = Vec::new();
    for (i, (x, y)) in [(0.0, 0.0), (300.0, 80.0), (620.0, 40.0)].into_iter().enumerate() {
        let r = format!("r{i}");
        nodes.insert(r.clone().into(), json!({ "id": r, "name": r, "type": "rectangle", "x": x, "y": y,
            "width": 200, "height": 140, "rotation": 0, "scale": 1, "fills": solid(COLORS[i]) }));
        roots.push(r);
    }
    nodes.insert("bg".into(), shader("bg", -40.0, 260.0));
    roots.push("bg".into());
    nodes.insert("glow".into(), json!({ "id": "glow", "name": "glow", "type": "group", "x": 0, "y": 0,
        "rotation": 0, "scale": 1, "blendMode": "screen", "children": ["g_fill", "g_dot"] }));
    nodes.insert("g_fill".into(), shader("g_fill", 180.0, 120.0));
    nodes.insert("g_dot".into(), json!({ "id": "g_dot", "name": "g_dot", "type": "ellipse", "x": 260, "y": 160,
        "width": 90, "height": 70, "rotation": 0, "scale": 1, "fills": solid("#ffd166") }));
    roots.push("glow".into());
    let doc = Doc { nodes, roots, ..Default::default() };
    let mut resident = ResidentScene::default();
    resident.update(&doc.to_json()).expect("scene");
    let mut caches = PlanCaches::default();
    let opts = editor_opts();
    for step in 0..40 {
        caches.begin_frame();
        let c = cam(-20.0 + step as f64 * 3.0, -10.0 + step as f64 * 1.5, 0.8);
        let (plan, _) = build_camera_plan_retained(&resident, &mut caches, &[], &c, &opts)
            .expect("plan")
            .expect("some");
        assert!(
            plan.items.iter().any(|i| matches!(i, crate::seams::DrawItem::BackdropSplit(_))),
            "the screen group splits"
        );
        let scene = resident.scene().expect("scene");
        let hashes = resident.hashes().expect("hashes");
        if let Some(d) = plan_divergence(scene, &[], &c, &opts, hashes, caches.camera_cull.as_ref(), caches.capture_view, &plan) {
            panic!("frame {step}: the served plan diverged: {d}");
        }
    }
    assert_eq!(caches.camera_rebuilds, 1, "only the first frame builds from scratch");
    assert!(caches.incr_patch_hits >= 39, "{} patched frames", caches.incr_patch_hits);
}

/// Blend splits turning on and off with no background blur: the slot switches the segment's split context and
/// re-plans only the units that read it (`CameraSlot::respec_segment`). The documents drop the background blurs
/// (a backdrop effect still rebuilds) and give paths strokes and translucency, so blend-mode, opacity and fill
/// edits flip the context between plans that hold plain paths, paint stacks, text, SDF shapes and blended groups.
/// Every frame is compared with a from-scratch build.
fn split_doc(rng: &mut Rng, top: usize, pages: usize) -> Doc {
    let mut doc = canvas_doc(rng, top);
    for p in 0..pages {
        let mut ids = Vec::new();
        for _ in 0..10 {
            let (x, y) = (rng.f(0.0, 1100.0), rng.f(0.0, 800.0));
            let id = if rng.chance(0.35) {
                doc.subtree(rng, x, y, 1)
            } else {
                let n = doc.leaf(rng, x, y);
                let id = n["id"].as_str().unwrap().to_string();
                doc.nodes.insert(id.clone().into(), n);
                id
            };
            ids.push(id);
        }
        doc.pages.push((format!("page-{p}"), ids));
    }
    // No split to start with: the edits turn one on (a blend mode, a fill under a translucent stroke) and off.
    for n in doc.nodes.values_mut() {
        let o = n.as_object_mut().expect("object");
        o.remove("blendMode");
        if let Some(Value::Array(fx)) = o.get_mut("effects") {
            fx.retain(|e| e["type"] != "background-blur");
        }
        if o["type"] == "path" && rng.chance(0.7) {
            // A translucent stroke-only path: a fill turns it into a grouped paint stack (a split).
            o.insert(
                "strokes".into(),
                json!([{ "id": "stroke-0", "paint": { "type": "solid", "color": "#222222" }, "width": 2 }]),
            );
            o.insert("fills".into(), json!([]));
            o.insert("opacity".into(), json!(0.6));
        }
    }
    doc
}

#[test]
fn split_context_flips_match_a_full_rebuild() {
    let mut rng = Rng(0x5eed_0005);
    let doc = split_doc(&mut rng, 14, 0);
    let (patched, _, _) = fuzz(doc, 0x5717, 1200, &[]);
    assert!(patched > 400, "the slot served {patched} patches");
    let mut rng = Rng(0x5eed_0006);
    let mut doc = split_doc(&mut rng, 0, 1);
    doc.layout = "vertical";
    let (patched, _, _) = fuzz(doc, 0x5718, 1000, &[0]);
    assert!(patched > 300, "the slot served {patched} patches");
}

/// Scripted split-context switches over every kind of node a switch could reach: a translucent stroke-only path
/// gains and loses a fill (Poplars' recolour and undo), a path's and a group's blend mode turn on and off, and a
/// translucent group's stroke-only child gains a fill. Around them sit plain paths, a pattern stack, rects and
/// ellipses (their SDF path follows the context), text, a drop-shadowed group, a layer-blurred leaf and a clip
/// frame. On the canvas and on a page, at two camera positions; every frame is compared with a from-scratch build, and the
/// switches must patch (no split-context rebuilds).
#[test]
fn scripted_split_context_switches_patch_exactly() {
    let path = |id: &str, x: f64, extra: Value| {
        let mut n = json!({ "id": id, "name": id, "type": "path", "x": x, "y": 40, "width": 60, "height": 50,
            "rotation": 0, "scale": 1, "closed": true,
            "points": [{ "type": "move", "x": 0, "y": 0 }, { "type": "line", "x": 60, "y": 10 },
                       { "type": "line", "x": 20, "y": 50 }],
            "fills": solid("#3a86ff") });
        for (k, v) in extra.as_object().expect("object") {
            n[k] = v.clone();
        }
        n
    };
    let stroke = json!([{ "id": "stroke-0", "paint": { "type": "solid", "color": "#222222" }, "width": 3 }]);
    let shape = |id: &str, ty: &str, x: f64, y: f64| {
        json!({ "id": id, "name": id, "type": ty, "x": x, "y": y, "width": 50, "height": 40, "rotation": 0,
            "scale": 1, "fills": solid("#ef476f"),
            "strokes": [{ "id": "stroke-0", "paint": { "type": "solid", "color": "#111111" }, "width": 2 }] })
    };
    for page in [false, true] {
        let mut nodes = Map::new();
        let mut add = |n: Value| {
            nodes.insert(n["id"].as_str().unwrap().into(), n);
        };
        add(path("toggle", 0.0, json!({ "fills": [], "strokes": stroke, "opacity": 0.6 })));
        add(path("plain", 80.0, json!({ "strokes": stroke })));
        add(path("stack", 160.0, json!({ "fills": [
            { "id": "fill-0", "paint": { "type": "solid", "color": "#06d6a0" } },
            { "id": "fill-1", "paint": { "type": "pattern", "patternId": "bricks" } }] })));
        add(path("blend", 240.0, json!({})));
        add(shape("rect", "rectangle", 320.0, 40.0));
        add(shape("ell", "ellipse", 400.0, 40.0));
        add(json!({ "id": "text", "name": "text", "type": "richtext", "x": 480, "y": 40, "width": 120,
            "height": 30, "rotation": 0, "scale": 1, "htmlContent": html("hello") }));
        add(json!({ "id": "blurred", "name": "blurred", "type": "ellipse", "x": 0, "y": 140, "width": 60,
            "height": 60, "rotation": 0, "scale": 1, "fills": solid("#ffbe0b"),
            "effects": [{ "id": "layer-blur-0", "type": "layer-blur", "radius": 4 }] }));
        // A group that turns blended, over a rect and a path.
        add(shape("g1-rect", "rectangle", 0.0, 0.0));
        add(path("g1-path", 40.0, json!({ "y": 10 })));
        add(json!({ "id": "g1", "name": "g1", "type": "group", "x": 100, "y": 140, "width": 120, "height": 80,
            "rotation": 0, "scale": 1, "children": ["g1-rect", "g1-path"] }));
        // A translucent stroke-only layer in a group that gains a fill (then a grouped stack under its opacity). A
        // translucent group always composites in its own split, so the opacity is the layer's.
        add(path("g2-kid", 0.0, json!({ "y": 0, "fills": [], "strokes": stroke, "opacity": 0.7 })));
        add(path("g2-other", 70.0, json!({ "y": 0 })));
        add(json!({ "id": "g2", "name": "g2", "type": "group", "x": 260, "y": 140, "width": 140, "height": 60,
            "rotation": 0, "scale": 1, "children": ["g2-kid", "g2-other"] }));
        // A drop-shadowed group and a clip frame.
        add(shape("g3-ell", "ellipse", 0.0, 0.0));
        add(json!({ "id": "g3", "name": "g3", "type": "group", "x": 420, "y": 140, "width": 60, "height": 50,
            "rotation": 0, "scale": 1, "children": ["g3-ell"],
            "effects": [{ "id": "drop-shadow-0", "type": "drop-shadow", "color": "#000000", "blur": 6,
                          "offset": { "x": 0, "y": 3 }, "opacity": 0.3 }] }));
        add(path("f-path", 10.0, json!({ "y": 10, "strokes": stroke })));
        add(shape("f-rect", "rectangle", 50.0, 20.0));
        add(json!({ "id": "frame", "name": "frame", "type": "container", "x": 500, "y": 140, "width": 120,
            "height": 80, "rotation": 0, "scale": 1, "clipContent": true, "fills": solid("#f4f4f5"),
            "children": ["f-path", "f-rect"] }));
        let tops: Vec<String> = ["toggle", "plain", "stack", "blend", "rect", "ell", "text", "blurred", "g1",
            "g2", "g3", "frame"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut doc = Doc { nodes, layout: if page { "vertical" } else { "freeform" }, ..Default::default() };
        let visible: &[usize] = if page {
            doc.pages.push(("page-0".into(), tops));
            &[0]
        } else {
            doc.roots = tops;
            &[]
        };
        let mut resident = ResidentScene::default();
        resident.update(&doc.to_json()).expect("scene");
        let mut caches = PlanCaches::default();
        caches.set_cull_min_nodes(Some(0));
        let opts = editor_opts();
        let red = json!([{ "id": "fill-0", "paint": { "type": "solid", "color": "#c2410c" } }]);
        let edits = [
            json!({ "toggle": { "fills": red } }),
            json!({ "toggle": { "fills": [] } }),
            json!({ "blend": { "blendMode": "multiply" } }),
            json!({ "blend": { "blendMode": "normal" } }),
            json!({ "g2-kid": { "fills": red } }),
            json!({ "rect": { "x": 330 } }),
            json!({ "g2-kid": { "fills": [] } }),
            json!({ "g1": { "blendMode": "screen" } }),
            json!({ "toggle": { "fills": red } }),
            json!({ "g1": { "blendMode": "normal" } }),
            json!({ "toggle": { "fills": [] } }),
        ];
        let respecs = || super::RESPECS.load(std::sync::atomic::Ordering::Relaxed);
        let (before, hits0) = (respecs(), caches.incr_patch_hits);
        let mut frame = 0;
        // A camera per round: all in view, then panned so the left half is culled, then back.
        for (round, c) in [cam(-50.0, -50.0, 1.0), cam(450.0, -50.0, 1.0), cam(-50.0, -50.0, 1.0)]
            .into_iter()
            .enumerate()
        {
            for e in &edits {
                resident.apply_props(&json!({ "nodes": e }).to_string()).expect("patch");
                {
                    caches.begin_frame();
                    let (plan, _w) = build_camera_plan_retained(&resident, &mut caches, visible, &c, &opts)
                        .expect("plan")
                        .expect("some");
                    let scene = resident.scene().expect("scene");
                    let hashes = resident.hashes().expect("hashes");
                    if let Some(d) = plan_divergence(
                        scene,
                        visible,
                        &c,
                        &opts,
                        hashes,
                        caches.camera_cull.as_ref(),
                        caches.capture_view,
                        &plan,
                    ) {
                        panic!("page {page}, round {round}, frame {frame} after {e}: {d}");
                    }
                    frame += 1;
                }
            }
        }
        let _ = &mut doc;
        // Other tests share the counter: a floor, not an exact count.
        assert!(respecs() - before >= 12, "{} split-context switches", respecs() - before);
        assert_eq!(
            caches.incr_last_fallback.filter(|w| *w == "split context changed"),
            None,
            "a switch rebuilt the plan"
        );
        assert!(caches.incr_patch_hits - hits0 >= 25, "{} patches", caches.incr_patch_hits - hits0);
    }
}

/// A blend's steps drawn from another zoom band's flattening while off-screen (a live frame after a band change)
/// are refined once the camera comes near them, as any curve is: the frames after that match a full rebuild.
#[test]
fn off_screen_blend_steps_refine_when_the_camera_comes_near() {
    let ellipse = |id: &str, x: f64, c: &str| {
        json!({ "id": id, "name": id, "type": "ellipse", "x": x, "y": 0, "width": 90, "height": 70,
            "rotation": 0, "scale": 1, "fills": solid(c) })
    };
    let mut nodes = Map::new();
    nodes.insert("a".into(), ellipse("a", 0.0, "#3a86ff"));
    nodes.insert("b".into(), ellipse("b", 300.0, "#ef476f"));
    nodes.insert("bl".into(), json!({ "id": "bl", "name": "bl", "type": "blend", "sourceIds": ["a", "b"],
        "spacing": "steps", "steps": 4 }));
    nodes.insert("far".into(), json!({ "id": "far", "name": "far", "type": "rectangle", "x": 2000, "y": 0,
        "width": 100, "height": 80, "rotation": 0, "scale": 1, "fills": solid("#06d6a0") }));
    let roots = vec!["a".to_string(), "b".into(), "bl".into(), "far".into()];
    let doc = Doc { nodes, roots, ..Default::default() };
    let mut resident = ResidentScene::default();
    resident.update(&doc.to_json()).expect("scene");
    let mut caches = PlanCaches::default();
    caches.set_hold_refine_while_zooming(false);
    let opts = editor_opts();
    let frame = |caches: &mut PlanCaches, c: &CameraParams| {
        caches.begin_frame();
        caches.set_outline_refine_budget_ms(Some(1000.0));
        let (plan, _) = build_camera_plan_retained(&resident, caches, &[], c, &opts).expect("plan").expect("some");
        let refining = caches.outline_refining();
        caches.set_outline_refine_budget_ms(None);
        let scene = resident.scene().expect("scene");
        let hashes = resident.hashes().expect("hashes");
        let d = plan_divergence(scene, &[], c, &opts, hashes, caches.camera_cull.as_ref(), caches.capture_view, &plan);
        (d, refining)
    };
    // At 1x over the blend: every step flattened at that band.
    let (d, _) = frame(&mut caches, &cam(-50.0, -100.0, 1.0));
    assert_eq!(d, None, "first frame");
    // At 4x over the far rectangle, the zoom held long enough to settle: the steps are far off-screen, so they
    // draw their 1x flattening and wait.
    for _ in 0..3 {
        std::thread::sleep(std::time::Duration::from_millis(250));
        frame(&mut caches, &cam(2000.0, 0.0, 4.0));
    }
    // Over the blend again, at 4x: within a few frames the steps are refined. (They were not: the queue named
    // the steps, which are no scene node, so the patch skipped them and dropped them from the queue; the 1x
    // flattening stayed until the next full rebuild.)
    let mut last = None;
    for _ in 0..6 {
        let (d, refining) = frame(&mut caches, &cam(-20.0, -40.0, 4.0));
        last = d;
        if last.is_none() && !refining {
            break;
        }
    }
    assert_eq!(last, None, "the steps are refined");
}
