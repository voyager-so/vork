//! Flipbooks (docs/design/motion-model.md §12.10): a `group` with a `flipbook` object draws ONE of its children, the
//! drawing at `flipbook.current` (absent = 0, clamped to the last drawing). The motion overlay writes `current` for
//! the time being drawn (moda-motion `flipbook.rs`), so the editor's canvas and every export pick the same drawing;
//! outside a timeline the stored `current` is the drawing shown at rest.
//!
//! **Onion skin** is a preview aid, never exported: a flipbook with `onionSkin: { before, after }` (or a number, both
//! sides) also draws its `before` previous and `after` next drawings behind the current one, faded and tinted (red
//! before, green after). [`expand_onion`] builds them when a scene is made: each is a copy of the drawing's subtree
//! (ids `<id>~onion<±k>`) with its paint recoloured, in a wrapper group (`<flipbook>~onion<±k>`) that fades it; every
//! node it adds carries [`ONION_KEY`] and is dropped again the next time the scene is made, so the expansion is
//! idempotent. Export lanes remove `onionSkin` first ([`strip_onion_skin`]).

use std::collections::HashMap;

use crate::value::{json, Map, Value};

use crate::types::{RawNode, SceneDocument};

/// The flipbook object on a group.
pub const FLIPBOOK_KEY: &str = "flipbook";
/// The onion skin request on a flipbook (editor state and preview option; never saved or exported).
pub const ONION_SKIN_KEY: &str = "onionSkin";
/// Marks a node the onion expansion added (its value: the flipbook's id).
pub const ONION_KEY: &str = "_onion";
/// The most drawings an onion skin shows on each side.
pub const ONION_MAX: usize = 8;
/// The tint of the drawings before the current one, and after it.
pub const ONION_BEFORE_TINT: &str = "#E5484D";
pub const ONION_AFTER_TINT: &str = "#30A46C";

/// Whether `node` is a flipbook: a group with a `flipbook` object.
pub fn is_flipbook(node: &RawNode) -> bool {
    node.node_type() == "group" && node.object(FLIPBOOK_KEY).is_some()
}

/// The drawing a flipbook shows among `count` drawings: its `current`, floored and clamped.
pub fn current_index(node: &RawNode, count: usize) -> usize {
    let k = node
        .object(FLIPBOOK_KEY)
        .and_then(|f| f.get("current"))
        .and_then(Value::as_f64)
        .filter(|k| k.is_finite() && *k > 0.0)
        .map_or(0, |k| k.floor().min(usize::MAX as f64) as usize);
    k.min(count.saturating_sub(1))
}

/// What a flipbook draws, back to front: its onion-skin wrappers (children marked [`ONION_KEY`]), then the current
/// drawing. `is_onion` says whether a child id is an onion wrapper.
pub fn drawn_children(node: &RawNode, children: &[String], is_onion: impl Fn(&str) -> bool) -> Vec<String> {
    let (onion, drawings): (Vec<&String>, Vec<&String>) = children.iter().partition(|c| is_onion(c));
    let mut out: Vec<String> = onion.into_iter().cloned().collect();
    if !drawings.is_empty() {
        out.push(drawings[current_index(node, drawings.len())].clone());
    }
    out
}

/// `(before, after)` from a flipbook's `onionSkin`: `{ before, after }` or a number for both; `None` when off.
pub fn onion_request(node: &RawNode) -> Option<(usize, usize)> {
    let count = |v: Option<&Value>| {
        v.and_then(Value::as_f64)
            .filter(|n| n.is_finite() && *n > 0.0)
            .map_or(0, |n| (n.round() as usize).min(ONION_MAX))
    };
    let (b, a) = match node.json_field(ONION_SKIN_KEY)? {
        Value::Object(o) => (count(o.get("before")), count(o.get("after"))),
        v @ Value::Number(_) => (count(Some(v)), count(Some(v))),
        Value::Bool(true) => (1, 1),
        _ => (0, 0),
    };
    (b + a > 0).then_some((b, a))
}

/// Whether `id` draws through an onion-skinned flipbook: the flipbook itself, anything inside it, or a node the
/// expansion added. Such an edit re-derives the copies through a full load (the resident scene can't patch them).
pub fn in_onion_flipbook(scene: &crate::scene::Scene, id: &str) -> bool {
    let mut cur = id;
    for _ in 0..512 {
        match scene.node(cur) {
            Some(n) if n.has(ONION_KEY) || (is_flipbook(n) && n.has(ONION_SKIN_KEY)) => return true,
            _ => {}
        }
        match scene.parent_map.get(cur) {
            Some(p) => cur = p,
            None => return false,
        }
    }
    false
}

/// Remove `onionSkin` from every node (and any expansion left in the document): what an export draws.
pub fn strip_onion_skin(doc: &mut SceneDocument) {
    for n in &mut doc.nodes {
        if n.has(ONION_SKIN_KEY) {
            n.remove(ONION_SKIN_KEY);
        }
    }
    remove_onion_expansion(doc);
}

/// Drop the nodes an earlier [`expand_onion`] added, and their ids from the flipbooks' children (the document as
/// stored, `onionSkin` kept). Whether there were any.
pub fn remove_onion_expansion(doc: &mut SceneDocument) -> bool {
    if !doc.nodes.iter().any(|n| n.has(ONION_KEY)) {
        return false;
    }
    let removed: std::collections::HashSet<String> =
        doc.nodes.iter().filter(|n| n.has(ONION_KEY)).map(|n| n.id().to_string()).collect();
    doc.nodes.retain(|n| !n.has(ONION_KEY));
    for n in &mut doc.nodes {
        if !is_flipbook(n) {
            continue;
        }
        if let Some(Value::Array(children)) = n.json_field("children") {
            if children.iter().any(|c| c.as_str().is_some_and(|c| removed.contains(c))) {
                let kept: Vec<Value> =
                    children.iter().filter(|c| !c.as_str().is_some_and(|c| removed.contains(c))).cloned().collect();
                n.set("children", Value::Array(kept));
            }
        }
    }
    true
}

/// Build the onion skins of every flipbook that asks for one (see the module docs). Idempotent: an earlier
/// expansion is removed first, so a document whose `current` or drawings changed expands afresh.
pub fn expand_onion(doc: &mut SceneDocument) {
    let had = remove_onion_expansion(doc);
    if !had && !doc.nodes.iter().any(|n| n.has(ONION_SKIN_KEY)) {
        return;
    }
    let index: HashMap<String, usize> = doc.nodes.iter().enumerate().map(|(i, n)| (n.id().to_string(), i)).collect();
    let mut added: Vec<RawNode> = Vec::new();
    let mut wrappers: Vec<(usize, Vec<String>)> = Vec::new();
    for (fi, fb) in doc.nodes.iter().enumerate() {
        if !is_flipbook(fb) {
            continue;
        }
        let Some((before, after)) = onion_request(fb) else { continue };
        let drawings: Vec<String> = fb
            .array("children")
            .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
            .unwrap_or_default();
        if drawings.len() < 2 {
            continue;
        }
        let current = current_index(fb, drawings.len()) as i64;
        let fb_id = fb.id().to_string();
        // Back to front: the farthest drawings first, before and after the current one alternating inwards.
        let mut offsets: Vec<i64> = Vec::new();
        for d in (1..=before.max(after) as i64).rev() {
            if d as usize <= before {
                offsets.push(-d);
            }
            if d as usize <= after {
                offsets.push(d);
            }
        }
        let mut ids = Vec::new();
        for o in offsets {
            let k = current + o;
            if k < 0 || k >= drawings.len() as i64 {
                continue;
            }
            let reach = if o < 0 { before } else { after };
            let tint = if o < 0 { ONION_BEFORE_TINT } else { ONION_AFTER_TINT };
            let suffix = format!("~onion{o:+}");
            let Some(root) = clone_subtree(doc, &index, &drawings[k as usize], &suffix, tint, &fb_id, &mut added) else {
                continue;
            };
            let wid = format!("{fb_id}{suffix}");
            let mut w = Map::new();
            w.insert("id".into(), Value::from(wid.clone()));
            w.insert("type".into(), Value::from("group"));
            w.insert("name".into(), Value::from(format!("Onion {o:+}")));
            for (k, v) in [("x", 0.0), ("y", 0.0), ("rotation", 0.0), ("scaleX", 1.0), ("scaleY", 1.0)] {
                w.insert(k.into(), Value::from(v));
            }
            w.insert("opacity".into(), Value::from(onion_opacity(o.unsigned_abs() as usize, reach)));
            w.insert("children".into(), json!([root]));
            w.insert(ONION_KEY.into(), Value::from(fb_id.clone()));
            added.push(RawNode::new(w));
            ids.push(wid);
        }
        if !ids.is_empty() {
            wrappers.push((fi, ids));
        }
    }
    for (fi, ids) in wrappers {
        let fb = &mut doc.nodes[fi];
        let mut children: Vec<Value> = ids.into_iter().map(Value::from).collect();
        if let Some(Value::Array(old)) = fb.json_field("children") {
            children.extend(old.iter().cloned());
        }
        fb.set("children", Value::Array(children));
    }
    doc.nodes.extend(added);
}

/// How opaque the onion drawing `d` steps from the current one is, of `reach` on its side: 0.4 next to it, fading
/// towards 0.1 at the far end.
pub fn onion_opacity(d: usize, reach: usize) -> f64 {
    let reach = reach.max(1) as f64;
    0.4 - 0.3 * (d.saturating_sub(1) as f64) / reach
}

/// Copy `id`'s subtree with `suffix` on every id and its paint turned to `tint`; returns the copy's root id.
fn clone_subtree(
    doc: &SceneDocument,
    index: &HashMap<String, usize>,
    id: &str,
    suffix: &str,
    tint: &str,
    flipbook: &str,
    out: &mut Vec<RawNode>,
) -> Option<String> {
    let mut stack = vec![id.to_string()];
    let mut depth_guard = 0usize;
    while let Some(cur) = stack.pop() {
        depth_guard += 1;
        if depth_guard > 100_000 {
            break;
        }
        let Some(&i) = index.get(&cur) else { continue };
        let src = &doc.nodes[i];
        let mut n = src.clone();
        n.set("id", Value::from(format!("{cur}{suffix}")));
        n.set(ONION_KEY, Value::from(flipbook));
        if let Some(Value::Array(children)) = src.json_field("children") {
            let ids: Vec<String> = children.iter().filter_map(Value::as_str).map(str::to_string).collect();
            n.set(
                "children",
                Value::Array(ids.iter().map(|c| Value::from(format!("{c}{suffix}"))).collect()),
            );
            stack.extend(ids);
        }
        tint_node(&mut n, tint);
        out.push(n);
    }
    index.contains_key(id).then(|| format!("{id}{suffix}"))
}

/// Turn a node's paint to `tint`: solid and gradient fills and strokes become the tint, images and videos get a tint
/// overlay, text draws in it; effects go (an onion drawing is a silhouette of the drawing, not a render of it).
fn tint_node(n: &mut RawNode, tint: &str) {
    for key in ["fills", "strokes"] {
        if let Some(Value::Array(list)) = n.json_field(key) {
            let list: Vec<Value> = list.iter().map(|e| tint_entry(e, tint)).collect();
            n.set(key, Value::Array(list));
        }
    }
    if n.node_type() == "richtext" {
        n.set("textFills", json!([{ "id": "onion", "paint": { "type": "solid", "color": tint } }]));
    }
    if n.has("effects") {
        n.set("effects", Value::Array(Vec::new()));
    }
    // A paint layer's copy draws its layer's pixels through the image fill (whose src names the original layer),
    // as a plain rectangle.
    if n.has("_paintLayer") {
        n.remove("_paintLayer");
        n.remove("surface");
    }
}

fn tint_entry(entry: &Value, tint: &str) -> Value {
    let mut e = entry.clone();
    let Some(obj) = e.as_object_mut() else { return e };
    let kind = obj.get("paint").and_then(|p| p.get("type")).and_then(Value::as_str).unwrap_or("solid").to_string();
    match kind.as_str() {
        "image" | "video" => {
            if let Some(Value::Object(p)) = obj.get_mut("paint") {
                let adjust = p.entry("adjust").or_insert_with(|| Value::Object(Map::new()));
                if let Value::Object(a) = adjust {
                    a.insert("overlay".into(), json!({ "enabled": true, "color": tint, "opacity": 0.85 }));
                }
            }
        }
        _ => {
            obj.insert("paint".into(), json!({ "type": "solid", "color": tint }));
        }
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::hydrate_document;

    fn doc(onion: Value, current: u64) -> SceneDocument {
        hydrate_document(&json!({
            "nodes": [
                {"id": "fb", "type": "group", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
                 "flipbook": {"mode": "loop", "current": current}, "onionSkin": onion, "children": ["a", "b", "c"]},
                {"id": "a", "type": "group", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1, "children": ["a1"]},
                {"id": "a1", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10, "rotation": 0, "scale": 1,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#123456"}}]},
                {"id": "b", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10, "rotation": 0, "scale": 1},
                {"id": "c", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10, "rotation": 0, "scale": 1}
            ],
            "pages": [{"id": "p", "nodeIds": ["fb"]}]
        }))
    }

    fn ids(d: &SceneDocument) -> Vec<String> {
        d.nodes.iter().map(|n| n.id().to_string()).collect()
    }

    #[test]
    fn current_is_clamped_and_floored() {
        let d = doc(Value::Null, 7);
        assert_eq!(current_index(&d.nodes[0], 3), 2);
        assert_eq!(current_index(&d.nodes[0], 0), 0);
        let children: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
        assert_eq!(drawn_children(&d.nodes[0], &children, |_| false), vec!["c".to_string()]);
    }

    #[test]
    fn onion_copies_neighbours_tinted_and_is_idempotent() {
        let mut d = doc(json!({"before": 1, "after": 1}), 1);
        expand_onion(&mut d);
        let fb = &d.nodes[0];
        let children: Vec<&str> = fb.array("children").unwrap().iter().filter_map(Value::as_str).collect();
        assert_eq!(children, vec!["fb~onion-1", "fb~onion+1", "a", "b", "c"]);
        let copy = d.nodes.iter().find(|n| n.id() == "a1~onion-1").expect("a's child copied");
        assert_eq!(copy.json_field("fills").unwrap()[0]["paint"]["color"], json!(ONION_BEFORE_TINT));
        let first = ids(&d);
        expand_onion(&mut d);
        assert_eq!(ids(&d), first, "a second expansion changes nothing");
        // Off: the expansion goes.
        strip_onion_skin(&mut d);
        assert_eq!(ids(&d), vec!["fb", "a", "a1", "b", "c"]);
        let children: Vec<&str> = d.nodes[0].array("children").unwrap().iter().filter_map(Value::as_str).collect();
        assert_eq!(children, vec!["a", "b", "c"]);
    }

    #[test]
    fn onion_stops_at_the_ends() {
        let mut d = doc(json!(2), 0);
        expand_onion(&mut d);
        let children: Vec<&str> = d.nodes[0].array("children").unwrap().iter().filter_map(Value::as_str).collect();
        assert_eq!(children, vec!["fb~onion+2", "fb~onion+1", "a", "b", "c"]);
        assert!((onion_opacity(1, 2) - 0.4).abs() < 1e-12);
        assert!(onion_opacity(2, 2) < 0.4);
    }
}
