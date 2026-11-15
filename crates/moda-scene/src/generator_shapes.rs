//! Layers as generator inputs (live-nodes.md §5.2, §11 rank 5): a generator's params name layers by link
//! (`"@Figure"`, `"@#id"`, several as `"@Figure, @Tree"`) for a stroke field's `region` and `avoid`. Each linked layer
//! becomes its outline, flattened, in the generator's local space: `{ "paths": [{ "points", "closed" }] }`, the shape
//! a moda-field `inside` / `distance` reads. Groups, frames and other containers give their shape descendants.
//!
//! One implementation for every host: moda-render calls [`linked_shapes`] on the document it loaded; the editor
//! calls [`generator_shapes_json`] (the engine's `generator_shapes` export) with the same nodes, so a stroke field
//! avoids the same outline in the app, the CLI and exports. Points are rounded to 1/1000 px so the platform's trig
//! (native or wasm) can't change a generator's output.

use std::collections::HashMap;

use crate::value::{json, Map, Value};

use crate::blend::{flatten_path_to_polyline, BlendPathPoint};
use crate::scene::is_parent_node;
use crate::shape_outline::node_outline;
use crate::transform::{apply_point, compose, from_trsk};
use crate::types::{Affine, RawNode, SceneDocument, Transform2D};

/// Most outline points one link may give (a guard against linking a huge drawing).
const MAX_POINTS: usize = 200_000;

/// The layer names a generator's params link to, as written (without the `@`).
pub fn link_names(params: &Map) -> Vec<String> {
    fn walk(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::String(s) => {
                for part in s.split(',') {
                    let part = part.trim();
                    if let Some(name) = part.strip_prefix('@').filter(|n| !n.is_empty()) {
                        if !out.iter().any(|o| o == name) {
                            out.push(name.to_string());
                        }
                    }
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            Value::Object(m) => {
                if let Some(Value::Object(from)) = m.get("from") {
                    if let Some(Value::String(l)) = from.get("layer") {
                        walk(&Value::String(format!("@{l}")), out);
                    }
                }
                m.values().for_each(|x| walk(x, out));
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    params.values().for_each(|v| walk(v, &mut out));
    out
}

struct Index<'a> {
    by_id: HashMap<&'a str, &'a RawNode>,
    parent: HashMap<&'a str, &'a str>,
}

impl<'a> Index<'a> {
    fn new(doc: &'a SceneDocument) -> Self {
        let mut by_id = HashMap::with_capacity(doc.nodes.len());
        let mut parent = HashMap::new();
        for n in &doc.nodes {
            by_id.insert(n.id(), n);
            if let Some(Value::Array(kids)) = n.json_field("children") {
                for k in kids.iter().filter_map(Value::as_str) {
                    parent.insert(k, n.id());
                }
            }
        }
        Index { by_id, parent }
    }

    /// A node's local → canvas affine (the renderer's `accumulated_affine`, from the document).
    fn affine(&self, node: &RawNode) -> Affine {
        let (sx, sy) = node.scale_xy();
        let local = Transform2D {
            x: node.f64("x").unwrap_or(0.0),
            y: node.f64("y").unwrap_or(0.0),
            rotation: node.rotation(),
            scale_x: sx,
            scale_y: sy,
        };
        let mut affine = from_trsk(local, node.skew_x());
        let mut current = node.id();
        let mut guard = 0;
        while let Some(pid) = self.parent.get(current) {
            guard += 1;
            let Some(parent) = self.by_id.get(pid) else {
                break;
            };
            if !is_parent_node(parent) || guard > 256 {
                break;
            }
            let (psx, psy) = if parent.node_type() == "group" {
                let nz = |v: f64| if v == 0.0 { 1.0 } else { v };
                (
                    nz(parent.f64("scaleX").unwrap_or(1.0)),
                    nz(parent.f64("scaleY").unwrap_or(1.0)),
                )
            } else {
                let s = parent.scale();
                (s, s)
            };
            let t = Transform2D {
                x: parent.f64("x").unwrap_or(0.0),
                y: parent.f64("y").unwrap_or(0.0),
                rotation: parent.rotation(),
                scale_x: psx,
                scale_y: psy,
            };
            affine = compose(from_trsk(t, parent.skew_x()), affine);
            current = pid;
        }
        affine
    }

    /// `#id`, an id, or the first layer (in document order) with that name.
    fn find(&self, doc: &'a SceneDocument, name: &str) -> Option<&'a RawNode> {
        if let Some(id) = name.strip_prefix('#') {
            return self.by_id.get(id).copied();
        }
        self.by_id.get(name).copied().or_else(|| {
            doc.nodes.iter().find(|n| {
                n.object("metadata")
                    .and_then(|m| m.get("name"))
                    .and_then(Value::as_str)
                    == Some(name)
            })
        })
    }
}

fn invert(a: &Affine) -> Option<Affine> {
    let det = a.a * a.d - a.b * a.c;
    if !det.is_finite() || det.abs() < 1e-18 {
        return None;
    }
    let (ia, ib, ic, id) = (a.d / det, -a.b / det, -a.c / det, a.a / det);
    Some(Affine {
        a: ia,
        b: ib,
        c: ic,
        d: id,
        e: -(ia * a.e + ic * a.f),
        f: -(ib * a.e + id * a.f),
    })
}

fn round(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

/// A layer's outlines (its own, or its shape descendants'), flattened, through `to_local`.
fn outlines(
    ix: &Index,
    node: &RawNode,
    to_local: &Affine,
    out: &mut Vec<Value>,
    budget: &mut usize,
    depth: u32,
) {
    // A hidden layer inside a linked group is left out; the linked layer itself is named on purpose (a repeater's
    // `shape: "@Petal"` whose original is hidden, a stroke field's hidden guide region), so its outline counts.
    if depth > 64 || (depth > 0 && node.truthy("hidden")) {
        return;
    }
    if let Some((pts, _closed)) = node_outline(node) {
        let pts: Vec<BlendPathPoint> = pts;
        let poly = flatten_path_to_polyline(&pts, true);
        if poly.len() < 3 || poly.len() > *budget {
            return;
        }
        *budget -= poly.len();
        let m = compose(*to_local, ix.affine(node));
        let points: Vec<Value> = poly
            .iter()
            .map(|&p| {
                let (x, y) = apply_point(p, m);
                json!([round(x), round(y)])
            })
            .collect();
        out.push(json!({ "points": points, "closed": true }));
        return;
    }
    if let Some(Value::Array(kids)) = node.json_field("children") {
        for k in kids.iter().filter_map(Value::as_str) {
            if let Some(child) = ix.by_id.get(k) {
                outlines(ix, child, to_local, out, budget, depth + 1);
            }
        }
    }
}

/// The shapes of the layers `params` link to, in the generator's local space, keyed as the links name them. Links
/// to `std:field` nodes are fields, not shapes, and are left out; so are links to nothing and to the generator itself.
pub fn linked_shapes(
    params: &Map,
    doc: &SceneDocument,
    generator_id: &str,
) -> Map {
    let names = link_names(params);
    let mut out = Map::new();
    if names.is_empty() {
        return out;
    }
    let ix = Index::new(doc);
    let Some(generator) = ix.by_id.get(generator_id) else {
        return out;
    };
    let Some(to_local) = invert(&ix.affine(generator)) else {
        return out;
    };
    for name in names {
        let Some(target) = ix.find(doc, &name) else {
            continue;
        };
        if target.id() == generator_id
            || (target.node_type() == "generator"
                && target
                    .str("code")
                    .is_some_and(|c| c.starts_with("std:field@")))
        {
            continue;
        }
        let mut paths = Vec::new();
        let mut budget = MAX_POINTS;
        outlines(&ix, target, &to_local, &mut paths, &mut budget, 0);
        if !paths.is_empty() {
            let mut shape = json!({ "paths": paths });
            // Its paint, for a generator that redraws the layer (std:repeater@1): the first solid fill and stroke of
            // the layer, or of its first shape descendant.
            if let Some((fill, stroke)) = paint_of(&ix, target, 0) {
                if let Some(f) = fill {
                    // not-a-node-key: ctx.shape's own paint
                    shape["fill"] = json!(f);
                }
                if let Some((c, w)) = stroke {
                    // not-a-node-key: ctx.shape's own paint, { color, width } as a generator item's stroke
                    shape["stroke"] = json!({ "color": c, "width": w });
                }
            }
            out.insert(name.into(), shape);
        }
    }
    out
}

/// A layer's first solid fill colour and first solid stroke (colour, width), or its first shape descendant's.
fn paint_of(ix: &Index<'_>, node: &RawNode, depth: usize) -> Option<(Option<String>, Option<(String, f64)>)> {
    let solid = |v: Option<&Value>| -> Option<String> {
        v?.as_array()?.iter().find_map(|l| {
            if l.get("visible").and_then(Value::as_bool) == Some(false) {
                return None;
            }
            let paint = l.get("paint")?;
            (paint.get("type").and_then(Value::as_str) == Some("solid")).then(|| paint.get("color")?.as_str().map(str::to_string))?
        })
    };
    let fill = solid(node.json_field("fills"));
    let stroke = solid(node.json_field("strokes")).map(|c| {
        let w = node
            .json_field("strokes")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(|s| s.get("width"))
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        (c, w)
    });
    if fill.is_some() || stroke.is_some() {
        return Some((fill, stroke));
    }
    if depth > 16 || !is_parent_node(node) {
        return None;
    }
    let kids = node.json_field("children").and_then(Value::as_array)?;
    kids.iter()
        .filter_map(|c| c.as_str().and_then(|id| ix.by_id.get(id)))
        .find_map(|c| paint_of(ix, c, depth + 1))
}

/// The engine export: `{ nodes: [the generator, the linked layers and their descendants, and every ancestor of
/// these], generator: id, params? }` → `{ status: "ok", shapes: { name: { paths } } }` (`params` defaults to the
/// generator node's own).
pub fn generator_shapes_json(request_json: &str) -> String {
    let Ok(req) = crate::value::from_str::<Value>(request_json) else {
        return json!({ "status": "error", "code": "invalid_json", "message": "request is not valid JSON" }).to_string();
    };
    let Some(id) = req.get("generator").and_then(Value::as_str) else {
        return json!({ "status": "error", "code": "invalid_request", "message": "generator (an id) is required" }).to_string();
    };
    // Only the transforms and outlines matter here: generators are masked so hydrating runs no code.
    let nodes: Vec<Value> = req
        .get("nodes")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|n| {
                    let mut n = n.clone();
                    if n.get("type").and_then(Value::as_str) == Some("generator") {
                        if let Some(o) = n.as_object_mut() {
                            o.remove("params");
                            o.insert("code".into(), json!(""));
                        }
                    }
                    n
                })
                .collect()
        })
        .unwrap_or_default();
    let params = req
        .get("params")
        .and_then(Value::as_object)
        .cloned()
        .or_else(|| {
            req.get("nodes")?
                .as_array()?
                .iter()
                .find(|n| n.get("id").and_then(Value::as_str) == Some(id))?
                .get("params")?
                .as_object()
                .cloned()
        })
        .unwrap_or_default();
    let doc = crate::scene::hydrate_document(&json!({ "nodes": nodes }));
    json!({ "status": "ok", "shapes": linked_shapes(&params, &doc, id) }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linked_layers_become_shapes_in_the_generators_space() {
        let doc = json!({
            "nodes": [
                { "id": "frame", "type": "container", "x": 100, "y": 50, "width": 800, "height": 600, "rotation": 0, "scale": 1,
                  "children": ["g", "fig", "grp"] },
                { "id": "g", "type": "generator", "x": 10, "y": 20, "width": 400, "height": 300, "rotation": 0, "scale": 1,
                  "code": "std:stroke-field@1", "params": { "avoid": "@Figure, @Pair", "region": "M0 0 H10 V10 Z" } },
                { "id": "fig", "type": "rectangle", "x": 60, "y": 70, "width": 20, "height": 40, "rotation": 0, "scale": 1,
                  "metadata": { "name": "Figure" } },
                { "id": "grp", "type": "group", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1, "children": ["a", "b"],
                  "metadata": { "name": "Pair" } },
                { "id": "a", "type": "ellipse", "x": 200, "y": 200, "width": 20, "height": 20, "rotation": 0, "scale": 1 },
                { "id": "b", "type": "rectangle", "x": 300, "y": 200, "width": 10, "height": 10, "rotation": 45, "scale": 1 },
            ],
            "canvasRootNodeIds": ["frame"],
        });
        let d = crate::scene::hydrate_document(&doc);
        let g = d.nodes.iter().find(|n| n.id() == "g").unwrap();
        let shapes = linked_shapes(g.object("params").unwrap(), &d, "g");
        assert_eq!(shapes.len(), 2, "{shapes:?}");
        // The figure: its box, minus the generator's offset in the frame.
        let fig = &shapes["Figure"]["paths"][0]["points"];
        let xs: Vec<f64> = fig
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p[0].as_f64().unwrap())
            .collect();
        let ys: Vec<f64> = fig
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p[1].as_f64().unwrap())
            .collect();
        let (x0, x1) = xs
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        let (y0, y1) = ys
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), &y| (a.min(y), b.max(y)));
        assert_eq!((x0, x1, y0, y1), (50.0, 70.0, 50.0, 90.0));
        // The group gives both children.
        assert_eq!(shapes["Pair"]["paths"].as_array().unwrap().len(), 2);
        // The export answers the same from the nodes alone.
        let r: Value = crate::value::from_str(&generator_shapes_json(
            &json!({ "nodes": doc["nodes"], "generator": "g" }).to_string(),
        ))
        .unwrap();
        assert_eq!(r["status"], "ok");
        assert_eq!(Value::Object(shapes), r["shapes"]);
    }

    /// A hidden layer that a param links to by name still gives its outline (guide.expressions: a repeater's
    /// `shape: "@Petal"`, "hide the original"); hidden layers inside a linked group don't. A hidden link once gave
    /// nothing, so the guide's repeater example drew no petals.
    #[test]
    fn a_hidden_linked_layer_gives_its_outline_and_hidden_children_do_not() {
        let doc = json!({
            "nodes": [
                { "id": "frame", "type": "container", "x": 0, "y": 0, "width": 400, "height": 400, "rotation": 0, "scale": 1,
                  "children": ["g", "petal", "grp"] },
                { "id": "g", "type": "generator", "x": 0, "y": 0, "width": 400, "height": 400, "rotation": 0, "scale": 1,
                  "code": "std:repeater@1", "params": { "shape": "@Petal", "avoid": "@Pair" } },
                { "id": "petal", "type": "ellipse", "x": 190, "y": 80, "width": 20, "height": 120, "rotation": 0, "scale": 1,
                  "hidden": true, "metadata": { "name": "Petal" },
                  "fills": [{ "id": "fill-0", "paint": { "type": "solid", "color": "#FF5A1F" } }] },
                { "id": "grp", "type": "group", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1, "children": ["a", "b"],
                  "metadata": { "name": "Pair" } },
                { "id": "a", "type": "rectangle", "x": 10, "y": 10, "width": 20, "height": 20, "rotation": 0, "scale": 1 },
                { "id": "b", "type": "rectangle", "x": 50, "y": 10, "width": 20, "height": 20, "rotation": 0, "scale": 1, "hidden": true },
            ],
            "canvasRootNodeIds": ["frame"],
        });
        let d = crate::scene::hydrate_document(&doc);
        let g = d.nodes.iter().find(|n| n.id() == "g").unwrap();
        let shapes = linked_shapes(g.object("params").unwrap(), &d, "g");
        assert_eq!(shapes["Petal"]["paths"].as_array().map(Vec::len), Some(1), "{shapes:?}");
        assert_eq!(shapes["Pair"]["paths"].as_array().map(Vec::len), Some(1), "the hidden child is left out: {shapes:?}");
    }
}
