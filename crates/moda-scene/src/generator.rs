//! Live generator nodes (docs/design/live-nodes.md §3, §7.1): a `generator` node's drawn content is its code's
//! output, drawn as transients of the node in its local space, like a chart's parts. It comes from, in order:
//! 1. `_generatorOutput` on the fed node: the editor's generator worker evaluated it (never saved);
//! 2. the native evaluator (the `code` feature: moda-render, the CLI), from the document's code table;
//! 3. otherwise (the editor's engine before its output lands, or after an error), a placeholder: the node's box,
//!    dashed, labelled "Generator · <name>" and the status (`_generatorStatus`), so a generator is never an
//!    invisible box.

use crate::value::{json, Map, Value};

use crate::types::{RawNode, SceneDocument};

/// Where a painted output's pixels go (live-nodes §14.4): the renderer's surface registry. `(node id, width px,
/// height px, tiles)`, every non-empty tile of the surface (the sink replaces what it had). moda-engine installs it
/// (`install_scene_seams`); without one, painted output draws transparent.
pub type SurfaceSink = fn(&str, u32, u32, &[(u32, u32, &[u8])]);

static SURFACE_SINK: std::sync::OnceLock<SurfaceSink> = std::sync::OnceLock::new();

/// Install the surface sink. First call wins.
pub fn install_surface_sink(sink: SurfaceSink) {
    let _ = SURFACE_SINK.set(sink);
}

/// A generator output node as the renderer takes it: brushes named by id; a painted output (`type: "paint"`)
/// normalized to the paint-layer rectangle that samples its surface texture (`paint-surface:<w>x<h>:<id>`).
fn output_node(m: Map, doc: &SceneDocument) -> RawNode {
    let mut n = RawNode::new(with_brush_ids(m, doc));
    if n.node_type() == "paint" {
        crate::normalize::normalize_node_in_place(&mut n);
    }
    n
}

/// The transient children a generator node draws (see the module docs).
pub fn expand_generator_node(node: &RawNode, doc: &SceneDocument) -> Vec<RawNode> {
    if let Some(Value::Array(fed)) = node.json_field("_generatorOutput") {
        // The host evaluated it, and hands a painted output's tiles to the renderer itself.
        return fed.iter().filter_map(|v| v.as_object().map(|m| output_node(m.clone(), doc))).collect();
    }
    #[cfg(feature = "code")]
    {
        let r = evaluate(node, doc);
        if let Some(p) = r.paint.as_ref().filter(|_| !helper_thread()) {
            send_paint(&format!("{}/~paint", node.id()), p);
        }
        r.nodes
            .into_iter()
            .filter_map(|v| match v {
                Value::Object(m) => Some(output_node(m, doc)),
                _ => None,
            })
            .collect()
    }
    #[cfg(not(feature = "code"))]
    {
        let _ = doc;
        placeholder(node)
    }
}

thread_local! {
    static HELPER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Mark this thread as a helper that evaluates generators ahead of the render (moda-render's generator prefetch):
/// its evaluations hand no painted output to the surface sink and don't set [`take_composition_time_read`], so they
/// change nothing the frame being drawn shows. The render thread's own evaluation of the same request does both.
pub fn mark_helper_thread() {
    HELPER.with(|h| h.set(true));
}

#[cfg_attr(not(feature = "code"), allow(dead_code))]
fn helper_thread() -> bool {
    HELPER.with(std::cell::Cell::get)
}

/// Whether this thread evaluates generators ahead of the render ([`mark_helper_thread`]).
pub fn is_helper_thread() -> bool {
    helper_thread()
}

/// Hands a painted output to the sink, unless the same pixels already went to that node.
#[cfg(feature = "code")]
fn send_paint(id: &str, p: &moda_code::paint_out::PaintOutput) {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static SENT: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);
    let Some(sink) = SURFACE_SINK.get() else { return };
    {
        let mut sent = SENT.lock().unwrap_or_else(|e| e.into_inner());
        let sent = sent.get_or_insert_with(HashMap::new);
        let key = format!("{}x{}:{}", p.width_px, p.height_px, p.pixel_hash);
        if sent.get(id) == Some(&key) {
            return;
        }
        sent.insert(id.to_string(), key);
    }
    let tiles: Vec<(u32, u32, &[u8])> = p.tiles.iter().map(|t| (t.tx, t.ty, t.rgba.as_slice())).collect();
    sink(id, p.width_px, p.height_px, &tiles);
}

/// Generator output names brushes as the code did (`brush: "Bristle Flat 06"`): a stroke's `brushId` that isn't a
/// document brush id but is a document brush's name (case-insensitive) becomes that brush's id. A brush application
/// with `markWidth: true` (the stroke field: its `width` is the mark's width) has its stroke width divided by the
/// brush's mark width per unit of weight, and the flag goes. Only a bristle brush sized by weight (`scaleWithWeight`)
/// has a multiplier other than 1: its band is `size (mm) × spread` per unit (brushMarkWidth.ts in the editor).
fn with_brush_ids(mut m: Map, doc: &SceneDocument) -> Map {
    if let Some(Value::Array(strokes)) = m.get_mut("strokes") {
        for st in strokes {
            let Some(app) = st.get_mut("brush").and_then(Value::as_object_mut) else { continue };
            let mark_width = app.remove("markWidth").and_then(|v| v.as_bool()).unwrap_or(false);
            let Some(mut id) = app.get("brushId").and_then(Value::as_str).map(str::to_string) else { continue };
            if doc.brush(&id).is_none() {
                let found = doc.brushes.iter().find(|b| {
                    b.get("name").and_then(Value::as_str).is_some_and(|n| n.eq_ignore_ascii_case(&id))
                });
                if let Some(found) = found.and_then(|b| b.get("id")).and_then(Value::as_str) {
                    id = found.to_string();
                    app.insert("brushId".into(), json!(id));
                }
            }
            if !mark_width {
                continue;
            }
            let per_weight = doc
                .brush(&id)
                .and_then(crate::stroke_brush::bristle::parse_bristle_brush)
                .filter(|b| b.scale_with_weight)
                .map(|b| b.size * crate::stroke_brush::bristle::MM * b.spread());
            if let (Some(k), Some(w)) = (per_weight, st.get("width").and_then(Value::as_f64)) {
                if k > 0.0 {
                    st["width"] = json!(w / k);
                }
            }
        }
    }
    m
}

/// The dashed, labelled box a generator draws while it has no output.
pub fn placeholder(node: &RawNode) -> Vec<RawNode> {
    let w = node.f64("width").unwrap_or(0.0).max(1.0);
    let h = node.f64("height").unwrap_or(0.0).max(1.0);
    let id = node.id();
    let name = node
        .object("metadata")
        .and_then(|m| m.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("Generator");
    let status = node.str("_generatorStatus").unwrap_or("pending");
    let (label, color) = match status.strip_prefix("error: ") {
        Some(msg) => (format!("Generator · {name} — {msg}"), "#dc2626"),
        None if status == "pending" => (format!("Generator · {name}"), "#7c3aed"),
        None => (format!("Generator · {name} — {status}"), "#7c3aed"),
    };
    let mut rect = Map::new();
    rect.insert("id".into(), json!(format!("{id}_gen_placeholder")));
    rect.insert("type".into(), json!("rectangle"));
    for (k, v) in [("x", 0.0), ("y", 0.0), ("width", w), ("height", h), ("rotation", 0.0), ("scale", 1.0)] {
        rect.insert(k.into(), json!(v));
    }
    rect.insert(
        "fills".into(),
        json!([{ "id": "fill-0", "paint": { "type": "solid", "color": "#7c3aed" }, "opacity": 0.05 }]),
    );
    rect.insert(
        "strokes".into(),
        json!([{ "id": "stroke-0", "paint": { "type": "solid", "color": color }, "width": 1.0, "dash": [6.0, 4.0], "align": "inside" }]),
    );
    let size = (h / 4.0).clamp(6.0, 14.0);
    let text = label.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
    let mut t = Map::new();
    t.insert("id".into(), json!(format!("{id}_gen_placeholder_label")));
    t.insert("type".into(), json!("richtext"));
    for (k, v) in [("x", 6.0), ("y", 4.0), ("width", (w - 12.0).max(1.0)), ("height", size * 1.4), ("rotation", 0.0), ("scale", 1.0)] {
        t.insert(k.into(), json!(v));
    }
    t.insert(
        "htmlContent".into(),
        json!(format!(
            "<p style=\"font-family: Inter; font-size: {size}px; text-align: left; color: {color}\">{text}</p>"
        )),
    );
    vec![RawNode::new(rect), RawNode::new(t)]
}

/// Runs one generator evaluation. The default evaluates in this process (`moda_code::generator::evaluate`); a host
/// that must survive hostile code installs its own (moda-render: a supervised child process with a memory cap, since
/// boa can't bound its heap and an allocation failure aborts the process).
#[cfg(feature = "code")]
pub type Evaluator = fn(&moda_code::generator::GenRequest) -> moda_code::generator::GenResult;

#[cfg(feature = "code")]
static EVALUATOR: std::sync::OnceLock<Evaluator> = std::sync::OnceLock::new();

/// Install the evaluator seam (moda-render at startup). First call wins.
#[cfg(feature = "code")]
pub fn install_evaluator(evaluator: Evaluator) {
    let _ = EVALUATOR.set(evaluator);
}

/// Evaluates a generator node against the document's code table.
#[cfg(feature = "code")]
pub fn evaluate(node: &RawNode, doc: &SceneDocument) -> moda_code::generator::GenResult {
    let code_ref = node.str("code").unwrap_or_default();
    let (source, std) = if code_ref.starts_with("std:") {
        (String::new(), Some(code_ref.to_string()))
    } else {
        let Some(def) = doc.code.get(code_ref) else {
            return moda_code::generator::missing_code(code_ref);
        };
        (def.get("source").and_then(Value::as_str).unwrap_or_default().to_string(), None)
    };
    let params = node.object("params").cloned().unwrap_or_default();
    let linked_fields = linked_fields(&params, doc);
    let linked_shapes = crate::generator_shapes::linked_shapes(&params, doc, node.id());
    let req = moda_code::generator::GenRequest {
        source,
        std,
        identity: Some(code_ref.to_string()),
        params,
        linked_fields,
        linked_shapes,
        mode: Some("clamp".into()),
        size: (node.f64("width").unwrap_or(0.0).max(0.0), node.f64("height").unwrap_or(0.0).max(0.0)),
        seed: node.f64("seed").map(|s| s.max(0.0) as u32),
        // The node's pinned seed identity (#1821): editing its code keeps its random stream.
        seed_identity: node.str("seedIdentity").map(str::to_string),
        id_prefix: Some(format!("{}/", node.id())),
        ..Default::default()
    };
    // Inside a composition (moda-render's graph lane writes it per frame): the time the code sees.
    let mut req = req;
    if let Some(c) = node.object("_compositionTime") {
        req.t = c.get("t").and_then(Value::as_f64);
        req.frame = c.get("frame").and_then(Value::as_i64);
        req.time = c.get("time").cloned();
        req.rate = c.get("rate").and_then(Value::as_array).and_then(|r| {
            Some((r.first()?.as_u64()? as u32, r.get(1)?.as_u64()? as u32))
        });
        req.duration = c.get("duration").and_then(Value::as_f64);
        req.graph_seed = c.get("graphSeed").and_then(Value::as_str).map(str::to_string);
    }
    let result = match EVALUATOR.get() {
        Some(evaluator) => evaluator(&req),
        None => moda_code::generator::evaluate(&req),
    };
    if req.t.is_some() && result.read_time && !helper_thread() {
        COMPOSITION_TIME_READ.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    result
}

/// Set when a generator drawn at a composition's time (`_compositionTime`) read the time (`ctx.t`, `ctx.frame`,
/// `ctx.time`): its output differs from frame to frame even when the graph's state doesn't.
#[cfg(feature = "code")]
static COMPOSITION_TIME_READ: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether a generator evaluated at a composition's time read the time since the last call; clears it. moda-render's
/// graph lane asks after drawing each frame: such a frame is never a copy of the one before.
pub fn take_composition_time_read() -> bool {
    #[cfg(feature = "code")]
    {
        COMPOSITION_TIME_READ.swap(false, std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(feature = "code"))]
    {
        false
    }
}

/// The field outputs a generator's params link to (`"@name"`, `"@#id"`, `{ from: { layer } }`), resolved to the
/// linked `std:field` nodes' field specs (live-nodes §5.3). Keyed as the link names them.
#[cfg(feature = "code")]
pub fn linked_fields(params: &crate::value::Map, doc: &SceneDocument) -> crate::value::Map {
    let mut names = Vec::new();
    fn walk(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::String(s) if s.starts_with('@') && s.len() > 1 => out.push(s[1..].to_string()),
            Value::Object(m) => {
                if let Some(Value::Object(from)) = m.get("from") {
                    if let Some(Value::String(l)) = from.get("layer") {
                        out.push(l.clone());
                    }
                }
                for x in m.values() {
                    walk(x, out);
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    for v in params.values() {
        walk(v, &mut names);
    }
    let mut out = crate::value::Map::new();
    for name in names {
        let target = match name.strip_prefix('#') {
            Some(id) => doc.nodes.iter().find(|n| n.id() == id),
            None => doc.nodes.iter().find(|n| {
                n.id() == name
                    || n.object("metadata").and_then(|m| m.get("name")).and_then(Value::as_str) == Some(name.as_str())
            }),
        };
        if let Some(t) = target.filter(|t| t.node_type() == "generator" && t.str("code") == Some("std:field@1")) {
            let p = t.object("params").cloned().unwrap_or_default();
            out.insert(name, moda_code::stdgen::field_output(&p));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::value::json;

    fn scene_of(doc: crate::value::Value) -> crate::scene::Scene {
        crate::scene::Scene::from_document(crate::scene::hydrate_document(&doc)).unwrap()
    }

    /// #1821: the renderer seeds a generator from its pinned `seedIdentity`, as the editor does, so a node whose code
    /// changed draws the stream it had before (and an export matches the canvas).
    #[cfg(feature = "code")]
    #[test]
    fn a_pinned_seed_identity_keeps_the_stream_when_the_code_changes() {
        let src = |tag: &str| {
            format!("// {tag}\nexport default (ctx) => [{{ key: 'a', type: 'rect', x: ctx.rand() * 100, y: ctx.noise(0.3, 0) * 100, w: 1, h: 1 }}];")
        };
        let eval = |source: String, pin: Option<&str>| {
            let mut node = json!({ "id": "g", "type": "generator", "x": 0, "y": 0, "width": 50, "height": 50, "rotation": 0, "code": "b3:c" });
            if let Some(pin) = pin {
                node["seedIdentity"] = json!(pin);
            }
            let doc = crate::scene::hydrate_document(&json!({
                "nodes": [node], "canvasRootNodeIds": ["g"],
                "code": { "b3:c": { "lang": "js", "kind": "generator", "source": source } },
            }));
            let node = doc.nodes.iter().find(|n| n.id() == "g").expect("the generator").clone();
            let r = super::evaluate(&node, &doc);
            assert!(r.ok, "{:?}", r.errors);
            r
        };
        let a = eval(src("A"), None);
        assert_ne!(eval(src("B: comment only"), None).output_hash, a.output_hash, "unpinned, the code's identity seeds");
        let pin = a.seed_identity.clone().expect("reported");
        assert_eq!(eval(src("B: comment only"), Some(&pin)).output_hash, a.output_hash);
    }

    #[test]
    fn fed_output_draws_as_transients() {
        let scene = scene_of(json!({
            "nodes": [{ "id": "g", "type": "generator", "x": 0, "y": 0, "width": 50, "height": 50, "rotation": 0, "code": "b3:x",
                        "_generatorOutput": [{ "id": "g/~ellipse0", "type": "ellipse", "x": 1, "y": 2, "width": 3, "height": 4, "rotation": 0, "scale": 1 }] }],
            "canvasRootNodeIds": ["g"],
        }));
        let kids = scene.transient_by_parent.get("g").cloned().unwrap_or_default();
        assert_eq!(kids, vec!["g/~ellipse0".to_string()]);
    }

    #[test]
    fn a_mark_width_stroke_is_sized_for_a_weight_scaled_bristle_brush() {
        let doc = crate::scene::hydrate_document(&json!({
            "nodes": [], "canvasRootNodeIds": [],
            "brushes": [
                { "id": "b-weight", "name": "Wide", "type": "bristle", "shape": "flat-point", "size": 4, "length": 1, "scaleWithWeight": true },
                { "id": "b-width", "name": "Narrow", "type": "bristle", "shape": "round-point", "size": 4, "length": 1 },
            ],
        }));
        let stroke = |name: &str, mark: bool| {
            let mut brush = json!({ "brushId": name, "seed": 2 });
            if mark {
                brush["markWidth"] = json!(true);
            }
            json!({ "id": "n", "type": "path", "strokes": [{ "id": "stroke-0", "width": 12.0, "brush": brush }] })
                .as_object()
                .unwrap()
                .clone()
        };
        let out = |name: &str, mark: bool| super::with_brush_ids(stroke(name, mark), &doc)["strokes"][0].clone();
        // Sized by weight: the band is size (mm) × spread per unit, so a 12 px mark is 12 / that.
        let wide = out("wide", true);
        let per_weight = 4.0 * crate::stroke_brush::bristle::MM * (0.7 + 0.5) * 1.15;
        assert_eq!(wide["brush"], json!({ "brushId": "b-weight", "seed": 2 }), "resolved, and the flag goes");
        assert!((wide["width"].as_f64().unwrap() - 12.0 / per_weight).abs() < 1e-9, "{}", wide["width"]);
        // Sized by the stroke width, or no flag: the width is left alone.
        assert_eq!(out("narrow", true)["width"], 12.0);
        assert_eq!(out("narrow", true)["brush"], json!({ "brushId": "b-width", "seed": 2 }));
        assert_eq!(out("wide", false)["width"], 12.0);
    }

    #[test]
    fn placeholder_is_labelled() {
        let node = crate::types::RawNode::new(
            json!({ "id": "g", "type": "generator", "width": 200, "height": 100, "metadata": { "name": "Sky" },
                    "_generatorStatus": "error: code_error: boom" })
            .as_object()
            .unwrap()
            .clone(),
        );
        let p = super::placeholder(&node);
        assert_eq!(p.len(), 2);
        assert!(p[1].str("htmlContent").unwrap().contains("Generator · Sky — code_error: boom"));
    }

    #[cfg(feature = "code")]
    #[test]
    fn a_generator_draws_its_output() {
        let src = "export const schema = { n: { type: 'int', default: 2 } };\nexport default (ctx, p) => Array.from({ length: p.n }, (_, i) => ({ type: 'rect', x: i * 10, y: 0, w: 5, h: 5, fill: '#ff0000' }));";
        let scene = scene_of(json!({
            "nodes": [{ "id": "g", "type": "generator", "x": 100, "y": 50, "width": 200, "height": 100, "rotation": 0,
                        "code": "b3:test", "params": { "n": 3 } }],
            "canvasRootNodeIds": ["g"],
            "code": { "b3:test": { "lang": "js", "kind": "generator", "source": src } },
        }));
        let kids = scene.transient_by_parent.get("g").cloned().unwrap_or_default();
        assert_eq!(kids.len(), 3, "{kids:?}");
        let n = &scene.transient_by_id[&kids[2]];
        assert_eq!(n.node_type(), "rectangle");
        assert_eq!(n.f64("x"), Some(20.0));
    }

    #[cfg(feature = "code")]
    #[test]
    fn std_field_nodes_draw_and_are_linked_by_name() {
        let src = "export const schema = { w: { type: 'number', default: 1, min: 0, max: 100, fieldable: true } };\nexport default (ctx, p) => [{ type: 'rect', x: 0, y: 0, w: p.w.at(0, 0), h: 5, fill: '#000' }];";
        let scene = scene_of(json!({
            "nodes": [
                { "id": "f", "type": "generator", "x": 0, "y": 0, "width": 100, "height": 100, "rotation": 0,
                  "code": "std:field@1", "params": { "field": 0.25, "cell": 50 }, "metadata": { "name": "Flow" } },
                { "id": "g", "type": "generator", "x": 0, "y": 0, "width": 100, "height": 100, "rotation": 0,
                  "code": "b3:w", "params": { "w": { "field": "@Flow", "remap": [0, 40] } } },
            ],
            "canvasRootNodeIds": ["f", "g"],
            "code": { "b3:w": { "lang": "js", "kind": "generator", "source": src } },
        }));
        // The field node draws its 2 × 2 preview cells.
        assert_eq!(scene.transient_by_parent.get("f").map(Vec::len), Some(4));
        // The linked generator reads the field: 0.25 remapped to 0..40 is 10.
        let kids = scene.transient_by_parent.get("g").cloned().unwrap_or_default();
        assert_eq!(scene.transient_by_id[&kids[0]].f64("width"), Some(10.0));
    }

    #[cfg(feature = "code")]
    #[test]
    fn two_identical_generators_each_draw() {
        // Same code, params, seed and box: one cached evaluation, but each node's parts have their own ids (the
        // first generator drew nothing when the second reused its ids).
        let src = "export default () => [{ type: 'rect', x: 0, y: 0, w: 5, h: 5, fill: '#ff0000' }];";
        let g = |id: &str, x: f64| json!({ "id": id, "type": "generator", "x": x, "y": 0, "width": 50, "height": 50, "rotation": 0, "code": "b3:same" });
        let doc = json!({
            "nodes": [g("g1", 0.0), g("g2", 100.0)],
            "canvasRootNodeIds": ["g1", "g2"],
            "code": { "b3:same": { "lang": "js", "kind": "generator", "source": src } },
        });
        let scene = crate::scene::Scene::from_document(crate::scene::hydrate_document(&doc)).unwrap();
        let a = scene.transient_by_parent.get("g1").cloned().unwrap_or_default();
        let b = scene.transient_by_parent.get("g2").cloned().unwrap_or_default();
        assert_eq!((a.len(), b.len()), (1, 1), "{a:?} {b:?}");
        assert_ne!(a[0], b[0]);
        assert!(scene.transient_by_id.contains_key(&a[0]) && scene.transient_by_id.contains_key(&b[0]));
    }

    /// A nodes-mode generator (its output stored as children, as `generator.create` makes it) drawn at a
    /// composition's time: the timed output draws, not the stored design, with the stored child before or after
    /// the host in the document's node list.
    #[cfg(feature = "code")]
    #[test]
    fn a_nodes_mode_generator_at_a_composition_time_draws_its_timed_output() {
        let src = "export default (ctx) => [{ key: 'dot', type: 'rect', x: 20 + 240 * ctx.t, y: 60, w: 40, h: 40, fill: '#ffffff' }];";
        let host = json!({ "id": "g", "type": "generator", "x": 0, "y": 0, "width": 320, "height": 180, "rotation": 0, "code": "b3:t",
                           "children": ["g/dot"], "_compositionTime": { "t": 0.5, "frame": 5, "rate": [10, 1], "duration": 1.0 } });
        let stored = json!({ "id": "g/dot", "type": "rectangle", "x": 20, "y": 60, "width": 40, "height": 40, "rotation": 0, "scale": 1, "genKey": "dot" });
        for nodes in [vec![host.clone(), stored.clone()], vec![stored.clone(), host.clone()]] {
            let scene = scene_of(json!({
                "nodes": nodes,
                "canvasRootNodeIds": ["g"],
                "code": { "b3:t": { "lang": "js", "kind": "generator", "source": src } },
            }));
            assert_eq!(scene.transient_by_parent.get("g").cloned().unwrap_or_default(), vec!["g/dot".to_string()]);
            let drawn = scene.effective_node("g/dot").unwrap();
            assert_eq!(drawn.f64("x"), Some(140.0), "the output at 0.5 s, not the stored x 20: {drawn:?}");
        }
        // Outside a composition the stored design draws.
        let mut plain = host.clone();
        plain.as_object_mut().unwrap().remove("_compositionTime");
        let scene = scene_of(json!({ "nodes": [plain, stored], "canvasRootNodeIds": ["g"],
                                     "code": { "b3:t": { "lang": "js", "kind": "generator", "source": src } } }));
        assert_eq!(scene.effective_node("g/dot").unwrap().f64("x"), Some(20.0));
    }

    #[cfg(feature = "code")]
    #[test]
    fn stroke_field_avoids_a_linked_layer_and_follows_it() {
        let doc = |fig_x: f64| {
            json!({
                "nodes": [
                    { "id": "frame", "type": "container", "x": 0, "y": 0, "width": 400, "height": 200, "rotation": 0, "scale": 1,
                      "children": ["s", "fig"] },
                    { "id": "s", "type": "generator", "x": 0, "y": 0, "width": 400, "height": 200, "rotation": 0, "code": "std:stroke-field@1",
                      "params": { "direction": 0, "length": [8, 12], "width": [2, 3], "density": 2, "avoid": "@Figure", "seed": 1 } },
                    { "id": "fig", "type": "rectangle", "x": fig_x, "y": 50, "width": 80, "height": 100, "rotation": 0, "scale": 1,
                      "metadata": { "name": "Figure" } },
                ],
                "canvasRootNodeIds": ["frame"],
            })
        };
        let marks = |fig_x: f64| {
            let scene = crate::scene::Scene::from_document(crate::scene::hydrate_document(&doc(fig_x))).unwrap();
            let kids = scene.transient_by_parent.get("s").cloned().unwrap_or_default();
            kids.iter()
                .map(|k| {
                    let n = &scene.transient_by_id[k];
                    (n.f64("x").unwrap(), n.f64("y").unwrap(), n.f64("width").unwrap(), n.f64("height").unwrap())
                })
                .collect::<Vec<_>>()
        };
        let a = marks(100.0);
        assert!(a.len() > 100, "{} marks", a.len());
        // No mark inside the figure (a mark stops at its edge; its box may touch it).
        let inside = |(x, y, w, h): (f64, f64, f64, f64), fx: f64| x > fx + 1.0 && x + w < fx + 79.0 && y > 51.0 && y + h < 149.0;
        assert!(!a.iter().any(|&m| inside(m, 100.0)));
        // Move the figure: the marks part around it where it is now.
        let b = marks(250.0);
        assert!(!b.iter().any(|&m| inside(m, 250.0)));
        assert!(b.iter().any(|&m| inside(m, 100.0)), "the old place fills in");
    }

    #[cfg(feature = "code")]
    #[test]
    fn stroke_field_marks_use_the_document_brush_named() {
        let doc = json!({
            "nodes": [{ "id": "s", "type": "generator", "x": 0, "y": 0, "width": 200, "height": 100, "rotation": 0,
                        "code": "std:stroke-field@1",
                        "params": { "direction": 0, "length": [10, 20], "width": [3, 4], "brush": "soft liner seed:2", "seed": 1 } }],
            "canvasRootNodeIds": ["s"],
            "brushes": [{ "id": "b-liner", "name": "Soft Liner", "type": "calligraphic", "angle": { "value": 30 },
                          "roundness": { "value": 0.4 }, "size": { "value": 2 } }],
        });
        let scene = crate::scene::Scene::from_document(crate::scene::hydrate_document(&doc)).unwrap();
        let kids = scene.transient_by_parent.get("s").cloned().unwrap_or_default();
        assert!(kids.len() > 20, "{} marks", kids.len());
        let first = &scene.transient_by_id[&kids[0]];
        let brush = first.get("strokes").and_then(|s| s.get(0)).and_then(|s| s.get("brush")).cloned();
        assert_eq!(brush, Some(json!({ "brushId": "b-liner", "seed": 2.0 })));
    }
}
