//! Area type from the scene (E8.14): a richtext node's `moda_text::area::AreaSpec` — its own outline and Area
//! Type Options (`area`), the frames before it in its thread (`threadPrev`), and the Text Wrap objects above it
//! (`wrapsText`), each in the frame's own local px.
//!
//! The editor builds the same input from its scene for the edit overlay and the caret
//! (`utils/text/areaTextResolve.ts`); both hand it to moda-text, which does all the geometry.
//!
//! Model (see `RichTextNode` in `core/scene/nodes.ts`):
//! - `area: { points?, boxWidth?, boxHeight?, inset?, firstBaseline?, minBaseline?, rows?, rowGutter?, columns?,
//!   columnGutter?, flow? }` makes a text node area type: its text fits the outline (`points`, drawn at `boxWidth`
//!   × `boxHeight` and scaled with the node; the box without them) and never spills out of it.
//! - `threadPrev: <id>` on an area text node continues the story of that one; the story's text is the thread
//!   head's `htmlContent`.
//! - `wrapsText: { offset?, invert? }` on any layer: area text and box text below it in stacking order flow around
//!   its outline (a group's members' outlines; a box for layers without one).

use moda_scene::types::{Affine, RawNode};
use moda_scene::Scene;
use moda_text::area::{AreaFrame, AreaSpec, AreaVAlign, FirstBaseline, Ring, WrapExclusion};
use moda_text::text_path::PathCmd;
use moda_scene::value::Value;

/// Illustrator's default Text Wrap offset (6 pt).
pub const DEFAULT_WRAP_OFFSET: f64 = 6.0;
/// The longest thread followed.
const MAX_THREAD: usize = 256;

fn is_area(node: &RawNode) -> bool {
    node.node_type() == "richtext"
        && node.json_field("area").is_some_and(Value::is_object)
        && node.json_field("textPath").is_none()
}

/// Whether text in this node wraps at all (not auto-width, not along a path).
fn wraps(node: &RawNode) -> bool {
    node.node_type() == "richtext"
        && !crate::text_draw::node_auto_width(node)
        && node.json_field("textPath").is_none()
}

/// The thread up to and including `node`, head first. A frame whose `threadPrev` is missing, not area type or a
/// loop starts its own story.
pub fn thread_chain<'s>(scene: &'s Scene, node: &'s RawNode) -> Vec<&'s RawNode> {
    let mut chain: Vec<&RawNode> = vec![node];
    if !is_area(node) {
        return chain;
    }
    let mut cur = node;
    while let Some(prev) = cur.str("threadPrev").filter(|s| !s.is_empty()) {
        if chain.len() >= MAX_THREAD || chain.iter().any(|n| n.id() == prev) {
            break;
        }
        let Some(p) = scene.effective_node(prev) else {
            break;
        };
        if !is_area(p) {
            break;
        }
        chain.push(p);
        cur = p;
    }
    chain.reverse();
    chain
}

/// The story's HTML for `node`: its thread head's (its own when it heads the thread or has none).
pub fn story_html<'s>(scene: &'s Scene, node: &'s RawNode) -> &'s str {
    thread_chain(scene, node)
        .first()
        .and_then(|h| h.str("htmlContent"))
        .unwrap_or("")
}

/// The node's area layout input, or `None` for box text (no `area`, no wrap object reaching it).
pub fn node_area_spec(scene: &Scene, node: &RawNode) -> Option<AreaSpec> {
    if !wraps(node) {
        return None;
    }
    let area = is_area(node);
    if !area && scene.text_flow().wrap_ids.is_empty() {
        return None;
    }
    let chain = thread_chain(scene, node);
    let frames: Vec<AreaFrame> = chain
        .iter()
        .map(|n| frame_for(scene, n, chain.len() > 1))
        .collect();
    if !area && frames.last().is_none_or(|f| f.exclusions.is_empty()) {
        return None;
    }
    let index = frames.len() - 1;
    Some(AreaSpec { frames, index })
}

/// One frame of a thread: the node's box, outline, options and wrap objects.
fn frame_for(scene: &Scene, n: &RawNode, threaded: bool) -> AreaFrame {
    let w = n
        .f64("width")
        .filter(|v| v.is_finite())
        .unwrap_or(400.0)
        .max(0.0);
    let h = n
        .f64("height")
        .filter(|v| v.is_finite())
        .unwrap_or(200.0)
        .max(0.0);
    let exclusions = wrap_exclusions(scene, n);
    let Some(a) = n.json_field("area").filter(|a| a.is_object() && is_area(n)) else {
        // Box text a wrap object reaches: it keeps box text's placement and never clips (it grows or spills).
        let mut f = AreaFrame::rect(w, f64::INFINITY);
        f.options.first_baseline = FirstBaseline::HalfLeading;
        f.exclusions = exclusions;
        return f;
    };
    let mut options = moda_text::area::options_from_json(&a.to_json());
    options.valign = if a.get("verticalJustify").and_then(Value::as_bool) == Some(true) {
        AreaVAlign::Justify
    } else {
        AreaVAlign::parse(n.str("verticalAlign").unwrap_or("top"))
    };
    let num = |k: &str| a.get(k).and_then(Value::as_f64).filter(|v| v.is_finite());
    let shape = a
        .get("points")
        .and_then(Value::as_array)
        .and_then(|points| {
            let sx = match num("boxWidth") {
                Some(bw) if bw > 1e-9 && w > 0.0 => w / bw,
                _ => 1.0,
            };
            let sy = match num("boxHeight") {
                Some(bh) if bh > 1e-9 && h > 0.0 => h / bh,
                _ => 1.0,
            };
            let rings = moda_text::area::path_points_to_rings(&points.iter().map(Value::to_json).collect::<Vec<_>>(), sx, sy);
            (!rings.is_empty()).then_some(rings)
        });
    // Box area type in its own (unthreaded) frame follows "Auto height" and grows downward.
    let auto_height = shape.is_none() && !threaded && n.str("resizeMode") == Some("resize-height");
    AreaFrame {
        width: w,
        height: if auto_height { f64::INFINITY } else { h },
        shape,
        exclusions,
        options,
    }
}

fn wrap_settings(node: &RawNode) -> Option<(f64, bool)> {
    let tw = node.object("wrapsText")?;
    let offset = tw
        .get("offset")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .unwrap_or(DEFAULT_WRAP_OFFSET)
        .max(0.0);
    let invert = tw.get("invert").and_then(Value::as_bool).unwrap_or(false);
    Some((offset, invert))
}

/// The Text Wrap objects above `text` that reach it, in its local px.
pub fn wrap_exclusions(scene: &Scene, text: &RawNode) -> Vec<WrapExclusion> {
    let mut out = Vec::new();
    if scene.text_flow().wrap_ids.is_empty() || !wraps(text) {
        return out;
    }
    let Some(text_aff) = scene.node_canvas_affine(text.id()) else {
        return out;
    };
    let Some(inv) = invert(text_aff) else {
        return out;
    };
    let w = text.f64("width").unwrap_or(0.0).max(0.0);
    let h = if text.json_field("area").is_some() {
        text.f64("height").unwrap_or(0.0).max(0.0)
    } else {
        // Box text reaches down as far as it needs.
        1.0e7
    };
    let text_box = moda_scene::transform::transformed_aabb((0.0, 0.0, w, h), text_aff);
    for wid in &scene.text_flow().wrap_ids {
        if wid == text.id() {
            continue;
        }
        let Some(wn) = scene.effective_node(wid) else {
            continue;
        };
        let Some((offset, invert_wrap)) = wrap_settings(wn) else {
            continue;
        };
        if wn.truthy("hidden")
            || wn.f64("opacity") == Some(0.0)
            || !paints_above(scene, wid, text.id())
        {
            continue;
        }
        let rings = canvas_rings(scene, wid, 0);
        if rings.is_empty() {
            continue;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for &(x, y) in rings.iter().flatten() {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        // An inverted wrap confines the text wherever it is; a plain one only matters where it overlaps.
        let pad = offset;
        let overlaps = x0 - pad < text_box.0 + text_box.2
            && x1 + pad > text_box.0
            && y0 - pad < text_box.1 + text_box.3
            && y1 + pad > text_box.1;
        if !overlaps && !invert_wrap {
            continue;
        }
        let local: Vec<Ring> = rings
            .into_iter()
            .map(|r| {
                r.into_iter()
                    .map(|p| moda_scene::transform::apply_point(p, inv))
                    .collect()
            })
            .collect();
        out.push(WrapExclusion {
            rings: local,
            offset,
            invert: invert_wrap,
        });
    }
    out
}

fn invert(a: Affine) -> Option<Affine> {
    let det = a.a * a.d - a.b * a.c;
    if det.abs() < 1e-12 || !det.is_finite() {
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

/// A layer's outline as canvas-px rings: a shape's or path's outline (a boolean's result), a group's members',
/// otherwise its box.
fn canvas_rings(scene: &Scene, id: &str, depth: usize) -> Vec<Ring> {
    let Some(node) = scene.effective_node(id) else {
        return Vec::new();
    };
    if depth > 32 {
        return Vec::new();
    }
    if node.node_type() == "group" {
        return scene
            .children_by_parent()
            .get(id)
            .map(|kids| {
                kids.iter()
                    .flat_map(|k| canvas_rings(scene, k, depth + 1))
                    .collect()
            })
            .unwrap_or_default();
    }
    let Some(aff) = scene.node_canvas_affine(id) else {
        return Vec::new();
    };
    let local: Vec<Ring> = match moda_scene::shape_outline::node_outline(node) {
        Some((points, _)) if points.len() >= 2 => {
            use moda_scene::blend::BlendPathPoint as B;
            let cmds: Vec<PathCmd> = points
                .iter()
                .enumerate()
                .map(|(i, p)| match *p {
                    B::Move { x, y } => PathCmd::Move(x, y),
                    _ if i == 0 => {
                        let (x, y) = end_of(p);
                        PathCmd::Move(x, y)
                    }
                    B::Line { x, y } | B::Arc { x, y, .. } => PathCmd::Line(x, y),
                    B::Curve { x, y, cp1, cp2 } => PathCmd::Cubic(cp1.0, cp1.1, cp2.0, cp2.1, x, y),
                    B::Quadratic { x, y, cp } => PathCmd::Quad(cp.0, cp.1, x, y),
                })
                .collect();
            moda_text::area::cmds_to_rings(&cmds)
        }
        _ => {
            let w = node.f64("width").unwrap_or(0.0);
            let h = node.f64("height").unwrap_or(0.0);
            if w > 0.0 && h > 0.0 {
                vec![vec![(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]]
            } else {
                Vec::new()
            }
        }
    };
    local
        .into_iter()
        .map(|r| {
            r.into_iter()
                .map(|p| moda_scene::transform::apply_point(p, aff))
                .collect()
        })
        .collect()
}

fn end_of(p: &moda_scene::blend::BlendPathPoint) -> (f64, f64) {
    use moda_scene::blend::BlendPathPoint as B;
    match *p {
        B::Move { x, y } | B::Line { x, y } | B::Arc { x, y, .. } => (x, y),
        B::Curve { x, y, .. } | B::Quadratic { x, y, .. } => (x, y),
    }
}

/// `id` and its ancestors, outermost first.
fn ancestry(scene: &Scene, id: &str) -> Vec<String> {
    let mut chain = vec![id.to_string()];
    let mut cur = id.to_string();
    while let Some(p) = scene.parent_map().get(&cur) {
        if chain.len() > 256 || chain.contains(p) {
            break;
        }
        chain.push(p.clone());
        cur = p.clone();
    }
    chain.reverse();
    chain
}

/// Whether `a` paints after (above) `b` in the same page or canvas.
pub fn paints_above(scene: &Scene, a: &str, b: &str) -> bool {
    let (ca, cb) = (ancestry(scene, a), ancestry(scene, b));
    let mut k = 0;
    while k < ca.len() && k < cb.len() && ca[k] == cb[k] {
        k += 1;
    }
    if k == ca.len() {
        return false; // `a` holds `b` (or is it): a parent paints first
    }
    if k == cb.len() {
        return true; // `b` holds `a`
    }
    let siblings: Vec<String> = if k == 0 {
        let (pa, pb) = (scene.page_of(&ca[0]), scene.page_of(&cb[0]));
        if pa != pb {
            return false;
        }
        match pa {
            Some(page) => scene.page_top_level_ids(&page),
            None => scene.doc().canvas_root_node_ids.clone(),
        }
    } else {
        let parent = &ca[k - 1];
        let mut kids = scene
            .children_by_parent()
            .get(parent)
            .cloned()
            .unwrap_or_default();
        if scene
            .canonical_or_transient(parent)
            .is_some_and(moda_scene::scene::first_child_on_top)
        {
            kids.reverse();
        }
        kids
    };
    let ia = siblings.iter().position(|s| *s == ca[k]);
    let ib = siblings.iter().position(|s| *s == cb[k]);
    matches!((ia, ib), (Some(ia), Some(ib)) if ia > ib)
}

/// The text nodes whose layout may change when `touched` nodes change: every wrappable text node when a Text
/// Wrap object is among them, and the later frames of the threads they belong to. The retained plan re-plans
/// these with the touched nodes.
pub fn text_dependents<'a>(
    scene: &Scene,
    touched: impl IntoIterator<Item = &'a str>,
) -> Vec<String> {
    let flow = &scene.text_flow();
    if flow.wrap_ids.is_empty() && flow.thread_ids.is_empty() {
        return Vec::new();
    }
    let touched: Vec<&str> = touched.into_iter().collect();
    let mut out: Vec<String> = Vec::new();
    let wrap_touched = touched.iter().any(|id| {
        flow.wrap_ids
            .binary_search_by(|x| x.as_str().cmp(id))
            .is_ok()
            || scene
                .effective_node(id)
                .is_some_and(|n| n.node_type() == "group")
    });
    if wrap_touched && !flow.wrap_ids.is_empty() {
        for n in &scene.doc().nodes {
            if wraps(n) && !touched.contains(&n.id()) {
                out.push(n.id().to_string());
            }
        }
    }
    if !flow.thread_ids.is_empty() {
        for fid in &flow.thread_ids {
            let Some(f) = scene.effective_node(fid) else {
                continue;
            };
            let chain = thread_chain(scene, f);
            if chain
                .iter()
                .any(|n| n.id() != fid && touched.contains(&n.id()))
                && !touched.contains(&fid.as_str())
            {
                out.push(fid.clone());
            }
        }
    }
    // Text wrapped around a thread member that moved is covered by the wrap scan; a text node that moved under a
    // wrap object is touched itself.
    out.sort();
    out.dedup();
    out
}

/// A stable hash of a node's area input (the text cache keys on it: a wrap object that moved, or an earlier frame
/// of the thread that changed, re-lays the text out).
pub fn area_key(spec: Option<&AreaSpec>, story_html: Option<&str>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    if let Some(spec) = spec {
        spec.index.hash(&mut h);
        for f in &spec.frames {
            for v in [
                f.width,
                f.height,
                f.options.inset,
                f.options.min_baseline,
                f.options.row_gutter,
                f.options.column_gutter,
            ] {
                v.to_bits().hash(&mut h);
            }
            (f.options.rows, f.options.columns).hash(&mut h);
            format!(
                "{:?}{:?}{:?}",
                f.options.first_baseline, f.options.flow, f.options.valign
            )
            .hash(&mut h);
            for r in f.shape.iter().flatten() {
                for &(x, y) in r {
                    (x.to_bits(), y.to_bits()).hash(&mut h);
                }
                u64::MAX.hash(&mut h);
            }
            for e in &f.exclusions {
                (e.offset.to_bits(), e.invert).hash(&mut h);
                for r in &e.rings {
                    for &(x, y) in r {
                        (x.to_bits(), y.to_bits()).hash(&mut h);
                    }
                    u64::MAX.hash(&mut h);
                }
            }
        }
    }
    story_html.hash(&mut h);
    h.finish()
}

/// Fill in a text spec's area input and, for a later frame of a thread, the story it shows.
pub fn apply_to_spec(scene: &Scene, node: &RawNode, spec: &mut moda_text::TextNodeSpec) {
    spec.area = node_area_spec(scene, node);
    if spec.area.as_ref().is_some_and(|a| a.frames.len() > 1) {
        spec.html = moda_text::lang::html_with_document_lang(
            story_html(scene, node),
            crate::text_draw::document_locale(scene),
        ).into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    fn scene(nodes: Value, page_nodes: Value) -> Scene {
        Scene::load(&json!({
            "pages": [{ "id": "p", "nodeIds": page_nodes, "width": 800, "height": 800 }],
            "nodes": nodes,
        })).unwrap()
    }

    fn text(id: &str, x: f64, y: f64, extra: Value) -> Value {
        let mut n = json!({
            "id": id, "type": "richtext", "x": x, "y": y, "width": 200, "height": 100, "rotation": 0, "scale": 1,
            "htmlContent": format!("<p>{id}</p>"),
        });
        for (k, v) in extra.as_object().unwrap() {
            n[k] = v.clone();
        }
        n
    }

    #[test]
    fn box_text_without_wrap_objects_is_not_area_type() {
        let s = scene(json!([text("t", 0.0, 0.0, json!({}))]), json!(["t"]));
        assert!(node_area_spec(&s, s.node("t").unwrap()).is_none());
        assert!(s.text_flow().wrap_ids.is_empty());
    }

    #[test]
    fn a_thread_lists_its_frames_head_first_and_shows_the_head_story() {
        let s = scene(
            json!([
                text("a", 0.0, 0.0, json!({ "area": {} })),
                text(
                    "b",
                    300.0,
                    0.0,
                    json!({ "area": {}, "threadPrev": "a", "htmlContent": "" })
                ),
                text(
                    "c",
                    600.0,
                    0.0,
                    json!({ "area": {}, "threadPrev": "b", "htmlContent": "" })
                ),
            ]),
            json!(["a", "b", "c"]),
        );
        let c = s.node("c").unwrap();
        let chain: Vec<&str> = thread_chain(&s, c).iter().map(|n| n.id()).collect();
        assert_eq!(chain, ["a", "b", "c"]);
        assert_eq!(story_html(&s, c), "<p>a</p>");
        let spec = node_area_spec(&s, c).unwrap();
        assert_eq!((spec.frames.len(), spec.index), (3, 2));
        assert_eq!(s.text_flow().thread_ids, ["b", "c"]);
        // Editing the head re-lays out the rest of the thread.
        assert_eq!(text_dependents(&s, ["a"]), ["b", "c"]);
        assert_eq!(text_dependents(&s, ["b"]), ["c"]);
    }

    #[test]
    fn a_wrap_object_above_the_text_cuts_it_in_the_text_local_space() {
        let s = scene(
            json!([
                text("t", 100.0, 100.0, json!({ "area": {} })),
                { "id": "r", "type": "rectangle", "x": 150.0, "y": 120.0, "width": 50, "height": 40, "rotation": 0,
                  "scale": 1, "wrapsText": { "offset": 4 } },
                { "id": "under", "type": "rectangle", "x": 150.0, "y": 120.0, "width": 50, "height": 40,
                  "rotation": 0, "scale": 1, "wrapsText": {} },
            ]),
            json!(["under", "t", "r"]),
        );
        assert!(paints_above(&s, "r", "t") && !paints_above(&s, "under", "t"));
        let spec = node_area_spec(&s, s.node("t").unwrap()).unwrap();
        let ex = &spec.frames[0].exclusions;
        assert_eq!(ex.len(), 1, "only the object above the text");
        assert_eq!(ex[0].offset, 4.0);
        let xs: Vec<f64> = ex[0].rings[0].iter().map(|p| p.0).collect();
        let ys: Vec<f64> = ex[0].rings[0].iter().map(|p| p.1).collect();
        assert!((xs.iter().cloned().fold(f64::MAX, f64::min) - 50.0).abs() < 1e-9);
        assert!((ys.iter().cloned().fold(f64::MAX, f64::min) - 20.0).abs() < 1e-9);
        // Box text below it wraps too, keeping box text's placement.
        let s2 = scene(
            json!([
                text("t", 100.0, 100.0, json!({})),
                { "id": "r", "type": "ellipse", "x": 150.0, "y": 120.0, "width": 50, "height": 40, "rotation": 0,
                  "scale": 1, "wrapsText": {} },
            ]),
            json!(["t", "r"]),
        );
        let spec = node_area_spec(&s2, s2.node("t").unwrap()).unwrap();
        assert_eq!(
            spec.frames[0].options.first_baseline,
            FirstBaseline::HalfLeading
        );
        assert!(spec.frames[0].height.is_infinite());
        assert_eq!(spec.frames[0].exclusions[0].offset, DEFAULT_WRAP_OFFSET);
        // Moving the wrap object re-lays the text out.
        assert_eq!(text_dependents(&s2, ["r"]), ["t"]);
    }
}
