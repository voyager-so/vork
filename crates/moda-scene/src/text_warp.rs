//! Live geometry effects on text layers (Warp and the rest of E8.1, Illustrator's Effect ▸ Warp on type).
//!
//! A rich text node with a visible geometry effect draws as a path: its glyph outlines, laid out by the vector
//! text engine exactly as the text lane would draw them, run through [`crate::live_effects::apply_effects_to_copies`]
//! as ONE piece of art (the warp measures the whole layer's bounds). The resolved record is a `path` with the node's
//! id, transform and drawing effects, so every lane that draws a path draws it: the canvas, PNG / PDF export,
//! shadows, masks. The canonical node stays rich text, so the text is still edited as text (the editor blanks
//! the text while its DOM editor covers it, so the unwarped text shows while editing, as in Illustrator's
//! envelope edit, and re-warps when editing ends).
//!
//! A text layer whose glyph fills include a vector pattern (E8.5) draws as its outlines the same way, with no
//! effect applied: the pattern paint then draws in the path's paint stack, clipped to the glyphs.
//!
//! Paint: each distinct run fill is one outline group, drawn with its own fill stack, one copy per group
//! ([`crate::live_effects::EFFECT_COPY_FILLS_KEY`]). A `textFills` stack paints every glyph, so it is one group.
//!
//! The glyph outlines come through the [`TextOutliner`] seam, which moda-engine installs (the text layout lives
//! there). It answers [`TextOutlines::Pending`] when the text cannot be outlined yet (its fonts are not installed):
//! the node then draws as plain text, and [`Scene::retry_pending_text_warps`] resolves it once they arrive.

use crate::blend::BlendPathPoint as PathPoint;
use crate::live_effects::{
    apply_effects_to_copies, contours_from_points, geometry_effects, has_geometry_effects,
    resolved_map, EFFECT_COPIES_KEY, EFFECT_COPY_FILLS_KEY,
};
use crate::scene::Scene;
use crate::types::RawNode;
use crate::value::Value;
use std::sync::OnceLock;

/// One paint's glyph outlines, node-local px.
#[derive(Debug, Clone)]
pub struct TextOutlineGroup {
    pub points: Vec<PathPoint>,
    /// The fill stack these glyphs draw with (a `fills` array).
    pub fills: Value,
}

/// What the outliner could make of a text node.
#[derive(Debug, Clone)]
pub enum TextOutlines {
    /// The glyph outlines, one group per fill (empty for empty text).
    Ready(Vec<TextOutlineGroup>),
    /// Not yet: the fonts it needs are not installed (or the text does not lay out through the vector lane).
    Pending,
}

/// A rich text node's glyph outlines (see the module docs).
pub type TextOutliner = fn(&Scene, &RawNode) -> TextOutlines;

static OUTLINER: OnceLock<TextOutliner> = OnceLock::new();

/// Install the text seam (moda-engine's `install_scene_seams`). First call wins.
pub fn install_text_outliner(outliner: TextOutliner) {
    let _ = OUTLINER.set(outliner);
}

/// The installed text seam, if any (`crate::envelope` outlines the text it bends).
pub fn outliner() -> Option<TextOutliner> {
    OUTLINER.get().copied()
}

/// Whether `node` is a text layer that draws as its glyph outlines: a geometry effect reshapes it, or a vector
/// pattern paints its glyphs (E8.5: the pattern is clipped to the outlines, exact at any zoom, which the text lane's
/// one glyph paint can't do).
pub fn is_effected_text(node: &RawNode) -> bool {
    node.node_type() == "richtext" && (has_geometry_effects(node) || has_pattern_text_fill(node))
}

/// Whether a rich text node's visible glyph fills (`textFills`) include a vector pattern.
pub fn has_pattern_text_fill(node: &RawNode) -> bool {
    node.node_type() == "richtext"
        && crate::paint::parse_fills(node.json_field("textFills"))
            .iter()
            .any(|f| crate::paint::shown(f.visible, f.opacity) && matches!(f.paint, crate::paint::Paint::Pattern(_)))
}

/// What [`resolve`] made of a node.
#[derive(Debug, Clone)]
pub enum TextWarp {
    /// Not an effected text layer: it draws as itself.
    None,
    /// Its outlines are not ready yet: it draws as plain text for now.
    Pending,
    /// The path it draws as.
    Resolved(RawNode),
}

/// The path record an effected text layer draws as (see the module docs).
pub fn resolve(scene: &Scene, node: &RawNode) -> TextWarp {
    if !is_effected_text(node) {
        return TextWarp::None;
    }
    let Some(outliner) = OUTLINER.get() else {
        return TextWarp::Pending;
    };
    let groups = match outliner(scene, node) {
        TextOutlines::Ready(groups) => groups,
        TextOutlines::Pending => return TextWarp::Pending,
    };
    TextWarp::Resolved(record(node, &groups))
}

/// [`resolve`] for a text node on its own, outside any document (the `geometry_effects` wasm export: the editor's
/// selection bounds, SVG / PDF export and Expand Appearance). `area` is the text's area input (its Text Wrap objects
/// and thread frames) when it has one; the caller resolves it from the document.
pub fn resolve_standalone(node: &RawNode, area: Option<&Value>) -> TextWarp {
    if !is_effected_text(node) {
        return TextWarp::None;
    }
    let mut m = node.fields().clone();
    if let Some(area) = area {
        m.insert(TEXT_AREA_KEY.into(), area.clone());
    }
    let scene = Scene::load(&crate::value::json!({
        "nodes": [Value::Object(m)],
        "pages": [{"id": "__text_warp", "width": 1, "height": 1, "nodeIds": [node.id()]}],
    })).expect("a standalone text node has no child hierarchy");
    match scene.node(node.id()).map(|n| resolve(&scene, n)) {
        Some(TextWarp::Resolved(mut r)) => {
            r.remove(TEXT_AREA_KEY);
            TextWarp::Resolved(r)
        }
        Some(other) => other,
        None => TextWarp::Pending,
    }
}

/// The node key [`resolve_standalone`] carries a text's area input under (Text Wrap objects and thread frames, which
/// live on other nodes, resolved by the caller: `moda_text::area::AreaSpec` JSON); the outliner lays the text out in
/// it instead of reading the document. Never stored.
pub const TEXT_AREA_KEY: &str = "__textArea";

/// The drawn record for `groups` (exposed for tests).
pub fn record(node: &RawNode, groups: &[TextOutlineGroup]) -> RawNode {
    let effects = geometry_effects(node);
    let copies = groups
        .iter()
        .map(|g| contours_from_points(&g.points, true))
        .collect();
    let g = apply_effects_to_copies(copies, true, false, &effects, node, None);
    let mut m = resolved_map(node, &g);
    // Rich-text-only paint: the glyphs' fills become the path's.
    m.remove("textFills");
    let fills: Vec<&Value> = g
        .groups
        .iter()
        .map(|&k| groups.get(k).map(|gr| &gr.fills).unwrap_or(&Value::Null))
        .collect();
    let first = fills.first().copied().cloned().unwrap_or(Value::Array(Vec::new()));
    // Distinct fills means several copies, which `resolved_map` lists under `EFFECT_COPIES_KEY`.
    let distinct = fills.iter().any(|f| **f != first);
    m.insert("fills".into(), first);
    if distinct && m.contains_key(EFFECT_COPIES_KEY) {
        m.insert(
            EFFECT_COPY_FILLS_KEY.into(),
            Value::Array(fills.into_iter().cloned().collect()),
        );
    } else {
        m.remove(EFFECT_COPY_FILLS_KEY);
    }
    RawNode::new(m)
}

#[cfg(test)]
#[path = "text_warp_tests.rs"]
mod tests;
