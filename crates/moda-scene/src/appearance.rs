//! Illustrator's Appearance model (E8.2, `docs/roadmap/appearance-model.md`): what the engine reads to draw
//! a node's fills and strokes as ONE ordered stack, with effects scoped to a single fill or stroke, and fills
//! and strokes on groups.
//!
//! - **Entry effects.** `Fill.effects` / `Stroke.effects` hold effects that apply to that layer alone. Geometry
//!   effects ([`crate::effect::GEOMETRY_EFFECT_TYPES`]) reshape the layer's outline after the node's own
//!   geometry effects; the drawing effects in [`ENTRY_DRAWING_EFFECT_TYPES`] filter the layer's pixels. Backdrop,
//!   shader and text-only effects stay node-level (they are ignored on an entry).
//! - **One order.** `appearanceOrder` lists the node's fill and stroke ids, and `contents` on a group, bottom
//!   to top. Absent, the order is canonical: the fills, then the contents, then the strokes. The stored list is
//!   repaired rather than trusted ([`ordered_entries`]): unknown or repeated ids are dropped and a missing id goes
//!   back at its canonical place.
//! - **Groups.** A group's fills and strokes paint its MERGED OUTLINE (`crate::boolean_node::group_merged_outline`),
//!   below or above its contents by the order. [`Scene::group_appearance`](crate::scene::Scene::group_appearance)
//!   holds the path each such group paints.
//!
//! A node without entry effects and in canonical order ([`has_custom_stack`] false) draws exactly as before
//! this model existed.

use crate::value::Value;

use crate::effect::{is_visible_json, GEOMETRY_EFFECT_TYPES};
use crate::paint::{parse_fills, parse_strokes, shown, Fill, Stroke};
use crate::types::RawNode;

/// The node field holding the stack order.
pub const APPEARANCE_ORDER_KEY: &str = "appearanceOrder";
/// The order id of a group's contents (its children).
pub const CONTENTS_ID: &str = "contents";
/// Marks the stand-in node the engine plans one entry's drawing effects on (never stored in a document).
pub const ENTRY_PROBE_KEY: &str = "_appearanceEntry";
/// The drawing effects a fill or stroke may carry (the rest of the effect kinds are node-level only).
pub const ENTRY_DRAWING_EFFECT_TYPES: &[&str] = &[
    "drop-shadow",
    "inner-shadow",
    "layer-blur",
    "glow",
    "inner-glow",
    "noise",
    "texture",
    "feather",
];

/// One entry of a node's stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entry {
    /// `fills[i]`.
    Fill(usize),
    /// `strokes[i]`.
    Stroke(usize),
    /// A group's children.
    Contents,
}

/// Whether an effect type may sit on a fill or stroke.
pub fn is_entry_effect_type(t: &str) -> bool {
    GEOMETRY_EFFECT_TYPES.contains(&t) || ENTRY_DRAWING_EFFECT_TYPES.contains(&t)
}

/// The visible effects of an entry that draw on an entry, in list order.
pub fn entry_effects(effects: Option<&Vec<Value>>) -> Vec<Value> {
    effects
        .map(|list| {
            list.iter()
                .filter(|e| {
                    is_visible_json(e)
                        && e.get("type")
                            .and_then(Value::as_str)
                            .is_some_and(is_entry_effect_type)
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

/// Split entry effects into (geometry, drawing), each in list order.
pub fn split_entry_effects(effects: &[Value]) -> (Vec<Value>, Vec<Value>) {
    effects.iter().cloned().partition(|e| {
        e.get("type")
            .and_then(Value::as_str)
            .is_some_and(|t| GEOMETRY_EFFECT_TYPES.contains(&t))
    })
}

/// A fill's visible entry effects.
pub fn fill_effects(fill: &Fill) -> Vec<Value> {
    entry_effects(fill.effects.as_ref())
}

/// A stroke's visible entry effects.
pub fn stroke_effects(stroke: &Stroke) -> Vec<Value> {
    entry_effects(stroke.effects.as_ref())
}

/// The canonical order: the fills, then the contents (groups), then the strokes.
fn canonical(fills: usize, strokes: usize, contents: bool) -> Vec<Entry> {
    let mut out: Vec<Entry> = (0..fills).map(Entry::Fill).collect();
    if contents {
        out.push(Entry::Contents);
    }
    out.extend((0..strokes).map(Entry::Stroke));
    out
}

/// The stored order list, if any.
pub fn stored_order(node: &RawNode) -> Option<Vec<&str>> {
    node.array(APPEARANCE_ORDER_KEY)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
}

/// The node's entries bottom to top: the stored `appearanceOrder` repaired against the layers that exist
/// (`contents` only when `contents` is true). An id names a fill before a stroke when both share it.
pub fn ordered_entries(
    node: &RawNode,
    fills: &[Fill],
    strokes: &[Stroke],
    contents: bool,
) -> Vec<Entry> {
    let canon = canonical(fills.len(), strokes.len(), contents);
    let Some(stored) = stored_order(node) else {
        return canon;
    };
    let mut out: Vec<Entry> = Vec::with_capacity(canon.len());
    for id in stored {
        let entry = if contents && id == CONTENTS_ID {
            Some(Entry::Contents)
        } else if let Some(i) = fills.iter().position(|f| f.id == id) {
            Some(Entry::Fill(i))
        } else {
            strokes.iter().position(|s| s.id == id).map(Entry::Stroke)
        };
        if let Some(e) = entry.filter(|e| !out.contains(e)) {
            out.push(e);
        }
    }
    // Missing entries go back right after their canonical predecessor (or first).
    for (k, e) in canon.iter().enumerate() {
        if out.contains(e) {
            continue;
        }
        let at = canon[..k]
            .iter()
            .rev()
            .find_map(|p| out.iter().position(|o| o == p))
            .map_or(0, |i| i + 1);
        out.insert(at, *e);
    }
    out
}

/// Whether the node's stack needs the appearance lane: an order that isn't the canonical one, or an
/// entry effect on a visible layer. False for every node written before this model.
pub fn has_custom_stack(node: &RawNode, fills: &[Fill], strokes: &[Stroke]) -> bool {
    fills
        .iter()
        .any(|f| shown(f.visible, f.opacity) && !fill_effects(f).is_empty())
        || strokes
            .iter()
            .any(|s| shown(s.visible, s.opacity) && !stroke_effects(s).is_empty())
        || (stored_order(node).is_some()
            && ordered_entries(node, fills, strokes, false)
                != canonical(fills.len(), strokes.len(), false))
}

/// Whether any visible fill or stroke of the node carries a visible entry DRAWING effect (a capture).
pub fn has_entry_drawing_effects(node: &RawNode) -> bool {
    let drawing = |e: &Vec<Value>| {
        entry_effects(Some(e)).iter().any(|e| {
            e.get("type")
                .and_then(Value::as_str)
                .is_some_and(|t| ENTRY_DRAWING_EFFECT_TYPES.contains(&t))
        })
    };
    let fills = parse_fills(node.json_field("fills"));
    let strokes = parse_strokes(node.json_field("strokes"));
    fills
        .iter()
        .any(|f| shown(f.visible, f.opacity) && f.effects.as_ref().is_some_and(drawing))
        || strokes
            .iter()
            .any(|s| shown(s.visible, s.opacity) && s.effects.as_ref().is_some_and(drawing))
}

/// A group's own fills (stored on the group; `paint::node_fills` answers none for groups).
pub fn group_fills(node: &RawNode) -> Vec<Fill> {
    if node.node_type() != "group" {
        return Vec::new();
    }
    parse_fills(node.json_field("fills"))
}

/// A group's own strokes.
pub fn group_strokes(node: &RawNode) -> Vec<Stroke> {
    if node.node_type() != "group" {
        return Vec::new();
    }
    parse_strokes(node.json_field("strokes"))
}

/// Whether a group paints its merged outline: a visible fill, or a visible stroke with a width.
pub fn group_has_appearance(node: &RawNode) -> bool {
    node.node_type() == "group"
        && (group_fills(node)
            .iter()
            .any(|f| shown(f.visible, f.opacity))
            || group_strokes(node)
                .iter()
                .any(|s| shown(s.visible, s.opacity) && s.width > 0.0))
}

/// A group's entries split at its contents: (the ids drawn under the children, the ids drawn over them),
/// bottom to top, visible layers only.
pub fn group_split(node: &RawNode) -> (Vec<Entry>, Vec<Entry>) {
    let fills = group_fills(node);
    let strokes = group_strokes(node);
    let order = ordered_entries(node, &fills, &strokes, true);
    let visible = |e: &Entry| match *e {
        Entry::Fill(i) => shown(fills[i].visible, fills[i].opacity),
        Entry::Stroke(i) => shown(strokes[i].visible, strokes[i].opacity),
        Entry::Contents => false,
    };
    let at = order
        .iter()
        .position(|e| *e == Entry::Contents)
        .unwrap_or(order.len());
    (
        order[..at].iter().copied().filter(visible).collect(),
        order[at + 1..].iter().copied().filter(visible).collect(),
    )
}

/// The ids of `entries`, for an `appearanceOrder` over one side of a group's contents.
pub fn entry_ids(entries: &[Entry], fills: &[Fill], strokes: &[Stroke]) -> Vec<String> {
    entries
        .iter()
        .filter_map(|e| match *e {
            Entry::Fill(i) => fills.get(i).map(|f| f.id.clone()),
            Entry::Stroke(i) => strokes.get(i).map(|s| s.id.clone()),
            Entry::Contents => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    fn node(v: Value) -> RawNode {
        RawNode::from_value(v).unwrap()
    }

    fn layers(n: &RawNode) -> (Vec<Fill>, Vec<Stroke>) {
        (parse_fills(n.json_field("fills")), parse_strokes(n.json_field("strokes")))
    }

    fn solid(id: &str) -> Value {
        json!({"id": id, "paint": {"type": "solid", "color": "#000"}})
    }

    fn stroke(id: &str) -> Value {
        json!({"id": id, "paint": {"type": "solid", "color": "#000"}, "width": 2})
    }

    #[test]
    fn absent_order_is_fills_contents_strokes() {
        let n = node(
            json!({"id": "g", "type": "group", "fills": [solid("f0"), solid("f1")], "strokes": [stroke("s0")]}),
        );
        let (f, s) = layers(&n);
        assert_eq!(
            ordered_entries(&n, &f, &s, true),
            [
                Entry::Fill(0),
                Entry::Fill(1),
                Entry::Contents,
                Entry::Stroke(0)
            ]
        );
        assert!(!has_custom_stack(&n, &f, &s));
    }

    #[test]
    fn stored_order_is_repaired() {
        // s0 under f0, an unknown id, a repeat, and f1 missing (it returns after f0, its canonical predecessor).
        let n = node(json!({"id": "r", "type": "rectangle",
            "fills": [solid("f0"), solid("f1")], "strokes": [stroke("s0"), stroke("s1")],
            "appearanceOrder": ["s0", "nope", "f0", "s0", "s1"]}));
        let (f, s) = layers(&n);
        assert_eq!(
            ordered_entries(&n, &f, &s, false),
            [
                Entry::Stroke(0),
                Entry::Fill(0),
                Entry::Fill(1),
                Entry::Stroke(1)
            ]
        );
        assert!(has_custom_stack(&n, &f, &s));
        // `contents` means nothing on a leaf.
        let n = node(
            json!({"id": "r", "type": "rectangle", "fills": [solid("f0")], "strokes": [stroke("s0")],
            "appearanceOrder": ["f0", "contents", "s0"]}),
        );
        let (f, s) = layers(&n);
        assert!(
            !has_custom_stack(&n, &f, &s),
            "a canonical stored order draws as before"
        );
    }

    #[test]
    fn entry_effects_keep_the_kinds_an_entry_draws() {
        let n = node(json!({"id": "r", "type": "rectangle", "strokes": [{
        "id": "s0", "paint": {"type": "solid", "color": "#0ff"}, "width": 4,
        "effects": [
            {"id": "b", "type": "layer-blur", "radius": 8},
            {"id": "h", "type": "layer-blur", "radius": 8, "visible": false},
            {"id": "g", "type": "background-blur", "radius": 8},
            {"id": "r", "type": "roughen", "size": 5}
        ]}]}));
        let (f, s) = layers(&n);
        let fx = stroke_effects(&s[0]);
        assert_eq!(fx.len(), 2);
        let (geometry, drawing) = split_entry_effects(&fx);
        assert_eq!(geometry[0]["id"], "r");
        assert_eq!(drawing[0]["id"], "b");
        assert!(has_custom_stack(&n, &f, &s));
        assert!(has_entry_drawing_effects(&n));
    }

    #[test]
    fn group_split_puts_entries_either_side_of_the_contents() {
        let n = node(json!({"id": "g", "type": "group", "children": [],
            "fills": [solid("f0")], "strokes": [stroke("s0"), stroke("s1")],
            "appearanceOrder": ["s0", "contents", "f0", "s1"]}));
        assert!(group_has_appearance(&n));
        let (under, over) = group_split(&n);
        assert_eq!(under, [Entry::Stroke(0)]);
        assert_eq!(over, [Entry::Fill(0), Entry::Stroke(1)]);
        let plain = node(json!({"id": "g", "type": "group", "children": []}));
        assert!(!group_has_appearance(&plain));
    }
}
