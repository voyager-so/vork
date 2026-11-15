//! The engine half of live geometry effects on text (`moda_scene::text_warp`): a rich text node's glyph outlines,
//! laid out by the vector text engine as the text lane draws them (`text_geometry::node_fill_outlines`), grouped by
//! the fill they draw with.
//!
//! Outlines are cached per node, keyed by its text content (every key but its placement), its area input (wrap
//! objects, threads) and the text-host generation (a font install re-lays text out): every re-feed builds a fresh
//! `Scene`, and an unchanged text layer is served from here.

use moda_scene::text_warp::{TextOutlineGroup, TextOutlines};
use moda_scene::types::RawNode;
use moda_scene::Scene;
use moda_scene::value::{json, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// The cache is cleared when it grows past this many text nodes.
const CACHE_CAP: usize = 1024;

thread_local! {
    static CACHE: RefCell<HashMap<String, (u64, Vec<TextOutlineGroup>)>> = RefCell::new(HashMap::new());
}

fn cache_key(scene: &Scene, node: &RawNode) -> u64 {
    let mut h = rustc_hash::FxHasher::default();
    crate::residency::node_hashes(node).text.hash(&mut h);
    let area = crate::text_area::node_area_spec(scene, node);
    let story = area
        .as_ref()
        .filter(|a| a.frames.len() > 1)
        .map(|_| crate::text_area::story_html(scene, node));
    crate::text_area::area_key(area.as_ref(), story).hash(&mut h);
    crate::text_draw::text_host_generation().hash(&mut h);
    h.finish()
}

/// `#rrggbb` for a straight RGBA 0..1 colour (alpha goes on the fill layer).
fn hex(c: [f32; 4]) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
}

/// A visible `textFills` stack paints every glyph: it is the one fill stack.
fn text_fills(node: &RawNode) -> Option<Value> {
    let fills = node.json_field("textFills")?.as_array()?;
    let shown = fills.iter().any(|f| {
        f.get("visible").and_then(Value::as_bool) != Some(false)
            && f.get("opacity").and_then(Value::as_f64).is_none_or(|o| o > 0.0)
    });
    shown.then(|| Value::Array(fills.clone()))
}

/// A rich text node's glyph outlines, one group per fill (the `moda_scene::text_warp` seam).
pub fn text_warp_outlines(scene: &Scene, node: &RawNode) -> TextOutlines {
    let key = cache_key(scene, node);
    let id = node.id().to_string();
    if let Some(hit) = CACHE.with(|c| {
        c.borrow()
            .get(&id)
            .filter(|(k, _)| *k == key)
            .map(|(_, g)| g.clone())
    }) {
        return TextOutlines::Ready(hit);
    }
    let Ok(fills) = crate::text_geometry::node_fill_outlines(scene, node) else {
        return TextOutlines::Pending;
    };
    let whole = text_fills(node);
    let mut keys: Vec<String> = Vec::new();
    let mut groups: Vec<TextOutlineGroup> = Vec::new();
    for f in &fills {
        let (k, paint) = match &whole {
            // A `textFills` stack paints the glyphs, not the echo copies and extrude walls behind them.
            Some(stack) if f.kind != "effect" => (String::new(), stack.clone()),
            _ => {
                let alpha = (f.color[3] * f.opacity).clamp(0.0, 1.0) as f64;
                let paint = match &f.gradient {
                    Some(g) => json!([{"id": "fill-0", "paint": g, "opacity": f.opacity as f64}]),
                    None => json!([{"id": "fill-0", "paint": {"type": "solid", "color": hex(f.color)}, "opacity": alpha}]),
                };
                (paint.to_string(), paint)
            }
        };
        let at = match keys.iter().position(|x| *x == k) {
            Some(i) => i,
            None => {
                keys.push(k);
                groups.push(TextOutlineGroup {
                    points: Vec::new(),
                    fills: paint,
                });
                groups.len() - 1
            }
        };
        for d in &f.paths {
            crate::boolean_operands::parse_glyph_d(d, &mut groups[at].points);
        }
    }
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() >= CACHE_CAP {
            c.clear();
        }
        c.insert(id, (key, groups.clone()));
    });
    TextOutlines::Ready(groups)
}

