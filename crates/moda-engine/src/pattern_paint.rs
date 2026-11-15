//! Vector pattern paints (E8.5, `moda_scene::pattern`): how the engine draws a fill or stroke layer whose paint
//! is `{ type: 'pattern', patternId, transform? }`.
//!
//! **Vectors, clipped by coverage.** A pattern layer draws as copies of the pattern's tile artwork (real
//! tessellated paths, like a brush's placed artwork), clipped to the layer's region by a coverage mask of that
//! region (the clipping-frame machinery, `effects::mask`, rendered at the frame's device scale and anti-aliased
//! at output resolution). Nothing is ever sampled from a texture, so a pattern is as sharp as any other vector at
//! every zoom and in every export: curves flatten per zoom band like the rest of the plan, and the clip's edge
//! is re-rendered when the zoom crosses a power of two (`effects::mask::coverage_scale`).
//!
//! The alternative, rendering one tile into a texture at device scale and sampling it with the image pattern
//! shader, draws in one quad whatever the tile count, but it re-renders on every zoom step, needs a texture
//! the size of the tile on screen (past the device's limit when zoomed far in), blurs when the pattern or the
//! shape is rotated (the texels no longer land on pixels), and can't express the brick and hex offsets
//! without a larger periodic cell. The price of vectors is plan size: one draw per tile copy per artwork
//! shape. A layer that would need more than [`MAX_PATTERN_DRAWS`] draws is not drawn (with a note naming the
//! pattern); the SVG and PDF exports write native pattern cells and have no such limit.
//!
//! **Where the copies go.** The region's node-local bounds, mapped into pattern space by the inverse of the
//! paint's transform, pick the lattice copies that reach into it (`Tile::copies`, in the pattern's overlap
//! order). Each copy's shapes are the tile's mapped into node-local space in f64 (the paint's transform after
//! the copy's lattice offset), all drawn under the node's own transform: neighbouring copies then share their
//! edges exactly, so tiles that meet edge to edge leave no seam at any rotation or zoom (a per-copy GPU matrix
//! would round each copy's edges differently), curves flatten for the size they are drawn at and strokes scale
//! with the tiles. The tile artwork itself is parsed once per pattern content ([`tile_art`]).
//!
//! **Opacity and blend.** The layer's opacity multiplies into each tile shape and its blend mode applies to each
//! shape without one (as a brush's placed artwork): where two shapes of the art overlap inside a translucent
//! layer, the lower one shows through.
//!
//! **Caches.** A node's hashes fold the content hash of every pattern it names (`residency::node_hashes_in`),
//! so an edit to a pattern definition re-keys every plan, tessellation and fingerprint that draws it. The clip
//! mask is keyed by its layer (`<node>::paint::<k>#pattern`), so the retained camera slot's mask bookkeeping
//! treats it like any clipping frame. Tile shapes key the per-node tessellation cache by their own ids (the
//! layer, the lattice cell and the shape) and their geometry hash.
//!
//! [`set_enabled`] switches the whole lane off (pattern layers draw nothing, as before patterns existed): the
//! regression tests compare pattern-free documents with it on and off.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

use moda_scene::paint::{Paint, PatternPaint};
use moda_scene::pattern::{PatternDef, TileArt};
use moda_scene::types::{Affine, RawNode};

use moda_scene::value::Value;

/// Most tile-shape draws one pattern layer makes (copies × artwork shapes).
pub const MAX_PATTERN_DRAWS: usize = 20_000;

/// Tile artworks kept prepared (per pattern content).
const TILE_CACHE_ENTRIES: usize = 64;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(true) };
    static TILES: RefCell<HashMap<u64, Arc<TileArt>>> = RefCell::new(HashMap::new());
    static LAYERS_PLANNED: Cell<u64> = const { Cell::new(0) };
}

/// Turn the pattern lane on or off (on by default; thread-local, for tests and A/Bs).
pub fn set_enabled(on: bool) {
    ENABLED.with(|c| c.set(on));
}

pub fn enabled() -> bool {
    ENABLED.with(Cell::get)
}

/// How many pattern layers this thread has planned (a counter for tests).
pub fn layers_planned() -> u64 {
    LAYERS_PLANNED.with(Cell::get)
}

pub(crate) fn note_layer_planned() {
    LAYERS_PLANNED.with(|c| c.set(c.get() + 1));
}

fn is_pattern(v: &Value) -> bool {
    v.get("paint").and_then(|p| p.get("type")).and_then(Value::as_str) == Some("pattern")
}

/// Whether any of the node's fill, stroke or glyph-fill layers is a pattern (visible or not).
pub fn has_pattern_layer(node: &RawNode) -> bool {
    enabled()
        && ["fills", "strokes", "textFills"]
            .iter()
            .any(|k| matches!(node.json_field(k), Some(Value::Array(ls)) if ls.iter().any(is_pattern)))
}

/// The node without its pattern layers (every other layer kept, in order).
pub fn without_pattern_layers(node: &RawNode) -> RawNode {
    let mut m = node.fields().clone();
    for key in ["fills", "strokes", "textFills"] {
        if let Some(Value::Array(layers)) = m.get_mut(key) {
            layers.retain(|l| !is_pattern(l));
        }
    }
    node.with_fields(m)
}

/// A paint-stack layer clone's pattern paint (`paint::layers`: the clone carries one fill or one stroke), with
/// whether it is the stroke. `None` for any other layer, or with the lane off.
pub fn layer_pattern(layer: &RawNode) -> Option<(PatternPaint, bool)> {
    if !enabled() {
        return None;
    }
    let fills = moda_scene::paint::node_fills(layer);
    let strokes = moda_scene::paint::node_strokes(layer);
    match (fills.as_slice(), strokes.as_slice()) {
        ([f], []) => match &f.paint {
            Paint::Pattern(p) => Some((p.clone(), false)),
            _ => None,
        },
        ([], [s]) => match &s.paint {
            Paint::Pattern(p) => Some((p.clone(), true)),
            _ => None,
        },
        _ => None,
    }
}

/// The layer clone painted opaque white at full opacity with normal blending: what its region's coverage is
/// built from (the same body lane, so dashes, trims, arrowheads, per-side weights and brushes are the clip).
pub fn white_layer(layer: &RawNode) -> RawNode {
    let mut m = layer.fields().clone();
    let white = moda_scene::value::json!({"type": "solid", "color": "#ffffff"});
    for key in ["fills", "strokes"] {
        if let Some(Value::Array(ls)) = m.get_mut(key) {
            for l in ls.iter_mut() {
                if let Some(o) = l.as_object_mut() {
                    o.insert("paint".into(), white.clone());
                    o.remove("opacity");
                    o.remove("blendMode");
                }
            }
        }
    }
    m.insert("opacity".into(), Value::from(1.0));
    m.remove("blendMode");
    layer.with_fields(m)
}

/// `a⁻¹`, or `None` when singular.
pub fn invert(a: Affine) -> Option<Affine> {
    let det = a.a * a.d - a.b * a.c;
    if !(det.abs() > 1e-12) || !det.is_finite() {
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

/// The tile artwork of a pattern definition (content hash `pattern_hash`), prepared once and cached.
pub fn tile_art(def: &PatternDef, pattern_hash: u64) -> Arc<TileArt> {
    if let Some(hit) = TILES.with(|t| t.borrow().get(&pattern_hash).cloned()) {
        return hit;
    }
    let art = Arc::new(moda_scene::pattern::tile_art(def, &crate::brush_artwork::brush_artwork_geometry));
    TILES.with(|t| {
        let mut t = t.borrow_mut();
        if t.len() >= TILE_CACHE_ENTRIES {
            t.clear();
        }
        t.insert(pattern_hash, art.clone());
    });
    art
}

/// Node-local positions of a clip's triangles: every `Body` item of the white layer's draw, a stroke layer's
/// stroke band only (a body that also carries fill triangles keeps its stroke triangles; one with no stroke
/// vertices at all, such as an arrowhead or a per-side stroke ring, is all stroke).
pub fn clip_positions(items: &[crate::seams::DrawItem], is_stroke: bool) -> Vec<f32> {
    let mut out = Vec::new();
    for item in items {
        let crate::seams::DrawItem::Body { geometry, .. } = item else { continue };
        let g = &**geometry;
        let n = (g.positions.len() / 2).min(g.vertex_count as usize);
        let has_stroke = g.vertex_types.iter().take(n).any(|t| *t >= 0.5);
        let keep_stroke_only = is_stroke && has_stroke && g.vertex_types.len() >= n;
        for tri in 0..n / 3 {
            let v = tri * 3;
            if keep_stroke_only && !(0..3).all(|k| g.vertex_types[v + k] >= 0.5) {
                continue;
            }
            out.extend_from_slice(&g.positions[v * 2..v * 2 + 6]);
        }
    }
    out
}

/// `[min_x, min_y, max_x, max_y]` of node-local positions (`None` when empty or not finite).
pub fn positions_bounds(positions: &[f32]) -> Option<[f64; 4]> {
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for p in positions.chunks_exact(2) {
        let (x, y) = (p[0] as f64, p[1] as f64);
        if !(x.is_finite() && y.is_finite()) {
            continue;
        }
        b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
    }
    (b[0] <= b[2] && b[1] <= b[3]).then_some(b)
}

/// The AABB of `b` (`[min_x, min_y, max_x, max_y]`) under `a`.
pub fn map_bounds(b: [f64; 4], a: Affine) -> [f64; 4] {
    let pts = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])];
    let mut out = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for p in pts {
        let (x, y) = moda_scene::transform::apply_point(p, a);
        out = [out[0].min(x), out[1].min(y), out[2].max(x), out[3].max(y)];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    fn node(v: Value) -> RawNode {
        RawNode::from_value(v).unwrap()
    }

    #[test]
    fn layer_pattern_reads_single_layer_clones() {
        let fill = node(json!({"id": "n::paint::0", "type": "rectangle", "width": 10, "height": 10,
            "fills": [{"id": "f", "paint": {"type": "pattern", "patternId": "p", "transform": {"scale": 2}}}]}));
        let (p, stroke) = layer_pattern(&fill).unwrap();
        assert_eq!((p.pattern_id.as_str(), stroke), ("p", false));
        assert_eq!(p.affine().a, 2.0);
        let stroke = node(json!({"id": "n::paint::1", "type": "rectangle", "width": 10, "height": 10,
            "strokes": [{"id": "s", "width": 3, "paint": {"type": "pattern", "patternId": "q"}}]}));
        assert!(layer_pattern(&stroke).unwrap().1);
        let solid = node(json!({"id": "n", "type": "rectangle", "fills": [{"id": "f", "paint": {"type": "solid", "color": "#000"}}]}));
        assert!(layer_pattern(&solid).is_none());
        set_enabled(false);
        assert!(layer_pattern(&fill).is_none() && !has_pattern_layer(&fill));
        set_enabled(true);
        assert!(has_pattern_layer(&fill));
        let stripped = without_pattern_layers(&fill);
        assert!(moda_scene::paint::node_fills(&stripped).is_empty());
    }

    #[test]
    fn white_layer_keeps_geometry_and_drops_paint_settings() {
        let s = node(json!({"id": "n::paint::1", "type": "path", "opacity": 0.5, "blendMode": "multiply",
            "strokes": [{"id": "s", "width": 3, "opacity": 0.2, "dash": [2, 1], "paint": {"type": "pattern", "patternId": "q"}}]}));
        let w = white_layer(&s);
        let st = &moda_scene::paint::node_strokes(&w)[0];
        assert_eq!(st.width, 3.0);
        assert_eq!(st.dash.as_deref(), Some(&[2.0, 1.0][..]));
        assert_eq!(st.opacity, None);
        assert!(matches!(st.paint, Paint::Solid(_)));
        assert_eq!(w.f64("opacity"), Some(1.0));
        assert!(w.json_field("blendMode").is_none());
    }

    #[test]
    fn inverse_and_bounds_round_trip() {
        let a = moda_scene::transform::from_trs(moda_scene::types::Transform2D {
            x: 5.0, y: -3.0, rotation: 30.0, scale_x: 2.0, scale_y: 2.0,
        });
        let i = invert(a).unwrap();
        let p = moda_scene::transform::apply_point(moda_scene::transform::apply_point((7.0, 11.0), a), i);
        assert!((p.0 - 7.0).abs() < 1e-9 && (p.1 - 11.0).abs() < 1e-9);
        assert!(invert(Affine { a: 0.0, b: 0.0, c: 0.0, d: 0.0, e: 0.0, f: 0.0 }).is_none());
        assert_eq!(map_bounds([0.0, 0.0, 10.0, 5.0], moda_scene::transform::translate(1.0, 2.0)), [1.0, 2.0, 11.0, 7.0]);
        assert_eq!(positions_bounds(&[1.0, 2.0, 3.0, -1.0]), Some([1.0, -1.0, 3.0, 2.0]));
        assert_eq!(positions_bounds(&[]), None);
    }
}
