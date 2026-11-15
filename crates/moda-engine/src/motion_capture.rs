//! motion_capture.rs — cached while moving, exact when still (docs/roadmap/motion-captures.md).
//!
//! A few layers cost the GPU most of a frame on some documents: a group with a blend mode splits the render pass
//! (the whole multisample target is resolved and reloaded around it), and a large procedural shader fill shades
//! every pixel it covers. While the camera moves, the live editor draws such a layer (a *unit*) from a texture
//! of it: the unit renders through the layer-capture lane (`layer_blur`) as an identity capture
//! (`LayerFilters::motion`) at the capture view's zoom step, over the capture window, and the capture cache
//! (`layer_blur::cache`, keyed by `residency::capture_key`) keeps the result across frames. A pan or a zoom then
//! composites the stored texture, translated or scaled, instead of drawing the unit again. A screen or multiply
//! group composites with that mode's fixed-function blend (no split); any other mode keeps its split around the
//! composite.
//!
//! Only the live camera lane (`residency::build_camera_plan_retained`, with `PlanCaches::motion_plans`) enters
//! this, and only while the host says the camera moves (`PlanCaches::set_motion_captures`). The motion frames
//! plan in a scene state of their own (`residency::ScenePlanState`), so the still frame after them is planned by
//! exactly the state (retained plan, memos) a still frame had before: byte-identical to the frame with this off.
//! Exports, stills, snapshots and every other lane never enter it.

use moda_scene::types::RawNode;
use moda_scene::Scene;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::hash::Hasher;
use std::sync::Arc;

/// A shader-filled leaf is a unit when its canvas box is at least this large (canvas units²). The cost of a
/// procedural fill is the area it shades; smaller fills (a dot, an icon) cost little to shade live and a capture
/// each to cache. Chosen from the corpus (docs/perf/motion-captures.md): the shader layers that cost frames there
/// are 456 × 456 and up.
pub const MIN_SHADER_AREA: f64 = 256.0 * 256.0;

/// The capture scale (a factor of the zoom step's) of a unit larger than the capture window: it is captured again
/// whenever a pan leaves the window, so its capture is kept cheap. `MODA_MOTION_WINDOWED_SCALE` overrides (A/Bs).
pub fn windowed_scale() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("MODA_MOTION_WINDOWED_SCALE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| *v > 0.0 && *v <= 1.0)
            .unwrap_or(WINDOWED_SCALE)
    })
}

/// See [`windowed_scale`].
pub const WINDOWED_SCALE: f64 = 0.5;

/// Multisampling of a motion capture (`MODA_MOTION_CAPTURE_MSAA` overrides, for A/Bs).
pub fn capture_samples() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("MODA_MOTION_CAPTURE_MSAA")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| matches!(v, 1 | 4))
            .unwrap_or(1)
    })
}

/// The layers a moving frame draws from captures.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MotionUnits {
    ids: HashSet<String>,
    /// A digest of `ids` (a plan built under these units keys on it).
    digest: u64,
}

impl MotionUnits {
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.ids.contains(id)
    }

    pub fn digest(&self) -> u64 {
        self.digest
    }

    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.ids.iter().map(String::as_str)
    }

    fn from_ids(ids: HashSet<String>) -> MotionUnits {
        let mut sorted: Vec<&String> = ids.iter().collect();
        sorted.sort();
        let mut h = rustc_hash::FxHasher::default();
        for id in sorted {
            h.write(id.as_bytes());
            h.write_u8(0xff);
        }
        let digest = h.finish();
        MotionUnits { ids, digest }
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<Arc<MotionUnits>>> = const { RefCell::new(None) };
    /// `ACTIVE.is_some()`, readable without a borrow (the predicate runs for every node of every scan).
    static ANY: Cell<bool> = const { Cell::new(false) };
}

/// Restores the previous units on drop.
pub struct Entered {
    prev: Option<Arc<MotionUnits>>,
}

impl Drop for Entered {
    fn drop(&mut self) {
        let prev = self.prev.take();
        ANY.with(|a| a.set(prev.is_some()));
        ACTIVE.with(|a| *a.borrow_mut() = prev);
    }
}

/// Plan under `units` until the guard drops (`None`: under none).
pub fn enter(units: Option<Arc<MotionUnits>>) -> Entered {
    let units = units.filter(|u| !u.is_empty());
    ANY.with(|a| a.set(units.is_some()));
    let prev = ACTIVE.with(|a| std::mem::replace(&mut *a.borrow_mut(), units));
    Entered { prev }
}

/// Whether a plan is being built under motion units.
pub fn active() -> bool {
    ANY.with(Cell::get)
}

/// Whether `id` is a unit of the plan being built.
pub fn is_unit(id: &str) -> bool {
    active() && ACTIVE.with(|a| a.borrow().as_ref().is_some_and(|u| u.contains(id)))
}

/// Fold the active units into a plan key (nothing when none are active, so keys are unchanged outside motion).
pub fn fold_key(h: &mut impl Hasher) {
    if !active() {
        return;
    }
    ACTIVE.with(|a| {
        if let Some(u) = a.borrow().as_ref() {
            h.write(b"motion-captures");
            h.write_u64(u.digest);
        }
    });
}

fn parent_of<'a>(scene: &'a Scene, id: &str) -> Option<&'a str> {
    scene
        .parent_map()
        .get(id)
        .or_else(|| scene.transient_parent_of().get(id))
        .map(String::as_str)
}

fn blended(node: &RawNode) -> bool {
    crate::seams::BlendMode::from_node_value(node.str("blendMode")) != crate::seams::BlendMode::Normal
}

/// A layer that renders through its own capture (a layer filter, a capture shadow, drawing effects) or reads the
/// target (a background blur): captures don't nest, and a capture can't read the target.
fn captures_or_reads(scene: &Scene, node: &RawNode) -> bool {
    crate::layer_blur::plan::needs_layer_capture_in(scene, node) || crate::backdrop::plan::has_active_background_blur(node)
}

/// A layer that can't draw inside a unit's capture: one that captures or reads the target, splits its paint stack,
/// or blends with a mode of its own (inside a capture the backdrop starts transparent).
fn unsafe_inside(scene: &Scene, id: &str, node: &RawNode) -> bool {
    captures_or_reads(scene, node)
        || blended(node)
        || crate::paint::layers::stack_needs_split(node, scene.ancestor_opacity_product(id), false)
}

fn shader_filled(node: &RawNode) -> bool {
    moda_scene::paint::node_fills_shared(node).iter().any(|f| {
        moda_scene::paint::shown(f.visible, f.opacity)
            && matches!(f.paint, moda_scene::paint::Paint::Shader(_))
    })
}

/// The units of `scene`: what a moving frame draws from captures.
///
/// - A group (or a frame with children) with a blend mode: it splits the pass on every frame.
/// - A leaf with a procedural shader fill at least [`MIN_SHADER_AREA`] large: it shades its whole area on every
///   frame. Such a leaf is taken up to the highest group above it (not a frame) that holds nothing a capture can't
///   draw, so a stack of shader layers in one group is one capture, not one each.
///
/// A unit's subtree must hold nothing that captures, reads the target, splits or blends on its own; a unit inside
/// another unit, or inside a layer that captures, is left to the outer one. Must be called outside [`enter`] (it
/// asks the plain capture predicates).
pub fn units(scene: &Scene) -> MotionUnits {
    debug_assert!(!active(), "motion units are computed outside a motion plan");
    let mut drawn: Vec<String> = Vec::new();
    for page in &scene.doc().pages {
        drawn.extend(scene.flattened_z_order_for_page(page.id()));
    }
    if scene.is_freeform() {
        drawn.extend(scene.flattened_z_order_for_canvas());
    }
    // Ancestors of a layer a capture can't draw: none of them can be a unit.
    let mut tainted: HashSet<&str> = HashSet::new();
    for id in &drawn {
        let Some(node) = scene.effective_node(id) else { continue };
        if !unsafe_inside(scene, id, node) {
            continue;
        }
        let mut cur = id.as_str();
        let mut guard = 0;
        while let Some(p) = parent_of(scene, cur) {
            if !tainted.insert(p) || guard > 256 {
                break;
            }
            guard += 1;
            cur = p;
        }
    }
    let groupish = |n: &RawNode| crate::backdrop::blend::is_blend_subtree_node(n);
    let mut picked: HashSet<String> = HashSet::new();
    for id in &drawn {
        let Some(node) = scene.effective_node(id) else { continue };
        if captures_or_reads(scene, node) {
            continue;
        }
        if groupish(node) {
            if blended(node) && !tainted.contains(id.as_str()) {
                picked.insert(id.clone());
            }
            continue;
        }
        if !shader_filled(node) || !crate::layer_blur::plan::capture_capable(node) {
            continue;
        }
        let big = crate::backdrop::plan::node_canvas_aabb(scene, node)
            .is_some_and(|b| b.finite() && b.w * b.h >= MIN_SHADER_AREA);
        if !big {
            continue;
        }
        // Up through the groups that can hold it (a blended leaf blends on its own: it stays one).
        let mut unit = id.as_str();
        if !blended(node) && !crate::paint::layers::stack_needs_split(node, scene.ancestor_opacity_product(id), false) {
            while let Some(p) = parent_of(scene, unit) {
                let Some(pn) = scene.effective_node(p) else { break };
                if pn.node_type() != "group" || tainted.contains(p) || captures_or_reads(scene, pn) {
                    break;
                }
                unit = p;
                if blended(pn) {
                    break; // its parent holds a blended layer
                }
            }
        }
        picked.insert(unit.to_string());
    }
    // A unit under another unit, or under a layer that captures, is drawn by that one.
    let ids: HashSet<String> = picked
        .iter()
        .filter(|id| {
            let mut cur = id.as_str();
            let mut guard = 0;
            while let Some(p) = parent_of(scene, cur) {
                if picked.contains(p) || scene.effective_node(p).is_some_and(|n| captures_or_reads(scene, n)) || guard > 256 {
                    return false;
                }
                guard += 1;
                cur = p;
            }
            true
        })
        .cloned()
        .collect();
    MotionUnits::from_ids(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    fn scene(nodes: moda_scene::value::Value, children: moda_scene::value::Value) -> Scene {
        let doc = json!({
            "nodes": nodes,
            "pages": [{"id": "p", "name": "p", "width": 1000, "height": 1000, "nodeIds": children,
                       "freeformX": 0, "freeformY": 0}],
            "canvasRootNodeIds": [],
            "layoutMode": "freeform",
            "variables": [], "brushes": [], "config": {},
        });
        Scene::load_str(&doc.to_string()).expect("scene")
    }

    fn shader_rect(id: &str, w: f64, extra: moda_scene::value::Value) -> moda_scene::value::Value {
        let mut n = json!({"id": id, "type": "rectangle", "x": 0, "y": 0, "width": w, "height": w,
            "fills": [{"id": "f", "paint": {"type": "shader", "shaderId": "mesh-gradient"}}]});
        if let (Some(m), Some(e)) = (n.as_object_mut(), extra.as_object()) {
            for (k, v) in e {
                m.insert(k.clone(), v.clone());
            }
        }
        n
    }

    #[test]
    fn blended_groups_and_large_shader_fills_are_units() {
        let s = scene(
            json!([
                shader_rect("bg", 1000.0, json!({})),
                shader_rect("dot", 20.0, json!({})),
                {"id": "g", "type": "group", "blendMode": "screen", "children": ["e"]},
                {"id": "e", "type": "ellipse", "x": 0, "y": 0, "width": 100, "height": 100,
                 "fills": [{"id": "f", "paint": {"type": "solid", "color": "#ff0000"}}]},
                {"id": "plain", "type": "group", "children": ["r"]},
                {"id": "r", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10},
                shader_rect("sl", 400.0, json!({"blendMode": "soft-light"})),
            ]),
            json!(["bg", "dot", "g", "plain", "sl"]),
        );
        let u = units(&s);
        let mut got: Vec<&str> = u.ids().collect();
        got.sort();
        assert_eq!(got, vec!["bg", "g", "sl"]);
    }

    #[test]
    fn shader_layers_in_one_group_are_one_unit() {
        let s = scene(
            json!([
                {"id": "outer", "type": "group", "children": ["a", "inner"]},
                shader_rect("a", 800.0, json!({})),
                {"id": "inner", "type": "group", "children": ["b", "band"]},
                shader_rect("b", 800.0, json!({})),
                {"id": "band", "type": "rectangle", "x": 0, "y": 0, "width": 800, "height": 4},
            ]),
            json!(["outer"]),
        );
        assert_eq!(units(&s).ids().collect::<Vec<_>>(), vec!["outer"]);
    }

    #[test]
    fn a_group_holding_what_a_capture_cannot_draw_is_not_a_unit() {
        let s = scene(
            json!([
                {"id": "g", "type": "group", "blendMode": "screen", "children": ["m", "n"]},
                {"id": "m", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10, "blendMode": "multiply"},
                shader_rect("n", 800.0, json!({})),
            ]),
            json!(["g"]),
        );
        // The group splits as before; the shader leaf inside it caches on its own.
        assert_eq!(units(&s).ids().collect::<Vec<_>>(), vec!["n"]);
    }

    #[test]
    fn the_predicate_answers_only_inside_enter() {
        let u = Arc::new(MotionUnits::from_ids(["a".to_string()].into_iter().collect()));
        assert!(!is_unit("a"));
        {
            let _g = enter(Some(u.clone()));
            assert!(active() && is_unit("a") && !is_unit("b"));
            {
                let _inner = enter(None);
                assert!(!active() && !is_unit("a"));
            }
            assert!(is_unit("a"));
        }
        assert!(!active() && !is_unit("a"));
    }
}
