//! Viewport culling of camera plans (docs/qa/perf.md, "Not fixed" 3).
//!
//! A camera plan built with a cull window leaves out every display-list node whose
//! world-space VISUAL bounds miss the window: the geometry box grown by everything
//! that paints outside it (stroke width with its miter / alignment reach, arrowheads,
//! drop shadow offset + blur, layer / background blur radius, text effects (all read
//! from `effects[]`), stroke
//! wave / wiggle / profile), mapped through the node's world affine. A parent node
//! (group / container) whose subtree bounds miss the window is skipped as ONE unit:
//! the plan builder jumps past its contiguous display-list range.
//!
//! Culling never changes a pixel inside the window: a culled node still advances the
//! render-z counter by exactly what it would have consumed, so every drawn item keeps
//! the depth it has in the unculled plan. Anything whose reach this module cannot
//! bound (a brushed stroke, an unknown effect entry, a blend node's expansion, a node
//! painted through another node's transform, non-finite geometry) gets infinite bounds
//! and is never culled — unless it sits inside a `clipContent` container: a
//! clipping container bounds its whole subtree by its own box (see
//! [`crate::effects::container_clip_aabb`]), so a brushed stroke or overflowing text
//! inside a frame is culled with the frame. Subtrees holding paint that does not go
//! through the container's clip (a layer-blur composite, a backdrop blur, a blend
//! expansion, structured `effects`, a node painted through another's transform) are
//! not narrowed.
//!
//! A placed brush (bristle, scatter, art, pattern) is bounded once it has been drawn: the plan builder records
//! the canvas box of what it tessellated (`PlanCaches::brush_bounds`, keyed on the node's geometry, brush and
//! transform), and a later window culls the stroke by that box, grown by the flattening tolerances (another zoom
//! band flattens the same shapes a little differently). Until then it is unbounded, as above.
//!
//! The editor camera lane culls to a window around the view
//! (`residency::build_camera_plan_retained`, which owns that policy); an export of a
//! canvas box culls to its own target (`frame::build_camera_box_plan`).

use std::collections::HashMap;

use moda_scene::effect;
use moda_scene::types::{Affine, DisplayList, PaintItem, PaintKind, RawNode};
use moda_scene::Scene;
use moda_scene::value::Value;

/// `[min_x, min_y, max_x, max_y]` in canvas (world) units.
pub type Aabb = [f64; 4];

pub const INFINITE: Aabb = [
    f64::NEG_INFINITY,
    f64::NEG_INFINITY,
    f64::INFINITY,
    f64::INFINITY,
];

const EMPTY: Aabb = [
    f64::INFINITY,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NEG_INFINITY,
];

fn union(a: Aabb, b: Aabb) -> Aabb {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

fn intersect(a: Aabb, b: Aabb) -> Aabb {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}

/// Whether some of `node`'s own paint bypasses an ancestor container's clip (see the
/// module doc): the layer-blur composite and the backdrop split take the plain page
/// clip, a blend expands into intermediates, and every other `effects[]` entry renders
/// through its own lane. Drop and inner shadows are drawn under the effective page
/// clip like the body, so they don't.
pub(crate) fn escapes_container_clip(node: &RawNode) -> bool {
    node.node_type() == "blend"
        || node
            .str("_transformSourceId")
            .is_some_and(|s| !s.is_empty() && s != node.id())
        || node.array("effects").is_some_and(|list| {
            list.iter().any(|e| {
                !matches!(
                    e.get("type").and_then(Value::as_str),
                    Some("drop-shadow" | "inner-shadow")
                )
            })
        })
}

/// Whether `a` holds no point.
pub fn is_empty(a: &Aabb) -> bool {
    !(a[0] <= a[2] && a[1] <= a[3])
}

/// Whether `a` and `b` overlap (touching counts as overlapping).
pub fn intersects(a: &Aabb, b: &Aabb) -> bool {
    !is_empty(a) && a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

/// Whether `outer` contains `inner`.
pub fn contains(outer: &Aabb, inner: &Aabb) -> bool {
    outer[0] <= inner[0] && outer[1] <= inner[1] && outer[2] >= inner[2] && outer[3] >= inner[3]
}

fn num(v: Option<&Value>) -> f64 {
    v.and_then(Value::as_f64)
        .filter(|x| x.is_finite())
        .unwrap_or(0.0)
        .abs()
}

/// Local-unit reach of one stroke entry (or the legacy top-level stroke fields).
/// `None` when the reach cannot be bounded (a brush).
fn stroke_reach(o: &moda_scene::value::Map, node: &RawNode) -> Option<f64> {
    if o.get("visible") == Some(&Value::Bool(false)) {
        return Some(0.0);
    }
    let w = num(o.get("width"));
    if o.get("brush").is_some_and(|b| !b.is_null()) {
        // Brush artwork (art / scatter / pattern) reaches by the artwork's own
        // proportions and scatter settings; not bounded here.
        return None;
    }
    // Miter joins reach `miterLimit * w / 2` past the centre line, an outside-aligned
    // stroke a full `w`; `w * max(miterLimit, 4)` covers both, square caps and the
    // width profile's widest marker (the engine's own outline estimate uses 4w).
    let miter = o
        .get("miterLimit")
        .and_then(Value::as_f64)
        .filter(|m| m.is_finite())
        .unwrap_or(10.0)
        .max(4.0);
    let mut reach = w * miter;
    if let Some(wave) = o.get("wave").and_then(Value::as_object) {
        reach += num(wave.get("amount")) * w.max(1.0);
    }
    if let Some(dynamic) = o.get("dynamic").and_then(Value::as_object) {
        reach += num(dynamic.get("wiggle")) * w.max(1.0);
    }
    let has_arrow = |k: &str| {
        o.get(k)
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty() && s != "none")
    };
    if has_arrow("startArrow") || has_arrow("endArrow") {
        let scale = o
            .get("arrowScale")
            .and_then(Value::as_f64)
            .filter(|s| s.is_finite() && *s > 0.0)
            .unwrap_or(1.0)
            .max(1.0);
        let pointer = num(node.json_field("pointerLength"))
            .max(num(node.json_field("pointerWidth")))
            .max(10.0);
        reach += pointer * scale + 4.0 * w;
    }
    Some(reach)
}

/// Local-unit reach of an `effects[]` entry drawn around the node (text glow / echo /
/// extrude, a shader). Offsets add, blur radii reach ~3σ, `count` repeats the offset.
/// The caller skips hidden entries.
fn effect_reach(o: &moda_scene::value::Map, box_extent: f64) -> f64 {
    let mut offset = 0.0;
    let mut blur = 0.0;
    let mut other = 0.0;
    for (k, v) in o {
        let Some(x) = v.as_f64().filter(|x| x.is_finite()) else {
            if let Some(obj) = v.as_object() {
                // `{ x, y }` offsets.
                if k.to_ascii_lowercase().contains("offset") {
                    offset += num(obj.get("x")) + num(obj.get("y"));
                }
            }
            continue;
        };
        let k = k.to_ascii_lowercase();
        if k.contains("offset") || k == "x" || k == "y" {
            offset += x.abs();
        } else if k.contains("radius") || k.contains("blur") {
            blur += x.abs();
        } else if k.contains("spread") || k.contains("depth") {
            other += x.abs();
        }
    }
    let count = num(o.get("count")).max(1.0);
    let scale_step = num(o.get("scaleStep"));
    count * offset + 3.0 * blur + other + count * scale_step * box_extent
}

/// Local-unit reach of a blur-like `effects[]` entry (`layer-blur`, or a backdrop:
/// `background-blur` / `glass` / `pixelate` / `dither`): three times its largest
/// radius-like number. The caller skips hidden entries.
fn blur_effect_reach(o: &moda_scene::value::Map) -> f64 {
    let mut r: f64 = 0.0;
    for (k, v) in o {
        if let Some(x) = v.as_f64().filter(|x| x.is_finite()) {
            let k = k.to_ascii_lowercase();
            if k.contains("radius") || k.contains("blur") || k.contains("amount") {
                r = r.max(x.abs());
            }
        }
    }
    3.0 * r
}

/// Local-unit reach of the effects on a node's fills and strokes (E8.2): the farthest entry's, each counted as
/// the node-level effects are (blur-like radii ×3, offsets, glows' radii; a geometry effect by its numbers).
fn entry_effects_reach(node: &RawNode, extent: f64) -> f64 {
    let mut reach: f64 = 0.0;
    for key in ["fills", "strokes"] {
        let Some(layers) = node.array(key) else {
            continue;
        };
        for l in layers {
            let Some(effects) = l.get("effects").and_then(Value::as_array) else {
                continue;
            };
            let mut r = 0.0;
            for e in effects {
                let Some(o) = e.as_object().filter(|_| effect::is_visible_json(e)) else {
                    continue;
                };
                r += match o.get("type").and_then(Value::as_str) {
                    Some("layer-blur") => blur_effect_reach(o),
                    Some("inner-shadow") => 0.0,
                    _ => effect_reach(o, extent),
                };
            }
            reach = reach.max(r);
        }
    }
    reach
}

/// How far (canvas units) a group's own appearance (E8.2) paints past its contents: its widest stroke plus
/// the farthest entry effect. `None` for a group without appearance.
fn group_appearance_reach(scene: &Scene, node: &RawNode) -> Option<f64> {
    scene.group_appearance(node.id())?;
    let mut stroke: f64 = 0.0;
    if let Some(strokes) = node.array("strokes") {
        for s in strokes {
            if let Some(o) = s.as_object() {
                stroke = stroke.max(stroke_reach(o, node)?);
            }
        }
    }
    Some(stroke + entry_effects_reach(node, 0.0))
}

/// How far paint can reach past the node's geometry box: in node-local units, plus in canvas units (a drop
/// shadow's offset, which is added after the node's transform), or `None` when it cannot be bounded.
fn local_reach(node: &RawNode, box_w: f64, box_h: f64, brush_learned: bool) -> Option<(f64, f64)> {
    let extent = box_w.abs().max(box_h.abs());
    let has_text = node.node_type() == "richtext"
        || matches!(node.json_field("textContent"), Some(Value::String(s)) if !s.is_empty());
    let mut reach = 0.0;
    // Strokes: every visible entry of the stack (the widest wins). A placed brush's stroke is bounded by its
    // learned box instead (`brush_learned`).
    let mut stroke: f64 = 0.0;
    if let Some(strokes) = node.array("strokes") {
        for s in strokes {
            if let Some(o) = s.as_object() {
                if brush_learned && o.get("brush").is_some_and(|b| !b.is_null()) {
                    continue;
                }
                stroke = stroke.max(stroke_reach(o, node)?);
            }
        }
    }
    reach += stroke;
    let stroke_w = node
        .array("strokes")
        .map(|a| a.iter().map(|s| num(s.get("width"))).fold(0.0, f64::max))
        .unwrap_or(0.0);
    // Drop shadows, around the stroke's outer edge: a positive spread, then the Gaussian's 3σ (σ = blur / 2, so
    // 1.5 × blur, where the shape lanes' quad and the capture lane stop: `path-shadow.wgsl`, `layer_blur::shadow`)
    // and a unit of anti-aliasing, moved by the offset in canvas units (`offset`). Text draws its shadows through
    // its own lane: 3 × blur, offset included, as before.
    let mut offset: f64 = 0.0;
    for sh in effect::drop_shadows(node) {
        let (ox, oy) = (sh.offset.x.abs(), sh.offset.y.abs());
        if has_text {
            reach = reach.max(ox + oy + sh.spread().max(0.0) + 3.0 * sh.blur.abs() + stroke_w);
            continue;
        }
        reach = reach.max(sh.spread().max(0.0) + 1.5 * sh.blur.abs() + 1.0 + stroke_w);
        offset = offset.max(ox.max(oy));
    }
    // Filter shadows (CSS `filter: drop-shadow()`, a layer filter) reach as a drop shadow does: 3σ around the
    // picture, moved by the canvas-space offset.
    for sh in effect::filter_shadows(node) {
        reach = reach.max(1.5 * sh.blur.abs() + 1.0 + stroke_w);
        offset = offset.max(sh.offset.x.abs().max(sh.offset.y.abs()));
    }
    // The rest of the visible `effects[]` list. An inner shadow paints inside the
    // shape, so it adds nothing.
    let mut extra = 0.0;
    for e in effect::node_effects_json(node) {
        let Some(o) = e.as_object().filter(|_| effect::is_visible_json(e)) else {
            continue;
        };
        match o.get("type").and_then(Value::as_str) {
            Some("drop-shadow" | "inner-shadow" | "filter-shadow") => {}
            Some(t) if t == "layer-blur" || effect::BACKDROP_EFFECT_TYPES.contains(&t) => {
                extra += blur_effect_reach(o);
            }
            _ => extra += effect_reach(o, extent),
        }
    }
    // Effects on one fill or stroke (E8.2) reach past it as a node's own do.
    extra += entry_effects_reach(node, extent);
    // Text can overflow its box (fixed-size frames, glyph overhang): a generous margin
    // in proportion to the box plus a constant for small boxes.
    if has_text {
        extra += 2.0 * extent + 512.0;
    }
    // Routed connectors (elbowed / curved with no stored waypoints) route around
    // their endpoints' box.
    if node.node_type() == "line" && node.str("routing").is_some_and(|r| r != "straight") {
        extra += extent + 64.0;
    }
    Some((reach + extra, offset))
}

/// Largest distance a unit of node-local length can span in canvas x or y under `a`.
fn affine_reach_scale(a: &Affine) -> f64 {
    (a.a.abs() + a.c.abs()).max(a.b.abs() + a.d.abs())
}

/// World-space visual bounds of one primary display-list item (its own paint only,
/// not its subtree), `EMPTY` for items that paint nothing themselves (groups), or
/// `INFINITE` when they cannot be bounded.
pub fn item_visual_bounds(scene: &Scene, item: &PaintItem) -> Aabb {
    visual_bounds(scene, item, false)
}

/// [`item_visual_bounds`]; `brush_learned` leaves the placed brush stroke out (its box is learned).
fn visual_bounds(scene: &Scene, item: &PaintItem, brush_learned: bool) -> Aabb {
    if item.kind == PaintKind::GroupBoundary {
        return EMPTY;
    }
    if item.kind == PaintKind::PageBackground {
        return INFINITE;
    }
    let Some(node) = scene.effective_node(&item.node_id) else {
        return EMPTY; // skipped by the plan builder without drawing
    };
    // A blend expands into intermediates between its sources; a node painted through
    // another node's transform is not where its box says.
    if node.node_type() == "blend"
        || node
            .str("_transformSourceId")
            .is_some_and(|s| !s.is_empty() && s != node.id())
    {
        return INFINITE;
    }
    let (Some(r), Some(a)) = (item.bounds_canvas.as_ref(), item.world_affine.as_ref()) else {
        return INFINITE;
    };
    if !(r.x.is_finite() && r.y.is_finite() && r.width.is_finite() && r.height.is_finite()) {
        return INFINITE;
    }
    let (box_w, box_h) = (
        node.f64("width").unwrap_or(r.width),
        node.f64("height").unwrap_or(r.height),
    );
    let Some((reach, offset)) = local_reach(node, box_w, box_h, brush_learned) else {
        return INFINITE;
    };
    let pad = reach * affine_reach_scale(a).max(1e-9) + offset;
    if !pad.is_finite() {
        return INFINITE;
    }
    [
        r.x.min(r.x + r.width) - pad,
        r.y.min(r.y + r.height) - pad,
        r.x.max(r.x + r.width) + pad,
        r.y.max(r.y + r.height) + pad,
    ]
}

/// How far (canvas units) a subtree-capture root's own shadows (a group's or
/// an unfilled frame's, `layer_blur::shadow::is_subtree_capture_node`) and
/// layer filters (any group's or frame's with children) paint past its subtree:
/// the farthest drop shadow (offset + 3σ-ish blur) and three layer-blur radii,
/// scaled by the root's world affine. `None` for every other item.
fn subtree_effect_reach(scene: &Scene, item: &PaintItem) -> Option<f64> {
    let node = scene.effective_node(&item.node_id)?;
    let appearance = if node.node_type() == "group" {
        match group_appearance_reach(scene, node) {
            Some(r) => Some(r),
            // A painting group whose reach can't be bounded (a brush): never culled.
            None if scene.group_appearance(node.id()).is_some() => Some(f64::INFINITY),
            None => None,
        }
    } else {
        None
    };
    let shadows = crate::layer_blur::shadow::is_subtree_capture_node(node);
    // Layer filters capture the subtree of any group or frame with children.
    if !shadows && !crate::backdrop::blend::is_blend_subtree_node(node) {
        return appearance.filter(|r| *r > 0.0).map(|r| {
            let scale = item.world_affine.as_ref().map_or(1.0, affine_reach_scale).max(1.0);
            r * scale
        });
    }
    let mut reach: f64 = appearance.unwrap_or(0.0);
    for sh in effect::drop_shadows(node).into_iter().filter(|_| shadows) {
        reach = reach.max(sh.offset.x.abs() + sh.offset.y.abs() + 3.0 * sh.blur.abs());
    }
    // A filter shadow is a layer filter: any group's or frame's with children, like its layer blur.
    for sh in effect::filter_shadows(node) {
        reach = reach.max(sh.offset.x.abs() + sh.offset.y.abs() + 3.0 * sh.blur.abs());
    }
    for e in effect::node_effects_json(node) {
        let Some(o) = e.as_object().filter(|_| effect::is_visible_json(e)) else {
            continue;
        };
        if o.get("type").and_then(Value::as_str) == Some("layer-blur") {
            reach += blur_effect_reach(o);
        }
    }
    if reach == 0.0 {
        return None;
    }
    let scale = item
        .world_affine
        .as_ref()
        .map_or(1.0, affine_reach_scale)
        .max(1.0);
    Some(reach * scale)
}

/// The skip decision for one display list under a cull window.
#[derive(Debug, Default)]
pub struct CullPlan {
    /// Per display-list index: `Some((end, z))` = skip items `index..end` and advance
    /// the render-z counter by `z`. Set only on primary items.
    skip: Vec<Option<(usize, u64)>>,
    /// Culled primary items (range roots only — a skipped parent's descendants are
    /// not listed).
    pub culled_roots: Vec<String>,
    /// Number of primary items inside skipped ranges.
    pub culled_items: usize,
    /// The placed brush strokes the decision reached (drawn, or skipped by their learned box): their cached
    /// placements and geometry stay warm (`PlanCaches::brush_kept`).
    pub brush_seen: Vec<String>,
}

impl CullPlan {
    pub fn skip_at(&self, index: usize) -> Option<(usize, u64)> {
        self.skip.get(index).copied().flatten()
    }
}

/// Per display-list index, the subtree facts the skip decision needs: visual bounds
/// (the item's own paint unioned with its descendants'), the render-z the subtree
/// consumes, where its contiguous range ends, and how many primary items it holds.
/// Independent of the window, so the camera lane keeps one per display list and
/// scene generation (`PlanCaches::cull_index`) and only re-decides on a pan.
#[derive(Debug, Default)]
pub struct CullIndex {
    bounds: Vec<Aabb>,
    z: Vec<u64>,
    end: Vec<usize>,
    primaries: Vec<usize>,
    /// The subtree is kept or culled as a whole: a layer-filtered group's
    /// capture blurs every descendant into its neighbours, so one outside the
    /// window can still paint inside it.
    whole: Vec<bool>,
    /// A placed brush stroke (a leaf): its learned-bounds key and the bounds of the rest of its paint (fills,
    /// shadows, other strokes).
    brush: Vec<Option<(u64, Aabb)>>,
}

/// The learned canvas box of a placed brush stroke (`PlanCaches::brush_bounds`) and the zoom band it was drawn
/// at, by its key.
pub type LearnedBrushBounds<'a> = &'a dyn Fn(&str, u64) -> Option<(Aabb, i32)>;

/// Decide which display-list ranges miss `window`. `z_cost(item)` is the render-z the
/// plan builder would consume for `item` alone (0 for filtered items and groups, 1 for
/// every drawn kind; never called for a blend, which is never culled).
pub fn plan_cull(
    scene: &Scene,
    dl: &DisplayList,
    window: &Aabb,
    z_cost: impl Fn(&PaintItem) -> u64,
) -> CullPlan {
    CullIndex::build(scene, dl, z_cost, |_| None).decide(dl, window, &|_, _| None)
}

impl CullIndex {
    /// `brush_key(item)`: the learned-bounds key of a placed brush stroke (`None` for anything else, or a stroke
    /// that can't be learned).
    pub fn build(
        scene: &Scene,
        dl: &DisplayList,
        z_cost: impl Fn(&PaintItem) -> u64,
        brush_key: impl Fn(&PaintItem) -> Option<u64>,
    ) -> CullIndex {
        let n = dl.items.len();
        let mut index_of: HashMap<&str, usize> = HashMap::with_capacity(n);
        for (i, item) in dl.items.iter().enumerate() {
            if item.is_primary_node_item {
                index_of.insert(item.node_id.as_str(), i);
            }
        }
        let mut bounds = vec![EMPTY; n];
        let mut z = vec![0u64; n];
        let mut end = vec![0usize; n];
        let mut primaries = vec![0usize; n];
        let mut whole = vec![false; n];
        let mut brush = vec![None; n];
        // Whether the subtree at i holds paint that escapes an ancestor container's clip.
        let mut escapes = vec![false; n];
        // The nearest ancestor that has its own display-list item (a table / generated
        // host pushes only its transient children).
        let parent_index = |id: &str| -> Option<usize> {
            let mut cur = id;
            for _ in 0..64 {
                let p = scene
                    .parent_map()
                    .get(cur)
                    .or_else(|| scene.transient_parent_of().get(cur))?;
                if let Some(&pi) = index_of.get(p.as_str()) {
                    return Some(pi);
                }
                cur = p.as_str();
            }
            None
        };
        // Reverse pre-order: every descendant is folded into its parent before the parent
        // itself is read (descendants follow their parent contiguously).
        let mut next_primary = n;
        for i in (0..n).rev() {
            let item = &dl.items[i];
            if !item.is_primary_node_item {
                continue;
            }
            // The item's own range ends before the next primary item (its shapeText /
            // connectorText entries ride along); descendants extend it below.
            end[i] = end[i].max(next_primary);
            next_primary = i;
            let own = item_visual_bounds(scene, item);
            bounds[i] = union(bounds[i], own);
            if own == INFINITE {
                brush[i] = brush_key(item).map(|k| (k, visual_bounds(scene, item, true)));
            }
            // A group's (or frame's) layer blur and a group's (or unfilled
            // frame's) shadows paint around its whole subtree (`layer_blur::shadow`): grow the folded
            // descendants' bounds by their reach.
            whole[i] = scene.effective_node(&item.node_id).is_some_and(|n| {
                crate::backdrop::blend::is_blend_subtree_node(n)
                    && crate::layer_blur::has_active_layer_filter(n)
            });
            if let Some(pad) = subtree_effect_reach(scene, item) {
                if !is_empty(&bounds[i]) {
                    bounds[i] = if pad.is_finite() {
                        [
                            bounds[i][0] - pad,
                            bounds[i][1] - pad,
                            bounds[i][2] + pad,
                            bounds[i][3] + pad,
                        ]
                    } else {
                        INFINITE
                    };
                }
            }
            z[i] += z_cost(item);
            primaries[i] += 1;
            let node = scene.effective_node(&item.node_id);
            escapes[i] |= node.is_some_and(escapes_container_clip);
            if let Some(pi) = parent_index(&item.node_id) {
                if pi < i {
                    // A clipping container limits everything below it to its box.
                    let clip = (!escapes[i])
                        .then(|| scene.effective_node(&dl.items[pi].node_id))
                        .flatten()
                        .and_then(|p| crate::effects::container_clip_aabb(scene, p));
                    let child = match clip {
                        Some(c) => {
                            let b = intersect(bounds[i], c);
                            if is_empty(&b) {
                                EMPTY
                            } else {
                                b
                            }
                        }
                        None => bounds[i],
                    };
                    escapes[pi] |= escapes[i];
                    bounds[pi] = union(bounds[pi], child);
                    z[pi] += z[i];
                    end[pi] = end[pi].max(end[i]);
                    primaries[pi] += primaries[i];
                }
            }
        }
        CullIndex {
            bounds,
            z,
            end,
            primaries,
            whole,
            brush,
        }
    }

    /// The display-list length this index was built for.
    pub fn len(&self) -> usize {
        self.bounds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bounds.is_empty()
    }

    /// Skip every range whose subtree bounds miss `window`, outermost first. A placed brush stroke with a learned
    /// box (`learned`) is skipped when that box, grown by the flattening tolerances, and the rest of its paint miss
    /// the window.
    pub fn decide(&self, dl: &DisplayList, window: &Aabb, learned: LearnedBrushBounds) -> CullPlan {
        let n = dl.items.len().min(self.bounds.len());
        let mut plan = CullPlan {
            skip: vec![None; n],
            ..Default::default()
        };
        let mut i = 0;
        while i < n {
            let item = &dl.items[i];
            let brush_misses = || {
                let Some((key, rest)) = self.brush.get(i).copied().flatten() else {
                    return false;
                };
                if self.primaries[i] != 1 {
                    return false;
                }
                let Some((b, band)) = learned(&item.node_id, key) else {
                    return false;
                };
                let own = union(grown_for_bands(b, band), rest);
                !intersects(&own, window)
            };
            if item.is_primary_node_item && self.brush.get(i).is_some_and(Option::is_some) {
                plan.brush_seen.push(item.node_id.clone());
            }
            if item.is_primary_node_item && (!intersects(&self.bounds[i], window) || brush_misses()) {
                let e = self.end[i].max(i + 1);
                plan.skip[i] = Some((e, self.z[i]));
                plan.culled_roots.push(item.node_id.clone());
                plan.culled_items += self.primaries[i];
                i = e;
            } else if item.is_primary_node_item && self.whole[i] {
                i = self.end[i].max(i + 1);
            } else {
                i += 1;
            }
        }
        plan
    }

    /// The canvas box holding everything the subtrees `drawn` keeps paint: each kept primary item's whole subtree
    /// bounds (the items under it may draw less, never more), the subtrees under a dropped item looked at one by
    /// one. `None` when some kept subtree can't be bounded; an empty box when nothing is kept.
    pub fn drawn_bounds(
        &self,
        dl: &DisplayList,
        drawn: impl Fn(&PaintItem) -> bool,
    ) -> Option<Aabb> {
        let n = dl.items.len().min(self.bounds.len());
        let mut b = EMPTY;
        let mut i = 0;
        while i < n {
            let item = &dl.items[i];
            if !item.is_primary_node_item || !drawn(item) {
                i += 1;
                continue;
            }
            let sub = self.bounds[i];
            if is_empty(&sub) {
                i = self.end[i].max(i + 1);
                continue;
            }
            if !sub.iter().all(|v| v.is_finite()) {
                return None;
            }
            b = union(b, sub);
            i = self.end[i].max(i + 1);
        }
        Some(b)
    }
}

/// A learned brush box drawn at `band`, grown to hold the same shapes flattened at any band: a flattening lies
/// within its tolerance of the curve, and a band's tolerance is at most the coarsest band's. Plus a hundredth of
/// its size and a canvas unit, for the joins of an outline and rounding.
fn grown_for_bands(b: Aabb, band: i32) -> Aabb {
    let tol = crate::tess::outline::tolerance_for_band(band) + crate::tess::outline::tolerance_for_band(i32::MIN);
    let pad = 2.0 * tol + 0.01 * (b[2] - b[0]).max(b[3] - b[1]) + 1.0;
    [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad]
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    fn load(nodes: Value, top: &[&str]) -> Scene {
        Scene::load(&json!({
            "nodes": nodes,
            "pages": [{ "id": "p1", "width": 2000, "height": 2000,
                        "fills": [{"id": "f", "paint": {"type": "solid", "color": "#ffffff"}}],
                        "nodeIds": top }],
            "config": {},
        })).unwrap()
    }

    fn bounds_of(s: &Scene, id: &str) -> Aabb {
        let dl = moda_scene::build_display_list(s, "p1");
        let item = dl
            .items
            .iter()
            .find(|i| i.node_id == id && i.is_primary_node_item)
            .expect("item");
        item_visual_bounds(s, item)
    }

    fn rect(id: &str, x: f64, extra: Value) -> Value {
        let mut v = json!({ "id": id, "type": "rectangle", "x": x, "y": 100, "width": 100, "height": 100,
            "fills": [{"id": "f", "paint": {"type": "solid", "color": "#ff0000"}}] });
        for (k, e) in extra.as_object().unwrap() {
            v[k] = e.clone();
        }
        v
    }

    fn solid_stroke(width: f64) -> Value {
        json!([{"id": "s", "paint": {"type": "solid", "color": "#000000"}, "width": width}])
    }

    #[test]
    fn shadows_stay_inside_a_clipping_frame_other_effects_escape_it() {
        let node = |effects: Value| {
            RawNode::new(
                rect("r", 0.0, json!({ "effects": effects }))
                    .as_object()
                    .unwrap()
                    .clone(),
            )
        };
        let shadow = json!({ "id": "d", "type": "drop-shadow", "color": "#000000", "blur": 8, "offset": {"x": 0, "y": 4} });
        let inner = json!({ "id": "i", "type": "inner-shadow", "color": "#000000", "blur": 4, "offset": {"x": 0, "y": 1} });
        assert!(!escapes_container_clip(&node(json!([]))));
        assert!(!escapes_container_clip(&node(json!([
            shadow.clone(),
            inner
        ]))));
        assert!(escapes_container_clip(&node(
            json!([shadow.clone(), { "id": "b", "type": "layer-blur", "radius": 6 }])
        )));
        assert!(escapes_container_clip(&node(
            json!([{ "id": "g", "type": "background-blur", "radius": 12 }])
        )));
        assert!(escapes_container_clip(&node(
            json!([{ "id": "t", "type": "glow", "color": "#fff", "radiusPx": 4 }])
        )));
    }

    #[test]
    fn plain_rect_bounds_are_its_box() {
        let s = load(json!([rect("r", 100.0, json!({}))]), &["r"]);
        assert_eq!(bounds_of(&s, "r"), [100.0, 100.0, 200.0, 200.0]);
    }

    #[test]
    fn stroke_shadow_and_blur_grow_the_bounds() {
        let s = load(
            json!([
                rect("stroke", 100.0, json!({ "strokes": solid_stroke(8.0) })),
                rect(
                    "shadow",
                    100.0,
                    json!({ "effects": [
                        { "id": "d0", "type": "drop-shadow", "color": "#000000", "blur": 2, "offset": {"x": 5, "y": 0} },
                        { "id": "d1", "type": "drop-shadow", "color": "#000000", "blur": 10, "offset": {"x": 30, "y": 0} },
                    ] })
                ),
                rect(
                    "blur",
                    100.0,
                    json!({ "effects": [{ "id": "b", "type": "layer-blur", "radius": 12 }] })
                ),
                rect(
                    "hidden",
                    100.0,
                    json!({ "effects": [
                        { "id": "d", "type": "drop-shadow", "color": "#000000", "blur": 10,
                          "offset": {"x": 30, "y": 0}, "visible": false },
                        { "id": "b", "type": "layer-blur", "radius": 12, "visible": false },
                    ] })
                ),
            ]),
            &["stroke", "shadow", "blur", "hidden"],
        );
        let st = bounds_of(&s, "stroke");
        assert!(
            st[0] <= 100.0 - 8.0 * 4.0 && st[2] >= 200.0 + 8.0 * 4.0,
            "{st:?}"
        );
        let sh = bounds_of(&s, "shadow");
        assert!(
            sh[2] >= 200.0 + 30.0 + 15.0 && sh[2] < 200.0 + 30.0 + 20.0,
            "shadow reaches offset + 3σ (σ = blur / 2), where its quad stops: {sh:?}"
        );
        let bl = bounds_of(&s, "blur");
        assert!(bl[0] <= 100.0 - 36.0, "layer blur reaches 3 radii: {bl:?}");
        let hi = bounds_of(&s, "hidden");
        assert!(
            hi[0] > 100.0 - 36.0 && hi[2] < 200.0 + 30.0,
            "hidden effects add no reach: {hi:?}"
        );
    }

    #[test]
    fn arrowheads_grow_a_line() {
        let s = load(
            json!([{ "id": "l", "type": "line", "points": [100, 100, 300, 100],
                "strokes": [{"id": "s", "paint": {"type": "solid", "color": "#000000"}, "width": 2,
                             "endArrow": "triangle", "arrowScale": 3}] }]),
            &["l"],
        );
        let b = bounds_of(&s, "l");
        assert!(
            b[2] >= 300.0 + 30.0,
            "the arrowhead reaches past the end point: {b:?}"
        );
        assert!(b[1] <= 100.0 - 30.0, "and across the line: {b:?}");
    }

    #[test]
    fn scale_maps_the_reach() {
        let s = load(
            json!([rect(
                "r",
                100.0,
                json!({ "scaleX": 2, "scaleY": 2, "strokes": solid_stroke(1.0) })
            )]),
            &["r"],
        );
        let b = bounds_of(&s, "r");
        // Scaled ×2: the 100 px box spans 200 canvas px and the stroke reach doubles.
        assert!(b[2] - b[0] >= 200.0 + 2.0 * 2.0 * 10.0 - 1e-9, "{b:?}");
    }

    #[test]
    fn brushed_strokes_are_never_culled() {
        let s = load(
            json!([rect(
                "b",
                100.0,
                json!({ "strokes": [{"id": "s", "paint": {"type": "solid", "color": "#000000"},
                "width": 4, "brush": {"brushId": "x"}}] })
            )]),
            &["b"],
        );
        assert_eq!(bounds_of(&s, "b"), INFINITE);
    }

    fn one_per_drawn(it: &PaintItem) -> u64 {
        (it.kind != PaintKind::GroupBoundary) as u64
    }

    #[test]
    fn a_group_outside_the_window_is_skipped_as_one_range_with_its_z() {
        let s = load(
            json!([
                rect("a", 100.0, json!({})),
                { "id": "g", "type": "group", "children": ["g1", "g2"] },
                rect("g1", 1500.0, json!({ "parentId": "g" })),
                rect("g2", 1700.0, json!({ "parentId": "g" })),
                rect("c", 300.0, json!({})),
            ]),
            &["a", "g", "c"],
        );
        let dl = moda_scene::build_display_list(&s, "p1");
        let gi = dl.items.iter().position(|i| i.node_id == "g").unwrap();
        let ci = dl.items.iter().position(|i| i.node_id == "c").unwrap();
        let plan = plan_cull(&s, &dl, &[0.0, 0.0, 1000.0, 1000.0], one_per_drawn);
        assert_eq!(plan.culled_roots, vec!["g".to_string()]);
        assert_eq!(plan.culled_items, 3);
        assert_eq!(plan.skip_at(gi), Some((ci, 2)), "skips g1 and g2, two z");
        // A window reaching one child keeps the group; the other child is culled alone.
        let plan = plan_cull(&s, &dl, &[1400.0, 0.0, 1650.0, 1000.0], one_per_drawn);
        assert!(plan.skip_at(gi).is_none());
        assert!(plan.culled_roots.contains(&"g2".to_string()));
        assert!(!plan.culled_roots.contains(&"g1".to_string()));
    }

    #[test]
    fn a_groups_shadow_entering_the_window_keeps_the_group() {
        // The group's children end at x = 200; its drop shadow reaches 150 + 3·20 past.
        let s = load(
            json!([
                { "id": "g", "type": "group", "children": ["g1"],
                  "effects": [{ "id": "d", "type": "drop-shadow", "color": "#000000", "blur": 20,
                                "offset": {"x": 150, "y": 0} }] },
                rect("g1", 100.0, json!({ "parentId": "g" })),
            ]),
            &["g"],
        );
        let dl = moda_scene::build_display_list(&s, "p1");
        let plan = plan_cull(&s, &dl, &[300.0, 0.0, 1000.0, 1000.0], one_per_drawn);
        // The group stays (its shadow is planned from the whole subtree, culled
        // children included); the child outside the window is culled alone.
        assert_eq!(
            plan.culled_roots,
            vec!["g1".to_string()],
            "the group's shadow is in the window"
        );
        let plan = plan_cull(&s, &dl, &[600.0, 0.0, 1000.0, 1000.0], one_per_drawn);
        assert_eq!(plan.culled_roots, vec!["g".to_string()]);
    }

    #[test]
    fn a_layer_blurred_group_is_kept_or_culled_whole() {
        // g1 is far outside the window, but the group's capture blurs it with g2,
        // which is inside: g1 must stay in the plan.
        let s = load(
            json!([
                { "id": "g", "type": "group", "children": ["g1", "g2"],
                  "effects": [{ "id": "b", "type": "layer-blur", "radius": 4 }] },
                rect("g1", 1500.0, json!({ "parentId": "g" })),
                rect("g2", 100.0, json!({ "parentId": "g" })),
            ]),
            &["g"],
        );
        let dl = moda_scene::build_display_list(&s, "p1");
        let plan = plan_cull(&s, &dl, &[0.0, 0.0, 500.0, 500.0], one_per_drawn);
        assert!(plan.culled_roots.is_empty(), "{:?}", plan.culled_roots);
        let plan = plan_cull(&s, &dl, &[0.0, 1000.0, 500.0, 1500.0], one_per_drawn);
        assert_eq!(plan.culled_roots, vec!["g".to_string()]);
    }

    #[test]
    fn a_shadow_entering_the_window_keeps_the_node() {
        // The rect's geometry ends at x = 200; its shadow reaches x = 200 + 150 + 3·20.
        let s = load(
            json!([rect(
                "r",
                100.0,
                json!({ "effects": [{ "id": "d", "type": "drop-shadow", "color": "#000000", "blur": 20,
                "offset": {"x": 150, "y": 0} }] })
            )]),
            &["r"],
        );
        let dl = moda_scene::build_display_list(&s, "p1");
        let plan = plan_cull(&s, &dl, &[300.0, 0.0, 1000.0, 1000.0], |_| 1);
        assert!(plan.culled_roots.is_empty(), "the shadow is in the window");
        let plan = plan_cull(&s, &dl, &[600.0, 0.0, 1000.0, 1000.0], |_| 1);
        assert_eq!(plan.culled_roots, vec!["r".to_string()]);
    }
}
