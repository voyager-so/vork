//! Display-list builder — port of `buildDisplayList`
//! (`core/rendering/displayList.ts:99-209`) with the page scope and renderer
//! defaults (visibleOnly=true, expandTransientNodes=true).
//!
//! The (nodeId, kind, zIndex) tuple stream over `items` is the M0c↔M0d
//! oracle contract (PLAN M0c Verify).

use crate::paint::Paint;
use crate::scene::{is_parent_node, HashMap, Scene, SeenIds};
use crate::transform;
use crate::types::{
    is_page_bg_id, Affine, ClipDescriptor, DisplayList, NodeId, PaintEffect, PaintEffectKind,
    PaintItem, PaintKind, PaintPhase, RawNode, Rect, Transform2D,
};
use crate::value::Value;
use std::collections::HashSet;

/// Group bounds union their children this many levels down, no further (see [`bounds_for`]).
const BOUNDS_MAX_DEPTH: usize = 32;

/// What one display-list build has already worked out, so a deep tree of groups costs each node's canvas
/// transform once instead of once per group whose bounds include it (#1715). Owned by one build and dropped with
/// it: nothing outlives the scene state it was computed from, so nothing needs invalidating. Every entry is the
/// value the uncached computation returns, bit for bit.
#[derive(Default)]
struct BuildMemo {
    /// [`Scene::node_canvas_affine_and_transform`] by node id.
    canvas: HashMap<NodeId, Option<(Affine, Transform2D)>>,
    /// Bounds that no depth cut-off truncated, with the height of the subtree they cover: the same at any depth
    /// that leaves room for that subtree.
    bounds: HashMap<NodeId, (Option<Rect>, usize)>,
}

impl BuildMemo {
    fn canvas(&mut self, scene: &Scene, id: &str) -> Option<(Affine, Transform2D)> {
        if let Some(c) = self.canvas.get(id) {
            return *c;
        }
        let c = scene.node_canvas_affine_and_transform(id);
        self.canvas.insert(id.to_string(), c);
        c
    }

    fn canvas_affine(&mut self, scene: &Scene, id: &str) -> Option<Affine> {
        self.canvas(scene, id).map(|(a, _)| a)
    }

    fn canvas_transform(&mut self, scene: &Scene, id: &str) -> Option<Transform2D> {
        self.canvas(scene, id).map(|(_, t)| t)
    }
}

/// Build the display list for one page (displayList.ts:99-122).
pub fn build_display_list(scene: &Scene, page_id: &str) -> DisplayList {
    let node_order = scene.flattened_z_order_for_page(page_id);
    let clip_mask_source_ids = collect_clip_mask_source_ids(scene);
    let mut memo = BuildMemo::default();
    let mut items = Vec::new();
    for (z_index, node_id) in node_order.iter().enumerate() {
        let Some(node) = scene.effective_node(node_id) else {
            continue;
        };
        items.extend(build_paint_items_for_node(
            scene,
            node,
            z_index,
            &clip_mask_source_ids,
            &mut memo,
        ));
    }
    let page_index = scene
        .doc
        .pages
        .iter()
        .position(|p| p.id() == page_id)
        .unwrap_or(0);
    DisplayList {
        page_id: page_id.to_string(),
        page_index,
        items,
    }
}

/// Build the display list for the freeform canvas root — the floating
/// (off-page) nodes that sit on no page. Mirrors `build_display_list` over the
/// canvas z-order (`flattened_z_order_for_canvas`) but emits NO page-background
/// item, since the pasteboard has no page fill. Lets the WASM camera
/// frame render off-page nodes, matching the WebGPU orchestrator's freeform
/// pass (`WebGPURenderOrchestrator.ts:959-977`).
pub fn build_display_list_for_canvas(scene: &Scene) -> DisplayList {
    build_display_list_for_canvas_with(scene, &mut BuildMemo::default())
}

fn build_display_list_for_canvas_with(scene: &Scene, memo: &mut BuildMemo) -> DisplayList {
    let node_order = scene.flattened_z_order_for_canvas();
    // No visible floating nodes → skip the whole-scene clip-mask-source scan.
    if node_order.is_empty() {
        return DisplayList {
            page_id: crate::types::SCENE_ROOT_ID.to_string(),
            page_index: 0,
            items: Vec::new(),
        };
    }
    let clip_mask_source_ids = collect_clip_mask_source_ids(scene);
    let mut items = Vec::new();
    for (z_index, node_id) in node_order.iter().enumerate() {
        let Some(node) = scene.effective_node(node_id) else {
            continue;
        };
        items.extend(build_paint_items_for_node(
            scene,
            node,
            z_index,
            &clip_mask_source_ids,
            memo,
        ));
    }
    DisplayList {
        // Synthetic: the canvas root is not a page. `page_index` is unused by
        // the camera-plan item builder (it loops over `items` only).
        page_id: crate::types::SCENE_ROOT_ID.to_string(),
        page_index: 0,
        items,
    }
}

/// The display-list items of just these flattened node ids (a slice of a
/// page's or the canvas root's z-order, [`Scene::flattened_z_order_for_ids`]),
/// as [`build_display_list`] builds them, `z_index` counting from
/// `z_index_start`. `clip_mask_source_ids` is [`clip_mask_source_ids`] of the
/// scene. Lets the engine's retained camera plan rebuild one changed subtree's
/// items without rebuilding the whole list.
pub fn display_list_items_for(
    scene: &Scene,
    node_order: &[crate::types::NodeId],
    z_index_start: usize,
    clip_mask_source_ids: &HashSet<String>,
) -> Vec<PaintItem> {
    let mut memo = BuildMemo::default();
    let mut items = Vec::new();
    for (k, node_id) in node_order.iter().enumerate() {
        let Some(node) = scene.effective_node(node_id) else {
            continue;
        };
        items.extend(build_paint_items_for_node(
            scene,
            node,
            z_index_start + k,
            clip_mask_source_ids,
            &mut memo,
        ));
    }
    items
}

/// The mask node a group node names (`clipMaskNodeId`) when it is a clip-mask
/// source for the display list's `text-mask-source` reason: the per-node term
/// of [`clip_mask_source_ids`], so a caller can keep the set up to date as
/// nodes change.
pub fn clip_mask_source_of(scene: &Scene, id: &str) -> Option<String> {
    let node = scene
        .resolved_by_id
        .get(id)
        .or_else(|| scene.node(id))
        .or_else(|| scene.transient_by_id.get(id))?;
    if node.node_type() != "group" {
        return None;
    }
    node.str("clipMaskNodeId")
        .filter(|m| !m.is_empty())
        .map(str::to_string)
}

/// Fixed margin (canvas px) framing the canvas-root content bounds — the
/// native FRAMING CONTRACT for page-less freeform renders: the output is
/// `(content bounds + margin) × scale`. Lives beside
/// [`canvas_root_content_bounds`] so the assessor's manifest frame and the
/// engine's render frame share ONE definition.
pub const CANVAS_ROOT_FRAME_MARGIN: f64 = 40.0;

/// Union canvas-absolute AABB of the freeform canvas-root (floating) nodes —
/// the bounds source the browser's page-less Freeform fit uses
/// (`fitAllFloatingContent` → `getSelectionCanvasBounds(scene,
/// scene.canvasRootNodeIds)`, viewSlice.ts:63-73). Groups union their children
/// (see `bounds_for`). Additionally unions the canvas-scope DISPLAY-LIST
/// items' bounds, which can exceed the raw node AABBs (routed connectors,
/// blend sources, expansion children) — never smaller, so the framing can only
/// widen. An item inside a clipping container counts only where that container
/// shows it: a layer parked far outside its frame would otherwise widen the
/// frame past the GPU limit and shrink the whole render. `None` when nothing
/// yields bounds — the caller's "nothing to frame" signal (mirrors
/// `fitAllFloatingContent` returning false).
pub fn canvas_root_content_bounds(scene: &Scene) -> Option<Rect> {
    fn fold(acc: &mut Option<(f64, f64, f64, f64)>, r: &Rect) {
        if !(r.x.is_finite() && r.y.is_finite() && r.width.is_finite() && r.height.is_finite()) {
            return;
        }
        let e = (r.x, r.y, r.x + r.width, r.y + r.height);
        *acc = Some(match *acc {
            None => e,
            Some(a) => (a.0.min(e.0), a.1.min(e.1), a.2.max(e.2), a.3.max(e.3)),
        });
    }
    // One memo for the roots' bounds and the display list: the list's group items need the same bounds again.
    let mut memo = BuildMemo::default();
    let mut acc: Option<(f64, f64, f64, f64)> = None;
    for id in &scene.doc.canvas_root_node_ids {
        let Some(node) = scene.effective_node(id) else {
            continue;
        };
        if let Some(r) = node_bounds_canvas(scene, node, &mut memo) {
            fold(&mut acc, &r);
        }
    }
    // Only pay the display-list walk when there is root content at all.
    if acc.is_some() {
        let dl = build_display_list_for_canvas_with(scene, &mut memo);
        for item in &dl.items {
            let Some(mut r) = item.bounds_canvas else {
                continue;
            };
            let mut shown = true;
            for clip in item.clip_chain.iter().filter(|c| c.mode == "container-clip") {
                let Some(owner) = scene.effective_node(&clip.owner_node_id) else {
                    continue;
                };
                let Some(o) = node_bounds_canvas(scene, owner, &mut memo) else {
                    continue;
                };
                let (x0, y0) = (r.x.max(o.x), r.y.max(o.y));
                let (x1, y1) = ((r.x + r.width).min(o.x + o.width), (r.y + r.height).min(o.y + o.height));
                shown &= x1 >= x0 && y1 >= y0;
                r = Rect { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
            }
            if shown {
                fold(&mut acc, &r);
            }
        }
    }
    acc.map(|(x0, y0, x1, y1)| Rect {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    })
}

/// The canvas-root FRAME size: content bounds + [`CANVAS_ROOT_FRAME_MARGIN`]
/// on every side — what a page-less freeform render's output measures at
/// scale 1. Emitted on the assess manifest so the backend can derive render
/// caps from the engine's authoritative framing instead of re-deriving scene
/// geometry.
pub fn canvas_root_frame_size(scene: &Scene) -> Option<(f64, f64)> {
    let b = canvas_root_content_bounds(scene)?;
    Some((
        b.width + CANVAS_ROOT_FRAME_MARGIN * 2.0,
        b.height + CANVAS_ROOT_FRAME_MARGIN * 2.0,
    ))
}

/// displayList.ts:142-176.
fn build_paint_items_for_node(
    scene: &Scene,
    node: &RawNode,
    z_index: usize,
    clip_mask_source_ids: &HashSet<String>,
    memo: &mut BuildMemo,
) -> Vec<PaintItem> {
    let base = build_base_paint_item(scene, node, z_index, clip_mask_source_ids, memo);
    let mut items = vec![base.clone()];

    if has_shape_text(node) {
        let mut reasons = base.fallback_reasons.clone();
        push_unique(&mut reasons, "intra-node-ordering");
        items.push(PaintItem {
            id: format!("{}:shapeText", node.id()),
            kind: PaintKind::ShapeText,
            phase: PaintPhase::Transparent,
            fallback_reasons: reasons,
            is_primary_node_item: false,
            ..base.clone()
        });
    }
    if has_connector_text(node) {
        let mut reasons = base.fallback_reasons.clone();
        push_unique(&mut reasons, "intra-node-ordering");
        items.push(PaintItem {
            id: format!("{}:connectorText", node.id()),
            kind: PaintKind::ConnectorText,
            phase: PaintPhase::Transparent,
            fallback_reasons: reasons,
            is_primary_node_item: false,
            ..base
        });
    }
    items
}

/// displayList.ts:178-209.
fn build_base_paint_item(
    scene: &Scene,
    node: &RawNode,
    z_index: usize,
    clip_mask_source_ids: &HashSet<String>,
    memo: &mut BuildMemo,
) -> PaintItem {
    let kind = primary_paint_kind(node);
    // One ancestor walk serves the effects, the clip chain and the fallback reasons.
    let parent_group_ids = collect_parent_group_ids(scene, node.id());
    let clip_chain = collect_clip_chain(scene, &parent_group_ids);
    let effects = collect_effects(scene, node, &parent_group_ids);
    let mut fallback_reasons = collect_fallback_reasons(
        scene,
        node,
        &effects,
        clip_mask_source_ids,
        &parent_group_ids,
        &clip_chain,
    );
    // collectTransientTextFallbackReasons (displayList.ts:220-223): transient
    // richtext (table cell contents) report 'unsupported-node-kind'.
    if kind == PaintKind::Richtext && scene.transient_by_id.contains_key(node.id()) {
        push_unique(&mut fallback_reasons, "unsupported-node-kind");
    }
    PaintItem {
        id: format!("{}:{}", node.id(), kind.as_str()),
        node_id: node.id().to_string(),
        node_type: if is_page_bg_id(node.id()) {
            "pageBackground".to_string()
        } else {
            node.node_type().to_string()
        },
        kind,
        z_index,
        phase: paint_phase(node, kind),
        bounds_canvas: node_bounds_canvas(scene, node, memo),
        world_transform: memo.canvas_transform(scene, node.id()),
        world_affine: memo.canvas_affine(scene, node.id()),
        opacity: node.opacity(),
        clip_chain,
        blend_mode: blend_mode_of(node),
        effects,
        fallback_reasons,
        is_primary_node_item: true,
    }
}

/// displayList.ts:211-218.
fn primary_paint_kind(node: &RawNode) -> PaintKind {
    if is_page_bg_id(node.id()) {
        return PaintKind::PageBackground;
    }
    match node.node_type() {
        "group" => PaintKind::GroupBoundary,
        "richtext" => PaintKind::Richtext,
        "generated" if node.str("renderMode") == Some("vector-asset") => PaintKind::Svg,
        "generated" if node.str("renderMode") == Some("raster-asset") => PaintKind::Image,
        _ => PaintKind::ShapeFill,
    }
}

/// displayList.ts:225-232.
fn paint_phase(node: &RawNode, kind: PaintKind) -> PaintPhase {
    match kind {
        PaintKind::PageBackground => return PaintPhase::Background,
        PaintKind::Richtext | PaintKind::ShapeText | PaintKind::ConnectorText => {
            return PaintPhase::Transparent
        }
        PaintKind::GroupBoundary => return PaintPhase::Overlay,
        _ => {}
    }
    if node.opacity() < 1.0 {
        return PaintPhase::Transparent;
    }
    if has_shadow(node) || blend_mode_of(node).is_some() || has_background_blur(node) {
        return PaintPhase::Transparent;
    }
    PaintPhase::Opaque
}

/// displayList.ts:234-242 — clip-mask source rectangles across the document.
pub fn clip_mask_source_ids(scene: &Scene) -> HashSet<String> {
    let mut ids: HashSet<String> = scene
        .node_order
        .iter()
        .filter_map(|id| clip_mask_source_of(scene, id))
        .collect();
    // Envelopes (E8.17): their transient groups' masks, and everything an envelope draws when the envelope
    // itself is a clip mask.
    if !scene.envelope_transients.is_empty() {
        ids.extend(scene.envelope_mask_sources.iter().cloned());
        let masked_envelopes: Vec<&String> =
            scene.envelope_transients.keys().filter(|e| ids.contains(*e)).collect();
        let extra: Vec<String> = masked_envelopes
            .into_iter()
            .flat_map(|e| scene.envelope_transients[e].iter().cloned())
            .collect();
        ids.extend(extra);
    }
    ids
}

fn collect_clip_mask_source_ids(scene: &Scene) -> HashSet<String> {
    clip_mask_source_ids(scene)
}

/// displayList.ts:244-268.
fn collect_fallback_reasons(
    scene: &Scene,
    node: &RawNode,
    effects: &[PaintEffect],
    clip_mask_source_ids: &HashSet<String>,
    parent_group_ids: &[String],
    clip_chain: &[ClipDescriptor],
) -> Vec<&'static str> {
    let mut reasons: Vec<&'static str> = Vec::new();
    if clip_mask_source_ids.contains(node.id())
        || parent_group_ids.iter().any(|id| clip_mask_source_ids.contains(id))
    {
        push_unique(&mut reasons, "text-mask-source");
    }
    if !clip_chain.is_empty() {
        push_unique(&mut reasons, "ancestor-clip-mask");
    }
    if has_group_opacity_ancestor(scene, parent_group_ids) {
        push_unique(&mut reasons, "group-opacity");
    }
    if blend_mode_of(node).is_some() || has_blend_mode_ancestor(scene, parent_group_ids) {
        push_unique(&mut reasons, "blend-mode");
    }
    if effects
        .iter()
        .any(|e| e.kind == PaintEffectKind::BackgroundBlur)
    {
        push_unique(&mut reasons, "backdrop-blur-source");
    }
    if effects.iter().any(|e| {
        matches!(
            e.kind,
            PaintEffectKind::Shadow
                | PaintEffectKind::Stroke
                | PaintEffectKind::Pattern
                | PaintEffectKind::Shader
        )
    }) {
        push_unique(&mut reasons, "visual-effect");
    }
    if has_gradient_text_fill(node) {
        push_unique(&mut reasons, "visual-effect");
    }
    if matches!(
        node.node_type(),
        "table" | "generated" | "generator" | "blend"
    ) {
        push_unique(&mut reasons, "unsupported-node-kind");
    }
    reasons
}

/// displayList.ts:270-287. `parent_group_ids` is [`collect_parent_group_ids`] of the node.
fn collect_effects(scene: &Scene, node: &RawNode, parent_group_ids: &[String]) -> Vec<PaintEffect> {
    let mut effects = Vec::new();
    let id = node.id().to_string();
    if has_shadow(node) {
        effects.push(PaintEffect {
            kind: PaintEffectKind::Shadow,
            source_node_id: id.clone(),
        });
    }
    if has_stroke(node) {
        effects.push(PaintEffect {
            kind: PaintEffectKind::Stroke,
            source_node_id: id.clone(),
        });
    }
    if has_background_blur(node) {
        effects.push(PaintEffect {
            kind: PaintEffectKind::BackgroundBlur,
            source_node_id: id.clone(),
        });
    }
    if blend_mode_of(node).is_some() {
        effects.push(PaintEffect {
            kind: PaintEffectKind::BlendMode,
            source_node_id: id.clone(),
        });
    }
    // The fill stack (E3.1 P3): an image fill is the pattern effect; a shader
    // fill (or a rich-text shader glyph fill) the shader effect.
    let fill = crate::paint::top_fill(node).map(|f| f.paint);
    if matches!(fill, Some(Paint::Image(_))) {
        effects.push(PaintEffect {
            kind: PaintEffectKind::Pattern,
            source_node_id: id.clone(),
        });
    }
    if matches!(fill, Some(Paint::Shader(_)))
        || matches!(
            crate::paint::node_text_fills(node).last(),
            Some(Paint::Shader(_))
        )
    {
        effects.push(PaintEffect {
            kind: PaintEffectKind::Shader,
            source_node_id: id.clone(),
        });
    }
    for group_id in parent_group_ids {
        // NOTE: TS reads the CANONICAL node here (scene.getNodeById,
        // displayList.ts:280-285).
        let Some(group) = scene.node(group_id) else {
            continue;
        };
        if group.opacity() < 1.0 {
            effects.push(PaintEffect {
                kind: PaintEffectKind::GroupOpacity,
                source_node_id: group_id.clone(),
            });
        }
        if group.node_type() == "group" && group.truthy("clipMaskNodeId") {
            effects.push(PaintEffect {
                kind: PaintEffectKind::Mask,
                source_node_id: group_id.clone(),
            });
        }
        if group.node_type() == "container" && group.truthy("clipContent") {
            effects.push(PaintEffect {
                kind: PaintEffectKind::Mask,
                source_node_id: group_id.clone(),
            });
        }
    }
    effects
}

/// displayList.ts:289-301, over [`collect_parent_group_ids`] of the node.
fn collect_clip_chain(scene: &Scene, parent_group_ids: &[String]) -> Vec<ClipDescriptor> {
    let mut descriptors = Vec::new();
    for parent_id in parent_group_ids {
        let Some(parent) = scene.canonical_or_transient(parent_id) else {
            continue;
        };
        if parent.node_type() == "group" && parent.truthy("clipMaskNodeId") {
            descriptors.push(ClipDescriptor {
                owner_node_id: parent_id.clone(),
                mask_node_id: parent.str("clipMaskNodeId").map(str::to_string),
                mode: "group-mask",
            });
        } else if parent.node_type() == "container" && parent.truthy("clipContent") {
            descriptors.push(ClipDescriptor {
                owner_node_id: parent_id.clone(),
                mask_node_id: None,
                mode: "container-clip",
            });
        }
    }
    descriptors
}

/// displayList.ts `collectParentGroupIds`. Walks
/// `getNodeGroupId ?? getParentIdForTransientNode`, pushing only ParentNodes
/// and stepping THROUGH non-parent transient hosts (table / generated).
fn collect_parent_group_ids(scene: &Scene, node_id: &str) -> Vec<String> {
    let mut ids = Vec::new();
    // Ancestor chains are usually a few links long, but a rig of groups inside groups can be a thousand: a list
    // scanned per step made each walk quadratic in the depth.
    let mut seen = SeenIds::default();
    let mut parent_id: Option<&str> = scene
        .parent_map
        .get(node_id)
        .or_else(|| scene.transient_parent_of.get(node_id))
        .map(String::as_str);
    while let Some(pid) = parent_id {
        if !seen.insert(pid) {
            break;
        }
        let Some(parent) = scene.canonical_or_transient(pid) else {
            break;
        };
        // Non-parent transient hosts (table / generated) are stepped THROUGH
        // so their expanded children still see clip/opacity/blend ancestors
        // above the host — parity with transform_parent_chain and the TS twin
        // (displayList.ts collectParentGroupIds).
        if is_parent_node(parent) {
            ids.push(pid.to_string());
        }
        parent_id = scene
            .parent_map
            .get(pid)
            .or_else(|| scene.transient_parent_of.get(pid))
            .map(String::as_str);
    }
    ids
}

fn has_group_opacity_ancestor(scene: &Scene, parent_group_ids: &[String]) -> bool {
    parent_group_ids.iter().any(|pid| {
        scene
            .canonical_or_transient(pid)
            .map(|p| p.opacity() < 1.0)
            .unwrap_or(false)
    })
}

fn has_blend_mode_ancestor(scene: &Scene, parent_group_ids: &[String]) -> bool {
    parent_group_ids.iter().any(|pid| {
        scene
            .canonical_or_transient(pid)
            .map(|p| blend_mode_of(p).is_some())
            .unwrap_or(false)
    })
}

/// displayList.ts:331-335: textContent present, non-line, non-empty string.
fn has_shape_text(node: &RawNode) -> bool {
    node.node_type() != "line"
        && matches!(node.json_field("textContent"), Some(Value::String(s)) if !s.is_empty())
}

/// displayList.ts:337-339.
fn has_connector_text(node: &RawNode) -> bool {
    node.node_type() == "line"
        && matches!(node.json_field("textContent"), Some(Value::String(s)) if !s.is_empty())
}

/// displayList.ts `hasShadow`: the node has a drop shadow that draws
/// (`readDropShadow`, `effect::drop_shadow`).
fn has_shadow(node: &RawNode) -> bool {
    crate::effect::drop_shadow(node).is_some()
}

/// displayList.ts `hasStroke`: a visible stroke with a positive width,
/// whatever its paint.
fn has_stroke(node: &RawNode) -> bool {
    crate::paint::with_top_stroke(node, |s| s.is_some_and(|s| s.width > 0.0))
}


/// displayList.ts `hasBackgroundBlur`: backdrop-capable type AND a visible
/// backdrop effect (`readBackdrop`, `effect::backdrop`) — NOT the
/// `backdropIsActive` test (copy the bug: an inert stored backdrop still
/// flips the phase to transparent).
fn has_background_blur(node: &RawNode) -> bool {
    matches!(
        node.node_type(),
        "rectangle" | "container" | "ellipse" | "star" | "polygon" | "path"
    ) && crate::effect::backdrop(node).is_some()
}

/// displayList.ts:361-365: a rich-text linear or radial glyph fill.
fn has_gradient_text_fill(node: &RawNode) -> bool {
    matches!(
        crate::paint::node_text_fills(node).last(),
        Some(Paint::Linear(_) | Paint::Radial(_))
    )
}

/// displayList.ts:367-369: Some only when present and != 'normal'.
fn blend_mode_of(node: &RawNode) -> Option<String> {
    match node.str("blendMode") {
        Some(m) if !m.is_empty() && m != "normal" => Some(m.to_string()),
        _ => None,
    }
}

fn push_unique(reasons: &mut Vec<&'static str>, r: &'static str) {
    if !reasons.contains(&r) {
        reasons.push(r);
    }
}

/// Canvas bounds approximation for PaintItem.boundsCanvas
/// (Scene.getNodeBoundsCanvas / nodeTransformUtils.ts:351-543). Implemented
/// for AABB shapes, lines (endpoints + waypoints), paths (control-point
/// hull) and groups (child union). DEVIATIONS (documented in the M0c
/// report): curved-line bezier extents, warp bounds (legacy corner-pin AND
/// the `distort` 3×3 matrix) and blend-source bounds fall back to
/// simpler hulls — boundsCanvas is advisory metadata for M1+, not part of
/// the (nodeId, kind, z) oracle contract. (The render-plan side computes
/// true warped bounds where it matters: effects/mask.rs + backdrop/plan.rs.)
fn node_bounds_canvas(scene: &Scene, node: &RawNode, memo: &mut BuildMemo) -> Option<Rect> {
    bounds_for(scene, node, 0, memo).0
}

/// The warped corner quad (node-LOCAL px) of a distorted node, or `None` for
/// undistorted/invalid/identity cases (callers fall back to the plain box).
/// Mirrors `getNodeDistortLocalQuad` (core/scene/distortGeometry.ts): validate
/// the STORED matrix, h33-normalize, treat approximate identity as no warp,
/// then map the unit corners through the homography and scale to local px.
/// Box-based distort-capable types only (`DISTORT_CAPABLE_NODE_TYPES`).
fn distorted_local_corners(node: &RawNode, w: f64, h: f64) -> Option<[(f64, f64); 4]> {
    const CAPABLE: [&str; 5] = ["rectangle", "ellipse", "star", "polygon", "richtext"];
    if !CAPABLE.contains(&node.node_type()) || !(w > 0.0 && h > 0.0) {
        return None;
    }
    let d = node.object("distort")?;
    if d.get("type").and_then(Value::as_str) != Some("matrix") {
        return None;
    }
    let raw: Vec<f64> = d
        .get("matrix")?
        .as_array()?
        .iter()
        .filter_map(Value::as_f64)
        .collect();
    let stored: [f64; 9] = raw.try_into().ok()?;
    if !crate::distort::is_valid_distort_matrix(&stored) {
        return None;
    }
    let m = crate::distort::normalize_distort_matrix(&stored)?;
    // Approximate identity is "no warp" (persisted no-op) — TS epsilon 1e-9.
    if m.iter()
        .zip(crate::distort::IDENTITY_DISTORT.iter())
        .all(|(a, b)| (a - b).abs() < 1e-9)
    {
        return None;
    }
    let mut out = [(0.0, 0.0); 4];
    for (i, (ux, uy)) in crate::distort::UNIT_QUAD.iter().enumerate() {
        let den = m[6] * ux + m[7] * uy + m[8];
        if !(den.is_finite() && den.abs() > 1e-12) {
            return None;
        }
        let px = (m[0] * ux + m[1] * uy + m[2]) / den;
        let py = (m[3] * ux + m[4] * uy + m[5]) / den;
        if !(px.is_finite() && py.is_finite()) {
            return None;
        }
        out[i] = (px * w, py * h);
    }
    Some(out)
}

/// A node's canvas bounds at `depth` levels below the [`node_bounds_canvas`] call (deeper than
/// [`BOUNDS_MAX_DEPTH`]: none), and the height of the subtree they cover, or `None` when the depth cut-off truncated
/// them. Untruncated bounds are the same at any depth that leaves room for their subtree, so `memo` keeps those.
fn bounds_for(
    scene: &Scene,
    node: &RawNode,
    depth: usize,
    memo: &mut BuildMemo,
) -> (Option<Rect>, Option<usize>) {
    if depth > BOUNDS_MAX_DEPTH {
        return (None, None);
    }
    if let Some(&(r, height)) = memo.bounds.get(node.id()) {
        if depth + height <= BOUNDS_MAX_DEPTH {
            return (r, Some(height));
        }
    }
    let (r, height) = bounds_uncached(scene, node, depth, memo);
    if let Some(height) = height {
        memo.bounds.insert(node.id().to_string(), (r, height));
    }
    (r, height)
}

/// [`bounds_for`] without the memo lookup.
fn bounds_uncached(
    scene: &Scene,
    node: &RawNode,
    depth: usize,
    memo: &mut BuildMemo,
) -> (Option<Rect>, Option<usize>) {
    let Some(affine) = memo.canvas_affine(scene, node.id()) else {
        return (None, Some(0));
    };
    if node.node_type() == "group" {
        let children: Vec<String> = node
            .array("children")
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let mut acc: Option<(f64, f64, f64, f64)> = None;
        // The deepest level below this group the union reached; `None` once the cut-off truncated it.
        let mut height = Some(0);
        for cid in children {
            let Some(child) = scene.effective_node(&cid) else {
                continue;
            };
            if child.node_type() == "blend" {
                continue; // skip blend bounds inside groups (simplified)
            }
            let (r, child_height) = bounds_for(scene, child, depth + 1, memo);
            height = match (height, child_height) {
                (Some(h), Some(c)) => Some(h.max(c + 1)),
                _ => None,
            };
            if let Some(r) = r {
                let e = (r.x, r.y, r.x + r.width, r.y + r.height);
                acc = Some(match acc {
                    None => e,
                    Some(a) => (a.0.min(e.0), a.1.min(e.1), a.2.max(e.2), a.3.max(e.3)),
                });
            }
        }
        let Some(t) = memo.canvas_transform(scene, node.id()) else {
            return (None, height);
        };
        let (x0, y0, x1, y1) = acc.unwrap_or((t.x, t.y, t.x, t.y));
        return (
            Some(Rect {
                x: x0,
                y: y0,
                width: x1 - x0,
                height: y1 - y0,
            }),
            height,
        );
    }
    (leaf_bounds(scene, node, affine, memo), Some(0))
}

/// [`bounds_for`] of a node that is not a group, given its canvas affine.
fn leaf_bounds(
    scene: &Scene,
    node: &RawNode,
    affine: Affine,
    memo: &mut BuildMemo,
) -> Option<Rect> {
    match node.node_type() {
        "line" => {
            let pts = node.points()?;
            if pts.len() < 4 {
                return None;
            }
            let p: Vec<f64> = pts.iter().filter_map(|v| v.as_f64()).collect();
            if p.len() < 4 {
                return None;
            }
            // Endpoints/waypoints are parent-space; transform through the
            // PARENT affine (line nodes have no own x/y).
            let parent_affine = scene
                .transform_parent_chain(node.id())
                .first()
                .and_then(|pid| memo.canvas_affine(scene, pid))
                .unwrap_or_else(|| {
                    let page_pos = scene
                        .page_of(node.id())
                        .and_then(|pid| scene.doc.pages.iter().position(|pg| pg.id() == pid))
                        .map(|i| scene.page_position(i))
                        .unwrap_or((0.0, 0.0));
                    transform::translate(page_pos.0, page_pos.1)
                });
            let mut points = vec![(p[0], p[1]), (p[2], p[3])];
            if let Some(wps) = node.array("waypoints") {
                for wp in wps {
                    if let Some(o) = wp.as_object() {
                        if let (Some(x), Some(y)) = (
                            o.get("x").and_then(Value::as_f64),
                            o.get("y").and_then(Value::as_f64),
                        ) {
                            points.push((x, y));
                        }
                    }
                }
            }
            let mut min = (f64::INFINITY, f64::INFINITY);
            let mut max = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for pt in points {
                let w = transform::apply_point(pt, parent_affine);
                min = (min.0.min(w.0), min.1.min(w.1));
                max = (max.0.max(w.0), max.1.max(w.1));
            }
            Some(Rect {
                x: min.0,
                y: min.1,
                width: max.0 - min.0,
                height: max.1 - min.1,
            })
        }
        "path" => {
            let b = transform::bounds_from_points(node.points()?)?;
            let (x0, y0, x1, y1) = transform::transformed_aabb(
                (b.min_x, b.min_y, b.max_x - b.min_x, b.max_y - b.min_y),
                affine,
            );
            Some(Rect {
                x: x0,
                y: y0,
                width: x1 - x0,
                height: y1 - y0,
            })
        }
        _ => {
            let w = node.f64("width");
            let h = node.f64("height");
            match (w, h) {
                (Some(w), Some(h)) => {
                    // Warped-node parity (scene.bounds.ts getNodeDistortCanvasBounds):
                    // a distorted node's canvas bounds are the AABB of its
                    // WARPED corner quad, not the unwarped box — without this,
                    // warp overhang would be cropped from canvas-root framing.
                    if let Some(corners) = distorted_local_corners(node, w, h) {
                        let mut min = (f64::INFINITY, f64::INFINITY);
                        let mut max = (f64::NEG_INFINITY, f64::NEG_INFINITY);
                        for c in corners {
                            let p = transform::apply_point(c, affine);
                            min = (min.0.min(p.0), min.1.min(p.1));
                            max = (max.0.max(p.0), max.1.max(p.1));
                        }
                        if min.0.is_finite() && min.1.is_finite() && max.0 > min.0 && max.1 > min.1
                        {
                            return Some(Rect {
                                x: min.0,
                                y: min.1,
                                width: max.0 - min.0,
                                height: max.1 - min.1,
                            });
                        }
                    }
                    let (x0, y0, x1, y1) = transform::transformed_aabb((0.0, 0.0, w, h), affine);
                    Some(Rect {
                        x: x0,
                        y: y0,
                        width: x1 - x0,
                        height: y1 - y0,
                    })
                }
                _ => {
                    let t = memo.canvas_transform(scene, node.id())?;
                    Some(Rect {
                        x: t.x,
                        y: t.y,
                        width: 0.0,
                        height: 0.0,
                    })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Scene;
    use crate::value::json;

    fn tuples(dl: &DisplayList) -> Vec<(String, String, usize)> {
        dl.items
            .iter()
            .map(|i| (i.node_id.clone(), i.kind.as_str().to_string(), i.z_index))
            .collect()
    }

    #[test]
    fn page_bg_first_then_nodes_in_order() {
        let s = Scene::load(&json!({
            "nodes": [
                {"id":"r1","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#f00"}}]},
                {"id":"t1","type":"richtext","x":0.0,"y":20.0,"width":100.0,"height":40.0,
                 "rotation":0,"scale":1,"htmlContent":"<p>hi</p>"}
            ],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                       "nodeIds":["r1","t1"]}]
        })).unwrap();
        let dl = build_display_list(&s, "p1");
        assert_eq!(
            tuples(&dl),
            vec![
                ("__page_bg_p1".to_string(), "pageBackground".to_string(), 0),
                ("r1".to_string(), "shapeFill".to_string(), 1),
                ("t1".to_string(), "richtext".to_string(), 2),
            ]
        );
        assert_eq!(dl.items[0].phase, PaintPhase::Background);
        assert_eq!(dl.items[1].phase, PaintPhase::Opaque);
        assert_eq!(dl.items[2].phase, PaintPhase::Transparent);
    }

    #[test]
    fn canvas_root_content_bounds_unions_floating_nodes() {
        // The page-less freeform framing source: union canvas-absolute AABB of
        // the canvas-root nodes (rect 100,80 300x200 ∪ ellipse 450,150 180x180
        // → (100,80)-(630,330)); pages (none here) contribute nothing.
        let s = Scene::load(&json!({
            "layoutMode": "freeform",
            "nodes": [
                {"id":"float1","type":"rectangle","x":100.0,"y":80.0,"width":300.0,"height":200.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#e94040"}}]},
                {"id":"float2","type":"ellipse","x":450.0,"y":150.0,"width":180.0,"height":180.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#2255ee"}}]}
            ],
            "pages": [],
            "canvasRootNodeIds": ["float1","float2"]
        })).unwrap();
        let b = canvas_root_content_bounds(&s).expect("floating nodes have bounds");
        assert_eq!((b.x, b.y, b.width, b.height), (100.0, 80.0, 530.0, 250.0));

        // No roots → nothing to frame.
        let empty = Scene::load(&json!({"nodes": [], "pages": [], "canvasRootNodeIds": []})).unwrap();
        assert!(canvas_root_content_bounds(&empty).is_none());
    }

    #[test]
    fn canvas_root_content_bounds_ignore_what_a_clipping_frame_hides() {
        // A layer parked 20000 px outside its clipping frame doesn't widen the framing (it widened it past the GPU
        // limit, so the frame rendered at a reduced scale); a layer straddling the edge counts up to the edge.
        let frame = |clip: bool| Scene::load(&json!({
            "layoutMode": "freeform",
            "nodes": [
                {"id":"F","type":"container","x":0.0,"y":0.0,"width":800.0,"height":600.0,"clipContent":clip,
                 "children":["far","edge"]},
                {"id":"far","type":"rectangle","x":20000.0,"y":10.0,"width":100.0,"height":100.0,
                 "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#e94040"}}]},
                {"id":"edge","type":"rectangle","x":700.0,"y":-50.0,"width":200.0,"height":100.0,
                 "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#2255ee"}}]}
            ],
            "pages": [],
            "canvasRootNodeIds": ["F"]
        })).unwrap();
        let b = canvas_root_content_bounds(&frame(true)).expect("bounds");
        assert_eq!((b.x, b.y, b.width, b.height), (0.0, 0.0, 800.0, 600.0));
        let b = canvas_root_content_bounds(&frame(false)).expect("bounds");
        assert_eq!((b.x, b.y, b.width, b.height), (0.0, -50.0, 20100.0, 650.0));
    }

    #[test]
    fn canvas_root_bounds_include_distort_warp_overhang() {
        // Warped-node framing parity (scene.bounds.ts getNodeDistortCanvasBounds):
        // a distorted root's warp overhang must widen the frame. The matrix
        // scales unit-x by 2, so a 100×50 node at (10,20) warps to local
        // corners spanning 0..200 → canvas bounds 10..210 (not 10..110).
        let s = Scene::load(&json!({
            "layoutMode": "freeform",
            "nodes": [
                {"id":"warped","type":"rectangle","x":10.0,"y":20.0,"width":100.0,"height":50.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#e94040"}}],
                 "distort":{"type":"matrix","matrix":[2.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]}}
            ],
            "pages": [],
            "canvasRootNodeIds": ["warped"]
        })).unwrap();
        let b = canvas_root_content_bounds(&s).expect("warped node has bounds");
        assert_eq!((b.x, b.y), (10.0, 20.0));
        assert!(
            (b.width - 200.0).abs() < 1e-6,
            "warp overhang must widen the frame (got width {})",
            b.width
        );
        assert!((b.height - 50.0).abs() < 1e-6, "got height {}", b.height);

        // An identity distort is "no warp" — plain box bounds.
        let s2 = Scene::load(&json!({
            "layoutMode": "freeform",
            "nodes": [
                {"id":"plain","type":"rectangle","x":10.0,"y":20.0,"width":100.0,"height":50.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#e94040"}}],
                 "distort":{"type":"matrix","matrix":[1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]}}
            ],
            "pages": [],
            "canvasRootNodeIds": ["plain"]
        })).unwrap();
        let b2 = canvas_root_content_bounds(&s2).expect("bounds");
        assert_eq!((b2.width, b2.height), (100.0, 50.0));
    }

    #[test]
    fn canvas_root_display_list_has_floating_nodes_no_page_bg() {
        // The freeform canvas list holds the floating (off-page) nodes
        // in z-order, with NO page-background prefix and NO page-parented nodes.
        let s = Scene::load(&json!({
            "layoutMode": "freeform",
            "nodes": [
                {"id":"onpage","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#f00"}}]},
                {"id":"floata","type":"rectangle","x":900.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#0f0"}}]},
                {"id":"floatb","type":"rectangle","x":920.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#00f"}}]}
            ],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                       "nodeIds":["onpage"],"freeformX":0.0,"freeformY":0.0}],
            "canvasRootNodeIds": ["floata","floatb"]
        })).unwrap();
        let dl = build_display_list_for_canvas(&s);
        let ids: Vec<&str> = dl.items.iter().map(|i| i.node_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["floata", "floatb"],
            "canvas list = floating nodes in order (no page bg, no page-parented nodes)"
        );
        assert!(
            dl.items.iter().all(|i| i.kind != PaintKind::PageBackground),
            "pasteboard has no page-background fill"
        );
    }

    #[test]
    fn shape_text_emits_secondary_item_with_same_z() {
        // displayList.ts:151-161
        let s = Scene::load(&json!({
            "nodes": [
                {"id":"r1","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#f00"}}],"textContent":"<p>in shape</p>"}
            ],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["r1"]}]
        })).unwrap();
        let dl = build_display_list(&s, "p1");
        assert_eq!(
            tuples(&dl)[1..],
            [
                ("r1".to_string(), "shapeFill".to_string(), 1),
                ("r1".to_string(), "shapeText".to_string(), 1),
            ]
        );
        assert!(!dl.items[2].is_primary_node_item);
        assert!(dl.items[2]
            .fallback_reasons
            .contains(&"intra-node-ordering"));
    }

    #[test]
    fn connector_text_item_for_lines() {
        let s = Scene::load(&json!({
            "nodes": [
                {"id":"l1","type":"line","points":[0.0,0.0,50.0,50.0],"strokes":[{"id":"stroke-0",
                 "paint":{"type":"solid","color":"#000"},"width":2.0}],"textContent":"label"}
            ],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["l1"]}]
        })).unwrap();
        let dl = build_display_list(&s, "p1");
        assert_eq!(dl.items[2].kind, PaintKind::ConnectorText);
        // stroke effect present (displayList.ts:273) → visual-effect reason
        assert!(dl.items[1].fallback_reasons.contains(&"visual-effect"));
    }

    #[test]
    fn blend_node_is_shape_fill_with_unsupported_reason() {
        let s = Scene::load(&json!({
            "nodes": [
                {"id":"a","type":"rectangle","x":0.0,"y":0.0,"width":5.0,"height":5.0,"rotation":0,"scale":1},
                {"id":"b","type":"rectangle","x":50.0,"y":50.0,"width":5.0,"height":5.0,"rotation":0,"scale":1},
                {"id":"bl","type":"blend","sourceIds":["a","b"],"spacing":"steps","steps":3}
            ],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["a","b","bl"]}]
        })).unwrap();
        let dl = build_display_list(&s, "p1");
        let blend_item = dl.items.iter().find(|i| i.node_id == "bl").unwrap();
        assert_eq!(blend_item.kind, PaintKind::ShapeFill);
        assert!(blend_item
            .fallback_reasons
            .contains(&"unsupported-node-kind"));
    }

    #[test]
    fn group_boundary_and_clip_chain() {
        let s = Scene::load(&json!({
            "nodes": [
                {"id":"g","type":"group","x":0.0,"y":0.0,"children":["m","c"],
                 "clipMaskNodeId":"m"},
                {"id":"m","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,"rotation":0,"scale":1},
                {"id":"c","type":"rectangle","x":2.0,"y":2.0,"width":5.0,"height":5.0,"rotation":0,"scale":1}
            ],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["g"]}]
        })).unwrap();
        let dl = build_display_list(&s, "p1");
        let g = dl.items.iter().find(|i| i.node_id == "g").unwrap();
        assert_eq!(g.kind, PaintKind::GroupBoundary);
        assert_eq!(g.phase, PaintPhase::Overlay);
        let m = dl.items.iter().find(|i| i.node_id == "m").unwrap();
        assert!(m.fallback_reasons.contains(&"text-mask-source"));
        let c = dl.items.iter().find(|i| i.node_id == "c").unwrap();
        assert!(c.fallback_reasons.contains(&"ancestor-clip-mask"));
        assert_eq!(c.clip_chain.len(), 1);
        assert_eq!(c.clip_chain[0].mode, "group-mask");
    }

    #[test]
    fn generated_raster_kind_is_image_when_not_expanded() {
        // With expandTransientNodes=true the host never appears; check the
        // kind classification path directly through a vector-asset host that
        // has no cachedAssetUrl... the kind logic itself:
        let n = RawNode::from_value(json!({
            "id":"gen","type":"generated","generatedKind":"latex","renderMode":"raster-asset"
        }))
        .unwrap();
        assert_eq!(super::primary_paint_kind(&n), PaintKind::Image);
        let n2 = RawNode::from_value(json!({
            "id":"gen","type":"generated","generatedKind":"latex","renderMode":"vector-asset"
        }))
        .unwrap();
        assert_eq!(super::primary_paint_kind(&n2), PaintKind::Svg);
    }

    #[test]
    fn opacity_blend_shadow_force_transparent_phase() {
        // displayList.ts:225-232
        let s = Scene::load(&json!({
            "nodes": [
                {"id":"o","type":"rectangle","x":0.0,"y":0.0,"width":1.0,"height":1.0,
                 "rotation":0,"scale":1,"opacity":0.5,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#000"}}]},
                {"id":"b","type":"rectangle","x":0.0,"y":0.0,"width":1.0,"height":1.0,
                 "rotation":0,"scale":1,"blendMode":"multiply","fills":[{"id":"fill-0","paint":{"type":"solid","color":"#000"}}]},
                {"id":"sh","type":"rectangle","x":0.0,"y":0.0,"width":1.0,"height":1.0,
                 "rotation":0,"scale":1,"effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000","blur":4.0}],"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#000"}}]}
            ],
            "pages": [{"id":"p1","name":"P","width":10.0,"height":10.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                       "nodeIds":["o","b","sh"]}]
        })).unwrap();
        let dl = build_display_list(&s, "p1");
        for id in ["o", "b", "sh"] {
            let item = dl.items.iter().find(|i| i.node_id == id).unwrap();
            assert_eq!(item.phase, PaintPhase::Transparent, "{id}");
        }
        let b = dl.items.iter().find(|i| i.node_id == "b").unwrap();
        assert_eq!(b.blend_mode.as_deref(), Some("multiply"));
        assert!(b.fallback_reasons.contains(&"blend-mode"));
    }

    /// `depth` groups, each inside the last and turned a little, each also holding a rect and a line, around a
    /// path: a rig of groups inside groups (#1715), deeper than the bounds cut-off.
    fn deep_rig(depth: usize) -> Scene {
        let mut nodes = Vec::new();
        for i in 0..depth {
            let inner = if i + 1 < depth {
                format!("g{}", i + 1)
            } else {
                "leaf".to_string()
            };
            nodes.push(json!({"id": format!("g{i}"), "type": "group", "x": 1.5, "y": 0.25,
                "rotation": 2.0, "scaleX": 1.01, "children": [inner, format!("r{i}"), format!("l{i}")]}));
            nodes.push(
                json!({"id": format!("r{i}"), "type": "rectangle", "x": 0.5, "y": -0.5,
                "width": 4.0, "height": 3.0, "rotation": 7.0, "scale": 1}),
            );
            nodes.push(
                json!({"id": format!("l{i}"), "type": "line", "points": [0.0, 0.0, 6.0, 2.0]}),
            );
        }
        nodes.push(json!({"id": "leaf", "type": "path", "x": 2.0, "y": 3.0,
            "points": [0.0, 0.0, 5.0, 5.0, 10.0, 0.0]}));
        Scene::load(
            &json!({"layoutMode": "freeform", "nodes": nodes, "pages": [],
            "canvasRootNodeIds": ["g0"]}),
        ).unwrap()
    }

    #[test]
    fn a_deep_rig_computes_each_canvas_transform_once() {
        // Each node's canvas transform is a walk up its whole ancestor chain. The builder used to redo it for
        // every group whose bounds include the node (up to 32 levels up) and twice more for the node's own
        // item: ~70 walks a node, a 1,000-deep chain took seconds a build. Now once a node, whatever the depth.
        for depth in [50usize, 400] {
            let scene = deep_rig(depth);
            let nodes = 3 * depth + 1;
            crate::scene::ACCUMULATED_AFFINE_CALLS.with(|n| n.set(0));
            let dl = build_display_list_for_canvas(&scene);
            let calls = crate::scene::ACCUMULATED_AFFINE_CALLS.with(|n| n.get());
            assert_eq!(dl.items.len(), nodes);
            assert!(
                calls <= nodes,
                "depth {depth}: {calls} transform walks for {nodes} nodes"
            );
            // The content bounds share one memo with the list they build.
            crate::scene::ACCUMULATED_AFFINE_CALLS.with(|n| n.set(0));
            assert!(canvas_root_content_bounds(&scene).is_some());
            let calls = crate::scene::ACCUMULATED_AFFINE_CALLS.with(|n| n.get());
            assert!(
                calls <= nodes,
                "depth {depth}: content bounds took {calls} transform walks for {nodes} nodes"
            );
        }
    }

    #[test]
    fn a_memoised_build_matches_each_node_on_its_own() {
        // The memo changes no value: every item's transforms are the scene's own, and its bounds those of the node
        // measured alone, bit for bit, past the bounds cut-off too.
        let scene = deep_rig(80);
        let dl = build_display_list_for_canvas(&scene);
        for item in &dl.items {
            let id = item.node_id.as_str();
            assert_eq!(item.world_affine, scene.node_canvas_affine(id), "{id}");
            assert_eq!(
                item.world_transform,
                scene.node_canvas_transform(id),
                "{id}"
            );
            let node = scene.effective_node(id).unwrap();
            let alone = node_bounds_canvas(&scene, node, &mut BuildMemo::default());
            assert_eq!(item.bounds_canvas, alone, "{id}");
        }
    }
}
