//! Exact backdrop-dependent blend modes, via the backdrop pass split.
//!
//! The body pipelines blend with fixed-function state (manifest `blend_modes`). `multiply` and `screen` can be
//! written that way. The other modes cannot: overlay picks multiply or screen from the BACKDROP's value,
//! darken/lighten compare the two colours, the dodges and burns divide by them, and the non-separable hue,
//! saturation, color and luminosity mix all three channels of both (§10.2). Such a node's body
//! (`BlendMode::split_id`) is instead drawn between two splits on the SEAMS §7 pass-split stream:
//!
//! 1. `BlendBegin`: copy the node's region of the target (the backdrop B) aside, then clear that region to
//!    transparent.
//! 2. The node's body items draw normally (source-over) into the cleared region, leaving exactly its own
//!    premultiplied paint S: fills, strokes, masks and inner shadows all go through their usual pipelines.
//! 3. `BlendEnd`: copy S out, and write the W3C compositing result for `mix-blend-mode` back into the region
//!    (premultiplied): `co = cs·(1 − αb) + cb·(1 − αs) + αs·αb·B(Cb, Cs)`, `αo = αs + αb·(1 − αs)`, with
//!    `B` the node's blend function (Compositing and Blending Level 1, §5.8 and §10.1). Blending runs on the
//!    stored (gamma) values, as CSS does.
//!
//! Unlike backdrop effects, pages carrying such a node keep MSAA: each split ends the pass (resolving the
//! multisample target), copies the region from the resolved target, and draws its clear / composite into the
//! multisample view with the frame target as resolve target, so every sample of a region pixel holds the
//! composite and the next segment (which loads the multisample view) builds on it. The only AA lost is inside a
//! split region, where each pixel's samples collapse to its resolved value, which is what the pixel shows anyway.
//! They also keep transparent exports: the formula is exact over a transparent backdrop.
//! Multiply joins them on a transparent target (`BlendMode::split_id_over`): its fixed-function state
//! (`cs·cb + cb·(1 − αs)`) drops the `cs·(1 − αb)` term, so over nothing it painted black.

use anyhow::{bail, Result};
use moda_scene::types::RawNode;
use moda_scene::Scene;
use std::cell::RefCell;

use super::plan::{node_canvas_aabb, PlanCtx, Rect};
use crate::seams::{BackdropOp, BackdropSplit, BlendMode, CoverageSource, INFINITE_CLIP};

/// The split-composited blend mode a node asks for, if any (`BlendMode::split_id`). Every mode but normal,
/// multiply and screen: those are exact fixed-function states over the opaque backdrops pages have.
pub fn split_blend_mode(node: &RawNode) -> Option<u32> {
    BlendMode::from_node_value(node.str("blendMode")).split_id()
}

/// As [`split_blend_mode`], for a target that may be transparent (`BlendMode::split_id_over`): multiply needs
/// the split there too.
pub fn split_blend_mode_over(node: &RawNode, transparent: bool) -> Option<u32> {
    BlendMode::from_node_value(node.str("blendMode")).split_id_over(transparent)
}

/// Whether a node's blend mode applies to its whole subtree, flattened (as in Figma): a group, or a frame with
/// children. Its descendants draw into a blend split that composites them together with the node's mode.
pub fn is_blend_subtree_node(node: &RawNode) -> bool {
    match node.node_type() {
        "group" => true,
        "container" => node.array("children").is_some_and(|c| !c.is_empty()),
        _ => false,
    }
}

/// Whether a leaf layer's own draw items can't carry its blend mode as a body's blend state, so they blend as one
/// layer inside a split (as in Figma, where a layer's blend mode applies to its whole drawn result): text (glyph
/// batches, or a vector-lane text's bodies, with its effects) and a line (its stroke, arrowheads and label).
pub fn is_blend_unit_leaf(node: &RawNode) -> bool {
    matches!(node.node_type(), "richtext" | "line")
}

/// The blend split id ([`BlendMode::exact_split_id`]) of a node whose blend mode no fixed-function state can
/// carry, whatever the mode: a subtree's ([`is_blend_subtree_node`]), a text's or a line's
/// ([`is_blend_unit_leaf`]) or a layer-filter capture's composite quad (the filtered body is blended as a whole).
/// `None` for normal blending and every other node.
pub fn composite_blend_split(node: &RawNode) -> Option<u32> {
    let mode = BlendMode::from_node_value(node.str("blendMode"));
    if mode == BlendMode::Normal {
        return None;
    }
    (is_blend_subtree_node(node)
        || is_blend_unit_leaf(node)
        || crate::layer_blur::has_active_layer_filter(node))
    .then(|| mode.exact_split_id())
}

/// The fixed-function composite blend of a layer-filtered leaf's capture, when its own blend mode has an exact
/// fixed-function state over the target: screen (`S + D (1 − S)`, exact over any backdrop) and, over an opaque
/// target, multiply (`S D + D (1 − αS)`). The filtered result then composites with that state in the frame's pass,
/// instead of inside a blend-split pair (two full-target passes and resolves per node, every frame, and a
/// camera-dependent page). Only for a leaf (not a group, frame with children, text or line), drawn over the
/// frame itself: no ancestor captures, blends, fades as a group or carries effects (their targets start cleared).
pub fn filter_composite_fixed_blend(
    scene: &Scene,
    id: &str,
    transparent: bool,
) -> Option<crate::seams::CaptureBlend> {
    use crate::seams::CaptureBlend;
    let node = scene.effective_node(id)?;
    let blend = match BlendMode::from_node_value(node.str("blendMode")) {
        BlendMode::Screen => CaptureBlend::Screen,
        BlendMode::Multiply if !transparent => CaptureBlend::Multiply,
        _ => return None,
    };
    // A moving frame's motion unit (`crate::motion_capture`) that is a group or a frame with children: its whole
    // subtree is the capture, so its mode blends the composite as it would the group's split.
    let motion_subtree = is_blend_subtree_node(node) && crate::motion_capture::is_unit(id);
    if (is_blend_subtree_node(node) && !motion_subtree)
        || is_blend_unit_leaf(node)
        || !crate::layer_blur::has_active_layer_filter(node)
        || !(motion_subtree
            || matches!(crate::paint::layers::plan(node), crate::paint::layers::LayerPlan::Single(_)))
    {
        return None;
    }
    let mut seen = moda_scene::scene::SeenIds::default();
    let parent = |id: &str| {
        scene
            .parent_map()
            .get(id)
            .or_else(|| scene.transient_parent_of().get(id))
            .map(String::as_str)
    };
    let mut cur = parent(id);
    while let Some(pid) = cur {
        if !seen.insert(pid) {
            break;
        }
        let Some(p) = scene.effective_node(pid) else {
            break;
        };
        let blended = BlendMode::from_node_value(p.str("blendMode")) != BlendMode::Normal;
        let effects = p.array("effects").is_some_and(|e| !e.is_empty());
        let faded = moda_scene::scene::is_parent_node(p) && p.f64("opacity").unwrap_or(1.0) < 1.0;
        if blended || effects || faded || crate::layer_blur::plan::needs_layer_capture(p) {
            return None;
        }
        cur = parent(pid);
    }
    Some(blend)
}

/// [`composite_blend_split`] of the node `id`, on a target that may be `transparent`: `None` too when its
/// composite blends with a fixed-function state ([`filter_composite_fixed_blend`]).
pub fn composite_blend_split_in(scene: &Scene, id: &str, transparent: bool) -> Option<u32> {
    let node = scene.effective_node(id)?;
    let split = composite_blend_split(node)?;
    filter_composite_fixed_blend(scene, id, transparent)
        .is_none()
        .then_some(split)
}

pub fn page_has_split_blend(scene: &Scene, page_id: &str) -> bool {
    page_has_split_blend_over(scene, page_id, false)
}

/// Whether a page renders any split blend when its target is `transparent` (a transparent export).
pub fn page_has_split_blend_over(scene: &Scene, page_id: &str, transparent: bool) -> bool {
    scene
        .flattened_z_order_for_page(page_id)
        .iter()
        .any(|id| node_splits(scene, id, transparent))
}

/// Whether a page's splits all come from paint stacks (E3.1 P6), none from a node's own blend mode: its rect
/// and ellipse bodies keep the analytic SDF path (their edges stay antialiased on the single-sample target).
pub fn page_splits_are_stacks_only(scene: &Scene, page_id: &str, transparent: bool) -> bool {
    !scene
        .flattened_z_order_for_page(page_id)
        .iter()
        .filter_map(|id| scene.effective_node(id).map(|n| (id, n)))
        .any(|(id, n)| {
            moda_scene::scene::is_opacity_group(n)
                || split_blend_mode_over(n, transparent).is_some()
                || composite_blend_split_in(scene, id, transparent).is_some()
        })
}

pub fn canvas_root_has_split_blend(scene: &Scene) -> bool {
    canvas_root_has_split_blend_over(scene, false)
}

/// Whether the canvas root renders any split blend when its target is `transparent`.
pub fn canvas_root_has_split_blend_over(scene: &Scene, transparent: bool) -> bool {
    scene
        .flattened_z_order_for_canvas()
        .iter()
        .any(|id| node_splits(scene, id, transparent))
}

/// Whether a node draws inside blend splits: its own blend mode (its body's, its subtree's or its layer-filter
/// composite's), or its paint stack's (E3.1 P6).
pub(crate) fn node_splits(scene: &Scene, id: &str, transparent: bool) -> bool {
    scene.effective_node(id).is_some_and(|n| {
        moda_scene::scene::is_opacity_group(n)
            || split_blend_mode_over(n, transparent).is_some()
            || composite_blend_split_in(scene, id, transparent).is_some()
            || crate::paint::layers::stack_needs_split(
                n,
                scene.ancestor_opacity_product(id),
                transparent,
            )
    })
}

/// Whether the node's canvas AABB bounds everything it draws. A warp is not in `node_canvas_aabb` (see its note), so
/// a warped node uses the whole target instead.
fn aabb_is_tight(node: &RawNode) -> bool {
    crate::paint::node_distort(node).is_none()
}

/// The `BlendBegin` / `BlendEnd` pair wrapping an overlay node's body, or `None` to leave the node on its
/// fixed-function fallback (not a split blend, or nothing of it lands on the target).
pub fn compute_blend_splits(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
) -> Option<(BackdropSplit, BackdropSplit)> {
    let mode = split_blend_mode_over(node, ctx.transparent_target)?;
    split_pair(scene, node, ctx, BackdropOp::BlendEnd { mode })
}

/// A `BlendBegin` / `StackEnd` pair over `node`'s region (E3.1 P6): a paint stack's transparency group, or one
/// blended layer of it. `mode` is [`BlendMode::exact_split_id`]; `opacity` scales the group. `None` when nothing
/// of the node lands on the target.
pub fn compute_stack_splits(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    mode: BlendMode,
    opacity: f64,
) -> Option<(BackdropSplit, BackdropSplit)> {
    split_pair(
        scene,
        node,
        ctx,
        BackdropOp::StackEnd {
            mode: mode.exact_split_id(),
            opacity: opacity.clamp(0.0, 1.0) as f32,
        },
    )
}

/// A `GroupBegin` / `StackEnd` pair over `node`'s region (E3.1 P6): a paint stack drawn onto the backdrop as a
/// non-isolated group, faded by the node's `opacity`. `None` when nothing of the node lands on the target.
pub fn compute_group_splits(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    opacity: f64,
) -> Option<(BackdropSplit, BackdropSplit)> {
    split_pair_from(
        scene,
        node,
        ctx,
        BackdropOp::GroupBegin,
        BackdropOp::StackEnd {
            mode: crate::seams::STACK_FADE_MODE,
            opacity: opacity.clamp(0.0, 1.0) as f32,
        },
    )
}

fn split_pair(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    end: BackdropOp,
) -> Option<(BackdropSplit, BackdropSplit)> {
    split_pair_from(scene, node, ctx, BackdropOp::BlendBegin, end)
}

/// A `BlendBegin` / `BlendEnd` pair blending everything drawn between them with split mode `mode`
/// ([`composite_blend_split`]) over the canvas rect `bounds` (a subtree's or a layer-filter composite's drawn
/// extent; `None` = the whole target). `None` when nothing of it lands on the target.
pub(crate) fn compute_region_blend_splits(
    node: &RawNode,
    ctx: &PlanCtx,
    bounds: Option<Rect>,
    mode: u32,
) -> Option<(BackdropSplit, BackdropSplit)> {
    let projected = bounds
        .and_then(|b| ctx.project_rect(&b))
        .filter(|p| p.finite())
        // 2px for antialiased edges.
        .map(|p| Rect {
            x: p.x - 2.0,
            y: p.y - 2.0,
            w: p.w + 4.0,
            h: p.h + 4.0,
        });
    split_pair_in(
        node,
        ctx,
        projected,
        BackdropOp::BlendBegin,
        BackdropOp::BlendEnd { mode },
    )
}

/// A `BlendBegin` / `StackEnd` pair compositing everything drawn between them as one layer, faded by `opacity`, with
/// split mode `mode` ([`BlendMode::exact_split_id`]; 0 is source-over) over the canvas rect `bounds`: a group whose
/// opacity applies to it as a whole ([`moda_scene::scene::is_opacity_group`]). `None` when nothing of it lands.
pub(crate) fn compute_region_fade_splits(
    node: &RawNode,
    ctx: &PlanCtx,
    bounds: Option<Rect>,
    mode: u32,
    opacity: f64,
) -> Option<(BackdropSplit, BackdropSplit)> {
    let projected = bounds
        .and_then(|b| ctx.project_rect(&b))
        .filter(|p| p.finite())
        .map(|p| Rect {
            x: p.x - 2.0,
            y: p.y - 2.0,
            w: p.w + 4.0,
            h: p.h + 4.0,
        });
    split_pair_in(
        node,
        ctx,
        projected,
        BackdropOp::BlendBegin,
        BackdropOp::StackEnd {
            mode,
            opacity: opacity.clamp(0.0, 1.0) as f32,
        },
    )
}

fn split_pair_from(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    begin: BackdropOp,
    end: BackdropOp,
) -> Option<(BackdropSplit, BackdropSplit)> {
    let mut projected = None;
    if aabb_is_tight(node) {
        if let (Some(bounds), Some(affine)) = (
            node_canvas_aabb(scene, node),
            scene.node_canvas_affine(node.id()),
        ) {
            // A stroke reaches up to its full width outside the geometry (outside alignment); widths are local
            // units, so scale by the larger affine axis. Every stroke layer draws, so the widest one pads.
            let axis = (affine.a.hypot(affine.b)).max(affine.c.hypot(affine.d));
            let pad = crate::paint::layers::widest_stroke(node)
                .map_or(0.0, |s| s.width)
                .max(0.0)
                * axis;
            let padded = Rect {
                x: bounds.x - pad,
                y: bounds.y - pad,
                w: bounds.w + pad * 2.0,
                h: bounds.h + pad * 2.0,
            };
            if let Some(p) = ctx.project_rect(&padded).filter(|p| p.finite()) {
                // 2px for antialiased edges.
                projected = Some(Rect {
                    x: p.x - 2.0,
                    y: p.y - 2.0,
                    w: p.w + 4.0,
                    h: p.h + 4.0,
                });
            }
        }
    }
    split_pair_in(node, ctx, projected, begin, end)
}

/// The split pair over `region` (target px; `None` = the whole target), narrowed to the page clip and the target.
fn split_pair_in(
    node: &RawNode,
    ctx: &PlanCtx,
    region: Option<Rect>,
    begin: BackdropOp,
    end: BackdropOp,
) -> Option<(BackdropSplit, BackdropSplit)> {
    let target = Rect {
        x: 0.0,
        y: 0.0,
        w: ctx.width_px as f64,
        h: ctx.height_px as f64,
    };
    // A fade pair exists whether or not it lands on this target: what is drawn between it is one layer wherever
    // it is drawn. Nothing landing here is an empty region, which the executor skips; a layer capture that takes
    // these items gives the pair its own region ([`retarget_isolation_splits`]).
    let landed = (|| {
        let mut region = region.unwrap_or(target);
        if ctx.page_clip != INFINITE_CLIP {
            let clip = Rect {
                x: ctx.page_clip[0] as f64,
                y: ctx.page_clip[1] as f64,
                w: ctx.page_clip[2] as f64,
                h: ctx.page_clip[3] as f64,
            };
            if let Some(p) = ctx.project_rect(&clip).filter(|p| p.finite()) {
                region = region.intersect(&p)?;
            }
        }
        let region = region.intersect(&target)?;
        let x0 = region.x.floor().max(0.0);
        let y0 = region.y.floor().max(0.0);
        let x1 = (region.x + region.w).ceil().min(target.w);
        let y1 = (region.y + region.h).ceil().min(target.h);
        (x1 > x0 && y1 > y0).then_some([x0, y0, x1, y1])
    })();
    // A FADE pair exists whether or not it lands here. A blend pair that lands nowhere is not made, as before:
    // its bodies keep their own blend, and a capture that reaches them plans as it did.
    let fade = matches!(end, BackdropOp::StackEnd { mode, .. } if mode == 0 || mode == crate::seams::STACK_FADE_MODE);
    let [x0, y0, x1, y1] = match landed {
        Some(rect) => rect,
        None if fade => [0.0; 4],
        None => return None,
    };
    Some((
        isolation_split(node.id(), begin, [x0, y0, x1, y1]),
        isolation_split(node.id(), end, [x0, y0, x1, y1]),
    ))
}

/// One split of an isolation pair over the whole-pixel target rect `[x0, y0, x1, y1]`.
fn isolation_split(node_id: &str, op: BackdropOp, [x0, y0, x1, y1]: [f64; 4]) -> BackdropSplit {
    let region_px = [x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32];
    let rect = [x0 as f32, y0 as f32, (x1 - x0) as f32, (y1 - y0) as f32];
    BackdropSplit {
        node_id: node_id.to_string(),
        op,
        region_px,
        coverage: CoverageSource::Rect {
            size: [rect[2], rect[3]],
            radii: [0.0; 4],
        },
        node_rect_px: rect,
        visible_rect_px: rect,
        rotation_rad: 0.0,
        local_frame_y_sign: 1.0,
        local_half_size_px: [rect[2] * 0.5, rect[3] * 0.5],
        clip_rrect_px: None,
    }
}

/// Whether a split is half of a FADE pair: what is drawn between the two composites as one layer, scaled by an
/// opacity, with plain source-over. A faded group's `BlendBegin` / `StackEnd` at mode 0
/// ([`compute_region_fade_splits`]) and a paint-stack group's `GroupBegin` / `StackEnd` at
/// `STACK_FADE_MODE` (`compute_group_splits`). A begin is shared with the blend pairs, whose end carries a blend
/// mode: a list is all fades when every one of its splits passes this.
pub(crate) fn is_fade_split(split: &BackdropSplit) -> bool {
    matches!(split.op, BackdropOp::BlendBegin | BackdropOp::GroupBegin) || is_fade_end(split)
}

/// Whether a split ends a fade pair ([`is_fade_split`]). The end is what tells a fade from a blend: a list
/// HOLDS a fade when one of its splits passes this.
pub(crate) fn is_fade_end(split: &BackdropSplit) -> bool {
    matches!(split.op, BackdropOp::StackEnd { mode, .. } if mode == 0 || mode == crate::seams::STACK_FADE_MODE)
}

/// Re-aim the fade pairs among a capture's `items` at the capture. A fade pair ([`is_fade_split`]) composites
/// what is drawn between it as one layer over what its target already holds; it reads nothing else and has no
/// blend mode chosen for a particular target, so it can run in any target. A backdrop effect (a blur, glass, an
/// adjustment) reads the page behind its node and cannot; a blend pair's formula is chosen for the page target. A pair is planned over the
/// page target: its region is in that target's pixels, cut to that target, and empty when nothing of it lands
/// there ([`split_pair_in`]). Diverted into a layer capture (a filtered group's, `frame.rs`
/// `divert_subtree_filter`), the same items draw into the capture's own target, whose pixels are the capture
/// `region_canvas` at `width_px × height_px`. Each pair's region becomes the canvas bounds of what is drawn
/// between it, in those pixels (two pixels of margin, as the fade's own), so a faded group or
/// a translucent fill-and-stroke composites as one layer inside the capture exactly as it does on the page,
/// on the page or off it. Every split among `items` must be a fade split; pairs nest. One pass: each
/// open pair gathers the bounds of what it holds and hands them to the pair around it.
pub(crate) fn retarget_isolation_splits(
    items: &mut [crate::seams::DrawItem],
    region_canvas: [f64; 4],
    width_px: u32,
    height_px: u32,
) {
    use crate::seams::DrawItem;
    let [rx, ry, rw, rh] = region_canvas;
    let (zx, zy) = (width_px as f64 / rw.max(1e-9), height_px as f64 / rh.max(1e-9));
    let union = |a: Option<Rect>, b: Option<Rect>| match (a, b) {
        (Some(a), Some(b)) => {
            let (x0, y0) = (a.x.min(b.x), a.y.min(b.y));
            let (x1, y1) = ((a.x + a.w).max(b.x + b.w), (a.y + a.h).max(b.y + b.h));
            Some(Rect { x: x0, y: y0, w: x1 - x0, h: y1 - y0 })
        }
        (a, b) => a.or(b),
    };
    // The open pairs, outermost first: each begin's index and the canvas bounds drawn inside it so far.
    let mut open: Vec<(usize, Option<Rect>)> = Vec::new();
    for index in 0..items.len() {
        let closes = match &items[index] {
            DrawItem::BackdropSplit(split) if matches!(split.op, BackdropOp::BlendBegin | BackdropOp::GroupBegin) => {
                open.push((index, None));
                continue;
            }
            DrawItem::BackdropSplit(_) => true,
            _ => false,
        };
        if !closes {
            if let Some((_, bounds)) = open.last_mut() {
                let drawn = crate::layer_blur::shadow::all_items_canvas_bounds(std::slice::from_ref(&items[index]));
                *bounds = union(*bounds, drawn);
            }
            continue;
        }
        let Some((begin, bounds)) = open.pop() else {
            continue;
        };
        if let Some((_, outer)) = open.last_mut() {
            *outer = union(*outer, bounds);
        }
        // Nothing measurable between the pair: an empty region, which the executor skips.
        let rect = bounds
            .and_then(|b| {
                let x0 = ((b.x - rx) * zx - 2.0).floor().max(0.0);
                let y0 = ((b.y - ry) * zy - 2.0).floor().max(0.0);
                let x1 = ((b.x + b.w - rx) * zx + 2.0).ceil().min(width_px as f64);
                let y1 = ((b.y + b.h - ry) * zy + 2.0).ceil().min(height_px as f64);
                (x1 > x0 && y1 > y0).then_some([x0, y0, x1, y1])
            })
            .unwrap_or([0.0; 4]);
        for at in [begin, index] {
            if let DrawItem::BackdropSplit(split) = &mut items[at] {
                **split = isolation_split(&split.node_id, split.op.clone(), rect);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Exec
// ---------------------------------------------------------------------------

/// Engine-owned (hand-authored) WGSL: no TS renderer counterpart to extract from.
const BLEND_WGSL: &str = r#"
struct BlendUniforms {
  // Region origin in target px, blend mode, the source's opacity (a paint-stack group's; 1 otherwise).
  origin: vec2<f32>,
  mode: f32,
  opacity: f32,
};
@group(0) @binding(0) var<uniform> u: BlendUniforms;
@group(0) @binding(1) var backdropTex: texture_2d<f32>;
@group(0) @binding(2) var sourceTex: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  // One oversized triangle; the scissor rect limits it to the region.
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_clear() -> @location(0) vec4<f32> {
  return vec4<f32>(0.0);
}

// Separable blend functions B(Cb, Cs), Compositing and Blending Level 1 §10.1 (per channel, unpremultiplied).
fn hard_light(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
  // Multiply below mid-grey source (by 2·Cs), screen above (by 2·Cs − 1).
  let lo = 2.0 * cb * cs;
  let hi = vec3<f32>(1.0) - 2.0 * (vec3<f32>(1.0) - cb) * (vec3<f32>(1.0) - cs);
  return select(hi, lo, cs <= vec3<f32>(0.5));
}

fn color_dodge_1(cb: f32, cs: f32) -> f32 {
  if (cb <= 0.0) { return 0.0; }
  if (cs >= 1.0) { return 1.0; }
  return min(1.0, cb / (1.0 - cs));
}

fn color_burn_1(cb: f32, cs: f32) -> f32 {
  if (cb >= 1.0) { return 1.0; }
  if (cs <= 0.0) { return 0.0; }
  return 1.0 - min(1.0, (1.0 - cb) / cs);
}

fn soft_light(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
  let d = select(sqrt(cb), ((16.0 * cb - 12.0) * cb + 4.0) * cb, cb <= vec3<f32>(0.25));
  let lo = cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb);
  let hi = cb + (2.0 * cs - 1.0) * (d - cb);
  return select(hi, lo, cs <= vec3<f32>(0.5));
}

// Non-separable helpers, §10.2 (Rec. 601 luma weights, as the spec).
fn lum(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.3, 0.59, 0.11));
}

fn clip_color(c_in: vec3<f32>) -> vec3<f32> {
  var c = c_in;
  let l = lum(c);
  let n = min(c.r, min(c.g, c.b));
  let x = max(c.r, max(c.g, c.b));
  if (n < 0.0) {
    c = l + (c - l) * l / max(l - n, 1e-6);
  }
  if (x > 1.0) {
    c = l + (c - l) * (1.0 - l) / max(x - l, 1e-6);
  }
  return c;
}

fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
  return clip_color(c + (l - lum(c)));
}

fn sat(c: vec3<f32>) -> f32 {
  return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
}

fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
  // The max channel becomes s, the min 0, the middle scales in between; a grey stays black.
  let n = min(c.r, min(c.g, c.b));
  let range = max(c.r, max(c.g, c.b)) - n;
  if (range <= 0.0) {
    return vec3<f32>(0.0);
  }
  return (c - n) * s / range;
}

// The naive process conversion (K = 1 − max(R, G, B), C, M, Y under it; greys and black are K alone) and back. The
// round trip is exact, so a colour whose every ink is above 0 % overprints as itself.
fn to_cmyk(c: vec3<f32>) -> vec4<f32> {
  let k = 1.0 - max(c.r, max(c.g, c.b));
  if (k >= 1.0) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  return vec4<f32>((vec3<f32>(1.0) - c - vec3<f32>(k)) / (1.0 - k), k);
}

fn from_cmyk(v: vec4<f32>) -> vec3<f32> {
  return (vec3<f32>(1.0) - v.xyz) * (1.0 - v.w);
}

// Mode ids: `BlendMode::split_id` / `split_id_over` (seams.rs).
fn blend(mode: u32, cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
  switch (mode) {
    // multiply: split only over a transparent target, where its fixed-function state is not exact.
    case 1u: { return cb * cs; }
    // screen: split only inside a paint-stack group (E3.1 P6), whose backdrop may be transparent.
    case 2u: { return cb + cs - cb * cs; }
    // overlay(Cb, Cs) = hard-light(Cs, Cb): multiply below mid-grey backdrop, screen above.
    case 3u: { return hard_light(cs, cb); }
    case 4u: { return min(cb, cs); }
    case 5u: { return max(cb, cs); }
    case 6u: {
      return vec3<f32>(color_dodge_1(cb.r, cs.r), color_dodge_1(cb.g, cs.g), color_dodge_1(cb.b, cs.b));
    }
    case 7u: {
      return vec3<f32>(color_burn_1(cb.r, cs.r), color_burn_1(cb.g, cs.g), color_burn_1(cb.b, cs.b));
    }
    case 8u: { return hard_light(cb, cs); }
    case 9u: { return soft_light(cb, cs); }
    case 10u: { return abs(cb - cs); }
    case 11u: { return cb + cs - 2.0 * cb * cs; }
    case 12u: { return set_lum(set_sat(cs, sat(cb)), lum(cb)); }
    case 13u: { return set_lum(set_sat(cb, sat(cs)), lum(cb)); }
    case 14u: { return set_lum(cs, lum(cb)); }
    case 15u: { return set_lum(cb, lum(cs)); }
    // 16 is the paint-stack fade (`STACK_FADE_MODE`), handled in `fs_composite`.
    // linear burn / linear dodge (Figma LINEAR_BURN / LINEAR_DODGE; CSS plus-darker / plus-lighter).
    case 17u: { return max(cb + cs - vec3<f32>(1.0), vec3<f32>(0.0)); }
    case 18u: { return min(cb + cs, vec3<f32>(1.0)); }
    // overprint preview (`BlendMode::Overprint`): each ink the source leaves at 0 % is the backdrop's (OPM 1).
    case 19u: {
      let s = to_cmyk(cs);
      return from_cmyk(select(s, to_cmyk(cb), s <= vec4<f32>(0.5 / 255.0)));
    }
    default: { return cs; }
  }
}

@fragment
fn fs_composite(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let texel = vec2<i32>(floor(pos.xy - u.origin));
  let b = textureLoad(backdropTex, texel, 0);
  let current = textureLoad(sourceTex, texel, 0);
  // A non-isolated paint-stack group (GroupBegin … StackEnd): the layers drew onto the backdrop; fade them.
  if (u32(u.mode + 0.5) == 16u) {
    return b + (current - b) * u.opacity;
  }
  let s = current * u.opacity;
  let cb = select(vec3<f32>(0.0), b.rgb / b.a, b.a > 0.0);
  let cs = select(vec3<f32>(0.0), s.rgb / s.a, s.a > 0.0);
  let mixed = blend(u32(u.mode + 0.5), clamp(cb, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(cs, vec3<f32>(0.0), vec3<f32>(1.0)));
  let co = s.rgb * (1.0 - b.a) + b.rgb * (1.0 - s.a) + s.a * b.a * mixed;
  return vec4<f32>(clamp(co, vec3<f32>(0.0), vec3<f32>(1.0)), s.a + b.a * (1.0 - s.a));
}
"#;

pub struct BlendExec {
    clear: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    frame_format: wgpu::TextureFormat,
    /// The backdrops saved by `BlendBegin`s, each consumed by its matching end. A stack: a paint-stack group
    /// (E3.1 P6) nests its blended layers' splits inside its own.
    saved: RefCell<Vec<(wgpu::Texture, [u32; 4])>>,
}

impl BlendExec {
    /// `sample_count` is the frame's: the clear and composite passes draw into its multisample view (every sample
    /// of a region pixel gets the pixel's value), resolving into the frame target.
    pub fn new(
        device: &wgpu::Device,
        frame_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> BlendExec {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blend-split"),
            source: wgpu::ShaderSource::Wgsl(BLEND_WGSL.into()),
        });
        let frag = wgpu::ShaderStages::FRAGMENT;
        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: frag,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blend Split BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: frag,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture_entry(1),
                texture_entry(2),
            ],
        });
        let empty_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blend Split Clear Layout"),
            ..Default::default()
        });
        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blend Split Composite Layout"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let pipeline = |label: &str, layout: &wgpu::PipelineLayout, fs: &str| {
            moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    // Replace: both passes write the final value for every region pixel.
                    targets: &[Some(wgpu::ColorTargetState {
                        format: frame_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: sample_count.max(1),
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            })
        };
        BlendExec {
            clear: pipeline("Blend Split Clear", &empty_layout, "fs_clear"),
            composite: pipeline("Blend Split Composite", &composite_layout, "fs_composite"),
            bgl,
            frame_format,
            saved: RefCell::new(Vec::new()),
        }
    }

    /// Drop any saved backdrop (a frame that failed between a begin and its end); called before a frame's splits.
    pub fn reset(&self) {
        self.saved.borrow_mut().clear();
    }

    fn copy_out(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
        [x, y, w, h]: [u32; 4],
        label: &str,
    ) -> wgpu::Texture {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.frame_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: target,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        tex
    }

    /// A pass over the region: into `msaa_view` resolving into `target_view` when the frame is multisampled (so the
    /// next segment, which loads the multisample view, keeps what this pass wrote), else into `target_view`.
    fn region_pass<'e>(
        encoder: &'e mut wgpu::CommandEncoder,
        target_view: &'e wgpu::TextureView,
        msaa_view: Option<&'e wgpu::TextureView>,
        [x, y, w, h]: [u32; 4],
    ) -> wgpu::RenderPass<'e> {
        let (view, resolve_target) = match msaa_view {
            Some(m) => (m, Some(target_view)),
            None => (target_view, None),
        };
        let profile_ts = crate::gpu_profile::pass_timestamps();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blend split"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        // The viewport too, not the scissor alone: llvmpipe's multisample scissor keeps only half the samples of
        // the rect's first row and column, so a composite clipped by the scissor alone left those pixels half
        // cleared (the edge of every group at an opacity, partly transparent in an opaque export).
        pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
        pass.set_scissor_rect(x, y, w, h);
        pass
    }

    /// Run one `BlendBegin` / `BlendEnd` split (the caller has ended the main pass and resumes it after).
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
        target_view: &wgpu::TextureView,
        msaa_view: Option<&wgpu::TextureView>,
        target_size: (u32, u32),
        split: &BackdropSplit,
    ) -> Result<()> {
        let [x, y, w, h] = split.region_px;
        // Plans are built for this target, but clamp anyway: a copy past the edge is a validation error.
        let w = w.min(target_size.0.saturating_sub(x));
        let h = h.min(target_size.1.saturating_sub(y));
        if w == 0 || h == 0 {
            // Nothing on the target (plans clip regions to it, so a begin and its end skip together).
            return Ok(());
        }
        let region = [x, y, w, h];
        match split.op {
            BackdropOp::BlendBegin => {
                let backdrop =
                    self.copy_out(device, encoder, target, region, "Blend Split Backdrop");
                self.saved.borrow_mut().push((backdrop, region));
                let mut pass = Self::region_pass(encoder, target_view, msaa_view, region);
                pass.set_pipeline(&self.clear);
                pass.draw(0..3, 0..1);
            }
            BackdropOp::GroupBegin => {
                // Saved, not cleared: the group's layers draw onto the backdrop.
                let backdrop =
                    self.copy_out(device, encoder, target, region, "Stack Group Backdrop");
                self.saved.borrow_mut().push((backdrop, region));
            }
            BackdropOp::BlendEnd { .. } | BackdropOp::StackEnd { .. } => {
                let (mode, opacity) = match split.op {
                    BackdropOp::StackEnd { mode, opacity } => (mode, opacity),
                    BackdropOp::BlendEnd { mode } => (mode, 1.0),
                    _ => unreachable!(),
                };
                let Some((backdrop, saved_region)) = self.saved.borrow_mut().pop() else {
                    bail!("blend split end for '{}' without a begin", split.node_id);
                };
                if saved_region != region {
                    bail!("blend split regions differ for '{}'", split.node_id);
                }
                let source = self.copy_out(device, encoder, target, region, "Blend Split Source");
                let uniforms: [f32; 4] = [x as f32, y as f32, mode as f32, opacity];
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Blend Split Uniforms"),
                    size: 16,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&uniforms));
                let backdrop_view = backdrop.create_view(&Default::default());
                let source_view = source.create_view(&Default::default());
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Blend Split BG"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&backdrop_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&source_view),
                        },
                    ],
                });
                let mut pass = Self::region_pass(encoder, target_view, msaa_view, region);
                pass.set_pipeline(&self.composite);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            _ => bail!("not a blend split"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    /// CPU reference of the shader's formula, pinned against values computed by hand from the W3C spec.
    fn overlay_ref(b: [f64; 4], s: [f64; 4]) -> [f64; 4] {
        let (ab, as_) = (b[3], s[3]);
        let mut out = [0.0; 4];
        for i in 0..3 {
            let cb = if ab > 0.0 { b[i] / ab } else { 0.0 };
            let cs = if as_ > 0.0 { s[i] / as_ } else { 0.0 };
            let mixed = if cb <= 0.5 {
                2.0 * cb * cs
            } else {
                1.0 - 2.0 * (1.0 - cb) * (1.0 - cs)
            };
            out[i] = s[i] * (1.0 - ab) + b[i] * (1.0 - as_) + as_ * ab * mixed;
        }
        out[3] = as_ + ab * (1.0 - as_);
        out
    }

    #[test]
    fn overlay_reference_matches_spec_values() {
        // Opaque: Cb 0.25 (dark → multiply 2·Cb·Cs), Cb 0.75 (light → screen), Cs 0.5 is identity.
        let o = overlay_ref([0.25, 0.75, 0.5, 1.0], [0.8, 0.8, 0.5, 1.0]);
        let expect = [0.4, 1.0 - 2.0 * 0.25 * 0.2, 0.5, 1.0];
        for i in 0..4 {
            assert!((o[i] - expect[i]).abs() < 1e-12, "{o:?} vs {expect:?}");
        }
        // Half-transparent source over opaque backdrop: αs·B + (1 − αs)·Cb.
        let o = overlay_ref([0.25, 0.25, 0.25, 1.0], [0.4, 0.4, 0.4, 0.5]);
        let mixed = 2.0 * 0.25 * 0.8;
        assert!((o[0] - (0.5 * mixed + 0.5 * 0.25)).abs() < 1e-12);
        assert!((o[3] - 1.0).abs() < 1e-12);
        // Transparent backdrop: the source passes through unchanged.
        assert_eq!(
            overlay_ref([0.0; 4], [0.3, 0.2, 0.1, 0.6]),
            [0.3, 0.2, 0.1, 0.6]
        );
    }

    /// The shader's `fs_composite` for color-dodge (6) / color-burn (7) / linear-burn (17) / linear-dodge (18), in
    /// f32 with the same branch and clamp order: unpremultiply, clamp Cb / Cs to [0, 1], B per channel, premultiplied source-over, clamp.
    fn dodge_burn_shader(mode: u32, b: [f32; 4], current: [f32; 4], opacity: f32) -> [f32; 4] {
        let s = current.map(|v| v * opacity);
        let mut out = [0.0f32; 4];
        for i in 0..3 {
            let cb = if b[3] > 0.0 { b[i] / b[3] } else { 0.0 }.clamp(0.0, 1.0);
            let cs = if s[3] > 0.0 { s[i] / s[3] } else { 0.0 }.clamp(0.0, 1.0);
            let mixed = match mode {
                6 if cb <= 0.0 => 0.0,
                6 if cs >= 1.0 => 1.0,
                6 => (cb / (1.0 - cs)).min(1.0),
                7 if cb >= 1.0 => 1.0,
                7 if cs <= 0.0 => 0.0,
                7 => 1.0 - ((1.0 - cb) / cs).min(1.0),
                17 => (cb + cs - 1.0).max(0.0),
                18 => (cb + cs).min(1.0),
                _ => unreachable!(),
            };
            let co = s[i] * (1.0 - b[3]) + b[i] * (1.0 - s[3]) + s[3] * b[3] * mixed;
            out[i] = co.clamp(0.0, 1.0);
        }
        out[3] = s[3] + b[3] * (1.0 - s[3]);
        out
    }

    /// W3C §10.1 dodge / burn (and Figma's linear burn / dodge) and premultiplied source-over in f64, written from
    /// the spec.
    fn dodge_burn_spec(mode: u32, b: [f64; 4], s: [f64; 4]) -> [f64; 4] {
        let mut out = [0.0; 4];
        for i in 0..3 {
            let cb = if b[3] > 0.0 { b[i] / b[3] } else { 0.0 };
            let cs = if s[3] > 0.0 { s[i] / s[3] } else { 0.0 };
            let mixed = match mode {
                6 if cb == 0.0 => 0.0,
                6 if cs == 1.0 => 1.0,
                6 => (cb / (1.0 - cs)).min(1.0),
                7 if cb == 1.0 => 1.0,
                7 if cs == 0.0 => 0.0,
                7 => 1.0 - ((1.0 - cb) / cs).min(1.0),
                17 => (cb + cs - 1.0).max(0.0),
                18 => (cb + cs).min(1.0),
                _ => unreachable!(),
            };
            out[i] = s[i] * (1.0 - b[3]) + b[i] * (1.0 - s[3]) + s[3] * b[3] * mixed;
        }
        out[3] = s[3] + b[3] * (1.0 - s[3]);
        out
    }

    #[test]
    fn dodge_and_burn_edge_cases_match_the_spec() {
        // Backdrop 0 / 1 / partial, source 0 / 1 / partial (straight colour), source alpha 1 / ½ / 0, and an opaque,
        // half-transparent and empty backdrop. Every combination, both modes; the f32 shader path is within 1e-6.
        let levels = [0.0, 1.0, 0.4, 0.6, 1.0 - 1e-6, 1e-6];
        for mode in [6, 7, 17, 18] {
            for &cb in &levels {
                for &cs in &levels {
                    for ab in [1.0, 0.5, 0.0] {
                        for as_ in [1.0, 0.5, 0.0] {
                            // The spec side takes the same f32 inputs: near 1, f32(1 − 1e-6) is 1 − 1.0133e-6.
                            let b = [cb * ab, cb * ab, cb * ab, ab].map(|v| v as f32);
                            let current = [cs, cs, cs, 1.0].map(|v| v as f32);
                            let got = dodge_burn_shader(mode, b, current, as_ as f32);
                            let s = current.map(|v| v as f64 * as_);
                            let want = dodge_burn_spec(mode, b.map(|v| v as f64), s);
                            for i in 0..4 {
                                assert!(
                                    got[i].is_finite() && (got[i] as f64 - want[i]).abs() < 1e-6,
                                    "mode {mode} Cb {cb} Cs {cs} αb {ab} αs {as_}: {got:?} vs {want:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
        // The spec's special cases, pinned: a black backdrop never dodges, a white source always does; a white
        // backdrop never burns, a black source always does (including 0/0 and 1/0 in B's quotient).
        let opaque = |c: f32| [c, c, c, 1.0];
        assert_eq!(dodge_burn_shader(6, opaque(0.0), opaque(1.0), 1.0)[0], 0.0);
        assert_eq!(dodge_burn_shader(6, opaque(0.2), opaque(1.0), 1.0)[0], 1.0);
        assert_eq!(dodge_burn_shader(7, opaque(1.0), opaque(0.0), 1.0)[0], 1.0);
        assert_eq!(dodge_burn_shader(7, opaque(0.8), opaque(0.0), 1.0)[0], 0.0);
    }

    #[test]
    fn linear_burn_and_dodge_match_the_spec() {
        let opaque = |c: f32| [c, c, c, 1.0];
        let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
        // Opaque over opaque: B itself. Burn floors Cb + Cs − 1 at 0, dodge caps Cb + Cs at 1.
        assert!(close(linear(17, 0.8, 0.5, 1.0, 1.0), 0.3));
        assert!(close(linear(17, 0.3, 0.5, 1.0, 1.0), 0.0));
        assert!(close(linear(18, 0.3, 0.5, 1.0, 1.0), 0.8));
        assert!(close(linear(18, 0.6, 0.5, 1.0, 1.0), 1.0));
        // White is burn's identity and black dodge's; black burns to black and white dodges to white.
        assert!(close(dodge_burn_shader(17, opaque(0.37), opaque(1.0), 1.0)[0], 0.37));
        assert!(close(dodge_burn_shader(18, opaque(0.37), opaque(0.0), 1.0)[0], 0.37));
        assert_eq!(dodge_burn_shader(17, opaque(0.37), opaque(0.0), 1.0)[0], 0.0);
        assert_eq!(dodge_burn_shader(18, opaque(0.37), opaque(1.0), 1.0)[0], 1.0);
        // Half-transparent source: (1 − αs)·Cb + αs·B. Dodge's clamp is on B, not on the sum (CSS plus-lighter
        // would give min(1, Cb + αs·Cs) = 1 here).
        assert!(close(linear(18, 0.8, 0.8, 0.5, 1.0), 0.5 * 0.8 + 0.5 * 1.0));
        assert!(close(linear(17, 0.8, 0.5, 0.5, 1.0), 0.5 * 0.8 + 0.5 * 0.3));
        // Over nothing the source passes through unchanged.
        let clear = [0.0; 4];
        assert_eq!(dodge_burn_shader(17, clear, [0.3, 0.2, 0.1, 1.0], 0.6), [0.3 * 0.6, 0.2 * 0.6, 0.1 * 0.6, 0.6]);
        assert_eq!(dodge_burn_shader(18, clear, [0.3, 0.2, 0.1, 1.0], 0.6), [0.3 * 0.6, 0.2 * 0.6, 0.1 * 0.6, 0.6]);
    }

    /// One channel of `dodge_burn_shader` for straight colours Cb / Cs at source α `as_` and backdrop α `ab`.
    fn linear(mode: u32, cb: f32, cs: f32, as_: f32, ab: f32) -> f32 {
        dodge_burn_shader(mode, [cb * ab, cb * ab, cb * ab, ab], [cs, cs, cs, 1.0], as_)[0]
    }

    /// blend_modes' `gpu_separable_modes_match_the_w3c_formula`, color-dodge α=½ of #1ad9f0 over #e62614, green:
    /// Cb = 38/255 and Cs = 217/255 put B exactly on dodge's knee (Cb = 1 − Cs, B = 1), where dB/dCs = Cb/(1 − Cs)²
    /// ≈ 6.7. The source reaches the shader premultiplied in the Rgba8Unorm frame: G = 108.5 and α = 127.5, two
    /// exact rounding ties. llvmpipe rounds ties to even (108, 128; read back from the shader); Metal passes the
    /// test, consistent with rounding G up (109 puts Cs past the knee and B clamps to 1). The shader is exact for
    /// either input: the 6-level spread is the 8-bit tie, amplified by the slope.
    #[test]
    fn dodge_knee_amplifies_the_8_bit_source_tie() {
        let back = [230.0 / 255.0, 38.0 / 255.0, 20.0 / 255.0, 1.0];
        let g = |stored_g: f32, stored_a: f32| {
            let current = [
                13.0 / 255.0,
                stored_g / 255.0,
                120.0 / 255.0,
                stored_a / 255.0,
            ];
            (dodge_burn_shader(6, back, current, 1.0)[1] * 255.0).round() as u8
        };
        // Ties to even (llvmpipe): 141. Ties up on G: 147. The unquantized W3C value is 146.5.
        assert_eq!(g(108.0, 128.0), 141);
        assert_eq!(g(109.0, 128.0), 147);
        let exact = dodge_burn_spec(
            6,
            [230.0 / 255.0, 38.0 / 255.0, 20.0 / 255.0, 1.0],
            [13.0 / 255.0, 108.5 / 255.0, 120.0 / 255.0, 0.5],
        );
        assert!((exact[1] * 255.0 - 146.5).abs() < 1e-9, "{exact:?}");
    }
}
