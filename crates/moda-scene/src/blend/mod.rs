//! Blends (E8.12, Illustrator's Object ▸ Blend): a live blend node draws the steps between two or more objects.
//!
//! A blend node (`type: "blend"`) references its objects by id in chain order (`sourceIds`) and, after Replace
//! Spine, the path they follow (`spineId`). The objects stay ordinary nodes that draw themselves; the blend node
//! draws only the steps, at its own place in the stacking order. Its options are Illustrator's Blend Options plus
//! the 30.7 Blend panel's easing: `spacing` (`smooth` / `steps` / `distance`), `steps`, `distance`, `orientation`
//! (`page` / `path`), `easeIn`, `easeOut` and `reverseStack` (Reverse Front to Back).
//!
//! - [`geometry`]: outlines as cubic contours, point correspondence, interpolation.
//! - [`spine`]: arc-length placement along the spine.
//! - [`paint_lerp`]: fills, strokes (paint stacks, gradients) and opacity.
//! - [`steps`]: spacing, easing, orientation and stacking — [`steps::blend_steps`], the one implementation. The
//!   engine draws its output ([`expand_blend_node`]); Object ▸ Blend ▸ Expand and the SVG writer read the same
//!   output through [`blend_steps_json`] (`moda-wasm` `blend_steps`).
//!
//! Objects are the shapes a path can stand for: paths, lines, rectangles, ellipses, polygons and stars. Anything
//! else in `sourceIds` is left out of the chain.

pub mod color;
pub mod geometry;
pub mod paint_lerp;
pub mod points;
pub mod spine;
pub mod steps;

#[cfg(test)]
mod tests;

pub use points::{flatten_path_to_polyline, BlendPathPoint};
pub use steps::{blend_steps, BlendOptions, BlendSource, BlendStep, Orientation, Spacing};

use crate::scene::Scene;
use crate::types::{JsonMap, RawNode};
use geometry::Shape;
use crate::value::{json, Value};

/// Node types a blend can blend (and a spine can be).
pub const BLENDABLE_TYPES: &[&str] = &["path", "line", "rectangle", "ellipse", "polygon", "star"];

/// `Scene.getPathRenderAbsolutePoints` — `computePathRenderAbsolutePoints`
/// (`core/scene/scene.transforms.ts`): the path as DRAWN, under its canvas affine.
///
/// `path` has an editable uniform `cornerRadius`, so the drawn outline is no longer the authored `points`; a blend
/// that interpolated the sharp corners would hand back steps that touch neither end shape.
pub fn path_absolute_points(scene: &Scene, node: &RawNode) -> Option<Vec<BlendPathPoint>> {
    if node.node_type() != "path" {
        return None;
    }
    let raw = node.points()?;
    let parsed: Vec<BlendPathPoint> = raw.iter().filter_map(BlendPathPoint::from_ref).collect();
    if parsed.len() != raw.len() {
        return None; // isPathPointArray guard
    }
    let t = scene.node_canvas_affine(node.id())?;
    // `path.cornerRadius || 0` (JS falsy: 0/NaN/missing → 0) and the anchors' own corners, matching
    // `moda_engine::tess::path_cmds_for_node`.
    let corners = crate::path_geom::round_corners::PathCorners::read(node);
    if !corners.rounds_anything() {
        return Some(parsed.iter().map(|p| p.apply_affine(t)).collect());
    }
    let rounded = crate::path_geom::round_corners::round_path_corners_with(
        &parsed,
        node.bool("closed").unwrap_or(false),
        &corners,
    );
    Some(rounded.iter().map(|p| p.apply_affine(t)).collect())
}

/// Canvas-space outline points for any blendable node (`shapeNodeToCanvasPathPoints` in
/// `core/utils/shapeToPath.ts`). The shape → path conversions live engine-side (`tess::shape_paths`,
/// `line_to_path`), so the engine injects this.
pub type ShapeToAbsPoints<'a> = dyn Fn(&RawNode) -> Option<Vec<BlendPathPoint>> + 'a;

/// Whether a node's outline is closed: a path's `closed`, never a line, always the box shapes.
pub fn node_is_closed(node: &RawNode) -> bool {
    match node.node_type() {
        "path" => node.bool("closed").unwrap_or(false),
        "line" => false,
        _ => true,
    }
}

/// The blend's object ids in chain order.
pub fn source_ids(node: &RawNode) -> Vec<&str> {
    node.array("sourceIds")
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Every node a blend node draws from: its objects and its spine. An edit to any of them changes the blend.
pub fn dependency_ids(node: &RawNode) -> Vec<&str> {
    let mut ids = source_ids(node);
    if let Some(spine) = node.str("spineId").filter(|s| !s.is_empty()) {
        ids.push(spine);
    }
    ids
}

/// Where a node lives: `Some(Some(page))` on a page, `Some(None)` on the canvas (a canvas root, or inside a
/// frame on a page-less Design document), `None` for a page id that no longer resolves.
fn space_of(scene: &Scene, id: &str) -> Option<Option<usize>> {
    match scene.page_of(id) {
        None => Some(None),
        Some(pid) => scene.doc.pages.iter().position(|p| p.id() == pid).map(Some),
    }
}

fn node_shape(node: &RawNode, shape_to_abs: &ShapeToAbsPoints) -> Option<Shape> {
    if !BLENDABLE_TYPES.contains(&node.node_type()) {
        return None;
    }
    let shape = Shape::from_points(&shape_to_abs(node)?, node_is_closed(node));
    (!shape.is_empty()).then_some(shape)
}

/// What a blend node draws.
pub enum BlendExpansion {
    /// The steps as synthetic PathNodes, back to front (ids `{blend}-s{i}`), points relative to the page, or in
    /// canvas space when the objects are on the canvas (`page_index: None`).
    Paths {
        page_index: Option<usize>,
        nodes: Vec<RawNode>,
    },
    /// Nothing to draw, and why.
    Skip(String),
}

/// The blend node a step id (`{blend}-s{i}`, [`expand_blend_node`]) belongs to, when `id` is one and not a node
/// of its own.
pub fn step_owner<'a>(scene: &Scene, id: &'a str) -> Option<&'a str> {
    if scene.effective_node(id).is_some() {
        return None;
    }
    let (owner, i) = id.rsplit_once("-s")?;
    (!i.is_empty() && i.bytes().all(|b| b.is_ascii_digit()))
        .then_some(owner)
        .filter(|o| scene.effective_node(o).is_some_and(|n| n.node_type() == "blend"))
}

/// A blend node's steps, from the scene. The objects (and the spine) must be on one page, or all on the canvas.
pub fn expand_blend_node(
    scene: &Scene,
    node: &RawNode,
    shape_to_abs: &ShapeToAbsPoints,
) -> BlendExpansion {
    let mut page: Option<Option<usize>> = None;
    let mut sources: Vec<BlendSource> = Vec::new();
    for id in source_ids(node) {
        let Some(src) = scene.effective_node(id) else {
            continue;
        };
        let Some(pi) = space_of(scene, id) else {
            continue;
        };
        if *page.get_or_insert(pi) != pi {
            return BlendExpansion::Skip("blend: objects on different pages".into());
        }
        if let Some(shape) = node_shape(src, shape_to_abs) {
            sources.push(BlendSource::new(shape, src.fields()));
        }
    }
    let Some(page_index) = page else {
        return BlendExpansion::Skip("blend: no objects".into());
    };
    if sources.len() < 2 {
        return BlendExpansion::Skip("blend: fewer than two blendable objects".into());
    }
    let spine = node
        .str("spineId")
        .and_then(|id| scene.effective_node(id))
        .filter(|s| space_of(scene, s.id()) == Some(page_index))
        .and_then(|s| node_shape(s, shape_to_abs));
    let opts = BlendOptions::from_json(node.fields());
    let origin = page_index.map_or((0.0, 0.0), |pi| scene.page_position(pi));
    // The blend node's own opacity and blend mode apply to every step.
    let opacity = node
        .f64("opacity")
        .filter(|v| v.is_finite())
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    let blend_mode = node.json_field("blendMode").cloned();
    let nodes = blend_steps(&sources, spine.as_ref(), &opts)
        .iter()
        .enumerate()
        .map(|(i, step)| {
            let mut step = step.clone();
            step.opacity *= opacity;
            let mut m = step.to_path_node(origin);
            m.insert("id".into(), json!(format!("{}-s{i}", node.id())));
            if let Some(mode) = &blend_mode {
                m.insert("blendMode".into(), mode.clone());
            }
            RawNode::new(m)
        })
        .collect();
    BlendExpansion::Paths { page_index, nodes }
}

fn error(code: &str, message: &str) -> String {
    json!({"status": "error", "code": code, "message": message}).to_string()
}

/// The `blend_steps` contract (`moda-wasm`, `core/geometry/blendSteps.ts`): the steps for objects given as outlines.
///
/// Request: `{ sources: [{ points, closed, fills?, strokes?, opacity?, fillRule? }], spine?: { points, closed },
/// options: { spacing, steps, distance, orientation, easeIn, easeOut, reverseStack } }`, all in one coordinate
/// space. Reply: `{ status: "ok", steps: [{ pair, t, points, closed, fillRule, opacity?, fills?, strokes? }] }` back
/// to front, or `{ status: "error", code, message }`.
pub fn blend_steps_json(request: &str) -> String {
    let Ok(req) = crate::value::from_str::<Value>(request) else {
        return error("bad_request", "request is not JSON");
    };
    let parse_shape = |v: &Value| -> Option<Shape> {
        let pts: Vec<BlendPathPoint> = v
            .get("points")?
            .as_array()?
            .iter()
            .filter_map(BlendPathPoint::from_json)
            .collect();
        let closed = v.get("closed").and_then(Value::as_bool).unwrap_or(false);
        Some(Shape::from_points(&pts, closed))
    };
    let Some(items) = req.get("sources").and_then(Value::as_array) else {
        return error("bad_request", "sources must be an array");
    };
    let mut sources = Vec::with_capacity(items.len());
    for item in items {
        let (Some(shape), Some(style)) = (parse_shape(item), item.as_object()) else {
            return error("bad_request", "each source needs points");
        };
        sources.push(BlendSource::new(shape, style));
    }
    if sources.len() < 2 {
        return error("too_few_sources", "a blend needs at least two objects");
    }
    let spine = req
        .get("spine")
        .filter(|v| !v.is_null())
        .and_then(parse_shape);
    let empty = JsonMap::new();
    let opts = BlendOptions::from_json(
        req.get("options")
            .and_then(Value::as_object)
            .unwrap_or(&empty),
    );
    let steps: Vec<Value> = blend_steps(&sources, spine.as_ref(), &opts)
        .iter()
        .map(|s| {
            let mut m = s.to_path_node((0.0, 0.0));
            m.remove("type");
            m.insert("pair".into(), json!(s.pair));
            m.insert("t".into(), json!(s.t));
            Value::Object(m)
        })
        .collect();
    json!({"status": "ok", "steps": steps}).to_string()
}
