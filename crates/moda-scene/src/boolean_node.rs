//! Live boolean groups (E7.1, Figma's model): a `boolean` node holds ONE operation and its
//! children, and draws the result of that operation over the children's silhouettes with its
//! own paint. The children keep their paint but are not drawn while inside.
//!
//! This module derives the result geometry with the `path_bool` solver:
//!
//! - Operands are the node's visible children, bottom to top, each in the boolean's LOCAL
//!   space (the child's own transform applied). A path contributes its drawn outline
//!   (`cornerRadius` baked); a nested boolean contributes its own result; rectangles,
//!   ellipses, polygons, stars and rich text come through the [`OperandOutliner`] seam,
//!   which moda-engine installs (the shape converters and the text layout live there).
//!   Anything else (lines, frames, images, tables, …) contributes nothing, as does a hidden
//!   child (Figma leaves hidden layers out of the result).
//! - `subtract` is the bottom child minus the rest; union / intersect / exclude fold bottom
//!   to top (the solver's contract).
//! - The result is CACHED per node, keyed by a fingerprint of the operation and every
//!   descendant's geometry. Paint, names and lock state are not in the fingerprint, so
//!   recolouring a child (or the boolean itself) never re-solves; moving, resizing or
//!   re-shaping any descendant does. The cache outlives a scene: every re-feed builds a
//!   fresh `Scene`, and an unchanged boolean is served from here.
//!
//! [`Scene`] substitutes each boolean node's resolved record with [`result_path_node`]: a
//! `path` carrying the result points and the boolean's own paint, transform and effects,
//! so every draw lane (fills, strokes, shadows, blurs, masks, exports) handles it as a path.

use crate::blend::BlendPathPoint as PathPoint;
use crate::path_bool::{
    boolean_paths, BooleanOp, BooleanOperand, BooleanOutcome, BooleanRequest, Bounds, FillRule,
};
use crate::scene::Scene;
use crate::transform;
use crate::types::{RawNode, Transform2D};
use crate::value::{json, Value};
use std::cell::{Cell, RefCell};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::OnceLock;

/// Node-local outline of a child the scene layer cannot convert itself (rectangle, ellipse,
/// polygon, star, rich text). `None` = contributes nothing.
pub type OperandOutliner = fn(&Scene, &RawNode) -> Option<Vec<PathPoint>>;

static OUTLINER: OnceLock<OperandOutliner> = OnceLock::new();

/// Install the operand seam (moda-engine's `install_scene_seams`). First call wins.
pub fn install_operand_outliner(outliner: OperandOutliner) {
    let _ = OUTLINER.set(outliner);
}

/// The installed fonts' generation (moda-engine's text-host generation): it moves whenever faces are installed.
pub type FontGeneration = fn() -> u64;

static FONT_GENERATION: OnceLock<FontGeneration> = OnceLock::new();

/// Install the font-generation seam (moda-engine's `install_scene_seams`). First call wins. A text operand's
/// outline depends on which faces are installed, so the generation is part of its fingerprint: a result solved
/// before a face arrived (the text left out, or drawn in another face) is never served from the cache after.
pub fn install_font_generation(generation: FontGeneration) {
    let _ = FONT_GENERATION.set(generation);
}

/// Deeper nesting than this contributes nothing (a cycle guard as much as a limit).
const MAX_DEPTH: usize = 32;
/// The cache is cleared when it grows past this many boolean nodes.
const CACHE_CAP: usize = 4096;

/// Keys that never change a boolean's geometry: paint, effects, labels and editor state.
/// Everything else on a descendant is in the fingerprint.
const NON_GEOMETRY_KEYS: &[&str] = &[
    "fills",
    "strokes",
    "textFills",
    "opacity",
    "blendMode",
    "effects",
    "metadata",
    "name",
    "locked",
    "exportSettings",
    "constraints",
    "layoutPositioning",
    "trimPathStart",
    "trimPathEnd",
    "trimPathOffset",
    "trimPathMode",
    "trimPathBy",
];

/// A boolean node's result, in the node's local space.
#[derive(Debug, Clone)]
pub struct BooleanResult {
    /// The result outline (even-odd); empty when nothing is left.
    pub points: Vec<PathPoint>,
    pub bounds: Option<Bounds>,
}

impl BooleanResult {
    fn empty() -> BooleanResult {
        BooleanResult {
            points: Vec::new(),
            bounds: None,
        }
    }
}

struct CacheEntry {
    fingerprint: u64,
    result: Rc<BooleanResult>,
}

thread_local! {
    static CACHE: RefCell<HashMap<String, CacheEntry>> = RefCell::new(HashMap::new());
    static SOLVES: Cell<u64> = const { Cell::new(0) };
}

/// How many times this thread has run the solver for a boolean node (a test and stats probe:
/// a cache hit leaves it unchanged).
pub fn solve_count() -> u64 {
    SOLVES.with(Cell::get)
}

/// Drop every cached result (tests; a host that wants its memory back).
pub fn clear_cache() {
    CACHE.with(|c| c.borrow_mut().clear());
}

fn children_ids(node: &RawNode) -> Vec<String> {
    node.array("children")
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// The record a child is read from: variables resolved, and for a boolean the ORIGINAL node
/// (never its substituted path record, whose points are this module's own output).
fn child_node<'a>(scene: &'a Scene, id: &str) -> Option<&'a RawNode> {
    let canonical = scene.node(id)?;
    if canonical.node_type() == "boolean" {
        return scene.pre_boolean_record(id).or(Some(canonical));
    }
    scene.effective_node(id).or(Some(canonical))
}

/// The node's own placement: it moves the drawn result, never the result in local space.
const OWN_TRANSFORM_KEYS: &[&str] = &[
    "x", "y", "rotation", "skewX", "scale", "scaleX", "scaleY", "pivotX", "pivotY",
];

fn hash_geometry(scene: &Scene, node: &RawNode, depth: usize, groups: bool, h: &mut DefaultHasher) {
    node.node_type().hash(h);
    if node.node_type() == "richtext" {
        FONT_GENERATION.get().map(|g| g()).hash(h);
    }
    for (k, v) in node.fields().iter() {
        if NON_GEOMETRY_KEYS.contains(&k.as_str()) || k == "children" {
            continue;
        }
        if depth == 0 && OWN_TRANSFORM_KEYS.contains(&k.as_str()) {
            continue;
        }
        k.hash(h);
        match node.points().filter(|_| k == "points" && node.is_compact()) {
            Some(points) => crate::value::to_string(&*points.json()).unwrap_or_default().hash(h),
            None => v.to_string().hash(h),
        }
    }
    let descends = node.node_type() == "boolean" || (groups && node.node_type() == "group");
    if descends && depth < MAX_DEPTH {
        for cid in children_ids(node) {
            cid.hash(h);
            match child_node(scene, &cid) {
                // Inside a boolean, groups contribute nothing again (the boolean's own rules).
                Some(child) => hash_geometry(
                    scene,
                    child,
                    depth + 1,
                    groups && node.node_type() == "group",
                    h,
                ),
                None => 0u8.hash(h),
            }
        }
    }
}

/// The fingerprint of `node`'s result: its operation and every descendant's geometry.
pub fn fingerprint(scene: &Scene, node: &RawNode) -> u64 {
    fingerprint_with(scene, node, false)
}

fn fingerprint_with(scene: &Scene, node: &RawNode, groups: bool) -> u64 {
    let mut h = DefaultHasher::new();
    groups.hash(&mut h);
    hash_geometry(scene, node, 0, groups, &mut h);
    h.finish()
}

/// The child's own transform (its parent is the boolean, whose space the operands share).
fn local_affine(node: &RawNode) -> crate::types::Affine {
    let (sx, sy) = node.scale_xy();
    transform::from_trsk(
        Transform2D {
            x: node.f64("x").unwrap_or(0.0),
            y: node.f64("y").unwrap_or(0.0),
            rotation: node.rotation(),
            scale_x: sx,
            scale_y: sy,
        },
        node.skew_x(),
    )
}

/// A path's drawn outline in its own local space (live corners baked, as it renders).
pub fn path_local_points(node: &RawNode) -> Option<Vec<PathPoint>> {
    let raw = node.points()?;
    let parsed: Vec<PathPoint> = raw.iter().filter_map(PathPoint::from_ref).collect();
    if parsed.len() != raw.len() || parsed.len() < 2 {
        return None;
    }
    let corners = crate::path_geom::round_corners::PathCorners::read(node);
    if !corners.rounds_anything() {
        return Some(parsed);
    }
    Some(crate::path_geom::round_corners::round_path_corners_with(
        &parsed,
        node.bool("closed").unwrap_or(false),
        &corners,
    ))
}

/// One operand: the child's outline in the boolean's local space, or `None` (it contributes
/// nothing). `complete` goes false when an outline that should exist could not be made (no
/// seam, fonts not loaded), so the result is not cached and the next load retries.
fn operand(
    scene: &Scene,
    child: &RawNode,
    depth: usize,
    groups: bool,
    complete: &mut bool,
) -> Option<BooleanOperand> {
    if child.truthy("hidden") {
        return None;
    }
    let (local, fill_rule) = match child.node_type() {
        // A group's merged outline (appearance on groups) goes through nested groups: the union of
        // their contents.
        "boolean" | "group" if child.node_type() == "boolean" || groups => {
            // A boolean inside keeps its own rules (its group children contribute nothing).
            let r = result_for(scene, child, depth + 1, child.node_type() == "group", complete);
            if r.points.is_empty() {
                return None;
            }
            (r.points.clone(), FillRule::EvenOdd)
        }
        "path" => {
            let rule = match child.str("fillRule") {
                Some("evenodd") => FillRule::EvenOdd,
                _ => FillRule::NonZero,
            };
            (path_local_points(child)?, rule)
        }
        "rectangle" | "ellipse" | "polygon" | "star" | "richtext" => {
            let Some(outliner) = OUTLINER.get() else {
                *complete = false;
                return None;
            };
            match outliner(scene, child) {
                Some(points) if !points.is_empty() => (points, FillRule::NonZero),
                Some(_) => return None,
                None => {
                    // Text can fail for want of a font that arrives later; shapes cannot.
                    if child.node_type() == "richtext" {
                        *complete = false;
                    }
                    return None;
                }
            }
        }
        _ => return None,
    };
    let affine = local_affine(child);
    Some(BooleanOperand {
        points: local.iter().map(|p| p.apply_affine(affine)).collect(),
        fill_rule,
    })
}

fn solve(
    scene: &Scene,
    node: &RawNode,
    depth: usize,
    groups: bool,
    complete: &mut bool,
) -> BooleanResult {
    if depth > MAX_DEPTH {
        return BooleanResult::empty();
    }
    let op = match node.node_type() {
        "group" => BooleanOp::Union,
        _ => node
            .str("booleanOperation")
            .and_then(BooleanOp::from_token)
            .unwrap_or(BooleanOp::Union),
    };
    let mut operands = Vec::new();
    for cid in children_ids(node) {
        if cid == node.id() {
            continue;
        }
        let Some(child) = child_node(scene, &cid) else {
            continue;
        };
        if let Some(o) = operand(scene, child, depth, groups, complete) {
            operands.push(o);
        }
    }
    if operands.is_empty() {
        return BooleanResult::empty();
    }
    SOLVES.with(|s| s.set(s.get() + 1));
    // One operand: every operation answers the shape itself (as in Figma). The solver wants
    // two, and a union with itself is that shape, normalized like any other result.
    let (op, operands) = if operands.len() == 1 {
        (
            BooleanOp::Union,
            vec![operands[0].clone(), operands[0].clone()],
        )
    } else {
        (op, operands)
    };
    let request = BooleanRequest {
        op,
        operands,
        accuracy: None,
    };
    match boolean_paths(&request) {
        Ok(BooleanOutcome::Shape { points, bounds, .. }) => BooleanResult {
            points,
            bounds: Some(bounds),
        },
        // Nothing left, or the solver refused the input: the node draws nothing.
        Ok(BooleanOutcome::Empty) | Err(_) => BooleanResult::empty(),
    }
}

fn result_for(
    scene: &Scene,
    node: &RawNode,
    depth: usize,
    groups: bool,
    complete: &mut bool,
) -> Rc<BooleanResult> {
    let fp = fingerprint_with(scene, node, groups);
    // A group's merged outline shares the cache under its own key (a group is never a boolean).
    let id = if node.node_type() == "group" {
        format!("{}{GROUP_OUTLINE_KEY_SUFFIX}", node.id())
    } else {
        node.id().to_string()
    };
    if let Some(hit) = CACHE.with(|c| {
        c.borrow()
            .get(&id)
            .filter(|e| e.fingerprint == fp)
            .map(|e| e.result.clone())
    }) {
        return hit;
    }
    let mut own_complete = true;
    let result = Rc::new(solve(scene, node, depth, groups, &mut own_complete));
    if own_complete {
        CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if c.len() >= CACHE_CAP {
                c.clear();
            }
            c.insert(
                id,
                CacheEntry {
                    fingerprint: fp,
                    result: result.clone(),
                },
            );
        });
    } else {
        *complete = false;
    }
    result
}

/// The result of boolean node `node` in its local space, from the cache when no descendant's
/// geometry changed.
pub fn boolean_result(scene: &Scene, node: &RawNode) -> Rc<BooleanResult> {
    let mut complete = true;
    result_for(scene, node, 0, false, &mut complete)
}

/// Whether a text layer is one of boolean `node`'s operands, directly or through nested booleans.
pub fn has_text_operand(scene: &Scene, node: &RawNode) -> bool {
    fn walk(scene: &Scene, node: &RawNode, depth: usize) -> bool {
        depth <= MAX_DEPTH
            && children_ids(node)
                .iter()
                .any(|cid| match child_node(scene, cid) {
                    Some(c) if c.node_type() == "richtext" => true,
                    Some(c) if c.node_type() == "boolean" && cid != node.id() => {
                        walk(scene, c, depth + 1)
                    }
                    _ => false,
                })
    }
    walk(scene, node, 0)
}

/// The cache key suffix of a group's merged outline.
const GROUP_OUTLINE_KEY_SUFFIX: &str = "::merged-outline";

/// A group's MERGED OUTLINE (appearance on groups, E8.2): the union of its visible children's filled
/// geometry, in the group's local space, through nested groups and after each child's own geometry
/// effects (the child's resolved record). Strokes and paint are not shape: they are left out, as in the
/// appearance RFC (Q4). Cached like a boolean's result, keyed on every descendant's geometry.
pub fn group_merged_outline(scene: &Scene, group: &RawNode) -> Rc<BooleanResult> {
    let mut complete = true;
    result_for(scene, group, 0, true, &mut complete)
}

/// The record a boolean node draws as: a closed even-odd `path` with the result points, and
/// the boolean's own transform, paint, effects and state.
pub fn result_path_node(node: &RawNode, result: &BooleanResult) -> RawNode {
    let mut m = node.fields().clone();
    m.insert("type".into(), json!("path"));
    m.remove("children");
    m.remove("booleanOperation");
    m.insert(
        "points".into(),
        Value::Array(result.points.iter().map(PathPoint::to_json).collect()),
    );
    m.insert("closed".into(), json!(true));
    m.insert("fillRule".into(), json!("evenodd"));
    RawNode::new(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    thread_local! {
        /// The test's font generation: 0 = the text's face is not installed yet.
        static FONTS: Cell<u64> = const { Cell::new(0) };
    }

    fn test_font_generation() -> u64 {
        FONTS.with(Cell::get)
    }

    /// The scene-layer tests stand in for the engine's converters with rectangles only, and text as its box once
    /// its face is installed (before that it lays out with no glyphs, as text whose face hasn't arrived does).
    fn test_outliner(_scene: &Scene, node: &RawNode) -> Option<Vec<PathPoint>> {
        if node.node_type() == "richtext" && FONTS.with(Cell::get) == 0 {
            return Some(Vec::new());
        }
        if !matches!(node.node_type(), "rectangle" | "richtext") {
            return None;
        }
        let w = node.f64("width")?;
        let h = node.f64("height")?;
        Some(vec![
            PathPoint::Move { x: 0.0, y: 0.0 },
            PathPoint::Line { x: w, y: 0.0 },
            PathPoint::Line { x: w, y: h },
            PathPoint::Line { x: 0.0, y: h },
        ])
    }

    fn rect(id: &str, x: f64, y: f64, w: f64, h: f64) -> Value {
        json!({"id": id, "type": "rectangle", "x": x, "y": y, "width": w, "height": h,
               "rotation": 0, "scale": 1, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]})
    }

    fn doc(op: &str, extra: Vec<Value>) -> Value {
        let mut nodes = vec![
            json!({"id": "b", "type": "boolean", "x": 10, "y": 20, "rotation": 0, "scaleX": 1, "scaleY": 1,
                   "children": ["r1", "r2"], "booleanOperation": op,
                   "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}]}),
            rect("r1", 0.0, 0.0, 100.0, 100.0),
            rect("r2", 50.0, 50.0, 100.0, 100.0),
        ];
        nodes.extend(extra);
        json!({"nodes": nodes, "pages": [{"id": "p", "width": 500, "height": 500, "nodeIds": ["b"]}]})
    }

    fn load(v: &Value) -> Scene {
        install_operand_outliner(test_outliner);
        install_font_generation(test_font_generation);
        Scene::load(v).unwrap()
    }

    fn area(points: &[PathPoint]) -> f64 {
        // Shoelace over the anchors, per subpath, absolute — enough for straight-edged results.
        let mut total = 0.0;
        let mut sub: Vec<(f64, f64)> = Vec::new();
        let flush = |sub: &mut Vec<(f64, f64)>, total: &mut f64| {
            let n = sub.len();
            let mut a = 0.0;
            for i in 0..n {
                let (x0, y0) = sub[i];
                let (x1, y1) = sub[(i + 1) % n];
                a += x0 * y1 - x1 * y0;
            }
            *total += (a / 2.0).abs();
            sub.clear();
        };
        for p in points {
            let (x, y) = match p {
                PathPoint::Move { x, y } => {
                    if !sub.is_empty() {
                        flush(&mut sub, &mut total);
                    }
                    (*x, *y)
                }
                PathPoint::Line { x, y }
                | PathPoint::Curve { x, y, .. }
                | PathPoint::Quadratic { x, y, .. }
                | PathPoint::Arc { x, y, .. } => (*x, *y),
            };
            sub.push((x, y));
        }
        if !sub.is_empty() {
            flush(&mut sub, &mut total);
        }
        total
    }

    fn drawn(scene: &Scene) -> (Vec<PathPoint>, RawNode) {
        let n = scene.effective_node("b").unwrap().clone();
        let pts = n
            .points()
            .unwrap()
            .iter()
            .filter_map(PathPoint::from_ref)
            .collect();
        (pts, n)
    }

    /// "Aa" minus a bar: the text is a 100 × 40 box (once its face is installed), the bar 120 × 10 across it.
    fn text_minus_bar() -> Value {
        json!({"nodes": [
            {"id": "b", "type": "boolean", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
             "children": ["t", "bar"], "booleanOperation": "subtract",
             "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#111111"}}]},
            {"id": "t", "type": "richtext", "x": 0, "y": 0, "rotation": 0, "scale": 1, "width": 100, "height": 40,
             "htmlContent": "<p>Aa</p>"},
            rect("bar", -10.0, 15.0, 120.0, 10.0),
        ], "pages": [{"id": "p", "width": 500, "height": 500, "nodeIds": ["b"]}]})
    }

    #[test]
    fn text_solved_before_its_face_arrived_is_solved_again_once_it_does() {
        clear_cache();
        FONTS.with(|f| f.set(0));
        let mut scene = load(&text_minus_bar());
        // Without the face the text contributes nothing: the bar alone (what the canvas drew).
        assert!((area(&drawn(&scene).0) - 1_200.0).abs() < 1.0);
        FONTS.with(|f| f.set(1));
        // A scene loaded again from the same document (a full feed) does not take the result from the cache…
        let reloaded = load(&text_minus_bar());
        let a = area(&drawn(&reloaded).0);
        assert!((a - 3_000.0).abs() < 1.0, "the text minus the bar: {a}");
        // …and the resident scene re-solves its text booleans in place, the others keeping their results.
        assert_eq!(scene.retry_text_booleans(), vec!["b".to_string()]);
        assert!((area(&drawn(&scene).0) - 3_000.0).abs() < 1.0);
        let mut plain = load(&doc("union", vec![]));
        assert!(plain.retry_text_booleans().is_empty());
        FONTS.with(|f| f.set(0));
    }

    #[test]
    fn each_operation_draws_its_region() {
        // Two 100×100 squares overlapping in a 50×50 corner.
        for (op, expected) in [
            ("union", 17_500.0),
            ("subtract", 7_500.0),
            ("intersect", 2_500.0),
            ("exclude", 15_000.0),
        ] {
            clear_cache();
            let scene = load(&doc(op, vec![]));
            let (pts, node) = drawn(&scene);
            assert_eq!(node.node_type(), "path", "{op}");
            assert!(
                (area(&pts) - expected).abs() < 1.0,
                "{op}: area {}",
                area(&pts)
            );
        }
    }

    #[test]
    fn the_result_wears_the_boolean_paint_and_transform() {
        clear_cache();
        let scene = load(&doc("union", vec![]));
        let (_, node) = drawn(&scene);
        assert_eq!(node.f64("x"), Some(10.0));
        assert_eq!(node.f64("y"), Some(20.0));
        assert_eq!(node.str("fillRule"), Some("evenodd"));
        assert_eq!(node.json_field("closed"), Some(&json!(true)));
        assert!(node.json_field("children").is_none());
        assert_eq!(
            node.array("fills").unwrap()[0]["paint"]["color"],
            json!("#0000ff")
        );
    }

    #[test]
    fn children_are_not_drawn_but_the_boolean_is() {
        clear_cache();
        let scene = load(&doc("union", vec![]));
        let z = scene.flattened_z_order_for_page("p");
        assert!(z.contains(&"b".to_string()));
        assert!(!z.contains(&"r1".to_string()) && !z.contains(&"r2".to_string()));
    }

    #[test]
    fn nested_booleans_feed_their_result_to_the_parent() {
        clear_cache();
        // outer = union(inner, r3) where inner = r1 ∩ r2 (50×50 at 50,50) and r3 a
        // disjoint 10×10 square: 2500 + 100.
        let v = json!({
            "nodes": [
                {"id": "b", "type": "boolean", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
                 "children": ["inner", "r3"], "booleanOperation": "union"},
                {"id": "inner", "type": "boolean", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
                 "children": ["r1", "r2"], "booleanOperation": "intersect"},
                rect("r1", 0.0, 0.0, 100.0, 100.0),
                rect("r2", 50.0, 50.0, 100.0, 100.0),
                rect("r3", 300.0, 300.0, 10.0, 10.0),
            ],
            "pages": [{"id": "p", "width": 500, "height": 500, "nodeIds": ["b"]}]
        });
        let scene = load(&v);
        let (pts, _) = drawn(&scene);
        assert!((area(&pts) - 2_600.0).abs() < 1.0, "area {}", area(&pts));
    }

    #[test]
    fn a_child_transform_moves_its_operand() {
        clear_cache();
        // Move r2 away: union becomes two disjoint squares (20 000).
        let mut v = doc("union", vec![]);
        v["nodes"][2]["x"] = json!(400);
        let scene = load(&v);
        let (pts, _) = drawn(&scene);
        assert!((area(&pts) - 20_000.0).abs() < 1.0);
    }

    #[test]
    fn hidden_children_are_left_out() {
        clear_cache();
        let mut v = doc("union", vec![]);
        v["nodes"][2]["hidden"] = json!(true);
        let scene = load(&v);
        let (pts, _) = drawn(&scene);
        assert!((area(&pts) - 10_000.0).abs() < 1.0);
    }

    #[test]
    fn nothing_left_draws_nothing() {
        clear_cache();
        let mut v = doc("intersect", vec![]);
        v["nodes"][2]["x"] = json!(400);
        let scene = load(&v);
        assert!(!scene
            .flattened_z_order_for_page("p")
            .contains(&"b".to_string()));
    }

    #[test]
    fn the_cache_recomputes_only_when_a_descendant_geometry_changes() {
        clear_cache();
        let v = doc("subtract", vec![]);
        let before = solve_count();
        let _ = load(&v);
        assert_eq!(solve_count(), before + 1, "first load solves");

        // Same document again (a re-feed): served from the cache.
        let _ = load(&v);
        assert_eq!(solve_count(), before + 1, "unchanged: no solve");

        // Paint on a child, and on the boolean itself: still no solve.
        let mut recolour = v.clone();
        recolour["nodes"][1]["fills"][0]["paint"]["color"] = json!("#00ff00");
        recolour["nodes"][0]["fills"][0]["paint"]["color"] = json!("#000000");
        recolour["nodes"][0]["opacity"] = json!(0.5);
        recolour["nodes"][1]["metadata"] = json!({"name": "renamed"});
        let _ = load(&recolour);
        assert_eq!(solve_count(), before + 1, "paint-only edits: no solve");

        // Moving or rotating the boolean itself changes nothing in its local space.
        let mut moved = v.clone();
        moved["nodes"][0]["x"] = json!(200);
        moved["nodes"][0]["rotation"] = json!(30);
        let _ = load(&moved);
        let after_move = solve_count();
        assert_eq!(after_move, before + 1, "moving the boolean: no solve");

        // A descendant's size: solves.
        let mut resized = moved.clone();
        resized["nodes"][2]["width"] = json!(60);
        let _ = load(&resized);
        assert_eq!(solve_count(), after_move + 1, "resize: solves");

        // The operation: solves.
        let mut op = resized.clone();
        op["nodes"][0]["booleanOperation"] = json!("union");
        let _ = load(&op);
        assert_eq!(solve_count(), after_move + 2, "operation change: solves");
    }

    #[test]
    fn a_nested_change_invalidates_every_ancestor() {
        clear_cache();
        let v = json!({
            "nodes": [
                {"id": "b", "type": "boolean", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
                 "children": ["inner", "r3"], "booleanOperation": "union"},
                {"id": "inner", "type": "boolean", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
                 "children": ["r1", "r2"], "booleanOperation": "intersect"},
                rect("r1", 0.0, 0.0, 100.0, 100.0),
                rect("r2", 50.0, 50.0, 100.0, 100.0),
                rect("r3", 300.0, 300.0, 10.0, 10.0),
            ],
            "pages": [{"id": "p", "width": 500, "height": 500, "nodeIds": ["b"]}]
        });
        let _ = load(&v);
        let base = solve_count();
        let _ = load(&v);
        assert_eq!(solve_count(), base);
        let mut deep = v.clone();
        deep["nodes"][3]["y"] = json!(60);
        let s = load(&deep);
        assert_eq!(solve_count(), base + 2, "inner and outer both re-solve");
        let (pts, _) = drawn(&s);
        // inner is now 50×40.
        assert!((area(&pts) - 2_100.0).abs() < 1.0, "area {}", area(&pts));
    }
}
