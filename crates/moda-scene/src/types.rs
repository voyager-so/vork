//! Core scene types for moda-render.
//!
//! ============================================================================
//! FROZEN AT M0c EXIT (PLAN.md §0 rule 4).
//!
//! This file is the shared-type contract between moda-scene and every other
//! crate in the workspace (`crates/moda-scene/src/types.rs` is frozen at M0c
//! exit; `crates/moda-engine/src/seams.rs` freezes at M1 exit). Changes after
//! M0c require a recorded interface-change note and sign-off from all
//! in-flight track owners (DECISION.md tripwire #5 applies by analogy).
//! ============================================================================
//!
//! Design: the scene document is parsed PERMISSIVELY (PLAN M0c: "permissive
//! serde — RawNode + unknown-field spillover"). Nodes are kept as ordered maps
//! of typed values ([`crate::value`]: numbers are numbers, known field names an
//! enum), mirroring the dynamic objects the TS `Scene` mutates in place during
//! migration, with typed accessors layered on top. Unknown fields are
//! detected against per-node-type known-field tables (`fields.rs`) and feed
//! the manifest's `unknown` verdict (PLAN §3).
//!
//! Ground truth: `research/inventory/scene-schema.md`; node definitions
//! `core/scene/nodes.ts`; envelope `core/scene/Scene.ts:2613-2796`.

use crate::value::{Map, Value};
use std::collections::BTreeMap;

/// Ordered object of typed values ([`crate::value::Map`]), mirroring JS object
/// insertion-order semantics that the TS migrations rely on.
pub type JsonMap = Map;

/// Node id — arbitrary string (`core/types.ts:2`).
pub type NodeId = String;

/// The scene-root pseudo-parent id (`core/scene/scene.constants.ts:1`).
pub const SCENE_ROOT_ID: &str = "canvas";

/// Page-background transient node id prefix (`core/utils/idGenerator.ts:23`).
pub const PAGE_BG_PREFIX: &str = "__page_bg_";

/// `computePageBgId` clone (`core/utils/idGenerator.ts:25-27`).
pub fn compute_page_bg_id(page_id: &str) -> NodeId {
    format!("{PAGE_BG_PREFIX}{page_id}")
}

/// `isPageBgId` clone (`core/utils/idGenerator.ts:29-31`).
pub fn is_page_bg_id(id: &str) -> bool {
    id.starts_with(PAGE_BG_PREFIX)
}

/// Valid node types — the Rust mirror of `SCENE_NODE_TYPES` in `core/scene/nodes.ts`
/// (the canonical roster, compile-time pinned to the TS `SceneNode` union there). Legacy
/// `'text'` and `'image'` are silently dropped at hydration — that is a *known* drop, not
/// an "unknown" manifest verdict; see `KNOWN_DROPPED_NODE_TYPES` below.
///
/// This roster must be a **superset** of the TS one, and the asymmetry is the whole design:
/// - A type in `nodes.ts` but **missing here is silently DROPPED** at native hydration
///   (`scene.rs`) while the browser renders it — a bug, and one no other gate caught before
/// - A type here but **not yet in `nodes.ts` is legitimate** — an engine must render a NEWER
///   doc's node as known-but-unsupported (hydrates, one manifest axis, no crash) rather than
///   dropping it and reporting a clean `full` manifest for a doc it did not actually draw.
///
/// `tests/scene_schema_parity.rs` enforces exactly that: TS ⊆ this list, with every
/// deliberately-ahead entry declared in its `RUST_AHEAD_OF_TS` list. So a missing mirror
/// fails the build, a typo here fails the build, and a declared forward-registration passes.
///
pub const VALID_NODE_TYPES: [&str; 16] = [
    "rectangle",
    "ellipse",
    "line",
    "path",
    "group",
    "boolean",
    "container",
    "section",
    "star",
    "polygon",
    "richtext",
    "blend",
    "table",
    "generated",
    // A live generator (live-nodes.md §3): code + params; its drawn content is derived (`crate::generator`).
    "generator",
    // Raster paint layer (docs/design/raster-layers.md §4.3; spike). Hydration rewrites it as a rectangle whose
    // one image fill samples the layer's surface texture (`normalize::normalize_paint_in_place`).
    "paint",
];

/// Legacy node types the browser knowingly skips (see `isValidNodeType` in
/// `core/scene/Scene.ts` and its call site in the `nodes` setter).
///
/// `camera` is reserved by format 26 (motion-model.md §2.1, §12.6) and built later: a document holding one loads, and
/// it draws nothing yet. (A flipbook is a `group` with a `flipbook` object since format 28, `crate::flipbook`.)
pub const KNOWN_DROPPED_NODE_TYPES: [&str; 3] = ["text", "image", "camera"];

// ---------------------------------------------------------------------------
// RawNode
// ---------------------------------------------------------------------------

/// A key of an object. Small objects (every node, most paints) are scanned in order instead of hashed
/// ([`crate::value::Map::get`]).
#[inline]
pub fn lookup<'a>(map: &'a JsonMap, key: &str) -> Option<&'a Value> {
    map.get(key)
}

/// [`lookup`] of a known field name: the scan compares integers, not text.
#[inline]
pub fn lookup_known(map: &JsonMap, key: crate::value::K) -> Option<&Value> {
    map.get_known(key)
}

/// A scene node held as a raw ordered JSON object, exactly as the TS `Scene`
/// holds dynamic node objects. Migrations mutate it in place; accessors apply
/// the render-time defaults table (scene-schema.md §15) on read.
///
/// A path's `points` may be held compactly ([`crate::path_points::CompactPoints`], see [`RawNode::compact`]): the
/// map then keeps `"points": null` in their place (so its keys, and their order, are the node's). Semantic reads
/// use [`RawNode::get`] or [`RawNode::points`]; the placeholder is never a generic field value. [`RawNode::fields`] is the map as held; [`RawNode::to_json_map`] /
/// [`RawNode::into_json_map`] give the node's JSON with its points, for code that hands the node on as JSON.
/// Clones share immutable fields; every mutable accessor detaches its copy before writing.
#[derive(Debug, Clone)]
pub struct RawNode(
    std::sync::Arc<JsonMap>,
    Option<std::sync::Arc<crate::path_points::CompactPoints>>,
    PaintMemo,
);

/// A node's parsed stroke, fill and effect stacks (`paint::node_strokes`, `paint::node_fills`,
/// `effect::node_effects`), kept with the node: a plan build reads a node's paints and effects many times (its width,
/// its brush, its top fill, whether its fills are images, its shadows, whether it has a layer filter), and each read
/// deserialized them from JSON again. Every change to the node's fields drops all three; a clone shares them.
#[derive(Clone, Default)]
pub(crate) struct PaintMemo(
    std::sync::OnceLock<std::sync::Arc<[crate::paint::Stroke]>>,
    std::sync::OnceLock<std::sync::Arc<[crate::paint::Fill]>>,
    std::sync::OnceLock<std::sync::Arc<[crate::effect::Effect]>>,
);

impl std::fmt::Debug for PaintMemo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PaintMemo")
    }
}

/// Equal as JSON (a compact node equals its JSON form).
impl PartialEq for RawNode {
    fn eq(&self, other: &RawNode) -> bool {
        match (&self.1, &other.1) {
            (None, None) => self.0 == other.0,
            (Some(a), Some(b)) => self.0 == other.0 && (std::sync::Arc::ptr_eq(a, b) || a == b),
            _ => self.0.len() == other.0.len() && self.0.keys().all(|key| self.same_field(other, key)),
        }
    }
}

/// A present semantic field. `None` from `RawNode::get` means absent;
/// `Json(Value::Null)` means authored null; compact points are never a placeholder.
#[derive(Debug, Clone, Copy)]
pub enum NodeField<'a> {
    Json(&'a Value),
    CompactPoints(&'a crate::path_points::CompactPoints),
}

impl NodeField<'_> {
    pub fn eq_json(self, value: &Value) -> bool {
        match self {
            Self::Json(v) => v == value,
            Self::CompactPoints(points) => value.as_array().is_some_and(|v| points.eq_json(v)),
        }
    }
}

impl PartialEq for NodeField<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Json(a), Self::Json(b)) => a == b,
            (Self::CompactPoints(a), Self::CompactPoints(b)) => std::ptr::eq(a, b) || a == b,
            (a, Self::Json(b)) | (Self::Json(b), a) => a.eq_json(b),
        }
    }
}

impl RawNode {
    /// A node from its JSON object (points held as JSON; [`RawNode::compact`] stores them compactly).
    pub fn new(map: JsonMap) -> RawNode {
        RawNode(std::sync::Arc::new(map), None, PaintMemo::default())
    }

    /// A path node whose points are `points`, held compactly: `map` keeps `"points": null` in their place.
    pub fn with_compact_points(mut map: JsonMap, points: crate::path_points::CompactPoints) -> RawNode {
        map.insert("points".into(), Value::Null);
        RawNode(std::sync::Arc::new(map), Some(std::sync::Arc::new(points)), PaintMemo::default())
    }

    /// A path node whose points are `points`, shared with every node that draws the same geometry (nothing is
    /// copied): `map` keeps `"points": null` in their place.
    pub fn with_shared_points(mut map: JsonMap, points: std::sync::Arc<crate::path_points::CompactPoints>) -> RawNode {
        map.insert("points".into(), Value::Null);
        RawNode(std::sync::Arc::new(map), Some(points), PaintMemo::default())
    }

    /// Replace the node's points with compact ones, shared as [`RawNode::with_shared_points`] shares them.
    pub fn set_shared_points(&mut self, points: std::sync::Arc<crate::path_points::CompactPoints>) {
        self.2 = PaintMemo::default();
        std::sync::Arc::make_mut(&mut self.0).insert("points".into(), Value::Null);
        self.1 = Some(points);
    }

    /// The compact points and their sharing, when the node holds them so.
    pub fn shared_points(&self) -> Option<&std::sync::Arc<crate::path_points::CompactPoints>> {
        self.1.as_ref()
    }

    /// `normalize::compact_map` of the node's fields, keeping compact points compact.
    pub fn compact_fields(&mut self) {
        self.2 = PaintMemo::default();
        crate::normalize::compact_map(std::sync::Arc::make_mut(&mut self.0));
    }

    pub fn from_value(v: impl Into<Value>) -> Option<RawNode> {
        match v.into() {
            Value::Object(m) => Some(RawNode(std::sync::Arc::new(m), None, PaintMemo::default())),
            _ => None,
        }
    }

    /// Store a path's `points` compactly when they all have a canonical shape (`path_points`); no-op otherwise.
    /// Returns whether the node is now compact.
    pub fn compact(&mut self) -> bool {
        if self.1.is_some() {
            return true;
        }
        if self.node_type() != "path" {
            return false;
        }
        let Some(c) = self
            .0
            .get("points")
            .and_then(Value::as_array)
            .and_then(|a| crate::path_points::CompactPoints::from_json(a))
        else {
            return false;
        };
        if let Some(slot) = std::sync::Arc::make_mut(&mut self.0).get_mut("points") {
            *slot = Value::Null;
        }
        self.1 = Some(std::sync::Arc::new(c));
        true
    }

    /// Whether the node's points are held compactly.
    pub fn is_compact(&self) -> bool {
        self.1.is_some()
    }

    /// Bring compact points back into the map as JSON.
    pub(crate) fn decompact(&mut self) {
        if let Some(c) = self.1.take() {
            if let Some(slot) = std::sync::Arc::make_mut(&mut self.0).get_mut("points") {
                *slot = Value::Array(c.to_json());
            }
        }
    }

    /// Storage-only map: compact points have a null placeholder. Do not compare
    /// or reconstruct nodes from this map; use `same_field`, node equality,
    /// `with_fields` for derivation, or `to_json_map` at a JSON boundary.
    pub fn fields(&self) -> &JsonMap {
        &self.0
    }

    /// The node's fields to change in place (its points are brought back to JSON first).
    pub fn fields_mut(&mut self) -> &mut JsonMap {
        self.decompact();
        self.2 = PaintMemo::default();
        std::sync::Arc::make_mut(&mut self.0)
    }

    /// The node as JSON, points included.
    pub fn to_json_map(&self) -> JsonMap {
        let mut m = self.0.as_ref().clone();
        if let Some(c) = &self.1 {
            if let Some(slot) = m.get_mut("points") {
                *slot = Value::Array(c.to_json());
            }
        }
        m
    }

    /// The node as JSON, points included.
    pub fn into_json_map(mut self) -> JsonMap {
        self.decompact();
        std::sync::Arc::unwrap_or_clone(self.0)
    }

    /// A node derived from this one: `fields` (typically a changed copy of [`RawNode::fields`]) sharing this
    /// node's compact points while `fields` still holds their `"points": null` placeholder.
    pub fn with_fields(&self, fields: JsonMap) -> RawNode {
        let keep = self.1.is_some() && matches!(fields.get("points"), Some(Value::Null));
        RawNode(std::sync::Arc::new(fields), if keep { self.1.clone() } else { None }, PaintMemo::default())
    }

    /// The compact points, when the node holds them so.
    pub fn compact_points(&self) -> Option<&crate::path_points::CompactPoints> {
        self.1.as_deref()
    }

    /// Whether field `key` is the same on both nodes (absent on both counts as the same), however each holds
    /// its points.
    pub fn same_field(&self, other: &RawNode, key: &str) -> bool {
        self.get(key) == other.get(key)
    }

    /// A semantic view without materializing compact geometry.
    pub fn get(&self, key: &str) -> Option<NodeField<'_>> {
        if key == "points" {
            if let Some(points) = self.compact_points() { return Some(NodeField::CompactPoints(points)); }
        }
        lookup(&self.0, key).map(NodeField::Json)
    }

    pub fn field_eq_json(&self, key: &str, value: &Value) -> bool {
        self.get(key).is_some_and(|field| field.eq_json(value))
    }

    /// Heap bytes the compact points hold (0 when held as JSON).
    pub fn compact_points_bytes(&self) -> usize {
        self.1.as_ref().map_or(0, |c| c.heap_bytes())
    }

    pub fn id(&self) -> &str {
        lookup_known(&self.0, crate::value::K::Id).and_then(Value::as_str).unwrap_or("")
    }

    pub fn node_type(&self) -> &str {
        lookup_known(&self.0, crate::value::K::Type).and_then(Value::as_str).unwrap_or("")
    }

    /// A number field named by a known key (`f64` of its name, without comparing text).
    pub fn f64_known(&self, key: crate::value::K) -> Option<f64> {
        lookup_known(&self.0, key).and_then(Value::as_f64)
    }

    /// Borrow a field known to be JSON-backed. Compact points are not one: debug builds panic, and
    /// release builds see the storage placeholder. Generic readers use `get`, `same_field` or `field_eq_json`.
    pub fn json_field(&self, key: &str) -> Option<&Value> {
        debug_assert!(
            !(key == "points" && self.1.is_some()),
            "a compact node's points are read through RawNode::points"
        );
        lookup(&self.0, key)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        if key == "points" {
            self.decompact();
        }
        self.2 = PaintMemo::default();
        std::sync::Arc::make_mut(&mut self.0).get_mut(key)
    }

    pub fn set(&mut self, key: &str, value: impl Into<Value>) {
        if key == "points" {
            self.1 = None;
        }
        self.2 = PaintMemo::default();
        std::sync::Arc::make_mut(&mut self.0).insert(key.into(), value.into());
    }

    /// Overwrite a placement number (`x`, `y`, `rotation`, `scaleX`, `scaleY`, `opacity`) in place, or add it.
    /// No paint, effect or geometry parse reads these, so the paint memo and compact points stay as they are.
    /// The caller builds the number as it would for a document edit (a host that keys `f32` values passes
    /// the number of `Value::from(f32)`, which reads back differently from the widened `f64` when numbers keep their text).
    ///
    /// Returns whether the field changed: a field that already holds this number is left as it is.
    pub fn set_placement_number(&mut self, key: &str, value: crate::value::Number) -> bool {
        debug_assert!(matches!(key, "x" | "y" | "rotation" | "scaleX" | "scaleY" | "opacity"));
        let map = std::sync::Arc::make_mut(&mut self.0);
        match map.get_mut(key) {
            Some(Value::Number(held)) if *held == value => false,
            Some(slot) => {
                *slot = Value::Number(value);
                true
            }
            None => {
                map.insert(key.into(), Value::Number(value));
                true
            }
        }
    }

    pub fn remove(&mut self, key: &str) -> Option<Value> {
        if key == "points" {
            self.decompact();
        }
        self.2 = PaintMemo::default();
        std::sync::Arc::make_mut(&mut self.0).shift_remove(key)
    }

    /// The node's parsed stroke stack, parsed once (see `PaintMemo`).
    pub(crate) fn strokes_memo(
        &self,
        parse: impl FnOnce() -> Vec<crate::paint::Stroke>,
    ) -> std::sync::Arc<[crate::paint::Stroke]> {
        self.2 .0.get_or_init(|| parse().into()).clone()
    }

    /// The node's parsed fill stack, parsed once (see `PaintMemo`).
    pub(crate) fn fills_memo(&self, parse: impl FnOnce() -> Vec<crate::paint::Fill>) -> std::sync::Arc<[crate::paint::Fill]> {
        self.2 .1.get_or_init(|| parse().into()).clone()
    }

    /// The node's parsed effect list, parsed once (see `PaintMemo`).
    pub(crate) fn effects_memo(&self, parse: impl FnOnce() -> Vec<crate::effect::Effect>) -> std::sync::Arc<[crate::effect::Effect]> {
        self.2 .2.get_or_init(|| parse().into()).clone()
    }

    pub fn has(&self, key: &str) -> bool {
        lookup(&self.0, key).is_some()
    }

    pub fn f64(&self, key: &str) -> Option<f64> {
        lookup(&self.0, key).and_then(Value::as_f64)
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        lookup(&self.0, key).and_then(Value::as_str)
    }

    pub fn bool(&self, key: &str) -> Option<bool> {
        lookup(&self.0, key).and_then(Value::as_bool)
    }

    pub fn array(&self, key: &str) -> Option<&Vec<Value>> {
        debug_assert!(
            !(key == "points" && self.1.is_some()),
            "a compact node's points are read through RawNode::points"
        );
        lookup(&self.0, key).and_then(Value::as_array)
    }

    /// The node's `points` (a path's point objects, a line's flat coordinates): read them through this, never
    /// through `get("points")` / `array("points")`, so the scene can store them compactly (`path_points`).
    pub fn points(&self) -> Option<crate::path_points::PathPoints<'_>> {
        if let Some(c) = &self.1 {
            return Some(crate::path_points::PathPoints::Compact(c));
        }
        lookup(&self.0, "points")
            .and_then(Value::as_array)
            .map(|a| crate::path_points::PathPoints::Json(a))
    }

    pub fn object(&self, key: &str) -> Option<&JsonMap> {
        lookup(&self.0, key).and_then(Value::as_object)
    }

    /// JS truthiness for a field value (used to clone `!!node.foo` checks).
    pub fn truthy(&self, key: &str) -> bool {
        if key == "points" && self.1.is_some() {
            return true; // an array
        }
        match lookup(&self.0, key) {
            None | Some(Value::Null) => false,
            Some(Value::Bool(b)) => *b,
            Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0 && !f.is_nan()).unwrap_or(false),
            Some(Value::String(s)) => !s.is_empty(),
            Some(Value::Array(_)) | Some(Value::Object(_)) => true,
        }
    }
}

// ---------------------------------------------------------------------------
// Pages
// ---------------------------------------------------------------------------

#[cfg(test)]
mod copy_on_write_tests {
    use super::*;
    use crate::value::json;
    use std::sync::Arc;

    fn node() -> RawNode {
        RawNode::from_value(json!({"id":"P","type":"path","metadata":{"name":"P"},"points":[{"type":"move","x":1,"y":2}],"opacity":1})).unwrap()
    }

    #[test]
    fn semantic_fields_distinguish_absence_null_and_compact_points() {
        let authored = node();
        let mut compact = authored.clone();
        assert!(compact.compact());
        assert!(matches!(compact.get("points"), Some(NodeField::CompactPoints(_))));
        assert!(compact.get("missing").is_none());
        assert!(compact.same_field(&authored, "points"));
        assert!(compact.field_eq_json("points", authored.json_field("points").unwrap()));
        assert!(!compact.field_eq_json("points", &Value::Null));
        let mut null = compact.clone();
        null.set("points", Value::Null);
        assert!(matches!(null.get("points"), Some(NodeField::Json(Value::Null))));
        assert!(!compact.same_field(&null, "points"));
        null.remove("points");
        assert!(null.get("points").is_none());
        let empty = RawNode::with_compact_points(JsonMap::new(), Default::default());
        assert!(empty.has("points"));
        assert!(empty.field_eq_json("points", &json!([])));
        assert!(!empty.same_field(&null, "points"));
        let derived = compact.with_fields(compact.fields().clone());
        assert_eq!(derived, authored);
        *compact.get_mut("points").unwrap() = json!([]);
        assert!(compact.field_eq_json("points", &json!([])));
        assert!(!compact.same_field(&authored, "points"));
    }

    #[test]
    fn compact_field_equality_matches_materialized_json() {
        for points in [
            json!([]),
            json!([{"type":"move","x":1,"y":2.0},{"type":"line","x":-3,"y":4}]),
            json!([{"type":"curve","x":1,"y":2,"cp1":{"x":3.0,"y":4},"cp2":{"x":5,"y":6}}]),
            json!([{"type":"quadratic","x":1,"y":2,"cp":{"x":3,"y":4}}]),
        ] {
            let json_node = RawNode::from_value(json!({"type":"path","points":points})).unwrap();
            let mut compact = json_node.clone();
            assert!(compact.compact());
            for candidate in [points.clone(), Value::Null, json!([]),
                json!([{"type":"move","x":1.0,"y":2.0}]),
                json!([{"type":"move","x":1,"y":2.0,"extra":0}])] {
                assert_eq!(compact.field_eq_json("points", &candidate), points == candidate);
            }
            assert!(compact.same_field(&json_node, "points"));
            assert!(json_node.same_field(&compact, "points"));
        }
    }

    #[test]
    fn field_mutators_detach_and_preserve_original_bytes_and_paint_memo() {
        let original = node();
        let bytes = crate::value::to_vec(&original.to_json_map()).unwrap();
        let strokes = original.strokes_memo(Vec::new);
        let fills = original.fills_memo(Vec::new);
        let effects = original.effects_memo(Vec::new);
        for mutate in [
            |node: &mut RawNode| { node.set("opacity", json!(0.5)); },
            |node: &mut RawNode| { *node.get_mut("opacity").unwrap() = json!(0.5); },
            |node: &mut RawNode| { node.remove("opacity"); },
            |node: &mut RawNode| { node.fields_mut().insert("opacity".into(), json!(0.5)); },
            |node: &mut RawNode| { node.compact_fields(); },
        ] {
            let mut changed = original.clone();
            assert!(Arc::ptr_eq(&original.0, &changed.0));
            assert!(Arc::ptr_eq(&strokes, &changed.strokes_memo(|| panic!("shared memo"))));
            assert!(Arc::ptr_eq(&fills, &changed.fills_memo(|| panic!("shared memo"))));
            assert!(Arc::ptr_eq(&effects, &changed.effects_memo(|| panic!("shared memo"))));
            mutate(&mut changed);
            assert!(!Arc::ptr_eq(&original.0, &changed.0));
            assert!(changed.2 .0.get().is_none());
            assert!(changed.2 .1.get().is_none());
            assert!(changed.2 .2.get().is_none());
            assert_eq!(crate::value::to_vec(&original.to_json_map()).unwrap(), bytes);
            assert!(Arc::ptr_eq(&strokes, &original.strokes_memo(|| panic!("original memo"))));
            assert!(Arc::ptr_eq(&fills, &original.fills_memo(|| panic!("original memo"))));
            assert!(Arc::ptr_eq(&effects, &original.effects_memo(|| panic!("original memo"))));
        }
    }

    #[test]
    fn compact_points_detach_on_compaction_and_every_mutable_path() {
        let original = node();
        let bytes = crate::value::to_vec(&original.to_json_map()).unwrap();
        let mut compact = original.clone();
        assert!(compact.compact());
        assert!(!Arc::ptr_eq(&original.0, &compact.0));
        assert!(!original.is_compact());
        assert_eq!(compact, original);
        assert_eq!(crate::value::to_vec(&compact.to_json_map()).unwrap(), bytes);
        for mutate in [
            |node: &mut RawNode| { node.set("points", json!([])); },
            |node: &mut RawNode| { *node.get_mut("points").unwrap() = json!([]); },
            |node: &mut RawNode| { node.remove("points"); },
            |node: &mut RawNode| { node.fields_mut().insert("points".into(), json!([])); },
        ] {
            let mut changed = compact.clone();
            assert!(Arc::ptr_eq(&compact.0, &changed.0));
            mutate(&mut changed);
            assert!(!changed.is_compact());
            assert_ne!(changed, compact);
            assert_eq!(crate::value::to_vec(&compact.to_json_map()).unwrap(), bytes);
        }
        let derived = compact.with_fields(compact.fields().clone());
        assert!(Arc::ptr_eq(compact.1.as_ref().unwrap(), derived.1.as_ref().unwrap()));
        assert_eq!(derived, original);
        assert_eq!(crate::value::to_vec(&derived.into_json_map()).unwrap(), bytes);
        assert_eq!(crate::value::to_vec(&compact.clone().into_json_map()).unwrap(), bytes);
        assert_eq!(crate::value::to_vec(&compact.into_json_map()).unwrap(), bytes);
        assert_eq!(crate::value::to_vec(&original.clone().into_json_map()).unwrap(), bytes);
        assert_eq!(crate::value::to_vec(&original.into_json_map()).unwrap(), bytes);
    }

    #[test]
    fn clones_keep_fields_alive_after_the_original_drops() {
        let mut shared = { let original = node(); original.clone() };
        let bytes = crate::value::to_vec(&shared.to_json_map()).unwrap();
        assert_eq!(Arc::strong_count(&shared.0), 1);
        let pointer = Arc::as_ptr(&shared.0);
        shared.set("opacity", json!(0.5));
        assert_eq!(Arc::as_ptr(&shared.0), pointer);
        assert_ne!(crate::value::to_vec(&shared.to_json_map()).unwrap(), bytes);
    }
}

/// A page, kept raw like nodes (page background fill keys are the full §3.5
/// surface — `core/scene/scene.pages.ts:73-101`). Typed accessors for the
/// structural fields.
#[derive(Debug, Clone, PartialEq)]
pub struct RawPage(pub JsonMap);


impl RawPage {
    pub fn id(&self) -> &str {
        self.0.get("id").and_then(Value::as_str).unwrap_or("")
    }
    pub fn width(&self) -> f64 {
        self.0.get("width").and_then(Value::as_f64).unwrap_or(0.0)
    }
    pub fn height(&self) -> f64 {
        self.0.get("height").and_then(Value::as_f64).unwrap_or(0.0)
    }
    pub fn hidden(&self) -> bool {
        self.0
            .get("hidden")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }
    pub fn node_ids(&self) -> Vec<String> {
        self.0
            .get("nodeIds")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }
    pub fn set_node_ids(&mut self, ids: Vec<String>) {
        self.0.insert("nodeIds".into(),
            Value::Array(ids.into_iter().map(Value::String).collect()),
        );
    }
}

// ---------------------------------------------------------------------------
// Scene config (`core/scene/scene.config.ts:79-128`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct SceneConfig {
    /// `config.renderPageClipping ?? true`.
    pub render_page_clipping: bool,
    /// Raw config object (unknown keys preserved/ignored gracefully).
    pub raw: JsonMap,
}

impl Default for SceneConfig {
    fn default() -> Self {
        SceneConfig {
            render_page_clipping: true,
            raw: JsonMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Scene document envelope (`canvas_data`, scene-schema.md §1.2)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct SceneDocument {
    /// Flat node array (z-order = array order), AFTER hydration filtering +
    /// migration. Hierarchy via group/container `children` id refs.
    pub nodes: Vec<RawNode>,
    pub pages: Vec<RawPage>,
    /// Document brush table (`brushes`, brushes.md §4.4): definition objects
    /// with a string `id`, in document order. Read by the stroke outliner
    /// (calligraphic nibs) through [`SceneDocument::brush`]. Shared, so a
    /// clone of the document (one per exported frame, more with 3D layers)
    /// doesn't copy the table's JSON; replace it whole to change it.
    pub brushes: std::sync::Arc<Vec<Value>>,
    /// A content hash per entry of `brushes` (same order), computed once at
    /// hydration: render caches key on it instead of re-serialising brushes.
    pub brush_hashes: Vec<u64>,
    /// Document pattern table (`patterns`, E8.5, [`crate::pattern`]): definition objects with a string `id`,
    /// in document order. Read by the engine's pattern paint through [`SceneDocument::pattern`]. Shared, as `brushes`.
    pub patterns: std::sync::Arc<Vec<Value>>,
    /// A content hash per entry of `patterns` (same order), computed once at hydration.
    pub pattern_hashes: Vec<u64>,
    /// Document code table (`code`, live-nodes.md §3.2): `CodeDef`s keyed by content hash (`b3:…`), read by
    /// generator nodes ([`crate::generator`]).
    pub code: crate::value::Map,
    pub config: SceneConfig,
    pub canvas_root_node_ids: Vec<NodeId>,
    /// 'vertical'|'horizontal'|'grid'|'pages'|'freeform'; absent => vertical
    /// behavior (`core/scene/scene.layout.ts:17-18`).
    pub layout_mode: Option<String>,
    pub grid_columns: Option<f64>,
    /// Envelope keys we did not recognize (manifest `unknown` feed).
    pub unknown_envelope_keys: Vec<String>,
    /// Node types that were dropped at hydration because they are not in
    /// `VALID_NODE_TYPES` and not in `KNOWN_DROPPED_NODE_TYPES` — these are
    /// schema-evolution signals for the manifest (`unknown:node-type`).
    pub unknown_node_types: Vec<(NodeId, String)>,
}

impl SceneDocument {
    /// The brush definition with this id (the first, if ids repeat).
    pub fn brush(&self, id: &str) -> Option<&Value> {
        self.brushes
            .iter()
            .find(|b| b.get("id").and_then(Value::as_str) == Some(id))
    }

    /// The pattern definition with this id (the first, if ids repeat).
    pub fn pattern(&self, id: &str) -> Option<&Value> {
        self.patterns
            .iter()
            .find(|p| p.get("id").and_then(Value::as_str) == Some(id))
    }

    /// The content hash of the pattern with this id (see `pattern_hashes`).
    pub fn pattern_hash(&self, id: &str) -> Option<u64> {
        self.patterns
            .iter()
            .position(|p| p.get("id").and_then(Value::as_str) == Some(id))
            .and_then(|i| self.pattern_hashes.get(i).copied())
    }

    /// The content hash of the brush with this id (see `brush_hashes`).
    pub fn brush_hash(&self, id: &str) -> Option<u64> {
        self.brushes
            .iter()
            .position(|b| b.get("id").and_then(Value::as_str) == Some(id))
            .and_then(|i| self.brush_hashes.get(i).copied())
    }
}

// ---------------------------------------------------------------------------
// Affine transform (`core/scene/affineTransform.ts:18-25, 35-41`)
// ---------------------------------------------------------------------------

/// 2x3 affine: x' = a*x + c*y + e ; y' = b*x + d*y + f.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

/// Decomposed TRS (degrees), `core/scene/affineTransform.ts:35-41`.
/// world = Translate(x,y) * Rotate(rotation) * Scale(scaleX,scaleY) * local.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2D {
    pub x: f64,
    pub y: f64,
    pub rotation: f64,
    pub scale_x: f64,
    pub scale_y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

// ---------------------------------------------------------------------------
// Display list (`core/rendering/displayList.ts`)
// ---------------------------------------------------------------------------

/// `PaintKind` (displayList.ts:10-20).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintKind {
    PageBackground,
    ShapeFill,
    ShapeStroke,
    ShapeShadow,
    Richtext,
    ShapeText,
    ConnectorText,
    Image,
    Svg,
    GroupBoundary,
}

impl PaintKind {
    /// Exact TS string literal — the oracle-dump contract.
    pub fn as_str(self) -> &'static str {
        match self {
            PaintKind::PageBackground => "pageBackground",
            PaintKind::ShapeFill => "shapeFill",
            PaintKind::ShapeStroke => "shapeStroke",
            PaintKind::ShapeShadow => "shapeShadow",
            PaintKind::Richtext => "richtext",
            PaintKind::ShapeText => "shapeText",
            PaintKind::ConnectorText => "connectorText",
            PaintKind::Image => "image",
            PaintKind::Svg => "svg",
            PaintKind::GroupBoundary => "groupBoundary",
        }
    }
}

/// `PaintPhase` (displayList.ts:22).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintPhase {
    Background,
    Opaque,
    Transparent,
    Overlay,
}

impl PaintPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            PaintPhase::Background => "background",
            PaintPhase::Opaque => "opaque",
            PaintPhase::Transparent => "transparent",
            PaintPhase::Overlay => "overlay",
        }
    }
}

/// `PaintEffectDescriptor.kind` (displayList.ts:35-38).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintEffectKind {
    Shadow,
    Stroke,
    BackgroundBlur,
    BlendMode,
    GroupOpacity,
    Mask,
    Pattern,
    Shader,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PaintEffect {
    pub kind: PaintEffectKind,
    pub source_node_id: NodeId,
}

/// `ClipDescriptor` (displayList.ts:40-44).
#[derive(Debug, Clone, PartialEq)]
pub struct ClipDescriptor {
    pub owner_node_id: NodeId,
    pub mask_node_id: Option<NodeId>,
    /// 'group-mask' | 'container-clip'
    pub mode: &'static str,
}

/// `PaintFallbackReason` (displayList.ts:24-33).
pub type FallbackReason = &'static str;

/// `PaintItem` (displayList.ts:54-70).
#[derive(Debug, Clone, PartialEq)]
pub struct PaintItem {
    pub id: String,
    pub node_id: NodeId,
    /// node type string, or "pageBackground" for the transient bg rect.
    pub node_type: String,
    pub kind: PaintKind,
    pub z_index: usize,
    pub phase: PaintPhase,
    pub bounds_canvas: Option<Rect>,
    pub world_transform: Option<Transform2D>,
    pub world_affine: Option<Affine>,
    pub opacity: f64,
    pub clip_chain: Vec<ClipDescriptor>,
    /// Some(mode) only when blendMode present and != 'normal'
    /// (displayList.ts:367-369).
    pub blend_mode: Option<String>,
    pub effects: Vec<PaintEffect>,
    pub fallback_reasons: Vec<FallbackReason>,
    pub is_primary_node_item: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DisplayList {
    pub page_id: String,
    pub page_index: usize,
    pub items: Vec<PaintItem>,
}

// ---------------------------------------------------------------------------
// Feature manifest (PLAN.md §3 — schema frozen at M0c)
// ---------------------------------------------------------------------------

/// Axis/page/document verdict. `Unknown` (permissive-serde spillover hit
/// something new) is distinct from gated/`Unsupported` (PLAN §3).
#[derive(Debug, Clone, PartialEq)]
pub enum AxisStatus {
    Full,
    Degraded { expected_ssim_band: (f64, f64) },
    Unsupported,
    Unknown,
}

impl AxisStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            AxisStatus::Full => "full",
            AxisStatus::Degraded { .. } => "degraded",
            AxisStatus::Unsupported => "unsupported",
            AxisStatus::Unknown => "unknown",
        }
    }
    /// Severity for verdict aggregation: unsupported > unknown > degraded > full.
    pub fn severity(&self) -> u8 {
        match self {
            AxisStatus::Full => 0,
            AxisStatus::Degraded { .. } => 1,
            AxisStatus::Unknown => 2,
            AxisStatus::Unsupported => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AxisEntry {
    /// e.g. "generated:chart", "node:blend", "shader:plasma-field",
    /// "text:fallback-engine", "unknown:node-field".
    pub axis: String,
    pub status: AxisStatus,
    pub node_ids: Vec<NodeId>,
    /// Free-form detail object (families, scripts, spec_type, field, ...).
    pub detail: JsonMap,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ManifestWarning {
    pub axis: String,
    pub detail: JsonMap,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PageManifest {
    pub page_index: usize,
    pub page_id: String,
    pub verdict: AxisStatus,
    pub axes: Vec<AxisEntry>,
    pub warnings: Vec<ManifestWarning>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    /// Always "moda-render".
    pub renderer: &'static str,
    /// Engine git sha — provided by the caller (CLI), None in library tests.
    pub engine_version: Option<String>,
    pub canvas_id: Option<String>,
    pub verdict: AxisStatus,
    pub pages: Vec<PageManifest>,
    /// Page-less freeform docs only: the canvas-root FRAME size (content
    /// bounds + margin, canvas px at scale 1 — `canvas_root_frame_size`).
    /// The backend derives render caps (e.g. the thumbnail preview dim) from
    /// this instead of re-deriving scene geometry python-side. `None` for
    /// paged docs.
    pub canvas_root_frame_px: Option<(f64, f64)>,
}

impl Manifest {
    /// `renderable = no unsupported axis on requested pages` (PLAN §3).
    pub fn renderable(&self) -> bool {
        !self.pages.iter().any(|p| {
            p.axes
                .iter()
                .any(|a| matches!(a.status, AxisStatus::Unsupported))
        })
    }
}

// ---------------------------------------------------------------------------
// Font catalog (headless response `fonts` — FontFamilyCatalogEntry[],
// scene-schema.md §1.1/§14; MSDF predicate `core/config/msdfFonts.ts:27-31`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct FontVariantEntry {
    pub weight: f64,
    /// 'normal' | 'italic'
    pub style: String,
    pub msdf_ready: bool,
    pub atlas_urls: Vec<String>,
    pub metrics_url: Option<String>,
    pub ttf_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FontCatalog {
    /// lowercase family name -> variants (`utils/fontStore.ts:223-225` —
    /// lookup is `familyName.toLowerCase()`).
    pub families: BTreeMap<String, Vec<FontVariantEntry>>,
}

impl FontCatalog {
    /// `isMsdfFamily` clone (`core/config/msdfFonts.ts:27-31`): any variant
    /// with `msdf_ready && atlas_urls non-empty`.
    pub fn is_msdf_family(&self, family: &str) -> bool {
        self.families
            .get(&family.to_lowercase())
            .map(|vs| vs.iter().any(|v| v.msdf_ready && !v.atlas_urls.is_empty()))
            .unwrap_or(false)
    }
}
