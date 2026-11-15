//! Scene hydration, indexes, sanitizers, resolved-node computation and
//! flattened z-order — ports of:
//! - hydration filter + normalization:  `core/scene/Scene.ts:521-558`
//! - `normalizePages`:                  Scene.ts:678-694
//! - `restoreSceneState` config merge:  Scene.ts:2761-2770
//! - sanitizers:                        `core/scene/scene.sanitizers.ts`
//! - `sanitizeCanvasRootNodeIds`:       Scene.ts:595-628
//! - index builders:                    `core/scene/scene.indexes.ts`
//! - `computeResolvedNodes`:            Scene.ts:3152-3271
//! - flattened z-order selectors:       `core/scene/scene.selectors.ts:58-124`
//! - page positions:                    `core/scene/scene.layout.ts:67-130`
//! - accumulated transforms:            `core/scene/nodeTransformUtils.ts:95-311`

use crate::charts::ticks::TickError;
use crate::expansion::{
    create_rectangle_from_page, expand_generated_graphic_node, expand_table_node,
};
use crate::normalize::normalize_node_in_place;
use crate::transform::{self, compose};
use crate::types::{
    compute_page_bg_id, Affine, NodeId, RawNode, RawPage, SceneConfig, SceneDocument, Transform2D,
    KNOWN_DROPPED_NODE_TYPES, VALID_NODE_TYPES,
};
use crate::value::Value;
/// The scene's id-keyed sets hash with FxHash too (see [`HashMap`]).
pub type HashSet<K> = std::collections::HashSet<K, rustc_hash::FxBuildHasher>;

/// The scene's id-keyed indexes hash with FxHash: they are looked up for every node on every plan build, and
/// std's SipHash was a sixth of a first frame's time in wasm. Keys are the document's own ids (no DoS concern).
pub type HashMap<K, V> = std::collections::HashMap<K, V, rustc_hash::FxBuildHasher>;

/// The ids one ancestor walk has seen (its cycle guard): a list while short, as most chains are, then a set, so a
/// deep chain stays linear.
#[derive(Default)]
pub struct SeenIds<'a> {
    list: Vec<&'a str>,
    set: Option<HashSet<&'a str>>,
}

impl<'a> SeenIds<'a> {
    /// Past this many ids the list becomes a set.
    const LIST_MAX: usize = 16;

    /// Record `id`; false if it was seen already.
    pub fn insert(&mut self, id: &'a str) -> bool {
        if let Some(set) = &mut self.set {
            return set.insert(id);
        }
        if self.list.contains(&id) {
            return false;
        }
        self.list.push(id);
        if self.list.len() > Self::LIST_MAX {
            self.set = Some(self.list.drain(..).collect());
        }
        true
    }
}

#[cfg(test)]
thread_local! {
    /// How many times this thread ran [`Scene::accumulated_affine`] (a full ancestor walk): lets a test check that
    /// a build computes each node's transform once.
    pub(crate) static ACCUMULATED_AFFINE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// A test's reader, run where a write derives from the nodes it wrote ([`Scene::refresh_resolved`]): it sees
    /// what a derivation reads there, between the write and its stamps.
    pub(crate) static DERIVATION_PROBE: std::cell::Cell<Option<fn(&Scene)>> = const { std::cell::Cell::new(None) };
}

// ---------------------------------------------------------------------------
// Document hydration
// ---------------------------------------------------------------------------

/// Parse + hydrate a `canvas_data` JSON document: filter invalid node types,
/// normalize nodes and pages, parse config permissively. The variable tables
/// (`variableCollections`, `variables`, `variableModes`) are not read: the engine
/// renders the cached literals (`variables.rs`). Clones each node and page;
/// [`hydrate_document_owned`] moves them out of a parsed value instead (the feed
/// paths, which own the value they just parsed).
pub fn hydrate_document(canvas_data: &Value) -> SceneDocument {
    let empty = crate::value::Map::new();
    let root = canvas_data.as_object().unwrap_or(&empty);
    let mut doc = SceneDocument::default();
    note_unknown_envelope_keys(&mut doc, root);
    if let Some(Value::Array(nodes)) = root.get("nodes") {
        for v in nodes {
            let Some(node) = v.as_object() else { continue };
            if admit_node_type(&mut doc, node) {
                push_node(&mut doc, node.clone());
            }
        }
    }
    if let Some(Value::Array(pages)) = root.get("pages") {
        for p in pages {
            let Some(obj) = p.as_object() else { continue };
            push_page(&mut doc, obj.clone());
        }
    }
    hydrate_envelope(&mut doc, root);
    doc
}

/// [`hydrate_document`] over an owned value: nodes and pages move into the
/// document instead of being deep-cloned (at 50k nodes the clones were most of
/// a hydrate: ~2.6 s of the asset scan's 3.8 s at open).
pub fn hydrate_document_owned(canvas_data: Value) -> SceneDocument {
    let Value::Object(mut root) = canvas_data else {
        return hydrate_document(&canvas_data);
    };
    let mut doc = SceneDocument::default();
    note_unknown_envelope_keys(&mut doc, &root);
    let nodes = root.remove("nodes");
    let pages = root.remove("pages");
    if let Some(Value::Array(nodes)) = nodes {
        for v in nodes {
            let Value::Object(node) = v else { continue };
            if admit_node_type(&mut doc, &node) {
                push_node(&mut doc, node);
            }
        }
    }
    if let Some(Value::Array(pages)) = pages {
        for p in pages {
            let Value::Object(obj) = p else { continue };
            push_page(&mut doc, obj);
        }
    }
    hydrate_envelope(&mut doc, &root);
    doc
}

/// [`hydrate_document_owned`] of `extract_canvas_data_owned(parse(json))`, straight from the JSON text: each node
/// is hydrated (normalized, its points compacted) as soon as it parses, so a load holds the scene plus one node's
/// JSON rather than the whole document's JSON tree (a 20k-layer vector document's tree was 4.7 GB, past wasm's
/// 4 GB, for a 0.2 GB scene).
pub fn hydrate_document_str(json: &str) -> Result<SceneDocument, crate::value::Error> {
    use serde::de::DeserializeSeed;
    let mut de = serde_json::Deserializer::from_str(json);
    let doc = stream::Root.deserialize(&mut de)?;
    de.end()?;
    Ok(doc)
}

mod stream {
    use super::{admit_node_type, hydrate_document, hydrate_envelope, note_unknown_envelope_keys, push_node, push_page};
    use crate::types::SceneDocument;
    use serde::de::{DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
    use crate::value::Value;
    use std::fmt;

    /// A document (or the `canvas_data` inside one).
    pub struct Root;

    impl<'de> DeserializeSeed<'de> for Root {
        type Value = SceneDocument;
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<SceneDocument, D::Error> {
            d.deserialize_any(RootVisitor)
        }
    }

    struct RootVisitor;

    /// What a document that is not an object hydrates to.
    fn not_an_object() -> SceneDocument {
        hydrate_document(&Value::Null)
    }

    impl<'de> Visitor<'de> for RootVisitor {
        type Value = SceneDocument;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a canvas document")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<SceneDocument, A::Error> {
            let mut doc = SceneDocument::default();
            let mut root = crate::value::Map::new();
            let mut pages: Option<Value> = None;
            let mut inner: Option<SceneDocument> = None;
            while let Some(k) = a.next_key::<String>()? {
                match k.as_str() {
                    "nodes" => {
                        // A repeated key replaces the earlier value, as parsing to a `Value` does.
                        doc.nodes.clear();
                        doc.unknown_node_types.clear();
                        a.next_value_seed(Nodes(&mut doc))?;
                        root.insert(k.into(), Value::Null);
                    }
                    "pages" => {
                        pages = Some(a.next_value()?);
                        root.insert(k.into(), Value::Null);
                    }
                    "canvas_data" => {
                        inner = Some(a.next_value_seed(Root)?);
                        root.insert(k.into(), Value::Null);
                    }
                    _ => {
                        let v: Value = a.next_value()?;
                        root.insert(k.into(), v);
                    }
                }
            }
            if let Some(inner) = inner {
                return Ok(inner);
            }
            note_unknown_envelope_keys(&mut doc, &root);
            root.shift_remove("nodes");
            root.shift_remove("pages");
            if let Some(Value::Array(pages)) = pages {
                for p in pages {
                    let Value::Object(obj) = p else { continue };
                    push_page(&mut doc, obj);
                }
            }
            hydrate_envelope(&mut doc, &root);
            Ok(doc)
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<SceneDocument, A::Error> {
            while a.next_element::<IgnoredAny>()?.is_some() {}
            Ok(not_an_object())
        }
        fn visit_unit<E>(self) -> Result<SceneDocument, E> {
            Ok(not_an_object())
        }
        fn visit_bool<E>(self, _: bool) -> Result<SceneDocument, E> {
            Ok(not_an_object())
        }
        fn visit_i64<E>(self, _: i64) -> Result<SceneDocument, E> {
            Ok(not_an_object())
        }
        fn visit_u64<E>(self, _: u64) -> Result<SceneDocument, E> {
            Ok(not_an_object())
        }
        fn visit_f64<E>(self, _: f64) -> Result<SceneDocument, E> {
            Ok(not_an_object())
        }
        fn visit_str<E>(self, _: &str) -> Result<SceneDocument, E> {
            Ok(not_an_object())
        }
    }

    /// The `nodes` array, each node hydrated as it parses.
    struct Nodes<'a>(&'a mut SceneDocument);

    impl<'de> DeserializeSeed<'de> for Nodes<'_> {
        type Value = ();
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
            d.deserialize_any(self)
        }
    }

    impl<'de> Visitor<'de> for Nodes<'_> {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("the nodes")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<(), A::Error> {
            while let Some(v) = a.next_element::<Value>()? {
                let Value::Object(node) = v else { continue };
                if admit_node_type(self.0, &node) {
                    push_node(self.0, node);
                }
            }
            Ok(())
        }
        // Anything but an array has no nodes.
        fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<(), A::Error> {
            while a.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
            Ok(())
        }
        fn visit_unit<E>(self) -> Result<(), E> {
            Ok(())
        }
        fn visit_bool<E>(self, _: bool) -> Result<(), E> {
            Ok(())
        }
        fn visit_i64<E>(self, _: i64) -> Result<(), E> {
            Ok(())
        }
        fn visit_u64<E>(self, _: u64) -> Result<(), E> {
            Ok(())
        }
        fn visit_f64<E>(self, _: f64) -> Result<(), E> {
            Ok(())
        }
        fn visit_str<E>(self, _: &str) -> Result<(), E> {
            Ok(())
        }
    }
}

/// Unknown envelope keys → manifest signal.
fn note_unknown_envelope_keys(doc: &mut SceneDocument, root: &crate::value::Map) {
    for k in root.keys() {
        if !crate::fields::ENVELOPE_KEYS.contains(&k.as_str()) {
            doc.unknown_envelope_keys.push(k.to_string());
        }
    }
}

/// Nodes: filter (Scene.ts:529-540). False for a type the scene drops (an
/// unknown one is recorded).
fn admit_node_type(doc: &mut SceneDocument, node: &crate::value::Map) -> bool {
    let node_type = node.get("type").and_then(Value::as_str).unwrap_or("");
    if VALID_NODE_TYPES.contains(&node_type) {
        return true;
    }
    if !KNOWN_DROPPED_NODE_TYPES.contains(&node_type) {
        let id = node.get("id").and_then(Value::as_str).unwrap_or("").to_string();
        doc.unknown_node_types.push((id, node_type.to_string()));
    }
    false // silently skipped, like the `nodes` setter in Scene.ts
}

/// Nodes: normalize (Scene.ts:529-540).
fn push_node(doc: &mut SceneDocument, node: crate::value::Map) {
    push_node_with(doc, node, None)
}

/// [`push_node`] for a node whose path points the host already holds compact: they are shared, not parsed.
fn push_node_with(
    doc: &mut SceneDocument,
    mut node: crate::value::Map,
    points: Option<std::sync::Arc<crate::path_points::CompactPoints>>,
) {
    let shared = points.is_some();
    let mut raw = match points {
        Some(points) => RawNode::with_shared_points(node, points),
        // A feed's packed points of canonical shapes become compact points directly (no JSON for each point).
        None => match crate::feed_points::take_feed_points(&mut node) {
            Some(points) => RawNode::with_compact_points(node, points),
            None => RawNode::new(node),
        },
    };
    normalize_node_in_place(&mut raw);
    raw.compact_fields();
    if crate::path_points::compact_on_load() {
        raw.compact();
    } else if shared {
        // Compaction is off (a test's reference run): hold the host's points as the JSON a document would carry.
        raw.decompact();
    }
    doc.nodes.push(raw);
}

/// [`hydrate_document_owned`] for a host that holds its document typed and never writes it as JSON: `envelope` is
/// the document without its `nodes`, and each node is its fields plus, for a path the host holds compact, its
/// points (the node's fields keep `"points": null` in their place). The result is what hydrating the same document
/// from JSON gives.
pub fn hydrate_document_nodes(
    mut envelope: crate::value::Map,
    nodes: impl IntoIterator<Item = (crate::value::Map, Option<std::sync::Arc<crate::path_points::CompactPoints>>)>,
) -> SceneDocument {
    let mut doc = SceneDocument::default();
    note_unknown_envelope_keys(&mut doc, &envelope);
    envelope.shift_remove("nodes");
    let pages = envelope.shift_remove("pages");
    for (node, points) in nodes {
        if admit_node_type(&mut doc, &node) {
            push_node_with(&mut doc, node, points);
        }
    }
    if let Some(Value::Array(pages)) = pages {
        for p in pages {
            let Value::Object(obj) = p else { continue };
            push_page(&mut doc, obj);
        }
    }
    hydrate_envelope(&mut doc, &envelope);
    doc
}

/// Pages: normalizePages (Scene.ts:678-694).
fn push_page(doc: &mut SceneDocument, obj: crate::value::Map) {
    // nodeIds: keep only strings.
    let ids: Vec<String> = obj
        .get("nodeIds")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let mut page = RawPage(obj);
    page.set_node_ids(ids);
    // The background is the page's `fills` as stored (none: a transparent page).
    doc.pages.push(page);
}

/// Everything but the nodes and pages: brushes, config, the canvas root and the
/// layout fields (the variable tables are not read: the engine renders literals).
fn hydrate_envelope(doc: &mut SceneDocument, root: &crate::value::Map) {
    // Brush table (permissive: objects with a string id; everything else is
    // dropped, as the TS normaliser does before it ever reaches a feed).
    if let Some(Value::Array(brushes)) = root.get("brushes") {
        doc.brushes = std::sync::Arc::new(
            brushes
                .iter()
                .filter(|b| b.get("id").and_then(Value::as_str).is_some())
                .cloned()
                .collect(),
        );
        doc.brush_hashes = doc
            .brushes
            .iter()
            .map(|b| {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                b.to_string().hash(&mut h);
                h.finish()
            })
            .collect();
    }

    // Pattern table (E8.5), as permissive as the brush table.
    if let Some(Value::Array(patterns)) = root.get("patterns") {
        doc.patterns = std::sync::Arc::new(
            patterns
                .iter()
                .filter(|p| p.get("id").and_then(Value::as_str).is_some())
                .cloned()
                .collect(),
        );
        doc.pattern_hashes = doc
            .patterns
            .iter()
            .map(|p| {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                p.to_string().hash(&mut h);
                h.finish()
            })
            .collect();
    }

    // Code table (live-nodes.md §3.2): `CodeDef`s by content hash, for generator nodes.
    if let Some(Value::Object(code)) = root.get("code") {
        doc.code = code.clone();
    }

    // Config: renderPageClipping ?? true (DEFAULT_SCENE_CONFIG, scene.config.ts).
    let raw_config = root
        .get("config")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let render_page_clipping = raw_config
        .get("renderPageClipping")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    doc.config = SceneConfig {
        render_page_clipping,
        raw: raw_config,
    };

    let ids = |v: &Value| -> Vec<NodeId> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    doc.canvas_root_node_ids = match root.get("canvasRootNodeIds") {
        Some(list) => ids(list),
        // A Design document keeps its root lists on its pages (`canvases`,
        // docs/roadmap/design-pages.md). One scene draws one page: the one
        // `activeCanvasId` names, else the first.
        None => active_canvas(root)
            .and_then(|c| c.get("rootNodeIds"))
            .map(ids)
            .unwrap_or_default(),
    };
    doc.layout_mode = root
        .get("layoutMode")
        .and_then(Value::as_str)
        .map(str::to_string);
    doc.grid_columns = root.get("gridColumns").and_then(Value::as_f64);
}

/// The page (`canvases[]` entry) a Design document draws: the one
/// `activeCanvasId` names, else the first. None without canvases.
pub fn active_canvas(root: &crate::value::Map) -> Option<&crate::value::Map> {
    let canvases = root.get("canvases")?.as_array()?;
    let wanted = root.get("activeCanvasId").and_then(Value::as_str);
    let by_id = wanted.and_then(|id| {
        canvases
            .iter()
            .find(|c| c.get("id").and_then(Value::as_str) == Some(id))
    });
    by_id.or_else(|| canvases.first())?.as_object()
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

/// Hydrated scene with indexes, sanitized page membership, resolved nodes and
/// transient expansion — the moda-scene equivalent of the TS `Scene` object
/// in its render-ready state.
pub struct Scene {
    /// What changed, by canonical node and column group ([`crate::stamps`]): the one record derived data is
    /// validated against. The contract: every write to this scene advances its clock. The fields of a scene are
    /// private to this crate, so no code outside it can write one without going through a method that marks.
    /// - A write to one canonical node's own fields or to its own derived record, re-deriving nothing that reads
    ///   other nodes, stamps that row (`write_placement`, `set_node_xy`, `node_mut`, `resolved_mut`,
    ///   `re_resolve_node`, `prune_resolved`, and an `apply_node_edits` that re-derives no boolean, painted group,
    ///   envelope, text flow or expansion).
    /// - Every other writer dirties everything (`mark_all`): edits that re-derive records reading other nodes,
    ///   structural and failed edits, font healing (`retry_*`, `refit_auto_tables`), re-expansion, envelopes,
    ///   pages and transients.
    /// - A writer that writes nothing marks nothing.
    /// - Inside this crate, code that writes a field directly or calls [`Scene::store_resolved`] itself must
    ///   mark. Outside it there is no such code: the writers offered ([`Scene::page_mut`],
    ///   [`Scene::resolved_mut`], [`Scene::transient_mut`]) mark before they hand anything out.
    pub(crate) stamps: crate::stamps::Stamps,
    pub(crate) doc: SceneDocument,
    /// Insertion-ordered node ids (mirrors JS `Map` iteration order of
    /// `nodesById`).
    pub(crate) node_order: Vec<NodeId>,
    pub(crate) node_index: HashMap<NodeId, usize>,
    /// group/container parent map (scene.indexes.ts:5-15).
    pub(crate) parent_map: HashMap<NodeId, NodeId>,
    /// pageId/parentId -> ordered children ids (scene.indexes.ts:17-26).
    pub(crate) children_by_parent: HashMap<NodeId, Vec<NodeId>>,
    /// node -> owning page id (scene.indexes.ts:28-42).
    pub(crate) page_id_by_node: HashMap<NodeId, String>,
    /// Derived records by id: ONLY the nodes whose render record differs from
    /// their canonical node — an HTML variable reference resolved to its
    /// fallback, a live geometry
    /// effect applied, a boolean drawn as its result path — plus a transient
    /// whose id collides with a canonical node (the transient wins, as in the
    /// TS `resolved` map). Every other node renders straight from `doc.nodes`
    /// through [`Scene::effective_node`], so a document without variable
    /// references or effects keeps one copy of each node, not two (this map used to hold a
    /// full clone of every node: ~47 % of the resident scene at 50k nodes).
    ///
    /// Invariant, kept by every writer ([`Scene::store_resolved`]): an entry
    /// for a canonical id is present iff it differs from the canonical node,
    /// so a patched scene and a cold load of the same document hold the same
    /// map.
    pub(crate) resolved_by_id: HashMap<NodeId, RawNode>,
    /// Transient nodes (table/generated expansion + page backgrounds).
    pub(crate) transient_by_id: HashMap<NodeId, RawNode>,
    /// parent id (node or page) -> transient child ids in order.
    pub(crate) transient_by_parent: HashMap<NodeId, Vec<NodeId>>,
    /// transient id -> parent id.
    pub(crate) transient_parent_of: HashMap<NodeId, NodeId>,
    /// transient id -> page id.
    pub(crate) transient_page_of: HashMap<NodeId, String>,
    /// Boolean nodes (E7.1): the variable-resolved `boolean` record, kept because
    /// `resolved_by_id` holds the path each one draws as (`boolean_node::result_path_node`).
    pub(crate) boolean_source_by_id: HashMap<NodeId, RawNode>,
    /// Groups with appearance (E8.2, `crate::appearance`): the path each paints its own fills and strokes
    /// on, their merged outline in the group's local space ([`Scene::group_appearance`]).
    pub(crate) group_appearance_by_id: HashMap<NodeId, RawNode>,
    /// Area type (E8.14): the nodes with Text Wrap (`wrapsText`) and the text frames that continue a thread
    /// (`threadPrev`), sorted. Text layout reads other nodes only through these, so a document without either
    /// pays nothing ([`Scene::refresh_text_flow_index`]).
    pub(crate) text_flow: TextFlowIndex,
    /// Envelopes (E8.17, `crate::envelope`): every transient each envelope group draws in place of its children,
    /// keyed by the envelope (present, possibly empty, for every resolved envelope).
    pub(crate) envelope_transients: HashMap<NodeId, Vec<NodeId>>,
    /// Each envelope's warped art as one compound path: what it clips with when it is a clip mask.
    pub(crate) envelope_outline_by_id: HashMap<NodeId, RawNode>,
    /// Envelopes still drawing their content unbent: text inside them waits for its fonts.
    pub(crate) envelope_pending: Vec<NodeId>,
    /// Transients that are clip-mask sources of an envelope's transient groups (never drawn themselves).
    pub(crate) envelope_mask_sources: Vec<NodeId>,
    /// The accumulated transforms by row, derived from the fields above under the stamps rule
    /// ([`crate::worlds`]).
    pub(crate) worlds: crate::worlds::Worlds,
}

/// The column group a write to field `key` of `node` belongs to ([`crate::stamps::Group`]): the one
/// classification every in-place write stamps by ([`Scene::write_row`], [`Scene::write_placement`]).
///
/// - Placement: the node's own transform, and the opacity of a node that draws a body itself.
/// - Content: everything else. A node that holds children composites them by its opacity, so that is content. A
///   `fills` or `strokes` field can change a paint's form or a stroke's geometry as well as a paint's values, so a
///   field write of one is content too; the paint group is for a write that names a value of a paint that keeps
///   its form.
fn field_group(node: &RawNode, key: &str) -> crate::stamps::Group {
    match key {
        "x" | "y" | "rotation" | "scaleX" | "scaleY" | "scale" | "skewX" => crate::stamps::Group::Placement,
        "opacity" if !is_parent_node(node) => crate::stamps::Group::Placement,
        _ => crate::stamps::Group::Content,
    }
}

/// The fields of one node, open for a write in place ([`Scene::write_row`]). It records what was written: a
/// field that already holds its value is not written, and the row is stamped in every group a written field
/// belongs to ([`field_group`]).
pub struct RowFields<'a> {
    node: &'a mut RawNode,
    /// Whether a field of the placement group, and of the content group, was written.
    wrote: [bool; 2],
    /// The write named a field that is structure.
    structure: bool,
}

impl RowFields<'_> {
    /// Set a field of the node. Its id, type, parent and children are not set: naming one refuses the write.
    pub fn set(&mut self, key: &str, value: impl Into<Value>) {
        if matches!(key, "id" | "type" | "parentId" | "children") {
            self.structure = true;
            return;
        }
        let value = value.into();
        if self.node.field_eq_json(key, &value) {
            return;
        }
        self.wrote[usize::from(field_group(self.node, key) != crate::stamps::Group::Placement)] = true;
        self.node.set(key, value);
    }

    /// Replace the node's path geometry with compact points it shares with other nodes.
    pub fn set_shared_points(&mut self, points: std::sync::Arc<crate::path_points::CompactPoints>) {
        if self.node.shared_points().is_some_and(|held| std::sync::Arc::ptr_eq(held, &points) || **held == *points) {
            return;
        }
        self.wrote[1] = true;
        self.node.set_shared_points(points);
    }

    /// The node as written so far.
    pub fn node(&self) -> &RawNode {
        self.node
    }
}

/// What a write to nodes re-derives besides each written node's own record ([`Scene::rederived_by`]).
#[derive(Clone, Copy, Default)]
struct Rederived {
    text_flow: bool,
    booleans: bool,
    envelopes: bool,
    painted_groups: bool,
}

impl Rederived {
    fn any(self) -> bool {
        self.text_flow || self.booleans || self.envelopes || self.painted_groups
    }
    fn or(self, other: Rederived) -> Rederived {
        Rederived {
            text_flow: self.text_flow || other.text_flow,
            booleans: self.booleans || other.booleans,
            envelopes: self.envelopes || other.envelopes,
            painted_groups: self.painted_groups || other.painted_groups,
        }
    }
}

/// See [`Scene::text_flow`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextFlowIndex {
    pub wrap_ids: Vec<NodeId>,
    pub thread_ids: Vec<NodeId>,
}

/// Whether a node takes part in text flow (Text Wrap or a thread link): an edit to it can move other text.
pub fn node_in_text_flow(node: &RawNode) -> bool {
    node.json_field("wrapsText").is_some_and(Value::is_object)
        || node.str("threadPrev").is_some_and(|s| !s.is_empty())
}

/// What [`Scene::reexpand_transient_host`] changed: the transient ids the host
/// had before, and the ones it has now — both in DISPLAY-LIST order, so a
/// caller compares them as SEQUENCES. Equal sequences mean the display list's
/// shape is unchanged and only per-item properties need patching; any
/// difference (length, order, or an id swap) means the host's slice of the list
/// has to be rebuilt.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReexpandOutcome {
    pub removed: Vec<NodeId>,
    pub added: Vec<NodeId>,
}

/// The scene's parts, to read. Writing goes through the methods that stamp.
impl Scene {
    pub fn doc(&self) -> &SceneDocument {
        &self.doc
    }
    pub fn stamps(&self) -> &crate::stamps::Stamps {
        &self.stamps
    }
    /// Record a change to everything. Marking more than was written is always safe; nothing offered outside
    /// this crate can move the clock back or replace the stamps.
    pub fn mark_all(&mut self) -> u64 {
        self.stamps.mark_all()
    }
    /// A dirty list of `group` for one consumer ([`crate::stamps::Stamps::subscribe`]).
    pub fn subscribe(&mut self, group: crate::stamps::Group) -> crate::stamps::Subscription {
        self.stamps.subscribe(group)
    }
    /// The consumer has followed its dirty rows ([`crate::stamps::Stamps::clear`]).
    pub fn clear_dirty(&mut self, subscription: crate::stamps::Subscription) {
        self.stamps.clear(subscription)
    }
    /// This scene replaces `earlier` for a consumer: it takes over `earlier`'s subscriptions (their lists start
    /// empty) and its clock continues past `earlier`'s, so a clock never repeats and everything made from
    /// `earlier` is stale.
    pub fn continue_stamps_from(&mut self, earlier: &Scene) {
        self.stamps.adopt_subscriptions(&earlier.stamps);
        self.stamps.continue_from(&earlier.stamps);
    }
    pub fn node_order(&self) -> &Vec<NodeId> {
        &self.node_order
    }
    pub fn parent_map(&self) -> &HashMap<NodeId, NodeId> {
        &self.parent_map
    }
    pub fn children_by_parent(&self) -> &HashMap<NodeId, Vec<NodeId>> {
        &self.children_by_parent
    }
    pub fn page_id_by_node(&self) -> &HashMap<NodeId, String> {
        &self.page_id_by_node
    }
    pub fn resolved_by_id(&self) -> &HashMap<NodeId, RawNode> {
        &self.resolved_by_id
    }
    pub fn transient_by_id(&self) -> &HashMap<NodeId, RawNode> {
        &self.transient_by_id
    }
    pub fn transient_by_parent(&self) -> &HashMap<NodeId, Vec<NodeId>> {
        &self.transient_by_parent
    }
    pub fn transient_parent_of(&self) -> &HashMap<NodeId, NodeId> {
        &self.transient_parent_of
    }
    pub fn transient_page_of(&self) -> &HashMap<NodeId, String> {
        &self.transient_page_of
    }
    pub fn boolean_source_by_id(&self) -> &HashMap<NodeId, RawNode> {
        &self.boolean_source_by_id
    }
    pub fn group_appearance_by_id(&self) -> &HashMap<NodeId, RawNode> {
        &self.group_appearance_by_id
    }
    pub fn text_flow(&self) -> &TextFlowIndex {
        &self.text_flow
    }
    pub fn envelope_transients(&self) -> &HashMap<NodeId, Vec<NodeId>> {
        &self.envelope_transients
    }
    pub fn envelope_outline_by_id(&self) -> &HashMap<NodeId, RawNode> {
        &self.envelope_outline_by_id
    }
    pub fn envelope_pending(&self) -> &Vec<NodeId> {
        &self.envelope_pending
    }
    pub fn envelope_mask_sources(&self) -> &Vec<NodeId> {
        &self.envelope_mask_sources
    }

    /// A page, to change: no canonical row holds a page, so everything is dirtied before it is handed out (an
    /// unknown page hands nothing out and leaves the clock as it was, as for the two accessors below). The
    /// caller re-derives what the page feeds ([`Scene::reexpand_page_background`]).
    pub fn page_mut(&mut self, id: &str) -> Option<&mut RawPage> {
        let at = self.doc.pages.iter().position(|page| page.id() == id)?;
        self.stamps.mark_all();
        self.doc.pages.get_mut(at)
    }

    /// A node's derived record, to change (only a node whose record differs from its canonical node has one).
    /// Its node's row is stamped before it is handed out, as for [`Scene::node_mut`]: the record is that row's
    /// drawn form. The caller keeps the map's invariant ([`Scene::prune_resolved`]).
    pub fn resolved_mut(&mut self, id: &str) -> Option<&mut RawNode> {
        if !self.resolved_by_id.contains_key(id) {
            return None;
        }
        self.mark_record(id);
        self.resolved_by_id.get_mut(id)
    }

    /// Stamp the row whose derived record is about to be written. A record is keyed by its canonical node, so
    /// the row is known; a record without one (none is made today) dirties everything.
    fn mark_record(&mut self, id: &str) {
        match self.node_index.get(id) {
            Some(&row) => self.stamps.mark_row(row as u32),
            None => self.stamps.mark_all(),
        };
    }

    /// A transient node, to change. Everything is dirtied before it is handed out.
    pub fn transient_mut(&mut self, id: &str) -> Option<&mut RawNode> {
        if !self.transient_by_id.contains_key(id) {
            return None;
        }
        self.stamps.mark_all();
        self.transient_by_id.get_mut(id)
    }

    /// Name `parent` as the transient parent of `child` (tests build small hierarchies this way).
    #[doc(hidden)]
    pub fn set_transient_parent(&mut self, child: &str, parent: &str) {
        self.stamps.mark_all();
        self.transient_parent_of.insert(child.to_string(), parent.to_string());
    }
}

impl Scene {
    pub fn load(canvas_data: &Value) -> Result<Scene, crate::value::Error> {
        Scene::from_document(hydrate_document(canvas_data))
    }

    /// [`Scene::load`] over an owned value (no deep clone of the nodes).
    /// A scene from typed nodes ([`hydrate_document_nodes`]): no JSON document is built or parsed.
    pub fn load_nodes(
        envelope: crate::value::Map,
        nodes: impl IntoIterator<Item = (crate::value::Map, Option<std::sync::Arc<crate::path_points::CompactPoints>>)>,
    ) -> Result<Scene, crate::value::Error> {
        Scene::from_document(hydrate_document_nodes(envelope, nodes))
    }

    pub fn load_owned(canvas_data: Value) -> Result<Scene, crate::value::Error> {
        Scene::from_document(hydrate_document_owned(canvas_data))
    }

    /// `Scene::load_owned(extract_canvas_data_owned(parse(json)))` without the document's JSON tree
    /// ([`hydrate_document_str`]): the feed paths' load.
    pub fn load_str(json: &str) -> Result<Scene, crate::value::Error> {
        Scene::from_document(hydrate_document_str(json)?)
    }

    /// Build a Scene from an already-hydrated document. Used by `load` and by
    /// the animation frame loop (WF6), which clones the hydrated document,
    /// applies per-frame property overlays, and rebuilds the scene so every
    /// downstream index / resolved-node / transient-expansion stays
    /// consistent with the overlaid values.
    pub fn from_document(mut doc: SceneDocument) -> Result<Scene, crate::value::Error> {
        validate_hierarchy(doc.nodes.iter()).map_err(<crate::value::Error as serde::de::Error>::custom)?;
        // Flipbook onion skins (a preview aid): copies of the neighbouring drawings, before any index is built.
        crate::flipbook::expand_onion(&mut doc);
        // -- indexes + sanitizers (Scene.ts:2772-2780 order) --
        let mut node_order: Vec<NodeId> = Vec::with_capacity(doc.nodes.len());
        let mut node_index: HashMap<NodeId, usize> = HashMap::default();
        for (i, n) in doc.nodes.iter().enumerate() {
            // JS Map.set: later duplicate ids overwrite earlier — keep last,
            // preserving FIRST insertion position (JS Map semantics).
            if let Some(&existing) = node_index.get(n.id()) {
                let _ = existing;
            } else {
                node_order.push(n.id().to_string());
            }
            node_index.insert(n.id().to_string(), i);
        }

        let parent_map = build_parent_map(&doc.nodes);
        sanitize_unique_page_membership(&mut doc.pages, &parent_map);
        sanitize_top_level_lists(&mut doc.pages, &doc.nodes, &node_index);
        let page_id_by_node = build_page_index(&doc.pages, &doc.nodes, &node_index);
        sanitize_canvas_root_node_ids(
            &mut doc.canvas_root_node_ids,
            &node_index,
            &parent_map,
            &page_id_by_node,
        );
        let children_by_parent = build_children_index(&doc.pages, &doc.nodes);

        let mut scene = Scene {
            stamps: crate::stamps::Stamps::new(doc.nodes.len(), 0),
            doc,
            node_order,
            node_index,
            parent_map,
            children_by_parent,
            page_id_by_node,
            resolved_by_id: HashMap::default(),
            transient_by_id: HashMap::default(),
            transient_by_parent: HashMap::default(),
            transient_parent_of: HashMap::default(),
            transient_page_of: HashMap::default(),
            boolean_source_by_id: HashMap::default(),
            group_appearance_by_id: HashMap::default(),
            text_flow: TextFlowIndex::default(),
            envelope_transients: HashMap::default(),
            envelope_outline_by_id: HashMap::default(),
            envelope_pending: Vec::new(),
            envelope_mask_sources: Vec::new(),
            worlds: Default::default(),
        };
        // The scene's own derived columns follow its writes, as any consumer does; they subscribe first.
        scene.worlds.subscribe(&mut scene.stamps);
        // Text flow first: a warped text layer's outlines (derived below, `crate::text_warp`) lay out in its wrap
        // objects and thread frames.
        scene.refresh_text_flow_index();
        scene.compute_resolved_nodes()?;
        scene.compute_boolean_nodes();
        scene.rederive_flowing_text_warps();
        scene.compute_envelopes();
        // After the envelopes: a group's merged outline reads its children as they draw.
        scene.compute_group_appearances();
        Ok(scene)
    }

    /// The path a group with appearance paints its own fills and strokes on (its merged outline, in the
    /// group's local space, with the group's `fills`, `strokes` and `appearanceOrder`), or `None` for a group
    /// without appearance and every other node.
    pub fn group_appearance(&self, id: &str) -> Option<&RawNode> {
        self.group_appearance_by_id.get(id)
    }

    /// Re-derive [`Scene::group_appearance_by_id`] from the groups that paint (an O(nodes) scan, run at load
    /// and by an edit that touches a group with appearance or what feeds one).
    fn compute_group_appearances(&mut self) {
        let ids: Vec<NodeId> = self
            .node_order
            .iter()
            .filter(|id| self.node(id).is_some_and(crate::appearance::group_has_appearance))
            .cloned()
            .collect();
        let mut out = HashMap::default();
        for id in ids {
            let Some(group) = self.effective_node(&id) else {
                continue;
            };
            let result = crate::boolean_node::group_merged_outline(self, group);
            out.insert(id, group_appearance_node(group, &result));
        }
        self.group_appearance_by_id = out;
    }

    /// Whether `id` sits inside a group with appearance (its geometry feeds that group's merged outline).
    pub fn has_group_appearance_ancestor(&self, id: &str) -> bool {
        if self.group_appearance_by_id.is_empty() {
            return false;
        }
        let mut current = id;
        let mut steps = 0;
        while let Some(p) = self.parent_map.get(current) {
            if self.group_appearance_by_id.contains_key(p) {
                return true;
            }
            current = p;
            steps += 1;
            if steps > 256 {
                break;
            }
        }
        false
    }

    /// Whether the scene has, or may now have, envelopes to resolve.
    fn has_envelopes(&self) -> bool {
        !self.envelope_transients.is_empty() || !self.envelope_pending.is_empty()
    }

    /// Drop every envelope's transients and records.
    fn clear_envelopes(&mut self) -> Vec<NodeId> {
        let mut removed = Vec::new();
        let old = std::mem::take(&mut self.envelope_transients);
        let mut envs: Vec<&NodeId> = old.keys().collect();
        envs.sort();
        for env in envs {
            for id in &old[env] {
                self.transient_by_id.remove(id);
                self.transient_parent_of.remove(id);
                self.transient_page_of.remove(id);
                self.transient_by_parent.remove(id);
                if self.node(id).is_none() {
                    self.resolved_by_id.remove(id);
                }
                removed.push(id.clone());
            }
            self.transient_by_parent.remove(env);
        }
        self.envelope_outline_by_id.clear();
        self.envelope_pending.clear();
        self.envelope_mask_sources.clear();
        removed
    }

    /// Resolve every envelope group (E8.17, `crate::envelope`), innermost first, replacing what they drew before.
    /// A document without envelopes pays one scan of its nodes. Returns the envelope and transient ids now drawn.
    pub fn compute_envelopes(&mut self) -> Vec<NodeId> {
        // With no envelope now and none to resolve, nothing is written: the clock stands. Otherwise derived
        // records are rewritten and no row can be attributed.
        if self.has_envelopes() || !crate::envelope::envelope_ids(self).is_empty() {
            self.stamps.mark_all();
        }
        if self.has_envelopes() {
            self.clear_envelopes();
        }
        let mut touched = Vec::new();
        for env in crate::envelope::envelope_ids(self) {
            // A repeat (`crate::repeat`) resolves in the same pass: its copies are transients drawn in its place.
            let r = if self.node(&env).is_some_and(crate::envelope::is_envelope) {
                crate::envelope::resolve(self, &env)
            } else {
                crate::repeat::resolve(self, &env)
            };
            if r.pending {
                self.envelope_pending.push(env.clone());
                if r.transients.is_empty() {
                    // Nothing bent yet (its text waits for fonts): it draws its content as it is for now.
                    continue;
                }
            }
            let page_id = self.page_of(&env);
            let mut order: Vec<NodeId> = Vec::new();
            let mut by_parent: HashMap<NodeId, Vec<RawNode>> = HashMap::default();
            let mut ids = Vec::with_capacity(r.transients.len());
            for (parent, n) in r.transients {
                if !by_parent.contains_key(&parent) {
                    order.push(parent.clone());
                }
                ids.push(n.id().to_string());
                by_parent.entry(parent).or_default().push(n);
            }
            for parent in order {
                let nodes = by_parent.remove(&parent).unwrap_or_default();
                self.add_transients(nodes, &parent, page_id.as_deref());
            }
            if let Some(o) = r.outline {
                self.envelope_outline_by_id.insert(env.clone(), o);
            }
            self.envelope_mask_sources.extend(r.mask_sources);
            touched.push(env.clone());
            touched.extend(ids.iter().cloned());
            self.envelope_transients.insert(env, ids);
        }
        touched
    }

    /// Re-resolve the envelopes waiting for fonts (after a font install); returns the ids that changed when any
    /// did (see [`Scene::retry_pending_text_warps`]).
    pub fn retry_pending_envelopes(&mut self) -> Vec<NodeId> {
        // Nothing pending writes nothing; `compute_envelopes` marks what it rewrites.
        if self.envelope_pending.is_empty() {
            return Vec::new();
        }
        let before = self.envelope_pending.clone();
        let touched = self.compute_envelopes();
        if self.envelope_pending == before && touched.iter().all(|id| !before.contains(id)) {
            // Still waiting on the same fonts.
            return Vec::new();
        }
        touched
    }

    /// Whether `id` draws through an envelope or a repeat: the group, anything inside one, or one of its transients.
    pub fn in_envelope(&self, id: &str) -> bool {
        if self.envelope_transients.contains_key(id) || self.envelope_pending.iter().any(|e| e == id) {
            return true;
        }
        if self
            .transient_by_id
            .get(id)
            .is_some_and(|n| n.has(crate::envelope::ENVELOPE_OF_KEY) || n.has(crate::repeat::REPEAT_OF_KEY))
        {
            return true;
        }
        let mut cur = id;
        let mut steps = 0;
        while let Some(p) = self.parent_map.get(cur) {
            if self.node(p).is_some_and(|n| crate::envelope::is_envelope(n) || crate::repeat::is_repeat(n)) {
                return true;
            }
            cur = p;
            steps += 1;
            if steps > 512 {
                break;
            }
        }
        self.node(id).is_some_and(|n| crate::envelope::is_envelope(n) || crate::repeat::is_repeat(n))
    }

    /// The compound path an envelope clips with when it is a clip mask (its warped art).
    pub fn envelope_mask_record(&self, id: &str) -> Option<&RawNode> {
        self.envelope_outline_by_id.get(id)
    }

    /// Warped text that lays out in the text flow (Text Wrap objects, threads) reads the OTHER nodes' drawn records
    /// (a wrap object's effected or boolean outline, an earlier thread frame), which may not have been derived yet
    /// when the text was: re-derive it once every other record is in. Only text with a geometry effect, and only
    /// when the document has text flow.
    fn rederive_flowing_text_warps(&mut self) {
        if self.text_flow.wrap_ids.is_empty() && self.text_flow.thread_ids.is_empty() {
            return;
        }
        let ids: Vec<NodeId> = self
            .node_order
            .iter()
            .filter(|id| self.node(id).is_some_and(crate::text_warp::is_effected_text))
            .cloned()
            .collect();
        for id in ids {
            self.re_resolve_node(&id);
        }
    }

    /// Rebuild [`Scene::text_flow`] from the nodes (an O(nodes) scan; edits call it only when a node in text flow
    /// changed).
    pub fn refresh_text_flow_index(&mut self) {
        let mut wrap_ids = Vec::new();
        let mut thread_ids = Vec::new();
        for n in &self.doc.nodes {
            if n.json_field("wrapsText").is_some_and(Value::is_object) {
                wrap_ids.push(n.id().to_string());
            }
            if n.str("threadPrev").is_some_and(|s| !s.is_empty()) {
                thread_ids.push(n.id().to_string());
            }
        }
        wrap_ids.sort();
        wrap_ids.dedup();
        thread_ids.sort();
        thread_ids.dedup();
        let index = TextFlowIndex { wrap_ids, thread_ids };
        if index != self.text_flow {
            // Text layout reads other nodes through this index: no row can be attributed. (The same index
            // again writes nothing.)
            self.stamps.mark_all();
            self.text_flow = index;
        }
    }

    /// Whether `id` is in the text-flow index.
    fn in_text_flow_index(&self, id: &str) -> bool {
        self.text_flow.wrap_ids.binary_search_by(|x| x.as_str().cmp(id)).is_ok()
            || self.text_flow.thread_ids.binary_search_by(|x| x.as_str().cmp(id)).is_ok()
    }

    pub fn node(&self, id: &str) -> Option<&RawNode> {
        self.node_index.get(id).map(|&i| &self.doc.nodes[i])
    }

    /// Task #95 S2-min: mutable canonical-node access for the retained-mode
    /// overlay patch lane (`residency::ResidentScene::apply_props`). Callers
    /// own index/hash consistency — the patch lane re-hashes patched ids and
    /// rejects structural keys, so every index stays valid.
    pub fn node_mut(&mut self, id: &str) -> Option<&mut RawNode> {
        let i = *self.node_index.get(id)?;
        // The caller may write any field: placement, paint and content are all stamped.
        self.stamps.mark_row(i as u32);
        self.doc.nodes.get_mut(i)
    }

    /// Live-drag fast path: set a node's parent-relative position on BOTH the
    /// canonical doc node and its resolved entry (`effective_node` prefers
    /// `resolved_by_id`), so the next per-frame plan rebuild reflects the move
    /// without a full scene reload. Position-only — geometry/structure untouched.
    pub fn set_node_xy(&mut self, id: &str, x: f64, y: f64) {
        // Held means the position the transform reads ([`Scene::accumulated_affine`]), however it is spelled: an
        // authored integer `4` holds `4.0`, and a coordinate left out holds zero.
        let at = (x, y);
        let holds = move |n: &RawNode| n.f64("x").unwrap_or(0.0) == at.0 && n.f64("y").unwrap_or(0.0) == at.1;
        let (x, y) = (crate::value::json!(x), crate::value::json!(y));
        if let Some(&i) = self.node_index.get(id) {
            // Already there: nothing is written and the clock stands.
            if self.doc.nodes.get(i).is_some_and(|n| !holds(n)) {
                self.stamps.mark(i as u32, crate::stamps::Group::Placement);
                if let Some(n) = self.doc.nodes.get_mut(i) {
                    n.set("x", x.clone());
                    n.set("y", y.clone());
                }
            }
        }
        if self.resolved_by_id.get(id).is_some_and(|r| !holds(r)) {
            // The node's derived record moves with it: the same row's placement.
            match self.node_index.get(id) {
                Some(&row) => self.stamps.mark(row as u32, crate::stamps::Group::Placement),
                None => self.stamps.mark_all(),
            };
            if let Some(r) = self.resolved_by_id.get_mut(id) {
                r.set("x", x);
                r.set("y", y);
            }
        }
    }

    /// The row of the canonical node named `id`: its index in `doc.nodes`, the integer everything at runtime
    /// names it by. Names are a side map read at install and by questions; a row stays the node's until an edit
    /// reports `structural` (or the scene is replaced).
    pub fn row_of(&self, id: &str) -> Option<u32> {
        self.node_index.get(id).map(|&row| row as u32)
    }

    /// The canonical node at `row`.
    pub fn row(&self, row: u32) -> Option<&RawNode> {
        self.doc.nodes.get(row as usize)
    }

    /// Whether the node at `row` is plain: a path, group or text node that is its own resolved record and feeds no
    /// derived record (expansions, booleans, group appearance, envelopes, repeats, text flow). Writing a plain
    /// row's own fields in place ([`Scene::write_placement`], [`Scene::write_row`]) leaves the scene
    /// [`Scene::apply_node_edits`] of the same fields leaves. The answer stands while nothing dirties everything
    /// ([`crate::stamps::Stamps::all`]) for rows written through [`Scene::write_placement`] and
    /// [`Scene::write_row`]: a row write that changes the answer for its row dirties everything. A whole-node
    /// edit ([`Scene::apply_node_edits`], [`Scene::node_mut`]) can change it with a content stamp alone; a caller
    /// that mixes those with row writes asks again after them.
    pub fn placeable(&self, row: u32) -> bool {
        let Some(node) = self.doc.nodes.get(row as usize) else { return false };
        let id = node.id();
        matches!(node.node_type(), "path" | "group" | "richtext")
            && self.node_index.get(id) == Some(&(row as usize))
            && !self.resolved_by_id.contains_key(id)
            && !self.transient_by_parent.contains_key(id)
            && !self.transient_parent_of.contains_key(id)
            && self.boolean_source_by_id.is_empty()
            && self.group_appearance_by_id.is_empty()
            && !crate::appearance::group_has_appearance(node)
            && !self.has_envelopes()
            && !crate::envelope::is_envelope(node)
            && !crate::repeat::is_repeat(node)
            && !node_in_text_flow(node)
            && !self.in_text_flow_index(id)
    }

    /// Write the placement of the plain node at `row` ([`Scene::placeable`]) as typed numbers: its local
    /// transform, its opacity, or both. No JSON edit, no normalization pass and no re-derivation: for these nodes
    /// those leave the same scene.
    ///
    /// A number the row already holds is not written, and a write that changes nothing leaves the clock alone:
    /// the same rule as a field write ([`RowFields::set`]). The row is stamped as a placement when its transform
    /// changed; a changed opacity is stamped as a field write of it is ([`field_group`]: placement for a node
    /// that draws a body itself, content for one that holds children). Returns whether anything changed.
    pub fn write_placement(&mut self, row: u32, transform: Option<NodePlacement>, opacity: Option<crate::value::Number>) -> bool {
        let Some(node) = self.doc.nodes.get_mut(row as usize) else { return false };
        let mut moved = false;
        if let Some(t) = transform {
            for (key, number) in [("x", t.x), ("y", t.y), ("rotation", t.rotation), ("scaleX", t.scale_x), ("scaleY", t.scale_y)] {
                moved |= node.set_placement_number(key, number);
            }
        }
        let faded = opacity.is_some_and(|opacity| node.set_placement_number("opacity", opacity));
        let opacity_group = field_group(node, "opacity");
        if moved || (faded && opacity_group == crate::stamps::Group::Placement) {
            self.stamps.mark(row, crate::stamps::Group::Placement);
        }
        if faded && opacity_group == crate::stamps::Group::Content {
            self.stamps.mark(row, crate::stamps::Group::Content);
        }
        moved || faded
    }

    /// Write fields of the node at `row` in place: `write` sets them on the node itself ([`RowFields`]). A node's
    /// id, type, parent and children are structure, not fields of a row write ([`Scene::apply_node_edits`] changes
    /// those): a write that names one is refused. Returns whether anything was written: a write that sets no
    /// field to a new value writes nothing and leaves the clock alone.
    ///
    /// There is one path for every row. The node is written where it stands, normalized as an edit normalizes
    /// it, and what is derived from it is derived again by the derivation an edit runs
    /// ([`Scene::derive_written`]). The scene after equals `apply_node_edits` of the node with those fields; no
    /// copy of the node is made and no edit batch is built. (The derived records themselves are still kept by
    /// name: re-deriving one reads the node's name from its row and looks it up in those maps.)
    ///
    /// The row is stamped in each group a written field belongs to ([`field_group`]) when the row is plain before and after
    /// ([`Scene::placeable`]) and nothing else was derived from it. Otherwise everything is dirty: other records
    /// changed, or what `placeable` answers for the row did. An error can leave fields partly written;
    /// everything is dirty then too.
    pub fn write_row(&mut self, row: u32, write: impl FnOnce(&mut RowFields<'_>)) -> Result<bool, String> {
        const STRUCTURE: &str = "a row write names the node's id, type, parent or children";
        let index = row as usize;
        if index >= self.doc.nodes.len() {
            return Err(format!("no node at row {row}"));
        }
        let plain = self.placeable(row);
        let node = &mut self.doc.nodes[index];
        let mut fields = RowFields { node: &mut *node, wrote: [false; 2], structure: false };
        write(&mut fields);
        let (wrote, structure) = (fields.wrote, fields.structure);
        if wrote == [false; 2] {
            return if structure { Err(STRUCTURE.into()) } else { Ok(false) };
        }
        crate::feed_points::resolve_node_feed_points(node);
        normalize_node_in_place(node);
        if crate::path_points::compact_on_load() {
            node.compact();
        }
        let node = &self.doc.nodes[index];
        let rederived = self.rederived_by(node.id(), Some(node));
        let mut out = NodeEditOutcome::default();
        let derived = self.derive_written(&[row], rederived, &mut out, &mut HashMap::default());
        let row_alone = !structure && derived.is_ok() && plain && self.placeable(row) && !rederived.any()
            && out.transients_removed.is_empty() && out.transients_added.is_empty();
        if row_alone {
            // Each group a written field belongs to: the groups are independent, and a consumer of one must
            // see a write to it whatever else was written with it.
            for (group, wrote) in [crate::stamps::Group::Placement, crate::stamps::Group::Content].into_iter().zip(wrote) {
                if wrote {
                    self.stamps.mark(row, group);
                }
            }
        } else {
            self.stamps.mark_all();
        }
        if structure {
            return Err(STRUCTURE.into());
        }
        derived.map(|()| true)
    }

    /// What writing the node named `id` re-derives besides its own record: asked of the node as it will stand
    /// (`None` for a removal), against the indexes as they stand before the write (they still know a removed
    /// node's parents). A boolean draws a path derived from its descendants and a group with appearance an
    /// outline of them; envelopes draw transients derived from everything inside them, so any write re-resolves
    /// them when the document has one or gains one; text flow (Text Wrap, threads) is indexed.
    fn rederived_by(&self, id: &str, node: Option<&RawNode>) -> Rederived {
        Rederived {
            text_flow: node.is_some_and(node_in_text_flow) || self.in_text_flow_index(id),
            booleans: node.is_some_and(|n| n.node_type() == "boolean")
                || (!self.boolean_source_by_id.is_empty()
                    && (self.boolean_source_by_id.contains_key(id) || self.has_boolean_ancestor(id))),
            envelopes: self.has_envelopes()
                || node.is_some_and(|n| crate::envelope::is_envelope(n) || crate::repeat::is_repeat(n)),
            painted_groups: node.is_some_and(crate::appearance::group_has_appearance)
                || (!self.group_appearance_by_id.is_empty()
                    && (self.group_appearance_by_id.contains_key(id) || self.has_group_appearance_ancestor(id))),
        }
    }

    /// Derive again what is derived from the canonical nodes at `rows`, which were just written where they
    /// stand: the one derivation after a write, for a batch of whole-node edits and for a row write alike. In
    /// hydration's order: the text-flow index (a warped text layer lays out in the text flow), each node's
    /// resolved record and expansion (`prepared` holds the ones a caller derived before writing), then booleans,
    /// envelopes and painted groups.
    fn derive_written(
        &mut self,
        rows: &[u32],
        rederived: Rederived,
        out: &mut NodeEditOutcome,
        prepared: &mut HashMap<String, (Option<RawNode>, Option<Vec<RawNode>>)>,
    ) -> Result<(), String> {
        if rows.is_empty() && !rederived.any() {
            return Ok(());
        }
        // The nodes are written and not yet stamped (the caller stamps what this reports). The derivations below
        // read the scene as written: advance the clock first, so that a derived column refreshed before the
        // write (the accumulated transforms) answers none of them with what a node held before it.
        self.stamps.begin();
        if rederived.text_flow {
            self.refresh_text_flow_index();
        }
        for &row in rows {
            self.refresh_resolved(row, out, prepared)?;
        }
        if rederived.booleans {
            self.compute_boolean_nodes();
        }
        if rederived.envelopes {
            self.envelope_edit(out);
        }
        if rederived.painted_groups {
            self.compute_group_appearances();
        }
        Ok(())
    }

    /// Write `id`'s derived record, keeping the [`Scene::resolved_by_id`]
    /// invariant: `None`, or a record equal to the canonical node, removes the
    /// entry (the node then renders from `doc.nodes`).
    /// Returns whether the stored record changed.
    pub(crate) fn store_resolved(&mut self, id: &str, record: Option<RawNode>) -> bool {
        let record = record.filter(|r| self.node(id) != Some(r));
        if self.resolved_by_id.get(id) == record.as_ref() {
            return false;
        }
        match record {
            Some(r) => {
                self.resolved_by_id.insert(id.to_string(), r);
            }
            None => {
                self.resolved_by_id.remove(id);
            }
        }
        true
    }

    /// Drop `id`'s derived record when an in-place merge made it equal to the
    /// canonical node again (the retained patch lane merges a prop patch into
    /// both), so the map matches what a cold load builds.
    pub fn prune_resolved(&mut self, id: &str) {
        if self
            .resolved_by_id
            .get(id)
            .is_some_and(|r| self.node(id) == Some(r))
        {
            // The node's derived record goes: its row. (Nothing to prune writes nothing.)
            self.mark_record(id);
            self.resolved_by_id.remove(id);
        }
    }

    /// The HTML variable fallbacks alone: `None` when `node` references no
    /// variable (the resolved record would equal the canonical one, so it isn't
    /// stored).
    fn variable_resolved(&self, node: &RawNode) -> Option<RawNode> {
        crate::variables::resolve_node_if_referenced(node).filter(|r| r != node)
    }

    /// `node`'s derived record — HTML variable fallbacks, then live geometry
    /// effects — and,
    /// for an expanding host, its transient children (expanded from the
    /// variable-resolved record). `None` for the record means the node renders
    /// as its canonical self.
    fn derive(&self, node: &RawNode) -> (Option<RawNode>, Result<Option<Vec<RawNode>>, TickError>) {
        self.derive_with(node, |base| {
            let expanded = match base.node_type() {
                "table" => Some(expand_table_node(base)),
                "generated" => Some(expand_generated_graphic_node(base)?),
                // A nodes-mode generator draws its stored children: nothing to evaluate, unless it is drawn at a time
                // inside a composition (`_compositionTime`, moda-render's graph lane) or handed a timed output
                // (`_generatorOutput`, the editor's playback): then the timed output replaces the stored design.
                "generator"
                    if !matches!(base.json_field("children"), Some(Value::Array(_)))
                        || base.json_field("_compositionTime").is_some()
                        || base.json_field("_generatorOutput").is_some() =>
                {
                    Some(crate::generator::expand_generator_node(base, &self.doc))
                }
                _ => None,
            };
            let expanded = expanded.map(|children| inherit_blend_mode(base, children));
            Ok(expanded)
        })
    }

    fn derive_record(&self, node: &RawNode) -> Option<RawNode> {
        self.derive_with(node, |_| ()).0
    }

    fn derive_with<T>(&self, node: &RawNode, expand: impl FnOnce(&RawNode) -> T) -> (Option<RawNode>, T) {
        let vars = self.variable_resolved(node);
        let fitted = vars.as_ref().unwrap_or(node);
        let fitted = if fitted.node_type() == "table" { crate::expansion::fit_table_rows(fitted) } else { None };
        let base = fitted.as_ref().or(vars.as_ref()).unwrap_or(node);
        let expanded = expand(base);
        // Live geometry effects (E8.1): after variables, a shape with a
        // visible geometry effect resolves to its effected path.
        // A text layer's effects reshape its glyph outlines (`crate::text_warp`).
        let effected = match crate::text_warp::resolve(self, base) {
            crate::text_warp::TextWarp::Resolved(path) => Some(path),
            crate::text_warp::TextWarp::Pending => None,
            crate::text_warp::TextWarp::None => crate::live_effects::resolve_node_in(base, &self.doc),
        };
        (effected.or(fitted).or(vars), expanded)
    }

    /// A boolean node's own (variable-resolved) record — `effective_node` answers the path it
    /// draws as. `None` for any other node.
    pub fn pre_boolean_record(&self, id: &str) -> Option<&RawNode> {
        self.boolean_source_by_id.get(id)
    }

    /// Substitute every boolean node's resolved record with the path of its result (E7.1,
    /// `crate::boolean_node`). Runs after variable resolution; results come from the
    /// per-node cache unless a descendant's geometry changed.
    ///
    /// Also the patch lane's ([`Scene::apply_node_edits`]) way to bring them up to date, so each
    /// source record is re-derived from the canonical node rather than read back from
    /// `resolved_by_id`, which by then holds the previous result's path.
    fn compute_boolean_nodes(&mut self) {
        let ids: Vec<NodeId> = self
            .node_order
            .iter()
            .filter(|id| self.node(id).is_some_and(|n| n.node_type() == "boolean"))
            .cloned()
            .collect();
        self.boolean_source_by_id.retain(|id, _| ids.contains(id));
        if ids.is_empty() {
            return;
        }
        for id in &ids {
            let Some(node) = self.node(id) else {
                continue;
            };
            let record = self.derive_record(node);
            let source = record.unwrap_or_else(|| node.clone());
            self.boolean_source_by_id.insert(id.clone(), source);
        }
        let mut drawn = Vec::with_capacity(ids.len());
        for id in &ids {
            let Some(source) = self.boolean_source_by_id.get(id) else {
                continue;
            };
            let result = crate::boolean_node::boolean_result(self, source);
            drawn.push((
                id.clone(),
                crate::boolean_node::result_path_node(source, &result),
            ));
        }
        for (id, node) in drawn {
            self.resolved_by_id.insert(id, node);
        }
    }

    /// Whether `id` is inside a boolean node (its geometry feeds that node's result).
    pub fn has_boolean_ancestor(&self, id: &str) -> bool {
        let mut current = id;
        let mut steps = 0;
        while let Some(p) = self.parent_map.get(current) {
            if self.node(p).is_some_and(|n| n.node_type() == "boolean") {
                return true;
            }
            current = p;
            steps += 1;
            if steps > 256 {
                break;
            }
        }
        false
    }

    /// Re-derive ONE leaf node's resolved record from its canonical node:
    /// HTML variable fallbacks, then live geometry effects (as [`Scene::load`] does). The
    /// incremental patch lane calls it after merging a patch into a node that
    /// has (or had) geometry effects, whose resolved `points` are derived and
    /// would otherwise go stale. Returns `false` (mutating nothing) for an
    /// unknown id.
    pub fn re_resolve_node(&mut self, id: &str) -> bool {
        if self.node(id).is_none() {
            return false;
        }
        let Some(node) = self.node(id) else {
            return false;
        };
        let record = self.derive_record(node);
        // The record is the node's own drawn form: its row is stamped, and only when the record changed.
        if self.store_resolved(id, record) {
            self.mark_record(id);
        }
        true
    }

    /// Text layers whose geometry effect is waiting for their fonts: effected text (`crate::text_warp`) still
    /// drawing as plain text because its outlines could not be made yet.
    pub fn pending_text_warps(&self) -> Vec<NodeId> {
        self.node_order
            .iter()
            .filter(|id| {
                self.node(id).is_some_and(crate::text_warp::is_effected_text)
                    && self
                        .effective_node(id)
                        .is_some_and(|n| n.node_type() == "richtext")
            })
            .cloned()
            .collect()
    }

    /// Re-resolve every [`Scene::pending_text_warps`] node (after a font install) and return the ones that now
    /// draw warped.
    pub fn retry_pending_text_warps(&mut self) -> Vec<NodeId> {
        // `re_resolve_node` and the envelope pass mark what they rewrite; with nothing pending the clock stands.
        let mut done = Vec::new();
        for id in self.pending_text_warps() {
            self.re_resolve_node(&id);
            if self
                .effective_node(&id)
                .is_some_and(|n| n.node_type() != "richtext")
            {
                done.push(id);
            }
        }
        // Envelopes bend text too: re-resolve them when a warped text healed or their own text waited.
        if !done.is_empty() && self.has_envelopes() {
            done.extend(self.compute_envelopes());
        } else {
            done.extend(self.retry_pending_envelopes());
        }
        done
    }

    /// Re-solve every boolean that cuts text (after a font install: its text operand was outlined with the faces
    /// installed then, or not at all) and return their ids. The others keep their cached results.
    pub fn retry_text_booleans(&mut self) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> = self
            .boolean_source_by_id
            .iter()
            .filter(|(_, source)| crate::boolean_node::has_text_operand(self, source))
            .map(|(id, _)| id.clone())
            .collect();
        ids.sort();
        if !ids.is_empty() {
            // Boolean results are solved again: no row can be attributed.
            self.stamps.mark_all();
            self.compute_boolean_nodes();
        }
        ids
    }

    /// resolved ?? canonical ?? transient (displayList.ts:112).
    pub fn effective_node(&self, id: &str) -> Option<&RawNode> {
        self.resolved_by_id
            .get(id)
            .or_else(|| self.node(id))
            .or_else(|| self.transient_by_id.get(id))
    }

    /// canonical ?? transient (selectors read this — scene.selectors.ts:67).
    pub fn canonical_or_transient(&self, id: &str) -> Option<&RawNode> {
        self.node(id).or_else(|| self.transient_by_id.get(id))
    }

    fn add_transients(&mut self, nodes: Vec<RawNode>, parent_id: &str, page_id: Option<&str>) {
        let mut ids = Vec::with_capacity(nodes.len());
        for n in nodes {
            let id = n.id().to_string();
            // `effective_node` reads resolved ?? canonical ?? transient, so a
            // transient needs a `resolved_by_id` mirror only to win over a
            // canonical node of the same id.
            if self.node(&id).is_some() {
                self.resolved_by_id.insert(id.clone(), n.clone());
            }
            self.transient_by_id.insert(id.clone(), n);
            self.transient_parent_of
                .insert(id.clone(), parent_id.to_string());
            if let Some(p) = page_id {
                self.transient_page_of.insert(id.clone(), p.to_string());
            }
            ids.push(id);
        }
        self.transient_by_parent.insert(parent_id.to_string(), ids);
    }

    /// Refit auto tables after a font install. Canonical records stay authored.
    pub fn refit_auto_tables(&mut self) -> Result<Vec<String>, TickError> {
        // `reexpand_transient_host` marks each host it re-expands.
        let ids: Vec<String> = self.doc.nodes.iter().filter(|n| n.node_type() == "table" && n.str("rowSizing") != Some("fixed")).map(|n| n.id().to_string()).collect();
        let mut changed = Vec::new();
        for id in ids {
            let before = self.effective_node(&id).cloned();
            self.reexpand_transient_host(&id)?;
            if self.effective_node(&id) != before.as_ref() { changed.push(id); }
        }
        Ok(changed)
    }

    /// Re-derive the transient children of ONE expanding host (a `table` or a
    /// `generated` node), in place, without touching the rest of the scene.
    ///
    /// **Why this exists.** Table cell / chart part geometry is *baked* at
    /// expansion time — cell coordinates fold in the host's `x`/`y`
    /// (`expansion::expand_table_node`), row/column sizes decide every border
    /// line, and chart parts are generated from `generatedSource`. So the
    /// incremental patch lane cannot just overlay a property onto the host's
    /// resolved record the way it does for a leaf node: the derived children
    /// would go stale. Before this, ANY property patch on a transient host
    /// forced a whole-scene JSON re-parse plus a cold plan rebuild (~150 ms on
    /// a 10k-cell table). This re-expands just that host instead, and reports
    /// which transient ids went away and which appeared so the caller can tell
    /// a *shape-preserving* edit (same id sequence → patch the existing display
    /// items) from a *structural* one (sequence changed → rebuild the list).
    ///
    /// **No ancestor context is needed.** Expansion is a pure function of the
    /// resolved host node alone — it never reads a parent. Group/container
    /// opacity and transform are applied at PAINT time by
    /// [`Scene::ancestor_opacity_product`] and
    /// [`Scene::transform_parent_chain`], both of which step *through* a
    /// transient host. A host nested in a translucent, translated group
    /// therefore re-expands to byte-identical children regardless of where it
    /// sits in the tree.
    ///
    /// **Byte-parity contract with [`Scene::load`].** For the maps this touches
    /// (`transient_by_id`, `transient_by_parent`, `transient_parent_of`,
    /// `transient_page_of`, and the host's + its transients' `resolved_by_id`
    /// entries), the post-call state is byte-identical to what
    /// `Scene::load(patched_json)` would have produced — that is the whole point
    /// of the scoped path, and `assert_reexpand_matches_cold_load` in this
    /// module's tests is the oracle. Hence the deliberate mirroring of
    /// `compute_resolved_nodes` below, statement for statement and in the same
    /// order.
    ///
    /// Returns `None` — having mutated NOTHING — when `id` is unknown or is not
    /// an expanding host; the caller then has nothing to patch.
    pub fn reexpand_transient_host(&mut self, id: &str) -> Result<Option<ReexpandOutcome>, TickError> {
        // Every early-out happens BEFORE the first mutation, so a `None` return
        // leaves the scene bit-identical (callers treat `None` as "not my job",
        // not as "the scene is now half-updated").
        let Some(node) = self.node(id) else { return Ok(None) };
        // Derived from CANONICAL, as `compute_resolved_nodes` does — never from
        // the previous resolved record, which may already carry a stale
        // expansion-relevant value.
        let (record, expanded) = self.derive(node);
        let Some(expanded) = expanded? else { return Ok(None) };
        // From here the host's transients are rewritten: no row can be attributed. (Every early-out above left
        // the scene, and its clock, as they were.)
        self.stamps.mark_all();

        let removed = self
            .transient_by_parent
            .get(id)
            .cloned()
            .unwrap_or_default();
        for old_id in &removed {
            self.transient_by_id.remove(old_id);
            self.transient_parent_of.remove(old_id);
            self.transient_page_of.remove(old_id);
            // Collision guard: `add_transients` mirrors every transient into
            // `resolved_by_id`, so that map is shared between transients and
            // canonical nodes. Nothing stops a doc from naming a real node
            // `c00_background`; in that case the canonical node owns the entry
            // and evicting it here would blank a node that has nothing to do
            // with this host.
            if self.node(old_id).is_none() {
                self.resolved_by_id.remove(old_id);
            }
        }

        // The order of these two statements is load-bearing for byte parity:
        // `compute_resolved_nodes` (:418-431) runs `add_transients` — which also
        // writes `resolved_by_id` for each transient — and inserts the HOST's
        // resolved entry afterwards. Swapping them would flip the winner
        // whenever a transient id collides with its own host's id.
        let page_id = self.page_id_by_node.get(id).cloned();
        self.add_transients(expanded, id, page_id.as_deref());
        self.store_resolved(id, record);

        // Display-list order: `add_transients` preserves emission order and
        // `visit_z` (:537-546) walks `transient_by_parent` as a sequence, so the
        // caller can compare `removed` vs `added` positionally.
        let added = self
            .transient_by_parent
            .get(id)
            .cloned()
            .unwrap_or_default();
        Ok(Some(ReexpandOutcome { removed, added }))
    }

    /// Re-derive ONE page's background rect — the transient
    /// `__page_bg_<pageId>` quad `compute_resolved_nodes` synthesizes from the
    /// page's own props — in place, without touching the rest of the scene.
    ///
    /// **Why this exists.** A page background is not a node in the document; it
    /// is DERIVED from the page ([`create_rectangle_from_page`]) and is first in
    /// every page's z-order ([`Scene::flattened_z_order_for_page_opts`]). So the
    /// incremental patch lane cannot mutate a page's props and stop there — the
    /// quad that actually paints the background would keep the OLD fill, the OLD
    /// size, and the OLD gradient keys. Re-deriving (rather than merging the
    /// patch onto the existing bg) is what makes a `linear-gradient` -> `color`
    /// transition correct: the derivation only copies the gradient keys the page
    /// still carries, and an insert-only merge can never REMOVE the ones it
    /// drops. The derived surface is wider than the paint alone — width/height
    /// come from the page size.

    ///
    /// **Why [`Scene::reexpand_transient_host`] cannot serve this.** That path
    /// starts at `self.node(id)?`, and a page id is not a node id — even though
    /// `transient_by_parent` happens to be keyed by the page id for backgrounds.
    ///
    /// **Byte-parity contract with [`Scene::load`].** Same contract as
    /// `reexpand_transient_host`: for `transient_by_id`, `transient_by_parent`,
    /// `transient_parent_of`, `transient_page_of`, and the `resolved_by_id`
    /// mirror, the post-call state is byte-identical to what
    /// `Scene::load(patched_json)` would have produced — hence the deliberate
    /// mirroring of `compute_resolved_nodes`'s page loop below.
    ///
    /// Returns `None` — having mutated NOTHING — when `page_id` names no page.
    pub fn reexpand_page_background(&mut self, page_id: &str) -> Option<ReexpandOutcome> {
        // Early-out BEFORE the first mutation, so `None` leaves the scene
        // bit-identical (callers treat it as "not my job", not "half-updated").
        let page = self.doc.pages.iter().find(|p| p.id() == page_id)?;
        // compute_resolved_nodes derives from the CANONICAL page — never from
        // the previous resolved record.
        let bg = create_rectangle_from_page(page);
        // From here the page's background is rewritten: no row can be attributed. (An unknown page left the
        // scene, and its clock, as they were.)
        self.stamps.mark_all();

        let removed = self
            .transient_by_parent
            .get(page_id)
            .cloned()
            .unwrap_or_default();
        for old_id in &removed {
            self.transient_by_id.remove(old_id);
            self.transient_parent_of.remove(old_id);
            self.transient_page_of.remove(old_id);
            // Collision guard, same as `reexpand_transient_host` (:386-391):
            // `resolved_by_id` is shared between transients and canonical nodes,
            // and nothing stops a doc from naming a real node `__page_bg_<id>`.
            if self.node(old_id).is_none() {
                self.resolved_by_id.remove(old_id);
            }
        }

        // `add_transients` writes `resolved_by_id` too, and in a cold load the
        // page loop runs LAST (:450-455) — after every canonical node's
        // `resolved_by_id` insert — so the background wins an id collision
        // there. Re-inserting here reproduces that same winner.
        self.add_transients(vec![bg], page_id, Some(page_id));

        let added = self
            .transient_by_parent
            .get(page_id)
            .cloned()
            .unwrap_or_default();
        Some(ReexpandOutcome { removed, added })
    }

    /// `computeResolvedNodes` (Scene.ts): HTML variable fallbacks first, then
    /// table/generated expansion, then page-background rects.
    fn compute_resolved_nodes(&mut self) -> Result<(), TickError> {
        for i in 0..self.node_order.len() {
            let id = &self.node_order[i];
            let Some(node) = self.node(id) else {
                continue;
            };
            let (record, expanded) = self.derive(node);
            let id = id.clone();
            if let Some(expanded) = expanded? {
                let page_id = self.page_id_by_node.get(&id).cloned();
                self.add_transients(expanded, &id, page_id.as_deref());
            }
            // After the host's own transients, so the host wins an id collision
            // with one of them (and a later canonical node wins over an earlier
            // host's transient), as in the TS `resolved` map.
            self.store_resolved_unless_regenerated(&id, record);
        }
        // Page backgrounds (Scene.ts:3247-3264).
        let bgs: Vec<(String, RawNode)> = self
            .doc
            .pages
            .iter()
            .map(|page| {
                (
                    page.id().to_string(),
                    create_rectangle_from_page(page),
                )
            })
            .collect();
        for (page_id, bg) in bgs {
            self.add_transients(vec![bg], &page_id, Some(&page_id));
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Flattened z-order (scene.selectors.ts:58-90)
    // -----------------------------------------------------------------------

    /// `getFlattenedZOrderForPage` with the renderer defaults
    /// (visibleOnly=true, expandTransientNodes=true — Scene.ts:1968-1971).
    pub fn flattened_z_order_for_page(&self, page_id: &str) -> Vec<NodeId> {
        self.flattened_z_order_for_page_opts(page_id, true, true)
    }

    /// `getLayoutContext().mode === 'freeform'` (the WASM feed always carries
    /// the resolved layout mode). Off-page canvas-root nodes are painted only in
    /// freeform — the render gate, the camera-plan fingerprint fold, and the
    /// residency cache-bypass must all agree, so they share this predicate;
    /// Drift here would mean a renderer/cache disagreement.
    pub fn is_freeform(&self) -> bool {
        self.doc.layout_mode.as_deref() == Some("freeform")
    }

    /// Flattened z-order for the freeform canvas root — the floating
    /// (off-page) nodes that live on the pasteboard, not on any page.
    /// `getFlattenedZOrderForCanvas` analog (scene.selectors.ts:92-124): visits
    /// `canvasRootNodeIds` in order with NO page-background prefix. Used by the
    /// WASM camera frame to render off-page nodes, matching the
    /// WebGPU orchestrator's freeform pass.
    pub fn flattened_z_order_for_canvas(&self) -> Vec<NodeId> {
        self.flattened_z_order_for_canvas_opts(true, true)
    }

    pub fn flattened_z_order_for_canvas_opts(
        &self,
        visible_only: bool,
        expand_transient: bool,
    ) -> Vec<NodeId> {
        let mut ordered: Vec<NodeId> = Vec::new();
        // `getChildren(SCENE_ROOT_ID)` (scene.selectors.ts:123): the canvas root
        // is keyed off the sanitized `canvasRootNodeIds`, not the children index
        // (`build_children_index` never populates SCENE_ROOT_ID).
        self.visit_z(
            &self.doc.canvas_root_node_ids,
            visible_only,
            expand_transient,
            &mut ordered,
        );
        ordered
    }

    pub fn flattened_z_order_for_page_opts(
        &self,
        page_id: &str,
        visible_only: bool,
        expand_transient: bool,
    ) -> Vec<NodeId> {
        let mut ordered: Vec<NodeId> = vec![compute_page_bg_id(page_id)];
        let top: Vec<NodeId> = self
            .children_by_parent
            .get(page_id)
            .cloned()
            .or_else(|| {
                self.doc
                    .pages
                    .iter()
                    .find(|p| p.id() == page_id)
                    .map(|p| p.node_ids())
            })
            .unwrap_or_default();
        self.visit_z(&top, visible_only, expand_transient, &mut ordered);
        ordered
    }

    /// The renderer's flattened z-order (visible only, transients expanded) of
    /// just these top-level ids, in order: the slice of a page's or the canvas
    /// root's z-order their subtrees occupy. The engine's retained camera plan
    /// rebuilds one changed subtree with it instead of the whole list.
    pub fn flattened_z_order_for_ids(&self, ids: &[NodeId]) -> Vec<NodeId> {
        let mut out = Vec::new();
        self.visit_z(ids, true, true, &mut out);
        out
    }

    /// The top-level ids a page's z-order visits (after its background), as
    /// `flattened_z_order_for_page_opts` reads them.
    pub fn page_top_level_ids(&self, page_id: &str) -> Vec<NodeId> {
        self.children_by_parent
            .get(page_id)
            .cloned()
            .or_else(|| {
                self.doc
                    .pages
                    .iter()
                    .find(|p| p.id() == page_id)
                    .map(|p| p.node_ids())
            })
            .unwrap_or_default()
    }

    fn visit_z(
        &self,
        ids: &[NodeId],
        visible_only: bool,
        expand_transient: bool,
        out: &mut Vec<NodeId>,
    ) {
        let mut stack: Vec<(NodeId, bool)> = ids.iter().rev().map(|id| (id.clone(), false)).collect();
        let mut active = HashSet::default();
        while let Some((id, exit)) = stack.pop() {
            if exit {
                active.remove(&id);
                continue;
            }
            if !active.insert(id.clone()) {
                continue;
            }
            stack.push((id.clone(), true));
            let id = &id;
            let Some(node) = self.canonical_or_transient(id) else {
                continue;
            };
            // `node.hidden || node.opacity === 0` (scene.selectors.ts:69)
            if visible_only && (node.truthy("hidden") || node.f64("opacity") == Some(0.0)) {
                continue;
            }
            if node.node_type() == "boolean" {
                // E7.1: the node draws its result (a path in `resolved_by_id`); its children
                // are operands, never drawn. An empty result draws nothing.
                let has_result = self
                    .resolved_by_id
                    .get(id)
                    .and_then(|n| n.points())
                    .is_some_and(|p| !p.is_empty());
                if has_result || !visible_only {
                    out.push(id.clone());
                }
            } else if expand_transient && self.envelope_transients.contains_key(id) {
                // E8.17: an envelope draws its bent content (transients), never its children as they are.
                out.push(id.clone());
                let children = self.transient_by_parent.get(id).cloned().unwrap_or_default();
                if !children.is_empty() {
                    stack.extend(children.into_iter().rev().map(|id| (id, false)));
                }
            } else if expand_transient
                && node.node_type() == "generator"
                && self.transient_by_parent.get(id).is_some_and(|c| !c.is_empty())
            {
                // A nodes-mode generator drawn at a composition time: its timed output, not its stored children.
                out.push(id.clone());
                let children = self.transient_by_parent.get(id).cloned().unwrap_or_default();
                stack.extend(children.into_iter().rev().map(|id| (id, false)));
            } else if is_parent_node(node) {
                out.push(id.clone());
                // A section with "Hide contents" (design-editor E7.10) draws its background only.
                if node.node_type() == "section" && node.truthy("contentsHidden") {
                    continue;
                }
                let mut children = self
                    .children_by_parent
                    .get(id)
                    .cloned()
                    .or_else(|| children_of(node))
                    .unwrap_or_default();
                // A flipbook (motion-model.md §12.10) draws one drawing, and its onion skin when it has one.
                if crate::flipbook::is_flipbook(node) {
                    children = crate::flipbook::drawn_children(node, &children, |c| {
                        self.canonical_or_transient(c).is_some_and(|n| n.has(crate::flipbook::ONION_KEY))
                    });
                }
                // Canvas stacking "first on top" (E7.15): paint the frame's children last to first.
                if first_child_on_top(node) {
                    children.reverse();
                }
                if !children.is_empty() {
                    stack.extend(children.into_iter().rev().map(|id| (id, false)));
                }
            } else if matches!(node.node_type(), "table" | "generated" | "generator") {
                if expand_transient {
                    let children = self
                        .transient_by_parent
                        .get(id)
                        .cloned()
                        .unwrap_or_default();
                    if !children.is_empty() {
                        stack.extend(children.into_iter().rev().map(|id| (id, false)));
                    }
                } else {
                    out.push(id.clone());
                }
            } else {
                out.push(id.clone());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Transforms (nodeTransformUtils.ts:95-311; transform2d.ts:19-53)
    // -----------------------------------------------------------------------

    /// `getNodeGroupId` equivalent.
    pub fn group_parent_of(&self, id: &str) -> Option<&NodeId> {
        self.parent_map.get(id)
    }

    /// The siblings `id` is stacked among, back to front: its group's or frame's children, its page's top-level
    /// layers, or the canvas root's.
    pub fn sibling_ids(&self, id: &str) -> Option<&[NodeId]> {
        if let Some(parent) = self.parent_map.get(id) {
            return self.children_by_parent.get(parent).map(Vec::as_slice);
        }
        if let Some(page) = self.page_id_by_node.get(id) {
            if let Some(kids) = self.children_by_parent.get(page) {
                return Some(kids.as_slice());
            }
        }
        self.doc
            .canvas_root_node_ids
            .iter()
            .any(|r| r == id)
            .then_some(self.doc.canvas_root_node_ids.as_slice())
    }

    /// Clip to the layer below (`clipToBelow`, Photoshop's clipping mask ⌥⌘G, Procreate's Clipping Mask): the base a
    /// clipped layer draws inside, the nearest sibling below it that is not clipped itself (so consecutive clipped
    /// layers share one base). `None` when `id` isn't clipped or has no such sibling (it then draws unclipped).
    pub fn clip_below_base(&self, id: &str) -> Option<&NodeId> {
        if !self.canonical_or_transient(id)?.truthy("clipToBelow") {
            return None;
        }
        let siblings = self.sibling_ids(id)?;
        let at = siblings.iter().position(|s| s == id)?;
        siblings[..at]
            .iter()
            .rev()
            .find(|s| !self.canonical_or_transient(s).is_some_and(|n| n.truthy("clipToBelow")))
    }

    /// Product of every ancestor parent-node's (group/container) opacity,
    /// walking from `id`'s parent up to the root. Returns 1.0 for a top-level
    /// node.
    ///
    /// Parity with the WebGPU path (`displayListAdapter.ts`
    /// `computeAncestorOpacityProduct`): the paint path applies only
    /// each node's OWN `opacity`, so ancestor group/container opacity is folded
    /// into a leaf's effective opacity (`own × this product`) as a leaf-only
    /// multiply-down — NOT a flattened transparency group, so overlapping
    /// descendants of a translucent group blend independently.
    ///
    /// The walk steps THROUGH transient hosts (tables / generated graphics):
    /// their expanded cells/shapes are independent paint items whose own
    /// opacity is already baked in at expansion time (`tableExpansion.ts` /
    /// `generatedGraphicExpansion.ts`), so the host contributes nothing here —
    /// but an enclosing group/container above it must still dim its contents.
    /// [`Self::ancestor_opacity_product`] without the ancestors `composited` says composite their subtree as one
    /// unit at their own opacity (the renderer fades that unit, so their layers draw at full opacity inside it).
    pub fn ancestor_opacity_product_except(&self, id: &str, composited: impl Fn(&str) -> bool) -> f64 {
        let mut product = 1.0_f64;
        let mut seen = SeenIds::default();
        let mut parent_id: Option<&str> = self
            .parent_map
            .get(id)
            .or_else(|| self.transient_parent_of.get(id))
            .map(String::as_str);
        while let Some(pid) = parent_id {
            if !seen.insert(pid) {
                break;
            }
            let Some(parent) = self.effective_node(pid) else {
                break;
            };
            if is_parent_node(parent) && !composited(pid) {
                product *= parent.f64("opacity").unwrap_or(1.0);
                if product <= 0.0 {
                    return 0.0;
                }
            }
            parent_id = self
                .parent_map
                .get(pid)
                .or_else(|| self.transient_parent_of.get(pid))
                .map(String::as_str);
        }
        product
    }

    pub fn ancestor_opacity_product(&self, id: &str) -> f64 {
        let mut product = 1.0_f64;
        // A list guards a short chain, a set a deep one: a list alone made a 1,000-deep chain of groups quadratic
        // per node, and the blend planner walks it for every node (#1715).
        let mut seen = SeenIds::default();
        let mut parent_id: Option<&str> = self
            .parent_map
            .get(id)
            .or_else(|| self.transient_parent_of.get(id))
            .map(String::as_str);
        while let Some(pid) = parent_id {
            if !seen.insert(pid) {
                break;
            }
            let Some(parent) = self.effective_node(pid) else {
                break;
            };
            if is_parent_node(parent) {
                product *= parent.f64("opacity").unwrap_or(1.0);
                if product <= 0.0 {
                    return 0.0; // a fully transparent ancestor hides everything below it
                }
            }
            // Otherwise `parent` is a transient host (table / generated graphic):
            // skip its already-baked own opacity but keep walking to its group.
            parent_id = self
                .parent_map
                .get(pid)
                .or_else(|| self.transient_parent_of.get(pid))
                .map(String::as_str);
        }
        product
    }

    /// Rust twin of `isExportNodeIsolated` (`renderers/webgpu/exportExclusion.ts:36-47`),
    /// the positive counterpart to `--exclude`: when an export isolates a set of
    /// element subtrees (per-element alpha plates for the AE export
    /// path), a paint item renders only when its node lies within one of
    /// them.
    ///
    /// A node qualifies when itself OR any of its group / transient-host
    /// ancestors is in `isolate` — so isolating a group keeps every descendant
    /// leaf, and isolating a table/generated host keeps its expanded transient
    /// cells. The chain is the same `parent_map ?? transient_parent_of` walk
    /// [`Self::ancestor_opacity_product`] uses (the TS
    /// `getNodeGroupId ?? getParentIdForTransientNode`), cycle-guarded.
    ///
    /// Unlike `collect_parent_group_ids` this checks EVERY id on the chain,
    /// including non-`ParentNode` transient hosts: a table host is exactly the
    /// id a caller isolates, and its cells must survive the filter.
    ///
    /// An empty `isolate` means "no isolation requested" → every node renders.
    /// Callers short-circuit on empty before walking; this returns `true`
    /// defensively so the semantics hold either way.
    ///
    /// NB: isolation filters node paint items only. It does not suppress the
    /// solid page-background fill (gated on `transparent`), so alpha-plate
    /// callers must pair isolation with `transparent: true` — exactly as
    /// `exclude` callers do.
    pub fn is_node_isolated<S: std::hash::BuildHasher>(&self, id: &str, isolate: &std::collections::HashSet<&str, S>) -> bool {
        if isolate.is_empty() {
            return true;
        }
        let mut seen: HashSet<&str> = HashSet::default();
        let mut current: &str = id;
        loop {
            if isolate.contains(current) {
                return true;
            }
            if !seen.insert(current) {
                return false; // cyclic parent data: refuse rather than loop
            }
            let next = self
                .parent_map
                .get(current)
                .or_else(|| self.transient_parent_of.get(current));
            match next {
                Some(pid) => current = pid.as_str(),
                None => return false,
            }
        }
    }

    /// `getParentChain` (nodeTransformUtils.ts:130-187): group/container
    /// ancestors; a transient child of a `native-children` generated host
    /// treats that host as a transform parent. Returns ancestor ids from
    /// immediate parent to root.
    pub fn transform_parent_chain(&self, id: &str) -> Vec<NodeId> {
        let mut out: Vec<NodeId> = Vec::new();
        // Ancestor chains are usually a few links long: a list is the cheapest cycle guard there, a set past that
        // (a list made a 1,000-deep chain of groups quadratic per node, ~a minute a frame).
        let mut visited = SeenIds::default();
        let mut current: &str = id;
        loop {
            let pid: &str;
            if let Some(p) = self.parent_map.get(current) {
                // group/container branch: parent must be a ParentNode
                // (nodeTransformUtils.ts:154-156) else the chain breaks.
                let Some(parent) = self.effective_node(p) else {
                    break;
                };
                // A boolean parent's resolved record is the path it draws; its canonical
                // node is the group-like transform parent (E7.1).
                if !is_parent_node(parent) && !self.boolean_source_by_id.contains_key(p) {
                    break;
                }
                pid = p.as_str();
            } else if let Some(tp) = self.transient_parent_of.get(current) {
                // transient branch: only native-children generated hosts act
                // as transform parents (nodeTransformUtils.ts:157-172).
                let Some(candidate) = self.effective_node(tp) else {
                    break;
                };
                // An envelope's transients sit in its local space (E8.17): the envelope and its transient groups are
                // their transform parents.
                // A repeat's copies too (`crate::repeat`): each copy's group carries the copy's transform.
                let envelope_parent = self.envelope_transients.contains_key(tp.as_str())
                    || candidate.has(crate::envelope::ENVELOPE_OF_KEY)
                    || candidate.has(crate::repeat::REPEAT_OF_KEY);
                // A generator's output is in its local space (live-nodes.md §6.4), like a chart's parts.
                if !envelope_parent
                    && candidate.node_type() != "generator"
                    && (candidate.node_type() != "generated"
                        || candidate.str("renderMode") != Some("native-children"))
                {
                    // Transient children of every OTHER host carry the host's
                    // transform themselves: table cells bake table.x/y
                    // (expansion.rs) and generated raster/vector proxies copy
                    // the host's x/y/rotation/scale. The host therefore
                    // contributes no transform of its own — but its
                    // group/container ancestors still do. Step THROUGH the
                    // host without pushing it, matching the ancestor-opacity
                    // walk above and the TS twin (nodeTransformUtils.ts
                    // getParentChain).
                    if !visited.insert(tp) {
                        break;
                    }
                    current = tp;
                    continue;
                }
                pid = tp.as_str();
            } else {
                break;
            }
            if !visited.insert(pid) {
                break; // circular dependency guard (:174-179)
            }
            out.push(pid.to_string());
            current = pid;
        }
        out
    }

    /// `calculateAccumulatedTransform` (nodeTransformUtils.ts:251-311):
    /// page-space affine (page offset applied separately).
    ///
    /// The transform is composed parent first: the node's local transform under the accumulated transform of
    /// what is above it ([`crate::worlds`]). A canonical node whose row the world columns answer for is read
    /// from them; any other node is walked by name, with the same composition in the same order, so both give
    /// the same numbers.
    pub fn accumulated_affine(&self, id: &str) -> Option<(Affine, Transform2D)> {
        #[cfg(test)]
        ACCUMULATED_AFFINE_CALLS.with(|n| n.set(n.get() + 1));
        if let Some(&row) = self.node_index.get(id) {
            if let Some(placed) = self.worlds.placed(self.stamps.clock(), row) {
                return Some(placed.answer());
            }
        }
        let node = self.effective_node(id)?;
        let above = self.frame_above(id);
        Some(crate::worlds::Placed::own(node).under(above.as_ref()).answer())
    }

    /// Page id owning a node (incl. transients + parent chain —
    /// nodeTransformUtils.ts:211-226).
    pub fn page_of(&self, id: &str) -> Option<String> {
        if let Some(p) = self.page_id_by_node.get(id) {
            return Some(p.clone());
        }
        if let Some(p) = self.transient_page_of.get(id) {
            return Some(p.clone());
        }
        let mut current = id.to_string();
        loop {
            let parent = self
                .parent_map
                .get(&current)
                .cloned()
                .or_else(|| self.transient_parent_of.get(&current).cloned());
            let Some(pid) = parent else { return None };
            if let Some(p) = self.page_id_by_node.get(&pid) {
                return Some(p.clone());
            }
            if let Some(p) = self.transient_page_of.get(&pid) {
                return Some(p.clone());
            }
            current = pid;
        }
    }

    /// `computePagePosition` (scene.layout.ts:67-130). Default page spacing
    /// 100 (canvasDefaults.ts:152); 'pages' mode forces 2500.
    pub fn page_position(&self, page_index: usize) -> (f64, f64) {
        let mode = self.doc.layout_mode.as_deref().unwrap_or("vertical");
        let spacing = if mode == "pages" {
            crate::defaults::page::PAGES_LAYOUT_SPACING
        } else {
            crate::defaults::page::PAGE_SPACING
        };
        match mode {
            "freeform" => {
                let Some(page) = self.doc.pages.get(page_index) else {
                    return (0.0, 0.0);
                };
                let fx = page.0.get("freeformX").and_then(Value::as_f64);
                let fy = page.0.get("freeformY").and_then(Value::as_f64);
                match (fx, fy) {
                    (Some(x), Some(y)) => (x, y),
                    _ => {
                        // fallback to vertical (scene.layout.ts:81-94)
                        let mut y = 0.0;
                        for i in 0..page_index {
                            y += self.doc.pages[i].height() + crate::defaults::page::PAGE_SPACING;
                        }
                        (0.0, y)
                    }
                }
            }
            "horizontal" => {
                let mut x = 0.0;
                for i in 0..page_index.min(self.doc.pages.len()) {
                    x += self.doc.pages[i].width() + spacing;
                }
                (x, 0.0)
            }
            "grid" => {
                let cols = self
                    .doc
                    .grid_columns
                    .filter(|c| *c >= 1.0)
                    .map(|c| c.floor() as usize)
                    .unwrap_or(2);
                let col = page_index % cols;
                let row = page_index / cols;
                let mut x = 0.0;
                for i in 0..col {
                    let idx = row * cols + i;
                    if let Some(page) = self.doc.pages.get(idx) {
                        x += page.width() + spacing;
                    }
                }

                let mut y = 0.0;
                for r in 0..row {
                    let mut row_max = 0.0;
                    for c in 0..cols {
                        let idx = r * cols + c;
                        if let Some(page) = self.doc.pages.get(idx) {
                            row_max = f64::max(row_max, page.height());
                        }
                    }
                    y += row_max + spacing;
                }

                (x, y)
            }
            // vertical | pages | default
            _ => {
                let mut y = 0.0;
                for i in 0..page_index.min(self.doc.pages.len()) {
                    y += self.doc.pages[i].height() + spacing;
                }
                (0.0, y)
            }
        }
    }

    /// Canvas-space affine incl. page offset (transform2d.ts:38-53).
    pub fn node_canvas_affine(&self, id: &str) -> Option<Affine> {
        self.node_canvas_affine_and_transform(id).map(|(a, _)| a)
    }

    /// Canvas-space TRS (transform2d.ts:19-31 / nodeTransformUtils.ts:54-66).
    pub fn node_canvas_transform(&self, id: &str) -> Option<Transform2D> {
        self.node_canvas_affine_and_transform(id).map(|(_, t)| t)
    }

    /// [`Self::node_canvas_affine`] and [`Self::node_canvas_transform`] together, from one ancestor walk (each
    /// alone walks the whole chain, and so does finding the node's page).
    pub fn node_canvas_affine_and_transform(&self, id: &str) -> Option<(Affine, Transform2D)> {
        let (affine, trs) = self.accumulated_affine(id)?;
        // A row the world columns answer for has its page's position beside its transform.
        let row = self.node_index.get(id).filter(|row| self.worlds.placed(self.stamps.clock(), **row).is_some());
        let page_pos = match row {
            Some(&row) => self.worlds.page(row),
            None => self
                .page_of(id)
                .and_then(|pid| self.doc.pages.iter().position(|p| p.id() == pid))
                .map(|i| self.page_position(i))
                .unwrap_or((0.0, 0.0)),
        };
        Some((
            compose(transform::translate(page_pos.0, page_pos.1), affine),
            Transform2D {
                x: trs.x + page_pos.0,
                y: trs.y + page_pos.1,
                ..trs
            },
        ))
    }
}

// ---------------------------------------------------------------------------
// Incremental node edits (the renderer's scene-sync patch lane)
// ---------------------------------------------------------------------------

/// A node's local transform as the numbers [`Scene::write_placement`] stores: position, rotation in degrees
/// and scale. Each is the number a document edit would carry (`None` from a non-finite value is not one).
#[derive(Debug, Clone, PartialEq)]
pub struct NodePlacement {
    pub x: crate::value::Number,
    pub y: crate::value::Number,
    pub rotation: crate::value::Number,
    pub scale_x: crate::value::Number,
    pub scale_y: crate::value::Number,
}

/// What [`Scene::apply_node_edits`] changed, for the caller's hash and cache
/// bookkeeping. Transient lists are in display-list order, host by host, so a
/// caller can compare `transients_removed` with `transients_added` as sequences
/// (equal → the hosts' slices of the display list kept their shape).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NodeEditOutcome {
    /// Canonical ids whose node and resolved record were (re)written.
    pub written: Vec<NodeId>,
    /// Canonical ids that no longer exist.
    pub removed: Vec<NodeId>,
    /// Transient ids dropped (a removed or re-expanded host's old expansion).
    pub transients_removed: Vec<NodeId>,
    /// Transient ids (re)created by re-expanding a host.
    pub transients_added: Vec<NodeId>,
    /// Membership, hierarchy, type or order changed: the indexes were rebuilt.
    pub structural: bool,
    /// Envelopes (E8.17) were re-resolved: every transient they draw may have changed.
    pub envelopes_rebent: bool,
}

/// What an edit did besides the rows it reports as written, for the stamps ([`Scene::apply_node_edits`]).
#[derive(Debug, Default)]
struct EditEffects {
    /// It passed every refusal and began to write the scene.
    began: bool,
    /// It re-derives records that read other nodes: boolean results, painted groups' outlines, envelopes, the
    /// text-flow index.
    derived: bool,
}

impl Scene {
    /// Apply one edit batch in place: `upserts` are whole nodes (added, or
    /// replacing the node with that id), `removes` are deleted ids,
    /// `page_node_ids` replaces pages' top-level lists (z-order, membership),
    /// and `canvas_root_node_ids`, when given, replaces the canvas root's list
    /// (a page-less Design document's top level: its frames and loose layers).
    ///
    /// The post-call state equals `Scene::load` of the document with the same
    /// edits applied (`residency::ResidentScene::divergence_from_full_load` is
    /// the oracle): upserts are normalized as hydration normalizes them, a node
    /// of an unknown type is dropped as hydration drops it, and every derived
    /// map is brought to what a cold load builds. The cost is O(edited nodes) for
    /// the resolved records and expansions, plus an O(nodes) index rebuild only
    /// when the structure changed: no JSON parse, no re-resolution of untouched
    /// nodes, no re-expansion of untouched tables and charts.
    ///
    /// Errors, having mutated nothing, on an upsert without an id or a page id
    /// that names no page.
    ///
    /// Stamps: a batch that only rewrites nodes nothing else is derived from stamps each written row's
    /// placement, paint and content, whatever else the scene holds (an untouched boolean elsewhere does not
    /// widen it). A structural edit, one that re-derives records reading other nodes (it touches a boolean's or
    /// a painted group's inputs, the text flow, a host of transients, or the scene has envelopes, which every
    /// edit re-resolves), and one that fails after it began to write dirty everything. An edit refused before
    /// its first write, for any reason, and a batch with nothing applicable (empty, unknown removals, membership as
    /// it already is) leave the scene and its clock as they were.
    pub fn apply_node_edits(
        &mut self,
        upserts: Vec<RawNode>,
        removes: &[NodeId],
        page_node_ids: &[(String, Vec<NodeId>)],
        canvas_root_node_ids: Option<&[NodeId]>,
    ) -> Result<NodeEditOutcome, String> {
        // The edit itself reports whether it began to write and whether it re-derived records that read other
        // nodes; nothing here guesses either from the scene's contents or repeats the edit's validation.
        let mut effects = EditEffects::default();
        let result = self.apply_node_edits_unmarked(upserts, removes, page_node_ids, canvas_root_node_ids, &mut effects);
        if !effects.began {
            // Refused before the first write, whatever the reason: the scene and its clock are as they were.
            return result;
        }
        match &result {
            // It began and wrote, removed and re-derived nothing.
            Ok(out) if !out.structural && out.written.is_empty() && out.removed.is_empty() && !effects.derived
                && !out.envelopes_rebent && out.transients_removed.is_empty() && out.transients_added.is_empty() => {}
            // Plain rows only: each written node is the whole of its change.
            Ok(out) if !out.structural && out.removed.is_empty() && !effects.derived && !out.envelopes_rebent
                && out.transients_removed.is_empty() && out.transients_added.is_empty()
                && out.written.iter().all(|id| !self.transient_by_parent.contains_key(id)) => {
                for id in &out.written {
                    // A whole-node edit may have changed any of the node's fields.
                    match self.node_index.get(id) {
                        Some(&row) => self.stamps.mark_row(row as u32),
                        None => self.stamps.mark_all(),
                    };
                }
            }
            // Structure changed (rows moved), records that read other nodes were re-derived, or the edit failed
            // after it began to write: no row can be attributed.
            _ => {
                self.stamps.resize(self.doc.nodes.len());
                self.stamps.mark_all();
            }
        }
        result
    }

    fn apply_node_edits_unmarked(
        &mut self,
        upserts: Vec<RawNode>,
        removes: &[NodeId],
        page_node_ids: &[(String, Vec<NodeId>)],
        canvas_root_node_ids: Option<&[NodeId]>,
        effects: &mut EditEffects,
    ) -> Result<NodeEditOutcome, String> {
        // Validate before the first mutation.
        for n in &upserts {
            if n.id().is_empty() {
                return Err("upsert without an id".into());
            }
        }
        // What the batch re-derives besides the written nodes' own records, asked before the edit: the parent
        // map still knows a removed node's parents.
        let rederived = upserts.iter().map(|n| self.rederived_by(n.id(), Some(n)))
            .chain(removes.iter().map(|id| self.rederived_by(id, None)))
            // A document with envelopes re-resolves them after any edit, one of page lists alone included.
            .fold(Rederived { envelopes: self.has_envelopes(), ..Rederived::default() }, Rederived::or);
        for (pid, _) in page_node_ids {
            if !self.doc.pages.iter().any(|p| p.id() == pid) {
                return Err(format!("unknown page '{pid}'"));
            }
        }
        // What this edit re-derives from nodes other than the ones it writes.
        effects.derived = rederived.any();
        let mut remove_set: HashSet<NodeId> = removes.iter().cloned().collect();
        let mut kept: Vec<RawNode> = Vec::with_capacity(upserts.len());
        for mut n in upserts {
            if !VALID_NODE_TYPES.contains(&n.node_type()) {
                // Hydration skips it; so does the patch (a removal if it existed).
                remove_set.insert(n.id().to_string());
                continue;
            }
            crate::feed_points::resolve_node_feed_points(&mut n);
            normalize_node_in_place(&mut n);
            if crate::path_points::compact_on_load() {
                n.compact();
            }
            // The upsert is the node's final state.
            remove_set.remove(n.id());
            kept.push(n);
        }
        let mut removing: Vec<NodeId> = remove_set
            .into_iter()
            .filter(|id| self.node_index.contains_key(id))
            .collect();
        removing.sort();
        let pages_changed = page_node_ids.iter().any(|(pid, ids)| {
            self.doc
                .pages
                .iter()
                .find(|p| p.id() == pid)
                .is_some_and(|p| p.node_ids() != *ids)
        });
        let restructured = kept.iter().any(|n| match self.node(n.id()) {
            None => true,
            Some(old) => {
                old.json_field("type") != n.json_field("type")
                    || old.json_field("parentId") != n.json_field("parentId")
                    || old.json_field("children") != n.json_field("children")
            }
        });
        let roots_changed = canvas_root_node_ids.is_some_and(|ids| self.doc.canvas_root_node_ids != ids);
        let structural = !removing.is_empty() || pages_changed || roots_changed || restructured;
        // Nothing applicable is left (an empty batch, removals of unknown ids, membership as it already is): this
        // is not an edit. Nothing is re-derived and the scene and its clock stand.
        if kept.is_empty() && !structural {
            return Ok(NodeEditOutcome::default());
        }

        if structural {
            let replacements: HashMap<&str, &RawNode> = kept.iter().map(|n| (n.id(), n)).collect();
            let removed: HashSet<&str> = removing.iter().map(String::as_str).collect();
            validate_hierarchy(
                self.doc.nodes.iter()
                    .filter(|n| !removed.contains(n.id()) && !replacements.contains_key(n.id()))
                    .chain(replacements.values().copied()),
            )?;
        }

        // Expand the final upsert for each generated host before mutation.
        // Its variables and brush table are unchanged by node edits; reuse this
        // derivation below so atomic refusal does not double chart work.
        let mut prepared = HashMap::default();
        let final_upserts: HashMap<&str, &RawNode> = kept.iter().map(|n| (n.id(), n)).collect();
        for node in final_upserts.into_values() {
            if node.node_type() == "generated" {
                let (record, expanded) = self.derive(node);
                prepared.insert(node.id().to_string(), (record, expanded?));
            }
        }

        // Every refusal is above this line; from here the scene is written.
        effects.began = true;
        let mut out = NodeEditOutcome {
            structural,
            ..NodeEditOutcome::default()
        };

        if !structural {
            // Property edits only: the indexes stand; rewrite each node and
            // re-derive its resolved record (and expansion, for a host).
            let mut rows = Vec::with_capacity(kept.len());
            for n in kept {
                let id = n.id().to_string();
                if let Some(&i) = self.node_index.get(&id) {
                    self.doc.nodes[i] = n;
                    rows.push(i as u32);
                }
                out.written.push(id);
            }
            self.derive_written(&rows, rederived, &mut out, &mut prepared)?;
            return Ok(out);
        }

        // 1. Removals: their expansions, resolved records, then the nodes.
        for id in &removing {
            self.drop_transients_of(id, &mut out.transients_removed);
            self.resolved_by_id.remove(id);
        }
        if !removing.is_empty() {
            let gone: HashSet<&str> = removing.iter().map(String::as_str).collect();
            self.doc.nodes.retain(|n| !gone.contains(n.id()));
            self.reindex_nodes();
        }
        out.removed = removing;

        // 2. Upserts: replace in place, or append (a JS Map appends a new key,
        // and the feed serializes nodes in that order).
        let mut written: Vec<NodeId> = Vec::with_capacity(kept.len());
        for n in kept {
            let id = n.id().to_string();
            match self.node_index.get(&id) {
                Some(&i) => self.doc.nodes[i] = n,
                None => {
                    self.node_index.insert(id.clone(), self.doc.nodes.len());
                    self.node_order.push(id.clone());
                    self.doc.nodes.push(n);
                }
            }
            written.push(id);
        }

        // 3. Page lists.
        for (pid, ids) in page_node_ids {
            if let Some(page) = self.doc.pages.iter_mut().find(|p| p.id() == pid) {
                page.set_node_ids(ids.clone());
            }
        }

        // 3b. The canvas root's list (sanitized with the page lists below).
        if let Some(ids) = canvas_root_node_ids {
            self.doc.canvas_root_node_ids = ids.to_vec();
        }

        // 4. Derived indexes and sanitizers, in `from_document`'s order.
        let old_page_of = std::mem::take(&mut self.page_id_by_node);
        self.parent_map = build_parent_map(&self.doc.nodes);
        sanitize_unique_page_membership(&mut self.doc.pages, &self.parent_map);
        sanitize_top_level_lists(&mut self.doc.pages, &self.doc.nodes, &self.node_index);
        self.page_id_by_node = build_page_index(&self.doc.pages, &self.doc.nodes, &self.node_index);
        sanitize_canvas_root_node_ids(
            &mut self.doc.canvas_root_node_ids,
            &self.node_index,
            &self.parent_map,
            &self.page_id_by_node,
        );
        self.children_by_parent = build_children_index(&self.doc.pages, &self.doc.nodes);

        // 5. Resolved records: the upserts, plus every untouched host that
        // changed page (its transients carry the page id).
        let mut seen: HashSet<NodeId> = written.iter().cloned().collect();
        let mut refresh = written.clone();
        let mut moved_hosts: Vec<NodeId> = self
            .transient_by_parent
            .keys()
            .filter(|h| self.node_index.contains_key(*h) && !seen.contains(*h))
            .filter(|h| old_page_of.get(*h) != self.page_id_by_node.get(*h))
            .cloned()
            .collect();
        moved_hosts.sort();
        for h in moved_hosts {
            seen.insert(h.clone());
            refresh.push(h);
        }
        let rows: Vec<u32> = refresh.iter().filter_map(|id| self.node_index.get(id).map(|&row| row as u32)).collect();
        self.derive_written(&rows, rederived, &mut out, &mut prepared)?;
        out.written = refresh;
        Ok(out)
    }

    /// Re-resolve every envelope after an edit, reporting its transients as removed and added.
    fn envelope_edit(&mut self, out: &mut NodeEditOutcome) {
        out.envelopes_rebent = true;
        let removed = self.clear_envelopes();
        out.transients_removed.extend(removed);
        let touched = self.compute_envelopes();
        out.transients_added.extend(touched.into_iter().filter(|id| self.transient_by_id.contains_key(id)));
    }

    /// `node_order` / `node_index` from `doc.nodes`, as `from_document` builds
    /// them (a duplicate id keeps its first position and points at its last copy).
    fn reindex_nodes(&mut self) {
        self.node_order.clear();
        self.node_index.clear();
        for (i, n) in self.doc.nodes.iter().enumerate() {
            if !self.node_index.contains_key(n.id()) {
                self.node_order.push(n.id().to_string());
            }
            self.node_index.insert(n.id().to_string(), i);
        }
    }

    /// Drop a host's transient records (appending their ids to `removed`).
    fn drop_transients_of(&mut self, id: &str, removed: &mut Vec<NodeId>) {
        let Some(old) = self.transient_by_parent.remove(id) else {
            return;
        };
        for t in old {
            self.transient_by_id.remove(&t);
            self.transient_parent_of.remove(&t);
            self.transient_page_of.remove(&t);
            // Collision guard (see `reexpand_transient_host`).
            if self.node(&t).is_none() {
                self.resolved_by_id.remove(&t);
            }
            removed.push(t);
        }
    }

    /// Re-derive one canonical node's resolved record and, for a host, its
    /// expansion, exactly as `compute_resolved_nodes` does for that id.
    /// Derive again the resolved record and the expansion of the canonical node at `row` (`prepared` may hold
    /// them already, by the node's name).
    fn refresh_resolved(
        &mut self,
        row: u32,
        out: &mut NodeEditOutcome,
        prepared: &mut HashMap<String, (Option<RawNode>, Option<Vec<RawNode>>)>,
    ) -> Result<(), TickError> {
        #[cfg(test)]
        if let Some(probe) = DERIVATION_PROBE.with(|probe| probe.get()) {
            probe(self);
        }
        let Some(node) = self.doc.nodes.get(row as usize) else {
            return Ok(());
        };
        let id = node.id().to_string();
        let id = id.as_str();
        let (record, expanded) = match prepared.remove(id) {
            Some(derived) => derived,
            None => {
                let (record, expanded) = self.derive(node);
                (record, expanded?)
            }
        };
        self.drop_transients_of(id, &mut out.transients_removed);
        if let Some(expanded) = expanded {
            let page_id = self.page_id_by_node.get(id).cloned();
            self.add_transients(expanded, id, page_id.as_deref());
            if let Some(ids) = self.transient_by_parent.get(id) {
                out.transients_added.extend(ids.iter().cloned());
            }
        }
        self.store_resolved_unless_regenerated(id, record);
        Ok(())
    }

    /// [`Scene::store_resolved`], except for a nodes-mode generator's stored child that its host's timed output
    /// (drawn at a composition's time, or fed by the editor's playback) regenerated under the same id: the output
    /// is what draws, so its transient keeps the resolved slot whatever order the two come in. Without this a stored
    /// child after its host in `doc.nodes` put the design's copy (the output at time 0) back, and a generator reading
    /// `ctx.t` drew the same image at every frame.
    fn store_resolved_unless_regenerated(&mut self, id: &str, record: Option<RawNode>) {
        let regenerated = self
            .transient_parent_of
            .get(id)
            .and_then(|host| self.node(host))
            .is_some_and(|host| host.node_type() == "generator");
        if regenerated {
            if let Some(t) = self.transient_by_id.get(id).cloned() {
                self.resolved_by_id.insert(id.to_string(), t);
            }
            return;
        }
        self.store_resolved(id, record);
    }
}

// ---------------------------------------------------------------------------
// helpers (ports of scene.indexes.ts / scene.sanitizers.ts / nodes.ts guards)
// ---------------------------------------------------------------------------

/// A group, or a frame with children, whose opacity is between 0 and 1: its layers composite as one unit, faded by
/// that opacity as a whole (as in Figma and After Effects), so where its layers overlap they don't show through each
/// other.
pub fn is_opacity_group(node: &RawNode) -> bool {
    let o = node.f64("opacity").unwrap_or(1.0);
    o > 0.0
        && o < 1.0
        && match node.node_type() {
            "group" => true,
            "container" => node.array("children").is_some_and(|c| !c.is_empty()),
            _ => false,
        }
}

/// `isParentNode` (nodes.ts): group/container/section with a children array, or a nodes-mode generator (its
/// `children` are its stored derived output, generated-content RFC §2).
pub fn is_parent_node(node: &RawNode) -> bool {
    matches!(node.node_type(), "group" | "boolean" | "container" | "section" | "generator")
        && matches!(node.json_field("children"), Some(Value::Array(_)))
}

/// `createNodeModelMatrix`'s transform-id redirect (`WebGPUPathRenderer.ts:2202-2208`;
/// also `WebGPUPathShadowRenderer.ts:98`): synthetic nodes — generated-graphic proxies
/// (`generatedGraphicExpansion.ts:81,102`) and line body/arrow parts — carry
/// `_transformSourceId` and resolve their model transform through the SOURCE node's
/// chain. The proxy's own x/y are ignored on that path (the source transform already
/// carries them). Critical for `raster-asset` generated hosts nested in containers:
/// the proxy's own parent chain BREAKS (`nodeTransformUtils.ts:157-172` only treats
/// `native-children` hosts as transform parents), so without the redirect the proxy
/// renders at host-LOCAL coordinates in page space (attribution sweep find, canvases
/// 3d0f70ef page-0 map / d31a991e page-3 latex).
pub fn paint_transform_id<'a>(node: &'a RawNode) -> &'a str {
    node.str("_transformSourceId")
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| node.id())
}

/// The record a group with appearance paints its fills and strokes on: a closed even-odd `path` of its merged
/// outline (group-local), carrying the group's `fills`, `strokes` and `appearanceOrder` and drawn through the
/// group's transform (`_transformSourceId`). The group's own effects, opacity and blend stay on the group, which
/// applies them to its whole appearance (a subtree capture), so the record carries none.
pub fn group_appearance_node(
    group: &RawNode,
    result: &crate::boolean_node::BooleanResult,
) -> RawNode {
    let mut m = crate::value::Map::new();
    for k in ["id", "fills", "strokes", crate::appearance::APPEARANCE_ORDER_KEY] {
        if let Some(v) = group.json_field(k) {
            m.insert(k.into(), v.clone());
        }
    }
    m.insert("type".into(), Value::from("path"));
    m.insert("_transformSourceId".into(), Value::from(group.id()));
    m.insert(
        "points".into(),
        Value::Array(result.points.iter().map(|p| p.to_json()).collect()),
    );
    m.insert("closed".into(), Value::from(true));
    m.insert("fillRule".into(), Value::from("evenodd"));
    RawNode::new(m)
}

/// Figma's canvas stacking "First on top" (`flexLayout.itemReverseZIndex`; `childrenInPaintOrder` in
/// nodes.ts): an auto-layout frame whose first child paints on top, so its children paint last to first.
/// Visual only: the children keep their order (and the layers panel its).
pub fn first_child_on_top(node: &RawNode) -> bool {
    node.node_type() == "container"
        && node
            .object("flexLayout")
            .and_then(|layout| layout.get("itemReverseZIndex"))
            .and_then(Value::as_bool)
            == Some(true)
}

fn children_of(node: &RawNode) -> Option<Vec<NodeId>> {
    node.array("children").map(|a| {
        a.iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect()
    })
}

/// Validate the final node graph before expanding or editing it. Last duplicate id
/// wins, matching node_index; shared descendants are allowed, cycles are not.
fn validate_hierarchy<'a>(nodes: impl IntoIterator<Item = &'a RawNode>) -> Result<(), String> {
    let nodes: HashMap<&str, &RawNode> = nodes.into_iter().map(|n| (n.id(), n)).collect();
    let mut state: HashMap<&str, u8> = HashMap::default();
    for &root in nodes.keys() {
        let mut stack = vec![(root, false)];
        while let Some((id, exit)) = stack.pop() {
            if exit {
                state.insert(id, 2);
                continue;
            }
            match state.get(id) {
                Some(1) => return Err(format!("cyclic node hierarchy at '{id}'; remove the child link that closes the cycle")),
                Some(2) => continue,
                _ => {}
            }
            let Some(node) = nodes.get(id) else { continue };
            state.insert(id, 1);
            stack.push((id, true));
            if is_parent_node(node) {
                if let Some(children) = node.array("children") {
                    stack.extend(children.iter().rev().filter_map(Value::as_str).map(|id| (id, false)));
                }
            }
        }
    }
    Ok(())
}

/// scene.indexes.ts:5-15.
pub fn build_parent_map(nodes: &[RawNode]) -> HashMap<NodeId, NodeId> {
    let mut map = HashMap::default();
    for node in nodes {
        if is_parent_node(node) {
            if let Some(children) = children_of(node) {
                for c in children {
                    map.insert(c, node.id().to_string());
                }
            }
        }
    }
    map
}

/// scene.indexes.ts:17-26.
pub fn build_children_index(pages: &[RawPage], nodes: &[RawNode]) -> HashMap<NodeId, Vec<NodeId>> {
    let mut map = HashMap::default();
    for page in pages {
        map.insert(page.id().to_string(), page.node_ids());
    }
    for node in nodes {
        if is_parent_node(node) {
            map.insert(node.id().to_string(), children_of(node).unwrap_or_default());
        }
    }
    map
}

/// scene.indexes.ts:28-42 (stack-based walk, duplicate-tolerant).
pub fn build_page_index(
    pages: &[RawPage],
    nodes: &[RawNode],
    node_index: &HashMap<NodeId, usize>,
) -> HashMap<NodeId, String> {
    let mut map = HashMap::default();
    for page in pages {
        let mut stack: Vec<NodeId> = page.node_ids();
        let mut seen = HashSet::default();
        while let Some(id) = stack.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            map.insert(id.clone(), page.id().to_string());
            if let Some(&i) = node_index.get(&id) {
                let node = &nodes[i];
                if is_parent_node(node) {
                    if let Some(children) = children_of(node) {
                        stack.extend(children);
                    }
                }
            }
        }
    }
    map
}

/// `sanitizeUniquePageMembership` (scene.sanitizers.ts:24-40): a node id may
/// appear in only one page's nodeIds, and only if it has no group parent.
pub fn sanitize_unique_page_membership(
    pages: &mut [RawPage],
    parent_map: &HashMap<NodeId, NodeId>,
) {
    let mut seen: HashSet<NodeId> = HashSet::default();
    for page in pages.iter_mut() {
        let ids = page.node_ids();
        if ids.is_empty() {
            continue;
        }
        let mut filtered = Vec::with_capacity(ids.len());
        for id in ids {
            if parent_map.contains_key(&id) {
                continue;
            }
            if seen.insert(id.clone()) {
                filtered.push(id);
            }
        }
        page.set_node_ids(filtered);
    }
}

/// `sanitizeTopLevelLists` (scene.sanitizers.ts:5-22). QUIRK: only
/// `type === 'group'` children are considered (containers are NOT — copy the
/// bug; scene.sanitizers.ts:12).
pub fn sanitize_top_level_lists(
    pages: &mut [RawPage],
    nodes: &[RawNode],
    node_index: &HashMap<NodeId, usize>,
) {
    for page in pages.iter_mut() {
        let ids = page.node_ids();
        if ids.is_empty() {
            continue;
        }
        let mut child_set: HashSet<NodeId> = HashSet::default();
        for id in &ids {
            if let Some(&i) = node_index.get(id) {
                let n = &nodes[i];
                if n.node_type() == "group" {
                    if let Some(children) = children_of(n) {
                        child_set.extend(children);
                    }
                }
            }
        }
        if child_set.is_empty() {
            continue;
        }
        let filtered: Vec<NodeId> = ids
            .into_iter()
            .filter(|id| !child_set.contains(id))
            .collect();
        page.set_node_ids(filtered);
    }
}

/// `sanitizeCanvasRootNodeIds` (Scene.ts:617-650), filter half: drop ids that
/// are missing, grouped, page-owned, or duplicated — preserving order. The
/// engine now RENDERS canvas-root (off-page) nodes, so — like the
/// page-membership sanitizers above — it must not trust the feed's exclusivity:
/// a stray id also parented to a page/group would otherwise render twice (page
/// loop + canvas-root loop) or surface a grouped child as a phantom top-level
/// node. (The editor's auto-tracking of untracked floating nodes is an
/// editor-scene concern; the feed already carries the complete sanitized list.)
pub fn sanitize_canvas_root_node_ids(
    canvas_root_node_ids: &mut Vec<NodeId>,
    node_index: &HashMap<NodeId, usize>,
    parent_map: &HashMap<NodeId, NodeId>,
    page_id_by_node: &HashMap<NodeId, String>,
) {
    // Page-based canvases (the overwhelming majority) have an empty list — skip
    // the allocation entirely. `from_document` runs per animation frame and on
    // the mobile FFI's per-frame rebuild, so the empty fast-path matters.
    if canvas_root_node_ids.is_empty() {
        return;
    }
    let mut seen: HashSet<NodeId> = HashSet::default();
    canvas_root_node_ids.retain(|id| {
        // `&&` short-circuits, so the dedup `insert` (and its clone) runs only
        // for ids that pass the cheap membership checks; `insert` returns false
        // for a duplicate, dropping it.
        node_index.contains_key(id)
            && !parent_map.contains_key(id)
            && !page_id_by_node.contains_key(id)
            && seen.insert(id.clone())
    });
}

/// Convenience: top-level entry — parse and hydrate, returning a render-ready
/// scene.
pub fn load_scene_json(canvas_data_json: &str) -> Result<Scene, crate::value::Error> {
    let v: Value = crate::value::from_str(canvas_data_json)?;
    Scene::load_owned(v)
}

/// Some headless payloads wrap the scene as `{ canvas_data: {...} }`
/// (`HeadlessCanvasResponse`, scene-schema.md §1.1). Accept both.
pub fn extract_canvas_data(root: &Value) -> &Value {
    root.get("canvas_data").unwrap_or(root)
}

/// [`extract_canvas_data`] by value.
pub fn extract_canvas_data_owned(mut root: Value) -> Value {
    match root.get_mut("canvas_data") {
        Some(inner) => inner.take(),
        None => root,
    }
}


/// A table's or generated graphic's blend mode, handed to the transient children it expands into (a child's
/// own mode wins). They draw in its place, each blending with the host's mode, as each carries the host's
/// opacity: exact for a single-child graphic (an image, a QR), per child for a table or a chart.
fn inherit_blend_mode(host: &RawNode, mut children: Vec<RawNode>) -> Vec<RawNode> {
    let Some(mode) = host.str("blendMode").filter(|m| !m.is_empty() && *m != "normal") else {
        return children;
    };
    for child in &mut children {
        if child.str("blendMode").is_none_or(|m| m.is_empty() || m == "normal") {
            child.set("blendMode", crate::value::json!(mode));
        }
    }
    children
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    #[test]
    fn cyclic_edits_are_atomic_and_disconnected_cycles_are_refused() {
        let doc = crate::value::json!({"pages":[{"id":"p","nodeIds":["a"]}],"nodes":[
            {"id":"a","type":"group","children":["b"]},
            {"id":"b","type":"group","children":[]}
        ]});
        let mut scene = Scene::load(&doc).unwrap();
        let before = scene.doc.nodes.clone();
        let bad = RawNode::new(crate::value::json!({"id":"b","type":"group","children":["a"]}).as_object().unwrap().clone());
        assert!(scene.apply_node_edits(vec![bad], &[], &[], None).is_err());
        assert_eq!(scene.doc.nodes, before);
        assert_eq!(scene.flattened_z_order_for_ids(&["a".into()]), vec!["a", "b"]);
        let disconnected = crate::value::json!({"nodes":[
            {"id":"a","type":"group","children":["b"]},
            {"id":"b","type":"group","children":["a"]}
        ]});
        assert!(Scene::load(&disconnected).is_err());
        assert!(Scene::load_owned(disconnected.clone()).is_err());
        assert!(Scene::from_document(hydrate_document(&disconnected)).is_err());
    }

    #[test]
    fn deep_hierarchy_uses_explicit_traversal_stacks() {
        let count = 10_000;
        let nodes: Vec<Value> = (0..count).map(|i| crate::value::json!({
            "id":format!("n{i}"), "type":"group",
            "children": if i + 1 < count {vec![format!("n{}", i+1)]} else {vec![]}
        })).collect();
        let doc = crate::value::json!({"nodes": nodes, "pages":[{"id":"p","nodeIds":["n0"]}]});
        let scene = Scene::load(&doc).unwrap();
        assert_eq!(scene.page_id_by_node.len(), count);
        let order = scene.flattened_z_order_for_ids(&["n0".into()]);
        assert_eq!(order.len(), count);
        assert_eq!(order.last().unwrap(), "n9999");
    }

    #[test]
    fn a_refused_edit_leaves_the_clock_alone_whatever_refuses_it() {
        let d = json!({"nodes": [
            {"id": "ga", "type": "group", "x": 0, "y": 0, "children": ["gb"]},
            {"id": "gb", "type": "group", "x": 0, "y": 0, "children": []},
            {"id": "r", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10}],
            "pages": [{"id": "p", "width": 100, "height": 100, "nodeIds": ["ga", "r"]}]});
        let mut scene = Scene::load(&d).unwrap();
        let moved = scene.subscribe(crate::stamps::Group::Placement);
        let r = scene.node_index["r"];
        scene.write_placement(r as u32, None, crate::value::Number::from_f64(0.5));
        let clock = scene.stamps.clock();
        // A cycle is refused by the hierarchy check, which is not one of the id or page checks.
        let cycle = RawNode::from_value(json!({"id": "gb", "type": "group", "x": 0, "y": 0, "children": ["ga"]})).unwrap();
        assert!(scene.apply_node_edits(vec![cycle], &[], &[], None).is_err());
        assert!(scene.apply_node_edits(vec![RawNode::from_value(json!({"type": "rectangle"})).unwrap()], &[], &[], None).is_err());
        assert!(scene.apply_node_edits(vec![], &[], &[("no such page".to_string(), vec![])], None).is_err());
        assert_eq!(scene.apply_node_edits(vec![], &[], &[], None).unwrap(), NodeEditOutcome::default());
        assert_eq!(scene.stamps.clock(), clock);
        // The refusals cleared no consumer's dirty rows either.
        assert_eq!(scene.stamps.dirty(moved), [r as u32]);
    }

    #[test]
    fn a_batch_with_nothing_applicable_re_derives_nothing() {
        // A repeat is resolved with the envelopes, which every edit re-resolves: a batch that comes to nothing
        // must not count as one.
        let d = json!({"nodes": [
            {"id": "rep", "type": "group", "x": 100, "y": 50, "children": ["a"], "repeat": {"type": "grid", "rows": 2, "columns": 3}},
            {"id": "a", "type": "rectangle", "x": 0, "y": 0, "width": 40, "height": 20},
            {"id": "plain", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10}],
            "pages": [{"id": "p1", "width": 500, "height": 400, "nodeIds": ["rep", "plain"]}]});
        let mut scene = Scene::load(&d).unwrap();
        assert!(scene.has_envelopes(), "the repeat is resolved");
        let clock = scene.stamps.clock();
        let transients = scene.transient_by_id.clone();
        let nothing = NodeEditOutcome::default();
        assert_eq!(scene.apply_node_edits(vec![], &[], &[], None).unwrap(), nothing);
        assert_eq!(scene.apply_node_edits(vec![], &["no such node".to_string()], &[], None).unwrap(), nothing);
        let same = [("p1".to_string(), vec!["rep".to_string(), "plain".to_string()])];
        assert_eq!(scene.apply_node_edits(vec![], &[], &same, None).unwrap(), nothing);
        assert_eq!(scene.stamps.clock(), clock);
        assert_eq!(scene.transient_by_id, transients);
        // A real edit there still dirties everything: the envelopes are re-resolved.
        let mut plain = scene.node("plain").unwrap().clone();
        plain.set("x", json!(5));
        scene.apply_node_edits(vec![plain], &[], &[], None).unwrap();
        assert!(scene.stamps.all() > clock);
    }

    #[test]
    fn a_derived_record_belongs_to_its_node_s_row() {
        use crate::stamps::Group;
        let d = json!({"nodes": [
            {"id": "a", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10},
            {"id": "b", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10}],
            "pages": [{"id": "p", "width": 100, "height": 100, "nodeIds": ["a", "b"]}]});
        let mut scene = Scene::load(&d).unwrap();
        // Give `a` a record that differs from its node, as a variable fallback or a geometry effect would.
        let mut record = scene.node("a").unwrap().clone();
        record.set("width", json!(20));
        assert!(scene.store_resolved("a", Some(record.clone())));
        assert!(!scene.store_resolved("a", Some(record)), "the same record again is not a write");
        // Moving the node moves its record: that row's placement, and no other row.
        let built = scene.stamps.clock();
        scene.set_node_xy("a", 3.0, 4.0);
        assert!(!scene.stamps.current(0, Group::Placement, built));
        assert!(scene.stamps.current(0, Group::Content, built) && scene.stamps.current(1, Group::Placement, built));
        assert!(scene.stamps.all() <= built);
        // The position it holds, spelled as an integer, is still the position it holds.
        let mut whole = scene.node("b").unwrap().clone();
        whole.set("x", json!(4));
        scene.apply_node_edits(vec![whole], &[], &[], None).unwrap();
        let clock = scene.stamps.clock();
        scene.set_node_xy("b", 4.0, 0.0);
        assert_eq!(scene.stamps.clock(), clock);
        assert_eq!(scene.node("b").unwrap().json_field("x"), Some(&json!(4)));
        // So is a coordinate left out, which reads as zero: nothing is written and no field appears.
        let d = json!({"nodes": [
            {"id": "no_y", "type": "rectangle", "x": 4, "width": 10, "height": 10},
            {"id": "neither", "type": "rectangle", "width": 10, "height": 10}],
            "pages": [{"id": "p", "width": 100, "height": 100, "nodeIds": ["no_y", "neither"]}]});
        let mut defaults = Scene::load(&d).unwrap();
        let (clock, before) = (defaults.stamps.clock(), defaults.doc.nodes.clone());
        defaults.set_node_xy("no_y", 4.0, 0.0);
        defaults.set_node_xy("neither", 0.0, 0.0);
        assert_eq!(defaults.stamps.clock(), clock);
        assert_eq!(defaults.doc.nodes, before);
        defaults.set_node_xy("neither", 0.0, 1.0);
        assert!(defaults.stamps.clock() > clock);
        // A mutable borrow of the record stamps the row, like a mutable borrow of the node.
        let built = scene.stamps.clock();
        assert!(scene.resolved_mut("a").is_some());
        assert!(!scene.stamps.current(0, Group::Content, built) && scene.stamps.current(1, Group::Content, built));
        assert!(scene.stamps.all() <= built);
        // Re-deriving drops the record (a plain rectangle has none): the row is stamped once, then nothing is written.
        let built = scene.stamps.clock();
        assert!(scene.re_resolve_node("a"));
        assert!(!scene.stamps.current(0, Group::Content, built) && scene.stamps.all() <= built);
        let clock = scene.stamps.clock();
        assert!(scene.re_resolve_node("a"));
        assert_eq!(scene.stamps.clock(), clock);
    }

    #[test]
    fn an_edit_is_as_wide_as_what_it_re_derives_not_as_what_the_scene_holds() {
        use crate::stamps::Group;
        let square = |id: &str, x: f64, w: f64| {
            json!({"id": id, "type": "path", "parentId": "b", "x": x, "y": x, "points": [
                {"type": "move", "x": 0, "y": 0}, {"type": "line", "x": w, "y": 0},
                {"type": "line", "x": w, "y": w}, {"type": "line", "x": 0, "y": w}], "closed": true})
        };
        let d = json!({"nodes": [
            {"id": "b", "type": "boolean", "x": 10, "y": 10, "booleanOperation": "subtract", "children": ["s1", "s2"],
             "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#7c3aed"}}]},
            square("s1", 0.0, 100.0), square("s2", 25.0, 50.0),
            {"id": "r", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10}],
            "pages": [], "canvasRootNodeIds": ["b", "r"]});
        let mut scene = Scene::load(&d).unwrap();
        let (r, s1) = (scene.node_index["r"] as u32, scene.node_index["s1"] as u32);
        // The rectangle feeds nothing: the boolean elsewhere in the scene does not widen its edit.
        let built = scene.stamps.clock();
        let mut rect = scene.node("r").unwrap().clone();
        rect.set("x", json!(5));
        scene.apply_node_edits(vec![rect], &[], &[], None).unwrap();
        assert!(scene.stamps.all() <= built, "an independent edit is attributed to its row");
        assert!(!scene.stamps.current(r, Group::Placement, built));
        assert!(scene.stamps.current(s1, Group::Content, built));
        // An operand feeds the boolean's solved path: its edit re-derives another record, so nothing is attributed.
        let built = scene.stamps.clock();
        let mut operand = scene.node("s2").unwrap().clone();
        operand.set("x", json!(80));
        scene.apply_node_edits(vec![operand], &[], &[], None).unwrap();
        assert!(scene.stamps.all() > built);
    }

    #[test]
    fn every_scene_writer_advances_the_clock() {
        use crate::stamps::Group;
        let d = json!({"nodes": [
            {"id": "a", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10},
            {"id": "b", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10}],
            "pages": [{"id": "p", "width": 100, "height": 100, "nodeIds": ["a", "b"]}]});
        let mut scene = Scene::load(&d).unwrap();
        // A plain whole-node edit stamps its own row (it may have changed any field) and no other row.
        let built = scene.stamps.clock();
        let mut a = scene.node("a").unwrap().clone();
        a.set("x", json!(5));
        scene.apply_node_edits(vec![a], &[], &[], None).unwrap();
        assert!([Group::Placement, Group::Paint, Group::Content].into_iter().all(|g| !scene.stamps.current(0, g, built)));
        assert!([Group::Placement, Group::Paint, Group::Content].into_iter().all(|g| scene.stamps.current(1, g, built)));
        assert!(scene.stamps.all() <= built, "a plain edit is attributed to its row");
        // A typed placement write stamps placement alone.
        let built = scene.stamps.clock();
        scene.write_placement(0, None, crate::value::Number::from_f64(0.75));
        assert!(!scene.stamps.current(0, Group::Placement, built));
        assert!(scene.stamps.current(0, Group::Content, built) && scene.stamps.current(0, Group::Paint, built));
        // A typed placement write and a mutable borrow stamp their row.
        let built = scene.stamps.clock();
        let row = scene.row_of("b").unwrap();
        scene.write_placement(row, None, crate::value::Number::from_f64(0.5));
        scene.node_mut("a");
        assert!(!scene.stamps.current(1, Group::Placement, built) && !scene.stamps.current(0, Group::Content, built));
        // An unrestricted write may have moved or repainted the node: what was derived from its placement or its
        // paint is stale too, whether it came through a mutable borrow or a whole-node edit.
        let built = scene.stamps.clock();
        scene.node_mut("b").unwrap().set("x", json!(7));
        assert!([Group::Placement, Group::Paint, Group::Content].into_iter().all(|g| !scene.stamps.current(1, g, built)));
        let built = scene.stamps.clock();
        let mut moved = scene.node("a").unwrap().clone();
        moved.set("x", json!(9));
        scene.apply_node_edits(vec![moved], &[], &[], None).unwrap();
        assert!([Group::Placement, Group::Paint, Group::Content].into_iter().all(|g| !scene.stamps.current(0, g, built)));
        // No write: an empty typed write, the same position again, a refused edit, an empty batch.
        let clock = scene.stamps.clock();
        scene.write_placement(0, None, None);
        scene.set_node_xy("a", 9.0, 0.0);
        scene.set_node_xy("a", 9.0, 0.0);
        let after_first_xy = scene.stamps.clock();
        scene.set_node_xy("a", 9.0, 0.0);
        assert_eq!(scene.stamps.clock(), after_first_xy, "the same position again writes nothing");
        assert!(scene.apply_node_edits(vec![RawNode::from_value(json!({"type": "rectangle"})).unwrap()], &[], &[], None).is_err());
        assert!(scene.apply_node_edits(vec![], &[], &[("no such page".to_string(), vec![])], None).is_err());
        scene.apply_node_edits(vec![], &[], &[], None).unwrap();
        assert_eq!(scene.stamps.clock(), after_first_xy);
        assert!(after_first_xy >= clock);
        // A row write that fails to find its row changes nothing.
        let built = scene.stamps.clock();
        assert!(scene.write_row(7, |_| {}).is_err());
        assert_eq!(scene.stamps.clock(), built);
        // Structure is not a field of a row: the write is refused.
        let row = scene.row_of("b").unwrap();
        assert!(scene.write_row(row, |node| node.set("children", json!([]))).is_err());
        assert!(scene.node("b").unwrap().json_field("children").is_none());
        // A row write that sets nothing writes nothing.
        let built = scene.stamps.clock();
        scene.write_row(row, |_| {}).unwrap();
        assert_eq!(scene.stamps.clock(), built);
        // A page's background and a structural edit dirty everything.
        for change in 0..2 {
            let built = scene.stamps.clock();
            match change {
                0 => { scene.reexpand_page_background("p"); }
                _ => { scene.apply_node_edits(vec![], &["b".to_string()], &[], None).unwrap(); }
            }
            assert!(scene.stamps.all() > built, "change {change}");
            assert!(!scene.stamps.current(0, Group::Content, built), "change {change}");
        }
        assert_eq!(scene.stamps.len(), 1);
        // A writer with nothing to write leaves the clock where it was: an unknown page, node or record, no
        // pending fonts, no booleans, no envelopes, the same text-flow index again.
        let clock = scene.stamps.clock();
        assert!(scene.page_mut("no such page").is_none());
        assert!(scene.resolved_mut("no such node").is_none());
        assert!(scene.transient_mut("no such node").is_none());
        assert!(scene.reexpand_page_background("no such page").is_none());
        assert!(matches!(scene.reexpand_transient_host("no such node"), Ok(None)));
        assert!(matches!(scene.reexpand_transient_host("a"), Ok(None)), "a rectangle expands into nothing");
        assert!(!scene.re_resolve_node("no such node"));
        assert!(scene.re_resolve_node("a"), "a plain rectangle re-derives to the record it has: none");
        assert_eq!(scene.apply_node_edits(vec![], &["no such node".to_string()], &[], None).unwrap(), NodeEditOutcome::default());
        scene.prune_resolved("a");
        assert!(scene.retry_pending_text_warps().is_empty());
        assert!(scene.retry_text_booleans().is_empty());
        assert!(scene.retry_pending_envelopes().is_empty());
        assert!(scene.compute_envelopes().is_empty());
        assert!(matches!(scene.refit_auto_tables(), Ok(changed) if changed.is_empty()));
        scene.refresh_text_flow_index();
        assert_eq!(scene.stamps.clock(), clock);
        // A replacement scene continues the clock; it never goes back.
        let mut next = Scene::load(&d).unwrap();
        let moved = scene.subscribe(Group::Placement);
        next.continue_stamps_from(&scene);
        assert!(next.stamps().clock() > scene.stamps().clock());
        next.write_placement(0, None, crate::value::Number::from_f64(0.25));
        assert_eq!(next.stamps().dirty(moved), [0]);
        // The accessors that do hand something out mark first.
        assert!(scene.page_mut("p").is_some());
        assert!(scene.stamps.clock() > clock && scene.stamps.all() > clock);
    }

    #[test]
    fn cyclic_hierarchy_is_refused_on_load() {
        let doc = crate::value::json!({"pages":[{"id":"p","nodeIds":["g"]}],"nodes":[
            {"id":"g","type":"group","children":["h"]},
            {"id":"h","type":"group","children":["h"]}
        ]});
        assert!(Scene::load_str(&doc.to_string()).is_err(), "cyclic document was accepted");
    }

    #[test]
    fn a_design_document_draws_its_active_page() {
        let rect = |id: &str| json!({"id": id, "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10});
        let doc = json!({
            "nodes": [rect("a"), rect("b"), rect("c")],
            "pages": [],
            "canvases": [
                {"id": "p1", "name": "Page 1", "rootNodeIds": ["a"]},
                {"id": "p2", "name": "Page 2", "rootNodeIds": ["b", "c"]}
            ]
        });
        assert_eq!(Scene::load(&doc).unwrap().doc.canvas_root_node_ids, vec!["a".to_string()]);
        let mut second = doc.clone();
        second["activeCanvasId"] = json!("p2");
        assert_eq!(Scene::load(&second).unwrap().doc.canvas_root_node_ids, vec!["b".to_string(), "c".to_string()]);
        // An explicit root list (the editor's feed) wins.
        let mut fed = doc.clone();
        fed["canvasRootNodeIds"] = json!(["c"]);
        assert_eq!(Scene::load(&fed).unwrap().doc.canvas_root_node_ids, vec!["c".to_string()]);
    }

    #[test]
    fn a_document_copy_shares_its_brush_and_pattern_tables_until_one_is_replaced() {
        let doc = hydrate_document(&json!({
            "nodes": [], "pages": [],
            "brushes": [{"id": "nib", "type": "calligraphic", "angle": 30}, {"name": "no id"}],
            "patterns": [{"id": "dots", "width": 4, "height": 4, "nodes": []}]
        }));
        assert_eq!(doc.brushes.len(), 1, "a brush without an id is dropped");
        assert_eq!(doc.brush_hashes.len(), 1);
        let mut copy = doc.clone();
        assert!(std::sync::Arc::ptr_eq(&doc.brushes, &copy.brushes));
        assert!(std::sync::Arc::ptr_eq(&doc.patterns, &copy.patterns));
        copy.brushes = std::sync::Arc::new(vec![json!({"id": "nib", "type": "calligraphic", "angle": 60})]);
        assert_eq!(doc.brush("nib").and_then(|b| b.get("angle")), Some(&json!(30)));
        assert_eq!(copy.brush("nib").and_then(|b| b.get("angle")), Some(&json!(60)));
        assert_eq!(copy.pattern("dots"), doc.pattern("dots"));
    }
    use std::collections::{BTreeMap, BTreeSet};

    fn doc() -> Value {
        json!({
            "nodes": [
                {"id":"r1","type":"rectangle","x":10.0,"y":10.0,"width":50.0,"height":50.0,"rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ff0000"}}]},
                {"id":"g1","type":"group","x":100.0,"y":100.0,"scaleX":2.0,"scaleY":1.0,"children":["r2"]},
                {"id":"r2","type":"rectangle","x":5.0,"y":5.0,"width":10.0,"height":10.0,"rotation":0,"scale":1},
                {"id":"old","type":"text","x":0,"y":0},
                {"id":"hid","type":"rectangle","x":0,"y":0,"width":1,"height":1,"rotation":0,"scale":1,"hidden":true},
                {"id":"zer","type":"rectangle","x":0,"y":0,"width":1,"height":1,"rotation":0,"scale":1,"opacity":0}
            ],
            "pages": [
                {"id":"p1","name":"Page 1","width":1080.0,"height":1080.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ffffff"}}],
                 "nodeIds":["r1","g1","r2","hid","zer"]}
            ]
        })
    }

    /// Typed placement writes leave the scene `apply_node_edits` of the same fields leaves, for every node they
    /// admit, over a random sequence; nodes with derived records are not admitted.
    #[test]
    fn typed_placement_writes_match_node_edits() {
        let source = json!({"nodes":[
            {"id":"g","type":"group","x":0,"y":0,"rotation":0,"scaleX":1,"scaleY":1,"opacity":1,"children":["h","t"]},
            {"id":"h","type":"group","x":3,"y":4,"children":["p"]},
            {"id":"p","type":"path","x":0.125,"y":-0.375,"rotation":17.9,"scaleX":0.3,"scaleY":1.7,"opacity":0.5,
             "points":[{"type":"move","x":0,"y":0},{"type":"line","x":11,"y":13},{"type":"line","x":0,"y":13}],
             "closed":true,"fills":[{"id":"f","paint":{"type":"solid","color":"#ff0000"}}]},
            {"id":"t","type":"richtext","x":1,"y":2,"width":40,"height":20},
            {"id":"r","type":"rectangle","x":0,"y":0,"width":4,"height":4}],
            "pages":[{"id":"page","width":128,"height":128,"nodeIds":["g","r"]}]});
        let mut typed = Scene::load(&source).unwrap();
        let mut edited = Scene::load(&source).unwrap();
        assert!(!typed.placeable(typed.row_of("r").unwrap()), "only path, group and text nodes are admitted");
        assert!(typed.row_of("missing").is_none());
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed >> 11) as f64 / (1u64 << 53) as f64 };
        for step in 0..200 {
            let id = ["g", "h", "p", "t"][step % 4];
            let slot = typed.row_of(id).unwrap();
            assert!(typed.placeable(slot));
            // A host keys some values as f32 and some as f64; each is the number its JSON edit would carry.
            let f32n = |v: f64| match Value::from(v as f32) { Value::Number(n) => n, _ => unreachable!() };
            let f64n = |v: f64| crate::value::Number::from_f64(v).unwrap();
            let placement = NodePlacement { x: f32n(next() * 200. - 100.), y: f32n(next()),
                rotation: f64n(next() * 720. - 360.), scale_x: f32n(next() * 4. - 2.), scale_y: f32n(next() * 3. + 0.25) };
            let transform = (step % 3 != 0).then(|| placement.clone());
            let opacity = (step % 3 != 1).then(|| f64n(next()));
            typed.write_placement(slot, transform.clone(), opacity.clone());
            let mut node = edited.node(id).unwrap().clone();
            if let Some(t) = transform {
                for (key, value) in [("x", t.x), ("y", t.y), ("rotation", t.rotation), ("scaleX", t.scale_x), ("scaleY", t.scale_y)] {
                    node.set(key, Value::Number(value));
                }
            }
            if let Some(opacity) = opacity { node.set("opacity", Value::Number(opacity)); }
            let outcome = edited.apply_node_edits(vec![node], &[], &[], None).unwrap();
            assert!(!outcome.structural && outcome.transients_added.is_empty() && outcome.transients_removed.is_empty());
            for id in &edited.node_order {
                assert_eq!(typed.effective_node(id).unwrap().to_json_map(), edited.effective_node(id).unwrap().to_json_map(), "step {step} node {id}");
                assert_eq!(format!("{:?}", typed.node_canvas_affine_and_transform(id)), format!("{:?}", edited.node_canvas_affine_and_transform(id)), "step {step} affine {id}");
            }
            assert_eq!(typed.resolved_by_id.len(), edited.resolved_by_id.len());
        }
    }

    /// Fields written in place on a row leave the scene a node edit of the same fields leaves, with the row
    /// stamped as the caller names it; a row with derived records goes through the general edit.
    #[test]
    fn row_writes_match_node_edits() {
        use crate::stamps::Group;
        let source = json!({"nodes":[
            {"id":"g","type":"group","x":0,"y":0,"rotation":0,"scaleX":1,"scaleY":1,"opacity":1,"children":["p","t"]},
            {"id":"p","type":"path","x":0.125,"y":-0.375,"rotation":17.9,"scaleX":0.3,"scaleY":1.7,"opacity":0.5,
             "points":[{"type":"move","x":0,"y":0},{"type":"line","x":11,"y":13},{"type":"line","x":0,"y":13}],
             "closed":true,"fills":[{"id":"f","paint":{"type":"solid","color":"#ff0000"}}],"strokes":[]},
            {"id":"t","type":"richtext","x":1,"y":2,"width":40,"height":20},
            {"id":"tbl","type":"rectangle","x":0,"y":0,"width":4,"height":4}],
            "pages":[{"id":"page","width":128,"height":128,"nodeIds":["g","tbl"]}]});
        let mut typed = Scene::load(&source).unwrap();
        let mut edited = Scene::load(&source).unwrap();
        let writes: Vec<(&str, Vec<(&str, Value)>)> = vec![
            ("p", vec![("fills", json!([{"id":"f","paint":{"type":"solid","color":"#00ff00"},"opacity":0.25}]))]),
            ("p", vec![("strokes", json!([{"id":"s","paint":{"type":"solid","color":"#0000ff"},"width":3.5,"cap":"round"}])),
                ("closed", json!(false))]),
            ("p", vec![("points", json!([{"type":"move","x":1,"y":1},{"type":"line","x":5,"y":9},{"type":"line","x":-2,"y":4}]))]),
            ("g", vec![("effects", json!([{"id":"e","type":"drop-shadow","color":"#000000","blur":3,"offset":{"x":1,"y":2}}])),
                ("opacity", json!(0.5))]),
            ("t", vec![("vorkText", json!([0, 0, 7]))]),
            ("g", vec![("x", json!(12.5)), ("y", json!(-4))]),
            // A position and a paint in one write: both groups are stamped.
            ("p", vec![("x", json!(3.25)), ("fills", json!([{"id":"f","paint":{"type":"solid","color":"#0000ff"}}]))]),
            // Not a plain row: the general edit runs.
            ("tbl", vec![("width", json!(9))]),
        ];
        for (step, (id, fields)) in writes.into_iter().enumerate() {
            let row = typed.row_of(id).unwrap();
            let plain = typed.placeable(row);
            let built = typed.stamps.clock();
            // Every group a written field belongs to is stamped.
            let placement = fields.iter().any(|(key, _)| matches!(*key, "x" | "y"));
            let content = fields.iter().any(|(key, _)| !matches!(*key, "x" | "y"));
            let written = fields.clone();
            typed.write_row(row, move |node| for (key, value) in written { node.set(key, value); }).unwrap();
            let mut node = edited.node(id).unwrap().clone();
            for (key, value) in fields { node.set(key, value); }
            edited.apply_node_edits(vec![node], &[], &[], None).unwrap();
            // Plain before and after: nothing else is derived from what was written.
            let plain = plain && typed.placeable(row);
            for id in &edited.node_order {
                assert_eq!(typed.effective_node(id).unwrap().to_json_map(), edited.effective_node(id).unwrap().to_json_map(), "step {step} node {id}");
                assert_eq!(typed.node(id).unwrap().is_compact(), edited.node(id).unwrap().is_compact(), "step {step} points {id}");
                assert_eq!(format!("{:?}", typed.node_canvas_affine_and_transform(id)), format!("{:?}", edited.node_canvas_affine_and_transform(id)), "step {step} affine {id}");
            }
            assert_eq!(typed.resolved_by_id.len(), edited.resolved_by_id.len(), "step {step}");
            for (group, wrote) in [(Group::Placement, placement), (Group::Content, content)] {
                // (A row that is not plain dirties everything, in every group.)
                assert_eq!(!typed.stamps.current(row, group, built), wrote || !plain, "step {step}: {group:?}");
            }
            if plain {
                assert!(typed.stamps.all() <= built, "step {step}: a plain row's write dirties that row alone");
                for other in (0..typed.doc.nodes.len() as u32).filter(|other| *other != row) {
                    assert!(typed.stamps.current(other, Group::Placement, built) && typed.stamps.current(other, Group::Content, built), "step {step}: row {other} stands");
                }
            }
        }
    }

    /// A row write that makes a plain row draw through a derived record, or back, dirties everything: what
    /// `placeable` answered before no longer stands, and a consumer that keeps the answer sees that.
    #[test]
    fn a_row_write_across_the_plain_boundary_dirties_everything() {
        use crate::stamps::Group;
        let source = json!({"nodes":[
            {"id":"p","type":"path","x":1,"y":2,"rotation":0,"scaleX":1,"scaleY":1,"opacity":1,
             "points":[{"type":"move","x":0,"y":0},{"type":"line","x":40,"y":0},{"type":"line","x":40,"y":30},{"type":"line","x":0,"y":30}],
             "closed":true,"fills":[{"id":"f","paint":{"type":"solid","color":"#ff0000"}}],"strokes":[]},
            {"id":"q","type":"path","x":0,"y":0,
             "points":[{"type":"move","x":0,"y":0},{"type":"line","x":4,"y":0},{"type":"line","x":4,"y":3}],
             "closed":true,"fills":[{"id":"f","paint":{"type":"solid","color":"#00ff00"}}],"strokes":[]}],
            "pages":[{"id":"page","width":128,"height":128,"nodeIds":["p","q"]}]});
        let mut typed = Scene::load(&source).unwrap();
        let mut edited = Scene::load(&source).unwrap();
        let row = typed.row_of("p").unwrap();
        assert!(typed.placeable(row));
        // A live effect redraws the path's outline: the path now draws through a derived record.
        let roughen = json!([{"id": "r", "type": "roughen", "size": 5, "sizeMode": "absolute", "detail": 10, "points": "corner", "seed": 1}]);
        for effects in [roughen, json!([])] {
            let before = (typed.placeable(row), typed.stamps.clock());
            let written = effects.clone();
            typed.write_row(row, move |node| node.set("effects", written)).unwrap();
            let mut node = edited.node("p").unwrap().clone();
            node.set("effects", effects);
            edited.apply_node_edits(vec![node], &[], &[], None).unwrap();
            for id in &edited.node_order {
                assert_eq!(typed.effective_node(id).unwrap().to_json_map(), edited.effective_node(id).unwrap().to_json_map(), "node {id}");
            }
            assert_ne!(typed.placeable(row), before.0, "the write crossed the plain boundary");
            assert!(typed.stamps.all() > before.1, "and dirtied everything");
            assert_eq!(typed.resolved_by_id.contains_key("p"), !typed.placeable(row));
        }
    }

    /// A node's `skewX` shears its own box, and a sheared group shears its children:
    /// the accumulated affine is `parent(T·R·K·S) · child(T·R·K·S)`.
    #[test]
    fn accumulated_affine_carries_skew() {
        let v = json!({
            "nodes": [
                {"id":"r","type":"rectangle","x":10.0,"y":20.0,"width":100.0,"height":50.0,"rotation":0,"scale":1,"skewX":30.0},
                {"id":"g","type":"group","x":0.0,"y":0.0,"rotation":90.0,"scaleX":1.0,"scaleY":1.0,"skewX":-20.0,"children":["c"]},
                {"id":"c","type":"rectangle","x":5.0,"y":0.0,"width":10.0,"height":10.0,"rotation":0,"scale":1}
            ],
            "pages": [{"id":"p1","name":"P","width":500.0,"height":500.0,"nodeIds":["r","g"]}]
        });
        let scene = Scene::load(&v).unwrap();
        let (a, _) = scene.accumulated_affine("r").unwrap();
        let k = 30f64.to_radians().tan();
        // The bottom-left corner (0, 50) lands k·50 to the right of the origin.
        let bl = transform::apply_point((0.0, 50.0), a);
        assert!(
            (bl.0 - (10.0 + k * 50.0)).abs() < 1e-9 && (bl.1 - 70.0).abs() < 1e-9,
            "{bl:?}"
        );
        let (child, _) = scene.accumulated_affine("c").unwrap();
        let group = transform::from_trsk(
            Transform2D {
                x: 0.0,
                y: 0.0,
                rotation: 90.0,
                scale_x: 1.0,
                scale_y: 1.0,
            },
            -20.0,
        );
        let want = transform::apply_point((5.0, 10.0), group);
        let got = transform::apply_point((0.0, 10.0), child);
        assert!(
            (got.0 - want.0).abs() < 1e-9 && (got.1 - want.1).abs() < 1e-9,
            "{got:?} vs {want:?}"
        );
    }

    /// The owned load (the feed paths) builds the document the borrowed one
    /// does, with or without the `canvas_data` wrapper.
    #[test]
    fn load_owned_matches_load() {
        let mut v = doc();
        v["canvasRootNodeIds"] = json!(["zer"]);
        v["variables"] = json!([]);
        v["futureKey"] = json!(1);
        v["nodes"].as_array_mut().unwrap().push(json!({"id":"p","type":"path","x":0,"y":0,"points":[
            {"type":"move","x":0,"y":0},{"type":"line","x":1,"y":1,"incomingHandle":{"x":0,"y":0}}]}));
        v["nodes"].as_array_mut().unwrap().push(json!({"id":"odd","type":"sparkle"}));
        v["nodes"].as_array_mut().unwrap().push(json!(7));
        for input in [v.clone(), json!({ "canvas_data": v, "other": 1 })] {
            let a = Scene::load(extract_canvas_data(&input)).unwrap();
            let b = Scene::load_owned(extract_canvas_data_owned(input.clone())).unwrap();
            assert_eq!(format!("{:?}", a.doc), format!("{:?}", b.doc));
            assert_eq!(a.node_order, b.node_order);
            assert_eq!(a.parent_map, b.parent_map);
            // Streamed straight from the text: the same document.
            let c = Scene::load_str(&input.to_string()).unwrap();
            assert_eq!(format!("{:?}", b.doc), format!("{:?}", c.doc));
            assert_eq!(b.node_order, c.node_order);
            assert_eq!(b.parent_map, c.parent_map);
        }
        // Shapes that are not a document, or have no nodes array.
        for input in [json!(null), json!([1, 2]), json!("x"), json!({"nodes": {"a": 1}, "pages": 3}), json!({"canvas_data": null})] {
            let b = Scene::load_owned(extract_canvas_data_owned(input.clone())).unwrap();
            let c = Scene::load_str(&input.to_string()).unwrap();
            assert_eq!(format!("{:?}", b.doc), format!("{:?}", c.doc), "{input}");
        }
        assert!(Scene::load_str("{\"nodes\": [").is_err());
    }

    #[test]
    fn property_edits_preserve_compact_geometry() {
        let document = json!({"nodes": [{"id": "path", "type": "path", "closed": false,
            "points": [{"type": "move", "x": 0, "y": 0},
                       {"type": "line", "x": 30, "y": 20}]}],
            "pages": [{"id": "page", "nodeIds": ["path"]}]});
        let mut scene = Scene::load(&document).unwrap();
        let mut original = scene.node("path").unwrap().clone();
        assert!(original.compact());
        scene.apply_node_edits(vec![original.clone()], &[], &[], None).unwrap();
        for (field, value) in [("x", json!(25)), ("opacity", json!(0.5)), ("rotation", json!(45))] {
            let mut edited = scene.node("path").unwrap().clone();
            edited.set(field, value);
            let expected = edited.to_json_map();
            scene.apply_node_edits(vec![edited], &[], &[], None).unwrap();
            let installed = scene.node("path").unwrap();
            assert!(std::ptr::eq(original.compact_points().unwrap(), installed.compact_points().unwrap()),
                "property edits must retain the shared point allocation");
            assert_eq!(installed.to_json_map(), expected);
        }
        let mut edited = scene.node("path").unwrap().clone();
        edited.set("points", json!([{"type": "move", "x": 7, "y": 9}]));
        scene.apply_node_edits(vec![edited], &[], &[], None).unwrap();
        assert_ne!(scene.node("path").unwrap().to_json_map()["points"], original.to_json_map()["points"]);
    }

    /// `apply_node_edits` leaves the scene as `Scene::load` of the edited
    /// document builds it — through the sanitizers (the fixture's page lists a
    /// grouped `r2`), an unknown type, a regroup and a removal. (The engine's
    /// `residency_patch_tests` run the long sequences against plans too.)
    #[test]
    fn apply_node_edits_matches_a_cold_load() {
        fn assert_same(a: &Scene, b: &Scene) {
            let nodes = |s: &Scene| -> BTreeMap<String, RawNode> {
                s.doc.nodes.iter().map(|n| (n.id().to_string(), n.clone())).collect()
            };
            assert_eq!(nodes(a), nodes(b));
            assert_eq!(a.doc.pages, b.doc.pages);
            assert_eq!(a.parent_map, b.parent_map);
            assert_eq!(a.children_by_parent, b.children_by_parent);
            assert_eq!(a.page_id_by_node, b.page_id_by_node);
            assert_eq!(a.resolved_by_id, b.resolved_by_id);
            assert_eq!(a.transient_by_id, b.transient_by_id);
            assert_eq!(a.transient_by_parent, b.transient_by_parent);
            assert_eq!(a.transient_page_of, b.transient_page_of);
        }
        let mut d = doc();
        let mut scene = Scene::load(&d).unwrap();

        // Move r1 into g1 (behind r2), add a node of an unknown type, drop `zer`.
        let r1 = {
            let mut n = d["nodes"][0].clone();
            n["parentId"] = json!("g1");
            n
        };
        d["nodes"][0] = r1.clone();
        d["nodes"][1]["children"] = json!(["r1", "r2"]);
        d["nodes"].as_array_mut().unwrap().retain(|n| n["id"] != "zer");
        d["pages"][0]["nodeIds"] = json!(["g1", "r2", "hid"]);
        let out = scene
            .apply_node_edits(
                vec![
                    RawNode::from_value(r1).unwrap(),
                    RawNode::from_value(d["nodes"][1].clone()).unwrap(),
                    RawNode::from_value(json!({"id": "odd", "type": "sparkle"})).unwrap(),
                ],
                &["zer".to_string()],
                &[("p1".to_string(), vec!["g1".into(), "r2".into(), "hid".into()])],
                None,
            )
            .unwrap();
        assert!(out.structural);
        assert_eq!(out.removed, vec!["zer".to_string()]);
        assert_same(&scene, &Scene::load(&d).unwrap());

        // A property-only replace keeps the indexes and is not structural.
        d["nodes"][2]["width"] = json!(99.0);
        let out = scene
            .apply_node_edits(vec![RawNode::from_value(d["nodes"][2].clone()).unwrap()], &[], &[], None)
            .unwrap();
        assert!(!out.structural);
        assert_eq!(out.written, vec!["r2".to_string()]);
        assert_same(&scene, &Scene::load(&d).unwrap());

        // Refused before mutating: an unknown page.
        let before = scene.doc.nodes.clone();
        assert!(scene
            .apply_node_edits(vec![], &["r2".to_string()], &[("nope".to_string(), vec![])], None)
            .is_err());
        assert_eq!(scene.doc.nodes, before);
    }

    /// A boolean node added to (or edited in) a loaded scene draws its result path, as a cold
    /// load of the edited document does: the patch lane used to leave its record a raw
    /// `boolean`, which no draw path handles, so a boolean made in an open document stayed
    /// invisible on the canvas until the document was reopened.
    #[test]
    fn apply_node_edits_solves_boolean_nodes() {
        let square = |id: &str, parent: &str, x: f64, w: f64| {
            json!({"id": id, "type": "path", "parentId": parent, "x": x, "y": x, "points": [
                {"type": "move", "x": 0, "y": 0}, {"type": "line", "x": w, "y": 0},
                {"type": "line", "x": w, "y": w}, {"type": "line", "x": 0, "y": w}], "closed": true})
        };
        let boolean = |op: &str| {
            json!({"id": "b", "type": "boolean", "x": 10, "y": 10, "booleanOperation": op,
                   "children": ["s1", "s2"], "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#7c3aed"}}]})
        };
        let mut d = json!({"nodes": [], "pages": [], "canvasRootNodeIds": []});
        let mut scene = Scene::load(&d).unwrap();

        d["nodes"] = json!([boolean("subtract"), square("s1", "b", 0.0, 100.0), square("s2", "b", 25.0, 50.0)]);
        d["canvasRootNodeIds"] = json!(["b"]);
        let upserts: Vec<RawNode> = d["nodes"].as_array().unwrap().iter().map(|n| RawNode::from_value(n.clone()).unwrap()).collect();
        scene.apply_node_edits(upserts, &[], &[], Some(&["b".to_string()])).unwrap();
        let cold = Scene::load(&d).unwrap();
        assert_eq!(scene.effective_node("b").map(RawNode::node_type), Some("path"));
        assert_eq!(scene.resolved_by_id, cold.resolved_by_id);

        // A property edit on the boolean and one on an operand re-solve it too.
        d["nodes"][0] = boolean("union");
        scene.apply_node_edits(vec![RawNode::from_value(d["nodes"][0].clone()).unwrap()], &[], &[], None).unwrap();
        assert_eq!(scene.resolved_by_id, Scene::load(&d).unwrap().resolved_by_id);
        d["nodes"][2] = square("s2", "b", 80.0, 50.0);
        scene.apply_node_edits(vec![RawNode::from_value(d["nodes"][2].clone()).unwrap()], &[], &[], None).unwrap();
        assert_eq!(scene.resolved_by_id, Scene::load(&d).unwrap().resolved_by_id);
    }

    /// A page-less Design document: adding, removing and reordering canvas-root
    /// nodes through the patch lane leaves the scene as a cold load of the
    /// edited document leaves it.
    #[test]
    fn apply_node_edits_canvas_root_matches_a_cold_load() {
        let rect = |id: &str, x: f64| json!({"id": id, "type": "rectangle", "x": x, "y": 0.0, "width": 10.0, "height": 10.0});
        let mut d = json!({
            "nodes": [rect("a", 0.0), rect("b", 20.0), rect("c", 40.0)],
            "pages": [],
            "canvasRootNodeIds": ["a", "b", "c"],
        });
        let mut scene = Scene::load(&d).unwrap();

        // Add `n` on top, remove `b`, move `a` above `c`.
        d["nodes"].as_array_mut().unwrap().retain(|n| n["id"] != "b");
        d["nodes"].as_array_mut().unwrap().push(rect("n", 60.0));
        d["canvasRootNodeIds"] = json!(["c", "a", "n"]);
        let roots: Vec<NodeId> = vec!["c".into(), "a".into(), "n".into()];
        let out = scene
            .apply_node_edits(vec![RawNode::from_value(rect("n", 60.0)).unwrap()], &["b".to_string()], &[], Some(&roots))
            .unwrap();
        assert!(out.structural);
        let cold = Scene::load(&d).unwrap();
        assert_eq!(scene.doc.canvas_root_node_ids, cold.doc.canvas_root_node_ids);
        assert_eq!(scene.doc.canvas_root_node_ids, roots);
        assert_eq!(scene.node_order, cold.node_order);
        assert_eq!(scene.parent_map, cold.parent_map);
        assert_eq!(scene.children_by_parent, cold.children_by_parent);
        assert_eq!(scene.resolved_by_id, cold.resolved_by_id);

        // A reorder alone is structural; the same list again is not.
        let reordered: Vec<NodeId> = vec!["n".into(), "c".into(), "a".into()];
        assert!(scene.apply_node_edits(vec![], &[], &[], Some(&reordered)).unwrap().structural);
        assert!(!scene.apply_node_edits(vec![], &[], &[], Some(&reordered)).unwrap().structural);
        // An id that isn't a node is sanitized away, as a cold load drops it.
        let stray: Vec<NodeId> = vec!["n".into(), "ghost".into(), "c".into(), "a".into()];
        scene.apply_node_edits(vec![], &[], &[], Some(&stray)).unwrap();
        assert_eq!(scene.doc.canvas_root_node_ids, reordered);
    }

    /// `is_node_isolated` is the Rust twin of `isExportNodeIsolated`
    /// (`renderers/webgpu/exportExclusion.ts:36-47`), the filter per-element
    /// alpha plates are built on. The three properties that make a plate
    /// correct: an EMPTY set isolates nothing (so `--isolate` unset leaves the
    /// plan untouched), isolating a GROUP keeps its descendants (a group plate
    /// must contain the group's contents, not an empty frame), and a sibling
    /// outside the set is filtered (a plate must not contain its neighbours).
    #[test]
    fn is_node_isolated_matches_the_browser_isolation_filter() {
        let scene = Scene::load(&doc()).unwrap();

        // Empty set = "no isolation requested" → everything renders.
        let none: HashSet<&str> = HashSet::default();
        for id in ["r1", "g1", "r2"] {
            assert!(
                scene.is_node_isolated(id, &none),
                "an empty isolate set must not filter {id}"
            );
        }

        // Isolating a leaf keeps only that leaf.
        let leaf: HashSet<&str> = ["r1"].into_iter().collect();
        assert!(scene.is_node_isolated("r1", &leaf));
        assert!(
            !scene.is_node_isolated("g1", &leaf),
            "a sibling group must be filtered out of a leaf's plate"
        );
        assert!(
            !scene.is_node_isolated("r2", &leaf),
            "another element's child must be filtered out of a leaf's plate"
        );

        // Isolating a GROUP keeps the group and every descendant — the whole
        // point of per-GROUP plates.
        let group: HashSet<&str> = ["g1"].into_iter().collect();
        assert!(scene.is_node_isolated("g1", &group));
        assert!(
            scene.is_node_isolated("r2", &group),
            "isolating a group must keep its children (r2 is g1's child)"
        );
        assert!(
            !scene.is_node_isolated("r1", &group),
            "a node outside the isolated group must be filtered"
        );

        // Multi-element selection is a union, and an unknown id isolates
        // nothing extra (it simply never matches).
        let multi: HashSet<&str> = ["r1", "g1", "nope"].into_iter().collect();
        for id in ["r1", "g1", "r2"] {
            assert!(scene.is_node_isolated(id, &multi), "{id} is in the union");
        }
    }

    /// Cyclic parent data must resolve to "filtered", never an infinite loop:
    /// the walk is the same shape as `ancestor_opacity_product`'s, whose guard
    /// this mirrors. A hang here would wedge an export worker, not fail it.
    #[test]
    fn is_node_isolated_terminates_on_cyclic_parents() {
        let mut scene = Scene::load(&doc()).unwrap();
        scene.parent_map.insert("r1".into(), "g1".into());
        scene.parent_map.insert("g1".into(), "r1".into());
        let isolate: HashSet<&str> = ["nope"].into_iter().collect();
        assert!(!scene.is_node_isolated("r1", &isolate));
    }

    #[test]
    fn page_fill_variable_ref_survives_hydration_even_with_an_empty_variable_id() {
        // review-guidelines 6.2. Hydration keeps a page's `fills` verbatim, so a
        // variable reference in a solid background survives it (even with an
        // empty or absent `variableId`) and reaches the sink, which misses it to
        // '#000000' — as the browser does. Rewriting it to a white default
        // would render the same page white natively and black on web.
        //
        // Driven through `Scene::load`/`hydrate_document` ON PURPOSE: the
        // expansion.rs page tests build `RawPage` directly.
        for color in [
            json!({"type":"variable","variableId":""}),
            json!({"type":"variable"}),
            json!({"type":"variable","variableId":"v_missing"}),
        ] {
            let fills = json!([{"id":"fill-0","paint":{"type":"solid","color":color}}]);
            let s = Scene::load(&json!({
                "nodes": [],
                "pages": [{"id":"p1","name":"P","width":10.0,"height":10.0,"nodeIds":[],"fills":fills}],
            })).unwrap();
            assert_eq!(
                s.doc.pages[0].0.get("fills"),
                Some(&fills),
                "hydration must preserve the reference so the sink can miss it to \
                 '#000000'; rewriting it to the white default diverges from TS"
            );
        }
    }

    #[test]
    fn page_without_fills_hydrates_without_a_default_background() {
        // Pages store `fills` (the white default of a colour page is written by
        // the document codemod, 024-paint-stacks `transformPage`); hydration no
        // longer normalises a page background. Replaces
        // `page_fill_non_reference_object_still_takes_the_white_default`, which
        // pinned the old flat `fill` admission rule [Q5].
        let s = Scene::load(&json!({
            "nodes": [],
            "pages": [{"id":"p1","name":"P","width":10.0,"height":10.0,"nodeIds":[],"fills":[]}],
        })).unwrap();
        assert_eq!(s.doc.pages[0].0.get("fills"), Some(&json!([])));
        assert!(crate::paint::top_page_fill(&s.doc.pages[0]).is_none());
    }

    #[test]
    fn invalid_node_types_dropped_silently() {
        // Scene.ts `nodes` setter — legacy 'text' skipped, no unknown verdict.
        let s = Scene::load(&doc()).unwrap();
        assert!(s.node("old").is_none());
        assert!(s.doc.unknown_node_types.is_empty());
    }

    #[test]
    fn canvas_root_ids_sanitized_against_page_and_group_membership() {
        // The engine renders canvas-root nodes, so a malformed feed
        // that lists a page-parented or grouped id (or a missing/duplicate id)
        // in canvasRootNodeIds must be filtered, or the node would render twice.
        let s = Scene::load(&json!({
            "layoutMode": "freeform",
            "nodes": [
                {"id":"float","type":"rectangle","x":900.0,"y":0.0,"width":10.0,"height":10.0,"rotation":0,"scale":1},
                {"id":"onpage","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,"rotation":0,"scale":1},
                {"id":"grp","type":"group","x":0.0,"y":0.0,"children":["child"]},
                {"id":"child","type":"rectangle","x":1.0,"y":1.0,"width":5.0,"height":5.0,"rotation":0,"scale":1}
            ],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                       "nodeIds":["onpage","grp"],"freeformX":0.0,"freeformY":0.0}],
            // Dirty list: a genuine float, a page-owned id, a grouped child, a
            // missing id, and a duplicate of the float.
            "canvasRootNodeIds": ["float","onpage","child","ghost","float"]
        })).unwrap();
        assert_eq!(
            s.doc.canvas_root_node_ids,
            vec!["float".to_string()],
            "only the genuine floating node survives sanitization"
        );
        // The canvas display list then renders exactly that node, once.
        let dl = crate::display_list::build_display_list_for_canvas(&s);
        let ids: Vec<&str> = dl.items.iter().map(|i| i.node_id.as_str()).collect();
        assert_eq!(ids, vec!["float"]);
    }

    #[test]
    fn unknown_node_types_flagged() {
        let s = Scene::load(&json!({
            "nodes": [{"id":"f","type":"hologram","x":0,"y":0}],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["f"]}]
        })).unwrap();
        assert_eq!(
            s.doc.unknown_node_types,
            vec![("f".to_string(), "hologram".to_string())]
        );
    }

    #[test]
    fn set_node_xy_moves_the_rendered_position() {
        // The native drag fast-path (moda-desktop NudgeNodes -> set_node_xy) must
        // move where the node RENDERS, not just its canonical record: the plan and
        // display list read `effective_node` (resolved ?? canonical ?? transient),
        // so the resolved cache must update too — otherwise a nudged node would
        // stay put on screen. Pins the engine half of the fast-path without a GPU.
        let mut s = Scene::load(&doc()).unwrap();
        assert_eq!(s.effective_node("r1").unwrap().f64("x"), Some(10.0));
        assert_eq!(s.effective_node("r1").unwrap().f64("y"), Some(10.0));

        s.set_node_xy("r1", 250.0, 175.0);

        // effective (what the renderer reads) reflects the move...
        assert_eq!(s.effective_node("r1").unwrap().f64("x"), Some(250.0));
        assert_eq!(s.effective_node("r1").unwrap().f64("y"), Some(175.0));
        // ...and so does canonical, so a later full re-feed / selectors agree.
        assert_eq!(s.node("r1").unwrap().f64("x"), Some(250.0));
        assert_eq!(s.node("r1").unwrap().f64("y"), Some(175.0));

        // A grouped child nudges in its parent-local space (engine applies the
        // group transform at render), so the resolved entry must update too.
        s.set_node_xy("r2", 9.0, 9.0);
        assert_eq!(s.effective_node("r2").unwrap().f64("x"), Some(9.0));
        assert_eq!(s.effective_node("r2").unwrap().f64("y"), Some(9.0));

        // An unknown id is a silent no-op (defensive — stale frontend selection).
        s.set_node_xy("does-not-exist", 1.0, 2.0);
    }

    #[test]
    fn nested_table_transients_inherit_container_transform() {
        // Table cells are expanded in the TABLE'S PARENT'S coordinate space
        // (expansion.rs bakes table.x/y into cell coords), so a table nested in
        // a container needs the container's translate applied by the transform
        // walk — transform_parent_chain must step THROUGH the table host to the
        // container (parity with nodeTransformUtils.ts getParentChain). Before
        // the fix the chain broke at the table and nested tables rendered at
        // container-local coords in page space.
        let s = Scene::load(&json!({
            "nodes": [
                {"id":"card","type":"container","x":70.0,"y":648.0,"width":940.0,"height":276.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ffffff"}}],"children":["tbl"]},
                {"id":"tbl","type":"table","x":28.0,"y":68.0,"width":60.0,"height":40.0,
                 "rotation":0,"scale":1,
                 "rows":[{"size":20.0},{"size":20.0}],
                 "columns":[{"size":30.0},{"size":30.0}],
                 "cells":[
                    [{"id":"c00","textContent":"<p>a</p>"},{"id":"c01","textContent":"<p>b</p>"}],
                    [{"id":"c10","textContent":"<p>c</p>"},{"id":"c11","textContent":"<p>d</p>"}]
                 ]}
            ],
            "pages": [{"id":"p1","name":"P","width":1080.0,"height":1080.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                       "nodeIds":["card"]}]
        })).unwrap();

        // The chain steps through the table host to the container, without
        // pushing the table itself.
        assert_eq!(
            s.transform_parent_chain("c00_background"),
            vec!["card".to_string()]
        );

        // Cell bg canvas position = container (70, 648) + baked table-parent-
        // relative coords (28, 68).
        let affine = s.node_canvas_affine("c00_background").unwrap();
        let origin = transform::apply_point((0.0, 0.0), affine);
        assert_eq!(origin, (98.0, 716.0));

        // A page-root table keeps its previous (already correct) placement:
        // empty chain, baked coords are page coords.
        let s2 = Scene::load(&json!({
            "nodes": [
                {"id":"tbl","type":"table","x":28.0,"y":68.0,"width":60.0,"height":40.0,
                 "rotation":0,"scale":1,
                 "rows":[{"size":20.0}],
                 "columns":[{"size":30.0}],
                 "cells":[[{"id":"c00","textContent":"<p>a</p>"}]]}
            ],
            "pages": [{"id":"p1","name":"P","width":1080.0,"height":1080.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                       "nodeIds":["tbl"]}]
        })).unwrap();
        assert!(s2.transform_parent_chain("c00_background").is_empty());
        let affine2 = s2.node_canvas_affine("c00_background").unwrap();
        assert_eq!(transform::apply_point((0.0, 0.0), affine2), (28.0, 68.0));
    }

    #[test]
    fn grouped_children_removed_from_top_level() {
        // sanitizeUniquePageMembership (scene.sanitizers.ts:24-40): r2 has a
        // group parent → dropped from page.nodeIds.
        let s = Scene::load(&doc()).unwrap();
        let ids = s.doc.pages[0].node_ids();
        assert!(!ids.contains(&"r2".to_string()));
        assert!(ids.contains(&"g1".to_string()));
    }

    #[test]
    fn z_order_starts_with_page_bg_and_walks_groups() {
        // scene.selectors.ts:58-90
        let s = Scene::load(&doc()).unwrap();
        let order = s.flattened_z_order_for_page("p1");
        assert_eq!(
            order,
            vec![
                "__page_bg_p1".to_string(),
                "r1".to_string(),
                "g1".to_string(), // parent pushed before children
                "r2".to_string(),
                // hid (hidden) and zer (opacity 0) excluded (selectors.ts:69)
            ]
        );
    }

    #[test]
    fn z_order_first_on_top_reverses_an_auto_layout_frames_children() {
        // Figma's canvas stacking "First on top" (flexLayout.itemReverseZIndex): scene.selectors.ts
        // childrenInPaintOrder. Only the frame's own children flip; a nested frame keeps its order.
        let load = |reverse: bool| {
            Scene::load(&json!({
                "nodes": [
                    {"id":"f","type":"container","x":0.0,"y":0.0,"width":100.0,"height":40.0,"rotation":0,"scale":1,
                     "clipContent":false,"children":["a","b","inner"],
                     "flexLayout":{"flexDirection":"row","gap":-10.0,"padding":{"top":0,"right":0,"bottom":0,"left":0},
                                   "alignItems":"flex-start","justifyContent":"flex-start","itemReverseZIndex":reverse}},
                    {"id":"a","type":"rectangle","x":0.0,"y":0.0,"width":40.0,"height":40.0,"rotation":0,"scale":1},
                    {"id":"b","type":"rectangle","x":30.0,"y":0.0,"width":40.0,"height":40.0,"rotation":0,"scale":1},
                    {"id":"inner","type":"container","x":60.0,"y":0.0,"width":40.0,"height":40.0,"rotation":0,"scale":1,
                     "clipContent":false,"children":["c","d"]},
                    {"id":"c","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,"rotation":0,"scale":1},
                    {"id":"d","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,"rotation":0,"scale":1}
                ],
                "pages": [{"id":"p1","name":"P","width":400.0,"height":400.0,"nodeIds":["f"]}]
            })).unwrap()
        };
        let ids = |s: &Scene| s.flattened_z_order_for_page("p1")[1..].to_vec();
        assert_eq!(ids(&load(false)), vec!["f", "a", "b", "inner", "c", "d"]);
        assert_eq!(ids(&load(true)), vec!["f", "inner", "c", "d", "b", "a"]);
    }

    // Ancestor group/container opacity folds into descendants.

    #[test]
    fn ancestor_opacity_top_level_is_one() {
        let s = Scene::load(&doc()).unwrap();
        assert_eq!(s.ancestor_opacity_product("r1"), 1.0);
    }

    #[test]
    fn ancestor_opacity_multiplies_group_into_child() {
        let s = Scene::load(&json!({
            "nodes": [
                {"id":"g","type":"group","x":0,"y":0,"opacity":0.5,"children":["c"]},
                {"id":"c","type":"rectangle","x":0,"y":0,"width":10,"height":10,"rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#f00"}}]}
            ],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["g"]}]
        })).unwrap();
        assert_eq!(s.ancestor_opacity_product("c"), 0.5);
    }

    #[test]
    fn ancestor_opacity_multiplies_nested_parents() {
        let s = Scene::load(&json!({
            "nodes": [
                {"id":"outer","type":"group","x":0,"y":0,"opacity":0.5,"children":["inner"]},
                {"id":"inner","type":"group","x":0,"y":0,"opacity":0.5,"children":["c"]},
                {"id":"c","type":"rectangle","x":0,"y":0,"width":10,"height":10,"rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#f00"}}]}
            ],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["outer"]}]
        })).unwrap();
        assert_eq!(s.ancestor_opacity_product("c"), 0.25); // 0.5 (inner) × 0.5 (outer)
        assert_eq!(s.ancestor_opacity_product("inner"), 0.5); // inner sees only outer above it
    }

    #[test]
    fn ancestor_opacity_excludes_self_so_container_own_fill_not_double_dimmed() {
        let s = Scene::load(&json!({
            "nodes": [
                {"id":"ct","type":"container","x":0,"y":0,"width":50,"height":50,"opacity":0.5,"children":["c"]},
                {"id":"c","type":"rectangle","x":0,"y":0,"width":10,"height":10,"rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#f00"}}]}
            ],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["ct"]}]
        })).unwrap();
        assert_eq!(s.ancestor_opacity_product("ct"), 1.0); // own fill dimmed only by its own opacity
        assert_eq!(s.ancestor_opacity_product("c"), 0.5); // child inherits container opacity
    }

    #[test]
    fn ancestor_opacity_zero_short_circuits() {
        let s = Scene::load(&json!({
            "nodes": [
                {"id":"g","type":"group","x":0,"y":0,"opacity":0.0,"children":["c"]},
                {"id":"c","type":"rectangle","x":0,"y":0,"width":10,"height":10,"rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#f00"}}]}
            ],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["g"]}]
        })).unwrap();
        assert_eq!(s.ancestor_opacity_product("c"), 0.0);
    }

    #[test]
    fn ancestor_opacity_walks_through_table_host_to_group() {
        // A table inside a translucent group: its expanded transient children
        // must inherit the GROUP's opacity even though the table host itself is
        // not a parent node (its own opacity is already baked into the children).
        let s = Scene::load(&json!({
            "nodes": [
                {"id":"g","type":"group","x":0,"y":0,"opacity":0.5,"children":["tbl"]},
                {"id":"tbl","type":"table","x":0,"y":0,"rows":[50.0],"columns":[50.0]}
            ],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["g"]}]
        })).unwrap();
        let child_ids: Vec<String> = s
            .transient_parent_of
            .iter()
            .filter(|(_, parent)| parent.as_str() == "tbl")
            .map(|(id, _)| id.clone())
            .collect();
        assert!(
            !child_ids.is_empty(),
            "table should expand into transient children"
        );
        for id in child_ids {
            assert_eq!(
                s.ancestor_opacity_product(&id),
                0.5,
                "transient child {id} should inherit the enclosing group's opacity"
            );
        }
    }

    #[test]
    fn group_transform_composes() {
        // nodeTransformUtils.ts:251-311: child (5,5) in group at (100,100)
        // scaled 2x1 → world (110, 105).
        let s = Scene::load(&doc()).unwrap();
        let t = s.node_canvas_transform("r2").unwrap();
        assert_eq!((t.x, t.y), (110.0, 105.0));
        assert_eq!((t.scale_x, t.scale_y), (2.0, 1.0));
    }

    #[test]
    fn table_expansion_replaces_host_in_z_order() {
        let s = Scene::load(&json!({
            "nodes": [{
                "id":"tbl","type":"table","x":0.0,"y":0.0,"width":20.0,"height":10.0,
                "rotation":0,"scale":1,
                "rows":[{"size":10.0}],"columns":[{"size":20.0}],
                "cells":[[{"id":"c0","textContent":"<p>x</p>"}]],
                "columnStrokeConfig":"none","rowStrokeConfig":"none"
            }],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["tbl"]}]
        })).unwrap();
        let order = s.flattened_z_order_for_page("p1");
        assert_eq!(
            order,
            vec![
                "__page_bg_p1".to_string(),
                "c0_background".to_string(),
                "c0_content".to_string(),
            ]
        );
    }

    #[test]
    fn a_nodes_mode_generator_with_a_timed_output_draws_that_instead_of_its_children() {
        // Inside a playing composition the editor hands the generator its timed output: that replaces the design.
        let s = Scene::load(&json!({
            "nodes": [
                { "id": "g", "type": "generator", "x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0, "rotation": 0, "scale": 1,
                  "code": "b3:missing", "params": {}, "children": ["g/bar:a"],
                  "_generatorOutput": [{ "id": "g/bar:a@t", "type": "rectangle", "x": 1.0, "y": 2.0, "width": 9.0, "height": 9.0, "rotation": 0, "scale": 1 }] },
                { "id": "g/bar:a", "type": "rectangle", "x": 1.0, "y": 2.0, "width": 5.0, "height": 5.0, "rotation": 0, "scale": 1,
                  "genKey": "bar:a", "genHash": "0123456789abcdef" }
            ],
            "pages": [{"id":"p1","name":"P","width":1000.0,"height":1000.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["g"]}]
        })).unwrap();
        let order = s.flattened_z_order_for_page("p1");
        assert!(order.contains(&"g/bar:a@t".to_string()), "{order:?}");
        assert!(!order.contains(&"g/bar:a".to_string()), "the stored design is replaced: {order:?}");
    }

    #[test]
    fn a_nodes_mode_generator_draws_its_stored_children_in_its_space() {
        // Format 29: the derived cache is the generator's children; nothing is evaluated.
        let s = Scene::load(&json!({
            "nodes": [
                { "id": "g", "type": "generator", "x": 50.0, "y": 60.0, "width": 100.0, "height": 100.0, "rotation": 0, "scale": 1,
                  "code": "b3:missing", "params": {}, "children": ["g/bar:a", "g/bar:b"], "outputKey": "b3:k" },
                { "id": "g/bar:a", "type": "rectangle", "x": 1.0, "y": 2.0, "width": 5.0, "height": 5.0, "rotation": 0, "scale": 1,
                  "genKey": "bar:a", "genHash": "0123456789abcdef" },
                { "id": "g/bar:b", "type": "rectangle", "x": 9.0, "y": 2.0, "width": 5.0, "height": 5.0, "rotation": 0, "scale": 1,
                  "genKey": "bar:b", "genHash": "fedcba9876543210" }
            ],
            "pages": [{"id":"p1","name":"P","width":1000.0,"height":1000.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["g"]}]
        })).unwrap();
        let order = s.flattened_z_order_for_page("p1");
        assert!(order.contains(&"g/bar:a".to_string()) && order.contains(&"g/bar:b".to_string()), "{order:?}");
        assert!(!order.iter().any(|id| id.contains("placeholder")), "no evaluation, no placeholder: {order:?}");
        let t = s.node_canvas_transform("g/bar:b").unwrap();
        assert_eq!((t.x, t.y), (59.0, 62.0));
    }

    #[test]
    fn chart_host_is_transform_parent_of_placeholder() {
        let s = Scene::load(&json!({
            "nodes": [{
                "id":"ch","type":"generated","generatedKind":"chart",
                "x":50.0,"y":60.0,"width":300.0,"height":200.0,"rotation":0,"scale":1,
                "renderMode":"native-children",
                "generatedSource":{"kind":"chart","chartVersion":1,"spec":{"type":"bar"},
                    "data":{"columns":[],"rows":[],"series":[],"categories":[],"valueRange":[0,0]}}
            }],
            "pages": [{"id":"p1","name":"P","width":1000.0,"height":1000.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["ch"]}]
        })).unwrap();
        let order = s.flattened_z_order_for_page("p1");
        assert_eq!(order[1], "ch_gen_proxy_0_placeholder");
        // host x/y propagate via the transform parent chain
        // (nodeTransformUtils.ts:157-172).
        let t = s
            .node_canvas_transform("ch_gen_proxy_0_placeholder")
            .unwrap();
        assert_eq!((t.x, t.y), (50.0, 60.0));
    }

    #[test]
    fn raster_asset_proxy_transform_resolves_via_transform_source_id() {
        // Attribution-sweep find (3d0f70ef p0 map / d31a991e p3 latex): a
        // `raster-asset` generated host nested in a container expands to a
        // pattern-rect proxy whose own parent chain BREAKS
        // (nodeTransformUtils.ts:157-172 — only `native-children` hosts are
        // transform parents), so the proxy's own affine is host-LOCAL. The
        // production model matrix instead resolves through
        // `_transformSourceId` (WebGPUPathRenderer.ts:2202-2208), landing the
        // proxy on the host's full container chain. Mirrors canvas 3d0f70ef:
        // container (16,2160) > generated LaTeX (18,48) => absolute (34,2208).
        let s = Scene::load(&json!({
            "nodes": [
                {"id":"ct","type":"container","x":16.0,"y":2160.0,"width":358.0,"height":288.0,
                 "children":["gen"]},
                {"id":"gen","type":"generated","generatedKind":"latex","x":18.0,"y":48.0,
                 "width":322.0,"height":150.0,"rotation":0,"scale":1,
                 "renderMode":"raster-asset",
                 "cachedAssetUrl":"https://assets/latex.png",
                 "cachedNaturalDimensions":{"width":644.0,"height":300.0},
                 "generatedSource":{"kind":"latex","latex":"x","displayMode":"display","fontSize":32}}
            ],
            "pages": [{"id":"p1","name":"P","width":390.0,"height":2956.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["ct"]}]
        })).unwrap();
        let proxy_id = "gen_gen_proxy_0";
        let proxy = s.transient_by_id.get(proxy_id).expect("proxy expanded");
        // the redirect target is the generated host
        assert_eq!(paint_transform_id(proxy), "gen");
        // The proxy's own chain now steps THROUGH the non-native-children host
        // to the container (transform_parent_chain), so bounds/hit-test
        // consumers that use the proxy's own id agree with paint. The proxy
        // copies the host's local TRS, so through-stepping reproduces the
        // host's accumulated transform exactly.
        let bare = s.node_canvas_transform(proxy_id).unwrap();
        assert_eq!((bare.x, bare.y), (34.0, 2208.0));
        // redirected: the host's container chain applies (browser placement)
        let redirected = s.node_canvas_transform(paint_transform_id(proxy)).unwrap();
        assert_eq!((redirected.x, redirected.y), (34.0, 2208.0));
        // nodes without `_transformSourceId` are unaffected ('' is falsy in
        // the TS `||` — covered by the filter)
        let host = s.node("gen").unwrap();
        assert_eq!(paint_transform_id(host), "gen");
    }

    #[test]
    fn page_positions_vertical_default() {
        // scene.layout.ts:99-106 + canvasDefaults.ts:152.
        let s = Scene::load(&json!({
            "nodes": [],
            "pages": [
                {"id":"p1","name":"a","width":100.0,"height":200.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":[]},
                {"id":"p2","name":"b","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":[]}
            ]
        })).unwrap();
        assert_eq!(s.page_position(0), (0.0, 0.0));
        assert_eq!(s.page_position(1), (0.0, 300.0)); // 200 + 100 spacing
    }

    #[test]
    fn page_positions_grid_match_browser_layout_for_variable_page_sizes() {
        // scene.layout.ts grid arm accumulates actual widths within the row
        // and row-local max heights for prior rows.
        let s = Scene::load(&json!({
            "nodes": [],
            "layoutMode": "grid",
            "gridColumns": 2.0,
            "pages": [
                {"id":"p1","name":"a","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":[]},
                {"id":"p2","name":"b","width":300.0,"height":80.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":[]},
                {"id":"p3","name":"c","width":200.0,"height":50.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":[]},
                {"id":"p4","name":"d","width":120.0,"height":200.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":[]}
            ]
        })).unwrap();

        assert_eq!(s.page_position(0), (0.0, 0.0));
        assert_eq!(s.page_position(1), (200.0, 0.0));
        assert_eq!(s.page_position(2), (0.0, 200.0));
        assert_eq!(s.page_position(3), (300.0, 200.0));
    }

    #[test]
    fn render_page_clipping_reads_config_and_defaults_on() {
        let s = Scene::load(&json!({"nodes":[],"pages":[]})).unwrap();
        assert!(s.doc.config.render_page_clipping);
        let s = Scene::load(&json!({"nodes":[],"pages":[],"config":{"renderPageClipping":false}})).unwrap();
        assert!(!s.doc.config.render_page_clipping);
    }

    #[test]
    fn unknown_envelope_key_is_render_invariant() {
        // Byte-exact proof for the `unknown:envelope-field` non-gating policy
        // (assess.rs `axis_elevates_verdict`, 2026-07-04): making a doc that carries
        // a foreign top-level key RENDER (instead of declining to Chromium) is safe
        // because hydration DROPS every unrecognized top-level key into
        // `unknown_envelope_keys` — a field the render-plan builder never reads. So
        // the hydrated `SceneDocument` the plan is built from is byte-identical with
        // and without the foreign key; only the observability vector differs. A scene
        // that renders identically to one that never had the key renders correct
        // (browser-matching) pixels.
        let base = json!({
            "nodes": [{
                "id": "r", "type": "rectangle",
                "x": 1.0, "y": 2.0, "width": 30.0, "height": 40.0,
                "rotation": 0, "scale": 1, "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#00ff00"}}], "cornerRadius": 4.0
            }],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,
                       "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["r"]}],
            "config": {"renderPageClipping": true},
            "variables": [],
            "canvasRootNodeIds": ["r"],
            "layoutMode": "grid",
            "gridColumns": 3
        });
        let mut with_foreign = base.as_object().unwrap().clone();
        // `schemaVersion` was the top REMAINING prod decline cause once node-field
        // stopped gating; a novel key exercises the general foreign-key path.
        with_foreign.insert("schemaVersion".into(), json!(7));
        with_foreign.insert("someFutureLegacyKey".into(), json!({"a": [1, 2, 3]}));

        let a = hydrate_document(&base);
        let b = hydrate_document(&Value::Object(with_foreign));

        // Every field the render-plan builder consumes is byte-identical.
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.pages, b.pages);
        assert_eq!(a.config, b.config);
        assert_eq!(a.canvas_root_node_ids, b.canvas_root_node_ids);
        assert_eq!(a.layout_mode, b.layout_mode);
        assert_eq!(a.grid_columns, b.grid_columns);
        // Only the non-render observability signal differs — `schemaVersion` is a
        // recognized ENVELOPE_KEY (never spills); the novel key is the sole foreign one.
        assert!(a.unknown_envelope_keys.is_empty());
        assert_eq!(
            b.unknown_envelope_keys,
            vec!["someFutureLegacyKey".to_string()]
        );
    }

    // -----------------------------------------------------------------------
    // Scoped transient re-expansion — CPU parity oracle.
    //
    // `reexpand_transient_host` exists to avoid a whole-scene reload when one
    // table/chart host is edited, so its ONLY correctness criterion is that the
    // scene it leaves behind is indistinguishable from the one
    // `Scene::load(patched_json)` would have built. Every case below asserts
    // exactly that, over all five transient maps plus `resolved_by_id`.
    // -----------------------------------------------------------------------

    fn table_doc() -> Value {
        json!({
            "nodes": [{
                "id":"tbl","type":"table",
                "x":100.0,"y":200.0,"width":60.0,"height":40.0,"rotation":0,"scale":1,
                "rows":[{"size":20.0},{"size":20.0}],
                "columns":[{"size":30.0},{"size":30.0}],
                "cells":[
                    [{"id":"c00","textContent":"<p>a</p>"},
                     {"id":"c01","textContent":"<p>b</p>","textHorizontalAlign":"center"}],
                    [{"id":"c10","textContent":"<p>c</p>"},
                     {"id":"c11","textContent":"<p>d</p>"}]
                ],
                "backgroundColors":["#eeeeee","#dddddd"],
                "headerBackgroundColor":"#222222",
                "rowStrokeConfig":"all","columnStrokeConfig":"all",
                "strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#000000"},"width":2.0}],"cornerRadius":8.0,"opacity":0.5
            }],
            "pages": [{"id":"p1","name":"P","width":1080.0,"height":1080.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                       "nodeIds":["tbl"]}]
        })
    }

    /// A genuinely-expanding bar chart (a `config` with a string palette is the
    /// minimum that clears `chart_config_issue`'s throw gate, so this emits real
    /// chart parts rather than the placeholder — see the non-vacuity assertion
    /// in `assert_reexpand_matches_cold_load`).
    fn chart_doc() -> Value {
        json!({
            "nodes": [{
                "id":"ch","type":"generated","generatedKind":"chart",
                "x":40.0,"y":60.0,"width":400.0,"height":300.0,"rotation":0,"scale":1,
                "renderMode":"native-children",
                "metadata":{"name":"Quarterly"},
                "generatedSource":{
                    "kind":"chart","chartVersion":1,
                    "spec":{"type":"bar","x":0,"y":0,"width":400,"height":300,
                            "config":{"palette":["#3b82f6","#ef4444"],
                                      "fontFamily":"Inter","fontSize":12,
                                      "showGrid":true,"showXAxis":true,"showYAxis":true,
                                      "showXLabels":true,"showYLabels":true,"yTickCount":5,
                                      "title":"Quarterly results"}},
                    "data":{"columns":["category","series","value"],
                            "rows":[{"category":"Q1","series":"Revenue","value":120},
                                    {"category":"Q2","series":"Revenue","value":180}],
                            "series":["Revenue"],"categories":["Q1","Q2"],
                            "valueRange":[0,180]}
                }
            }],
            "pages": [{"id":"p1","name":"P","width":800.0,"height":600.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                       "nodeIds":["ch"]}]
        })
    }

    fn host_node_mut<'a>(doc: &'a mut Value, host_id: &str) -> &'a mut Value {
        doc.get_mut("nodes")
            .and_then(Value::as_array_mut)
            .expect("fixture carries a nodes array")
            .iter_mut()
            .find(|n| n.get("id").and_then(Value::as_str) == Some(host_id))
            .expect("fixture carries the host node")
    }

    fn as_list_map(m: &HashMap<NodeId, Vec<NodeId>>) -> BTreeMap<&str, &Vec<NodeId>> {
        m.iter().map(|(k, v)| (k.as_str(), v)).collect()
    }

    fn as_str_map(m: &HashMap<NodeId, String>) -> BTreeMap<&str, &str> {
        m.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
    }

    /// Compare two id → node maps key by key, so a failure names the offending
    /// id and field instead of dumping two whole scenes.
    fn assert_node_maps_eq(
        label: &str,
        live: &HashMap<NodeId, RawNode>,
        cold: &HashMap<NodeId, RawNode>,
    ) {
        let live_ids: BTreeSet<&str> = live.keys().map(String::as_str).collect();
        let cold_ids: BTreeSet<&str> = cold.keys().map(String::as_str).collect();
        assert_eq!(
            live_ids, cold_ids,
            "{label}: id sets differ between re-expansion and cold load"
        );
        for id in live_ids {
            let (a, b) = (&live[id], &cold[id]);
            if a == b {
                continue;
            }
            let keys: BTreeSet<&str> = a.fields().keys().chain(b.fields().keys()).map(|k| k.as_str()).collect();
            for k in keys {
                assert_eq!(
                    a.to_json_map().get(k),
                    b.to_json_map().get(k),
                    "{label}[{id}].{k}: re-expansion (left) diverged from cold load (right)"
                );
            }
            assert_eq!(a, b, "{label}[{id}] differs but every field compared equal");
        }
    }

    /// The oracle: patch the host's canonical JSON, cold-load that, apply the
    /// same property merge to a live scene, re-expand just the host, and require
    /// the two to be indistinguishable.
    fn assert_reexpand_matches_cold_load(
        base: &Value,
        host_id: &str,
        patch: impl Fn(&mut Value),
    ) -> ReexpandOutcome {
        let mut scene = Scene::load(base).unwrap();
        assert!(
            scene
                .transient_by_parent
                .get(host_id)
                .is_some_and(|ids| !ids.is_empty()),
            "fixture host {host_id} must expand to at least one transient, or the \
             parity assertions below are vacuous"
        );

        let mut patched = base.clone();
        patch(host_node_mut(&mut patched, host_id));
        let patched_host = host_node_mut(&mut patched, host_id).clone();
        let cold = Scene::load(&patched).unwrap();
        assert!(
            cold.transient_by_parent
                .get(host_id)
                .is_some_and(|ids| !ids.is_empty()),
            "patched fixture host {host_id} must still expand to at least one transient"
        );

        // Mirror the engine's incremental patch lane: the patched top-level
        // props are merged into the canonical node (`ResidentScene::apply_props`
        // → `Scene::node_mut`), and only then is the host re-expanded.
        {
            let live = scene.node_mut(host_id).expect("host node is canonical");
            for (k, v) in patched_host.as_object().expect("host node is an object") {
                live.set(k, v.clone());
            }
        }
        assert_eq!(
            scene.node(host_id),
            cold.node(host_id),
            "the merged canonical node must equal the cold-loaded one — otherwise \
             this test measures hydration/migration drift, not re-expansion parity"
        );

        let outcome = scene
            .reexpand_transient_host(host_id).unwrap()
            .expect("a table/generated host re-expands");

        assert_node_maps_eq(
            "transient_by_id",
            &scene.transient_by_id,
            &cold.transient_by_id,
        );
        assert_node_maps_eq(
            "resolved_by_id",
            &scene.resolved_by_id,
            &cold.resolved_by_id,
        );
        assert_eq!(
            as_list_map(&scene.transient_by_parent),
            as_list_map(&cold.transient_by_parent),
            "transient_by_parent"
        );
        assert_eq!(
            as_str_map(&scene.transient_parent_of),
            as_str_map(&cold.transient_parent_of),
            "transient_parent_of"
        );
        assert_eq!(
            as_str_map(&scene.transient_page_of),
            as_str_map(&cold.transient_page_of),
            "transient_page_of"
        );
        assert_eq!(
            Some(&outcome.added),
            cold.transient_by_parent.get(host_id),
            "the reported `added` sequence must be the cold load's transient order \
             for this host — the caller splices the display list by it"
        );
        outcome
    }

    /// The page-background analogue of
    /// `assert_reexpand_matches_cold_load`. A page background is DERIVED from the
    /// page rather than stored as a node, so the incremental patch lane has to
    /// re-derive it after a page-prop patch; this asserts the scoped path leaves
    /// every transient map byte-identical to a cold `Scene::load` of the same
    /// patched document.
    ///
    /// It cannot reuse the host oracle above: that one starts at
    /// `self.node(host_id)`, and a page id is not a node id.
    fn assert_page_bg_reexpand_matches_cold_load(
        base: &Value,
        page_id: &str,
        patch: impl Fn(&mut Value),
    ) -> ReexpandOutcome {
        let bg_id = compute_page_bg_id(page_id);
        let mut scene = Scene::load(base).unwrap();
        assert!(
            scene.transient_by_id.contains_key(&bg_id),
            "fixture page {page_id} must derive a background transient, or the \
             parity assertions below are vacuous"
        );

        let mut patched = base.clone();
        fn page_of<'a>(doc: &'a mut Value, page_id: &str) -> &'a mut Value {
            doc["pages"]
                .as_array_mut()
                .expect("pages array")
                .iter_mut()
                .find(|p| p["id"] == json!(page_id))
                .expect("page in fixture")
        }
        patch(page_of(&mut patched, page_id));
        let patched_page = page_of(&mut patched, page_id).clone();
        let cold = Scene::load(&patched).unwrap();

        // Mirror the engine's incremental patch lane: the patched props are
        // merged into the canonical PAGE (`ResidentScene::apply_props` →
        // `doc.pages`), and only then is the background re-derived.
        {
            let live = scene
                .doc
                .pages
                .iter_mut()
                .find(|p| p.id() == page_id)
                .expect("page is canonical");
            for (k, v) in patched_page.as_object().expect("page is an object") {
                live.0.insert(k.clone(), v.clone());
            }
        }

        let outcome = scene
            .reexpand_page_background(page_id)
            .expect("a known page re-derives its background");

        assert_node_maps_eq(
            "transient_by_id",
            &scene.transient_by_id,
            &cold.transient_by_id,
        );
        assert_node_maps_eq(
            "resolved_by_id",
            &scene.resolved_by_id,
            &cold.resolved_by_id,
        );
        assert_eq!(
            as_list_map(&scene.transient_by_parent),
            as_list_map(&cold.transient_by_parent),
            "transient_by_parent"
        );
        assert_eq!(
            as_str_map(&scene.transient_parent_of),
            as_str_map(&cold.transient_parent_of),
            "transient_parent_of"
        );
        assert_eq!(
            as_str_map(&scene.transient_page_of),
            as_str_map(&cold.transient_page_of),
            "transient_page_of"
        );
        // `visit_z` walks `transient_by_parent` as a SEQUENCE, and the background
        // is first in every page's z-order — so the reported `added` list must be
        // the cold load's, not merely the same set.
        assert_eq!(
            Some(&outcome.added),
            cold.transient_by_parent.get(page_id),
            "the reported `added` sequence must be the cold load's for this page"
        );
        outcome
    }

    fn page_bg_doc() -> Value {
        json!({
            "nodes": [{"id":"a","type":"rectangle","x":10.0,"y":10.0,
                       "width":50.0,"height":50.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#f00"}}]}],
            "pages": [{"id":"p1","name":"P","width":800.0,"height":600.0,
                       "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#112233"}}],"nodeIds":["a"]}]
        })
    }

    #[test]
    fn reexpand_page_background_fill_matches_cold_load() {
        let outcome = assert_page_bg_reexpand_matches_cold_load(&page_bg_doc(), "p1", |page| {
            page["fills"] = json!([{"id":"fill-0","paint":{"type":"solid","color":"#00ff00"}}]);
        });
        // The bg id is derived from the page id, so a fill change is always
        // shape-preserving — the caller may patch the existing display item.
        assert_eq!(outcome.removed, outcome.added);
    }

    #[test]
    fn reexpand_page_background_size_matches_cold_load() {
        // width/height are derived from the PAGE, so a resize moves the quad
        // even though no bg prop was named.
        assert_page_bg_reexpand_matches_cold_load(&page_bg_doc(), "p1", |page| {
            page["width"] = json!(1000.0);
            page["height"] = json!(250.0);
        });
    }

    #[test]
    fn reexpand_page_background_gradient_matches_cold_load() {
        assert_page_bg_reexpand_matches_cold_load(&page_bg_doc(), "p1", |page| {
            page["fills"] = json!([{"id": "fill-0", "paint": {"type": "linear",
                "start": {"x": 0.0, "y": 0.0}, "end": {"x": 1.0, "y": 1.0},
                "stops": [{"offset": 0.0, "color": "#ff0000"}, {"offset": 1.0, "color": "#0000ff"}]}}]);
        });
    }

    /// Freeform canvas-root nodes have no background prefix
    /// (`flattened_z_order_for_canvas`), and re-deriving one page must not touch
    /// another page's records. `None` — for an unknown page id — must mutate
    /// NOTHING, so a caller can treat it as "not my job".
    #[test]
    fn reexpand_page_background_is_scoped_and_bails_clean() {
        let doc = json!({
            "nodes": [{"id":"a","type":"rectangle","x":0.0,"y":0.0,"width":50.0,"height":50.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#f00"}}]},
                      {"id":"b","type":"rectangle","x":0.0,"y":0.0,"width":50.0,"height":50.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#0f0"}}]}],
            "pages": [{"id":"p1","width":800.0,"height":600.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#111111"}}],"nodeIds":["a"]},
                      {"id":"p2","width":600.0,"height":300.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#222222"}}],"nodeIds":["b"]}]
        });
        let mut scene = Scene::load(&doc).unwrap();
        let p2_before = scene.transient_by_id.get("__page_bg_p2").cloned();
        let a_before = scene.resolved_by_id.get("a").cloned();

        assert!(
            scene.reexpand_page_background("nope").is_none(),
            "an unknown page id is not this method's job"
        );
        assert_eq!(
            scene.transient_by_id.get("__page_bg_p2").cloned(),
            p2_before,
            "a `None` return must mutate nothing"
        );

        scene.reexpand_page_background("p1").expect("p1 re-derives");
        assert_eq!(
            scene.transient_by_id.get("__page_bg_p2").cloned(),
            p2_before,
            "re-deriving p1 must leave p2's background untouched"
        );
        assert_eq!(
            scene.resolved_by_id.get("a").cloned(),
            a_before,
            "re-deriving a page must not disturb canonical nodes"
        );
    }

    #[test]
    fn reexpand_table_cell_text_matches_cold_load() {
        let outcome = assert_reexpand_matches_cold_load(&table_doc(), "tbl", |host| {
            host["cells"][0][0]["textContent"] = json!("<p>edited</p>");
        });
        // A cell-text edit is shape-preserving: the ids are byte-identical, so a
        // caller may patch the existing display items in place.
        assert_eq!(outcome.removed, outcome.added);
    }

    #[test]
    fn reexpand_table_row_resize_matches_cold_load() {
        // Row heights drive every cell y, every content inset, and both
        // horizontal border spans.
        assert_reexpand_matches_cold_load(&table_doc(), "tbl", |host| {
            host["rows"] = json!([{"size": 45.0}, {"size": 12.0}]);
            host["height"] = json!(57.0);
        });
    }

    #[test]
    fn reexpand_table_columns_matches_cold_load() {
        assert_reexpand_matches_cold_load(&table_doc(), "tbl", |host| {
            host["columns"] = json!([{"size": 80.0}, {"size": 30.0}]);
            host["width"] = json!(110.0);
        });
    }

    #[test]
    fn an_edit_that_re_derives_other_nodes_dirties_everything() {
        use crate::stamps::Group;
        // A table's cells are transients derived from the host: an edit of the host is never one row's change.
        let mut scene = Scene::load(&table_doc()).unwrap();
        let built = scene.stamps.clock();
        let mut host = scene.node("tbl").unwrap().clone();
        host.set("x", json!(333.0));
        scene.apply_node_edits(vec![host], &[], &[], None).unwrap();
        assert!(scene.stamps.all() > built);
        assert!(!scene.stamps.current(0, Group::Content, built));
    }

    #[test]
    fn reexpand_table_host_move_matches_cold_load() {
        // The load-bearing case for re-expansion existing at all: cell and
        // border coordinates BAKE the host's x/y (expansion.rs:204-205,
        // 260-263), so a move cannot be served by overlaying x/y on the host.
        let outcome = assert_reexpand_matches_cold_load(&table_doc(), "tbl", |host| {
            host["x"] = json!(333.0);
            host["y"] = json!(-44.5);
        });
        assert_eq!(
            outcome.removed, outcome.added,
            "a move keeps the display-list shape — only geometry changes"
        );
    }

    #[test]
    fn reexpand_table_row_stroke_config_matches_cold_load() {
        // `lineShouldRender` vocabulary (expansion.rs:37-56): dropping from
        // 'all' to 'horizontal'-style configs removes border nodes entirely, so
        // the transient id sequence shrinks.
        for cfg in ["none", "outside", "inside", "after-first"] {
            let outcome = assert_reexpand_matches_cold_load(&table_doc(), "tbl", |host| {
                host["rowStrokeConfig"] = json!(cfg);
            });
            assert!(
                outcome.added.len() < outcome.removed.len(),
                "rowStrokeConfig='{cfg}' must drop at least one border line"
            );
        }
    }

    #[test]
    fn reexpand_table_add_row_matches_cold_load() {
        let outcome = assert_reexpand_matches_cold_load(&table_doc(), "tbl", |host| {
            host["rows"] = json!([{"size": 20.0}, {"size": 20.0}, {"size": 20.0}]);
            host["cells"].as_array_mut().unwrap().push(json!([
                {"id":"c20","textContent":"<p>e</p>"},
                {"id":"c21","textContent":"<p>f</p>"}
            ]));
            host["height"] = json!(60.0);
        });
        assert!(
            outcome.added.len() > outcome.removed.len(),
            "adding a row must add transients — the caller has to rebuild the slice"
        );
        assert_ne!(outcome.removed, outcome.added);
    }

    #[test]
    fn reexpand_table_remove_row_matches_cold_load() {
        let outcome = assert_reexpand_matches_cold_load(&table_doc(), "tbl", |host| {
            host["rows"] = json!([{"size": 20.0}]);
            host["cells"] = json!([[
                {"id":"c00","textContent":"<p>a</p>"},
                {"id":"c01","textContent":"<p>b</p>","textHorizontalAlign":"center"}
            ]]);
            host["height"] = json!(20.0);
        });
        assert!(
            outcome.added.len() < outcome.removed.len(),
            "removing a row must retire transients"
        );
        // The retired records are provably evicted rather than merely dropped
        // from the host's list: the helper asserts the live transient/resolved id
        // SETS equal the cold load's, which contains no trace of them.
    }

    /// A chart whose config/data the generator rejects silently degrades to ONE
    /// placeholder rect, which would still satisfy the "expands to something"
    /// guard — so the chart cases additionally require real generated parts.
    fn assert_real_chart_parts(outcome: &ReexpandOutcome) {
        assert!(
            outcome.added.len() > 1 && !outcome.added.iter().any(|id| id.ends_with("_placeholder")),
            "chart fixture degraded to the placeholder ({:?}) — the parity assertions \
             would then never exercise the chart generator",
            outcome.added
        );
    }

    #[test]
    fn reexpand_chart_data_matches_cold_load() {
        // Chart parts are generated from `generatedSource`, so a data edit
        // re-runs the whole generator.
        let outcome = assert_reexpand_matches_cold_load(&chart_doc(), "ch", |host| {
            host["generatedSource"]["data"] = json!({
                "columns":["category","series","value"],
                "rows":[{"category":"Q1","series":"Revenue","value":10},
                        {"category":"Q2","series":"Revenue","value":20},
                        {"category":"Q3","series":"Revenue","value":30}],
                "series":["Revenue"],"categories":["Q1","Q2","Q3"],
                "valueRange":[0,30]
            });
        });
        assert_real_chart_parts(&outcome);
    }

    #[test]
    fn reexpand_chart_host_move_matches_cold_load() {
        let outcome = assert_reexpand_matches_cold_load(&chart_doc(), "ch", |host| {
            host["x"] = json!(210.0);
            host["y"] = json!(15.0);
        });
        assert_real_chart_parts(&outcome);
    }

    #[test]
    fn reexpand_chart_resize_matches_cold_load() {
        // Width feeds the chart layout (and the placeholder fallback geometry),
        // so a resize re-lays-out every part.
        let outcome = assert_reexpand_matches_cold_load(&chart_doc(), "ch", |host| {
            host["width"] = json!(640.0);
            host["generatedSource"]["spec"]["width"] = json!(640);
        });
        assert_real_chart_parts(&outcome);
    }

    #[test]
    fn reexpand_table_inside_translucent_group_is_ancestor_free() {
        // The doc comment's claim that no ancestor context is needed: the SAME
        // table nested under a translated container inside a translucent group
        // must re-expand to the same records a cold load produces. Ancestor
        // opacity/transform are applied at paint time
        // (`ancestor_opacity_product` / `transform_parent_chain`), never baked.
        let nested = json!({
            "nodes": [
                {"id":"g","type":"group","x":0.0,"y":0.0,"opacity":0.4,"children":["card"]},
                {"id":"card","type":"container","x":70.0,"y":648.0,"width":940.0,"height":276.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ffffff"}}],"children":["tbl"]},
                {"id":"tbl","type":"table","x":28.0,"y":68.0,"width":60.0,"height":40.0,
                 "rotation":0,"scale":1,
                 "rows":[{"size":20.0},{"size":20.0}],
                 "columns":[{"size":30.0},{"size":30.0}],
                 "cells":[
                    [{"id":"c00","textContent":"<p>a</p>"},{"id":"c01","textContent":"<p>b</p>"}],
                    [{"id":"c10","textContent":"<p>c</p>"},{"id":"c11","textContent":"<p>d</p>"}]
                 ]}
            ],
            "pages": [{"id":"p1","name":"P","width":1080.0,"height":1080.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                       "nodeIds":["g"]}]
        });
        assert_reexpand_matches_cold_load(&nested, "tbl", |host| {
            host["cells"][1][1]["textContent"] = json!("<p>nested edit</p>");
            host["x"] = json!(31.0);
        });
    }

    #[test]
    fn reexpand_non_host_is_noop() {
        // A `None` return must leave the scene bit-identical: the caller uses
        // `None` to mean "this id owns no transients", and any partial mutation
        // here would silently corrupt an unrelated node's records.
        // A plain rectangle alongside the table, so the scene carries both.
        let mut doc_with_rect = table_doc();
        doc_with_rect["nodes"].as_array_mut().unwrap().push(json!(
            {"id":"r1","type":"rectangle","x":1.0,"y":2.0,"width":3.0,"height":4.0,
             "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#f00"}}]}
        ));
        doc_with_rect["pages"][0]["nodeIds"] = json!(["tbl", "r1"]);
        let mut s = Scene::load(&doc_with_rect).unwrap();

        let before = (
            s.transient_by_id.clone(),
            s.transient_by_parent.clone(),
            s.transient_parent_of.clone(),
            s.transient_page_of.clone(),
            s.resolved_by_id.clone(),
        );

        assert!(s.reexpand_transient_host("r1").unwrap().is_none());
        assert!(s.reexpand_transient_host("does-not-exist").unwrap().is_none());
        assert!(
            s.reexpand_transient_host("p1").unwrap().is_none(),
            "a page is not a node"
        );

        assert_eq!(before.0, s.transient_by_id);
        assert_eq!(before.1, s.transient_by_parent);
        assert_eq!(before.2, s.transient_parent_of);
        assert_eq!(before.3, s.transient_page_of);
        assert_eq!(before.4, s.resolved_by_id);
    }

}
