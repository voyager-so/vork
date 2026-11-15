//! residency.rs — Stage-1 retained-mode engine residency (task #95).
//!
//! All caching here is **keyed by content hashes computed from the re-fed
//! scene**, so every cache is *self-invalidating by construction*: there are
//! no invalidation messages to get wrong. Hash discipline (PLAN §2.2):
//! per-node hashes use **deny-list exclusion, never allow-list inclusion** —
//! unknown/new props therefore OVER-invalidate (safe, costs a re-compute)
//! instead of under-invalidating (stale pixels).
//!
//! Layers (PLAN §1):
//! - R1: `ResidentScene` — parse-once-per-sweep resident hydrated Scene +
//!   per-node hash maps (geom / text / full).
//! - R3: per-node tessellation cache (`PlanCaches::tess` / `line_tess`) —
//!   ONE entry per node, evict-and-replace on GeomKey change. Transform-only
//!   changes (x/y/rotation/scale*) do NOT re-key (deliberately better than
//!   the browser, which over-invalidates on x/y — MAP §5.1). `strokeWidth`
//!   IS in the key (the browser's latent stale-geometry hazard, MAP §5.2).
//! - R4: text item cache (full `TextQuadBatch` outcomes keyed by text-hash;
//!   transform/z/clip patched on hit) + the shaping cache inside moda-text.
//! - R5: per-page plan cache keyed by a display-list fingerprint.
//! - R6: blend-expansion tessellation rides R3 keyed by source-node hashes.
//!
//! A one-shot render builds no caches. An export job (`export_job`: moda-cli
//! `batch` and `slices`) builds one `PlanCaches` for its pages, keyed by
//! content; plan building through it is byte-identical to building without.

/// The plan caches' and resident scene's id-keyed maps hash with FxHash (std's SipHash showed up across a
/// first frame's plan build in wasm). Keys are the document's own ids and content hashes.
pub type HashMap<K, V> = std::collections::HashMap<K, V, rustc_hash::FxBuildHasher>;
use std::hash::Hasher;
use std::sync::Arc;

use moda_scene::types::RawNode;
use moda_scene::Scene;
use moda_scene::value::Value;

use crate::frame::{PlanWarnings, RenderOptions};
use crate::seams::{CoveragePlan, TessellatedGeometry};

// ---------------------------------------------------------------------------
// Hashing (FxHasher — same hasher family as the production fxhash plan)
// ---------------------------------------------------------------------------

type FxHasher = rustc_hash::FxHasher;

/// Resident generations come from one counter for every [`ResidentScene`] in the process, so no two scenes
/// ever share a generation. The plan caches key memos, display lists and cull indexes on the generation; a
/// scene parked for a Design page (`page_cache`) and brought back later must never meet a memo that another
/// scene left at the same number.
static NEXT_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_generation() -> u64 {
    NEXT_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Hash a whole byte slice (the R1 scene-JSON identity check).
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = FxHasher::default();
    h.write(bytes);
    h.finish()
}

/// TRANSFORM-ONLY props (PLAN §2.1): model-matrix inputs only — tessellation
/// and text residency survive changes to these.
///
/// `unitTransforms` (KineticText) is deliberately NOT here: the per-glyph
/// transform/opacity are baked into the analytic instance records at build time
/// (not applied as a separate uniform), so a per-frame change MUST invalidate the
/// text cache and rebuild — keeping it out of this set is required for cache
/// safety. The consequence is that animating `unitTransforms` re-shapes + re-packs
/// the whole vector batch every frame; an optimization (patch only the per-glyph
/// record fields on an otherwise-identical text hash, instead of a full rebuild)
/// is deferred to the slice that ships the emitter, where the perf actually bites.
const TRANSFORM_ONLY: &[&str] = &["x", "y", "rotation", "scale", "scaleX", "scaleY"];

/// Known PAINT-ONLY props (PLAN §2.1): rebuilt per frame into instance /
/// uniform data from the resident plan — never tessellation inputs. This is
/// a DENY list for the geom hash: anything NOT listed here (and not
/// transform-only) re-keys tessellation, so unknown props over-invalidate.
///
/// The paint stacks (`fills`, `strokes`, `textFills`) are hashed through
/// [`hash_paint_geometry`]: their COLOURS are paint-only, but the fill's
/// paint TYPE (routing), an image/video placement (the texture quad), and the
/// stroke's width, caps, joins, dash, alignment, width model and arrowheads
/// (baked into stroke triangles) are tessellation inputs.
///
/// NOTE deliberately KEPT in the geom hash (i.e. absent below): `distort` (a
/// distort warp rewrites the tessellated geometry through the local
/// homography), `fillRule`, and every text prop (lines carve label gaps into
/// their body geometry).
///
/// The effect list (`effects`) is hashed through [`hash_paint_geometry`]
/// too: its drop / inner shadows are paint-only (own DrawItems and instance
/// data, rebuilt per frame), every other entry (blurs, backdrops, shaders,
/// text effects) re-keys tessellation.
const PAINT_ONLY: &[&str] = &["opacity", "blendMode"];

/// Effect types that feed only per-frame paint (see [`PAINT_ONLY`]).
const PAINT_ONLY_EFFECT_TYPES: &[&str] = &["drop-shadow", "inner-shadow"];

/// The entries of an effect list that change the display-list STRUCTURE: a
/// backdrop adds/removes a pass split (and flips the page's sample count /
/// transparency), a layer filter (layer blur, noise, texture, feather, filter shadow) diverts the
/// node's items into a capture group.
fn structural_effects(effects: Option<&Value>) -> Vec<&Value> {
    effects
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter(|e| {
                    e.get("type").and_then(Value::as_str).is_some_and(|t| {
                        matches!(t, "layer-blur" | "noise" | "texture" | "feather" | "filter-shadow")
                            || moda_scene::effect::BACKDROP_EFFECT_TYPES.contains(&t)
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// A key that feeds only per-frame paint.
fn is_paint_only(key: &str) -> bool {
    PAINT_ONLY.contains(&key)
}

/// Hash the tessellation-relevant part of a paint stack (see [`PAINT_ONLY`]) into `h`:
/// per fill layer its visibility, paint type and image/video placement plus
/// the recolour/overlay switches (which pick the texture pipeline);
/// per stroke layer everything but its paint's colours (the paint type stays:
/// a gradient stroke routes differently). Returns false, having hashed nothing, for keys that
/// are not a paint stack.
///
/// The part is hashed where it stands: no value is built to be hashed and dropped (one small map
/// per paint layer of every node, on every hash of the node). It is written through the one
/// encoding of a value ([`ValueHash`]), as the part would hash written out as a value with an
/// absent field as `null`.
///
/// Node hashes are what the tessellation and plan caches key on until those caches follow the
/// scene's stamps (renderer plan 2.4); this function goes with them.
fn hash_paint_geometry(key: &str, v: &Value, h: &mut FxHasher) -> bool {
    if !matches!(key, "fills" | "strokes" | "textFills" | "effects") {
        return false;
    }
    let Some(layers) = v.as_array() else { return false };
    let kept = |effect: &Value| {
        !effect.get("type").and_then(Value::as_str).is_some_and(|t| PAINT_ONLY_EFFECT_TYPES.contains(&t))
    };
    let mut h = ValueHash(h);
    h.array(match key {
        "effects" => layers.iter().filter(|effect| kept(*effect)).count(),
        _ => layers.len(),
    });
    for layer in layers {
        let paint = layer.get("paint");
        match key {
            "fills" => {
                let adjust = paint.and_then(|p| p.get("adjust"));
                let flag = |a: &str, b: &str| adjust.and_then(|x| x.get(a)).and_then(|x| x.get(b));
                h.object(5);
                h.field("visible", layer.get("visible"));
                h.field("type", paint.and_then(|p| p.get("type")));
                h.field("placement", paint.and_then(|p| p.get("placement")));
                h.field("overlay", flag("overlay", "enabled"));
                h.field("recolor", flag("recolor", "enabled"));
            }
            "strokes" => {
                // Every field but the layer's opacity and blend mode, in the layer's order, with the paint as
                // its type alone; a layer without a paint has one of type `null`, last.
                let fields = layer.as_object();
                let own = |k: &str| !matches!(k, "opacity" | "blendMode" | "paint");
                h.object(fields.map_or(0, |o| o.iter().filter(|(k, _)| own(k.as_str())).count()) + 1);
                let paint_type = |h: &mut ValueHash<'_>| {
                    h.key("paint");
                    h.object(1);
                    h.field("type", paint.and_then(|p| p.get("type")));
                };
                let mut painted = false;
                for (k, value) in fields.into_iter().flat_map(|o| o.iter()) {
                    match k.as_str() {
                        "opacity" | "blendMode" => {}
                        "paint" => {
                            paint_type(&mut h);
                            painted = true;
                        }
                        k => h.field(k, Some(value)),
                    }
                }
                if !painted {
                    paint_type(&mut h);
                }
            }
            "textFills" => {
                h.object(2);
                h.field("visible", layer.get("visible"));
                h.field("type", paint.and_then(|p| p.get("type")));
            }
            _ => {
                if kept(layer) {
                    h.value(layer);
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod paint_geometry_tests {
    use super::*;
    use moda_scene::value::json;

    fn in_place(key: &str, v: &Value) -> Option<u64> {
        let mut h = FxHasher::default();
        hash_paint_geometry(key, v, &mut h).then(|| h.finish())
    }
    fn of_value(v: &Value) -> u64 {
        let mut h = FxHasher::default();
        hash_value(v, &mut h);
        h.finish()
    }

    /// The part hashed in place is the hash of the part written out as a value: fills, text fills and effects.
    #[test]
    fn the_paint_part_hashes_as_the_value_it_stands_for() {
        let fills = json!([
            {"id": "a", "visible": true, "opacity": 0.5, "paint": {"type": "solid", "color": {"r": 1, "g": 0, "b": 0, "a": 1}}},
            {"id": "b", "paint": {"type": "image", "placement": {"mode": "fill"}, "adjust": {"overlay": {"enabled": true}, "recolor": {"enabled": false}}}},
            {"id": "c"},
            7,
        ]);
        assert_eq!(in_place("fills", &fills), Some(of_value(&json!([
            {"visible": true, "type": "solid", "placement": null, "overlay": null, "recolor": null},
            {"visible": null, "type": "image", "placement": {"mode": "fill"}, "overlay": true, "recolor": false},
            {"visible": null, "type": null, "placement": null, "overlay": null, "recolor": null},
            {"visible": null, "type": null, "placement": null, "overlay": null, "recolor": null},
        ]))));
        let text = json!([{"visible": false, "paint": {"type": "gradient", "stops": []}}, {}]);
        assert_eq!(in_place("textFills", &text), Some(of_value(&json!([
            {"visible": false, "type": "gradient"}, {"visible": null, "type": null},
        ]))));
        let effects = json!([{"type": "blur", "radius": 4}, {"type": PAINT_ONLY_EFFECT_TYPES[0]}, {"radius": 1}]);
        assert_eq!(in_place("effects", &effects), Some(of_value(&json!([{"type": "blur", "radius": 4}, {"radius": 1}]))));
        // Not a paint stack: nothing is hashed.
        assert_eq!(in_place("fills", &json!({"not": "a list"})), None);
        assert_eq!(in_place("points", &json!([])), None);
    }

    /// A stroke layer hashes everything but its colours, its opacity and its blend mode.
    #[test]
    fn a_stroke_layer_hashes_its_shape_and_paint_type_only() {
        let stroke = |color: f64, opacity: f64, width: f64, kind: &str| json!([
            {"id": "s", "width": width, "opacity": opacity, "blendMode": "normal", "paint": {"type": kind, "color": color}},
        ]);
        let base = in_place("strokes", &stroke(1., 1., 2., "solid"));
        assert!(base.is_some());
        assert_eq!(base, in_place("strokes", &stroke(0.25, 0.5, 2., "solid")), "colour and opacity are paint only");
        assert_ne!(base, in_place("strokes", &stroke(1., 1., 3., "solid")), "width is geometry");
        assert_ne!(base, in_place("strokes", &stroke(1., 1., 2., "gradient")), "the paint's type routes the stroke");
        assert_ne!(in_place("strokes", &json!([{"width": 2}])), in_place("strokes", &json!([{"width": 2}, {"width": 2}])));
    }
}

/// The three per-node content hashes (PLAN §2.2/§2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeHashes {
    /// Hash of the ENTIRE node value (R5 fingerprints, R6 keys).
    pub full: u64,
    /// Full minus TRANSFORM-ONLY minus known PAINT-ONLY (the GeomKey).
    pub geom: u64,
    /// Full minus TRANSFORM-ONLY (text caches: paint IS baked into quads).
    pub text: u64,
}

/// The one encoding of a JSON value into a hasher. Everything that hashes a value, or a part of one where it
/// stands, writes through this, so a value and its parts hash alike and there is one encoding to change.
struct ValueHash<'a>(&'a mut FxHasher);

impl ValueHash<'_> {
    fn null(&mut self) {
        self.0.write_u8(0);
    }
    fn bool(&mut self, b: bool) {
        self.0.write_u8(1);
        self.0.write_u8(b as u8);
    }
    /// Canonical numeric identity: f64 bits (JSON numbers are f64 in every producer we feed from).
    fn number(&mut self, bits: u64) {
        self.0.write_u8(2);
        self.0.write_u64(bits);
    }
    fn string(&mut self, s: &str) {
        self.0.write_u8(3);
        self.0.write(s.as_bytes());
    }
    /// An array of `len` items: the items follow.
    fn array(&mut self, len: usize) {
        self.0.write_u8(4);
        self.0.write_usize(len);
    }
    /// An object of `len` fields: each field's key, then its value, follow.
    fn object(&mut self, len: usize) {
        self.0.write_u8(5);
        self.0.write_usize(len);
    }
    fn key(&mut self, key: &str) {
        self.0.write(key.as_bytes());
    }
    /// A field of an object: its key and its value, `null` when the value is absent.
    fn field(&mut self, key: &str, value: Option<&Value>) {
        self.key(key);
        match value {
            Some(value) => self.value(value),
            None => self.null(),
        }
    }
    fn value(&mut self, v: &Value) {
        match v {
            Value::Null => self.null(),
            Value::Bool(b) => self.bool(*b),
            Value::Number(n) => self.number(n.as_f64().map(f64::to_bits).unwrap_or(0)),
            Value::String(s) => self.string(s),
            Value::Array(a) => {
                self.array(a.len());
                for item in a {
                    self.value(item);
                }
            }
            Value::Object(o) => {
                self.object(o.len());
                // preserve_order Map: iteration order == insertion order — stable
                // for the same JSON; a re-ordered but equal doc only
                // over-invalidates.
                for (k, val) in o {
                    self.field(k.as_str(), Some(val));
                }
            }
        }
    }
}

fn hash_value(v: &Value, h: &mut FxHasher) {
    ValueHash(h).value(v);
}

/// [`hash_value`] of compact points' JSON, from the compact form (so a node hashes the same however it holds
/// its points).
fn hash_compact_points(points: &moda_scene::path_points::CompactPoints, h: &mut FxHasher) {
    let mut h = ValueHash(h);
    let num = |h: &mut ValueHash<'_>, key: &str, v: f64| {
        h.key(key);
        h.number(v.to_bits());
    };
    h.array(points.len());
    for i in 0..points.len() {
        let (kind, c) = points.point(i);
        h.object(kind.keys().len());
        h.key("type");
        h.string(kind.name());
        num(&mut h, "x", c[0]);
        num(&mut h, "y", c[1]);
        for (j, key) in kind.controls().iter().enumerate() {
            h.key(key);
            h.object(2);
            num(&mut h, "x", c[2 + 2 * j]);
            num(&mut h, "y", c[3 + 2 * j]);
        }
    }
}

/// Per-node effective hashes for an already-hydrated [`Scene`] — the map the
/// plan-cache fingerprint and R3/R4 caches key on. Mirrors what
/// [`ResidentScene::update`] builds from JSON, but for a `Scene` constructed
/// any other way (e.g. the mobile FFI's per-frame `Scene::from_document`
/// overlay lane). Hashes the resolved-??-canonical view (`effective_node`),
/// exactly what the plan builder reads.
pub fn node_hashes_for_scene(scene: &Scene) -> HashMap<String, NodeHashes> {
    let mut hashes = HashMap::with_capacity_and_hasher(
        scene.node_order().len() + scene.transient_by_id().len(),
        Default::default(),
    );
    for id in scene.node_order() {
        if let Some(node) = scene.effective_node(id) {
            hashes.insert(id.clone(), node_hashes_in(scene, node));
        }
    }
    for (id, node) in scene.transient_by_id() {
        // Resolved transients are what effective_node returns; hash those.
        let node = scene.effective_node(id).unwrap_or(node);
        hashes.insert(id.clone(), node_hashes_in(scene, node));
    }
    hashes
}

/// [`node_hashes`] plus the document state a node's drawing depends on
/// beyond its own fields: the brush definition its `strokeBrush` names
/// (brush plan P8 — a brush-table edit must re-key the strokes that use it,
/// in the tess cache and in every plan fingerprint).
pub fn node_hashes_in(scene: &Scene, node: &RawNode) -> NodeHashes {
    node_hashes_in_skipping(scene, node, &[])
}

/// The fields [`content_hashes_for_scene`] leaves out: labels that name a layer without changing what it draws.
const LABEL_FIELDS: &[&str] = &["id", "name"];

/// [`node_hashes_for_scene`] without the layers' labels (`id`, `name`): two layers that draw the same thing hash
/// the same. An export job's caches (`export_job`) key on these, so a page reuses what another page with the same
/// content built; everything a layer draws from is still in its hashes.
pub fn content_hashes_for_scene(scene: &Scene) -> HashMap<String, NodeHashes> {
    let mut hashes = HashMap::with_capacity_and_hasher(
        scene.node_order().len() + scene.transient_by_id().len(),
        Default::default(),
    );
    for id in scene.node_order() {
        if let Some(node) = scene.effective_node(id) {
            hashes.insert(id.clone(), node_hashes_in_skipping(scene, node, LABEL_FIELDS));
        }
    }
    for (id, node) in scene.transient_by_id() {
        let node = scene.effective_node(id).unwrap_or(node);
        hashes.insert(id.clone(), node_hashes_in_skipping(scene, node, LABEL_FIELDS));
    }
    hashes
}

fn node_hashes_in_skipping(scene: &Scene, node: &RawNode, skip: &[&str]) -> NodeHashes {
    let mut nh = node_hashes_skipping(node, skip);
    // A text layer lays out in the document's locale (the language of text that declares none, which picks its CJK
    // faces: `text_draw::document_locale`), so a locale change re-keys it like an edit of its own.
    if node.node_type() == "richtext" {
        if let Some(locale) = crate::text_draw::document_locale(scene) {
            let mix = |x: u64| {
                let mut m = FxHasher::default();
                m.write_u64(x);
                m.write(b"locale");
                m.write(locale.as_bytes());
                m.finish()
            };
            nh.full = mix(nh.full);
            nh.geom = mix(nh.geom);
            nh.text = mix(nh.text);
        }
    }
    if let Some(id) = crate::tess::outline::brush_id(node).as_deref() {
        let mut h = FxHasher::default();
        h.write(b"brush");
        match scene.doc().brush_hash(id) {
            Some(bh) => h.write_u64(bh),
            None => h.write_u8(0),
        }
        let b = h.finish();
        let mix = |x: u64| {
            let mut m = FxHasher::default();
            m.write_u64(x);
            m.write_u64(b);
            m.finish()
        };
        nh.full = mix(nh.full);
        nh.geom = mix(nh.geom);
    }
    // E8.5: the pattern definitions its paints name (a pattern edit re-keys every node that draws it). Only the
    // full hash: the node's own geometry doesn't depend on a pattern (the tile shapes hash their own).
    if crate::pattern_paint::enabled() {
        let ids = moda_scene::pattern::node_pattern_ids(node);
        if !ids.is_empty() {
            let mut h = FxHasher::default();
            h.write_u64(nh.full);
            h.write(b"patterns");
            for id in &ids {
                h.write(id.as_bytes());
                match scene.doc().pattern_hash(id) {
                    Some(ph) => h.write_u64(ph),
                    None => h.write_u8(0),
                }
            }
            nh.full = h.finish();
        }
    }
    nh
}

/// Compute all three hashes in one pass over the node's top-level entries.
pub fn node_hashes(node: &RawNode) -> NodeHashes {
    node_hashes_skipping(node, &[])
}

fn node_hashes_skipping(node: &RawNode, skip: &[&str]) -> NodeHashes {
    let mut full = FxHasher::default();
    let mut geom = FxHasher::default();
    let mut text = FxHasher::default();
    for (k, v) in node.fields() {
        if skip.contains(&k.as_str()) {
            continue;
        }
        let mut entry = FxHasher::default();
        entry.write(k.as_bytes());
        match node.compact_points().filter(|_| k == "points") {
            Some(points) => hash_compact_points(points, &mut entry),
            None => hash_value(v, &mut entry),
        }
        let eh = entry.finish();
        full.write_u64(eh);
        let is_transform = TRANSFORM_ONLY.contains(&k.as_str());
        if !is_transform {
            text.write_u64(eh);
            let mut g = FxHasher::default();
            g.write(k.as_bytes());
            if hash_paint_geometry(k, v, &mut g) {
                geom.write_u64(g.finish());
            } else if !is_paint_only(k) {
                geom.write_u64(eh);
            }
        }
    }

    NodeHashes {
        full: full.finish(),
        geom: geom.finish(),
        text: text.finish(),
    }
}

// ---------------------------------------------------------------------------
// R1 — resident scene
// ---------------------------------------------------------------------------

/// Parse-once-per-sweep residency: the hydrated Scene + per-node hash maps,
/// keyed by the fxhash of the raw scene JSON. `update` re-parses only when
/// the JSON content changed; a multi-page sweep re-feeding the same string
/// hits the resident copy (Stage 0 re-parsed once per page).
pub struct ResidentScene {
    state: Option<Resident>,
    /// M1 fingerprint memo: a monotonic counter bumped on every resident-content
    /// change — a real re-parse (`update`), an in-place scalar patch
    /// (`apply_props`), or a `clear`. An unchanged value proves the scene content
    /// is unchanged, the cheap superset signal the camera fingerprint memo keys
    /// on. Over-bumps only cost a recompute; under-bumps would risk a stale
    /// reuse, so the bump sites are deliberately conservative.
    ///
    /// INVARIANT (construction-enforced, not type-enforced): this counter pairs
    /// with `PlanCaches.camera_frame_key`, which lives in a SEPARATE struct.
    /// `clear_residency` clears both together. If a future refactor ever replaces
    /// the `ResidentScene` wholesale (this counter resets to 0) WITHOUT also
    /// dropping the caches' `camera_frame_key`, a generation collision could
    /// reuse a stale fingerprint — so reset the memo (`camera_clear`) on any such
    /// swap.
    generation: u64,
    /// rust-edit-incremental-z4 (incremental plan patching): the generation of the
    /// most recent change that could alter the DISPLAY-LIST STRUCTURE (z-order,
    /// membership, visibility) — a full `update`/`clear`, a `zIndex` edit, a
    /// visibility flip (`visible`/`hidden`), an `opacity` crossing 0, or any page
    /// patch. A cached plan index built at generation `g` is only incrementally
    /// patchable when `last_structural_gen <= g` (otherwise the display list it
    /// indexed may have changed shape → full rebuild). Conservative: over-bumping
    /// only forces a (correct) full rebuild.
    last_structural_gen: u64,
    /// rust-edit-incremental-z4: a rolling log of `(generation, node_id)` for every
    /// node a non-structural `apply_props` touched, newest at the back. Lets the
    /// render path compute "which nodes changed since generation g" without an O(n)
    /// scan. Bounded by `DIRTY_LOG_CAP`; when it overflows, the window start rises
    /// and any plan index older than it falls back (safe).
    recent_edits: std::collections::VecDeque<(u64, String, bool)>,
    /// Scale RFC stage C: the generation of the most recent change the retained
    /// camera plan cannot patch at all — a full `update`/`clear`, a page patch, or
    /// a patch that failed part-way. Every other change, structural or not, is in
    /// `recent_edits` with its node ids (`dirty_detail`).
    last_hard_gen: u64,
    /// rust-edit-incremental-z4: the oldest generation `recent_edits` still fully
    /// covers. A `dirty_since(g)` query with `g < dirty_window_start_gen` cannot be
    /// answered precisely → treated as structural (full rebuild).
    dirty_window_start_gen: u64,
    /// Scoped re-expansion kill switch (`?wasmHostReexpand=0`). Default ENABLED.
    /// When off, `apply_props` reverts to the earlier behaviour: bail on any node with a
    /// transient expansion so the host heals with a whole-scene re-feed. The switch
    /// lives on `ResidentScene` (not `PlanCaches`) because the decision is taken
    /// inside `apply_props`, which only ever sees the resident scene.
    host_reexpand_enabled: bool,
    /// Scoped re-expansion counters, surfaced through the wasm `residency_stats` JSON:
    /// how many transient-host prop patches took the scoped re-expansion path, and
    /// how many of those turned out STRUCTURAL (the re-expansion changed the
    /// transient id sequence → the plan index is unusable, full rebuild). A
    /// `host_reexpands` that stays 0 while a table drag is running means the lane
    /// is not engaging; a `host_reexpand_structural` that tracks `host_reexpands`
    /// 1:1 means we re-expanded but still paid the rebuild.
    host_reexpands: u64,
    host_reexpand_structural: u64,
}

/// rust-edit-incremental-z4: cap on the rolling edit log. Generous — a render
/// keeps up with edits in the steady state, so the log usually holds ≤ a handful
/// of entries between frames; the cap only guards a pathological edit burst with
/// no intervening render (which then falls back to a full rebuild — still
/// correct).
const DIRTY_LOG_CAP: usize = 4096;

/// renderer-perf-more-z6: the ONLY keys an edit to a PARENT (group/container) node
/// may carry and still take the incremental SUBTREE fast path
/// (`collect_affected_item_nodes`, which rebuilds the parent's already-item-
/// producing descendants in place). These are the inherited-down transform/opacity
/// keys: they RE-VALUE existing items (new model matrix / baked alpha) but can
/// never change WHICH items a node produces. `apply_props` classifies a parent
/// edit carrying ANY other key as STRUCTURAL → full rebuild, because such a key can
/// change item PRODUCTION that the subtree walk + post-rebuild `items_shape_match`
/// would miss (a fill-less `container` gaining a `fill` → a new body the parent had
/// no span for; a `clipMaskNodeId`/effect change re-masking descendants).
///
/// MAINTENANCE INVARIANT (deny-by-default — keep this list MINIMAL): only add a key
/// here if editing it on a parent provably cannot add/remove/re-mask any item in its
/// subtree, only re-value existing ones. In particular `opacity` is safe ONLY while
/// group/container opacity is applied value-wise per descendant item (the leaf
/// multiply-down in `frame.rs::append_node_draw_items`) — if a future
/// isolated-group-compositing feature ever makes a group render to an offscreen
/// layer, `opacity` (and the layer-triggering key) must move OUT of this set, since
/// a 1.0→0.5 edit would then change item production the subtree patch can't see.
const SAFE_PARENT_KEYS: &[&str] = &["x", "y", "rotation", "scale", "scaleX", "scaleY", "opacity"];

/// Whether an opacity edit on `node` (to `new_op`) changes which items it produces: a group or a frame with children
/// at an opacity between 0 and 1 composites as one unit in a fade split (`moda_scene::scene::is_opacity_group`), so an
/// edit that makes or keeps it one (the split carries the opacity) needs a full rebuild, not a subtree patch.
fn opacity_group_edit(node: &RawNode, new_op: f64) -> bool {
    let child_bearing = match node.node_type() {
        "group" => true,
        "container" => node.array("children").is_some_and(|c| !c.is_empty()),
        _ => false,
    };
    child_bearing && (moda_scene::scene::is_opacity_group(node) || (new_op > 0.0 && new_op < 1.0))
}

/// rust-edit-incremental-z4: the answer to "what changed since generation g?".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirtySince {
    /// Nothing changed since `g` — the cached plan is current as-is.
    Unchanged,
    /// A structural change occurred (or the window can't cover `g`) — the cached
    /// plan index is unusable; do a full rebuild.
    Structural,
    /// Exactly these node ids had non-structural property edits since `g`.
    Nodes(Vec<String>),
}

/// Scale RFC stage C: [`ResidentScene::dirty_detail`]'s answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SceneDirt {
    /// Nothing changed.
    Unchanged,
    /// A change only a full rebuild handles.
    Hard,
    /// These ids changed (each once), `true` when an edit to it could change the
    /// display list's shape.
    Nodes(Vec<(String, bool)>),
}

struct Resident {
    json_hash: u64,
    scene: Scene,
    /// Effective-node hashes (resolved ?? canonical) + transients — exactly
    /// what the plan builder reads through `effective_node`.
    hashes: HashMap<String, NodeHashes>,
    /// S2-min: true once `apply_props` mutated the resident scene — the next
    /// full feed must re-parse even when its hash matches the last fed
    /// string (the resident content has diverged from that string).
    patched: bool,
    /// The text-host generation the scene's warped text was last resolved against
    /// ([`ResidentScene::heal_text_warps`]).
    text_host_gen: u64,
}

/// Re-derive ONE transient host's expansion in place and repair the
/// resident hash map, returning whether the re-expansion was STRUCTURAL (the
/// display list changed shape → the caller forces a full plan rebuild).
///
/// The caller has already merged the patch into the host's CANONICAL node;
/// `Scene::reexpand_transient_host` re-resolves the host from canonical (variables
/// included) and rebuilds its transient records exactly as `Scene::load` of the
/// patched JSON would. See `collect_affected_item_nodes` for why splicing the
/// re-expanded children into a retained plan is byte-exact.
///
/// STRUCTURAL iff the transient id list changed **as a SEQUENCE**, not as a set:
/// `transient_by_parent` order IS display-list order (`Scene::visit_z` walks it in
/// order), so a permutation reorders draw items and invalidates every cached span
/// index just as surely as an add/remove does.
fn reexpand_host_in_place(st: &mut Resident, id: &str) -> anyhow::Result<bool> {
    use anyhow::anyhow;
    let outcome = st
        .scene
        .reexpand_transient_host(id)?
        .ok_or_else(|| anyhow!("'{id}' could not be re-expanded — resync required"))?;
    let structural = outcome.removed != outcome.added;
    // Drop hashes for transients that no longer exist. Only for ids NOT in `added`:
    // a same-id-different-content transient (the common case — a cell whose text
    // changed keeps its `<table>_cell_r_c` id) must be RE-hashed, not removed, and
    // the insert loop below does that.
    let kept: std::collections::HashSet<&str> = outcome.added.iter().map(String::as_str).collect();
    for old in &outcome.removed {
        if !kept.contains(old.as_str()) {
            st.hashes.remove(old);
        }
    }
    // Re-hash every transient the fresh expansion produced (added-or-kept alike:
    // an unchanged id can still carry changed geometry, e.g. every cell rect after
    // a host move, since `expand_table_node` bakes the host x/y into cell coords).
    for new_id in &outcome.added {
        let nh = st
            .scene
            .effective_node(new_id)
            .map(node_hashes)
            .ok_or_else(|| anyhow!("transient '{new_id}' vanished mid-re-expansion"))?;
        st.hashes.insert(new_id.clone(), nh);
    }
    // ...and the host itself. Its hash is never a draw item's own hash (it is
    // expanded away), but `fold_display_list_fingerprint` folds a `native-children`
    // host's `full` hash per transient item, so a stale host hash there
    // would make a host MOVE fingerprint-identical.
    let host_hash = st
        .scene
        .effective_node(id)
        .map(node_hashes)
        .ok_or_else(|| anyhow!("host '{id}' vanished mid-re-expansion"))?;
    st.hashes.insert(id.to_string(), host_hash);
    Ok(structural)
}

/// Prop keys whose edit changes the display-list SHAPE (z-order, membership,
/// visibility) — see `apply_props`. An `effects` edit is structural only when
/// its backdrop / layer-blur entries change ([`structural_effects`]).
fn key_changes_display_list(k: &str) -> bool {
    matches!(k, "zIndex" | "visible" | "hidden")
}

/// The whole-node half of a scene patch (`upserts`, `removes`, `pageNodeIds`,
/// `canvasRootNodeIds`).
struct SceneEdits {
    upserts: Vec<RawNode>,
    removes: Vec<String>,
    page_node_ids: Vec<(String, Vec<String>)>,
    /// The canvas root's new top-level list (a page-less Design document), when it changed.
    canvas_root_node_ids: Option<Vec<String>>,
}

/// Parse the whole-node half of a patch, refusing a malformed payload before anything in the scene is mutated.
/// `None` when the patch carries none. The upserted nodes are moved out of `patch` (it is only read for its
/// `nodes` and `pages` halves after this), not cloned: a patch that moves every layer of a large selection
/// carries thousands of whole nodes, and copying each one and then dropping the original cost as much as
/// parsing them.
///
/// ```json
/// { "upserts": [ { "id": "a", "type": "rectangle", ... } ],
///   "removes": [ "b" ],
///   "pageNodeIds": { "page-1": [ "a", "c" ] },
///   "canvasRootNodeIds": [ "f1", "a" ] }
/// ```
fn parse_scene_edits(patch: &mut Value) -> anyhow::Result<Option<SceneEdits>> {
    use anyhow::anyhow;
    let (u, r, p) = (patch.get("upserts"), patch.get("removes"), patch.get("pageNodeIds"));
    let roots = patch.get("canvasRootNodeIds");
    if u.is_none() && r.is_none() && p.is_none() && roots.is_none() {
        return Ok(None);
    }
    let mut edits = SceneEdits {
        upserts: Vec::new(),
        removes: Vec::new(),
        page_node_ids: Vec::new(),
        canvas_root_node_ids: None,
    };
    if let Some(roots) = roots {
        let ids = roots
            .as_array()
            .ok_or_else(|| anyhow!("patch.canvasRootNodeIds must be an array"))?
            .iter()
            .map(|v| v.as_str().map(str::to_string))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| anyhow!("canvasRootNodeIds entries must be strings"))?;
        edits.canvas_root_node_ids = Some(ids);
    }
    if let Some(r) = r {
        for v in r.as_array().ok_or_else(|| anyhow!("patch.removes must be an array"))? {
            let id = v.as_str().ok_or_else(|| anyhow!("patch.removes entries must be strings"))?;
            edits.removes.push(id.to_string());
        }
    }
    if let Some(p) = p {
        for (pid, ids) in p.as_object().ok_or_else(|| anyhow!("patch.pageNodeIds must be an object"))? {
            let ids = ids
                .as_array()
                .ok_or_else(|| anyhow!("pageNodeIds['{pid}'] must be an array"))?
                .iter()
                .map(|v| v.as_str().map(str::to_string))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| anyhow!("pageNodeIds['{pid}'] entries must be strings"))?;
            edits.page_node_ids.push((pid.to_string(), ids));
        }
    }
    if let Some(u) = patch.get_mut("upserts") {
        let u = u.as_array_mut().ok_or_else(|| anyhow!("patch.upserts must be an array"))?;
        edits.upserts.reserve(u.len());
        for v in u.iter_mut() {
            let node = RawNode::from_value(std::mem::take(v)).ok_or_else(|| anyhow!("upsert must be an object"))?;
            if node.id().is_empty() {
                return Err(anyhow!("upsert without an id — resync required"));
            }
            edits.upserts.push(node);
        }
    }
    Ok(Some(edits))
}

/// Apply whole-node edits to the resident scene and repair the hash map for
/// exactly the ids they touched: removed nodes and dropped transients lose their
/// hashes, written nodes and re-expanded transients are re-hashed, and nothing
/// else is. Every tessellation, text and outline cache is keyed by those
/// hashes, so an untouched node keeps its cache entries across the edit.
///
/// Returns (display-list shape may have changed, the written and removed
/// canonical ids with whether each one's edit could change the display list's
/// shape). A replace that keeps a node's structure is classified key by key, as
/// a `nodes` prop patch is, so a fill or position edit still takes the
/// incremental plan patch; an add, remove, reparent, reorder or type change is
/// structural.
fn apply_scene_edits(
    st: &mut Resident,
    edits: SceneEdits,
) -> anyhow::Result<(bool, Vec<(String, bool)>)> {
    // A flipbook's onion skin copies its neighbouring drawings when the scene is made (`moda_scene::flipbook`): an
    // edit to it or inside it re-derives them through a full feed.
    // The upserts that carry an onion skin, by id (one pass: a patch can upsert every node of the document).
    let onion_upserts: std::collections::HashSet<&str> = edits
        .upserts
        .iter()
        .filter(|n| n.has(moda_scene::flipbook::ONION_SKIN_KEY))
        .map(|n| n.id())
        .collect();
    for id in edits.upserts.iter().map(|n| n.id()).chain(edits.removes.iter().map(String::as_str)) {
        if moda_scene::flipbook::in_onion_flipbook(&st.scene, id) || onion_upserts.contains(id) {
            anyhow::bail!("'{id}' is in an onion-skinned flipbook — resync required");
        }
    }
    let mut structural = false;
    // Upsert id → whether its own change is structural (an add is).
    let mut upsert_structural: HashMap<String, bool> = HashMap::default();
    for n in &edits.upserts {
        let Some(old) = st.scene.node(n.id()) else {
            // An add: structural, decided by the scene below.
            upsert_structural.insert(n.id().to_string(), true);
            continue;
        };
        let was = structural;
        structural = false;
        let changed = old
            .fields()
            .keys()
            .filter(|k| !old.same_field(n, k))
            .map(|k| k.as_str())
            .chain(
                n.fields()
                    .keys()
                    .filter(|k| !old.fields().contains_key(*k))
                    .map(|k| k.as_str()),
            );
        let parent = moda_scene::scene::is_parent_node(old);
        for k in changed {
            if key_changes_display_list(k) || (parent && !SAFE_PARENT_KEYS.contains(&k)) {
                structural = true;
            }
        }
        if structural_effects(old.json_field("effects")) != structural_effects(n.json_field("effects")) {
            structural = true;
        }
        let old_op = old.f64("opacity").unwrap_or(1.0);
        let new_op = n.f64("opacity").unwrap_or(1.0);
        if (old_op <= 0.0) != (new_op <= 0.0) || (old_op != new_op && opacity_group_edit(old, new_op)) {
            structural = true;
        }
        upsert_structural.insert(n.id().to_string(), structural);
        structural |= was;
    }
    // The top-level lists a patch names: the retained camera plan diffs them.
    let mut lists: Vec<(String, bool)> = edits
        .page_node_ids
        .iter()
        .map(|(pid, _)| (crate::camera_slot::page_list_key(pid), true))
        .collect();
    if edits.canvas_root_node_ids.is_some() {
        lists.push((crate::camera_slot::CANVAS_LIST_KEY.to_string(), true));
    }
    // What each existing upsert draws as: a text layer gaining or losing a geometry effect switches between text
    // and a path (`moda_scene::text_warp`), which changes its display-list item kind.
    let drawn_before: Vec<(String, String)> = edits
        .upserts
        .iter()
        .filter_map(|n| {
            let t = st.scene.effective_node(n.id())?.node_type().to_string();
            Some((n.id().to_string(), t))
        })
        .collect();
    let outcome = st
        .scene
        .apply_node_edits(
            edits.upserts,
            &edits.removes,
            &edits.page_node_ids,
            edits.canvas_root_node_ids.as_deref(),
        )
        .map_err(|e| anyhow::anyhow!("{e} — resync required"))?;
    // A host whose expansion changed shape reorders or resizes its slice of the
    // display list, exactly as `reexpand_host_in_place` reports it.
    // An envelope (E8.17) re-bent by the edit: its transients are derived from everything inside it, so the plan
    // rebuilds from scratch rather than patching them one by one.
    let envelope_rebent = outcome.envelopes_rebent;
    let reshaped = outcome.structural || envelope_rebent || outcome.transients_removed != outcome.transients_added;
    structural |= reshaped;
    for (id, was) in &drawn_before {
        if st.scene.effective_node(id).is_some_and(|n| n.node_type() != was) {
            upsert_structural.insert(id.clone(), true);
            structural = true;
        }
    }
    for id in outcome.removed.iter().chain(&outcome.transients_removed) {
        st.hashes.remove(id);
    }
    for id in outcome.transients_added.iter().chain(&outcome.written) {
        let nh = st
            .scene
            .effective_node(id)
            .map(|n| node_hashes_in(&st.scene, n))
            .ok_or_else(|| anyhow::anyhow!("node '{id}' vanished mid-edit"))?;
        st.hashes.insert(id.clone(), nh);
    }
    // A reshaped scene (membership, hierarchy or order moved) may have moved any
    // written node; a removed node's slot is gone.
    let mut touched: Vec<(String, bool)> = outcome
        .written
        .into_iter()
        .map(|id| {
            let s = reshaped || upsert_structural.get(&id).copied().unwrap_or(true);
            (id, s)
        })
        .collect();
    touched.extend(outcome.removed.into_iter().map(|id| (id, true)));
    touched.extend(lists);
    Ok((structural, touched))
}

/// First difference between two maps, for [`ResidentScene::divergence_from_full_load`].
fn map_divergence<V: PartialEq + std::fmt::Debug, S: std::hash::BuildHasher>(
    what: &str,
    patched: &std::collections::HashMap<String, V, S>,
    full: &std::collections::HashMap<String, V, S>,
) -> Option<String> {
    if let Some(k) = patched.keys().find(|k| !full.contains_key(*k)) {
        return Some(format!("{what}: '{k}' only in the patched scene"));
    }
    if let Some(k) = full.keys().find(|k| !patched.contains_key(*k)) {
        return Some(format!("{what}: '{k}' missing from the patched scene"));
    }
    let mut keys: Vec<&String> = full.keys().collect();
    keys.sort();
    keys.into_iter()
        .find(|k| patched[*k] != full[*k])
        .map(|k| format!("{what}: '{k}' differs: patched {:?} vs full {:?}", patched[k], full[k]))
}

impl Default for ResidentScene {
    fn default() -> Self {
        ResidentScene {
            state: None,
            generation: 0,
            last_structural_gen: 0,
            recent_edits: std::collections::VecDeque::new(),
            last_hard_gen: 0,
            dirty_window_start_gen: 0,
            host_reexpand_enabled: true,
            host_reexpands: 0,
            host_reexpand_structural: 0,
        }
    }
}

impl ResidentScene {
    /// Returns the resident scene for `scene_json`, re-hydrating only when
    /// the content hash changed. Err = JSON parse failure (caller falls back
    /// to the Stage-0 path / surfaces the error like Stage 0 did).
    pub fn update(&mut self, scene_json: &str) -> Result<(), moda_scene::value::Error> {
        self.update_metered(scene_json, None).map(|_| ())
    }

    /// [`Self::update`], and when it re-parsed, the new scene's heap footprint by `live_bytes` (the host's
    /// allocator count): what it holds after the parse over what it held once the old scene was gone.
    /// `Ok(None)` when the resident scene already was this document.
    pub fn update_metered(
        &mut self,
        scene_json: &str,
        live_bytes: Option<&dyn Fn() -> usize>,
    ) -> Result<Option<usize>, moda_scene::value::Error> {
        let json_hash = hash_bytes(scene_json.as_bytes());
        if matches!(&self.state, Some(r) if r.json_hash == json_hash && !r.patched) {
            self.heal_text_warps()?;
            return Ok(None);
        }
        // The old scene goes BEFORE the new document parses, so a full feed's peak is
        // the parsed JSON plus the scene built from it — not also the old scene (at
        // 100k nodes that is half a gigabyte of wasm linear memory, which never
        // shrinks). A feed that fails to parse leaves no resident scene; the host's
        // next full feed rebuilds it.
        self.state = None;
        let base = live_bytes.map(|f| f());
        let scene = Scene::load_str(scene_json)?;
        let hashes = node_hashes_for_scene(&scene);
        self.state = Some(Resident {
            json_hash,
            scene,
            hashes,
            patched: false,
            text_host_gen: crate::text_draw::text_host_generation(),
        });
        // M1: a real re-parse changed the resident content (the early return
        // above covers the no-change case; a parse error returns before here).
        self.generation = next_generation();
        // rust-edit-incremental-z4: a full re-feed can change anything (z-order,
        // membership, …), so it is structural; reset the dirty window to here and
        // drop the per-node log (its history predates a wholesale re-parse).
        self.last_structural_gen = self.generation;
        self.last_hard_gen = self.generation;
        self.recent_edits.clear();
        self.dirty_window_start_gen = self.generation;
        Ok(Some(match (base, live_bytes) {
            (Some(base), Some(f)) => f().saturating_sub(base),
            _ => 0,
        }))
    }

    /// Live geometry effects on text (`moda_scene::text_warp`) and booleans cutting text: once fonts install (the
    /// text-host generation moves), re-resolve the text layers that could not be outlined before and the booleans
    /// whose text operand was outlined with the earlier faces, and replan them (a structural change). Cheap when no
    /// font was installed since the last call. Returns whether any healed.
    pub fn heal_text_warps(&mut self) -> Result<bool, moda_scene::charts::ticks::TickError> {
        let now = crate::text_draw::text_host_generation();
        let Some(st) = self.state.as_mut() else {
            return Ok(false);
        };
        if st.text_host_gen == now {
            return Ok(false);
        }
        let mut healed = st.scene.retry_pending_text_warps();
        healed.extend(st.scene.retry_text_booleans());
        let tables = st.scene.refit_auto_tables()?;
        st.text_host_gen = now;
        if !tables.is_empty() { st.hashes = node_hashes_for_scene(&st.scene); }
        healed.extend(tables);
        if healed.is_empty() {
            return Ok(false);
        }
        for id in &healed {
            if let Some(nh) = st.scene.effective_node(id).map(|n| node_hashes_in(&st.scene, n)) {
                st.hashes.insert(id.clone(), nh);
            }
        }
        let gen = next_generation();
        self.generation = gen;
        self.last_structural_gen = gen;
        for id in healed {
            self.record_edit(gen, id, true);
        }
        Ok(true)
    }

    /// rust-edit-incremental-z4: append a non-structural node edit to the rolling
    /// dirty log, evicting + advancing the coverage window when it overflows.
    fn record_edit(&mut self, gen: u64, id: String, structural: bool) {
        self.recent_edits.push_back((gen, id, structural));
        while self.recent_edits.len() > DIRTY_LOG_CAP {
            if let Some((g, _, _)) = self.recent_edits.pop_front() {
                self.dirty_window_start_gen = self.dirty_window_start_gen.max(g);
            }
        }
    }

    /// rust-edit-incremental-z4: which nodes changed since a plan index was built
    /// at generation `since_gen` — the gate for the incremental patcher. Returns
    /// `Structural` (→ full rebuild) on any display-list-shape change since then,
    /// or when the log can't cover `since_gen`; `Unchanged` when nothing changed;
    /// else the exact set of property-edited node ids.
    pub fn dirty_since(&self, since_gen: u64) -> DirtySince {
        if since_gen == self.generation {
            return DirtySince::Unchanged;
        }
        if self.last_structural_gen > since_gen || since_gen < self.dirty_window_start_gen {
            return DirtySince::Structural;
        }
        let mut seen = std::collections::HashSet::new();
        let mut nodes = Vec::new();
        for (g, id, _) in &self.recent_edits {
            if *g > since_gen && seen.insert(id.as_str()) {
                nodes.push(id.clone());
            }
        }
        DirtySince::Nodes(nodes)
    }

    /// Scale RFC stage C: what changed since generation `since_gen`, for the
    /// retained camera plan (`camera_slot`), which patches structural edits too.
    /// `Hard` = a change it cannot patch (a full feed, a page patch, a partial
    /// failure) or a log that no longer reaches back that far. Otherwise every
    /// node id an edit touched — written, re-expanded or removed — once, flagged
    /// when any of its edits could change the display list's shape (added,
    /// removed, reparented, reordered, hidden/shown, a parent's non-transform
    /// key: see `apply_props`).
    pub fn dirty_detail(&self, since_gen: u64) -> SceneDirt {
        if since_gen == self.generation {
            return SceneDirt::Unchanged;
        }
        if self.last_hard_gen > since_gen || since_gen < self.dirty_window_start_gen {
            return SceneDirt::Hard;
        }
        let mut at: HashMap<&str, usize> = HashMap::default();
        let mut nodes: Vec<(String, bool)> = Vec::new();
        for (g, id, structural) in &self.recent_edits {
            if *g <= since_gen {
                continue;
            }
            match at.get(id.as_str()) {
                Some(&i) => nodes[i].1 |= *structural,
                None => {
                    at.insert(id.as_str(), nodes.len());
                    nodes.push((id.clone(), *structural));
                }
            }
        }
        SceneDirt::Nodes(nodes)
    }

    /// S2-min (task #95): patches against the resident scene (`set_node_props`).
    ///
    /// - `upserts` / `removes` / `pageNodeIds` (the editor's scene-sync lane):
    ///   whole nodes added or replaced, nodes removed, pages' top-level lists
    ///   replaced — see [`apply_scene_edits`]. Applied first.
    /// - `nodes` / `pages` (the override lane during playback): merge top-level
    ///   props into canonical + resolved + transient copies.
    ///
    /// Both re-hash exactly the touched ids, so every hash-keyed cache
    /// self-corrects and untouched nodes keep theirs. ANY doubt (unknown id,
    /// structural key in a prop merge, malformed payload) errors — the host
    /// falls back to a full feed. Always.
    pub fn apply_props(&mut self, patch_json: &str) -> anyhow::Result<()> {
        use anyhow::{anyhow, bail};
        if self.state.is_none() {
            bail!("no resident scene to patch");
        }
        // Parse first — it's pure and mutates nothing, so a malformed payload
        // bails WITHOUT needlessly invalidating the fingerprint memo.
        let mut patch: Value = moda_scene::value::from_str(patch_json)?;
        let edits = parse_scene_edits(&mut patch)?;
        // M1: a patch can still mutate some nodes before a later node's
        // structural-key bail, so bump the generation now (after the parse,
        // before any mutation) — any mutation, even a partial-then-error one,
        // must invalidate.
        self.generation = next_generation();
        let gen = self.generation;
        // Read the kill switch + take the counters OUT of `self` before the
        // closure below borrows `self.state` mutably. The counters are locals so they
        // still count a patch that mutated some hosts and then bailed on a later node
        // (the partial-mutation path) — they are diagnostics, and an undercount there
        // would hide exactly the case worth seeing.
        let host_reexpand_enabled = self.host_reexpand_enabled;
        let mut host_reexpands: u64 = 0;
        let mut host_reexpand_structural: u64 = 0;
        // rust-edit-incremental-z4: do the mutation in an inner scope that returns
        // (dirty node set, whether the edit could change the DISPLAY-LIST STRUCTURE).
        // Wrapping it lets us treat ANY error (incl. a partial-then-`bail!`) as a
        // structural invalidation below: a partial mutation is already applied to
        // the resident scene, so the next render MUST do a full rebuild rather than
        // serve a stale plan index — even if the host hasn't re-fed yet.
        let result: anyhow::Result<(Vec<(String, bool)>, bool, bool)> = (|| {
            // (id, whether its edit changes the display list's shape)
            let mut touched: Vec<(String, bool)> = Vec::new();
            let mut structural = false;
            // A change the retained camera plan cannot patch (`last_hard_gen`).
            let mut hard = false;
            let st = self
                .state
                .as_mut()
                .expect("resident scene present (checked above)");
            // Mark patched BEFORE any mutation: a patch can mutate earlier nodes and
            // then bail on a later one (structural key, unknown id, transient-expansion
            // parent). The host heals a rejected `set_node_props` by re-feeding the
            // SAME scene JSON, so if `patched` were still false the hash-skip in
            // `update` would skip re-parsing and strand the partial mutation. Setting
            // it here forces that re-feed to re-hydrate. (On full success it means the
            // same thing it always did: the resident content diverged from any fed
            // string.)
            st.patched = true;
            const STRUCTURAL: &[&str] = &["id", "type", "parentId", "children", "nodeIds"];
            // Whole-node edits (the editor's scene-sync lane): added, replaced and
            // removed nodes and page order, applied before any `nodes` prop merge.
            if let Some(edits) = edits {
                let (edit_structural, written) = apply_scene_edits(st, edits)?;
                structural |= edit_structural;
                touched.extend(written);
            }
            // The per-node classification below: every `structural = true` in the
            // loop is this node's.
            let mut node_structural;
            if let Some(nodes) = patch.get("nodes") {
                let nodes = nodes
                    .as_object()
                    .ok_or_else(|| anyhow!("patch.nodes must be an object"))?;
                for (id, props) in nodes {
                    node_structural = false;
                    let props = props
                        .as_object()
                        .ok_or_else(|| anyhow!("patch for '{id}' must be an object"))?;
                    if props.keys().any(|k| STRUCTURAL.contains(&k.as_str())) {
                        bail!("structural key in patch for '{id}' — resync required");
                    }
                    // A paint layer (raster spike) hydrated as a rectangle whose fill, strokes and corner radius are
                    // derived from its size and `surface`: re-derive them for the merged node, so a property patch
                    // lands where a full load of the patched node would.
                    let paint_props;
                    let props = match st.scene.effective_node(id) {
                        Some(n) if n.json_field("_paintLayer").and_then(Value::as_bool) == Some(true) => {
                            let mut merged = n.to_json_map();
                            for (k, v) in props {
                                merged.insert(k.clone(), v.clone());
                            }
                            let mut raw = moda_scene::types::RawNode::new(merged);
                            moda_scene::normalize::normalize_node_in_place(&mut raw);
                            let mut p = props.clone();
                            for k in ["fills", "strokes", "cornerRadius"] {
                                if let Some(v) = raw.json_field(k) {
                                    p.insert(k.into(), v.clone());
                                }
                            }
                            paint_props = p;
                            &paint_props
                        }
                        _ => props,
                    };
                    // E7.1: a boolean node's drawn path is DERIVED from its descendants'
                    // geometry and its operation, so a patch to either re-solves it
                    // through a full feed (the result cache keeps that to the one node).
                    // E8.17: an envelope draws transients derived from everything inside it; an edit to it or
                    // to its content re-bends it through a full feed.
                    // A repeat's copies are transients of the same kind, derived from its children.
                    if st.scene.in_envelope(id) || props.get("envelope").is_some() || props.get("repeat").is_some() {
                        bail!("'{id}' feeds an envelope or a repeat — resync required");
                    }
                    // A flipbook's onion skin (`moda_scene::flipbook`) is copied from its drawings when the scene is
                    // made, so an edit to it, inside it, or turning it on re-derives it through a full feed.
                    if moda_scene::flipbook::in_onion_flipbook(&st.scene, id)
                        || props.contains_key(moda_scene::flipbook::ONION_SKIN_KEY)
                    {
                        bail!("'{id}' is in an onion-skinned flipbook — resync required");
                    }
                    if st.scene.has_boolean_ancestor(id)
                        || (st.scene.pre_boolean_record(id).is_some()
                            && props.contains_key("booleanOperation"))
                    {
                        bail!("'{id}' feeds a boolean node's result — resync required");
                    }
                    // E8.2: a group's appearance paints an outline DERIVED from its descendants, and draws
                    // outside every node span, so an edit to the group or to anything inside it re-plans
                    // through a full feed (documents without group appearance never take this).
                    if st.scene.group_appearance(id).is_some()
                        || st.scene.has_group_appearance_ancestor(id)
                        || (st.scene.effective_node(id).is_some_and(|n| n.node_type() == "group")
                            && ["fills", "strokes", moda_scene::appearance::APPEARANCE_ORDER_KEY]
                                .iter()
                                .any(|k| props.contains_key(*k)))
                    {
                        bail!("'{id}' feeds a group's appearance — resync required");
                    }
                    // Nodes with generated/transient children (tables,
                    // charts) need their expansion RE-DERIVED, because expansion
                    // bakes host state into the children — a table's cell rects
                    // carry the host x/y (`expand_table_node`), its row/column
                    // sizes, its stroke config, its cell text. Previously this
                    // bailed and the host healed with a whole-scene JSON re-feed +
                    // cold plan rebuild (~23 ms on a 2000-item table, ~150 ms at
                    // 10000 — the cost tracked the WHOLE SCENE, not the edited
                    // host). Now we re-expand just this host; see
                    // `reexpand_host_in_place` for the correctness argument.
                    let is_transient_host = st.scene.transient_by_parent().contains_key(id.as_str());
                    if is_transient_host && !host_reexpand_enabled {
                        bail!("'{id}' has transient expansion — resync required (host re-expansion disabled)");
                    }
                    // rust-edit-incremental-z4: classify the edit. `zIndex`,
                    // `visible`/`hidden`, an `opacity` crossing 0 (all change the
                    // display-list shape — `visit_z` skips opacity==0 / hidden nodes,
                    // zIndex reorders), or an `effects` edit that changes its
                    // backdrop entries (adds/removes a backdrop split + flips the
                    // page sample_count/transparent metadata) or its layer blurs
                    // (diverts/undiverts the node's items into a capture group,
                    // [`structural_effects`]) are STRUCTURAL → force a full
                    // rebuild. All other property edits are incrementally
                    // patchable.
                    for k in props.keys() {
                        if matches!(k.as_str(), "zIndex" | "visible" | "hidden") {
                            node_structural = true;
                        }
                    }
                    // E8.2: effects on a fill or stroke (their captures divert the layer's items, and turn the
                    // page's capture context on) and the stack order re-plan the node from scratch.
                    let entry_fx = |v: Option<&Value>| {
                        v.and_then(Value::as_array).is_some_and(|a| {
                            a.iter().any(|l| l.get("effects").is_some_and(|e| !e.is_null()))
                        })
                    };
                    let old_node = st.scene.effective_node(id);
                    if props.contains_key(moda_scene::appearance::APPEARANCE_ORDER_KEY)
                        || ["fills", "strokes"].iter().any(|k| {
                            props.contains_key(*k)
                                && (entry_fx(props.get(*k)) || entry_fx(old_node.and_then(|n| n.json_field(k))))
                        })
                    {
                        node_structural = true;
                    }
                    if let Some(new_effects) = props.get("effects") {
                        let old_effects = st.scene.effective_node(id).and_then(|n| n.json_field("effects"));
                        if structural_effects(old_effects) != structural_effects(Some(new_effects)) {
                            node_structural = true;
                        }
                    }
                    if let Some(new_op) = props.get("opacity").and_then(Value::as_f64) {
                        let old_node = st.scene.effective_node(id);
                        let old_op = old_node.and_then(|n| n.f64("opacity")).unwrap_or(1.0);
                        if (old_op <= 0.0) != (new_op <= 0.0)
                            || (old_op != new_op && old_node.is_some_and(|n| opacity_group_edit(n, new_op)))
                        {
                            node_structural = true;
                        }
                    }
                    // renderer-perf-more-z6: a PARENT (group/container) edit takes the
                    // incremental SUBTREE fast path only for inherited transform/opacity
                    // keys; any other key can change item production the subtree walk would
                    // miss, so force a full rebuild. See `SAFE_PARENT_KEYS` (module-level)
                    // for the full rationale + maintenance invariant. Leaf edits are
                    // unaffected (a leaf always has a span; post-rebuild shape-match catches
                    // any item-count change).
                    let is_parent = st
                        .scene
                        .effective_node(id)
                        .is_some_and(moda_scene::scene::is_parent_node);
                    if is_parent
                        && props
                            .keys()
                            .any(|k| !SAFE_PARENT_KEYS.contains(&k.as_str()))
                    {
                        node_structural = true;
                    }
                    touched.push((id.to_string(), false));
                    let touched_at = touched.len() - 1;
                    if is_transient_host {
                        // DELIBERATE ASYMMETRY vs the non-host branch
                        // below. A host's props are merged into the CANONICAL node
                        // ONLY; its `resolved_by_id` entry is then replaced wholesale
                        // by `reexpand_transient_host`, which re-runs
                        // `variables::resolve_node` from canonical. That is STRICTER
                        // than the blind resolved-merge used for non-hosts, and it has
                        // to be: a cell's HTML resolves to a different value than the
                        // raw patch carries (its `var(--nf-variable-…, <literal>)`
                        // runs become the literals), and the expansion consumes the
                        // RESOLVED node. Blind-merging the raw value into `resolved`
                        // would bake the unresolved CSS references into the cells.
                        // (A host also never has a `transient_by_id` entry of its own.
                        // A `table`/`generated` host is additionally expanded AWAY —
                        // `Scene::visit_z` substitutes its transients, so it owns no
                        // display-list item.)
                        let Some(node) = st.scene.node_mut(id) else {
                            bail!("unknown node '{id}' in patch — resync required");
                        };
                        for (k, v) in props {
                            node.set(k, v.clone());
                        }
                        host_reexpands = host_reexpands.saturating_add(1);
                        if reexpand_host_in_place(st, id)? {
                            node_structural = true;
                            host_reexpand_structural = host_reexpand_structural.saturating_add(1);
                        }
                        // NOTE: only the HOST lands in `touched` (→ the dirty log), not
                        // its transients. `collect_affected_item_nodes` walks the host
                        // down into its transient children, so one entry covers them
                        // all — and it MUST, because a 20x20 table expands to ~1.6k
                        // transients and logging each would blow `DIRTY_LOG_CAP` (4096)
                        // in three edits, forcing every plan index older than the
                        // window to fall back to a full rebuild. Pinned by
                        // `apply_props_host_patch_logs_only_the_host`.
                        touched[touched_at].1 = node_structural;
                        structural |= node_structural;
                        continue;
                    }
                    let mut found = false;
                    // Live geometry effects (E8.1) derive the resolved record's
                    // `type` / `points` from the canonical node: a node that has
                    // them before or after this patch re-resolves from canonical
                    // instead of taking the blind merge below.
                    // A text layer with a pattern glyph fill resolves to its outlines too (`text_warp`).
                    let derives_path = |n: &RawNode| {
                        moda_scene::live_effects::has_geometry_effects(n) || moda_scene::text_warp::is_effected_text(n)
                    };
                    let had_geometry_effects = st.scene.node(id).is_some_and(derives_path);
                    if let Some(node) = st.scene.node_mut(id) {
                        for (k, v) in props {
                            node.set(k, v.clone());
                        }
                        found = true;
                    }
                    let geometry_effects = had_geometry_effects || st.scene.node(id).is_some_and(derives_path);
                    if geometry_effects {
                        let drawn_type = |s: &moda_scene::Scene| {
                            s.effective_node(id).map(|n| n.node_type().to_string())
                        };
                        let before = drawn_type(&st.scene);
                        st.scene.re_resolve_node(id);
                        // A text layer's effects make it draw as a path (`moda_scene::text_warp`): a patch that
                        // adds or drops them changes its display-list item kind (text ↔ shape), a rebuild.
                        if drawn_type(&st.scene) != before {
                            node_structural = true;
                        }
                    } else if let Some(node) = st.scene.resolved_mut(id.as_str()) {
                        // Only a node whose record differs from canonical has an
                        // entry (`Scene::resolved_by_id`); the rest render from
                        // the canonical node merged above.
                        for (k, v) in props {
                            node.set(k, v.clone());
                        }
                        found = true;
                        st.scene.prune_resolved(id);
                    }
                    if let Some(node) = st.scene.transient_mut(id.as_str()) {
                        for (k, v) in props {
                            node.set(k, v.clone());
                        }
                        found = true;
                    }
                    if !found {
                        bail!("unknown node '{id}' in patch — resync required");
                    }
                    let nh = st
                        .scene
                        .effective_node(id)
                        .map(|n| node_hashes_in(&st.scene, n))
                        .ok_or_else(|| anyhow!("node '{id}' vanished mid-patch"))?;
                    st.hashes.insert(id.to_string(), nh);
                    touched[touched_at].1 = node_structural;
                    structural |= node_structural;
                }
            }
            // Area type (E8.14): a prop patch that adds or drops Text Wrap or a thread link re-indexes text flow.
            if patch.get("nodes").and_then(Value::as_object).is_some_and(|nodes| {
                nodes.values().any(|p| {
                    p.as_object()
                        .is_some_and(|p| p.contains_key("wrapsText") || p.contains_key("threadPrev"))
                })
            }) {
                st.scene.refresh_text_flow_index();
            }
            // The text whose lines fit around, or continue from, what the patch touched re-lays out with it.
            let deps = crate::text_area::text_dependents(&st.scene, touched.iter().map(|(id, _)| id.as_str()));
            for id in deps {
                if !touched.iter().any(|(t, _)| *t == id) {
                    // A warped text layer's path is laid out from the same area input.
                    if st
                        .scene
                        .node(&id)
                        .is_some_and(moda_scene::text_warp::is_effected_text)
                    {
                        st.scene.re_resolve_node(&id);
                        if let Some(nh) = st.scene.effective_node(&id).map(|n| node_hashes_in(&st.scene, n)) {
                            st.hashes.insert(id.clone(), nh);
                        }
                    }
                    touched.push((id, false));
                }
            }
            if let Some(pages) = patch.get("pages") {
                // rust-edit-incremental-z4: a page edit (fill / size / position) shifts
                // the page-fill quad and the page clip on EVERY node — treat as
                // structural so the incremental patcher falls back to a full rebuild.
                structural = true;
                hard = true;
                let pages = pages
                    .as_object()
                    .ok_or_else(|| anyhow!("patch.pages must be an object"))?;
                for (id, props) in pages {
                    let props = props
                        .as_object()
                        .ok_or_else(|| anyhow!("page patch for '{id}' must be an object"))?;
                    if props.keys().any(|k| STRUCTURAL.contains(&k.as_str())) {
                        bail!("structural key in page patch for '{id}' — resync required");
                    }
                    // A page has no canonical row: `page_mut` dirties everything.
                    let page = st
                        .scene
                        .page_mut(id.as_ref())
                        .ok_or_else(|| anyhow!("unknown page '{id}' in patch — resync required"))?;
                    for (k, v) in props {
                        page.0.insert(k.clone(), v.clone());
                    }
                    // The page background is DERIVED, not stored — the
                    // `__page_bg_<pageId>` transient rect that actually paints the
                    // background is synthesized from the page's own props
                    // (`create_rectangle_from_page`) and sits first in the page's
                    // z-order. Writing the page props above and stopping there would
                    // leave the engine holding new page props and the OLD background
                    // quad. `structural = true` does NOT save us: it forces a full plan
                    // rebuild FROM THE RESIDENT SCENE, and the resident scene's bg is
                    // exactly what is stale.
                    //
                    // RE-DERIVE, never merge: `apply_props` is insert-only, so merging
                    // the patch onto the existing bg could not drop the
                    // `fillLinearGradient*` keys a `linear-gradient` -> `color` transition
                    // removes. Re-derivation also picks up the parts of the surface the
                    // raw patch does not carry — width/height from the page size, the
                    // `fillPriority` gate and the nullish `#ffffff` default.
                    //
                    // Only the pages this patch names are re-derived. Bookkeeping
                    // follows `reexpand_host_in_place`.
                    let outcome = st.scene.reexpand_page_background(id).ok_or_else(|| {
                        anyhow!("page '{id}' background could not be re-derived — resync required")
                    })?;
                    let kept: std::collections::HashSet<&str> =
                        outcome.added.iter().map(String::as_str).collect();
                    for old in &outcome.removed {
                        if !kept.contains(old.as_str()) {
                            st.hashes.remove(old);
                        }
                    }
                    for bg_id in &outcome.added {
                        let nh =
                            st.scene
                                .effective_node(bg_id)
                                .map(node_hashes)
                                .ok_or_else(|| {
                                    anyhow!("page background '{bg_id}' vanished mid-re-derive")
                                })?;
                        st.hashes.insert(bg_id.clone(), nh);
                        // The bg is a real display-list item of its own (unlike a
                        // table host, which is expanded away), so the dirty log has
                        // to name it for the node lane to agree with the page lane.
                        touched.push((bg_id.clone(), true));
                    }
                }
            }
            Ok((touched, structural, hard))
        })();
        self.host_reexpands = self.host_reexpands.wrapping_add(host_reexpands);
        self.host_reexpand_structural = self
            .host_reexpand_structural
            .wrapping_add(host_reexpand_structural);
        match result {
            Ok((touched, structural, hard)) => {
                // rust-edit-incremental-z4: commit the dirty bookkeeping. A structural
                // edit bumps the watermark (invalidating every prior plan index); every
                // touched node lands in the rolling log so the next render patches
                // exactly those nodes.
                if structural {
                    self.last_structural_gen = gen;
                }
                if hard {
                    self.last_hard_gen = gen;
                }
                for (id, node_structural) in touched {
                    self.record_edit(gen, id, node_structural);
                }
                Ok(())
            }
            Err(e) => {
                // rust-edit-incremental-z4: a partial-then-error patch already mutated
                // the resident scene but recorded nothing in the dirty log. Invalidate
                // every prior plan index so the next render does a correct full rebuild
                // regardless of when (or whether) the host re-feeds.
                self.last_structural_gen = gen;
                self.last_hard_gen = gen;
                Err(e)
            }
        }
    }

    pub fn scene(&self) -> Option<&Scene> {
        self.state.as_ref().map(|r| &r.scene)
    }

    /// The patch lane's oracle: compare the resident (patched) scene with a
    /// cold `Scene::load` of `scene_json` — nodes, pages, every index, the
    /// resolved and transient records, and the per-node hashes every cache keys
    /// on. `None` when they agree; otherwise the first difference. Node ORDER in
    /// the flat array is not compared (nothing renders from it: z-order comes
    /// from page lists and `children`), only membership.
    ///
    /// For tests, benches and the editor's opt-in verify mode; it parses and
    /// hydrates the whole document, so never on a hot path.
    pub fn divergence_from_full_load(&self, scene_json: &str) -> Option<String> {
        let Some(st) = &self.state else {
            return Some("no resident scene".into());
        };
        let value: Value = match moda_scene::value::from_str(scene_json) {
            Ok(v) => v,
            Err(e) => return Some(format!("scene JSON does not parse: {e}")),
        };
        let full = match Scene::load(moda_scene::extract_canvas_data(&value)) {
            Ok(full) => full,
            Err(error) => return Some(format!("full scene load refused: {error}")),
        };
        let a = &st.scene;
        let nodes = |s: &Scene| -> HashMap<String, RawNode> {
            s.doc().nodes.iter().map(|n| (n.id().to_string(), n.clone())).collect()
        };
        if a.doc().pages != full.doc().pages {
            let i = a
                .doc()
                .pages
                .iter()
                .zip(&full.doc().pages)
                .position(|(x, y)| x != y)
                .unwrap_or(a.doc().pages.len().min(full.doc().pages.len()));
            return Some(format!("pages differ at index {i}"));
        }
        if a.doc().canvas_root_node_ids != full.doc().canvas_root_node_ids {
            return Some("canvasRootNodeIds differ".into());
        }
        let mut order_a = a.node_order().clone();
        let mut order_b = full.node_order().clone();
        order_a.sort();
        order_b.sort();
        if order_a != order_b {
            return Some("node membership differs".into());
        }
        map_divergence("node", &nodes(a), &nodes(&full))
            .or_else(|| map_divergence("parent_map", &a.parent_map(), &full.parent_map()))
            .or_else(|| {
                map_divergence(
                    "children_by_parent",
                    &a.children_by_parent(),
                    &full.children_by_parent(),
                )
            })
            .or_else(|| {
                map_divergence("page_id_by_node", &a.page_id_by_node(), &full.page_id_by_node())
            })
            .or_else(|| map_divergence("resolved", &a.resolved_by_id(), &full.resolved_by_id()))
            .or_else(|| map_divergence("transient", &a.transient_by_id(), &full.transient_by_id()))
            .or_else(|| {
                map_divergence(
                    "transient_by_parent",
                    &a.transient_by_parent(),
                    &full.transient_by_parent(),
                )
            })
            .or_else(|| {
                map_divergence(
                    "transient_parent_of",
                    &a.transient_parent_of(),
                    &full.transient_parent_of(),
                )
            })
            .or_else(|| {
                map_divergence(
                    "transient_page_of",
                    &a.transient_page_of(),
                    &full.transient_page_of(),
                )
            })
            .or_else(|| map_divergence("hash", &st.hashes, &node_hashes_for_scene(&full)))
            .or_else(|| {
                (a.text_flow() != full.text_flow()).then(|| {
                    format!("text_flow: patched {:?} vs full {:?}", a.text_flow(), full.text_flow())
                })
            })
    }

    pub fn hashes(&self) -> Option<&HashMap<String, NodeHashes>> {
        self.state.as_ref().map(|r| &r.hashes)
    }

    /// M1: the monotonic resident-generation counter (see the field doc). The
    /// camera fingerprint memo reads this to cheaply detect an unchanged scene.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Scoped re-expansion kill switch (`?wasmHostReexpand=0`). Default ENABLED —
    /// the host calls this with `false` only to force the earlier behaviour (every
    /// transient-host prop patch rejected → whole-scene re-feed heal).
    pub fn set_host_reexpand_enabled(&mut self, on: bool) {
        self.host_reexpand_enabled = on;
    }

    /// Counters: (scoped re-expansions, of which structural).
    pub fn host_reexpand_stats(&self) -> (u64, u64) {
        (self.host_reexpands, self.host_reexpand_structural)
    }

    /// Swap in another resident scene (a parked Design page coming back, or an empty one) and return the one
    /// that was resident. Settings and diagnostics stay with `self`: the host's re-expansion switch applies to
    /// whichever scene is resident, and its counters keep counting across swaps.
    pub fn swap_scene(&mut self, mut other: ResidentScene) -> ResidentScene {
        other.host_reexpand_enabled = self.host_reexpand_enabled;
        std::mem::swap(&mut other.host_reexpands, &mut self.host_reexpands);
        std::mem::swap(&mut other.host_reexpand_structural, &mut self.host_reexpand_structural);
        std::mem::replace(self, other)
    }

    /// A resident scene holding `scene` (one built in process, not fed as JSON: `moda-render serve`'s motion handles
    /// keep the graph lane's own hydrated document resident). As a full feed, at a new generation; it counts as
    /// patched, so a later [`Self::update`] always re-parses.
    pub fn from_scene(scene: Scene) -> ResidentScene {
        let mut r = ResidentScene::default();
        let hashes = node_hashes_for_scene(&scene);
        r.state = Some(Resident {
            json_hash: 0,
            scene,
            hashes,
            patched: true,
            text_host_gen: crate::text_draw::text_host_generation(),
        });
        r.generation = next_generation();
        r.last_structural_gen = r.generation;
        r.last_hard_gen = r.generation;
        r.dirty_window_start_gen = r.generation;
        r
    }

    /// A resident scene with nothing loaded, at a generation no other scene has had.
    pub fn empty() -> ResidentScene {
        let mut r = ResidentScene::default();
        r.clear();
        r
    }

    pub fn clear(&mut self) {
        self.state = None;
        // M1: dropping the resident scene is a content change — bump so a
        // clear → rebuild cycle never collides with the pre-clear generation.
        self.generation = next_generation();
        // rust-edit-incremental-z4: a clear is structural; reset the dirty window.
        self.last_structural_gen = self.generation;
        self.last_hard_gen = self.generation;
        self.recent_edits.clear();
        self.dirty_window_start_gen = self.generation;
    }
}

// ---------------------------------------------------------------------------
// R3/R4/R5/R6 — plan-level caches
// ---------------------------------------------------------------------------

/// Memory budgets (PLAN §4). Eviction is never correctness-bearing — a miss
/// only re-computes.
/// What [`PlanCaches::placed_get`] found for a brush stroke.
pub(crate) enum PlacedLookup {
    /// Its placement (current, or the previous band's / trim's while it refines).
    Hit(Arc<crate::tess::placed::Placed>),
    /// Place it now.
    Place,
    /// Never placed and past the frame's placement budget (or placing in a host worker): draw nothing for it this
    /// frame.
    Defer,
    /// Never placed, and a host worker can place it: request it ([`PlanCaches::placement_request`]) and draw
    /// nothing for it until the placement arrives.
    Request,
    /// Placed at another band (the same geometry, brush and paint: its `base`), and a host worker can place it:
    /// draw this placement and request the current band's ([`PlanCaches::placement_request`]).
    Refine(Arc<crate::tess::placed::Placed>),
}

/// A worker's placement waiting its turn ([`PlanCaches::provide_placement`]): `None` for a request that will not
/// arrive (failed, superseded or dropped).
type StagedPlacement = Option<(String, u64, u64, i32, crate::tess::placed::Placed, Option<TessellatedGeometry>)>;

/// A node whose outline the current camera plan draws stale (another zoom
/// band, or a uniform placeholder off-screen), with its canvas bbox so later
/// frames refine what is on screen first.
#[derive(Debug, Clone)]
pub(crate) struct PendingOutline {
    pub id: String,
    pub bbox: [f64; 4],
    /// A brush stroke never placed yet, left out of a live frame past its placement budget
    /// ([`PlanCaches::placed_get`]): these fill in first, in paint order.
    pub unplaced: bool,
}

/// A text layer a live frame deferred, with the plan inputs its layout (and cache key) depends on.
#[derive(Debug, Clone)]
pub(crate) struct TextPrewarm {
    id: String,
    best_effort: bool,
    render_scale: f64,
}

/// Lay out text layers live frames deferred ([`PlanCaches::text_deferrable`]) into the text cache, between frames,
/// for up to `budget_ms` (at least one): the next frame's re-plan of those layers is then a cache hit. A frame lays
/// out only a refine budget's worth and pays its whole plan patch, prepared-frame rebuild and upload for it, so
/// thousands of first-visit text layers (a 20k-layer text document at fit) took 15–20 s of frames to converge; laid
/// out here, in idle time, the frames only draw. The z, clip and shader clock of a cached batch are patched on hit,
/// so a layout built here serves the frame exactly (the key covers everything else). Returns the layers left.
pub fn prewarm_deferred_text(resident: &ResidentScene, caches: &mut PlanCaches, budget_ms: f64) -> usize {
    let (Some(scene), Some(hashes)) = (resident.scene(), resident.hashes()) else {
        return caches.text_prewarm.len();
    };
    let clock = caches.outline_clock.unwrap_or(default_clock);
    let deadline = clock() + budget_ms.max(0.0);
    let progressive = std::mem::replace(&mut caches.text_progressive, false);
    let mut first = true;
    loop {
        if !first && clock() >= deadline {
            break;
        }
        first = false;
        let Some(job) = caches.text_prewarm.pop_front() else {
            break;
        };
        caches.text_prewarm_ids.remove(&job.id);
        let Some(node) = scene.canonical_or_transient(&job.id) else {
            continue;
        };
        if node.node_type() != "richtext" {
            continue;
        }
        // A layer a frame laid out meanwhile is a cache hit here (cheap).
        let _ = crate::text_draw::build_richtext_item(
            scene,
            node,
            0.0,
            [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY],
            job.best_effort,
            0.0,
            job.render_scale,
            Some((&mut *caches, hashes)),
        );
        caches.text_prewarmed += 1;
        caches.text_ready.insert(job.id);
    }
    caches.text_progressive = progressive;
    caches.text_prewarm.len()
}

/// Stale outlines and deferred text a live frame re-patches. Both refine within the frame's time budget and the
/// rest stay queued, so candidates past what a budget can lay out (a few hundred text layers at the idle budget)
/// are re-planned only to be queued again; a first visit's 5,700 queued text layers made every refining frame
/// re-plan all of them. (At 64 a first visit's text converged over 50 frames.)
const OUTLINE_REFINE_CANDIDATES: usize = 1024;
/// Candidates a frame re-plans per ms of its refine budget (6 ms while the canvas moves: 96; 28 ms when idle: 448).
const OUTLINE_REFINE_CANDIDATES_PER_MS: f64 = 16.0;

/// A live frame places brush strokes never placed before (a document opening) for this many times its refine
/// budget (the host's: 28 ms idle gives 140 ms, 6 ms while interacting 30 ms); the rest draw nothing that frame and
/// fill in over the next frames, back to front. A stroke placed once is never left out again.
const PLACEMENT_BUDGET_SCALE: f64 = 5.0;

/// The frames that fill in the strokes an earlier frame left out take this many times the refine budget (560 ms
/// idle): each fill-in frame also re-plans and re-uploads, so fewer, longer ones settle sooner.
const PLACEMENT_FILL_SCALE: f64 = 20.0;

/// Never-placed strokes a refine frame re-plans, per ms of its placement budget (a placement is ~0.15 ms).
const PLACEMENT_CANDIDATES_PER_MS: f64 = 10.0;
const TESS_BUDGET_BYTES: usize = 96 * 1024 * 1024;
/// The text cache holds a view's worth of laid-out text layers: a 20k-layer text document at fit shows ~11k of them
/// (~170 MB of batches). At 48 MB the layers laid out between frames (`prewarm_deferred_text`) were evicted before a
/// frame re-planned them, so the view never finished filling in.
const TEXT_BUDGET_BYTES: usize = 192 * 1024 * 1024;
const PAGE_PLAN_BUDGET_BYTES: usize = 16 * 1024 * 1024;
const PAGE_PLAN_BUDGET_ENTRIES: usize = 64;
/// The camera plan cache is a SINGLE slot holding the current viewport's combined
/// plan — which the engine builds transiently every frame regardless, so caching
/// it merely retains one copy rather than dropping it. Bounding it by the
/// 64-entry page-cache budget (16 MB) was far too tight: a dense viewport
/// (~2k visible nodes) combines a plan well over 16 MB, so `camera_put` refused
/// it and the plan was RE-COMBINED from scratch every frame — the camera-cache-
/// miss-every-frame throughput wall (a 2k-node pan ran ~5 fps because of this).
/// Give the single slot its own generous budget; one plan at this size is safe
/// versus the coarse wasm memory guard, and a still-larger plan keeps the old
/// behaviour (not retained → rebuilt each frame).
const CAMERA_PLAN_BUDGET_BYTES: usize = 128 * 1024 * 1024;
/// The retained camera plan (`camera_slot`) holds the viewport's plan too, and is
/// what keeps a large document's edits and blurred pans off the full rebuild. Its
/// byte count (`plan_bytes`) includes the bodies' geometry and text batches,
/// which it shares with the tessellation and text caches (the same `Arc`s), so it
/// overstates what the slot adds; a Design document of 50,000 nodes at fit counts
/// ~150 MB. A larger plan is not retained (every change rebuilds, as before).
const CAMERA_SLOT_BUDGET_BYTES: usize = 512 * 1024 * 1024;
/// What the culled camera builds keep of each node's draw items for later windows
/// (`PlanCaches::node_items`), by [`owned_items_bytes`]. It had no bound: a view of
/// a whole 100k-node document at fit kept a deep copy of every node's text batches
/// and instances beside the plan, 1.4 GB of wasm heap — past the memory guard's
/// alarm on its own. A pan or zoom-out copies what an earlier window built; at this
/// budget that is every node of several dense viewports, and the rest are rebuilt.
const NODE_ITEMS_BUDGET_BYTES: usize = 64 * 1024 * 1024;

pub struct TessEntry {
    pub key: u64,
    /// `Arc`-shared with the plan's `DrawItem::Body.geometry` — a cache hit is a
    /// pointer-bump, and the plan never deep-copies the vertex stream.
    pub geometry: Arc<TessellatedGeometry>,
    tick: u64,
    /// Brush plan P8: for an outlined stroke, its key WITHOUT the zoom band
    /// (geometry + brush). A live camera frame may draw this entry while the
    /// outline for the current band is still being refined.
    outline_base: Option<u64>,
    /// Brush plan P8: an outlined stroke's second band (same base) — e.g. the
    /// camera's while a page capture / thumbnail renders at its own scale —
    /// so the two don't evict each other every frame.
    outline_alt: Option<(u64, Arc<TessellatedGeometry>)>,
    /// Zoom bands of `geometry` and `outline_alt` (`None` = placeholder).
    outline_bands: (Option<i32>, Option<i32>),
    /// Live frames this entry was drawn as an EDITED stroke's pre-edit outline
    /// ([`PlanCaches::outline_prev_get`]); at [`OUTLINE_MAX_PREV_FRAMES`] the
    /// stroke is computed over budget, so a continuous multi-select gesture
    /// never starves the strokes late in document order.
    prev_frames: u32,
}

/// Live frames an edited outlined stroke may draw its pre-edit outline before
/// it is computed over budget (see [`TessEntry::prev_frames`]).
pub(crate) const OUTLINE_MAX_PREV_FRAMES: u32 = 2;

/// Brush plan P8 follow-up: an art / scatter brush's placed artwork for one
/// stroke node — the synthetic path nodes the plan draws in its slot.
pub struct PlacedEntry {
    pub key: u64,
    /// Key without the zoom band (geometry + brush), for stale reuse.
    base: u64,
    pub placed: Arc<crate::tess::placed::Placed>,
    bytes: usize,
    tick: u64,
    /// Live frames this entry has been served stale (over budget) since it
    /// was placed; at [`PLACED_MAX_STALE_FRAMES`] it is re-placed regardless
    /// of the budget, so a small art stroke never lags a whole playback
    /// behind bigger ones that keep taking the budget first.
    stale_frames: u32,
}

/// Live frames an art placement may be served stale before it is re-placed
/// over budget (see [`PlacedEntry::stale_frames`]).
pub(crate) const PLACED_MAX_STALE_FRAMES: u32 = 2;

pub struct LineTessEntry {
    pub key: u64,
    pub geometries: Vec<TessellatedGeometry>,
    tick: u64,
}

pub struct TextEntry {
    pub key: u64,
    /// The cached batch with STALE z/model/clip/mask — patched on hit.
    pub batch: Box<crate::seams::TextQuadBatch>,
    /// Connector labels only: the carved body text-gap (start_t, end_t)
    /// measured with the batch (font metrics ride the key's generation).
    pub gap: Option<(f64, f64)>,
    tick: u64,
}

/// An export job's text batch, shared by every layer with its content key (`export_job`).
pub struct JobTextEntry {
    batch: Box<crate::seams::TextQuadBatch>,
    /// What the batch's build recorded, replayed for each layer it serves.
    events: Arc<[moda_assets::report::Captured]>,
    tick: u64,
}

pub struct PagePlanEntry {
    pub fingerprint: u64,
    /// Whether any item in the plan bakes the shader clock (procedural fills,
    /// shader-filled text) — such plans only hit when the clock matches.
    pub uses_shader_time: bool,
    pub shader_time_s: f64,
    pub plan: Arc<CoveragePlan>,
    pub warnings: Arc<PlanWarnings>,
    bytes: usize,
    tick: u64,
}

/// Residency: the combined multi-page camera plan cache. There is
/// exactly ONE live camera frame per engine, so this is a single slot (no LRU)
/// keyed by the visible-page set × per-page content fingerprints. Camera params
/// (cameraX/Y, zoom, viewport, dpr→size, clear) are DELIBERATELY excluded from
/// the key — the draw items are canvas-absolute, so a pan/zoom reuses this entry
/// and the host re-presents it via `CameraPresentOverride` (no rebuild).
pub struct CameraPlanEntry {
    pub fingerprint: u64,
    pub uses_shader_time: bool,
    pub shader_time_s: f64,
    /// Every shader-clock consumer in this plan is a procedural-fill
    /// body with a direct-copy clock the present path re-threads (no MSDF/vector
    /// text shader fill, no non-linear clock). When true the clock is NOT in the
    /// hit key — an animating shader reuses this entry and the live clock rides
    /// the present override. When false the entry keeps the exact-clock-match
    /// behavior (rebuild per clock) so a non-re-threadable shader never freezes.
    pub shader_time_rethreadable: bool,
    pub plan: Arc<CoveragePlan>,
    pub warnings: Arc<PlanWarnings>,
}

/// Monotonic ms for the refine budget when the host set no clock (native).
/// On wasm32 without a host clock there is no time source: report 0, so a
/// budget never expires (outlines are computed, never drawn stale).
fn default_clock() -> f64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        static START: OnceLock<std::time::Instant> = OnceLock::new();
        START
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_secs_f64()
            * 1e3
    }
    #[cfg(target_arch = "wasm32")]
    {
        0.0
    }
}

/// The live camera lane's history in [`PlanCaches`] (see [`PlanCaches::enter_side_camera`]).
#[derive(Debug, Clone, Copy)]
pub struct CameraHistory {
    last_camera_zoom: Option<f64>,
    last_zoom_change_ms: Option<f64>,
    last_plan_scale: Option<f64>,
}

/// All Stage-1 plan caches. Owned by the wasm host (one per engine instance);
/// the default export path never builds one.
#[derive(Default)]
pub struct PlanCaches {
    tick: u64,
    /// Brush plan P8: (resident generation, scene has outlined strokes).
    outline_memo: Option<(u64, (bool, bool))>,
    /// (resident generation, visible page indices, whether those pages make the combined camera plan
    /// camera-dependent). Each predicate walks a page's whole z-order, three walks per frame, which on a
    /// 10,000-path page was most of a pan frame's plan time; the answer depends on the scene alone.
    camera_dependent_memo: Option<(u64, Vec<usize>, bool)>,
    /// (resident generation, visible page indices, whether those pages hold layer captures — layer
    /// filters or capture shadows): memoised like `camera_dependent_memo`.
    capture_memo: Option<(u64, Vec<usize>, bool)>,
    /// The capture projection (zoom bucket + window) the cached camera plan's layer captures were
    /// planned under (`frame::CaptureView`); a camera it still serves reuses the plan.
    pub(crate) capture_view: Option<crate::frame::CaptureView>,
    /// The camera scale bucket (`text_draw::degraded_camera_bucket`) the cached camera plan's
    /// Degraded text was sized for, when any of it was drawn coarser than full quality or skipped
    /// as too small; a camera zoomed in past it rebuilds. `None` = the plan is zoom-invariant.
    degraded_zoom_bucket: Option<f64>,
    /// Brush plan P8: the time (host clock, ms) until which a live camera
    /// build refines outlines after a zoom-band change; past it, a node whose
    /// geometry is unchanged draws its previous band's outline and is queued.
    /// `None` (exports, captures, video) = compute every outline exactly. A
    /// node whose geometry changed is always computed. Set per frame
    /// ([`PlanCaches::set_outline_refine_budget_ms`]).
    outline_refine_deadline: Option<f64>,
    /// The per-frame refine budget the deadline is started from.
    outline_refine_budget_ms: Option<f64>,
    /// A change of zoom band alone rebuilds the camera plan instead of keeping the retained slot (off by default;
    /// for A/Bs), and how many times the slot was kept across one.
    pub band_rekey_disabled: bool,
    pub band_rekeys: u64,
    /// The deadline for placing never-placed brush strokes this frame (`PLACEMENT_BUDGET_SCALE` times the refine
    /// budget, started at the frame's first placement); `None` off the live lane (exports place everything).
    placement_deadline: Option<f64>,
    /// This frame fills in strokes an earlier one left out (`PLACEMENT_FILL_SCALE`).
    placement_filling: bool,
    /// Strokes placed at least once (a stroke that has drawn is never left out of a frame).
    placed_ever: std::collections::HashSet<String>,
    /// Never-placed strokes a host worker is placing: (key, request sequence number). An entry stays until its
    /// placement is promoted into `placed`.
    placement_requested: HashMap<String, (u64, u64)>,
    /// The next placement request's sequence number, and the next one to promote: placements arrive from several
    /// workers out of order and are promoted in request (paint) order, so a painting still fills in back to front.
    placement_seq: u64,
    placement_promoted: u64,
    placement_staged: std::collections::BTreeMap<u64, StagedPlacement>,
    /// Brush definitions as request JSON, by content hash (`SceneDocument::brush_hash`).
    placement_brush_json: HashMap<u64, String>,
    /// Never-placed strokes the last camera build left out (diagnostics; the frame is `outline_refining`).
    pub placement_deferred: usize,
    /// The zoom of the last live camera frame (refinement holds while it changes).
    last_camera_zoom: Option<f64>,
    /// Refine while the zoom changes too (default off: refinement holds; `set_hold_refine_while_zooming`).
    refine_while_zooming: bool,
    /// The host clock the deadline is measured on.
    outline_clock: Option<fn() -> f64>,
    /// Set while a live camera frame builds its plan from scratch: text layout past the refine budget is
    /// deferred to the next frame ([`Self::text_deferrable`]).
    text_progressive: bool,
    /// Text layers left for the next frame by a full rebuild past its refine budget (diagnostics).
    pub text_deferred: u64,
    /// The deferred text layers and the inputs their layout needs, for [`prewarm_deferred_text`] to lay out
    /// between frames (in the order they were deferred; `text_prewarm_ids` dedupes).
    text_prewarm: std::collections::VecDeque<TextPrewarm>,
    text_prewarm_ids: std::collections::HashSet<String>,
    /// Deferred text layers laid out between frames, not re-planned yet (the next frame takes them all).
    text_ready: std::collections::HashSet<String>,
    /// The text cache's budget in bytes; 0 = [`TEXT_BUDGET_BYTES`] (tests set a small one).
    pub(crate) text_budget_bytes: usize,
    /// Text layers laid out between frames (diagnostics).
    pub text_prewarmed: u64,
    /// Placed art / scatter brush artwork per stroke node.
    placed: HashMap<String, PlacedEntry>,
    placed_bytes: usize,
    /// Outlines drawn stale in the last camera build.
    outline_stale_served: usize,
    /// Brush plan P8: outlines are computed off the main thread (the host runs
    /// the WASM `outline_tessellation` export in workers): past the refine
    /// budget a live frame requests the outline instead of computing it, and
    /// draws the previous band's outline or a pending body meanwhile.
    outline_worker: bool,
    /// Worker results accepted (diagnostics: proves the worker path runs).
    outline_worker_provided: u64,
    /// Nodes a worker could not outline (a non-trap error): computed on the
    /// main thread from now on.
    outline_no_worker: std::collections::HashSet<String>,
    /// Requests the host has not taken yet (JSON objects).
    outline_requests: Vec<String>,
    /// node id → outline key requested and not yet provided (dedupe).
    outline_requested: HashMap<String, u64>,
    /// The viewport of the last camera build (for `outline_refining` read
    /// after the build).
    outline_view: Option<[f64; 4]>,
    /// The zoom band of the last camera build with outlines (diagnostics).
    outline_band: Option<i32>,
    /// Notes (`skipped` warnings) of cached outlines, by node: (key, note).
    outline_notes: HashMap<String, (u64, String)>,
    /// Nodes whose outline the current camera plan draws stale; non-empty = the
    /// host should render another frame ([`PlanCaches::outline_refining`]). The
    /// next live frame patches up to the refine budget of them through the
    /// incremental camera slot instead of rebuilding the whole plan.
    outline_pending: Vec<PendingOutline>,
    pub(crate) tess: HashMap<String, TessEntry>,
    tess_bytes: usize,
    /// The frame (tick) whose eviction pass found only the working set left over budget: no further pass that
    /// frame (each one walks and sorts every entry).
    tess_evict_blocked: Option<u64>,
    /// The same for the placed-brush and line caches.
    placed_evict_blocked: Option<u64>,
    line_evict_blocked: Option<u64>,
    pub(crate) line_tess: HashMap<String, LineTessEntry>,
    line_tess_bytes: usize,
    pub(crate) text: HashMap<String, TextEntry>,
    text_bytes: usize,
    /// An export job's text batches by content key alone (`export_job`); `Some` turns the job's sharing on.
    job_text: Option<HashMap<u64, JobTextEntry>>,
    job_text_bytes: usize,
    pages: HashMap<String, PagePlanEntry>,
    page_bytes: usize,
    /// The single combined camera-frame plan (PLAN §3 #21 / R13).
    camera: Option<CameraPlanEntry>,
    /// Marginal heap of the cached camera plan (single slot, surfaced in
    /// `stats_json` so the residency telemetry / memory watermark accounts for
    /// what can be the largest single plan — all visible pages combined).
    camera_bytes: usize,
    // Stats (battery part 2 reads these through the wasm stats surface).
    pub tess_hits: u64,
    pub tess_misses: u64,
    pub text_hits: u64,
    pub text_misses: u64,
    pub page_hits: u64,
    pub page_misses: u64,
    pub camera_hits: u64,
    pub camera_misses: u64,
    pub evictions: u64,
    /// M1 fingerprint memo: when `false` (the DEFAULT — so the memo is ON), an
    /// unchanged `camera_frame_key` lets `build_camera_plan_retained` skip the
    /// O(n) `camera_fingerprint` recompute. The host sets this `true` only for the
    /// `?wasmFingerprintMemo=0` kill switch. Config — preserved across `clear()`.
    fingerprint_memo_disabled: bool,
    /// M1: the last camera frame's cheap generation key paired with the
    /// `camera_fingerprint` it produced. Unchanged key ⇒ reuse the stored
    /// fingerprint. `None` until first use; reset when the camera slot drops.
    camera_frame_key: Option<(u64, u64)>,
    /// M1: camera frames whose `camera_fingerprint` recompute was skipped via the
    /// generation-key memo (the M1 signal — rises on a static pan).
    pub camera_fp_skips: u64,
    /// rust-render-perf-g8 (E3): the page-path twin of `camera_frame_key`. Maps a
    /// page id to its last cheap generation key paired with the `page_fingerprint`
    /// it produced. An unchanged key (resident generation + page index + plan opts
    /// + asset/text generations — a conservative SUPERSET of the fingerprint's
    /// inputs) lets `build_plan_retained` reuse the stored fingerprint and skip the
    /// O(n) per-frame display-list rebuild on static single-page playback. Governed
    /// by the SAME `fingerprint_memo_disabled` kill switch as the camera memo.
    page_frame_key: HashMap<String, (u64, u64)>,
    /// rust-render-perf-g8 (E3): page frames whose `page_fingerprint` recompute was
    /// skipped via the generation-key memo (rises on static page playback).
    pub page_fp_skips: u64,
    /// rust-edit-incremental-z4: the live single-page editor slot — the last full
    /// page plan + its node index, kept in lockstep with the resident generation so
    /// a structure-preserving property edit patches O(dirty) DrawItems in place
    /// (via `Arc::make_mut`) instead of rebuilding the whole O(n) plan. `None`
    /// until the first full build; dropped on any structural change. `take`/restore
    /// during a patch sidesteps the `&mut self` borrow that `rebuild_node_items`
    /// needs for the R3/R4 caches.
    incremental_page: Option<IncrementalEntry>,
    /// Scale RFC stage C: the retained camera plan (`camera_slot`) — the editor
    /// viewport's plan, patched in place by edits and camera moves instead of
    /// rebuilt: property edits and structural ones (add / delete / reorder),
    /// multi-page and canvas-root views, split and capture pages alike.
    camera_slot: Option<Box<crate::camera_slot::CameraSlot>>,
    /// The scene-side index of a retained camera plan dropped for a camera move
    /// past its cull window, for the next seed at the same scene generation.
    camera_scene_index: Option<Box<crate::camera_slot::SceneIndex>>,
    /// Frames the retained camera slot served since its plan was built: a split
    /// page's plan rebuilt after one frame or less (a fast pan) culls to the view
    /// itself next time instead of a window around it.
    camera_window_frames: u32,
    /// The zoom a split page's plan was last built at (its window choice holds
    /// for a pan at that zoom).
    camera_window_zoom: Option<f64>,
    /// rust-edit-incremental-z4: edits served entirely from the index (no rebuild):
    /// static re-renders (`incr_static_hits`) and structure-preserving patches
    /// (`incr_patch_hits`, touching `incr_patch_nodes` nodes total). `incr_fallbacks`
    /// counts edits that fell back to a full rebuild (structural / ineligible /
    /// shape-verify miss). Surfaced in `stats_json` for telemetry.
    pub incr_static_hits: u64,
    pub incr_patch_hits: u64,
    pub incr_patch_nodes: u64,
    /// Nodes whose certified placement changed without rebuilding their draw items.
    pub incr_placement_nodes: u64,
    pub incr_fallbacks: u64,
    /// Scale RFC stage C: why the retained camera slot last fell back to a full
    /// rebuild (diagnostics; `camera_bench` reports it per lane).
    pub incr_last_fallback: Option<&'static str>,
    /// Camera frames planned from scratch (no cached plan, no retained slot to patch): a camera-dependent page
    /// rebuilds on every frame its slot can't serve.
    pub camera_rebuilds: u64,
    /// rust-edit-incremental-z4: marginal heap retained by the incremental page /
    /// camera slots (the live plan each holds). Tracked + surfaced in `stats_json`
    /// so the slots are VISIBLE to the wasm memory watermark — and seeding is gated
    /// on the plan being within the same byte budget the LRU caches enforce, so a
    /// slot never retains an over-budget plan the LRU itself refused.
    pub incr_page_bytes: usize,
    pub incr_camera_bytes: usize,
    /// The shader clock (`f64` bits) of the last camera build, and whether this build's is the same: a plan whose
    /// clock the present can't re-thread is retained only while the clock holds still (the editor's, outside
    /// playback), keyed on it (`CameraSlot::clock`).
    last_camera_clock: Option<u64>,
    camera_clock_steady: bool,
    /// rust-edit-incremental-z4: `true` disables the incremental patcher (the
    /// `?wasmIncrementalPlan=0` kill switch), forcing the legacy full-rebuild path.
    /// Config — preserved across `clear()`.
    incremental_disabled: bool,
    /// Viewport culling (`crate::cull`) of the camera plan: `true` turns it off
    /// (kill switch / A-B measurement). Config — preserved across `clear()`.
    cull_disabled: bool,
    /// Scenes with fewer nodes than this draw everything (culling buys nothing
    /// there and every window change would rebuild the plan). `None` =
    /// [`CULL_MIN_NODES`]. Config — preserved across `clear()`.
    cull_min_nodes: Option<usize>,
    /// The cull window (canvas units) the cached camera plan and the camera
    /// incremental slot were built with; `None` when they are unculled. A camera
    /// whose view the window still contains reuses them.
    pub(crate) camera_cull: Option<crate::cull::Aabb>,
    /// The cull window the last camera plan build used (`None`: unculled). The motion cache checks its region
    /// against it.
    last_plan_window: Option<crate::cull::Aabb>,
    /// Camera plan builds that culled, and what the last one considered / left out.
    pub cull_builds: u64,
    /// The last camera frame was drawn mid-zoom with curves at a band behind the camera's: not exact, so the
    /// host renders again.
    pub motion_plan: bool,
    /// Mid-zoom frames keep the plan's flattening band (`motion_plan`): the live editor's camera lane turns it
    /// on; off, every band crossing rebuilds the camera plan.
    pub motion_plans: bool,
    /// The camera scale (px per canvas unit) the last camera build flattened curves at.
    last_plan_scale: Option<f64>,
    /// When (refine clock, ms) the camera's zoom last changed.
    last_zoom_change_ms: Option<f64>,
    /// The last live camera frame was planned while the zoom changes (within `MOTION_ZOOM_HOLD_MS` of a change).
    zooming: bool,
    pub cull_last_considered: usize,
    pub cull_last_culled: usize,
    /// Display lists by page id (and the canvas root), for the resident generation
    /// `dl_cache_gen`. Consulted only while `dl_live_gen` is set — by the camera lane,
    /// for the build in progress — so exports and other lanes never see it.
    dl_cache: HashMap<String, Arc<moda_scene::DisplayList>>,
    dl_cache_gen: Option<u64>,
    dl_live_gen: Option<u64>,
    /// The canvas root's display list its caller built for the plan in progress
    /// ([`crate::frame::build_camera_box_plan_cached_with_dl`]), with the address of the scene it was built from.
    /// Set and cleared within that one call.
    frame_canvas_dl: Option<(usize, Arc<moda_scene::DisplayList>)>,
    /// Scene predicates for the same generation (`memo_flag`).
    flag_memo: HashMap<(String, u8), bool>,
    /// Viewport-cull subtree indexes for the same generation, by display-list key
    /// and the render options that decide which items consume z.
    cull_index: HashMap<(String, u64), Arc<crate::cull::CullIndex>>,
    /// Placed brush strokes' drawn canvas boxes, by node: the key (`frame::brush_bounds_key`: geometry, brush,
    /// transform), the box and the zoom band it was drawn at. The cull reads them (`crate::cull`); they outlive
    /// generations, since the key says when one no longer applies.
    brush_bounds: HashMap<String, (u64, crate::cull::Aabb, i32)>,
    /// Placed brush strokes a culled camera build has reached: eviction treats their placements and geometry as
    /// in use. The retained plan draws them across frames without touching their entries, and culling one must not
    /// make a pan re-place every stroke it brings back.
    brush_kept: std::collections::HashSet<String>,
    /// The plan index's reverse-dependency scan, for one resident generation.
    reverse_deps: Option<(u64, ReverseDeps)>,
    /// Culled camera plans: each node's built items, for the resident generation,
    /// plan options and resources in `node_items_key`. A plan rebuilt for a new cull
    /// window copies every node an earlier window already built (a pan back, a zoom
    /// out) and builds only the nodes it has never drawn.
    node_items: HashMap<String, CachedNodeItems>,
    /// [`owned_items_bytes`] of `node_items`, bounded by [`NODE_ITEMS_BUDGET_BYTES`].
    node_items_bytes: usize,
    node_items_key: Option<(u64, u64, u64)>,
    /// Set by the camera lane while a culled, camera-independent plan builds.
    node_items_live: bool,
    /// Nodes whose items a culled build copied from `node_items`.
    pub cull_copied_nodes: u64,
    /// Cached while moving (`crate::motion_capture`, [`PlanCaches::set_motion_captures`]).
    motion: MotionState,
}

/// The live camera lane's motion captures (`crate::motion_capture`): moving frames plan their units as captures, in
/// a scene state of their own, so a still frame plans with exactly the state still frames had.
#[derive(Default)]
struct MotionState {
    /// The host asks for motion captures on the live camera frames (the camera moves and the setting is on).
    requested: bool,
    /// The caches hold the moving frames' scene state; `other` holds the still frames'.
    active: bool,
    /// The scene state of the mode not in use: the still frames' while moving, the last motion's while still (the
    /// next gesture starts from it: its captures are keyed alike).
    other: Option<ScenePlanState>,
    /// The resident generation's units.
    units: Option<(u64, Arc<crate::motion_capture::MotionUnits>)>,
    /// The units the moving frames' scene state was planned with (another set starts it afresh).
    state_digest: Option<u64>,
    /// The shader clock the motion froze at its first frame: a unit's shaders hold still while the camera moves.
    clock: Option<f64>,
    /// The capture scale the moving frames hold (`hold_scale`).
    scale: Option<f64>,
    /// Gestures and frames planned with motion captures (diagnostics).
    gestures: u64,
    frames: u64,
}

/// How far in (as a factor of scale) a moving frame keeps compositing captures made at the scale it holds before it
/// captures again at the camera's: an octave, so a zoom in draws them magnified up to 2× for a moment.
const MOTION_HOLD_ZOOM_IN: f64 = 2.0;
/// How far out: one capture zoom step (a larger view at a held scale would capture more pixels than the screen shows).
const MOTION_HOLD_ZOOM_OUT: f64 = 1.1892071150027210; // 2^(1/4)

impl MotionState {
    /// The capture scale a moving frame plans at, for the camera's bucket `bucket`.
    fn hold_scale(&mut self, bucket: f64) -> f64 {
        match self.scale {
            Some(s) if bucket <= s * MOTION_HOLD_ZOOM_IN * (1.0 + 1e-9) && bucket * MOTION_HOLD_ZOOM_OUT * (1.0 + 1e-9) >= s => s,
            _ => {
                self.scale = Some(bucket);
                bucket
            }
        }
    }
}

/// One node's draw items as a culled camera build made them (`PlanCaches::node_items`).
pub(crate) struct CachedNodeItems {
    /// [`owned_items_bytes`] of `items`.
    bytes: usize,
    /// The render-z the node was built at and the z it consumed.
    pub z: u64,
    pub z_consumed: u64,
    pub items: Vec<crate::seams::DrawItem>,
    /// The warnings its build raised, raised again when it is copied.
    pub skipped: Vec<String>,
}

/// Below this many scene nodes (canonical + transient) the camera plan is not
/// culled.
pub const CULL_MIN_NODES: usize = 256;

/// The cull window built around a view: the view grown by half its size on every
/// side, so a pan of up to half a viewport reuses the plan.
fn cull_window_for(view: crate::cull::Aabb) -> crate::cull::Aabb {
    crate::tess::outline::near_view(view)
}

/// How long after the zoom last changed a frame still counts as mid-zoom (`motion_plan`), ms.
const MOTION_ZOOM_HOLD_MS: f64 = 200.0;
/// How many zoom bands a mid-zoom frame may draw curves behind the camera's.
const MOTION_BAND_LAG: i32 = 2;

/// Whether a plan culled to `window` can present `view`: the window contains the
/// view and is not so much larger (after a zoom in) that it draws mostly off-screen
/// content.
fn cull_window_serves(window: &crate::cull::Aabb, view: &crate::cull::Aabb) -> bool {
    let area = |a: &crate::cull::Aabb| (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    crate::cull::contains(window, view) && area(window) <= 16.0 * area(view)
}

/// Fold a capture view (zoom bucket + window) into a camera plan fingerprint.
fn captured_fingerprint(fingerprint: u64, view: Option<&crate::frame::CaptureView>) -> u64 {
    match view {
        None => fingerprint,
        Some(v) => {
            let mut h = FxHasher::default();
            h.write_u64(fingerprint);
            h.write(b"capture");
            h.write_u64(v.scale.to_bits());
            for x in v.window {
                h.write_u64(x.to_bits());
            }
            h.finish()
        }
    }
}

/// Fold a cull window into a camera plan fingerprint.
fn culled_fingerprint(fingerprint: u64, window: Option<&crate::cull::Aabb>) -> u64 {
    match window {
        None => fingerprint,
        Some(w) => {
            let mut h = FxHasher::default();
            h.write_u64(fingerprint);
            h.write(b"cull");
            for v in w {
                h.write_u64(v.to_bits());
            }
            h.finish()
        }
    }
}

/// rust-edit-incremental-z4: the live editor slot — a built page plan plus the
/// index that lets the patcher rebuild individual nodes in place.
struct IncrementalEntry {
    page_id: String,
    plan: Arc<CoveragePlan>,
    warnings: Arc<PlanWarnings>,
    index: PlanIndex,
}

/// rust-edit-incremental-z4: everything the incremental patcher needs to rebuild a
/// single node's items and splice them into the cached plan without rebuilding the
/// display list or re-deriving page geometry.
struct PlanIndex {
    /// Per item-producing primary node, in plan-item order.
    spans: Vec<crate::frame::NodeSpan>,
    /// node_id → index into `spans`.
    by_node: HashMap<String, usize>,
    page_index: usize,
    page_clip: [f32; 4],
    transparent: bool,
    /// Plan-relevant `RenderOptions` fingerprint the plan was built with — an opts
    /// change (scale/sample/text-engine/…) forces a full rebuild.
    opts_key: u64,
    /// rust-edit-incremental-z4: the external-resource generations the plan baked in
    /// (`moda_assets::engine_seam::store_generation()` ⊕ `text_draw::text_host_generation()`).
    /// These bump when a pattern/image asset or a font SETTLES asynchronously —
    /// WITHOUT a scene edit — which `page_fingerprint` folds in but the cheap
    /// generation-based dirty tracking does not. If it changes, the fast path
    /// falls back so a settled asset/font re-renders instead of serving a stale
    /// placeholder/un-shaped batch.
    resource_key: u64,
    /// Resident generation the plan reflects. `dirty_since(built_at_generation)`
    /// drives the fast path.
    built_at_generation: u64,
    /// Editing one of these changes OTHER nodes the host may not have marked dirty,
    /// so a dirty node in any of them forces a full rebuild:
    ///   - `blend_related_ids`: blend nodes + their objects and spine
    ///     (a blend renders its sources' geometry at plan time);
    ///   - `mask_source_ids`: `clipMaskNodeId` targets (a group mask's coverage is
    ///     this node's geometry — editing it re-shapes the mask for masked siblings).
    /// (Connector/line anchors are NOT here: the engine reads pre-baked line points
    /// node-locally — `LineSpec::from_json` — so a shape move can't silently change a
    /// connector's plan items.)
    blend_related_ids: Arc<std::collections::HashSet<String>>,
    mask_source_ids: Arc<std::collections::HashSet<String>>,
    /// Viewport culling: roots of the display-list ranges the build skipped. A
    /// culled node has no span, which the patcher would otherwise read as "draws
    /// nothing" — an edit that touches one (or one of their descendants) falls back.
    culled_roots: std::collections::HashSet<String>,
}

/// `(blend_related_ids, mask_source_ids)` of a scene (`scan_reverse_deps`).
type ReverseDeps = (
    Arc<std::collections::HashSet<String>>,
    Arc<std::collections::HashSet<String>>,
);

/// Whether `id` or one of its ancestors was culled from the indexed plan.
fn under_culled_root(scene: &Scene, index: &PlanIndex, id: &str) -> bool {
    if index.culled_roots.is_empty() {
        return false;
    }
    let mut cur = id;
    for _ in 0..64 {
        if index.culled_roots.contains(cur) {
            return true;
        }
        match scene
            .parent_map()
            .get(cur)
            .or_else(|| scene.transient_parent_of().get(cur))
        {
            Some(p) => cur = p.as_str(),
            None => return false,
        }
    }
    true // runaway chain: be conservative
}

/// A tess entry's bytes, including an outline's second-band slot.
fn tess_entry_bytes(e: &TessEntry) -> usize {
    geometry_bytes(&e.geometry) + e.outline_alt.as_ref().map_or(0, |(_, g)| geometry_bytes(g))
}

fn geometry_bytes(g: &TessellatedGeometry) -> usize {
    g.positions.len() * 4 + g.vertex_types.len() * 4 + 64
}

fn batch_bytes(b: &crate::seams::TextQuadBatch) -> usize {
    // Marginal heap of a cached text batch, by lane. The MSDF lane carries
    // `quads` + the `ops` draw stream; the vector lane (`b.vector`) carries no
    // quads/ops and instead owns a per-node quadratic-curve point buffer plus
    // per-glyph instance records — count those ACTUAL `Vec` payloads.
    //
    // This replaces a flat 64 KiB/vector-node estimate that charged a 3-glyph
    // label the same as a full paragraph. A realistic node is ~5-10 KiB, so the
    // flat estimate saturated the 48 MB text LRU at ~750 nodes and thrashed
    // dense decks: a 1k-richtext deck overflowed the budget, evicting live
    // batches it immediately re-prepared (a periodic full text re-shape, the
    // intermittent-drag-jank spike). Curve points are still duplicated per node
    // — sharing them across nodes (the real RSS win) is future work.
    let mut bytes = 256;
    bytes += b.ops.len() * 512;
    // A shader text fill carries a fixed packed instance
    // (`PackedFill`'s two `[f32; 64]` arrays + colors/bounds + a module id) —
    // ~0.5 KiB, not trivial on a shader-text deck, so count it explicitly.
    // vork: moda-text-draw's TextQuadBatch carries no shader fill (text shader fills are not ported).
    if let Some(v) = &b.vector {
        bytes += vector_batch_bytes(v);
    }
    // A Degraded-tier batch owns its RGBA raster (the node's box at its raster
    // scale). Uncounted, a text cache of them never reached its budget: a 100k-node
    // document planned before its fonts arrived kept 16k rasters, 1.3 GB of wasm
    // heap, against a counted 4 MB.
    if let Some(d) = &b.degraded {
        bytes += d.raster.rgba.len() + std::mem::size_of::<moda_text::degraded::DegradedRaster>();
    }
    bytes
}

/// Marginal heap of a vector text node's draw ops: the per-node curve buffer
/// (6 f32 per quadratic segment), the `VectorOp` structs themselves, and each
/// op's per-glyph instance records + local-space triangle list. Counted as
/// `len * size_of::<T>()` (consistent with `geometry_bytes` and the `quads`
/// term above) so it scales with content rather than being a flat constant.
///
/// Contract: this is *logical* accounting like the rest of the file — it counts
/// the dominant variable-length payloads and excludes `Vec` spare capacity and
/// the small fixed side payloads (`gradient`, `node_id`/key strings). The batch's
/// `fill_shader` is counted by the caller (`batch_bytes`) since it lives on
/// `TextQuadBatch`. Excluded items are bounded per node and the budgets are
/// deliberately conservative, so any drift is small and non-correctness-bearing
/// (eviction only re-computes). If a future `VectorNodeOps` field carries
/// significant heap, add it here.
fn vector_batch_bytes(v: &moda_text::vector::VectorNodeOps) -> usize {
    let mut bytes = v.curve_points.len() * std::mem::size_of::<f32>();
    bytes += v.ops.len() * std::mem::size_of::<moda_text::vector::VectorOp>();
    for op in &v.ops {
        bytes +=
            op.records.len() * std::mem::size_of::<moda_text::vector::ops::VectorAnalyticRecord>();
        bytes += op.tris.len() * std::mem::size_of::<f32>();
    }
    for e in &v.emojis {
        bytes += std::mem::size_of::<moda_text::vector::VectorEmoji>() + e.cluster.len();
    }
    bytes
}

/// Marginal heap of a cached page plan (PLAN §4: page plans ≤16 MB). Text batches
/// are owned by the plan's DrawItems; body geometry is now `Arc`-SHARED with the
/// tess cache (renderer-perf-more-z6), so a body's `geometry_bytes` is counted here
/// AND in `tess_bytes` — i.e. shared geometry is double-counted across budgets, and
/// a tess-LRU eviction frees 0 RSS while a retained plan still holds the `Arc`. This
/// makes the byte budgets (`PAGE_PLAN_BUDGET_BYTES` etc.) deliberately CONSERVATIVE
/// (over-count → evict/drop-slot sooner than real RSS demands) — accepted as the
/// safe direction; deduping by `Arc::strong_count` is possible future work but risks
/// under-counting. Instances/masks counted coarsely.
fn plan_bytes(plan: &CoveragePlan) -> usize {
    1024 + plan.masks.len() * 4096 + items_bytes(&plan.items)
}

/// Heap a copy of `items` owns: [`items_bytes`] without body geometry, which is an
/// `Arc` shared with the tess cache and the plan (a copy is a pointer bump).
fn owned_items_bytes(items: &[crate::seams::DrawItem]) -> usize {
    use crate::seams::DrawItem;
    items
        .iter()
        .map(|item| match item {
            DrawItem::Body { .. } => std::mem::size_of::<crate::seams::PaintInstance>() + 128,
            DrawItem::LayerBlurComposite { group, .. } => owned_items_bytes(&group.items) + 256,
            other => items_bytes(std::slice::from_ref(other)),
        })
        .sum::<usize>()
        + items.len() * std::mem::size_of::<crate::seams::DrawItem>()
}

/// rust-edit-incremental-z4: marginal heap of a slice of draw items — the per-item
/// part of [`plan_bytes`], factored out so the incremental patcher can DELTA-update
/// the slot's retained byte count (old span vs new span) in O(patched-items) rather
/// than re-walking the whole plan each edit (keeps the page lane O(dirty)).
pub(crate) fn items_bytes(items: &[crate::seams::DrawItem]) -> usize {
    use crate::seams::DrawItem;
    let mut bytes = 0;
    for item in items {
        bytes += match item {
            DrawItem::Body { geometry, .. } => {
                geometry_bytes(geometry) + std::mem::size_of::<crate::seams::PaintInstance>() + 128
            }
            DrawItem::Text(b) => batch_bytes(b),
            // SDF bodies carry only a boxed instance (no geometry).
            DrawItem::SolidShapeSdf { .. } => {
                std::mem::size_of::<crate::paint::shape_sdf::ShapeSolidSdfInstance>() + 128
            }
            DrawItem::GradientShapeSdf { .. } => {
                std::mem::size_of::<crate::paint::shape_sdf::ShapeGradientSdfInstance>() + 128
            }
            // The diverted items' geometry/text live inside the
            // group — recurse so the page/camera byte budgets see them.
            DrawItem::LayerBlurComposite { group, .. } => items_bytes(&group.items) + 256,
            _ => 256,
        };
    }
    bytes
}

/// The part of [`PlanCaches`] that belongs to ONE resident scene: the retained camera plan and every memo keyed
/// on that scene's generation. [`PlanCaches::take_scene_state`] moves it out when the scene is parked (a Design
/// page that is no longer shown, `page_cache`) and [`PlanCaches::restore_scene_state`] moves it back when the
/// scene is shown again, so the page's first frame back reuses its plan instead of rebuilding it. The
/// content-hashed caches (tessellation, text, outlines) are shared by every scene and stay put.
#[derive(Default)]
pub struct ScenePlanState {
    camera: Option<CameraPlanEntry>,
    camera_bytes: usize,
    camera_frame_key: Option<(u64, u64)>,
    camera_slot: Option<Box<crate::camera_slot::CameraSlot>>,
    incr_camera_bytes: usize,
    camera_scene_index: Option<Box<crate::camera_slot::SceneIndex>>,
    camera_cull: Option<crate::cull::Aabb>,
    capture_view: Option<crate::frame::CaptureView>,
    degraded_zoom_bucket: Option<f64>,
    camera_window_frames: u32,
    camera_window_zoom: Option<f64>,
    outline_memo: Option<(u64, (bool, bool))>,
    camera_dependent_memo: Option<(u64, Vec<usize>, bool)>,
    capture_memo: Option<(u64, Vec<usize>, bool)>,
    outline_pending: Vec<PendingOutline>,
    reverse_deps: Option<(u64, ReverseDeps)>,
    dl_cache: HashMap<String, Arc<moda_scene::DisplayList>>,
    dl_cache_gen: Option<u64>,
    flag_memo: HashMap<(String, u8), bool>,
    cull_index: HashMap<(String, u64), Arc<crate::cull::CullIndex>>,
}

impl ScenePlanState {
    /// Marginal heap of the retained camera plan it holds (the resident scene is counted by its owner).
    pub fn plan_bytes(&self) -> usize {
        self.incr_camera_bytes.max(self.camera_bytes)
    }

    /// Whether it holds a retained camera plan.
    pub fn has_plan(&self) -> bool {
        self.camera_slot.is_some() || self.camera.is_some()
    }
}

impl PlanCaches {
    /// Move out the state that belongs to the resident scene the caches last planned (see [`ScenePlanState`]),
    /// leaving these caches as they are for a scene they have never seen. The shared content-hashed caches stay.
    pub fn take_scene_state(&mut self) -> ScenePlanState {
        // The still frames' state is the scene's: a parked scene comes back still, and its motion state is dropped.
        self.motion_reset();
        self.take_scene_state_raw()
    }

    /// Put back a scene's state taken by [`Self::take_scene_state`], replacing whatever the caches hold for
    /// the scene they planned since (which the caller parks or drops first).
    pub fn restore_scene_state(&mut self, s: ScenePlanState) {
        self.motion_reset();
        self.restore_scene_state_raw(s);
    }

    /// Back to the still frames' scene state, dropping the moving frames' (a scene change, a config change).
    fn motion_reset(&mut self) {
        if self.motion.active {
            let _moving = self.take_scene_state_raw();
            if let Some(still) = self.motion.other.take() {
                self.restore_scene_state_raw(still);
            }
        }
        let requested = self.motion.requested;
        let (gestures, frames) = (self.motion.gestures, self.motion.frames);
        self.motion = MotionState {
            requested,
            gestures,
            frames,
            ..MotionState::default()
        };
    }

    /// Cached while moving (`crate::motion_capture`): the host says whether the next live camera frames move (the
    /// camera pans or zooms, with the setting on). While they do, the layers that cost the GPU most (blended groups,
    /// large procedural fills) draw from captures kept across the frames, and shaders in them hold the clock of
    /// the first moving frame. The first still frame after is planned exactly as with this off. Only the live
    /// camera lane (`motion_plans`) takes it; exports, readbacks and every other lane never do.
    pub fn set_motion_captures(&mut self, on: bool) {
        self.motion.requested = on;
    }

    /// Whether the last live camera frame was planned with motion captures.
    pub fn motion_captures_active(&self) -> bool {
        self.motion.active
    }

    /// The units the last live camera frame planned under (`None`: a still frame). With [`Self::motion_clock`],
    /// what a from-scratch reference build of that frame enters (the camera-slot fuzz's oracle).
    #[cfg(test)]
    pub(crate) fn motion_units(&self) -> Option<Arc<crate::motion_capture::MotionUnits>> {
        self.motion
            .active
            .then(|| self.motion.units.as_ref().map(|(_, u)| u.clone()))
            .flatten()
    }

    /// The shader clock the current motion froze at (`None`: a still frame).
    #[cfg(test)]
    pub(crate) fn motion_clock(&self) -> Option<f64> {
        self.motion.active.then_some(self.motion.clock).flatten()
    }

    /// Whether the last live camera frame was planned while the zoom changes (`motion_plans` on, the zoom changed
    /// within `MOTION_ZOOM_HOLD_MS`): its layer captures may composite an earlier zoom step's result
    /// (`Engine::set_capture_stand_ins_mid_zoom`).
    pub fn zooming(&self) -> bool {
        self.zooming
    }

    /// Diagnostics: `{active, units, gestures, frames, scale}`.
    pub fn motion_captures_json(&self) -> String {
        format!(
            "{{\"active\":{},\"requested\":{},\"units\":{},\"gestures\":{},\"frames\":{},\"scale\":{}}}",
            self.motion.active,
            self.motion.requested,
            self.motion.units.as_ref().map_or(0, |(_, u)| u.len()),
            self.motion.gestures,
            self.motion.frames,
            self.motion.scale.map_or("null".to_string(), |s| format!("{s}")),
        )
    }

    /// The motion units a live camera frame of `resident` plans under, switching the scene state when the frame
    /// starts or ends a motion (see [`MotionState`]). `None`: a still frame, or nothing to capture.
    fn motion_frame(
        &mut self,
        resident: &ResidentScene,
        shader_time_s: f64,
    ) -> Option<Arc<crate::motion_capture::MotionUnits>> {
        let units = if self.motion.requested && self.motion_plans {
            let gen = resident.generation();
            match &self.motion.units {
                Some((g, u)) if *g == gen => Some(u.clone()),
                _ => resident.scene().map(|scene| {
                    let u = Arc::new(crate::motion_capture::units(scene));
                    self.motion.units = Some((gen, u.clone()));
                    u
                }),
            }
            .filter(|u| !u.is_empty())
        } else {
            None
        };
        match (&units, self.motion.active) {
            (Some(u), false) => {
                let still = self.take_scene_state_raw();
                // The last motion's state serves this one when its units are the same.
                if let Some(moving) = self
                    .motion
                    .other
                    .take()
                    .filter(|_| self.motion.state_digest == Some(u.digest()))
                {
                    self.restore_scene_state_raw(moving);
                }
                self.motion.other = Some(still);
                self.motion.state_digest = Some(u.digest());
                self.motion.active = true;
                self.motion.clock = Some(shader_time_s);
                self.motion.scale = None;
                self.motion.gestures = self.motion.gestures.wrapping_add(1);
            }
            (Some(u), true) => {
                if self.motion.state_digest != Some(u.digest()) {
                    // An edit changed what is a unit: plan the moving frames afresh.
                    let _stale = self.take_scene_state_raw();
                    self.motion.state_digest = Some(u.digest());
                    self.motion.scale = None;
                }
            }
            (None, true) => {
                let moving = self.take_scene_state_raw();
                if let Some(still) = self.motion.other.take() {
                    self.restore_scene_state_raw(still);
                }
                self.motion.other = Some(moving);
                self.motion.active = false;
                self.motion.clock = None;
                self.motion.scale = None;
            }
            (None, false) => {}
        }
        if units.is_some() {
            self.motion.frames = self.motion.frames.wrapping_add(1);
        }
        units
    }

    fn take_scene_state_raw(&mut self) -> ScenePlanState {
        ScenePlanState {
            camera: self.camera.take(),
            camera_bytes: std::mem::take(&mut self.camera_bytes),
            camera_frame_key: self.camera_frame_key.take(),
            camera_slot: self.camera_slot.take(),
            incr_camera_bytes: std::mem::take(&mut self.incr_camera_bytes),
            camera_scene_index: self.camera_scene_index.take(),
            camera_cull: self.camera_cull.take(),
            capture_view: self.capture_view.take(),
            degraded_zoom_bucket: self.degraded_zoom_bucket.take(),
            camera_window_frames: std::mem::take(&mut self.camera_window_frames),
            camera_window_zoom: self.camera_window_zoom.take(),
            outline_memo: self.outline_memo.take(),
            camera_dependent_memo: self.camera_dependent_memo.take(),
            capture_memo: self.capture_memo.take(),
            outline_pending: std::mem::take(&mut self.outline_pending),
            reverse_deps: self.reverse_deps.take(),
            dl_cache: std::mem::take(&mut self.dl_cache),
            dl_cache_gen: self.dl_cache_gen.take(),
            flag_memo: std::mem::take(&mut self.flag_memo),
            cull_index: std::mem::take(&mut self.cull_index),
        }
    }

    fn restore_scene_state_raw(&mut self, s: ScenePlanState) {
        self.camera = s.camera;
        self.camera_bytes = s.camera_bytes;
        self.camera_frame_key = s.camera_frame_key;
        self.camera_slot = s.camera_slot;
        self.incr_camera_bytes = s.incr_camera_bytes;
        self.camera_scene_index = s.camera_scene_index;
        self.camera_cull = s.camera_cull;
        self.capture_view = s.capture_view;
        self.degraded_zoom_bucket = s.degraded_zoom_bucket;
        self.camera_window_frames = s.camera_window_frames;
        self.camera_window_zoom = s.camera_window_zoom;
        self.outline_memo = s.outline_memo;
        self.camera_dependent_memo = s.camera_dependent_memo;
        self.capture_memo = s.capture_memo;
        self.outline_pending = s.outline_pending;
        self.reverse_deps = s.reverse_deps;
        self.dl_cache = s.dl_cache;
        self.dl_cache_gen = s.dl_cache_gen;
        self.flag_memo = s.flag_memo;
        self.cull_index = s.cull_index;
    }

    /// Advance the frame tick (call once per render sweep) — drives LRU.
    pub fn begin_frame(&mut self) {
        self.tick += 1;
    }

    pub fn clear(&mut self) {
        *self = PlanCaches {
            tess_hits: self.tess_hits,
            tess_misses: self.tess_misses,
            text_hits: self.text_hits,
            text_misses: self.text_misses,
            page_hits: self.page_hits,
            page_misses: self.page_misses,
            camera_hits: self.camera_hits,
            camera_misses: self.camera_misses,
            camera_fp_skips: self.camera_fp_skips,
            page_fp_skips: self.page_fp_skips,
            evictions: self.evictions,
            // rust-edit-incremental-z4: stats survive a clear; the live editor slot
            // (`incremental_page`) is cache data and resets via `..default()`.
            incr_static_hits: self.incr_static_hits,
            incr_patch_hits: self.incr_patch_hits,
            incr_patch_nodes: self.incr_patch_nodes,
            incr_placement_nodes: self.incr_placement_nodes,
            incr_fallbacks: self.incr_fallbacks,
            incremental_disabled: self.incremental_disabled,
            cull_disabled: self.cull_disabled,
            band_rekey_disabled: self.band_rekey_disabled,
            cull_min_nodes: self.cull_min_nodes,
            cull_builds: self.cull_builds,
            motion_plans: self.motion_plans,
            // Config, not cache data — the kill switch must survive a cache clear
            // (e.g. a text-engine switch). The memo (`camera_frame_key`) resets
            // via `..default()` and self-rebuilds on the next frame.
            fingerprint_memo_disabled: self.fingerprint_memo_disabled,
            // Config set once by the host when its outline workers are ready —
            // a cache clear (text-engine switch, memory guard) must not turn
            // them off for the rest of the session. Open requests are cache
            // data: they reset (the engine re-requests).
            outline_worker: self.outline_worker,
            outline_clock: self.outline_clock,
            // Node ids a worker failed on. After a document switch they may
            // name nodes that no longer exist — harmless (never matched).
            outline_no_worker: std::mem::take(&mut self.outline_no_worker),
            // An export job's sharing is config; its entries are cache data.
            job_text: self.job_text.as_ref().map(|_| HashMap::default()),
            // The host's per-frame request, not cache data.
            motion: MotionState {
                requested: self.motion.requested,
                ..MotionState::default()
            },
            ..PlanCaches::default()
        };
    }

    /// M1: enable/disable the camera fingerprint memo. Default ENABLED; the host
    /// disables it for the `?wasmFingerprintMemo=0` kill switch. Re-enabling needs
    /// no reset: every input to the key is re-read on the re-enabled frame, so any
    /// change during the disabled window moves it — a scene edit (resident
    /// generation), a render-option change, OR an asset/pattern/font settle (the
    /// folded `store_generation` / `text_host_generation`) — and the next key
    /// differs from the stale stored one, forcing a recompute.
    pub fn set_fingerprint_memo_enabled(&mut self, on: bool) {
        self.fingerprint_memo_disabled = !on;
    }

    /// Viewport culling of the camera plan (default ON). Turning it off (or on)
    /// drops the cached camera plan and incremental slot, which were built with the
    /// other setting.
    pub fn set_viewport_cull_enabled(&mut self, on: bool) {
        if self.cull_disabled == on {
            self.motion_reset();
            self.cull_disabled = !on;
            self.camera_clear();
            self.camera_cull = None;
        }
    }

    /// Override [`CULL_MIN_NODES`] (tests use 0 to cull small scenes).
    pub fn set_cull_min_nodes(&mut self, n: Option<usize>) {
        self.cull_min_nodes = n;
    }

    /// rust-edit-incremental-z4: enable/disable the incremental plan patcher.
    /// Default ENABLED; the host disables it for the `?wasmIncrementalPlan=0` kill
    /// switch (forcing the legacy full-rebuild path). Disabling drops the live slot
    /// so re-enabling starts clean.
    pub fn set_incremental_enabled(&mut self, on: bool) {
        self.incremental_disabled = !on;
        if !on {
            self.motion_reset();
            self.incremental_page = None;
            self.camera_slot = None;
            self.incr_page_bytes = 0;
            self.incr_camera_bytes = 0;
        }
    }

    // --- Brush plan P8: progressive outline refinement --------------------

    /// Set the clock the refine budget is measured on (ms; the WASM host
    /// passes `Date.now`, native builds default to a monotonic clock).
    pub fn set_outline_clock(&mut self, clock: fn() -> f64) {
        self.outline_clock = Some(clock);
    }

    /// Let the next camera builds spend about `ms` refining outlines after a
    /// zoom-band change, drawing the rest from their previous band until
    /// refined. `None` restores "compute everything" (the default).
    /// Enter a build at another camera than the live lane's (a motion-cache tile, a readback): the live lane's
    /// camera history (the zoom it last drew, when that last changed, the scale its plan flattened at) is set aside
    /// and this build starts from its own zoom, so it neither sees a zoom change nor records one. Hand the result
    /// to [`PlanCaches::restore_camera_history`] after the build. A tile fill at the view it was anchored on used
    /// to read as a zoom change to the next live frame, which held its previous band and refined nothing, and the
    /// next fill flipped it back: an idle canvas re-tessellated every curve every few seconds.
    pub fn enter_side_camera(&mut self, zoom: f64) -> CameraHistory {
        let h = CameraHistory {
            last_camera_zoom: self.last_camera_zoom,
            last_zoom_change_ms: self.last_zoom_change_ms,
            last_plan_scale: self.last_plan_scale,
        };
        self.last_camera_zoom = Some(zoom);
        self.last_zoom_change_ms = None;
        h
    }

    /// Put back the live lane's camera history set aside by [`PlanCaches::enter_side_camera`].
    pub fn restore_camera_history(&mut self, h: CameraHistory) {
        self.last_camera_zoom = h.last_camera_zoom;
        self.last_zoom_change_ms = h.last_zoom_change_ms;
        self.last_plan_scale = h.last_plan_scale;
    }

    /// Hold outline / curve refinement while the zoom changes (default on; off for tests and A/Bs).
    pub fn set_hold_refine_while_zooming(&mut self, on: bool) {
        self.refine_while_zooming = !on;
    }

    pub fn set_outline_refine_budget_ms(&mut self, ms: Option<f64>) {
        // The clock starts at the first outline the frame refines, not at the
        // frame's start: a plan rebuild or patch clone that takes longer than
        // the budget must not starve refinement (it would never converge).
        self.outline_refine_budget_ms = ms;
        self.outline_refine_deadline = ms.map(|_| f64::INFINITY);
        self.placement_deadline = ms.map(|_| f64::INFINITY);
    }

    fn placement_scale(&self) -> f64 {
        if self.placement_filling {
            PLACEMENT_FILL_SCALE
        } else {
            PLACEMENT_BUDGET_SCALE
        }
    }

    /// Whether `id` is the stroke this build last left out ([`PlacedLookup::Defer`]).
    pub(crate) fn placement_deferred_last(&self, id: &str) -> bool {
        self.outline_pending.last().is_some_and(|p| p.unplaced && p.id == id)
    }

    /// Take a share of the frame's placement budget for a never-placed stroke; `false` = spent (leave it out and
    /// place it on a later frame). Always `true` off the live lane. The clock starts at the first placement.
    fn placement_budget_take(&mut self) -> bool {
        let Some(d) = self.placement_deadline else {
            return true;
        };
        let now = (self.outline_clock.unwrap_or(default_clock))();
        if d == f64::INFINITY {
            let ms = self.outline_refine_budget_ms.unwrap_or(0.0) * self.placement_scale();
            self.placement_deadline = Some(now + ms);
            return true;
        }
        now < d
    }

    /// Whether the last camera plan holds every node that meets canvas rect `region` (its cull window contains
    /// it, or it is unculled). The motion cache renders pieces of `region` from it.
    pub fn plan_holds(&self, region: &crate::cull::Aabb) -> bool {
        self.last_plan_window
            .is_none_or(|w| crate::cull::contains(&w, region))
    }

    /// Make the next camera build cull around its own view (the motion cache re-centres the window on its
    /// anchor so the region around it is in the plan).
    pub fn drop_cull_window(&mut self) {
        self.camera_cull = None;
    }

    /// Whether the last camera build drew any outline stale (or was served for a zoom in progress, see
    /// `motion_plan`): the host keeps rendering frames until this is false.
    pub fn outline_refining(&self) -> bool {
        if self.motion_plan {
            return true;
        }
        let view = crate::tess::outline::camera_view().or(self.outline_view);
        match view {
            // Off-screen strokes wait until the camera comes near them.
            Some(v) => {
                let near = crate::tess::outline::near_view(v);
                self.outline_pending
                    .iter()
                    .any(|p| crate::tess::outline::visible_area(p.bbox, near) > 0.0)
            }
            None => !self.outline_pending.is_empty(),
        }
    }

    /// Stale outlines the last camera frame showed on screen.
    pub fn outline_visible_pending(&self) -> usize {
        match crate::tess::outline::camera_view().or(self.outline_view) {
            Some(v) => self
                .outline_pending
                .iter()
                .filter(|p| crate::tess::outline::visible_area(p.bbox, v) > 0.0)
                .count(),
            None => self.outline_pending.len(),
        }
    }

    /// Compute deferred outlines in host workers ([`Self::take_outline_requests`] /
    /// [`Self::provide_outline`]) instead of on the main thread.
    pub fn set_outline_worker(&mut self, on: bool) {
        self.outline_worker = on;
        if !on {
            self.outline_requests.clear();
            self.outline_requested.clear();
            // In-flight placements will not arrive: promote past them (their strokes re-request or place here).
            let seqs: Vec<u64> = self.placement_requested.drain().map(|(_, (_, seq))| seq).collect();
            for seq in seqs {
                self.placement_staged.entry(seq).or_insert(None);
            }
            self.promote_placements();
        }
    }

    pub(crate) fn outline_worker(&self) -> bool {
        self.outline_worker && self.outline_refine_deadline.is_some()
    }

    /// Queue a worker request for `node`'s outline at `band` (once per key).
    pub(crate) fn outline_request(
        &mut self,
        node: &RawNode,
        key: u64,
        base: u64,
        band: i32,
        brush: Option<&Value>,
    ) {
        if self.outline_requested.get(node.id()) == Some(&key)
            || self.outline_no_worker.contains(node.id())
        {
            return;
        }
        self.outline_requested.insert(node.id().to_string(), key);
        self.outline_requests.push(
            moda_scene::value::json!({
                "id": node.id(),
                "key": key.to_string(),
                "base": base.to_string(),
                "band": band,
                "node": Value::Object(node.to_json_map()),
                "brush": brush,
            })
            .to_string(),
        );
    }

    /// A worker could not outline `node_id` (a non-trap error): forget the
    /// request and keep that node on the main thread.
    pub fn outline_request_failed(&mut self, node_id: &str) {
        self.outline_requested.remove(node_id);
        self.outline_no_worker.insert(node_id.to_string());
        if let Some((_, seq)) = self.placement_requested.remove(node_id) {
            self.placement_staged.entry(seq).or_insert(None);
            self.promote_placements();
        }
    }

    /// Queue a host worker's placement of `kind` stroke `node` (`task: "place"`, served by the worker's
    /// `tess::placed::placement_job`), under its placement `key` / `base`: a never-placed stroke, or a band change's
    /// re-placement of an art stroke (its warp is fitted to the band's tolerance). Its strands' body is tessellated
    /// there too, at the current zoom band.
    pub(crate) fn placement_request(
        &mut self,
        node: &RawNode,
        kind: crate::tess::placed::PlacedKind,
        def: &Value,
        brush_hash: Option<u64>,
        key: u64,
        base: u64,
    ) {
        if self.placement_requested.get(node.id()).is_some_and(|r| r.0 == key) {
            return;
        }
        let seq = self.placement_seq;
        self.placement_seq += 1;
        if let Some((_, old)) = self.placement_requested.insert(node.id().to_string(), (key, seq)) {
            // Superseded (the stroke changed while in flight): the old one is dropped when it arrives.
            self.placement_staged.entry(old).or_insert(None);
        }
        // Written directly (a painting opening asks for thousands at once): the brush's JSON once per brush, not a
        // deep copy of it into every request.
        let brush = match brush_hash {
            Some(h) => self
                .placement_brush_json
                .entry(h)
                .or_insert_with(|| def.to_string())
                .clone(),
            None => def.to_string(),
        };
        let node_json = moda_scene::value::to_string(&node.to_json_map()).unwrap_or_else(|_| "null".into());
        let id = Value::String(node.id().to_string());
        let band = crate::tess::outline::current_band();
        let kind = match kind {
            crate::tess::placed::PlacedKind::Art => "art",
            _ => "bristle",
        };
        self.outline_requests.push(format!(
            r#"{{"task":"place","id":{id},"key":"{key}","base":"{base}","band":{band},"kind":"{kind}","node":{node_json},"brush":{brush}}}"#
        ));
    }

    /// Queue a host worker's strand body of placed bristle stroke `node` at zoom band `band` (`task: "strands"`,
    /// served by the worker's `tess::placed::strands_job`): a band change's refinement, off the main thread. `id` is
    /// the strands' cache entry (`<node>::strands`), `key` / `base` its strand key and base
    /// (`tess::placed::strand_run_key` / `strand_run_base`); the reply is an outline's (`provide_outline` under `id`).
    /// The worker places the stroke again (as the main thread did: placement is deterministic) and refuses when its
    /// placement's strand key isn't `key`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn strands_request(
        &mut self,
        id: &str,
        node: &RawNode,
        def: &Value,
        brush_hash: Option<u64>,
        key: u64,
        base: u64,
        band: i32,
    ) {
        if self.outline_requested.get(id) == Some(&key) || self.outline_no_worker.contains(id) {
            return;
        }
        self.outline_requested.insert(id.to_string(), key);
        let brush = match brush_hash {
            Some(h) => self
                .placement_brush_json
                .entry(h)
                .or_insert_with(|| def.to_string())
                .clone(),
            None => def.to_string(),
        };
        let node_json = moda_scene::value::to_string(&node.to_json_map()).unwrap_or_else(|_| "null".into());
        let id = Value::String(id.to_string());
        self.outline_requests.push(format!(
            r#"{{"task":"strands","id":{id},"key":"{key}","base":"{base}","band":{band},"node":{node_json},"brush":{brush}}}"#
        ));
    }

    /// Whether `id` (an outline or a strand body) went back to the main thread (a worker couldn't do it).
    pub(crate) fn outline_no_worker_has(&self, id: &str) -> bool {
        self.outline_no_worker.contains(id)
    }

    /// A worker's placement for a `task: "place"` request: staged, and promoted into the placement cache (with its
    /// strand body) once every earlier request is in, so strokes appear in paint order. `false` when it is not the
    /// stroke's current request (superseded while in flight) and dropped.
    pub fn provide_placement(
        &mut self,
        node_id: &str,
        key: &str,
        base: &str,
        band: i32,
        wire: &crate::tess::placed::PlacementWire,
        strands: Option<TessellatedGeometry>,
    ) -> bool {
        let (Ok(key), Ok(base)) = (key.parse::<u64>(), base.parse::<u64>()) else {
            return false;
        };
        let Some(&(k, seq)) = self.placement_requested.get(node_id) else {
            return false;
        };
        if k != key || self.placement_staged.contains_key(&seq) {
            return false;
        }
        let staged = match crate::tess::placed::placed_from_wire(node_id, wire) {
            Some(p) => Some((node_id.to_string(), key, base, band, p, strands)),
            None => {
                // Malformed: this stroke places on the main thread from now on.
                self.placement_requested.remove(node_id);
                self.outline_no_worker.insert(node_id.to_string());
                None
            }
        };
        self.placement_staged.insert(seq, staged);
        self.promote_placements();
        true
    }

    /// Promote staged placements in request order, up to the first still in flight.
    fn promote_placements(&mut self) {
        while let Some(entry) = self.placement_staged.remove(&self.placement_promoted) {
            self.placement_promoted += 1;
            let Some((id, key, base, band, placed, strands)) = entry else {
                continue;
            };
            if self.placement_requested.get(&id).is_some_and(|r| r.0 == key) {
                self.placement_requested.remove(&id);
            }
            if let Some(g) = strands {
                let k = crate::tess::placed::strand_run_key(&placed, band);
                let b = crate::tess::placed::strand_run_base(&placed);
                self.outline_put(&format!("{id}::strands"), k, b, Some(band), Arc::new(g));
            }
            self.outline_worker_provided += 1;
            self.placed_put(&id, key, base, Arc::new(placed));
        }
    }

    /// Whether never-placed stroke `id` is with a host worker (requested, not promoted yet).
    pub(crate) fn placement_in_flight(&self, id: &str) -> bool {
        self.placement_requested.contains_key(id)
    }

    /// The outline requests since the last call, as a JSON array (`[]` when
    /// none): `[{ id, key, base, band, node, brush }]` — `key` / `base` are
    /// u64 decimal strings, handed back verbatim to [`Self::provide_outline`].
    pub fn take_outline_requests(&mut self) -> String {
        if self.outline_requests.is_empty() {
            return "[]".to_string();
        }
        format!("[{}]", std::mem::take(&mut self.outline_requests).join(","))
    }

    /// A worker's result for a request: cached under its key, so the next
    /// live frame draws it (its node is still queued as pending). Ignored when
    /// the node has moved on to another key since (edited while in flight).
    pub fn provide_outline(
        &mut self,
        node_id: &str,
        key: &str,
        base: &str,
        band: i32,
        geometry: TessellatedGeometry,
        note: Option<String>,
    ) -> bool {
        let (Ok(key), Ok(base)) = (key.parse::<u64>(), base.parse::<u64>()) else {
            return false;
        };
        if self.outline_requested.get(node_id) != Some(&key) {
            return false;
        }
        self.outline_requested.remove(node_id);
        // Still the node's latest request: any content computed since (for
        // another key) would have dropped it (`outline_put`), and a newer
        // edit re-requests under its own key. The entry may hold another
        // `base` — the pre-edit outline an edited stroke draws meanwhile.
        self.outline_worker_provided += 1;
        self.outline_put(node_id, key, base, Some(band), Arc::new(geometry));
        self.outline_note_put(node_id, key, note);
        true
    }

    /// The text cache's byte budget (`text_budget_bytes`, or [`TEXT_BUDGET_BYTES`]). An over-budget insert trims to
    /// 7/8 of it, so eviction scans are amortised.
    fn text_budget(&self) -> usize {
        if self.text_budget_bytes > 0 {
            self.text_budget_bytes
        } else {
            TEXT_BUDGET_BYTES
        }
    }

    /// Queue a deferred text layer for [`prewarm_deferred_text`], with the inputs its cache key depends on.
    pub(crate) fn text_prewarm_push(&mut self, id: &str, best_effort: bool, render_scale: f64) {
        if self.text_prewarm_ids.insert(id.to_string()) {
            self.text_prewarm.push_back(TextPrewarm {
                id: id.to_string(),
                best_effort,
                render_scale,
            });
        }
    }

    /// Deferred text layers not laid out yet.
    pub fn text_prewarm_len(&self) -> usize {
        self.text_prewarm.len()
    }

    /// Whether this build may leave a text layer for the next frame: a live camera frame's full rebuild whose
    /// refine budget is spent (the first layout of a frame always runs).
    pub(crate) fn text_deferrable(&mut self) -> bool {
        let defer = self.text_progressive && !self.outline_budget_take();
        if defer {
            self.text_deferred += 1;
        }
        defer
    }

    /// Whether this build may draw outlines stale (a live camera frame).
    pub(crate) fn outline_progressive(&self) -> bool {
        self.outline_refine_deadline.is_some()
    }

    /// Queue a node drawn stale this build.
    pub(crate) fn outline_pending_push(&mut self, id: &str, bbox: [f64; 4]) {
        self.outline_stale_served += 1;
        self.outline_pending.push(PendingOutline {
            id: id.to_string(),
            bbox,
            unplaced: false,
        });
    }

    /// Take one unit of the refine budget; `false` = spent (draw stale).
    pub(crate) fn outline_budget_take(&mut self) -> bool {
        let Some(d) = self.outline_refine_deadline else {
            return true;
        };
        let now = (self.outline_clock.unwrap_or(default_clock))();
        if d == f64::INFINITY {
            // First refinement this frame: always allowed; start the clock.
            self.outline_refine_deadline = Some(now + self.outline_refine_budget_ms.unwrap_or(0.0));
            return true;
        }
        now < d
    }

    pub(crate) fn outline_note(&self, node_id: &str, key: u64) -> Option<String> {
        self.outline_notes
            .get(node_id)
            .filter(|(k, _)| *k == key)
            .map(|(_, n)| n.clone())
    }

    pub(crate) fn outline_note_put(&mut self, node_id: &str, key: u64, note: Option<String>) {
        match note {
            Some(n) => {
                self.outline_notes.insert(node_id.to_string(), (key, n));
            }
            None => {
                self.outline_notes.remove(node_id);
            }
        }
    }

    /// Live frames: a cached outline of the same geometry and brush fitted at
    /// a FINER band (up to three bands finer) is exact at `band` too — just
    /// denser — so zooming out never waits for refinement.
    pub(crate) fn outline_finer_get(
        &mut self,
        node_id: &str,
        base: u64,
        band: i32,
    ) -> Option<Arc<TessellatedGeometry>> {
        let e = self.tess.get_mut(node_id)?;
        if e.outline_base != Some(base) {
            return None;
        }
        let ok = |b: Option<i32>| b.is_some_and(|b| b >= band && b <= band + 3);
        if ok(e.outline_bands.0) {
            e.tick = self.tick;
            return Some(e.geometry.clone());
        }
        if ok(e.outline_bands.1) {
            e.tick = self.tick;
            return e.outline_alt.as_ref().map(|(_, g)| g.clone());
        }
        None
    }

    /// The node's cached outline for any geometry (its previous content):
    /// an edited stroke past the frame budget draws this until refined.
    pub(crate) fn outline_prev_get(&mut self, node_id: &str) -> Option<Arc<TessellatedGeometry>> {
        let tick = self.tick;
        let e = self.tess.get_mut(node_id)?;
        if e.outline_base.is_none()
            || e.outline_bands.0.is_none()
            || e.prev_frames >= OUTLINE_MAX_PREV_FRAMES
        {
            return None;
        }
        e.prev_frames += 1;
        e.tick = tick;
        Some(e.geometry.clone())
    }

    /// Whether a cached outline of this node exists for the same geometry and
    /// brush (at any band).
    pub(crate) fn outline_has_base(&self, node_id: &str, base: u64) -> bool {
        self.tess
            .get(node_id)
            .is_some_and(|e| e.outline_base == Some(base))
    }

    /// A cached outline of this node for the same geometry and brush at any
    /// band (the caller queues it with [`Self::outline_pending_push`]).
    pub(crate) fn outline_stale_get(
        &mut self,
        node_id: &str,
        base: u64,
    ) -> Option<Arc<TessellatedGeometry>> {
        let e = self.tess.get_mut(node_id)?;
        if e.outline_base != Some(base) {
            return None;
        }
        e.tick = self.tick;
        Some(e.geometry.clone())
    }

    /// Put an outlined node's geometry (`key` includes the band; `base` not).
    pub(crate) fn outline_put(
        &mut self,
        node_id: &str,
        key: u64,
        base: u64,
        band: Option<i32>,
        geometry: Arc<TessellatedGeometry>,
    ) {
        // Keep the previous band of the same outline as the second slot.
        let keep = self.tess.get_mut(node_id).and_then(|e| {
            (e.outline_base == Some(base) && e.key != key)
                .then(|| (e.key, e.geometry.clone(), e.outline_bands.0))
        });
        // Anything in flight for another key of this node is now stale.
        if self
            .outline_requested
            .get(node_id)
            .is_some_and(|k| *k != key)
        {
            self.outline_requested.remove(node_id);
        }
        self.tess_put(node_id, key, geometry);
        if let Some(e) = self.tess.get_mut(node_id) {
            e.outline_base = Some(base);
            if let Some((_, g, _)) = &keep {
                self.tess_bytes += geometry_bytes(g);
            }
            e.outline_bands = (band, keep.as_ref().and_then(|k| k.2));
            e.outline_alt = keep.map(|(k, g, _)| (k, g));
        }
    }

    // --- Brush plan P8: placed art / scatter brush artwork ---------------

    /// The placement for `key`; on a miss past the refine budget, the node's
    /// placement for the same geometry and brush at another band (served
    /// stale and queued for refinement). `None` = compute it.
    pub(crate) fn placed_get(
        &mut self,
        node_id: &str,
        key: u64,
        base: u64,
        bbox: [f64; 4],
        // A host worker can place it (`tess::placed::worker_placeable`: a bristle or art stroke without trim / dash).
        worker_placeable: bool,
    ) -> PlacedLookup {
        let tick = self.tick;
        if let Some(e) = self.placed.get_mut(node_id) {
            if e.key == key {
                e.tick = tick;
                e.stale_frames = 0;
                return PlacedLookup::Hit(e.placed.clone());
            }
        }
        // Never placed (a document opening): within the frame's placement budget, place it; past it, leave it out
        // this frame (it never drew, so nothing disappears) and place it on a later one, back to front.
        if !self.placed_ever.contains(node_id) {
            // With host workers, every never-placed stroke they can place goes to them (in paint order, promoted
            // in that order): the main thread draws what has arrived.
            if worker_placeable && self.outline_worker() && !self.outline_no_worker.contains(node_id) {
                let in_flight = self.placement_requested.get(node_id).is_some_and(|r| r.0 == key);
                self.placement_deferred += 1;
                self.outline_pending.push(PendingOutline {
                    id: node_id.to_string(),
                    bbox,
                    unplaced: true,
                });
                return if in_flight {
                    PlacedLookup::Defer
                } else {
                    PlacedLookup::Request
                };
            }
            if self.placement_budget_take() {
                return PlacedLookup::Place;
            }
            self.placement_deferred += 1;
            self.outline_pending.push(PendingOutline {
                id: node_id.to_string(),
                bbox,
                unplaced: true,
            });
            return PlacedLookup::Defer;
        }
        // Placed at another band (art): with host workers, re-placed there; this placement draws until the new one
        // arrives (a worker's reply is due, so no stale-frame cap), and none is placed on the main thread.
        if worker_placeable && self.outline_worker() && !self.outline_no_worker.contains(node_id) {
            if let Some(e) = self.placed.get_mut(node_id).filter(|e| e.base == base) {
                e.tick = tick;
                let placed = e.placed.clone();
                let in_flight = self.placement_requested.get(node_id).is_some_and(|r| r.0 == key);
                self.outline_pending_push(node_id, bbox);
                return if in_flight {
                    PlacedLookup::Hit(placed)
                } else {
                    PlacedLookup::Refine(placed)
                };
            }
        }
        if self.outline_budget_take() {
            return PlacedLookup::Place;
        }
        let Some(e) = self.placed.get_mut(node_id) else {
            return PlacedLookup::Place;
        };
        if e.base != base || e.stale_frames >= PLACED_MAX_STALE_FRAMES {
            return PlacedLookup::Place;
        }
        e.stale_frames += 1;
        e.tick = tick;
        let placed = e.placed.clone();
        self.outline_pending_push(node_id, bbox);
        PlacedLookup::Hit(placed)
    }

    pub(crate) fn placed_put(
        &mut self,
        node_id: &str,
        key: u64,
        base: u64,
        placed: Arc<crate::tess::placed::Placed>,
    ) {
        let bytes: usize = placed
            .nodes
            .iter()
            .map(|n| 64 + n.points().map_or(0, |p| p.len() * 96))
            .sum();
        // Fewer shapes than before: drop the tess entries of the copies that
        // no longer exist (`<node>::brush::<k>`).
        self.placed_ever.insert(node_id.to_string());
        let old_len = self.placed.get(node_id).map_or(0, |e| e.placed.nodes.len());
        for k in placed.nodes.len()..old_len {
            if let Some(e) = self.tess.remove(&format!("{node_id}::brush::{k}")) {
                self.tess_bytes = self.tess_bytes.saturating_sub(geometry_bytes(&e.geometry));
            }
        }
        if let Some(old) = self.placed.insert(
            node_id.to_string(),
            PlacedEntry {
                key,
                base,
                placed,
                bytes,
                tick: self.tick,
                stale_frames: 0,
            },
        ) {
            self.placed_bytes = self.placed_bytes.saturating_sub(old.bytes);
        }
        self.placed_bytes += bytes;
        // As the tessellation cache: down to a low-water mark (`evict_oldest` walks and sorts every entry, and
        // trimming just under the budget re-ran it on every placement once a painting's strokes outgrew it: O(n²)
        // per frame), and never a placement this frame or the last used (a view whose placements outgrow the
        // budget evicted its own mid-frame and placed every stroke again on the next zoom or pan).
        if self.placed_bytes > TESS_BUDGET_BYTES / 4 && self.placed_evict_blocked != Some(self.tick) {
            let culled = &self.brush_kept;
            let evicted = evict_stale(
                &mut self.placed,
                self.tick,
                |e| e.tick,
                |e| e.bytes,
                &mut self.placed_bytes,
                TESS_BUDGET_BYTES / 4 / 8 * 7,
                |k| Self::brush_kept_has(culled, k),
            );
            self.evictions += evicted;
            if self.placed_bytes > TESS_BUDGET_BYTES / 4 {
                self.placed_evict_blocked = Some(self.tick);
            }
        }
    }

    // --- R3: body tessellation -------------------------------------------
    pub(crate) fn tess_get(&mut self, node_id: &str, key: u64) -> Option<Arc<TessellatedGeometry>> {
        match self.tess.get_mut(node_id) {
            Some(e) if e.key == key => {
                e.tick = self.tick;
                self.tess_hits += 1;
                Some(e.geometry.clone()) // Arc pointer-bump, not a vertex-stream copy
            }
            // An outline's second band: swap it to the front (LRU of two).
            Some(e) if e.outline_alt.as_ref().is_some_and(|(k, _)| *k == key) => {
                let (k, g) = e.outline_alt.take().expect("checked");
                let old = std::mem::replace(&mut e.geometry, g);
                e.outline_alt = Some((e.key, old));
                e.key = k;
                e.outline_bands = (e.outline_bands.1, e.outline_bands.0);
                e.tick = self.tick;
                self.tess_hits += 1;
                Some(e.geometry.clone())
            }
            _ => {
                self.tess_misses += 1;
                None
            }
        }
    }

    pub(crate) fn tess_put(&mut self, node_id: &str, key: u64, geometry: Arc<TessellatedGeometry>) {
        let bytes = geometry_bytes(&geometry);
        if let Some(old) = self.tess.insert(
            node_id.to_string(),
            TessEntry {
                key,
                geometry,
                tick: self.tick,
                outline_base: None,
                outline_alt: None,
                outline_bands: (None, None),
                prev_frames: 0,
            },
        ) {
            self.tess_bytes = self.tess_bytes.saturating_sub(tess_entry_bytes(&old));
        }
        self.tess_bytes += bytes;
        if self.tess_bytes > TESS_BUDGET_BYTES && self.tess_evict_blocked != Some(self.tick) {
            // Down to a low-water mark, and never the geometry this frame or the last drew: the frame's plan holds
            // those `Arc`s anyway (dropping them frees nothing), and a view larger than the budget (a 20k-layer
            // document at fit) evicted its own geometry mid-frame and re-tessellated all of it every frame.
            let culled = &self.brush_kept;
            let evicted = evict_stale(
                &mut self.tess,
                self.tick,
                |e| e.tick,
                tess_entry_bytes,
                &mut self.tess_bytes,
                TESS_BUDGET_BYTES / 8 * 7,
                |k| Self::brush_kept_has(culled, k),
            );
            self.evictions += evicted;
            if self.tess_bytes > TESS_BUDGET_BYTES {
                self.tess_evict_blocked = Some(self.tick);
            }
        }
    }

    // --- R3: line tessellation (multi-part) --------------------------------
    pub(crate) fn line_tess_get(
        &mut self,
        node_id: &str,
        key: u64,
    ) -> Option<Vec<TessellatedGeometry>> {
        match self.line_tess.get_mut(node_id) {
            Some(e) if e.key == key => {
                e.tick = self.tick;
                self.tess_hits += 1;
                Some(e.geometries.clone())
            }
            _ => {
                self.tess_misses += 1;
                None
            }
        }
    }

    pub(crate) fn line_tess_put(
        &mut self,
        node_id: &str,
        key: u64,
        geometries: &[TessellatedGeometry],
    ) {
        let bytes: usize = geometries.iter().map(geometry_bytes).sum();
        if let Some(old) = self.line_tess.insert(
            node_id.to_string(),
            LineTessEntry {
                key,
                geometries: geometries.to_vec(),
                tick: self.tick,
            },
        ) {
            self.line_tess_bytes = self
                .line_tess_bytes
                .saturating_sub(old.geometries.iter().map(geometry_bytes).sum());
        }
        self.line_tess_bytes += bytes;
        if self.line_tess_bytes > TESS_BUDGET_BYTES / 4 && self.line_evict_blocked != Some(self.tick) {
            let evicted = evict_stale(
                &mut self.line_tess,
                self.tick,
                |e| e.tick,
                |e| e.geometries.iter().map(geometry_bytes).sum(),
                &mut self.line_tess_bytes,
                TESS_BUDGET_BYTES / 4 / 8 * 7,
                |_| false,
            );
            self.evictions += evicted;
            if self.line_tess_bytes > TESS_BUDGET_BYTES / 4 {
                self.line_evict_blocked = Some(self.tick);
            }
        }
    }

    // --- export job sharing (`export_job`) -----------------------------------
    /// Turn on an export job's sharing: text batches are looked up by content key alone and replay their build's
    /// resource events, and plain shapes tessellate through the process-wide command-keyed cache. The caller keys
    /// the caches on [`content_hashes_for_scene`].
    pub fn enable_job_sharing(&mut self) {
        self.job_text.get_or_insert_with(HashMap::default);
    }

    pub(crate) fn job_sharing(&self) -> bool {
        self.job_text.is_some()
    }

    /// End an export job's page: drop every entry the page didn't use (two-generation eviction: what
    /// the next page can reuse is what this one drew), so the caches hold one page's working set.
    pub(crate) fn end_job_page(&mut self) {
        let now = self.tick;
        self.tess.retain(|_, e| e.tick == now);
        self.tess_bytes = self.tess.values().map(tess_entry_bytes).sum();
        self.line_tess.retain(|_, e| e.tick == now);
        self.line_tess_bytes = self
            .line_tess
            .values()
            .map(|e| e.geometries.iter().map(geometry_bytes).sum::<usize>())
            .sum();
        self.placed.retain(|_, e| e.tick == now);
        self.placed_bytes = self.placed.values().map(|e| e.bytes).sum();
        self.text.retain(|_, e| e.tick == now);
        self.text_bytes = self.text.values().map(|e| batch_bytes(&e.batch)).sum();

        self.pages.retain(|_, e| e.tick == now);
        self.page_bytes = self.pages.values().map(|e| e.bytes).sum();
        if let Some(job) = self.job_text.as_mut() {
            job.retain(|_, e| e.tick == now);
            self.job_text_bytes = job.values().map(|e| batch_bytes(&e.batch)).sum();
        }
        // Per-node notes of entries that may be gone (re-made with the entry).
        let tess = &self.tess;
        self.outline_notes.retain(|k, _| tess.contains_key(k));
    }

    /// Bytes an export job's caches hold.
    pub(crate) fn job_bytes(&self) -> usize {
        self.tess_bytes + self.line_tess_bytes + self.placed_bytes + self.text_bytes + self.page_bytes + self.job_text_bytes
    }

    /// The text batch for `item_id` with content `key`: in an export job any layer's with that key (its
    /// `node_id` set to `item_id`, its build's resource events recorded again as `item_id`'s), else
    /// [`Self::text_get`].
    pub(crate) fn text_get_content(&mut self, item_id: &str, key: u64) -> Option<Box<crate::seams::TextQuadBatch>> {
        let Some(job) = self.job_text.as_mut() else {
            return self.text_get(item_id, key);
        };
        match job.get_mut(&key) {
            Some(e) => {
                e.tick = self.tick;
                self.text_hits += 1;
                moda_assets::report::replay(&e.events, item_id);
                let mut batch = e.batch.clone();
                batch.node_id = item_id.to_string();
                Some(batch)
            }
            None => {
                self.text_misses += 1;
                None
            }
        }
    }

    /// Before building a text batch [`Self::text_put_content`] may keep: in an export job, start capturing the
    /// resource events the build records.
    pub(crate) fn text_build_begin(&self) {
        if self.job_sharing() {
            moda_assets::report::capture_begin();
        }
    }

    /// After the build [`Self::text_build_begin`] opened: keep `batch` (when it built one) under `key`, with
    /// the events its build recorded in an export job. Always call it once after the begin, built or not.
    pub(crate) fn text_build_end(&mut self, item_id: &str, key: u64, batch: Option<&crate::seams::TextQuadBatch>) {
        if self.job_text.is_none() {
            if let Some(batch) = batch {
                self.text_put(item_id, key, batch);
            }
            return;
        }
        let events = moda_assets::report::capture_end();
        let Some(batch) = batch else { return };
        // A layer-quality warning names its layer in its message, which a replay as another layer can't
        // rename: build such a layer every time (a text build records none today).
        if events.iter().any(|e| {
            matches!(
                e,
                moda_assets::report::Captured::Event(moda_assets::report::RuntimeEvent::RenderQuality { .. })
            )
        }) {
            return;
        }
        let bytes = batch_bytes(batch);
        let job = self.job_text.as_mut().expect("checked");
        if let Some(old) = job.insert(
            key,
            JobTextEntry {
                batch: Box::new(batch.clone()),
                events: events.into(),
                tick: self.tick,
            },
        ) {
            self.job_text_bytes = self.job_text_bytes.saturating_sub(batch_bytes(&old.batch));
        }
        self.job_text_bytes += bytes;
    }

    // --- R4: text item cache ------------------------------------------------
    pub(crate) fn text_get(
        &mut self,
        item_id: &str,
        key: u64,
    ) -> Option<Box<crate::seams::TextQuadBatch>> {
        self.text_get_with_gap(item_id, key).map(|(b, _)| b)
    }

    pub(crate) fn text_get_with_gap(
        &mut self,
        item_id: &str,
        key: u64,
    ) -> Option<(Box<crate::seams::TextQuadBatch>, Option<(f64, f64)>)> {
        // An export job keeps text by content (`text_get_content`); the id-keyed entries (connector labels) would
        // serve without replaying their build's resource events, so a job builds those every time.
        if self.job_sharing() {
            self.text_misses += 1;
            return None;
        }
        match self.text.get_mut(item_id) {
            Some(e) if e.key == key => {
                e.tick = self.tick;
                self.text_hits += 1;
                Some((e.batch.clone(), e.gap))
            }
            _ => {
                self.text_misses += 1;
                None
            }
        }
    }

    pub(crate) fn text_put(
        &mut self,
        item_id: &str,
        key: u64,
        batch: &crate::seams::TextQuadBatch,
    ) {
        self.text_put_with_gap(item_id, key, batch, None)
    }

    pub(crate) fn text_put_with_gap(
        &mut self,
        item_id: &str,
        key: u64,
        batch: &crate::seams::TextQuadBatch,
        gap: Option<(f64, f64)>,
    ) {
        if self.job_sharing() {
            return;
        }
        let bytes = batch_bytes(batch);
        if let Some(old) = self.text.insert(
            item_id.to_string(),
            TextEntry {
                key,
                batch: Box::new(batch.clone()),
                gap,
                tick: self.tick,
            },
        ) {
            self.text_bytes = self.text_bytes.saturating_sub(batch_bytes(&old.batch));
        }
        self.text_bytes += bytes;
        let budget = self.text_budget();
        if self.text_bytes > budget {
            // Down to the low-water mark, not just under the budget: `evict_oldest` clones and sorts every key,
            // and trimming only to the budget re-ran it on every insert once text outgrew it (thousands of text
            // nodes: O(n²) per frame).
            let evicted = evict_oldest(
                &mut self.text,
                self.tick,
                |e| e.tick,
                |e| batch_bytes(&e.batch),
                &mut self.text_bytes,
                budget / 8 * 7,
            );
            self.evictions += evicted;
        }
    }

    // --- R5: page plan cache -------------------------------------------------
    pub fn page_get(
        &mut self,
        page_id: &str,
        fingerprint: u64,
        shader_time_s: f64,
    ) -> Option<(Arc<CoveragePlan>, Arc<PlanWarnings>)> {
        match self.pages.get_mut(page_id) {
            Some(e)
                if e.fingerprint == fingerprint
                    && (!e.uses_shader_time || e.shader_time_s == shader_time_s) =>
            {
                e.tick = self.tick;
                self.page_hits += 1;
                Some((e.plan.clone(), e.warnings.clone()))
            }
            _ => {
                self.page_misses += 1;
                None
            }
        }
    }

    pub fn page_put(
        &mut self,
        page_id: &str,
        fingerprint: u64,
        shader_time_s: f64,
        plan: Arc<CoveragePlan>,
        warnings: Arc<PlanWarnings>,
    ) {
        let uses_shader_time = plan_uses_shader_time(&plan);
        let bytes = plan_bytes(&plan);
        if let Some(old) = self.pages.insert(
            page_id.to_string(),
            PagePlanEntry {
                fingerprint,
                uses_shader_time,
                shader_time_s,
                plan,
                warnings,
                bytes,
                tick: self.tick,
            },
        ) {
            self.page_bytes = self.page_bytes.saturating_sub(old.bytes);
        }
        self.page_bytes += bytes;
        // PLAN §4: page plans ≤16 MB (byte LRU), entry cap as backstop.
        if self.page_bytes > PAGE_PLAN_BUDGET_BYTES {
            let evicted = evict_oldest(
                &mut self.pages,
                self.tick,
                |e| e.tick,
                |e| e.bytes,
                &mut self.page_bytes,
                PAGE_PLAN_BUDGET_BYTES,
            );
            self.evictions += evicted;
        }
        if self.pages.len() > PAGE_PLAN_BUDGET_ENTRIES {
            let mut entries: Vec<(String, u64)> = self
                .pages
                .iter()
                .map(|(k, e)| (k.clone(), e.tick))
                .collect();
            entries.sort_by_key(|(_, t)| *t);
            for (k, _) in entries
                .iter()
                .take(self.pages.len() - PAGE_PLAN_BUDGET_ENTRIES)
            {
                if let Some(old) = self.pages.remove(k) {
                    self.page_bytes = self.page_bytes.saturating_sub(old.bytes);
                    self.evictions += 1;
                }
            }
        }
    }

    // --- whole-frame plan cache (camera, single slot) ----------------------
    pub fn camera_get(
        &mut self,
        fingerprint: u64,
        shader_time_s: f64,
    ) -> Option<(Arc<CoveragePlan>, Arc<PlanWarnings>)> {
        match &self.camera {
            // A re-threadable plan ignores the clock in the hit key —
            // an animating procedural-fill shader reuses the cached plan and the
            // live clock rides the present override. A non-re-threadable plan
            // (text shader fill / non-linear clock) keeps the exact-clock-match
            // condition so it rebuilds per clock rather than freezing.
            Some(e)
                if e.fingerprint == fingerprint
                    && (!e.uses_shader_time
                        || e.shader_time_rethreadable
                        || e.shader_time_s == shader_time_s) =>
            {
                self.camera_hits += 1;
                Some((e.plan.clone(), e.warnings.clone()))
            }
            _ => {
                self.camera_misses += 1;
                None
            }
        }
    }

    pub fn camera_put(
        &mut self,
        fingerprint: u64,
        shader_time_s: f64,
        plan: Arc<CoveragePlan>,
        warnings: Arc<PlanWarnings>,
    ) {
        let bytes = plan_bytes(&plan);
        // Bound the single camera slot with its OWN (generous) budget — see
        // CAMERA_PLAN_BUDGET_BYTES. A truly pathological plan still won't be
        // retained, so it can't pin memory and prematurely trip the coarse wasm
        // memory guard (which disables ALL residency one-way); it rebuilds each
        // frame instead (the per-node R3/R4 tessellation/text caches survive, so
        // it only re-combines, never re-tessellates). But a normal dense viewport
        // now fits and is cached, so a static pan reuses it instead of re-combining
        // every frame.
        if bytes > CAMERA_PLAN_BUDGET_BYTES {
            self.camera_clear();
            return;
        }
        let uses_shader_time = plan_uses_shader_time(&plan);
        let shader_time_rethreadable = plan_shader_time_rethreadable(&plan);
        // Single slot: the new plan's bytes simply replace the old slot's.
        self.camera_bytes = bytes;
        self.camera = Some(CameraPlanEntry {
            fingerprint,
            uses_shader_time,
            shader_time_s,
            shader_time_rethreadable,
            plan,
            warnings,
        });
    }

    /// Drop the cached camera plan + free its memory. Used when entering the
    /// backdrop bypass path (`build_camera_plan_retained`), where the slot is no
    /// longer consulted: an otherwise-retained non-backdrop plan would be dead
    /// weight (the largest single plan — all visible pages) for the whole
    /// backdrop session. Idempotent.
    /// Start the live-generation memos over when the resident generation moved.
    fn live_memo_gen(&mut self) -> Option<u64> {
        let gen = self.dl_live_gen?;
        if self.dl_cache_gen != Some(gen) {
            self.dl_cache.clear();
            self.flag_memo.clear();
            self.cull_index.clear();
            self.dl_cache_gen = Some(gen);
        }
        Some(gen)
    }

    /// A placed brush stroke's learned canvas box and the band it was drawn at, when `key` still describes it.
    pub(crate) fn brush_bounds_get(&self, node_id: &str, key: u64) -> Option<(crate::cull::Aabb, i32)> {
        self.brush_bounds
            .get(node_id)
            .filter(|(k, _, _)| *k == key)
            .map(|(_, b, band)| (*b, *band))
    }

    /// Note the placed brush strokes a culled camera build reached (see `brush_kept`).
    pub(crate) fn brush_kept_note(&mut self, ids: &[String]) {
        for id in ids {
            if !self.brush_kept.contains(id) {
                self.brush_kept.insert(id.clone());
            }
        }
    }

    /// Whether cache entry `key` (a node id, or `<node id>::…` for its parts) belongs to a kept stroke.
    fn brush_kept_has(culled: &std::collections::HashSet<String>, key: &str) -> bool {
        culled.contains(key.split_once("::").map_or(key, |(n, _)| n))
    }

    /// Record a placed brush stroke's drawn canvas box (see `brush_bounds`).
    pub(crate) fn brush_bounds_put(&mut self, node_id: &str, key: u64, bounds: crate::cull::Aabb, band: i32) {
        self.brush_bounds.insert(node_id.to_string(), (key, bounds, band));
    }

    /// The viewport-cull index for display list `key` under options `opts_key`,
    /// built by `build` unless this generation already has it.
    pub(crate) fn cull_index(
        &mut self,
        key: &str,
        opts_key: u64,
        build: impl FnOnce() -> crate::cull::CullIndex,
    ) -> Arc<crate::cull::CullIndex> {
        if self.live_memo_gen().is_none() {
            return Arc::new(build());
        }
        let k = (key.to_string(), opts_key);
        if let Some(ix) = self.cull_index.get(&k) {
            return ix.clone();
        }
        let ix = Arc::new(build());
        self.cull_index.insert(k, ix.clone());
        ix
    }

    /// The display list cached for `key` at the live resident generation.
    pub(crate) fn dl_get(&mut self, key: &str) -> Option<Arc<moda_scene::DisplayList>> {
        self.live_memo_gen()?;
        self.dl_cache.get(key).cloned()
    }

    /// A scene predicate (a page's backdrop / blend / layer-blur content, …) memoised
    /// for the live resident generation; computed fresh outside the camera lane.
    pub(crate) fn memo_flag(&mut self, key: &str, which: u8, f: impl FnOnce() -> bool) -> bool {
        if self.live_memo_gen().is_none() {
            return f();
        }
        if let Some(&v) = self.flag_memo.get(&(key.to_string(), which)) {
            return v;
        }
        let v = f();
        self.flag_memo.insert((key.to_string(), which), v);
        v
    }

    pub(crate) fn set_frame_canvas_dl(&mut self, scene: &moda_scene::Scene, dl: Option<Arc<moda_scene::DisplayList>>) {
        self.frame_canvas_dl = dl.map(|dl| (std::ptr::from_ref(scene) as usize, dl));
    }

    /// The canvas root's display list the caller built for this plan, if it was built from `scene`.
    pub(crate) fn frame_canvas_dl(&self, scene: &moda_scene::Scene) -> Option<Arc<moda_scene::DisplayList>> {
        self.frame_canvas_dl
            .as_ref()
            .filter(|(at, _)| *at == std::ptr::from_ref(scene) as usize)
            .map(|(_, dl)| dl.clone())
    }

    pub(crate) fn dl_put(&mut self, key: &str, dl: Arc<moda_scene::DisplayList>) {
        if self.dl_live_gen.is_some() && self.dl_cache_gen == self.dl_live_gen {
            self.dl_cache.insert(key.to_string(), dl);
        }
    }

    /// A node's items from an earlier culled build of this generation, when they
    /// were built at render-z `z` (`None` outside a culled camera build).
    pub(crate) fn node_items_get(&self, node_id: &str, z: u64) -> Option<&CachedNodeItems> {
        if !self.node_items_live {
            return None;
        }
        self.node_items.get(node_id).filter(|c| c.z == z)
    }

    /// Keep a node's freshly built items for later culled builds of this generation:
    /// only self-contained ones (no mask reference into this plan's mask table, no
    /// split or capture group) that draw no stale stand-in awaiting refinement.
    pub(crate) fn node_items_put(
        &mut self,
        node_id: &str,
        z: u64,
        z_consumed: u64,
        items: &[crate::seams::DrawItem],
        skipped: &[String],
        pending_before: usize,
    ) {
        use crate::seams::DrawItem;
        // Shader-clock items bake the build's clock; they are rebuilt each time.
        if !self.node_items_live
            || self.outline_pending.len() != pending_before
            || items_use_shader_time(items)
        {
            return;
        }
        if items.iter().any(|i| {
            drawitem_mask(i).is_some()
                || matches!(
                    i,
                    DrawItem::BackdropSplit(_) | DrawItem::LayerBlurComposite { .. }
                )
        }) {
            return;
        }
        let bytes = owned_items_bytes(items) + node_id.len() + 64;
        let replaced = self.node_items.get(node_id).map_or(0, |c| c.bytes);
        // Over budget: this node is rebuilt when a later window needs it.
        if self.node_items_bytes - replaced + bytes > NODE_ITEMS_BUDGET_BYTES {
            return;
        }
        self.node_items_bytes = self.node_items_bytes - replaced + bytes;
        self.node_items.insert(
            node_id.to_string(),
            CachedNodeItems {
                bytes,
                z,
                z_consumed,
                items: items.to_vec(),
                skipped: skipped.to_vec(),
            },
        );
    }

    /// Entries and [`owned_items_bytes`] of the culled builds' per-node item cache.
    pub fn node_items_stats(&self) -> (usize, usize) {
        (self.node_items.len(), self.node_items_bytes)
    }

    /// Outline refinements queued so far (a node that queued one drew a stand-in).
    pub(crate) fn outline_pending_len(&self) -> usize {
        self.outline_pending.len()
    }

    /// The retained camera slot (tests).
    #[cfg(test)]
    pub(crate) fn camera_slot_ref(&self) -> Option<&crate::camera_slot::CameraSlot> {
        self.camera_slot.as_deref()
    }

    /// The retained camera slot fell back to a full rebuild.
    fn note_slot_fallback(&mut self, why: &'static str) {
        self.incr_fallbacks = self.incr_fallbacks.wrapping_add(1);
        self.incr_last_fallback = Some(why);
    }

    fn note_cull(&mut self, out: &crate::frame::CullOut) {
        self.cull_builds = self.cull_builds.wrapping_add(1);
        self.cull_last_considered = out.considered;
        self.cull_last_culled = out.items;
        self.cull_copied_nodes = self.cull_copied_nodes.wrapping_add(out.copied_nodes as u64);
    }

    pub fn camera_clear(&mut self) {
        self.camera = None;
        self.camera_bytes = 0;
        // M1: the memo's stored fingerprint pairs with the (now-dropped) slot —
        // drop it too so the next non-backdrop frame recomputes against a fresh
        // slot rather than reusing a fingerprint whose plan is gone.
        self.camera_frame_key = None;
        // rust-edit-incremental-z4: the incremental camera slot pairs with the same
        // lane; drop it so a backdrop session (which bypasses the camera cache)
        // doesn't leave a stale incremental plan to be patched on its way out.
        self.camera_slot = None;
        self.incr_camera_bytes = 0;
    }

    pub fn stats_json(&self) -> String {
        format!(
            "{{\"tess_entries\":{},\"tess_bytes\":{},\"line_entries\":{},\"line_bytes\":{},\"text_entries\":{},\"text_bytes\":{},\"page_entries\":{},\"page_bytes\":{},\"camera_bytes\":{},\"tess_hits\":{},\"tess_misses\":{},\"text_hits\":{},\"text_misses\":{},\"page_hits\":{},\"page_misses\":{},\"camera_hits\":{},\"camera_misses\":{},\"camera_fp_skips\":{},\"page_fp_skips\":{},\"evictions\":{},\"incr_static_hits\":{},\"incr_patch_hits\":{},\"incr_patch_nodes\":{},\"incr_placement_nodes\":{},\"incr_fallbacks\":{},\"incr_page_bytes\":{},\"incr_camera_bytes\":{},\"outline_refining\":{},\"outline_stale_served\":{},\"outline_visible_pending\":{},\"outline_pending\":{},\"outline_band\":{},\"outline_worker_provided\":{},\"cull_builds\":{},\"cull_last_considered\":{},\"cull_last_culled\":{},\"cull_copied_nodes\":{},\"node_items_entries\":{},\"node_items_bytes\":{},\"text_deferred\":{},\"text_prewarmed\":{},\"text_prewarm_queued\":{},\"incr_last_fallback\":{},\"camera_dependent\":{},\"camera_rebuilds\":{},\"placement_deferred\":{}}}",
            self.tess.len(),
            self.tess_bytes,
            self.line_tess.len(),
            self.line_tess_bytes,
            self.text.len(),
            self.text_bytes,
            self.pages.len(),
            self.page_bytes,
            self.camera.as_ref().map_or(0, |_| self.camera_bytes),
            self.tess_hits,
            self.tess_misses,
            self.text_hits,
            self.text_misses,
            self.page_hits,
            self.page_misses,
            self.camera_hits,
            self.camera_misses,
            self.camera_fp_skips,
            self.page_fp_skips,
            self.evictions,
            self.incr_static_hits,
            self.incr_patch_hits,
            self.incr_patch_nodes,
            self.incr_placement_nodes,
            self.incr_fallbacks,
            self.incr_page_bytes,
            self.incr_camera_bytes,
            self.outline_refining(),
            self.outline_stale_served,
            self.outline_visible_pending(),
            self.outline_pending.len(),
            self.outline_band.map_or("null".to_string(), |b| b.to_string()),
            self.outline_worker_provided,
            self.cull_builds,
            self.cull_last_considered,
            self.cull_last_culled,
            self.cull_copied_nodes,
            self.node_items.len(),
            self.node_items_bytes,
            self.text_deferred,
            self.text_prewarmed,
            self.text_prewarm.len(),
            self.incr_last_fallback
                .map_or("null".to_string(), |w| moda_scene::value::Value::from(w).to_string()),
            self.camera_dependent_memo.as_ref().is_some_and(|m| m.2),
            self.camera_rebuilds,
            self.placement_deferred,
        )
    }
}

/// Evict least-recently-used entries until `bytes` <= `budget`. Returns the
/// eviction count. Linear scan — runs only on budget crossings.
/// [`evict_oldest`] of the entries not used this frame or the last (`tick_of(v) + 1 < now`) and not `in_use`: the
/// working set stays, even over `budget`.
fn evict_stale<V>(
    map: &mut HashMap<String, V>,
    now: u64,
    tick_of: impl Fn(&V) -> u64,
    bytes_of: impl Fn(&V) -> usize,
    bytes: &mut usize,
    budget: usize,
    in_use: impl Fn(&str) -> bool,
) -> u64 {
    let mut entries: Vec<(String, u64)> = map
        .iter()
        .filter(|(k, v)| tick_of(v).saturating_add(1) < now && !in_use(k))
        .map(|(k, v)| (k.clone(), tick_of(v)))
        .collect();
    entries.sort_by_key(|(_, t)| *t);
    let mut evicted = 0;
    for (k, _) in entries {
        if *bytes <= budget {
            break;
        }
        if let Some(v) = map.remove(&k) {
            *bytes = bytes.saturating_sub(bytes_of(&v));
            evicted += 1;
        }
    }
    evicted
}

fn evict_oldest<V>(
    map: &mut HashMap<String, V>,
    _now: u64,
    tick_of: impl Fn(&V) -> u64,
    bytes_of: impl Fn(&V) -> usize,
    bytes: &mut usize,
    budget: usize,
) -> u64 {
    let mut entries: Vec<(String, u64)> =
        map.iter().map(|(k, v)| (k.clone(), tick_of(v))).collect();
    entries.sort_by_key(|(_, t)| *t);
    let mut evicted = 0;
    for (k, _) in entries {
        if *bytes <= budget {
            break;
        }
        if let Some(v) = map.remove(&k) {
            *bytes = bytes.saturating_sub(bytes_of(&v));
            evicted += 1;
        }
    }
    evicted
}

/// The content key of one layer capture (`LayerBlurGroup::cache_key`): a hash
/// of everything its pixels are made of, so the retained engine can reuse the
/// filtered result (`layer_blur::cache`) whenever it recurs.
///
/// - each captured node (`ids`: the node, or a subtree root and its
///   descendants): its content hash (the resident map's, or computed), its
///   canvas affine (a parent's move or resize reaches every descendant this
///   way), its effective opacity (ancestors folded in) and clip;
/// - the capture: region, size and filters — the camera's part (the zoom
///   bucket and the capture window the region was cut to) arrives through
///   these;
/// - scene-wide inputs: plan-relevant render options (the curve band
///   included), the asset store and the font host generations.
///
/// `None` without resident hashes (exports and one-shot renders re-render
/// every capture, as before) or when the capture holds a shader-clock
/// consumer (its pixels change every frame).
///
/// Returns (the content key, the identity key): the identity key folds what
/// the capture draws — the nodes, placement, opacity, clips and resources —
/// but not the capture's geometry (region, size, filter px) or the zoom band,
/// so two zoom steps of one capture share it (the live lane's stand-in while a
/// step's new result waits, `layer_blur::cache`).
pub(crate) fn capture_key<'a>(
    scene: &Scene,
    hashes: Option<&HashMap<String, NodeHashes>>,
    ids: impl IntoIterator<Item = &'a str>,
    group: &crate::seams::LayerBlurGroup,
    page_clip: [f32; 4],
    opts: &RenderOptions,
) -> (Option<u64>, Option<u64>) {
    let Some(hashes) = hashes else {
        return (None, None);
    };
    // A motion capture (`crate::motion_capture`) holds its shaders at the clock the motion froze
    // (`PlanCaches::set_motion_captures`): keyed on that clock, it is kept while the camera moves.
    let clocked = items_use_shader_time(&group.items);
    if clocked && !group.filters.motion {
        return (None, None);
    }
    let mut h = FxHasher::default();
    h.write(b"layer-capture");
    if clocked {
        h.write_u64(opts.shader_time_s.to_bits());
    }
    // Paint-layer pixels (raster spike): any stroke recaptures (a per-layer epoch of the captured ids is the finer key).
    h.write_u64(crate::surface::epoch_total());
    for id in ids {
        h.write(id.as_bytes());
        h.write_u8(0xff);
        match hashes.get(id) {
            Some(nh) => h.write_u64(nh.full),
            None => match scene.effective_node(id) {
                Some(n) => h.write_u64(node_hashes_in(scene, n).full),
                None => h.write_u8(1),
            },
        }
        if let Some(a) = scene.node_canvas_affine(id) {
            for v in [a.a, a.b, a.c, a.d, a.e, a.f] {
                h.write_u64(v.to_bits());
            }
        }
        h.write_u64(scene.ancestor_opacity_product(id).to_bits());
        match crate::effects::effective_page_clip(scene, id, page_clip) {
            Some(c) => c.iter().for_each(|v| h.write_u32(v.to_bits())),
            None => h.write_u8(2),
        }
        // The rounded clip of the innermost rounded clipping frame above: a
        // subtree's descendants (and nested composites) are cut by it inside
        // the capture.
        fold_clip_rrect(&mut h, crate::effects::clip_rounded_rect(scene, id));
    }
    // The identity: what the capture draws, without its geometry (region, size,
    // filter px) or the zoom band.
    let mut identity = h.clone();
    identity.write_u64(opts.scale.to_bits());
    identity.write_u64(moda_assets::engine_seam::store_generation());
    identity.write_u64(crate::text_draw::text_host_generation());
    fn fold_group(h: &mut FxHasher, g: &crate::seams::LayerBlurGroup) {
        for v in g.region_canvas {
            h.write_u64(v.to_bits());
        }
        h.write_u32(g.width_px);
        h.write_u32(g.height_px);
        // Includes the kind and shadow blend, as well as filter parameters.
        h.write(format!("{:?}", g.filters).as_bytes());
        // Every descendant capture's geometry and clips affect these pixels. A descendant that has its own
        // content key folds that key: it covers the descendant's filters, geometry and items. A capture can hold
        // a copy of items that include captures of copies (a filter shadow's capture of a node whose child has a
        // filter shadow), so walking into each one would visit a chain of n of them 2^(n-1) times.
        for item in &g.items {
            if let crate::seams::DrawItem::LayerBlurComposite {
                group: nested, page_clip_bounds, clip_rrect, ..
            } = item {
                match nested.cache_key {
                    Some(key) => h.write_u64(key),
                    None => fold_group(h, nested),
                }
                page_clip_bounds.iter().for_each(|v| h.write_u32(v.to_bits()));
                fold_clip_rrect(h, *clip_rrect);
            }
        }
    }
    fold_group(&mut h, group);
    // What the captured items draw with: a live frame may plan them from an outline, curve or strand body it is
    // still refining (the previous band's, or a placeholder) and refine them on later frames; the nodes' hashes
    // don't change then, but these do, so a capture of the stand-in isn't served for the refined items.
    fold_item_geometry(&mut h, &group.items);
    fold_render_options_fingerprint(&mut h, opts);
    h.write_i32(crate::tess::outline::current_band());
    h.write_u64(moda_assets::engine_seam::store_generation());
    h.write_u64(crate::text_draw::text_host_generation());
    (Some(h.finish()), Some(identity.finish()))
}

/// The geometry a capture's items draw with, into its content key: each body's tessellation (its `Arc`, vertex
/// count, bounds and a sample of its positions: the same tessellation is the same `Arc` while a cache holds it, and
/// the sample keeps a reused address from passing for another) and each text batch, nested captures' items
/// included. A text batch is named by what it draws: its content key (`TextQuadBatch::content_key`, the key its
/// plan cache keeps it under) and its placement (z, model, clip). A cache hit is a copy in a new allocation, so
/// its address says nothing: named by address, an unchanged text looked new on every rebuilt plan and a capture
/// holding it was rendered again each time. A batch with no content key (built outside a plan cache) has only
/// its address.
fn fold_item_geometry(h: &mut FxHasher, items: &[crate::seams::DrawItem]) {
    for item in items {
        match item {
            crate::seams::DrawItem::Body { geometry, .. } => {
                h.write_usize(Arc::as_ptr(geometry) as usize);
                h.write_u32(geometry.vertex_count);
                geometry.bounds.iter().for_each(|v| h.write_u32(v.to_bits()));
                let p = &geometry.positions;
                for i in [0, p.len() / 3, p.len() / 2, p.len().saturating_sub(1)] {
                    if let Some(v) = p.get(i) {
                        h.write_u32(v.to_bits());
                    }
                }
            }
            crate::seams::DrawItem::Text(batch) => match batch.content_key {
                Some(key) => {
                    h.write_u64(key);
                    h.write_u64(batch.z.to_bits());
                    batch.model_matrix.iter().chain(&batch.page_clip_bounds).for_each(|v| h.write_u32(v.to_bits()));
                }
                None => h.write_usize(Arc::as_ptr(batch) as usize),
            },
            // A nested capture's own content key already holds its items' geometry (see `fold_group`).
            crate::seams::DrawItem::LayerBlurComposite { group, .. } => match group.cache_key {
                Some(key) => h.write_u64(key),
                None => fold_item_geometry(h, &group.items),
            },
            _ => {}
        }
    }
}

/// A capture composite's rounded clip (`seams::ClipRRect`) into a content key.
fn fold_clip_rrect(h: &mut FxHasher, rr: Option<crate::seams::ClipRRect>) {
    match rr {
        Some(r) => {
            h.write_u8(1);
            r.rect
                .iter()
                .chain(r.radii.iter())
                .for_each(|v| h.write_u32(v.to_bits()));
        }
        None => h.write_u8(0),
    }
}

/// Whether a built plan bakes the shader clock anywhere (R5 hit condition).
pub fn plan_uses_shader_time(plan: &CoveragePlan) -> bool {
    items_use_shader_time(&plan.items)
        || plan.masks.iter().any(|mask| mask.image.as_ref().is_some_and(|image| image.rendered.is_some()))
}

/// rust-edit-incremental-z4: whether any of these draw items bakes the shader
/// clock (procedural fill / shader-filled text). Used both for the whole-plan
/// check and — applied to just the freshly-rebuilt items (O(k)) — to detect a
/// patch that introduces ANY shader into the slot, so the incremental slot is
/// conservatively dropped (served once, rebuilt next frame) rather than risk
/// serving a patched-in shader whose clock-rethreadability was not re-verified
/// per-patch. This patch-time guard applies to BOTH lanes and is intentionally
/// broad even though the CAMERA seed gate may admit re-threadable-shader
/// plans (the PAGE seed gate stays non-shader-only).
pub(crate) fn items_use_shader_time(items: &[crate::seams::DrawItem]) -> bool {
    use crate::seams::{BodyPaint, DrawItem};
    items.iter().any(|item| match item {
        DrawItem::Body { paint, .. } => matches!(paint, BodyPaint::Shader(_)),
        // vork: text shader fills are not ported; a text batch reads no clock.
        DrawItem::Text(_) => false,
        // The divert moves a blurred node's shader-fill body INTO
        // the capture group — without recursing, a page whose only clock
        // consumer is a blurred shader node would cache as clock-free and
        // freeze the shader animation on LRU hits.
        DrawItem::LayerBlurComposite { group, .. } => {
            // An agent shader layer effect bakes the clock into its slots (`time()`).
            // A capture that has a content key and is not a motion capture was found clock-free when the key was
            // made (`capture_key`), so its items are not walked again: a capture can hold copies of captures
            // (`fold_group`), and walking each would visit a chain of n of them 2^(n-1) times.
            !group.filters.shaders.is_empty()
                || ((group.cache_key.is_none() || group.filters.motion) && items_use_shader_time(&group.items))
        }
        _ => false,
    })
}

/// Whether EVERY shader-clock consumer in this plan can be re-threaded
/// at present (`render_plan_to_texture_impl`) — i.e. each is a procedural-fill
/// body whose clock is a direct-copy `shader_params` slot (`clock_offsets:
/// Some`). True lets `camera_get` drop the clock from its hit key (an animating
/// shader reuses the cached plan; the live clock rides the present override).
///
/// Returns false when a consumer is NOT re-threaded by the present path:
/// * an MSDF/vector **text** shader fill (the text executor packs its own
///   instance from the batch's baked clock — not patched here), or
/// * a procedural fill whose clock is not a clean direct copy
///   (`clock_offsets: None`, a manifest-drift guard).
/// Such a plan keeps the exact-clock-match cache behavior (rebuild per clock) so
/// the shader animates correctly instead of freezing.
///
/// Vacuously true for a plan with no shader fills; pair with `plan_uses_shader_time`.
pub fn plan_shader_time_rethreadable(plan: &CoveragePlan) -> bool {
    items_shader_time_rethreadable(&plan.items)
}

/// [`plan_shader_time_rethreadable`] for some of a plan's items (a retained slot's re-planned units).
pub(crate) fn items_shader_time_rethreadable(items: &[crate::seams::DrawItem]) -> bool {
    use crate::seams::{BodyPaint, DrawItem};
    items.iter().all(|item| match item {
        DrawItem::Body {
            paint: BodyPaint::Shader(body),
            ..
        } => body.clock_offsets.is_some(),
        // vork: text shader fills are not ported.
        DrawItem::Text(_) => true,
        // A shader fill diverted into a layer-blur capture is NOT
        // re-threaded by the present path (the capture mini-render packs its
        // instances from the plan's baked clock), so such a plan must keep the
        // exact-clock-match cache behavior (and its capture takes no content
        // key, `capture_key`, so it re-renders every frame).
        DrawItem::LayerBlurComposite { group, .. } => group.filters.shaders.is_empty() && !items_use_shader_time(&group.items),
        _ => true,
    })
}

// ---------------------------------------------------------------------------
// R5 — page fingerprint
// ---------------------------------------------------------------------------

/// Fold the plan-relevant `RenderOptions` fields (everything except the shader
/// clock, which the `uses_shader_time` hit condition handles) into a
/// fingerprint. Shared by `page_fingerprint` and the canvas-root fold so an
/// option change re-fingerprints both — including a viewport over ONLY floating
/// content, where no `page_fingerprint` runs.
fn fold_render_options_fingerprint(h: &mut FxHasher, opts: &RenderOptions) {
    h.write_u64(opts.scale.to_bits());
    // Brush plan P8: inside a camera build of a scene with outlined strokes,
    // the outline zoom band is plan content (outlines are refit per band).
    // Absent otherwise, so other scenes keep zoom-invariant camera plans.
    if let Some(band) = crate::tess::outline::camera_key_band() {
        h.write(b"outline-band");
        h.write_i32(band);
    }
    h.write_u8(opts.transparent as u8);
    h.write_u8(opts.preserve_page_fill as u8);
    h.write_u32(opts.sample_count);
    h.write_u8(opts.best_effort_text as u8);
    h.write_u8(match opts.text_masks_available {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    });
    // A plan for an engine in a working colour plans its luminance masks differently.
    h.write_u8(opts.working.is_some() as u8);
    for e in &opts.exclude {
        h.write(e.as_bytes());
    }
    // `isolate` is plan-relevant exactly as `exclude` is: two renders of the
    // same scene isolated to different elements produce different plans, so a
    // cached plan must not be served across them. The list length separates it
    // from an `exclude` list of the same ids, and each id is length-prefixed in
    // turn — concatenating ids without a delimiter makes `["ab", "c"]` and
    // `["a", "bc"]` fold identically, which would serve one isolation's cached
    // plan for another's render (the silently-wrong-plate outcome).
    h.write_usize(opts.isolate.len());
    for i in &opts.isolate {
        h.write_usize(i.len());
        h.write(i.as_bytes());
    }
}

/// Fold a display list (z-order × per-node full hashes, plus blend source
/// hashes) into a fingerprint hasher. Shared by `page_fingerprint` and the
/// freeform canvas-root fold in `camera_fingerprint` so both detect
/// the same node-level edits.
fn fold_display_list_fingerprint(
    h: &mut FxHasher,
    scene: &Scene,
    dl: &moda_scene::DisplayList,
    hashes: &HashMap<String, NodeHashes>,
) {
    h.write_usize(dl.items.len());
    for item in &dl.items {
        h.write(item.node_id.as_bytes());
        h.write_u8(item.is_primary_node_item as u8);
        let nh = hashes
            .get(&item.node_id)
            .copied()
            .or_else(|| scene.effective_node(&item.node_id).map(node_hashes));
        h.write_u64(nh.map(|x| x.full).unwrap_or(0));
        // A transient child of a `native-children` generated host
        // (a chart's expanded shapes / placeholder) is emitted in host-LOCAL
        // coordinates — the host's own x/y/rotation/scale reach the child's
        // draw-item model matrix via `transform_parent_chain`, NOT the child's
        // node hash. The host is expanded away (never its own DL item), so a
        // host MOVE leaves every child hash unchanged and, without this fold,
        // the page/camera fingerprint is identical → the retained plan is reused
        // with stale chart pixels. Fold the host's full hash so a host transform
        // edit re-fingerprints. (raster-/vector-asset proxies are unaffected:
        // they are their OWN DL items and bake the host x/y into their OWN
        // coords, so their hash already moves with the host; any group/container
        // ancestor move is caught via that ancestor's own DL item. Only
        // `native-children` hosts are transform parents per
        // `transform_parent_chain`, and only they are expanded away.)
        //
        // UPDATE: transient hosts DO now reach the incremental patcher
        // (`apply_props` re-expands them in place instead of bailing), and the
        // note above once warned that would need "its own fix here". It
        // does not, for two independent reasons — both worth stating, because the
        // absence of a change here is easy to misread as an oversight:
        //   1. `reexpand_host_in_place` re-hashes the HOST as well as its
        //      transients, so the fold below still sees a moved host's new hash.
        //      The fold is correct either way, patched or rebuilt.
        //   2. The two paths are not even in competition. The incremental patcher
        //      keys on `dirty_since`, NOT on the fingerprint; a page/camera
        //      fingerprint is only computed on the route the patcher DECLINED,
        //      which is a full rebuild anyway. So this fold can never be the thing
        //      that serves a stale patched plan.
        // It remains the guard for the full-rebuild route (a host re-fed via
        // `update`, or a structural re-expansion), so it stays exactly as is.
        if let Some(host_id) = scene.transient_parent_of().get(&item.node_id) {
            if let Some(host) = scene.effective_node(host_id) {
                if host.node_type() == "generator"
                    || host.node_type() == "generated"
                        && host.str("renderMode") == Some("native-children")
                {
                    // `host` is in hand, so the hash is infallible: prefer the
                    // precomputed map entry, else compute it directly.
                    let hh = hashes
                        .get(host_id)
                        .copied()
                        .unwrap_or_else(|| node_hashes(host));
                    h.write_u64(hh.full);
                }
            }
        }
        // Blend nodes draw OTHER nodes' geometry (possibly cross-page / hidden, so
        // not otherwise in this DL): fold their objects' and spine's full hashes
        // (`moda_scene::blend::dependency_ids`, what `expand_blend_node` reads) so an
        // edit to one re-fingerprints (else a cross-page source edit can leave the
        // fingerprint unchanged → stale hit).
        if let Some(node) = scene.effective_node(&item.node_id) {
            if node.node_type() == "blend" {
                for id in moda_scene::blend::dependency_ids(node) {
                    let sh = hashes
                        .get(id)
                        .copied()
                        .or_else(|| scene.effective_node(id).map(node_hashes));
                    h.write_u64(sh.map(|x| x.full).unwrap_or(0));
                }
            }
        }
    }
}

/// Fingerprint of everything (except the shader clock — handled by the
/// `uses_shader_time` hit condition) that feeds `build_coverage_plan` for one
/// page: the display list in z-order × per-node full hashes (blend nodes fold
/// their cross-page source hashes), page geometry/props, and the
/// plan-relevant `RenderOptions` fields.
pub fn page_fingerprint(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
    hashes: &HashMap<String, NodeHashes>,
) -> u64 {
    let Some(page) = scene.doc().pages.get(page_index) else {
        return 0;
    };
    // Display list in z-order × node hashes. The DL itself is rebuilt per
    // frame (cheap); the fingerprint guards the EXPENSIVE plan build.
    let dl = moda_scene::build_display_list(scene, page.id());
    page_fingerprint_with_dl(scene, page_index, opts, hashes, &dl)
}

/// renderer-perf-more-z6: [`page_fingerprint`] computed from an ALREADY-BUILT
/// display list. The DL is a pure fn of (scene, page) and the fold order here is
/// identical to `page_fingerprint`'s, so the fingerprint is byte-identical — but
/// the cold/structural-rebuild path can now build the DL once (for both the
/// fingerprint AND the plan build) instead of twice (~21 ms@10k of waste/edit).
pub fn page_fingerprint_with_dl(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
    hashes: &HashMap<String, NodeHashes>,
    dl: &moda_scene::DisplayList,
) -> u64 {
    let mut h = FxHasher::default();
    let Some(page) = scene.doc().pages.get(page_index) else {
        return 0;
    };
    // Page raw value (size, fill, name-level props) + position (camera/clip
    // inputs depend on neighbouring pages via the layout walk).
    hash_value(&Value::Object(page.0.clone()), &mut h);
    let (px, py) = scene.page_position(page_index);
    h.write_u64(px.to_bits());
    h.write_u64(py.to_bits());
    h.write_u8(scene.doc().config.render_page_clipping as u8);
    // Plan-relevant options.
    fold_render_options_fingerprint(&mut h, opts);
    fold_display_list_fingerprint(&mut h, scene, dl, hashes);
    // Asset/font residency: pattern texture settle states feed pattern
    // instances at plan-build time, so a settle flip must re-fingerprint.
    h.write_u64(moda_assets::engine_seam::store_generation());
    h.write_u64(crate::text_draw::text_host_generation());
    h.finish()
}

// ---------------------------------------------------------------------------
// Retained plan building (the wasm render path + resident_smoke test path)
// ---------------------------------------------------------------------------

/// Build (or reuse) the coverage plan for one page through every Stage-1
/// cache layer. The caller must have `resident.update(json)`-ed first.
pub fn build_plan_retained(
    resident: &ResidentScene,
    caches: &mut PlanCaches,
    page_index: usize,
    opts: &RenderOptions,
) -> anyhow::Result<(Arc<CoveragePlan>, Arc<PlanWarnings>)> {
    let scene = resident
        .scene()
        .ok_or_else(|| anyhow::anyhow!("no resident scene (update first)"))?;
    let hashes = resident.hashes().expect("hashes exist with scene");
    let page_id = scene
        .doc()
        .pages
        .get(page_index)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "page index {page_index} out of range ({} pages)",
                scene.doc().pages.len()
            )
        })?
        .id()
        .to_string();
    // rust-edit-incremental-z4: INCREMENTAL FAST PATH. When the live editor slot
    // holds this page's plan + index and the change since it was built is a small,
    // structure-preserving property edit, patch only the affected nodes' DrawItems
    // in place (O(dirty)) and return — skipping the O(n) display-list build +
    // node-hash fold + plan assembly the fallback below does. Any doubt → fall
    // through to the byte-identical full rebuild (which re-seeds the index).
    if !caches.incremental_disabled {
        if let Some(out) = try_incremental_fast_path(resident, caches, page_index, &page_id, opts) {
            debug_verify_incremental(scene, page_index, opts, hashes, &out.0);
            return Ok(out);
        }
    }
    // E3 (rust-render-perf-g8) fingerprint memo: skip the O(n) `page_fingerprint`
    // (display-list rebuild + node-hash fold) when the cheap generation key is
    // unchanged — the page-path twin of the camera memo above. Same kill switch
    // (`fingerprint_memo_disabled`). Conservative superset ⇒ an unchanged key
    // guarantees an unchanged fingerprint; a false miss only recomputes (safe).
    // The reused fingerprint still flows through `page_get`, which re-checks
    // `shader_time`.
    // renderer-perf-more-z6: build the display list AT MOST ONCE on the cold /
    // structural-rebuild path. Both the fingerprint and the full plan build need
    // it; building it twice (`page_fingerprint` + `build_coverage_plan_inner`) was
    // ~21 ms@10k of pure waste on every cold / resync / structural-fallback
    // rebuild. The DL is a pure fn of (scene, page), so sharing it is byte-
    // identical. `shared_dl` is `Some` only when we built it for the fingerprint
    // (the memo-miss path); on a memo HIT we built no DL, so the plan builder
    // builds it once itself — no double build on any path.
    let mut shared_dl: Option<moda_scene::DisplayList> = None;
    let fingerprint = if caches.fingerprint_memo_disabled {
        let dl = moda_scene::build_display_list(scene, &page_id);
        let fp = page_fingerprint_with_dl(scene, page_index, opts, hashes, &dl);
        shared_dl = Some(dl);
        fp
    } else {
        let frame_key = page_frame_key(resident.generation(), page_index, opts);
        match caches.page_frame_key.get(&page_id) {
            // Hit: reuse the stored fingerprint — no clone, no re-insert on the
            // hot static-playback path (the stored (key, fp) is already current).
            Some(&(k, fp)) if k == frame_key => {
                caches.page_fp_skips = caches.page_fp_skips.wrapping_add(1);
                fp
            }
            // Miss: recompute and store (key changed or first frame for this page).
            _ => {
                let dl = moda_scene::build_display_list(scene, &page_id);
                let fp = page_fingerprint_with_dl(scene, page_index, opts, hashes, &dl);
                caches
                    .page_frame_key
                    .insert(page_id.clone(), (frame_key, fp));
                shared_dl = Some(dl);
                fp
            }
        }
    };
    if let Some(hit) = caches.page_get(&page_id, fingerprint, opts.shader_time_s) {
        // rust-edit-incremental-z4: a fingerprint hit means the cached plan is
        // already current; if the live editor slot doesn't already hold it (e.g.
        // first edit after a non-editor render seeded only the LRU cache), leave it
        // — the next full rebuild will seed the index. Returning the hit is correct
        // regardless (the index is an accelerator, never a source of truth).
        // (A `shared_dl` we may have built for the fingerprint is dropped here; it
        // was needed for the fingerprint regardless, so this costs nothing extra.)
        return Ok(hit);
    }
    // rust-edit-incremental-z4: full rebuild — use the INDEXED builder so we capture
    // the per-node spans + page constants and (re)seed the incremental editor slot.
    // Skip seeding for shader-animated plans (clock advances by opts, not the
    // resident generation, so they'd churn the index every frame for no edit win).
    // renderer-perf-more-z6: hand it the DL we already built for the fingerprint.
    let (plan, warnings, aux) = crate::frame::build_coverage_plan_indexed_with_dl(
        scene,
        page_index,
        opts,
        Some((caches, hashes)),
        shared_dl,
    )?;
    let plan = Arc::new(plan);
    let warnings = Arc::new(warnings);
    caches.page_put(
        &page_id,
        fingerprint,
        opts.shader_time_s,
        plan.clone(),
        warnings.clone(),
    );
    // rust-edit-incremental-z4: only seed the incremental slot for a static,
    // backdrop-free plan. Shader-animated plans advance by opts (not the resident
    // generation) so they'd churn the index every frame; backdrop pages need a
    // per-node `PlanCtx` to emit `BackdropSplit`s (which `rebuild_node_items`
    // deliberately does not thread) and carry page-level sample_count/transparent
    // state a single-node patch can't recompute — so they always full-rebuild.
    // The PAGE lane stays non-shader-only (unlike the CAMERA lane, which
    // now seeds re-threadable shaders). The page present path
    // (`present_page`/`present_page_to_canvas`/`render_page`) NEVER passes a present
    // override (`present_plan(.., None, None)` / `render_plan_to_texture(&plan)`),
    // so it can't re-thread a cached clock — seeding a shader plan here would freeze
    // the shader at the seed clock during slideshow/page playback. (`page_get`
    // likewise has no re-threadable bypass.)
    // rust-edit-incremental-z4: gate seeding on the SAME byte budget the LRU page
    // cache enforces, so the slot never retains an over-budget plan that `page_put`
    // itself would have evicted (else the big plan stays alive invisibly to the
    // memory watermark). `plan_bytes` is the marginal heap of the retained plan.
    let page_slot_bytes = plan_bytes(&plan);
    if !caches.incremental_disabled
        && !plan_uses_shader_time(&plan)
        && !aux.has_backdrop
        // A layer-blur page's diverted spans aren't patchable in
        // isolation (and its plan is camera/scale-dependent) — never seed.
        && !aux.has_layer_blur
        // A group painting its appearance (E8.2) draws an outline derived from its descendants, outside every node
        // span: an edit inside it can't be patched in isolation — never seed.
        && !page_has_group_appearance(scene, &page_id)
        && page_slot_bytes <= PAGE_PLAN_BUDGET_BYTES
    {
        let index = build_plan_index(
            scene,
            page_index,
            opts,
            aux,
            resident.generation(),
            &mut caches.reverse_deps,
        );
        caches.incremental_page = Some(IncrementalEntry {
            page_id: page_id.clone(),
            plan: plan.clone(),
            warnings: warnings.clone(),
            index,
        });
        caches.incr_page_bytes = page_slot_bytes;
    } else {
        // A shader-animated / backdrop / over-budget / kill-switched rebuild
        // invalidates any stale slot.
        caches.incremental_page = None;
        caches.incr_page_bytes = 0;
    }
    Ok((plan, warnings))
}

/// The plan of a host-owned offscreen scene, with its index. The index owns the plan: the host plans through
/// it, and between frames nothing else holds the plan, so a placement or a node prepared again writes the plan's
/// items in place (`Arc::make_mut`: a caller that still holds an earlier state keeps it, and that write copies).
/// A plan over the whole-frame byte budget is not held and is planned again each frame.
#[derive(Default)]
pub struct FramePlanIndex {
    entry: Option<FrameIndexEntry>,
    /// Scratch kept between placements, so a placement allocates for what moves and not for the plan: the moved
    /// rows' ranges.
    ranges: Vec<(u32, u32)>,
    /// What the last `place` or `patch` wrote into the plan ([`Self::take_written`]).
    written: PlanWritten,
}

/// The items a plan's owner wrote in place, by item index, between two states of the plan (the engine's
/// `frame::ChangedItems::Written`).
#[derive(Default)]
pub struct PlanWritten {
    /// Bodies whose placement values were written by their recipe, ascending: their draw stands.
    pub placed: Vec<u32>,
    /// Items the planner prepared again, each with the item it replaced.
    pub replaced: Vec<(u32, crate::seams::DrawItem)>,
}

struct FrameIndexEntry {
    plan: Arc<CoveragePlan>,
    warnings: Arc<PlanWarnings>,
    /// The host's key for the scene state the plan stands at, and the shader clock it was planned for.
    key: u64,
    shader_time_s: f64,
    uses_shader_time: bool,
    bytes: usize,
    /// What following a change in place needs; `None` for a plan that is only held (a plan that reads the
    /// shader clock, or incremental planning switched off).
    patchable: Option<Patchable>,
}

struct Patchable {
    index: PlanIndex,
    surface_epoch: u64,
    /// `Some` for a page with captures, cut-outs or groups painting their appearance. Such a unit bakes its
    /// content's placement, so it and everything under it is planned whole; every other plain span still
    /// follows its recipe (`place`) or is prepared node by node (`patch`).
    units: Option<Units>,
    /// The plan's spans by scene row, made with the plan. `None` for a plan that is not placed by rows.
    rows: Option<PlacedRows>,
}

/// A plan's prepared items by scene row: which span a row's node owns, and what the scene says about following
/// a placement by recipe. It is made with the plan (`FramePlanIndex::seed`) and holds only what stands for the
/// plan's life: a node's span and its place in the scene do not change while the planner prepares nodes again
/// in place (a patch keeps every span's items the same shape, for the same node), and a change of structure
/// plans whole and seeds again. Whether a span holds a recipe right now is not kept here; it is read from the
/// span. Everything a placement would ask the scene by name is asked here once, so a placement frame works on row
/// and span indices alone.
struct PlacedRows {
    /// The scene's hierarchy by row these were read under (`Worlds::hierarchy`); a placement under another one
    /// is declined.
    hierarchy: u64,
    /// Per row, the span its node owns, or `u32::MAX`.
    span: Vec<u32>,
    /// Per span, the row of its node; `u32::MAX` when the span must not follow a placement by recipe (a blend or
    /// mask source or something under one, a warned node, a span inside a unit, a span that does not hold only
    /// its node's items) or its node is not a row.
    row: Vec<u32>,
    /// Per row: moving the node is planned whole. It owns a unit, sits above one or inside one; or the build
    /// culled it or a node above it, so it has no span and a move may bring it into the window.
    unit: Vec<bool>,
    /// Where each culled node sits in the scene's parent-before-child order, ascending: a moved row whose range
    /// holds one is planned whole (the culled node under it may move into the window).
    culled: Vec<u32>,
}

impl PlacedRows {
    fn read(scene: &Scene, index: &PlanIndex, units: Option<&Units>, warnings: &PlanWarnings, plan: &CoveragePlan) -> Option<Self> {
        // The scene's rows, when its owner keeps the world columns current (`Scene::refresh_worlds`).
        let worlds = scene.world_rows()?;
        let rows = worlds.len();
        // What the build culled, by row: the culled node and everything under it must not move by recipe, and
        // neither must anything above it. Rows beside it are free. A culled node that is not a row (an
        // expansion's part) is reached by no row's range: such a plan is not placed by rows.
        let mut unit = vec![false; rows];
        let mut culled = Vec::with_capacity(index.culled_roots.len());
        for id in &index.culled_roots {
            let (from, to) = worlds.range(scene.row_of(id)?)?;
            culled.push(from);
            for &under in worlds.ordered(from, to) { *unit.get_mut(under as usize)? = true; }
        }
        culled.sort_unstable();
        let mut span = vec![u32::MAX; rows];
        let mut row_of_span = vec![u32::MAX; index.spans.len()];
        let mut restriction_memo = HashMap::default();
        let warned = !warnings.skipped.is_empty() || !warnings.dropped.is_empty();
        for (si, entry) in index.spans.iter().enumerate() {
            let id = entry.node_id.as_str();
            // A span of something that is not a row (a table's cell) is reached by no row's subtree.
            let Some(row) = scene.row_of(id) else { continue };
            if let Some(slot) = span.get_mut(row as usize) { *slot = si as u32; }
            let Some(items) = plan.items.get(entry.start..entry.start + entry.len) else { continue };
            let follows = !index.blend_related_ids.contains(id) && !index.mask_source_ids.contains(id)
                && !has_blend_or_mask_ancestor(scene, id, index, &mut restriction_memo)
                && !(warned && (warnings_mention_node(&warnings.skipped, id) || warnings_mention_node(&warnings.dropped, id)))
                // The span names this node's own items (a unit's cut-out shifts the items inside it).
                && items.iter().all(|item| drawitem_node_id(item) == Some(id))
                && units.is_none_or(|units| !units.holds(scene, id));
            if follows { row_of_span[si] = row; }
        }
        if let Some(units) = units {
            for (row, slot) in unit.iter_mut().enumerate() {
                let id = scene.row(row as u32)?.id();
                // A unit that moves, sits under a node that moves, or holds one, is planned again whole. A moved
                // node is asked directly: one that draws nothing yet has no span to ask.
                *slot |= units.above.contains(id) || units.holds(scene, id);
            }
        }
        Some(Self { hierarchy: worlds.hierarchy(), span, row: row_of_span, unit, culled })
    }
}

/// The nodes of a page that own a unit, from the scene rather than from what the plan emitted: a unit that
/// draws nothing yet (a background effect outside the target) still splits the plan once it moves into view.
struct Units {
    /// A fading group, a blurred, shadowed or mask-captured node, a background effect, a split blend, a group
    /// painting its appearance.
    owners: std::collections::HashSet<String>,
    /// Every node above an owner: moving one moves the unit.
    above: std::collections::HashSet<String>,
}

impl Units {
    fn of(scene: &Scene, plan: &CoveragePlan) -> Self {
        use crate::seams::DrawItem;
        let mut owners: std::collections::HashSet<String> = plan.items.iter().filter_map(|item| match item {
            DrawItem::BackdropSplit(split) => Some(split.node_id.clone()),
            DrawItem::LayerBlurComposite { node_id, .. } => Some(node_id.clone()),
            _ => None,
        }).chain(scene.group_appearance_by_id().keys().cloned()).collect();
        for id in scene.flattened_z_order_for_page(&plan.page_id) {
            if owners.contains(id.as_str()) { continue; }
            let Some(node) = scene.effective_node(&id) else { continue };
            // A split blend depends on the target's alpha mode; either answer makes the node an owner.
            if crate::backdrop::plan::has_active_background_blur(node) || crate::layer_blur::plan::needs_layer_capture_in(scene, node)
                || crate::backdrop::blend::node_splits(scene, &id, false) || crate::backdrop::blend::node_splits(scene, &id, true) {
                owners.insert(id.to_string());
            }
        }
        let mut above = std::collections::HashSet::new();
        for owner in &owners {
            let mut cursor = owner.as_str();
            // Stop at a chain another owner already walked.
            while let Some(parent) = scene.parent_map().get(cursor).or_else(|| scene.transient_parent_of().get(cursor)) {
                if !above.insert(parent.to_string()) { break; }
                cursor = parent;
            }
        }
        Self { owners, above }
    }

    /// Whether `id` is a unit owner or sits under one.
    fn holds(&self, scene: &Scene, id: &str) -> bool {
        let mut cursor = id;
        // A chain longer than any real hierarchy counts as inside one.
        for _ in 0..4096 {
            if self.owners.contains(cursor) { return true; }
            match scene.parent_map().get(cursor).or_else(|| scene.transient_parent_of().get(cursor)) {
                Some(parent) => cursor = parent,
                None => return false,
            }
        }
        true
    }
}

impl FramePlanIndex {
    /// Take `plan`, made whole of the scene state `key` names, as the scene's plan. Returns the plan it takes the
    /// place of, when one was held: the host hands it to the engine, which compares the two.
    #[allow(clippy::too_many_arguments)]
    pub fn seed(&mut self, scene: &Scene, opts: &RenderOptions,
        plan: &Arc<CoveragePlan>, warnings: &Arc<PlanWarnings>, aux: crate::frame::PlanBuildAux,
        generation: u64, key: u64, caches: &mut PlanCaches) -> Option<Arc<CoveragePlan>> {
        let replaced = self.entry.take().map(|entry| entry.plan);
        self.written = PlanWritten::default();
        let bytes = plan_bytes(plan);
        // The same bound the whole-frame slot keeps: a plan past it is not held.
        if bytes > CAMERA_PLAN_BUDGET_BYTES { return replaced; }
        let uses_shader_time = plan_uses_shader_time(plan);
        let patchable = (!caches.incremental_disabled && !uses_shader_time).then(|| {
            // One blur or fading group no longer turns the index off for the canvas: it becomes a unit.
            let units = (aux.has_backdrop || aux.has_layer_blur || page_has_group_appearance(scene, &plan.page_id))
                .then(|| Units::of(scene, plan));
            let index = build_plan_index(scene, plan.page_index, opts, aux, generation, &mut caches.reverse_deps);
            let rows = PlacedRows::read(scene, &index, units.as_ref(), warnings, plan);
            Patchable { units, rows, index, surface_epoch: crate::surface::epoch_total() }
        });
        self.entry = Some(FrameIndexEntry {
            plan: plan.clone(), warnings: warnings.clone(), key, shader_time_s: opts.shader_time_s,
            uses_shader_time, bytes, patchable,
        });
        replaced
    }

    /// The plan held for the scene state `key` names, when nothing was written since it was made or last
    /// brought there. An offscreen frame has no live clock to thread into a plan, so a plan that reads the
    /// shader clock is the plan of its exact clock only.
    pub fn held(&self, key: u64, shader_time_s: f64) -> Option<(Arc<CoveragePlan>, Arc<PlanWarnings>)> {
        let entry = self.entry.as_ref()?;
        (entry.key == key && (!entry.uses_shader_time || entry.shader_time_s == shader_time_s))
            .then(|| (entry.plan.clone(), entry.warnings.clone()))
    }

    /// Whether `plan` is the plan held here. A host must not keep a command packet for a plan that is not (one
    /// past the byte budget is planned again each frame).
    pub fn holds(&self, plan: &Arc<CoveragePlan>) -> bool {
        self.entry.as_ref().is_some_and(|entry| Arc::ptr_eq(&entry.plan, plan))
    }

    /// Follow placement writes without preparing anything: `rows` is the complete batch of scene rows
    /// (`Scene::row_of`; the scene's placement dirty list) whose own transform changed since the last plan, or
    /// (for a node that draws a body itself) whose opacity did. Every item-producing node under them must hold a
    /// placement recipe that admits its new placement (`frame::placed_span_row`); then only those items' matrices
    /// and solid alphas are written, in the plan held here, which then stands at `key`. Otherwise nothing
    /// changes, the plan and index stay, and `None` is returned for the caller to try [`Self::patch`].
    ///
    /// Nothing is looked up by name, and the work follows the rows that move: each moved row is one range of the
    /// scene's parent-before-child order, the ranges of a batch are visited as their union (a moved node under a
    /// moved node is inside its range, not walked again), and each matrix is read from the scene's world column.
    /// The scene's owner keeps that column current (`Scene::refresh_worlds`); a batch the rows do not answer (the
    /// column is not current, something under a moved row is not a row, a derived record stands in for a node)
    /// is declined like any other.
    ///
    /// The caller proves the batch holds nothing else: no content, paint, effect, visibility or group-opacity
    /// change, and no node whose placement feeds another node's fields. Node hashes are not read.
    #[allow(clippy::too_many_arguments)]
    pub fn place(&mut self, scene: &Scene, opts: &RenderOptions, generation: u64, key: u64, rows: &[u32],
        caches: &mut PlanCaches,
    ) -> Option<(Arc<CoveragePlan>, Arc<PlanWarnings>)> {
        self.written = PlanWritten::default();
        let entry = self.entry.as_mut()?;
        let patchable = entry.patchable.as_mut()?;
        if patchable.index.opts_key != render_options_key(opts)
            || patchable.index.resource_key != resource_key()
            || patchable.surface_epoch != crate::surface::epoch_total() { return None; }
        let read = patchable.rows.as_ref()?;
        let worlds = scene.world_rows()?;
        // The rows were read under the structure the plan was made of; a change of structure plans whole.
        if read.hierarchy != worlds.hierarchy() { return None; }
        let index = &mut patchable.index;
        // Each moved row's range, after asking the moved row itself what only it can answer.
        let ranges = &mut self.ranges;
        ranges.clear();
        for &row in rows {
            if *read.unit.get(row as usize)? { return None; }
            // A node that draws itself and has no span may have just started to draw. A node that holds
            // children need not draw: a move of it with nothing drawn under it changes nothing.
            if !worlds.is_parent(row) && *read.span.get(row as usize)? == u32::MAX { return None; }
            // Something under it that is not a row is not reached by rows.
            if !worlds.whole(row) { return None; }
            let (from, to) = worlds.range(row)?;
            // A culled node under it may move into the window.
            let next = read.culled.partition_point(|&at| at < from);
            if read.culled.get(next).is_some_and(|&at| at < to) { return None; }
            ranges.push((from, to));
        }
        // The union of the ranges: a range starts inside an earlier one only when it lies wholly inside it.
        ranges.sort_unstable();
        let mut affected: Vec<u32> = Vec::with_capacity(rows.len());
        let mut covered = 0u32;
        for &(from, to) in ranges.iter() {
            if to <= covered { continue; }
            for &under in worlds.ordered(from.max(covered), to) {
                let si = *read.span.get(under as usize)?;
                if si != u32::MAX { affected.push(si); }
            }
            covered = to;
        }
        // Decide every span before the first write.
        let mut placed = Vec::with_capacity(affected.len());
        for &si in &affected {
            let span = index.spans.get(si as usize)?;
            let row = *read.row.get(si as usize).filter(|row| **row != u32::MAX)?;
            let items = entry.plan.items.get(span.start..span.start + span.len)?;
            placed.push(crate::frame::placed_span_row(scene, row, span.placement.as_ref()?, items)?);
        }
        let mut written = Vec::with_capacity(affected.len());
        // In place: nothing but this index holds the plan between frames.
        let plan_mut = Arc::make_mut(&mut entry.plan);
        for (&si, placed) in affected.iter().zip(&placed) {
            let span = &mut index.spans[si as usize];
            let recipe = span.placement.as_mut().expect("checked above");
            crate::frame::place_span(recipe, placed, &mut plan_mut.items[span.start..span.start + span.len]);
            written.extend((span.start..span.start + span.len).map(|item| item as u32));
        }
        written.sort_unstable();
        written.dedup();
        self.written.placed = written;
        index.built_at_generation = generation;
        entry.key = key;
        caches.incr_patch_hits = caches.incr_patch_hits.wrapping_add(1);
        caches.incr_patch_nodes = caches.incr_patch_nodes.wrapping_add(affected.len() as u64);
        caches.incr_placement_nodes = caches.incr_placement_nodes.wrapping_add(affected.len() as u64);
        Some((entry.plan.clone(), entry.warnings.clone()))
    }

    /// What the last successful [`Self::place`] or [`Self::patch`] wrote into the plan, for the engine
    /// (`frame::ChangedItems::Written`). Taken once; anything else that plans clears it.
    pub fn take_written(&mut self) -> PlanWritten {
        std::mem::take(&mut self.written)
    }

    /// `changed` is the complete batch since the last plan, with no bounded
    /// edit-log truncation. The caller proves these are dependency-safe
    /// translations: unchanged content, opacity, topology and derived records.
    /// General topology-preserving edits are not sufficient (text flow and
    /// boolean sources can change other nodes). The plan held here is written in place and then stands at `key`.
    /// A patch that fails part way leaves nothing held: the caller plans whole.
    #[allow(clippy::too_many_arguments)]
    pub fn patch(&mut self, scene: &Scene, hashes: &HashMap<String, NodeHashes>,
        opts: &RenderOptions, generation: u64, key: u64, changed: Vec<String>, caches: &mut PlanCaches,
    ) -> Option<(Arc<CoveragePlan>, Arc<PlanWarnings>)> {
        self.written = PlanWritten::default();
        // Preparing a node alone is safe beside a unit, but not inside one or above one: those placements
        // change the unit's capture. Ask only the changed nodes, including unit owners that emitted no items.
        // Decline before taking the entry so the caller can replace it with a whole plan.
        let patchable = self.entry.as_ref()?.patchable.as_ref()?;
        if patchable.units.as_ref().is_some_and(|units| changed.iter()
            .any(|id| units.above.contains(id) || units.holds(scene, id)))
            || patchable.index.opts_key != render_options_key(opts)
            || patchable.index.resource_key != resource_key()
            || patchable.surface_epoch != crate::surface::epoch_total() { return None; }
        let mut entry = self.entry.take()?;
        let index = &mut entry.patchable.as_mut()?.index;
        let changes = if generation == index.built_at_generation { DirtySince::Unchanged }
            else { DirtySince::Nodes(changed) };
        let mut written = PlanWritten::default();
        match apply_incremental_patch(index, &mut entry.plan, &entry.warnings,
            changes, generation, true, scene, hashes, caches, opts, CAMERA_PLAN_BUDGET_BYTES, entry.bytes, &[],
            Some(&mut written)) {
            IncrementalOutcome::Fallback => None,
            // Served this once; past the budget, or a shader clock now in it: not held.
            IncrementalOutcome::ServedDropSlot => Some((entry.plan, entry.warnings)),
            IncrementalOutcome::Served { bytes } => {
                if let Some(bytes) = bytes { entry.bytes = bytes; }
                entry.key = key;
                let served = (entry.plan.clone(), entry.warnings.clone());
                self.entry = Some(entry);
                self.written = written;
                Some(served)
            }
        }
    }
}

/// rust-edit-incremental-z4: the external-resource generation key folded into the
/// incremental slot's validity — pattern/image asset settle + font/text-host
/// settle. Mirrors what `page_fingerprint` folds (`store_generation` ⊕
/// `text_host_generation`); a change means an async resource settled and the
/// cached plan must be rebuilt even with no scene edit.
fn resource_key() -> u64 {
    let mut h = FxHasher::default();
    h.write_u64(moda_assets::engine_seam::store_generation());
    h.write_u64(crate::text_draw::text_host_generation());
    h.finish()
}

/// What a camera frame's pixels depend on besides the camera: the resident scene generation, the render options,
/// and the settled resources (assets, fonts). The motion cache keys on it.
pub fn motion_content_key(resident: &ResidentScene, opts: &RenderOptions) -> u64 {
    let mut h = FxHasher::default();
    h.write_u64(resident.generation());
    h.write_u64(render_options_key(opts));
    h.write_u64(resource_key());
    // Paint-layer pixels (raster spike): a stroke changes pixels without a scene edit.
    h.write_u64(crate::surface::epoch_total());
    h.finish()
}

/// rust-edit-incremental-z4: assemble the per-page plan index from a full build's
/// `PlanBuildAux` + a one-time reverse-dependency scan of the scene.
fn build_plan_index(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
    aux: crate::frame::PlanBuildAux,
    generation: u64,
    deps_memo: &mut Option<(u64, ReverseDeps)>,
) -> PlanIndex {
    let mut by_node = HashMap::with_capacity_and_hasher(aux.spans.len(), Default::default());
    for (i, span) in aux.spans.iter().enumerate() {
        // A node id appears once as a primary item; last-wins is harmless.
        by_node.insert(span.node_id.clone(), i);
    }
    // The scan is O(scene) and depends on the scene alone: once per generation (a
    // culled camera plan re-seeds its index on every pan past its window).
    let (blend_related_ids, mask_source_ids) = match deps_memo {
        Some((gen, deps)) if *gen == generation => deps.clone(),
        _ => {
            let (b, m) = scan_reverse_deps(scene);
            let deps = (Arc::new(b), Arc::new(m));
            *deps_memo = Some((generation, deps.clone()));
            deps
        }
    };
    PlanIndex {
        culled_roots: aux.culled_roots.into_iter().collect(),
        spans: aux.spans,
        by_node,
        page_index,
        page_clip: aux.page_clip,
        transparent: aux.transparent,
        opts_key: render_options_key(opts),
        resource_key: resource_key(),
        built_at_generation: generation,
        blend_related_ids,
        mask_source_ids,
    }
}

/// rust-edit-incremental-z4: collect the cross-node reverse dependencies that make
/// an edit's effect NON-local (so a dirty node in either set forces a full
/// rebuild). O(n), run once per full build. `blend_related_ids`: every blend node
/// + the objects and spine whose geometry it renders.
/// `mask_source_ids`: every `clipMaskNodeId` target (a group mask's coverage is
/// that node's geometry), and every clipped layer and its base (clip to the layer below). Connector anchors are deliberately absent — the engine
/// reads pre-baked line points node-locally, so a shape move never silently
/// re-routes a connector at plan time.
fn scan_reverse_deps(
    scene: &Scene,
) -> (
    std::collections::HashSet<String>,
    std::collections::HashSet<String>,
) {
    let mut blend = std::collections::HashSet::new();
    let mut mask_src = std::collections::HashSet::new();
    for id in scene.node_order() {
        let Some(node) = scene.effective_node(id) else {
            continue;
        };
        if node.node_type() == "blend" {
            blend.insert(id.clone());
            for src in moda_scene::blend::dependency_ids(node) {
                blend.insert(src.to_string());
            }
        }
        if let Some(m) = node.str("clipMaskNodeId") {
            if !m.is_empty() {
                mask_src.insert(m.to_string());
            }
        }
        // Clip to the layer below: the base's coverage clips the layers above it, and a clipped layer's toggle or
        // move changes which base the ones above it share, so an edit to any of them is non-local.
        if node.truthy("clipToBelow") {
            mask_src.insert(id.clone());
            if let Some(base) = scene.clip_below_base(id) {
                mask_src.insert(base.clone());
            }
        }
    }
    (blend, mask_src)
}

fn render_options_key(opts: &RenderOptions) -> u64 {
    let mut h = FxHasher::default();
    fold_render_options_fingerprint(&mut h, opts);
    // A moving frame's motion units (`crate::motion_capture`) change what its nodes plan to (nothing outside one).
    crate::motion_capture::fold_key(&mut h);
    h.finish()
}

/// rust-edit-incremental-z4: does any plan warning belong to node `id`? Warnings are
/// formatted `"{node_id} (...): ..."` / `"{node_id}: ..."` / `"{node_id}:innershadow ..."`,
/// so a node id followed by a space or colon is its prefix. The space/colon delimiter
/// keeps `n1` from matching `n10`'s warnings.
fn warnings_mention_node(skipped: &[String], id: &str) -> bool {
    skipped.iter().any(|w| {
        w.strip_prefix(id)
            // `rest.is_empty()` covers a warning that is EXACTLY the node id (no
            // suffix): `strip_prefix` yields `Some("")`, which neither delimiter
            // check would catch. Treating it as a match keeps the fallback strictly
            // conservative — a warned node never silently takes the patch path.
            .is_some_and(|rest| rest.is_empty() || rest.starts_with(' ') || rest.starts_with(':'))
    })
}

/// rust-edit-incremental-z4: is `id` a DESCENDANT of a blend- or mask-source node?
/// `scan_reverse_deps` records only the DIRECT source ids (a blend's objects and
/// spine, a `clipMaskNodeId` target), so a leaf nested inside a source GROUP is
/// not in those sets and would otherwise look eligible. But editing such a leaf
/// changes its ancestor's rendered geometry, which the blend/mask CONSUMER
/// composites — so the consumer's items (not the leaf's own span) go stale under an
/// in-place patch. Walk the parent chain and bail if any ancestor is a source.
/// Cycle-guarded: corrupt cyclic parent data resolves to "restricted" (fall back),
/// never an infinite loop.
fn has_blend_or_mask_ancestor(scene: &Scene, id: &str, index: &PlanIndex,
    memo: &mut HashMap<String, bool>) -> bool {
    if index.blend_related_ids.is_empty() && index.mask_source_ids.is_empty() { return false; }
    let mut chain = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut cursor = id;
    let restricted = loop {
        if let Some(&known) = memo.get(cursor) { break known; }
        if !seen.insert(cursor) { break true; }
        chain.push(cursor);
        if index.blend_related_ids.contains(cursor) || index.mask_source_ids.contains(cursor) { break true; }
        match scene.group_parent_of(cursor).or_else(|| scene.transient_parent_of().get(cursor)) {
            Some(parent) => cursor = parent,
            None => break false,
        }
    };
    for node in chain { memo.insert(node.to_string(), restricted); }
    restricted
}

/// renderer-perf-more-z6 (subtree/group incremental edits): expand a host-marked
/// dirty node into the set of ITEM-PRODUCING nodes whose DrawItems the edit can
/// actually change, appending each node's SPAN INDEX (an index into `index.spans`)
/// to `out` (deduped via `out_seen`) — span indices, not ids, so the walk allocates
/// nothing and the caller reads the id from the span directly.
///
/// - A dirty LEAF (no children) maps to itself: it must have a cached span, else
///   we cannot know its item shape stayed stable (it might have just started
///   producing items) — return `false` so the caller falls back. This preserves
///   the exact pre-existing single-leaf behaviour.
/// - A dirty PARENT (`group`/`container` with `children`) maps to every
///   item-producing node in its subtree (itself, if it has a body span — a
///   container with a fill/stroke does — plus all descendants present in the
///   cached index). An inherited transform/opacity edit on a parent flows down to
///   each descendant; `rebuild_node_items` re-reads the full ancestor product, so
///   each rebuilt descendant is byte-identical to a from-scratch rebuild. A parent
///   edit only reaches this path when it is transform/opacity-ONLY: `apply_props`
///   classifies any other key on a parent (fill/stroke/`clipMaskNodeId`/effects/…)
///   as STRUCTURAL via its `SAFE_PARENT_KEYS` gate (and `opacity` crossing 0,
///   `zIndex`, `visible`/`hidden` are structural for any node). A transform/opacity
///   edit never changes WHICH items the parent OR a descendant produces, only their
///   values — so the item set in `by_node` is stable and skipping a parent with no
///   span (a group, or a fill-less container) can never drop a needed item. This is
///   what makes an empty-`affected` expansion (a transform/opacity edit on a subtree
///   with no item-producing descendants) a correct no-op rather than a stale serve.
///
/// - A dirty TRANSIENT HOST (a `table`, or a `generated` chart — anything
///   with a `transient_by_parent` entry) maps to its expanded transient children.
///   A host is NOT `is_parent_node` (that is group/container only) and has no span of
///   its own — it is expanded AWAY, never its own draw item — so it contributes
///   nothing itself and everything through its transients. An EMPTY expansion is a
///   correct no-op for the same reason a span-less group is.
///
/// BYTE-EXACTNESS of splicing a RE-EXPANDED host's children into a retained plan.
/// Four points, in order:
///
/// (a) **The transient records are fully re-derived, not mutated.**
///     `apply_props` merges the patch into the host's CANONICAL node and then calls
///     `Scene::reexpand_transient_host`, which re-resolves the host through
///     `variables::resolve_node` and re-runs `expand_table_node` /
///     `expand_generated_graphic_node` from scratch. Expansion is a pure function of
///     the resolved host, and is ancestor-free (a transient inherits ancestor opacity
///     and transform at PAINT time, via `ancestor_opacity_product` /
///     `transform_parent_chain` — never at expansion time). So every transient record
///     is bit-identical to what `Scene::load(patched_json)` would have produced. A CPU
///     parity oracle in `moda-scene` asserts exactly that, over all five transient maps
///     plus `resolved_by_id`.
///
/// (b) **An identical id SEQUENCE means the display list is unchanged in membership
///     AND order,** so every cached plan-index span stays valid. `Scene::visit_z` emits
///     a host's transients by walking `transient_by_parent[host]` IN ORDER, so that
///     vector IS the display-list order for the host's slice. `apply_props` compares
///     `removed` vs `added` as sequences (not sets) and classifies any difference —
///     add, remove, or permutation — as STRUCTURAL → full rebuild. A same-length,
///     same-order re-expansion therefore patches spans that still describe the same
///     items.
///
/// (c) **`rebuild_node_items` re-reads the ancestor transform/opacity fresh** for each
///     rebuilt transient, so a `native-children` generated host's children get the
///     right model matrix even though the host moved and the children's own coords are
///     host-LOCAL. (Table transients bake the host x/y into their own coords instead;
///     re-expansion is what makes that correct, and their re-hash is what makes the
///     fingerprint move — see `reexpand_host_in_place`.)
///
/// (d) **`items_shape_match` still gates the splice** on item count, per-item variant,
///     node id and mask-freedom. So a re-expansion that keeps the id sequence but
///     changes what a transient PRODUCES — the canonical case being a cell whose text
///     went empty↔non-empty, which adds/removes a text item — is rejected and falls
///     back to a full rebuild rather than serving stale pixels. Pinned by a test.
///
/// Bounded, cycle-guarded subtree walk: a corrupt cyclic structure or a walk that
/// would exceed the indexed node count resolves to `false` (fall back), never an
/// infinite loop. Returns `false` (→ full rebuild) on any doubt.
fn collect_affected_item_nodes<'a>(
    scene: &'a Scene,
    index: &PlanIndex,
    root: &'a str,
    out: &mut Vec<usize>,
    out_seen: &mut std::collections::HashSet<usize>,
    walked: &mut std::collections::HashSet<&'a str>,
) -> bool {
    let Some(root_node) = scene.effective_node(root) else {
        return false; // unknown id — be conservative
    };
    // Viewport culling: a node the build culled (or one under a culled parent) has
    // no span; its edit may bring it into the window.
    if under_culled_root(scene, index, root) {
        return false;
    }
    // A dirty leaf must already be an item-producing node; if it has no span the
    // edit may have changed its item shape (e.g. empty → non-empty) and we cannot
    // patch it safely. (Identical to the pre-existing single-node guard.) We collect
    // SPAN INDICES (not ids): zero per-node allocation on this hot subtree-walk path,
    // and the caller reads `index.spans[si].node_id` directly (no second `by_node`
    // lookup).
    // A transient host walks its subtree too (its transient children),
    // even though it is not `is_parent_node` and has no span of its own.
    let walk_subtree = moda_scene::scene::is_parent_node(root_node)
        || scene.transient_by_parent().contains_key(root);
    if !walk_subtree {
        let Some(&si) = index.by_node.get(root) else {
            return false;
        };
        if out_seen.insert(si) {
            out.push(si);
        }
        return true;
    }
    // Parent: walk the subtree (inclusive) and collect every node that has a span.
    // Cap the walk at the TOTAL scene node count (+slack), not just the item-
    // producing span count: a deeply-nested hierarchy of span-less group/container
    // boundaries is a legitimate subtree but has more nodes than spans, so capping
    // on `spans.len()` would trip a spurious (correctness-safe but slow) fallback.
    // The total node count is the true upper bound; exceeding it means corrupt/
    // cyclic data.
    //
    // TRANSIENTS COUNT. `node_order` is CANONICAL nodes only, and a
    // transient host is one canonical node that expands into hundreds (a 4x3 table
    // → 30, a 20x20 → ~1.6k). Before this term the cap on a two-node page holding
    // one small table was 18 while the walk legitimately visited 31, so EVERY host
    // patch tripped the runaway guard and fell back — silently giving back the whole
    // win with no error anywhere. Canonical + transient is the true upper bound.
    let cap = scene.node_order().len() + scene.transient_by_id().len() + 16;
    let mut stack: Vec<&'a str> = vec![root];
    while let Some(id) = stack.pop() {
        if !walked.insert(id) {
            continue; // already visited — cycle-safe
        }
        if walked.len() > cap {
            return false; // runaway / corrupt structure → fall back
        }
        if index.culled_roots.contains(id) {
            return false; // a culled descendant may move into the window
        }
        // Self contributes only if it produced items (a pure group boundary does
        // not; a container body / leaf does).
        if let Some(&si) = index.by_node.get(id) {
            if out_seen.insert(si) {
                out.push(si);
            }
        }
        if let Some(kids) = scene.children_by_parent().get(id) {
            for k in kids {
                stack.push(k.as_str());
            }
        }
        if let Some(kids) = scene.transient_by_parent().get(id) {
            for k in kids {
                stack.push(k.as_str());
            }
        }
    }
    true
}

/// rust-edit-incremental-z4: outcome of an incremental patch attempt on a
/// taken-out slot's `index` + `plan`.
enum IncrementalOutcome {
    /// The slot serves this render (unchanged or patched in place) — KEEP it.
    /// `bytes` is the patched plan's recomputed marginal heap when a patch ran
    /// (so the caller updates its `incr_*_bytes` accounting), or `None` for a
    /// no-op static re-render (the byte count is unchanged).
    Served { bytes: Option<usize> },
    /// Served, but DROP the slot afterwards — either a rebuilt span now contains a
    /// shader fill (`became_shader` drops on ANY rebuilt shader span,
    /// including an in-place edit of an already-re-threadable shader node — a
    /// deliberately conservative simplicity choice, NOT a clock-freeze necessity;
    /// the camera lane re-threads the clock via the present override) OR the patched
    /// plan grew past the byte budget (don't retain an over-budget plan). Serve once,
    /// rebuild next frame.
    ServedDropSlot,
    /// Cannot serve incrementally — the caller full-rebuilds + reseeds.
    Fallback,
}

/// rust-edit-incremental-z4: the shared incremental-patch CORE for BOTH the page
/// (`build_plan_retained`) and camera (`build_camera_plan_retained`) lanes. Given a
/// taken-out slot's `index` + `plan` + the seed build's warnings, applies
/// the dirty-node patches IN PLACE (via `Arc::make_mut`) and reports whether the
/// slot can serve this render. Callers do the slot-identity / opts / resource-key
/// gate first; this handles only `dirty_since` → eligibility → splice + stats.
#[allow(clippy::too_many_arguments)]
fn apply_incremental_patch(
    index: &mut PlanIndex,
    plan: &mut Arc<CoveragePlan>,
    seed_warnings: &crate::frame::PlanWarnings,
    changes: DirtySince,
    generation: u64,
    translation_only: bool,
    scene: &Scene,
    hashes: &HashMap<String, NodeHashes>,
    caches: &mut PlanCaches,
    opts: &RenderOptions,
    budget_bytes: usize,
    current_bytes: usize,
    // Brush plan P8: nodes to rebuild although the resident scene did not
    // change them (outlines refined since the plan drew them stale).
    extra_dirty: &[String],
    // Where to note the items written, for a host that tells the engine what changed (`FramePlanIndex::patch`).
    mut written: Option<&mut PlanWritten>,
) -> IncrementalOutcome {
    let since = match (changes, extra_dirty) {
        (d, []) => d,
        (DirtySince::Unchanged, extra) => DirtySince::Nodes(extra.to_vec()),
        (DirtySince::Nodes(mut v), extra) => {
            for id in extra {
                if !v.contains(id) {
                    v.push(id.clone());
                }
            }
            DirtySince::Nodes(v)
        }
        (DirtySince::Structural, _) => DirtySince::Structural,
    };
    match since {
        DirtySince::Unchanged => {
            // Static re-render — nothing changed since the slot was built.
            caches.incr_static_hits = caches.incr_static_hits.wrapping_add(1);
            IncrementalOutcome::Served { bytes: None }
        }
        DirtySince::Structural => {
            caches.incr_fallbacks = caches.incr_fallbacks.wrapping_add(1);
            IncrementalOutcome::Fallback
        }
        DirtySince::Nodes(dirty) => {
            // renderer-perf-more-z6: expand each host-marked dirty node into the
            // item-producing nodes whose DrawItems the edit can change — a dirty
            // leaf is itself; a dirty PARENT (group/container) is its whole subtree
            // of item-producers, so a group/multi-node transform-or-opacity drag
            // patches O(subtree) leaves instead of falling back to a full O(n)
            // rebuild. Each expanded node is rebuilt independently below and
            // shape-checked; byte-exactness is proven by `MODA_INCREMENTAL_VERIFY`
            // + the `mutation_replay` GPU gate. Any doubt (unknown id, dirty leaf
            // with no span, cyclic/runaway subtree) → full rebuild.
            let mut affected: Vec<usize> = Vec::with_capacity(dirty.len());
            let mut affected_seen: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
            let mut walked = std::collections::HashSet::new();
            for id in &dirty {
                if !collect_affected_item_nodes(scene, index, id, &mut affected, &mut affected_seen, &mut walked)
                {
                    caches.incr_fallbacks = caches.incr_fallbacks.wrapping_add(1);
                    return IncrementalOutcome::Fallback;
                }
            }
            // Gather the spans to patch, bailing (full rebuild) on any node that is
            // not fast-path-eligible. Also bail if a node carried a warning in the
            // SEED build (`skipped` OR a fail-closed `dropped` refusal): the patch
            // path returns the seed warnings unchanged, so a node that warned then
            // could leave a stale entry once it stops warning — fall back to keep the
            // host-visible warning set authoritative (the rebuild below independently
            // bails if a patched node produces a NEW warning of either kind).
            // `affected` holds span indices, so the node id comes straight from the
            // span (no second `by_node` lookup, no alloc).
            let mut to_patch: Vec<usize> = Vec::with_capacity(affected.len());
            let mut restriction_memo = HashMap::default();
            for &si in &affected {
                // Checked span lookup (`.get`, not `[]`) so a desynced index can
                // never panic — any miss just falls back to the full rebuild.
                let Some(span) = index.spans.get(si) else {
                    caches.incr_fallbacks = caches.incr_fallbacks.wrapping_add(1);
                    return IncrementalOutcome::Fallback;
                };
                let id = span.node_id.as_str();
                // Eligibility: the node must not be a blend/mask source nor nested
                // inside one (editing it changes a CONSUMER's composite, not just
                // this node's own span). Container bodies (a parent WITH a span) are
                // allowed — `rebuild_node_items` drives the exact same
                // `append_node_draw_items` route for them, and the post-rebuild
                // shape/mask checks below catch anything unexpected.
                let eligible = !index.blend_related_ids.contains(id)
                    && !index.mask_source_ids.contains(id)
                    && !has_blend_or_mask_ancestor(scene, id, index, &mut restriction_memo);
                if !eligible
                    || warnings_mention_node(&seed_warnings.skipped, id)
                    || warnings_mention_node(&seed_warnings.dropped, id)
                {
                    caches.incr_fallbacks = caches.incr_fallbacks.wrapping_add(1);
                    return IncrementalOutcome::Fallback;
                }
                to_patch.push(si);
            }
            let page_clip = index.page_clip;
            let transparent = index.transparent;
            // `make_mut` is in-place when the slot UNIQUELY owns the Arc (the page
            // editor lane between frames → true O(k)). On the CAMERA lane the
            // engine's prepared-frame holds a second strong ref, so it clones —
            // which is REQUIRED there anyway: a fresh Arc pointer makes the
            // prepared-frame `ptr_eq` correctly MISS and rebuild the GPU buffers,
            // whereas an in-place mutate would alias the same pointer and present
            // stale buffers. Either way correct; the camera clone copies N DrawItems
            // — but since renderer-perf-more-z6 Arc-shared the body geometry, that is
            // O(n) pointer-bumps for the (large) vertex streams plus the smaller
            // per-item `Box<PaintInstance>` + `node_id` String copies, not an O(n)
            // deep vertex copy. It still skips the DL build + fingerprint + per-node
            // instance assembly a full rebuild pays.
            let plan_mut = Arc::make_mut(plan);
            let mut became_shader = false;
            // Delta byte-accounting: accumulate (new span bytes − old span bytes)
            // over the patched spans only, so the slot's retained size is kept
            // accurate in O(patched-items) instead of an O(n) full re-walk per edit
            // (preserves the page lane's O(dirty) cost).
            let mut delta_bytes: i64 = 0;
            // Owner id → index in the plan's mask table, built on the first masked node.
            let mut mask_owners: Option<HashMap<String, usize>> = None;
            for &si in &to_patch {
                let span = &mut index.spans[si];
                if translation_only {
                    if let (Some(recipe), Some(items)) = (span.placement.as_mut(),
                        plan_mut.items.get_mut(span.start..span.start + span.len)) {
                        let _placement = crate::prof::span("canvas.plan.placement");
                        // Capture cut-outs can shift a span. Recipes are certified only for the
                        // owner's plain items; generated paint/brush items instead rebuild below,
                        // where items_shape_match compares their actual old/new draw identities.
                        if items.iter().all(|item| drawitem_node_id(item) == Some(span.node_id.as_str()))
                            && crate::frame::translate_span(scene, recipe, items) {
                            caches.incr_placement_nodes = caches.incr_placement_nodes.wrapping_add(1);
                            if let Some(written) = written.as_deref_mut() {
                                written.placed.extend((span.start..span.start + span.len).map(|item| item as u32));
                            }
                            continue;
                        }
                    }
                }
                // Scratch mask table: an unmasked node never adds to it; a masked
                // node (inside a clipping frame / mask group) rebuilds its owner's
                // mask here, and `remap_scratch_masks` maps it onto the plan's.
                let mut scratch = crate::effects::PlanMasks::with_text_masks(
                    opts.text_masks_available.unwrap_or(true),
                )
                .with_vector_text_masks(true)
                .in_working_color(opts.working.is_some());
                let (mut new_items, z_consumed, w) = crate::frame::rebuild_node_items(
                    scene,
                    &span.primary,
                    opts,
                    transparent,
                    page_clip,
                    None,
                    &mut scratch,
                    Some((caches, hashes)),
                    span.z,
                );
                // A node inside a clipping frame / mask group rebuilt its owner's mask into the scratch
                // table; point its items at the plan's own entry, provided the owner's mask is unchanged.
                let mut mask_warnings = crate::frame::PlanWarnings::default();
                let scratch_masks = scratch.into_masks(&mut mask_warnings);
                if !mask_warnings.is_empty()
                    || !remap_scratch_masks(
                        &mut new_items,
                        &scratch_masks,
                        &plan_mut.masks,
                        &mut mask_owners,
                    )
                {
                    caches.incr_fallbacks = caches.incr_fallbacks.wrapping_add(1);
                    return IncrementalOutcome::Fallback;
                }
                let Some(old) = plan_mut.items.get(span.start..span.start + span.len) else {
                    caches.incr_fallbacks = caches.incr_fallbacks.wrapping_add(1);
                    return IncrementalOutcome::Fallback;
                };
                if z_consumed != span.z_consumed
                    || !w.is_empty()
                    || !items_shape_match(&new_items, old)
                {
                    caches.incr_fallbacks = caches.incr_fallbacks.wrapping_add(1);
                    return IncrementalOutcome::Fallback;
                }
                // Accumulate the byte delta while `old` is still borrowed (before the
                // splice below overwrites it).
                delta_bytes += items_bytes(&new_items) as i64 - items_bytes(old) as i64;
                // Dropping the slot on ANY rebuilt shader span (below) is
                // INTENTIONAL conservative simplicity — including an in-place edit of
                // an ALREADY re-threadable shader node, which could in principle be
                // retained. So a shader-node edit serves once then rebuilds; only
                // drags of UNRELATED (non-shader) nodes retain the slot. Refining
                // this to keep the slot on re-threadable in-place edits is a perf
                // optimization deliberately left out of scope.
                became_shader |= items_use_shader_time(&new_items);
                let Some(dest) = plan_mut.items.get_mut(span.start..span.start + span.len) else {
                    caches.incr_fallbacks = caches.incr_fallbacks.wrapping_add(1);
                    return IncrementalOutcome::Fallback;
                };
                // MOVE (not clone) the freshly built items into the plan: `new_items`
                // is discarded right after, so deep-cloning their geometry/text
                // batches would be pure waste. `items_shape_match` guaranteed equal
                // lengths, so the zip overwrites every dest slot exactly.
                for (at, (slot, item)) in dest.iter_mut().zip(new_items).enumerate() {
                    let old = std::mem::replace(slot, item);
                    if let Some(written) = written.as_deref_mut() {
                        written.replaced.push(((span.start + at) as u32, old));
                    }
                }
                // The planner prepared the span again: its recipe describes the new items, or is dropped.
                span.placement = span.placement.take().and_then(|recipe|
                    crate::frame::recertify_placement(scene, recipe, &*dest, page_clip));
            }
            index.built_at_generation = generation;
            caches.incr_patch_hits = caches.incr_patch_hits.wrapping_add(1);
            caches.incr_patch_nodes = caches.incr_patch_nodes.wrapping_add(to_patch.len() as u64);
            // A patch can GROW the plan while preserving shape (e.g. a richtext
            // content edit → more glyph quads, or a path `points` edit → more
            // vertices). Apply the accumulated delta to the slot's retained size; if
            // it now exceeds the byte budget, serve this frame but DROP the slot so
            // we never retain an over-budget plan (the same guard the seed applies).
            let new_bytes = (current_bytes as i64 + delta_bytes).max(0) as usize;
            if became_shader || new_bytes > budget_bytes {
                IncrementalOutcome::ServedDropSlot
            } else {
                IncrementalOutcome::Served {
                    bytes: Some(new_bytes),
                }
            }
        }
    }
}

/// rust-edit-incremental-z4: the PAGE incremental patcher (`render_page`). Returns
/// `Some(plan)` when the live page slot can serve this render and `None` to fall
/// back to the full rebuild (which re-seeds the slot). The slot is `take`n out so
/// `rebuild_node_items` can borrow `&mut caches` (R3/R4 caches); restored on serve,
/// dropped on fallback / shader-introduction.
fn try_incremental_fast_path(
    resident: &ResidentScene,
    caches: &mut PlanCaches,
    page_index: usize,
    page_id: &str,
    opts: &RenderOptions,
) -> Option<(Arc<CoveragePlan>, Arc<PlanWarnings>)> {
    let mut slot = caches.incremental_page.take()?;
    // A different page / opts than what we're rendering now — NOT our slot. RESTORE
    // it (don't drop) so an interleaved render of another page (e.g. a thumbnail or
    // export pass sharing this `PlanCaches`, served from a cache hit) doesn't evict
    // the editor page's still-valid slot. NOTE shader_time is intentionally NOT gated
    // here: the PAGE index is only ever seeded for plans that do NOT use the shader
    // clock (`!plan_uses_shader_time` at the seed site — the page present path can't
    // re-thread a cached clock, so shader plans are never seeded here), so a changing
    // `shader_time` (the live animation clock the editor advances every frame) does
    // not affect this page's plan — gating on it would wrongly force a full rebuild
    // every frame.
    if slot.page_id != page_id
        || slot.index.page_index != page_index
        || slot.index.opts_key != render_options_key(opts)
    {
        caches.incremental_page = Some(slot);
        return None;
    }
    // Same page, but an external resource (pattern/image asset or font) settled
    // asynchronously since the slot was built (no scene edit) — the cached plan may
    // hold a stale placeholder / un-shaped text batch. Fall back so the full rebuild
    // reseeds with the settled resource.
    if slot.index.resource_key != resource_key() {
        caches.incr_fallbacks = caches.incr_fallbacks.wrapping_add(1);
        caches.incr_page_bytes = 0; // slot dropped — zero its accounting too
        return None;
    }
    let scene = resident.scene()?;
    let hashes = resident.hashes()?;
    let current_bytes = caches.incr_page_bytes;
    let changes = resident.dirty_since(slot.index.built_at_generation);
    let outcome = apply_incremental_patch(
        &mut slot.index,
        &mut slot.plan,
        &slot.warnings,
        changes,
        resident.generation(),
        false,
        scene,
        hashes,
        caches,
        opts,
        PAGE_PLAN_BUDGET_BYTES,
        current_bytes,
        &[],
        None,
    );
    match outcome {
        IncrementalOutcome::Served { bytes } => {
            if let Some(b) = bytes {
                caches.incr_page_bytes = b; // patched: keep the watermark accurate
            }
            let out = (slot.plan.clone(), slot.warnings.clone());
            caches.incremental_page = Some(slot);
            Some(out)
        }
        // Slot dropped (not restored) → drop its byte accounting too. A subsequent
        // full rebuild re-seeds both; a ServedDropSlot frame returns the patched plan
        // but lets the next frame rebuild via the shader-aware / re-seed path.
        IncrementalOutcome::ServedDropSlot => {
            caches.incr_page_bytes = 0;
            Some((slot.plan.clone(), slot.warnings.clone()))
        }
        IncrementalOutcome::Fallback => {
            caches.incr_page_bytes = 0;
            None
        }
    }
}

/// Whether a page holds a group painting its appearance (E8.2): the scene keeps them all, so a document without one
/// answers without a walk.
fn page_has_group_appearance(scene: &Scene, page_id: &str) -> bool {
    !scene.group_appearance_by_id().is_empty()
        && scene
            .group_appearance_by_id()
            .keys()
            .any(|id| scene.page_of(id).as_deref() == Some(page_id))
}

/// rust-edit-incremental-z4: structural match between a freshly-rebuilt node's items
/// and the cached ones — same count, same per-item variant + node id, and NEITHER
/// side carrying a mask ref (eligible nodes are unmasked; a mask ref means the node
/// is clipped and the scratch-table rebuild can't be spliced safely). Values
/// (geometry/transform/paint) legitimately differ after an edit — those are proven
/// equal to a full rebuild by `MODA_INCREMENTAL_VERIFY` + the mutation_replay gate.
fn items_shape_match(new: &[crate::seams::DrawItem], old: &[crate::seams::DrawItem]) -> bool {
    if new.len() != old.len() {
        return false;
    }
    new.iter().zip(old).all(|(n, o)| {
        std::mem::discriminant(n) == std::mem::discriminant(o)
            && drawitem_node_id(n) == drawitem_node_id(o)
            && drawitem_mask(n) == drawitem_mask(o)
    })
}

/// Re-point a patched node's items from the scratch mask table its rebuild filled onto the plan's table.
/// Every scratch mask must equal the plan's mask for the same owner, field for field, with its ancestor
/// chain mapped the same way (an edited clip owner changes its mask, so the patch falls back and the full
/// rebuild re-plans the mask). False → fall back. `owners` caches the plan's owner → index map.
fn remap_scratch_masks(
    items: &mut [crate::seams::DrawItem],
    scratch: &[crate::seams::MaskPlan],
    plan: &[crate::seams::MaskPlan],
    owners: &mut Option<HashMap<String, usize>>,
) -> bool {
    if scratch.is_empty() {
        return true;
    }
    let owners = owners.get_or_insert_with(|| {
        let mut m = HashMap::with_capacity_and_hasher(plan.len(), Default::default());
        for (j, p) in plan.iter().enumerate() {
            m.entry(p.owner_node_id.clone()).or_insert(j);
        }
        m
    });
    // Scratch ancestors are built before their dependants (a smaller index), so `map` has them.
    let mut map: Vec<usize> = Vec::with_capacity(scratch.len());
    for m in scratch {
        let Some(&j) = owners.get(&m.owner_node_id) else {
            return false;
        };
        let ancestor = match m.ancestor {
            Some(a) => match map.get(a) {
                Some(&ja) => Some(ja),
                None => return false,
            },
            None => None,
        };
        let mut mapped = m.clone();
        mapped.ancestor = ancestor;
        if mapped != plan[j] {
            return false;
        }
        map.push(j);
    }
    for item in items.iter_mut() {
        let Some(id) = crate::camera_slot::item_mask_id(item) else { continue };
        match map.get(id as usize) {
            Some(&j) => crate::camera_slot::set_item_mask_id(item, j as u64),
            None => return false,
        }
    }
    true
}

pub(crate) fn drawitem_node_id(item: &crate::seams::DrawItem) -> Option<&str> {
    use crate::seams::DrawItem;
    match item {
        DrawItem::PageFill { .. } => None,
        DrawItem::Body { node_id, .. } => Some(node_id),
        DrawItem::Shadow { node_id, .. } => Some(node_id),
        DrawItem::Text(b) => Some(&b.node_id),
        DrawItem::BackdropSplit(s) => Some(&s.node_id),
        // SDF bodies are unmasked (an active mask disqualifies them),
        // so they're eligible for the incremental shape-match — keyed by node id
        // like any other body.
        DrawItem::SolidShapeSdf { node_id, .. } => Some(node_id),
        DrawItem::GradientShapeSdf { node_id, .. } => Some(node_id),
        // Unreachable through the incremental lanes (layer-blur pages
        // never seed the slot), keyed defensively anyway.
        DrawItem::LayerBlurComposite { node_id, .. } => Some(node_id),
    }
}

fn drawitem_mask(item: &crate::seams::DrawItem) -> Option<&crate::seams::MaskRef> {
    use crate::seams::DrawItem;
    match item {
        DrawItem::Body { mask, .. } => mask.as_ref(),
        // vork: a text batch carries no mask (moda-text-draw's TextQuadBatch; see PROVENANCE.md).
        DrawItem::Text(_) => None,
        DrawItem::LayerBlurComposite { mask, .. } => mask.as_ref(),
        _ => None,
    }
}

/// rust-edit-incremental-z4: dev correctness oracle. When `MODA_INCREMENTAL_VERIFY`
/// is set, rebuild the page plan from scratch and assert the incremental plan's
/// items are byte-for-byte (fingerprint-)identical. Off by default (a no-op) — the
/// shipped proof is the `mutation_replay` GPU gate; this is the fast CPU loop.
fn debug_verify_incremental(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
    hashes: &HashMap<String, NodeHashes>,
    got: &Arc<CoveragePlan>,
) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let on = *ENABLED.get_or_init(|| std::env::var("MODA_INCREMENTAL_VERIFY").is_ok());
    if !on {
        return;
    }
    let mut fresh_caches = PlanCaches::default();
    let (full, _w, _aux) = crate::frame::build_coverage_plan_indexed(
        scene,
        page_index,
        opts,
        Some((&mut fresh_caches, hashes)),
    )
    .expect("verify: full rebuild");
    // The PAGE lane is non-shader-only (never seeds a shader plan
    // here), so a served page plan carries no re-threaded clock — the raw
    // fingerprint is correct (and strictly tighter than the normalized variant the
    // CAMERA oracle needs).
    let gi = plan_items_fingerprint(got);
    let fi = plan_items_fingerprint(&full);
    if gi != fi {
        let n = got.items.len().min(full.items.len());
        let first_div = (0..n)
            .find(|&i| drawitem_fingerprint(&got.items[i]) != drawitem_fingerprint(&full.items[i]));
        panic!(
            "MODA_INCREMENTAL_VERIFY: incremental plan diverged from full rebuild \
             (page {page_index}): incr {} items / full {} items; first divergent item index {:?}",
            got.items.len(),
            full.items.len(),
            first_div
        );
    }
}

/// rust-edit-incremental-z4: diagnostic — rebuild the page plan from scratch and
/// return the first draw-item index whose Debug encoding differs from `got`, with
/// both encodings. `None` when identical. Used by the `edit_bench --replay`
/// harness to localize a divergence to a specific item (no GPU needed).
pub fn first_item_divergence(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
    hashes: &HashMap<String, NodeHashes>,
    got: &CoveragePlan,
) -> Option<(usize, String, String)> {
    let mut fresh = PlanCaches::default();
    let (full, _w, _aux) = crate::frame::build_coverage_plan_indexed(
        scene,
        page_index,
        opts,
        Some((&mut fresh, hashes)),
    )
    .ok()?;
    let n = got.items.len().max(full.items.len());
    for i in 0..n {
        let a = got.items.get(i).map(|x| format!("{x:?}"));
        let b = full.items.get(i).map(|x| format!("{x:?}"));
        if a != b {
            return Some((
                i,
                a.unwrap_or_else(|| "<none>".into()),
                b.unwrap_or_else(|| "<none>".into()),
            ));
        }
    }
    None
}

/// rust-edit-incremental-z4: the CAMERA twin of [`first_item_divergence`] — rebuild
/// the combined camera plan from scratch and return the first draw-item index that
/// DIVERGES from `got`, with both items' raw Debug encodings as the human-readable
/// report. Used by `edit_bench --replay --camera` to prove the camera incremental
/// patcher is byte-exact MODULO the re-threaded shader clock (no GPU needed):
/// The divergence DECISION uses the clock-normalized
/// `drawitem_fingerprint_verify` (a re-threadable shader's seed-clock vs
/// current-clock `shader_params` slots ride the camera present override, so they
/// are intentionally NOT a divergence), while a genuine non-clock difference IS
/// reported. The returned strings are the raw (un-normalized) Debug, so the report
/// shows the real item even though the compare was normalized.
pub fn first_camera_item_divergence(
    scene: &Scene,
    visible_page_indices: &[usize],
    cam: &crate::frame::CameraParams,
    opts: &RenderOptions,
    hashes: &HashMap<String, NodeHashes>,
    got: &CoveragePlan,
) -> Option<(usize, String, String)> {
    let mut fresh = PlanCaches::default();
    let full = crate::frame::build_camera_plan_cached(
        scene,
        visible_page_indices,
        cam,
        opts,
        Some((&mut fresh, hashes)),
        None,
    )
    .ok()??;
    let full = full.0;
    let n = got.items.len().max(full.items.len());
    for i in 0..n {
        let ga = got.items.get(i);
        let fb = full.items.get(i);
        // Decide divergence on the CLOCK-NORMALIZED per-item fingerprint
        // (same as the camera verify oracle), so a re-threadable shader's
        // seed-clock vs current-clock `shader_params` slot difference is NOT
        // reported (it rides the camera present override), while a genuine
        // non-clock difference still is. The reported strings stay the RAW Debug
        // encodings so the diagnostic shows the actual item.
        let same = match (ga, fb) {
            (Some(g), Some(f)) => drawitem_fingerprint_verify(g) == drawitem_fingerprint_verify(f),
            (None, None) => true,
            _ => false,
        };
        if !same {
            return Some((
                i,
                ga.map(|x| format!("{x:?}"))
                    .unwrap_or_else(|| "<none>".into()),
                fb.map(|x| format!("{x:?}"))
                    .unwrap_or_else(|| "<none>".into()),
            ));
        }
    }
    None
}

/// rust-edit-incremental-z4: a deterministic fingerprint of a plan's draw stream
/// (item-by-item, including f32 bits), for the verifier + tests.
fn plan_items_fingerprint(plan: &CoveragePlan) -> u64 {
    let mut h = FxHasher::default();
    h.write_usize(plan.items.len());
    for it in &plan.items {
        h.write_u64(drawitem_fingerprint(it));
    }
    h.write_usize(plan.masks.len());
    h.finish()
}

/// rust-edit-incremental-z4: fingerprint one DrawItem via its Debug encoding — a
/// pragmatic deterministic digest that covers all fields (positions, matrices,
/// colors) without per-type PartialEq. Debug is derived on every DrawItem field, so
/// equal Debug ⇒ structurally equal; f32s render losslessly enough for this dev
/// oracle (the shipped proof is the GPU byte gate).
fn drawitem_fingerprint(item: &crate::seams::DrawItem) -> u64 {
    let mut h = FxHasher::default();
    h.write(format!("{item:?}").as_bytes());
    h.finish()
}

/// verify-ONLY DrawItem fingerprint that normalizes the re-threaded
/// shader clock so the incremental oracle doesn't false-positive. The CAMERA
/// incremental slot serves a plan whose re-threadable shader body still carries
/// the SEED clock in its `PaintInstance.shader_params` clock slots; the live clock
/// is re-threaded into a COPY at present (`finish_camera_frame` →
/// `cam.present_override(shader_time_s)`), not into the cached plan items. A
/// from-scratch rebuild at the CURRENT clock bakes a different value into those
/// exact slots, so a raw Debug compare would FALSELY diverge even though the
/// presented pixels match. Zero the clock slots (precisely what the override
/// overwrites) before hashing — for RE-THREADABLE bodies only (`clock_offsets:
/// Some(non-empty)`). A non-re-threadable baked clock (`clock_offsets: None`), a
/// clockless `Some([])` body, or a text shader fill is left untouched, so a
/// GENUINE clock freeze still diverges and is caught.
///
/// This touches ONLY the dev verify oracles — the real cache keys
/// (`page_fingerprint`/`camera_fingerprint`/`render_options_key`), the served plan,
/// and the render output are all unaffected.
pub(crate) fn drawitem_fingerprint_verify(item: &crate::seams::DrawItem) -> u64 {
    use crate::seams::{BodyPaint, DrawItem};
    if let DrawItem::Body {
        paint: BodyPaint::Shader(body),
        ..
    } = item
    {
        if let Some(offsets) = &body.clock_offsets {
            if !offsets.is_empty() {
                let mut normalized = item.clone();
                if let DrawItem::Body { instance, .. } = &mut normalized {
                    for &off in offsets {
                        if let Some(slot) = instance.shader_params.get_mut(off as usize) {
                            *slot = 0.0;
                        }
                    }
                }
                return drawitem_fingerprint(&normalized);
            }
        }
    }
    drawitem_fingerprint(item)
}

/// verify-ONLY plan fingerprint built from [`drawitem_fingerprint_verify`]
/// (re-threaded shader clock normalized), for the tests' oracles.
#[cfg(test)]
fn plan_items_fingerprint_verify(plan: &CoveragePlan) -> u64 {
    let mut h = FxHasher::default();
    h.write_usize(plan.items.len());
    for it in &plan.items {
        h.write_u64(drawitem_fingerprint_verify(it));
    }
    h.write_usize(plan.masks.len());
    h.finish()
}

/// Residency: fingerprint of the combined camera plan's CONTENT —
/// the visible-page set in order × each page's `page_fingerprint`. The camera
/// pose (cameraX/Y, zoom, viewport, dpr, clear) is excluded on purpose: the
/// draw items are canvas-absolute, so a pan/zoom does NOT change the content and
/// must hit the cache (the host re-presents via `CameraPresentOverride`). dpr
/// and sample_count ARE folded in (via `page_fingerprint`'s `opts`), so a
/// quality-ladder MSAA flip or a dpr change rebuilds once — correct, and cheap
/// because the per-node tessellation/text caches survive it.
/// M1 fingerprint memo: a CHEAP key that is a conservative SUPERSET of every
/// input to [`camera_fingerprint`], so an unchanged key proves an unchanged
/// fingerprint WITHOUT the O(n) per-page display-list rebuild. It folds:
/// - the resident `generation` — ⊇ ALL scene content the fingerprint reads
///   (per-node hashes, z-order, add/remove, page props/position, clip config,
///   blend/canvas-root sources): any change goes through `update`/`apply_props`
///   and bumps the generation;
/// - the visible page set (all indices, incl. out-of-range — only ever
///   over-detects vs the fingerprint's in-range filter, which is safe);
/// - the plan-relevant render options (the SAME `fold_render_options_fingerprint`
///   the fingerprint uses, so an opts change moves both);
/// - the asset-store + text-host generations (pattern settle / font load).
///
/// `shader_time` is intentionally NOT folded — `camera_get` gates it separately —
/// so the key stays stable across an animating shader's frames AND as the host
/// advances the clock on any scene. The worst case is a needless recompute
/// (safe), never a stale reuse.
fn camera_frame_key(generation: u64, visible_page_indices: &[usize], opts: &RenderOptions) -> u64 {
    let mut h = FxHasher::default();
    h.write(b"camera-frame-key");
    h.write_u64(generation);
    h.write_usize(visible_page_indices.len());
    for &idx in visible_page_indices {
        h.write_usize(idx);
    }
    fold_render_options_fingerprint(&mut h, opts);
    h.write_u64(moda_assets::engine_seam::store_generation());
    h.write_u64(crate::text_draw::text_host_generation());
    h.finish()
}

/// rust-render-perf-g8 (E3): the page-path twin of [`camera_frame_key`]. A cheap
/// key that is a conservative SUPERSET of every input to [`page_fingerprint`] for
/// a single page, so an unchanged key proves an unchanged fingerprint WITHOUT the
/// O(n) display-list rebuild + node-hash fold. It folds:
/// - the resident `generation` — ⊇ ALL scene content the fingerprint reads (per-node
///   hashes, z-order, add/remove, page props/position via the layout walk, clip
///   config): any change goes through `update`/`apply_props` and bumps it;
/// - the page index (the page being rendered);
/// - the plan-relevant render options (same `fold_render_options_fingerprint`);
/// - the asset-store + text-host generations (pattern settle / font load).
///
/// `shader_time` is intentionally NOT folded — `page_get` re-checks it — so the key
/// stays stable across an animating shader's frames. Worst case is a needless
/// recompute (safe), never a stale reuse.
/// Callers outside `ResidentScene` must advance `generation` for every scene mutation.
pub fn page_frame_key(generation: u64, page_index: usize, opts: &RenderOptions) -> u64 {
    let mut h = FxHasher::default();
    h.write(b"page-frame-key");
    h.write_u64(generation);
    h.write_usize(page_index);
    fold_render_options_fingerprint(&mut h, opts);
    h.write_u64(moda_assets::engine_seam::store_generation());
    h.write_u64(crate::text_draw::text_host_generation());
    h.finish()
}

pub fn camera_fingerprint(
    scene: &Scene,
    visible_page_indices: &[usize],
    opts: &RenderOptions,
    hashes: &HashMap<String, NodeHashes>,
) -> u64 {
    camera_fingerprint_with(scene, visible_page_indices, opts, hashes, None)
}

/// The display-list key of the freeform canvas root in `PlanCaches::dl_get`.
pub(crate) const CANVAS_ROOT_DL_KEY: &str = "\u{0}canvas-root";

/// [`camera_fingerprint`], reading display lists through the camera lane's
/// per-generation cache when `caches` is given (a zoom-band change re-keys the
/// frame without changing the scene).
fn camera_fingerprint_with(
    scene: &Scene,
    visible_page_indices: &[usize],
    opts: &RenderOptions,
    hashes: &HashMap<String, NodeHashes>,
    mut caches: Option<&mut PlanCaches>,
) -> u64 {
    let mut dl_for =
        |key: &str, build: &dyn Fn() -> moda_scene::DisplayList| match caches.as_deref_mut() {
            Some(c) => match c.dl_get(key) {
                Some(dl) => dl,
                None => {
                    let dl = Arc::new(build());
                    c.dl_put(key, dl.clone());
                    dl
                }
            },
            None => Arc::new(build()),
        };
    let mut h = FxHasher::default();
    h.write(b"camera-frame");
    // Fold ONLY in-range indices, matching `build_camera_plan_cached`'s skip of
    // stale host-cull indices, so the fingerprint reflects exactly the pages the
    // plan renders (a stale index dropping out of the cull then doesn't force a
    // needless rebuild). `page_fingerprint` already returns 0 for an out-of-range
    // index — this is a consistency/robustness guard, not a panic fix.
    // NOTE for an empty in-range set (panning over blank canvas) the fingerprint
    // folds nothing from `opts` — harmless: the plan is then just the flat
    // backdrop clear (no draw items), identical at any sample_count/dpr, and the
    // live dims/clear still ride the present override.
    let page_count = scene.doc().pages.len();
    h.write_usize(
        visible_page_indices
            .iter()
            .filter(|&&idx| idx < page_count)
            .count(),
    );
    for &idx in visible_page_indices.iter().filter(|&&idx| idx < page_count) {
        h.write_usize(idx);
        let page_id = scene.doc().pages[idx].id().to_string();
        let dl = dl_for(&page_id, &|| {
            moda_scene::build_display_list(scene, &page_id)
        });
        h.write_u64(page_fingerprint_with_dl(scene, idx, opts, hashes, &dl));
    }
    // The combined camera plan also renders freeform canvas-root
    // (off-page) nodes — and only in freeform layout — so fold them in too, or a
    // floating-node edit (with pages unchanged) would serve a stale cached plan.
    if scene.is_freeform() && !scene.doc().canvas_root_node_ids.is_empty() {
        h.write(b"canvas-root");
        // With NO visible pages (panned to the pasteboard) the page loop above
        // folds nothing, so fold the same plan-relevant options + asset/font
        // generations here that page_fingerprint folds — otherwise a render-option
        // change or an image/pattern/font settling would reuse a stale plan for a
        // canvas-root-only view.
        fold_render_options_fingerprint(&mut h, opts);
        let dl = dl_for(CANVAS_ROOT_DL_KEY, &|| {
            moda_scene::build_display_list_for_canvas(scene)
        });
        fold_display_list_fingerprint(&mut h, scene, &dl, hashes);
        h.write_u64(moda_assets::engine_seam::store_generation());
        h.write_u64(crate::text_draw::text_host_generation());
    }
    h.finish()
}

/// Whether the visible pages make the combined camera plan camera-dependent (`build_camera_plan_retained`):
/// a backdrop or split blend on a visible page or on the freeform canvas root. (Layer captures are
/// planned under a zoom-bucketed capture window instead, [`capture_pages`].)
fn camera_dependent_pages(scene: &Scene, visible_page_indices: &[usize]) -> bool {
    visible_page_indices.iter().any(|&i| {
        scene
            .doc()
            .pages
            .get(i)
            .is_some_and(|p| {
                crate::backdrop::page_has_active_backdrop(scene, p.id())
                    || crate::backdrop::page_has_split_blend(scene, p.id())
            })
    })
    // A freeform canvas-root backdrop node also makes the plan
    // camera-dependent (its split is camera-projected), so bypass the cache too.
        || (scene.is_freeform()
            && !scene.doc().canvas_root_node_ids.is_empty()
            && (crate::backdrop::canvas_root_has_active_backdrop(scene)
                || crate::backdrop::canvas_root_has_split_blend(scene)))
}

/// Whether the visible pages (or the freeform canvas root) hold layer captures — layer filters or
/// capture shadows. Their regions and scales depend on the camera, so the camera lane plans them under
/// a [`crate::frame::CaptureView`] (a zoom bucket and a window around the view) and caches the plan for
/// every camera that view serves.
fn capture_pages(scene: &Scene, visible_page_indices: &[usize]) -> bool {
    visible_page_indices.iter().any(|&i| {
            scene
                .doc()
                .pages
                .get(i)
                .is_some_and(|p| crate::layer_blur::page_has_active_layer_filter(scene, p.id()))
        })
        || (scene.is_freeform()
            && !scene.doc().canvas_root_node_ids.is_empty()
            && crate::layer_blur::canvas_root_has_active_layer_filter(scene))
}

/// Residency: build (or reuse) the combined multi-page camera plan
/// through every Stage-1 cache layer — the camera twin of `build_plan_retained`.
/// The caller must have `resident.update(json)`-ed first. On a content hit
/// (static pan/zoom of a non-backdrop scene) no draw lists are rebuilt; the host
/// applies the live camera via `CameraPresentOverride` at present time. Area 3:
/// a visible BACKDROP page makes the combined plan camera-dependent (its
/// `BackdropSplit` is in target pixels), so it bypasses the camera plan cache and
/// is built fresh each frame (per-node R3/R4 caches still apply). `Ok(None)` is
/// retained only as a defensive snapshot-fallback signal (no longer used for
/// backdrop).
pub fn build_camera_plan_retained(
    resident: &ResidentScene,
    caches: &mut PlanCaches,
    visible_page_indices: &[usize],
    cam: &crate::frame::CameraParams,
    opts: &RenderOptions,
) -> anyhow::Result<Option<(Arc<CoveragePlan>, Arc<PlanWarnings>)>> {
    // Cached while moving (`crate::motion_capture`): a moving frame plans its units as captures, in the moving
    // frames' scene state, with the clock the motion froze. A still frame enters nothing: it plans exactly as with
    // motion captures off.
    let units = caches.motion_frame(resident, opts.shader_time_s);
    let frozen;
    let opts = match (&units, caches.motion.clock) {
        (Some(_), Some(t)) if t.to_bits() != opts.shader_time_s.to_bits() => {
            frozen = RenderOptions {
                shader_time_s: t,
                ..opts.clone()
            };
            &frozen
        }
        _ => opts,
    };
    let _motion = crate::motion_capture::enter(units);
    // Display lists are reused across this lane's builds for one resident
    // generation (a culled plan rebuilds when the camera leaves its window).
    caches.dl_live_gen = Some(resident.generation());
    let clock = opts.shader_time_s.to_bits();
    // A first build counts as steady: a playing clock drops the slot on the next frame and seeds no more.
    caches.camera_clock_steady = caches.last_camera_clock.replace(clock).is_none_or(|c| c == clock);
    let out = camera_plan_retained(resident, caches, visible_page_indices, cam, opts);
    caches.dl_live_gen = None;
    caches.text_progressive = false;
    out
}

fn camera_plan_retained(
    resident: &ResidentScene,
    caches: &mut PlanCaches,
    visible_page_indices: &[usize],
    cam: &crate::frame::CameraParams,
    opts: &RenderOptions,
) -> anyhow::Result<Option<(Arc<CoveragePlan>, Arc<PlanWarnings>)>> {
    let scene = resident
        .scene()
        .ok_or_else(|| anyhow::anyhow!("no resident scene (update first)"))?;
    let hashes = resident.hashes().expect("hashes exist with scene");
    // The visible canvas rect: outlines refine on-screen first.
    let view = [
        cam.camera_x,
        cam.camera_y,
        cam.camera_x + cam.viewport_w / cam.zoom.max(1e-9),
        cam.camera_y + cam.viewport_h / cam.zoom.max(1e-9),
    ];
    // Viewport culling (`crate::cull`): on a large scene the plan draws only nodes
    // whose visual bounds meet a window around the view. A camera the current window
    // still serves reuses the cached plan / retained slot built with it; one it
    // doesn't (a pan past the margin, a zoom out, a deep zoom in) rebuilds around
    // the new view.
    let culling = !caches.cull_disabled
        && scene.node_order().len() + scene.transient_by_id().len()
            >= caches.cull_min_nodes.unwrap_or(CULL_MIN_NODES)
        && view.iter().all(|v| v.is_finite());
    // While the zoom changes (a wheel or pinch zoom), curves keep the flattening band the plan has, up to
    // MOTION_BAND_LAG bands from the camera's: a band crossing rebuilt the whole camera plan (its key holds the
    // band), a 60-280 ms frame at 20k layers. Such a frame asks the host for another (`outline_refining`), and the
    // first frame after the zoom holds is exact. Only the live camera lane turns this on (`motion_plans`).
    // "While the zoom changes": it changed within the last MOTION_ZOOM_HOLD_MS (a wheel zoom's frames don't
    // all see a new zoom), by the refine clock.
    let now_ms = (caches.outline_clock.unwrap_or(default_clock))();
    let zoom_changed = caches.last_camera_zoom.is_some_and(|z| z != cam.zoom);
    if zoom_changed {
        caches.last_zoom_change_ms = Some(now_ms);
    }
    let zooming_now = caches.motion_plans
        && caches
            .last_zoom_change_ms
            .is_some_and(|t| now_ms - t >= 0.0 && now_ms - t < MOTION_ZOOM_HOLD_MS);
    caches.zooming = zooming_now;
    caches.motion_plan = false;
    let serving_window: Option<crate::cull::Aabb> = if culling {
        caches.camera_cull.filter(|w| cull_window_serves(w, &view))
    } else {
        None
    };
    // The cached plan / slot is usable at this camera. (A capture page also
    // needs its capture view to serve: checked below, once it is known.)
    // Degraded text (a node the vector lane declined — every text node before the
    // document's fonts arrive) is rasterised at the screen's density and skipped
    // under a pixel. A plan that did either is good only up to the zoom octave it
    // was built for.
    let degraded_bucket = crate::text_draw::degraded_camera_bucket(cam.zoom * cam.dpr);
    let _degraded_camera = crate::text_draw::enter_degraded_camera(Some(degraded_bucket));
    let degraded_serves = caches
        .degraded_zoom_bucket
        .is_none_or(|b| degraded_bucket <= b);
    let window_serves = (!culling || serving_window.is_some()) && degraded_serves;
    // Outlines the previous frame drew stale: re-patch the most visible ones
    // (largest on-screen area first, then those near the viewport) through the
    // retained slot; off-screen ones wait. A full rebuild below recomputes the
    // list.
    //
    // By index into the queue (thousands of entries after a band change on a painting): each entry's area is
    // measured once, and the entries move (never clone) into whichever list the frame keeps.
    let pending = std::mem::take(&mut caches.outline_pending);
    caches.placement_filling = pending.iter().any(|p| p.unplaced);
    if pending.is_empty() {
        caches.text_ready.clear(); // nothing waits on them any more
    }
    let near = crate::tess::outline::near_view(view);
    let mut soon_by_area: Vec<(f64, usize)> = Vec::new();
    let mut soon_unplaced: Vec<usize> = Vec::new();
    let mut later: Vec<usize> = Vec::new();
    for (i, p) in pending.iter().enumerate() {
        // Strokes a host worker is still placing wait (they re-plan once their placement is promoted).
        if crate::tess::outline::visible_area(p.bbox, near) > 0.0 && !(p.unplaced && caches.placement_in_flight(&p.id)) {
            if p.unplaced {
                soon_unplaced.push(i);
            } else {
                soon_by_area.push((crate::tess::outline::visible_area(p.bbox, view), i));
            }
        } else {
            later.push(i);
        }
    }
    // Stable: equal areas keep the queue's order.
    soon_by_area.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    // Never-placed strokes first, in the paint order they were left out in (a painting fills in back to front,
    // and a stroke once drawn stays: no flicker); then the rest by on-screen area.
    let unplaced_soon = soon_unplaced.len();
    let mut soon: Vec<usize> = soon_unplaced;
    soon.extend(soon_by_area.into_iter().map(|(_, i)| i));
    let mut pending: Vec<Option<PendingOutline>> = pending.into_iter().map(Some).collect();
    // Candidates in proportion to the frame's refine budget: re-planning a unit (a curve re-flattened for the
    // new zoom band, a text layer laid out) costs ~0.06 ms at 20k layers whether or not the budget lets it
    // refine, so a zoom that crosses a band re-planned 1,024 units every frame (60 ms) while interacting.
    //
    // While the zoom changes, a live frame refines nothing: the curves and outlines it would refit are for a band
    // the next frame leaves, and every refit patches the plan, which costs the frame its prepared pass (a
    // replay). They draw from their previous band (detail reduced while moving) and refine once the zoom holds;
    // a still frame is exact.
    let zooming = caches.last_camera_zoom.is_some_and(|z| z != cam.zoom);
    caches.last_camera_zoom = Some(cam.zoom);
    let cap = match (caches.outline_refine_deadline, caches.outline_refine_budget_ms) {
        (Some(_), _) if zooming && !caches.refine_while_zooming => 0,
        (Some(_), Some(ms)) => ((ms * OUTLINE_REFINE_CANDIDATES_PER_MS) as usize).clamp(32, OUTLINE_REFINE_CANDIDATES),
        (Some(_), None) => OUTLINE_REFINE_CANDIDATES,
        (None, _) => usize::MAX,
    };
    // Never-placed strokes take candidates in proportion to the placement budget (they lead `soon`).
    let cap = match caches.outline_refine_budget_ms {
        Some(ms) if cap > 0 && cap != usize::MAX => {
            let unplaced = unplaced_soon;
            let placements = (ms * caches.placement_scale() * PLACEMENT_CANDIDATES_PER_MS) as usize;
            cap.max(unplaced.min(placements))
        }
        _ => cap,
    };
    // Nothing near the viewport pending: the cached camera plan (whose far
    // stand-ins nobody sees) is served as usual.
    let near_pending = !soon.is_empty();
    // The queue as the frame found it (the most visible first), kept for a cached plan that still draws them.
    let prior: Vec<usize> = soon.iter().chain(later.iter()).copied().collect();
    let id_of = |pending: &[Option<PendingOutline>], i: usize| -> String {
        pending[i].as_ref().map(|p| p.id.clone()).unwrap_or_default()
    };
    // Text laid out between frames (`prewarm_deferred_text`) re-plans as a cache hit: past the cap, all of it (a
    // first view of thousands of text layers fills in at once rather than a cap's worth a frame).
    if cap > 0 && !caches.text_ready.is_empty() {
        let (ready, rest): (Vec<usize>, Vec<usize>) = soon.into_iter().partition(|&i| {
            pending[i]
                .as_ref()
                .is_some_and(|p| caches.text_ready.contains(&p.id))
        });
        for &i in &ready {
            caches.text_ready.remove(&id_of(&pending, i));
        }
        soon = rest;
        later.extend(soon.split_off(soon.len().min(cap)));
        soon.extend(ready);
    } else {
        later.extend(soon.split_off(soon.len().min(cap)));
    }
    // Pending entries name draw items; a paint-stack layer clone (`n::paint::k`), a placed brush shape
    // (`n::brush::k`) or a shadow / glow silhouette (`n` + `\u{1}silhouette`) is not a scene node, so hand the patch
    // its owner (which rebuilds all of the node's layers, shapes or effects), as is a blend step (`{blend}-s{i}`).
    // The patch skips an unknown id: a silhouette drawn from another band stayed coarse forever (how coarse
    // depended on how much the frames that zoomed had time to refine).
    let mut carried: Vec<String> = Vec::with_capacity(soon.len());
    let mut carried_set: std::collections::HashSet<&str, rustc_hash::FxBuildHasher> = Default::default();
    for &i in &soon {
        let Some(p) = pending[i].as_ref() else { continue };
        let owner = crate::paint::layers::layer_owner_id(crate::tess::placed::placed_owner_id(
            crate::frame::silhouette_owner_id(&p.id),
        ));
        // A blend's step (`{blend}-s{i}`): its blend node. Skipped as unknown, a step drawn from another band's
        // flattening while off-screen was never refined (the patch dropped it from the queue).
        let owner = moda_scene::blend::step_owner(scene, owner).unwrap_or(owner);
        if carried_set.insert(owner) {
            carried.push(owner.to_string());
        }
    }
    drop(carried_set);
    drop(soon);

    // Scale RFC stage C: the retained camera plan (`camera_slot`). Bring it up to
    // the scene first — it knows, in O(edit), the scene-wide flags the frame is
    // planned under, which otherwise take a walk of every node per edit.
    let mut slot_prep: Option<(
        Box<crate::camera_slot::CameraSlot>,
        crate::camera_slot::Prepared,
    )> = None;
    if let Some(mut slot) = caches.camera_slot.take() {
        caches.incr_camera_bytes = 0;
        let same_frame = slot.visible_pages == visible_page_indices;
        if slot.clock.is_some_and(|c| c != opts.shader_time_s.to_bits()) {
            // Its items bake a clock the present can't re-thread, and the clock moved.
            caches.note_slot_fallback("shader clock moved");
        } else if !(window_serves && slot.cull == serving_window) {
            // A camera move past the window: the plan rebuilds, but what the slot
            // knows about the scene still holds at this generation.
            if slot.built_at_generation == resident.generation() {
                caches.camera_scene_index = Some(Box::new(slot.into_index()));
            }
        } else if !caches.incremental_disabled && same_frame {
            if slot.resource_key != resource_key() {
                // An image or font settled: its placeholder items are stale.
                caches.note_slot_fallback("an image or font settled");
            } else {
                match slot.prepare(
                    scene,
                    resident.dirty_detail(slot.built_at_generation),
                    &carried,
                ) {
                    Ok(p) => slot_prep = Some((slot, p)),
                    Err(why) => caches.note_slot_fallback(why),
                }
            }
        }
    }

    // Brush plan P8: outlined strokes refit per zoom band — memoised per
    // resident generation so a static pan does not rescan the scene.
    let (has_outlines, has_curves) = match (&slot_prep, caches.outline_memo) {
        (Some((_, p)), _) => {
            caches.outline_memo = Some((resident.generation(), p.scene_flags));
            p.scene_flags
        }
        (None, Some((gen, has))) if gen == resident.generation() => has,
        (None, _) => {
            let has = (
                crate::tess::outline::scene_has_outlines(scene),
                crate::tess::outline::scene_has_curves(scene),
            );
            caches.outline_memo = Some((resident.generation(), has));
            has
        }
    };
    let camera_scale = crate::tess::outline::camera_scale_for(has_outlines, has_curves, cam.zoom * cam.dpr);
    let camera_scale = match (camera_scale, caches.last_plan_scale) {
        (Some(now), Some(last))
            if zooming_now
                && (crate::tess::outline::band_for_scale(now) - crate::tess::outline::band_for_scale(last)).abs()
                    <= MOTION_BAND_LAG =>
        {
            if crate::tess::outline::band_for_scale(now) != crate::tess::outline::band_for_scale(last) {
                caches.motion_plan = true;
            }
            Some(last)
        }
        _ => camera_scale,
    };
    caches.last_plan_scale = camera_scale;
    let _outline_scale = crate::tess::outline::enter_camera(camera_scale);
    caches.outline_stale_served = 0;
    caches.placement_deferred = 0;
    // Banded curves (#87) refine like outlines: record the view for them too,
    // or off-screen curved strokes are never "far" and `outline_refining`
    // stays true forever.
    let refines = has_outlines || has_curves;
    let _outline_view = crate::tess::outline::enter_view(refines.then_some(view));
    caches.outline_view = refines.then_some(view);
    caches.outline_band = refines.then(crate::tess::outline::current_band);
    // The options key folds the zoom band once the camera's is entered: a plan
    // built at another band (outlined strokes and curves refit per band) or
    // under other options re-plans from scratch.
    let opts_key = render_options_key(opts);
    // The same options at another zoom band: the slot stays, re-keyed, and queues its units to refine at the new
    // band (a few hundred a frame, most visible first), drawing the old band's meanwhile, instead of a rebuild of
    // the whole plan in one frame (370 ms at fit → 100 % on a 15k-stroke painting).
    let bandless_key = {
        let _no_band = crate::tess::outline::enter_camera(None);
        render_options_key(opts)
    };
    if let Some((s, prep)) = slot_prep.as_mut() {
        // Not a plan with layer captures: kept, it would keep capturing over the old view's window (a capture's
        // pixels depend on its window: its region is cut to it, and grain and blur follow), where a build at the new
        // band captures over a window around the new view.
        if s.opts_key != opts_key
            && s.bandless_key == Some(bandless_key)
            && !caches.band_rekey_disabled
            && !prep.has_captures
        {
            s.opts_key = opts_key;
            // This frame refines the refine budget's worth, the most visible first, as the rebuild it replaces
            // would have (while the zoom changes too: the band only moves once the zoom has run past the lag); the
            // rest queue.
            let mut candidates = s.band_refine_candidates();
            candidates.sort_by(|a, b| {
                let (va, vb) = (
                    crate::tess::outline::visible_area(a.1, view),
                    crate::tess::outline::visible_area(b.1, view),
                );
                vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
            });
            let rekey_cap = match caches.outline_refine_budget_ms {
                Some(ms) => ((ms * OUTLINE_REFINE_CANDIDATES_PER_MS) as usize).clamp(32, OUTLINE_REFINE_CANDIDATES),
                None => usize::MAX,
            };
            let now = rekey_cap.saturating_sub(carried.len()).min(candidates.len());
            let rest = candidates.split_off(now);
            prep.add_unit_refines(candidates.into_iter().map(|(id, _)| id));
            for (id, bbox) in rest {
                caches.outline_pending.push(PendingOutline { id, bbox, unplaced: false });
            }
            caches.band_rekeys = caches.band_rekeys.wrapping_add(1);
        }
    }
    if slot_prep
        .as_ref()
        .is_some_and(|(s, _)| s.opts_key != opts_key)
    {
        // The plan rebuilds, but what the slot knows about the scene still holds at this generation (a zoom band
        // change re-seeded every node's flags: a third of a band change on a 15k-stroke painting).
        if let Some((slot, _)) = slot_prep.take() {
            if slot.built_at_generation == resident.generation() {
                caches.camera_scene_index = Some(Box::new(slot.into_index()));
            }
        }
        caches.note_slot_fallback("zoom band or render options changed");
    }

    // A visible backdrop page makes the combined plan camera-DEPENDENT — its
    // `BackdropSplit` is in camera target pixels, so it changes on every
    // pan/zoom. The retained slot re-plans just the split units per camera; a
    // full build is never cached by fingerprint.
    let memo_hit = match &caches.camera_dependent_memo {
        Some((gen, pages, dependent))
            if *gen == resident.generation() && pages.as_slice() == visible_page_indices =>
        {
            Some(*dependent)
        }
        _ => None,
    };
    let camera_dependent_plan = match &slot_prep {
        Some((_, p)) => {
            caches.camera_dependent_memo = Some((
                resident.generation(),
                visible_page_indices.to_vec(),
                p.camera_dependent,
            ));
            p.camera_dependent
        }
        None => memo_hit.unwrap_or_else(|| {
            let dependent = camera_dependent_pages(scene, visible_page_indices);
            caches.camera_dependent_memo = Some((
                resident.generation(),
                visible_page_indices.to_vec(),
                dependent,
            ));
            dependent
        }),
    };
    // Layer captures (layer filters, capture shadows) are planned under a
    // capture view — the camera's scale rounded up to a zoom bucket, over a
    // window around the view — so the plan is reused while the camera stays in
    // the window and the bucket, and the engine reuses each capture's result
    // while its content key holds (`layer_blur::cache`).
    let has_captures = match (&slot_prep, &caches.capture_memo) {
        (Some((_, p)), _) => {
            caches.capture_memo = Some((
                resident.generation(),
                visible_page_indices.to_vec(),
                p.has_captures,
            ));
            p.has_captures
        }
        (None, Some((gen, pages, has)))
            if *gen == resident.generation() && pages.as_slice() == visible_page_indices =>
        {
            *has
        }
        (None, _) => {
            let has = capture_pages(scene, visible_page_indices);
            caches.capture_memo = Some((
                resident.generation(),
                visible_page_indices.to_vec(),
                has,
            ));
            has
        }
    };
    let capture_scale = crate::frame::capture_scale_bucket(cam.zoom * cam.dpr);
    // Moving frames keep their captures' scale through a zoom (`MotionState::hold_scale`): the composite scales
    // them instead of capturing again at every step.
    let capture_scale = if crate::motion_capture::active() {
        caches.motion.hold_scale(capture_scale)
    } else {
        capture_scale
    };
    let serving_capture: Option<crate::frame::CaptureView> = if has_captures {
        caches
            .capture_view
            .filter(|v| v.scale == capture_scale && cull_window_serves(&v.window, &view))
    } else {
        None
    };
    if !has_captures {
        caches.capture_view = None;
    }

    // Serve from the retained slot: an edit re-plans the units it touched, a
    // camera move the split units, a zoom-step crossing the capture units.
    if let Some((mut slot, prep)) = slot_prep {
        // Captures keep the slot's capture window (the cull window); a zoom step
        // re-plans them at the new bucket.
        let capture = slot
            .capture
            .filter(|v| cull_window_serves(&v.window, &view))
            .map(|v| crate::frame::CaptureView {
                scale: capture_scale,
                window: v.window,
            });
        if has_captures == capture.is_some() {
            // A live frame lays text out within the refine budget here too: the text a first frame queued
            // (thousands of layers on a first visit) lands over several frames instead of one long one.
            caches.text_progressive = caches.outline_refine_deadline.is_some();
            let outcome = slot.apply(
                prep,
                scene,
                resident.generation(),
                hashes,
                caches,
                opts,
                cam,
                capture,
            );
            let drop_after = match outcome {
                crate::camera_slot::SlotOutcome::Unchanged => {
                    caches.incr_static_hits = caches.incr_static_hits.wrapping_add(1);
                    Some(false)
                }
                crate::camera_slot::SlotOutcome::Patched { units, drop_after } => {
                    caches.incr_patch_hits = caches.incr_patch_hits.wrapping_add(1);
                    caches.incr_patch_nodes = caches.incr_patch_nodes.wrapping_add(units as u64);
                    // Units that bake a clock the present can't re-thread: while the clock holds still, the slot
                    // keeps serving, keyed on it; a moving clock drops it after this frame, as before.
                    let clock_drop = drop_after && !caches.camera_clock_steady;
                    if drop_after && !clock_drop {
                        slot.clock = Some(opts.shader_time_s.to_bits());
                    }
                    Some(clock_drop || slot.bytes > CAMERA_SLOT_BUDGET_BYTES)
                }
                crate::camera_slot::SlotOutcome::Fallback(why) => {
                    caches.note_slot_fallback(why);
                    None
                }
            };
            if let Some(drop_after) = drop_after {
                // Degraded text the patch re-planned at this zoom bounds how far in
                // the plan serves.
                if crate::text_draw::degraded_zoom_limited() {
                    caches.degraded_zoom_bucket = Some(
                        caches
                            .degraded_zoom_bucket
                            .map_or(degraded_bucket, |b| b.min(degraded_bucket)),
                    );
                }
                caches.camera_window_frames = caches.camera_window_frames.saturating_add(1);
                caches.capture_view = capture;
                caches
                    .outline_pending
                    .extend(later.iter().filter_map(|&i| pending[i].take()));
                if caches.outline_pending.is_empty() {
                    debug_verify_camera_slot(
                        scene,
                        visible_page_indices,
                        cam,
                        opts,
                        hashes,
                        slot.cull.as_ref(),
                        capture,
                        &slot.plan,
                    );
                }
                let out = (slot.plan.clone(), slot.warnings.clone());
                if !drop_after {
                    caches.incr_camera_bytes = slot.bytes;
                    caches.camera_slot = Some(slot);
                }
                return Ok(Some(out));
            }
        } else {
            caches.note_slot_fallback("capture window no longer serves");
        }
    }
    // Full rebuild: it recomputes which outlines are drawn stale.
    caches.camera_rebuilds = caches.camera_rebuilds.wrapping_add(1);
    caches.outline_pending.clear();
    // A full rebuild lays text out within the refine budget and queues the rest; the next frames (patches of
    // the retained plan) lay out what was queued, within their budgets. See `text_draw::build_richtext_item`.
    caches.text_progressive = caches.outline_refine_deadline.is_some();

    // What the build records for the retained slot.
    let seeding = !caches.incremental_disabled;
    let mut plan_aux = crate::frame::CameraPlanAux::default();

    if camera_dependent_plan {
        // Rebuilt on every camera move without the slot: never cached by
        // fingerprint. Drop any stale camera-independent plan so it isn't
        // retained for the whole backdrop session.
        caches.camera_clear();
        // With the slot, cull to a window around the view (the slot re-projects
        // the splits while the camera stays in it); without, to the view itself
        // (the target holds only the view, and nothing outside it can be sampled
        // by a split). Also to the view in a fast pan — the last window, at this
        // zoom, served one frame or none and culled much: a window four views
        // large then costs more to build than it saves. Where the last build
        // culled little (a view at fit), the window costs nothing extra.
        let mut out = crate::frame::CullOut::default();
        let fast_pan = caches.camera_window_zoom == Some(cam.zoom)
            && caches.camera_window_frames <= 1
            && caches.cull_last_culled * 4 >= caches.cull_last_considered;
        let wide = seeding && !fast_pan;
        caches.camera_window_frames = 0;
        caches.camera_window_zoom = Some(cam.zoom);
        let window = culling.then(|| if wide { cull_window_for(view) } else { view });
        // Captures keep their window across frames all the same, so their
        // content keys (and cached results) hold while the camera pans.
        let capture_view = has_captures.then(|| {
            serving_capture
                .filter(|v| !wide || window.is_none_or(|w| v.window == w))
                .unwrap_or(crate::frame::CaptureView {
                    scale: capture_scale,
                    window: window
                        .filter(|_| wide)
                        .unwrap_or_else(|| cull_window_for(view)),
                })
        });
        caches.capture_view = capture_view;
        // A window rebuild copies the camera-independent nodes' items an earlier
        // window's build made (`PlanCaches::node_items`), as below.
        let node_items = seeding && window.is_some();
        if node_items {
            let key = (
                resident.generation(),
                render_options_key(opts),
                resource_key(),
            );
            if caches.node_items_key != Some(key) {
                caches.node_items.clear();
                caches.node_items_key = Some(key);
            }
            caches.node_items_live = true;
        }
        let built = crate::frame::build_camera_plan_culled_with(
            scene,
            visible_page_indices,
            cam,
            opts,
            Some((&mut *caches, hashes)),
            None,
            window.as_ref().map(|w| (w, &mut out)),
            capture_view,
            seeding.then_some(&mut plan_aux),
        );
        caches.node_items_live = false;
        let built = built?;
        caches.degraded_zoom_bucket =
            crate::text_draw::degraded_zoom_limited().then_some(degraded_bucket);
        if culling {
            caches.note_cull(&out);
            caches.camera_cull = if seeding { window } else { None };
            caches.last_plan_window = window;
        } else {
            caches.camera_cull = None;
            caches.last_plan_window = None;
        }
        let Some((plan, warnings)) = built else {
            return Ok(None);
        };
        let (plan, warnings) = (Arc::new(plan), Arc::new(warnings));
        if seeding {
            seed_camera_slot(
                caches,
                resident,
                scene,
                &plan,
                &warnings,
                plan_aux,
                visible_page_indices,
                opts_key,
                window,
                capture_view,
                cam,
                (has_outlines, has_curves),
            );
        }
        return Ok(Some((plan, warnings)));
    }

    // M1 fingerprint memo: skip the O(n) `camera_fingerprint` (which rebuilds each
    // visible page's display list) when the cheap generation key is unchanged. The
    // key is a conservative SUPERSET of the fingerprint's inputs, so an unchanged
    // key GUARANTEES an unchanged fingerprint — reuse it. A false miss (key
    // changed but fingerprint would not) only recomputes — safe. The reused
    // fingerprint still flows through `camera_get`, which re-checks `shader_time`.
    let fingerprint = if caches.fingerprint_memo_disabled {
        camera_fingerprint_with(
            scene,
            visible_page_indices,
            opts,
            hashes,
            Some(&mut *caches),
        )
    } else {
        let frame_key = camera_frame_key(resident.generation(), visible_page_indices, opts);
        let fp = match caches.camera_frame_key {
            Some((k, fp)) if k == frame_key => {
                caches.camera_fp_skips = caches.camera_fp_skips.wrapping_add(1);
                fp
            }
            _ => camera_fingerprint_with(
                scene,
                visible_page_indices,
                opts,
                hashes,
                Some(&mut *caches),
            ),
        };
        caches.camera_frame_key = Some((frame_key, fp));
        fp
    };
    if !near_pending && window_serves && (!has_captures || serving_capture.is_some()) {
        if let Some(hit) = caches.camera_get(
            captured_fingerprint(
                culled_fingerprint(fingerprint, serving_window.as_ref()),
                serving_capture.as_ref(),
            ),
            opts.shader_time_s,
        ) {
            // The cached plan still draws the far stand-ins: keep them queued.
            caches.outline_pending = prior.iter().filter_map(|&i| pending[i].take()).collect();
            return Ok(Some(hit));
        }
    }
    // A rebuild: cull to a fresh window around this view (a rebuild costs the same
    // whether or not the old window still served).
    let window = culling.then(|| cull_window_for(view));
    // A capture page's captures take the same window (one rebuild re-windows
    // both), or their own around the view when the scene is too small to cull.
    let capture_view = has_captures.then(|| crate::frame::CaptureView {
        scale: capture_scale,
        window: window.unwrap_or_else(|| cull_window_for(view)),
    });
    caches.capture_view = capture_view;
    let fingerprint = captured_fingerprint(
        culled_fingerprint(fingerprint, window.as_ref()),
        capture_view.as_ref(),
    );
    let mut cull_out = crate::frame::CullOut::default();
    // A culled build copies the items of nodes an earlier culled build of this
    // scene generation, options (zoom band included) and resources already made,
    // and keeps the ones it makes (`PlanCaches::node_items`).
    if window.is_some() {
        let key = (
            resident.generation(),
            render_options_key(opts),
            resource_key(),
        );
        if caches.node_items_key != Some(key) {
            caches.node_items.clear();
            caches.node_items_bytes = 0;
            caches.node_items_key = Some(key);
        }
        caches.node_items_live = true;
    } else if caches.node_items_key.is_some() {
        caches.node_items.clear();
        caches.node_items_bytes = 0;
        caches.node_items_key = None;
    }
    let built = crate::frame::build_camera_plan_culled_with(
        scene,
        visible_page_indices,
        cam,
        opts,
        // Reborrow so `caches` stays usable for the `camera_put` below.
        Some((&mut *caches, hashes)),
        None,
        window.as_ref().map(|w| (w, &mut cull_out)),
        capture_view,
        seeding.then_some(&mut plan_aux),
    );
    caches.node_items_live = false;
    let built = built?;
    caches.degraded_zoom_bucket =
        crate::text_draw::degraded_zoom_limited().then_some(degraded_bucket);
    caches.last_plan_window = window;
    caches.camera_cull = window;
    if window.is_some() {
        caches.note_cull(&cull_out);
    }
    let Some((plan, warnings)) = built else {
        // Defensive: build_camera_plan_cached no longer returns None for backdrop,
        // but keep the snapshot-fallback signal for any future degenerate case.
        return Ok(None);
    };
    let plan = Arc::new(plan);
    let warnings = Arc::new(warnings);
    // Brush plan P8: a plan that draws outlines stale NEAR the viewport is
    // transitional — never cache it by fingerprint; the retained slot seeded
    // below refines it (or the next frame rebuilds it). Far stand-ins alone
    // don't stop caching; a pan that brings them near bypasses the cache
    // (`near_pending`).
    if !caches.outline_refining() {
        caches.camera_put(
            fingerprint,
            opts.shader_time_s,
            plan.clone(),
            warnings.clone(),
        );
    } else {
        caches.camera_frame_key = None;
    }
    if seeding {
        seed_camera_slot(
            caches,
            resident,
            scene,
            &plan,
            &warnings,
            plan_aux,
            visible_page_indices,
            opts_key,
            window,
            capture_view,
            cam,
            (has_outlines, has_curves),
        );
        if let Some(slot) = caches.camera_slot.as_mut() {
            slot.bandless_key = Some(bandless_key);
        }
    }
    Ok(Some((plan, warnings)))
}

/// Seed the retained camera slot from a full build, for a plan whose shader
/// clock the present re-threads (so the retained items are clock-independent)
/// and within the camera byte budget (as the LRU camera slot enforces, so an
/// over-budget plan is not retained invisibly to the memory watermark).
#[allow(clippy::too_many_arguments)]
fn seed_camera_slot(
    caches: &mut PlanCaches,
    resident: &ResidentScene,
    scene: &Scene,
    plan: &Arc<CoveragePlan>,
    warnings: &Arc<PlanWarnings>,
    plan_aux: crate::frame::CameraPlanAux,
    visible_page_indices: &[usize],
    opts_key: u64,
    window: Option<crate::cull::Aabb>,
    capture_view: Option<crate::frame::CaptureView>,
    cam: &crate::frame::CameraParams,
    scene_flags: (bool, bool),
) {
    caches.camera_slot = None;
    caches.incr_camera_bytes = 0;
    let bytes = plan_bytes(plan);
    // A plan baking a clock the present can't re-thread (a procedural fill inside a layer capture: a noise, a
    // blur) is exact only at that clock: it is retained while the clock holds still, keyed on it.
    let clock = if plan_shader_time_rethreadable(plan) {
        None
    } else if caches.camera_clock_steady {
        caches.last_camera_clock
    } else {
        caches.incr_last_fallback = Some("seed: a shader clock the present can't re-thread, and it moves");
        return;
    };
    if bytes > CAMERA_SLOT_BUDGET_BYTES {
        caches.incr_last_fallback = Some("seed: the plan is over the camera byte budget");
        return;
    }
    let reuse = caches.camera_scene_index.take().map(|b| *b);
    match crate::camera_slot::CameraSlot::seed(
        scene,
        resident.generation(),
        plan.clone(),
        warnings.clone(),
        plan_aux,
        visible_page_indices,
        opts_key,
        resource_key(),
        window,
        capture_view,
        *cam,
        scene_flags,
        bytes,
        reuse,
    ) {
        Ok(mut slot) => {
            slot.clock = clock;
            caches.incr_camera_bytes = bytes;
            caches.camera_slot = Some(Box::new(slot));
        }
        Err(why) => caches.incr_last_fallback = Some(why),
    }
}

/// The retained camera slot's dev oracle: with `MODA_INCREMENTAL_VERIFY` set,
/// rebuild the frame from scratch and panic on any difference from the served
/// plan ([`crate::camera_slot::plan_divergence`]). No-op otherwise.
#[allow(clippy::too_many_arguments)]
fn debug_verify_camera_slot(
    scene: &Scene,
    visible_page_indices: &[usize],
    cam: &crate::frame::CameraParams,
    opts: &RenderOptions,
    hashes: &HashMap<String, NodeHashes>,
    cull: Option<&crate::cull::Aabb>,
    capture: Option<crate::frame::CaptureView>,
    got: &Arc<CoveragePlan>,
) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("MODA_INCREMENTAL_VERIFY").is_ok()) {
        return;
    }
    if let Some(d) = crate::camera_slot::plan_divergence(
        scene,
        visible_page_indices,
        cam,
        opts,
        hashes,
        cull,
        capture,
        got,
    ) {
        panic!(
            "MODA_INCREMENTAL_VERIFY (camera): retained plan diverged from a full rebuild — {d}"
        );
    }
}

#[cfg(test)]
#[path = "residency_patch_tests.rs"]
mod scene_patch_tests;

#[cfg(test)]
#[path = "text_defer_tests.rs"]
mod text_defer_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    /// The colour of a node's top visible solid fill.
    trait FillColor {
        fn fill_color(&self) -> Option<&str>;
    }
    impl FillColor for RawNode {
        fn fill_color(&self) -> Option<&str> {
            moda_scene::paint::top_fill_paint_json(self)
                .filter(|p| p.get("type").and_then(Value::as_str) == Some("solid"))
                .and_then(|p| p.get("color"))
                .and_then(Value::as_str)
        }
    }

    /// A one-layer solid `fills` stack.
    fn solid(color: &str) -> Value {
        json!([{"id": "fill-0", "paint": {"type": "solid", "color": color}}])
    }

    fn raw(v: Value) -> RawNode {
        match v {
            Value::Object(m) => RawNode::new(m),
            _ => panic!("node must be an object"),
        }
    }

    /// An opaque `DrawItem::Body` with a zeroed instance — the shared body shape
    /// for the residency fixtures. `node_id`, `paint`, and `geometry` are the only
    /// values that vary between tests, so they stay caller arguments.
    fn body_item(
        node_id: &str,
        paint: crate::seams::BodyPaint,
        geometry: Arc<crate::seams::TessellatedGeometry>,
    ) -> crate::seams::DrawItem {
        crate::seams::DrawItem::Body {
            node_id: node_id.to_string(),
            z: 0.0,
            phase: crate::seams::Phase::Opaque,
            paint,
            geometry,
            instance: Box::new(bytemuck::Zeroable::zeroed()),
            blend_mode: crate::seams::BlendMode::Normal,
            mask: None,
        }
    }

    /// The shared 100×100 `CoveragePlan` shell. `page_id` and `items` are the only
    /// values that vary between tests; every other field is the fixture default.
    fn coverage_plan(page_id: &str, items: Vec<crate::seams::DrawItem>) -> CoveragePlan {
        CoveragePlan {
            page_id: page_id.to_string(),
            page_index: 0,
            width_px: 100,
            height_px: 100,
            page_size: [100.0, 100.0],
            scale: 1.0,
            transparent: false,
            sample_count: 1,
            camera: [0.0, 0.0],
            items,
            masks: Vec::new(),
            camera_frame: None,
            clear_rgba: None,
            region_viewport: None,
        }
    }

    /// A `DrawItem::Text` batch with empty atlases/quads/ops — the shared text
    /// shape for the rethreadability fixtures.
    fn text_item() -> crate::seams::DrawItem {
        crate::seams::DrawItem::Text(std::sync::Arc::new(crate::seams::TextQuadBatch {
            node_id: "t".to_string(),
            z: 1.0,
            model_matrix: [0.0; 16],
            page_clip_bounds: [0.0; 4],
            ops: Vec::new(),
            emojis: Vec::new(),
            degraded: None,
            vector: None,
            distort: None,
            content_key: None,
        }))
    }

    /// A vector-lane text batch with `glyphs` per-glyph instance records,
    /// `curve_floats` curve points, and `tris_floats` solid-triangle floats —
    /// the three variable heap payloads `vector_batch_bytes` counts (plus the
    /// fixed-size `VectorOp` struct itself).
    fn vector_text_batch(
        node_id: &str,
        glyphs: usize,
        curve_floats: usize,
        tris_floats: usize,
    ) -> crate::seams::TextQuadBatch {
        use moda_text::vector::ops::VectorAnalyticRecord;
        use moda_text::vector::{VectorNodeOps, VectorOp, VectorOpPipeline};
        let op = VectorOp {
            pipeline: VectorOpPipeline::Analytic,
            color: [0.0, 0.0, 0.0, 1.0],
            opacity: 1.0,
            blur_px: 0.0,
            stroke_px: 0.0,
            additive: false,
            tris: vec![0.0_f32; tris_floats],
            records: vec![
                VectorAnalyticRecord {
                    pen_x: 0.0,
                    baseline_y: 0.0,
                    font_size_px: 16.0,
                    em_box: [0.0, 0.0, 1.0, 1.0],
                    first_curve: 0,
                    curve_count: 1,
                    // KineticText per-glyph transform: identity.
                    glyph_lin: [1.0, 0.0, 0.0, 1.0],
                    glyph_off: [0.0, 0.0],
                    glyph_opacity: 1.0,
                };
                glyphs
            ],
            gradient: None,
            paint: None,
        };
        crate::seams::TextQuadBatch {
            node_id: node_id.to_string(),
            z: 1.0,
            model_matrix: [0.0; 16],
            page_clip_bounds: [0.0; 4],
            ops: Vec::new(),
            emojis: Vec::new(),
            degraded: None,
            vector: Some(VectorNodeOps {
                ops: vec![op],
                curve_points: vec![0.0_f32; curve_floats].into(),
                gradient: None,
                emojis: Vec::new(),
            }),
            distort: None,
            content_key: None,
        }
    }

    #[test]
    fn vector_batch_bytes_track_actual_payload_not_flat_estimate() {
        use moda_text::vector::ops::VectorAnalyticRecord;

        // A realistic short label ("hello world" ≈ 10 glyphs, ~7 unique glyphs ×
        // ~30 quad segments × 6 floats ≈ 1260 curve floats) must cost far less
        // than the old flat 64 KiB/node estimate — that estimate saturated the
        // 48 MB text LRU at ~750 nodes and thrashed dense decks.
        let small = batch_bytes(&vector_text_batch("a", 10, 1260, 0));
        assert!(
            small < 16 * 1024,
            "a short vector label should be well under the old 64 KiB flat estimate, got {small}"
        );

        // Cost must SCALE with content, not be flat: a paragraph costs more.
        let big = batch_bytes(&vector_text_batch("b", 200, 60_000, 0));
        assert!(
            big > small,
            "a paragraph ({big}) must cost more than a short label ({small})"
        );

        // Each variable payload is pinned independently at `len * size_of::<T>()`
        // so removing any one term is caught (curve points otherwise dominate).
        let base = batch_bytes(&vector_text_batch("c", 10, 1000, 0));
        let more_curves = batch_bytes(&vector_text_batch("c", 10, 2000, 0));
        assert_eq!(
            more_curves - base,
            1000 * std::mem::size_of::<f32>(),
            "curve-point delta"
        );
        let more_glyphs = batch_bytes(&vector_text_batch("c", 60, 1000, 0));
        assert_eq!(
            more_glyphs - base,
            50 * std::mem::size_of::<VectorAnalyticRecord>(),
            "glyph-record delta"
        );
        let more_tris = batch_bytes(&vector_text_batch("c", 10, 1000, 300));
        assert_eq!(
            more_tris - base,
            300 * std::mem::size_of::<f32>(),
            "tess-triangle delta"
        );
    }

    /// A multi-op vector batch: one `VectorOp` per entry in `op_glyphs`, each
    /// carrying that many instance records (no curves/tris). Lets a test pin the
    /// per-op term and prove the accounting sums ALL ops, not just the first.
    fn vector_text_batch_multi_op(
        node_id: &str,
        op_glyphs: &[usize],
    ) -> crate::seams::TextQuadBatch {
        use moda_text::vector::ops::VectorAnalyticRecord;
        use moda_text::vector::{VectorNodeOps, VectorOp, VectorOpPipeline};
        let rec = VectorAnalyticRecord {
            pen_x: 0.0,
            baseline_y: 0.0,
            font_size_px: 16.0,
            em_box: [0.0, 0.0, 1.0, 1.0],
            first_curve: 0,
            curve_count: 1,
            // KineticText per-glyph transform: identity.
            glyph_lin: [1.0, 0.0, 0.0, 1.0],
            glyph_off: [0.0, 0.0],
            glyph_opacity: 1.0,
        };
        let ops = op_glyphs
            .iter()
            .map(|&n| VectorOp {
                pipeline: VectorOpPipeline::Analytic,
                color: [0.0, 0.0, 0.0, 1.0],
                opacity: 1.0,
                blur_px: 0.0,
                stroke_px: 0.0,
                additive: false,
                tris: Vec::new(),
                records: vec![rec; n],
                gradient: None,
                paint: None,
            })
            .collect();
        let mut batch = vector_text_batch(node_id, 0, 0, 0);
        batch.vector = Some(VectorNodeOps {
            ops,
            curve_points: Default::default(),
            gradient: None,
            emojis: Vec::new(),
        });
        batch
    }

    #[test]
    fn vector_batch_bytes_sums_all_ops_and_the_per_op_struct() {
        use moda_text::vector::VectorOp;
        // Same total records (60) and zero curves/tris, split 1-op vs 2-op. The
        // ONLY difference is one extra `VectorOp` struct — which pins that term
        // AND proves the loop sums every op (a "first op only" regression would
        // undercount the 2-op batch's records and break this equality).
        let one = batch_bytes(&vector_text_batch_multi_op("m", &[60]));
        let two = batch_bytes(&vector_text_batch_multi_op("m", &[30, 30]));
        assert_eq!(
            two - one,
            std::mem::size_of::<VectorOp>(),
            "second op adds exactly one VectorOp struct; records summed across ops"
        );
    }

    /// A Degraded-tier text batch owning a `w`×`h` RGBA raster.
    fn degraded_text_batch(id: &str, w: u32, h: u32) -> crate::seams::TextQuadBatch {
        let mut b = vector_text_batch(id, 0, 0, 0);
        b.vector = None;
        b.degraded = Some(crate::seams::DegradedQuad {
            raster: Arc::new(moda_text::degraded::DegradedRaster {
                rgba: vec![0; (w * h * 4) as usize],
                width: w,
                height: h,
                logical_w: w as f64 / 2.0,
                logical_h: h as f64 / 2.0,
                quality_scale: 2.0,
                pad_top: 0.0,
                pad_bottom: 0.0,
                fill_alpha: 1.0,
            }),
            offset: [0.0, 0.0],
            opacity: 1.0,
        });
        b
    }

    #[test]
    fn batch_bytes_counts_a_degraded_raster() {
        let b = degraded_text_batch("d", 200, 100);
        assert!(batch_bytes(&b) >= 200 * 100 * 4, "the raster is the batch's heap");
    }

    /// The text budget the eviction tests run at (the real one is larger; batches this size keep them light).
    const TEST_TEXT_BUDGET: usize = 48 * 1024 * 1024;

    #[test]
    fn text_lru_bounds_degraded_rasters_by_their_bytes() {
        // 16k texts planned before their fonts arrived (Degraded-tier rasters,
        // ~80 KB each) kept 1.3 GB in a cache that counted 4 MB. Counted, the
        // cache stays inside its budget.
        let mut caches = PlanCaches::default();
        caches.text_budget_bytes = TEST_TEXT_BUDGET;
        for i in 0..2_000u64 {
            let id = format!("n{i}");
            caches.text_put(&id, i, &degraded_text_batch(&id, 140, 140));
        }
        let held: usize = caches
            .text
            .values()
            .map(|e| e.batch.degraded.as_ref().map_or(0, |d| d.raster.rgba.len()))
            .sum();
        assert!(caches.text_bytes <= TEST_TEXT_BUDGET);
        assert!(held <= TEST_TEXT_BUDGET, "rasters held: {held}");
        assert!(caches.evictions > 0);
    }

    #[test]
    fn node_items_cache_stays_inside_its_budget() {
        // A culled camera build keeps each node's items for later windows. At a
        // 100k-node fit that was a deep copy of every node's text batches.
        let mut caches = PlanCaches::default();
        caches.node_items_live = true;
        let mut kept = 0;
        for i in 0..5_000u64 {
            let id = format!("n{i}");
            let item = crate::seams::DrawItem::Text(std::sync::Arc::new(degraded_text_batch(&id, 100, 100)));
            caches.node_items_put(&id, i, 1, std::slice::from_ref(&item), &[], 0);
            kept += usize::from(caches.node_items_get(&id, i).is_some());
        }
        let (entries, bytes) = caches.node_items_stats();
        assert_eq!(entries, kept);
        assert!(bytes <= NODE_ITEMS_BUDGET_BYTES, "{bytes}");
        assert!(entries > 100 && entries < 5_000, "{entries}");
        // A different generation starts the cache over.
        caches.node_items.clear();
        caches.node_items_bytes = 0;
        assert_eq!(caches.node_items_stats(), (0, 0));
    }

    #[test]
    fn text_lru_holds_a_dense_richtext_deck_without_thrash() {
        // Regression guard for the intermittent-drag-jank fix: ~1k short vector
        // text nodes must fit the 48 MB text budget with zero evictions. The old
        // flat 64 KiB estimate evicted past ~750 nodes, forcing a periodic full
        // text re-prepare mid-drag (the ~1s frame spikes).
        let mut caches = PlanCaches::default();
        for i in 0..1024u64 {
            let id = format!("n{i}");
            let batch = vector_text_batch(&id, 12, 1500, 0);
            caches.text_put(&id, i, &batch);
        }
        assert_eq!(caches.text.len(), 1024, "all nodes should stay resident");
        assert_eq!(caches.evictions, 0, "no eviction thrash under the budget");
    }

    #[test]
    fn text_put_keeps_vector_text_bytes_exact_across_replacement() {
        // Replacing a cached vector batch must subtract the OLD size and add the
        // NEW one exactly — both directions. A bug in the subtract/replace path
        // would silently drift `text_bytes` from the retained heap and skew every
        // future eviction decision.
        let mut caches = PlanCaches::default();
        let big = vector_text_batch("n", 200, 60_000, 400);
        let small = vector_text_batch("n", 10, 1_000, 0);

        caches.text_put("n", 1, &big);
        assert_eq!(caches.text_bytes, batch_bytes(&big));

        caches.text_put("n", 2, &small); // larger → smaller
        assert_eq!(caches.text_bytes, batch_bytes(&small));
        assert_eq!(caches.text.len(), 1, "replacement keeps a single entry");

        caches.text_put("n", 3, &big); // smaller → larger
        assert_eq!(caches.text_bytes, batch_bytes(&big));
        assert_eq!(caches.evictions, 0, "a single replaced entry never evicts");
    }

    #[test]
    fn text_eviction_trims_to_the_low_water_mark() {
        // Crossing the budget trims to 7/8 of it (oldest first), so the next inserts don't rescan the cache: an
        // eviction per insert once text outgrew the budget made a frame O(n²) in text nodes.
        const CURVE_FLOATS: usize = 256 * 1024; // ~1 MiB per batch
        let mut caches = PlanCaches::default();
        caches.text_budget_bytes = TEST_TEXT_BUDGET;
        let mut first_trim = None;
        for i in 0..80u64 {
            caches.begin_frame();
            let id = format!("t{i}");
            let before = caches.evictions;
            caches.text_put(&id, i, &vector_text_batch(&id, 1, CURVE_FLOATS, 0));
            if caches.evictions > before && first_trim.is_none() {
                first_trim = Some(i);
                assert!(
                    caches.text_bytes <= (TEST_TEXT_BUDGET / 8 * 7),
                    "an over-budget insert trims to the low-water mark, got {}",
                    caches.text_bytes
                );
                // The next insert fits under the budget again: no rescan.
                let trimmed = caches.evictions;
                let id = format!("t{i}b");
                caches.text_put(&id, i, &vector_text_batch(&id, 1, CURVE_FLOATS, 0));
                assert_eq!(caches.evictions, trimmed, "no eviction right after a trim");
            }
            assert!(caches.text_bytes <= TEST_TEXT_BUDGET);
        }
        first_trim.expect("80 MiB of text crosses the 48 MiB budget");
        // The oldest entries went first; the newest is still cached.
        assert!(!caches.text.contains_key("t0"));
        assert!(caches.text.contains_key("t79"));
    }

    #[test]
    fn text_lru_evicts_when_aggregate_vector_bytes_exceed_budget() {
        // Eviction boundary (distinct from the 1k-small-node no-thrash case): the
        // AGGREGATE of a few vector batches crosses TEST_TEXT_BUDGET and evicts,
        // and `text_bytes` stays within budget afterward. Each batch is well under
        // budget individually (~28 MiB = 7M f32 curve points); two of them
        // (~56 MiB) cross the 48 MiB budget. Kept to two batches to bound peak
        // test memory (each `text_put` clones the batch into the cache).
        const CURVE_FLOATS: usize = 7_000_000;
        let mut caches = PlanCaches::default();
        caches.text_budget_bytes = TEST_TEXT_BUDGET;
        let per = batch_bytes(&vector_text_batch("x", 1, CURVE_FLOATS, 0));
        assert!(
            per < TEST_TEXT_BUDGET,
            "one batch must fit individually ({per})"
        );
        assert!(
            2 * per > TEST_TEXT_BUDGET,
            "two batches must exceed the budget"
        );
        for i in 0..2u64 {
            let id = format!("big{i}");
            caches.text_put(&id, i, &vector_text_batch(&id, 1, CURVE_FLOATS, 0));
        }
        assert!(
            caches.evictions >= 1,
            "aggregate over-budget puts must evict"
        );
        assert!(
            caches.text_bytes <= TEST_TEXT_BUDGET,
            "text_bytes must stay within budget after eviction, got {}",
            caches.text_bytes
        );
    }

    /// PLAN §4: page plans are byte-bounded (≤16 MB) with LRU eviction, and
    /// accounting stays exact across replacement. Eviction is never
    /// correctness-bearing — an evicted page just re-plans.
    /// Regression for the camera-cache-miss-every-frame throughput wall: a dense
    /// viewport's combined plan whose bytes exceed the 16 MB PAGE budget but fit
    /// the 128 MB CAMERA-slot budget MUST be retained (cached), not refused. Before
    /// the dedicated camera budget, such a plan was rejected and re-combined every
    /// frame (a 2k-node pan ran ~5 fps because of this).
    #[test]
    fn camera_plan_over_page_budget_is_retained() {
        use crate::seams::{BodyPaint, TessellatedGeometry};
        // ~20 MB of f32 geometry: over PAGE_PLAN_BUDGET_BYTES (16 MB), under
        // CAMERA_PLAN_BUDGET_BYTES (128 MB).
        let positions = 5_000_000;
        let plan = Arc::new(coverage_plan(
            "cam",
            vec![body_item(
                "n",
                BodyPaint::Solid,
                Arc::new(TessellatedGeometry {
                    positions: vec![0.0; positions],
                    ..Default::default()
                }),
            )],
        ));
        let bytes = plan_bytes(&plan);
        assert!(
            bytes > PAGE_PLAN_BUDGET_BYTES,
            "test plan must exceed the page budget ({bytes} <= {PAGE_PLAN_BUDGET_BYTES})"
        );
        assert!(
            bytes <= CAMERA_PLAN_BUDGET_BYTES,
            "test plan must fit the camera budget ({bytes} > {CAMERA_PLAN_BUDGET_BYTES})"
        );
        let warnings = Arc::new(PlanWarnings::default());
        let mut c = PlanCaches::default();
        c.camera_put(123, 0.0, plan, warnings);
        assert!(
            c.camera_get(123, 0.0).is_some(),
            "a dense-viewport plan over the 16 MB page budget but under the 128 MB \
             camera budget must be RETAINED (else it re-combines every frame)"
        );
        assert_eq!(c.camera_hits, 1);
    }

    #[test]
    fn page_plan_byte_budget_lru() {
        fn big_plan(page_id: &str, positions: usize) -> Arc<CoveragePlan> {
            use crate::seams::{BodyPaint, TessellatedGeometry};
            Arc::new(coverage_plan(
                page_id,
                vec![body_item(
                    "n",
                    BodyPaint::Solid,
                    Arc::new(TessellatedGeometry {
                        positions: vec![0.0; positions],
                        ..Default::default()
                    }),
                )],
            ))
        }
        let warnings = Arc::new(PlanWarnings::default());
        let mut c = PlanCaches::default();
        // ~6 MB per plan; three plans cross the 16 MB budget → oldest evicted.
        let n = 1_500_000;
        for (i, id) in ["a", "b", "c"].iter().enumerate() {
            c.begin_frame();
            c.page_put(id, i as u64, 0.0, big_plan(id, n), warnings.clone());
        }
        assert!(
            c.page_bytes <= PAGE_PLAN_BUDGET_BYTES,
            "page bytes over budget: {}",
            c.page_bytes
        );
        assert!(c.page_get("a", 0, 0.0).is_none(), "oldest must be evicted");
        assert!(c.page_get("b", 1, 0.0).is_some());
        assert!(c.page_get("c", 2, 0.0).is_some());
        assert!(c.evictions >= 1);
        // Replacement must not double-count.
        let before = c.page_bytes;
        c.begin_frame();
        c.page_put("c", 3, 0.0, big_plan("c", n), warnings.clone());
        assert_eq!(c.page_bytes, before, "replace must swap bytes, not add");
        // clear() resets the byte ledger.
        c.clear();
        assert_eq!(c.page_bytes, 0);
    }

    /// The camera fingerprint must fold freeform canvas-root nodes
    /// (they now render), but ONLY in freeform layout (they aren't rendered
    /// otherwise). Without folding, editing/adding a floating node with pages
    /// unchanged would serve a stale cached camera plan.
    #[test]
    fn camera_fingerprint_folds_freeform_canvas_root_only() {
        let mk = |mode: &str, with_float: bool| -> (Scene, HashMap<String, NodeHashes>) {
            let mut nodes = vec![json!({"id":"onpage","type":"rectangle","x":0.0,"y":0.0,
                "width":10.0,"height":10.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]})];
            let mut roots: Vec<&str> = vec![];
            if with_float {
                nodes.push(json!({"id":"floata","type":"rectangle","x":900.0,"y":0.0,
                    "width":10.0,"height":10.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00f"}}]}));
                roots.push("floata");
            }
            let doc = json!({
                "layoutMode": mode,
                "nodes": nodes,
                "pages": [{"id":"p1","width":100.0,"height":100.0,"nodeIds":["onpage"],
                           "freeformX":0.0,"freeformY":0.0}],
                "canvasRootNodeIds": roots,
                "config": {},
            });
            let scene = Scene::load(&doc).unwrap();
            let hashes = node_hashes_for_scene(&scene);
            (scene, hashes)
        };
        let opts = RenderOptions {
            text_masks_available: Some(true),
            ..Default::default()
        };
        // Freeform: a floating node is rendered, so it must change the fingerprint.
        let (a, ha) = mk("freeform", false);
        let (b, hb) = mk("freeform", true);
        assert_ne!(
            camera_fingerprint(&a, &[0], &opts, &ha),
            camera_fingerprint(&b, &[0], &opts, &hb),
            "freeform floating node folds into the camera fingerprint"
        );
        // Vertical: the same floating node is NOT rendered, so it must NOT shift
        // the fingerprint (no spurious cache invalidation).
        let (c, hc) = mk("vertical", false);
        let (d, hd) = mk("vertical", true);
        assert_eq!(
            camera_fingerprint(&c, &[0], &opts, &hc),
            camera_fingerprint(&d, &[0], &opts, &hd),
            "outside freeform the canvas root is neither rendered nor fingerprinted"
        );
        // Reordering two same-content floating nodes changes the fingerprint:
        // paint order matters, and the fold writes node ids in canvas z-order.
        let two = |order: [&str; 2]| -> (Scene, HashMap<String, NodeHashes>) {
            let doc = json!({
                "layoutMode": "freeform",
                "nodes": [
                    {"id":"fa","type":"rectangle","x":900.0,"y":0.0,"width":10.0,"height":10.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00f"}}]},
                    {"id":"fb","type":"rectangle","x":920.0,"y":0.0,"width":10.0,"height":10.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00f"}}]}
                ],
                "pages": [{"id":"p1","width":100.0,"height":100.0,"nodeIds":[],
                           "freeformX":0.0,"freeformY":0.0}],
                "canvasRootNodeIds": order,
                "config": {},
            });
            let scene = Scene::load(&doc).unwrap();
            let hashes = node_hashes_for_scene(&scene);
            (scene, hashes)
        };
        let (e, he) = two(["fa", "fb"]);
        let (f, hf) = two(["fb", "fa"]);
        assert_ne!(
            camera_fingerprint(&e, &[0], &opts, &he),
            camera_fingerprint(&f, &[0], &opts, &hf),
            "reordering floating nodes changes the camera fingerprint (paint order)"
        );
    }

    /// A viewport panned to the pasteboard (NO visible pages) still
    /// renders floating nodes, so the camera fingerprint must fold the
    /// plan-relevant render options even with `visible_page_indices == []` —
    /// otherwise a sample-count / option change (or an asset/font settling)
    /// would serve a stale cached plan. Page_fingerprint folds these, but it
    /// never runs when there are no visible pages.
    #[test]
    fn camera_fingerprint_folds_options_for_canvas_root_only_view() {
        let doc = json!({
            "layoutMode": "freeform",
            "nodes": [
                {"id":"floata","type":"rectangle","x":900.0,"y":900.0,
                 "width":10.0,"height":10.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00f"}}]}
            ],
            "pages": [{"id":"p1","width":100.0,"height":100.0,"nodeIds":[],
                       "freeformX":0.0,"freeformY":0.0}],
            "canvasRootNodeIds": ["floata"],
            "config": {},
        });
        let scene = Scene::load(&doc).unwrap();
        let hashes = node_hashes_for_scene(&scene);
        let opts4 = RenderOptions {
            sample_count: 4,
            text_masks_available: Some(true),
            ..Default::default()
        };
        let opts1 = RenderOptions {
            sample_count: 1,
            text_masks_available: Some(true),
            ..Default::default()
        };
        // No visible pages, but the floating node renders — the option change
        // must still re-fingerprint.
        assert_ne!(
            camera_fingerprint(&scene, &[], &opts4, &hashes),
            camera_fingerprint(&scene, &[], &opts1, &hashes),
            "sample_count change invalidates the canvas-root-only camera plan"
        );
    }

    #[test]
    fn hash_is_deterministic() {
        let n = raw(
            json!({"id": "a", "type": "rectangle", "x": 1.0, "width": 10.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}]}),
        );
        assert_eq!(node_hashes(&n), node_hashes(&n));
        let m = raw(
            json!({"id": "a", "type": "rectangle", "x": 1.0, "width": 10.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}]}),
        );
        assert_eq!(node_hashes(&n), node_hashes(&m));
    }

    #[test]
    fn transform_only_props_do_not_change_geom_or_text_hash() {
        let a = raw(
            json!({"id": "a", "type": "rectangle", "x": 1.0, "y": 2.0, "rotation": 0.0, "scale": 1.0, "width": 10.0}),
        );
        let b = raw(
            json!({"id": "a", "type": "rectangle", "x": 99.0, "y": -4.0, "rotation": 45.0, "scale": 2.0, "width": 10.0}),
        );
        let (ha, hb) = (node_hashes(&a), node_hashes(&b));
        assert_eq!(
            ha.geom, hb.geom,
            "x/y/rotation/scale must not re-key tessellation"
        );
        assert_eq!(ha.text, hb.text);
        assert_ne!(
            ha.full, hb.full,
            "full hash must still see transforms (repaint identity)"
        );
    }

    #[test]
    fn paint_only_props_do_not_change_geom_hash_but_do_change_text_hash() {
        let a = raw(
            json!({"id": "a", "type": "rectangle", "width": 10.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}], "opacity": 1.0}),
        );
        let b = raw(
            json!({"id": "a", "type": "rectangle", "width": 10.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}], "opacity": 0.5}),
        );
        let (ha, hb) = (node_hashes(&a), node_hashes(&b));
        assert_eq!(
            ha.geom, hb.geom,
            "fill/opacity must not re-key tessellation"
        );
        assert_ne!(ha.text, hb.text, "paint IS baked into text quads");
        assert_ne!(ha.full, hb.full);
    }

    #[test]
    fn geometry_class_props_re_key() {
        // Stroke width (the browser's latent stale-geometry hazard — MAP §5.2),
        // dash, cap/join, fillRule, stroke align, width, points, type,
        // distort, the fill's paint type: ALL must change the geom hash.
        let rect = |extra: Value, stroke: Value| {
            let mut layer = json!({"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": 1.0});
            for (k, v) in stroke.as_object().unwrap() {
                layer[k] = v.clone();
            }
            let mut n = json!({"id": "a", "type": "rectangle", "width": 10.0, "height": 5.0, "strokes": [layer]});
            for (k, v) in extra.as_object().unwrap() {
                n[k] = v.clone();
            }
            n
        };
        let none = json!({});
        let base = rect(none.clone(), none.clone());
        let variants = vec![
            rect(none.clone(), json!({"width": 3.0})),
            rect(none.clone(), json!({"dash": [5.0, 5.0]})),
            rect(none.clone(), json!({"cap": "round"})),
            rect(none.clone(), json!({"join": "bevel"})),
            rect(json!({"fillRule": "evenodd"}), none.clone()),
            rect(none.clone(), json!({"align": "inner"})),
            rect(json!({"width": 12.0}), none.clone()),
            rect(json!({"type": "ellipse"}), none.clone()),
            rect(
                json!({"distort": {"type": "matrix", "matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2, 0.0, 1.0]}}),
                none.clone(),
            ),
            rect(
                json!({"fills": [{"id": "fill-0", "paint": {"type": "image", "placement": {}}}]}),
                none.clone(),
            ),
            // Trim paths + dash offset are geometry inputs (baked into stroke
            // triangles), so they MUST re-key tessellation or animating them shows
            // stale frames. Guards the invariant that they stay out of the
            // TRANSFORM_ONLY / PAINT_ONLY deny-lists.
            rect(json!({"trimPathStart": 0.25}), none.clone()),
            rect(json!({"trimPathEnd": 0.75}), none.clone()),
            rect(json!({"trimPathOffset": 0.5}), none.clone()),
            rect(none.clone(), json!({"dashOffset": 3.0})),
        ];
        let h0 = node_hashes(&raw(base)).geom;
        for v in variants {
            assert_ne!(
                h0,
                node_hashes(&raw(v.clone())).geom,
                "geometry-class prop failed to re-key: {v}"
            );
        }
    }

    #[test]
    fn stroke_paint_route_presence_re_keys_geom_but_its_value_does_not() {
        // Stroke tessellation is gated on a stroke existing, so gaining/losing a
        // stroke layer MUST re-key the tess cache (else a path given its first
        // stroke keeps the cached band-less geometry), while a colour-to-colour
        // edit must NOT. The projection keeps the stroke's paint TYPE (a
        // gradient stroke routes differently), so a type swap re-keys too
        // (conservative; it used to key only on "has a route").
        let path = |paint: Value| {
            raw(json!({"id": "a", "type": "path", "points": [],
                "strokes": [{"id": "stroke-0", "paint": paint, "width": 2}]}))
        };
        let stops = json!([{"offset": 0, "color": "#ff0000"}, {"offset": 1, "color": "#0000ff"}]);
        let none = raw(json!({"id": "a", "type": "path", "points": []}));
        let red = path(json!({"type": "solid", "color": "#ff0000"}));
        let green = path(json!({"type": "solid", "color": "#00ff00"}));
        let gradient = path(
            json!({"type": "linear", "start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 0},
            "stops": stops}),
        );
        let gradient2 = path(
            json!({"type": "linear", "start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 100},
            "stops": [{"offset": 0, "color": "#00ff00"}, {"offset": 1, "color": "#000000"}]}),
        );
        assert_ne!(
            node_hashes(&none).geom,
            node_hashes(&red).geom,
            "gaining a stroke must re-key tessellation"
        );
        assert_ne!(
            node_hashes(&none).geom,
            node_hashes(&gradient).geom,
            "gaining a gradient stroke must re-key tessellation"
        );
        assert_eq!(
            node_hashes(&red).geom,
            node_hashes(&green).geom,
            "a stroke colour edit stays paint-only"
        );
        assert_ne!(
            node_hashes(&red).geom,
            node_hashes(&gradient).geom,
            "swapping a solid stroke for a gradient re-keys (the route changes)"
        );
        assert_eq!(
            node_hashes(&gradient).geom,
            node_hashes(&gradient2).geom,
            "a gradient's placement / stops edit stays paint-only"
        );
    }

    #[test]
    fn unknown_props_over_invalidate() {
        let a = raw(json!({"id": "a", "type": "rectangle", "width": 10.0}));
        let b = raw(json!({"id": "a", "type": "rectangle", "width": 10.0, "someFutureProp": 1.0}));
        assert_ne!(
            node_hashes(&a).geom,
            node_hashes(&b).geom,
            "unknown props must over-invalidate (deny-list law)"
        );
    }

    /// R5: fingerprint sensitivity — z-order, add/remove, prop edits, page
    /// props, scale all re-fingerprint; an identical scene does not.
    #[test]
    fn page_fingerprint_sensitivity() {
        let mk =
            |nodes: Value, node_ids: Value, page_w: f64| -> (Scene, HashMap<String, NodeHashes>) {
                let doc = json!({
                    "nodes": nodes,
                    "pages": [{"id": "p1", "width": page_w, "height": 400.0, "nodeIds": node_ids}],
                    "config": {},
                });
                let scene = Scene::load(&doc).unwrap();
                let mut hashes = HashMap::default();
                for id in scene.node_order() {
                    if let Some(n) = scene.effective_node(id) {
                        hashes.insert(id.clone(), node_hashes(n));
                    }
                }
                (scene, hashes)
            };
        let nodes = json!([
            {"id": "a", "type": "rectangle", "x": 0.0, "y": 0.0, "width": 50.0, "height": 50.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]},
            {"id": "b", "type": "ellipse", "x": 60.0, "y": 0.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0f0"}}]},
        ]);
        let opts = RenderOptions::default();
        let (s0, h0) = mk(nodes.clone(), json!(["a", "b"]), 800.0);
        let fp0 = page_fingerprint(&s0, 0, &opts, &h0);
        // Identical scene → identical fingerprint.
        let (s1, h1) = mk(nodes.clone(), json!(["a", "b"]), 800.0);
        assert_eq!(fp0, page_fingerprint(&s1, 0, &opts, &h1));
        // Z-reorder.
        let (s2, h2) = mk(nodes.clone(), json!(["b", "a"]), 800.0);
        assert_ne!(
            fp0,
            page_fingerprint(&s2, 0, &opts, &h2),
            "z-order must re-fingerprint"
        );
        // Remove.
        let (s3, h3) = mk(nodes.clone(), json!(["a"]), 800.0);
        assert_ne!(
            fp0,
            page_fingerprint(&s3, 0, &opts, &h3),
            "remove must re-fingerprint"
        );
        // Prop edit (even transform-only — repaint identity).
        let nodes_moved = json!([
            {"id": "a", "type": "rectangle", "x": 5.0, "y": 0.0, "width": 50.0, "height": 50.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]},
            {"id": "b", "type": "ellipse", "x": 60.0, "y": 0.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0f0"}}]},
        ]);
        let (s4, h4) = mk(nodes_moved, json!(["a", "b"]), 800.0);
        assert_ne!(
            fp0,
            page_fingerprint(&s4, 0, &opts, &h4),
            "node move must re-fingerprint"
        );
        // Page prop.
        let (s5, h5) = mk(nodes.clone(), json!(["a", "b"]), 900.0);
        assert_ne!(
            fp0,
            page_fingerprint(&s5, 0, &opts, &h5),
            "page size must re-fingerprint"
        );
        // Render options.
        let mut opts2 = RenderOptions::default();
        opts2.scale = 2.0;
        assert_ne!(
            fp0,
            page_fingerprint(&s0, 0, &opts2, &h0),
            "scale must re-fingerprint"
        );
    }

    /// Moving a `native-children` generated host (a chart) must
    /// re-fingerprint the page/camera plan. The host's expanded children live
    /// in host-LOCAL coordinates — their own hashes are position-invariant, and
    /// the host is expanded away (never its own DL item), so the host transform
    /// reaches the children's draw-item model matrices ONLY through
    /// `transform_parent_chain`. Without folding the host hash here, a host MOVE
    /// left the fingerprint unchanged and the retained plan was reused with
    /// stale chart pixels.
    #[test]
    fn page_fingerprint_folds_native_children_host_transform() {
        // The transient child is host-LOCAL, so its own hash is invariant to ALL
        // host transform fields — the fold must cover the whole transform surface
        // (x/y/rotation/scale), not just translation. Parameterize all four.
        let mk =
            |x: f64, y: f64, rotation: f64, scale: f64| -> (Scene, HashMap<String, NodeHashes>) {
                let doc = json!({
                    "nodes": [{
                        "id": "ch1", "type": "generated", "generatedKind": "chart",
                        "x": x, "y": y, "width": 300.0, "height": 200.0,
                        "rotation": rotation, "scale": scale,
                        "renderMode": "native-children",
                        "generatedSource": {
                            "kind": "chart", "chartVersion": 1, "spec": {"type": "bar"},
                            "data": {"columns": [], "rows": [], "series": [],
                                     "categories": [], "valueRange": [0, 0]}
                        }
                    }],
                    "pages": [{"id": "p1", "width": 800.0, "height": 400.0, "nodeIds": ["ch1"]}],
                    "config": {},
                });
                let scene = Scene::load(&doc).unwrap();
                let hashes = node_hashes_for_scene(&scene);
                (scene, hashes)
            };
        let opts = RenderOptions::default();
        let (s0, h0) = mk(40.0, 40.0, 0.0, 1.0);
        let fp0 = page_fingerprint(&s0, 0, &opts, &h0);

        // Setup sanity: the host expands into a host-LOCAL transient child that is
        // what actually lands in the display list (the host never does). Whether
        // the chart renders 1 placeholder (empty data, here) or N real shapes is
        // irrelevant: the fold runs once per DL item keyed on `transient_parent_of`
        // → host, so the placeholder exercises the identical code path.
        let dl = moda_scene::build_display_list(&s0, "p1");
        let child_id = dl
            .items
            .iter()
            .map(|i| i.node_id.as_str())
            .find(|id| s0.transient_parent_of().get(*id).map(String::as_str) == Some("ch1"))
            .expect("chart host must expand into a transient child in the display list");
        // The child's OWN hash is invariant to host transform — proving the bug
        // premise: a host move cannot be caught by the child's hash alone.
        let (s_moved, _) = mk(140.0, 90.0, 45.0, 2.0);
        assert_eq!(
            node_hashes(s0.effective_node(child_id).unwrap()).full,
            node_hashes(s_moved.effective_node(child_id).unwrap()).full,
            "transient child hash must be host-transform-invariant (bug premise)"
        );

        // Identical scene → identical fingerprint.
        let (s1, h1) = mk(40.0, 40.0, 0.0, 1.0);
        assert_eq!(fp0, page_fingerprint(&s1, 0, &opts, &h1));
        // Changing ONLY a single host transform field MUST re-fingerprint — one
        // case per field, since the child hash carries none of them.
        for (label, (s, h)) in [
            ("x", mk(140.0, 40.0, 0.0, 1.0)),
            ("y", mk(40.0, 140.0, 0.0, 1.0)),
            ("rotation", mk(40.0, 40.0, 30.0, 1.0)),
            ("scale", mk(40.0, 40.0, 0.0, 1.5)),
        ] {
            assert_ne!(
                fp0,
                page_fingerprint(&s, 0, &opts, &h),
                "chart host {label} change must re-fingerprint"
            );
            // The camera plan folds page fingerprints, so it must track it too.
            assert_ne!(
                camera_fingerprint(&s0, &[0], &opts, &h0),
                camera_fingerprint(&s, &[0], &opts, &h),
                "camera fingerprint must track the chart host {label} change"
            );
        }
    }

    /// S2-min: patch-lane guard rails — every doubt path errors (resync).
    #[test]
    fn apply_props_guard_rails() {
        let doc = json!({
            "nodes": [{"id": "a", "type": "rectangle", "x": 0.0, "width": 50.0, "height": 50.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]}],
            "pages": [{"id": "p1", "width": 800.0, "height": 400.0, "nodeIds": ["a"]}],
            "config": {},
        });
        let json = moda_scene::value::to_string(&doc).unwrap();
        let mut r = ResidentScene::default();
        assert!(
            r.apply_props("{\"nodes\":{}}").is_err(),
            "no resident scene → error"
        );
        r.update(&json).unwrap();
        assert!(
            r.apply_props("{\"nodes\":{\"missing\":{\"fills\":[{\"id\":\"fill-0\",\"paint\":{\"type\":\"solid\",\"color\":\"#000\"}}]}}}")
                .is_err(),
            "unknown id → error"
        );
        assert!(
            r.apply_props("{\"nodes\":{\"a\":{\"type\":\"ellipse\"}}}")
                .is_err(),
            "structural key → error"
        );
        assert!(r.apply_props("not json").is_err(), "malformed → error");
        // Happy path: fill patch lands in the scene + re-hashes the node.
        let before = *r.hashes().unwrap().get("a").unwrap();
        r.apply_props("{\"nodes\":{\"a\":{\"fills\":[{\"id\":\"fill-0\",\"paint\":{\"type\":\"solid\",\"color\":\"#00ff00\"}}]}}}")
            .unwrap();
        let scene = r.scene().unwrap();
        assert_eq!(
            scene.effective_node("a").unwrap().fill_color(),
            Some("#00ff00")
        );
        let after = *r.hashes().unwrap().get("a").unwrap();
        assert_ne!(before.full, after.full, "patch must re-hash");
        assert_eq!(
            before.geom, after.geom,
            "fill is paint-only — tess key survives"
        );
        // A full feed of the ORIGINAL string must re-parse (patched flag),
        // restoring the canonical fill.
        r.update(&json).unwrap();
        assert_eq!(
            r.scene().unwrap().effective_node("a").unwrap().fill_color(),
            Some("#f00")
        );
    }

    /// A patch can mutate an earlier node then bail on a later
    /// one (here: a structural key), leaving a PARTIAL mutation resident. Because
    /// `patched` is set BEFORE the mutation loop, the host's reject-heal —
    /// re-feeding the SAME scene JSON — re-parses instead of hash-skipping, so the
    /// partial mutation is healed. Guards the `set_node_props` reject-heal contract
    /// both the snapshot and camera patch lanes rely on.
    #[test]
    fn apply_props_partial_mutation_then_error_heals_on_refeed() {
        let doc = json!({
            "nodes": [
                {"id": "a", "type": "rectangle", "x": 0.0, "width": 50.0, "height": 50.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]},
                {"id": "b", "type": "rectangle", "x": 0.0, "width": 50.0, "height": 50.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0f0"}}]}
            ],
            "pages": [{"id": "p1", "width": 800.0, "height": 400.0, "nodeIds": ["a", "b"]}],
            "config": {},
        });
        let json = moda_scene::value::to_string(&doc).unwrap();
        let mut r = ResidentScene::default();
        r.update(&json).unwrap();
        // "a" is processed before "b" (id order), so its fill is already changed
        // when the structural-key bail on "b" hits — a partial mutation.
        assert!(
            r.apply_props(
                "{\"nodes\":{\"a\":{\"fills\":[{\"id\":\"fill-0\",\"paint\":{\"type\":\"solid\",\"color\":\"#00ff00\"}}]},\"b\":{\"type\":\"ellipse\"}}}"
            )
            .is_err(),
            "a structural key on a later node still errors"
        );
        assert_eq!(
            r.scene().unwrap().effective_node("a").unwrap().fill_color(),
            Some("#00ff00"),
            "the partial mutation IS resident (a was mutated before b bailed)"
        );
        // The host heals a rejected patch by re-feeding the SAME JSON. Without
        // `patched` set on the partial mutation, the hash-skip in `update` would
        // strand "a" at #00ff00.
        r.update(&json).unwrap();
        assert_eq!(
            r.scene().unwrap().effective_node("a").unwrap().fill_color(),
            Some("#f00"),
            "re-feeding the same JSON re-parses and restores canonical (no hash-skip)"
        );
    }

    /// Residency: the camera plan cache hits across a pan/zoom (the
    /// content is unchanged — camera params are NOT in the key), returns the SAME
    /// Arc (no rebuild), stays byte-parity with a fresh Stage-0 build, and
    /// invalidates on a sample_count flip or a node edit.
    #[test]
    fn camera_plan_retained_caches_and_pan_hits() {
        let doc = json!({
            "nodes": [
                {"id":"a","type":"rectangle","x":0.0,"y":0.0,"width":50.0,"height":50.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]},
                {"id":"b","type":"ellipse","x":60.0,"y":0.0,"width":40.0,"height":40.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0f0"}}]},
                {"id":"c","type":"path","x":0.0,"y":0.0,"width":30.0,"height":30.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00f"}}],
                 "pathData":"M0 0 L30 0 L30 30 Z"},
            ],
            "pages": [
                {"id":"p1","width":800.0,"height":600.0,"nodeIds":["a","b"],
                 "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ffffff"}}]},
                {"id":"p2","width":800.0,"height":600.0,"nodeIds":["c"],
                 "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ffffff"}}]},
            ],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&moda_scene::value::to_string(&doc).unwrap()).unwrap();
        let mut caches = PlanCaches::default();
        let opts = RenderOptions {
            sample_count: 4,
            best_effort_text: true,
            text_masks_available: Some(true),
            ..Default::default()
        };
        let cam = |x: f64, zoom: f64| crate::frame::CameraParams {
            camera_x: x,
            camera_y: 0.0,
            zoom,
            viewport_w: 800.0,
            viewport_h: 600.0,
            dpr: 2.0,
            clear_rgb: [0.1, 0.2, 0.3],
        };
        let indices = [0usize, 1];

        // Cold build → miss.
        caches.begin_frame();
        let (cold, _w) =
            build_camera_plan_retained(&r, &mut caches, &indices, &cam(0.0, 1.0), &opts)
                .unwrap()
                .expect("no backdrop → Some");
        assert_eq!(caches.camera_misses, 1);
        assert_eq!(caches.camera_hits, 0);

        // Warm build at a PANNED + ZOOMED camera → HIT, same Arc (the camera
        // pose is applied at present via the override, not baked per frame).
        // The zoom stays in the same band (2 → 2.4 device px per unit): the
        // scene has curved fills, which refit per band above 1×.
        caches.begin_frame();
        let (warm, _w) =
            build_camera_plan_retained(&r, &mut caches, &indices, &cam(321.0, 1.2), &opts)
                .unwrap()
                .expect("Some");
        // The retained camera slot (`camera_slot`) serves it before the camera
        // plan cache: either is a hit on the same plan.
        assert_eq!(
            caches.camera_hits + caches.incr_static_hits,
            1,
            "pan/zoom must reuse the cached plan"
        );
        assert!(
            Arc::ptr_eq(&cold, &warm),
            "pan/zoom reuses the cached plan with no rebuild"
        );

        // Parity: the cached plan matches a fresh Stage-0 build (pure memoization).
        let scene = r.scene().unwrap();
        let (stage0, _w) = crate::frame::build_camera_plan(scene, &indices, &cam(0.0, 1.0), &opts)
            .unwrap()
            .expect("Some");
        assert_eq!(
            cold.items.len(),
            stage0.items.len(),
            "cached vs Stage-0 item count parity"
        );
        assert_eq!(cold.masks.len(), stage0.masks.len(), "mask count parity");
        let fills = |p: &CoveragePlan| {
            p.items
                .iter()
                .filter(|i| matches!(i, crate::seams::DrawItem::PageFill { .. }))
                .count()
        };
        assert_eq!(fills(&cold), 2, "one page-fill per visible page");
        assert_eq!(fills(&cold), fills(&stage0));

        // A quality-ladder sample_count flip re-fingerprints (rebuild).
        let opts1 = RenderOptions {
            sample_count: 1,
            ..opts.clone()
        };
        caches.begin_frame();
        build_camera_plan_retained(&r, &mut caches, &indices, &cam(0.0, 1.0), &opts1).unwrap();
        assert_eq!(caches.camera_misses, 2, "sample_count flip rebuilds");
        // Same opts again → hit.
        caches.begin_frame();
        build_camera_plan_retained(&r, &mut caches, &indices, &cam(9.0, 1.3), &opts1).unwrap();
        assert_eq!(caches.camera_hits + caches.incr_static_hits, 2);

        // A node edit (full re-feed) invalidates the camera cache.
        let doc2 = json!({
            "nodes": [
                {"id":"a","type":"rectangle","x":0.0,"y":0.0,"width":50.0,"height":50.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]},
                {"id":"b","type":"ellipse","x":99.0,"y":0.0,"width":40.0,"height":40.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0f0"}}]},
                {"id":"c","type":"path","x":0.0,"y":0.0,"width":30.0,"height":30.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00f"}}],
                 "pathData":"M0 0 L30 0 L30 30 Z"},
            ],
            "pages": [
                {"id":"p1","width":800.0,"height":600.0,"nodeIds":["a","b"]},
                {"id":"p2","width":800.0,"height":600.0,"nodeIds":["c"]},
            ],
            "config": {},
        });
        r.update(&moda_scene::value::to_string(&doc2).unwrap()).unwrap();
        caches.begin_frame();
        build_camera_plan_retained(&r, &mut caches, &indices, &cam(9.0, 1.3), &opts1).unwrap();
        assert_eq!(caches.camera_misses, 3, "node edit re-fingerprints");
    }

    /// Shader playback: an animating procedural-fill shader HITS the
    /// camera plan cache when ONLY the clock advances (no per-frame rebuild) —
    /// the live clock rides the present override — while a real draw-list change
    /// still MISSES. This is the primary non-GPU correctness artifact for Lever 1.
    #[test]
    fn camera_plan_shader_clock_only_change_hits() {
        let doc = json!({
            "nodes": [
                {"id":"s","type":"rectangle","x":0.0,"y":0.0,"width":200.0,"height":150.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "shader", "shaderId": "clouds"}}]},
            ],
            "pages": [{"id":"p1","width":800.0,"height":600.0,"nodeIds":["s"]}],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&moda_scene::value::to_string(&doc).unwrap()).unwrap();
        let mut caches = PlanCaches::default();
        let opts = |shader_time_s: f64| RenderOptions {
            sample_count: 4,
            shader_time_s,
            ..Default::default()
        };
        let cam = crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom: 1.0,
            viewport_w: 800.0,
            viewport_h: 600.0,
            dpr: 2.0,
            clear_rgb: [0.1, 0.1, 0.1],
        };
        let indices = [0usize];

        // Cold build at clock 0 → miss; the cached fill exposes a re-threadable
        // clock offset (clouds packs `time` at shader_params[0]).
        caches.begin_frame();
        let (cold, _w) = build_camera_plan_retained(&r, &mut caches, &indices, &cam, &opts(0.0))
            .unwrap()
            .expect("no backdrop → Some");
        assert_eq!(caches.camera_misses, 1);
        assert_eq!(caches.camera_hits, 0);
        assert!(
            plan_uses_shader_time(&cold),
            "a shader fill bakes the clock"
        );
        assert!(
            plan_shader_time_rethreadable(&cold),
            "a procedural-fill clock is re-threadable"
        );
        let offsets = cold.items.iter().find_map(|i| match i {
            crate::seams::DrawItem::Body {
                paint: crate::seams::BodyPaint::Shader(b),
                ..
            } => Some(b.clock_offsets.clone()),
            _ => None,
        });
        assert_eq!(
            offsets,
            Some(Some(vec![0u16])),
            "clouds exposes its clock slot (shader_params[0])"
        );

        // The cold build now ALSO seeds the incremental slot (the shader
        // is re-threadable), so clock-only frames are served by the incremental
        // STATIC lane (which runs BEFORE `camera_get`) rather than the camera plan
        // cache. Either lane is correct — the clock rides the present override — but
        // the serving counter is now `incr_static_hits`, not `camera_hits`.
        assert!(
            caches.camera_slot.is_some(),
            "re-threadable shader plan must seed the incremental slot"
        );
        // Clock advances (only `shader_time_s` differs) → served, same Arc, no
        // rebuild: the new clock is applied at present, not baked per frame.
        for (i, t) in [0.016_f64, 0.033, 1.5, 99.0].into_iter().enumerate() {
            caches.begin_frame();
            let (warm, _w) = build_camera_plan_retained(&r, &mut caches, &indices, &cam, &opts(t))
                .unwrap()
                .expect("Some");
            assert_eq!(
                caches.incr_static_hits,
                i as u64 + 1,
                "clock-only frame must serve incrementally (t={t})"
            );
            assert_eq!(
                caches.camera_hits, 0,
                "clock-only frame is served before the camera plan cache"
            );
            assert_eq!(caches.camera_misses, 1, "clock-only frame must NOT rebuild");
            assert!(
                Arc::ptr_eq(&cold, &warm),
                "clock-only frame reuses the cached plan Arc"
            );
        }

        // A real draw-list change (node moved) is no longer served stale. NOTE this
        // applies the change via `r.update()` (a full re-feed), which is STRUCTURAL
        // (`last_structural_gen` is bumped), so the slot is correctly invalidated and
        // the frame FULL-REBUILDS — it is NOT served by the patch lane. (The patch
        // lane for a structure-preserving drag is exercised by the `apply_props` path
        // in `camera_incremental_serves_when_unrelated_shader_present`.) The point
        // here: the changed content is not served stale from the clock-only-hit slot.
        let doc2 = json!({
            "nodes": [
                {"id":"s","type":"rectangle","x":40.0,"y":0.0,"width":200.0,"height":150.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "shader", "shaderId": "clouds"}}]},
            ],
            "pages": [{"id":"p1","width":800.0,"height":600.0,"nodeIds":["s"]}],
            "config": {},
        });
        let patch_before = caches.incr_patch_hits;
        let misses_before = caches.camera_misses;
        r.update(&moda_scene::value::to_string(&doc2).unwrap()).unwrap();
        caches.begin_frame();
        let (moved, _w) = build_camera_plan_retained(&r, &mut caches, &indices, &cam, &opts(99.0))
            .unwrap()
            .expect("Some");
        // Served-path expectation: a structural re-feed does NOT patch; it forces
        // exactly one full camera rebuild (so a silent stale-slot serve can't pass).
        assert_eq!(
            caches.incr_patch_hits, patch_before,
            "a structural re-feed must NOT serve via the patch lane"
        );
        assert_eq!(
            caches.camera_misses,
            misses_before + 1,
            "a structural re-feed must force exactly one full camera rebuild"
        );
        // The served plan reflects the move (different from the pre-move plan)...
        assert_ne!(
            plan_items_fingerprint(&cold),
            plan_items_fingerprint(&moved),
            "the moved node must change the served plan"
        );
        // ...and is byte-exact vs a from-scratch combined camera build.
        let scene = r.scene().unwrap();
        let hashes = r.hashes().unwrap();
        let (full, _w) = crate::frame::build_camera_plan_cached(
            scene,
            &indices,
            &cam,
            &opts(99.0),
            Some((&mut PlanCaches::default(), hashes)),
            None,
        )
        .unwrap()
        .expect("full camera build");
        assert_eq!(
            plan_items_fingerprint(&moved),
            plan_items_fingerprint(&full),
            "served shader plan after a move diverged from a full rebuild"
        );
    }

    /// The cache-key logic in isolation — a re-threadable plan drops
    /// the clock from its hit key (animates via the present override), while a
    /// NON-re-threadable plan (a clock use the present path can't patch, modelled
    /// here by `clock_offsets: None`) keeps the exact-clock-match behavior so it
    /// rebuilds per clock rather than freezing.
    #[test]
    fn camera_cache_key_honours_rethreadability() {
        use crate::seams::{BodyPaint, ShaderFillBody, TessellatedGeometry};
        fn shader_plan(clock_offsets: Option<Vec<u16>>) -> Arc<CoveragePlan> {
            Arc::new(coverage_plan(
                "p",
                vec![body_item(
                    "n",
                    BodyPaint::Shader(Box::new(ShaderFillBody {
                        module_id: "clouds".to_string(),
                        image: None,
                        clock_offsets,
                    })),
                    Arc::new(TessellatedGeometry::default()),
                )],
            ))
        }
        let warn = Arc::new(PlanWarnings::default());
        let fp = 0xABCDu64;

        // Re-threadable: a clock-only change HITS (the present override animates).
        let rethreadable = shader_plan(Some(vec![0]));
        assert!(plan_uses_shader_time(&rethreadable));
        assert!(plan_shader_time_rethreadable(&rethreadable));
        let mut caches = PlanCaches::default();
        caches.camera_put(fp, 0.0, rethreadable, warn.clone());
        assert!(
            caches.camera_get(fp, 0.5).is_some(),
            "re-threadable: clock change hits"
        );
        assert_eq!(caches.camera_hits, 1);
        assert_eq!(caches.camera_misses, 0);

        // Offscreen frames have no present override, even for re-threadable fills: a host's plan index holds
        // such a plan for its exact clock only.
        let mut index = FramePlanIndex::default();
        index.entry = Some(FrameIndexEntry {
            plan: shader_plan(Some(vec![0])), warnings: warn.clone(), key: fp, shader_time_s: 0.0,
            uses_shader_time: true, bytes: 0, patchable: None,
        });
        assert!(index.held(fp, 0.5).is_none());
        assert!(index.held(fp, 0.0).is_some());
        assert!(index.held(fp + 1, 0.0).is_none());

        // Non-re-threadable: a clock-only change MISSES; the same clock HITS.
        let frozen = shader_plan(None);
        assert!(plan_uses_shader_time(&frozen));
        assert!(!plan_shader_time_rethreadable(&frozen));
        let mut caches = PlanCaches::default();
        caches.camera_put(fp, 0.0, frozen, warn.clone());
        assert!(
            caches.camera_get(fp, 0.5).is_none(),
            "non-re-threadable: clock change must rebuild (no freeze)"
        );
        assert!(
            caches.camera_get(fp, 0.0).is_some(),
            "non-re-threadable: identical clock still hits"
        );
        assert_eq!(caches.camera_hits, 1);
        assert_eq!(caches.camera_misses, 1);
    }

    /// A plan is re-threadable ONLY when EVERY clock consumer is. A
    /// re-threadable procedural-fill body mixed with a NON-re-threadable consumer
    /// (a body fill whose clock isn't a direct copy) makes the WHOLE plan
    /// non-re-threadable — so the clock stays in the cache key.
    #[test]
    fn mixed_plan_with_any_non_rethreadable_consumer_is_not_rethreadable() {
        use crate::seams::{BodyPaint, ShaderFillBody, TessellatedGeometry};

        // A re-threadable body fill ALONE → re-threadable.
        assert!(plan_shader_time_rethreadable(&coverage_plan(
            "p",
            vec![body_item(
                "b",
                BodyPaint::Shader(Box::new(ShaderFillBody {
                    module_id: "clouds".to_string(),
                    image: None,
                    clock_offsets: Some(vec![0]),
                })),
                Arc::new(TessellatedGeometry::default()),
            )]
        )));

        // Re-threadable body fill + a non-direct-copy body fill → NOT re-threadable.
        assert!(!plan_shader_time_rethreadable(&coverage_plan(
            "p",
            vec![
                body_item(
                    "b",
                    BodyPaint::Shader(Box::new(ShaderFillBody {
                        module_id: "clouds".to_string(),
                        image: None,
                        clock_offsets: Some(vec![0]),
                    })),
                    Arc::new(TessellatedGeometry::default()),
                ),
                body_item(
                    "b",
                    BodyPaint::Shader(Box::new(ShaderFillBody {
                        module_id: "clouds".to_string(),
                        image: None,
                        clock_offsets: None,
                    })),
                    Arc::new(TessellatedGeometry::default()),
                ),
            ]
        )));

        // A plain text fill (no shader) does NOT block re-threading.
        assert!(plan_shader_time_rethreadable(&coverage_plan(
            "p",
            vec![
                body_item(
                    "b",
                    BodyPaint::Shader(Box::new(ShaderFillBody {
                        module_id: "clouds".to_string(),
                        image: None,
                        clock_offsets: Some(vec![0]),
                    })),
                    Arc::new(TessellatedGeometry::default()),
                ),
                text_item(),
            ]
        )));
    }

    /// The incremental-slot SEED gate moved from `!plan_uses_shader_time`
    /// (a shader-PRESENCE check) to `plan_shader_time_rethreadable`. Prove the new
    /// predicate is a STRICT SUPERSET of the old one at the seed site:
    ///   * a re-threadable shader plan is now ADMITTED (old gate rejected it),
    ///   * a non-re-threadable shader plan is STILL rejected,
    ///   * a non-shader plan is STILL admitted (unchanged),
    /// and `!plan_uses_shader_time(p) ⇒ plan_shader_time_rethreadable(p)` for each.
    #[test]
    fn seed_gate_for_rethreadable_shader_is_superset_of_old_gate() {
        use crate::seams::{BodyPaint, ShaderFillBody, TessellatedGeometry};

        // The OLD seed gate; the NEW one is `plan_shader_time_rethreadable`.
        let old_gate = |p: &CoveragePlan| !plan_uses_shader_time(p);
        let new_gate = plan_shader_time_rethreadable;

        // (1) Re-threadable shader: NEW admits, OLD rejected → the actual fix.
        let rethreadable = coverage_plan(
            "p",
            vec![body_item(
                "b",
                BodyPaint::Shader(Box::new(ShaderFillBody {
                    module_id: "clouds".to_string(),
                    image: None,
                    clock_offsets: Some(vec![0]),
                })),
                Arc::new(TessellatedGeometry::default()),
            )],
        );
        assert!(plan_uses_shader_time(&rethreadable));
        assert!(
            new_gate(&rethreadable),
            "new gate now seeds re-threadable shader"
        );
        assert!(
            !old_gate(&rethreadable),
            "old gate (shader-presence) refused to seed it — the ENG-3601 bug"
        );

        // (2) Non-re-threadable shader: both gates still reject (clock baked).
        let frozen_body = coverage_plan(
            "p",
            vec![body_item(
                "b",
                BodyPaint::Shader(Box::new(ShaderFillBody {
                    module_id: "clouds".to_string(),
                    image: None,
                    clock_offsets: None,
                })),
                Arc::new(TessellatedGeometry::default()),
            )],
        );
        assert!(!new_gate(&frozen_body), "non-direct-copy clock not seeded");
        assert!(!old_gate(&frozen_body));
        // (3) Non-shader plan: both gates admit (unchanged seeding). Empty items
        // is the canonical no-shader case: `all()` is vacuously true, `any()` false.
        let non_shader = coverage_plan("p", vec![]);
        assert!(!plan_uses_shader_time(&non_shader));
        assert!(new_gate(&non_shader), "non-shader plan still seeds");
        assert!(old_gate(&non_shader));

        // Superset invariant: `!uses_shader_time ⇒ rethreadable` for every case.
        for p in [&rethreadable, &frozen_body, &non_shader] {
            assert!(
                !old_gate(p) || new_gate(p),
                "new gate must admit everything the old gate admitted"
            );
        }
    }

    /// M1: the cheap generation key is sensitive to EVERY input the O(n)
    /// `camera_fingerprint` folds (so it can never under-detect a change into a
    /// stale reuse), and deliberately NOT to `shader_time` (which `camera_get`
    /// re-checks). Pure logic — no scene/GPU needed.
    #[test]
    fn camera_frame_key_sensitivity() {
        let opts = RenderOptions {
            scale: 2.0,
            transparent: false,
            sample_count: 4,
            best_effort_text: true,
            text_masks_available: Some(true),
            exclude: vec!["x".to_string()],
            isolate: Vec::new(),
            ..Default::default()
        };
        let indices = [0usize, 1, 2];
        let base = camera_frame_key(7, &indices, &opts);
        assert_eq!(base, camera_frame_key(7, &indices, &opts), "deterministic");
        assert_ne!(base, camera_frame_key(8, &indices, &opts), "generation");
        assert_ne!(
            base,
            camera_frame_key(7, &[0usize, 1, 3], &opts),
            "visible membership"
        );
        assert_ne!(
            base,
            camera_frame_key(7, &[0usize, 1], &opts),
            "visible length"
        );
        let key = |o: &RenderOptions| camera_frame_key(7, &indices, o);
        let mut o = opts.clone();
        o.scale = 1.5;
        assert_ne!(base, key(&o), "scale/dpr");
        let mut o = opts.clone();
        o.transparent = true;
        assert_ne!(base, key(&o), "transparent");
        let mut o = opts.clone();
        o.sample_count = 1;
        assert_ne!(base, key(&o), "sample_count");
        let mut o = opts.clone();
        o.best_effort_text = false;
        assert_ne!(base, key(&o), "best_effort_text");
        let mut o = opts.clone();
        o.text_masks_available = None;
        assert_ne!(base, key(&o), "text_masks_available");
        let mut o = opts.clone();
        o.exclude = vec!["y".to_string()];
        assert_ne!(base, key(&o), "exclude");
        // `isolate` is plan-relevant exactly as `exclude` is: a cached plan
        // must never be served across two different per-element isolations.
        let mut o = opts.clone();
        o.isolate = vec!["y".to_string()];
        assert_ne!(base, key(&o), "isolate");
        // ...and an `isolate` list must not collide with the SAME ids as an
        // `exclude` list (opposite filters, opposite plans).
        let mut a = opts.clone();
        a.exclude = vec!["z".to_string()];
        let mut b = opts.clone();
        b.isolate = vec!["z".to_string()];
        assert_ne!(key(&a), key(&b), "exclude vs isolate must not collide");
        // Per-id length prefixes: without them the ids concatenate and two
        // different isolations fold to the same key, so one's cached plan
        // would be served for the other's render.
        let mut c = opts.clone();
        c.isolate = vec!["ab".to_string(), "c".to_string()];
        let mut d = opts.clone();
        d.isolate = vec!["a".to_string(), "bc".to_string()];
        assert_ne!(key(&c), key(&d), "isolate id boundaries must be hashed");
        // shader_time is intentionally excluded (camera_get gates it instead).
        let mut o = opts.clone();
        o.shader_time_s = 9.0;
        assert_eq!(base, key(&o), "shader_time must NOT move the key");
    }

    /// rust-render-perf-g8 (E3): the page-path memo key must move on every input
    /// `page_fingerprint` reads (generation ⊇ all scene content, page index, plan
    /// options) and stay stable across `shader_time_s` (page_get gates it instead),
    /// mirroring `camera_frame_key_sensitivity`. A superset violation here would
    /// let a stale fingerprint be reused — the one correctness risk of the memo.
    #[test]
    fn page_frame_key_sensitivity() {
        let opts = RenderOptions {
            scale: 2.0,
            transparent: false,
            sample_count: 4,
            best_effort_text: true,
            text_masks_available: Some(true),
            exclude: vec!["x".to_string()],
            isolate: Vec::new(),
            ..Default::default()
        };
        let base = page_frame_key(7, 0, &opts);
        assert_eq!(base, page_frame_key(7, 0, &opts), "deterministic");
        assert_ne!(base, page_frame_key(8, 0, &opts), "generation");
        assert_ne!(base, page_frame_key(7, 1, &opts), "page index");
        let key = |o: &RenderOptions| page_frame_key(7, 0, o);
        let mut o = opts.clone();
        o.scale = 1.5;
        assert_ne!(base, key(&o), "scale/dpr");
        let mut o = opts.clone();
        o.transparent = true;
        assert_ne!(base, key(&o), "transparent");
        let mut o = opts.clone();
        o.sample_count = 1;
        assert_ne!(base, key(&o), "sample_count");
        let mut o = opts.clone();
        o.best_effort_text = false;
        assert_ne!(base, key(&o), "best_effort_text");
        let mut o = opts.clone();
        o.text_masks_available = None;
        assert_ne!(base, key(&o), "text_masks_available");
        let mut o = opts.clone();
        o.exclude = vec!["y".to_string()];
        assert_ne!(base, key(&o), "exclude");
        // `isolate` is plan-relevant exactly as `exclude` is: a cached plan
        // must never be served across two different per-element isolations.
        let mut o = opts.clone();
        o.isolate = vec!["y".to_string()];
        assert_ne!(base, key(&o), "isolate");
        // ...and an `isolate` list must not collide with the SAME ids as an
        // `exclude` list (opposite filters, opposite plans).
        let mut a = opts.clone();
        a.exclude = vec!["z".to_string()];
        let mut b = opts.clone();
        b.isolate = vec!["z".to_string()];
        assert_ne!(key(&a), key(&b), "exclude vs isolate must not collide");
        // Per-id length prefixes: without them the ids concatenate and two
        // different isolations fold to the same key, so one's cached plan
        // would be served for the other's render.
        let mut c = opts.clone();
        c.isolate = vec!["ab".to_string(), "c".to_string()];
        let mut d = opts.clone();
        d.isolate = vec!["a".to_string(), "bc".to_string()];
        assert_ne!(key(&c), key(&d), "isolate id boundaries must be hashed");
        // shader_time is intentionally excluded (page_get gates it instead).
        let mut o = opts.clone();
        o.shader_time_s = 9.0;
        assert_eq!(base, key(&o), "shader_time must NOT move the key");
    }

    /// M1: the resident-generation counter bumps on every content change (a real
    /// re-parse, a scalar patch, a clear) and stays put on a no-op feed.
    #[test]
    fn resident_generation_tracks_content_changes() {
        let doc = |fill: &str| {
            moda_scene::value::to_string(&json!({
                "nodes": [{"id":"a","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": fill}}]}],
                "pages": [{"id":"p1","width":100.0,"height":100.0,"nodeIds":["a"]}],
                "config": {},
            }))
            .unwrap()
        };
        let mut r = ResidentScene::default();
        assert_eq!(r.generation(), 0);
        r.update(&doc("#f00")).unwrap();
        let g1 = r.generation();
        assert!(g1 > 0, "first parse bumps the generation");
        r.update(&doc("#f00")).unwrap();
        assert_eq!(r.generation(), g1, "an unchanged feed must not bump");
        r.update(&doc("#0f0")).unwrap();
        let g2 = r.generation();
        assert!(g2 > g1, "a content change bumps");
        r.apply_props(&moda_scene::value::to_string(&json!({"nodes":{"a":{"opacity":0.5}}})).unwrap())
            .unwrap();
        let g3 = r.generation();
        assert!(g3 > g2, "an apply_props patch bumps");
        r.clear();
        assert!(r.generation() > g3, "clear bumps");
    }

    /// M1: with the memo ON (default), a static pan SKIPS the O(n)
    /// `camera_fingerprint` recompute yet returns the SAME cached plan; a
    /// target-param flip or a node edit must NOT skip and must rebuild; and the
    /// kill switch falls back to recompute while staying correct.
    #[test]
    fn fingerprint_memo_skips_recompute_on_static_pan() {
        let doc = |bx: f64| {
            moda_scene::value::to_string(&json!({
                "nodes": [
                    {"id":"a","type":"rectangle","x":0.0,"y":0.0,"width":50.0,"height":50.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]},
                    {"id":"b","type":"ellipse","x":bx,"y":0.0,"width":40.0,"height":40.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0f0"}}]},
                ],
                "pages": [{"id":"p1","width":800.0,"height":600.0,"nodeIds":["a","b"]}],
                "config": {},
            }))
            .unwrap()
        };
        let mut r = ResidentScene::default();
        r.update(&doc(60.0)).unwrap();
        // `PlanCaches::default()` has the fingerprint memo ON. Isolate it: the
        // (newer) camera incremental fast path would otherwise shadow the memo on
        // this single-page, non-backdrop static-pan frame — serving the same cached
        // plan via an even cheaper `incr_static_hits` exit before the memo runs, so
        // `camera_fp_skips`/`camera_hits` never tick. That layering is correct (the
        // camera plan is camera-independent), but this test exercises the memo,
        // which stays live for multi-page, backdrop, shader, and over-budget frames
        // where the incremental slot is not seeded.
        let mut caches = PlanCaches::default();
        caches.set_incremental_enabled(false);
        let opts = RenderOptions {
            sample_count: 4,
            best_effort_text: true,
            text_masks_available: Some(true),
            ..Default::default()
        };
        let cam = |x: f64, zoom: f64| crate::frame::CameraParams {
            camera_x: x,
            camera_y: 0.0,
            zoom,
            viewport_w: 800.0,
            viewport_h: 600.0,
            dpr: 2.0,
            clear_rgb: [0.1, 0.2, 0.3],
        };
        let indices = [0usize];

        caches.begin_frame();
        let (cold, _w) =
            build_camera_plan_retained(&r, &mut caches, &indices, &cam(0.0, 1.0), &opts)
                .unwrap()
                .expect("no backdrop → Some");
        assert_eq!(
            caches.camera_fp_skips, 0,
            "cold build computes the fingerprint"
        );
        assert_eq!(caches.camera_misses, 1);

        caches.begin_frame();
        let (warm, _w) =
            build_camera_plan_retained(&r, &mut caches, &indices, &cam(321.0, 1.2), &opts)
                .unwrap()
                .expect("Some");
        assert_eq!(
            caches.camera_fp_skips, 1,
            "a static pan skips the recompute"
        );
        assert_eq!(
            caches.camera_hits, 1,
            "the memo still hits the camera cache"
        );
        assert!(
            Arc::ptr_eq(&cold, &warm),
            "memo path returns the cached plan"
        );

        let opts1 = RenderOptions {
            sample_count: 1,
            ..opts.clone()
        };
        caches.begin_frame();
        build_camera_plan_retained(&r, &mut caches, &indices, &cam(0.0, 1.0), &opts1).unwrap();
        assert_eq!(
            caches.camera_fp_skips, 1,
            "a sample_count flip must NOT skip"
        );
        assert_eq!(caches.camera_misses, 2, "a sample_count flip rebuilds");

        r.update(&doc(99.0)).unwrap();
        caches.begin_frame();
        build_camera_plan_retained(&r, &mut caches, &indices, &cam(0.0, 1.0), &opts1).unwrap();
        assert_eq!(caches.camera_fp_skips, 1, "a node edit must NOT skip");
        assert_eq!(caches.camera_misses, 3, "a node edit rebuilds");

        // Kill switch: no further skips, but the recomputed fingerprint still
        // matches the slot → the camera cache hit is unaffected.
        caches.set_fingerprint_memo_enabled(false);
        caches.begin_frame();
        build_camera_plan_retained(&r, &mut caches, &indices, &cam(7.0, 1.1), &opts1).unwrap();
        assert_eq!(
            caches.camera_fp_skips, 1,
            "kill switch: memo no longer skips"
        );
        assert_eq!(
            caches.camera_hits, 2,
            "kill switch still hits via recomputed fingerprint"
        );
    }

    /// The camera-dependence check is memoised per resident generation: a static pan reuses the answer,
    /// and a feed that adds (or removes) a backdrop is seen on the next frame.
    #[test]
    fn camera_dependence_memo_follows_the_resident_generation() {
        let doc = |blur: bool| {
            let mut node = json!({ "id": "r", "type": "rectangle", "x": 100.0, "y": 80.0, "width": 200.0, "height": 150.0, "fill": "#ff0000" });
            if blur {
                node["effects"] = json!([{ "id": "backdrop-0", "type": "background-blur", "radius": 24 }]);
            }
            moda_scene::value::to_string(&json!({
                "nodes": [node],
                "pages": [{ "id": "p1", "width": 800.0, "height": 600.0, "fill": "#ffffff", "nodeIds": ["r"] }],
                "config": {},
            }))
            .unwrap()
        };
        let opts = RenderOptions {
            sample_count: 4,
            best_effort_text: true,
            text_masks_available: Some(true),
            ..Default::default()
        };
        let cam = |x: f64| crate::frame::CameraParams {
            camera_x: x,
            camera_y: 0.0,
            zoom: 1.0,
            viewport_w: 800.0,
            viewport_h: 600.0,
            dpr: 2.0,
            clear_rgb: [0.1, 0.1, 0.1],
        };
        let has_split = |plan: &CoveragePlan| {
            plan.items
                .iter()
                .any(|i| matches!(i, crate::seams::DrawItem::BackdropSplit(_)))
        };
        let mut r = ResidentScene::default();
        let mut caches = PlanCaches::default();
        let frame = |r: &ResidentScene, caches: &mut PlanCaches, x: f64| {
            caches.begin_frame();
            build_camera_plan_retained(r, caches, &[0], &cam(x), &opts)
                .unwrap()
                .expect("Some")
                .0
        };

        r.update(&doc(false)).unwrap();
        let first = frame(&r, &mut caches, 0.0);
        assert!(!has_split(&first));
        assert_eq!(
            caches.camera_dependent_memo.as_ref().map(|m| m.2),
            Some(false)
        );
        let panned = frame(&r, &mut caches, 50.0);
        assert!(
            Arc::ptr_eq(&first, &panned),
            "a pan still hits the camera plan cache"
        );

        r.update(&doc(true)).unwrap();
        let blurred = frame(&r, &mut caches, 50.0);
        assert!(
            has_split(&blurred),
            "the new backdrop is seen on the next frame"
        );
        assert_eq!(
            caches.camera_dependent_memo.as_ref().map(|m| (m.0, m.2)),
            Some((r.generation(), true))
        );

        r.update(&doc(false)).unwrap();
        let cleared = frame(&r, &mut caches, 50.0);
        assert!(!has_split(&cleared));
        assert_eq!(
            caches.camera_dependent_memo.as_ref().map(|m| m.2),
            Some(false)
        );
    }

    /// A backdrop scene BYPASSES the camera plan cache (the
    /// split is camera-dependent), so it never probes/caches it, and each camera
    /// yields a distinct camera-projected split. Guards the load-bearing bypass.
    #[test]
    fn camera_backdrop_scene_bypasses_plan_cache() {
        let doc = json!({
            "nodes": [
                { "id": "r", "type": "rectangle", "x": 100.0, "y": 80.0, "width": 200.0, "height": 150.0,
                  "effects": [{ "id": "backdrop-0", "type": "background-blur", "radius": 24 }] },
            ],
            "pages": [{ "id": "p1", "width": 800.0, "height": 600.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}], "nodeIds": ["r"] }],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&moda_scene::value::to_string(&doc).unwrap()).unwrap();
        let mut caches = PlanCaches::default();
        let opts = RenderOptions {
            sample_count: 4,
            best_effort_text: true,
            text_masks_available: Some(true),
            ..Default::default()
        };
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 800.0,
            viewport_h: 600.0,
            dpr: 2.0,
            clear_rgb: [0.1, 0.1, 0.1],
        };
        let split_rect = |plan: &CoveragePlan| -> [f32; 4] {
            plan.items
                .iter()
                .find_map(|i| match i {
                    crate::seams::DrawItem::BackdropSplit(s) => Some(s.node_rect_px),
                    _ => None,
                })
                .expect("backdrop split present")
        };

        caches.begin_frame();
        let (p1, _w) = build_camera_plan_retained(&r, &mut caches, &[0], &cam(1.0), &opts)
            .unwrap()
            .expect("backdrop renders in-frame (Some)");
        assert_eq!(
            p1.sample_count, 1,
            "backdrop camera frame is sample_count 1"
        );
        let rect1 = split_rect(&p1);

        caches.begin_frame();
        let (p2, _w) = build_camera_plan_retained(&r, &mut caches, &[0], &cam(2.0), &opts)
            .unwrap()
            .expect("Some");
        let rect2 = split_rect(&p2);

        // The bypass never probes or writes the camera plan cache.
        assert_eq!(
            caches.camera_hits, 0,
            "backdrop must not hit the camera plan cache"
        );
        assert_eq!(
            caches.camera_misses, 0,
            "backdrop must not even probe the camera plan cache"
        );
        // Camera-projected split → node rect scales with zoom (per-page would not).
        assert!(
            (rect2[2] - rect1[2] * 2.0).abs() < 1.0,
            "split node rect width scales with zoom: {} vs {}",
            rect2[2],
            rect1[2]
        );
    }

    #[test]
    fn text_layout_props_change_text_hash() {
        let a = raw(
            json!({"id": "t", "type": "richtext", "htmlContent": "<p>hi</p>", "width": 100.0, "height": 40.0}),
        );
        for v in [
            json!({"id": "t", "type": "richtext", "htmlContent": "<p>ho</p>", "width": 100.0, "height": 40.0}),
            json!({"id": "t", "type": "richtext", "htmlContent": "<p>hi</p>", "width": 80.0, "height": 40.0}),
            json!({"id": "t", "type": "richtext", "htmlContent": "<p>hi</p>", "width": 100.0, "height": 30.0}),
            json!({"id": "t", "type": "richtext", "htmlContent": "<p>hi</p>", "width": 100.0, "height": 40.0, "verticalAlign": "bottom"}),
            json!({"id": "t", "type": "richtext", "htmlContent": "<p>hi</p>", "width": 100.0, "height": 40.0, "paragraphSpacing": 8.0}),
        ] {
            assert_ne!(node_hashes(&a).text, node_hashes(&raw(v)).text);
        }
    }

    // -----------------------------------------------------------------------
    // rust-edit-incremental-z4: incremental plan patching correctness (CPU, no
    // GPU) — the fast oracle complementing the mutation_replay byte-exact gate.
    // -----------------------------------------------------------------------

    fn editor_opts() -> RenderOptions {
        RenderOptions {
            preserve_page_fill: false,
            scale: 1.0,
            transparent: false,
            sample_count: 4,
            shader_time_s: 0.0,
            exclude: Vec::new(),
            isolate: Vec::new(),
            best_effort_text: true,
            text_masks_available: Some(true),
            working: None,
        }
    }

    fn rect_scene(n: usize) -> String {
        let nodes: Vec<Value> = (0..n)
            .map(|i| {
                json!({
                    "id": format!("r{i}"), "type": "rectangle",
                    "x": (i * 13 % 700) as f64, "y": (i * 7 % 500) as f64,
                    "width": 40.0, "height": 30.0, "rotation": 0, "scale": 1,
                    "cornerRadius": 3, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}]
                })
            })
            .collect();
        let ids: Vec<String> = (0..n).map(|i| format!("r{i}")).collect();
        moda_scene::value::to_string(&json!({
            "nodes": nodes,
            "pages": [{"id": "page-1", "name": "p", "width": 800.0, "height": 600.0,
                       "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}], "nodeIds": ids}],
            "config": {"renderPageClipping": true},
            "canvasRootNodeIds": ids, "layoutMode": "freeform", "variables": [],
        }))
        .unwrap()
    }

    /// Like `rect_scene` but every rect carries a visible stroke on a SHARP rect,
    /// so each body is tessellation-routed (routes only CURVED solid
    /// strokes to the SDF ring; a sharp-rect stroke still tessellates) — used where
    /// a test needs an actual `DrawItem::Body` carrying cacheable tessellated
    /// geometry (plain solid rects are analytic SDF bodies and never hit the tess
    /// cache).
    fn stroked_rect_scene(n: usize) -> String {
        // SHARP rects (no corner radius): a solid stroke on a sharp rect stays on
        // tessellation (routes only CURVED stroked bodies to the SDF
        // ring), so these give the test a `DrawItem::Body` with tess geometry.
        let nodes: Vec<Value> = (0..n)
            .map(|i| {
                json!({
                    "id": format!("r{i}"), "type": "rectangle",
                    "x": (i * 13 % 700) as f64, "y": (i * 7 % 500) as f64,
                    "width": 40.0, "height": 30.0, "rotation": 0, "scale": 1,
                    "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}],
                    "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#222222"}, "width": 2}]
                })
            })
            .collect();
        let ids: Vec<String> = (0..n).map(|i| format!("r{i}")).collect();
        moda_scene::value::to_string(&json!({
            "nodes": nodes,
            "pages": [{"id": "page-1", "name": "p", "width": 800.0, "height": 600.0,
                       "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}], "nodeIds": ids}],
            "config": {"renderPageClipping": true},
            "canvasRootNodeIds": ids, "layoutMode": "freeform", "variables": [],
        }))
        .unwrap()
    }

    /// Every structure-preserving property edit must (a) engage the incremental
    /// patch and (b) produce a plan byte-(fingerprint-)identical to a from-scratch
    /// build of the edited scene.
    #[test]
    fn incremental_property_edits_match_full_rebuild() {
        let opts = editor_opts();
        let mut resident = ResidentScene::default();
        resident.update(&rect_scene(40)).unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        build_plan_retained(&resident, &mut caches, 0, &opts).unwrap(); // seed index

        // These edits all keep their node on the SAME DrawItem variant
        // (a plain solid rect stays an analytic SDF body under transform/fill/
        // size/rotation/opacity/corner-radius edits), so the incremental patch
        // engages. Adding a SOLID stroke to a CURVED rect ALSO stays SDF now (the
        // stroke ring rides on the same instance), so it too engages. An edit that
        // flips the variant (e.g. a DASHED stroke routes the body from SDF to
        // tessellation) is structural and falls back — see the explicit case after
        // this loop.
        let edits = [
            json!({"nodes": {"r5": {"x": 123.0}}}),
            json!({"nodes": {"r5": {"fills": solid("#ff0000")}}}),
            json!({"nodes": {"r10": {"width": 88.0, "height": 12.0}}}),
            json!({"nodes": {"r0": {"rotation": 30.0}}}),
            json!({"nodes": {"r39": {"x": 7.0, "y": 9.0, "opacity": 0.5}}}),
            json!({"nodes": {"r5": {"cornerRadius": 10.0}}}),
            // A solid stroke on a curved rect stays an SDF body (the
            // ring packs into the same instance) — the patch must still engage.
            json!({"nodes": {"r1": {"strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": 2.0}]}}}),
            // Multi-node edit in one patch (a marquee drag).
            json!({"nodes": {"r2": {"x": 1.0}, "r3": {"x": 2.0}, "r4": {"y": 3.0}}}),
        ];
        for (i, e) in edits.iter().enumerate() {
            let patches_before = caches.incr_patch_hits;
            resident
                .apply_props(&moda_scene::value::to_string(e).unwrap())
                .unwrap();
            caches.begin_frame();
            let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
            assert!(
                caches.incr_patch_hits > patches_before,
                "edit {i}: expected the incremental patch to engage (got a full rebuild)"
            );
            let scene = resident.scene().unwrap();
            let (full, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
            assert_eq!(
                plan_items_fingerprint(&incr),
                plan_items_fingerprint(&full),
                "edit {i} ({e}): incremental plan diverged from full rebuild"
            );
        }

        // A DASHED stroke is out of the SDF ring's scope, so adding a
        // dash flips r1 (already a solid-stroked SDF body from the loop above)
        // from the analytic SDF body to a tessellated path (a DrawItem variant
        // change). The patch's shape-match correctly rejects it and falls back to
        // a full rebuild — which must still be correct.
        let fb_before = caches.incr_fallbacks;
        resident
            .apply_props(
                &json!({"nodes": {"r1": {"strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"},
                    "width": 2.0, "dash": [4.0, 2.0]}]}}})
                .to_string(),
            )
            .unwrap();
        caches.begin_frame();
        let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        assert!(
            caches.incr_fallbacks > fb_before,
            "dashed-stroke-add (SDF→tessellation variant flip) must fall back to a full rebuild"
        );
        let scene = resident.scene().unwrap();
        let (full, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
        assert_eq!(
            plan_items_fingerprint(&incr),
            plan_items_fingerprint(&full),
            "stroke-add fallback diverged from full rebuild"
        );
        // A static re-render (no edit) is served from the slot with no rebuild.
        let static_before = caches.incr_static_hits;
        caches.begin_frame();
        build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        assert!(
            caches.incr_static_hits > static_before,
            "static re-render should hit the slot"
        );
    }

    /// A z-order edit changes the display-list shape, so it MUST fall back to a
    /// full rebuild — and still be correct.
    #[test]
    fn incremental_falls_back_on_zindex() {
        let opts = editor_opts();
        let mut resident = ResidentScene::default();
        resident.update(&rect_scene(20)).unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();

        let fb_before = caches.incr_fallbacks;
        let patch_before = caches.incr_patch_hits;
        resident
            .apply_props(&json!({"nodes": {"r5": {"zIndex": 999}}}).to_string())
            .unwrap();
        caches.begin_frame();
        let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        assert_eq!(
            caches.incr_patch_hits, patch_before,
            "zIndex must NOT patch"
        );
        assert!(caches.incr_fallbacks > fb_before, "zIndex must fall back");
        let scene = resident.scene().unwrap();
        let (full, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
        assert_eq!(plan_items_fingerprint(&incr), plan_items_fingerprint(&full));
    }

    /// Editing a node that a BLEND node renders (one of its `sourceIds`) changes
    /// the blend's geometry, so it must fall back — and stay correct.
    #[test]
    fn incremental_falls_back_on_blend_source_edit() {
        let opts = editor_opts();
        let scene_json = moda_scene::value::to_string(&json!({
            "nodes": [
                {"id": "a", "type": "rectangle", "x": 10.0, "y": 10.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]},
                {"id": "b", "type": "rectangle", "x": 200.0, "y": 200.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}]},
                {"id": "bl", "type": "blend", "sourceIds": ["a", "b"], "spacing": "steps", "steps": 4},
                {"id": "c", "type": "rectangle", "x": 400.0, "y": 50.0, "width": 30.0, "height": 30.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00ff00"}}]}
            ],
            "pages": [{"id": "page-1", "name": "p", "width": 800.0, "height": 600.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                       "nodeIds": ["a", "b", "bl", "c"]}],
            "config": {"renderPageClipping": true},
            "canvasRootNodeIds": ["a", "b", "bl", "c"], "layoutMode": "freeform", "variables": [],
        })).unwrap();
        let mut resident = ResidentScene::default();
        resident.update(&scene_json).unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();

        // Editing 'a' (a blend source) must fall back; editing 'c' (independent
        // leaf) must patch.
        let fb_before = caches.incr_fallbacks;
        resident
            .apply_props(&json!({"nodes": {"a": {"x": 60.0}}}).to_string())
            .unwrap();
        caches.begin_frame();
        let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        assert!(
            caches.incr_fallbacks > fb_before,
            "blend-source edit must fall back"
        );
        let scene = resident.scene().unwrap();
        let (full, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
        assert_eq!(
            plan_items_fingerprint(&incr),
            plan_items_fingerprint(&full),
            "blend fallback diverged"
        );

        let patch_before = caches.incr_patch_hits;
        resident
            .apply_props(&json!({"nodes": {"c": {"x": 410.0}}}).to_string())
            .unwrap();
        caches.begin_frame();
        let (incr2, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        assert!(
            caches.incr_patch_hits > patch_before,
            "independent leaf must patch"
        );
        let scene = resident.scene().unwrap();
        let (full2, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
        assert_eq!(
            plan_items_fingerprint(&incr2),
            plan_items_fingerprint(&full2)
        );
    }

    /// A leaf nested INSIDE a blend source group is not itself in
    /// `blend_related_ids` (only the group is), but editing it changes the blend's
    /// geometry — so the ancestor guard must force a fallback, not an in-place
    /// patch, and the result must stay byte-exact with a full rebuild.
    #[test]
    fn incremental_falls_back_on_blend_source_descendant_edit() {
        let opts = editor_opts();
        let scene_json = moda_scene::value::to_string(&json!({
            "nodes": [
                {"id": "g", "type": "group", "x": 10.0, "y": 10.0, "children": ["child"]},
                {"id": "child", "type": "rectangle", "x": 0.0, "y": 0.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]},
                {"id": "b", "type": "rectangle", "x": 200.0, "y": 200.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}]},
                {"id": "bl", "type": "blend", "sourceIds": ["g", "b"], "spacing": "steps", "steps": 4},
                {"id": "c", "type": "rectangle", "x": 400.0, "y": 50.0, "width": 30.0, "height": 30.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00ff00"}}]}
            ],
            "pages": [{"id": "page-1", "name": "p", "width": 800.0, "height": 600.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                       "nodeIds": ["g", "b", "bl", "c"]}],
            "config": {"renderPageClipping": true},
            "canvasRootNodeIds": ["g", "b", "bl", "c"], "layoutMode": "freeform", "variables": [],
        })).unwrap();
        let mut resident = ResidentScene::default();
        resident.update(&scene_json).unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();

        // Direct guard check on the seeded index: 'child' (descendant of blend
        // source group 'g') is restricted; 'c' (independent leaf) is not.
        {
            let scene = resident.scene().unwrap();
            let index = &caches
                .incremental_page
                .as_ref()
                .expect("page slot seeded")
                .index;
            assert!(
                has_blend_or_mask_ancestor(scene, "child", index, &mut HashMap::default()),
                "a blend-source descendant must be flagged restricted"
            );
            assert!(
                !has_blend_or_mask_ancestor(scene, "c", index, &mut HashMap::default()),
                "an independent leaf must not be flagged restricted"
            );
        }

        // Editing 'child' (nested in blend source 'g') must fall back, not patch.
        let fb_before = caches.incr_fallbacks;
        resident
            .apply_props(&json!({"nodes": {"child": {"x": 5.0}}}).to_string())
            .unwrap();
        caches.begin_frame();
        let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        assert!(
            caches.incr_fallbacks > fb_before,
            "blend-source descendant edit must fall back"
        );
        let scene = resident.scene().unwrap();
        let (full, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
        assert_eq!(
            plan_items_fingerprint(&incr),
            plan_items_fingerprint(&full),
            "descendant fallback diverged from full rebuild"
        );
    }

    /// The kill switch forces the legacy full-rebuild path (never the patcher).
    #[test]
    fn incremental_kill_switch_forces_full_rebuild() {
        let opts = editor_opts();
        let mut resident = ResidentScene::default();
        resident.update(&rect_scene(10)).unwrap();
        let mut caches = PlanCaches::default();
        caches.set_incremental_enabled(false);
        caches.begin_frame();
        build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        resident
            .apply_props(&json!({"nodes": {"r2": {"x": 9.0}}}).to_string())
            .unwrap();
        caches.begin_frame();
        let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        assert_eq!(
            caches.incr_patch_hits, 0,
            "kill switch must prevent patching"
        );
        let scene = resident.scene().unwrap();
        let (full, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
        assert_eq!(plan_items_fingerprint(&incr), plan_items_fingerprint(&full));
    }

    /// renderer-perf-more-z6: a GROUP transform/opacity edit takes the subtree
    /// incremental path — it patches the group's leaf descendants (not a full
    /// rebuild) and stays byte-exact. The dominant interactive op (group drag).
    #[test]
    fn incremental_group_transform_patches_subtree() {
        let opts = editor_opts();
        let scene_json = moda_scene::value::to_string(&json!({
            "nodes": [
                {"id": "g", "type": "group", "x": 10.0, "y": 10.0, "rotation": 0, "scale": 1,
                 "opacity": 1, "children": ["a", "b", "c"]},
                {"id": "a", "type": "rectangle", "x": 0.0, "y": 0.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]},
                {"id": "b", "type": "rectangle", "x": 50.0, "y": 0.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00ff00"}}]},
                {"id": "c", "type": "rectangle", "x": 100.0, "y": 0.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}]}
            ],
            "pages": [{"id": "page-1", "name": "p", "width": 800.0, "height": 600.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                       "nodeIds": ["g"]}],
            "config": {"renderPageClipping": true},
            "canvasRootNodeIds": ["g"], "layoutMode": "freeform", "variables": [],
        }))
        .unwrap();
        let mut resident = ResidentScene::default();
        resident.update(&scene_json).unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();

        // A group transform/opacity drag patches the subtree (3 leaves), byte-exact.
        for e in [
            json!({"nodes": {"g": {"x": 25.0}}}),
            json!({"nodes": {"g": {"x": 30.0, "y": 5.0}}}),
            json!({"nodes": {"g": {"rotation": 15.0}}}),
        ] {
            let patch_before = caches.incr_patch_hits;
            let nodes_before = caches.incr_patch_nodes;
            resident.apply_props(&e.to_string()).unwrap();
            caches.begin_frame();
            let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
            assert!(
                caches.incr_patch_hits > patch_before,
                "group edit {e} should patch incrementally, not full-rebuild"
            );
            assert_eq!(
                caches.incr_patch_nodes,
                nodes_before + 3,
                "group edit {e} should rebuild EXACTLY the 3 subtree leaves (no over-expansion)"
            );
            let scene = resident.scene().unwrap();
            let (full, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
            assert_eq!(
                plan_items_fingerprint(&incr),
                plan_items_fingerprint(&full),
                "group edit {e}: incremental subtree patch diverged from full rebuild"
            );
        }
        // An opacity edit that makes the group composite as one unit adds its fade split: a full rebuild, which
        // matches a from-scratch plan.
        let patch_before = caches.incr_patch_hits;
        resident.apply_props(&json!({"nodes": {"g": {"opacity": 0.5}}}).to_string()).unwrap();
        caches.begin_frame();
        let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        assert_eq!(caches.incr_patch_hits, patch_before, "a group opacity edit rebuilds, it doesn't patch");
        let (full, _w) = crate::frame::build_coverage_plan(resident.scene().unwrap(), 0, &opts).unwrap();
        assert_eq!(plan_items_fingerprint(&incr), plan_items_fingerprint(&full));
    }

    /// renderer-perf-more-z6 (review finding): a CONTAINER with no fill/stroke emits
    /// NO body (so has no span in the index). A non-transform edit that makes it
    /// GAIN a body (`fill`/`stroke`) must NOT take the subtree fast path — the new
    /// body would be silently dropped (the parent isn't in `by_node`, so the walk
    /// skips it). The `SAFE_PARENT_KEYS` gate classifies it STRUCTURAL → full
    /// rebuild, which correctly emits the new body.
    #[test]
    fn incremental_falls_back_on_container_fill_gain() {
        let opts = editor_opts();
        let scene_json = moda_scene::value::to_string(&json!({
            "nodes": [
                // Container with children but no fill/stroke → emits no body item.
                {"id": "ct", "type": "container", "x": 50.0, "y": 50.0, "width": 200.0,
                 "height": 200.0, "rotation": 0, "scale": 1, "children": ["leaf"]},
                {"id": "leaf", "type": "rectangle", "x": 0.0, "y": 0.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]}
            ],
            "pages": [{"id": "page-1", "name": "p", "width": 800.0, "height": 600.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                       "nodeIds": ["ct"]}],
            "config": {"renderPageClipping": true},
            "canvasRootNodeIds": ["ct"], "layoutMode": "freeform", "variables": [],
        }))
        .unwrap();
        let mut resident = ResidentScene::default();
        resident.update(&scene_json).unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        let (before, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();

        let fb_before = caches.incr_fallbacks;
        let patch_before = caches.incr_patch_hits;
        // Add a fill to the previously body-less container.
        resident
            .apply_props(&json!({"nodes": {"ct": {"fills": solid("#00ff00")}}}).to_string())
            .unwrap();
        caches.begin_frame();
        let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        assert_eq!(
            caches.incr_patch_hits, patch_before,
            "a container fill-gain must NOT take the subtree patch path"
        );
        assert!(
            caches.incr_fallbacks > fb_before,
            "a container fill-gain must fall back to a full rebuild"
        );
        let scene = resident.scene().unwrap();
        let (full, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
        assert_eq!(
            plan_items_fingerprint(&incr),
            plan_items_fingerprint(&full),
            "container fill-gain fallback diverged from full rebuild"
        );
        // The fill-gain actually added an item (the container body) — proving the
        // fast path WOULD have served a stale (item-missing) plan without the gate.
        assert!(
            full.items.len() > before.items.len(),
            "the container fill-gain must add a body item to the plan"
        );
    }

    /// Nodes inside clipping frames (a container with `clipContent`, nested ones too) carry mask refs. An edit
    /// to one patches in place against the plan's own mask table (the owner's mask is unchanged), and matches a
    /// full rebuild; an edit to the clip owner itself changes its mask and falls back.
    #[test]
    fn incremental_patches_nodes_inside_clipping_frames() {
        let opts = editor_opts();
        let solid = |c: &str| json!([{"id": "fill-0", "paint": {"type": "solid", "color": c}}]);
        let scene_json = moda_scene::value::to_string(&json!({
            "nodes": [
                {"id": "f1", "type": "container", "x": 20.0, "y": 20.0, "width": 300.0, "height": 200.0, "rotation": 0, "scale": 1,
                 "clipContent": true, "cornerRadius": 12, "fills": solid("#eeeeee"), "children": ["a", "g", "f2"]},
                {"id": "a", "type": "rectangle", "x": 250.0, "y": 10.0, "width": 80.0, "height": 40.0, "rotation": 0, "scale": 1, "fills": solid("#ff0000")},
                {"id": "g", "type": "group", "x": 10.0, "y": 60.0, "rotation": 0, "scale": 1, "children": ["b"]},
                {"id": "b", "type": "path", "x": 0.0, "y": 0.0, "width": 40.0, "height": 40.0, "rotation": 0, "scale": 1, "closed": true,
                 "points": [{"type": "move", "x": 0, "y": 0}, {"type": "curve", "x": 40, "y": 40, "cp1": {"x": 30, "y": 0}, "cp2": {"x": 40, "y": 10}}, {"type": "line", "x": 0, "y": 40}],
                 "fills": solid("#00aa00")},
                {"id": "f2", "type": "container", "x": 120.0, "y": 100.0, "width": 100.0, "height": 60.0, "rotation": 0, "scale": 1,
                 "clipContent": true, "fills": solid("#dddddd"), "children": ["c"]},
                {"id": "c", "type": "rectangle", "x": 60.0, "y": 30.0, "width": 80.0, "height": 50.0, "rotation": 0, "scale": 1,
                 "fills": [{"id": "fill-0", "paint": {"type": "linear", "start": {"x": 0, "y": 0}, "end": {"x": 1, "y": 1},
                   "stops": [{"offset": 0, "color": "#0000ff"}, {"offset": 1, "color": "#00ffff"}]}}]}
            ],
            "pages": [{"id": "page-1", "name": "p", "width": 800.0, "height": 600.0, "fills": solid("#ffffff"), "nodeIds": ["f1"]}],
            "config": {"renderPageClipping": true},
            "canvasRootNodeIds": ["f1"], "layoutMode": "freeform", "variables": [],
        }))
        .unwrap();
        let mut resident = ResidentScene::default();
        resident.update(&scene_json).unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        let (seed, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        assert_eq!(seed.masks.len(), 2, "one mask per clipping frame");

        let edits = [
            json!({"nodes": {"a": {"x": 200.0}}}),
            json!({"nodes": {"a": {"fills": solid("#123456")}}}),
            json!({"nodes": {"b": {"y": 5.0}}}),
            json!({"nodes": {"g": {"x": 30.0}}}),
            // Inside the nested frame: its mask composes f1's.
            json!({"nodes": {"c": {"x": 10.0, "width": 120.0}}}),
            json!({"nodes": {"a": {"x": 1.0}, "c": {"y": 2.0}}}),
        ];
        for (i, e) in edits.iter().enumerate() {
            let patches_before = caches.incr_patch_hits;
            resident.apply_props(&e.to_string()).unwrap();
            caches.begin_frame();
            let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
            assert!(
                caches.incr_patch_hits > patches_before,
                "edit {i} ({e}): expected the incremental patch to engage"
            );
            let (full, _w) =
                crate::frame::build_coverage_plan(resident.scene().unwrap(), 0, &opts).unwrap();
            assert_eq!(
                plan_items_fingerprint(&incr),
                plan_items_fingerprint(&full),
                "edit {i} ({e}): incremental plan diverged from full rebuild"
            );
            assert_eq!(incr.masks, full.masks, "edit {i}: mask table");
        }

        // Resizing a clip owner changes its mask (and its nested frame's): full rebuild.
        for e in [
            json!({"nodes": {"f2": {"width": 140.0}}}),
            json!({"nodes": {"f1": {"height": 150.0}}}),
        ] {
            let fb_before = caches.incr_fallbacks;
            resident.apply_props(&e.to_string()).unwrap();
            caches.begin_frame();
            let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
            assert!(
                caches.incr_fallbacks > fb_before,
                "{e}: a clip owner edit must fall back"
            );
            let (full, _w) =
                crate::frame::build_coverage_plan(resident.scene().unwrap(), 0, &opts).unwrap();
            assert_eq!(
                plan_items_fingerprint(&incr),
                plan_items_fingerprint(&full),
                "{e}"
            );
            assert_eq!(incr.masks, full.masks, "{e}: mask table");
        }
    }

    /// renderer-perf-more-z6 (review finding): adding a `clipMaskNodeId` to a GROUP
    /// re-masks its descendants — a non-transform parent edit the subtree fast path
    /// cannot reason about, so the `SAFE_PARENT_KEYS` gate forces a full rebuild.
    #[test]
    fn incremental_falls_back_on_group_clip_mask() {
        let opts = editor_opts();
        let scene_json = moda_scene::value::to_string(&json!({
            "nodes": [
                {"id": "mask", "type": "rectangle", "x": 0.0, "y": 0.0, "width": 60.0, "height": 60.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}]},
                {"id": "g", "type": "group", "x": 10.0, "y": 10.0, "rotation": 0, "scale": 1, "children": ["a"]},
                {"id": "a", "type": "rectangle", "x": 0.0, "y": 0.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]}
            ],
            "pages": [{"id": "page-1", "name": "p", "width": 800.0, "height": 600.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                       "nodeIds": ["mask", "g"]}],
            "config": {"renderPageClipping": true},
            "canvasRootNodeIds": ["mask", "g"], "layoutMode": "freeform", "variables": [],
        }))
        .unwrap();
        let mut resident = ResidentScene::default();
        resident.update(&scene_json).unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();

        let fb_before = caches.incr_fallbacks;
        let patch_before = caches.incr_patch_hits;
        resident
            .apply_props(&json!({"nodes": {"g": {"clipMaskNodeId": "mask"}}}).to_string())
            .unwrap();
        caches.begin_frame();
        let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        assert_eq!(
            caches.incr_patch_hits, patch_before,
            "a group clipMaskNodeId edit must NOT take the subtree patch path"
        );
        assert!(
            caches.incr_fallbacks > fb_before,
            "a group clipMaskNodeId edit must fall back to a full rebuild"
        );
        let scene = resident.scene().unwrap();
        let (full, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
        assert_eq!(
            plan_items_fingerprint(&incr),
            plan_items_fingerprint(&full),
            "group clip-mask fallback diverged from full rebuild"
        );
    }

    /// renderer-perf-more-z6 (review gap 5.6): NESTED groups (group → group → leaf).
    /// The subtree walk must recurse past depth 1, so an edit to the OUTER group
    /// rebuilds the deep leaves and stays byte-exact.
    #[test]
    fn incremental_nested_group_transform_matches_full_rebuild() {
        let opts = editor_opts();
        let scene_json = moda_scene::value::to_string(&json!({
            "nodes": [
                {"id": "g1", "type": "group", "x": 10.0, "y": 10.0, "rotation": 0, "scale": 1,
                 "opacity": 1, "children": ["g2", "d"]},
                {"id": "g2", "type": "group", "x": 5.0, "y": 5.0, "rotation": 0, "scale": 1,
                 "opacity": 1, "children": ["a", "b"]},
                {"id": "a", "type": "rectangle", "x": 0.0, "y": 0.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]},
                {"id": "b", "type": "rectangle", "x": 50.0, "y": 0.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00ff00"}}]},
                {"id": "d", "type": "rectangle", "x": 0.0, "y": 60.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}]}
            ],
            "pages": [{"id": "page-1", "name": "p", "width": 800.0, "height": 600.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                       "nodeIds": ["g1"]}],
            "config": {"renderPageClipping": true},
            "canvasRootNodeIds": ["g1"], "layoutMode": "freeform", "variables": [],
        }))
        .unwrap();
        let mut resident = ResidentScene::default();
        resident.update(&scene_json).unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();

        // Edit the OUTER group: its subtree is {g2→{a,b}, d} = 3 item-producing
        // leaves nested up to 2 deep. All three must be patched (the walk recurses).
        let patch_before = caches.incr_patch_hits;
        let nodes_before = caches.incr_patch_nodes;
        resident
            .apply_props(&json!({"nodes": {"g1": {"x": 33.0, "rotation": 10.0}}}).to_string())
            .unwrap();
        caches.begin_frame();
        let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        assert!(
            caches.incr_patch_hits > patch_before,
            "nested-group edit should patch incrementally"
        );
        assert_eq!(
            caches.incr_patch_nodes,
            nodes_before + 3,
            "the outer-group walk must reach all 3 nested leaves"
        );
        let scene = resident.scene().unwrap();
        let (full, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
        assert_eq!(
            plan_items_fingerprint(&incr),
            plan_items_fingerprint(&full),
            "nested-group subtree patch diverged from full rebuild"
        );

        // And an edit to the INNER group patches only ITS two leaves.
        let nodes_before = caches.incr_patch_nodes;
        resident
            .apply_props(&json!({"nodes": {"g2": {"x": 7.0}}}).to_string())
            .unwrap();
        caches.begin_frame();
        let (incr2, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        assert_eq!(
            caches.incr_patch_nodes,
            nodes_before + 2,
            "inner-group edit must patch only its 2 leaves"
        );
        let scene = resident.scene().unwrap();
        let (full2, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
        assert_eq!(
            plan_items_fingerprint(&incr2),
            plan_items_fingerprint(&full2),
            "inner-group subtree patch diverged from full rebuild"
        );
    }

    /// renderer-perf-more-z6 (review gap 5.1): the CAMERA lane (the editor's real
    /// viewport) subtree-incremental path is byte-exact for a group edit. The page
    /// lane has unit coverage; the camera lane shares `apply_incremental_patch` but
    /// runs the `Arc::make_mut` clone path — verify its draw stream matches a
    /// from-scratch combined camera build.
    #[test]
    fn incremental_camera_group_edit_matches_full_rebuild() {
        let opts = editor_opts();
        let scene_json = moda_scene::value::to_string(&json!({
            "nodes": [
                {"id": "g", "type": "group", "x": 10.0, "y": 10.0, "rotation": 0, "scale": 1,
                 "opacity": 1, "children": ["a", "b", "c"]},
                {"id": "a", "type": "rectangle", "x": 0.0, "y": 0.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]},
                {"id": "b", "type": "rectangle", "x": 50.0, "y": 0.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00ff00"}}]},
                {"id": "c", "type": "rectangle", "x": 100.0, "y": 0.0, "width": 40.0, "height": 40.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}]}
            ],
            "pages": [{"id": "page-1", "name": "p", "width": 800.0, "height": 600.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                       "nodeIds": ["g"]}],
            "config": {"renderPageClipping": true},
            "canvasRootNodeIds": ["g"], "layoutMode": "freeform", "variables": [],
        }))
        .unwrap();
        let cam = crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom: 1.0,
            viewport_w: 800.0,
            viewport_h: 600.0,
            dpr: 1.0,
            clear_rgb: [1.0, 1.0, 1.0],
        };
        let mut resident = ResidentScene::default();
        resident.update(&scene_json).unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        // Seed the camera incremental slot.
        build_camera_plan_retained(&resident, &mut caches, &[0], &cam, &opts)
            .unwrap()
            .expect("camera plan built");

        let patch_before = caches.incr_patch_hits;
        resident
            .apply_props(&json!({"nodes": {"g": {"x": 40.0, "opacity": 0.6}}}).to_string())
            .unwrap();
        caches.begin_frame();
        let (incr, _w) = build_camera_plan_retained(&resident, &mut caches, &[0], &cam, &opts)
            .unwrap()
            .expect("camera re-plan");
        assert!(
            caches.incr_patch_hits > patch_before,
            "camera group edit should patch the subtree incrementally"
        );
        // Byte-exact vs a from-scratch combined camera build.
        let scene = resident.scene().unwrap();
        let hashes = resident.hashes().unwrap();
        let (full, _w) = crate::frame::build_camera_plan_cached(
            scene,
            &[0],
            &cam,
            &opts,
            Some((&mut PlanCaches::default(), hashes)),
            None,
        )
        .unwrap()
        .expect("full camera build");
        assert_eq!(
            plan_items_fingerprint(&incr),
            plan_items_fingerprint(&full),
            "camera-lane group subtree patch diverged from a full camera rebuild"
        );
    }

    /// A SINGLE re-threadable shader fill on the canvas must NOT disable
    /// incrementality for the whole frame. The CAMERA seed gate now admits a
    /// re-threadable shader plan, so dragging an UNRELATED non-shader node patches
    /// incrementally (`incr_patch_hits` increments) instead of falling back to a full
    /// O(scene) rebuild every frame (the reported bug: `incr_patch_hits` stuck at 0).
    /// Exercises the CLOCK-ADVANCE path (seed at T0, serve at T1) where the bug
    /// lives, proves the verify-oracle normalization reconciles the seed-clock plan
    /// with a current-clock rebuild (MB-2), and guards that the PAGE lane still
    /// refuses to seed a shader plan (Fix 1).
    #[test]
    fn camera_incremental_serves_when_unrelated_shader_present() {
        let opts = editor_opts();
        let scene_json = moda_scene::value::to_string(&json!({
            "nodes": [
                {"id": "s", "type": "rectangle", "x": 0.0, "y": 0.0, "width": 200.0, "height": 150.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "shader", "shaderId": "clouds"}}]},
                {"id": "a", "type": "rectangle", "x": 300.0, "y": 0.0, "width": 40.0, "height": 40.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]}
            ],
            "pages": [{"id": "page-1", "name": "p", "width": 800.0, "height": 600.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                       "nodeIds": ["s", "a"]}],
            "config": {"renderPageClipping": true},
            "layoutMode": "freeform", "variables": [],
        }))
        .unwrap();
        let cam = crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom: 1.0,
            viewport_w: 800.0,
            viewport_h: 600.0,
            dpr: 1.0,
            clear_rgb: [1.0, 1.0, 1.0],
        };
        let mut resident = ResidentScene::default();
        resident.update(&scene_json).unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        // Cold build seeds the camera incremental slot — even though a shader fill
        // is present, because that shader is re-threadable.
        let (cold, _w) = build_camera_plan_retained(&resident, &mut caches, &[0], &cam, &opts)
            .unwrap()
            .expect("camera plan built");
        assert!(
            plan_uses_shader_time(&cold) && plan_shader_time_rethreadable(&cold),
            "scene has a re-threadable shader fill"
        );
        assert!(
            caches.camera_slot.is_some(),
            "ENG-3601: re-threadable shader plan must seed the incremental slot"
        );

        // A static re-render (no edit, clock unchanged) serves from the slot.
        let static_before = caches.incr_static_hits;
        caches.begin_frame();
        build_camera_plan_retained(&resident, &mut caches, &[0], &cam, &opts)
            .unwrap()
            .expect("camera re-plan");
        assert_eq!(
            caches.incr_static_hits,
            static_before + 1,
            "static re-render with a shader present must serve via exactly one incremental static hit"
        );

        // Clock-advance — the path the bug lives on: seeded at clock
        // T0=0.0, serve a static re-render at a DIFFERENT clock T1. A re-threadable
        // shader drops the clock from the slot's identity (it rides the CAMERA
        // present override, applied to a copy at present), so this STILL serves via
        // the incremental static lane, reusing the SEED-clock plan Arc.
        let opts_t1 = RenderOptions {
            shader_time_s: 0.75,
            ..editor_opts()
        };
        let static_before = caches.incr_static_hits;
        caches.begin_frame();
        let (warm_t1, _w) =
            build_camera_plan_retained(&resident, &mut caches, &[0], &cam, &opts_t1)
                .unwrap()
                .expect("camera re-plan at T1");
        assert_eq!(
            caches.incr_static_hits,
            static_before + 1,
            "clock-advance static frame must serve via exactly one incremental static hit (clock not in identity)"
        );
        assert!(
            Arc::ptr_eq(&cold, &warm_t1),
            "clock-advance serve reuses the seed-clock plan Arc (override re-threads at present)"
        );
        // The served plan is verify-CONSISTENT with a fresh build at T1 once the
        // re-threaded clock slots are normalized — the cached items legitimately
        // carry the SEED clock; the override applies T1 to a copy.
        let scene = resident.scene().unwrap();
        let hashes = resident.hashes().unwrap();
        let (full_t1, _w) = crate::frame::build_camera_plan_cached(
            scene,
            &[0],
            &cam,
            &opts_t1,
            Some((&mut PlanCaches::default(), hashes)),
            None,
        )
        .unwrap()
        .expect("full camera build at T1");
        assert_eq!(
            plan_items_fingerprint_verify(&warm_t1),
            plan_items_fingerprint_verify(&full_t1),
            "clock-normalized served plan diverged from a fresh build at T1"
        );
        // ...and the RAW fingerprints DIFFER, proving the clock genuinely advanced in
        // the rebuild (so the normalization is doing real work — exactly the false
        // positive `MODA_INCREMENTAL_VERIFY` would otherwise panic on, MB-2).
        assert_ne!(
            plan_items_fingerprint(&warm_t1),
            plan_items_fingerprint(&full_t1),
            "raw fingerprints must differ across clocks (else the normalization proves nothing)"
        );

        // Drag the UNRELATED non-shader node at the ADVANCED clock T1 — the reported
        // gesture, exercising the mixed path: the seed-clock shader span for `s` is
        // reused untouched while `a` is freshly rebuilt at the current clock. It must
        // be SERVED BY THE PATCH LANE (exactly one patch hit, no full-rebuild
        // fallback), not silently full-rebuild (the exact bug this guards against).
        let patch_before = caches.incr_patch_hits;
        let misses_before = caches.camera_misses;
        resident
            .apply_props(&json!({"nodes": {"a": {"x": 320.0}}}).to_string())
            .unwrap();
        caches.begin_frame();
        let (incr, _w) = build_camera_plan_retained(&resident, &mut caches, &[0], &cam, &opts_t1)
            .unwrap()
            .expect("camera re-plan");
        assert_eq!(
            caches.incr_patch_hits,
            patch_before + 1,
            "ENG-3601: dragging an unrelated node with a shader present must serve via exactly one patch"
        );
        assert_eq!(
            caches.camera_misses, misses_before,
            "the drag must NOT fall back to a full camera rebuild"
        );

        // Byte-exact MODULO the re-threaded shader clock vs a from-scratch build at
        // T1: `incr` carries `s` at the SEED clock (the override re-threads at
        // present), the fresh `full` bakes T1 — so compare with the clock-normalized
        // fingerprint (the same the camera verify oracle uses).
        let scene = resident.scene().unwrap();
        let hashes = resident.hashes().unwrap();
        let (full, _w) = crate::frame::build_camera_plan_cached(
            scene,
            &[0],
            &cam,
            &opts_t1,
            Some((&mut PlanCaches::default(), hashes)),
            None,
        )
        .unwrap()
        .expect("full camera build");
        assert_eq!(
            plan_items_fingerprint_verify(&incr),
            plan_items_fingerprint_verify(&full),
            "incrementally-served plan (shader present) diverged from a full rebuild at T1"
        );

        // Guard: the PAGE lane must NOT seed a shader plan — its
        // present path (`present_page`/`present_page_to_canvas`/`render_page`) never
        // passes a present override, so a seeded re-threadable shader would FREEZE at
        // the seed clock during page/slideshow playback. Use a fresh cache so this is
        // independent of the camera flow above.
        let mut page_caches = PlanCaches::default();
        page_caches.begin_frame();
        let (page_plan, _w) = build_plan_retained(&resident, &mut page_caches, 0, &opts).unwrap();
        assert!(
            plan_uses_shader_time(&page_plan),
            "the page plan still carries the shader fill"
        );
        assert!(
            page_caches.incremental_page.is_none(),
            "ENG-3601 Fix 1: the PAGE lane must NOT seed a re-threadable shader plan"
        );
    }

    /// FIX A: the `first_camera_item_divergence` replay diagnostic must
    /// use the SAME clock-normalized compare as the camera verify oracle — a served
    /// seed-clock plan vs a current-clock rebuild differs ONLY in a re-threadable
    /// shader's `shader_params` clock slots (which ride the present override), so it
    /// must NOT be reported as a divergence. (Latent today because `edit_bench`
    /// pins `shader_time_s: 0.0`; this pins the contract.)
    #[test]
    fn first_camera_item_divergence_ignores_rethreaded_clock() {
        let scene_json = moda_scene::value::to_string(&json!({
            "nodes": [
                {"id": "s", "type": "rectangle", "x": 0.0, "y": 0.0, "width": 200.0, "height": 150.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "shader", "shaderId": "clouds"}}]}
            ],
            "pages": [{"id": "page-1", "name": "p", "width": 800.0, "height": 600.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                       "nodeIds": ["s"]}],
            "config": {"renderPageClipping": true},
            "layoutMode": "freeform", "variables": [],
        }))
        .unwrap();
        let cam = crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom: 1.0,
            viewport_w: 800.0,
            viewport_h: 600.0,
            dpr: 1.0,
            clear_rgb: [1.0, 1.0, 1.0],
        };
        let mut resident = ResidentScene::default();
        resident.update(&scene_json).unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        // The "got" plan: built (and clock-baked) at T0 = 0.0.
        let opts_t0 = editor_opts();
        let (got, _w) = build_camera_plan_retained(&resident, &mut caches, &[0], &cam, &opts_t0)
            .unwrap()
            .expect("camera plan built");
        assert!(
            plan_uses_shader_time(&got) && plan_shader_time_rethreadable(&got),
            "scene has a re-threadable shader fill"
        );

        // A from-scratch rebuild at T1 = 0.75 bakes a DIFFERENT clock — raw
        // fingerprints differ, so the diagnostic would (pre-FIX-A) report a bogus
        // divergence...
        let opts_t1 = RenderOptions {
            shader_time_s: 0.75,
            ..editor_opts()
        };
        let scene = resident.scene().unwrap();
        let hashes = resident.hashes().unwrap();
        let (full_t1, _w) = crate::frame::build_camera_plan_cached(
            scene,
            &[0],
            &cam,
            &opts_t1,
            Some((&mut PlanCaches::default(), hashes)),
            None,
        )
        .unwrap()
        .expect("full camera build at T1");
        assert_ne!(
            plan_items_fingerprint(&got),
            plan_items_fingerprint(&full_t1),
            "raw fingerprints must differ across clocks (else the test proves nothing)"
        );

        // ...but FIX A normalizes the re-threaded clock slots, so no divergence is
        // reported for the clock-only difference.
        assert_eq!(
            first_camera_item_divergence(scene, &[0], &cam, &opts_t1, hashes, &got),
            None,
            "clock-only re-threaded difference must NOT be reported as a divergence"
        );
    }

    /// renderer-perf-more-z6 (review gap 5.2): the cold-build display-list dedup
    /// (Experiment C) is byte-identical. The fingerprint computed from a prebuilt DL
    /// equals the one that builds its own DL, and a plan built with the shared DL
    /// equals one built without — the whole correctness argument is "the DL is a
    /// pure fn of (scene, page)", so pin it.
    #[test]
    fn cold_dl_dedup_is_byte_identical() {
        let opts = editor_opts();
        let value: Value = moda_scene::value::from_str(&rect_scene(30)).unwrap();
        let scene = Scene::load(moda_scene::extract_canvas_data(&value)).unwrap();
        let hashes = node_hashes_for_scene(&scene);
        let page_id = scene.doc().pages[0].id().to_string();
        let dl = moda_scene::build_display_list(&scene, &page_id);

        // Fingerprint: with vs without a prebuilt DL.
        assert_eq!(
            page_fingerprint(&scene, 0, &opts, &hashes),
            page_fingerprint_with_dl(&scene, 0, &opts, &hashes, &dl),
            "page_fingerprint_with_dl diverged from page_fingerprint"
        );

        // Plan: build with the shared DL vs let the builder build its own.
        let (without_index, _) = crate::frame::build_coverage_plan_cached_with_dl(
            &scene,
            0,
            &opts,
            Some((&mut PlanCaches::default(), &hashes)),
            Some(dl.clone()),
        )
        .unwrap();
        let (with_dl, _w, _aux) = crate::frame::build_coverage_plan_indexed_with_dl(
            &scene,
            0,
            &opts,
            Some((&mut PlanCaches::default(), &hashes)),
            Some(dl),
        )
        .unwrap();
        let (without_dl, _w, _aux) = crate::frame::build_coverage_plan_indexed(
            &scene,
            0,
            &opts,
            Some((&mut PlanCaches::default(), &hashes)),
        )
        .unwrap();
        assert_eq!(
            plan_items_fingerprint(&with_dl),
            plan_items_fingerprint(&without_dl),
            "prebuilt-DL plan diverged from a self-built-DL plan"
        );
        assert_eq!(
            plan_items_fingerprint(&without_index),
            plan_items_fingerprint(&with_dl),
            "omitting the unused plan index changed the plan"
        );
    }

    /// renderer-perf-more-z6 (review gap 5.3): a tess-cache HIT must hand the plan
    /// the SAME geometry `Arc` (pointer-equal), not a deep copy — that sharing is
    /// the whole point of the Arc-geometry change (Experiment D). Two builds over a
    /// warm cache should yield ptr-equal body geometry; a regression to deep copies
    /// would fail here while still passing every byte-exact test.
    ///
    /// Uses `stroked_rect_scene`: plain solid rects (and now solid-stroked CURVED
    /// rects) are analytic SDF bodies that never populate the tess
    /// cache, so a SHARP stroked rect — which stays on tessellation — is what
    /// gives this test a `DrawItem::Body` to share.
    #[test]
    fn tess_cache_shares_geometry_arc() {
        use crate::seams::DrawItem;
        let opts = editor_opts();
        let value: Value = moda_scene::value::from_str(&stroked_rect_scene(8)).unwrap();
        let scene = Scene::load(moda_scene::extract_canvas_data(&value)).unwrap();
        let hashes = node_hashes_for_scene(&scene);
        let mut caches = PlanCaches::default();

        // First build populates the tess cache; second build hits it.
        let (p1, _w, _a) = crate::frame::build_coverage_plan_indexed(
            &scene,
            0,
            &opts,
            Some((&mut caches, &hashes)),
        )
        .unwrap();
        let (p2, _w, _a) = crate::frame::build_coverage_plan_indexed(
            &scene,
            0,
            &opts,
            Some((&mut caches, &hashes)),
        )
        .unwrap();

        let first_body_geom = |plan: &CoveragePlan| {
            plan.items.iter().find_map(|it| match it {
                DrawItem::Body { geometry, .. } => Some(geometry.clone()),
                _ => None,
            })
        };
        let g1 = first_body_geom(&p1).expect("a body in plan 1");
        let g2 = first_body_geom(&p2).expect("a body in plan 2");
        assert!(
            std::sync::Arc::ptr_eq(&g1, &g2),
            "a tess-cache hit must share the geometry Arc with the plan (no deep copy)"
        );
    }

    // -----------------------------------------------------------------------
    // Scoped transient-host re-expansion
    // -----------------------------------------------------------------------

    /// One `table` host with `n_rows` x `n_cols` cells, plus a plain rect so the
    /// page is never all-transient. Mirrors the shape the stress-scene generator
    /// and the fuzz corpus emit (`tools/bench/gen-stress-scenes.py::_table`), with
    /// the same load-bearing invariant: `width`/`height` ARE the axis sums, so a
    /// row-resize patch must move `height` too or the scene is malformed.
    fn table_scene(n_rows: usize, n_cols: usize) -> Value {
        let row_size = 24.0_f64;
        let col_size = 60.0_f64;
        let cells: Vec<Value> = (0..n_rows)
            .map(|r| {
                let row: Vec<Value> = (0..n_cols).map(|c| table_cell(r, c)).collect();
                json!(row)
            })
            .collect();
        json!({
            "nodes": [
                {
                    "id": "t1", "type": "table", "x": 20.0, "y": 30.0,
                    "width": col_size * n_cols as f64, "height": row_size * n_rows as f64,
                    "rotation": 0, "scale": 1,
                    "rows": (0..n_rows).map(|_| json!({"size": row_size})).collect::<Vec<_>>(),
                    "columns": (0..n_cols).map(|_| json!({"size": col_size})).collect::<Vec<_>>(),
                    "cells": cells,
                    "rowStrokeConfig": "all", "columnStrokeConfig": "all",
                    "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#c9c9c9"}, "width": 1}], "backgroundColors": ["#ffffff", "#f4f4f6"],
                    "headerBackgroundColor": "#3a86ff",
                },
                {"id": "r0", "type": "rectangle", "x": 500.0, "y": 10.0,
                 "width": 40.0, "height": 30.0, "rotation": 0, "scale": 1, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}]},
            ],
            "pages": [{"id": "page-1", "name": "p", "width": 900.0, "height": 700.0,
                       "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}], "nodeIds": ["t1", "r0"]}],
            "config": {"renderPageClipping": true},
            "canvasRootNodeIds": ["t1", "r0"], "layoutMode": "freeform", "variables": [],
        })
    }

    /// Checkerboard cell text (half the cells carry richtext) so the expansion
    /// exercises both the text and the bare-rect lanes — and so an "empty cell
    /// gains text" case exists to prove the shape-match fallback.
    fn table_cell(r: usize, c: usize) -> Value {
        if (r + c) % 2 == 0 {
            table_text_cell(r, c, &format!("R{r}C{c}"))
        } else {
            json!({"id": format!("t1-c{r}-{c}")})
        }
    }

    fn table_text_cell(r: usize, c: usize, text: &str) -> Value {
        json!({
            "id": format!("t1-c{r}-{c}"),
            "textContent": format!(
                "<p style=\"font-family: Inter; font-size: 13px; color: #222222\">{text}</p>"
            ),
            "textHorizontalAlign": "left",
            "textVerticalAlign": "middle",
            "textPadding": 6,
        })
    }

    /// One `generated` chart host (`native-children`, so its transients are
    /// host-LOCAL and inherit the host transform at paint time) plus a rect.
    fn chart_scene() -> Value {
        json!({
            "nodes": [
                {
                    "id": "ch1", "type": "generated", "generatedKind": "chart",
                    "x": 40.0, "y": 40.0, "width": 300.0, "height": 200.0,
                    "rotation": 0, "scale": 1,
                    "renderMode": "native-children",
                    "generatedSource": {
                        "kind": "chart", "chartVersion": 1, "spec": {"type": "bar"},
                        "data": {"columns": [], "rows": [], "series": [],
                                 "categories": [], "valueRange": [0, 0]}
                    }
                },
                {"id": "r0", "type": "rectangle", "x": 500.0, "y": 10.0,
                 "width": 40.0, "height": 30.0, "rotation": 0, "scale": 1, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}]},
            ],
            "pages": [{"id": "page-1", "name": "p", "width": 900.0, "height": 700.0,
                       "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}], "nodeIds": ["ch1", "r0"]}],
            "config": {"renderPageClipping": true},
            "canvasRootNodeIds": ["ch1", "r0"], "layoutMode": "freeform", "variables": [],
        })
    }

    /// The headline scoped re-expansion guarantee: a structure-preserving prop edit on a
    /// TRANSIENT HOST (table cell text, an axis resize, a host move, a stroke-config
    /// flip) engages the incremental patch instead of forcing a whole-scene resync,
    /// AND the patched plan is byte-(fingerprint-)identical to a from-scratch build
    /// of the edited scene. Previously every one of these bailed outright.
    #[test]
    fn incremental_table_host_edits_match_full_rebuild() {
        let opts = editor_opts();
        let mut resident = ResidentScene::default();
        resident
            .update(&moda_scene::value::to_string(&table_scene(4, 3)).unwrap())
            .unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        build_plan_retained(&resident, &mut caches, 0, &opts).unwrap(); // seed index

        // The full 4x3 checkerboard with one cell RE-TYPED (c0-0 keeps text, so no
        // item appears or disappears — see the empty->non-empty fallback test).
        let retyped: Vec<Value> = (0..4)
            .map(|r| {
                let row: Vec<Value> = (0..3)
                    .map(|c| {
                        if (r, c) == (0, 0) {
                            table_text_cell(r, c, "EDITED")
                        } else {
                            table_cell(r, c)
                        }
                    })
                    .collect();
                json!(row)
            })
            .collect();

        // Every edit keeps the row/column COUNT (→ the same transient id sequence),
        // so each must take the patch lane. The axis resizes carry `height`/`width`
        // with `rows`/`columns` to preserve the sum invariant the expansion assumes.
        let edits = [
            json!({"nodes": {"t1": {"cells": retyped}}}),
            json!({"nodes": {"t1": {
                "rows": [{"size": 40.0}, {"size": 24.0}, {"size": 24.0}, {"size": 24.0}],
                "height": 112.0
            }}}),
            json!({"nodes": {"t1": {
                "columns": [{"size": 90.0}, {"size": 60.0}, {"size": 60.0}],
                "width": 210.0
            }}}),
            // Host MOVE. `expand_table_node` BAKES the host x/y into every cell and
            // border coordinate, so this is precisely the edit that NEEDS
            // re-expansion — a "transform-only edits are cheap" shortcut that reused
            // the old subtree would render the table at its old position.
            json!({"nodes": {"t1": {"x": 120.0, "y": 90.0}}}),
            // Host opacity: multiplies into every transient's baked opacity.
            json!({"nodes": {"t1": {"opacity": 0.5}}}),
        ];
        for (i, e) in edits.iter().enumerate() {
            let patches_before = caches.incr_patch_hits;
            resident
                .apply_props(&moda_scene::value::to_string(e).unwrap())
                .unwrap();
            caches.begin_frame();
            let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
            assert!(
                caches.incr_patch_hits > patches_before,
                "table-host edit {i} ({e}): expected the incremental patch to engage \
                 (got a full rebuild)"
            );
            let scene = resident.scene().unwrap();
            let (full, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
            assert_eq!(
                plan_items_fingerprint(&incr),
                plan_items_fingerprint(&full),
                "table-host edit {i} ({e}): incremental plan diverged from a full rebuild"
            );
        }
        let (reexpands, structural) = resident.host_reexpand_stats();
        assert_eq!(
            reexpands,
            edits.len() as u64,
            "every host patch must have re-expanded exactly once"
        );
        assert_eq!(
            structural, 0,
            "none of these count-preserving edits may be classified structural"
        );
    }

    /// A chart (`generated`, `native-children`) host re-expands too — including a
    /// `generatedSource.data` change, the edit that actually reshapes the chart.
    #[test]
    fn incremental_chart_host_edits_match_full_rebuild() {
        let opts = editor_opts();
        let mut resident = ResidentScene::default();
        resident
            .update(&moda_scene::value::to_string(&chart_scene()).unwrap())
            .unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        // Setup sanity: the host really does expand (else this test is vacuous).
        assert!(
            resident
                .scene()
                .unwrap()
                .transient_by_parent()
                .get("ch1")
                .is_some_and(|kids| !kids.is_empty()),
            "chart host must expand into at least one transient child"
        );

        let edits = [
            json!({"nodes": {"ch1": {"x": 140.0}}}),
            json!({"nodes": {"ch1": {"width": 360.0}}}),
            json!({"nodes": {"ch1": {"generatedSource": {
                "kind": "chart", "chartVersion": 1, "spec": {"type": "bar"},
                "data": {"columns": ["a"], "rows": [[1.0]], "series": ["s"],
                         "categories": ["c"], "valueRange": [0.0, 1.0]}
            }}}}),
        ];
        for (i, e) in edits.iter().enumerate() {
            let fb_before = caches.incr_fallbacks;
            let patch_before = caches.incr_patch_hits;
            resident
                .apply_props(&moda_scene::value::to_string(e).unwrap())
                .unwrap();
            caches.begin_frame();
            let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
            // A DATA edit may legitimately RESHAPE the expansion (an empty-data
            // placeholder becoming real bars changes the transient id sequence), so
            // assert correctness unconditionally and only require that ONE of the two
            // routes ran — never a silent no-op serving stale pixels.
            assert!(
                caches.incr_patch_hits > patch_before || caches.incr_fallbacks > fb_before,
                "chart-host edit {i} ({e}): neither patched nor fell back — stale serve"
            );
            let scene = resident.scene().unwrap();
            let (full, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
            assert_eq!(
                plan_items_fingerprint(&incr),
                plan_items_fingerprint(&full),
                "chart-host edit {i} ({e}): plan diverged from a full rebuild"
            );
        }
        let (reexpands, _) = resident.host_reexpand_stats();
        assert_eq!(
            reexpands,
            edits.len() as u64,
            "every chart-host patch must have re-expanded"
        );
    }

    /// The dirty log records ONLY the host, never its transients. A 20x20 table
    /// expands to ~1.6k children; logging each would exhaust `DIRTY_LOG_CAP` (4096)
    /// within a few edits and force every plan index older than the window into a
    /// full rebuild — i.e. it would hand back the whole win. `dirty_since` must
    /// report exactly one node for a one-host patch.
    #[test]
    fn apply_props_under_a_boolean_node_resyncs() {
        // E7.1: a boolean draws a path derived from its children, so a child patch (or a new
        // operation) must re-feed; paint on the boolean itself patches in place.
        crate::install_scene_seams();
        let doc = json!({
            "nodes": [
                {"id": "b", "type": "boolean", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
                 "children": ["r1", "r2"], "booleanOperation": "union"},
                {"id": "r1", "type": "rectangle", "x": 0, "y": 0, "width": 50, "height": 50, "rotation": 0, "scale": 1},
                {"id": "r2", "type": "rectangle", "x": 25, "y": 25, "width": 50, "height": 50, "rotation": 0, "scale": 1},
            ],
            "pages": [{"id": "p", "width": 200, "height": 200, "nodeIds": ["b"]}]
        });
        let mut resident = ResidentScene::default();
        resident.update(&doc.to_string()).unwrap();
        assert!(resident
            .apply_props(&json!({"nodes": {"r2": {"x": 60.0}}}).to_string())
            .is_err());
        resident.update(&doc.to_string()).unwrap();
        assert!(resident
            .apply_props(&json!({"nodes": {"b": {"booleanOperation": "subtract"}}}).to_string())
            .is_err());
        resident.update(&doc.to_string()).unwrap();
        resident
            .apply_props(&json!({"nodes": {"b": {"opacity": 0.5}}}).to_string())
            .unwrap();
        let drawn = resident.scene().unwrap().effective_node("b").unwrap();
        assert_eq!(drawn.node_type(), "path");
        assert_eq!(drawn.f64("opacity"), Some(0.5));
    }

    /// The motion graph's chart parts (moda-motion host.rs "Chart parts") reach the editor's resident scene as a
    /// `_motionParts` prop on the chart: the host re-expands with each keyed part's fields written in.
    #[test]
    fn apply_props_motion_parts_re_expand_a_chart() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/chart-bar.json"
        );
        let doc: Value = moda_scene::value::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let doc = doc.get("canvasData").cloned().unwrap_or(doc);
        let mut resident = ResidentScene::default();
        resident.update(&doc.to_string()).unwrap();
        let bar = |r: &ResidentScene| {
            let s = r.scene().unwrap();
            s.transient_by_parent()["gn-chart"]
                .iter()
                .filter_map(|id| s.effective_node(id))
                .find(|n| {
                    n.object("metadata")
                        .and_then(|m| m.get("name"))
                        .and_then(Value::as_str)
                        == Some("Bar Chart Bar Q2 Profit")
                })
                .map(|n| (n.id().to_string(), n.f64("height").unwrap()))
                .unwrap()
        };
        let (id, h) = bar(&resident);
        assert!(h > 10.0, "{h}");
        resident
            .apply_props(&json!({"nodes": {"gn-chart": {"_motionParts": {"bar:Q2:Profit": {"height": 7.5}}}}}).to_string())
            .unwrap();
        assert_eq!(
            bar(&resident),
            (id, 7.5),
            "the part's field, on the same transient id"
        );
    }

    #[test]
    fn apply_props_host_patch_logs_only_the_host() {
        let mut resident = ResidentScene::default();
        resident
            .update(&moda_scene::value::to_string(&table_scene(20, 20)).unwrap())
            .unwrap();
        let expanded = resident
            .scene()
            .unwrap()
            .transient_by_parent()
            .get("t1")
            .map(Vec::len)
            .unwrap_or(0);
        assert!(
            expanded > 400,
            "setup: a 20x20 table must expand widely (got {expanded}) or the cap \
             argument this test guards is vacuous"
        );
        let gen_before = resident.generation();
        resident
            .apply_props(&json!({"nodes": {"t1": {"x": 33.0}}}).to_string())
            .unwrap();
        match resident.dirty_since(gen_before) {
            DirtySince::Nodes(nodes) => assert_eq!(
                nodes,
                vec!["t1".to_string()],
                "a host patch must log exactly one dirty node (the host)"
            ),
            other => panic!("expected Nodes([host]), got {other:?}"),
        }
    }

    /// Adding a row changes the transient id SEQUENCE, so the re-expansion is
    /// STRUCTURAL: the cached plan index's spans no longer describe the display
    /// list. It must fall back to a full rebuild — and still be correct.
    #[test]
    fn table_row_add_is_structural_and_still_correct() {
        let opts = editor_opts();
        let mut resident = ResidentScene::default();
        resident
            .update(&moda_scene::value::to_string(&table_scene(3, 2)).unwrap())
            .unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();

        let fb_before = caches.incr_fallbacks;
        let patch_before = caches.incr_patch_hits;
        let grown: Vec<Value> = (0..4)
            .map(|r| json!((0..2).map(|c| table_cell(r, c)).collect::<Vec<_>>()))
            .collect();
        resident
            .apply_props(
                &json!({"nodes": {"t1": {
                    "rows": [{"size": 24.0}, {"size": 24.0}, {"size": 24.0}, {"size": 24.0}],
                    "height": 96.0,
                    "cells": grown
                }}})
                .to_string(),
            )
            .unwrap();
        assert_eq!(
            resident.host_reexpand_stats().1,
            1,
            "a row ADD must be classified structural"
        );
        caches.begin_frame();
        let (plan, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        assert_eq!(
            caches.incr_patch_hits, patch_before,
            "a structural re-expansion must not patch"
        );
        assert!(
            caches.incr_fallbacks > fb_before,
            "a structural re-expansion must fall back to a full rebuild"
        );
        let scene = resident.scene().unwrap();
        let (full, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
        assert_eq!(
            plan_items_fingerprint(&plan),
            plan_items_fingerprint(&full),
            "post-row-add plan must match a full rebuild"
        );
    }

    /// A cell going EMPTY -> NON-EMPTY keeps the transient id sequence (so the edit
    /// is NOT structural) but changes what that transient PRODUCES: a text item
    /// appears. `items_shape_match` must reject the splice and fall back — the
    /// failure mode this pins is serving stale pixels, not a crash.
    #[test]
    fn table_cell_text_appearing_falls_back_not_stale() {
        let opts = editor_opts();
        let mut resident = ResidentScene::default();
        resident
            .update(&moda_scene::value::to_string(&table_scene(2, 2)).unwrap())
            .unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();

        // t1-c0-1 starts EMPTY (checkerboard: (0+1) % 2 != 0). Give it text.
        let fb_before = caches.incr_fallbacks;
        let with_new_text = json!([
            [table_cell(0, 0), table_text_cell(0, 1, "NEW")],
            [table_cell(1, 0), table_cell(1, 1)],
        ]);
        resident
            .apply_props(&json!({"nodes": {"t1": {"cells": with_new_text}}}).to_string())
            .unwrap();
        assert_eq!(
            resident.host_reexpand_stats().1,
            0,
            "the id sequence is unchanged, so this is NOT structural — the fallback \
             below must come from the item shape-match, which is the point"
        );
        caches.begin_frame();
        let (plan, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        assert!(
            caches.incr_fallbacks > fb_before,
            "an empty->non-empty cell changes item production — the shape-match must \
             reject the splice and fall back"
        );
        let scene = resident.scene().unwrap();
        let (full, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
        assert_eq!(
            plan_items_fingerprint(&plan),
            plan_items_fingerprint(&full),
            "the fallback must serve a correct plan, not stale pixels"
        );
    }

    /// A cell whose HTML carries a variable colour run must resolve through
    /// `variables::resolve_node` on re-expansion, not be blind-merged raw. This is
    /// why the host branch of `apply_props` deliberately does NOT merge into
    /// `resolved_by_id` the way the non-host branch does.
    #[test]
    fn host_reexpansion_resolves_variables() {
        let opts = editor_opts();
        let mut doc = table_scene(2, 2);
        doc["nodes"][0]["cells"][0][0]["textContent"] =
            json!("<p style=\"color: var(--nf-variable-v-text, #ff0000)\">x</p>");
        let mut resident = ResidentScene::default();
        resident
            .update(&moda_scene::value::to_string(&doc).unwrap())
            .unwrap();
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        resident
            .apply_props(&json!({"nodes": {"t1": {"x": 77.0}}}).to_string())
            .unwrap();
        caches.begin_frame();
        let (plan, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
        let scene = resident.scene().unwrap();
        let (full, _w) = crate::frame::build_coverage_plan(scene, 0, &opts).unwrap();
        assert_eq!(
            plan_items_fingerprint(&plan),
            plan_items_fingerprint(&full),
            "a variable-bound host prop must survive re-expansion identically to a \
             cold load (a blind resolved-merge would bake the unresolved ref)"
        );
    }

    /// `?wasmHostReexpand=0` restores the earlier contract exactly: every
    /// transient-host prop patch is REJECTED so the host heals with a full re-feed.
    #[test]
    fn host_reexpand_kill_switch_restores_rejection() {
        let json = moda_scene::value::to_string(&table_scene(2, 2)).unwrap();
        let mut resident = ResidentScene::default();
        resident.update(&json).unwrap();
        resident.set_host_reexpand_enabled(false);
        assert!(
            resident
                .apply_props(&json!({"nodes": {"t1": {"x": 5.0}}}).to_string())
                .is_err(),
            "with the kill switch off a transient-host patch must be rejected"
        );
        assert_eq!(
            resident.host_reexpand_stats().0,
            0,
            "a rejected patch must not count as a re-expansion"
        );
        // Re-enabled (the default): the same patch now succeeds.
        let mut resident = ResidentScene::default();
        resident.update(&json).unwrap();
        resident
            .apply_props(&json!({"nodes": {"t1": {"x": 5.0}}}).to_string())
            .expect("with the switch on the same patch is accepted");
        assert_eq!(resident.host_reexpand_stats().0, 1);
    }

    /// Guard rails specific to the host lane: the PRE-EXISTING gates still run and
    /// still win. A structural key on a host is a hard reject; `zIndex` / a
    /// visibility flip still escalate to structural rather than sneaking through the
    /// re-expansion path as a cheap patch.
    #[test]
    fn apply_props_host_guard_rails() {
        let json = moda_scene::value::to_string(&table_scene(2, 2)).unwrap();
        let mut resident = ResidentScene::default();
        resident.update(&json).unwrap();
        assert!(
            resident
                .apply_props(&json!({"nodes": {"t1": {"type": "rectangle"}}}).to_string())
                .is_err(),
            "a structural key on a host is still a hard reject"
        );
        for (key, value) in [
            ("zIndex", json!(9.0)),
            ("visible", json!(false)),
            ("hidden", json!(true)),
        ] {
            let mut resident = ResidentScene::default();
            resident.update(&json).unwrap();
            let gen_before = resident.generation();
            resident
                .apply_props(&json!({"nodes": {"t1": {key: value}}}).to_string())
                .unwrap();
            assert_eq!(
                resident.dirty_since(gen_before),
                DirtySince::Structural,
                "'{key}' on a host must still escalate to structural"
            );
        }
    }

    // -----------------------------------------------------------------------
    // The `pages` branch of `apply_props`.
    //
    // A page background is DERIVED, not stored: `compute_resolved_nodes`
    // synthesizes a transient `__page_bg_<pageId>` rect from the page's own
    // props and puts it first in the page's z-order. Previously the pages
    // branch wrote the page props and stopped, so the engine held new page props
    // and the OLD background quad — a stale background on the snapshot lane.
    //
    // Every test below asserts on BOTH surfaces the rest of the engine reads:
    // `scene.effective_node("__page_bg_<pageId>")` (what the plan builder paints)
    // and the entry in `hashes` (what the plan/tessellation caches key on). A fix
    // that updates one without the other serves stale pixels from cache.
    // -----------------------------------------------------------------------

    /// A one-page, one-node doc whose page carries `page_props` on top of the
    /// required id/size/nodeIds.
    fn page_bg_doc(page_props: Value) -> Value {
        let mut page = json!({"id": "p1", "width": 800.0, "height": 400.0, "nodeIds": ["a"]});
        for (k, v) in page_props.as_object().expect("page props object") {
            page.as_object_mut().unwrap().insert(k.clone(), v.clone());
        }
        json!({
            "nodes": [{"id": "a", "type": "rectangle", "x": 10.0, "y": 10.0,
                       "width": 50.0, "height": 50.0, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]}],
            "pages": [page],
            "config": {},
        })
    }

    fn resident_for(doc: &Value) -> ResidentScene {
        let mut r = ResidentScene::default();
        r.update(&moda_scene::value::to_string(doc).unwrap()).unwrap();
        r
    }

    fn bg<'a>(r: &'a ResidentScene, page_id: &str) -> &'a RawNode {
        r.scene()
            .unwrap()
            .effective_node(&moda_scene::types::compute_page_bg_id(page_id))
            .expect("page background transient exists")
    }

    fn bg_hash(r: &ResidentScene, page_id: &str) -> NodeHashes {
        *r.hashes()
            .unwrap()
            .get(&moda_scene::types::compute_page_bg_id(page_id))
            .expect("page background is hashed")
    }

    /// THE oracle for all of the above: after a pages patch, the resident scene's
    /// derived background must be byte-identical to what a COLD feed of the same
    /// post-patch document produces — node AND hash. Everything else in this
    /// section is a targeted probe of one derivation rule; this is the general
    /// statement, and it is what `mutation_replay`'s byte gate enforces at the
    /// pixel level.
    fn assert_bg_matches_cold_feed(r: &ResidentScene, cold_doc: &Value, page_id: &str) {
        let cold = resident_for(cold_doc);
        assert_eq!(
            bg(r, page_id),
            bg(&cold, page_id),
            "patched background diverged from a cold feed of the same doc"
        );
        assert_eq!(
            bg_hash(r, page_id).full,
            bg_hash(&cold, page_id).full,
            "patched background HASH diverged from a cold feed (caches would serve stale)"
        );
    }

    /// A page `fill` patch must repaint the derived background — the base case,
    /// and the one a keyframed page-background animation hits every frame.
    #[test]
    fn apply_props_page_fill_rederives_background() {
        let doc = page_bg_doc(json!({"fills": solid("#112233")}));
        let mut r = resident_for(&doc);
        assert_eq!(bg(&r, "p1").fill_color(), Some("#112233"));
        let before = bg_hash(&r, "p1");

        r.apply_props(r##"{"pages":{"p1":{"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#00ff00"}}]}}}"##)
            .unwrap();

        assert_eq!(
            bg(&r, "p1").fill_color(),
            Some("#00ff00"),
            "the derived bg still carries the OLD fill — this is the ENG-4355 bug"
        );
        assert_ne!(
            before.full,
            bg_hash(&r, "p1").full,
            "the bg hash must move too, or the plan cache serves the old quad"
        );
        assert_bg_matches_cold_feed(
            &r,
            &page_bg_doc(json!({"fills": solid("#00ff00")})),
            "p1",
        );
    }

    /// `create_rectangle_from_page` derives the quad's `width`/`height` from the
    /// PAGE size, so a page resize moves the background even though the patch
    /// never names a width on the bg.
    #[test]
    fn apply_props_page_resize_rederives_background_geometry() {
        let doc = page_bg_doc(json!({"fills": solid("#112233")}));
        let mut r = resident_for(&doc);
        let before = bg_hash(&r, "p1");

        r.apply_props(r##"{"pages":{"p1":{"width":1000.0,"height":250.0}}}"##)
            .unwrap();

        assert_eq!(bg(&r, "p1").f64("width"), Some(1000.0));
        assert_eq!(bg(&r, "p1").f64("height"), Some(250.0));
        assert_ne!(
            before.geom,
            bg_hash(&r, "p1").geom,
            "a size change must move the GEOM hash — it re-keys tessellation"
        );
    }

    /// A gradient round trip: the patch REPLACES the page's `fills` stack, and
    /// the bg is re-derived from it, so a page that returns to a flat colour
    /// carries no trace of its old gradient (with the flat keys an insert-only
    /// merge could not remove `fillLinearGradient*`; a stack swap has nothing
    /// stale left to remove, and the re-derivation must agree with a cold feed).
    #[test]
    fn apply_props_page_gradient_round_trip_removes_stale_keys() {
        let doc = page_bg_doc(json!({"fills": solid("#112233")}));
        let mut r = resident_for(&doc);
        let top_type = |r: &ResidentScene| {
            moda_scene::paint::top_fill(bg(r, "p1")).map(|f| match f.paint {
                moda_scene::paint::Paint::Linear(_) => "linear",
                moda_scene::paint::Paint::Solid(_) => "solid",
                _ => "other",
            })
        };
        assert_eq!(top_type(&r), Some("solid"), "baseline is a flat colour");

        r.apply_props(
            r##"{"pages":{"p1":{"fills":[{"id":"fill-0","paint":{"type":"linear",
                               "start":{"x":0.0,"y":0.0},"end":{"x":1.0,"y":1.0},
                               "stops":[{"offset":0.0,"color":"#ff0000"},{"offset":1.0,"color":"#0000ff"}]}}]}}}"##,
        )
        .unwrap();
        assert_eq!(
            top_type(&r),
            Some("linear"),
            "the gradient must reach the derived bg"
        );

        // Back to a flat colour.
        r.apply_props(r##"{"pages":{"p1":{"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#112233"}}]}}}"##)
            .unwrap();
        assert_eq!(top_type(&r), Some("solid"));
        assert_eq!(bg(&r, "p1").fill_color(), Some("#112233"));
        assert!(
            !bg(&r, "p1")
                .json_field("fills")
                .unwrap()
                .to_string()
                .contains("stops"),
            "no stale gradient on the derived bg"
        );
        assert_bg_matches_cold_feed(
            &r,
            &page_bg_doc(json!({"fills": solid("#112233")})),
            "p1",
        );
    }

    /// An empty `fills` stack is no background ([Q5]: there is no default
    /// fill; this used to pin the nullish `fill` → `#ffffff` default). The
    /// derived bg carries no paint, and agrees with a cold feed.
    #[test]
    fn apply_props_page_empty_fills_derive_no_background() {
        let doc = page_bg_doc(json!({"fills": solid("#112233")}));
        let mut r = resident_for(&doc);

        r.apply_props(r##"{"pages":{"p1":{"fills":[]}}}"##).unwrap();

        assert!(
            moda_scene::paint::top_fill(bg(&r, "p1")).is_none(),
            "an empty page stack derives a bg that paints nothing"
        );
        assert_bg_matches_cold_feed(&r, &page_bg_doc(json!({"fills": []})), "p1");
    }

    /// An image background replaces the colour: the bg's top fill is the image,
    /// not the page's stale colour (this used to pin the non-`color`
    /// `fillPriority` gate dropping the `fill` key).
    #[test]
    fn apply_props_page_image_fill_replaces_the_colour() {
        let doc = page_bg_doc(json!({"fills": solid("#112233")}));
        let mut r = resident_for(&doc);
        assert_eq!(bg(&r, "p1").fill_color(), Some("#112233"));

        r.apply_props(r##"{"pages":{"p1":{"fills":[{"id":"fill-0","paint":{"type":"image","src":"data:x"}}]}}}"##)
            .unwrap();

        assert_eq!(
            bg(&r, "p1").fill_color(),
            None,
            "the old colour must not survive"
        );
        assert!(matches!(
            moda_scene::paint::top_fill(bg(&r, "p1")).map(|f| f.paint),
            Some(moda_scene::paint::Paint::Image(_))
        ));
    }

    /// Only the pages the patch NAMES are re-derived. A multi-page doc's untouched
    /// pages must be bit-identical afterwards — both the derived node and its hash
    /// — so the cost of a page edit tracks the edit, not the document.
    #[test]
    fn apply_props_page_patch_rederives_only_the_patched_pages() {
        let doc = json!({
            "nodes": [
                {"id":"a","type":"rectangle","x":0.0,"y":0.0,"width":50.0,"height":50.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]},
                {"id":"b","type":"rectangle","x":0.0,"y":0.0,"width":50.0,"height":50.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0f0"}}]}
            ],
            "pages": [
                {"id":"p1","width":800.0,"height":400.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#111111"}}],"nodeIds":["a"]},
                {"id":"p2","width":600.0,"height":300.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#222222"}}],"nodeIds":["b"]}
            ],
            "config": {},
        });
        let mut r = resident_for(&doc);
        let p2_before = bg(&r, "p2").clone();
        let p2_hash_before = bg_hash(&r, "p2");

        r.apply_props(r##"{"pages":{"p1":{"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#00ff00"}}]}}}"##)
            .unwrap();

        assert_eq!(bg(&r, "p1").fill_color(), Some("#00ff00"));
        assert_eq!(&p2_before, bg(&r, "p2"), "an unpatched page must not move");
        assert_eq!(
            p2_hash_before.full,
            bg_hash(&r, "p2").full,
            "an unpatched page's bg hash must not move either"
        );
    }

    /// The derived bg is a real display-list item of its own (unlike a table host,
    /// which `visit_z` expands away), so the dirty log has to name it — otherwise
    /// the node lane and the page lane disagree about what changed. Belt-and-braces
    /// alongside the branch's unconditional `structural`, which is what actually
    /// forces the rebuild today; this keeps the two consistent if that ever relaxes.
    #[test]
    fn apply_props_page_patch_logs_the_derived_background() {
        let doc = page_bg_doc(json!({"fills": solid("#112233")}));
        let mut r = resident_for(&doc);
        let gen_before = r.generation();

        r.apply_props(r##"{"pages":{"p1":{"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#00ff00"}}]}}}"##)
            .unwrap();

        // The branch is unconditionally structural, so `dirty_since` reports
        // Structural rather than a node list — assert that FIRST so the log
        // assertion below is not mistaken for the invalidation guarantee.
        assert!(
            matches!(r.dirty_since(gen_before), DirtySince::Structural),
            "a page patch stays structural (the page clip moves on every node)"
        );
        assert!(
            r.recent_edits.iter().any(|(_, id, _)| id == "__page_bg_p1"),
            "the derived bg must land in the dirty log"
        );
    }

    /// Partial-then-bail: a patch that mutates page 1 and then hits a structural
    /// key on page 2 must still leave the resident scene invalidated, so the next
    /// render does a correct full rebuild regardless of when the host re-feeds.
    #[test]
    fn apply_props_page_partial_then_error_still_invalidates() {
        let doc = json!({
            "nodes": [{"id":"a","type":"rectangle","x":0.0,"y":0.0,"width":50.0,"height":50.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]}],
            "pages": [
                {"id":"p1","width":800.0,"height":400.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#111111"}}],"nodeIds":["a"]},
                {"id":"p2","width":600.0,"height":300.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#222222"}}],"nodeIds":[]}
            ],
            "config": {},
        });
        let json_str = moda_scene::value::to_string(&doc).unwrap();
        let mut r = resident_for(&doc);
        let gen_before = r.generation();

        // "p1" is processed before "p2" (id order), so its bg is already
        // re-derived when the structural-key bail on "p2" hits.
        assert!(
            r.apply_props(r##"{"pages":{"p1":{"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#00ff00"}}]},"p2":{"nodeIds":[]}}}"##)
                .is_err(),
            "a structural key on a later page still errors"
        );
        assert_eq!(
            bg(&r, "p1").fill_color(),
            Some("#00ff00"),
            "the partial re-derive IS resident"
        );
        assert!(
            matches!(r.dirty_since(gen_before), DirtySince::Structural),
            "a partial-then-bail page patch must invalidate every prior plan index"
        );
        // …and the host's reject-heal (re-feed the SAME JSON) must re-parse.
        r.update(&json_str).unwrap();
        assert_eq!(bg(&r, "p1").fill_color(), Some("#111111"));
    }

    /// The mobile-FFI page overlay algorithm, on the CPU lane.
    ///
    /// `moda-mobile-ffi` sets `[lib] test = false` (its test surface is
    /// `scripts/host_smoke.py` against the built cdylib), and
    /// `tests/anim_overlay_parity.rs` needs a GPU adapter and returns early
    /// without one — so on CI's Linux runner NEITHER covers
    /// `render_overlay_frame`'s page half. This test replicates that algorithm
    /// against the plan builder instead of the rasterizer, so it runs everywhere:
    ///
    ///   * per frame, patch this playhead's `page_changes` PLUS a revert of any
    ///     page key overlaid last frame that dropped out (to the pristine base
    ///     value — `apply_props` is insert-only, so a key with no base value is
    ///     what forces the clone+rebuild fallback in the real session);
    ///   * emit the `pages` key ONLY when non-empty, because `apply_props` marks a
    ///     patch structural for the mere presence of `pages` and an unconditional
    ///     key would make every v2 playback frame structural;
    ///   * assert the retained plan equals a cold full rebuild of the same doc.
    ///
    /// The timeline: a colour tween, a flip to a gradient stack (the derived rect
    /// must LOSE its solid while the patch only ever writes keys), a flip back,
    /// and empty frames at both ends.
    #[test]
    fn overlay_page_changes_match_a_cold_rebuild_every_frame() {
        use moda_scene::value::Map;
        use std::collections::BTreeSet;

        // The page carries its full fill surface so every key the timeline drops
        // has a base value to revert TO; without that the real session would
        // (correctly) take the fallback and this test would prove nothing.
        let base = json!({
            "nodes": [{"id":"a","type":"rectangle","x":10.0,"y":10.0,
                       "width":50.0,"height":50.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]}],
            "pages": [{"id":"p1","width":480.0,"height":320.0,"nodeIds":["a"],
                       "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#eeeeee"}}]}],
            "config": {},
        });
        let base_json = moda_scene::value::to_string(&base).unwrap();
        let base_scene = Scene::load(moda_scene::extract_canvas_data(&base)).unwrap();

        let frames: [Value; 6] = [
            json!({}),
            json!({"fills": solid("#c0d0e0")}),
            json!({"fills": solid("#a0b0c0")}),
            // The solid drops out here: the derived rect must lose it.
            json!({"fills": [{"id": "fill-0", "paint": {"type": "linear",
                   "start": {"x": 0.0, "y": 50.0}, "end": {"x": 100.0, "y": 50.0},
                   "stops": [{"offset": 0.0, "color": "#ff0000"}, {"offset": 1.0, "color": "#0000ff"}]}}]}),
            // ...and the gradient drops out here.
            json!({"fills": solid("#90ee90")}),
            json!({}),
        ];

        let opts = RenderOptions {
            text_masks_available: Some(true),
            ..Default::default()
        };
        let mut resident = ResidentScene::default();
        resident.update(&base_json).unwrap();
        let mut caches = PlanCaches::default();
        let mut applied_pages: BTreeSet<String> = BTreeSet::new();

        for (frame, changes) in frames.iter().enumerate() {
            let changes = changes.as_object().expect("frame props object");
            let mut props: Map = changes.clone();
            // Revert keys overlaid last frame but absent now, to the base value.
            for k in &applied_pages {
                if changes.contains_key(k) {
                    continue;
                }
                let base_v = base_scene
                    .doc()
                    .pages
                    .iter()
                    .find(|p| p.id() == "p1")
                    .and_then(|p| p.0.get(k))
                    .cloned()
                    .expect("base page has every reverted key (see the fixture)");
                props.insert(k.clone().into(), base_v);
            }
            let mut patch = Map::new();
            patch.insert("nodes".into(), Value::Object(Map::new()));
            if !props.is_empty() {
                let mut pages = Map::new();
                pages.insert("p1".into(), Value::Object(props.clone()));
                patch.insert("pages".into(), Value::Object(pages));
            }
            resident
                .apply_props(&Value::Object(patch).to_string())
                .unwrap_or_else(|e| panic!("frame {frame}: apply_props rejected: {e:#}"));
            applied_pages = changes.keys().map(|k| k.to_string()).collect();

            // Ground truth: the same page state, cold.
            let mut cold_doc = base.clone();
            let cold_page = cold_doc["pages"][0].as_object_mut().unwrap();
            for (k, v) in &props {
                cold_page.insert(k.clone(), v.clone());
            }
            let cold_scene = Scene::load(moda_scene::extract_canvas_data(&cold_doc)).unwrap();

            caches.begin_frame();
            let (incr, _w) = build_plan_retained(&resident, &mut caches, 0, &opts).unwrap();
            let (full, _w) = crate::frame::build_coverage_plan(&cold_scene, 0, &opts).unwrap();
            assert_eq!(
                plan_items_fingerprint(&incr),
                plan_items_fingerprint(&full),
                "frame {frame} ({props:?}): the overlaid plan diverged from a cold rebuild"
            );
            // Non-vacuity: the derived background must actually be tracking the
            // page, or every frame above could agree on a plan that paints nothing.
            assert_eq!(
                resident
                    .scene()
                    .unwrap()
                    .effective_node("__page_bg_p1")
                    .unwrap(),
                cold_scene.effective_node("__page_bg_p1").unwrap(),
                "frame {frame}: derived background diverged from the cold scene"
            );
        }
    }

    /// An unknown page id errors instead of silently no-op'ing, and does so
    /// through the same `Err` arm that invalidates.
    #[test]
    fn apply_props_page_unknown_id_errors() {
        let doc = page_bg_doc(json!({"fills": solid("#112233")}));
        let mut r = resident_for(&doc);
        assert!(r
            .apply_props(r##"{"pages":{"nope":{"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#00ff00"}}]}}}"##)
            .is_err());
    }

    // -----------------------------------------------------------------------
    // Brush plan P8: outlined (variable-width / brushed) strokes
    // -----------------------------------------------------------------------

    fn outline_doc(nib_angle: f64) -> Value {
        json!({
            "nodes": [
                {"id":"v","type":"path","x":10.0,"y":10.0,"width":200.0,"height":50.0,
                 "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#123456"}, "width": 12, "cap": "round", "profile": {"preset": "eye"}}],"points":[{"type":"move","x":0,"y":0},{"type":"curve","x":200,"y":0,
                            "cp1":{"x":60,"y":-80},"cp2":{"x":140,"y":80}}]},
                {"id":"c","type":"path","x":10.0,"y":120.0,"width":200.0,"height":50.0,
                 "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#654321"}, "width": 8, "brush": {"brushId": "nib"}}],"points":[{"type":"move","x":0,"y":0},{"type":"line","x":200,"y":40}]},
            ],
            "pages": [{"id":"p1","width":400.0,"height":300.0,"nodeIds":["v","c"]}],
            "brushes": [{"id":"nib","name":"Nib","type":"calligraphic",
                         "angle":{"value":nib_angle},"roundness":{"value":0.2},"size":{"value":1}}],
            "config": {},
        })
    }

    fn body_geometry<'a>(
        plan: &'a CoveragePlan,
        id: &str,
    ) -> &'a crate::seams::TessellatedGeometry {
        plan.items
            .iter()
            .find_map(|i| match i {
                crate::seams::DrawItem::Body {
                    node_id, geometry, ..
                } if node_id == id => Some(&**geometry),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no body for {id}"))
    }

    /// A straight line or path whose stroke gets round caps (an edit, after frames at 1×) refines them per
    /// zoom band in the live lane: each zoom's plan equals one built cold at that zoom, and the caps gain
    /// vertices as the zoom grows (the line cache and the per-node tess cache key round ends on the band).
    #[test]
    fn round_ends_on_straight_strokes_refine_with_the_zoom() {
        let stroke = |cap: &str, width: f64| {
            json!([{"id": "stroke-0", "cap": cap, "join": cap, "paint": {"type": "solid", "color": "#000000"}, "width": width}])
        };
        let doc = json!({"nodes": [
                {"id": "l", "type": "line", "x": 0, "y": 0, "width": 200, "height": 0, "rotation": 0, "scale": 1,
                 "points": [100, 200, 300, 200], "strokes": stroke("butt", 1.0)},
                {"id": "p", "type": "path", "x": 0, "y": 0, "width": 200, "height": 40, "rotation": 0, "scale": 1,
                 "points": [{"type": "move", "x": 100, "y": 240}, {"type": "line", "x": 200, "y": 210}, {"type": "line", "x": 300, "y": 240}],
                 "strokes": stroke("butt", 1.0)}],
            "pages": [], "canvasRootNodeIds": ["l", "p"], "config": {}, "layoutMode": "freeform", "variables": []});
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let mut caches = PlanCaches::default();
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 280.0,
            camera_y: 190.0,
            zoom,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 2.0,
            clear_rgb: [1.0; 3],
        };
        caches.begin_frame();
        build_camera_plan_retained(&r, &mut caches, &[], &cam(1.0), &RenderOptions::default()).unwrap();
        r.apply_props(
            &json!({"nodes": {"l": {"strokes": stroke("round", 2.0)}, "p": {"strokes": stroke("round", 2.0)}}}).to_string(),
        )
        .unwrap();
        let vertices = |id: &str, p: &CoveragePlan| body_geometry(p, id).positions.len();
        let mut last = (0, 0);
        for zoom in [1.0, 16.0, 16.0, 64.0] {
            caches.begin_frame();
            let (live, _) = build_camera_plan_retained(&r, &mut caches, &[], &cam(zoom), &RenderOptions::default())
                .unwrap()
                .unwrap();
            let (cold, _) =
                build_camera_plan_retained(&r, &mut PlanCaches::default(), &[], &cam(zoom), &RenderOptions::default())
                    .unwrap()
                    .unwrap();
            assert_eq!(plan_items_fingerprint(&live), plan_items_fingerprint(&cold), "zoom {zoom}");
            let now = (vertices("l", &live), vertices("p", &live));
            assert!(now.0 >= last.0 && now.1 >= last.1, "zoom {zoom}: {now:?} after {last:?}");
            last = now;
        }
        let (one, _) = build_camera_plan_retained(&r, &mut PlanCaches::default(), &[], &cam(1.0), &RenderOptions::default())
            .unwrap()
            .unwrap();
        assert!(last.0 > vertices("l", &one) && last.1 > vertices("p", &one), "{last:?}");
    }

    #[test]
    fn outlined_strokes_draw_as_stroke_vertices_and_key_on_band_and_brush() {
        let mut r = ResidentScene::default();
        r.update(&outline_doc(30.0).to_string()).unwrap();
        let mut caches = PlanCaches::default();
        let opts = RenderOptions {
            sample_count: 4,
            ..Default::default()
        };
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 2.0,
            clear_rgb: [0.0; 3],
        };
        caches.begin_frame();
        let (p1, w1) = build_camera_plan_retained(&r, &mut caches, &[0], &cam(1.0), &opts)
            .unwrap()
            .unwrap();
        assert!(w1.dropped.is_empty() && w1.skipped.is_empty(), "{w1:?}");
        let v1 = body_geometry(&p1, "v");
        assert!(v1.vertex_types.iter().filter(|t| **t == 1.0).count() > 100);
        // The calligraphic brush resolved from the table: a nib at 30° on a
        // line at ~11° is wider than a hairline and narrower than the weight.
        let c1 = body_geometry(&p1, "c");
        assert!(c1.vertex_count > 0);
        // Same band (zoom 1 → 1.2 at dpr 2: 2 → 2.4 px/unit) → camera hit.
        caches.begin_frame();
        let (p2, _) = build_camera_plan_retained(&r, &mut caches, &[0], &cam(1.2), &opts)
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&p1, &p2), "same zoom band reuses the plan");
        // A higher band refits the outline finer → rebuild with more vertices.
        caches.begin_frame();
        let (p3, _) = build_camera_plan_retained(&r, &mut caches, &[0], &cam(8.0), &opts)
            .unwrap()
            .unwrap();
        assert!(!Arc::ptr_eq(&p1, &p3), "a new zoom band rebuilds");
        assert!(body_geometry(&p3, "v").vertex_count > v1.vertex_count);
        // Back to band 1: the plan rebuilds (single camera slot) at the band-1 outline.
        caches.begin_frame();
        let (p4, _) = build_camera_plan_retained(&r, &mut caches, &[0], &cam(1.0), &opts)
            .unwrap()
            .unwrap();
        assert_eq!(body_geometry(&p4, "v").positions, v1.positions);

        // A brush-table edit re-keys the strokes that use the brush (hash, tess
        // cache, camera plan) and leaves the others alone.
        let h_before = r.hashes().unwrap().clone();
        r.update(&outline_doc(80.0).to_string()).unwrap();
        let h_after = r.hashes().unwrap();
        assert_ne!(h_before["c"].geom, h_after["c"].geom);
        assert_eq!(h_before["v"].geom, h_after["v"].geom);
        caches.begin_frame();
        let (p5, _) = build_camera_plan_retained(&r, &mut caches, &[0], &cam(1.0), &opts)
            .unwrap()
            .unwrap();
        assert_ne!(body_geometry(&p5, "c").positions, c1.positions);
        assert_eq!(body_geometry(&p5, "v").positions, v1.positions);

        // Export at pixel ratio 2 = the camera at zoom 1 × dpr 2 (same band):
        // canvas = export geometry.
        let export = RenderOptions {
            scale: 2.0,
            ..opts.clone()
        };
        let (page, _) = crate::frame::build_coverage_plan(r.scene().unwrap(), 0, &export).unwrap();
        assert_eq!(body_geometry(&page, "v").positions, v1.positions);
    }

    #[test]
    fn scenes_without_outlines_keep_zoom_invariant_camera_plans() {
        // Straight strokes with straight ends (and fills of any shape) do not depend on the zoom. Round caps
        // and joins do (their arcs flatten per band): below.
        let mut doc = outline_doc(30.0);
        for n in doc["nodes"].as_array_mut().unwrap() {
            let o = n.as_object_mut().unwrap();
            if let Some(stroke) = o["strokes"][0].as_object_mut() {
                stroke.remove("profile");
                stroke.remove("brush");
                stroke.insert("cap".into(), json!("butt"));
                stroke.insert("join".into(), json!("miter"));
            }
            o.insert(
                "points".into(),
                json!([{"type":"move","x":0,"y":0},{"type":"line","x":200,"y":40}]),
            );
        }
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let mut caches = PlanCaches::default();
        let opts = RenderOptions::default();
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 2.0,
            clear_rgb: [0.0; 3],
        };
        let (a, _) = build_camera_plan_retained(&r, &mut caches, &[0], &cam(1.0), &opts)
            .unwrap()
            .unwrap();
        let (b, _) = build_camera_plan_retained(&r, &mut caches, &[0], &cam(16.0), &opts)
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&a, &b));

        // A round cap on the same straight stroke refines with the zoom.
        for n in doc["nodes"].as_array_mut().unwrap() {
            if let Some(stroke) = n["strokes"][0].as_object_mut() {
                stroke.insert("cap".into(), json!("round"));
            }
        }
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let mut caches = PlanCaches::default();
        let (a, _) = build_camera_plan_retained(&r, &mut caches, &[0], &cam(1.0), &opts)
            .unwrap()
            .unwrap();
        let (b, _) = build_camera_plan_retained(&r, &mut caches, &[0], &cam(16.0), &opts)
            .unwrap()
            .unwrap();
        assert!(!Arc::ptr_eq(&a, &b));
        assert_ne!(plan_items_fingerprint(&a), plan_items_fingerprint(&b));
    }

    #[test]
    fn mid_zoom_frames_keep_the_zooms_plan_until_the_zoom_holds() {
        // With motion plans on (the live editor), frames while the zoom changes keep the band the zoom started at
        // (up to MOTION_BAND_LAG bands behind) and ask for another frame; once the zoom has held for
        // MOTION_ZOOM_HOLD_MS the frame is exact: the plan a caches without motion plans builds.
        use std::sync::atomic::{AtomicU64, Ordering};
        static NOW: AtomicU64 = AtomicU64::new(0);
        fn clock() -> f64 {
            NOW.load(Ordering::Relaxed) as f64
        }
        let doc = json!({
            "nodes": [{"id":"e","type":"path","x":10.0,"y":10.0,"width":200.0,"height":120.0,"closed":true,
                       "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#336699"}}],
                       "points": [{"type":"move","x":0,"y":60},{"type":"curve","x":200,"y":60,"cp1":{"x":0,"y":-20},"cp2":{"x":200,"y":-20}},
                                  {"type":"curve","x":0,"y":60,"cp1":{"x":200,"y":140},"cp2":{"x":0,"y":140}}]}],
            "pages": [{"id":"p1","width":400.0,"height":300.0,"nodeIds":["e"]}],
            "config": {},
        });
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 1.0,
            clear_rgb: [0.0; 3],
        };
        let opts = RenderOptions::default();
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let exact = |z: f64| {
            let mut c = PlanCaches::default();
            build_camera_plan_retained(&r, &mut c, &[0], &cam(z), &opts).unwrap().unwrap().0
        };
        let mut caches = PlanCaches::default();
        caches.motion_plans = true;
        caches.set_outline_clock(clock);
        let mut at = |z: f64, t: u64, caches: &mut PlanCaches| {
            NOW.store(t, Ordering::Relaxed);
            build_camera_plan_retained(&r, caches, &[0], &cam(z), &opts).unwrap().unwrap().0
        };
        let n = |p: &CoveragePlan| body_geometry(p, "e").positions.len();
        let start = at(1.0, 1000, &mut caches);
        assert!(!caches.outline_refining(), "a still first frame is exact");
        // Zooming in 3x over a few frames (about 1.6 bands): the curve keeps band 0's flattening.
        let mid = at(1.7, 1016, &mut caches);
        assert!(caches.motion_plan && caches.outline_refining(), "a mid-zoom frame asks for another");
        let late = at(3.0, 1032, &mut caches);
        assert_eq!(n(&mid), n(&start));
        assert_eq!(n(&late), n(&start), "within the band lag: the zoom's first flattening");
        // Past the band lag (8x: three bands), the frame flattens for the camera again.
        let deep = at(8.0, 1048, &mut caches);
        assert_eq!(n(&deep), n(&exact(8.0)));
        // Back out to 3x (a band behind the flattening the 8x frame made): mid-zoom, the 8x flattening serves.
        let back = at(3.0, 1064, &mut caches);
        assert_eq!(n(&back), n(&deep));
        assert!(caches.outline_refining());
        // Within the hold the frame is still the zoom's; after it, exact, and it asks for nothing more.
        let _ = at(3.0, 1100, &mut caches);
        assert!(caches.outline_refining(), "within the hold, still mid-zoom");
        let held = at(3.0, 1300, &mut caches);
        assert!(!caches.motion_plan && !caches.outline_refining(), "held: exact");
        assert_eq!(n(&held), n(&exact(3.0)));
        // Without motion plans every zoom step flattens for its camera.
        let mut plain = PlanCaches::default();
        plain.set_outline_clock(clock);
        let _ = at(1.0, 2000, &mut plain);
        assert_eq!(n(&at(3.0, 2016, &mut plain)), n(&exact(3.0)));
        assert!(!plain.outline_refining());
    }

    /// A build at another camera (a motion-cache tile of the view an open showed before it fit) between live frames:
    /// it must not read as a zoom change to the live lane (which then held its previous band and refined nothing, and
    /// the next such build flipped it back, re-tessellating every curve every few seconds while idle). Set apart
    /// with `enter_side_camera`, the live frames settle once and then re-tessellate nothing.
    #[test]
    fn side_camera_builds_leave_the_live_lanes_zoom_history_alone() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NOW: AtomicU64 = AtomicU64::new(0);
        fn clock() -> f64 {
            NOW.load(Ordering::Relaxed) as f64
        }
        let doc = json!({
            "nodes": [{"id":"e","type":"path","x":10.0,"y":10.0,"width":200.0,"height":120.0,"closed":true,
                       "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#336699"}}],
                       "points": [{"type":"move","x":0,"y":60},{"type":"curve","x":200,"y":60,"cp1":{"x":0,"y":-20},"cp2":{"x":200,"y":-20}},
                                  {"type":"curve","x":0,"y":60,"cp1":{"x":200,"y":140},"cp2":{"x":0,"y":140}}]}],
            "pages": [{"id":"p1","width":400.0,"height":300.0,"nodeIds":["e"]}],
            "config": {},
        });
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 2.0,
            clear_rgb: [0.0; 3],
        };
        let opts = RenderOptions::default();
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let mut caches = PlanCaches::default();
        caches.set_outline_clock(clock);
        let live = |z: f64, t: u64, caches: &mut PlanCaches| {
            NOW.store(t, Ordering::Relaxed);
            caches.motion_plans = true;
            caches.begin_frame();
            build_camera_plan_retained(&r, caches, &[0], &cam(z), &opts).unwrap().unwrap().0
        };
        let side = |z: f64, t: u64, caches: &mut PlanCaches| {
            NOW.store(t, Ordering::Relaxed);
            caches.motion_plans = false;
            let h = caches.enter_side_camera(z);
            caches.begin_frame();
            let _ = build_camera_plan_retained(&r, caches, &[0], &cam(z), &opts).unwrap();
            caches.restore_camera_history(h);
        };
        // An open's first frame at 1x, then the fit (0.46x: another band), held until the zoom holds.
        let _ = live(1.0, 1000, &mut caches);
        let _ = live(0.46, 1016, &mut caches);
        assert!(caches.motion_plan, "mid-zoom");
        let settled = live(0.46, 1300, &mut caches);
        assert!(!caches.outline_refining(), "held: exact");
        let misses = caches.tess_misses;
        // Idle: tile fills of the pre-fit view (1x) between live frames at 0.46x.
        for k in 0..6u64 {
            side(1.0, 2000 + k * 1000, &mut caches);
            let p = live(0.46, 2500 + k * 1000, &mut caches);
            assert!(!caches.motion_plan && !caches.outline_refining(), "frame {k}: exact, nothing more to do");
            assert_eq!(
                body_geometry(&p, "e").positions,
                body_geometry(&settled, "e").positions,
                "frame {k}: the settled flattening"
            );
        }
        // The fills flatten their own band once; the live lane's geometry is never re-tessellated.
        assert!(caches.tess_misses <= misses + 1, "{} re-tessellations", caches.tess_misses - misses);
    }

    #[test]
    fn curved_fills_refit_per_band() {
        // A filled curved path with no stroke: fills flatten per band (F4.1),
        // so the camera plan follows the band, finer above 1× and coarser
        // below. A straight-only fill stays zoom-invariant.
        let doc = |ty: &str| {
            let points = if ty == "curved" {
                json!([{"type":"move","x":0,"y":60},{"type":"curve","x":200,"y":60,"cp1":{"x":0,"y":-20},"cp2":{"x":200,"y":-20}},
                       {"type":"curve","x":0,"y":60,"cp1":{"x":200,"y":140},"cp2":{"x":0,"y":140}}])
            } else {
                json!([{"type":"move","x":0,"y":0},{"type":"line","x":200,"y":0},{"type":"line","x":100,"y":120}])
            };
            json!({
                "nodes": [{"id":"e","type":"path","x":10.0,"y":10.0,"width":200.0,"height":120.0,"closed":true,
                           "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#336699"}}],"points": points}],
                "pages": [{"id":"p1","width":400.0,"height":300.0,"nodeIds":["e"]}],
                "config": {},
            })
        };
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 1.0,
            clear_rgb: [0.0; 3],
        };
        let opts = RenderOptions::default();
        for (ty, bands) in [("curved", true), ("straight", false)] {
            let mut r = ResidentScene::default();
            r.update(&doc(ty).to_string()).unwrap();
            let mut caches = PlanCaches::default();
            let mut plan = |z: f64| {
                build_camera_plan_retained(&r, &mut caches, &[0], &cam(z), &opts)
                    .unwrap()
                    .unwrap()
                    .0
            };
            let (a, b, c) = (plan(0.25), plan(1.0), plan(4.0));
            assert_eq!(
                !Arc::ptr_eq(&a, &b),
                bands,
                "{ty}: 1/4× rebuilds only with curves"
            );
            assert_eq!(
                !Arc::ptr_eq(&b, &c),
                bands,
                "{ty}: 4× rebuilds only with curves"
            );
            if bands {
                let n = |p: &CoveragePlan| body_geometry(p, "e").positions.len();
                assert!(
                    n(&c) > n(&b) && n(&b) > n(&a),
                    "the fill flattens finer with the band: {} < {} < {}",
                    n(&a),
                    n(&b),
                    n(&c)
                );
            }
        }
    }

    #[test]
    fn curved_strokes_refit_per_band() {
        // A uniform stroke on a curve flattens to 0.5 / 2^band units (F4.1):
        // the camera plan follows the band, finer above 1× and coarser below,
        // and one plan serves every zoom within a band.
        let mut doc = outline_doc(30.0);
        doc["nodes"][0]["strokes"][0]
            .as_object_mut()
            .unwrap()
            .remove("profile");
        doc["nodes"][1]["strokes"][0]
            .as_object_mut()
            .unwrap()
            .remove("brush");
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let mut caches = PlanCaches::default();
        let opts = RenderOptions::default();
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 1.0,
            clear_rgb: [0.0; 3],
        };
        let mut plan = |z: f64| {
            build_camera_plan_retained(&r, &mut caches, &[0], &cam(z), &opts)
                .unwrap()
                .unwrap()
                .0
        };
        let (a, b) = (plan(0.25), plan(1.0));
        assert!(!Arc::ptr_eq(&a, &b), "a coarser band rebuilds");
        let n = |p: &CoveragePlan| body_geometry(p, "v").positions.len();
        assert!(
            n(&a) < n(&b),
            "1/4× flattens the curve coarser: {} vs {}",
            n(&a),
            n(&b)
        );
        let c = plan(4.0);
        assert!(!Arc::ptr_eq(&b, &c), "a finer band rebuilds");
        assert!(
            n(&c) > n(&b),
            "4× flattens the curve finer: {} vs {}",
            n(&c),
            n(&b)
        );
        let d = plan(4.3);
        assert!(Arc::ptr_eq(&c, &d), "same band, same plan");
    }

    /// F4.1: curved tessellations (a curved path and a curved line route) are
    /// cached per zoom band. Panning and zooming within a band reuse them, even
    /// when the camera plan itself is rebuilt; another band tessellates again.
    #[test]
    fn curved_tessellations_are_cached_per_zoom_band() {
        let doc = json!({
            "nodes": [
                {"id":"c","type":"path","x":10.0,"y":10.0,"width":200.0,"height":120.0,"closed":true,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#336699"}}],"strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#123456"}, "width": 4}],"points":[{"type":"move","x":0,"y":60},
                           {"type":"curve","x":200,"y":60,"cp1":{"x":0,"y":-20},"cp2":{"x":200,"y":-20}},
                           {"type":"curve","x":0,"y":60,"cp1":{"x":200,"y":140},"cp2":{"x":0,"y":140}}]},
                {"id":"l","type":"line","points":[20.0,200.0,300.0,260.0],"strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": 3}],"routing":"curved","waypoints":[{"x":150.0,"y":170.0}]},
            ],
            "pages": [{"id":"p1","width":400.0,"height":300.0,"nodeIds":["c","l"]}],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        assert!(crate::tess::outline::scene_has_curves(r.scene().unwrap()));
        let opts = RenderOptions::default();
        let cam = |zoom: f64, x: f64| crate::frame::CameraParams {
            camera_x: x,
            camera_y: x / 2.0,
            zoom,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 2.0,
            clear_rgb: [0.0; 3],
        };
        let mut caches = PlanCaches::default();
        // Every build rebuilds the camera plan, so the per-node caches answer.
        let build = |caches: &mut PlanCaches, zoom: f64, x: f64| {
            caches.camera_clear();
            build_camera_plan_retained(&r, caches, &[0], &cam(zoom, x), &opts)
                .unwrap()
                .unwrap()
                .0
        };
        // zoom 4 × dpr 2 = 8 px per unit: band 3.
        let first = build(&mut caches, 4.0, 0.0);
        let (hits, misses) = (caches.tess_hits, caches.tess_misses);
        assert!(misses >= 2, "both curves tessellate once: {misses}");
        // Pans, and zooms that stay in band 3 (6.9 ..= 11.3 px per unit).
        for (zoom, x) in [(4.0, 35.0), (4.0, -60.0), (3.6, 12.0), (5.2, 80.0)] {
            let p = build(&mut caches, zoom, x);
            assert!(Arc::ptr_eq(&plan_body(&first, "c"), &plan_body(&p, "c")));
        }
        assert_eq!(
            caches.tess_misses, misses,
            "no re-tessellation within the band"
        );
        assert_eq!(caches.tess_hits, hits + 8, "both curves hit on every build");
        // Another band (32 px per unit: band 5) tessellates both again, finer.
        let finer = build(&mut caches, 16.0, 0.0);
        assert_eq!(caches.tess_misses, misses + 2);
        assert!(
            body_geometry(&finer, "c").positions.len() > body_geometry(&first, "c").positions.len()
        );
        // And a zoom-out below 1× (band -2) too, coarser.
        let coarse = build(&mut caches, 0.125, 0.0);
        assert_eq!(caches.tess_misses, misses + 4);
        assert!(
            body_geometry(&coarse, "c").positions.len()
                < body_geometry(&first, "c").positions.len()
        );
    }

    /// E8.5: a vector pattern's tile shapes flatten per zoom band like any curve. A scene whose only curves were in
    /// pattern tiles counted as straight (its camera plan never re-keyed on the band), and the tile shapes, synthetic
    /// and cached under their content hash, kept the band they were first built at: zoomed in, circles showed facets.
    #[test]
    fn pattern_tile_curves_flatten_per_zoom_band() {
        let doc = json!({
            "nodes": [
                {"id":"r","type":"rectangle","x":0.0,"y":0.0,"width":200.0,"height":150.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "pattern", "patternId": "dots"}}]},
            ],
            "patterns": [{ "id": "dots", "name": "Dots",
                "tile": { "type": "grid", "width": 24, "height": 24 },
                "artwork": { "nodes": [{ "id": "a", "type": "ellipse", "x": 4, "y": 4, "width": 16, "height": 16,
                              "fills": [{ "id": "fill-0", "paint": { "type": "solid", "color": "#facc15" } }] }] } }],
            "pages": [{"id":"p1","width":400.0,"height":300.0,"nodeIds":["r"]}],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let opts = RenderOptions::default();
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 2.0,
            clear_rgb: [0.0; 3],
        };
        let mut caches = PlanCaches::default();
        // The most vertices any one tile shape has in a build (one circle).
        let tile_vertices = |caches: &mut PlanCaches, zoom: f64| {
            caches.camera_clear();
            let plan = build_camera_plan_retained(&r, caches, &[0], &cam(zoom), &opts).unwrap().unwrap().0;
            plan.items
                .iter()
                .filter_map(|i| match i {
                    crate::seams::DrawItem::Body { node_id, geometry, .. } if node_id.contains("#tile:") => {
                        Some(geometry.positions.len())
                    }
                    _ => None,
                })
                .max()
                .expect("tile bodies")
        };
        let at_1x = tile_vertices(&mut caches, 1.0);
        let at_16x = tile_vertices(&mut caches, 16.0);
        assert!(at_16x > at_1x, "zoomed in, the tile's circle flattens finer: {at_16x} vs {at_1x}");
        // And back at 1× the coarse vertices serve again (same band, same content).
        assert_eq!(tile_vertices(&mut caches, 1.0), at_1x);
    }

    fn plan_body(plan: &CoveragePlan, id: &str) -> Arc<crate::seams::TessellatedGeometry> {
        plan.items
            .iter()
            .find_map(|i| match i {
                crate::seams::DrawItem::Body {
                    node_id, geometry, ..
                } if node_id == id => Some(geometry.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no body for {id}"))
    }

    fn uniform_curves_doc(n: usize) -> Value {
        let nodes: Vec<Value> = (0..n)
            .map(|i| {
                json!({"id": format!("u{i}"), "type":"path","x": 10.0 + (i % 20) as f64 * 18.0,"y": 10.0 + (i / 20) as f64 * 12.0,
                       "width":16.0,"height":10.0,"strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#123456"},"width":3.0}],
                       "points":[{"type":"move","x":0,"y":0},{"type":"curve","x":16,"y":0,
                                  "cp1":{"x":4,"y":-10},"cp2":{"x":12,"y":10}}]})
            })
            .collect();
        let ids: Vec<String> = (0..n).map(|i| format!("u{i}")).collect();
        json!({"nodes": nodes, "pages": [{"id":"p1","width":400.0,"height":300.0,"nodeIds": ids}], "config": {}})
    }

    #[test]
    fn uniform_curved_strokes_refine_progressively_after_a_band_change() {
        let mut r = ResidentScene::default();
        r.update(&uniform_curves_doc(3).to_string()).unwrap();
        let opts = RenderOptions::default();
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 1.0,
            clear_rgb: [0.0; 3],
        };
        let reference = |zoom: f64| {
            let mut full = PlanCaches::default();
            build_camera_plan_retained(&r, &mut full, &[0], &cam(zoom), &opts)
                .unwrap()
                .unwrap()
                .0
        };
        let (want1, want8) = (reference(1.0), reference(8.0));
        let mut caches = PlanCaches::default();
        let frame = |caches: &mut PlanCaches, zoom: f64, ms: f64| {
            caches.begin_frame();
            caches.set_outline_refine_budget_ms(Some(ms));
            let (p, _) = build_camera_plan_retained(&r, caches, &[0], &cam(zoom), &opts)
                .unwrap()
                .unwrap();
            caches.set_outline_refine_budget_ms(None);
            p
        };
        let p1 = frame(&mut caches, 1.0, 0.0);
        assert!(!caches.outline_refining(), "cold strokes are computed");
        assert_eq!(
            body_geometry(&p1, "u0").positions,
            body_geometry(&want1, "u0").positions
        );
        // Past the budget, a band change draws the previous band's flattening…
        let z1 = frame(&mut caches, 8.0, 0.0);
        assert!(caches.outline_refining());
        assert_eq!(
            body_geometry(&z1, "u1").positions,
            body_geometry(&want1, "u1").positions
        );
        // …and refines once the budget allows.
        let z2 = frame(&mut caches, 8.0, 1e6);
        assert!(!caches.outline_refining());
        assert_eq!(
            body_geometry(&z2, "u1").positions,
            body_geometry(&want8, "u1").positions
        );
        assert_ne!(
            body_geometry(&want8, "u1").positions,
            body_geometry(&want1, "u1").positions
        );
    }

    /// #87 review repro: a scene with curved strokes and no outlines, most of
    /// them off-screen after zooming in. Off-screen strokes must count as far
    /// (they wait for the camera) so live refinement converges instead of
    /// asking the host for frames forever.
    #[test]
    fn curved_strokes_off_screen_do_not_keep_refining() {
        let mut r = ResidentScene::default();
        r.update(&uniform_curves_doc(2000).to_string()).unwrap();
        let opts = RenderOptions::default();
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 1200.0,
            viewport_h: 800.0,
            dpr: 1.0,
            clear_rgb: [0.0; 3],
        };
        let mut caches = PlanCaches::default();
        let _ = build_camera_plan_retained(&r, &mut caches, &[0], &cam(1.0), &opts);
        let mut frames = 0;
        loop {
            caches.begin_frame();
            caches.set_outline_refine_budget_ms(Some(4.0));
            let _ = build_camera_plan_retained(&r, &mut caches, &[0], &cam(16.0), &opts);
            caches.set_outline_refine_budget_ms(None);
            frames += 1;
            if !caches.outline_refining() {
                break;
            }
            assert!(
                frames < 500,
                "still refining after {frames} frames: {} visible pending",
                caches.outline_visible_pending()
            );
        }
    }

    /// `cargo test -p moda-engine --release -- --ignored band_change_cost --nocapture`
    #[test]
    #[ignore]
    fn band_change_cost() {
        let mut r = ResidentScene::default();
        r.update(&uniform_curves_doc(2000).to_string()).unwrap();
        let opts = RenderOptions::default();
        let cam = |zoom: f64, vw: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 400.0 * vw,
            viewport_h: 300.0 * vw,
            dpr: 1.0,
            clear_rgb: [0.0; 3],
        };
        // All on screen (vw = zoom), and a normal viewport (vw = 1, most strokes
        // off-screen: they wait until the camera comes near them).
        for (budget, on_screen) in [
            (None, true),
            (Some(4.0), true),
            (None, false),
            (Some(4.0), false),
        ] {
            let mut caches = PlanCaches::default();
            let _ = build_camera_plan_retained(
                &r,
                &mut caches,
                &[0],
                &cam(1.0, if on_screen { 16.0 } else { 1.0 }),
                &opts,
            );
            let mut frames = 0;
            let t = std::time::Instant::now();
            let mut first = 0.0;
            loop {
                caches.begin_frame();
                caches.set_outline_refine_budget_ms(budget);
                let _ = build_camera_plan_retained(
                    &r,
                    &mut caches,
                    &[0],
                    &cam(16.0, if on_screen { 16.0 } else { 1.0 }),
                    &opts,
                );
                caches.set_outline_refine_budget_ms(None);
                frames += 1;
                if frames == 1 {
                    first = t.elapsed().as_secs_f64() * 1e3;
                }
                if !caches.outline_refining() || frames >= 200 {
                    break;
                }
            }
            eprintln!(
                "  pending after {frames} frames: {} visible",
                caches.outline_visible_pending()
            );
            eprintln!(
                "2000 curved strokes, 1× → 16×, budget {budget:?}, all on screen {on_screen}: first frame {first:.1} ms, {frames} frames, {:.1} ms total",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    #[test]
    fn live_frames_refine_outlines_progressively() {
        let mut r = ResidentScene::default();
        r.update(&outline_doc(30.0).to_string()).unwrap();
        let opts = RenderOptions {
            sample_count: 4,
            ..Default::default()
        };
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 2.0,
            clear_rgb: [0.0; 3],
        };
        let reference = |zoom: f64| {
            let mut full = PlanCaches::default();
            build_camera_plan_retained(&r, &mut full, &[0], &cam(zoom), &opts)
                .unwrap()
                .unwrap()
                .0
        };
        let (want1, want8) = (reference(1.0), reference(8.0));

        // Live frames with a zero refine budget. A cold open computes every
        // outline anyway: nothing is ever drawn as a placeholder.
        let mut caches = PlanCaches::default();
        let frame = |r: &ResidentScene, caches: &mut PlanCaches, zoom: f64, ms: f64| {
            caches.begin_frame();
            caches.set_outline_refine_budget_ms(Some(ms));
            let (p, _) = build_camera_plan_retained(r, caches, &[0], &cam(zoom), &opts)
                .unwrap()
                .unwrap();
            caches.set_outline_refine_budget_ms(None);
            p
        };
        let p1 = frame(&r, &mut caches, 1.0, 0.0);
        assert!(
            !caches.outline_refining(),
            "cold outlines are computed, not deferred"
        );
        assert_eq!(
            body_geometry(&p1, "v").positions,
            body_geometry(&want1, "v").positions
        );
        assert_eq!(
            body_geometry(&p1, "c").positions,
            body_geometry(&want1, "c").positions
        );
        let p2 = frame(&r, &mut caches, 1.1, 0.0);
        assert!(Arc::ptr_eq(&p1, &p2), "converged plan is cached");

        // A band change (zoom 1 → 2; both strokes still on screen) past the
        // budget: one outline still refines (guaranteed progress), the other is
        // drawn from the previous band…
        let want2 = reference(2.0);
        let z1 = frame(&r, &mut caches, 2.0, 0.0);
        assert!(caches.outline_refining());
        assert_eq!(caches.outline_visible_pending(), 1);
        assert_eq!(
            body_geometry(&z1, "v").positions,
            body_geometry(&want2, "v").positions
        );
        assert_eq!(
            body_geometry(&z1, "c").positions,
            body_geometry(&want1, "c").positions
        );
        // …and refines once the budget allows.
        let z2 = frame(&r, &mut caches, 2.0, 1e6);
        assert!(!caches.outline_refining());
        assert_eq!(
            body_geometry(&z2, "c").positions,
            body_geometry(&want2, "c").positions
        );
        // Far off-screen strokes wait for the camera: at zoom 8 the 50×37-unit
        // view shows only "v".
        let z3 = frame(&r, &mut caches, 8.0, 1e6);
        assert!(
            !caches.outline_refining(),
            "nothing on or near screen left to refine"
        );
        assert_eq!(
            body_geometry(&z3, "v").positions,
            body_geometry(&want8, "v").positions
        );
        assert_eq!(
            body_geometry(&z3, "c").positions,
            body_geometry(&want2, "c").positions
        );
        let at_c = crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 110.0,
            ..cam(8.0)
        };
        caches.begin_frame();
        caches.set_outline_refine_budget_ms(Some(1e6));
        let (z4, _) = build_camera_plan_retained(&r, &mut caches, &[0], &at_c, &opts)
            .unwrap()
            .unwrap();
        caches.set_outline_refine_budget_ms(None);
        let mut fresh = PlanCaches::default();
        let (want_c, _) = build_camera_plan_retained(&r, &mut fresh, &[0], &at_c, &opts)
            .unwrap()
            .unwrap();
        assert_eq!(
            body_geometry(&z4, "c").positions,
            body_geometry(&want_c, "c").positions
        );

        // Geometry that changes every frame (an animated trim) is computed every
        // frame, whatever the budget.
        let mut doc = outline_doc(30.0);
        for k in 0..3 {
            doc["nodes"][0]["trimPathEnd"] = json!(0.5 + 0.1 * k as f64);
            r.update(&doc.to_string()).unwrap();
            let p = frame(&r, &mut caches, 8.0, 0.0);
            assert!(!caches.outline_refining(), "frame {k}");
            let mut fresh = PlanCaches::default();
            let (want, _) = build_camera_plan_retained(&r, &mut fresh, &[0], &cam(8.0), &opts)
                .unwrap()
                .unwrap();
            assert_eq!(
                body_geometry(&p, "v").positions,
                body_geometry(&want, "v").positions
            );
        }
    }

    #[test]
    fn cold_live_frames_outline_only_what_is_near_the_viewport() {
        let mut r = ResidentScene::default();
        r.update(&outline_doc(30.0).to_string()).unwrap();
        let opts = RenderOptions::default();
        let cam = |y: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: y,
            zoom: 8.0,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 2.0,
            clear_rgb: [0.0; 3],
        };
        let exact = |y: f64| {
            let mut full = PlanCaches::default();
            build_camera_plan_retained(&r, &mut full, &[0], &cam(y), &opts)
                .unwrap()
                .unwrap()
                .0
        };
        let mut caches = PlanCaches::default();
        caches.set_outline_refine_budget_ms(Some(1e6));
        let (p, _) = build_camera_plan_retained(&r, &mut caches, &[0], &cam(0.0), &opts)
            .unwrap()
            .unwrap();
        caches.set_outline_refine_budget_ms(None);
        let want = exact(0.0);
        assert_eq!(
            body_geometry(&p, "v").positions,
            body_geometry(&want, "v").positions
        );
        assert_ne!(
            body_geometry(&p, "c").positions,
            body_geometry(&want, "c").positions,
            "the off-screen stroke is a placeholder"
        );
        assert!(!caches.outline_refining() && caches.outline_visible_pending() == 0);
        caches.begin_frame();
        caches.set_outline_refine_budget_ms(Some(1e6));
        let (p2, _) = build_camera_plan_retained(&r, &mut caches, &[0], &cam(110.0), &opts)
            .unwrap()
            .unwrap();
        caches.set_outline_refine_budget_ms(None);
        assert_eq!(
            body_geometry(&p2, "c").positions,
            body_geometry(&exact(110.0), "c").positions,
            "outlined once the camera reaches it"
        );
    }

    #[test]
    fn a_page_capture_at_another_scale_does_not_evict_the_cameras_outlines() {
        let mut r = ResidentScene::default();
        r.update(&outline_doc(30.0).to_string()).unwrap();
        let opts = RenderOptions::default();
        let cam = crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom: 1.0,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 2.0,
            clear_rgb: [0.0; 3],
        };
        let mut caches = PlanCaches::default();
        let live = |caches: &mut PlanCaches| {
            caches.begin_frame();
            caches.set_outline_refine_budget_ms(Some(0.0));
            let p = build_camera_plan_retained(&r, caches, &[0], &cam, &opts)
                .unwrap()
                .unwrap()
                .0;
            caches.set_outline_refine_budget_ms(None);
            p
        };
        let p1 = live(&mut caches);
        // A thumbnail-scale capture through the same caches.
        let thumb = RenderOptions {
            scale: 0.25,
            ..RenderOptions::default()
        };
        {
            let scene = r.scene().unwrap();
            let hashes = r.hashes().unwrap();
            crate::frame::build_coverage_plan_cached(scene, 0, &thumb, Some((&mut caches, hashes)))
                .unwrap();
        }
        caches.camera_clear();
        let misses = caches.tess_misses;
        let p2 = live(&mut caches);
        assert!(
            !caches.outline_refining(),
            "no stale outlines after the capture"
        );
        assert_eq!(
            caches.tess_misses, misses,
            "the camera's band is still cached"
        );
        assert_eq!(
            body_geometry(&p1, "v").positions,
            body_geometry(&p2, "v").positions
        );
    }

    #[test]
    fn zooming_out_reuses_finer_outlines_without_refinement() {
        let mut r = ResidentScene::default();
        r.update(&outline_doc(30.0).to_string()).unwrap();
        let opts = RenderOptions::default();
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 800.0,
            viewport_h: 600.0,
            dpr: 2.0,
            clear_rgb: [0.0; 3],
        };
        let mut caches = PlanCaches::default();
        let mut live = |zoom: f64| {
            caches.begin_frame();
            caches.set_outline_refine_budget_ms(Some(0.0));
            let p = build_camera_plan_retained(&r, &mut caches, &[0], &cam(zoom), &opts)
                .unwrap()
                .unwrap()
                .0;
            caches.set_outline_refine_budget_ms(None);
            (p, caches.outline_refining(), caches.tess_misses)
        };
        let (p2, _, _) = live(2.0);
        let (p1, refining, misses_before) = live(1.0);
        assert!(!refining, "zooming out never waits for refinement");
        assert_eq!(
            body_geometry(&p1, "v").positions,
            body_geometry(&p2, "v").positions
        );
        let (_, refining, misses_after) = live(0.5);
        assert!(!refining);
        assert!(misses_after >= misses_before);
    }

    fn cam_at(zoom: f64, y: f64) -> crate::frame::CameraParams {
        crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: y,
            zoom,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 2.0,
            clear_rgb: [0.0; 3],
        }
    }

    fn live_frame(
        r: &ResidentScene,
        caches: &mut PlanCaches,
        cam: &crate::frame::CameraParams,
        pages: &[usize],
    ) -> Arc<CoveragePlan> {
        caches.begin_frame();
        caches.set_outline_refine_budget_ms(Some(0.0));
        let p = build_camera_plan_retained(r, caches, pages, cam, &RenderOptions::default())
            .unwrap()
            .unwrap()
            .0;
        caches.set_outline_refine_budget_ms(None);
        p
    }

    fn exact_frame(
        r: &ResidentScene,
        cam: &crate::frame::CameraParams,
        pages: &[usize],
    ) -> Arc<CoveragePlan> {
        let mut full = PlanCaches::default();
        build_camera_plan_retained(r, &mut full, pages, cam, &RenderOptions::default())
            .unwrap()
            .unwrap()
            .0
    }

    fn serve_requests(caches: &mut PlanCaches) -> usize {
        let reqs: Vec<Value> = moda_scene::value::from_str(&caches.take_outline_requests()).unwrap();
        let mut n = 0;
        for q in &reqs {
            let node = RawNode::new(q["node"].as_object().unwrap().clone());
            let band = q["band"].as_i64().unwrap() as i32;
            let (g, note) = crate::tess::outline::tessellate_outlined_node(
                &node,
                q.get("brush").filter(|b| !b.is_null()),
                band,
            );
            if caches.provide_outline(
                q["id"].as_str().unwrap(),
                q["key"].as_str().unwrap(),
                q["base"].as_str().unwrap(),
                band,
                g,
                note,
            ) {
                n += 1;
            }
        }
        n
    }

    #[test]
    fn worker_outlines_round_trip_and_match_the_main_thread() {
        let mut r = ResidentScene::default();
        r.update(&outline_doc(30.0).to_string()).unwrap();
        let mut caches = PlanCaches::default();
        caches.set_outline_worker(true);
        // Cold strokes compute on the main thread, workers or not.
        live_frame(&r, &mut caches, &cam_at(1.0, 0.0), &[0]);
        assert!(!caches.outline_refining());
        assert_eq!(caches.take_outline_requests(), "[]");
        // A band change past the budget: one refines on the main thread, the
        // other goes to a worker and draws its previous band meanwhile.
        let z2 = cam_at(2.0, 0.0);
        live_frame(&r, &mut caches, &z2, &[0]);
        assert!(caches.outline_refining());
        assert_eq!(serve_requests(&mut caches), 1);
        let p = live_frame(&r, &mut caches, &z2, &[0]);
        assert!(!caches.outline_refining());
        let want = exact_frame(&r, &z2, &[0]);
        for id in ["v", "c"] {
            assert_eq!(
                body_geometry(&p, id).positions,
                body_geometry(&want, id).positions,
                "{id}"
            );
        }
    }

    #[test]
    fn edited_strokes_are_never_drawn_without_their_stroke_in_worker_mode() {
        let mut r = ResidentScene::default();
        r.update(&outline_doc(30.0).to_string()).unwrap();
        let mut caches = PlanCaches::default();
        caches.set_outline_worker(true);
        live_frame(&r, &mut caches, &cam_at(1.0, 0.0), &[0]);
        let stroke_verts = |p: &CoveragePlan, id: &str| {
            body_geometry(p, id)
                .vertex_types
                .iter()
                .filter(|t| **t == 1.0)
                .count()
        };
        // A brush edit re-keys "c"; a budget-0 frame draws it (exactly, as the
        // frame's one guaranteed outline) — never without its stroke.
        for angle in [40.0, 50.0, 60.0] {
            r.update(&outline_doc(angle).to_string()).unwrap();
            let p = live_frame(&r, &mut caches, &cam_at(1.0, 0.0), &[0]);
            assert!(stroke_verts(&p, "c") > 6, "angle {angle}");
            let want = exact_frame(&r, &cam_at(1.0, 0.0), &[0]);
            assert_eq!(
                body_geometry(&p, "c").positions,
                body_geometry(&want, "c").positions,
                "angle {angle}"
            );
        }
        assert_eq!(
            caches.take_outline_requests(),
            "[]",
            "edits are computed, not requested"
        );
        // Editing BOTH at once at budget 0: one is exact, the other draws its
        // pre-edit outline (queued), and the next frame converges.
        let mut doc = outline_doc(60.0);
        doc["nodes"][0]["strokes"][0]["width"] = json!(20.0);
        doc["nodes"][1]["strokes"][0]["width"] = json!(12.0);
        r.update(&doc.to_string()).unwrap();
        let p = live_frame(&r, &mut caches, &cam_at(1.0, 0.0), &[0]);
        assert!(stroke_verts(&p, "v") > 6 && stroke_verts(&p, "c") > 6);
        assert!(caches.outline_refining());
        assert_eq!(
            serve_requests(&mut caches),
            1,
            "the stroke drawing its pre-edit outline is requested from the workers"
        );
        let mut p = p;
        for _ in 0..3 {
            if !caches.outline_refining() {
                break;
            }
            p = live_frame(&r, &mut caches, &cam_at(1.0, 0.0), &[0]);
        }
        let want = exact_frame(&r, &cam_at(1.0, 0.0), &[0]);
        for id in ["v", "c"] {
            assert_eq!(
                body_geometry(&p, id).positions,
                body_geometry(&want, id).positions,
                "{id}"
            );
        }
    }

    #[test]
    fn a_continuous_multi_select_edit_starves_no_stroke() {
        // A gesture re-keys BOTH strokes every frame at budget 0: the frame's
        // one granted refine goes to the first in document order; the other
        // draws its pre-edit outline — but only for OUTLINE_MAX_PREV_FRAMES
        // frames in a row, then it is computed over budget.
        let mut r = ResidentScene::default();
        r.update(&outline_doc(30.0).to_string()).unwrap();
        let mut caches = PlanCaches::default();
        live_frame(&r, &mut caches, &cam_at(1.0, 0.0), &[0]);
        let mut behind = [0u32; 2];
        for f in 0..10 {
            let mut doc = outline_doc(30.0);
            doc["nodes"][0]["strokes"][0]["width"] = json!(10.0 + f as f64);
            doc["nodes"][1]["strokes"][0]["width"] = json!(6.0 + f as f64);
            r.update(&doc.to_string()).unwrap();
            let p = live_frame(&r, &mut caches, &cam_at(1.0, 0.0), &[0]);
            let want = exact_frame(&r, &cam_at(1.0, 0.0), &[0]);
            for (i, id) in ["v", "c"].into_iter().enumerate() {
                if body_geometry(&p, id).positions == body_geometry(&want, id).positions {
                    behind[i] = 0;
                } else {
                    behind[i] += 1;
                    assert!(
                        behind[i] <= OUTLINE_MAX_PREV_FRAMES,
                        "frame {f}: {id} drew its pre-edit outline {} frames in a row",
                        behind[i]
                    );
                }
            }
        }
    }

    #[test]
    fn late_worker_results_for_edited_strokes_are_dropped() {
        let mut r = ResidentScene::default();
        r.update(&outline_doc(30.0).to_string()).unwrap();
        let mut caches = PlanCaches::default();
        caches.set_outline_worker(true);
        live_frame(&r, &mut caches, &cam_at(1.0, 0.0), &[0]);
        live_frame(&r, &mut caches, &cam_at(2.0, 0.0), &[0]);
        let reqs = caches.take_outline_requests();
        assert_ne!(reqs, "[]");
        // Edit everything, converge on the main thread, then the old results land.
        r.update(&outline_doc(70.0).to_string()).unwrap();
        let mut doc = outline_doc(70.0);
        doc["nodes"][0]["strokes"][0]["width"] = json!(20.0);
        r.update(&doc.to_string()).unwrap();
        for _ in 0..4 {
            live_frame(&r, &mut caches, &cam_at(2.0, 0.0), &[0]);
        }
        // (Requests for the edited content itself are current: drop them here.)
        caches.take_outline_requests();
        caches.outline_requests.push(
            reqs.trim_start_matches('[')
                .trim_end_matches(']')
                .to_string(),
        );
        assert_eq!(serve_requests(&mut caches), 0, "stale results rejected");
        let p = live_frame(&r, &mut caches, &cam_at(2.0, 0.0), &[0]);
        let want = exact_frame(&r, &cam_at(2.0, 0.0), &[0]);
        assert_eq!(
            body_geometry(&p, "v").positions,
            body_geometry(&want, "v").positions
        );
    }

    #[test]
    fn far_pending_outlines_keep_multi_page_camera_plans_cached() {
        let mut doc = outline_doc(30.0);
        doc["pages"] = json!([
            {"id":"p1","width":400.0,"height":300.0,"nodeIds":["v","c"]},
            {"id":"p2","width":400.0,"height":300.0,"nodeIds":[]},
        ]);
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let mut caches = PlanCaches::default();
        // Zoom 8 at the origin: "c" is far off-screen (a stand-in, pending).
        let p1 = live_frame(&r, &mut caches, &cam_at(8.0, 0.0), &[0, 1]);
        assert!(!caches.outline_refining());
        let p2 = live_frame(&r, &mut caches, &cam_at(8.0, 0.0), &[0, 1]);
        assert!(
            Arc::ptr_eq(&p1, &p2),
            "cached although a far stand-in is pending"
        );
        // Panning to "c" bypasses the cache and outlines it.
        let at_c = cam_at(8.0, 110.0);
        let mut p3 = live_frame(&r, &mut caches, &at_c, &[0, 1]);
        for _ in 0..3 {
            if !caches.outline_refining() {
                break;
            }
            p3 = live_frame(&r, &mut caches, &at_c, &[0, 1]);
        }
        let want = exact_frame(&r, &at_c, &[0, 1]);
        assert_eq!(
            body_geometry(&p3, "c").positions,
            body_geometry(&want, "c").positions
        );
    }

    /// `MODA_PERF_SCENE=<scene.json> cargo test --release -p moda-engine --lib
    /// perf_scene_camera_costs -- --ignored --nocapture` (scene from
    /// scripts/e2e/perf-brush-strokes.mjs `buildScene`).
    #[test]
    #[ignore]
    fn perf_scene_camera_costs() {
        let Ok(path) = std::env::var("MODA_PERF_SCENE") else {
            return;
        };
        let json = std::fs::read_to_string(path).unwrap();
        let mut r = ResidentScene::default();
        r.update(&json).unwrap();
        let opts = RenderOptions::default();
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 1400.0,
            viewport_h: 900.0,
            dpr: 2.0,
            clear_rgb: [0.0; 3],
        };
        let mut caches = PlanCaches::default();
        let time = |caches: &mut PlanCaches, zoom: f64, budget: Option<f64>| {
            caches.begin_frame();
            caches.set_outline_refine_budget_ms(budget);
            let t = std::time::Instant::now();
            build_camera_plan_retained(&r, caches, &[0], &cam(zoom), &opts).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            caches.set_outline_refine_budget_ms(None);
            (ms, caches.outline_refining())
        };
        eprintln!(
            "cold, all outlines: {:.1} ms",
            time(&mut caches, 1.0, None).0
        );
        eprintln!("warm hit: {:.2} ms", time(&mut caches, 1.0, None).0);
        caches.camera_clear();
        eprintln!(
            "rebuild, all tess hits: {:.1} ms",
            time(&mut caches, 1.0, None).0
        );
        let mut n = 0;
        let mut total = 0.0;
        loop {
            let (ms, refining) = time(&mut caches, 4.0, Some(6.0));
            n += 1;
            total += ms;
            if n <= 3 {
                eprintln!("band change frame {n}: {ms:.1} ms");
            }
            if !refining || n > 500 {
                break;
            }
        }
        eprintln!(
            "band change: {n} frames, {:.1} ms mean; incr patch hits {} fallbacks {}",
            total / n as f64,
            caches.incr_patch_hits,
            caches.incr_fallbacks
        );
        let mut fresh = PlanCaches::default();
        let (ms, _) = time(&mut fresh, 1.0, Some(6.0));
        eprintln!("cold with a 6 ms budget (all computed): {ms:.1} ms");
    }

    fn placed_doc(color: &str) -> Value {
        json!({
            "nodes": [
                {"id":"a","type":"path","x":10.0,"y":10.0,"width":300.0,"height":40.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffcc00"}}],"strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#123456"}, "width": 1, "brush": {"brushId": "art"}}],"closed":true,
                 "points":[{"type":"move","x":0,"y":0},{"type":"line","x":300,"y":0},{"type":"line","x":300,"y":40}]},
                {"id":"s","type":"path","x":10.0,"y":120.0,"width":300.0,"height":10.0,
                 "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#654321"}, "width": 1, "brush": {"brushId": "dots", "seed": 1}}],"points":[{"type":"move","x":0,"y":0},{"type":"line","x":300,"y":0}]},
            ],
            "pages": [{"id":"p1","width":400.0,"height":300.0,"nodeIds":["a","s"]}],
            "brushes": [
                {"id":"art","name":"Art","type":"art","scaleWithWeight":true,"scale":{"value":1},"fit":"stretch","direction":"right",
                 "overlap":"none","colorize":{"method":"none"},
                 "artwork":{"nodes":[{"id":"t","type":"path","closed":true,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": color}}],
                   "points":[{"type":"move","x":0,"y":0},{"type":"line","x":100,"y":10},{"type":"line","x":0,"y":20}]}]}},
                {"id":"dots","name":"Dots","type":"scatter","scaleWithWeight":true,"size":{"value":1},"spacing":{"value":1},
                 "scatter":{"value":0},"rotation":{"value":0},"rotationRelativeTo":"path","colorize":{"method":"none"},
                 "artwork":{"nodes":[{"id":"d","type":"ellipse","x":0,"y":0,"width":10,"height":10,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}]}]}},
            ],
            "config": {},
        })
    }

    #[test]
    fn generated_draw_identities_keep_node_preparation() {
        for brush in [false, true] {
            for effect in [false, true] {
                for page_lane in [false, true] {
                    // The editor page cache does not seed an incremental slot for effect plans.
                    if page_lane && effect { continue; }
                    let mut doc = placed_doc("#ff0000");
                    if !brush {
                        doc["nodes"][0]["strokes"] = json!([]);
                        doc["nodes"][0]["fills"] = json!([
                            {"id":"red","paint":{"type":"solid","color":"#ff0000"}},
                            {"id":"blue","paint":{"type":"linear","start":{"x":0,"y":0},"end":{"x":1,"y":1},
                                "stops":[{"offset":0,"color":"#0000ff80"},{"offset":1,"color":"#00ff0080"}]}}
                        ]);
                    }
                    if effect {
                        doc["nodes"].as_array_mut().unwrap().push(json!({"id":"unit","type":"rectangle",
                            "x":340,"y":200,"width":30,"height":30,
                            "fills":[{"id":"f","paint":{"type":"solid","color":"#abcdef"}}],
                            "effects":[{"id":"e","type":"layer-blur","radius":5}]}));
                        doc["pages"][0]["nodeIds"].as_array_mut().unwrap().push(json!("unit"));
                    }
                    let mut resident = ResidentScene::default();
                    resident.update(&doc.to_string()).unwrap();
                    resident.state.as_mut().unwrap().scene.refresh_worlds();
                    let opts = editor_opts();
                    let mut caches = PlanCaches::default();
                    let mut index = FramePlanIndex::default();
                    let seed = if page_lane {
                        build_plan_retained(&resident, &mut caches, 0, &opts).unwrap().0
                    } else {
                        let scene = resident.scene().unwrap();
                        let (plan, warnings, aux) = crate::frame::build_coverage_plan_indexed(scene, 0, &opts,
                            Some((&mut caches, resident.hashes().unwrap()))).unwrap();
                        let (plan, warnings) = (Arc::new(plan), Arc::new(warnings));
                        index.seed(scene, &opts, &plan, &warnings, aux, 0, 0, &mut caches);
                        plan
                    };
                    let prefix = if brush { "a::brush::" } else { "a::paint::" };
                    assert!(seed.items.iter().any(|i| drawitem_node_id(i).is_some_and(|id| id.starts_with(prefix))));
                    drop(seed);
                    for (generation, edit) in [json!({"x":30}), json!({"y":25})].iter().enumerate() {
                        resident.apply_props(&json!({"nodes":{"a":edit}}).to_string()).unwrap();
                        resident.state.as_mut().unwrap().scene.refresh_worlds();
                        caches.begin_frame();
                        let before = caches.incr_patch_hits;
                        let plan = if page_lane {
                            build_plan_retained(&resident, &mut caches, 0, &opts).unwrap().0
                        } else {
                            index.patch(resident.scene().unwrap(), resident.hashes().unwrap(), &opts,
                                generation as u64 + 1, generation as u64 + 1, vec!["a".into()], &mut caches)
                                .expect("generated identities must permit node preparation").0
                        };
                        assert!(caches.incr_patch_hits > before, "brush={brush} effect={effect} page={page_lane}");
                        let mut fresh_caches = PlanCaches::default();
                        let (fresh, _, _) = crate::frame::build_coverage_plan_indexed(resident.scene().unwrap(), 0, &opts,
                            Some((&mut fresh_caches, resident.hashes().unwrap()))).unwrap();
                        assert_eq!(plan_items_fingerprint(&plan), plan_items_fingerprint(&fresh),
                            "brush={brush} effect={effect} page={page_lane} edit={edit}");
                        assert_eq!(plan.masks, fresh.masks);
                    }
                }
            }
        }
    }

    /// Progressive placement: a live frame places never-placed brush strokes (a painting opening) within its
    /// placement budget and leaves the rest out; the frames after fill them in back to front, in paint order, a
    /// stroke once drawn never leaving, until the plan is a from-scratch build's. A build without a budget (an
    /// export) places them all at once.
    #[test]
    fn unplaced_strokes_fill_in_back_to_front_across_live_frames() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NOW: AtomicU64 = AtomicU64::new(0);
        // Every read of the clock advances it 1 ms: a placement budget of 5 ms places a handful a frame.
        fn clock() -> f64 {
            NOW.fetch_add(1, Ordering::Relaxed) as f64
        }
        let ids: Vec<String> = (0..48).map(|k| format!("b{k:02}")).collect();
        let nodes: Vec<Value> = ids
            .iter()
            .enumerate()
            .map(|(k, id)| {
                json!({"id": id, "type": "path", "x": 10.0, "y": 10.0 + 13.0 * k as f64, "width": 300.0, "height": 10.0,
                       "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#223344"}, "width": 6,
                                    "brush": {"brushId": "round", "seed": k}}],
                       "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 300, "y": 10}]})
            })
            .collect();
        let doc = json!({
            "nodes": nodes,
            "pages": [{"id": "p1", "width": 400.0, "height": 700.0, "nodeIds": ids}],
            "brushes": [{"id": "round", "name": "Round", "type": "bristle", "shape": "flat-fan", "size": 3,
                         "length": 1.5, "density": 0.5, "thickness": 0.5, "opacity": 0.8, "stiffness": 0.4}],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let opts = RenderOptions::default();
        let cam = crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom: 1.0,
            viewport_w: 400.0,
            viewport_h: 700.0,
            dpr: 1.0,
            clear_rgb: [1.0; 3],
        };
        let drawn = |p: &CoveragePlan| -> Vec<String> {
            let mut out: Vec<String> = Vec::new();
            for i in &p.items {
                if let crate::seams::DrawItem::Body { node_id, .. } = i {
                    let owner = node_id.split("::").next().unwrap_or("").to_string();
                    if out.last() != Some(&owner) {
                        out.push(owner);
                    }
                }
            }
            out
        };
        // No budget (an export, moda-render): every stroke in the first build.
        let mut export = PlanCaches::default();
        let full = build_camera_plan_retained(&r, &mut export, &[0], &cam, &opts).unwrap().unwrap().0;
        assert_eq!(drawn(&full), ids, "an unbudgeted build places everything");
        assert_eq!(export.placement_deferred, 0);
        // Live frames at a 1 ms refine budget (5 ms of placements).
        let mut caches = PlanCaches::default();
        caches.set_outline_clock(clock);
        let mut frames = 0;
        let mut before: Vec<String> = Vec::new();
        let last = loop {
            caches.begin_frame();
            caches.set_outline_refine_budget_ms(Some(1.0));
            let plan = build_camera_plan_retained(&r, &mut caches, &[0], &cam, &opts).unwrap().unwrap().0;
            caches.set_outline_refine_budget_ms(None);
            let now = drawn(&plan);
            // Back to front: what is drawn is always the first strokes in paint order, and only grows.
            assert_eq!(now.as_slice(), &ids[..now.len()], "frame {frames}: {now:?}");
            assert!(now.len() >= before.len(), "frame {frames}: a stroke left");
            if frames == 0 {
                assert!(!now.is_empty() && now.len() < ids.len(), "the first frame is partial: {now:?}");
                assert_eq!(caches.placement_deferred, ids.len() - now.len());
                assert!(caches.outline_refining(), "a partial frame asks for another");
            }
            frames += 1;
            before = now;
            if !caches.outline_refining() {
                break plan;
            }
            assert!(frames < 40, "never settled");
        };
        assert!(frames > 2, "filled in over {frames} frames");
        // Settled: the from-scratch build.
        let scene = r.scene().unwrap();
        let hashes = r.hashes().unwrap();
        assert_eq!(
            crate::camera_slot::plan_divergence(scene, &[0], &cam, &opts, hashes, None, None, &last),
            None
        );
        assert_eq!(drawn(&last), ids);
    }

    /// Placement in host workers: with worker mode on, a live frame sends every never-placed bristle stroke to the
    /// workers (in paint order) and draws none of them; the placements come back out of order (several workers),
    /// are promoted in request order, and the painting still fills in back to front, a stroke once drawn never
    /// leaving, until the plan is a from-scratch build's. A failed request places on the main thread instead.
    #[test]
    fn worker_placements_fill_in_back_to_front_whatever_order_they_arrive_in() {
        let ids: Vec<String> = (0..24).map(|k| format!("b{k:02}")).collect();
        let nodes: Vec<Value> = ids
            .iter()
            .enumerate()
            .map(|(k, id)| {
                json!({"id": id, "type": "path", "x": 10.0, "y": 10.0 + 12.0 * k as f64, "width": 300.0, "height": 10.0,
                       "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#223344"}, "width": 6,
                                    "brush": {"brushId": "round", "seed": k}}],
                       "points": [{"type": "move", "x": 0, "y": 0}, {"type": "curve", "x": 300, "y": 10,
                                   "cp1": {"x": 100, "y": -20}, "cp2": {"x": 200, "y": 30}}]})
            })
            .collect();
        let doc = json!({
            "nodes": nodes,
            "pages": [{"id": "p1", "width": 400.0, "height": 320.0, "nodeIds": ids}],
            "brushes": [{"id": "round", "name": "Round", "type": "bristle", "shape": "flat-fan", "size": 3,
                         "length": 1.5, "density": 0.5, "thickness": 0.5, "opacity": 0.8, "stiffness": 0.4}],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let opts = RenderOptions::default();
        let cam = crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom: 1.0,
            viewport_w: 400.0,
            viewport_h: 320.0,
            dpr: 1.0,
            clear_rgb: [1.0; 3],
        };
        let drawn = |p: &CoveragePlan| -> Vec<String> {
            let mut out: Vec<String> = Vec::new();
            for i in &p.items {
                if let crate::seams::DrawItem::Body { node_id, .. } = i {
                    let owner = node_id.split("::").next().unwrap_or("").to_string();
                    if out.last() != Some(&owner) {
                        out.push(owner);
                    }
                }
            }
            out
        };
        let mut caches = PlanCaches::default();
        caches.set_outline_worker(true);
        let frame = |caches: &mut PlanCaches| {
            caches.begin_frame();
            caches.set_outline_refine_budget_ms(Some(28.0));
            let plan = build_camera_plan_retained(&r, caches, &[0], &cam, &opts).unwrap().unwrap().0;
            caches.set_outline_refine_budget_ms(None);
            plan
        };
        let first = frame(&mut caches);
        assert!(drawn(&first).is_empty(), "every stroke went to the workers");
        let requests: Vec<Value> = moda_scene::value::from_str(&caches.take_outline_requests()).unwrap();
        assert_eq!(requests.len(), ids.len());
        let order: Vec<&str> = requests.iter().map(|q| q["id"].as_str().unwrap()).collect();
        assert_eq!(order, ids.iter().map(String::as_str).collect::<Vec<_>>(), "requested in paint order");
        assert!(caches.outline_refining());
        // A second frame before anything arrives re-requests nothing.
        let _ = frame(&mut caches);
        assert_eq!(caches.take_outline_requests(), "[]");
        // Replies in a scrambled order (three workers); one worker fails its stroke (placed here instead).
        let mut replies: Vec<&Value> = requests.iter().collect();
        replies.sort_by_key(|q| {
            let k: usize = q["id"].as_str().unwrap()[1..].parse().unwrap();
            (k % 3, std::cmp::Reverse(k))
        });
        let mut before: Vec<String> = Vec::new();
        for (n, q) in replies.iter().enumerate() {
            let id = q["id"].as_str().unwrap();
            if id == "b05" {
                caches.outline_request_failed(id);
            } else {
                let job = crate::tess::placed::placement_job(q).unwrap();
                assert!(caches.provide_placement(
                    id,
                    q["key"].as_str().unwrap(),
                    q["base"].as_str().unwrap(),
                    q["band"].as_i64().unwrap() as i32,
                    &job.wire,
                    job.strands,
                ));
            }
            if n % 4 == 3 {
                let plan = frame(&mut caches);
                let now = drawn(&plan);
                assert_eq!(now.as_slice(), &ids[..now.len()], "after {n} replies: {now:?}");
                assert!(now.len() >= before.len(), "a stroke left");
                before = now;
            }
        }
        let mut last = frame(&mut caches);
        for _ in 0..8 {
            if !caches.outline_refining() {
                break;
            }
            last = frame(&mut caches);
        }
        assert!(!caches.outline_refining(), "settled");
        assert_eq!(drawn(&last), ids);
        let scene = r.scene().unwrap();
        let hashes = r.hashes().unwrap();
        assert_eq!(
            crate::camera_slot::plan_divergence(scene, &[0], &cam, &opts, hashes, None, None, &last),
            None
        );
        assert!(caches.outline_worker_provided >= ids.len() as u64 - 1);
    }

    /// A band change with host workers on: every on-screen bristle stroke's strands refine in the workers
    /// (`task: "strands"`), none on the main thread; each draws its previous band's strands until its reply lands,
    /// and the settled plan is a from-scratch build's at the new band. A stroke with a trim (the workers can't place
    /// it as this thread did) refines here; a worker's refusal sends its stroke back here too.
    #[test]
    fn a_band_change_refines_bristle_strands_in_the_workers() {
        let ids: Vec<String> = (0..12).map(|k| format!("b{k:02}")).collect();
        let nodes: Vec<Value> = ids
            .iter()
            .enumerate()
            .map(|(k, id)| {
                let mut n = json!({"id": id, "type": "path", "x": 10.0, "y": 10.0 + 12.0 * k as f64, "width": 300.0,
                       "height": 10.0,
                       "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#223344"}, "width": 6,
                                    "brush": {"brushId": "round", "seed": k}}],
                       "points": [{"type": "move", "x": 0, "y": 0}, {"type": "curve", "x": 300, "y": 10,
                                   "cp1": {"x": 100, "y": -20}, "cp2": {"x": 200, "y": 30}}]});
                if k == 3 {
                    n["trimPathStart"] = json!(0.1);
                    n["trimPathEnd"] = json!(0.8);
                }
                n
            })
            .collect();
        let doc = json!({
            "nodes": nodes,
            "pages": [{"id": "p1", "width": 400.0, "height": 320.0, "nodeIds": ids}],
            "brushes": [{"id": "round", "name": "Round", "type": "bristle", "shape": "flat-fan", "size": 3,
                         "length": 1.5, "density": 0.5, "thickness": 0.5, "opacity": 0.8, "stiffness": 0.4}],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let opts = RenderOptions::default();
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 1600.0,
            viewport_h: 1400.0,
            dpr: 1.0,
            clear_rgb: [1.0; 3],
        };
        let mut caches = PlanCaches::default();
        let frame = |caches: &mut PlanCaches, zoom: f64| {
            caches.begin_frame();
            caches.set_outline_refine_budget_ms(Some(28.0));
            let plan = build_camera_plan_retained(&r, caches, &[0], &cam(zoom), &opts).unwrap().unwrap().0;
            caches.set_outline_refine_budget_ms(None);
            plan
        };
        // Placed and settled at 1× on the main thread (over as many frames as the budget takes); then the workers
        // come on.
        let _ = frame(&mut caches, 1.0);
        for _ in 0..40 {
            if !caches.outline_refining() {
                break;
            }
            let _ = frame(&mut caches, 1.0);
        }
        assert!(!caches.outline_refining());
        caches.set_outline_worker(true);
        // Zoomed in across bands: the strands go to the workers.
        let _ = frame(&mut caches, 4.0);
        let requests: Vec<Value> = moda_scene::value::from_str(&caches.take_outline_requests()).unwrap();
        let asked: Vec<&str> = requests.iter().map(|q| q["id"].as_str().unwrap()).collect();
        assert!(requests.iter().all(|q| q["task"] == "strands"), "{asked:?}");
        let mut want: Vec<String> = ids.iter().filter(|id| *id != "b03").map(|id| format!("{id}::strands")).collect();
        want.sort();
        let mut got: Vec<String> = asked.iter().map(|s| s.to_string()).collect();
        got.sort();
        assert_eq!(got, want, "every untrimmed stroke's strands, and not the trimmed one's");
        assert!(caches.outline_refining(), "drawing the previous band meanwhile");
        // Nothing re-requested before a reply.
        let _ = frame(&mut caches, 4.0);
        assert_eq!(caches.take_outline_requests(), "[]");
        // The workers reply; one refuses (its stroke refines here instead).
        for q in &requests {
            let id = q["id"].as_str().unwrap();
            if id == "b07::strands" {
                caches.outline_request_failed(id);
                continue;
            }
            let g = crate::tess::placed::strands_job(q).unwrap();
            assert!(caches.provide_outline(
                id,
                q["key"].as_str().unwrap(),
                q["base"].as_str().unwrap(),
                q["band"].as_i64().unwrap() as i32,
                g,
                None
            ));
        }
        let mut last = frame(&mut caches, 4.0);
        for _ in 0..20 {
            if !caches.outline_refining() {
                break;
            }
            last = frame(&mut caches, 4.0);
        }
        assert!(!caches.outline_refining(), "settled");
        assert_eq!(caches.take_outline_requests(), "[]", "the refused stroke isn't asked again");
        let scene = r.scene().unwrap();
        let hashes = r.hashes().unwrap();
        assert_eq!(
            crate::camera_slot::plan_divergence(scene, &[0], &cam(4.0), &opts, hashes, None, None, &last),
            None
        );
        // A request for strands the worker would place differently is refused, not answered wrongly.
        let mut wrong = requests[0].clone();
        wrong["key"] = json!("12345");
        assert!(crate::tess::placed::strands_job(&wrong).is_err());
    }

    /// A band change with host workers on: every art stroke is re-placed in the workers (`task: "place"`,
    /// `kind: "art"`: its warp refits its cubics to the new band's tolerance), none on the main thread; each draws
    /// its previous band's placement (and strands) until its reply lands, and the settled plan is a from-scratch
    /// build's at the new band. A stroke with a trim (the workers can't place it as this thread did) re-places here;
    /// a worker's refusal sends its stroke back here too.
    #[test]
    fn a_band_change_re_places_art_in_the_workers() {
        let ids: Vec<String> = (0..10).map(|k| format!("a{k:02}")).collect();
        let nodes: Vec<Value> = ids
            .iter()
            .enumerate()
            .map(|(k, id)| {
                let mut n = json!({"id": id, "type": "path", "x": 10.0, "y": 10.0 + 14.0 * k as f64, "width": 300.0,
                       "height": 10.0,
                       "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#335577"}, "width": 4,
                                    "brush": {"brushId": "dry"}}],
                       "points": [{"type": "move", "x": 0, "y": 0}, {"type": "curve", "x": 300, "y": 10,
                                   "cp1": {"x": 100, "y": -20.0 - k as f64}, "cp2": {"x": 200, "y": 30}}]});
                if k == 3 {
                    n["trimPathStart"] = json!(0.1);
                    n["trimPathEnd"] = json!(0.8);
                }
                n
            })
            .collect();
        // Three one-colour curved strokes, as the library's dry brush (`lib-art-dry-brush`).
        let shape = |y: f64, len: f64, h: f64| {
            json!({"type": "path", "x": 0, "y": y, "closed": true, "fillRule": "nonzero",
                   "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}], "strokes": [],
                   "points": [{"type": "move", "x": 0, "y": h}, {"type": "curve", "cp1": {"x": 5, "y": -0.3},
                               "cp2": {"x": 0.8 * len, "y": -0.3}, "x": len, "y": h},
                              {"type": "curve", "cp1": {"x": 0.8 * len, "y": 1.6 * h}, "cp2": {"x": 5, "y": 1.6 * h},
                               "x": 0, "y": h}]})
        };
        let mut artwork = vec![shape(0.25, 100.0, 1.05), shape(2.1, 90.0, 0.9), shape(3.9, 97.0, 1.2)];
        for (k, n) in artwork.iter_mut().enumerate() {
            n["id"] = json!(format!("n{k}"));
        }
        let doc = json!({
            "nodes": nodes,
            "pages": [{"id": "p1", "width": 400.0, "height": 320.0, "nodeIds": ids}],
            "brushes": [{"id": "dry", "name": "Dry Brush", "type": "art", "scale": {"value": 1}, "fit": "stretch",
                         "direction": "right", "overlap": "none", "colorize": {"method": "tints"},
                         "artwork": {"nodes": artwork}}],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let opts = RenderOptions::default();
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 1600.0,
            viewport_h: 1400.0,
            dpr: 1.0,
            clear_rgb: [1.0; 3],
        };
        let mut caches = PlanCaches::default();
        let frame = |caches: &mut PlanCaches, zoom: f64| {
            caches.begin_frame();
            caches.set_outline_refine_budget_ms(Some(28.0));
            let plan = build_camera_plan_retained(&r, caches, &[0], &cam(zoom), &opts).unwrap().unwrap().0;
            caches.set_outline_refine_budget_ms(None);
            plan
        };
        // Placed and settled at 1× on the main thread; then the workers come on.
        let _ = frame(&mut caches, 1.0);
        for _ in 0..40 {
            if !caches.outline_refining() {
                break;
            }
            let _ = frame(&mut caches, 1.0);
        }
        assert!(!caches.outline_refining());
        let placed_at_1x: HashMap<String, u64> = ids.iter().map(|id| (id.clone(), caches.placed[id].key)).collect();
        caches.set_outline_worker(true);
        // Zoomed in across bands: the art goes to the workers.
        let _ = frame(&mut caches, 4.0);
        let requests: Vec<Value> = moda_scene::value::from_str(&caches.take_outline_requests()).unwrap();
        let asked: Vec<&str> = requests.iter().map(|q| q["id"].as_str().unwrap()).collect();
        assert!(requests.iter().all(|q| q["task"] == "place" && q["kind"] == "art"), "{asked:?}");
        let want: Vec<&str> = ids.iter().map(String::as_str).filter(|id| *id != "a03").collect();
        assert_eq!(asked, want, "every untrimmed stroke, in paint order, and not the trimmed one");
        assert!(caches.outline_refining(), "drawing the previous band meanwhile");
        // Nothing re-requested before a reply, and nothing placed here meanwhile.
        let _ = frame(&mut caches, 4.0);
        assert_eq!(caches.take_outline_requests(), "[]");
        for id in &want {
            assert_eq!(caches.placed[*id].key, placed_at_1x[*id], "{id} placed on the main thread");
        }
        // The workers reply (in a scrambled order); one refuses (its stroke re-places here instead).
        let (mut wire_bytes, mut strand_bytes) = (0usize, 0usize);
        let mut replies: Vec<&Value> = requests.iter().collect();
        replies.reverse();
        for q in replies {
            let id = q["id"].as_str().unwrap();
            if id == "a06" {
                caches.outline_request_failed(id);
                continue;
            }
            let job = crate::tess::placed::placement_job(q).unwrap();
            assert!(job.strands.is_some(), "one-colour art draws as one body");
            strand_bytes += job.strands.as_ref().map_or(0, |g| 4 * (g.positions.len() + g.vertex_types.len()));
            let w = &job.wire;
            wire_bytes += w.meta.len()
                + 4 * (w.paint.len() + w.source.len() + w.counts.len() + w.hashes.len())
                + w.closed.len()
                + w.kinds.len()
                + 8 * (w.at.len() + w.coords.len());
            assert!(caches.provide_placement(
                id,
                q["key"].as_str().unwrap(),
                q["base"].as_str().unwrap(),
                q["band"].as_i64().unwrap() as i32,
                &job.wire,
                job.strands,
            ));
        }
        eprintln!("art placement wire: {wire_bytes} B, strands {strand_bytes} B, for {} strokes", want.len() - 1);
        let mut last = frame(&mut caches, 4.0);
        for _ in 0..20 {
            if !caches.outline_refining() {
                break;
            }
            last = frame(&mut caches, 4.0);
        }
        assert!(!caches.outline_refining(), "settled");
        assert_eq!(caches.take_outline_requests(), "[]", "the refused stroke isn't asked again");
        let scene = r.scene().unwrap();
        let hashes = r.hashes().unwrap();
        assert_eq!(
            crate::camera_slot::plan_divergence(scene, &[0], &cam(4.0), &opts, hashes, None, None, &last),
            None
        );
        // Back out to 1×: re-placed in the workers again, from the 4× placements.
        let _ = frame(&mut caches, 1.0);
        let requests: Vec<Value> = moda_scene::value::from_str(&caches.take_outline_requests()).unwrap();
        assert!(requests.iter().all(|q| q["task"] == "place"));
        assert!(!requests.is_empty() && requests.iter().all(|q| q["id"] != "a06" && q["id"] != "a03"));
    }

    /// A unit that draws nothing at one band (a stroke whose geometry flattens away there) may draw at another: a
    /// change of band alone queues it with the banded units, so the settled plan has it (Poplars lost a few
    /// strokes at fit → 100 % without this).
    #[test]
    fn a_band_change_refines_units_that_drew_nothing() {
        let doc = json!({
            "nodes": [
                {"id": "curve", "type": "path", "x": 10.0, "y": 10.0, "width": 100.0, "height": 40.0,
                 "fills": [{"id": "f", "paint": {"type": "solid", "color": "#3a86ff"}}], "closed": true,
                 "points": [{"type": "move", "x": 0, "y": 20}, {"type": "curve", "x": 100, "y": 20,
                             "cp1": {"x": 20, "y": -20}, "cp2": {"x": 80, "y": 60}}]},
                // A fill with no area: no geometry at this band ("empty geometry"), but a unit all the same.
                {"id": "dot", "type": "path", "x": 200.0, "y": 50.0, "width": 20.0, "height": 0.0, "closed": true,
                 "fills": [{"id": "f", "paint": {"type": "solid", "color": "#111111"}}],
                 "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 10, "y": 0}, {"type": "line", "x": 20, "y": 0}]},
                {"id": "box", "type": "rectangle", "x": 250.0, "y": 10.0, "width": 40.0, "height": 30.0,
                 "fills": [{"id": "f", "paint": {"type": "solid", "color": "#ef476f"}}]},
            ],
            "pages": [{"id": "p1", "width": 400.0, "height": 300.0, "nodeIds": ["curve", "dot", "box"]}],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let cam = crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom: 1.0,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 1.0,
            clear_rgb: [1.0; 3],
        };
        let mut caches = PlanCaches::default();
        caches.begin_frame();
        let (plan, _) = build_camera_plan_retained(&r, &mut caches, &[0], &cam, &RenderOptions::default())
            .unwrap()
            .unwrap();
        assert!(
            !plan.items.iter().any(|i| matches!(i, crate::seams::DrawItem::Body { node_id, .. } if node_id == "dot")),
            "the flat path draws nothing"
        );
        let slot = caches.camera_slot_ref().expect("a retained slot");
        let ids: Vec<String> = slot.band_refine_candidates().into_iter().map(|(id, _)| id).collect();
        assert!(ids.iter().any(|i| i == "curve"), "a tessellated body: {ids:?}");
        assert!(ids.iter().any(|i| i == "dot"), "a unit that drew nothing: {ids:?}");
        assert!(!ids.iter().any(|i| i == "box"), "an SDF shape doesn't depend on the band: {ids:?}");
    }

    /// A layer capture's content key follows the geometry its items draw with. A live frame may plan a blurred
    /// group's paths from geometry it is still refining (the previous band's) and refine it on later frames without
    /// any node changing; keyed on the nodes alone, the capture of the stand-in was served for good (anime-dusk:
    /// 107k px off a fresh engine after fit → 100 %).
    #[test]
    fn a_capture_key_follows_its_items_geometry() {
        let doc = json!({
            "nodes": [
                {"id": "g", "type": "group", "x": 0.0, "y": 0.0, "children": ["p"],
                 "effects": [{"id": "b", "type": "layer-blur", "radius": 6}]},
                {"id": "p", "type": "path", "x": 20.0, "y": 20.0, "width": 100.0, "height": 40.0, "closed": true,
                 "fills": [{"id": "f", "paint": {"type": "solid", "color": "#3a86ff"}}],
                 "points": [{"type": "move", "x": 0, "y": 20}, {"type": "curve", "x": 100, "y": 20,
                             "cp1": {"x": 20, "y": -20}, "cp2": {"x": 80, "y": 60}}]},
            ],
            "pages": [{"id": "p1", "width": 400.0, "height": 300.0, "nodeIds": ["g"]}],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let opts = RenderOptions::default();
        let (plan, _) = crate::frame::build_coverage_plan(r.scene().unwrap(), 0, &opts).unwrap();
        let group = plan
            .items
            .iter()
            .find_map(|i| match i {
                crate::seams::DrawItem::LayerBlurComposite { group, .. } => Some(group.clone()),
                _ => None,
            })
            .expect("a layer capture");
        let scene = r.scene().unwrap();
        let hashes = r.hashes();
        let key = |g: &crate::seams::LayerBlurGroup| {
            capture_key(scene, hashes, ["g", "p"], g, [0.0, 0.0, 400.0, 300.0], &opts)
        };
        let before = key(&group);
        assert!(before.0.is_some());
        assert_eq!(key(&group), before, "the same items, the same key");
        // The same path drawn from another tessellation (a coarser band's stand-in, then its refinement).
        let mut refined = (*group).clone();
        let mut swapped = false;
        for item in &mut refined.items {
            if let crate::seams::DrawItem::Body { geometry, .. } = item {
                let mut g = (**geometry).clone();
                g.positions.iter_mut().for_each(|v| *v += 0.25);
                *geometry = Arc::new(g);
                swapped = true;
            }
        }
        assert!(swapped, "the capture holds the path's body");
        let after = key(&refined);
        assert_ne!(after.0, before.0, "another geometry, another capture");
        assert_eq!(after.1, before.1, "the identity (a zoom step's stand-in) stays");

        // A grandchild capture's target region is also part of the outer pixels.
        let wrap = |child: crate::seams::LayerBlurGroup| {
            let mut parent = (*group).clone();
            parent.items = vec![crate::seams::DrawItem::LayerBlurComposite {
                node_id: "nested".into(), z: 0., group: Arc::new(child),
                page_clip_bounds: [0., 0., 400., 300.], clip_rrect: None, mask: None,
            }];
            parent
        };
        let nested = wrap(wrap((*group).clone()));
        let mut moved = (*group).clone();
        moved.region_canvas[0] += 1.;
        let moved = wrap(wrap(moved));
        assert_ne!(key(&nested).0, key(&moved).0, "deep capture regions affect the key");
        assert_eq!(key(&nested).1, key(&moved).1, "content identity stays");
    }

    /// The same document at the same zoom settles to a from-scratch build's plan however much each frame had time to
    /// refine. Glowing curves: a glow draws its node's silhouette, tessellated through the per-node cache under its
    /// own id (`<node>\u{1}silhouette`); drawn from the previous band past the budget, it queued under that id,
    /// which names no node, so the re-plan skipped it and the glow stayed coarse for good (anime-dusk: 98k px off a
    /// from-scratch build at 100 % on a slow machine, none on a fast one).
    #[test]
    fn settling_reaches_a_fresh_build_whatever_the_refine_budget() {
        let nodes: Vec<Value> = (0..8)
            .map(|k| {
                json!({"id": format!("n{k}"), "type": "path", "x": 20.0 + (k % 4) as f64 * 90.0,
                       "y": 30.0 + (k / 4) as f64 * 120.0, "width": 60.0, "height": 80.0, "closed": true,
                       "fills": [{"id": "f", "paint": {"type": "solid", "color": "#fff4dc"}}],
                       "points": [{"type": "move", "x": 30, "y": 0},
                                  {"type": "curve", "x": 10, "y": 70, "cp1": {"x": 0, "y": 10}, "cp2": {"x": -10, "y": 50}},
                                  {"type": "curve", "x": 60, "y": 60, "cp1": {"x": 30, "y": 90}, "cp2": {"x": 55, "y": 85}},
                                  {"type": "curve", "x": 30, "y": 0, "cp1": {"x": 40, "y": 40}, "cp2": {"x": 20, "y": 20}}],
                       "effects": [{"type": "glow", "id": "g", "color": "#ffe0b0", "radiusPx": 12, "intensity": 1, "opacity": 0.8}]})
            })
            .collect();
        let ids: Vec<String> = (0..8).map(|k| format!("n{k}")).collect();
        let doc = json!({
            "nodes": nodes,
            "pages": [{"id": "p1", "width": 400.0, "height": 300.0, "nodeIds": ids}],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let opts = RenderOptions { sample_count: 4, ..Default::default() };
        // The whole page on screen at every zoom (an off-screen body waits for the camera to refine, by design).
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 1600.0,
            viewport_h: 1200.0,
            dpr: 1.0,
            clear_rgb: [1.0; 3],
        };
        // No refine budget (every frame refines what it can: one refinement, then the deadline has passed), and an
        // unlimited one.
        for budget in [Some(0.0), None] {
            let mut caches = PlanCaches::default();
            let mut frame = |zoom: f64, caches: &mut PlanCaches| {
                caches.begin_frame();
                caches.set_outline_refine_budget_ms(budget);
                let p = build_camera_plan_retained(&r, caches, &[0], &cam(zoom), &opts).unwrap().unwrap().0;
                caches.set_outline_refine_budget_ms(None);
                p
            };
            for zoom in [0.25, 1.0, 4.0] {
                let mut p = frame(zoom, &mut caches);
                for _ in 0..200 {
                    if !caches.outline_refining() {
                        break;
                    }
                    p = frame(zoom, &mut caches);
                }
                assert!(!caches.outline_refining(), "budget {budget:?}: settled at {zoom}");
                let scene = r.scene().unwrap();
                let hashes = r.hashes().unwrap();
                assert_eq!(
                    crate::camera_slot::plan_divergence(
                        scene,
                        &[0],
                        &cam(zoom),
                        &opts,
                        hashes,
                        caches.camera_cull.as_ref(),
                        caches.capture_view,
                        &p
                    ),
                    None,
                    "budget {budget:?}, zoom {zoom}"
                );
            }
        }
    }

    /// A body inside a layer capture reaches the screen through the capture's composite, whatever its own box: after
    /// a band change it refines as if on screen, and the frame keeps refining until it has. Off screen on its own, it
    /// was left drawing its previous band inside an on-screen capture for good (anime-dusk: 4.7k px off a
    /// from-scratch build at 100 %).
    #[test]
    fn bodies_inside_a_capture_refine_even_off_screen() {
        let curve = |id: &str, x: f64| {
            json!({"id": id, "type": "path", "x": x, "y": 40.0, "width": 100.0, "height": 40.0, "closed": true,
                   "fills": [{"id": "f", "paint": {"type": "solid", "color": "#3a86ff"}}],
                   "points": [{"type": "move", "x": 0, "y": 20}, {"type": "curve", "x": 100, "y": 20,
                               "cp1": {"x": 20, "y": -20}, "cp2": {"x": 80, "y": 60}}]})
        };
        let doc = json!({
            "nodes": [
                {"id": "g", "type": "group", "x": 0.0, "y": 0.0, "children": ["near", "far"],
                 "effects": [{"id": "b", "type": "layer-blur", "radius": 6}]},
                curve("near", 20.0),
                // Far past the view (and its refinement margin) on its own.
                curve("far", 3000.0),
            ],
            "pages": [{"id": "p1", "width": 3200.0, "height": 300.0, "nodeIds": ["g"]}],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let opts = RenderOptions { sample_count: 4, ..Default::default() };
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 1.0,
            clear_rgb: [1.0; 3],
        };
        let mut caches = PlanCaches::default();
        let mut frame = |zoom: f64, caches: &mut PlanCaches| {
            caches.begin_frame();
            caches.set_outline_refine_budget_ms(Some(0.0));
            let p = build_camera_plan_retained(&r, caches, &[0], &cam(zoom), &opts).unwrap().unwrap().0;
            caches.set_outline_refine_budget_ms(None);
            p
        };
        for zoom in [1.0, 4.0] {
            let mut p = frame(zoom, &mut caches);
            for _ in 0..20 {
                if !caches.outline_refining() {
                    break;
                }
                p = frame(zoom, &mut caches);
            }
            assert!(!caches.outline_refining(), "settled at {zoom}");
            let scene = r.scene().unwrap();
            let hashes = r.hashes().unwrap();
            assert_eq!(
                crate::camera_slot::plan_divergence(
                    scene,
                    &[0],
                    &cam(zoom),
                    &opts,
                    hashes,
                    caches.camera_cull.as_ref(),
                    caches.capture_view,
                    &p
                ),
                None,
                "zoom {zoom}"
            );
        }
    }

    /// A plan with layer captures rebuilds at a change of zoom band instead of keeping the slot: a capture's pixels
    /// depend on its window (its region is cut to it; grain and blur follow), and the kept slot's window is the old
    /// view's (sumi-e and portrait settled 1.8k and 85 px off a from-scratch build at 100 %).
    #[test]
    fn a_band_change_rebuilds_a_plan_with_captures() {
        let doc = json!({
            "nodes": [
                {"id": "g", "type": "group", "x": 0.0, "y": 0.0, "children": ["p"],
                 "effects": [{"id": "b", "type": "layer-blur", "radius": 6}]},
                {"id": "p", "type": "path", "x": 20.0, "y": 40.0, "width": 100.0, "height": 40.0, "closed": true,
                 "fills": [{"id": "f", "paint": {"type": "solid", "color": "#3a86ff"}}],
                 "points": [{"type": "move", "x": 0, "y": 20}, {"type": "curve", "x": 100, "y": 20,
                             "cp1": {"x": 20, "y": -20}, "cp2": {"x": 80, "y": 60}}]},
            ],
            "pages": [{"id": "p1", "width": 400.0, "height": 300.0, "nodeIds": ["g"]}],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let opts = RenderOptions { sample_count: 4, ..Default::default() };
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 1.0,
            clear_rgb: [1.0; 3],
        };
        let mut caches = PlanCaches::default();
        for zoom in [1.0, 1.0, 4.0] {
            caches.begin_frame();
            let _ = build_camera_plan_retained(&r, &mut caches, &[0], &cam(zoom), &opts).unwrap();
        }
        assert_eq!(caches.band_rekeys, 0, "not kept across the band");
        assert_eq!(caches.camera_rebuilds, 2, "rebuilt at the new band");
    }

    /// Diagnostic (ignored): `SETTLE_SCENE=<scene json>` — settle at fit, step to 100 %, settle, and report where the
    /// settled plan differs from a fresh build. `SETTLE_REKEY=off` takes the rebuild path.
    #[test]
    #[ignore]
    fn diag_settled_step_matches_a_fresh_build() {
        let Some(path) = std::env::var_os("SETTLE_SCENE") else { return };
        let json = std::fs::read_to_string(path).unwrap();
        let mut r = ResidentScene::default();
        r.update(&json).unwrap();
        let scene = r.scene().unwrap();
        let (ox, oy, pw, ph, pages) = match scene.doc().pages.first() {
            Some(p) => {
                let (x, y) = scene.page_position(0);
                (x, y, p.width(), p.height(), vec![0usize])
            }
            None => {
                let b = moda_scene::display_list::canvas_root_content_bounds(scene).unwrap();
                (b.x, b.y, b.width, b.height, Vec::new())
            }
        };
        let (vw, vh, dpr) = (1440.0, 900.0, 2.0);
        let fit = (vw / pw).min(vh / ph) * 0.9;
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: ox + pw / 2.0 - vw / zoom / 2.0,
            camera_y: oy + ph / 2.0 - vh / zoom / 2.0,
            zoom,
            viewport_w: vw,
            viewport_h: vh,
            dpr,
            clear_rgb: [0.93, 0.93, 0.93],
        };
        let opts = RenderOptions {
            preserve_page_fill: false,
            scale: dpr,
            transparent: false,
            sample_count: 4,
            shader_time_s: 0.0,
            exclude: Vec::new(),
            isolate: Vec::new(),
            best_effort_text: true,
            text_masks_available: Some(true),
            working: None,
        };
        let mut caches = PlanCaches::default();
        caches.band_rekey_disabled = std::env::var("SETTLE_REKEY").is_ok_and(|v| v == "off");
        // SETTLE_FAKE_CLOCK=<step>: every clock read advances `step` ms (a loaded machine: little refines a frame).
        if let Some(step) = std::env::var("SETTLE_FAKE_CLOCK").ok().and_then(|v| v.parse::<u64>().ok()) {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NOW: AtomicU64 = AtomicU64::new(0);
            static STEP: AtomicU64 = AtomicU64::new(1);
            STEP.store(step, Ordering::Relaxed);
            fn clock() -> f64 {
                NOW.fetch_add(STEP.load(Ordering::Relaxed), Ordering::Relaxed) as f64
            }
            caches.set_outline_clock(clock);
        }
        let ms: f64 = std::env::var("SETTLE_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(6.0);
        // SETTLE_GPU=1: present every frame through a live engine (its prepared frame patched frame to frame, as in
        // the editor), and at the end compare its last frame with a fresh engine drawing the same plan.
        let gpu = std::env::var_os("SETTLE_GPU").is_some();
        let new_engine = || {
            let g = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("gpu");
            let mut e = crate::Engine::new(g).expect("engine");
            e.set_retained(true);
            if std::env::var("SETTLE_RUNS").is_ok_and(|v| v == "off") {
                e.set_solid_run_merging(false);
            }
            e
        };
        let live = gpu.then(new_engine);
        if std::env::var("SETTLE_RESIDENCY").is_ok_and(|v| v == "off") {
            if let Some(e) = live.as_ref() {
                e.set_vertex_residency(false);
            }
        }
        if std::env::var("SETTLE_CAPCACHE").is_ok_and(|v| v == "off") {
            if let Some(e) = live.as_ref() {
                e.set_capture_cache_budget(0);
            }
        }
        if std::env::var("SETTLE_PATCH").is_ok_and(|v| v == "off") {
            if let Some(e) = live.as_ref() {
                e.set_prepared_patch(false);
            }
        }
        let present = |engine: &crate::Engine, plan: &Arc<CoveragePlan>, zoom: f64| -> Vec<u8> {
            let c = cam(zoom);
            let (w, h) = (crate::export_scale::px_ceil(c.viewport_w * c.dpr) as u32, crate::export_scale::px_ceil(c.viewport_h * c.dpr) as u32);
            let surface = engine.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("diag surface"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: engine.frame_format(),
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = surface.create_view(&wgpu::TextureViewDescriptor::default());
            engine
                .render_plan_to_view(plan, Some(&c.present_override(0.0)), &view, Some(&surface), Some(plan), true)
                .expect("present");
            crate::readback::read_texture_rgba(&engine.gpu.device, &engine.gpu.queue, &surface, w, h).expect("readback")
        };
        let last_px: std::cell::RefCell<Vec<u8>> = Default::default();
        let budget = std::cell::Cell::new(ms);
        let mut frame = |zoom: f64, caches: &mut PlanCaches| {
            caches.begin_frame();
            caches.set_outline_refine_budget_ms(Some(budget.get()));
            let p = build_camera_plan_retained(&r, caches, &pages, &cam(zoom), &opts).unwrap().unwrap().0;
            caches.set_outline_refine_budget_ms(None);
            if let Some(e) = live.as_ref() {
                *last_px.borrow_mut() = present(e, &p, zoom);
                if std::env::var_os("SETTLE_PF").is_some() {
                    let st = e.prepared_frame_stats();
                    eprintln!(
                        "  pf: hits {} rebuilds {} patches {} patched items {} replays {} fallback {:?}; items {}",
                        st.hits, st.rebuilds, st.patches, st.patched_items, st.bundle_replays, e.patch_fallback_reason(), p.items.len()
                    );
                }
            }
            p
        };
        // SETTLE_SEQ: `zoom:budget` steps (`fit` for the fit zoom), default the bench's `fit:28,fit:6,1:6`.
        let seq: Vec<(f64, f64)> = std::env::var("SETTLE_SEQ")
            .unwrap_or_else(|_| "fit:28,fit:6,1:6".into())
            .split(',')
            .map(|p| {
                let (z, b) = p.split_once(':').unwrap();
                (if z == "fit" { fit } else { z.parse().unwrap() }, b.parse().unwrap())
            })
            .collect();
        for (zoom, b) in seq {
            budget.set(b);
            let mut n = 0;
            let mut p = frame(zoom, &mut caches);
            let t0 = std::time::Instant::now();
            while (caches.outline_refining() || live.as_ref().is_some_and(|e| e.captures_pending())) && n < 2000 {
                p = frame(zoom, &mut caches);
                n += 1;
                if n % 100 == 0 {
                    eprintln!("  frame {n}: pending {} visible {} patch nodes {} ({:.0} ms)", caches.outline_pending.len(), caches.outline_visible_pending(), caches.incr_patch_nodes, t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            let extra: usize = std::env::var("SETTLE_EXTRA").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            for _ in 0..extra {
                p = frame(zoom, &mut caches);
            }
            eprintln!(
                "zoom {zoom:.3}: {n} frames, refining {}, pending {}, rebuilds {}, rekeys {}, last fallback {:?}",
                caches.outline_refining(),
                caches.outline_pending.len(),
                caches.camera_rebuilds,
                caches.band_rekeys,
                caches.incr_last_fallback
            );
            for q in caches.outline_pending.iter().take(5) {
                eprintln!("  pending {} bbox {:?} unplaced {}; view {:?}", q.id, q.bbox, q.unplaced, caches.outline_view);
            }
            let scene = r.scene().unwrap();
            let hashes = r.hashes().unwrap();
            let d = crate::camera_slot::plan_divergence(
                scene,
                &pages,
                &cam(zoom),
                &opts,
                hashes,
                caches.camera_cull.as_ref(),
                caches.capture_view,
                &p,
            );
            eprintln!("zoom {zoom:.3}: divergence {}", d.as_deref().map_or("none".into(), |s| s.chars().take(1500).collect::<String>()));
            // Every node whose bodies differ from a from-scratch plan's (not just the first).
            {
                let mut cold = PlanCaches::default();
                let fresh = build_camera_plan_retained(&r, &mut cold, &pages, &cam(zoom), &opts).unwrap().unwrap().0;
                let bodies = |p: &CoveragePlan| {
                    let mut m: std::collections::BTreeMap<String, Vec<u32>> = Default::default();
                    for i in &p.items {
                        if let crate::seams::DrawItem::Body { node_id, geometry, .. } = i {
                            m.entry(node_id.clone()).or_default().extend(geometry.positions.iter().map(|v| v.to_bits()));
                        }
                    }
                    m
                };
                let (a, b) = (bodies(&p), bodies(&fresh));
                let differ: Vec<&String> = a.keys().chain(b.keys()).filter(|k| a.get(*k) != b.get(*k)).collect();
                let mut differ: Vec<&String> = differ;
                differ.dedup();
                eprintln!("zoom {zoom:.3}: {} nodes' bodies differ from a from-scratch plan", differ.len());
                for id in differ.iter().take(8) {
                    let pend = caches.outline_pending.iter().find(|q| &q.id == *id).map(|q| q.bbox);
                    eprintln!("   {id}: live {} floats, fresh {} floats, pending {:?}", a.get(*id).map_or(0, |v| v.len()), b.get(*id).map_or(0, |v| v.len()), pend);
                }
            }
            if gpu {
                let fresh = new_engine();
                let want = present(&fresh, &p, zoom);
                let got = last_px.borrow();
                let diffs = got.chunks(4).zip(want.chunks(4)).filter(|(a, b)| a != b).count();
                eprintln!("zoom {zoom:.3}: live engine's last frame vs a fresh engine's: {diffs} px differ");
                if let Some(dir) = std::env::var_os("SETTLE_DUMP") {
                    let c = cam(zoom);
                    let (w, h) = (crate::export_scale::px_ceil(c.viewport_w * c.dpr) as u32, crate::export_scale::px_ceil(c.viewport_h * c.dpr) as u32);
                    let dir = std::path::PathBuf::from(dir);
                    let _ = std::fs::create_dir_all(&dir);
                    std::fs::write(dir.join(format!("live-{zoom:.3}.rgba")), &*got).unwrap();
                    std::fs::write(dir.join(format!("fresh-{zoom:.3}.rgba")), &want).unwrap();
                    eprintln!("dumped {w}x{h} rgba");
                }
                // And against a from-scratch plan (no refine budget) in a fresh engine: what a fresh open shows.
                let mut cold = PlanCaches::default();
                let fresh_plan = build_camera_plan_retained(&r, &mut cold, &pages, &cam(zoom), &opts).unwrap().unwrap().0;
                let want = present(&new_engine(), &fresh_plan, zoom);
                let diffs = got.chunks(4).zip(want.chunks(4)).filter(|(a, b)| a != b).count();
                eprintln!("zoom {zoom:.3}: live engine's last frame vs a from-scratch build's: {diffs} px differ");
                if let Some(dir) = std::env::var_os("SETTLE_DUMP") {
                    std::fs::write(std::path::PathBuf::from(dir).join(format!("scratch-{zoom:.3}.rgba")), &want).unwrap();
                }
            }
        }
    }

    #[test]
    fn bristle_brushes_draw_strands_in_the_node_slot() {
        let doc = json!({
            "nodes": [
                {"id":"b","type":"path","x":10.0,"y":10.0,"width":300.0,"height":10.0,
                 "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#223344"}, "width": 2, "brush": {"brushId": "round", "seed": 5}}],
                 "points":[{"type":"move","x":0,"y":0},{"type":"line","x":300,"y":0}]},
            ],
            "pages": [{"id":"p1","width":400.0,"height":300.0,"nodeIds":["b"]}],
            "brushes": [
                {"id":"round","name":"Round","type":"bristle","shape":"flat-fan","size":3,"length":1.5,
                 "density":0.5,"thickness":0.5,"opacity":0.8,"stiffness":0.4},
            ],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let opts = RenderOptions::default();
        let strand_bodies = |runs: bool| {
            crate::frame::set_strand_runs(runs);
            let (plan, w) = crate::frame::build_coverage_plan(r.scene().unwrap(), 0, &opts).unwrap();
            crate::frame::set_strand_runs(true);
            assert!(w.dropped.is_empty() && w.skipped.is_empty(), "{w:?}");
            let bodies: Vec<u32> = plan
                .items
                .iter()
                .filter_map(|i| match i {
                    crate::seams::DrawItem::Body { node_id, geometry, .. } if node_id.starts_with("b::brush::") => {
                        Some(geometry.vertex_count)
                    }
                    _ => None,
                })
                .collect();
            bodies
        };
        // One body per strand, round(36 × 0.5) of them; as one run, one body of all their triangles.
        let one_by_one = strand_bodies(false);
        assert_eq!(one_by_one.len(), 18, "round(36 x 0.5) strands");
        assert_eq!(strand_bodies(true), vec![one_by_one.iter().sum::<u32>()]);
    }

    /// After a zoom-band change a live frame tessellates the strands its refine budget allows and draws the rest from
    /// their previous band, queued; further frames refine them until the plan is a fresh build's.
    /// A change of zoom band alone keeps the retained slot: the first frame at the new band is a patch that
    /// re-plans nothing yet (the old band's items draw meanwhile), and the next frames refine every banded unit, a
    /// budget's worth a frame, until the plan is a from-scratch build's at the new band. Curves, round joins,
    /// bristle strands and SDF shapes, all on screen (off-screen units wait for the camera, as refinement does);
    /// in across two bands (zooming out reuses a finer band's strands by design, #1268). With the switch off, a band
    /// change rebuilds, and settles to the same plan.
    #[test]
    fn a_band_change_keeps_the_slot_and_refines_to_a_fresh_build() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NOW: AtomicU64 = AtomicU64::new(0);
        fn clock() -> f64 {
            NOW.fetch_add(1, Ordering::Relaxed) as f64
        }
        let mut nodes: Vec<Value> = Vec::new();
        let mut ids: Vec<String> = Vec::new();
        for k in 0..30 {
            let id = format!("n{k:02}");
            let (x, y) = (10.0 + (k % 6) as f64 * 60.0, 10.0 + (k / 6) as f64 * 55.0);
            let n = match k % 5 {
                0 => json!({"id": id, "type": "path", "x": x, "y": y, "width": 50.0, "height": 40.0,
                    "fills": [{"id": "f", "paint": {"type": "solid", "color": "#3a86ff"}}],
                    "points": [{"type": "move", "x": 0, "y": 20}, {"type": "curve", "x": 50, "y": 20,
                                "cp1": {"x": 10, "y": -10}, "cp2": {"x": 40, "y": 50}}], "closed": true}),
                1 => json!({"id": id, "type": "path", "x": x, "y": y, "width": 50.0, "height": 40.0,
                    "strokes": [{"id": "s", "paint": {"type": "solid", "color": "#222222"}, "width": 5, "join": "round", "cap": "round"}],
                    "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 50, "y": 10}, {"type": "line", "x": 10, "y": 40}]}),
                2 => json!({"id": id, "type": "path", "x": x, "y": y, "width": 50.0, "height": 10.0,
                    "strokes": [{"id": "s", "paint": {"type": "solid", "color": "#223344"}, "width": 3,
                                 "brush": {"brushId": "round", "seed": k}}],
                    "points": [{"type": "move", "x": 0, "y": 0}, {"type": "curve", "x": 50, "y": 0,
                                "cp1": {"x": 15, "y": -10}, "cp2": {"x": 35, "y": 10}}]}),
                3 => json!({"id": id, "type": "rectangle", "x": x, "y": y, "width": 40.0, "height": 30.0,
                    "fills": [{"id": "f", "paint": {"type": "solid", "color": "#ef476f"}}]}),
                _ => json!({"id": id, "type": "ellipse", "x": x, "y": y, "width": 40.0, "height": 30.0,
                    "fills": [{"id": "f", "paint": {"type": "solid", "color": "#06d6a0"}}],
                    "strokes": [{"id": "s", "paint": {"type": "solid", "color": "#111111"}, "width": 2}]}),
            };
            nodes.push(n);
            ids.push(id);
        }
        let doc = json!({
            "nodes": nodes,
            "pages": [{"id": "p1", "width": 400.0, "height": 340.0, "nodeIds": ids}],
            "brushes": [{"id": "round", "name": "Round", "type": "bristle", "shape": "flat-fan", "size": 3,
                         "length": 1.5, "density": 0.5, "thickness": 0.5, "opacity": 0.8, "stiffness": 0.4}],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let opts = RenderOptions { sample_count: 4, ..Default::default() };
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            // Wide enough to hold the whole page at 4x.
            viewport_w: 1600.0,
            viewport_h: 1360.0,
            dpr: 1.0,
            clear_rgb: [1.0; 3],
        };
        for rekey in [true, false] {
            let mut caches = PlanCaches::default();
            caches.set_outline_clock(clock);
            caches.band_rekey_disabled = !rekey;
            let mut frame = |zoom: f64, caches: &mut PlanCaches| {
                caches.begin_frame();
                caches.set_outline_refine_budget_ms(Some(4.0));
                let p = build_camera_plan_retained(&r, caches, &[0], &cam(zoom), &opts).unwrap().unwrap().0;
                caches.set_outline_refine_budget_ms(None);
                p
            };
            let settle = |zoom: f64, caches: &mut PlanCaches, frame: &mut dyn FnMut(f64, &mut PlanCaches) -> Arc<CoveragePlan>| {
                let mut p = frame(zoom, caches);
                for _ in 0..200 {
                    if !caches.outline_refining() {
                        break;
                    }
                    p = frame(zoom, caches);
                }
                assert!(!caches.outline_refining(), "settled at {zoom}");
                p
            };
            let _ = settle(1.0, &mut caches, &mut frame);
            for zoom in [2.0, 4.0] {
                let rekeys = caches.band_rekeys;
                let rebuilds = caches.camera_rebuilds;
                let _ = frame(zoom, &mut caches);
                if rekey {
                    assert_eq!(caches.band_rekeys, rekeys + 1, "zoom {zoom}: the slot was kept");
                    assert_eq!(caches.camera_rebuilds, rebuilds, "zoom {zoom}: no rebuild");
                } else {
                    assert!(caches.camera_rebuilds > rebuilds, "zoom {zoom}: rebuilt");
                }
                let last = settle(zoom, &mut caches, &mut frame);
                let scene = r.scene().unwrap();
                let hashes = r.hashes().unwrap();
                let d = crate::camera_slot::plan_divergence(scene, &[0], &cam(zoom), &opts, hashes, None, None, &last);
                assert_eq!(d, None, "rekey {rekey}, zoom {zoom}");
            }
        }
    }

    #[test]
    fn bristle_strands_refine_progressively_after_a_band_change() {
        let stroke = |id: &str, y: f64| {
            json!({"id": id, "type": "path", "x": 10.0, "y": y, "width": 300.0, "height": 10.0,
                   "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#223344"}, "width": 2,
                                "brush": {"brushId": "round", "seed": 5}}],
                   "points": [{"type": "move", "x": 0, "y": 0},
                              {"type": "curve", "x": 300, "y": 0, "cp1": {"x": 100, "y": -40}, "cp2": {"x": 200, "y": 40}}]})
        };
        let doc = json!({
            "nodes": [stroke("b", 40.0), stroke("c", 120.0)],
            "pages": [{"id":"p1","width":400.0,"height":300.0,"nodeIds":["b", "c"]}],
            "brushes": [
                {"id":"round","name":"Round","type":"bristle","shape":"flat-fan","size":3,"length":1.5,
                 "density":0.5,"thickness":0.5,"opacity":0.8,"stiffness":0.4},
            ],
            "config": {},
        });
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let opts = RenderOptions {
            sample_count: 4,
            ..Default::default()
        };
        let cam = |zoom: f64| crate::frame::CameraParams {
            camera_x: 0.0,
            camera_y: 0.0,
            zoom,
            viewport_w: 400.0,
            viewport_h: 300.0,
            dpr: 2.0,
            clear_rgb: [0.0; 3],
        };
        let strands = |p: &CoveragePlan| -> Vec<(String, Vec<f32>)> {
            p.items
                .iter()
                .filter_map(|i| match i {
                    crate::seams::DrawItem::Body {
                        node_id, geometry, ..
                    } if node_id.contains("::brush::") => Some((node_id.clone(), geometry.positions.clone())),
                    _ => None,
                })
                .collect()
        };
        let reference = |zoom: f64| {
            let mut full = PlanCaches::default();
            strands(&build_camera_plan_retained(&r, &mut full, &[0], &cam(zoom), &opts).unwrap().unwrap().0)
        };
        let mut caches = PlanCaches::default();
        let mut frame = |zoom: f64| {
            caches.begin_frame();
            caches.set_outline_refine_budget_ms(Some(0.0));
            let (p, _) = build_camera_plan_retained(&r, &mut caches, &[0], &cam(zoom), &opts)
                .unwrap()
                .unwrap();
            caches.set_outline_refine_budget_ms(None);
            (strands(&p), caches.outline_refining())
        };
        // Opening: the strokes place (one a frame at a zero budget, `placed_get`), their strands tessellated.
        let (mut s1, mut refining) = frame(1.0);
        for _ in 0..4 {
            if !refining {
                break;
            }
            (s1, refining) = frame(1.0);
        }
        assert!(!refining);
        assert_eq!(s1, reference(1.0));
        // A band change past the budget: one stroke's strands refine (guaranteed progress), the other's draw from
        // the previous band and wait.
        let want = reference(4.0);
        assert_ne!(want, s1, "the strands flatten per band");
        let (s2, refining) = frame(4.0);
        assert!(refining, "a stroke waits");
        assert_eq!(s2.len(), 2);
        assert_eq!((s2[0] == want[0]) as u8 + (s2[1] == want[1]) as u8, 1, "one refined, one stale");
        assert!(s2.iter().any(|x| s1.contains(x)), "the stale one is its previous band's");
        // Held there, the rest refine: the plan is a fresh build's.
        let mut last = s2;
        for _ in 0..8 {
            let (s, refining) = frame(4.0);
            last = s;
            if !refining {
                break;
            }
        }
        assert_eq!(last, want);
    }

    #[test]
    fn placed_brushes_draw_artwork_in_the_node_slot() {
        let mut r = ResidentScene::default();
        r.update(&placed_doc("#ff0000").to_string()).unwrap();
        let opts = RenderOptions::default();
        let (plan, w) = crate::frame::build_coverage_plan(r.scene().unwrap(), 0, &opts).unwrap();
        assert!(w.dropped.is_empty(), "{w:?}");
        let bodies: Vec<(&str, f64, &crate::seams::TessellatedGeometry)> = plan
            .items
            .iter()
            .filter_map(|i| match i {
                crate::seams::DrawItem::Body {
                    node_id,
                    z,
                    geometry,
                    ..
                } => Some((node_id.as_str(), *z, &**geometry)),
                _ => None,
            })
            .collect();
        // "a": its fill body (no stroke vertices), then the warped artwork above it.
        let a = bodies.iter().find(|b| b.0 == "a").expect("fill body");
        assert!(
            a.2.vertex_types.iter().all(|t| *t == 0.0),
            "the brush replaces the stroke band"
        );
        let art: Vec<_> = bodies
            .iter()
            .filter(|b| b.0.starts_with("a::brush::"))
            .collect();
        assert_eq!(art.len(), 1, "stretch: one warped copy along the path");
        assert!(art.iter().all(|b| b.1 > a.1 && b.1 < a.1 + 0.5));
        let dots = bodies
            .iter()
            .filter(|b| b.0.starts_with("s::brush::"))
            .count();
        assert!(dots >= 10, "{dots} scatter copies");
        assert!(bodies.iter().all(|b| b.0 != "s"
            || b.2.vertex_count == 0
            || b.2.vertex_types.iter().all(|t| *t == 0.0)));

        // A brush-table edit re-keys the art stroke only.
        let h0 = r.hashes().unwrap().clone();
        r.update(&placed_doc("#00ff00").to_string()).unwrap();
        let h1 = r.hashes().unwrap();
        assert_ne!(h0["a"].full, h1["a"].full);
        assert_eq!(h0["s"].full, h1["s"].full);
    }

    #[test]
    fn refused_placements_draw_the_plain_stroke_with_a_note() {
        let mut doc = placed_doc("#ff0000");
        // ~30,000 copies of a 10 px dot: past the 10,000-copy budget.
        doc["nodes"][1]["points"] =
            json!([{"type":"move","x":0,"y":0},{"type":"line","x":300000,"y":0}]);
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let (plan, w) =
            crate::frame::build_coverage_plan(r.scene().unwrap(), 0, &RenderOptions::default())
                .unwrap();
        assert!(
            w.skipped
                .iter()
                .any(|m| m.starts_with("s (brush)") && m.contains("too complex")),
            "{w:?}"
        );
        let s_items: Vec<_> = plan
            .items
            .iter()
            .filter_map(|i| match i {
                crate::seams::DrawItem::Body {
                    node_id, geometry, ..
                } if node_id.starts_with('s') => Some((node_id.clone(), geometry.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(s_items.len(), 1, "no placed copies");
        assert!(
            s_items[0].1.vertex_types.iter().any(|t| *t == 1.0),
            "the plain stroke band"
        );
    }

    fn placed_fill_alphas(plan: &CoveragePlan, prefix: &str) -> Vec<f32> {
        plan.items
            .iter()
            .filter_map(|i| match i {
                crate::seams::DrawItem::Body {
                    node_id, instance, ..
                } if node_id.starts_with(prefix) => Some(instance.fill_color[3]),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn paint_only_edits_of_a_brushed_stroke_redraw_without_re_placing() {
        let mut r = ResidentScene::default();
        let mut doc = placed_doc("#ff0000");
        r.update(&doc.to_string()).unwrap();
        let mut caches = PlanCaches::default();
        let opts = RenderOptions::default();
        let build = |r: &ResidentScene, caches: &mut PlanCaches| {
            let scene = r.scene().unwrap();
            let hashes = r.hashes().unwrap();
            crate::frame::build_coverage_plan_cached(scene, 0, &opts, Some((caches, hashes)))
                .unwrap()
                .0
        };
        let p1 = build(&r, &mut caches);
        assert!(placed_fill_alphas(&p1, "s::brush::")
            .iter()
            .all(|a| (*a - 1.0).abs() < 1e-6));
        let placed_before = caches.placed.get("s").map(|e| Arc::as_ptr(&e.placed));
        doc["nodes"][1]["opacity"] = json!(0.25);
        doc["nodes"][1]["blendMode"] = json!("multiply");
        r.update(&doc.to_string()).unwrap();
        let p2 = build(&r, &mut caches);
        let alphas = placed_fill_alphas(&p2, "s::brush::");
        assert!(
            !alphas.is_empty() && alphas.iter().all(|a| (*a - 0.25).abs() < 1e-6),
            "{alphas:?}"
        );
        assert_eq!(
            caches.placed.get("s").map(|e| Arc::as_ptr(&e.placed)),
            placed_before,
            "opacity / blend edits reuse the placement"
        );
    }

    #[test]
    fn a_placed_shape_the_fill_refuses_falls_back_to_the_plain_stroke() {
        // A tight zig-zag art brush (6-unit pitch) stretched along a long path:
        // one warped shape with thousands of crossings.
        let mut pts = vec![json!({"type":"move","x":0,"y":0})];
        for i in 1..400 {
            pts.push(json!({"type":"line","x": i as f64 * 6.0, "y": if i % 2 == 0 { 0.0 } else { 20.0 }}));
        }
        pts.push(json!({"type":"line","x":0,"y":10}));
        let mut doc = placed_doc("#ff0000");
        doc["brushes"][0]["artwork"]["nodes"][0]["points"] = json!(pts);
        doc["nodes"][0]["points"] = json!([{"type":"move","x":0,"y":0},{"type":"curve","x":300,"y":0,
            "cp1":{"x":100,"y":-300},"cp2":{"x":200,"y":300}}]);
        doc["nodes"][0]["strokes"][0]["width"] = json!(8.0);
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        // The old 50,000-vertex cap (#2151 raised the real one), so the zig-zag still trips it.
        let (plan, w) = crate::tess::with_fill_vertex_cap(50_000, || {
            crate::frame::build_coverage_plan(r.scene().unwrap(), 0, &RenderOptions::default())
        })
        .unwrap();
        assert!(w.dropped.is_empty(), "never a page refusal: {w:?}");
        eprintln!("zigzag notes: {:?}", w.skipped);
        assert!(w.skipped.iter().any(|m| m.contains("too complex")), "{w:?}");
        assert!(placed_fill_alphas(&plan, "a::brush::").is_empty());
    }

    fn placed_x_extent(plan: &CoveragePlan, prefix: &str) -> (usize, f32) {
        let mut n = 0;
        let mut max_x = f32::MIN;
        for i in &plan.items {
            if let crate::seams::DrawItem::Body {
                node_id, geometry, ..
            } = i
            {
                if node_id.starts_with(prefix) {
                    n += 1;
                    for p in geometry.positions.chunks(2) {
                        max_x = max_x.max(p[0]);
                    }
                }
            }
        }
        (n, max_x)
    }

    #[test]
    fn trim_and_dash_on_placed_brushes() {
        let render = |mut doc: Value, edit: &dyn Fn(&mut Value)| {
            edit(&mut doc);
            let mut r = ResidentScene::default();
            r.update(&doc.to_string()).unwrap();
            let (plan, w) =
                crate::frame::build_coverage_plan(r.scene().unwrap(), 0, &RenderOptions::default())
                    .unwrap();
            assert!(w.dropped.is_empty(), "{w:?}");
            assert!(!w.skipped.iter().any(|m| m.contains("not drawn")), "{w:?}");
            plan
        };
        let doc = placed_doc("#ff0000");
        // Art "a" (a triangle stretched along an L path): trimming to the first
        // half reveals only the part of the art on that half.
        let full = render(doc.clone(), &|_| {});
        let half = render(doc.clone(), &|d| d["nodes"][0]["trimPathEnd"] = json!(0.3));
        let (n_full, x_full) = placed_x_extent(&full, "a::brush::");
        let (n_half, x_half) = placed_x_extent(&half, "a::brush::");
        assert!(n_full >= 1 && n_half >= 1);
        assert!(
            x_half < x_full - 50.0,
            "revealed art stops early: {x_half} vs {x_full}"
        );
        // Scatter "s" (dots every 10 units along 300, centred at 0, 10, …):
        // a [20, 20] dash at width 1 is 10 on / 10 off, so every copy sits on
        // a dash boundary; windows are half-open, so each belongs to the dash
        // starting there → the copies at 0, 20, 40, … (about half).
        let s_full = placed_x_extent(&full, "s::brush::").0;
        let s_dash = placed_x_extent(
            &render(doc.clone(), &|d| {
                d["nodes"][1]["strokes"][0]["dash"] = json!([20, 20])
            }),
            "s::brush::",
        )
        .0;
        assert!(
            s_dash > 0 && s_dash < s_full * 3 / 4,
            "{s_dash} of {s_full}"
        );
        // 31 copies (0 … 300), 15 dashes: the end copy at 300 sits in the last
        // gap [290, 300) — it is the path's end, not its start.
        assert_eq!(s_dash, s_full / 2, "one copy per dash");
        // An animated trim on scatter doesn't re-place: filtered per frame.
        let s_trim = placed_x_extent(
            &render(doc, &|d| d["nodes"][1]["trimPathEnd"] = json!(0.25)),
            "s::brush::",
        )
        .0;
        assert!(s_trim > 0 && s_trim < s_full / 2);
    }

    #[test]
    fn pattern_brushes_draw_tiles_with_corners() {
        let mut doc = placed_doc("#ff0000");
        doc["brushes"].as_array_mut().unwrap().push(json!({
            "id": "rope", "name": "Rope", "type": "pattern", "scale": {"value": 1}, "spacing": 0, "fit": "stretch",
            "colorize": {"method": "none"},
            "tiles": {
                "side": {"nodes": [{"id": "band", "type": "rectangle", "x": 0, "y": 0, "width": 20, "height": 10, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#e9d8a6"}}]}]},
                "outerCorner": {"nodes": [{"id": "sq", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#005f73"}}]}]}
            }
        }));
        doc["nodes"][0]["strokes"][0]["brush"] = json!({"brushId": "rope"});
        let mut r = ResidentScene::default();
        r.update(&doc.to_string()).unwrap();
        let (plan, w) =
            crate::frame::build_coverage_plan(r.scene().unwrap(), 0, &RenderOptions::default())
                .unwrap();
        assert!(w.dropped.is_empty(), "{w:?}");
        let (n, _) = placed_x_extent(&plan, "a::brush::");
        assert!(n > 10, "side tiles + corner tiles: {n}");
    }

    #[test]
    fn a_cache_clear_keeps_worker_mode() {
        let mut caches = PlanCaches::default();
        caches.set_outline_worker(true);
        caches.outline_request_failed("x");
        caches.clear();
        assert!(
            caches.outline_worker,
            "worker mode is host config, not cache data"
        );
        assert!(caches.outline_no_worker.contains("x"));
        assert!(caches.outline_requested.is_empty());
    }

    #[test]
    fn a_stale_placement_is_replaced_within_a_few_frames() {
        // Budget 0: each frame's first take is granted (spent here by a bigger
        // stroke); every later placement is served stale — but only for
        // PLACED_MAX_STALE_FRAMES frames, then re-placed over budget.
        let mut c = PlanCaches::default();
        c.placed_put("a", 1, 7, Arc::new(crate::tess::placed::Placed::default()));
        let bbox = [0.0, 0.0, 10.0, 10.0];
        let mut stale = 0;
        for frame in 0..10 {
            c.begin_frame();
            c.set_outline_refine_budget_ms(Some(0.0));
            assert!(c.outline_budget_take(), "frame {frame}: first take");
            if matches!(c.placed_get("a", 2 + frame, 7, bbox, false), PlacedLookup::Hit(_)) {
                stale += 1;
            } else {
                // Re-placed now (the trim moved on): a fresh entry.
                c.placed_put(
                    "a",
                    2 + frame,
                    7,
                    Arc::new(crate::tess::placed::Placed::default()),
                );
                assert_eq!(stale, PLACED_MAX_STALE_FRAMES, "frame {frame}");
                stale = 0;
            }
        }
    }
}

#[cfg(test)]
mod unit_placement_tests {
    use super::*;
    use moda_scene::value::json;

    fn path(id: &str, x: f64, y: f64, color: &str) -> Value {
        json!({"id": id, "type": "path", "x": x, "y": y, "width": 70, "height": 60, "closed": true,
            "points": [{"type":"move","x":0,"y":0},{"type":"line","x":70,"y":0},{"type":"line","x":70,"y":60},{"type":"line","x":0,"y":60}],
            "fills": [{"id":"fill","paint":{"type":"solid","color":color}}]})
    }

    struct Host { resident: ResidentScene, caches: PlanCaches, index: FramePlanIndex, opts: RenderOptions, generation: u64 }
    impl Host {
        fn new(nodes: Vec<Value>, roots: &[&str]) -> Self {
            let source = json!({"nodes": nodes, "pages": [{"id":"page","width":320,"height":240,"nodeIds":roots,
                "fills":[{"id":"fill","paint":{"type":"solid","color":"#eeeeee"}}]}],
                "config":{"renderPageClipping":true},"layoutMode":"freeform","canvasRootNodeIds":[]});
            let mut resident = ResidentScene::default();
            resident.update(&source.to_string()).unwrap();
            Self { resident, caches: Default::default(), index: Default::default(),
                opts: RenderOptions { sample_count: 4, transparent: false, ..Default::default() }, generation: 0 }
        }
        /// Moves `id` to `x` and plans as a canvas does: follow recipes, else plan whole and seed. Returns the
        /// plan and whether recipes were followed.
        fn step(&mut self, id: Option<(&str, f64)>) -> (Arc<CoveragePlan>, bool) {
            let mut ids = Vec::new();
            if let Some((id, x)) = id {
                self.resident.apply_props(&json!({"nodes": {id: {"x": x}}}).to_string()).unwrap();
                ids.push(id.to_string());
            }
            self.generation += 1;
            self.caches.begin_frame();
            // A host keeps the scene's world columns current, as a canvas does before it plans.
            self.resident.state.as_mut().unwrap().scene.refresh_worlds();
            let scene = self.resident.scene().unwrap();
            if !ids.is_empty() {
                let rows: Vec<u32> = ids.iter().filter_map(|id| scene.row_of(id)).collect();
                if let Some((plan, _)) = self.index.place(scene, &self.opts, self.generation, self.generation, &rows, &mut self.caches) {
                    self.index.take_written();
                    return (plan, true);
                }
            }
            let (plan, warnings, aux) = crate::frame::build_coverage_plan_indexed(scene, 0, &self.opts,
                Some((&mut self.caches, self.resident.hashes().unwrap()))).unwrap();
            let (plan, warnings) = (Arc::new(plan), Arc::new(warnings));
            self.index.seed(scene, &self.opts, &plan, &warnings, aux, self.generation, self.generation, &mut self.caches);
            (plan, false)
        }
        fn fresh(&self) -> String {
            let mut caches = PlanCaches::default();
            let (plan, ..) = crate::frame::build_coverage_plan_indexed(self.resident.scene().unwrap(), 0, &self.opts,
                Some((&mut caches, self.resident.hashes().unwrap()))).unwrap();
            format!("{plan:?}")
        }
    }

    #[test]
    fn shifted_span_cannot_write_another_nodes_items() {
        for scale in [1., 4.] {
            let mut host = Host::new(vec![path("free", 10., 10., "#255fe8"),
                path("other", 100., 10., "#ff7700")], &["free", "other"]);
            host.step(None);
            // Model a capture cut-out leaving a stale span aimed at the next node.
            let entry = host.index.entry.as_mut().unwrap();
            let spans = &mut entry.patchable.as_mut().unwrap().index.spans;
            let other_start = spans.iter().find(|s| s.node_id == "other").unwrap().start;
            spans.iter_mut().find(|s| s.node_id == "free").unwrap().start = other_start;
            host.resident.apply_props(&json!({"nodes":{"free":{"x":20,"scaleX":scale}}}).to_string()).unwrap();
            host.resident.state.as_mut().unwrap().scene.refresh_worlds();
            assert!(host.index.patch(host.resident.scene().unwrap(), host.resident.hashes().unwrap(),
                &host.opts, 2, 2, vec!["free".into()], &mut host.caches).is_none());
        }
    }

    /// A scaled path can decline its recipe without invalidating a neighbouring capture. The same
    /// dependency gate must refuse the capture, its descendants and ancestors, even when it is culled.
    #[test]
    fn node_preparation_beside_units_matches_fresh_planning() {
        for effect in ["layer-blur", "background-blur"] {
            for unit_x in [100., -500.] {
                let nodes = vec![path("free", 10., 10., "#255fe8"),
                    json!({"id":"above","type":"group","children":["unit"]}),
                    json!({"id":"unit","type":"container","x":unit_x,"y":80,"width":100,"height":100,
                        "children":["inside"],"effects":[{"id":"e","type":effect,"radius":5}]}),
                    path("inside", 5., 5., "#ff7700")];
                for mover in ["free", "unit", "above", "inside"] {
                    let mut host = Host::new(nodes.clone(), &["free", "above"]);
                    host.step(None);
                    for scale in [4., 0.25, 8.] {
                        host.resident.apply_props(&json!({"nodes":{mover:{"scaleX":scale,"scaleY":0.8}}}).to_string()).unwrap();
                        host.generation += 1;
                        host.resident.state.as_mut().unwrap().scene.refresh_worlds();
                        let scene = host.resident.scene().unwrap();
                        let patched = host.index.patch(scene, host.resident.hashes().unwrap(), &host.opts,
                            host.generation, host.generation, vec![mover.into()], &mut host.caches);
                        if mover == "free" {
                            let (plan, _) = patched.expect("an unrelated node can prepare beside a unit");
                            let written = host.index.take_written();
                            assert!(!written.replaced.is_empty(), "scale must exercise node preparation");
                            assert_eq!(format!("{plan:?}"), host.fresh());
                        } else {
                            assert!(patched.is_none(), "{mover} feeds a unit and must plan whole");
                            break;
                        }
                    }
                }
            }
        }
    }

    /// A background effect that starts outside the target emits no split and, on an unpainted container, no
    /// span. Moving it or a node above it into view must plan whole; unrelated paths still follow recipes.
    #[test]
    fn a_unit_that_draws_nothing_yet_still_declines_placement_when_it_or_its_ancestor_moves() {
        let nodes = || vec![path("before", 25., 70., "#e02a40"), json!({"id":"ancestor","type":"group","children":["unit"]}),
            json!({"id":"unit","type":"container","x":-500,"y":75,"width":100,"height":100,"children":[],
                "effects":[{"id":"e","type":"background-blur","radius":12}]}),
            path("after", 190., 85., "#255fe8")];
        let splits = |plan: &CoveragePlan| plan.items.iter().filter(|item| matches!(item, crate::seams::DrawItem::BackdropSplit(_))).count();
        for mover in ["unit", "ancestor"] {
            let mut host = Host::new(nodes(), &["before", "ancestor", "after"]);
            let (first, _) = host.step(None);
            assert_eq!(splits(&first), 0, "the effect starts outside the target");
            let (plan, placed) = host.step(Some(("after", 200.)));
            assert!(placed, "an unrelated path follows its recipe beside the unit");
            assert_eq!(format!("{:?}", *plan), host.fresh());
            let (plan, placed) = host.step(Some((mover, if mover == "unit" { 20. } else { 600. })));
            assert!(!placed, "moving {mover} plans whole");
            assert_eq!(splits(&plan), 1, "the effect is in view after moving {mover}");
            assert_eq!(format!("{:?}", *plan), host.fresh());
            let (plan, placed) = host.step(Some(("before", 40.)));
            assert!(placed);
            assert_eq!(format!("{:?}", *plan), host.fresh());
        }
    }

    /// The guard asks the changed nodes, not every unit: its cost does not grow with units that stay put.
    #[test]
    fn many_stationary_units_do_not_block_or_slow_unrelated_placements() {
        let mut nodes = vec![path("free", 10., 10., "#255fe8")];
        let mut roots = vec!["free".to_string()];
        for i in 0..64 {
            let id = format!("unit{i}");
            let mut unit = path(&id, (i % 8) as f64 * 30., 80. + (i / 8) as f64 * 15., "#ffffff70");
            unit["effects"] = json!([{"id":"e","type":"layer-blur","radius":2}]);
            nodes.push(unit);
            roots.push(id);
        }
        let roots: Vec<&str> = roots.iter().map(String::as_str).collect();
        let mut host = Host::new(nodes, &roots);
        host.step(None);
        for x in [20., 35., 50.] {
            let (plan, placed) = host.step(Some(("free", x)));
            assert!(placed);
            assert_eq!(format!("{:?}", *plan), host.fresh());
        }
        let (plan, placed) = host.step(Some(("unit3", 5.)));
        assert!(!placed, "a unit that moves plans whole");
        assert_eq!(format!("{:?}", *plan), host.fresh());
    }
}
