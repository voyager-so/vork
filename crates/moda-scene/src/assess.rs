//! `assess()` — the feature-manifest walker (PLAN.md §3, schema frozen at
//! M0c). A PURE document walk: no GPU, no asset fetches. Computed from the
//! same predicates the browser uses for its own routing (DECISION.md §1.6):
//!
//! - MSDF text gate  = `isMsdfFamily` (`core/config/msdfFonts.ts:27-31`) +
//!   the codepoint coverage scan concept (`utils/msdf/coverageScanner.ts:69-84`).
//!   Without atlas metrics (an asset fetch), coverage is approximated by
//!   script-class detection — the same fallback classes DECISION §1.4 names
//!   (CJK / Greek / Cyrillic / Arabic / non-catalog families).
//! - chart gate      = unregistered `spec.type` only. bar/line/pie/scatter/
//!   area/combo generators are ported (`crate::charts`, incl. time/date X
//!   axes); an unported spec type gates with a `spec_type` detail
//!   (`chart_gate_reason`). (Was "ALL chart types gated until M4" at M0c — no
//!   longer true.)
//! - shader gate     = extraction-manifest id registry incl. aliases
//!   (`core/shaders/shaderIdAliases.ts:1-18`); a built-in default registry is
//!   compiled from the production shader modules and can be replaced by the
//!   M0a extraction manifest at runtime.
//!
//! Verdicts: Full | Degraded{expected_ssim_band} | Unsupported | Unknown.
//! `unknown` (permissive-serde spillover) is distinct from gated.

use crate::scene::Scene;
use crate::types::{
    AxisEntry, AxisStatus, FontCatalog, Manifest, ManifestWarning, NodeId, PageManifest, RawNode,
};
use crate::value::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// Shader-id registry: canonical ids + legacy alias map. The default set is
/// compiled from `renderers/webgpu/shaders/{mesh,light,pattern}/*` module
/// registrations (49 modules; was 35 at DECISION.md S9). Every production
/// assess site (moda-cli, moda-py, moda-wasm) now wires
/// `moda_engine::extracted_shader_registry()` — the same sets derived from the
/// GENERATED moda-shaders extraction manifest — so a new shader landing via
/// extraction can never render live yet assess Unknown server-side. This
/// hand-maintained default survives only for library tests and is pinned to
/// the manifest by moda-engine's `default_registry_matches_extraction_manifest`
/// tripwire (delete-const follow-up tracked there).
#[derive(Debug, Clone)]
pub struct ShaderRegistry {
    pub known_ids: HashSet<String>,
    pub aliases: BTreeMap<String, String>,
    /// Image-backed fills gated until M4 (the set is now empty).
    pub gated_ids: HashSet<String>,
    /// MSDF text-fill INELIGIBLE ids — the browser predicate
    /// `isTextShaderFillEligible` (descriptor `shade` shader with no
    /// `primaryImage` resource, `defineProceduralShader.ts:326-328`) fails
    /// only for image-backed shaders among the 49 registered shaders (all
    /// 49 are `shade`). Ineligible shader ids on a richtext fill render
    /// per-run colors in the browser (`WebGPUTextSystem.ts:599-621`).
    pub text_fill_ineligible_ids: HashSet<String>,
}

impl Default for ShaderRegistry {
    fn default() -> Self {
        const IDS: [&str; 49] = [
            "analog-grain-gradient",
            "band-dither-panels",
            "center-wave-bars",
            "clouds",
            "cmyk-halftone-press",
            "cross-hatching-image",
            "diagonal-panel-warp",
            "dither-image",
            "dithered-wave-ribbon",
            "glass-tile-glow",
            "gradient-map-image",
            "grid-displacement-image",
            "gooey-halftone-cells",
            "halftone-cell-wall",
            "halftone-image",
            "heatmap-image",
            "hex-panel-warp",
            "lens-distortion-image",
            "light-rays",
            "light-rays-advanced",
            "liquid-metal-image",
            "luminous-sine-veils",
            "mesh-glass-rods",
            "mesh-gradient",
            "mesh-gradient-advanced",
            "mesh-grid",
            "mesh-rbf",
            "metaball-field",
            "metal-plates",
            "opening-gates",
            "perspective-grid",
            "pixel-stretch-image",
            "plasma-field",
            "prismatic-swirl-panels",
            "ring-warp-glass",
            "segment-display-field",
            "shape-grid",
            "shape-tiles",
            "simplex-blob",
            "slice-shift-image",
            "smooth-sine-lines",
            "soft-glass-panels",
            "soft-pixel-field",
            "soft-voronoi",
            "stars",
            "swirl-edge-bands-gradient",
            "swirl-shape-gradient",
            "swirl-stripe-gradient",
            "warped-grid",
        ];
        let mut aliases = BTreeMap::new();
        // LEGACY_SHADER_ID_ALIASES (core/shaders/shaderIdAliases.ts:1-9)
        aliases.insert("mesh-spot".into(),
            "mesh-gradient-advanced".to_string(),
        );
        aliases.insert("mesh-spot-simple".into(), "mesh-gradient".to_string());
        aliases.insert("shape-grid-dots".into(), "shape-grid".to_string());
        aliases.insert("shape-grid-tiles".into(), "shape-tiles".to_string());
        // Image-backed shaders are UN-GATED — the `imageProcessing.ts` port +
        // shader-image store landed in M4e. `gated_ids` stays as a mechanism
        // for future gates.
        ShaderRegistry {
            known_ids: IDS.iter().map(|s| s.to_string()).collect(),
            aliases,
            gated_ids: HashSet::new(),
            text_fill_ineligible_ids: [
                "cross-hatching-image",
                "dither-image",
                "gradient-map-image",
                "grid-displacement-image",
                "halftone-image",
                "heatmap-image",
                "lens-distortion-image",
                "liquid-metal-image",
                "pixel-stretch-image",
                "slice-shift-image",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        }
    }
}

impl ShaderRegistry {
    /// `resolveShaderId` (shaderIdAliases.ts:13-17).
    pub fn resolve<'a>(&'a self, id: &'a str) -> &'a str {
        self.aliases.get(id).map(String::as_str).unwrap_or(id)
    }

    /// `ShaderManager.supportsTextShaderFill` (ShaderManager.ts:181-184) for
    /// a RESOLVED id: known + not in the ineligible (image-backed) set.
    pub fn supports_text_fill(&self, resolved_id: &str) -> bool {
        self.known_ids.contains(resolved_id) && !self.text_fill_ineligible_ids.contains(resolved_id)
    }
}

/// Inputs the walker may use. Everything optional — absent inputs degrade to
/// conservative verdicts, never to silent Full.
#[derive(Default)]
pub struct AssessOptions {
    pub canvas_id: Option<String>,
    pub engine_version: Option<String>,
    pub shader_registry: Option<ShaderRegistry>,
    /// Font catalog from the headless response (`fonts`). When absent, every
    /// non-default family is treated as non-MSDF (degraded), matching the
    /// browser's behavior with an empty catalog (`findFontByName` miss ⇒
    /// `isMsdfFamily` false, msdfFonts.ts:27-31).
    pub font_catalog: Option<FontCatalog>,
    /// Task #72 (`--input` + vendored-catalog mode only): engine-truth probe
    /// returning the vendored-VARIANT gaps one html blob hits at render time
    /// — catalog-promised (family, weight, style) keys the render host has no
    /// atlas bytes for (`moda_text::vendored_variant_gaps`). Non-empty ⇒ the
    /// node's runs render via the cosmic Degraded tier (fallback face) while
    /// the browser (same catalog) renders the real variant via MSDF: surfaced
    /// as `text:variant-gap`, Degraded — a FIDELITY loss, not content loss
    /// (build_batch routes variant-gapped runs through the degraded raster
    /// unconditionally; before that fix the strict lane silently DROPPED
    /// them). None (the `--fonts` / `--canvas-id` modes) skips the check —
    /// materialization fetches the bytes, so there is no gap to surface.
    pub variant_gap_probe: Option<Box<dyn Fn(&str) -> Vec<String>>>,
    /// Task #128 (`--text-engine hybrid` only): engine-truth probe answering
    /// "would the hybrid's vector arm serve this html blob?" — the CLI wires
    /// `moda_text::build_node_ops` (fallback-class? eligible reason?) +
    /// `moda_text::vector::build_vector_node_ops` (does the vector lane
    /// resolve every run — TTF present, per-script fallback families in the
    /// catalog, no shaping throw e.g. the Q-105/Q-118 Arabic tripwire?).
    /// `true` means the vector arm serves the node — the exact engine the GA
    /// default lane ships as verdict FULL — so the family/script degrades are
    /// SUPPRESSED (no degraded axis; parity with `text_engine_is_vector`) and
    /// a `text:fallback-engine:vector` ManifestWarning carries the
    /// families/scripts detail for hybrid-lane diagnostics. `false`/absent
    /// keeps the `text:fallback-engine` degrade (0.90-0.97).
    /// None on the msdf/vector engines — the warning must not appear when the
    /// hybrid arm cannot fire (Q-115 engine-truth injection precedent).
    pub hybrid_vector_probe: Option<Box<dyn Fn(&str) -> bool>>,
    /// The render lane uses the VECTOR text engine (the GA default, matching the
    /// in-browser WASM renderer). Vector draws any font's glyph outlines directly
    /// — a family with a TTF but no MSDF atlas (custom uploads included) renders
    /// identically to the browser — so a font's MSDF-atlas readiness / catalog
    /// membership is IRRELEVANT to whether the text renders. When set, the
    /// FONT-FAMILY half of the `text:fallback-engine` degrade (an MSDF-era
    /// artifact) is suppressed: it could only FALSE-decline renderable text.
    /// MSDF/hybrid keep it (they need a per-font atlas and diverge on
    /// non-vendored families). Does NOT suppress the non-Latin-SCRIPT or
    /// emoji-coverage halves: those are content-coverage divergences (the native
    /// renderer only has the catalog's fonts, and an unserviceable run is skipped
    /// at render — the output ships text-missing, not a fail-open), so they keep
    /// degrading under every engine. Default false (the conservative MSDF-era
    /// behavior for tests / internal callers).
    pub text_engine_is_vector: bool,
}

/// Per-page axis accumulator.
struct AxisAcc {
    status: AxisStatus,
    node_ids: Vec<NodeId>,
    detail: Map,
}

struct PageAcc {
    axes: BTreeMap<String, AxisAcc>,
    warnings: Vec<ManifestWarning>,
}

impl PageAcc {
    fn new() -> Self {
        PageAcc {
            axes: BTreeMap::new(),
            warnings: Vec::new(),
        }
    }
    fn add(&mut self, axis: &str, status: AxisStatus, node_id: &str, detail: Map) {
        let entry = self
            .axes
            .entry(axis.to_string())
            .or_insert_with(|| AxisAcc {
                status: status.clone(),
                node_ids: Vec::new(),
                detail,
            });
        if status.severity() > entry.status.severity() {
            entry.status = status;
        }
        if !node_id.is_empty() && !entry.node_ids.contains(&node_id.to_string()) {
            entry.node_ids.push(node_id.to_string());
        }
    }
}

/// Axes that record a schema-novelty signal but MUST NOT elevate the page/doc
/// verdict (so they never decline a doc to the browser fallback).
///
/// `unknown:node-field` fires when a NODE carries a key the assessor's per-type
/// field tables (`fields.rs`) don't list. The `tests/scene_schema_parity.rs` CI
/// gate parses `core/scene/nodes.ts` and asserts those tables mirror EVERY field
/// on every `SceneNode` variant (its ignore-list is empty), so a key that spills
/// HERE is provably NON-canonical — a foreign / legacy / agent-lane key that is
/// not part of the render schema. Neither renderer reads such a key: the browser
/// WebGPU/WASM golden reads only the typed `nodes.ts` field surface (verified —
/// e.g. `fillColor` exists only as an internal GPU uniform computed from `fill`,
/// never read off a node; `parentId` is derived from the scene graph, never read
/// off a node). So the native raster is pixel-identical to the browser's, and
/// declining would burn a Chromium render for ZERO fidelity gain. We still EMIT
/// the axis (drift observability), it just no longer gates.
///
/// `unknown:envelope-field` is NON-gating too, for the same parity reason — the
/// gate hole the previous note called out is now CLOSED. `tests/scene_schema_parity.rs`
/// asserts `ENVELOPE_KEYS` is a superset of BOTH authoritative top-level producers:
/// `projection.py` `META_KEYS` (the collab round-trip) AND the frontend serializer's
/// own `getSceneState()` return-type key set (`scene_state_serializer_keys_are_covered…`).
/// Together those are the complete set of CANONICAL top-level `canvas_data` keys, so a
/// key that spills HERE is provably produced by NO canonical serializer — a foreign /
/// legacy / agent-lane key (e.g. the PPTX `metadata` blob) that the browser's own scene
/// loader ignores exactly as the native `SceneDocument` parser does (both drop unknown
/// top-level keys — the parser parks them in `unknown_envelope_keys`, never read by plan
/// building). So the native raster is pixel-identical to the browser's and declining would
/// burn a Chromium render for ZERO fidelity — and, once the browser renderer retires, would
/// leave the doc NOWHERE to render. This was the top REMAINING `manifest_unsupported` cause
/// in prod once `unknown:node-field` stopped gating: `schemaVersion` alone was ~99% of the
/// envelope-field declines (it kept being re-declined until each new benign top-level key
/// was hand-added to `ENVELOPE_KEYS`). We still EMIT the axis (drift observability), it just
/// no longer gates — and the parity gate now FAILS CI if a genuinely render-affecting
/// top-level key is ever added to the serializer without teaching the native parser.
///
/// `effects:shader-effects` is ALSO non-gating (parity-safe by the same
/// argument). It fires for `shader` entries of `effects` that are NOT the one
/// renderable source-mask effect — post-process-mode entries and extra
/// source-mask entries beyond the first. The browser reads
/// ONLY the first renderable source-mask effect (`getRenderableSourceMaskShaderEffect`
/// → `.find(...)`, `WebGPUPathRenderer.ts`) and never touches the others, so
/// every entry this axis flags renders NOTHING in the browser; the native
/// engine's `effective_shader_fill` mirrors that exactly (the flagged entries
/// render nothing here too, the node renders its underlying fill / the first
/// source-mask effect). So the native raster is pixel-identical to the browser's
/// and declining would burn a Chromium render for ZERO fidelity — and, once the
/// browser renderer is retired, would leave the doc with NOWHERE to render. The
/// first source-mask effect's own shader-id is still recorded on the separate
/// `shader:<id>` axis; an UNKNOWN one no longer gates either (see the
/// `shader:*` note below), it serves as a white box exactly as the browser would
/// with the same engine. We still EMIT this axis (drift observability — how many
/// docs carry non-source-mask entries), it just no longer gates. RE-GATE
/// TRIGGER: this exclusion is only correct while
/// those extra modes render NOTHING in either engine — whoever implements a
/// currently-inert mode (e.g. post-process) must revisit it in the same PR; see
/// the shader-effects row in
/// `docs/architecture/feature-implementation-playbooks.md` §1.3.
///
/// `unknown:node-type` is NON-gating for the same parity reason. It fires for a
/// node whose `type` is outside `VALID_NODE_TYPES` (and not a KNOWN legacy drop
/// like `text`/`image`): `hydrate_document` drops the node at hydration
/// (`scene.rs`, `continue`) exactly as the browser's `Scene.ts` hydration does
/// (`isValidNodeType` filters at Scene.ts:693/1278/2090 — the node is silently
/// skipped and never reaches either renderer). `tests/scene_schema_parity.rs`
/// pins the TS `SceneNode` roster equal to the Rust roster ("SceneNode node-type
/// roster drifted"), so a type that spills HERE is provably not in the shipped
/// TS union either — a foreign / legacy / agent-lane type NEITHER renderer
/// draws. The native raster is therefore pixel-identical to the browser's
/// (both render the doc without the node), declining would burn a Chromium
/// render for ZERO fidelity — and, post browser-renderer retirement, would
/// leave the doc NOWHERE to render. We still EMIT the axis with the
/// `node_type` detail (drift observability: a genuinely NEW 13th node type
/// shows up as a prod rate spike on a new detail value during the
/// frontend/backend deploy-skew window — only a minutes-scale deploy-ordering
/// window remains, since the roster parity gate blocks a merge that doesn't
/// teach the Rust engine), it just no longer gates.
///
/// An unknown `shader:<id>` (status `Unknown`) is ALSO non-gating.
/// It was long treated as "something the engine cannot draw but the browser
/// CAN" — but that stopped being true once the browser began rendering with the
/// SAME `moda-engine` (one-engine-everywhere): the browser draws the identical
/// white box, so declining bought byte-identical pixels for a wasted Chromium
/// render (prod: 0 shader declines in 90d). The extraction-parity gate
/// (`extractor-drift` CI + `default_registry_matches_extraction_manifest`) makes
/// a browser-only shader structurally unmergeable, so the only residual is a
/// minutes-scale frontend/backend deploy-skew window — the same class as
/// `unknown:node-type`. The `native_render.shader_drift_axis` WARNING (router
/// `_summarize_manifest`/`_Decision.emit`) is retained as the sole signal for
/// this now-served condition. Only a *gated* KNOWN shader (status `Unsupported`,
/// `gated_ids`) still gates — a deliberate native-capability gate, matched here
/// by the `status`-aware check in `axis_elevates_verdict`.
/// (`unknown:generated-kind` is NOT in that class: the expansion draws a
/// browser-parity placeholder rect for it — expansion.rs
/// `create_placeholder_rect` — so it reports as Degraded, which still elevates
/// the verdict. Keeping it elevating (rather than Full) preserves the
/// fail-closed window for strict export consumers if a future 5th
/// `GeneratedGraphicKind` ever ships TS-first. `generated:chart` splits by
/// `chart_gate_reason`'s reason: unregistered types emit `Full` (identical
/// placeholder in both lanes) while raw-array-data emits `Unsupported` — that
/// reason now only fires for arrays the ported `rowsToParsedChartData`
/// coercion fails closed on (TS-throw shapes), but the browser is still the
/// safer lane there, so the axis's elevating classification is LOAD-BEARING;
/// never add it to the exclusion list below.)
///
/// WIRE CONTRACT: `manifest_to_json` stamps this predicate per-axis as the
/// `gating` bit, which is what the backend decision-record attribution reads.
/// Its `_NON_GATING_AXES` fallback set (router.py) deliberately mirrors the
/// exclusion list AS OF the last pre-`gating`-key engine (it omits
/// `unknown:node-type`, which flipped in the same change as the key) — when
/// flipping another axis here, do NOT add it to that fallback set.
fn axis_elevates_verdict(axis: &str, status: &AxisStatus) -> bool {
    // An UNRECOGNIZED `shader:<id>` (`AxisStatus::Unknown`) is non-gating: native
    // drops the node to a white box, but so does the browser — the browser
    // renders with the SAME `moda-engine` (one-engine-everywhere), so it cannot
    // draw a shader the native engine can't. Declining would burn a Chromium
    // render for byte-identical output (0 occurrences in 90d, and no
    // fidelity when it would fire). The `native_render.shader_drift_axis` WARNING
    // (router `_summarize_manifest`/`_Decision.emit`) remains the ONLY signal for
    // this condition — it now fires on the native SERVE, not a decline. A *gated*
    // KNOWN shader (`AxisStatus::Unsupported` via `gated_ids`) STILL gates: that
    // is a deliberate native-capability gate the browser genuinely used to serve,
    // not un-extracted-shader drift (and `renderable()` already excludes it).
    if axis.starts_with("shader:") && matches!(status, AxisStatus::Unknown) {
        return false;
    }
    !matches!(
        axis,
        "unknown:node-field"
            | "unknown:envelope-field"
            | "unknown:node-type"
            | "effects:shader-effects"
    )
}

/// The manifest walker. Walks the hydrated scene per page (flattened z-order
/// WITHOUT transient expansion — host nodes, not proxies, carry the verdicts)
/// and emits the PLAN §3 manifest.
pub fn assess(scene: &Scene, opts: &AssessOptions) -> Manifest {
    let registry = opts.shader_registry.clone().unwrap_or_default();
    let empty_catalog = FontCatalog::default();
    let catalog = opts.font_catalog.as_ref().unwrap_or(&empty_catalog);

    let mut pages = Vec::new();
    for (page_index, page) in scene.doc.pages.iter().enumerate() {
        let mut acc = PageAcc::new();

        // Page-level fill axes (pages carry the same fill surface).
        assess_fill_surface(
            &RawNode::new(page.0.clone()),
            &compute_bg_pseudo_id(page.id()),
            &registry,
            &mut acc,
        );

        // Parser side-channel signals land on page 0 only (document-scoped).
        if page_index == 0 {
            add_document_signals(scene, &mut acc);
        }

        // Walk the page's nodes (hosts, not expansion proxies).
        let order = scene.flattened_z_order_for_page_opts(page.id(), true, false);
        assess_node_order(scene, &order, &registry, catalog, opts, &mut acc);

        pages.push(finish_surface_manifest(page_index, page.id(), acc));
    }

    // Page-less freeform docs (legacy Freeform/Design): the canvas-root
    // floating nodes are the ONLY rendered content, served natively by the
    // camera-lane entry point (`moda_engine::render_canvas_root`). Assess them
    // as ONE synthetic surface (page_id = SCENE_ROOT_ID, index 0) so that lane
    // can never render unvetted features (the decline→500 incident class).
    // Scoped to `pages.is_empty()`: with pages present the server export lanes
    // render pages only, and floating nodes never reach their output.
    let mut canvas_root_frame_px: Option<(f64, f64)> = None;
    if scene.doc.pages.is_empty() && !scene.doc.canvas_root_node_ids.is_empty() {
        let mut acc = PageAcc::new();
        // Document-scoped parser signals normally land on page 0 — with no
        // pages, the canvas-root surface is the document's only manifest home.
        add_document_signals(scene, &mut acc);
        // Hosts, not expansion proxies — mirrors the page walk's opts.
        let order = scene.flattened_z_order_for_canvas_opts(true, false);
        assess_node_order(scene, &order, &registry, catalog, opts, &mut acc);
        pages.push(finish_surface_manifest(0, crate::types::SCENE_ROOT_ID, acc));
        // The authoritative frame size for this surface (content bounds +
        // margin) — the backend derives render caps from it instead of
        // re-deriving scene geometry.
        canvas_root_frame_px = crate::display_list::canvas_root_frame_size(scene);
    }

    let verdict = pages
        .iter()
        .map(|p| p.verdict.clone())
        .max_by_key(AxisStatus::severity)
        .unwrap_or(AxisStatus::Full);
    Manifest {
        renderer: "moda-render",
        engine_version: opts.engine_version.clone(),
        canvas_id: opts.canvas_id.clone(),
        verdict,
        pages,
        canvas_root_frame_px,
    }
}

fn compute_bg_pseudo_id(page_id: &str) -> String {
    crate::types::compute_page_bg_id(page_id)
}

/// Parser side-channel signals (document-scoped; emitted once per manifest —
/// on page 0, or on the synthetic canvas-root surface of a page-less doc).
/// BOTH axes here are renderable-informational and NON-gating (see
/// `axis_elevates_verdict`): the parity guards prove `ENVELOPE_KEYS` covers
/// every canonical top-level producer (projection `META_KEYS` + the
/// `getSceneState()` serializer) and the node-type roster mirrors the TS
/// `SceneNode` union, so a key/type spilling here is provably foreign and
/// dropped identically by both renderers at hydration.
fn add_document_signals(scene: &Scene, acc: &mut PageAcc) {
    for k in &scene.doc.unknown_envelope_keys {
        let mut d = Map::new();
        d.insert("field".into(), json!(k));
        d.insert("scope".into(), json!("envelope"));
        acc.add("unknown:envelope-field", AxisStatus::Unknown, "", d);
    }
    for (id, t) in &scene.doc.unknown_node_types {
        let mut d = Map::new();
        d.insert("node_type".into(), json!(t));
        acc.add("unknown:node-type", AxisStatus::Unknown, id, d);
    }
}

/// Assess one surface's flattened node order (a page's, or the freeform
/// canvas root's) into `acc`.
#[allow(clippy::too_many_arguments)]
fn assess_node_order(
    scene: &Scene,
    order: &[String],
    registry: &ShaderRegistry,
    catalog: &FontCatalog,
    opts: &AssessOptions,
    acc: &mut PageAcc,
) {
    // Q-28: active richtext clip-mask sources on this surface. Their
    // glyphs never render as TEXT in either renderer (shouldSkipRender,
    // WebGPUShapeSystem.ts:283-299), so they must not contribute
    // text:* axes — their pixel impact is the mask coverage
    // (mask:text-alpha carries the band), on every page.
    let mut richtext_mask_sources: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for id in order {
        let Some(node) = scene.node(id) else { continue };
        if node.node_type() == "group" {
            if let Some(mask_id) = node.str("clipMaskNodeId") {
                if scene
                    .node(mask_id)
                    .map(|m| m.node_type() == "richtext")
                    .unwrap_or(false)
                {
                    richtext_mask_sources.insert(mask_id.to_string());
                }
            }
        }
    }

    for id in order {
        if id.starts_with("__page_bg_") {
            continue;
        }
        let Some(node) = scene.node(id) else { continue };
        assess_node(
            scene,
            node,
            registry,
            catalog,
            opts.variant_gap_probe.as_deref(),
            opts.hybrid_vector_probe.as_deref(),
            opts.text_engine_is_vector,
            acc,
            &richtext_mask_sources,
        );
    }
}

/// Fold a surface's accumulated axes into its `PageManifest`.
///
/// Verdict aggregation EXCLUDES the renderable-informational `unknown:node-field`
/// and `unknown:envelope-field` axes (`axis_elevates_verdict`): an unrecognized
/// NODE field or top-level ENVELOPE key is recorded as an axis for observability
/// but must never elevate the verdict or decline the doc — the CI schema-parity
/// gate proves both are non-canonical and rendered identically (ignored) by both
/// renderers.
fn finish_surface_manifest(page_index: usize, page_id: &str, acc: PageAcc) -> PageManifest {
    let verdict = acc
        .axes
        .iter()
        .filter(|(name, a)| axis_elevates_verdict(name, &a.status))
        .map(|(_, a)| a.status.clone())
        .max_by_key(AxisStatus::severity)
        .unwrap_or(AxisStatus::Full);
    PageManifest {
        page_index,
        page_id: page_id.to_string(),
        verdict,
        axes: acc
            .axes
            .into_iter()
            .map(|(axis, a)| AxisEntry {
                axis,
                status: a.status,
                node_ids: a.node_ids,
                detail: a.detail,
            })
            .collect(),
        warnings: acc.warnings,
    }
}

#[allow(clippy::too_many_arguments)]
fn assess_node(
    scene: &Scene,
    node: &RawNode,
    registry: &ShaderRegistry,
    catalog: &FontCatalog,
    variant_gap_probe: Option<&dyn Fn(&str) -> Vec<String>>,
    hybrid_vector_probe: Option<&dyn Fn(&str) -> bool>,
    text_engine_is_vector: bool,
    acc: &mut PageAcc,
    richtext_mask_sources: &std::collections::HashSet<String>,
) {
    let id = node.id();

    // Unrecognized node-field spillover: recorded as a renderable-informational
    // axis (see `axis_elevates_verdict`) — the CI schema-parity gate proves any
    // field spilling here is non-canonical (foreign / legacy / agent-lane) and is
    // ignored identically by both renderers, so it must NOT decline the doc.
    for field in crate::fields::unknown_fields(node) {
        let mut d = Map::new();
        d.insert("field".into(), json!(field));
        acc.add("unknown:node-field", AxisStatus::Unknown, id, d);
    }

    // A geometry effect stack that stopped before its end (`live_effects::StackWarning`: more effects than
    // `MAX_GEOMETRY_EFFECTS`, or one past `MAX_SEGMENTS`). Every lane draws the same outline (the effects before
    // the stop, through the one implementation), so it is a warning, never a degrade.
    let stack_warnings = scene
        .effective_node(id)
        .and_then(|n| n.json_field(crate::live_effects::EFFECT_WARNINGS_KEY))
        .and_then(Value::as_array);
    for w in stack_warnings.into_iter().flatten() {
        let mut d = w.as_object().cloned().unwrap_or_default();
        d.insert("node".into(), json!(id));
        let code = w.get("code").and_then(Value::as_str).unwrap_or("unknown");
        acc.warnings.push(ManifestWarning {
            axis: format!("effects:{code}"),
            detail: d,
        });
    }

    // M4d — text-as-mask (Q-28): a group clipping by a richtext source
    // builds its R8 coverage from the Canvas2D standalone texture in the
    // browser; natively from a text raster. On page index 0:
    // - Under the GA VECTOR engine, a vector-servable mask source renders
    //   its raster from the exact vector glyph outlines
    //   (`moda_text::vector::render_mask_raster`) — the same letterforms the
    //   node draws as visible text, measured at 0.99955 SSIM vs the browser
    //   golden on `masks/text-alpha` (the Degraded raster measured 0.96508)
    //   — the same analytic-AA-residual class the vector visible-text lane
    //   ships as Full, so the axis is Full with an `engine` detail.
    // - Fallback-class sources (non-vector families/scripts/emoji, effects,
    //   shadow/stroke/background/gradient) keep the Degraded-tier cosmic
    //   raster and the honest Degraded band. The MANIFEST-driven catalog
    //   family alias (moda-text `degraded::load_catalog_fonts`) fixed that
    //   arm's worst deficit — Noto silently substituting for opsz-bracketed
    //   catalog families ("Inter" vs name-table "Inter 24pt", the
    //   probe-proven (+6,+6)px coverage misplacement; the fixture's degraded
    //   arm now measures SSIM ~0.9995 png / ~0.9956 jpg) — but the band stays
    //   for the class: arbitrary fallback content (non-catalog families,
    //   scripts, effects) still rasterizes with cosmic letterforms that are
    //   not the browser's.
    // Every page builds the coverage: the export path dropped it off page 0
    // (children baked UNCLIPPED, a copied browser quirk, Q-95) while the
    // editor clipped, so page 2 of a PDF differed from its preview.
    if node.node_type() == "group" {
        if let Some(mask_id) = node.str("clipMaskNodeId") {
            if let Some(mask_node) = scene.node(mask_id) {
                if mask_node.node_type() == "richtext" {
                    let mut d = Map::new();
                    d.insert("mask_node".into(), json!(mask_id));
                    let status = if text_engine_is_vector
                        && vector_text_mask_servable(mask_node, catalog)
                    {
                        d.insert("engine".into(), json!("vector"));
                        AxisStatus::Full
                    } else {
                        AxisStatus::Degraded {
                            expected_ssim_band: (0.90, 0.97),
                        }
                    };
                    acc.add("mask:text-alpha", status, id, d);
                }
            }
        }
    }

    match node.node_type() {
        "blend" => {
            // Blend nodes are renderable (`crate::blend`) and contribute no
            // manifest axis. A blend with missing or cross-page objects draws
            // nothing, so it stays unreported too.
            return;
        }
        "generator" => {
            // Live generators (live-nodes.md §7.1) are evaluated natively (`crate::generator`, the `code` feature
            // moda-render builds with) and drawn as transients: nothing to route elsewhere.
            return;
        }
        "generated" => {
            match node.str("generatedKind") {
                Some("chart") => {
                    // bar/line/pie/scatter/area/combo generators are ported
                    // (crate::charts), including time (date) X axes — the FULL
                    // production registry (utils/markup/elements/charts/types.ts,
                    // generators/register.ts). A spec type outside that set has
                    // no generator in EITHER lane: the browser renders the same
                    // placeholder rect (chartExpansion.ts:110-159) our expansion
                    // mirrors byte-equally, so routing to the browser buys zero
                    // fidelity. Per the Q-95 precedent (browser-parity drop),
                    // the axis is Full with a `dropped` detail note — the
                    // spec_type detail is kept for decision-event attribution.
                    let empty = Map::new();
                    let src = node.object("generatedSource").unwrap_or(&empty);
                    if let Some(mut detail) = crate::charts::chart_gate_reason(src) {
                        if detail.get("reason").and_then(Value::as_str) == Some("raw-array-data") {
                            // A raw row array that SURVIVED scene-load
                            // normalization — i.e. one `rowsToParsedChartData`
                            // (ported in charts::datarows, wired in
                            // normalize.rs) could not coerce because the TS
                            // would THROW on it. Fail closed: gate to the
                            // browser rather than render malformed data.
                            // Coercible arrays are rewritten to the ParsedData
                            // bundle at load and render natively (Full).
                            acc.add("generated:chart", AxisStatus::Unsupported, id, detail);
                        } else if detail.get("reason").and_then(Value::as_str)
                            == Some("conditional-config")
                        {
                            // ChartConfigIssue::BrowserSafer: whether the
                            // browser renders a real chart or the placeholder
                            // here depends on values this mirror does not
                            // model (per-type seriesStyles short-circuits, JS
                            // string palette/explode semantics). The browser
                            // is right in every subcase, so gate to it —
                            // same classification as raw-array-data.
                            acc.add("generated:chart", AxisStatus::Unsupported, id, detail);
                        } else if detail.get("reason").and_then(Value::as_str)
                            == Some("missing-config")
                        {
                            // Config-less spec with real data: the browser
                            // generator THROWS on its first `spec.config`
                            // read and the catch renders the placeholder
                            // (chartExpansion.ts:68-76); native draws the
                            // identical placeholder pre-dispatch
                            // (charts::expand_chart_generated_node). Same
                            // output in both lanes ⇒ Full / non-gating —
                            // but the axis keeps the state OBSERVABLE (it
                            // used to render as a silent ~1px blank with a
                            // zero-axis FULL manifest).
                            detail.insert(
                                "dropped".into(),
                                json!(
                                    "config-less spec — the browser generator throws and \
                                     renders the placeholder rect, browser parity \
                                     (chartExpansion.ts:68-76)"
                                ),
                            );
                            acc.add("generated:chart", AxisStatus::Full, id, detail);
                        } else {
                            detail.insert(
                                "dropped".into(),
                                json!(
                                    "no generator in either lane — placeholder rect, \
                                     browser parity (chartExpansion.ts:110)"
                                ),
                            );
                            acc.add("generated:chart", AxisStatus::Full, id, detail);
                        }
                    }
                    return;
                }
                // A locally generated QR (cachedVectorPath) expands into a plain rect +
                // path pair (expansion.rs expand_qr_vector) that every lane draws natively.
                Some("qr") if crate::expansion::is_vector_qr(node) => {}
                // A QR without a stored vector draws the placeholder rect in both
                // renderers (there is no cached-asset QR any more).
                Some("qr") => {
                    let mut d = Map::new();
                    d.insert("kind".into(), json!("qr"));
                    d.insert("reason".into(), json!("missing cachedVectorPath → placeholder"));
                    acc.add("generated:placeholder", AxisStatus::Full, id, d);
                }
                Some("latex") => {
                    // Render via cachedAssetUrl pattern rect. SVG cached
                    // assets (LaTeX vector-asset) need rasterization — flag
                    // as a warning for the asset pipeline's usvg probe.
                    if let Some(url) = node.str("cachedAssetUrl") {
                        if is_svg_url(url) || node.str("renderMode") == Some("vector-asset") {
                            let mut d = Map::new();
                            d.insert("asset".into(), json!(url));
                            acc.warnings.push(ManifestWarning {
                                axis: "svg:generated-asset".to_string(),
                                detail: d,
                            });
                        }
                    } else {
                        // Missing cached asset → placeholder rect renders.
                        // Parity-Full, not a fidelity gap: BOTH renderers draw
                        // the byte-identical placeholder rect (expansion.rs
                        // create_placeholder_rect == generatedGraphicExpansion.ts
                        // createPlaceholderRect), and cachedAssetUrl is written
                        // only at generation/edit time — no lane regenerates it
                        // at render time, so the browser cannot out-render this.
                        // Mirrors the mask:text-alpha browser-parity precedent
                        // above. The axis stays in the manifest for
                        // observability of missing-asset docs; re-examine this
                        // marking if a render-time asset (re)generation
                        // pipeline ever lands on a browser/export path.
                        let mut d = Map::new();
                        d.insert(
                            "kind".into(),
                            json!(node.str("generatedKind").unwrap_or("")),
                        );
                        d.insert(
                            "reason".into(),
                            json!("missing cachedAssetUrl → placeholder"),
                        );
                        d.insert(
                            "parity".into(),
                            json!(
                                "parity-full: both renderers draw the identical placeholder rect \
                                 (expansion.rs:create_placeholder_rect == generatedGraphicExpansion.ts); \
                                 no lane regenerates cachedAssetUrl at render time"
                            ),
                        );
                        acc.add("generated:placeholder", AxisStatus::Full, id, d);
                    }
                }
                Some(other) => {
                    // Unrecognized kind → the expansion draws the SAME placeholder
                    // rect the browser draws for its default branch
                    // (expansion.rs `create_placeholder_rect` ↔
                    // generatedGraphicExpansion.ts default arm) — pixel-identical
                    // lanes, so this is Degraded (placeholder-band), not Unknown.
                    let mut d = Map::new();
                    d.insert("kind".into(), json!(other));
                    d.insert(
                        "reason".into(),
                        json!(
                            "unrecognized generatedKind → placeholder rect (browser parity: generatedGraphicExpansion.ts default branch)"
                        ),
                    );
                    acc.add(
                        "unknown:generated-kind",
                        AxisStatus::Degraded {
                            expected_ssim_band: (0.90, 0.99),
                        },
                        id,
                        d,
                    );
                }
                None => {
                    let mut d = Map::new();
                    d.insert(
                        "reason".into(),
                        json!(
                            "unrecognized generatedKind → placeholder rect (browser parity: generatedGraphicExpansion.ts default branch)"
                        ),
                    );
                    acc.add(
                        "unknown:generated-kind",
                        AxisStatus::Degraded {
                            expected_ssim_band: (0.90, 0.99),
                        },
                        id,
                        d,
                    );
                }
            }
        }
        "table" => {
            // Tables expand to rect/richtext/line transients — assess the
            // cell text against the text predicates.
            if let Some(rows) = node.array("cells") {
                for row in rows {
                    let Some(cells) = row.as_array() else {
                        continue;
                    };
                    for cell in cells {
                        if let Some(text) = cell
                            .as_object()
                            .and_then(|c| c.get("textContent"))
                            .and_then(Value::as_str)
                        {
                            assess_text_content(
                                text,
                                id,
                                catalog,
                                variant_gap_probe,
                                hybrid_vector_probe,
                                text_engine_is_vector,
                                acc,
                            );
                        }
                    }
                }
            }
        }
        "richtext" => {
            // Q-95/Q-28: an active clip-mask SOURCE never renders as text in
            // either renderer (shouldSkipRender) — its pixel impact (if any)
            // is the mask coverage, reported on the owning group as
            // mask:text-alpha. Suppress its text axes so a never-drawn node
            // can't band the page.
            if !richtext_mask_sources.contains(id) {
                if let Some(html) = node.str("htmlContent") {
                    assess_text_content(
                        html,
                        id,
                        catalog,
                        variant_gap_probe,
                        hybrid_vector_probe,
                        text_engine_is_vector,
                        acc,
                    );
                }
            }
            if let Some(crate::paint::Paint::Shader(s)) = crate::paint::top_text_fill(node) {
                assess_shader_id(Some(&s.shader_id), id, registry, acc, true);
            }
            // The text lane draws one glyph fill and one glyph stroke (E3.1
            // P6): a `textFills` stack of solids folds into one colour (the
            // engine's `paint::layers::text_fill_paint`); any other stack, and
            // a stack of strokes, renders its top layer only.
            let shown_strokes = crate::paint::node_strokes(node)
                .iter()
                .filter(|s| crate::paint::shown(s.visible, s.opacity))
                .count();
            let text_paints = crate::paint::node_text_fills(node);
            let text_fills = text_paints.len();
            let fills_fold = text_paints
                .iter()
                .all(|p| matches!(p, crate::paint::Paint::Solid(_)));
            if (text_fills > 1 && !fills_fold) || shown_strokes > 1 {
                let mut d = Map::new();
                d.insert(
                    "reason".into(),
                    json!("text draws its top textFills layer and top stroke only"),
                );
                d.insert("textFills".into(), json!(text_fills));
                d.insert("strokes".into(), json!(shown_strokes));
                acc.add(
                    "paint:text-stack",
                    AxisStatus::Degraded {
                        expected_ssim_band: (0.90, 0.99),
                    },
                    id,
                    d,
                );
            }
        }
        _ => {}
    }

    // Shape text frames share the richtext pipeline.
    if node.node_type() != "richtext" && node.node_type() != "table" {
        if let Some(text) = node.str("textContent") {
            if !text.is_empty() {
                assess_text_content(
                    text,
                    id,
                    catalog,
                    variant_gap_probe,
                    hybrid_vector_probe,
                    text_engine_is_vector,
                    acc,
                );
            }
        }
    }

    // Fill/shader/effects axes for shape-like nodes.
    if matches!(
        node.node_type(),
        "rectangle" | "container" | "section" | "ellipse" | "star" | "polygon" | "path" | "line"
    ) {
        assess_fill_surface(node, id, registry, acc);
    }

    // `shader` entries of `effects` (effect.ts). M4e: the FIRST renderable
    // source-mask effect routes the node through the custom-shader pipeline
    // (image-backed fills) — it gates via the shader-id registry like any
    // fill shader (`assess_shader_id`); a stale self-asset effect falls
    // through to the underlying fill (renders fine, no axis). Everything
    // else among the shader entries (post-process mode, extra source-mask
    // entries beyond the first) is inert-but-unrendered: the
    // browser reads ONLY the first renderable source-mask effect
    // (`getRenderableSourceMaskShaderEffect`) and never touches the others, so
    // the native engine renders these docs pixel-identically (the flagged
    // entries render nothing in EITHER engine — proven byte-exact). We still
    // record the `effects:shader-effects` axis for drift observability, but it
    // is NON-gating (`axis_elevates_verdict`): declining would only burn a
    // Chromium render, and once the browser renderer retires it would leave
    // the doc unrenderable. Status is Unknown (a novelty signal, not a
    // capability gap — the same posture as `unknown:node-field`).
    if crate::shader_effects::shader_effect_entries(node).next().is_some() {
        let renderable = crate::shader_effects::renderable_source_mask_effect(node);
        if let Some(effect) = renderable {
            let shader_id = effect.get("shaderId").and_then(Value::as_str);
            if shader_id.is_some_and(|sid| !sid.is_empty()) {
                assess_shader_id(shader_id, id, registry, acc, false);
            }
        }
        {
            let unrendered: Vec<&Value> = crate::shader_effects::shader_effect_entries(node)
                .filter(|e| match renderable {
                    Some(r) => !std::ptr::eq(*e, r),
                    None => {
                        // A stale self-asset source-mask effect is render-
                        // inert (falls through to the fill) — not gated.
                        e.get("mode").and_then(Value::as_str)
                            != Some(crate::shader_effects::SOURCE_MASK_SHADER_EFFECT_MODE)
                    }
                })
                .collect();
            if !unrendered.is_empty() {
                let mut d = Map::new();
                let ids: Vec<&str> = unrendered
                    .iter()
                    .filter_map(|e| {
                        e.as_object()
                            .and_then(|o| o.get("shaderId"))
                            .and_then(Value::as_str)
                    })
                    .collect();
                d.insert("shader_ids".into(), json!(ids));
                // Surface the inert entries' modes/types in the decision log so a
                // FUTURE renderer that starts drawing a new mode (making this
                // fail-open non-gate wrong) is caught as a ramp in telemetry
                // before it can diverge — the observability half of leaving this
                // axis non-gating (`axis_elevates_verdict`).
                let modes: Vec<&str> = unrendered
                    .iter()
                    .map(|e| {
                        e.get("mode")
                            .and_then(Value::as_str)
                            .or_else(|| e.get("type").and_then(Value::as_str))
                            .unwrap_or("(none)")
                    })
                    .collect();
                d.insert("modes".into(), json!(modes));
                acc.add("effects:shader-effects", AxisStatus::Unknown, id, d);
            }
        }
    }

    // Structured node effects `effects[]` — glow / echo / extrude:
    // PARITY-FULL, no axis. The Q-123 `effects:structured` gate
    // (stage 1) was removed in the same PR that ported the render path,
    // exactly as the quirk instructed:
    // - richtext: the vector ops walk renders the full pass model
    //   (`moda_text::vector::ops::resolve_effect_passes`, the port of
    //   vectorTextEffects.ts:168-250 + vectorTextExtrude.ts), and EVERY
    //   engine lane routes active-effects nodes through the vector arm
    //   (build_richtext_item, Q-124) — the browser's flag-on routing, the
    //   only configuration that renders effects[] at all. Per-node vector
    //   declines fall back to MSDF without effects, byte-matching the
    //   browser's renderWithVector → renderWithMsdf fallback.
    // - non-richtext kinds: glow / echo / extrude entries are text effects;
    //   the browser ignores them on shapes too — parity-Full.

    let _ = scene;
}

fn is_svg_url(url: &str) -> bool {
    let lower = url.to_lowercase();
    let path = lower.split(['?', '#']).next().unwrap_or("");
    path.ends_with(".svg") || lower.starts_with("data:image/svg")
}

/// Fill-surface axes shared by nodes and pages: shader fills (registry gate),
/// SVG pattern images (warning), backdrop variants. Reads the `fills` stack
/// (a page's too: it is passed here as a node-shaped map).
fn assess_fill_surface(node: &RawNode, id: &str, registry: &ShaderRegistry, acc: &mut PageAcc) {
    // Every visible layer draws (E3.1 P6): each shader layer is gated.
    let fills = crate::paint::parse_fills(node.json_field("fills"));
    for f in fills
        .iter()
        .filter(|f| crate::paint::shown(f.visible, f.opacity))
    {
        if let crate::paint::Paint::Shader(s) = &f.paint {
            assess_shader_id(Some(&s.shader_id), id, registry, acc, false);
        }
    }
    // SVG pattern fills: WARNING ONLY, never an axis — structurally NON-GATING
    // (verdict aggregation reads `acc.axes` only, and router policy reads only
    // verdict/renderable), so an SVG fill can never route a render to the
    // browser lane. That is deliberate: BOTH lanes rasterize SVG through the
    // same `moda_assets::svg::prepare_svg` (resvg) — the wasm editor via
    // `decode_pattern_bytes`, the server via `fetch::load_one` — and the
    // editor's canonical baseline IS resvg (the WASM engine is the default
    // renderer), so server-vs-editor parity is exact by construction EXCEPT
    // SVG `<text>`: the server rasterizes with a CATALOG-SEEDED fontdb while
    // the wasm lane uses the deliberately empty `default_fontdb()` (renders
    // fontless there), so the server can render MORE text, never less. The
    // byte-level `moda_assets::probe::probe_svg` (server prefetch lane) emits
    // `svg:text` for exactly that case, plus `svg:parse-error` /
    // `svg:embedded-image` / `svg:sanitized-content`. This URL-heuristic
    // warning survives only as an assess-time observability breadcrumb for
    // resvg-vs-Chromium deltas on the (draining) browser-fallback lane — the
    // reason `thresholds.toml` gates SVG fixtures at 0.97 — and undercounts
    // extension-less SVGs by design (decode-time sniff still renders them).
    for f in &fills {
        let crate::paint::Paint::Image(image) = &f.paint else {
            continue;
        };
        let Some(src) = image.src.as_deref() else {
            continue;
        };
        if is_svg_url(src) {
            let mut d = Map::new();
            d.insert("asset".into(), json!(src));
            acc.warnings.push(ManifestWarning {
                axis: "svg:pattern-image".to_string(),
                detail: d,
            });
        }
    }
    // Backdrop effects (`backdropIsActive`, effect.ts): the node's first
    // visible backdrop entry of `effects`. A hidden one (`visible: false`) is
    // inert.
    if let Some(backdrop) = crate::effect::backdrop(node) {
        if crate::effect::backdrop_is_active(&backdrop) {
            if let crate::effect::EffectKind::Glass(glass) = &backdrop.kind {
                // DECISION S11 sanctioned Degraded entries only "for exotic
                // params"; the original code over-applied it to EVERY active
                // glass, elevating whole-doc verdicts and pushing export lanes
                // to the browser. The measured golden ratchet floors
                // (fixtures/thresholds.toml [ratchet], 2026-06-12 standard
                // golden lane: backdrop/glass 0.99566–0.99690,
                // backdrop/glass-params 0.99467–0.99528) sit above the 0.98
                // global bar — S11's pre-implementation 0.97–0.98 projection
                // is falsified for the fixture-covered parameter envelope, and
                // the editor's WASM lane already shows users this same
                // engine's glass live. So: common glass is Full (no
                // gating axis) with a non-gating warning kept for
                // observability; only params beyond the measured fixture
                // envelope retain a Degraded entry, its band narrowed to what
                // the covered neighborhood actually measures. The envelope
                // covers EVERY render-affecting scalar knob at the
                // fixtures/parity/design/scenes/backdrop/glass*.json maxima — frost 30,
                // refraction 90, depth 70, dispersion 80, lightIntensity 80,
                // splay 50 (lightAngle is directional and the fixtures span
                // the circle; all three refractionStyles are covered) — so no
                // unmeasured parameter neighborhood is promoted to Full.
                let exotic = glass.frost > 30.0
                    || glass.refraction > 90.0
                    || glass.depth > 70.0
                    || glass.dispersion > 80.0
                    || glass.light_intensity > 80.0
                    || glass.splay > 50.0;
                let mut d = Map::new();
                d.insert("type".into(), json!("glass"));
                if exotic {
                    acc.add(
                        "backdrop:glass",
                        AxisStatus::Degraded {
                            expected_ssim_band: (0.99, 0.995),
                        },
                        id,
                        d,
                    );
                } else {
                    acc.warnings.push(ManifestWarning {
                        axis: "backdrop:glass".to_string(),
                        detail: d,
                    });
                }
            }
            // background-blur/pixelate/dither: Full (extracted WGSL path, M3e) — no
            // axis entry needed; absence of an axis means Full.
        }
    }
}

fn assess_shader_id(
    shader_id: Option<&str>,
    node_id: &str,
    registry: &ShaderRegistry,
    acc: &mut PageAcc,
    is_text_fill: bool,
) {
    let Some(raw_id) = shader_id.filter(|s| !s.is_empty()) else {
        return; // inactive shader fill (hasActiveRoundedRectSurfaceFill)
    };
    let resolved = registry.resolve(raw_id).to_string();
    let axis = format!("shader:{resolved}");
    let mut d = Map::new();
    d.insert("shader_id".into(), json!(resolved));
    if raw_id != resolved {
        d.insert("alias_of".into(), json!(raw_id));
    }
    if is_text_fill {
        d.insert("text_fill".into(), json!(true));
    }
    if !registry.known_ids.contains(&resolved) {
        acc.add(&axis, AxisStatus::Unknown, node_id, d);
    } else if is_text_fill && !registry.supports_text_fill(&resolved) {
        // M4 "richtext shader fills": text-fill eligibility is the browser's
        // own predicate (`supportsMsdfTextFill` — descriptor `shade`, no
        // primaryImage). Eligible ids render via the extracted
        // `<id>.msdf-text-fill` module → Full. Image-backed shaders are
        // ineligible in the BROWSER too: it silently falls back to per-run
        // colors (`WebGPUTextSystem.resolveTextNodeFill`), and the engine
        // byte-copies that exact fallback (text_draw.rs `shader_fill` →
        // None → per-run colors). Native output is IDENTICAL to the browser
        // lane, so this is parity-full BY CONSTRUCTION and must not gate —
        // declining would reproduce the same pixels via Chromium. Emit a
        // ManifestWarning (like `svg:pattern-image`) so observability of the
        // fallback is retained without elevating the verdict.
        d.insert(
            "reason".into(),
            json!(
                "image-backed shader ineligible for text fill (supportsMsdfTextFill); \
                 renders browser-identical per-run colors"
            ),
        );
        acc.warnings.push(ManifestWarning {
            axis: "shader:text-fill-fallback".to_string(),
            detail: d,
        });
    } else if registry.gated_ids.contains(&resolved) {
        acc.add(&axis, AxisStatus::Unsupported, node_id, d);
    }
    // known + not gated → Full → no axis entry.
}

// ---------------------------------------------------------------------------
// Text predicates (msdfFonts.ts + coverageScanner.ts clones)
// ---------------------------------------------------------------------------

/// Codepoint fallback classes per DECISION §1.4 (non-catalog families,
/// CJK/Greek/Cyrillic/Arabic, …). `isWhitespaceOrControl` analog skipped.
fn codepoint_script_class(cp: u32) -> Option<&'static str> {
    match cp {
        0x0370..=0x03FF | 0x1F00..=0x1FFF => Some("Greek"),
        0x0400..=0x04FF | 0x0500..=0x052F => Some("Cyrillic"),
        0x0590..=0x05FF | 0xFB1D..=0xFB4F => Some("Hebrew"),
        0x0600..=0x06FF
        | 0x0750..=0x077F
        | 0x0870..=0x089F
        | 0x08A0..=0x08FF
        | 0xFB50..=0xFDFF
        | 0xFE70..=0xFEFF => Some("Arabic"),
        0x0700..=0x074F | 0x0860..=0x086F => Some("Syriac"),
        0x0780..=0x07BF => Some("Thaana"),
        0x07C0..=0x07FF => Some("NKo"),
        0x0800..=0x083F => Some("Samaritan"),
        0x0840..=0x085F => Some("Mandaic"),
        // Brahmic / Indic — need complex shaping; the vector lane renders them
        // via cosmic-text, MSDF/hybrid cannot (so they degrade under those).
        // Ranges MUST stay in sync with `moda_text::complex_script`.
        0x0900..=0x097F | 0xA8E0..=0xA8FF => Some("Devanagari"),
        0x0980..=0x09FF => Some("Bengali"),
        0x0A00..=0x0A7F => Some("Gurmukhi"),
        0x0A80..=0x0AFF => Some("Gujarati"),
        0x0B00..=0x0B7F => Some("Oriya"),
        0x0B80..=0x0BFF => Some("Tamil"),
        0x0C00..=0x0C7F => Some("Telugu"),
        0x0C80..=0x0CFF => Some("Kannada"),
        0x0D00..=0x0D7F => Some("Malayalam"),
        0x0D80..=0x0DFF => Some("Sinhala"),
        0x11000..=0x110FF => Some("Brahmi"),
        // Southeast Asian
        0x0E00..=0x0E7F => Some("Thai"),
        0x0E80..=0x0EFF => Some("Lao"),
        0x0F00..=0x0FFF => Some("Tibetan"),
        0x1000..=0x109F | 0xA9E0..=0xA9FF | 0xAA60..=0xAA7F => Some("Myanmar"),
        0x1780..=0x17FF | 0x19E0..=0x19FF => Some("Khmer"),
        0x1800..=0x18AF => Some("Mongolian"),
        0x1900..=0x194F => Some("Limbu"),
        0x1A20..=0x1AAF => Some("Tai Tham"),
        0x10D00..=0x10D3F => Some("Hanifi Rohingya"),
        0x1E900..=0x1E95F => Some("Adlam"),
        0x1100..=0x11FF | 0xAC00..=0xD7AF | 0x3130..=0x318F => Some("Hangul"),
        0x3040..=0x309F | 0x30A0..=0x30FF | 0x31F0..=0x31FF => Some("Kana"),
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F => Some("Han"),
        _ => None,
    }
}

fn is_emoji_codepoint(cp: u32) -> bool {
    matches!(cp,
        0x1F300..=0x1FAFF | 0x2600..=0x27BF | 0x1F1E6..=0x1F1FF | 0x2190..=0x21FF | 0xFE0F | 0x200D)
}

/// Length of a well-formed tag run at `from` — one or more U+E0020–E007E then
/// the U+E007F terminator — or 0 when there is none. Mirrors `tag_run_end` in
/// `moda_text::emoji`, which the RENDER path uses to absorb the run into its
/// U+1F3F4 cluster.
fn tag_run_len(chars: &[char], from: usize) -> usize {
    let mut j = from;
    while j < chars.len() && matches!(chars[j] as u32, 0xE0020..=0xE007E) {
        j += 1;
    }
    if j > from && j < chars.len() && chars[j] as u32 == 0xE007F {
        j + 1 - from
    } else {
        0
    }
}

/// Extract `font-family: ...` declarations from inline styles in an HTML
/// subset string (renderer parser semantics: quotes stripped —
/// utils/richtext/parser.ts:70-152).
fn extract_font_families(html: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let lower = html.to_lowercase();
    let mut idx = 0;
    while let Some(rel) = lower[idx..].find("font-family") {
        let start = idx + rel + "font-family".len();
        let rest = &html[start..];
        let Some(colon) = rest.find(':') else {
            idx = start;
            continue;
        };
        let after = &rest[colon + 1..];
        let end = after
            .find(|c| c == ';' || c == '"' || c == '\'')
            .unwrap_or(after.len());
        let fam = after[..end].trim().trim_matches(['"', '\'', '`']).trim();
        if !fam.is_empty() {
            out.insert(fam.to_string());
        }
        idx = start;
    }
    out
}

/// Strip tags to approximate visible text (the manifest only needs the
/// codepoint population, not layout).
fn visible_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    // Minimal entity decoding for the common cases.
    out.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", "\u{a0}")
        .replace("&quot;", "\"")
}

/// mask:text-alpha better tier: would the render lane's vector mask arm
/// (`moda_text::vector::render_mask_raster`, reached only under the GA
/// vector engine) serve this mask-source richtext node? Conservative mirror
/// of the render-side decline gates plus the vector routing's content
/// classes, evaluated from node fields: the axis flips to Full only when
/// every signal says the vector arm fires; any uncertainty keeps the honest
/// Degraded band (never a false Full — the render arm itself falls back to
/// the Degraded raster on ANY decline).
fn vector_text_mask_servable(node: &RawNode, catalog: &FontCatalog) -> bool {
    // Silhouette-affecting features → the render arm declines (mask.rs /
    // mask_raster.rs gates: drop shadow, legacy stroke, gradient fill, structured
    // effects, kinetic per-glyph transforms).
    if crate::paint::top_stroke(node).is_some() {
        return false;
    }
    if matches!(
        crate::paint::top_text_fill(node),
        Some(crate::paint::Paint::Linear(_) | crate::paint::Paint::Radial(_))
    ) {
        return false;
    }

    // A drop shadow or a glow / echo / extrude entry (visible or not) changes
    // the silhouette the mask samples.
    if crate::effect::node_effects_json(node).iter().any(|e| {
        matches!(
            e.get("type").and_then(Value::as_str),
            Some("drop-shadow" | "glow" | "echo" | "extrude")
        )
    }) {
        return false;
    }
    if node.array("unitTransforms").is_some_and(|a| !a.is_empty()) {
        return false;
    }
    let Some(html) = node.str("htmlContent") else {
        return false;
    };
    if html.is_empty() {
        return false;
    }
    // Content classes that route runs off the vector lane
    // (`select_text_engine` ≠ vector / run-resolution fallbacks): non-catalog
    // families (incl. the fallback-class 'sans-serif' default), emoji and
    // regional-indicator codepoints, and non-Latin fallback scripts.
    let mut families = extract_font_families(html);
    if families.is_empty() {
        families.insert(crate::defaults::richtext::FONT_FAMILY.to_string());
    }
    if families.iter().any(|f| !catalog.is_msdf_family(f)) {
        return false;
    }
    for ch in visible_text(html).chars() {
        let cp = ch as u32;
        if ch.is_whitespace() || ch.is_control() {
            continue;
        }
        if matches!(cp, 0x1F1E6..=0x1F1FF) || is_emoji_codepoint(cp) {
            return false;
        }
        if codepoint_script_class(cp).is_some() {
            return false;
        }
    }
    true
}

fn assess_text_content(
    html: &str,
    node_id: &str,
    catalog: &FontCatalog,
    variant_gap_probe: Option<&dyn Fn(&str) -> Vec<String>>,
    hybrid_vector_probe: Option<&dyn Fn(&str) -> bool>,
    text_engine_is_vector: bool,
    acc: &mut PageAcc,
) {
    if html.is_empty() {
        return;
    }
    // Task #72: vendored-variant gap (`--input` without `--fonts` only). The
    // CLI's probe replays the ENGINE's run mapping (same parser, same
    // nearest-variant snap, same `<b>` bump) against the vendored host: a
    // catalog-promised variant without vendored atlas bytes renders via the
    // cosmic Degraded tier (fallback face — build_batch routes variant-gapped
    // runs through the degraded raster unconditionally; it used to silently
    // DROP them on the strict lane), so the manifest must degrade instead of
    // reporting `full`. The (0.90, 0.97) band is truthful for the substituted
    // typeface. This is a MISSING-ATLAS-BYTES artifact, so — like the family
    // gate below — it is irrelevant under vector (vector draws the variant's
    // TTF outlines directly).
    if !text_engine_is_vector {
        if let Some(probe) = variant_gap_probe {
            let gaps = probe(html);
            if !gaps.is_empty() {
                let mut d = Map::new();
                d.insert("missing_vendored_keys".into(), json!(gaps));
                acc.add(
                    "text:variant-gap",
                    AxisStatus::Degraded {
                        expected_ssim_band: (0.90, 0.97),
                    },
                    node_id,
                    d,
                );
            }
        }
    }
    // Family gate: every family mentioned (or the sans-serif default when
    // none) must be MSDF-ready (msdfFonts.ts:27-31). The default family is
    // 'sans-serif' (utils/richtext/types.ts:30-39) which is never a catalog
    // family → fallback-class (DECISION §1.4).
    let mut families = extract_font_families(html);
    if families.is_empty() {
        families.insert(crate::defaults::richtext::FONT_FAMILY.to_string());
    }
    let non_msdf: Vec<String> = families
        .iter()
        .filter(|f| !catalog.is_msdf_family(f))
        .cloned()
        .collect();

    // Coverage-class gate: scan codepoints for fallback scripts
    // (coverageScanner.ts concept; whitespace/control skipped).
    let text = visible_text(html);
    let mut scripts: BTreeSet<&'static str> = BTreeSet::new();
    let mut has_emoji = false;
    let mut emoji_coverage_gap = false;
    let chars: Vec<char> = text.chars().collect();
    let mut ci = 0usize;
    while ci < chars.len() {
        let ch = chars[ci];
        let cp = ch as u32;
        if ch.is_whitespace() || ch.is_control() {
            ci += 1;
            continue;
        }
        // Q-42 (aligned with the RENDER path): the cluster regex absorbs
        // skin-tone modifiers into their base pictograph's cluster
        // (utils/msdf/emoji.ts:6), so skin-toned emoji render via the Noto SVG
        // image pipeline like any other emoji — they fall through to the
        // `is_emoji_codepoint` warning below, NOT a coverage gap. Regional
        // indicators now PAIR greedily left-to-right exactly like the
        // segmenter (emoji.ts `RI_PAIR` / moda-text emoji.rs): a pair with a
        // shipped waved-flag SVG (`crate::flags`, generated from the emoji
        // pack + mirrored to the CDN) clusters and renders via the image
        // pipeline — renderable, warning only. An unshipped pair (e.g. 🇽🇽)
        // and a lone RI stay in TEXT runs and route the node to the fallback
        // lane (fallback class → Degraded tier), keeping the coverage gap.
        if matches!(cp, 0x1F1E6..=0x1F1FF) {
            let next = chars.get(ci + 1).map(|c| *c as u32);
            if let Some(n) = next.filter(|n| matches!(n, 0x1F1E6..=0x1F1FF)) {
                if crate::flags::is_valid_flag_pair(cp, n) {
                    has_emoji = true;
                } else {
                    emoji_coverage_gap = true;
                }
                ci += 2;
                continue;
            }
            emoji_coverage_gap = true;
            ci += 1;
            continue;
        }
        // Tag-sequence subdivision flags (🏴󠁧󠁢󠁳󠁣󠁴󠁿): the segmenter absorbs a
        // well-formed tag run into the U+1F3F4 cluster. A shipped sequence renders
        // its own SVG; an unshipped one degrades to the bare 🏴 SVG. BOTH render via
        // the image pipeline, so this is renderable either way — warning only, no
        // coverage gap (unlike lone / unshipped RI pairs, which stay in TEXT runs).
        if cp == 0x1F3F4 {
            has_emoji = true;
            ci += 1 + tag_run_len(&chars, ci + 1);
            continue;
        }
        // Any tag char still reachable here is an orphan — no preceding 🏴, or an
        // unterminated run. Text runs strip those before layout, so they must not
        // invent a script class or a coverage gap. Skip explicitly instead of
        // relying on `codepoint_script_class` happening to return None for the Tags
        // block, which is what left them classified into no branch at all.
        if matches!(cp, 0xE0020..=0xE007F) {
            ci += 1;
            continue;
        }
        if is_emoji_codepoint(cp) {
            has_emoji = true;
            ci += 1;
            continue;
        }
        if let Some(s) = codepoint_script_class(cp) {
            scripts.insert(s);
        }
        ci += 1;
    }

    // Only the FONT MSDF-readiness / catalog-membership gate (`non_msdf`) is a
    // pure atlas-era artifact the VECTOR engine (the GA default) makes moot: it
    // draws any font's outlines directly (a family with a TTF but no MSDF atlas
    // renders identically to the browser), so gating on it under vector could
    // only FALSE-decline renderable text. Suppress THAT under vector; MSDF/hybrid
    // keep it (they need a per-font atlas and diverge on non-vendored families).
    //
    // Non-Latin SCRIPT coverage also renders correctly on the native VECTOR
    // path now, so it no longer forces a decline under vector:
    //   - SIMPLE scripts (Greek/Cyrillic/CJK/Kana/Hangul) draw via the vector
    //     outline lane + per-script Noto coverage fallback;
    //   - COMPLEX scripts (Arabic/Hebrew/all Indic/Thai/Khmer/…) route to the
    //     cosmic-text tier (rustybuzz + `unicode-bidi`), the ONLY correct native
    //     shaper for them, whose bundled Noto fonts are baked into the render
    //     host (`/opt/moda/native/fonts`) — proven by rendered evidence.
    // MSDF/hybrid can't render these, so they KEEP the script degrade.
    //
    // `emoji_coverage_gap` (lone / unshipped-pair regional indicators — Q-42)
    // is still a genuine content gap on every engine: those codepoints do not
    // cluster (only pairs with a shipped waved-flag SVG do), so the
    // text-routed codepoints need a color-emoji face the native host does not
    // carry — so it keeps degrading.
    // Task #128 / drain-the-browser-lane: under `--text-engine hybrid` the
    // engine-truth probe answers "does the hybrid's VECTOR arm serve this
    // node?". When it does, the node renders on the exact same vector engine
    // the GA default lane (server `NATIVE_RENDER_TEXT_ENGINE=vector` + the
    // in-browser WASM renderer) ships as verdict FULL — so degrading it here
    // was pure inconsistency, never a capability gap. Vector-served nodes now
    // suppress the family/script degrades entirely (parity with the
    // `text_engine_is_vector` treatment above) and surface a
    // `text:fallback-engine:vector` ManifestWarning instead, preserving the
    // hybrid-lane diagnostics. (History: this used to emit a Degraded axis
    // with a 0.94-0.995 band, floor calibrated by the §5.3 real-corpus
    // measurement — worst served page aad05362 p1 at 0.94814, the analytic-AA
    // ink-weight residual on small light-on-dark text. That same divergence
    // ships un-flagged as Full on the default vector lane, so the Degraded
    // marking only ever false-declined renderable text.) Nodes that stay
    // cosmic (Q-90 non-catalog families, vector-error throws) keep the
    // classic degrade. The probe is None on msdf/vector engines, and is only
    // invoked when a family/script cause exists (the manifest walk stays
    // cheap — the probe replays full shaping).
    let vector_served = (!non_msdf.is_empty() || !scripts.is_empty())
        && hybrid_vector_probe.map(|p| p(html)).unwrap_or(false);
    let font_family_degrade = !text_engine_is_vector && !vector_served && !non_msdf.is_empty();
    let script_degrade = !text_engine_is_vector && !vector_served && !scripts.is_empty();
    if vector_served {
        let mut d = Map::new();
        d.insert("node".into(), json!(node_id));
        if !non_msdf.is_empty() {
            d.insert("families".into(), json!(non_msdf));
        }
        if !scripts.is_empty() {
            d.insert("scripts".into(), json!(scripts.iter().collect::<Vec<_>>()));
        }
        acc.warnings.push(ManifestWarning {
            axis: "text:fallback-engine:vector".to_string(),
            detail: d,
        });
    }
    if font_family_degrade || script_degrade || emoji_coverage_gap {
        let mut d = Map::new();
        if font_family_degrade {
            d.insert("families".into(), json!(non_msdf));
        }
        if script_degrade {
            d.insert("scripts".into(), json!(scripts.iter().collect::<Vec<_>>()));
        }
        if emoji_coverage_gap {
            d.insert("emoji_coverage_gap".into(), json!(true));
        }
        acc.add(
            "text:fallback-engine",
            AxisStatus::Degraded {
                expected_ssim_band: (0.90, 0.97),
            },
            node_id,
            d,
        );
    }
    if has_emoji {
        // Emoji (incl. skin-toned, ZWJ clusters, and RI-pair flags with a
        // shipped waved-flag SVG) render via the image pipeline (per-cluster
        // Noto SVGs) on BOTH native lanes: the baked lane from the bundled
        // full Noto set (`moda_fonts::emoji_bundle`), the catalog lane from
        // the CDN (bundle-first, CDN fallback). So this is renderable — a
        // warning only, never a degrade. (Lone regional indicators and pairs
        // without a shipped SVG are the exception: they do not cluster and are
        // handled by `emoji_coverage_gap` above, so they degrade.)
        let mut d = Map::new();
        d.insert("node".into(), json!(node_id));
        acc.warnings.push(ManifestWarning {
            axis: "text:emoji".to_string(),
            detail: d,
        });
    }
}

// ---------------------------------------------------------------------------
// JSON serialization (the CLI prints this; format per PLAN §3)
// ---------------------------------------------------------------------------

pub fn manifest_to_json(m: &Manifest) -> Value {
    let mut out = json!({
        "renderer": m.renderer,
        "engine_version": m.engine_version,
        "canvas_id": m.canvas_id,
        "verdict": m.verdict.as_str(),
        "renderable": m.renderable(),
        "pages": m.pages.iter().map(|p| json!({
            "page_index": p.page_index,
            "page_id": p.page_id,
            "verdict": p.verdict.as_str(),
            "axes": p.axes.iter().map(|a| {
                let mut o = Map::new();
                o.insert("axis".into(), json!(a.axis));
                o.insert("status".into(), json!(a.status.as_str()));
                // Whether this axis participates in verdict aggregation (can decline
                // a doc). Informational axes (`axis_elevates_verdict` == false) are
                // emitted for drift observability only; downstream attribution (the
                // router's `native_render.decision` summarizer) must not blame them
                // for a decline. Additive key — older consumers ignore it.
                o.insert("gating".into(), json!(axis_elevates_verdict(&a.axis, &a.status)));
                if let AxisStatus::Degraded { expected_ssim_band } = &a.status {
                    o.insert("expected_ssim_band".into(), json!([expected_ssim_band.0, expected_ssim_band.1]));
                }
                o.insert("node_ids".into(), json!(a.node_ids));
                o.insert("detail".into(), Value::Object(a.detail.clone()));
                Value::Object(o)
            }).collect::<Vec<_>>(),
            "warnings": p.warnings.iter().map(|w| json!({
                "axis": w.axis,
                "detail": w.detail,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    });
    // Additive key (older consumers ignore it): the page-less canvas-root
    // frame size — see `Manifest::canvas_root_frame_px`.
    if let Some((w, h)) = m.canvas_root_frame_px {
        out["canvas_root_frame_px"] = json!([w, h]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Scene;
    use crate::types::FontVariantEntry;
    use crate::value::json;

    fn page_with(nodes: Value, node_ids: Value) -> Value {
        json!({
            "nodes": nodes,
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                       "nodeIds": node_ids}]
        })
    }

    fn find_axis<'a>(m: &'a Manifest, axis: &str) -> Option<&'a AxisEntry> {
        m.pages[0].axes.iter().find(|a| a.axis == axis)
    }

    #[test]
    fn pageless_canvas_root_doc_assessed_as_synthetic_surface() {
        // A page-less freeform doc's floating nodes are walked as ONE
        // synthetic surface (page_id = SCENE_ROOT_ID, index 0) so the
        // canvas-root render lane is manifest-gated like any page.
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
        let m = assess(&s, &AssessOptions::default());
        assert_eq!(m.pages.len(), 1);
        assert_eq!(m.pages[0].page_index, 0);
        assert_eq!(m.pages[0].page_id, crate::types::SCENE_ROOT_ID);
        assert!(m.renderable());
        assert_eq!(m.verdict.as_str(), "full");
    }

    #[test]
    fn pageless_canvas_root_unsupported_feature_gates() {
        // The new lane must not bypass gating: an unsupported feature on a
        // FLOATING node (raw-array chart data — the browser out-renders
        // native) declines the doc exactly as it would on a page.
        let s = Scene::load(&json!({
            "layoutMode": "freeform",
            "nodes": [{
                "id":"ch","type":"generated","generatedKind":"chart",
                "x":0.0,"y":0.0,"width":10.0,"height":10.0,"rotation":0,"scale":1,
                "renderMode":"native-children",
                "generatedSource":{"kind":"chart","chartVersion":1,"spec":{"type":"bar"},
                    "data":[{"Category":"A","Value":1}]}
            }],
            "pages": [],
            "canvasRootNodeIds": ["ch"]
        })).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert_eq!(m.pages.len(), 1);
        let a = find_axis(&m, "generated:chart").expect("floating chart is assessed");
        assert_eq!(a.status, AxisStatus::Unsupported);
        assert!(!m.renderable());
    }

    #[test]
    fn paged_doc_gets_no_synthetic_canvas_root_surface() {
        // With pages present the server export lanes render pages only —
        // floating nodes never reach their output, so no synthetic surface
        // (and no gating from off-page content) is added.
        let s = Scene::load(&json!({
            "layoutMode": "freeform",
            "nodes": [
                {"id":"onpage","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#f00"}}]},
                {"id":"floata","type":"rectangle","x":900.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#0f0"}}]}
            ],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                       "nodeIds":["onpage"],"freeformX":0.0,"freeformY":0.0}],
            "canvasRootNodeIds": ["floata"]
        })).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert_eq!(m.pages.len(), 1);
        assert_eq!(m.pages[0].page_id, "p1");
    }

    #[test]
    fn genuinely_empty_scene_has_no_synthetic_surface() {
        let s = Scene::load(&json!({"nodes": [], "pages": [], "canvasRootNodeIds": []})).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert!(m.pages.is_empty());
    }

    #[test]
    fn chart_unregistered_type_full_with_dropped_detail() {
        // A spec type outside the ported/production registry has no generator
        // in EITHER lane — both renderers draw the identical placeholder rect
        // (charts::unknown_type_yields_placeholder + the chart_oracle_parity
        // fixtures exercise the pixels), so the axis reports Full with a
        // `dropped` note per the Q-95 browser-parity precedent, and the doc
        // stays on the native lane. Ported types (bar/line/pie/scatter/area/
        // combo) contribute no axis at all.
        let chart_node = |spec: Value| {
            json!([{
                "id":"ch","type":"generated","generatedKind":"chart",
                "x":0.0,"y":0.0,"width":10.0,"height":10.0,"rotation":0,"scale":1,
                "renderMode":"native-children",
                "generatedSource":{"kind":"chart","chartVersion":1,"spec":spec,
                    "data":{"columns":[],"rows":[],"series":[],"categories":[],"valueRange":[0,0]}}
            }])
        };
        // an unregistered/unported type → Full + dropped detail (browser parity)
        let s = Scene::load(&page_with(
            chart_node(json!({"type":"radar"})),
            json!(["ch"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        let a = find_axis(&m, "generated:chart").unwrap();
        assert_eq!(a.status, AxisStatus::Full);
        assert_eq!(a.detail.get("spec_type"), Some(&json!("radar")));
        assert!(a
            .detail
            .get("dropped")
            .and_then(Value::as_str)
            .is_some_and(|d| d.contains("browser parity")));
        assert_eq!(a.node_ids, vec!["ch".to_string()]);
        assert!(m.renderable()); // no exit-2 / browser routing for this class
        assert_eq!(m.verdict.as_str(), "full");

        // a PORTED type with an UNCOERCIBLE raw array still GATES: the keys
        // are non-standard (capitalized), so the TS rowsToParsedChartData
        // THROWS — scene-load normalization keeps the array (fail closed)
        // and the doc routes to the browser rather than rendering wrong data.
        let s = Scene::load(&page_with(
            json!([{
                "id":"ch","type":"generated","generatedKind":"chart",
                "x":0.0,"y":0.0,"width":10.0,"height":10.0,"rotation":0,"scale":1,
                "renderMode":"native-children",
                "generatedSource":{"kind":"chart","chartVersion":1,
                    "spec":{"type":"bar","config":{"palette":["#123456"]}},
                    "data":[{"Category":"A","Value":1}]}
            }]),
            json!(["ch"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        let a = find_axis(&m, "generated:chart").unwrap();
        assert_eq!(a.status, AxisStatus::Unsupported);
        assert_eq!(a.detail.get("reason"), Some(&json!("raw-array-data")));
        assert!(!m.renderable());

        // a COERCIBLE raw row array (the agent-written shape) with a
        // generator-safe config is normalized into the ParsedData bundle at
        // scene load and renders natively — no generated:chart axis at all,
        // verdict full. (Config-less variants of the same shape land in the
        // missing-config classification below instead — the browser coerces
        // at hydration and then its generator throws on the missing config.)
        let s = Scene::load(&page_with(
            json!([{
                "id":"ch","type":"generated","generatedKind":"chart",
                "x":0.0,"y":0.0,"width":10.0,"height":10.0,"rotation":0,"scale":1,
                "renderMode":"native-children",
                "generatedSource":{"kind":"chart","chartVersion":1,
                    "spec":{"type":"bar","config":{"palette":["#123456"]}},
                    "data":[{"category":"Q1","value":70},{"category":"Q2","value":80}]}
            }]),
            json!(["ch"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert!(find_axis(&m, "generated:chart").is_none());
        assert!(m.renderable());
        assert_eq!(m.verdict.as_str(), "full");

        // a PORTED type with real bundle rows but a MISSING `spec.config`:
        // the browser generator throws on its first config read and renders
        // the placeholder; native mirrors that placeholder pre-dispatch.
        // Identical output in both lanes ⇒ Full / non-gating — but the axis
        // stays OBSERVABLE (this shape used to render as a silent ~1px blank
        // with a zero-axis FULL manifest).
        let s = Scene::load(&page_with(
            json!([{
                "id":"ch","type":"generated","generatedKind":"chart",
                "x":0.0,"y":0.0,"width":10.0,"height":10.0,"rotation":0,"scale":1,
                "renderMode":"native-children",
                "generatedSource":{"kind":"chart","chartVersion":1,"spec":{"type":"bar"},
                    "data":{"columns":["category","value"],
                            "rows":[{"category":"Q1","value":120.0}],
                            "series":[],"categories":["Q1"],"valueRange":[0,120]}}
            }]),
            json!(["ch"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        let a = find_axis(&m, "generated:chart").unwrap();
        assert_eq!(a.status, AxisStatus::Full);
        assert_eq!(a.detail.get("reason"), Some(&json!("missing-config")));
        assert!(a
            .detail
            .get("dropped")
            .and_then(Value::as_str)
            .is_some_and(|d| d.contains("browser parity")));
        assert!(m.renderable());
        assert_eq!(m.verdict.as_str(), "full");

        // CONDITIONALLY divergent config (ChartConfigIssue::BrowserSafer):
        // palette missing but seriesStyles present — the browser may render a
        // REAL chart via the `seriesStyle?.fill ||` short-circuit
        // (barChart.ts:346), so this must GATE to the browser (Unsupported),
        // never claim placeholder parity.
        let s = Scene::load(&page_with(
            json!([{
                "id":"ch","type":"generated","generatedKind":"chart",
                "x":0.0,"y":0.0,"width":10.0,"height":10.0,"rotation":0,"scale":1,
                "renderMode":"native-children",
                "generatedSource":{"kind":"chart","chartVersion":1,
                    "spec":{"type":"bar","config":{"fontSize":12}},
                    "seriesStyles":{"__single__":{"fill":"#ff0000"}},
                    "data":{"columns":["category","value"],
                            "rows":[{"category":"Q1","value":120.0}],
                            "series":[],"categories":["Q1"],"valueRange":[0,120]}}
            }]),
            json!(["ch"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        let a = find_axis(&m, "generated:chart").unwrap();
        assert_eq!(a.status, AxisStatus::Unsupported);
        assert_eq!(a.detail.get("reason"), Some(&json!("conditional-config")));
        assert_eq!(a.detail.get("missing"), Some(&json!("palette")));
        assert!(!m.renderable());

        // time X axis on a ported type → ported now (no generated:chart axis)
        let s = Scene::load(&page_with(
            chart_node(json!({"type":"line","xScale":"time"})),
            json!(["ch"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert!(find_axis(&m, "generated:chart").is_none());

        // ported scatter → no generated:chart axis
        let s = Scene::load(&page_with(
            chart_node(json!({"type":"scatter"})),
            json!(["ch"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert!(find_axis(&m, "generated:chart").is_none());

        // ported combo → no generated:chart axis
        let s = Scene::load(&page_with(
            chart_node(json!({"type":"combo"})),
            json!(["ch"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert!(find_axis(&m, "generated:chart").is_none());

        // combo + time X axis is ported too (no longer gated) — the no-bar arm
        // uses the time scale; a bar+time combo falls back to categorical.
        let s = Scene::load(&page_with(
            chart_node(json!({"type":"combo","xScale":"time"})),
            json!(["ch"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert!(find_axis(&m, "generated:chart").is_none());
    }

    #[test]
    fn generated_placeholder_missing_asset_is_parity_full() {
        // A latex generated node with NO cachedAssetUrl expands to the
        // placeholder rect in BOTH renderers (expansion.rs:create_placeholder_rect
        // == generatedGraphicExpansion.ts) and no
        // lane regenerates the asset at render time — so the axis is
        // parity-Full and must NOT push export lanes to the browser
        // (the accepts_degraded=False policies: PDF / node-raster / animation).
        let s = Scene::load(&page_with(
            json!([{
                "id":"lx1","type":"generated","generatedKind":"latex",
                "x":0.0,"y":0.0,"width":50.0,"height":50.0,"rotation":0,"scale":1,
                "renderMode":"raster-asset",
                "generatedSource":{"kind":"latex","latex":"x","displayMode":"display","fontSize":32}
            }]),
            json!(["lx1"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        // The axis stays in the manifest (missing-asset observability)…
        let a = find_axis(&m, "generated:placeholder").unwrap();
        assert_eq!(a.status, AxisStatus::Full);
        assert_eq!(a.node_ids, vec!["lx1".to_string()]);
        assert_eq!(
            a.detail.get("reason"),
            Some(&json!("missing cachedAssetUrl → placeholder"))
        );
        assert!(a
            .detail
            .get("parity")
            .and_then(Value::as_str)
            .is_some_and(|p| p.starts_with("parity-full")));
        // …but the verdict is full: the doc renders natively on every lane.
        assert_eq!(m.verdict.as_str(), "full");
        assert_eq!(m.pages[0].verdict.as_str(), "full");
        assert!(m.renderable());
    }

    #[test]
    fn generated_with_cached_asset_emits_no_placeholder_axis() {
        // With a cachedAssetUrl present the node renders as a pattern rect —
        // no generated:placeholder axis (unchanged behavior).
        let s = Scene::load(&page_with(
            json!([{
                "id":"lx1","type":"generated","generatedKind":"latex",
                "x":0.0,"y":0.0,"width":50.0,"height":50.0,"rotation":0,"scale":1,
                "renderMode":"raster-asset",
                "cachedAssetUrl":"https://assets/latex.png",
                "generatedSource":{"kind":"latex","latex":"x","displayMode":"display","fontSize":32}
            }]),
            json!(["lx1"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert!(find_axis(&m, "generated:placeholder").is_none());
        assert_eq!(m.verdict.as_str(), "full");
    }

    #[test]
    fn blend_node_supported_since_m4b() {
        let s = Scene::load(&page_with(
            json!([
                {"id":"a","type":"rectangle","x":0.0,"y":0.0,"width":1.0,"height":1.0,"rotation":0,"scale":1},
                {"id":"b","type":"rectangle","x":5.0,"y":5.0,"width":1.0,"height":1.0,"rotation":0,"scale":1},
                {"id":"bl","type":"blend","sourceIds":["a","b"],"spacing":"steps","steps":2}
            ]),
            json!(["a", "b", "bl"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        // M4b un-gated blend nodes: the expansion (crate::blend) renders
        // them, so the manifest reports no node:blend axis and the page
        // stays renderable.
        assert!(find_axis(&m, "node:blend").is_none());
        assert!(m.renderable());
    }

    #[test]
    fn unknown_node_field_is_observed_but_does_not_decline() {
        // Policy (owner-directed, 2026-07-04): an unrecognized node FIELD must NOT
        // decline a doc to the browser. `tests/scene_schema_parity.rs` is a CI gate
        // that forces the field tables to mirror every `nodes.ts` field, so a field
        // that spills here is provably non-canonical (foreign / legacy / agent-lane)
        // and is ignored identically by BOTH renderers — native == browser pixels.
        // The axis is still emitted (drift observability) but is non-elevating.
        let s = Scene::load(&page_with(
            json!([{"id":"r","type":"rectangle","x":0.0,"y":0.0,"width":1.0,"height":1.0,
                "rotation":0,"scale":1,"futureProp":7}]),
            json!(["r"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        // Observability: the axis is still recorded, with its field detail…
        let a = find_axis(&m, "unknown:node-field").unwrap();
        assert_eq!(a.status, AxisStatus::Unknown);
        assert_eq!(a.detail.get("field"), Some(&json!("futureProp")));
        // …but it does NOT elevate the verdict — the doc renders natively.
        assert_eq!(m.verdict.as_str(), "full");
        assert!(m.renderable());
    }

    #[test]
    fn svg_pattern_fill_warns_but_never_gates() {
        // Contract lock: an SVG pattern fill emits the `svg:pattern-image`
        // WARNING (observability breadcrumb) and NOTHING else — no axis entry,
        // so it structurally cannot elevate the page/doc verdict or decline
        // the doc to the browser lane. Both lanes rasterize SVG via the same
        // resvg pipeline (`moda_assets::svg::prepare_svg`); render fidelity is
        // exercised by moda-assets' svg.rs / fetch.rs pixel tests.
        let s = Scene::load(&page_with(
            json!([{"id":"r","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"image","src":"https://x/icon.svg"}}]}]),
            json!(["r"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        // No axis of any kind — the page is axis-empty…
        assert!(m.pages[0].axes.is_empty());
        // …the warning is present with asset attribution…
        let w = m.pages[0]
            .warnings
            .iter()
            .find(|w| w.axis == "svg:pattern-image")
            .expect("svg:pattern-image warning");
        assert_eq!(w.detail.get("asset"), Some(&json!("https://x/icon.svg")));
        // …and the verdict stays Full / renderable (native lane).
        assert_eq!(m.pages[0].verdict, AxisStatus::Full);
        assert_eq!(m.verdict.as_str(), "full");
        assert!(m.renderable());

        // The URL heuristic covers data: and querystring forms too…
        assert!(is_svg_url("data:image/svg+xml;base64,PHN2Zz4="));
        assert!(is_svg_url("https://x/a.svg?v=2"));
        assert!(is_svg_url("https://x/a.SVG#frag"));
        // …and stays silent for raster sources.
        assert!(!is_svg_url("https://x/photo.png"));
        assert!(!is_svg_url("https://x/photo.png?name=a.svg"));
    }

    #[test]
    fn unknown_node_type_does_not_gate_verdict() {
        // A node whose `type` is outside VALID_NODE_TYPES is dropped at
        // hydration by BOTH renderers (scene.rs `hydrate_document` `continue`;
        // Scene.ts isValidNodeType filters at 693/1278/2090), and the CI roster
        // pin (tests/scene_schema_parity.rs) proves such a type is not in the
        // shipped TS union either. Native output == browser output (both render
        // the doc without the node), so the axis must be observed but must NOT
        // elevate the verdict / decline the doc to the browser.
        let rect = json!({"id":"n1","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
            "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#123456"}}]});
        let s = Scene::load(&json!({
            "nodes": [rect.clone(), {"id":"n2","type":"sticker3d","x":0.0,"y":0.0}],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                       "nodeIds":["n1","n2"]}]
        })).unwrap();
        let m = assess(&s, &AssessOptions::default());
        // Observability: the axis is still recorded, with the node-type detail…
        let a = find_axis(&m, "unknown:node-type").unwrap();
        assert_eq!(a.status, AxisStatus::Unknown);
        assert_eq!(a.detail.get("node_type"), Some(&json!("sticker3d")));
        assert_eq!(a.node_ids, vec!["n2".to_string()]);
        // …but it does NOT elevate the verdict — the doc renders natively.
        // (`verdict == full` is the load-bearing assertion: router.py gates on
        // verdict ∈ {full, degraded}.)
        assert_eq!(m.verdict.as_str(), "full");
        assert!(m.renderable());

        // Pixel evidence (CPU oracle): the foreign node is dropped at hydration,
        // so the display list — the sole input plan building walks — is
        // IDENTICAL to the same doc without the node. Identical display lists ⇒
        // identical plans ⇒ identical pixels; the flip cannot change output.
        assert!(s.node("n2").is_none());
        let s_without = Scene::load(&json!({
            "nodes": [rect],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                       "nodeIds":["n1","n2"]}]
        })).unwrap();
        let ids = |sc: &Scene| -> Vec<String> {
            crate::display_list::build_display_list(sc, "p1")
                .items
                .iter()
                .map(|i| i.node_id.clone())
                .collect()
        };
        assert_eq!(ids(&s), ids(&s_without));

        // KNOWN legacy drops ('text'/'image', Scene.ts `isValidNodeType`) stay axis-free:
        // no unknown:node-type entry, verdict full.
        let s2 = Scene::load(&json!({
            "nodes": [{"id":"t1","type":"text","x":0.0,"y":0.0},
                      {"id":"i1","type":"image","x":0.0,"y":0.0}],
            "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                       "nodeIds":["t1","i1"]}]
        })).unwrap();
        let m2 = assess(&s2, &AssessOptions::default());
        assert!(find_axis(&m2, "unknown:node-type").is_none());
        assert_eq!(m2.verdict.as_str(), "full");
    }

    #[test]
    fn instancer_node_is_an_unknown_node_type() {
        // The inert `instancer` kind was retired (live-nodes.md §3.3/§14.5): a doc
        // that still carries one meets it as any other foreign type — dropped at
        // hydration, reported as `unknown:node-type`, no crash.
        let s = Scene::load(&page_with(
            json!([
                {"id":"r","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#123456"}}]},
                {"id":"inst1","type":"instancer","x":10.0,"y":20.0,"width":40.0,"height":40.0,
                 "count":12,"distribution":{"kind":"ring","radius":30.0}}
            ]),
            json!(["r", "inst1"]),
        )).unwrap();
        assert!(s.node("inst1").is_none());
        assert!(s.node("r").is_some());
        let m = assess(&s, &AssessOptions::default());
        let a = find_axis(&m, "unknown:node-type").expect("unknown:node-type axis");
        assert_eq!(a.detail.get("node_type"), Some(&json!("instancer")));
        assert_eq!(a.node_ids, vec!["inst1".to_string()]);
        assert!(find_axis(&m, "node:instancer").is_none());
    }

    #[test]
    fn format_26_motion_fields_load_and_draw_the_static_design() {
        // Format 26 (motion-model.md §2.1): a frame's and a slide's `graph`, a layer's reserved `transform3d` and the
        // reserved `camera` node type; format 28's flipbook (a group with `flipbook`, its drawings with `hold`). The
        // document loads with a clean manifest and draws its static design: the graph and transform3d are ignored,
        // the camera draws nothing yet, and the flipbook draws its drawing at rest (`current`, here the second).
        let graph = json!({
            "v": 1,
            "timeline": {"duration": {"value": "1411200000", "timescale": 141120000}, "rate": {"num": 30, "den": 1},
                         "markers": [{"at": {"value": "0", "timescale": 141120000}, "name": "go", "kind": "stop"}]},
            "nodes": [{"id": "fade", "type": "keys", "params": {"value": {"keys": [
                {"at": {"value": "0", "timescale": 141120000}, "v": 0.0},
                {"at": {"value": "70560000", "timescale": 141120000}, "v": 1.0, "ease": "out"}]}}}],
            "bindings": [{"target": {"layer": "r", "path": "opacity"}, "op": "set", "from": {"node": "fade"}}],
            "camera": "cam"
        });
        let solid = json!([{"id":"fill-0","paint":{"type":"solid","color":"#123456"}}]);
        let mut doc = page_with(
            json!([
                {"id":"f","type":"container","x":0.0,"y":0.0,"width":80.0,"height":80.0,"rotation":0,"scale":1,
                 "clipContent":true,"children":["r","cam"],"fills":solid,"graph":graph},
                {"id":"r","type":"rectangle","x":10.0,"y":10.0,"width":10.0,"height":10.0,"rotation":0,"scale":1,
                 "fills":solid,"transform3d":{"z":40.0,"rotation":{"x":10.0,"y":0.0,"z":0.0},"anchor":{"x":0.5,"y":0.5,"z":0.0}}},
                {"id":"cam","type":"camera","position":{"x":0.0,"y":0.0,"z":-500.0},
                 "pointOfInterest":{"x":0.0,"y":0.0,"z":0.0},"zoom":1.0},
                {"id":"fb","type":"group","x":0.0,"y":0.0,"rotation":0,"scaleX":1.0,"scaleY":1.0,
                 "flipbook":{"mode":"pingpong","current":1},"onionSkin":{"before":1,"after":0},"children":["d1","d2"]},
                {"id":"d1","type":"group","x":0.0,"y":0.0,"rotation":0,"scaleX":1.0,"scaleY":1.0,"children":["s1"],"hold":2},
                {"id":"s1","type":"rectangle","x":0.0,"y":0.0,"width":5.0,"height":5.0,"rotation":0,"scale":1,"fills":solid},
                {"id":"d2","type":"group","x":0.0,"y":0.0,"rotation":0,"scaleX":1.0,"scaleY":1.0,"children":[],"hold":3}
            ]),
            json!(["f", "fb"]),
        );
        doc["pages"][0]["graph"] = graph.clone();
        // At rest, without the onion skin (a preview aid).
        doc["nodes"][3].as_object_mut().unwrap().remove("onionSkin");
        let s = Scene::load(&doc).unwrap();
        assert!(s.node("cam").is_none() && s.node("fb").is_some());
        assert!(s.doc.unknown_node_types.is_empty(), "{:?}", s.doc.unknown_node_types);
        let m = assess(&s, &AssessOptions::default());
        assert!(find_axis(&m, "unknown:node-type").is_none());
        assert!(find_axis(&m, "unknown:node-field").is_none(), "{:?}", m.pages[0].axes);
        assert_eq!(m.verdict.as_str(), "full");
        let drawn: Vec<String> = crate::display_list::build_display_list(&s, "p1")
            .items
            .iter()
            .map(|i| i.node_id.clone())
            .collect();
        assert!(drawn.contains(&"r".to_string()), "{drawn:?}");
        assert!(!drawn.iter().any(|id| ["cam", "d1", "s1"].contains(&id.as_str())), "{drawn:?}");
        // The flipbook's second drawing (an empty group) is current: nothing of the first draws.
        let z = s.flattened_z_order_for_ids(&["fb".to_string()]);
        assert_eq!(z, vec!["fb".to_string(), "d2".to_string()]);
        // With an onion skin, the first drawing's copy draws behind it.
        let mut onion = doc.clone();
        onion["nodes"][3]["onionSkin"] = json!({"before": 1, "after": 0});
        let so = Scene::load(&onion).unwrap();
        let z = so.flattened_z_order_for_ids(&["fb".to_string()]);
        assert_eq!(z, vec!["fb", "fb~onion-1", "d1~onion-1", "s1~onion-1", "d2"]);
        assert!(find_axis(&assess(&so, &AssessOptions::default()), "unknown:node-field").is_none());

        // The same design without any of it draws the same list.
        let plain = page_with(
            json!([
                {"id":"f","type":"container","x":0.0,"y":0.0,"width":80.0,"height":80.0,"rotation":0,"scale":1,
                 "clipContent":true,"children":["r"],"fills":solid},
                {"id":"r","type":"rectangle","x":10.0,"y":10.0,"width":10.0,"height":10.0,"rotation":0,"scale":1,"fills":solid}
            ]),
            json!(["f"]),
        );
        let plain_drawn: Vec<String> = crate::display_list::build_display_list(&Scene::load(&plain).unwrap(), "p1")
            .items
            .iter()
            .map(|i| i.node_id.clone())
            .collect();
        // Beside the flipbook (whose current drawing is an empty group), the same list.
        let drawn: Vec<String> = drawn.into_iter().filter(|id| id != "fb" && id != "d2").collect();
        assert_eq!(drawn, plain_drawn);
    }

    #[test]
    fn manifest_json_marks_informational_axes_non_gating() {
        // The router's decision summarizer attributes a decline to the
        // max-severity GATING axis; without this bit it cannot tell an
        // informational axis (never declines) from a gating one, and `unknown`
        // severity outranks `degraded` — misattributing real declines to e.g.
        // `unknown:node-field`. Assert the bit per axis class.
        let axis_gating = |m: &Manifest, axis: &str| -> bool {
            let j = manifest_to_json(m);
            j["pages"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|p| p["axes"].as_array().unwrap())
                .find(|a| a["axis"] == json!(axis))
                .unwrap_or_else(|| panic!("axis {axis} not in manifest"))["gating"]
                .as_bool()
                .unwrap_or_else(|| panic!("axis {axis} missing bool gating bit"))
        };

        // Informational: an unrecognized node field ⇒ gating=false.
        let s = Scene::load(&page_with(
            json!([{"id":"r","type":"rectangle","x":0.0,"y":0.0,"width":1.0,"height":1.0,
                "rotation":0,"scale":1,"futureProp":7}]),
            json!(["r"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert!(!axis_gating(&m, "unknown:node-field"));

        // Informational: an unrecognized node TYPE ⇒ gating=false (this train's
        // flip — both lanes drop the node at hydration).
        let s = Scene::load(&page_with(
            json!([{"id":"h","type":"hologram","x":0.0,"y":0.0}]),
            json!(["h"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert!(!axis_gating(&m, "unknown:node-type"));

        // Informational: an unknown shader id ⇒ gating=false (native
        // draws the identical white box the browser would with the same engine,
        // so declining bought nothing; the shader_drift_axis WARNING is the
        // remaining signal).
        let s = Scene::load(&page_with(
            json!([{
                "id":"r","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                "rotation":0,"scale":1,
                "fills":[{"id":"fill-0","paint":{"type":"shader","shaderId":"totally-unknown-shader"}}]
            }]),
            json!(["r"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert!(!axis_gating(&m, "shader:totally-unknown-shader"));
        // …and its presence no longer elevates the verdict — the doc serves.
        assert_eq!(m.verdict.as_str(), "full");
        assert!(m.renderable());

        // Gating: unknown:generated-kind (Degraded, still elevating) ⇒ true —
        // and generated:chart, emitted only at Full now, carries true too.
        let s = Scene::load(&page_with(
            json!([{
                "id":"g","type":"generated","generatedKind":"future-widget",
                "x":0.0,"y":0.0,"width":10.0,"height":10.0,"rotation":0,"scale":1,
                "renderMode":"native-children"
            }]),
            json!(["g"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert!(axis_gating(&m, "unknown:generated-kind"));
    }

    #[test]
    fn foreign_fields_render_but_genuine_feature_gaps_still_gate() {
        // The prod residual (2026-07-04, native_render.decision manifest_unsupported):
        // recognized node types carrying FOREIGN, non-`nodes.ts` keys — `fillColor` /
        // `tint` / `parentId` / `fillPatternFit`, and `resizeMode` on a NON-richtext
        // node (a legacy / agent-optimizer shape). Neither renderer reads these keys,
        // so the doc must render natively rather than burn a Chromium fallback.
        let s = Scene::load(&page_with(
            json!([
                {"id":"rect-1","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#123456"}}],
                 "fillColor":"#abcdef","tint":0.5,"parentId":"grp-9","resizeMode":"fit",
                 "fillPatternFit":"cover"},
                {"id":"line-1","type":"line","points":[0.0,0.0,5.0,5.0],
                 "strokes":[{"id":"stroke-0","paint":{"type":"solid","color":"#000"},"width":1.0}],"height":5.0,"width":5.0}
            ]),
            json!(["rect-1", "line-1"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        // The foreign fields ARE observed (axis emitted for drift monitoring)…
        assert_eq!(
            find_axis(&m, "unknown:node-field").unwrap().status,
            AxisStatus::Unknown
        );
        // …but the doc renders natively — zero browser decline.
        assert_eq!(m.verdict.as_str(), "full");
        assert!(m.renderable());

        // A GENUINE capability gap — an UNSUPPORTED axis (raw-array chart data the
        // browser out-renders) on a foreign-field node — STILL gates: the policy
        // relaxes unrecognized FIELDS (and unknown shaders that
        // white-box identically in both engines), never a feature the browser
        // renders MORE of than native.
        let s2 = Scene::load(&page_with(
            json!([{
                "id":"ch","type":"generated","generatedKind":"chart","tint":0.5,
                "x":0.0,"y":0.0,"width":10.0,"height":10.0,"rotation":0,"scale":1,
                "renderMode":"native-children",
                "generatedSource":{"kind":"chart","chartVersion":1,
                    "spec":{"type":"bar","config":{"palette":["#123456"]}},
                    "data":[{"Category":"A","Value":1}]}
            }]),
            json!(["ch"]),
        )).unwrap();
        let m2 = assess(&s2, &AssessOptions::default());
        // Verdict is driven by the unsupported chart (a genuinely elevating axis).
        assert_eq!(
            find_axis(&m2, "generated:chart").unwrap().status,
            AxisStatus::Unsupported
        );
        assert!(!m2.renderable());
    }

    #[test]
    fn unknown_generated_kind_is_degraded_placeholder() {
        // An unrecognized (or absent) `generatedKind` expands to the SAME
        // placeholder rect in both lanes (expansion.rs `create_placeholder_rect`
        // ↔ generatedGraphicExpansion.ts default branch, #f0f0f0/#cccccc), so
        // the honest verdict is Degraded — renderable natively — not Unknown
        // (which false-declines the whole doc to Chromium; the 2026-07 glass
        // incident class). It stays ELEVATING (Degraded, not Full) so strict
        // export consumers keep their fail-closed window for any future
        // TS-first 5th GeneratedGraphicKind.
        let gen_node = |extra: Value| {
            let mut n = json!({
                "id":"g","type":"generated",
                "x":0.0,"y":0.0,"width":10.0,"height":10.0,"rotation":0,"scale":1
            });
            n.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            json!([n])
        };

        // Unrecognized kind → Degraded placeholder, kind preserved in detail.
        let s = Scene::load(&page_with(
            gen_node(json!({"generatedKind":"weird"})),
            json!(["g"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        let a = find_axis(&m, "unknown:generated-kind").unwrap();
        assert!(matches!(a.status, AxisStatus::Degraded { .. }));
        assert_eq!(a.detail.get("kind"), Some(&json!("weird")));
        assert_eq!(m.verdict.as_str(), "degraded");
        assert!(m.renderable());

        // Absent kind (malformed doc) → same Degraded placeholder verdict.
        let s2 = Scene::load(&page_with(gen_node(json!({})), json!(["g"]))).unwrap();
        let m2 = assess(&s2, &AssessOptions::default());
        let a2 = find_axis(&m2, "unknown:generated-kind").unwrap();
        assert!(matches!(a2.status, AxisStatus::Degraded { .. }));
        assert_eq!(m2.verdict.as_str(), "degraded");
        assert!(m2.renderable());
    }

    #[test]
    fn benign_metadata_fields_do_not_force_unknown_verdict() {
        // Regression (live-prod #1 fallback cause): a collab-projected doc carries
        // benign top-level `name` / `schemaVersion` / `kind` / … metadata and a
        // node with a top-level `name` + node-level `pivotX`/`pivotY`. These are
        // non-render-affecting (or native-supported), so the walker must NOT emit
        // any `unknown:*` axis — an `unknown` verdict makes the native-render
        // router false-decline the whole doc to Chromium (router.py: verdict not
        // in {full, degraded}). The doc has only a solid-fill rect, so the honest
        // verdict is `full`.
        let s = Scene::load(&json!({
            "name": "My Canvas",
            "schemaVersion": 7,
            "kind": "canvas",
            "documentType": "design",
            "width": 1080,
            "height": 1080,
            // Arbitrary document-metadata blob (e.g. PPTX import) parked in the
            // projection's `meta` catchall — benign, never read by the walker.
            "metadata": {"title": "Deck", "author": "someone"},
            "nodes": [{
                "id": "r", "type": "rectangle",
                "x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0,
                "rotation": 0, "scale": 1, "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ff0000"}}],
                // `originalId`: a duplicated node's source-node reference —
                // bookkeeping only, no render effect.
                "name": "Rect 1", "pivotX": 0.25, "pivotY": 0.75,
                "originalId": "orig-r"
            }],
            "pages": [{"id":"p1","name":"P","width":1080.0,"height":1080.0,
                       "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":["r"]}]
        })).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert!(find_axis(&m, "unknown:envelope-field").is_none());
        assert!(find_axis(&m, "unknown:node-field").is_none());
        assert_eq!(m.verdict.as_str(), "full");
        assert!(m.renderable());

        // Control: a genuinely-unrecognized ENVELOPE key is still RECORDED (its field
        // detail is preserved for drift observability) but no longer declines (owner
        // policy 2026-07-04, now that the parity gate covers the `getSceneState()`
        // serializer surface too — `scene_schema_parity.rs`). A key spilling here is
        // produced by no canonical serializer, so it is foreign/legacy/agent-lane and
        // ignored identically by both renderers; the honest verdict is `full`. Any
        // *canonical* top-level key silently going unmirrored is caught at build time by
        // that gate, not by a runtime decline.
        let s2 = Scene::load(&json!({
            "totallyNewEnvelopeKey": {"v": 1},
            "nodes": [], "pages": [{"id":"p1","name":"P","width":10.0,"height":10.0,
                       "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],"nodeIds":[]}]
        })).unwrap();
        let m2 = assess(&s2, &AssessOptions::default());
        let env_axis = find_axis(&m2, "unknown:envelope-field").unwrap();
        assert_eq!(env_axis.status, AxisStatus::Unknown);
        assert_eq!(
            env_axis.detail.get("field"),
            Some(&json!("totallyNewEnvelopeKey"))
        );
        assert_eq!(m2.verdict.as_str(), "full");
        assert!(m2.renderable());
    }

    #[test]
    fn internal_pattern_image_info_field_does_not_force_unknown_verdict() {
        // Regression (live-prod #1 fallback cause, 2026-07-03): a pattern-image
        // fill node carries `_patternImageInfo` — an internal, non-render-affecting
        // annotation the image placement path stashes for pattern-editing UX
        // (ImageService.ts). It must NOT emit `unknown:node-field` (an `unknown`
        // verdict false-declines the whole doc to Chromium: router.py verdict not
        // in {full, degraded}). The node is a solid-fill rect, so verdict is `full`.
        let s = Scene::load(&page_with(
            json!([{
                "id":"r","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ff0000"}}],
                "_patternImageInfo":{"width":221,"height":247}
            }]),
            json!(["r"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert!(find_axis(&m, "unknown:node-field").is_none());
        assert_eq!(m.verdict.as_str(), "full");
        assert!(m.renderable());

        // Control: an arbitrary unrecognized `_`-field is still RECORDED (its field
        // detail is preserved for drift observability) but no longer declines (owner
        // policy 2026-07-04). Any *canonical* `_`-field silently going unmirrored is
        // caught at build time by `scene_schema_parity.rs`, not by a runtime decline.
        let s2 = Scene::load(&page_with(
            json!([{
                "id":"r","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ff0000"}}],"_futureInternalProp":7
            }]),
            json!(["r"]),
        )).unwrap();
        let m2 = assess(&s2, &AssessOptions::default());
        assert_eq!(
            find_axis(&m2, "unknown:node-field")
                .unwrap()
                .detail
                .get("field"),
            Some(&json!("_futureInternalProp"))
        );
        assert_eq!(m2.verdict.as_str(), "full");
        assert!(m2.renderable());
    }

    #[test]
    fn known_shader_full_alias_resolved_unknown_shader_flagged() {
        let s = Scene::load(&page_with(
            json!([
                {"id":"s1","type":"rectangle","x":0.0,"y":0.0,"width":1.0,"height":1.0,"rotation":0,"scale":1,
                 "fills":[{"id":"fill-0","paint":{"type":"shader","shaderId":"mesh-spot"}}]},
                {"id":"s2","type":"rectangle","x":0.0,"y":0.0,"width":1.0,"height":1.0,"rotation":0,"scale":1,
                 "fills":[{"id":"fill-0","paint":{"type":"shader","shaderId":"never-heard-of-it"}}]},
                {"id":"s3","type":"rectangle","x":0.0,"y":0.0,"width":1.0,"height":1.0,"rotation":0,"scale":1,
                 "fills":[{"id":"fill-0","paint":{"type":"shader","shaderId":"heatmap-image"}}]}
            ]),
            json!(["s1", "s2", "s3"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        // alias mesh-spot → mesh-gradient-advanced: known, not gated → no axis
        assert!(find_axis(&m, "shader:mesh-gradient-advanced").is_none());
        // unknown id → Unknown
        assert_eq!(
            find_axis(&m, "shader:never-heard-of-it").unwrap().status,
            AxisStatus::Unknown
        );
        // M4e: image-backed trio is un-gated → known, not gated → no axis
        assert!(find_axis(&m, "shader:heatmap-image").is_none());
    }

    fn find_warning<'a>(m: &'a Manifest, axis: &str) -> Option<&'a ManifestWarning> {
        m.pages[0].warnings.iter().find(|w| w.axis == axis)
    }

    #[test]
    fn richtext_shader_fill_uses_browser_eligibility_predicate() {
        // M4 "richtext shader fills": eligible descriptor shaders render via
        // the extracted `<id>.msdf-text-fill` module → Full (no axis entry);
        // the image-backed trio fails `supportsMsdfTextFill` in the browser
        // too (per-run-color fallback, which the engine byte-copies) → NO
        // gating axis, just a `shader:text-fill-fallback` warning; unknown
        // ids stay Unknown.
        let s = Scene::load(&page_with(
            json!([
                {"id":"t1","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,
                 "htmlContent":"<p>shader</p>",
                 "textFills":[{"id":"fill-0","paint":{"type":"shader","shaderId":"light-rays"}}]},
                {"id":"t2","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,
                 "htmlContent":"<p>shader</p>",
                 "textFills":[{"id":"fill-0","paint":{"type":"shader","shaderId":"dither-image"}}]},
                {"id":"t3","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,
                 "htmlContent":"<p>shader</p>",
                 "textFills":[{"id":"fill-0","paint":{"type":"shader","shaderId":"mesh-spot"}}]},
                {"id":"t4","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,
                 "htmlContent":"<p>shader</p>",
                 "textFills":[{"id":"fill-0","paint":{"type":"shader","shaderId":"no-such-shader"}}]}
            ]),
            json!(["t1", "t2", "t3", "t4"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        // Eligible (incl. via legacy alias) → Full → no axis entry.
        assert!(find_axis(&m, "shader:light-rays").is_none());
        assert!(find_axis(&m, "shader:mesh-gradient-advanced").is_none());
        // Image-backed → ineligible in BOTH lanes (browser per-run-color
        // fallback, byte-copied by the engine) → no gating axis, warning only.
        assert!(find_axis(&m, "shader:dither-image").is_none());
        let w = find_warning(&m, "shader:text-fill-fallback").unwrap();
        assert_eq!(w.detail.get("shader_id"), Some(&json!("dither-image")));
        assert_eq!(w.detail.get("text_fill"), Some(&json!(true)));
        assert!(w.detail.get("reason").is_some());
        // Unknown id stays Unknown (schema told us something new).
        assert_eq!(
            find_axis(&m, "shader:no-such-shader").unwrap().status,
            AxisStatus::Unknown
        );
    }

    #[test]
    fn richtext_trio_text_fill_is_renderable_with_warning() {
        // The incident class this flip eliminates: a doc whose ONLY special
        // feature is an image-backed-trio shader fill on richtext must stay
        // renderable (verdict full, no exit-2 / browser decline) — the engine
        // renders the SAME per-run-color fallback the browser does. The
        // warning keeps the fallback observable.
        let s = Scene::load(&page_with(
            json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                "rotation":0,"scale":1,
                "htmlContent":"<p>shader</p>",
                "textFills":[{"id":"fill-0","paint":{"type":"shader","shaderId":"liquid-metal-image"}}]}]),
            json!(["t"]),
        )).unwrap();
        // vector text engine: keeps the (orthogonal) MSDF-readiness text gate
        // out of the verdict so the assertion isolates the shader axis.
        let m = assess(
            &s,
            &AssessOptions {
                text_engine_is_vector: true,
                ..Default::default()
            },
        );
        assert!(m.renderable());
        assert_eq!(m.verdict.as_str(), "full");
        assert!(find_axis(&m, "shader:liquid-metal-image").is_none());
        let w = find_warning(&m, "shader:text-fill-fallback").unwrap();
        assert_eq!(
            w.detail.get("shader_id"),
            Some(&json!("liquid-metal-image"))
        );

        // Alias-resolved detail: a registry that aliases a legacy id onto a
        // trio member still resolves before the eligibility check and records
        // the alias on the warning.
        let mut registry = ShaderRegistry::default();
        registry.aliases.insert("legacy-liquid".into(),
            "liquid-metal-image".to_string(),
        );
        let s2 = Scene::load(&page_with(
            json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                "rotation":0,"scale":1,
                "htmlContent":"<p>shader</p>",
                "textFills":[{"id":"fill-0","paint":{"type":"shader","shaderId":"legacy-liquid"}}]}]),
            json!(["t"]),
        )).unwrap();
        let m2 = assess(
            &s2,
            &AssessOptions {
                shader_registry: Some(registry),
                text_engine_is_vector: true,
                ..Default::default()
            },
        );
        assert!(m2.renderable());
        assert_eq!(m2.verdict.as_str(), "full");
        let w2 = find_warning(&m2, "shader:text-fill-fallback").unwrap();
        assert_eq!(
            w2.detail.get("shader_id"),
            Some(&json!("liquid-metal-image"))
        );
        assert_eq!(w2.detail.get("alias_of"), Some(&json!("legacy-liquid")));
    }

    #[test]
    fn text_fallback_degraded_with_band() {
        // Non-catalog family + CJK both degrade with the 0.90–0.97 band
        // (DECISION §1.4).
        let s = Scene::load(&page_with(
            json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                "rotation":0,"scale":1,
                "htmlContent":"<p style=\"font-family: Mystery Sans\">你好</p>"}]),
            json!(["t"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        let a = find_axis(&m, "text:fallback-engine").unwrap();
        match &a.status {
            AxisStatus::Degraded { expected_ssim_band } => {
                assert_eq!(*expected_ssim_band, (0.90, 0.97));
            }
            other => panic!("expected degraded, got {other:?}"),
        }
        assert_eq!(a.detail.get("scripts"), Some(&json!(["Han"])));
    }

    #[test]
    fn vector_engine_never_degrades_text_on_font_msdf_readiness() {
        // A non-catalog LATIN family degrades under the msdf-era default (no MSDF
        // atlas) but assesses FULL under vector: vector draws the TTF outlines
        // directly, so MSDF-atlas readiness is irrelevant and the gate could only
        // false-decline renderable text.
        let s = Scene::load(&page_with(
            json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                "rotation":0,"scale":1,
                "htmlContent":"<p style=\"font-family: Mystery Sans\">Hello world</p>"}]),
            json!(["t"]),
        )).unwrap();
        // msdf-era default: degraded on the family gate.
        let dm = assess(&s, &AssessOptions::default());
        assert!(find_axis(&dm, "text:fallback-engine").is_some());
        // vector: the family gate is moot → full.
        let vm = assess(
            &s,
            &AssessOptions {
                text_engine_is_vector: true,
                ..Default::default()
            },
        );
        assert!(find_axis(&vm, "text:fallback-engine").is_none());
        assert_eq!(vm.verdict.as_str(), "full");
    }

    #[test]
    fn vector_engine_renders_non_latin_scripts_natively() {
        // Non-Latin script coverage renders CORRECTLY on the native vector path:
        // simple scripts (CJK) via the vector outline lane + Noto coverage
        // fallback, complex scripts (Arabic/Hebrew/Indic/Thai/…) via the
        // cosmic-text tier. So under vector they earn a FULL / renderable verdict
        // and no longer decline to the browser — while MSDF (which cannot render
        // them) keeps the script degrade.
        let cases = [
            "<p style=\"font-family: Inter\">你好</p>", // Han (simple)
            "<p style=\"font-family: Inter\">مرحبا</p>", // Arabic (complex/RTL)
            "<p style=\"font-family: Inter\">שלום</p>", // Hebrew (complex/RTL)
            "<p style=\"font-family: Inter\">வணக்கம்</p>", // Tamil (complex)
            "<p style=\"font-family: Inter\">สวัสดี</p>", // Thai (complex)
        ];
        for html in cases {
            let s = Scene::load(&page_with(
                json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                    "rotation":0,"scale":1, "htmlContent": html}]),
                json!(["t"]),
            )).unwrap();
            // Vector: renders natively → full, no fallback-engine decline.
            let vm = assess(
                &s,
                &AssessOptions {
                    text_engine_is_vector: true,
                    ..Default::default()
                },
            );
            assert!(
                find_axis(&vm, "text:fallback-engine").is_none(),
                "vector must render {html:?} natively (no decline)"
            );
            assert_eq!(vm.verdict.as_str(), "full", "{html:?}");
            assert!(vm.renderable(), "{html:?}");
            // MSDF: cannot render the script → keeps degrading.
            let mm = assess(&s, &AssessOptions::default());
            assert!(
                find_axis(&mm, "text:fallback-engine").is_some(),
                "msdf keeps the script degrade for {html:?}"
            );
        }
    }

    #[test]
    fn hybrid_probe_suppresses_vector_served_degrade() {
        // Task #128 / drain-the-browser-lane: a node the engine-truth probe
        // proves the vector arm serves renders on the SAME engine the GA
        // default lane ships as verdict full — so it must NOT degrade. It
        // surfaces a `text:fallback-engine:vector` ManifestWarning (the
        // hybrid-lane diagnostics) instead of a degraded axis; declined
        // nodes keep the classic axis/band. Probe absent (msdf/vector
        // engines) → classic axis only, no warning.
        let nodes = json!([
            {"id":"cjk","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
             "rotation":0,"scale":1,
             "htmlContent":"<p style=\"font-family: Inter\">你好</p>"},
            {"id":"fake","type":"richtext","x":0.0,"y":20.0,"width":10.0,"height":10.0,
             "rotation":0,"scale":1,
             "htmlContent":"<p style=\"font-family: Mystery Sans\">plain</p>"}
        ]);
        let s = Scene::load(&page_with(nodes, json!(["cjk", "fake"]))).unwrap();
        // Probe: serve anything containing CJK (engine truth is injected by
        // the CLI; here a stand-in with the same shape).
        let m = assess(
            &s,
            &AssessOptions {
                hybrid_vector_probe: Some(Box::new(|html: &str| html.contains('你'))),
                ..Default::default()
            },
        );
        // The vector-served node no longer emits ANY degraded axis…
        assert!(find_axis(&m, "text:fallback-engine:vector").is_none());
        let classic = find_axis(&m, "text:fallback-engine").expect("classic axis");
        assert_eq!(classic.node_ids, vec!["fake".to_string()]);
        // …but the hybrid-lane diagnostics survive as a warning carrying the
        // suppressed causes (the cjk node's script detail).
        let w = m.pages[0]
            .warnings
            .iter()
            .find(|w| w.axis == "text:fallback-engine:vector")
            .expect("vector-served warning");
        assert_eq!(w.detail.get("node"), Some(&json!("cjk")));
        assert!(w.detail.get("scripts").is_some());

        // No probe → no vector warning, both nodes on the classic axis.
        let m2 = assess(&s, &AssessOptions::default());
        assert!(find_axis(&m2, "text:fallback-engine:vector").is_none());
        assert!(!m2.pages[0]
            .warnings
            .iter()
            .any(|w| w.axis == "text:fallback-engine:vector"));
        let classic2 = find_axis(&m2, "text:fallback-engine").unwrap();
        assert_eq!(classic2.node_ids.len(), 2);
    }

    #[test]
    fn hybrid_probe_vector_served_node_is_full_and_renderable() {
        // Renderer-parity evidence for the assessment flip: a page whose only
        // fallback-class node is vector-served must grade FULL + renderable —
        // identical to what the GA default vector engine reports for the same
        // node (`vector_engine_renders_non_latin_scripts_natively`), because
        // both lanes render it with the same vector engine and ship the same
        // pixels. Before the flip this page graded Degraded and could be
        // routed to the (draining) browser lane by any accepts_degraded=False
        // consumer.
        let s = Scene::load(&page_with(
            json!([{"id":"cjk","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                "rotation":0,"scale":1,
                "htmlContent":"<p style=\"font-family: Inter\">你好</p>"}]),
            json!(["cjk"]),
        )).unwrap();
        let m = assess(
            &s,
            &AssessOptions {
                hybrid_vector_probe: Some(Box::new(|_| true)),
                ..Default::default()
            },
        );
        assert_eq!(m.verdict.as_str(), "full");
        assert!(m.renderable());
        // Parity target: the GA default vector engine on the same scene.
        let vm = assess(
            &s,
            &AssessOptions {
                text_engine_is_vector: true,
                ..Default::default()
            },
        );
        assert_eq!(vm.verdict.as_str(), m.verdict.as_str());
    }

    #[test]
    fn q42_valid_flags_renderable_lone_or_invalid_ri_fallback_class() {
        // Q-42: the cluster regex absorbs skin-tone modifiers, so a skin-toned
        // emoji renders via the Noto SVG pipeline — a `text:emoji` warning only,
        // NO fallback-engine degrade. Regional-indicator PAIRS with a shipped
        // waved-flag SVG now cluster the same way (emoji.ts `RI_PAIR` +
        // moda-text mirror) — renderable, warning only. A lone RI or an
        // unshipped pair (e.g. 🇽🇽) stays in TEXT runs -> coverage gap -> the
        // whole node is fallback-class (Degraded), matching the render path.
        let mut catalog = FontCatalog::default();
        catalog.families.insert("inter".into(),
            vec![FontVariantEntry {
                weight: 400.0,
                style: "normal".to_string(),
                msdf_ready: true,
                atlas_urls: vec!["https://cdn/atlas0.png".to_string()],
                metrics_url: Some("https://cdn/metrics.json".to_string()),
                ttf_url: None,
            }],
        );

        // Skin-toned wave — renderable, warning only, on both engines.
        let skin = Scene::load(&page_with(
            json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                "rotation":0,"scale":1,
                "htmlContent":"<p style=\"font-family: Inter\">wave \u{1F44B}\u{1F3FD}</p>"}]),
            json!(["t"]),
        )).unwrap();
        for vector in [false, true] {
            let m = assess(
                &skin,
                &AssessOptions {
                    font_catalog: Some(catalog.clone()),
                    text_engine_is_vector: vector,
                    ..Default::default()
                },
            );
            assert!(
                find_axis(&m, "text:fallback-engine").is_none(),
                "skin-toned emoji must not degrade (vector={vector})"
            );
            assert!(m.pages[0].warnings.iter().any(|w| w.axis == "text:emoji"));
        }

        // A valid flag pair (🇺🇸) — clusters, renders via the image pipeline:
        // renderable, `text:emoji` warning only, on both engines.
        let flag = Scene::load(&page_with(
            json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                "rotation":0,"scale":1,
                "htmlContent":"<p style=\"font-family: Inter\">flag \u{1F1FA}\u{1F1F8}</p>"}]),
            json!(["t"]),
        )).unwrap();
        for vector in [false, true] {
            let m = assess(
                &flag,
                &AssessOptions {
                    font_catalog: Some(catalog.clone()),
                    text_engine_is_vector: vector,
                    ..Default::default()
                },
            );
            assert!(
                find_axis(&m, "text:fallback-engine").is_none(),
                "shipped flag pair must not degrade (vector={vector})"
            );
            assert!(m.pages[0].warnings.iter().any(|w| w.axis == "text:emoji"));
        }

        // A lone RI and an unshipped pair (🇽🇽 — no waved-flag SVG) — still a
        // coverage gap -> fallback-class, on the default AND vector engines
        // (they route to the fallback raster — a content divergence
        // independent of the font engine).
        for text in ["lone \u{1F1FA}", "nope \u{1F1FD}\u{1F1FD}"] {
            let scene = Scene::load(&page_with(
                json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                    "rotation":0,"scale":1,
                    "htmlContent":format!("<p style=\"font-family: Inter\">{text}</p>")}]),
                json!(["t"]),
            )).unwrap();
            for vector in [false, true] {
                let m = assess(
                    &scene,
                    &AssessOptions {
                        font_catalog: Some(catalog.clone()),
                        text_engine_is_vector: vector,
                        ..Default::default()
                    },
                );
                let a = find_axis(&m, "text:fallback-engine")
                    .unwrap_or_else(|| panic!("{text:?} must degrade (vector={vector})"));
                assert!(matches!(a.status, AxisStatus::Degraded { .. }));
                assert_eq!(a.detail.get("emoji_coverage_gap"), Some(&json!(true)));
                // font/script detail must NOT be present (only the emoji gap fired).
                assert!(a.detail.get("families").is_none());
            }
        }
    }

    #[test]
    fn pure_emoji_canvas_stays_renderable_full() {
        // The bundle guarantee: an Extended_Pictographic-only canvas (no
        // skin-tone / regional-indicator / uncovered-script text) is
        // provider-serviceable on both native lanes, so the VERDICT the router
        // gates on stays `full` + renderable — it must never route such a canvas
        // to the browser. (The router declines any `degraded` verdict for
        // non-degraded-accepting output.) A face + a ZWJ sequence, no modifiers.
        let mut catalog = FontCatalog::default();
        catalog.families.insert("inter".into(),
            vec![FontVariantEntry {
                weight: 400.0,
                style: "normal".to_string(),
                msdf_ready: true,
                atlas_urls: vec!["https://cdn/atlas0.png".to_string()],
                metrics_url: Some("https://cdn/metrics.json".to_string()),
                ttf_url: None,
            }],
        );
        let s = Scene::load(&page_with(
            json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                "rotation":0,"scale":1,
                "htmlContent":"<p style=\"font-family: Inter\">party \u{1F389} \u{1F600} family \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}</p>"}]),
            json!(["t"]),
        )).unwrap();
        for vector in [false, true] {
            let m = assess(
                &s,
                &AssessOptions {
                    font_catalog: Some(catalog.clone()),
                    text_engine_is_vector: vector,
                    ..Default::default()
                },
            );
            assert_eq!(m.verdict.as_str(), "full", "vector={vector}");
            assert!(m.renderable(), "vector={vector}");
            assert!(find_axis(&m, "text:fallback-engine").is_none());
            assert!(m.pages[0].warnings.iter().any(|w| w.axis == "text:emoji"));
        }
    }

    #[test]
    fn text_as_mask_reports_degraded_band() {
        // Q-28 text-as-mask (M4d): under the non-vector engines the mask
        // raster is the Degraded (cosmic) tier — honest Degraded band. (The
        // default AssessOptions has `text_engine_is_vector: false`, and the
        // default FontCatalog has no families, so the vector arm can't fire.)
        let s = Scene::load(&page_with(
            json!([
                {"id":"g","type":"group","x":0.0,"y":0.0,"children":["m","c"],
                 "clipMaskNodeId":"m"},
                {"id":"m","type":"richtext","x":0.0,"y":0.0,"width":100.0,"height":40.0,
                 "rotation":0,"scale":1,
                 "htmlContent":"<p style=\"font-family: Inter\">MODA</p>"},
                {"id":"c","type":"rectangle","x":0.0,"y":0.0,"width":100.0,"height":40.0,
                 "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ff006e"}}]}
            ]),
            json!(["g"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        let a = find_axis(&m, "mask:text-alpha").expect("text-as-mask axis");
        match &a.status {
            AxisStatus::Degraded { expected_ssim_band } => {
                assert_eq!(*expected_ssim_band, (0.90, 0.97));
            }
            other => panic!("expected degraded, got {other:?}"),
        }
        assert!(a.detail.get("dropped").is_none(), "page 0 mask is LIVE");
        assert_eq!(a.detail.get("mask_node"), Some(&json!("m")));
        // Q-95/Q-28: the mask SOURCE never renders as text (shouldSkipRender)
        // — its non-catalog family must not also band the page as
        // text:fallback-engine (the mask axis carries the band).
        assert!(find_axis(&m, "text:fallback-engine").is_none());
    }

    /// A catalog whose `inter` family is vector/msdf-servable (the shape the
    /// mask-tier tests need).
    fn inter_catalog() -> FontCatalog {
        let mut catalog = FontCatalog::default();
        catalog.families.insert("inter".into(),
            vec![FontVariantEntry {
                weight: 400.0,
                style: "normal".to_string(),
                msdf_ready: true,
                atlas_urls: vec!["https://cdn/atlas0.png".to_string()],
                metrics_url: Some("https://cdn/metrics.json".to_string()),
                ttf_url: Some("https://cdn/inter.ttf".to_string()),
            }],
        );
        catalog
    }

    fn mask_scene(mask_node: Value) -> Scene {
        Scene::load(&page_with(
            json!([
                {"id":"g","type":"group","x":0.0,"y":0.0,"children":["m","c"],
                 "clipMaskNodeId":"m"},
                mask_node,
                {"id":"c","type":"rectangle","x":0.0,"y":0.0,"width":100.0,"height":40.0,
                 "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ff006e"}}]}
            ]),
            json!(["g"]),
        )).unwrap()
    }

    #[test]
    fn text_as_mask_vector_servable_is_full_under_vector_engine() {
        // The better tier (vector mask raster): a vector-servable mask source
        // under the GA vector engine renders the exact vector letterforms —
        // same divergence class the visible vector lane ships as Full.
        let s = mask_scene(json!(
            {"id":"m","type":"richtext","x":0.0,"y":0.0,"width":100.0,"height":40.0,
             "rotation":0,"scale":1,
             "htmlContent":"<p style=\"font-family: Inter\">MODA</p>"}
        ));
        let m = assess(
            &s,
            &AssessOptions {
                font_catalog: Some(inter_catalog()),
                text_engine_is_vector: true,
                ..Default::default()
            },
        );
        let a = find_axis(&m, "mask:text-alpha").expect("text-as-mask axis");
        assert!(
            matches!(a.status, AxisStatus::Full),
            "expected Full, got {:?}",
            a.status
        );
        assert_eq!(a.detail.get("engine"), Some(&json!("vector")));
        assert_eq!(a.detail.get("mask_node"), Some(&json!("m")));
    }

    #[test]
    fn text_as_mask_shader_fill_is_vector_servable() {
        // A shader-fill mask source is Full under the vector engine BY
        // DESIGN, not an oversight: the mask spec seam never routes shader
        // fills (`node_to_text_spec` pins `shader_fill: false` for the
        // mask/Degraded consumers — moda-engine
        // `text_draw::mask_spec_never_routes_shader_fill`), so the render arm
        // rasterizes the plain glyph silhouette with per-run colors — exactly
        // what the browser's Canvas2D standalone mask texture does (Canvas2D
        // cannot run shaders). No AnalyticShaderFill op ever reaches the
        // rasterizer's decline gate for a MASK source.
        let s = mask_scene(json!(
            {"id":"m","type":"richtext","x":0.0,"y":0.0,"width":100.0,"height":40.0,
             "rotation":0,"scale":1,
             "textFills":[{"id":"fill-0","paint":{"type":"shader","shaderId":"mesh-gradient"}}],
             "htmlContent":"<p style=\"font-family: Inter\">MODA</p>"}
        ));
        let m = assess(
            &s,
            &AssessOptions {
                font_catalog: Some(inter_catalog()),
                text_engine_is_vector: true,
                ..Default::default()
            },
        );
        let a = find_axis(&m, "mask:text-alpha").expect("text-as-mask axis");
        assert!(
            matches!(a.status, AxisStatus::Full),
            "expected Full, got {:?}",
            a.status
        );
        assert_eq!(a.detail.get("engine"), Some(&json!("vector")));
    }

    #[test]
    fn text_as_mask_fallback_class_keeps_degraded_band_under_vector_engine() {
        // Fallback-class arms stay honestly Degraded even under the vector
        // engine: the render arm declines and uses the Degraded raster.
        let cases: Vec<Value> = vec![
            // Non-catalog family → run resolution declines the vector lane.
            json!({"id":"m","type":"richtext","x":0.0,"y":0.0,"width":100.0,"height":40.0,
                "rotation":0,"scale":1,
                "htmlContent":"<p style=\"font-family: Season Sans\">MODA</p>"}),
            // Legacy shadow → silhouette gate declines.
            json!({"id":"m","type":"richtext","x":0.0,"y":0.0,"width":100.0,"height":40.0,
                "rotation":0,"scale":1,
                "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000000","blur":4,"offset":{"x":0,"y":0}}],
                "htmlContent":"<p style=\"font-family: Inter\">MODA</p>"}),
            // Fallback-script content → vector routing declines the runs.
            json!({"id":"m","type":"richtext","x":0.0,"y":0.0,"width":100.0,"height":40.0,
                "rotation":0,"scale":1,
                "htmlContent":"<p style=\"font-family: Inter\">مرحبا</p>"}),
        ];
        for mask_node in cases {
            let s = mask_scene(mask_node.clone());
            let m = assess(
                &s,
                &AssessOptions {
                    font_catalog: Some(inter_catalog()),
                    text_engine_is_vector: true,
                    ..Default::default()
                },
            );
            let a = find_axis(&m, "mask:text-alpha").expect("text-as-mask axis");
            match &a.status {
                AxisStatus::Degraded { expected_ssim_band } => {
                    assert_eq!(*expected_ssim_band, (0.90, 0.97), "case {mask_node}");
                }
                other => panic!("case {mask_node}: expected degraded, got {other:?}"),
            }
        }
    }

    #[test]
    fn text_as_mask_off_first_page_is_assessed_as_on_the_first() {
        // A text clip mask on page 2 builds its coverage as on page 1 (the
        // export path dropped it there: children baked UNCLIPPED), so it gets
        // the same band and no `dropped` detail.
        let s = Scene::load(&json!({
            "nodes": [
                {"id":"first","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                 "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ffffff"}}]},
                {"id":"g","type":"group","x":0.0,"y":0.0,"children":["m","c"],
                 "clipMaskNodeId":"m"},
                {"id":"m","type":"richtext","x":0.0,"y":0.0,"width":100.0,"height":40.0,
                 "rotation":0,"scale":1,
                 "htmlContent":"<p style=\"font-family: Season Sans\">UGC</p>"},
                {"id":"c","type":"rectangle","x":0.0,"y":0.0,"width":100.0,"height":40.0,
                 "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ff006e"}}]}
            ],
            "pages": [
                {"id":"p1","name":"P1","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                 "nodeIds": ["first"]},
                {"id":"p2","name":"P2","width":100.0,"height":100.0,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fff"}}],
                 "nodeIds": ["g"]}
            ]
        })).unwrap();
        let m = assess(&s, &AssessOptions::default());
        let a = m.pages[1]
            .axes
            .iter()
            .find(|a| a.axis == "mask:text-alpha")
            .expect("text-as-mask axis on page 1");
        assert!(
            matches!(a.status, AxisStatus::Degraded { .. }),
            "expected the page-0 band, got {:?}",
            a.status
        );
        assert!(a.detail.get("dropped").is_none());
        // The mask source never renders as text: no text axes.
        assert!(m.pages[1].axes.iter().all(|a| !a.axis.starts_with("text:")));
    }

    #[test]
    fn msdf_catalog_family_is_full() {
        let mut catalog = FontCatalog::default();
        catalog.families.insert("montserrat".into(),
            vec![FontVariantEntry {
                weight: 400.0,
                style: "normal".to_string(),
                msdf_ready: true,
                atlas_urls: vec!["https://cdn/atlas0.png".to_string()],
                metrics_url: Some("https://cdn/metrics.json".to_string()),
                ttf_url: None,
            }],
        );
        let s = Scene::load(&page_with(
            json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                "rotation":0,"scale":1,
                "htmlContent":"<p style=\"font-family: Montserrat\">Hello world</p>"}]),
            json!(["t"]),
        )).unwrap();
        let m = assess(
            &s,
            &AssessOptions {
                font_catalog: Some(catalog),
                ..Default::default()
            },
        );
        assert!(find_axis(&m, "text:fallback-engine").is_none());
        assert_eq!(m.verdict.as_str(), "full");
        assert!(m.renderable());
    }

    #[test]
    fn variant_gap_probe_degrades_text() {
        // Task #72: an msdf catalog family requested at a variant the render
        // host has no vendored bytes for must degrade (text:variant-gap), not
        // report `full`. The probe is engine-truth supplied by the CLI; here
        // a stub stands in for `moda_text::vendored_variant_gaps`.
        let mut catalog = FontCatalog::default();
        catalog.families.insert(
            "fraunces".into(),
            vec![crate::types::FontVariantEntry {
                weight: 400.0,
                style: "normal".into(),
                msdf_ready: true,
                atlas_urls: vec!["a.png".into()],
                metrics_url: None,
                ttf_url: None,
            }],
        );
        let s = Scene::load(&page_with(
            json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                "rotation":0,"scale":1,
                "htmlContent":"<p style=\"font-family: Fraunces\"><b>Bold</b></p>"}]),
            json!(["t"]),
        )).unwrap();
        let probe: Box<dyn Fn(&str) -> Vec<String>> = Box::new(|html: &str| {
            if html.contains("<b>") {
                vec!["Fraunces-700-normal".to_string()]
            } else {
                Vec::new()
            }
        });
        let m = assess(
            &s,
            &AssessOptions {
                font_catalog: Some(catalog.clone()),
                variant_gap_probe: Some(probe),
                ..Default::default()
            },
        );
        let axis = find_axis(&m, "text:variant-gap").expect("variant-gap axis");
        assert!(matches!(axis.status, AxisStatus::Degraded { .. }));
        assert_eq!(
            axis.detail["missing_vendored_keys"],
            json!(["Fraunces-700-normal"])
        );
        assert_eq!(m.verdict.as_str(), "degraded");
        // No probe (envelope / --canvas-id modes) → no variant-gap axis.
        let m2 = assess(
            &s,
            &AssessOptions {
                font_catalog: Some(catalog),
                ..Default::default()
            },
        );
        assert!(find_axis(&m2, "text:variant-gap").is_none());
    }

    #[test]
    fn variant_gap_suppressed_under_vector_engine() {
        // The variant gap is a MISSING-ATLAS-BYTES artifact: under the vector
        // engine (the GA default of BOTH the in-browser WASM renderer and the
        // server lane's NATIVE_RENDER_TEXT_ENGINE) glyphs draw from the
        // variant's TTF outlines directly, so a catalog-promised variant with
        // no vendored MSDF atlas renders at full fidelity — the axis must NOT
        // fire even when the probe reports gaps. Render-path evidence for the
        // suppression: the vector lane is the same engine the editor renders
        // these variants with, and even the MSDF best-effort lane now draws a
        // Degraded fallback face instead of silently dropping the run
        // (moda-text/src/ops.rs). Under msdf/hybrid (vector=false) the axis
        // keeps its exact Degraded band — the dev/oracle vendored-fixture
        // lane genuinely drops runs there.
        let mut catalog = FontCatalog::default();
        catalog.families.insert(
            "fraunces".into(),
            vec![crate::types::FontVariantEntry {
                weight: 400.0,
                style: "normal".into(),
                msdf_ready: true,
                atlas_urls: vec!["a.png".into()],
                metrics_url: None,
                ttf_url: None,
            }],
        );
        let s = Scene::load(&page_with(
            json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                "rotation":0,"scale":1,
                "htmlContent":"<p style=\"font-family: Fraunces\"><b>Bold</b></p>"}]),
            json!(["t"]),
        )).unwrap();
        let probe = || -> Box<dyn Fn(&str) -> Vec<String>> {
            Box::new(|_html: &str| vec!["Fraunces-700-normal".to_string()])
        };
        // vector engine + a probe that ALWAYS reports a gap → no axis, doc Full.
        let m = assess(
            &s,
            &AssessOptions {
                font_catalog: Some(catalog.clone()),
                variant_gap_probe: Some(probe()),
                text_engine_is_vector: true,
                ..Default::default()
            },
        );
        assert!(find_axis(&m, "text:variant-gap").is_none());
        assert_eq!(m.verdict.as_str(), "full");
        assert!(m.renderable());
        // Same probe under msdf/hybrid (vector=false) → the exact Degraded
        // band + missing_vendored_keys detail (the marking stays truthful for
        // the vendored-fixture dev lane).
        let m2 = assess(
            &s,
            &AssessOptions {
                font_catalog: Some(catalog),
                variant_gap_probe: Some(probe()),
                text_engine_is_vector: false,
                ..Default::default()
            },
        );
        let axis = find_axis(&m2, "text:variant-gap").expect("variant-gap axis");
        assert_eq!(
            axis.status,
            AxisStatus::Degraded {
                expected_ssim_band: (0.90, 0.97)
            }
        );
        assert_eq!(
            axis.detail["missing_vendored_keys"],
            json!(["Fraunces-700-normal"])
        );
        assert_eq!(m2.verdict.as_str(), "degraded");
    }

    /// Catalog covering the effects fixtures' families (Inter 400/700,
    /// Fraunces 400) so the font axes stay Full and the tests isolate the
    /// effects posture.
    fn effects_test_catalog() -> FontCatalog {
        let mut catalog = FontCatalog::default();
        let entry = |weight: f64| FontVariantEntry {
            weight,
            style: "normal".to_string(),
            msdf_ready: true,
            atlas_urls: vec!["https://cdn/atlas0.png".to_string()],
            metrics_url: Some("https://cdn/metrics.json".to_string()),
            ttf_url: Some("https://cdn/font.ttf".to_string()),
        };
        catalog
            .families
            .insert("inter".into(), vec![entry(400.0), entry(700.0)]);
        catalog
            .families
            .insert("fraunces".into(), vec![entry(400.0)]);
        catalog
    }

    #[test]
    fn structured_effects_full_on_richtext_and_inert_on_shapes() {
        // Task #142 stage 2: the Q-123 `effects:structured` gate is GONE —
        // the vector ops walk renders effects[] and every lane routes
        // active-effects richtext through the vector arm (Q-124). A richtext
        // glow is parity-Full: no axis at all (effects is known schema on
        // richtext, nodes.ts:1139-1141).
        let s = Scene::load(&page_with(
            json!([
                {"id":"t","type":"richtext","x":0.0,"y":0.0,"width":100.0,"height":40.0,
                 "rotation":0,"scale":1,
                 "htmlContent":"<p style=\"font-family: Inter\">NEON</p>",
                 "effects":[{"type":"glow","id":"fx1","color":"#36f9c5","radiusPx":18.0}]}
            ]),
            json!(["t"]),
        )).unwrap();
        let m = assess(
            &s,
            &AssessOptions {
                font_catalog: Some(effects_test_catalog()),
                ..Default::default()
            },
        );
        assert!(find_axis(&m, "effects:structured").is_none());
        assert!(find_axis(&m, "unknown:node-field").is_none());
        assert_eq!(m.verdict.as_str(), "full");
        assert!(m.renderable());

        // A text effect on a shape (a rect with a glow): `effects` is schema
        // on every effect-capable kind (E-M.7), and the browser draws no text
        // effect on a shape either — parity-Full, no spillover.
        let s2 = Scene::load(&page_with(
            json!([
                {"id":"r","type":"rectangle","x":0.0,"y":50.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#3a86ff"}}],
                 "effects":[{"type":"glow","id":"fx2","color":"#ffbe0b","radiusPx":12.0}]}
            ]),
            json!(["r"]),
        )).unwrap();
        let m2 = assess(&s2, &AssessOptions::default());
        assert!(find_axis(&m2, "effects:structured").is_none());
        assert!(find_axis(&m2, "unknown:node-field").is_none());
        assert_eq!(m2.verdict.as_str(), "full");
        assert!(m2.renderable());
    }

    #[test]
    fn distort_field_is_known_schema() {
        // A plain rectangle round-tripped through a backend `model_dump()` carries `distort: null`
        // (and other unset optionals). That inert null must NOT spill as `unknown:node-field` —
        // otherwise every saved doc with a shape would false-decline.
        let with_null = Scene::load(&page_with(
            json!([{"id":"r","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                    "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#3a86ff"}}],"distort":null}]),
            json!(["r"]),
        )).unwrap();
        let m = assess(&with_null, &AssessOptions::default());
        assert!(
            find_axis(&m, "unknown:node-field").is_none(),
            "distort:null must not spill"
        );

        // A REAL (non-null) distort is KNOWN schema as of node-distort PR 2:
        // the native engine renders the warp, so the doc must NOT decline.
        let with_real = Scene::load(&page_with(
            json!([{"id":"r","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                    "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#3a86ff"}}],
                    "distort":{"type":"matrix","matrix":[1,0,0,0,1,0,0.3,0.1,1]}}]),
            json!(["r"]),
        )).unwrap();
        let m2 = assess(&with_real, &AssessOptions::default());
        assert!(
            find_axis(&m2, "unknown:node-field").is_none(),
            "a distort-carrying node renders natively as of PR 2 — no spill"
        );
        assert!(m2.renderable());

        // On a NON-capable type `distort` is out-of-schema and still spills
        // (declines to the fallback rather than rendering silently unwarped).
        let on_line = Scene::load(&page_with(
            json!([{"id":"l","type":"line","points":[0.0,0.0,10.0,10.0],
                    "distort":{"type":"matrix","matrix":[1,0,0,0,1,0,0.3,0.1,1]}}]),
            json!(["l"]),
        )).unwrap();
        let m3 = assess(&on_line, &AssessOptions::default());
        assert!(
            find_axis(&m3, "unknown:node-field").is_some(),
            "distort on a non-capable type must spill"
        );
    }

    #[test]
    fn structured_effects_fixtures_render_full() {
        // Pin the committed effects fixtures (the golden-lane scenes) to the
        // un-gated posture: richtext-only effects ⇒ verdict full, renderable.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .to_path_buf();
        for name in [
            "structured-glow",
            "structured-echo",
            "structured-extrude",
            "structured-combo",
            "structured-animated",
        ] {
            let raw =
                std::fs::read_to_string(root.join(format!("tests/fixtures/{name}.json")))
                    .expect("fixture present");
            let parsed: Value = crate::value::from_str(&raw).unwrap();
            let s = Scene::load(crate::extract_canvas_data(&parsed)).unwrap();
            let m = assess(
                &s,
                &AssessOptions {
                    font_catalog: Some(effects_test_catalog()),
                    ..Default::default()
                },
            );
            assert!(
                find_axis(&m, "effects:structured").is_none(),
                "{name}: gate must be gone"
            );
            assert_eq!(m.verdict.as_str(), "full", "{name}");
            assert!(m.renderable(), "{name}");
        }
        // The non-adopter fixture (rect with glow): `effects` is schema on a
        // rect too (E-M.7) and the glow draws nothing there — no spillover.
        let raw = std::fs::read_to_string(
            root.join("tests/fixtures/structured-non-adopter.json"),
        )
        .expect("fixture present");
        let parsed: Value = crate::value::from_str(&raw).unwrap();
        let s = Scene::load(crate::extract_canvas_data(&parsed)).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert!(find_axis(&m, "effects:structured").is_none());
        assert!(find_axis(&m, "unknown:node-field").is_none());
        assert!(m.renderable());
    }

    #[test]
    fn inert_shader_effects_render_natively_without_declining() {
        // The browser reads ONLY the first renderable source-mask effect
        // (`getRenderableSourceMaskShaderEffect` → `.find(...)`) and ignores
        // every other `shader` entry of `effects`, so a doc whose extra entries are
        // inert (post-process mode, or extras beyond the first source-mask)
        // renders pixel-identically natively. `effects:shader-effects` is
        // recorded for observability but MUST NOT decline the doc — declining
        // burns a Chromium render and, once the browser renderer retires,
        // leaves the doc unrenderable.

        // (a) post-process-only: no renderable source-mask effect at all. The
        // node renders its underlying fill; native === browser (both inert).
        let pp = Scene::load(&page_with(
            json!([{"id":"r","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                    "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#3a86ff"}}],
                    "effects":[{"id":"pp","type":"shader","shaderId":"some-post",
                        "mode":"post-process","source":{"type":"rendered-node","nodeId":"r"},
                        "params":{}}]}]),
            json!(["r"]),
        )).unwrap();
        let m = assess(&pp, &AssessOptions::default());
        let axis =
            find_axis(&m, "effects:shader-effects").expect("axis recorded for observability");
        assert_eq!(
            axis.status,
            AxisStatus::Unknown,
            "novelty signal, not a capability gap"
        );
        assert_eq!(m.verdict.as_str(), "full");
        assert!(
            m.renderable(),
            "post-process-only doc must render natively, not decline"
        );

        // (b) a renderable source-mask effect (KNOWN shader) FOLLOWED by an
        // inert post-process entry: the first effect renders via the shader
        // pipeline; the trailing entry is inert. The doc must NOT decline.
        let mixed = Scene::load(&page_with(
            json!([{"id":"r","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
            "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"image","src":"data:image/svg+xml;base64,AAAA"}}],
            "effects":[
                {"id":"sm","type":"shader","shaderId":"heatmap-image",
                 "mode":"source-mask-material","source":{"type":"self-asset"},"params":{}},
                {"id":"pp","type":"shader","shaderId":"some-post",
                 "mode":"post-process","source":{"type":"rendered-node","nodeId":"r"},"params":{}}
            ]}]),
            json!(["r"]),
        )).unwrap();
        let m = assess(&mixed, &AssessOptions::default());
        assert!(find_axis(&m, "effects:shader-effects").is_some());
        assert!(
            find_axis(&m, "shader:heatmap-image").is_none(),
            "the known first source-mask shader does not gate"
        );
        assert_eq!(m.verdict.as_str(), "full");
        assert!(
            m.renderable(),
            "renderable-first + inert-trailing must not decline"
        );

        // (c) an UNKNOWN shader on the first renderable source-mask effect
        // surfaces on the separate `shader:<id>` axis (not `effects:shader-effects`)
        // — but that axis is NON-gating too: native draws the
        // identical white box the browser would with the same engine, so the doc
        // SERVES rather than burning a Chromium render. The `shader:no-such-shader`
        // axis is still emitted (drift observability → shader_drift_axis WARNING).
        let unknown = Scene::load(&page_with(
            json!([{"id":"r","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                    "rotation":0,"scale":1,"fills":[{"id":"fill-0","paint":{"type":"image","src":"data:image/svg+xml;base64,AAAA"}}],
                    "effects":[{"id":"sm","type":"shader","shaderId":"no-such-shader",
                        "mode":"source-mask-material","source":{"type":"self-asset"},"params":{}}]}]),
            json!(["r"]),
        )).unwrap();
        let m = assess(&unknown, &AssessOptions::default());
        let shader_axis = find_axis(&m, "shader:no-such-shader")
            .expect("unknown shader on the renderable effect surfaces as shader:<id>");
        assert_eq!(shader_axis.status, AxisStatus::Unknown);
        assert_eq!(
            m.verdict.as_str(),
            "full",
            "unknown source-mask shader is non-gating — the doc serves natively (ENG-4036)"
        );
        assert!(m.renderable());
    }

    #[test]
    fn glass_backdrop_common_full_exotic_degraded() {
        // Common glass (params inside the measured fixture envelope — the
        // golden ratchet floors pin backdrop/glass* at >= 0.99467) is Full:
        // no gating axis, only a non-gating observability warning.
        let s = Scene::load(&page_with(
            json!([
                {"id":"g","type":"rectangle","x":0.0,"y":0.0,"width":1.0,"height":1.0,"rotation":0,"scale":1,
                 "effects":[{"id":"backdrop-0","type":"glass","frost":30.0,"refraction":60.0,"depth":40.0,
                    "lightAngle":0.0,"lightIntensity":40.0,"dispersion":20.0,"splay":20.0}]},
                {"id":"u","type":"rectangle","x":0.0,"y":0.0,"width":1.0,"height":1.0,"rotation":0,"scale":1,
                 "effects":[{"id":"backdrop-0","type":"background-blur","radius":10.0}]},
                {"id":"inert","type":"rectangle","x":0.0,"y":0.0,"width":1.0,"height":1.0,"rotation":0,"scale":1,
                 "effects":[{"id":"backdrop-0","type":"background-blur","radius":10.0,"visible":false}]}
            ]),
            json!(["g", "u", "inert"]),
        )).unwrap();
        let m = assess(&s, &AssessOptions::default());
        assert!(
            find_axis(&m, "backdrop:glass").is_none(),
            "fixture-covered glass must not gate"
        );
        // an active and a hidden background blur produce no gating axes either
        assert_eq!(m.pages[0].axes.len(), 0);
        assert_eq!(m.verdict.as_str(), "full");
        assert!(
            m.pages[0]
                .warnings
                .iter()
                .any(|w| w.axis == "backdrop:glass"),
            "observability warning survives the promotion"
        );

        // Params beyond the measured fixture envelope (frost > 30 /
        // refraction > 90 / depth > 70 / dispersion > 80 / lightIntensity > 80
        // / splay > 50) keep a narrowed Degraded band — including a SINGLE
        // out-of-envelope knob while every other one is in range.
        let exotic_scenes = [
            json!({"id":"backdrop-0","type":"glass","frost":10.0,"refraction":100.0,"depth":80.0,
                "lightAngle":0.0,"lightIntensity":40.0,"dispersion":100.0,"splay":20.0}),
            json!({"id":"backdrop-0","type":"glass","frost":100.0,"refraction":60.0,"depth":40.0,
                "lightAngle":0.0,"lightIntensity":40.0,"dispersion":20.0,"splay":20.0}),
            json!({"id":"backdrop-0","type":"glass","frost":10.0,"refraction":60.0,"depth":40.0,
                "lightAngle":0.0,"lightIntensity":100.0,"dispersion":20.0,"splay":100.0}),
        ];
        for blur in exotic_scenes {
            let exotic = Scene::load(&page_with(
                json!([
                    {"id":"x","type":"rectangle","x":0.0,"y":0.0,"width":1.0,"height":1.0,"rotation":0,"scale":1,
                     "effects":[blur]}
                ]),
                json!(["x"]),
            )).unwrap();
            let m = assess(&exotic, &AssessOptions::default());
            match &find_axis(&m, "backdrop:glass").unwrap().status {
                AxisStatus::Degraded { expected_ssim_band } => {
                    assert_eq!(*expected_ssim_band, (0.99, 0.995));
                }
                other => panic!("beyond-envelope glass must stay Degraded, got {other:?}"),
            }
            assert_eq!(m.verdict.as_str(), "degraded");
        }
    }

    #[test]
    fn manifest_json_shape() {
        let s = Scene::load(&page_with(json!([]), json!([]))).unwrap();
        let m = assess(
            &s,
            &AssessOptions {
                canvas_id: Some("c-1".to_string()),
                engine_version: Some("deadbeef".to_string()),
                ..Default::default()
            },
        );
        let j = manifest_to_json(&m);
        assert_eq!(j["renderer"], json!("moda-render"));
        assert_eq!(j["canvas_id"], json!("c-1"));
        assert_eq!(j["verdict"], json!("full"));
        assert_eq!(j["pages"][0]["page_index"], json!(0));
    }
}
