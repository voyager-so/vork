//! text_draw.rs — text rendering (consumes `seams::TextQuadBatch`).
//!
//! Two halves:
//!
//! 1. **Plan side** (no GPU): `build_richtext_item` / `build_shape_text_item` / `plan_connector_text` lay a text
//!    node out on the vector lane (`moda_text::vector::build_vector_node_ops`: glyph outlines drawn analytically)
//!    and emit `DrawItem::Text(TextQuadBatch)`. A node the vector lane declines (no usable font for a run, a
//!    shaping failure) falls to the degraded tier: one cosmic-text raster quad (`build_degraded_batch`).
//!
//! 2. **GPU side**: `TextDraw` builds the extracted text pipelines (`vector-text-*`, `<id>.vector-text-fill`,
//!    `text-decoration`, `text-emoji` — blend/depth/vertex state from the manifest descriptors, never hand-coded),
//!    uploads glyph curves / emoji cells / degraded rasters, packs a 256-aligned uniform arena (model/color/
//!    params/clip per draw — the production dynamic-offset ring), and draws batches inside the transparent bucket
//!    loop AFTER paths (`renderTransparentItemsWithText`).
//!
//! Fonts come from a `moda_text::TextHost` — vendored bytes under `fixtures/oracle/fonts` natively (env
//! `MODA_FONTS_DIR` overrides), the installed VFS on wasm.

use std::cell::RefCell;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use moda_scene::types::RawNode;
use moda_scene::Scene;
use moda_text::ops::{parse_gradient_stops, GradientSpec, TextNodeSpec, TextOpPipeline};
use moda_text::TextHost;
use moda_scene::value::Value;

use crate::seams::{DegradedQuad, DrawItem, TextQuadBatch};

// ---------------------------------------------------------------------------
// Font host discovery (Gate 2; M4 replaces with catalog ingest)
// ---------------------------------------------------------------------------

// Pinned on the first SUCCESSFUL load (process-wide, like the browser's font
// cache) or on an explicit `install_text_host` (the wasm editor). A FAILED
// lazy load is NOT cached: embedders (the mobile FFI) materialize fonts
// per-canvas over the network, and a transient failure on the first text
// canvas must not poison every later text render in the process.
static HOST: std::sync::RwLock<Option<Arc<TextHost>>> = std::sync::RwLock::new(None);

/// Process-wide programmatic fonts-dir override, consulted BEFORE the
/// `MODA_FONTS_DIR` env fallback. Last write wins, and the dir in effect at
/// the first SUCCESSFUL text-host load is pinned for the process lifetime
/// (see [`HOST`]). Hosts that are already multithreaded (e.g. the iOS FFI,
/// where CFNetwork threads call `getenv` concurrently) MUST use this instead
/// of mutating `MODA_FONTS_DIR` — `setenv` after the process goes
/// multithreaded is the classic setenv/getenv use-after-free UB.
static FONTS_DIR_OVERRIDE: std::sync::RwLock<Option<std::path::PathBuf>> =
    std::sync::RwLock::new(None);

/// Set the process-wide fonts directory programmatically (last write wins).
/// Takes effect for text-host loads that happen after it; once a load
/// succeeds the host is pinned and later sets have no effect. Consulted
/// before `MODA_FONTS_DIR`.
pub fn set_fonts_dir(dir: impl Into<std::path::PathBuf>) {
    if let Ok(mut guard) = FONTS_DIR_OVERRIDE.write() {
        *guard = Some(dir.into());
    }
}

#[cfg(test)]
fn discover_fonts_dir() -> Option<std::path::PathBuf> {
    crate::test_fonts();
    Some(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fonts"))
}

#[cfg(not(test))]
fn discover_fonts_dir() -> Option<std::path::PathBuf> {
    if let Some(dir) = FONTS_DIR_OVERRIDE.read().ok().and_then(|g| g.clone()) {
        return Some(dir);
    }
    if let Ok(dir) = std::env::var("MODA_FONTS_DIR") {
        return Some(std::path::PathBuf::from(dir));
    }
    let mut dir = std::env::current_dir().ok()?;
    loop {
        for probe in [
            dir.join("fixtures/oracle/fonts"),
            dir.join("native-renderer/fixtures/oracle/fonts"),
        ] {
            if probe.join("MANIFEST.json").is_file() {
                return Some(probe);
            }
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// WASM-EDITOR PROTOTYPE (research/wasm-editor/SPIKE-wasm-render.md): install
/// an explicitly-constructed TextHost (e.g. loaded from an in-memory file map
/// on wasm32, where directory discovery is meaningless). REPLACES any
/// previously-installed host — the wasm editor re-installs after each
/// incremental font ingest (per-canvas on-demand feeding); readers clone the
/// `Arc` per frame, so a swap between frames is safe. Native callers install
/// once (or rely on lazy directory discovery below).
pub fn install_text_host(host: Arc<TextHost>) {
    let mut guard = HOST.write().unwrap();
    // Task #95 residency: a host swap (font install) changes atlas page
    // contents/UVs — retained text batches key on this generation. Bump it
    // while holding the host lock so `text_host_snapshot` cannot observe a
    // host paired with the preceding/following generation.
    TEXT_HOST_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    *guard = Some(host);
}

/// Monotonic install counter (task #95): retained text caches fold this into
/// their keys so font arrivals self-invalidate. 0 = lazy-discovered host.
static TEXT_HOST_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn text_host_generation() -> u64 {
    TEXT_HOST_GENERATION.load(std::sync::atomic::Ordering::Relaxed)
}

/// Task #95 R4: cache context threaded from the plan builder (same shape as
/// frame.rs `CacheCtx`).
pub type TextCacheCtx<'a> = (
    &'a mut crate::residency::PlanCaches,
    &'a crate::residency::HashMap<String, crate::residency::NodeHashes>,
);

/// R4 cache key for one text item: the node's TEXT content hash (deny-list:
/// full minus transform-only — paint IS baked into quads) ⊕ the plan inputs
/// that change the built batch ⊕ the text-host generation (font installs
/// swap atlases/UVs). The shader-fill instance and model matrix are PATCHED
/// on every hit instead of keyed, so transform moves and the shader clock
/// keep text residency.
/// Min real local→screen axis scale of a model matrix — the model half of
/// `analyticPadScale` (`try_build_vector_batch` bakes `min(sx, sy) ×
/// render_scale` into every glyph's em_box AA padding). Shared by the pad
/// computation and [`text_cache_key`] so the two derivations cannot drift.
pub(crate) fn model_min_axis_scale(model_matrix: &[f32; 16]) -> f64 {
    let sx = (model_matrix[0] as f64).hypot(model_matrix[1] as f64);
    let sy = (model_matrix[4] as f64).hypot(model_matrix[5] as f64);
    sx.min(sy)
}

#[allow(clippy::too_many_arguments)]
fn text_cache_key(
    scene: &Scene,
    hashes: &crate::residency::HashMap<String, crate::residency::NodeHashes>,
    node: &RawNode,
    best_effort: bool,
    render_scale: f64,
    model_min_scale: f64,
    area_key: u64,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let text_hash = hashes
        .get(node.id())
        .map(|h| h.text)
        .unwrap_or_else(|| crate::residency::node_hashes_in(scene, node).text);
    let mut h = rustc_hash::FxHasher::default();
    text_hash.hash(&mut h);
    // rust-edit-incremental-z4: the resident `hashes` map is computed from the
    // UN-folded effective node, but the plan builder bakes ANCESTOR opacity into
    // this node's `opacity` (frame.rs leaf-multiply-down) before shaping
    // it into quads — so the text batch's baked alpha depends on it. Fold the
    // (possibly ancestor-multiplied) opacity into the key so a parent/group opacity
    // edit re-keys the descendant's batch instead of serving one with stale alpha.
    // Pre-existing latent residency-cache bug surfaced by the incremental-edit
    // mutation_replay scenes (group opacity over text/shape children). Monotonic:
    // it can only ADD cache misses (a re-shape), never reuse a stale batch.
    node.f64("opacity").unwrap_or(1.0).to_bits().hash(&mut h);
    best_effort.hash(&mut h);
    render_scale.to_bits().hash(&mut h);
    // The MODEL half of analyticPadScale (node/ancestor scale is
    // TRANSFORM_ONLY, so the text hash alone cannot re-key it): a batch built
    // at one effective scale carries that scale's AA padding in every em_box
    // and must never be served at another — the cache-hit patch only updates
    // the model matrix (lever 10.5, caught by review on the retained animate
    // lanes where `scale` tweens hit this every frame).
    model_min_scale.to_bits().hash(&mut h);
    text_host_generation().hash(&mut h);
    // Area type (E8.14): the shape, thread and wrap objects the node's lines fit (they live on other nodes).
    area_key.hash(&mut h);
    h.finish()
}

/// On-hit patch of the volatile (non-content) batch fields.
fn patch_cached_batch(
    batch: &mut crate::seams::TextQuadBatch,
    z: f64,
    model: [f32; 16],
    clip: [f32; 4],
    fill_shader: Option<crate::seams::TextShaderFill>,
) {
    batch.z = z;
    batch.model_matrix = model;
    batch.page_clip_bounds = clip;
    // vork: moda-text-draw's TextQuadBatch has no fill_shader (text shader fills are not ported).
    let _ = fill_shader;
}

/// The process-wide text host (vendored catalog + atlases + emoji).
/// Lazily discovered from disk on native when nothing was installed.
fn text_host_snapshot() -> Result<(Arc<TextHost>, u64)> {
    if let Ok(guard) = HOST.read() {
        if let Some(host) = guard.as_ref() {
            return Ok((host.clone(), text_host_generation()));
        }
    }
    let mut guard = HOST
        .write()
        .map_err(|_| anyhow!("text host: lock poisoned"))?;
    if let Some(host) = guard.as_ref() {
        return Ok((host.clone(), text_host_generation()));
    }
    let dir = discover_fonts_dir().ok_or_else(|| {
        anyhow!(
            "text host: no vendored fonts dir found (set MODA_FONTS_DIR or run inside the repo)"
        )
    })?;
    let host = TextHost::load(&dir)
        .map(Arc::new)
        .map_err(|e| anyhow!("text host: {e}"))?;
    *guard = Some(host.clone());
    Ok((host, text_host_generation()))
}

pub fn font_substitutions(html: &str) -> Vec<(String, String)> {
    text_host_snapshot()
        .map(|(host, _)| moda_text::runs::substituted_families(html, &host.catalog))
        .unwrap_or_default()
}

pub fn text_host() -> Result<Arc<TextHost>> {
    text_host_snapshot().map(|(host, _)| host)
}

// Per-thread vector font store (`--text-engine vector`): parsed opentype.js-
// exact fonts over the same vendored TTF bytes the browser's VectorFontCache
// fetches. Thread-local (not the global host) because the store caches
// `Rc<OtjsFont>` and per-font RefCell curve caches — renders are
// single-threaded per plan. LENIENT vendor gaps: a catalog TTF that is not
// vendored behaves like the browser's not-yet-loaded font (per-node MSDF
// fallback) instead of the oracle's hard panic.
thread_local! {
    // The cached store plus the TEXT_HOST_GENERATION it was built at — reused only
    // while the GLOBAL generation is unchanged, so a host swap (`install_text_host`
    // on a new OR refreshed canvas's catalog — the mobile FFI, the warm server
    // lane) rebuilds it. Keying on the generation (not the dir) means a SAME-PATH
    // content change (a canvas re-materialized in place on a live swap/retry) still
    // rebuilds, and it is correct across threads — the generation is global, unlike
    // a per-thread dir (the mobile apps may create the session and render on
    // different threads).
    static VECTOR_STORE: RefCell<
        Option<(u64, std::rc::Rc<moda_text::vector::VectorFontStore>)>,
    > = const { RefCell::new(None) };
}

/// The vector font store paired with the text host it was built from.
///
/// These two are ONE snapshot on purpose. The store resolves a variant
/// by looking the CATALOG's `ttf_url` string up in `MANIFEST.files`, so a store
/// built over a different fonts dir than the catalog misses EVERY url and each
/// node silently degrades to the deprecated MSDF lane (`font-unavailable`). That
/// is exactly what the warm server lane did: `moda-py` swaps in a per-canvas
/// catalog host via `install_text_host` while leaving `MODA_FONTS_DIR` pinned at
/// the baked `/opt/moda/native/fonts`, so every non-baked family (Roboto, Open
/// Sans, Poppins, …) rendered MSDF. Returning the pair from a SINGLE host read
/// also means a host swap mid-node can't hand a caller store-A + catalog-B, and
/// the generation stamp means a swap since the store was built always rebuilds it.
///
/// This is the ONLY way to get a store, on every lane — there is deliberately no
/// `install_vector_store` escape hatch any more. It existed for the wasm editor
/// (whose fonts live in an in-memory VFS, where `discover_fonts_dir` is
/// meaningless), but the store it installed became byte-identical to this one the
/// moment the host started handing stores out: `load_url` reads through
/// `host::read_file_bytes`, which is VFS-first, so the lazy path sees exactly the
/// bytes the feeder installed. Keeping it would have left the pairing invariant
/// "structural for the lazy store, only as good as its install site otherwise" —
/// i.e. exactly the per-call-site convention that failed here in the first place.
pub(crate) fn vector_store_and_host() -> Result<(
    std::rc::Rc<moda_text::vector::VectorFontStore>,
    Arc<TextHost>,
)> {
    let (host, generation) = text_host_snapshot()?;
    let cached = VECTOR_STORE.with(|cell| match cell.borrow().as_ref() {
        Some((cached_gen, store)) if *cached_gen == generation => Some(store.clone()),
        _ => None,
    });
    if let Some(store) = cached {
        return Ok((store, host));
    }
    // Built from the map the host extracted at load — never a fresh read of the
    // fonts dir, which a warm host is explicitly allowed to outlive
    // (`moda_fonts::prune_fonts_by_canvas` may sweep a materialized per-canvas
    // dir from another process while this one still holds the parsed host).
    let store = std::rc::Rc::new(host.vector_font_store_lenient());
    VECTOR_STORE.with(|cell| *cell.borrow_mut() = Some((generation, store.clone())));
    Ok((store, host))
}

/// The loaded fonts as a text measurer needs them (moda-layout's `ModaText::vector`): the vector font store and the
/// text host whose catalog it was built from. `None` when no fonts are installed.
pub fn layout_fonts() -> Option<(
    std::rc::Rc<moda_text::vector::VectorFontStore>,
    Arc<TextHost>,
)> {
    vector_store_and_host().ok()
}

fn try_build_vector_batch(
    spec: &TextNodeSpec,
    model_matrix: [f32; 16],
    z: f64,
    page_clip: [f32; 4],
    fill_shader: Option<crate::seams::TextShaderFill>,
    render_scale: f64,
    fall_label: &str,
    placement: moda_text::vector::VectorPlacement,
) -> Result<Option<TextBuildOutcome>> {
    // vork: the vector lane's batch is built by moda-text-draw's `try_build_vector_batch` (the same code, ported
    // there with the GPU side); text shader fills are not ported. See PROVENANCE.md.
    if fill_shader.is_some() {
        bail!("text shader fills are not ported");
    }
    // One snapshot: store and catalog must come from the SAME host.
    let (store, host) = vector_store_and_host()?;
    // Text in a family this catalog can't draw draws in a stand-in: say so (whichever lane draws it).
    for (requested, drawn) in moda_text::runs::substituted_families(&spec.html, &host.catalog) {
        moda_assets::report::record(moda_assets::report::RuntimeEvent::FontSubstituted {
            node_id: spec.node_id.clone(),
            requested,
            drawn,
        });
    }
    match moda_text_draw::try_build_vector_batch(&store, &host, spec, model_matrix, z, page_clip, render_scale, placement)? {
        Ok(moda_text_draw::TextBuildOutcome::Item(batch)) => {
            if fall_label == moda_assets::report::VECTOR_LANE_LABEL {
                moda_assets::report::record_vector_text_served(&spec.node_id);
            }
            Ok(Some(TextBuildOutcome::Item(Box::new(DrawItem::Text(batch)))))
        }
        Ok(moda_text_draw::TextBuildOutcome::Empty) => {
            if fall_label == moda_assets::report::VECTOR_LANE_LABEL {
                moda_assets::report::record_vector_text_served(&spec.node_id);
            }
            Ok(Some(TextBuildOutcome::Empty))
        }
        Ok(moda_text_draw::TextBuildOutcome::Degraded { reason }) => Ok(Some(TextBuildOutcome::Degraded { reason })),
        Err(reason) => {
            moda_assets::report::record(moda_assets::report::RuntimeEvent::FontFallbackClass {
                node_id: spec.node_id.clone(),
                reason: format!("{fall_label}: {reason}"),
                cause: format!("{fall_label}: {reason}"),
                families: Vec::new(),
            });
            Ok(None)
        }
    }
}

// ---------------------------------------------------------------------------
// Plan side — node → TextQuadBatch
// ---------------------------------------------------------------------------

/// vork: a richtext builder a host installs for this thread (`install_text_builder`): given the node, its model
/// matrix, z, effective clip and render scale, it returns the node's draw. When using plan caches, a host
/// must put the identity/revision of externally held content on the node so it participates in the text key.
/// See PROVENANCE.md.
pub type TextBuilder = std::rc::Rc<dyn Fn(&RawNode, [f32; 16], f64, [f32; 4], f64) -> Result<TextBuildOutcome>>;

thread_local! {
    static TEXT_BUILDER: RefCell<Option<TextBuilder>> = const { RefCell::new(None) };
}

/// Bounds supplied by the host's actual drawing, for captures of externally laid-out text.
pub fn host_text_bounds(node: &RawNode) -> Result<Option<[f64; 4]>> {
    let Some(build) = TEXT_BUILDER.with(|b| b.borrow().clone()) else { return Ok(None) };
    let identity = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.];
    match build(node, identity, 0., [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY], 1.)? {
        TextBuildOutcome::Item(item) => match *item {
            crate::seams::DrawItem::Text(batch) => Ok(batch.local_bounds()),
            _ => Ok(None),
        },
        _ => Ok(None),
    }
}

/// vork: install (or, with `None`, remove) this thread's richtext builder; [`build_richtext_item`] hands
/// uncached richtext nodes to it.
pub fn install_text_builder(builder: Option<TextBuilder>) {
    TEXT_BUILDER.with(|b| *b.borrow_mut() = builder);
}

/// Outcome of planning one text item.
pub enum TextBuildOutcome {
    Item(Box<DrawItem>),
    /// Browser renders this node via Canvas2D — fallback class; the page must
    /// report Degraded/unsupported in the manifest, never silently differ.
    Degraded {
        reason: String,
    },
    /// Nothing to draw (no content / fully empty layout).
    Empty,
}

/// `affineToMatrix4x4` of the node's canvas affine. The affine already carries
/// the node's OWN uniform `scale` (folded into the local TRS by
/// `accumulated_affine` → `scale_xy`), so the model matrix must NOT re-apply it.
///
/// This used to post-multiply by `Scale(node.scale)`, but
/// `node_canvas_affine` ALSO folds the node's own scale, so it double-applied —
/// text rendered at `scale²` whenever `scale != 1`. That stayed latent until
/// animation began driving text `scale` (every shape renders `scale¹`), at which
/// point animated text grew quadratically AND off-center. Building the model
/// straight from the canvas affine applies `scale` exactly once.
fn richtext_model_matrix(scene: &Scene, node_id: &str) -> Option<[f32; 16]> {
    let a = scene.node_canvas_affine(node_id)?;
    Some([
        a.a as f32, a.b as f32, 0.0, 0.0, //
        a.c as f32, a.d as f32, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        a.e as f32, a.f as f32, 0.0, 1.0,
    ])
}

/// Extract the renderWithMsdf fill parameters (`resolveTextNodeFill`,
/// `WebGPUTextSystem.ts:526-624`). Shader fills (`fillPriority: 'shader'`)
/// land in M4 (PLAN "richtext shader fills") — per-run colors apply, matching
/// the production ineligible-shader fallback.
fn extract_fill(node: &RawNode) -> (Option<String>, Option<GradientSpec>) {
    // The glyph-colour override: the `textFills` stack folded to one paint
    // (E3.1 P6; its top layer when it doesn't fold).
    use moda_scene::paint::Paint;
    match crate::paint::layers::text_fill_paint(node) {
        Some(Paint::Solid(s)) => (s.color.as_str().map(str::to_string), None),
        Some(Paint::Linear(l)) => (
            None,
            Some(GradientSpec {
                fill_type: 1,
                start: (l.start.x, l.start.y),
                end: (l.end.x, l.end.y),
                rotation: 0.0,
                stops: text_paint_stops(&l.stops, l.options.repeat),
                flags: l.options.shader_flags(),
            }),
        ),
        Some(Paint::Radial(r)) => {
            // `node.fillRadialGradientRotation ? deg·π/180 : 0` (falsy).
            let rotation = if r.rotation != 0.0 {
                r.rotation * std::f64::consts::PI / 180.0
            } else {
                0.0
            };
            (
                None,
                Some(GradientSpec {
                    fill_type: 2,
                    start: (r.center.x, r.center.y),
                    end: (r.radius_x, r.radius_y),
                    rotation,
                    stops: text_paint_stops(&r.stops, r.options.repeat),
                    flags: r.options.shader_flags(),
                }),
            )
        }
        // Shader fills pack separately (`extract_shader_fill_with`); per-run
        // colours apply underneath.
        _ => (None, None),
    }
}

fn text_paint_stops(stops: &[moda_scene::paint::GradientStop], repeat: Option<bool>) -> Vec<(f64, [f64; 4])> {
    if repeat != Some(true) {
        return parse_gradient_stops(Some(&moda_scene::paint::stops_to_flat(stops).iter().map(Value::to_json).collect()));
    }
    crate::paint::gradient::parse_repeating_stops(stops, repeat)
        .iter()
        .map(|stop| (stop.offset, [stop.color.r, stop.color.g, stop.color.b, stop.color.a]))
        .collect()
}

/// The text stroke pass's colour: the top stroke's solid colour. The text
/// lane has one stroke pass and no gradient stroke pipeline: a gradient stroke
/// draws its first stop, and a stack of glyph strokes draws its top layer (the
/// manifest reports `paint:text-stack`; E3.1 P6).
fn text_stroke_color(node: &RawNode) -> Option<String> {
    moda_scene::paint::top_stroke(node)
        .and_then(|s| moda_scene::paint::representative_color(&s.paint).map(str::to_string))
}

/// `||0` falsy coercion for numeric node fields (NaN → 0).
fn falsy_f64(node: &RawNode, key: &str) -> f64 {
    match node.json_field(key) {
        Some(Value::Number(n)) => {
            let v = n.as_f64().unwrap_or(0.0);
            if v.is_nan() {
                0.0
            } else {
                v
            }
        }
        Some(Value::Bool(true)) => 1.0,
        _ => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Degraded rasters sized to the screen (camera frames)
// ---------------------------------------------------------------------------

thread_local! {
    /// Camera frames: device px per canvas unit (zoom × dpr) rounded UP to a power
    /// of two, set for the duration of a camera plan build. Export / page plans
    /// leave it unset and rasterise Degraded text at its full `QUALITY_SCALE`.
    static DEGRADED_CAMERA_SCALE: std::cell::Cell<Option<f64>> = const { std::cell::Cell::new(None) };
    /// How many Degraded text items in the current camera build were drawn coarser
    /// than full quality or skipped as too small — non-zero, the plan depends on
    /// the zoom. A count, so a caller can tell whether one node's build added any.
    static DEGRADED_ZOOM_LIMITED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// `px_per_unit` (zoom × dpr) rounded up to its power of two: the camera scale
/// Degraded rasters are sized for, so a zoom inside one octave reuses them and a
/// raster is never coarser than the screen.
pub fn degraded_camera_bucket(px_per_unit: f64) -> f64 {
    if !(px_per_unit.is_finite() && px_per_unit > 0.0) {
        return 1.0;
    }
    (px_per_unit.log2() - 1e-9).ceil().exp2()
}

/// Restores the previous camera scale on drop.
pub struct DegradedCameraGuard(Option<f64>, u64);

impl Drop for DegradedCameraGuard {
    fn drop(&mut self) {
        DEGRADED_CAMERA_SCALE.with(|c| c.set(self.0));
        DEGRADED_ZOOM_LIMITED.with(|c| c.set(self.1));
    }
}

/// Enter a camera plan build at `bucket` ([`degraded_camera_bucket`]) device px
/// per canvas unit; `None` restores full-size rasters. Zeroes the zoom-limited
/// count ([`degraded_zoom_limited_count`]) for this build.
pub fn enter_degraded_camera(bucket: Option<f64>) -> DegradedCameraGuard {
    DegradedCameraGuard(
        DEGRADED_CAMERA_SCALE.with(|c| c.replace(bucket)),
        DEGRADED_ZOOM_LIMITED.with(|c| c.replace(0)),
    )
}

/// Degraded text items the current camera build drew below full quality or
/// skipped as too small (so a finer zoom must rebuild them).
pub fn degraded_zoom_limited_count() -> u64 {
    DEGRADED_ZOOM_LIMITED.with(std::cell::Cell::get)
}

/// Whether the current camera build has any zoom-limited Degraded text.
pub fn degraded_zoom_limited() -> bool {
    degraded_zoom_limited_count() > 0
}

fn note_degraded_zoom_limited() {
    DEGRADED_ZOOM_LIMITED.with(|c| c.set(c.get() + 1));
}

/// Max real local→screen axis scale of a model matrix.
fn model_max_axis_scale(model_matrix: &[f32; 16]) -> f64 {
    let sx = (model_matrix[0] as f64).hypot(model_matrix[1] as f64);
    let sy = (model_matrix[4] as f64).hypot(model_matrix[5] as f64);
    sx.max(sy)
}

/// Device px per logical text px for a node with `model_matrix` in the current
/// camera build, or `None` outside one (full-size rasters).
fn degraded_screen_scale(model_matrix: &[f32; 16]) -> Option<f64> {
    DEGRADED_CAMERA_SCALE
        .with(std::cell::Cell::get)
        .map(|cam| cam * model_max_axis_scale(model_matrix))
}

/// In a camera build, a text node whose box is under
/// [`moda_text::degraded::MIN_ON_SCREEN_PX`] on screen in either direction draws
/// nothing, on every lane: its glyphs would cover less than a pixel. At fit on a
/// 100k-node document that is most of its text, and its vector glyph data was most
/// of the fit plan's memory. Counted as zoom-limited, so zooming in rebuilds it.
fn skip_text_too_small(spec: &TextNodeSpec, model_matrix: &[f32; 16]) -> bool {
    let Some(screen) = degraded_screen_scale(model_matrix) else {
        return false;
    };
    let (Some(w), Some(h)) = (spec.width, spec.height) else {
        return false;
    };
    let min = moda_text::degraded::MIN_ON_SCREEN_PX;
    // A zero or unset box says nothing about how big the text draws (it overflows).
    if w > 0.0
        && h > 0.0
        && w.is_finite()
        && h.is_finite()
        && (w * screen < min || h * screen < min)
    {
        note_degraded_zoom_limited();
        return true;
    }
    false
}

/// A cached text batch is still good to serve here: it carries no Degraded raster,
/// or one at least as fine as the current camera needs. (A raster built zoomed out
/// is coarse; zooming in must re-rasterise it.)
fn degraded_hit_serves(batch: &crate::seams::TextQuadBatch, model_matrix: &[f32; 16]) -> bool {
    let Some(d) = &batch.degraded else {
        return true;
    };
    let q = d.raster.quality_scale;
    if q < moda_text::degraded::QUALITY_SCALE {
        match degraded_screen_scale(model_matrix) {
            Some(need) if q >= need.min(moda_text::degraded::QUALITY_SCALE) * (1.0 - 1e-6) => {
                note_degraded_zoom_limited()
            }
            _ => return false,
        }
    }
    true
}

/// A text node's draw: the vector lane (glyph outlines), or, when it declines (complex script, a glyph no
/// fallback face covers), the degraded raster tier with the bundled fallback set.
fn vector_or_degraded(
    spec: &TextNodeSpec,
    model: [f32; 16],
    z: f64,
    clip: [f32; 4],
    fill_shader: Option<crate::seams::TextShaderFill>,
    render_scale: f64,
) -> Result<TextBuildOutcome> {
    if let Some(out) = try_build_vector_batch(
        spec,
        model,
        z,
        clip,
        fill_shader,
        render_scale,
        moda_assets::report::VECTOR_LANE_LABEL,
        moda_text::vector::VectorPlacement::VectorParity,
    )? {
        return Ok(out);
    }
    build_degraded_batch(spec, model, z, clip, render_scale).or_else(|e| {
        Ok(TextBuildOutcome::Degraded {
            reason: format!("vector declined; degraded raster failed: {e}"),
        })
    })
}

/// M4d: build the Degraded-tier raster batch for a fallback-class node
/// (`--best-effort-text`). Placement mirrors the Canvas2D display quad
/// (`richtext/transforms.ts:77-117`): quad = texture logical size, alignment
/// offsets against the node box, opacity × the raster's uniform-alpha
/// `fill_alpha`.
///
/// In a camera build ([`enter_degraded_camera`]) the raster is sized to the
/// screen and text under about a pixel is skipped (`Empty`). Elsewhere (an export
/// or capture at `render_scale` px per canvas unit) it is as fine as the output
/// ([`moda_text::degraded::output_raster_scale`]). The scale it uses, `render_scale`
/// × the model's MIN axis scale, is what [`text_cache_key`] keys, so a cached raster
/// is never served at another scale.
fn build_degraded_batch(
    spec: &TextNodeSpec,
    model_matrix: [f32; 16],
    z: f64,
    page_clip: [f32; 4],
    render_scale: f64,
) -> Result<TextBuildOutcome> {
    // vork: moda-text-draw's `build_degraded_batch` (the same code, ported there) builds the raster batch; it has no
    // camera-build branch (vork's engine draws no live camera). See PROVENANCE.md.
    #[cfg(test)]
    crate::test_fonts();
    let engine = moda_text::degraded::engine().map_err(|e| anyhow!("degraded engine: {e}"))?;
    Ok(match moda_text_draw::build_degraded_batch(&engine, spec, model_matrix, z, page_clip, render_scale)? {
        moda_text_draw::TextBuildOutcome::Item(batch) => TextBuildOutcome::Item(Box::new(DrawItem::Text(batch))),
        moda_text_draw::TextBuildOutcome::Empty => TextBuildOutcome::Empty,
        moda_text_draw::TextBuildOutcome::Degraded { reason } => TextBuildOutcome::Degraded { reason },
    })
}

/// `WebGPUTextSystem.resolveNodeEffects` (WebGPUTextSystem.ts:551-585):
/// resolve a node's text effects (`effects` entries of type drop-shadow,
/// glow, echo, extrude) for the text renderers — drop hidden entries
/// (`visible: false`), drop shadows that draw nothing (no colour, or neither
/// a blur nor an offset: `effect::drop_shadow_draws`) and every other type,
/// keep node array order (back-to-front). Effect colors are literals (the
/// editor keeps one next to every variable binding), so a
/// missing/non-string glow / extrude color falls to the browser's
/// `?? '#000000'` default here; a drop shadow needs a string colour.
pub(crate) fn extract_node_effects(node: &RawNode) -> Vec<moda_text::TextEffectSpec> {
    use moda_text::TextEffectSpec as E;
    let mut out = Vec::new();
    for entry in moda_scene::effect::node_effects_json(node) {
        let Some(o) = entry.as_object() else {
            continue;
        };
        // `isEffectVisible`: only a strict `visible: false` hides.
        if !moda_scene::effect::is_visible_json(entry) {
            continue;
        }
        let f = |k: &str| o.get(k).and_then(Value::as_f64);
        let s = |k: &str| o.get(k).and_then(Value::as_str).map(str::to_string);
        match o.get("type").and_then(Value::as_str) {
            Some("drop-shadow") => {
                use serde::Deserialize as _;
                let Ok(shadow) = moda_scene::effect::DropShadow::deserialize(entry) else {
                    continue;
                };
                let Some(color) = shadow.color.as_str().filter(|c| !c.is_empty()) else {
                    continue;
                };
                // Text takes no spread (Figma): a shadow that only spreads
                // draws nothing on glyphs.
                if !moda_scene::effect::drop_shadow_draws(&shadow)
                    || (shadow.blur <= 0.0 && shadow.offset.is_zero())
                {
                    continue;
                }
                out.push(E::DropShadow(moda_text::TextDropShadow {
                    color: color.to_string(),
                    offset_x: shadow.offset.x,
                    offset_y: shadow.offset.y,
                    // × viewZoom — viewZoom = 1 on the export path.
                    blur: shadow.blur,
                    opacity: shadow.opacity(),
                }));
            }
            // Glows (outer and inner) are the capture lane's, as on every
            // shape (`layer_blur::shadow::capture_glows`): the glyphs'
            // silhouette blurred, composited with the glow's mode.
            Some("echo") => out.push(E::Echo {
                count: f("count"),
                offset_x: f("offsetX"),
                offset_y: f("offsetY"),
                scale_step: f("scaleStep"),
                color_start: s("colorStart"),
                color_end: s("colorEnd"),
                opacity_step: f("opacityStep"),
            }),
            Some("extrude") => out.push(E::Extrude {
                depth_px: f("depthPx"),
                angle_deg: f("angleDeg"),
                color: s("color").unwrap_or_else(|| "#000000".to_string()),
            }),
            _ => {} // unknown type — render-inert in the browser too
        }
    }
    out
}

/// KineticText per-glyph deltas: read the node's `unitTransforms`
/// array (a dense `GlyphTransform[]`, animation-mutated onto the node by the Rust
/// v3 evaluator) into resolved {@link moda_text::vector::ops::GlyphXform}s, one per
/// code point of the canonical plain text. The vector lane maps each glyph → its
/// source code point and indexes this. A null / non-object / missing-field entry
/// resolves to identity so the array stays dense and index-aligned. Empty (the
/// common case) means a static node → byte-exact, no per-glyph cost.
fn extract_unit_transforms(node: &RawNode) -> Vec<moda_text::vector::ops::GlyphXform> {
    use moda_text::vector::ops::GlyphXform;
    let Some(arr) = node.array("unitTransforms") else {
        return Vec::new();
    };
    // Re-apply the evaluator's length cap: a malformed/hostile imported scene can
    // carry an oversized `unitTransforms` array straight to the renderer without
    // passing through the v3 evaluator, so bound per-frame allocation here too
    // (by the text's own length past the fixed cap).
    arr.iter()
        .take(unit_transforms_cap(node, arr.len()))
        .map(|entry| {
            let Some(o) = entry.as_object() else {
                return GlyphXform::IDENTITY; // null / non-object → identity (keep dense)
            };
            let f = |k: &str, d: f64| {
                o.get(k)
                    .and_then(Value::as_f64)
                    .filter(|v| v.is_finite())
                    .unwrap_or(d)
            };
            GlyphXform::resolve(
                f("offsetX", 0.0),
                f("offsetY", 0.0),
                f("scale", 1.0),
                f("rotation", 0.0),
                f("opacity", 1.0),
            )
        })
        .collect()
}

/// The `unitTransforms` entries honored for `node` (`moda_text::vector::ops::unit_transforms_cap`).
fn unit_transforms_cap(node: &RawNode, len: usize) -> usize {
    moda_text::vector::ops::unit_transforms_cap(len, node.str("htmlContent").unwrap_or(""))
}

/// KineticText units: each `unitTransforms` entry's `unit` (the first code point of the word or line it moves
/// with), the entry's own index where it has none. Empty when no entry names another unit (every glyph its own).
fn extract_unit_groups(node: &RawNode) -> Vec<u32> {
    let Some(arr) = node.array("unitTransforms") else {
        return Vec::new();
    };
    let groups: Vec<u32> = arr
        .iter()
        .take(unit_transforms_cap(node, arr.len()))
        .enumerate()
        .map(|(i, entry)| {
            entry
                .get("unit")
                .and_then(Value::as_u64)
                .filter(|u| (*u as usize) <= i)
                .map_or(i as u32, |u| u as u32)
        })
        .collect();
    if groups.iter().enumerate().all(|(i, g)| *g as usize == i) {
        return Vec::new();
    }
    groups
}

/// KineticText fill: each `unitTransforms` entry's `fill` (how much of the glyph's fill draws, 0…1; 1 where it has
/// none). Empty when no entry sets one.
fn extract_unit_fill(node: &RawNode) -> Vec<f32> {
    use moda_text::vector::ops::UNIT_TRANSFORMS_MAX_LENGTH;
    let Some(arr) = node.array("unitTransforms") else {
        return Vec::new();
    };
    // not-a-node-key: a unitTransforms entry's fill amount (0…1)
    if !arr.iter().any(|e| e.get("fill").is_some()) {
        return Vec::new();
    }
    arr.iter()
        .take(UNIT_TRANSFORMS_MAX_LENGTH)
        .map(|e| {
            // not-a-node-key: a unitTransforms entry's fill amount (0…1)
            e.get("fill")
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite())
                .map_or(1.0, |v| v.clamp(0.0, 1.0) as f32)
        })
        .collect()
}

/// Fill wipe (`fillWipe`: 0…1, `fillWipeAngle`: degrees, 0 = left to right, 90 = top to bottom): the glyphs' fill
/// shows only behind a straight edge that crosses the text's lines as `fillWipe` goes from 0 (no fill) to 1 (all of
/// it); the text's stroke and effects draw throughout, so a stroked text goes from outline to filled. The fill is
/// the text's solid colour (its fill paint, else its first character's colour) as a hard-stop gradient over the
/// laid-out lines' box; a gradient- or shader-filled text is left as it is.
fn apply_fill_wipe(node: &RawNode, spec: &mut TextNodeSpec) {
    let Some(p) = node.f64("fillWipe").filter(|p| p.is_finite()) else {
        return;
    };
    let p = p.clamp(0.0, 1.0);
    if p >= 1.0 || spec.gradient.is_some() || spec.shader_fill {
        return;
    }
    let color = spec.fill_override.clone().or_else(|| {
        let (_, host) = vector_store_and_host().ok()?;
        moda_text::runs::extract_msdf_runs_by_paragraph(&spec.html, &host.catalog)
            .first()
            .map(|r| r.color.clone())
    });
    let Some(c) = color.as_deref().and_then(moda_text::color::css_color_to_normalized_rgba) else {
        return;
    };
    let (w, h) = (spec.width.unwrap_or(400.0).max(1.0), spec.height.unwrap_or(200.0).max(1.0));
    // The lines' box, else the node's.
    let rich = moda_text::RichTextNodeSpec {
        node_id: spec.node_id.clone(),
        html_content: spec.html.to_string(),
        width: spec.width,
        height: spec.height,
        paragraph_spacing: spec.paragraph_spacing,
        line_height_policy: spec.line_height_policy,
        vertical_align: spec.vertical_align.clone(),
        truncation: None,
        auto_width: spec.auto_width,
        area: spec.area.clone(),
    };
    let extent = crate::text_metrics::rich_text_metrics(&rich)
        .ok()
        .filter(|m| !m.lines.is_empty())
        .map(|m| {
            m.lines.iter().fold(
                [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
                |b, l| [b[0].min(l.x), b[1].min(l.top), b[2].max(l.x + l.width), b[3].max(l.top + l.height)],
            )
        })
        .unwrap_or([0.0, 0.0, w, h]);
    let (start, end) = wipe_line(extent, node.f64("fillWipeAngle").filter(|a| a.is_finite()).unwrap_or(0.0));
    let clear = [c[0], c[1], c[2], 0.0];
    spec.gradient = Some(GradientSpec {
        fill_type: 1,
        start: (start.0 / w * 100.0, start.1 / h * 100.0),
        end: (end.0 / w * 100.0, end.1 / h * 100.0),
        rotation: 0.0,
        stops: vec![(0.0, c), (p, c), (p, clear), (1.0, clear)],
        flags: 0.0,
    });
    spec.fill_override = None;
}

/// A wipe's gradient line over the box `[x0, y0, x1, y1]` at `angle_deg`: through its centre, from the corner the
/// edge leaves to the one it reaches last (px).
pub(crate) fn wipe_line(b: [f64; 4], angle_deg: f64) -> ((f64, f64), (f64, f64)) {
    let (sin, cos) = angle_deg.to_radians().sin_cos();
    let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
    let half = ((b[2] - b[0]).abs() * cos.abs() + (b[3] - b[1]).abs() * sin.abs()) / 2.0;
    ((cx - cos * half, cy - sin * half), (cx + cos * half, cy + sin * half))
}

/// E6.3 "Truncate text" (`textTruncation: 'ending'`): lines past `maxLines`
/// or below the node box are hidden and the last shown line ends in `…`.
/// `None` when truncation is off.
pub fn node_truncation(node: &RawNode) -> Option<moda_text::truncation::TextTruncation> {
    if node.str("textTruncation") != Some("ending") {
        return None;
    }
    let max_lines = node
        .f64("maxLines")
        .filter(|n| n.is_finite() && *n >= 1.0)
        .map(|n| n.floor() as usize);
    let max_height = node.f64("height").filter(|h| h.is_finite() && *h >= 0.0);
    Some(moda_text::truncation::TextTruncation {
        max_lines,
        max_height,
    })
}

/// F7.2 type on a path: the node's `textPath` — `points` (PathPoint JSON, node-local px at the node size
/// `boxWidth` × `boxHeight`, scaled with the node when it is resized), `closed`, `startOffset` (0..1), `flip`,
/// `align` — flattened for `moda_text::text_path`. `None` without one, or when the path has no length.
pub fn node_text_path(node: &RawNode) -> Option<moda_text::text_path::TextPathSpec> {
    use moda_text::text_path::{flatten, PathCmd, TextPathAlign, TextPathSpec};
    let tp = node.json_field("textPath")?;
    let points = tp.get("points")?.as_array()?;
    let num = |v: Option<&Value>| v.and_then(Value::as_f64).filter(|n| n.is_finite());
    let (w, h) = (
        node.f64("width").unwrap_or(0.0),
        node.f64("height").unwrap_or(0.0),
    );
    let sx = match num(tp.get("boxWidth")) {
        Some(bw) if bw > 1e-9 && w > 0.0 => w / bw,
        _ => 1.0,
    };
    let sy = match num(tp.get("boxHeight")) {
        Some(bh) if bh > 1e-9 && h > 0.0 => h / bh,
        _ => 1.0,
    };
    let xy =
        |v: &Value| -> Option<(f64, f64)> { Some((num(v.get("x"))? * sx, num(v.get("y"))? * sy)) };
    let mut cmds = Vec::with_capacity(points.len());
    for (i, p) in points.iter().enumerate() {
        let (x, y) = xy(p)?;
        let kind = p.get("type").and_then(Value::as_str).unwrap_or("line");
        cmds.push(match kind {
            _ if i == 0 => PathCmd::Move(x, y),
            "move" => PathCmd::Move(x, y),
            "curve" => {
                let (c1, c2) = (xy(p.get("cp1")?)?, xy(p.get("cp2")?)?);
                PathCmd::Cubic(c1.0, c1.1, c2.0, c2.1, x, y)
            }
            "quadratic" => {
                let c = xy(p.get("cp")?)?;
                PathCmd::Quad(c.0, c.1, x, y)
            }
            // Arcs (rare in drawn paths) follow their chord.
            _ => PathCmd::Line(x, y),
        });
    }
    let points = flatten(&cmds);
    let closed = tp.get("closed").and_then(Value::as_bool).unwrap_or(false);
    moda_text::text_path::PathMeasure::new(&points, closed, false)?;
    Some(TextPathSpec {
        points,
        closed,
        start_offset: num(tp.get("startOffset")).unwrap_or(0.0).clamp(0.0, 1.0),
        flip: tp.get("flip").and_then(Value::as_bool).unwrap_or(false),
        align: TextPathAlign::parse(
            tp.get("align")
                .and_then(Value::as_str)
                .unwrap_or("baseline"),
        ),
    })
}

/// Extract the `TextNodeSpec` for a richtext node — shared by the MSDF
/// plan path, the Degraded tier, and M4d's text-as-mask source
/// (`effects::mask::text_mask_source`).
/// E7.20: `resizeMode: 'auto-width'` — the text never wraps at the box edge, until the box reaches its
/// `maxWidth` (Figma's max width): the editor clamps the box there and the lines wrap at it. Mirrors
/// `autoWidthWrapsAtMax` / `isUnwrappedAutoWidth` in the editor's `core/scene/textAutoWidth.ts`.
pub fn node_auto_width(node: &RawNode) -> bool {
    if node.str("resizeMode") != Some("auto-width") {
        return false;
    }
    match node.f64("maxWidth") {
        Some(max) if max > 0.0 => node.f64("width").unwrap_or(0.0) < max - MAX_WIDTH_REACHED_TOLERANCE_PX,
        _ => true,
    }
}

/// A box within this of its max width has reached it (the editor's `MAX_WIDTH_REACHED_TOLERANCE_PX`).
const MAX_WIDTH_REACHED_TOLERANCE_PX: f64 = 0.5;

/// The document's `locale` (`config.locale`): the language of text that declares none (`moda_text::lang`), so its
/// CJK falls back to that region's face.
pub fn document_locale(scene: &Scene) -> Option<&str> {
    scene.doc().config.raw.get("locale").and_then(|v| v.as_str())
}

/// A richtext node's HTML as it lays out in `scene`: its `htmlContent` with the document's locale folded in
/// (`moda_text::lang::html_with_document_lang`).
pub fn node_html_in(scene: &Scene, node: &RawNode) -> String {
    moda_text::lang::html_with_document_lang(node.str("htmlContent").unwrap_or(""), document_locale(scene))
}

/// [`node_to_text_spec`] in its document: the HTML takes the document's locale ([`node_html_in`]).
pub fn node_to_text_spec_in(scene: &Scene, node: &RawNode) -> TextNodeSpec {
    let mut spec = node_to_text_spec(node);
    spec.html = moda_text::lang::html_with_document_lang(&spec.html, document_locale(scene)).into();
    spec
}

pub fn node_to_text_spec(node: &RawNode) -> TextNodeSpec {
    let (fill_override, gradient) = extract_fill(node);
    TextNodeSpec {
        node_id: node.id().to_string(),
        html: node.str("htmlContent").unwrap_or("").into(),
        width: node.f64("width"),
        height: node.f64("height"),
        paragraph_spacing: falsy_f64(node, "paragraphSpacing"),
        line_height_policy: Default::default(),
        vertical_align: node.str("verticalAlign").map(str::to_string),
        truncation: node_truncation(node),
        text_path: node_text_path(node),
        auto_width: node_auto_width(node),
        opacity: node.f64("opacity").unwrap_or(1.0),
        stroke_color: text_stroke_color(node),
        stroke_width: moda_scene::paint::node_stroke_style(node).width,
        fill_override,
        gradient,
        // Set by build_richtext_item AFTER the eligibility check + packing
        // (extract_shader_fill_with); the mask/Degraded consumers never route
        // through the shader-fill pipeline.
        shader_fill: false,
        effects: extract_node_effects(node),
        unit_transforms: extract_unit_transforms(node).into(),
        unit_groups: extract_unit_groups(node).into(),
        unit_fill: extract_unit_fill(node).into(),
        // A Moda node's runs draw in their HTML's colours.
        run_colors: Default::default(),
        unit_range: None,
        // The node's distort warp as f32 rows for the vector lane's
        // DrawUniforms. Layout is untouched — the warp applies at draw time.
        // `node_distort` h33-normalizes post-validation, so these casts can't
        // overflow for a valid (possibly scale-represented) stored matrix.
        distort: crate::paint::node_distort(node).map(|d| {
            let t = d.local;
            [
                t.h11 as f32,
                t.h12 as f32,
                t.h13 as f32,
                t.h21 as f32,
                t.h22 as f32,
                t.h23 as f32,
                t.h31 as f32,
                t.h32 as f32,
                t.h33 as f32,
            ]
        }),
        area: None,
    }
}

/// Vector lane: identical extraction with the VECTOR eligibility predicate
/// (`supports_vector_text_fill` — the manifest asserts the eligible-id sets
/// are identical, but each engine consults its own production predicate).
fn extract_shader_fill_with(
    scene: &Scene,
    node: &RawNode,
    shader_time_s: f64,
) -> Option<crate::seams::TextShaderFill> {
    let Some(moda_scene::paint::Paint::Shader(shader)) =
        moda_scene::paint::node_text_fills(node).pop()
    else {
        return None;
    };
    let shader_id = shader.shader_id.as_str();
    let params = Value::Object(shader.params.clone().unwrap_or_default());
    let packed = if crate::agent_fill::is_code_ref(shader_id) {
        // An agent shader (a code key): every agent fill draws on text (crate::agent_fill).
        match crate::agent_fill::pack(scene, node, shader_id, &params, shader_time_s) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("moda-engine: warn: {e}");
                return None;
            }
        }
    } else {
        // `supportsVectorTextFill`: the browser predicate as extracted manifest data (descriptor `shade`, no
        // primaryImage).
        if !moda_shaders::runtime::supports_vector_text_fill(shader_id) {
            return None;
        }
        moda_shaders::runtime::pack_fill(shader_id, &params.to_json(), shader_time_s).ok()?
    };

    let node_opacity = node.f64("opacity").unwrap_or(1.0);
    // strokeCss feeds only the packed strokeColor (ABI parity — the fill
    // fragment never reads it; stroke is a separate solid pass).
    let stroke_color: [f32; 4] = text_stroke_color(node)
        .as_deref()
        .and_then(moda_text::color::css_color_to_normalized_rgba)
        .map(|c| {
            [
                c[0] as f32,
                c[1] as f32,
                c[2] as f32,
                (c[3] * node_opacity) as f32,
            ]
        })
        .unwrap_or([0.0; 4]);
    // boxW/boxH exactly as renderWithMsdf computes them
    // (`WebGPUTextSystem.ts:657-659`); bounds dims clamped >= 1 (zero-size
    // box NaN guard, textShaderFillData.ts).
    let box_w = node.f64("width").unwrap_or(400.0).max(0.0);
    let box_h = node.f64("height").unwrap_or(200.0).max(0.0);
    Some(crate::seams::TextShaderFill {
        packed,
        fill_color: [1.0, 1.0, 1.0, node_opacity as f32],
        stroke_color,
        bounds: [0.0, 0.0, box_w.max(1.0) as f32, box_h.max(1.0) as f32],
    })
}

/// Plan one richtext node (`renderWithMsdf` orchestration). `page_clip` must
/// be the EFFECTIVE clip (page ∧ ancestor rect-mask AABBs) — text is NOT
/// clipped to the node rect (WebGPUTextSystem.ts:890-891). `shader_time_s`
/// only matters for shader fills (renderer-clock params; 0 on exports).
#[allow(clippy::too_many_arguments)]
pub fn build_richtext_item(
    scene: &Scene,
    node: &RawNode,
    z: f64,
    effective_clip: [f32; 4],
    best_effort: bool,
    shader_time_s: f64,
    render_scale: f64,
    caches: Option<TextCacheCtx<'_>>,
) -> Result<TextBuildOutcome> {
    // vork: a host that lays out its own text installs a builder (`install_text_builder`), which draws the node
    // at the model matrix this lane computes. See PROVENANCE.md.
    if let Some(build) = TEXT_BUILDER.with(|b| b.borrow().clone()) {
        let model = richtext_model_matrix(scene, node.id())
            .ok_or_else(|| anyhow!("no canvas affine for richtext {}", node.id()))?;
        // The host supplies a content revision on the node for fields held outside the Moda record.
        // Use the editor's key, bounded cache and placement patch unchanged for these batches too.
        let mut slot = None;
        if let Some((cache, hashes)) = caches {
            let key = text_cache_key(scene, hashes, node, best_effort, render_scale, model_min_axis_scale(&model), 0);
            if let Some(mut batch) = cache.text_get_content(node.id(), key).filter(|b| degraded_hit_serves(b, &model)) {
                patch_cached_batch(&mut batch, z, model, effective_clip, None);
                return Ok(TextBuildOutcome::Item(Box::new(crate::seams::DrawItem::Text(batch.into()))));
            }
            cache.text_build_begin();
            slot = Some((cache, key));
        }
        let mut built = build(node, model, z, effective_clip, render_scale);
        if let Some((cache, key)) = slot {
            name_built_text(&mut built, key);
            cache.text_build_end(node.id(), key, built_text_batch(&built));
        }
        return built;
    }
    let mut spec = node_to_text_spec_in(scene, node);
    // Area type (E8.14): the outline, thread and wrap objects the lines fit; a later frame shows the story of the
    // thread's head.
    crate::text_area::apply_to_spec(scene, node, &mut spec);
    if spec.html.is_empty() {
        return Ok(TextBuildOutcome::Empty);
    }
    // Motion's text transitions: a reflow between two widths (folded into the glyph offsets) and a fill wipe.
    crate::text_metrics::apply_reflow(node, &mut spec);
    let area_key = if spec.area.is_some() {
        crate::text_area::area_key(spec.area.as_ref(), Some(&spec.html))
    } else {
        0
    };
    let threaded = spec.area.as_ref().is_some_and(|a| a.frames.len() > 1);
    let fill_shader = extract_shader_fill_with(scene, node, shader_time_s);
    spec.shader_fill = fill_shader.is_some();
    apply_fill_wipe(node, &mut spec);
    let model = richtext_model_matrix(scene, node.id())
        .ok_or_else(|| anyhow!("no canvas affine for richtext {}", node.id()))?;
    if skip_text_too_small(&spec, &model) {
        return Ok(TextBuildOutcome::Empty);
    }
    // R4 (task #95): full-item residency. Transform/z/clip/shader-clock are
    // patched on hit, so moves and shader-time ticks keep the cached layout.
    let mut cache_slot: Option<(&mut crate::residency::PlanCaches, u64)> = None;
    if let Some((cache, hashes)) = caches {
        let key = text_cache_key(
            scene,
            hashes,
            node,
            best_effort,
            render_scale,
            model_min_axis_scale(&model),
            area_key,
        );
        if let Some(mut batch) = cache
            .text_get_content(node.id(), key)
            .filter(|b| degraded_hit_serves(b, &model))
        {
            // NB: this returns BEFORE `try_build_vector_batch`, where the vector-served
            // and vector→MSDF decline events are recorded: an export job's hit replays
            // what the build recorded (`text_get_content`); the editor's lanes don't
            // collect. See the coupling note on `moda_assets::report::record_vector_text_served`.
            patch_cached_batch(&mut batch, z, model, effective_clip, fill_shader);
            return Ok(TextBuildOutcome::Item(Box::new(
                crate::seams::DrawItem::Text(batch.into()),
            )));
        }
        // A live camera frame lays text out within the frame's refine budget (shared with outlines): past it, a
        // layer not laid out yet draws nothing this frame and is queued, and the next frames lay the queue out
        // through the retained plan, a budget's worth each. A page shown for the first time (thousands of text
        // layers) presents its shapes first and its text over the next frames, instead of blocking on all of it.
        // The first layout of a frame always runs (an edited text layer never waits). Exports and captures lay
        // everything out.
        if cache.text_deferrable() {
            let bbox = scene
                .node_canvas_affine(node.id())
                .map(|a| {
                    let (w, h) = (node.f64("width").unwrap_or(0.0), node.f64("height").unwrap_or(0.0));
                    crate::tess::outline::canvas_bbox([0.0, 0.0, w as f32, h as f32], &a, 0.0)
                })
                .unwrap_or([f64::NEG_INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY]);
            cache.outline_pending_push(node.id(), bbox);
            cache.text_prewarm_push(node.id(), best_effort, render_scale);
            return Ok(TextBuildOutcome::Empty);
        }
        cache.text_build_begin();
        cache_slot = Some((cache, key));
    }
    let built = (|| {
        // The vector lane (glyph outlines; effects[] and per-glyph transforms draw only there). A node it declines
        // (complex script, emoji, a glyph no fallback face covers) draws through the degraded raster tier, the last
        // resort, with the bundled fallback set.
        if let Some(out) = try_build_vector_batch(
            &spec,
            model,
            z,
            effective_clip,
            fill_shader.clone(),
            render_scale,
            moda_assets::report::VECTOR_LANE_LABEL,
            moda_text::vector::VectorPlacement::VectorParity,
        )? {
            return Ok(out);
        }
        if threaded {
            // A later frame of a thread holds the whole story; box text would draw all of it here.
            return Ok(TextBuildOutcome::Empty);
        }
        let _ = best_effort;
        build_degraded_batch(&spec, model, z, effective_clip, render_scale).or_else(|e| {
            Ok(TextBuildOutcome::Degraded {
                reason: format!("vector declined; degraded raster failed: {e}"),
            })
        })
    })();
    let mut built = built;
    if let Some((cache, key)) = cache_slot {
        name_built_text(&mut built, key);
        cache.text_build_end(node.id(), key, built_text_batch(&built));
    }
    built
}

/// Name the batch a build made by its content key (`TextQuadBatch::content_key`), before the cache keeps it: the
/// batch this plan draws and every later hit's copy then carry the same name.
fn name_built_text(built: &mut Result<TextBuildOutcome>, key: u64) {
    if let Ok(TextBuildOutcome::Item(item)) = built {
        if let crate::seams::DrawItem::Text(batch) = item.as_mut() {
            std::sync::Arc::make_mut(batch).content_key = Some(key);
        }
    }
}

/// The text batch a build made, if it made one (what the text cache keeps).
fn built_text_batch(built: &Result<TextBuildOutcome>) -> Option<&crate::seams::TextQuadBatch> {
    match built {
        Ok(TextBuildOutcome::Item(item)) => match item.as_ref() {
            crate::seams::DrawItem::Text(batch) => Some(batch.as_ref()),
            _ => None,
        },
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Shape text (text-in-shape frames) — `WebGPUShapeSystem.renderShapeText`
// (:1258-1343) + `core/scene/textInShapeHelpers.ts`
// ---------------------------------------------------------------------------

/// `getTextBoundsFromShape` (textInShapeHelpers.ts:82-182) — local-space
/// text-area bounds per shape type. `SHAPE_DEFAULTS.textFrame.padding = 8`.
fn text_bounds_from_shape(node: &RawNode, padding: f64) -> Option<(f64, f64, f64, f64)> {
    let w = node.f64("width").unwrap_or(0.0);
    let h = node.f64("height").unwrap_or(0.0);
    match node.node_type() {
        "rectangle" => Some((
            padding,
            padding,
            (w - padding * 2.0).max(0.0),
            (h - padding * 2.0).max(0.0),
        )),
        "ellipse" => {
            let inscribed = std::f64::consts::FRAC_1_SQRT_2;
            let (rx, ry) = (w / 2.0, h / 2.0);
            let irx = (rx * inscribed - padding).max(0.0);
            let iry = (ry * inscribed - padding).max(0.0);
            Some((
                rx - irx,
                ry - iry,
                (irx * 2.0).max(0.0),
                (iry * 2.0).max(0.0),
            ))
        }
        "star" => {
            let (rx, ry) = (w / 2.0, h / 2.0);
            let ratio = node
                .f64("innerRadiusRatio")
                .filter(|r| r.is_finite())
                .map(|r| r.clamp(0.0, 1.0))
                .unwrap_or(0.4);
            let irx = (rx * ratio - padding).max(0.0);
            let iry = (ry * ratio - padding).max(0.0);
            let (sw, sh) = ((irx * 2.0).max(0.0), (iry * 2.0).max(0.0));
            Some((rx - sw / 2.0, ry - sh / 2.0, sw, sh))
        }
        "polygon" => {
            let (rx, ry) = (w / 2.0, h / 2.0);
            let sides = node.f64("sides").unwrap_or(3.0).max(3.0);
            let c = (std::f64::consts::PI / sides).cos();
            let ax = (rx * c - padding).max(0.0);
            let ay = (ry * c - padding).max(0.0);
            let (sw, sh) = ((ax * 2.0).max(0.0), (ay * 2.0).max(0.0));
            Some((rx - sw / 2.0, ry - sh / 2.0, sw, sh))
        }
        "path" => {
            // centroid + min distance to edges (textInShapeHelpers.ts:159-180)
            let points = node.points()?;
            if points.is_empty() {
                return None;
            }
            let pts: Vec<(f64, f64)> = points
                .iter()
                .filter_map(|p| Some((p.f64("x")?, p.f64("y")?)))
                .collect();
            if pts.is_empty() {
                return None;
            }
            let (ox, oy) = (
                node.f64("x").filter(|v| v.is_finite()).unwrap_or(0.0),
                node.f64("y").filter(|v| v.is_finite()).unwrap_or(0.0),
            );
            let n = pts.len() as f64;
            let cx = pts.iter().map(|p| p.0).sum::<f64>() / n + ox;
            let cy = pts.iter().map(|p| p.1).sum::<f64>() / n + oy;
            let closed = node.bool("closed").unwrap_or(false);
            let local_c = (cx - ox, cy - oy);
            let min_dist = if !closed && pts.len() < 3 {
                let (mut minx, mut miny, mut maxx, mut maxy) = (
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                );
                for p in &pts {
                    minx = minx.min(p.0);
                    miny = miny.min(p.1);
                    maxx = maxx.max(p.0);
                    maxy = maxy.max(p.1);
                }
                ((maxx - minx).min(maxy - miny)) / 4.0
            } else {
                let mut min_d = f64::INFINITY;
                let np = pts.len();
                for i in 0..np {
                    let next = if closed { (i + 1) % np } else { i + 1 };
                    if next >= np {
                        continue;
                    }
                    min_d = min_d.min(dist_point_segment(local_c, pts[i], pts[next]));
                }
                if min_d.is_infinite() {
                    0.0
                } else {
                    min_d
                }
            };
            let safe_radius = (min_dist - padding).max(0.0);
            let size = safe_radius * 2.0;
            Some((cx - safe_radius, cy - safe_radius, size, size))
        }
        _ => None,
    }
}

fn dist_point_segment(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len_sq = dx * dx + dy * dy;
    if len_sq == 0.0 {
        return ((p.0 - a.0).powi(2) + (p.1 - a.1).powi(2)).sqrt();
    }
    let t = (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len_sq).clamp(0.0, 1.0);
    let (px, py) = (a.0 + t * dx, a.1 + t * dy);
    ((p.0 - px).powi(2) + (p.1 - py).powi(2)).sqrt()
}

/// `hasTextContent` (textInShapeHelpers.ts:42-53).
fn has_text_content(node: &RawNode) -> bool {
    let Some(content) = node.str("textContent") else {
        return false;
    };
    strip_tags(content).trim() != ""
}

fn strip_tags(html: &str) -> String {
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
    out
}

/// Plan one shape's text frame at `z = shape z + SHAPE_TEXT_Z_OFFSET`
/// (`renderShapeText`). Returns Empty when the shape has no text. The
/// synthetic node takes the PLAIN page clip (`_pageClipBoundsOverride` =
/// `getPageClipFromScene`), padding 0 (the frame bounds already inset), and
/// vertical align default 'middle'. Horizontal alignment is NOT applied
/// outside the HTML (copy the bug — `renderShapeText` line 1334-1339 is a
/// no-op; the golden's "centered" frame text is left-aligned).
#[allow(clippy::too_many_arguments)]
pub fn build_shape_text_item(
    scene: &Scene,
    node: &RawNode,
    z: f64,
    page_clip: [f32; 4],
    best_effort: bool,
    render_scale: f64,
    caches: Option<TextCacheCtx<'_>>,
) -> Result<TextBuildOutcome> {
    if !matches!(
        node.node_type(),
        "rectangle" | "ellipse" | "star" | "polygon" | "path"
    ) {
        return Ok(TextBuildOutcome::Empty);
    }
    if !has_text_content(node) {
        return Ok(TextBuildOutcome::Empty);
    }
    // getTextPadding (textInShapeHelpers.ts:284-315): shape.textPadding, else a default
    // capped relative to the shape so badge-sized shapes keep a usable text area.
    let padding = node.text_frame_padding();
    let Some((bx, by, bw, bh)) = text_bounds_from_shape(node, padding) else {
        return Ok(TextBuildOutcome::Empty);
    };
    if bw <= 0.0 || bh <= 0.0 {
        return Ok(TextBuildOutcome::Empty);
    }
    // getShapeTextFrameWorldTransform: world = abs + rotate(localOffset)·scale
    // (+ page offset). node_canvas_transform already includes the page offset.
    let trs = scene
        .node_canvas_transform(node.id())
        .ok_or_else(|| anyhow!("no canvas transform for {}", node.id()))?;
    let rad = trs.rotation * std::f64::consts::PI / 180.0;
    let (cos, sin) = (rad.cos(), rad.sin());
    let (rox, roy) = if trs.rotation == 0.0 {
        (bx, by)
    } else {
        (bx * cos - by * sin, bx * sin + by * cos)
    };
    let world_x = trs.x + rox * trs.scale_x;
    let world_y = trs.y + roy * trs.scale_y;
    // affineFromTRS (affineTransform.ts:55-73) → affineToMatrix4x4.
    let (a, b, c, d) = (
        cos * trs.scale_x,
        sin * trs.scale_x,
        -sin * trs.scale_y,
        cos * trs.scale_y,
    );
    let model: [f32; 16] = [
        a as f32,
        b as f32,
        0.0,
        0.0, //
        c as f32,
        d as f32,
        0.0,
        0.0, //
        0.0,
        0.0,
        1.0,
        0.0, //
        world_x as f32,
        world_y as f32,
        0.0,
        1.0,
    ];
    let spec = TextNodeSpec {
        node_id: format!("{}-shape-text", node.id()),
        html: node.str("textContent").unwrap_or("").into(),
        width: Some(bw),
        height: Some(bh),
        paragraph_spacing: 0.0,
        vertical_align: Some(
            node.str("textVerticalAlign")
                .unwrap_or("middle")
                .to_string(),
        ),
        opacity: node.f64("opacity").unwrap_or(1.0),
        // ACCEPTED GAP: shape-text labels do NOT
        // warp — `distort` stays None here, so a distorted labeled shape
        // renders a warped body under a flat label at the unwarped frame
        // position. Wiring it needs the node's homography conjugated by the
        // text-frame origin (T(-b) ∘ H ∘ T(b)) plus warped frame bounds; that
        // ships with (or before) the PR 3 authoring surface, which is when a
        // user can first create the combination. Until then only hand-authored
        // docs can hit it. (MSDF/Degraded/shader-fill text share the unwarped
        // posture — see seams.rs `TextQuadBatch::distort`.)
        ..Default::default()
    };
    if skip_text_too_small(&spec, &model) {
        return Ok(TextBuildOutcome::Empty);
    }
    // R4 (task #95): same full-item residency as richtext, keyed by the
    // SHAPE node's text hash (textContent/textPadding/size/sides/... are all
    // non-transform props). Shape text never carries a shader fill.
    let cache_id = spec.node_id.clone();
    let mut cache_slot: Option<(&mut crate::residency::PlanCaches, u64)> = None;
    if let Some((cache, hashes)) = caches {
        let key = text_cache_key(
            scene,
            hashes,
            node,
            best_effort,
            render_scale,
            model_min_axis_scale(&model),
            0,
        );
        if let Some(mut batch) = cache
            .text_get_content(&cache_id, key)
            .filter(|b| degraded_hit_serves(b, &model))
        {
            patch_cached_batch(&mut batch, z, model, page_clip, None);
            return Ok(TextBuildOutcome::Item(Box::new(
                crate::seams::DrawItem::Text(batch.into()),
            )));
        }
        cache.text_build_begin();
        cache_slot = Some((cache, key));
    }
    // Shape text draws as richtext does (`WebGPUShapeSystem.renderShapeText` pushes a synthetic RichTextNode into
    // the text batch). Shape-text synthetic nodes carry no fillPriority: never a shader fill.
    let mut built = vector_or_degraded(&spec, model, z, page_clip, None, render_scale);
    if let Some((cache, key)) = cache_slot {
        name_built_text(&mut built, key);
        cache.text_build_end(&cache_id, key, built_text_batch(&built));
    }
    built
}

// ---------------------------------------------------------------------------
// M4b — connector labels (`WebGPUShapeSystem.renderLine` text half +
// `renderConnectorText` :1349-1473) + the connector HTML normalizer
// (`core/scene/connectorTextHelpers.ts:886-1027, 1306-1373`) + the intrinsic
// content measurement (`utils/canvas2dRichTextRenderer.ts
// measureRichTextContentSize` :419-444 over `utils/richtext/intrinsicSize.ts
// measureParagraphsIntrinsicSize` :5-82).
//
// MEASUREMENT NOTE (recorded deviation, not a quirk): production measures
// label content with Canvas2D (full Chromium shaping/kerning) even for
// MSDF-eligible families; natively the MSDF naive-advance measurer is
// injected into the SAME shared layout math. Gap extents can differ by the
// kerning delta (typically sub-pixel to ~1px per label) — the PNG SSIM gate
// arbitrates. The drawn glyphs themselves are MSDF in both worlds.
// ---------------------------------------------------------------------------

/// Result of planning one connector's label: the body text-gap to carve and
/// the label text batch (None = no label).
#[derive(Default)]
pub struct ConnectorTextPlan {
    pub gap: Option<crate::tess::line_to_path::TextGap>,
    pub outcome: Option<TextBuildOutcome>,
}

/// `hasConnectorText` (connectorTextHelpers.ts:71-76).
fn has_connector_text(node: &RawNode) -> bool {
    match node.str("textContent") {
        Some(html) => !strip_tags(html).trim().is_empty(),
        None => false,
    }
}

/// `measureParagraphsIntrinsicSize` (`utils/richtext/intrinsicSize.ts:5-82`) with the vector lane's measurer and
/// metrics (the fonts the label draws with).
fn measure_paragraphs_intrinsic_size(
    paragraphs: &[moda_text::Paragraph],
    max_width: f64,
    paragraph_spacing_px: f64,
    store: &moda_text::vector::VectorFontStore,
    catalog: &moda_text::Catalog,
) -> (f64, f64) {
    use moda_text::line_metrics::FontMetricsProvider;
    use moda_text::measure::TextMeasurer;
    use moda_text::positioning::{compute_list_indent_px, compute_marker_metrics};

    let measurer = moda_text::vector::measure::VectorMeasurer::new(store, catalog);
    let metrics = moda_text::vector::node_layout::VectorMetricsProvider { store, catalog };
    let default_style = moda_text::types::default_style();

    let mut total_height = 0.0f64;
    let mut max_line_width = 0.0f64;

    for (para_index, para) in paragraphs.iter().enumerate() {
        let list = para.list_metadata.as_ref();
        let is_ul = list.map(|l| l.kind == "ul").unwrap_or(false);
        // `para.runs[0]?.style.fontSize || defaultStyle.fontSize` (falsy).
        let list_font_size = para
            .runs
            .first()
            .map(|r| r.style.font_size)
            .filter(|f| *f != 0.0 && !f.is_nan())
            .unwrap_or(default_style.font_size);
        let effective_max_width = if is_ul {
            let level = list.map(|l| l.level).unwrap_or(0);
            let indent_px = compute_list_indent_px(level, list_font_size);
            let mm = compute_marker_metrics(list_font_size, None, None);
            (max_width - indent_px - mm.marker_box_width_px).max(0.0)
        } else {
            max_width
        };

        let lines = moda_text::wrap::layout_paragraph(para, effective_max_width, &measurer);
        if lines.is_empty() {
            let s = para.base_style.as_ref().unwrap_or(&default_style);
            total_height += moda_text::types::round_line_height_px(s.font_size * s.line_height);
        }
        let mut last_line_height_px = 0.0f64;
        let mut last_content_box = 0.0f64;
        for line in &lines {
            let mut w = 0.0f64;
            for seg in &line.segments {
                w += measurer.measure(&seg.text, &seg.style);
            }
            // The paragraph's indents and the every-line composer's word spacing (`moda_text::paragraph_format`).
            if let Some(px) = line.space_px {
                w += px * moda_text::positioning::justify_opportunities(line) as f64;
            }
            if line.indent_start != 0.0 || line.indent_end != 0.0 {
                w += line.indent_start + line.indent_end;
            }
            let effective_width = if is_ul {
                let level = list.map(|l| l.level).unwrap_or(0);
                let indent_px = compute_list_indent_px(level, list_font_size);
                let mm = compute_marker_metrics(list_font_size, None, None);
                w + indent_px + mm.marker_box_width_px
            } else {
                w
            };
            if effective_width > max_line_width {
                max_line_width = effective_width;
            }
            let max_font_size = line
                .segments
                .iter()
                .fold(default_style.font_size, |m, s| m.max(s.style.font_size));
            let line_height_px = if line.line_height_px > 0.0 {
                line.line_height_px
            } else {
                max_font_size * default_style.line_height
            };
            let mut max_ascent = 0.0f64;
            let mut max_descent = 0.0f64;
            for seg in &line.segments {
                let (a, d) = metrics.metrics(&seg.style);
                if a > max_ascent {
                    max_ascent = a;
                }
                if d > max_descent {
                    max_descent = d;
                }
            }
            let content_box = max_ascent + max_descent;
            total_height += line_height_px;
            last_line_height_px = line_height_px;
            last_content_box = content_box;
        }
        if !lines.is_empty() {
            total_height += (last_content_box - last_line_height_px).max(0.0);
        }
        if para_index < paragraphs.len() - 1 {
            total_height += paragraph_spacing_px.max(0.0);
        }
    }

    (max_line_width, total_height)
}

/// `measureRichTextContentSize` (canvas2dRichTextRenderer.ts:419-444) for
/// the connector temp node: width 10000 (prevent wrapping), no padding,
/// paragraphSpacing 0; returns CEILed dimensions.
fn measure_connector_text_size(
    store: &moda_text::vector::VectorFontStore,
    catalog: &moda_text::Catalog,
    html: &str,
) -> (f64, f64) {
    let paragraphs = moda_text::parser::parse_rich_text_to_paragraphs(html);
    let (w, h) = measure_paragraphs_intrinsic_size(&paragraphs, 10000.0, 0.0, store, catalog);
    (w.ceil(), h.ceil())
}

// --- connector HTML normalizer (connectorTextHelpers.ts:911-1057) ----------

/// `STYLE_ATTR_RE` = `\sstyle\s*=\s*(['"])(.*?)\1` (non-greedy) over an
/// opening tag. Returns (start, end, quote, content-range) of the matched
/// attribute within `tag`.
fn find_style_attr(tag: &str) -> Option<(usize, usize, char, std::ops::Range<usize>)> {
    let lower = tag.to_lowercase();
    let bytes = lower.as_bytes();
    let mut i = 0usize;
    while let Some(rel) = lower[i..].find("style") {
        let at = i + rel;
        // preceding char must be whitespace (the `\s` in the regex)
        let ws_before = at > 0 && bytes[at - 1].is_ascii_whitespace();
        if !ws_before {
            i = at + 5;
            continue;
        }
        let mut j = at + 5;
        while j < tag.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if j >= tag.len() || bytes[j] != b'=' {
            i = at + 5;
            continue;
        }
        j += 1;
        while j < tag.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if j >= tag.len() || (bytes[j] != b'"' && bytes[j] != b'\'') {
            i = at + 5;
            continue;
        }
        let quote = bytes[j] as char;
        let content_start = j + 1;
        if let Some(close_rel) = tag[content_start..].find(quote) {
            let content_end = content_start + close_rel;
            // match span starts at the whitespace before `style`
            return Some((at - 1, content_end + 1, quote, content_start..content_end));
        }
        i = at + 5;
    }
    None
}

/// `extractConnectorAlignFromStyleContent` (:930-937): first
/// `text-align: <value>`; justify → left.
fn extract_align_from_style_content(style: &str) -> Option<&'static str> {
    let lower = style.to_lowercase();
    let mut i = 0usize;
    while let Some(rel) = lower[i..].find("text-align") {
        let at = i + rel;
        let mut j = at + "text-align".len();
        let bytes = lower.as_bytes();
        while j < lower.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if j < lower.len() && bytes[j] == b':' {
            j += 1;
            while j < lower.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            for v in ["left", "center", "right", "justify"] {
                if lower[j..].starts_with(v) {
                    return Some(match v {
                        "justify" => "left",
                        "left" => "left",
                        "center" => "center",
                        _ => "right",
                    });
                }
            }
        }
        i = at + "text-align".len();
    }
    None
}

/// `injectConnectorTextAlignIntoOpeningTag` (:1029-1044).
fn inject_text_align_into_opening_tag(opening_tag: &str, tag: &str, align: &str) -> String {
    // `^<([a-z0-9]+)\b([^>]*)>$` — attrs are everything after the tag name.
    let inner = &opening_tag[1..opening_tag.len() - 1]; // strip < >
    let name_len = inner
        .char_indices()
        .take_while(|(_, c)| c.is_ascii_alphanumeric())
        .count();
    let attrs = &inner[name_len..];
    if let Some((_, _, quote, content)) = find_style_attr(opening_tag) {
        // splice into the existing style attribute
        let existing = opening_tag[content.clone()].trim().to_string();
        let needs_semicolon = !existing.is_empty() && !existing.ends_with(';');
        let next_style = format!(
            "{existing}{}{}text-align: {align};",
            if needs_semicolon { ";" } else { "" },
            if existing.is_empty() { "" } else { " " }
        );
        // rebuild the attrs with the style attr replaced
        let (mstart, mend, _, _) = find_style_attr(opening_tag).unwrap();
        let before = &opening_tag[..mstart];
        let after = &opening_tag[mend..];
        return format!("{before} style={quote}{next_style}{quote}{after}");
    }
    let trimmed = attrs.trim();
    let prefix = if trimmed.is_empty() { "" } else { " " };
    format!("<{tag}{prefix}{trimmed} style=\"text-align: {align};\">")
}

const NORMALIZER_BLOCK_TAGS: [&str; 5] = ["p", "li", "div", "ul", "ol"];

struct NormFrame {
    tag: String,
    own_explicit: Option<&'static str>,
    effective: &'static str,
    continuation: &'static str,
}

fn intern_align(a: &str) -> &'static str {
    match a {
        "center" => "center",
        "right" => "right",
        _ => "left",
    }
}

/// `normalizeConnectorParagraphTextAlign` (connectorTextHelpers.ts:970-1027):
/// every block opening tag (`p|li|div|ul|ol`) gets an explicit `text-align`,
/// with block-local continuation semantics and the ul/ol-wrapper li rule.
pub fn normalize_connector_paragraph_text_align(html: &str, align: &str) -> String {
    if html.is_empty() {
        return html.to_string();
    }
    let bootstrap = intern_align(align);
    let mut frames: Vec<NormFrame> = vec![NormFrame {
        tag: "#root".into(),
        own_explicit: None,
        effective: bootstrap,
        continuation: bootstrap,
    }];

    let mut out = String::with_capacity(html.len() + 64);
    let bytes = html.as_bytes();
    let mut i = 0usize;
    while i < html.len() {
        let Some(rel) = html[i..].find('<') else {
            out.push_str(&html[i..]);
            break;
        };
        let tag_start = i + rel;
        // try to match `</?(p|li|div|ul|ol)\b[^>]*>`
        let rest = &html[tag_start..];
        let mut k = 1usize;
        let is_closing = rest.as_bytes().get(1) == Some(&b'/');
        if is_closing {
            k = 2;
        }
        let name_start = k;
        while let Some(c) = rest.as_bytes().get(k) {
            if c.is_ascii_alphanumeric() {
                k += 1;
            } else {
                break;
            }
        }
        let name = rest[name_start..k].to_lowercase();
        let boundary_ok = !matches!(rest.as_bytes().get(k), Some(c) if c.is_ascii_alphanumeric());
        let close = rest.find('>');
        let matched = boundary_ok
            && NORMALIZER_BLOCK_TAGS.contains(&name.as_str())
            && k > name_start
            && close.is_some();
        if !matched {
            // not a block tag — emit through the '<' and continue scanning
            out.push_str(&html[i..tag_start + 1]);
            i = tag_start + 1;
            continue;
        }
        let close = close.unwrap();
        let full = &rest[..close + 1];
        out.push_str(&html[i..tag_start]);

        if is_closing {
            out.push_str(full);
            let closed = frames.pop();
            if let (Some(closed), Some(parent)) = (closed, frames.last_mut()) {
                // shouldPropagateClosedBlockAlignment (:1046-1057)
                let suppress = (parent.tag == "ul" || parent.tag == "ol")
                    && closed.tag == "li"
                    && parent.own_explicit.is_some();
                if !suppress {
                    parent.continuation = closed.effective;
                }
            }
        } else {
            let parent_continuation = frames.last().map(|f| f.continuation).unwrap_or(bootstrap);
            let own = find_style_attr(full)
                .and_then(|(_, _, _, content)| extract_align_from_style_content(&full[content]));
            let effective = own.unwrap_or(parent_continuation);
            if own.is_some() {
                out.push_str(full);
            } else {
                out.push_str(&inject_text_align_into_opening_tag(full, &name, effective));
            }
            frames.push(NormFrame {
                tag: name,
                own_explicit: own,
                effective,
                continuation: effective,
            });
        }
        i = tag_start + close + 1;
    }
    let _ = bytes;
    out
}

/// `getEffectiveConnectorTextAlignFromNormalizedHtml`
/// (connectorTextHelpers.ts:1352-1373): the FIRST parsed paragraph's align
/// (justify → left), else `line.textAlign ?? 'center'`.
fn effective_connector_text_align(normalized_html: &str, node: &RawNode) -> &'static str {
    if !normalized_html.is_empty() {
        let paragraphs = moda_text::parser::parse_rich_text_to_paragraphs(normalized_html);
        if let Some(first) = paragraphs.first() {
            match first.align {
                "left" => return "left",
                "center" => return "center",
                "right" => return "right",
                "justify" => return "left",
                _ => {}
            }
        }
    }
    intern_align(node.str("textAlign").unwrap_or("center"))
}

/// `renderLine`'s text half + `renderConnectorText` — plan one connector's
/// gap + label. `z` is the line's z + SHAPE_TEXT_Z_OFFSET; `page_clip` is
/// the PLAIN page clip (renderConnectorText computes its own
/// `getPageClipFromScene`).
#[allow(clippy::too_many_arguments)]
pub fn plan_connector_text(
    scene: &Scene,
    node: &RawNode,
    z: f64,
    page_clip: [f32; 4],
    render_scale: f64,
    caches: Option<TextCacheCtx<'_>>,
) -> Result<ConnectorTextPlan> {
    use crate::tess::connector_text as ct;
    use crate::tess::line_to_path::LineSpec;

    if node.node_type() != "line" {
        return Ok(ConnectorTextPlan::default());
    }
    let spec = LineSpec::from_json(|k| node.json_field(k));
    // `shouldRenderText` (renderLine :879-881; export pages are always
    // text-enabled).
    if !has_connector_text(node) || !ct::is_line_long_enough_for_text(&spec) {
        return Ok(ConnectorTextPlan::default());
    }
    let html_raw = node.str("textContent").unwrap_or("");

    // t + canvas position/angle (transform-dependent — computed fresh every
    // frame; no font measure needed).
    let t = ct::get_connector_text_position_t(&spec, node.f64("textPosition"));
    let abs = abs_line_geometry(scene, node, &spec);
    let text_pos = ct::get_connector_text_position(&spec, &abs, t);

    // R6 (task #95): connector-label residency — the cached entry carries
    // BOTH the batch and the measured gap (the measure rides font metrics,
    // which ride the key's text-host generation). Model/z/clip patched on
    // hit; geometry/content edits re-key via the node text hash.
    let cache_id = format!("{}-connector-text", node.id());
    // Hoisted above the cache lookup: the label model bakes trs.scale into
    // its columns, so the key must carry the same min-axis scale the built
    // batch's AA padding was computed at (see `model_min_axis_scale`).
    let trs = scene
        .accumulated_affine(node.id())
        .map(|(_, t)| t)
        .ok_or_else(|| anyhow!("no transform for line {}", node.id()))?;
    let (scale_x, scale_y) = (trs.scale_x, trs.scale_y);
    let mut cache_slot: Option<(&mut crate::residency::PlanCaches, u64)> = None;
    let mut cached: Option<(Box<crate::seams::TextQuadBatch>, Option<(f64, f64)>)> = None;
    if let Some((cache, hashes)) = caches {
        let base_key = text_cache_key(
            scene,
            hashes,
            node,
            false,
            render_scale,
            scale_x.abs().min(scale_y.abs()),
            0,
        );
        // The cached entry carries the measured GAP, and the gap is a
        // function of the label's tangent angle (`calculate_line_text_gap`
        // reads `text_pos.angle`) — which is transform-derived, so the node
        // text hash cannot re-key it. Fold the effective angle in so a
        // rotating line (or ancestor) re-measures instead of carving the
        // body with the prior frame's gap (review round 3, lever 10.5).
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = rustc_hash::FxHasher::default();
            base_key.hash(&mut h);
            text_pos.angle.to_bits().hash(&mut h);
            h.finish()
        };
        cached = cache.text_get_with_gap(&cache_id, key);
        if cached.is_none() {
            cache_slot = Some((cache, key));
        }
    }

    let (store, host) = vector_store_and_host()?;
    let gap = match &cached {
        Some((_, gap)) => {
            gap.map(|(start_t, end_t)| crate::tess::line_to_path::TextGap { start_t, end_t })
        }
        None => {
            // Measure RAW textContent (renderLine :893-915), min 20×16.
            let (mw, mh) = measure_connector_text_size(&store, &host.catalog, html_raw);
            let text_width = mw.max(20.0);
            let text_height = mh.max(16.0);
            ct::calculate_line_text_gap(
                &spec,
                text_width,
                text_height,
                Some(text_pos.angle),
                Some(t),
                true,
                node.f64("textPosition"),
                node.f64("textPadding"),
                node.str("textRotation"),
            )
        }
    };

    // --- renderConnectorText (:1349-1473) --- (trs/scale hoisted above the
    // cache lookup so the key and the model share one derivation)
    let offset_x = node.f64("textOffsetX").unwrap_or(0.0);
    let offset_y = node.f64("textOffsetY").unwrap_or(0.0);
    let final_pos = ct::apply_text_offsets(&text_pos, offset_x, offset_y);

    let rotation = ct::calculate_text_rotation(text_pos.angle, node.str("textRotation"));

    let normalized = normalize_connector_paragraph_text_align(
        html_raw,
        node.str("textAlign").unwrap_or("center"),
    );
    let text_align = effective_connector_text_align(&normalized, node);
    let anchor_local_x = ct::connector_text_anchor_local_x(text_align, ct::CONNECTOR_TEXT_WIDTH);
    let half_height = ct::CONNECTOR_TEXT_HEIGHT / 2.0;

    // rotatedAnchorOffset = affineFromTRS(0,0,rotDeg,sx,sy)·(anchorX, halfH).
    let (sin, cos) = rotation.sin_cos();
    let rot_off_x = cos * scale_x * anchor_local_x + (-sin * scale_y) * half_height;
    let rot_off_y = sin * scale_x * anchor_local_x + cos * scale_y * half_height;
    let world_x = final_pos.0 - rot_off_x;
    let world_y = final_pos.1 - rot_off_y;

    // model = affineToMatrix4x4(affineFromTRS(world, rotDeg, sx, sy)).
    let model: [f32; 16] = [
        (cos * scale_x) as f32,
        (sin * scale_x) as f32,
        0.0,
        0.0, //
        (-sin * scale_y) as f32,
        (cos * scale_y) as f32,
        0.0,
        0.0, //
        0.0,
        0.0,
        1.0,
        0.0, //
        world_x as f32,
        world_y as f32,
        0.0,
        1.0,
    ];

    let spec_text = TextNodeSpec {
        node_id: format!("{}-connector-text", node.id()),
        html: normalized.into(),
        width: Some(ct::CONNECTOR_TEXT_WIDTH),
        height: Some(ct::CONNECTOR_TEXT_HEIGHT),
        paragraph_spacing: 0.0,
        vertical_align: Some("middle".to_string()),
        opacity: node.f64("opacity").unwrap_or(1.0),
        ..Default::default()
    };
    // R6 hit: patch the volatile fields and return the cached gap + batch.
    if let Some((mut batch, _)) = cached.filter(|(b, _)| degraded_hit_serves(b, &model)) {
        patch_cached_batch(&mut batch, z, model, page_clip, None);
        return Ok(ConnectorTextPlan {
            gap,
            outcome: Some(TextBuildOutcome::Item(Box::new(
                crate::seams::DrawItem::Text(batch.into()),
            ))),
        });
    }
    // NOTE `textBackground: 'solid'` is renderer-inert (Q-77): no WebGPU
    // consumer reads it; the synthetic node carries no background. Copy the
    // bug — the rl-curved golden shows a gap with INVISIBLE white text.
    // Connector labels are synthetic RichTextNodes through the same
    // `renderText` routing — vector lane first under the flag.
    let outcome = vector_or_degraded(&spec_text, model, z, page_clip, None, render_scale);
    if let (Some((cache, key)), Ok(TextBuildOutcome::Item(item))) = (cache_slot, &outcome) {
        if let crate::seams::DrawItem::Text(batch) = item.as_ref() {
            cache.text_put_with_gap(
                &cache_id,
                key,
                batch,
                gap.as_ref().map(|g| (g.start_t, g.end_t)),
            );
        }
    }
    Ok(ConnectorTextPlan {
        gap,
        outcome: Some(outcome?),
    })
}

/// `getAbsoluteLineCoordinates` (connectorTextHelpers.ts:190-248): node
/// canvas affine applied to endpoints + waypoints; affine kept for the
/// rounded-elbow local-space sampler.
fn abs_line_geometry(
    scene: &Scene,
    node: &RawNode,
    spec: &crate::tess::line_to_path::LineSpec,
) -> crate::tess::connector_text::AbsLineGeometry {
    use crate::tess::connector_text::AbsLineGeometry;
    let (x1, y1, x2, y2) = (
        spec.points.first().copied().unwrap_or(0.0),
        spec.points.get(1).copied().unwrap_or(0.0),
        spec.points.get(2).copied().unwrap_or(0.0),
        spec.points.get(3).copied().unwrap_or(0.0),
    );
    match scene.node_canvas_affine(node.id()) {
        Some(a) => {
            let m = [a.a, a.b, a.c, a.d, a.e, a.f];
            let ap = |x: f64, y: f64| -> [f64; 2] {
                [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]
            };
            let p1 = ap(x1, y1);
            let p2 = ap(x2, y2);
            AbsLineGeometry {
                x1: p1[0],
                y1: p1[1],
                x2: p2[0],
                y2: p2[1],
                waypoints: spec.waypoints.iter().map(|w| ap(w[0], w[1])).collect(),
                canvas_affine: Some(m),
            }
        }
        None => AbsLineGeometry {
            x1,
            y1,
            x2,
            y2,
            waypoints: spec.waypoints.clone(),
            canvas_affine: None,
        },
    }
}

// ---------------------------------------------------------------------------
// Engine text layout query (editor measurement, edit-overlay parity)
// ---------------------------------------------------------------------------

/// The vector lane's layout of one richtext node as JSON — the same walk the
/// draw path places glyphs with (`dump_rich_text_node_vector`, browser-vector
/// placement): paragraphs → lines → segments with their text, start x, width
/// and placed glyph x positions, plus each line's baseline. A node the vector
/// lane would not draw comes back as `{ engine, reason }` with no paragraphs.
///
/// The editor measures with this so a fitted box or an edit overlay agrees
/// with the pixels by construction (docs/qa/text-editing.md).
/// E8.22: the glyph-alternates query (`moda_text::vector::alternates`) over the installed fonts: the alternates the
/// resolved variant `family` / `weight` / `style` offers for `text`, starting from the run's `font-feature-settings`
/// (`features_css`, empty for none). `{ "default": …, "alternates": [ { feature, value, fontFeatures, glyphIds,
/// glyphNames, advance, path } ] }`; an error when the variant has no installed font.
pub fn glyph_alternates_json(
    family: &str,
    weight: i64,
    style: &str,
    text: &str,
    features_css: &str,
) -> Result<serde_json::Value> {
    let (store, host) = vector_store_and_host()?;
    let features = moda_text::font_features::parse_font_feature_settings(features_css);
    match moda_text::vector::alternates::glyph_alternates(&store, &host.catalog, family, weight, style, text, &features)
    {
        Some(a) => Ok(a.to_json()),
        None => Err(anyhow!("no installed font for {family} {weight} {style}")),
    }
}

/// A richtext node's laid-out lines, each line's text (its segments joined), in paragraph order: the lines the
/// canvas draws, soft wraps included (motion's `text.units(by: line)`). `None` for a node that isn't text.
pub fn node_layout_lines(scene: &moda_scene::Scene, node_id: &str) -> Option<Vec<String>> {
    let node = scene.effective_node(node_id)?;
    if node.str("type") != Some("richtext") {
        return None;
    }
    let spec = moda_text::RichTextNodeSpec {
        node_id: node.id().to_string(),
        html_content: node_html_in(scene, node),
        width: node.f64("width"),
        height: node.f64("height"),
        paragraph_spacing: node.f64("paragraphSpacing").unwrap_or(0.0),
        line_height_policy: Default::default(),
        vertical_align: node.str("verticalAlign").map(str::to_string),
        truncation: None,
        auto_width: node_auto_width(node),
        area: crate::text_area::node_area_spec(scene, node),
    };
    // moda-text reports its layout as serde_json.
    use serde_json::Value;
    let layout = layout_rich_text_json(&spec).ok()?;
    let mut out = Vec::new();
    for para in layout.get("paragraphs")?.as_array()? {
        for line in para.get("lines").and_then(Value::as_array).into_iter().flatten() {
            let text: String = line
                .get("segments")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|seg| seg.get("text").and_then(Value::as_str))
                .collect();
            out.push(text);
        }
    }
    Some(out)
}

pub fn layout_rich_text_json(spec: &moda_text::RichTextNodeSpec) -> Result<serde_json::Value> {
    // moda-text reports its layout as serde_json.
    use serde_json::Value;
    let (store, host) = vector_store_and_host()?;
    let vector = moda_text::vector::layout_rich_text_node_vector(spec, &store, &host.catalog);
    if vector.get("engine").and_then(Value::as_str) == Some("vector") {
        return Ok(vector);
    }
    // The vector lane declined (emoji, a complex script, a glyph no face covers): the canvas draws the node
    // through the degraded tier (`build_degraded_batch`), so report ITS lines, placed in the node the way the
    // raster is placed.
    let text_spec = TextNodeSpec {
        node_id: spec.node_id.clone(),
        html: spec.html_content.clone().into(),
        width: spec.width,
        height: spec.height,
        paragraph_spacing: spec.paragraph_spacing,
        line_height_policy: spec.line_height_policy,
        vertical_align: spec.vertical_align.clone(),
        auto_width: spec.auto_width,
        ..Default::default()
    };
    #[cfg(test)]
    crate::test_fonts();
    let Ok(engine) = moda_text::degraded::engine() else {
        return Ok(vector);
    };
    let Ok(mut layout) = engine.layout_json(&text_spec) else {
        return Ok(vector);
    };
    let logical_w = layout["logicalW"].as_f64().unwrap_or(0.0);
    let logical_h = layout["logicalH"].as_f64().unwrap_or(0.0);
    let (dx, dy) = moda_text::degraded::raster_offset(&text_spec, logical_w, logical_h);
    let shift = |v: &mut Value, d: f64| {
        if let Some(n) = v.as_f64() {
            *v = serde_json::json!(n + d);
        }
    };
    if let Some(paragraphs) = layout["paragraphs"].as_array_mut() {
        for p in paragraphs {
            for l in p["lines"].as_array_mut().into_iter().flatten() {
                shift(&mut l["lineStartX"], dx);
                shift(&mut l["lineTopY"], dy);
                shift(&mut l["baselineY"], dy);
                for s in l["segments"].as_array_mut().into_iter().flatten() {
                    shift(&mut s["x"], dx);
                }
            }
        }
    }
    layout["vectorDecline"] = vector.get("reason").cloned().unwrap_or(Value::Null);
    Ok(layout)
}

// ---------------------------------------------------------------------------
// Gate-1 verification dump (`--dump-text-quads`)
// ---------------------------------------------------------------------------

/// Serialize text batches as JSON (no GPU). Used by the CLI's
/// `--dump-text-quads PATH` mode (PLAN §2 / SEAMS §6).
pub fn dump_batches_json(batches: &[&TextQuadBatch]) -> Value {
    use moda_scene::value::json;
    let ops_json = |b: &TextQuadBatch| -> Vec<Value> {
        b.ops
            .iter()
            .map(|o| {
                let (pipeline, emoji) = match o.pipeline {
                    TextOpPipeline::Solid => ("text-decoration", None),
                    TextOpPipeline::Emoji { emoji } => ("text-emoji", Some(emoji)),
                };
                json!({
                    "pipeline": pipeline,
                    "emoji": emoji,
                    "color": o.color,
                    "opacity": o.opacity,
                    "blurPx": o.blur_px,
                    "strokePx": o.stroke_px,
                    "vertexCount": o.verts.len() / 4,
                    "indexCount": o.indices.len(),
                    "verts": o.verts,
                })
            })
            .collect()
    };
    let vector_json = |v: &moda_text::vector::VectorNodeOps| -> Value {
        json!({
            "curveCount": v.curve_points.len() / 6,
            "ops": v.ops.iter().map(|o| json!({
                "pipeline": match o.pipeline {
                    moda_text::vector::VectorOpPipeline::Tess => "vector-text-tess",
                    moda_text::vector::VectorOpPipeline::Analytic if o.additive => "vector-text-analytic-additive",
                    moda_text::vector::VectorOpPipeline::Analytic => "vector-text-analytic",
                    moda_text::vector::VectorOpPipeline::AnalyticGradient => "vector-text-analytic-gradient",
                    moda_text::vector::VectorOpPipeline::AnalyticShaderFill => "vector-text-fill",
                },
                "color": o.color,
                "opacity": o.opacity,
                "blurPx": o.blur_px,
                "strokePx": o.stroke_px,
                "triangleVerts": o.tris.len() / 2,
                "gradient": o.gradient.as_ref().map(|g| json!({
                    "textBounds": g.text_bounds,
                    "params": g.params,
                    "config": g.config,
                    "stops": g.stops,
                })),
                "instances": o.records.iter().map(|r| json!({
                    "pen": [r.pen_x, r.baseline_y],
                    "fontSizePx": r.font_size_px,
                    "emBox": r.em_box,
                    "firstCurve": r.first_curve,
                    "curveCount": r.curve_count,
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        })
    };
    json!({
        "batches": batches.iter().map(|b| json!({
            "nodeId": b.node_id,
            "z": b.z,
            "modelMatrix": b.model_matrix,
            "pageClipBounds": b.page_clip_bounds,
            "emojis": b.emojis,
            "vector": b.vector.as_ref().map(vector_json),
            "degraded": b.degraded.as_ref().map(|d| json!({
                "logicalSize": [d.raster.logical_w, d.raster.logical_h],
                "qualityScale": d.raster.quality_scale,
                "offset": d.offset,
                "opacity": d.opacity,
                "fillAlpha": d.raster.fill_alpha,
            })),
            "ops": ops_json(b),
        })).collect::<Vec<_>>(),
    })
}

// ---------------------------------------------------------------------------
// GPU side
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GPU side: moda-text-draw (vork) holds it; see PROVENANCE.md.
// ---------------------------------------------------------------------------

pub use moda_text_draw::text_draw::{PreparedText, TextDraw};

// ---------------------------------------------------------------------------
// Motion greeking ("Simplify small details while moving"): while the canvas moves, a vector text layer whose
// type is under a few device pixels per em draws each word as one translucent bar instead of its glyphs. Only
// live motion frames do this (`Engine::set_motion_text_greek_px`); the still frame after motion, exports,
// thumbnails and snapshots always draw the glyphs.
// ---------------------------------------------------------------------------

/// The bar's band above the baseline, in em (about Inter's x-height, where most of a word's ink is).
const GREEK_BAND_EM: f32 = 0.55;
/// The bar's opacity: about the share of that band a run of lowercase text inks.
const GREEK_INK: f32 = 0.45;
/// A gap wider than this (em) between two glyphs' boxes starts a new word.
const GREEK_WORD_GAP_EM: f32 = 0.18;

/// The largest font size (px, layer-local) of a vector text layer that can be greeked, or `None` when it can't:
/// not vector text, or drawn with anything but plain solid glyph fills and decorations (gradients, shader fills,
/// strokes, shadows, glows, emoji, per-glyph transforms, a distort warp or a mask).
pub fn greek_font_px(b: &crate::seams::TextQuadBatch) -> Option<f32> {
    let v = b.vector.as_ref()?;
    if b.distort.is_some() || !v.emojis.is_empty() || v.gradient.is_some() {
        return None;
    }
    let mut max = 0.0f32;
    for op in &v.ops {
        match op.pipeline {
            moda_text::vector::VectorOpPipeline::Tess => {}
            moda_text::vector::VectorOpPipeline::Analytic
                if !op.additive && op.blur_px == 0.0 && op.stroke_px == 0.0 && op.gradient.is_none() =>
            {
                for r in &op.records {
                    if r.glyph_lin != [1.0, 0.0, 0.0, 1.0] || r.glyph_off != [0.0, 0.0] {
                        return None;
                    }
                    max = max.max(r.font_size_px);
                }
            }
            _ => return None,
        }
    }
    (max > 0.0).then_some(max)
}

/// The batch's device px per em at `device_px_per_canvas_px`, when it can be greeked.
pub fn greek_em_px(b: &crate::seams::TextQuadBatch, device_px_per_canvas_px: f64) -> Option<f64> {
    let font = greek_font_px(b)? as f64;
    Some(font * model_min_axis_scale(&b.model_matrix) * device_px_per_canvas_px)
}

/// The greeked copy of a greekable vector batch ([`greek_font_px`]): each glyph fill op becomes one solid
/// triangle op of per-word bars (the same colour, `GREEK_INK` of its opacity); decorations stay.
pub fn greek_batch(b: &crate::seams::TextQuadBatch) -> Option<crate::seams::TextQuadBatch> {
    greek_font_px(b)?;
    let v = b.vector.as_ref()?;
    let mut ops = Vec::with_capacity(v.ops.len());
    for op in &v.ops {
        if op.pipeline == moda_text::vector::VectorOpPipeline::Tess {
            ops.push(op.clone());
            continue;
        }
        let mut tris: Vec<f32> = Vec::new();
        let mut push_bar = |x0: f32, x1: f32, base: f32, size: f32| {
            let (y0, y1) = (base - GREEK_BAND_EM * size, base);
            tris.extend_from_slice(&[x0, y0, x1, y0, x1, y1, x0, y0, x1, y1, x0, y1]);
        };
        // Current word: (left, right, baseline, size).
        let mut word: Option<(f32, f32, f32, f32)> = None;
        for r in &op.records {
            let s = r.font_size_px;
            let (l, rt) = (r.pen_x + r.em_box[0] * s, r.pen_x + r.em_box[2] * s);
            word = match word {
                Some((wl, wr, base, size))
                    if base == r.baseline_y && size == s && l - wr <= GREEK_WORD_GAP_EM * s =>
                {
                    Some((wl, wr.max(rt), base, size))
                }
                Some((wl, wr, base, size)) => {
                    push_bar(wl, wr, base, size);
                    Some((l, rt, r.baseline_y, s))
                }
                None => Some((l, rt, r.baseline_y, s)),
            };
        }
        if let Some((wl, wr, base, size)) = word {
            push_bar(wl, wr, base, size);
        }
        if tris.is_empty() {
            continue;
        }
        ops.push(moda_text::vector::VectorOp {
            pipeline: moda_text::vector::VectorOpPipeline::Tess,
            color: op.color,
            opacity: op.opacity * GREEK_INK,
            blur_px: 0.0,
            stroke_px: 0.0,
            additive: false,
            tris,
            records: Vec::new(),
            gradient: None,
            paint: None,
        });
    }
    let mut out = b.clone();
    out.vector = Some(moda_text::vector::VectorNodeOps {
        ops,
        curve_points: Default::default(),
        gradient: None,
        emojis: Vec::new(),
    });
    Some(out)
}
