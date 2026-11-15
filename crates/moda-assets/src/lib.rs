//! moda-assets — the moda-render asset layer (PLAN.md M3c).
//!
//! Pipeline: LOCAL source resolution (data URI / vendored file; an http(s)
//! source is a per-asset failure — documents render offline) → bounded
//! parallel prefetch → decode (`image`) → ICC→sRGB (`moxcms`) → premultiply → production
//! sizing policy resize → `rgba8unorm` upload, or resvg rasterization for
//! SVG. Placeholder semantics are state-driven, exactly like
//! `WebGPUTextureManager.getImageFillState`.
//!
//! Public surface the engine/paint tracks consume (SEAMS-compatible):
//! - [`AssetStore`] — post-prefetch lookup: `fill_state` / `dimensions` /
//!   `scale_factor` / `image` (heap-decoded), plus `external` for a
//!   browser-decoded, GPU-resident raster (wasm editor lane —
//!   `insert_external` / `clear_external`, no pixels on the heap).
//! - [`pattern::build_pattern_paint`] → [`pattern::PatternPaint`] — the
//!   per-node pattern instance (416-byte `PathPatternData` ABI) + texture
//!   key + sampler choice.
//! - [`collect_pattern_requests`] + [`fetch::prefetch_blocking`] — scene scan
//!   and prefetch entry points (used by moda-cli).
//! - [`probe::probe_svg`] — usvg per-asset manifest probe.

pub mod bitmap_budget;
pub mod constants;
pub mod decode;
pub mod engine_seam;
#[cfg(feature = "native-io")]
pub mod fetch;
pub mod pattern;
pub mod probe;
pub mod report;
pub mod shader_image;
pub mod sizing;
pub mod source;
pub mod svg;

use std::collections::HashMap;
use std::sync::Arc;

pub use decode::PreparedImage;
#[cfg(feature = "native-io")]
pub use fetch::{
    fetch_media_bytes_blocking, prefetch_blocking, AssetFailure, BlankAsset, PrefetchConfig,
    PrefetchReport, SvgProbeWarning, REMOTE_URL_NOT_ALLOWED,
};
pub use pattern::{ImageFillState, PatternInstance, PatternPaint};

/// The fontdb byte-decode callers (the wasm editor lane) lay out SVG `<text>` with: the text
/// engine's fonts ([`svg::engine_fontdb`]).
fn default_fontdb() -> Arc<resvg::usvg::fontdb::Database> {
    svg::engine_fontdb()
}

/// Decode pattern-fill bytes that an external host fetched (the wasm
/// editor's browser-fetch feeder) into a [`PreparedImage`] — the same
/// SVG-sniff + decode/ICC/premultiply/sizing pipeline `fetch::load_one`
/// runs after its own fetch. No IO, so not `native-io`-gated.
pub fn decode_pattern_bytes(
    src: &str,
    bytes: &[u8],
    min_raster_size: Option<(u32, u32)>,
) -> anyhow::Result<PreparedImage> {
    anyhow::ensure!(!bytes.is_empty(), "empty asset body for {src}");
    let is_svg = source::is_likely_svg(src) || source::sniff_svg_bytes(bytes);
    let prepared = if is_svg {
        svg::prepare_svg(bytes, min_raster_size, &default_fontdb())
    } else {
        decode::prepare_raster(bytes)
    };
    // A decode that succeeded and paints NOTHING. Recorded on the
    // runtime-event channel (deduped per src, inert unless a report lane armed
    // the collector) so the wasm editor lane — whose fontless SVG rasterization
    // is the main way this happens — is no longer silent about it.
    if let Ok(img) = &prepared {
        if img.coverage_permille == 0 {
            report::record(report::RuntimeEvent::BlankRasterDecoded {
                src: report::sanitized_asset_ref(src),
                coverage_permille: img.coverage_permille,
                cause: img.blank_cause.to_string(),
                shader_image: false,
            });
        }
    }
    prepared
}

/// M4e shader-image analog of [`decode_pattern_bytes`] (`fetch::load_job`'s
/// shader arm): the bytes run through the `imageProcessing.ts` port. Empty
/// bytes are legal — the production `!source.src` arm never fetches.
pub fn decode_shader_image_bytes(
    src: &str,
    bytes: &[u8],
    processing: &moda_shaders::runtime::image_processing::Processing,
) -> anyhow::Result<moda_shaders::runtime::image_processing::PreparedShaderImage> {
    shader_image::load_shader_image(
        src,
        bytes,
        source::is_likely_svg(src),
        processing,
    )
}

/// One asset request from the scene scan ([`collect_pattern_requests`]).
/// Lives here (not in the `native-io`-gated `fetch` module) because the scan
/// itself does no IO: the wasm editor collects requests engine-side and lets
/// the BROWSER fetch the bytes (`renderers/wasm/wasmAssetFeeder.ts`), while
/// the CLI feeds the same requests to [`fetch::prefetch_blocking`].
#[derive(Debug, Clone)]
pub struct PrefetchRequest {
    pub src: String,
    /// Desired min raster size (SVG only; `computeMinRasterSizeForPatternFill`).
    pub min_raster_size: Option<(u32, u32)>,
    /// M4e: when set, this is a SHADER-IMAGE request (image-backed procedural
    /// fill): the bytes run through `imageProcessing.ts`'s port instead of
    /// the pattern pipeline, and the result lands in the store's
    /// shader-image map keyed by `moda_shaders::runtime::image_resource_key`.
    pub shader_processing: Option<moda_shaders::runtime::image_processing::Processing>,
    /// Requesting node id (WF5 r3 resource report attribution).
    pub node_id: Option<String>,
}

/// The store key of raster `src` prepared without the sizing policy and without 8-bit association: its decoded
/// texels, straight alpha, at their natural size ([`decode::prepare_raster_natural`]). A request for this key
/// fetches `src` and prepares it so; the plain `src` keys the same source as a browser prepares a bitmap. The
/// two are different pixels of one file, so they are two entries.
pub fn natural_key(src: &str) -> String {
    format!("{NATURAL_KEY}{src}")
}

/// The source a [`natural_key`] names; `None` for any other key.
pub fn natural_source(key: &str) -> Option<&str> {
    key.strip_prefix(NATURAL_KEY)
}

const NATURAL_KEY: &str = "moda-natural:";

/// Terminal state of one prefetched asset.
#[derive(Debug, Clone)]
pub enum AssetEntry {
    /// Decoded, premultiplied bytes resident on the heap (the native CLI
    /// lane, and the wasm SVG/shader-image paths).
    Loaded(Arc<PreparedImage>),
    /// A browser-decoded raster uploaded straight to a GPU texture (the wasm
    /// editor lane): the store holds only this metadata,
    /// so NO decoded RGBA `Vec` ever lands on the wasm heap. The pixels are
    /// pulled from the installed `ExternalImageUploader` at bind time, keyed
    /// by the store key.
    LoadedExternal(ExternalImageMeta),
    /// Committed failure (remote URL, decode error, missing file) →
    /// rose placeholder.
    Failed,
}

/// Metadata for a [`AssetEntry::LoadedExternal`] raster — exactly what the
/// engine seam consumes to size the texture and pack the pattern instance,
/// carrying NO pixel bytes. The decoded pixels live in the browser (an
/// `ImageBitmap` held by the wasm feeder) and go ImageBitmap → GPU texture via
/// `copy_external_image_to_texture`, never through the wasm linear heap.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalImageMeta {
    /// Uploaded texel dimensions (post sizing policy = the ImageBitmap size).
    pub width: u32,
    pub height: u32,
    /// uploaded / natural per axis (`CachedTexture.scaleFactor`) — the pattern
    /// instance divides `fillPatternScale` by this.
    pub scale_factor: (f64, f64),
    /// Monotonic per-install id — the retained-cache identity for the external
    /// texture. The heap lane keys the texture cache on the decoded-pixel
    /// `Arc` pointer; an external upload has no such `Arc`, so a re-install
    /// bumps this id to self-invalidate the cached texture (MAP §5.7 parity).
    pub upload_id: u64,
}

/// Immutable post-prefetch asset lookup. The native analog of
/// `WebGPUTextureManager`'s cache+state surface, with the async machinery
/// stripped: by render time every source has settled. M4e adds the
/// shader-image side (`ShaderImageResourceManager` analog), keyed by the
/// production resource key (`createImageResourceKey`).
#[derive(Debug, Clone, Default)]
pub struct AssetStore {
    entries: HashMap<String, AssetEntry>,
    shader_images: HashMap<String, shader_image::ShaderImageEntry>,
}

impl AssetStore {
    pub fn extend(&mut self, other: AssetStore) {
        self.entries.extend(other.entries);
        self.shader_images.extend(other.shader_images);
    }

    pub fn insert(&mut self, src: String, entry: AssetEntry) {
        self.entries.insert(src, entry);
    }

    /// Insert a browser-decoded, GPU-resident raster (wasm editor lane): the
    /// store carries only `meta`, never the decoded bytes.
    pub fn insert_external(&mut self, src: String, meta: ExternalImageMeta) {
        self.entries.insert(src, AssetEntry::LoadedExternal(meta));
    }

    /// M4e: settled shader-image entry under its resource key.
    pub fn insert_shader_image(&mut self, key: String, entry: shader_image::ShaderImageEntry) {
        self.shader_images.insert(key, entry);
    }

    /// M4e: lookup by production resource key. `None` = never requested
    /// (callers treat as the manager's fallback entry, like Fallback).
    pub fn shader_image(&self, key: &str) -> Option<&shader_image::ShaderImageEntry> {
        self.shader_images.get(key)
    }

    pub fn len(&self) -> usize {
        self.entries.len() + self.shader_images.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.shader_images.is_empty()
    }

    /// `getImageFillState` (`WebGPUTextureManager.ts:445-459`): never
    /// requested → Loading; committed failure → Failed. A
    /// GPU-resident external raster is `Loaded` (real pixels, just not on the
    /// heap).
    pub fn fill_state(&self, src: &str) -> ImageFillState {
        match self.entries.get(src) {
            Some(AssetEntry::Loaded(_)) | Some(AssetEntry::LoadedExternal(_)) => {
                ImageFillState::Loaded
            }
            Some(AssetEntry::Failed) => ImageFillState::Failed,
            None => ImageFillState::Loading,
        }
    }

    /// `getTextureDimensions` — uploaded texel dims (heap OR external).
    pub fn dimensions(&self, src: &str) -> Option<(u32, u32)> {
        match self.entries.get(src) {
            Some(AssetEntry::Loaded(i)) => Some((i.width, i.height)),
            Some(AssetEntry::LoadedExternal(m)) => Some((m.width, m.height)),
            _ => None,
        }
    }

    /// `getTextureScaleFactor` — uploaded / natural, per axis (heap OR external).
    pub fn scale_factor(&self, src: &str) -> Option<(f64, f64)> {
        match self.entries.get(src) {
            Some(AssetEntry::Loaded(i)) => Some(i.scale_factor),
            Some(AssetEntry::LoadedExternal(m)) => Some(m.scale_factor),
            _ => None,
        }
    }

    /// The decoded heap image, if this source settled as heap-resident. An
    /// external (GPU-resident) raster returns `None` here — use
    /// [`AssetStore::external`] for its metadata.
    pub fn image(&self, src: &str) -> Option<&PreparedImage> {
        match self.entries.get(src) {
            Some(AssetEntry::Loaded(img)) => Some(img),
            _ => None,
        }
    }

    /// The metadata for a browser-decoded, GPU-resident raster (wasm editor
    /// lane). `None` for heap/failed/never-requested sources.
    pub fn external(&self, src: &str) -> Option<&ExternalImageMeta> {
        match self.entries.get(src) {
            Some(AssetEntry::LoadedExternal(m)) => Some(m),
            _ => None,
        }
    }

    /// Drop every browser-decoded external raster entry, returning how many were
    /// removed. Pairs with the wasm `clear_external_images`: the session's
    /// `ImageBitmap` registry and this metadata must be torn down together. A
    /// retained `LoadedExternal` keeps `fill_state` reporting `Loaded`, so
    /// `collect_pattern_requests` would treat the key as settled and never
    /// re-feed it — and on a reused wasm instance the engine would then resolve
    /// an external texture whose bitmap is gone, rendering transparent. Heap,
    /// shader, and failed entries are untouched (their pixels live in the store,
    /// independent of any bitmap).
    pub fn clear_external(&mut self) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|_, e| !matches!(e, AssetEntry::LoadedExternal(_)));
        before - self.entries.len()
    }

    /// Drop a SINGLE entry by key, returning whether one was present. The
    /// per-key analog of [`AssetStore::clear_external`]: the live-video lane
    /// releases one node's streaming texture (`video:<nodeId>`) on stop / node
    /// delete / page leave without disturbing other videos or static assets.
    /// Reverts the key to `Loading` at the seam (the next `collect`/feed would
    /// re-settle it) — the wasm `release_external_image` pairs this with tearing
    /// down the matching `<video>`/registry entry so the engine can't resolve a
    /// texture whose source element is gone.
    pub fn remove(&mut self, src: &str) -> bool {
        self.entries.remove(src).is_some()
    }

    /// Drop every entry whose key starts with `prefix`, returning how many were
    /// removed. Used by the wasm `clear_external_images` to tear down the
    /// live-video streaming keys (`video:<nodeId>`) on canvas teardown:
    /// HEAP-fed frames settle as `Loaded` (not `LoadedExternal`), so
    /// [`AssetStore::clear_external`] alone leaves them resident in the
    /// process-global store across a canvas switch (the wasm instance is reused).
    /// Generic over the prefix to keep the store unaware of the `video:`
    /// convention.
    pub fn clear_prefixed(&mut self, prefix: &str) -> usize {
        // An empty prefix matches every key (`str::starts_with("")` is always
        // true) and would silently wipe the whole store. No caller passes one
        // today; the guard documents the invariant and trips a future misuse
        // (e.g. a computed prefix from the planned video-export path) in
        // debug/test builds without changing release behavior.
        debug_assert!(
            !prefix.is_empty(),
            "clear_prefixed called with an empty prefix (would wipe the store)"
        );
        if prefix.is_empty() {
            return 0;
        }
        let before = self.entries.len();
        self.entries.retain(|k, _| !k.starts_with(prefix));
        before - self.entries.len()
    }
}

/// Whether the server render lanes scope asset prefetch to the page(s) being
/// rendered (default ON). Read from `MODA_ASSET_SCOPE_TO_RENDERED_PAGES`: unset
/// or a truthy value (`1`/`true`/`yes`/`on`) ⇒ scoped; `0`/`false`/`no`/`off` ⇒
/// the old all-pages prefetch. A runtime kill switch — a single flip on the
/// render service reverts to the pre-change behavior with no rebuild. Read only
/// at the lane call sites (once per render), never in the pure scan.
pub fn scope_assets_to_rendered_pages_enabled() -> bool {
    scope_flag_from_env(
        std::env::var("MODA_ASSET_SCOPE_TO_RENDERED_PAGES")
            .ok()
            .as_deref(),
    )
}

/// Pure parse of the scope flag (split out so it is testable without mutating
/// the process env — which would race the other tests reading `MODA_ASSET_*`).
/// `None` (unset) ⇒ ON; only an explicit falsey value opts out.
fn scope_flag_from_env(value: Option<&str>) -> bool {
    match value {
        Some(v) => {
            let t = v.trim();
            !["0", "false", "no", "off"]
                .iter()
                .any(|off| t.eq_ignore_ascii_case(off))
        }
        None => true,
    }
}

/// Scan a hydrated scene for asset sources to prefetch: pattern fills (with
/// the per-node desired min raster size for SVG —
/// `WebGPUShapeSystem.computeMinRasterSizeForPatternFill`,
/// `WebGPUShapeSystem.ts:640-684`) and, M4e, SHADER-IMAGE sources for the
/// image-backed procedural fills (`ShaderImageResourceManager.get` analog;
/// requests carry their `Processing` and store under the production resource
/// key).
///
/// Pure scene scan — no IO — so it is NOT `native-io`-gated: the wasm32
/// build uses it to tell the browser what to fetch. Scans EVERY page; the
/// server render lanes use [`collect_pattern_requests_for_pages`] to scope to
/// the page(s) actually being rendered.
pub fn collect_pattern_requests(
    scene: &moda_scene::Scene,
    export_scale: f64,
) -> Vec<PrefetchRequest> {
    let all_pages: Vec<usize> = (0..scene.doc().pages.len()).collect();
    collect_pattern_requests_for_pages(scene, export_scale, &all_pages)
}

/// Like [`collect_pattern_requests`] but scoped to `page_indices` — the assets
/// DRAWN on those pages only. Each page's display list already includes every
/// asset it draws (a shared image, or a master/background applied to that page),
/// so a per-page render scoped this way never misses one of its own assets; it
/// only skips assets on pages it is NOT rendering. Out-of-range indices are
/// ignored, and duplicate indices are harmless (requests dedupe downstream by
/// src / resource key).
///
/// The server render lanes (moda-py's `settle_assets`, the CLI) call this so an
/// N-page doc rendered one page per subprocess stops prefetching + decoding all
/// N pages' assets on every per-page render (~Nx asset work → 1x). Gated by
/// [`scope_assets_to_rendered_pages_enabled`] at those call sites; the wasm
/// editor keeps using the all-pages [`collect_pattern_requests`].
pub fn collect_pattern_requests_for_pages(
    scene: &moda_scene::Scene,
    export_scale: f64,
    page_indices: &[usize],
) -> Vec<PrefetchRequest> {
    let mut out: Vec<PrefetchRequest> = Vec::new();
    let effective_pixel_ratio = sizing::effective_svg_raster_pixel_ratio(Some(export_scale));
    // Skip a repeated page index (batch's `parse_page_list` may pass e.g.
    // `3,3`): building the display list twice is wasted scene-tree traversal,
    // and downstream already dedupes requests by src — so this only avoids
    // redundant work, never changes the collected set.
    let mut visited = vec![false; scene.doc().pages.len()];

    for &page_index in page_indices {
        if page_index >= visited.len() || visited[page_index] {
            continue;
        }
        visited[page_index] = true;
        let Some(page) = scene.doc().pages.get(page_index) else {
            continue;
        };
        let page_id = page.id().to_string();
        let dl = moda_scene::build_display_list(scene, &page_id);
        collect_requests_from_display_list(scene, &dl, effective_pixel_ratio, &mut out);
    }
    out
}

/// Asset sources drawn by the freeform CANVAS-ROOT (floating, off-page) nodes
/// — the page-scoped collectors above never see them (they walk page display
/// lists), so the page-less canvas-root render lane
/// (`moda_engine::render_canvas_root`) collects through this instead. Same
/// per-item semantics as the page walk, over
/// [`moda_scene::build_display_list_for_canvas`].
pub fn collect_pattern_requests_for_canvas_root(
    scene: &moda_scene::Scene,
    export_scale: f64,
) -> Vec<PrefetchRequest> {
    let dl = moda_scene::build_display_list_for_canvas(scene);
    collect_pattern_requests_for_canvas_list(scene, &dl, export_scale)
}

/// [`collect_pattern_requests_for_canvas_root`] over the canvas root's display list `dl`
/// (`moda_scene::build_display_list_for_canvas(scene)`), for a caller that has it already.
pub fn collect_pattern_requests_for_canvas_list(
    scene: &moda_scene::Scene,
    dl: &moda_scene::DisplayList,
    export_scale: f64,
) -> Vec<PrefetchRequest> {
    let mut out: Vec<PrefetchRequest> = Vec::new();
    let effective_pixel_ratio = sizing::effective_svg_raster_pixel_ratio(Some(export_scale));
    collect_requests_from_display_list(scene, dl, effective_pixel_ratio, &mut out);
    out
}

/// Every asset the editor may draw: all pages ([`collect_pattern_requests`])
/// plus the canvas root ([`collect_pattern_requests_for_canvas_root`]). The
/// wasm editor lane draws both (a Design document has no pages at all: its
/// frames are canvas-root nodes), so it scans both. Requests may repeat a src;
/// callers dedupe by src / resource key as they already do.
pub fn collect_pattern_requests_with_canvas_root(
    scene: &moda_scene::Scene,
    export_scale: f64,
) -> Vec<PrefetchRequest> {
    let mut out = collect_pattern_requests(scene, export_scale);
    out.extend(collect_pattern_requests_for_canvas_root(
        scene,
        export_scale,
    ));
    out
}

/// The shared per-display-list scan behind the page and canvas-root
/// collectors (one item-processing implementation, so the two lanes can't
/// drift).
fn collect_requests_from_display_list(
    scene: &moda_scene::Scene,
    dl: &moda_scene::DisplayList,
    effective_pixel_ratio: f64,
    out: &mut Vec<PrefetchRequest>,
) {
    for item in &dl.items {
        if !item.is_primary_node_item {
            continue;
        }
        let Some(node) = scene.effective_node(&item.node_id) else {
            continue;
        };
        // M4e: image-backed shader fill? Only a renderable source-mask
        // EFFECT carries an image source (`WebGPUPathRenderer.ts:2050` —
        // the fillShaderId-only route always binds the fallback texture).
        if let Some((shader_id, raw_params, Some(effect))) =
            moda_scene::shader_effects::effective_shader_fill(&node)
        {
            let empty = moda_scene::value::Value::Object(moda_scene::value::Map::new());
            let params = raw_params.unwrap_or(&empty);
            if let Some(processing) =
                moda_shaders::runtime::shader_image_processing(shader_id, &params.to_json())
            {
                if let Some(src) = moda_scene::shader_effects::effect_image_src(&node, effect) {
                    out.push(PrefetchRequest {
                        src: src.to_string(),
                        min_raster_size: None,
                        shader_processing: Some(processing),
                        node_id: Some(node.id().to_string()),
                    });
                }
            }
        }
        // Every shown fill's image, not only the top one (E3.1 P3): a fill stack draws each layer (an image under a
        // blended tint, `paint::layers`), and an image the prefetch never saw drew as a placeholder, the export
        // failing with a missing resource. A video fill prefetches its poster through this same path: P0 renders
        // the poster via the URL-keyed pattern texture, so the browser must fetch it exactly as it fetches an
        // image's src. (Posters are raster, so the SVG min-raster-size hint below resolves to None for them.)
        let fills = moda_scene::paint::node_fills_shared(&node);
        for fill in fills.iter().filter(|f| moda_scene::paint::shown(f.visible, f.opacity)) {
            let (src, placement, is_pattern_fill) = match &fill.paint {
                moda_scene::paint::Paint::Image(p) => (p.src.as_deref(), &p.placement, true),
                moda_scene::paint::Paint::Video(p) => (p.poster_src.as_deref(), &p.placement, false),
                _ => continue,
            };
            // A paint layer's surface (raster spike) is filled by the engine's surface registry, never fetched.
            let Some(src) = src.filter(|s| !s.is_empty() && !s.starts_with(moda_scene::normalize::PAINT_SURFACE_SRC_PREFIX)) else {
                continue;
            };

            // The node's on-screen footprint: non-tiling pattern fills only
            // (`fillPatternRepeat && !== 'no-repeat'` bails; missing repeat does NOT
            // bail — WebGPUShapeSystem.ts:650-651).
            //
            // Deliberately NOT gated on `is_likely_svg(src)`. That gate reads the URL,
            // and in production a pattern's src is an extension-less capability URL
            // (`/api/v2/images/ref/<uuid>?v=…&s=…`), so it answered `false` for every
            // uploaded SVG on a real canvas — the hint reached `resolve_target_size` as
            // None precisely for the assets whose size most needed bounding. Format is
            // only knowable from the bytes, which this request builder has not fetched
            // yet, so compute the footprint for every candidate and let the decoder
            // decide: `resolve_target_size` reads it only under `is_svg`, leaving raster
            // sizing untouched. Video posters stay None — that arm is raster by
            // construction, and the decode cache keys off this value.
            let tiling = placement.mode() == "tile";
            let min_raster_size = if is_pattern_fill && !tiling {
                match placement.natural_size.map(|s| (s.width, s.height)) {
                    Some((nw, nh)) if nw > 0.0 && nh > 0.0 => {
                        // The scale the image draws at in the node's box (Fill / Fit resolved).
                        let resolved = placement.resolve(
                            moda_scene::paint::node_image_box(&node).unwrap_or([0.0; 4]),
                            None,
                        );
                        let (psx, psy) = (resolved.scale.0.abs(), resolved.scale.1.abs());
                        let abs = scene
                            .node_canvas_affine(node.id())
                            .map(|a| {
                                (
                                    (a.a * a.a + a.b * a.b).sqrt(),
                                    (a.c * a.c + a.d * a.d).sqrt(),
                                )
                            })
                            .unwrap_or((1.0, 1.0));
                        sizing::compute_desired_svg_raster_size(
                            nw,
                            nh,
                            (psx, psy),
                            abs,
                            effective_pixel_ratio,
                            // The LIVE grant, not the pinned constant: on a
                            // baseline-tier device (WEB-NEXTJS-21G) an SVG
                            // raster sized past the grant would fail texture
                            // creation.
                            Some(constants::live_max_texture_dimension() as f64),
                        )
                    }
                    _ => None,
                }
            } else {
                None
            };

            out.push(PrefetchRequest {
                src: src.to_string(),
                min_raster_size,
                shader_processing: None,
                node_id: Some(node.id().to_string()),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_entry_accessors() {
        let mut store = AssetStore::default();
        store.insert_external(
            "k".into(),
            ExternalImageMeta {
                width: 256,
                height: 128,
                scale_factor: (2.0, 4.0),
                upload_id: 3,
            },
        );
        // A browser-decoded, GPU-resident raster reports Loaded with its
        // uploaded dims + scale factor, but carries NO heap image (the pixels
        // never entered the wasm heap).
        assert_eq!(store.fill_state("k"), ImageFillState::Loaded);
        assert_eq!(store.dimensions("k"), Some((256, 128)));
        assert_eq!(store.scale_factor("k"), Some((2.0, 4.0)));
        assert!(store.image("k").is_none());
        let meta = store.external("k").expect("external meta present");
        assert_eq!((meta.width, meta.height), (256, 128));
        assert_eq!(meta.upload_id, 3);

        // A heap entry is not external; a never-requested key is neither.
        store.insert(
            "heap".into(),
            AssetEntry::Loaded(Arc::new(PreparedImage {
                width: 4,
                height: 4,
                natural_width: 4,
                natural_height: 4,
                scale_factor: (1.0, 1.0),
                rgba_premul: Arc::new(vec![0; 4 * 4 * 4]),
                is_svg: false,
                icc_applied: false,
                coverage_permille: 1000,
                straight: false,
                blank_cause: "",
            })),
        );
        assert!(store.external("heap").is_none());
        assert!(store.image("heap").is_some());
        assert!(store.external("missing").is_none());
        assert_eq!(store.fill_state("missing"), ImageFillState::Loading);
    }

    #[test]
    fn clear_external_drops_only_external_entries_and_reverts_to_loading() {
        let mut store = AssetStore::default();
        store.insert_external(
            "ext".into(),
            ExternalImageMeta {
                width: 8,
                height: 8,
                scale_factor: (1.0, 1.0),
                upload_id: 1,
            },
        );
        store.insert(
            "heap".into(),
            AssetEntry::Loaded(Arc::new(PreparedImage {
                width: 2,
                height: 2,
                natural_width: 2,
                natural_height: 2,
                scale_factor: (1.0, 1.0),
                rgba_premul: Arc::new(vec![0; 2 * 2 * 4]),
                is_svg: false,
                icc_applied: false,
                coverage_permille: 1000,
                straight: false,
                blank_cause: "",
            })),
        );
        store.insert("failed".into(), AssetEntry::Failed);

        // Mirrors `clear_external_images` teardown on a reused wasm instance:
        // only the external metadata is dropped, and it must revert to Loading
        // so `collect_pattern_requests` re-feeds the now-closed bitmap rather
        // than resolving a bitmap-less external texture (which renders
        // transparent). Heap and failed entries are untouched.
        assert_eq!(store.clear_external(), 1);
        assert_eq!(store.fill_state("ext"), ImageFillState::Loading);
        assert!(store.external("ext").is_none());
        assert!(store.image("heap").is_some());
        assert_eq!(store.fill_state("failed"), ImageFillState::Failed);

        // Idempotent: a second clear removes nothing.
        assert_eq!(store.clear_external(), 0);
    }

    #[test]
    fn clear_prefixed_drops_streaming_keys_heap_and_external_only() {
        let mut store = AssetStore::default();
        // A heap-fed live video frame — `Loaded`, NOT `LoadedExternal`.
        store.insert(
            "video:n1".into(),
            AssetEntry::Loaded(Arc::new(PreparedImage {
                width: 4,
                height: 4,
                natural_width: 4,
                natural_height: 4,
                scale_factor: (1.0, 1.0),
                rgba_premul: Arc::new(vec![0; 4 * 4 * 4]),
                is_svg: false,
                icc_applied: false,
                coverage_permille: 1000,
                straight: false,
                blank_cause: "",
            })),
        );
        // An external live video frame.
        store.insert_external(
            "video:n2".into(),
            ExternalImageMeta {
                width: 8,
                height: 8,
                scale_factor: (1.0, 1.0),
                upload_id: 9,
            },
        );
        // A normal (non-video) image — must survive.
        store.insert(
            "https://cdn/img.png".into(),
            AssetEntry::Loaded(Arc::new(PreparedImage {
                width: 2,
                height: 2,
                natural_width: 2,
                natural_height: 2,
                scale_factor: (1.0, 1.0),
                rgba_premul: Arc::new(vec![0; 2 * 2 * 4]),
                is_svg: false,
                icc_applied: false,
                coverage_permille: 1000,
                straight: false,
                blank_cause: "",
            })),
        );

        // `clear_external` alone would leave the HEAP video frame resident — the
        // canvas-switch leak. `clear_prefixed("video:")` drops the whole streaming
        // key space (heap + external) and nothing else.
        assert_eq!(store.clear_prefixed("video:"), 2);
        assert_eq!(store.fill_state("video:n1"), ImageFillState::Loading);
        assert_eq!(store.fill_state("video:n2"), ImageFillState::Loading);
        assert!(store.image("https://cdn/img.png").is_some());
        assert_eq!(store.clear_prefixed("video:"), 0); // idempotent
    }

    #[test]
    #[cfg(debug_assertions)] // release builds compile the debug_assert (and this test) out.
    #[should_panic(expected = "empty prefix")]
    fn clear_prefixed_rejects_an_empty_prefix() {
        // The guard stops a future caller from silently wiping the whole store
        // with `clear_prefixed("")`; release builds fall through to the no-op.
        let mut store = AssetStore::default();
        store.clear_prefixed("");
    }

    fn scene_from(nodes: moda_scene::value::Value) -> moda_scene::Scene {
        let json = moda_scene::value::json!({
            "nodes": nodes,
            "pages": [{"id": "p1", "name": "P1", "width": 400, "height": 300, "nodeIds": ["v"]}],
            "config": {}, "variables": [], "canvasRootNodeIds": [],
            "layoutMode": "vertical", "category": null, "animations": null
        });
        moda_scene::Scene::load(moda_scene::extract_canvas_data(&json)).unwrap()
    }

    #[test]
    fn video_fill_prefetches_its_poster() {
        // A 'video' fill's poster URL must be collected for prefetch
        // through the SAME scene scan as an image pattern's src — otherwise the
        // browser is never told to fetch it and the poster texture never
        // arrives (it would bake the Loading placeholder forever). Uses an
        // https:// poster on purpose: a data: URI would decode inline and mask
        // the collection seam.
        let scene = scene_from(moda_scene::value::json!([{
            "id": "v", "type": "rectangle", "x": 0, "y": 0, "width": 320, "height": 180,
            "fills": [{"id": "fill-0", "paint": {"type": "video", "src": "https://cdn.example.com/clip.mp4", "posterSrc": "https://cdn.example.com/poster.jpg", "playback": {}, "placement": {"repeat": "no-repeat"}}}]
        }]));
        let reqs = collect_pattern_requests(&scene, 1.0);
        assert_eq!(
            reqs.len(),
            1,
            "the video poster should be collected for prefetch"
        );
        assert_eq!(reqs[0].src, "https://cdn.example.com/poster.jpg");
        // Posters are raster, so the SVG min-raster hint stays None.
        assert!(reqs[0].min_raster_size.is_none());

        // A video fill with no poster collects nothing (mirrors a srcless pattern).
        let scene = scene_from(moda_scene::value::json!([{
            "id": "v", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10,
            "fills": [{"id": "fill-0", "paint": {"type": "video", "playback": {}, "placement": {}}}]
        }]));
        assert!(collect_pattern_requests(&scene, 1.0).is_empty());
    }

    #[test]
    fn every_shown_fill_in_a_stack_prefetches_its_image() {
        // An image under a blended tint (`<fill paint="image(…)"/><fill paint="#d4a017" blend-mode="multiply"/>`)
        // draws as a layer of its own: it must be fetched though it isn't the top fill (QA: the run preview drew a
        // flat tan disc and the export failed with "1 missing resource(s)"). A hidden layer draws nothing.
        let scene = scene_from(moda_scene::value::json!([{
            "id": "v", "type": "ellipse", "x": 0, "y": 0, "width": 280, "height": 280,
            "fills": [
                {"id": "fill-0", "paint": {"type": "image", "src": "https://cdn.example.com/crowd.jpg", "placement": {"repeat": "no-repeat"}}},
                {"id": "fill-1", "paint": {"type": "image", "src": "https://cdn.example.com/hidden.jpg", "placement": {"repeat": "no-repeat"}}, "visible": false},
                {"id": "fill-2", "paint": {"type": "solid", "color": "#d4a017"}, "blendMode": "multiply"}
            ]
        }]));
        assert_eq!(srcs(&collect_pattern_requests(&scene, 1.0)), set(&["https://cdn.example.com/crowd.jpg"]));
    }

    #[test]
    fn with_canvas_root_collects_a_page_less_design_documents_images() {
        // A Design document has no pages: its frames (and their children) are
        // canvas-root nodes. The editor's scan must still see their images
        // (QA 2026-09-26: every image in a Design document stayed the grey
        // Loading placeholder because the wasm lane scanned pages only).
        let json = moda_scene::value::json!({
            "nodes": [
                {"id": "f", "type": "container", "x": 0, "y": 0, "width": 400, "height": 300,
                 "clipContent": true, "children": ["img"],
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}]},
                {"id": "img", "type": "rectangle", "x": 10, "y": 10, "width": 200, "height": 100,
                 "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "https://cdn.example.com/photo.jpg", "placement": {"repeat": "no-repeat"}}}]},
                {"id": "loose", "type": "rectangle", "x": 500, "y": 0, "width": 50, "height": 50,
                 "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "https://cdn.example.com/loose.png", "placement": {"repeat": "no-repeat"}}}]}
            ],
            "pages": [], "config": {}, "variables": [], "canvasRootNodeIds": ["f", "loose"],
            "layoutMode": "freeform", "category": "design", "animations": null
        });
        let scene = moda_scene::Scene::load(moda_scene::extract_canvas_data(&json)).unwrap();
        assert!(
            collect_pattern_requests(&scene, 1.0).is_empty(),
            "no pages, nothing page-scoped"
        );
        assert_eq!(
            srcs(&collect_pattern_requests_with_canvas_root(&scene, 1.0)),
            set(&[
                "https://cdn.example.com/loose.png",
                "https://cdn.example.com/photo.jpg"
            ])
        );
    }

    #[test]
    fn extensionless_pattern_src_still_gets_a_footprint() {
        // Production pattern srcs are extension-less capability URLs, so the old
        // `is_likely_svg(src)` gate on this hint answered false for every
        // uploaded SVG on a real canvas (see `source::tests` — it asserts false
        // for exactly this URL shape). The footprint then reached
        // `resolve_target_size` as None, which is the only reason a 4255x5971
        // artboard could raster at 8510x11942 in a 157x255 box.
        //
        // The values are canvas c4d6996f's actual node: box 157x254.8 with
        // fillPatternScale 0.046.
        let scene = scene_from(moda_scene::value::json!([{
            "id": "v", "type": "rectangle", "x": 0, "y": 0, "width": 157, "height": 254.8,
            "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "https://api.moda.app/api/v2/images/ref/e6da8f7e-7ea9-4221-8844-4f15b284f6ad?v=55ce7652&s=a3ad6a69", "placement": {"fit": "crop", "scale": {"x": 0.0460397788411424, "y": 0.0460397788411424}, "naturalSize": {"width": 4255, "height": 5971}}}}]
        }]));
        let reqs = collect_pattern_requests(&scene, 2.0);
        assert_eq!(reqs.len(), 1);
        let footprint = reqs[0]
            .min_raster_size
            .expect("an extension-less pattern src must still carry its footprint");
        assert_eq!(footprint, (392, 550));

        // End to end: that footprint is what bounds the raster.
        assert_eq!(
            sizing::resolve_target_size(4255, 5971, true, Some(footprint)),
            (1567, 2200)
        );
    }

    #[test]
    fn tiling_pattern_still_reports_no_footprint() {
        // A repeating pattern's texture IS its tile, so it has no displayed
        // extent to size against and must keep passing None.
        let scene = scene_from(moda_scene::value::json!([{
            "id": "v", "type": "rectangle", "x": 0, "y": 0, "width": 157, "height": 255,
            "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "https://api.moda.app/api/v2/images/ref/abc", "placement": {"fit": "tile", "scale": {"x": 0.046, "y": 0.046}, "naturalSize": {"width": 4255, "height": 5971}}}}]
        }]));
        let reqs = collect_pattern_requests(&scene, 2.0);
        assert_eq!(reqs.len(), 1);
        assert!(reqs[0].min_raster_size.is_none());
    }

    fn img_node(id: &str, src: &str) -> moda_scene::value::Value {
        moda_scene::value::json!({
            "id": id, "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10,
            "fills": [{"id": "fill-0", "paint": {"type": "image", "src": src, "placement": {"repeat": "no-repeat"}}}]
        })
    }

    /// p0: image A + shared S; p1: image B; p2: image C + shared S (a second
    /// node with the same src). Sharing is per-src (requests dedupe), so S must
    /// appear whenever a page that draws it is rendered.
    fn multi_page_scene() -> moda_scene::Scene {
        let json = moda_scene::value::json!({
            "nodes": [
                img_node("a", "https://x/a.png"),
                img_node("s0", "https://x/shared.png"),
                img_node("b", "https://x/b.png"),
                img_node("c", "https://x/c.png"),
                img_node("s2", "https://x/shared.png"),
            ],
            "pages": [
                {"id": "p0", "name": "P0", "width": 400, "height": 300, "nodeIds": ["a", "s0"]},
                {"id": "p1", "name": "P1", "width": 400, "height": 300, "nodeIds": ["b"]},
                {"id": "p2", "name": "P2", "width": 400, "height": 300, "nodeIds": ["c", "s2"]},
            ],
            "config": {}, "variables": [], "canvasRootNodeIds": [],
            "layoutMode": "vertical", "category": null, "animations": null
        });
        moda_scene::Scene::load(moda_scene::extract_canvas_data(&json)).unwrap()
    }

    fn srcs(reqs: &[PrefetchRequest]) -> std::collections::BTreeSet<String> {
        reqs.iter().map(|r| r.src.clone()).collect()
    }

    fn set(items: &[&str]) -> std::collections::BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn all_pages_collect_sees_every_pages_assets() {
        let scene = multi_page_scene();
        assert_eq!(
            srcs(&collect_pattern_requests(&scene, 1.0)),
            set(&[
                "https://x/a.png",
                "https://x/b.png",
                "https://x/c.png",
                "https://x/shared.png",
            ])
        );
    }

    #[test]
    fn scoping_to_one_page_collects_only_that_page() {
        let scene = multi_page_scene();
        // Page 1 draws only image B — not A, C, or the shared S. This is the Nx
        // win: a page-1 render no longer fetches+decodes the other pages' assets.
        assert_eq!(
            srcs(&collect_pattern_requests_for_pages(&scene, 1.0, &[1])),
            set(&["https://x/b.png"])
        );
    }

    #[test]
    fn scoping_still_loads_a_pages_shared_asset() {
        // Correctness invariant: a shared image loads for EVERY page that draws
        // it, and never for a page that does not — no missing images.
        let scene = multi_page_scene();
        assert!(srcs(&collect_pattern_requests_for_pages(&scene, 1.0, &[0]))
            .contains("https://x/shared.png"));
        assert!(srcs(&collect_pattern_requests_for_pages(&scene, 1.0, &[2]))
            .contains("https://x/shared.png"));
        assert!(
            !srcs(&collect_pattern_requests_for_pages(&scene, 1.0, &[1]))
                .contains("https://x/shared.png")
        );
    }

    /// Every page carries the SAME background image (a page-level pattern fill —
    /// the "master / global background applied to all pages" case). The scene
    /// synthesizes a locked page-background rect per page from the page's own
    /// fill surface (`create_rectangle_from_page` passes `fillPatternImageSrc`
    /// through), so the bg image is DRAWN — and must be prefetched — on every
    /// page, plus each page has its own unique foreground image.
    fn scene_with_global_page_background() -> moda_scene::Scene {
        let page = |id: &str, fg: &str| {
            moda_scene::value::json!({
                "id": id, "name": id, "width": 400, "height": 300, "nodeIds": [fg],
                "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "https://x/bg.png", "placement": {"repeat": "no-repeat"}}}]
            })
        };
        let json = moda_scene::value::json!({
            "nodes": [
                img_node("fg0", "https://x/fg0.png"),
                img_node("fg1", "https://x/fg1.png"),
                img_node("fg2", "https://x/fg2.png"),
            ],
            "pages": [
                page("p0", "fg0"),
                page("p1", "fg1"),
                page("p2", "fg2"),
            ],
            "config": {}, "variables": [], "canvasRootNodeIds": [],
            "layoutMode": "vertical", "category": null, "animations": null
        });
        moda_scene::Scene::load(moda_scene::extract_canvas_data(&json)).unwrap()
    }

    #[test]
    fn scoping_loads_the_global_page_background_for_every_page() {
        // Correctness invariant (the master/global-asset case): a background
        // image applied to ALL pages must load for EVERY page's scoped render —
        // never a missing background. Each page also loads only its own fg.
        let scene = scene_with_global_page_background();
        for (idx, fg) in [
            (0usize, "https://x/fg0.png"),
            (1, "https://x/fg1.png"),
            (2, "https://x/fg2.png"),
        ] {
            let got = srcs(&collect_pattern_requests_for_pages(&scene, 1.0, &[idx]));
            assert_eq!(
                got,
                set(&["https://x/bg.png", fg]),
                "page {idx} scoped must load its global bg + only its own fg"
            );
        }
    }

    #[test]
    fn scoping_the_full_page_set_equals_all_pages() {
        // The CLI all-pages invocation (or the kill switch OFF path) must be
        // byte-for-byte the unscoped scan — no asset dropped.
        let scene = multi_page_scene();
        assert_eq!(
            srcs(&collect_pattern_requests_for_pages(&scene, 1.0, &[0, 1, 2])),
            srcs(&collect_pattern_requests(&scene, 1.0))
        );
    }

    #[test]
    fn out_of_range_and_duplicate_indices_are_safe() {
        let scene = multi_page_scene();
        assert!(collect_pattern_requests_for_pages(&scene, 1.0, &[99]).is_empty());
        // Duplicate index dedupes downstream by src.
        assert_eq!(
            srcs(&collect_pattern_requests_for_pages(&scene, 1.0, &[1, 1])),
            set(&["https://x/b.png"])
        );
    }

    #[test]
    fn scope_flag_defaults_on_and_honors_opt_out() {
        assert!(scope_flag_from_env(None), "unset ⇒ scoping ON");
        assert!(scope_flag_from_env(Some("1")));
        assert!(scope_flag_from_env(Some("true")));
        for off in ["0", "false", "no", "off", "OFF", " 0 "] {
            assert!(!scope_flag_from_env(Some(off)), "{off:?} must opt out");
        }
    }
}
