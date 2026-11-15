//! TextHost — the font/asset host: vendored catalog + families index + the MANIFEST snapshot the vector lane's
//! font store and the degraded tier read, plus the Noto emoji SVGs (`EmojiTextureManager.ts`), rasterized and
//! cached here.
//!
//! On wasm32 the JS glue installs the font files as an in-memory overlay (`install_vfs`); natively they come from a
//! fonts dir (`fixtures/oracle/fonts` layout).

use crate::catalog::Catalog;
use crate::lru::ByteLruCache;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

// ---------------------------------------------------------------------------
// WASM-EDITOR PROTOTYPE (research/wasm-editor/SPIKE-wasm-render.md): in-memory
// file overlay. On wasm32 there is no filesystem; the JS glue fetches font
// assets (catalog.json / MANIFEST.json / families-index.json / metrics /
// TTFs / atlas PNGs / emoji SVGs) and installs them as a path→bytes map.
// Installs MERGE (new keys added, existing keys overwritten) so the editor
// can feed fonts per-canvas on demand — mid-session font changes re-ingest
// the delta and re-load the TextHost. Every fs read below checks the overlay
// first, then falls through to std::fs — native behavior is unchanged when
// no overlay is installed.
// ---------------------------------------------------------------------------

// Bytes are stored behind `Arc` so the degraded cosmic-text engine can SHARE
// the VFS allocation (via `fontdb::Source::Binary`) instead of holding a second
// persistent copy of every TTF/OTF — see `vfs_font_sources_sorted`.
static VFS: OnceLock<std::sync::RwLock<HashMap<String, Arc<Vec<u8>>>>> = OnceLock::new();

/// Install (merge) the in-memory file overlay. Keys must be the exact string
/// form of the paths the host will ask for (e.g. `/fonts/catalog.json` when
/// `TextHost::load(Path::new("/fonts"))`). Existing keys are overwritten;
/// keys absent from `files` are kept (incremental per-canvas font feeding).
pub fn install_vfs(files: HashMap<String, Vec<u8>>) {
    let lock = VFS.get_or_init(|| std::sync::RwLock::new(HashMap::new()));
    lock.write()
        .unwrap()
        .extend(files.into_iter().map(|(k, v)| (k, Arc::new(v))));
}

// Scene overlays are immutable and owned by hosts and their lazy font stores. The registry
// only locates live files; it never keeps a retired document's resources alive.
struct ScopedFile {
    bytes: Arc<Vec<u8>>,
}
static SCOPED_VFS: OnceLock<Mutex<HashMap<String, std::sync::Weak<ScopedFile>>>> = OnceLock::new();

pub(crate) struct VfsLease(Vec<(String, Arc<ScopedFile>)>);
impl VfsLease {
    fn install(files: HashMap<String, Vec<u8>>) -> Self {
        let mut registry = SCOPED_VFS.get_or_init(Default::default).lock().unwrap();
        registry.retain(|_, file| file.strong_count() > 0);
        Self(files.into_iter().map(|(path, bytes)| {
            let file = registry.get(&path).and_then(std::sync::Weak::upgrade).unwrap_or_else(|| {
                let file = Arc::new(ScopedFile { bytes: Arc::new(bytes.clone()) });
                registry.insert(path.clone(), Arc::downgrade(&file));
                file
            });
            assert_eq!(file.bytes.as_slice(), bytes.as_slice(), "immutable font overlay path changed");
            (path, file)
        }).collect())
    }
}
impl Drop for VfsLease {
    fn drop(&mut self) {
        let mut registry = SCOPED_VFS.get().unwrap().lock().unwrap();
        for (path, file) in &self.0 {
            if Arc::strong_count(file) == 1 {
                registry.remove(path);
            }
        }
    }
}

/// Read immutable font bytes without making another persistent binary copy.
pub fn read_file_shared(path: &Path) -> Result<Arc<Vec<u8>>, String> {
    let key = path.to_string_lossy();
    if let Some(vfs) = SCOPED_VFS.get() {
        if let Some(file) = vfs.lock().unwrap().get(key.as_ref()).and_then(std::sync::Weak::upgrade) {
            return Ok(file.bytes.clone());
        }
    }
    if let Some(vfs) = VFS.get() {
        if let Some(bytes) = vfs.read().unwrap().get(key.as_ref()) {
            return Ok(bytes.clone());
        }
    }
    let bytes = if path.extension().is_some_and(|extension| extension == "font-face") {
        crate::collection::read(path)?
    } else {
        std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?
    };
    Ok(Arc::new(bytes))
}

/// VFS-first byte read (wasm-editor): the in-memory overlay is consulted
/// before `std::fs`. `pub` so the wasm entry point can read the installed
/// MANIFEST.json to build the vector store, and the vector store can read
/// vendored TTFs from the same overlay. Native behavior is unchanged when no
/// overlay is installed.
pub fn read_file_bytes(path: &Path) -> Result<Vec<u8>, String> {
    read_file_shared(path).map(|bytes| bytes.as_ref().clone())
}

/// Every installed font binary (`.ttf`/`.otf`/`.ttc`) as a SHARED handle, in
/// stable key order. Lets the degraded engine be rebuilt straight from the VFS
/// (`fontdb::Source::Binary`) with no second persistent copy of the font bytes:
/// `Database::load_font_data(v)` is exactly `load_font_source(Source::Binary(
/// Arc::new(v)))`, so passing the VFS's own `Arc` is byte-for-byte equivalent
/// while sharing the allocation. Cumulative (the VFS merges across installs) and
/// sorted so the cosmic-text fallback-face resolution stays deterministic across
/// captures/loads (the order the old per-install TTF accumulator guaranteed).
pub fn vfs_font_sources_sorted() -> Vec<Arc<Vec<u8>>> {
    let Some(vfs) = VFS.get() else {
        return Vec::new();
    };
    let map = vfs.read().unwrap();
    let mut entries: Vec<(&String, &Arc<Vec<u8>>)> = map
        .iter()
        .filter(|(k, _)| {
            let lower = k.to_lowercase();
            lower.ends_with(".ttf") || lower.ends_with(".otf") || lower.ends_with(".ttc")
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries.into_iter().map(|(_, v)| Arc::clone(v)).collect()
}

pub(crate) fn read_file_string(path: &Path) -> Result<String, String> {
    let bytes = read_file_bytes(path)?;
    String::from_utf8(bytes).map_err(|e| format!("utf8 {path:?}: {e}"))
}

// ---------------------------------------------------------------------------
// Emoji SVG byte provider seam.
//
// `emoji_rgba` reads a cluster's SVG from the loose vendored `emoji/` dir (or
// the wasm VFS) first, then — if that misses — from an OPTIONAL process-global
// provider installed by native code. `moda-fonts` installs one that serves the
// bundled full Noto SVG pack (`moda_fonts::emoji_bundle`), so the baked lane
// renders every emoji instead of a transparent hole.
//
// This is a bare function-pointer seam so `moda-text` stays a leaf on the web
// lane: on wasm32 nothing installs a provider (the browser feeds emoji SVGs
// into the VFS on demand), so this is inert there and pulls in no HTTP/archive
// code. `moda-text` gains no dependency.
// ---------------------------------------------------------------------------
type EmojiBytesProvider = Box<dyn Fn(&str) -> Option<Vec<u8>> + Send + Sync>;
static EMOJI_PROVIDER: OnceLock<EmojiBytesProvider> = OnceLock::new();

/// Install the process-global emoji SVG byte provider (first call wins). The
/// provider is consulted by [`TextHost::emoji_rgba`] only when the vendored
/// loose dir / VFS has no bytes for the cluster.
pub fn install_emoji_bytes_provider(provider: EmojiBytesProvider) {
    let _ = EMOJI_PROVIDER.set(provider);
}

// A miss RECORDER, installed by native code alongside the provider. It fires
// whenever `emoji_rgba` yields no image — the SINGLE choke both the planning
// site (`ops.rs`, which drops the emoji op) and the draw site
// (`moda-engine::text_draw::emoji_bind_group`) route through — so a miss is
// reported (classified into a browser-parity coverage gap vs. a real missing
// resource by the installed backend) no matter which site observes it. Recording
// HERE (not only at the draw site) is required because the planning site drops
// unrenderable emoji before they ever reach a draw. moda-text can't depend on
// the report crate, so this stays a bare seam (inert on wasm — nothing installs
// it there, and the browser has its own emoji fetch path).
type EmojiMissRecorder = Box<dyn Fn(&str) + Send + Sync>;
static EMOJI_MISS_RECORDER: OnceLock<EmojiMissRecorder> = OnceLock::new();

/// Install the process-global emoji-miss recorder (first call wins). Called by
/// [`TextHost::emoji_rgba`] every time a cluster yields no image.
pub fn install_emoji_miss_recorder(recorder: EmojiMissRecorder) {
    let _ = EMOJI_MISS_RECORDER.set(recorder);
}

fn record_emoji_miss(cluster: &str) {
    if let Some(rec) = EMOJI_MISS_RECORDER.get() {
        rec(cluster);
    }
}

fn emoji_provider_bytes(cluster: &str) -> Option<Vec<u8>> {
    EMOJI_PROVIDER.get().and_then(|f| f(cluster))
}

/// Decoded RGBA8 image (tightly packed).
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

pub struct TextHost {
    vfs_lease: Option<Arc<VfsLease>>,
    pub catalog: Catalog,
    pub families_index: serde_json::Map<String, Value>,
    /// The fonts dir this host was loaded from — the dir whose `catalog.json`
    /// produced [`Self::catalog`] and whose `MANIFEST.json` vendored its bytes.
    /// Carried so a consumer that needs a SECOND view of the same vendored set
    /// (the vector lane's `VectorFontStore`, which resolves TTF outlines out of
    /// `MANIFEST.files` keyed by the catalog's `ttf_url`) can build it from THIS
    /// dir instead of re-discovering one. Those two must agree: a store built
    /// from a different dir looks the catalog's URLs up in the wrong manifest,
    /// misses every one of them, and silently degrades the whole document to the
    /// degraded tier.
    root: PathBuf,
    /// `MANIFEST.files` (ttf url → vendored path) — the vector lane's half of the
    /// same snapshot [`Self::catalog`] came from. Kept HERE, extracted during
    /// [`Self::load`], so building the (thread-local, non-`Sync`) store never
    /// re-reads the fonts dir: a materialized per-canvas dir is deliberately not
    /// load-bearing once a warm host holds it (`moda_fonts::prune_fonts_by_canvas`
    /// may sweep it from another process), so a lazy re-read would turn a pruned
    /// dir into a hard vector-lane failure.
    vector_files: HashMap<String, String>,
    manifest: Value,
    emoji_dir: PathBuf,
    /// decoded emoji cache: cluster → premultiplied RgbaImage (the browser
    /// uploads premultiplied — `copyExternalImageSafe(premultipliedAlpha:
    /// true)`; resvg/tiny-skia output is premultiplied already). Byte-capped
    /// LRU; `None` (missing-SVG placeholder) is charged ~0 bytes so it stays
    /// cached cheaply.
    emoji_cache: Mutex<ByteLruCache<String, Option<std::sync::Arc<RgbaImage>>>>,
}

/// Decoded emoji RGBA cache caps. The GPU is the working copy; this CPU cache only needs to hold the emoji a frame
/// may re-upload, bounded so an emoji-heavy canvas can't pin decoded RGBA in the never-shrinking wasm arena. Native
/// builds may override the byte cap via env for profiling. The entry cap matters as much: missing-glyph lookups cache
/// a `None` placeholder charged ~0 bytes, which the byte cap can never evict.
const DEFAULT_EMOJI_CACHE_BYTES: u64 = 32 * 1024 * 1024;
pub const EMOJI_CACHE_MAX_ENTRIES: usize = 4096;

#[cfg(not(target_family = "wasm"))]
fn cache_cap_bytes(env_key: &str, default_mb_bytes: u64) -> u64 {
    std::env::var(env_key)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(default_mb_bytes)
}

#[cfg(target_family = "wasm")]
fn cache_cap_bytes(_env_key: &str, default_mb_bytes: u64) -> u64 {
    default_mb_bytes
}

/// Byte cap for decoded emoji. `pub` for the engine's GPU emoji bind-group cache (the GPU texture and the CPU
/// RGBA are the same `w*h*4` bytes).
pub fn emoji_cap_bytes() -> u64 {
    cache_cap_bytes("MODA_TEXT_EMOJI_CACHE_MB", DEFAULT_EMOJI_CACHE_BYTES)
}

#[inline]
fn rgba_byte_size(img: &RgbaImage) -> u64 {
    img.rgba.len() as u64
}

impl TextHost {
    /// Load immutable in-memory files, retained until the host and its lazy stores retire.
    pub fn load_with_files(fonts_root: &Path, files: HashMap<String, Vec<u8>>) -> Result<TextHost, String> {
        let lease = Arc::new(VfsLease::install(files));
        let mut host = Self::load(fonts_root)?;
        host.vfs_lease = Some(lease);
        Ok(host)
    }

    /// Load from a vendored fonts dir (`fixtures/oracle/fonts` layout:
    /// catalog.json + MANIFEST.json + families-index.json + per-family dirs).
    pub fn load(fonts_root: &Path) -> Result<TextHost, String> {
        let read_json = |name: &str| -> Result<Value, String> {
            let p = fonts_root.join(name);
            let s = read_file_string(&p)?;
            serde_json::from_str(&s).map_err(|e| format!("parse {p:?}: {e}"))
        };
        let catalog_json = read_json("catalog.json")?;
        let catalog = Catalog::from_catalog_json(&catalog_json);
        let manifest = read_json("MANIFEST.json")?;
        let families_index = read_json("families-index.json")?
            .get("families")
            .and_then(|f| f.as_object())
            .cloned()
            .ok_or("families-index.json missing families object")?;

        Ok(TextHost {
            vfs_lease: None,
            catalog,
            families_index,
            root: fonts_root.to_path_buf(),
            vector_files: crate::vector::manifest_files(&manifest),
            manifest,
            emoji_dir: fonts_root.join("emoji"),
            emoji_cache: Mutex::new(ByteLruCache::new(
                emoji_cap_bytes(),
                EMOJI_CACHE_MAX_ENTRIES,
            )),
        })
    }

    /// A vector font store over THIS host's vendored snapshot — the same dir and
    /// the same `MANIFEST.files` that produced [`Self::catalog`].
    ///
    /// The store is the vector lane's other half: it resolves a variant by
    /// looking the catalog's `ttf_url` STRING up in `MANIFEST.files`. Sourcing
    /// both from one host is what makes them impossible to mispair — a store
    /// built over a different dir misses every url and silently degrades the
    /// whole document to the degraded tier.
    ///
    /// Lenient: an un-vendored catalog TTF falls back per node (browser parity
    /// with a not-yet-loaded font), never a panic.
    pub fn vector_font_store_lenient(&self) -> crate::vector::VectorFontStore {
        let mut store = crate::vector::VectorFontStore::from_files(&self.root, self.vector_files.clone(), true);
        store.vfs_lease = self.vfs_lease.clone();
        store
    }

    /// Build the degraded text tier from this host's retained manifest, so a
    /// concurrent directory writer cannot change the set between text systems.
    pub fn degraded_engine(&self) -> crate::degraded::DegradedEngine {
        crate::degraded::build_catalog_engine_from_snapshot(&self.root, &self.manifest)
    }

    /// Compare against the manifest snapshot already loaded by this host, without
    /// rereading a shared directory. Includes TTF-only resources, which need not
    /// have an atlas key. Byte-read failures remain the vector store's concern.
    pub fn covers_vendored_files(&self, files: &HashMap<String, String>) -> bool {
        files
            .iter()
            .all(|(url, path)| self.vector_files.get(url) == Some(path))
    }

    /// Rasterize (and cache) one emoji cluster from its Noto SVG at intrinsic
    /// size (Chromium rasterizes the blob at intrinsic size; Noto SVGs have a
    /// 128×128 viewBox). PREMULTIPLIED rgba, matching the browser's
    /// `premultipliedAlpha: true` texture upload. None = the browser's 1×1
    /// transparent failure placeholder.
    ///
    /// SVG source order: the vendored loose `emoji/` dir (also the wasm VFS,
    /// where the browser feeds per-cluster SVGs) FIRST, then the installed
    /// [`install_emoji_bytes_provider`] provider — the native bundled full Noto
    /// pack. Without the provider (e.g. wasm) only the loose dir/VFS is
    /// consulted, so behavior there is unchanged.
    pub fn emoji_rgba(&self, cluster: &str) -> Option<std::sync::Arc<RgbaImage>> {
        if let Some(cached) = self.emoji_cache.lock().unwrap().get(cluster) {
            // Re-record a cached miss too: the negative cache lives for the
            // host's lifetime (a process-global default host outlives many
            // renders) while the report resets per render, so classifying only
            // when the cache is first populated would drop the miss from every
            // later render. The recorder dedupes within a render.
            if cached.is_none() {
                record_emoji_miss(cluster);
            }
            return cached;
        }
        let fname = emoji_cluster_to_noto_filename(cluster);
        let svg = read_file_bytes(&self.emoji_dir.join(&fname))
            .ok()
            .or_else(|| emoji_provider_bytes(cluster));
        let result = svg
            .and_then(|bytes| rasterize_svg_bytes(&bytes))
            .map(std::sync::Arc::new);
        if result.is_none() {
            record_emoji_miss(cluster);
        }
        // Charge the decoded bytes (or ~0 for the missing-SVG `None` placeholder,
        // so negatives stay cached cheaply but can still be evicted).
        let bytes = result.as_ref().map(|img| rgba_byte_size(img)).unwrap_or(0);
        self.emoji_cache
            .lock()
            .unwrap()
            .insert(cluster.to_string(), result.clone(), bytes);
        result
    }
}

/// `emojiClusterToNotoUrl` filename part (`utils/msdf/emoji.ts:72-80`):
/// codepoints joined by `_`, VS15/VS16 stripped, lowercase hex zero-padded to at
/// least 4 digits (Noto convention — e.g. `emoji_u00a9.svg`, `emoji_u0031_20e3.svg`;
/// unpadded bases below 0x1000 404 on the CDN).
pub fn emoji_cluster_to_noto_filename(cluster: &str) -> String {
    let hex: Vec<String> = cluster
        .chars()
        .map(|c| c as u32)
        .filter(|cp| *cp != 0xFE0E && *cp != 0xFE0F)
        .map(|cp| format!("{cp:04x}"))
        .collect();
    format!("emoji_u{}.svg", hex.join("_"))
}

fn rasterize_svg_bytes(data: &[u8]) -> Option<RgbaImage> {
    let opt = resvg::usvg::Options::default();
    let tree = resvg::usvg::Tree::from_data(data, &opt).ok()?;
    let size = tree.size();
    let (w, h) = (size.width().ceil() as u32, size.height().ceil() as u32);
    if w == 0 || h == 0 {
        return None;
    }
    let mut pixmap = resvg::tiny_skia::Pixmap::new(w, h)?;
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::identity(),
        &mut pixmap.as_mut(),
    );
    // tiny-skia stores premultiplied RGBA — exactly what the browser uploads.
    Some(RgbaImage {
        width: w,
        height: h,
        rgba: pixmap.data().to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::oracle_fonts_root;

    /// The host hands out a vector store over its OWN vendored snapshot,
    /// so the store's `MANIFEST.files` and the host's `catalog` can never be two
    /// different dirs. Asserted through the store's ability to RESOLVE a catalog
    /// variant, not through a getter — a getter with no production caller could not
    /// fail if this regressed.
    #[test]
    #[ignore = "needs unported Moda fixture oracle/fonts/catalog.json, MANIFEST.json and families-index.json"]
    fn hands_out_a_vector_store_over_its_own_snapshot() {
        let host = TextHost::load(&oracle_fonts_root()).expect("host load");
        let store = host.vector_font_store_lenient();
        assert!(
            store
                .font_for_variant(&host.catalog, "Inter", 400, "normal", None)
                .is_some(),
            "the host's own store must resolve the host's own catalog"
        );
    }

    #[test]
    fn emoji_filename_strips_vs16_and_zero_pads() {
        // VS16 stripped; codepoints below 0x1000 zero-padded to 4 digits (Noto
        // convention — the unpadded `emoji_ua9.svg` / `emoji_u31_20e3.svg` 404).
        assert_eq!(
            emoji_cluster_to_noto_filename("©\u{FE0F}"),
            "emoji_u00a9.svg"
        );
        assert_eq!(
            emoji_cluster_to_noto_filename("1\u{FE0F}\u{20E3}"),
            "emoji_u0031_20e3.svg"
        );
        assert_eq!(emoji_cluster_to_noto_filename("🎨"), "emoji_u1f3a8.svg");
        assert_eq!(
            emoji_cluster_to_noto_filename("👨\u{200D}👩\u{200D}👧"),
            "emoji_u1f468_200d_1f469_200d_1f467.svg"
        );
        // 👩‍🔧 profession ZWJ (woman + ZWJ + wrench): ZWJ is PRESERVED in the
        // filename (Noto names ZWJ sequences with the 200d joiner).
        assert_eq!(
            emoji_cluster_to_noto_filename("\u{1F469}\u{200D}\u{1F527}"),
            "emoji_u1f469_200d_1f527.svg"
        );
        // 👩🏽‍❤️‍💋‍👩🏼 multi-person skin-tone kiss: skin-tone modifiers kept,
        // the two ZWJs kept, VS16 (FE0F) on the heart stripped.
        assert_eq!(
            emoji_cluster_to_noto_filename(
                "\u{1F469}\u{1F3FD}\u{200D}\u{2764}\u{FE0F}\u{200D}\u{1F48B}\u{200D}\u{1F469}\u{1F3FC}"
            ),
            "emoji_u1f469_1f3fd_200d_2764_200d_1f48b_200d_1f469_1f3fc.svg"
        );
    }

    #[test]
    #[ignore = "needs unported Moda fixture oracle/fonts/emoji and catalog metadata"]
    fn emoji_rasterizes_at_intrinsic_size() {
        let host = TextHost::load(&oracle_fonts_root()).expect("host load");
        let img = host.emoji_rgba("🎨").expect("vendored emoji svg");
        assert_eq!((img.width, img.height), (128, 128));
        // A cluster with no loose SVG AND no installed provider (this test does
        // not install one — the `_provider_serves...` test uses a distinct
        // cluster) → None (browser bakes a transparent placeholder).
        assert!(host.emoji_rgba("\u{1F9F1}").is_none());
    }

    #[test]
    #[ignore = "needs unported Moda fixture oracle/fonts/catalog.json, MANIFEST.json and families-index.json"]
    fn emoji_miss_invokes_the_recorder_fresh_and_cached() {
        // The planning site drops unrenderable emoji, so the miss must be
        // reported from `emoji_rgba` itself — on both the fresh resolve and the
        // negative-cache hit (the report resets per render but the cache
        // outlives it). A distinct cluster keeps other tests' misses out of the
        // count.
        static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
        install_emoji_miss_recorder(Box::new(|c| SEEN.lock().unwrap().push(c.to_string())));
        let host = TextHost::load(&oracle_fonts_root()).expect("host load");
        let cluster = "\u{1FAE9}"; // shaking-face — not vendored, no provider bytes
        assert!(host.emoji_rgba(cluster).is_none()); // fresh miss
        assert!(host.emoji_rgba(cluster).is_none()); // cached-None miss
        let n = SEEN
            .lock()
            .unwrap()
            .iter()
            .filter(|c| *c == cluster)
            .count();
        assert!(n >= 2, "recorded the fresh AND the cached miss, got {n}");
    }

    #[test]
    #[ignore = "needs unported Moda fixture oracle/fonts/catalog.json, MANIFEST.json and families-index.json"]
    fn emoji_provider_serves_loose_dir_miss() {
        // The installed provider (native lanes: the bundled Noto pack) is
        // consulted only when the loose dir / VFS has no bytes for the cluster.
        // A distinct cluster keeps this from perturbing the `is_none()` check
        // above (the provider answers only its own cluster).
        install_emoji_bytes_provider(Box::new(|cluster: &str| {
            (cluster == "\u{1F9CA}").then(|| {
                br#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10" fill="red"/></svg>"#.to_vec()
            })
        }));
        let host = TextHost::load(&oracle_fonts_root()).expect("host load");
        let img = host.emoji_rgba("\u{1F9CA}").expect("provider-served emoji");
        assert_eq!((img.width, img.height), (10, 10));
    }
}
