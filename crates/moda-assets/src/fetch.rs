//! Asset prefetch — the native replacement for the browser's two-render
//! protocol (render → wait `assetWaitTimeoutMs` → re-render,
//! `WebGPUExportCoordinator.ts:710-743`). Every source is resolved up front
//! from LOCAL bytes only — `data:` URIs, files under `--assets-dir`, absolute
//! paths — and decoded on a bounded pool of worker threads. Desktop Moda
//! renders offline: an http(s) source is never fetched; it is a committed
//! per-asset failure (rose placeholder + resource-report entry), exactly like
//! a missing file.

use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::constants::PREFETCH_CONCURRENCY;
use crate::decode::{prepare_raster, PreparedImage};
use crate::source::{
    classify, decode_data_uri, is_likely_svg, resolve_file, sniff_svg_bytes, AssetSource,
};
use crate::svg::prepare_svg;
// `PrefetchRequest` lives in lib.rs (un-gated) — the scan side is shared with
// the wasm editor's browser-fetch feeder.
use crate::{AssetEntry, AssetStore, PrefetchRequest};

/// Failure reason prefix for an http(s) source (followed by `: <url>`).
pub const REMOTE_URL_NOT_ALLOWED: &str = "remote URL not allowed (documents render offline)";

/// A single asset whose read+decode exceeds this budget is logged (WARN) so a
/// slow asset is visible. It also drives the `slow_asset_count` /
/// `slowest_asset_ms` fields on the prefetch report.
const SLOW_ASSET_WARN_MS: u64 = 1_000;

/// Per-asset read/decode split (wall-clock of each phase for one job).
/// Aggregated across the concurrent prefetch into [`PrefetchReport`]; the
/// summed values are aggregate WORK and can exceed the settle wall-clock.
#[derive(Debug, Default, Clone, Copy)]
struct Span {
    fetch_ms: u64,
    decode_ms: u64,
}

/// One failed prefetch, with node attribution — feeds the
/// CLI's resource report (WF5 r3 "scream on missing resources").
#[derive(Debug, Clone)]
pub struct AssetFailure {
    /// The original source URL (pattern src, or the shader-image src behind
    /// a resource key), log-truncated NOWHERE — the report needs the full
    /// URL to be actionable.
    pub url: String,
    pub shader_image: bool,
    /// Full error chain.
    pub reason: String,
    /// `Some(kind)` when the decode failure carries a positive browser-
    /// undecodable sniff (`moda_assets::decode::BrowserUndecodable`: EMF/WMF/
    /// HTML bytes, or a structurally complete PNG corrupt at rest). Chromium
    /// rejects the same bytes and bakes the identical failed-fill placeholder,
    /// so the miss is PIXEL PARITY, not a native quality gap: the
    /// `fail_on_missing_assets` gates (moda-py + moda-cli `failure_count`)
    /// leave it OUT of the count, exactly like fallback-class font routing,
    /// while it stays in the report so the placeholder summary still fires.
    /// Typed, not parsed back out of `reason`.
    pub browser_undecodable: Option<&'static str>,
    /// Node ids that referenced this source.
    pub nodes: Vec<String>,
}

/// Byte-accurate SVG hazard telemetry for one successfully-decoded pattern
/// asset (`moda_assets::probe::probe_svg` over the fetched bytes, server
/// prefetch lane only — the wasm editor's `decode_pattern_bytes` hot path is
/// deliberately NOT probed). These are the only SVG cases genuinely degraded
/// in BOTH lanes (usvg-unsupported features; `<text>` whose fidelity depends
/// on the catalog-seeded fontdb) — observability only, NEVER a failure: the
/// pixels shipped, and routing must not read this (the assess-time
/// `svg:pattern-image` manifest warning is likewise non-gating).
#[derive(Debug, Clone, PartialEq)]
pub struct SvgProbeWarning {
    /// SANITIZED asset reference (`sanitized_asset_ref`): unlike
    /// `AssetFailure::url` (actionable, may need the full signed URL to
    /// reproduce a fetch failure), this is telemetry for assets that LOADED —
    /// so the query string (where presigned-GCS / capability-token credentials
    /// live) is stripped, data: URIs collapse to a scheme+size stub, and the
    /// result is length-capped so it can never bloat the bounded report.
    pub url: String,
    /// Probe warning strings (`svg:text (…)`, `svg:embedded-image (…)`,
    /// `svg:sanitized-content (…)`).
    pub warnings: Vec<String>,
}

/// One LOADED asset that paints nothing — the decode succeeded and
/// every pixel came out fully transparent. Observability only, NEVER a failure:
/// the pixels shipped, and routing must not read this (same contract as
/// [`SvgProbeWarning`], whose emit site this one sits beside).
#[derive(Debug, Clone, PartialEq)]
pub struct BlankAsset {
    /// SANITIZED asset reference ([`crate::report::sanitized_asset_ref`]) — the
    /// asset LOADED, so like `SvgProbeWarning::url` this is telemetry and the
    /// signed-URL credentials are stripped.
    pub url: String,
    /// Node ids that referenced this source (free from `PrefetchRequest::node_id`).
    pub nodes: Vec<String>,
    /// Always 0 today (that is what makes it blank); carried so the report can
    /// distinguish a true zero from a future near-zero threshold.
    pub coverage_permille: u16,
    /// Low-cardinality cause (`PreparedImage::blank_cause`).
    pub cause: String,
}

/// Prefetch outcome summary returned alongside the store.
#[derive(Debug, Clone, Default)]
pub struct PrefetchReport {
    /// Unique fetch jobs (deduped sources / shader resource keys).
    pub referenced: usize,
    pub loaded: usize,
    pub failures: Vec<AssetFailure>,
    /// Per-asset SVG byte-probe hazard warnings (loaded assets only —
    /// observability, not failures; see [`SvgProbeWarning`]).
    pub svg_warnings: Vec<SvgProbeWarning>,
    /// Assets that decoded to fully transparent pixels (loaded assets only —
    /// observability, not failures; see [`BlankAsset`]).
    pub blank: Vec<BlankAsset>,
    /// Aggregate byte-retrieval work (file reads / data-URI decode), summed
    /// across the concurrent prefetch. Telemetry only.
    pub fetch_ms: u64,
    /// Aggregate image/SVG decode work, summed the same way. Telemetry only.
    pub decode_ms: u64,
    /// Number of assets whose own fetch+decode exceeded [`SLOW_ASSET_WARN_MS`].
    pub slow_asset_count: u32,
    /// The single slowest asset's fetch+decode wall (ms). Telemetry only.
    pub slowest_asset_ms: u64,
}

#[derive(Clone)]
pub struct PrefetchConfig {
    /// Worker threads reading + decoding sources in parallel.
    pub concurrency: usize,
    /// A single asset whose read+decode meets/exceeds this is WARN-logged and
    /// counted in `PrefetchReport::slow_asset_count`. Telemetry only (default
    /// [`SLOW_ASSET_WARN_MS`]); tests pin it to 0 to exercise the path.
    pub slow_warn_ms: u64,
    /// Vendored asset directory (`--assets-dir`, PLAN §2).
    pub assets_dir: Option<PathBuf>,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        PrefetchConfig {
            concurrency: PREFETCH_CONCURRENCY,
            slow_warn_ms: SLOW_ASSET_WARN_MS,
            assets_dir: None,
        }
    }
}

impl std::fmt::Debug for PrefetchConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefetchConfig")
            .field("concurrency", &self.concurrency)
            .field("assets_dir", &self.assets_dir)
            .finish()
    }
}

enum Outcome {
    /// Decoded pattern image + byte-probe warnings (SVG only; see
    /// [`SvgProbeWarning`] — empty for raster and for clean SVGs).
    Loaded(PreparedImage, Vec<String>),
    /// M4e: a settled shader-image request (stored under its resource key).
    LoadedShaderImage(moda_shaders::runtime::image_processing::PreparedShaderImage),
    /// Committed failure: the flattened error chain plus the typed browser-
    /// undecodable sniff kind extracted from the LIVE error (see
    /// [`AssetFailure::browser_undecodable`]).
    Failed {
        reason: String,
        browser_undecodable: Option<&'static str>,
    },
}

/// One deduped read+decode job.
enum Job {
    /// Pattern texture: store key = src; min-raster max-merged like
    /// `requestedMinRasterSize` (`WebGPUTextureManager.ts:263-271`).
    Pattern {
        src: String,
        min: Option<(u32, u32)>,
    },
    /// M4e shader image: store key = `image_resource_key(src, processing)`
    /// (dedup identical to `ShaderImageResourceManager`'s entry map).
    Shader {
        src: String,
        key: String,
        processing: moda_shaders::runtime::image_processing::Processing,
    },
}

/// Per-source cap for raw media bytes ([`fetch_media_bytes_blocking`]) —
/// bounds the RETAINED memory per unique video source. Generous for any
/// product video fill; an oversized source errors instead of exhausting the
/// export worker.
pub const MAX_MEDIA_SOURCE_BYTES: usize = 256 * 1024 * 1024;

/// Read one source's RAW bytes — data URI / vendored file — with no image
/// decode. The animated-export video lane uses this for `fillVideoSrc` (MP4
/// bytes feed `moda_video::decode`, not the pattern pipeline).
pub fn fetch_media_bytes_blocking(src: &str, config: &PrefetchConfig) -> Result<Vec<u8>> {
    let (bytes, _svg_hint) = fetch_bytes(config, src)?;
    anyhow::ensure!(!bytes.is_empty(), "empty media body for {}", truncate(src));
    anyhow::ensure!(
        bytes.len() <= MAX_MEDIA_SOURCE_BYTES,
        "media body for {} is {} bytes (cap {MAX_MEDIA_SOURCE_BYTES})",
        truncate(src),
        bytes.len()
    );
    Ok(bytes)
}

/// Resolve + decode every request (deduped; pattern requests by src,
/// shader-image requests by resource key). Returns the settled store plus a
/// [`PrefetchReport`] with full failure details — the store keeps placeholder
/// semantics for the pixels; the report is the scream channel.
pub fn prefetch_blocking(
    requests: Vec<PrefetchRequest>,
    config: &PrefetchConfig,
) -> (AssetStore, PrefetchReport) {
    let mut merged: HashMap<String, Option<(u32, u32)>> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut shader_jobs: Vec<Job> = Vec::new();
    let mut shader_seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    // store key → (original src, referencing node ids) for the report.
    let mut nodes_by_key: HashMap<String, Vec<String>> = HashMap::new();
    let mut src_by_key: HashMap<String, String> = HashMap::new();
    let mut note_node = |key: &str, src: &str, node: &Option<String>| {
        src_by_key
            .entry(key.to_string())
            .or_insert_with(|| src.to_string());
        let nodes = nodes_by_key.entry(key.to_string()).or_default();
        if let Some(n) = node {
            if !nodes.contains(n) {
                nodes.push(n.clone());
            }
        }
    };
    for r in requests {
        if let Some(processing) = r.shader_processing {
            let key = moda_shaders::runtime::image_resource_key(&r.src, &processing);
            note_node(&key, &r.src, &r.node_id);
            if shader_seen.insert(key.clone()) {
                shader_jobs.push(Job::Shader {
                    src: r.src,
                    key,
                    processing,
                });
            }
            continue;
        }
        note_node(&r.src, &r.src, &r.node_id);
        let entry = merged.entry(r.src.clone()).or_insert_with(|| {
            order.push(r.src.clone());
            None
        });
        *entry = match (*entry, r.min_raster_size) {
            (Some((aw, ah)), Some((bw, bh))) => Some((aw.max(bw), ah.max(bh))),
            (a, b) => a.or(b),
        };
    }
    if order.is_empty() && shader_jobs.is_empty() {
        return (AssetStore::default(), PrefetchReport::default());
    }

    let jobs: Vec<Job> = order
        .into_iter()
        .map(|src| {
            let min = merged[&src];
            Job::Pattern { src, min }
        })
        .chain(shader_jobs)
        .collect();
    let results = run_jobs(&jobs, config);

    let mut store = AssetStore::default();
    let mut report = PrefetchReport {
        referenced: merged.len() + shader_seen.len(),
        ..PrefetchReport::default()
    };
    for (key, is_shader, outcome, span) in results {
        // Aggregate the read/decode split + slow-asset telemetry.
        let total_ms = span.fetch_ms + span.decode_ms;
        report.fetch_ms += span.fetch_ms;
        report.decode_ms += span.decode_ms;
        if total_ms >= config.slow_warn_ms {
            report.slow_asset_count += 1;
        }
        report.slowest_asset_ms = report.slowest_asset_ms.max(total_ms);
        let url = src_by_key.get(&key).cloned().unwrap_or_else(|| key.clone());
        let nodes = nodes_by_key.get(&key).cloned().unwrap_or_default();
        match outcome {
            Outcome::Loaded(img, probe_warnings) => {
                report.loaded += 1;
                if !probe_warnings.is_empty() {
                    report.svg_warnings.push(SvgProbeWarning {
                        url: crate::report::sanitized_asset_ref(&url),
                        warnings: probe_warnings,
                    });
                }
                // A successful decode that paints NOTHING. Reported
                // beside `svg_warnings` and under the same contract — the asset
                // loaded, the placeholder semantics below do not apply, and this
                // must never reach a failure count.
                if img.coverage_permille == 0 {
                    report.blank.push(BlankAsset {
                        url: crate::report::sanitized_asset_ref(&url),
                        nodes,
                        coverage_permille: img.coverage_permille,
                        cause: img.blank_cause.to_string(),
                    });
                }
                store.insert(key, AssetEntry::Loaded(Arc::new(img)))
            }
            Outcome::LoadedShaderImage(img) => {
                report.loaded += 1;
                store.insert_shader_image(
                    key,
                    crate::shader_image::ShaderImageEntry::Loaded(Arc::new(img)),
                )
            }
            Outcome::Failed {
                reason,
                browser_undecodable,
            } => {
                eprintln!(
                    "moda-assets: failed loading '{}' — baking {} placeholder: {reason}",
                    truncate(&key),
                    if is_shader {
                        "shader-image fallback"
                    } else {
                        "failed"
                    }
                );
                report.failures.push(AssetFailure {
                    url,
                    shader_image: is_shader,
                    reason,
                    browser_undecodable,
                    nodes,
                });
                if is_shader {
                    store.insert_shader_image(key, crate::shader_image::ShaderImageEntry::Fallback)
                } else {
                    store.insert(key, AssetEntry::Failed)
                }
            }
        }
    }
    (store, report)
}

/// Resolve every job on up to `config.concurrency` scoped worker threads,
/// returning `(store_key, is_shader, outcome, span)` per job, ordered by store
/// key so the report is deterministic.
fn run_jobs(jobs: &[Job], config: &PrefetchConfig) -> Vec<(String, bool, Outcome, Span)> {
    let next = AtomicUsize::new(0);
    let workers = config.concurrency.max(1).min(jobs.len());
    let mut results: Vec<(String, bool, Outcome, Span)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut out = Vec::new();
                    while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                        out.push(run_job(job, config));
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("prefetch worker panicked"))
            .collect()
    });
    results.sort_by(|a, b| a.0.cmp(&b.0));
    results
}

fn run_job(job: &Job, config: &PrefetchConfig) -> (String, bool, Outcome, Span) {
    let (store_key, is_shader) = match job {
        Job::Pattern { src, .. } => (src.clone(), false),
        Job::Shader { key, .. } => (key.clone(), true),
    };
    let started = Instant::now();
    // A decoder panic on hostile bytes is a per-asset failure, not a crashed
    // render.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load_job(config, job)))
        .unwrap_or_else(|_| Err(anyhow!("decode panicked for {}", truncate(&store_key))));
    let (outcome, span) = match result {
        Ok(done) => done,
        Err(e) => (
            Outcome::Failed {
                reason: format!("{e:#}"),
                browser_undecodable: crate::decode::browser_undecodable_kind(&e),
            },
            Span {
                fetch_ms: started.elapsed().as_millis() as u64,
                decode_ms: 0,
            },
        ),
    };
    let total_ms = span.fetch_ms + span.decode_ms;
    if total_ms >= config.slow_warn_ms {
        eprintln!(
            "moda-assets: slow asset '{}' — {} ms (read {} ms, decode {} ms)",
            truncate(&store_key),
            total_ms,
            span.fetch_ms,
            span.decode_ms,
        );
    }
    (store_key, is_shader, outcome, span)
}

fn truncate(s: &str) -> String {
    // Mirror the production log hygiene (`src.slice(0, 80)`).
    s.chars().take(80).collect()
}

fn load_job(config: &PrefetchConfig, job: &Job) -> Result<(Outcome, Span)> {
    match job {
        Job::Pattern { src, min } => {
            let (img, span, probe_warnings) = load_one(config, src, *min)?;
            Ok((Outcome::Loaded(img, probe_warnings), span))
        }
        Job::Shader {
            src, processing, ..
        } => {
            // The production `!source.src` arm never fetches.
            let fetch_started = Instant::now();
            let (bytes, svg_hint) = if src.is_empty() {
                (Vec::new(), false)
            } else {
                let (bytes, svg_hint) = fetch_bytes(config, src)?;
                if bytes.is_empty() {
                    bail!("empty asset body for {}", truncate(src));
                }
                (bytes, svg_hint)
            };
            let fetch_ms = fetch_started.elapsed().as_millis() as u64;
            let decode_started = Instant::now();
            let img = crate::shader_image::load_shader_image(
                src,
                &bytes,
                svg_hint,
                processing,
            )?;
            Ok((
                Outcome::LoadedShaderImage(img),
                Span {
                    fetch_ms,
                    decode_ms: decode_started.elapsed().as_millis() as u64,
                },
            ))
        }
    }
}

/// Resolve a source to raw bytes (+ SVG hint) — data URI / vendored file. An
/// http(s) source is refused: Desktop Moda never touches the network.
fn fetch_bytes(config: &PrefetchConfig, src: &str) -> Result<(Vec<u8>, bool)> {
    match classify(src) {
        AssetSource::DataUri => {
            let (mime, bytes) = decode_data_uri(src)?;
            Ok((bytes, mime.contains("svg")))
        }
        AssetSource::Http => bail!("{REMOTE_URL_NOT_ALLOWED}: {src}"),
        AssetSource::File => {
            let path = resolve_file(src, config.assets_dir.as_deref())?;
            Ok((
                std::fs::read(&path).with_context(|| format!("read {}", path.display()))?,
                is_likely_svg(src),
            ))
        }
    }
}

fn load_one(
    config: &PrefetchConfig,
    src: &str,
    min_raster: Option<(u32, u32)>,
) -> Result<(PreparedImage, Span, Vec<String>)> {
    // A natural key names its source and how it is prepared (`crate::natural_key`).
    let (src, natural) = match crate::natural_source(src) {
        Some(source) => (source, true),
        None => (src, false),
    };
    let fetch_started = Instant::now();
    let (bytes, svg_hint) = fetch_bytes(config, src)?;
    let fetch_ms = fetch_started.elapsed().as_millis() as u64;

    if bytes.is_empty() {
        bail!("empty asset body for {}", truncate(src));
    }

    // SVG detection: src hint (extension / data mime) or content sniff —
    // the production path also detects SVG via blob Content-Type for
    // extension-less proxy URLs.
    let is_svg = svg_hint || sniff_svg_bytes(&bytes);
    if natural && is_svg {
        bail!("{} is an SVG: it has no decoded texels to keep at a natural size", truncate(src));
    }
    let decode_started = Instant::now();
    let (img, probe_warnings) = if is_svg {
        // SVG <text> lays out with the text engine's fonts, resolved here: building that engine is the
        // first SVG's cost, not every prefetch's (most decode no SVG).
        let fontdb = crate::svg::engine_fontdb();
        let img = prepare_svg(&bytes, min_raster, &fontdb)?;
        // Byte-accurate hazard telemetry (probe.rs) alongside the decoded
        // pixels — prefetch lane only (the wasm editor's `decode_pattern_bytes`
        // hot path stays probe-free). The probe parses independently of
        // `prepare_svg` and cannot perturb the decode output; a parse FAILURE
        // surfaces as the decode error above, so successful decodes only ever
        // carry fidelity warnings (`svg:text` / `svg:embedded-image` /
        // `svg:sanitized-content`).
        let warnings = crate::probe::probe_svg(&bytes, &fontdb).warnings;
        (img, warnings)
    } else {
        (decoded_raster(&bytes, natural)?, Vec::new())
    };
    Ok((
        img,
        Span {
            fetch_ms,
            decode_ms: decode_started.elapsed().as_millis() as u64,
        },
        probe_warnings,
    ))
}

/// Bytes of decoded pixels [`decoded_raster`] keeps for reuse, across every prefetch of the process.
const DECODED_RASTER_BUDGET: usize = 128 << 20;

/// Recently decoded rasters, least recently used first, by the BLAKE3 hash of their source bytes, and their pixel
/// bytes in total.
static DECODED_RASTERS: std::sync::Mutex<(Vec<([u8; 32], PreparedImage)>, usize)> =
    std::sync::Mutex::new((Vec::new(), 0));

/// [`prepare_raster`] of `bytes` (`natural`: [`crate::decode::prepare_raster_natural`]), decoded once per process
/// while it stays among the recent ones. A renderer that lives across documents and edits (`serve`) prefetches
/// again for each new scene and scale; the pixels of bytes it has seen are reused. The key is the bytes' hash
/// and the preparation, so other bytes (an edited file at the same path) decode afresh.
fn decoded_raster(bytes: &[u8], natural: bool) -> Result<PreparedImage> {
    let key = {
        let mut hash = blake3::Hasher::new();
        hash.update(bytes);
        if natural {
            hash.update(b"natural");
        }
        *hash.finalize().as_bytes()
    };
    let cache = || DECODED_RASTERS.lock().unwrap_or_else(|e| e.into_inner());
    {
        let mut c = cache();
        if let Some(at) = c.0.iter().position(|(k, _)| *k == key) {
            let hit = c.0.remove(at);
            c.0.push(hit.clone());
            return Ok(hit.1);
        }
    }
    let img = if natural { crate::decode::prepare_raster_natural(bytes)? } else { prepare_raster(bytes)? };
    let size = img.rgba_premul.len();
    if size <= DECODED_RASTER_BUDGET {
        let mut c = cache();
        if !c.0.iter().any(|(k, _)| *k == key) {
            c.0.push((key, img.clone()));
            c.1 += size;
            while c.1 > DECODED_RASTER_BUDGET {
                let (_, old) = c.0.remove(0);
                c.1 -= old.rgba_premul.len();
            }
        }
    }
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PNG_1PX: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    #[test]
    fn data_uri_prefetch_loads() {
        let _report = crate::report::tests::serial_globals();
        let (store, report) = prefetch_blocking(
            vec![PrefetchRequest {
                src: PNG_1PX.into(),
                min_raster_size: None,
                shader_processing: None,
                node_id: None,
            }],
            &PrefetchConfig::default(),
        );
        assert_eq!(store.fill_state(PNG_1PX), crate::ImageFillState::Loaded);
        assert_eq!((report.referenced, report.loaded), (1, 1));
        assert!(report.failures.is_empty());
        // 1x1 → small-image policy → 512x512
        assert_eq!(store.dimensions(PNG_1PX), Some((512, 512)));
        assert_eq!(store.scale_factor(PNG_1PX), Some((512.0, 512.0)));
    }

    #[test]
    fn svg_text_decode_probes_warning_alongside_loaded_image() {
        let _report = crate::report::tests::serial_globals();
        // An SVG with <text> and an unseeded fontdb: the decode SUCCEEDS (the
        // pixels ship, fontless — browser-parity placeholder semantics do not
        // apply) and the byte probe attaches the `svg:text` fidelity warning
        // to the report. Observability only: loaded, zero failures.
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><rect width="20" height="20" fill="#336699"/><text x="2" y="12">hi</text></svg>"##;
        let src = format!("data:image/svg+xml;utf8,{svg}");
        let (store, report) = prefetch_blocking(
            vec![PrefetchRequest {
                src: src.clone(),
                min_raster_size: None,
                shader_processing: None,
                node_id: Some("n1".into()),
            }],
            &PrefetchConfig::default(),
        );
        assert_eq!(store.fill_state(&src), crate::ImageFillState::Loaded);
        assert!(report.failures.is_empty());
        assert_eq!(report.loaded, 1);
        assert_eq!(report.svg_warnings.len(), 1);
        // The reported ref is SANITIZED: a data: URI collapses to a bounded
        // scheme+size stub (never the full — potentially huge — payload).
        assert_eq!(
            report.svg_warnings[0].url,
            format!("data:image/svg+xml ({} bytes)", src.len())
        );
        assert!(report.svg_warnings[0]
            .warnings
            .iter()
            .any(|w| w.starts_with("svg:text")));

        // The probe must not perturb the decode: pixels are byte-identical to
        // a direct `prepare_svg` of the same bytes (probe-free path).
        let direct = crate::svg::prepare_svg(
            svg.as_bytes(),
            None,
            &Arc::new(resvg::usvg::fontdb::Database::new()),
        )
        .unwrap();
        let via_prefetch = store.image(&src).expect("loaded image");
        assert_eq!(
            (via_prefetch.width, via_prefetch.height),
            (direct.width, direct.height)
        );
        assert_eq!(
            via_prefetch.rgba_premul.as_slice(),
            direct.rgba_premul.as_slice()
        );

        // A clean (text-free) SVG carries no probe warnings.
        let clean = r##"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><rect width="20" height="20" fill="#336699"/></svg>"##;
        let clean_src = format!("data:image/svg+xml;utf8,{clean}");
        let (store, report) = prefetch_blocking(
            vec![PrefetchRequest {
                src: clean_src.clone(),
                min_raster_size: None,
                shader_processing: None,
                node_id: None,
            }],
            &PrefetchConfig::default(),
        );
        assert_eq!(store.fill_state(&clean_src), crate::ImageFillState::Loaded);
        assert!(report.svg_warnings.is_empty());

        // Raster assets never probe.
        let (_, report) = prefetch_blocking(
            vec![PrefetchRequest {
                src: PNG_1PX.into(),
                min_raster_size: None,
                shader_processing: None,
                node_id: None,
            }],
            &PrefetchConfig::default(),
        );
        assert!(report.svg_warnings.is_empty());
    }

    /// A text-only SVG with an unseeded fontdb LOADS CLEAN and paints
    /// nothing. The report must say so on the `blank` channel — loaded, zero
    /// failures, node attributed, url sanitized — because on every other channel
    /// this render is indistinguishable from a healthy one.
    #[test]
    fn blank_svg_decode_reports_on_the_blank_channel_without_failing() {
        let _report = crate::report::tests::serial_globals();
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20"><text x="2" y="14">Acme</text></svg>"##;
        let src = format!("data:image/svg+xml;utf8,{svg}");
        let (store, report) = prefetch_blocking(
            vec![PrefetchRequest {
                src: src.clone(),
                min_raster_size: None,
                shader_processing: None,
                node_id: Some("n7".into()),
            }],
            &PrefetchConfig::default(),
        );
        // The asset LOADED — placeholder semantics do not apply.
        assert_eq!(store.fill_state(&src), crate::ImageFillState::Loaded);
        assert_eq!(report.loaded, 1);
        assert!(report.failures.is_empty());

        assert_eq!(report.blank.len(), 1);
        let b = &report.blank[0];
        assert_eq!(b.coverage_permille, 0);
        assert_eq!(b.cause, crate::decode::BLANK_CAUSE_SVG_TEXT_NO_FONTDB);
        assert_eq!(b.nodes, vec!["n7".to_string()]);
        // Sanitized like `SvgProbeWarning::url` — the asset loaded, so this is
        // telemetry and must not carry a payload or credentials.
        assert_eq!(b.url, format!("data:image/svg+xml ({} bytes)", src.len()));

        // A PAINTING asset reports nothing on this channel — the regression
        // sentinel against a false-positive blank.
        let (_, clean) = prefetch_blocking(
            vec![PrefetchRequest {
                src: PNG_1PX.into(),
                min_raster_size: None,
                shader_processing: None,
                node_id: None,
            }],
            &PrefetchConfig::default(),
        );
        assert!(clean.blank.is_empty());
    }

    #[test]
    fn bad_data_uri_fails() {
        let _report = crate::report::tests::serial_globals();
        let src = "data:image/png;base64,notbase64!!!";
        let (store, report) = prefetch_blocking(
            vec![PrefetchRequest {
                src: src.into(),
                min_raster_size: None,
                shader_processing: None,
                node_id: Some("node-a".into()),
            }],
            &PrefetchConfig::default(),
        );
        assert_eq!(store.fill_state(src), crate::ImageFillState::Failed);
        // The report screams: full URL, reason, node attribution.
        assert_eq!(report.failures.len(), 1);
        let f = &report.failures[0];
        assert_eq!(f.url, src);
        assert_eq!(f.nodes, vec!["node-a".to_string()]);
        // Not a decode sniff — a bad body keeps its decline/retry path.
        assert_eq!(f.browser_undecodable, None);
        assert_eq!(report.loaded, 0);
    }

    /// The typed browser-undecodable sniff crosses from the LIVE decode error
    /// into the `AssetFailure` the report carries (the fetch lane flattens the
    /// chain to a string right after, so this is the only place it can be read
    /// typed). EMF bytes are the smallest positively-sniffed fixture.
    #[test]
    fn browser_undecodable_sniff_reaches_the_asset_failure() {
        let _report = crate::report::tests::serial_globals();
        use base64::Engine as _;
        let mut emf = vec![0u8; 88];
        emf[0] = 0x01;
        emf[40..44].copy_from_slice(b" EMF");
        let src = format!(
            "data:image/x-emf;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&emf)
        );
        let (store, report) = prefetch_blocking(
            vec![PrefetchRequest {
                src: src.clone(),
                min_raster_size: None,
                shader_processing: None,
                node_id: Some("node-a".into()),
            }],
            &PrefetchConfig::default(),
        );
        assert_eq!(store.fill_state(&src), crate::ImageFillState::Failed);
        assert_eq!(report.failures.len(), 1);
        let f = &report.failures[0];
        assert_eq!(f.browser_undecodable, Some("emf"));
        assert!(
            f.reason
                .contains("image decoder: undecodable format (sniff: emf)"),
            "reason grammar unchanged: {}",
            f.reason
        );
    }

    /// Documents render offline: an http(s) source is a committed
    /// per-asset failure naming the URL, never a network attempt: a local
    /// listener stands in for the remote host and must see no connection.
    #[test]
    fn remote_url_fails_without_a_network_attempt() {
        let _report = crate::report::tests::serial_globals();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        for src in [
            format!("http://{addr}/x.png"),
            format!("https://{addr}/x.svg"),
        ] {
            let (store, report) = prefetch_blocking(
                vec![PrefetchRequest {
                    src: src.clone(),
                    min_raster_size: None,
                    shader_processing: None,
                    node_id: Some("n1".into()),
                }],
                &PrefetchConfig::default(),
            );
            assert_eq!(store.fill_state(&src), crate::ImageFillState::Failed);
            assert_eq!(report.failures.len(), 1);
            let f = &report.failures[0];
            assert_eq!(f.url, src);
            assert_eq!(f.reason, format!("{REMOTE_URL_NOT_ALLOWED}: {src}"));
            assert_eq!(f.nodes, vec!["n1".to_string()]);
            assert!(fetch_media_bytes_blocking(&src, &PrefetchConfig::default())
                .unwrap_err()
                .to_string()
                .starts_with(REMOTE_URL_NOT_ALLOWED));
        }
        assert!(
            listener.accept().is_err(),
            "the prefetch must not connect to a remote source"
        );
    }

    /// Relative sources resolve under `--assets-dir`; absolute paths resolve
    /// as-is. The report carries the read/decode split + slow-asset telemetry
    /// (`slow_warn_ms: 0` makes every asset count as slow, so the fields are
    /// deterministic regardless of machine speed).
    #[test]
    fn local_files_load_with_timing() {
        let _report = crate::report::tests::serial_globals();
        use base64::Engine as _;
        let png = base64::engine::general_purpose::STANDARD
            .decode(PNG_1PX.strip_prefix("data:image/png;base64,").unwrap())
            .unwrap();
        let dir = std::env::temp_dir().join(format!("moda-assets-fetch-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("img")).unwrap();
        std::fs::write(dir.join("img/a.png"), &png).unwrap();
        let absolute = dir.join("img/a.png").to_string_lossy().into_owned();
        let config = PrefetchConfig {
            slow_warn_ms: 0,
            assets_dir: Some(dir.clone()),
            ..PrefetchConfig::default()
        };
        let (store, report) = prefetch_blocking(
            ["img/a.png".to_string(), absolute.clone()]
                .into_iter()
                .map(|src| PrefetchRequest {
                    src,
                    min_raster_size: None,
                    shader_processing: None,
                    node_id: None,
                })
                .collect(),
            &config,
        );
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(store.fill_state("img/a.png"), crate::ImageFillState::Loaded);
        assert_eq!(store.fill_state(&absolute), crate::ImageFillState::Loaded);
        assert_eq!((report.referenced, report.loaded), (2, 2));
        assert_eq!(report.slow_asset_count, 2);
        assert!(report.slowest_asset_ms <= report.fetch_ms + report.decode_ms);
    }

    #[test]
    fn min_raster_requests_max_merge() {
        let _report = crate::report::tests::serial_globals();
        // Two requests for the same SVG src — the larger min wins per axis.
        let svg = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4MCIgaGVpZ2h0PSI4MCI+PC9zdmc+";
        let (store, report) = prefetch_blocking(
            vec![
                PrefetchRequest {
                    src: svg.into(),
                    min_raster_size: Some((1100, 200)),
                    shader_processing: None,
                    node_id: None,
                },
                PrefetchRequest {
                    src: svg.into(),
                    min_raster_size: Some((200, 1200)),
                    shader_processing: None,
                    node_id: None,
                },
            ],
            &PrefetchConfig::default(),
        );
        // dedup: two requests, one job.
        assert_eq!((report.referenced, report.loaded), (1, 1));
        // policy floor is 1024; merged request (1100, 1200) raises both axes.
        assert_eq!(store.dimensions(svg), Some((1100, 1200)));
    }

    /// A PNG of one colour, `w`×`h`.
    fn solid_png(rgba: [u8; 4], w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbaImage::from_pixel(w, h, image::Rgba(rgba));
        let mut out = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .unwrap();
        out
    }

    fn file_prefetch(dir: &std::path::Path, src: &str) -> AssetStore {
        let config = PrefetchConfig {
            assets_dir: Some(dir.to_path_buf()),
            ..PrefetchConfig::default()
        };
        let request = PrefetchRequest {
            src: src.into(),
            min_raster_size: None,
            shader_processing: None,
            node_id: None,
        };
        prefetch_blocking(vec![request], &config).0
    }

    /// The same bytes, under any name and in any later prefetch, are the pixels decoded the first time.
    #[test]
    fn the_same_bytes_decode_once() {
        let _report = crate::report::tests::serial_globals();
        let png = solid_png([12, 200, 77, 255], 600, 520);
        let dir = std::env::temp_dir().join(format!("moda-assets-same-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.png"), &png).unwrap();
        std::fs::write(dir.join("b.png"), &png).unwrap();
        let first = file_prefetch(&dir, "a.png");
        let again = file_prefetch(&dir, "a.png");
        let other_name = file_prefetch(&dir, "b.png");
        std::fs::remove_dir_all(&dir).unwrap();
        let pixels =
            |store: &AssetStore, src: &str| Arc::clone(&store.image(src).unwrap().rgba_premul);
        assert!(Arc::ptr_eq(
            &pixels(&first, "a.png"),
            &pixels(&again, "a.png")
        ));
        assert!(Arc::ptr_eq(
            &pixels(&first, "a.png"),
            &pixels(&other_name, "b.png")
        ));
        assert_eq!(first.dimensions("a.png"), Some((600, 520)));
    }

    /// A file replaced at the same path is decoded again: the next prefetch has the new pixels and size.
    #[test]
    fn an_edited_file_decodes_its_new_bytes() {
        let _report = crate::report::tests::serial_globals();
        let dir = std::env::temp_dir().join(format!("moda-assets-edit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("photo.png"), solid_png([255, 0, 0, 255], 600, 520)).unwrap();
        let before = file_prefetch(&dir, "photo.png");
        std::fs::write(dir.join("photo.png"), solid_png([0, 0, 255, 255], 640, 530)).unwrap();
        let after = file_prefetch(&dir, "photo.png");
        std::fs::remove_dir_all(&dir).unwrap();
        let first_pixel =
            |store: &AssetStore| store.image("photo.png").unwrap().rgba_premul[..4].to_vec();
        assert_eq!(first_pixel(&before), [255, 0, 0, 255]);
        assert_eq!(first_pixel(&after), [0, 0, 255, 255]);
        assert_eq!(after.dimensions("photo.png"), Some((640, 530)));
    }
}
