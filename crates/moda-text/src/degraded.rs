//! degraded.rs — the M4 **Degraded text tier** (DECISION §1.4 / PLAN M4).
//!
//! Fallback-class richtext nodes (non-catalog families incl. the default
//! `sans-serif`, codepoints outside the MSDF atlas coverage — Q-28/Q-42 —
//! and text-as-mask sources) are rendered by the **browser's Canvas2D
//! engine** (full Chromium shaping). This module is the native stand-in:
//! a CPU raster of the node produced by
//!
//! - the SAME shared layout core the Canvas2D engine injects measurement
//!   into (`utils/richtext/layout.ts` → `crate::wrap`,
//!   `positioning.ts` → `crate::positioning`,
//!   `lineMetrics.ts` → `crate::line_metrics`,
//!   `decorations.ts` → `crate::decorations`), and
//! - **cosmic-text 0.19** (`ShapeLine` + swash) as the measurement/shaping/
//!   raster engine — the spike-validated recipe
//!   (`research/spikes/cosmic-text.md` §3–4, §9: shape once, wrap with OUR
//!   ported greedy tokenizer, place lines with the shared half-leading
//!   formula, rasterize via swash).
//!
//! The parity target is `utils/canvas2dRichTextRenderer.ts`
//! (`renderRichTextNodeToCanvasWithFixedSize`) + `core/text/textSizing.ts`
//! (`getRichTextCanvasLogicalSize`): texture logical size = intrinsic
//! content size (+6px horizontal texture padding each side), rasterized at
//! `qualityScale = 2`. Output is honestly **Degraded** — the feature
//! manifest reports `expected_ssim_band [0.90, 0.97]` (DECISION §1.6) and
//! pixels are emitted only behind `--best-effort-text` (PLAN §2), except for
//! mask coverage where the browser ALWAYS uses the Canvas2D texture.
//!
//! Deliberate Canvas2D parity behaviors (copy, don't fix):
//! - letterSpacing is added **between graphemes only** (measurement.ts:
//!   197-213) — the opposite of the MSDF trailing-spacing quirk Q-10 — and a
//!   non-zero spacing draws per-grapheme (kerning/ligatures break exactly
//!   like `drawTextWithLetterSpacing`, utils/canvas/text.ts:3-31);
//! - uniform-alpha runs rasterize **opaque** and the average alpha ships
//!   separately (`fill_alpha`) to be applied at the display quad — masks
//!   sample the OPAQUE texture (canvas2dRichTextRenderer.ts:299-321,
//!   `detectMixedAlpha`/`computeAverageAlpha`);
//! - run colors parse with the FORGIVING parser (`parseColorInput`-class;
//!   cf. Q-57), black-ish fallback to the raw string is replaced by the
//!   TS DEFAULT_STYLE color on parse failure;
//! - `text-align: justify` ≡ left, list paragraphs force left (Q-11).
//!
//! Font sources, in fontdb load order: OS fonts (macOS dev/CI — the same
//! families the golden-capture Chromium fell back to: Helvetica, Georgia,
//! Hiragino, Geeza Pro, Apple Color Emoji), then the vendored Noto fallback
//! set under `fixtures/fonts-degraded/` (MANIFEST.json — makes Linux
//! deterministic; cosmic-text's unix fallback lists name Noto families),
//! then the vendored MSDF catalog TTFs (`fixtures/oracle/fonts`, so
//! `Inter` etc. resolve identically to the browser's webfonts).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock, RwLock};

use cosmic_text::fontdb;
use cosmic_text::{
    Attrs, AttrsList, Family, FontSystem, Hinting, ShapeLine, Shaping, Style, SwashCache,
    SwashContent, Weight, Wrap as CosmicWrap,
};
use unicode_segmentation::UnicodeSegmentation;

use crate::color::css_color_to_normalized_rgba;
use crate::line_metrics::{compute_baseline_for_line_with_fallback, FontMetricsProvider};
use crate::measure::TextMeasurer;
use crate::ops::TextNodeSpec;
use crate::parser::parse_rich_text_to_paragraphs;
use crate::positioning::{
    advance_pen_y_for_paragraph_break, compute_bullet_center_x, compute_bullet_metrics,
    compute_line_start_x, compute_list_indent_px, compute_list_indentation, compute_marker_metrics,
    compute_segment_positions_justified, justify_space_px, measure_line_width, JustifiedMeasurer,
};
use crate::types::{default_style, InlineStyle, Line, Paragraph};
use crate::wrap::layout_paragraph_with_policy;

/// `canvas2dRichTextRenderer.ts:107` — `const pixelRatio = 2`.
pub const QUALITY_SCALE: f64 = 2.0;
/// `core/text/textSizing.ts:20` — constant horizontal texture padding.
const HORIZONTAL_TEXTURE_PADDING_PX: f64 = 6.0;

/// The edge a degraded raster's quad is anchored to in its node (`getRichTextAlignmentFromHtml`,
/// `core/text/textSizing.ts:35-40`): best-effort detection; explicit right beats center. `start` / `end` follow the
/// first paragraph's direction (`crate::paragraph_direction`).
pub fn rich_text_alignment_from_html(html: &str) -> &'static str {
    let lower = html.to_lowercase();
    let has = |needle: &str| -> bool {
        // `text-align\s*:\s*<value>` with optional whitespace.
        let mut rest = lower.as_str();
        while let Some(i) = rest.find("text-align") {
            let after = rest[i + "text-align".len()..].trim_start();
            if let Some(after) = after.strip_prefix(':') {
                if after.trim_start().starts_with(needle) {
                    return true;
                }
            }
            rest = &rest[i + "text-align".len()..];
        }
        false
    };
    if has("start") || has("end") {
        if let Some(first) = parse_rich_text_to_paragraphs(html).first() {
            if first.direction.logical_align.is_some() {
                return first.align;
            }
        }
    }
    if has("right") {
        return "right";
    }
    if has("center") || lower.contains("<center>") || lower.contains("<center ") {
        return "center";
    }
    "left"
}

/// Where a degraded raster of `logical_w` × `logical_h` sits in its node: the texture's top-left in node space
/// (`richtext/transforms.ts:77-117`), from the quad's anchor ([`rich_text_alignment_from_html`]) and `verticalAlign`.
pub fn raster_offset(spec: &TextNodeSpec, logical_w: f64, logical_h: f64) -> (f64, f64) {
    let width_difference = spec.width.unwrap_or(400.0) - logical_w;
    let height_difference = spec.height.unwrap_or(200.0) - logical_h;
    let x = match rich_text_alignment_from_html(&spec.html) {
        "center" => width_difference / 2.0,
        "right" => width_difference,
        _ => 0.0,
    };
    let y = match spec.vertical_align.as_deref() {
        Some("middle") => height_difference / 2.0,
        Some("bottom") => height_difference,
        _ => 0.0,
    };
    (x, y)
}

/// Where `raster`'s texture goes in its node: [`raster_offset`] of the lines' box, moved up by the ink drawn above
/// them ([`DegradedRaster::pad_top`]).
pub fn raster_quad_offset(spec: &TextNodeSpec, raster: &DegradedRaster) -> (f64, f64) {
    let lines_h = raster.logical_h - raster.pad_top - raster.pad_bottom;
    let (x, y) = raster_offset(spec, raster.logical_w, lines_h);
    (x, y - raster.pad_top)
}

/// A degraded raster's horizontal geometry: the texture's logical width, the width lines wrap at, and the box lines
/// align in (`box_x`, `box_w`, texture space).
struct RasterGeometry {
    width: f64,
    height: f64,
    wrap_width: f64,
    box_x: f64,
    box_w: f64,
}
/// `utils/richtext/alpha.ts:11` — MIXED_ALPHA_TOLERANCE. `pub(crate)`: the
/// vector mask raster (`crate::vector::mask_raster`) mirrors the same
/// `detectMixedAlpha` rule (Q-69) over its glyph ops.
pub(crate) const MIXED_ALPHA_TOLERANCE: f64 = 0.025;

/// One rasterized Degraded node — the native analogue of the Canvas2D
/// standalone texture (`WebGPURichTextSystem.getStandaloneTextureForMask`).
/// `rgba` is straight (non-premultiplied) RGBA8, tightly packed.
#[derive(Debug, Clone, PartialEq)]
pub struct DegradedRaster {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// `logicalSize` — texture px / qualityScale (canvas CSS px).
    pub logical_w: f64,
    pub logical_h: f64,
    pub quality_scale: f64,
    /// Logical px of ink above the text's top and below its bottom (both inside `logical_h`): a mark stacked over
    /// a tight line-height (a reph, a Thai tone mark over an upper vowel, Arabic harakat) draws past the lines,
    /// and the texture keeps it rather than cutting it off. [`raster_quad_offset`] places the texture by them.
    pub pad_top: f64,
    pub pad_bottom: f64,
    /// `__nfRichTextFillAlpha` — uniform run alpha applied at the DISPLAY
    /// quad only (`transforms.ts:119-131`); 1.0 when alphas are mixed (the
    /// raster then carries per-run alpha itself). Masks ignore this — they
    /// sample the opaque texture (copy the bug).
    pub fill_alpha: f32,
}

// ---------------------------------------------------------------------------
// Engine (process-wide font system + caches)
// ---------------------------------------------------------------------------

pub struct DegradedEngine {
    inner: Mutex<EngineInner>,
    id: u64,
    /// What the per-canvas MANIFEST declared and what actually loaded, so a caller can
    /// check its required keys against THIS font system too — see
    /// [`DegradedEngine::unresolved_catalog_keys`]. `catalog_all` is every key the
    /// manifest listed, `catalog_with_ttf` those carrying a `ttfPath`, `catalog_loaded`
    /// the subset whose bytes read. All three empty means no manifest was read.
    catalog_all: std::collections::HashSet<String>,
    catalog_with_ttf: std::collections::HashSet<String>,
    catalog_loaded: std::collections::HashSet<String>,
    /// Heap bytes of the catalog TTFs this engine loaded.
    ///
    /// `load_catalog_fonts` does `std::fs::read` into `Source::Binary(Arc::new(..))`,
    /// so every catalog face a canvas resolved is held in full, per engine — NOT
    /// memory-mapped. A cache that holds engines therefore has to see these bytes or
    /// its byte budget is fiction; `moda-py`'s host cache adds this to its own
    /// accounting.
    catalog_bytes: u64,
}

struct EngineInner {
    font_system: FontSystem,
    swash: SwashCache,
    /// (font-string key, text) → width at font-size 1 (advances are linear
    /// in font size on the unhinted path).
    width_cache: HashMap<(String, String), f64>,
    /// font-string key → (ascent, descent) at font-size 1.
    /// The font box per em, `None` when the font had none (the 0.82 / 0.18 fallback).
    metrics_cache: HashMap<String, Option<(f64, f64)>>,
    /// Lowercased catalog family → the face's actual name-table family, for
    /// catalog TTFs whose internal name differs from the catalog family
    /// (opsz-bracketed releases: the "Inter" catalog entries ship
    /// `Inter_24pt-*.ttf` whose name IDs 1/16 read "Inter 24pt"). The browser
    /// resolves these via `@font-face { font-family: Inter }` for the same
    /// file; cosmic-text matches fontdb name-table families, so without the
    /// alias it silently substitutes a fallback face (Noto Sans) with
    /// different metrics — the probe-proven (+6,+6)px mask:text-alpha
    /// misplacement. Built from MANIFEST.json by [`load_catalog_fonts`].
    family_aliases: HashMap<String, String>,
    /// Emoji cluster → its vector-lane image cell (`emoji_cell_rgba`), misses included.
    emoji_cells: HashMap<String, Option<std::sync::Arc<crate::host::RgbaImage>>>,
    /// `EmojiEffect::key` → its effect copy (`emoji_effect_rgba`).
    emoji_effects: HashMap<String, std::sync::Arc<crate::host::RgbaImage>>,
    /// vork: while [`DegradedEngine::glyph_outlines`] runs the draw walk, every glyph it draws.
    recording: Option<GlyphRecording>,
}

/// vork: the glyphs one draw walk drew ([`DegradedEngine::glyph_outlines`]): each glyph's face, size and pen (logical
/// px of the raster before its ink margins are trimmed), and the y of the lines' top in that space.
#[derive(Default)]
struct GlyphRecording {
    origin_y: f64,
    glyphs: Vec<RecordedGlyph>,
}

struct RecordedGlyph {
    font_id: fontdb::ID,
    glyph_id: u16,
    font_size: f32,
    font_weight: fontdb::Weight,
    flags: cosmic_text::CacheKeyFlags,
    x: f64,
    y: f64,
}

/// vork: one glyph's outline ([`DegradedEngine::glyph_outlines`]): its pen (the glyph's origin on its baseline) and
/// its segments, node px, y down.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphOutline {
    pub pen: [f32; 2],
    pub segments: Vec<OutlineSegment>,
}

/// vork: one segment of a glyph outline ([`DegradedEngine::glyph_outlines`]), node px, y down.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutlineSegment {
    MoveTo([f32; 2]),
    LineTo([f32; 2]),
    QuadTo([f32; 2], [f32; 2]),
    CurveTo([f32; 2], [f32; 2], [f32; 2]),
    Close,
}

/// Source of [`DegradedEngine::id`].
static NEXT_ENGINE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Texels per em of an emoji image cell ([`DegradedEngine::emoji_cell_rgba`]): the
/// Noto SVG set's 128 px, near the colour fonts' largest strikes (Noto's 109 ppem, Apple's
/// 160 px sbix).
pub const EMOJI_CELL_TEXELS: u32 = 128;

static ENGINE:OnceLock<std::sync::RwLock<Option<std::sync::Arc<DegradedEngine>>>> =
    OnceLock::new();

fn engine_slot() -> &'static std::sync::RwLock<Option<std::sync::Arc<DegradedEngine>>> {
    ENGINE.get_or_init(|| std::sync::RwLock::new(None))
}

/// Programmatic fonts-dir overrides, consulted BEFORE the env fallbacks in
/// [`engine`]. Hosts that are already multithreaded (e.g. the iOS FFI, where
/// CFNetwork threads call `getenv` concurrently) MUST use these instead of
/// mutating `MODA_FONTS_DIR` / `MODA_DEGRADED_FONTS_DIR` — `setenv` after the
/// process goes multithreaded is the classic setenv/getenv use-after-free UB.
/// Mirrors `moda_engine::text_draw::set_fonts_dir`, which wires the MSDF/vector
/// host the same way; the degraded engine is a SEPARATE font system, so it
/// needs its own override or it loads with only OS fonts (and on iOS that
/// leaves cosmic-text with no default family → `ShapeLine` panics
/// "no default font found").
static CATALOG_DIR_OVERRIDE: RwLock<Option<PathBuf>> = RwLock::new(None);
static DEGRADED_DIR_OVERRIDE: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Point the degraded engine's catalog fonts dir (the one carrying
/// `MANIFEST.json` with `keys.*.ttfPath`) at `dir` programmatically. Must run
/// BEFORE the first degraded text render (the engine is a lazy `OnceLock`);
/// last write wins until then. Consulted before `MODA_FONTS_DIR`.
pub fn set_catalog_fonts_dir(dir: impl Into<PathBuf>) {
    if let Ok(mut guard) = CATALOG_DIR_OVERRIDE.write() {
        *guard = Some(dir.into());
    }
}

/// Point the degraded engine's Noto-fallback fonts dir at `dir`
/// programmatically. Same lifetime/precedence rules as
/// [`set_catalog_fonts_dir`]; consulted before `MODA_DEGRADED_FONTS_DIR`.
pub fn set_degraded_fallback_fonts_dir(dir: impl Into<PathBuf>) {
    if let Ok(mut guard) = DEGRADED_DIR_OVERRIDE.write() {
        *guard = Some(dir.into());
    }
}

fn walk_up_for(rel: &[&str]) -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        for probe in rel {
            let p = dir.join(probe);
            if p.join("MANIFEST.json").is_file() {
                return Some(p);
            }
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// The Noto-fallback dir, by the precedence [`engine`] documents: programmatic
/// override, then `MODA_DEGRADED_FONTS_DIR`, then a walk up from cwd. Extracted so a
/// per-canvas install ([`build_catalog_engine`]) resolves it the SAME way rather than
/// carrying a second copy of the rules — the fallback set is not per-canvas, only the
/// catalog is.
fn resolve_degraded_fallback_dir() -> Option<PathBuf> {
    DEGRADED_DIR_OVERRIDE
        .read()
        .ok()
        .and_then(|g| g.clone())
        .or_else(|| {
            std::env::var("MODA_DEGRADED_FONTS_DIR")
                .ok()
                .map(PathBuf::from)
        })
        .or_else(|| {
            walk_up_for(&[
                "fixtures/fonts-degraded",
                "native-renderer/fixtures/fonts-degraded",
            ])
        })
}

/// The catalog fonts dir, by the same precedence: programmatic override, then
/// `MODA_FONTS_DIR`, then a walk up from cwd.
fn resolve_catalog_dir() -> Option<PathBuf> {
    CATALOG_DIR_OVERRIDE
        .read()
        .ok()
        .and_then(|g| g.clone())
        .or_else(|| std::env::var("MODA_FONTS_DIR").ok().map(PathBuf::from))
        .or_else(|| {
            walk_up_for(&[
                "fixtures/oracle/fonts",
                "native-renderer/fixtures/oracle/fonts",
            ])
        })
}

/// The process-wide Degraded engine. Discovery mirrors
/// `moda_engine::text_draw::discover_fonts_dir`: programmatic overrides
/// ([`set_degraded_fallback_fonts_dir`], [`set_catalog_fonts_dir`]) first,
/// then env (`MODA_DEGRADED_FONTS_DIR`, `MODA_FONTS_DIR`), else walk up from
/// cwd. The programmatic path is how multithreaded hosts (the iOS FFI) wire
/// fonts without setenv/getenv UB.
/// On wasm32 (no filesystem) discovery finds nothing; the wasm editor
/// swap-installs a VFS-fed engine via [`install_engine`] after every font
/// install, and the empty-db guard in [`DegradedEngine::render_node`] keeps
/// a pre-install call an Err (browser "texture not ready" parity) rather
/// than a cosmic-text "no default font found" panic (= abort on wasm).
pub fn engine() -> Result<std::sync::Arc<DegradedEngine>, String> {
    if let Some(e) = engine_slot().read().expect("degraded engine slot").as_ref() {
        return Ok(e.clone());
    }
    let mut guard = engine_slot().write().expect("degraded engine slot");
    if let Some(e) = guard.as_ref() {
        return Ok(e.clone());
    }
    // Through the SHARED path, so a process holds exactly one copy of the ~46 MB baked
    // catalog however many engines reference it. Passing the baked dir as the
    // per-canvas dir hits `compose_catalog_engine`'s `baked_dir == catalog_dir` branch,
    // which skips the second read — the lazy engine is the baked set and nothing more.
    let degraded_dir = resolve_degraded_fallback_dir();
    let baked = baked_catalog();
    let built = std::sync::Arc::new(match baked {
        Some(b) => compose_catalog_engine(degraded_dir.as_deref(), baked, &b.dir.clone()),
        None => DegradedEngine::load(degraded_dir.as_deref(), None),
    });
    *guard = Some(built.clone());
    Ok(built)
}

/// Swap-install the process-wide Degraded engine.
///
/// Two callers, for the same reason — a host that knows its fonts better than the lazy
/// [`engine`] discovery can:
///
/// * The **wasm editor** has no filesystem, so its font feeder rebuilds the engine from
///   the VFS's catalog TTF bytes ([`DegradedEngine::from_font_data`]) on every
///   `install_fonts`, mirroring the TextHost swap.
/// * The **warm pooled native lane** (`moda-py`) renders many canvases in one process
///   and installs a per-canvas engine via [`build_catalog_engine`] +
///   [`install_engine_arc`], because the lazy
///   discovery resolves ONCE per process and would otherwise pin every later canvas to
///   whichever fonts dir was ambient at the first complex-script render. That was
///   The tier stayed on the baked Latin-complete set for the worker's life,
///   so a customer's Arabic in Cairo came back in a vendored Noto face with the job
///   reporting success.
pub fn install_engine(engine: DegradedEngine) {
    *engine_slot().write().expect("degraded engine slot") = Some(std::sync::Arc::new(engine));
}

/// The engine currently installed, WITHOUT building one when none is (unlike [`engine`]). The wasm
/// editor lane reads this: its engine arrives with the first font install, and building one before
/// that would cache an empty font system.
pub fn installed_engine() -> Option<std::sync::Arc<DegradedEngine>> {
    engine_slot().read().expect("degraded engine slot").clone()
}

/// Install a pre-built engine (the LRU-cached per-canvas case — no rebuild).
pub fn install_engine_arc(engine: std::sync::Arc<DegradedEngine>) {
    *engine_slot().write().expect("degraded engine slot") = Some(engine);
}

/// The ambient BAKED catalog, read once per process and shared by every per-canvas
/// engine.
///
/// Layering the baked set into each engine is what keeps rare-script coverage, but
/// reading it per engine is not affordable: `fixtures/oracle/fonts` (what prod bakes at
/// `/opt/moda/native/fonts`) is 57 TTFs and roughly 40-50 MB, ~30 MB of it the four CJK
/// variable fonts. A fresh `std::fs::read` per cached engine would multiply that by the
/// LRU's occupancy and — now that those bytes are correctly charged to the byte budget
/// — collapse a 256 MB cache to four or five entries, defeating the warm lane it exists
/// to serve.
///
/// So the blobs are read once and handed to each engine as shared `Arc`s via
/// `Source::Binary`, the same sharing `from_font_sources_with_aliases` already does for
/// the VFS lane. One copy in memory, however many engines reference it.
struct BakedCatalog {
    sources: Vec<std::sync::Arc<Vec<u8>>>,
    aliases: HashMap<String, String>,
    dir: PathBuf,
}

fn baked_catalog() -> Option<&'static BakedCatalog> {
    static BAKED: std::sync::OnceLock<Option<BakedCatalog>> = std::sync::OnceLock::new();
    BAKED
        .get_or_init(|| {
            let dir = resolve_catalog_dir()?;
            let raw = crate::host::read_file_string(&dir.join("MANIFEST.json")).ok()?;
            let manifest = serde_json::from_str::<serde_json::Value>(&raw).ok()?;
            let keys = manifest.get("keys")?.as_object()?;

            // Several keys share one ttfPath (multi-weight files) — read each once.
            let mut seen: std::collections::BTreeSet<String> = Default::default();
            let mut sources = Vec::new();
            for (_, entry) in fallback_set_first(keys) {
                let Some(rel) = entry.get("ttfPath").and_then(|p| p.as_str()) else {
                    continue;
                };
                if !seen.insert(rel.to_string()) {
                    continue;
                }
                if let Ok(bytes) = crate::host::read_file_bytes(&dir.join(rel)) {
                    sources.push(std::sync::Arc::new(bytes));
                }
            }
            let aliases = manifest_family_aliases(&manifest, |rel| {
                crate::host::read_file_bytes(&dir.join(rel)).ok()
            });
            Some(BakedCatalog {
                sources,
                aliases,
                dir,
            })
        })
        .as_ref()
}

/// Build a Degraded engine layering `catalog_dir` OVER the shared baked catalog, keeping the
/// Noto fallback set the ambient configuration resolves.
///
/// This is the per-canvas counterpart to the lazy [`engine`] discovery: the fallback dir
/// is a process property, the catalog is not. Callers cache the result — see
/// [`install_engine_arc`] — because the build re-scans system fonts (~54 ms measured on
/// the vendored oracle set), which is cheap next to a CLI process spawn but not free
/// enough to repeat per render.
pub fn build_catalog_engine(catalog_dir: &std::path::Path) -> DegradedEngine {
    // The one-shot CLI can point MODA_FONTS_DIR at this very catalog. It is
    // the per-render snapshot, not a distinct baked fallback: load its keys
    // through the checked per-canvas path instead of hiding them in OnceLock.
    let baked = if resolve_catalog_dir().as_deref() == Some(catalog_dir) {
        None
    } else {
        baked_catalog()
    };
    compose_catalog_engine(
        resolve_degraded_fallback_dir().as_deref(),
        baked,
        catalog_dir,
    )
}

/// The layering itself, with the baked catalog passed IN rather than resolved.
///
/// Split out so the precedence rules can be tested against a synthetic baked set: the
/// real one is a process-wide `OnceLock`, so a test cannot vary it, and the ordering
/// and alias-clearing properties below would otherwise be unguarded.
fn compose_catalog_engine(
    fallback_dir: Option<&std::path::Path>,
    baked: Option<&BakedCatalog>,
    catalog_dir: &std::path::Path,
) -> DegradedEngine {
    compose_catalog_engine_from_snapshot(fallback_dir, baked, catalog_dir, None)
}

/// The degraded engine for `catalog_dir`'s fonts as `manifest` describes them (the snapshot a `TextHost` loaded),
/// with the Noto fallback set. `manifest`'s `"systemFonts": false` keeps the OS fonts out.
pub fn build_catalog_engine_from_snapshot(
    catalog_dir: &std::path::Path,
    manifest: &serde_json::Value,
) -> DegradedEngine {
    // Match the directory-based builder: the current request's catalog must
    // pass through the checked per-canvas layer, even on the one-shot CLI.
    let baked = if resolve_catalog_dir().as_deref() == Some(catalog_dir) {
        None
    } else {
        baked_catalog()
    };
    compose_catalog_engine_from_snapshot(
        resolve_degraded_fallback_dir().as_deref(),
        baked,
        catalog_dir,
        Some(manifest),
    )
}

/// Desktop fork: a fonts dir whose `MANIFEST.json` says `"systemFonts": false`
/// keeps the OS fonts out of the degraded tier, so a native render falls back to
/// exactly the faces the fonts dir ships — the same set the wasm editor has in its
/// VFS (which never sees OS fonts). Without it, native and editor pick different
/// fallback faces for the same uncovered glyph.
fn manifest_allows_system_fonts(manifest: &serde_json::Value) -> bool {
    manifest.get("systemFonts").and_then(|v| v.as_bool()) != Some(false)
}

fn dir_allows_system_fonts(dir: &std::path::Path) -> bool {
    crate::host::read_file_string(&dir.join("MANIFEST.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .is_none_or(|m| manifest_allows_system_fonts(&m))
}

fn compose_catalog_engine_from_snapshot(
    fallback_dir: Option<&std::path::Path>,
    baked: Option<&BakedCatalog>,
    catalog_dir: &std::path::Path,
    manifest: Option<&serde_json::Value>,
) -> DegradedEngine {
    let mut db = fontdb::Database::new();
    let allow_system = manifest
        .map(manifest_allows_system_fonts)
        .unwrap_or_else(|| dir_allows_system_fonts(catalog_dir));
    if allow_system {
        db.load_system_fonts();
    }
    if let Some(dir) = fallback_dir {
        db.load_fonts_dir(dir);
    }

    // Baked first (shared blobs, one copy per process), per-canvas second so it WINS
    // the families it supplies — `a_per_canvas_family_beats_the_baked_one_of_the_same_name`
    // covers the alias route; the raw tie-break itself is tested separately.
    let mut aliases = HashMap::new();
    let baked_dir = baked.map(|b| {
        for src in &b.sources {
            let shared: std::sync::Arc<dyn AsRef<[u8]> + Send + Sync> = src.clone();
            db.load_font_source(fontdb::Source::Binary(shared));
        }
        aliases.extend(b.aliases.clone());
        b.dir.as_path()
    });

    // Only the per-canvas bytes are charged to the caller's cache budget; the baked
    // blobs above are shared, so charging them per entry would over-count by ~46 MB an
    // engine. Skipped entirely when this IS the baked dir (a catalog-less render).
    let mut catalog_bytes = 0;
    let mut declared_all = std::collections::HashSet::new();
    let mut declared_with_ttf = std::collections::HashSet::new();
    let mut loaded_keys = std::collections::HashSet::new();
    if manifest.is_some() || baked_dir != Some(catalog_dir) {
        let load = match manifest {
            Some(manifest) => load_catalog_fonts_from_manifest(&mut db, catalog_dir, manifest),
            None => load_catalog_fonts(&mut db, catalog_dir),
        };
        // Drop the baked layer's redirects for families this canvas actually SUPPLIED a
        // face for, before taking its own aliases. `load_catalog_fonts` only emits an
        // alias when the face's internal family differs from the catalog name, so a
        // canvas whose face is named exactly like its catalog family contributes NO
        // entry — and the baked redirect would survive underneath and keep winning.
        // Today that is one family (`inter` -> `Inter 24pt`, the only alias the baked
        // set emits), and Inter is the product's default family, so a team font of that
        // name is the case this closes. Extending alone cannot: an absent key overrides
        // nothing.
        //
        // Keyed on LOADED, not declared: a declared face whose bytes do not read (the
        // partial-writer race) would otherwise have its baked redirect removed with
        // nothing to replace it, resolving the family to a generic sans — worse than
        // leaving the baked face in place.
        for family in &load.loaded_families {
            aliases.remove(family);
        }
        aliases.extend(load.aliases);
        catalog_bytes = load.bytes;
        declared_all = load.all;
        declared_with_ttf = load.with_ttf;
        loaded_keys = load.loaded;
    }
    DegradedEngine::from_parts(
        db,
        aliases,
        catalog_bytes,
        declared_all,
        declared_with_ttf,
        loaded_keys,
    )
}

/// The `wght` axis range of a variable face, or None for a static one.
fn wght_range(db: &fontdb::Database, id: fontdb::ID) -> Option<(f32, f32)> {
    db.with_face_data(id, |data, index| {
        let face = rustybuzz::ttf_parser::Face::parse(data, index).ok()?;
        let axis = face
            .variation_axes()
            .into_iter()
            .find(|a| a.tag == rustybuzz::ttf_parser::Tag::from_bytes(b"wght"))?;
        Some((axis.min_value, axis.max_value))
    })
    .flatten()
}

/// Remove each static face whose weight a variable face of the same family, style and stretch also draws. fontdb
/// lists a variable face at its default weight (400), so with a static face of the family beside it (the bundled
/// fallback Noto Sans Arabic Regular next to the installed variable Noto Sans Arabic) the CSS match, and
/// cosmic-text's after it, landed on the static face for every asked weight: bold Arabic drew at 400, silently.
/// Only families with more than one face of a style are read.
fn drop_static_faces_a_variable_face_draws(db: &mut fontdb::Database) {
    let mut groups: HashMap<(String, fontdb::Style, u16), Vec<(fontdb::ID, u16)>> = HashMap::new();
    for face in db.faces() {
        if let Some((family, _)) = face.families.first() {
            groups
                .entry((family.to_lowercase(), face.style, face.stretch.to_number()))
                .or_default()
                .push((face.id, face.weight.0));
        }
    }
    let mut superseded = Vec::new();
    for faces in groups.values().filter(|faces| faces.len() > 1) {
        let ranges: Vec<_> = faces.iter().map(|&(id, _)| wght_range(db, id)).collect();
        for (&(id, weight), range) in faces.iter().zip(&ranges) {
            let weight = f32::from(weight);
            if range.is_none() && ranges.iter().flatten().any(|&(min, max)| (min..=max).contains(&weight)) {
                superseded.push(id);
            }
        }
    }
    for id in superseded {
        db.remove_face(id);
    }
}

/// Point the CSS generic families at faces this font system has, where fontdb's defaults (Arial, Times New Roman)
/// aren't in it — the desktop's render fonts dir and the editor's engine load no OS fonts. Text in the default family
/// (`sans-serif`: markup with no font-family) that lands on this tier (a right-to-left or complex-script run in it)
/// otherwise asks for a family nobody has, and cosmic-text falls back per character through its platform lists — on
/// macOS none of them is in the set either, so Latin came out in whichever face the set lists first (a condensed
/// display face) and digits in the emoji face. The picks are the vector lane's own for the same family
/// (`runs::resolve_renderable_family`: Noto Serif for a serif name where it's present, else Noto Sans), so a
/// paragraph draws in the same face whichever tier draws it. A default that is present (OS fonts loaded) is kept.
fn settle_generic_families(db: &mut fontdb::Database) {
    let has = |db: &fontdb::Database, name: &str| {
        db.faces()
            .any(|f| f.families.iter().any(|(n, _)| n.eq_ignore_ascii_case(name)))
    };
    let pick = |db: &fontdb::Database, names: &[&str]| -> Option<String> {
        names.iter().find(|n| has(db, n)).map(|n| n.to_string())
    };
    if !has(db, db.family_name(&Family::SansSerif)) {
        if let Some(name) = pick(db, &["Noto Sans", "Inter"]) {
            db.set_sans_serif_family(name);
        }
    }
    if !has(db, db.family_name(&Family::Serif)) {
        if let Some(name) = pick(db, &["Noto Serif", "Noto Sans", "Inter"]) {
            db.set_serif_family(name);
        }
    }
}

/// Drop any swapped-in engine so the next [`engine`] call resolves from the ambient
/// configuration again.
///
/// This is the cross-canvas isolation half of the per-canvas install: without it, a
/// canvas whose text needs no catalog would keep rendering against the PREVIOUS
/// canvas's fonts, which is the same silent-substitution failure in the other
/// direction. `moda-py` calls this whenever it restores its default text host.
pub fn uninstall_engine() {
    *engine_slot().write().expect("degraded engine slot") = None;
}

impl DegradedEngine {
    /// Build the font system: OS fonts → vendored Noto fallback set →
    /// vendored MSDF catalog TTFs (MANIFEST.json `keys.*.ttfPath`).
    pub fn load(
        degraded_fonts_dir: Option<&std::path::Path>,
        catalog_fonts_dir: Option<&std::path::Path>,
    ) -> DegradedEngine {
        match catalog_fonts_dir {
            Some(dir) => Self::load_with_catalogs(degraded_fonts_dir, &[dir]),
            None => Self::load_with_catalogs(degraded_fonts_dir, &[]),
        }
    }

    /// [`load`](Self::load) over SEVERAL catalog dirs, applied in order so a later dir
    /// wins the family name (replaces this ordering rule with explicit face removal).
    ///
    /// The per-canvas lane needs this. Its materialized dir carries only the families
    /// `materialize_fonts_dir` resolved for the scene, which is a strict SUBSET of the
    /// baked set: `script_fallback_families` has no mapping for Syriac, Thaana, N'Ko,
    /// Samaritan, Mandaic, Brahmi, Limbu, Tai Tham, Mongolian, Adlam or Hanifi
    /// Rohingya, and the vendored `fonts-degraded` dir carries none of them either —
    /// only NotoSans, NotoSansArabic, NotoSansCJKsc and the emoji face. Loading ONLY a
    /// per-canvas dir would therefore take away the last face those scripts have and
    /// render them as `.notdef` tofu, which is a worse bug than the wrong typeface this
    /// lane exists to fix. Layering keeps the baked catalog as the coverage floor.
    pub fn load_with_catalogs(
        degraded_fonts_dir: Option<&std::path::Path>,
        catalog_fonts_dirs: &[&std::path::Path],
    ) -> DegradedEngine {
        let mut db = fontdb::Database::new();
        // The same system families the golden-capture Chromium used — unless a
        // catalog dir opts out (see `manifest_allows_system_fonts`).
        if catalog_fonts_dirs
            .iter()
            .all(|d| dir_allows_system_fonts(d))
        {
            db.load_system_fonts();
        }
        // fontdb's load_system_fonts is a NO-OP on Android (no android branch
        // as of fontdb 0.23) — load the platform font dirs explicitly so the
        // degraded lane always has fallback faces. cosmic-text PANICS on an
        // empty font db ("no default font found"), and that panic poisons the
        // process-wide engine mutex for every later render.
        #[cfg(target_os = "android")]
        {
            db.load_fonts_dir("/system/fonts");
            db.load_fonts_dir("/system/font");
            db.load_fonts_dir("/product/fonts");
        }
        if let Some(dir) = degraded_fonts_dir {
            db.load_fonts_dir(dir);
        }
        // Catalog TTFs only (atlas pages/emoji SVGs are MSDF-path data). A later dir's
        // aliases overwrite an earlier one's, matching the face precedence fontdb gives
        // the later `load_font_data`.
        let mut aliases = HashMap::new();
        let mut catalog_bytes: u64 = 0;
        for dir in catalog_fonts_dirs {
            let load = load_catalog_fonts(&mut db, dir);
            aliases.extend(load.aliases);
            catalog_bytes = catalog_bytes.saturating_add(load.bytes);
        }
        Self::from_db_with_aliases_and_bytes(db, aliases, catalog_bytes)
    }

    /// Build from in-memory font blobs (the wasm editor's VFS catalog TTFs —
    /// no filesystem). Same FontSystem construction as [`DegradedEngine::load`].
    pub fn from_font_data(fonts: Vec<Vec<u8>>) -> DegradedEngine {
        let mut db = fontdb::Database::new();
        db.load_system_fonts(); // no-op on wasm32; native symmetry
        for bytes in fonts {
            db.load_font_data(bytes);
        }
        Self::from_db(db)
    }

    /// Build from SHARED font handles (the wasm editor's VFS `Arc`s). Identical
    /// to [`from_font_data`] — `load_font_data(v)` is `load_font_source(
    /// Source::Binary(Arc::new(v)))` — but the fontdb SHARES the VFS allocation
    /// instead of copying every TTF/OTF a second time into the engine.
    pub fn from_font_sources(sources: Vec<std::sync::Arc<Vec<u8>>>) -> DegradedEngine {
        Self::from_font_sources_with_aliases(sources, HashMap::new())
    }

    /// [`from_font_sources`] plus catalog family aliases — the wasm editor
    /// derives them from its in-memory MANIFEST.json via
    /// [`manifest_family_aliases`] so bracket-named catalog TTFs ("Inter" →
    /// name-table "Inter 24pt") resolve to the same face `@font-face` binds,
    /// exactly like the filesystem lane's [`DegradedEngine::load`].
    pub fn from_font_sources_with_aliases(
        sources: Vec<std::sync::Arc<Vec<u8>>>,
        family_aliases: HashMap<String, String>,
    ) -> DegradedEngine {
        let mut db = fontdb::Database::new();
        db.load_system_fonts(); // no-op on wasm32; native symmetry
        for src in sources {
            let shared: std::sync::Arc<dyn AsRef<[u8]> + Send + Sync> = src;
            db.load_font_source(fontdb::Source::Binary(shared));
        }
        Self::from_db_with_aliases(db, family_aliases)
    }

    fn from_db(db: fontdb::Database) -> DegradedEngine {
        Self::from_db_with_aliases(db, HashMap::new())
    }

    fn from_db_with_aliases(
        db: fontdb::Database,
        family_aliases: HashMap<String, String>,
    ) -> DegradedEngine {
        Self::from_db_with_aliases_and_bytes(db, family_aliases, 0)
    }

    fn from_db_with_aliases_and_bytes(
        db: fontdb::Database,
        family_aliases: HashMap<String, String>,
        catalog_bytes: u64,
    ) -> DegradedEngine {
        Self::from_parts(
            db,
            family_aliases,
            catalog_bytes,
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }

    fn from_parts(
        mut db: fontdb::Database,
        family_aliases: HashMap<String, String>,
        catalog_bytes: u64,
        catalog_all: std::collections::HashSet<String>,
        catalog_with_ttf: std::collections::HashSet<String>,
        catalog_loaded: std::collections::HashSet<String>,
    ) -> DegradedEngine {
        settle_generic_families(&mut db);
        drop_static_faces_a_variable_face_draws(&mut db);
        let font_system = FontSystem::new_with_locale_and_db("en-US".to_string(), db);
        DegradedEngine {
            inner: Mutex::new(EngineInner {
                font_system,
                swash: SwashCache::new(),
                width_cache: HashMap::new(),
                metrics_cache: HashMap::new(),
                family_aliases,
                emoji_cells: HashMap::new(),
                emoji_effects: HashMap::new(),
                recording: None,
            }),
            id: NEXT_ENGINE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            catalog_all,
            catalog_with_ttf,
            catalog_loaded,
            catalog_bytes,
        }
    }

    /// This engine's fonts as a fontdb for SVG `<text>` (resvg lays out SVG images' text through it):
    /// the same faces and CSS generic families this engine shapes with, plus each catalog family alias
    /// ("Inter" for a face named "Inter 24pt") as an extra family name on its faces, so an SVG asking
    /// for a catalog family by its catalog name finds the face text nodes draw with.
    pub fn svg_font_database(&self) -> fontdb::Database {
        let inner = self.inner.lock().expect("degraded engine");
        let mut db = inner.font_system.db().clone();
        let mut aliased = Vec::new();
        for face in db.faces() {
            let extra: Vec<String> = inner
                .family_aliases
                .iter()
                .filter(|(catalog, drawn)| {
                    face.families
                        .iter()
                        .any(|(n, _)| n.eq_ignore_ascii_case(drawn))
                        && !face
                            .families
                            .iter()
                            .any(|(n, _)| n.eq_ignore_ascii_case(catalog))
                })
                .map(|(catalog, _)| catalog.clone())
                .collect();
            if !extra.is_empty() {
                aliased.push((face.id, extra));
            }
        }
        for (id, extra) in aliased {
            let Some(mut info) = db.face(id).cloned() else {
                continue;
            };
            db.remove_face(id);
            info.families.extend(
                extra
                    .into_iter()
                    .map(|n| (n, fontdb::Language::English_UnitedStates)),
            );
            db.push_face_info(info);
        }
        db
    }

    /// This engine's process-unique id: a cache of anything it drew (the GPU emoji
    /// textures) keys on it, so a swapped-in engine (the wasm editor rebuilds one on
    /// every font install, e.g. once the colour emoji face arrives) draws afresh.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// One colour emoji cluster as the vector lane's image cell
    /// (`vector::VectorEmoji`): straight RGBA, [`EMOJI_CELL_TEXELS`] texels per em
    /// wide and `EMOJI_CELL_HEIGHT_EM / EMOJI_CELL_WIDTH_EM` of that tall, the baseline
    /// `EMOJI_CELL_ASCENT_EM` below the top.
    ///
    /// The glyph comes from this font system's colour emoji face (Apple Color Emoji's
    /// sbix strikes or the bundled Noto Color Emoji's CBDT ones natively; the wasm
    /// editor's fontdb holds only the bundled Noto face), shaped with rustybuzz so ZWJ
    /// sequences, skin tones, keycaps and flags resolve to their ligature glyph. It is
    /// scaled so its advance fits the one-em layout slot and centred in it.
    ///
    /// `None` when no face draws the cluster in colour (only a monochrome or notdef glyph
    /// resolves, or the font system is still empty): there is no cell, and the draw that
    /// asked for it fails (`moda-text-draw`'s `emoji_image`). Cached per engine, misses included.
    pub fn emoji_cell_rgba(&self, cluster: &str) -> Option<std::sync::Arc<crate::host::RgbaImage>> {
        let mut inner = self.inner.lock().expect("degraded engine poisoned");
        if let Some(hit) = inner.emoji_cells.get(cluster) {
            return hit.clone();
        }
        let made = if inner.font_system.db().faces().next().is_none() {
            None
        } else {
            rasterize_emoji_cell(&mut inner, cluster).map(std::sync::Arc::new)
        };
        inner.emoji_cells.insert(cluster.to_string(), made.clone());
        made
    }

    /// An effect pass's copy of an emoji cell ([`crate::vector::EmojiEffect`]): the
    /// cell's silhouette at [`crate::vector::EMOJI_EFFECT_TEXELS_PER_EM`], shaped with the
    /// analytic glyph shader's own formulas over its signed distance (a drop shadow or
    /// glow softens the edge over ± the blur, a stroke is the band outside the edge), in
    /// the pass colour with its opacity baked in, grown by `pad_texels` on every side.
    /// Straight RGBA; a glow's copy draws through `text-emoji-additive`, which adds it
    /// like the glyph glow's additive pipeline.
    pub fn emoji_effect_rgba(
        &self,
        cluster: &str,
        effect: &crate::vector::EmojiEffect,
    ) -> Option<std::sync::Arc<crate::host::RgbaImage>> {
        let cell = self.emoji_cell_rgba(cluster)?;
        let key = effect.key(cluster);
        let mut inner = self.inner.lock().expect("degraded engine poisoned");
        if let Some(hit) = inner.emoji_effects.get(&key) {
            return Some(hit.clone());
        }
        let made = std::sync::Arc::new(emoji_effect_raster(&cell, effect));
        // Effect parameters follow slider drags; keep the recent ones only.
        if inner.emoji_effects.len() >= 128 {
            inner.emoji_effects.clear();
        }
        inner.emoji_effects.insert(key, made.clone());
        Some(made)
    }

    /// Which of `required` this engine has no face for.
    ///
    /// The caller already validates its TextHost against the same required set; this is
    /// the symmetric check for the OTHER font system, and it exists because the two are
    /// read from the dir at different moments. A key fails two ways: it was declared
    /// with a `ttfPath` whose bytes would not read, or the manifest this engine read
    /// never listed it at all — a concurrent same-key writer having replaced the dir
    /// with a partial snapshot in between. A key the manifest lists WITHOUT a `ttfPath`
    /// is legitimately faceless here (MSDF-only) and is not reported.
    ///
    /// A caller with no per-canvas expectations passes an empty required set.
    /// An empty/unreadable manifest cannot prove coverage of nonempty requirements:
    /// it may be the partial writer's outcome, even when every fetch failed.
    pub fn unresolved_catalog_keys(&self, required: &[String]) -> Vec<String> {
        required
            .iter()
            .filter(|id| {
                if self.catalog_loaded.contains(*id) {
                    return false;
                }
                // Declared with a ttf that did not read, or absent from the manifest
                // entirely. Declared without a ttf is fine.
                self.catalog_with_ttf.contains(*id) || !self.catalog_all.contains(*id)
            })
            .cloned()
            .collect()
    }

    /// Heap bytes of the catalog TTFs this engine holds — what a cache that keeps
    /// engines must add to its budget. Excludes the shared system-font db and the
    /// swash/metrics caches, which grow with use; the entry-count cap bounds those.
    pub fn catalog_bytes(&self) -> u64 {
        self.catalog_bytes
    }

    /// Whether the engine's fontdb has any faces loaded. `canvas_logical_size`
    /// shapes via cosmic-text, which PANICS on an empty db ("no default font
    /// found") — callers that size before rendering (the vector mask-raster
    /// arm) must check this first; [`Self::render_node`] guards internally.
    pub fn has_fonts(&self) -> bool {
        let inner = self.inner.lock().expect("degraded engine poisoned");
        let loaded = inner.font_system.db().faces().next().is_some();
        loaded
    }

    /// `getRichTextCanvasLogicalSize` (textSizing.ts:18-29): intrinsic
    /// content size at the node's wrap width, plus 6px horizontal texture
    /// padding each side. Returns (width_logical, height_logical).
    pub fn canvas_logical_size(&self, spec: &TextNodeSpec) -> (f64, f64) {
        let paragraphs = parse_rich_text_to_paragraphs(&spec.html);
        let max_width = spec.wrap_width();
        let (iw, ih) = {
            let mut inner = self.inner.lock().expect("degraded engine poisoned");
            measure_paragraphs_intrinsic_size(
                &mut inner,
                &paragraphs,
                max_width,
                spec.paragraph_spacing.max(0.0),
                spec.line_height_policy,
            )
        };
        // measureRichTextContentSize returns ceil(content).
        let content_w = iw.ceil();
        let content_h = ih.ceil();
        (content_w + HORIZONTAL_TEXTURE_PADDING_PX * 2.0, content_h)
    }

    /// The texture a node's raster is drawn into ([`RasterGeometry`]). Lines align in the node's own box, placed in
    /// the texture where [`raster_offset`] puts the texture in the node, so a paragraph lands where it would in the
    /// box whatever the quad is anchored to. The texture is the widest line plus padding
    /// ([`Self::canvas_logical_size`]), widened to the box when a paragraph aligns to another edge than the quad's
    /// anchor: a justified paragraph fills the box, and a right-aligned paragraph under a left-aligned one ends at
    /// the box's right edge. Auto-width text has no box: its lines align in the texture.
    fn raster_geometry(&self, spec: &TextNodeSpec, paragraphs: &[Paragraph]) -> RasterGeometry {
        let (natural_w, height) = self.canvas_logical_size(spec);
        let wrap_width = natural_w.max(0.0).min(spec.wrap_width());
        let fixed_box = spec
            .width
            .filter(|w| !spec.auto_width && w.is_finite() && *w > 0.0);
        let Some(node_w) = fixed_box else {
            return RasterGeometry {
                width: natural_w,
                height,
                wrap_width,
                box_x: 0.0,
                box_w: natural_w.max(0.0),
            };
        };
        let anchor = rich_text_alignment_from_html(&spec.html);
        let elsewhere = paragraphs.iter().any(|p| {
            let is_ul = p.list_metadata.as_ref().is_some_and(|m| m.kind == "ul");
            let align = if is_ul { "left" } else { p.align };
            align != anchor
        });
        let width = if elsewhere {
            natural_w.max(node_w + HORIZONTAL_TEXTURE_PADDING_PX * 2.0)
        } else {
            natural_w
        };
        let (dx, _) = raster_offset(spec, width, height);
        RasterGeometry {
            width,
            height,
            wrap_width,
            box_x: -dx,
            box_w: node_w,
        }
    }

    /// `renderRichTextNodeToCanvasWithFixedSize` — the full Canvas2D draw
    /// walk onto a `qualityScale`-supersampled RGBA raster.
    pub fn render_node(&self, spec: &TextNodeSpec) -> Result<DegradedRaster, String> {
        self.render_node_scaled(spec, RasterScale::Quality)
            .map(|r| r.expect("an unscaled raster is never skipped"))
    }

    /// [`Self::render_node`] for an export or capture drawn at `output_scale` output px per logical px (the export
    /// scale times the node's own): [`QUALITY_SCALE`] texels per logical px, or more when the output is finer (a PNG
    /// at 300 dpi, a PDF's 300 ppi picture), so the text is as sharp as the output, up to [`MAX_RASTER_TEXELS`].
    pub fn render_node_at(&self, spec: &TextNodeSpec, output_scale: f64) -> Result<DegradedRaster, String> {
        self.render_node_scaled(spec, RasterScale::Output(output_scale))
            .map(|r| r.expect("an output raster is never skipped"))
    }

    /// [`Self::render_node`] for a node drawn at `screen_scale` device pixels per
    /// logical pixel (a camera frame: zoom × dpr × the node's own scale). The raster
    /// is never finer than the screen shows it — `min(QUALITY_SCALE, screen_scale)`
    /// texels per logical pixel — and a node whose raster would cover less than
    /// [`MIN_ON_SCREEN_PX`] on screen in either direction is skipped (`Ok(None)`).
    /// Without this, a zoomed-out document of 100k text nodes rasterised every one of
    /// them at 2× its full size before its fonts arrived: gigabytes of bitmaps for text
    /// a few pixels (or less) tall.
    pub fn render_node_on_screen(
        &self,
        spec: &TextNodeSpec,
        screen_scale: f64,
    ) -> Result<Option<DegradedRaster>, String> {
        self.render_node_scaled(spec, RasterScale::Screen(screen_scale))
    }

    /// vork: the outline of every glyph [`Self::render_node`] draws, placed where its raster puts it in the node
    /// ([`raster_quad_offset`]), px, y down, in draw order. Read from the same draw
    /// walk (the glyphs, faces, sizes and pens the raster is drawn from), each outline unhinted at the glyph's size
    /// from its face (swash, cosmic-text's own outline scaler: a variable face at the drawn weight, a synthetic
    /// italic skewed). For text drawn as geometry: this tier's text as paths.
    pub fn glyph_outlines(&self, spec: &TextNodeSpec) -> Result<Vec<GlyphOutline>, String> {
        self.inner.lock().expect("degraded engine poisoned").recording = Some(GlyphRecording::default());
        let raster = self.render_node(spec);
        let recording = self.inner.lock().expect("degraded engine poisoned").recording.take();
        let raster = raster?;
        let recording = recording.unwrap_or_default();
        let lines_h = raster.logical_h - raster.pad_top - raster.pad_bottom;
        let (offset_x, offset_y) = raster_offset(spec, raster.logical_w, lines_h);
        let mut inner = self.inner.lock().expect("degraded engine poisoned");
        let EngineInner { font_system, swash, .. } = &mut *inner;
        let mut outlines = Vec::with_capacity(recording.glyphs.len());
        for g in &recording.glyphs {
            let (key, _, _) = cosmic_text::CacheKey::new(
                g.font_id,
                g.glyph_id,
                g.font_size,
                (0.0, 0.0),
                g.font_weight,
                g.flags | cosmic_text::CacheKeyFlags::DISABLE_HINTING,
            );
            let Some(commands) = swash.get_outline_commands_uncached(font_system, key) else {
                continue;
            };
            // swash's y points up from the pen.
            let (x0, y0) = (g.x + offset_x, g.y - recording.origin_y + offset_y);
            let place = |x: f32, y: f32| [(x0 + f64::from(x)) as f32, (y0 - f64::from(y)) as f32];
            outlines.push(GlyphOutline {
                pen: place(0.0, 0.0),
                segments: commands
                    .iter()
                    .map(|c| match *c {
                        cosmic_text::Command::MoveTo(p) => OutlineSegment::MoveTo(place(p.x, p.y)),
                        cosmic_text::Command::LineTo(p) => OutlineSegment::LineTo(place(p.x, p.y)),
                        cosmic_text::Command::QuadTo(c, p) => OutlineSegment::QuadTo(place(c.x, c.y), place(p.x, p.y)),
                        cosmic_text::Command::CurveTo(a, b, p) => {
                            OutlineSegment::CurveTo(place(a.x, a.y), place(b.x, b.y), place(p.x, p.y))
                        }
                        cosmic_text::Command::Close => OutlineSegment::Close,
                    })
                    .collect(),
            });
        }
        Ok(outlines)
    }

    fn render_node_scaled(
        &self,
        spec: &TextNodeSpec,
        raster_scale: RasterScale,
    ) -> Result<Option<DegradedRaster>, String> {
        let mut paragraphs = parse_rich_text_to_paragraphs(&spec.html);
        self.validate_paragraphs(&paragraphs)?;
        apply_fill_override(&mut paragraphs, spec.fill_override.as_deref());
        let geometry = self.raster_geometry(spec, &paragraphs);
        let (width_logical, height_logical) = (geometry.width, geometry.height);
        let scale = match raster_scale {
            RasterScale::Quality => QUALITY_SCALE,
            RasterScale::Output(s) => {
                output_raster_scale(width_logical, height_logical + 2.0 * max_font_size(&paragraphs), s)
            }
            RasterScale::Screen(s) => match on_screen_raster_scale(width_logical, height_logical, s) {
                Some(scale) => scale,
                None => return Ok(None),
            },
        };
        let canvas_w = ((width_logical * scale).floor() as i64).max(1) as u32;
        let canvas_h = ((height_logical * scale).floor() as i64).max(1) as u32;
        if canvas_w > 8192 || canvas_h > 8192 {
            return Err(format!(
                "degraded raster {canvas_w}x{canvas_h} exceeds texture budget"
            ));
        }
        // The lines draw between margins of the largest font size above and below, so ink past the first or last
        // line (marks over a line-height under the font's own) lands in the texture; [`trim_ink_margins`] then
        // keeps only the margin rows that drew. Whole pixels, so the lines' own texels are unchanged.
        let margin_px = ((max_font_size(&paragraphs) * scale).ceil().max(0.0) as u32)
            .min(8192u32.saturating_sub(canvas_h) / 2);
        let mut canvas = Canvas::new(canvas_w, canvas_h + 2 * margin_px);
        let mut inner = self.inner.lock().expect("degraded engine poisoned");
        if let Some(recording) = inner.recording.as_mut() {
            recording.origin_y = margin_px as f64 / scale;
        }

        // Layout (canvas2dRichTextRenderer.ts:165-198): lines align in the node's box ([`RasterGeometry`]). They
        // WRAP at the node's own wrap width, never wider: wrapping at the texture (the widest line plus 12px of
        // padding) pulled a word up onto a line the box can't hold (a box fitted to its text re-wrapped —
        // docs/qa/text-editing.md).
        let wrap_width = geometry.wrap_width;
        let mut per_para: Vec<Vec<Line>> = Vec::with_capacity(paragraphs.len());
        for para in &paragraphs {
            per_para.push(layout_para_canvas2d(&mut inner, para, wrap_width, spec.line_height_policy));
        }

        // detectMixedAlpha across all segments (alpha.ts:29-39).
        let mut first_alpha = -1.0f64;
        let mut mixed = false;
        for lines in &per_para {
            for line in lines {
                for seg in &line.segments {
                    let a = alpha_from_color(&seg.style.color);
                    if first_alpha < 0.0 {
                        first_alpha = a;
                    } else if (a - first_alpha).abs() > MIXED_ALPHA_TOLERANCE {
                        mixed = true;
                    }
                }
            }
        }

        // Draw walk (canvas2dRichTextRenderer.ts:203-386).
        let default = default_style();
        let mut y = margin_px as f64 / scale;
        let para_spacing = spec.paragraph_spacing.max(0.0);
        for (p_index, para) in paragraphs.iter().enumerate() {
            y = advance_pen_y_for_paragraph_break(y, p_index == 0, para_spacing);
            let lines = &per_para[p_index];
            let list_kind = para
                .list_metadata
                .as_ref()
                .filter(|m| m.kind == "ul" || m.kind == "ol")
                .map(|m| m.kind.clone());
            let list_level = para.list_metadata.as_ref().map(|m| m.level).unwrap_or(0);
            let is_list_paragraph = list_kind.as_deref() == Some("ul");
            // Paragraph-level UAX #9 resolution for the line-assembly reorder
            // (UAX #9's W/N rules are paragraph-scoped — a European number
            // after Arabic stays in the RTL run even when a wrap puts it at a
            // line start). `None` for LTR-only paragraphs keeps the draw walk
            // byte-identical. The TS twin is `utils/richtext/bidiLayout.ts`
            // (`computeParagraphBidiIfRtl` + `computeVisualLine` in
            // `canvas2dRichTextRenderer.ts`) — keep in lockstep.
            let para_text: String = para.runs.iter().map(|r| r.text.as_str()).collect();
            let para_bidi = crate::bidi_visual::paragraph_bidi_if_rtl(&para_text, para.direction.rtl);
            // Bytes consumed by earlier lines (the wrap only drops whitespace);
            // `None` once a line fails to match — the rest draws logically.
            let mut bidi_cursor: Option<usize> = Some(0);
            for (l_index, line) in lines.iter().enumerate() {
                let max_font_size = line
                    .segments
                    .iter()
                    .fold(default.font_size, |m, s| m.max(s.style.font_size));
                let line_height_px = if line.line_height_px > 0.0 {
                    line.line_height_px
                } else {
                    max_font_size * default.line_height
                };
                let baseline = {
                    let provider = ProviderRef(&mut inner);
                    compute_baseline_for_line_with_fallback(
                        line,
                        line_height_px,
                        max_font_size,
                        &provider,
                    )
                    .baseline_from_top_px
                };
                let baseline_y_abs = y + baseline;

                let measurer = MeasurerRef(&mut inner);
                let total_width = measure_line_width(line, &measurer);
                // The paragraph's indents (`crate::paragraph_format`).
                let (indent_x, line_width) =
                    crate::positioning::indented_container(line, geometry.box_x, geometry.box_w);
                // A justified RTL paragraph's unstretched lines sit right (`crate::paragraph_direction`).
                let line_start_x = compute_line_start_x(
                    crate::paragraph_direction::line_start_align(para.align, para.direction.rtl, line),
                    total_width,
                    indent_x,
                    line_width,
                    is_list_paragraph,
                );
                let font_size = para
                    .runs
                    .first()
                    .map(|r| r.style.font_size)
                    .unwrap_or(default.font_size);
                let first_seg_style = line
                    .segments
                    .first()
                    .map(|s| s.style.clone())
                    .or_else(|| para.runs.first().map(|r| r.style.clone()))
                    .or_else(|| para.base_style.clone())
                    .unwrap_or_else(default_style);
                let (first_ascent, _fd) = {
                    let provider = ProviderRef(&mut inner);
                    provider.metrics(&first_seg_style)
                };
                let ind = compute_list_indentation(
                    list_kind.as_deref(),
                    list_level,
                    font_size,
                    None,
                    Some(first_ascent),
                );

                // UL bullet on the first visual line.
                if is_list_paragraph && l_index == 0 {
                    let raw_bullet_color = para
                        .runs
                        .first()
                        .map(|r| r.style.color.clone())
                        .or_else(|| para.base_style.as_ref().map(|s| s.color.clone()))
                        .unwrap_or_else(|| default.color.clone());
                    let bullet_color = css_color_to_normalized_rgba(&raw_bullet_color)
                        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                    let bm = compute_bullet_metrics(font_size, None, Some(first_ascent));
                    let center_y = baseline_y_abs
                        + bm.center_y_offset_from_baseline
                        + bm.vertical_adjustment_px;
                    let text_start_x = line_start_x + ind.left_offset_px;
                    let center_x = compute_bullet_center_x(
                        text_start_x,
                        ind.marker_box_width_px,
                        ind.marker_gap_px,
                        ind.left_offset_px,
                        bm.horizontal_adjustment_px,
                    );
                    canvas.fill_circle(
                        center_x * scale,
                        center_y * scale,
                        bm.radius_px * scale,
                        bullet_color,
                    );
                }

                // Line-level UAX #9 reordering (display-only): the wrap emits
                // one segment per word/space in LOGICAL order, so without this
                // an Arabic/Hebrew line draws its words left-to-right in
                // logical order (letters join per segment, word order wrong).
                // `visual_line` splits segments into single-level pieces and
                // assigns their x's in visual order against the PARAGRAPH's
                // resolution; `None` (no RTL content / identity resolution)
                // keeps the logical walk byte-identical. See `crate::bidi_visual`.
                // Justify widens the spaces of the line, in logical or visual order alike.
                let justify_px =
                    justify_space_px(para.align, line, total_width, line_width - ind.left_offset_px);
                let visual = match (&para_bidi, bidi_cursor) {
                    (Some(bidi), Some(cursor)) => {
                        match crate::bidi_visual::match_line_segment_ranges(
                            &para_text,
                            &line.segments,
                            cursor,
                        ) {
                            Some((ranges, next_cursor)) => {
                                bidi_cursor = Some(next_cursor);
                                let measurer = MeasurerRef(&mut inner);
                                let measurer = JustifiedMeasurer {
                                    inner: &measurer,
                                    space_px: justify_px,
                                };
                                crate::bidi_visual::visual_line(
                                    bidi,
                                    &para_text,
                                    &ranges,
                                    line,
                                    line_start_x + ind.left_offset_px,
                                    &measurer,
                                )
                            }
                            None => {
                                bidi_cursor = None;
                                None
                            }
                        }
                    }
                    _ => None,
                };
                let draw_line: &Line = visual.as_ref().map(|v| &v.line).unwrap_or(line);
                let positions = match &visual {
                    Some(v) => v.positions.clone(),
                    None => {
                        let measurer = MeasurerRef(&mut inner);
                        compute_segment_positions_justified(
                            line,
                            line_start_x,
                            ind.left_offset_px,
                            &measurer,
                            justify_px,
                        )
                    }
                };
                for (i, seg) in draw_line.segments.iter().enumerate() {
                    // resolveColor → parseColorInput; failure keeps the raw
                    // string which canvas treats as no-op — we fall back to
                    // DEFAULT_STYLE black (degraded-tier approximation).
                    let mut color = css_color_to_normalized_rgba(&seg.style.color)
                        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                    if !mixed {
                        // Force opaque rasterization; avg alpha ships in
                        // fill_alpha (canvas2dRichTextRenderer.ts:299-314).
                        color[3] = 1.0;
                    }
                    draw_segment_text(
                        &mut inner,
                        &mut canvas,
                        &seg.text,
                        &seg.style,
                        positions[i].start_x,
                        baseline_y_abs,
                        color,
                        scale,
                    );
                    // Tab leaders (`crate::tab_stops`), copy by copy on the paragraph's grid.
                    for (text, x) in crate::tab_stops::leader_pieces(seg.tab.as_ref(), positions[i].start_x) {
                        draw_segment_text(
                            &mut inner,
                            &mut canvas,
                            text,
                            &seg.style,
                            x,
                            baseline_y_abs,
                            color,
                            scale,
                        );
                    }
                }

                // Decorations (underline then strikethrough), drawn as
                // thickness-centered rects exactly like the ctx strokes. The
                // logical helpers return line-relative x's (offset by base_x
                // at draw); the visual-order builder reads the reordered
                // positions and returns ABSOLUTE x's (base 0).
                let (rects, rect_base_x) = match &visual {
                    Some(v) => {
                        let inner_measurer = MeasurerRef(&mut inner);
                        let measurer = JustifiedMeasurer {
                            inner: &inner_measurer,
                            space_px: justify_px,
                        };
                        let mut r = crate::bidi_visual::compute_visual_underline_rects(
                            &v.line,
                            &v.order,
                            &v.positions,
                            baseline_y_abs,
                            &measurer,
                        );
                        r.extend(crate::bidi_visual::compute_visual_strikethrough_rects(
                            &v.line,
                            &v.order,
                            &v.positions,
                            baseline_y_abs,
                            &measurer,
                        ));
                        (r, 0.0)
                    }
                    None => {
                        let inner_measurer = MeasurerRef(&mut inner);
                        let measurer = JustifiedMeasurer {
                            inner: &inner_measurer,
                            space_px: justify_px,
                        };
                        let mut r = crate::decorations::compute_underline_rects(
                            line,
                            baseline_y_abs,
                            &measurer,
                        );
                        r.extend(crate::decorations::compute_strikethrough_rects(
                            line,
                            baseline_y_abs,
                            &measurer,
                        ));
                        (r, line_start_x + ind.left_offset_px)
                    }
                };
                for r in &rects {
                    let color =
                        css_color_to_normalized_rgba(&r.color).unwrap_or([0.0, 0.0, 0.0, 1.0]);
                    // Each style's shapes (crate::decorations): a solid line is one rect, as before.
                    for shape in crate::decorations::decoration_shapes(r, rect_base_x, r.x1) {
                        match shape {
                            crate::decorations::DecorationShape::Rect { x0, y0, x1, y1 } => {
                                canvas.fill_rect(x0 * scale, y0 * scale, x1 * scale, y1 * scale, color)
                            }
                            crate::decorations::DecorationShape::Dot { cx, cy, r } => {
                                canvas.fill_circle(cx * scale, cy * scale, r * scale, color)
                            }
                            crate::decorations::DecorationShape::Wave { points, width } => {
                                let pts: Vec<[f64; 2]> =
                                    points.iter().map(|p| [p[0] * scale, p[1] * scale]).collect();
                                canvas.stroke_polyline(&pts, width * scale, color);
                            }
                        }
                    }
                }

                y += line_height_px;
            }
            if lines.is_empty() {
                let s = para.base_style.clone().unwrap_or_else(default_style);
                let lh = if s.line_height > 0.0 {
                    s.line_height
                } else {
                    default.line_height
                };
                y += s.font_size * lh;
            }
        }

        // computeAverageAlpha (alpha.ts:41-55) — width-weighted.
        let fill_alpha = if mixed {
            1.0
        } else {
            let mut weighted = 0.0f64;
            let mut weight = 0.0f64;
            for lines in &per_para {
                for line in lines {
                    for seg in &line.segments {
                        let a = alpha_from_color(&seg.style.color);
                        let measurer = MeasurerRef(&mut inner);
                        let w = crate::positioning::segment_advance(seg, &measurer).max(1.0);
                        weighted += a * w;
                        weight += w;
                    }
                }
            }
            if weight == 0.0 {
                1.0
            } else {
                (weighted / weight).clamp(0.0, 1.0)
            }
        };

        let (mut rgba, pad_top_px, pad_bottom_px) =
            trim_ink_margins(canvas.data, canvas_w, canvas_h, margin_px);
        bleed_into_transparent(&mut rgba, canvas_w, canvas_h + pad_top_px + pad_bottom_px);
        let (pad_top, pad_bottom) = (pad_top_px as f64 / scale, pad_bottom_px as f64 / scale);
        Ok(Some(DegradedRaster {
            rgba,
            width: canvas_w,
            height: canvas_h + pad_top_px + pad_bottom_px,
            logical_w: width_logical,
            logical_h: height_logical + pad_top + pad_bottom,
            quality_scale: scale,
            pad_top,
            pad_bottom,
            fill_alpha: fill_alpha as f32,
        }))
    }
}

/// A Degraded raster smaller than this on screen (device px, either direction) is
/// not drawn at all — see [`DegradedEngine::render_node_on_screen`].
pub const MIN_ON_SCREEN_PX: f64 = 1.0;

/// The largest font size any run or paragraph of `paragraphs` sets.
fn max_font_size(paragraphs: &[Paragraph]) -> f64 {
    paragraphs
        .iter()
        .flat_map(|p| p.runs.iter().map(|r| r.style.font_size).chain(p.base_style.as_ref().map(|s| s.font_size)))
        .filter(|s| s.is_finite())
        .fold(default_style().font_size, f64::max)
}

/// A `w`-wide RGBA texture of `h` line rows between `margin` rows above and below, cut to the line rows plus the
/// margin rows next to them that hold ink. Returns the pixels and how many rows of ink are kept above and below.
fn trim_ink_margins(rgba: Vec<u8>, w: u32, h: u32, margin: u32) -> (Vec<u8>, u32, u32) {
    if margin == 0 {
        return (rgba, 0, 0);
    }
    let row = w as usize * 4;
    let inked = |r: u32| rgba[r as usize * row..(r as usize + 1) * row].chunks_exact(4).any(|px| px[3] != 0);
    let top = (0..margin).find(|&r| inked(r)).unwrap_or(margin);
    let total = h + 2 * margin;
    let bottom = (margin + h..total).rev().find(|&r| inked(r)).map_or(margin + h, |r| r + 1);
    let kept = rgba[top as usize * row..bottom as usize * row].to_vec();
    (kept, margin - top, bottom - (margin + h))
}

/// Gives each fully transparent texel of a straight-alpha `w` × `h` raster the colour of an inked neighbour (alpha
/// stays 0). The raster is drawn through a linear sampler between texels, which mixes a transparent texel's RGB into
/// the edge texel next to it: with the canvas's (0, 0, 0, 0) that darkened every glyph edge drawn at another scale
/// than the raster's, a dark seam where white text crossed other ink. Pixels sampled at texel centres are unchanged.
fn bleed_into_transparent(rgba: &mut [u8], w: u32, h: u32) {
    let (w, h) = (w as usize, h as usize);
    let src = rgba.to_vec();
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            if src[i + 3] != 0 {
                continue;
            }
            let mut sum = [0u32; 3];
            let mut n = 0u32;
            for ny in y.saturating_sub(1)..(y + 2).min(h) {
                for nx in x.saturating_sub(1)..(x + 2).min(w) {
                    let j = (ny * w + nx) * 4;
                    if src[j + 3] != 0 {
                        for c in 0..3 {
                            sum[c] += src[j + c] as u32;
                        }
                        n += 1;
                    }
                }
            }
            if n > 0 {
                for c in 0..3 {
                    rgba[i + c] = ((sum[c] + n / 2) / n) as u8;
                }
            }
        }
    }
}

/// The node's solid text fill (`textFills`, folded to `spec.fill_override`) colours every run, as the vector lane
/// draws it (`vector/ops.rs`: the override replaces each run's colour). This tier (complex scripts, emoji, glyphs
/// only a fallback face covers) drew the HTML's run colours instead: white `textFills` over the HTML's black drew
/// black. The text-as-mask raster clears the override first (`vector/mask_raster.rs`), so masks keep their coverage.
fn apply_fill_override(paragraphs: &mut [Paragraph], fill_override: Option<&str>) {
    let Some(color) = fill_override.filter(|c| css_color_to_normalized_rgba(c).is_some()) else {
        return;
    };
    for para in paragraphs {
        if let Some(base) = para.base_style.as_mut() {
            base.color = color.to_string();
        }
        for run in &mut para.runs {
            run.style.color = color.to_string();
        }
    }
}

/// How finely [`DegradedEngine::render_node_scaled`] rasterises.
#[derive(Debug, Clone, Copy)]
enum RasterScale {
    /// [`QUALITY_SCALE`] texels per logical px.
    Quality,
    /// An export or capture at this many output px per logical px ([`output_raster_scale`]).
    Output(f64),
    /// A camera frame at this many device px per logical px ([`on_screen_raster_scale`]).
    Screen(f64),
}

/// The most texels an output raster ([`DegradedEngine::render_node_at`]) is made finer to (64 MB of RGBA). Past it
/// the raster stays at the finest scale that fits, never under [`QUALITY_SCALE`].
pub const MAX_RASTER_TEXELS: f64 = 4096.0 * 4096.0;

/// Texels per logical pixel for a `logical_w` × `logical_h` Degraded raster (its ink margins included) drawn into an
/// output at `output_scale` px per logical px: the output's own density, at least [`QUALITY_SCALE`] (what the raster
/// always had), at most what keeps it within 8192 texels a side and [`MAX_RASTER_TEXELS`]. A raster coarser than its
/// output is upscaled and blurs: Thai or Hindi text in a 300 dpi PNG drew at 144 dpi beside sharp Latin text.
pub fn output_raster_scale(logical_w: f64, logical_h: f64, output_scale: f64) -> f64 {
    if !(output_scale.is_finite() && output_scale > QUALITY_SCALE) {
        return QUALITY_SCALE;
    }
    let (w, h) = (logical_w.max(1.0), logical_h.max(1.0));
    let fits = (8192.0 / w).min(8192.0 / h).min((MAX_RASTER_TEXELS / (w * h)).sqrt());
    output_scale.min(fits).max(QUALITY_SCALE)
}

/// Texels per logical pixel for a `logical_w` × `logical_h` Degraded raster shown at
/// `screen_scale` device px per logical px: the screen's own density, capped at
/// [`QUALITY_SCALE`]. `None` when the raster would be under [`MIN_ON_SCREEN_PX`] on
/// screen (or the scale is degenerate), i.e. nothing worth drawing.
pub fn on_screen_raster_scale(logical_w: f64, logical_h: f64, screen_scale: f64) -> Option<f64> {
    if !(screen_scale.is_finite() && screen_scale > 0.0) {
        return None;
    }
    if logical_w * screen_scale < MIN_ON_SCREEN_PX || logical_h * screen_scale < MIN_ON_SCREEN_PX {
        return None;
    }
    Some(screen_scale.min(QUALITY_SCALE))
}

/// One shaped cluster of [`DegradedEngine::resolve_faces`]: the byte range of the text it covers, the face cosmic-text
/// drew it with (the asked one, or the fallback it found by coverage) and whether that face had a glyph for it
/// (`false`: `.notdef`, a box).
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedCluster {
    pub start: usize,
    pub end: usize,
    pub family: Option<String>,
    /// The face's own weight and whether it is italic (a fallback face may not have the asked ones).
    pub weight: u16,
    pub italic: bool,
    pub has_glyph: bool,
}

/// The weight a glyph of `face` is drawn at when shaped for `asked`: a variable face is instanced at `asked`
/// clamped to its `wght` axis (cosmic-text's `Font::new` sets that location), a static face at its own weight.
/// `face.weight` alone is the default instance's `usWeightClass` — 900 for the bundled Fraunces, whatever was
/// asked.
fn drawn_weight(db: &fontdb::Database, face: &fontdb::FaceInfo, asked: u16) -> u16 {
    db.with_face_data(face.id, |data, index| {
        let parsed = rustybuzz::ttf_parser::Face::parse(data, index).ok()?;
        let axis = parsed
            .variation_axes()
            .into_iter()
            .find(|a| a.tag == rustybuzz::ttf_parser::Tag::from_bytes(b"wght"))?;
        Some(f32::from(asked).clamp(axis.min_value, axis.max_value).round() as u16)
    })
    .flatten()
    .unwrap_or(face.weight.0)
}

impl DegradedEngine {
    /// Which face this tier draws each part of `text` in, set in `family` / `weight` / `italic`: the same attrs and
    /// shaping as [`Self::render_node`] (catalog aliases, cosmic-text's coverage fallback). Read-only; one line, no
    /// wrapping. For `moda-render font-resolve` (the SDK's `fonts.has`).
    pub fn resolve_faces(
        &self,
        family: &str,
        weight: u16,
        italic: bool,
        text: &str,
    ) -> Result<Vec<ResolvedCluster>, String> {
        let mut inner = self.inner.lock().expect("degraded engine poisoned");
        if inner.font_system.db().faces().next().is_none() {
            return Err("degraded engine has no fonts loaded".to_string());
        }
        let style = InlineStyle {
            font_family: family.to_string(),
            font_weight: weight.to_string(),
            font_style: if italic { "italic" } else { "normal" }.to_string(),
            ..default_style()
        };
        let aliases = inner.family_aliases.clone();
        let attrs = attrs_for_style(&style, &aliases);
        let list = attrs_list_for(&inner.font_system, text, &attrs);
        let shape = ShapeLine::new(&mut inner.font_system, text, &list, Shaping::Advanced, 8);
        let mut out = Vec::new();
        for span in &shape.spans {
            for word in &span.words {
                for g in &word.glyphs {
                    let db = inner.font_system.db();
                    let face = db.face(g.font_id);
                    out.push(ResolvedCluster {
                        start: g.start,
                        end: g.end,
                        family: face.and_then(|f| f.families.first().map(|(name, _)| name.clone())),
                        weight: face
                            .map(|f| drawn_weight(db, f, g.font_weight.0))
                            .unwrap_or(weight),
                        italic: face
                            .map(|f| f.style != fontdb::Style::Normal)
                            .unwrap_or(italic),
                        has_glyph: g.glyph_id != 0,
                    });
                }
            }
        }
        out.sort_by_key(|c| (c.start, c.end));
        Ok(out)
    }

    /// Refuse requests this lane cannot faithfully draw before measuring or allocating a raster.
    /// Inspect shaped clusters, not just cmap coverage: this is the same face selection and feature
    /// application as the draw walk, including fallback among the installed fonts.
    fn validate_paragraphs(&self, paragraphs: &[Paragraph]) -> Result<(), String> {
        let mut inner = self.inner.lock().expect("degraded engine poisoned");
        if inner.font_system.db().faces().next().is_none() {
            return Err("degraded engine has no fonts loaded".to_string());
        }
        let aliases = inner.family_aliases.clone();
        let mut missing = String::new();
        for run in paragraphs.iter().flat_map(|p| &p.runs) {
            if !run.style.font_variations.is_empty() {
                return Err("font-variation-settings are not supported by the degraded text lane".to_string());
            }
            // This lane spaces independently shaped graphemes. Scripts whose forms depend on
            // their neighbours cannot use that algorithm (the same contextual scripts for which
            // the parser preserves ZWNJ); keep their normal whole-run shaping at zero spacing.
            if run.style.letter_spacing != 0.0 && run.text.chars().any(crate::parser::zwnj_shapes_after) {
                return Err("nonzero letter-spacing on contextually shaped text is not supported by the degraded text lane".to_string());
            }
            let attrs = attrs_for_style(&run.style, &aliases);
            let list = attrs_list_for(&inner.font_system, &run.text, &attrs);
            let shape = ShapeLine::new(&mut inner.font_system, &run.text, &list, Shaping::Advanced, 8);
            for glyph in shape.spans.iter().flat_map(|s| &s.words).flat_map(|w| &w.glyphs) {
                if glyph.glyph_id == 0 {
                    for ch in run.text[glyph.start..glyph.end].chars() {
                        if !missing.contains(ch) {
                            missing.push(ch);
                        }
                    }
                }
            }
        }
        if !missing.is_empty() {
            return Err(format!("font coverage missing for characters {missing:?}"));
        }
        Ok(())
    }

    /// The lines [`Self::render_node`] draws, as JSON in the vector layout dump's shape
    /// (`crate::vector::dump_rich_text_node_vector`): paragraphs → lines (`lineTopY`, `lineHeightPx`,
    /// `baselineY`, `lineStartX`, `totalWidth`, `listLeftOffset`) → segments (`text`, `x`, `width`). Positions are
    /// in the raster's own space; the caller places the raster (`logicalW` / `logicalH`) in the node. Paragraphs
    /// with right-to-left text come back `"bidi": true` (their segments are in logical order, not drawn order).
    pub fn layout_json(&self, spec: &TextNodeSpec) -> Result<serde_json::Value, String> {
        use serde_json::{json, Value};
        let paragraphs = parse_rich_text_to_paragraphs(&spec.html);
        self.validate_paragraphs(&paragraphs)?;
        let geometry = self.raster_geometry(spec, &paragraphs);
        let (width_logical, height_logical) = (geometry.width, geometry.height);
        let wrap_width = geometry.wrap_width;
        let mut inner = self.inner.lock().expect("degraded engine poisoned");
        let default = default_style();
        let para_spacing = spec.paragraph_spacing.max(0.0);
        let mut y = 0.0;
        let mut paragraphs_out = Vec::with_capacity(paragraphs.len());
        for (p_index, para) in paragraphs.iter().enumerate() {
            y = advance_pen_y_for_paragraph_break(y, p_index == 0, para_spacing);
            let lines = layout_para_canvas2d(&mut inner, para, wrap_width, spec.line_height_policy);
            let list_kind = para
                .list_metadata
                .as_ref()
                .filter(|m| m.kind == "ul" || m.kind == "ol")
                .map(|m| m.kind.clone());
            let list_level = para.list_metadata.as_ref().map(|m| m.level).unwrap_or(0);
            let is_list_paragraph = list_kind.as_deref() == Some("ul");
            let para_text: String = para.runs.iter().map(|r| r.text.as_str()).collect();
            let bidi = para.direction.rtl
                || crate::bidi_visual::paragraph_bidi_if_rtl(&para_text, false).is_some();
            let font_size = para
                .runs
                .first()
                .map(|r| r.style.font_size)
                .unwrap_or(default.font_size);
            let mut lines_out = Vec::with_capacity(lines.len());
            for line in &lines {
                let max_font_size = line
                    .segments
                    .iter()
                    .fold(default.font_size, |m, s| m.max(s.style.font_size));
                let line_height_px = if line.line_height_px > 0.0 {
                    line.line_height_px
                } else {
                    max_font_size * default.line_height
                };
                let baseline = {
                    let provider = ProviderRef(&mut inner);
                    compute_baseline_for_line_with_fallback(
                        line,
                        line_height_px,
                        max_font_size,
                        &provider,
                    )
                    .baseline_from_top_px
                };
                let measurer = MeasurerRef(&mut inner);
                let total_width = measure_line_width(line, &measurer);
                // The paragraph's indents (`crate::paragraph_format`).
                let (indent_x, line_width) =
                    crate::positioning::indented_container(line, geometry.box_x, geometry.box_w);
                // A justified RTL paragraph's unstretched lines sit right (`crate::paragraph_direction`).
                let line_start_x = compute_line_start_x(
                    crate::paragraph_direction::line_start_align(para.align, para.direction.rtl, line),
                    total_width,
                    indent_x,
                    line_width,
                    is_list_paragraph,
                );
                let first_seg_style = line
                    .segments
                    .first()
                    .map(|s| s.style.clone())
                    .or_else(|| para.runs.first().map(|r| r.style.clone()))
                    .or_else(|| para.base_style.clone())
                    .unwrap_or_else(default_style);
                let (first_ascent, _) = ProviderRef(&mut inner).metrics(&first_seg_style);
                let ind = compute_list_indentation(
                    list_kind.as_deref(),
                    list_level,
                    font_size,
                    None,
                    Some(first_ascent),
                );
                let measurer = MeasurerRef(&mut inner);
                let justify_px =
                    justify_space_px(para.align, line, total_width, line_width - ind.left_offset_px);
                let positions = compute_segment_positions_justified(
                    line,
                    line_start_x,
                    ind.left_offset_px,
                    &measurer,
                    justify_px,
                );
                let segments: Vec<Value> = line
                    .segments
                    .iter()
                    .zip(positions.iter())
                    .map(|(s, p)| {
                        if s.hyphen {
                            json!({ "text": s.text, "x": p.start_x, "width": p.width, "hyphen": true })
                        } else {
                            json!({ "text": s.text, "x": p.start_x, "width": p.width, "sourceIndices": s.style.source.values() })
                        }
                    })
                    .collect();
                let mut line_out = json!({
                    "lineTopY": y,
                    "lineHeightPx": line_height_px,
                    "baselineY": y + baseline,
                    "lineStartX": line_start_x,
                    "totalWidth": total_width,
                    "listLeftOffset": ind.left_offset_px,
                });
                line_out["segments"] = Value::Array(segments);
                if justify_px != 0.0 {
                    line_out["justifySpacePx"] = json!(justify_px);
                }
                lines_out.push(line_out);
                y += line_height_px;
            }
            if lines.is_empty() {
                let s = para.base_style.clone().unwrap_or_else(default_style);
                let lh = if s.line_height > 0.0 {
                    s.line_height
                } else {
                    default.line_height
                };
                y += s.font_size * lh;
            }
            // The arrays are moved in, not through `json!` (which serializes them into a deep copy).
            let mut para_out = json!({
                "paragraphIndex": p_index,
                "align": para.align,
                "listKind": list_kind,
                "listLevel": list_level,
                "fontSize": font_size,
                "bidi": bidi,
            });
            para_out["lines"] = Value::Array(lines_out);
            paragraphs_out.push(para_out);
        }
        let mut out = json!({
            "engine": "degraded",
            "logicalW": width_logical,
            "logicalH": height_logical,
            "totalHeightPx": y,
        });
        out["paragraphs"] = Value::Array(paragraphs_out);
        Ok(out)
    }
}

/// `layoutListParagraph` (layoutList.ts:6-22) / `layoutParagraph` dispatch —
/// the Canvas2D loop (canvas2dRichTextRenderer.ts:180-198).
fn layout_para_canvas2d(
    inner: &mut EngineInner,
    para: &Paragraph,
    content_width: f64,
    policy: crate::types::LineHeightPolicy,
) -> Vec<Line> {
    let is_ul = para
        .list_metadata
        .as_ref()
        .map(|m| m.kind == "ul")
        .unwrap_or(false);
    let measurer = MeasurerRef(&mut *inner);
    let lines = if is_ul {
        let font_size = para
            .runs
            .first()
            .map(|r| r.style.font_size)
            .unwrap_or_else(|| default_style().font_size);
        let level = para.list_metadata.as_ref().map(|m| m.level).unwrap_or(0);
        let indent_px = compute_list_indent_px(level, font_size);
        let m = compute_marker_metrics(font_size, None, None);
        let available = (content_width - indent_px - m.marker_box_width_px).max(0.0);
        layout_paragraph_with_policy(para, available, &measurer, Some(&crate::vector::cjk::segment_text_for_wrap_with), policy)
    } else {
        layout_paragraph_with_policy(para, content_width, &measurer, Some(&crate::vector::cjk::segment_text_for_wrap_with), policy)
    };
    if lines.is_empty() && policy == crate::types::LineHeightPolicy::Css {
        // An empty CSS paragraph still consumes its line box, just as the vector walk does.
        crate::paragraph_layout::layout_paragraphs_with(
            std::slice::from_ref(para), 0., &measurer, false, |_, _, _| Vec::new(),
        ).paragraphs.remove(0).lines
    } else {
        lines
    }
}

/// `measureParagraphsIntrinsicSize` (`utils/richtext/intrinsicSize.ts`).
fn measure_paragraphs_intrinsic_size(
    inner: &mut EngineInner,
    paragraphs: &[Paragraph],
    max_width: f64,
    paragraph_spacing_px: f64,
    policy: crate::types::LineHeightPolicy,
) -> (f64, f64) {
    let default = default_style();
    let mut total_height = 0.0f64;
    let mut max_line_width = 0.0f64;
    for (idx, para) in paragraphs.iter().enumerate() {
        let list = para
            .list_metadata
            .as_ref()
            .filter(|m| m.kind == "ul" || m.kind == "ol")
            .map(|m| (m.kind.clone(), m.level));
        let is_ul = matches!(&list, Some((k, _)) if k == "ul");
        let font_size = para
            .runs
            .first()
            .map(|r| r.style.font_size)
            .unwrap_or(default.font_size);
        let mut effective_max_width = max_width;
        let mut list_extra = 0.0;
        if is_ul {
            let level = list.as_ref().map(|(_, l)| *l).unwrap_or(0);
            let indent_px = compute_list_indent_px(level, font_size);
            let m = compute_marker_metrics(font_size, None, None);
            effective_max_width = (max_width - indent_px - m.marker_box_width_px).max(0.0);
            list_extra = indent_px + m.marker_box_width_px;
        }
        let lines = {
            let measurer = MeasurerRef(&mut *inner);
            layout_paragraph_with_policy(para, effective_max_width, &measurer, Some(&crate::vector::cjk::segment_text_for_wrap_with), policy)
        };
        if lines.is_empty() {
            total_height += if policy == crate::types::LineHeightPolicy::Css {
                crate::paragraph_layout::empty_line_height_px(para, &MeasurerRef(&mut *inner))
            } else {
                let s = para.base_style.clone().unwrap_or_else(default_style);
                policy.resolve(s.font_size * s.line_height)
            };
        }
        let mut last_line_height = 0.0f64;
        let mut last_content_box = 0.0f64;
        for line in &lines {
            let w = {
                let measurer = MeasurerRef(&mut *inner);
                // Indents included (`crate::paragraph_format`).
                crate::positioning::line_extent_px(line, &measurer)
            };
            let effective_width = w + list_extra;
            if effective_width > max_line_width {
                max_line_width = effective_width;
            }
            let max_font_size = line
                .segments
                .iter()
                .fold(default.font_size, |m, s| m.max(s.style.font_size));
            let line_height_px = if line.line_height_px > 0.0 {
                line.line_height_px
            } else {
                max_font_size * default.line_height
            };
            let mut max_ascent = 0.0f64;
            let mut max_descent = 0.0f64;
            for seg in &line.segments {
                let provider = ProviderRef(&mut *inner);
                let (a, d) = provider.metrics(&seg.style);
                max_ascent = max_ascent.max(a);
                max_descent = max_descent.max(d);
            }
            total_height += line_height_px;
            last_line_height = line_height_px;
            last_content_box = max_ascent + max_descent;
        }
        if !lines.is_empty() {
            total_height += (last_content_box - last_line_height).max(0.0);
        }
        if idx < paragraphs.len() - 1 {
            total_height += paragraph_spacing_px.max(0.0);
        }
    }
    (max_line_width, total_height)
}

// ---------------------------------------------------------------------------
// cosmic-text measurement / metrics / raster
// ---------------------------------------------------------------------------

/// Load the catalog TTFs listed in `dir/MANIFEST.json` (`keys.*.ttfPath`)
/// into `db` and return the catalog family aliases: lowercased manifest
/// `family` → the loaded face's actual fontdb family name, for every entry
/// whose name-table family differs from the catalog family (opsz-bracketed
/// releases like `Inter_24pt-*.ttf` → "Inter 24pt"). The browser's
/// `@font-face` declares the CATALOG family for the same file, so Canvas2D
/// resolves "Inter" to this face; cosmic-text matches name-table families,
/// so the alias is what keeps the degraded lane on the same face instead of
/// silently substituting Noto Sans (wrong ascent/lsb → the probe-proven
/// (+6,+6)px mask:text-alpha misplacement). First manifest entry wins when
/// several weights of one catalog family carry different internal names.
/// What one catalog dir contributed to an engine.
#[derive(Default)]
struct CatalogLoad {
    aliases: HashMap<String, String>,
    bytes: u64,
    /// Every key the manifest listed.
    all: std::collections::HashSet<String>,
    /// Those carrying a `ttfPath`.
    with_ttf: std::collections::HashSet<String>,
    /// Those whose bytes actually read.
    loaded: std::collections::HashSet<String>,
    /// Lowercased families this dir actually SUPPLIED a face for. What an earlier
    /// layer's redirects should be cleared against — clearing on merely-declared
    /// families removes a working baked redirect and leaves nothing in its place.
    loaded_families: std::collections::HashSet<String>,
}

/// The manifest prefix the desktop's render fonts dir gives the bundled Noto fallback set
/// (`apps/desktop/main/fonts/index.cjs`, `writeRenderFontsDir`: `_fallback/<file>`).
const FALLBACK_SET_KEY_PREFIX: &str = "_fallback/";

/// A manifest's `keys` in load order: the bundled fallback set first, then the catalog
/// faces, each group in manifest order.
///
/// Load order is fontdb id order, and cosmic-text's last coverage stage (any face, after the
/// named fallback lists) walks equally good faces in id order. The editor's engine loads the
/// fallback set first (its VFS paths `_degraded/NN-…` sort before every catalog face's
/// `<slug>/…`, `host::vfs_font_sources_sorted`), so a glyph the asked face lacks goes to the
/// fallback set — `→` and `✓` beside a Fraunces `⑩` draw in Noto Sans CJK SC. The desktop
/// manifest lists the fallback set AFTER the catalog, so loading it in manifest order put
/// every bundled family ahead of it and the export drew `→` in Anton and `✓` in IBM Plex
/// Mono (#1761). Same rule as `DegradedEngine::load`: fallback set before catalog TTFs.
fn fallback_set_first(
    keys: &serde_json::Map<String, serde_json::Value>,
) -> impl Iterator<Item = (&String, &serde_json::Value)> {
    let is_fallback = |k: &String| k.starts_with(FALLBACK_SET_KEY_PREFIX);
    keys.iter()
        .filter(move |(k, _)| is_fallback(k))
        .chain(keys.iter().filter(move |(k, _)| !is_fallback(k)))
}

fn load_catalog_fonts(db: &mut fontdb::Database, dir: &std::path::Path) -> CatalogLoad {
    // VFS-FIRST, like `TextHost::load`. When the fonts-cache volume is read-only or
    // full, `materialize_fonts_dir` keeps the fetched bytes in the in-memory overlay
    // (`moda_text::host::install_vfs`) rather than on disk. Reading through `std::fs`
    // here would find nothing, load zero catalog faces, and — because the caller caches
    // the pair as a clean catalog host — pin that silent substitution for every later
    // render of the canvas, while the TextHost beside it loaded perfectly.
    let Ok(raw) = crate::host::read_file_string(&dir.join("MANIFEST.json")) else {
        return CatalogLoad::default();
    };
    let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return CatalogLoad::default();
    };
    load_catalog_fonts_from_manifest(db, dir, &manifest)
}

fn load_catalog_fonts_from_manifest(
    db: &mut fontdb::Database,
    dir: &std::path::Path,
    manifest: &serde_json::Value,
) -> CatalogLoad {
    let mut out = CatalogLoad::default();
    let aliases = &mut out.aliases;
    let mut bytes_read: u64 = 0;
    let Some(keys) = manifest.get("keys").and_then(|k| k.as_object()) else {
        return out;
    };
    // Several manifest keys can share one ttfPath (multi-weight files);
    // read + load each file once and reuse its face ids for every key.
    let mut loaded: HashMap<String, Vec<fontdb::ID>> = HashMap::new();
    for (key_id, entry) in fallback_set_first(keys) {
        out.all.insert(key_id.clone());
        let Some(rel) = entry.get("ttfPath").and_then(|p| p.as_str()) else {
            continue;
        };
        out.with_ttf.insert(key_id.clone());
        let ids = match loaded.get(rel) {
            Some(ids) => ids.clone(),
            None => {
                let path = dir.join(rel);
                if path
                    .extension()
                    .is_some_and(|extension| extension == "font-face")
                {
                    let Ok(face) = crate::collection::descriptor(&path) else {
                        continue;
                    };
                    let mut metadata_db = fontdb::Database::new();
                    let ids = metadata_db.load_font_source(fontdb::Source::Binary(
                        std::sync::Arc::new(face.metadata),
                    ));
                    let infos: Vec<_> = ids
                        .iter()
                        .filter_map(|id| metadata_db.face(*id).cloned())
                        .collect();
                    let ids: Vec<_> = infos
                        .into_iter()
                        .map(|mut info| {
                            info.source = fontdb::Source::File(face.source.clone());
                            info.index = face.index;
                            db.push_face_info(info)
                        })
                        .collect();
                    loaded.insert(rel.to_string(), ids.clone());
                    ids
                } else {
                    let Ok(bytes) = crate::host::read_file_shared(&dir.join(rel)) else {
                        loaded.insert(rel.to_string(), Vec::new());
                        continue;
                    };
                    bytes_read = bytes_read.saturating_add(bytes.len() as u64);
                    let ids =
                        db.load_font_source(fontdb::Source::Binary(bytes));
                    let ids: Vec<fontdb::ID> = ids.to_vec();
                    loaded.insert(rel.to_string(), ids.clone());
                    ids
                }
            }
        };
        let Some(catalog_family) = entry.get("family").and_then(|f| f.as_str()) else {
            continue;
        };
        // Marked here, not at read time: several manifest keys can share ONE ttfPath
        // (multi-weight files), and the reader only runs for the first of them — marking
        // there left every later key looking unloaded, so a perfectly good multi-weight
        // catalog would report incomplete and never cache. Keyed on face ids being
        // non-empty so bytes that read but fontdb refused are not counted as success
        // either.
        if !ids.is_empty() {
            out.loaded.insert(key_id.clone());
            if let Some(family) = entry.get("family").and_then(|f| f.as_str()) {
                out.loaded_families.insert(family.to_lowercase());
            }
        }
        for id in ids {
            let Some(face) = db.face(id) else { continue };
            let Some((face_family, _)) = face.families.first() else {
                continue;
            };
            if !face_family.eq_ignore_ascii_case(catalog_family) {
                aliases
                    .entry(catalog_family.to_lowercase())
                    .or_insert_with(|| face_family.clone());
            }
        }
    }
    out.bytes = bytes_read;
    out
}

/// The filesystem-free twin of [`load_catalog_fonts`]'s alias derivation, for
/// hosts whose catalog lives behind a byte reader instead of a directory (the
/// wasm editor's VFS: `moda_text::host::read_file_bytes`). Walks the
/// MANIFEST.json `keys.*` entries, parses each `ttfPath`'s bytes in a scratch
/// fontdb, and returns lowercased catalog family → the face's actual
/// name-table family for every entry whose names differ (opsz-bracketed
/// releases). Same first-entry-wins rule as [`load_catalog_fonts`] — pinned
/// identical by `manifest_alias_reader_matches_fs_lane`.
pub fn manifest_family_aliases(
    manifest: &serde_json::Value,
    mut read: impl FnMut(&str) -> Option<Vec<u8>>,
) -> HashMap<String, String> {
    let mut aliases = HashMap::new();
    let Some(keys) = manifest.get("keys").and_then(|k| k.as_object()) else {
        return aliases;
    };
    let mut parsed: HashMap<String, Vec<String>> = HashMap::new();
    for entry in keys.values() {
        let Some(rel) = entry.get("ttfPath").and_then(|p| p.as_str()) else {
            continue;
        };
        let Some(catalog_family) = entry.get("family").and_then(|f| f.as_str()) else {
            continue;
        };
        if aliases.contains_key(&catalog_family.to_lowercase()) {
            continue; // first manifest entry wins; skip the byte read
        }
        // Several keys can share one ttfPath (multi-weight files): parse each
        // file once and reuse its face families.
        let families = match parsed.get(rel) {
            Some(f) => f.clone(),
            None => {
                let families: Vec<String> = match read(rel) {
                    Some(bytes) => {
                        let mut scratch = fontdb::Database::new();
                        let ids = scratch
                            .load_font_source(fontdb::Source::Binary(std::sync::Arc::new(bytes)));
                        ids.iter()
                            .filter_map(|id| scratch.face(*id))
                            .filter_map(|face| face.families.first().map(|(f, _)| f.clone()))
                            .collect()
                    }
                    None => Vec::new(),
                };
                parsed.insert(rel.to_string(), families.clone());
                families
            }
        };
        for face_family in families {
            if !face_family.eq_ignore_ascii_case(catalog_family) {
                aliases
                    .entry(catalog_family.to_lowercase())
                    .or_insert_with(|| face_family);
            }
        }
    }
    aliases
}

/// The face CSS font matching picks for `attrs` (its family, the nearest weight it has, its style), when the
/// family is in the set.
fn matched_face<'f>(db: &'f fontdb::Database, attrs: &Attrs) -> Option<&'f fontdb::FaceInfo> {
    let query = fontdb::Query {
        families: &[attrs.family],
        weight: attrs.weight,
        stretch: attrs.stretch,
        style: attrs.style,
    };
    db.query(&query).and_then(|id| db.face(id))
}

/// `attrs` at the weight its family's matched face draws at ([`drawn_weight`]). cosmic-text takes a family's face
/// only at the asked weight (or a variable face spanning it): Anton (Regular only) asked at 700 was skipped and the
/// whole run drew in whichever bold face came first, where CSS — and the vector lane's nearest-weight snap — draws
/// Anton Regular.
fn settle_weight<'a>(db: &fontdb::Database, attrs: Attrs<'a>) -> Attrs<'a> {
    match matched_face(db, &attrs) {
        Some(face) => {
            let weight = drawn_weight(db, face, attrs.weight.0);
            attrs.weight(Weight(weight))
        }
        None => attrs,
    }
}

fn face_has_glyph(db: &fontdb::Database, id: fontdb::ID, c: char) -> bool {
    db.with_face_data(id, |data, index| {
        rustybuzz::ttf_parser::Face::parse(data, index)
            .ok()
            .and_then(|f| f.glyph_index(c))
            .is_some()
    })
    .unwrap_or(false)
}

fn attrs_for_style<'a>(style: &'a InlineStyle, aliases: &'a HashMap<String, String>) -> Attrs<'a> {
    let family = match style.font_family.trim() {
        "" | "sans-serif" => Family::SansSerif,
        "serif" => Family::Serif,
        "monospace" => Family::Monospace,
        "cursive" => Family::Cursive,
        "fantasy" => Family::Fantasy,
        // The catalog alias (when present) redirects to the face's real
        // name-table family — the same face the browser's `@font-face` binds
        // the catalog name to.
        name => match aliases.get(&name.to_lowercase()) {
            Some(alias) => Family::Name(alias),
            None => Family::Name(name),
        },
    };
    let weight = match style.font_weight.trim() {
        "bold" => 700,
        "normal" | "" => 400,
        w => crate::jsnum::js_parse_int(w).unwrap_or(400).clamp(1, 1000),
    };
    let s = if style.font_style.contains("italic") || style.font_style.contains("oblique") {
        Style::Italic
    } else {
        Style::Normal
    };
    let attrs = Attrs::new()
        .family(family)
        .weight(Weight(weight as u16))
        .style(s);
    if style.font_features.is_empty() {
        return attrs;
    }
    // F7.3: the Degraded tier shapes through cosmic-text's rustybuzz, which
    // takes the run's font-feature-settings as they are.
    let mut features = cosmic_text::FontFeatures::new();
    for f in &style.font_features {
        features.set(cosmic_text::FeatureTag::new(&f.tag), f.value);
    }
    attrs.font_features(features)
}

fn font_string_key(style: &InlineStyle) -> String {
    if !style.font_features.is_empty() {
        return format!(
            "{}|{}|{}|{}",
            style.font_family,
            style.font_weight,
            style.font_style,
            crate::font_features::serialize_font_feature_settings(&style.font_features)
        );
    }
    format!(
        "{}|{}|{}",
        style.font_family, style.font_weight, style.font_style
    )
}

impl EngineInner {
    /// Shaped width at font-size 1 (advances scale linearly, unhinted).
    fn width_em(&mut self, text: &str, style: &InlineStyle) -> f64 {
        let key = (font_string_key(style), text.to_string());
        if let Some(w) = self.width_cache.get(&key) {
            return *w;
        }
        let attrs = attrs_for_style(style, &self.family_aliases);
        let list = attrs_list_for(&self.font_system, text, &attrs);
        let shape = ShapeLine::new(&mut self.font_system, text, &list, Shaping::Advanced, 8);
        let lines = shape.layout(1.0, None, CosmicWrap::None, None, None, Hinting::Disabled);
        let w = lines.iter().map(|l| l.w as f64).fold(0.0, f64::max);
        self.width_cache.insert(key, w);
        w
    }

    /// Letter-spaced width: spacing after EVERY grapheme, the last one
    /// included, as the vector lane places it (`glyphPlacement.ts`) and as
    /// `draw_segment_text` advances. This used to add it between graphemes
    /// only (`measureTextWithLetterSpacing`), but the wrap measures a line
    /// word by word, so each word and each space lost its trailing gap: with
    /// tracking wider than a space (`letter-spacing: 6px` at 20 px) the word
    /// gap came out narrower than the letter gaps and the words ran together.
    fn measure_canvas2d(&mut self, text: &str, style: &InlineStyle) -> f64 {
        if style.letter_spacing == 0.0 || text.is_empty() {
            return self.width_em(text, style) * style.font_size;
        }
        let mut width = 0.0;
        for g in text.graphemes(true) {
            width += self.width_em(g, style) * style.font_size + style.letter_spacing;
        }
        width
    }

    /// `getCanvasFontMetrics` (canvasFontMetrics.ts:14-29): fontBoundingBox
    /// ascent/descent of the STYLED font via the 'HMygjpq' sample — the
    /// requested font's metrics, never the fallback's (spike §4).
    fn canvas_font_metrics(&mut self, style: &InlineStyle) -> (f64, f64) {
        let (a1, d1) = self.canvas_font_box_em(style).unwrap_or((0.82, 0.18)); // ts fallback ratios
        (a1 * style.font_size, d1 * style.font_size)
    }

    /// The styled font's box (ascent, descent) per em, `None` when the font has no metrics.
    fn canvas_font_box_em(&mut self, style: &InlineStyle) -> Option<(f64, f64)> {
        let key = font_string_key(style);
        if let Some(v) = self.metrics_cache.get(&key) {
            *v
        } else {
            let attrs = attrs_for_style(style, &self.family_aliases);
            let list = AttrsList::new(&settle_weight(self.font_system.db(), attrs));
            let shape = ShapeLine::new(
                &mut self.font_system,
                "HMygjpq",
                &list,
                Shaping::Advanced,
                8,
            );
            let lines = shape.layout(1.0, None, CosmicWrap::None, None, None, Hinting::Disabled);
            let v = lines
                .first()
                .map(|l| (l.max_ascent as f64, l.max_descent as f64))
                .filter(|(a, d)| *a > 0.0 || *d > 0.0);
            self.metrics_cache.insert(key, v);
            v
        }
    }
}

/// `TextMeasurer` adapter over the locked engine (canvas2d semantics).
struct MeasurerRef<'a> {
    inner: *mut EngineInner,
    _marker: std::marker::PhantomData<&'a mut EngineInner>,
}

#[allow(non_snake_case)]
fn MeasurerRef(inner: &mut EngineInner) -> MeasurerRef<'_> {
    MeasurerRef {
        inner,
        _marker: std::marker::PhantomData,
    }
}

impl TextMeasurer for MeasurerRef<'_> {
    fn measure(&self, text: &str, style: &InlineStyle) -> f64 {
        // SAFETY: MeasurerRef borrows the EngineInner exclusively for its
        // lifetime (constructed from &mut); TextMeasurer's &self receiver
        // forces the raw-pointer interior mutability.
        unsafe { (*self.inner).measure_canvas2d(text, style) }
    }

    fn auto_line_height_px(&self, style: &InlineStyle) -> Option<f64> {
        // The canvas lane has no line gap: its font box (ascent + descent) at the line's size; none without the
        // font's metrics (the style's multiple applies).
        // SAFETY: see above.
        let (a, d) = unsafe { (*self.inner).canvas_font_box_em(style) }?;
        Some((a + d) * style.line_font_size()).filter(|px| *px > 0.0)
    }
}

struct ProviderRef<'a> {
    inner: *mut EngineInner,
    _marker: std::marker::PhantomData<&'a mut EngineInner>,
}

#[allow(non_snake_case)]
fn ProviderRef(inner: &mut EngineInner) -> ProviderRef<'_> {
    ProviderRef {
        inner,
        _marker: std::marker::PhantomData,
    }
}

impl FontMetricsProvider for ProviderRef<'_> {
    fn metrics(&self, style: &InlineStyle) -> (f64, f64) {
        // SAFETY: see MeasurerRef.
        unsafe { (*self.inner).canvas_font_metrics(style) }
    }
}

/// Draw one segment at (x, baseline) logical px — `drawTextWithLetterSpacing`
/// (utils/canvas/text.ts:3-31): whole-run shaping at spacing 0, else
/// per-grapheme placement (kerning/ligatures break, copy the behavior). A segment is one direction (`bidi_visual`
/// splits a line at level changes); a right-to-left one (its first strong character) runs from its right edge, so
/// its graphemes are placed left to right in reverse and a Hebrew word keeps its letter order.
#[allow(clippy::too_many_arguments)]
fn draw_segment_text(
    inner: &mut EngineInner,
    canvas: &mut Canvas,
    text: &str,
    style: &InlineStyle,
    x: f64,
    baseline_y: f64,
    color: [f64; 4],
    scale: f64,
) {
    if text.is_empty() {
        return;
    }
    if style.letter_spacing == 0.0 {
        draw_shaped_run(inner, canvas, text, style, x, baseline_y, color, scale);
        return;
    }
    let mut cursor = x;
    let mut graphemes: Vec<&str> = text.graphemes(true).collect();
    if crate::paragraph_direction::first_strong_direction(text) == Some(true) {
        graphemes.reverse();
    }
    for g in graphemes {
        draw_shaped_run(inner, canvas, g, style, cursor, baseline_y, color, scale);
        cursor += inner.width_em(g, style) * style.font_size + style.letter_spacing;
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_shaped_run(
    inner: &mut EngineInner,
    canvas: &mut Canvas,
    text: &str,
    style: &InlineStyle,
    x: f64,
    baseline_y: f64,
    color: [f64; 4],
    scale: f64,
) {
    let EngineInner {
        font_system,
        swash,
        family_aliases,
        recording,
        ..
    } = inner;
    let attrs = attrs_for_style(style, family_aliases);
    let list = attrs_list_for(font_system, text, &attrs);
    let shape = ShapeLine::new(font_system, text, &list, Shaping::Advanced, 8);
    let lines = shape.layout(
        style.font_size as f32,
        None,
        CosmicWrap::None,
        None,
        None,
        Hinting::Disabled,
    );
    for ll in &lines {
        for glyph in &ll.glyphs {
            if let Some(recording) = recording.as_mut() {
                recording.glyphs.push(RecordedGlyph {
                    font_id: glyph.font_id,
                    glyph_id: glyph.glyph_id,
                    font_size: glyph.font_size,
                    font_weight: glyph.font_weight,
                    flags: glyph.cache_key_flags,
                    x: x + f64::from(glyph.x + glyph.font_size * glyph.x_offset),
                    y: baseline_y + f64::from(glyph.y - glyph.font_size * glyph.y_offset),
                });
            }
            let (cache_key, px, py) = glyph_raster_key(glyph, (x * scale) as f32, (baseline_y * scale) as f32, scale as f32);
            let Some(img) = swash.get_image(font_system, cache_key).clone() else {
                continue;
            };
            let gx = px + img.placement.left;
            let gy = py - img.placement.top;
            let gw = img.placement.width as i64;
            let gh = img.placement.height as i64;
            match img.content {
                SwashContent::Mask => {
                    for row in 0..gh {
                        for col in 0..gw {
                            let cov = img.data[(row * gw + col) as usize] as f64 / 255.0;
                            if cov > 0.0 {
                                canvas.blend(gx as i64 + col, gy as i64 + row, color, cov);
                            }
                        }
                    }
                }
                SwashContent::Color => {
                    // Color emoji bitmaps: straight RGBA, own colors (the
                    // canvas2d fillStyle does not tint emoji).
                    for row in 0..gh {
                        for col in 0..gw {
                            let i = ((row * gw + col) * 4) as usize;
                            let a = img.data[i + 3] as f64 / 255.0;
                            if a > 0.0 {
                                let c = [
                                    img.data[i] as f64 / 255.0,
                                    img.data[i + 1] as f64 / 255.0,
                                    img.data[i + 2] as f64 / 255.0,
                                    a,
                                ];
                                canvas.blend(gx as i64 + col, gy as i64 + row, c, 1.0);
                            }
                        }
                    }
                }
                SwashContent::SubpixelMask => {
                    // Not produced by cosmic's render order; treat as mask
                    // via the green channel if it ever appears.
                    for row in 0..gh {
                        for col in 0..gw {
                            let i = ((row * gw + col) * 4) as usize;
                            let cov = img.data[i + 1] as f64 / 255.0;
                            if cov > 0.0 {
                                canvas.blend(gx as i64 + col, gy as i64 + row, color, cov);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The swash cache key of `glyph` drawn at `scale` texels per px with its pen at (`x`, `baseline_y`) texels, and the
/// whole texel its image's placement is relative to. `LayoutGlyph::physical` truncates y to a whole texel (hinting
/// for screen text), which put this tier's lines up to a texel above where the vector lane draws them, by a
/// different fraction of a px at every raster scale. Here y is placed to a quarter texel as x is: the outline is
/// rendered shifted up by the bin from the whole texel below the pen (swash's y axis points up).
fn glyph_raster_key(glyph: &cosmic_text::LayoutGlyph, x: f32, baseline_y: f32, scale: f32) -> (cosmic_text::CacheKey, i32, i32) {
    let pen_x = (glyph.x + glyph.font_size * glyph.x_offset).mul_add(scale, x);
    let pen_y = (glyph.y - glyph.font_size * glyph.y_offset).mul_add(scale, baseline_y);
    let below = pen_y.ceil();
    let (key, px, up) = cosmic_text::CacheKey::new(
        glyph.font_id,
        glyph.glyph_id,
        glyph.font_size * scale,
        (pen_x, below - pen_y),
        glyph.font_weight,
        glyph.cache_key_flags,
    );
    (key, px, below as i32 - up)
}

/// Colour emoji faces, most preferred first. The bundled Noto face leads so native renders
/// and exports draw the same artwork as the wasm editor, whose fontdb has no OS fonts.
const COLOR_EMOJI_FAMILIES: [&str; 3] = ["Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji"];

/// The first of [`COLOR_EMOJI_FAMILIES`] this font system holds.
fn color_emoji_family(font_system: &FontSystem) -> Option<&'static str> {
    COLOR_EMOJI_FAMILIES.into_iter().find(|name| {
        font_system
            .db()
            .faces()
            .any(|f| f.families.iter().any(|(n, _)| n == name))
    })
}

/// Characters the run's face lacks, in a script with its own fallback family (`Noto Sans Thai` for Thai, …), get
/// that family where the set has it, at its nearest weight: the vector lane's chain
/// (`script_fallback::list_fallback_variants`). cosmic-text's own script fallback takes a face only at the asked
/// weight, so bold or medium Thai skipped the static Noto Sans Thai and drew in any face of that weight covering
/// Thai (FreeSerif Bold, a serif, beside sans text). Other characters keep cosmic-text's fallback.
fn script_fallback_spans(db: &fontdb::Database, text: &str, attrs: &Attrs, list: &mut AttrsList) {
    use crate::vector::script_fallback::script_fallback_families;
    use crate::vector::unicode_scripts::{classify_codepoint, UnicodeScript};
    let primary = matched_face(db, attrs).map(|f| f.id);
    let mut fallbacks: HashMap<&'static str, Option<(fontdb::ID, u16)>> = HashMap::new();
    let mut runs: Vec<(std::ops::Range<usize>, &'static str, u16)> = Vec::new();
    for (i, c) in text.char_indices() {
        if c.is_whitespace() || primary.is_some_and(|id| face_has_glyph(db, id, c)) {
            continue;
        }
        let script = classify_codepoint(c as u32);
        // Symbol chains are vector-lane coverage choices, not script faces. Keep cosmic-text's existing
        // coverage fallback for them: both canvas and export load the bundled fallback set first, so a symbol
        // it covers keeps that face even when a catalog math or symbol font also covers it.
        if matches!(script, UnicodeScript::MathSymbols | UnicodeScript::Symbols) {
            continue;
        }
        let found = script_fallback_families(script).iter().find_map(|&family| {
            let face = *fallbacks.entry(family).or_insert_with(|| {
                matched_face(db, &attrs.clone().family(Family::Name(family)))
                    .map(|f| (f.id, drawn_weight(db, f, attrs.weight.0)))
            });
            face.filter(|&(id, _)| face_has_glyph(db, id, c)).map(|(_, w)| (family, w))
        });
        let Some((family, weight)) = found else { continue };
        let end = i + c.len_utf8();
        match runs.last_mut() {
            Some((r, f, w)) if r.end == i && *f == family && *w == weight => r.end = end,
            _ => runs.push((i..end, family, weight)),
        }
    }
    for (range, family, weight) in runs {
        list.add_span(range, &attrs.clone().family(Family::Name(family)).weight(Weight(weight)));
    }
}

/// `attrs` over `text`, with each emoji grapheme (an image-lane cluster by the vector lane's own rule,
/// `emoji::split_text_into_segments`) set in the colour emoji face. Through cosmic-text's coverage fallback an
/// emoji took the first face that maps it: on Linux GNU FreeSerif's monochrome outline (🔥) and its boxed letters
/// for a flag's regional indicators, on a Mac Apple Color Emoji, where the vector lane and the wasm editor draw
/// the bundled Noto artwork.
fn attrs_list_for(font_system: &FontSystem, text: &str, attrs: &Attrs) -> AttrsList {
    let mut list = AttrsList::new(&settle_weight(font_system.db(), attrs.clone()));
    script_fallback_spans(font_system.db(), text, attrs, &mut list);
    let maybe_emoji = |c: char| {
        let cp = c as u32;
        crate::emoji::is_extended_pictographic(cp) || (0x1F1E6..=0x1F1FF).contains(&cp) || c == '\u{20E3}'
    };
    if !text.chars().any(maybe_emoji) {
        return list;
    }
    let Some(family) = color_emoji_family(font_system) else {
        return list;
    };
    let emoji = settle_weight(font_system.db(), attrs.clone().family(Family::Name(family)));
    for (start, g) in text.grapheme_indices(true) {
        if g.chars().any(maybe_emoji)
            && matches!(crate::emoji::split_text_into_segments(g).as_slice(), [crate::emoji::Segment::Emoji(_)])
        {
            list.add_span(start..start + g.len(), &emoji);
        }
    }
    list
}

/// [`DegradedEngine::emoji_cell_rgba`]'s raster.
fn rasterize_emoji_cell(inner: &mut EngineInner, cluster: &str) -> Option<crate::host::RgbaImage> {
    use crate::vector::ops::{EMOJI_CELL_ASCENT_EM, EMOJI_CELL_HEIGHT_EM, EMOJI_CELL_WIDTH_EM};
    let EngineInner {
        font_system, swash, ..
    } = inner;
    // Name the colour face outright: through the generic family a VS16 sequence whose base
    // the text face also covers (❤️, ☀️) would shape to that face's monochrome glyph.
    let family = color_emoji_family(font_system);
    let attrs = match family {
        Some(name) => Attrs::new().family(Family::Name(name)),
        None => Attrs::new(),
    };
    let list = AttrsList::new(&attrs);
    let shape = ShapeLine::new(font_system, cluster, &list, Shaping::Advanced, 8);
    let layout = |size: f64| shape.layout(size as f32, None, CosmicWrap::None, None, None, Hinting::Disabled);
    // Scale the cluster so its advance fits the one-em cell, then centre it.
    let advance_em: f64 = layout(1.0)
        .iter()
        .flat_map(|l| l.glyphs.iter())
        .map(|g| g.w as f64)
        .sum();
    let em_px = EMOJI_CELL_TEXELS as f64 / EMOJI_CELL_WIDTH_EM;
    let size = em_px * EMOJI_CELL_WIDTH_EM / advance_em.max(EMOJI_CELL_WIDTH_EM);
    let width = EMOJI_CELL_TEXELS;
    let height = (em_px * EMOJI_CELL_HEIGHT_EM).round() as u32;
    let origin = (
        ((width as f64 - advance_em * size) / 2.0) as f32,
        (em_px * EMOJI_CELL_ASCENT_EM) as f32,
    );
    let mut canvas = Canvas::new(width, height);
    let mut drew = false;
    for line in &layout(size) {
        for glyph in &line.glyphs {
            let phys = glyph.physical(origin, 1.0);
            let img = swash.get_image(font_system, phys.cache_key).clone()?;
            // A monochrome glyph (notdef, or a text face's symbol) is not an emoji image.
            if img.content != SwashContent::Color {
                return None;
            }
            let (gx, gy) = (phys.x + img.placement.left, phys.y - img.placement.top);
            let (gw, gh) = (img.placement.width as i64, img.placement.height as i64);
            for row in 0..gh {
                for col in 0..gw {
                    let i = ((row * gw + col) * 4) as usize;
                    let a = img.data[i + 3] as f64 / 255.0;
                    if a > 0.0 {
                        let c = [
                            img.data[i] as f64 / 255.0,
                            img.data[i + 1] as f64 / 255.0,
                            img.data[i + 2] as f64 / 255.0,
                            a,
                        ];
                        canvas.blend(gx as i64 + col, gy as i64 + row, c, 1.0);
                        drew = true;
                    }
                }
            }
        }
    }
    if !drew {
        return None;
    }
    // Straight RGBA: the `text-emoji` pipeline's colour blend is `src-alpha,
    // one-minus-src-alpha` (a premultiplied cell would darken its soft edges).
    let rgba = canvas.data;
    Some(crate::host::RgbaImage {
        width,
        height,
        rgba,
    })
}

/// [`DegradedEngine::emoji_effect_rgba`]'s raster.
fn emoji_effect_raster(
    cell: &crate::host::RgbaImage,
    effect: &crate::vector::EmojiEffect,
) -> crate::host::RgbaImage {
    // The cell's alpha at the effect resolution (a box filter over `down` × `down`).
    let down = ((EMOJI_CELL_TEXELS as f64 / crate::vector::EMOJI_EFFECT_TEXELS_PER_EM).round() as usize).max(1);
    let (cw, ch) = (cell.width as usize / down, cell.height as usize / down);
    let pad = effect.pad_texels as usize;
    let (w, h) = (cw + 2 * pad, ch + 2 * pad);
    let mut mask = vec![0f32; w * h];
    for y in 0..ch {
        for x in 0..cw {
            let mut sum = 0u32;
            for dy in 0..down {
                for dx in 0..down {
                    sum += cell.rgba[((y * down + dy) * cell.width as usize + x * down + dx) * 4 + 3] as u32;
                }
            }
            mask[(y + pad) * w + x + pad] = sum as f32 / (255.0 * (down * down) as f32);
        }
    }
    let shaped = effect.blur_texels > 0.0 || effect.stroke_texels > 0.0;
    let alpha: Vec<f32> = if !shaped {
        // An echo copy: the plain silhouette (the shader's un-effected coverage).
        mask
    } else {
        // Signed distance, positive inside, in texels: exact distance transforms of the
        // inside and outside sets, with the edge texels' own coverage for the last texel.
        let inside: Vec<bool> = mask.iter().map(|&a| a >= 0.5).collect();
        let to_inside = squared_distance_transform(&inside, w, h);
        let outside: Vec<bool> = inside.iter().map(|&i| !i).collect();
        let to_outside = squared_distance_transform(&outside, w, h);
        let smooth = |e0: f32, e1: f32, x: f32| {
            let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let width = effect.blur_texels.max(0.5);
        (0..w * h)
            .map(|i| {
                let sd = if inside[i] {
                    to_outside[i].sqrt() - 0.5
                } else {
                    -(to_inside[i].sqrt() - 0.5)
                };
                let sd = if sd.abs() <= 1.0 { mask[i] - 0.5 } else { sd };
                if effect.stroke_texels > 0.0 {
                    let outer = smooth(-width, width, sd + effect.stroke_texels);
                    let inner = smooth(-width, width, sd - 0.5);
                    (outer - inner).clamp(0.0, 1.0)
                } else {
                    smooth(-width, width, sd)
                }
            })
            .collect()
    };
    let c = effect.color;
    let mut rgba = vec![0u8; w * h * 4];
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    for (i, a) in alpha.iter().enumerate() {
        // Straight RGBA in the pass colour; a glow's copy draws through the additive
        // `text-emoji-additive` pipeline, which adds `rgb × a` as the glyph glow does.
        let v = a * c[3] * effect.gain;
        rgba[i * 4..i * 4 + 4].copy_from_slice(&[byte(c[0]), byte(c[1]), byte(c[2]), byte(v)]);
    }
    crate::host::RgbaImage {
        width: w as u32,
        height: h as u32,
        rgba,
    }
}

/// Squared Euclidean distance from every texel to the nearest texel of `set`
/// (Felzenszwalb–Huttenlocher, separable). Texels far from any get a large finite value.
fn squared_distance_transform(set: &[bool], w: usize, h: usize) -> Vec<f32> {
    const FAR: f32 = 1.0e12;
    let mut grid: Vec<f32> = set.iter().map(|&s| if s { 0.0 } else { FAR }).collect();
    let n = w.max(h);
    let (mut f, mut d) = (vec![0f32; n], vec![0f32; n]);
    let (mut v, mut z) = (vec![0usize; n], vec![0f32; n + 1]);
    let mut transform_1d = |f: &[f32], len: usize, d: &mut [f32]| {
        let mut k = 0usize;
        v[0] = 0;
        z[0] = f32::NEG_INFINITY;
        z[1] = f32::INFINITY;
        for q in 1..len {
            let mut s;
            loop {
                let p = v[k];
                s = ((f[q] + (q * q) as f32) - (f[p] + (p * p) as f32)) / (2.0 * (q as f32 - p as f32));
                if s <= z[k] && k > 0 {
                    k -= 1;
                } else {
                    break;
                }
            }
            // z[0] is −∞, so the loop stops with s > z[k] (FAR keeps s finite).
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f32::INFINITY;
        }
        k = 0;
        for q in 0..len {
            while z[k + 1] < q as f32 {
                k += 1;
            }
            let p = v[k];
            d[q] = (q as f32 - p as f32).powi(2) + f[p];
        }
    };
    for x in 0..w {
        for y in 0..h {
            f[y] = grid[y * w + x];
        }
        transform_1d(&f, h, &mut d);
        for y in 0..h {
            grid[y * w + x] = d[y];
        }
    }
    for y in 0..h {
        f[..w].copy_from_slice(&grid[y * w..(y + 1) * w]);
        transform_1d(&f, w, &mut d);
        grid[y * w..(y + 1) * w].copy_from_slice(&d[..w]);
    }
    grid
}

fn alpha_from_color(color: &str) -> f64 {
    css_color_to_normalized_rgba(color)
        .map(|c| c[3].clamp(0.0, 1.0))
        .unwrap_or(1.0)
}

// ---------------------------------------------------------------------------
// CPU canvas (straight-alpha source-over, like the 2D context)
// ---------------------------------------------------------------------------

struct Canvas {
    w: u32,
    h: u32,
    data: Vec<u8>,
}

impl Canvas {
    fn new(w: u32, h: u32) -> Canvas {
        Canvas {
            w,
            h,
            data: vec![0; (w as usize) * (h as usize) * 4],
        }
    }

    fn blend(&mut self, x: i64, y: i64, color: [f64; 4], coverage: f64) {
        if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 {
            return;
        }
        let sa = (color[3] * coverage).clamp(0.0, 1.0);
        if sa <= 0.0 {
            return;
        }
        let i = ((y as usize) * (self.w as usize) + x as usize) * 4;
        let da = self.data[i + 3] as f64 / 255.0;
        let oa = sa + da * (1.0 - sa);
        if oa <= 0.0 {
            return;
        }
        for c in 0..3 {
            let sc = color[c];
            let dc = self.data[i + c] as f64 / 255.0;
            let oc = (sc * sa + dc * da * (1.0 - sa)) / oa;
            self.data[i + c] = (oc * 255.0 + 0.5) as u8;
        }
        self.data[i + 3] = (oa * 255.0 + 0.5) as u8;
    }

    /// Axis-aligned rect with edge antialiasing (device px coords).
    fn fill_rect(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, color: [f64; 4]) {
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        let yi0 = y0.floor() as i64;
        let yi1 = (y1.ceil() as i64).max(yi0 + 1);
        let xi0 = x0.floor() as i64;
        let xi1 = (x1.ceil() as i64).max(xi0 + 1);
        for py in yi0..yi1 {
            let cy = ((py as f64 + 1.0).min(y1) - (py as f64).max(y0)).clamp(0.0, 1.0);
            for px in xi0..xi1 {
                let cx = ((px as f64 + 1.0).min(x1) - (px as f64).max(x0)).clamp(0.0, 1.0);
                let cov = cx * cy;
                if cov > 0.0 {
                    self.blend(px, py, color, cov);
                }
            }
        }
    }

    /// A polyline stroked at `width` (device px coords), antialiased by distance — a wavy decoration.
    fn stroke_polyline(&mut self, points: &[[f64; 2]], width: f64, color: [f64; 4]) {
        if points.len() < 2 || width <= 0.0 {
            return;
        }
        let h = width * 0.5;
        let (mut bx0, mut by0, mut bx1, mut by1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for p in points {
            bx0 = bx0.min(p[0]);
            by0 = by0.min(p[1]);
            bx1 = bx1.max(p[0]);
            by1 = by1.max(p[1]);
        }
        let (x0, x1) = (points[0][0], points[points.len() - 1][0]);
        for py in (by0 - h - 1.0).floor() as i64..(by1 + h + 1.0).ceil() as i64 {
            for px in (bx0 - 1.0).floor() as i64..(bx1 + 1.0).ceil() as i64 {
                let (cx, cy) = (px as f64 + 0.5, py as f64 + 0.5);
                let mut d = f64::MAX;
                for s in points.windows(2) {
                    let (ax, ay, bx, by) = (s[0][0], s[0][1], s[1][0], s[1][1]);
                    let (dx, dy) = (bx - ax, by - ay);
                    let l2 = dx * dx + dy * dy;
                    let t = if l2 > 0.0 {
                        (((cx - ax) * dx + (cy - ay) * dy) / l2).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let (qx, qy) = (ax + t * dx - cx, ay + t * dy - cy);
                    d = d.min((qx * qx + qy * qy).sqrt());
                }
                // Butt ends: the stroke stops at the first and last x.
                let ends = ((cx - x0 + 0.5).min(x1 - cx + 0.5)).clamp(0.0, 1.0);
                let cov = (h - d + 0.5).clamp(0.0, 1.0) * ends;
                if cov > 0.0 {
                    self.blend(px, py, color, cov);
                }
            }
        }
    }

    /// Filled antialiased disc (device px coords) — `ctx.arc` + fill.
    fn fill_circle(&mut self, cx: f64, cy: f64, r: f64, color: [f64; 4]) {
        if r <= 0.0 {
            return;
        }
        let x0 = (cx - r - 1.0).floor() as i64;
        let x1 = (cx + r + 1.0).ceil() as i64;
        let y0 = (cy - r - 1.0).floor() as i64;
        let y1 = (cy + r + 1.0).ceil() as i64;
        for py in y0..y1 {
            for px in x0..x1 {
                let d =
                    (((px as f64 + 0.5) - cx).powi(2) + ((py as f64 + 0.5) - cy).powi(2)).sqrt();
                let cov = (r - d + 0.5).clamp(0.0, 1.0);
                if cov > 0.0 {
                    self.blend(px, py, color, cov);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg_attr(not(vork_private_tests), allow(dead_code, unused_imports))]
mod tests {
    use super::*;

    fn fixtures_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap()
            .to_path_buf()
    }

    /// The engine over Moda's degraded and oracle font directories, for the tests that wait on them.
    fn oracle_engine() -> DegradedEngine {
        DegradedEngine::load(
            Some(&fixtures_root().join("fixtures/fonts-degraded")),
            Some(&fixtures_root().join("fixtures/oracle/fonts")),
        )
    }

    fn test_engine() -> DegradedEngine {
        // Use only shipped faces: missing Moda fixtures must not silently fall
        // through to whichever fonts happen to be installed on the test host.
        let root = fixtures_root();
        let mut db = fontdb::Database::new();
        for path in [
            "crates/moda-engine/tests/fixtures/fonts/inter/Inter_24pt-Regular.ttf",
            "crates/moda-engine/tests/fixtures/fonts/inter/Inter_24pt-ExtraBold.ttf",
            "crates/moda-engine/tests/fixtures/fonts/noto-sans/NotoSans-Regular.ttf",
            "crates/vork-std/tests/fonts/NotoSansArabic-Regular.ttf",
            crate::world_type_fonts::WORLD_TYPE_HEBREW,
            crate::world_type_fonts::WORLD_TYPE_THAI,
            crate::world_type_fonts::WORLD_TYPE_DEVANAGARI,
            "crates/moda-text/tests/fonts/HanTest.otf",
        ] {
            db.load_font_data(std::fs::read(root.join(path)).unwrap());
        }
        DegradedEngine::from_db_with_aliases(db, HashMap::from([("inter".into(), "Inter 24pt".into())]))
    }

    /// `resolve_faces` names the face each part of the text is drawn in: the asked family where it has the glyphs, a
    /// coverage fallback (here a CJK face) where it doesn't, with byte ranges of the text.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn resolve_faces_reports_the_asked_face_and_the_fallback() {
        let engine = test_engine();
        let clusters = engine
            .resolve_faces("Inter", 400, false, "A漢")
            .expect("resolve");
        let first = clusters.iter().find(|c| c.start == 0).expect("A");
        // The face's own family name ("Inter 24pt" for the optical-size cut the catalog ships).
        let is_inter =
            |c: &ResolvedCluster| c.family.as_deref().is_some_and(|f| f.starts_with("Inter"));
        assert!(is_inter(first), "{clusters:?}");
        assert!(first.has_glyph);
        let han = clusters.iter().find(|c| c.start == 1).expect("漢");
        assert!(!is_inter(han), "{clusters:?}");
        assert!(han.has_glyph, "{clusters:?}");
        assert_eq!(han.end, 4);
    }

    /// #1761: a run the vector lane declines (Fraunces with a `⑩` nothing on that lane covers) draws on this tier,
    /// SVG images lay their `<text>` out with the engine's faces, generic families included, and a
    /// catalog family asked for by its catalog name finds the face its alias names.
    #[test]
    fn svg_font_database_carries_the_engines_faces_and_aliases() {
        let font = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../moda-engine/tests/fixtures/fonts/noto-sans/NotoSans-Regular.ttf");
        let bytes = std::sync::Arc::new(std::fs::read(font).unwrap());
        let mut aliases = HashMap::new();
        aliases.insert("house sans".to_string(), "Noto Sans".to_string());
        let engine = DegradedEngine::from_font_sources_with_aliases(vec![bytes], aliases);
        let db = engine.svg_font_database();
        let names: Vec<&str> = db
            .faces()
            .flat_map(|f| f.families.iter().map(|(n, _)| n.as_str()))
            .collect();
        assert!(names.contains(&"Noto Sans"), "{names:?}");
        assert!(names.contains(&"house sans"), "{names:?}");
        // The generic sans family names a face the database has (the engine's settled pick, or an
        // OS default that is present).
        let sans = db.family_name(&fontdb::Family::SansSerif);
        assert!(names.contains(&sans), "{sans} in {names:?}");
    }

    /// and the export must pick the faces the editor's canvas picks for the glyphs Fraunces lacks. The editor's
    /// engine loads the bundled fallback set first (VFS `_degraded/NN-…`, sorted before every catalog face) beside
    /// only the faces the scene was fed; the desktop's render fonts dir lists every bundled family and then the
    /// fallback set as `_fallback/<file>`. Loaded in manifest order, the export drew `→` in Anton and `✓` in IBM
    /// Plex Mono while the canvas drew both, and `⑩`, in Noto Sans CJK SC. Also: a variable face reports the
    /// weight it is drawn at (Fraunces 400, not its default instance's 900).
    #[test]
    #[ignore = "needs unported Moda fixture editor font bundle (Fraunces variable, CJK, emoji, math and symbols)"]
    fn export_and_canvas_pick_the_same_faces_beside_a_fallback_tier_glyph() {
        let repo = fixtures_root().parent().unwrap().to_path_buf();
        let public = repo.join("packages/editor/public");
        let fallback = public.join("wasm-degraded-fonts");
        let fraunces = public.join("fonts/google/fraunces/Fraunces[SOFT,WONK,opsz,wght].ttf");
        let math = public.join("fonts/noto/NotoSansMath-Regular.ttf");
        let symbols = public.join("fonts/noto/NotoSansSymbols2-Regular.ttf");

        // The editor: `_degraded/` faces (the scene has CJK and dingbats, so CJK SC and emoji are installed), then
        // the faces fed for the scene, in VFS path order.
        let canvas_files = [
            fallback.join("NotoSans-Regular.ttf"),
            fallback.join("NotoSans-Bold.ttf"),
            fallback.join("NotoSansArabic-Regular.ttf"),
            fallback.join("NotoSansCJKsc-Regular.otf"),
            fallback.join("NotoColorEmoji.ttf"),
            fraunces.clone(),
            math.clone(),
            symbols.clone(),
        ];
        // `from_font_sources` without its OS fonts (a no-op on wasm, not here).
        let mut db = fontdb::Database::new();
        for path in &canvas_files {
            db.load_font_data(std::fs::read(path).unwrap());
        }
        let canvas = DegradedEngine::from_db(db);

        // The export: `writeRenderFontsDir`'s manifest — bundled families first, the fallback set last.
        let dir = std::env::temp_dir().join(format!("degraded-1761-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let key = |family: &str, path: &std::path::Path| {
            serde_json::json!({ "family": family, "weight": 400, "style": "normal", "vectorOnly": true, "ttfPath": path })
        };
        let mut keys = serde_json::Map::new();
        for (family, path) in [
            ("Anton", public.join("fonts/google/anton/Anton-Regular.ttf")),
            ("Fraunces", fraunces.clone()),
            ("IBM Plex Mono", public.join("fonts/google/ibmplexmono/IBMPlexMono-Regular.ttf")),
            ("Noto Sans Math", math.clone()),
            ("Noto Sans Symbols 2", symbols.clone()),
        ] {
            keys.insert(format!("{family}-400-normal"), key(family, &path));
        }
        for file in [
            "NotoColorEmoji.ttf",
            "NotoSans-Bold.ttf",
            "NotoSans-BoldItalic.ttf",
            "NotoSans-Italic.ttf",
            "NotoSans-Regular.ttf",
            "NotoSansArabic-Regular.ttf",
            "NotoSansCJKsc-Regular.otf",
        ] {
            let family = file.rsplit_once('.').unwrap().0;
            keys.insert(format!("_fallback/{file}"), key(family, &fallback.join(file)));
        }
        let manifest = serde_json::json!({ "systemFonts": false, "keys": keys });
        std::fs::write(dir.join("MANIFEST.json"), manifest.to_string()).unwrap();
        let export = DegradedEngine::load_with_catalogs(None, &[&dir]);

        let text = "Roadmap → ⑩ shipped ✓";
        let faces = |engine: &DegradedEngine| -> Vec<(String, Option<String>, u16)> {
            engine
                .resolve_faces("Fraunces", 400, false, text)
                .expect("resolve")
                .into_iter()
                .filter(|c| !text[c.start..c.end].trim().is_empty())
                .map(|c| (text[c.start..c.end].to_string(), c.family, c.weight))
                .collect()
        };
        let (canvas, export) = (faces(&canvas), faces(&export));
        std::fs::remove_dir_all(&dir).ok();
        for (part, family, weight) in &canvas {
            let want = if part.chars().all(|c| c.is_ascii_alphabetic()) {
                "Fraunces"
            } else {
                "Noto Sans CJK SC"
            };
            assert_eq!(family.as_deref(), Some(want), "canvas {part:?}: {canvas:?}");
            assert_eq!(*weight, 400, "canvas {part:?}: {canvas:?}");
        }
        assert_eq!(export, canvas, "export vs canvas faces");
    }

    /// Text in the default family that lands on this tier (a Hebrew word in it) draws its Latin and digits in Noto
    /// Sans, the vector lane's face for the default family, when no OS fonts are loaded (the desktop's render fonts
    /// dir, `systemFonts: false`) — not in whichever face the set lists first (a display face) and the emoji face,
    /// which is where cosmic-text's macOS fallback lists, naming only OS fonts, left them. `serif` too, with no Noto
    /// Serif in the set.
    #[test]
    #[ignore = "needs unported Moda fixture full Fraunces Bold and fonts-degraded/NotoColorEmoji.ttf"]
    fn the_default_family_draws_in_noto_sans_without_os_fonts() {
        let root = fixtures_root().join("fixtures");
        let fonts = root.join("oracle/fonts");
        let dir =
            std::env::temp_dir().join(format!("degraded-generic-families-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let key = |family: &str, path: std::path::PathBuf| {
            serde_json::json!({ "family": family, "weight": 400, "style": "normal", "vectorOnly": true, "ttfPath": path })
        };
        let manifest = serde_json::json!({ "systemFonts": false, "keys": {
            "a-display": key("Fraunces", fonts.join("fraunces/Fraunces[SOFT,WONK,opsz,wght]-Bold.ttf")),
            "b-emoji": key("NotoColorEmoji", root.join("fonts-degraded/NotoColorEmoji.ttf")),
            "c-noto": key("Noto Sans", fonts.join("noto-sans/NotoSans-Regular.ttf")),
            "d-hebrew": key("Noto Sans Hebrew", fonts.join("noto-sans-hebrew/NotoSansHebrew[wdth,wght]-Regular.ttf")),
        }});
        std::fs::write(dir.join("MANIFEST.json"), manifest.to_string()).unwrap();
        let engine = DegradedEngine::load_with_catalogs(None, &[&dir]);
        let text = "Price שלום 12.00 total";
        for family in ["", "sans-serif", "serif"] {
            let clusters = engine.resolve_faces(family, 400, false, text).expect("resolve");
            for c in &clusters {
                let part = &text[c.start..c.end];
                if part.trim().is_empty() {
                    continue;
                }
                let want = if part.chars().any(|ch| ('\u{0590}'..='\u{05FF}').contains(&ch)) {
                    "Noto Sans Hebrew"
                } else {
                    "Noto Sans"
                };
                assert_eq!(
                    c.family.as_deref(),
                    Some(want),
                    "{family:?}: {part:?} in {clusters:?}"
                );
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `layout_json` reports the lines the tier draws, and they wrap at the NODE's width, not the padded
    /// texture's: a box fitted to its widest line keeps its lines (docs/qa/text-editing.md, shrink-wrap).
    /// Thai has no spaces between words: lines break at the dictionary's word boundaries, never inside a word
    /// (it broke per character: "ร้า|น", "ครอบครัวต|ลอด").
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn thai_wraps_between_words() {
        let engine = test_engine();
        let text = "งานนี้รวมร้านอาหารริมทางชื่อดังจากกรุงเทพฯ มาพร้อมดนตรีสดและกิจกรรมสำหรับครอบครัวตลอดทั้งคืน";
        let words: Vec<String> = text
            .split(' ')
            .flat_map(|w| crate::sea_words::segment_sea_words(w, Default::default()))
            .collect();
        // Every place a line may end: after a whole word.
        let mut ends = std::collections::HashSet::new();
        let mut joined = String::new();
        for w in &words {
            joined.push_str(w);
            ends.insert(joined.clone());
        }
        for width in [120.0, 160.0, 210.0] {
            let spec = TextNodeSpec {
                node_id: "t".to_string(),
                html: format!("<p><span style=\"font-size: 16px\">{text}</span></p>").into(),
                width: Some(width),
                height: Some(400.0),
                ..Default::default()
            };
            let layout = engine.layout_json(&spec).expect("layout");
            assert_eq!(layout["engine"], "degraded");
            let lines: Vec<String> = layout["paragraphs"][0]["lines"]
                .as_array()
                .unwrap()
                .iter()
                .map(|l| l["segments"].as_array().unwrap().iter().map(|s| s["text"].as_str().unwrap()).collect::<String>())
                .collect();
            assert!(lines.len() > 2, "{width}px: {lines:?}");
            let mut so_far = String::new();
            for line in &lines[..lines.len() - 1] {
                so_far.push_str(&line.replace(' ', ""));
                assert!(ends.contains(&so_far), "{width}px: a line ends inside a word: {lines:?}");
            }
        }
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn layout_json_wraps_at_the_node_width_and_survives_a_fit() {
        let engine = test_engine();
        let spec = |width: f64| {
            TextNodeSpec {
            node_id: "t".to_string(),
            html: "<p><span style=\"font-size: 20px\">The quick brown fox jumps over a lazy dog as it is in a big old den</span></p>".into(),
            width: Some(width),
            height: Some(200.0),
            ..Default::default()
        }
        };
        let lines_of = |layout: &serde_json::Value| -> Vec<(String, f64)> {
            layout["paragraphs"][0]["lines"]
                .as_array()
                .unwrap()
                .iter()
                .map(|l| {
                    let text: String = l["segments"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|s| s["text"].as_str().unwrap())
                        .collect();
                    (text.trim().to_string(), l["totalWidth"].as_f64().unwrap())
                })
                .collect()
        };
        // Many widths, so some line has room for a short word inside the texture's 24px of padding.
        for width in (120..=320).step_by(9).map(f64::from) {
            let layout = engine.layout_json(&spec(width)).expect("layout");
            assert_eq!(layout["engine"], "degraded");
            let lines = lines_of(&layout);
            for (text, w) in &lines {
                assert!(
                    *w <= width + 1e-6,
                    "line {text:?} is {w}px, wider than the {width}px node"
                );
            }
            let joined: Vec<&str> = lines
                .iter()
                .flat_map(|(t, _)| t.split_whitespace())
                .collect();
            assert_eq!(
                joined.join(" "),
                "The quick brown fox jumps over a lazy dog as it is in a big old den"
            );

            // Fit the node to its widest line: the same lines come back.
            let widest = lines.iter().map(|(_, w)| *w).fold(0.0, f64::max).ceil();
            let fitted = lines_of(&engine.layout_json(&spec(widest)).expect("layout"));
            let texts = |v: &[(String, f64)]| v.iter().map(|(t, _)| t.clone()).collect::<Vec<_>>();
            assert_eq!(
                texts(&fitted),
                texts(&lines),
                "fitting the {width}px box to {widest}px re-wrapped it"
            );
        }
    }

    /// Letter-spaced words keep their spaces (text the vector lane declines: an emoji, a complex script, a face it
    /// can't read). The wrap measures a line word by word and each word dropped its trailing letter gap, so a word
    /// gap (a space, no gaps) was narrower than the letter gaps around it: `BIG WORLD` measured 1 px narrower than
    /// `BIGWORLD` and drew as one word.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn letter_spaced_word_spaces_keep_their_width() {
        let engine = test_engine();
        let width = |text: &str, spacing: f64| {
            let spec = TextNodeSpec {
                node_id: "t".to_string(),
                html: format!(
                    "<p><span style=\"font-size: 20px; letter-spacing: {spacing}px; text-transform: uppercase\">{text}</span></p>"
                ).into(),
                width: Some(2000.0),
                height: Some(200.0),
                ..Default::default()
            };
            let layout = engine.layout_json(&spec).expect("layout");
            layout["paragraphs"][0]["lines"][0]["totalWidth"]
                .as_f64()
                .unwrap()
        };
        let (spaced, joined) = (width("big world", 6.0), width("bigworld", 6.0));
        // One more grapheme: this font's space advance and one more 6 px gap.
        // The vendored face is not required to have a quarter-em space.
        let space = width("big world", 0.0) - width("bigworld", 0.0);
        assert!(
            (spaced - joined - (space + 6.0)).abs() < 0.5,
            "\"BIG WORLD\" is {spaced}px, \"BIGWORLD\" {joined}px: the space has no width"
        );
    }

    /// The engine slot is process-global, so every test that swaps it must run alone.
    /// (Rust runs tests in parallel by default; without this they would clobber each
    /// other's installs and fail intermittently rather than meaningfully.)
    static SLOT: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The tier used to be a lazy singleton — built once per process from
    /// whatever fonts dir was ambient at the FIRST complex-script render, and never
    /// replaceable. A warm worker renders many canvases, so every later one drew its
    /// Arabic against the first one's fonts (in prod: the baked Latin-complete set, so
    /// a customer's Cairo came back as vendored Noto with the job reporting success).
    /// These pin the property that fixes it: the installed engine is per-canvas, and
    /// nothing survives the canvas it belongs to.
    #[test]
    fn install_replaces_the_lazily_discovered_engine() {
        let _g = SLOT.lock().unwrap_or_else(|e| e.into_inner());
        uninstall_engine();

        let discovered = engine().expect("lazy discovery");
        install_engine(build_catalog_engine(
            &fixtures_root().join("fixtures/oracle/fonts"),
        ));
        let installed = engine().expect("installed");

        assert!(
            !std::sync::Arc::ptr_eq(&discovered, &installed),
            "installing a per-canvas engine did not replace the engine — this is the ENG-4288 \
             singleton: the first render's fonts would serve every later canvas"
        );
        uninstall_engine();
    }

    #[test]
    fn a_second_install_replaces_the_first() {
        let _g = SLOT.lock().unwrap_or_else(|e| e.into_inner());
        let oracle = fixtures_root().join("fixtures/oracle/fonts");

        install_engine(build_catalog_engine(&oracle));
        let first = engine().expect("first canvas");
        install_engine(build_catalog_engine(&oracle));
        let second = engine().expect("second canvas");

        // Same DIR, but a genuinely re-installed engine: the warm lane installs
        // per canvas, and a stale Arc here would mean canvas N drew with canvas N-1's
        // fonts whenever their keys differed.
        assert!(
            !std::sync::Arc::ptr_eq(&first, &second),
            "the second install reused the first engine"
        );
        uninstall_engine();
    }

    #[test]
    fn uninstall_restores_ambient_discovery() {
        let _g = SLOT.lock().unwrap_or_else(|e| e.into_inner());
        install_engine(build_catalog_engine(
            &fixtures_root().join("fixtures/oracle/fonts"),
        ));
        let installed = engine().expect("installed");

        uninstall_engine();
        let after = engine().expect("rediscovered");

        // The cross-canvas isolation half: a canvas with no catalog must NOT keep
        // rendering against the previous canvas's fonts.
        assert!(
            !std::sync::Arc::ptr_eq(&installed, &after),
            "uninstall left the previous canvas's engine installed"
        );
        uninstall_engine();
    }

    /// The symmetric coverage check for the OTHER font system.
    ///
    /// moda-py validates its TextHost against `materialized.resolved` and then builds
    /// this engine from the same dir at a LATER moment. A concurrent same-key writer
    /// can replace the dir with a partial snapshot in between, so the host passes while
    /// the engine quietly loads fewer faces — and caching that pair as clean pins the
    /// substitution for every later render of the canvas. This is what lets the caller
    /// notice.
    ///
    /// Three cases, because the interesting one is the key that is simply ABSENT: a
    /// naive "declared but did not load" check misses exactly the partial-manifest race
    /// it exists for.
    #[test]
    fn unresolved_catalog_keys_reports_the_partial_writer() {
        let root = fixtures_root();
        let face = root.join("crates/moda-engine/tests/fixtures/fonts/noto-sans/NotoSans-Regular.ttf");
        assert!(face.is_file(), "fixture missing: {}", face.display());

        let dir = std::env::temp_dir().join(format!(
            "moda-cover-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("f")).expect("dir");
        std::fs::copy(&face, dir.join("f/face.ttf")).expect("copy");
        // Present-and-loadable, present-but-file-missing, and present-without-a-ttf.
        std::fs::write(
            dir.join("MANIFEST.json"),
            r#"{"keys":{
                 "Good-400-normal":{"family":"Good","ttfPath":"f/face.ttf"},
                 "Gone-400-normal":{"family":"Gone","ttfPath":"f/missing.ttf"},
                 "MsdfOnly-400-normal":{"family":"MsdfOnly","metricsPath":"f/m.json"}
               }}"#,
        )
        .expect("manifest");

        let engine = compose_catalog_engine(None, None, &dir);
        let required = |ids: &[&str]| ids.iter().map(|s| s.to_string()).collect::<Vec<_>>();

        assert!(
            engine
                .unresolved_catalog_keys(&required(&["Good-400-normal"]))
                .is_empty(),
            "a key whose face loaded must not be reported"
        );
        assert_eq!(
            engine.unresolved_catalog_keys(&required(&["Gone-400-normal"])),
            vec!["Gone-400-normal".to_string()],
            "a declared ttf that would not read must be reported"
        );
        assert!(
            engine
                .unresolved_catalog_keys(&required(&["MsdfOnly-400-normal"]))
                .is_empty(),
            "an MSDF-only key is legitimately faceless here and must NOT be reported"
        );
        // THE RACE: the host resolved a key this manifest never listed.
        assert_eq!(
            engine.unresolved_catalog_keys(&required(&["Vanished-400-normal"])),
            vec!["Vanished-400-normal".to_string()],
            "a key absent from the manifest is the partial-writer case and must be reported"
        );

        // TWO KEYS, ONE FILE — the multi-weight catalog shape. The reader only runs for
        // the first of them, so marking "loaded" there left every later key looking
        // absent: a perfectly good catalog reported incomplete and never cached.
        let shared = std::env::temp_dir().join(format!(
            "moda-shared-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&shared);
        std::fs::create_dir_all(shared.join("f")).expect("dir");
        std::fs::copy(&face, shared.join("f/face.ttf")).expect("copy");
        std::fs::write(
            shared.join("MANIFEST.json"),
            r#"{"keys":{
                 "Shared-400-normal":{"family":"Shared","ttfPath":"f/face.ttf"},
                 "Shared-700-normal":{"family":"Shared","ttfPath":"f/face.ttf"}
               }}"#,
        )
        .expect("manifest");
        let shared_engine = compose_catalog_engine(None, None, &shared);
        assert!(
            shared_engine
                .unresolved_catalog_keys(&required(&["Shared-400-normal", "Shared-700-normal"]))
                .is_empty(),
            "two keys sharing one TTF must BOTH count as loaded, or a multi-weight catalog \
             never caches and re-materializes on every render"
        );
        let _ = std::fs::remove_dir_all(&shared);

        // A fully empty partial manifest must not masquerade as a catalog-less
        // engine when the successful caller has concrete resolved requirements.
        for manifest in [Some(r#"{"keys":{},"files":{}}"#), None] {
            if let Some(manifest) = manifest {
                std::fs::write(dir.join("MANIFEST.json"), manifest).unwrap();
            } else {
                std::fs::remove_file(dir.join("MANIFEST.json")).unwrap();
            }
            let empty = compose_catalog_engine(None, None, &dir);
            assert_eq!(
                empty.unresolved_catalog_keys(&required(&["Good-400-normal"])),
                vec!["Good-400-normal"]
            );
            assert!(empty.unresolved_catalog_keys(&[]).is_empty());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A canvas's own family must win even when its face needs NO alias of its own.
    ///
    /// `load_catalog_fonts` emits an alias only when the face's internal family differs
    /// from the catalog name, so a canvas whose face is named like its family
    /// contributes no entry — and `extend` cannot override with a key that is absent.
    /// The baked layer's redirect then survives underneath and keeps winning. Today the
    /// baked set emits exactly one such redirect, `inter` -> `Inter 24pt`, and Inter is
    /// the product's default family, so a team font of that name is the live case.
    ///
    /// Modelled with real faces: a synthetic BAKED layer that redirects "Noto Sans" to
    /// Lato's face, against a per-canvas layer supplying Noto Sans under its own name
    /// (so it emits nothing). Only clearing the superseded redirect can make the render
    /// come out as Noto Sans.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn a_canvas_family_that_needs_no_alias_still_clears_the_baked_redirect() {
        let root = fixtures_root();
        let lato = root.join(crate::world_type_fonts::WORLD_TYPE_LATO);
        let mont = root.join("crates/moda-engine/tests/fixtures/fonts/noto-sans/NotoSans-Regular.ttf");
        assert!(lato.is_file() && mont.is_file(), "fixtures missing");

        let base = std::env::temp_dir().join(format!(
            "moda-alias-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let mk = |name: &str, src: &std::path::Path, family: &str| {
            let d = base.join(name);
            std::fs::create_dir_all(d.join("f")).expect("dir");
            std::fs::copy(src, d.join("f/face.ttf")).expect("copy");
            std::fs::write(
                d.join("MANIFEST.json"),
                format!(
                    r#"{{"keys":{{"{family}-400-normal":{{"family":"{family}","ttfPath":"f/face.ttf"}}}}}}"#
                ),
            )
            .expect("manifest");
            d
        };
        // Baked: claims "Noto Sans" but is backed by Lato -> emits noto sans -> Lato.
        let baked_dir = mk("baked", &lato, "Noto Sans");
        // Per-canvas: real Noto Sans under its own name -> emits NOTHING.
        let per = mk("per", &mont, "Noto Sans");

        let baked_aliases = load_catalog_fonts(&mut fontdb::Database::new(), &baked_dir).aliases;
        assert_eq!(
            baked_aliases.get("noto sans").map(String::as_str),
            Some("Lato"),
            "fixture is wrong: the baked layer must actually emit a redirect"
        );
        let per_aliases = load_catalog_fonts(&mut fontdb::Database::new(), &per).aliases;
        assert!(
            !per_aliases.contains_key("noto sans"),
            "fixture is wrong: the per-canvas face must need no alias, or nothing is tested"
        );

        let baked = BakedCatalog {
            sources: vec![std::sync::Arc::new(std::fs::read(&lato).expect("read"))],
            aliases: baked_aliases,
            dir: baked_dir,
        };
        let html =
            "<p style=\"font-family: Noto Sans; font-size: 40px; color: #111\">Handgloves</p>";
        let got = compose_catalog_engine(None, Some(&baked), &per)
            .render_node(&spec(html, 900.0, 90.0))
            .expect("render")
            .rgba;
        let want_noto_sans = compose_catalog_engine(None, None, &per)
            .render_node(&spec(html, 900.0, 90.0))
            .expect("render")
            .rgba;

        assert_eq!(
            got, want_noto_sans,
            "the baked redirect survived and shaped the text in Lato — a canvas supplying \
             its own copy of a redirected family silently gets the wrong typeface"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// A face that does not ARRIVE must not cost the baked redirect.
    ///
    /// The clearing keys off families the canvas actually supplied. Keyed on merely
    /// DECLARED families instead — which is how I first wrote it — a manifest entry
    /// whose bytes do not read (the partial-writer race) removes the baked redirect and
    /// puts nothing in its place, so the family resolves to a generic sans. That is
    /// worse than leaving the baked face alone, which would at least have shaped the
    /// text in the right typeface.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn a_declared_but_unloadable_face_leaves_the_baked_redirect_alone() {
        let root = fixtures_root();
        let lato = root.join(crate::world_type_fonts::WORLD_TYPE_LATO);
        assert!(lato.is_file(), "fixture missing: {}", lato.display());

        let base = std::env::temp_dir().join(format!(
            "moda-unloadable-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&base);

        // Baked: redirects "Montserrat" to Lato's face.
        let baked_dir = base.join("baked");
        std::fs::create_dir_all(baked_dir.join("f")).expect("dir");
        std::fs::copy(&lato, baked_dir.join("f/face.ttf")).expect("copy");
        std::fs::write(
            baked_dir.join("MANIFEST.json"),
            r#"{"keys":{"Montserrat-400-normal":{"family":"Montserrat","ttfPath":"f/face.ttf"}}}"#,
        )
        .expect("manifest");
        let baked_aliases = load_catalog_fonts(&mut fontdb::Database::new(), &baked_dir).aliases;
        assert_eq!(
            baked_aliases.get("montserrat").map(String::as_str),
            Some("Lato"),
            "fixture is wrong: the baked layer must emit a redirect to clear"
        );

        // Per-canvas: DECLARES Montserrat, but its ttf does not exist.
        let per = base.join("per");
        std::fs::create_dir_all(&per).expect("dir");
        std::fs::write(
            per.join("MANIFEST.json"),
            r#"{"keys":{"Montserrat-400-normal":{"family":"Montserrat","ttfPath":"f/gone.ttf"}}}"#,
        )
        .expect("manifest");
        assert!(
            load_catalog_fonts(&mut fontdb::Database::new(), &per)
                .loaded_families
                .is_empty(),
            "fixture is wrong: nothing must load, or the test proves nothing"
        );

        let baked = BakedCatalog {
            sources: vec![std::sync::Arc::new(std::fs::read(&lato).expect("read"))],
            aliases: baked_aliases,
            dir: baked_dir,
        };
        let html =
            "<p style=\"font-family: Montserrat; font-size: 40px; color: #111\">Handgloves</p>";
        let with_failed_layer = compose_catalog_engine(None, Some(&baked), &per)
            .render_node(&spec(html, 900.0, 90.0))
            .expect("render")
            .rgba;
        let baked_alone = compose_catalog_engine(None, Some(&baked), &base.join("empty-missing"))
            .render_node(&spec(html, 900.0, 90.0))
            .expect("render")
            .rgba;

        assert_eq!(
            with_failed_layer, baked_alone,
            "the baked redirect was cleared by a face that never arrived — the family now \
             resolves to a generic sans instead of the typeface it had before"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// The baked catalog is SHARED, not copied into every engine.
    ///
    /// Layering it in per engine is what keeps rare-script coverage, but the baked set
    /// is 49.5 MB on disk (measured; ~30 MB of it four CJK variable fonts). Read fresh
    /// per engine and charged to the host LRU's 256 MB budget, that would collapse the
    /// warm cache to four or five entries and defeat the lane it serves. So the blobs
    /// are read once per process and handed out as shared `Arc`s, and only the
    /// PER-CANVAS bytes are charged.
    ///
    /// Both directions matter: charging zero for everything would "fix" the budget by
    /// making it blind, so the per-canvas side is asserted exactly.
    #[test]
    #[ignore = "needs unported Moda fixture oracle/fonts baked catalog manifest and Montserrat-Regular.ttf"]
    fn the_baked_catalog_is_shared_and_only_per_canvas_bytes_are_charged() {
        let _g = SLOT.lock().unwrap_or_else(|e| e.into_inner());
        let root = fixtures_root();
        std::env::set_var("MODA_FONTS_DIR", root.join("fixtures/oracle/fonts"));
        let face = root.join("fixtures/oracle/fonts/montserrat/Montserrat-Regular.ttf");
        assert!(face.is_file(), "fixture missing: {}", face.display());

        let base = std::env::temp_dir().join(format!(
            "moda-bytes-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let empty = base.join("empty");
        std::fs::create_dir_all(&empty).expect("dir");
        let per = base.join("per");
        std::fs::create_dir_all(per.join("f")).expect("dir");
        std::fs::copy(&face, per.join("f/face.ttf")).expect("copy");
        std::fs::write(
            per.join("MANIFEST.json"),
            r#"{"keys":{"Solo Family-400-normal":{"family":"Solo Family","ttfPath":"f/face.ttf"}}}"#,
        )
        .expect("manifest");

        assert_eq!(
            build_catalog_engine(&empty).catalog_bytes(),
            0,
            "the shared baked catalog was charged to this engine — with ~50 MB per entry \
             the host LRU collapses to a handful of canvases"
        );
        let face_len = std::fs::metadata(per.join("f/face.ttf"))
            .expect("meta")
            .len();
        assert_eq!(
            build_catalog_engine(&per).catalog_bytes(),
            face_len,
            "the per-canvas face was not charged — a budget that counts nothing is not a budget"
        );

        let _ = std::fs::remove_dir_all(&base);
        uninstall_engine();
    }

    /// A per-canvas family must BEAT the baked one of the same name.
    ///
    /// This is the layering's whole point and the half a tie-free test cannot reach:
    /// `a_per_canvas_family_reaches_the_tier_and_changes_the_glyphs` uses a name nothing
    /// else supplies, so it never forms a collision. Here the per-canvas catalog claims
    /// "Lato" — a family the baked oracle set really has — backed by Montserrat's face,
    /// so the only way the render can differ from the baked one is if the per-canvas
    /// entry won.
    ///
    /// Goes through `build_catalog_engine`, the production entry point, rather than a
    /// helper: after the shared-baked-blob change that is where load ORDER is decided,
    /// and a helper-level test would no longer cover it.
    #[test]
    #[ignore = "needs unported Moda fixture montserrat/Montserrat-Regular.ttf"]
    fn a_per_canvas_family_beats_the_baked_one_of_the_same_name() {
        let _g = SLOT.lock().unwrap_or_else(|e| e.into_inner());
        let root = fixtures_root();
        let montserrat = root.join("fixtures/oracle/fonts/montserrat/Montserrat-Regular.ttf");
        assert!(
            montserrat.is_file(),
            "fixture missing: {}",
            montserrat.display()
        );
        std::env::set_var("MODA_FONTS_DIR", root.join("fixtures/oracle/fonts"));

        let base = std::env::temp_dir().join(format!(
            "moda-override-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let over = base.join("over");
        std::fs::create_dir_all(over.join("f")).expect("dir");
        std::fs::copy(&montserrat, over.join("f/face.ttf")).expect("copy");
        // Claims the BAKED family name, backed by a different face.
        std::fs::write(
            over.join("MANIFEST.json"),
            r#"{"keys":{"Lato-400-normal":{"family":"Lato","ttfPath":"f/face.ttf"}}}"#,
        )
        .expect("manifest");
        let plain = base.join("plain");
        std::fs::create_dir_all(&plain).expect("dir");

        let html = "<p style=\"font-family: Lato; font-size: 40px; color: #111\">Handgloves</p>";
        let overridden = build_catalog_engine(&over)
            .render_node(&spec(html, 900.0, 90.0))
            .expect("override render")
            .rgba;
        let baked_only = build_catalog_engine(&plain)
            .render_node(&spec(html, 900.0, 90.0))
            .expect("baked render")
            .rgba;

        assert_ne!(
            overridden, baked_only,
            "the per-canvas 'Lato' did not win — the baked face still shaped the text, so a \
             canvas overriding a baked family silently gets the wrong typeface"
        );
        let _ = std::fs::remove_dir_all(&base);
        uninstall_engine();
    }

    #[test]
    #[ignore = "needs unported Moda fixture fonts-degraded fallback directory"]
    fn a_catalog_engine_keeps_the_noto_fallback_set() {
        let _g = SLOT.lock().unwrap_or_else(|e| e.into_inner());
        // The Noto fallback set is a PROCESS property and must survive a per-canvas
        // catalog install, or pointing the tier at a canvas's fonts would take away the
        // faces that cover complex scripts — trading one substitution bug for a worse
        // one (tofu).
        //
        // Isolated to the fallback dir ALONE — no catalog dirs and, crucially, NO SYSTEM
        // FONTS. Two things would otherwise supply this Arabic and leave the test green
        // with the fallback deleted: the ambient oracle catalog that layering now loads
        // (it carries NotoSansArabic), and any dev box whose OS ships Noto — mine does,
        // which is exactly how an earlier version of this assertion passed under that
        // mutation. `load_with_catalogs` always scans system fonts, so the db is built
        // by hand here, the same way `catalog_only_engine` does for the same reason.
        let mut db = fontdb::Database::new();
        db.load_fonts_dir(resolve_degraded_fallback_dir().expect("vendored fallback dir"));
        let e = DegradedEngine::from_db(db);

        let r = e
            .render_node(&spec(
                "<p style=\"font-family: Cairo; font-size: 30px\">مرحبا بالعالم</p>",
                1080.0,
                110.0,
            ))
            .expect("raster");
        let ink = r.rgba.chunks_exact(4).filter(|p| p[3] > 0).count();
        assert!(
            ink > 200,
            "Arabic drew no ink from an engine whose catalog dir is empty — the Noto \
             fallback set was lost, got {ink} px"
        );
    }

    fn spec(html: &str, w: f64, h: f64) -> TextNodeSpec {
        TextNodeSpec {
            node_id: "t".into(),
            html: html.into(),
            width: Some(w),
            height: Some(h),
            opacity: 1.0,
            ..Default::default()
        }
    }

    /// The ink's vertical centre in logical px from the raster's line box top.
    fn ink_centre_y(r: &DegradedRaster) -> f64 {
        let (mut sum, mut weight) = (0.0, 0.0);
        for (i, px) in r.rgba.chunks_exact(4).enumerate() {
            let a = px[3] as f64;
            sum += a * ((i / r.width as usize) as f64 + 0.5);
            weight += a;
        }
        assert!(weight > 0.0, "the text drew");
        sum / weight / r.quality_scale - r.pad_top
    }

    /// A line sits where its baseline puts it at every raster scale. Glyphs were placed at the whole texel above the
    /// pen (`LayoutGlyph::physical` truncates y), so a 2x raster drew a line up to half a px higher than an 8x one,
    /// and higher than the vector lane draws the same text.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn lines_sit_at_the_same_height_at_every_raster_scale() {
        let e = test_engine();
        for size in [17.0, 23.0, 31.0, 48.0] {
            let s = spec(&format!("<p style=\"font-size: {size}px; color: #ffffff\">שלום עולם ภาษาไทย</p>"), 600.0, size * 2.0);
            let fine = ink_centre_y(&e.render_node_at(&s, 8.0).unwrap());
            for scale in [2.0, 3.0, 4.0] {
                let r = e.render_node_at(&s, scale).unwrap();
                assert_eq!(r.quality_scale, scale);
                let y = ink_centre_y(&r);
                assert!((y - fine).abs() < 0.1, "{size} px at {scale}x: ink centre {y:.3} px, at 8x {fine:.3} px");
            }
        }
    }

    /// The transparent texels around white glyphs are white (alpha 0), so the linear sampler between an edge texel
    /// and its empty neighbour blends white, never black: drawn at another scale, edges had a dark fringe.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn transparent_texels_next_to_ink_carry_its_colour() {
        let e = test_engine();
        let r = e.render_node(&spec("<p style=\"font-size: 30px; color: #ffffff\">שלום עולם</p>", 400.0, 60.0)).unwrap();
        let (w, h) = (r.width as usize, r.height as usize);
        let alpha = |x: usize, y: usize| r.rgba[(y * w + x) * 4 + 3];
        let mut fringe = 0;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let i = (y * w + x) * 4;
                if alpha(x, y) == 0 && (alpha(x - 1, y) > 0 || alpha(x + 1, y) > 0 || alpha(x, y - 1) > 0 || alpha(x, y + 1) > 0) {
                    fringe += 1;
                    assert_eq!(&r.rgba[i..i + 4], &[255, 255, 255, 0], "texel ({x}, {y}) next to ink");
                }
            }
        }
        assert!(fringe > 100, "{fringe} transparent texels border the glyphs");
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn renders_cjk_fallback_text() {
        let e = test_engine();
        let r = e
            .render_node(&spec(
                "<p style=\"font-family: Inter; font-size: 30px\">漢字漢字漢字</p>",
                1080.0,
                110.0,
            ))
            .expect("raster");
        assert_eq!(r.quality_scale, 2.0);
        assert!(r.width > 0 && r.height > 0);
        // Ink present (some non-transparent pixels).
        let ink = r.rgba.chunks_exact(4).filter(|p| p[3] > 0).count();
        assert!(ink > 500, "expected CJK ink, got {ink} px");
        // logicalSize = ceil(intrinsic) + 12 horizontal texture padding.
        assert!(r.logical_w > 12.0 && r.logical_w <= 1080.0 + 12.0);
        assert_eq!(r.width, (r.logical_w * 2.0).floor() as u32);
    }

    /// Prod-faithful engine: ONLY the catalog (oracle) TTFs — no OS fonts, no
    /// dedicated fallback dir. Production loads exactly this into cosmic
    /// (`MODA_FONTS_DIR=/opt/moda/native/fonts`, which bakes the oracle font
    /// set — Dockerfile.agent-worker.base). Building the db with NO system
    /// fonts proves cosmic-text resolves the bundled Noto complex-script fonts
    /// by coverage, with nothing to mask a missing face.
    fn catalog_only_engine() -> DegradedEngine {
        let mut db = fontdb::Database::new();
        for bytes in oracle_ttfs() {
            db.load_font_data(bytes);
        }
        DegradedEngine::from_db(db)
    }

    #[test]
    #[ignore = "needs unported Moda fixture oracle/fonts/MANIFEST.json and broad-script font set"]
    fn cosmic_renders_complex_scripts_from_catalog_only() {
        // Every complex script a customer can type must (a) have a baked Noto
        // face whose cmap covers it in the prod-faithful catalog-only oracle,
        // and (b) produce ink on the cosmic engine — i.e. the face is found and
        // shaped, not another font's `.notdef` box (tofu). Shaping CORRECTNESS
        // (joining, bidi, conjuncts, marks) is proven by the rendered-image
        // evidence in the PR writeup; this pins that the fonts RESOLVE in prod.
        //
        // The complex-script set is `moda_text::complex_script` (the runs routed
        // to this cosmic lane). A face missing here renders tofu in prod, so the
        // coverage assertion is the real regression guard (ink alone can't tell
        // a shaped glyph from a `.notdef` box).
        //
        // LANE SCOPE: this is the baked catalog dir (MODA_FONTS_DIR=
        // /opt/moda/native/fonts) that the degraded engine loads on the warm
        // moda-py lane AND the non-catalog CLI lane — under NATIVE_RENDER_WARM_FONTS
        // (default ON) both preserve MODA_FONTS_DIR (the CLI lane only runs with
        // uses_catalog=False there), so both cover these scripts. The ONLY lane that
        // wouldn't is the catalog-materialized CLI subprocess (MODA_FONTS_DIR
        // stripped for the per-canvas --fonts dir), reached only under the
        // WARM_FONTS kill switch; carrying these vectorOnly fallback TTFs into
        // materialized dirs belongs with the moda-fonts materialize path (a separate
        // concern from the baked set) and is not a regression (that lane rendered
        // these scripts as tofu before this change too).
        let ttfs = oracle_ttfs();
        // Parse each baked face once, then reuse across the per-script coverage
        // checks (the fontdb `db` in `catalog_only_engine` owns its own copies).
        let faces: Vec<rustybuzz::Face<'_>> = ttfs
            .iter()
            .filter_map(|b| rustybuzz::Face::from_slice(b, 0))
            .collect();
        let e = catalog_only_engine();
        let base = "font-family: Inter; font-size: 30px; color: #111";
        // (name, sample text, a representative codepoint that MUST be covered)
        let cases: &[(&str, &str, u32)] = &[
            ("Arabic", "مرحبا بالعالم", 0x0645),
            ("Hebrew", "שלום עולם", 0x05E9),
            ("Devanagari", "नमस्ते दुनिया", 0x0928),
            ("Thai", "สวัสดีชาวโลก", 0x0E2A),
            ("Tamil", "வணக்கம் உலகம்", 0x0BB5),
            ("Telugu", "నమస్కారం", 0x0C28),
            ("Bengali", "নমস্কার বিশ্ব", 0x09A8),
            ("Khmer", "ជំរាបសួរ", 0x1787),
            ("mixed-bidi", "hello مرحبا world 123", 0x0645),
            // #7275 residual — rare complex scripts, baked TTF-only (vectorOnly):
            // routed to this lane by `complex_script` but previously unbaked, so
            // they rendered `.notdef` tofu in prod. Now covered → 100%.
            ("Syriac", "ܫܠܡܐ ܥܠܡܐ", 0x0710),
            ("Thaana", "ދިވެހިބަސް", 0x0780),
            ("NKo", "ߒߞߏ ߞߊ߲", 0x07C1),
            ("Samaritan", "ࠀࠁࠋ", 0x0800),
            ("Mandaic", "ࡀࡁࡂ", 0x0840),
            ("Adlam", "𞤀𞤁𞤂", 0x1E900),
            ("Mongolian", "ᠮᠣᠩᠭᠣᠯ", 0x1820),
            ("Brahmi", "𑀓𑀔𑀕", 0x11013),
            ("Limbu", "ᤁᤂᤃ", 0x1901),
            ("Tai Tham", "ᨠᨡᨢ", 0x1A20),
            ("Hanifi Rohingya", "𐴀𐴁𐴂", 0x10D00),
        ];
        for (name, text, cp) in cases {
            let ch = char::from_u32(*cp).unwrap();
            assert!(
                any_oracle_face_covers(&faces, ch),
                "{name}: no baked oracle face covers U+{cp:04X} — renders tofu in prod"
            );
            let html = format!("<p style=\"{base}\">{text}</p>");
            let r = e
                .render_node(&spec(&html, 900.0, 80.0))
                .unwrap_or_else(|e| panic!("{name}: render failed: {e}"));
            let ink = r.rgba.chunks_exact(4).filter(|p| p[3] > 0).count();
            assert!(
                ink > 500,
                "{name}: expected complex-script ink (font resolved), got {ink}px"
            );
        }
    }

    /// Marks drawn past the lines at a tight line-height (Devanagari's reph and i-matra, Thai tone marks over an
    /// upper vowel) stay in the texture: it grows above the first line by the ink there, and the quad moves up by as
    /// much, so the lines sit where they did. Text whose ink stays in its lines keeps the texture it had.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn marks_above_a_tight_line_stay_in_the_raster() {
        let e = test_engine();
        for text in ["कीर्ति हिंदी", "ญี่ปุ่น น้ำ"] {
            let html = format!("<p style=\"font-size: 48px; line-height: 0.9; color: #000\">{text}</p>");
            let node = spec(&html, 600.0, 60.0);
            let r = e.render_node(&node).expect("render");
            assert!(r.pad_top > 0.0, "{text}: no ink above the line");
            let row = r.width as usize * 4;
            assert!(r.rgba[..row].chunks_exact(4).any(|p| p[3] > 0), "{text}: the top row is the ink's");
            assert_eq!(r.height as usize * row, r.rgba.len());
            let (_, y) = raster_quad_offset(&node, &r);
            assert_eq!(y, -r.pad_top, "{text}: the lines stay at the node's top");
        }
        // Arabic also overflows below its default line box. Keep that ink, while a taller line is unchanged.
        let arabic = e
            .render_node(&spec("<p style=\"font-size: 30px; color: #000\">مرحبا بالعالم</p>", 600.0, 60.0))
            .expect("render");
        assert!(arabic.pad_bottom > 0.0);
        let last_row = (arabic.height - 1) as usize * arabic.width as usize * 4;
        assert!(arabic.rgba[last_row..].chunks_exact(4).any(|p| p[3] > 0));
        let plain = e
            .render_node(&spec("<p style=\"font-size: 30px; line-height: 2; color: #000\">مرحبا بالعالم</p>", 600.0, 60.0))
            .expect("render");
        assert_eq!((plain.pad_top, plain.pad_bottom), (0.0, 0.0));
        assert_eq!(plain.height, (plain.logical_h * plain.quality_scale).floor() as u32);
    }

    /// A below-base ra (ra-phala) is a conjunct, not a virama and a full ra. The Indic shaper finds the below-base
    /// consonant by asking whether `blwf` would substitute virama + ra; a font whose `blwf` is a format-3 context
    /// lookup (Noto Sans Bengali 3.011, the Google Fonts release) was told no, so গ্র drew as গ্‌র (a visible virama).
    /// harfrust's `would_apply` fix: `native/vendor/harfrust/MODA-PATCH.md`.
    #[test]
    #[ignore = "needs unported Moda fixture fonts-indic/NotoSansBengali-ContextFormat3Test.ttf"]
    fn format3_context_blwf_forms_ra_phala() {
        let font = std::fs::read(
            fixtures_root().join("fixtures/fonts-indic/NotoSansBengali-ContextFormat3Test.ttf"),
        )
        .unwrap();
        let e = DegradedEngine::from_font_data(vec![font]);
        let ink_width = |text: &str| {
            let style = "font-family: 'Noto Sans Bengali'; font-size: 48px; color: #111";
            let html = format!("<p style=\"{style}\">{text}</p>");
            let r = e
                .render_node(&spec(&html, 600.0, 80.0))
                .unwrap_or_else(|err| panic!("{text}: render failed: {err}"));
            let w = r.width as usize;
            let columns: Vec<usize> = r
                .rgba
                .chunks_exact(4)
                .enumerate()
                .filter(|(_, p)| p[3] > 0)
                .map(|(i, _)| i % w)
                .collect();
            columns.iter().max().unwrap() - columns.iter().min().unwrap()
        };
        // The conjunct (half ga + ra-phala) is narrower than ga, a visible virama and ra (a ZWNJ asks for those).
        for (joined, explicit) in [("গ্র", "গ্\u{200C}র"), ("প্রিয়", "প্\u{200C}রিয়")] {
            let (j, x) = (ink_width(joined), ink_width(explicit));
            assert!(j < x, "{joined}: ink {j} px, not narrower than the explicit-virama form ({x} px)");
        }
    }

    /// A ZERO WIDTH NON-JOINER is the author's "don't join here" (Persian هفت‌سین, می‌روم; a Devanagari
    /// half form क्‌ष): it reaches the shaper, so the word draws unjoined, not as if the ZWNJ were absent.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn zwnj_keeps_letters_unjoined() {
        let e = test_engine();
        let base = "font-family: Inter; font-size: 40px; color: #111";
        for (with, without) in [("هفت\u{200C}سین می\u{200C}روم", "هفتسین میروم"), ("क्\u{200C}ष", "क्ष")] {
            let draw = |text: &str| {
                e.render_node(&spec(&format!("<p style=\"{base}\">{text}</p>"), 600.0, 80.0))
                    .unwrap_or_else(|err| panic!("{text}: render failed: {err}"))
                    .rgba
            };
            assert_ne!(draw(with), draw(without), "{with:?} drew as if its ZWNJ were not there");
        }
    }

    /// A long-lived engine (`moda-render serve`, behind `look` and the run preview) draws text after many sizes as a
    /// fresh one does. swash 0.2.8's scale context kept an evicted hinting instance's old size once more than 8
    /// (face, size) pairs had been drawn, so a later size's new glyphs came out hinted at another size.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn text_after_many_sizes_draws_as_in_a_fresh_engine() {
        let draw = |e: &DegradedEngine, size: u32, text: &str| {
            let html = format!("<p style=\"font-family: Inter; font-size: {size}px; color: #111\">{text}</p>");
            e.render_node(&spec(&html, 900.0, 160.0))
                .unwrap_or_else(|err| panic!("{text} at {size}px: render failed: {err}"))
                .rgba
        };
        let long_lived = test_engine();
        for size in 20..34 {
            draw(&long_lived, size, "Handgloves शुरू");
        }
        // New glyphs at the first size: nothing of them is in the glyph cache yet, so they are scaled again.
        let text = "Quiz ₹1,299 से दिवाली";
        assert!(
            draw(&long_lived, 20, text) == draw(&test_engine(), 20, text),
            "after 14 sizes, new glyphs at 20px drew differently from a fresh engine (a stale hinting instance)"
        );
    }

    /// Every catalog TTF the prod worker bakes (`keys.*.ttfPath` in the oracle
    /// `MANIFEST.json` → `/opt/moda/native/fonts`) — what cosmic loads in prod.
    fn oracle_ttfs() -> Vec<Vec<u8>> {
        let catalog = fixtures_root().join("fixtures/oracle/fonts");
        let raw = std::fs::read_to_string(catalog.join("MANIFEST.json")).unwrap();
        let m: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let mut out = Vec::new();
        for entry in m["keys"].as_object().unwrap().values() {
            if let Some(rel) = entry.get("ttfPath").and_then(|p| p.as_str()) {
                if let Ok(bytes) = std::fs::read(catalog.join(rel)) {
                    out.push(bytes);
                }
            }
        }
        out
    }

    /// True when at least one baked oracle face's cmap covers `ch` — the exact
    /// condition cosmic-text needs to resolve a face by coverage instead of
    /// falling back to another font's `.notdef` (tofu). Faces are parsed once by
    /// the caller and reused across the per-script checks.
    fn any_oracle_face_covers(faces: &[rustybuzz::Face<'_>], ch: char) -> bool {
        faces.iter().any(|f| f.glyph_index(ch).is_some())
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn renders_default_sans_serif() {
        let e = test_engine();
        let r = e
            .render_node(&spec(
                "<p style=\"font-size: 30px; color: #1d2733\">Default sans-serif text.</p>",
                1080.0,
                110.0,
            ))
            .expect("raster");
        let ink = r.rgba.chunks_exact(4).filter(|p| p[3] > 0).count();
        assert!(ink > 500, "expected latin ink, got {ink} px");
        assert_eq!(r.fill_alpha, 1.0);
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn uniform_alpha_rasterizes_opaque_with_fill_alpha() {
        let e = test_engine();
        let r = e
            .render_node(&spec(
                "<p style=\"font-size: 40px; color: rgba(0, 0, 0, 0.5)\">Half alpha</p>",
                400.0,
                60.0,
            ))
            .expect("raster");
        // detectMixedAlpha=false → glyphs drawn OPAQUE, avg alpha in
        // fill_alpha (display-quad only; masks sample opaque pixels).
        assert!(
            (r.fill_alpha - 0.5).abs() < 0.01,
            "fill_alpha={}",
            r.fill_alpha
        );
        let max_a = r.rgba.chunks_exact(4).map(|p| p[3]).max().unwrap_or(0);
        assert!(max_a > 250, "raster should be opaque, max alpha {max_a}");
    }

    /// The node's solid text fill (`fill_override`, from `textFills`) colours every run, as on the vector lane:
    /// white over the HTML's black draws white, and translucent fills keep the uniform-alpha path.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn fill_override_colours_every_run() {
        let e = test_engine();
        let html = "<p style=\"font-size: 40px; color: #000000\">Fill <span style=\"color: #ff0000\">me</span></p>";
        let mut s = spec(html, 400.0, 60.0);
        s.fill_override = Some("#ffffff".into());
        let r = e.render_node(&s).expect("raster");
        let solid: Vec<&[u8]> = r.rgba.chunks_exact(4).filter(|p| p[3] > 250).collect();
        assert!(!solid.is_empty(), "no opaque glyph pixels");
        assert!(
            solid.iter().all(|p| p[0] > 250 && p[1] > 250 && p[2] > 250),
            "a glyph pixel kept its run colour under a white fill"
        );
        s.fill_override = Some("rgba(255, 255, 255, 0.5)".into());
        let r = e.render_node(&s).expect("raster");
        assert!((r.fill_alpha - 0.5).abs() < 0.01, "fill_alpha={}", r.fill_alpha);
        // An unreadable fill leaves the run colours.
        s.fill_override = Some("not a colour".into());
        let r = e.render_node(&s).expect("raster");
        assert!(r.rgba.chunks_exact(4).any(|p| p[3] > 250 && p[0] < 5 && p[1] < 5 && p[2] < 5));
    }

    /// Build a DegradedEngine from the bundled fallback set + catalog TTFs WITH
    /// NO system fonts — the exact fontdb the wasm editor's `from_font_data`
    /// path produces (`load_system_fonts` is a no-op on wasm32). This is the
    /// only configuration that proves the bundled NotoColorEmoji.ttf (not a
    /// macOS Apple-Color-Emoji system face) carries the color emoji, and that a
    /// non-catalog display alias gets a coherent fallback face.
    fn wasm_like_engine() -> DegradedEngine {
        let mut db = fontdb::Database::new();
        // Bundled fallback set in MANIFEST order (NotoColorEmoji LAST), then
        // catalog TTFs — mirroring the wasm `_degraded/NN-*` sort order.
        let degraded = fixtures_root().join("fixtures/fonts-degraded");
        for name in [
            "NotoSans-Regular.ttf",
            "NotoSans-Bold.ttf",
            "NotoSansArabic-Regular.ttf",
            "NotoSansCJKsc-Regular.otf",
            "NotoColorEmoji.ttf",
        ] {
            db.load_font_data(std::fs::read(degraded.join(name)).expect("bundled font"));
        }
        let catalog = fixtures_root().join("fixtures/oracle/fonts");
        if let Ok(raw) = std::fs::read_to_string(catalog.join("MANIFEST.json")) {
            if let Ok(m) = serde_json::from_str::<serde_json::Value>(&raw) {
                if let Some(keys) = m.get("keys").and_then(|k| k.as_object()) {
                    for entry in keys.values() {
                        if let Some(rel) = entry.get("ttfPath").and_then(|p| p.as_str()) {
                            if let Ok(bytes) = std::fs::read(catalog.join(rel)) {
                                db.load_font_data(bytes);
                            }
                        }
                    }
                }
            }
        }
        DegradedEngine::from_db(db)
    }

    #[test]
    #[ignore = "needs unported Moda fixture fonts-degraded/NotoColorEmoji.ttf"]
    fn wasm_emoji_rasterizes_in_color_without_system_fonts() {
        // Regression: the wasm degraded engine was fed ONLY catalog TTFs, so
        // emoji shaped to notdef (tofu). With the bundled NotoColorEmoji.ttf in
        // the fontdb (and no OS color-emoji font to mask the bug), the plain,
        // ZWJ, flag, and skin-tone clusters must all produce colored pixels.
        let e = wasm_like_engine();
        let r = e
            .render_node(&spec(
                "<p style=\"font-family: Inter; font-size: 36px; color: #1d2733\">\
                 e \u{1F3A8} \u{1F680} \u{1F354} \u{1F44B}\u{1F3FD} \u{1F1FA}\u{1F1F8}</p>",
                600.0,
                80.0,
            ))
            .expect("raster");
        let colored = r
            .rgba
            .chunks_exact(4)
            .filter(|p| p[3] > 128 && (p[0] as i32 - p[2] as i32).abs() > 30)
            .count();
        assert!(
            colored > 200,
            "expected colored emoji pixels (no OS fonts), got {colored}"
        );
    }

    #[test]
    #[ignore = "needs unported Moda fixture fonts-degraded/NotoColorEmoji.ttf"]
    fn emoji_cells_come_from_the_colour_face() {
        // The vector lane's emoji image: the bundled Noto Color Emoji glyph (CBDT), in a
        // 128 × 160 straight-alpha cell, for plain, skin-tone, ZWJ, flag and VS16 clusters.
        let e = wasm_like_engine();
        // (Noto draws the ZWJ family as grey silhouettes on a tile, so it is checked for
        // ink, not hue: one glyph, not three shrunk side by side, fills most of the cell.)
        let clusters = [
            ("\u{1F680}", true),
            ("\u{1F44B}\u{1F3FD}", true),
            ("\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}", false),
            ("\u{1F1EF}\u{1F1F5}", true),
            ("\u{2764}\u{FE0F}", true),
        ];
        for (cluster, hued) in clusters {
            let img = e.emoji_cell_rgba(cluster).unwrap_or_else(|| panic!("no cell for {cluster}"));
            assert_eq!((img.width, img.height), (EMOJI_CELL_TEXELS, 160));
            let colored = img
                .rgba
                .chunks_exact(4)
                .filter(|p| p[3] > 128 && (p[0] as i32 - p[2] as i32).abs() > 30)
                .count();
            let inked = img.rgba.chunks_exact(4).filter(|p| p[3] > 128).count();
            assert!(inked > 3000, "{cluster}: {inked} inked texels");
            assert!(!hued || colored > 500, "{cluster}: {colored} coloured texels");
            // Straight alpha: a soft edge texel keeps a colour brighter than its alpha.
            assert!(
                !hued || img.rgba.chunks_exact(4).any(|p| p[3] > 0 && p[3] < 128 && p[0].max(p[1]).max(p[2]) > p[3]),
                "{cluster}: straight alpha"
            );
            // Nothing spills past the one-em advance's cell edges.
            let ink_cols: Vec<usize> = (0..img.width as usize)
                .filter(|&x| (0..img.height as usize).any(|y| img.rgba[(y * img.width as usize + x) * 4 + 3] > 0))
                .collect();
            assert!(ink_cols.first() > Some(&0) && ink_cols.last() < Some(&(img.width as usize - 1)), "{cluster}");
        }
        // A letter has no colour glyph: no cell (the caller records the miss).
        assert!(e.emoji_cell_rgba("a").is_none());
        // Cached, and a new engine gets a new id.
        assert!(e.emoji_cell_rgba("\u{1F680}").is_some());
        assert_ne!(e.id(), wasm_like_engine().id());
    }

    /// An emoji in a run this tier shapes takes the colour face even where the asked family or an earlier face maps
    /// it: EB Garamond maps the regional indicators (as boxed letters), so 🇯🇵 drew as "J" "P" in boxes. A text
    /// presentation mark (©) stays in the text face.
    #[test]
    #[ignore = "needs unported Moda fixture NotoColorEmoji.ttf and EBGaramond[wght].ttf"]
    fn emoji_in_shaped_runs_take_the_colour_face() {
        let mut db = fontdb::Database::new();
        for name in ["NotoSans-Regular.ttf", "NotoColorEmoji.ttf"] {
            db.load_font_data(std::fs::read(fixtures_root().join("fixtures/fonts-degraded").join(name)).expect("font"));
        }
        let garamond = fixtures_root().join("../packages/editor/public/fonts/google/ebgaramond/EBGaramond[wght].ttf");
        db.load_font_data(std::fs::read(garamond).expect("EB Garamond"));
        let engine = DegradedEngine::from_db(db);
        let text = "Tokyo \u{1F1EF}\u{1F1F5} \u{1F525} \u{A9}";
        let clusters = engine.resolve_faces("EB Garamond", 400, false, text).expect("resolve");
        let face_of = |part: &str| {
            let at = text.find(part).unwrap();
            clusters.iter().find(|c| c.start == at).and_then(|c| c.family.clone()).unwrap_or_default()
        };
        assert_eq!(face_of("\u{1F1EF}"), "Noto Color Emoji", "{clusters:?}");
        assert_eq!(face_of("\u{1F525}"), "Noto Color Emoji", "{clusters:?}");
        assert!(face_of("T").starts_with("EB Garamond"), "{clusters:?}");
        assert!(face_of("\u{A9}").starts_with("EB Garamond"), "{clusters:?}");
    }

    #[test]
    #[ignore = "needs unported Moda fixture fonts-degraded/NotoColorEmoji.ttf"]
    fn emoji_effect_copies_follow_the_silhouette() {
        use crate::vector::EmojiEffect;
        let e = wasm_like_engine();
        let fx = |blur: f32, stroke: f32, additive: bool| EmojiEffect {
            color: [1.0, 0.0, 0.0, 1.0],
            gain: 1.0,
            blur_texels: blur,
            stroke_texels: stroke,
            pad_texels: (blur + stroke) as u32 + 1,
            additive,
        };
        let a = |img: &crate::host::RgbaImage, x: usize, y: usize| img.rgba[(y * img.width as usize + x) * 4 + 3];
        // Stroke: a ring just outside the silhouette, empty deep inside and far outside.
        let ring = e.emoji_effect_rgba("\u{1F534}", &fx(0.0, 4.0, false)).expect("red circle");
        assert_eq!((ring.width, ring.height), (64 + 10, 80 + 10));
        let (cx, cy) = (ring.width as usize / 2, 5 + 64 - 26); // inside the disc
        assert_eq!(a(&ring, cx, cy), 0, "stroke is outside only");
        assert_eq!(a(&ring, 0, 0), 0);
        let row: Vec<u8> = (0..ring.width as usize).map(|x| a(&ring, x, cy)).collect();
        assert!(row.iter().filter(|&&v| v > 200).count() >= 4, "two ring crossings: {row:?}");
        assert!(ring.rgba.chunks_exact(4).all(|p| p[0] == 255 && p[1] == 0 && p[2] == 0), "straight red");
        // Shadow blur: soft beyond the edge, solid inside.
        let soft = e.emoji_effect_rgba("\u{1F534}", &fx(6.0, 0.0, false)).unwrap();
        let row: Vec<u8> = (0..soft.width as usize).map(|x| a(&soft, x, cy)).collect();
        assert_eq!(a(&soft, soft.width as usize / 2, cy), 255);
        assert!(row.iter().filter(|&&v| v > 10 && v < 245).count() >= 8, "a ramp: {row:?}");
    }

    #[test]
    #[ignore = "needs unported Moda fixture fonts-degraded/NotoColorEmoji.ttf and NotoSansCJKsc-Regular.otf"]
    fn wasm_non_catalog_alias_has_coherent_advances() {
        // Regression (634d2b14 "Roboto Bk" smear): a non-catalog display alias
        // with no exact catalog face must fall back to a Noto face with REAL
        // metrics, not collapse to overlapping/zero advances. Measure a known
        // string and assert the rendered ink spans a plausible horizontal
        // extent (no crush-onto-each-other smear).
        let e = wasm_like_engine();
        let r = e
            .render_node(&spec(
                "<p style=\"font-family: Roboto Bk; font-size: 48px; color: #1d2733\">CONSUMERS</p>",
                900.0,
                90.0,
            ))
            .expect("raster");
        // Find the horizontal ink span (first..last column with an opaque pixel).
        let w = r.width as usize;
        let h = r.height as usize;
        let mut min_x = w;
        let mut max_x = 0usize;
        for y in 0..h {
            for x in 0..w {
                let a = r.rgba[(y * w + x) * 4 + 3];
                if a > 64 {
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                }
            }
        }
        let span = max_x.saturating_sub(min_x);
        // "CONSUMERS" at 48px/2x quality should span hundreds of px, not a
        // few-dozen-px overlapping smear. Floor chosen well below a coherent
        // render and well above a collapsed-advance smear.
        assert!(
            span > 300,
            "ink span too narrow ({span}px) — overlapping/smeared advances"
        );
    }

    /// Regression (probe-proven 2026-07-17, mask:text-alpha (+6,+6)px): the
    /// oracle/prod Inter TTFs are opsz-bracketed releases whose name-table
    /// family is "Inter 24pt" (name IDs 1/16), while the catalog — like the
    /// browser's `@font-face` for the SAME file — declares the family
    /// "Inter". Without the MANIFEST-driven alias, cosmic-text finds no
    /// "Inter" face and silently substitutes Noto Sans, whose metrics differ
    /// (ascent ≈1.069em vs Inter's ≈0.969em, lsb 14.4px vs 8.2px at 160px):
    /// every degraded raster of a catalog-bracketed family lays out with the
    /// wrong font, which misplaced text-shaped mask coverage by (+6,+6)px
    /// against both the browser golden and native's own vector text lane.
    #[test]
    fn load_resolves_bracketed_catalog_inter_not_noto_fallback() {
        let root = fixtures_root();
        let manifest = serde_json::json!({ "systemFonts": false, "keys": {
            "Inter-800-normal": { "family": "Inter", "weight": 800, "style": "normal",
                "ttfPath": root.join("crates/moda-engine/tests/fixtures/fonts/inter/Inter_24pt-ExtraBold.ttf") },
            "Noto Sans-400-normal": { "family": "Noto Sans", "weight": 400, "style": "normal",
                "ttfPath": root.join("crates/moda-engine/tests/fixtures/fonts/noto-sans/NotoSans-Regular.ttf") }
        }});
        let e = build_catalog_engine_from_snapshot(&root, &manifest);
        let style = InlineStyle {
            font_family: "Inter".into(),
            font_weight: "800".into(),
            font_size: 160.0,
            ..default_style()
        };
        let mut inner = e.inner.lock().unwrap();
        let (ascent, _descent) = inner.canvas_font_metrics(&style);
        // Inter ascent ≈ 155px at 160px; the Noto Sans substitution ≈ 171px.
        assert!(
            (145.0..165.0).contains(&ascent),
            "ascent {ascent}px at 160px — Inter face not resolved (Noto substitution gives ~171px)"
        );
    }

    /// Deterministic variant (NO system fonts — a box with a real "Inter"
    /// installed can't mask a broken alias): the MANIFEST-driven catalog load
    /// aliases "inter" → "Inter 24pt", leaves matching names unaliased, and
    /// metrics resolve through the alias.
    #[test]
    #[ignore = "needs unported Moda fixture oracle/fonts/MANIFEST.json"]
    fn manifest_alias_covers_bracketed_families_only() {
        let mut db = fontdb::Database::new();
        let aliases =
            load_catalog_fonts(&mut db, &fixtures_root().join("fixtures/oracle/fonts")).aliases;
        assert_eq!(aliases.get("inter").map(String::as_str), Some("Inter 24pt"));
        // Families whose name-table family equals the catalog name need none.
        assert!(!aliases.contains_key("noto sans"), "spurious: {aliases:?}");
        let e = DegradedEngine::from_db_with_aliases(db, aliases);
        let style = InlineStyle {
            font_family: "Inter".into(),
            font_weight: "800".into(),
            font_size: 160.0,
            ..default_style()
        };
        let mut inner = e.inner.lock().unwrap();
        let (ascent, _descent) = inner.canvas_font_metrics(&style);
        assert!(
            (145.0..165.0).contains(&ascent),
            "ascent {ascent}px at 160px — alias not applied"
        );
    }

    /// Drift pin: the wasm lane's reader-based alias derivation
    /// (`manifest_family_aliases`, fed by the VFS in `install_fonts`) must
    /// produce EXACTLY the aliases the filesystem lane derives while loading
    /// (`load_catalog_fonts`) — otherwise the wasm editor's degraded
    /// rasters/mask sizing resolve different faces than the server lanes.
    #[test]
    #[ignore = "needs unported Moda fixture oracle/fonts/MANIFEST.json"]
    fn manifest_alias_reader_matches_fs_lane() {
        let dir = fixtures_root().join("fixtures/oracle/fonts");
        let mut db = fontdb::Database::new();
        let fs_aliases = load_catalog_fonts(&mut db, &dir).aliases;
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("MANIFEST.json")).unwrap())
                .unwrap();
        let reader_aliases =
            manifest_family_aliases(&manifest, |rel| std::fs::read(dir.join(rel)).ok());
        assert_eq!(fs_aliases, reader_aliases);
        // And the engine built the wasm way (shared sources + reader aliases)
        // resolves the bracket-named catalog family like the fs engine.
        let sources: Vec<std::sync::Arc<Vec<u8>>> = manifest["keys"]
            .as_object()
            .unwrap()
            .values()
            .filter_map(|e| e.get("ttfPath").and_then(|p| p.as_str()))
            .filter_map(|rel| std::fs::read(dir.join(rel)).ok().map(std::sync::Arc::new))
            .collect();
        let e = DegradedEngine::from_font_sources_with_aliases(sources, reader_aliases);
        let style = InlineStyle {
            font_family: "Inter".into(),
            font_weight: "800".into(),
            font_size: 160.0,
            ..default_style()
        };
        let mut inner = e.inner.lock().unwrap();
        let (ascent, _descent) = inner.canvas_font_metrics(&style);
        assert!(
            (145.0..165.0).contains(&ascent),
            "ascent {ascent}px at 160px — wasm-lane alias not applied"
        );
    }

    /// The mask:text-alpha placement pin: the degraded raster of the
    /// `fixtures/parity/design/scenes/masks/text-alpha.json` mask node must put its first
    /// ink where the browser golden does. The golden's mask coverage starts
    /// at canvas (188,276) with the node at (180,240) → logical (8,36) inside
    /// the raster (x = Inter's 160px 'M' lsb ≈ 8px, y = ascent − cap rise).
    /// The Noto substitution put it at ≈(13,42) — the (+6,+6) canvas offset.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn inter_mask_raster_ink_origin_matches_browser_golden() {
        let e = test_engine();
        let r = e
            .render_node(&spec(
                "<p style=\"font-family: Inter; font-size: 160px; font-weight: 800; color: #000000\">MODA</p>",
                920.0,
                320.0,
            ))
            .expect("raster");
        let w = r.width as usize;
        let (mut min_x, mut min_y) = (usize::MAX, usize::MAX);
        for (i, px) in r.rgba.chunks_exact(4).enumerate() {
            if px[3] > 64 {
                min_x = min_x.min(i % w);
                min_y = min_y.min(i / w);
            }
        }
        let ink_x = min_x as f64 / r.quality_scale;
        let ink_y = min_y as f64 / r.quality_scale;
        assert!(
            (ink_x - 8.0).abs() <= 2.0 && (ink_y - 36.0).abs() <= 2.0,
            "ink origin ({ink_x},{ink_y}) logical px — expected ≈(8,36); \
             (13,42)-class values mean the Noto substitution is back"
        );
    }

    /// RTL word order (the `bidi_visual` line-assembly seam): a two-word
    /// Arabic line must place the LOGICALLY-FIRST word on the RIGHT. The wrap
    /// emits per-word segments, so without the visual reorder the words draw
    /// left-to-right in logical order (letters joined, order wrong — the prod
    /// field bug). Pinned with a short first word and a long second word:
    /// after the fix the LEFT ink cluster (logical-last word) is the wide one.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn arabic_two_word_line_renders_rtl_word_order() {
        let e = test_engine();
        // "به" (2 letters) then "بببببببب" (8 letters) — joined beh runs with
        // clearly different widths, separated by a space.
        let short = "\u{0628}\u{0647}";
        let long = "\u{0628}\u{0628}\u{0628}\u{0628}\u{0628}\u{0628}\u{0628}\u{0628}";
        let html = format!(
            "<p style=\"font-family: Inter; font-size: 40px; color: #000\">{short} {long}</p>"
        );
        let r = e.render_node(&spec(&html, 600.0, 80.0)).expect("raster");
        let w = r.width as usize;
        let h = r.height as usize;
        // Ink coverage per column.
        let mut col_ink = vec![0usize; w];
        for y in 0..h {
            for x in 0..w {
                if r.rgba[(y * w + x) * 4 + 3] > 64 {
                    col_ink[x] += 1;
                }
            }
        }
        let cols: Vec<usize> = (0..w).filter(|&x| col_ink[x] > 0).collect();
        let (first, last) = (*cols.first().expect("ink"), *cols.last().expect("ink"));
        // Find the widest empty gap between ink columns — the inter-word space.
        let mut gap_at = None;
        let mut gap_len = 0usize;
        let mut run_start = None;
        for x in first..=last {
            if col_ink[x] == 0 {
                if run_start.is_none() {
                    run_start = Some(x);
                }
            } else if let Some(s) = run_start.take() {
                if x - s > gap_len {
                    gap_len = x - s;
                    gap_at = Some((s + x) / 2);
                }
            }
        }
        let split = gap_at.expect("two ink clusters separated by the space");
        let left_w = (first..split).filter(|&x| col_ink[x] > 0).count();
        let right_w = (split..=last).filter(|&x| col_ink[x] > 0).count();
        assert!(
            left_w > right_w * 2,
            "RTL word order: the LONG (logical-last) word must be LEFT of the \
             short one; got left {left_w}px vs right {right_w}px ink"
        );
    }

    /// Justified right-to-left text is flush on both edges: every line of a justified Arabic paragraph reaches the
    /// box's right edge, as the same paragraph set `text-align: right` does (its start edge).
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn justified_arabic_lines_reach_the_right_edge() {
        let e = test_engine();
        let words = "\u{0647}\u{0630}\u{0627} \u{0646}\u{0635} \u{0639}\u{0631}\u{0628}\u{064A} \u{0637}\u{0648}\u{064A}\u{0644}";
        let text = format!("{words} {words}. {words}, {words} {words} {words}");
        // The rightmost ink of each line, in node space (the raster placed as the canvas places it).
        let right_edges = |align: &str| -> Vec<f64> {
            let html = format!(
                "<p style=\"font-family: Noto Sans Arabic; font-size: 20px; line-height: 2; color: #000; text-align: {align}\">{text}</p>"
            );
            let node = spec(&html, 300.0, 300.0);
            let r = e.render_node(&node).expect("raster");
            let (dx, _) = raster_offset(&node, r.logical_w, r.logical_h);
            let (w, h) = (r.width as usize, r.height as usize);
            let rows: Vec<Option<usize>> = (0..h)
                .map(|y| (0..w).rev().find(|&x| r.rgba[(y * w + x) * 4 + 3] > 64))
                .collect();
            // Lines (20px apart at line-height 2): ink rows, joining gaps shorter than 5 logical px (Arabic dots sit
            // apart from their letters).
            let gap = (5.0 * r.quality_scale) as usize;
            let mut edges = Vec::new();
            let (mut band, mut blank) = (None::<usize>, 0usize);
            for row in rows.iter().chain(std::iter::repeat(&None).take(gap + 1)) {
                match (row, band) {
                    (Some(x), m) => {
                        band = Some(m.map_or(*x, |m| m.max(*x)));
                        blank = 0;
                    }
                    (None, Some(m)) => {
                        blank += 1;
                        if blank > gap {
                            edges.push(dx + m as f64 / r.quality_scale);
                            band = None;
                        }
                    }
                    (None, None) => {}
                }
            }
            edges
        };
        let right = right_edges("right");
        let justified = right_edges("justify");
        assert!(
            justified.len() >= 3,
            "a paragraph of several lines: {justified:?}"
        );
        assert_eq!(justified.len(), right.len(), "{justified:?} vs {right:?}");
        for (j, r) in justified.iter().zip(&right) {
            assert!(
                (j - r).abs() <= 1.5,
                "justified right edges {justified:?} vs right-aligned {right:?}"
            );
            assert!(
                (j - 300.0).abs() <= 3.0,
                "lines reach the box's right edge: {justified:?}"
            );
        }
    }

    /// Paragraphs aligned to different edges in one node each align in the node's box: a left paragraph under a
    /// right-aligned one starts at the box's left edge, not where the right-anchored texture begins.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn mixed_alignment_paragraphs_align_in_the_node_box() {
        let e = test_engine();
        let html = "<p style=\"font-family: Noto Sans Arabic; font-size: 20px; color: #000; text-align: right\">\u{0646}\u{0635}</p>\
                    <p style=\"font-family: Noto Sans Arabic; font-size: 20px; color: #000; text-align: left\">\u{0647}\u{0630}\u{0627} \u{0646}\u{0635}</p>";
        let node = spec(html, 400.0, 100.0);
        let r = e.render_node(&node).expect("raster");
        let (dx, _) = raster_offset(&node, r.logical_w, r.logical_h);
        let w = r.width as usize;
        let ink: Vec<usize> = r
            .rgba
            .chunks_exact(4)
            .enumerate()
            .filter(|(_, p)| p[3] > 64)
            .map(|(i, _)| i % w)
            .collect();
        let left = dx + *ink.iter().min().expect("ink") as f64 / r.quality_scale;
        let right = dx + *ink.iter().max().expect("ink") as f64 / r.quality_scale;
        assert!(
            left.abs() <= 4.0,
            "the left paragraph starts at the box's left edge: {left}"
        );
        assert!(
            (right - 400.0).abs() <= 4.0,
            "the right paragraph ends at the box's right edge: {right}"
        );
    }

    /// Letter-spacing places one grapheme at a time; in a right-to-left piece they run from its right edge. Hebrew
    /// "ומממממ" (a narrow vav, then five wide mems) keeps the vav, the logically first letter, at the RIGHT end.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn letter_spaced_hebrew_keeps_its_letter_order() {
        let e = test_engine();
        let word = "\u{05D5}\u{05DE}\u{05DE}\u{05DE}\u{05DE}\u{05DE}";
        for spacing in [0.0, 12.0] {
            let html = format!(
                "<p style=\"font-family: Inter; font-size: 40px; letter-spacing: {spacing}px; color: #000\">{word}</p>"
            );
            let r = e.render_node(&spec(&html, 600.0, 80.0)).expect("raster");
            let (w, h) = (r.width as usize, r.height as usize);
            let inked = |x: usize| (0..h).any(|y| r.rgba[(y * w + x) * 4 + 3] > 64);
            // Ink clusters left to right: each letter is its own (Hebrew doesn't join).
            let mut clusters: Vec<(usize, usize)> = Vec::new();
            for x in 0..w {
                if inked(x) {
                    match clusters.last_mut() {
                        Some((_, end)) if *end + 1 == x => *end = x,
                        _ => clusters.push((x, x)),
                    }
                }
            }
            assert_eq!(
                clusters.len(),
                6,
                "six letters at spacing {spacing}: {clusters:?}"
            );
            let width = |c: (usize, usize)| c.1 - c.0 + 1;
            let (first, last) = (clusters[0], clusters[5]);
            assert!(
                width(last) * 2 < width(first),
                "at spacing {spacing} the narrow vav (logically first) draws at the right: left {first:?}, right {last:?}"
            );
        }
    }

    #[test]
    #[ignore = "needs unported Moda fixture fonts-degraded/NotoColorEmoji.ttf"]
    fn emoji_rasterize_in_color() {
        let e = oracle_engine();
        let r = e
            .render_node(&spec(
                "<p style=\"font-family: Inter; font-size: 36px; color: #1d2733\">Wave: 👋🏽</p>",
                400.0,
                60.0,
            ))
            .expect("raster");
        // The skin-tone cluster (Q-42 coverage-gap class) must produce
        // colored, non-text-color pixels (sbix/CBDT bitmap strike).
        let colored = r
            .rgba
            .chunks_exact(4)
            .filter(|p| p[3] > 128 && (p[0] as i32 - p[2] as i32).abs() > 30)
            .count();
        assert!(colored > 50, "expected colored emoji pixels, got {colored}");
    }

    fn oracle_fonts(files: &[&str]) -> DegradedEngine {
        let dir = fixtures_root().join("fixtures/oracle/fonts");
        DegradedEngine::from_font_data(files.iter().map(|f| std::fs::read(dir.join(f)).unwrap()).collect())
    }

    /// A family asked at a weight it lacks draws in its nearest face, as CSS and the vector lane do: Instrument
    /// Serif (Regular only) at 700 drew in the first bold face in the set (DejaVu Sans Bold here), Latin and all.
    #[test]
    #[ignore = "needs unported Moda fixture instrument-serif/InstrumentSerif-Regular.ttf and Inter Bold"]
    fn a_weight_the_family_lacks_draws_in_its_nearest_face() {
        let e = oracle_fonts(&[
            "inter/Inter_24pt-Bold.ttf",
            "inter/Inter_24pt-Regular.ttf",
            "instrument-serif/InstrumentSerif-Regular.ttf",
        ]);
        for weight in [400, 500, 700, 900] {
            let clusters = e.resolve_faces("Instrument Serif", weight, false, "Sea Glass").unwrap();
            for c in &clusters {
                assert_eq!(c.family.as_deref(), Some("Instrument Serif"), "weight {weight}: {c:?}");
                assert_eq!(c.weight, 400, "weight {weight}: {c:?}");
            }
        }
        // Raw font blobs use the name-table family; the catalog's "Inter" alias is not installed here.
        let bold = e.resolve_faces("Inter 24pt", 700, false, "Sea").unwrap();
        assert!(!bold.is_empty());
        assert!(bold.iter().all(|c| c.weight == 700 && c.family.as_deref() == Some("Inter 24pt")), "{bold:?}");
    }

    /// Thai the run's face lacks goes to Noto Sans Thai at its nearest weight whatever weight was asked;
    /// cosmic-text's own script fallback skipped the static Regular at 500 or 700 for any face of that weight
    /// covering Thai (FreeSerif Bold on Linux).
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn script_fallback_takes_the_scripts_family_at_its_nearest_weight() {
        let mut db = fontdb::Database::new();
        for path in [
            "crates/moda-engine/tests/fixtures/fonts/inter/Inter_24pt-ExtraBold.ttf",
            "crates/moda-engine/tests/fixtures/fonts/inter/Inter_24pt-Regular.ttf",
            crate::world_type_fonts::WORLD_TYPE_THAI,
        ] {
            db.load_font_data(std::fs::read(fixtures_root().join(path)).unwrap());
        }
        let e = DegradedEngine::from_db(db);
        let text = "Sea วงดนตรี Glass";
        let thai = text.find('ว').unwrap()..text.find(" Glass").unwrap();
        let inner = e.inner.lock().unwrap();
        for weight in [400u16, 500, 700] {
            let attrs = Attrs::new().family(Family::Name("Inter")).weight(Weight(weight));
            let list = attrs_list_for(&inner.font_system, text, &attrs);
            for (i, c) in text.char_indices() {
                let got = list.get_span(i);
                if thai.contains(&i) {
                    assert_eq!(got.family, Family::Name("Noto Sans Thai"), "{c} at {weight}");
                    assert_eq!(got.weight, Weight(400), "{c} at {weight}");
                } else if !c.is_whitespace() {
                    assert_eq!(got.family, Family::Name("Inter"), "{c} at {weight}");
                }
            }
        }
        drop(inner);
        let clusters = e.resolve_faces("Inter", 700, false, text).unwrap();
        for c in clusters.iter().filter(|c| thai.contains(&c.start)) {
            assert_eq!(c.family.as_deref(), Some("Noto Sans Thai"), "{c:?}");
        }
    }

    /// A static face of a family beside a variable face of it (the bundled fallback Noto Sans Arabic Regular and
    /// the installed variable Noto Sans Arabic) took every weight: fontdb lists the variable face at its default
    /// 400, so bold drew Regular, silently. The variable face now draws each weight it spans, whichever loaded first.
    #[test]
    #[ignore = "needs unported Moda fixture Lora-Regular.ttf and Lora[wght].ttf"]
    fn a_variable_face_draws_the_weights_a_static_face_of_its_family_shadowed() {
        let static_regular = fixtures_root().join("fixtures/oracle/fonts/lora/Lora-Regular.ttf");
        let public = fixtures_root().parent().unwrap().join("packages/editor/public");
        let variable = public.join("fonts/google/lora/Lora[wght].ttf");
        let (static_regular, variable) = (std::fs::read(static_regular).unwrap(), std::fs::read(variable).unwrap());
        for fonts in [vec![static_regular.clone(), variable.clone()], vec![variable, static_regular]] {
            let e = DegradedEngine::from_font_data(fonts);
            let mut ink = Vec::new();
            for weight in [400u16, 700] {
                let clusters = e.resolve_faces("Lora", weight, false, "Sea Glass").unwrap();
                let drawn = |c: &ResolvedCluster| c.family.as_deref() == Some("Lora") && c.weight == weight;
                assert!(clusters.iter().all(drawn), "{weight}: {clusters:?}");
                let style = format!("font-family: Lora; font-size: 40px; font-weight: {weight}");
                let html = format!("<p style=\"{style}\">Sea Glass</p>");
                let r = e.render_node(&spec(&html, 400.0, 60.0)).expect("raster");
                ink.push(r.rgba.chunks_exact(4).map(|p| u64::from(p[3])).sum::<u64>());
            }
            assert!(ink[1] * 10 > ink[0] * 12, "700 draws heavier than 400: {ink:?}");
        }
    }
}
