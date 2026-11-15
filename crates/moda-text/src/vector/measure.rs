//! The vector engine's font store, measurer and glyph placement — ports of
//! `renderers/webgpu/text/vector/vectorLayout.ts` (createVectorMeasurer,
//! VectorGlyphShaper, normalizeTextForShaping, styleToVariant),
//! `renderers/webgpu/text/vector/glyphPlacement.ts` (Phase 3: ShapedGlyph,
//! shapeWithKerning, placeGlyphs / measureGlyphRun) and the store side of
//! `VectorFontCache.ts`.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;
use std::path::PathBuf;
use std::rc::Rc;

use serde_json::Value;

use crate::catalog::Catalog;
use crate::measure::TextMeasurer;
use crate::ttf::TtfMetrics;
use crate::types::InlineStyle;
use crate::vector::cjk::js_whitespace;
use crate::vector::hb::{shape_canvas2d_hb, shape_vector_hb, HbFace};
use crate::vector::otjs::{OtjsFont, ShapeError};
use crate::vector::varfont::{variation_axes, AxisInfo, Variation};

/// `styleToVariant` (vectorLayout.ts:41-47): `parseInt(fontWeight, 10) || 400`
/// — NOTE no 'bold' handling (unlike `styleToFontKey`); 'bold' → NaN → 400.
pub fn style_to_variant(style: &InlineStyle) -> (String, i64, &'static str) {
    let weight = crate::jsnum::js_parse_int(&style.font_weight)
        .filter(|w| *w != 0)
        .unwrap_or(400);
    let st = match style.font_style.as_str() {
        "italic" => "italic",
        "oblique" => "oblique",
        _ => "normal",
    };
    (style.font_family.clone(), weight, st)
}

/// `styleFontKey` (vectorLayout.ts:50-53).
pub fn style_font_key(style: &InlineStyle) -> String {
    let (family, weight, st) = style_to_variant(style);
    format!("{family}|{weight}|{st}")
}

/// E6.3: how many parsed faces the store keeps before dropping the variable
/// instances (a slider drag parses one per value).
const MAX_CACHED_INSTANCES: usize = 256;

/// Which instance of a variant to load: the run's font size (automatic
/// optical size) and its `font-variation-settings` (E6.3). `Option<f64>`
/// converts, for callers that only know a size.
#[derive(Debug, Clone, Copy, Default)]
pub struct Instance<'a> {
    pub size_px: Option<f64>,
    pub variations: &'a [Variation],
}

impl From<Option<f64>> for Instance<'_> {
    fn from(size_px: Option<f64>) -> Self {
        Instance {
            size_px,
            variations: &[],
        }
    }
}

impl<'a> Instance<'a> {
    /// A run style's instance: its size and its axis settings.
    pub fn of(style: &'a InlineStyle) -> Self {
        Instance {
            size_px: Some(style.font_size),
            variations: &style.font_variations,
        }
    }
}

/// Width fallback ratio for unresolved fonts (vectorLayout.ts:86).
const MISSING_FONT_ADVANCE_RATIO: f64 = 0.5;

/// The vendored-TTF font store — the oracle-lane mirror of `VectorFontCache`
/// and `createVectorFontHost` (tools/oracle/dump-linebox.mjs): resolve a
/// variant through the catalog's EXACT-match `getVariantTtfPath`, read the
/// vendored bytes, parse. A catalog TTF that is NOT vendored panics (the JS
/// dump's `VendorGapError` — never a silent fallback); a TTF that fails to
/// parse mirrors `VectorFontCache.failed` (font unavailable → fallback).
pub struct VectorFontStore {
    pub(crate) vfs_lease: Option<Arc<crate::host::VfsLease>>,
    fonts_root: PathBuf,
    /// ttf url → vendored relative path (MANIFEST.files).
    files: HashMap<String, String>,
    /// instance key (ttf url, plus variable-font coordinates when the face is
    /// variable) → parsed font (None = parse failed, mirrors `failed`).
    cache: RefCell<HashMap<String, Option<Rc<StoreEntry>>>>,
    /// Desktop fork: ttf url → bytes, shared by every instance of a variable
    /// font (named instances and optical sizes are separate cache entries over
    /// one buffer).
    bytes: RefCell<HashMap<String, (Arc<Vec<u8>>, Rc<Vec<AxisInfo>>)>>,
    /// Track C: vendor-gap policy. The ORACLE store panics on a catalog TTF
    /// that is not vendored (drift must scream); the ENGINE store is lenient
    /// — a missing vendored TTF behaves like the browser's not-yet-loaded
    /// font (None → per-node MSDF fallback, `font-unavailable` class).
    lenient_vendor_gaps: bool,
    /// The last variant [`Self::load_variant`] resolved, by its exact arguments. The vector lane asks for one variant
    /// thousands of times in a row (each shaping, measure and metrics call of every word), and its answer is a
    /// function of the arguments, the catalog (by [`Catalog::id`]) and this store's files.
    last_variant: RefCell<Option<LastVariant>>,
    /// Test-only: pretend HarfBuzz face creation always fails, so the vector
    /// lane's degraded codepoint backstop is exercised for an
    /// otjs-incompatible font even though rustybuzz can in fact parse it.
    #[cfg(test)]
    force_no_hb_face: Cell<bool>,
}

/// [`VectorFontStore::load_variant`]'s arguments and the entry they resolved to.
struct LastVariant {
    catalog: u64,
    family: String,
    weight: i64,
    style: String,
    size_px: Option<f64>,
    variations: Vec<Variation>,
    entry: Rc<StoreEntry>,
}

/// One parsed font plus everything the Phase 2 lane needs from the same
/// vendored bytes: the raw binary (HB faces shape from it — the mirror of
/// `harfbuzzShaper.ts::registerShapingFontData`) and the lazily-built
/// rustybuzz face (`getHbFont`'s WeakMap cache; `Some(None)` = creation
/// failed once, don't retry — callers fall back to the opentype path).
struct StoreEntry {
    font: Rc<OtjsFont>,
    bytes: Arc<Vec<u8>>,
    /// The instance coordinates `font` was parsed at (the HB face must match).
    variations: Vec<Variation>,
    hb: RefCell<Option<Option<Rc<HbFace>>>>,
    /// Per-font opentype.js shaping-capability memo (the
    /// mirror of `fontShapingCapability.ts`): `Some(true)` = route this font's
    /// runs through rustybuzz, either because the otjs walk cannot shape it at
    /// all (type-7-GSUB class) OR because it carries a REQUIRED GSUB feature the
    /// otjs walk never applies (`rvrn`/`rclt`) but HarfBuzz/browsers always do.
    needs_hb: Cell<Option<bool>>,
    /// The GSUB feature tags the font declares, read once on first ask
    /// (synthesized small caps, `crate::small_caps`).
    gsub_tags: RefCell<Option<Rc<Vec<[u8; 4]>>>>,
}

/// Why the vector lane could not produce a parsed font for a variant.
///
/// The lane used to collapse four structurally different causes into
/// one undifferentiated `font-unavailable`, which is why a production
/// investigation could not tell "the catalog has no TTF at this exact
/// weight/style" from "the catalog's URL is absent from the vendored manifest"
/// from "the bytes don't parse". Each has a different owner and a different fix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorFontMiss {
    /// `getVariantTtfPath` found no EXACT weight/style variant carrying a
    /// `ttf_url` — a catalog/vendoring gap, not a render fault.
    NoCatalogTtf,
    /// The catalog's `ttf_url` is not a key of `MANIFEST.files`: the catalog and
    /// the vendored snapshot disagree. Either the TTF was never vendored, or the
    /// store was built over a DIFFERENT fonts dir than the catalog came from.
    NotVendored { url: String },
    /// Vendored, but the bytes could not be read (missing file / VFS race).
    ReadFailed { url: String },
    /// Bytes read, but `OtjsFont::parse` rejected them.
    ParseFailed { url: String },
}

impl VectorFontMiss {
    /// Stable, low-cardinality class name — safe to group/alert on.
    pub fn class(&self) -> &'static str {
        match self {
            VectorFontMiss::NoCatalogTtf => "no-catalog-ttf",
            VectorFontMiss::NotVendored { .. } => "not-vendored",
            VectorFontMiss::ReadFailed { .. } => "read-failed",
            VectorFontMiss::ParseFailed { .. } => "parse-failed",
        }
    }

    /// The class plus a credential-free label for the URL that missed, so a log
    /// line names WHICH url the manifest lookup was keyed on without leaking the
    /// signature a raw font URL carries.
    pub fn detail(&self) -> String {
        match self {
            VectorFontMiss::NoCatalogTtf => self.class().to_string(),
            VectorFontMiss::NotVendored { url }
            | VectorFontMiss::ReadFailed { url }
            | VectorFontMiss::ParseFailed { url } => {
                format!("{} {}", self.class(), font_url_label(url))
            }
        }
    }
}

/// Longest url label we ever put in a log line.
const URL_LABEL_MAX_CHARS: usize = 96;

/// A credential-free label for a font URL: host + final path segment, with the
/// query and fragment DROPPED.
///
/// Team-font catalog URLs are signed CDN/GCS URLs whose credential lives entirely
/// in the query (`Signature=`/`Expires=`/`X-Goog-*`), and the repo's standing rule
/// is that a raw asset URL never reaches a log line. The host and the file name are
/// what actually identify the miss (`fonts.moda.app/OpenSans-Bold.ttf`), so keep
/// those and drop everything else.
///
/// A short digest of the FULL url is appended, because dropping the query alone
/// would erase the one distinction a font-miss diagnosis turns on: two
/// urls differing only in `Signature=`/`Expires=` produce an identical label, so a
/// `not-vendored` record could not tell signature skew (a re-signed catalog against
/// a reused fonts dir) from a genuine vendoring gap. The digest discriminates them
/// without carrying the credential — same trick as the Python side's
/// `placeholder_ref_label`.
pub fn font_url_label(url: &str) -> String {
    // A data: URI is all payload and no identity — labelling it would dump ~96
    // chars of base64 into a log line and say nothing about which font missed.
    // `get` (not a slice) — a `[..5]` index panics when byte 5 lands inside a
    // multi-byte char, and a percent-decoded object key can be non-ASCII.
    if url
        .get(..5)
        .is_some_and(|p| p.eq_ignore_ascii_case("data:"))
    {
        return "data-uri".to_string();
    }
    let no_frag = url.split('#').next().unwrap_or(url);
    let no_query = no_frag.split('?').next().unwrap_or(no_frag);
    let after_scheme = no_query
        .split_once("://")
        .map_or(no_query, |(_, rest)| rest);
    let mut parts = after_scheme.splitn(2, '/');
    let authority = parts.next().unwrap_or("");
    // `user:pass@host` — userinfo is the one other place a URL can carry a
    // credential, so drop it. (Today's font catalog never uses it; the point is
    // that this function is the seam that must be safe for ANY url it is handed.)
    let host = authority.rsplit('@').next().unwrap_or(authority);
    let path = parts.next().unwrap_or("");
    let basename = path.rsplit('/').next().unwrap_or("");
    let label = match (host.is_empty(), basename.is_empty()) {
        (true, true) => "unparseable".to_string(),
        (_, true) => host.to_string(),
        (true, _) => basename.to_string(),
        _ => format!("{host}/{basename}"),
    };
    // Truncate on a CHAR boundary — a percent-decoded object key can be non-ASCII.
    let label: String = if label.chars().count() <= URL_LABEL_MAX_CHARS {
        label
    } else {
        label.chars().take(URL_LABEL_MAX_CHARS).collect()
    };
    format!("{label}#{}", url_digest(url))
}

/// A short, stable, non-reversible tag for a full url. Not cryptographic — it only
/// has to distinguish two urls in a log line.
fn url_digest(url: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    url.hash(&mut h);
    format!("{:08x}", h.finish() as u32)
}

/// `MANIFEST.files` as the vector store consumes it: ttf url → vendored path
/// (relative to the fonts dir, or absolute for a shared CAS blob).
///
/// Extracted here so `TextHost::load` and the store agree by construction — the
/// host keeps the map and the store is built from it, rather than each parsing
/// the file independently.
pub fn manifest_files(manifest: &Value) -> HashMap<String, String> {
    let mut files = HashMap::new();
    if let Some(map) = manifest.get("files").and_then(|f| f.as_object()) {
        for (url, entry) in map {
            if let Some(path) = entry.get("path").and_then(|p| p.as_str()) {
                files.insert(url.clone(), path.to_string());
            }
        }
    }
    files
}

/// The capability probe text (mirror of `OPENTYPE_SHAPING_PROBE_TEXT`): a
/// Latin word exercising the latn 'liga' lookup walk, where opentype.js
/// throws for unsupported lookup formats regardless of text content.
const OPENTYPE_SHAPING_PROBE_TEXT: &str = "ffi";

impl StoreEntry {
    fn gsub_tags(&self) -> Rc<Vec<[u8; 4]>> {
        if let Some(cached) = self.gsub_tags.borrow().as_ref() {
            return cached.clone();
        }
        let tags = Rc::new(crate::font_features::gsub_feature_tags(&self.bytes));
        *self.gsub_tags.borrow_mut() = Some(tags.clone());
        tags
    }

    fn hb_face(&self) -> Option<Rc<HbFace>> {
        if let Some(cached) = self.hb.borrow().as_ref() {
            return cached.clone();
        }
        let created =
            HbFace::from_bytes_with_variations(self.bytes.clone(), &self.variations).map(Rc::new);
        *self.hb.borrow_mut() = Some(created.clone());
        created
    }

    /// One-shot probe, memoized per parsed font (`fontNeedsHarfBuzz`). Routes a
    /// font through rustybuzz when the otjs walk throws OR when the font carries
    /// a required GSUB feature the otjs walk skips (`rvrn`/`rclt`) — see
    /// `fontShapingCapability.ts`.
    fn needs_harfbuzz(&self) -> bool {
        if let Some(cached) = self.needs_hb.get() {
            return cached;
        }
        let needs = self
            .font
            .string_to_glyphs(OPENTYPE_SHAPING_PROBE_TEXT)
            .is_err()
            || self.font.has_unsupported_required_feature();
        self.needs_hb.set(Some(needs));
        needs
    }
}

impl VectorFontStore {
    /// TEST/ORACLE ONLY — the engine must NOT use this.
    ///
    /// Named for what it is because it cannot be `cfg(test)`-gated like its lenient
    /// sibling: the parity/oracle suites in `moda-text/tests/` are separate crates, so
    /// the constructor has to stay `pub` for them. It takes an arbitrary `fonts_root`
    /// + manifest, which is precisely how the store and the catalog came to be built
    /// over different dirs, so the name is the enforcement: a production
    /// call site would have to write `_for_tests`, which review and `grep` both catch.
    /// Every production lane goes through [`crate::TextHost::vector_font_store_lenient`].
    #[doc(hidden)]
    pub fn load_from_manifest_for_tests(
        fonts_root: &std::path::Path,
        manifest: &Value,
    ) -> VectorFontStore {
        Self::load_with_policy(fonts_root, manifest, false)
    }

    /// TEST/ORACLE ONLY — the engine must NOT use this.
    ///
    /// It takes an arbitrary `fonts_root` + manifest, which is precisely how the
    /// store and the catalog came to be built over different dirs. Every
    /// production lane goes through [`crate::TextHost::vector_font_store_lenient`],
    /// which can only ever hand out a store over the host's OWN snapshot; this has
    /// no production callers left. Lenient: vendor gaps fall back per node instead
    /// of panicking (see `lenient_vendor_gaps`).
    ///
    /// `cfg(test)`-gated so that "must NOT use this" is enforced by the COMPILER
    /// rather than by this comment — a doc-comment invariant is precisely the
    /// enforcement mode that failed before.
    #[cfg(test)]
    pub fn load_from_manifest_lenient(
        fonts_root: &std::path::Path,
        manifest: &Value,
    ) -> VectorFontStore {
        Self::load_with_policy(fonts_root, manifest, true)
    }

    fn load_with_policy(
        fonts_root: &std::path::Path,
        manifest: &Value,
        lenient_vendor_gaps: bool,
    ) -> VectorFontStore {
        Self::from_files(fonts_root, manifest_files(manifest), lenient_vendor_gaps)
    }

    /// Build over an ALREADY-EXTRACTED `MANIFEST.files` map (see
    /// [`manifest_files`]).
    ///
    /// `pub(crate)`, not `pub`: [`crate::TextHost::vector_font_store_lenient`] is the
    /// only caller that can guarantee the map and the catalog are ONE snapshot, and
    /// leaving this reachable from other crates left the store/catalog pairing as a
    /// per-call-site convention — the enforcement mode that failed before.
    ///
    /// This is the constructor the engine uses: `TextHost` extracts the map once
    /// at load and hands it over, so rebuilding the store never re-reads the
    /// fonts dir. That matters beyond the saved syscall — a materialized
    /// per-canvas dir is explicitly NOT load-bearing after the host has loaded
    /// (`moda_fonts::prune_fonts_by_canvas` may sweep it from another process,
    /// because the manifest addresses bytes by absolute CAS path), so a lazy
    /// re-read would make a warm host's vector lane fail on a pruned dir.
    pub(crate) fn from_files(
        fonts_root: &std::path::Path,
        files: HashMap<String, String>,
        lenient_vendor_gaps: bool,
    ) -> VectorFontStore {
        VectorFontStore {
            vfs_lease: None,
            fonts_root: fonts_root.to_path_buf(),
            files,
            cache: RefCell::new(HashMap::new()),
            bytes: RefCell::new(HashMap::new()),
            lenient_vendor_gaps,
            last_variant: RefCell::new(None),
            #[cfg(test)]
            force_no_hb_face: Cell::new(false),
        }
    }

    /// Test-only: force every variant's HarfBuzz face to be treated as
    /// unavailable, so otjs-incompatible fonts take the degraded codepoint
    /// backstop instead of the HB reroute.
    #[cfg(test)]
    #[cfg_attr(not(vork_private_tests), allow(dead_code))] // Its one caller is a private test.
    fn force_no_hb_face(&self) {
        self.force_no_hb_face.set(true);
    }

    /// The bytes (and `fvar` axes) behind `url`, read once and shared by every
    /// instance. Read misses are never cached (see the lenient returns).
    fn url_bytes(
        &self,
        url: &str,
        family: &str,
        weight: i64,
        style: &str,
    ) -> Result<(Arc<Vec<u8>>, Rc<Vec<AxisInfo>>), VectorFontMiss> {
        if let Some(hit) = self.bytes.borrow().get(url) {
            return Ok(hit.clone());
        }
        let Some(rel) = self.files.get(url) else {
            if self.lenient_vendor_gaps {
                // Wasm-editor (GAP#2 font-feed race): do NOT negative-cache a
                // not-yet-vendored TTF. The feeder vendors fonts on demand and
                // re-installs a fresh store, but a render that probes this url
                // BEFORE its bytes land would otherwise poison the cache with a
                // permanent `None` — the run then stays on per-glyph fallback
                // for the whole session (the nondeterministic "CONSUMERS VOTE."
                // → "c ONs Uм ERS" degrade). Returning None WITHOUT caching lets
                // the next frame (after the bytes arrive) resolve the real font.
                return Err(VectorFontMiss::NotVendored { url: url.into() });
            }
            panic!(
                "[VectorFontStore] TTF for {family} {weight} {style} ({url}) is in the catalog \
                 but not vendored — re-run fetch-fonts.mjs --vector-fallbacks"
            );
        };
        // VFS-first (wasm-editor): the in-memory overlay the TextHost loads
        // from is consulted before std::fs, so the vector store reads the same
        // browser-fetched bytes. Native behavior is unchanged when no overlay
        // is installed (`read_file_bytes` falls through to std::fs).
        let bytes = match crate::host::read_file_shared(&self.fonts_root.join(rel)) {
            Ok(b) => b,
            Err(e) => {
                if self.lenient_vendor_gaps {
                    // Same race rationale as above: a transient read miss (bytes
                    // not yet in the VFS overlay) must not be cached, so the
                    // font resolves once its bytes are installed.
                    let _ = e;
                    return Err(VectorFontMiss::ReadFailed { url: url.into() });
                }
                panic!("[VectorFontStore] read {rel}: {e}")
            }
        };
        let axes = Rc::new(variation_axes(&bytes));
        self.bytes
            .borrow_mut()
            .insert(url.to_string(), (bytes.clone(), axes.clone()));
        Ok((bytes, axes))
    }

    /// What tells this variant's instance at `instance` apart from its other instances: `""` for a static face (one
    /// instance), else `#` and the instance's coordinates (axis settings, optical size from the font size). A cache
    /// of outlines per glyph keys on this beside the variant: `Roboto|400|normal` is one instance at `wght` 900 and
    /// another at 400, with different outlines under one glyph id.
    pub fn instance_suffix<'v>(
        &self,
        catalog: &Catalog,
        family: &str,
        weight: i64,
        style: &str,
        instance: impl Into<Instance<'v>>,
    ) -> String {
        let Some(source) = catalog.get_variant_ttf_source(family, weight, style) else {
            return String::new();
        };
        let (url, variations) = (source.0.as_str(), &source.1);
        let Ok((_, axes)) = self.url_bytes(url, family, weight, style) else {
            return String::new();
        };
        let coords = instance_coords(&axes, variations, instance.into());
        instance_key(url, &coords)
            .strip_prefix(url)
            .unwrap_or_default()
            .to_string()
    }

    /// Resolve `family`/`weight`/`style` to a parsed instance. `size_px` drives
    /// optical sizing: a variable face with an `opsz` axis the variant does not
    /// pin is instanced at `opsz = font size in px` (clamped to the axis), as
    /// CSS `font-optical-sizing: auto` does. `None` keeps the axis default.
    fn load_variant<'v>(
        &self,
        catalog: &Catalog,
        family: &str,
        weight: i64,
        style: &str,
        size_px: impl Into<Instance<'v>>,
    ) -> Result<Rc<StoreEntry>, VectorFontMiss> {
        let instance: Instance<'_> = size_px.into();
        if let Some(last) = self.last_variant.borrow().as_ref() {
            if last.catalog == catalog.id()
                && last.weight == weight
                && last.size_px == instance.size_px
                && last.family == family
                && last.style == style
                && last.variations.as_slice() == instance.variations
            {
                return Ok(last.entry.clone());
            }
        }
        let entry = self.resolve_variant(catalog, family, weight, style, instance)?;
        *self.last_variant.borrow_mut() = Some(LastVariant {
            catalog: catalog.id(),
            family: family.to_string(),
            weight,
            style: style.to_string(),
            size_px: instance.size_px,
            variations: instance.variations.to_vec(),
            entry: entry.clone(),
        });
        Ok(entry)
    }

    /// [`Self::load_variant`] without the last-variant memo.
    fn resolve_variant(
        &self,
        catalog: &Catalog,
        family: &str,
        weight: i64,
        style: &str,
        instance: Instance<'_>,
    ) -> Result<Rc<StoreEntry>, VectorFontMiss> {
        let Some(source) = catalog.get_variant_ttf_source(family, weight, style) else {
            return Err(VectorFontMiss::NoCatalogTtf);
        };
        let (url, variations) = (source.0.as_str(), &source.1);
        let (bytes, axes) = self.url_bytes(url, family, weight, style)?;
        let coords = instance_coords(&axes, variations, instance);
        // A static face's instance key is its url: looked up as it is, with no key built.
        let key = if coords.is_empty() {
            Cow::Borrowed(url)
        } else {
            Cow::Owned(instance_key(url, &coords))
        };
        if let Some(cached) = self.cache.borrow().get(key.as_ref()) {
            // Only PARSE outcomes are ever cached, so a cached `None` is always
            // a parse failure.
            return cached.clone().ok_or_else(|| VectorFontMiss::ParseFailed {
                url: url.to_string(),
            });
        }
        let parsed = OtjsFont::parse_instance(&bytes, &coords).ok().map(|font| {
            Rc::new(StoreEntry {
                font: Rc::new(font),
                bytes,
                variations: coords,
                hb: RefCell::new(None),
                needs_hb: Cell::new(None),
                gsub_tags: RefCell::new(None),
            })
        });
        let mut cache = self.cache.borrow_mut();
        // E6.3: axis sliders make a new instance per value; keep the instanced
        // entries bounded (the static faces, keyed by bare url, always stay).
        if cache.len() >= MAX_CACHED_INSTANCES {
            cache.retain(|k, _| !k.contains('#'));
        }
        cache.insert(key.into_owned(), parsed.clone());
        drop(cache);
        parsed.ok_or_else(|| VectorFontMiss::ParseFailed {
            url: url.to_string(),
        })
    }

    /// `VectorFontCache.getFontForVariant` (VectorFontCache.ts:65-67) in a
    /// settled frame (loads are synchronous here). `size_px` is the run's font
    /// size, which picks a variable font's optical size (see `load_variant`).
    pub fn font_for_variant<'v>(
        &self,
        catalog: &Catalog,
        family: &str,
        weight: i64,
        style: &str,
        size_px: impl Into<Instance<'v>>,
    ) -> Option<Rc<OtjsFont>> {
        self.font_for_variant_result(catalog, family, weight, style, size_px)
            .ok()
    }

    /// [`Self::font_for_variant`] with the miss CLASSIFIED — the
    /// routing decision is identical, but the caller can say WHY the node is
    /// about to fall back.
    pub fn font_for_variant_result<'v>(
        &self,
        catalog: &Catalog,
        family: &str,
        weight: i64,
        style: &str,
        size_px: impl Into<Instance<'v>>,
    ) -> Result<Rc<OtjsFont>, VectorFontMiss> {
        self.load_variant(catalog, family, weight, style, size_px)
            .map(|e| e.font.clone())
    }

    /// The raw vendored TTF bytes for a variant (the SAME bytes the otjs parse
    /// and HB faces read) — the PDF text layer embeds these so the invisible
    /// glyph runs resolve against exactly the font the raster was drawn with.
    /// `None` mirrors `font_for_variant`'s unresolved/parse-failed cases.
    pub fn bytes_for_variant(
        &self,
        catalog: &Catalog,
        family: &str,
        weight: i64,
        style: &str,
    ) -> Option<Arc<Vec<u8>>> {
        self.load_variant(catalog, family, weight, style, None)
            .ok()
            .map(|e| e.bytes.clone())
    }

    /// Whether a variant's font declares the GSUB feature `tag` (any script).
    /// `false` when the variant has no parsed font.
    pub fn variant_has_gsub_feature(
        &self,
        catalog: &Catalog,
        family: &str,
        weight: i64,
        style: &str,
        tag: &[u8; 4],
    ) -> bool {
        self.load_variant(catalog, family, weight, style, None)
            .map(|e| e.gsub_tags().contains(tag))
            .unwrap_or(false)
    }

    /// Phase 2 (Q-122): the rustybuzz face over the SAME vendored bytes the
    /// otjs parse used — the hybrid arm's `Canvas2dParity` ls==0 shaper.
    /// `None` (no TTF, parse failed, or face creation failed) sends callers
    /// down the opentype path, mirroring `shapeWithHarfBuzz`'s null returns.
    pub fn hb_face_for_variant<'v>(
        &self,
        catalog: &Catalog,
        family: &str,
        weight: i64,
        style: &str,
        size_px: impl Into<Instance<'v>>,
    ) -> Option<Rc<HbFace>> {
        self.load_variant(catalog, family, weight, style, size_px)
            .ok()?
            .hb_face()
    }

    /// True when the VECTOR lane must reroute this font's
    /// runs through rustybuzz — either the otjs walk cannot shape it (type-7
    /// GSUB class) or it carries a required GSUB feature the otjs walk skips
    /// (`rvrn`/`rclt`). Mirror of `fontNeedsHarfBuzz` (probe memoized per parsed
    /// font). An unresolved / unloaded variant reports false (the caller's font
    /// lookup already handled that case).
    pub fn font_needs_harfbuzz<'v>(
        &self,
        catalog: &Catalog,
        family: &str,
        weight: i64,
        style: &str,
        size_px: impl Into<Instance<'v>>,
    ) -> bool {
        self.load_variant(catalog, family, weight, style, size_px)
            .is_ok_and(|e| e.needs_harfbuzz())
    }

    /// The vector lane's shaping route for a variant, resolved in ONE
    /// catalog/cache lookup. The `needs_harfbuzz` probe is memoized per parsed
    /// font (the `needs_hb` cell), so once a font is known to be otjs-
    /// incompatible (type-7 GSUB) subsequent frames skip the throwing opentype
    /// shaping entirely — they go straight to HarfBuzz, or to the degraded
    /// codepoint backstop when no HB face is available. This memo IS the
    /// negative cache that stops the per-frame re-attempt-and-error churn.
    fn vector_shape_route<'v>(
        &self,
        catalog: &Catalog,
        family: &str,
        weight: i64,
        style: &str,
        size_px: impl Into<Instance<'v>>,
    ) -> VectorShapeRoute {
        let Ok(entry) = self.load_variant(catalog, family, weight, style, size_px) else {
            // Unresolved variant: the caller's font lookup already handled it;
            // keep the otjs path (its own error/empty surfaces as before).
            return VectorShapeRoute::Otjs;
        };
        if !entry.needs_harfbuzz() {
            return VectorShapeRoute::Otjs;
        }
        // otjs-incompatible: prefer HarfBuzz (full GSUB/GPOS), else the
        // GSUB-free degraded backstop — NEVER the throwing otjs path.
        #[cfg(test)]
        if self.force_no_hb_face.get() {
            return VectorShapeRoute::Degraded;
        }
        match entry.hb_face() {
            Some(hb) => VectorShapeRoute::HarfBuzz(hb),
            None => VectorShapeRoute::Degraded,
        }
    }

    /// `VectorFontCache.getMetricsForVariant` (VectorFontCache.ts:73-75).
    pub fn metrics_for_variant<'v>(
        &self,
        catalog: &Catalog,
        family: &str,
        weight: i64,
        style: &str,
        size_px: impl Into<Instance<'v>>,
    ) -> Option<TtfMetrics> {
        self.font_for_variant(catalog, family, weight, style, size_px)
            .and_then(|f| f.metrics.clone())
    }
}

/// Instance coordinates for a face with `axes`: the catalog variant's pinned
/// coordinates (dropping tags the face lacks), plus `opsz` from the font size
/// when the face has an optical-size axis the variant leaves free. Empty for a
/// static face, so a static face's instance key is just its url.
fn instance_coords(
    axes: &[AxisInfo],
    variations: &[Variation],
    instance: Instance<'_>,
) -> Vec<Variation> {
    let size_px = instance.size_px;
    if axes.is_empty() {
        return Vec::new();
    }
    let mut coords: Vec<Variation> = variations
        .iter()
        .filter(|(tag, _)| axes.iter().any(|a| &a.tag == tag))
        .cloned()
        .collect();
    // E6.3: the run's `font-variation-settings` win over the variant's own
    // coordinates (as CSS puts them over `font-weight`), clamped to the axis.
    for (tag, v) in instance.variations {
        if let Some(axis) = axes.iter().find(|a| &a.tag == tag) {
            coords.retain(|(t, _)| t != tag);
            coords.push((tag.clone(), v.clamp(axis.min, axis.max)));
        }
    }
    if let (Some(opsz), Some(size)) = (axes.iter().find(|a| a.tag == "opsz"), size_px) {
        if !coords.iter().any(|(t, _)| t == "opsz") && size.is_finite() && size > 0.0 {
            // Quarter-px steps bound the instance count for animated sizes.
            let v = ((size * 4.0).round() / 4.0) as f32;
            coords.push(("opsz".to_string(), v.clamp(opsz.min, opsz.max)));
        }
    }
    coords.sort_by(|a, b| a.0.cmp(&b.0));
    coords
}

fn instance_key(url: &str, coords: &[Variation]) -> String {
    if coords.is_empty() {
        return url.to_string();
    }
    let mut key = String::from(url);
    key.push('#');
    for (i, (tag, v)) in coords.iter().enumerate() {
        if i > 0 {
            key.push(',');
        }
        key.push_str(&format!("{tag}={v}"));
    }
    key
}

/// `normalizeTextForShaping` (vectorLayout.ts:69-81): tabs → space; other
/// whitespace the font can't map → space; non-whitespace never substituted.
/// The two identical branches are kept separate on purpose — they mirror the
/// TS conditions line-for-line.
#[allow(clippy::if_same_then_else)]
pub fn normalize_text_for_shaping(font: &OtjsFont, text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch == '\t' {
            out.push(' ');
        } else if js_whitespace(ch) && font.char_to_glyph_index(ch) == 0 {
            out.push(' ');
        } else {
            out.push(ch);
        }
    }
    out
}

/// `ShapedGlyph` (glyphPlacement.ts:13-28, Phase 3) — one shaped glyph ready
/// for placement, in font units. `x_offset`/`y_offset` are 0 on the opentype.js
/// path (no GPOS). `cluster` is NOT the glyph ordinal — it is the source
/// CODE-POINT index, uniform across all shaping paths (see the field doc below);
/// the per-glyph KineticText transform is addressed by it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapedGlyph {
    /// Glyph index (curve/tess cache key).
    pub glyph: u32,
    /// Pen advance in font units (kerning toward the NEXT glyph folded in on
    /// the opentype path).
    pub x_advance: f64,
    /// Placement offset from the pen in font units (HarfBuzz GPOS; 0 here).
    pub x_offset: f64,
    /// Offset from the baseline in font units, y-up (HarfBuzz GPOS; 0 here).
    pub y_offset: f64,
    /// Source CODE-POINT index of this glyph in its run text — letter spacing
    /// applies once per cluster (glyphPlacement.ts:23-27), and KineticText maps
    /// `unitTransforms[cluster]`. On the otjs path it is the source
    /// code point (a ligature survivor keeps its FIRST component's index, the
    /// rest deleted); the GSUB-free degraded path emits one glyph per code point
    /// so it coincides with the glyph index; both HarfBuzz shapers
    /// (`shape_vector_hb` / `shape_canvas2d_hb`) convert rustybuzz BYTE clusters
    /// to code-point indices so the value is uniform across every shaping path.
    pub cluster: usize,
}

/// `shapeWithKerning` (glyphPlacement.ts:43-51): wrap opentype-shaped glyph
/// ids as `ShapedGlyph`s, folding each pair's kerning into the leading
/// glyph's advance — pen positions identical to the historical
/// kern-before-glyph walk.
pub fn shape_with_kerning(font: &OtjsFont, gids: &[u32]) -> Result<Vec<ShapedGlyph>, ShapeError> {
    // One glyph per code point (no GSUB substitution), so the source cluster IS
    // the glyph index — used by the GSUB-free degraded path.
    let pairs: Vec<(u32, usize)> = gids.iter().enumerate().map(|(i, &g)| (g, i)).collect();
    shape_with_kerning_clustered(font, &pairs)
}

/// `shapeWithKerning` carrying each glyph's SOURCE CODE-POINT cluster (from
/// `string_to_glyphs_clustered`), so a ligature survivor reports its first
/// component's code point (KineticText). Kerning + advances are
/// glyph-based, so pen positions are identical to [`shape_with_kerning`];
/// only `cluster` carries the source index instead of the glyph ordinal.
pub fn shape_with_kerning_clustered(
    font: &OtjsFont,
    pairs: &[(u32, usize)],
) -> Result<Vec<ShapedGlyph>, ShapeError> {
    let mut shaped = Vec::with_capacity(pairs.len());
    for (i, &(g, cluster)) in pairs.iter().enumerate() {
        // `font.getKerningValue(glyph, glyphs[i + 1]) || 0`
        let kern = if i + 1 < pairs.len() {
            font.get_kerning_value(g, pairs[i + 1].0)?
        } else {
            0.0
        };
        shaped.push(ShapedGlyph {
            glyph: g,
            // `(glyph.advanceWidth || 0) + kern`
            x_advance: font.advance_width(g) + kern,
            x_offset: 0.0,
            y_offset: 0.0,
            cluster,
        });
    }
    Ok(shaped)
}

/// `VectorGlyphShaper.shape` (vectorLayout.ts:115-135) minus the cache:
/// normalize + `stringToGlyphs` + `shapeWithKerning`. The production
/// dispatcher's complex-script / RTL HarfBuzz branch is deliberately absent
/// — complex text that would diverge is unreachable (coverage fallback) or
/// errors loudly (otjs.rs Arabic guard, Q-118). The FONT-capability HarfBuzz
/// branch lives in [`shape_vector`], which is the vector
/// lane's entry point; this bare otjs walk also serves the hybrid lane's
/// probe-free paths.
pub fn shape(font: &OtjsFont, text: &str) -> Result<Vec<ShapedGlyph>, ShapeError> {
    // Clustered so each glyph carries its source code-point index (ligature
    // survivor → first component), which KineticText maps to unitTransforms.
    let pairs = font.string_to_glyphs_clustered(&normalize_text_for_shaping(font, text))?;
    shape_with_kerning_clustered(font, &pairs)
}

/// Backstop: GSUB-free codepoint shaping for fonts the otjs walk
/// cannot shape (type-7 GSUB extension lookups, e.g. Lato) WHEN no HarfBuzz
/// reroute face is available. Maps each char straight through the cmap (the
/// raw `charToGlyphIndex` step `string_to_glyphs` uses, with NO feature
/// substitution) and applies kerning via `shape_with_kerning`, producing the
/// exact `ShapedGlyph` shape the opentype path emits. The run keeps rendering
/// in the vector engine (legible Latin, loses ligatures/contextual subs)
/// instead of erroring → whole-node MSDF fallback EVERY frame; it self-heals
/// to HarfBuzz the moment a reroute face becomes available (see
/// [`shape_vector`]). Infallible by construction — the cmap walk never runs a
/// throwing lookup, and `shape_with_kerning` only errors on malformed GPOS
/// kern tables, which a font reaching this backstop has already parsed.
pub fn shape_degraded(font: &OtjsFont, text: &str) -> Result<Vec<ShapedGlyph>, ShapeError> {
    let gids: Vec<u32> = normalize_text_for_shaping(font, text)
        .chars()
        .map(|ch| font.char_to_glyph_index(ch))
        .collect();
    shape_with_kerning(font, &gids)
}

/// How the vector lane shapes one variant's run (resolved once, per frame).
enum VectorShapeRoute {
    /// otjs-compatible font: the byte-exact opentype codepoint+kern walk.
    Otjs,
    /// otjs-incompatible (type-7 GSUB), HarfBuzz face available: full reroute.
    HarfBuzz(Rc<HbFace>),
    /// otjs-incompatible AND no HarfBuzz face: the GSUB-free degraded backstop.
    Degraded,
}

/// The VECTOR lane's shaping dispatch — the mirror of
/// `VectorGlyphShaper.shape`'s font-capability branch. Fonts the otjs walk
/// cannot shape (per-font probe memo: Lato's type-7 GSUB extension lookups)
/// route through rustybuzz against the same vendored bytes. When such a font
/// has NO usable HarfBuzz face (face creation failed), the run falls to the
/// GSUB-free degraded codepoint backstop ([`shape_degraded`]) instead of
/// re-throwing the opentype `UnsupportedLookup` every frame: the node keeps
/// rendering legible (degraded) Latin in the vector engine rather than
/// declining whole-node to MSDF on every frame, and self-heals to HarfBuzz
/// once a face becomes available. otjs-compatible fonts keep the opentype
/// codepoint path byte-exact. The capability probe is memoized per parsed
/// font (`needs_hb` cell), so the throwing opentype path is never re-run for a
/// known-incompatible font — that memo is the negative cache.
#[allow(clippy::too_many_arguments)]
pub fn shape_vector<'v>(
    store: &VectorFontStore,
    catalog: &Catalog,
    family: &str,
    weight: i64,
    style: &str,
    size_px: impl Into<Instance<'v>>,
    font: &OtjsFont,
    text: &str,
) -> Result<Vec<ShapedGlyph>, ShapeError> {
    match store.vector_shape_route(catalog, family, weight, style, size_px) {
        VectorShapeRoute::Otjs => shape(font, text),
        VectorShapeRoute::HarfBuzz(hb) => shape_vector_hb(&hb, font, text),
        VectorShapeRoute::Degraded => shape_degraded(font, text),
    }
}

/// Shape with HarfBuzz's default OpenType features plus the run's explicit overrides.
/// An empty feature list means normal shaping, not a different shaper: kerning
/// (including GPOS extension lookups), ligatures and contextual substitutions
/// must agree with the same run explicitly requesting a default feature.
/// Shared by drawing, measurement, layout dumps and text extraction.
///
/// `None` only when the variant has no HarfBuzz face; the caller then uses its
/// existing font fallback.
#[allow(clippy::too_many_arguments)]
pub fn shape_featured<'v>(
    store: &VectorFontStore,
    catalog: &Catalog,
    family: &str,
    weight: i64,
    style: &str,
    size_px: impl Into<Instance<'v>>,
    font: &OtjsFont,
    text: &str,
    features: &[crate::font_features::FontFeature],
) -> Option<Result<Vec<ShapedGlyph>, ShapeError>> {
    let hb = store.hb_face_for_variant(catalog, family, weight, style, size_px)?;
    Some(crate::vector::hb::shape_features_hb(
        &hb, font, text, features,
    ))
}

/// Shape one laid-out segment for the draw (`ops.rs`), the layout dump (`node_layout.rs`) and extraction
/// (`extract.rs`), with the run's character attributes applied (`crate::character`): one dispatch, so what is drawn
/// is what the caret and the exports see. All runs shape through rustybuzz with
/// default features and explicit overrides (including optical / no kerning).
#[allow(clippy::too_many_arguments)]
pub fn shape_segment<'v>(
    store: &VectorFontStore,
    catalog: &Catalog,
    family: &str,
    weight: i64,
    st: &str,
    size_px: impl Into<Instance<'v>> + Copy,
    font: &OtjsFont,
    text: &str,
    style: &InlineStyle,
    placement: crate::vector::ops::VectorPlacement,
) -> Result<Vec<ShapedGlyph>, ShapeError> {
    use crate::vector::ops::VectorPlacement;
    let features = crate::character::shaping_features(&style.font_features, &style.character);
    // Vertical text: upright pieces shape with `vert` (`crate::vertical`).
    let vertical_features = crate::vertical::shaping_features(&features, &style.vertical);
    let features: &[crate::font_features::FontFeature] = vertical_features.as_deref().unwrap_or(&features);
    let mut shaped = if let Some(r) =
        shape_featured(store, catalog, family, weight, st, size_px, font, text, features)
    {
        r?
    } else if placement == VectorPlacement::Canvas2dParity && style.letter_spacing != 0.0 {
        shape_canvas2d_letter_spacing(store, catalog, family, weight, st, size_px, font, text)?
    } else if placement == VectorPlacement::Canvas2dParity {
        match store.hb_face_for_variant(catalog, family, weight, st, size_px) {
            Some(hb) => shape_canvas2d_hb(&hb, font, text)?,
            None => shape(font, text)?,
        }
    } else {
        shape_vector(store, catalog, family, weight, st, size_px, font, text)?
    };
    crate::character::adjust_shaped(font, &mut shaped, &style.character);
    crate::vertical::adjust_shaped(font, &mut shaped, &style.vertical);
    crate::ruby::adjust_shaped(font, &mut shaped, style);
    Ok(shaped)
}

/// Q-121 (FIX r3): the Canvas2D pipeline's letter-spacing lane. When
/// `letterSpacing != 0` the browser's fallback raster draws PER CHARACTER
/// (`drawTextWithLetterSpacing`, utils/canvas/text.ts:26-28 — one `fillText`
/// per grapheme advanced by `measureText(char).width + letterSpacing`) and
/// measures the same way (`measureTextWithLetterSpacing`,
/// utils/richtext/measurement.ts:197-213 — per-char widths + spacing BETWEEN
/// chars only). Per-char shaping means NO kerning, NO cross-character GSUB
/// (liga/calt can never see a context) — so this lane maps each code point
/// straight through cmap and uses bare hmtx advances. One cluster per code
/// point; combining sequences therefore over-segment relative to the
/// browser's grapheme segmenter (recorded deviation, quirk Q-121).
/// ELIGIBILITY mirrors the browser vector engine's shaping dispatch: the
/// [`shape_vector`] walk runs first and its errors (Q-118 Arabic guard,
/// malformed tables) still decline the node to the Degraded tier exactly
/// like `VectorParity` — a node the browser's vector engine cannot shape
/// must not become vector-eligible just because the hybrid arm would place
/// it per-char (FIX r3 round-2 finding: bypassing the guard re-routed Arabic
/// canvases to ISOLATED forms, aad05362 −0.009..−0.024 vs msdf). Now
/// the browser dispatch reroutes otjs-incompatible fonts (Lato's
/// type-7 liga lookups) through HarfBuzz instead of throwing, so the probe
/// here does too — letter-spaced Lato nodes are now vector-eligible on the
/// hybrid arm, matching the regenerated oracle.
#[allow(clippy::too_many_arguments)]
pub fn shape_canvas2d_letter_spacing<'v>(
    store: &VectorFontStore,
    catalog: &Catalog,
    family: &str,
    weight: i64,
    style: &str,
    size_px: impl Into<Instance<'v>>,
    font: &OtjsFont,
    text: &str,
) -> Result<Vec<ShapedGlyph>, ShapeError> {
    // Eligibility-parity probe (result discarded; metrics below are per-char).
    let _ = shape_vector(store, catalog, family, weight, style, size_px, font, text)?;
    Ok(normalize_text_for_shaping(font, text)
        .chars()
        .enumerate()
        .map(|(i, ch)| {
            let gid = font.char_to_glyph_index(ch);
            ShapedGlyph {
                glyph: gid,
                x_advance: font.advance_width(gid),
                x_offset: 0.0,
                y_offset: 0.0,
                cluster: i,
            }
        })
        .collect())
}

/// `placeGlyphs` (glyphPlacement.ts:74-91, Phase 3): pen walk over pre-shaped
/// glyphs — `visit` gets each glyph's placed x origin (pen + shaping
/// x-offset) and y-up baseline offset in px; pen advances by
/// `xAdvance·scale`, plus letter spacing after every CLUSTER including the
/// last (per-glyph on the opentype path, Q-106). Returns the final pen x.
pub fn place_glyphs<F: FnMut(u32, f64, f64, usize)>(
    font: &OtjsFont,
    shaped: &[ShapedGlyph],
    font_size_px: f64,
    start_x: f64,
    mut visit: F,
    letter_spacing_px: f64,
) -> f64 {
    // `fontSizePx / (font.unitsPerEm || 1000)` (glyphPlacement.ts:82)
    let upm = if font.units_per_em != 0 {
        font.units_per_em as f64
    } else {
        1000.0
    };
    let scale = font_size_px / upm;
    let mut pen_x = start_x;
    for (i, s) in shaped.iter().enumerate() {
        visit(s.glyph, pen_x + s.x_offset * scale, s.y_offset * scale, i);
        pen_x += s.x_advance * scale;
        if i == shaped.len() - 1 || shaped[i + 1].cluster != s.cluster {
            pen_x += letter_spacing_px;
        }
    }
    pen_x
}

/// `measureGlyphRun` (glyphPlacement.ts:99-101).
///
/// MIRRORED IN PYTHON: `backend/app/services/pptx/font_metrics.py` reimplements
/// this accumulation (and `place_glyphs` above) so the PDF importer can size a
/// text box to the width this lane will draw. It is pinned to constants captured
/// FROM here, so it cannot notice this file changing underneath it -- if you
/// alter how advances, kerning, ligatures or letter spacing accumulate, that
/// module and `backend/app/services/pptx/tests/test_font_metrics.py` need the
/// same change and a re-capture.
pub fn measure_glyph_run(
    font: &OtjsFont,
    shaped: &[ShapedGlyph],
    font_size_px: f64,
    letter_spacing_px: f64,
) -> f64 {
    place_glyphs(
        font,
        shaped,
        font_size_px,
        0.0,
        |_, _, _, _| {},
        letter_spacing_px,
    )
}

/// Q-121: `measureTextWithLetterSpacing` (utils/richtext/measurement.ts:
/// 197-213) — per-char widths with letter spacing BETWEEN clusters only
/// (`if (i < chars.length - 1) width += letterSpacing`), no trailing term.
/// The vector engine's `measureGlyphRun` adds spacing after EVERY cluster
/// including the last; the Canvas2D pipeline does not — a one-`ls` per-run
/// difference that compounds at every word gap (word + trailing-ls + space
/// + trailing-ls vs word + space).
pub fn measure_glyph_run_canvas2d(
    font: &OtjsFont,
    shaped: &[ShapedGlyph],
    font_size_px: f64,
    letter_spacing_px: f64,
) -> f64 {
    let w = measure_glyph_run(font, shaped, font_size_px, letter_spacing_px);
    if shaped.is_empty() {
        w
    } else {
        w - letter_spacing_px
    }
}

/// `createVectorMeasurer` (vectorLayout.ts:146-181) behind the shared
/// `TextMeasurer` trait. The JS measurer can THROW (opentype.js shaping on a
/// font the capability probe missed, malformed tables) and
/// `renderWithVector`'s catch turns that into a whole-node MSDF fallback;
/// the trait returns plain f64, so the
/// first error is recorded as poison (and 0 returned) for the caller to
/// surface — the call order matches JS, so the poisoned error is the same
/// error JS would have thrown first.
pub struct VectorMeasurer<'a> {
    pub store: &'a VectorFontStore,
    pub catalog: &'a Catalog,
    pub poison: RefCell<Option<ShapeError>>,
    /// Q-121: under `Canvas2dParity` (the hybrid arm) letter-spaced text
    /// measures with the Canvas2D pipeline's semantics (per-char unkerned,
    /// spacing between clusters only); `VectorParity` keeps the browser
    /// vector engine's `measureGlyphRun` byte-exact.
    pub placement: crate::vector::ops::VectorPlacement,
}

impl<'a> VectorMeasurer<'a> {
    pub fn new(store: &'a VectorFontStore, catalog: &'a Catalog) -> Self {
        VectorMeasurer {
            store,
            catalog,
            poison: RefCell::new(None),
            placement: crate::vector::ops::VectorPlacement::VectorParity,
        }
    }

    pub fn with_placement(
        store: &'a VectorFontStore,
        catalog: &'a Catalog,
        placement: crate::vector::ops::VectorPlacement,
    ) -> Self {
        VectorMeasurer {
            store,
            catalog,
            poison: RefCell::new(None),
            placement,
        }
    }

    pub fn take_poison(&self) -> Option<ShapeError> {
        self.poison.borrow().clone()
    }

    fn poison_with(&self, e: ShapeError) {
        let mut p = self.poison.borrow_mut();
        if p.is_none() {
            *p = Some(e);
        }
    }
}

impl TextMeasurer for VectorMeasurer<'_> {
    fn auto_line_height_px(&self, style: &InlineStyle) -> Option<f64> {
        let (family, weight, st) = style_to_variant(style);
        let m = self
            .store
            .metrics_for_variant(self.catalog, &family, weight, st, Instance::of(style))?;
        crate::measure::auto_line_height_from_metrics(&m, style.line_font_size())
    }

    fn measure(&self, text: &str, style: &InlineStyle) -> f64 {
        if text.is_empty() {
            return 0.0;
        }
        // Emoji placeholders (`runs::EMOJI_PLACEHOLDER`) take 1 em + letter spacing each,
        // the rest shapes as text (vectorLayout.ts `makeVectorMeasurer`).
        if text.contains(crate::runs::EMOJI_PLACEHOLDER) {
            let mut width = 0.0;
            for (i, chunk) in text.split(crate::runs::EMOJI_PLACEHOLDER).enumerate() {
                if i > 0 {
                    width += style.font_size + style.letter_spacing;
                }
                width += self.measure(chunk, style);
            }
            return width;
        }
        let letter_spacing = style.letter_spacing; // `style.letterSpacing || 0`
        let (family, weight, st) = style_to_variant(style);
        let Some(font) =
            self.store
                .font_for_variant(self.catalog, &family, weight, st, Instance::of(style))
        else {
            // vectorLayout.ts:149-154 — count code points (for..of)
            let count = text.chars().count() as f64;
            return count * (style.font_size * MISSING_FONT_ADVANCE_RATIO + letter_spacing);
        };
        // Runs measure with the glyphs the draw places
        // (`shape_featured` — the same call in `ops.rs`), with the vector
        // engine's per-cluster spacing, on either placement.
        // E8.22: optical / no kerning turn `kern` off here too (`character::shaping_features`), and every measured
        // run gets the same advance changes the draw places (`character::adjust_shaped`).
        let features = crate::character::shaping_features(&style.font_features, &style.character);
        let vertical_features = crate::vertical::shaping_features(&features, &style.vertical);
        let features: &[crate::font_features::FontFeature] = vertical_features.as_deref().unwrap_or(&features);
        let adjust = |mut shaped: Vec<ShapedGlyph>| {
            crate::character::adjust_shaped(&font, &mut shaped, &style.character);
            crate::vertical::adjust_shaped(&font, &mut shaped, &style.vertical);
            crate::ruby::adjust_shaped(&font, &mut shaped, style);
            shaped
        };
        if let Some(r) = shape_featured(
            self.store,
            self.catalog,
            &family,
            weight,
            st,
            Instance::of(style),
            &font,
            text,
            features,
        ) {
            return match r {
                Ok(shaped) => measure_glyph_run(&font, &adjust(shaped), style.font_size, letter_spacing),
                Err(e) => {
                    self.poison_with(e);
                    0.0
                }
            };
        }
        let canvas2d = matches!(
            self.placement,
            crate::vector::ops::VectorPlacement::Canvas2dParity
        );
        // Q-121: the hybrid arm measures letter-spaced text the way the
        // Canvas2D pipeline (the prod golden's engine for fallback-class
        // nodes) does — per-char unkerned, spacing between clusters only.
        if canvas2d && letter_spacing != 0.0 {
            match shape_canvas2d_letter_spacing(
                self.store,
                self.catalog,
                &family,
                weight,
                st,
                Instance::of(style),
                &font,
                text,
            ) {
                Ok(shaped) => {
                    return measure_glyph_run_canvas2d(
                        &font,
                        &adjust(shaped),
                        style.font_size,
                        letter_spacing,
                    );
                }
                Err(e) => {
                    // Same poison-decline path as the vector-parity lane.
                    self.poison_with(e);
                    return 0.0;
                }
            }
        }
        // Q-122 (Phase 2): ls==0 runs on the hybrid arm measure with the
        // Canvas2D pipeline's HarfBuzz semantics (`measureText` — kerning
        // through extension lookups + default GSUB). No otjs shape gate on
        // this lane: eligibility is HB's, exactly like the golden's engine.
        // A missing HB face falls through to the opentype path (the browser
        // shaper's null fallback, harfbuzzShaper.ts:163-165).
        if canvas2d && letter_spacing == 0.0 {
            if let Some(hb) = self.store.hb_face_for_variant(
                self.catalog,
                &family,
                weight,
                st,
                Instance::of(style),
            ) {
                match shape_canvas2d_hb(&hb, &font, text) {
                    Ok(shaped) => {
                        return measure_glyph_run(&font, &adjust(shaped), style.font_size, letter_spacing);
                    }
                    Err(e) => {
                        self.poison_with(e);
                        return 0.0;
                    }
                }
            }
        }
        // VectorParity (and the hybrid arm's HB-face-less fallthrough): the
        // vector lane's dispatch — otjs, with the HarfBuzz reroute
        // for fonts otjs cannot shape.
        match shape_vector(
            self.store,
            self.catalog,
            &family,
            weight,
            st,
            Instance::of(style),
            &font,
            text,
        ) {
            Ok(shaped) => measure_glyph_run(&font, &adjust(shaped), style.font_size, letter_spacing),
            Err(e) => {
                self.poison_with(e);
                0.0
            }
        }
    }
}

#[cfg(test)]
#[cfg_attr(not(vork_private_tests), allow(dead_code, unused_imports))]
mod tests {
    use super::*;
    use crate::test_fixtures::{fonts_root, store_and_catalog};

    /// A url label must identify the miss (host + file name) while
    /// dropping the query, where a signed font URL's credential lives.
    #[test]
    fn font_url_label_keeps_host_and_basename_drops_the_signature() {
        // `<host>/<basename>#<digest-of-the-full-url>`.
        let label = |u: &str| font_url_label(u);
        let head = |u: &str| label(u).split('#').next().unwrap().to_string();
        assert_eq!(
            head("https://fonts.moda.app/platform/OpenSans-Bold.ttf"),
            "fonts.moda.app/OpenSans-Bold.ttf"
        );
        let signed = "https://cdn.moda.app/team/abc/Roboto.ttf?Expires=1&Signature=SECRET&x=y#frag";
        assert_eq!(head(signed), "cdn.moda.app/Roboto.ttf");
        // No part of the credential survives, anywhere in the label.
        assert!(!label(signed).contains("SECRET") && !label(signed).contains("Expires"));
        // But signature skew is still DISTINGUISHABLE from a vendoring gap — the
        // §7 hypothesis this label exists to test needs exactly that.
        let resigned = "https://cdn.moda.app/team/abc/Roboto.ttf?Expires=2&Signature=OTHER";
        assert_ne!(label(signed), label(resigned));
        assert_eq!(head(signed), head(resigned));
        // Stable across calls, so two log lines for the same url match up.
        assert_eq!(label(signed), label(signed));
        // Degenerate shapes must not panic or leak the query.
        assert_eq!(head("https://cdn.moda.app"), "cdn.moda.app");
        assert_eq!(head("Inter.ttf?sig=s"), "Inter.ttf");
        assert_eq!(head(""), "unparseable");
        // Userinfo is the one other credential-bearing URL component.
        assert_eq!(
            head("https://user:pass@cdn.moda.app/a/Roboto.ttf"),
            "cdn.moda.app/Roboto.ttf"
        );
        // A data: URI is payload, not identity — never echo it.
        assert_eq!(font_url_label("data:font/ttf;base64,AAEAAAAL"), "data-uri");
        assert_eq!(font_url_label("DATA:font/ttf;base64,AAEA"), "data-uri");
        // A non-ASCII lead byte must not panic the data: probe.
        assert_eq!(head("日本語フォント.ttf"), "日本語フォント.ttf");
        // Truncation is char-boundary safe (a percent-decoded key can be non-ASCII).
        let long = format!("h.example/{}.ttf", "é".repeat(200));
        assert_eq!(head(&long).chars().count(), URL_LABEL_MAX_CHARS);
    }

    /// The last-variant memo answers only the arguments and the catalog it was filled with: a different weight, size or
    /// catalog after it gets its own font, and the first arguments again get theirs.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn load_variant_memo_follows_the_arguments_and_the_catalog() {
        let (store, catalog) = store_and_catalog();
        let font = |catalog: &Catalog, weight: i64, size: Option<f64>| {
            store
                .font_for_variant(catalog, "Inter", weight, "normal", size)
                .expect("Inter is vendored")
        };
        let regular = font(&catalog, 400, None);
        let bold = font(&catalog, 800, None);
        assert!(!Rc::ptr_eq(&regular, &bold));
        assert!(Rc::ptr_eq(&font(&catalog, 400, Some(12.0)), &regular));
        assert!(Rc::ptr_eq(&font(&catalog, 800, Some(12.0)), &bold));
        // Another catalog whose Inter 400 is the bold file, asked right after this catalog's Inter 400.
        let bold_url = catalog.get_variant_ttf_path("Inter", 800, "normal").unwrap();
        let other = Catalog::from_catalog_json(&serde_json::json!({ "platform_fonts": [{
            "family_name": "Inter",
            "variants": [{ "weight": 400, "style": "normal", "ttf_url": bold_url }],
        }] }));
        assert_ne!(other.id(), catalog.id());
        assert!(Rc::ptr_eq(&font(&catalog, 400, None), &regular));
        assert!(Rc::ptr_eq(&font(&other, 400, None), &bold));
        assert!(Rc::ptr_eq(&font(&catalog, 400, None), &regular));
        // A family the catalog lacks misses after a hit, and the hit stays.
        assert!(store
            .font_for_variant(&catalog, "No Such Family", 400, "normal", None)
            .is_none());
        assert!(Rc::ptr_eq(&font(&catalog, 400, None), &regular));
    }

    /// The three ways a vendored TTF fails to become a parsed font are
    /// reported DISTINCTLY instead of collapsing into one `font-unavailable`.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn font_for_variant_result_classifies_each_miss() {
        let (store, catalog) = store_and_catalog();
        // Resolvable variant — the control.
        assert!(store
            .font_for_variant_result(&catalog, "Inter", 400, "normal", None)
            .is_ok());
        // A weight the catalog has no EXACT variant for: a catalog gap, not a
        // vendoring or parse fault (`get_variant_ttf_path` never snaps).
        assert_eq!(
            store
                .font_for_variant_result(&catalog, "Inter", 137, "normal", None)
                .map(|_| ()),
            Err(VectorFontMiss::NoCatalogTtf)
        );
        // The catalog offers a url the manifest doesn't vendor — the not-vendored
        // class (a store built over a different fonts dir than the catalog).
        let empty = VectorFontStore::load_from_manifest_lenient(
            &fonts_root(),
            &serde_json::json!({ "files": {} }),
        );
        let miss = empty
            .font_for_variant_result(&catalog, "Inter", 400, "normal", None)
            .err()
            .expect("an un-vendored url must miss");
        assert_eq!(miss.class(), "not-vendored");
        assert!(
            miss.detail().starts_with("not-vendored "),
            "detail names the url that missed: {}",
            miss.detail()
        );
        // Vendored but unparseable bytes. PID-scoped so parallel test binaries
        // (and sibling worktrees) never share — or delete — each other's dir.
        let bad_dir =
            std::env::temp_dir().join(format!("moda-eng4041-badfont-{}", std::process::id()));
        std::fs::create_dir_all(&bad_dir).unwrap();
        std::fs::write(bad_dir.join("bad.ttf"), b"not a font").unwrap();
        let url = catalog
            .get_variant_ttf_path("Inter", 400, "normal")
            .expect("Inter 400 has a ttf_url");
        let broken = VectorFontStore::load_from_manifest_lenient(
            &bad_dir,
            &serde_json::json!({ "files": { url.clone(): { "path": "bad.ttf" } } }),
        );
        // Twice: the parse failure is negative-cached, and the SECOND lookup
        // must still report `parse-failed` rather than a different class.
        for _ in 0..2 {
            assert_eq!(
                broken
                    .font_for_variant_result(&catalog, "Inter", 400, "normal", None)
                    .err()
                    .map(|m| m.class()),
                Some("parse-failed")
            );
        }
        let _ = std::fs::remove_dir_all(&bad_dir);
    }

    /// CROSS-LANGUAGE CONTRACT: the `cause` strings the backend's WARN
    /// allowlist matches literally (`router._TEXT_FALLBACK_WARN_CAUSES`).
    ///
    /// Renaming one silently disables the alert forever — the share still
    /// publishes and the histogram still populates, so nothing looks broken; the
    /// alert simply never fires again. That is this ticket's own failure shape, so
    /// the composed value is pinned here rather than only the enum variant name.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn font_unavailable_cause_strings_match_the_backend_allowlist() {
        use crate::vector::{build_vector_node_ops, VectorEngineDecision, VectorPlacement};

        let (_, catalog) = store_and_catalog();
        // A store that vendors nothing: every catalog url misses (the class).
        let empty = VectorFontStore::load_from_manifest_lenient(
            &fonts_root(),
            &serde_json::json!({ "files": {} }),
        );
        let spec = crate::ops::TextNodeSpec {
            node_id: "n".into(),
            html: "<p><span style=\"font-family: Inter; font-weight: 400\">hi</span></p>".into(),
            ..Default::default()
        };
        let VectorEngineDecision::Fallback {
            cause, families, ..
        } = build_vector_node_ops(&spec, &empty, &catalog, 1.0, VectorPlacement::VectorParity)
        else {
            panic!("an un-vendored store must decline");
        };
        assert_eq!(cause, "font-unavailable:not-vendored");
        assert_eq!(families, vec!["Inter".to_string()]);

        // The other three classes, composed the same way. `read-failed` and
        // `parse-failed` are in the backend allowlist too; `no-catalog-ttf` is
        // deliberately NOT (a stable catalog property, not a fault).
        for class in [
            "not-vendored",
            "read-failed",
            "parse-failed",
            "no-catalog-ttf",
        ] {
            assert!(
                format!("font-unavailable:{class}").starts_with("font-unavailable:"),
                "the backend matches on the `font-unavailable:` stem"
            );
        }
        assert_eq!(VectorFontMiss::NoCatalogTtf.class(), "no-catalog-ttf");
        assert_eq!(
            VectorFontMiss::NotVendored { url: String::new() }.class(),
            "not-vendored"
        );
        assert_eq!(
            VectorFontMiss::ReadFailed { url: String::new() }.class(),
            "read-failed"
        );
        assert_eq!(
            VectorFontMiss::ParseFailed { url: String::new() }.class(),
            "parse-failed"
        );
    }

    /// Backstop: a font whose `string_to_glyphs` throws
    /// `UnsupportedLookup` (Lato's type-7 GSUB extension liga lookups — the
    /// exact `lookupType: 7` throw) yields a NON-EMPTY degraded run via the
    /// GSUB-free codepoint shaper, Ok — it never propagates the error. The
    /// degraded run maps each code point straight through the cmap (no
    /// ligatures), so "Office" stays 6 glyphs (vs HB's ffi-ligated < 6).
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn lato_degraded_shaper_is_ok_and_nonempty_where_otjs_throws() {
        let (store, catalog) = store_and_catalog();
        let font = store
            .font_for_variant(&catalog, "Lato", 400, "normal", None)
            .unwrap();

        // The opentype walk still throws (port parity).
        assert!(
            matches!(
                shape(&font, "Office"),
                Err(ShapeError::UnsupportedLookup { lookup_type: 7, .. })
            ),
            "Lato's liga must be a type-7 throw"
        );

        // The degraded backstop is Ok, non-empty, and GSUB-free (no ligature).
        let degraded = shape_degraded(&font, "Office").expect("degraded shaping must not error");
        assert_eq!(
            degraded.len(),
            "Office".chars().count(),
            "degraded is codepoint-only — no ffi ligature"
        );
        // Each glyph is the raw cmap mapping, kerning folded into the advance.
        for (i, ch) in "Office".chars().enumerate() {
            assert_eq!(degraded[i].glyph, font.char_to_glyph_index(ch));
        }
    }

    /// Negative cache: once a font is known otjs-incompatible, the
    /// vector dispatch never re-runs the throwing opentype path. With the HB
    /// face forced unavailable, `shape_vector` returns the degraded run (Ok)
    /// instead of erroring — so a measure/ops call would render via the vector
    /// engine degraded, not fall back to MSDF every frame.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn shape_vector_falls_to_degraded_when_no_hb_face() {
        let (store, catalog) = store_and_catalog();
        store.force_no_hb_face();
        let font = store
            .font_for_variant(&catalog, "Lato", 400, "normal", None)
            .unwrap();

        let shaped = shape_vector(
            &store, &catalog, "Lato", 400, "normal", None, &font, "Office",
        )
        .expect("degraded backstop must keep shape_vector Ok, not propagate the type-7 throw");
        assert_eq!(shaped.len(), "Office".chars().count());

        // And the measurer yields a positive width (renders) rather than the
        // poison-decline 0.0 the old re-throw-every-frame path produced.
        let measurer = VectorMeasurer::new(&store, &catalog);
        let style = InlineStyle {
            font_family: "Lato".into(),
            font_size: 24.0,
            ..crate::types::default_style()
        };
        let w = measurer.measure("Office", &style);
        assert!(w > 0.0, "degraded Lato run must measure > 0, got {w}");
        assert!(
            measurer.take_poison().is_none(),
            "degraded path must not poison the measurer"
        );
    }
}
