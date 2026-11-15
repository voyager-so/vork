//! Font-catalog consumption — ports of `utils/fontStore.ts:223-225`
//! (case-insensitive family lookup), `core/config/msdfFonts.ts` (MSDF
//! readiness + nearest-variant snapping) and `core/config/msdfFallbacks.ts`
//! (the Noto punctuation-fallback cascade).

use crate::family_normalization::{family_name_candidates, CandidateKind};
use crate::types::FontKey;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// CSS default weight — i.e. "the caller did not ask for a particular weight"
/// (`DEFAULT_FONT_WEIGHT`, `utils/fontStore.ts`).
const DEFAULT_FONT_WEIGHT: i64 = 400;

/// Upper bound on the normalized-lookup memo. Family names come from document
/// content, so this is generous; the cap just keeps a pathological doc bounded.
const RESOLUTION_CACHE_CAP: usize = 512;

/// A variant's outline file and its variable-font coordinates ([`Catalog::get_variant_ttf_source`]), shared by every
/// lookup of the same variant.
pub type TtfSource = Arc<(String, Vec<crate::vector::varfont::Variation>)>;

/// The variants one asked-for family name was looked up at: weight, style, source.
type TtfSourcesOfFamily = Vec<(i64, String, Option<TtfSource>)>;

#[derive(Debug, Clone)]
pub struct CatalogVariant {
    pub weight: i64,
    /// 'normal' | 'italic'
    pub style: String,
    pub msdf_ready: bool,
    pub atlas_urls: Vec<String>,
    pub metrics_url: Option<String>,
    pub ttf_url: Option<String>,
    /// Desktop fork: variable-font instance coordinates for `ttf_url` (axis
    /// tag → user-space value, e.g. a named instance's `wght`/`wdth`). Empty for
    /// a static face. `opsz` is normally absent: it then follows the font size
    /// (see `VectorFontStore`).
    pub variations: Vec<crate::vector::varfont::Variation>,
}

#[derive(Debug, Clone)]
pub struct CatalogFamily {
    pub family_name: String,
    pub variants: Vec<CatalogVariant>,
    /// The catalog's `category` (`serif`, `sans-serif`, `monospace`, …), lowercased; empty when absent.
    pub category: String,
}

/// A catalog variant's `variations` object (`{"wght": 700, "wdth": 75}`).
fn parse_variations(v: Option<&Value>) -> Vec<crate::vector::varfont::Variation> {
    let Some(obj) = v.and_then(|v| v.as_object()) else {
        return Vec::new();
    };
    let mut out: Vec<crate::vector::varfont::Variation> = obj
        .iter()
        .filter_map(|(tag, value)| Some((tag.clone(), value.as_f64()? as f32)))
        .filter(|(tag, _)| tag.len() == 4)
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// `MsdfVariant` (`msdfFonts.ts:12`).
#[derive(Debug, Clone, PartialEq)]
pub struct MsdfVariant {
    pub weight: i64,
    pub style: String,
}

/// A normalized (non-exact) family match: the catalog key it landed on plus the
/// weight/style the dropped name suffix encoded.
#[derive(Debug, Clone)]
struct NormalizedHit {
    key: String,
    weight: Option<i64>,
    style: Option<&'static str>,
    kind: CandidateKind,
}

#[derive(Debug, Default)]
pub struct Catalog {
    /// Which catalog this is ([`Self::id`]): unique per catalog read, 0 for the empty default one.
    id: u64,
    substitutes: HashMap<String, String>,
    /// lowercased family name → family (`fontStore.ts`); built platform-then-team
    /// in catalog order, later entries overwrite.
    by_name: HashMap<String, CatalogFamily>,
    /// Memo of the NORMALIZED lookup (`familyResolutionCache`, `fontStore.ts`).
    /// Only names that miss the exact map ever reach it, so the hot path is
    /// untouched; `None` records a genuine miss so it stays as cheap as a hit.
    normalized_cache: RwLock<HashMap<String, Option<NormalizedHit>>>,
    /// Memo of [`Self::get_variant_ttf_source`] by the family name as asked, then weight and style. The vector text
    /// lane asks it several times per word (each font, shaping-route and metrics lookup of a run); the catalog never
    /// changes, so an answer stays right for its lifetime.
    ttf_sources: RwLock<HashMap<String, TtfSourcesOfFamily>>,
}

impl Catalog {
    /// Build from the vendored `catalog.json`
    /// (`{platform_fonts: [...], team_fonts: [...]}` — the shape
    /// `setFontCatalog` ingests; `getAllFonts` = platform ++ team,
    /// `fontStore.ts:200-204`).
    pub fn from_catalog_json(v: &Value) -> Catalog {
        let mut by_name = HashMap::new();
        let mut substitutes = HashMap::new();
        for list_key in ["platform_fonts", "team_fonts"] {
            let Some(arr) = v.get(list_key).and_then(|x| x.as_array()) else {
                continue;
            };
            for fam in arr {
                let name = fam
                    .get("family_name")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                if name.is_empty() {
                    continue;
                }
                if let Some(drawn) = fam.get("substitute").and_then(Value::as_str) {
                    substitutes.insert(name.to_lowercase(), drawn.to_string());
                } else {
                    substitutes.remove(&name.to_lowercase());
                }
                let variants = fam
                    .get("variants")
                    .and_then(|x| x.as_array())
                    .map(|vs| {
                        vs.iter()
                            .filter_map(|vv| {
                                Some(CatalogVariant {
                                    weight: vv.get("weight")?.as_f64()? as i64,
                                    style: vv
                                        .get("style")
                                        .and_then(|s| s.as_str())
                                        .unwrap_or("normal")
                                        .to_string(),
                                    msdf_ready: vv
                                        .get("msdf_ready")
                                        .and_then(|b| b.as_bool())
                                        .unwrap_or(false),
                                    atlas_urls: vv
                                        .get("atlas_urls")
                                        .and_then(|a| a.as_array())
                                        .map(|a| {
                                            a.iter()
                                                .filter_map(|u| u.as_str().map(|s| s.to_string()))
                                                .collect()
                                        })
                                        .unwrap_or_default(),
                                    metrics_url: vv
                                        .get("metrics_url")
                                        .and_then(|s| s.as_str())
                                        .map(|s| s.to_string()),
                                    ttf_url: vv
                                        .get("ttf_url")
                                        .and_then(|s| s.as_str())
                                        .map(|s| s.to_string()),
                                    variations: parse_variations(vv.get("variations")),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                by_name.insert(
                    name.to_lowercase(),
                    CatalogFamily {
                        family_name: name,
                        variants,
                        category: fam
                            .get("category")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_ascii_lowercase(),
                    },
                );
            }
        }
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Catalog {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            substitutes,
            by_name,
            normalized_cache: RwLock::new(HashMap::new()),
            ttf_sources: RwLock::new(HashMap::new()),
        }
    }

    /// An identity for memos of this catalog's answers: two catalogs with the same id answer every lookup the same
    /// way (a catalog never changes once read; every empty default catalog shares 0).
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn substitute_family(&self, family: &str) -> Option<&str> {
        self.substitutes
            .get(&family.to_lowercase())
            .map(String::as_str)
    }

    /// The catalog key `family` resolves to by NORMALIZATION, when the exact
    /// lookup missed. Mirrors `resolveFontFamily`'s fallback arm
    /// (`utils/fontStore.ts`) — see `family_normalization` for the conservatism
    /// contract (fonts we deliberately do not host keep missing).
    fn normalized_hit(&self, family: &str) -> Option<NormalizedHit> {
        // ONE key for both halves — reading with the raw name while writing the
        // lowercased one means every mixed-case family misses its own memo, takes an
        // exclusive write lock, and churns the map to the cap forever.
        let cache_key = family.to_lowercase();
        if let Ok(cache) = self.normalized_cache.read() {
            if let Some(hit) = cache.get(&cache_key) {
                return hit.clone();
            }
        }

        let resolved = family_name_candidates(family).into_iter().find_map(|c| {
            let key = c.name.to_lowercase();
            if !self.by_name.contains_key(&key) {
                return None;
            }
            Some(NormalizedHit {
                key,
                weight: c.weight,
                style: c.style,
                kind: c.kind,
            })
        });

        if let Ok(mut cache) = self.normalized_cache.write() {
            // Clear rather than freeze at the cap: refusing to insert would make
            // every later lookup recompute candidates forever (`fontStore.ts` does
            // the same).
            if cache.len() >= RESOLUTION_CACHE_CAP {
                cache.clear();
            }
            cache.insert(cache_key, resolved.clone());
        }
        resolved
    }

    /// `findFontByName` (`fontStore.ts`): exact first, then a conservative
    /// normalization of the name (weight suffix, PostScript name, Google rename,
    /// unresolvable OOXML theme token).
    pub fn find_font_by_name(&self, family: &str) -> Option<&CatalogFamily> {
        if let Some(exact) = self.by_name.get(&family.to_lowercase()) {
            return Some(exact);
        }
        self.by_name.get(&self.normalized_hit(family)?.key)
    }

    /// `canonicalFamilyName` (`utils/fontStore.ts`): the catalog's own name for
    /// `family`, or `family` unchanged when it does not resolve. Resolving to the
    /// canonical name is what makes a second `effective_variant` call a no-op, so a
    /// variant that has already been snapped can be looked up again safely.
    pub fn canonical_family_name(&self, family: &str) -> String {
        self.find_font_by_name(family)
            .map(|f| f.family_name.clone())
            .unwrap_or_else(|| family.to_string())
    }

    /// The weight/style the NAME encodes, before checking what the family carries.
    /// An explicitly requested weight wins over the suffix (the PPTX importer's
    /// precedence: an explicit bold overrides the suffix weight).
    fn implied_variant<'a>(hit: &NormalizedHit, weight: i64, style: &'a str) -> (i64, &'a str)
    where
        'static: 'a,
    {
        let w = match hit.weight {
            Some(w) if weight == DEFAULT_FONT_WEIGHT => w,
            _ => weight,
        };
        let s = match hit.style {
            Some(s) if style == "normal" => s,
            _ => style,
        };
        (w, s)
    }

    /// `resolveSnapVariant` (`utils/fontStore.ts`): what a request MEANT, for a
    /// caller that is about to SNAP.
    ///
    /// The snapping pools cross-style-snap by themselves, so each axis is checked
    /// independently — insisting the exact pair exist would silently discard the
    /// Light in `Oswald Light` the moment the run is italicised (Oswald ships no
    /// italics at all).
    pub fn snap_variant(&self, family: &str, weight: i64, style: &str) -> (i64, String) {
        let fallback = (weight, style.to_string());
        if self.by_name.contains_key(&family.to_lowercase()) {
            return fallback;
        }
        let Some(hit) = self.normalized_hit(family) else {
            return fallback;
        };
        let Some(entry) = self.by_name.get(&hit.key) else {
            return fallback;
        };

        let (w, s) = Self::implied_variant(&hit, weight, style);
        (
            if entry.variants.iter().any(|v| v.weight == w) {
                w
            } else {
                weight
            },
            if entry.variants.iter().any(|v| v.style == s) {
                s.to_string()
            } else {
                style.to_string()
            },
        )
    }

    /// Preserve fractional CSS requests until the actual face is selected.
    pub fn snap_variant_precise(&self, family: &str, weight: f64, style: &str) -> (f64, String) {
        let (snapped, style) = self.snap_variant(family, weight as i64, style);
        (
            if weight.fract() == 0.0 {
                snapped as f64
            } else {
                weight
            },
            style,
        )
    }

    /// `resolveEffectiveVariant` (`utils/fontStore.ts`): the variant an EXACT
    /// (non-snapping) accessor should serve for `family`.
    ///
    /// Unlike [`Self::snap_variant`] this may never name a variant the caller cannot
    /// use, so each candidate must exist AND satisfy `eligible` — the SAME predicate
    /// the caller is about to apply. That is also what makes a second application
    /// harmless: a snap that landed on 400 in some pool did so because the implied
    /// face was absent from that pool, and `eligible` is that pool's membership test,
    /// so the implied face is refused here too.
    pub fn effective_variant(
        &self,
        family: &str,
        weight: i64,
        style: &str,
        eligible: impl Fn(&CatalogVariant) -> bool,
    ) -> (i64, String) {
        let fallback = (weight, style.to_string());
        if self.by_name.contains_key(&family.to_lowercase()) {
            return fallback;
        }
        let Some(hit) = self.normalized_hit(family) else {
            return fallback;
        };
        let Some(entry) = self.by_name.get(&hit.key) else {
            return fallback;
        };

        let (weight2, style2) = Self::implied_variant(&hit, weight, style);
        // Most-specific interpretation first, then back off one axis at a time.
        for (w, s) in [(weight2, style2), (weight2, style), (weight, style2)] {
            if entry
                .variants
                .iter()
                .any(|v| v.weight == w && v.style == s && eligible(v))
            {
                return (w, s.to_string());
            }
        }
        fallback
    }

    /// Whether `family` resolved to a DIFFERENT typeface than the one named — today
    /// only an unresolvable OOXML theme token (`+mn-lt`). Exposed for diagnostics.
    pub fn is_substituted_family(&self, family: &str) -> bool {
        !self.by_name.contains_key(&family.to_lowercase())
            && self
                .normalized_hit(family)
                .is_some_and(|hit| hit.kind == CandidateKind::Substituted)
    }

    /// `isMsdfFamily` (`msdfFonts.ts:27-31`).
    pub fn is_msdf_family(&self, family: &str) -> bool {
        match self.find_font_by_name(family) {
            None => false,
            Some(f) => f
                .variants
                .iter()
                .any(|v| v.msdf_ready && !v.atlas_urls.is_empty()),
        }
    }

    /// `getMsdfVariants` (`msdfFonts.ts:47-61`) — catalog order preserved.
    pub fn get_msdf_variants(&self, family: &str) -> Vec<MsdfVariant> {
        let Some(f) = self.find_font_by_name(family) else {
            return Vec::new();
        };
        f.variants
            .iter()
            .filter(|v| v.msdf_ready && !v.atlas_urls.is_empty())
            .map(|v| MsdfVariant {
                weight: v.weight,
                style: v.style.clone(),
            })
            .collect()
    }

    /// `resolveNearestVariant` (`msdfFonts.ts:108-114`) via
    /// `pickNearestVariant` (`msdfFonts.ts:79-103`): same-style pool
    /// preferred, min |Δweight|, ties → heavier.
    pub fn resolve_nearest_variant(
        &self,
        family: &str,
        requested_weight: i64,
        style: &str,
    ) -> Option<MsdfVariant> {
        self.resolve_nearest_variant_precise(family, requested_weight as f64, style)
    }

    pub fn resolve_nearest_variant_precise(
        &self,
        family: &str,
        requested_weight: f64,
        style: &str,
    ) -> Option<MsdfVariant> {
        let (weight, style) = self.snap_variant_precise(family, requested_weight, style);
        pick_nearest_variant_precise(&self.get_msdf_variants(family), weight, &style)
    }

    /// `hasTtfFamily` (`msdfFonts.ts:139-143`) — the family has at least one
    /// variant with a TTF URL (the vector engine reads outlines straight from
    /// the TTF, so these render even without an MSDF atlas).
    pub fn has_ttf_family(&self, family: &str) -> bool {
        match self.find_font_by_name(family) {
            None => false,
            Some(f) => f.variants.iter().any(|v| v.ttf_url.is_some()),
        }
    }

    /// `getTtfVariants` (`msdfFonts.ts:121-132`) — every variant with a TTF
    /// URL, NOT gated on MSDF readiness (the vector engine reads outlines
    /// straight from the TTF).
    pub fn get_ttf_variants(&self, family: &str) -> Vec<MsdfVariant> {
        let Some(f) = self.find_font_by_name(family) else {
            return Vec::new();
        };
        f.variants
            .iter()
            .filter(|v| v.ttf_url.is_some())
            .map(|v| MsdfVariant {
                weight: v.weight,
                style: v.style.clone(),
            })
            .collect()
    }

    /// Whether `family` is a serif face by its catalog category (`isSerifFamily`).
    pub fn is_serif_family(&self, family: &str) -> bool {
        self.find_font_by_name(family).is_some_and(|f| f.category == "serif")
    }

    /// `resolveNearestTtfVariant` (`msdfFonts.ts:138-144`).
    pub fn resolve_nearest_ttf_variant(
        &self,
        family: &str,
        requested_weight: i64,
        style: &str,
    ) -> Option<MsdfVariant> {
        self.resolve_nearest_ttf_variant_precise(family, requested_weight as f64, style)
    }

    pub fn resolve_nearest_ttf_variant_precise(
        &self,
        family: &str,
        requested_weight: f64,
        style: &str,
    ) -> Option<MsdfVariant> {
        let (weight, style) = self.snap_variant_precise(family, requested_weight, style);
        pick_nearest_variant_precise(&self.get_ttf_variants(family), weight, &style)
    }

    /// `getVariantTtfPath` (`msdfFonts.ts`) — EXACT weight+style match (no
    /// snapping), `null` when the variant has no `ttf_url`. The requested variant
    /// goes through `effective_variant` first, so a weight the family NAME encoded
    /// (`Open Sans Light`) is honored.
    pub fn get_variant_ttf_path(&self, family: &str, weight: i64, style: &str) -> Option<String> {
        self.get_variant_ttf_source(family, weight, style)
            .map(|source| source.0.clone())
    }

    /// Desktop fork: [`Self::get_variant_ttf_path`] plus the variant's
    /// variable-font coordinates (empty for a static face).
    pub fn get_variant_ttf_source(
        &self,
        family: &str,
        weight: i64,
        style: &str,
    ) -> Option<TtfSource> {
        let known = |sources: &TtfSourcesOfFamily| {
            sources
                .iter()
                .find(|(w, s, _)| *w == weight && s == style)
                .map(|(_, _, source)| source.clone())
        };
        if let Ok(memo) = self.ttf_sources.read() {
            if let Some(source) = memo.get(family).and_then(known) {
                return source;
            }
        }
        let source = self.resolve_variant_ttf_source(family, weight, style);
        if let Ok(mut memo) = self.ttf_sources.write() {
            // Bounded as the normalized memo is: names and weights come from document content.
            if memo.len() >= RESOLUTION_CACHE_CAP {
                memo.clear();
            }
            let sources = memo.entry(family.to_string()).or_default();
            if sources.len() >= RESOLUTION_CACHE_CAP {
                sources.clear();
            }
            sources.push((weight, style.to_string(), source.clone()));
        }
        source
    }

    fn resolve_variant_ttf_source(
        &self,
        family: &str,
        weight: i64,
        style: &str,
    ) -> Option<TtfSource> {
        let f = self.find_font_by_name(family)?;
        // The vector/export pool: a variant with outlines.
        let (weight, style) =
            self.effective_variant(family, weight, style, |v| v.ttf_url.is_some());
        let v = f
            .variants
            .iter()
            .find(|v| v.weight == weight && v.style == style)?;
        Some(Arc::new((v.ttf_url.clone()?, v.variations.clone())))
    }

    /// `getFallbacksForKey` (`msdfFallbacks.ts:14-38`).
    pub fn get_fallbacks_for_key(&self, key: &FontKey) -> Vec<FontKey> {
        let use_serif = crate::runs::is_serif_family_name(&key.family);
        let fallback_family = if use_serif { "Noto Serif" } else { "Noto Sans" };
        let style = if key.style == "italic" {
            "italic"
        } else {
            "normal"
        };

        let variants: Vec<MsdfVariant> = self
            .get_msdf_variants(fallback_family)
            .into_iter()
            .filter(|v| v.style == style)
            .collect();
        let mut weights: Vec<i64> = variants.iter().map(|v| v.weight).collect();
        weights.sort();

        let mut ordered: Vec<i64> = Vec::new();
        if let Some(nearest) = self.resolve_nearest_variant(fallback_family, key.weight, style) {
            ordered.push(nearest.weight);
        }
        for w in [700, 400] {
            if weights.contains(&w) && !ordered.contains(&w) {
                ordered.push(w);
            }
        }
        let mut remaining: Vec<i64> = weights
            .iter()
            .copied()
            .filter(|w| !ordered.contains(w))
            .collect();
        // JS Array.sort is stable; sort_by on (|Δ|) with stable Rust sort matches.
        remaining.sort_by_key(|w| (w - key.weight).abs());
        ordered.extend(remaining);

        ordered
            .into_iter()
            .take(3)
            .map(|w| FontKey {
                family: fallback_family.to_string(),
                weight: w,
                style: style.to_string(),
            })
            .collect()
    }
}

/// `pickNearestVariant` (`msdfFonts.ts:79-103`).
// MIRRORED IN PYTHON: `backend/scripts/pdf_font_snap.py::snap_variant`, for the PDF
// fidelity harness — it has to predict which face moda-text will draw in order to decide
// whether a page's SSIM is meaningful. Change the rule here and that copy is
// wrong: it will start suppressing scores for pages that render, or publishing scores for
// pages that do not.
pub fn pick_nearest_variant(
    variants: &[MsdfVariant],
    requested_weight: i64,
    style: &str,
) -> Option<MsdfVariant> {
    pick_nearest_variant_precise(variants, requested_weight as f64, style)
}

fn pick_nearest_variant_precise(
    variants: &[MsdfVariant],
    requested_weight: f64,
    style: &str,
) -> Option<MsdfVariant> {
    if variants.is_empty() {
        return None;
    }
    let same_style: Vec<&MsdfVariant> = variants.iter().filter(|v| v.style == style).collect();
    let pool: Vec<&MsdfVariant> = if !same_style.is_empty() {
        same_style
    } else {
        variants.iter().collect()
    };

    let mut best: Option<&MsdfVariant> = None;
    let mut best_delta = f64::INFINITY;
    for v in pool {
        let d = (v.weight as f64 - requested_weight).abs();
        if d < best_delta {
            best = Some(v);
            best_delta = d;
        } else if d == best_delta {
            if let Some(b) = best {
                if v.weight > b.weight {
                    // tie → heavier (msdfFonts.ts:97-100)
                    best = Some(v);
                }
            }
        }
    }
    best.cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractional_weight_selection_does_not_round_across_a_face_midpoint() {
        let pool = variants(&[(400, "normal"), (500, "normal")]);
        assert_eq!(pick_nearest_variant_precise(&pool, 449.9, "normal").unwrap().weight, 400);
        assert_eq!(pick_nearest_variant_precise(&pool, 450.0, "normal").unwrap().weight, 500);
        assert_eq!(pick_nearest_variant_precise(&pool, 450.1, "normal").unwrap().weight, 500);
    }

    fn variants(ws: &[(i64, &str)]) -> Vec<MsdfVariant> {
        ws.iter()
            .map(|(w, s)| MsdfVariant {
                weight: *w,
                style: s.to_string(),
            })
            .collect()
    }

    #[test]
    fn nearest_prefers_same_style_then_heavier_tie() {
        // msdfFonts.ts:79-103
        let vs = variants(&[(400, "normal"), (700, "normal"), (300, "italic")]);
        let v = pick_nearest_variant(&vs, 550, "normal").unwrap();
        assert_eq!(v.weight, 700); // |Δ|=150 tie → heavier
        let v = pick_nearest_variant(&vs, 500, "italic").unwrap();
        assert_eq!(v.weight, 300); // cross-style pool not used (italic exists)
        let vs2 = variants(&[(400, "normal")]);
        let v = pick_nearest_variant(&vs2, 700, "italic").unwrap();
        assert_eq!((v.weight, v.style.as_str()), (400, "normal")); // cross-style fallback
    }

    /// Mirrors `tests/vitest/utils/fontStoreResolution.test.ts` — the two engines
    /// MUST resolve a family name the same way or the canvas and the exported file
    /// render different fonts.
    /// Predicate for the tests that only care about variant EXISTENCE.
    fn any_variant(_: &CatalogVariant) -> bool {
        true
    }

    fn resolution_catalog() -> Catalog {
        fn fam(name: &str, weights: &[i64], italic: bool) -> serde_json::Value {
            let mut variants: Vec<serde_json::Value> = weights
                .iter()
                .map(|w| {
                    serde_json::json!({
                        "weight": w, "style": "normal", "msdf_ready": false,
                        "atlas_urls": [], "ttf_url": format!("https://cdn/{name}-{w}.ttf"),
                    })
                })
                .collect();
            if italic {
                variants.push(serde_json::json!({
                    "weight": 400, "style": "italic", "msdf_ready": false,
                    "atlas_urls": [], "ttf_url": format!("https://cdn/{name}-400i.ttf"),
                }));
            }
            serde_json::json!({ "family_name": name, "variants": variants })
        }
        Catalog::from_catalog_json(&serde_json::json!({
            "platform_fonts": [
                fam("Open Sans", &[300, 400, 700], true),
                fam("Source Sans 3", &[300, 400], false),
                fam("Inter", &[400], false),
                fam("Archivo", &[400], false),
                fam("Archivo Black", &[400], false),
                fam("Lato", &[400, 700], false),
                // Implied weight 300 exists, but ONLY on the upright axis.
                fam("Brand Sans", &[300, 400, 700], true),
                // No italic axis at all (Oswald-shaped).
                fam("Upright Only", &[300, 400, 700], false),
            ],
            "team_fonts": [],
        }))
    }

    #[test]
    fn exact_match_always_wins_over_normalization() {
        let c = resolution_catalog();
        assert_eq!(
            c.find_font_by_name("Open Sans").unwrap().family_name,
            "Open Sans"
        );
        assert_eq!(
            c.find_font_by_name("open sans").unwrap().family_name,
            "Open Sans"
        );
        // "Archivo Black" is its own family — stripping "Black" would be wrong.
        assert_eq!(
            c.find_font_by_name("Archivo Black").unwrap().family_name,
            "Archivo Black"
        );
        assert_eq!(
            c.effective_variant("Archivo Black", 400, "normal", any_variant),
            (400, "normal".into())
        );
    }

    #[test]
    fn normalized_names_resolve_and_carry_their_weight() {
        let c = resolution_catalog();
        assert_eq!(
            c.find_font_by_name("Open Sans Light").unwrap().family_name,
            "Open Sans"
        );
        assert_eq!(
            c.effective_variant("Open Sans Light", 400, "normal", any_variant),
            (300, "normal".into())
        );
        // Explicit weight wins over the suffix (the PPTX importer's precedence).
        assert_eq!(
            c.effective_variant("Open Sans Light", 700, "normal", any_variant),
            (700, "normal".into())
        );
        assert_eq!(
            c.effective_variant("Open Sans Italic", 400, "normal", any_variant),
            (400, "italic".into())
        );

        assert_eq!(
            c.find_font_by_name("Source Sans Pro").unwrap().family_name,
            "Source Sans 3"
        );
        assert_eq!(
            c.find_font_by_name("OpenSans-Light").unwrap().family_name,
            "Open Sans"
        );
        assert_eq!(
            c.effective_variant("ABCDEF+OpenSans-Light", 400, "normal", any_variant),
            (300, "normal".into())
        );
    }

    #[test]
    fn implied_weight_the_family_lacks_is_dropped() {
        let c = resolution_catalog();
        // Lato has no 300 — substituting it would turn a working lookup into a miss.
        assert_eq!(
            c.effective_variant("Lato Light", 400, "normal", any_variant),
            (400, "normal".into())
        );
        assert_eq!(
            c.get_variant_ttf_path("Lato Light", 400, "normal"),
            c.get_variant_ttf_path("Lato", 400, "normal")
        );
    }

    #[test]
    fn accessors_serve_the_implied_weight() {
        let c = resolution_catalog();
        assert_eq!(
            c.get_variant_ttf_path("Open Sans Light", 400, "normal"),
            c.get_variant_ttf_path("Open Sans", 300, "normal")
        );
        assert_eq!(
            c.resolve_nearest_ttf_variant("Open Sans Light", 400, "normal")
                .unwrap()
                .weight,
            300
        );
        assert!(c.has_ttf_family("Open Sans Light"));
    }

    #[test]
    fn ooxml_theme_tokens_resolve_to_the_default_family() {
        let c = resolution_catalog();
        for token in ["+mn-lt", "+mj-lt"] {
            assert_eq!(
                c.find_font_by_name(token).unwrap().family_name,
                "Inter",
                "{token}"
            );
            assert!(c.is_substituted_family(token), "{token}");
            assert_eq!(
                c.get_variant_ttf_path(token, 400, "normal"),
                c.get_variant_ttf_path("Inter", 400, "normal")
            );
        }
    }

    /// Mirrors "the snap → exact-lookup pipeline cannot re-apply the implied
    /// weight" in `tests/vitest/utils/fontStoreResolution.test.ts`.
    #[test]
    fn implied_weight_is_never_applied_across_the_style_axis() {
        let c = resolution_catalog();
        // "Brand Sans Light" styled italic: 300 exists only upright, so the EXACT
        // accessors must stay at (400, italic) rather than name a missing variant.
        assert_eq!(
            c.effective_variant("Brand Sans Light", 400, "italic", any_variant),
            (400, "italic".into())
        );
        assert!(c
            .get_variant_ttf_path("Brand Sans Light", 400, "italic")
            .is_some());
    }

    /// Mirrors "keeps the implied weight when the family has no italic axis at all".
    #[test]
    fn snap_variant_keeps_the_implied_weight_across_the_style_axis() {
        let c = resolution_catalog();
        // The snapping pools cross-style-snap themselves, so italicising the run must
        // not throw away the Light the name asked for.
        assert_eq!(
            c.snap_variant("Upright Only Light", 400, "italic"),
            (300, "italic".into())
        );
        let v = c
            .resolve_nearest_ttf_variant("Upright Only Light", 400, "italic")
            .unwrap();
        assert_eq!((v.weight, v.style.as_str()), (300, "normal"));
    }

    /// The MSDF pool is a strict subset of the family's variants: the snap lands on
    /// 400 and re-looking THAT up must not be rewritten back to the implied 300.
    #[test]
    fn an_exact_accessor_never_names_a_variant_the_requested_pool_lacks() {
        let c = Catalog::from_catalog_json(&serde_json::json!({
            "platform_fonts": [{
                "family_name": "Partial Atlas",
                "variants": [
                    { "weight": 300, "style": "normal", "msdf_ready": false, "atlas_urls": [],
                      "ttf_url": "https://cdn/pa-300.ttf" },
                    { "weight": 400, "style": "normal", "msdf_ready": true, "atlas_urls": ["a.png"],
                      "metrics_url": "a.json", "ttf_url": "https://cdn/pa-400.ttf" },
                ],
            }],
            "team_fonts": [],
        }));
        let snapped = c
            .resolve_nearest_variant("Partial Atlas Light", 400, "normal")
            .unwrap();
        assert_eq!(snapped.weight, 400);
        assert_eq!(
            c.effective_variant("Partial Atlas Light", snapped.weight, &snapped.style, |v| v
                .msdf_ready
                && !v.atlas_urls.is_empty()),
            (400, "normal".into())
        );
        // The TTF pool DOES have 300, so the outline lookup still honours the name.
        assert_eq!(
            c.get_variant_ttf_path("Partial Atlas Light", 400, "normal"),
            Some("https://cdn/pa-300.ttf".to_string())
        );
    }

    #[test]
    fn the_normalized_memo_is_keyed_lowercased_so_it_actually_hits() {
        let c = resolution_catalog();
        assert!(c.find_font_by_name("Open Sans Light").is_some());
        assert!(c
            .normalized_cache
            .read()
            .unwrap()
            .contains_key("open sans light"));
        // A second lookup (and a differently-cased one) is served from the memo.
        assert!(c.find_font_by_name("open sans light").is_some());
        assert_eq!(c.normalized_cache.read().unwrap().len(), 1);
    }

    #[test]
    fn snapping_returns_the_canonical_family_so_the_relookup_is_exact() {
        let c = resolution_catalog();
        assert_eq!(c.canonical_family_name("Open Sans Light"), "Open Sans");
        assert_eq!(c.canonical_family_name("Avenir"), "Avenir");
        // `effective_variant` on a canonical name is a guaranteed no-op, which is
        // what makes a second lookup of an already-snapped variant safe.
        assert_eq!(
            c.effective_variant(
                &c.canonical_family_name("Open Sans Light"),
                400,
                "normal",
                any_variant
            ),
            (400, "normal".into())
        );
    }

    #[test]
    fn effective_variant_is_a_fixed_point() {
        let c = resolution_catalog();
        for (family, w, s) in [
            ("Open Sans Light", 400, "normal"),
            ("Brand Sans Light", 400, "italic"),
            ("Lato Light", 400, "normal"),
            ("+mn-lt", 400, "normal"),
        ] {
            let once = c.effective_variant(family, w, s, any_variant);
            let twice = c.effective_variant(family, once.0, &once.1, any_variant);
            assert_eq!(once, twice, "{family} {w} {s}");
        }
    }

    #[test]
    fn licensed_and_generic_families_keep_missing() {
        let c = resolution_catalog();
        for name in [
            "Avenir",
            "Avenir Next",
            "Gotham",
            "Gotham Pro",
            "Helvetica Neue",
            "Helvetica Neue Light",
            "sans-serif",
            "serif",
        ] {
            assert!(
                c.find_font_by_name(name).is_none(),
                "{name} must stay unresolved"
            );
            assert!(!c.has_ttf_family(name), "{name}");
            assert!(!c.is_substituted_family(name), "{name}");
        }
    }

    fn oracle_fonts_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap()
            .join("fixtures/oracle/fonts")
    }

    /// Every common weight of a family in `NATIVE_RENDER_BAKED_FONT_FAMILIES`
    /// (backend config) must resolve EXACTLY — no weight/style snap — to a TTF
    /// that is VENDORED on disk (present in `MANIFEST.files` and existing). That
    /// is the precise condition the warm vector lane needs to render the family
    /// from the baked `/opt/moda/native/fonts` dir with ZERO network fetch. A gap
    /// (a weight in the catalog but not vendored) would snap to the wrong weight
    /// or force a per-canvas CDN fetch, so this guards the baked-set claim.
    #[test]
    #[ignore = "needs unported Moda fixture oracle/fonts/catalog.json and MANIFEST.json"]
    fn baked_common_families_resolve_from_vendored_ttfs() {
        let root = oracle_fonts_root();
        let catalog_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("catalog.json")).unwrap())
                .unwrap();
        let catalog = Catalog::from_catalog_json(&catalog_json);
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("MANIFEST.json")).unwrap())
                .unwrap();
        let files = manifest["files"].as_object().unwrap();
        // MUST stay in sync with NATIVE_RENDER_BAKED_FONT_FAMILIES (backend).
        let families = ["Inter", "Noto Sans", "Montserrat", "Lato"];
        // The common weight range design canvases actually use.
        let combos = [
            (400, "normal"),
            (500, "normal"),
            (600, "normal"),
            (700, "normal"),
            (400, "italic"),
        ];
        for fam in families {
            for (w, style) in combos {
                let v = catalog
                    .resolve_nearest_ttf_variant(fam, w, style)
                    .unwrap_or_else(|| panic!("{fam} {w} {style}: no ttf variant in catalog"));
                assert_eq!(
                    (v.weight, v.style.as_str()),
                    (w, style),
                    "{fam} {w} {style}: snapped to {} {} (that weight/style is not baked)",
                    v.weight,
                    v.style
                );
                let url = catalog
                    .get_variant_ttf_path(fam, v.weight, &v.style)
                    .unwrap_or_else(|| panic!("{fam} {w} {style}: variant has no ttf_url"));
                let entry = files.get(&url).unwrap_or_else(|| {
                    panic!("{fam} {w} {style}: {url} not vendored in MANIFEST.files (would fetch)")
                });
                let rel = entry["path"].as_str().unwrap();
                assert!(
                    root.join(rel).is_file(),
                    "{fam} {w} {style}: vendored file {rel} missing"
                );
            }
        }
    }
}
