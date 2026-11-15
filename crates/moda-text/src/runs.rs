//! `Paragraph[]` → MSDF runs → layout paragraphs — ports of
//! `utils/msdf/parserRuns.ts`, `utils/msdf/runsToParagraphs.ts`,
//! `utils/msdf/resolveRunVariant.ts` and `utils/msdf/msdfFontMetrics.ts`'s
//! `styleToFontKey`.
//!
//! Deviation (parity-irrelevant, documented): run colors are carried as raw
//! CSS strings instead of normalized `[r,g,b,a]` floats — color participates
//! in zero layout decisions (the measure-cache key at `layout.ts:33-34`
//! excludes it) and never appears in `linebox.json`.

use crate::catalog::Catalog;
use crate::emoji::{split_text_into_segments, split_text_into_segments_with_ranges, text_segment, Segment};
use crate::font_weight::parse_font_weight_number;
use crate::parser::parse_rich_text_to_paragraphs;
use crate::types::{FontKey, InlineRun, InlineStyle, Paragraph};
use std::collections::BTreeMap;

/// `EMOJI_PLACEHOLDER` (`runsToParagraphs.ts:6`).
pub const EMOJI_PLACEHOLDER: char = '\u{200D}';

/// `MsdfTextRun` / `MsdfEmojiRun` (`utils/msdf/types.ts`), merged.
#[derive(Debug, Clone)]
pub struct MsdfRun {
    pub source: crate::types::SourceIndices,
    /// 'text' | 'emoji'
    pub kind: RunKind,
    /// text for text runs, the emoji cluster for emoji runs
    pub text: String,
    pub font_size_px: f64,
    pub color: String,
    /// always set on text runs (InlineStyle.letterSpacing is non-optional)
    pub letter_spacing: f64,
    pub line_height_multiple: f64,
    /// `InlineStyle::line_height_auto`.
    pub line_height_auto: bool,
    pub paragraph_break_before: bool,
    pub paragraph_index: usize,
    pub paragraph_align: &'static str,
    /// E6.3: the paragraph's `text-wrap`, carried like its alignment.
    pub paragraph_wrap: crate::text_wrap::TextWrap,
    /// The paragraph's tab stops (`crate::tab_stops`), carried like its alignment.
    pub paragraph_tab_stops: Option<std::sync::Arc<crate::tab_stops::TabStops>>,
    /// The paragraph's indents, hyphenation, composer and word spacing (`crate::paragraph_format`), carried like its
    /// alignment.
    pub paragraph_format: crate::paragraph_format::ParagraphFormat,
    /// The paragraph's direction (`crate::paragraph_direction`), carried like its alignment.
    pub paragraph_direction: crate::paragraph_direction::ParagraphDirection,
    /// Some only on text runs
    pub font_weight: Option<f64>,
    /// 'normal' | 'italic'
    pub font_style: String,
    pub font_family: String,
    pub text_decoration: Option<String>,
    /// The run's decoration styling (`InlineStyle::decoration`).
    pub decoration: crate::decorations::DecorationStyle,
    /// F7.3: the run's `font-feature-settings` (empty = font defaults).
    pub font_features: Vec<crate::font_features::FontFeature>,
    /// E6.3: the run's `font-variation-settings` (empty = the variant's own instance).
    pub font_variations: Vec<crate::vector::varfont::Variation>,
    /// The run's gradient (`--nf-text-fill`, `crate::run_fill`); `color` stays its first stop.
    pub fill: Option<crate::ops::GradientSpec>,
    /// The run's paint (`--nf-paint`, `InlineStyle::paint`).
    pub paint: Option<u32>,
    /// Synthesized small caps: the full size of a reduced run
    /// (`InlineStyle::small_caps_full_size`).
    pub small_caps_full_size: Option<f64>,
    /// E8.22: the run's character attributes (`InlineStyle::character`).
    pub character: crate::character::CharacterStyle,
    /// The run's language (`InlineStyle::lang`): which regional face a CJK fallback draws it in (`crate::lang`).
    pub lang: Option<String>,
    /// The run's vertical-text styles (`InlineStyle::vertical`).
    pub vertical: crate::vertical::VerticalStyle,
    /// The run's ruby (`InlineStyle::ruby`).
    pub ruby: crate::ruby::RubyStyle,
    /// 'ul' | 'ol' (E7.16: numbered lists propagate too, so they indent and draw their counter)
    pub list_kind: Option<String>,
    pub list_level: Option<i64>,
    /// E7.16: the item's number (ol) and marker style.
    pub list_index: Option<i64>,
    pub list_style: Option<crate::list_markers::ListStyle>,
    /// resolved variant — set by the snapping pass (dump-linebox.mjs:118-132)
    pub font_key: Option<FontKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunKind {
    Text,
    Emoji,
}

/// `decodeHtmlEntities` (parserRuns.ts:10-18) — case-insensitive for the
/// named entities (`/gi`), exact for the numeric ones.
fn decode_html_entities(s: &str) -> String {
    let mut out = replace_ci(s, "&quot;", "\"");
    out = out.replace("&#34;", "\"");
    out = replace_ci(&out, "&apos;", "'");
    out = out.replace("&#39;", "'");
    out = replace_ci(&out, "&nbsp;", " ");
    out = replace_ci(&out, "&amp;", "&");
    out
}

fn replace_ci(haystack: &str, needle: &str, replacement: &str) -> String {
    let lower = haystack.to_lowercase();
    let needle_lower = needle.to_lowercase();
    let mut out = String::with_capacity(haystack.len());
    let mut last = 0usize;
    let mut from = 0usize;
    while let Some(pos) = lower[from..].find(&needle_lower) {
        let abs = from + pos;
        out.push_str(&haystack[last..abs]);
        out.push_str(replacement);
        last = abs + needle.len();
        from = last;
    }
    out.push_str(&haystack[last..]);
    out
}

/// `normalizeFontFamily` (parserRuns.ts:27-41).
pub(crate) fn normalize_font_family(raw: &str, catalog: &Catalog) -> String {
    let decoded = decode_html_entities(raw);
    let decoded = decoded.trim();
    if decoded.is_empty() {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::new();
    for p in decoded.split(',') {
        let p = p.trim();
        if p.is_empty() {
            continue;
        }
        // .replace(/^"|"$/g,'') then ^'|'$ then trim
        let mut s = p;
        if s.starts_with('"') {
            s = &s[1..];
        }
        if s.ends_with('"') {
            s = &s[..s.len() - 1];
        }
        if s.starts_with('\'') {
            s = &s[1..];
        }
        if s.ends_with('\'') {
            s = &s[..s.len() - 1];
        }
        let s = s.trim();
        if !s.is_empty() {
            parts.push(s.to_string());
        }
    }
    if parts.is_empty() {
        return String::new();
    }
    // Desktop fork: the first family the catalog can draw (it has outlines), like a
    // CSS font stack — there are no MSDF-baked families to prefer.
    for p in &parts {
        if catalog.has_ttf_family(p) {
            return p.clone();
        }
    }
    parts[0].clone()
}

/// `isSerifFamilyName` (`msdfFallbacks.ts:4-10`). Treats names containing
/// "serif" or "slab" as serif, but keeps the CSS generic "sans-serif" / "sans
/// serif" (and the richtext default family) classified as sans — mirrors the TS
/// negative lookbehind `(?<!sans[\s-])serif|slab`: a "serif" match preceded by
/// "sans" + a space/hyphen does not count.
pub(crate) fn is_serif_family_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    if lower.contains("slab") {
        return true;
    }
    for (idx, _) in lower.match_indices("serif") {
        let preceded_by_sans = lower[..idx]
            .strip_suffix(|c: char| c == '-' || c.is_whitespace())
            .map_or(false, |rest| rest.ends_with("sans"));
        if !preceded_by_sans {
            return true;
        }
    }
    false
}

/// `isRenderableFamily` (parserRuns.ts): the vector/MSDF engine can draw the
/// family — it has a baked MSDF atlas or a TTF in the catalog.
fn is_renderable_family(fam: &str, catalog: &Catalog) -> bool {
    catalog.is_msdf_family(fam) || catalog.has_ttf_family(fam)
}

/// `resolveRenderableFamily` (parserRuns.ts): pick the family a run should
/// actually render with so text is NEVER dropped — the run's own family if
/// renderable, else the paragraph/node baseStyle family if renderable, else the
/// default fallback (Noto Serif for serif-looking names when the catalog has it,
/// else Noto Sans). The serif decision keys off the run's own `fam`, not the
/// baseStyle family.
fn resolve_renderable_family(fam: &str, base_style_fam: Option<&str>, catalog: &Catalog) -> String {
    if !fam.is_empty() && is_renderable_family(fam, catalog) {
        return fam.to_string();
    }
    if let Some(base) = base_style_fam {
        if !base.is_empty() && is_renderable_family(base, catalog) {
            return base.to_string();
        }
    }
    // Noto Serif only where the catalog has it: the desktop app ships Noto Sans but no Noto
    // Serif, and a family nobody can load declines the vector lane (drawn by the Degraded tier).
    if is_serif_family_name(fam) && is_renderable_family("Noto Serif", catalog) {
        "Noto Serif".to_string()
    } else {
        "Noto Sans".to_string()
    }
}

/// What every run consumer reads an absent `font_weight` as (`DEFAULT_RUN_FONT_WEIGHT`,
/// `parserRuns.ts`).
const DEFAULT_RUN_FONT_WEIGHT: f64 = 400.0;

/// Resolve CSS's registered weight axis before face matching can replace the
/// requested weight with a catalog face's weight. Explicit axis settings win;
/// the font store ignores axes absent from a static face and clamps variable axes.
fn weight_variations(
    settings: &[crate::vector::varfont::Variation],
    weight: Option<f64>,
) -> Vec<crate::vector::varfont::Variation> {
    let mut variations = settings.to_vec();
    if !variations.iter().any(|(tag, _)| tag == "wght") {
        variations.push(("wght".into(), weight.unwrap_or(DEFAULT_RUN_FONT_WEIGHT) as f32));
    }
    variations
}

/// `resolveRunFont` (`parserRuns.ts`): fully resolve a run's font identity — the
/// CANONICAL catalog family plus the weight/style the family NAME encoded
/// (`Open Sans Light` → Open Sans at 300).
///
/// This is the ONE seam where a normalized name becomes a catalog name. Every key
/// derived downstream (`style_to_font_key`, `measurer_font_key`, the atlas key from
/// `resolve_msdf_run_variant`, the vector store's TTF lookup) is an EXACT string key
/// with no snapping of its own, so resolving anywhere later would leave the atlas
/// installed under one name and measured under another — which corrupts advances
/// silently instead of failing.
///
/// A `None` weight means the run declared none, which every consumer reads as 400 —
/// so ask with 400 but hand `None` back unless the NAME actually implied something,
/// keeping the run byte-identical for the overwhelmingly common case.
fn resolve_run_font(
    fam: &str,
    base_style_fam: Option<&str>,
    weight: Option<f64>,
    style: &str,
    catalog: &Catalog,
) -> (String, Option<f64>, String) {
    let render_fam = resolve_renderable_family(fam, base_style_fam, catalog);
    let requested = weight.unwrap_or(DEFAULT_RUN_FONT_WEIGHT);
    let (w, s) = catalog.snap_variant_precise(&render_fam, requested, style);
    let emitted = if w == requested { weight } else { Some(w) };
    (catalog.canonical_family_name(&render_fam), emitted, s)
}

/// `mapParagraphsToMsdfRuns` (parserRuns.ts:50-161).
pub fn map_paragraphs_to_msdf_runs(paras: &[Paragraph], catalog: &Catalog) -> Vec<MsdfRun> {
    let mut out: Vec<MsdfRun> = Vec::new();
    for (p_idx, para) in paras.iter().enumerate() {
        let paragraph_align = para.align;
        let paragraph_wrap = para.wrap;
        let paragraph_tab_stops = para.tab_stops.clone();
        let paragraph_format = para.format;
        let paragraph_direction = para.direction;
        let mut emitted_in_para = false;
        let list_meta = para
            .list_metadata
            .as_ref()
            .filter(|m| m.kind == "ul" || m.kind == "ol");
        let list_kind = list_meta.map(|m| m.kind.clone());
        let list_level = list_meta.map(|m| m.level);
        let list_index = list_meta.and_then(|m| m.index);
        let list_style = list_meta.map(|m| m.style);
        // baseStyle family is the second-choice renderable family for every run
        // in this paragraph (parserRuns.ts:96 passes it to resolveRenderableFamily).
        let base_style_fam = para
            .base_style
            .as_ref()
            .map(|bs| normalize_font_family(&bs.font_family, catalog));

        // Presentation belongs to the canonical paragraph, not its styling boundaries.
        // Project text spans back onto their source runs; an emoji cluster takes its base's style.
        let text: String = para.runs.iter().map(|run| run.text.as_str()).collect();
        let chars: Vec<char> = text.chars().collect();
        let segments = split_text_into_segments_with_ranges(&text);
        let paragraph_source = if segments.iter().any(|(_, segment)| matches!(segment, Segment::Emoji(_))) {
            crate::types::SourceIndices::new(para.runs.iter().flat_map(|r| r.style.source.values()).collect())
        } else { Default::default() };
        let mut source_start = 0;
        let mut segment_index = 0;
        for run in &para.runs {
            let source_end = source_start + run.text.chars().count();
            while segment_index < segments.len() && segments[segment_index].0.end <= source_start {
                segment_index += 1;
            }
            let run_segments: Vec<(Segment, crate::types::SourceIndices)> = segments[segment_index..].iter()
                .take_while(|(range, _)| range.start < source_end)
                .filter_map(|(range, segment)| {
                    let start = range.start.max(source_start);
                    let end = range.end.min(source_end);
                    let source = run.style.source.slice(start - source_start..end - source_start);
                    match segment {
                        Segment::Text(_) => text_segment(&chars[start..end]).map(|text| {
                            let source = if chars[start..end].iter().all(|&c| crate::emoji::is_text_character(c)) {
                                source
                            } else {
                                crate::types::SourceIndices::new(chars[start..end].iter().enumerate()
                                    .filter_map(|(i, &c)| crate::emoji::is_text_character(c).then(|| source.get(i))).collect())
                            };
                            (text, source)
                        }),
                        Segment::Emoji(_) if range.start >= source_start => Some((segment.clone(), paragraph_source.slice(range.clone()))),
                        Segment::Emoji(_) => None,
                    }
                }).collect();
            source_start = source_end;
            let fam = normalize_font_family(&run.style.font_family, catalog);
            let font_size_px = run.style.font_size;
            let letter_spacing = run.style.letter_spacing;
            let line_height_multiple = run.style.line_height;
            let line_height_auto = run.style.line_height_auto;
            let font_weight = parse_font_weight_number(&run.style.font_weight, None);
            let font_style = match run.style.font_style.as_str() {
                "italic" => "italic",
                "oblique" => "oblique",
                _ => "normal",
            };
            let td = run.style.text_decoration.clone().filter(|d| d != "none");

            // parserRuns.ts:73 — captured once per run
            let para_break = !out.is_empty() && !emitted_in_para;
            // Never drop text: resolve to a renderable family (the run's own,
            // else the node baseStyle's, else the default fallback) — the MSDF
            // feeder re-gates and the vector engine draws TTF-only families.
            let (run_family, run_weight, run_style) = resolve_run_font(
                &fam,
                base_style_fam.as_deref(),
                font_weight,
                font_style,
                catalog,
            );
            for (seg, source) in run_segments {
                match seg {
                    Segment::Emoji(cluster) => {
                        out.push(MsdfRun {
                            source: source.clone(),
                            kind: RunKind::Emoji,
                            text: cluster,
                            font_size_px,
                            color: run.style.color.clone(),
                            letter_spacing: 0.0,
                            line_height_multiple,
                            line_height_auto,
                            paragraph_break_before: if !emitted_in_para {
                                para_break
                            } else {
                                false
                            },
                            paragraph_index: p_idx,
                            paragraph_align,
                            paragraph_wrap,
                    paragraph_tab_stops: paragraph_tab_stops.clone(),
                    paragraph_format,
                    paragraph_direction,
                            font_weight: None,
                            font_style: "normal".to_string(),
                            font_family: String::new(),
                            text_decoration: td.clone(),
                            decoration: Default::default(),
                            font_features: Vec::new(),
                            font_variations: Vec::new(),
                            fill: None,
                            paint: run.style.paint,
                            small_caps_full_size: None,
                            character: Default::default(),
                            lang: run.style.lang.clone(),
                            vertical: Default::default(),
                            ruby: Default::default(),
                            list_kind: list_kind.clone(),
                            list_level,
                            list_index,
                            list_style,
                            font_key: None,
                        });
                        emitted_in_para = true;
                    }
                    Segment::Text(value) => {
                        if !value.is_empty() {
                            out.push(MsdfRun {
                            source: source.clone(),
                                kind: RunKind::Text,
                                text: value,
                                font_size_px,
                                color: run.style.color.clone(),
                                letter_spacing,
                                line_height_multiple,
                                line_height_auto,
                                paragraph_break_before: if !emitted_in_para {
                                    para_break
                                } else {
                                    false
                                },
                                paragraph_index: p_idx,
                                paragraph_align,
                                paragraph_wrap,
                    paragraph_tab_stops: paragraph_tab_stops.clone(),
                    paragraph_format,
                    paragraph_direction,
                                font_weight: run_weight,
                                font_style: run_style.clone(),
                                font_family: run_family.clone(),
                                text_decoration: td.clone(),
                                decoration: run.style.decoration.clone(),
                                font_features: run.style.font_features.clone(),
                                font_variations: weight_variations(&run.style.font_variations, run_weight),
                                fill: run.style.fill.clone(),
                                paint: run.style.paint,
                                small_caps_full_size: None,
                                character: run.style.character,
                                lang: run.style.lang.clone(),
                                vertical: run.style.vertical,
                                ruby: run.style.ruby.clone(),
                                list_kind: list_kind.clone(),
                                list_level,
                                list_index,
                                list_style,
                                font_key: None,
                            });
                            emitted_in_para = true;
                        }
                    }
                }
            }
        }

        // Empty paragraph carrying baseStyle (parserRuns.ts:123-158)
        if !emitted_in_para {
            if let Some(bs) = &para.base_style {
                let base_fam = normalize_font_family(&bs.font_family, catalog);
                // Never drop the spacing run: substitute a renderable family.
                let bs_weight = parse_font_weight_number(&bs.font_weight, None);
                let bs_style = match bs.font_style.as_str() {
                    "italic" => "italic",
                    "oblique" => "oblique",
                    _ => "normal",
                };
                let (run_family, run_weight, run_style) =
                    resolve_run_font(&base_fam, None, bs_weight, bs_style, catalog);
                let td = bs.text_decoration.clone().filter(|d| d != "none");
                out.push(MsdfRun {
                    source: Default::default(),
                    kind: RunKind::Text,
                    text: String::new(),
                    font_size_px: bs.font_size,
                    color: bs.color.clone(),
                    letter_spacing: bs.letter_spacing,
                    line_height_multiple: bs.line_height,
                    line_height_auto: bs.line_height_auto,
                    paragraph_break_before: {
                        // out is non-empty ⇒ true (parserRuns.ts:144)
                        !out.is_empty()
                    },
                    paragraph_index: p_idx,
                    paragraph_align,
                    paragraph_wrap,
                    paragraph_tab_stops: paragraph_tab_stops.clone(),
                    paragraph_format,
                    paragraph_direction,
                    font_weight: run_weight,
                    font_style: run_style,
                    font_family: run_family,
                    text_decoration: td,
                    decoration: bs.decoration.clone(),
                    font_features: bs.font_features.clone(),
                    font_variations: weight_variations(&bs.font_variations, run_weight),
                    fill: None,
                    paint: bs.paint,
                    small_caps_full_size: None,
                    character: bs.character,
                    lang: bs.lang.clone(),
                    vertical: bs.vertical,
                    ruby: bs.ruby.clone(),
                    list_kind: list_kind.clone(),
                    list_level,
                    list_index,
                    list_style,
                    font_key: None,
                });
            }
        }
    }
    out
}

/// `extractMsdfRunsByParagraph` (parserRuns.ts:163-166).
pub fn extract_msdf_runs_by_paragraph(html: &str, catalog: &Catalog) -> Vec<MsdfRun> {
    let paras = parse_rich_text_to_paragraphs(html);
    map_paragraphs_to_msdf_runs(&paras, catalog)
}

/// Families with actual text content in `html` whose runs were REJECTED at
/// extraction because `is_msdf_family` said no (`mapParagraphsToMsdfRuns`
/// only emits text runs for MSDF families — parserRuns.ts:95). These never
/// reach the variant-snapping `dropped` list, so fallback decisions could not
/// name them and the vendor-sanity check could not catch a stale/incomplete
/// vendored catalog (the wasm-editor font bug). First-seen order, deduped.
pub fn extraction_rejected_families(html: &str, catalog: &Catalog) -> Vec<String> {
    let paras = parse_rich_text_to_paragraphs(html);
    let mut out: Vec<String> = Vec::new();
    let consider = |raw: &str, out: &mut Vec<String>| {
        let fam = normalize_font_family(raw, catalog);
        if !fam.is_empty() && !catalog.is_msdf_family(&fam) && !out.contains(&fam) {
            out.push(fam);
        }
    };
    for para in &paras {
        let mut emitted_text = false;
        for run in &para.runs {
            let has_text = split_text_into_segments(&run.text)
                .iter()
                .any(|s| matches!(s, Segment::Text(t) if !t.is_empty()));
            if has_text {
                consider(&run.style.font_family, &mut out);
                emitted_text = true;
            }
        }
        // Empty paragraph carrying baseStyle (parserRuns.ts:123-158) — its
        // family also routes the node when rejected.
        if !emitted_text {
            if let Some(bs) = &para.base_style {
                consider(&bs.font_family, &mut out);
            }
        }
    }
    out
}

/// CSS generic families: asking for one names no particular font, so drawing it in the default family is no
/// substitution.
fn is_generic_family(fam: &str) -> bool {
    matches!(
        fam.to_ascii_lowercase().as_str(),
        "serif"
            | "sans-serif"
            | "monospace"
            | "cursive"
            | "fantasy"
            | "system-ui"
            | "ui-serif"
            | "ui-sans-serif"
            | "ui-monospace"
            | "ui-rounded"
            | "math"
            | "emoji"
            | "fangsong"
            | "inherit"
            | "initial"
            | "unset"
    )
}

/// The families a node's text asks for that the catalog can't draw, each with the family the runs draw in instead
/// (`resolve_renderable_family`: the paragraph's base family, else Noto Serif / Noto Sans), as `(requested, drawn)`,
/// first-seen and deduped. Only runs with visible text count; generic families (`sans-serif`, …) never do.
pub fn substituted_families(html: &str, catalog: &Catalog) -> Vec<(String, String)> {
    let paras = parse_rich_text_to_paragraphs(html);
    let mut out: Vec<(String, String)> = Vec::new();
    for para in &paras {
        let base = para
            .base_style
            .as_ref()
            .map(|bs| normalize_font_family(&bs.font_family, catalog));
        for run in &para.runs {
            if run.text.trim().is_empty() {
                continue;
            }
            let fam = normalize_font_family(&run.style.font_family, catalog);
            if let Some(drawn) = catalog.substitute_family(&fam) {
                if !out.iter().any(|(requested, _)| requested == &fam) {
                    out.push((fam, drawn.to_string()));
                }
                continue;
            }
            if fam.is_empty() || is_generic_family(&fam) || is_renderable_family(&fam, catalog) {
                continue;
            }
            let drawn = catalog.canonical_family_name(&resolve_renderable_family(
                &fam,
                base.as_deref(),
                catalog,
            ));
            if !out.iter().any(|(r, _)| r == &fam) {
                out.push((fam, drawn));
            }
        }
    }
    out
}

/// `resolveMsdfRunVariant` (`resolveRunVariant.ts:24-33`).
pub fn resolve_msdf_run_variant(run: &MsdfRun, catalog: &Catalog) -> Option<FontKey> {
    if run.kind != RunKind::Text {
        return None;
    }
    let family = run.font_family.trim();
    if family.is_empty() || !catalog.is_msdf_family(family) {
        return None;
    }
    let requested_weight = run.font_weight.unwrap_or(400.0);
    let style = match run.font_style.as_str() {
        "italic" => "italic",
        "oblique" => "oblique",
        _ => "normal",
    };
    let resolved = catalog.resolve_nearest_variant_precise(family, requested_weight, style)?;
    Some(FontKey {
        // The run's own family, unchanged: it is ALREADY canonical (`resolve_run_font`
        // resolves it once, where the run is built). Canonicalizing HERE instead was a
        // bug — the atlas would be keyed on the canonical name while the measurer keys
        // off the run's `font_family`, and those are exact string keys.
        family: family.to_string(),
        weight: resolved.weight,
        style: resolved.style,
    })
}

/// `resolveVectorRunVariant` (`resolveRunVariant.ts`): the MSDF variant for an
/// atlas-baked family (so both engines draw the same face), otherwise the nearest
/// TTF-backed variant. The vector engine reads outlines straight from the TTF, so a
/// family with no MSDF atlas (every installed font in the desktop app) still resolves.
pub fn resolve_vector_run_variant(run: &MsdfRun, catalog: &Catalog) -> Option<FontKey> {
    if let Some(key) = resolve_msdf_run_variant(run, catalog) {
        return Some(key);
    }
    if run.kind != RunKind::Text {
        return None;
    }
    let family = run.font_family.trim();
    if family.is_empty() {
        return None;
    }
    let requested_weight = run.font_weight.unwrap_or(400.0);
    let style = match run.font_style.as_str() {
        "italic" => "italic",
        "oblique" => "oblique",
        _ => "normal",
    };
    let resolved = catalog.resolve_nearest_ttf_variant_precise(family, requested_weight, style)?;
    Some(FontKey {
        family: family.to_string(),
        weight: resolved.weight,
        style: resolved.style,
    })
}

/// `convertMsdfRunsToParagraphs` (`runsToParagraphs.ts:15-122`).
/// The emoji map (placeholder → run) is unused by Gate-1 numerics; allocation
/// order is preserved implicitly by run order.
pub fn convert_msdf_runs_to_paragraphs(runs: &[MsdfRun]) -> Vec<Paragraph> {
    if runs.is_empty() {
        return Vec::new();
    }
    // Group by paragraphIndex; JS Map keeps insertion order, then keys are
    // sorted numerically (runsToParagraphs.ts:40) — BTreeMap gives sorted.
    let mut groups: BTreeMap<usize, Vec<&MsdfRun>> = BTreeMap::new();
    for run in runs {
        groups.entry(run.paragraph_index).or_default().push(run);
    }

    let mut paragraphs = Vec::new();
    for (_idx, runs_in_para) in groups {
        if runs_in_para.is_empty() {
            continue;
        }
        let first = runs_in_para[0];
        let align = first.paragraph_align;
        let wrap = first.paragraph_wrap;
        let tab_stops = first.paragraph_tab_stops.clone();
        let format = first.paragraph_format;
        let direction = first.paragraph_direction;
        let list_kind = first.list_kind.clone();
        let list_level = first.list_level.unwrap_or(0);
        let list_index = first.list_index;
        let list_style = first.list_style;

        let mut inline_runs: Vec<InlineRun> = Vec::new();
        for run in &runs_in_para {
            match run.kind {
                RunKind::Text => {
                    inline_runs.push(InlineRun {
                        text: run.text.clone(),
                        style: InlineStyle {
                            source: run.source.clone(),
                            font_family: if run.font_family.is_empty() {
                                "sans-serif".to_string()
                            } else {
                                run.font_family.clone()
                            },
                            font_size: run.font_size_px,
                            // String(textRun.fontWeight || 400)
                            font_weight: run
                                .font_weight
                                .filter(|w| *w != 0.0)
                                .unwrap_or(400.0)
                                .to_string(),
                            font_style: if run.font_style.is_empty() {
                                "normal".to_string()
                            } else {
                                run.font_style.clone()
                            },
                            color: run.color.clone(),
                            letter_spacing: run.letter_spacing,
                            line_height: if run.line_height_multiple != 0.0 {
                                run.line_height_multiple
                            } else {
                                1.2
                            },
                            line_height_auto: run.line_height_auto,
                            text_decoration: Some(
                                run.text_decoration
                                    .clone()
                                    .unwrap_or_else(|| "none".to_string()),
                            ),
                            decoration: run.decoration.clone(),
                            font_features: run.font_features.clone(),
                            font_variations: run.font_variations.clone(),
                            // The run text is already transformed (parse).
                            text_transform: Default::default(),
                            text_wrap: Default::default(),
                            tab_stops: None,
                            paragraph: Default::default(),
                            fill: run.fill.clone(),
                            paint: run.paint,
                            small_caps_full_size: run.small_caps_full_size,
                            character: run.character,
                            lang: run.lang.clone(),
                            vertical: run.vertical,
                            ruby: run.ruby.clone(),
                        },
                    });
                }
                RunKind::Emoji => {
                    inline_runs.push(InlineRun {
                        text: EMOJI_PLACEHOLDER.to_string(),
                        style: InlineStyle {
                            source: crate::types::SourceIndices::new(vec![run.source.get(0)]),
                            font_family: "sans-serif".to_string(),
                            font_size: run.font_size_px,
                            font_weight: "400".to_string(),
                            font_style: "normal".to_string(),
                            color: run.color.clone(),
                            letter_spacing: 0.0,
                            line_height: if run.line_height_multiple != 0.0 {
                                run.line_height_multiple
                            } else {
                                1.2
                            },
                            line_height_auto: run.line_height_auto,
                            // emoji style omits textDecoration (runsToParagraphs.ts:78-86)
                            text_decoration: None,
                            decoration: Default::default(),
                            font_features: Vec::new(),
                            font_variations: Vec::new(),
                            text_transform: Default::default(),
                            text_wrap: Default::default(),
                            tab_stops: None,
                            paragraph: Default::default(),
                            fill: None,
                            paint: None,
                            small_caps_full_size: None,
                            character: Default::default(),
                            lang: run.lang.clone(),
                            vertical: run.vertical,
                            ruby: run.ruby.clone(),
                        },
                    });
                }
            }
        }

        let list_metadata = list_kind.map(|kind| crate::types::ListMetadata {
            style: list_style.unwrap_or_else(|| {
                crate::list_markers::ListStyle::level_default(&kind, list_level)
            }),
            kind,
            level: list_level,
            index: list_index,
        });
        paragraphs.push(Paragraph {
            align,
            runs: inline_runs,
            base_style: None,
            list_metadata,
            wrap,
            tab_stops,
            format,
            direction,
        });
    }
    paragraphs
}

/// `styleToFontKey` (`utils/msdf/msdfFontMetrics.ts:18-28`).
pub fn style_to_font_key(style: &InlineStyle) -> FontKey {
    let parsed = crate::jsnum::js_parse_int(&style.font_weight);
    let weight = match parsed {
        Some(w) => w,
        None => {
            if style.font_weight == "bold" {
                700
            } else {
                400
            }
        }
    };
    FontKey {
        family: style.font_family.clone(),
        weight,
        style: match style.font_style.as_str() {
            "italic" => "italic",
            "oblique" => "oblique",
            _ => "normal",
        }.to_string(),
    }
}

/// The measurer's font key (`layoutCore.ts:20-24`):
/// `parseInt(style.fontWeight, 10) || 400` — note `|| 400` (0 → 400), unlike
/// styleToFontKey's NaN-only fallback.
pub fn measurer_font_key(style: &InlineStyle) -> FontKey {
    let weight = crate::jsnum::js_parse_int(&style.font_weight)
        .filter(|w| *w != 0)
        .unwrap_or(400);
    FontKey {
        family: style.font_family.clone(),
        weight,
        style: match style.font_style.as_str() {
            "italic" => "italic",
            "oblique" => "oblique",
            _ => "normal",
        }.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> Catalog {
        let variants: Vec<serde_json::Value> = [300, 400, 700]
            .iter()
            .map(|w| {
                serde_json::json!({
                    "weight": w, "style": "normal", "msdf_ready": true,
                    "atlas_urls": ["a.png"], "metrics_url": "a.json",
                    "ttf_url": format!("https://cdn/os-{w}.ttf"),
                })
            })
            .collect();
        Catalog::from_catalog_json(&serde_json::json!({
            "platform_fonts": [{ "family_name": "Open Sans", "variants": variants }],
            "team_fonts": [],
        }))
    }

    #[test]
    fn a_materialized_stand_in_reports_the_real_family() {
        let catalog = Catalog::from_catalog_json(&serde_json::json!({
            "platform_fonts": [{
                "family_name": "Absent Family", "missing": true, "substitute": "Inter",
                "variants": [{ "weight": 400, "style": "normal", "ttf_url": "inter.ttf" }]
            }]
        }));
        let html = "<p style=\"font-family: Absent Family\">Visible text</p>";
        assert_eq!(
            substituted_families(html, &catalog),
            vec![("Absent Family".to_string(), "Inter".to_string())]
        );
        assert_eq!(
            extract_msdf_runs_by_paragraph(html, &catalog)[0].font_family,
            "Absent Family"
        );
    }

    #[test]
    fn a_family_the_catalog_cannot_draw_is_named_with_what_draws_instead() {
        let c = catalog();
        let html = "<p><span style=\"font-family: 'SF Pro Rounded'\">Hi</span> <span style=\"font-family: Open Sans\">there</span></p>\
                    <p><span style=\"font-family: sans-serif\">generic</span><span style=\"font-family: Nope Serif\">x</span>\
                    <span style=\"font-family: Missing Blank\"> </span></p>";
        assert_eq!(
            substituted_families(html, &c),
            vec![
                ("SF Pro Rounded".to_string(), "Noto Sans".to_string()),
                ("Nope Serif".to_string(), "Noto Sans".to_string()),
            ],
            "generic families and blank runs don't count; a drawable family doesn't"
        );
        // A stack whose later member the catalog draws draws in that member: no substitution.
        assert!(substituted_families("<p style=\"font-family: 'SF Pro Rounded', 'Open Sans'\">Hi</p>", &c).is_empty());
        // `drawn` is the family the run mapper emits for those runs (what the renderer shapes with).
        let with_base = "<p style=\"font-family: Open Sans\"><span style=\"font-family: Gone\">Hi</span></p>";
        let drawn = extract_msdf_runs_by_paragraph(with_base, &c)
            .into_iter()
            .find(|r| r.kind == RunKind::Text && !r.text.trim().is_empty())
            .unwrap()
            .font_family;
        assert_eq!(substituted_families(with_base, &c), vec![("Gone".to_string(), drawn)]);
    }

    /// Mirrors "the run mapper is the single seam that canonicalizes" and "the atlas
    /// key and the measurer key are the same string"
    /// (`tests/vitest/utils/fontStoreResolution.test.ts`).
    #[test]
    fn the_run_mapper_emits_the_canonical_family_and_the_implied_weight() {
        let c = catalog();
        let runs =
            extract_msdf_runs_by_paragraph("<p style=\"font-family: Open Sans Light\">Ag</p>", &c);
        let run = runs.iter().find(|r| r.kind == RunKind::Text).unwrap();
        assert_eq!(run.font_family, "Open Sans");
        assert_eq!(run.font_weight, Some(300.0));

        // The atlas key and the measurer key are exact strings with no snapping of
        // their own, so they must be the SAME string for a normalized family.
        let atlas_key = resolve_msdf_run_variant(run, &c).unwrap();
        assert_eq!(atlas_key.family, run.font_family);
        assert_eq!(atlas_key.weight, 300);
    }

    /// A missing serif family resolves to Noto Serif only when the catalog has it; the
    /// desktop catalog ships Noto Sans alone, and an unloadable Noto Serif sent the run to
    /// the Degraded tier.
    #[test]
    fn a_missing_serif_family_falls_back_to_a_face_the_catalog_has() {
        let face = |family: &str| {
            serde_json::json!({ "family_name": family, "variants": [
                { "weight": 400, "style": "normal", "ttf_url": format!("https://cdn/{family}.ttf") }] })
        };
        let sans_only = Catalog::from_catalog_json(&serde_json::json!({
            "platform_fonts": [face("Noto Sans")], "team_fonts": [] }));
        assert_eq!(resolve_renderable_family("Missing Serif", None, &sans_only), "Noto Sans");
        let both = Catalog::from_catalog_json(&serde_json::json!({
            "platform_fonts": [face("Noto Sans"), face("Noto Serif")], "team_fonts": [] }));
        assert_eq!(resolve_renderable_family("Missing Serif", None, &both), "Noto Serif");
        assert_eq!(resolve_renderable_family("Missing Grotesk", None, &both), "Noto Sans");
    }

    #[test]
    fn a_run_whose_name_implies_nothing_is_left_exactly_as_parsed() {
        let c = catalog();
        let plain =
            extract_msdf_runs_by_paragraph("<p style=\"font-family: Open Sans\">Ag</p>", &c);
        let run = plain.iter().find(|r| r.kind == RunKind::Text).unwrap();
        assert_eq!(
            (run.font_family.as_str(), run.font_weight),
            ("Open Sans", Some(400.0))
        );

        // An explicitly declared weight wins over the suffix (the importer's precedence).
        let bold = extract_msdf_runs_by_paragraph(
            "<p style=\"font-family: Open Sans Light; font-weight: 700\">Ag</p>",
            &c,
        );
        let run = bold.iter().find(|r| r.kind == RunKind::Text).unwrap();
        assert_eq!(
            (run.font_family.as_str(), run.font_weight),
            ("Open Sans", Some(700.0))
        );
    }

    /// A family with TTFs but no MSDF atlas is vector-eligible: the vector resolver
    /// snaps within the TTF pool while the MSDF resolver declines.
    #[test]
    fn a_ttf_only_family_resolves_on_the_vector_lane() {
        let c = Catalog::from_catalog_json(&serde_json::json!({
            "platform_fonts": [{ "family_name": "Georgia", "variants": [
                { "weight": 400, "style": "normal", "msdf_ready": false, "atlas_urls": [],
                  "ttf_url": "https://cdn/georgia-400.ttf" },
                { "weight": 700, "style": "normal", "msdf_ready": false, "atlas_urls": [],
                  "ttf_url": "https://cdn/georgia-700.ttf" },
            ]}],
            "team_fonts": [],
        }));
        let runs = extract_msdf_runs_by_paragraph(
            "<p style=\"font-family: Georgia; font-weight: 800\">Ag</p>",
            &c,
        );
        let run = runs.iter().find(|r| r.kind == RunKind::Text).unwrap();
        assert!(resolve_msdf_run_variant(run, &c).is_none());
        let key = resolve_vector_run_variant(run, &c).unwrap();
        assert_eq!(
            (key.family.as_str(), key.weight, key.style.as_str()),
            ("Georgia", 700, "normal")
        );
    }
}
