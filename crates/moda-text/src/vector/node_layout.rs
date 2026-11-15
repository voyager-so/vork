//! Per-node VECTOR layout walk — the Rust mirror of the oracle's
//! `dumpRichTextNodeVector` (`tools/oracle/dump-linebox.mjs`, `--engine
//! vector` lane), which itself mirrors `WebGPUTextSystem.renderWithVector`
//! (WebGPUTextSystem.ts:633-762) → `VectorTextRenderer.drawRuns/buildLayout`
//! (VectorTextRenderer.ts:352-378, 1073-1133) → `walkLines` (387-494).
//!
//! Every emitted JSON key matches the oracle dump exactly (Gate 1 compares
//! key sets and numbers at 1e-3 px).

use serde_json::{json, Map, Value};

use crate::catalog::Catalog;
use crate::line_metrics::{compute_baseline_for_line_with_fallback, FontMetricsProvider};
use crate::line_metrics::{DEFAULT_ASCENT_FALLBACK_RATIO, DEFAULT_DESCENT_FALLBACK_RATIO};
use crate::node_layout::RichTextNodeSpec;
use crate::paragraph_layout::layout_paragraphs_to_lines_with_policy;
use crate::positioning::{
    advance_pen_y_for_paragraph_break, compute_bullet_center_x, compute_bullet_metrics,
    compute_line_start_x, compute_list_indentation, compute_segment_positions_justified,
    justify_space_px, measure_line_width,
    XHEIGHT_TO_ASCENT_FALLBACK_RATIO,
};
use crate::runs::{
    convert_msdf_runs_to_paragraphs, extract_msdf_runs_by_paragraph, resolve_vector_run_variant,
    MsdfRun, RunKind,
};
use crate::types::InlineStyle;
use crate::vector::cjk::{js_whitespace, segment_text_for_wrap_with};
use crate::vector::measure::{
    place_glyphs, style_font_key, style_to_variant, VectorFontStore, VectorMeasurer,
};
use crate::vector::otjs::ShapeError;
use crate::vector::script_fallback::{
    resolve_coverage_chunks_in, FallbackContext, FallbackVariant,
};

/// E8.22: per cluster, how much more advance (px) `adjusted` gives it than `base` (the same text shaped without the
/// character attributes and without kerning: the font's bare advances): `[[cluster, px], …]`, the nonzero ones, in
/// cluster order.
fn cluster_extra_px(
    font: &crate::vector::otjs::OtjsFont,
    base: &[crate::vector::measure::ShapedGlyph],
    adjusted: &[crate::vector::measure::ShapedGlyph],
    font_size_px: f64,
) -> Vec<(usize, f64)> {
    let upm = if font.units_per_em != 0 { font.units_per_em as f64 } else { 1000.0 };
    let scale = font_size_px / upm;
    let mut sums: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    for g in adjusted {
        *sums.entry(g.cluster).or_default() += g.x_advance * scale;
    }
    for g in base {
        *sums.entry(g.cluster).or_default() -= g.x_advance * scale;
    }
    sums.into_iter().filter(|(_, px)| px.abs() > 1e-6).collect()
}

/// `sanitizeText` (utils/richtext/parser.ts:192-194).
pub(crate) fn sanitize_text(text: &str) -> String {
    crate::parser::sanitize_text(text)
}

/// Mirror of `VectorTextRenderer.makeMetricsProvider`
/// (VectorTextRenderer.ts:1143-1159): TTF baseline metrics per style
/// variant, MSDF-formula scaling, 0.8/0.2-em fallback.
pub struct VectorMetricsProvider<'a> {
    pub store: &'a VectorFontStore,
    pub catalog: &'a Catalog,
}

impl FontMetricsProvider for VectorMetricsProvider<'_> {
    fn metrics(&self, style: &InlineStyle) -> (f64, f64) {
        let (family, weight, st) = style_to_variant(style);
        match self.store.metrics_for_variant(
            self.catalog,
            &family,
            weight,
            st,
            crate::vector::measure::Instance::of(style),
        ) {
            // Synthesized small caps: the line keeps the full size's metrics.
            Some(m) if m.units_per_em != 0.0 => (
                (m.ascent / m.units_per_em) * style.line_font_size(),
                (m.descent.abs() / m.units_per_em) * style.line_font_size(),
            ),
            _ => (
                style.line_font_size() * DEFAULT_ASCENT_FALLBACK_RATIO,
                style.line_font_size() * DEFAULT_DESCENT_FALLBACK_RATIO,
            ),
        }
    }

    fn cap_x_heights(&self, style: &InlineStyle) -> (f64, f64) {
        let (family, weight, st) = style_to_variant(style);
        let m = self.store.metrics_for_variant(
            self.catalog,
            &family,
            weight,
            st,
            crate::vector::measure::Instance::of(style),
        );
        let em = |v: Option<f64>, fallback: f64| match (&m, v) {
            (Some(m), Some(v)) if m.units_per_em != 0.0 && v > 0.0 => {
                v / m.units_per_em * style.line_font_size()
            }
            _ => style.line_font_size() * fallback,
        };
        (
            em(m.as_ref().and_then(|m| m.cap_height), 0.7),
            em(m.as_ref().and_then(|m| m.x_height), 0.5),
        )
    }
}

/// Q-125: `getCanvasFontMetrics` truth for the hybrid arm's `Canvas2dParity`
/// walks. The default-config golden draws fallback-class nodes through the
/// Canvas2D raster pipeline, whose per-line baselines AND intrinsic content
/// height come from Chromium's `TextMetrics.fontBoundingBox{Ascent,Descent}`
/// (`getCanvasFontMetrics`, canvasFontMetrics.ts:22-25) — and Blink returns
/// those ROUNDED to integer px (`SimpleFontData::PlatformInit`; verified
/// against Chromium: Tajawal 24px → 15/9 where the TTF floats are
/// 15.432/8.568, 50px → 32/18, 9.3px → 6/3). The vector engine's own lane
/// (`VectorParity`) keeps the float TTF metrics (`makeMetricsProvider`).
/// Recorded deviation: the rounding is applied to `chooseBaselineMetrics`
/// floats — for fonts where Chromium's fontBoundingBox source differs from
/// the TTF baseline-metrics choice the residual is the source delta, not
/// the rounding.
pub(crate) struct Canvas2dFontMetricsProvider<'a>(pub(crate) VectorMetricsProvider<'a>);

impl FontMetricsProvider for Canvas2dFontMetricsProvider<'_> {
    fn metrics(&self, style: &InlineStyle) -> (f64, f64) {
        let (a, d) = self.0.metrics(style);
        // JS `Math.round` / Blink's metric rounding: floor(x + 0.5) on the
        // positive ascent/descent magnitudes.
        ((a + 0.5).floor(), (d + 0.5).floor())
    }

    fn cap_x_heights(&self, style: &InlineStyle) -> (f64, f64) {
        self.0.cap_x_heights(style)
    }
}

/// Mirror of `VectorTextRenderer.bulletXHeightPx`
/// (VectorTextRenderer.ts:1167-1178).
pub(crate) fn bullet_x_height_px(
    store: &VectorFontStore,
    catalog: &Catalog,
    style: Option<&InlineStyle>,
    font_size_px: f64,
) -> Option<f64> {
    let style = style?;
    let (family, weight, st) = style_to_variant(style);
    let m = store.metrics_for_variant(
        catalog,
        &family,
        weight,
        st,
        crate::vector::measure::Instance::of(style),
    )?;
    if m.units_per_em == 0.0 {
        return None;
    }
    match m.x_height {
        Some(x) if x > 0.0 => Some((x / m.units_per_em) * font_size_px),
        _ => Some((m.ascent / m.units_per_em) * font_size_px * XHEIGHT_TO_ASCENT_FALLBACK_RATIO),
    }
}

/// Desktop fork: the text engine is vector-only (the MSDF atlas lane is gone —
/// system fonts have no atlases). A node with any shapeable run in a catalog
/// family draws on the vector lane; everything else (no catalog family at all,
/// emoji-only text) goes to the degraded tier, which rasterizes color emoji from
/// the bundled fallback set. Studio's `selectTextEngine` returned 'msdf' for
/// emoji-only text, whose emoji textures came from the CDN.
pub(crate) fn select_text_engine(html: &str, catalog: &Catalog) -> &'static str {
    select_text_engine_with(html, catalog, EmojiRuns::Decline)
}

/// [`select_text_engine`] for a lane that draws emoji (`EmojiRuns::Draw`): emoji-only
/// text is vector text too, its clusters drawn as colour-font image cells.
pub(crate) fn select_text_engine_with(
    html: &str,
    catalog: &Catalog,
    emoji: EmojiRuns,
) -> &'static str {
    let probe_runs = extract_msdf_runs_by_paragraph(html, catalog);
    if probe_runs
        .iter()
        .any(|r| r.kind == RunKind::Text || (emoji == EmojiRuns::Draw && r.kind == RunKind::Emoji))
    {
        "vector"
    } else {
        "canvas2d"
    }
}

pub(crate) enum RunResolution {
    Resolved(Vec<MsdfRun>),
    Fallback {
        reason: &'static str,
        family: Option<String>,
        /// Fallback diagnostics.
        ///
        /// Deliberately SEPARATE from `reason`/`family`: those two are the
        /// oracle dump's byte-compared JSON contract (`fixtures/oracle-vector`),
        /// so the dump ignores this field while the engine folds it into the
        /// resource report.
        diag: FallbackDiagnostics,
    },
}

/// What a vector decline needs to be actionable without a canvas join:
/// Production records carried only the coarse reason and an empty
/// `families`, so neither the failing weight/style nor the actual cause survived.
pub(crate) struct FallbackDiagnostics {
    /// Low-cardinality cause class, safe to group and alert on —
    /// `font-unavailable:not-vendored`, `coverage`, `unresolved-variant`, …
    /// Always the coarse `reason`, refined with a sub-class where one exists.
    pub cause: String,
    /// Free-text detail: the variant (weight/style) and, for a font miss, the
    /// credential-free label of the url the manifest lookup missed on.
    pub detail: Option<String>,
    /// The families this decline is about, for the report's structured `families`.
    ///
    /// Separate from the sibling `family` field, which is the ORACLE dump's
    /// byte-compared JSON contract and is `None` for `coverage` (the TS oracle
    /// emits no family there). Reporting it only through that field is what left
    /// coverage declines with `families: []` — the very gap this closes.
    ///
    /// A LIST, not an `Option`, because a decline is a property of the whole NODE:
    /// one declining run sends every run to MSDF, so a mixed-family node has more
    /// than one family affected. Single-family causes push exactly one element.
    pub families: Vec<String>,
}

/// `"<weight> <style>"` — the variant granularity every fallback record was
/// missing (both values were already in scope at the emit site, just dropped).
fn variant_label(weight: impl std::fmt::Display, style: &str) -> String {
    format!("{weight} {style}")
}

/// The families to report for an `emoji-run` decline: EVERY distinct non-empty
/// `font_family` among the node's runs, in first-seen order.
///
/// Emoji runs carry no family of their own (`runs.rs` gives them
/// `String::new()`; only text segments get the resolved `render_fam`), but one
/// emoji declines the WHOLE node — so the families that actually lost the primary
/// engine are all of the surrounding text's, not just the first. A heading node
/// mixing a display face with a body face loses both, and reporting one of them
/// sends triage after the wrong font. Empty for an emoji-only node, which is the
/// one case where there genuinely is no text family to name.
///
/// Diagnostics only: this never reaches the oracle dump's byte-compared
/// `family` field, and never influences shaping, layout or pixels.
fn emoji_decline_families(runs: &[MsdfRun]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for run in runs {
        // Deduped: a node alternating two families across six runs has two
        // affected families, and `families_total` is meant to answer "one team's
        // font or the whole catalog?" — repeats would inflate it into nonsense.
        if !run.font_family.is_empty() && !out.iter().any(|f| f == &run.font_family) {
            out.push(run.font_family.clone());
        }
    }
    out
}

/// Mirror of `VectorTextRenderer.resolveVectorRuns`
/// (VectorTextRenderer.ts:1098-1133) / the dump's
/// `resolveVectorRunsForDump`.
pub(crate) fn resolve_vector_runs(
    runs: &[MsdfRun],
    store: &VectorFontStore,
    catalog: &Catalog,
) -> RunResolution {
    resolve_vector_runs_with(runs, store, catalog, EmojiRuns::Decline)
}

/// What the vector lane does with an emoji run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmojiRuns {
    /// Decline the whole node (`emoji-run`): the TS oracle's contract, kept by the
    /// consumers that only understand glyph outlines (the text-layer extractor).
    Decline,
    /// Keep the node: each emoji cluster lays out as a 1 em placeholder and draws as a
    /// colour-bitmap quad beside the outlined text (`VectorNodeOps::emojis`).
    Draw,
}

pub(crate) fn resolve_vector_runs_with(
    runs: &[MsdfRun],
    store: &VectorFontStore,
    catalog: &Catalog,
    emoji: EmojiRuns,
) -> RunResolution {
    // Synthesized small caps (`crate::small_caps`): a run asking for `smcp` /
    // `c2sc` from a font without them splits into full-size and reduced
    // capital sub-runs before anything below resolves them.
    let expanded = crate::small_caps::expand_runs(runs, |run| {
        let Some(v) = resolve_vector_run_variant(run, catalog) else {
            return Default::default();
        };
        crate::small_caps::synthesis_for(&run.font_features, |tag| {
            store.variant_has_gsub_feature(catalog, &v.family, v.weight, &v.style, tag)
        })
    });
    let runs: &[MsdfRun] = expanded.as_deref().unwrap_or(runs);
    let mut resolved: Vec<MsdfRun> = Vec::new();
    for run in runs {
        if run.kind == RunKind::Emoji && emoji == EmojiRuns::Draw {
            resolved.push(run.clone());
            continue;
        }
        if run.kind != RunKind::Text {
            return RunResolution::Fallback {
                reason: "emoji-run",
                family: None,
                diag: FallbackDiagnostics {
                    cause: "emoji-run".to_string(),
                    // The node's OWN families, not the emoji's: one emoji declines
                    // the whole node, so these are the families whose text just lost
                    // the primary engine. In practice the most frequent decline of
                    // all, and it was reporting `families: []` — the gap this
                    // closes.
                    //
                    // Read from the node's TEXT runs, never from `run` itself: an
                    // emoji run is built with `font_family: String::new()`
                    // (`runs.rs`, where only the text segments receive the resolved
                    // `render_fam`), so keying on the declining run reproduced the
                    // exact `families: []` this is here to close.
                    families: emoji_decline_families(runs),
                    detail: None,
                },
            };
        }
        let Some(variant) = resolve_vector_run_variant(run, catalog) else {
            return RunResolution::Fallback {
                reason: "unresolved-variant",
                // JS `run.fontFamily ?? null` — always a string on text runs.
                family: Some(run.font_family.clone()),
                diag: FallbackDiagnostics {
                    cause: "unresolved-variant".to_string(),
                    families: vec![run.font_family.clone()],
                    detail: Some(variant_label(
                        run.font_weight.unwrap_or(400.0),
                        if run.font_style == "italic" {
                            "italic"
                        } else {
                            "normal"
                        },
                    )),
                },
            };
        };
        let font = match store.font_for_variant_result(
            catalog,
            &variant.family,
            variant.weight,
            &variant.style,
            Some(run.font_size_px),
        ) {
            Ok(font) => font,
            Err(miss) => {
                return RunResolution::Fallback {
                    reason: "font-unavailable",
                    family: Some(variant.family.clone()),
                    diag: FallbackDiagnostics {
                        // The sub-class is the whole point: it splits
                        // "the catalog has no TTF here" from "the vendored
                        // manifest doesn't carry the catalog's url" from "the
                        // bytes don't parse", which used to be one string.
                        cause: format!("font-unavailable:{}", miss.class()),
                        families: vec![variant.family.clone()],
                        detail: Some(format!(
                            "{}; {}",
                            variant_label(variant.weight, &variant.style),
                            miss.detail()
                        )),
                    },
                };
            }
        };

        // Soft hyphens stay: break opportunities, hyphenating or not (`crate::hyphenate`); the wrap drops them.
        let text = crate::parser::sanitize_text_keeping(&run.text, true);
        let primary_variant = FallbackVariant {
            family: variant.family.clone(),
            weight: variant.weight,
            style: variant.style.clone(),
        };
        // CJK falls back to the face of the run's language (`crate::lang`).
        let ctx = FallbackContext::for_run(run.lang.as_deref(), &variant.family, catalog);
        let Some(chunks) = resolve_coverage_chunks_in(store, catalog, &font, &primary_variant, &text, ctx)
        else {
            return RunResolution::Fallback {
                reason: "coverage",
                family: None,
                diag: FallbackDiagnostics {
                    cause: "coverage".to_string(),
                    families: vec![variant.family.clone()],
                    detail: Some(format!(
                        "{} {}",
                        variant.family,
                        variant_label(variant.weight, &variant.style)
                    )),
                },
            };
        };

        let mut break_before = run.paragraph_break_before;
        let wants_slant = matches!(run.font_style.as_str(), "italic" | "oblique");
        let mut source_offset = 0;
        for chunk in chunks {
            let mut r = run.clone();
            let source_end = source_offset + chunk.text.chars().count();
            r.source = run.source.slice(source_offset..source_end);
            source_offset = source_end;
            r.text = chunk.text;
            r.font_family = chunk.variant.family;
            r.font_weight = Some(chunk.variant.weight as f64);
            // An italic or oblique request resolving to an upright face synthesizes a slant.
            // An actual slanted face must not receive a second slant.
            r.font_style = if wants_slant && !matches!(chunk.variant.style.as_str(), "italic" | "oblique") {
                crate::vector::ops::SYNTHETIC_OBLIQUE.to_string()
            } else {
                chunk.variant.style
            };
            r.paragraph_break_before = break_before;
            resolved.push(r);
            break_before = false;
        }
    }
    RunResolution::Resolved(resolved)
}

fn fallback_entry(node_id: &str, engine: &str, reason: &str) -> Value {
    json!({ "nodeId": node_id, "engine": engine, "reason": reason })
}

/// `dumpRichTextNodeVector` (tools/oracle/dump-linebox.mjs vector lane) —
/// browser-vector placement (`VectorPlacement::VectorParity`), the
/// `fixtures/oracle-vector` contract.
pub fn dump_rich_text_node_vector(
    spec: &RichTextNodeSpec,
    store: &VectorFontStore,
    catalog: &Catalog,
) -> Value {
    dump_rich_text_node_vector_with_placement(
        spec,
        store,
        catalog,
        crate::vector::ops::VectorPlacement::VectorParity,
    )
}

/// The vector dump with an explicit vertical-placement semantics
/// (`VectorParity`, or `Canvas2dParity`: the Canvas2D raster's placement, Q-120).
pub fn dump_rich_text_node_vector_with_placement(
    spec: &RichTextNodeSpec,
    store: &VectorFontStore,
    catalog: &Catalog,
    placement: crate::vector::ops::VectorPlacement,
) -> Value {
    dump_with(spec, store, catalog, placement, EmojiRuns::Decline, false)
}

/// The layout the canvas draws: [`dump_rich_text_node_vector`] with emoji kept in the
/// vector lane (`build_vector_node_ops` draws them as image cells), so the editor measures
/// the lines the canvas paints. An emoji segment is flagged `"emoji": true` and carries
/// the clusters themselves as its `text` (each laid out 1 em wide), so line text reads as
/// the document's.
pub fn layout_rich_text_node_vector(
    spec: &RichTextNodeSpec,
    store: &VectorFontStore,
    catalog: &Catalog,
) -> Value {
    layout_with(spec, store, catalog, false)
}

/// [`layout_rich_text_node_vector`] with the metrics a script reads (`moda_engine::text_metrics`): each glyph's
/// source code point in its segment's text (`c`) and advance in px (`a`, letter spacing included), each segment's
/// font ascent and descent in px (`ascent`, `descent`), and each line's descent. Kept out of the plain layout, whose
/// JSON is compared key for key with the TS oracle's.
pub fn layout_rich_text_node_detail(
    spec: &RichTextNodeSpec,
    store: &VectorFontStore,
    catalog: &Catalog,
) -> Value {
    layout_with(spec, store, catalog, true)
}

fn layout_with(
    spec: &RichTextNodeSpec,
    store: &VectorFontStore,
    catalog: &Catalog,
    detail: bool,
) -> Value {
    // A right-to-left paragraph whose base level reorders its text draws through the cosmic-text lane
    // (`build_vector_node_ops` declines it): report that lane's layout, not this one's logical order.
    if crate::paragraph_direction::html_needs_rtl_reorder(&spec.html_content) {
        return fallback_entry(&spec.node_id, "canvas2d", "rtl-paragraph");
    }
    // Likewise complex-script text (Indic conjuncts, joining, marks): `build_vector_node_ops` declines it to the
    // cosmic-text lane, which shapes it. This lane's otjs walk doesn't, so with a font it can walk (Noto Sans
    // Malayalam) each conjunct measured as its separate letters, about a quarter wider than it draws.
    if crate::complex_script::text_needs_complex_shaping(&spec.html_content) {
        return fallback_entry(&spec.node_id, "canvas2d", "complex-script");
    }
    let mut dump = dump_with(
        spec,
        store,
        catalog,
        crate::vector::ops::VectorPlacement::VectorParity,
        EmojiRuns::Draw,
        detail,
    );
    if dump.get("engine").and_then(Value::as_str) != Some("vector") {
        return dump;
    }
    // Placeholders take the emoji runs in layout order (`emoji_<n>`), hidden lines included.
    let runs = extract_msdf_runs_by_paragraph(&spec.html_content, catalog);
    let clusters: Vec<(String, crate::types::SourceIndices)> = crate::ops::emoji_runs_in_layout_order(&runs)
        .into_iter()
        .map(|r| (r.text.clone(), r.source.clone()))
        .collect();
    let mut next = 0usize;
    let paragraphs = dump.get_mut("paragraphs").and_then(Value::as_array_mut);
    for para in paragraphs.into_iter().flatten() {
        let lines = para.get_mut("lines").and_then(Value::as_array_mut);
        for line in lines.into_iter().flatten() {
            let segments = line.get_mut("segments").and_then(Value::as_array_mut);
            for seg in segments.into_iter().flatten() {
                let Some(text) = seg.get("text").and_then(Value::as_str) else {
                    continue;
                };
                if !text.contains(crate::runs::EMOJI_PLACEHOLDER) {
                    continue;
                }
                let mut out = String::new();
                let old_source = seg["sourceIndices"].as_array().cloned().unwrap_or_default();
                let mut source = Vec::new();
                for (i, ch) in text.chars().enumerate() {
                    match (ch == crate::runs::EMOJI_PLACEHOLDER).then(|| clusters.get(next)) {
                        Some(cluster) => {
                            next += 1;
                            if let Some((text, indices)) = cluster {
                                out.push_str(text);
                                source.extend(indices.values());
                            }
                        }
                        None => {
                            out.push(ch);
                            source.push(old_source.get(i).and_then(Value::as_u64).map(|n| n as usize));
                        },
                    }
                }
                seg["text"] = Value::String(out);
                seg["sourceIndices"] = json!(source);
            }
        }
    }
    dump
}

fn dump_with(
    spec: &RichTextNodeSpec,
    store: &VectorFontStore,
    catalog: &Catalog,
    placement: crate::vector::ops::VectorPlacement,
    emoji: EmojiRuns,
    detail: bool,
) -> Value {
    let html = &spec.html_content;
    let engine_choice = select_text_engine_with(html, catalog, emoji);
    if engine_choice != "vector" {
        return fallback_entry(&spec.node_id, engine_choice, "select-text-engine");
    }

    let runs = extract_msdf_runs_by_paragraph(html, catalog);
    if runs.is_empty() {
        return fallback_entry(&spec.node_id, "msdf-fallback", "no-runs");
    }

    let resolved = match resolve_vector_runs_with(&runs, store, catalog, emoji) {
        // Vertical text: the pieces the draw lays out (`crate::vertical::split_runs`); area type stays horizontal.
        RunResolution::Resolved(r) => match crate::vertical::runs_writing_mode(&r) {
            mode if mode.is_vertical() && spec.area.is_none() => {
                crate::vertical::split_runs(&r, mode)
            }
            _ => r,
        },
        // `diag` is deliberately DROPPED here: this dump's JSON is byte-compared
        // against the TS oracle fixtures, which have no such fields.
        RunResolution::Fallback { reason, family, .. } => {
            let mut entry = fallback_entry(&spec.node_id, "msdf-fallback", reason);
            if let Some(f) = family {
                entry
                    .as_object_mut()
                    .unwrap()
                    .insert("family".to_string(), Value::String(f));
            }
            return entry;
        }
    };

    match dump_vector_layout(spec, &resolved, store, catalog, placement, detail) {
        Ok(v) => v,
        Err(e) => {
            // renderWithVector's catch (WebGPUTextSystem.ts:758-761): ANY
            // vector-path throw → MSDF fallback for the node, error recorded.
            let mut entry = fallback_entry(&spec.node_id, "msdf-fallback", "vector-error");
            entry
                .as_object_mut()
                .unwrap()
                .insert("error".to_string(), Value::String(e.message()));
            entry
        }
    }
}

fn dump_vector_layout(
    spec: &RichTextNodeSpec,
    resolved: &[MsdfRun],
    store: &VectorFontStore,
    catalog: &Catalog,
    placement: crate::vector::ops::VectorPlacement,
    detail: bool,
) -> Result<Value, ShapeError> {
    // renderWithVector:657-659 box coercions.
    let box_w = spec.width.unwrap_or(400.0).max(0.0);
    let box_h = spec.height.unwrap_or(200.0).max(0.0);
    let paragraph_spacing_px = spec.paragraph_spacing.max(0.0);
    // Vertical text: lines wrap across the node's height and stack across its width (`crate::vertical`); the dump's
    // coordinates stay those of the horizontal pass (x along a line, y across the lines), as `writingMode` says.
    let writing_mode = crate::vertical::runs_writing_mode(resolved);
    let vertical = writing_mode.is_vertical() && resolved.iter().any(|r| r.vertical.glyph.is_some());
    let (box_w, box_h) = if vertical { (box_h, box_w) } else { (box_w, box_h) };

    // buildLayout (VectorTextRenderer.ts:1077-1086).
    let paragraphs = convert_msdf_runs_to_paragraphs(resolved);
    // Q-121: the dump models the same parity split as the render walk —
    // `Canvas2dParity` (hybrid lane) measures letter-spaced runs with the
    // Canvas2D pipeline's semantics.
    let measurer = VectorMeasurer::with_placement(store, catalog, placement);
    let segmenter = |word: &str, rules| segment_text_for_wrap_with(word, rules);

    // Same metrics split as the render walk (ops.rs): float TTF metrics on
    // the vector lane, Chromium-rounded canvas metrics (Q-125) on the hybrid
    // lane's Canvas2dParity.
    let vector_metrics = VectorMetricsProvider { store, catalog };
    let canvas2d_metrics = Canvas2dFontMetricsProvider(VectorMetricsProvider { store, catalog });
    let metrics_provider: &dyn FontMetricsProvider = match placement {
        crate::vector::ops::VectorPlacement::VectorParity => &vector_metrics,
        crate::vector::ops::VectorPlacement::Canvas2dParity => &canvas2d_metrics,
    };

    // Area type fits each line into its shape (`crate::area`); box text wraps at the box width.
    let (layout, area) = match &spec.area {
        Some(a) => {
            let (l, al) = crate::area::layout_area(
                &paragraphs,
                a,
                paragraph_spacing_px,
                &measurer,
                &metrics_provider,
                Some(&segmenter),
            );
            (l, Some(al))
        }
        None => (
            layout_paragraphs_to_lines_with_policy(
                &paragraphs,
                if vertical { spec.inline_wrap_width() } else { spec.wrap_width() },
                paragraph_spacing_px,
                &measurer,
                Some(&segmenter),
                spec.line_height_policy,
            ),
            None,
        ),
    };
    if let Some(e) = measurer.take_poison() {
        return Err(e);
    }

    // drawRuns:364-367 — vertical offset from the LAID-OUT height; under
    // `Canvas2dParity` (the hybrid lane, Q-120 + Q-125) the Canvas2D
    // pipeline's raster-quad alignment instead (ops.rs::placement_v_offset).
    let vertical_align = match spec.vertical_align.as_deref() {
        Some(s) if !s.is_empty() => s,
        _ => "top",
    };
    let v_offset = if area.is_some() {
        0.0
    } else {
        crate::vector::ops::placement_v_offset(
            placement,
            vertical_align,
            box_h,
            &layout,
            paragraph_spacing_px,
            metrics_provider,
        )
    };

    // walkLines (387-494).
    let x = 0.0;
    let mut pen_y = v_offset;
    let mut paragraphs_out: Vec<Value> = Vec::new();
    for (p, para) in layout.paragraphs.iter().enumerate() {
        pen_y = advance_pen_y_for_paragraph_break(pen_y, p == 0, paragraph_spacing_px);

        let is_list_paragraph = para.list_kind.is_some();
        let bullet_style = para.list_style.filter(|s| s.is_bullet());
        let indent = compute_list_indentation(
            para.list_kind.as_deref(),
            para.list_level,
            para.font_size,
            None,
            None,
        );

        let mut lines_out: Vec<Value> = Vec::new();
        for (line_index, line) in para.lines.iter().enumerate() {
            let total_width = measure_line_width(line, &measurer);
            // Area type: the line's slot in this frame; a line another frame of the thread draws (or the overflow)
            // is listed without placement, so the editor can still match the text line by line.
            let slot = match &area {
                Some(a) => match a.visible(p, line_index) {
                    Some(s) => Some(*s),
                    None => {
                        let frame = a.slots[p][line_index].map(|s| s.frame);
                        lines_out.push(json!({
                            "lineIndex": line_index,
                            "hidden": true,
                            "frame": frame,
                            "lineHeightPx": line.line_height_px,
                            "totalWidth": total_width,
                            "segments": line.segments.iter().map(|s| if s.hyphen { json!({ "text": s.text, "hyphen": true }) } else { json!({ "text": s.text }) }).collect::<Vec<_>>(),
                        }));
                        continue;
                    }
                },
                None => None,
            };
            let (container_x, container_w) = slot.map(|s| (s.x, s.width)).unwrap_or((x, box_w));
            // The paragraph's indents (`crate::paragraph_format`).
            let (container_x, container_w) =
                crate::positioning::indented_container(line, container_x, container_w);
            let line_start_x = compute_line_start_x(
                crate::paragraph_direction::line_start_align(para.align, para.rtl, line),
                total_width,
                container_x,
                container_w,
                is_list_paragraph,
            );
            let justify_px = justify_space_px(
                para.align,
                line,
                total_width,
                container_w - indent.left_offset_px,
            );
            let positions = compute_segment_positions_justified(
                line,
                line_start_x,
                indent.left_offset_px,
                &measurer,
                justify_px,
            );
            let baseline = compute_baseline_for_line_with_fallback(
                line,
                line.line_height_px,
                para.font_size,
                &metrics_provider,
            );
            let line_top = slot.map(|s| s.top()).unwrap_or(pen_y);
            let baseline_y = line_top + baseline.baseline_from_top_px;

            // UL bullet on the first visual line (walkLines:433-453).
            let bullet: Value = if let Some(bullet_style) = bullet_style.filter(|_| line_index == 0)
            {
                let style = line.segments.first().map(|s| &s.style);
                let m = compute_bullet_metrics(
                    para.font_size,
                    bullet_x_height_px(store, catalog, style, para.font_size),
                    Some(baseline.ascent),
                );
                let center_y =
                    baseline_y + m.center_y_offset_from_baseline + m.vertical_adjustment_px;
                let text_start_x = line_start_x + indent.left_offset_px;
                let center_x = compute_bullet_center_x(
                    text_start_x,
                    indent.marker_box_width_px,
                    indent.marker_gap_px,
                    indent.left_offset_px,
                    m.horizontal_adjustment_px,
                );
                let mut b =
                    json!({ "centerX": center_x, "centerY": center_y, "radius": m.radius_px });
                if bullet_style != crate::list_markers::ListStyle::Disc {
                    b["shape"] = json!(bullet_style.css());
                }
                b
            } else {
                Value::Null
            };
            // E7.16: a numbered item's counter, right-aligned in the hanging indent.
            let marker = if line_index == 0 {
                crate::list_markers::counter_marker_segment(
                    para,
                    line_start_x + indent.left_offset_px,
                    indent.marker_gap_px,
                    &measurer,
                )
            } else {
                None
            };

            // Segment + glyph emission.
            let mut segments_out: Vec<Value> = Vec::new();
            for (i, seg) in line.segments.iter().enumerate() {
                let mut out = Map::new();
                out.insert("text".to_string(), json!(seg.text));
                out.insert("sourceIndices".to_string(), json!(seg.style.source.values()));
                if seg.hyphen {
                    // A hyphenated line's hyphen: drawn, but not in the source text (the edit overlay skips it).
                    out.insert("hyphen".to_string(), Value::Bool(true));
                }
                out.insert("x".to_string(), json!(positions[i].start_x));
                out.insert("width".to_string(), json!(positions[i].width));
                out.insert("fontKey".to_string(), json!(style_font_key(&seg.style)));
                out.insert("fontSize".to_string(), json!(seg.style.font_size));
                out.insert("letterSpacing".to_string(), json!(seg.style.letter_spacing));
                out.insert("lineHeight".to_string(), json!(seg.style.line_height));
                if seg.text.contains(crate::runs::EMOJI_PLACEHOLDER) {
                    // An emoji image slot (EmojiRuns::Draw only): no glyphs to shape.
                    out.insert("emoji".to_string(), Value::Bool(true));
                    segments_out.push(Value::Object(out));
                    continue;
                }
                let whitespace_only = seg.text.is_empty() || seg.text.chars().all(js_whitespace);
                if whitespace_only {
                    out.insert("whitespaceOnly".to_string(), Value::Bool(true));
                    segments_out.push(Value::Object(out));
                    continue;
                }
                let (family, weight, st) = style_to_variant(&seg.style);
                let size = Some(seg.style.font_size);
                let Some(font) = store.font_for_variant(catalog, &family, weight, st, size) else {
                    // unreachable: the pre-pass resolved every font
                    segments_out.push(Value::Object(out));
                    continue;
                };
                // Q-121: hybrid lane (`Canvas2dParity`) — letter-spaced
                // segments shape per-char like the Canvas2D raster.
                // Q-122: ls==0 hybrid segments shape with HarfBuzz semantics
                // (rustybuzz; oracle mirror is harfbuzzjs in the dump tool).
                // F7.3 / Q-121 / Q-122 and E8.22: the draw walk's own dispatch (`measure::shape_segment`).
                let shaped = crate::vector::measure::shape_segment(
                    store, catalog, &family, weight, st, size, &font, &seg.text, &seg.style, placement,
                )?;
                let mut glyphs_out: Vec<Value> = Vec::new();
                let final_pen_x = place_glyphs(
                    &font,
                    &shaped,
                    seg.style.font_size,
                    positions[i].start_x,
                    |g, pen_x, _y_offset_px, _| glyphs_out.push(json!({ "g": g, "x": pen_x })),
                    seg.style.letter_spacing, // `|| 0`
                );
                if detail {
                    // Each glyph's code point in `text` and its advance: to the next glyph's pen, the last to the
                    // segment's end (letter spacing included).
                    let xs: Vec<f64> = glyphs_out.iter().map(|g| g["x"].as_f64().unwrap_or(0.0)).collect();
                    for (k, g) in glyphs_out.iter_mut().enumerate() {
                        let next = xs.get(k + 1).copied().unwrap_or(final_pen_x);
                        g["c"] = json!(shaped.get(k).map_or(k, |s| s.cluster));
                        g["a"] = json!(next - xs[k]);
                    }
                }
                out.insert("glyphs".to_string(), Value::Array(glyphs_out));
                out.insert("finalPenX".to_string(), json!(final_pen_x));
                if detail {
                    let (ascent, descent) = metrics_provider.metrics(&seg.style);
                    out.insert("ascent".to_string(), json!(ascent));
                    out.insert("descent".to_string(), json!(descent));
                }
                // E8.22: the segment's character attributes, and for the edit overlay how much wider each cluster is
                // than the font's bare advance (the horizontal scale and the kerning method, per code-point index into
                // `text`). The overlay draws such a segment unkerned (`font-kerning: none`): the browser kerns only
                // inside each of its text pieces, and the per-character letter spacing splits them.
                let ch = seg.style.character;
                if !ch.is_default() {
                    let mut c = Map::new();
                    for (prop, value) in ch.css_declarations() {
                        c.insert(prop.to_string(), json!(value));
                    }
                    out.insert("character".to_string(), Value::Object(c));
                    if ch.changes_advances() {
                        let mut plain = seg.style.clone();
                        plain.character = crate::character::CharacterStyle {
                            kerning: crate::character::Kerning::None,
                            ..crate::character::CharacterStyle::DEFAULT
                        };
                        let base = crate::vector::measure::shape_segment(
                            store, catalog, &family, weight, st, size, &font, &seg.text, &plain, placement,
                        )?;
                        out.insert(
                            "clusterExtraPx".to_string(),
                            json!(cluster_extra_px(&font, &base, &shaped, seg.style.font_size)),
                        );
                    }
                }
                segments_out.push(Value::Object(out));
            }
            // A poisoned measure during the walk is the same first-throw the
            // browser would have hit.
            if let Some(e) = measurer.take_poison() {
                return Err(e);
            }

            let mut line_out = json!({
                "lineIndex": line_index,
                "lineTopY": line_top,
                "lineHeightPx": line.line_height_px,
                "baselineFromTopPx": baseline.baseline_from_top_px,
                "baselineY": baseline_y,
                "ascent": baseline.ascent,
                "lineStartX": line_start_x,
                "totalWidth": total_width,
                "listLeftOffset": indent.left_offset_px,
                "bullet": bullet,
            });
            // Moved in, not through `json!` (which serializes `&segments_out` into a deep copy of every glyph).
            line_out["segments"] = Value::Array(segments_out);
            if let Some((seg, marker_x)) = &marker {
                line_out["marker"] = json!({ "text": seg.text, "x": marker_x });
            }
            // The paragraph's indents (`crate::paragraph_format`): the page checks count them as the line's room.
            if line.indent_start != 0.0 {
                line_out["indentStartPx"] = json!(line.indent_start);
            }
            if line.indent_end != 0.0 {
                line_out["indentEndPx"] = json!(line.indent_end);
            }
            // Justified: how much wider each space is (the edit overlay widens its spaces to match); the every-line
            // composer may also narrow them.
            if justify_px != 0.0 {
                line_out["justifySpacePx"] = json!(justify_px);
            }
            if let Some(s) = &slot {
                line_out["slot"] = json!({ "x": s.x, "width": s.width, "descent": s.descent });
            }
            if detail {
                line_out["descent"] = json!(baseline.descent);
            }
            lines_out.push(line_out);
            pen_y += line.line_height_px;
        }
        let mut para_out = json!({
            "paragraphIndex": p,
            "align": para.align,
            "listKind": para.list_kind.clone().map(Value::String).unwrap_or(Value::Null),
            "listLevel": para.list_level,
            "fontSize": para.font_size,
        });
        para_out["lines"] = Value::Array(lines_out);
        // A right-to-left paragraph (`crate::paragraph_direction`): the editor's overlay can't follow its lines with
        // left-anchored spacers, as with the degraded tier's bidi paragraphs.
        if para.rtl {
            para_out["bidi"] = json!(true);
        }
        paragraphs_out.push(para_out);
    }

    let resolved_runs: Vec<Value> = resolved
        .iter()
        .map(|r| {
            let style = if r.font_style == "italic" {
                "italic"
            } else {
                "normal"
            };
            json!({
                "text": sanitize_text(&r.text),
                "fontKey": format!("{}|{}|{}", r.font_family, r.font_weight.unwrap_or(400.0), style),
            })
        })
        .collect();

    let mut out = json!({
        "nodeId": spec.node_id,
        "engine": "vector",
        "box": {
            // Text boxes have no padding; the oracle linebox schema keeps the key.
            "padding": 0.0,
            "boxW": box_w,
            "boxH": box_h,
            "vOffset": v_offset,
            "verticalAlign": vertical_align,
            "paragraphSpacingPx": paragraph_spacing_px,
        },
        "totalHeightPx": layout.total_height_px,
        "maxLineWidthPx": layout.max_line_width_px,
    });
    out["resolvedRuns"] = Value::Array(resolved_runs);
    out["paragraphs"] = Value::Array(paragraphs_out);
    if let Some(a) = &area {
        out["area"] = area_summary(a);
        out["totalHeightPx"] = json!(area_content_bottom(a));
    }
    if vertical {
        // `boxW` / `boxH` above are the inline (the node's height) and block (its width) sizes.
        out["writingMode"] = json!(writing_mode.css());
    }
    Ok(out)
}

/// The dump's `area` record: which frame of the thread this is, whether text is left over past the last frame,
/// and whether any line lands here.
pub(crate) fn area_summary(a: &crate::area::AreaLayout) -> Value {
    json!({
        "index": a.index,
        "overflow": a.overflow,
        "empty": !a.has_visible_lines(),
    })
}

/// The bottom of the last line drawn in this frame (the height auto-height area text grows to).
pub(crate) fn area_content_bottom(a: &crate::area::AreaLayout) -> f64 {
    a.slots
        .iter()
        .flatten()
        .flatten()
        .filter(|s| s.frame == a.index)
        .map(|s| s.top() + s.line_height)
        .fold(0.0, f64::max)
}

#[cfg(test)]
#[cfg_attr(not(vork_private_tests), allow(dead_code, unused_imports))]
mod complex_script_layout_tests {
    use super::*;
    use crate::ops::TextNodeSpec;
    use crate::test_fixtures::store_and_catalog;
    use crate::vector::ops::{build_vector_node_ops, VectorEngineDecision, VectorPlacement};

    fn html(family: &str, text: &str) -> String {
        format!("<p><span style=\"font-family: {family}; font-size: 60px\">{text}</span></p>")
    }

    fn layout(html: &str, store: &VectorFontStore, catalog: &Catalog) -> Value {
        let spec = RichTextNodeSpec {
            node_id: "t".into(),
            html_content: html.into(),
            width: Some(2000.0),
            height: Some(200.0),
            ..Default::default()
        };
        layout_rich_text_node_vector(&spec, store, catalog)
    }

    fn draw_declines(html: &str, store: &VectorFontStore, catalog: &Catalog) -> bool {
        let spec = TextNodeSpec {
            node_id: "t".into(),
            html: html.into(),
            width: Some(2000.0),
            height: Some(200.0),
            opacity: 1.0,
            ..Default::default()
        };
        matches!(
            build_vector_node_ops(&spec, store, catalog, 1.0, VectorPlacement::VectorParity),
            VectorEngineDecision::Fallback { .. }
        )
    }

    /// The editor's layout reports the lane that draws: complex-script text (Malayalam conjuncts here, in a font the
    /// otjs walk can read, so nothing threw) declines like the draw, and its lines come from the cosmic-text lane.
    /// Before, the otjs walk laid "ക്കക്കക്ക" out as six separate letters, 374 px at 60 px for 271 drawn.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn complex_script_text_declines_like_the_draw() {
        let (store, catalog) = store_and_catalog();
        for text in ["ക്കക്കക്ക", "ആദ്യം ബജറ്റ്", "नमस्ते"] {
            let h = html("Noto Sans Malayalam", text);
            assert!(draw_declines(&h, &store, &catalog), "{text}: the draw declines");
            let d = layout(&h, &store, &catalog);
            assert_ne!(d["engine"], "vector", "{text}: {d}");
            assert_eq!(d["reason"], "complex-script", "{text}: {d}");
        }
        // Latin stays on the vector lane.
        let d = layout(&html("Inter", "Hello"), &store, &catalog);
        assert_eq!(d["engine"], "vector", "{d}");
    }
}
