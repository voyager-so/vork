//! Structured VECTOR text-run extraction — the PDF text layer's data source
//! (flatten=False sandwich, DESIGN-backend.md §12.8).
//!
//! Walks the SAME layout the vector render/dump walk draws
//! (`vector::node_layout::dump_vector_layout`, `VectorParity` placement:
//! `layout_paragraphs_to_lines_with_segmenter` → `compute_segment_positions` →
//! `shape_vector` → `place_glyphs`) but returns structs instead of the
//! oracle-contract JSON: per segment the placed glyph ids/origins plus the
//! cluster→byte ranges a PDF ToUnicode CMap needs. Because the positions come
//! from the same layout the raster was drawn from, an invisible PDF overlay
//! built from this aligns with the rendered pixels by construction.
//!
//! Fallback semantics mirror the render walk: a node the vector lane would not
//! draw (engine choice, complex-script routing, run resolution, shape error)
//! returns `None` — the
//! caller simply omits it from the text layer (the raster still shows it),
//! never a decline. A single segment whose shaped output contains `.notdef`
//! (gid 0) is dropped per-segment (the per-run mirror of the browser's
//! `pdfGlyphCoverage` auto-flatten).

use std::ops::Range;

use crate::catalog::Catalog;
use crate::line_metrics::compute_baseline_for_line_with_fallback;
use crate::node_layout::RichTextNodeSpec;
use crate::positioning::{
    advance_pen_y_for_paragraph_break, compute_line_start_x, compute_list_indentation,
    compute_segment_positions_justified, justify_space_px, measure_line_width,
};
use crate::runs::{convert_msdf_runs_to_paragraphs, extract_msdf_runs_by_paragraph};
use crate::vector::cjk::{js_whitespace, segment_text_for_wrap_with};
use crate::vector::measure::{
    normalize_text_for_shaping, place_glyphs, style_to_variant, VectorFontStore, VectorMeasurer,
};
use crate::vector::node_layout::{
    resolve_vector_runs, select_text_engine, RunResolution, VectorMetricsProvider,
};
use crate::vector::ops::VectorPlacement;

/// One placed glyph, node-local px (pen origin at the node's top-left).
#[derive(Debug, Clone)]
pub struct ExtractedGlyph {
    pub glyph_id: u32,
    /// Placed x origin in node-local canvas px (pen + shaping x-offset).
    pub x_px: f64,
    /// Y-UP baseline offset in px (shaping yOffset, plus the run's baseline shift, E8.22).
    pub y_offset_px: f64,
    /// E8.22 character rotation: where the glyph's origin lands once the glyph is turned about its own centre
    /// (node-local px, added to `x_px` and the shifted baseline), so a live-text writer can rotate the glyph about its
    /// origin (SVG `rotate`) and still match the draw. (0, 0) without rotation.
    pub rotation_origin_px: (f64, f64),
    /// Byte range into [`ExtractedRun::text`] covered by this glyph's cluster.
    pub text_range: Range<usize>,
    /// Original decoded source character, retained through wrap and shaping.
    pub source_code_point: Option<usize>,
}

/// One laid-out segment's glyph run.
#[derive(Debug, Clone)]
pub struct ExtractedRun {
    /// The text the glyphs shape (shaping-normalized segment text — the string
    /// [`ExtractedGlyph::text_range`] indexes into).
    pub text: String,
    pub family: String,
    pub weight: i64,
    /// "normal" | "italic".
    pub style: String,
    pub font_size_px: f64,
    /// The run's CSS colour as authored (`InlineStyle::color`) — for exporters that write live text.
    pub color: String,
    /// Baseline y in node-local canvas px (y-down from the node top; includes
    /// the vertical-align offset — the y the raster drew this line at).
    pub baseline_y_px: f64,
    /// Final pen x after the last glyph (node-local px, from `place_glyphs`) —
    /// gives the last glyph a real advance in the PDF layer, so its selection
    /// box isn't zero-width.
    pub end_x_px: f64,
    /// F7.3: the run's `font-feature-settings` as canonical CSS (`"smcp" 1`), empty for the font's defaults. The
    /// glyphs are already shaped with them; exporters writing live text (SVG `<text>`) pass them on.
    pub font_features: String,
    /// E6.3: the run's `font-variation-settings` as canonical CSS, empty for the variant's own instance. The glyphs
    /// are already drawn at that instance; live-text exporters pass it on.
    pub font_variations: String,
    /// The run's letter-spacing, px (added after every glyph; already in the glyphs' x). Live-text exporters write it
    /// as `letter-spacing` so the text keeps its tracking when it is edited or re-laid out.
    pub letter_spacing_px: f64,
    /// The run's gradient (`--nf-text-fill`, `crate::run_fill`) over the node box; `None` = `color`.
    pub fill: Option<crate::ops::GradientSpec>,
    /// E8.22: the run's character attributes. The glyph x positions already carry the scale and kerning, and
    /// `y_offset_px` the baseline shift; the scale and rotation of the outlines are the writer's (live text can say a
    /// rotation per glyph, not a scale: `crate::character::CharacterStyle::transforms_glyphs`).
    pub character: crate::character::CharacterStyle,
    pub glyphs: Vec<ExtractedGlyph>,
}

/// E8.22: a turned glyph's origin offset — `(I − R)·C` for the glyph's em-box centre `C` (px, y-down, from the pen),
/// the translation that turns a rotation about the origin into one about the centre. (0, 0) without rotation.
fn rotation_origin_px(
    font: &crate::vector::otjs::OtjsFont,
    gid: u32,
    font_size_px: f64,
    ch: &crate::character::CharacterStyle,
) -> (f64, f64) {
    if ch.rotation == 0.0 {
        return (0.0, 0.0);
    }
    let Ok(q) = font.glyph_quadratics(gid) else {
        return (0.0, 0.0);
    };
    let x = ch.glyph_xform([q.min_x, q.min_y, q.max_x, q.max_y], font_size_px as f32);
    // The origin (0, 0) under `lin·(p − C) + C + off`.
    let cx = (q.min_x as f64 + q.max_x as f64) * 0.5 * font_size_px;
    let cy = (q.min_y as f64 + q.max_y as f64) * 0.5 * font_size_px;
    let [a, c, b, d] = x.lin.map(f64::from);
    (
        -(a * cx + c * cy) + cx + x.off[0] as f64,
        -(b * cx + d * cy) + cy + x.off[1] as f64,
    )
}

/// Cluster (code-point index into `text`) → byte range of that cluster.
///
/// A cluster covers the chars from its own index up to the next DISTINCT
/// cluster value present in the shaped output (in ascending order — robust to
/// multi-glyph clusters and reordered emission), or the end of the text.
fn cluster_byte_ranges(text: &str, clusters: &[usize]) -> Vec<Range<usize>> {
    let char_offsets: Vec<usize> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(text.len()))
        .collect();
    let mut sorted: Vec<usize> = clusters.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let byte_at = |char_idx: usize| -> usize {
        *char_offsets
            .get(char_idx)
            .unwrap_or(char_offsets.last().unwrap_or(&0))
    };
    clusters
        .iter()
        .map(|&c| {
            let end_char = match sorted.iter().position(|&s| s == c) {
                Some(i) => sorted.get(i + 1).copied(),
                None => None,
            };
            let start = byte_at(c);
            let end = end_char.map(byte_at).unwrap_or(text.len());
            start..end.max(start)
        })
        .collect()
}

/// Extract every vector-lane glyph run of one richtext node, or `None` when
/// the vector walk would not draw the node (the caller omits it from the text
/// layer — the raster still shows the pixels).
pub fn extract_vector_text_runs(
    spec: &RichTextNodeSpec,
    store: &VectorFontStore,
    catalog: &Catalog,
) -> Option<Vec<ExtractedRun>> {
    let html = &spec.html_content;
    if select_text_engine(html, catalog) != "vector" {
        return None;
    }
    // Complex-script / RTL / bidi gate — same predicate, same position as the
    // canonical draw walk (`build_vector_node_ops`): these scripts render via
    // the cosmic-text lane, so a vector-layout text layer would misalign (and
    // mis-order) against the raster. Omit the node.
    if crate::complex_script::text_needs_complex_shaping(html)
        || crate::paragraph_direction::html_needs_rtl_reorder(html)
    {
        return None;
    }
    // Vertical text and ruby place glyphs off the horizontal line these runs describe: exporters draw the engine's
    // outlines (SVG, PDF) and leave the node out of a text layer.
    if crate::vertical::html_has_placed_glyphs(html) {
        return None;
    }
    let runs = extract_msdf_runs_by_paragraph(html, catalog);
    if runs.is_empty() {
        return None;
    }
    let resolved = match resolve_vector_runs(&runs, store, catalog) {
        RunResolution::Resolved(r) => r,
        RunResolution::Fallback { .. } => return None,
    };

    // Box coercions + layout, exactly as dump_vector_layout (VectorParity).
    let box_w = spec.width.unwrap_or(400.0).max(0.0);
    let box_h = spec.height.unwrap_or(200.0).max(0.0);
    let paragraph_spacing_px = spec.paragraph_spacing.max(0.0);

    let placement = VectorPlacement::VectorParity;
    let measurer = VectorMeasurer::with_placement(store, catalog, placement);
    let segmenter = |word: &str, rules| segment_text_for_wrap_with(word, rules);
    let paragraphs = convert_msdf_runs_to_paragraphs(&resolved);
    let metrics = VectorMetricsProvider { store, catalog };
    // Area type: the lines the draw walk places in this frame (`crate::area`).
    let (layout, area) = match &spec.area {
        Some(a) => {
            let (l, al) = crate::area::layout_area(
                &paragraphs,
                a,
                paragraph_spacing_px,
                &measurer,
                &metrics,
                Some(&segmenter),
            );
            (l, Some(al))
        }
        None => (
            crate::paragraph_layout::layout_paragraphs_to_lines_with_segmenter(
                &paragraphs,
                spec.wrap_width(),
                paragraph_spacing_px,
                &measurer,
                Some(&segmenter),
            ),
            None,
        ),
    };
    if measurer.take_poison().is_some() {
        return None;
    }
    // E6.3: cut exactly where the draw walk cuts (`place_vector_layout`).
    let mut layout = layout;
    if let (Some(t), None) = (&spec.truncation, &area) {
        crate::truncation::truncate_layout(&mut layout, t, box_w, paragraph_spacing_px, &measurer);
    }

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
            &metrics,
        )
    };

    let x = 0.0;
    let mut pen_y = v_offset;
    let mut out: Vec<ExtractedRun> = Vec::new();
    for (p, para) in layout.paragraphs.iter().enumerate() {
        pen_y = advance_pen_y_for_paragraph_break(pen_y, p == 0, paragraph_spacing_px);
        let is_list_paragraph = para.list_kind.is_some();
        let indent = compute_list_indentation(
            para.list_kind.as_deref(),
            para.list_level,
            para.font_size,
            None,
            None,
        );
        for (line_index, line) in para.lines.iter().enumerate() {
            let slot = match &area {
                Some(a) => match a.visible(p, line_index) {
                    Some(s) => Some(*s),
                    None => continue,
                },
                None => None,
            };
            let (container_x, box_w) = slot.map(|s| (s.x, s.width)).unwrap_or((x, box_w));
            // The paragraph's indents (`crate::paragraph_format`).
            let (container_x, box_w) = crate::positioning::indented_container(line, container_x, box_w);
            let total_width = measure_line_width(line, &measurer);
            let line_start_x =
                compute_line_start_x(
                    crate::paragraph_direction::line_start_align(para.align, para.rtl, line),
                    total_width,
                    container_x,
                    box_w,
                    is_list_paragraph,
                );
            let justify_px =
                justify_space_px(para.align, line, total_width, box_w - indent.left_offset_px);
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
                &metrics,
            );
            let baseline_y = slot.map(|s| s.top()).unwrap_or(pen_y) + baseline.baseline_from_top_px;

            // E7.16: a numbered item's counter is text too (its own run, as the draw walk shapes it).
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
            // Tab leaders (`crate::tab_stops`): each copy placed on the grid as the draw walk places it, right after its
            // tab; the copies of one tab become one run (`leader` = the tab's index), so live text reads `Soup....6.50`.
            // The list counter is its own run: the first word never joins it.
            let mut draw_segments: Vec<(std::borrow::Cow<'_, crate::types::LineSegment>, f64, Option<usize>, bool)> =
                Vec::new();
            if let Some((seg, x)) = &marker {
                draw_segments.push((std::borrow::Cow::Borrowed(seg), *x, None, true));
            }
            for (i, (seg, pos)) in line.segments.iter().zip(positions.iter()).enumerate() {
                draw_segments.push((std::borrow::Cow::Borrowed(seg), pos.start_x, None, false));
                for (text, x) in crate::tab_stops::leader_pieces(seg.tab.as_ref(), pos.start_x) {
                    let copy = crate::types::LineSegment::new(text.to_string(), seg.style.without_source());
                    draw_segments.push((std::borrow::Cow::Owned(copy), x, Some(i), false));
                }
            }
            let mut last_leader: Option<usize> = None;
            // The style of the run just pushed when it was this line's word right before the next segment, with no
            // space between them. A piece the wrap split off without a space (`t-` `shirt`, each CJK character)
            // joins it in that style, and a hyphenated line's hyphen (`LineSegment::hyphen`) in any: extractors put
            // a space between runs, so live text read `t- shirt`, `日 本 語` and `hyphen -`.
            let mut word_run: Option<crate::types::InlineStyle> = None;
            for (seg, seg_start_x, leader_of, is_marker) in draw_segments {
                let seg: &crate::types::LineSegment = &seg;
                if seg.text.is_empty() || seg.text.chars().all(js_whitespace) {
                    word_run = None;
                    continue; // no glyphs; extractors infer spaces from gaps
                }
                let (family, weight, st) = style_to_variant(&seg.style);
                let size = crate::vector::measure::Instance::of(&seg.style);
                let Some(font) = store.font_for_variant(catalog, &family, weight, st, size) else {
                    continue; // pre-pass resolved every font; defensive
                };
                let shaped = crate::vector::measure::shape_segment(
                    store, catalog, &family, weight, st, size, &font, &seg.text, &seg.style, placement,
                );
                let Ok(shaped) = shaped else {
                    // The render walk's throw would fall the NODE back to MSDF.
                    return None;
                };
                if shaped.is_empty() || shaped.iter().any(|g| g.glyph == 0) {
                    // Coverage guard: a run with .notdef never enters the text
                    // layer (per-run mirror of pdfGlyphCoverage auto-flatten).
                    continue;
                }
                let normalized = normalize_text_for_shaping(&font, &seg.text);
                let clusters: Vec<usize> = shaped.iter().map(|s| s.cluster).collect();
                let ranges = cluster_byte_ranges(&normalized, &clusters);
                let mut glyphs: Vec<ExtractedGlyph> = Vec::with_capacity(shaped.len());
                let end_x_px = place_glyphs(
                    &font,
                    &shaped,
                    seg.style.font_size,
                    seg_start_x,
                    |g, pen_x, y_offset_px, ordinal| {
                        glyphs.push(ExtractedGlyph {
                            source_code_point: shaped.get(ordinal).and_then(|g| seg.style.source.get(g.cluster)),
                            glyph_id: g,
                            x_px: pen_x,
                            y_offset_px: y_offset_px + seg.style.character.baseline_shift,
                            rotation_origin_px: rotation_origin_px(&font, g, seg.style.font_size, &seg.style.character),
                            text_range: ranges.get(ordinal).cloned().unwrap_or(0..0),
                        });
                    },
                    seg.style.letter_spacing,
                );
                if let (Some(tab), Some(prev)) = (leader_of, out.last_mut()) {
                    if last_leader == Some(tab) {
                        // The next copy of the same leader: one run, its glyphs pinned where the draw walk put them.
                        let base = prev.text.len();
                        prev.text.push_str(&normalized);
                        prev.glyphs.extend(glyphs.into_iter().map(|mut g| {
                            g.text_range = g.text_range.start + base..g.text_range.end + base;
                            g
                        }));
                        prev.end_x_px = end_x_px;
                        continue;
                    }
                }
                let joins = leader_of.is_none()
                    && word_run.as_ref().is_some_and(|style| seg.hyphen || *style == seg.style);
                if joins {
                    if let Some(prev) = out.last_mut() {
                        let base = prev.text.len();
                        prev.text.push_str(&normalized);
                        prev.glyphs.extend(glyphs.into_iter().map(|mut g| {
                            g.text_range = g.text_range.start + base..g.text_range.end + base;
                            g
                        }));
                        prev.end_x_px = end_x_px;
                        if seg.hyphen {
                            word_run = None;
                        }
                        continue;
                    }
                }
                word_run = (leader_of.is_none() && !is_marker).then(|| seg.style.clone());
                last_leader = leader_of;
                out.push(ExtractedRun {
                    text: normalized,
                    family: family.clone(),
                    weight,
                    style: st.to_string(),
                    font_size_px: seg.style.font_size,
                    color: seg.style.color.clone(),
                    fill: seg.style.fill.clone(),
                    baseline_y_px: baseline_y,
                    end_x_px,
                    letter_spacing_px: seg.style.letter_spacing,
                    font_features: if seg.style.font_features.is_empty() {
                        String::new()
                    } else {
                        crate::font_features::serialize_font_feature_settings(
                            &seg.style.font_features,
                        )
                    },
                    font_variations: if seg.style.font_variations.is_empty() {
                        String::new()
                    } else {
                        crate::font_variations::serialize_font_variation_settings(
                            &seg.style.font_variations,
                        )
                    },
                    character: seg.style.character,
                    glyphs,
                });
            }
            if measurer.take_poison().is_some() {
                return None;
            }
            pen_y += line.line_height_px;
        }
    }
    Some(out)
}
