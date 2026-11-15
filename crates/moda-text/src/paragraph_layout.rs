//! Paragraph aggregation — ports of `utils/richtext/paragraphLayout.ts`
//! (`layoutParagraphsToLines`, the measurement-injected shared core) and
//! `utils/richtext/layoutList.ts` (`layoutListParagraph` — UL-only wrap-width
//! reduction).

use crate::measure::TextMeasurer;
use crate::types::{Line, Paragraph};
use crate::wrap::{layout_paragraph_with_policy, WordSegmenter};

pub struct LaidOutParagraph {
    pub lines: Vec<Line>,
    pub align: &'static str,
    /// The paragraph's resolved direction is right-to-left (`crate::paragraph_direction`).
    pub rtl: bool,
    /// Some("ul") | Some("ol") | None
    pub list_kind: Option<String>,
    pub list_level: i64,
    pub font_size: f64,
    /// `paragraphColor` (paragraphLayout.ts:64): runs[0].style.color ??
    /// baseStyle.color — the bullet-color fallback (Gate 2).
    pub paragraph_color: Option<String>,
    /// E7.16: the item's number (ol) and its marker style (`crate::list_markers`).
    pub list_index: Option<i64>,
    pub list_style: Option<crate::list_markers::ListStyle>,
    /// E7.16: the style a counter marker is drawn in — the item's first run's (else its base style).
    pub marker_style: Option<crate::types::InlineStyle>,
}

pub struct ParagraphLayoutResult {
    pub paragraphs: Vec<LaidOutParagraph>,
    pub total_height_px: f64,
    pub max_line_width_px: f64,
}

/// The empty line uses the paragraph's style and the same Auto metrics as populated lines.
pub(crate) fn empty_line_height_px<M: TextMeasurer>(para: &Paragraph, measurer: &M) -> f64 {
    para.runs.first().map(|r| &r.style).or(para.base_style.as_ref())
        .map(|s| s.line_box_px(|s| measurer.auto_line_height_px(s)))
        .unwrap_or(16.0 * 1.2)
}

/// `getParagraphListStyle` (`lists.ts:65-71`).
fn paragraph_list_style(para: &Paragraph) -> Option<(&str, i64)> {
    let meta = para.list_metadata.as_ref()?;
    if meta.kind != "ul" && meta.kind != "ol" {
        return None;
    }
    Some((meta.kind.as_str(), meta.level))
}

/// `layoutListParagraph` (`layoutList.ts:6-22`): a list item wraps in the
/// width its hanging indent leaves (E7.16: numbered items too).
fn layout_list_paragraph<M: TextMeasurer>(
    para: &Paragraph,
    max_width_px: f64,
    measurer: &M,
    font_size_px: f64,
    word_segmenter: Option<WordSegmenter<'_>>,
    policy: crate::types::LineHeightPolicy,
) -> Vec<Line> {
    let Some((kind, level)) = paragraph_list_style(para) else {
        return layout_paragraph_with_policy(para, max_width_px, measurer, word_segmenter, policy);
    };
    if kind != "ul" && kind != "ol" {
        return layout_paragraph_with_policy(para, max_width_px, measurer, word_segmenter, policy);
    }
    // The text wraps in what the hanging indent leaves: the draw walks start it at `left_offset_px`.
    let indent =
        crate::positioning::compute_list_indentation(Some(kind), level, font_size_px, None, None);
    let available = (max_width_px - indent.left_offset_px).max(0.0);
    layout_paragraph_with_policy(para, available, measurer, word_segmenter, policy)
}

/// `layoutParagraphsToLines` (`paragraphLayout.ts:30-83`) without a word
/// segmenter — the MSDF path.
pub fn layout_paragraphs_to_lines<M: TextMeasurer>(
    paragraphs: &[Paragraph],
    box_width_px: f64,
    paragraph_spacing_px: f64,
    measurer: &M,
) -> ParagraphLayoutResult {
    layout_paragraphs_to_lines_with_segmenter(
        paragraphs,
        box_width_px,
        paragraph_spacing_px,
        measurer,
        None,
    )
}

/// `layoutParagraphsToLines` with the vector engine's optional word
/// segmenter threaded through (paragraphLayout.ts:35,52-54).
pub fn layout_paragraphs_to_lines_with_segmenter<M: TextMeasurer>(
    paragraphs: &[Paragraph],
    box_width_px: f64,
    paragraph_spacing_px: f64,
    measurer: &M,
    word_segmenter: Option<WordSegmenter<'_>>,
) -> ParagraphLayoutResult {
    layout_paragraphs_to_lines_with_policy(
        paragraphs, box_width_px, paragraph_spacing_px, measurer, word_segmenter, Default::default(),
    )
}

/// Aggregate wrapped and empty lines using one line-height convention.
pub(crate) fn layout_paragraphs_to_lines_with_policy<M: TextMeasurer>(
    paragraphs: &[Paragraph],
    box_width_px: f64,
    paragraph_spacing_px: f64,
    measurer: &M,
    word_segmenter: Option<WordSegmenter<'_>>,
    policy: crate::types::LineHeightPolicy,
) -> ParagraphLayoutResult {
    layout_paragraphs_with(
        paragraphs,
        paragraph_spacing_px,
        measurer,
        policy == crate::types::LineHeightPolicy::Figma,
        |para, list_indent, base_font_size| {
            if list_indent.is_some() {
                layout_list_paragraph(para, box_width_px, measurer, base_font_size, word_segmenter, policy)
            } else {
                layout_paragraph_with_policy(para, box_width_px, measurer, word_segmenter, policy)
            }
        },
    )
}

/// `layoutParagraphsToLines`' aggregation with the line breaking supplied: `wrap(paragraph, list indent, base
/// font size)` returns a paragraph's lines (`list indent` is the hanging indent a list item's lines wrap inside,
/// `None` for a plain paragraph). Box text wraps at one width (above); area type (`crate::area`) at the widths of
/// its shape, line by line. `whole_px_lines`: box text rounds each line's height (Figma); area type keeps exact
/// leading (Illustrator).
pub(crate) fn layout_paragraphs_with<M: TextMeasurer>(
    paragraphs: &[Paragraph],
    paragraph_spacing_px: f64,
    measurer: &M,
    whole_px_lines: bool,
    mut wrap: impl FnMut(&Paragraph, Option<f64>, f64) -> Vec<Line>,
) -> ParagraphLayoutResult {
    let paragraph_spacing_px = paragraph_spacing_px.max(0.0);
    if paragraphs.is_empty() {
        return ParagraphLayoutResult {
            paragraphs: Vec::new(),
            total_height_px: 0.0,
            max_line_width_px: 0.0,
        };
    }

    let mut laid_out: Vec<LaidOutParagraph> = Vec::new();
    for para in paragraphs {
        // paragraphLayout.ts:45-46 — `??` (nullish): runs[0] else baseStyle else defaults
        let base_font_size = para
            .runs
            .first()
            .map(|r| r.style.line_font_size())
            .or_else(|| para.base_style.as_ref().map(|s| s.line_font_size()))
            .unwrap_or(16.0);
        let base_line_box_px = empty_line_height_px(para, measurer);

        let list = paragraph_list_style(para);
        let list_kind = list.map(|(k, _)| k.to_string());
        let list_level = list.map(|(_, l)| l).unwrap_or(0);

        let list_indent = list.map(|(kind, level)| {
            crate::positioning::compute_list_indentation(Some(kind), level, base_font_size, None, None)
                .left_offset_px
        });
        let mut lines = wrap(para, list_indent, base_font_size);
        if lines.is_empty() {
            // empty paragraph → one synthetic line (paragraphLayout.ts:56-58); box text's lines are whole pixels
            let lh = base_line_box_px;
            lines = vec![Line {
                segments: Vec::new(),
                line_height_px: if whole_px_lines {
                    crate::types::round_line_height_px(lh)
                } else {
                    lh
                },
                soft_wrap: false,
                // An empty paragraph's caret sits at its first-line indent.
                indent_start: para.format.start_indent(true),
                indent_end: para.format.indent_right,
                space_px: None,
            }];
        }
        laid_out.push(LaidOutParagraph {
            lines,
            align: para.align,
            rtl: para.direction.rtl,
            list_level,
            font_size: base_font_size,
            paragraph_color: para
                .runs
                .first()
                .map(|r| r.style.color.clone())
                .or_else(|| para.base_style.as_ref().map(|s| s.color.clone())),
            list_index: para.list_metadata.as_ref().and_then(|m| m.index),
            list_style: para
                .list_metadata
                .as_ref()
                .filter(|_| list_kind.is_some())
                .map(|m| m.style),
            marker_style: para
                .runs
                .iter()
                .find(|r| !r.text.is_empty())
                .or(para.runs.first())
                .map(|r| r.style.clone())
                .or_else(|| para.base_style.clone()),
            list_kind,
        });
    }

    // Aggregate the same way the draw paths advance penY (paragraphLayout.ts:69-80)
    let mut total_height_px = 0.0;
    let mut max_line_width_px = 0.0f64;
    for (p, lp) in laid_out.iter().enumerate() {
        if p > 0 {
            total_height_px += paragraph_spacing_px;
        }
        for line in &lp.lines {
            // Indents included: a box that hugs its text holds them (`line_extent_px`).
            let lw = crate::positioning::line_extent_px(line, measurer);
            if lw > max_line_width_px {
                max_line_width_px = lw;
            }
            total_height_px += line.line_height_px;
        }
    }

    ParagraphLayoutResult {
        paragraphs: laid_out,
        total_height_px,
        max_line_width_px,
    }
}

#[cfg(test)]
mod auto_line_height_tests {
    use super::*;
    use crate::parser::parse_rich_text_to_paragraphs;
    use crate::types::InlineStyle;

    /// Every char 10 px; Auto is a font whose ascender − descender + line gap is 1.15 em (Arial's hhea).
    struct ArialLike;
    impl TextMeasurer for ArialLike {
        fn measure(&self, text: &str, _style: &InlineStyle) -> f64 {
            text.chars().count() as f64 * 10.0
        }
        fn auto_line_height_px(&self, style: &InlineStyle) -> Option<f64> {
            Some(style.line_font_size() * 1.15)
        }
    }

    fn heights(html: &str) -> Vec<f64> {
        let paragraphs = parse_rich_text_to_paragraphs(html);
        layout_paragraphs_to_lines(&paragraphs, 1000.0, 0.0, &ArialLike)
            .paragraphs
            .iter()
            .flat_map(|p| p.lines.iter().map(|l| l.line_height_px))
            .collect()
    }

    #[test]
    fn normal_is_the_fonts_own_line_spacing_rounded_per_line() {
        // Figma: Arial 100 Auto is 115; 16 Auto is 18 (18.4); 13 is 15 (14.95).
        assert_eq!(heights(r#"<p style="font-size: 100px; line-height: normal">Hg</p>"#), vec![115.0]);
        assert_eq!(heights(r#"<p style="font-size: 16px; line-height: normal">Hg</p>"#), vec![18.0]);
        assert_eq!(heights(r#"<p style="font-size: 13px; line-height: normal">Hg</p>"#), vec![15.0]);
        // An empty Auto paragraph takes the same line.
        assert_eq!(heights(r#"<p style="font-size: 16px; line-height: normal"></p>"#), vec![18.0]);
    }

    #[test]
    fn a_multiple_or_px_overrides_auto_and_unset_is_auto() {
        assert_eq!(
            heights(r#"<p style="font-size: 16px; line-height: normal"><span style="line-height: 1.5">Hg</span></p>"#),
            vec![24.0]
        );
        assert_eq!(
            heights(r#"<p style="font-size: 16px; line-height: normal"><span style="line-height: 30px">Hg</span></p>"#),
            vec![30.0]
        );
        // Unset is Auto (Figma's default); an explicit 1.2 is 19.2 → 19.
        assert_eq!(heights(r#"<p style="font-size: 16px">Hg</p>"#), vec![18.0]);
        assert_eq!(heights(r#"<p style="font-size: 16px; line-height: 1.2">Hg</p>"#), vec![19.0]);
        // A span inherits Auto; the biggest line box on the line wins (40 × 1.15 = 46).
        assert_eq!(
            heights(r#"<p style="font-size: 16px; line-height: normal">a <span style="font-size: 40px">B</span></p>"#),
            vec![46.0]
        );
    }
}
