//! E6.3 text truncation, like Figma's "Truncate text": when a text box's
//! content overflows, the lines that don't fit are hidden and the last shown
//! line ends in an ellipsis. Mirrored by `utils/richtext/truncation.ts`.
//!
//! A line is shown while it is within `max_lines` (Figma's Max lines) and its
//! bottom fits the box height (`max_height`, `None` for no height limit). The
//! first line always shows. When anything was hidden, the last shown line
//! drops graphemes from its end until it plus `…` fits its wrap width, trailing
//! spaces go, and `…` joins its last segment (in that segment's style).
//!
//! Runs on an already wrapped layout, so every lane that lays a node out (the
//! vector draw, SVG extraction) cuts it at the same line.

use unicode_segmentation::UnicodeSegmentation;

use crate::measure::TextMeasurer;
use crate::paragraph_layout::ParagraphLayoutResult;
use crate::positioning::{compute_list_indent_px, compute_marker_metrics, measure_line_width};
use crate::types::{Line, LineSegment};

/// The ellipsis a truncated line ends in.
pub const ELLIPSIS: &str = "\u{2026}";

/// How far a line may pass the box's bottom and still count as fitting: line
/// heights are font size × line height on every lane, so only float noise
/// needs absorbing.
const HEIGHT_EPSILON_PX: f64 = 0.5;

/// A node's truncation limits.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TextTruncation {
    /// Figma's Max lines (`maxLines`); `None` = no line limit.
    pub max_lines: Option<usize>,
    /// The box height lines must fit in; `None` = no height limit.
    pub max_height: Option<f64>,
}

/// The width a paragraph's lines wrapped to (`layoutListParagraph` narrows
/// bulleted paragraphs by their indent and marker box).
fn wrap_width(list_kind: Option<&str>, level: i64, font_size: f64, box_width_px: f64) -> f64 {
    if list_kind != Some("ul") {
        return box_width_px;
    }
    let indent = compute_list_indent_px(level, font_size);
    let m = compute_marker_metrics(font_size, None, None);
    (box_width_px - indent - m.marker_box_width_px).max(0.0)
}

/// Cut `line` so it plus the ellipsis fits `width`, then append the ellipsis.
pub fn ellipsize_line<M: TextMeasurer>(line: &mut Line, width: f64, measurer: &M) {
    let Some(style) = line.segments.last().map(|s| s.style.clone()) else {
        return;
    };
    let ellipsis_w = measurer.measure(ELLIPSIS, &style);
    // The ellipsis ends the text: a justified paragraph leaves this line as it is, and it replaces a hyphen.
    line.soft_wrap = false;
    line.space_px = None;
    if line.segments.last().is_some_and(|s| s.hyphen) {
        line.segments.pop();
    }
    while !line.segments.is_empty() && measure_line_width(line, measurer) + ellipsis_w > width {
        pop_grapheme(line);
    }
    // No space before the ellipsis.
    while let Some(seg) = line.segments.last_mut() {
        let trimmed = seg.text.trim_end().len();
        seg.text.truncate(trimmed);
        seg.style.source = seg.style.source.slice(0..seg.text.chars().count());
        if !seg.text.is_empty() {
            break;
        }
        line.segments.pop();
    }
    match line.segments.last_mut() {
        Some(seg) => {
            seg.style.source = seg.style.source.slice(0..seg.text.chars().count());
            seg.text.push_str(ELLIPSIS);
        },
        None => line.segments.push(LineSegment {
            text: ELLIPSIS.to_string(),
            style: style.without_source(),
            tab: None,
            hyphen: false,
        }),
    }
}

fn pop_grapheme(line: &mut Line) {
    let Some(seg) = line.segments.last_mut() else {
        return;
    };
    match seg.text.grapheme_indices(true).next_back() {
        Some((i, _)) => seg.text.truncate(i),
        None => seg.text.clear(),
    }
    if seg.text.is_empty() {
        line.segments.pop();
    }
}

/// Apply `truncation` to a laid-out node. Returns whether anything was hidden
/// (the layout's totals are recomputed when so).
pub fn truncate_layout<M: TextMeasurer>(
    layout: &mut ParagraphLayoutResult,
    truncation: &TextTruncation,
    box_width_px: f64,
    paragraph_spacing_px: f64,
    measurer: &M,
) -> bool {
    let max_lines = truncation.max_lines.unwrap_or(usize::MAX).max(1);
    let max_height = truncation.max_height.unwrap_or(f64::INFINITY);
    let spacing = paragraph_spacing_px.max(0.0);

    // Find the last line that shows: (paragraph, line) indices.
    let mut shown = 0usize;
    let mut height = 0.0f64;
    let mut cut: Option<(usize, usize)> = None;
    'walk: for (p, para) in layout.paragraphs.iter().enumerate() {
        let gap = if p > 0 { spacing } else { 0.0 };
        for (l, line) in para.lines.iter().enumerate() {
            let bottom = height + if l == 0 { gap } else { 0.0 } + line.line_height_px;
            let fits =
                shown < max_lines && (shown == 0 || bottom <= max_height + HEIGHT_EPSILON_PX);
            if !fits {
                cut = Some((p, l));
                break 'walk;
            }
            height = bottom;
            shown += 1;
        }
    }
    let Some((cut_p, cut_l)) = cut else {
        return false;
    };

    // Drop everything from the cut on; the last shown line is the one before it.
    if cut_l == 0 {
        layout.paragraphs.truncate(cut_p);
    } else {
        layout.paragraphs.truncate(cut_p + 1);
        layout.paragraphs[cut_p].lines.truncate(cut_l);
    }
    if let Some(para) = layout.paragraphs.last_mut() {
        let width = wrap_width(
            para.list_kind.as_deref(),
            para.list_level,
            para.font_size,
            box_width_px,
        );
        if let Some(line) = para.lines.last_mut() {
            // Within the paragraph's indents (`crate::paragraph_format`).
            let width = width - line.indent_start - line.indent_end;
            ellipsize_line(line, width, measurer);
        }
    }

    let mut total = 0.0;
    let mut max_w = 0.0f64;
    for (p, para) in layout.paragraphs.iter().enumerate() {
        if p > 0 {
            total += spacing;
        }
        for line in &para.lines {
            total += line.line_height_px;
            max_w = max_w.max(crate::positioning::line_extent_px(line, measurer));
        }
    }
    layout.total_height_px = total;
    layout.max_line_width_px = max_w;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paragraph_layout::layout_paragraphs_to_lines;
    use crate::parser::parse_rich_text_to_paragraphs;
    use crate::types::InlineStyle;

    /// Every char is 10px wide, the ellipsis too.
    struct Mono;
    impl TextMeasurer for Mono {
        fn measure(&self, text: &str, _style: &InlineStyle) -> f64 {
            text.chars().count() as f64 * 10.0
        }
    }

    fn lines(layout: &ParagraphLayoutResult) -> Vec<String> {
        layout
            .paragraphs
            .iter()
            .flat_map(|p| p.lines.iter())
            .map(|l| l.segments.iter().map(|s| s.text.as_str()).collect())
            .collect()
    }

    fn layout(html: &str, width: f64) -> ParagraphLayoutResult {
        layout_paragraphs_to_lines(&parse_rich_text_to_paragraphs(html), width, 0.0, &Mono)
    }

    const HTML: &str = "<p style=\"font-size: 10px; line-height: 2\">aaa bbb ccc ddd eee</p>";

    #[test]
    fn max_lines_cuts_and_ellipsizes() {
        // 80px wide: "aaa bbb" / "ccc ddd" / "eee"; lines are 20px tall.
        let mut l = layout(HTML, 80.0);
        assert_eq!(lines(&l), ["aaa bbb", "ccc ddd", "eee"]);
        let t = TextTruncation {
            max_lines: Some(2),
            max_height: None,
        };
        assert!(truncate_layout(&mut l, &t, 80.0, 0.0, &Mono));
        // "ccc ddd…" is 80px: fits exactly.
        assert_eq!(lines(&l), ["aaa bbb", "ccc ddd…"]);
        assert_eq!(l.total_height_px, 40.0);
    }

    #[test]
    fn the_ellipsis_replaces_what_no_longer_fits() {
        let mut l = layout(HTML, 70.0);
        let t = TextTruncation {
            max_lines: Some(1),
            max_height: None,
        };
        assert!(truncate_layout(&mut l, &t, 70.0, 0.0, &Mono));
        // "aaa bbb…" is 80 > 70: drop "b", then the space-free "aaa bb…" fits.
        assert_eq!(lines(&l), ["aaa bb…"]);
    }

    #[test]
    fn the_box_height_hides_lines_that_do_not_fit() {
        let mut l = layout(HTML, 80.0);
        let t = TextTruncation {
            max_lines: None,
            max_height: Some(45.0),
        };
        assert!(truncate_layout(&mut l, &t, 80.0, 0.0, &Mono));
        assert_eq!(lines(&l), ["aaa bbb", "ccc ddd…"]);
        // Everything fits: nothing changes.
        let mut all = layout(HTML, 80.0);
        let roomy = TextTruncation {
            max_lines: Some(3),
            max_height: Some(60.0),
        };
        assert!(!truncate_layout(&mut all, &roomy, 80.0, 0.0, &Mono));
        assert_eq!(lines(&all), ["aaa bbb", "ccc ddd", "eee"]);
    }

    #[test]
    fn the_first_line_always_shows_and_later_paragraphs_go() {
        let html = "<p style=\"font-size: 10px; line-height: 2\">one two</p><p>three</p>";
        let mut l = layout(html, 200.0);
        let t = TextTruncation {
            max_lines: None,
            max_height: Some(5.0),
        };
        assert!(truncate_layout(&mut l, &t, 200.0, 0.0, &Mono));
        assert_eq!(lines(&l), ["one two…"]);
        assert_eq!(l.paragraphs.len(), 1);
    }

    #[test]
    fn a_trailing_space_goes_before_the_ellipsis() {
        let mut line = Line {
            segments: vec![LineSegment {
                text: "ab ".to_string(),
                style: crate::types::default_style(),
                tab: None,
                hyphen: false,
            }],
            line_height_px: 20.0,
            soft_wrap: false,
            ..Default::default()
        };
        ellipsize_line(&mut line, 100.0, &Mono);
        assert_eq!(line.segments[0].text, "ab…");
    }
}
