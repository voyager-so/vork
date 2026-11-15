//! Alignment, segment positioning, list indentation and bullet metrics —
//! ports of `utils/richtext/positioning.ts`, `utils/richtext/lists.ts` and
//! `utils/richtext/bulletMetrics.ts`.

use crate::measure::TextMeasurer;
use crate::types::Line;

/// `LIST_PADDING_LEFT_MIN_PX` (`lists.ts:16`).
pub const LIST_PADDING_LEFT_MIN_PX: f64 = 24.0;
/// `INDENT_PER_LEVEL_FACTOR` (`lists.ts:17`).
pub const INDENT_PER_LEVEL_FACTOR: f64 = 1.5;
/// `MARKER_BOX_FACTOR` (`lists.ts:18`).
pub const MARKER_BOX_FACTOR: f64 = 0.9;
/// `MARKER_BOX_MAX_FACTOR` (`lists.ts:19`).
pub const MARKER_BOX_MAX_FACTOR: f64 = 1.0;
/// `MARKER_GAP_MIN` (`lists.ts:20`).
pub const MARKER_GAP_MIN: f64 = 4.0;
/// `MARKER_GAP_FACTOR` (`lists.ts:21`).
pub const MARKER_GAP_FACTOR: f64 = 0.25;

/// `BULLET_DIAMETER_TO_XHEIGHT_RATIO` (`bulletMetrics.ts:26`).
pub const BULLET_DIAMETER_TO_XHEIGHT_RATIO: f64 = 0.6;
/// `BULLET_MIN_DIAMETER_PX` (`bulletMetrics.ts:31`).
pub const BULLET_MIN_DIAMETER_PX: f64 = 3.0;
/// `BULLET_VERTICAL_OFFSET_FACTOR` (`bulletMetrics.ts:40`) — the −0.1·xHeight
/// empirical browser nudge.
pub const BULLET_VERTICAL_OFFSET_FACTOR: f64 = 0.1;
/// `BULLET_HORIZONTAL_OFFSET_FACTOR` (`bulletMetrics.ts:49`) — the
/// −0.15·diameter empirical browser nudge.
pub const BULLET_HORIZONTAL_OFFSET_FACTOR: f64 = 0.15;

/// `XHEIGHT_TO_ASCENT_FALLBACK_RATIO` (`metricsConstants.ts:11`).
pub const XHEIGHT_TO_ASCENT_FALLBACK_RATIO: f64 = 0.52;

/// `computeLineStartX` (`positioning.ts:16-48`): a justified line starts at
/// the left edge (its word spaces widen instead, [`justify_space_px`]); list
/// paragraphs always left.
pub fn compute_line_start_x(
    align: &str,
    total_line_width: f64,
    container_x: f64,
    container_width: f64,
    is_list_paragraph: bool,
) -> f64 {
    if is_list_paragraph {
        return container_x;
    }
    match align {
        "left" | "justify" => container_x,
        "center" => container_x + (container_width - total_line_width) / 2.0,
        "right" => container_x + container_width - total_line_width,
        _ => container_x,
    }
}

/// A segment's width (`segmentAdvance`): a resolved tab's advance (`crate::tab_stops`), else the measured text.
pub fn segment_advance<M: TextMeasurer + ?Sized>(seg: &crate::types::LineSegment, measurer: &M) -> f64 {
    match &seg.tab {
        Some(fill) => fill.advance,
        None => measurer.measure(&seg.text, &seg.style),
    }
}

/// `measureLineWidth` (`positioning.ts:58-64`): the line as drawn, with the composer's word spacing
/// (`Line::space_px`).
pub fn measure_line_width<M: TextMeasurer>(line: &Line, measurer: &M) -> f64 {
    let mut total = 0.0;
    for seg in &line.segments {
        total += segment_advance(seg, measurer);
    }
    if let Some(px) = line.space_px {
        total += px * justify_opportunities(line) as f64;
    }
    total
}

/// A line's container after its paragraph's indents (`crate::paragraph_format`): `(x, width)`. Every draw walk
/// places a line in this; without indents it is the container itself.
pub fn indented_container(line: &Line, container_x: f64, container_width: f64) -> (f64, f64) {
    if line.indent_start == 0.0 && line.indent_end == 0.0 {
        return (container_x, container_width);
    }
    (
        container_x + line.indent_start,
        container_width - line.indent_start - line.indent_end,
    )
}

/// How wide a line reaches from its container's left edge, indents included: what a box that hugs its text must
/// hold (`lineExtentPx`).
pub fn line_extent_px<M: TextMeasurer>(line: &Line, measurer: &M) -> f64 {
    let w = measure_line_width(line, measurer);
    if line.indent_start == 0.0 && line.indent_end == 0.0 {
        w
    } else {
        w + line.indent_start + line.indent_end
    }
}

/// `computeListIndentPx` (`lists.ts:28-31`): `Math.max(0, level|0) × 1.5em`.
pub fn compute_list_indent_px(level: i64, font_size_px: f64) -> f64 {
    let safe_level = level.max(0) as f64;
    safe_level * (INDENT_PER_LEVEL_FACTOR * font_size_px)
}

pub struct BulletMetrics {
    pub radius_px: f64,
    pub diameter_px: f64,
    pub center_y_offset_from_baseline: f64,
    pub vertical_adjustment_px: f64,
    pub horizontal_adjustment_px: f64,
}

/// `computeBulletMetrics` (`bulletMetrics.ts:108-151`).
pub fn compute_bullet_metrics(
    font_size_px: f64,
    x_height_px: Option<f64>,
    ascent_px: Option<f64>,
) -> BulletMetrics {
    let effective_x_height_px = match x_height_px {
        Some(x) if x > 0.0 => x,
        _ => {
            let effective_ascent = match ascent_px {
                Some(a) if a > 0.0 => a,
                _ => font_size_px * 0.8,
            };
            effective_ascent * XHEIGHT_TO_ASCENT_FALLBACK_RATIO
        }
    };
    let diameter_px =
        BULLET_MIN_DIAMETER_PX.max(effective_x_height_px * BULLET_DIAMETER_TO_XHEIGHT_RATIO);
    BulletMetrics {
        radius_px: diameter_px / 2.0,
        diameter_px,
        center_y_offset_from_baseline: -(effective_x_height_px / 2.0),
        vertical_adjustment_px: -(effective_x_height_px * BULLET_VERTICAL_OFFSET_FACTOR),
        horizontal_adjustment_px: -(diameter_px * BULLET_HORIZONTAL_OFFSET_FACTOR),
    }
}

pub struct MarkerMetrics {
    pub marker_box_width_px: f64,
    pub marker_gap_px: f64,
    pub bullet_radius_px: f64,
}

/// `computeMarkerMetrics` (`lists.ts:40-63`).
pub fn compute_marker_metrics(
    font_size_px: f64,
    x_height_px: Option<f64>,
    ascent_px: Option<f64>,
) -> MarkerMetrics {
    let marker_box_by_indent = MARKER_BOX_FACTOR * (INDENT_PER_LEVEL_FACTOR * font_size_px);
    let marker_box_by_font = MARKER_BOX_MAX_FACTOR * font_size_px;
    let marker_box_width_px = marker_box_by_indent.min(marker_box_by_font);
    let marker_gap_px = MARKER_GAP_MIN.max(MARKER_GAP_FACTOR * font_size_px);
    let bullet = compute_bullet_metrics(font_size_px, x_height_px, ascent_px);
    MarkerMetrics {
        marker_box_width_px,
        marker_gap_px,
        bullet_radius_px: bullet.radius_px,
    }
}

pub struct ListIndentation {
    pub left_offset_px: f64,
    pub indent_px: f64,
    pub marker_box_width_px: f64,
    pub marker_gap_px: f64,
    pub bullet_radius_px: f64,
}

/// The hanging indent of a list's items at level 0: `max(24px, 1em)` for bullets (ProseMirror padding-left
/// parity), `max(24px, 1.5em)` for numbered items so a counter like `10.` fits before the text (E7.16; the
/// editing surface pads `<ol>` the same, `ProseMirrorEditor.tsx`).
pub fn list_padding_left_px(list_kind: Option<&str>, font_size: f64) -> f64 {
    let em = if list_kind == Some("ol") {
        ORDERED_LIST_PADDING_FACTOR * font_size
    } else {
        font_size
    };
    LIST_PADDING_LEFT_MIN_PX.max(crate::jsnum::js_math_round(em))
}

/// `ORDERED_LIST_PADDING_FACTOR` (`lists.ts`): a numbered item's hanging indent, in ems.
pub const ORDERED_LIST_PADDING_FACTOR: f64 = 1.5;

/// `computeListIndentation` (`positioning.ts:77-123`): UL **and** OL get the
/// indent math; text starts at `indent + max(24px, round(fontSize))`
/// (ProseMirror padding-left parity, positioning.ts:106-114).
pub fn compute_list_indentation(
    list_kind: Option<&str>,
    list_level: i64,
    font_size: f64,
    x_height_px: Option<f64>,
    ascent_px: Option<f64>,
) -> ListIndentation {
    let is_list = matches!(list_kind, Some("ul") | Some("ol"));
    if !is_list {
        return ListIndentation {
            left_offset_px: 0.0,
            indent_px: 0.0,
            marker_box_width_px: 0.0,
            marker_gap_px: 0.0,
            bullet_radius_px: 0.0,
        };
    }
    let indent_px = compute_list_indent_px(list_level, font_size);
    let list_padding_left_px = list_padding_left_px(list_kind, font_size);
    let m = compute_marker_metrics(font_size, x_height_px, ascent_px);
    ListIndentation {
        left_offset_px: indent_px + list_padding_left_px,
        indent_px,
        marker_box_width_px: m.marker_box_width_px,
        marker_gap_px: m.marker_gap_px,
        bullet_radius_px: m.bullet_radius_px,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentPosition {
    pub start_x: f64,
    pub width: f64,
}

/// `computeSegmentPositions` (`positioning.ts:135-154`): pure prefix sums of
/// measured widths.
pub fn compute_segment_positions<M: TextMeasurer>(
    line: &Line,
    line_start_x: f64,
    list_left_offset: f64,
    measurer: &M,
) -> Vec<SegmentPosition> {
    let mut positions = Vec::with_capacity(line.segments.len());
    let mut cursor = line_start_x + list_left_offset;
    for seg in &line.segments {
        let width = segment_advance(seg, measurer);
        positions.push(SegmentPosition {
            start_x: cursor,
            width,
        });
        cursor += width;
    }
    positions
}

/// The spaces a justified line widens: every U+0020 / U+00A0 in its
/// whitespace-only segments (the wrap puts each run of spaces in a segment of
/// its own; tabs keep their width).
pub fn justify_opportunities(line: &Line) -> usize {
    line.segments.iter().map(|seg| space_count(&seg.text)).sum()
}

/// The widening spaces of one segment: its U+0020 / U+00A0 when it is all
/// whitespace, else none.
pub fn space_count(text: &str) -> usize {
    if text.is_empty() || !text.chars().all(|c| c == ' ' || c == '\t' || c == '\u{00A0}') {
        return 0;
    }
    text.chars().filter(|&c| c == ' ' || c == '\u{00A0}').count()
}

/// `text-align: justify` (as in Figma and CSS): how far each space of `line`
/// widens so the line fills `available_width`. Only lines the wrap broke
/// (`Line::soft_wrap`) justify; a paragraph's last line and a line before a
/// hard break stay flush left. Never negative: an overfull line (one word
/// wider than the box) keeps its spaces.
pub fn justify_space_px(align: &str, line: &Line, total_width: f64, available_width: f64) -> f64 {
    // The every-line composer set the line's spacing itself (`crate::composer`).
    if let Some(px) = line.space_px {
        return px;
    }
    if align != "justify" || !line.soft_wrap {
        return 0.0;
    }
    let n = justify_opportunities(line);
    if n == 0 {
        return 0.0;
    }
    ((available_width - total_width) / n as f64).max(0.0)
}

/// A measurer whose whitespace-only segments are `space_px` wider per space:
/// what a justified line's segment positions and decorations measure with.
pub struct JustifiedMeasurer<'a, M: TextMeasurer + ?Sized> {
    pub inner: &'a M,
    pub space_px: f64,
}

impl<M: TextMeasurer + ?Sized> TextMeasurer for JustifiedMeasurer<'_, M> {
    fn measure(&self, text: &str, style: &crate::types::InlineStyle) -> f64 {
        let w = self.inner.measure(text, style);
        // Negative: the every-line composer shrank the line's spaces.
        if self.space_px != 0.0 {
            w + self.space_px * space_count(text) as f64
        } else {
            w
        }
    }

    fn auto_line_height_px(&self, style: &crate::types::InlineStyle) -> Option<f64> {
        self.inner.auto_line_height_px(style)
    }
}

/// [`compute_segment_positions`] for a justified line: each space of a
/// whitespace-only segment is `space_px` wider ([`justify_space_px`]).
pub fn compute_segment_positions_justified<M: TextMeasurer + ?Sized>(
    line: &Line,
    line_start_x: f64,
    list_left_offset: f64,
    measurer: &M,
    space_px: f64,
) -> Vec<SegmentPosition> {
    let j = JustifiedMeasurer { inner: measurer, space_px };
    compute_segment_positions(line, line_start_x, list_left_offset, &j)
}

/// `computeBulletCenterX` (`bulletMetrics.ts:176-189`).
pub fn compute_bullet_center_x(
    text_start_x: f64,
    marker_box_width_px: f64,
    marker_gap_px: f64,
    left_offset_px: f64,
    horizontal_adjustment_px: f64,
) -> f64 {
    let marker_span_px = marker_box_width_px + marker_gap_px;
    let base_center_x = if marker_span_px > 0.0 && marker_span_px <= left_offset_px {
        text_start_x - marker_span_px * 0.5
    } else {
        text_start_x - marker_box_width_px * 0.5
    };
    base_center_x + horizontal_adjustment_px
}

/// `advancePenYForParagraphBreak` (`utils/richtext/paragraphSpacing.ts:1-9`).
pub fn advance_pen_y_for_paragraph_break(
    pen_y: f64,
    is_first_paragraph: bool,
    paragraph_spacing_px: f64,
) -> f64 {
    if is_first_paragraph {
        return pen_y;
    }
    pen_y + paragraph_spacing_px.max(0.0)
}
