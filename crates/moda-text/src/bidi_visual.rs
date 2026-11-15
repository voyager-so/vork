//! Line-level UAX #9 visual reordering of layout segments — the native twin of
//! `utils/richtext/bidiLayout.ts` (`computeVisualLine` /
//! `computeVisualDecorationRects`), used by the cosmic-text Degraded lane
//! (`crate::degraded`), the only native lane that renders RTL scripts (the
//! vector/MSDF lanes decline complex-script nodes — `crate::complex_script`).
//!
//! WHY: the shared layout core wraps in LOGICAL order and emits one segment
//! per word group / space / style run (`crate::wrap`), and the draw walk
//! assigns x positions in that logical order (`compute_segment_positions`
//! prefix sums). Each segment shapes correctly on its own — cosmic-text
//! ShapeLine handles joining and within-segment bidi — but a segment is
//! typically ONE WORD, so Arabic/Hebrew lines rendered their words
//! left-to-right in logical order: letters joined, word order wrong.
//!
//! The fix is display-only and enters at the line-assembly seam. UAX #9's W/N
//! rules are PARAGRAPH-scoped (W2 turns a European number after Arabic into AN
//! even across a line wrap), so the resolution is per paragraph
//! ([`paragraph_bidi_if_rtl`], at the paragraph's base level —
//! `crate::paragraph_direction`, as `bidiLayout.ts`; alignment is separate, so an
//! unaligned all-Arabic paragraph stays left) with L1/L2 applied per line (`visual_runs` over the line's
//! byte range). [`visual_line`] maps a wrapped line's segments back to their
//! paragraph ranges ([`match_line_segment_ranges`]), splits them into
//! single-embedding-level pieces (a trailing `،` resolves to the base level
//! and must detach visually from its word — UAX #9 N1/N2), and assigns each
//! piece's `start_x` in visual run order. Wrapping, line breaking, alignment
//! (`line_start_x` still comes from the logical `measure_line_width`),
//! intrinsic sizing, and glyph emission are untouched — an LTR-only paragraph
//! short-circuits to `None` and the draw walk is byte-identical.
//!
//! Known v1 divergence from the TS twin: bidi-js iterates UTF-16 code units
//! and defaults astral surrogate halves to class L, so a pictographic emoji
//! between two RTL words stays an LTR island there; `unicode-bidi` resolves
//! real char classes (ON — the emoji reorders with the surrounding neutral
//! run). Both are strictly better than the pre-fix behavior (no reordering).

use crate::decorations::{decoration_rect_for, trim_widths, DecorationRect};
use crate::measure::TextMeasurer;
use crate::positioning::SegmentPosition;
use crate::types::{Line, LineSegment};

/// Every RTL-relevant code point (strong RTL scripts, presentation forms, and
/// the RLM/ALM/RLE/RLO/RLI controls) is ≥ U+0590, so this is a sound cheap
/// gate: below it, no reordering can occur. False positives (CJK, Devanagari…)
/// just pay one UBA resolution that returns "all LTR".
fn maybe_rtl(text: &str) -> bool {
    text.chars().any(|c| (c as u32) >= 0x0590)
}

/// A line prepared for visual-order drawing: `line.segments` are the original
/// segments split into single-embedding-level pieces (style cloned, logical
/// order), `positions` carry each piece's LEFT edge assigned in visual order,
/// and `order` is the visual (left-to-right display) order as indices into
/// `line.segments`.
pub struct VisualLine {
    pub line: Line,
    pub positions: Vec<SegmentPosition>,
    pub order: Vec<usize>,
}

/// Paragraph-level UBA resolution for the line-assembly reorder, or `None`
/// when it can never reorder anything: no RTL-relevant code points, or every
/// resolved level is even (all-LTR). `rtl`: the paragraph's base direction. The TS twin is
/// `bidiLayout.ts::computeParagraphBidiIfRtl`. Callers resolve the paragraph
/// ONCE and feed each wrapped line its ranges — per-line resolution would
/// mis-order digits/Latin at line starts (UAX #9's W/N rules are
/// paragraph-scoped).
pub fn paragraph_bidi_if_rtl(text: &str, rtl: bool) -> Option<unicode_bidi::BidiInfo<'_>> {
    // An RTL base (`crate::paragraph_direction`) reorders even all-LTR text: neutrals at its ends take the base level.
    if text.is_empty() || (!rtl && !maybe_rtl(text)) {
        return None;
    }
    let base = if rtl {
        unicode_bidi::Level::rtl()
    } else {
        unicode_bidi::Level::ltr()
    };
    let bidi = unicode_bidi::BidiInfo::new(text, Some(base));
    if bidi.levels.iter().all(|l| !l.is_rtl()) {
        return None;
    }
    // NOTE: the text can legitimately resolve to MULTIPLE Unicode paragraphs —
    // a `<br>` hard break is kept as a `"\n"` run inside one rich-text
    // Paragraph (parser.rs `br` handling), and UAX #9 treats `\n` as a
    // paragraph separator. `visual_line` selects the bidi paragraph that
    // contains each wrapped line (lines never span a hard break — the wrap
    // breaks on `\n`).
    Some(bidi)
}

/// Map one wrapped line's segments back to byte ranges in the paragraph text,
/// starting from `cursor` (bytes consumed by earlier lines). The wrap only
/// ever drops whitespace and U+200B (collapsed spaces and explicit break points), so a mismatch is
/// healed by skipping those characters; anything else returns `None` and the caller
/// draws the rest of the paragraph in logical order (defensive — should not
/// happen). Mirror of `bidiLayout.ts::matchLineSegmentRanges`.
pub fn match_line_segment_ranges(
    para_text: &str,
    segments: &[LineSegment],
    mut cursor: usize,
) -> Option<(Vec<std::ops::Range<usize>>, usize)> {
    let mut ranges = Vec::with_capacity(segments.len());
    for seg in segments {
        while cursor < para_text.len() && !para_text[cursor..].starts_with(seg.text.as_str()) {
            match para_text[cursor..].chars().next() {
                Some(c) if c.is_whitespace() || c == '\u{200B}' => cursor += c.len_utf8(),
                _ => return None,
            }
        }
        if !para_text[cursor..].starts_with(seg.text.as_str()) {
            return None;
        }
        ranges.push(cursor..cursor + seg.text.len());
        cursor += seg.text.len();
    }
    Some((ranges, cursor))
}

fn piece_style(seg: &LineSegment, range: &std::ops::Range<usize>, start: usize, end: usize, rtl: bool) -> crate::types::InlineStyle {
    use unicode_segmentation::UnicodeSegmentation;
    let from = seg.text[..start-range.start].chars().count();
    let text = &seg.text[start-range.start..end-range.start];
    let mut style = seg.style.source_slice(from..from+text.chars().count());
    if rtl && crate::paragraph_direction::first_strong_direction(text) != Some(true) {
        let mut offset = 0;
        let mut pieces = Vec::new();
        for g in text.graphemes(true) {
            let end = offset + g.chars().count();
            pieces.push(style.source.slice(offset..end));
            offset = end;
        }
        style.source = crate::types::SourceIndices::new(pieces.iter().rev().flat_map(|s| s.values()).collect());
    }
    style
}

/// Resolve a wrapped line for visual-order drawing against its PARAGRAPH's
/// UBA resolution ([`paragraph_bidi_if_rtl`]) and the line's segment ranges
/// within it ([`match_line_segment_ranges`]). `None` means "draw the logical
/// layout unchanged": an all-even-level line, an identity resolution with no
/// splits, or a failed consistency check (defensive — logical order is always
/// safe).
///
/// Segments are split at embedding-level boundaries
/// (`bidiLayout.ts::splitTextByEmbeddingLevel` semantics — exact level, not
/// parity: level-2 digits and level-0 Latin both render LTR but sit on
/// opposite sides of an RTL run), so every emitted piece stays contiguous
/// under the UAX #9 L2 reversals; L1/L2 run per line via
/// `visual_runs(para, line_range)`. Piece widths come from `measurer`, with
/// one letter-spacing restored at each split boundary so the pieces sum to
/// the unsplit measure that drove wrapping/alignment. Drawing a piece per
/// level run is what Chromium does inside one `fillText`, so per-piece
/// shaping matches the parity target (no shaping crosses a direction
/// boundary).
pub fn visual_line<M: TextMeasurer>(
    bidi: &unicode_bidi::BidiInfo<'_>,
    para_text: &str,
    ranges: &[std::ops::Range<usize>],
    line: &Line,
    start_x: f64,
    measurer: &M,
) -> Option<VisualLine> {
    if line.segments.is_empty() || ranges.len() != line.segments.len() {
        return None;
    }
    let line_start = ranges.first()?.start;
    let line_end = ranges.last()?.end;

    // Split each segment into maximal single-level pieces (resolved levels are
    // char-uniform, so boundaries land on char boundaries). `piece_of_byte` is
    // indexed by paragraph byte − line_start; bytes the wrap dropped between
    // segments (healed whitespace) map to no piece.
    let mut pieces: Vec<LineSegment> = Vec::with_capacity(line.segments.len());
    // Letter-spacing restored at split boundaries (non-final pieces of a split
    // segment): the unsplit measure that drove wrapping/alignment adds spacing
    // between ALL graphemes, so each split boundary loses exactly one.
    let mut width_extra: Vec<f64> = Vec::with_capacity(line.segments.len());
    let mut piece_of_byte: Vec<Option<usize>> = vec![None; line_end - line_start];
    for (seg, range) in line.segments.iter().zip(ranges) {
        if seg.tab.is_some() {
            // A resolved tab (`crate::tab_stops`) stays one piece with its fill: its advance and leader.
            pieces.push(seg.clone());
            width_extra.push(0.0);
            for pb in range.clone() {
                piece_of_byte[pb - line_start] = Some(pieces.len() - 1);
            }
            continue;
        }
        let mut piece_start = range.start;
        for (rel, _) in seg.text.char_indices() {
            let b = range.start + rel;
            if b > piece_start && bidi.levels[b] != bidi.levels[piece_start] {
                pieces.push(LineSegment {
                    text: piece_text(
                        &para_text[piece_start..b],
                        bidi.levels[piece_start].is_rtl(),
                    ),
                    style: piece_style(seg, range, piece_start, b, bidi.levels[piece_start].is_rtl()),
                    tab: None,
                    hyphen: false,
                });
                width_extra.push(seg.style.letter_spacing);
                for pb in piece_start..b {
                    piece_of_byte[pb - line_start] = Some(pieces.len() - 1);
                }
                piece_start = b;
            }
        }
        if range.end > piece_start {
            pieces.push(LineSegment {
                text: piece_text(
                    &para_text[piece_start..range.end],
                    bidi.levels[piece_start].is_rtl(),
                ),
                style: piece_style(seg, range, piece_start, range.end, bidi.levels[piece_start].is_rtl()),
                tab: None,
                hyphen: false,
            });
            width_extra.push(0.0);
            for pb in piece_start..range.end {
                piece_of_byte[pb - line_start] = Some(pieces.len() - 1);
            }
        }
    }

    // Visual order of the pieces: L1/L2 over THIS line's byte range with the
    // paragraph-resolved levels; walk the visual level runs — an RTL run
    // displays its pieces in reverse logical order. A `<br>` hard break makes
    // the text multi-paragraph under UAX #9, so select the bidi paragraph
    // CONTAINING this line (lines never span a hard break; anything else is
    // the defensive logical fallback).
    let para = bidi
        .paragraphs
        .iter()
        .find(|p| p.range.start <= line_start && line_end <= p.range.end)?;
    let (levels, runs) = bidi.visual_runs(para, line_start..line_end);
    let mut order: Vec<usize> = Vec::with_capacity(pieces.len());
    let mut seen = vec![false; pieces.len()];
    let mut prev = usize::MAX;
    for run in runs {
        let rtl = levels[run.start].is_rtl();
        // An RTL run displays its bytes in reverse logical order. Index
        // arithmetic rather than a boxed iterator: this is the draw path, and
        // one heap allocation per run per line buys nothing here.
        for i in 0..(run.end - run.start) {
            let b = if rtl { run.end - 1 - i } else { run.start + i };
            let Some(p) = piece_of_byte[b - line_start] else {
                continue; // a wrap-dropped whitespace byte between segments
            };
            if p == prev {
                continue;
            }
            if seen[p] {
                // A piece torn non-contiguous in visual space (only reachable
                // via explicit embedding controls) — logical order is the
                // safe fallback.
                return None;
            }
            seen[p] = true;
            prev = p;
            order.push(p);
        }
    }
    if order.len() != pieces.len() {
        return None;
    }
    // All-even line (e.g. a digits-only continuation line) or identity with no
    // splits ⇒ the logical draw already renders this exactly — keep the
    // untouched (byte-exact) path.
    if pieces.len() == line.segments.len() && order.iter().enumerate().all(|(i, p)| i == *p) {
        return None;
    }

    let mut positions = vec![
        SegmentPosition {
            start_x: 0.0,
            width: 0.0,
        };
        pieces.len()
    ];
    let mut cursor = start_x;
    for &p in &order {
        let width = crate::positioning::segment_advance(&pieces[p], measurer) + width_extra[p];
        positions[p] = SegmentPosition {
            start_x: cursor,
            width,
        };
        cursor += width;
    }

    Some(VisualLine {
        line: Line {
            segments: pieces,
            line_height_px: line.line_height_px,
            soft_wrap: line.soft_wrap,
            ..Default::default()
        },
        positions,
        order,
    })
}

/// A piece's text as it is drawn. The shaper takes a piece's direction from its first strong character, so a
/// right-to-left piece with none (only neutrals: the brackets around a Latin name in a Hebrew line, a lone `!`) would
/// draw left to right and unmirrored, `(Avi)` as `)Avi(`. Such a piece is given in display order, reversed with
/// each character mirrored (UAX #9 L3/L4); a piece that starts right-to-left is the shaper's to mirror.
fn piece_text(text: &str, rtl: bool) -> String {
    use unicode_segmentation::UnicodeSegmentation;
    if !rtl || crate::paragraph_direction::first_strong_direction(text) == Some(true) {
        return text.to_string();
    }
    text.graphemes(true)
        .rev()
        .flat_map(|g| {
            g.chars()
                .map(|c| unicode_bidi_mirroring::get_mirrored(c).unwrap_or(c))
        })
        .collect()
}

/// Underline/strikethrough rects for a visually reordered line — the native
/// twin of `bidiLayout.ts::computeVisualDecorationRects`. The shared helpers
/// (`crate::decorations`) accumulate x in LOGICAL order, wrong for a reordered
/// line; this builds the same rect model from the already-assigned visual
/// positions instead: visually-adjacent decorated pieces merge into one rect
/// (underlines continue under spaces, CSS behavior) with whitespace trimmed
/// only at the group's visual edges; thickness/offset model unchanged; group
/// color follows the LAST visual piece (the shared helpers' end-segment
/// color, in visual space). Returned x's are ABSOLUTE (from `positions`), not
/// line-relative like the logical helpers.
fn compute_visual_decoration_rects<M: TextMeasurer>(
    line: &Line,
    order: &[usize],
    positions: &[SegmentPosition],
    baseline_y: f64,
    measurer: &M,
    wants: fn(&str) -> bool,
    underline: bool,
) -> Vec<DecorationRect> {
    let mut rects: Vec<DecorationRect> = Vec::new();
    let mut group: Vec<usize> = Vec::new();

    let mut flush = |group: &mut Vec<usize>| {
        if group.is_empty() {
            return;
        }
        let first = group[0];
        let last = group[group.len() - 1];
        let first_seg = &line.segments[first];
        let last_seg = &line.segments[last];
        let (leading_width, _) =
            trim_widths(&first_seg.text, |t| measurer.measure(t, &first_seg.style));
        let (_, trailing_width) =
            trim_widths(&last_seg.text, |t| measurer.measure(t, &last_seg.style));
        let x1 = positions[first].start_x + leading_width;
        let x2 = positions[last].start_x + positions[last].width - trailing_width;
        if x2 > x1 {
            rects.push(decoration_rect_for(
                group.iter().map(|&i| &line.segments[i].style),
                &last_seg.style,
                x1,
                x2,
                baseline_y,
                underline,
                false,
            ));
        }
        group.clear();
    };

    for &i in order {
        let dec = line.segments[i]
            .style
            .text_decoration
            .as_deref()
            .unwrap_or("none");
        if wants(dec) {
            // A group breaks where the decoration styling changes (crate::decorations).
            if let Some(&g) = group.first() {
                if line.segments[g].style.decoration != line.segments[i].style.decoration {
                    flush(&mut group);
                }
            }
            group.push(i);
        } else {
            flush(&mut group);
        }
    }
    flush(&mut group);
    rects
}

/// Visual-order underline rects (`underlineY = baseline + 0.10 × maxFontSize`).
pub fn compute_visual_underline_rects<M: TextMeasurer>(
    line: &Line,
    order: &[usize],
    positions: &[SegmentPosition],
    baseline_y: f64,
    measurer: &M,
) -> Vec<DecorationRect> {
    compute_visual_decoration_rects(
        line,
        order,
        positions,
        baseline_y,
        measurer,
        |dec| dec.contains("underline"),
        true,
    )
}

/// Visual-order strikethrough rects (`strikeY = baseline − 0.30 × maxFontSize`).
pub fn compute_visual_strikethrough_rects<M: TextMeasurer>(
    line: &Line,
    order: &[usize],
    positions: &[SegmentPosition],
    baseline_y: f64,
    measurer: &M,
) -> Vec<DecorationRect> {
    compute_visual_decoration_rects(
        line,
        order,
        positions,
        baseline_y,
        measurer,
        |dec| dec.contains("line-through"),
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{default_style, InlineStyle};

    struct CharWidth;
    impl TextMeasurer for CharWidth {
        fn measure(&self, text: &str, _style: &InlineStyle) -> f64 {
            text.chars().count() as f64 * 10.0
        }
    }

    fn line(texts: &[&str]) -> Line {
        Line {
            segments: texts
                .iter()
                .map(|t| LineSegment {
                    text: (*t).to_string(),
                    style: default_style(),
                    tab: None,
                    hyphen: false,
                })
                .collect(),
            line_height_px: 19.2,
            soft_wrap: false,
            ..Default::default()
        }
    }

    fn texts(v: &VisualLine) -> Vec<&str> {
        v.line.segments.iter().map(|s| s.text.as_str()).collect()
    }

    /// Piece texts in VISUAL (left-to-right display) order.
    fn visual_texts(v: &VisualLine) -> Vec<&str> {
        v.order
            .iter()
            .map(|&i| v.line.segments[i].text.as_str())
            .collect()
    }

    const AR_1: &str = "\u{0647}\u{062F}\u{064A}\u{0629}"; // هدية
    const AR_2: &str = "\u{0628}\u{0633}\u{064A}\u{0637}\u{0629}"; // بسيطة
    const AR_COMMA: &str = "\u{060C}"; // ،
    const HE: &str = "\u{05D0}\u{05D1}"; // אב

    /// One-line paragraph resolution (the common case): paragraph text is the
    /// line's own concatenated segments.
    fn resolve_one_line<M: TextMeasurer>(
        l: &Line,
        start_x: f64,
        measurer: &M,
    ) -> Option<VisualLine> {
        let para_text: String = l.segments.iter().map(|s| s.text.as_str()).collect();
        let bidi = paragraph_bidi_if_rtl(&para_text, false)?;
        let (ranges, _) = match_line_segment_ranges(&para_text, &l.segments, 0).expect("ranges");
        visual_line(&bidi, &para_text, &ranges, l, start_x, measurer)
    }

    fn resolve(l: &Line, start_x: f64) -> Option<VisualLine> {
        resolve_one_line(l, start_x, &CharWidth)
    }

    /// Pure-LTR lines and empty lines never reorder (fast-path byte-exactness
    /// for all existing content).
    #[test]
    fn ltr_and_empty_lines_stay_logical() {
        assert!(resolve(&line(&["hello", " ", "world"]), 0.0).is_none());
        assert!(paragraph_bidi_if_rtl("", false).is_none());
        // CJK is ≥ U+0590 (pays the UBA call) but resolves all-LTR.
        assert!(resolve(&line(&["你好", " ", "世界"]), 0.0).is_none());
        assert!(paragraph_bidi_if_rtl("hello world", false).is_none());
        // A single RTL island between Latin anchors resolves to the identity
        // (no split, no reorder) — logical draw kept byte-exact.
        assert!(resolve(&line(&["ab", " ", HE, " ", "cd"]), 0.0).is_none());
    }

    /// A menu line in Arabic, `هدية بسيطة<tab>12`, with a right stop and a dot leader: the reorder keeps the tab
    /// one piece with its advance and leader (it was re-measured as a space-less "\t" and lost its dots).
    #[test]
    fn a_resolved_tab_keeps_its_advance_and_leader_through_the_reorder() {
        let mut l = line(&[AR_1, " ", AR_2, "\t", "12"]);
        let fill = crate::tab_stops::TabFill {
            advance: 200.0,
            leader: Some(crate::tab_stops::Leader {
                text: ".".to_string(),
                offset: 5.0,
                unit: 10.0,
                count: 19,
                head: None,
                tail: None,
            }),
        };
        l.segments[3].tab = Some(fill.clone());
        let v = resolve(&l, 0.0).expect("reorders");
        let tab = v
            .line
            .segments
            .iter()
            .position(|s| s.text == "\t")
            .expect("the tab piece");
        assert_eq!(v.line.segments[tab].tab.as_ref(), Some(&fill));
        assert_eq!(v.positions[tab].width, 200.0);
        // The words (40 + 10 + 50) then the tab's 200 then "12": the line is as wide as the wrap measured it.
        let end = v
            .positions
            .iter()
            .map(|p| p.start_x + p.width)
            .fold(0.0, f64::max);
        assert_eq!(end, 40.0 + 10.0 + 50.0 + 200.0 + 20.0);
    }

    /// The prod field case, `هدية بسيطة،` authored logically: per-word wrap
    /// segments must display as `[بسيطة][ ][هدية]` left-to-right with the
    /// trailing comma DETACHED to the far right (U+060C is class CS → neutral
    /// at eot → base level, UAX #9 N1/N2) — exactly Chromium's resolution of
    /// the same string with an LTR base.
    #[test]
    fn pure_arabic_line_reverses_word_segments_and_detaches_trailing_comma() {
        let ar2_comma = format!("{AR_2}{AR_COMMA}");
        let l = line(&[AR_1, " ", &ar2_comma]);
        let v = resolve(&l, 0.0).expect("reorders");
        // The comma splits off its word (different embedding level).
        assert_eq!(texts(&v), vec![AR_1, " ", AR_2, AR_COMMA]);
        assert_eq!(visual_texts(&v), vec![AR_2, " ", AR_1, AR_COMMA]);
        // Positions assigned left-to-right in visual order from start_x.
        assert_eq!(v.positions[2].start_x, 0.0); // بسيطة (5 chars → 50)
        assert_eq!(v.positions[1].start_x, 50.0); // space
        assert_eq!(v.positions[0].start_x, 60.0); // هدية (4 chars → 40)
        assert_eq!(v.positions[3].start_x, 100.0); // ، far right
    }

    /// Mixed direction under the LTR base: Latin anchors keep their place,
    /// the RTL island's word segments reverse in place (UAX #9 L2).
    #[test]
    fn mixed_bidi_reverses_only_the_rtl_island() {
        let l = line(&["ab", " ", AR_1, " ", AR_2, " ", "cd"]);
        let v = resolve(&l, 0.0).expect("reorders");
        assert_eq!(v.order, vec![0, 1, 4, 3, 2, 5, 6]);
    }

    #[test]
    fn brackets_around_latin_in_an_rtl_line_draw_mirrored() {
        // "אברהם (Avi) Cohen": the brackets resolve right-to-left (N0) but hold no strong character, so the shaper
        // would draw them as typed; their pieces are mirrored, and the line reads "Cohen (Avi) אברהם".
        let he = "\u{05D0}\u{05D1}\u{05E8}\u{05D4}\u{05DD}";
        let l = line(&[he, " ", "(Avi)", " ", "Cohen"]);
        let para_text: String = l.segments.iter().map(|s| s.text.as_str()).collect();
        let bidi = paragraph_bidi_if_rtl(&para_text, true).expect("rtl");
        let (ranges, _) = match_line_segment_ranges(&para_text, &l.segments, 0).expect("ranges");
        let v = visual_line(&bidi, &para_text, &ranges, &l, 0.0, &CharWidth).expect("reorders");
        assert_eq!(
            visual_texts(&v),
            vec!["Cohen", " ", "(", "Avi", ")", " ", he]
        );
        // A piece that starts right-to-left is left to the shaper.
        assert_eq!(piece_text("(\u{05E9})", true), "(\u{05E9})");
        assert_eq!(piece_text("[{", true), "}]");
        assert_eq!(piece_text("(", false), "(");
    }

    /// European digits between Arabic words resolve to level 2 (AN inside the
    /// RTL run): the number reads LTR internally but takes its RTL slot in
    /// the word order.
    #[test]
    fn digits_keep_ltr_but_take_rtl_slot() {
        let l = line(&[AR_1, " ", "123", " ", AR_2]);
        let v = resolve(&l, 0.0).expect("reorders");
        assert_eq!(v.order, vec![4, 3, 2, 1, 0]);
    }

    /// Multi-level SEGMENTS split into single-level pieces so the visual walk
    /// never tears a drawn run: "ab هدية" + "بسيطة cd" interleave visually
    /// under the LTR base (the two RTL word runs merge between the Latin
    /// anchors).
    #[test]
    fn multi_level_segments_split_and_reorder() {
        let l = line(&[&format!("ab {AR_1}"), &format!("{AR_2} cd")]);
        let v = resolve(&l, 0.0).expect("reorders");
        assert_eq!(texts(&v), vec!["ab ", AR_1, AR_2, " cd"]);
        // Visual: [ab ] [بسيطة] [هدية] [ cd] — the merged RTL run reverses.
        assert_eq!(v.order, vec![0, 2, 1, 3]);
    }

    #[test]
    fn total_width_is_invariant_under_reordering() {
        let ar2_comma = format!("{AR_2}{AR_COMMA}");
        let l = line(&[AR_1, " ", &ar2_comma]);
        let v = resolve(&l, 7.0).expect("reorders");
        let right_edge = v
            .positions
            .iter()
            .map(|p| p.start_x + p.width)
            .fold(0.0, f64::max);
        // 11 chars × 10 starting at 7.
        assert_eq!(right_edge, 7.0 + 110.0);
    }

    /// PARAGRAPH-scoped resolution across a wrap (UAX #9 W2): a European
    /// number that starts a continuation line still follows Arabic in the
    /// paragraph, so it resolves to AN (level 2) inside the RTL run and
    /// renders at the line's RIGHT — a per-line resolution would see EN at
    /// start-of-text, give it the base level, and mis-place it at the LEFT.
    #[test]
    fn wrapped_continuation_line_keeps_paragraph_context() {
        // Paragraph "هدية 123 بسيطة" wrapped after the first word; the wrap
        // drops the break space, so line 2's segments start at byte 9.
        let para_text = format!("{AR_1} 123 {AR_2}");
        let line2 = line(&["123", " ", AR_2]);
        let bidi = paragraph_bidi_if_rtl(&para_text, false).expect("rtl paragraph");
        // Line 1 consumed AR_1; cursor heals the dropped space.
        let (ranges, cursor) =
            match_line_segment_ranges(&para_text, &line2.segments, AR_1.len()).expect("ranges");
        assert_eq!(cursor, para_text.len());
        let v = visual_line(&bidi, &para_text, &ranges, &line2, 0.0, &CharWidth)
            .expect("paragraph context reorders the continuation line");
        // Visual: [بسيطة] [ ] [123] — digits keep their RTL slot (right).
        assert_eq!(v.order, vec![2, 1, 0]);
    }

    /// A `<br>` hard break is a `"\n"` run INSIDE one rich-text Paragraph, and
    /// UAX #9 treats `\n` as a paragraph separator — the resolution must not
    /// bail on multi-paragraph text (that regressed every RTL paragraph
    /// containing a hard break to logical order). Each wrapped line selects
    /// its containing bidi paragraph.
    #[test]
    fn hard_break_paragraph_reorders_lines_on_both_sides_of_the_break() {
        let para_text = format!("{AR_1} {AR_2}\n{AR_2} {AR_1}");
        let bidi = paragraph_bidi_if_rtl(&para_text, false).expect("multi-paragraph text must resolve");
        // Line 1: before the break.
        let line1 = line(&[AR_1, " ", AR_2]);
        let (r1, c1) = match_line_segment_ranges(&para_text, &line1.segments, 0).expect("ranges");
        let v1 = visual_line(&bidi, &para_text, &r1, &line1, 0.0, &CharWidth).expect("reorders");
        assert_eq!(v1.order, vec![2, 1, 0]);
        // Line 2: after the break (cursor heals the consumed `\n`).
        let line2 = line(&[AR_2, " ", AR_1]);
        let (r2, _) = match_line_segment_ranges(&para_text, &line2.segments, c1).expect("ranges");
        let v2 = visual_line(&bidi, &para_text, &r2, &line2, 0.0, &CharWidth).expect("reorders");
        assert_eq!(v2.order, vec![2, 1, 0]);
    }

    /// Letter-spacing at a split boundary: the unsplit measure (which drove
    /// wrapping/alignment) adds spacing between all graphemes, so each split
    /// restores one spacing on the non-final piece — the pieces sum exactly
    /// to the unsplit width.
    #[test]
    fn split_pieces_preserve_letter_spacing_width() {
        struct SpacedWidth;
        impl TextMeasurer for SpacedWidth {
            fn measure(&self, text: &str, style: &InlineStyle) -> f64 {
                let n = text.chars().count() as f64;
                n * 10.0 + style.letter_spacing * (n - 1.0).max(0.0)
            }
        }
        let ar2_comma = format!("{AR_2}{AR_COMMA}");
        let mut l = line(&[AR_1, " ", &ar2_comma]);
        for seg in &mut l.segments {
            seg.style.letter_spacing = 5.0;
        }
        let v = resolve_one_line(&l, 0.0, &SpacedWidth).expect("reorders");
        // Unsplit logical width: (4·10+3·5) + 10 + (6·10+5·5) = 55+10+85.
        let logical: f64 = l
            .segments
            .iter()
            .map(|s| SpacedWidth.measure(&s.text, &s.style))
            .sum();
        let right_edge = v
            .positions
            .iter()
            .map(|p| p.start_x + p.width)
            .fold(0.0, f64::max);
        assert_eq!(right_edge, logical, "split must not lose boundary spacing");
    }

    /// Visual decoration rects: visually-adjacent decorated pieces merge,
    /// whitespace trims only at the group's visual edges, color follows the
    /// LAST VISUAL piece, x's are absolute from the positions.
    #[test]
    fn visual_decoration_rects_follow_visual_adjacency() {
        let mut l = line(&[AR_1, " ", AR_2]);
        for seg in &mut l.segments {
            seg.style.text_decoration = Some("underline".to_string());
        }
        l.segments[0].style.color = "#111".to_string();
        l.segments[2].style.color = "#222".to_string();
        let v = resolve(&l, 0.0).expect("reorders");
        let rects =
            compute_visual_underline_rects(&v.line, &v.order, &v.positions, 12.0, &CharWidth);
        assert_eq!(rects.len(), 1, "one merged rect: {rects:?}");
        let r = &rects[0];
        assert_eq!(r.x1, 0.0);
        assert_eq!(r.x2, 100.0); // 10 chars × 10 (4 + 1 + 5)
                                 // Last VISUAL piece is logical segment 0 (هدية).
        assert_eq!(r.color, "#111");
        assert_eq!(
            r.y,
            12.0 + default_style().font_size * crate::decorations::UNDERLINE_OFFSET_RATIO
        );
    }
}
