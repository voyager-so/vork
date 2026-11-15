//! Paragraph direction — the twin of `utils/richtext/paragraphDirection.ts` (keep them in lockstep; the shared
//! parity fixture `text-measurement-parity/cases.json` pins both).
//!
//! A paragraph's `dir` attribute (HTML, CSS `direction`) is `auto` (the default: the first strong bidi character of
//! the paragraph decides, as HTML dir=auto / UAX #9 P2-P3), `ltr` or `rtl`. The resolved direction is the
//! paragraph's bidi base level ([`crate::bidi_visual::paragraph_bidi`]), and `text-align: start` / `end` resolve
//! against it (start = right in RTL). Physical alignment never changes with direction, and an unaligned paragraph
//! stays left as before. A justified RTL paragraph sets the lines it does not stretch flush right
//! ([`line_start_align`]).

use crate::types::{Align, Line, Paragraph};

/// The direction a paragraph declares (`dir`); `Auto` without one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeclaredDirection {
    #[default]
    Auto,
    Ltr,
    Rtl,
}

/// `text-align: start` / `end` as written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalAlign {
    Start,
    End,
}

/// A paragraph's direction: as declared, its logical alignment (if any), and resolved ([`resolve_paragraph_direction`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ParagraphDirection {
    pub declared: DeclaredDirection,
    pub logical_align: Option<LogicalAlign>,
    /// The resolved direction is right-to-left: the paragraph's bidi base level.
    pub rtl: bool,
}

/// A `dir` attribute's value (`parseDirAttribute`): `ltr` / `rtl`, anything else `Auto`.
pub fn parse_dir_attribute(value: &str) -> DeclaredDirection {
    match value.trim().to_ascii_lowercase().as_str() {
        "ltr" => DeclaredDirection::Ltr,
        "rtl" => DeclaredDirection::Rtl,
        _ => DeclaredDirection::Auto,
    }
}

/// A block's CSS `direction: ltr | rtl` declaration (`directionFromStyle`: the markup / agent form, which the
/// sanitizer folds into `dir`); `None` without one.
pub fn direction_from_style(style_attr: Option<&str>) -> Option<DeclaredDirection> {
    for decl in style_attr?.split(';') {
        let Some((prop, value)) = decl.split_once(':') else {
            continue;
        };
        if prop.trim().eq_ignore_ascii_case("direction") {
            match value.trim().to_ascii_lowercase().as_str() {
                "ltr" => return Some(DeclaredDirection::Ltr),
                "rtl" => return Some(DeclaredDirection::Rtl),
                _ => {}
            }
        }
    }
    None
}

/// Strong class of one character (`strongClass`): `Some(false)` L, `Some(true)` R / AL, `None` weak or neutral.
/// Astral code points are classed by block exactly as the TS twin does (bidi-js reads UTF-16 code units).
fn strong_class(c: char) -> Option<bool> {
    let cp = c as u32;
    if cp >= 0x10000 {
        if (0x10800..=0x10FFF).contains(&cp) || (0x1E800..=0x1EFFF).contains(&cp) {
            if (0x10D30..=0x10D39).contains(&cp) || (0x10E60..=0x10E7E).contains(&cp) {
                return None;
            }
            return Some(true);
        }
        if (0x1F000..=0x1FAFF).contains(&cp) || (0xE0000..=0xE0FFF).contains(&cp) {
            return None;
        }
        return Some(false);
    }
    use unicode_bidi::BidiClass;
    match unicode_bidi::bidi_class(c) {
        BidiClass::L => Some(false),
        BidiClass::R | BidiClass::AL => Some(true),
        _ => None,
    }
}

/// UAX #9 P2-P3 (`firstStrongDirection`): whether the first strong character of `text` is right-to-left, skipping
/// isolates (LRI / RLI / FSI … PDI); `None` without one. The whole paragraph is scanned, as HTML dir=auto.
pub fn first_strong_direction(text: &str) -> Option<bool> {
    let mut isolate_depth = 0usize;
    for c in text.chars() {
        match c {
            '\u{2066}' | '\u{2067}' | '\u{2068}' => {
                isolate_depth += 1;
                continue;
            }
            '\u{2069}' => {
                isolate_depth = isolate_depth.saturating_sub(1);
                continue;
            }
            _ => {}
        }
        if isolate_depth > 0 {
            continue;
        }
        if c.is_ascii() && !c.is_ascii_alphabetic() {
            continue;
        }
        if let Some(rtl) = strong_class(c) {
            return Some(rtl);
        }
    }
    None
}

/// Whether a paragraph is right-to-left (`isParagraphRtl`): declared, else its first strong character; LTR without.
pub fn is_paragraph_rtl(p: &Paragraph) -> bool {
    match p.direction.declared {
        DeclaredDirection::Ltr => false,
        DeclaredDirection::Rtl => true,
        DeclaredDirection::Auto => {
            let text: String = p.runs.iter().map(|r| r.text.as_str()).collect();
            first_strong_direction(&text) == Some(true)
        }
    }
}

/// `start` / `end` against a direction (`physicalAlign`).
pub fn physical_align(align: LogicalAlign, rtl: bool) -> Align {
    if (align == LogicalAlign::Start) != rtl {
        "left"
    } else {
        "right"
    }
}

/// Resolve a parsed paragraph's direction in place (`resolveParagraphDirection`): `direction.rtl` and a logical
/// alignment (the parser leaves `"start"` / `"end"` in `align`) to `left` / `right`. Idempotent.
pub fn resolve_paragraph_direction(p: &mut Paragraph) {
    match p.align {
        "start" => p.direction.logical_align = Some(LogicalAlign::Start),
        "end" => p.direction.logical_align = Some(LogicalAlign::End),
        _ => {}
    }
    let rtl = is_paragraph_rtl(p);
    p.direction.rtl = rtl;
    if let Some(logical) = p.direction.logical_align {
        p.align = physical_align(logical, rtl);
    }
}

/// The alignment a line's start x comes from (`lineStartAlign`): the paragraph's, except that a justified RTL
/// paragraph puts a line it does not stretch (its last line, a line before a hard break, a line with no space to
/// widen) flush right, as CSS `text-align-last: auto` does.
pub fn line_start_align(align: Align, rtl: bool, line: &Line) -> Align {
    if !rtl || align != "justify" || line.space_px.is_some() {
        return align;
    }
    if !line.soft_wrap || crate::positioning::justify_opportunities(line) == 0 {
        return "right";
    }
    align
}

/// Whether any paragraph has odd bidi levels. The vector lane draws logical order, so both
/// RTL bases and embedded overrides in LTR paragraphs need the shaping renderer.
pub fn html_needs_rtl_reorder(html: &str) -> bool {
    crate::parser::parse_rich_text_to_paragraphs(html)
        .iter()
        .any(|p| {
            let text: String = p.runs.iter().map(|r| r.text.as_str()).collect();
            crate::bidi_visual::paragraph_bidi_if_rtl(&text, p.direction.rtl).is_some()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_strong_skips_neutrals_digits_and_isolates() {
        assert_eq!(first_strong_direction("שלום world"), Some(true));
        assert_eq!(first_strong_direction("Hello שלום"), Some(false));
        assert_eq!(first_strong_direction("123, (!) مرحبا"), Some(true));
        assert_eq!(
            first_strong_direction("\u{2067}abc\u{2069} שלום"),
            Some(true)
        );
        assert_eq!(
            first_strong_direction("\u{2066}שלום\u{2069} abc"),
            Some(false)
        );
        assert_eq!(first_strong_direction("123 !?"), None);
        assert_eq!(first_strong_direction(""), None);
        // Emoji are neutral; an astral RTL letter (Adlam) is strong.
        assert_eq!(first_strong_direction("😀 שלום"), Some(true));
        assert_eq!(first_strong_direction("\u{1E900}"), Some(true));
    }

    #[test]
    fn logical_alignment_follows_direction() {
        assert_eq!(physical_align(LogicalAlign::Start, false), "left");
        assert_eq!(physical_align(LogicalAlign::Start, true), "right");
        assert_eq!(physical_align(LogicalAlign::End, false), "right");
        assert_eq!(physical_align(LogicalAlign::End, true), "left");
    }

    #[test]
    fn rtl_reordering_is_detected_only_where_it_moves_something() {
        assert!(html_needs_rtl_reorder("<p dir=\"rtl\">Hello!</p>"));
        assert!(html_needs_rtl_reorder(
            "<p style=\"direction: rtl\">(1) one</p>"
        ));
        assert!(
            !html_needs_rtl_reorder("<p dir=\"rtl\">Hello world</p>"),
            "all level 2: drawn as is"
        );
        assert!(!html_needs_rtl_reorder("<p>Hello!</p>"));
        assert!(!html_needs_rtl_reorder("<p dir=\"ltr\">Hello!</p>"));
    }
}
