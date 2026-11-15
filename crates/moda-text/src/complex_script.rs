//! Complex-script detection — which runs MUST be shaped by the cosmic-text
//! (rustybuzz + `unicode-bidi`) engine because the otjs/MSDF lanes render them
//! WRONG or blank.
//!
//! The otjs vector shaper maps code points 1:1 through cmap with NO bidi, NO
//! joining, NO reordering and NO GPOS mark positioning (see
//! `vector/otjs.rs::string_to_glyphs_clustered` — it even hard-throws on
//! Arabic). That is correct for Latin/Greek/Cyrillic/CJK/Kana/Hangul (simple,
//! LTR, 1:1) but produces broken output for:
//!
//! - **RTL scripts** (need UAX #9 bidi + cursive joining): Arabic, Hebrew,
//!   Syriac, Thaana, N'Ko, Samaritan, Mandaic, plus the RTL SMP blocks
//!   (Adlam, Mandaic, Hanifi Rohingya, …);
//! - **Brahmic / Indic scripts** (conjuncts, matra reordering, marks):
//!   Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada,
//!   Malayalam, Sinhala;
//! - **Southeast-Asian scripts** (above/below marks, reordering, stacking):
//!   Thai, Lao, Khmer, Myanmar, Tibetan.
//!
//! Any run containing one of these MUST route to the cosmic-text lane
//! (`crate::degraded`), which shapes them correctly (HarfBuzz-class rustybuzz +
//! full-paragraph bidi). Rendered evidence: the cosmic lane produces correct
//! Arabic joining, Hebrew/mixed bidi, Devanagari conjuncts, Thai marks and
//! Khmer stacking where the otjs lane does not.
//!
//! - **Combining marks and conjoining jamo** in any script (decomposed `e\u{301}`,
//!   Vietnamese stacked marks, Hangul written as jamo): the mark needs GPOS
//!   attachment or composition, the jamo need composing into a syllable.
//!
//! Scripts deliberately EXCLUDED (the otjs lane renders them correctly, so they
//! keep the crisp GPU-outline path): Latin, Greek, Cyrillic, Georgian,
//! Armenian, Ethiopic (1:1 syllabic, LTR), Han/CJK, Hiragana, Katakana, Hangul
//! syllables — as long as their text carries no combining mark.

/// True when `cp` belongs to a script that requires HarfBuzz complex shaping
/// and/or bidi reordering the otjs/MSDF lanes cannot perform.
///
/// MIRRORED IN PYTHON: `_SIMPLE_RANGES` in
/// `backend/app/services/pptx/font_metrics.py` is the COMPLEMENT of this — the
/// codepoints the PDF importer will measure by summing advances. It is written
/// as an allow-list precisely so that growing this matcher makes that module
/// decline to measure rather than measure wrongly, but a script MOVING from
/// complex to simple here will not reach it.
///
/// MIRROR: `COMPLEX_SHAPING_RANGES` in
/// `renderers/webgpu/text/vector/complexScripts.ts` is a TypeScript twin of this
/// table, read by the WebGPU text lane and by the PDF export's raster-fallback gate.
/// It must stay a SUPERSET of the ranges below — narrower means text silently drawn
/// on a lane that cannot shape it. The two drifted once (Mongolian, Syriac Supplement
/// and Brahmi were here and missing there), so
/// `tests/vitest/text/complexScripts.ownerParity.test.ts` parses this function and
/// fails when the TS side stops covering it. ADD A RANGE HERE, ADD IT THERE TOO.
///
/// The ranges are inclusive and cover the BMP scripts plus the SMP RTL/Brahmic
/// blocks a customer can realistically type; it is intentionally broader than
/// `vector::unicode_scripts::classify_codepoint` (which omits some SMP/RTL
/// blocks) so nothing in the "complex" space slips through onto the broken
/// otjs path.
pub fn codepoint_needs_complex_shaping(cp: u32) -> bool {
    matches!(cp,
        // ── RTL scripts (bidi + joining) ──────────────────────────────────
        0x0590..=0x05FF   // Hebrew
        | 0x0600..=0x06FF // Arabic
        | 0x0700..=0x074F // Syriac
        | 0x0750..=0x077F // Arabic Supplement
        | 0x0780..=0x07BF // Thaana
        | 0x07C0..=0x07FF // NKo
        | 0x0800..=0x083F // Samaritan
        | 0x0840..=0x085F // Mandaic
        | 0x0860..=0x086F // Syriac Supplement
        | 0x0870..=0x089F // Arabic Extended-B/-A adjoining
        | 0x08A0..=0x08FF // Arabic Extended-A
        | 0xFB1D..=0xFB4F // Hebrew presentation forms
        | 0xFB50..=0xFDFF // Arabic presentation forms-A
        | 0xFE70..=0xFEFF // Arabic presentation forms-B
        // ── Brahmic / Indic (conjuncts, reordering, marks) ────────────────
        | 0x0900..=0x097F // Devanagari
        | 0x0980..=0x09FF // Bengali
        | 0x0A00..=0x0A7F // Gurmukhi
        | 0x0A80..=0x0AFF // Gujarati
        | 0x0B00..=0x0B7F // Oriya
        | 0x0B80..=0x0BFF // Tamil
        | 0x0C00..=0x0C7F // Telugu
        | 0x0C80..=0x0CFF // Kannada
        | 0x0D00..=0x0D7F // Malayalam
        | 0x0D80..=0x0DFF // Sinhala
        | 0xA8E0..=0xA8FF // Devanagari Extended
        | 0x11000..=0x110FF // Brahmi
        // ── Southeast Asian (marks, reordering, stacking) ─────────────────
        | 0x0E00..=0x0E7F // Thai
        | 0x0E80..=0x0EFF // Lao
        | 0x0F00..=0x0FFF // Tibetan
        | 0x1000..=0x109F // Myanmar
        | 0x1780..=0x17FF // Khmer
        | 0x1800..=0x18AF // Mongolian (cursive joining)
        | 0x1900..=0x194F // Limbu
        | 0x19E0..=0x19FF // Khmer Symbols
        | 0x1A20..=0x1AAF // Tai Tham
        | 0xA9E0..=0xA9FF // Myanmar Extended-B
        | 0xAA60..=0xAA7F // Myanmar Extended-A
        // ── SMP RTL / complex blocks ──────────────────────────────────────
        | 0x10D00..=0x10D3F // Hanifi Rohingya
        | 0x1E900..=0x1E95F // Adlam
        // ── Combining marks and conjoining jamo (any script) ──────────────
        // Decomposed text (NFD: macOS file names, text copied out of PDFs) and
        // marks with no precomposed form (Yoruba ẹ́, m̐, Z̵) need GPOS mark
        // attachment or composition; drawn 1:1 the mark lands beside its base
        // and Hangul jamo never join into syllables.
        | 0x0300..=0x036F // Combining Diacritical Marks
        | 0x0483..=0x0489 // Cyrillic combining marks
        | 0x1100..=0x11FF // Hangul Jamo
        | 0x1AB0..=0x1AFF // Combining Diacritical Marks Extended
        | 0x1DC0..=0x1DFF // Combining Diacritical Marks Supplement
        | 0xA960..=0xA97F // Hangul Jamo Extended-A
        | 0xD7B0..=0xD7FF // Hangul Jamo Extended-B
        | 0xFE20..=0xFE2F // Combining Half Marks
    )
}

/// True when the VISIBLE text of `html` contains any code point whose script
/// needs complex shaping (see [`codepoint_needs_complex_shaping`]).
///
/// Tag-aware (and quote-aware): code points inside `<...>` markup are skipped so
/// a complex-script character in a tag/attribute (e.g. a non-ASCII `font-family`
/// NAME in a `style=` attribute) never false-routes an otherwise-Latin node to
/// the cosmic lane. A `>` inside a quoted attribute value does NOT end the tag
/// (so `<span title="a > b" style="font-family:'نوتو'">` stays fully skipped).
///
/// Entity-aware: a NUMERIC character reference (`&#1575;` / `&#x627;`) that
/// decodes to a complex-script code point counts — imported / agent-authored
/// `htmlContent` can encode non-ASCII that way, and the downstream parser
/// decodes it before shaping, so a raw-only scan would blank/mis-shape it.
/// (Named entities can't encode complex scripts, so only numeric refs are
/// checked.)
///
/// Allocation-free and short-circuits on the first complex code point, so it is
/// cheap enough to run per text node on the plan path. Accepts raw `htmlContent`
/// OR already-stripped plain text (plain text has no `<`, so the tag skip is
/// inert).
pub fn text_needs_complex_shaping(html: &str) -> bool {
    let mut in_tag = false;
    let mut in_quote: Option<char> = None;
    for (i, c) in html.char_indices() {
        match c {
            '<' if !in_tag => {
                in_tag = true;
                in_quote = None;
            }
            '>' if in_tag && in_quote.is_none() => in_tag = false,
            '"' | '\'' if in_tag => match in_quote {
                Some(q) if q == c => in_quote = None,
                Some(_) => {}
                None => in_quote = Some(c),
            },
            '&' if !in_tag => {
                if let Some(cp) = numeric_char_ref_at(&html[i..]) {
                    if codepoint_needs_complex_shaping(cp) {
                        return true;
                    }
                }
            }
            _ if !in_tag && codepoint_needs_complex_shaping(c as u32) => return true,
            _ => {}
        }
    }
    false
}

/// Decode a leading numeric character reference `&#NNN;` (decimal) or
/// `&#xHHH;` (hex) from `s` (which starts at the `&`). Returns the referenced
/// code point, or `None` if `s` is not a well-formed numeric ref. Allocation-free.
fn numeric_char_ref_at(s: &str) -> Option<u32> {
    let rest = s.strip_prefix("&#")?;
    let (radix, digits) = match rest.strip_prefix(['x', 'X']) {
        Some(hex) => (16u32, hex),
        None => (10u32, rest),
    };
    let is_digit = |c: char| {
        if radix == 16 {
            c.is_ascii_hexdigit()
        } else {
            c.is_ascii_digit()
        }
    };
    let end = digits.find(|c: char| !is_digit(c)).unwrap_or(digits.len());
    // At least one digit, and a terminating ';'.
    if end == 0 || digits.as_bytes().get(end) != Some(&b';') {
        return None;
    }
    u32::from_str_radix(&digits[..end], radix).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_rtl_and_brahmic_and_sea() {
        // RTL
        assert!(text_needs_complex_shaping("مرحبا")); // Arabic
        assert!(text_needs_complex_shaping("שלום")); // Hebrew
        assert!(text_needs_complex_shaping("ܫܠܡܐ")); // Syriac
        assert!(text_needs_complex_shaping("ދިވެހި")); // Thaana
                                                    // Brahmic
        assert!(text_needs_complex_shaping("नमस्ते")); // Devanagari
        assert!(text_needs_complex_shaping("বাংলা")); // Bengali
        assert!(text_needs_complex_shaping("வணக்கம்")); // Tamil
        assert!(text_needs_complex_shaping("నమస్కారం")); // Telugu
        assert!(text_needs_complex_shaping("ಕನ್ನಡ")); // Kannada
                                                     // SE Asian
        assert!(text_needs_complex_shaping("สวัสดี")); // Thai
        assert!(text_needs_complex_shaping("ຂໍໃຫ້")); // Lao
        assert!(text_needs_complex_shaping("ជំរាបសួរ")); // Khmer
        assert!(text_needs_complex_shaping("မင်္ဂလာ")); // Myanmar
        assert!(text_needs_complex_shaping("ᠮᠣᠩᠭᠣᠯ")); // Mongolian (cursive joining)
    }

    #[test]
    fn excludes_simple_scripts() {
        // The otjs lane renders these correctly — must NOT route to cosmic.
        assert!(!text_needs_complex_shaping("Hello, world 123!"));
        assert!(!text_needs_complex_shaping("Привет мир")); // Cyrillic
        assert!(!text_needs_complex_shaping("Γειά σου")); // Greek
        assert!(!text_needs_complex_shaping("你好世界")); // Han
        assert!(!text_needs_complex_shaping("こんにちは")); // Hiragana
        assert!(!text_needs_complex_shaping("안녕하세요")); // Hangul
        assert!(!text_needs_complex_shaping(" გამარჯობა")); // Georgian
    }

    #[test]
    fn detects_combining_marks_and_conjoining_jamo() {
        // Decomposed (NFD) text: the mark must attach to its base.
        assert!(text_needs_complex_shaping("a\u{308}")); // ä
        assert!(text_needs_complex_shaping("O\u{302}\u{300}")); // Vietnamese Ồ
        assert!(text_needs_complex_shaping("\u{438}\u{306}")); // Cyrillic й
        assert!(text_needs_complex_shaping("m\u{310}")); // no precomposed form
        assert!(text_needs_complex_shaping("\u{1112}\u{1161}\u{11AB}")); // 한 as jamo
        assert!(text_needs_complex_shaping("x\u{1DC4}y\u{FE20}")); // supplement, half marks
        // The same text precomposed (NFC) keeps the vector lane.
        assert!(!text_needs_complex_shaping("ä Ồ й 한"));
    }

    #[test]
    fn detects_complex_inside_html_and_mixed() {
        assert!(text_needs_complex_shaping(
            "<p style=\"font-family: Inter\">hello مرحبا world</p>"
        ));
        // Markup alone never trips it.
        assert!(!text_needs_complex_shaping(
            "<p style=\"font-family: Noto Sans Arabic\">plain latin</p>"
        ));
        // A complex-script code point INSIDE a tag/attribute (a non-ASCII
        // font-family name) must NOT route a Latin node to cosmic.
        assert!(!text_needs_complex_shaping(
            "<p style=\"font-family: 'نوتو'\">plain latin only</p>"
        ));
        // A `>` inside a quoted attribute value must not prematurely end the tag
        // skip and leak the attribute's complex-script chars as visible text.
        assert!(!text_needs_complex_shaping(
            "<span title=\"a > b\" style=\"font-family:'نوتو'\">plain latin</span>"
        ));
        // But real complex-script VISIBLE text after such an attribute is caught.
        assert!(text_needs_complex_shaping(
            "<span title=\"a > b\">مرحبا</span>"
        ));
    }

    #[test]
    fn detects_numeric_char_ref_complex_scripts() {
        // Entity-encoded complex text (imported / agent-authored) is decoded
        // downstream, so it must be caught here too.
        assert!(text_needs_complex_shaping("<p>&#1575;&#1604;</p>")); // Arabic (decimal)
        assert!(text_needs_complex_shaping("<p>&#x05D0;&#x05D1;</p>")); // Hebrew (hex)
        assert!(text_needs_complex_shaping("&#x0E01;")); // Thai (bare hex)
                                                         // Latin numeric refs must NOT trip it.
        assert!(!text_needs_complex_shaping(
            "<p>&#65;&#66;&#67; &amp; &lt;</p>"
        )); // ABC & <
        assert!(!text_needs_complex_shaping("<p>&#x41;&#x42;</p>")); // AB
                                                                     // Malformed refs are inert (no panic, no false positive).
        assert!(!text_needs_complex_shaping("<p>&# &#x &#;&#xZZ;</p>"));
    }
}
