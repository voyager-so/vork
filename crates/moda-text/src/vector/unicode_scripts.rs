//! Unicode script classification — port of `utils/msdf/unicodeScripts.ts`
//! (`classifyCodepoint` over the full `SCRIPT_RANGES` table, first match
//! wins). Phase 3's per-script fallback (`script_fallback.rs`) consumes this
//! where Phase 1a only needed the CJK subset.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnicodeScript {
    Cjk,
    Hiragana,
    Katakana,
    Korean,
    Cyrillic,
    Arabic,
    Devanagari,
    Thai,
    Hebrew,
    Greek,
    Bengali,
    Tamil,
    Gurmukhi,
    Gujarati,
    Oriya,
    Telugu,
    Kannada,
    Malayalam,
    Georgian,
    Armenian,
    Ethiopic,
    Tibetan,
    Myanmar,
    Khmer,
    Lao,
    Sinhala,
    MathSymbols,
    Symbols,
    Unknown,
}

/// `SCRIPT_RANGES` (unicodeScripts.ts:34-162), order preserved (first match
/// wins — e.g. 0x0d80-0x0dff hits 'sinhala' only because malayalam's range
/// ends at 0x0d7f, but the scan order is part of the contract).
const SCRIPT_RANGES: &[(u32, u32, UnicodeScript)] = &[
    // CJK Unified Ideographs (+ extensions, compatibility)
    (0x4e00, 0x9fff, UnicodeScript::Cjk),
    (0x3400, 0x4dbf, UnicodeScript::Cjk),
    (0x20000, 0x2a6df, UnicodeScript::Cjk),
    (0x2a700, 0x2b73f, UnicodeScript::Cjk),
    (0x2b740, 0x2b81f, UnicodeScript::Cjk),
    (0x2b820, 0x2ceaf, UnicodeScript::Cjk),
    (0xf900, 0xfaff, UnicodeScript::Cjk),
    (0x2f800, 0x2fa1f, UnicodeScript::Cjk),
    // CJK symbols, radicals, strokes
    (0x2e80, 0x2eff, UnicodeScript::Cjk),
    (0x2f00, 0x2fdf, UnicodeScript::Cjk),
    (0x31c0, 0x31ef, UnicodeScript::Cjk),
    (0x3200, 0x32ff, UnicodeScript::Cjk),
    (0x3300, 0x33ff, UnicodeScript::Cjk),
    (0xfe30, 0xfe4f, UnicodeScript::Cjk),
    // Japanese
    (0x3040, 0x309f, UnicodeScript::Hiragana),
    (0x30a0, 0x30ff, UnicodeScript::Katakana),
    (0x31f0, 0x31ff, UnicodeScript::Katakana),
    (0xff65, 0xff9f, UnicodeScript::Katakana),
    // Korean
    (0xac00, 0xd7af, UnicodeScript::Korean),
    (0x1100, 0x11ff, UnicodeScript::Korean),
    (0x3130, 0x318f, UnicodeScript::Korean),
    (0xa960, 0xa97f, UnicodeScript::Korean),
    (0xd7b0, 0xd7ff, UnicodeScript::Korean),
    // Cyrillic
    (0x0400, 0x04ff, UnicodeScript::Cyrillic),
    (0x0500, 0x052f, UnicodeScript::Cyrillic),
    (0x2de0, 0x2dff, UnicodeScript::Cyrillic),
    (0xa640, 0xa69f, UnicodeScript::Cyrillic),
    // Arabic
    (0x0600, 0x06ff, UnicodeScript::Arabic),
    (0x0750, 0x077f, UnicodeScript::Arabic),
    (0x08a0, 0x08ff, UnicodeScript::Arabic),
    (0xfb50, 0xfdff, UnicodeScript::Arabic),
    (0xfe70, 0xfeff, UnicodeScript::Arabic),
    // Devanagari
    (0x0900, 0x097f, UnicodeScript::Devanagari),
    (0xa8e0, 0xa8ff, UnicodeScript::Devanagari),
    // Thai
    (0x0e00, 0x0e7f, UnicodeScript::Thai),
    // Hebrew
    (0x0590, 0x05ff, UnicodeScript::Hebrew),
    (0xfb1d, 0xfb4f, UnicodeScript::Hebrew),
    // Greek
    (0x0370, 0x03ff, UnicodeScript::Greek),
    (0x1f00, 0x1fff, UnicodeScript::Greek),
    // Bengali
    (0x0980, 0x09ff, UnicodeScript::Bengali),
    // Tamil
    (0x0b80, 0x0bff, UnicodeScript::Tamil),
    // Other Indic scripts
    (0x0a00, 0x0a7f, UnicodeScript::Gurmukhi),
    (0x0a80, 0x0aff, UnicodeScript::Gujarati),
    (0x0b00, 0x0b7f, UnicodeScript::Oriya),
    (0x0c00, 0x0c7f, UnicodeScript::Telugu),
    (0x0c80, 0x0cff, UnicodeScript::Kannada),
    (0x0d00, 0x0d7f, UnicodeScript::Malayalam),
    // Georgian
    (0x10a0, 0x10ff, UnicodeScript::Georgian),
    (0x2d00, 0x2d2f, UnicodeScript::Georgian),
    // Armenian
    (0x0530, 0x058f, UnicodeScript::Armenian),
    // Ethiopic
    (0x1200, 0x137f, UnicodeScript::Ethiopic),
    (0x1380, 0x139f, UnicodeScript::Ethiopic),
    (0x2d80, 0x2ddf, UnicodeScript::Ethiopic),
    // Tibetan
    (0x0f00, 0x0fff, UnicodeScript::Tibetan),
    // Myanmar
    (0x1000, 0x109f, UnicodeScript::Myanmar),
    (0xaa60, 0xaa7f, UnicodeScript::Myanmar),
    // Khmer
    (0x1780, 0x17ff, UnicodeScript::Khmer),
    (0x19e0, 0x19ff, UnicodeScript::Khmer),
    // Lao
    (0x0e80, 0x0eff, UnicodeScript::Lao),
    // Sinhala
    (0x0d80, 0x0dff, UnicodeScript::Sinhala),
    // Symbols. Two classes, NOT one, because the fallback chain requires a
    // SINGLE family to cover an entire script-class piece (see
    // `resolve_script_fallback`), and no single Noto family covers the whole
    // symbol space. Measured cmap coverage split cleanly along this boundary:
    // 'Noto Sans Math' has Arrows 112/112, MathOps 256/256 but Dingbats 9/192,
    // while 'Noto Sans Symbols 2' has Dingbats 145/192, MiscSymArrows 219/256
    // but Arrows 13/112. Merging them would leave a mixed piece like "→ ➜" with
    // no covering family and blank the whole node. Keep the blocks partitioned
    // by which family covers them.
    //
    // U+2000–206F General Punctuation is DELIBERATELY excluded: the em-dash and
    // friends must keep classifying 'unknown' so they ride along with the
    // surrounding script run.
    // Blocks below are ordered by codepoint; measured 'Noto Sans Math' /
    // 'Noto Sans Symbols 2' cmap coverage is noted so the next reader can
    // re-derive the partition.
    //
    // U+2800–28FF Braille is likewise deliberately unclassified for now: 'Noto
    // Sans Symbols 2' covers it 256/256, but nothing in the corpus exercises
    // it, so it stays on the safe whole-node path.
    (0x2190, 0x21ff, UnicodeScript::MathSymbols), // Arrows                        — Math 112/112, Sym2 13/112
    (0x2200, 0x22ff, UnicodeScript::MathSymbols), // Mathematical Operators        — Math 256/256, Sym2 7/256
    (0x2300, 0x23ff, UnicodeScript::MathSymbols), // Miscellaneous Technical       — Math 141/256, Sym2 37/256
    (0x25a0, 0x25ff, UnicodeScript::Symbols), // Geometric Shapes              — Sym2 96/96,   Math 96/96
    (0x2600, 0x26ff, UnicodeScript::Symbols), // Miscellaneous Symbols         — Sym2 121/256, Math 41/256
    (0x2700, 0x27bf, UnicodeScript::Symbols), // Dingbats                      — Sym2 145/192, Math 9/192
    (0x27c0, 0x27ef, UnicodeScript::MathSymbols), // Misc Mathematical Symbols-A   — Math 48/48,   Sym2 0/48
    (0x27f0, 0x27ff, UnicodeScript::MathSymbols), // Supplemental Arrows-A         — Math 16/16,   Sym2 0/16
    (0x2900, 0x297f, UnicodeScript::MathSymbols), // Supplemental Arrows-B         — Math 128/128, Sym2 0/128
    (0x2980, 0x29ff, UnicodeScript::MathSymbols), // Misc Mathematical Symbols-B   — Math 128/128, Sym2 3/128
    (0x2a00, 0x2aff, UnicodeScript::MathSymbols), // Supplemental Mathematical Ops — Math 256/256, Sym2 0/256
    (0x2b00, 0x2bff, UnicodeScript::Symbols), // Misc Symbols and Arrows       — Sym2 219/256, Math 74/256
    // ASTRAL symbol blocks (Q-131 follow-up). Before these rows the table
    // stopped at 0x2BFF, so an astral pictograph demoted off the emoji lane by
    // the shipped-SVG gate (`emoji.rs` / `emoji.ts`) classified `Unknown`,
    // resolved an EMPTY fallback chain, and — when it was the only uncovered
    // codepoint — declined the whole node to the degraded tier instead of
    // reaching the per-codepoint retry. 'Noto Sans Symbols 2' actually covers
    // these blocks (measured below on the vendored face), so classifying them
    // routes the demoted glyph onto a face that draws it, per glyph, with the
    // node staying on the vector engine. Blocks the vendored faces mostly lack
    // (Alchemical 8/128, Supp Symbols & Pictographs 14/256) stay deliberately
    // unclassified on the safe whole-node path.
    (0x1f000, 0x1f02f, UnicodeScript::Symbols), // Mahjong Tiles              — Sym2 44/48,   Math 0/48
    (0x1f030, 0x1f09f, UnicodeScript::Symbols), // Domino Tiles               — Sym2 100/112, Math 0/112
    (0x1f0a0, 0x1f0ff, UnicodeScript::Symbols), // Playing Cards              — Sym2 82/96,   Math 0/96
    (0x1f650, 0x1f67f, UnicodeScript::Symbols), // Ornamental Dingbats        — Sym2 48/48,   Math 0/48
    (0x1f780, 0x1f7ff, UnicodeScript::Symbols), // Geometric Shapes Extended  — Sym2 102/128, Math 89/128
    (0x1f800, 0x1f8ff, UnicodeScript::Symbols), // Supplemental Arrows-C      — Sym2 150/256, Math 0/256
    (0x1fa00, 0x1fa6f, UnicodeScript::Symbols), // Chess Symbols              — Sym2 98/112,  Math 0/112
];

/// `classifyCodepoint` (unicodeScripts.ts:164-169).
pub fn classify_codepoint(cp: u32) -> UnicodeScript {
    if cp == 0x00b1 {
        return UnicodeScript::MathSymbols;
    }
    for (start, end, script) in SCRIPT_RANGES {
        if cp >= *start && cp <= *end {
            return *script;
        }
    }
    UnicodeScript::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plus_minus_uses_the_math_fallback_chain() {
        assert_eq!(classify_codepoint('±' as u32), UnicodeScript::MathSymbols);
    }
}
