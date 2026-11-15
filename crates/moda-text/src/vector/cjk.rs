//! CJK wrap segmentation — port of
//! `renderers/webgpu/text/vector/cjkLineBreaking.ts`. The coverage split +
//! font fallback that used to live here (`cjkFallback.ts`) moved to
//! `script_fallback.rs` when main generalized it per-script (Phase 3,
//! `scriptFallback.ts`); script classification is `unicode_scripts.rs`.

use crate::paragraph_format::{LineBreak, LineBreaking, WordBreak};

/// JS `/\s/` (the exact set RegExp matches): used by `scriptFallback.ts`,
/// `vectorLayout.normalizeTextForShaping` and `segmentTextForWrap`.
pub fn js_whitespace(ch: char) -> bool {
    matches!(
        ch,
        '\t' | '\n' | '\u{000B}' | '\u{000C}' | '\r' | ' ' | '\u{00A0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200A}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202F}'
                | '\u{205F}'
                | '\u{3000}'
                | '\u{FEFF}'
    )
}

/// Spaces a line never breaks at (UAX #14 GL): no-break space, figure space, narrow no-break space
/// (`isNoBreakSpace`). [`js_whitespace`] holds them, so without this they split a word like any space.
pub fn is_no_break_space(ch: char) -> bool {
    matches!(ch, '\u{00A0}' | '\u{2007}' | '\u{202F}')
}

/// `CJK_BREAKABLE_RANGES` (cjkLineBreaking.ts:17-33).
const CJK_BREAKABLE_RANGES: &[(u32, u32)] = &[
    (0x2e80, 0x2fdf),
    (0x3000, 0x30ff),
    (0x3130, 0x318f),
    (0x31c0, 0x31ff),
    (0x3200, 0x33ff),
    (0x3400, 0x4dbf),
    (0x4e00, 0x9fff),
    (0xac00, 0xd7af),
    (0xf900, 0xfaff),
    (0xfe30, 0xfe4f),
    (0xff00, 0xff9f),
    (0x20000, 0x3ffff),
];

/// `isCjkBreakableCodepoint` (cjkLineBreaking.ts:36-41).
pub fn is_cjk_breakable_codepoint(cp: u32) -> bool {
    CJK_BREAKABLE_RANGES
        .iter()
        .any(|(s, e)| cp >= *s && cp <= *e)
}

/// Closing brackets and quotes and sentence punctuation: never at a line's start under any `line-break` but
/// `anywhere` (UAX #14 CL, CP, IS, EX and the closing QU). `KINSOKU_ALWAYS_NO_START` in cjkLineBreaking.ts.
const KINSOKU_ALWAYS_NO_START: CharSet =
    CharSet::new(")]}）］｝〉》」』】〕〗〙〛｠｣”’»›〞〟、。，．,.:;!?！？｡､");
/// Kept off a line's start by `normal` and `strict` (UAX #14 NS, CJ, IN and PO): small kana, the prolonged sound
/// mark, iteration marks, middle dots and fullwidth colons, ellipses, wave dashes, `%`, and the halfwidth forms.
const KINSOKU_NORMAL_NO_START: CharSet = CharSet::new("ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿｧｨｩｪｫｬｭｮｯーｰ々ゝゞヽヾ〻・：；･‥…〜～％%ﾞﾟ");
/// Kept off a line's start by `strict` only: hyphens and doubled exclamation / question marks.
const KINSOKU_STRICT_NO_START: CharSet = CharSet::new("‐–゠‼⁇⁈⁉");
/// Opening brackets and quotes: never at a line's end (UAX #14 OP and the opening QU). `KINSOKU_NO_END` in
/// cjkLineBreaking.ts.
const KINSOKU_NO_END: CharSet = CharSet::new("([{（［｛〈《「『【〔〖〘〚｟｢“‘«‹〝");

/// A kinsoku set: its ASCII members as a bit mask, the rest scanned. Most text is ASCII, and scanning the UTF-8 lists
/// for every character was about 6% of line layout.
struct CharSet {
    ascii: u128,
    chars: &'static str,
}

impl CharSet {
    const fn new(chars: &'static str) -> Self {
        // UTF-8 bytes below 0x80 are exactly the ASCII characters of the string.
        let bytes = chars.as_bytes();
        let mut ascii = 0u128;
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] < 0x80 {
                ascii |= 1 << bytes[i];
            }
            i += 1;
        }
        CharSet { ascii, chars }
    }
}

fn in_set(set: &CharSet, ch: char) -> bool {
    if ch.is_ascii() {
        return (set.ascii >> (ch as u32)) & 1 == 1;
    }
    set.chars.chars().any(|c| c == ch)
}

/// Whether `ch` may not start a line under `line_break` (`isKinsokuNoStart`).
pub fn is_kinsoku_no_start(ch: char, line_break: LineBreak) -> bool {
    match line_break {
        LineBreak::Anywhere => false,
        LineBreak::Loose => in_set(&KINSOKU_ALWAYS_NO_START, ch),
        LineBreak::Normal => {
            in_set(&KINSOKU_ALWAYS_NO_START, ch) || in_set(&KINSOKU_NORMAL_NO_START, ch)
        }
        LineBreak::Strict => {
            in_set(&KINSOKU_ALWAYS_NO_START, ch)
                || in_set(&KINSOKU_NORMAL_NO_START, ch)
                || in_set(&KINSOKU_STRICT_NO_START, ch)
        }
    }
}

/// Whether `ch` may not end a line under `line_break` (`isKinsokuNoEnd`).
pub fn is_kinsoku_no_end(ch: char, line_break: LineBreak) -> bool {
    line_break != LineBreak::Anywhere && in_set(&KINSOKU_NO_END, ch)
}

/// Hangul syllables and compatibility jamo (`isHangulCodepoint`).
pub fn is_hangul_codepoint(cp: u32) -> bool {
    (0xac00..=0xd7af).contains(&cp) || (0x3130..=0x318f).contains(&cp)
}

/// The CJK characters that are letters rather than punctuation, for `keep-all` (`isCjkLetterCodepoint`): Han,
/// kana, Hangul, iteration marks and fullwidth letters and digits.
pub fn is_cjk_letter_codepoint(cp: u32) -> bool {
    matches!(cp,
        0x3005..=0x3007 | 0x303b | 0x3041..=0x309f | 0x30a1..=0x30fa | 0x30fc..=0x30ff | 0x31f0..=0x31ff
        | 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff | 0x20000..=0x3ffff
        | 0xff10..=0xff19 | 0xff21..=0xff3a | 0xff41..=0xff5a | 0xff66..=0xff9d)
        || is_hangul_codepoint(cp)
}

/// A character that belongs to the one before it and is never split from it (`isClusterContinuation`): a
/// combining mark (the dakuten U+3099 included), a zero-width joiner, a variation selector (marks too) or an emoji
/// skin-tone modifier. What follows a zero-width joiner is handled by the caller.
pub fn is_cluster_continuation(ch: char) -> bool {
    use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};
    ch == '\u{200D}'
        || ('\u{1F3FB}'..='\u{1F3FF}').contains(&ch)
        || ch.general_category_group() == GeneralCategoryGroup::Mark
}

fn is_regional_indicator(ch: char) -> bool {
    ('\u{1F1E6}'..='\u{1F1FF}').contains(&ch)
}

/// Dashes a line may break AFTER inside a word (UAX #14 HY / BA / B2): hyphen-minus, hyphen, en dash, em dash.
/// `isBreakAfterDash` in cjkLineBreaking.ts.
pub fn is_break_after_dash(ch: char) -> bool {
    matches!(ch, '-' | '\u{2010}' | '\u{2013}' | '\u{2014}')
}

/// [`segment_text_for_wrap_with`] under the default rules (`word-break: auto`, `line-break: normal`).
pub fn segment_text_for_wrap(text: &str) -> Vec<String> {
    segment_text_for_wrap_with(text, LineBreaking::DEFAULT)
}

/// The letter class of the last character, for whether the next letter joins it.
#[derive(PartialEq, Clone, Copy)]
enum Letter {
    /// Not a letter: whitespace, CJK punctuation, or the start.
    None,
    /// A character of a space-separated script (Latin, digits, their punctuation, emoji): chains into a word.
    Word,
    /// A Hangul syllable or jamo.
    Hangul,
    /// Han, kana, or a fullwidth letter or digit.
    Cjk,
}

/// `segmentTextForWrap` (cjkLineBreaking.ts): split a space-free word into wrap tokens, a break opportunity
/// between each two. Tokens concatenate back to `text`.
///
/// - Whitespace runs are tokens of their own; a no-break space ([`is_no_break_space`]) joins the tokens on both
///   sides (`line-break: anywhere` still breaks around it).
/// - CJK characters break between each other (`word-break: normal`); under `auto`, the default, Hangul keeps its
///   words whole (`keep-all`), so Korean breaks at spaces; `keep-all` keeps every script's letters together;
///   `break-all` breaks between any two letters, Latin too.
/// - Kinsoku (`line-break`): a character that may not start a line joins the token before it, and one that may
///   not end a line joins the token after it ([`is_kinsoku_no_start`], [`is_kinsoku_no_end`]). `anywhere` breaks
///   between any two characters.
/// - A combining mark, joiner or emoji modifier never starts a token; nor does the second of a flag's two regional
///   indicators.
/// - Latin words also break after a dash that sits between a letter or digit and a letter ("state-of-the-art",
///   "en–dash", "word—word"), as UAX #14 and every browser do; a dash that leads a word ("-5", "--flag") or
///   precedes a digit ("10-20") keeps the word whole.
/// - Spaceless SEA runs (Thai, Lao, Khmer, Myanmar) split at the dictionary's word boundaries
///   ([`crate::sea_words::segment_sea_words`]), as the complex-script tier that draws them wraps.
pub fn segment_text_for_wrap_with(text: &str, rules: LineBreaking) -> Vec<String> {
    let LineBreaking {
        word_break,
        line_break,
    } = rules;
    let mut tokens: Vec<String> = Vec::new();
    // Whether the last token is whitespace (`None`: no token yet).
    let mut last_ws: Option<bool> = None;
    let mut bind_next = false;
    let mut last_letter = Letter::None;
    // The previous two characters of the current word (a dash breaks only between word characters).
    let mut prev: Option<char> = None;
    let mut prev2: Option<char> = None;
    // Regional indicators in a row: an even count closes a flag.
    let mut ri_run = 0usize;
    // Consecutive Thai, Lao, Khmer or Myanmar characters, cut at the dictionary's word boundaries on flush.
    let mut sea_run = String::new();
    let flush_sea_run = |sea_run: &mut String,
                         tokens: &mut Vec<String>,
                         last_ws: &mut Option<bool>,
                         bind_next: &mut bool,
                         last_letter: &mut Letter| {
        if sea_run.is_empty() {
            return;
        }
        for (i, part) in crate::sea_words::segment_sea_words(sea_run, rules).into_iter().enumerate() {
            // Only the run's first token can join the one before (an opening bracket bound it); the dictionary's
            // boundaries inside the run always break.
            if i == 0 && *last_ws == Some(false) && *bind_next {
                tokens.last_mut().unwrap().push_str(&part);
            } else {
                tokens.push(part);
            }
        }
        sea_run.clear();
        *last_ws = Some(false);
        *bind_next = false;
        // A SEA run's edges are break opportunities.
        *last_letter = Letter::None;
    };

    for ch in text.chars() {
        let cp = ch as u32;
        if crate::sea_words::is_sea_script_codepoint(cp) {
            sea_run.push(ch);
            continue;
        }
        flush_sea_run(&mut sea_run, &mut tokens, &mut last_ws, &mut bind_next, &mut last_letter);
        if is_no_break_space(ch) {
            // A no-break space (French « 30 mg », a figure space between digits) holds both sides together.
            if last_ws == Some(false) && line_break != LineBreak::Anywhere {
                tokens.last_mut().unwrap().push(ch);
            } else {
                tokens.push(ch.to_string());
            }
            last_ws = Some(false);
            bind_next = line_break != LineBreak::Anywhere;
            last_letter = Letter::None;
            prev = None;
            prev2 = None;
            ri_run = 0;
            continue;
        }
        if js_whitespace(ch) {
            if last_ws == Some(true) {
                tokens.last_mut().unwrap().push(ch);
            } else {
                tokens.push(ch.to_string());
            }
            last_ws = Some(true);
            bind_next = false;
            last_letter = Letter::None;
            prev = None;
            prev2 = None;
            ri_run = 0;
            continue;
        }
        let ri = is_regional_indicator(ch);
        let continuation = last_ws == Some(false)
            && (is_cluster_continuation(ch) || prev == Some('\u{200D}') || (ri && ri_run % 2 == 1));
        ri_run = if ri { ri_run + 1 } else { 0 };
        if continuation {
            // Part of the character before: no break, and the classes stay the base character's.
            tokens.last_mut().unwrap().push(ch);
            prev2 = prev;
            prev = Some(ch);
            continue;
        }
        let cjk = is_cjk_breakable_codepoint(cp);
        let letter = if !cjk {
            Letter::Word
        } else if is_hangul_codepoint(cp) {
            Letter::Hangul
        } else if is_cjk_letter_codepoint(cp) {
            Letter::Cjk
        } else {
            Letter::None
        };
        let append = last_ws == Some(false)
            && line_break != LineBreak::Anywhere
            && (bind_next
                || is_kinsoku_no_start(ch, line_break)
                || letters_join(last_letter, letter, word_break)
                    && !(letter == Letter::Word
                        && last_letter == Letter::Word
                        && ch.is_alphabetic()
                        && prev.is_some_and(is_break_after_dash)
                        && prev2.is_some_and(|c| c.is_alphanumeric())));
        prev2 = prev;
        prev = Some(ch);
        if append {
            tokens.last_mut().unwrap().push(ch);
        } else {
            tokens.push(ch.to_string());
        }
        last_ws = Some(false);
        bind_next = is_kinsoku_no_end(ch, line_break);
        last_letter = letter;
    }
    flush_sea_run(&mut sea_run, &mut tokens, &mut last_ws, &mut bind_next, &mut last_letter);
    tokens
}

/// Whether a `cur` letter stays with the `last` character before it (no break between them) under `word_break`
/// (`lettersJoin`). `last` is never whitespace here.
fn letters_join(last: Letter, cur: Letter, word_break: WordBreak) -> bool {
    use Letter::*;
    match word_break {
        WordBreak::BreakAll => false,
        WordBreak::Normal => last == Word && cur == Word,
        // Hangul also stays with the CJK punctuation before it (a particle after a closing bracket: 「한국어」는).
        WordBreak::KeepAll => (last != None && cur != None) || cur == Hangul,
        WordBreak::Auto => matches!(
            (last, cur),
            (Word, Word) | (Hangul, Hangul) | (Hangul, Word) | (Word, Hangul) | (None, Hangul)
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::segment_text_for_wrap;
    use super::segment_text_for_wrap_with;
    use crate::paragraph_format::{LineBreak, LineBreaking};
    use super::{
        in_set, CharSet, KINSOKU_ALWAYS_NO_START, KINSOKU_NORMAL_NO_START, KINSOKU_NO_END,
        KINSOKU_STRICT_NO_START,
    };

    #[test]
    fn kinsoku_sets_hold_exactly_their_characters() {
        let sets: [&CharSet; 4] = [
            &KINSOKU_ALWAYS_NO_START,
            &KINSOKU_NORMAL_NO_START,
            &KINSOKU_STRICT_NO_START,
            &KINSOKU_NO_END,
        ];
        for set in sets {
            for ch in (0..=0x10ffffu32).filter_map(char::from_u32) {
                assert_eq!(
                    in_set(set, ch),
                    set.chars.contains(ch),
                    "{ch:?} in {:?}",
                    set.chars
                );
            }
        }
    }

    fn seg(s: &str) -> Vec<String> {
        segment_text_for_wrap(s)
    }

    #[test]
    fn latin_words_and_spaces_stay_whole() {
        assert_eq!(seg("the quick  fox"), ["the", " ", "quick", "  ", "fox"]);
    }

    #[test]
    fn a_dash_between_letters_is_a_break_opportunity() {
        assert_eq!(seg("state-of-the-art"), ["state-", "of-", "the-", "art"]);
        assert_eq!(seg("en–dash em—dash"), ["en–", "dash", " ", "em—", "dash"]);
        assert_eq!(seg("x86-64 COVID-19"), ["x86-64", " ", "COVID-19"]);
        assert_eq!(seg("-5 --flag 10-20"), ["-5", " ", "--flag", " ", "10-20"]);
        assert_eq!(seg("e-mail"), ["e-", "mail"]);
    }

    #[test]
    fn thai_and_other_spaceless_sea_runs_break_between_words() {
        // The vector lane's tokens (`text.metrics`, checks, a PDF's invisible words) break where the complex-script
        // tier draws the lines, never inside a word ("เว้นวร|รค").
        assert_eq!(seg("ไม่มีการเว้นวรรค"), ["ไม่มี", "การ", "เว้น", "วรรค"]);
        assert_eq!(seg("ผัดไทย 50 บาท"), ["ผัด", "ไทย", " ", "50", " ", "บาท"]);
        // An opening bracket binds the run's first word; a closing one joins its last.
        assert_eq!(seg("(ผัดไทย)"), ["(ผัด", "ไทย)"]);
        assert_eq!(seg("ខ្ញុំស្រលាញ់អ្នក"), ["ខ្ញុំ", "ស្រលាញ់", "អ្នក"]);
    }

    #[test]
    fn no_break_spaces_hold_their_neighbours() {
        // cjkLineBreaking.ts `isNoBreakSpace` (textBreakRules.test.ts has the same cases).
        assert_eq!(seg("30\u{202F}mg"), ["30\u{202F}mg"]);
        assert_eq!(seg("1\u{2007}000\u{00A0}€"), ["1\u{2007}000\u{00A0}€"]);
        assert_eq!(seg("«\u{202F}oui\u{202F}»"), ["«\u{202F}oui\u{202F}»"]);
        assert_eq!(seg("a b\u{202F}c"), ["a", " ", "b\u{202F}c"]);
        assert_eq!(seg("日本\u{202F}語"), ["日", "本\u{202F}語"]);
        let anywhere = LineBreaking {
            line_break: LineBreak::Anywhere,
            ..LineBreaking::default()
        };
        assert_eq!(segment_text_for_wrap_with("a\u{202F}b", anywhere), ["a", "\u{202F}", "b"]);
    }

    #[test]
    fn cjk_still_breaks_per_character_with_kinsoku() {
        assert_eq!(seg("日本語。"), ["日", "本", "語。"]);
    }
}
