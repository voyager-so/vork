//! Hyphenation: Liang's algorithm over TeX's hyphenation patterns (`hyph/<lang>.json`, from hyph-utf8's
//! `hyph-<lang>.tex`; each file keeps its source's notice). Mirrored by `utils/richtext/hyphenation.ts`, which reads
//! the same files, so both lanes break words at the same points.
//!
//! The patterns are the paragraph's language's (`--nf-lang`, [`HyphLang`]): US and British English, German (the 1996
//! spelling), French, Spanish, Italian, Dutch, Portuguese, Danish, Polish, Norwegian, Estonian and Lithuanian,
//! each under a permissive licence
//! (THIRD-PARTY-NOTICES.md). No language: US English. A language with no patterns here: no pattern points (soft and
//! hard hyphens still break).
//!
//! [`word_breaks`] answers "where may this word group break": a word group is what the wrap keeps together (the
//! text between spaces, `crate::wrap`), and
//!
//! - a soft hyphen (U+00AD) in it is a break that draws a hyphen when used; a word with soft hyphens breaks only
//!   there, as in TeX and InDesign;
//! - otherwise its core (the letters between leading and trailing punctuation) must be letter runs joined by hard
//!   hyphens (`self-contained`): each run of at least `min_word` letters takes the patterns' points at least
//!   `after_first` letters from its start and `before_last` from its end (capitalized runs only with `capitalized`),
//!   and each hard hyphen is a break after it that draws nothing more. Anything else (digits, apostrophes, URLs)
//!   doesn't break.
//!
//! Letters are ASCII and Latin-1 / Latin Extended-A / B letters (U+00C0–U+024F); a letter matches a pattern by its
//! lowercase form (one that lowercases to more than one char matches none).

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::paragraph_format::{HyphLang, Hyphenation};

/// The shared pattern files (also imported by `utils/richtext/hyphenation.ts`), in [`HyphLang::ALL`]'s order.
pub const PATTERNS_JSON: [&str; HyphLang::COUNT] = [
    include_str!("hyph/en-us.json"),
    include_str!("hyph/en-gb.json"),
    include_str!("hyph/de.json"),
    include_str!("hyph/fr.json"),
    include_str!("hyph/es.json"),
    include_str!("hyph/it.json"),
    include_str!("hyph/nl.json"),
    include_str!("hyph/pt.json"),
    include_str!("hyph/da.json"),
    include_str!("hyph/pl.json"),
    include_str!("hyph/nb.json"),
    include_str!("hyph/et.json"),
    include_str!("hyph/lt.json"),
];

/// The soft hyphen: a break opportunity that draws a hyphen only when used.
pub const SOFT_HYPHEN: char = '\u{00AD}';

struct Patterns {
    /// Letters (and `.` for a word edge) → the digit after each position (len + 1 values).
    table: HashMap<String, Vec<u8>>,
    /// Exceptions: the word → its break positions.
    exceptions: HashMap<String, Vec<usize>>,
    /// The longest pattern, in chars.
    max_len: usize,
}

fn load(json: &str) -> Patterns {
    let json: serde_json::Value = serde_json::from_str(json).expect("hyphenation patterns parse");
    let mut table = HashMap::new();
    let mut max_len = 0;
    for pat in json["patterns"].as_str().unwrap_or("").split_whitespace() {
        let mut letters = String::new();
        let mut digits: Vec<u8> = vec![0];
        for c in pat.chars() {
            if let Some(d) = c.to_digit(10) {
                *digits.last_mut().unwrap() = d as u8;
            } else {
                letters.push(c);
                digits.push(0);
            }
        }
        max_len = max_len.max(digits.len() - 1);
        table.insert(letters, digits);
    }
    let mut exceptions = HashMap::new();
    for ex in json["exceptions"].as_str().unwrap_or("").split_whitespace() {
        let mut word = String::new();
        let mut points = Vec::new();
        for c in ex.chars() {
            if c == '-' {
                points.push(word.chars().count());
            } else {
                word.push(c);
            }
        }
        exceptions.insert(word, points);
    }
    Patterns {
        table,
        exceptions,
        max_len,
    }
}

fn patterns(lang: HyphLang) -> &'static Patterns {
    static P: [OnceLock<Patterns>; HyphLang::COUNT] = [const { OnceLock::new() }; HyphLang::COUNT];
    let i = lang as usize;
    P[i].get_or_init(|| load(PATTERNS_JSON[i]))
}

/// A letter a word may hyphenate across: ASCII, or Latin-1 / Latin Extended-A / B (U+00C0–U+024F, not × ÷).
pub fn is_hyphen_letter(c: char) -> bool {
    c.is_ascii_alphabetic()
        || (('\u{00C0}'..='\u{024F}').contains(&c) && c != '\u{00D7}' && c != '\u{00F7}')
}

/// Uppercase for the capitalized-word rule (a letter that lowercases to something else).
fn is_upper(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_uppercase();
    }
    let mut lower = c.to_lowercase();
    !(lower.next() == Some(c) && lower.next().is_none())
}

/// A letter as the patterns name it: its lowercase form, or `#` (no pattern's) when that isn't one char.
fn pattern_letter(c: char) -> char {
    if c.is_ascii() {
        return c.to_ascii_lowercase();
    }
    let mut lower = c.to_lowercase();
    match (lower.next(), lower.next()) {
        (Some(l), None) => l,
        _ => '#',
    }
}

/// Liang's points for a run of letters in `lang`: the positions `p` (1..n) where the run may break before its `p`th
/// letter, before `after_first` / `before_last` limits.
fn liang_points(letters: &[char], lang: HyphLang) -> Vec<usize> {
    if lang == HyphLang::None {
        return Vec::new();
    }
    let pats = patterns(lang);
    let key: String = letters.iter().map(|&c| pattern_letter(c)).collect();
    if let Some(points) = pats.exceptions.get(&key) {
        return points.clone();
    }
    let word: Vec<char> = std::iter::once('.')
        .chain(key.chars())
        .chain(std::iter::once('.'))
        .collect();
    let n = word.len();
    let mut values = vec![0u8; n + 1];
    let mut sub = String::new();
    for i in 0..n {
        sub.clear();
        for j in i..n.min(i + pats.max_len) {
            sub.push(word[j]);
            if let Some(digits) = pats.table.get(&sub) {
                for (k, d) in digits.iter().enumerate() {
                    if *d > values[i + k] {
                        values[i + k] = *d;
                    }
                }
            }
        }
    }
    // values[i] sits before word[i]; letter p of the run is word[p + 1], so the break before it is values[p + 1].
    (1..letters.len())
        .filter(|&p| values[p + 1] % 2 == 1)
        .collect()
}

/// Where a word may break inside the run `letters` under `h` (positions before a letter, 1..n).
pub fn hyphenation_points(letters: &[char], h: &Hyphenation, lang: HyphLang) -> Vec<usize> {
    let n = letters.len();
    if n < h.min_word as usize || n < 2 {
        return Vec::new();
    }
    if !h.capitalized && letters.first().is_some_and(|c| is_upper(*c)) {
        return Vec::new();
    }
    let (lo, hi) = (
        h.after_first as usize,
        n.saturating_sub(h.before_last as usize),
    );
    liang_points(letters, lang)
        .into_iter()
        .filter(|&p| p >= lo && p <= hi)
        .collect()
}

/// A place a word group may break: after `offset` chars (of the text without soft hyphens). `hyphen`: the line
/// draws a hyphen there (a pattern point or a soft hyphen); a break after a hard hyphen draws nothing more.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WordBreak {
    pub offset: usize,
    pub hyphen: bool,
}

const LEADING_PUNCT: &[char] = &[
    '(', '[', '{', '"', '\'', '\u{201C}', '\u{2018}', '\u{00AB}', '\u{00BF}', '\u{00A1}',
];
const TRAILING_PUNCT: &[char] = &[
    ')', ']', '}', '"', '\'', '\u{201D}', '\u{2019}', '\u{00BB}', '.', ',', ';', ':', '!', '?',
    '\u{2026}',
];

/// A word group's text with its soft hyphens removed, and where it may break (in order). `text` is the group's
/// text as the wrap sees it (soft hyphens included); `lang` picks the patterns.
pub fn word_breaks(text: &str, h: &Hyphenation, lang: HyphLang) -> (String, Vec<WordBreak>) {
    if text.contains(SOFT_HYPHEN) {
        let mut clean = String::with_capacity(text.len());
        let mut n = 0usize;
        let mut breaks: Vec<WordBreak> = Vec::new();
        for c in text.chars() {
            if c == SOFT_HYPHEN {
                if n > 0 && breaks.last().map(|b| b.offset) != Some(n) {
                    breaks.push(WordBreak {
                        offset: n,
                        hyphen: true,
                    });
                }
            } else {
                clean.push(c);
                n += 1;
            }
        }
        // A soft hyphen at the end breaks nothing.
        breaks.retain(|b| b.offset < n);
        return (clean, breaks);
    }
    let chars: Vec<char> = text.chars().collect();
    let mut start = 0;
    while start < chars.len() && LEADING_PUNCT.contains(&chars[start]) {
        start += 1;
    }
    let mut end = chars.len();
    while end > start && TRAILING_PUNCT.contains(&chars[end - 1]) {
        end -= 1;
    }
    let core = &chars[start..end];
    // The core: letter runs joined by single hard hyphens.
    let valid = !core.is_empty()
        && core.iter().all(|&c| c == '-' || is_hyphen_letter(c))
        && core.first() != Some(&'-')
        && core.last() != Some(&'-')
        && !core.windows(2).any(|w| w[0] == '-' && w[1] == '-');
    let mut breaks = Vec::new();
    if valid {
        let mut run_start = 0;
        for i in 0..=core.len() {
            if i < core.len() && core[i] != '-' {
                continue;
            }
            for p in hyphenation_points(&core[run_start..i], h, lang) {
                breaks.push(WordBreak {
                    offset: start + run_start + p,
                    hyphen: true,
                });
            }
            if i < core.len() {
                // The break after a hard hyphen.
                breaks.push(WordBreak {
                    offset: start + i + 1,
                    hyphen: false,
                });
            }
            run_start = i + 1;
        }
    }
    (text.to_string(), breaks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hyphenate(word: &str) -> String {
        hyphenate_in(word, HyphLang::EnUs)
    }

    fn hyphenate_in(word: &str, lang: HyphLang) -> String {
        let (clean, breaks) = word_breaks(word, &Hyphenation::DEFAULT, lang);
        let chars: Vec<char> = clean.chars().collect();
        let mut out = String::new();
        let mut last = 0;
        for b in &breaks {
            out.extend(&chars[last..b.offset]);
            out.push(if b.hyphen { '-' } else { '|' });
            last = b.offset;
        }
        out.extend(&chars[last..]);
        out
    }

    #[test]
    fn liang_on_a_word_list() {
        // TeX's own answers for these words (plain TeX's \showhyphens with the same patterns and 2/2 limits).
        let cases = [
            ("hyphenation", "hy-phen-ation"),
            ("typesetting", "type-set-ting"),
            ("paragraph", "para-graph"),
            ("composition", "com-po-si-tion"),
            ("justification", "jus-ti-fi-ca-tion"),
            ("algorithm", "al-go-rithm"),
            ("computer", "com-put-er"),
            ("associate", "as-so-ciate"),
            ("table", "table"),
            ("project", "project"),
            ("line", "line"),
        ];
        for (word, want) in cases {
            assert_eq!(hyphenate(word), want, "{word}");
        }
    }

    #[test]
    fn limits_capitals_punctuation_and_soft_hyphens() {
        let h = Hyphenation {
            after_first: 3,
            before_last: 4,
            ..Hyphenation::DEFAULT
        };
        let (_, b) = word_breaks("hyphenation", &h, HyphLang::EnUs);
        assert_eq!(b.iter().map(|b| b.offset).collect::<Vec<_>>(), vec![6]); // hyphen-ation
        let (_, b) = word_breaks(
            "Paragraph",
            &Hyphenation {
                capitalized: false,
                ..Hyphenation::DEFAULT
            },
            HyphLang::EnUs,
        );
        assert!(b.is_empty());
        assert_eq!(hyphenate("Paragraph"), "Para-graph");
        assert_eq!(hyphenate("(paragraph),"), "(para-graph),");
        assert_eq!(hyphenate("self-contained"), "self-|con-tained");
        assert_eq!(hyphenate("paragraph's"), "paragraph's");
        assert_eq!(hyphenate("example.com"), "example.com");
        assert_eq!(hyphenate("abc123def"), "abc123def");
        // A word shorter than Words Longer Than.
        assert_eq!(hyphenate("rabbit"), "rab-bit");
        assert_eq!(hyphenate("tiger"), "tiger");
        // Soft hyphens: the only points, and dropped from the text.
        assert_eq!(hyphenate("ty\u{AD}pe\u{AD}set\u{AD}ting"), "ty-pe-set-ting");
        assert_eq!(hyphenate("\u{AD}word\u{AD}"), "word");
    }

    /// The words of `crates/moda-text/tests/fixtures/hyphenation-languages.json`, shared with the TS lane's test.
    pub(crate) fn language_cases() -> Vec<(HyphLang, String, String)> {
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/hyphenation-languages.json")
            ).unwrap())
                .unwrap();
        let mut out = Vec::new();
        for (tag, words) in json["cases"].as_object().unwrap() {
            let lang = HyphLang::from_tag(tag).unwrap();
            for (word, want) in words.as_object().unwrap() {
                out.push((lang, word.clone(), want.as_str().unwrap().to_string()));
            }
        }
        out
    }

    #[test]
    #[ignore = "needs unported Moda fixture tests/fixtures/hyphenation-languages.json"]
    fn patterns_per_language() {
        let cases = language_cases();
        assert!(cases.len() > 30);
        for (lang, word, want) in cases {
            assert_eq!(hyphenate_in(&word, lang), want, "{lang:?} {word}");
        }
        // A language with no patterns here: only soft and hard hyphens break.
        assert_eq!(hyphenate_in("hyphenation", HyphLang::None), "hyphenation");
        assert_eq!(
            hyphenate_in("self-contained", HyphLang::None),
            "self-|contained"
        );
        // German capitalizes nouns: Hyphenate Capitalized Words off leaves them whole.
        let no_caps = Hyphenation {
            capitalized: false,
            ..Hyphenation::DEFAULT
        };
        assert!(word_breaks("Silbentrennung", &no_caps, HyphLang::De)
            .1
            .is_empty());
        // A letter that lowercases to two chars (İ) matches no pattern, in either lane.
        assert_eq!(pattern_letter('\u{0130}'), '#');
    }
}
