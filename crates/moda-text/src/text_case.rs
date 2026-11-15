//! E6.3 text case: CSS `text-transform` on a run (`none | uppercase |
//! lowercase | capitalize`), mirrored by `utils/richtext/textCase.ts`.
//!
//! The transform is non-destructive: `htmlContent` keeps the typed text and
//! the parse (`parser::parse_rich_text_to_paragraphs`) rewrites each run's text
//! once, after the walk, so every lane downstream (wrap, shaping, drawing,
//! extraction for SVG/PDF) sees the same transformed characters. Small caps are
//! not a transform here: they are the OpenType `smcp`/`c2sc` features
//! (`font-feature-settings`), which the text-case control writes directly, and
//! which the vector lane synthesizes for a font without them
//! (`crate::small_caps`).
//!
//! Upper and lower case use the full Unicode mappings of `str::to_uppercase` /
//! `str::to_lowercase` (JS `toUpperCase`/`toLowerCase` agree: `ß` → `SS`,
//! final sigma in context). Capitalize upper-cases the first letter of each
//! word, like CSS: a letter starts a word unless the character before it (in
//! the same paragraph, across runs) is a letter, digit, mark or apostrophe, so
//! `don't` stays `Don't` and `e-mail` becomes `E-Mail`.
//!
//! Greek capitals carry no accents: upper case drops a Greek letter's tonos,
//! other accents and breathings and keeps its diaeresis (`Ελιά` → `ΕΛΙΑ`), and
//! a vowel whose accent kept it apart from a following ι or υ gives that letter
//! a diaeresis (`ρολόι` → `ΡΟΛΟΪ`). A lone `ή` ("or") keeps its tonos. These are
//! the rules of ICU's and Firefox's Greek upper-casing; `str::to_uppercase`
//! alone keeps every accent (`ΕΛΙΆ`).
//!
//! Turkish and Azerbaijani (a run whose language is `tr` or `az`, from `lang`,
//! `--nf-lang` or the document's locale) pair dotted and dotless i: upper case
//! maps `i` → `İ` and `ı` → `I`, lower case `I` → `ı` and `İ` → `i`, title case
//! a word's first `i` → `İ` (`izmir` → `İZMİR`, not `IZMIR`). These are CSS's
//! and ICU's language-sensitive mappings; `str::to_uppercase` alone gives `I`.

use unicode_normalization::UnicodeNormalization;

use crate::types::Paragraph;

/// A run's CSS `text-transform`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextTransform {
    #[default]
    None,
    Uppercase,
    Lowercase,
    Capitalize,
}

impl TextTransform {
    /// Parse a CSS `text-transform` value. Unknown values (`full-width`, …)
    /// read as `none`, like the TS twin.
    pub fn parse(value: &str) -> TextTransform {
        match value.trim().to_ascii_lowercase().as_str() {
            "uppercase" => TextTransform::Uppercase,
            "lowercase" => TextTransform::Lowercase,
            "capitalize" => TextTransform::Capitalize,
            _ => TextTransform::None,
        }
    }
}

/// Whether `c` continues a word for `capitalize` (see the module docs).
fn continues_word(c: char) -> bool {
    c.is_alphanumeric() || c == '\'' || c == '\u{2019}' || is_combining_mark(c)
}

/// Combining marks (Mn/Mc/Me) — enough for word continuity without a
/// general-category table: the combining blocks `capitalize` meets in practice.
pub(crate) fn is_combining_mark(c: char) -> bool {
    matches!(c as u32,
        0x0300..=0x036F | 0x0483..=0x0489 | 0x0591..=0x05BD | 0x0610..=0x061A
        | 0x064B..=0x065F | 0x0900..=0x0903 | 0x093A..=0x094F | 0x1AB0..=0x1AFF
        | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F)
}

/// Greek and Coptic or Greek Extended (with `is_alphabetic`: the letters whose
/// capitals drop their accents; `;` U+037E and the spacing accents pass through).
fn is_greek(c: char) -> bool {
    matches!(c as u32, 0x0370..=0x03FF | 0x1F00..=0x1FFF)
}

/// The accents and breathings a Greek capital drops (the diaeresis U+0308 stays).
fn is_greek_accent(c: char) -> bool {
    matches!(
        c as u32,
        0x0300 | 0x0301 | 0x0342 | 0x0313 | 0x0314 | 0x0343
    )
}

/// Upper case with Greek capitals unaccented (see the module docs). `text` is
/// already upper-cased; each Greek letter with the marks after it is one
/// cluster, rewritten and recomposed. Everything else passes through as it is.
fn greek_upper(text: &str, source: &crate::types::SourceIndices) -> (String, crate::types::SourceIndices) {
    if !text.chars().any(is_greek) {
        return (text.to_string(), source.clone());
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut indices = Vec::new();
    // The previous cluster was a vowel that lost an accent (a following Ι or Υ is
    // then not part of a diphthong with it).
    let mut accented_vowel = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if !is_greek(c) || !c.is_alphabetic() {
            out.push(c);
            indices.push(source.get(i));
            accented_vowel = false;
            i += 1;
            continue;
        }
        let mut end = i + 1;
        while end < chars.len() && is_combining_mark(chars[end]) {
            end += 1;
        }
        let cluster: Vec<char> = chars[i..end]
            .iter()
            .copied()
            .collect::<String>()
            .nfd()
            .collect();
        let (base, marks) = (cluster[0], &cluster[1..]);
        let had_accent = marks.iter().any(|&m| is_greek_accent(m) || m == '\u{0344}');
        let mut diaeresis = marks.iter().any(|&m| m == '\u{0308}' || m == '\u{0344}');
        let lone =
            |at: Option<&char>| !at.is_some_and(|&n| n.is_alphabetic() || is_combining_mark(n));
        let keep = base == 'Η'
            && had_accent
            && lone(i.checked_sub(1).map(|p| &chars[p]))
            && lone(chars.get(end));
        if !diaeresis && accented_vowel && (base == 'Ι' || base == 'Υ') && !had_accent {
            diaeresis = true;
        }
        let mut rebuilt = String::new();
        rebuilt.push(base);
        for &m in marks {
            if keep || !(is_greek_accent(m) || m == '\u{0308}' || m == '\u{0344}') {
                rebuilt.push(m);
            }
        }
        if diaeresis && !keep {
            rebuilt.push('\u{0308}');
        }
        let rebuilt: String = rebuilt.nfc().collect();
        indices.extend(std::iter::repeat_n(source.get(i), rebuilt.chars().count()));
        out.push_str(&rebuilt);
        accented_vowel = had_accent && !keep && matches!(base, 'Α' | 'Ε' | 'Η' | 'Ο' | 'Υ' | 'Ω');
        i = end;
    }
    (out, crate::types::SourceIndices::new(indices))
}

/// Whether a canonical language tag is Turkish or Azerbaijani, whose i has a
/// dotted and a dotless case pair.
fn turkic(lang: Option<&str>) -> bool {
    matches!(lang.and_then(|l| l.split('-').next()), Some("tr" | "az"))
}

/// Apply one transform to a run's text (`prev` = the paragraph's character
/// before the run, for `capitalize`; `lang` = the run's canonical language).
pub fn transform_text(
    text: &str,
    transform: TextTransform,
    prev: Option<char>,
    lang: Option<&str>,
) -> String {
    transform_with_source(text, &Default::default(), transform, prev, lang).0
}

fn transform_with_source(
    text: &str, source: &crate::types::SourceIndices, transform: TextTransform,
    mut prev: Option<char>, lang: Option<&str>,
) -> (String, crate::types::SourceIndices) {
    use crate::types::SourceIndices;
    let dotted = turkic(lang);
    if transform == TextTransform::None { return (text.to_string(), source.clone()); }
    let mut out = String::with_capacity(text.len());
    let mut indices = Vec::new();
    let mut chars = text.chars().enumerate().peekable();
    while let Some((i, c)) = chars.next() {
        let mapped: String = match transform {
            TextTransform::Uppercase if dotted && c == 'i' => "İ".into(),
            TextTransform::Uppercase => c.to_uppercase().collect(),
            TextTransform::Lowercase if dotted && c == 'İ' => "i".into(),
            TextTransform::Lowercase if dotted && c == 'I' => {
                if chars.peek().is_some_and(|(_, c)| *c == '\u{307}') {
                    chars.next(); "i".into()
                } else { "ı".into() }
            },
            TextTransform::Lowercase => c.to_lowercase().collect(),
            TextTransform::Capitalize if !prev.is_some_and(continues_word) && dotted && c == 'i' => "İ".into(),
            TextTransform::Capitalize if !prev.is_some_and(continues_word) && c.is_alphabetic() => c.to_uppercase().collect(),
            _ => c.to_string(),
        };
        indices.extend(std::iter::repeat_n(source.get(i), mapped.chars().count()));
        out.push_str(&mapped);
        prev = Some(c);
    }
    let indices = SourceIndices::new(indices);
    if transform == TextTransform::Uppercase { return greek_upper(&out, &indices); }
    if transform == TextTransform::Lowercase {
        // Rust's whole-string lowercasing supplies contextual final sigma. Its contextual
        // choice is one character; expansions and the Turkic contraction above own their maps.
        let original = if dotted { text.replace('İ', "i").replace("I\u{307}", "i").replace('I', "ı") } else { text.to_string() };
        out = original.to_lowercase();
    }
    (out, indices)
}

/// Rewrite every run's text by its style's transform. Word context for
/// `capitalize` runs across the runs of a paragraph and uses the ORIGINAL
/// characters, so a transform on one run never changes its neighbour's.
pub fn apply_text_transforms(paragraphs: &mut [Paragraph]) {
    for para in paragraphs {
        let mut prev: Option<char> = None;
        for run in &mut para.runs {
            let last = run.text.chars().last();
            if run.style.text_transform != TextTransform::None {
                let (text, source) = transform_with_source(
                    &run.text, &run.style.source, run.style.text_transform, prev, run.style.lang.as_deref());
                run.text = text;
                run.style.source = source;
            }
            if last.is_some() {
                prev = last;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::parse_rich_text_to_paragraphs;

    fn texts(html: &str) -> Vec<String> {
        parse_rich_text_to_paragraphs(html)
            .into_iter()
            .map(|p| p.runs.into_iter().map(|r| r.text).collect::<String>())
            .collect()
    }

    #[test]
    fn upper_lower_and_none() {
        assert_eq!(
            texts(r#"<p style="text-transform: uppercase">Hello wörld straße</p>"#),
            vec!["HELLO WÖRLD STRASSE"]
        );
        assert_eq!(
            texts(r#"<p style="text-transform: lowercase">HeLLo ΣΟΦΟΣ</p>"#),
            vec!["hello σοφος"]
        );
        assert_eq!(
            texts(r#"<p style="text-transform: none">MiXed</p>"#),
            vec!["MiXed"]
        );
    }

    #[test]
    fn greek_capitals_drop_accents_and_keep_diaeresis() {
        let up = |t: &str| super::transform_text(t, super::TextTransform::Uppercase, None, None);
        assert_eq!(up("Ελαιόλαδο Σητείας"), "ΕΛΑΙΟΛΑΔΟ ΣΗΤΕΙΑΣ");
        assert_eq!(up("Ελιά, Ηράκλειο"), "ΕΛΙΑ, ΗΡΑΚΛΕΙΟ");
        // A kept diaeresis, and ΐ / ΰ lose only their tonos.
        assert_eq!(up("Ϊ ϊ ΐ ΰ προϊόν"), "Ϊ Ϊ Ϊ Ϋ ΠΡΟΪΟΝ");
        // An accented vowel before ι / υ: not a diphthong, so the ι / υ gets a diaeresis.
        assert_eq!(up("ρολόι κέικ"), "ΡΟΛΟΪ ΚΕΪΚ");
        // A diphthong accented on its second letter stays as it is.
        assert_eq!(up("Αίγυπτος ευρώ"), "ΑΙΓΥΠΤΟΣ ΕΥΡΩ");
        // Polytonic breathings and accents go too.
        assert_eq!(up("ἀρχὴ ὕδωρ"), "ΑΡΧΗ ΥΔΩΡ");
        // The lone ή ("or") keeps its tonos; ή inside a word doesn't.
        assert_eq!(up("άσπρο ή μαύρο, ήλιος"), "ΑΣΠΡΟ Ή ΜΑΥΡΟ, ΗΛΙΟΣ");
        // Accents on other scripts stay.
        assert_eq!(up("café Ελιά"), "CAFÉ ΕΛΙΑ");
        // Greek punctuation passes through untouched (U+037E is not rewritten to `;`).
        assert_eq!(up("τι;\u{037E} ά·"), "ΤΙ;\u{037E} Α·");
    }

    #[test]
    fn turkish_and_azerbaijani_pair_dotted_and_dotless_i() {
        assert_eq!(
            texts(r#"<p lang="tr" style="text-transform: uppercase">sıcacık izmir</p>"#),
            vec!["SICACIK İZMİR"]
        );
        assert_eq!(
            texts(r#"<p style="--nf-lang: az-Latn; text-transform: lowercase">İSTİQLAL IŞIQ</p>"#),
            vec!["istiqlal ışıq"]
        );
        assert_eq!(
            texts(r#"<p lang="tr-TR" style="text-transform: capitalize">ilk ısık</p>"#),
            vec!["İlk Isık"]
        );
        // The document's locale stands in for text that sets no language.
        assert_eq!(
            texts(&crate::lang::html_with_document_lang(
                r#"<p style="text-transform: uppercase">izmir</p>"#,
                Some("tr")
            )),
            vec!["İZMİR"]
        );
        // Other languages keep the default mapping.
        assert_eq!(
            texts(r#"<p lang="en" style="text-transform: uppercase">izmir</p>"#),
            vec!["IZMIR"]
        );
    }

    #[test]
    fn capitalize_follows_css_word_starts() {
        assert_eq!(
            texts(r#"<p style="text-transform: capitalize">don't e-mail the 3rd état</p>"#),
            vec!["Don't E-Mail The 3rd État"]
        );
    }

    #[test]
    fn capitalize_word_context_spans_runs() {
        // "hel" + "lo" is one word: the second run must not capitalize its `l`.
        assert_eq!(
            texts(
                r#"<p style="text-transform: capitalize">hel<span style="color: red">lo world</span></p>"#
            ),
            vec!["Hello World"]
        );
    }

    #[test]
    fn transform_inherits_and_a_span_can_reset_it() {
        assert_eq!(
            texts(
                r#"<p style="text-transform: uppercase">shout <span style="text-transform: none">quiet</span> shout</p>"#
            ),
            vec!["SHOUT quiet SHOUT"]
        );
    }
}
