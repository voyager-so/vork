//! Illustrator's Paragraph panel beyond alignment: indents, hyphenation, the composer and word spacing. Mirrored by
//! `utils/richtext/paragraphFormat.ts` (same properties, same parse, same canonical form), so the editor measures
//! what the engine draws.
//!
//! Stored as inherited CSS on the paragraph (the paragraph a run opens takes it, like `text-wrap`):
//!
//! - `text-indent: -12px`: the first line's indent (negative: a hanging indent), px or em.
//! - `--nf-indent-left: 24px`, `--nf-indent-right: 0px`: every line's indents, px or em.
//! - `--nf-hyphenate: auto [min-word N] [after-first N] [before-last N] [limit N] [no-capitalized]` or `none`:
//!   Illustrator's Hyphenation dialog (Words Longer Than 6, After First 2, Before Last 2, Hyphen Limit 2 (0: no
//!   limit), Hyphenate Capitalized Words on). `crate::hyphenate` finds the points.
//! - `--nf-lang: de`: the paragraph's language (a BCP 47 tag, as HTML's `lang`), which picks the hyphenation patterns
//!   ([`HyphLang`]); no language: US English.
//! - `--nf-composer: every-line | single-line`: Adobe's Every-line Composer (`crate::composer`, Knuth–Plass) or the
//!   Single-line Composer, the greedy wrap (`crate::wrap`, the default, as in Figma).
//! - `--nf-word-spacing: 80% 100% 133%`: Illustrator's Justification ▸ Word Spacing (minimum, desired, maximum), what
//!   the every-line composer lets a space shrink or stretch to.
//! - `word-break: auto | normal | keep-all | break-all` and `line-break: auto | loose | normal | strict | anywhere`:
//!   where Chinese, Japanese and Korean lines may break and how strict the kinsoku rules are ([`LineBreaking`],
//!   applied by `crate::vector::cjk::segment_text_for_wrap_with`).
//! - `text-spacing-trim: space-all | normal | space-first | trim-start | trim-both` and `text-autospace: no-autospace |
//!   normal | ideograph-alpha | ideograph-numeric`: fullwidth punctuation trimming and the space between CJK and Latin
//!   ([`CjkSpacing`], applied by `crate::cjk_spacing`).
//!
//! Every default leaves the paragraph exactly as before: no property, no change.

pub const TEXT_INDENT_PROPERTY: &str = "text-indent";
pub const INDENT_LEFT_PROPERTY: &str = "--nf-indent-left";
pub const INDENT_RIGHT_PROPERTY: &str = "--nf-indent-right";
pub const HYPHENATE_PROPERTY: &str = "--nf-hyphenate";
pub const COMPOSER_PROPERTY: &str = "--nf-composer";
pub const LANG_PROPERTY: &str = "--nf-lang";
pub const WORD_SPACING_PROPERTY: &str = "--nf-word-spacing";
pub const WORD_BREAK_PROPERTY: &str = "word-break";
pub const LINE_BREAK_PROPERTY: &str = "line-break";
pub const TEXT_SPACING_TRIM_PROPERTY: &str = "text-spacing-trim";
pub const TEXT_AUTOSPACE_PROPERTY: &str = "text-autospace";

/// The largest indent, px (Illustrator: 1296 pt).
pub const MAX_INDENT_PX: f64 = 1296.0;

/// Illustrator's Hyphenation options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hyphenation {
    /// Words Longer Than: a word needs at least this many letters (3–25).
    pub min_word: u32,
    /// After First: letters before a break (1–15).
    pub after_first: u32,
    /// Before Last: letters after a break (1–15).
    pub before_last: u32,
    /// Hyphen Limit: the most consecutive lines ending in a hyphen; 0 = no limit (0–25).
    pub limit: u32,
    /// Hyphenate Capitalized Words.
    pub capitalized: bool,
}

impl Hyphenation {
    pub const DEFAULT: Hyphenation = Hyphenation {
        min_word: 6,
        after_first: 2,
        before_last: 2,
        limit: 2,
        capitalized: true,
    };

    /// Whether another hyphenated line may follow `run` consecutive ones.
    pub fn allows(&self, run: u32) -> bool {
        self.limit == 0 || run < self.limit
    }
}

impl Default for Hyphenation {
    fn default() -> Self {
        Hyphenation::DEFAULT
    }
}

/// The hyphenation pattern sets (`crate::hyphenate`), picked by the paragraph's language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HyphLang {
    /// US English: no language, `en`, and every `en-*` but `en-GB`.
    #[default]
    EnUs,
    /// British English: `en-GB`.
    EnGb,
    /// German, the 1996 spelling: `de`, `de-*`.
    De,
    Fr,
    Es,
    It,
    Nl,
    Pt,
    Da,
    Pl,
    /// Norwegian (hyph-no): `nb`, `nn` and `no`.
    Nb,
    Et,
    Lt,
    /// A language with no patterns here: no pattern points.
    None,
}

impl HyphLang {
    /// The sets with patterns (not `None`), in `crate::hyphenate::PATTERNS_JSON`'s order.
    pub const COUNT: usize = 13;

    /// A language tag's patterns (`de`, `de-AT`, `en_GB`; any case); `None` (the Rust option) for a malformed tag.
    pub fn from_tag(tag: &str) -> Option<HyphLang> {
        let t = tag.trim().to_ascii_lowercase().replace('_', "-");
        let mut parts = t.split('-');
        let primary = parts.next()?;
        if !(2..=3).contains(&primary.len()) || !primary.chars().all(|c| c.is_ascii_lowercase()) {
            return None;
        }
        let rest: Vec<&str> = parts.collect();
        if rest
            .iter()
            .any(|p| p.is_empty() || p.len() > 8 || !p.chars().all(|c| c.is_ascii_alphanumeric()))
        {
            return None;
        }
        Some(match primary {
            "en" if rest.first() == Some(&"gb") => HyphLang::EnGb,
            "en" => HyphLang::EnUs,
            "de" => HyphLang::De,
            "fr" => HyphLang::Fr,
            "es" => HyphLang::Es,
            "it" => HyphLang::It,
            "nl" => HyphLang::Nl,
            "pt" => HyphLang::Pt,
            "da" => HyphLang::Da,
            "pl" => HyphLang::Pl,
            "nb" | "nn" | "no" => HyphLang::Nb,
            "et" => HyphLang::Et,
            "lt" => HyphLang::Lt,
            _ => HyphLang::None,
        })
    }
}

/// Adobe's two composers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Composer {
    /// The greedy wrap, one line at a time (Figma, and every existing document).
    #[default]
    SingleLine,
    /// Knuth–Plass total fit over the paragraph (`crate::composer`).
    EveryLine,
}

impl Composer {
    pub fn css(self) -> &'static str {
        match self {
            Composer::SingleLine => "single-line",
            Composer::EveryLine => "every-line",
        }
    }
}

/// Illustrator's Justification ▸ Word Spacing, percent of the font's space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WordSpacing {
    pub min: f64,
    pub desired: f64,
    pub max: f64,
}

impl WordSpacing {
    pub const DEFAULT: WordSpacing = WordSpacing {
        min: 80.0,
        desired: 100.0,
        max: 133.0,
    };
}

impl Default for WordSpacing {
    fn default() -> Self {
        WordSpacing::DEFAULT
    }
}

/// CSS `word-break`: where a line may break between letters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WordBreak {
    /// The default: Korean (Hangul) keeps its words whole and breaks at spaces, as `keep-all`; Chinese and Japanese
    /// break between characters, as `normal`.
    #[default]
    Auto,
    /// Breaks between any two CJK characters, Hangul syllables included; other scripts break at spaces.
    Normal,
    /// No break between letters of any script: CJK text breaks at spaces and after punctuation only.
    KeepAll,
    /// Breaks between any two letters of any script (Latin words too).
    BreakAll,
}

impl WordBreak {
    pub fn css(self) -> &'static str {
        match self {
            WordBreak::Auto => "auto",
            WordBreak::Normal => "normal",
            WordBreak::KeepAll => "keep-all",
            WordBreak::BreakAll => "break-all",
        }
    }

    /// `auto | normal | keep-all | break-all`; anything else (`break-word`, …) is unreadable.
    pub fn parse(value: &str) -> Option<WordBreak> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(WordBreak::Auto),
            "normal" => Some(WordBreak::Normal),
            "keep-all" => Some(WordBreak::KeepAll),
            "break-all" => Some(WordBreak::BreakAll),
            _ => None,
        }
    }
}

/// CSS `line-break`: how strict the kinsoku rules (characters that may not start or end a line) are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineBreak {
    /// Closing brackets and quotes, sentence punctuation, small kana, the prolonged sound mark, iteration marks,
    /// middle dots and `%` never start a line; opening brackets and quotes never end one. `auto` reads as this.
    #[default]
    Normal,
    /// As `normal`, and hyphens (`‐ – ゠`) and `‼ ⁇ ⁈ ⁉` never start a line either.
    Strict,
    /// Only closing brackets and quotes and sentence punctuation (`、。，．！？` and their ASCII forms) are kept off a
    /// line's start: small kana, `ー`, iteration marks, `・：；`, `…‥`, `〜` and `%` may start one.
    Loose,
    /// A line may break between any two characters, with no kinsoku rules.
    Anywhere,
}

impl LineBreak {
    pub fn css(self) -> &'static str {
        match self {
            LineBreak::Normal => "normal",
            LineBreak::Strict => "strict",
            LineBreak::Loose => "loose",
            LineBreak::Anywhere => "anywhere",
        }
    }

    /// `auto | loose | normal | strict | anywhere` (`auto` is `normal`).
    pub fn parse(value: &str) -> Option<LineBreak> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" | "normal" => Some(LineBreak::Normal),
            "strict" => Some(LineBreak::Strict),
            "loose" => Some(LineBreak::Loose),
            "anywhere" => Some(LineBreak::Anywhere),
            _ => None,
        }
    }
}

/// A paragraph's CJK line-breaking rules: `word-break` and `line-break`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LineBreaking {
    pub word_break: WordBreak,
    pub line_break: LineBreak,
}

impl LineBreaking {
    pub const DEFAULT: LineBreaking = LineBreaking {
        word_break: WordBreak::Auto,
        line_break: LineBreak::Normal,
    };
}

/// CSS `text-spacing-trim`: which fullwidth punctuation sets half-width (its blank half removed, `crate::cjk_spacing`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpacingTrim {
    /// The default (as in Figma): every fullwidth punctuation keeps its full width.
    #[default]
    SpaceAll,
    /// Adjacent pairs only: a closing mark before another closing mark or a middle dot, and an opening mark after
    /// an opening, closing or middle mark, lose the blank half between them (`」「`, `。」`, `（「`).
    Normal,
    /// As `normal`, and an opening mark that starts a line loses its blank left half, except on the paragraph's first
    /// line (and the first line after a hard break).
    SpaceFirst,
    /// As `space-first`, on every line.
    TrimStart,
    /// As `trim-start`, and a closing mark that ends a line loses its blank right half.
    TrimBoth,
}

impl SpacingTrim {
    pub fn css(self) -> &'static str {
        match self {
            SpacingTrim::SpaceAll => "space-all",
            SpacingTrim::Normal => "normal",
            SpacingTrim::SpaceFirst => "space-first",
            SpacingTrim::TrimStart => "trim-start",
            SpacingTrim::TrimBoth => "trim-both",
        }
    }

    /// `space-all | normal | space-first | trim-start | trim-both`; CSS's other values are unreadable here.
    pub fn parse(value: &str) -> Option<SpacingTrim> {
        match value.trim().to_ascii_lowercase().as_str() {
            "space-all" => Some(SpacingTrim::SpaceAll),
            "normal" => Some(SpacingTrim::Normal),
            "space-first" => Some(SpacingTrim::SpaceFirst),
            "trim-start" => Some(SpacingTrim::TrimStart),
            "trim-both" => Some(SpacingTrim::TrimBoth),
            _ => None,
        }
    }
}

/// CSS `text-autospace`: an eighth of an em between ideographs (Han and kana) and Latin letters and / or digits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Autospace {
    /// `ideograph-alpha`: between an ideograph and a letter of a space-separated script.
    pub alpha: bool,
    /// `ideograph-numeric`: between an ideograph and a (non-fullwidth) digit.
    pub numeric: bool,
}

impl Autospace {
    pub const NONE: Autospace = Autospace {
        alpha: false,
        numeric: false,
    };

    pub fn css(self) -> &'static str {
        match (self.alpha, self.numeric) {
            (false, false) => "no-autospace",
            (true, true) => "normal",
            (true, false) => "ideograph-alpha",
            (false, true) => "ideograph-numeric",
        }
    }

    /// `no-autospace`, `normal` (or `auto`: both), or `ideograph-alpha` and / or `ideograph-numeric`.
    pub fn parse(value: &str) -> Option<Autospace> {
        let v = value.trim().to_ascii_lowercase();
        match v.as_str() {
            "no-autospace" => return Some(Autospace::NONE),
            "normal" | "auto" => {
                return Some(Autospace {
                    alpha: true,
                    numeric: true,
                })
            }
            _ => {}
        }
        let mut out = Autospace::NONE;
        for token in v.split_whitespace() {
            match token {
                "ideograph-alpha" if !out.alpha => out.alpha = true,
                "ideograph-numeric" if !out.numeric => out.numeric = true,
                _ => return None,
            }
        }
        (out != Autospace::NONE).then_some(out)
    }
}

/// A paragraph's CJK spacing: `text-spacing-trim` and `text-autospace`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CjkSpacing {
    pub trim: SpacingTrim,
    pub autospace: Autospace,
}

impl CjkSpacing {
    pub const DEFAULT: CjkSpacing = CjkSpacing {
        trim: SpacingTrim::SpaceAll,
        autospace: Autospace::NONE,
    };

    pub fn is_default(&self) -> bool {
        *self == CjkSpacing::DEFAULT
    }
}

/// A paragraph's indents, hyphenation, composer and word spacing. The default changes nothing.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ParagraphFormat {
    /// px, every line.
    pub indent_left: f64,
    /// px, every line.
    pub indent_right: f64,
    /// px, added to the first line's left indent (negative: a hanging indent).
    pub indent_first: f64,
    /// `None`: no hyphenation (and a soft hyphen is dropped, as before).
    pub hyphenate: Option<Hyphenation>,
    pub composer: Composer,
    pub word_spacing: WordSpacing,
    /// `word-break` and `line-break`.
    pub breaking: LineBreaking,
    /// `text-spacing-trim` and `text-autospace`.
    pub spacing: CjkSpacing,
    /// The hyphenation patterns, from `--nf-lang`.
    pub lang: HyphLang,
}

impl ParagraphFormat {
    pub const DEFAULT: ParagraphFormat = ParagraphFormat {
        indent_left: 0.0,
        indent_right: 0.0,
        indent_first: 0.0,
        hyphenate: None,
        composer: Composer::SingleLine,
        word_spacing: WordSpacing::DEFAULT,
        breaking: LineBreaking::DEFAULT,
        spacing: CjkSpacing::DEFAULT,
        lang: HyphLang::EnUs,
    };

    pub fn is_default(&self) -> bool {
        *self == ParagraphFormat::DEFAULT
    }

    /// Whether any indent is set.
    pub fn has_indents(&self) -> bool {
        self.indent_left != 0.0 || self.indent_right != 0.0 || self.indent_first != 0.0
    }

    /// Line `first`'s left indent: the left indent, plus the first-line indent on the paragraph's first line.
    pub fn start_indent(&self, first: bool) -> f64 {
        if first {
            self.indent_left + self.indent_first
        } else {
            self.indent_left
        }
    }

    /// How much narrower a line is than its container.
    pub fn inset(&self, first: bool) -> f64 {
        self.start_indent(first) + self.indent_right
    }

    /// Apply one declaration; false when `prop` isn't one of these. An unreadable value leaves the format alone.
    pub fn apply_declaration(&mut self, prop: &str, value: &str, font_size_px: f64) -> bool {
        match prop {
            TEXT_INDENT_PROPERTY => {
                if let Some(px) = parse_length_px(value, font_size_px) {
                    self.indent_first = px;
                }
            }
            INDENT_LEFT_PROPERTY => {
                if let Some(px) = parse_length_px(value, font_size_px) {
                    self.indent_left = px;
                }
            }
            INDENT_RIGHT_PROPERTY => {
                if let Some(px) = parse_length_px(value, font_size_px) {
                    self.indent_right = px;
                }
            }
            HYPHENATE_PROPERTY => {
                if let Some(h) = parse_hyphenate(value) {
                    self.hyphenate = h;
                }
            }
            COMPOSER_PROPERTY => {
                if let Some(c) = parse_composer(value) {
                    self.composer = c;
                }
            }
            WORD_SPACING_PROPERTY => {
                if let Some(ws) = parse_word_spacing(value) {
                    self.word_spacing = ws;
                }
            }
            WORD_BREAK_PROPERTY => {
                if let Some(wb) = WordBreak::parse(value) {
                    self.breaking.word_break = wb;
                }
            }
            LINE_BREAK_PROPERTY => {
                if let Some(lb) = LineBreak::parse(value) {
                    self.breaking.line_break = lb;
                }
            }
            TEXT_SPACING_TRIM_PROPERTY => {
                if let Some(t) = SpacingTrim::parse(value) {
                    self.spacing.trim = t;
                }
            }
            TEXT_AUTOSPACE_PROPERTY => {
                if let Some(a) = Autospace::parse(value) {
                    self.spacing.autospace = a;
                }
            }
            LANG_PROPERTY => {
                if let Some(lang) = HyphLang::from_tag(value) {
                    self.lang = lang;
                }
            }
            _ => return false,
        }
        true
    }
}

/// A length in px: `12px`, `-12`, `0`, `1.5em` (of `font_size_px`), clamped to ±`MAX_INDENT_PX`. Other units: none.
pub fn parse_length_px(value: &str, font_size_px: f64) -> Option<f64> {
    let v = value.trim().to_ascii_lowercase();
    let (num, scale) = if let Some(n) = v.strip_suffix("px") {
        (n, 1.0)
    } else if let Some(n) = v.strip_suffix("em") {
        (n, font_size_px)
    } else {
        (v.as_str(), 1.0)
    };
    let n = num.trim();
    if n.is_empty()
        || !n
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.' || c == '-' || c == '+')
    {
        return None;
    }
    let x: f64 = n.parse().ok()?;
    let px = x * scale;
    if !px.is_finite() {
        return None;
    }
    Some(px.clamp(-MAX_INDENT_PX, MAX_INDENT_PX))
}

/// `every-line` / `single-line` (also Adobe's names `paragraph` / `single`).
pub fn parse_composer(value: &str) -> Option<Composer> {
    match value.trim().to_ascii_lowercase().as_str() {
        "every-line" | "paragraph" => Some(Composer::EveryLine),
        "single-line" | "single" => Some(Composer::SingleLine),
        _ => None,
    }
}

fn parse_count(token: Option<&str>, lo: u32, hi: u32) -> Option<u32> {
    let t = token?;
    if t.is_empty() || !t.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    // All digits: only an overflow fails, and that clamps like any big number.
    let n: u64 = t.parse().unwrap_or(u64::MAX);
    Some((n.min(hi as u64) as u32).max(lo))
}

/// `none` → `Some(None)`; `auto …` → `Some(Some(options))`; anything unreadable → `None`. Options: `min-word N`
/// (3–25), `after-first N`, `before-last N` (1–15), `limit N` (0–25, 0 = none), `capitalized` / `no-capitalized`.
pub fn parse_hyphenate(value: &str) -> Option<Option<Hyphenation>> {
    let v = value.trim().to_ascii_lowercase();
    let mut tokens = v.split_whitespace();
    match tokens.next()? {
        "none" | "manual" => return Some(None),
        "auto" => {}
        _ => return None,
    }
    let mut h = Hyphenation::DEFAULT;
    while let Some(t) = tokens.next() {
        match t {
            "min-word" => h.min_word = parse_count(tokens.next(), 3, 25)?,
            "after-first" => h.after_first = parse_count(tokens.next(), 1, 15)?,
            "before-last" => h.before_last = parse_count(tokens.next(), 1, 15)?,
            "limit" => h.limit = parse_count(tokens.next(), 0, 25)?,
            "capitalized" => h.capitalized = true,
            "no-capitalized" => h.capitalized = false,
            _ => return None,
        }
    }
    Some(Some(h))
}

/// The canonical `--nf-hyphenate` value: `auto` and the options that differ from the defaults, or `none`.
pub fn serialize_hyphenate(h: Option<Hyphenation>) -> String {
    let Some(h) = h else {
        return "none".to_string();
    };
    let d = Hyphenation::DEFAULT;
    let mut out = String::from("auto");
    if h.min_word != d.min_word {
        out.push_str(&format!(" min-word {}", h.min_word));
    }
    if h.after_first != d.after_first {
        out.push_str(&format!(" after-first {}", h.after_first));
    }
    if h.before_last != d.before_last {
        out.push_str(&format!(" before-last {}", h.before_last));
    }
    if h.limit != d.limit {
        out.push_str(&format!(" limit {}", h.limit));
    }
    if !h.capitalized {
        out.push_str(" no-capitalized");
    }
    out
}

/// `80% 100% 133%` (the `%` optional): minimum ≤ desired ≤ maximum, each 0–1000. One value sets all three.
pub fn parse_word_spacing(value: &str) -> Option<WordSpacing> {
    let nums: Vec<f64> = value
        .split_whitespace()
        .map(|t| {
            let n = t.strip_suffix('%').unwrap_or(t);
            if n.is_empty() || !n.chars().all(|c| c.is_ascii_digit() || c == '.') {
                return None;
            }
            n.parse::<f64>()
                .ok()
                .filter(|x| x.is_finite() && (0.0..=1000.0).contains(x))
        })
        .collect::<Option<Vec<f64>>>()?;
    let ws = match nums.as_slice() {
        [d] => WordSpacing {
            min: *d,
            desired: *d,
            max: *d,
        },
        [min, desired, max] => WordSpacing {
            min: *min,
            desired: *desired,
            max: *max,
        },
        _ => return None,
    };
    (ws.min <= ws.desired && ws.desired <= ws.max).then_some(ws)
}

/// The canonical `--nf-word-spacing` value.
pub fn serialize_word_spacing(ws: WordSpacing) -> String {
    format!("{}% {}% {}%", ws.min, ws.desired, ws.max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_serializes_hyphenation() {
        assert_eq!(parse_hyphenate("none"), Some(None));
        assert_eq!(parse_hyphenate("auto"), Some(Some(Hyphenation::DEFAULT)));
        let h =
            parse_hyphenate("auto min-word 5 after-first 3 before-last 3 limit 0 no-capitalized")
                .unwrap()
                .unwrap();
        assert_eq!(
            h,
            Hyphenation {
                min_word: 5,
                after_first: 3,
                before_last: 3,
                limit: 0,
                capitalized: false
            }
        );
        assert_eq!(
            serialize_hyphenate(Some(h)),
            "auto min-word 5 after-first 3 before-last 3 limit 0 no-capitalized"
        );
        assert_eq!(serialize_hyphenate(Some(Hyphenation::DEFAULT)), "auto");
        assert_eq!(serialize_hyphenate(None), "none");
        // Clamped to Illustrator's ranges; junk is unreadable.
        assert_eq!(
            parse_hyphenate("auto min-word 99")
                .unwrap()
                .unwrap()
                .min_word,
            25
        );
        assert_eq!(parse_hyphenate("auto min-word x"), None);
        assert_eq!(parse_hyphenate("sometimes"), None);
    }

    #[test]
    fn parses_word_spacing_and_lengths() {
        assert_eq!(
            parse_word_spacing("80% 100% 133%"),
            Some(WordSpacing::DEFAULT)
        );
        assert_eq!(parse_word_spacing("80 100 133"), Some(WordSpacing::DEFAULT));
        assert_eq!(
            parse_word_spacing("120% 100% 133%"),
            None,
            "min above desired"
        );
        assert_eq!(
            serialize_word_spacing(WordSpacing::DEFAULT),
            "80% 100% 133%"
        );
        assert_eq!(parse_length_px("-12px", 16.0), Some(-12.0));
        assert_eq!(parse_length_px("1.5em", 20.0), Some(30.0));
        assert_eq!(parse_length_px("0", 20.0), Some(0.0));
        assert_eq!(parse_length_px("10%", 20.0), None);
    }

    #[test]
    fn language_tags_pick_patterns() {
        assert_eq!(HyphLang::from_tag("de"), Some(HyphLang::De));
        assert_eq!(HyphLang::from_tag("de-AT"), Some(HyphLang::De));
        assert_eq!(HyphLang::from_tag("en_GB"), Some(HyphLang::EnGb));
        assert_eq!(HyphLang::from_tag("en-AU"), Some(HyphLang::EnUs));
        assert_eq!(HyphLang::from_tag("FR-ca"), Some(HyphLang::Fr));
        assert_eq!(HyphLang::from_tag("sv"), Some(HyphLang::None));
        assert_eq!(HyphLang::from_tag("et-EE"), Some(HyphLang::Et));
        assert_eq!(HyphLang::from_tag("nn"), Some(HyphLang::Nb));
        assert_eq!(HyphLang::from_tag("no"), Some(HyphLang::Nb));
        assert_eq!(HyphLang::from_tag("german"), None);
        assert_eq!(HyphLang::from_tag("de--x"), None);
        let mut f = ParagraphFormat::DEFAULT;
        assert!(f.apply_declaration(LANG_PROPERTY, "fr", 16.0));
        assert_eq!(f.lang, HyphLang::Fr);
        // Unreadable: the language stays.
        assert!(f.apply_declaration(LANG_PROPERTY, "not a tag", 16.0));
        assert_eq!(f.lang, HyphLang::Fr);
    }

    #[test]
    fn indents_per_line() {
        let f = ParagraphFormat {
            indent_left: 20.0,
            indent_right: 10.0,
            indent_first: -8.0,
            ..ParagraphFormat::DEFAULT
        };
        assert_eq!(f.start_indent(true), 12.0);
        assert_eq!(f.start_indent(false), 20.0);
        assert_eq!(f.inset(true), 22.0);
        assert_eq!(f.inset(false), 30.0);
    }
}
