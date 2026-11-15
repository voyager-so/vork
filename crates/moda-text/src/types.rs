//! Shared rich-text types — line-for-line port of `utils/richtext/types.ts`.
//!
//! `InlineStyle` keeps the TS field shapes deliberately: `font_weight` and
//! `font_style` stay strings because the TS renderer passes raw CSS values
//! through and only interprets them at consumption points
//! (`utils/richtext/types.ts:6-7`); copying the types copies the bugs.

/// Character provenance carried through shaping and wrapping. This is metadata, not a visual
/// style property: comparing styles deliberately ignores it so layout coalescing stays unchanged.
#[derive(Debug, Clone, Default)]
pub struct SourceIndices(std::sync::Arc<Vec<Option<usize>>>, std::ops::Range<usize>);
impl PartialEq for SourceIndices {
    fn eq(&self, _: &Self) -> bool { true }
}
impl SourceIndices {
    pub fn new(indices: Vec<Option<usize>>) -> Self {
        let len = indices.len();
        Self(std::sync::Arc::new(indices), 0..len)
    }
    pub fn get(&self, index: usize) -> Option<usize> {
        (index < self.1.len()).then(|| self.0[self.1.start + index]).flatten()
    }
    pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
        let start = range.start.min(self.1.len());
        let end = range.end.min(self.1.len()).max(start);
        Self(self.0.clone(), self.1.start + start..self.1.start + end)
    }
    pub fn values(&self) -> Vec<Option<usize>> { self.0[self.1.clone()].to_vec() }
    pub fn append(&mut self, other: &Self) {
        if self.1.start != 0 || self.1.end != self.0.len() {
            *self = Self::new(self.values());
        }
        std::sync::Arc::make_mut(&mut self.0).extend_from_slice(&other.0[other.1.clone()]);
        self.1.end = self.0.len();
    }
    pub fn filter(&self, text: &str, keep: impl Fn(char) -> bool) -> Self {
        Self::new(text.chars().enumerate().filter_map(|(i,c)| keep(c).then(|| self.get(i))).collect())
    }
}

/// The rich-text style naming a range's paint: `--nf-paint: <index>` ([`InlineStyle::paint`]). A host that supplies
/// colours with the draw writes it in place of `color`, so text that differs only in its colours is one layout.
pub const PAINT_PROPERTY: &str = "--nf-paint";

/// `InlineStyle` (`utils/richtext/types.ts:3-12`).
#[derive(Debug, Clone, PartialEq)]
pub struct InlineStyle {
    pub source: SourceIndices,
    pub font_family: String,
    /// px
    pub font_size: f64,
    /// numeric string or 'normal'/'bold' (raw CSS passthrough possible)
    pub font_weight: String,
    /// 'normal' | 'italic' | raw CSS passthrough (e.g. 'oblique')
    pub font_style: String,
    /// canvas-compatible color string — irrelevant to layout, kept for parity
    pub color: String,
    /// px
    pub letter_spacing: f64,
    /// multiple (e.g. 1.2). With `line_height_auto`, the fallback a lane without the font's metrics uses.
    pub line_height: f64,
    /// Figma's Auto line height (CSS `line-height: normal`): the font's own line spacing, ascender − descender +
    /// line gap ([`crate::measure::TextMeasurer::auto_line_height_px`]), rounded per line like any other. Inherits
    /// like `line_height`; an explicit multiple or px clears it.
    pub line_height_auto: bool,
    /// None ⇔ TS `undefined`; values: 'underline' | 'line-through' |
    /// 'underline line-through' | 'none'
    pub text_decoration: Option<String>,
    /// CSS `text-decoration-style` / `-color` / `-thickness` and `text-underline-offset`
    /// (`crate::decorations::DecorationStyle`), inherited like `text_decoration`. The default draws what every
    /// decoration drew before.
    pub decoration: crate::decorations::DecorationStyle,
    /// F7.3: CSS `font-feature-settings`, parsed (`crate::font_features`).
    /// Empty ⇔ `normal` — the font's default features, the pre-F7.3 shaping.
    pub font_features: Vec<crate::font_features::FontFeature>,
    /// E6.3: CSS `text-transform` (text case). Applied to the run text once, at
    /// the end of the HTML parse (`crate::text_case`), so every lane downstream
    /// lays out, draws and exports the transformed text.
    pub text_transform: crate::text_case::TextTransform,
    /// E6.3: CSS `text-wrap` (inherited like CSS); the paragraph a run opens
    /// takes it (`Paragraph::wrap`).
    pub text_wrap: crate::text_wrap::TextWrap,
    /// Tab stops (`--nf-tab-stops`, `crate::tab_stops`), inherited; the paragraph a run opens takes them
    /// (`Paragraph::tab_stops`).
    pub tab_stops: Option<std::sync::Arc<crate::tab_stops::TabStops>>,
    /// Indents, hyphenation, the composer and word spacing (`crate::paragraph_format`), inherited; the paragraph a
    /// run opens takes them (`Paragraph::format`).
    pub paragraph: crate::paragraph_format::ParagraphFormat,
    /// E6.3: CSS `font-variation-settings`, parsed
    /// (`crate::font_variations`). Empty ⇔ `normal`: the catalog variant's own
    /// instance.
    pub font_variations: Vec<crate::vector::varfont::Variation>,
    /// A gradient on this range (`--nf-text-fill`, `crate::run_fill`), over the
    /// node's box. `None` = the solid `color`.
    pub fill: Option<crate::ops::GradientSpec>,
    /// Which of the draw's paints colours this range (`--nf-paint`, [`PAINT_PROPERTY`]): an index into
    /// `TextNodeSpec::run_colors`. Everything drawn in the range's colour carries it: its glyphs, a hyphen the wrap
    /// adds, its decorations and its list marker. `None` = the solid `color`.
    pub paint: Option<u32>,
    /// Synthesized small caps (`crate::small_caps`): on a run whose glyphs were
    /// scaled down because the font has no `smcp`/`c2sc`, the run's full font
    /// size. `font_size` is then the reduced glyph size (shaping, advances,
    /// drawing); line boxes, baselines and decorations use this one
    /// ([`InlineStyle::line_font_size`]). `None` on every other run.
    pub small_caps_full_size: Option<f64>,
    /// E8.22: baseline shift, horizontal / vertical scale, character rotation and the kerning method
    /// (`--nf-*` properties, `crate::character`). The defaults leave every path untouched.
    pub character: crate::character::CharacterStyle,
    /// The run's language (`--nf-lang`, or an element's `lang` attribute; `crate::lang`), canonical, inherited.
    /// `None`: none declared (the document's locale, folded in as a wrapper by the caller, or nothing).
    pub lang: Option<String>,
    /// Vertical text: the writing mode, text orientation and tate-chu-yoko (`crate::vertical`), and in a vertical
    /// node how this piece draws. The default leaves every path untouched.
    pub vertical: crate::vertical::VerticalStyle,
    /// Ruby (furigana) over this run (`crate::ruby`). The default (none) leaves every path untouched.
    pub ruby: crate::ruby::RubyStyle,
}

impl InlineStyle {
    pub fn source_slice(&self, range: std::ops::Range<usize>) -> Self {
        let mut style = self.clone();
        style.source = self.source.slice(range);
        style
    }
    pub fn without_source(&self) -> Self {
        let mut style = self.clone();
        style.source = Default::default();
        style
    }

    /// Whether the wrap must keep this run's text on one line: a combined vertical piece, or a ruby base.
    pub fn unbreakable(&self) -> bool {
        self.vertical.unbreakable() || self.ruby.text.is_some()
    }

    /// The size line metrics use: the full size for a synthesized small-caps
    /// run, else `font_size`.
    pub fn line_font_size(&self) -> f64 {
        self.small_caps_full_size.unwrap_or(self.font_size)
    }

    /// The line box this style asks for, before per-line rounding: the font's own line spacing for Auto (when
    /// `auto_px` has the font's metrics), else the size times the multiple.
    pub fn line_box_px(&self, auto_px: impl FnOnce(&InlineStyle) -> Option<f64>) -> f64 {
        if self.line_height_auto {
            if let Some(px) = auto_px(self).filter(|px| px.is_finite() && *px > 0.0) {
                return px;
            }
        }
        self.line_font_size() * self.line_height
    }
}

/// Internal line-box convention, independent of shaping and baseline placement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LineHeightPolicy {
    /// Moda's Figma-compatible box text rounds each line to a whole pixel.
    #[default]
    Figma,
    /// CSS line-height retains fractional CSS pixels on every line.
    Css,
}
impl LineHeightPolicy {
    pub fn resolve(self, px: f64) -> f64 {
        match self {
            Self::Figma => round_line_height_px(px),
            Self::Css => px,
        }
    }
}

/// A box-text line's height in whole pixels, as Figma rounds each line (measured: Inter 16 at 1.2 is 19 px a line,
/// 57 for three; 150% of 13 is 20 a line, 60 for three). `roundLineHeightPx` (`utils/richtext/types.ts`) is the twin:
/// JS `Math.round` and `f64::round` agree on positive halves.
pub fn round_line_height_px(px: f64) -> f64 {
    if px > 0.0 {
        px.round().max(1.0)
    } else {
        px
    }
}

/// `DEFAULT_STYLE` (`utils/richtext/types.ts:30-39`).
pub fn default_style() -> InlineStyle {
    InlineStyle {
        source: Default::default(),
        font_family: "sans-serif".to_string(),
        font_size: 16.0,
        font_weight: "400".to_string(),
        font_style: "normal".to_string(),
        color: "#000000".to_string(),
        letter_spacing: 0.0,
        // Unset line height is Figma's Auto: the font's own spacing; 1.2 is the fallback without the font's metrics.
        line_height: 1.2,
        line_height_auto: true,
        text_decoration: Some("none".to_string()),
        decoration: crate::decorations::DecorationStyle {
            line_style: crate::decorations::DecorationLineStyle::Solid,
            color: None,
            thickness: None,
            underline_offset: None,
        },
        font_features: Vec::new(),
        text_transform: crate::text_case::TextTransform::None,
        text_wrap: crate::text_wrap::TextWrap::Auto,
        tab_stops: None,
        paragraph: crate::paragraph_format::ParagraphFormat::DEFAULT,
        font_variations: Vec::new(),
        fill: None,
        paint: None,
        small_caps_full_size: None,
        character: crate::character::CharacterStyle::DEFAULT,
        lang: None,
        vertical: crate::vertical::VerticalStyle::DEFAULT,
        ruby: crate::ruby::RubyStyle::default(),
    }
}

/// `InlineRun` (`utils/richtext/types.ts:16-19`).
#[derive(Debug, Clone, PartialEq)]
pub struct InlineRun {
    pub text: String,
    pub style: InlineStyle,
}

/// Paragraph alignment (`utils/richtext/types.ts:22`).
pub type Align = &'static str; // 'left' | 'center' | 'right' | 'justify' ('start' | 'end' only mid-parse)

/// `Paragraph.listMetadata` (`utils/richtext/types.ts:27`).
#[derive(Debug, Clone, PartialEq)]
pub struct ListMetadata {
    /// 'ul' | 'ol'
    pub kind: String,
    pub level: i64,
    pub index: Option<i64>,
    /// E7.16: the marker, resolved (the list's `list-style-type`, else the level default).
    pub style: crate::list_markers::ListStyle,
}

/// `Paragraph` (`utils/richtext/types.ts:21-28`).
#[derive(Debug, Clone, PartialEq)]
pub struct Paragraph {
    pub align: Align,
    pub runs: Vec<InlineRun>,
    pub base_style: Option<InlineStyle>,
    pub list_metadata: Option<ListMetadata>,
    /// E6.3: how the paragraph's lines break (`crate::text_wrap`).
    pub wrap: crate::text_wrap::TextWrap,
    /// Tab stops (`crate::tab_stops`): each `\t` advances to the next stop; the parser gives a paragraph with tabs and none of its own the default stops.
    pub tab_stops: Option<std::sync::Arc<crate::tab_stops::TabStops>>,
    /// Indents, hyphenation, the composer and word spacing (`crate::paragraph_format`).
    pub format: crate::paragraph_format::ParagraphFormat,
    /// Its direction (`crate::paragraph_direction`): declared, logical alignment, resolved.
    pub direction: crate::paragraph_direction::ParagraphDirection,
}

/// `LineSegment` (`utils/richtext/types.ts:42-45`).
#[derive(Debug, Clone, PartialEq)]
pub struct LineSegment {
    pub text: String,
    pub style: InlineStyle,
    /// A `\t` of a paragraph with tab stops, resolved on its line (`crate::tab_stops`): its width and leader.
    pub tab: Option<crate::tab_stops::TabFill>,
    /// The hyphen a hyphenated line ends with (`crate::hyphenate`): drawn and measured like text, but not in the
    /// source text (the editor's line map skips it).
    pub hyphen: bool,
}

impl LineSegment {
    /// A segment of `text` in `style` (no tab fill).
    pub fn new(text: impl Into<String>, style: InlineStyle) -> Self {
        LineSegment {
            text: text.into(),
            style,
            tab: None,
            hyphen: false,
        }
    }
}

/// `Line` (`utils/richtext/types.ts:47-50`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Line {
    pub segments: Vec<LineSegment>,
    pub line_height_px: f64,
    /// The wrap ended this line (the next word did not fit), not a hard break
    /// or the paragraph's end: a justified paragraph stretches it
    /// (`positioning::justify_space_px`).
    pub soft_wrap: bool,
    /// The paragraph's indents on this line (`crate::paragraph_format`), px: its container starts this far right
    /// and ends `indent_end` short (`positioning::indented_container`). 0 without indents.
    pub indent_start: f64,
    pub indent_end: f64,
    /// The every-line composer's word spacing (`crate::composer`): how much wider each justifiable space is, px
    /// (negative: narrower). `None`: the draw's own (`positioning::justify_space_px`).
    pub space_px: Option<f64>,
}

impl Line {
    /// A line of `segments`, without indents or composed spacing.
    pub fn new(segments: Vec<LineSegment>, line_height_px: f64, soft_wrap: bool) -> Self {
        Line {
            segments,
            line_height_px,
            soft_wrap,
            ..Default::default()
        }
    }
}

/// `MsdfFontKey` (`utils/msdf/types.ts`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FontKey {
    pub family: String,
    pub weight: i64,
    /// 'normal' | 'italic'
    pub style: String,
}

impl FontKey {
    /// `MsdfAtlasManager.keyToId` (`MsdfAtlasManager.ts:153-155`).
    pub fn id(&self) -> String {
        format!("{}-{}-{}", self.family, self.weight, self.style)
    }
}
