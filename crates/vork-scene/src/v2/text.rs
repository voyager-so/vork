//! The text node (END-STATE-API §7, text): CSS's names for the properties of text, with CSS's
//! meanings, on a node in a 2D canvas, and the SVG paint properties its glyph outlines take
//! (`fill`, `stroke`, `stroke-width`). `content` is a string or styled runs. Moda's text engine
//! lays it out and draws it. Every glyph, word and line is addressable by a track (`glyphs`,
//! `words`, `lines`; §7). Values the engine does not build yet are refused with one line naming
//! them.
use super::*;

/// `content`: a string, or styled runs laid out as one text.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Runs(Vec<Run>),
}
impl Content {
    /// The text of every run, in order, as authored.
    pub fn text(&self) -> String {
        match self { Self::Text(text) => text.clone(), Self::Runs(runs) => runs.iter().map(|run| run.content.as_str()).collect() }
    }
    /// The runs; a string is one run that sets nothing.
    pub fn runs(&self) -> std::borrow::Cow<'_, [Run]> {
        match self {
            Self::Text(text) => std::borrow::Cow::Owned(vec![Run { content: text.clone(), ..Default::default() }]),
            Self::Runs(runs) => std::borrow::Cow::Borrowed(runs),
        }
    }
}
/// One styled run of `content`: its text and the CSS and SVG properties it sets; where it sets none, the
/// node's apply.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Run {
    #[schemars(extend("x-source" = "END-STATE-API §7 (content is a string or styled runs)", "x-help-path" = "run.content", "x-help" = "string"))]
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "END-STATE-API §7 (a font file or a list of files)", "x-help-path" = "run.font", "x-help" = "file-path|file-path[]; default the node's"))]
    pub font: Option<FontFiles>,
    #[serde(rename = "font-size", default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "CSS font-size", "x-help-path" = "run.font-size", "x-help" = "number; px; default the node's"))]
    pub font_size: Option<f32>,
    #[serde(rename = "font-weight", default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "CSS font-weight", "x-help-path" = "run.font-weight", "x-help" = "number|normal|bold; default the node's"))]
    pub font_weight: Option<FontWeight>,
    #[serde(rename = "font-style", default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "CSS font-style", "x-help-path" = "run.font-style", "x-help" = "normal|italic|oblique; default the node's"))]
    pub font_style: Option<FontStyle>,
    #[serde(rename = "font-variation-settings", default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "CSS font-variation-settings", "x-help-path" = "run.font-variation-settings", "x-help" = "string; default the node's"))]
    pub font_variation_settings: Option<String>,
    #[serde(rename = "font-feature-settings", default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "CSS font-feature-settings", "x-help-path" = "run.font-feature-settings", "x-help" = "string; default the node's"))]
    pub font_feature_settings: Option<String>,
    #[serde(rename = "letter-spacing", default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "CSS letter-spacing", "x-help-path" = "run.letter-spacing", "x-help" = "number; px; default the node's"))]
    pub letter_spacing: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "SVG 2 fill; CSS Color 4", "x-help-path" = "run.fill", "x-help" = "\"none\"|CSS colour|CSS gradient|{image, fit}; default the node's"))]
    pub fill: Option<Paint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "SVG 2 stroke; CSS Color 4", "x-help-path" = "run.stroke", "x-help" = "\"none\"|CSS colour|CSS gradient|{image, fit}; default the node's"))]
    pub stroke: Option<Paint>,
}

fn validate_paint(value: &Paint) -> Result<()> {
    if let Paint::Css(css) = value {
        if !css.contains("gradient(") { paint(css)?; }
    }
    Ok(())
}

/// `font`: a font file (.ttf/.otf), or a list of files.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum FontFiles {
    File(String),
    List(Vec<String>),
}
impl FontFiles {
    pub fn files(&self) -> &[String] {
        match self { Self::File(file) => std::slice::from_ref(file), Self::List(files) => files }
    }
    pub fn files_mut(&mut self) -> Vec<&mut String> {
        match self { Self::File(file) => vec![file], Self::List(files) => files.iter_mut().collect() }
    }
}
/// CSS `font-weight`: a number from 1 to 1000, `normal` (400) or `bold` (700).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum FontWeight {
    Keyword(FontWeightKeyword),
    Number(f32),
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum FontWeightKeyword {
    #[default]
    Normal,
    Bold,
}
impl Default for FontWeight {
    fn default() -> Self { Self::Keyword(FontWeightKeyword::Normal) }
}
impl FontWeight {
    /// The numeric weight (`normal` 400, `bold` 700).
    pub fn value(&self) -> f32 {
        match self { Self::Keyword(FontWeightKeyword::Normal) => 400., Self::Keyword(FontWeightKeyword::Bold) => 700., Self::Number(value) => *value }
    }
}
/// CSS `font-style`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum FontStyle {
    #[default]
    Normal,
    Italic,
    Oblique,
}
/// CSS `line-height`: `normal` (the font's own line metrics) or a multiple of the font size.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum LineHeight {
    Normal(LineHeightNormal),
    Multiple(f32),
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum LineHeightNormal {
    #[default]
    Normal,
}
impl Default for LineHeight {
    fn default() -> Self { Self::Normal(LineHeightNormal::Normal) }
}
/// CSS `text-align`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum TextAlign {
    #[default]
    Start,
    End,
    Left,
    Right,
    Center,
    Justify,
}
impl TextAlign {
    pub fn css(self) -> &'static str {
        match self { Self::Start => "start", Self::End => "end", Self::Left => "left", Self::Right => "right", Self::Center => "center", Self::Justify => "justify" }
    }
}
/// CSS `direction`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    #[default]
    Ltr,
    Rtl,
}
/// CSS `white-space`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum WhiteSpace {
    #[default]
    Normal,
    Nowrap,
    Pre,
    PreWrap,
    PreLine,
    BreakSpaces,
}
/// CSS `word-break`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum WordBreak {
    #[default]
    Normal,
    BreakAll,
    KeepAll,
    BreakWord,
}
/// CSS `line-break`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum LineBreak {
    #[default]
    Auto,
    Loose,
    Normal,
    Strict,
    Anywhere,
}

pub(super) fn font_size() -> f32 { 16. }
pub(super) fn normal() -> String { "normal".into() }
pub(super) fn is_normal(value: &String) -> bool { value == "normal" }
pub(super) fn one() -> f32 { 1. }

/// Parse an SVG text paint using CSS Color 4 as straight-alpha sRGB.
/// Validation and drawing share this parser.
/// `none` is the SVG no-paint keyword. Context-dependent and other colour spaces refuse.
pub fn paint(paint: &str) -> Result<Option<[f32; 4]>> {
    use cssparser::{Parser, ParserInput};
    use cssparser_color::{Color, RgbaLegacy, hsl_to_rgb, hwb_to_rgb};
    let mut input = ParserInput::new(paint);
    let mut parser = Parser::new(&mut input);
    if parser.try_parse(|p| p.expect_ident_matching("none")).is_ok() {
        ensure!(parser.is_exhausted(), "invalid CSS paint {paint:?}");
        return Ok(None);
    }
    let color = parser.parse_entirely(Color::parse)
        .map_err(|_| anyhow::anyhow!("invalid or unbuilt CSS colour {paint:?}"))?;
    let rgba = match color {
        Color::Rgba(rgba) => rgba,
        Color::Hsl(hsl) => {
            let (r, g, b) = hsl_to_rgb(hsl.hue.unwrap_or(0.) / 360., hsl.saturation.unwrap_or(0.), hsl.lightness.unwrap_or(0.));
            RgbaLegacy::from_floats(r, g, b, hsl.alpha.unwrap_or(0.))
        }
        Color::Hwb(hwb) => {
            let (r, g, b) = hwb_to_rgb(hwb.hue.unwrap_or(0.) / 360., hwb.whiteness.unwrap_or(0.), hwb.blackness.unwrap_or(0.));
            RgbaLegacy::from_floats(r, g, b, hwb.alpha.unwrap_or(0.))
        }
        _ => bail!("CSS colour {paint:?}: only named, hex, rgb(), hsl() and hwb() sRGB paints are built"),
    };
    Ok(Some([f32::from(rgba.red) / 255., f32::from(rgba.green) / 255., f32::from(rgba.blue) / 255., rgba.alpha]))
}

#[cfg(test)]
mod paint_tests {
    use super::paint;

    #[test]
    fn css_color_four_alpha_clamping_and_missing_components() {
        for (value, expected) in [
            ("rgb(100% 0% 0% / 50%)", [1., 0., 0., 0.5]),
            ("rgba(255, 0, 0, .5)", [1., 0., 0., 0.5]),
            ("hsl(0.5turn 100% 50% / .5)", [0., 1., 1., 0.5]),
            ("hwb(200grad 0% 0% / 50%)", [0., 1., 1., 0.5]),
            ("#ff000080", [1., 0., 0., 128. / 255.]),
            ("rgb(300 -5 0 / 200%)", [1., 0., 0., 1.]),
            ("rgb(none 0 0 / none)", [0.; 4]),
            ("transparent", [0.; 4]), ("ReBeccAPurple", [102. / 255., 51. / 255., 153. / 255., 1.]),
        ] {
            assert_eq!(paint(value).unwrap(), Some(expected), "{value}");
        }
        assert_eq!(paint(" NONE /* no outline */ ").unwrap(), None);
    }
}

/// Which addressable pieces of a text node a track names (§7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TextUnit {
    /// Letters as a reader sees them: Unicode grapheme clusters (UAX #29).
    Glyphs,
    /// UAX #29 words.
    Words,
    /// The lines after wrapping.
    Lines,
}
impl TextUnit {
    pub fn segment(self) -> &'static str {
        match self { Self::Glyphs => "glyphs", Self::Words => "words", Self::Lines => "lines" }
    }
    pub(super) fn parse(segment: &str) -> Option<Self> {
        match segment { "glyphs" => Some(Self::Glyphs), "words" => Some(Self::Words), "lines" => Some(Self::Lines), _ => None }
    }
}
/// A field of one glyph, word or line that a track keys (§7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnitProperty {
    Translation,
    Rotation,
    Scale,
    Opacity,
    /// A colour: four numbers (r, g, b, a from 0 to 1, as CSS's sRGB), or CSS colour strings keyed STEP.
    Fill,
    Stroke,
    /// CSS `transform-origin` strings, keyed STEP ([`transform_origin`]).
    TransformOrigin,
}
impl UnitProperty {
    pub fn segment(self) -> &'static str {
        match self {
            Self::Translation => "translation", Self::Rotation => "rotation", Self::Scale => "scale", Self::Opacity => "opacity",
            Self::Fill => "fill", Self::Stroke => "stroke", Self::TransformOrigin => "transform-origin",
        }
    }
    pub(super) fn parse(segment: &str) -> Option<Self> {
        [Self::Translation, Self::Rotation, Self::Scale, Self::Opacity, Self::Fill, Self::Stroke, Self::TransformOrigin]
            .into_iter().find(|property| property.segment() == segment)
    }
}

/// CSS `transform-origin` of a glyph, word or line (§7), whose reference box is the piece's ink box: one or
/// two of `left`, `center`, `right`, `top`, `bottom`, a percentage of the box, or a length in px from the
/// box's top-left. Per axis (x, y): the fraction of the box and the px added to it.
pub fn transform_origin(value: &str) -> Result<[(f32, f32); 2]> {
    #[derive(Clone, Copy, PartialEq)]
    enum Axis { X, Y, Either }
    let parse = |token: &str| -> Result<(Axis, f32, f32)> {
        Ok(match token {
            "left" => (Axis::X, 0., 0.),
            "right" => (Axis::X, 1., 0.),
            "top" => (Axis::Y, 0., 0.),
            "bottom" => (Axis::Y, 1., 0.),
            "center" => (Axis::Either, 0.5, 0.),
            "0" => (Axis::Either, 0., 0.),
            _ => {
                let number = |text: &str| text.parse::<f32>().ok().filter(|n| n.is_finite());
                if let Some(percent) = token.strip_suffix('%').and_then(number) {
                    (Axis::Either, percent / 100., 0.)
                } else if let Some(px) = token.strip_suffix("px").and_then(number) {
                    (Axis::Either, 0., px)
                } else {
                    bail!("transform-origin {value:?}: {token:?} is not left, center, right, top, bottom, a percentage or a length in px")
                }
            }
        })
    };
    let tokens: Vec<&str> = value.split_whitespace().collect();
    let (mut x, mut y) = match tokens.as_slice() {
        [one] => match parse(one)? {
            vertical @ (Axis::Y, ..) => ((Axis::Either, 0.5, 0.), vertical),
            other => (other, (Axis::Either, 0.5, 0.)),
        },
        [first, second] => (parse(first)?, parse(second)?),
        [_, _, _] => bail!("transform-origin {value:?}: a z offset is not built in a 2D canvas; give x and y"),
        _ => bail!("transform-origin {value:?}: give one or two values"),
    };
    if x.0 == Axis::Y || y.0 == Axis::X {
        std::mem::swap(&mut x, &mut y);
    }
    ensure!(x.0 != Axis::Y && y.0 != Axis::X, "transform-origin {value:?}: give one horizontal and one vertical value");
    Ok([(x.1, x.2), (y.1, y.2)])
}

/// The text a node lays out, after `white-space` (CSS Text 3 §4.1.1), each character with the index of
/// the run it comes from: `normal` and `nowrap` collapse every run of white space, newlines included, to
/// one space; `pre-line` collapses spaces and tabs and keeps newlines as line breaks. Spaces at the start
/// and end of a line are removed. A collapsed space belongs to the run of the first white space it stands
/// for, across run boundaries as CSS collapses across inline boxes.
pub fn laid_out_runs(content: &Content, white_space: WhiteSpace) -> Vec<(char, usize)> {
    let keep_newlines = white_space == WhiteSpace::PreLine;
    let mut out = Vec::new();
    let mut gap: Option<usize> = None;
    let mut line_start = true;
    let mut previous_cr = false;
    for (index, run) in content.runs().iter().enumerate() {
        for c in run.content.chars() {
            let crlf = previous_cr && c == '\n';
            previous_cr = c == '\r';
            if crlf { continue; }
            let c = if c == '\r' { '\n' } else { c };
            if c == '\n' && keep_newlines {
                out.push(('\n', index));
                gap = None;
                line_start = true;
            } else if matches!(c, ' ' | '\t' | '\r' | '\n') {
                if !line_start && gap.is_none() {
                    gap = Some(index);
                }
            } else {
                if let Some(run) = gap.take() {
                    out.push((' ', run));
                }
                out.push((c, index));
                line_start = false;
            }
        }
    }
    out
}

/// [`laid_out_runs`]'s text.
pub fn laid_out_text(content: &Content, white_space: WhiteSpace) -> String {
    laid_out_runs(content, white_space).into_iter().map(|(c, _)| c).collect()
}

/// The byte ranges of a text's glyphs (UAX #29 grapheme clusters) or words (UAX #29 words), in order.
/// Newlines are neither glyphs nor words.
pub fn unit_ranges(text: &str, unit: TextUnit) -> Vec<std::ops::Range<usize>> {
    use unicode_segmentation::UnicodeSegmentation;
    match unit {
        TextUnit::Glyphs => text.grapheme_indices(true).filter(|(_, g)| *g != "\n" && *g != "\r\n").map(|(i, g)| i..i + g.len()).collect(),
        TextUnit::Words => text.unicode_word_indices().map(|(i, w)| i..i + w.len()).collect(),
        TextUnit::Lines => Vec::new(),
    }
}

impl Scene {
    /// One text node's checks: the fields Moda's engine does not build refuse in one line each.
    /// Validate both authored and sampled text styles before shaping.
    pub fn validate_text(node: &Node, images: &Table<Image>) -> Result<()> {
        let NodeKind::Text { content, font, font_size, font_weight, font_variation_settings: _, font_feature_settings: _, letter_spacing, word_spacing,
            line_height, white_space, word_break, line_break, width, fill, stroke, stroke_width, .. } = &node.kind else { return Ok(()) };
        let name = &node.name;
        let check_font = |here: &str, font: &FontFiles, font_size: f32, font_weight: FontWeight, letter_spacing: f32| -> Result<()> {
            ensure!(!font.files().is_empty(), "{here}: font names no file; give a .ttf or .otf file");
            ensure!(font_size.is_finite() && font_size > 0., "{here}: font-size must be a positive number of px");
            let weight = font_weight.value();
            ensure!(weight.is_finite() && (1. ..=1000.).contains(&weight), "{here}: font-weight must be from 1 to 1000, normal or bold");
            ensure!(letter_spacing.is_finite(), "{here}: letter-spacing must be a finite number of px");
            Ok(())
        };
        check_font(&format!("node {name:?}"), font, *font_size, *font_weight, *letter_spacing)?;
        if let Content::Runs(runs) = content {
            for (index, run) in runs.iter().enumerate() {
                let here = format!("node {name:?}: content[{index}]");
                check_font(&here, run.font.as_ref().unwrap_or(font), run.font_size.unwrap_or(*font_size), run.font_weight.unwrap_or(*font_weight),
                    run.letter_spacing.unwrap_or(*letter_spacing))?;
                for (field, value) in [("fill", &run.fill), ("stroke", &run.stroke)] {
                    if let Some(value) = value {
                        validate_paint(value).map_err(|error| anyhow::anyhow!("{here}.{field}: {error}"))?;
                    }
                }
            }
        }
        ensure!(*word_spacing == 0., "node {name:?}: word-spacing other than 0 is not built yet");
        if let LineHeight::Multiple(multiple) = line_height {
            ensure!(multiple.is_finite() && *multiple > 0., "node {name:?}: line-height must be normal or a positive multiple of the font size");
        }
        ensure!(!matches!(white_space, WhiteSpace::Pre | WhiteSpace::PreWrap | WhiteSpace::BreakSpaces),
            "node {name:?}: white-space {:?} is not built yet; normal, nowrap and pre-line are", serde_json::to_value(white_space)?.as_str().unwrap_or_default());
        ensure!(*word_break == WordBreak::Normal, "node {name:?}: word-break other than normal is not built yet");
        ensure!(*line_break == LineBreak::Auto, "node {name:?}: line-break other than auto is not built yet");
        if let Some(width) = width {
            ensure!(width.is_finite() && *width > 0., "node {name:?}: width must be a positive number of px, or null for no wrapping");
        }
        ensure!(stroke_width.is_finite() && *stroke_width >= 0., "node {name:?}: stroke-width must be a nonnegative number of px");
        for value in [fill, stroke].into_iter().chain(content.runs().iter().flat_map(|run| run.fill.iter().chain(&run.stroke))) {
            if let Paint::Image(image) = value {
                ensure!(images.contains_key(&image.image), "node {name:?}: paint names image {:?}, which is not in images", image.image);
            }
        }
        for (field, value) in [("fill", fill), ("stroke", stroke)] {
            validate_paint(value).map_err(|error| anyhow::anyhow!("node {name:?}: {field}: {error}"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod whitespace_tests {
    use super::*;

    #[test]
    fn pre_line_normalizes_crlf_and_lone_cr_across_runs() {
        let text = Content::Text("first\r\nsecond\rthird".into());
        assert_eq!(laid_out_text(&text, WhiteSpace::PreLine), "first\nsecond\nthird");
        assert_eq!(laid_out_text(&text, WhiteSpace::Normal), "first second third");
        let runs = Content::Runs(vec![
            Run { content: "first\r".into(), ..Default::default() },
            Run { content: "\nsecond\r".into(), ..Default::default() },
            Run { content: "third".into(), ..Default::default() },
        ]);
        assert_eq!(laid_out_text(&runs, WhiteSpace::PreLine), "first\nsecond\nthird");
        assert_eq!(laid_out_runs(&runs, WhiteSpace::PreLine)[5], ('\n', 0));
    }
}
