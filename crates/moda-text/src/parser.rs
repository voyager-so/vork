//! HTML-subset → `Paragraph[]` — line-for-line port of
//! `utils/richtext/parser.ts` over html5ever (via `scraper`, the pinned
//! WHATWG-parity parser — DECISION §6). Every function cites its TS lines.

use crate::font_weight::{
    normalize_font_weight_value, normalize_letter_spacing_value, normalize_typography_font_style,
    normalize_typography_font_weight,
};
use crate::jsnum::js_parse_float;
use crate::paragraph_direction::{DeclaredDirection, ParagraphDirection};
use crate::types::{default_style, Align, InlineRun, InlineStyle, ListMetadata, Paragraph};
use scraper::{Html, Node};

/// `INLINE_TAGS` (parser.ts:7).
fn is_inline_tag(tag: &str) -> bool {
    matches!(
        tag,
        "span" | "b" | "strong" | "i" | "em" | "u" | "s" | "a" | "br"
    )
}

/// `BLOCK_TAGS` (parser.ts:8).
fn is_block_tag(tag: &str) -> bool {
    matches!(tag, "p" | "div" | "li" | "ul" | "ol")
}

/// `parseLetterSpacingPx` (parser.ts:10-30): normalize, then
/// `^(-?(?:\d+\.?\d*|\.\d+))(px|em)?$`.
fn parse_letter_spacing_px(value: &str, font_size_px: f64) -> Option<f64> {
    let normalized = normalize_letter_spacing_value(value)?;
    let raw = normalized.trim();
    // The TS regex allows '-' but not '+'.
    let (num, unit) = match_letter_spacing_number(raw)?;
    let amount = js_parse_float(num)?;
    match unit.map(|u| u.to_lowercase()).as_deref() {
        None | Some("px") => Some(amount),
        Some("em") => Some(amount * font_size_px),
        _ => None,
    }
}

/// `^(-?(?:\d+\.?\d*|\.\d+))(px|em)?$` (parser.ts:14) — case-insensitive.
fn match_letter_spacing_number(raw: &str) -> Option<(&str, Option<&str>)> {
    let bytes = raw.as_bytes();
    let mut i = 0usize;
    if i < bytes.len() && bytes[i] == b'-' {
        i += 1;
    }
    let ds = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i > ds {
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
    } else if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        let fs = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == fs {
            return None;
        }
    } else {
        return None;
    }
    let num = &raw[..i];
    let rest = &raw[i..];
    if rest.is_empty() {
        return Some((num, None));
    }
    if rest.eq_ignore_ascii_case("px") || rest.eq_ignore_ascii_case("em") {
        return Some((num, Some(rest)));
    }
    None
}

/// `parseFontSizePx` (parser.ts:32-56): `^([0-9]*\.?[0-9]+)(px|em|pt|%)?$`.
fn parse_font_size_px(value: &str, inherited_font_size_px: f64) -> Option<f64> {
    let (num, unit) = match_css_number_with_unit_unsigned(value.trim(), &["px", "em", "pt", "%"])?;
    let amount = js_parse_float(num)?;
    match unit.map(|u| u.to_lowercase()).as_deref() {
        None | Some("px") => Some(amount),
        Some("pt") => Some(amount), // a point is a px (72 per inch)
        Some("em") => Some(amount * inherited_font_size_px),
        // `%` is `em` scaled by 100. Keep this arm in lockstep with the TS twin:
        // dropping it renders a `font-size: 60%` exponent at the full inherited size.
        Some("%") => Some(amount / 100.0 * inherited_font_size_px),
        _ => None,
    }
}

/// `^([0-9]*\.?[0-9]+)(unit)?$` (no sign — parser.ts:33/56).
fn match_css_number_with_unit_unsigned<'a>(
    raw: &'a str,
    allowed: &[&str],
) -> Option<(&'a str, Option<&'a str>)> {
    let bytes = raw.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
    }
    let ds = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == ds {
        // must end with at least one digit after optional '.'
        // (covers both "12" — ds==i means no trailing digits after '.' or none at all)
        // "12" case: digits consumed before '.', but the regex requires [0-9]+ at end;
        // re-check: for "12", we consumed "12", no '.', ds == i == 2 with zero digits after — invalid per scan.
        // Handle by requiring the last consumed char to be a digit instead:
        return match_trailing_digit_form(raw, allowed);
    }
    let num = &raw[..i];
    let rest = &raw[i..];
    if rest.is_empty() {
        return Some((num, None));
    }
    for u in allowed {
        if rest.eq_ignore_ascii_case(u) {
            return Some((num, Some(rest)));
        }
    }
    None
}

/// Fallback for the digits-only / digits-no-dot form of
/// `[0-9]*\.?[0-9]+` (e.g. "12", "12px").
fn match_trailing_digit_form<'a>(
    raw: &'a str,
    allowed: &[&str],
) -> Option<(&'a str, Option<&'a str>)> {
    let bytes = raw.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 {
        return None;
    }
    let num = &raw[..i];
    let rest = &raw[i..];
    if rest.is_empty() {
        return Some((num, None));
    }
    for u in allowed {
        if rest.eq_ignore_ascii_case(u) {
            return Some((num, Some(rest)));
        }
    }
    None
}

enum LineHeightValue {
    Px(f64),
    Multiple(f64),
    /// `normal`: Figma's Auto (`InlineStyle::line_height_auto`).
    Auto,
}

/// `parseLineHeightValue` (parser.ts:53-68): `^([0-9]*\.?[0-9]+)(px|em|%)?$`;
/// `em`/unitless → multiple (legacy `Nem` behavior, parser.ts:65-67).
fn parse_line_height_value(value: &str) -> Option<LineHeightValue> {
    if value.trim().eq_ignore_ascii_case("normal") {
        return Some(LineHeightValue::Auto);
    }
    let (num, unit) = match_css_number_with_unit_unsigned(value.trim(), &["px", "em", "%"])?;
    let amount = js_parse_float(num)?;
    if !amount.is_finite() || amount <= 0.0 {
        return None;
    }
    match unit.map(|u| u.to_lowercase()).as_deref() {
        Some("px") => Some(LineHeightValue::Px(amount)),
        Some("%") => Some(LineHeightValue::Multiple(amount / 100.0)),
        _ => Some(LineHeightValue::Multiple(amount)),
    }
}

/// `parseInlineStyle` (parser.ts:70-152).
pub fn parse_inline_style(style_attr: Option<&str>, inherited: &InlineStyle) -> InlineStyle {
    let mut style = inherited.clone();
    let Some(style_attr) = style_attr else {
        return style;
    };
    let mut pending_line_height_px: Option<f64> = None;
    let mut pending_letter_spacing: Option<String> = None;
    // `--nf-text-fill` (crate::run_fill): a declared fill wins; a `color`
    // declared without one clears an inherited fill.
    let mut declared_color = false;
    let mut declared_fill: Option<Option<crate::ops::GradientSpec>> = None;
    // E8.22 character attributes (`crate::character`), applied after the walk so `em` sees this span's font size.
    let mut pending_character: Vec<(String, String)> = Vec::new();
    // Paragraph indents, hyphenation, composer, word spacing (`crate::paragraph_format`), after the walk too.
    let mut pending_paragraph: Vec<(String, String)> = Vec::new();
    for rule in style_attr.split(';') {
        // `const [rawProp, rawValue] = rule.split(':')` — second element only.
        let mut it = rule.split(':');
        let raw_prop = it.next().unwrap_or("");
        let raw_value = it.next().unwrap_or("");
        if raw_prop.is_empty() || raw_value.is_empty() {
            continue;
        }
        let prop = raw_prop.trim().to_lowercase();
        let value = raw_value.trim().to_string();
        match prop.as_str() {
            "font-family" => {
                // value.replace(/^["']|["']$/g, '') — one leading and one
                // trailing quote char of either kind.
                let mut v = value.as_str();
                if v.starts_with('"') || v.starts_with('\'') {
                    v = &v[1..];
                }
                if v.ends_with('"') || v.ends_with('\'') {
                    v = &v[..v.len() - 1];
                }
                style.font_family = v.to_string();
            }
            "font-size" => {
                let inherited_fs = if inherited.font_size != 0.0 {
                    inherited.font_size
                } else {
                    16.0 // DEFAULT_STYLE.fontSize || 16 (parser.ts:86)
                };
                if let Some(fs) = parse_font_size_px(&value, inherited_fs) {
                    style.font_size = fs;
                }
            }
            "font-weight" => {
                // parser.ts:91-108
                let relative_weight =
                    normalize_font_weight_value(&value, Some(inherited.font_weight.as_str()));
                let normalized = normalize_typography_font_weight(&value);
                let vt = value.trim().to_lowercase();
                if vt == "bolder" || vt == "lighter" {
                    if let Some(rw) = &relative_weight {
                        style.font_weight = rw.clone();
                    }
                } else if let Some(w) = normalized.font_weight {
                    style.font_weight = w.to_string();
                } else {
                    style.font_weight = relative_weight.unwrap_or_else(|| value.clone());
                }
                if normalized.font_weight.is_some()
                    && value.trim().chars().any(|c| c.is_whitespace())
                {
                    if let Some(fs) = normalized.font_style {
                        style.font_style = fs.to_string();
                    }
                }
            }
            "font-style" => {
                // parser.ts:110-115
                match normalize_typography_font_style(&value) {
                    Some(s) => style.font_style = s.to_string(),
                    None => style.font_style = value.clone(),
                }
            }
            "color" => {
                style.color = value.clone();
                declared_color = true;
            }
            crate::character::BASELINE_SHIFT_PROPERTY
            | crate::character::HORIZONTAL_SCALE_PROPERTY
            | crate::character::VERTICAL_SCALE_PROPERTY
            | crate::character::CHARACTER_ROTATION_PROPERTY
            | crate::character::KERNING_PROPERTY => {
                pending_character.push((prop.clone(), value.clone()));
            }
            crate::paragraph_format::TEXT_INDENT_PROPERTY
            | crate::paragraph_format::INDENT_LEFT_PROPERTY
            | crate::paragraph_format::INDENT_RIGHT_PROPERTY
            | crate::paragraph_format::HYPHENATE_PROPERTY
            | crate::paragraph_format::COMPOSER_PROPERTY
            | crate::paragraph_format::WORD_SPACING_PROPERTY
            | crate::paragraph_format::WORD_BREAK_PROPERTY
            | crate::paragraph_format::LINE_BREAK_PROPERTY
            | crate::paragraph_format::TEXT_SPACING_TRIM_PROPERTY
            | crate::paragraph_format::TEXT_AUTOSPACE_PROPERTY => {
                // parser.ts: inherits; the paragraph a run opens takes it, like `text-wrap`.
                pending_paragraph.push((prop.clone(), value.clone()));
            }
            crate::lang::LANG_PROPERTY => {
                // The run's language (its CJK faces) inherits; `auto` (or an empty value) clears it, an unreadable tag
                // keeps the inherited one. It is also a paragraph format property (the hyphenation patterns).
                let v = value.trim();
                if v.is_empty() || v.eq_ignore_ascii_case("auto") {
                    style.lang = None;
                } else if let Some(lang) = crate::lang::canonical_lang(v) {
                    style.lang = Some(lang);
                }
                pending_paragraph.push((prop.clone(), value.clone()));
            }
            crate::vertical::WRITING_MODE_PROPERTY
            | crate::vertical::TEXT_ORIENTATION_PROPERTY
            | crate::vertical::TEXT_COMBINE_UPRIGHT_PROPERTY => {
                // Vertical text (parser.ts): inherits; the node takes its first run's writing mode.
                style.vertical.apply_declaration(&prop, &value);
            }
            crate::ruby::RUBY_PROPERTY | crate::ruby::RUBY_ALIGN_PROPERTY | crate::ruby::RUBY_SIZE_PROPERTY => {
                // Ruby (parser.ts): on the span it annotates.
                style.ruby.apply_declaration(&prop, &value);
            }
            crate::run_fill::TEXT_FILL_PROPERTY => {
                declared_fill = Some(crate::run_fill::parse_text_fill(&value));
            }
            crate::types::PAINT_PROPERTY => {
                style.paint = value.trim().parse().ok();
            }
            "letter-spacing" => {
                pending_letter_spacing = Some(value.clone());
            }
            "line-height" => match parse_line_height_value(&value) {
                Some(LineHeightValue::Px(px)) => {
                    pending_line_height_px = Some(px);
                    style.line_height_auto = false;
                }
                Some(LineHeightValue::Multiple(m)) => {
                    style.line_height = m;
                    style.line_height_auto = false;
                }
                Some(LineHeightValue::Auto) => {
                    style.line_height = crate::types::default_style().line_height;
                    style.line_height_auto = true;
                }
                None => {}
            },
            "font-feature-settings" => {
                // F7.3 (parser.ts `font-feature-settings`): inherits like
                // every other run style; `normal` clears.
                style.font_features = crate::font_features::parse_font_feature_settings(&value);
            }
            "font-variation-settings" => {
                // E6.3 (parser.ts): inherits; `normal` clears.
                style.font_variations =
                    crate::font_variations::parse_font_variation_settings(&value);
            }
            crate::tab_stops::TAB_STOPS_PROPERTY => {
                // parser.ts: inherits; the paragraph a run opens takes it, like `text-wrap`.
                style.tab_stops = crate::tab_stops::parse_tab_stops(&value).map(std::sync::Arc::new);
            }
            "text-wrap" | "text-wrap-style" => {
                // E6.3 (parser.ts): inherits; the paragraph a run opens takes it.
                style.text_wrap = crate::text_wrap::TextWrap::parse(&value);
            }
            "text-transform" => {
                // E6.3 text case (parser.ts `text-transform`): inherits; the
                // run text is rewritten after the walk (`crate::text_case`).
                style.text_transform = crate::text_case::TextTransform::parse(&value);
            }
            "text-decoration" => {
                // parser.ts:129-135
                let v = value.to_lowercase();
                style.text_decoration = Some(
                    if v.contains("underline") && v.contains("line-through") {
                        "underline line-through"
                    } else if v.contains("underline") {
                        "underline"
                    } else if v.contains("line-through") {
                        "line-through"
                    } else {
                        "none"
                    }
                    .to_string(),
                );
                // CSS: the shorthand resets the style, colour and thickness, then sets what it names.
                style.decoration.apply_shorthand(&value);
            }
            crate::decorations::TEXT_DECORATION_STYLE_PROPERTY
            | crate::decorations::TEXT_DECORATION_COLOR_PROPERTY
            | crate::decorations::TEXT_DECORATION_THICKNESS_PROPERTY
            | crate::decorations::TEXT_UNDERLINE_OFFSET_PROPERTY => {
                // parser.ts: inherits like `text-decoration`.
                style.decoration.apply_declaration(&prop, &value);
            }
            _ => {}
        }
    }
    match declared_fill {
        Some(fill) => style.fill = fill,
        None if declared_color => style.fill = None,
        None => {}
    }
    // parser.ts:141-150
    if let Some(px) = pending_line_height_px {
        let fs = if style.font_size != 0.0 {
            style.font_size
        } else if inherited.font_size != 0.0 {
            inherited.font_size
        } else {
            16.0
        };
        if fs > 0.0 {
            style.line_height = px / fs;
        }
    }
    if let Some(pls) = pending_letter_spacing {
        let fs = if style.font_size != 0.0 {
            style.font_size
        } else if inherited.font_size != 0.0 {
            inherited.font_size
        } else {
            16.0
        };
        if let Some(ls) = parse_letter_spacing_px(&pls, fs) {
            style.letter_spacing = ls;
        }
    }
    let fs = if style.font_size != 0.0 { style.font_size } else { 16.0 };
    for (prop, value) in &pending_character {
        style.character.apply_declaration(prop, value, fs);
    }
    for (prop, value) in &pending_paragraph {
        style.paragraph.apply_declaration(prop, value, fs);
    }
    style
}

/// `applyTagStyle` (parser.ts:154-183). `<a>` implies nothing: as in Figma, a link is just a link, drawn
/// underlined only when its text is styled so.
pub fn apply_tag_style(tag: &str, style: &InlineStyle) -> InlineStyle {
    let mut updated = style.clone();
    match tag {
        "b" | "strong" => {
            if updated.font_weight == "400" || updated.font_weight == "normal" {
                updated.font_weight = "700".to_string();
            }
        }
        "i" | "em" => updated.font_style = "italic".to_string(),
        "u" => {
            updated.text_decoration = Some(
                if updated.text_decoration.as_deref() == Some("line-through") {
                    "underline line-through"
                } else {
                    "underline"
                }
                .to_string(),
            );
        }
        "s" => {
            updated.text_decoration = Some(
                if updated.text_decoration.as_deref() == Some("underline") {
                    "underline line-through"
                } else {
                    "line-through"
                }
                .to_string(),
            );
        }
        _ => {}
    }
    updated
}

/// `extractParagraphAlignFromStyle` (parser.ts):
/// `/text-align\s*:\s*(left|center|right|justify|start|end)/i` searched anywhere. `start` / `end` stay logical until
/// the paragraph's direction is known (`crate::paragraph_direction::resolve_paragraph_direction`).
pub fn extract_align_from_style(style_attr: Option<&str>) -> Align {
    let Some(attr) = style_attr else {
        return "left";
    };
    let lower = attr.to_lowercase();
    let needle = "text-align";
    let bytes = lower.as_bytes();
    let mut from = 0usize;
    while let Some(pos) = lower[from..].find(needle) {
        let mut i = from + pos + needle.len();
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b':' {
            i += 1;
            while i < bytes.len() && (bytes[i] as char).is_whitespace() {
                i += 1;
            }
            for cand in ["left", "center", "right", "justify", "start", "end"] {
                if lower[i..].starts_with(cand) {
                    return match cand {
                        "left" => "left",
                        "center" => "center",
                        "right" => "right",
                        "start" => "start",
                        "end" => "end",
                        _ => "justify",
                    };
                }
            }
        }
        from += pos + 1;
    }
    "left"
}

/// `sanitizeText` (parser.ts): `\r\n?` → `\n`, strip U+00AD/U+2060/U+FEFF
/// (but NOT U+200D — see below), and U+200C unless it follows a character of a
/// script it shapes ([`zwnj_shapes_after`]). Keep U+200B for wrapping.
pub fn sanitize_text(text: &str) -> String {
    sanitize_text_keeping(text, false)
}

/// [`sanitize_text`], keeping soft hyphens when `soft_hyphens`: a paragraph breaks at them (`crate::hyphenate`; the
/// wrap drops them from what it lays out).
pub fn sanitize_text_keeping(text: &str, soft_hyphens: bool) -> String {
    sanitize_with_source(text, soft_hyphens, 0).0
}

fn sanitize_with_source(text: &str, soft_hyphens: bool, source_start: usize) -> (String, crate::types::SourceIndices) {
    let mut out = String::with_capacity(text.len());
    let mut indices = Vec::new();
    let mut chars = text.chars().enumerate().peekable();
    while let Some((index, c)) = chars.next() {
        let before = out.len();
        match c {
            '\r' => {
                if chars.peek().is_some_and(|(_, c)| *c == '\n') {
                    chars.next();
                }
                out.push('\n');
            }
            // U+200D (ZWJ) is deliberately NOT stripped here: it is the emoji
            // join char and must survive to `split_text_into_segments` so ZWJ
            // sequences cluster as one emoji. A stray ZWJ in plain text is
            // stripped there, from text segments only.
            '\u{00AD}' if soft_hyphens => out.push(c),
            // ZWNJ is the author's "don't join here" in Persian, Urdu and the Brahmic
            // scripts: the shaper honours it. Anywhere else it is an invisible leftover.
            '\u{200C}' if out.chars().next_back().is_some_and(zwnj_shapes_after) => out.push(c),
            '\u{200C}' | '\u{FEFF}' | '\u{00AD}' | '\u{2060}' => {}
            _ => out.push(c),
        }
        if out.len() > before { indices.push(Some(source_start + index)); }
    }
    (out, crate::types::SourceIndices::new(indices))
}

/// Scripts whose shaping a ZERO WIDTH NON-JOINER changes: the cursive ones (Arabic,
/// Syriac, Thaana, N'Ko, Mandaic, Mongolian, Hanifi Rohingya, Adlam) and the Brahmic
/// ones (Devanagari … Sinhala, Myanmar, Khmer). Code-point blocks, as parser.ts
/// `zwnjShapesAfter`.
pub fn zwnj_shapes_after(c: char) -> bool {
    matches!(
        c as u32,
        0x0600..=0x0DFF
            | 0x1000..=0x109F
            | 0x1780..=0x18AF
            | 0xFB50..=0xFDFF
            | 0xFE70..=0xFEFE
            | 0x10D00..=0x10D3F
            | 0x1E900..=0x1E95F
    )
}

struct ListCtx {
    /// 'ul' | 'ol'
    list_type: &'static str,
    level: i64,
    next_index: Option<i64>,
    /// E7.16: the marker, resolved (declared `list-style-type` / `type`, else the level default).
    style: crate::list_markers::ListStyle,
}

struct WalkState {
    source_cursor: usize,
    source_text: String,
    source_reserved: bool,
    paragraphs: Vec<Paragraph>,
    last_observed_style: InlineStyle,
    list_stack: Vec<ListCtx>,
    pending_list_meta: Option<ListMetadata>,
}

impl WalkState {
    fn begin_source_paragraph(&mut self) {
        if !self.source_reserved {
            if !self.paragraphs.is_empty() {
                self.source_cursor += 1;
                self.source_text.push('\n');
            }
            self.source_reserved = true;
        }
    }

    /// `ensureParagraph` (parser.ts:208-217).
    fn ensure_paragraph(
        &mut self,
        current: Option<usize>,
        align: Align,
        dir: DeclaredDirection,
        style: &InlineStyle,
    ) -> usize {
        if let Some(i) = current {
            return i;
        }
        let mut p = Paragraph {
            align,
            runs: Vec::new(),
            base_style: None,
            list_metadata: None,
            wrap: style.text_wrap,
            tab_stops: style.tab_stops.clone(),
            format: style.paragraph,
            direction: ParagraphDirection {
                declared: dir,
                ..Default::default()
            },
        };
        if let Some(meta) = self.pending_list_meta.take() {
            p.list_metadata = Some(meta);
        }
        self.begin_source_paragraph();
        self.source_reserved = false;
        self.paragraphs.push(p);
        self.paragraphs.len() - 1
    }
}

/// `parseHtmlToParagraphs` (parser.ts:196-333). The LRU paragraph cache
/// (parser.ts:336-356) is a perf detail with no semantic effect — omitted.
pub struct ParsedText {
    pub paragraphs: Vec<Paragraph>,
    /// Decoded source, with paragraph separators, before sanitization and case transforms.
    pub source_text: String,
}
pub fn parse_rich_text_to_paragraphs(html: &str) -> Vec<Paragraph> {
    parse_rich_text_with_source(html).paragraphs
}
pub fn source_code_point_count(html: &str) -> usize {
    parse_rich_text_with_source(html).source_text.chars().count()
}
pub fn parse_rich_text_with_source(html: &str) -> ParsedText {
    let doc = Html::parse_document(html);
    let mut state = WalkState {
        source_cursor: 0,
        source_text: String::new(),
        source_reserved: false,
        paragraphs: Vec::new(),
        last_observed_style: default_style(),
        list_stack: Vec::new(),
        pending_list_meta: None,
    };

    // DOMParser('text/html').body — find <body> under the root <html>.
    let root = doc.tree.root();
    let mut body = None;
    for child in root.children() {
        if let Node::Element(el) = child.value() {
            if el.name() == "html" {
                for sub in child.children() {
                    if let Node::Element(el2) = sub.value() {
                        if el2.name() == "body" {
                            body = Some(sub);
                        }
                    }
                }
            }
        }
    }
    let Some(body) = body else {
        return ParsedText { paragraphs: state.paragraphs, source_text: state.source_text };
    };

    let default = default_style();
    let mut para: Option<usize> = None;
    for child in body.children() {
        para = walk(&mut state, child, &default, para, "left", DeclaredDirection::Auto);
    }
    let _ = para;
    crate::text_case::apply_text_transforms(&mut state.paragraphs);
    // A tab in a paragraph with no stops of its own goes to the default stops (`crate::tab_stops`), never a space's
    // width.
    let mut default_stops = None;
    for p in &mut state.paragraphs {
        if p.tab_stops.is_none() && p.runs.iter().any(|r| r.text.contains('\t')) {
            p.tab_stops = Some(default_stops.get_or_insert_with(crate::tab_stops::default_tab_stops).clone());
        }
    }
    // Direction (`crate::paragraph_direction`): the base level from `dir` or the first strong character; start / end.
    for p in &mut state.paragraphs {
        crate::paragraph_direction::resolve_paragraph_direction(p);
    }
    ParsedText { paragraphs: state.paragraphs, source_text: state.source_text }
}

/// `walk` (parser.ts:219-325). `currentPara` is an index into
/// `state.paragraphs` (paragraphs are pushed on creation, so indices are
/// stable references).
fn walk(
    state: &mut WalkState,
    node: ego_tree::NodeRef<'_, Node>,
    inherited_style: &InlineStyle,
    current_para: Option<usize>,
    parent_align: Align,
    parent_dir: DeclaredDirection,
) -> Option<usize> {
    match node.value() {
        Node::Text(t) => {
            if sanitize_text(t).is_empty() {
                if current_para.is_none() { state.begin_source_paragraph(); }
                state.source_text.push_str(t);
                state.source_cursor += t.chars().count();
                return current_para;
            }
            let idx = state.ensure_paragraph(current_para, parent_align, parent_dir, inherited_style);
            let (text, source) = sanitize_with_source(t, t.contains('\u{00AD}'), state.source_cursor);
            state.source_text.push_str(t);
            state.source_cursor += t.chars().count();
            if text.is_empty() { return Some(idx); }
            let mut style = inherited_style.clone();
            style.source = source;
            state.paragraphs[idx].runs.push(InlineRun { text, style });
            state.last_observed_style = inherited_style.clone();
            Some(idx)
        }
        Node::Element(el) => {
            let tag = el.name().to_lowercase();

            if tag == "br" {
                let idx =
                    state.ensure_paragraph(current_para, parent_align, parent_dir, inherited_style);
                state.paragraphs[idx].runs.push(InlineRun {
                    text: "\n".to_string(),
                    style: { let mut style = inherited_style.clone();
                        style.source = crate::types::SourceIndices::new(vec![Some(state.source_cursor)]);
                        state.source_cursor += 1;
                        state.source_text.push('\n');
                        style },
                });
                return Some(idx);
            }

            let is_block = is_block_tag(&tag);
            let style_attr = el.attr("style");
            let align = if is_block {
                extract_align_from_style(style_attr)
            } else {
                parent_align
            };
            // `dir` inherits through blocks (HTML); `auto` on a block leaves its paragraphs auto.
            let dir = if is_block {
                match el.attr("dir") {
                    Some(v) => crate::paragraph_direction::parse_dir_attribute(v),
                    None => crate::paragraph_direction::direction_from_style(style_attr)
                        .unwrap_or(parent_dir),
                }
            } else {
                parent_dir
            };
            // An element's `lang` attribute (HTML) sets its language before its own `style` (whose `--nf-lang` wins);
            // `lang=""` declares none.
            let with_lang;
            let inherited_style = match el.attr("lang") {
                Some(v) => {
                    let mut s = inherited_style.clone();
                    s.lang = crate::lang::canonical_lang(v);
                    with_lang = s;
                    &with_lang
                }
                None => inherited_style,
            };
            let style_after_attr = parse_inline_style(style_attr, inherited_style);
            let style_after_tag = if is_inline_tag(&tag) {
apply_tag_style(&tag, &style_after_attr)
            } else {
                style_after_attr
            };

            let mut para = current_para;
            let mut before_count = 0usize;

            // List containers (parser.ts:253-276)
            if tag == "ul" {
                let level = state.list_stack.len() as i64;
                state.list_stack.push(ListCtx {
                    list_type: "ul",
                    level,
                    next_index: None,
                    style: crate::list_markers::declared_list_style(style_attr, el.attr("type"))
                        .unwrap_or_else(|| {
                            crate::list_markers::ListStyle::level_default("ul", level)
                        }),
                });
                for child in node.children() {
                    para = walk(state, child, &style_after_tag, para, align, dir);
                }
                state.list_stack.pop();
                return para;
            }
            if tag == "ol" {
                let mut start_idx = 1i64;
                if let Some(start_attr) = el.attr("start") {
                    if let Some(v) = crate::jsnum::js_parse_int(start_attr) {
                        if v > 0 {
                            start_idx = v;
                        }
                    }
                }
                let level = state.list_stack.len() as i64;
                state.list_stack.push(ListCtx {
                    list_type: "ol",
                    level,
                    next_index: Some(start_idx),
                    style: crate::list_markers::declared_list_style(style_attr, el.attr("type"))
                        .unwrap_or_else(|| {
                            crate::list_markers::ListStyle::level_default("ol", level)
                        }),
                });
                for child in node.children() {
                    para = walk(state, child, &style_after_tag, para, align, dir);
                }
                state.list_stack.pop();
                return para;
            }

            if is_block {
                para = None;
                before_count = state.paragraphs.len();
            }

            // parser.ts:285-290
            if tag == "li" {
                if let Some(top) = state.list_stack.last() {
                    if top.list_type == "ul" {
                        state.pending_list_meta = Some(ListMetadata {
                            kind: "ul".to_string(),
                            level: top.level,
                            index: None,
                            style: top.style,
                        });
                    } else if top.list_type == "ol" {
                        state.pending_list_meta = Some(ListMetadata {
                            kind: "ol".to_string(),
                            level: top.level,
                            index: Some(top.next_index.unwrap_or(1).max(1)),
                            style: top.style,
                        });
                    }
                }
            }

            for child in node.children() {
                para = walk(state, child, &style_after_tag, para, align, dir);
            }

            if is_block {
                // parser.ts:296-315 — empty block still emits a paragraph
                if state.paragraphs.len() == before_count {
                    let mut base = style_after_tag.clone();
                    if tag == "p" {
                        let has_inline_style = style_attr.is_some();
                        if !has_inline_style {
                            base = state.last_observed_style.clone();
                        }
                    } else {
                        base = state.last_observed_style.clone();
                    }
                    let mut p = Paragraph {
                        align,
                        runs: Vec::new(),
                        wrap: base.text_wrap,
                        tab_stops: base.tab_stops.clone(),
                        format: base.paragraph,
                        base_style: Some(base),
                        list_metadata: None,
                        direction: ParagraphDirection {
                            declared: dir,
                            ..Default::default()
                        },
                    };
                    if tag == "li" {
                        if let Some(top) = state.list_stack.last() {
                            if top.list_type == "ul" {
                                p.list_metadata = Some(ListMetadata {
                                    kind: "ul".to_string(),
                                    level: top.level,
                                    index: None,
                                    style: top.style,
                                });
                            } else if top.list_type == "ol" {
                                p.list_metadata = Some(ListMetadata {
                                    kind: "ol".to_string(),
                                    level: top.level,
                                    index: Some(top.next_index.unwrap_or(1).max(1)),
                                    style: top.style,
                                });
                            }
                        }
                    }
                    state.begin_source_paragraph();
                    state.source_reserved = false;
                    state.paragraphs.push(p);
                }
                para = None;
                // parser.ts:319-322 — advance OL index after closing the li
                if tag == "li" {
                    if let Some(top) = state.list_stack.last_mut() {
                        if top.list_type == "ol" {
                            top.next_index = Some(top.next_index.unwrap_or(1) + 1);
                        }
                    }
                }
            }
            para
        }
        _ => current_para, // comments / doctype / PI — parser.ts:234
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tab_with_no_stops_of_its_own_takes_the_default_stops() {
        // parser.ts twin (tabStops.test.ts): only the paragraph with a tab, and never in place of its own stops.
        let paras = parse_rich_text_to_paragraphs("<p>a\tb</p><p>c</p>");
        let stops = paras[0].tab_stops.as_deref().expect("default stops");
        assert!(stops.stops.is_empty());
        assert_eq!(stops.interval, None);
        assert!(paras[1].tab_stops.is_none());
        let own = parse_rich_text_to_paragraphs("<p style=\"--nf-tab-stops: 200 right\">a\tb</p>");
        assert_eq!(own[0].tab_stops.as_deref().unwrap().stops[0].position, 200.0);
        let stop = crate::tab_stops::next_tab_stop(stops, 10.0);
        assert_eq!(stop.position, 36.0);
    }

    #[test]
    fn zwnj_survives_only_where_it_shapes() {
        // parser.ts `keepMeaningfulZwnj` (textBreakRules.test.ts has the same cases).
        assert_eq!(sanitize_text("a\u{200C}b"), "ab");
        assert_eq!(sanitize_text("\u{200C}هفت"), "هفت");
        assert_eq!(sanitize_text("هفت\u{200C}سین"), "هفت\u{200C}سین");
        assert_eq!(sanitize_text("می\u{200C}\u{200C}روم"), "می\u{200C}روم");
        assert_eq!(sanitize_text("क्\u{200C}ष"), "क्\u{200C}ष");
        assert_eq!(sanitize_text("ok\u{200C} هفت"), "ok هفت");
        assert_eq!(sanitize_text_keeping("زنده\u{200C}ی\u{00AD}", true), "زنده\u{200C}ی\u{00AD}");
    }

    #[test]
    fn pt_em_nem_units() {
        // parser.ts:44-47 (pt = px at 72 per inch, em ×inherited), :65-67 (legacy Nem)
        let d = default_style();
        let s = parse_inline_style(Some("font-size: 24pt"), &d);
        assert_eq!(s.font_size, 24.0);
        let s2 = parse_inline_style(Some("font-size: 1.5em"), &s);
        assert_eq!(s2.font_size, 36.0);
        let s3 = parse_inline_style(Some("line-height: 1.4em"), &d);
        assert_eq!(s3.line_height, 1.4);
        let s4 = parse_inline_style(Some("line-height: 24px; font-size: 12px"), &d);
        assert_eq!(s4.line_height, 2.0);
        let s5 = parse_inline_style(Some("line-height: 150%"), &d);
        assert_eq!(s5.line_height, 1.5);
    }

    #[test]
    fn percent_font_size_scales_inherited() {
        // parser.ts:48-51 — `%` resolves against the parent size, like `em`.
        // Dropping it rendered a `font-size: 60%` exponent at the numeral's own size.
        let mut d = default_style();
        d.font_size = 40.0;
        assert_eq!(
            parse_inline_style(Some("font-size: 50%"), &d).font_size,
            20.0
        );
        assert_eq!(
            parse_inline_style(Some("font-size: 62.5%"), &d).font_size,
            25.0
        );
        assert_eq!(
            parse_inline_style(Some("font-size: 150%"), &d).font_size,
            60.0
        );
        // Nested spans compound: 40 → 80 → 20.
        let outer = parse_inline_style(Some("font-size: 200%"), &d);
        assert_eq!(outer.font_size, 80.0);
        assert_eq!(
            parse_inline_style(Some("font-size: 25%"), &outer).font_size,
            20.0
        );
        // An unparseable value still leaves the inherited size untouched.
        assert_eq!(
            parse_inline_style(Some("font-size: xx-large"), &d).font_size,
            40.0
        );
        assert_eq!(
            parse_inline_style(Some("font-size: 12rem"), &d).font_size,
            40.0
        );
    }

    #[test]
    fn percent_font_size_through_paragraph_parse() {
        let paras = parse_rich_text_to_paragraphs(
            "<p style=\"font-size: 40px\">100 m<span style=\"font-size: 50%\">2</span></p>",
        );
        let runs = &paras[0].runs;
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].style.font_size, 40.0);
        assert_eq!(runs[1].style.font_size, 20.0);
    }

    #[test]
    fn b_only_bumps_400() {
        // parser.ts:157-160
        let d = default_style();
        let bolded = apply_tag_style("b", &d);
        assert_eq!(bolded.font_weight, "700");
        let mut light = d.clone();
        light.font_weight = "300".to_string();
        let still_light = apply_tag_style("b", &light);
        assert_eq!(still_light.font_weight, "300");
    }

    #[test]
    fn anchor_draws_no_decoration_of_its_own() {
        // As in Figma: a link is underlined only when its text is styled so.
        let plain = parse_rich_text_to_paragraphs("<p><a href=\"https://x.test\">Click</a></p>");
        assert_eq!(plain[0].runs[0].style.text_decoration.as_deref(), Some("none"));

        let styled = parse_rich_text_to_paragraphs(
            "<p><a href=\"https://x.test\"><span style=\"text-decoration: underline\">Click</span></a></p>",
        );
        assert_eq!(styled[0].runs[0].style.text_decoration.as_deref(), Some("underline"));

        let on_anchor = parse_rich_text_to_paragraphs(
            "<p><a href=\"https://x.test\" style=\"text-decoration: underline; font-weight: 700\">Click</a></p>",
        );
        assert_eq!(on_anchor[0].runs[0].style.text_decoration.as_deref(), Some("underline"));
        assert_eq!(on_anchor[0].runs[0].style.font_weight, "700");
    }

    #[test]
    fn br_and_zero_width_break_survive() {
        // parser.ts:238-242, :192-194
        let paras = parse_rich_text_to_paragraphs("<p>a<br>b\u{200B}c</p>");
        assert_eq!(paras.len(), 1);
        let texts: Vec<&str> = paras[0].runs.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(texts, vec!["a", "\n", "b\u{200B}c"]);
    }

    #[test]
    fn list_stack_and_ol_start() {
        // parser.ts:253-322
        let paras = parse_rich_text_to_paragraphs(
            "<ul><li>a</li><ul><li>b</li></ul></ul><ol start=\"3\"><li>c</li><li>d</li></ol>",
        );
        assert_eq!(paras.len(), 4);
        let m0 = paras[0].list_metadata.as_ref().unwrap();
        assert_eq!((m0.kind.as_str(), m0.level), ("ul", 0));
        let m1 = paras[1].list_metadata.as_ref().unwrap();
        assert_eq!((m1.kind.as_str(), m1.level), ("ul", 1));
        let m2 = paras[2].list_metadata.as_ref().unwrap();
        assert_eq!((m2.kind.as_str(), m2.level, m2.index), ("ol", 0, Some(3)));
        let m3 = paras[3].list_metadata.as_ref().unwrap();
        assert_eq!(m3.index, Some(4));
        // E7.16: level defaults (disc, circle; decimal) and nothing inherited across lists.
        use crate::list_markers::ListStyle;
        assert_eq!(m0.style, ListStyle::Disc);
        assert_eq!(m1.style, ListStyle::Circle);
        assert_eq!(m2.style, ListStyle::Decimal);
    }

    #[test]
    fn list_styles_and_nested_numbering() {
        use crate::list_markers::ListStyle;
        let paras = parse_rich_text_to_paragraphs(concat!(
            "<ol style=\"list-style-type: upper-roman\"><li><p>A</p>",
            "<ol><li><p>x</p></li><li><p>y</p></li></ol></li>",
            "<li><p>B</p><ul type=\"square\"><li><p>s</p></li></ul></li></ol>",
            "<ol type=\"a\"><li><p>z</p></li></ol>"
        ));
        let got: Vec<(i64, Option<i64>, ListStyle)> = paras
            .iter()
            .map(|p| {
                let m = p.list_metadata.as_ref().unwrap();
                (m.level, m.index, m.style)
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (0, Some(1), ListStyle::UpperRoman),
                (1, Some(1), ListStyle::LowerAlpha),
                (1, Some(2), ListStyle::LowerAlpha),
                (0, Some(2), ListStyle::UpperRoman),
                (1, None, ListStyle::Square),
                (0, Some(1), ListStyle::LowerAlpha),
            ]
        );
    }

    #[test]
    fn empty_p_inherits_last_observed_style() {
        // parser.ts:296-315
        let paras = parse_rich_text_to_paragraphs("<p style=\"font-size: 40px\">x</p><p></p>");
        assert_eq!(paras.len(), 2);
        let base = paras[1].base_style.as_ref().unwrap();
        assert_eq!(base.font_size, 40.0);
    }
}
