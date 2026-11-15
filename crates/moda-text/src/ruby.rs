//! Ruby (furigana): small annotation text set over its base text (to the right of it in vertical text), as
//! Illustrator's and InDesign's Ruby and HTML's `<ruby>` / `<rt>`.
//!
//! A run stores it as `--nf-*` custom properties in the rich-text HTML's `<span style>` (markup `<ruby>東京<rt>とう
//! きょう</rt></ruby>` and `ruby="…"` on a span become this):
//!
//! ```text
//! --nf-ruby: 'とう|きょう'       the annotation; `|` splits it per base character (jukugo / mono ruby)
//! --nf-ruby-align: center         center (the default) | distribute (spread 1:2:1 over a wider base)
//! --nf-ruby-size: 50%             the annotation's size, a share of the base's (the default 50%)
//! ```
//!
//! How a run with ruby lays out and draws ([`layout`]):
//!
//! - The base run stays on one line (it is one unit of the wrap).
//! - Group ruby (no `|`, or segments that don't match the base's characters): an annotation no wider than its base is
//!   centred over it (`center`) or spread over it with 1:2:1 spacing (`distribute`: half a gap at either end, a whole
//!   gap between characters). A wider annotation spreads the base instead: each base character gets the same space
//!   on either side (1:2:1 again), so the base is exactly as wide as the annotation.
//! - Jukugo ruby (`|` segments, one per base character): when every segment fits over its character, each sits over
//!   its own character (centred or spread); when one doesn't, the whole word's annotation is set as group ruby.
//! - Widths: base characters by their shaped advances, annotation characters by their advances at the ruby size
//!   (vertical advances in vertical text).
//! - Placement: in horizontal text the annotation's baseline is its descent above the top of the base's em box (the
//!   base's ascender above its baseline); in vertical text it is a column of upright glyphs on the base column's right,
//!   their em centred half a base em plus half a ruby em right of the column's centre line. The line's height doesn't
//!   grow: the annotation sits in the leading (give the lines room with the line height).
//! - The annotation is drawn in the base run's font, colour and fill.
//!
//! Mirrored in `utils/richtext/ruby.ts`; the parse and [`layout`] are pinned against each other by
//! `native/fixtures/vertical-text/cases.json` (`ruby`).

use std::sync::Arc;

use crate::vector::measure::ShapedGlyph;
use crate::vector::otjs::OtjsFont;

pub const RUBY_PROPERTY: &str = "--nf-ruby";
pub const RUBY_ALIGN_PROPERTY: &str = "--nf-ruby-align";
pub const RUBY_SIZE_PROPERTY: &str = "--nf-ruby-size";
/// The annotation's size, a share of the base's.
pub const DEFAULT_RUBY_SIZE: f64 = 0.5;
/// The annotation's size is kept to 10%–100% of the base's.
pub const MIN_RUBY_SIZE: f64 = 0.1;
pub const MAX_RUBY_SIZE: f64 = 1.0;

/// How an annotation narrower than its base sits over it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RubyAlign {
    #[default]
    Center,
    /// 1:2:1: half a gap at either end, a whole gap between characters.
    Distribute,
}

impl RubyAlign {
    pub fn parse(value: &str) -> Option<RubyAlign> {
        match value.trim().to_ascii_lowercase().as_str() {
            "center" | "centre" => Some(RubyAlign::Center),
            "distribute" | "space-around" | "1-2-1" => Some(RubyAlign::Distribute),
            _ => None,
        }
    }

    pub fn css(self) -> &'static str {
        match self {
            RubyAlign::Center => "center",
            RubyAlign::Distribute => "distribute",
        }
    }
}

/// A run's ruby. [`RubyStyle::default`] is every run without one.
#[derive(Debug, Clone, PartialEq)]
pub struct RubyStyle {
    /// The annotation, `|` between per-character segments; `None`: no ruby.
    pub text: Option<Arc<str>>,
    pub align: RubyAlign,
    /// A share of the base's size.
    pub size: f64,
}

impl Default for RubyStyle {
    fn default() -> Self {
        RubyStyle {
            text: None,
            align: RubyAlign::Center,
            size: DEFAULT_RUBY_SIZE,
        }
    }
}

/// Decode a CSS string: quoted (`'…'` or `"…"`) with `\HH ` hex escapes, or bare.
pub fn parse_css_string(value: &str) -> String {
    let v = value.trim();
    let inner = if v.len() >= 2
        && ((v.starts_with('\'') && v.ends_with('\'')) || (v.starts_with('"') && v.ends_with('"')))
    {
        &v[1..v.len() - 1]
    } else {
        v
    };
    let mut out = String::with_capacity(inner.len());
    let chars: Vec<char> = inner.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && i + 1 < chars.len() {
            let mut j = i + 1;
            while j < chars.len() && j < i + 7 && chars[j].is_ascii_hexdigit() {
                j += 1;
            }
            if j > i + 1 {
                let hex: String = chars[i + 1..j].iter().collect();
                if let Some(c) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                    out.push(c);
                }
                // One whitespace after a hex escape belongs to it.
                if j < chars.len() && chars[j] == ' ' {
                    j += 1;
                }
                i = j;
                continue;
            }
            out.push(chars[i + 1]);
            i += 2;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// A CSS string for `text`, single-quoted, with the characters that could end a declaration or break out of an
/// attribute hex-escaped (`'` `"` `\` `;` `:` `<` `>` `&`).
pub fn css_string(text: &str) -> String {
    let mut out = String::from("'");
    for c in text.chars() {
        if matches!(c, '\'' | '"' | '\\' | ';' | ':' | '<' | '>' | '&') {
            out.push_str(&format!("\\{:x} ", c as u32));
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

impl RubyStyle {
    /// Apply one CSS declaration; false when `prop` isn't one of ours. An invalid value keeps the inherited one.
    pub fn apply_declaration(&mut self, prop: &str, value: &str) -> bool {
        match prop {
            RUBY_PROPERTY => {
                let v = value.trim();
                if v.eq_ignore_ascii_case("none") {
                    self.text = None;
                } else {
                    let text = parse_css_string(v);
                    self.text = if text.trim().is_empty() { None } else { Some(Arc::from(text.as_str())) };
                }
            }
            RUBY_ALIGN_PROPERTY => {
                if let Some(a) = RubyAlign::parse(value) {
                    self.align = a;
                }
            }
            RUBY_SIZE_PROPERTY => {
                if let Some(s) = crate::character::parse_scale(value) {
                    self.size = s.clamp(MIN_RUBY_SIZE, MAX_RUBY_SIZE);
                }
            }
            _ => return false,
        }
        true
    }

    /// The non-default values as CSS declarations, for exporters and dumps.
    pub fn css_declarations(&self) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        if let Some(t) = &self.text {
            out.push((RUBY_PROPERTY, css_string(t)));
        }
        if self.align != RubyAlign::Center {
            out.push((RUBY_ALIGN_PROPERTY, self.align.css().to_string()));
        }
        if self.size != DEFAULT_RUBY_SIZE {
            out.push((RUBY_SIZE_PROPERTY, format!("{}%", crate::character::css_number(self.size * 100.0))));
        }
        out
    }

    /// The annotation's per-character segments (one when it has no `|`); empty without ruby.
    pub fn segments(&self) -> Vec<String> {
        match &self.text {
            Some(t) => t.split('|').map(str::to_string).collect(),
            None => Vec::new(),
        }
    }

    /// The annotation without its `|` separators.
    pub fn plain(&self) -> String {
        self.text.as_deref().unwrap_or("").replace('|', "")
    }

    /// A measurement-cache key part ('' without ruby).
    pub fn cache_key(&self) -> String {
        match &self.text {
            Some(t) => format!("|rb:{}:{}:{}", t, self.align.css(), self.size),
            None => String::new(),
        }
    }
}

/// Where an annotation goes over its base: each base character's extra space on either side, and each annotation
/// character's start, from the base run's start (before that space). Units are the caller's (font units or px).
#[derive(Debug, Clone, PartialEq)]
pub struct RubyLayout {
    pub base_pad: f64,
    pub ruby_x: Vec<f64>,
}

/// `chars` set in `width` from `origin`: centred, or spread 1:2:1.
fn spread(origin: f64, width: f64, chars: &[f64], align: RubyAlign, out: &mut Vec<f64>) {
    let total: f64 = chars.iter().sum();
    let mut x = origin;
    match align {
        RubyAlign::Center => {
            x += (width - total) * 0.5;
            for w in chars {
                out.push(x);
                x += w;
            }
        }
        RubyAlign::Distribute => {
            let gap = if chars.is_empty() { 0.0 } else { (width - total) / chars.len() as f64 };
            x += gap * 0.5;
            for w in chars {
                out.push(x);
                x += w + gap;
            }
        }
    }
}

/// Lay an annotation out over its base (see the module docs): `base` the base characters' widths (clusters, in
/// order), `segments` the annotation's characters' widths, per `|` segment.
pub fn layout(base: &[f64], segments: &[Vec<f64>], align: RubyAlign) -> RubyLayout {
    let n = base.len();
    let chars: Vec<f64> = segments.iter().flatten().copied().collect();
    let mut ruby_x = Vec::with_capacity(chars.len());
    if n == 0 {
        return RubyLayout {
            base_pad: 0.0,
            ruby_x: vec![0.0; chars.len()],
        };
    }
    // Jukugo: a segment over each base character, when every one fits.
    if segments.len() == n && n > 1 && segments.iter().zip(base).all(|(s, w)| s.iter().sum::<f64>() <= w + 1e-9) {
        let mut origin = 0.0;
        for (s, w) in segments.iter().zip(base) {
            spread(origin, *w, s, align, &mut ruby_x);
            origin += w;
        }
        return RubyLayout { base_pad: 0.0, ruby_x };
    }
    let b: f64 = base.iter().sum();
    let r: f64 = chars.iter().sum();
    if r <= b + 1e-9 {
        spread(0.0, b, &chars, align, &mut ruby_x);
        return RubyLayout { base_pad: 0.0, ruby_x };
    }
    // A wider annotation spreads the base, 1:2:1.
    spread(0.0, r, &chars, RubyAlign::Center, &mut ruby_x);
    RubyLayout {
        base_pad: (r - b) / (2.0 * n as f64),
        ruby_x,
    }
}

/// The base's character (cluster) widths, font units, from its shaped glyphs.
fn cluster_widths(shaped: &[ShapedGlyph]) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::new();
    for (i, g) in shaped.iter().enumerate() {
        if i == 0 || shaped[i - 1].cluster != g.cluster {
            out.push(0.0);
        }
        *out.last_mut().unwrap() += g.x_advance;
    }
    out
}

/// The annotation's characters' widths at its size, base font units: horizontal advances, or vertical ones in
/// vertical text.
fn ruby_widths(font: &OtjsFont, style: &crate::types::InlineStyle) -> Vec<Vec<f64>> {
    let vertical = style.vertical.glyph.is_some();
    let vm = font.vertical_metrics();
    style
        .ruby
        .segments()
        .iter()
        .map(|seg| {
            seg.chars()
                .map(|ch| {
                    let gid = font.char_to_glyph_index(ch);
                    let adv = match (vertical, vm) {
                        (true, Some(m)) => m.advance(gid),
                        _ => font.advance_width(gid),
                    };
                    adv * style.ruby.size
                })
                .collect()
        })
        .collect()
}

/// The ruby layout of a base run from its shaped glyphs as shaped without the ruby (font units); `None` without ruby.
pub fn plan(font: &OtjsFont, shaped: &[ShapedGlyph], style: &crate::types::InlineStyle) -> Option<RubyLayout> {
    style.ruby.text.as_ref()?;
    Some(layout(&cluster_widths(shaped), &ruby_widths(font, style), style.ruby.align))
}

/// Give a base run the space its ruby needs: each character `base_pad` on either side. A no-op without ruby (or
/// when the annotation fits), so every caller can run it unconditionally after the other advance changes.
pub fn adjust_shaped(font: &OtjsFont, shaped: &mut [ShapedGlyph], style: &crate::types::InlineStyle) {
    let Some(plan) = plan(font, shaped, style) else { return };
    if plan.base_pad <= 0.0 {
        return;
    }
    for i in 0..shaped.len() {
        if i == 0 || shaped[i - 1].cluster != shaped[i].cluster {
            shaped[i].x_offset += plan.base_pad;
        }
        if i + 1 == shaped.len() || shaped[i + 1].cluster != shaped[i].cluster {
            shaped[i].x_advance += 2.0 * plan.base_pad;
        }
    }
}

/// The annotation's run style: the base's at the ruby size, no letter spacing or character attributes, upright in
/// vertical text.
pub fn annotation_style(base: &crate::types::InlineStyle, vertical: bool) -> crate::types::InlineStyle {
    let mut s = base.without_source();
    s.ruby = RubyStyle::default();
    s.font_size = base.font_size * base.ruby.size;
    s.small_caps_full_size = None;
    s.letter_spacing = 0.0;
    s.character = crate::character::CharacterStyle::DEFAULT;
    s.vertical.glyph = vertical.then_some(crate::vertical::GlyphOrient::Upright);
    s.vertical.combine = crate::vertical::TextCombine::None;
    s
}

/// Where the annotation's baseline is in horizontal text: its descent above the top of the base's em box.
pub fn baseline_above(base_baseline_y: f64, base_ascent_px: f64, ruby_descent_px: f64) -> f64 {
    base_baseline_y - base_ascent_px - ruby_descent_px
}

/// How far right of the base column's centre line the annotation's column centre is in vertical text.
pub fn column_offset(base_size_px: f64, ruby_size_px: f64) -> f64 {
    (base_size_px + ruby_size_px) * 0.5
}
