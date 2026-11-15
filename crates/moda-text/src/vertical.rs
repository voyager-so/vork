//! Vertical text (Illustrator's Vertical Type, CSS `writing-mode: vertical-rl | vertical-lr`): lines run top to
//! bottom and stack right to left (`vertical-rl`) or left to right (`vertical-lr`).
//!
//! A run stores it as CSS in the rich-text HTML, inherited like any other run style:
//!
//! ```text
//! writing-mode: vertical-rl         horizontal-tb (the default) | vertical-rl | vertical-lr
//! text-orientation: mixed           mixed (the default) | upright | sideways
//! text-combine-upright: all         none (the default) | all | digits [2-4]  (tate-chu-yoko)
//! ```
//!
//! The node's writing mode is its first run's. How a vertical node lays out and draws:
//!
//! - The text wraps as horizontal text would in a box as wide as the node is tall (the inline size), so line breaking,
//!   the composer, alignment and justification are the horizontal ones. Each line then becomes a column: line `k`'s
//!   box (top `t`, height `h` in the horizontal pass) is the column from `x = W − t − h` to `W − t` in a
//!   `vertical-rl` node `W` wide, or from `t` to `t + h` in a `vertical-lr` one; the inline position along the line is
//!   the distance down the column ([`ColumnFrame`]).
//! - [`split_runs`] cuts each run into pieces drawn upright, sideways or combined ([`GlyphOrient`]). With
//!   `text-orientation: mixed` a character is upright when its Unicode vertical orientation (UAX #50) is U or Tu / Tr
//!   ([`is_upright`]: CJK, kana, Hangul, full-width forms, CJK punctuation and symbols) and sideways otherwise (Latin,
//!   digits, half-width forms); `upright` and `sideways` set every character. A `text-combine-upright` run (all of it,
//!   or each run of at most N ASCII digits for `digits N`) is combined: set horizontally, upright, in one em.
//! - Upright pieces shape with the OpenType `vert` feature (vertical alternates: the corner-hugging 、。, the turned
//!   ー and brackets 「」, the vertical small kana) and advance by the glyph's vertical advance ([`VerticalMetrics`]:
//!   `vmtx`, else the font's ascender − descender); each glyph is centred on the column's centre line and its top is
//!   its vertical origin above the baseline (`VORG`, else the ascender). Sideways pieces shape as horizontal text and
//!   turn 90° clockwise, their em box (ascender to descender) centred on the column. A combined piece advances one em
//!   down the column; its horizontal text is centred on the column and on that em, squeezed to one em wide when wider.
//! - A baseline shift moves glyphs toward the column's right (positive) or left.
//!
//! Mirrored in `utils/richtext/verticalText.ts`: the parse, [`is_upright`], [`split_text`], the metrics fallbacks and
//! [`place`] are pinned against each other by `native/fixtures/vertical-text/cases.json`.

use crate::font_features::FontFeature;
use crate::vector::measure::ShapedGlyph;
use crate::vector::otjs::OtjsFont;

pub const WRITING_MODE_PROPERTY: &str = "writing-mode";
pub const TEXT_ORIENTATION_PROPERTY: &str = "text-orientation";
pub const TEXT_COMBINE_UPRIGHT_PROPERTY: &str = "text-combine-upright";

/// CSS `writing-mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WritingMode {
    #[default]
    HorizontalTb,
    VerticalRl,
    VerticalLr,
}

impl WritingMode {
    /// CSS values, plus the SVG 1.1 / legacy ones (`tb-rl`, `tb`, `lr-tb` …).
    pub fn parse(value: &str) -> Option<WritingMode> {
        match value.trim().to_ascii_lowercase().as_str() {
            "horizontal-tb" | "lr" | "lr-tb" | "rl" | "rl-tb" | "horizontal" => Some(WritingMode::HorizontalTb),
            "vertical-rl" | "tb" | "tb-rl" | "vertical" => Some(WritingMode::VerticalRl),
            "vertical-lr" | "tb-lr" => Some(WritingMode::VerticalLr),
            _ => None,
        }
    }

    pub fn css(self) -> &'static str {
        match self {
            WritingMode::HorizontalTb => "horizontal-tb",
            WritingMode::VerticalRl => "vertical-rl",
            WritingMode::VerticalLr => "vertical-lr",
        }
    }

    pub fn is_vertical(self) -> bool {
        self != WritingMode::HorizontalTb
    }
}

/// CSS `text-orientation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextOrientation {
    #[default]
    Mixed,
    Upright,
    Sideways,
}

impl TextOrientation {
    pub fn parse(value: &str) -> Option<TextOrientation> {
        match value.trim().to_ascii_lowercase().as_str() {
            "mixed" => Some(TextOrientation::Mixed),
            "upright" => Some(TextOrientation::Upright),
            "sideways" | "sideways-right" => Some(TextOrientation::Sideways),
            _ => None,
        }
    }

    pub fn css(self) -> &'static str {
        match self {
            TextOrientation::Mixed => "mixed",
            TextOrientation::Upright => "upright",
            TextOrientation::Sideways => "sideways",
        }
    }
}

/// CSS `text-combine-upright` (tate-chu-yoko).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextCombine {
    #[default]
    None,
    /// The whole run, in one em.
    All,
    /// Each run of at most this many ASCII digits (2–4).
    Digits(u8),
}

impl TextCombine {
    pub fn parse(value: &str) -> Option<TextCombine> {
        let v = value.trim().to_ascii_lowercase();
        let mut words = v.split_whitespace();
        match words.next()? {
            "none" => Some(TextCombine::None),
            "all" => Some(TextCombine::All),
            "digits" => match words.next() {
                None => Some(TextCombine::Digits(2)),
                Some(n) => match n.parse::<u8>() {
                    Ok(n) if (2..=4).contains(&n) => Some(TextCombine::Digits(n)),
                    _ => None,
                },
            },
            _ => None,
        }
    }

    pub fn css(self) -> String {
        match self {
            TextCombine::None => "none".to_string(),
            TextCombine::All => "all".to_string(),
            TextCombine::Digits(n) => format!("digits {n}"),
        }
    }
}

/// How a piece of a vertical line's text is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlyphOrient {
    Upright,
    Sideways,
    Combined,
}

impl GlyphOrient {
    pub fn name(self) -> &'static str {
        match self {
            GlyphOrient::Upright => "upright",
            GlyphOrient::Sideways => "sideways",
            GlyphOrient::Combined => "combined",
        }
    }
}

/// A run's vertical-text styles. [`VerticalStyle::DEFAULT`] is every horizontal run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VerticalStyle {
    pub writing_mode: WritingMode,
    pub orientation: TextOrientation,
    pub combine: TextCombine,
    /// Set by [`split_runs`] on the runs of a vertical node: how this piece draws. `None` on horizontal text.
    pub glyph: Option<GlyphOrient>,
}

impl VerticalStyle {
    pub const DEFAULT: VerticalStyle = VerticalStyle {
        writing_mode: WritingMode::HorizontalTb,
        orientation: TextOrientation::Mixed,
        combine: TextCombine::None,
        glyph: None,
    };

    /// Apply one CSS declaration; false when `prop` isn't one of ours. An invalid value keeps the inherited one.
    pub fn apply_declaration(&mut self, prop: &str, value: &str) -> bool {
        match prop {
            WRITING_MODE_PROPERTY => {
                if let Some(m) = WritingMode::parse(value) {
                    self.writing_mode = m;
                }
            }
            TEXT_ORIENTATION_PROPERTY => {
                if let Some(o) = TextOrientation::parse(value) {
                    self.orientation = o;
                }
            }
            TEXT_COMBINE_UPRIGHT_PROPERTY => {
                if let Some(c) = TextCombine::parse(value) {
                    self.combine = c;
                }
            }
            _ => return false,
        }
        true
    }

    /// The non-default styles as CSS declarations, for exporters and dumps.
    pub fn css_declarations(&self) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        if self.writing_mode != WritingMode::HorizontalTb {
            out.push((WRITING_MODE_PROPERTY, self.writing_mode.css().to_string()));
        }
        if self.orientation != TextOrientation::Mixed {
            out.push((TEXT_ORIENTATION_PROPERTY, self.orientation.css().to_string()));
        }
        if self.combine != TextCombine::None {
            out.push((TEXT_COMBINE_UPRIGHT_PROPERTY, self.combine.css()));
        }
        out
    }

    /// Whether the run's text must stay on one line (a combined piece is one em of the column).
    pub fn unbreakable(&self) -> bool {
        self.glyph == Some(GlyphOrient::Combined)
    }

    /// A key for measurement caches: how this run shapes differently from horizontal text. Empty for horizontal text.
    pub fn cache_key(&self) -> &'static str {
        match self.glyph {
            None => "",
            Some(GlyphOrient::Upright) => "|v:u",
            Some(GlyphOrient::Sideways) => "|v:s",
            Some(GlyphOrient::Combined) => "|v:c",
        }
    }
}

/// Code points drawn upright in mixed orientation: Unicode vertical orientation (UAX #50) U, Tu and Tr. The Tr ones
/// (brackets, the long-vowel mark, the wave dash) take their `vert` alternate. Inclusive ranges, sorted.
const UPRIGHT_RANGES: &[(u32, u32)] = &[
    (0x00A7, 0x00A7),
    (0x00A9, 0x00A9),
    (0x00AE, 0x00AE),
    (0x00B1, 0x00B1),
    (0x00BC, 0x00BE),
    (0x00D7, 0x00D7),
    (0x00F7, 0x00F7),
    (0x02EA, 0x02EB),
    (0x1100, 0x11FF),
    (0x1401, 0x167F),
    (0x18B0, 0x18FF),
    (0x2016, 0x2016),
    (0x2020, 0x2021),
    (0x2030, 0x2031),
    (0x203B, 0x203C),
    (0x2042, 0x2042),
    (0x2047, 0x2049),
    (0x2051, 0x2051),
    (0x2100, 0x2101),
    (0x2103, 0x2109),
    (0x210F, 0x210F),
    (0x2113, 0x2114),
    (0x2116, 0x2117),
    (0x211E, 0x2123),
    (0x2125, 0x2125),
    (0x2127, 0x2127),
    (0x2129, 0x2129),
    (0x212E, 0x212E),
    (0x2135, 0x213F),
    (0x2145, 0x214A),
    (0x214C, 0x214D),
    (0x214F, 0x2189),
    (0x218C, 0x218F),
    (0x221E, 0x221E),
    (0x2234, 0x2235),
    (0x2300, 0x2307),
    (0x230C, 0x231F),
    (0x2324, 0x2328),
    (0x232B, 0x232B),
    (0x237D, 0x239A),
    (0x23BE, 0x23CD),
    (0x23CF, 0x23CF),
    (0x23D1, 0x23DB),
    (0x23E2, 0x2422),
    (0x2424, 0x24FF),
    (0x25A0, 0x2619),
    (0x2620, 0x2767),
    (0x2776, 0x2793),
    (0x2B12, 0x2B2F),
    (0x2B50, 0x2B59),
    (0x2BB8, 0x2BFF),
    (0x2E80, 0xA4CF),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7FF),
    (0xE000, 0xFAFF),
    (0xFE10, 0xFE1F),
    (0xFE30, 0xFE6F),
    (0xFF01, 0xFF60),
    (0xFFE0, 0xFFE7),
    (0x1F000, 0x1FAFF),
    (0x20000, 0x3FFFD),
];

/// Whether `ch` is drawn upright in a vertical line with `text-orientation: mixed` (UAX #50 U / Tu / Tr).
pub fn is_upright(ch: char) -> bool {
    let cp = ch as u32;
    // Binary search over the sorted, disjoint ranges.
    let mut lo = 0usize;
    let mut hi = UPRIGHT_RANGES.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let (s, e) = UPRIGHT_RANGES[mid];
        if cp < s {
            hi = mid;
        } else if cp > e {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    false
}

fn is_space(ch: char) -> bool {
    matches!(ch, ' ' | '\t' | '\u{00A0}' | '\n')
}

/// Cut a vertical run's text into pieces by how they draw, in order (concatenating back to `text`). Spaces join the
/// piece before them (a leading space the piece after it), so a space never becomes a piece of its own.
pub fn split_text(text: &str, style: &VerticalStyle) -> Vec<(String, GlyphOrient)> {
    if text.is_empty() {
        return Vec::new();
    }
    if style.combine == TextCombine::All {
        return vec![(text.to_string(), GlyphOrient::Combined)];
    }
    let chars: Vec<char> = text.chars().collect();
    let base = |ch: char| match style.orientation {
        TextOrientation::Upright => GlyphOrient::Upright,
        TextOrientation::Sideways => GlyphOrient::Sideways,
        TextOrientation::Mixed => {
            if is_upright(ch) || ch == crate::runs::EMOJI_PLACEHOLDER {
                GlyphOrient::Upright
            } else {
                GlyphOrient::Sideways
            }
        }
    };
    let mut kinds: Vec<Option<GlyphOrient>> =
        chars.iter().map(|&c| if is_space(c) { None } else { Some(base(c)) }).collect();
    // `digits N`: a run of 1..=N ASCII digits with no digit on either side.
    if let TextCombine::Digits(n) = style.combine {
        let mut i = 0usize;
        while i < chars.len() {
            if chars[i].is_ascii_digit() {
                let mut j = i;
                while j < chars.len() && chars[j].is_ascii_digit() {
                    j += 1;
                }
                if j - i <= n as usize {
                    for k in kinds.iter_mut().take(j).skip(i) {
                        *k = Some(GlyphOrient::Combined);
                    }
                }
                i = j;
            } else {
                i += 1;
            }
        }
    }
    // Spaces take the kind before them (the first non-space's when leading).
    let first = kinds.iter().flatten().next().copied().unwrap_or(GlyphOrient::Sideways);
    let mut prev = first;
    for k in kinds.iter_mut() {
        match k {
            Some(o) => prev = *o,
            None => *k = Some(prev),
        }
    }
    let mut out: Vec<(String, GlyphOrient)> = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let kind = kinds[i].unwrap();
        let mut j = i + 1;
        // Each combined run of digits is its own piece (two numbers a space apart are two ems).
        while j < chars.len() && kinds[j] == Some(kind) {
            if kind == GlyphOrient::Combined && chars[j].is_ascii_digit() != chars[j - 1].is_ascii_digit() {
                break;
            }
            j += 1;
        }
        out.push((chars[i..j].iter().collect(), kind));
        i = j;
    }
    // A space that ended up in a combined digits piece moves to its own sideways piece.
    if matches!(style.combine, TextCombine::Digits(_)) {
        out = out
            .into_iter()
            .map(|(t, k)| {
                if k == GlyphOrient::Combined && !t.chars().any(|c| c.is_ascii_digit()) {
                    (t, GlyphOrient::Sideways)
                } else {
                    (t, k)
                }
            })
            .collect();
    }
    out
}

/// Whether rich-text HTML has vertical text or ruby anywhere: glyphs placed off a horizontal line, which exporters
/// that write live text from the horizontal runs (`vector::extract`) can't place. A cheap probe on the stored CSS.
pub fn html_has_placed_glyphs(html: &str) -> bool {
    if html.contains(crate::ruby::RUBY_PROPERTY) {
        return true;
    }
    let lower = html.to_ascii_lowercase();
    lower.match_indices(WRITING_MODE_PROPERTY).any(|(i, _)| {
        let rest = lower[i + WRITING_MODE_PROPERTY.len()..].trim_start();
        let Some(rest) = rest.strip_prefix(':') else { return false };
        let value: String = rest.trim_start().chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '-').collect();
        WritingMode::parse(&value).is_some_and(|m| m.is_vertical())
    })
}

/// The writing mode of a node's runs: its first text run's.
pub fn runs_writing_mode(runs: &[crate::runs::MsdfRun]) -> WritingMode {
    runs.iter()
        .find(|r| r.kind == crate::runs::RunKind::Text)
        .or(runs.first())
        .map(|r| r.vertical.writing_mode)
        .unwrap_or_default()
}

/// The runs of a node laid out in `mode`: for a vertical mode, each run cut into [`split_text`] pieces with
/// [`VerticalStyle::glyph`] set; horizontal runs come back unchanged (and with no glyph orientation).
pub fn split_runs(runs: &[crate::runs::MsdfRun], mode: WritingMode) -> Vec<crate::runs::MsdfRun> {
    if !mode.is_vertical() {
        return runs.to_vec();
    }
    let mut out = Vec::with_capacity(runs.len());
    for run in runs {
        if run.kind == crate::runs::RunKind::Emoji {
            let mut r = run.clone();
            r.vertical.writing_mode = mode;
            r.vertical.glyph = Some(GlyphOrient::Upright);
            out.push(r);
            continue;
        }
        let mut style = run.vertical;
        style.writing_mode = mode;
        // A ruby base stays one piece (its annotation spans it): upright when its first character is.
        if run.ruby.text.is_some() {
            let mut r = run.clone();
            let first = split_text(&run.text, &style).first().map(|p| p.1).unwrap_or(GlyphOrient::Upright);
            r.vertical = VerticalStyle {
                glyph: Some(if first == GlyphOrient::Combined { GlyphOrient::Upright } else { first }),
                ..style
            };
            out.push(r);
            continue;
        }
        let mut offset = 0;
        for (text, glyph) in split_text(&run.text, &style) {
            let mut r = run.clone();
            let end = offset + text.chars().count();
            r.source = run.source.slice(offset..end);
            offset = end;
            r.text = text;
            r.vertical = VerticalStyle {
                glyph: Some(glyph),
                ..style
            };
            out.push(r);
        }
    }
    out
}

/// The features a vertical piece adds to its run's: `vert` for upright glyphs (unless the run sets `vert` itself).
pub fn shaping_features(features: &[FontFeature], style: &VerticalStyle) -> Option<Vec<FontFeature>> {
    if style.glyph != Some(GlyphOrient::Upright) || features.iter().any(|f| &f.tag == b"vert") {
        return None;
    }
    let mut out = features.to_vec();
    out.push(FontFeature {
        tag: *b"vert",
        value: 1,
    });
    Some(out)
}

/// A font's vertical metrics, font units: `vmtx` advances, `VORG` origins, and the ascender / descender used where
/// the font has neither (OS/2 typo, else `hhea`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VerticalMetrics {
    pub units_per_em: f64,
    /// `vmtx` advance heights (a glyph past the list reuses the last); empty without `vhea` / `vmtx`.
    pub advances: Vec<u16>,
    /// `VORG`: the default vertical origin and the per-glyph ones.
    pub vorg_default: Option<i16>,
    pub vorg: Vec<(u16, i16)>,
    pub ascender: f64,
    /// Negative below the baseline.
    pub descender: f64,
}

fn rd_u16(d: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*d.get(o)?, *d.get(o + 1)?]))
}
fn rd_i16(d: &[u8], o: usize) -> Option<i16> {
    rd_u16(d, o).map(|v| v as i16)
}
fn rd_u32(d: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_be_bytes([*d.get(o)?, *d.get(o + 1)?, *d.get(o + 2)?, *d.get(o + 3)?]))
}

/// The `[offset, length]` of table `tag` in an sfnt (the first face of a collection).
fn table(data: &[u8], tag: &[u8; 4]) -> Option<(usize, usize)> {
    let mut base = 0usize;
    if data.get(0..4) == Some(b"ttcf") {
        base = rd_u32(data, 12)? as usize;
    }
    let n = rd_u16(data, base + 4)? as usize;
    for i in 0..n {
        let rec = base + 12 + i * 16;
        if data.get(rec..rec + 4)? == tag {
            let off = rd_u32(data, rec + 8)? as usize;
            let len = rd_u32(data, rec + 12)? as usize;
            if off.checked_add(len)? > data.len() {
                return None;
            }
            return Some((off, len));
        }
    }
    None
}

impl VerticalMetrics {
    /// Read from font bytes. `None` when the font has no `head`.
    pub fn parse(data: &[u8]) -> Option<VerticalMetrics> {
        let (head, _) = table(data, b"head")?;
        let upm = rd_u16(data, head + 18).filter(|&u| u != 0).unwrap_or(1000) as f64;
        let mut m = VerticalMetrics {
            units_per_em: upm,
            ascender: upm * 0.88,
            descender: -upm * 0.12,
            ..Default::default()
        };
        let os2 = table(data, b"OS/2").filter(|&(_, len)| len >= 72);
        if let Some((o, _)) = os2 {
            m.ascender = rd_i16(data, o + 68)? as f64;
            m.descender = rd_i16(data, o + 70)? as f64;
        } else if let Some((h, _)) = table(data, b"hhea") {
            m.ascender = rd_i16(data, h + 4)? as f64;
            m.descender = rd_i16(data, h + 6)? as f64;
        }
        if m.ascender - m.descender <= 0.0 {
            m.ascender = upm * 0.88;
            m.descender = -upm * 0.12;
        }
        if let (Some((vhea, vlen)), Some((vmtx, len))) = (table(data, b"vhea"), table(data, b"vmtx")) {
            if vlen >= 36 {
                let count = rd_u16(data, vhea + 34).unwrap_or(0) as usize;
                let count = count.min(len / 4);
                m.advances = (0..count).filter_map(|i| rd_u16(data, vmtx + i * 4)).collect();
            }
        }
        if let Some((vorg, len)) = table(data, b"VORG") {
            if len >= 8 {
                m.vorg_default = rd_i16(data, vorg + 4);
                let n = rd_u16(data, vorg + 6).unwrap_or(0) as usize;
                m.vorg = (0..n)
                    .filter_map(|i| Some((rd_u16(data, vorg + 8 + i * 4)?, rd_i16(data, vorg + 10 + i * 4)?)))
                    .collect();
            }
        }
        Some(m)
    }

    /// The glyph's vertical advance, font units: `vmtx`, else ascender − descender.
    pub fn advance(&self, gid: u32) -> f64 {
        if self.advances.is_empty() {
            return self.ascender - self.descender;
        }
        let i = (gid as usize).min(self.advances.len() - 1);
        self.advances[i] as f64
    }

    /// The glyph's vertical origin: how far above its baseline the top of its vertical advance is, font units
    /// (`VORG`, else the ascender).
    pub fn origin(&self, gid: u32) -> f64 {
        match self.vorg_default {
            Some(d) => match self.vorg.binary_search_by_key(&(gid as u16), |&(g, _)| g) {
                Ok(i) => self.vorg[i].1 as f64,
                Err(_) => d as f64,
            },
            None => self.ascender,
        }
    }
}

/// Turn a vertical piece's shaped glyphs (font units) into its advances down the column: upright glyphs advance by
/// their vertical advance (horizontal offsets dropped: GPOS placed them for a horizontal line); a combined piece is
/// one em, all on its last glyph (every glyph's pen is the piece's start), and one cluster (letter spacing once). Sideways pieces keep their horizontal
/// advances. A no-op for horizontal text.
pub fn adjust_shaped(font: &OtjsFont, shaped: &mut [ShapedGlyph], style: &VerticalStyle) {
    match style.glyph {
        Some(GlyphOrient::Upright) => {
            let Some(vm) = font.vertical_metrics() else { return };
            for g in shaped.iter_mut() {
                g.x_advance = vm.advance(g.glyph);
                g.x_offset = 0.0;
                g.y_offset = 0.0;
            }
        }
        Some(GlyphOrient::Combined) => {
            let upm = if font.units_per_em != 0 {
                font.units_per_em as f64
            } else {
                1000.0
            };
            let first = shaped.first().map(|g| g.cluster).unwrap_or(0);
            let last = shaped.len().saturating_sub(1);
            for (i, g) in shaped.iter_mut().enumerate() {
                // Every glyph keeps the piece's start as its pen; the em is on the last one.
                g.x_advance = if i == last { upm } else { 0.0 };
                g.x_offset = 0.0;
                g.y_offset = 0.0;
                g.cluster = first;
            }
        }
        _ => {}
    }
}

/// Where a line's column is: `mode`, the node's width `block_size` (W), and the line's box in the horizontal pass
/// (`line_top`, `line_height`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColumnFrame {
    pub mode: WritingMode,
    pub block_size: f64,
    pub line_top: f64,
    pub line_height: f64,
}

impl ColumnFrame {
    /// The column's left edge.
    pub fn left(&self) -> f64 {
        match self.mode {
            WritingMode::VerticalLr => self.line_top,
            _ => self.block_size - self.line_top - self.line_height,
        }
    }

    /// The column's centre line.
    pub fn centre_x(&self) -> f64 {
        self.left() + self.line_height * 0.5
    }

    /// The same column moved `d` px toward its over side (the right; ruby sits there).
    pub fn shifted(&self, d: f64) -> ColumnFrame {
        let line_top = match self.mode {
            WritingMode::VerticalLr => self.line_top + d,
            _ => self.line_top - d,
        };
        ColumnFrame { line_top, ..*self }
    }

    /// A horizontal-pass point `(s, y)` of this line, turned into the column (the line's top edge to the column's
    /// right edge, its inline position `s` down the column). Decorations and other line-relative geometry use it.
    pub fn map_point(&self, s: f64, y: f64) -> (f64, f64) {
        (self.left() + self.line_top + self.line_height - y, s)
    }
}

/// One glyph to place in a column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphPlace {
    pub orient: GlyphOrient,
    /// The horizontal pass's pen (the glyph's origin, letter spacing and offsets in), px.
    pub pen_x: f64,
    /// The glyph's baseline in the horizontal pass (baseline shift in), and its line's baseline, px.
    pub glyph_baseline_y: f64,
    pub line_baseline_y: f64,
    pub font_size_px: f64,
    pub units_per_em: f64,
    /// The glyph's horizontal advance, font units (horizontal scale in).
    pub h_advance: f64,
    /// Upright: the glyph's vertical origin, font units ([`VerticalMetrics::origin`]).
    pub v_origin: f64,
    /// The run's ascender and descender, px (both positive).
    pub ascent_px: f64,
    pub descent_px: f64,
    /// Combined: this glyph's horizontal offset in the piece, and the piece's horizontal width, px.
    pub combined_offset_px: f64,
    pub combined_width_px: f64,
}

/// Where the glyph goes: the linear map and the origin, so a horizontal-pass point `q` of the glyph lands at
/// `lin·(q − P0) + P1`, with `P0 = (pen_x, glyph_baseline_y)`. `lin` is row-major `[a, c, b, d]`
/// (`x' = a·x + c·y`, `y' = b·x + d·y`), the renderer's `GlyphXform` convention.
pub fn place(frame: &ColumnFrame, g: &GlyphPlace) -> ([f64; 4], (f64, f64)) {
    let scale = g.font_size_px / if g.units_per_em > 0.0 { g.units_per_em } else { 1000.0 };
    let cx = frame.centre_x();
    // A baseline shift (the glyph above its line's baseline) moves it toward the column's right.
    let shift = g.line_baseline_y - g.glyph_baseline_y;
    match g.orient {
        GlyphOrient::Upright => (
            [1.0, 0.0, 0.0, 1.0],
            (cx - g.h_advance * scale * 0.5 + shift, g.pen_x + g.v_origin * scale),
        ),
        GlyphOrient::Sideways => (
            // 90° clockwise on a y-down screen: (x, y) ↦ (−y, x).
            [0.0, -1.0, 1.0, 0.0],
            (cx - (g.ascent_px - g.descent_px) * 0.5 + shift, g.pen_x),
        ),
        GlyphOrient::Combined => {
            let em = g.font_size_px;
            let squeeze = if g.combined_width_px > em && g.combined_width_px > 0.0 {
                em / g.combined_width_px
            } else {
                1.0
            };
            (
                [squeeze, 0.0, 0.0, 1.0],
                (
                    cx - g.combined_width_px * squeeze * 0.5 + g.combined_offset_px * squeeze + shift,
                    g.pen_x + em * 0.5 + (g.ascent_px - g.descent_px) * 0.5,
                ),
            )
        }
    }
}

/// Compose the column placement onto a glyph's own transform (`lin_x`, `off_x`, about its em-box centre `c`, the
/// renderer's `GlyphXform`): the result maps `q ↦ lin·(lin_x·(q − c) + c + off_x − p0) + p1`, as a new
/// (`lin`, `off`) about the same centre.
pub fn compose(
    lin_x: [f64; 4],
    off_x: (f64, f64),
    c: (f64, f64),
    p0: (f64, f64),
    lin: [f64; 4],
    p1: (f64, f64),
) -> ([f64; 4], (f64, f64)) {
    let [a, cc, b, d] = lin;
    let [xa, xc, xb, xd] = lin_x;
    let out = [a * xa + cc * xb, a * xc + cc * xd, b * xa + d * xb, b * xc + d * xd];
    let (vx, vy) = (c.0 + off_x.0 - p0.0, c.1 + off_x.1 - p0.1);
    let off = (a * vx + cc * vy + p1.0 - c.0, b * vx + d * vy + p1.1 - c.1);
    (out, off)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_three_properties() {
        let mut v = VerticalStyle::DEFAULT;
        assert!(v.apply_declaration(WRITING_MODE_PROPERTY, "vertical-rl"));
        assert_eq!(v.writing_mode, WritingMode::VerticalRl);
        v.apply_declaration(WRITING_MODE_PROPERTY, "sideways-lr");
        assert_eq!(v.writing_mode, WritingMode::VerticalRl, "an unknown value keeps the inherited one");
        v.apply_declaration(TEXT_ORIENTATION_PROPERTY, "upright");
        assert_eq!(v.orientation, TextOrientation::Upright);
        v.apply_declaration(TEXT_COMBINE_UPRIGHT_PROPERTY, "digits 3");
        assert_eq!(v.combine, TextCombine::Digits(3));
        v.apply_declaration(TEXT_COMBINE_UPRIGHT_PROPERTY, "digits 9");
        assert_eq!(v.combine, TextCombine::Digits(3));
        assert!(!v.apply_declaration("color", "red"));
        assert_eq!(
            v.css_declarations(),
            vec![
                (WRITING_MODE_PROPERTY, "vertical-rl".to_string()),
                (TEXT_ORIENTATION_PROPERTY, "upright".to_string()),
                (TEXT_COMBINE_UPRIGHT_PROPERTY, "digits 3".to_string()),
            ]
        );
    }

    #[test]
    fn mixed_orientation_turns_latin_and_keeps_cjk_upright() {
        for ch in ['縦', 'か', 'カ', '、', '。', 'ー', '「', '」', '（', 'Ａ', '１', '한', '※'] {
            assert!(is_upright(ch), "{ch} is upright");
        }
        for ch in ['A', 'z', '1', '-', '(', 'ｶ', 'é', 'Ж'] {
            assert!(!is_upright(ch), "{ch} is sideways");
        }
        let v = VerticalStyle {
            writing_mode: WritingMode::VerticalRl,
            ..VerticalStyle::DEFAULT
        };
        assert_eq!(
            split_text("縦書きでHTML 5を", &v),
            vec![
                ("縦書きで".to_string(), GlyphOrient::Upright),
                ("HTML 5".to_string(), GlyphOrient::Sideways),
                ("を".to_string(), GlyphOrient::Upright),
            ]
        );
    }

    #[test]
    fn digits_combine_only_short_runs() {
        let v = VerticalStyle {
            writing_mode: WritingMode::VerticalRl,
            combine: TextCombine::Digits(2),
            ..VerticalStyle::DEFAULT
        };
        assert_eq!(
            split_text("令和6年12月2026年", &v),
            vec![
                ("令和".to_string(), GlyphOrient::Upright),
                ("6".to_string(), GlyphOrient::Combined),
                ("年".to_string(), GlyphOrient::Upright),
                ("12".to_string(), GlyphOrient::Combined),
                ("月".to_string(), GlyphOrient::Upright),
                ("2026".to_string(), GlyphOrient::Sideways),
                ("年".to_string(), GlyphOrient::Upright),
            ]
        );
    }

    #[test]
    fn placed_glyphs_probe_finds_vertical_modes_and_ruby() {
        assert!(html_has_placed_glyphs("<p style=\"writing-mode: vertical-rl\">縦</p>"));
        assert!(html_has_placed_glyphs("<p style=\"WRITING-MODE:tb-rl\">縦</p>"));
        assert!(html_has_placed_glyphs("<p><span style=\"--nf-ruby: 'か'\">花</span></p>"));
        assert!(!html_has_placed_glyphs("<p style=\"writing-mode: horizontal-tb\">横</p>"));
        assert!(!html_has_placed_glyphs("<p>writing-mode is a CSS property</p>"));
    }

    #[test]
    fn columns_stack_right_to_left_or_left_to_right() {
        let rl = ColumnFrame {
            mode: WritingMode::VerticalRl,
            block_size: 300.0,
            line_top: 40.0,
            line_height: 30.0,
        };
        assert_eq!(rl.left(), 230.0);
        assert_eq!(rl.centre_x(), 245.0);
        assert_eq!(rl.map_point(12.0, 40.0), (260.0, 12.0));
        let lr = ColumnFrame {
            mode: WritingMode::VerticalLr,
            ..rl
        };
        assert_eq!(lr.left(), 40.0);
        assert_eq!(lr.map_point(12.0, 70.0), (40.0, 12.0));
    }

    #[test]
    fn compose_is_place_after_the_glyph_transform() {
        let lin = [0.0, -1.0, 1.0, 0.0];
        let (l, off) = compose([1.0, 0.0, 0.0, 1.0], (0.0, 0.0), (5.0, -5.0), (0.0, 0.0), lin, (100.0, 20.0));
        // The pen (0, 0) lands at P1: lin·(0 − c) + c + off.
        let px = l[0] * (0.0 - 5.0) + l[1] * (0.0 + 5.0) + 5.0 + off.0;
        let py = l[2] * (0.0 - 5.0) + l[3] * (0.0 + 5.0) - 5.0 + off.1;
        assert!((px - 100.0).abs() < 1e-9 && (py - 20.0).abs() < 1e-9, "{px} {py}");
    }
}
