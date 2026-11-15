//! Underline / strikethrough geometry — port of
//! `utils/richtext/decorations.ts` (+ the quad builder from
//! `utils/richtext/decorationGeometry.ts:52-99`). Not exercised by the Gate-1
//! oracle (linebox.json carries no decoration rects); pinned by unit tests
//! here and gated visually in M3a Gate 2.
//!
//! Quirks preserved: contiguous decorated segments merge into ONE rect;
//! leading/trailing whitespace (JS `\s`) trimmed off the rect; color = LAST
//! segment's color; thickness = max over the group of
//! `max(1, round(fontSize·0.06)) × weightFactor` re-rounded.
//!
//! CSS Text Decoration 3/4 styling ([`DecorationStyle`]): `text-decoration-style`
//! (solid | double | dotted | dashed | wavy), `text-decoration-color`,
//! `text-decoration-thickness` and `text-underline-offset` (auto | px | % of the
//! em), from the longhands or the `text-decoration` shorthand. A run at the
//! defaults draws exactly what it drew before; a group breaks where the styling
//! changes. [`decoration_shapes`] turns a rect into what each style draws, and
//! every lane draws those shapes (the TS twin is `decorationShapes`).

use crate::measure::TextMeasurer;
use crate::types::Line;

/// `UNDERLINE_OFFSET_RATIO` (decorations.ts:3).
pub const UNDERLINE_OFFSET_RATIO: f64 = 0.1;
/// `STRIKETHROUGH_OFFSET_RATIO` (decorations.ts:4).
pub const STRIKETHROUGH_OFFSET_RATIO: f64 = 0.3;

pub const TEXT_DECORATION_STYLE_PROPERTY: &str = "text-decoration-style";
pub const TEXT_DECORATION_COLOR_PROPERTY: &str = "text-decoration-color";
pub const TEXT_DECORATION_THICKNESS_PROPERTY: &str = "text-decoration-thickness";
pub const TEXT_UNDERLINE_OFFSET_PROPERTY: &str = "text-underline-offset";

/// Pattern geometry, in units of `u = max(thickness, 1)` (`DECORATION_PATTERN` in decorations.ts).
/// Double: two lines of the thickness, a thickness apart.
pub const DOUBLE_GAP: f64 = 1.0;
/// Dashed: dashes 3u long, 2u apart.
pub const DASH_LENGTH: f64 = 3.0;
pub const DASH_GAP: f64 = 2.0;
/// Dotted: round dots one thickness across, 2u centre to centre.
pub const DOT_SPACING: f64 = 2.0;
/// Wavy: a sine 1.25u from its centre line, 6u a wavelength, stroked at the thickness.
pub const WAVE_AMPLITUDE: f64 = 1.25;
pub const WAVE_LENGTH: f64 = 6.0;
/// Samples per wavelength of the wave's centre line.
pub const WAVE_SAMPLES: usize = 16;
/// Sides of a dot's polygon (the triangle builders and the outlines).
pub const DOT_SEGMENTS: usize = 12;

/// `text-decoration-style`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecorationLineStyle {
    #[default]
    Solid,
    Double,
    Dotted,
    Dashed,
    Wavy,
}

impl DecorationLineStyle {
    pub fn parse(value: &str) -> Option<DecorationLineStyle> {
        match value.trim().to_ascii_lowercase().as_str() {
            "solid" => Some(DecorationLineStyle::Solid),
            "double" => Some(DecorationLineStyle::Double),
            "dotted" => Some(DecorationLineStyle::Dotted),
            "dashed" => Some(DecorationLineStyle::Dashed),
            "wavy" => Some(DecorationLineStyle::Wavy),
            _ => None,
        }
    }

    pub fn css(self) -> &'static str {
        match self {
            DecorationLineStyle::Solid => "solid",
            DecorationLineStyle::Double => "double",
            DecorationLineStyle::Dotted => "dotted",
            DecorationLineStyle::Dashed => "dashed",
            DecorationLineStyle::Wavy => "wavy",
        }
    }
}

/// A `text-decoration-thickness` / `text-underline-offset` length: px, or a percentage of the em (`em` is stored as
/// one: `0.1em` = `10%`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecorationLength {
    Px(f64),
    Percent(f64),
}

impl DecorationLength {
    /// `Some(None)` for `auto` / `from-font` (the default), `Some(Some(len))` for a length, `None` when unreadable.
    /// `parseDecorationLength` in decorations.ts is the twin.
    pub fn parse(value: &str) -> Option<Option<DecorationLength>> {
        let v = value.trim().to_ascii_lowercase();
        if v == "auto" || v == "from-font" {
            return Some(None);
        }
        let split = v
            .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+'))
            .unwrap_or(v.len());
        let (num, unit) = v.split_at(split);
        if !num.chars().any(|c| c.is_ascii_digit()) {
            return None;
        }
        let n: f64 = num.parse().ok()?;
        if !n.is_finite() {
            return None;
        }
        match unit.trim() {
            "px" | "" => Some(Some(DecorationLength::Px(n))),
            "%" => Some(Some(DecorationLength::Percent(n))),
            "em" => Some(Some(DecorationLength::Percent(n * 100.0))),
            "pt" => Some(Some(DecorationLength::Px(n))), // a point is a px (72 per inch)
            _ => None,
        }
    }

    /// px at `font_size`.
    pub fn resolve(self, font_size: f64) -> f64 {
        match self {
            DecorationLength::Px(px) => px,
            DecorationLength::Percent(p) => p / 100.0 * font_size,
        }
    }

    fn is_negative(self) -> bool {
        match self {
            DecorationLength::Px(v) | DecorationLength::Percent(v) => v < 0.0,
        }
    }
}

/// A run's decoration styling beyond the line itself. [`DecorationStyle::default`] is every run without it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DecorationStyle {
    pub line_style: DecorationLineStyle,
    /// `text-decoration-color`; `None` = the text colour (`currentcolor`).
    pub color: Option<String>,
    /// `text-decoration-thickness`; `None` = auto (the size × weight rule).
    pub thickness: Option<DecorationLength>,
    /// `text-underline-offset` (from the baseline to the underline's top edge); `None` = auto.
    pub underline_offset: Option<DecorationLength>,
}

fn is_line_keyword(token: &str) -> bool {
    matches!(token, "underline" | "overline" | "line-through" | "none" | "blink")
}

/// Whitespace-separated tokens, keeping `rgb(1, 2, 3)` whole.
fn css_tokens(value: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0i32;
    for c in value.chars() {
        match c {
            '(' => {
                depth += 1;
                cur.push(c);
            }
            ')' => {
                depth -= 1;
                cur.push(c);
            }
            c if c.is_whitespace() && depth <= 0 => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// A colour value worth keeping: not a CSS-wide keyword, `currentcolor` (the text colour) or empty.
fn decoration_color(value: &str) -> Option<String> {
    let v = value.trim();
    let lower = v.to_ascii_lowercase();
    if v.is_empty()
        || matches!(
            lower.as_str(),
            "currentcolor" | "inherit" | "initial" | "unset" | "revert" | "auto"
        )
    {
        return None;
    }
    Some(v.to_string())
}

impl DecorationStyle {
    /// The `text-decoration` shorthand's style, colour and thickness (CSS resets the three, then sets what it names;
    /// `text-underline-offset` is not part of it). The line itself is read by the caller, as before.
    pub fn apply_shorthand(&mut self, value: &str) {
        self.line_style = DecorationLineStyle::Solid;
        self.color = None;
        self.thickness = None;
        for token in css_tokens(value) {
            let lower = token.to_ascii_lowercase();
            if is_line_keyword(&lower) {
                continue;
            }
            if let Some(style) = DecorationLineStyle::parse(&lower) {
                self.line_style = style;
            } else if let Some(len) = DecorationLength::parse(&lower) {
                if !len.is_some_and(|l| l.is_negative()) {
                    self.thickness = len;
                }
            } else {
                self.color = decoration_color(&token);
            }
        }
    }

    /// A longhand declaration; `false` when `prop` isn't one.
    pub fn apply_declaration(&mut self, prop: &str, value: &str) -> bool {
        match prop {
            TEXT_DECORATION_STYLE_PROPERTY => {
                if let Some(style) = DecorationLineStyle::parse(value) {
                    self.line_style = style;
                }
            }
            TEXT_DECORATION_COLOR_PROPERTY => self.color = decoration_color(value),
            TEXT_DECORATION_THICKNESS_PROPERTY => {
                if let Some(len) = DecorationLength::parse(value) {
                    if !len.is_some_and(|l| l.is_negative()) {
                        self.thickness = len;
                    }
                }
            }
            TEXT_UNDERLINE_OFFSET_PROPERTY => {
                if let Some(len) = DecorationLength::parse(value) {
                    self.underline_offset = len;
                }
            }
            _ => return false,
        }
        true
    }

    pub fn is_default(&self) -> bool {
        *self == DecorationStyle::default()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecorationRect {
    pub x1: f64,
    pub x2: f64,
    /// The centre of the line (of the upper line for a double underline; the wave's centre line sits
    /// [`WAVE_AMPLITUDE`] lower for a wavy underline, so its top stays at the line's top).
    pub y: f64,
    pub thickness: f64,
    pub color: String,
    /// The paint of the run whose colour the line takes (`InlineStyle::paint`); `None` for a line with a colour of
    /// its own.
    pub paint: Option<u32>,
    pub line_style: DecorationLineStyle,
    /// An underline (double and wavy grow away from the text: downward); else a line-through.
    pub underline: bool,
}

/// What a decoration draws, in the caller's coordinates.
#[derive(Debug, Clone, PartialEq)]
pub enum DecorationShape {
    Rect { x0: f64, y0: f64, x1: f64, y1: f64 },
    Dot { cx: f64, cy: f64, r: f64 },
    /// A polyline stroked at `width` (butt ends).
    Wave { points: Vec<[f64; 2]>, width: f64 },
}

/// The shapes `rect` draws, shifted by `offset_x`; dash, dot and wave patterns are phased from `origin_x` (in the
/// rect's own x, before the offset), so pieces of one line drawn separately (the PDF lane's words) line up.
/// `decorationShapes` in decorations.ts is the twin.
pub fn decoration_shapes(rect: &DecorationRect, offset_x: f64, origin_x: f64) -> Vec<DecorationShape> {
    let t = rect.thickness;
    let u = t.max(1.0);
    let (x1, x2) = (rect.x1, rect.x2);
    let mut out = Vec::new();
    if !(x2 > x1) || !(t > 0.0) {
        return out;
    }
    let band = |y: f64, a: f64, b: f64| DecorationShape::Rect {
        x0: offset_x + a,
        y0: y - t * 0.5,
        x1: offset_x + b,
        y1: y + t * 0.5,
    };
    match rect.line_style {
        DecorationLineStyle::Solid => out.push(band(rect.y, x1, x2)),
        DecorationLineStyle::Double => {
            let step = t * (1.0 + DOUBLE_GAP);
            if rect.underline {
                out.push(band(rect.y, x1, x2));
                out.push(band(rect.y + step, x1, x2));
            } else {
                out.push(band(rect.y - step * 0.5, x1, x2));
                out.push(band(rect.y + step * 0.5, x1, x2));
            }
        }
        DecorationLineStyle::Dashed => {
            let period = u * (DASH_LENGTH + DASH_GAP);
            let mut k = ((x1 - origin_x) / period).floor();
            loop {
                let a = origin_x + k * period;
                if a >= x2 {
                    break;
                }
                let (s, e) = (a.max(x1), (a + u * DASH_LENGTH).min(x2));
                if e > s {
                    out.push(band(rect.y, s, e));
                }
                k += 1.0;
            }
        }
        DecorationLineStyle::Dotted => {
            let period = u * DOT_SPACING;
            let first = origin_x + u * 0.5;
            let mut k = ((x1 - first) / period).ceil();
            loop {
                let cx = first + k * period;
                if cx >= x2 {
                    break;
                }
                if cx >= x1 {
                    out.push(DecorationShape::Dot {
                        cx: offset_x + cx,
                        cy: rect.y,
                        r: t * 0.5,
                    });
                }
                k += 1.0;
            }
        }
        DecorationLineStyle::Wavy => {
            let amp = u * WAVE_AMPLITUDE;
            let wavelength = u * WAVE_LENGTH;
            let cy = if rect.underline { rect.y + amp } else { rect.y };
            let step = wavelength / WAVE_SAMPLES as f64;
            let n = ((x2 - x1) / step).ceil().max(1.0) as usize;
            let mut points = Vec::with_capacity(n + 1);
            for i in 0..=n {
                let x = if i == n { x2 } else { x1 + i as f64 * step };
                let phase = (x - origin_x) / wavelength * std::f64::consts::TAU;
                points.push([offset_x + x, cy - amp * phase.sin()]);
            }
            out.push(DecorationShape::Wave { points, width: t });
        }
    }
    out
}

/// A dot's polygon ([`DOT_SEGMENTS`] sides).
pub fn dot_polygon(cx: f64, cy: f64, r: f64) -> Vec<[f64; 2]> {
    (0..DOT_SEGMENTS)
        .map(|i| {
            let a = i as f64 / DOT_SEGMENTS as f64 * std::f64::consts::TAU;
            [cx + r * a.cos(), cy + r * a.sin()]
        })
        .collect()
}

/// A stroked polyline's two edges, each point offset half the width along the normal of its neighbours' chord.
pub fn wave_edges(points: &[[f64; 2]], width: f64) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let h = width * 0.5;
    let n = points.len();
    let mut upper = Vec::with_capacity(n);
    let mut lower = Vec::with_capacity(n);
    for i in 0..n {
        let prev = points[i.saturating_sub(1)];
        let next = points[(i + 1).min(n - 1)];
        let (dx, dy) = (next[0] - prev[0], next[1] - prev[1]);
        let len = (dx * dx + dy * dy).sqrt();
        let (nx, ny) = if len > 0.0 {
            (dy / len, -dx / len)
        } else {
            (0.0, -1.0)
        };
        let p = points[i];
        upper.push([p[0] + nx * h, p[1] + ny * h]);
        lower.push([p[0] - nx * h, p[1] - ny * h]);
    }
    (upper, lower)
}

/// The closed outline of each shape (rects and dots as polygons, a wave as its stroke's outline) — what the outline
/// lanes fill (`decorationPolygons` in decorations.ts).
pub fn decoration_polygons(shapes: &[DecorationShape]) -> Vec<Vec<[f64; 2]>> {
    shapes
        .iter()
        .map(|s| match s {
            DecorationShape::Rect { x0, y0, x1, y1 } => {
                vec![[*x0, *y0], [*x1, *y0], [*x1, *y1], [*x0, *y1]]
            }
            DecorationShape::Dot { cx, cy, r } => dot_polygon(*cx, *cy, *r),
            DecorationShape::Wave { points, width } => {
                let (upper, mut lower) = wave_edges(points, *width);
                lower.reverse();
                upper.into_iter().chain(lower).collect()
            }
        })
        .collect()
}

/// `numericWeightOf` (decorations.ts:14-18).
fn numeric_weight_of(font_weight: &str) -> i64 {
    match crate::jsnum::js_parse_int(font_weight) {
        Some(w) => w,
        None => {
            if font_weight == "bold" {
                700
            } else {
                400
            }
        }
    }
}

/// `thicknessFor` (decorations.ts:20-36): weightFactor ∈
/// {1.0, 1.2, 1.4, 1.6, 1.8, 2.0} at <500/≥500/≥600/≥700/≥800/≥900.
pub fn thickness_for(font_size: f64, font_weight: &str) -> f64 {
    let numeric_weight = numeric_weight_of(font_weight);
    let base_thickness = crate::jsnum::js_math_round(font_size * 0.06).max(1.0);
    let weight_factor = if numeric_weight >= 900 {
        2.0
    } else if numeric_weight >= 800 {
        1.8
    } else if numeric_weight >= 700 {
        1.6
    } else if numeric_weight >= 600 {
        1.4
    } else if numeric_weight >= 500 {
        1.2
    } else {
        1.0
    };
    crate::jsnum::js_math_round(base_thickness * weight_factor).max(1.0)
}

/// The thinnest explicit `text-decoration-thickness` drawn, px.
pub const MIN_EXPLICIT_THICKNESS: f64 = 0.5;

/// One decoration group's rect from its segments' styles (`decorationRectFor` in decorations.ts): the auto rules
/// (thickness = max over the group of [`thickness_for`], underline 0.10 em below the baseline, line-through 0.30 em
/// above it, the last segment's colour) unless the group's [`DecorationStyle`] (one per group) says otherwise.
pub(crate) fn decoration_rect_for<'a>(
    styles: impl Iterator<Item = &'a crate::types::InlineStyle>,
    last: &crate::types::InlineStyle,
    x1: f64,
    x2: f64,
    baseline_y: f64,
    underline: bool,
    line_font_size: bool,
) -> DecorationRect {
    let mut max_font_size = 0.0f64;
    let mut max_thickness = 1.0f64;
    for st in styles {
        // Synthesized small caps keep the full size's line (the visual-order builder reads `font_size`, as before).
        let size = if line_font_size {
            st.line_font_size()
        } else {
            st.font_size
        };
        if size > max_font_size {
            max_font_size = size;
        }
        let th = thickness_for(size, &st.font_weight);
        if th > max_thickness {
            max_thickness = th;
        }
    }
    let deco = &last.decoration;
    let thickness = match deco.thickness {
        Some(len) => len.resolve(max_font_size).max(MIN_EXPLICIT_THICKNESS),
        None => max_thickness,
    };
    let y = if underline {
        match deco.underline_offset {
            Some(off) => baseline_y + off.resolve(max_font_size) + thickness * 0.5,
            None => baseline_y + max_font_size * UNDERLINE_OFFSET_RATIO,
        }
    } else {
        baseline_y - max_font_size * STRIKETHROUGH_OFFSET_RATIO
    };
    DecorationRect {
        x1,
        x2,
        y,
        thickness,
        color: deco.color.clone().unwrap_or_else(|| last.color.clone()),
        paint: if deco.color.is_none() { last.paint } else { None },
        line_style: deco.line_style,
        underline,
    }
}

fn is_js_whitespace(c: char) -> bool {
    matches!(
        c,
        '\u{0009}'
            | '\u{000A}'
            | '\u{000B}'
            | '\u{000C}'
            | '\u{000D}'
            | '\u{0020}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'
            ..='\u{200A}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202F}'
                | '\u{205F}'
                | '\u{3000}'
                | '\u{FEFF}'
    )
}

/// `computeTrimWidth` (decorations.ts:38-49) — `/^\s+/` and `/\s+$/`.
/// `pub(crate)`: the visual-order decoration builder (`crate::bidi_visual`)
/// trims group edges with the same JS-whitespace rule.
pub(crate) fn trim_widths<F: FnMut(&str) -> f64>(text: &str, mut measure: F) -> (f64, f64) {
    let leading: String = text.chars().take_while(|c| is_js_whitespace(*c)).collect();
    let trailing: String = {
        let rev: Vec<char> = text
            .chars()
            .rev()
            .take_while(|c| is_js_whitespace(*c))
            .collect();
        rev.into_iter().rev().collect()
    };
    // a fully-whitespace segment is BOTH leading and trailing (the TS regexes
    // both match the whole string) — keep that behavior.
    let leading_width = if leading.is_empty() {
        0.0
    } else {
        measure(&leading)
    };
    let trailing_width = if trailing.is_empty() {
        0.0
    } else {
        measure(&trailing)
    };
    (leading_width, trailing_width)
}

/// Shared body of `computeUnderlineRects` / `computeStrikethroughRects`
/// (decorations.ts:51-168) — they differ only in the decoration predicate and
/// the y-offset sign/ratio. A group also breaks where the decoration styling
/// ([`DecorationStyle`]) changes.
fn compute_decoration_rects<M: TextMeasurer>(
    line: &Line,
    baseline_y: f64,
    measurer: &M,
    wants: fn(&str) -> bool,
    underline: bool,
) -> Vec<DecorationRect> {
    let mut rects = Vec::new();
    let mut group_start: Option<usize> = None;
    let mut group_end: usize = 0;

    let flush =
        |group_start: &mut Option<usize>, group_end: usize, rects: &mut Vec<DecorationRect>| {
            let Some(gs) = *group_start else { return };
            let start_seg = &line.segments[gs];
            let end_seg = &line.segments[group_end];
            let mut end_x = 0.0;
            for seg in &line.segments[..gs] {
                end_x += crate::positioning::segment_advance(seg, measurer);
            }
            let mut start_x = end_x;
            for seg in &line.segments[gs..=group_end] {
                end_x += crate::positioning::segment_advance(seg, measurer);
            }
            let (leading_width, _) =
                trim_widths(&start_seg.text, |t| measurer.measure(t, &start_seg.style));
            let (_, trailing_width) =
                trim_widths(&end_seg.text, |t| measurer.measure(t, &end_seg.style));
            start_x += leading_width;
            end_x -= trailing_width;
            if end_x > start_x {
                rects.push(decoration_rect_for(
                    line.segments[gs..=group_end].iter().map(|s| &s.style),
                    &end_seg.style,
                    start_x,
                    end_x,
                    baseline_y,
                    underline,
                    true,
                ));
            }
            *group_start = None;
        };

    for (i, seg) in line.segments.iter().enumerate() {
        let dec = seg.style.text_decoration.as_deref().unwrap_or("none");
        if wants(dec) {
            if let Some(gs) = group_start {
                if line.segments[gs].style.decoration != seg.style.decoration {
                    flush(&mut group_start, group_end, &mut rects);
                }
            }
            if group_start.is_none() {
                group_start = Some(i);
            }
            group_end = i;
        } else {
            flush(&mut group_start, group_end, &mut rects);
        }
    }
    flush(&mut group_start, group_end, &mut rects);
    rects
}

/// `computeUnderlineRects` (decorations.ts:51-109):
/// `underlineY = baseline + 0.10 × maxFontSize`.
pub fn compute_underline_rects<M: TextMeasurer>(
    line: &Line,
    baseline_y: f64,
    measurer: &M,
) -> Vec<DecorationRect> {
    compute_decoration_rects(
        line,
        baseline_y,
        measurer,
        |dec| dec.contains("underline"),
        true,
    )
}

/// `computeStrikethroughRects` (decorations.ts:111-168):
/// `strikeY = baseline − 0.30 × maxFontSize`.
pub fn compute_strikethrough_rects<M: TextMeasurer>(
    line: &Line,
    baseline_y: f64,
    measurer: &M,
) -> Vec<DecorationRect> {
    compute_decoration_rects(
        line,
        baseline_y,
        measurer,
        |dec| dec.contains("line-through"),
        false,
    )
}

fn push_quad(verts: &mut Vec<f32>, idx: &mut Vec<u32>, q: [[f64; 2]; 4]) {
    let base = (verts.len() / 4) as u32;
    for p in q {
        verts.extend_from_slice(&[p[0] as f32, p[1] as f32, 0.0, 0.0]);
    }
    idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// `buildDecorationGeometry` (decorationGeometry.ts:52-99): solid quads
/// centered on y, height = thickness, [x,y,u=0,v=0] vertices + 6 indices.
/// The other styles triangulate their [`decoration_shapes`] (dots as fans,
/// a wave as a strip of quads); no two triangles overlap, so a translucent
/// colour blends once.
pub fn build_decoration_geometry(offset_x: f64, rects: &[DecorationRect]) -> (Vec<f32>, Vec<u32>) {
    let mut verts: Vec<f32> = Vec::with_capacity(rects.len() * 16);
    let mut idx: Vec<u32> = Vec::with_capacity(rects.len() * 6);
    for r in rects {
        if r.line_style == DecorationLineStyle::Solid {
            let x0 = offset_x + r.x1;
            let x1f = offset_x + r.x2;
            let y0 = r.y - r.thickness * 0.5;
            let y1f = r.y + r.thickness * 0.5;
            push_quad(
                &mut verts,
                &mut idx,
                [[x0, y0], [x1f, y0], [x1f, y1f], [x0, y1f]],
            );
            continue;
        }
        for shape in decoration_shapes(r, offset_x, r.x1) {
            match shape {
                DecorationShape::Rect { x0, y0, x1, y1 } => {
                    push_quad(&mut verts, &mut idx, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                }
                DecorationShape::Dot { cx, cy, r } => {
                    let ring = dot_polygon(cx, cy, r);
                    let base = (verts.len() / 4) as u32;
                    verts.extend_from_slice(&[cx as f32, cy as f32, 0.0, 0.0]);
                    for p in &ring {
                        verts.extend_from_slice(&[p[0] as f32, p[1] as f32, 0.0, 0.0]);
                    }
                    let n = ring.len() as u32;
                    for i in 0..n {
                        idx.extend_from_slice(&[base, base + 1 + i, base + 1 + (i + 1) % n]);
                    }
                }
                DecorationShape::Wave { points, width } => {
                    let (upper, lower) = wave_edges(&points, width);
                    for i in 0..points.len().saturating_sub(1) {
                        push_quad(
                            &mut verts,
                            &mut idx,
                            [upper[i], upper[i + 1], lower[i + 1], lower[i]],
                        );
                    }
                }
            }
        }
    }
    (verts, idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{default_style, InlineStyle, Line, LineSegment};

    struct CharWidth;
    impl TextMeasurer for CharWidth {
        fn measure(&self, text: &str, _style: &InlineStyle) -> f64 {
            text.chars().count() as f64 * 10.0
        }
    }

    fn seg(text: &str, deco: Option<&str>, weight: &str, size: f64, color: &str) -> LineSegment {
        let mut style = default_style();
        style.text_decoration = deco.map(|d| d.to_string());
        style.font_weight = weight.to_string();
        style.font_size = size;
        style.color = color.to_string();
        LineSegment {
            text: text.to_string(),
            style,
            tab: None,
            hyphen: false,
        }
    }

    #[test]
    fn thickness_table() {
        // decorations.ts:20-36
        assert_eq!(thickness_for(30.0, "400"), 2.0); // round(1.8)=2 ×1.0
        assert_eq!(thickness_for(30.0, "700"), 3.0); // 2×1.6=3.2→3
        assert_eq!(thickness_for(30.0, "900"), 4.0); // 2×2.0=4
        assert_eq!(thickness_for(10.0, "400"), 1.0); // max(1, round(0.6))=1
    }

    #[test]
    fn merged_rect_trims_whitespace_and_uses_last_color() {
        // decorations.ts:60-95 — "ab " + "cd" underlined, trailing space of
        // the LAST decorated segment trimmed, color from last segment
        let line = Line {
            segments: vec![
                seg("xx", None, "400", 16.0, "#000"),
                seg("ab ", Some("underline"), "400", 16.0, "#111"),
                seg("cd ", Some("underline"), "400", 16.0, "#222"),
                seg("zz", None, "400", 16.0, "#000"),
            ],
            line_height_px: 19.2,
            soft_wrap: false,
            ..Default::default()
        };
        let rects = compute_underline_rects(&line, 12.0, &CharWidth);
        assert_eq!(rects.len(), 1);
        let r = &rects[0];
        assert_eq!(r.x1, 20.0); // after "xx"
        assert_eq!(r.x2, 20.0 + 30.0 + 30.0 - 10.0); // minus trailing " "
        assert_eq!(r.color, "#222");
        assert_eq!(r.y, 12.0 + 16.0 * UNDERLINE_OFFSET_RATIO);
    }

    #[test]
    fn strikethrough_offset_sign() {
        let line = Line {
            segments: vec![seg("ab", Some("line-through"), "400", 20.0, "#000")],
            line_height_px: 24.0,
            soft_wrap: false,
            ..Default::default()
        };
        let rects = compute_strikethrough_rects(&line, 15.0, &CharWidth);
        assert_eq!(rects.len(), 1);
        assert_eq!(rects[0].y, 15.0 - 20.0 * STRIKETHROUGH_OFFSET_RATIO);
    }

    fn rect(style: DecorationLineStyle, underline: bool, x1: f64, x2: f64, t: f64) -> DecorationRect {
        DecorationRect {
            x1,
            x2,
            y: 20.0,
            thickness: t,
            color: "#000".into(),
            paint: None,
            line_style: style,
            underline,
        }
    }

    fn styled(css: &str) -> crate::types::InlineStyle {
        crate::parser::parse_inline_style(Some(css), &default_style())
    }

    #[test]
    fn parses_longhands_and_the_shorthand() {
        let s = styled(
            "text-decoration: underline; text-decoration-style: wavy; text-decoration-color: rgba(255, 0, 0, 0.5); \
             text-decoration-thickness: 3px; text-underline-offset: 0.25em",
        );
        assert_eq!(s.text_decoration.as_deref(), Some("underline"));
        assert_eq!(s.decoration.line_style, DecorationLineStyle::Wavy);
        assert_eq!(s.decoration.color.as_deref(), Some("rgba(255, 0, 0, 0.5)"));
        assert_eq!(s.decoration.thickness, Some(DecorationLength::Px(3.0)));
        assert_eq!(s.decoration.underline_offset, Some(DecorationLength::Percent(25.0)));

        let s = styled("text-decoration: underline dotted #00f 10%");
        assert_eq!(s.text_decoration.as_deref(), Some("underline"));
        assert_eq!(s.decoration.line_style, DecorationLineStyle::Dotted);
        assert_eq!(s.decoration.color.as_deref(), Some("#00f"));
        assert_eq!(s.decoration.thickness, Some(DecorationLength::Percent(10.0)));

        // The shorthand resets style / colour / thickness (not the offset); auto and from-font are the default.
        let mut inherited = styled("text-decoration: underline double red 4px; text-underline-offset: 2px");
        inherited.font_size = 20.0;
        let s = crate::parser::parse_inline_style(Some("text-decoration: underline"), &inherited);
        assert_eq!(s.decoration.line_style, DecorationLineStyle::Solid);
        assert_eq!(s.decoration.color, None);
        assert_eq!(s.decoration.thickness, None);
        assert_eq!(s.decoration.underline_offset, Some(DecorationLength::Px(2.0)));
        let s = styled("text-decoration-thickness: from-font; text-underline-offset: auto; text-decoration-color: currentcolor");
        assert!(s.decoration.is_default());
        // Unreadable values are ignored; a negative thickness too (a negative offset is fine).
        let s = styled("text-decoration-style: zigzag; text-decoration-thickness: -2px; text-underline-offset: -3px");
        assert_eq!(s.decoration.line_style, DecorationLineStyle::Solid);
        assert_eq!(s.decoration.thickness, None);
        assert_eq!(s.decoration.underline_offset, Some(DecorationLength::Px(-3.0)));
        // A plain underline is the default styling: what every existing document has.
        assert!(styled("text-decoration: underline").decoration.is_default());
    }

    #[test]
    fn explicit_thickness_offset_and_colour() {
        let mut a = seg("ab", Some("underline"), "400", 20.0, "#111");
        a.style.decoration.thickness = Some(DecorationLength::Percent(10.0)); // 2px at 20
        a.style.decoration.underline_offset = Some(DecorationLength::Px(4.0));
        a.style.decoration.color = Some("#f00".into());
        let line = Line {
            segments: vec![a],
            line_height_px: 24.0,
            ..Default::default()
        };
        let r = &compute_underline_rects(&line, 15.0, &CharWidth)[0];
        assert_eq!(r.thickness, 2.0);
        assert_eq!(r.y, 15.0 + 4.0 + 1.0, "top edge at baseline + offset");
        assert_eq!(r.color, "#f00");
        // The offset is an underline's only; a line-through keeps its place.
        let mut s = line.clone();
        s.segments[0].style.text_decoration = Some("line-through".into());
        let r = &compute_strikethrough_rects(&s, 15.0, &CharWidth)[0];
        assert_eq!(r.y, 15.0 - 20.0 * STRIKETHROUGH_OFFSET_RATIO);
        assert_eq!(r.thickness, 2.0);
    }

    #[test]
    fn a_group_breaks_where_the_styling_changes() {
        let a = seg("ab", Some("underline"), "400", 16.0, "#000");
        let mut b = seg("cd", Some("underline"), "400", 16.0, "#000");
        b.style.decoration.line_style = DecorationLineStyle::Wavy;
        let line = Line {
            segments: vec![a.clone(), a, b],
            line_height_px: 19.2,
            ..Default::default()
        };
        let rects = compute_underline_rects(&line, 12.0, &CharWidth);
        assert_eq!(rects.len(), 2);
        assert_eq!((rects[0].x1, rects[0].x2), (0.0, 40.0));
        assert_eq!(rects[0].line_style, DecorationLineStyle::Solid);
        assert_eq!((rects[1].x1, rects[1].x2), (40.0, 60.0));
        assert_eq!(rects[1].line_style, DecorationLineStyle::Wavy);
    }

    #[test]
    fn solid_geometry_is_unchanged() {
        // The one quad every decoration drew before styles existed.
        let r = rect(DecorationLineStyle::Solid, true, 10.0, 50.0, 2.0);
        let (v, i) = build_decoration_geometry(5.0, std::slice::from_ref(&r));
        assert_eq!(
            v,
            vec![15.0, 19.0, 0.0, 0.0, 55.0, 19.0, 0.0, 0.0, 55.0, 21.0, 0.0, 0.0, 15.0, 21.0, 0.0, 0.0]
        );
        assert_eq!(i, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(
            decoration_shapes(&r, 5.0, r.x1),
            vec![DecorationShape::Rect { x0: 15.0, y0: 19.0, x1: 55.0, y1: 21.0 }]
        );
    }

    #[test]
    fn double_geometry() {
        // Underline: two lines of the thickness a thickness apart, the second further from the text.
        let r = rect(DecorationLineStyle::Double, true, 0.0, 40.0, 2.0);
        assert_eq!(
            decoration_shapes(&r, 0.0, 0.0),
            vec![
                DecorationShape::Rect { x0: 0.0, y0: 19.0, x1: 40.0, y1: 21.0 },
                DecorationShape::Rect { x0: 0.0, y0: 23.0, x1: 40.0, y1: 25.0 },
            ]
        );
        // Line-through: centred on the line's place.
        let r = rect(DecorationLineStyle::Double, false, 0.0, 40.0, 2.0);
        assert_eq!(
            decoration_shapes(&r, 0.0, 0.0),
            vec![
                DecorationShape::Rect { x0: 0.0, y0: 17.0, x1: 40.0, y1: 19.0 },
                DecorationShape::Rect { x0: 0.0, y0: 21.0, x1: 40.0, y1: 23.0 },
            ]
        );
        let (v, i) = build_decoration_geometry(0.0, &[r]);
        assert_eq!((v.len(), i.len()), (32, 12));
    }

    #[test]
    fn dashed_geometry() {
        // t = 2: dashes 6 long, 4 apart, from the rect's start; the last one is cut at the end.
        let r = rect(DecorationLineStyle::Dashed, true, 0.0, 33.0, 2.0);
        let xs: Vec<(f64, f64)> = decoration_shapes(&r, 0.0, 0.0)
            .into_iter()
            .map(|s| match s {
                DecorationShape::Rect { x0, x1, y0, y1 } => {
                    assert_eq!((y0, y1), (19.0, 21.0));
                    (x0, x1)
                }
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(xs, vec![(0.0, 6.0), (10.0, 16.0), (20.0, 26.0), (30.0, 33.0)]);
        // Phased from an origin: a piece of the line starting mid-gap draws the same dashes.
        let piece = rect(DecorationLineStyle::Dashed, true, 8.0, 18.0, 2.0);
        let xs: Vec<f64> = decoration_shapes(&piece, 0.0, 0.0)
            .into_iter()
            .map(|s| match s {
                DecorationShape::Rect { x0, .. } => x0,
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(xs, vec![10.0]);
        // A thin line keeps a 1px pattern unit.
        let thin = rect(DecorationLineStyle::Dashed, true, 0.0, 10.0, 0.5);
        assert_eq!(decoration_shapes(&thin, 0.0, 0.0).len(), 2);
    }

    #[test]
    fn dotted_geometry() {
        // t = 2: round dots 2 across, centres 4 apart from 1.
        let r = rect(DecorationLineStyle::Dotted, true, 0.0, 16.0, 2.0);
        let dots = decoration_shapes(&r, 3.0, 0.0);
        assert_eq!(
            dots,
            [1.0, 5.0, 9.0, 13.0]
                .iter()
                .map(|cx| DecorationShape::Dot { cx: cx + 3.0, cy: 20.0, r: 1.0 })
                .collect::<Vec<_>>()
        );
        let (v, i) = build_decoration_geometry(0.0, &[r]);
        assert_eq!(v.len(), 4 * 4 * (DOT_SEGMENTS + 1));
        assert_eq!(i.len(), 4 * 3 * DOT_SEGMENTS);
    }

    #[test]
    fn wavy_geometry() {
        let t = 2.0;
        let r = rect(DecorationLineStyle::Wavy, true, 0.0, 24.0, t);
        let shapes = decoration_shapes(&r, 0.0, 0.0);
        let [DecorationShape::Wave { points, width }] = shapes.as_slice() else {
            panic!("{shapes:?}")
        };
        assert_eq!(*width, t);
        let amp = t * WAVE_AMPLITUDE;
        // Two wavelengths of 12, 16 samples each; the centre line sits one amplitude below the line, so the
        // wave's crest is where a solid underline's centre would be.
        assert_eq!(points.len(), 2 * WAVE_SAMPLES + 1);
        assert_eq!(points[0], [0.0, 20.0 + amp]);
        assert!((points[4][1] - 20.0).abs() < 1e-9, "crest at a quarter wave");
        assert!((points[12][1] - (20.0 + 2.0 * amp)).abs() < 1e-9, "trough at three quarters");
        assert_eq!(points.last().unwrap()[0], 24.0);
        for p in points {
            assert!(p[1] >= 20.0 - 1e-9 && p[1] <= 20.0 + 2.0 * amp + 1e-9);
        }
        // A line-through waves about its own centre.
        let s = rect(DecorationLineStyle::Wavy, false, 0.0, 24.0, t);
        let shapes = decoration_shapes(&s, 0.0, 0.0);
        let [DecorationShape::Wave { points, .. }] = shapes.as_slice() else {
            unreachable!()
        };
        assert_eq!(points[0], [0.0, 20.0]);
        // Stroke: one quad per segment, each edge half the thickness from the centre line.
        let (v, i) = build_decoration_geometry(0.0, &[r]);
        assert_eq!(i.len(), 2 * WAVE_SAMPLES * 6);
        assert_eq!(v.len(), 2 * WAVE_SAMPLES * 16);
        let poly = &decoration_polygons(&decoration_shapes(&rect(DecorationLineStyle::Wavy, true, 0.0, 24.0, t), 0.0, 0.0))[0];
        assert_eq!(poly.len(), 2 * (2 * WAVE_SAMPLES + 1));
        // At the crest the tangent is flat: the edges are straight above and below.
        assert!((poly[4][1] - (20.0 - t / 2.0)).abs() < 1e-6);
    }

    /// The TS twin (`decorations.ts` `decorationShapes` / `decorationPolygons`) reads the same fixture
    /// (packages/editor/test/decorationStyles.test.ts): both lanes draw the same shapes. `MODA_WRITE_DECORATION_FIXTURE=1`
    /// rewrites it.
    #[test]
    #[ignore = "needs unported Moda fixture decoration-shapes.json"]
    fn shapes_match_the_shared_fixture() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/decoration-shapes.json");
        let mut cases = Vec::new();
        for (style, underline) in [
            (DecorationLineStyle::Solid, true),
            (DecorationLineStyle::Double, true),
            (DecorationLineStyle::Double, false),
            (DecorationLineStyle::Dotted, true),
            (DecorationLineStyle::Dashed, true),
            (DecorationLineStyle::Wavy, true),
            (DecorationLineStyle::Wavy, false),
        ] {
            for (t, x1, x2, origin) in [(1.0, 0.0, 37.5, 0.0), (2.5, 12.25, 61.0, 3.0), (0.5, 5.0, 19.0, 5.0)] {
                let r = DecorationRect {
                    x1,
                    x2,
                    y: 30.0,
                    thickness: t,
                    color: String::new(),
                    paint: None,
                    line_style: style,
                    underline,
                };
                let polys = decoration_polygons(&decoration_shapes(&r, 7.0, origin));
                cases.push(serde_json::json!({
                    "rect": { "x1": x1, "x2": x2, "y": 30.0, "thickness": t, "lineStyle": style.css(), "underline": underline },
                    "offsetX": 7.0, "originX": origin,
                    "polygons": polys.iter().map(|p| p.iter().map(|q| [round6(q[0]), round6(q[1])]).collect::<Vec<_>>()).collect::<Vec<_>>(),
                }));
            }
        }
        let json = serde_json::to_string(&serde_json::json!({ "cases": cases })).unwrap() + "\n";
        if std::env::var_os("MODA_WRITE_DECORATION_FIXTURE").is_some() {
            std::fs::write(&path, &json).unwrap();
            return;
        }
        let pinned = std::fs::read_to_string(&path).expect("native/fixtures/decoration-shapes.json");
        let a: serde_json::Value = serde_json::from_str(&pinned).unwrap();
        let b: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(a, b, "decoration shapes changed: rewrite the fixture and check the TS twin");
    }

    fn round6(v: f64) -> f64 {
        (v * 1e6).round() / 1e6
    }

    #[test]
    fn decoration_length_parse() {
        assert_eq!(DecorationLength::parse("auto"), Some(None));
        assert_eq!(DecorationLength::parse("from-font"), Some(None));
        assert_eq!(DecorationLength::parse("2px"), Some(Some(DecorationLength::Px(2.0))));
        assert_eq!(DecorationLength::parse("0"), Some(Some(DecorationLength::Px(0.0))));
        assert_eq!(DecorationLength::parse("0.1em"), Some(Some(DecorationLength::Percent(10.0))));
        assert_eq!(DecorationLength::parse("15%"), Some(Some(DecorationLength::Percent(15.0))));
        assert_eq!(DecorationLength::parse("-1.5px"), Some(Some(DecorationLength::Px(-1.5))));
        assert_eq!(DecorationLength::parse("red"), None);
        assert_eq!(DecorationLength::parse("2vw"), None);
        assert_eq!(DecorationLength::Percent(10.0).resolve(30.0), 3.0);
    }
}
