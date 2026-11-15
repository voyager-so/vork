//! List markers (design-editor E7.16): what a list paragraph draws in its hanging indent.
//!
//! A `<ul>` / `<ol>` item's marker comes from the list's CSS `list-style-type` (or the HTML `type`
//! attribute); a list without one takes the level default, as Figma and Google Docs cycle them:
//!
//! - bulleted: disc, circle, square, then disc again;
//! - numbered: `1.`, `a.`, `i.`, then `1.` again.
//!
//! Bullets are geometry (the draw walks tessellate a disc, a ring or a square); numbers are text,
//! shaped in the item's first run style and right-aligned against the text start, one marker gap
//! before it. The editing surface draws the same thing with native CSS list markers (the rules in
//! `packages/editor/src/components/text/richtext-editor.css`); the TS twin of this module is
//! `packages/editor/src/utils/richtext/listMarkers.ts`.

/// A list's marker style (CSS `list-style-type`, the subset Moda draws).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ListStyle {
    Disc,
    Circle,
    Square,
    Decimal,
    LowerAlpha,
    UpperAlpha,
    LowerRoman,
    UpperRoman,
}

impl ListStyle {
    /// A CSS `list-style-type` value (`lower-latin` / `upper-latin` are the alpha aliases). Any other value
    /// (`none`, a string, an unsupported counter style) is `None`: the level default applies.
    pub fn from_css(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "disc" => Some(Self::Disc),
            "circle" => Some(Self::Circle),
            "square" => Some(Self::Square),
            "decimal" => Some(Self::Decimal),
            "lower-alpha" | "lower-latin" => Some(Self::LowerAlpha),
            "upper-alpha" | "upper-latin" => Some(Self::UpperAlpha),
            "lower-roman" => Some(Self::LowerRoman),
            "upper-roman" => Some(Self::UpperRoman),
            _ => None,
        }
    }

    /// The HTML `type` attribute of `<ol>` (`1 a A i I`, case-sensitive) or `<ul>` (`disc circle square`).
    pub fn from_type_attr(value: &str) -> Option<Self> {
        match value.trim() {
            "1" => Some(Self::Decimal),
            "a" => Some(Self::LowerAlpha),
            "A" => Some(Self::UpperAlpha),
            "i" => Some(Self::LowerRoman),
            "I" => Some(Self::UpperRoman),
            other => match other.to_ascii_lowercase().as_str() {
                "disc" => Some(Self::Disc),
                "circle" => Some(Self::Circle),
                "square" => Some(Self::Square),
                _ => None,
            },
        }
    }

    /// The CSS name.
    pub fn css(self) -> &'static str {
        match self {
            Self::Disc => "disc",
            Self::Circle => "circle",
            Self::Square => "square",
            Self::Decimal => "decimal",
            Self::LowerAlpha => "lower-alpha",
            Self::UpperAlpha => "upper-alpha",
            Self::LowerRoman => "lower-roman",
            Self::UpperRoman => "upper-roman",
        }
    }

    /// The style a list at `level` (0 = top) takes when it names none.
    pub fn level_default(kind: &str, level: i64) -> Self {
        let i = level.rem_euclid(3);
        if kind == "ol" {
            [Self::Decimal, Self::LowerAlpha, Self::LowerRoman][i as usize]
        } else {
            [Self::Disc, Self::Circle, Self::Square][i as usize]
        }
    }

    /// A bullet (drawn as geometry) rather than a counter (drawn as text).
    pub fn is_bullet(self) -> bool {
        matches!(self, Self::Disc | Self::Circle | Self::Square)
    }

    /// The counter text for the item numbered `index` (1-based), with its `.`; `None` for bullets.
    pub fn marker_text(self, index: i64) -> Option<String> {
        let n = index.max(1);
        let body = match self {
            Self::Disc | Self::Circle | Self::Square => return None,
            Self::Decimal => n.to_string(),
            Self::LowerAlpha => alphabetic(n, false),
            Self::UpperAlpha => alphabetic(n, true),
            Self::LowerRoman => roman(n, false),
            Self::UpperRoman => roman(n, true),
        };
        Some(format!("{body}."))
    }
}

/// CSS `lower-alpha`: a…z, aa, ab, … (bijective base 26).
fn alphabetic(mut n: i64, upper: bool) -> String {
    let base = if upper { b'A' } else { b'a' };
    let mut out = Vec::new();
    while n > 0 {
        n -= 1;
        out.push((base + (n % 26) as u8) as char);
        n /= 26;
    }
    out.iter().rev().collect()
}

/// CSS `lower-roman` (1–3999; outside that range CSS falls back to decimal).
fn roman(n: i64, upper: bool) -> String {
    if !(1..=3999).contains(&n) {
        return n.to_string();
    }
    const TABLE: [(i64, &str); 13] = [
        (1000, "m"),
        (900, "cm"),
        (500, "d"),
        (400, "cd"),
        (100, "c"),
        (90, "xc"),
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    ];
    let mut rest = n;
    let mut out = String::new();
    for (value, digits) in TABLE {
        while rest >= value {
            out.push_str(digits);
            rest -= value;
        }
    }
    if upper {
        out.to_ascii_uppercase()
    } else {
        out
    }
}

/// The marker style a list element declares: CSS `list-style-type` (in `style`, or the `list-style` shorthand)
/// wins over the `type` attribute, as in a browser.
pub fn declared_list_style(style_attr: Option<&str>, type_attr: Option<&str>) -> Option<ListStyle> {
    if let Some(style) = style_attr {
        for decl in style.split(';') {
            let Some((prop, value)) = decl.split_once(':') else {
                continue;
            };
            let prop = prop.trim().to_ascii_lowercase();
            if prop == "list-style-type" {
                if let Some(s) = ListStyle::from_css(value) {
                    return Some(s);
                }
            } else if prop == "list-style" {
                if let Some(s) = value.split_whitespace().find_map(ListStyle::from_css) {
                    return Some(s);
                }
            }
        }
    }
    type_attr.and_then(ListStyle::from_type_attr)
}

/// Bullet geometry, `[x, y, 0, 0]` vertices and triangle indices, like `ops::generate_bullet_disc`: a filled disc,
/// a ring (the circle marker, stroked at a fifth of its diameter, at least 1px) or a filled square of the disc's
/// diameter.
pub fn bullet_geometry(
    style: ListStyle,
    center_x: f64,
    center_y: f64,
    radius: f64,
    segments: u32,
) -> (Vec<f32>, Vec<u32>) {
    match style {
        ListStyle::Circle => {
            let n = segments.max(3);
            let thickness = (radius * 0.4).max(1.0).min(radius);
            let inner = (radius - thickness).max(0.0);
            let mut verts: Vec<f32> = Vec::with_capacity(n as usize * 8);
            let mut indices: Vec<u32> = Vec::with_capacity(n as usize * 6);
            for i in 0..n {
                let theta = (i as f64 / n as f64) * std::f64::consts::PI * 2.0;
                let (s, c) = theta.sin_cos();
                verts.extend_from_slice(&[
                    (center_x + c * radius) as f32,
                    (center_y + s * radius) as f32,
                    0.0,
                    0.0,
                ]);
                verts.extend_from_slice(&[
                    (center_x + c * inner) as f32,
                    (center_y + s * inner) as f32,
                    0.0,
                    0.0,
                ]);
            }
            for i in 0..n {
                let (o0, i0) = (2 * i, 2 * i + 1);
                let j = (i + 1) % n;
                let (o1, i1) = (2 * j, 2 * j + 1);
                indices.extend_from_slice(&[o0, o1, i1, o0, i1, i0]);
            }
            (verts, indices)
        }
        ListStyle::Square => {
            // The square reads as heavy as the disc at about 0.85 of its diameter.
            let h = radius * 0.85;
            let (x0, x1) = ((center_x - h) as f32, (center_x + h) as f32);
            let (y0, y1) = ((center_y - h) as f32, (center_y + h) as f32);
            (
                vec![
                    x0, y0, 0.0, 0.0, x1, y0, 0.0, 0.0, x1, y1, 0.0, 0.0, x0, y1, 0.0, 0.0,
                ],
                vec![0, 1, 2, 0, 2, 3],
            )
        }
        _ => crate::ops::generate_bullet_disc(center_x, center_y, radius, segments),
    }
}

/// The text marker of a laid-out list paragraph's first line: the counter string in the item's first run style
/// (decorations off), and its left x — right-aligned one marker gap before `text_start_x`. `None` for bullets and
/// plain paragraphs.
pub fn counter_marker_segment<M: crate::measure::TextMeasurer>(
    para: &crate::paragraph_layout::LaidOutParagraph,
    text_start_x: f64,
    marker_gap_px: f64,
    measurer: &M,
) -> Option<(crate::types::LineSegment, f64)> {
    let style = para.list_style?;
    let text = style.marker_text(para.list_index.unwrap_or(1))?;
    let mut marker_style = para.marker_style.clone()?;
    marker_style.text_decoration = Some("none".to_string());
    marker_style.letter_spacing = 0.0;
    let width = measurer.measure(&text, &marker_style);
    Some((
        crate::types::LineSegment {
            text,
            style: marker_style,
            tab: None,
            hyphen: false,
        },
        text_start_x - marker_gap_px - width,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters() {
        assert_eq!(ListStyle::Decimal.marker_text(12).as_deref(), Some("12."));
        assert_eq!(ListStyle::LowerAlpha.marker_text(1).as_deref(), Some("a."));
        assert_eq!(ListStyle::LowerAlpha.marker_text(26).as_deref(), Some("z."));
        assert_eq!(
            ListStyle::LowerAlpha.marker_text(27).as_deref(),
            Some("aa.")
        );
        assert_eq!(
            ListStyle::UpperAlpha.marker_text(28).as_deref(),
            Some("AB.")
        );
        assert_eq!(ListStyle::LowerRoman.marker_text(4).as_deref(), Some("iv."));
        assert_eq!(
            ListStyle::UpperRoman.marker_text(1994).as_deref(),
            Some("MCMXCIV.")
        );
        assert_eq!(
            ListStyle::UpperRoman.marker_text(4000).as_deref(),
            Some("4000.")
        );
        assert_eq!(ListStyle::Disc.marker_text(3), None);
    }

    #[test]
    fn level_defaults_cycle_like_figma() {
        let ol: Vec<_> = (0..4).map(|l| ListStyle::level_default("ol", l)).collect();
        assert_eq!(
            ol,
            [
                ListStyle::Decimal,
                ListStyle::LowerAlpha,
                ListStyle::LowerRoman,
                ListStyle::Decimal
            ]
        );
        let ul: Vec<_> = (0..4).map(|l| ListStyle::level_default("ul", l)).collect();
        assert_eq!(
            ul,
            [
                ListStyle::Disc,
                ListStyle::Circle,
                ListStyle::Square,
                ListStyle::Disc
            ]
        );
    }

    #[test]
    fn declared_styles() {
        assert_eq!(
            declared_list_style(Some("color: red; list-style-type: upper-roman"), None),
            Some(ListStyle::UpperRoman)
        );
        assert_eq!(
            declared_list_style(Some("list-style: square inside"), None),
            Some(ListStyle::Square)
        );
        assert_eq!(
            declared_list_style(None, Some("a")),
            Some(ListStyle::LowerAlpha)
        );
        assert_eq!(
            declared_list_style(None, Some("A")),
            Some(ListStyle::UpperAlpha)
        );
        // CSS wins over the attribute; an unknown value leaves the default.
        assert_eq!(
            declared_list_style(Some("list-style-type: lower-roman"), Some("1")),
            Some(ListStyle::LowerRoman)
        );
        assert_eq!(
            declared_list_style(Some("list-style-type: georgian"), None),
            None
        );
    }

    #[test]
    fn ring_and_square_geometry() {
        let (v, i) = bullet_geometry(ListStyle::Circle, 10.0, 10.0, 4.0, 16);
        assert_eq!(v.len(), 16 * 2 * 4);
        assert_eq!(i.len(), 16 * 6);
        let (v, i) = bullet_geometry(ListStyle::Square, 10.0, 10.0, 4.0, 16);
        assert_eq!((v.len(), i.len()), (16, 6));
        assert!((v[0] - 6.6).abs() < 1e-5 && (v[8] - 13.4).abs() < 1e-5);
    }
}
