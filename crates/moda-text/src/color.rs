//! Text-pipeline color parsing — line-for-line port of the FORGIVING parser
//! chain the MSDF text renderer uses (`utils/colorHelpers.ts`):
//!
//!   `cssColorToNormalizedRgba` = `parseColorInput` → `colorToNormalizedRgba`
//!   (colorHelpers.ts:709-737, 578-628, 377-399)
//!
//! and the run-color quantization round-trip
//! (`parserRuns.ts normalizeColor` → `rgbaNormalizedArrayToCss`
//! (colorHelpers.ts:700-707) → re-parse at draw).
//!
//! COPY THE BUGS (this parser is deliberately DIFFERENT from the path-fill
//! parser in `moda-engine/src/color.rs`, which rejects named colors):
//! - named CSS colors DO parse here (CSS_NAMED_COLORS table, parseColorInput);
//! - `transparent` → rgba(0,0,0,0);
//! - rgb()/rgba() accepts INTEGER rgb components only (regex `\d+`), alpha
//!   `[\d.]+`; out-of-range rejects;
//! - hex via JS `parseInt(s, 16)` PREFIX parsing: `"#1g2733"` parses r from
//!   `"1g"` → 1 (parseInt stops at the first non-hex char, empty prefix →
//!   NaN → null);
//! - bare hex without `#` is accepted only when it matches
//!   `^[0-9a-f]{3|4|6|8}$` (parseColorInput:608-611);
//! - hsl()/hsla() comma and modern space syntax, s/l clamped 0-100, hue
//!   wrapped mod 360;
//! - `cssColorToNormalizedRgba` rejects non-finite / out-of-[0,1] channels
//!   (colorHelpers.ts:723-733).

/// JS `parseInt(s, 16)` prefix semantics (no sign handling needed here —
/// callers pass digit-only substrings).
fn js_parse_int_16(s: &str) -> Option<u32> {
    let mut val: Option<u32> = None;
    for c in s.chars() {
        let Some(d) = c.to_digit(16) else { break };
        val = Some(val.unwrap_or(0).saturating_mul(16).saturating_add(d));
    }
    val
}

/// `hexToRgba` (colorHelpers.ts:324-369): (r,g,b 0-255, a 0-1) or None.
fn hex_to_rgba(hex: &str) -> Option<(f64, f64, f64, f64)> {
    if hex.is_empty() {
        return None;
    }
    // JS `hex.replace('#', '')` — first occurrence only.
    let clean = hex.replacen('#', "", 1);
    let chars: Vec<char> = clean.chars().collect();
    let two = |a: char, b: char| js_parse_int_16(&format!("{a}{b}")).map(|v| v as f64);
    let dup = |c: char| js_parse_int_16(&format!("{c}{c}")).map(|v| v as f64);
    let (r, g, b, a) = match chars.len() {
        3 => (dup(chars[0]), dup(chars[1]), dup(chars[2]), Some(1.0)),
        4 => (
            dup(chars[0]),
            dup(chars[1]),
            dup(chars[2]),
            dup(chars[3]).map(|v| v / 255.0),
        ),
        6 => (
            two(chars[0], chars[1]),
            two(chars[2], chars[3]),
            two(chars[4], chars[5]),
            Some(1.0),
        ),
        8 => (
            two(chars[0], chars[1]),
            two(chars[2], chars[3]),
            two(chars[4], chars[5]),
            two(chars[6], chars[7]).map(|v| v / 255.0),
        ),
        _ => return None,
    };
    // `isNaN(...)` validation — None ⇔ NaN.
    Some((r?, g?, b?, a?))
}

/// `parseRgbaColor` (colorHelpers.ts:285-318): integer rgb, `[\d.]+` alpha.
fn parse_rgba_color(s: &str) -> Option<(f64, f64, f64, f64)> {
    let t = s.trim();
    let lower = t.to_lowercase();
    let rest = lower
        .strip_prefix("rgba")
        .or_else(|| lower.strip_prefix("rgb"))?;
    // ^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+))?\s*\)$
    // NOTE: the production regex has no /i flag and is matched against the
    // trimmed ORIGINAL string; prefixes are lowercase-only. Mirror that:
    if !(t.starts_with("rgb") || t.starts_with("rgba")) {
        return None;
    }
    let rest = rest.strip_prefix('(')?;
    let rest = rest.strip_suffix(')')?;
    let parts: Vec<&str> = rest.split(',').collect();
    if parts.len() != 3 && parts.len() != 4 {
        return None;
    }
    let int_part = |p: &str| -> Option<f64> {
        let p = p.trim();
        if p.is_empty() || !p.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        p.parse::<f64>().ok()
    };
    let r = int_part(parts[0])?;
    let g = int_part(parts[1])?;
    let b = int_part(parts[2])?;
    let a = if parts.len() == 4 {
        let p = parts[3].trim();
        if p.is_empty() || !p.chars().all(|c| c.is_ascii_digit() || c == '.') {
            return None;
        }
        // JS parseFloat over [\d.]+ — "1.2.3" parses to 1.2.
        js_parse_float(p)?
    } else {
        1.0
    };
    if !(0.0..=255.0).contains(&r) || !(0.0..=255.0).contains(&g) || !(0.0..=255.0).contains(&b) {
        return None;
    }
    if !(0.0..=1.0).contains(&a) {
        return None;
    }
    Some((r, g, b, a))
}

/// JS `parseFloat` prefix semantics over a `[\d.]+` capture.
fn js_parse_float(s: &str) -> Option<f64> {
    let mut end = 0;
    let mut seen_dot = false;
    let mut seen_digit = false;
    for (i, c) in s.char_indices() {
        if c.is_ascii_digit() {
            seen_digit = true;
            end = i + 1;
        } else if c == '.' && !seen_dot {
            seen_dot = true;
            end = i + 1;
        } else {
            break;
        }
    }
    if !seen_digit {
        return None;
    }
    s[..end]
        .trim_end_matches('.')
        .parse::<f64>()
        .ok()
        .or(Some(0.0))
}

/// `hslToRgb` (colorHelpers.ts) — standard CSS HSL → RGB (0-255).
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (f64, f64, f64) {
    let s = s / 100.0;
    let l = l / 100.0;
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match hp as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    (
        ((r1 + m) * 255.0).round(),
        ((g1 + m) * 255.0).round(),
        ((b1 + m) * 255.0).round(),
    )
}

/// `parseHslString` (colorHelpers.ts:922-944) — comma and modern syntax.
fn parse_hsl_string(s: &str) -> Option<(f64, f64, f64, f64)> {
    let t = s.trim();
    let lower = t.to_lowercase();
    let body = lower
        .strip_prefix("hsla")
        .or_else(|| lower.strip_prefix("hsl"))?;
    let body = body.trim().strip_prefix('(')?.strip_suffix(')')?;
    let parse_num = |p: &str| -> Option<f64> {
        let p = p
            .trim()
            .trim_end_matches("deg")
            .trim_end_matches('%')
            .trim();
        p.parse::<f64>().ok()
    };
    let (h, sat, light, alpha, alpha_pct) = if body.contains(',') {
        let parts: Vec<&str> = body.split(',').collect();
        if parts.len() != 3 && parts.len() != 4 {
            return None;
        }
        let a_raw = parts.get(3).map(|p| p.trim().to_string());
        let pct = a_raw.as_deref().map(|p| p.ends_with('%')).unwrap_or(false);
        (
            parse_num(parts[0])?,
            parse_num(parts[1])?,
            parse_num(parts[2])?,
            match a_raw {
                Some(p) => parse_num(&p)?,
                None => 1.0,
            },
            pct,
        )
    } else {
        let (main, a_part) = match body.split_once('/') {
            Some((m, a)) => (m, Some(a)),
            None => (body, None),
        };
        let parts: Vec<&str> = main.split_whitespace().collect();
        if parts.len() != 3 {
            return None;
        }
        let pct = a_part.map(|p| p.trim().ends_with('%')).unwrap_or(false);
        (
            parse_num(parts[0])?,
            parse_num(parts[1])?,
            parse_num(parts[2])?,
            match a_part {
                Some(p) => parse_num(p)?,
                None => 1.0,
            },
            pct,
        )
    };
    if !h.is_finite() || !sat.is_finite() || !light.is_finite() || !alpha.is_finite() {
        return None;
    }
    let a = if alpha_pct { alpha / 100.0 } else { alpha };
    let s_c = sat.clamp(0.0, 100.0);
    let l_c = light.clamp(0.0, 100.0);
    let h_w = ((h % 360.0) + 360.0) % 360.0;
    let (r, g, b) = hsl_to_rgb(h_w, s_c, l_c);
    Some((r, g, b, a.clamp(0.0, 1.0)))
}

/// CSS named colors (colorHelpers.ts CSS_NAMED_COLORS — Color Module 4).
fn named_color(name: &str) -> Option<&'static str> {
    Some(match name {
        "black" => "#000000",
        "silver" => "#c0c0c0",
        "gray" => "#808080",
        "white" => "#ffffff",
        "maroon" => "#800000",
        "red" => "#ff0000",
        "purple" => "#800080",
        "fuchsia" => "#ff00ff",
        "green" => "#008000",
        "lime" => "#00ff00",
        "olive" => "#808000",
        "yellow" => "#ffff00",
        "navy" => "#000080",
        "blue" => "#0000ff",
        "teal" => "#008080",
        "aqua" => "#00ffff",
        "aliceblue" => "#f0f8ff",
        "antiquewhite" => "#faebd7",
        "aquamarine" => "#7fffd4",
        "azure" => "#f0ffff",
        "beige" => "#f5f5dc",
        "bisque" => "#ffe4c4",
        "blanchedalmond" => "#ffebcd",
        "blueviolet" => "#8a2be2",
        "brown" => "#a52a2a",
        "burlywood" => "#deb887",
        "cadetblue" => "#5f9ea0",
        "chartreuse" => "#7fff00",
        "chocolate" => "#d2691e",
        "coral" => "#ff7f50",
        "cornflowerblue" => "#6495ed",
        "cornsilk" => "#fff8dc",
        "crimson" => "#dc143c",
        "cyan" => "#00ffff",
        "darkblue" => "#00008b",
        "darkcyan" => "#008b8b",
        "darkgoldenrod" => "#b8860b",
        "darkgray" => "#a9a9a9",
        "darkgrey" => "#a9a9a9",
        "darkgreen" => "#006400",
        "darkkhaki" => "#bdb76b",
        "darkmagenta" => "#8b008b",
        "darkolivegreen" => "#556b2f",
        "darkorange" => "#ff8c00",
        "darkorchid" => "#9932cc",
        "darkred" => "#8b0000",
        "darksalmon" => "#e9967a",
        "darkseagreen" => "#8fbc8f",
        "darkslateblue" => "#483d8b",
        "darkslategray" => "#2f4f4f",
        "darkslategrey" => "#2f4f4f",
        "darkturquoise" => "#00ced1",
        "darkviolet" => "#9400d3",
        "deeppink" => "#ff1493",
        "deepskyblue" => "#00bfff",
        "dimgray" => "#696969",
        "dimgrey" => "#696969",
        "dodgerblue" => "#1e90ff",
        "firebrick" => "#b22222",
        "floralwhite" => "#fffaf0",
        "forestgreen" => "#228b22",
        "gainsboro" => "#dcdcdc",
        "ghostwhite" => "#f8f8ff",
        "gold" => "#ffd700",
        "goldenrod" => "#daa520",
        "greenyellow" => "#adff2f",
        "grey" => "#808080",
        "honeydew" => "#f0fff0",
        "hotpink" => "#ff69b4",
        "indianred" => "#cd5c5c",
        "indigo" => "#4b0082",
        "ivory" => "#fffff0",
        "khaki" => "#f0e68c",
        "lavender" => "#e6e6fa",
        "lavenderblush" => "#fff0f5",
        "lawngreen" => "#7cfc00",
        "lemonchiffon" => "#fffacd",
        "lightblue" => "#add8e6",
        "lightcoral" => "#f08080",
        "lightcyan" => "#e0ffff",
        "lightgoldenrodyellow" => "#fafad2",
        "lightgray" => "#d3d3d3",
        "lightgrey" => "#d3d3d3",
        "lightgreen" => "#90ee90",
        "lightpink" => "#ffb6c1",
        "lightsalmon" => "#ffa07a",
        "lightseagreen" => "#20b2aa",
        "lightskyblue" => "#87cefa",
        "lightslategray" => "#778899",
        "lightslategrey" => "#778899",
        "lightsteelblue" => "#b0c4de",
        "lightyellow" => "#ffffe0",
        "limegreen" => "#32cd32",
        "linen" => "#faf0e6",
        "magenta" => "#ff00ff",
        "mediumaquamarine" => "#66cdaa",
        "mediumblue" => "#0000cd",
        "mediumorchid" => "#ba55d3",
        "mediumpurple" => "#9370db",
        "mediumseagreen" => "#3cb371",
        "mediumslateblue" => "#7b68ee",
        "mediumspringgreen" => "#00fa9a",
        "mediumturquoise" => "#48d1cc",
        "mediumvioletred" => "#c71585",
        "midnightblue" => "#191970",
        "mintcream" => "#f5fffa",
        "mistyrose" => "#ffe4e1",
        "moccasin" => "#ffe4b5",
        "navajowhite" => "#ffdead",
        "oldlace" => "#fdf5e6",
        "olivedrab" => "#6b8e23",
        "orange" => "#ffa500",
        "orangered" => "#ff4500",
        "orchid" => "#da70d6",
        "palegoldenrod" => "#eee8aa",
        "palegreen" => "#98fb98",
        "paleturquoise" => "#afeeee",
        "palevioletred" => "#db7093",
        "papayawhip" => "#ffefd5",
        "peachpuff" => "#ffdab9",
        "peru" => "#cd853f",
        "pink" => "#ffc0cb",
        "plum" => "#dda0dd",
        "powderblue" => "#b0e0e6",
        "rosybrown" => "#bc8f8f",
        "royalblue" => "#4169e1",
        "saddlebrown" => "#8b4513",
        "salmon" => "#fa8072",
        "sandybrown" => "#f4a460",
        "seagreen" => "#2e8b57",
        "seashell" => "#fff5ee",
        "sienna" => "#a0522d",
        "skyblue" => "#87ceeb",
        "slateblue" => "#6a5acd",
        "slategray" => "#708090",
        "slategrey" => "#708090",
        "snow" => "#fffafa",
        "springgreen" => "#00ff7f",
        "steelblue" => "#4682b4",
        "tan" => "#d2b48c",
        "thistle" => "#d8bfd8",
        "tomato" => "#ff6347",
        "turquoise" => "#40e0d0",
        "violet" => "#ee82ee",
        "wheat" => "#f5deb3",
        "whitesmoke" => "#f5f5f5",
        "yellowgreen" => "#9acd32",
        "rebeccapurple" => "#663399",
        _ => return None,
    })
}

/// `parseColorInput` (colorHelpers.ts:578-628), collapsed to its numeric
/// result: the (r,g,b 0-255 ints-or-rounded, a 0-1) tuple the emitted
/// `rgba(r, g, b, a)` string carries.
fn parse_color_input(input: &str) -> Option<(f64, f64, f64, f64)> {
    if input.is_empty() {
        return None;
    }
    let clean = input.trim().to_lowercase();
    if clean == "transparent" {
        return Some((0.0, 0.0, 0.0, 0.0));
    }
    if let Some(hex) = named_color(&clean) {
        if let Some(rgba) = hex_to_rgba(hex) {
            return Some(rgba);
        }
    }
    // already rgb()/rgba()? (matched against the ORIGINAL input)
    if let Some(rgba) = parse_rgba_color(input) {
        return Some(rgba);
    }
    // bare hex without '#'
    let mut hex_input = clean.clone();
    let is_bare_hex =
        matches!(clean.len(), 3 | 4 | 6 | 8) && clean.chars().all(|c| c.is_ascii_hexdigit());
    if is_bare_hex {
        hex_input = format!("#{clean}");
    }
    if let Some(rgba) = hex_to_rgba(&hex_input) {
        return Some(rgba);
    }
    if let Some(rgba) = parse_hsl_string(input) {
        return Some(rgba);
    }
    None
}

/// `colorToNormalizedRgba` (colorHelpers.ts:377-399) — the STRICT parse used
/// directly by gradient-stop parsing (`WebGPUGradientSystem.
/// parseGradientStops`): hex or integer-rgb()/rgba() ONLY — named colors,
/// `transparent`, bare hex regex and hsl() do NOT apply on this path.
pub fn strict_color_to_normalized_rgba(color: &str) -> Option<[f64; 4]> {
    if let Some((r, g, b, a)) = hex_to_rgba(color) {
        return Some([r / 255.0, g / 255.0, b / 255.0, a]);
    }
    if let Some((r, g, b, a)) = parse_rgba_color(color) {
        return Some([r / 255.0, g / 255.0, b / 255.0, a]);
    }
    None
}

/// `cssColorToNormalizedRgba` (colorHelpers.ts:709-737): the forgiving parse,
/// normalized to [0,1] floats; None on failure or out-of-range channels.
pub fn css_color_to_normalized_rgba(color: &str) -> Option<[f64; 4]> {
    let (r, g, b, a) = parse_color_input(color)?;
    let (r, g, b) = (r / 255.0, g / 255.0, b / 255.0);
    for v in [r, g, b, a] {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return None;
        }
    }
    Some([r, g, b, a])
}

/// `normalizeColor` (parserRuns.ts:43-48): parseColorInput(color || '#000')
/// → colorToNormalizedRgba, else [0,0,0,1].
pub fn normalize_run_color(color: &str) -> [f64; 4] {
    let input = if color.is_empty() { "#000" } else { color };
    match parse_color_input(input) {
        Some((r, g, b, a)) => [r / 255.0, g / 255.0, b / 255.0, a],
        None => [0.0, 0.0, 0.0, 1.0],
    }
}

/// `rgbaNormalizedArrayToCss` (colorHelpers.ts:700-707): rgb channels are
/// quantized via `Math.round(c * 255)`; alpha passes through. Returns the
/// css string used as `InlineStyle.color` (group keys for decoration draws).
pub fn rgba_normalized_array_to_css(rgba: [f64; 4]) -> String {
    let r = (rgba[0] * 255.0).round();
    let g = (rgba[1] * 255.0).round();
    let b = (rgba[2] * 255.0).round();
    format!("rgba({}, {}, {}, {})", r, g, b, fmt_js_number(rgba[3]))
}

/// JS template-literal number formatting (shortest round-trip; "1" not "1.0").
fn fmt_js_number(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e21 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_forms() {
        assert_eq!(
            css_color_to_normalized_rgba("#1d2733").unwrap(),
            [29.0 / 255.0, 39.0 / 255.0, 51.0 / 255.0, 1.0]
        );
        assert_eq!(
            css_color_to_normalized_rgba("#fff").unwrap(),
            [1.0, 1.0, 1.0, 1.0]
        );
        // 8-digit alpha
        let c = css_color_to_normalized_rgba("#ff000080").unwrap();
        assert!((c[3] - 128.0 / 255.0).abs() < 1e-12);
        // bare hex (parseColorInput adds '#')
        assert_eq!(
            css_color_to_normalized_rgba("ff0000").unwrap(),
            [1.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn named_and_transparent() {
        assert_eq!(
            css_color_to_normalized_rgba("black").unwrap(),
            [0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            css_color_to_normalized_rgba("RED").unwrap(),
            [1.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            css_color_to_normalized_rgba("transparent").unwrap(),
            [0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn rgba_forms() {
        assert_eq!(
            css_color_to_normalized_rgba("rgba(255, 0, 0, 0.5)").unwrap(),
            [1.0, 0.0, 0.0, 0.5]
        );
        assert_eq!(
            css_color_to_normalized_rgba("rgb(0,128,255)").unwrap(),
            [0.0, 128.0 / 255.0, 1.0, 1.0]
        );
        // float rgb components rejected (regex \d+)
        assert!(css_color_to_normalized_rgba("rgb(1.5, 0, 0)").is_none());
        // out-of-range rejected
        assert!(css_color_to_normalized_rgba("rgb(300, 0, 0)").is_none());
    }

    #[test]
    fn parse_int_prefix_quirk() {
        // parseInt('1g', 16) === 1 in JS — the whole-string parse must not
        // reject prefix-hex (colorHelpers hexToRgba via parseInt).
        let c = css_color_to_normalized_rgba("#1g2733");
        assert_eq!(c.unwrap()[0], 1.0 / 255.0);
    }

    #[test]
    fn quantization_round_trip() {
        // run color: raw css → normalizeColor → rgbaNormalizedArrayToCss →
        // cssColorToNormalizedRgba (the draw-time chain).
        let n = normalize_run_color("#1d2733");
        let css = rgba_normalized_array_to_css(n);
        assert_eq!(css, "rgba(29, 39, 51, 1)");
        assert_eq!(
            css_color_to_normalized_rgba(&css).unwrap(),
            [29.0 / 255.0, 39.0 / 255.0, 51.0 / 255.0, 1.0]
        );
        // unparsable → [0,0,0,1]
        assert_eq!(normalize_run_color("nonsense"), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn hsl_forms() {
        assert_eq!(
            css_color_to_normalized_rgba("hsl(0, 100%, 50%)").unwrap(),
            [1.0, 0.0, 0.0, 1.0]
        );
        let c = css_color_to_normalized_rgba("hsl(120 100% 50% / 50%)").unwrap();
        assert_eq!(c[1], 1.0);
        assert_eq!(c[3], 0.5);
    }
}
