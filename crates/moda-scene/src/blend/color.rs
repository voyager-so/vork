//! Colour parsing and interpolation for blends and the v3 `color` channel.
//!
//! The forgiving parse chain is a copy of the moda-text port (`crates/moda-text/src/color.rs`, a line-for-line
//! port of `utils/colorHelpers.ts`): moda-scene cannot depend on moda-text (moda-text dev-depends on
//! moda-scene). Keep the two in sync.

// ---------------------------------------------------------------------------
// Color (forgiving parseColorInput chain — copied port, see module docs)
// ---------------------------------------------------------------------------

/// TS RGBA shape: 0-255 channels, 0-1 alpha (`colorHelpers.RGBAColor`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgba255 {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

/// JS `Math.round` (round half toward +inf).
pub(crate) fn js_round(x: f64) -> f64 {
    (x + 0.5).floor()
}

/// JS number stringification is shortest-round-trip, same as Rust `{}`.
fn fmt_js(x: f64) -> String {
    if x == x.floor() && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

/// `parseColorInput` (colorHelpers.ts:578-628) → normalized
/// `rgba(r, g, b, a)` string, or None.
fn parse_color_input(input: &str) -> Option<String> {
    if input.is_empty() {
        return None;
    }
    let clean = input.trim().to_lowercase();
    if clean == "transparent" {
        return Some("rgba(0, 0, 0, 0)".to_string());
    }
    if let Some(hex) = named_color(&clean) {
        if let Some((r, g, b, a)) = hex_to_rgba(hex) {
            return Some(format!(
                "rgba({}, {}, {}, {})",
                fmt_js(r),
                fmt_js(g),
                fmt_js(b),
                fmt_js(a)
            ));
        }
    }
    if let Some((r, g, b, a)) = parse_rgba_color(input) {
        return Some(format!(
            "rgba({}, {}, {}, {})",
            fmt_js(r),
            fmt_js(g),
            fmt_js(b),
            fmt_js(a)
        ));
    }
    let hex_input = if clean.len() == 3 || clean.len() == 4 || clean.len() == 6 || clean.len() == 8
    {
        if clean.chars().all(|c| c.is_ascii_hexdigit()) {
            format!("#{clean}")
        } else {
            clean.clone()
        }
    } else {
        clean.clone()
    };
    if let Some((r, g, b, a)) = hex_to_rgba(&hex_input) {
        return Some(format!(
            "rgba({}, {}, {}, {})",
            fmt_js(r),
            fmt_js(g),
            fmt_js(b),
            fmt_js(a)
        ));
    }
    if let Some((r, g, b, a)) = parse_hsl_string(input) {
        return Some(format!(
            "rgba({}, {}, {}, {})",
            fmt_js(r),
            fmt_js(g),
            fmt_js(b),
            fmt_js(a)
        ));
    }
    None
}

/// `parseRgbaString` (blendUtils.ts:25-31).
pub(crate) fn parse_rgba_string(input: Option<&str>) -> Option<Rgba255> {
    let input = input?;
    if input.is_empty() {
        return None;
    }
    let normalized = parse_color_input(input).unwrap_or_else(|| input.to_string());
    let (r, g, b, a) = parse_rgba_color(&normalized)?;
    Some(Rgba255 { r, g, b, a })
}

/// `rgbaToCss` (blendUtils.ts:33-35).
pub(crate) fn rgba_to_css(c: Rgba255) -> String {
    format!(
        "rgba({}, {}, {}, {})",
        fmt_js(js_round(c.r)),
        fmt_js(js_round(c.g)),
        fmt_js(js_round(c.b)),
        fmt_js(c.a)
    )
}

// ---------------------------------------------------------------------------
// Forgiving color-parse helpers — verbatim copies of the moda-text ports
// (`crates/moda-text/src/color.rs`, themselves line-for-line ports of
// `utils/colorHelpers.ts`). moda-scene cannot depend on moda-text (moda-text
// dev-depends on moda-scene), so the ~150 lines are duplicated rather than
// re-exported. Keep in sync with the moda-text copy.
// ---------------------------------------------------------------------------

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
pub fn named_color(name: &str) -> Option<&'static str> {
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
