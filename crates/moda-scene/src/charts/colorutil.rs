//! Color + shadow helpers for the chart port:
//!  - `utils/colorHelpers.ts` subset (hexToRgb/hexToRgba/parseRgbaColor/
//!    rgbToHex/normalizeColorString/brightness/contrast). Named CSS colors
//!    are NOT ported: chart palettes/series fills are hex or rgba in
//!    practice, and `normalizeColorString` returns its input unchanged for
//!    anything unparseable — which is also what this port does (the only
//!    divergence is named colors, which the engine's color parser rejects
//!    anyway, SEAMS §3).
//!  - `utils/markup/elements/charts/colors.ts` (palette/contrast/shadow).

use crate::value::{json, Map, Value};

use super::jsmath::{js_num_to_string, js_round};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgba {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

/// `hexToRgb` (colorHelpers.ts:25-53) — 3- or 6-digit hex.
pub fn hex_to_rgb(hex: &str) -> Option<(u8, u8, u8)> {
    if hex.is_empty() {
        return None;
    }
    let clean = hex.replacen('#', "", 1);
    let full: String = if clean.len() == 3 {
        clean.chars().flat_map(|c| [c, c]).collect()
    } else {
        clean
    };
    if full.len() != 6 || !full.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let r = u8::from_str_radix(&full[0..2], 16).ok()?;
    let g = u8::from_str_radix(&full[2..4], 16).ok()?;
    let b = u8::from_str_radix(&full[4..6], 16).ok()?;
    Some((r, g, b))
}

/// `hexToRgba` (colorHelpers.ts:324+) — 3/4/6/8-digit hex.
pub fn hex_to_rgba(hex: &str) -> Option<Rgba> {
    let trimmed = hex.trim();
    if !trimmed.starts_with('#') && !trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let clean = trimmed.strip_prefix('#').unwrap_or(trimmed);
    let expanded: String = match clean.len() {
        3 | 4 => clean.chars().flat_map(|c| [c, c]).collect(),
        6 | 8 => clean.to_string(),
        _ => return None,
    };
    if !expanded.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let r = u8::from_str_radix(&expanded[0..2], 16).ok()? as f64;
    let g = u8::from_str_radix(&expanded[2..4], 16).ok()? as f64;
    let b = u8::from_str_radix(&expanded[4..6], 16).ok()? as f64;
    let a = if expanded.len() == 8 {
        // alpha = parseInt(hex, 16) / 255, rounded to 2 decimals in the TS
        let raw = u8::from_str_radix(&expanded[6..8], 16).ok()? as f64 / 255.0;
        js_round(raw * 100.0) / 100.0
    } else {
        1.0
    };
    Some(Rgba { r, g, b, a })
}

/// `parseRgbaColor` — rgb()/rgba() forms.
pub fn parse_rgba_color(color: &str) -> Option<Rgba> {
    let t = color.trim().to_lowercase();
    let inner = t
        .strip_prefix("rgba(")
        .or_else(|| t.strip_prefix("rgb("))?
        .strip_suffix(')')?;
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() < 3 {
        return None;
    }
    let r: f64 = parts[0].parse().ok()?;
    let g: f64 = parts[1].parse().ok()?;
    let b: f64 = parts[2].parse().ok()?;
    let a: f64 = if parts.len() > 3 {
        parts[3].parse().ok()?
    } else {
        1.0
    };
    Some(Rgba { r, g, b, a })
}

/// `rgbToHex` (colorHelpers.ts:62-75).
pub fn rgb_to_hex(r: f64, g: f64, b: f64) -> String {
    let clamp = |n: f64| -> u8 {
        let rounded = js_round(n);
        rounded.clamp(0.0, 255.0) as u8
    };
    format!("#{:02x}{:02x}{:02x}", clamp(r), clamp(g), clamp(b))
}

/// `normalizeColorString` (colorHelpers.ts:983-1023) minus HSL/named colors
/// (see module docs). 'transparent' is handled by the caller (`parse_fill`).
pub fn normalize_color_string(color: &str) -> String {
    if color.is_empty() {
        return color.to_string();
    }
    let parsed = parse_rgba_color(color).or_else(|| hex_to_rgba(color));
    if let Some(p) = parsed {
        if p.a < 1.0 {
            return format!(
                "rgba({}, {}, {}, {})",
                js_num_to_string(p.r),
                js_num_to_string(p.g),
                js_num_to_string(p.b),
                js_num_to_string(p.a)
            );
        }
        return rgb_to_hex(p.r, p.g, p.b);
    }
    if color.trim().eq_ignore_ascii_case("transparent") {
        return "rgba(0, 0, 0, 0)".to_string();
    }
    color.to_string()
}

/// `calculateBrightness` + `isLightColor` + `getContrastTextColor`
/// (colorHelpers.ts:77-113). Invalid hex → brightness 0 → "light" false.
pub fn get_contrast_text_color(background: &str) -> &'static str {
    let brightness = match hex_to_rgb(background) {
        Some((r, g, b)) => (r as f64 * 299.0 + g as f64 * 587.0 + b as f64 * 114.0) / 1000.0,
        None => 0.0,
    };
    if brightness > 128.0 {
        "#000000"
    } else {
        "#ffffff"
    }
}

// ---------------------------------------------------------------------------
// charts/colors.ts
// ---------------------------------------------------------------------------

/// `resolvePaletteColor` (colors.ts:10-13).
pub fn resolve_palette_color<'a>(palette: &'a [String], index: usize) -> &'a str {
    if palette.is_empty() {
        return "#3b82f6";
    }
    palette
        .get(index % palette.len())
        .map(String::as_str)
        .unwrap_or_else(|| palette.first().map(String::as_str).unwrap_or("#3b82f6"))
}

/// `extractPrimaryColor` (colors.ts:18-27): first #hex found (3-6 hex chars).
pub fn extract_primary_color(fill: &str) -> Option<String> {
    if fill.starts_with('#') {
        return Some(fill.to_string());
    }
    // regex /#[0-9a-fA-F]{3,6}/ — first match, greedy up to 6
    let bytes = fill.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'#' {
            let mut j = i + 1;
            while j < bytes.len() && j - i <= 6 && bytes[j].is_ascii_hexdigit() {
                j += 1;
            }
            if j - i - 1 >= 3 {
                return Some(fill[i..j].to_string());
            }
        }
    }
    None
}

/// `getAutoContrastColor` (colors.ts:32-36).
pub fn get_auto_contrast_color(fill: &str) -> String {
    match extract_primary_color(fill) {
        None => "#1a1a1a".to_string(),
        Some(primary) => get_contrast_text_color(&primary).to_string(),
    }
}

/// `parseBoxShadow` (colors.ts:67-122).
fn parse_box_shadow(value: &str) -> (Option<f64>, Option<f64>, Option<f64>, Option<String>) {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed == "none" {
        return (None, None, None, None);
    }
    let mut color: Option<String> = None;
    let mut remaining = trimmed.to_string();

    // rgba?(...) first
    if let Some(start) = find_rgb_call(trimmed) {
        let (s, e) = start;
        color = Some(trimmed[s..e].to_string());
        remaining = format!("{}{}", &trimmed[..s], &trimmed[e..])
            .trim()
            .to_string();
    } else {
        // hex at end: /#[0-9a-fA-F]{3,8}\s*$/
        let t = remaining.trim_end();
        if let Some(hash) = t.rfind('#') {
            let tail = &t[hash + 1..];
            if (3..=8).contains(&tail.len()) && tail.bytes().all(|b| b.is_ascii_hexdigit()) {
                color = Some(t[hash..].trim().to_string());
                remaining = t[..hash].trim().to_string();
            }
        }
    }

    let mut numbers: Vec<f64> = Vec::new();
    for part in remaining.split_whitespace() {
        let p = part.strip_suffix("px").unwrap_or(part);
        // /^(-?\d+(?:\.\d+)?)(px)?$/
        if !p.is_empty()
            && p.bytes()
                .enumerate()
                .all(|(i, c)| c.is_ascii_digit() || c == b'.' || (i == 0 && c == b'-'))
            && p.bytes().filter(|&c| c == b'.').count() <= 1
            && p != "-"
        {
            if let Ok(v) = p.parse::<f64>() {
                numbers.push(v);
            }
        }
    }

    let x = numbers.first().copied();
    let y = numbers.get(1).copied();
    let blur = numbers.get(2).copied();
    if numbers.is_empty() && color.is_none() {
        return (None, None, None, None);
    }
    (x, y, blur, color)
}

fn find_rgb_call(s: &str) -> Option<(usize, usize)> {
    let lower = s.to_lowercase();
    let start = lower.find("rgba(").or_else(|| lower.find("rgb("))?;
    let close = s[start..].find(')')? + start + 1;
    Some((start, close))
}

/// `resolveShadowProps` (colors.ts) → props map to merge: `effects` with one
/// `drop-shadow` entry, when the CSS box-shadow parses to a shadow that draws
/// (a colour, and a blur or an offset; `effect::drop_shadow_draws`).
pub fn resolve_shadow_props(shadow: Option<&str>) -> Map {
    let mut props = Map::new();
    let Some(shadow) = shadow else {
        return props;
    };
    if shadow.is_empty() {
        return props;
    }
    let (x, y, blur, color) = parse_box_shadow(shadow);
    let Some(color) = color else {
        return props;
    };
    use crate::effect::{drop_shadow_draws, set_effects, DropShadow, Effect, EffectKind, EffectOffset};
    let shadow = DropShadow {
        color: json!(color),
        blur: blur.unwrap_or(0.0),
        offset: EffectOffset { x: x.unwrap_or(0.0), y: y.unwrap_or(0.0) },
        spread: None,
        opacity: None,
        show_behind_node: None,
    };
    if drop_shadow_draws(&shadow) {
        let id = "drop-shadow-0".to_string();
        set_effects(&mut props, vec![Effect { id, visible: None, kind: EffectKind::DropShadow(shadow) }]);
    }
    props
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_hex_roundtrip() {
        assert_eq!(normalize_color_string("#3B82F6"), "#3b82f6");
        assert_eq!(normalize_color_string("#f0a"), "#ff00aa");
        assert_eq!(
            normalize_color_string("rgba(0, 0, 0, 0.5)"),
            "rgba(0, 0, 0, 0.5)"
        );
        assert_eq!(normalize_color_string("rgb(255, 0, 0)"), "#ff0000");
        assert_eq!(normalize_color_string("transparent"), "rgba(0, 0, 0, 0)");
        assert_eq!(normalize_color_string("not-a-color"), "not-a-color");
    }

    #[test]
    fn contrast() {
        assert_eq!(get_contrast_text_color("#ffffff"), "#000000");
        assert_eq!(get_contrast_text_color("#1a1a1a"), "#ffffff");
        assert_eq!(get_auto_contrast_color("#3b82f6"), "#ffffff");
        assert_eq!(get_auto_contrast_color("no-hex-here"), "#1a1a1a");
    }

    #[test]
    fn box_shadow() {
        let p = resolve_shadow_props(Some("0 4px 20px rgba(0,0,0,0.15)"));
        assert_eq!(
            p.get("effects"),
            Some(&json!([{"id": "drop-shadow-0", "type": "drop-shadow", "color": "rgba(0,0,0,0.15)",
                "blur": 20.0, "offset": {"x": 0.0, "y": 4.0}}]))
        );
        // No colour, or neither a blur nor an offset: nothing draws.
        assert!(resolve_shadow_props(Some("0 4px 20px")).is_empty());
        assert!(resolve_shadow_props(Some("0 0 0 #000")).is_empty());
        assert!(resolve_shadow_props(Some("none")).is_empty());
        assert!(resolve_shadow_props(None).is_empty());
    }
}
