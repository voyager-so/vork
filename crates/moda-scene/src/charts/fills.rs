//! Port of `utils/markup/fills.ts` `parseFill` (solid + linear/radial/conic
//! gradient string syntax) — charts feed palette colors and user series fills
//! through this before spreading onto nodes. The result is a prop bag holding
//! the fill stack (`paint_bag`).

use super::paint_bag;
use crate::paint::{
    ConicGradientPaint, GradientStop, LinearGradientPaint, Paint, PaintPoint, RadialGradientPaint,
};
use crate::value::{json, Map, Value};


use super::colorutil::normalize_color_string;

fn transparent_solid() -> Map {
    let mut m = Map::new();
    paint_bag::set_fill_color(&mut m, json!("rgba(0, 0, 0, 0)"));
    m
}

fn with_paint(paint: Paint) -> Map {
    let mut m = Map::new();
    paint_bag::set_fill_paint(&mut m, Some(paint));
    m
}

fn pt(p: (f64, f64)) -> PaintPoint {
    PaintPoint { x: p.0, y: p.1 }
}

/// `parseFill` (fills.ts:117-149) → props map to spread onto a node.
pub fn parse_fill(value: &str) -> Map {
    let trimmed = value.trim();
    let normalized = trimmed.to_lowercase();
    if normalized == "none" || normalized == "transparent" {
        return transparent_solid();
    }
    if trimmed.starts_with("linear-gradient(") {
        return parse_linear_gradient(trimmed);
    }
    if trimmed.starts_with("radial-gradient(") {
        return parse_radial_gradient(trimmed);
    }
    if trimmed.starts_with("conic-gradient(") {
        return parse_conic_gradient(trimmed);
    }
    let mut m = Map::new();
    paint_bag::set_fill_color(&mut m, json!(normalize_color_string(trimmed)));
    m
}

fn gradient_content<'a>(value: &'a str, prefix: &str) -> Option<&'a str> {
    // /^<kind>-gradient\((.+)\)$/i — anchored, content non-empty
    let lower = value.to_lowercase();
    if !lower.starts_with(prefix) || !value.ends_with(')') {
        return None;
    }
    let inner = &value[prefix.len()..value.len() - 1];
    if inner.is_empty() {
        None
    } else {
        Some(inner)
    }
}

/// `splitGradientArgs` (fills.ts:350-375).
fn split_gradient_args(content: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32;
    for c in content.chars() {
        match c {
            '(' => {
                depth += 1;
                current.push(c);
            }
            ')' => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                parts.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(c),
        }
    }
    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }
    parts
}

/// `parseColorStops` (fills.ts:448-515).
fn parse_color_stops(args: &[String]) -> Vec<GradientStop> {
    #[derive(Clone)]
    struct Stop {
        color: String,
        position: Option<f64>,
    }
    let normalize_stop = |raw: &str| -> String {
        let t = raw.trim();
        if t.eq_ignore_ascii_case("none") {
            return normalize_color_string("transparent");
        }
        normalize_color_string(t)
    };
    let mut stops: Vec<Stop> = Vec::new();
    for arg in args {
        let trimmed = arg.trim();
        if trimmed.is_empty() {
            continue;
        }
        // /^(.+?)\s+(\d+(?:\.\d+)?)\s*%$/ — color + trailing percent
        let mut parsed = None;
        if let Some(no_pct) = trimmed.strip_suffix('%') {
            let no_pct = no_pct.trim_end();
            if let Some(ws) = no_pct.rfind(char::is_whitespace) {
                let num = &no_pct[ws + 1..];
                if !num.is_empty()
                    && num.bytes().all(|b| b.is_ascii_digit() || b == b'.')
                    && num.bytes().filter(|&b| b == b'.').count() <= 1
                    && num.as_bytes()[0].is_ascii_digit()
                {
                    if let Ok(v) = num.parse::<f64>() {
                        parsed = Some(Stop {
                            color: normalize_stop(&no_pct[..ws]),
                            position: Some(v / 100.0),
                        });
                    }
                }
            }
        }
        stops.push(parsed.unwrap_or_else(|| Stop {
            color: normalize_stop(trimmed),
            position: None,
        }));
    }
    if stops.is_empty() {
        return Vec::new();
    }
    let n = stops.len();
    if stops[0].position.is_none() {
        stops[0].position = Some(0.0);
    }
    if stops[n - 1].position.is_none() {
        stops[n - 1].position = Some(1.0);
    }
    let mut last_known = 0usize;
    for i in 1..n {
        if stops[i].position.is_some() {
            let gap = i - last_known;
            if gap > 1 {
                let start_pos = stops[last_known].position.unwrap();
                let end_pos = stops[i].position.unwrap();
                let step = (end_pos - start_pos) / gap as f64;
                for j in (last_known + 1)..i {
                    stops[j].position = Some(start_pos + step * (j - last_known) as f64);
                }
            }
            last_known = i;
        }
    }
    stops
        .into_iter()
        .map(|s| GradientStop {
            offset: s.position.unwrap(),
            color: json!(s.color),
        })
        .collect()
}

/// `angleToPoints` (fills.ts:427-437): rotate the unit up vector by CSS deg
/// (clockwise in screen space — `rotatePointAroundPivot`).
fn angle_to_points(degrees: f64) -> ((f64, f64), (f64, f64)) {
    // rotatePointAroundPivot({x:0,y:-1},{0,0},deg) — screen-space CW rotation
    let rad = degrees * std::f64::consts::PI / 180.0;
    let (sin, cos) = (rad.sin(), rad.cos());
    // rotating (0,-1): x' = x cos − y sin = sin; y' = x sin + y cos = −cos
    let dir = (sin, -cos);
    let dx = dir.0 * 50.0;
    let dy = dir.1 * 50.0;
    ((50.0 - dx, 50.0 - dy), (50.0 + dx, 50.0 + dy))
}

/// `parseGradientDirection` (fills.ts:381-421).
fn parse_gradient_direction(first_arg: &str) -> ((f64, f64), (f64, f64), usize) {
    let arg = first_arg.to_lowercase();
    let arg = arg.trim();
    let table: [(&str, (f64, f64), (f64, f64)); 8] = [
        ("to right", (0.0, 50.0), (100.0, 50.0)),
        ("to left", (100.0, 50.0), (0.0, 50.0)),
        ("to bottom", (50.0, 0.0), (50.0, 100.0)),
        ("to top", (50.0, 100.0), (50.0, 0.0)),
        ("to bottom right", (0.0, 0.0), (100.0, 100.0)),
        ("to bottom left", (100.0, 0.0), (0.0, 100.0)),
        ("to top right", (0.0, 100.0), (100.0, 0.0)),
        ("to top left", (100.0, 100.0), (0.0, 0.0)),
    ];
    for (kw, s, e) in table {
        if arg == kw {
            return (s, e, 1);
        }
    }
    // /^(-?\d+(?:\.\d+)?)\s*(?:deg)?$/
    let num = arg.strip_suffix("deg").map(str::trim_end).unwrap_or(arg);
    let valid = !num.is_empty()
        && num
            .bytes()
            .enumerate()
            .all(|(i, b)| b.is_ascii_digit() || b == b'.' || (i == 0 && b == b'-'))
        && num.bytes().filter(|&b| b == b'.').count() <= 1
        && num != "-";
    if valid {
        if let Ok(angle) = num.parse::<f64>() {
            let (s, e) = angle_to_points(angle);
            return (s, e, 1);
        }
    }
    ((50.0, 0.0), (50.0, 100.0), 0)
}


fn parse_linear_gradient(value: &str) -> Map {
    let Some(content) = gradient_content(value, "linear-gradient(") else {
        return transparent_solid();
    };
    let parts = split_gradient_args(content);
    if parts.len() < 2 {
        return transparent_solid();
    }
    let (start, end, color_start) = parse_gradient_direction(&parts[0]);
    let stops = parse_color_stops(&parts[color_start..]);
    if stops.is_empty() {
        return transparent_solid();
    }
    with_paint(Paint::Linear(LinearGradientPaint {
        start: pt(start),
        end: pt(end),
        stops,
        options: Default::default(),
    }))
}

fn parse_radial_gradient(value: &str) -> Map {
    let Some(content) = gradient_content(value, "radial-gradient(") else {
        return transparent_solid();
    };
    let parts = split_gradient_args(content);
    if parts.len() < 2 {
        return transparent_solid();
    }
    let first = parts[0].to_lowercase();
    let mut color_start = 0usize;
    if first.contains("circle") || first.contains("ellipse") || first.contains("at") {
        color_start = 1;
    }
    let stops = parse_color_stops(&parts[color_start..]);
    if stops.is_empty() {
        return transparent_solid();
    }
    with_paint(Paint::Radial(RadialGradientPaint {
        center: pt((50.0, 50.0)),
        radius_x: 50.0,
        radius_y: 50.0,
        rotation: 0.0,
        stops,
        options: Default::default(),
    }))
}

fn parse_conic_gradient(value: &str) -> Map {
    let Some(content) = gradient_content(value, "conic-gradient(") else {
        return transparent_solid();
    };
    let parts = split_gradient_args(content);
    if parts.len() < 2 {
        return transparent_solid();
    }
    let mut color_start = 0usize;
    let mut angle = 0.0f64;
    let first = parts[0].to_lowercase();
    let first = first.trim();
    if first.starts_with("from") || first.starts_with("at") {
        color_start = 1;
        // /from\s+(-?\d+(?:\.\d+)?)\s*(?:deg)?/
        if let Some(rest) = first.strip_prefix("from") {
            let rest = rest.trim_start();
            let num: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                .collect();
            if let Ok(v) = num.parse::<f64>() {
                angle = v;
            }
        }
    }
    let stops = parse_color_stops(&parts[color_start..]);
    if stops.is_empty() {
        return transparent_solid();
    }
    let mut normalized = angle % 360.0;
    if normalized < 0.0 {
        normalized += 360.0;
    }
    with_paint(Paint::Conic(ConicGradientPaint {
        center: pt((50.0, 50.0)),
        angle: normalized,
        stops,
        options: Default::default(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::parse_fills;

    /// The one fill layer's paint of a parsed bag.
    fn paint_of(m: &Map) -> Paint {
        let fills = parse_fills(m.get("fills"));
        assert_eq!(fills.len(), 1, "one fill layer: {m:?}");
        fills[0].paint.clone()
    }

    fn solid(color: &str) -> Paint {
        Paint::Solid(crate::paint::SolidPaint { color: json!(color) })
    }

    #[test]
    fn solid_and_transparent() {
        assert_eq!(paint_of(&parse_fill("#3B82F6")), solid("#3b82f6"));
        assert_eq!(paint_of(&parse_fill("none")), solid("rgba(0, 0, 0, 0)"));
    }

    #[test]
    fn linear_gradient() {
        let m = parse_fill("linear-gradient(to right, #ff0000, #0000ff 80%)");
        let Paint::Linear(g) = paint_of(&m) else {
            panic!("a linear gradient paint: {m:?}");
        };
        assert_eq!(g.start, PaintPoint { x: 0.0, y: 50.0 });
        assert_eq!(
            g.stops,
            vec![
                GradientStop { offset: 0.0, color: json!("#ff0000") },
                GradientStop { offset: 0.8, color: json!("#0000ff") },
            ]
        );
    }
}
