//! Brush colourisation (brushes.md §2.1 "Colorization", §4.4 `Colorize`):
//! Illustrator's four methods for art, scatter and pattern brushes, applied to
//! the paints of the placed artwork. This is the ONE implementation of the
//! colour math; art, scatter and pattern all call [`apply_to_node`].
//!
//! The methods, given the stroke colour S (the brushed node's `stroke`):
//!
//! - **none**: paints unchanged.
//! - **tints**: black → S, white stays white, everything else a tint of S by
//!   its luma: `mix(white, S, 1 − L)`.
//! - **tints-shades**: black and white stay, mid-grey (L = 0.5) → S, darker
//!   colours blend S toward black and lighter ones toward white.
//! - **hue-shift**: the key colour K (defaults to the most prominent chromatic
//!   paint of the artwork) becomes S; every other colour turns by the same hue
//!   angle, its saturation is remapped piecewise-linearly so K's saturation
//!   lands on S's, and its lightness moves toward S's mapping in proportion to
//!   its saturation relative to K's (so greys, black and white stay put).
//!
//! L is Rec. 601 luma of the gamma-encoded channels. The result's alpha is
//! the art colour's alpha times the stroke colour's.
//! Paints that are not plain colour strings (variable references, unparseable
//! strings) pass through unchanged; the caller resolves variables first. When
//! the stroke has no parseable colour, colourisation is skipped.

use crate::value::{Map, Value};

pub use crate::charts::colorutil::Rgba;
use crate::charts::colorutil::{hex_to_rgba, parse_rgba_color};
use crate::paint::{map_paint_colors, visit_paint_colors, ColorEdit, PaintColorSlot};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorizeMethod {
    None,
    Tints,
    TintsShades,
    HueShift,
}

/// `Colorize` (nodes.ts). `key_color` keeps the raw value (a string or a
/// variable reference); only strings are used.
#[derive(Debug, Clone, PartialEq)]
pub struct Colorize {
    pub method: ColorizeMethod,
    pub key_color: Option<Value>,
}

impl Default for Colorize {
    fn default() -> Self {
        Colorize {
            method: ColorizeMethod::None,
            key_color: None,
        }
    }
}

/// `normalizeColorize` (strokeModel.ts).
pub fn parse_colorize(raw: Option<&Value>) -> Colorize {
    let Some(o) = raw.and_then(Value::as_object) else {
        return Colorize::default();
    };
    let method = match o.get("method").and_then(Value::as_str) {
        Some("none") => ColorizeMethod::None,
        Some("tints") => ColorizeMethod::Tints,
        Some("tints-shades") => ColorizeMethod::TintsShades,
        Some("hue-shift") => ColorizeMethod::HueShift,
        _ => return Colorize::default(),
    };
    let key_color = match o.get("keyColor") {
        Some(Value::String(s)) if !s.is_empty() => Some(Value::String(s.clone())),
        Some(v @ Value::Object(_)) => Some(v.clone()),
        _ => None,
    };
    Colorize { method, key_color }
}

/// The paint colours colourisation reads and rewrites: every solid fill,
/// stroke and glyph colour and every gradient stop, through
/// `paint::visit_paint_colors` / `map_paint_colors`. The image overlay colour
/// is not artwork colour.
fn is_artwork_color(slot: PaintColorSlot) -> bool {
    !matches!(slot, PaintColorSlot::ImageOverlay { .. })
}


/// Parse a CSS colour string: `#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`,
/// `rgb()`, `rgba()`, `transparent`. Channels 0..255, alpha 0..1.
pub fn parse_color(s: &str) -> Option<Rgba> {
    let t = s.trim();
    if t.eq_ignore_ascii_case("transparent") {
        return Some(Rgba {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        });
    }
    let c = if t.starts_with('#') {
        hex_to_rgba(t)
    } else {
        parse_rgba_color(t)
    }?;
    let ok = [c.r, c.g, c.b, c.a].iter().all(|v| v.is_finite());
    ok.then(|| Rgba {
        r: c.r.clamp(0.0, 255.0),
        g: c.g.clamp(0.0, 255.0),
        b: c.b.clamp(0.0, 255.0),
        a: c.a.clamp(0.0, 1.0),
    })
}

/// `#rrggbb` when opaque, else `rgba(r, g, b, a)` (alpha to 4 decimals).
pub fn format_color(c: Rgba) -> String {
    let q = |v: f64| v.round().clamp(0.0, 255.0) as u8;
    if c.a >= 1.0 {
        format!("#{:02x}{:02x}{:02x}", q(c.r), q(c.g), q(c.b))
    } else {
        let a = (c.a.clamp(0.0, 1.0) * 10000.0).round() / 10000.0;
        format!("rgba({}, {}, {}, {})", q(c.r), q(c.g), q(c.b), a)
    }
}

/// Rec. 601 luma, 0..1.
pub fn luma(c: Rgba) -> f64 {
    (0.299 * c.r + 0.587 * c.g + 0.114 * c.b) / 255.0
}

fn mix(a: Rgba, b: Rgba, t: f64) -> Rgba {
    Rgba {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a,
    }
}

const WHITE: Rgba = Rgba {
    r: 255.0,
    g: 255.0,
    b: 255.0,
    a: 1.0,
};
const BLACK: Rgba = Rgba {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};

/// HSL, h in degrees [0, 360), s and l in 0..1.
pub fn to_hsl(c: Rgba) -> (f64, f64, f64) {
    let (r, g, b) = (c.r / 255.0, c.g / 255.0, c.b / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;
    let d = max - min;
    if d < 1e-12 {
        return (0.0, 0.0, l);
    }
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };
    let h = if max == r {
        ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    (h * 60.0, s.clamp(0.0, 1.0), l.clamp(0.0, 1.0))
}

pub fn from_hsl(h: f64, s: f64, l: f64, a: f64) -> Rgba {
    let h = h.rem_euclid(360.0) / 360.0;
    let (s, l) = (s.clamp(0.0, 1.0), l.clamp(0.0, 1.0));
    if s <= 0.0 {
        return Rgba {
            r: l * 255.0,
            g: l * 255.0,
            b: l * 255.0,
            a,
        };
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let hue = |t: f64| {
        let t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    Rgba {
        r: hue(h + 1.0 / 3.0) * 255.0,
        g: hue(h) * 255.0,
        b: hue(h - 1.0 / 3.0) * 255.0,
        a,
    }
}

/// Piecewise-linear map of [0, 1] fixing 0 and 1 and sending `from` to `to`.
fn remap01(x: f64, from: f64, to: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    if from <= 1e-9 {
        return to + (1.0 - to) * x;
    }
    if from >= 1.0 - 1e-9 {
        return to * x;
    }
    if x <= from {
        to * x / from
    } else {
        to + (1.0 - to) * (x - from) / (1.0 - from)
    }
}

/// The resolved inputs for one colourisation.
#[derive(Debug, Clone, Copy)]
pub struct ColorizeCtx {
    pub method: ColorizeMethod,
    pub stroke: Rgba,
    pub key: Rgba,
}

/// Colourise one colour. The single place the four methods are defined.
pub fn colorize_rgba(c: Rgba, ctx: &ColorizeCtx) -> Rgba {
    if ctx.method == ColorizeMethod::None {
        return c;
    }
    // The stroke colour's alpha multiplies the art's (a half-transparent
    // stroke colourises to half-transparent art).
    let alpha = c.a * ctx.stroke.a;
    let s = Rgba {
        a: alpha,
        ..ctx.stroke
    };
    let white = Rgba { a: alpha, ..WHITE };
    let black = Rgba { a: alpha, ..BLACK };
    match ctx.method {
        ColorizeMethod::None => c,
        ColorizeMethod::Tints => mix(white, s, 1.0 - luma(c)),
        ColorizeMethod::TintsShades => {
            let l = luma(c);
            if l <= 0.5 {
                mix(black, s, l / 0.5)
            } else {
                mix(s, white, (l - 0.5) / 0.5)
            }
        }
        ColorizeMethod::HueShift => {
            let (h, sat, l) = to_hsl(c);
            let (kh, ks, kl) = to_hsl(ctx.key);
            let (sh, ss, sl) = to_hsl(ctx.stroke);
            if ks <= 1e-9 {
                // An achromatic key has no hue to shift from.
                return Rgba { a: alpha, ..c };
            }
            let w = (sat / ks).min(1.0);
            let l2 = l + (remap01(l, kl, sl) - l) * w;
            let s2 = remap01(sat, ks, ss);
            let h2 = if ss <= 1e-9 { h } else { h + (sh - kh) };
            from_hsl(h2, s2, l2, alpha)
        }
    }
}

/// Colourise a colour string; `None` when it is not a parseable colour.
pub fn colorize_css(s: &str, ctx: &ColorizeCtx) -> Option<String> {
    parse_color(s).map(|c| format_color(colorize_rgba(c, ctx)))
}

/// Every colour string in a node's paints, in key order (fill, stroke, stops).
pub fn node_colors(node: &Map) -> Vec<Rgba> {
    let mut out = Vec::new();
    visit_paint_colors(node, &mut |slot, v| {
        if is_artwork_color(slot) {
            out.extend(v.as_str().and_then(parse_color));
        }
    });
    out
}

/// Illustrator's default key colour: the most prominent colour of the artwork.
/// `weighted` pairs each node's colours with a weight (its area or length);
/// chromatic colours win over greys; ties keep the first seen.
pub fn prominent_color(weighted: &[(Vec<Rgba>, f64)]) -> Option<Rgba> {
    let mut tally: Vec<(Rgba, f64, bool)> = Vec::new();
    for (colors, w) in weighted {
        for c in colors {
            let chromatic = to_hsl(*c).1 > 0.05;
            let q = |v: f64| v.round();
            let same = |a: &Rgba| q(a.r) == q(c.r) && q(a.g) == q(c.g) && q(a.b) == q(c.b);
            match tally.iter_mut().find(|(a, _, _)| same(a)) {
                Some(entry) => entry.1 += w.max(1e-9),
                None => tally.push((*c, w.max(1e-9), chromatic)),
            }
        }
    }
    let any_chromatic = tally.iter().any(|e| e.2);
    let mut best: Option<(Rgba, f64)> = None;
    for (c, w, chromatic) in tally {
        if any_chromatic && !chromatic {
            continue;
        }
        if best.is_none_or(|(_, bw)| w > bw) {
            best = Some((c, w));
        }
    }
    best.map(|(c, _)| Rgba { a: 1.0, ..c })
}

/// Resolve a `Colorize` against the stroke colour and the artwork; `None`
/// when nothing changes (method none, or no parseable stroke colour).
pub fn resolve(
    colorize: &Colorize,
    stroke: Option<&Value>,
    weighted_colors: &[(Vec<Rgba>, f64)],
) -> Option<ColorizeCtx> {
    if colorize.method == ColorizeMethod::None {
        return None;
    }
    let stroke = stroke.and_then(Value::as_str).and_then(parse_color)?;
    let key = colorize
        .key_color
        .as_ref()
        .and_then(Value::as_str)
        .and_then(parse_color)
        .or_else(|| prominent_color(weighted_colors))
        .unwrap_or(BLACK);
    Some(ColorizeCtx {
        method: colorize.method,
        stroke,
        key,
    })
}

/// Colourise every colour string of a node's paints in place.
pub fn apply_to_node(node: &mut Map, ctx: &ColorizeCtx) {
    map_paint_colors(
        node,
        &mut |slot, v| match v.as_str().filter(|_| is_artwork_color(slot)) {
            Some(s) => {
                colorize_css(s, ctx).map_or(ColorEdit::Keep, |c| ColorEdit::Set(Value::String(c)))
            }
            None => ColorEdit::Keep,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    fn c(s: &str) -> Rgba {
        parse_color(s).unwrap()
    }
    fn ctx(method: ColorizeMethod, stroke: &str, key: &str) -> ColorizeCtx {
        ColorizeCtx {
            method,
            stroke: c(stroke),
            key: c(key),
        }
    }
    fn close(a: Rgba, b: Rgba) -> bool {
        (a.r - b.r).abs() < 0.6
            && (a.g - b.g).abs() < 0.6
            && (a.b - b.b).abs() < 0.6
            && (a.a - b.a).abs() < 1e-9
    }

    #[test]
    fn parse_and_format() {
        assert_eq!(format_color(c("#f00")), "#ff0000");
        assert_eq!(
            format_color(c("rgba(0, 128, 255, 0.5)")),
            "rgba(0, 128, 255, 0.5)"
        );
        assert_eq!(format_color(c("#11223380")), "rgba(17, 34, 51, 0.5)");
        assert!(parse_color("bogus").is_none());
        assert_eq!(c("transparent").a, 0.0);
    }

    #[test]
    fn hsl_round_trips() {
        for s in [
            "#ff0000", "#12ab34", "#808080", "#ffffff", "#000000", "#3355cc",
        ] {
            let x = c(s);
            let (h, sa, l) = to_hsl(x);
            assert!(close(from_hsl(h, sa, l, 1.0), x), "{s}");
        }
    }

    #[test]
    fn tints() {
        let k = ctx(ColorizeMethod::Tints, "#ff0000", "#000");
        assert!(close(colorize_rgba(c("#000000"), &k), c("#ff0000")));
        assert!(close(colorize_rgba(c("#ffffff"), &k), c("#ffffff")));
        let grey = colorize_rgba(c("#808080"), &k);
        assert!(grey.r > 250.0 && (grey.g - 127.0).abs() < 2.0, "{grey:?}");
        assert_eq!(colorize_rgba(c("rgba(0,0,0,0.25)"), &k).a, 0.25);
        // A translucent stroke colour carries its alpha into the art.
        let half = ctx(ColorizeMethod::Tints, "rgba(255, 0, 0, 0.5)", "#000");
        assert_eq!(colorize_rgba(c("rgba(0,0,0,0.5)"), &half).a, 0.25);
        for m in [ColorizeMethod::TintsShades, ColorizeMethod::HueShift] {
            let k = ColorizeCtx { method: m, ..half };
            assert_eq!(colorize_rgba(c("#d03a2f"), &k).a, 0.5, "{m:?}");
        }
    }

    #[test]
    fn tints_and_shades() {
        let k = ctx(ColorizeMethod::TintsShades, "#3366cc", "#000");
        assert!(close(colorize_rgba(c("#000000"), &k), c("#000000")));
        assert!(close(colorize_rgba(c("#ffffff"), &k), c("#ffffff")));
        // Exactly mid luma → the stroke colour.
        let mid = Rgba {
            r: 127.5,
            g: 127.5,
            b: 127.5,
            a: 1.0,
        };
        assert!(close(colorize_rgba(mid, &k), c("#3366cc")));
    }

    #[test]
    fn hue_shift_maps_key_to_stroke_and_keeps_greys() {
        let k = ctx(ColorizeMethod::HueShift, "#2a9d4a", "#d03a2f");
        assert!(close(colorize_rgba(c("#d03a2f"), &k), c("#2a9d4a")));
        for g in ["#000000", "#ffffff", "#777777"] {
            assert!(close(colorize_rgba(c(g), &k), c(g)), "{g}");
        }
        // Another hue turns by the same angle.
        let (h0, _, _) = to_hsl(c("#d0b02f"));
        let (h1, _, _) = to_hsl(colorize_rgba(c("#d0b02f"), &k));
        let (kh, _, _) = to_hsl(c("#d03a2f"));
        let (sh, _, _) = to_hsl(c("#2a9d4a"));
        let turn = (h1 - h0 - (sh - kh)).rem_euclid(360.0);
        assert!(!(1.0..=359.0).contains(&turn), "turned {turn}");
    }

    #[test]
    fn prominent_prefers_heaviest_chromatic() {
        let w = vec![
            (vec![c("#000000")], 100.0),
            (vec![c("#ff0000")], 5.0),
            (vec![c("#00ff00")], 10.0),
            (vec![c("#ff0000")], 6.0),
        ];
        assert!(close(prominent_color(&w).unwrap(), c("#ff0000")));
        assert!(close(
            prominent_color(&[(vec![c("#222")], 1.0)]).unwrap(),
            c("#222")
        ));
        assert!(prominent_color(&[]).is_none());
    }

    #[test]
    fn apply_touches_fill_stroke_and_stops_only() {
        let mut n = json!({
            "type": "rectangle",
            "fills": [
                {"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}},
                {"id": "fill-1", "paint": {"type": "linear", "start": {"x": 0.0, "y": 0.0}, "end": {"x": 1.0, "y": 0.0},
                    "stops": [{"offset": 0.0, "color": "#000000"}, {"offset": 1.0, "color": {"id": "var"}}]}},
                {"id": "fill-2", "paint": {"type": "image", "src": "a.png",
                    "adjust": {"overlay": {"enabled": true, "color": "#000000"}}}}
            ],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#ffffff"}, "width": 1.0}],
            "effects": [{"id": "drop-shadow-0", "type": "drop-shadow", "color": "#000000", "blur": 4.0}],
        })
        .as_object()
        .unwrap()
        .clone();
        apply_to_node(&mut n, &ctx(ColorizeMethod::Tints, "#0000ff", "#000"));
        assert_eq!(n["fills"][0]["paint"]["color"], "#0000ff");
        assert_eq!(n["strokes"][0]["paint"]["color"], "#ffffff");
        assert_eq!(n["effects"][0]["color"], "#000000");
        assert_eq!(
            n["fills"][1]["paint"]["stops"],
            json!([{"offset": 0.0, "color": "#0000ff"}, {"offset": 1.0, "color": {"id": "var"}}])
        );
        // The image overlay colour is not artwork colour.
        assert_eq!(n["fills"][2]["paint"]["adjust"]["overlay"]["color"], "#000000");
    }

    #[test]
    fn resolve_needs_a_stroke_colour() {
        let tints = parse_colorize(Some(&json!({"method": "tints"})));
        assert!(resolve(&tints, Some(&json!({"id": "v"})), &[]).is_none());
        assert!(resolve(&tints, Some(&json!("#f00")), &[]).is_some());
        assert!(resolve(&Colorize::default(), Some(&json!("#f00")), &[]).is_none());
        let bad = parse_colorize(Some(&json!({"method": "sepia"})));
        assert_eq!(bad.method, ColorizeMethod::None);
    }
}
