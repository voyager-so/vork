//! A gradient on a text range: the rich-text `--nf-text-fill` style (twin of
//! `utils/richtext/textFill.ts`).
//!
//! A span (or paragraph) carries its gradient as a custom CSS property, in the
//! node's 0-100 box space like every Moda paint (Figma maps a range's gradient
//! over the whole text node's box too):
//!
//! ```text
//! --nf-text-fill: linear(<x1>% <y1>%, <x2>% <y2>%, <color> <offset>%, …)
//! --nf-text-fill: radial(<cx>% <cy>%, <rx>% <ry>%, <rotation>deg, <color> <offset>%, …)
//! ```
//!
//! It inherits like `color`; a `color` declared without a fill clears an
//! inherited one (the run is solid again). The run's `color` stays the first
//! stop, so lanes that draw no run gradients (MSDF, degraded) still draw a
//! sensible solid.

use crate::ops::{parse_gradient_stops, GradientSpec, TextGradientUniforms};
use serde_json::Value;

/// The CSS property.
pub const TEXT_FILL_PROPERTY: &str = "--nf-text-fill";

/// Split `s` at top-level commas (commas inside parentheses stay).
fn split_top_level(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(s[start..].trim());
    out
}

/// `<n>%` or `<n>` → n.
fn parse_percent(s: &str) -> Option<f64> {
    let t = s.trim();
    let t = t.strip_suffix('%').unwrap_or(t);
    let v: f64 = t.trim().parse().ok()?;
    v.is_finite().then_some(v)
}

/// `<a>% <b>%` → (a, b).
fn parse_pair(s: &str) -> Option<(f64, f64)> {
    let mut it = s.split_whitespace();
    let a = parse_percent(it.next()?)?;
    let b = parse_percent(it.next()?)?;
    it.next().is_none().then_some((a, b))
}

/// `<color> <offset>%` → (css color, 0..1 offset). The color may hold spaces
/// inside its parentheses, so the offset is the text after the last `)` or space.
fn parse_stop(s: &str) -> Option<(String, f64)> {
    let t = s.trim();
    let split = t.rfind(|c: char| c.is_whitespace())?;
    let (color, offset) = (t[..split].trim(), t[split..].trim());
    if color.is_empty() {
        return None;
    }
    Some((color.to_string(), parse_percent(offset)? / 100.0))
}

/// Parse a `--nf-text-fill` value. `None` for anything else (`none`, malformed).
pub fn parse_text_fill(value: &str) -> Option<GradientSpec> {
    let v = value.trim();
    let lower = v.to_ascii_lowercase();
    let (fill_type, body) = if lower.starts_with("linear(") {
        (1u32, &v["linear(".len()..])
    } else if lower.starts_with("radial(") {
        (2u32, &v["radial(".len()..])
    } else {
        return None;
    };
    let body = body.trim_end().strip_suffix(')')?;
    let mut parts = split_top_level(body);
    // An optional last part `flags <n>`: the interpolation space + 8 × dither (textFill.ts writes it when non-zero).
    let mut flags = 0.0f32;
    if let Some(last) = parts.last().and_then(|p| p.trim().strip_prefix("flags ")) {
        flags = last
            .trim()
            .parse()
            .ok()
            .filter(|f: &f32| f.is_finite() && *f >= 0.0)?;
        parts.pop();
    }
    let (start, end, rotation, stop_parts) = if fill_type == 1 {
        if parts.len() < 3 {
            return None;
        }
        (
            parse_pair(parts[0])?,
            parse_pair(parts[1])?,
            0.0,
            &parts[2..],
        )
    } else {
        if parts.len() < 4 {
            return None;
        }
        let deg: f64 = parts[2]
            .trim()
            .strip_suffix("deg")
            .unwrap_or(parts[2])
            .trim()
            .parse()
            .ok()?;
        (
            parse_pair(parts[0])?,
            parse_pair(parts[1])?,
            // The shader wants radians (text_draw.rs `extract_fill`).
            deg * std::f64::consts::PI / 180.0,
            &parts[3..],
        )
    };
    let mut flat: Vec<Value> = Vec::new();
    for p in stop_parts {
        let (color, offset) = parse_stop(p)?;
        flat.push(Value::from(offset));
        flat.push(Value::from(color));
    }
    if flat.is_empty() {
        return None;
    }
    let mut stops = parse_gradient_stops(Some(&flat));
    if flags as u32 & 16 != 0 {
        stops = flat.chunks(2).map(|pair| {
            let color = parse_gradient_stops(Some(&vec![Value::from(0), pair[1].clone()]))[0].1;
            (pair[0].as_f64().unwrap_or(0.0), color)
        }).take(8).collect();
    }
    Some(GradientSpec {
        fill_type,
        start,
        end,
        rotation,
        stops,
        flags,
    })
}

/// The gradient's shader uniforms over a `box_w` × `box_h` text box
/// (`packTextGradientUniformFloats`; the node-level gradient packs the same way).
pub fn gradient_uniforms(g: &GradientSpec, box_w: f64, box_h: f64) -> TextGradientUniforms {
    const MAX_STOPS: usize = 8;
    let num_stops = g.stops.len().min(MAX_STOPS);
    let mut stops = [[0.0f32; 4]; 8];
    for (i, (off, c)) in g.stops.iter().take(MAX_STOPS).enumerate() {
        let packed = (c[2] * 255.0).floor() + c[3] * 0.99999;
        stops[i] = [*off as f32, c[0] as f32, c[1] as f32, packed as f32];
    }
    TextGradientUniforms {
        text_bounds: [0.0, 0.0, box_w as f32, box_h as f32],
        params: [
            g.start.0 as f32,
            g.start.1 as f32,
            g.end.0 as f32,
            g.end.1 as f32,
        ],
        config: [
            num_stops as f32,
            g.fill_type as f32,
            g.rotation as f32,
            g.flags,
        ],
        stops,
    }
}

#[cfg(test)]
#[cfg_attr(not(vork_private_tests), allow(dead_code, unused_imports))]
mod tests {
    use super::*;

    #[test]
    fn repeating_run_fill_preserves_negative_and_coincident_stops() {
        let gradient = parse_text_fill("linear(0% 50%, 100% 50%, #000000 -10%, #000000 0%, #ffffff 0%, #ffffff 10%, flags 16)").unwrap();
        assert_eq!(gradient.flags, 16.0);
        assert_eq!(gradient.stops.iter().map(|stop| stop.0).collect::<Vec<_>>(), [-0.1, 0.0, 0.0, 0.1]);
    }

    #[test]
    fn a_trailing_flags_part_carries_interpolation_and_dither() {
        let g = parse_text_fill("linear(0% 50%, 100% 50%, rgb(255, 0, 0) 0%, rgb(0, 255, 255) 100%, flags 11)").unwrap();
        assert_eq!(g.flags, 11.0);
        assert_eq!(g.stops.len(), 2);
        assert_eq!(gradient_uniforms(&g, 10.0, 10.0).config[3], 11.0);
        let plain = parse_text_fill("linear(0% 50%, 100% 50%, rgb(255, 0, 0) 0%, rgb(0, 255, 255) 100%)").unwrap();
        assert_eq!(plain.flags, 0.0);
        assert!(parse_text_fill("linear(0% 50%, 100% 50%, rgb(255, 0, 0) 0%, flags x)").is_none());
    }

    #[test]
    fn parses_linear() {
        let g = parse_text_fill(
            "linear(0% 0%, 100% 0%, rgb(167, 139, 250) 0%, rgba(244, 114, 182, 0.5) 100%)",
        )
        .unwrap();
        assert_eq!(g.fill_type, 1);
        assert_eq!(g.start, (0.0, 0.0));
        assert_eq!(g.end, (100.0, 0.0));
        assert_eq!(g.stops.len(), 2);
        assert!((g.stops[0].1[0] - 167.0 / 255.0).abs() < 1e-9);
        assert_eq!(g.stops[1].0, 1.0);
        assert!((g.stops[1].1[3] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn parses_radial() {
        let g =
            parse_text_fill("radial(50% 40%, 30% 20%, 90deg, #ff0000 0%, #0000ff 100%)").unwrap();
        assert_eq!(g.fill_type, 2);
        assert_eq!(g.start, (50.0, 40.0));
        assert_eq!(g.end, (30.0, 20.0));
        assert!((g.rotation - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        assert_eq!(g.stops[1].1, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn inherits_like_color_and_a_declared_color_clears_it() {
        use crate::parser::parse_rich_text_to_paragraphs;
        let html = "<p style=\"color: #111111; --nf-text-fill: linear(0% 0%, 100% 0%, #ff0000 0%, #0000ff 100%)\">\
                    A<span style=\"font-weight: 700\">B</span><span style=\"color: #00ff00\">C</span></p>";
        let paras = parse_rich_text_to_paragraphs(html);
        let runs = &paras[0].runs;
        let fill_of = |t: &str| {
            runs.iter()
                .find(|r| r.text == t)
                .unwrap()
                .style
                .fill
                .is_some()
        };
        assert!(fill_of("A"));
        assert!(fill_of("B"), "a nested span inherits the fill");
        assert!(!fill_of("C"), "a declared colour without a fill is solid");
    }

    /// A gradient on one range draws that range's glyphs through a gradient op (its own uniforms, over the node
    /// box) and leaves the rest of the text solid.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn a_range_gradient_draws_only_that_range() {
        use crate::ops::TextNodeSpec;
        use crate::vector::{
            build_vector_node_ops, VectorEngineDecision, VectorOpPipeline, VectorPlacement,
        };
        let (store, catalog) = crate::test_fixtures::store_and_catalog();
        let spec = TextNodeSpec {
            node_id: "t".into(),
            html: "<p style=\"font-family: Inter; font-size: 32px; color: rgb(255, 255, 255)\">Plan \
                   <span style=\"color: rgb(167, 139, 250); --nf-text-fill: linear(0% 0%, 100% 0%, rgb(167, 139, 250) 0%, rgb(244, 114, 182) 100%)\">one</span></p>"
                .into(),
            width: Some(400.0),
            height: Some(96.0),
            opacity: 1.0,
            ..Default::default()
        };
        let VectorEngineDecision::Vector(ops) =
            build_vector_node_ops(&spec, &store, &catalog, 1.0, VectorPlacement::VectorParity)
        else {
            panic!("vector lane");
        };
        assert!(ops.gradient.is_none(), "no node-level gradient");
        let solid: Vec<_> = ops
            .ops
            .iter()
            .filter(|o| o.pipeline == VectorOpPipeline::Analytic)
            .collect();
        let grad: Vec<_> = ops
            .ops
            .iter()
            .filter(|o| o.pipeline == VectorOpPipeline::AnalyticGradient)
            .collect();
        assert_eq!(solid.len(), 1);
        assert_eq!(grad.len(), 1);
        assert_eq!(solid[0].records.len(), 4, "P l a n");
        assert_eq!(grad[0].records.len(), 3, "o n e");
        assert_eq!(solid[0].color, [1.0, 1.0, 1.0, 1.0]);
        let g = grad[0].gradient.as_ref().expect("the range's own uniforms");
        assert_eq!(g.text_bounds, [0.0, 0.0, 400.0, 96.0]);
        assert_eq!(g.params, [0.0, 0.0, 100.0, 0.0]);
        assert_eq!(g.config[..2], [2.0, 1.0]);
        // The range's glyphs sit right of the solid ones.
        let max_solid = solid[0]
            .records
            .iter()
            .map(|r| r.pen_x)
            .fold(f32::MIN, f32::max);
        assert!(grad[0].records.iter().all(|r| r.pen_x > max_solid));

        // A node-level fill overrides every run, the range's gradient included.
        let mut overridden = spec.clone();
        overridden.fill_override = Some("#000000".into());
        let VectorEngineDecision::Vector(ops) = build_vector_node_ops(
            &overridden,
            &store,
            &catalog,
            1.0,
            VectorPlacement::VectorParity,
        ) else {
            panic!("vector lane");
        };
        assert!(ops
            .ops
            .iter()
            .all(|o| o.pipeline != VectorOpPipeline::AnalyticGradient));
    }

    #[test]
    fn rejects_other_values() {
        assert!(parse_text_fill("none").is_none());
        assert!(parse_text_fill("linear(0% 0%)").is_none());
        assert!(parse_text_fill("conic(50% 50%, 0deg, #fff 0%)").is_none());
    }
}
