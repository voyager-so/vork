//! A laid-out text layer's metrics, per line and per code point, for scripts (`text.metrics`, moda-wasm
//! `text_metrics`) and for the reflow transition (`reflow` / `reflowWidth`, [`apply_reflow`]).
//!
//! Everything comes from the engine's own layout (`moda_text::vector::layout_rich_text_node_detail`, the walk the
//! canvas draws), in node-local px, y down:
//!
//! - a **line**: its text, left edge and width, top and height, baseline, and its ascent and descent (the largest
//!   of its runs' fonts, px above and below the baseline);
//! - a **code point** of the layer's plain text (the index space of `text.units` and `unitTransforms`): its pen x,
//!   advance (letter spacing included; the code points of a ligature share its advance evenly), baseline, its
//!   font's ascent and descent and its font size, and its line. A code point the layout drops (collapsed or
//!   wrapped whitespace) has none.
//!
//! Units (`char`, `word`, `line`) group code points the way `text.units` does: whitespace never belongs to one,
//! a word is a run of non-whitespace, a line is a laid-out line (soft wraps included).

use moda_scene::types::RawNode;
use moda_scene::Scene;
use moda_text::RichTextNodeSpec;
use serde_json::{json, Value};

/// One laid-out line.
#[derive(Debug, Clone, PartialEq)]
pub struct LineMetrics {
    pub text: String,
    pub x: f64,
    pub width: f64,
    pub top: f64,
    pub height: f64,
    pub baseline: f64,
    pub ascent: f64,
    pub descent: f64,
}

/// One code point where the layout placed it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharMetrics {
    pub x: f64,
    pub advance: f64,
    pub baseline: f64,
    pub ascent: f64,
    pub descent: f64,
    pub font_size: f64,
    pub line: usize,
}

/// A text layer's layout: its plain text, its lines, and each code point's place (`None`: not laid out).
#[derive(Debug, Clone, Default)]
pub struct TextMetrics {
    pub text: Vec<char>,
    pub lines: Vec<LineMetrics>,
    pub chars: Vec<Option<CharMetrics>>,
    pub height: f64,
}

/// The unit kinds of `text.units`.
pub const UNIT_KINDS: &[&str] = &["char", "word", "line"];

/// The layout spec of a richtext node, as the canvas lays it out (its area type included).
pub fn node_rich_spec(scene: &Scene, node: &RawNode) -> RichTextNodeSpec {
    RichTextNodeSpec {
        node_id: node.id().to_string(),
        html_content: crate::text_draw::node_html_in(scene, node),
        width: node.f64("width"),
        height: node.f64("height"),
        paragraph_spacing: node.f64("paragraphSpacing").unwrap_or(0.0),
        line_height_policy: Default::default(),
        vertical_align: node.str("verticalAlign").map(str::to_string),
        truncation: None,
        auto_width: crate::text_draw::node_auto_width(node),
        area: crate::text_area::node_area_spec(scene, node),
    }
}

/// The layout's metrics, or why there are none (text the vector lane doesn't lay out: an unknown font, a complex
/// script, colour emoji).
pub fn rich_text_metrics(spec: &RichTextNodeSpec) -> Result<TextMetrics, String> {
    let (store, host) = crate::text_draw::vector_store_and_host().map_err(|e| format!("{e:#}"))?;
    let dump = moda_text::vector::layout_rich_text_node_detail(spec, &store, &host.catalog);
    if dump.get("engine").and_then(Value::as_str) != Some("vector") {
        let why = dump.get("reason").and_then(Value::as_str).unwrap_or("not vector text");
        return Err(format!("the text engine doesn't lay this text out as vector text ({why})"));
    }
    Ok(metrics_from_layout(&moda_text::parser::parse_rich_text_with_source(&spec.html_content).source_text, &dump))
}

fn f(v: &Value, k: &str) -> f64 {
    v.get(k).and_then(Value::as_f64).unwrap_or(0.0)
}

fn arr<'a>(v: &'a Value, k: &str) -> &'a [Value] {
    v.get(k).and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

/// Metrics use the source indices carried by the detailed layout, including whitespace
/// and case expansions; source characters removed by layout remain unplaced.
pub fn metrics_from_layout(plain: &str, dump: &Value) -> TextMetrics {
    let text: Vec<char> = plain.chars().collect();
    let mut chars: Vec<Option<CharMetrics>> = vec![None; text.len()];
    let mut lines = Vec::new();
    for para in arr(dump, "paragraphs") {
        for line in arr(para, "lines") {
            if line.get("hidden").and_then(Value::as_bool) == Some(true) {
                continue;
            }
            let index = lines.len();
            let (baseline, ascent, descent) = (f(line, "baselineY"), f(line, "ascent"), f(line, "descent"));
            let segments = arr(line, "segments");
            let mut line_text = String::new();
            for seg in segments {
                let seg_text: Vec<char> = seg.get("text").and_then(Value::as_str).unwrap_or("").chars().collect();
                line_text.extend(seg_text.iter());
                let plain_of: Vec<Option<usize>> = arr(seg, "sourceIndices").iter().map(|v| v.as_u64().map(|i| i as usize)).collect();
                if seg.get("whitespaceOnly").and_then(Value::as_bool) == Some(true) {
                    let gap: Vec<usize> = plain_of.iter().copied().flatten().filter(|&cp| cp < chars.len()).collect();
                    place_whitespace(&mut chars, &gap, &[seg], index, baseline, ascent, descent);
                    continue;
                }
                let (seg_ascent, seg_descent) = match (seg.get("ascent"), seg.get("descent")) {
                    (Some(a), Some(d)) => (a.as_f64().unwrap_or(ascent), d.as_f64().unwrap_or(descent)),
                    _ => (ascent, descent),
                };
                let size = f(seg, "fontSize");
                if seg.get("emoji").and_then(Value::as_bool) == Some(true) {
                    // An emoji segment has no glyphs: each emoji (a grapheme, ZWJ sequences whole) takes an equal cell
                    // of the segment. Without this its code points had no metrics, so a reflow left it behind.
                    use unicode_segmentation::UnicodeSegmentation;
                    let s: String = seg_text.iter().collect();
                    let cells: Vec<usize> = s.graphemes(true).map(|g| g.chars().count()).collect();
                    let cell = f(seg, "width") / cells.len().max(1) as f64;
                    let mut cp = 0;
                    for (k, n) in cells.into_iter().enumerate() {
                        for p in plain_of.iter().skip(cp).take(n).copied().flatten() {
                            let Some(slot) = chars.get_mut(p) else { continue };
                            *slot = Some(CharMetrics {
                                x: f(seg, "x") + cell * k as f64,
                                advance: cell,
                                baseline,
                                ascent: seg_ascent,
                                descent: seg_descent,
                                font_size: size,
                                line: index,
                            });
                        }
                        cp += n;
                    }
                    continue;
                }
                let glyphs = arr(seg, "glyphs");
                let clusters: Vec<usize> = glyphs.iter().map(|g| g.get("c").and_then(Value::as_u64).unwrap_or(0) as usize).collect();
                for (k, g) in glyphs.iter().enumerate() {
                    let c = clusters[k];
                    // A cluster's first glyph places it; its code points share the cluster's advance.
                    if k > 0 && clusters[k - 1] == c {
                        continue;
                    }
                    let end_k = (k..glyphs.len()).find(|&j| clusters[j] != c).unwrap_or(glyphs.len());
                    let advance: f64 = glyphs[k..end_k].iter().map(|g| f(g, "a")).sum();
                    let next_c = clusters.get(end_k).copied().unwrap_or(seg_text.len()).max(c + 1);
                    let n = (next_c - c) as f64;
                    for (j, cp) in (c..next_c).enumerate() {
                        // The plain code point this char draws; the chars of one expansion (`SS` for `ß`) add up.
                        let Some(p) = plain_of.get(cp).copied().flatten() else { continue };
                        let expands = cp > 0 && plain_of.get(cp - 1).copied().flatten() == Some(p);
                        if let Some(slot) = chars.get_mut(p) {
                            match slot {
                                Some(m) if expands => m.advance += advance / n,
                                _ => {
                                    *slot = Some(CharMetrics {
                                        x: f(g, "x") + advance * j as f64 / n,
                                        advance: advance / n,
                                        baseline,
                                        ascent: seg_ascent,
                                        descent: seg_descent,
                                        font_size: size,
                                        line: index,
                                    })
                                }
                            }
                        }
                    }
                }
            }
            let x = segments.first().map_or(f(line, "lineStartX"), |s| f(s, "x"));
            lines.push(LineMetrics {
                text: line_text,
                x,
                width: f(line, "totalWidth"),
                top: f(line, "lineTopY"),
                height: f(line, "lineHeightPx"),
                baseline,
                ascent,
                descent,
            });
        }
    }
    TextMetrics { text, lines, chars, height: f(dump, "totalHeightPx") }
}

/// Whitespace code points `gap` laid out by the whitespace-only segments `segs` (each segment's width spread over
/// its characters, in order); the code points left over stay unplaced.
fn place_whitespace(
    chars: &mut [Option<CharMetrics>],
    gap: &[usize],
    segs: &[&Value],
    line: usize,
    baseline: f64,
    ascent: f64,
    descent: f64,
) {
    let mut at = gap.iter();
    for seg in segs {
        let n = seg.get("text").and_then(Value::as_str).unwrap_or("").chars().count();
        if n == 0 {
            continue;
        }
        let each = f(seg, "width") / n as f64;
        for j in 0..n {
            let Some(&i) = at.next() else { return };
            chars[i] = Some(CharMetrics {
                x: f(seg, "x") + each * j as f64,
                advance: each,
                baseline,
                ascent,
                descent,
                font_size: f(seg, "fontSize"),
                line,
            });
        }
    }
}

/// The code points of each unit, by `char`, `word` or `line` (`text.units`' units; lines as laid out).
pub fn units(m: &TextMetrics, by: &str) -> Vec<Vec<usize>> {
    // A char is an extended grapheme cluster (a letter with its marks, an emoji sequence), as `text.units` counts.
    let mut grapheme_start = vec![false; m.text.len()];
    if by == "char" {
        use unicode_segmentation::UnicodeSegmentation;
        let text: String = m.text.iter().collect();
        let mut i = 0usize;
        for g in text.graphemes(true) {
            if let Some(s) = grapheme_start.get_mut(i) {
                *s = true;
            }
            i += g.chars().count();
        }
    }
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut open = false;
    let mut last_line = usize::MAX;
    for (i, c) in m.text.iter().enumerate() {
        if c.is_whitespace() {
            open = false;
            continue;
        }
        match by {
            "char" if open && !grapheme_start[i] => {}
            "line" => {
                let Some(line) = m.chars[i].map(|c| c.line) else { continue };
                if line != last_line || out.is_empty() {
                    out.push(Vec::new());
                    last_line = line;
                }
            }
            "word" if open => {}
            _ => out.push(Vec::new()),
        }
        open = true;
        out.last_mut().expect("a unit").push(i);
    }
    out
}

fn round(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

/// The metrics as scripts read them: the layer (`width`, `height`, `firstBaseline`, `lastBaseline`, `ascent`,
/// `descent`, `lineCount`), its `lines`, and its `units` by `by` (1-based `unit` numbers as `text.units(units:)`
/// counts them; `start` / `end` code point indices, end exclusive; `x`, `advance`, `baseline`, `ascent`, `descent`,
/// `top` (baseline − ascent), `bottom` (baseline + descent), `fontSize`, `line`, all px in the layer).
pub fn metrics_json(m: &TextMetrics, by: &str) -> Value {
    let lines: Vec<Value> = m
        .lines
        .iter()
        .enumerate()
        .map(|(i, l)| {
            json!({
                "line": i + 1, "text": l.text, "x": round(l.x), "width": round(l.width), "top": round(l.top),
                "height": round(l.height), "baseline": round(l.baseline), "ascent": round(l.ascent),
                "descent": round(l.descent),
            })
        })
        .collect();
    let units: Vec<Value> = units(m, by)
        .iter()
        .enumerate()
        .filter_map(|(u, cps)| {
            let placed: Vec<(usize, CharMetrics)> = cps.iter().filter_map(|&i| m.chars[i].map(|c| (i, c))).collect();
            let (_, first) = *placed.first()?;
            let x0 = placed.iter().map(|(_, c)| c.x).fold(f64::INFINITY, f64::min);
            let x1 = placed.iter().map(|(_, c)| c.x + c.advance).fold(f64::NEG_INFINITY, f64::max);
            let ascent = placed.iter().map(|(_, c)| c.ascent).fold(0.0, f64::max);
            let descent = placed.iter().map(|(_, c)| c.descent).fold(0.0, f64::max);
            let size = placed.iter().map(|(_, c)| c.font_size).fold(0.0, f64::max);
            let start = cps[0];
            let end = cps[cps.len() - 1] + 1;
            Some(json!({
                "unit": u + 1, "text": m.text[start..end].iter().collect::<String>(), "start": start, "end": end,
                "x": round(x0), "advance": round(x1 - x0), "baseline": round(first.baseline), "ascent": round(ascent),
                "descent": round(descent), "top": round(first.baseline - ascent), "bottom": round(first.baseline + descent),
                "fontSize": round(size), "line": first.line + 1,
            }))
        })
        .collect();
    let width = m.lines.iter().map(|l| l.width).fold(0.0, f64::max);
    json!({
        "text": m.text.iter().collect::<String>(),
        "by": by,
        "layer": {
            "width": round(width),
            "height": round(m.height),
            "lineCount": m.lines.len(),
            "firstBaseline": m.lines.first().map(|l| round(l.baseline)),
            "lastBaseline": m.lines.last().map(|l| round(l.baseline)),
            "ascent": m.lines.first().map(|l| round(l.ascent)),
            "descent": m.lines.last().map(|l| round(l.descent)),
        },
        "lines": lines,
        "units": units,
    })
}

/// A richtext node's metrics JSON ([`metrics_json`]).
pub fn node_text_metrics(scene: &Scene, node_id: &str, by: &str) -> Result<Value, String> {
    let node = scene.effective_node(node_id).ok_or_else(|| format!("no node {node_id}"))?;
    if node.str("type") != Some("richtext") {
        return Err(format!("{node_id} isn't a text layer"));
    }
    let m = rich_text_metrics(&node_rich_spec(scene, node))?;
    Ok(metrics_json(&m, by))
}

/// The reflow transition (`reflow`: 0…1, `reflowWidth`: px): each code point moves from where the text lays out
/// at the node's width to where it lays out at `reflowWidth` (wrapped and aligned in a box that wide, from the same
/// top-left), `reflow` of the way, on top of any `unitTransforms`. At 0, or without a `reflowWidth`, nothing
/// changes. A code point laid out in only one of the two stays where it is.
pub fn apply_reflow(node: &RawNode, spec: &mut moda_text::TextNodeSpec) {
    let progress = node.f64("reflow").filter(|p| p.is_finite()).unwrap_or(0.0);
    let Some(to) = node.f64("reflowWidth").filter(|w| w.is_finite() && *w > 0.0) else {
        return;
    };
    if progress == 0.0 || spec.text_path.is_some() {
        return;
    }
    let rich = |width: Option<f64>, auto_width: bool| RichTextNodeSpec {
        node_id: spec.node_id.clone(),
        html_content: spec.html.to_string(),
        width,
        height: spec.height,
        paragraph_spacing: spec.paragraph_spacing,
        line_height_policy: spec.line_height_policy,
        vertical_align: spec.vertical_align.clone(),
        truncation: None,
        auto_width,
        area: spec.area.clone(),
    };
    let (Ok(a), Ok(b)) = (rich_text_metrics(&rich(spec.width, spec.auto_width)), rich_text_metrics(&rich(Some(to), false))) else {
        return;
    };
    let deltas = reflow_deltas(&a, &b, progress);
    if deltas.iter().all(|d| d.is_none()) {
        return;
    }
    fold_offsets(std::sync::Arc::make_mut(&mut spec.unit_transforms), &deltas);
}

/// Each code point's move from layout `a` to layout `b`, `progress` of the way (`None`: not laid out in both).
pub fn reflow_deltas(a: &TextMetrics, b: &TextMetrics, progress: f64) -> Vec<Option<(f64, f64)>> {
    a.chars
        .iter()
        .zip(b.chars.iter())
        .map(|(a, b)| match (a, b) {
            (Some(a), Some(b)) => Some(((b.x - a.x) * progress, (b.baseline - a.baseline) * progress)),
            _ => None,
        })
        .collect()
}

/// Add per-code-point offsets to the node's glyph transforms (identity where it has none).
pub fn fold_offsets(xforms: &mut Vec<moda_text::vector::ops::GlyphXform>, deltas: &[Option<(f64, f64)>]) {
    use moda_text::vector::ops::GlyphXform;
    if xforms.len() < deltas.len() {
        xforms.resize(deltas.len(), GlyphXform::IDENTITY);
    }
    for (x, d) in xforms.iter_mut().zip(deltas) {
        if let Some((dx, dy)) = d {
            x.off[0] += *dx as f32;
            x.off[1] += *dy as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_sources(mut dump: Value, sources: &[&[usize]]) -> Value {
        let mut source = sources.iter();
        for p in dump["paragraphs"].as_array_mut().unwrap() {
            for l in p["lines"].as_array_mut().unwrap() {
                for s in l["segments"].as_array_mut().unwrap() {
                    s["sourceIndices"] = json!(source.next().unwrap());
                }
            }
        }
        assert!(source.next().is_none());
        dump
    }

    fn seg(text: &str, x: f64, adv: f64) -> Value {
        let glyphs: Vec<Value> =
            text.chars().enumerate().map(|(i, _)| json!({ "g": 1, "x": x + adv * i as f64, "c": i, "a": adv })).collect();
        json!({ "text": text, "x": x, "width": adv * text.chars().count() as f64, "fontSize": 20.0,
                "ascent": 18.0, "descent": 5.0, "glyphs": glyphs })
    }

    fn space(x: f64, w: f64) -> Value {
        json!({ "text": " ", "x": x, "width": w, "fontSize": 20.0, "whitespaceOnly": true })
    }

    /// "ab cd" wrapped to two lines (the space dropped at the wrap).
    fn two_lines() -> Value {
        with_sources(json!({ "engine": "vector", "totalHeightPx": 48.0, "paragraphs": [{ "lines": [
            { "lineTopY": 0.0, "lineHeightPx": 24.0, "baselineY": 19.0, "ascent": 18.0, "descent": 5.0,
              "totalWidth": 20.0, "segments": [seg("ab", 0.0, 10.0)] },
            { "lineTopY": 24.0, "lineHeightPx": 24.0, "baselineY": 43.0, "ascent": 18.0, "descent": 5.0,
              "totalWidth": 20.0, "segments": [seg("cd", 0.0, 10.0)] },
        ]}]}), &[&[0,1], &[3,4]])
    }

    #[test]
    fn code_points_take_their_glyphs_places() {
        let m = metrics_from_layout("ab cd", &two_lines());
        let x: Vec<Option<(f64, f64)>> = m.chars.iter().map(|c| c.map(|c| (c.x, c.baseline))).collect();
        assert_eq!(x, vec![Some((0.0, 19.0)), Some((10.0, 19.0)), None, Some((0.0, 43.0)), Some((10.0, 43.0))]);
        assert_eq!(m.chars[4].unwrap().line, 1);
        assert_eq!(m.chars[0].unwrap().advance, 10.0);
        assert_eq!(m.lines.len(), 2);
        assert_eq!(m.lines[1].text, "cd");
    }

    #[test]
    fn a_char_unit_is_a_grapheme_as_text_units_counts() {
        // "e" + U+0301 then "x": two chars (the accent stays with its letter), numbered as text.units numbers them.
        let dump = json!({ "engine": "vector", "totalHeightPx": 24.0, "paragraphs": [{ "lines": [
            { "lineTopY": 0.0, "lineHeightPx": 24.0, "baselineY": 19.0, "ascent": 18.0, "descent": 5.0,
              "totalWidth": 30.0, "segments": [seg("e\u{301}x", 0.0, 10.0)] }]}]});
        let dump = with_sources(dump, &[&[0,1,2]]);
        let m = metrics_from_layout("e\u{301}x", &dump);
        assert_eq!(units(&m, "char"), vec![vec![0, 1], vec![2]]);
        assert_eq!(units(&m, "word"), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn an_emoji_takes_its_cell_so_it_reflows_with_the_text() {
        // An emoji segment has no glyphs; its code points take its cell (a ZWJ sequence all of them). Before, the emoji
        // had no metrics, so a reflow left it where it was while the text moved.
        let emoji = |text: &str, x: f64, w: f64| json!({ "text": text, "x": x, "width": w, "fontSize": 20.0, "emoji": true });
        let dump = json!({ "engine": "vector", "totalHeightPx": 24.0, "paragraphs": [{ "lines": [
            { "lineTopY": 0.0, "lineHeightPx": 24.0, "baselineY": 19.0, "ascent": 18.0, "descent": 5.0,
              "totalWidth": 110.0, "segments": [seg("ab", 0.0, 10.0), space(20.0, 10.0), emoji("🚀", 30.0, 40.0), space(70.0, 10.0), seg("cd", 80.0, 10.0)] }]}]});
        let dump = with_sources(dump, &[&[0,1], &[2], &[3], &[4], &[5,6]]);
        let m = metrics_from_layout("ab 🚀 cd", &dump);
        let x: Vec<Option<f64>> = m.chars.iter().map(|c| c.map(|c| c.x)).collect();
        assert_eq!(x, vec![Some(0.0), Some(10.0), Some(20.0), Some(30.0), Some(70.0), Some(80.0), Some(90.0)]);
        let family = "👨\u{200D}👩\u{200D}👧";
        let dump = json!({ "engine": "vector", "totalHeightPx": 24.0, "paragraphs": [{ "lines": [
            { "lineTopY": 0.0, "lineHeightPx": 24.0, "baselineY": 19.0, "ascent": 18.0, "descent": 5.0,
              "totalWidth": 90.0, "segments": [emoji(&format!("{family}🚀"), 0.0, 80.0), seg("x", 80.0, 10.0)] }]}]});
        let dump = with_sources(dump, &[&[0,1,2,3,4,5], &[6]]);
        let m = metrics_from_layout(&format!("{family}🚀x"), &dump);
        let x: Vec<Option<f64>> = m.chars.iter().map(|c| c.map(|c| c.x)).collect();
        assert_eq!(x, [vec![Some(0.0); 5], vec![Some(40.0), Some(80.0)]].concat());
    }

    #[test]
    fn a_ligature_shares_its_advance_and_spaces_take_their_segment() {
        // "fi x" with an "fi" ligature (one glyph for code points 0 and 1), a space segment, then "x".
        let lig = json!({ "text": "fi", "x": 0.0, "width": 12.0, "fontSize": 20.0, "ascent": 18.0, "descent": 5.0,
                          "glyphs": [{ "g": 7, "x": 0.0, "c": 0, "a": 12.0 }] });
        let dump = json!({ "engine": "vector", "totalHeightPx": 24.0, "paragraphs": [{ "lines": [
            { "lineTopY": 0.0, "lineHeightPx": 24.0, "baselineY": 19.0, "ascent": 18.0, "descent": 5.0,
              "totalWidth": 28.0, "segments": [lig, space(12.0, 6.0), seg("x", 18.0, 10.0)] }]}]});
        let dump = with_sources(dump, &[&[0,1], &[2], &[3]]);
        let m = metrics_from_layout("fi x", &dump);
        let placed: Vec<(f64, f64)> = m.chars.iter().map(|c| c.map(|c| (c.x, c.advance)).unwrap()).collect();
        assert_eq!(placed, vec![(0.0, 6.0), (6.0, 6.0), (12.0, 6.0), (18.0, 10.0)]);
    }

    #[test]
    fn a_case_mapping_that_changes_length_keeps_its_code_points() {
        // text-case="upper": the layout draws "STRASSE FINE", the plain text is "straße ﬁne" (ß → SS, ﬁ → FI). The
        // segments used to match nowhere, so both words had no metrics and text.metrics(by: 'word') dropped them.
        let dump = json!({ "engine": "vector", "totalHeightPx": 24.0, "paragraphs": [{ "lines": [
            { "lineTopY": 0.0, "lineHeightPx": 24.0, "baselineY": 19.0, "ascent": 18.0, "descent": 5.0,
              "totalWidth": 120.0, "segments": [seg("STRASSE", 0.0, 10.0), space(70.0, 10.0), seg("FINE", 80.0, 10.0)] }]}]});
        let dump = with_sources(dump, &[&[0,1,2,3,4,4,5], &[6], &[7,7,8,9]]);
        let m = metrics_from_layout("straße ﬁne", &dump);
        let placed: Vec<(f64, f64)> = m.chars.iter().map(|c| c.map(|c| (c.x, c.advance)).unwrap()).collect();
        // s t r a ß(SS) e ␠ ﬁ(FI) n e
        assert_eq!(
            placed,
            vec![(0.0, 10.0), (10.0, 10.0), (20.0, 10.0), (30.0, 10.0), (40.0, 20.0), (60.0, 10.0), (70.0, 10.0), (80.0, 20.0), (100.0, 10.0), (110.0, 10.0)]
        );
        assert_eq!(units(&m, "word"), vec![vec![0, 1, 2, 3, 4, 5], vec![7, 8, 9]]);
        // Lowercase: "İ" draws as "i̇".
        let dump = json!({ "engine": "vector", "totalHeightPx": 24.0, "paragraphs": [{ "lines": [
            { "lineTopY": 0.0, "lineHeightPx": 24.0, "baselineY": 19.0, "ascent": 18.0, "descent": 5.0,
              "totalWidth": 30.0, "segments": [seg("i\u{307}x", 0.0, 10.0)] }]}]});
        let dump = with_sources(dump, &[&[0,0,1]]);
        let m = metrics_from_layout("İx", &dump);
        assert_eq!(m.chars.iter().map(|c| c.map(|c| c.x)).collect::<Vec<_>>(), vec![Some(0.0), Some(20.0)]);
    }

    #[test]
    fn units_group_like_text_units() {
        let m = metrics_from_layout("ab cd", &two_lines());
        assert_eq!(units(&m, "char"), vec![vec![0], vec![1], vec![3], vec![4]]);
        assert_eq!(units(&m, "word"), vec![vec![0, 1], vec![3, 4]]);
        assert_eq!(units(&m, "line"), vec![vec![0, 1], vec![3, 4]]);
        let j = metrics_json(&m, "word");
        assert_eq!(j["units"][1]["text"], "cd");
        assert_eq!(j["units"][1]["baseline"], 43.0);
        assert_eq!(j["units"][1]["advance"], 20.0);
        assert_eq!(j["units"][1]["top"], 25.0);
        assert_eq!(j["units"][1]["bottom"], 48.0);
        assert_eq!(j["layer"]["firstBaseline"], 19.0);
        assert_eq!(j["layer"]["lastBaseline"], 43.0);
        assert_eq!(j["layer"]["lineCount"], 2);
    }

    fn inter(text: &str, width: f64) -> RichTextNodeSpec {
        RichTextNodeSpec {
            node_id: "t".into(),
            html_content: format!("<p style=\"font-family: Inter; font-size: 40px; line-height: 50px; color: #000000\">{text}</p>"),
            width: Some(width),
            height: Some(300.0),
            paragraph_spacing: 0.0,
            line_height_policy: Default::default(),
            vertical_align: None,
            truncation: None,
            auto_width: false,
            area: None,
        }
    }

    #[test]
    fn a_laid_out_text_measures_its_lines_and_units() {
        let m = rich_text_metrics(&inter("Hello world again", 150.0)).expect("Inter lays out");
        assert_eq!(m.lines.len(), 3, "{:?}", m.lines);
        assert!((m.lines[1].baseline - m.lines[0].baseline - 50.0).abs() < 1e-6, "50 px lines");
        let first = &m.lines[0];
        assert!(first.ascent > 30.0 && first.ascent < 45.0 && first.descent > 5.0 && first.descent < 15.0, "{first:?}");
        // Every placed code point sits on its line's baseline, and a line's letters advance to its width.
        for c in m.chars.iter().flatten() {
            assert_eq!(c.baseline, m.lines[c.line].baseline);
            assert_eq!(c.font_size, 40.0);
        }
        let hello: f64 = m.chars[0..5].iter().map(|c| c.unwrap().advance).sum();
        assert!((hello - first.width).abs() < 0.01, "\"Hello\" spans its line: {hello} vs {}", first.width);
        for w in m.chars[0..5].windows(2) {
            let (a, b) = (w[0].unwrap(), w[1].unwrap());
            assert!((a.x + a.advance - b.x).abs() < 1e-6, "pen to pen");
        }
        let j = metrics_json(&m, "word");
        assert_eq!(j["units"].as_array().unwrap().len(), 3);
        assert_eq!(j["units"][2]["text"], "again");
        assert_eq!(j["units"][2]["line"], 3);
        assert_eq!(metrics_json(&m, "line")["units"].as_array().unwrap().len(), 3);
        assert_eq!(metrics_json(&m, "char")["units"].as_array().unwrap().len(), 15);
    }

    #[test]
    fn reflowing_to_a_wider_box_brings_the_words_onto_one_line() {
        let narrow = rich_text_metrics(&inter("Hello world", 150.0)).unwrap();
        let wide = rich_text_metrics(&inter("Hello world", 600.0)).unwrap();
        assert_eq!((narrow.lines.len(), wide.lines.len()), (2, 1));
        let d = reflow_deltas(&narrow, &wide, 1.0);
        assert_eq!(d[0], Some((0.0, 0.0)), "the first letter stays");
        let (dx, dy) = d[6].unwrap();
        assert!(dx > 50.0 && (dy + 50.0).abs() < 1e-6, "\"world\" moves right and up a line: {dx}, {dy}");
        let (hx, hy) = reflow_deltas(&narrow, &wide, 0.5)[6].unwrap();
        assert!((hx - dx / 2.0).abs() < 1e-9 && (hy - dy / 2.0).abs() < 1e-9, "half of the way at 0.5");
    }

    #[test]
    fn reflow_moves_each_code_point_part_of_the_way() {
        // One line "ab cd" (a 10 px space) against the two-line layout.
        let one = json!({ "engine": "vector", "totalHeightPx": 24.0, "paragraphs": [{ "lines": [
            { "lineTopY": 0.0, "lineHeightPx": 24.0, "baselineY": 19.0, "ascent": 18.0, "descent": 5.0,
              "totalWidth": 50.0, "segments": [seg("ab", 0.0, 10.0), space(20.0, 10.0), seg("cd", 30.0, 10.0)] }]}]});
        let one = with_sources(one, &[&[0,1], &[2], &[3,4]]);
        let a = metrics_from_layout("ab cd", &one);
        let b = metrics_from_layout("ab cd", &two_lines());
        let d = reflow_deltas(&a, &b, 0.5);
        assert_eq!(d[0], Some((0.0, 0.0)));
        assert_eq!(d[2], None, "the wrapped space isn't laid out in the second layout");
        assert_eq!(d[3], Some((-15.0, 12.0)));
        assert_eq!(reflow_deltas(&a, &b, 1.0)[4], Some((-30.0, 24.0)));
        let mut x = Vec::new();
        fold_offsets(&mut x, &d);
        assert_eq!(x.len(), 5);
        assert_eq!(x[3].off, [-15.0, 12.0]);
        assert!(x[0].is_identity());
    }
}

/// Intrinsic wrapped cell height, independent of the row's current height/alignment.
pub fn table_text_height(node: &RawNode) -> Option<f64> {
    if moda_text::plain_text::parse_plain_text(node.str("htmlContent").unwrap_or("")).trim().is_empty() {
        return Some(0.0);
    }
    let spec = RichTextNodeSpec {
        node_id: node.id().to_string(),
        html_content: node.str("htmlContent").unwrap_or("").to_string(),
        width: node.f64("width"), height: Some(0.0),
        paragraph_spacing: node.f64("paragraphSpacing").unwrap_or(0.0),
        line_height_policy: Default::default(),
        vertical_align: Some("top".into()), truncation: None, auto_width: false, area: None,
    };
    let dump = crate::text_draw::layout_rich_text_json(&spec).ok()?;
    if !matches!(dump.get("engine")?.as_str()?, "vector" | "degraded") { return None; }
    dump.get("totalHeightPx")?.as_f64()
}
