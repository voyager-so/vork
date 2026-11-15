//! A chart's parts as the motion graph addresses them (docs/design/motion-model.md §1.3 "Chart parts").
//!
//! A `<chart>` draws transient children, so its bars are not layers. The graph host (moda-motion `host.rs`) turns
//! the parts worth animating into derived layers with keyed ids `<chartId>/<key>`, the id scheme of generated
//! nodes (docs/rfc/generated-content-and-hand-edits.md §2, §11: "a chart's bar is `bar:Q3`"), so a binding keeps
//! pointing at the same bar when the data changes order and, once charts are nodes-mode generators, at the stored
//! derived node with the same id.
//!
//! | Role (`metadata.chartRole`) | Key | Name (the part's own, `metadata.name`) |
//! |---|---|---|
//! | `bar` | `bar:<category>` (`bar:<category>:<series>` with a series dimension) | `Sales Bar Q1` |
//! | `value-label` | `value:<category>[:<series>]` | `Sales Value Q1` |
//! | `slice` | `slice:<category>` | `Sales Slice Apples` |
//! | `slice-label` | `label:<category>` | `Sales Label Apples` |
//! | `point-marker` | `point:<category>[:<series>]` | `Sales Point Q1` |
//! | `line-series` | `line[:<series>]` | `Sales Line Revenue` |
//! | `area-series` | `area[:<series>]` | `Sales Area Revenue` |
//!
//! A key that repeats (a combo chart drawing a category twice) names only its first part.
//!
//! A frame's values reach the renderer on the chart node itself: `_motionParts` (`{ key: { field: value } }`, the
//! part's changed top-level fields; the render feed only, never saved), which [`apply_motion_parts`] writes onto the
//! expanded children. Every host already hands the renderer the overlaid chart node, so nothing else changes.

use super::ticks::TickError;
use std::collections::HashSet;

use crate::value::{Map, Value};

use super::cfg::JsonMap;
use super::defs::{def_to_node, BarHint};
use crate::types::RawNode;

/// The chart node's field that carries a frame's part values.
pub const MOTION_PARTS_KEY: &str = "_motionParts";

/// One addressable part of a chart.
#[derive(Clone, Debug)]
pub struct ChartPart {
    /// `bar:Q1`, `slice:Apples`, … (see the module docs).
    pub key: String,
    /// The part's `metadata.chartRole`.
    pub role: String,
    /// The part's own name (`Sales Bar Q1`).
    pub name: String,
    /// The drawn node in the chart's LOCAL coordinates, as the expansion draws it (its transient id).
    pub node: JsonMap,
    /// For a bar: the side it grows from and its value.
    pub bar: Option<BarHint>,
}

/// The key of a drawn chart child, from its metadata; `None` for chrome (axes, grid, legend, titles).
pub fn part_key(node: &JsonMap) -> Option<(String, String)> {
    let meta = node.get("metadata")?.as_object()?;
    let role = meta.get("chartRole")?.as_str()?;
    let text = |k: &str| {
        meta.get(k).and_then(|v| match v {
            Value::String(s) => Some(s.clone()),
            Value::Number(n) => Some(n.to_string()),
            _ => None,
        })
    };
    let prefix = match role {
        "bar" => "bar",
        "value-label" => "value",
        "slice" => "slice",
        "slice-label" => "label",
        "point-marker" => "point",
        "line-series" => "line",
        "area-series" => "area",
        _ => return None,
    };
    let mut key = prefix.to_string();
    if !matches!(role, "line-series" | "area-series") {
        key.push(':');
        key.push_str(&text("chartCategory")?);
    }
    if let Some(series) = text("chartSeries") {
        key.push(':');
        key.push_str(&series);
    }
    Some((key, role.to_string()))
}

/// Each child's key, or `None` for chrome and for a repeated key (only its first part is addressable).
fn keys_of<'a>(children: impl Iterator<Item = &'a JsonMap>) -> Vec<Option<(String, String)>> {
    let mut seen = HashSet::new();
    children
        .map(|n| part_key(n).filter(|(k, _)| seen.insert(k.clone())))
        .collect()
}

/// The chart's addressable parts, in draw order (bars in category order, then series), with the geometry the
/// expansion draws (LOCAL coordinates). Empty for a host that isn't a chart or draws the placeholder.
pub fn chart_parts(node: &RawNode) -> Result<Vec<ChartPart>, TickError> {
    let Some(defs) = super::chart_definitions(node)? else {
        return Ok(Vec::new());
    };
    let host_opacity = node.opacity();
    let drawn: Vec<(JsonMap, Option<BarHint>)> = defs
        .iter()
        .enumerate()
        .filter_map(|(i, d)| def_to_node(d, node.id(), i, host_opacity).map(|n| (n, d.bar)))
        .collect();
    let keys = keys_of(drawn.iter().map(|(n, _)| n));
    Ok(drawn
        .into_iter()
        .zip(keys)
        .filter_map(|((n, bar), key)| {
            let (key, role) = key?;
            let name = n
                .get("metadata")
                .and_then(|m| m.get("name"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| key.clone());
            let bar = if role == "bar" { bar } else { None };
            Some(ChartPart {
                key,
                role,
                name,
                node: n,
                bar,
            })
        })
        .collect())
}

/// Write a frame's part values (`_motionParts`: key → changed fields) onto the expanded children.
pub fn apply_motion_parts(children: &mut [RawNode], parts: &Map) {
    if parts.is_empty() {
        return;
    }
    let maps: Vec<JsonMap> = children.iter().map(RawNode::to_json_map).collect();
    let keys = keys_of(maps.iter());
    for (child, key) in children.iter_mut().zip(keys) {
        let Some((key, _)) = key else { continue };
        let Some(fields) = parts.get(&key).and_then(Value::as_object) else {
            continue;
        };
        for (k, v) in fields {
            if matches!(k.as_str(), "id" | "type" | "children") {
                continue;
            }
            if v.is_null() {
                child.remove(k);
            } else {
                child.set(k, v.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::charts::defs::BarBase;
    use crate::value::json;

    /// The parity corpus's bar chart config (every field a generator reads), with `spec_extra` merged in.
    fn chart(spec_extra: Value, data: Value) -> RawNode {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/chart-bar.json"
        );
        let doc: Value = crate::value::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let nodes = doc.get("canvasData").unwrap_or(&doc)["nodes"]
            .as_array()
            .unwrap()
            .clone();
        let host = nodes.iter().find(|n| n["type"] == "generated").unwrap();
        let mut spec = host["generatedSource"]["spec"].clone();
        spec["config"]["showLegend"] = json!(false);
        for (k, v) in spec_extra.as_object().unwrap() {
            if k == "config" {
                for (ck, cv) in v.as_object().unwrap() {
                    spec["config"][ck] = cv.clone();
                }
            } else {
                spec[k] = v.clone();
            }
        }
        RawNode::from_value(json!({
            "id": "ch", "type": "generated", "generatedKind": "chart", "renderMode": "native-children",
            "x": 0, "y": 0, "width": 880, "height": 520, "rotation": 0, "scale": 1,
            "metadata": { "name": "Sales" },
            "generatedSource": { "kind": "chart", "chartVersion": 1, "spec": spec, "data": data }
        }))
        .unwrap()
    }

    fn rows(values: &[(&str, f64)]) -> Value {
        let lo = values.iter().map(|v| v.1).fold(0.0, f64::min);
        let hi = values.iter().map(|v| v.1).fold(0.0, f64::max);
        json!({
            "columns": ["category", "value"],
            "rows": values.iter().map(|(c, v)| json!({ "category": c, "value": v })).collect::<Vec<_>>(),
            "series": [], "categories": values.iter().map(|v| v.0).collect::<Vec<_>>(), "valueRange": [lo, hi]
        })
    }

    #[test]
    fn bars_are_keyed_in_category_order_and_grow_from_the_baseline() {
        let parts = chart_parts(&chart(
            json!({}),
            rows(&[("Q1", 10.0), ("Q2", -5.0), ("Q3", 20.0)]),
        )).unwrap();
        let bars: Vec<&ChartPart> = parts.iter().filter(|p| p.role == "bar").collect();
        assert_eq!(
            bars.iter().map(|p| p.key.as_str()).collect::<Vec<_>>(),
            ["bar:Q1", "bar:Q2", "bar:Q3"],
            "{parts:?}"
        );
        assert_eq!(bars[0].name, "Sales Bar Q1");
        assert_eq!(bars[0].bar.map(|b| b.base), Some(BarBase::Bottom));
        assert_eq!(
            bars[1].bar.map(|b| (b.base, b.value)),
            Some((BarBase::Top, -5.0)),
            "a negative bar hangs from the baseline"
        );
        // The positive bar's bottom and the negative bar's top are the same baseline.
        let f = |p: &ChartPart, k: &str| p.node.get(k).and_then(Value::as_f64).unwrap();
        assert!((f(bars[0], "y") + f(bars[0], "height") - f(bars[1], "y")).abs() < 1e-9);
        assert!(
            parts.iter().any(|p| p.key == "value:Q1"),
            "value labels are parts too: {parts:?}"
        );
    }

    #[test]
    fn horizontal_bars_grow_from_the_left_and_negative_ones_from_the_right() {
        let parts = chart_parts(&chart(
            json!({ "orientation": "horizontal" }),
            rows(&[("A", 4.0), ("B", -2.0)]),
        )).unwrap();
        let base = |k: &str| {
            parts
                .iter()
                .find(|p| p.key == k)
                .and_then(|p| p.bar)
                .map(|b| b.base)
        };
        assert_eq!(base("bar:A"), Some(BarBase::Left), "{parts:?}");
        assert_eq!(base("bar:B"), Some(BarBase::Right));
    }

    #[test]
    fn motion_parts_write_onto_the_keyed_children_only() {
        let mut node = chart(json!({}), rows(&[("Q1", 10.0), ("Q2", 20.0)]));
        let plain = crate::charts::expand_chart_generated_node(&node).unwrap();
        node.set(
            MOTION_PARTS_KEY,
            json!({ "bar:Q2": { "height": 7.0, "opacity": 0.5 }, "bar:Nope": { "height": 1.0 } }),
        );
        let moved = crate::charts::expand_chart_generated_node(&node).unwrap();
        assert_eq!(plain.len(), moved.len());
        let mut changed = 0;
        for (a, b) in plain.iter().zip(&moved) {
            assert_eq!(a.id(), b.id(), "ids stay the transient ones");
            if a.fields() != b.fields() {
                changed += 1;
                assert_eq!(b.f64("height"), Some(7.0));
                assert_eq!(b.f64("opacity"), Some(0.5));
                assert_eq!(
                    part_key(&b.to_json_map()).map(|k| k.0).as_deref(),
                    Some("bar:Q2")
                );
            }
        }
        assert_eq!(changed, 1);
    }

    #[test]
    fn pie_slices_and_line_series_are_parts() {
        let pie = chart_parts(&chart(
            json!({ "type": "pie" }),
            rows(&[("Apples", 3.0), ("Pears", 1.0)]),
        )).unwrap();
        assert!(
            pie.iter()
                .any(|p| p.key == "slice:Apples" && p.name == "Sales Slice Apples"),
            "{pie:?}"
        );
        let line = chart_parts(&chart(
            json!({ "type": "line", "config": { "showPoints": true } }),
            rows(&[("Q1", 3.0), ("Q2", 5.0)]),
        )).unwrap();
        assert!(line.iter().any(|p| p.key == "line"), "{line:?}");
        assert!(line.iter().any(|p| p.key == "point:Q2"), "{line:?}");
    }
}
