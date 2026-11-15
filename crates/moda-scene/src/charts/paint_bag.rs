//! Paint on a chart definition's prop bag (E3.1 paint stacks).
//!
//! The generators build a node's props a piece at a time (a parsed fill, then a
//! stroke colour, a width, a dash from the series style, …), and a later piece
//! may clear a helper default by setting it to `null` (the TS "explicitly
//! undefined"). The bag therefore holds the paint in its stack shape while it is
//! being built: `fills` (one layer, or `null` = explicitly none) and a PARTIAL
//! `strokes[0]` (`paint` / `width` / `dash` / `cap` / `join`, each possibly
//! `null`). [`merge`] composes bags (and a helper's defaults) field by field;
//! [`finish`] turns the result into a stored stack: the stroke is kept only with
//! a paint and a positive width, `null` fields are dropped.

use crate::value::{json, Map, Value};

use crate::paint::{Fill, Paint, SolidPaint};

type Bag = Map;

const FILLS: &str = "fills";
const STROKES: &str = "strokes";

/// One fill layer from a paint.
fn fill_layer(paint: Paint) -> Value {
    json!([Fill {
        id: "fill-0".into(),
        visible: None,
        opacity: None,
        blend_mode: None,
        overprint: None,
        paint,
        effects: None,
    }])
}

/// Set the fill to `paint` (`None` = explicitly no fill).
pub fn set_fill_paint(bag: &mut Bag, paint: Option<Paint>) {
    bag.insert(FILLS.into(), paint.map_or(Value::Null, fill_layer));
}

/// Set the fill to a solid colour (`null` = explicitly no fill).
pub fn set_fill_color(bag: &mut Bag, color: Value) {
    let paint = (!color.is_null()).then(|| Paint::Solid(SolidPaint { color }));
    set_fill_paint(bag, paint);
}

fn partial_stroke(bag: &mut Bag) -> &mut Map {
    let strokes = bag
        .entry(STROKES)
        .or_insert_with(|| json!([{"id": "stroke-0"}]));
    if !strokes.as_array().is_some_and(|a| a.first().is_some_and(Value::is_object)) {
        *strokes = json!([{"id": "stroke-0"}]);
    }
    strokes.as_array_mut().unwrap()[0].as_object_mut().unwrap()
}

/// Set one field of the stroke being built (`null` clears a helper default).
fn set_stroke_field(bag: &mut Bag, field: &str, v: Value) {
    partial_stroke(bag).insert(field.into(), v);
}

/// The stroke colour (a solid paint; `null` = no stroke colour).
pub fn set_stroke_color(bag: &mut Bag, color: Value) {
    let paint = if color.is_null() {
        Value::Null
    } else {
        json!({"type": "solid", "color": color})
    };
    set_stroke_field(bag, "paint", paint);
}

pub fn set_stroke_width(bag: &mut Bag, v: Value) {
    set_stroke_field(bag, "width", v);
}

pub fn set_stroke_dash(bag: &mut Bag, v: Value) {
    set_stroke_field(bag, "dash", v);
}

pub fn set_stroke_cap(bag: &mut Bag, v: Value) {
    set_stroke_field(bag, "cap", v);
}

pub fn set_stroke_join(bag: &mut Bag, v: Value) {
    set_stroke_field(bag, "join", v);
}

/// The stroke colour being built (`None` without one).
pub fn stroke_color(bag: &Bag) -> Option<Value> {
    partial(bag)
        .and_then(|s| s.get("paint"))
        .and_then(|p| p.get("color"))
        .cloned()
}

/// The stroke width being built.
pub fn stroke_width(bag: &Bag) -> Option<&Value> {
    partial(bag).and_then(|s| s.get("width"))
}

fn partial(bag: &Bag) -> Option<&Map> {
    bag.get(STROKES)?.as_array()?.first()?.as_object()
}

/// Whether `key` is one of the paint keys this module owns.
pub fn is_paint_key(key: &str) -> bool {
    key == FILLS || key == STROKES
}

/// Merge `v` for `key` into `dst` (a node or a bag): the stroke merges field by
/// field into the stroke being built; everything else replaces.
pub fn merge_key(dst: &mut Bag, key: &str, v: &Value) {
    if key == STROKES {
        if let Some(src) = v.as_array().and_then(|a| a.first()).and_then(Value::as_object) {
            let target = partial_stroke(dst);
            for (k, fv) in src {
                target.insert(k.clone(), fv.clone());
            }
            return;
        }
    }
    dst.insert(key.into(), v.clone());
}

/// Turn the built paint into a stored stack: a `null` fill goes, the stroke is
/// kept only with a paint and a positive width (its `null` fields dropped).
pub fn finish(node: &mut Bag) {
    if node.get(FILLS).is_some_and(Value::is_null) {
        node.remove(FILLS);
    }
    let Some(stroke) = partial(node).cloned() else {
        node.remove(STROKES);
        return;
    };
    let mut stroke: Map = stroke.into_iter().filter(|(_, v)| !v.is_null()).collect();
    if let Some(Value::Array(dash)) = stroke.get("dash") {
        let nums: Vec<Value> = dash.iter().filter(|v| v.is_number()).cloned().collect();
        stroke.insert("dash".into(), Value::Array(nums));
    }
    let width = stroke.get("width").and_then(Value::as_f64).unwrap_or(0.0);
    if stroke.contains_key("paint") && width > 0.0 {
        node.insert(STROKES.into(), json!([stroke]));
    } else {
        node.remove(STROKES);
    }
}
