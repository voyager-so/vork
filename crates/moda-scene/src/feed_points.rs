//! Path points that come beside a scene feed instead of inside its JSON.
//!
//! The editor holds a path's points packed (packages/editor/src/core/scene/pathPoints.ts): per point a shape id,
//! and every coordinate in one float array. A 100k-layer vector document's points are more text than one JavaScript
//! string can hold, so the feed carries a packed path's points as a placeholder, `"points": {"$pts": <id>}`, and the
//! host hands them over when a node is hydrated ([`set_feed_points_resolver`]). [`resolve_feed_points`] turns a
//! placeholder back into exactly the JSON array the text would have parsed to: the same keys in the same order, a
//! whole number as an integer and anything else as a float, a non-finite number as `null`, a key whose value was
//! `undefined` left out (as `JSON.stringify` does).

use crate::path_points::{CompactKind, CompactPoints};
use crate::types::RawNode;
use crate::value::{Map, Number, Value};
use std::cell::{Cell, RefCell};

/// The key of a points placeholder.
pub const PLACEHOLDER_KEY: &str = "$pts";

/// One field of a point shape.
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeField {
    /// A number: one coordinate.
    Num(String),
    /// An `{x, y}` pair: two coordinates.
    Xy(String),
    /// The same value on every point of this shape (`type`, a `handleStyle`, …).
    Const(String, Value),
    /// A key whose value was `undefined`: absent from the JSON.
    Skip(String),
}

/// A point shape: its fields in key order.
pub type Shape = Vec<ShapeField>;

/// The shapes from the host's table: `[[{"key", "kind": "num" | "xy" | "const" | "skip", "value"?}, …], …]`.
pub fn shapes_from_json(json: &str) -> Result<Vec<Shape>, String> {
    let v: Value = crate::value::from_str(json).map_err(|e| e.to_string())?;
    let Value::Array(shapes) = v else { return Err("point shapes: not an array".into()) };
    shapes
        .iter()
        .map(|s| {
            let Value::Array(fields) = s else { return Err("point shape: not an array".to_string()) };
            fields
                .iter()
                .map(|f| {
                    let key = f.get("key").and_then(Value::as_str).ok_or("point shape field: no key")?.to_string();
                    Ok(match f.get("kind").and_then(Value::as_str) {
                        Some("num") => ShapeField::Num(key),
                        Some("xy") => ShapeField::Xy(key),
                        Some("const") => ShapeField::Const(key, f.get("value").cloned().unwrap_or(Value::Null)),
                        Some("skip") => ShapeField::Skip(key),
                        other => return Err(format!("point shape field: kind {other:?}")),
                    })
                })
                .collect()
        })
        .collect()
}

/// A JavaScript number as `JSON.stringify` then `serde_json` would give it back.
pub fn js_number(v: f64) -> Value {
    if !v.is_finite() {
        return Value::Null;
    }
    // JSON.stringify writes a whole number below 1e21 without a fraction or exponent, which parses as an integer
    // when it fits one (-0 prints as 0).
    if v.fract() == 0.0 && v.abs() < 1e21 {
        if v >= 0.0 && v < 18446744073709551616.0 {
            return Value::Number(Number::from(v as u64));
        }
        if v < 0.0 && v >= -9223372036854775808.0 {
            return Value::Number(Number::from(v as i64));
        }
    }
    Number::from_f64(v).map_or(Value::Null, Value::Number)
}

/// Points from their packed form: per point its shape id, and the coordinates in order. `None` when the ids or
/// coordinates don't fit the shapes.
pub fn points_from_packed(shapes: &[Shape], ids: &[u8], coords: &[f64]) -> Option<Vec<Value>> {
    let mut at = 0;
    let mut next = || {
        let v = *coords.get(at)?;
        at += 1;
        Some(v)
    };
    let mut out = Vec::with_capacity(ids.len());
    for &id in ids {
        let shape = shapes.get(id as usize)?;
        let mut p = Map::with_capacity(shape.len());
        for f in shape {
            match f {
                ShapeField::Num(k) => {
                    p.insert(k.clone().into(), js_number(next()?));
                }
                ShapeField::Xy(k) => {
                    let (x, y) = (next()?, next()?);
                    let mut xy = Map::with_capacity(2);
                    xy.insert("x".into(), js_number(x));
                    xy.insert("y".into(), js_number(y));
                    p.insert(k.clone().into(), Value::Object(xy));
                }
                ShapeField::Const(k, v) => {
                    p.insert(k.clone().into(), v.clone());
                }
                ShapeField::Skip(_) => {}
            }
        }
        out.push(Value::Object(p));
    }
    (at == coords.len()).then_some(out)
}

/// A path's points as the host holds them: per point a shape id, the coordinates in order, and the shape table.
pub struct Packed {
    pub ids: Vec<u8>,
    pub coords: Vec<f64>,
    pub shapes: std::rc::Rc<Vec<Shape>>,
}

/// Where a placeholder's points come from: the host, by id.
pub type Resolver = Box<dyn Fn(u64) -> Option<Packed>>;

/// The compact kind of a point shape, when its fields are exactly a canonical point's (`type` first, as a constant,
/// then `x`, `y` and the kind's control points as pairs, in the order [`CompactKind::keys`] gives).
pub fn canonical_kind(shape: &Shape) -> Option<CompactKind> {
    let (ShapeField::Const(k, Value::String(name)), rest) = shape.split_first()? else { return None };
    if k != "type" {
        return None;
    }
    let kind = CompactKind::from_name(name)?;
    let keys = &kind.keys()[1..];
    if rest.len() != keys.len() {
        return None;
    }
    let fits = rest.iter().zip(keys).enumerate().all(|(i, (f, key))| match f {
        ShapeField::Num(k) => i < 2 && k == key,
        ShapeField::Xy(k) => i >= 2 && k == key,
        _ => false,
    });
    fits.then_some(kind)
}

/// Compact points straight from packed ones, when every shape they use is canonical: what hydrating their JSON
/// and compacting it would give.
pub fn compact_from_packed(p: &Packed) -> Option<CompactPoints> {
    let kinds_by_shape: Vec<Option<CompactKind>> = p.shapes.iter().map(canonical_kind).collect();
    let kinds: Option<Vec<CompactKind>> = p.ids.iter().map(|&id| kinds_by_shape.get(id as usize).copied().flatten()).collect();
    CompactPoints::from_js_coords(&kinds?, &p.coords)
}

thread_local! {
    static RESOLVER: RefCell<Option<Resolver>> = const { RefCell::new(None) };
    static MISSES: Cell<u64> = const { Cell::new(0) };
}

/// Install (or remove) the host's resolver for this thread.
pub fn set_feed_points_resolver(resolver: Option<Resolver>) {
    RESOLVER.with(|r| *r.borrow_mut() = resolver);
}

/// Placeholders no points could be found for, on this thread, since it started (their node gets no points).
pub fn feed_points_misses() -> u64 {
    MISSES.with(Cell::get)
}

/// The id of a points placeholder.
pub fn placeholder_id(points: &Value) -> Option<u64> {
    let Value::Object(m) = points else { return None };
    if m.len() != 1 {
        return None;
    }
    m.get(PLACEHOLDER_KEY)?.as_u64()
}

/// Replace a node's points placeholder with the points it names (a no-op for any other node). A placeholder with
/// no points to be found becomes an empty list, and counts as a miss ([`feed_points_misses`]).
pub fn resolve_feed_points(node: &mut Map) {
    let _ = take_feed_points_as(node, false);
}

/// Resolve a placeholder on an installed node without expanding unchanged compact geometry.
pub(crate) fn resolve_node_feed_points(node: &mut RawNode) {
    if node.fields().get("points").and_then(placeholder_id).is_some() {
        resolve_feed_points(node.fields_mut());
    }
}

/// [`resolve_feed_points`] for a node about to be hydrated: points of canonical shapes come back compact (the
/// node's `points` left `null`, for [`crate::types::RawNode::with_compact_points`]) when nodes are compacted on
/// load; anything else is resolved to JSON in place and gives `None`.
pub fn take_feed_points(node: &mut Map) -> Option<CompactPoints> {
    let compact = crate::path_points::compact_on_load() && node.get("type").and_then(Value::as_str) == Some("path");
    take_feed_points_as(node, compact)
}

fn take_feed_points_as(node: &mut Map, compact: bool) -> Option<CompactPoints> {
    let slot = node.get_mut("points")?;
    let id = placeholder_id(slot)?;
    let packed = RESOLVER.with(|r| r.borrow().as_ref().and_then(|resolve| resolve(id)));
    let Some(packed) = packed else {
        MISSES.with(|m| m.set(m.get() + 1));
        *slot = Value::Array(Vec::new());
        return None;
    };
    if compact {
        if let Some(c) = compact_from_packed(&packed) {
            *slot = Value::Null;
            return Some(c);
        }
    }
    *slot = Value::Array(points_from_packed(&packed.shapes, &packed.ids, &packed.coords).unwrap_or_else(|| {
        MISSES.with(|m| m.set(m.get() + 1));
        Vec::new()
    }));
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    fn shapes() -> Vec<Shape> {
        shapes_from_json(
            r#"[
              [{"key":"type","kind":"const","value":"move"},{"key":"x","kind":"num"},{"key":"y","kind":"num"}],
              [{"key":"x","kind":"num"},{"key":"y","kind":"num"},{"key":"type","kind":"const","value":"curve"},
               {"key":"cp1","kind":"xy"},{"key":"cp2","kind":"xy"},{"key":"handleStyle","kind":"skip"},
               {"key":"smooth","kind":"const","value":true}]
            ]"#,
        )
        .unwrap()
    }

    #[test]
    fn packed_points_come_back_as_the_json_text_would_parse() {
        let pts = points_from_packed(
            &shapes(),
            &[0, 1],
            &[0.0, -0.0, 1.5, 1e21, 3.0, 4.25, f64::NAN, -7.0],
        )
        .unwrap();
        // What JSON.stringify of the same objects gives, parsed.
        let text = r#"[{"type":"move","x":0,"y":0},{"x":1.5,"y":1e21,"type":"curve","cp1":{"x":3,"y":4.25},"cp2":{"x":null,"y":-7},"smooth":true}]"#;
        let want: Value = crate::value::from_str(text).unwrap();
        assert_eq!(Value::Array(pts.clone()), want);
        assert_eq!(crate::value::to_string(&Value::Array(pts)).unwrap(), crate::value::to_string(&want).unwrap());
        assert_eq!(js_number(9007199254740992.0), json!(9007199254740992u64));
        assert_eq!(js_number(-3.0), json!(-3));
        assert!(js_number(-3.0).is_i64());
        assert!(js_number(0.5).is_f64());
    }

    #[test]
    fn ids_or_coordinates_that_do_not_fit_are_refused() {
        assert!(points_from_packed(&shapes(), &[2], &[]).is_none());
        assert!(points_from_packed(&shapes(), &[0], &[1.0]).is_none());
        assert!(points_from_packed(&shapes(), &[0], &[1.0, 2.0, 3.0]).is_none());
        assert_eq!(points_from_packed(&shapes(), &[], &[]), Some(vec![]));
    }

    /// The canonical shapes the editor packs, and a non-canonical one.
    fn canonical_shapes() -> Vec<Shape> {
        shapes_from_json(
            r#"[
              [{"key":"type","kind":"const","value":"move"},{"key":"x","kind":"num"},{"key":"y","kind":"num"}],
              [{"key":"type","kind":"const","value":"line"},{"key":"x","kind":"num"},{"key":"y","kind":"num"}],
              [{"key":"type","kind":"const","value":"curve"},{"key":"x","kind":"num"},{"key":"y","kind":"num"},
               {"key":"cp1","kind":"xy"},{"key":"cp2","kind":"xy"}],
              [{"key":"type","kind":"const","value":"quadratic"},{"key":"x","kind":"num"},{"key":"y","kind":"num"},
               {"key":"cp","kind":"xy"}],
              [{"key":"type","kind":"const","value":"line"},{"key":"x","kind":"num"},{"key":"y","kind":"num"},
               {"key":"handleStyle","kind":"const","value":"smooth"}]
            ]"#,
        )
        .unwrap()
    }

    fn packed(shapes: &[Shape], ids: &[u8], coords: &[f64]) -> Packed {
        Packed { ids: ids.to_vec(), coords: coords.to_vec(), shapes: std::rc::Rc::new(shapes.to_vec()) }
    }

    #[test]
    fn canonical_packed_points_compact_as_their_json_would() {
        let shapes = canonical_shapes();
        assert_eq!(
            shapes.iter().map(canonical_kind).collect::<Vec<_>>(),
            vec![Some(CompactKind::Move), Some(CompactKind::Line), Some(CompactKind::Curve), Some(CompactKind::Quadratic), None]
        );
        assert_eq!(canonical_kind(&super::tests::shapes()[1]), None, "keys out of order");
        let values = [
            0.0, -0.0, 1.5, -1.5, 3.0, -3.0, 1e-9, 123456.789, 9007199254740991.0, -9007199254740991.0,
            9007199254740992.0, -9007199254740992.0, 1e20, 1e21, 1e300, -1e300, 18446744073709551616.0, 5e-324,
        ];
        let mut cases: Vec<(Vec<u8>, Vec<f64>)> = Vec::new();
        for (i, &v) in values.iter().enumerate() {
            let w = values[(i * 7 + 3) % values.len()];
            cases.push((vec![0, 2, 3, 1], vec![v, w, 1.0, 2.0, w, v, 3.0, 4.5, v, 6.0, 7.0, 8.0, -v, w, 9.0, 10.0]));
        }
        cases.push((vec![0], vec![f64::NAN, 1.0]));
        cases.push((vec![0], vec![f64::INFINITY, 1.0]));
        cases.push((vec![0, 4], vec![1.0, 2.0, 3.0, 4.0]));
        cases.push((vec![9], vec![1.0, 2.0]));
        cases.push((vec![0], vec![1.0]));
        for (ids, coords) in cases {
            let via_json = points_from_packed(&shapes, &ids, &coords).and_then(|p| CompactPoints::from_json(&p));
            let direct = compact_from_packed(&packed(&shapes, &ids, &coords));
            assert_eq!(direct, via_json, "{ids:?} {coords:?}");
            if let Some(c) = &direct {
                assert_eq!(Value::Array(c.to_json()), Value::Array(points_from_packed(&shapes, &ids, &coords).unwrap()));
            }
        }
    }

    #[test]
    fn placeholders_resolve_through_the_host_and_misses_are_counted() {
        let shapes = canonical_shapes();
        set_feed_points_resolver(Some(Box::new(move |id| (id == 7).then(|| packed(&shapes, &[0], &[1.0, 2.0])))));
        let mut node = json!({"id":"p","type":"path","points":{"$pts":7},"x":1}).as_object().unwrap().clone();
        resolve_feed_points(&mut node);
        assert_eq!(Value::Object(node), json!({"id":"p","type":"path","points":[{"type":"move","x":1,"y":2}],"x":1}));
        let mut node = json!({"id":"p","type":"path","points":{"$pts":7},"x":1}).as_object().unwrap().clone();
        let c = take_feed_points(&mut node).expect("canonical: compact");
        assert_eq!(c.to_json(), vec![json!({"type":"move","x":1,"y":2})]);
        assert_eq!(node.get("points"), Some(&Value::Null));
        let mut installed = RawNode::from_value(json!({"id":"p","type":"path","points":{"$pts":7}})).unwrap();
        resolve_node_feed_points(&mut installed);
        assert_eq!(installed.to_json_map()["points"], json!([{"type":"move","x":1,"y":2}]));
        let before = feed_points_misses();
        let mut missing = json!({"type":"path","points":{"$pts":8}}).as_object().unwrap().clone();
        assert!(take_feed_points(&mut missing).is_none());
        assert_eq!(missing.get("points"), Some(&json!([])));
        assert_eq!(feed_points_misses(), before + 1);
        let mut plain = json!({"points":{"$pts":7,"x":1}}).as_object().unwrap().clone();
        resolve_feed_points(&mut plain);
        assert_eq!(plain.get("points"), Some(&json!({"$pts":7,"x":1})), "not a placeholder");
        set_feed_points_resolver(None);
    }

    #[test]
    fn a_feed_with_placeholders_hydrates_to_the_scene_its_text_would() {
        for compact in [true, false] {
            crate::path_points::set_compact_on_load(compact);
            let shapes = canonical_shapes();
            let odd = super::tests::shapes();
            let cases: Vec<(Vec<Shape>, Vec<u8>, Vec<f64>)> = vec![
                (shapes.clone(), vec![0, 2, 1], vec![1.0, 2.0, 3.5, -4.0, 0.0, 1.0, 2.25, 3.0, -0.0, 7.0]),
                (shapes.clone(), vec![0, 4], vec![7.0, 1e-9, 1.0, 2.0]),
                (shapes.clone(), vec![0, 1], vec![f64::NAN, 1.0, 2.0, 3.0]),
                (odd.clone(), vec![0, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            ];
            let text_points: Vec<Value> =
                cases.iter().map(|(sh, i, c)| Value::Array(points_from_packed(sh, i, c).unwrap())).collect();
            let doc = |points: &[Value]| {
                let nodes: Vec<Value> = points
                    .iter()
                    .enumerate()
                    .map(|(k, p)| json!({"id": format!("n{k}"), "type":"path","x":k,"y":0,"points":p}))
                    .collect();
                let ids: Vec<String> = (0..points.len()).map(|k| format!("n{k}")).collect();
                json!({"nodes": nodes, "pages":[{"id":"pg","width":10,"height":10,"nodeIds":ids}]}).to_string()
            };
            let text = doc(&text_points);
            let placeholders: Vec<Value> = (1..=cases.len()).map(|k| json!({"$pts": k})).collect();
            let feed = doc(&placeholders);
            set_feed_points_resolver(Some(Box::new(move |id| {
                let (sh, i, c) = cases.get((id as usize).checked_sub(1)?)?;
                Some(packed(sh, i, c))
            })));
            let misses = feed_points_misses();
            let from_feed = crate::scene::hydrate_document_str(&feed).unwrap();
            let from_text = crate::scene::hydrate_document_str(&text).unwrap();
            set_feed_points_resolver(None);
            crate::path_points::set_compact_on_load(true);
            assert_eq!(feed_points_misses(), misses);
            assert_eq!(from_feed.nodes.len(), 4);
            for (a, b) in from_feed.nodes.iter().zip(&from_text.nodes) {
                assert_eq!(a, b);
                assert_eq!(a.to_json_map(), b.to_json_map());
                assert_eq!(a.is_compact(), b.is_compact(), "{}", a.id());
                assert_eq!(a.fields(), b.fields());
            }
            if compact {
                let compacted: Vec<bool> = from_feed.nodes.iter().map(RawNode::is_compact).collect();
                // A handleStyle or odd key order keeps JSON; a NaN point is sanitized away, then the rest compacts.
                assert_eq!(compacted, vec![true, false, true, false]);
            }
        }
    }
}
