//! `ChartNodeDefinition` (the generators' output) and its conversion to
//! transient scene nodes — port of `chartDefinitionToSceneNode` +
//! `applyPassthroughProps` + `finalizeNode` (core/scene/chartExpansion.ts:
//! 212-403) over the shape-helper factories (core/scene/shapeHelpers.ts,
//! core/scene/pathHelpers.ts `createPathNode`).
//!
//! Prop convention: a generator inserting `Value::Null` models the TS
//! "explicitly set to undefined" — passthrough then REMOVES any helper
//! default for that key (chartExpansion.ts:344-360); an absent key leaves
//! the helper default in place.

use crate::value::{json, Map, Value};

use super::cfg::JsonMap;
use super::paint_bag;

use super::jsmath::js_max;
use crate::expansion::transient_proxy_id;

#[derive(Clone, Debug)]
pub struct Def {
    pub ty: &'static str, // rectangle | ellipse | line | richtext | path | polygon | star
    pub props: JsonMap,
    /// A bar's base and value (never drawn; `parts::chart_parts` reads it for the motion graph).
    pub bar: Option<BarHint>,
}

/// Which side of a bar sits on the value axis's baseline: the side it grows from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarBase {
    Bottom,
    Top,
    Left,
    Right,
}

/// What a bar is, beyond what it draws: its base side and its datum's value (a stacked segment's own value).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarHint {
    pub base: BarBase,
    pub value: f64,
}

impl Def {
    pub fn new(ty: &'static str) -> Def {
        Def {
            ty,
            props: Map::new(),
            bar: None,
        }
    }

    /// Mark this def as a bar growing from `base` (see [`BarHint`]).
    pub fn grows_from(&mut self, base: BarBase, value: f64) -> &mut Def {
        self.bar = Some(BarHint { base, value });
        self
    }
    pub fn set(&mut self, key: &str, v: Value) -> &mut Def {
        if paint_bag::is_paint_key(key) {
            paint_bag::merge_key(&mut self.props, key, &v);
        } else {
            self.props.insert(key.into(), v);
        }
        self
    }

    /// A solid fill colour (`null` = explicitly no fill).
    pub fn fill_color(&mut self, color: Value) -> &mut Def {
        paint_bag::set_fill_color(&mut self.props, color);
        self
    }

    /// The stroke colour (`null` clears the helper default).
    pub fn stroke_color(&mut self, color: Value) -> &mut Def {
        paint_bag::set_stroke_color(&mut self.props, color);
        self
    }

    pub fn stroke_width(&mut self, v: Value) -> &mut Def {
        paint_bag::set_stroke_width(&mut self.props, v);
        self
    }

    pub fn stroke_dash(&mut self, v: Value) -> &mut Def {
        paint_bag::set_stroke_dash(&mut self.props, v);
        self
    }

    pub fn stroke_cap(&mut self, v: Value) -> &mut Def {
        paint_bag::set_stroke_cap(&mut self.props, v);
        self
    }

    pub fn stroke_join(&mut self, v: Value) -> &mut Def {
        paint_bag::set_stroke_join(&mut self.props, v);
        self
    }
}

/// Keys handled by the shape-helper call (chartExpansion.ts:331-342).
const HANDLED_BY_HELPER: [&str; 10] = [
    "id", "type", "x", "y", "width", "height", "points", "metadata", "name", "opacity",
];

fn number_prop(props: &JsonMap, key: &str, default: f64) -> f64 {
    match props.get(key) {
        Some(Value::Number(n)) => {
            let v = n.as_f64().unwrap_or(f64::NAN);
            if v.is_finite() {
                v
            } else {
                default
            }
        }
        _ => default,
    }
}

fn string_prop<'a>(props: &'a JsonMap, key: &str) -> Option<&'a str> {
    props.get(key).and_then(Value::as_str)
}

/// `applyHostOpacity` (chartExpansion.ts:397-402).
fn apply_host_opacity(child: Option<f64>, host: f64) -> Option<f64> {
    let child = child.filter(|c| c.is_finite());
    if host == 1.0 {
        return child;
    }
    match child {
        None => Some(host),
        Some(c) => Some(c * host),
    }
}

/// `applyPassthroughProps` (chartExpansion.ts:344-360).
fn apply_passthrough(node: &mut JsonMap, props: &JsonMap, extra_skip: &[&str]) {
    for (k, v) in props {
        if HANDLED_BY_HELPER.contains(&k.as_str()) || extra_skip.contains(&k.as_str()) {
            continue;
        }
        if paint_bag::is_paint_key(k) {
            // Paint merges into the helper's defaults field by field (a `null`
            // field clears that default).
            paint_bag::merge_key(node, k, v);
        } else if v.is_null() {
            // explicit `undefined` — clears any helper default
            node.remove(k);
        } else {
            node.insert(k.clone(), v.clone());
        }
    }
}

/// `finalizeNode` (chartExpansion.ts:362-376).
fn finalize_node(
    node: &mut JsonMap,
    opacity: Option<f64>,
    name: Option<&str>,
    incoming_metadata: Option<&JsonMap>,
) {
    let mut merged: JsonMap = node
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(im) = incoming_metadata {
        for (k, v) in im {
            if v.is_null() {
                // JS spread keeps `undefined` values, but they vanish on the
                // JSON boundary — omit.
                merged.remove(k);
                continue;
            }
            merged.insert(k.clone(), v.clone());
        }
    }
    if let Some(n) = name {
        merged.insert("name".into(), json!(n));
    }
    if merged.is_empty() {
        node.remove("metadata");
    } else {
        node.insert("metadata".into(), Value::Object(merged));
    }
    if let Some(op) = opacity {
        node.insert("opacity".into(), json!(op));
    }
}

/// `chartDefinitionToSceneNode` (chartExpansion.ts:212-319). Returns None
/// for defs the TS would skip (line defs with < 4 points).
pub fn def_to_node(def: &Def, host_id: &str, index: usize, host_opacity: f64) -> Option<JsonMap> {
    let id = transient_proxy_id(host_id, index);
    let props = &def.props;
    let name = string_prop(props, "name");
    let opacity_prop = match props.get("opacity") {
        Some(Value::Number(n)) => n.as_f64(),
        _ => None,
    };
    let opacity = apply_host_opacity(opacity_prop, host_opacity);
    let incoming_metadata = props.get("metadata").and_then(Value::as_object);

    let mut node: JsonMap = Map::new();
    node.insert("id".into(), json!(id));
    match def.ty {
        "rectangle" | "ellipse" | "polygon" | "star" | "richtext" => {
            node.insert("type".into(), json!(def.ty));
            node.insert("x".into(), json!(number_prop(props, "x", 0.0)));
            node.insert("y".into(), json!(number_prop(props, "y", 0.0)));
            node.insert("width".into(), json!(number_prop(props, "width", 1.0)));
            node.insert("height".into(),
                json!(number_prop(props, "height", 1.0)),
            );
            if def.ty == "polygon" {
                node.insert("sides".into(), json!(number_prop(props, "sides", 3.0)));
            }
            if def.ty == "star" {
                node.insert("numPoints".into(),
                    json!(number_prop(props, "numPoints", 5.0)),
                );
                node.insert("innerRadiusRatio".into(),
                    json!(number_prop(props, "innerRadiusRatio", 0.5)),
                );
            }
            if def.ty == "richtext" {
                node.insert("htmlContent".into(),
                    json!(string_prop(props, "htmlContent").unwrap_or("")),
                );
            }
            node.insert("rotation".into(), json!(0.0));
            node.insert("scale".into(), json!(1.0));
            apply_passthrough(&mut node, props, &[]);
        }
        "line" => {
            let pts = props.get("points").and_then(Value::as_array)?;
            if pts.len() < 4 {
                return None;
            }
            let nums: Vec<Value> = pts.iter().take(4).cloned().collect();
            node.insert("type".into(), json!("line"));
            node.insert("points".into(), Value::Array(nums));
            apply_passthrough(&mut node, props, &["points"]);
        }
        "path" => {
            let pts: Vec<Value> = props
                .get("points")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let x = number_prop(props, "x", 0.0);
            let y = number_prop(props, "y", 0.0);
            create_path_node(&mut node, &pts, x, y);
            apply_passthrough(&mut node, props, &["points"]);
        }
        _ => return None,
    }
    paint_bag::finish(&mut node);
    finalize_node(&mut node, opacity, name, incoming_metadata);
    apply_fill_opacity(&mut node);
    Some(node)
}

/// `applyFillOpacity` (chartExpansion.ts): an area's `fillOpacity` (area and
/// combo charts, `fill-opacity`) is its fill layers' opacity, clamped to 0–1;
/// no scene node reads a node-level `fillOpacity`.
fn apply_fill_opacity(node: &mut JsonMap) {
    let Some(fill_opacity) = node.remove("fillOpacity") else {
        return;
    };
    let Some(alpha) = fill_opacity.as_f64().filter(|v| v.is_finite()) else {
        return;
    };
    let alpha = alpha.clamp(0.0, 1.0);
    let Some(fills) = node.get_mut("fills").and_then(Value::as_array_mut) else {
        return;
    };
    for fill in fills.iter_mut().filter_map(Value::as_object_mut) {
        let layer = fill.get("opacity").and_then(Value::as_f64).unwrap_or(1.0);
        fill.insert("opacity".into(), json!(layer * alpha));
    }
}

// ---------------------------------------------------------------------------
// createPathNode (core/scene/pathHelpers.ts:254-330) — chart subset: the
// generators emit well-formed move/line/curve points, explicit x/y origin,
// no stroke/fill/closed options (those arrive via passthrough afterwards).
// ---------------------------------------------------------------------------

fn pt_f(p: &JsonMap, key: &str) -> f64 {
    p.get(key).and_then(Value::as_f64).unwrap_or(f64::NAN)
}

/// `normalizeClosedPathPoints` (pathHelpers.ts:234-249) — helper-time
/// `closed` is the SHAPE default `true` (the def's own `closed` only lands
/// later via passthrough; copy the bug).
fn normalize_closed_points(points: Vec<Value>) -> Vec<Value> {
    if points.len() < 2 {
        return points;
    }
    let first = points[0].as_object();
    let last = points[points.len() - 1].as_object();
    let (Some(first), Some(last)) = (first, last) else {
        return points;
    };
    if first.get("type").and_then(Value::as_str) != Some("move") {
        return points;
    }
    if last.get("type").and_then(Value::as_str) == Some("line")
        && pt_f(first, "x") == pt_f(last, "x")
        && pt_f(first, "y") == pt_f(last, "y")
    {
        let mut p = points;
        p.pop();
        return p;
    }
    points
}

/// `boundsFromPath` (utils/geometryHelpers.ts:692-851) for move/line/curve.
fn bounds_from_path(points: &[Value]) -> Option<(f64, f64)> {
    if points.is_empty() {
        return None;
    }
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let update = |x: f64, y: f64, mins: &mut (f64, f64, f64, f64)| {
        mins.0 = super::jsmath::js_min(mins.0, x);
        mins.1 = super::jsmath::js_min(mins.1, y);
        mins.2 = js_max(mins.2, x);
        mins.3 = js_max(mins.3, y);
    };
    let mut acc = (min_x, min_y, max_x, max_y);
    for v in points {
        let Some(p) = v.as_object() else { continue };
        let x = pt_f(p, "x");
        let y = pt_f(p, "y");
        if x.is_nan() || y.is_nan() {
            continue;
        }
        match p.get("type").and_then(Value::as_str) {
            Some("move") => {
                cx = x;
                cy = y;
                update(cx, cy, &mut acc);
            }
            Some("line") => {
                update(x, y, &mut acc);
                cx = x;
                cy = y;
            }
            Some("curve") => {
                if let (Some(cp1), Some(cp2)) = (
                    p.get("cp1").and_then(Value::as_object),
                    p.get("cp2").and_then(Value::as_object),
                ) {
                    update(cx, cy, &mut acc);
                    update(x, y, &mut acc);
                    let cp1x = pt_f(cp1, "x");
                    let cp1y = pt_f(cp1, "y");
                    let cp2x = pt_f(cp2, "x");
                    let cp2y = pt_f(cp2, "y");
                    for t in cubic_extrema(cx, cp1x, cp2x, x) {
                        update(
                            eval_cubic(cx, cp1x, cp2x, x, t),
                            eval_cubic(cy, cp1y, cp2y, y, t),
                            &mut acc,
                        );
                    }
                    for t in cubic_extrema(cy, cp1y, cp2y, y) {
                        update(
                            eval_cubic(cx, cp1x, cp2x, x, t),
                            eval_cubic(cy, cp1y, cp2y, y, t),
                            &mut acc,
                        );
                    }
                }
                cx = x;
                cy = y;
            }
            _ => {
                update(x, y, &mut acc);
                cx = x;
                cy = y;
            }
        }
    }
    (min_x, min_y, max_x, max_y) = acc;
    let _ = (max_x, max_y);
    if min_x.is_infinite() || min_y.is_infinite() {
        return None;
    }
    Some((min_x, min_y))
}

fn cubic_extrema(p0: f64, p1: f64, p2: f64, p3: f64) -> Vec<f64> {
    let a = -3.0 * p0 + 9.0 * p1 - 9.0 * p2 + 3.0 * p3;
    let b = 6.0 * p0 - 12.0 * p1 + 6.0 * p2;
    let c = -3.0 * p0 + 3.0 * p1;
    let mut extrema = Vec::new();
    if a.abs() < 1e-10 {
        if b.abs() > 1e-10 {
            let t = -c / b;
            if (0.0..=1.0).contains(&t) {
                extrema.push(t);
            }
        }
    } else {
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            let t1 = (-b + sq) / (2.0 * a);
            let t2 = (-b - sq) / (2.0 * a);
            if (0.0..=1.0).contains(&t1) {
                extrema.push(t1);
            }
            if (0.0..=1.0).contains(&t2) {
                extrema.push(t2);
            }
        }
    }
    extrema
}

fn eval_cubic(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
    let mt = 1.0 - t;
    mt * mt * mt * p0 + 3.0 * mt * mt * t * p1 + 3.0 * mt * t * t * p2 + t * t * t * p3
}

/// `offsetPathPoints` for move/line/curve.
fn offset_points(points: &[Value], dx: f64, dy: f64) -> Vec<Value> {
    points
        .iter()
        .map(|v| {
            let Some(p) = v.as_object() else {
                return v.clone();
            };
            let mut out = p.clone();
            if let Some(x) = p.get("x").and_then(Value::as_f64) {
                out.insert("x".into(), json!(x + dx));
            }
            if let Some(y) = p.get("y").and_then(Value::as_f64) {
                out.insert("y".into(), json!(y + dy));
            }
            for cp in ["cp1", "cp2", "cp"] {
                if let Some(c) = p.get(cp).and_then(Value::as_object) {
                    let mut co = c.clone();
                    if let Some(x) = c.get("x").and_then(Value::as_f64) {
                        co.insert("x".into(), json!(x + dx));
                    }
                    if let Some(y) = c.get("y").and_then(Value::as_f64) {
                        co.insert("y".into(), json!(y + dy));
                    }
                    out.insert(cp.into(), Value::Object(co));
                }
            }
            Value::Object(out)
        })
        .collect()
}

fn create_path_node(node: &mut JsonMap, points: &[Value], opt_x: f64, opt_y: f64) {
    // sanitizePathPoints: chart generators emit well-formed points; the
    // sanitizer is a structural no-op for them.
    let normalized = normalize_closed_points(points.to_vec());
    let has_explicit_origin = opt_x.is_finite() && opt_y.is_finite();
    let mut x = if has_explicit_origin { opt_x } else { 0.0 };
    let mut y = if has_explicit_origin { opt_y } else { 0.0 };
    let mut local_points = normalized.clone();
    if let Some((min_x, min_y)) = bounds_from_path(&normalized) {
        if min_x.is_finite() && min_y.is_finite() {
            if has_explicit_origin {
                x += min_x;
                y += min_y;
            } else {
                x = min_x;
                y = min_y;
            }
            local_points = offset_points(&normalized, -min_x, -min_y);
        }
    }
    node.insert("type".into(), json!("path"));
    node.insert("x".into(), json!(x));
    node.insert("y".into(), json!(y));
    node.insert("points".into(), Value::Array(local_points));
    // SHAPE_DEFAULTS.path (core/config/shapeDefaults.ts:134-141)
    paint_bag::set_stroke_color(node, json!("#000000"));
    paint_bag::set_stroke_width(node, json!(2.0));
    paint_bag::set_fill_color(node, json!("#cccccc"));
    node.insert("fillRule".into(), json!("nonzero"));
    node.insert("closed".into(), json!(true));
    node.insert("opacity".into(), json!(1.0));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_def_with_passthrough() {
        let mut def = Def::new("rectangle");
        def.set("x", json!(10.0))
            .set("y", json!(20.0))
            .set("width", json!(30.0))
            .set("height", json!(40.0))
            .set("fill", json!("#3b82f6"))
            .set("fillPriority", json!("color"))
            .set("name", json!("Chart Bar Q1"))
            .set("metadata", json!({"chartRole": "bar"}));
        let node = def_to_node(&def, "host", 0, 1.0).unwrap();
        assert_eq!(node.get("id"), Some(&json!("host_gen_proxy_0")));
        assert_eq!(node.get("rotation"), Some(&json!(0.0)));
        assert_eq!(node.get("fill"), Some(&json!("#3b82f6")));
        assert_eq!(
            node.get("metadata"),
            Some(&json!({"chartRole": "bar", "name": "Chart Bar Q1"}))
        );
        assert!(!node.contains_key("opacity"));
    }

    #[test]
    fn explicit_undefined_clears_path_defaults() {
        let mut def = Def::new("path");
        def.set(
            "points",
            json!([
                {"type": "move", "x": 5.0, "y": 5.0},
                {"type": "line", "x": 15.0, "y": 25.0}
            ]),
        )
        .set("x", json!(100.0))
        .set("y", json!(200.0))
        .set("stroke", Value::Null)
        .set("strokeWidth", Value::Null)
        .set("fill", json!("transparent"))
        .set("closed", json!(false));
        let node = def_to_node(&def, "h", 1, 1.0).unwrap();
        // bounds normalization: x += minX
        assert_eq!(node.get("x"), Some(&json!(105.0)));
        assert_eq!(node.get("y"), Some(&json!(205.0)));
        assert!(!node.contains_key("stroke"));
        assert!(!node.contains_key("strokeWidth"));
        assert_eq!(node.get("fill"), Some(&json!("transparent")));
        assert_eq!(node.get("closed"), Some(&json!(false)));
        assert_eq!(node.get("opacity"), Some(&json!(1.0))); // path helper default
        let pts = node.get("points").unwrap().as_array().unwrap();
        assert_eq!(pts[0], json!({"type": "move", "x": 0.0, "y": 0.0}));
    }

    #[test]
    fn fill_opacity_is_the_fill_layers_opacity() {
        // chartAreaFillOpacity.test.ts: a node-level `fillOpacity` is read by
        // no scene node, so it becomes the fill layers' opacity (clamped).
        let area = |fill_opacity: Value| {
            let mut def = Def::new("path");
            def.set(
                "points",
                json!([
                    {"type": "move", "x": 0.0, "y": 0.0},
                    {"type": "line", "x": 10.0, "y": 0.0},
                    {"type": "line", "x": 10.0, "y": 10.0}
                ]),
            )
            .fill_color(json!("#3b82f6"))
            .set("fillOpacity", fill_opacity)
            .set("closed", json!(true));
            def_to_node(&def, "h", 0, 1.0).unwrap()
        };
        let node = area(json!(0.3));
        assert!(!node.contains_key("fillOpacity"));
        assert_eq!(
            node.get("fills"),
            Some(&json!([{"id": "fill-0", "paint": {"type": "solid", "color": "#3b82f6"}, "opacity": 0.3}]))
        );
        assert_eq!(area(json!(4.0))["fills"][0]["opacity"], json!(1.0));
        assert_eq!(area(json!(-1.0))["fills"][0]["opacity"], json!(0.0));
    }

    #[test]
    fn host_opacity_composes() {
        let mut def = Def::new("ellipse");
        def.set("x", json!(0.0))
            .set("y", json!(0.0))
            .set("width", json!(10.0))
            .set("height", json!(10.0))
            .set("opacity", json!(0.5));
        let node = def_to_node(&def, "h", 2, 0.5).unwrap();
        assert_eq!(node.get("opacity"), Some(&json!(0.25)));
        let node2 = def_to_node(&def, "h", 2, 1.0).unwrap();
        assert_eq!(node2.get("opacity"), Some(&json!(0.5)));
    }

    #[test]
    fn line_def_needs_four_points() {
        let mut def = Def::new("line");
        def.set("points", json!([1.0, 2.0]));
        assert!(def_to_node(&def, "h", 0, 1.0).is_none());
    }
}
