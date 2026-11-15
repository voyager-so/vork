//! Node normalization at load — port of `normalizeNodeInPlace`
//! (`core/scene/scene.normalize.ts`) and its helper modules. These repair
//! current-schema input (agent-written, imported or malformed), not old
//! document shapes: there are no migrations (docs/document-format.md). All
//! passes are idempotent in-place mutations, run in the exact TS order.
//!
//!  1. generatedSource `kind` injection  (scene.normalize.ts `normalizeGeneratedSourceInPlace`)
//!  2. chart renderMode invariant        (scene.normalize.ts `normalizeGeneratedRenderModeInPlace`)
//!  3. path point sanitization           (scene.normalize.ts + pathHelpers.ts `sanitizePathPoints`)
//!  4. line anchor validation            (lineAnchorNormalize.ts)
//!  5. table healing                     (tableNormalize.ts)
//!
//! `dash` has no pass: every writer stores `number[]` (the agent paths parse
//! string presets through `parseDashPattern` before storing).
//!
//! The brush / variable-width stroke fields have no pass here: the TS editor
//! normalizes them on write (scene.mutations.ts, Scene.admitNode on insert),
//! and `stroke_model.rs` applies the same repairs when it parses them.

use crate::types::{JsonMap, RawNode};
use crate::value::{json, Map, Value};
use std::collections::HashSet;

/// Re-home a parsed JSON value at exactly the size it needs. serde_json builds
/// objects and arrays by pushing, so each one keeps the capacity its last
/// doubling left it (an object of 17 keys holds room for 28); across a
/// document's nodes that slack was about a third of the node table. Moves
/// every key and value (no string copies), depth first.
pub fn compact_value(v: &mut Value) {
    match v {
        Value::Object(m) => compact_map(m),
        Value::Array(a) => {
            for x in a.iter_mut() {
                compact_value(x);
            }
            a.shrink_to_fit();
        }
        _ => {}
    }
}

/// [`compact_value`] for an object.
pub fn compact_map(m: &mut JsonMap) {
    let old = std::mem::take(m);
    let mut exact = Map::with_capacity(old.len());
    for (k, mut v) in old {
        compact_value(&mut v);
        exact.insert(k, v);
    }
    *m = exact;
}

/// Entry point — `normalizeNodeInPlace` (scene.normalize.ts:173-230).
pub fn normalize_node_in_place(node: &mut RawNode) {
    normalize_generated_source_in_place(node);
    normalize_generated_render_mode_in_place(node);
    normalize_path_points_in_place(node);
    if node.node_type() == "line" {
        normalize_line_anchors_in_place(node);
    }
    if node.node_type() == "table" {
        normalize_table_in_place(node);
    }
    // A property patch merges into the already-rewritten node, so re-derive it from `_paintLayer` too.
    if node.node_type() == "paint" || node.json_field("_paintLayer").and_then(Value::as_bool) == Some(true) {
        normalize_paint_in_place(node);
    }
}

/// The prefix of a paint layer's surface source (`paint-surface:<w>x<h>:<node id>`): an image fill src the
/// engine resolves to the layer's surface texture, never to an asset.
pub const PAINT_SURFACE_SRC_PREFIX: &str = "paint-surface:";

/// Largest paint layer side in pixels (docs/design/raster-layers.md §6.3, Q5).
pub const PAINT_MAX_SIDE_PX: u32 = 8192;

/// A paint layer's pixel size: `ceil(width × density) × ceil(height × density)`, clamped to 1..=8192.
pub fn paint_surface_px(width: f64, height: f64, density: f64) -> (u32, u32) {
    let d = if density.is_finite() && density > 0.0 { density } else { 2.0 };
    let side = |v: f64| {
        let v = if v.is_finite() && v > 0.0 { v } else { 1.0 };
        ((v * d).ceil() as u32).clamp(1, PAINT_MAX_SIDE_PX)
    };
    (side(width), side(height))
}

/// The image-fill src naming a paint layer's surface. It depends on the node id and pixel size only, never on the
/// pixels (rule 2 of the raster design): a stroke never changes the draw item.
pub fn paint_surface_src(node_id: &str, w_px: u32, h_px: u32) -> String {
    format!("{PAINT_SURFACE_SRC_PREFIX}{w_px}x{h_px}:{node_id}")
}

/// Parse [`paint_surface_src`]: `(node id, width px, height px)`.
pub fn parse_paint_surface_src(src: &str) -> Option<(&str, u32, u32)> {
    let rest = src.strip_prefix(PAINT_SURFACE_SRC_PREFIX)?;
    let (size, id) = rest.split_once(':')?;
    let (w, h) = size.split_once('x')?;
    Some((id, w.parse().ok()?, h.parse().ok()?))
}

/// A paint node's pixel grid: its stored `surface.widthPx × surface.heightPx` (fixed when the layer is made; resizing
/// the node scales the pixels, it doesn't change the grid), clamped to 1..=8192. A surface without them (hand-written
/// test scenes) derives it from `width × density` (default density 2).
pub fn paint_node_px(width: f64, height: f64, surface: Option<&Value>) -> (u32, u32) {
    let int = |k: &str| {
        surface
            .and_then(|s| s.get(k))
            .and_then(Value::as_f64)
            .filter(|f| f.is_finite() && *f >= 1.0)
            .map(|f| (f.round() as u32).clamp(1, PAINT_MAX_SIDE_PX))
    };
    match (int("widthPx"), int("heightPx")) {
        (Some(w), Some(h)) => (w, h),
        _ => {
            let density = surface.and_then(|s| s.get("density")).and_then(Value::as_f64).unwrap_or(2.0);
            paint_surface_px(width, height, density)
        }
    }
}

/// A raster paint layer hydrates as a rectangle with one image fill that samples its surface texture, stretched over
/// the node's box (one texel is `width / widthPx` canvas units across). It keeps `surface` (so a committed stroke is
/// a one-node property edit) and gets no strokes, corner radius or other fills. The src names the node and its pixel
/// grid only, so neither a stroke nor a resize changes the draw item's texture.
fn normalize_paint_in_place(node: &mut RawNode) {
    let num = |v: Option<&Value>| v.and_then(Value::as_f64).filter(|f| f.is_finite() && *f > 0.0);
    let width = num(node.json_field("width")).unwrap_or(1.0);
    let height = num(node.json_field("height")).unwrap_or(1.0);
    let (w_px, h_px) = paint_node_px(width, height, node.json_field("surface"));
    let src = paint_surface_src(node.id(), w_px, h_px);
    node.set("type", Value::from("rectangle"));
    // The derived keys go FIRST, in a fixed order: a property patch merged into the node later (new keys append at
    // the end) then keeps the same key order as a full load of the patched node, so their hashes agree.
    let fields = node.fields_mut();
    let rest = std::mem::take(fields);
    fields.insert("_paintLayer".into(), Value::from(true));
    fields.insert("cornerRadius".into(), Value::from(0));
    fields.insert("strokes".into(), Value::Array(Vec::new()));
    fields.insert(
        "fills".into(),
        json!([{ "id": "paint-surface", "paint": {
            "type": "image",
            "src": src,
            "placement": {
                "fit": "crop",
                "offset": { "x": 0.0, "y": 0.0 },
                "scale": { "x": width / w_px as f64, "y": height / h_px as f64 },
                "box": { "width": width, "height": height },
                "naturalSize": { "width": w_px, "height": h_px },
            },
        }}]),
    );
    for (k, v) in rest {
        if !matches!(k.as_str(), "_paintLayer" | "cornerRadius" | "strokes" | "fills") {
            fields.insert(k, v);
        }
    }
}

fn is_finite_number(v: Option<&Value>) -> Option<f64> {
    v.and_then(Value::as_f64).filter(|f| f.is_finite())
}

// ---------------------------------------------------------------------------
// 1. generatedSource normalization (scene.normalize.ts:71-81 +
//    normalizeGeneratedGraphicSource, nodes.ts:1517-1703)
// ---------------------------------------------------------------------------

fn normalize_generated_source_in_place(node: &mut RawNode) {
    if node.node_type() != "generated" {
        return;
    }
    let kind = node.str("generatedKind").unwrap_or("").to_string();
    let needs = match node.json_field("generatedSource") {
        Some(Value::Object(src)) => {
            // TS trigger: `kind` mismatch (scene.normalize.ts:80-89). The TS
            // additionally normalizes EVERY generated source at node hydration
            // (`createGeneratedGraphicNode`, shapeHelpers.ts:801), which is
            // where agent-written chart sources with a raw row-array `data`
            // get coerced through `rowsToParsedChartData`. This mirror's only
            // hydration seam is the normalize pass, so trigger the normalize
            // for that chart shape here too; matching sources with a proper
            // ParsedData bundle stay untouched (byte-stability for existing
            // docs).
            src.get("kind").and_then(Value::as_str) != Some(kind.as_str())
                || (kind == "chart" && matches!(src.get("data"), Some(Value::Array(_))))
        }
        _ => false,
    };
    if !needs {
        return;
    }
    let src = node
        .json_field("generatedSource")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let normalized = normalize_generated_graphic_source(&kind, &src);
    node.set("generatedSource", Value::Object(normalized));
}

/// `normalizeGeneratedGraphicSource` (nodes.ts:1517-1703).
///
/// For `chart` sources whose `data` is a raw row ARRAY (the agent-written
/// "promoted" shape), the TS routes it through `rowsToParsedChartData`
/// (nodes.ts:1954-1963); this mirror does the same via the ported
/// `charts::datarows`, so native renders the identical real chart instead of
/// the old title-only ghost. Where the TS coercion THROWS (missing required
/// columns, uncoercible value cells) the port returns `None` and the raw
/// array is kept unchanged — fail closed: `chart_gate_reason` still reports
/// `generated:chart` Unsupported / `raw-array-data` and the doc routes to the
/// browser instead of silently rendering wrong data.
pub fn normalize_generated_graphic_source(kind: &str, source: &JsonMap) -> JsonMap {
    let mut out = Map::new();
    match kind {
        "qr" => {
            out.insert("kind".into(), json!("qr"));
            out.insert(
                "data".into(),
                json!(source.get("data").and_then(Value::as_str).unwrap_or("")),
            );
            // `?? DEFAULT_QR_ERROR_CORRECTION` — any non-null value passes
            // through unchanged (TS cast, nodes.ts:1526-1528).
            let ec = match source.get("errorCorrection") {
                Some(Value::Null) | None => json!(crate::defaults::generated::QR_ERROR_CORRECTION),
                Some(v) => v.clone(),
            };
            out.insert("errorCorrection".into(), ec);
            if let Some(Value::String(s)) = source.get("foregroundColor") {
                out.insert("foregroundColor".into(), json!(s));
            }
            if let Some(Value::String(s)) = source.get("backgroundColor") {
                out.insert("backgroundColor".into(), json!(s));
            }
            if let Some(Value::Number(n)) = source.get("borderModules") {
                out.insert("borderModules".into(), Value::Number(n.clone()));
            }
        }
        "latex" => {
            out.insert("kind".into(), json!("latex"));
            out.insert(
                "latex".into(),
                json!(source.get("latex").and_then(Value::as_str).unwrap_or("")),
            );
            let dm = match source.get("displayMode") {
                Some(Value::Null) | None => json!("display"),
                Some(v) => v.clone(),
            };
            out.insert("displayMode".into(), dm);
            let fs = match source.get("fontSize") {
                Some(Value::Number(n)) => Value::Number(n.clone()),
                _ => json!(crate::defaults::generated::LATEX_FONT_SIZE),
            };
            out.insert("fontSize".into(), fs);
            let color = match source.get("color") {
                Some(Value::String(s)) => json!(s),
                _ => json!(crate::defaults::generated::LATEX_COLOR),
            };
            out.insert("color".into(), color);
        }
        "chart" => {
            // nodes.ts:1556-1618
            let spec_raw = source
                .get("spec")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let top_styles = extract_series_styles(source.get("seriesStyles"));
            let spec_styles = extract_series_styles(spec_raw.get("seriesStyles"));
            let series_styles = top_styles.or(spec_styles);
            let mut spec = spec_raw.clone();
            spec.shift_remove("seriesStyles");
            spec.shift_remove("data");
            // A spec with a missing or PARTIAL `config` is what an IMPORTED
            // chart looks like: the backend importers describe what to plot,
            // and at most a title, not how Moda should style it. The TS mirror
            // completes it here (nodes.ts:1954-1957) and this must too — every
            // export byte is rendered server-side from the persisted source, so
            // repairing on one side only means a chart that draws in the editor
            // and exports as a placeholder.
            //
            // COMPLETED, not replaced: the generators index into config fields
            // directly, so a present-but-partial config throws where an absent
            // one would have been repaired. Authored values win.
            // `chart_config_issue`'s `Throws("config")` stays correct for a
            // spec that reaches the generator WITHOUT normalizing; this only
            // stops the normalizer producing one.
            {
                let authored = match spec.shift_remove("config") {
                    Some(Value::Object(map)) => map,
                    _ => JsonMap::new(),
                };
                let mut config = match crate::defaults::generated::default_chart_config() {
                    Value::Object(map) => map.clone(),
                    _ => JsonMap::new(),
                };
                for (key, value) in authored {
                    config.insert(key, value);
                }
                spec.insert("config".into(), Value::Object(config));
            }

            // `spec.xScale === 'time'` passes through to the coercion so
            // date-valued categories parse to `xTime` (nodes.ts:1956-1962).
            // Read from the pre-strip spec: stripping only removes
            // seriesStyles/data, so xScale is identical either way.
            let time_mode = spec_raw.get("xScale").and_then(Value::as_str) == Some("time");
            out.insert("kind".into(), json!("chart"));
            out.insert("chartVersion".into(), json!(1));
            out.insert("spec".into(), Value::Object(spec));
            let data = match source.get("data") {
                Some(Value::Array(a)) => {
                    match crate::charts::datarows::rows_to_parsed_chart_data(a, time_mode) {
                        Some(bundle) => Value::Object(bundle),
                        // TS throws for this shape — fail closed: keep the
                        // raw array so `chart_gate_reason` gates the doc to
                        // the browser (see the doc comment above).
                        None => Value::Array(a.clone()),
                    }
                }
                Some(Value::Object(o)) => Value::Object(o.clone()),
                _ => {
                    json!({"columns": [], "rows": [], "series": [], "categories": [], "valueRange": [0, 0]})
                }
            };
            out.insert("data".into(), data);
            if let Some(styles) = series_styles {
                if !styles.is_empty() {
                    out.insert("seriesStyles".into(), Value::Object(styles));
                }
            }
        }
        _ => {
            // Unknown kind: pass through unchanged (TS would hit the
            // exhaustive-check never branch; only reachable from malformed
            // docs — the manifest flags unknown generatedKind separately).
            return source.clone();
        }
    }
    out
}

/// `extractSeriesStyles` (nodes.ts:1496-1515): object-and-not-array
/// passthrough (Map instances can't occur in JSON input).
fn extract_series_styles(v: Option<&Value>) -> Option<JsonMap> {
    match v {
        Some(Value::Object(o)) => Some(o.clone()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// 2. chart renderMode invariant (scene.normalize.ts:92-98)
// ---------------------------------------------------------------------------

fn normalize_generated_render_mode_in_place(node: &mut RawNode) {
    if node.node_type() != "generated" {
        return;
    }
    if node.str("generatedKind") == Some("chart")
        && node.str("renderMode") != Some("native-children")
    {
        node.set("renderMode", json!("native-children"));
    }
}

// ---------------------------------------------------------------------------
// 3. path point sanitization (pathHelpers.ts:8-196 via scene.normalize.ts:100-127)
// ---------------------------------------------------------------------------

fn is_finite_xy_point(v: Option<&Value>) -> bool {
    matches!(v, Some(Value::Object(o))
        if is_finite_number(o.get("x")).is_some() && is_finite_number(o.get("y")).is_some())
}

/// `isValidPathPoint` (pathHelpers.ts:14-51).
fn is_valid_path_point(v: &Value) -> bool {
    let Some(o) = v.as_object() else { return false };
    if is_finite_number(o.get("x")).is_none() || is_finite_number(o.get("y")).is_none() {
        return false;
    }
    match o.get("type").and_then(Value::as_str) {
        Some("move") | Some("line") => {}
        Some("curve") => {
            if !is_finite_xy_point(o.get("cp1")) || !is_finite_xy_point(o.get("cp2")) {
                return false;
            }
        }
        Some("quadratic") => {
            if !is_finite_xy_point(o.get("cp")) {
                return false;
            }
        }
        Some("arc") => {
            if is_finite_number(o.get("rx")).is_none()
                || is_finite_number(o.get("ry")).is_none()
                || is_finite_number(o.get("xAxisRotation")).is_none()
                || !matches!(o.get("largeArc"), Some(Value::Bool(_)))
                || !matches!(o.get("sweep"), Some(Value::Bool(_)))
            {
                return false;
            }
        }
        _ => return false,
    }
    if let Some(h) = o.get("outgoingHandle") {
        if !h.is_null() && !is_finite_xy_point(Some(h)) {
            return false;
        }
    }
    true
}

/// `sanitizePathPoints` (pathHelpers.ts:136-196): strip `incomingHandle`,
/// strip malformed `outgoingHandle`, coerce arc 0/1 flags to booleans, drop
/// invalid points.
pub fn sanitize_path_points(value: Option<&Value>) -> (Vec<Value>, bool) {
    let Some(Value::Array(arr)) = value else {
        return (Vec::new(), true);
    };
    let mut out: Vec<Value> = Vec::with_capacity(arr.len());
    let mut changed = false;
    for candidate in arr {
        let Some(rec) = candidate.as_object() else {
            changed = true;
            continue;
        };
        let mut rec = rec.clone();
        let mut item_changed = false;
        if rec.shift_remove("incomingHandle").is_some() {
            item_changed = true;
        }
        if let Some(h) = rec.get("outgoingHandle") {
            if !h.is_null() && !is_finite_xy_point(Some(h)) {
                rec.shift_remove("outgoingHandle");
                item_changed = true;
            }
        }
        if rec.get("type").and_then(Value::as_str) == Some("arc") {
            for flag in ["largeArc", "sweep"] {
                let coerced = match rec.get(flag).and_then(Value::as_f64) {
                    Some(f) if f == 0.0 || f == 1.0 => Some(f == 1.0),
                    _ => None,
                };
                if let Some(b) = coerced {
                    rec.insert(flag.into(), Value::Bool(b));
                    item_changed = true;
                }
            }
        }
        let v = Value::Object(rec);
        if !is_valid_path_point(&v) {
            changed = true;
            continue;
        }
        if item_changed {
            changed = true;
        }
        out.push(v);
    }
    (out, changed)
}

/// Whether [`sanitize_path_points`] would change `value`, without cloning a
/// point: a clean path (every hydrate of every path) skips the rebuild.
fn path_points_need_sanitizing(value: Option<&Value>) -> bool {
    let Some(Value::Array(arr)) = value else {
        return true;
    };
    arr.iter().any(|candidate| {
        let Some(rec) = candidate.as_object() else {
            return true;
        };
        if rec.contains_key("incomingHandle") {
            return true;
        }
        if let Some(h) = rec.get("outgoingHandle") {
            if !h.is_null() && !is_finite_xy_point(Some(h)) {
                return true;
            }
        }
        if rec.get("type").and_then(Value::as_str) == Some("arc")
            && ["largeArc", "sweep"]
                .iter()
                .any(|flag| matches!(rec.get(*flag).and_then(Value::as_f64), Some(f) if f == 0.0 || f == 1.0))
        {
            return true;
        }
        !is_valid_path_point(candidate)
    })
}

fn normalize_path_points_in_place(node: &mut RawNode) {
    // Compact points are canonical (finite anchors and control points of move / line / curve / quadratic points),
    // which is always valid.
    if node.node_type() != "path" || node.is_compact() {
        return;
    }
    if !path_points_need_sanitizing(node.json_field("points")) {
        return;
    }
    let (points, changed) = sanitize_path_points(node.json_field("points"));
    if changed {
        node.set("points", Value::Array(points));
    }
}

// ---------------------------------------------------------------------------
// 11. legacy line anchors (lineAnchorNormalize.ts)
// ---------------------------------------------------------------------------

const KNOWN_PRESETS: [&str; 9] = [
    "top",
    "bottom",
    "left",
    "right",
    "center",
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
];

fn is_finite_integer(v: Option<&Value>) -> Option<f64> {
    is_finite_number(v).filter(|f| f.fract() == 0.0)
}

/// `normalizePerimeterParam` (lineAnchorNormalize.ts:54-69).
fn normalize_perimeter_param(raw: Option<&Value>) -> Option<JsonMap> {
    let o = raw?.as_object()?;
    match o.get("kind").and_then(Value::as_str) {
        Some("scalar") => {
            is_finite_number(o.get("t"))?;
            let mut m = o.clone();
            m.insert("kind".into(), json!("scalar"));
            Some(m)
        }
        Some("edge") => {
            let idx = is_finite_integer(o.get("edgeIndex"))?;
            if idx < 0.0 {
                return None;
            }
            is_finite_number(o.get("localT"))?;
            let mut m = o.clone();
            m.insert("kind".into(), json!("edge"));
            Some(m)
        }
        _ => None,
    }
}

/// `normalizeAnchor` (lineAnchorNormalize.ts:81-161). Returns the normalized
/// anchor or None (caller decides drop-vs-keep per :199-235).
fn normalize_anchor(raw: &Value) -> Option<Value> {
    let a = raw.as_object()?;
    let node_id = a.get("nodeId").and_then(Value::as_str)?.to_string();
    let sub_path = a.get("subPath").and_then(Value::as_str).map(str::to_string);

    let with_subpath = |mut m: JsonMap| -> Value {
        if let Some(sp) = &sub_path {
            m.insert("subPath".into(), json!(sp));
        }
        Value::Object(m)
    };

    match a.get("kind").and_then(Value::as_str) {
        Some("preset") => {
            let preset = a.get("preset").and_then(Value::as_str)?;
            if !KNOWN_PRESETS.contains(&preset) {
                return None;
            }
            let mut m = Map::new();
            m.insert("nodeId".into(), json!(node_id));
            m.insert("kind".into(), json!("preset"));
            m.insert("preset".into(), json!(preset));
            Some(with_subpath(m))
        }
        Some("bbox") => {
            let u = is_finite_number(a.get("u"))?;
            let v = is_finite_number(a.get("v"))?;
            let mut m = Map::new();
            m.insert("nodeId".into(), json!(node_id));
            m.insert("kind".into(), json!("bbox"));
            m.insert("u".into(), json!(u));
            m.insert("v".into(), json!(v));
            Some(with_subpath(m))
        }
        Some("perimeter") => {
            if let Some(param) = normalize_perimeter_param(a.get("param")) {
                let mut m = Map::new();
                m.insert("nodeId".into(), json!(node_id));
                m.insert("kind".into(), json!("perimeter"));
                m.insert("param".into(), Value::Object(param));
                return Some(with_subpath(m));
            }
            // Forward-compat: unknown param kinds preserved unchanged
            // (lineAnchorNormalize.ts:126-144).
            if let Some(Value::Object(p)) = a.get("param") {
                if let Some(kind) = p.get("kind").and_then(Value::as_str) {
                    if kind != "scalar" && kind != "edge" {
                        let mut m = Map::new();
                        m.insert("nodeId".into(), json!(node_id));
                        m.insert("kind".into(), json!("perimeter"));
                        m.insert("param".into(), Value::Object(p.clone()));
                        return Some(with_subpath(m));
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn is_perimeter_kind(raw: Option<&Value>) -> bool {
    matches!(raw, Some(Value::Object(o)) if o.get("kind").and_then(Value::as_str) == Some("perimeter"))
}

/// lineAnchorNormalize.ts:199-235.
pub fn normalize_line_anchors_in_place(node: &mut RawNode) -> bool {
    if node.node_type() != "line" {
        return false;
    }
    let mut changed = false;
    for key in ["startAnchor", "endAnchor"] {
        let raw = node.json_field(key).cloned();
        if is_perimeter_kind(raw.as_ref()) {
            let raw = raw.unwrap();
            if let Some(normalized) = normalize_anchor(&raw) {
                node.set(key, normalized);
                changed = true;
            } else {
                // Malformed known-perimeter payloads are dropped (set
                // undefined in TS → remove the key here).
                node.remove(key);
                changed = true;
            }
        }
    }
    changed
}

// ---------------------------------------------------------------------------
// 13. legacy table healing (tableNormalize.ts)
// ---------------------------------------------------------------------------

/// `isWellFormedTable` (tableNormalize.ts) — the allocation-free fast path.
/// Deliberately walks the grid without cloning it: the overwhelming majority of
/// tables are already sound, and this pass runs on every node of every document
/// at hydration (`scene.rs:61`).
fn table_is_well_formed(node: &RawNode) -> bool {
    let axis_ok = |key: &str| {
        node.array(key).is_some_and(|entries| {
            entries.iter().all(|entry| {
                matches!(entry, Value::Object(o) if is_finite_number(o.get("size")).is_some())
            })
        })
    };
    if !axis_ok("rows") || !axis_ok("columns") {
        return false;
    }
    let row_count = node.array("rows").map_or(0, Vec::len);
    let col_count = node.array("columns").map_or(0, Vec::len);
    let Some(cells) = node.array("cells") else {
        return false;
    };
    if cells.len() != row_count {
        return false;
    }
    let mut seen: HashSet<&str> = HashSet::new();
    for row in cells {
        let Some(row) = row.as_array() else {
            return false;
        };
        if row.len() != col_count {
            return false;
        }
        for cell in row {
            let Some(cell) = cell.as_object() else {
                return false;
            };
            match cell.get("id").and_then(Value::as_str) {
                Some(id) if !id.is_empty() => {
                    if !seen.insert(id) {
                        return false; // duplicate id across the whole grid
                    }
                }
                _ => return false,
            }
        }
    }
    true
}

/// One axis entry, normalized. AXIS ARRAYS WIN WHEN PRESENT — this only fills
/// the slots the caller asked for, and an existing entry OBJECT is kept whole so
/// an authored `rows[i].fill` (which `expand_table_node` reads for the per-row
/// background) survives; only a non-finite `size` is replaced.
fn normalize_table_axis(raw: Option<&Vec<Value>>, count: usize, default_size: f64) -> Vec<Value> {
    (0..count)
        .map(|i| match raw.and_then(|entries| entries.get(i)) {
            // A bare numeric entry is the oldest authored shape.
            Some(v @ Value::Number(_)) => match is_finite_number(Some(v)) {
                Some(size) => json!({ "size": size }),
                None => json!({ "size": default_size }),
            },
            Some(Value::Object(o)) => {
                let mut o = o.clone();
                if is_finite_number(o.get("size")).is_none() {
                    // NaN/Infinity are NOT finite. A NaN size propagates through
                    // every position accumulator in `expand_table_node` and
                    // poisons the whole table's geometry, so it can never survive.
                    o.insert("size".into(), json!(default_size));
                }
                Value::Object(o)
            }
            // missing / null / string / array
            _ => json!({ "size": default_size }),
        })
        .collect()
}

/// The padded cell, byte-identical to `createTableNodeCell`
/// (`core/scene/shapeHelpers.ts:706`) minus the random id — the id is minted
/// positionally in step 4 instead, so healing is a pure function of the table.
/// The empty-string `id` placeholder holds the key's position (JS object key
/// order) and reads as falsy, so step 4 always replaces it.
fn default_table_cell() -> Value {
    json!({
        "id": "",
        "textContent": crate::defaults::table::CELL_HTML_CONTENT,
        "textResizeMode": "no-resize",
        "textPadding": crate::defaults::table::CELL_TEXT_PADDING,
    })
}

/// `normalizeTableInPlace` (tableNormalize.ts). Tables were the
/// only node type with no entry in any scene normalization layer, so a malformed
/// grid (id-less or duplicate cell ids, ragged/oversized `cells`, a missing
/// axis, a NaN size) reached `expand_table_node` raw and dropped cells or
/// rendered NaN geometry. This heals the node in place at the load boundary;
/// unlike the TS twin it reports nothing — the browser owns the Sentry latch.
///
/// Never reorders or prepends: zebra striping (`backgroundColors[rowIdx % n]`)
/// and the header row are index-based, so padding always goes at the END and
/// overflow is dropped from the END.
pub fn normalize_table_in_place(node: &mut RawNode) -> bool {
    if node.node_type() != "table" || table_is_well_formed(node) {
        return false;
    }

    let rows_raw = node.array("rows").cloned();
    let cols_raw = node.array("columns").cloned();
    let cells_raw = node.array("cells").cloned();

    // Axis arrays win when present; the grid only synthesizes a MISSING axis.
    // With both absent the table becomes 0x0 (expands to nothing) rather than
    // inventing a size the author never wrote.
    let row_count = match (&rows_raw, &cells_raw) {
        (Some(rows), _) => rows.len(),
        (None, Some(cells)) => cells.len(),
        (None, None) => 0,
    };
    let col_count = match (&cols_raw, &cells_raw) {
        (Some(cols), _) => cols.len(),
        (None, Some(cells)) => cells
            .iter()
            .map(|row| row.as_array().map_or(0, Vec::len))
            .max()
            .unwrap_or(0),
        (None, None) => 0,
    };

    let rows = normalize_table_axis(
        rows_raw.as_ref(),
        row_count,
        crate::defaults::table::CELL_HEIGHT,
    );
    let columns = normalize_table_axis(
        cols_raw.as_ref(),
        col_count,
        crate::defaults::table::CELL_WIDTH,
    );

    let mut grid: Vec<Vec<Value>> = Vec::with_capacity(row_count);
    for r in 0..row_count {
        let src_row = cells_raw
            .as_ref()
            .and_then(|cells| cells.get(r))
            .and_then(Value::as_array);
        let mut row = Vec::with_capacity(col_count);
        for c in 0..col_count {
            match src_row.and_then(|src| src.get(c)) {
                // Reuse the authored cell whole — its text content, padding and
                // styling are never rewritten.
                Some(Value::Object(o)) => row.push(Value::Object(o.clone())),
                _ => row.push(default_table_cell()),
            }
        }
        grid.push(row);
    }

    // Cell ids — byte-identical to `resolveTableCellIds` (utils/tableSelection.ts:46-73) and its
    // inline twin in `expansion.rs:346-359`. The fallback is a pure function of
    // (tableId, row, col) and NEVER `generateNodeId`, which is what makes a second
    // pass a no-op even when a fallback collides with a real id elsewhere in the grid.
    let table_id = node.id().to_string();
    let mut seen: HashSet<String> = HashSet::new();
    for (r, row) in grid.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            let Some(cell) = cell.as_object_mut() else {
                continue;
            };
            let mut resolved = match cell.get("id").and_then(Value::as_str) {
                Some(id) if !id.is_empty() && !seen.contains(id) => id.to_string(),
                _ => format!("{table_id}_cell_{r}_{c}"),
            };
            // The positional fallback is not automatically free: an AUTHORED id elsewhere in
            // the grid can literally be `<tableId>_cell_<row>_<col>`, and this pass is the
            // first writer to persist that shape into `cell.id`. Claiming it unprobed would
            // emit the very duplicate this pass exists to remove — and because this lane
            // reports nothing, it would do so silently. Matches the TS twin byte for byte.
            if seen.contains(&resolved) {
                let base = resolved.clone();
                let mut disambiguator = 2;
                while seen.contains(&resolved) {
                    resolved = format!("{base}_{disambiguator}");
                    disambiguator += 1;
                }
            }
            if cell.get("id").and_then(Value::as_str) != Some(resolved.as_str()) {
                cell.insert("id".into(), json!(resolved));
            }
            seen.insert(resolved);
        }
    }

    node.set("rows", Value::Array(rows));
    node.set("columns", Value::Array(columns));
    node.set(
        "cells",
        Value::Array(grid.into_iter().map(Value::Array).collect()),
    );
    true
}

#[cfg(test)]
mod tests {
    /// A paint layer's pixel grid is stored: resizing the node scales the pixels over the new box and keeps the same
    /// surface src (so the same texture); a resize patch re-derived on the merged node equals a fresh load of it.
    #[test]
    fn paint_layer_resize_keeps_its_grid_and_src() {
        use super::*;
        use crate::value::json;
        let node = |w: f64, h: f64| {
            json!({ "id": "p1", "type": "paint", "x": 0, "y": 0, "width": w, "height": h, "rotation": 0, "scale": 1,
                    "surface": { "manifest": "", "widthPx": 800, "heightPx": 600, "tileSize": 256, "format": "rgba8-srgb" } })
        };
        let load = |v: crate::value::Value| {
            let mut raw = crate::types::RawNode::new(v.as_object().unwrap().clone().into_iter().collect());
            normalize_node_in_place(&mut raw);
            raw
        };
        let a = load(node(400.0, 300.0));
        let fill = |r: &crate::types::RawNode| r.json_field("fills").unwrap()[0]["paint"].clone();
        assert_eq!(fill(&a)["src"], json!("paint-surface:800x600:p1"));
        assert_eq!(fill(&a)["placement"]["scale"], json!({ "x": 0.5, "y": 0.5 }));
        // Resize: a patch merged into the normalized node, re-derived (as the resident scene does).
        let mut merged = a.to_json_map();
        merged.insert("width".into(), json!(1000.0));
        merged.insert("height".into(), json!(150.0));
        let mut patched = crate::types::RawNode::new(merged);
        normalize_node_in_place(&mut patched);
        let fresh = load(node(1000.0, 150.0));
        assert_eq!(fill(&patched), fill(&fresh));
        assert_eq!(fill(&fresh)["src"], json!("paint-surface:800x600:p1"), "same grid, same texture");
        assert_eq!(fill(&fresh)["placement"]["scale"], json!({ "x": 1.25, "y": 0.25 }));
        assert_eq!(fill(&fresh)["placement"]["naturalSize"], json!({ "width": 800, "height": 600 }));
        // Without a stored grid (hand-written scenes): width × density.
        assert_eq!(paint_node_px(100.0, 50.5, Some(&json!({ "density": 2 }))), (200, 101));
        assert_eq!(paint_node_px(100.0, 50.0, Some(&json!({ "widthPx": 99999, "heightPx": 3 }))), (8192, 3));
    }

    /// The no-clone pre-check agrees with the sanitizer on every case it handles.
    #[test]
    fn the_path_point_precheck_agrees_with_the_sanitizer() {
        use crate::value::json;
        let cases = [
            json!([{"type":"move","x":0,"y":0},{"type":"line","x":1,"y":2}]),
            json!([{"type":"curve","x":0,"y":0,"cp1":{"x":1,"y":1},"cp2":{"x":2,"y":2}}]),
            json!([{"type":"line","x":0,"y":0,"incomingHandle":{"x":1,"y":1}}]),
            json!([{"type":"line","x":0,"y":0,"outgoingHandle":{"x":"a","y":1}}]),
            json!([{"type":"line","x":0,"y":0,"outgoingHandle":null}]),
            json!([{"type":"arc","x":0,"y":0,"rx":1,"ry":1,"xAxisRotation":0,"largeArc":1,"sweep":false}]),
            json!([{"type":"arc","x":0,"y":0,"rx":1,"ry":1,"xAxisRotation":0,"largeArc":true,"sweep":false}]),
            json!([{"type":"line","x":0}]),
            json!([3, {"type":"line","x":0,"y":0}]),
            json!([]),
            json!("nope"),
        ];
        for c in &cases {
            assert_eq!(
                super::path_points_need_sanitizing(Some(c)),
                super::sanitize_path_points(Some(c)).1,
                "{c}"
            );
        }
        assert!(super::path_points_need_sanitizing(None));
    }

    /// The EXACT source the xlsx importer emits, through this lane.
    ///
    /// Both reviewers asked for this by name, and the reason is worth keeping:
    /// the import persists a config-less spec with raw row data, every export
    /// byte is rendered server-side from that persisted source, and a fill on
    /// the TS side only meant the editor drew a chart while the export drew a
    /// placeholder box. Nothing covered the emitted node at all.
    #[test]
    fn an_imported_chart_source_normalizes_with_a_config_in_this_lane_too() {
        let source = crate::value::json!({
            "kind": "chart",
            "chartVersion": 1,
            "spec": {"type": "bar", "x": 0, "y": 0, "width": 880.0, "height": 621.0, "title": "Revenue"},
            "data": [
                {"category": "Q1", "value": 1200.0, "series": "Revenue"},
                {"category": "Q2", "value": 2050.0, "series": "Revenue"}
            ]
        });
        let map = source.as_object().unwrap().clone();

        let out = normalize_generated_graphic_source("chart", &map);

        let spec = out
            .get("spec")
            .and_then(Value::as_object)
            .expect("spec survives");
        let config = spec
            .get("config")
            .and_then(Value::as_object)
            .expect("an imported chart must arrive with a config, or it exports as a placeholder");
        // Spot-check rather than the whole table: the table itself is pinned
        // against nodes.ts by `default_chart_config_matches_the_typescript`.
        assert_eq!(
            config.get("fontFamily").and_then(Value::as_str),
            Some("Inter")
        );
        assert!(config
            .get("palette")
            .and_then(Value::as_array)
            .is_some_and(|p| !p.is_empty()));

        // And the raw rows still become a real bundle rather than staying an array.
        let data = out
            .get("data")
            .and_then(Value::as_object)
            .expect("rows coerce to a bundle");
        assert!(data.contains_key("categories"));
    }

    /// The control: an authored config's own values always win.
    #[test]
    fn an_authored_chart_config_is_not_overwritten() {
        let source = crate::value::json!({
            "kind": "chart",
            "chartVersion": 1,
            "spec": {"type": "bar", "config": {"fontFamily": "Georgia"}},
            "data": [{"category": "Q1", "value": 1.0}]
        });
        let map = source.as_object().unwrap().clone();

        let out = normalize_generated_graphic_source("chart", &map);

        let config = out
            .get("spec")
            .and_then(Value::as_object)
            .and_then(|s| s.get("config"))
            .and_then(Value::as_object)
            .expect("config survives");
        assert_eq!(
            config.get("fontFamily").and_then(Value::as_str),
            Some("Georgia"),
            "the author's value must win over the default"
        );
        // And the fields they did NOT set are completed rather than left
        // missing — the generators index into them directly, so a partial
        // config is the shape that throws.
        assert!(
            config.len() > 60,
            "a partial config must be completed, got {} fields",
            config.len()
        );
        assert!(config.contains_key("palette"));
    }

    use super::*;
    use crate::value::json;

    fn node(v: Value) -> RawNode {
        RawNode::from_value(v).unwrap()
    }

    fn normalized(v: Value) -> RawNode {
        let mut n = node(v);
        normalize_node_in_place(&mut n);
        n
    }

    /// Every pass must be idempotent (PLAN M0c: "idempotent in-place passes").
    fn assert_idempotent(v: Value) {
        let mut once = node(v);
        normalize_node_in_place(&mut once);
        let mut twice = once.clone();
        normalize_node_in_place(&mut twice);
        assert_eq!(once, twice);
    }

    // 1. generatedSource kind injection (scene.normalize.ts:71-81)
    #[test]
    fn generated_source_kind_injected() {
        let n = normalized(json!({
            "id":"q","type":"generated","generatedKind":"qr",
            "x":0,"y":0,"width":10,"height":10,"rotation":0,"scale":1,
            "generatedVersion":1,"renderMode":"raster-asset",
            "generatedSource": {"data": "https://x", "backgroundColor": "#fff"}
        }));
        let src = n.object("generatedSource").unwrap();
        assert_eq!(src.get("kind"), Some(&json!("qr")));
        assert_eq!(src.get("errorCorrection"), Some(&json!("M"))); // nodes.ts:1182
        assert_eq!(src.get("backgroundColor"), Some(&json!("#fff")));
        assert_idempotent(json!({
            "id":"q","type":"generated","generatedKind":"qr",
            "x":0,"y":0,"width":10,"height":10,"rotation":0,"scale":1,
            "generatedVersion":1,"renderMode":"raster-asset",
            "generatedSource": {"data": "https://x"}
        }));
    }

    // 2. chart renderMode invariant (scene.normalize.ts:92-98)
    #[test]
    fn chart_render_mode_forced_native_children() {
        let n = normalized(json!({
            "id":"c","type":"generated","generatedKind":"chart",
            "x":0,"y":0,"width":10,"height":10,"rotation":0,"scale":1,
            "generatedVersion":1,"renderMode":"raster-asset",
            "generatedSource": {"kind":"chart","chartVersion":1,"spec":{"type":"bar"},"data":{"columns":[],"rows":[],"series":[],"categories":[],"valueRange":[0,0]}}
        }));
        assert_eq!(n.str("renderMode"), Some("native-children"));
        assert_idempotent(json!({
            "id":"c","type":"generated","generatedKind":"chart",
            "x":0,"y":0,"width":10,"height":10,"rotation":0,"scale":1,
            "generatedVersion":1,"renderMode":"raster-asset",
            "generatedSource": {"kind":"chart","chartVersion":1,"spec":{"type":"bar"},"data":{"columns":[],"rows":[],"series":[],"categories":[],"valueRange":[0,0]}}
        }));
    }

    #[test]
    fn chart_source_hoists_series_styles_and_strips_spec_data() {
        // nodes.ts:1556-1578
        let n = normalized(json!({
            "id":"c","type":"generated","generatedKind":"chart",
            "x":0,"y":0,"width":10,"height":10,"rotation":0,"scale":1,
            "generatedVersion":1,"renderMode":"native-children",
            "generatedSource": {
                "spec": {"type":"bar","data":[{"category":"a","value":1}],"seriesStyles":{"s1":{"color":"#f00"}}},
                "data": {"columns":[],"rows":[],"series":[],"categories":[],"valueRange":[0,0]}
            }
        }));
        let src = n.object("generatedSource").unwrap();
        let spec = src["spec"].as_object().unwrap();
        assert!(!spec.contains_key("data"));
        assert!(!spec.contains_key("seriesStyles"));
        assert_eq!(src["seriesStyles"]["s1"]["color"], json!("#f00"));
    }

    // 3. path point sanitization (pathHelpers.ts:136-196)
    #[test]
    fn path_points_sanitized() {
        let n = normalized(json!({"id":"p","type":"path","x":0,"y":0,"points":[
            {"type":"move","x":0,"y":0,"incomingHandle":{"x":1,"y":1}},
            {"type":"line","x":"oops","y":2},
            {"type":"arc","x":5,"y":5,"rx":1,"ry":1,"xAxisRotation":0,"largeArc":1,"sweep":0},
            {"type":"curve","x":9,"y":9,"cp1":{"x":1,"y":1},"cp2":{"x":2,"y":2},"outgoingHandle":"bad"}
        ]}));
        let pts = n.array("points").unwrap();
        assert_eq!(pts.len(), 3); // malformed 'line' dropped
        assert!(!pts[0].as_object().unwrap().contains_key("incomingHandle"));
        assert_eq!(pts[1]["largeArc"], json!(true)); // 0/1 coerced
        assert_eq!(pts[1]["sweep"], json!(false));
        assert!(!pts[2].as_object().unwrap().contains_key("outgoingHandle"));
        assert_idempotent(json!({"id":"p","type":"path","x":0,"y":0,"points":[
            {"type":"move","x":0,"y":0},
            {"type":"arc","x":5,"y":5,"rx":1,"ry":1,"xAxisRotation":0,"largeArc":1,"sweep":0}
        ]}));
    }

    // 11. legacy line anchors (lineAnchorNormalize.ts)
    #[test]
    fn line_perimeter_anchors_validated() {
        let v = json!({"id":"l","type":"line","points":[0,0,1,1],
            "endAnchor":{"nodeId":"b","kind":"perimeter","param":{"kind":"scalar","t":"NaN-ish"}}});
        let n = normalized(v);
        // malformed perimeter dropped (lineAnchorNormalize.ts:211-216)
        assert!(!n.has("endAnchor"));
        // unknown future param kind preserved (lineAnchorNormalize.ts:133-144)
        let v = json!({"id":"l","type":"line","points":[0,0,1,1],
            "endAnchor":{"nodeId":"b","kind":"perimeter","param":{"kind":"bezier","q":1}}});
        let n = normalized(v.clone());
        assert_eq!(
            n.json_field("endAnchor"),
            Some(&json!({"nodeId":"b","kind":"perimeter","param":{"kind":"bezier","q":1}}))
        );
        assert_idempotent(v);
    }

    // 13. legacy table healing (tableNormalize.ts)

    /// A cell id at (r, c), exactly as `resolveTableCellIds` mints it.
    fn cell_id(node: &RawNode, r: usize, c: usize) -> String {
        node.array("cells").unwrap()[r].as_array().unwrap()[c]["id"]
            .as_str()
            .unwrap()
            .to_string()
    }

    fn grid_dims(node: &RawNode) -> (usize, usize) {
        let cells = node.array("cells").unwrap();
        let cols = cells
            .first()
            .map_or(0, |r| r.as_array().map_or(0, Vec::len));
        (cells.len(), cols)
    }

    fn axis_sizes(node: &RawNode, key: &str) -> Vec<f64> {
        node.array(key)
            .unwrap()
            .iter()
            .map(|e| e["size"].as_f64().unwrap())
            .collect()
    }

    fn axis(count: usize, size: f64) -> Value {
        Value::Array((0..count).map(|_| json!({"size": size})).collect())
    }

    fn cells(rows: usize, cols: usize) -> Value {
        Value::Array(
            (0..rows)
                .map(|r| {
                    Value::Array(
                        (0..cols)
                            .map(|c| json!({"id": format!("authored_{r}_{c}")}))
                            .collect(),
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn well_formed_table_untouched() {
        let v = json!({"id":"t","type":"table","x":0,"y":0,
            "rows": axis(2, 50.0), "columns": axis(3, 150.0), "cells": cells(2, 3)});
        let mut n = node(v.clone());
        assert!(!normalize_table_in_place(&mut n));
        assert_eq!(n, node(v));
    }

    #[test]
    fn id_less_cells_get_positional_ids() {
        let n = normalized(json!({"id":"t","type":"table","x":0,"y":0,
            "rows": axis(2, 50.0), "columns": axis(2, 150.0),
            "cells": [[{}, {"id": ""}], [{"id": null}, {"textContent": "<p>x</p>"}]]}));
        assert_eq!(cell_id(&n, 0, 0), "t_cell_0_0");
        assert_eq!(cell_id(&n, 0, 1), "t_cell_0_1");
        assert_eq!(cell_id(&n, 1, 0), "t_cell_1_0");
        assert_eq!(cell_id(&n, 1, 1), "t_cell_1_1");
        // Authored content on an id-less cell is never rewritten.
        assert_eq!(
            n.array("cells").unwrap()[1].as_array().unwrap()[1]["textContent"],
            json!("<p>x</p>")
        );
    }

    #[test]
    fn duplicate_ids_deduped_positionally() {
        let n = normalized(json!({"id":"t","type":"table","x":0,"y":0,
            "rows": axis(1, 50.0), "columns": axis(3, 150.0),
            "cells": [[{"id":"dup"}, {"id":"dup"}, {"id":"dup"}]]}));
        // First occurrence keeps the authored id; the rest fall back positionally.
        assert_eq!(cell_id(&n, 0, 0), "dup");
        assert_eq!(cell_id(&n, 0, 1), "t_cell_0_1");
        assert_eq!(cell_id(&n, 0, 2), "t_cell_0_2");
    }

    /// The positional fallback can collide with an AUTHORED id elsewhere in the grid.
    /// This pass is the first writer to persist `<tableId>_cell_<r>_<c>` into
    /// `cell.id`, so that precondition is reachable — and a healed grid that still holds
    /// duplicates has not been healed. Mirrors the TS
    /// `disambiguates a fallback an AUTHORED id elsewhere already claimed`.
    #[test]
    fn fallback_colliding_with_an_authored_id_is_disambiguated() {
        let n = normalized(json!({"id":"t","type":"table","x":0,"y":0,
            "rows": axis(1, 50.0), "columns": axis(2, 150.0),
            "cells": [[{"id":"t_cell_0_1"}]]}));
        // (0,1) is padded, so its positional fallback is the id (0,0) already authored.
        assert_eq!(cell_id(&n, 0, 0), "t_cell_0_1");
        assert_eq!(cell_id(&n, 0, 1), "t_cell_0_1_2");

        let ids: HashSet<String> = (0..1)
            .flat_map(|r| (0..2).map(move |c| (r, c)))
            .map(|(r, c)| cell_id(&n, r, c))
            .collect();
        assert_eq!(ids.len(), 2, "healed grid must hold no duplicate cell ids");
    }

    /// A disambiguated grid must pass the well-formed fast path on the next hydration,
    /// or the table re-heals on every load and never converges.
    #[test]
    fn a_disambiguated_grid_converges() {
        let mut n = node(json!({"id":"t","type":"table","x":0,"y":0,
            "rows": axis(1, 50.0), "columns": axis(2, 150.0),
            "cells": [[{"id":"t_cell_0_1"}]]}));
        assert!(normalize_table_in_place(&mut n));
        let healed = n.clone();
        assert!(!normalize_table_in_place(&mut n));
        assert_eq!(n, healed);
    }

    #[test]
    fn ragged_grid_padded_to_column_count() {
        let n = normalized(json!({"id":"t","type":"table","x":0,"y":0,
            "rows": axis(3, 50.0), "columns": axis(3, 150.0),
            "cells": [[{"id":"a"}], [{"id":"b"},{"id":"c"},{"id":"d"}], "not-a-row"]}));
        assert_eq!(grid_dims(&n), (3, 3));
        let padded = &n.array("cells").unwrap()[0].as_array().unwrap()[1];
        assert_eq!(padded["id"], json!("t_cell_0_1"));
        assert_eq!(
            padded["textContent"],
            json!(crate::defaults::table::CELL_HTML_CONTENT)
        );
        assert_eq!(padded["textResizeMode"], json!("no-resize"));
        assert_eq!(padded["textPadding"], json!(8.0));
        // A non-array row is replaced wholesale by default cells.
        assert_eq!(cell_id(&n, 2, 0), "t_cell_2_0");
    }

    #[test]
    fn oversized_grid_truncated_from_the_end() {
        // Extra rows AND extra columns are dropped; nothing is reordered, so the
        // surviving cells keep their zebra-striping row index.
        let n = normalized(json!({"id":"t","type":"table","x":0,"y":0,
            "rows": axis(2, 50.0), "columns": axis(2, 150.0), "cells": cells(4, 5)}));
        assert_eq!(grid_dims(&n), (2, 2));
        assert_eq!(cell_id(&n, 0, 0), "authored_0_0");
        assert_eq!(cell_id(&n, 1, 1), "authored_1_1");
    }

    #[test]
    fn missing_cells_synthesized_from_the_axes() {
        let n = normalized(
            json!({"id":"t","type":"table","x":0,"y":0,"rows": axis(2, 50.0), "columns": axis(2, 150.0)}),
        );
        assert_eq!(grid_dims(&n), (2, 2));
        assert_eq!(cell_id(&n, 1, 1), "t_cell_1_1");
    }

    #[test]
    fn missing_rows_synthesized_from_the_grid() {
        let n = normalized(json!({"id":"t","type":"table","x":0,"y":0,
            "columns": axis(2, 150.0), "cells": cells(3, 2)}));
        assert_eq!(axis_sizes(&n, "rows"), vec![50.0, 50.0, 50.0]);
        // The columns axis wins over the grid's own width.
        assert_eq!(axis_sizes(&n, "columns"), vec![150.0, 150.0]);
        assert_eq!(grid_dims(&n), (3, 2));
    }

    #[test]
    fn missing_columns_synthesized_from_the_widest_row() {
        let n = normalized(json!({"id":"t","type":"table","x":0,"y":0,
            "rows": axis(2, 40.0),
            "cells": [[{"id":"a"}], [{"id":"b"},{"id":"c"},{"id":"d"}]]}));
        assert_eq!(axis_sizes(&n, "columns"), vec![150.0, 150.0, 150.0]);
        // The authored rows axis is preserved, not re-defaulted.
        assert_eq!(axis_sizes(&n, "rows"), vec![40.0, 40.0]);
        assert_eq!(grid_dims(&n), (2, 3));
    }

    #[test]
    fn both_axes_absent_becomes_zero_by_zero() {
        let n = normalized(json!({"id":"t","type":"table","x":0,"y":0}));
        assert_eq!(grid_dims(&n), (0, 0));
        assert!(n.array("rows").unwrap().is_empty());
        assert!(n.array("columns").unwrap().is_empty());
    }

    #[test]
    fn numeric_and_null_axis_entries_coerced() {
        let n = normalized(json!({"id":"t","type":"table","x":0,"y":0,
            "rows": [40, {"size": null}, null, "60"],
            "columns": [120, {"size": 90, "fill": "#eee"}],
            "cells": cells(4, 2)}));
        assert_eq!(axis_sizes(&n, "rows"), vec![40.0, 50.0, 50.0, 50.0]);
        assert_eq!(axis_sizes(&n, "columns"), vec![120.0, 90.0]);
        // An existing axis OBJECT is kept whole so authored `fill` survives.
        assert_eq!(n.array("columns").unwrap()[1]["fill"], json!("#eee"));
    }

    #[test]
    fn non_finite_axis_sizes_replaced() {
        // serde_json cannot hold NaN, but a string / bool `size` is the same
        // "not a finite number" branch and must not reach the accumulators.
        let n = normalized(json!({"id":"t","type":"table","x":0,"y":0,
            "rows": [{"size": "NaN", "fill": "#f00"}, {"size": true}],
            "columns": [{"size": []}],
            "cells": cells(2, 1)}));
        assert_eq!(axis_sizes(&n, "rows"), vec![50.0, 50.0]);
        assert_eq!(axis_sizes(&n, "columns"), vec![150.0]);
        assert_eq!(n.array("rows").unwrap()[0]["fill"], json!("#f00"));
    }

    #[test]
    fn row_count_grid_count_mismatch_both_directions() {
        // More rows than grid rows → the grid is padded.
        let n = normalized(json!({"id":"t","type":"table","x":0,"y":0,
            "rows": axis(4, 50.0), "columns": axis(1, 150.0), "cells": cells(2, 1)}));
        assert_eq!(grid_dims(&n), (4, 1));
        assert_eq!(cell_id(&n, 0, 0), "authored_0_0");
        assert_eq!(cell_id(&n, 3, 0), "t_cell_3_0");

        // Fewer rows than grid rows → the grid is truncated (rows axis wins).
        let n = normalized(json!({"id":"t","type":"table","x":0,"y":0,
            "rows": axis(1, 50.0), "columns": axis(1, 150.0), "cells": cells(3, 1)}));
        assert_eq!(grid_dims(&n), (1, 1));
        assert_eq!(axis_sizes(&n, "rows").len(), 1);
    }

    #[test]
    fn table_healing_is_idempotent() {
        assert_idempotent(json!({"id":"t","type":"table","x":0,"y":0,
            "rows": [50, {"size": null}], "columns": axis(3, 150.0),
            "cells": [[{"id":"dup"},{"id":"dup"}], [{}]]}));
        // A positional fallback that COLLIDES with a real id elsewhere still
        // settles after one pass, because the fallback is a pure function of
        // (tableId, row, col).
        assert_idempotent(json!({"id":"t","type":"table","x":0,"y":0,
            "rows": axis(2, 50.0), "columns": axis(2, 150.0),
            "cells": [[{},{"id":"t_cell_0_0"}], [{"id":"t_cell_0_0"},{}]]}));
    }

    /// The whole point of the pass: a healed table must expand to finite
    /// geometry. A NaN size used to propagate through every position
    /// accumulator in `expand_table_node` and poison the entire table.
    #[test]
    fn healed_table_expands_to_finite_geometry() {
        let n = normalized(json!({"id":"t","type":"table","x":10,"y":20,
            "rows": [{"size": "oops"}, 60, null],
            "cells": [[{"id":"a"}], [], [{"id":"a"},{"id":"b"},{}]],
            "backgroundColors": ["#fff", "#eee"], "headerBackgroundColor": "#ccc"}));
        let expanded = crate::expansion::expand_table_node(&n);
        assert!(!expanded.is_empty());
        for child in &expanded {
            for (key, value) in &child.to_json_map() {
                match value {
                    Value::Number(num) => assert!(
                        num.as_f64().is_some_and(f64::is_finite),
                        "{}.{key} is not finite: {value}",
                        child.id()
                    ),
                    Value::Array(items) => {
                        for item in items {
                            if let Value::Number(num) = item {
                                assert!(
                                    num.as_f64().is_some_and(f64::is_finite),
                                    "{}.{key} holds a non-finite entry: {value}",
                                    child.id()
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}
