//! Chart generators (PLAN M4 "chart generators ported by measured corpus
//! frequency"). The full production registry is ported: bar, line, pie/donut,
//! scatter, area, and combo — including time (date) X axes, via the d3-time
//! scale + multi-scale tick format ported in `timeticks`/`timefmt`.
//!
//! This module is the moda-scene equivalent of the production generator
//! registry (`utils/markup/elements/charts/generators/register.ts`) plus
//! `expandChartGeneratedNode` (`core/scene/chartExpansion.ts:60-159`):
//!
//! - children are emitted in the host's LOCAL coordinate space — the host is
//!   a transform parent (`nodeTransformUtils.getParentChain`), handled by
//!   `Scene::transform_parent_chain`;
//! - unknown chart types have no generator here → placeholder rect, exactly
//!   like a TS context that never registered them. Because that placeholder is
//!   byte-equal to the browser's (`chartExpansion.ts:131-159`), the manifest
//!   reports them `AxisStatus::Full` with a `dropped` + `spec_type` detail
//!   (assess.rs `generated:chart`, Q-95 browser-parity precedent) — they no
//!   longer route to the browser lane;
//! - time-axis charts (`spec.xScale === 'time'`) now render natively via the
//!   ported d3-time scale + multi-scale tick format (`timeticks`/`timefmt`),
//!   across line / area / scatter / combo. (Combo + time + bar series falls
//!   back to a categorical X axis, matching the TS — bars need band widths.)
//!
//! All numeric work happens in f64 with JS semantics (NaN-propagating maxima,
//! exact-decimal toFixed, …) so malformed persisted specs degrade exactly
//! like the browser (the `generated/chart-gated` fixture's broken spec must
//! yield ZERO definitions → placeholder, byte-equal to the TS output).

pub mod area_path;
pub mod axes;
pub mod cfg;
pub mod colorutil;
pub mod curves;
pub mod datarows;
pub mod defs;
pub mod fills;
pub mod format;
pub mod gen_area;
pub mod gen_bar;
pub mod gen_combo;
pub mod gen_line;
pub mod gen_pie;
pub mod gen_scatter;
pub mod html;
pub mod jsmath;
pub mod layout;
pub mod motion_parts;
pub mod parts;
pub mod scales;
pub mod paint_bag;
pub mod stack;
pub mod ticks;
pub mod timefmt;
pub mod timeticks;
pub mod value_axis_bounds;

use ticks::TickError;
use crate::value::{json, Map, Value};

use crate::expansion::transient_proxy_id;
use crate::types::RawNode;
use cfg::{DataView, SeriesStyles, SpecView};
use defs::{def_to_node, Def};

/// One generator signature for every ported chart type.
type ChartGenFn = fn(&SpecView, &DataView, SeriesStyles, &str) -> Result<Vec<Def>, TickError>;

/// THE porting seam: the chart-type NAME and its GENERATOR land together, so
/// "ported" (what `chart_gate_reason` / assess consider renderable) and the
/// actual expansion dispatch can never drift — adding a name here without a
/// real generator function does not compile. The full production registry
/// (bar, line, pie, scatter, area, combo); time X axes are ported too (see
/// `chart_gate_reason` / `timeticks`).
pub const PORTED_CHART_GENERATORS: [(&str, ChartGenFn); 6] = [
    ("bar", gen_bar::generate_bar_chart_nodes),
    ("line", gen_line::generate_line_chart_nodes),
    ("pie", gen_pie::generate_pie_chart_nodes),
    ("scatter", gen_scatter::generate_scatter_chart_nodes),
    ("area", gen_area::generate_area_chart_nodes),
    ("combo", gen_combo::generate_combo_chart_nodes),
];

/// The ported type names, derived from the generator table (single seam).
pub const PORTED_CHART_TYPES: [&str; 6] = [
    PORTED_CHART_GENERATORS[0].0,
    PORTED_CHART_GENERATORS[1].0,
    PORTED_CHART_GENERATORS[2].0,
    PORTED_CHART_GENERATORS[3].0,
    PORTED_CHART_GENERATORS[4].0,
    PORTED_CHART_GENERATORS[5].0,
];

pub fn is_ported_chart_type(ty: &str) -> bool {
    PORTED_CHART_GENERATORS.iter().any(|(name, _)| *name == ty)
}

fn chart_generator_for(ty: &str) -> Option<ChartGenFn> {
    PORTED_CHART_GENERATORS
        .iter()
        .find(|(name, _)| *name == ty)
        .map(|(_, f)| *f)
}

/// `expandChartGeneratedNode` (chartExpansion.ts:60-101). Returns the
/// transient children (LOCAL coordinates) or the placeholder rect.
///
/// A host carrying `_motionParts` (the motion graph's frame values for its
/// parts, `motion_parts.rs`; the render feed only, never saved) has those
/// fields written onto the part with each key.
pub fn expand_chart_generated_node(node: &RawNode) -> Result<Vec<RawNode>, TickError> {
    let Some(definitions) = chart_definitions(node)? else {
        // chartExpansion.ts:61 — kind mismatch returns [] (no placeholder).
        return Ok(Vec::new());
    };
    if definitions.is_empty() {
        return Ok(vec![create_chart_placeholder(node)]);
    }

    let host_opacity = node.opacity();
    let mut result: Vec<RawNode> = Vec::new();
    for (index, def) in definitions.iter().enumerate() {
        if let Some(n) = def_to_node(def, node.id(), index, host_opacity) {
            result.push(RawNode::new(n));
        }
    }
    if let Some(parts) = node.object(motion_parts::MOTION_PARTS_KEY) {
        motion_parts::apply_motion_parts(&mut result, parts);
    }
    Ok(result)
}

/// The chart's definitions (the generator's output, in draw order): `None` when
/// the host isn't a chart (kind mismatch), empty when it draws the placeholder.
pub(crate) fn chart_definitions(node: &RawNode) -> Result<Option<Vec<Def>>, TickError> {
    let src = node.object("generatedSource");
    let src_kind = src.and_then(|s| s.get("kind")).and_then(Value::as_str);
    if node.str("generatedKind") != Some("chart") || src_kind != Some("chart") {
        return Ok(None);
    }
    let src = src.unwrap();

    let host_w = node.f64("width").unwrap_or(f64::NAN);
    let host_h = node.f64("height").unwrap_or(f64::NAN);
    let chart_name = node
        .object("metadata")
        .and_then(|m| m.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("Chart")
        .to_string();

    let empty_map: Map = Map::new();
    let spec_map = src
        .get("spec")
        .and_then(Value::as_object)
        .unwrap_or(&empty_map);
    let data_map = src
        .get("data")
        .and_then(Value::as_object)
        .unwrap_or(&empty_map);
    // Categories are positions: a repeated label is its own point, bar or
    // slice, never merged into the first (`positionalCategories`).
    let positional = cfg::positional_categories(data_map);
    let data_map = positional.as_ref().unwrap_or(data_map);
    let spec = SpecView::new(spec_map, &empty_map, host_w, host_h);
    let data = DataView::new(data_map);
    let styles = SeriesStyles(src.get("seriesStyles").and_then(Value::as_object));

    // `generateChartDefinitions` — dispatched through the SAME table that
    // defines portedness (PORTED_CHART_GENERATORS), so assessment and
    // expansion cannot disagree; unregistered type → [] (placeholder).
    let definitions: Vec<Def> = match chart_generator_for(spec.ty()) {
        // Browser mirror for a config the browser generator THROWS on
        // unconditionally (`ChartConfigIssue::Throws`: absent/null config;
        // palette reading undefined/null with no seriesStyles to
        // short-circuit it; pie explode unusable with explodeAll falsy):
        // `expandChartGeneratedNode` catches the throw and renders the
        // placeholder (chartExpansion.ts:68-76). Rust has no throw at the
        // read site, so gate before dispatch. Without this, the NaN
        // degradation meant for missing SCALAR config fields applied here
        // too and emitted NaN-geometry definitions that rendered as a
        // silent ~1px blank (the config-less chart bug). The CONDITIONAL
        // class (`BrowserSafer`) keeps the degradation path — the browser
        // may render a real chart there, so a forced placeholder would be
        // wrong; assess routes those docs to the browser instead.
        Some(_)
            if matches!(
                chart_config_issue(
                    spec_map,
                    chart_series_styles_nonempty(src),
                    chart_palette_deref_reached(spec.ty(), src),
                ),
                Some(ChartConfigIssue::Throws(_))
            ) =>
        {
            Vec::new()
        }
        // Its numbers in its language's separators (`<chart lang>`, config.locale).
        Some(generate) => format::with_chart_locale(spec.config.s("locale"), || {
            generate(&spec, &data, styles, &chart_name)
        })?,
        None => Vec::new(),
    };
    Ok(Some(definitions))
}

/// `createChartPlaceholder` (chartExpansion.ts:131-159) — LOCAL coords.
fn create_chart_placeholder(node: &RawNode) -> RawNode {
    let mut m = Map::new();
    m.insert(
        "id".into(),
        json!(format!("{}_placeholder", transient_proxy_id(node.id(), 0))),
    );
    m.insert("type".into(), json!("rectangle"));
    m.insert("x".into(), json!(0.0));
    m.insert("y".into(), json!(0.0));
    m.insert(
        "width".into(),
        json!(node.f64("width").unwrap_or(0.0).max(1.0)),
    );
    m.insert(
        "height".into(),
        json!(node.f64("height").unwrap_or(0.0).max(1.0)),
    );
    m.insert("rotation".into(), json!(0.0));
    m.insert("scale".into(), json!(1.0));
    paint_bag::set_fill_color(&mut m, json!(crate::defaults::generated::CHART_PLACEHOLDER_FILL),
    );
    paint_bag::set_stroke_color(&mut m, json!(crate::defaults::generated::CHART_PLACEHOLDER_STROKE),
    );
    paint_bag::set_stroke_width(&mut m, json!(1.0));
    paint_bag::finish(&mut m);
    m.insert("opacity".into(), json!(node.opacity()));

    m.insert("metadata".into(), json!({"chartPlaceholder": true}));
    RawNode::new(m)
}

/// How an unusable `spec.config` diverges from the browser (see
/// `chart_config_issue`).
pub enum ChartConfigIssue {
    /// The browser generator THROWS unconditionally on this config —
    /// `expandChartGeneratedNode` catches it and renders the placeholder
    /// (chartExpansion.ts:68-76). Expansion mirrors the placeholder
    /// pre-dispatch; identical output in both lanes ⇒ assess reports
    /// Full/non-gating. Payload = which piece was missing.
    Throws(&'static str),
    /// Divergence is possible but CONDITIONAL on values this static mirror
    /// does not model (per-type seriesStyles short-circuits, JS string
    /// `.length`/`.includes` semantics) — the browser either renders a real
    /// chart where native would degrade, or placeholders where native would
    /// render chrome. Either way the browser lane is at least as good:
    /// assess reports Unsupported (gates to the browser); expansion keeps
    /// the shared NaN degradation (best effort for unrouted callers).
    BrowserSafer(&'static str),
}

/// JS-throw mirror for the config derefs the ported generators cannot
/// express (probed empirically against the production TS generators, all 6
/// types, real data). `None` = generator-safe: missing SCALAR config fields
/// NaN-degrade identically in both lanes and never throw, and a non-array
/// non-null palette of a non-string type resolves to the same
/// `'#3b82f6'` default in both lanes (colors.ts:12's `?? palette[0] ?? ...`
/// chain ≡ the empty-palette read here).
///
/// The throw/divergence surface is exactly the ARRAY-typed config fields:
/// - `"config"` — `spec.config` absent/null: every generator dereferences it
///   on a straight-line path BEFORE any data-dependent early return
///   (layout + `generateBackgroundNode` config reads — barChart.ts:71/76,
///   lineChart.ts:41, pieChart.ts:219, …), so JS throws for every data
///   shape → `Throws`. (A non-object config never throws at its own derefs
///   — JS property reads on primitives yield undefined, ≡ `{}` — so it
///   falls through to the palette/explode classification below.)
/// - `"explode"` (pie only; checked first — pieChart derefs explode at :285/
///   :356 BEFORE the palette read at :363): `config.explode.includes(...)`
///   throws when explode reads undefined/null or has no `.includes` method
///   (number/bool/object) → `Throws` — UNLESS `config.explodeAll` is truthy,
///   which short-circuits both derefs (`explodeAll || explode...`) → fine. A
///   STRING explode does not throw but uses JS substring `.includes`
///   semantics this port does not model → `BrowserSafer`.
/// - `"palette"` — `config.palette` reads undefined/null: series colors
///   resolve through `resolvePaletteColor`, whose `palette.length`
///   (colors.ts:11) throws → `Throws` — UNLESS the source carries
///   `seriesStyles`, whose per-type style fields short-circuit the palette
///   deref (`seriesStyle?.fill || resolvePaletteColor(...)` barChart.ts:346;
///   `?.stroke ||` lineChart.ts:182; …). Whether the browser then renders or
///   throws depends on which elements are styled with which fields →
///   `BrowserSafer`. A STRING palette does not throw but chooses colors by
///   JS char indexing (`palette[i % len]` → single characters) that this
///   port does not model → `BrowserSafer`. An ARRAY palette with any
///   non-string entry, or an OBJECT palette, also → `BrowserSafer`: TS
///   feeds the raw entry to `parseFill` (a `5` throws on `.trim` →
///   placeholder; a `null` picks the blue default; an array-like object
///   indexes real colors) while this port reads ""/empty — divergent, and
///   the browser is right in each subcase.
///
/// The palette/explode throws only fire if the generator actually REACHES
/// the deref: the generators emit background/title/source-footer nodes and
/// then early-return on empty/degenerate data (barChart.ts:104-112 — probed:
/// bar with a background, no palette and empty categories renders the REAL
/// background node, and a zero-total pie returns before touching explode).
/// `deref_reached` (see `chart_palette_deref_reached`) says the per-element
/// color/explode loop is provably reached; when it is not, `Throws` demotes
/// to `BrowserSafer` — never a false placeholder, and the browser lane is
/// right in every subcase.
fn chart_config_issue(
    spec: &Map,
    styles_nonempty: bool,
    deref_reached: bool,
) -> Option<ChartConfigIssue> {
    use ChartConfigIssue::{BrowserSafer, Throws};
    // Reach-aware certainty: a data-dependent throw is only "certain" when
    // the color/explode loop is provably reached.
    let throws_if_reached = |piece: &'static str| {
        Some(if deref_reached {
            Throws(piece)
        } else {
            BrowserSafer(piece)
        })
    };
    let cfg = match spec.get("config") {
        Some(Value::Object(cfg)) => Some(cfg),
        None | Some(Value::Null) => return Some(Throws("config")),
        // Non-object primitive ≡ {} for every JS property read.
        Some(_) => None,
    };
    let get = |key: &str| cfg.and_then(|c| c.get(key));
    if spec.get("type").and_then(Value::as_str) == Some("pie") && !cfg::truthy(get("explodeAll")) {
        match get("explode") {
            Some(Value::Array(_)) => {}
            Some(Value::String(_)) => return Some(BrowserSafer("explode")),
            // undefined/null (`.includes` deref throws) and non-array
            // non-string (`.includes` is not a function) both throw.
            _ => return throws_if_reached("explode"),
        }
    }
    match get("palette") {
        None | Some(Value::Null) => {
            if styles_nonempty {
                Some(BrowserSafer("palette"))
            } else {
                throws_if_reached("palette")
            }
        }
        Some(Value::String(_)) => Some(BrowserSafer("palette")),
        Some(Value::Array(entries)) => {
            if entries.iter().all(Value::is_string) {
                None
            } else {
                Some(BrowserSafer("palette"))
            }
        }
        Some(Value::Object(_)) => Some(BrowserSafer("palette")),
        // number/bool: both lanes resolve the identical '#3b82f6' default.
        Some(_) => None,
    }
}

/// Does the source carry any `seriesStyles` entries (the hoisted top-level
/// record — `ChartGeneratedSource.seriesStyles`)? Non-empty styles can
/// short-circuit the browser's palette deref, so they demote a missing
/// palette from a certain throw to a conditional divergence.
fn chart_series_styles_nonempty(src: &Map) -> bool {
    src.get("seriesStyles")
        .and_then(Value::as_object)
        .is_some_and(|styles| !styles.is_empty())
}

/// Is the per-element color/explode loop — the site of the palette/explode
/// throws — provably REACHED for this data? Conservative mirror of the
/// generators' data-dependent early returns (they emit background/title
/// chrome and return before the loop on degenerate data):
/// - every type needs non-empty `rows`;
/// - pie builds slices from positive-valued rows (a zero-total pie
///   early-returns before the explode/palette derefs — probed);
/// - scatter builds points from rows with numeric x AND y;
/// - the categorical types (bar/line/area/combo) early-return when
///   `orderedCategories` is empty, i.e. when `categories` and `series` are
///   BOTH empty (`resolveChartDimensions`, barChart.ts:110-112).
/// `false` means "not provably reached" — the classification then stays on
/// the browser-safer side, never a false placeholder. (A raw-array `data`
/// is not a bundle → `false`; its divergence is owned by `raw-array-data`.)
fn chart_palette_deref_reached(spec_type: &str, src: &Map) -> bool {
    let Some(Value::Object(data)) = src.get("data") else {
        return false;
    };
    let Some(rows) = data
        .get("rows")
        .and_then(Value::as_array)
        .filter(|rows| !rows.is_empty())
    else {
        return false;
    };
    let num = |row: &Value, key: &str| {
        row.as_object()
            .and_then(|m| m.get(key))
            .and_then(Value::as_f64)
    };
    match spec_type {
        "pie" => rows
            .iter()
            .any(|row| num(row, "value").is_some_and(|v| v > 0.0)),
        "scatter" => rows
            .iter()
            .any(|row| num(row, "x").is_some() && num(row, "y").is_some()),
        _ => {
            let nonempty = |key: &str| {
                data.get(key)
                    .and_then(Value::as_array)
                    .is_some_and(|list| !list.is_empty())
            };
            nonempty("categories") || nonempty("series")
        }
    }
}

/// Does the persisted `data` carry any rows (either the raw-array shape or a
/// ParsedData bundle)? With EMPTY data both lanes draw the placeholder no
/// matter what the config says, so a missing config is only *causal* — and
/// only worth an axis — when there was real data to lose.
fn chart_data_non_empty(src: &Map) -> bool {
    match src.get("data") {
        Some(Value::Array(rows)) => !rows.is_empty(),
        Some(Value::Object(bundle)) => bundle
            .get("rows")
            .and_then(Value::as_array)
            .is_some_and(|rows| !rows.is_empty()),
        _ => false,
    }
}

/// Manifest predicate (assess.rs): does this chart deviate from the fully
/// ported render path? Returns a detail with a `reason` discriminant, or None
/// when fully renderable by the ported set. assess.rs splits on `reason`:
/// - `"unregistered-type"` → the browser has no generator either — both lanes
///   draw the identical placeholder, so the axis is `Full` with a
///   browser-parity `dropped` note (Q-95 precedent), NOT a routing gate;
/// - `"raw-array-data"` → an UNCOERCIBLE raw row array. Coercible arrays no
///   longer reach this predicate: `normalize_generated_graphic_source`
///   (normalize.rs) routes them through the ported `rowsToParsedChartData`
///   (`charts::datarows`) at scene load, exactly like the TS hydration path,
///   so native renders the same real chart the browser does. An array that
///   SURVIVES normalization is one the TS coercion would THROW on (missing
///   value/category columns, bad value cells) — fail closed and keep gating
///   to the browser rather than silently rendering wrong data. Checked
///   BEFORE the config classification below: the TS failure for these docs
///   is a HYDRATION throw (createGeneratedGraphicNode), upstream of the
///   generator, so no generator-level placeholder-parity claim can hold;
/// - `"missing-config"` → ported type, real BUNDLE data, but the config
///   makes the browser generator throw UNCONDITIONALLY
///   (`ChartConfigIssue::Throws`; the `missing` detail says which piece):
///   the browser catch renders the placeholder and expansion here mirrors
///   it pre-dispatch. Identical output in both lanes → `Full` (non-gating),
///   but observable — this state used to render as a silent ~1px blank with
///   a zero-axis manifest;
/// - `"conditional-config"` → the config's divergence from the browser is
///   CONDITIONAL (`ChartConfigIssue::BrowserSafer`: seriesStyles palette
///   short-circuits, JS string palette/explode semantics, unproven deref
///   reach) — the browser may render a real chart where native degrades, so
///   the axis is `Unsupported` (gates to the browser, the lane that is
///   right in every subcase).
pub fn chart_gate_reason(src: &Map) -> Option<Map> {
    let spec = src.get("spec").and_then(Value::as_object);
    let spec_type = spec
        .and_then(|s| s.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let mut detail = Map::new();
    detail.insert("spec_type".into(), json!(spec_type));
    if !is_ported_chart_type(spec_type) {
        detail.insert("reason".into(), json!("unregistered-type"));
        return Some(detail);
    }
    // Ported type but RAW-ARRAY `data` that survived normalization: scene
    // load (`normalize_generated_graphic_source`) coerces every array the TS
    // `rowsToParsedChartData` accepts into the ParsedData bundle, so an array
    // still present here is one the TS coercion would THROW on. assess.rs
    // keeps this case GATING (Unsupported) — fail closed to the browser
    // rather than guess at malformed data. An EMPTY array is benign (it
    // coerces to the empty bundle → zero definitions in both lanes).
    // Checked BEFORE the config classification: the TS failure here is a
    // HYDRATION throw, upstream of the generator, so the generator-level
    // placeholder-parity reasoning below cannot apply.
    if src
        .get("data")
        .and_then(Value::as_array)
        .is_some_and(|rows| !rows.is_empty())
    {
        detail.insert("reason".into(), json!("raw-array-data"));
        return Some(detail);
    }
    // Ported type but an UNUSABLE `spec.config` with real data (see
    // `chart_config_issue`): unconditional browser throws report
    // `missing-config` (placeholder parity, non-gating), conditional or
    // unproven divergence reports `conditional-config` (Unsupported —
    // browser lane). With EMPTY data both lanes produce zero definitions →
    // the identical placeholder whatever the config says, so no axis fires
    // (parity-benign, like the empty raw-array case). `spec` is Some here —
    // a missing spec has spec_type "" and returned `unregistered-type`
    // above.
    let issue = spec.and_then(|s| {
        chart_config_issue(
            s,
            chart_series_styles_nonempty(src),
            chart_palette_deref_reached(spec_type, src),
        )
    });
    if let Some(issue) = issue {
        if chart_data_non_empty(src) {
            let (reason, piece) = match issue {
                ChartConfigIssue::Throws(piece) => ("missing-config", piece),
                ChartConfigIssue::BrowserSafer(piece) => ("conditional-config", piece),
            };
            detail.insert("reason".into(), json!(reason));
            detail.insert("missing".into(), json!(piece));
            return Some(detail);
        }
    }
    // Time X axes (`spec.xScale === 'time'`) are now ported (d3-time scale +
    // multi-scale tick format — see `timeticks`/`timefmt`), so they no longer
    // gate. Combo + time + bar series falls back to a categorical X axis inside
    // the generator (bars need band widths), matching the TS — still a real
    // render, so still ungated.
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RawNode;

    #[test]
    fn excessive_tick_counts_refuse_all_chart_generators_and_scene_edits() {
        for fixture in [
            include_str!("../../tests/fixtures/chart-bar.json"),
            include_str!("../../tests/fixtures/chart-line.json"),
            include_str!("../../tests/fixtures/chart-area.json"),
            include_str!("../../tests/fixtures/chart-scatter.json"),
            include_str!("../../tests/fixtures/chart-combo.json"),
        ] {
            let mut doc: Value = crate::value::from_str(fixture).unwrap();
            let mut scene = crate::Scene::load(&doc).unwrap();
            let before = scene.doc.nodes.clone();
            doc["nodes"][0]["generatedSource"]["spec"]["config"]["yTickCount"] = json!(1e12);
            let node = RawNode::from_value(doc["nodes"][0].clone()).unwrap();
            assert_eq!(expand_chart_generated_node(&node).unwrap_err(), TickError);
            assert!(crate::Scene::load(&doc).err().unwrap().to_string().contains("tick"));
            assert!(scene.apply_node_edits(vec![node.clone()], &[], &[], None).is_err());
            assert_eq!(scene.doc.nodes, before);
            // Only the final upsert for an id is authoritative.
            assert!(scene.apply_node_edits(vec![node, before[0].clone()], &[], &[], None).is_ok());
            assert_eq!(scene.doc.nodes, before);
        }
    }

    #[test]
    fn tick_budget_covers_numeric_log_time_and_label_fitting() {
        use scales::*;
        for scale in [create_linear_scale((0.0, 1.0), 100.0), create_log_scale((1.0, 1.1), 100.0), create_time_scale((0.0, 1e9), 100.0)] {
            assert_eq!(scale.ticks(1e12), Err(TickError));
        }
        assert_eq!(timeticks::utc_ticks(0, 1000000000, 1e12), Err(TickError));
        assert_eq!(generate_ticks(0.0, 1.0, 1e12), Err(TickError));
        assert!(axes::fit_bottom_tick_count(&create_linear_scale((0.0, 1.0), 0.0), (0.0, 1.0), 1000.0, None, 12.0).is_err());
    }

    fn chart_host(spec: Value, data: Value) -> RawNode {
        RawNode::from_value(json!({
            "id": "ch1", "type": "generated", "generatedKind": "chart",
            "x": 40.0, "y": 40.0, "width": 400.0, "height": 300.0,
            "rotation": 0, "scale": 1,
            "renderMode": "native-children",
            "generatedSource": {
                "kind": "chart", "chartVersion": 1,
                "spec": spec, "data": data
            }
        }))
        .unwrap()
    }

    /// `config.locale` (`<chart lang>`): value labels and ticks in its
    /// separators (TS: test/chartNumberLocale.test.ts).
    #[test]
    fn a_chart_locale_writes_its_numbers_in_that_language() {
        let data: Value = crate::value::from_str(r##"{"columns":["category","value"],"rows":[{"category":"A","value":1234.5},{"category":"B","value":2500}],"series":[],"categories":["A","B"],"valueRange":[0,2500]}"##).unwrap();
        let labels = |locale: Option<&str>| -> (Vec<String>, Vec<String>) {
            let mut spec = json!({"type": "bar", "x": 0, "y": 0, "width": 600, "height": 400});
            spec["config"] = crate::value::from_str::<Value>(r##"{"fontFamily":"Inter","fontSize":12,"labelColor":"#666666","titleColor":"#1a1a1a","subtitleColor":"#666666","titleSize":16,"titleWeight":600,"titleAlign":"left","subtitleSize":12,"palette":["#3b82f6"],"background":"transparent","showGrid":true,"gridColor":"#e5e5e5","gridStyle":"solid","gridWidth":1,"showXAxis":true,"showYAxis":true,"axisColor":"#cccccc","axisWidth":1,"showXLabels":true,"showYLabels":true,"yTickCount":5,"yFormat":"auto","yScale":"linear","barRadius":0,"barGap":0.2,"barGroupGap":0.1,"showLabels":true,"labelPosition":"top","labelFormat":"auto","valueFormat":"auto","valueDensity":"auto","valueLabelColor":"#666666","valueLabelSize":11,"showLegend":"auto","legendPosition":"bottom","legendAlign":"center","legendGap":8,"legendMarker":"square","legendMarkerSize":12,"legendColor":"#666666","padding":16}"##).unwrap();
            if let Some(locale) = locale {
                spec["config"]["locale"] = json!(locale);
            }
            let out = expand_chart_generated_node(&chart_host(spec, data.clone())).unwrap();
            let texts = |role: &str| -> Vec<String> {
                out.iter()
                    .map(|n| n.to_json_map())
                    .filter(|m| m.get("metadata").and_then(|x| x.get("chartRole")) == Some(&json!(role)))
                    .map(|m| {
                        let html = m.get("htmlContent").and_then(Value::as_str).unwrap_or("");
                        html.split('>').nth(1).and_then(|t| t.split('<').next()).unwrap_or("").to_string()
                    })
                    .collect()
            };
            (texts("value-label"), texts("y-label"))
        };
        let (values, ticks) = labels(Some("de"));
        assert_eq!(values, vec!["1.234,5", "2.500"]);
        assert!(ticks.iter().any(|t| t.len() == 5 && t.as_bytes()[1] == b'.'), "{ticks:?}");
        assert_eq!(labels(None).0, vec!["1,234.5", "2,500"]);
    }

    /// A scatter's x axis is its own quantity: `yFormat: percent` with no
    /// `xFormat` leaves the x ticks plain (TS: test/chartScatterXFormat.test.ts).
    #[test]
    fn a_scatter_x_axis_does_not_take_the_y_format() {
        let data: Value = crate::value::from_str(r##"{"columns":["x","y"],"rows":[{"x":100,"y":0.25},{"x":450,"y":0.5},{"x":900,"y":0.75}],"series":[],"categories":[],"valueRange":[0.25,0.75]}"##).unwrap();
        let labels = |x_format: Option<&str>, role: &str| -> Vec<String> {
            let mut spec = json!({"type": "scatter", "x": 0, "y": 0, "width": 500, "height": 320});
            spec["config"] = crate::value::from_str::<Value>(r##"{"fontFamily":"Inter","fontSize":12,"labelColor":"#666666","titleColor":"#1a1a1a","subtitleColor":"#666666","titleSize":16,"titleWeight":600,"titleAlign":"left","subtitleSize":12,"palette":["#3b82f6"],"background":"transparent","showGrid":true,"gridColor":"#e5e5e5","gridStyle":"solid","gridWidth":1,"showXAxis":true,"showYAxis":true,"axisColor":"#cccccc","axisWidth":1,"showXLabels":true,"showYLabels":true,"yTickCount":5,"xTickCount":5,"yFormat":"percent","yScale":"linear","showLabels":false,"labelFormat":"auto","valueFormat":"auto","valueLabelColor":"#666666","valueLabelSize":11,"showLegend":"auto","legendPosition":"bottom","legendAlign":"center","legendGap":8,"legendMarker":"square","legendMarkerSize":12,"legendColor":"#666666","pointSize":6,"pointShape":"circle","pointFill":"","pointStroke":"","pointStrokeWidth":0,"pointOpacity":0.7,"minPointSize":8,"maxPointSize":40,"padding":16}"##).unwrap();
            if let Some(f) = x_format {
                spec["xFormat"] = json!(f);
            }
            expand_chart_generated_node(&chart_host(spec, data.clone())).unwrap()
                .iter()
                .map(|n| n.to_json_map())
                .filter(|m| m.get("metadata").and_then(|x| x.get("chartRole")) == Some(&json!(role)))
                .map(|m| {
                    let html = m.get("htmlContent").and_then(Value::as_str).unwrap_or("");
                    html.split('>').nth(1).and_then(|t| t.split('<').next()).unwrap_or("").to_string()
                })
                .collect()
        };
        let x = labels(None, "x-label");
        assert!(!x.is_empty() && x.iter().all(|t| !t.contains('%')), "{x:?}");
        assert!(labels(None, "y-label").iter().any(|t| t.contains('%')));
        assert!(labels(Some("currency"), "x-label").iter().any(|t| t.contains('$')));
    }

    /// `show-labels` with `value-format="compact-currency"`: an empty cell is a
    /// gap with no value label, and the compact axis ticks use the value
    /// labels' K / M / B suffixes (TS: test/chartCompactLabels.test.ts).
    #[test]
    fn compact_currency_labels_skip_empty_cells_and_match_the_axis() {
        let data: Value = crate::value::from_str(r##"{"columns":["category","value"],"rows":[{"category":"Q1","value":98000},{"category":"Q2","value":0,"missing":true},{"category":"Q3","value":120000}],"series":[],"categories":["Q1","Q2","Q3"],"valueRange":[0,120000]}"##).unwrap();
        let mut spec = json!({"type": "bar", "x": 0, "y": 0, "width": 800, "height": 500});
        spec["config"] = crate::value::from_str::<Value>(r##"{"fontFamily":"Inter","fontSize":12,"labelColor":"#666666","titleColor":"#1a1a1a","subtitleColor":"#666666","titleSize":16,"titleWeight":600,"titleAlign":"left","subtitleSize":12,"palette":["#3b82f6"],"background":"transparent","showGrid":true,"gridColor":"#e5e5e5","gridStyle":"solid","gridWidth":1,"showXAxis":true,"showYAxis":true,"axisColor":"#cccccc","axisWidth":1,"showXLabels":true,"showYLabels":true,"yTickCount":5,"yFormat":"compact-currency","yScale":"linear","barRadius":0,"barGap":0.2,"barGroupGap":0.1,"showLabels":true,"labelPosition":"top","labelFormat":"auto","valueFormat":"compact-currency","valueDensity":"auto","valueLabelColor":"#666666","valueLabelSize":11,"showLegend":"auto","legendPosition":"bottom","legendAlign":"center","legendGap":8,"legendMarker":"square","legendMarkerSize":12,"legendColor":"#666666","padding":16}"##).unwrap();
        let out = expand_chart_generated_node(&chart_host(spec, data)).unwrap();
        let texts = |role: &str| -> Vec<(String, String)> {
            out.iter()
                .map(|n| n.to_json_map())
                .filter(|m| {
                    m.get("metadata").and_then(|x| x.get("chartRole")) == Some(&json!(role))
                })
                .map(|m| {
                    let category = m["metadata"]
                        .get("chartCategory")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let html = m.get("htmlContent").and_then(Value::as_str).unwrap_or("");
                    let text = html
                        .split('>')
                        .nth(1)
                        .and_then(|t| t.split('<').next())
                        .unwrap_or("")
                        .to_string();
                    (category, text)
                })
                .collect()
        };
        let values = texts("value-label");
        assert_eq!(
            values,
            vec![
                ("Q1".to_string(), "$98K".to_string()),
                ("Q3".to_string(), "$120K".to_string())
            ],
            "no label for the empty Q2"
        );
        let ticks: Vec<String> = texts("y-label").into_iter().map(|(_, t)| t).collect();
        assert!(ticks.iter().any(|t| t.ends_with('K')), "{ticks:?}");
        assert!(!ticks.iter().any(|t| t.ends_with('k')), "{ticks:?}");
    }

    /// A negative bar's value label sits past the bar's end, below it, as a positive bar's sits above it; with the
    /// default label position it was drawn above the bar's top, on the zero line over the positive area. Where below
    /// the bar would run into the category labels (a bar reaching the axis's end), it goes inside the bar's end. The TS
    /// generator matches (editor test chartNegativeBarLabels.test.ts).
    #[test]
    fn negative_bar_labels_sit_below_their_bars() {
        // The axis ends at -50: Q1's bar reaches it, so there's no room for its label below; Q4's has room.
        let data: Value = crate::value::from_str(r##"{"columns":["category","value"],"rows":[{"category":"Q1","value":-45},{"category":"Q2","value":120},{"category":"Q3","value":88},{"category":"Q4","value":-12}],"series":[],"categories":["Q1","Q2","Q3","Q4"],"valueRange":[-45,120]}"##).unwrap();
        for position in ["top", "outside"] {
            let mut spec = json!({"type": "bar", "x": 0, "y": 0, "width": 560, "height": 360});
            spec["config"] = crate::value::from_str::<Value>(r##"{"fontFamily":"Inter","fontSize":12,"labelColor":"#666666","titleColor":"#1a1a1a","subtitleColor":"#666666","titleSize":16,"titleWeight":600,"titleAlign":"left","subtitleSize":12,"palette":["#3b82f6"],"background":"transparent","showGrid":true,"gridColor":"#e5e5e5","gridStyle":"solid","gridWidth":1,"showXAxis":true,"showYAxis":true,"axisColor":"#cccccc","axisWidth":1,"showXLabels":true,"showYLabels":true,"yTickCount":5,"yFormat":"auto","yScale":"linear","barRadius":0,"barGap":0.2,"barGroupGap":0.1,"showLabels":true,"labelPosition":"top","labelFormat":"auto","valueFormat":"auto","valueDensity":"auto","valueLabelColor":"#666666","valueLabelSize":11,"showLegend":"auto","legendPosition":"bottom","legendAlign":"center","legendGap":8,"legendMarker":"square","legendMarkerSize":12,"legendColor":"#666666","padding":16}"##).unwrap();
            spec["config"]["labelPosition"] = json!(position);
            let out = expand_chart_generated_node(&chart_host(spec, data.clone())).unwrap();
            let maps: Vec<_> = out.iter().map(|n| n.to_json_map()).collect();
            let role = |m: &crate::value::Map| {
                m.get("metadata").and_then(|x| x.get("chartRole")).and_then(Value::as_str).unwrap_or("").to_string()
            };
            let boxes = |want: &str| -> std::collections::HashMap<String, (f64, f64)> {
                maps.iter()
                    .filter(|m| role(m) == want)
                    .map(|m| {
                        let category = m["metadata"]["chartCategory"].as_str().unwrap_or("").to_string();
                        (category, (m["y"].as_f64().unwrap(), m["height"].as_f64().unwrap()))
                    })
                    .collect()
            };
            let category_top = maps
                .iter()
                .filter(|m| role(m) == "x-label")
                .map(|m| m["y"].as_f64().unwrap())
                .fold(f64::INFINITY, f64::min);
            assert!(category_top.is_finite(), "{position}: no category labels");
            let (labels, bars) = (boxes("value-label"), boxes("bar"));
            for category in ["Q1", "Q4"] {
                let ((ly, lh), (by, _)) = (labels[category], bars[category]);
                assert!(ly >= by, "{position} {category}: label y {ly} above the zero line {by}");
                assert!(ly + lh <= category_top, "{position} {category}: label bottom {} past the category labels at {category_top}", ly + lh);
            }
            let ((ly, _), (by, bh)) = (labels["Q4"], bars["Q4"]);
            assert!(ly >= by + bh, "{position} Q4: room below the bar, label y {ly} above its end {}", by + bh);
            for category in ["Q2", "Q3"] {
                let ((ly, lh), (by, _)) = (labels[category], bars[category]);
                assert!(ly + lh <= by, "{position} {category}: label bottom {} below the bar's top {by}", ly + lh);
            }
        }
    }

    /// A vertical bar's value label is centred on its bar. Its box is at least `MIN_VALUE_LABEL_WIDTH` wide; on a
    /// narrower bar (a grouped chart in a small box) it started at the bar's left edge, so the label sat half the
    /// overhang to the right, over the next bar. The TS generator matches (editor test chartBarLabelCentre.test.ts).
    #[test]
    fn narrow_bar_value_labels_are_centred_on_their_bars() {
        let data: Value = crate::value::from_str(r##"{"columns":["category","series","value"],"rows":[{"category":"2019","value":12.1,"series":"Fund"},{"category":"2019","value":10.4,"series":"Benchmark"},{"category":"2020","value":4.2,"series":"Fund"},{"category":"2020","value":3.1,"series":"Benchmark"},{"category":"2021","value":15.7,"series":"Fund"},{"category":"2021","value":13.2,"series":"Benchmark"},{"category":"2022","value":-8.3,"series":"Fund"},{"category":"2022","value":-5,"series":"Benchmark"},{"category":"2023","value":9.9,"series":"Fund"},{"category":"2023","value":11.4,"series":"Benchmark"},{"category":"2024","value":7.2,"series":"Fund"},{"category":"2024","value":6.5,"series":"Benchmark"},{"category":"2025","value":-2.4,"series":"Fund"},{"category":"2025","value":1.8,"series":"Benchmark"}],"series":["Fund","Benchmark"],"categories":["2019","2020","2021","2022","2023","2024","2025"],"valueRange":[-8.3,15.7],"sourceColumns":["Year","Fund","Benchmark"]}"##).unwrap();
        let base: Value = crate::value::from_str(r##"{"type":"bar","mode":"grouped","x":0,"y":0,"width":310,"height":190,"config":{"fontFamily":"Inter","fontSize":8,"labelColor":"#666666","titleColor":"#1a1a1a","subtitleColor":"#666666","titleSize":16,"titleWeight":600,"titleAlign":"left","subtitleSize":12,"palette":["#3b82f6","#ef4444","#22c55e","#f59e0b"],"background":"transparent","showGrid":true,"gridColor":"#e5e5e5","gridStyle":"solid","gridWidth":1,"showXAxis":true,"showYAxis":true,"axisColor":"#cccccc","axisWidth":1,"showXLabels":true,"showYLabels":true,"yTickCount":5,"yFormat":"auto","yScale":"linear","barRadius":0,"barGap":0.2,"barGroupGap":0.1,"showLabels":true,"labelPosition":"top","labelFormat":"auto","valueFormat":"auto","valueDensity":"auto","valueLabelColor":"#666666","valueLabelSize":11,"showLegend":"auto","legendPosition":"bottom","legendAlign":"center","legendGap":8,"legendMarker":"square","legendMarkerSize":12,"legendColor":"#666666","strokeWidth":2,"strokeStyle":"solid","curve":"linear","strokeCap":"round","strokeJoin":"round","showPoints":false,"pointSize":6,"pointShape":"circle","pointFill":"","pointStroke":"","pointStrokeWidth":0,"fillOpacity":0.3,"baseline":0,"pointOpacity":0.7,"minPointSize":8,"maxPointSize":40,"xTickCount":5,"innerRadius":0,"startAngle":-90,"direction":"cw","padAngle":0,"explode":[],"explodeDistance":10,"explodeAll":false,"centerLabel":"","centerSublabel":"","centerColor":"#1a1a1a","centerLabelColor":"#1a1a1a","centerSublabelColor":"#666666","centerSize":24,"pieLabelPosition":"outside","pieLabelFormat":"percent","minLabelAngle":10,"padding":16,"cornerRadius":0}}"##).unwrap();
        for position in ["top", "outside", "inside"] {
            let mut spec = base.clone();
            spec["config"]["labelPosition"] = json!(position);
            let out = expand_chart_generated_node(&chart_host(spec, data.clone())).unwrap();
            let maps: Vec<_> = out.iter().map(|n| n.to_json_map()).collect();
            let boxes = |want: &str| -> std::collections::HashMap<String, (f64, f64)> {
                maps.iter()
                    .filter(|m| m.get("metadata").and_then(|x| x.get("chartRole")).and_then(Value::as_str) == Some(want))
                    .map(|m| {
                        let meta = &m["metadata"];
                        let category = meta["chartCategory"].as_str().unwrap_or("");
                        let key = format!("{category}/{}", meta["chartSeries"].as_str().unwrap_or(""));
                        (key, (m["x"].as_f64().unwrap(), m["width"].as_f64().unwrap()))
                    })
                    .collect()
            };
            let (labels, bars) = (boxes("value-label"), boxes("bar"));
            assert_eq!(labels.len(), 14, "{position}");
            let mut narrow = 0;
            for (key, (lx, lw)) in &labels {
                let (bx, bw) = bars[key];
                if bw < *lw {
                    narrow += 1;
                }
                let (label_centre, bar_centre) = (lx + lw / 2.0, bx + bw / 2.0);
                assert!((label_centre - bar_centre).abs() < 1e-9, "{position} {key}: label centre {label_centre}, bar centre {bar_centre}");
            }
            assert!(narrow > 0, "{position}: the bars are narrower than a label box");
        }
    }

    /// Explicit empty cells are gaps (data.ts `missing`), as the TS generators
    /// draw them: series B is `105, _, 110, _` over Q1..Q4. Expected points are
    /// the production TS `expandChartGeneratedNode` output for this source
    /// (the empty Q2 splits the line into single points; the trailing Q4 adds
    /// no point at the floor). Before the port, the native path drew B through
    /// value 0 at Q2 and Q4.
    /// A value axis along the bottom thins its ticks until its labels clear each other
    /// (chartBottomAxisFit.test.ts `fitBottomTickCount`): horizontal bars beside long category
    /// names at 20 px drew "0 200400600800 1,000". The grid takes the same ticks; at 12 px every
    /// tick stays.
    #[test]
    fn bottom_value_axis_thins_ticks_until_labels_fit() {
        let data: Value = crate::value::from_str(r##"{"columns":["category","value"],"rows":[{"category":"Cenozoic","value":66},{"category":"Mesozoic","value":186},{"category":"Paleozoic","value":289},{"category":"Neoproterozoic","value":461},{"category":"Paleoproterozoic","value":900},{"category":"Hadean","value":500}],"series":[],"categories":["Cenozoic","Mesozoic","Paleozoic","Neoproterozoic","Paleoproterozoic","Hadean"],"valueRange":[0,900]}"##).unwrap();
        let cases = [
            (20, vec![0.0, 500.0, 1000.0]),
            (12, vec![0.0, 200.0, 400.0, 600.0, 800.0, 1000.0]),
        ];
        for (font_size, expected) in cases {
            let config: Value = crate::value::from_str(
                &r##"{"fontFamily":"Inter","fontSize":FS,"labelColor":"#666666","titleColor":"#1a1a1a","subtitleColor":"#666666","titleSize":16,"titleWeight":600,"titleAlign":"left","subtitleSize":12,"palette":["#3b82f6","#ef4444","#22c55e","#f59e0b"],"background":"transparent","showGrid":true,"gridColor":"#e5e5e5","gridStyle":"solid","gridWidth":1,"showXAxis":true,"showYAxis":true,"axisColor":"#cccccc","axisWidth":1,"showXLabels":true,"showYLabels":true,"yTickCount":5,"yFormat":"auto","yScale":"linear","barRadius":0,"barGap":0.2,"barGroupGap":0.1,"showLabels":false,"labelPosition":"top","labelFormat":"auto","valueFormat":"auto","valueDensity":"auto","valueLabelColor":"#666666","valueLabelSize":11,"showLegend":"auto","legendPosition":"bottom","legendAlign":"center","legendGap":8,"legendMarker":"square","legendMarkerSize":12,"legendColor":"#666666","strokeWidth":2,"strokeStyle":"solid","curve":"linear","strokeCap":"round","strokeJoin":"round","showPoints":false,"pointSize":6,"pointShape":"circle","pointFill":"","pointStroke":"","pointStrokeWidth":0,"fillOpacity":0.3,"baseline":0,"pointOpacity":0.7,"minPointSize":8,"maxPointSize":40,"xTickCount":5,"innerRadius":0,"startAngle":-90,"direction":"cw","padAngle":0,"explode":[],"explodeDistance":10,"explodeAll":false,"centerLabel":"","centerSublabel":"","centerColor":"#1a1a1a","centerLabelColor":"#1a1a1a","centerSublabelColor":"#666666","centerSize":24,"pieLabelPosition":"outside","pieLabelFormat":"percent","minLabelAngle":10,"padding":16,"cornerRadius":0,"stroke":"#000000"}"##.replace("FS", &font_size.to_string()),
            )
            .unwrap();
            let mut spec = json!({"type": "bar", "orientation": "horizontal", "x": 0, "y": 0, "width": 400, "height": 300});
            spec["config"] = config;
            let out = expand_chart_generated_node(&chart_host(spec, data.clone())).unwrap();
            let values = |role: &str| -> Vec<f64> {
                out.iter()
                    .filter_map(|n| {
                        let meta = n.json_field("metadata")?;
                        (meta.get("chartRole") == Some(&json!(role)))
                            .then(|| meta.get("chartValue").and_then(Value::as_f64).unwrap_or(f64::NAN))
                    })
                    .collect()
            };
            assert_eq!(values("x-label"), expected, "{font_size} px labels");
            assert_eq!(values("grid-line"), expected, "{font_size} px grid");
        }
    }

    /// `x-label` / `y-label` draw on every chart with axes, each in its own band: the X title under the X tick
    /// labels, the turned Y title left of the Y tick labels (TS: `test/chartAxisTitles.test.ts`).
    #[test]
    fn axis_titles_draw_in_their_own_bands() {
        let data: Value = crate::value::from_str(r##"{"columns":["category","value"],"rows":[{"category":"Downtown","value":1204},{"category":"Airport","value":842},{"category":"Riverside","value":610}],"series":[],"categories":["Downtown","Airport","Riverside"],"valueRange":[0,1204]}"##).unwrap();
        let config: Value = crate::value::from_str(
            r##"{"fontFamily":"Inter","fontSize":FS,"labelColor":"#666666","titleColor":"#1a1a1a","subtitleColor":"#666666","titleSize":16,"titleWeight":600,"titleAlign":"left","subtitleSize":12,"palette":["#3b82f6","#ef4444","#22c55e","#f59e0b"],"background":"transparent","showGrid":true,"gridColor":"#e5e5e5","gridStyle":"solid","gridWidth":1,"showXAxis":true,"showYAxis":true,"axisColor":"#cccccc","axisWidth":1,"showXLabels":true,"showYLabels":true,"yTickCount":5,"yFormat":"auto","yScale":"linear","barRadius":0,"barGap":0.2,"barGroupGap":0.1,"showLabels":false,"labelPosition":"top","labelFormat":"auto","valueFormat":"auto","valueDensity":"auto","valueLabelColor":"#666666","valueLabelSize":11,"showLegend":"auto","legendPosition":"bottom","legendAlign":"center","legendGap":8,"legendMarker":"square","legendMarkerSize":12,"legendColor":"#666666","strokeWidth":2,"strokeStyle":"solid","curve":"linear","strokeCap":"round","strokeJoin":"round","showPoints":false,"pointSize":6,"pointShape":"circle","pointFill":"","pointStroke":"","pointStrokeWidth":0,"fillOpacity":0.3,"baseline":0,"pointOpacity":0.7,"minPointSize":8,"maxPointSize":40,"xTickCount":5,"innerRadius":0,"startAngle":-90,"direction":"cw","padAngle":0,"explode":[],"explodeDistance":10,"explodeAll":false,"centerLabel":"","centerSublabel":"","centerColor":"#1a1a1a","centerLabelColor":"#1a1a1a","centerSublabelColor":"#666666","centerSize":24,"pieLabelPosition":"outside","pieLabelFormat":"percent","minLabelAngle":10,"padding":16,"cornerRadius":0,"stroke":"#000000"}"##.replace("FS", "12").as_str(),
        )
        .unwrap();
        for (ty, orientation) in [("bar", "vertical"), ("bar", "horizontal"), ("line", "vertical"), ("area", "vertical")] {
            let mut spec = json!({"type": ty, "orientation": orientation, "x": 0, "y": 0, "width": 600, "height": 400,
                "xLabel": "Route", "yLabel": "Riders (k)"});
            spec["config"] = config.clone();
            let out = expand_chart_generated_node(&chart_host(spec, data.clone())).unwrap();
            let with_role = |role: &str| -> Vec<&crate::types::RawNode> {
                out.iter()
                    .filter(|n| n.json_field("metadata").and_then(|m| m.get("chartRole")) == Some(&json!(role)))
                    .collect()
            };
            let num = |n: &crate::types::RawNode, k: &str| n.json_field(k).and_then(Value::as_f64).unwrap_or(f64::NAN);
            let (x_titles, y_titles) = (with_role("x-axis-title"), with_role("y-axis-title"));
            assert_eq!((x_titles.len(), y_titles.len()), (1, 1), "{ty} {orientation}");
            let (x_title, y_title) = (x_titles[0], y_titles[0]);
            assert_eq!(num(y_title, "rotation"), -90.0);
            for tick in with_role("x-label") {
                assert!(num(x_title, "y") >= num(tick, "y") + num(tick, "height"), "{ty} {orientation}: x title under the ticks");
            }
            let y_ticks = with_role("y-label");
            assert!(!y_ticks.is_empty());
            for tick in y_ticks {
                assert!(num(tick, "x") >= num(y_title, "x") + num(y_title, "height"), "{ty} {orientation}: y title left of the ticks");
            }
        }
    }

    #[test]
    fn empty_cells_break_lines_and_areas_like_the_ts() {
        let config: Value = crate::value::from_str(r##"{"fontFamily":"Inter","fontSize":12,"labelColor":"#666666","titleColor":"#1a1a1a","subtitleColor":"#666666","titleSize":16,"titleWeight":600,"titleAlign":"left","subtitleSize":12,"palette":["#3b82f6","#ef4444","#22c55e","#f59e0b"],"background":"transparent","showGrid":true,"gridColor":"#e5e5e5","gridStyle":"solid","gridWidth":1,"showXAxis":true,"showYAxis":true,"axisColor":"#cccccc","axisWidth":1,"showXLabels":true,"showYLabels":true,"yTickCount":5,"yFormat":"auto","yScale":"linear","barRadius":0,"barGap":0.2,"barGroupGap":0.1,"showLabels":false,"labelPosition":"top","labelFormat":"auto","valueFormat":"auto","valueDensity":"auto","valueLabelColor":"#666666","valueLabelSize":11,"showLegend":"auto","legendPosition":"bottom","legendAlign":"center","legendGap":8,"legendMarker":"square","legendMarkerSize":12,"legendColor":"#666666","strokeWidth":2,"strokeStyle":"solid","curve":"linear","strokeCap":"round","strokeJoin":"round","showPoints":false,"pointSize":6,"pointShape":"circle","pointFill":"","pointStroke":"","pointStrokeWidth":0,"fillOpacity":0.3,"baseline":0,"pointOpacity":0.7,"minPointSize":8,"maxPointSize":40,"xTickCount":5,"innerRadius":0,"startAngle":-90,"direction":"cw","padAngle":0,"explode":[],"explodeDistance":10,"explodeAll":false,"centerLabel":"","centerSublabel":"","centerColor":"#1a1a1a","centerLabelColor":"#1a1a1a","centerSublabelColor":"#666666","centerSize":24,"pieLabelPosition":"outside","pieLabelFormat":"percent","minLabelAngle":10,"padding":16,"cornerRadius":0,"stroke":"#000000"}"##).unwrap();
        let data: Value = crate::value::from_str(r##"{"columns":["category","series","value"],"rows":[{"category":"Q1","value":100,"series":"A"},{"category":"Q1","value":105,"series":"B"},{"category":"Q2","value":100,"series":"A"},{"category":"Q2","value":0,"series":"B","missing":true},{"category":"Q3","value":100,"series":"A"},{"category":"Q3","value":110,"series":"B"},{"category":"Q4","value":100,"series":"A"},{"category":"Q4","value":0,"series":"B","missing":true}],"series":["A","B"],"categories":["Q1","Q2","Q3","Q4"],"valueRange":[0,110]}"##).unwrap();
        let b_line = [("move", 0.0, 9.083333333333314), ("move", 162.0, 0.0)];
        for (chart_type, role) in [("line", "line-series"), ("area", "area-series-line")] {
            let mut spec = json!({"type": chart_type, "x": 0, "y": 0, "width": 400, "height": 300});
            if chart_type == "line" {
                spec["mode"] = json!("grouped");
            }
            spec["config"] = config.clone();
            let out = expand_chart_generated_node(&chart_host(spec, data.clone())).unwrap();
            let series_b: Vec<_> = out
                .iter()
                .filter(|n| {
                    let meta = n.json_field("metadata");
                    meta.and_then(|m| m.get("chartRole")) == Some(&json!(role))
                        && meta.and_then(|m| m.get("chartSeries")) == Some(&json!("B"))
                })
                .collect();
            assert_eq!(series_b.len(), 1, "{chart_type}: one {role} for B");
            let node = series_b[0].to_json_map();
            let points: Vec<(String, f64, f64)> = node
                .get("points")
                .and_then(Value::as_array)
                .expect("points")
                .iter()
                .map(|p| {
                    let f = |k: &str| p.get(k).and_then(Value::as_f64).unwrap_or(f64::NAN);
                    (p["type"].as_str().unwrap_or("").to_string(), f("x"), f("y"))
                })
                .collect();
            assert_eq!(points.len(), b_line.len(), "{chart_type}: B breaks at the gaps: {points:?}");
            for ((t, x, y), (et, ex, ey)) in points.iter().zip(b_line) {
                assert!(t == et && (x - ex).abs() < 1e-6 && (y - ey).abs() < 1e-6, "{chart_type}: {points:?}");
            }
        }
    }

    /// A pie's slice labels are its value labels: an explicit `value-label-color` colours them (pieChart.ts
    /// `sliceLabelColor`); without one they keep the axis label colour.
    #[test]
    fn pie_slice_labels_take_an_explicit_value_label_color() {
        let data = json!({
            "columns": ["category", "value"],
            "rows": [{"category": "Direct", "value": 38.0}, {"category": "Wholesale", "value": 41.0}, {"category": "Online", "value": 21.0}],
            "series": [], "categories": ["Direct", "Wholesale", "Online"], "valueRange": [0.0, 41.0]
        });
        let label_html = |extra: Value| -> Vec<String> {
            let mut config = json!({"fontFamily": "Inter", "fontSize": 12, "labelColor": "#666666", "palette": ["#3b82f6", "#ef4444", "#22c55e"],
                "background": "transparent", "showLabels": true, "valueFormat": "auto", "valueDensity": "auto", "valueLabelColor": "#666666",
                "valueLabelSize": 11, "showLegend": false, "innerRadius": 0.5, "startAngle": -90, "direction": "cw", "padAngle": 0,
                "explode": [], "explodeDistance": 10, "explodeAll": false, "pieLabelPosition": "outside", "pieLabelFormat": "percent",
                "minLabelAngle": 10, "padding": 16, "cornerRadius": 0});
            for (k, v) in extra.as_object().unwrap() {
                config[k] = v.clone();
            }
            let spec = json!({"type": "pie", "x": 0, "y": 0, "width": 400, "height": 300, "config": config});
            expand_chart_generated_node(&chart_host(spec, data.clone())).unwrap()
                .iter()
                .filter(|n| n.json_field("metadata").and_then(|m| m.get("chartRole")) == Some(&json!("slice-label")))
                .map(|n| n.to_json_map().get("htmlContent").and_then(Value::as_str).unwrap_or("").to_string())
                .collect()
        };
        let plain = label_html(json!({}));
        assert_eq!(plain.len(), 3, "{plain:?}");
        assert!(plain.iter().all(|h| h.contains("#666666")), "{plain:?}");
        for position in ["outside", "inside"] {
            let coloured = label_html(json!({"valueLabelColor": "#1a1a1a", "valueLabelColorExplicit": true, "pieLabelPosition": position}));
            assert_eq!(coloured.len(), 3, "{position}: {coloured:?}");
            assert!(coloured.iter().all(|h| h.contains("#1a1a1a") && !h.contains("#666666")), "{position}: {coloured:?}");
        }
    }

    /// An explicit `value-label-size` sizes a pie's slice labels (pieChart.ts `sliceLabelSize`); without one they take
    /// `fontSize`.
    #[test]
    fn pie_slice_labels_take_an_explicit_value_label_size() {
        let data = json!({
            "columns": ["category", "value"],
            "rows": [{"category": "Direct", "value": 38.0}, {"category": "Wholesale", "value": 41.0}, {"category": "Online", "value": 21.0}],
            "series": [], "categories": ["Direct", "Wholesale", "Online"], "valueRange": [0.0, 41.0]
        });
        let label_html = |extra: Value| -> Vec<String> {
            let mut config = json!({"fontFamily": "Inter", "fontSize": 20, "labelColor": "#666666", "palette": ["#3b82f6", "#ef4444", "#22c55e"],
                "background": "transparent", "showLabels": true, "valueFormat": "auto", "valueDensity": "auto", "valueLabelColor": "#666666",
                "valueLabelSize": 11, "showLegend": false, "innerRadius": 0.5, "startAngle": -90, "direction": "cw", "padAngle": 0,
                "explode": [], "explodeDistance": 10, "explodeAll": false, "pieLabelPosition": "outside", "pieLabelFormat": "percent",
                "minLabelAngle": 10, "padding": 16, "cornerRadius": 0});
            for (k, v) in extra.as_object().unwrap() {
                config[k] = v.clone();
            }
            let spec = json!({"type": "pie", "x": 0, "y": 0, "width": 400, "height": 300, "config": config});
            expand_chart_generated_node(&chart_host(spec, data.clone())).unwrap()
                .iter()
                .filter(|n| n.json_field("metadata").and_then(|m| m.get("chartRole")) == Some(&json!("slice-label")))
                .map(|n| n.to_json_map().get("htmlContent").and_then(Value::as_str).unwrap_or("").to_string())
                .collect()
        };
        let plain = label_html(json!({}));
        assert_eq!(plain.len(), 3, "{plain:?}");
        assert!(plain.iter().all(|h| h.contains("font-size: 20px")), "{plain:?}");
        for position in ["outside", "inside"] {
            let sized = label_html(json!({"valueLabelSize": 38, "valueLabelSizeExplicit": true, "pieLabelPosition": position}));
            assert_eq!(sized.len(), 3, "{position}: {sized:?}");
            assert!(sized.iter().all(|h| h.contains("font-size: 38px")), "{position}: {sized:?}");
        }
    }

    #[test]
    fn missing_config_yields_placeholder() {
        // The config-less-chart silent-blank bug: a ported type with REAL
        // ParsedData but NO `spec.config`. In the browser every ported
        // generator dereferences `spec.config` (barChart.ts:71/131), so a
        // missing config THROWS and `expandChartGeneratedNode` catches it and
        // renders the visible placeholder (chartExpansion.ts:68-76). The NaN
        // degradation this crate uses for missing config FIELDS must not
        // apply to a missing config OBJECT — that used to emit NaN-geometry
        // definitions that rendered as a silent ~1px dot.
        let n = chart_host(
            json!({"type": "bar", "x": 0, "y": 0, "width": 400, "height": 300}),
            json!({
                "columns": ["category", "value"],
                "rows": [
                    {"category": "Q1", "value": 120.0},
                    {"category": "Q2", "value": 180.0}
                ],
                "series": [], "categories": ["Q1", "Q2"], "valueRange": [0.0, 180.0]
            }),
        );
        let out = expand_chart_generated_node(&n).unwrap();
        assert_eq!(
            out.len(),
            1,
            "config-less chart must render the placeholder, got {} nodes",
            out.len()
        );
        assert_eq!(out[0].id(), "ch1_gen_proxy_0_placeholder");
        // Full host size — visible, locatable, exactly what the browser draws.
        assert_eq!(out[0].f64("width"), Some(400.0));
        assert_eq!(out[0].f64("height"), Some(300.0));
    }

    #[test]
    fn null_config_yields_placeholder() {
        // JS throws on property access of null exactly like undefined —
        // `config: null` is the same browser placeholder path.
        let n = chart_host(
            json!({"type": "bar", "config": null}),
            json!({
                "columns": ["category", "value"],
                "rows": [{"category": "Q1", "value": 120.0}],
                "series": [], "categories": ["Q1"], "valueRange": [0.0, 120.0]
            }),
        );
        let out = expand_chart_generated_node(&n).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].id().ends_with("_placeholder"));
    }

    #[test]
    fn unusable_config_shapes_yield_placeholder() {
        // Probed against the production TS generators (all 6 types, real
        // data): a non-object config (`5`), an empty object (`{}`), and an
        // object with a null palette all leave `config.palette` reading
        // undefined/null, so `resolvePaletteColor`'s `palette.length`
        // (colors.ts:11) throws → the browser renders the placeholder.
        for cfg in [
            json!(5),
            json!({}),
            json!({"palette": null, "fontSize": 12}),
        ] {
            let n = chart_host(
                json!({"type": "bar", "config": cfg}),
                json!({
                    "columns": ["category", "value"],
                    "rows": [{"category": "Q1", "value": 120.0}],
                    "series": [], "categories": ["Q1"], "valueRange": [0.0, 120.0]
                }),
            );
            let out = expand_chart_generated_node(&n).unwrap();
            assert_eq!(out.len(), 1, "config {cfg} must placeholder");
            assert!(out[0].id().ends_with("_placeholder"), "config {cfg}");
        }
    }

    #[test]
    fn pie_missing_explode_yields_placeholder_but_bar_does_not() {
        // pie is the one generator with a second required ARRAY field:
        // `config.explode.includes(...)` throws when explode reads
        // undefined/null (probed: pie palette-only → placeholder in TS,
        // while bar/line/area/combo palette-only render real defs).
        let data = json!({
            "columns": ["category", "value"],
            "rows": [{"category": "Q1", "value": 120.0}, {"category": "Q2", "value": 80.0}],
            "series": [], "categories": ["Q1", "Q2"], "valueRange": [0.0, 120.0]
        });
        let pie = chart_host(
            json!({"type": "pie", "config": {"palette": ["#123456"]}}),
            data.clone(),
        );
        let out = expand_chart_generated_node(&pie).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].id().ends_with("_placeholder"));

        // bar with a palette-only config renders REAL definitions in the
        // browser (NaN degradation for the missing scalars, no throw) — the
        // gate must not overreach.
        let bar = chart_host(
            json!({"type": "bar", "config": {"palette": ["#123456"]}}),
            data,
        );
        let out = expand_chart_generated_node(&bar).unwrap();
        assert!(
            !out.iter().any(|nd| nd.id().ends_with("_placeholder")),
            "bar palette-only must keep the browser's NaN degradation, not the placeholder"
        );
    }

    fn styled_chart_host(spec: Value, data: Value, series_styles: Value) -> RawNode {
        RawNode::from_value(json!({
            "id": "ch1", "type": "generated", "generatedKind": "chart",
            "x": 40.0, "y": 40.0, "width": 400.0, "height": 300.0,
            "rotation": 0, "scale": 1,
            "renderMode": "native-children",
            "generatedSource": {
                "kind": "chart", "chartVersion": 1,
                "spec": spec, "data": data, "seriesStyles": series_styles
            }
        }))
        .unwrap()
    }

    #[test]
    fn styled_palette_less_chart_keeps_degradation_and_gates_to_browser() {
        // Probed: a bar with NO palette but a `__single__` seriesStyle fill
        // renders a REAL chart in the browser — `seriesStyle?.fill ||
        // resolvePaletteColor(...)` (barChart.ts:346) short-circuits the
        // throwing palette deref. The pre-dispatch placeholder gate must NOT
        // fire (that would out-placeholder the browser); instead the shape is
        // conditionally divergent, so the gate reason is `conditional-config`
        // (assess → Unsupported → browser lane, which is right in every
        // subcase — probed: line + fill-only styles still throws in TS
        // because line guards with `?.stroke`, not `?.fill`).
        let data = json!({
            "columns": ["category", "value"],
            "rows": [{"category": "Q1", "value": 120.0}, {"category": "Q2", "value": 80.0}],
            "series": [], "categories": ["Q1", "Q2"], "valueRange": [0.0, 120.0]
        });
        let n = styled_chart_host(
            json!({"type": "bar", "config": {"fontSize": 12}}),
            data,
            json!({"__single__": {"fill": "#ff0000"}}),
        );
        let out = expand_chart_generated_node(&n).unwrap();
        assert!(
            !out.iter().any(|nd| nd.id().ends_with("_placeholder")),
            "styled palette-less bar must not be force-placeholdered"
        );
        let src = n.object("generatedSource").unwrap();
        let d = chart_gate_reason(src).unwrap();
        assert_eq!(d.get("reason"), Some(&json!("conditional-config")));
        assert_eq!(d.get("missing"), Some(&json!("palette")));
    }

    #[test]
    fn pie_explode_all_short_circuits_missing_explode() {
        // Probed: pie with `explodeAll: true` and NO explode renders real
        // slices in the browser — `config.explodeAll || config.explode...`
        // short-circuits both throwing derefs (pieChart.ts:285/356). No gate.
        let data = json!({
            "columns": ["category", "value"],
            "rows": [{"category": "Q1", "value": 120.0}, {"category": "Q2", "value": 80.0}],
            "series": [], "categories": ["Q1", "Q2"], "valueRange": [0.0, 120.0]
        });
        let n = chart_host(
            json!({"type": "pie", "config": {"palette": ["#123456"], "explodeAll": true}}),
            data,
        );
        let out = expand_chart_generated_node(&n).unwrap();
        assert!(
            !out.iter().any(|nd| nd.id().ends_with("_placeholder")),
            "explodeAll pie must render real slices"
        );
        let src = n.object("generatedSource").unwrap();
        assert!(chart_gate_reason(src).is_none());
    }

    #[test]
    fn pie_wrong_type_explode_yields_placeholder_string_gates_to_browser() {
        // Probed: `explode: 5` throws in TS (`.includes` is not a function)
        // → placeholder; `explode: "Q1"` does NOT throw (strings have
        // `.includes`) and explodes by substring match this port does not
        // model → conditional-config (browser lane).
        let data = json!({
            "columns": ["category", "value"],
            "rows": [{"category": "Q1", "value": 120.0}, {"category": "Q2", "value": 80.0}],
            "series": [], "categories": ["Q1", "Q2"], "valueRange": [0.0, 120.0]
        });
        let bad = chart_host(
            json!({"type": "pie", "config": {"palette": ["#123456"], "explode": 5}}),
            data.clone(),
        );
        let out = expand_chart_generated_node(&bad).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].id().ends_with("_placeholder"));
        let d = chart_gate_reason(bad.object("generatedSource").unwrap()).unwrap();
        assert_eq!(d.get("reason"), Some(&json!("missing-config")));
        assert_eq!(d.get("missing"), Some(&json!("explode")));

        let stringy = chart_host(
            json!({"type": "pie", "config": {"palette": ["#123456"], "explode": "Q1"}}),
            data,
        );
        let out = expand_chart_generated_node(&stringy).unwrap();
        assert!(!out.iter().any(|nd| nd.id().ends_with("_placeholder")));
        let d = chart_gate_reason(stringy.object("generatedSource").unwrap()).unwrap();
        assert_eq!(d.get("reason"), Some(&json!("conditional-config")));
        assert_eq!(d.get("missing"), Some(&json!("explode")));
    }

    #[test]
    fn string_palette_gates_to_browser() {
        // Probed: a STRING palette renders a real chart in TS via char
        // indexing (`palette[i % len]` → "#", "f", …) — semantics this port
        // does not model (it reads an empty palette → default blue). Both
        // lanes render, differently → conditional-config (browser lane). A
        // non-string non-array palette (e.g. 5) resolves to the identical
        // '#3b82f6' default in both lanes (colors.ts:12) → no gate.
        let data = json!({
            "columns": ["category", "value"],
            "rows": [{"category": "Q1", "value": 120.0}],
            "series": [], "categories": ["Q1"], "valueRange": [0.0, 120.0]
        });
        let stringy = chart_host(
            json!({"type": "bar", "config": {"palette": "#ff0000"}}),
            data.clone(),
        );
        let out = expand_chart_generated_node(&stringy).unwrap();
        assert!(!out.iter().any(|nd| nd.id().ends_with("_placeholder")));
        let d = chart_gate_reason(stringy.object("generatedSource").unwrap()).unwrap();
        assert_eq!(d.get("reason"), Some(&json!("conditional-config")));
        assert_eq!(d.get("missing"), Some(&json!("palette")));

        let numeric = chart_host(json!({"type": "bar", "config": {"palette": 5}}), data);
        assert!(chart_gate_reason(numeric.object("generatedSource").unwrap()).is_none());
        let out = expand_chart_generated_node(&numeric).unwrap();
        assert!(!out.iter().any(|nd| nd.id().ends_with("_placeholder")));
    }

    #[test]
    fn unreached_palette_deref_keeps_generator_output() {
        // Probed: a bar with `config.background`, NO palette, and data whose
        // categories+series are EMPTY renders the REAL background node in
        // TS — the generator emits background/title chrome and early-returns
        // (barChart.ts:104-112) BEFORE the throwing palette deref. The
        // placeholder gate must not fire; the ported generator produces the
        // identical chrome. With rows present the axis reports the
        // conditional (browser-safer) class, not false placeholder parity.
        let n = chart_host(
            json!({"type": "bar", "config": {"background": "#ffeeee"}}),
            json!({
                "columns": ["category", "value"],
                "rows": [{"category": "Q1", "value": 120.0}],
                "series": [], "categories": [], "valueRange": [0.0, 120.0]
            }),
        );
        let out = expand_chart_generated_node(&n).unwrap();
        assert!(
            !out.iter().any(|nd| nd.id().ends_with("_placeholder")),
            "unreached palette deref must keep the generator's real output"
        );
        assert!(
            out.iter().any(|nd| {
                nd.object("metadata")
                    .and_then(|m| m.get("chartRole"))
                    .and_then(Value::as_str)
                    == Some("background")
            }),
            "the real background node must render, mirroring the browser"
        );
        let d = chart_gate_reason(n.object("generatedSource").unwrap()).unwrap();
        assert_eq!(d.get("reason"), Some(&json!("conditional-config")));

        // EMPTY rows: identical generator output in both lanes (background
        // node) and no axis at all — the config was never causal.
        let empty = chart_host(
            json!({"type": "bar", "config": {"background": "#ffeeee"}}),
            json!({"columns": [], "rows": [], "series": [], "categories": [], "valueRange": [0, 0]}),
        );
        let out = expand_chart_generated_node(&empty).unwrap();
        assert!(!out.iter().any(|nd| nd.id().ends_with("_placeholder")));
        assert!(chart_gate_reason(empty.object("generatedSource").unwrap()).is_none());
    }

    #[test]
    fn zero_total_pie_missing_explode_gates_but_keeps_output() {
        // Probed: a zero-total pie early-returns (after background chrome)
        // BEFORE the explode deref — no throw in TS. Not provably reached ⇒
        // conditional-config (browser lane), never a forced placeholder.
        let n = chart_host(
            json!({"type": "pie", "config": {"background": "#ffeeee", "palette": ["#123456"]}}),
            json!({
                "columns": ["category", "value"],
                "rows": [{"category": "Q1", "value": 0.0}],
                "series": [], "categories": ["Q1"], "valueRange": [0.0, 0.0]
            }),
        );
        let out = expand_chart_generated_node(&n).unwrap();
        assert!(!out.iter().any(|nd| nd.id().ends_with("_placeholder")));
        let d = chart_gate_reason(n.object("generatedSource").unwrap()).unwrap();
        assert_eq!(d.get("reason"), Some(&json!("conditional-config")));
        assert_eq!(d.get("missing"), Some(&json!("explode")));
    }

    #[test]
    fn malformed_palette_entries_gate_to_browser() {
        // Probed TS behavior for non-string palette ENTRIES with real data:
        // `[5]` → parseFill(5) throws on `.trim` → placeholder; `[null]` →
        // the `?? palette[0] ?? '#3b82f6'` chain picks blue; an array-like
        // OBJECT palette indexes real colors. This port reads ""/empty for
        // all three — divergent, browser right in each subcase →
        // conditional-config. A canonical all-string array stays ungated.
        let data = json!({
            "columns": ["category", "value"],
            "rows": [{"category": "Q1", "value": 120.0}],
            "series": [], "categories": ["Q1"], "valueRange": [0.0, 120.0]
        });
        for palette in [
            json!([5]),
            json!([null]),
            json!({"0": "#00ff00", "length": 1}),
        ] {
            let n = chart_host(
                json!({"type": "bar", "config": {"palette": palette}}),
                data.clone(),
            );
            let d = chart_gate_reason(n.object("generatedSource").unwrap())
                .unwrap_or_else(|| panic!("palette {palette} must gate"));
            assert_eq!(d.get("reason"), Some(&json!("conditional-config")));
            assert_eq!(d.get("missing"), Some(&json!("palette")));
            let out = expand_chart_generated_node(&n).unwrap();
            assert!(!out.iter().any(|nd| nd.id().ends_with("_placeholder")));
        }
    }

    #[test]
    fn gate_reason_missing_config() {
        // Ported type + missing/null config → observable `missing-config`
        // reason (assess marks it Full/non-gating: both lanes render the
        // identical placeholder). Takes precedence over raw-array-data —
        // with no config the browser placeholders regardless of data shape,
        // so gating to the browser buys zero fidelity.
        let missing = json!({"spec": {"type": "bar"},
                             "data": {"columns": [], "rows": [{"category": "A", "value": 1.0}],
                                      "series": [], "categories": ["A"], "valueRange": [0, 1]}})
        .as_object()
        .unwrap()
        .clone();
        let d = chart_gate_reason(&missing).unwrap();
        assert_eq!(d.get("reason"), Some(&json!("missing-config")));
        assert_eq!(d.get("spec_type"), Some(&json!("bar")));

        let null_cfg = json!({"spec": {"type": "bar", "config": null},
                              "data": {"columns": [], "rows": [{"category": "A", "value": 1.0}],
                                       "series": [], "categories": ["A"], "valueRange": [0, 1]}})
        .as_object()
        .unwrap()
        .clone();
        let d = chart_gate_reason(&null_cfg).unwrap();
        assert_eq!(d.get("reason"), Some(&json!("missing-config")));

        // With EMPTY data the placeholder is caused by the empty data — both
        // lanes draw it whatever the config says (the browser generator
        // throws on the missing config, native emits zero definitions; same
        // placeholder either way), so no axis fires: parity-benign, exactly
        // like the empty raw-array case.
        let missing_empty = json!({"spec": {"type": "bar"},
                                   "data": {"columns": [], "rows": [], "series": [],
                                            "categories": [], "valueRange": [0, 0]}})
        .as_object()
        .unwrap()
        .clone();
        assert!(chart_gate_reason(&missing_empty).is_none());
        let missing_no_data = json!({"spec": {"type": "bar"}})
            .as_object()
            .unwrap()
            .clone();
        assert!(chart_gate_reason(&missing_no_data).is_none());

        // raw-array-data outranks missing-config: any raw array still
        // present post-normalization is one the TS HYDRATION coercion throws
        // on — upstream of the generator, so the generator-level placeholder
        // parity behind missing-config cannot be claimed. (This uncoercible
        // shape uses capitalized keys; a coercible array never reaches the
        // gate — scene load rewrites it into the ParsedData bundle.)
        let raw_and_missing = json!({"spec": {"type": "bar"}, "data": [{"Category": "A", "V": 1}]})
            .as_object()
            .unwrap()
            .clone();
        let d = chart_gate_reason(&raw_and_missing).unwrap();
        assert_eq!(d.get("reason"), Some(&json!("raw-array-data")));

        // unregistered-type still outranks missing-config (no generator in
        // either lane — identical placeholder, existing precedent).
        let radar = json!({"spec": {"type": "radar"}, "data": {}})
            .as_object()
            .unwrap()
            .clone();
        let d = chart_gate_reason(&radar).unwrap();
        assert_eq!(d.get("reason"), Some(&json!("unregistered-type")));

        // an EMPTY config object with real rows reports missing-config too:
        // `config.palette` reads undefined and the browser generator throws
        // at `palette.length` (colors.ts:11) → placeholder in both lanes.
        let empty_cfg = json!({"spec": {"type": "bar", "config": {}},
                               "data": {"columns": ["category", "value"],
                                        "rows": [{"category": "A", "value": 1.0}],
                                        "series": [], "categories": ["A"], "valueRange": [0, 1]}})
        .as_object()
        .unwrap()
        .clone();
        let d = chart_gate_reason(&empty_cfg).unwrap();
        assert_eq!(d.get("reason"), Some(&json!("missing-config")));
        assert_eq!(d.get("missing"), Some(&json!("palette")));

        // pie's second required array field: palette present, explode absent.
        let pie_no_explode = json!({"spec": {"type": "pie", "config": {"palette": ["#123456"]}},
                                    "data": {"columns": ["category", "value"],
                                             "rows": [{"category": "A", "value": 1.0}],
                                             "series": [], "categories": ["A"], "valueRange": [0, 1]}})
        .as_object()
        .unwrap()
        .clone();
        let d = chart_gate_reason(&pie_no_explode).unwrap();
        assert_eq!(d.get("reason"), Some(&json!("missing-config")));
        assert_eq!(d.get("missing"), Some(&json!("explode")));

        // a generator-safe config (palette present; non-pie) never reports
        // missing-config: missing SCALAR fields NaN-degrade in both lanes.
        let safe_cfg = json!({"spec": {"type": "bar", "config": {"palette": ["#123456"]}},
                              "data": {"columns": ["category", "value"],
                                       "rows": [{"category": "A", "value": 1.0}],
                                       "series": [], "categories": ["A"], "valueRange": [0, 1]}})
        .as_object()
        .unwrap()
        .clone();
        assert!(chart_gate_reason(&safe_cfg).is_none());
    }

    #[test]
    fn empty_data_yields_placeholder() {
        // chartExpansion.ts:81-83 — zero definitions → placeholder
        let n = chart_host(
            json!({"type": "bar"}),
            json!({"columns": [], "rows": [], "series": [], "categories": [], "valueRange": [0, 0]}),
        );
        let out = expand_chart_generated_node(&n).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id(), "ch1_gen_proxy_0_placeholder");
        let fill = crate::paint::top_fill(&out[0]).map(|f| f.paint);
        assert_eq!(
            fill.as_ref().and_then(crate::paint::solid_color),
            Some("#f5f5f5")
        );
    }

    #[test]
    fn unknown_type_yields_placeholder() {
        // A type with no ported generator (and not registered in production)
        // still falls back to the placeholder rect.
        let n = chart_host(
            json!({"type": "radar", "config": {}}),
            json!({"columns": ["category", "value"],
                   "rows": [{"category": "a", "value": 1.0}],
                   "series": [], "categories": ["a"], "valueRange": [0, 1]}),
        );
        let out = expand_chart_generated_node(&n).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].id().ends_with("_placeholder"));
    }

    #[test]
    fn combo_expands_to_real_nodes() {
        // Combo is ported (M4-followup): a populated combo spec expands to
        // chart children, not the placeholder rect.
        let n = chart_host(
            json!({"type": "combo", "config": {"palette": ["#3a86ff", "#fb5607"]}}),
            json!({"columns": ["category", "series", "value"],
                   "rows": [
                       {"category": "Q1", "series": "Revenue", "value": 120.0},
                       {"category": "Q2", "series": "Revenue", "value": 180.0},
                       {"category": "Q1", "series": "Margin", "value": 30.0},
                       {"category": "Q2", "series": "Margin", "value": 45.0}
                   ],
                   "series": ["Revenue", "Margin"], "categories": ["Q1", "Q2"],
                   "valueRange": [0, 180]}),
        );
        let out = expand_chart_generated_node(&n).unwrap();
        assert!(out.len() > 1, "combo must expand, got {} nodes", out.len());
        assert!(
            !out[0].id().ends_with("_placeholder"),
            "combo must not fall back to placeholder"
        );
    }

    /// A reasonably complete chart config so the axes/labels actually render
    /// in the combo tests below.
    fn combo_test_config(y_scale: &str) -> Value {
        json!({
            "fontFamily": "Inter", "fontSize": 12, "labelColor": "#666666",
            "palette": ["#3b82f6", "#ef4444", "#06a77d"],
            "padding": 16, "yTickCount": 5, "xTickCount": 6, "yScale": y_scale, "yFormat": "auto",
            "showGrid": false, "gridColor": "#e5e5e5", "gridWidth": 1, "gridStyle": "solid",
            "showXAxis": true, "showYAxis": true, "axisColor": "#cccccc", "axisWidth": 1,
            "showXLabels": true, "showYLabels": true,
            "barGap": 0.2, "barGroupGap": 0.1, "barRadius": 0,
            "strokeWidth": 2, "strokeStyle": "solid", "strokeCap": "round", "strokeJoin": "round",
            "curve": "linear", "fillOpacity": 0.3, "baseline": 0, "showPoints": false,
            "showLegend": "auto", "legendPosition": "bottom", "legendAlign": "center",
            "legendGap": 8, "legendMarker": "square", "legendMarkerSize": 12, "legendColor": "#666666"
        })
    }

    fn node_role(n: &RawNode) -> Option<&str> {
        n.object("metadata")
            .and_then(|m| m.get("chartRole"))
            .and_then(Value::as_str)
    }

    #[test]
    fn headings_clear_plot_and_tick_and_value_labels() {
        for chart_type in ["bar", "line", "area", "scatter", "combo"] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                "tests/fixtures/chart-{chart_type}.json"
            ));
            let fixture: Value =
                crate::value::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let source = &fixture["nodes"][0]["generatedSource"];
            for size in [12.0, 24.0, 40.0] {
                for (title, subtitle) in [(true, true), (true, false), (false, true)] {
                    let mut spec = source["spec"].clone();
                    let config = spec["config"].as_object_mut().unwrap();
                    config.remove("title");
                    config.remove("subtitle");
                    if title {
                        config.insert("title".into(), json!("Heading"));
                    }
                    if subtitle {
                        config.insert("subtitle".into(), json!("Details"));
                    }
                    config.insert("fontSize".into(), json!(size));
                    config.insert("titleSize".into(), json!(size * 1.5));
                    config.insert("subtitleSize".into(), json!(size));
                    config.insert("valueLabelSize".into(), json!(size));
                    config.insert("showLabels".into(), json!(true));
                    let empty = Map::new();
                    let spec_view = SpecView::new(spec.as_object().unwrap(), &empty, 800.0, 600.0);
                    let data_view = DataView::new(source["data"].as_object().unwrap());
                    let layout =
                        layout::calculate_chart_layout(&spec_view, &data_view).unwrap();
                    let mut host = chart_host(spec, source["data"].clone());
                    host.set("width", json!(800));
                    host.set("height", json!(600));
                    let nodes = expand_chart_generated_node(&host).unwrap();
                    let headings: Vec<_> = nodes
                        .iter()
                        .filter(|node| matches!(node_role(node), Some("title" | "subtitle")))
                        .collect();
                    assert_eq!(headings.len(), usize::from(title) + usize::from(subtitle));
                    let labels: Vec<_> = nodes
                        .iter()
                        .filter(|node| {
                            matches!(
                                node_role(node),
                                Some(
                                    "x-label"
                                        | "y-label"
                                        | "y-label-left"
                                        | "y-label-right"
                                        | "value-label"
                                )
                            )
                        })
                        .collect();
                    assert!(!labels.is_empty());
                    for heading in headings {
                        let bottom = heading.f64("y").unwrap() + heading.f64("height").unwrap();
                        assert!(
                            bottom < layout.plot_area.y,
                            "{chart_type} {size}: heading overlaps plot"
                        );
                        for label in &labels {
                            assert!(
                                bottom < label.f64("y").unwrap(),
                                "{chart_type} {size}: heading overlaps {:?}",
                                node_role(label)
                            );
                        }
                    }
                }
            }
        }
    }

    /// 2024 Jan/Feb/Mar (15th) as ms since epoch, the X-time fixtures below use.
    const T_JAN: f64 = 1_705_276_800_000.0;
    const T_FEB: f64 = 1_707_955_200_000.0;
    const T_MAR: f64 = 1_710_460_800_000.0;

    fn time_line_data() -> Value {
        json!({
            "columns": ["category", "value"],
            "rows": [
                {"category": "2024-01-15", "value": 120.0, "xTime": T_JAN},
                {"category": "2024-02-15", "value": 80.0, "xTime": T_FEB},
                {"category": "2024-03-15", "value": 150.0, "xTime": T_MAR}
            ],
            "series": [],
            "categories": ["2024-01-15", "2024-02-15", "2024-03-15"],
            "valueRange": [0.0, 150.0],
            "timeRange": [T_JAN, T_MAR]
        })
    }

    fn x_labels(out: &[RawNode]) -> Vec<&RawNode> {
        out.iter()
            .filter(|n| node_role(n) == Some("x-label"))
            .collect()
    }

    /// A time X axis must emit numeric (ms) ticks with multi-scale date labels,
    /// not categorical band labels.
    #[test]
    fn line_time_axis_emits_date_labels() {
        let n = chart_host(
            json!({"type": "line", "xScale": "time", "config": combo_test_config("linear")}),
            time_line_data(),
        );
        let out = expand_chart_generated_node(&n).unwrap();
        assert!(out.len() > 1, "time line must expand, not placeholder");

        // The line series renders as a path (chronological).
        assert!(out.iter().any(|nd| node_role(nd) == Some("line-series")));

        let labels = x_labels(&out);
        assert!(!labels.is_empty(), "time axis must emit x labels");
        // chartValue is an ms instant (large number), not a category string.
        assert!(
            labels.iter().all(|nd| nd
                .object("metadata")
                .and_then(|m| m.get("chartValue"))
                .and_then(Value::as_f64)
                .map(|v| v > 1.0e12)
                .unwrap_or(false)),
            "x labels carry ms chartValue on a time axis"
        );
        // The label text is a multi-scale date (week-style "%b %d" here), not
        // the raw "2024-01-15" category. node + d3: display range nices to
        // [2024-01-14, 2024-03-17] with weekly ticks "Jan 14" … "Mar 17".
        let text: String = labels
            .iter()
            .filter_map(|nd| nd.str("htmlContent"))
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            !text.contains("2024-01-15"),
            "must not fall back to categorical ISO labels, got: {text}"
        );
        assert!(
            text.contains("Jan 14") && text.contains("Mar 17"),
            "multi-scale week labels expected (Jan 14 … Mar 17), got: {text}"
        );
    }

    /// An explicit `xTimeFormat` overrides the multi-scale formatter on every
    /// tick (here every label is `YYYY-MM`).
    #[test]
    fn line_time_axis_honors_x_time_format() {
        let n = chart_host(
            json!({"type": "line", "xScale": "time", "xTimeFormat": "%Y-%m",
                   "config": combo_test_config("linear")}),
            time_line_data(),
        );
        let out = expand_chart_generated_node(&n).unwrap();
        let labels = x_labels(&out);
        assert!(!labels.is_empty());
        for nd in &labels {
            let html = nd.str("htmlContent").unwrap_or("");
            assert!(
                html.contains("2024-0") || html.contains("2023-1"),
                "every label uses the %Y-%m specifier, got: {html}"
            );
        }
    }

    /// Combo + time + bar series falls back to categorical X (bars need bands),
    /// matching the TS; line-only combo + time uses the real time axis.
    #[test]
    fn combo_time_bar_series_falls_back_to_categorical() {
        let bar_time = chart_host(
            json!({"type": "combo", "xScale": "time", "config": combo_test_config("linear")}),
            json!({
                "columns": ["category", "series", "value"],
                "rows": [
                    {"category": "2024-01-15", "series": "A", "value": 10.0, "xTime": T_JAN},
                    {"category": "2024-02-15", "series": "A", "value": 20.0, "xTime": T_FEB}
                ],
                "series": ["A"], "categories": ["2024-01-15", "2024-02-15"],
                "valueRange": [0.0, 20.0], "timeRange": [T_JAN, T_FEB]
            }),
        );
        // Series A defaults to a bar chartType → categorical fallback: the X
        // labels are the raw category strings, not ms.
        let out = expand_chart_generated_node(&bar_time).unwrap();
        let labels = x_labels(&out);
        assert!(labels
            .iter()
            .any(|nd| nd.str("htmlContent").unwrap_or("").contains("2024-01-15")));
    }

    #[test]
    fn combo_dual_axis_remaps_spines_and_tags_yaxis() {
        // Exercises the combo-specific paths no other chart type has: dual-axis
        // layout, the remap_axis_roles spine→rectangle conversion, per-series
        // `chartYAxis` metadata, the right-axis y-label rewrite, and the
        // left/right axis title labels. Mix bar (left) + line (right) + area
        // (left).
        let host = RawNode::from_value(json!({
            "id": "ch1", "type": "generated", "generatedKind": "chart",
            "x": 40.0, "y": 40.0, "width": 480.0, "height": 320.0, "rotation": 0, "scale": 1,
            "renderMode": "native-children",
            "generatedSource": {
                "kind": "chart", "chartVersion": 1,
                "spec": {"type": "combo", "config": combo_test_config("linear"),
                         "dualAxis": true, "leftAxisLabel": "Rev", "rightAxisLabel": "Pct"},
                "data": {
                    "columns": ["category", "series", "value"],
                    "rows": [
                        {"category": "Q1", "series": "Rev", "value": 120.0},
                        {"category": "Q2", "series": "Rev", "value": 180.0},
                        {"category": "Q1", "series": "Growth", "value": 12.0},
                        {"category": "Q2", "series": "Growth", "value": 30.0},
                        {"category": "Q1", "series": "Base", "value": 40.0},
                        {"category": "Q2", "series": "Base", "value": 50.0}
                    ],
                    "series": ["Rev", "Growth", "Base"],
                    "categories": ["Q1", "Q2"], "valueRange": [0, 180]
                },
                "seriesStyles": {
                    "Rev": {"chartType": "bar", "yAxis": "left"},
                    "Growth": {"chartType": "line", "yAxis": "right"},
                    "Base": {"chartType": "area", "yAxis": "left"}
                }
            }
        }))
        .unwrap();
        let out = expand_chart_generated_node(&host).unwrap();
        let by_role = |r: &str| out.iter().find(|n| node_role(n) == Some(r));
        let yaxis_of = |n: &RawNode| {
            n.object("metadata")
                .and_then(|m| m.get("chartYAxis"))
                .and_then(Value::as_str)
                .map(str::to_string)
        };

        // remap_axis_roles converts BOTH Y-axis spines from line → rectangle.
        assert_eq!(
            by_role("axis-y-left").expect("left spine").node_type(),
            "rectangle"
        );
        assert_eq!(
            by_role("axis-y-right").expect("right spine").node_type(),
            "rectangle"
        );

        // Right and left axis labels are present under the remapped roles.
        assert!(by_role("y-label-left").is_some());
        assert!(by_role("y-label-right").is_some());

        // Per-series chartYAxis metadata splits the two axes.
        assert_eq!(
            yaxis_of(by_role("bar").expect("bar")).as_deref(),
            Some("left")
        );
        assert_eq!(
            yaxis_of(by_role("line-series").expect("line")).as_deref(),
            Some("right")
        );
        assert_eq!(
            yaxis_of(by_role("area-series").expect("area")).as_deref(),
            Some("left")
        );

        // Dual-axis title labels.
        assert!(by_role("axis-label-left").is_some());
        assert!(by_role("axis-label-right").is_some());
    }

    #[test]
    fn combo_log_scale_expands_to_bars() {
        // yScale=log with a strictly-positive domain: bars anchor to the
        // display-range floor (not config.baseline). Must expand, not panic.
        let n = chart_host(
            json!({"type": "combo", "config": combo_test_config("log")}),
            json!({
                "columns": ["category", "series", "value"],
                "rows": [
                    {"category": "Q1", "series": "A", "value": 10.0},
                    {"category": "Q2", "series": "A", "value": 1000.0}
                ],
                "series": ["A"], "categories": ["Q1", "Q2"], "valueRange": [0, 1000]
            }),
        );
        let out = expand_chart_generated_node(&n).unwrap();
        assert!(
            out.iter().any(|node| node_role(node) == Some("bar")),
            "log-scale combo must emit bars"
        );
    }

    #[test]
    fn chart_gated_fixture_spec_yields_placeholder() {
        // The seeded `generated/chart-gated` fixture: malformed ParsedData
        // (rows are arrays, config nearly empty). The browser generator
        // returns zero definitions for it (NaN layout + unkeyable rows), so
        // the placeholder must be emitted — pixel-equal to the golden.
        let n = chart_host(
            json!({"type": "bar", "config": {"palette": ["#3a86ff", "#fb5607", "#06a77d"], "showLegend": true}}),
            json!({
                "columns": [{"name": "quarter", "type": "string"}, {"name": "revenue", "type": "number"}],
                "rows": [["Q1", 120], ["Q2", 180], ["Q3", 150], ["Q4", 210]],
                "series": ["revenue"],
                "categories": ["Q1", "Q2", "Q3", "Q4"],
                "valueRange": {"min": 0, "max": 210}
            }),
        );
        let out = expand_chart_generated_node(&n).unwrap();
        assert_eq!(
            out.len(),
            1,
            "must fall back to placeholder, got {} nodes",
            out.len()
        );
        assert_eq!(out[0].id(), "ch1_gen_proxy_0_placeholder");
    }

    #[test]
    fn simple_bar_chart_expands() {
        let config: Value = crate::value::from_str(
            r##"{
            "fontFamily": "Inter", "fontSize": 12, "labelColor": "#666666",
            "titleColor": "#1a1a1a", "subtitleColor": "#666666",
            "titleSize": 16, "titleWeight": 600, "titleAlign": "left", "subtitleSize": 12,
            "palette": ["#3b82f6", "#ef4444"], "background": "transparent",
            "showGrid": true, "gridColor": "#e5e5e5", "gridStyle": "solid", "gridWidth": 1,
            "showXAxis": true, "showYAxis": true, "axisColor": "#cccccc", "axisWidth": 1,
            "showXLabels": true, "showYLabels": true, "yTickCount": 5,
            "yFormat": "auto", "yScale": "linear",
            "barRadius": 0, "barGap": 0.2, "barGroupGap": 0.1,
            "showLabels": false, "labelPosition": "top", "labelFormat": "auto",
            "valueFormat": "auto", "valueDensity": "auto",
            "valueLabelColor": "#666666", "valueLabelSize": 11,
            "showLegend": "auto", "legendPosition": "bottom", "legendAlign": "center",
            "legendGap": 8, "legendMarker": "square", "legendMarkerSize": 12,
            "legendColor": "#666666",
            "strokeWidth": 2, "strokeStyle": "solid", "curve": "linear",
            "strokeCap": "round", "strokeJoin": "round",
            "showPoints": false, "pointSize": 8, "pointShape": "circle",
            "pointFill": "", "pointStroke": "", "pointStrokeWidth": 0,
            "fillOpacity": 0.3, "baseline": 0,
            "pointOpacity": 0.7, "minPointSize": 8, "maxPointSize": 40, "xTickCount": 5,
            "innerRadius": 0, "startAngle": -90, "direction": "cw", "padAngle": 0,
            "explode": [], "explodeDistance": 10, "explodeAll": false,
            "centerLabel": "", "centerSublabel": "", "centerColor": "#1a1a1a",
            "centerLabelColor": "#1a1a1a", "centerSublabelColor": "#666666", "centerSize": 24,
            "pieLabelPosition": "outside", "pieLabelFormat": "percent", "minLabelAngle": 10,
            "padding": 16, "cornerRadius": 0
        }"##,
        )
        .unwrap();
        let n = chart_host(
            json!({"type": "bar", "x": 0, "y": 0, "width": 400, "height": 300, "config": config}),
            json!({
                "columns": ["category", "value"],
                "rows": [
                    {"category": "Q1", "value": 120.0},
                    {"category": "Q2", "value": 180.0},
                    {"category": "Q3", "value": 150.0}
                ],
                "series": [], "categories": ["Q1", "Q2", "Q3"], "valueRange": [0.0, 180.0]
            }),
        );
        let out = expand_chart_generated_node(&n).unwrap();
        // grid(6 ticks 0..250? domain [0,180]→nice[0,200]: 0,50,100,150,200 = 5)
        // + 3 bars + y axis(line+5 ticks+5 labels) + x axis(line+3 ticks+3 labels)
        let bars: Vec<&RawNode> = out
            .iter()
            .filter(|n| {
                n.object("metadata")
                    .and_then(|m| m.get("chartRole"))
                    .and_then(Value::as_str)
                    == Some("bar")
            })
            .collect();
        assert_eq!(bars.len(), 3);
        assert_eq!(bars[0].node_type(), "rectangle");
        // ids are sequential transient proxies
        assert!(out[0].id().starts_with("ch1_gen_proxy_0"));
        // grid lines present
        assert!(out.iter().any(|n| {
            n.object("metadata")
                .and_then(|m| m.get("chartRole"))
                .and_then(Value::as_str)
                == Some("grid-line")
        }));
        // axis labels are richtext with Inter html
        let label = out
            .iter()
            .find(|n| {
                n.object("metadata")
                    .and_then(|m| m.get("chartRole"))
                    .and_then(Value::as_str)
                    == Some("x-label")
            })
            .expect("x labels");
        assert!(label
            .str("htmlContent")
            .unwrap()
            .contains("font-size: 12px; font-family: Inter; text-align: center; color: #666666"));
    }


    #[test]
    fn gate_reason() {
        let ported = json!({"spec": {"type": "bar"}})
            .as_object()
            .unwrap()
            .clone();
        assert!(chart_gate_reason(&ported).is_none());
        // combo is ported now → not gated by type.
        let combo = json!({"spec": {"type": "combo"}})
            .as_object()
            .unwrap()
            .clone();
        assert!(chart_gate_reason(&combo).is_none());
        // an unported/unregistered type reports with spec_type + reason detail.
        let radar = json!({"spec": {"type": "radar"}})
            .as_object()
            .unwrap()
            .clone();
        let d = chart_gate_reason(&radar).unwrap();
        assert_eq!(d.get("spec_type"), Some(&json!("radar")));
        assert_eq!(d.get("reason"), Some(&json!("unregistered-type")));
        // a PORTED type with a non-empty raw array that normalization could
        // NOT coerce (capitalized keys → TS rowsToParsedChartData throws):
        // fail-closed gating raw-array-data reason. Coercible arrays never
        // reach the gate — scene load rewrites them into the ParsedData
        // bundle (see normalize.rs). An empty array is benign. Fires even
        // with a missing config: the TS failure is a HYDRATION throw,
        // upstream of the generator, so raw-array-data outranks the config
        // classification (see `gate_reason_missing_config`).
        let raw_rows = json!({"spec": {"type": "bar"}, "data": [{"Category": "A", "V": 1}]})
            .as_object()
            .unwrap()
            .clone();
        let d = chart_gate_reason(&raw_rows).unwrap();
        assert_eq!(d.get("reason"), Some(&json!("raw-array-data")));
        assert_eq!(d.get("spec_type"), Some(&json!("bar")));
        let empty_rows = json!({"spec": {"type": "bar"}, "data": []})
            .as_object()
            .unwrap()
            .clone();
        assert!(chart_gate_reason(&empty_rows).is_none());
        // time X axis is ported now → no longer gated (line / area / scatter /
        // combo). Combo + time falls back to categorical inside the generator
        // when bar series are present, but still renders (ungated).
        for ty in ["line", "area", "scatter", "combo"] {
            let time = json!({"spec": {"type": ty, "xScale": "time"}})
                .as_object()
                .unwrap()
                .clone();
            assert!(
                chart_gate_reason(&time).is_none(),
                "{ty} + time must not gate"
            );
        }
    }

    /// A stacked bar chart's value axis gutter is sized for the ticks it draws: a 7.5 + 5.3 + 2.1
    /// stack draws "20" ticks, which a gutter sized from the rows' largest value (7.5, "8") wrapped
    /// a digit per line. The layout takes the generator's own domain (`bar_value_domain`), as the TS
    /// does (packages/editor/test/chartStackedAxisGutter.test.ts).
    #[test]
    fn stacked_bars_reserve_the_gutter_of_their_totals() {
        let config = {
            let mut c = crate::defaults::generated::default_chart_config().clone();
            c["fontSize"] = json!(24);
            c
        };
        let plot_left = |data: Value, mode: Option<&str>| {
            let mut spec = json!({"type": "bar", "x": 0, "y": 0, "width": 960, "height": 820, "config": config});
            if let Some(mode) = mode {
                spec["mode"] = json!(mode);
            }
            chart_definitions(&chart_host(spec, data)).unwrap()
                .expect("chart definitions")
                .iter()
                .filter(|d| d.props["metadata"]["chartRole"].as_str() == Some("bar"))
                .map(|d| d.props["x"].as_f64().unwrap())
                .fold(f64::INFINITY, f64::min)
        };
        let row = |c: &str, s: &str, v: f64| json!({"category": c, "series": s, "value": v});
        let stacks = json!({
            "columns": ["category", "series", "value"],
            "rows": [row("Q1", "A", 6.1), row("Q1", "B", 4.0), row("Q1", "C", 1.2),
                     row("Q2", "A", 7.5), row("Q2", "B", 5.3), row("Q2", "C", 2.1)],
            "series": ["A", "B", "C"], "categories": ["Q1", "Q2"], "valueRange": [1.2, 7.5]
        });
        let single = json!({
            "columns": ["category", "value"],
            "rows": [{"category": "Q1", "value": 11.3}, {"category": "Q2", "value": 14.9}],
            "series": [], "categories": ["Q1", "Q2"], "valueRange": [11.3, 14.9]
        });
        let stacked = plot_left(stacks.clone(), Some("stacked"));
        assert_eq!(stacked, plot_left(single, None), "ticks to 20: two digits wide");
        assert!(stacked > plot_left(stacks.clone(), Some("grouped")));
        assert!(plot_left(stacks, Some("stacked-percent")) > stacked, "\"100%\" is wider than \"20\"");
    }

    /// #2116: repeated category labels merged into one position (a 12-month
    /// axis labelled J F M A M J J A S O N D drew 8 points). Every row is its
    /// own position and the axis draws the label as written.
    #[test]
    fn repeated_category_labels_are_positions() {
        let months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"];
        let rows: Vec<Value> = months
            .iter()
            .enumerate()
            .map(|(i, m)| json!({"category": m, "value": (i + 1) as f64}))
            .collect();
        let data = json!({
            "columns": ["category", "value"], "rows": rows, "series": [],
            "categories": ["J", "F", "M", "A", "S", "O", "N", "D"], "valueRange": [0.0, 12.0]
        });
        let mut config = crate::defaults::generated::default_chart_config().clone();
        config["showPoints"] = json!(true);
        let defs = |ty: &str| {
            let spec = json!({"type": ty, "x": 0, "y": 0, "width": 400, "height": 300, "config": config});
            chart_definitions(&chart_host(spec, data.clone())).unwrap().expect("chart definitions")
        };
        let role = |d: &Def| {
            d.props
                .get("metadata")
                .and_then(|m| m.get("chartRole"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        };
        let label_text = |d: &Def| {
            let html = d.props["htmlContent"].as_str().unwrap_or("");
            let mut text = String::new();
            let mut in_tag = false;
            for c in html.chars() {
                match c {
                    '<' => in_tag = true,
                    '>' => in_tag = false,
                    c if !in_tag => text.push(c),
                    _ => {}
                }
            }
            text
        };

        let line = defs("line");
        let points: Vec<&Def> = line.iter().filter(|d| role(d) == "point-marker").collect();
        assert_eq!(points.len(), 12, "one point per row");
        let values: Vec<f64> = points
            .iter()
            .map(|d| d.props["metadata"]["chartValue"].as_f64().unwrap())
            .collect();
        assert_eq!(values, (1..=12).map(f64::from).collect::<Vec<_>>());
        let xs: Vec<f64> = points.iter().map(|d| d.props["x"].as_f64().unwrap()).collect();
        assert!(xs.windows(2).all(|w| w[1] > w[0]), "left to right: {xs:?}");
        let labels: Vec<String> = line
            .iter()
            .filter(|d| role(d) == "x-label")
            .map(label_text)
            .collect();
        assert_eq!(labels, months.map(String::from).to_vec());

        let bar = defs("bar");
        let bars: Vec<&Def> = bar.iter().filter(|d| role(d) == "bar").collect();
        assert_eq!(bars.len(), 12, "a repeated label is its own bar, not a sum");

        // point-colors keys by the label, so every repeat of it takes the colour.
        let spec = json!({"type": "bar", "x": 0, "y": 0, "width": 400, "height": 300,
            "pointColors": {"M": "#e0301e"}, "config": config});
        let red: Vec<String> = chart_definitions(&chart_host(spec, data.clone())).unwrap()
            .expect("chart definitions")
            .iter()
            .filter(|d| role(d) == "bar" && Value::Object(d.props.clone()).to_string().contains("#e0301e"))
            .map(|d| d.props["metadata"]["chartCategory"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(red, ["M", "M (2)"]);
    }

    /// #2114: a combo with no `<series chart-type>` tags drew every series as
    /// grouped bars, exactly like `type="bar"`. It now draws the first series
    /// as bars and every later one as a line; an authored type still wins.
    #[test]
    fn combo_without_series_types_is_bars_then_lines() {
        let data = json!({
            "columns": ["category", "series", "value"],
            "rows": [
                {"category": "A", "series": "Bars", "value": 10.0},
                {"category": "A", "series": "Line", "value": 4.0},
                {"category": "B", "series": "Bars", "value": 12.0},
                {"category": "B", "series": "Line", "value": 8.0},
                {"category": "C", "series": "Bars", "value": 9.0},
                {"category": "C", "series": "Line", "value": 12.0}
            ],
            "series": ["Bars", "Line"], "categories": ["A", "B", "C"], "valueRange": [0.0, 12.0]
        });
        let spec = json!({
            "type": "combo", "x": 0, "y": 0, "width": 400, "height": 300,
            "config": crate::defaults::generated::default_chart_config().clone()
        });
        let roles = |n: &RawNode| -> Vec<(String, String)> {
            chart_definitions(n).unwrap()
                .expect("chart definitions")
                .iter()
                .filter_map(|d| {
                    let m = d.props.get("metadata")?;
                    let role = m.get("chartRole")?.as_str()?;
                    let series = m.get("chartSeries")?.as_str()?;
                    (role == "bar" || role == "line-series")
                        .then(|| (role.to_string(), series.to_string()))
                })
                .collect()
        };
        let drawn = roles(&chart_host(spec.clone(), data.clone()));
        let bars: Vec<_> = drawn.iter().filter(|(r, _)| r == "bar").collect();
        assert_eq!(bars.len(), 3, "{drawn:?}");
        assert!(bars.iter().all(|(_, s)| s == "Bars"), "{drawn:?}");
        assert_eq!(
            drawn.iter().filter(|(r, _)| r == "line-series").collect::<Vec<_>>(),
            vec![&("line-series".to_string(), "Line".to_string())]
        );

        // An authored chart type wins over the default.
        let styled = styled_chart_host(spec, data, json!({"Line": {"chartType": "bar"}}));
        let drawn = roles(&styled);
        assert_eq!(drawn.iter().filter(|(r, _)| r == "bar").count(), 6, "{drawn:?}");
    }

    /// Every drawn node of a role, as (category, serialized props). Chart
    /// definitions carry the category in `metadata.chartCategory`.
    fn drawn_by_role(spec: Value, data: Value, role: &str) -> Vec<(String, String)> {
        let defs = chart_definitions(&chart_host(spec, data)).unwrap().expect("chart definitions");
        defs.iter()
            .filter(|d| {
                d.props
                    .get("metadata")
                    .and_then(|m| m.get("chartRole"))
                    .and_then(Value::as_str)
                    == Some(role)
            })
            .map(|d| {
                let category = d.props["metadata"]["chartCategory"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                (category, Value::Object(d.props.clone()).to_string())
            })
            .collect()
    }

    /// #2115: `point-colors` was honoured by the TS generators (the editor's
    /// preview) but ignored by this port (export), so a highlighted bar
    /// exported in the series colour. Vertical and horizontal bars, donut
    /// slices and line markers all take the category's colour.
    #[test]
    fn point_colors_colour_their_category() {
        let red = "#e0301e";
        let data = json!({
            "columns": ["category", "value"],
            "rows": [
                {"category": "A", "value": 10.0},
                {"category": "B", "value": 12.0},
                {"category": "C", "value": 9.0}
            ],
            "series": [], "categories": ["A", "B", "C"], "valueRange": [0.0, 12.0]
        });
        let spec = |ty: &str, extra: Value| {
            let mut spec = json!({
                "type": ty, "x": 0, "y": 0, "width": 400, "height": 300,
                "pointColors": {"B": red},
                "config": crate::defaults::generated::default_chart_config().clone()
            });
            for (k, v) in extra.as_object().unwrap() {
                if k == "showPoints" {
                    spec["config"][k] = v.clone();
                } else {
                    spec[k] = v.clone();
                }
            }
            spec
        };
        let cases = [
            ("bar", json!({}), "bar"),
            ("bar", json!({"orientation": "horizontal"}), "bar"),
            ("pie", json!({}), "slice"),
            ("line", json!({"showPoints": true}), "point-marker"),
        ];
        for (ty, extra, role) in cases {
            let drawn = drawn_by_role(spec(ty, extra.clone()), data.clone(), role);
            assert_eq!(drawn.len(), 3, "{ty} {extra}: three {role} nodes");
            for (category, props) in &drawn {
                assert_eq!(
                    props.contains(red),
                    category == "B",
                    "{ty} {extra}: {role} {category} must be red iff it is B: {props}"
                );
            }
        }
    }
}
