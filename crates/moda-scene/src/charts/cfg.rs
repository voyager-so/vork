//! JSON views over the persisted chart pieces (`ChartGeneratedSource`):
//! spec (`PersistedChartSpec`), config (`ChartConfig`), data (`ParsedData`),
//! seriesStyles. Accessors mirror JS property reads: a missing/wrong-typed
//! numeric field reads as NaN (undefined arithmetic), strings/bools as
//! Option/truthiness — so malformed persisted specs degrade exactly like the
//! TS generators do (e.g. the `generated/chart-gated` fixture).

use crate::value::{Map, Value};
use std::collections::HashMap;

use super::jsmath::{js_max, js_min};

pub type JsonMap = Map;

/// JS-ish numeric read: number → value, anything else → NaN.
pub fn num(v: Option<&Value>) -> f64 {
    match v {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN),
        _ => f64::NAN,
    }
}

/// JS truthiness for config booleans.
pub fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map(|x| x != 0.0 && !x.is_nan()).unwrap_or(false),
        Some(Value::String(s)) => !s.is_empty(),
        Some(_) => true,
    }
}

#[derive(Clone, Copy)]
pub struct ConfigView<'a>(pub &'a JsonMap);

impl<'a> ConfigView<'a> {
    pub fn get(&self, key: &str) -> Option<&'a Value> {
        self.0.get(key)
    }
    pub fn f(&self, key: &str) -> f64 {
        num(self.0.get(key))
    }
    pub fn b(&self, key: &str) -> bool {
        truthy(self.0.get(key))
    }
    pub fn s(&self, key: &str) -> Option<&'a str> {
        self.0.get(key).and_then(Value::as_str)
    }
    /// Truthy string (e.g. `config.title` checks).
    pub fn s_truthy(&self, key: &str) -> Option<&'a str> {
        self.s(key).filter(|s| !s.is_empty())
    }
    /// `config.palette` — string entries (non-strings keep their slot so
    /// palette indexing stays aligned; JS would hand the raw value through,
    /// but only string fills are renderable, so coerce to "").
    pub fn palette(&self) -> Vec<String> {
        self.0
            .get("palette")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|v| v.as_str().unwrap_or("").to_string())
                    .collect()
            })
            .unwrap_or_default()
    }
    /// `config.explode` (string list).
    pub fn explode_list(&self) -> Vec<&'a str> {
        self.0
            .get("explode")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default()
    }
    /// `config.showLegend`: true | false | 'auto' | undefined.
    pub fn show_legend_true(&self) -> bool {
        matches!(self.0.get("showLegend"), Some(Value::Bool(true)))
    }
    pub fn show_legend_auto(&self) -> bool {
        self.s("showLegend") == Some("auto")
    }
    pub fn show_legend_false(&self) -> bool {
        matches!(self.0.get("showLegend"), Some(Value::Bool(false)))
    }
}

#[derive(Clone, Copy)]
pub struct SpecView<'a> {
    pub map: &'a JsonMap,
    pub config: ConfigView<'a>,
    pub width: f64,
    pub height: f64,
}

impl<'a> SpecView<'a> {
    /// Rebuild the runtime spec view (`rebuildChartSpec`,
    /// chartExpansion.ts:178-203): host width/height override; x/y zeroed.
    pub fn new(
        spec_map: &'a JsonMap,
        empty: &'a JsonMap,
        host_w: f64,
        host_h: f64,
    ) -> SpecView<'a> {
        let config = spec_map
            .get("config")
            .and_then(Value::as_object)
            .unwrap_or(empty);
        SpecView {
            map: spec_map,
            config: ConfigView(config),
            width: host_w,
            height: host_h,
        }
    }
    pub fn ty(&self) -> &'a str {
        self.map.get("type").and_then(Value::as_str).unwrap_or("")
    }
    pub fn s(&self, key: &str) -> Option<&'a str> {
        self.map.get(key).and_then(Value::as_str)
    }
    pub fn f_opt(&self, key: &str) -> Option<f64> {
        match self.map.get(key) {
            Some(Value::Number(n)) => n.as_f64(),
            _ => None,
        }
    }
    pub fn b(&self, key: &str) -> bool {
        truthy(self.map.get(key))
    }
    pub fn series_order(&self) -> Option<Vec<String>> {
        self.map
            .get("seriesOrder")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
    }
    /// `spec.pointColors`: per-category colours, keyed `category` or
    /// `series/category`.
    pub fn point_colors(&self) -> Option<&'a JsonMap> {
        self.map.get("pointColors").and_then(Value::as_object)
    }
    /// `spec.pointColors?.[`${series}/${category}`] ?? spec.pointColors?.[category]`:
    /// a series/category key overrides the category key.
    pub fn point_color(&self, series: &str, category: &str) -> Option<&'a str> {
        point_color(self.point_colors(), series, category)
    }
    /// `spec.content?.source` → (label, url).
    pub fn source(&self) -> Option<(&'a str, Option<&'a str>)> {
        let src = self
            .map
            .get("content")
            .and_then(Value::as_object)?
            .get("source")
            .and_then(Value::as_object)?;
        let label = src.get("label").and_then(Value::as_str)?;
        Some((label, src.get("url").and_then(Value::as_str)))
    }
}

/// `pointColors?.[`${series}/${category}`] ?? pointColors?.[category]`.
pub fn point_color<'a>(
    point_colors: Option<&'a JsonMap>,
    series: &str,
    category: &str,
) -> Option<&'a str> {
    let pc = point_colors?;
    pc.get(&format!("{series}/{category}"))
        .and_then(Value::as_str)
        .or_else(|| pc.get(category).and_then(Value::as_str))
}

/// `SeriesStyle` lookups (`spec.seriesStyles` Map — persisted as a record).
#[derive(Clone, Copy)]
pub struct SeriesStyles<'a>(pub Option<&'a JsonMap>);

impl<'a> SeriesStyles<'a> {
    pub fn get(&self, key: &str) -> Option<SeriesStyleView<'a>> {
        self.0
            .and_then(|m| m.get(key))
            .and_then(Value::as_object)
            .map(SeriesStyleView)
    }
}

#[derive(Clone, Copy)]
pub struct SeriesStyleView<'a>(pub &'a JsonMap);

impl<'a> SeriesStyleView<'a> {
    pub fn s(&self, key: &str) -> Option<&'a str> {
        self.0.get(key).and_then(Value::as_str)
    }
    pub fn s_truthy(&self, key: &str) -> Option<&'a str> {
        self.s(key).filter(|s| !s.is_empty())
    }
    pub fn f_opt(&self, key: &str) -> Option<f64> {
        match self.0.get(key) {
            Some(Value::Number(n)) => n.as_f64(),
            _ => None,
        }
    }
    pub fn b_opt(&self, key: &str) -> Option<bool> {
        self.0.get(key).and_then(Value::as_bool)
    }
    pub fn has(&self, key: &str) -> bool {
        // `!== undefined` checks
        self.0.contains_key(key) && !matches!(self.0.get(key), Some(Value::Null))
    }
    pub fn value(&self, key: &str) -> Option<&'a Value> {
        self.0.get(key)
    }
}

/// `ParsedData` view (`source.data`).
pub struct DataView<'a> {
    pub map: &'a JsonMap,
    pub rows: Vec<RowView<'a>>,
    pub series: Vec<String>,
    pub categories: Vec<String>,
    pub columns: Vec<&'a str>,
}

impl<'a> DataView<'a> {
    pub fn new(map: &'a JsonMap) -> DataView<'a> {
        let rows = map
            .get("rows")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(RowView).collect())
            .unwrap_or_default();
        let series = string_list(map.get("series"));
        let categories = string_list(map.get("categories"));
        let columns = map
            .get("columns")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        DataView {
            map,
            rows,
            series,
            categories,
            columns,
        }
    }
    pub fn x_range(&self) -> Option<(f64, f64)> {
        pair(self.map.get("xRange"))
    }
    pub fn y_range(&self) -> Option<(f64, f64)> {
        pair(self.map.get("yRange"))
    }
    pub fn size_range(&self) -> Option<(f64, f64)> {
        pair(self.map.get("sizeRange"))
    }
    pub fn time_range(&self) -> Option<(f64, f64)> {
        pair(self.map.get("timeRange"))
    }
    pub fn columns_include(&self, name: &str) -> bool {
        self.columns.contains(&name)
    }
    /// `data.categoryLabels` (render-time only, set by [`positional_categories`]).
    pub fn category_labels(&self) -> Option<&'a JsonMap> {
        self.map.get("categoryLabels").and_then(Value::as_object)
    }
    /// `categoryLabel(data, key)`: the text a category key draws.
    pub fn category_label<'k>(&self, key: &'k str) -> &'k str
    where
        'a: 'k,
    {
        category_label(self.category_labels(), key)
    }
}

/// `categoryLabels?.[key] ?? key`.
pub fn category_label<'k>(labels: Option<&'k JsonMap>, key: &'k str) -> &'k str {
    labels
        .and_then(|m| m.get(key))
        .and_then(Value::as_str)
        .unwrap_or(key)
}

/// `positionalCategories` (data.ts): categories are positions, never merged
/// by label. A label's nth occurrence (n > 0) in one series gets its own key
/// (`"M (2)"`, unique against every label), shared by the nth occurrence in
/// every other series; `categoryLabels` maps each new key back to its label.
/// `None` when no label repeats (the data is drawn as stored).
pub fn positional_categories(data: &JsonMap) -> Option<JsonMap> {
    if truthy(data.get("isScatterFormat")) {
        return None;
    }
    let rows = data.get("rows")?.as_array()?;
    let by_category = data
        .get("categories")
        .and_then(Value::as_array)
        .is_some_and(|c| !c.is_empty());
    let label_of = |row: &Value| -> String {
        let field = if by_category { "category" } else { "series" };
        row.get(field)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    let mut occurrences: HashMap<(String, String), usize> = HashMap::new();
    let mut repeated = false;
    let nth: Vec<usize> = rows
        .iter()
        .map(|row| {
            let label = label_of(row);
            if label.is_empty() {
                return 0;
            }
            let bucket = if by_category {
                row.get("series")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string()
            } else {
                String::new()
            };
            let n = occurrences.entry((bucket, label)).or_insert(0);
            let this = *n;
            *n += 1;
            repeated |= this > 0;
            this
        })
        .collect();
    if !repeated {
        return None;
    }

    let mut used: std::collections::HashSet<String> = rows.iter().map(label_of).collect();
    let mut keys: HashMap<(String, usize), String> = HashMap::new();
    let mut order: Vec<Value> = Vec::new();
    let mut category_labels = JsonMap::new();
    let mut out_rows: Vec<Value> = Vec::with_capacity(rows.len());
    for (row, &n) in rows.iter().zip(&nth) {
        let label = label_of(row);
        if label.is_empty() {
            out_rows.push(row.clone());
            continue;
        }
        let key = keys
            .entry((label.clone(), n))
            .or_insert_with(|| {
                let mut key = label.clone();
                if n > 0 {
                    let mut k = n + 1;
                    while used.contains(&key) {
                        key = format!("{label} ({k})");
                        k += 1;
                    }
                    used.insert(key.clone());
                    category_labels.insert(key.clone().into(), Value::String(label.clone()));
                }
                order.push(Value::String(key.clone()));
                key
            })
            .clone();
        let mut row = row.clone();
        if let Some(obj) = row.as_object_mut() {
            obj.insert(
                (if by_category { "category" } else { "series" }).into(),
                Value::String(key),
            );
        }
        out_rows.push(row);
    }
    let mut out = data.clone();
    out.insert("rows".into(), Value::Array(out_rows));
    out.insert(
        (if by_category { "categories" } else { "series" }).into(),
        Value::Array(order),
    );
    out.insert("categoryLabels".into(), Value::Object(category_labels));
    Some(out)
}

fn pair(v: Option<&Value>) -> Option<(f64, f64)> {
    let a = v?.as_array()?;
    if a.len() < 2 {
        return None;
    }
    Some((num(a.first()), num(a.get(1))))
}

fn string_list(v: Option<&Value>) -> Vec<String> {
    v.and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|x| x.as_str().unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// One `DataPoint` row. Rows may be malformed (e.g. arrays in the
/// chart-gated fixture) — every accessor degrades like a JS property read on
/// a non-object (undefined).
#[derive(Clone, Copy)]
pub struct RowView<'a>(pub &'a Value);

impl<'a> RowView<'a> {
    fn obj(&self) -> Option<&'a JsonMap> {
        self.0.as_object()
    }
    pub fn s(&self, key: &str) -> Option<&'a str> {
        self.obj()?.get(key).and_then(Value::as_str)
    }
    pub fn s_truthy(&self, key: &str) -> Option<&'a str> {
        self.s(key).filter(|s| !s.is_empty())
    }
    /// JS `row.value` style numeric read → NaN when missing.
    pub fn f(&self, key: &str) -> f64 {
        num(self.obj().and_then(|m| m.get(key)))
    }
    pub fn f_opt(&self, key: &str) -> Option<f64> {
        match self.obj().and_then(|m| m.get(key)) {
            Some(Value::Number(n)) => n.as_f64(),
            _ => None,
        }
    }
    /// `row.missing`: an explicit empty cell, drawn as a gap (data.ts).
    pub fn missing(&self) -> bool {
        truthy(self.obj().and_then(|m| m.get("missing")))
    }
    pub fn metadata(&self) -> Option<&'a JsonMap> {
        self.obj()?.get("metadata").and_then(Value::as_object)
    }
    /// `getCategoryKey` (markers.ts:45): `row.category || row.series || ''`.
    pub fn category_key(&self) -> &'a str {
        self.s_truthy("category")
            .or_else(|| self.s_truthy("series"))
            .unwrap_or("")
    }
}

/// `resolveSeriesOrder` (series.ts:9-26).
pub fn resolve_series_order(series: &[String], series_order: Option<&[String]>) -> Vec<String> {
    let Some(order) = series_order.filter(|o| !o.is_empty()) else {
        return series.to_vec();
    };
    let mut ordered = Vec::new();
    let mut seen: HashMap<&str, ()> = HashMap::new();
    for name in order {
        if series.contains(name) && !seen.contains_key(name.as_str()) {
            ordered.push(name.clone());
            seen.insert(name, ());
        }
    }
    for name in series {
        if !seen.contains_key(name.as_str()) {
            ordered.push(name.clone());
            seen.insert(name, ());
        }
    }
    ordered
}

pub const SINGLE_SERIES_KEY: &str = "__single__";

/// `resolveChartDimensions` (generators/shared.ts:32-46).
pub struct ChartDims {
    pub has_category_dimension: bool,
    pub has_series_dimension: bool,
    pub ordered_categories: Vec<String>,
    pub ordered_series: Vec<String>,
    pub series_keys: Vec<String>,
}

/// `BarValueEntry` (generators/shared.ts).
#[derive(Clone, Debug)]
pub struct BarValueEntry {
    pub value: f64,
    pub label: Option<String>,
}

/// `aggregateBarValues` (generators/shared.ts).
///
/// Rows sharing a (category, series) pair are summed into one bar and the last
/// explicit `label` wins, so a bar's label text comes from the aggregate.
/// Shared by `gen_bar` (which draws the bars) and `calculate_chart_layout`
/// (which reserves room for horizontal outside value labels).
pub fn aggregate_bar_values(
    data: &DataView,
    has_series_dimension: bool,
) -> HashMap<String, HashMap<String, BarValueEntry>> {
    let mut values_by_category: HashMap<String, HashMap<String, BarValueEntry>> = HashMap::new();
    for row in &data.rows {
        let category_key = row.category_key();
        if category_key.is_empty() || row.missing() {
            continue;
        }
        let series_key = if has_series_dimension {
            row.s("series").unwrap_or("")
        } else {
            SINGLE_SERIES_KEY
        };
        let cat_map = values_by_category
            .entry(category_key.to_string())
            .or_default();
        let entry = cat_map
            .entry(series_key.to_string())
            .or_insert(BarValueEntry {
                value: 0.0,
                label: None,
            });
        entry.value += row.f("value");
        if let Some(l) = row.s_truthy("label") {
            entry.label = Some(l.to_string());
        }
    }
    values_by_category
}

/// `resolveBarStacking` (generators/shared.ts): `(is_stacked, is_stacked_percent)`; a bar chart stacks only with a
/// series dimension (its mode defaults to grouped).
pub fn resolve_bar_stacking(spec: &SpecView, has_series_dimension: bool) -> (bool, bool) {
    let resolved_mode = spec.s("mode").or(if has_series_dimension {
        Some("grouped")
    } else {
        None
    });
    (
        has_series_dimension
            && (resolved_mode == Some("stacked") || resolved_mode == Some("stacked-percent")),
        has_series_dimension && resolved_mode == Some("stacked-percent"),
    )
}

/// `barValueDomain` (generators/shared.ts): a bar chart's value domain before bounds, each bar's value, or with
/// stacking each stack's positive and negative totals (stacked-percent: 0 to 100). Shared by `gen_bar` (its axis)
/// and `calculate_chart_layout` (the value axis's label gutter).
pub fn bar_value_domain(
    values_by_category: &HashMap<String, HashMap<String, BarValueEntry>>,
    ordered_categories: &[String],
    (is_stacked, is_stacked_percent): (bool, bool),
) -> (f64, f64) {
    if is_stacked_percent {
        return (0.0, 100.0);
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for category in ordered_categories {
        let Some(cat_map) = values_by_category.get(category) else {
            continue;
        };
        if is_stacked {
            let mut positive_sum = 0.0;
            let mut negative_sum = 0.0;
            for entry in cat_map.values() {
                if entry.value >= 0.0 {
                    positive_sum += entry.value;
                } else {
                    negative_sum += entry.value;
                }
            }
            min = js_min(min, negative_sum);
            max = js_max(max, positive_sum);
        } else {
            for entry in cat_map.values() {
                min = js_min(min, entry.value);
                max = js_max(max, entry.value);
            }
        }
    }
    (min, max)
}

pub fn resolve_chart_dimensions(data: &DataView, spec: &SpecView) -> ChartDims {
    let has_category_dimension = !data.categories.is_empty();
    let has_series_dimension = !data.series.is_empty() && has_category_dimension;
    let ordered_categories = if has_category_dimension {
        data.categories.clone()
    } else {
        data.series.clone()
    };
    let ordered_series = if has_series_dimension {
        resolve_series_order(&data.series, spec.series_order().as_deref())
    } else {
        Vec::new()
    };
    let series_keys = if has_series_dimension {
        ordered_series.clone()
    } else {
        vec![SINGLE_SERIES_KEY.to_string()]
    };
    ChartDims {
        has_category_dimension,
        has_series_dimension,
        ordered_categories,
        ordered_series,
        series_keys,
    }
}
