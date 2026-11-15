//! Port of `utils/markup/elements/charts/generators/comboChart.ts`.
//!
//! Combo charts overlay bar / line / area series on a shared categorical X
//! axis, optionally with a second (right) Y axis. The generator reuses the
//! already-ported leaf helpers (scales, axes, area regions, curves, legend,
//! title, footer, background, markers) and only adds combo-specific layout
//! (dual-axis reservation, the `remap_axis_roles` spine→rect conversion, the
//! per-axis label-width estimate).
//!
//! Time X axes (`spec.xScale === 'time'`) are ported for the no-bar case (a
//! d3-time X scale; see `timeticks`/`timefmt`). A time-requested combo WITH bar
//! series falls back to the categorical X branch — bars need band widths —
//! matching the TS warn-and-fall-back.

use super::ticks::TickError;
use crate::value::{json, Map, Value};
use std::collections::HashMap;

use super::axes::{
    create_axis_and_font_config, generate_categorical_axis_nodes, generate_grid_nodes,
    generate_numeric_axis_nodes, resolve_dash_pattern, GridConfig,
};
use super::cfg::{
    num, resolve_chart_dimensions, ConfigView, DataView, JsonMap, SeriesStyleView, SeriesStyles,
    SpecView, SINGLE_SERIES_KEY,
};
use super::colorutil::{resolve_palette_color, resolve_shadow_props};
use super::paint_bag;

use super::defs::{BarBase, Def};
use super::fills::parse_fill;
use super::format::{create_time_tick_formatter, format_axis_tick, get_effective_axis_format};
use super::html::{build_styled_html, escape_html, raw_size, TextStyleProps};
use super::jsmath::{js_max, js_min};
use super::layout::{
    calculate_chart_layout, x_axis_band_height, Bounds, LABEL_WIDTH_PADDING,
    MIN_Y_AXIS_LABEL_WIDTH, SOURCE_FOOTER_GAP, TITLE_BOTTOM_GAP, TITLE_SUBTITLE_GAP,
    VALUE_LABEL_GAP,
};
use super::parts::{
    append_line_series_path, append_unstacked_area_series, build_legend_items, chart_point_groups,
    generate_background_node, generate_legend_nodes, generate_series_point_markers,
    generate_source_footer_nodes, generate_title_nodes, resolve_series_curve,
    resolve_series_fill_opacity, resolve_series_show_points, AreaFillOrder, SeriesPoint,
};
use super::scales::{
    create_linear_scale, create_log_scale, create_time_scale, get_display_range_from_ticks,
    is_log_domain_valid, CategoricalScale, ValueScale,
};
use super::value_axis_bounds::{
    apply_value_axis_bounds, clamp_scale_to_plot, resolve_value_axis_bounds, ValueAxisBounds,
};

/// `spec.dualAxis === true` (strict — truthy non-booleans must NOT enable it).
fn strict_true(v: Option<&Value>) -> bool {
    matches!(v, Some(Value::Bool(true)))
}

/// One resolved combo series: its chart kind + axis side + palette index.
struct ComboSeries<'a> {
    name: String,
    chart_type: &'static str, // "bar" | "line" | "area"
    y_axis: &'static str,     // "left" | "right"
    style: Option<SeriesStyleView<'a>>,
    index: usize,
}

struct ComboEntry {
    value: f64,
    #[allow(dead_code)]
    label: Option<String>,
    x_time: Option<f64>,
}

#[derive(Default)]
struct ComboLayout {
    plot_area: Bounds,
    title_area: Option<Bounds>,
    legend_area: Option<Bounds>,
    x_axis_area: Option<Bounds>,
    y_axis_area_left: Option<Bounds>,
    y_axis_area_right: Option<Bounds>,
    source_footer_area: Option<Bounds>,
}

/// `resolveBoxValue` (comboChart.ts:76-82) → (top, right, bottom, left).
fn resolve_box_value(value: Option<&Value>) -> (f64, f64, f64, f64) {
    match value {
        Some(Value::Array(a)) => (num(a.first()), num(a.get(1)), num(a.get(2)), num(a.get(3))),
        Some(Value::Number(n)) => {
            let v = n.as_f64().unwrap_or(f64::NAN);
            (v, v, v, v)
        }
        _ => (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
    }
}

/// `shouldShowLegend` (comboChart.ts:84-88).
fn should_show_legend(config: ConfigView, data: &DataView) -> bool {
    if config.show_legend_true() && data.rows.is_empty() {
        return true;
    }
    if data.series.len() <= 1 {
        return false;
    }
    config.show_legend_true() || config.show_legend_auto()
}

/// `cornerRadius` resolution shared with the bar generator: a series style
/// value (number | array) wins, else `config.barRadius`, else absent (Null,
/// which `def_to_node` passthrough drops).
fn corner_radius_value(style: Option<SeriesStyleView>, config: ConfigView) -> Value {
    style
        .and_then(|s| s.value("cornerRadius"))
        .cloned()
        .or_else(|| config.get("barRadius").cloned())
        .unwrap_or(Value::Null)
}

/// `estimateYAxisLabelWidthFromValues` (comboChart.ts:90-128).
fn estimate_y_axis_label_width_from_values(
    values: &[f64],
    config: ConfigView,
    format: Option<&str>,
    scale_kind: &str,
    // Authored y-min/y-max for this axis (left only — see the render path).
    // Reserved width has to be measured against the ticks the pinned domain
    // produces, or a pinned axis reserves space for the wrong labels.
    bounds: &ValueAxisBounds,
) -> Result<f64, TickError> {
    if values.is_empty() {
        return Ok(MIN_Y_AXIS_LABEL_WIDTH);
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in values {
        min = js_min(min, v);
        max = js_max(max, v);
    }
    if !min.is_finite() || !max.is_finite() {
        return Ok(MIN_Y_AXIS_LABEL_WIDTH);
    }
    let (bounded_min, bounded_max) = apply_value_axis_bounds((min, max), bounds);
    min = bounded_min;
    max = bounded_max;
    if scale_kind != "log" {
        if bounds.min.is_none() {
            min = js_min(min, 0.0);
        }
        if bounds.max.is_none() {
            max = js_max(max, 0.0);
        }
    }
    let display_range = apply_value_axis_bounds(
        get_display_range_from_ticks((min, max), config.f("yTickCount"), scale_kind),
        bounds,
    );
    let (ticks, effective_format) =
        get_effective_axis_format(display_range, config.f("yTickCount"), format)?;
    if ticks.is_empty() {
        return Ok(MIN_Y_AXIS_LABEL_WIDTH);
    }
    let mut max_label_length = 0.0f64;
    for tick in &ticks {
        let formatted = format_axis_tick(*tick, effective_format);
        max_label_length = js_max(max_label_length, formatted.encode_utf16().count() as f64);
    }
    const AXIS_BUFFER: f64 = 8.0;
    let estimated =
        (max_label_length * config.f("fontSize") * 0.7).ceil() + LABEL_WIDTH_PADDING + AXIS_BUFFER;
    Ok(js_max(MIN_Y_AXIS_LABEL_WIDTH, estimated))
}

/// `buildComboLayout` (comboChart.ts:130-254).
#[allow(clippy::too_many_arguments)]
fn build_combo_layout(
    spec: &SpecView,
    data: &DataView,
    left_axis_width: f64,
    right_axis_width: f64,
    reserve_right_axis: bool,
) -> Result<ComboLayout, TickError> {
    let config = spec.config;
    let bounds = Bounds {
        x: 0.0,
        y: 0.0,
        width: spec.width,
        height: spec.height,
    };
    let (pt, pr, pb, pl) = resolve_box_value(config.get("padding"));
    let mut top = bounds.y + pt;
    let mut bottom = bounds.y + bounds.height - pb;
    let mut left = bounds.x + pl;
    let mut right = bounds.x + bounds.width - pr;

    let mut title_area = None;
    let has_title = config.s_truthy("title").is_some();
    let has_subtitle = config.s_truthy("subtitle").is_some();
    if has_title || has_subtitle {
        let title_height = (if has_title {
            config.f("titleSize")
        } else {
            0.0
        }) + (if has_subtitle {
            config.f("subtitleSize")
        } else {
            0.0
        }) + (if has_title && has_subtitle {
            TITLE_SUBTITLE_GAP
        } else {
            0.0
        });
        title_area = Some(Bounds {
            x: left,
            y: top,
            width: right - left,
            height: title_height,
        });
        let title_gap = js_max(
            TITLE_BOTTOM_GAP,
            js_max(
                if has_title {
                    config.f("titleSize")
                } else {
                    0.0
                },
                if has_subtitle {
                    config.f("subtitleSize")
                } else {
                    0.0
                },
            ) / 2.0,
        ) + js_max(
            config.f("fontSize") / 2.0,
            if config.b("showLabels") {
                config.f("valueLabelSize") + 2.0 + VALUE_LABEL_GAP
            } else {
                0.0
            },
        );
        top += title_height + title_gap;
    }

    let mut source_footer_area = None;
    if let Some((label, _)) = spec.source() {
        if !label.trim().is_empty() {
            let text_height = config.f("subtitleSize");
            source_footer_area = Some(Bounds {
                x: left,
                y: bottom - text_height,
                width: js_max(0.0, right - left),
                height: text_height,
            });
            bottom -= text_height + SOURCE_FOOTER_GAP;
        }
    }

    let mut legend_area = None;
    if should_show_legend(config, data) {
        // Recursive layout pass on the same spec (x/y already 0) to reuse the
        // shared legend-thickness arithmetic, exactly like the TS generator.
        let legend_layout = calculate_chart_layout(spec, data)?;
        legend_area = legend_layout.legend_area;
        if let Some(la) = legend_area {
            match config.s("legendPosition").unwrap_or("") {
                "top" => top = la.y + la.height,
                "bottom" => bottom = la.y,
                "left" => left = la.x + la.width,
                "right" => right = la.x,
                _ => {}
            }
        }
    }

    let has_data = !data.rows.is_empty();
    let should_reserve_y = has_data && (config.b("showYLabels") || config.b("showYAxis"));

    let mut y_axis_area_left = None;
    if should_reserve_y && left_axis_width > 0.0 {
        y_axis_area_left = Some(Bounds {
            x: left,
            y: top,
            width: left_axis_width,
            height: bottom - top,
        });
        left += left_axis_width;
    }

    let mut y_axis_area_right = None;
    if should_reserve_y && reserve_right_axis && right_axis_width > 0.0 {
        y_axis_area_right = Some(Bounds {
            x: right - right_axis_width,
            y: top,
            width: right_axis_width,
            height: bottom - top,
        });
        right -= right_axis_width;
    }

    let mut x_axis_area = None;
    let should_reserve_x = has_data && (config.b("showXLabels") || config.b("showXAxis"));
    let x_axis_height = if should_reserve_x {
        if config.b("showXLabels") {
            x_axis_band_height(config.f("fontSize"))
        } else {
            config.f("axisWidth")
        }
    } else {
        0.0
    };
    if x_axis_height > 0.0 {
        x_axis_area = Some(Bounds {
            x: left,
            y: bottom - x_axis_height,
            width: right - left,
            height: x_axis_height,
        });
        bottom -= x_axis_height;
    }

    let plot_area = Bounds {
        x: left,
        y: top,
        width: js_max(0.0, right - left),
        height: js_max(0.0, bottom - top),
    };

    Ok(ComboLayout {
        plot_area,
        title_area,
        legend_area,
        x_axis_area,
        y_axis_area_left,
        y_axis_area_right,
        source_footer_area,
    })
}

/// `remapAxisRoles` (comboChart.ts:256-289): rename the chart-role metadata
/// and convert a remapped Y-axis SPINE line into a thin rectangle (so a
/// dual-axis chart's two spines render as filled rects, matching the browser).
fn remap_axis_roles(nodes: Vec<Def>, role_map: &[(&str, &str)]) -> Vec<Def> {
    nodes
        .into_iter()
        .map(|mut node| {
            let role = node
                .props
                .get("metadata")
                .and_then(Value::as_object)
                .and_then(|m| m.get("chartRole"))
                .and_then(Value::as_str)
                .map(str::to_string);
            let Some(role) = role else {
                return node;
            };
            let Some(&(_, mapped)) = role_map.iter().find(|(k, _)| *k == role) else {
                return node;
            };
            if let Some(meta) = node
                .props
                .get_mut("metadata")
                .and_then(Value::as_object_mut)
            {
                meta.insert("chartRole".into(), json!(mapped));
            }
            if (mapped == "axis-y-left" || mapped == "axis-y-right") && node.ty == "line" {
                if let Some(points) = node.props.get("points").and_then(Value::as_array) {
                    if points.len() >= 4 {
                        let x1 = points[0].as_f64().unwrap_or(f64::NAN);
                        let y1 = points[1].as_f64().unwrap_or(f64::NAN);
                        let y2 = points[3].as_f64().unwrap_or(f64::NAN);
                        let height = (y2 - y1).abs();
                        let y = js_min(y1, y2);
                        let stroke_width = match paint_bag::stroke_width(&node.props) {
                            Some(Value::Number(n)) => n.as_f64().unwrap_or(1.0),
                            _ => 1.0,
                        };
                        let width = js_max(1.0, stroke_width);
                        let x = x1 - width / 2.0;
                        let fill = paint_bag::stroke_color(&node.props).unwrap_or(Value::Null);

                        let name = node.props.get("name").cloned().unwrap_or(Value::Null);
                        let metadata = node.props.get("metadata").cloned().unwrap_or(Value::Null);
                        let mut d = Def::new("rectangle");
                        d.set("x", json!(x))
                            .set("y", json!(y))
                            .set("width", json!(width))
                            .set("height", json!(height))
                            .fill_color(fill)
                            .set("name", name)
                            .set("metadata", metadata);
                        return d;
                    }
                }
            }
            node
        })
        .collect()
}

/// `stripHtmlText` (comboChart.ts:291): drop tags, trim.
fn strip_html_text(html: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.trim().to_string()
}

/// `generateAxisLabelNode` (comboChart.ts:293-324) — a left/right axis title.
fn generate_axis_label_node(
    label: &str,
    area: Bounds,
    align: &str,
    role: &str,
    chart_name: &str,
    config: ConfigView,
) -> Def {
    let height = config.f("fontSize") + 2.0;
    let gap = 4.0;
    let mut d = Def::new("richtext");
    d.set("x", json!(area.x))
        .set("y", json!(area.y - height - gap))
        .set("width", json!(area.width))
        .set("height", json!(height))
        .set(
            "htmlContent",
            json!(build_styled_html(
                &TextStyleProps {
                    font_size: raw_size(config.f("fontSize")),
                    font_family: config.s("fontFamily"),
                    text_align: Some(align),
                    color: config.s("labelColor"),
                    ..Default::default()
                },
                &escape_html(label),
            )),
        )
        .set("name", json!(format!("{chart_name} Axis Label {label}")))
        .set(
            "metadata",
            json!({"chartRole": role, "chartName": chart_name}),
        );
    d
}

/// `generateComboChartNodes` (comboChart.ts:326-1088).
pub fn generate_combo_chart_nodes(
    spec: &SpecView,
    data: &DataView,
    styles: SeriesStyles,
    chart_name: &str,
) -> Result<Vec<Def>, TickError> {
    let config = spec.config;
    let mut nodes: Vec<Def> = Vec::new();

    let dims = resolve_chart_dimensions(data, spec);
    if dims.ordered_categories.is_empty() {
        return Ok(nodes);
    }

    // seriesConfigs in seriesKeys order (mirrors the TS Map iteration order).
    let mut series_configs: Vec<ComboSeries> = Vec::with_capacity(dims.series_keys.len());
    for (index, series_key) in dims.series_keys.iter().enumerate() {
        let style = styles.get(series_key);
        // `comboSeriesChartType` (types.ts): the authored `chartType`, else
        // bars for the first series and a line for every later one.
        let chart_type = match style.and_then(|s| s.s("chartType")) {
            Some("bar") => "bar",
            Some("line") => "line",
            Some("area") => "area",
            _ if index == 0 => "bar",
            _ => "line",
        };
        let y_axis = if style.and_then(|s| s.s("yAxis")) == Some("right") {
            "right"
        } else {
            "left"
        };
        series_configs.push(ComboSeries {
            name: series_key.clone(),
            chart_type,
            y_axis,
            style,
            index,
        });
    }

    // valuesByCategory: category → series → entry (last write wins, NOT summed).
    let mut values_by_category: HashMap<String, HashMap<String, ComboEntry>> = HashMap::new();
    for row in &data.rows {
        if row.missing() {
            continue;
        }
        let category_key = row.category_key();
        if category_key.is_empty() {
            continue;
        }
        let series_key = if dims.has_series_dimension {
            row.s("series").unwrap_or("")
        } else {
            SINGLE_SERIES_KEY
        };
        values_by_category
            .entry(category_key.to_string())
            .or_default()
            .insert(
                series_key.to_string(),
                ComboEntry {
                    value: row.f("value"),
                    label: row.s_truthy("label").map(str::to_string),
                    x_time: row.f_opt("xTime"),
                },
            );
    }

    // Time X axis (comboChart.ts:370-375): only the no-bar arm uses a real time
    // scale — bars need band widths, so a time-requested combo WITH bar series
    // warns (warn elided here, no logger — see scales.rs) and falls back to the
    // categorical X branch.
    let has_bar_series = series_configs.iter().any(|s| s.chart_type == "bar");
    let time_axis_requested = spec.s("xScale") == Some("time") && data.time_range().is_some();
    let is_time_axis = time_axis_requested && !has_bar_series;

    let dual_axis_enabled = strict_true(spec.map.get("dualAxis"));
    if !dual_axis_enabled {
        for series in series_configs.iter_mut() {
            series.y_axis = "left";
        }
    }

    let collect_axis_values = |which_axis: &str| -> Vec<f64> {
        // Pre-filter the series once (mirrors the TS `leftSeries`/`rightSeries`
        // lists passed into `collectAxisValues`), then walk categories.
        let filtered_series: Vec<&ComboSeries> = series_configs
            .iter()
            .filter(|s| s.y_axis == which_axis)
            .collect();
        let mut values = Vec::new();
        for category in &dims.ordered_categories {
            let Some(cat_map) = values_by_category.get(category) else {
                continue;
            };
            for series in &filtered_series {
                if let Some(entry) = cat_map.get(&series.name) {
                    values.push(entry.value);
                }
            }
        }
        values
    };
    let left_values = collect_axis_values("left");
    let right_values = collect_axis_values("right");
    let has_right_series = series_configs.iter().any(|s| s.y_axis == "right");

    let left_format = spec.s("leftAxisFormat").or(config.s("yFormat"));
    let right_format = spec.s("rightAxisFormat").or(config.s("yFormat"));

    // Shared log-vs-linear decision used for axis-WIDTH estimation.
    let axis_values_for_scale = |values: &[f64]| -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &v in values {
            min = js_min(min, v);
            max = js_max(max, v);
        }
        (min, max)
    };
    // Authored `y-min`/`y-max` pin the LEFT (primary) axis. A dual-axis combo's
    // right axis carries a different unit, so one pair of bounds cannot
    // describe both; the right axis keeps following its own data.
    let (left_data_min, left_data_max) = axis_values_for_scale(if !left_values.is_empty() {
        &left_values
    } else {
        &right_values
    });
    let y_bounds = resolve_value_axis_bounds(
        spec.f_opt("yMin"),
        spec.f_opt("yMax"),
        (left_data_min, left_data_max),
    );
    let (left_min_raw, left_max_raw) =
        apply_value_axis_bounds((left_data_min, left_data_max), &y_bounds);
    let left_valid_for_log =
        left_min_raw.is_finite() && is_log_domain_valid(left_min_raw, left_max_raw);
    let mut right_valid_for_log = true;
    if dual_axis_enabled && has_right_series {
        let (r_min, r_max) = axis_values_for_scale(&right_values);
        right_valid_for_log = r_min.is_finite() && is_log_domain_valid(r_min, r_max);
    }
    let shared_scale_kind =
        if config.s("yScale") == Some("log") && left_valid_for_log && right_valid_for_log {
            "log"
        } else {
            "linear"
        };

    let left_axis_width = if dual_axis_enabled && !series_configs.iter().any(|s| s.y_axis == "left")
    {
        0.0
    } else {
        estimate_y_axis_label_width_from_values(
            &left_values,
            config,
            left_format,
            shared_scale_kind,
            &y_bounds,
        )?
    };
    let right_axis_width = if dual_axis_enabled && has_right_series {
        js_max(
            20.0,
            estimate_y_axis_label_width_from_values(
                &right_values,
                config,
                right_format,
                shared_scale_kind,
                &ValueAxisBounds::default(),
            )?,
        )
    } else {
        0.0
    };

    let layout = build_combo_layout(
        spec,
        data,
        left_axis_width,
        right_axis_width,
        dual_axis_enabled && has_right_series,
    )?;

    if let Some(bg) = generate_background_node(config, spec.width, spec.height, chart_name) {
        nodes.push(bg);
    }
    if let Some(title_area) = layout.title_area {
        nodes.extend(generate_title_nodes(config, title_area, chart_name));
    }
    nodes.extend(generate_source_footer_nodes(
        spec.source(),
        config,
        layout.source_footer_area,
        chart_name,
    ));

    if layout.plot_area.width <= 0.0 || layout.plot_area.height <= 0.0 {
        return Ok(nodes);
    }

    // Left (primary) Y domain from the raw data of whichever axis has values.
    let left_domain_values = if !left_values.is_empty() {
        &left_values
    } else {
        &right_values
    };
    let mut min_left = f64::INFINITY;
    let mut max_left = f64::NEG_INFINITY;
    for &v in left_domain_values {
        min_left = js_min(min_left, v);
        max_left = js_max(max_left, v);
    }
    if !min_left.is_finite() || !max_left.is_finite() {
        min_left = 0.0;
        max_left = 0.0;
    }
    let (bounded_left_min, bounded_left_max) =
        apply_value_axis_bounds((min_left, max_left), &y_bounds);
    min_left = bounded_left_min;
    max_left = bounded_left_max;

    let baseline = config.f("baseline");
    let mut preliminary_right_valid = true;
    if dual_axis_enabled && has_right_series {
        let mut r_min = f64::INFINITY;
        let mut r_max = f64::NEG_INFINITY;
        for &v in &right_values {
            r_min = js_min(r_min, v);
            r_max = js_max(r_max, v);
        }
        preliminary_right_valid = is_log_domain_valid(r_min, r_max);
    }
    let effective_scale = if config.s("yScale") == Some("log")
        && is_log_domain_valid(min_left, max_left)
        && preliminary_right_valid
    {
        "log"
    } else {
        "linear"
    };
    let combo_log_baseline_valid = effective_scale == "log" && baseline > 0.0;
    // A pinned end is the author's floor/ceiling: neither zero nor the
    // baseline may widen it back out.
    if effective_scale != "log" {
        if y_bounds.min.is_none() {
            min_left = js_min(js_min(min_left, 0.0), baseline);
        }
        if y_bounds.max.is_none() {
            max_left = js_max(js_max(max_left, 0.0), baseline);
        }
    } else if combo_log_baseline_valid {
        if y_bounds.min.is_none() {
            min_left = js_min(min_left, baseline);
        }
        if y_bounds.max.is_none() {
            max_left = js_max(max_left, baseline);
        }
    }
    // Re-pin after nicing, which would round `y-min = 3` back out to 0.
    let left_display_range = apply_value_axis_bounds(
        get_display_range_from_ticks(
            (min_left, max_left),
            config.f("yTickCount"),
            effective_scale,
        ),
        &y_bounds,
    );
    let (_, effective_left_format) =
        get_effective_axis_format(left_display_range, config.f("yTickCount"), left_format)?;

    let mut right_display_range: Option<(f64, f64)> = None;
    let mut effective_right_format = right_format;
    if dual_axis_enabled && has_right_series {
        let mut min_right = f64::INFINITY;
        let mut max_right = f64::NEG_INFINITY;
        for &v in &right_values {
            min_right = js_min(min_right, v);
            max_right = js_max(max_right, v);
        }
        if !min_right.is_finite() || !max_right.is_finite() {
            min_right = 0.0;
            max_right = 0.0;
        }
        if effective_scale != "log" {
            min_right = js_min(js_min(min_right, 0.0), baseline);
            max_right = js_max(js_max(max_right, 0.0), baseline);
        } else if combo_log_baseline_valid {
            min_right = js_min(min_right, baseline);
            max_right = js_max(max_right, baseline);
        }
        let range = get_display_range_from_ticks(
            (min_right, max_right),
            config.f("yTickCount"),
            effective_scale,
        );
        right_display_range = Some(range);
        let (_, fmt) = get_effective_axis_format(range, config.f("yTickCount"), right_format)?;
        effective_right_format = fmt;
    }

    let x_scale = CategoricalScale::new(
        &dims.ordered_categories,
        layout.plot_area.width,
        config.f("barGroupGap"),
    );
    let build_value_scale = |range: (f64, f64), range_pixels: f64| -> ValueScale {
        if effective_scale == "log" {
            create_log_scale(range, range_pixels)
        } else {
            create_linear_scale(range, range_pixels)
        }
    };
    // Only the left axis can be pinned, so only it needs marks clamped in.
    let y_scale_left = clamp_scale_to_plot(
        build_value_scale(left_display_range, layout.plot_area.height),
        layout.plot_area.height,
        &y_bounds,
        left_display_range,
    );
    let y_scale_right = match right_display_range {
        Some(range) => build_value_scale(range, layout.plot_area.height),
        None => y_scale_left.clone(),
    };

    let left_baseline_value = if effective_scale == "log" {
        left_display_range.0
    } else {
        0.0
    };
    let baseline_y_left =
        layout.plot_area.y + layout.plot_area.height - y_scale_left.scale(left_baseline_value);

    nodes.extend(generate_grid_nodes(
        &y_scale_left,
        left_display_range,
        config.f("yTickCount"),
        "horizontal",
        layout.plot_area,
        &GridConfig {
            show: config.b("showGrid"),
            color: config.s("gridColor"),
            width: config.f("gridWidth"),
            style: config.s("gridStyle"),
        },
        chart_name,
    )?);

    let (axis_config_y, axis_config_x, font_config) = create_axis_and_font_config(config);

    let band_width = x_scale.bandwidth();
    // No-bar time arm: build the time scale (comboChart.ts:558-562). Line/area
    // series position points by it; bars are absent when `is_time_axis`.
    let x_tick_count_for_time = spec.f_opt("xTickCount").unwrap_or(config.f("xTickCount"));
    let time_display_range = if is_time_axis {
        Some(get_display_range_from_ticks(
            data.time_range().unwrap(),
            x_tick_count_for_time,
            "time",
        ))
    } else {
        None
    };
    let time_x_scale = time_display_range.map(|dr| create_time_scale(dr, layout.plot_area.width));
    let bar_series_count = js_max(
        1.0,
        series_configs
            .iter()
            .filter(|s| s.chart_type == "bar")
            .count() as f64,
    );
    let clamped_bar_gap = js_max(0.0, js_min(1.0, config.f("barGap")));
    let sub_band = band_width / bar_series_count;
    let bar_width = js_max(0.0, sub_band * (1.0 - clamped_bar_gap));
    let bar_offset = (sub_band - bar_width) / 2.0;

    let palette = config.palette();

    // ---- area series ------------------------------------------------------
    for series in series_configs.iter().filter(|s| s.chart_type == "area") {
        emit_area_series(
            &mut nodes,
            series,
            data,
            &dims.ordered_categories,
            &values_by_category,
            &x_scale,
            band_width,
            if series.y_axis == "right" {
                &y_scale_right
            } else {
                &y_scale_left
            },
            left_display_range,
            right_display_range,
            effective_scale,
            baseline,
            &layout,
            config,
            &palette,
            dims.has_series_dimension,
            chart_name,
            spec.point_colors(),
            data.category_labels(),
            time_x_scale.as_ref(),
        );
    }

    // ---- bar series -------------------------------------------------------
    let bar_series: Vec<&ComboSeries> = series_configs
        .iter()
        .filter(|s| s.chart_type == "bar")
        .collect();
    for category in &dims.ordered_categories {
        let Some(cat_map) = values_by_category.get(category) else {
            continue;
        };
        for (bar_index, series) in bar_series.iter().enumerate() {
            let Some(entry) = cat_map.get(&series.name) else {
                continue;
            };
            let y_scale = if series.y_axis == "right" {
                &y_scale_right
            } else {
                &y_scale_left
            };
            let axis_range = if series.y_axis == "right" {
                right_display_range.unwrap_or(left_display_range)
            } else {
                left_display_range
            };
            let series_baseline_value = if effective_scale == "log" {
                axis_range.0
            } else {
                0.0
            };
            let baseline_y =
                layout.plot_area.y + layout.plot_area.height - y_scale.scale(series_baseline_value);
            let value = entry.value;
            let scaled_value = y_scale.scale(value);
            let scaled_floor = y_scale.scale(series_baseline_value);
            let bar_height = (scaled_value - scaled_floor).abs();
            let bar_x = layout.plot_area.x
                + x_scale.scale(category)
                + bar_index as f64 * sub_band
                + bar_offset;
            let bar_y = if value >= series_baseline_value {
                baseline_y - bar_height
            } else {
                baseline_y
            };

            let style = series.style;
            let fill_value = style
                .and_then(|s| s.s_truthy("fill"))
                .map(str::to_string)
                .unwrap_or_else(|| resolve_palette_color(&palette, series.index).to_string());

            let mut d = Def::new("rectangle");
            d.grows_from(
                if value >= series_baseline_value {
                    BarBase::Bottom
                } else {
                    BarBase::Top
                },
                value,
            );
            d.set("x", json!(bar_x))
                .set("y", json!(bar_y))
                .set("width", json!(bar_width))
                .set("height", json!(bar_height))
                .set("cornerRadius", corner_radius_value(style, config))
                .set(
                    "name",
                    json!(format!("{chart_name} Bar {category} {}", series.name)),
                )
                .set(
                    "metadata",
                    json!({
                        "chartRole": "bar",
                        "chartCategory": category,
                        "chartSeries": series.name,
                        "chartYAxis": series.y_axis,
                        "chartName": chart_name,
                    }),
                );
            let bar_fill = spec
                .point_color(&series.name, data.category_label(category))
                .unwrap_or(&fill_value);
            for (k, v) in &parse_fill(bar_fill) {
                d.set(k, v.clone());
            }
            if let Some(shadow) = style.and_then(|s| s.s("shadow")) {
                for (k, v) in resolve_shadow_props(Some(shadow)) {
                    d.set(&k, v);
                }
            }
            if let Some(stroke) = style.and_then(|s| s.s_truthy("stroke")) {
                d.stroke_color(json!(stroke));
            }
            if let Some(sw) = style.and_then(|s| s.f_opt("strokeWidth")) {
                d.stroke_width(json!(sw));
            }
            if let Some(op) = style.and_then(|s| s.f_opt("opacity")) {
                d.set("opacity", json!(op));
            }
            nodes.push(d);
        }
    }

    // ---- line series ------------------------------------------------------
    for series in series_configs.iter().filter(|s| s.chart_type == "line") {
        emit_line_series(
            &mut nodes,
            series,
            data,
            &dims.ordered_categories,
            &values_by_category,
            &x_scale,
            band_width,
            if series.y_axis == "right" {
                &y_scale_right
            } else {
                &y_scale_left
            },
            &layout,
            config,
            &palette,
            dims.has_series_dimension,
            chart_name,
            spec.point_colors(),
            data.category_labels(),
            time_x_scale.as_ref(),
        );
    }

    // ---- axes -------------------------------------------------------------
    if dual_axis_enabled {
        if let Some(area) = layout.y_axis_area_left {
            let left_axis_nodes = generate_numeric_axis_nodes(
                &y_scale_left,
                left_display_range,
                config.f("yTickCount"),
                effective_left_format,
                None,
                "left",
                layout.plot_area,
                Some(area),
                &axis_config_y,
                &font_config,
                chart_name,
                None,
            )?;
            nodes.extend(remap_axis_roles(
                left_axis_nodes,
                &[
                    ("axis-y", "axis-y-left"),
                    ("axis-y-tick", "axis-y-tick-left"),
                    ("y-label", "y-label-left"),
                ],
            ));
        }
    }

    if dual_axis_enabled {
        if let (Some(area), Some(range)) = (layout.y_axis_area_right, right_display_range) {
            let right_axis_nodes = generate_numeric_axis_nodes(
                &y_scale_right,
                range,
                config.f("yTickCount"),
                effective_right_format,
                None,
                "right",
                layout.plot_area,
                Some(area),
                &axis_config_y,
                &font_config,
                chart_name,
                None,
            )?;
            let remapped = remap_axis_roles(
                right_axis_nodes,
                &[
                    ("axis-y", "axis-y-right"),
                    ("axis-y-tick", "axis-y-tick-right"),
                    ("y-label", "y-label-right"),
                ],
            );
            // Rebuild y-label-right htmlContent with a right alignment (the TS
            // generator re-wraps the label text through buildStyledHtml).
            for mut node in remapped {
                let is_ylabel_right = node
                    .props
                    .get("metadata")
                    .and_then(Value::as_object)
                    .and_then(|m| m.get("chartRole"))
                    .and_then(Value::as_str)
                    == Some("y-label-right");
                if is_ylabel_right {
                    let html = node
                        .props
                        .get("htmlContent")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let label_text = strip_html_text(html);
                    let rebuilt = build_styled_html(
                        &TextStyleProps {
                            font_size: raw_size(font_config.font_size),
                            font_family: font_config.font_family,
                            text_align: Some("right"),
                            color: font_config.label_color,
                            ..Default::default()
                        },
                        &escape_html(&label_text),
                    );
                    node.set("htmlContent", json!(rebuilt));
                }
                nodes.push(node);
            }

            // Explicit zero-label inject on the right axis (linear scales only).
            if config.b("showYLabels") && effective_scale != "log" {
                let zero_label = format_axis_tick(0.0, effective_right_format);
                if !zero_label.is_empty() {
                    let label_width = js_max(
                        0.0,
                        area.width - axis_config_y.tick_length - axis_config_y.label_gap,
                    );
                    let y = layout.plot_area.y + layout.plot_area.height - y_scale_right.scale(0.0);
                    let mut d = Def::new("richtext");
                    d.set("x", json!(area.x))
                        .set("y", json!(y - font_config.font_size / 2.0))
                        .set("width", json!(label_width))
                        .set("height", json!(font_config.font_size + 2.0))
                        .set(
                            "htmlContent",
                            json!(build_styled_html(
                                &TextStyleProps {
                                    font_size: raw_size(font_config.font_size),
                                    font_family: font_config.font_family,
                                    text_align: Some("right"),
                                    color: font_config.label_color,
                                    ..Default::default()
                                },
                                &escape_html(&zero_label),
                            )),
                        )
                        .set(
                            "name",
                            json!(format!("{chart_name} Right Axis Label {zero_label}")),
                        )
                        .set(
                            "metadata",
                            json!({"chartRole": "y-label-right", "chartValue": 0.0, "chartName": chart_name}),
                        );
                    nodes.push(d);
                }
            }
        }
    }

    if !dual_axis_enabled {
        if let Some(area) = layout.y_axis_area_left {
            let (_, effective_single_format) = get_effective_axis_format(
                left_display_range,
                config.f("yTickCount"),
                config.s("yFormat"),
            )?;
            nodes.extend(generate_numeric_axis_nodes(
                &y_scale_left,
                left_display_range,
                config.f("yTickCount"),
                effective_single_format,
                None,
                "left",
                layout.plot_area,
                Some(area),
                &axis_config_y,
                &font_config,
                chart_name,
                None,
            )?);
        }
    }

    // Time X axis: multi-scale (or xTimeFormat) labels anchored at the Y-space
    // baseline like the categorical fallback (comboChart.ts time arm).
    if let (Some(time_scale), Some(time_range)) = (&time_x_scale, time_display_range) {
        let time_formatter = create_time_tick_formatter(spec.s("xTimeFormat"));
        nodes.extend(generate_numeric_axis_nodes(
            time_scale,
            time_range,
            x_tick_count_for_time,
            Some("auto"),
            Some(&*time_formatter),
            "bottom",
            layout.plot_area,
            layout.x_axis_area,
            &axis_config_x,
            &font_config,
            chart_name,
            Some(baseline_y_left),
        )?);
    } else {
        nodes.extend(generate_categorical_axis_nodes(
            &dims.ordered_categories,
            &x_scale,
            "bottom",
            layout.plot_area,
            layout.x_axis_area,
            &axis_config_x,
            &font_config,
            chart_name,
            Some(baseline_y_left),
            data.category_labels(),
        ));
    }

    if dual_axis_enabled {
        if let Some(area) = layout.y_axis_area_left {
            if let Some(label) = spec.s("leftAxisLabel").filter(|s| !s.is_empty()) {
                nodes.push(generate_axis_label_node(
                    label,
                    area,
                    "center",
                    "axis-label-left",
                    chart_name,
                    config,
                ));
            }
        }
        if let Some(area) = layout.y_axis_area_right {
            if let Some(label) = spec.s("rightAxisLabel").filter(|s| !s.is_empty()) {
                nodes.push(generate_axis_label_node(
                    label,
                    area,
                    "center",
                    "axis-label-right",
                    chart_name,
                    config,
                ));
            }
        }
    }

    // ---- legend -----------------------------------------------------------
    if let Some(legend_area) = layout.legend_area {
        let base_items = build_legend_items(&dims.ordered_series, styles, &palette);
        let items = base_items
            .into_iter()
            .map(|mut item| {
                let Some(series) = series_configs.iter().find(|s| s.name == item.name) else {
                    return item;
                };
                let style = series.style;
                if series.chart_type == "line" {
                    let stroke = style
                        .and_then(|s| s.s_truthy("stroke"))
                        .or_else(|| style.and_then(|s| s.s_truthy("fill")))
                        .map(str::to_string)
                        .unwrap_or_else(|| {
                            resolve_palette_color(&palette, series.index).to_string()
                        });
                    item.fill = stroke.clone();
                    item.stroke = Some(stroke);
                    item.stroke_width = Some(
                        style
                            .and_then(|s| s.f_opt("strokeWidth"))
                            .unwrap_or(config.f("strokeWidth")),
                    );
                    item.dash = resolve_dash_pattern(
                        style
                            .and_then(|s| s.s("strokeStyle"))
                            .or(config.s("strokeStyle")),
                    );
                } else {
                    item.fill = style
                        .and_then(|s| s.s_truthy("fill"))
                        .map(str::to_string)
                        .unwrap_or_else(|| {
                            resolve_palette_color(&palette, series.index).to_string()
                        });
                }
                item
            })
            .collect::<Vec<_>>();
        nodes.extend(generate_legend_nodes(
            &items,
            config,
            legend_area,
            chart_name,
            None,
        ));
    }

    Ok(nodes)
}

/// One area series (combo.ts:603-748) — categorical, sign-crossing regions.
#[allow(clippy::too_many_arguments)]
fn emit_area_series(
    nodes: &mut Vec<Def>,
    series: &ComboSeries,
    data: &DataView,
    ordered_categories: &[String],
    values_by_category: &HashMap<String, HashMap<String, ComboEntry>>,
    x_scale: &CategoricalScale,
    band_width: f64,
    y_scale: &ValueScale,
    left_display_range: (f64, f64),
    right_display_range: Option<(f64, f64)>,
    effective_scale: &str,
    baseline: f64,
    layout: &ComboLayout,
    config: ConfigView,
    palette: &[String],
    has_series_dimension: bool,
    chart_name: &str,
    point_colors: Option<&JsonMap>,
    category_labels: Option<&JsonMap>,
    time_x_scale: Option<&ValueScale>,
) {
    let style = series.style;
    let fill_color = style
        .and_then(|s| s.s_truthy("fill"))
        .map(str::to_string)
        .unwrap_or_else(|| resolve_palette_color(palette, series.index).to_string());
    let fill_opacity = resolve_series_fill_opacity(style, config.f("fillOpacity"));
    let stroke = style.and_then(|s| s.s("stroke")).or(config.s("stroke"));
    let stroke_truthy = stroke.filter(|s| !s.is_empty());
    let stroke_width = stroke_truthy.map(|_| {
        style
            .and_then(|s| s.f_opt("strokeWidth"))
            .unwrap_or(config.f("strokeWidth"))
    });
    let stroke_style = style
        .and_then(|s| s.s("strokeStyle"))
        .or(config.s("strokeStyle"));
    let dash = if stroke_truthy.is_some() {
        resolve_dash_pattern(stroke_style)
    } else {
        None
    };
    let opacity = style.and_then(|s| s.f_opt("opacity"));
    let curve = resolve_series_curve(style, config.s("curve"));

    let series_axis_range = if series.y_axis == "right" {
        right_display_range.unwrap_or(left_display_range)
    } else {
        left_display_range
    };
    let area_baseline_value = if effective_scale == "log" {
        if baseline > 0.0 {
            baseline
        } else {
            series_axis_range.0
        }
    } else {
        baseline
    };
    let baseline_y_pixel = layout.plot_area.height - y_scale.scale(area_baseline_value);

    let mut series_points: Vec<SeriesPoint> = Vec::new();
    let mut local_top_points: Vec<(f64, f64)> = Vec::new();
    for category in ordered_categories {
        let Some(entry) = values_by_category
            .get(category)
            .and_then(|m| m.get(&series.name))
        else {
            continue;
        };
        // Time axis: position by the parsed xTime; skip points without one
        // (comboChart.ts:629-634). Categorical centers in the band.
        let x = if let Some(ts) = time_x_scale {
            match entry.x_time {
                Some(xt) => ts.scale(xt),
                None => continue,
            }
        } else {
            x_scale.scale(category) + band_width / 2.0
        };
        let y = layout.plot_area.height - y_scale.scale(entry.value);
        series_points.push(SeriesPoint {
            category: category.clone(),
            value: entry.value,
            x: layout.plot_area.x + x,
            y: layout.plot_area.y + y,
            x_time: entry.x_time,
        });
        local_top_points.push((x, y));
    }

    // Time area fills must trace left-to-right; reorder by pixel-x ascending
    // (comboChart.ts:645-656).
    if time_x_scale.is_some() {
        let mut indices: Vec<usize> = (0..local_top_points.len()).collect();
        indices.sort_by(|&a, &b| {
            local_top_points[a]
                .0
                .partial_cmp(&local_top_points[b].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        series_points = indices.iter().map(|&i| series_points[i].clone()).collect();
        local_top_points = indices.iter().map(|&i| local_top_points[i]).collect();
    }

    if !local_top_points.is_empty() {
        let base_name = format!("{chart_name} Area {}", series.name);
        let shadow_props = resolve_shadow_props(style.and_then(|s| s.s("shadow")));
        let mut metadata = Map::new();
        metadata.insert("chartRole".into(), json!("area-series"));
        metadata.insert("chartSeries".into(), json!(series.name));
        metadata.insert("chartYAxis".into(), json!(series.y_axis));
        metadata.insert("chartName".into(), json!(chart_name));
        let groups = chart_point_groups(
            &series_points,
            data,
            &series.name,
            has_series_dimension,
            ordered_categories,
        );
        append_unstacked_area_series(
            nodes,
            &local_top_points,
            &groups,
            baseline_y_pixel,
            curve,
            layout.plot_area,
            &fill_color,
            fill_opacity,
            stroke_truthy,
            stroke_width,
            dash.as_deref(),
            config.s("strokeCap"),
            config.s("strokeJoin"),
            opacity,
            &base_name,
            &metadata,
            &shadow_props,
            AreaFillOrder::BeforeTail,
        );
    }

    if resolve_series_show_points(style, config.b("showPoints")) {
        nodes.extend(generate_series_point_markers(
            &series_points,
            style,
            config,
            &series.name,
            has_series_dimension,
            chart_name,
            &fill_color,
            point_colors,
            category_labels,
        ));
    }
}

/// One line series (combo.ts:799-878) — categorical.
#[allow(clippy::too_many_arguments)]
fn emit_line_series(
    nodes: &mut Vec<Def>,
    series: &ComboSeries,
    data: &DataView,
    ordered_categories: &[String],
    values_by_category: &HashMap<String, HashMap<String, ComboEntry>>,
    x_scale: &CategoricalScale,
    band_width: f64,
    y_scale: &ValueScale,
    layout: &ComboLayout,
    config: ConfigView,
    palette: &[String],
    has_series_dimension: bool,
    chart_name: &str,
    point_colors: Option<&JsonMap>,
    category_labels: Option<&JsonMap>,
    time_x_scale: Option<&ValueScale>,
) {
    let style = series.style;
    let stroke = style
        .and_then(|s| s.s_truthy("stroke"))
        .or_else(|| style.and_then(|s| s.s_truthy("fill")))
        .map(str::to_string)
        .unwrap_or_else(|| resolve_palette_color(palette, series.index).to_string());
    let stroke_width = style
        .and_then(|s| s.f_opt("strokeWidth"))
        .unwrap_or(config.f("strokeWidth"));
    let stroke_style = style
        .and_then(|s| s.s("strokeStyle"))
        .or(config.s("strokeStyle"));
    let dash = resolve_dash_pattern(stroke_style);
    let opacity = style.and_then(|s| s.f_opt("opacity"));
    let curve = resolve_series_curve(style, config.s("curve"));

    let mut series_points: Vec<SeriesPoint> = Vec::new();
    let mut local_points: Vec<(f64, f64)> = Vec::new();
    for category in ordered_categories {
        let Some(entry) = values_by_category
            .get(category)
            .and_then(|m| m.get(&series.name))
        else {
            continue;
        };
        // Time axis: position by the parsed xTime; skip points without one
        // (comboChart.ts:816-821). Categorical centers in the band.
        let x = if let Some(ts) = time_x_scale {
            match entry.x_time {
                Some(xt) => layout.plot_area.x + ts.scale(xt),
                None => continue,
            }
        } else {
            layout.plot_area.x + x_scale.scale(category) + band_width / 2.0
        };
        let y = layout.plot_area.y + layout.plot_area.height - y_scale.scale(entry.value);
        series_points.push(SeriesPoint {
            category: category.clone(),
            value: entry.value,
            x,
            y,
            x_time: entry.x_time,
        });
        local_points.push((x - layout.plot_area.x, y - layout.plot_area.y));
    }

    // Time series draw chronologically; reorder both arrays by pixel-x
    // (comboChart.ts:826-835).
    if time_x_scale.is_some() {
        let mut indices: Vec<usize> = (0..local_points.len()).collect();
        indices.sort_by(|&a, &b| {
            series_points[a]
                .x
                .partial_cmp(&series_points[b].x)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        series_points = indices.iter().map(|&i| series_points[i].clone()).collect();
        local_points = indices.iter().map(|&i| local_points[i]).collect();
    }

    let mut metadata = Map::new();
    metadata.insert("chartRole".into(), json!("line-series"));
    metadata.insert("chartSeries".into(), json!(series.name));
    metadata.insert("chartYAxis".into(), json!(series.y_axis));
    metadata.insert("chartName".into(), json!(chart_name));
    let groups = chart_point_groups(
        &series_points,
        data,
        &series.name,
        has_series_dimension,
        ordered_categories,
    );
    append_line_series_path(
        nodes,
        &local_points,
        &groups,
        curve,
        layout.plot_area,
        &stroke,
        stroke_width,
        dash.as_deref(),
        config.s("strokeCap"),
        config.s("strokeJoin"),
        opacity,
        format!("{chart_name} Line {}", series.name),
        metadata,
        &resolve_shadow_props(style.and_then(|s| s.s("shadow"))),
    );

    if resolve_series_show_points(style, config.b("showPoints")) {
        nodes.extend(generate_series_point_markers(
            &series_points,
            style,
            config,
            &series.name,
            has_series_dimension,
            chart_name,
            &stroke,
            point_colors,
            category_labels,
        ));
    }
}
