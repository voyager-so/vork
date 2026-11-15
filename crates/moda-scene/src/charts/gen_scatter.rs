//! Port of `utils/markup/elements/charts/generators/scatterChart.ts`, including
//! the `x-scale="time"` arm (a d3-time X scale over the pre-parsed ms x values;
//! see `timeticks`/`timefmt`). Y stays numeric.

use super::ticks::TickError;
use crate::value::{json, Map, Value};
use std::collections::HashMap;

use super::axes::{
    create_axis_and_font_config, dash_value, fit_bottom_tick_count, generate_axis_title_nodes,
    generate_grid_nodes, generate_numeric_axis_nodes, resolve_dash_pattern,
};
use super::cfg::{
    resolve_series_order, ConfigView, DataView, SeriesStyles, SpecView, SINGLE_SERIES_KEY,
};
use super::colorutil::resolve_palette_color;
use super::curves::{path_pts_to_value, PathPt};
use super::defs::Def;
use super::format::{create_time_tick_formatter, get_effective_axis_format};
use super::html::{build_styled_html, escape_html, raw_size, TextStyleProps};
use super::jsmath::{js_max, js_min, js_num_to_string};
use super::layout::{calculate_chart_layout, Bounds};
use super::parts::generate_background_node;
use super::parts::{
    generate_legend_nodes, generate_source_footer_nodes, generate_title_nodes,
    get_point_marker_node, linear_regression, resolve_point_shape, resolve_point_size, LegendItem,
    PointMarkerArgs,
};
use super::scales::{
    create_linear_scale, create_log_scale, create_size_scale, create_time_scale,
    get_display_range_from_ticks, is_log_domain_valid, ValueScale,
};
use super::timeticks::utc_date_parts;
use super::value_axis_bounds::{
    apply_value_axis_bounds, clamp_scale_to_plot, resolve_value_axis_bounds,
};

struct ScatterPoint {
    x: f64,
    y: f64,
    series: Option<String>,
    category: Option<String>,
    size: Option<f64>,
    label: Option<String>,
    screen_x: f64,
    screen_y: f64,
}

/// `Date.toISOString()` year field: 4 digits for `[0, 9999]`, else the extended
/// signed 6-digit form (`+012345` / `-000005`). Matters only in principle here
/// — `parseDateString` only yields years 0-9999 — but keeps the sign out of the
/// zero-pad width (the naive `{:04}` would render year −5 as `-005`, not the
/// `Date.toISOString()` `-000005`).
fn iso_year(year: i64) -> String {
    if (0..=9999).contains(&year) {
        format!("{year:04}")
    } else {
        format!(
            "{}{:06}",
            if year < 0 { '-' } else { '+' },
            year.unsigned_abs()
        )
    }
}

/// `xForName` (scatterChart.ts:409-419): a time-axis point's human-readable
/// node name uses an ISO date string instead of the raw ms integer. Midnight
/// UTC renders `toISOString().slice(0, 10)`; otherwise the full
/// `Date.toISOString()`, dropping the `.000` sub-second tail when there are no
/// milliseconds.
fn scatter_time_name(ms: f64) -> String {
    let p = utc_date_parts(ms as i64);
    let iso = format!(
        "{}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        iso_year(p.year),
        p.month0 + 1,
        p.day,
        p.hours,
        p.minutes,
        p.seconds,
        p.millis
    );
    if p.hours == 0 && p.minutes == 0 && p.seconds == 0 && p.millis == 0 {
        // toISOString().slice(0, 10)
        return iso.chars().take(10).collect();
    }
    if p.millis == 0 {
        // toISOString().replace(/\.000Z$/, 'Z')
        iso.strip_suffix(".000Z")
            .map(|s| format!("{s}Z"))
            .unwrap_or(iso)
    } else {
        iso
    }
}

fn get_size_value(row: &super::cfg::RowView, size_column: Option<&str>) -> Option<f64> {
    if let Some(col) = size_column {
        if let Some(meta) = row.metadata() {
            if let Some(v) = meta.get(col) {
                return v.as_f64();
            }
        }
    }
    row.f_opt("size")
}

pub fn generate_scatter_chart_nodes(
    spec: &SpecView,
    data: &DataView,
    styles: SeriesStyles,
    chart_name: &str,
) -> Result<Vec<Def>, TickError> {
    let config = spec.config;
    let mut nodes: Vec<Def> = Vec::new();

    let layout = calculate_chart_layout(spec, data)?;

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

    let has_series_dimension = !data.series.is_empty();
    let order = spec.series_order();
    let ordered_series = if has_series_dimension {
        resolve_series_order(&data.series, order.as_deref())
    } else {
        Vec::new()
    };
    let series_keys: Vec<String> = if has_series_dimension {
        ordered_series.clone()
    } else {
        vec![SINGLE_SERIES_KEY.to_string()]
    };
    let series_index_map: HashMap<&str, usize> = series_keys
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let mut x_min = spec
        .f_opt("xMin")
        .or(data.x_range().map(|r| r.0))
        .unwrap_or(0.0);
    let mut x_max = spec
        .f_opt("xMax")
        .or(data.x_range().map(|r| r.1))
        .unwrap_or(100.0);
    // Y bounds route through the shared value-axis helper so scatter
    // validates, re-pins and clamps them exactly like the other generators —
    // `y-min`/`y-max` are one documented attribute pair, not five dialects.
    let y_data_min = data.y_range().map(|r| r.0).unwrap_or(0.0);
    let y_data_max = data.y_range().map(|r| r.1).unwrap_or(100.0);
    let y_bounds = resolve_value_axis_bounds(
        spec.f_opt("yMin"),
        spec.f_opt("yMax"),
        (y_data_min, y_data_max),
    );
    let (mut y_min, mut y_max) = apply_value_axis_bounds((y_data_min, y_data_max), &y_bounds);

    let requested_x_scale = spec.s("xScale").unwrap_or("linear");
    let requested_y_scale = config.s("yScale");

    if x_min == x_max {
        if requested_x_scale == "log" && x_min > 0.0 {
            x_min /= 10.0;
            x_max *= 10.0;
        } else if requested_x_scale == "time" {
            const ONE_DAY_MS: f64 = 24.0 * 60.0 * 60.0 * 1000.0;
            x_min -= ONE_DAY_MS;
            x_max += ONE_DAY_MS;
        } else {
            x_min -= 10.0;
            x_max += 10.0;
        }
    }
    if y_min == y_max {
        if requested_y_scale == Some("log") && y_min > 0.0 {
            y_min /= 10.0;
            y_max *= 10.0;
        } else {
            y_min -= 10.0;
            y_max += 10.0;
        }
    }

    // Time is resolved only on the X axis; scatter Y stays numeric
    // (scatterChart.ts:185-186).
    let x_scale_kind = if requested_x_scale == "time" {
        "time"
    } else if requested_x_scale == "log" && is_log_domain_valid(x_min, x_max) {
        "log"
    } else {
        "linear"
    };
    let y_scale_kind = if requested_y_scale == Some("log") && is_log_domain_valid(y_min, y_max) {
        "log"
    } else {
        "linear"
    };

    let x_tick_count = spec.f_opt("xTickCount").unwrap_or(config.f("xTickCount"));
    let x_display_range = get_display_range_from_ticks((x_min, x_max), x_tick_count, x_scale_kind);
    // Re-pin after nicing so an authored bound is not rounded away.
    let y_display_range = apply_value_axis_bounds(
        get_display_range_from_ticks((y_min, y_max), config.f("yTickCount"), y_scale_kind),
        &y_bounds,
    );

    let (_, effective_x_format) = get_effective_axis_format(
        x_display_range,
        x_tick_count,
        // x is its own quantity: `yFormat` (percent, currency) is the value axis's, never the x axis's.
        spec.s("xFormat").or(Some("auto")),
    )?;
    let (_, effective_y_format) =
        get_effective_axis_format(y_display_range, config.f("yTickCount"), config.s("yFormat"))?;

    let x_scale = if x_scale_kind == "time" {
        create_time_scale(x_display_range, layout.plot_area.width)
    } else if x_scale_kind == "log" {
        create_log_scale(x_display_range, layout.plot_area.width)
    } else {
        create_linear_scale(x_display_range, layout.plot_area.width)
    };
    let y_scale = clamp_scale_to_plot(
        if y_scale_kind == "log" {
            create_log_scale(y_display_range, layout.plot_area.height)
        } else {
            create_linear_scale(y_display_range, layout.plot_area.height)
        },
        layout.plot_area.height,
        &y_bounds,
        y_display_range,
    );

    // Size scale
    let size_column = spec.s("sizeColumn");
    let has_size_data = data.columns_include("size")
        || size_column
            .map(|c| data.columns_include(c))
            .unwrap_or(false);
    let mut actual_size_range: Option<(f64, f64)> = None;
    if has_size_data {
        let mut size_min = f64::INFINITY;
        let mut size_max = f64::NEG_INFINITY;
        for row in &data.rows {
            if row.missing() {
                continue;
            }
            if let Some(v) = get_size_value(row, size_column) {
                size_min = js_min(size_min, v);
                size_max = js_max(size_max, v);
            }
        }
        if size_min.is_finite() && size_max.is_finite() {
            actual_size_range = Some((size_min, size_max));
        }
    }
    let min_point_size = config.f("minPointSize");
    let max_point_size = config.f("maxPointSize");
    let size_scale_kind = spec.s("sizeScale").unwrap_or("area");
    let size_scale_fn: Box<dyn Fn(Option<f64>) -> f64> = match actual_size_range {
        None => Box::new(move |_| min_point_size),
        Some(range) => {
            let scale = create_size_scale(range, (min_point_size, max_point_size), size_scale_kind);
            Box::new(move |v| match v {
                None => min_point_size,
                Some(v) => scale(v),
            })
        }
    };

    let (axis_config_y, axis_config_x, font_config) = create_axis_and_font_config(config);
    // A numeric x axis draws fewer ticks when its labels would run together
    // (the grid follows; `fitBottomTickCount`); a time axis keeps its own.
    let x_value_tick_count = if x_scale_kind != "time" && axis_config_x.show_labels {
        fit_bottom_tick_count(
            &x_scale,
            x_display_range,
            x_tick_count,
            effective_x_format,
            font_config.font_size,
        )?
    } else {
        x_tick_count
    };

    // Grids (y then x, each with gridAxis metadata)
    let grid_cfg = super::axes::GridConfig {
        show: config.b("showGrid"),
        color: config.s("gridColor"),
        width: config.f("gridWidth"),
        style: config.s("gridStyle"),
    };
    for mut d in generate_grid_nodes(
        &y_scale,
        y_display_range,
        config.f("yTickCount"),
        "horizontal",
        layout.plot_area,
        &grid_cfg,
        chart_name,
    )? {
        add_grid_axis(&mut d, "y");
        nodes.push(d);
    }
    for mut d in generate_grid_nodes(
        &x_scale,
        x_display_range,
        x_value_tick_count,
        "vertical",
        layout.plot_area,
        &grid_cfg,
        chart_name,
    )? {
        add_grid_axis(&mut d, "x");
        nodes.push(d);
    }

    // Quadrants (before points)
    if spec.b("showQuadrants") {
        nodes.extend(generate_quadrant_nodes(
            spec,
            data,
            layout.plot_area,
            &x_scale,
            &y_scale,
            x_display_range,
            y_display_range,
            chart_name,
            config,
            x_scale_kind,
            y_scale_kind,
        ));
    }

    // Screen-space points
    let mut scatter_points: Vec<ScatterPoint> = Vec::new();
    for row in &data.rows {
        if row.missing() {
            continue;
        }
        let (Some(x), Some(y)) = (row.f_opt("x"), row.f_opt("y")) else {
            continue;
        };
        if (x_scale_kind == "log" && x <= 0.0) || (y_scale_kind == "log" && y <= 0.0) {
            continue;
        }
        let screen_x = layout.plot_area.x + x_scale.scale(x);
        let screen_y = layout.plot_area.y + layout.plot_area.height - y_scale.scale(y);
        scatter_points.push(ScatterPoint {
            x,
            y,
            series: row.s("series").map(str::to_string),
            category: row.s_truthy("category").map(str::to_string),
            size: get_size_value(row, size_column),
            label: row.s_truthy("label").map(str::to_string),
            screen_x,
            screen_y,
        });
    }

    // Trend line (before points)
    if spec.b("showTrendLine") && scatter_points.len() >= 2 {
        nodes.extend(generate_trend_line_nodes(
            &scatter_points,
            layout.plot_area,
            &x_scale,
            &y_scale,
            x_display_range,
            y_display_range,
            spec,
            chart_name,
            x_scale_kind,
            y_scale_kind,
        ));
    }

    // Points
    for point in &scatter_points {
        let series_key: &str = if has_series_dimension {
            point.series.as_deref().unwrap_or(SINGLE_SERIES_KEY)
        } else {
            SINGLE_SERIES_KEY
        };
        let series_index = *series_index_map.get(series_key).unwrap_or(&0);
        let series_style = styles.get(series_key);

        let point_size = if has_size_data {
            size_scale_fn(point.size)
        } else {
            resolve_point_size(series_style, config.f("pointSize"))
        };
        let palette = config.palette();
        // `point.category || point.label || String(point.x)`.
        let point_color_key = point
            .category
            .clone()
            .or_else(|| point.label.clone())
            .unwrap_or_else(|| js_num_to_string(point.x));
        let point_fill = spec
            .point_color(series_key, &point_color_key)
            .or_else(|| series_style.and_then(|s| s.s_truthy("pointFill")))
            .or(config.s("pointFill").filter(|s| !s.is_empty()))
            .map(str::to_string)
            .unwrap_or_else(|| resolve_palette_color(&palette, series_index).to_string());
        let point_stroke = series_style
            .and_then(|s| s.s_truthy("pointStroke"))
            .or(config.s("pointStroke").filter(|s| !s.is_empty()))
            .unwrap_or("");
        let point_stroke_width = series_style
            .and_then(|s| s.f_opt("pointStrokeWidth"))
            .unwrap_or(config.f("pointStrokeWidth"));
        let point_shape = resolve_point_shape(series_style, config.s("pointShape"));
        let point_opacity = config.f("pointOpacity");

        let mut metadata = Map::new();
        metadata.insert("chartRole".into(), json!("scatter-point"));
        metadata.insert("chartX".into(), json!(point.x));
        metadata.insert("chartY".into(), json!(point.y));
        metadata.insert("chartName".into(), json!(chart_name));
        if has_series_dimension {
            metadata.insert("chartSeries".into(), json!(series_key));
        }
        if let Some(size) = point.size {
            metadata.insert("chartSize".into(), json!(size));
        }
        let x_for_name = if x_scale_kind == "time" {
            scatter_time_name(point.x)
        } else {
            js_num_to_string(point.x)
        };
        let mut marker = get_point_marker_node(PointMarkerArgs {
            shape: point_shape,
            size: point_size,
            x: point.screen_x - point_size / 2.0,
            y: point.screen_y - point_size / 2.0,
            fill: &point_fill,
            stroke: point_stroke,
            stroke_width: point_stroke_width,
            name: format!(
                "{chart_name} Point ({x_for_name}, {})",
                js_num_to_string(point.y)
            ),
            metadata: Value::Object(metadata),
        });
        marker.set("opacity", json!(point_opacity));
        nodes.push(marker);
    }

    // Point labels
    if spec.b("showPointLabels") {
        for point in &scatter_points {
            let Some(label) = &point.label else { continue };
            let label_offset = 8.0;
            let mut d = Def::new("richtext");
            d.set("x", json!(point.screen_x + label_offset))
                .set("y", json!(point.screen_y - config.f("fontSize") / 2.0))
                .set("width", json!(100.0))
                .set("height", json!(config.f("fontSize") + 4.0))
                .set(
                    "htmlContent",
                    json!(build_styled_html(
                        &TextStyleProps {
                            font_size: raw_size(config.f("fontSize")),
                            font_family: config.s("fontFamily"),
                            text_align: Some("left"),
                            color: config.s("labelColor"),
                            ..Default::default()
                        },
                        &escape_html(label),
                    )),
                )
                .set("name", json!(format!("{chart_name} Point Label {label}")))
                .set(
                    "metadata",
                    json!({"chartRole": "point-label", "chartName": chart_name, "chartLabel": label}),
                );
            nodes.push(d);
        }
    }

    nodes.extend(generate_numeric_axis_nodes(
        &y_scale,
        y_display_range,
        config.f("yTickCount"),
        effective_y_format,
        None,
        "left",
        layout.plot_area,
        layout.y_axis_area,
        &axis_config_y,
        &font_config,
        chart_name,
        None,
    )?);
    // Time X axis routes tick labels through d3-time-format instead of the
    // numeric `effective_x_format` (scatterChart.ts:463-482).
    let x_time_formatter = if x_scale_kind == "time" {
        Some(create_time_tick_formatter(spec.s("xTimeFormat")))
    } else {
        None
    };
    nodes.extend(generate_numeric_axis_nodes(
        &x_scale,
        x_display_range,
        x_value_tick_count,
        effective_x_format,
        x_time_formatter.as_deref(),
        "bottom",
        layout.plot_area,
        layout.x_axis_area,
        &axis_config_x,
        &font_config,
        chart_name,
        None,
    )?);

    nodes.extend(generate_axis_title_nodes(
        spec.s("xLabel"),
        spec.s("yLabel"),
        &layout,
        &font_config,
        chart_name,
    ));

    // Legend
    if let Some(legend_area) = layout.legend_area {
        if has_series_dimension && !ordered_series.is_empty() {
            let palette = config.palette();
            let items: Vec<LegendItem> = ordered_series
                .iter()
                .enumerate()
                .map(|(index, name)| {
                    let style = styles.get(name);
                    let fill = style
                        .and_then(|s| s.s_truthy("pointFill"))
                        .or_else(|| style.and_then(|s| s.s_truthy("fill")))
                        .map(str::to_string)
                        .unwrap_or_else(|| resolve_palette_color(&palette, index).to_string());
                    LegendItem {
                        name: name.clone(),
                        fill,
                        stroke: style.and_then(|s| s.s("stroke")).map(str::to_string),
                        stroke_width: style.and_then(|s| s.f_opt("strokeWidth")),
                        opacity: style.and_then(|s| s.f_opt("opacity")),
                        dash: None,
                    }
                })
                .collect();
            nodes.extend(generate_legend_nodes(
                &items,
                config,
                legend_area,
                chart_name,
                Some("circle"),
            ));
        }
    }

    Ok(nodes)
}

fn add_grid_axis(d: &mut Def, axis: &str) {
    if let Some(Value::Object(meta)) = d.props.get_mut("metadata") {
        meta.insert("gridAxis".into(), json!(axis));
    }
}

/// The x interval over which `y = slope·x + intercept` stays inside `y_range`, within `x_range`; `None` when the line
/// never enters the plot (or only touches it at one point).
fn linear_span_in_range(
    slope: f64,
    intercept: f64,
    x_range: (f64, f64),
    y_range: (f64, f64),
) -> Option<(f64, f64)> {
    let (y_lo, y_hi) = y_range;
    let mut from = x_range.0.min(x_range.1);
    let mut to = x_range.0.max(x_range.1);
    if slope == 0.0 {
        if intercept < y_lo || intercept > y_hi {
            return None;
        }
    } else {
        let a = (y_lo - intercept) / slope;
        let b = (y_hi - intercept) / slope;
        from = from.max(a.min(b));
        to = to.min(a.max(b));
    }
    (from.is_finite() && to.is_finite() && to > from).then_some((from, to))
}

#[allow(clippy::too_many_arguments)]
fn generate_trend_line_nodes(
    points: &[ScatterPoint],
    plot: Bounds,
    x_scale: &ValueScale,
    y_scale: &ValueScale,
    x_display_range: (f64, f64),
    y_display_range: (f64, f64),
    spec: &SpecView,
    chart_name: &str,
    x_scale_kind: &str,
    y_scale_kind: &str,
) -> Vec<Def> {
    let mut nodes = Vec::new();
    if points.len() < 2 {
        return nodes;
    }
    let pts: Vec<(f64, f64)> = points.iter().map(|p| (p.x, p.y)).collect();
    let Some(reg) = linear_regression(&pts) else {
        return nodes;
    };
    let (slope, intercept) = (reg.slope, reg.intercept);
    let (x_start, x_end) = x_display_range;

    let trend_color = spec.s("trendColor").unwrap_or("#94a3b8");
    let trend_style = spec.s("trendStyle").unwrap_or("dashed");
    let dash = resolve_dash_pattern(Some(trend_style));

    if x_scale_kind == "log" || y_scale_kind == "log" {
        let (y_lo, y_hi) = y_display_range;
        const SAMPLE_COUNT: usize = 64;
        let log_x = x_scale_kind == "log";
        let x_start_t = if log_x { x_start.log10() } else { x_start };
        let x_end_t = if log_x { x_end.log10() } else { x_end };
        let mut samples: Vec<PathPt> = Vec::new();
        for i in 0..=SAMPLE_COUNT {
            let t_x = x_start_t + ((x_end_t - x_start_t) * i as f64) / SAMPLE_COUNT as f64;
            let x_val = if log_x { 10f64.powf(t_x) } else { t_x };
            let y_val = slope * x_val + intercept;
            if !y_val.is_finite() || y_val < y_lo || y_val > y_hi {
                continue;
            }
            let px = x_scale.scale(x_val);
            let py = plot.height - y_scale.scale(y_val);
            if samples.is_empty() {
                samples.push(PathPt::Move { x: px, y: py });
            } else {
                samples.push(PathPt::Line { x: px, y: py });
            }
        }
        if samples.len() < 2 {
            return nodes;
        }
        let mut d = Def::new("path");
        d.set("x", json!(plot.x))
            .set("y", json!(plot.y))
            .set("points", path_pts_to_value(&samples))
            .stroke_color(json!(trend_color))
            .stroke_width(json!(2.0))
            .stroke_dash(dash_value(&dash))
            .fill_color(json!("transparent"))
            .set("closed", json!(false))
            .set("name", json!(format!("{chart_name} Trend Line")))
            .set(
                "metadata",
                json!({
                    "chartRole": "trend-line",
                    "chartName": chart_name,
                    "slope": slope,
                    "intercept": intercept,
                    "strokeStyle": trend_style,
                }),
            );
        nodes.push(d);
        return nodes;
    }

    // Clipped to the plot: across the x domain, cut where the line leaves the y range (a steep fit would otherwise
    // run past the axes).
    let Some((x_from, x_to)) =
        linear_span_in_range(slope, intercept, x_display_range, y_display_range)
    else {
        return nodes;
    };
    let y1 = slope * x_from + intercept;
    let y2 = slope * x_to + intercept;
    let screen_x1 = plot.x + x_scale.scale(x_from);
    let screen_y1 = plot.y + plot.height - y_scale.scale(y1);
    let screen_x2 = plot.x + x_scale.scale(x_to);
    let screen_y2 = plot.y + plot.height - y_scale.scale(y2);
    let mut d = Def::new("line");
    d.set(
        "points",
        json!([screen_x1, screen_y1, screen_x2, screen_y2]),
    )
    .stroke_color(json!(trend_color))
    .stroke_width(json!(2.0))
    .stroke_dash(dash_value(&dash))
    .set("name", json!(format!("{chart_name} Trend Line")))
    .set(
        "metadata",
        json!({
            "chartRole": "trend-line",
            "chartName": chart_name,
            "slope": slope,
            "intercept": intercept,
            "strokeStyle": trend_style,
        }),
    );
    nodes.push(d);
    nodes
}

#[allow(clippy::too_many_arguments)]
fn generate_quadrant_nodes(
    spec: &SpecView,
    data: &DataView,
    plot: Bounds,
    x_scale: &ValueScale,
    y_scale: &ValueScale,
    x_display_range: (f64, f64),
    y_display_range: (f64, f64),
    chart_name: &str,
    config: ConfigView,
    x_scale_kind: &str,
    y_scale_kind: &str,
) -> Vec<Def> {
    let mut nodes = Vec::new();
    let mut quadrant_x = spec.f_opt("quadrantX");
    let mut quadrant_y = spec.f_opt("quadrantY");

    if quadrant_x.is_none() || quadrant_y.is_none() {
        let mut x_values: Vec<f64> = data.rows.iter().filter_map(|r| r.f_opt("x")).collect();
        let mut y_values: Vec<f64> = data.rows.iter().filter_map(|r| r.f_opt("y")).collect();
        x_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        y_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if quadrant_x.is_none() && !x_values.is_empty() {
            quadrant_x = Some(x_values[x_values.len() / 2]);
        }
        if quadrant_y.is_none() && !y_values.is_empty() {
            quadrant_y = Some(y_values[y_values.len() / 2]);
        }
    }
    let (Some(quadrant_x), Some(quadrant_y)) = (quadrant_x, quadrant_y) else {
        return nodes;
    };
    if x_scale_kind == "log"
        && (quadrant_x <= 0.0 || quadrant_x < x_display_range.0 || quadrant_x > x_display_range.1)
    {
        return nodes;
    }
    if y_scale_kind == "log"
        && (quadrant_y <= 0.0 || quadrant_y < y_display_range.0 || quadrant_y > y_display_range.1)
    {
        return nodes;
    }

    let screen_x = plot.x + x_scale.scale(quadrant_x);
    let screen_y = plot.y + plot.height - y_scale.scale(quadrant_y);
    let quadrant_color = "#94a3b8";
    let quadrant_dash = resolve_dash_pattern(Some("dashed"));

    let mut vline = Def::new("line");
    vline
        .set("points", json!([screen_x, plot.y, screen_x, plot.y + plot.height]))
        .stroke_color(json!(quadrant_color))
        .stroke_width(json!(1.5))
        .stroke_dash(dash_value(&quadrant_dash))
        .set("name", json!(format!("{chart_name} Quadrant X Line")))
        .set(
            "metadata",
            json!({"chartRole": "quadrant-line", "chartName": chart_name, "quadrantAxis": "x", "quadrantValue": quadrant_x}),
        );
    nodes.push(vline);

    let mut hline = Def::new("line");
    hline
        .set("points", json!([plot.x, screen_y, plot.x + plot.width, screen_y]))
        .stroke_color(json!(quadrant_color))
        .stroke_width(json!(1.5))
        .stroke_dash(dash_value(&quadrant_dash))
        .set("name", json!(format!("{chart_name} Quadrant Y Line")))
        .set(
            "metadata",
            json!({"chartRole": "quadrant-line", "chartName": chart_name, "quadrantAxis": "y", "quadrantValue": quadrant_y}),
        );
    nodes.push(hline);

    if spec.b("quadrantLabels") {
        let label_padding = 8.0;
        let label_font_size = config.f("fontSize") - 2.0;
        let labels: [(&str, f64, f64, &str); 4] = [
            (
                "top-right",
                screen_x + label_padding,
                plot.y + label_padding,
                spec.s("quadrantLabelTopRight").unwrap_or("Q1"),
            ),
            (
                "top-left",
                plot.x + label_padding,
                plot.y + label_padding,
                spec.s("quadrantLabelTopLeft").unwrap_or("Q2"),
            ),
            (
                "bottom-left",
                plot.x + label_padding,
                screen_y + label_padding,
                spec.s("quadrantLabelBottomLeft").unwrap_or("Q3"),
            ),
            (
                "bottom-right",
                screen_x + label_padding,
                screen_y + label_padding,
                spec.s("quadrantLabelBottomRight").unwrap_or("Q4"),
            ),
        ];
        for (position, x, y, text) in labels {
            let mut d = Def::new("richtext");
            d.set("x", json!(x))
                .set("y", json!(y))
                .set("width", json!(100.0))
                .set("height", json!(label_font_size + 4.0))
                .set(
                    "htmlContent",
                    json!(build_styled_html(
                        &TextStyleProps {
                            font_size: Some(label_font_size),
                            font_family: config.s("fontFamily"),
                            text_align: Some("left"),
                            color: Some(quadrant_color),
                            ..Default::default()
                        },
                        &escape_html(text),
                    )),
                )
                .set("name", json!(format!("{chart_name} Quadrant Label {position}")))
                .set(
                    "metadata",
                    json!({"chartRole": "quadrant-label", "chartName": chart_name, "quadrantPosition": position}),
                );
            nodes.push(d);
        }
    }
    nodes
}

#[cfg(test)]
mod trend_line_tests {
    use super::linear_span_in_range;

    // Mirrors packages/editor/test/chartTrendLineClip.test.ts: the trend line is cut where it leaves the y range.
    #[test]
    fn trend_line_span_stays_in_the_y_range() {
        assert_eq!(
            linear_span_in_range(1.0, 0.0, (0.0, 10.0), (0.0, 5.0)),
            Some((0.0, 5.0))
        );
        assert_eq!(
            linear_span_in_range(-2.0, 10.0, (0.0, 10.0), (0.0, 10.0)),
            Some((0.0, 5.0))
        );
        assert_eq!(
            linear_span_in_range(0.0, 3.0, (0.0, 10.0), (0.0, 5.0)),
            Some((0.0, 10.0))
        );
        assert_eq!(
            linear_span_in_range(0.0, 7.0, (0.0, 10.0), (0.0, 5.0)),
            None
        );
        assert_eq!(
            linear_span_in_range(1.0, 100.0, (0.0, 10.0), (0.0, 5.0)),
            None
        );
    }
}
