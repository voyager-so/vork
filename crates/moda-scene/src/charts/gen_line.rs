//! Port of `utils/markup/elements/charts/generators/lineChart.ts`, including
//! the `spec.xScale === 'time'` arm (a d3-time X scale instead of a band
//! scale; see `timeticks`/`timefmt`).

use super::ticks::TickError;
use crate::value::{json, Map};
use std::collections::HashMap;

use super::axes::{
    create_axis_and_font_config, generate_categorical_axis_nodes, generate_axis_title_nodes, generate_grid_nodes,
    generate_numeric_axis_nodes, resolve_dash_pattern, GridConfig,
};
use super::cfg::{resolve_chart_dimensions, DataView, SeriesStyles, SpecView, SINGLE_SERIES_KEY};
use super::colorutil::{resolve_palette_color, resolve_shadow_props};
use super::defs::Def;
use super::format::{create_time_tick_formatter, get_effective_axis_format};
use super::jsmath::{js_max, js_min};
use super::layout::calculate_chart_layout;
use super::parts::{
    append_line_series_path, chart_point_groups, generate_background_node, generate_legend_nodes,
    generate_series_point_markers, generate_source_footer_nodes, generate_title_nodes,
    resolve_series_curve, resolve_series_show_points, LegendItem, SeriesPoint,
};
use super::scales::{
    create_linear_scale, create_log_scale, create_time_scale, get_display_range_from_ticks,
    is_log_domain_valid, CategoricalScale,
};
use super::value_axis_bounds::{
    apply_value_axis_bounds, clamp_scale_to_plot, resolve_value_axis_bounds,
};

struct LineEntry {
    value: f64,
    #[allow(dead_code)]
    label: Option<String>,
    x_time: Option<f64>,
}

pub fn generate_line_chart_nodes(
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

    let dims = resolve_chart_dimensions(data, spec);
    if dims.ordered_categories.is_empty() {
        return Ok(nodes);
    }

    let mut values_by_category: HashMap<String, HashMap<String, LineEntry>> = HashMap::new();
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
                LineEntry {
                    value: row.f("value"),
                    label: row.s_truthy("label").map(str::to_string),
                    x_time: row.f_opt("xTime"),
                },
            );
    }

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for category in &dims.ordered_categories {
        let Some(cat_map) = values_by_category.get(category) else {
            continue;
        };
        for entry in cat_map.values() {
            min = js_min(min, entry.value);
            max = js_max(max, entry.value);
        }
    }
    if !min.is_finite() || !max.is_finite() {
        min = 0.0;
        max = 0.0;
    }
    let baseline = config.f("baseline");
    // Authored `y-min`/`y-max` replace the matching end up front so the
    // log-validity check, the baseline extension and the scale all see the
    // window that will actually be drawn (valueAxisBounds.ts).
    let y_bounds = resolve_value_axis_bounds(spec.f_opt("yMin"), spec.f_opt("yMax"), (min, max));
    let (bounded_min, bounded_max) = apply_value_axis_bounds((min, max), &y_bounds);
    min = bounded_min;
    max = bounded_max;
    let effective_scale = if config.s("yScale") == Some("log") && is_log_domain_valid(min, max) {
        "log"
    } else {
        "linear"
    };
    let log_baseline_is_valid = effective_scale == "log" && baseline > 0.0;
    // A pinned end is the author's floor/ceiling: neither zero nor the
    // baseline may widen it back out.
    if effective_scale != "log" {
        if y_bounds.min.is_none() {
            min = js_min(js_min(min, 0.0), baseline);
        }
        if y_bounds.max.is_none() {
            max = js_max(js_max(max, 0.0), baseline);
        }
    } else if log_baseline_is_valid {
        if y_bounds.min.is_none() {
            min = js_min(min, baseline);
        }
        if y_bounds.max.is_none() {
            max = js_max(max, baseline);
        }
    }
    // Re-pin after nicing, which would round `y-min = 3` back out to 0.
    let display_range = apply_value_axis_bounds(
        get_display_range_from_ticks((min, max), config.f("yTickCount"), effective_scale),
        &y_bounds,
    );
    let (_, effective_y_format) =
        get_effective_axis_format(display_range, config.f("yTickCount"), config.s("yFormat"))?;

    let categorical_x_scale =
        CategoricalScale::new(&dims.ordered_categories, layout.plot_area.width, 0.0);
    let y_scale = clamp_scale_to_plot(
        if effective_scale == "log" {
            create_log_scale(display_range, layout.plot_area.height)
        } else {
            create_linear_scale(display_range, layout.plot_area.height)
        },
        layout.plot_area.height,
        &y_bounds,
        display_range,
    );
    let baseline_value = if effective_scale == "log" {
        if log_baseline_is_valid {
            baseline
        } else {
            display_range.0
        }
    } else {
        baseline
    };
    let baseline_y = layout.plot_area.y + layout.plot_area.height - y_scale.scale(baseline_value);
    let band_width = categorical_x_scale.bandwidth();

    // Time X axis (lineChart.ts:82, 142-146): when x-scale="time" and the data
    // carries a parsed time range, the X axis becomes a numeric time scale
    // (ms-since-epoch) snapped to nice date boundaries instead of a band scale.
    let is_time_axis = spec.s("xScale") == Some("time") && data.time_range().is_some();
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

    nodes.extend(generate_grid_nodes(
        &y_scale,
        display_range,
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
    let palette = config.palette();

    for (series_index, series_key) in dims.series_keys.iter().enumerate() {
        let series_style = styles.get(series_key);
        let stroke = series_style
            .and_then(|s| s.s_truthy("stroke"))
            .map(str::to_string)
            .unwrap_or_else(|| resolve_palette_color(&palette, series_index).to_string());
        let stroke_width = series_style
            .and_then(|s| s.f_opt("strokeWidth"))
            .unwrap_or(config.f("strokeWidth"));
        let stroke_style = series_style
            .and_then(|s| s.s("strokeStyle"))
            .or(config.s("strokeStyle"));
        let dash = resolve_dash_pattern(stroke_style);
        let opacity = series_style.and_then(|s| s.f_opt("opacity"));
        let curve = resolve_series_curve(series_style, config.s("curve"));

        let mut series_points: Vec<SeriesPoint> = Vec::new();
        for category in &dims.ordered_categories {
            let Some(entry) = values_by_category
                .get(category)
                .and_then(|m| m.get(series_key.as_str()))
            else {
                continue;
            };
            // Time axis: x is the pixel offset of the date value directly (no
            // band-center offset). Falls back to the band center when a point
            // has no parsed xTime (lineChart.ts:194-201).
            let x = match (&time_x_scale, entry.x_time) {
                (Some(ts), Some(xt)) => layout.plot_area.x + ts.scale(xt),
                _ => layout.plot_area.x + categorical_x_scale.scale(category) + band_width / 2.0,
            };
            let y = layout.plot_area.y + layout.plot_area.height - y_scale.scale(entry.value);
            series_points.push(SeriesPoint {
                category: category.clone(),
                value: entry.value,
                x,
                y,
                x_time: entry.x_time,
            });
        }

        // Time series may arrive out of chronological order; draw left-to-right
        // (lineChart.ts:207-209). Category order is preserved otherwise.
        if time_x_scale.is_some() {
            series_points
                .sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
        }

        let local_points: Vec<(f64, f64)> = series_points
            .iter()
            .map(|p| (p.x - layout.plot_area.x, p.y - layout.plot_area.y))
            .collect();
        let mut metadata = Map::new();
        metadata.insert("chartRole".into(), json!("line-series"));
        if dims.has_series_dimension {
            metadata.insert("chartSeries".into(), json!(series_key));
        }
        metadata.insert("chartName".into(), json!(chart_name));
        let groups = chart_point_groups(
            &series_points,
            data,
            series_key,
            dims.has_series_dimension,
            &dims.ordered_categories,
        );
        append_line_series_path(
            &mut nodes,
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
            format!("{chart_name} Line {series_key}"),
            metadata,
            &resolve_shadow_props(series_style.and_then(|s| s.s("shadow"))),
        );

        let show_points = resolve_series_show_points(series_style, config.b("showPoints"));
        if show_points {
            nodes.extend(generate_series_point_markers(
                &series_points,
                series_style,
                config,
                series_key,
                dims.has_series_dimension,
                chart_name,
                &stroke,
                spec.point_colors(),
                data.category_labels(),
            ));
        }
    }

    nodes.extend(generate_numeric_axis_nodes(
        &y_scale,
        display_range,
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
    // Time X axis uses d3-scaleUtc's multi-scale formatter by default (a
    // specifier per tick); spec.xTimeFormat overrides with one specifier for
    // every tick. The axis line/ticks anchor at the Y-space baseline, matching
    // the categorical path (lineChart.ts:273-314).
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
            Some(baseline_y),
        )?);
    } else {
        nodes.extend(generate_categorical_axis_nodes(
            &dims.ordered_categories,
            &categorical_x_scale,
            "bottom",
            layout.plot_area,
            layout.x_axis_area,
            &axis_config_x,
            &font_config,
            chart_name,
            Some(baseline_y),
            data.category_labels(),
        ));
    }

    if let Some(legend_area) = layout.legend_area {
        let series_names: &[String] = if dims.has_series_dimension {
            &dims.ordered_series
        } else {
            &[]
        };
        if !series_names.is_empty() {
            let items: Vec<LegendItem> = series_names
                .iter()
                .enumerate()
                .map(|(index, name)| {
                    let style = styles.get(name);
                    let stroke = style
                        .and_then(|s| s.s_truthy("stroke"))
                        .map(str::to_string)
                        .unwrap_or_else(|| resolve_palette_color(&palette, index).to_string());
                    LegendItem {
                        name: name.clone(),
                        fill: style
                            .and_then(|s| s.s_truthy("fill"))
                            .map(str::to_string)
                            .unwrap_or(stroke),
                        stroke: style.and_then(|s| s.s("stroke")).map(str::to_string),
                        stroke_width: style.and_then(|s| s.f_opt("strokeWidth")),
                        opacity: style.and_then(|s| s.f_opt("opacity")),
                        dash: resolve_dash_pattern(
                            style
                                .and_then(|s| s.s("strokeStyle"))
                                .or(config.s("strokeStyle")),
                        ),
                    }
                })
                .collect();
            nodes.extend(generate_legend_nodes(
                &items,
                config,
                legend_area,
                chart_name,
                config.s("legendMarker"),
            ));
        }
    }

    nodes.extend(generate_axis_title_nodes(
        spec.s("xLabel"),
        spec.s("yLabel"),
        &layout,
        &font_config,
        chart_name,
    ));

    Ok(nodes)
}
