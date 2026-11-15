//! Port of `utils/markup/elements/charts/generators/areaChart.ts`, including
//! the `spec.xScale === 'time'` arm (a d3-time X scale, with chronological
//! point reordering so fills stay valid polygons; see `timeticks`/`timefmt`).

use super::ticks::TickError;
use crate::value::{json, Map, Value};
use std::collections::HashMap;

use super::axes::{
    create_axis_and_font_config, dash_value, generate_categorical_axis_nodes, generate_axis_title_nodes, generate_grid_nodes,
    generate_numeric_axis_nodes, resolve_dash_pattern, GridConfig,
};
use super::cfg::{resolve_chart_dimensions, DataView, SeriesStyles, SpecView, SINGLE_SERIES_KEY};
use super::colorutil::{resolve_palette_color, resolve_shadow_props};
use super::curves::{generate_curve_points, path_pts_to_value, PathPt};
use super::defs::Def;
use super::fills::parse_fill;
use super::format::{create_time_tick_formatter, get_effective_axis_format};
use super::jsmath::{js_max, js_min};
use super::layout::calculate_chart_layout;
use super::parts::{
    append_unstacked_area_series, chart_point_groups, generate_background_node,
    generate_legend_nodes, generate_series_point_markers, generate_source_footer_nodes,
    generate_title_nodes, resolve_series_curve, resolve_series_fill_opacity,
    resolve_series_show_points, AreaFillOrder, LegendItem, SeriesPoint,
};
use super::scales::{
    create_linear_scale, create_log_scale, create_time_scale, get_display_range_from_ticks,
    is_log_domain_valid, CategoricalScale,
};
use super::stack::compute_stack;
use super::value_axis_bounds::{
    apply_value_axis_bounds, clamp_scale_to_plot, resolve_value_axis_bounds,
};

/// `generateStackedAreaPathPoints` (areaChart.ts:49-67).
fn generate_stacked_area_path_points(
    top: &[(f64, f64)],
    bottom: &[(f64, f64)],
    curve: &str,
) -> Vec<PathPt> {
    if top.is_empty() {
        return Vec::new();
    }
    let top_points = generate_curve_points(top, curve);
    let reversed_bottom: Vec<(f64, f64)> = bottom.iter().rev().copied().collect();
    let bottom_curve = generate_curve_points(&reversed_bottom, curve);
    let mut out = top_points;
    for (i, pt) in bottom_curve.into_iter().enumerate() {
        if i == 0 {
            // replace the leading move with a line
            out.push(match pt {
                PathPt::Move { x, y } => PathPt::Line { x, y },
                other => other,
            });
        } else {
            out.push(pt);
        }
    }
    out
}

struct AreaEntry {
    value: f64,
    #[allow(dead_code)]
    label: Option<String>,
    x_time: Option<f64>,
}

#[derive(Clone, Copy)]
struct StackedValue {
    bottom: f64,
    top: f64,
    value: f64,
}

pub fn generate_area_chart_nodes(
    spec: &SpecView,
    data: &DataView,
    styles: SeriesStyles,
    chart_name: &str,
) -> Result<Vec<Def>, TickError> {
    let config = spec.config;
    let mut nodes: Vec<Def> = Vec::new();

    let is_stacked = matches!(spec.s("mode"), Some("stacked") | Some("stacked-percent"));
    let is_stacked_percent = spec.s("mode") == Some("stacked-percent");

    // areaChart.ts mutates config.yFormat in place for stacked-percent.
    let y_format: Option<&str> = if is_stacked_percent && config.s("yFormat") == Some("auto") {
        Some("percent")
    } else {
        config.s("yFormat")
    };
    let baseline = config.f("baseline");

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

    let mut values_by_category: HashMap<String, HashMap<String, AreaEntry>> = HashMap::new();
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
                AreaEntry {
                    value: row.f("value"),
                    label: row.s_truthy("label").map(str::to_string),
                    x_time: row.f_opt("xTime"),
                },
            );
    }

    // Stacked values
    let mut stacked_values: HashMap<String, HashMap<String, StackedValue>> = HashMap::new();
    if is_stacked {
        let mut zero_total_percent: Vec<&String> = Vec::new();
        if is_stacked_percent {
            for category in &dims.ordered_categories {
                let cat_map = values_by_category.get(category);
                let mut category_total = 0.0;
                for series_key in &dims.series_keys {
                    category_total += cat_map
                        .and_then(|m| m.get(series_key.as_str()))
                        .map(|e| e.value)
                        .unwrap_or(0.0);
                }
                if category_total.abs() < f64::EPSILON {
                    zero_total_percent.push(category);
                }
            }
        }
        let stacked = compute_stack(
            &dims.ordered_categories,
            &dims.series_keys,
            |category, series_key| {
                values_by_category
                    .get(category)
                    .and_then(|m| m.get(series_key))
                    .map(|e| e.value)
            },
            if is_stacked_percent {
                "stacked-percent"
            } else {
                "stacked"
            },
            "running-sum",
        );
        for category in &dims.ordered_categories {
            let mut stack_map = HashMap::new();
            for series_key in &dims.series_keys {
                if zero_total_percent.contains(&category) {
                    stack_map.insert(
                        series_key.clone(),
                        StackedValue {
                            bottom: 0.0,
                            top: 0.0,
                            value: 0.0,
                        },
                    );
                    continue;
                }
                if let Some(segment) = stacked
                    .get(category.as_str())
                    .and_then(|m| m.get(series_key.as_str()))
                {
                    stack_map.insert(
                        series_key.clone(),
                        StackedValue {
                            bottom: segment.y0,
                            top: segment.y1,
                            value: segment.value,
                        },
                    );
                }
            }
            stacked_values.insert(category.clone(), stack_map);
        }
    }

    // Y domain
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    if is_stacked {
        for category in &dims.ordered_categories {
            let Some(stack_map) = stacked_values.get(category) else {
                continue;
            };
            for s in stack_map.values() {
                min = js_min(min, s.bottom);
                max = js_max(max, s.top);
            }
        }
    } else {
        for category in &dims.ordered_categories {
            let Some(cat_map) = values_by_category.get(category) else {
                continue;
            };
            for entry in cat_map.values() {
                min = js_min(min, entry.value);
                max = js_max(max, entry.value);
            }
        }
    }
    if !min.is_finite() || !max.is_finite() {
        min = 0.0;
        max = 0.0;
    }

    // Authored `y-min`/`y-max` replace the matching end up front so the
    // log-validity check, the baseline extension and the scale all see the
    // window that will actually be drawn (valueAxisBounds.ts).
    let y_bounds = resolve_value_axis_bounds(spec.f_opt("yMin"), spec.f_opt("yMax"), (min, max));
    let (bounded_min, bounded_max) = apply_value_axis_bounds((min, max), &y_bounds);
    min = bounded_min;
    max = bounded_max;

    let effective_scale = if !is_stacked_percent
        && !is_stacked
        && config.s("yScale") == Some("log")
        && is_log_domain_valid(min, max)
    {
        "log"
    } else {
        "linear"
    };
    let log_baseline_is_valid = effective_scale == "log" && baseline > 0.0;

    // A pinned end is the author's floor/ceiling: neither zero nor the
    // baseline may widen it back out.
    if is_stacked_percent {
        min = y_bounds.min.unwrap_or(0.0);
        max = y_bounds.max.unwrap_or(100.0);
    } else {
        if !is_stacked && (effective_scale != "log" || log_baseline_is_valid) {
            if y_bounds.min.is_none() {
                min = js_min(min, baseline);
            }
            if y_bounds.max.is_none() {
                max = js_max(max, baseline);
            }
        }
        if effective_scale != "log" {
            if y_bounds.min.is_none() {
                min = js_min(min, 0.0);
            }
            if y_bounds.max.is_none() {
                max = js_max(max, 0.0);
            }
        }
    }

    // Re-pin after nicing, which would round `y-min = 3` back out to 0.
    let display_range = apply_value_axis_bounds(
        get_display_range_from_ticks((min, max), config.f("yTickCount"), effective_scale),
        &y_bounds,
    );
    let (_, effective_y_format) =
        get_effective_axis_format(display_range, config.f("yTickCount"), y_format)?;

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
    let baseline_y_pixel = layout.plot_area.height - y_scale.scale(baseline_value);
    let band_width = categorical_x_scale.bandwidth();

    // Time X axis (areaChart.ts:133, 282-286): swap the band scale for a time
    // scale when x-scale="time" and the data carries a parsed time range.
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
        let fill_color = series_style
            .and_then(|s| s.s_truthy("fill"))
            .map(str::to_string)
            .unwrap_or_else(|| resolve_palette_color(&palette, series_index).to_string());
        let fill_opacity = resolve_series_fill_opacity(series_style, config.f("fillOpacity"));
        // not-a-node-key: the chart spec's series style and config `stroke`
        let stroke = if series_style.map(|s| s.has("stroke")).unwrap_or(false) {
            series_style.and_then(|s| s.s("stroke"))
        } else {
            config.s("stroke")
        };
        let stroke_truthy = stroke.filter(|s| !s.is_empty());
        let stroke_width = stroke_truthy.map(|_| {
            series_style
                .and_then(|s| s.f_opt("strokeWidth"))
                .unwrap_or(config.f("strokeWidth"))
        });
        let stroke_style = series_style
            .and_then(|s| s.s("strokeStyle"))
            .or(config.s("strokeStyle"));
        let dash = if stroke_truthy.is_some() {
            resolve_dash_pattern(stroke_style)
        } else {
            None
        };
        let opacity = series_style.and_then(|s| s.f_opt("opacity"));
        let curve = resolve_series_curve(series_style, config.s("curve"));

        let mut series_points: Vec<SeriesPoint> = Vec::new();
        let mut local_top_points: Vec<(f64, f64)> = Vec::new();
        let mut local_bottom_points: Vec<(f64, f64)> = Vec::new();

        for category in &dims.ordered_categories {
            let row_entry = values_by_category
                .get(category)
                .and_then(|m| m.get(series_key.as_str()));
            // Time axis: use the parsed xTime directly (no band offset); skip
            // points without a parsed date (areaChart.ts:341-351). Categorical
            // axis centers in the band.
            let x = if let Some(ts) = &time_x_scale {
                match row_entry.and_then(|e| e.x_time) {
                    Some(xt) => ts.scale(xt),
                    None => continue,
                }
            } else {
                categorical_x_scale.scale(category) + band_width / 2.0
            };
            if is_stacked {
                let Some(stacked) = stacked_values
                    .get(category)
                    .and_then(|m| m.get(series_key.as_str()))
                else {
                    continue;
                };
                let top_y = layout.plot_area.height - y_scale.scale(stacked.top);
                let bottom_y = layout.plot_area.height - y_scale.scale(stacked.bottom);
                series_points.push(SeriesPoint {
                    category: category.clone(),
                    value: stacked.value,
                    x: layout.plot_area.x + x,
                    y: layout.plot_area.y + top_y,
                    x_time: row_entry.and_then(|e| e.x_time),
                });
                local_top_points.push((x, top_y));
                local_bottom_points.push((x, bottom_y));
            } else {
                let Some(entry) = row_entry else { continue };
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
        }

        // On time axes, rows may arrive out of chronological order. Area fills
        // must trace top points left-to-right (and bottom right-to-left),
        // otherwise the polygon self-intersects. Reorder all three parallel
        // arrays by pixel-x ascending (areaChart.ts:388-405).
        if time_x_scale.is_some() && local_top_points.len() > 1 {
            let mut indices: Vec<usize> = (0..local_top_points.len()).collect();
            indices.sort_by(|&a, &b| {
                local_top_points[a]
                    .0
                    .partial_cmp(&local_top_points[b].0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            series_points = indices.iter().map(|&i| series_points[i].clone()).collect();
            local_top_points = indices.iter().map(|&i| local_top_points[i]).collect();
            if is_stacked {
                local_bottom_points = indices.iter().map(|&i| local_bottom_points[i]).collect();
            }
        }

        let point_groups = chart_point_groups(
            &series_points,
            data,
            series_key,
            dims.has_series_dimension,
            &dims.ordered_categories,
        );
        if !local_top_points.is_empty() {
            let series_name = if dims.has_series_dimension {
                series_key.as_str()
            } else {
                "Series"
            };
            let base_name = format!("{chart_name} Area {series_name}");
            let shadow_props = resolve_shadow_props(series_style.and_then(|s| s.s("shadow")));
            if is_stacked {
                let path_points: Vec<PathPt> = point_groups
                    .iter()
                    .flat_map(|g| {
                        let top: Vec<(f64, f64)> = g.iter().map(|&i| local_top_points[i]).collect();
                        let bottom: Vec<(f64, f64)> =
                            g.iter().map(|&i| local_bottom_points[i]).collect();
                        generate_stacked_area_path_points(&top, &bottom, curve)
                    })
                    .collect();
                let mut metadata = Map::new();
                metadata.insert("chartRole".into(), json!("area-series"));
                if dims.has_series_dimension {
                    metadata.insert("chartSeries".into(), json!(series_key));
                }
                metadata.insert("chartName".into(), json!(chart_name));
                let mut d = Def::new("path");
                d.set("x", json!(layout.plot_area.x))
                    .set("y", json!(layout.plot_area.y))
                    .set("points", path_pts_to_value(&path_points))
                    .stroke_color(stroke_truthy.map(|s| json!(s)).unwrap_or(Value::Null),
                    )
                    .stroke_width(stroke_width.map(|w| json!(w)).unwrap_or(Value::Null),
                    )
                    .stroke_dash(dash_value(&dash))
                    .stroke_cap(config
                            .s("strokeCap")
                            .map(|v| json!(v))
                            .unwrap_or(Value::Null),
                    )
                    .stroke_join(config
                            .s("strokeJoin")
                            .map(|v| json!(v))
                            .unwrap_or(Value::Null),
                    )
                    .set("fillOpacity", json!(fill_opacity))
                    .set("closed", json!(true))
                    .set("opacity", opacity.map(|v| json!(v)).unwrap_or(Value::Null))
                    .set("name", json!(base_name))
                    .set("metadata", Value::Object(metadata));
                for (k, v) in &parse_fill(&fill_color) {
                    d.set(k, v.clone());
                }
                for (k, v) in &shadow_props {
                    d.set(k, v.clone());
                }
                nodes.push(d);
            } else {
                let mut metadata = Map::new();
                metadata.insert("chartRole".into(), json!("area-series"));
                if dims.has_series_dimension {
                    metadata.insert("chartSeries".into(), json!(series_key));
                }
                metadata.insert("chartName".into(), json!(chart_name));
                append_unstacked_area_series(
                    &mut nodes,
                    &local_top_points,
                    &point_groups,
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
                    AreaFillOrder::AfterTail,
                );
            }
        }

        let show_points = resolve_series_show_points(series_style, config.b("showPoints"));
        if show_points {
            nodes.extend(generate_series_point_markers(
                &series_points,
                series_style,
                config,
                series_key,
                dims.has_series_dimension,
                chart_name,
                &fill_color,
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

    let x_axis_baseline_value = if effective_scale == "log" {
        display_range.0
    } else {
        0.0
    };
    let zero_y =
        layout.plot_area.y + layout.plot_area.height - y_scale.scale(x_axis_baseline_value);
    // Time X axis: multi-scale (or xTimeFormat) labels, anchored at the data-
    // space baseline like the categorical fallback (areaChart.ts:558-578).
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
            Some(zero_y),
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
            Some(zero_y),
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
                    let fill_color = style
                        .and_then(|s| s.s_truthy("fill"))
                        .map(str::to_string)
                        .unwrap_or_else(|| resolve_palette_color(&palette, index).to_string());
                    LegendItem {
                        name: name.clone(),
                        fill: fill_color,
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
                Some("square"),
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
