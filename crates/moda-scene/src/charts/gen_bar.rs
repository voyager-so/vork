//! Port of `utils/markup/elements/charts/generators/barChart.ts`.

use super::ticks::TickError;
use super::paint_bag;
use crate::value::{json, Map, Value};
use std::collections::HashMap;

use super::axes::{
    create_axis_and_font_config, fit_bottom_tick_count, generate_categorical_axis_nodes,
    generate_axis_title_nodes, generate_grid_nodes, generate_numeric_axis_nodes, GridConfig,
};
use super::cfg::{
    aggregate_bar_values, bar_value_domain, resolve_bar_stacking, resolve_chart_dimensions,
    DataView, SeriesStyles, SpecView,
};
use super::colorutil::{get_auto_contrast_color, resolve_palette_color, resolve_shadow_props};
use super::defs::{BarBase, Def};
use super::fills::parse_fill;
use super::format::{
    format_number_with_style, get_effective_axis_format, resolve_effective_value_label_format,
};
use super::html::{build_styled_html, escape_html, raw_size, TextStyleProps};
use super::jsmath::{js_max, js_min};
use super::layout::{
    calculate_chart_layout, estimate_value_label_width, MIN_VALUE_LABEL_WIDTH,
    OUTSIDE_VALUE_LABEL_WIDTH, VALUE_LABEL_GAP,
};
use super::parts::{
    build_legend_items, generate_background_node, generate_legend_nodes,
    generate_source_footer_nodes, generate_title_nodes,
};
use super::scales::{
    create_linear_scale, create_log_scale, get_display_range_from_ticks, is_log_domain_valid,
    CategoricalScale, ValueScale,
};
use super::stack::{compute_stack, StackResult};
use super::value_axis_bounds::{
    apply_value_axis_bounds, clamp_scale_to_plot, resolve_value_axis_bounds,
};

fn clamp01(v: f64) -> f64 {
    js_max(0.0, js_min(1.0, v))
}

fn normalize_corner_radius(value: Option<&Value>) -> [f64; 4] {
    match value {
        Some(Value::Array(a)) => [
            super::cfg::num(a.first()),
            super::cfg::num(a.get(1)),
            super::cfg::num(a.get(2)),
            super::cfg::num(a.get(3)),
        ],
        Some(Value::Number(n)) => {
            let r = n.as_f64().unwrap_or(f64::NAN);
            [r, r, r, r]
        }
        _ => [0.0; 4],
    }
}

fn resolve_stacked_corner_radius(
    value: Option<&Value>,
    segment_index: usize,
    segment_count: usize,
    horizontal: bool,
) -> [f64; 4] {
    let [tl, tr, br, bl] = normalize_corner_radius(value);
    if segment_count <= 1 {
        return [tl, tr, br, bl];
    }
    if horizontal {
        if segment_index == 0 {
            return [tl, 0.0, 0.0, bl];
        }
        if segment_index == segment_count - 1 {
            return [0.0, tr, br, 0.0];
        }
        return [0.0; 4];
    }
    if segment_index == 0 {
        return [0.0, 0.0, br, bl];
    }
    if segment_index == segment_count - 1 {
        return [tl, tr, 0.0, 0.0];
    }
    [0.0; 4]
}

fn corner_radius_value(value: Option<&Value>) -> Value {
    // grouped bars keep the raw config/series value (number | array | absent)
    match value {
        Some(v) => v.clone(),
        None => Value::Null,
    }
}

struct SeriesVisuals {
    props: Map,
    corner_radius: Option<Value>,
    label_color: Option<String>,
    label_color_explicit: bool,
    fill_value: String,
}

fn get_series_visuals(
    spec: &SpecView,
    styles: SeriesStyles,
    palette: &[String],
    bar_shadow: Option<&str>,
    series_key: &str,
    series_index: usize,
    category: &str,
) -> SeriesVisuals {
    let style = styles.get(series_key);
    let fill_value = style
        .and_then(|s| s.s_truthy("fill"))
        .map(str::to_string)
        .unwrap_or_else(|| resolve_palette_color(palette, series_index).to_string());
    let mut props = parse_fill(&fill_value);
    let resolved_shadow = match style.map(|s| s.has("shadow")) {
        Some(true) => style.and_then(|s| s.s("shadow")),
        _ => bar_shadow,
    };
    if let Some(sh) = resolved_shadow {
        if !sh.is_empty() && sh != "none" {
            for (k, v) in resolve_shadow_props(Some(sh)) {
                props.insert(k, v);
            }
        }
    }
    if let Some(stroke) = style.and_then(|s| s.s_truthy("stroke")) {
        paint_bag::set_stroke_color(&mut props, json!(stroke));
    }
    if let Some(sw) = style.and_then(|s| s.f_opt("strokeWidth")) {
        paint_bag::set_stroke_width(&mut props, json!(sw));
    }
    if let Some(op) = style.and_then(|s| s.f_opt("opacity")) {
        props.insert("opacity".into(), json!(op));
    }
    let corner_radius = style
        .and_then(|s| s.value("cornerRadius"))
        .cloned()
        .or_else(|| spec.config.get("barRadius").cloned());
    let label_color = style.and_then(|s| s.s("labelColor")).map(str::to_string);
    // `point-colors` overrides the series fill for this one bar
    // (`{ ...visual.props, ...parseFill(pointColor) }`, barChart.ts emitBar);
    // the value label's auto-contrast reads the same colour.
    let fill_value = match spec.point_color(series_key, category) {
        Some(point_color) => {
            for (k, v) in parse_fill(point_color) {
                props.insert(k, v);
            }
            point_color.to_string()
        }
        None => fill_value,
    };
    SeriesVisuals {
        props,
        corner_radius,
        label_color_explicit: label_color.is_some(),
        label_color,
        fill_value,
    }
}

#[allow(clippy::too_many_arguments)]
struct ValueLabelArgs<'a> {
    bar_x: f64,
    bar_y: f64,
    bar_width: f64,
    bar_height: f64,
    value: f64,
    label: Option<&'a str>,
    category: &'a str,
    series: Option<&'a str>,
    label_color: Option<&'a str>,
    label_color_explicit: bool,
    bar_fill: Option<&'a str>,
}

pub fn generate_bar_chart_nodes(
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

    let orientation = spec.s("orientation").unwrap_or("vertical");
    let dims = resolve_chart_dimensions(data, spec);
    if dims.ordered_categories.is_empty() {
        return Ok(nodes);
    }

    // valuesByCategory: category → series → {value (summed), label}
    let values_by_category = aggregate_bar_values(data, dims.has_series_dimension);

    let stacking = resolve_bar_stacking(spec, dims.has_series_dimension);
    let (is_stacked, is_stacked_percent) = stacking;
    let base_axis_format: Option<&str> =
        if is_stacked_percent && config.s("yFormat") == Some("auto") {
            Some("percent")
        } else {
            config.s("yFormat")
        };

    let (mut min, mut max) =
        bar_value_domain(&values_by_category, &dims.ordered_categories, stacking);
    if !min.is_finite() || !max.is_finite() {
        min = 0.0;
        max = 0.0;
    }
    // Authored `y-min`/`y-max` replace the matching end before anything reads
    // the domain, so the log-validity check and the zero extension below both
    // see the window that will actually be drawn (valueAxisBounds.ts).
    let y_bounds = resolve_value_axis_bounds(spec.f_opt("yMin"), spec.f_opt("yMax"), (min, max));
    let (bounded_min, bounded_max) = apply_value_axis_bounds((min, max), &y_bounds);
    min = bounded_min;
    max = bounded_max;
    let effective_scale =
        if config.s("yScale") == Some("log") && !is_stacked && is_log_domain_valid(min, max) {
            "log"
        } else {
            "linear"
        };
    if effective_scale != "log" {
        // A pinned end is the author's floor/ceiling — do not widen it to 0.
        if y_bounds.min.is_none() {
            min = js_min(min, 0.0);
        }
        if y_bounds.max.is_none() {
            max = js_max(max, 0.0);
        }
    }

    // Re-pin after nicing, which would round `y-min = 3` back out to 0.
    let display_range = apply_value_axis_bounds(
        get_display_range_from_ticks((min, max), config.f("yTickCount"), effective_scale),
        &y_bounds,
    );
    let build_value_scale = |range: f64| -> ValueScale {
        let inner = if effective_scale == "log" {
            create_log_scale(display_range, range)
        } else {
            create_linear_scale(display_range, range)
        };
        clamp_scale_to_plot(inner, range, &y_bounds, display_range)
    };
    let (_, axis_format) =
        get_effective_axis_format(display_range, config.f("yTickCount"), base_axis_format)?;

    let clamped_bar_gap = clamp01(config.f("barGap"));
    let preferred_generated_label_format = if config.s("labelFormat") == Some("auto") {
        config.s("valueFormat")
    } else {
        config.s("labelFormat")
    };
    let generated_values: Vec<f64> = dims
        .ordered_categories
        .iter()
        .flat_map(|category| {
            values_by_category
                .get(category)
                .map(|m| {
                    m.values()
                        .filter(|e| e.label.is_none())
                        .map(|e| e.value)
                        .collect::<Vec<f64>>()
                })
                .unwrap_or_default()
        })
        .collect();
    let resolve_generated_label_format = |available_width: f64| -> &str {
        resolve_effective_value_label_format(
            &generated_values,
            preferred_generated_label_format,
            config.s("valueDensity").unwrap_or("auto"),
            available_width,
            config.f("valueLabelSize"),
        )
    };
    // TS initializes this to config.labelFormat, but every orientation branch
    // overwrites it before the first value label is emitted
    // (barChart.ts:199/429/494/606/672).
    #[allow(unused_assignments)]
    let mut effective_generated_label_format: &str = config.s("labelFormat").unwrap_or("");

    let value_label_color_explicit = config.b("valueLabelColorExplicit");
    let show_labels = config.b("showLabels");
    let label_position = config.s("labelPosition");

    // A negative bar's label below the plot would run into the category labels.
    let plot_bottom = layout.plot_area.y + layout.plot_area.height;
    let add_value_label = |nodes: &mut Vec<Def>, args: ValueLabelArgs, effective_format: &str| {
        if !show_labels {
            return;
        }
        let label_text = match args.label {
            Some(l) => l.to_string(),
            None => format_number_with_style(args.value, effective_format),
        };
        if label_text.is_empty() {
            return;
        }
        let label_height = config.f("valueLabelSize") + 2.0;
        let estimated_label_width = js_max(
            MIN_VALUE_LABEL_WIDTH,
            estimate_value_label_width(&label_text, config.f("valueLabelSize")),
        );
        let mut label_x;
        let label_y;
        let mut text_align = "center";
        let mut is_inside_placement = label_position == Some("inside");
        let mut label_width: f64;

        if orientation == "vertical" {
            let should_fallback_outside = label_position == Some("inside")
                && (args.bar_width < estimated_label_width
                    || args.bar_height < label_height + VALUE_LABEL_GAP * 2.0);
            label_width = js_max(MIN_VALUE_LABEL_WIDTH, args.bar_width);
            // Centred on the bar: a bar narrower than the label's minimum width has the label overhang both sides
            // equally (from the bar's left edge, it sat half the overhang to the right, over the next bar).
            let centered_x = args.bar_x + (args.bar_width - label_width) / 2.0;
            if label_position == Some("inside") && !should_fallback_outside {
                label_x = centered_x;
                label_y = args.bar_y + (args.bar_height - label_height) / 2.0;
            } else if label_position == Some("inside") && should_fallback_outside {
                is_inside_placement = false;
                label_x = centered_x;
                label_y = if args.value < 0.0 {
                    args.bar_y + args.bar_height + VALUE_LABEL_GAP
                } else {
                    args.bar_y - label_height - VALUE_LABEL_GAP
                };
            } else {
                // "top" and "outside": past the bar's end, which is its bottom for a negative value (above it, the
                // label sat on the zero line over the positive area). Below the plot it would run into the category
                // labels: then it goes inside the bar's end, or, on a bar too short to hold it, above the zero line.
                is_inside_placement = false;
                label_x = centered_x;
                let mut y = args.bar_y - label_height - VALUE_LABEL_GAP;
                if args.value < 0.0 {
                    let below = args.bar_y + args.bar_height + VALUE_LABEL_GAP;
                    if below + label_height <= plot_bottom {
                        y = below;
                    } else if args.bar_height >= label_height + VALUE_LABEL_GAP * 2.0 {
                        is_inside_placement = true;
                        y = args.bar_y + args.bar_height - label_height - VALUE_LABEL_GAP;
                    }
                }
                label_y = y;
            }
        } else {
            let centered_y = args.bar_y + (args.bar_height - label_height) / 2.0;
            let should_fallback_outside = label_position == Some("inside")
                && args.bar_width < estimated_label_width + VALUE_LABEL_GAP * 2.0;
            if label_position == Some("inside") && !should_fallback_outside {
                label_width = js_max(0.0, args.bar_width);
                label_x = args.bar_x;
                label_y = centered_y;
            } else {
                is_inside_placement = false;
                label_width = js_max(OUTSIDE_VALUE_LABEL_WIDTH, estimated_label_width);
                label_y = centered_y;
                if args.value < 0.0 {
                    label_x = args.bar_x - label_width - VALUE_LABEL_GAP;
                    text_align = "right";
                } else {
                    label_x = args.bar_x + args.bar_width + VALUE_LABEL_GAP;
                    text_align = "left";
                }
                // `calculate_chart_layout` reserves a right-edge band sized for
                // exactly this placement, but the band is capped so a very
                // narrow chart keeps a usable plot area. When that cap binds —
                // or a negative label runs past the left edge — pull the label
                // back inside the chart bounds.
                // `clamp(value, min, max)` is `Math.max(min, Math.min(max, value))`
                // (utils/math.ts) — mirrored argument-for-argument.
                label_width = js_min(label_width, spec.width);
                label_x = js_max(0.0, js_min(spec.width - label_width, label_x));
            }
        }

        let has_explicit_color = args.label_color_explicit || value_label_color_explicit;
        let color: String = if has_explicit_color {
            args.label_color
                .or(config.s("valueLabelColor"))
                .unwrap_or("")
                .to_string()
        } else if is_inside_placement && args.bar_fill.is_some() {
            get_auto_contrast_color(args.bar_fill.unwrap())
        } else {
            args.label_color
                .or(config.s("valueLabelColor"))
                .unwrap_or("")
                .to_string()
        };

        let mut metadata = Map::new();
        metadata.insert("chartRole".into(), json!("value-label"));
        metadata.insert("chartCategory".into(), json!(args.category));
        metadata.insert("chartName".into(), json!(chart_name));
        if let Some(series) = args.series {
            metadata.insert("chartSeries".into(), json!(series));
        }
        let mut d = Def::new("richtext");
        d.set("x", json!(label_x))
            .set("y", json!(label_y))
            .set("width", json!(label_width))
            .set("height", json!(label_height))
            .set(
                "htmlContent",
                json!(build_styled_html(
                    &TextStyleProps {
                        font_size: raw_size(config.f("valueLabelSize")),
                        font_weight: Some(400.0),
                        font_family: config.s("fontFamily"),
                        text_align: Some(text_align),
                        color: if color.is_empty() { None } else { Some(&color) },
                    },
                    &escape_html(&label_text),
                )),
            )
            // A number never wraps: at a bar's width, "3.35" broke as "3.3" over "5", the "5" lost on a dark bar.
            .set("resizeMode", json!("auto-width"))
            .set(
                "name",
                json!(format!(
                    "{chart_name} Value {}{}",
                    args.category,
                    args.series.map(|s| format!(" {s}")).unwrap_or_default()
                )),
            )
            .set("metadata", Value::Object(metadata));
        nodes.push(d);
    };

    let should_render_legend = dims.has_series_dimension
        && dims.ordered_series.len() > 1
        && (config.show_legend_true() || config.show_legend_auto());
    if should_render_legend {
        if let Some(legend_area) = layout.legend_area {
            let items = build_legend_items(&dims.ordered_series, styles, &config.palette());
            nodes.extend(generate_legend_nodes(
                &items,
                config,
                legend_area,
                chart_name,
                None,
            ));
        }
    }

    let (axis_config_y, axis_config_x, font_config) = create_axis_and_font_config(config);

    let mut series_index_map: HashMap<&str, usize> = HashMap::new();
    let mut stacked_segments: Option<StackResult> = None;
    if is_stacked {
        for (i, s) in dims.ordered_series.iter().enumerate() {
            series_index_map.insert(s.as_str(), i);
        }
        stacked_segments = Some(compute_stack(
            &dims.ordered_categories,
            &dims.ordered_series,
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
            "diverging",
        ));
    }

    let palette = config.palette();
    let bar_shadow = config.s("barShadow");
    let grid_config = GridConfig {
        show: config.b("showGrid"),
        color: config.s("gridColor"),
        width: config.f("gridWidth"),
        style: config.s("gridStyle"),
    };

    if orientation == "vertical" {
        let x_scale = CategoricalScale::new(
            &dims.ordered_categories,
            layout.plot_area.width,
            config.f("barGroupGap"),
        );
        let y_scale = build_value_scale(layout.plot_area.height);
        let baseline_value = if effective_scale == "log" {
            display_range.0
        } else {
            0.0
        };
        let baseline_y =
            layout.plot_area.y + layout.plot_area.height - y_scale.scale(baseline_value);
        let band_width = x_scale.bandwidth();

        nodes.extend(generate_grid_nodes(
            &y_scale,
            display_range,
            config.f("yTickCount"),
            "horizontal",
            layout.plot_area,
            &grid_config,
            chart_name,
        )?);

        if is_stacked {
            let bar_width = js_max(0.0, band_width * (1.0 - clamped_bar_gap));
            let bar_offset = (band_width - bar_width) / 2.0;
            effective_generated_label_format =
                resolve_generated_label_format(js_max(MIN_VALUE_LABEL_WIDTH, bar_width));
            for category in &dims.ordered_categories {
                let Some(cat_map) = values_by_category.get(category) else {
                    continue;
                };
                let active_series: Vec<&String> = dims
                    .ordered_series
                    .iter()
                    .filter(|s| cat_map.contains_key(s.as_str()))
                    .collect();
                let segment_count = active_series.len();
                for (segment_index, series_key) in active_series.iter().enumerate() {
                    let Some(entry) = cat_map.get(series_key.as_str()) else {
                        continue;
                    };
                    let series_index = *series_index_map
                        .get(series_key.as_str())
                        .unwrap_or(&segment_index);
                    let Some(segment) = stacked_segments
                        .as_ref()
                        .and_then(|s| s.get(category.as_str()))
                        .and_then(|m| m.get(series_key.as_str()))
                    else {
                        continue;
                    };
                    let start_y =
                        layout.plot_area.y + layout.plot_area.height - y_scale.scale(segment.y0);
                    let end_y =
                        layout.plot_area.y + layout.plot_area.height - y_scale.scale(segment.y1);
                    let bar_height = (end_y - start_y).abs();
                    let bar_y = js_min(start_y, end_y);
                    let bar_x = layout.plot_area.x + x_scale.scale(category) + bar_offset;

                    let vis = get_series_visuals(
                        spec,
                        styles,
                        &palette,
                        bar_shadow,
                        series_key,
                        series_index,
                        data.category_label(category),
                    );
                    let scr = resolve_stacked_corner_radius(
                        vis.corner_radius.as_ref(),
                        segment_index,
                        segment_count,
                        false,
                    );
                    let mut d = Def::new("rectangle");
                    // The segment grows from its stack base (`y0`), below it in value.
                    d.grows_from(
                        if start_y >= end_y {
                            BarBase::Bottom
                        } else {
                            BarBase::Top
                        },
                        segment.value,
                    );
                    d.set("x", json!(bar_x))
                        .set("y", json!(bar_y))
                        .set("width", json!(bar_width))
                        .set("height", json!(bar_height))
                        .set("cornerRadius", json!(scr.to_vec()))
                        .set("name", json!(format!("{chart_name} Bar {category} {series_key}")))
                        .set(
                            "metadata",
                            json!({"chartRole": "bar", "chartCategory": category, "chartName": chart_name, "chartSeries": series_key}),
                        );
                    for (k, v) in &vis.props {
                        d.set(k, v.clone());
                    }
                    nodes.push(d);

                    add_value_label(
                        &mut nodes,
                        ValueLabelArgs {
                            bar_x,
                            bar_y,
                            bar_width,
                            bar_height,
                            value: segment.value,
                            label: entry.label.as_deref(),
                            category,
                            series: Some(series_key),
                            label_color: vis.label_color.as_deref(),
                            label_color_explicit: vis.label_color_explicit,
                            bar_fill: Some(&vis.fill_value),
                        },
                        effective_generated_label_format,
                    );
                }
            }
        } else {
            let series_count = dims.series_keys.len() as f64;
            let sub_band = band_width / series_count;
            let bar_width = js_max(0.0, sub_band * (1.0 - clamped_bar_gap));
            let bar_offset = (sub_band - bar_width) / 2.0;
            effective_generated_label_format =
                resolve_generated_label_format(js_max(MIN_VALUE_LABEL_WIDTH, bar_width));

            for category in &dims.ordered_categories {
                let Some(cat_map) = values_by_category.get(category) else {
                    continue;
                };
                for (series_index, series_key) in dims.series_keys.iter().enumerate() {
                    let Some(entry) = cat_map.get(series_key.as_str()) else {
                        continue;
                    };
                    let value = entry.value;
                    let scaled_value = y_scale.scale(value);
                    let scaled_floor = y_scale.scale(baseline_value);
                    let bar_height = (scaled_value - scaled_floor).abs();
                    let bar_x = layout.plot_area.x
                        + x_scale.scale(category)
                        + series_index as f64 * sub_band
                        + bar_offset;
                    let bar_y = if value >= baseline_value {
                        baseline_y - bar_height
                    } else {
                        baseline_y
                    };

                    let vis = get_series_visuals(
                        spec,
                        styles,
                        &palette,
                        bar_shadow,
                        series_key,
                        series_index,
                        data.category_label(category),
                    );
                    let mut metadata = Map::new();
                    metadata.insert("chartRole".into(), json!("bar"));
                    metadata.insert("chartCategory".into(), json!(category));
                    metadata.insert("chartName".into(), json!(chart_name));
                    if dims.has_series_dimension {
                        metadata.insert("chartSeries".into(), json!(series_key));
                    }
                    let mut d = Def::new("rectangle");
                    d.grows_from(
                        if value >= baseline_value {
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
                        .set(
                            "cornerRadius",
                            corner_radius_value(vis.corner_radius.as_ref()),
                        )
                        .set(
                            "name",
                            json!(format!(
                                "{chart_name} Bar {category}{}",
                                if dims.has_series_dimension {
                                    format!(" {series_key}")
                                } else {
                                    String::new()
                                }
                            )),
                        )
                        .set("metadata", Value::Object(metadata));
                    for (k, v) in &vis.props {
                        d.set(k, v.clone());
                    }
                    nodes.push(d);

                    add_value_label(
                        &mut nodes,
                        ValueLabelArgs {
                            bar_x,
                            bar_y,
                            bar_width,
                            bar_height,
                            value,
                            label: entry.label.as_deref(),
                            category,
                            series: if dims.has_series_dimension {
                                Some(series_key)
                            } else {
                                None
                            },
                            label_color: vis.label_color.as_deref(),
                            label_color_explicit: vis.label_color_explicit,
                            bar_fill: Some(&vis.fill_value),
                        },
                        effective_generated_label_format,
                    );
                }
            }
        }

        nodes.extend(generate_numeric_axis_nodes(
            &y_scale,
            display_range,
            config.f("yTickCount"),
            axis_format,
            None,
            "left",
            layout.plot_area,
            layout.y_axis_area,
            &axis_config_y,
            &font_config,
            chart_name,
            None,
        )?);
        nodes.extend(generate_categorical_axis_nodes(
            &dims.ordered_categories,
            &x_scale,
            "bottom",
            layout.plot_area,
            layout.x_axis_area,
            &axis_config_x,
            &font_config,
            chart_name,
            Some(baseline_y),
            data.category_labels(),
        ));
    } else {
        let y_scale_cat = CategoricalScale::new(
            &dims.ordered_categories,
            layout.plot_area.height,
            config.f("barGroupGap"),
        );
        let x_scale = build_value_scale(layout.plot_area.width);
        let baseline_value_h = if effective_scale == "log" {
            display_range.0
        } else {
            0.0
        };
        let baseline_x = layout.plot_area.x + x_scale.scale(baseline_value_h);
        let band_height = y_scale_cat.bandwidth();
        // The value axis runs along the bottom: fewer ticks when its labels
        // would run together (the grid follows; `fitBottomTickCount`).
        let value_tick_count = if axis_config_x.show_labels {
            fit_bottom_tick_count(
                &x_scale,
                display_range,
                config.f("yTickCount"),
                axis_format,
                font_config.font_size,
            )?
        } else {
            config.f("yTickCount")
        };

        nodes.extend(generate_grid_nodes(
            &x_scale,
            display_range,
            value_tick_count,
            "vertical",
            layout.plot_area,
            &grid_config,
            chart_name,
        )?);

        if is_stacked {
            let bar_height = js_max(0.0, band_height * (1.0 - clamped_bar_gap));
            let bar_offset = (band_height - bar_height) / 2.0;
            effective_generated_label_format = resolve_generated_label_format(js_max(
                MIN_VALUE_LABEL_WIDTH,
                layout.plot_area.width,
            ));
            for category in &dims.ordered_categories {
                let Some(cat_map) = values_by_category.get(category) else {
                    continue;
                };
                let active_series: Vec<&String> = dims
                    .ordered_series
                    .iter()
                    .filter(|s| cat_map.contains_key(s.as_str()))
                    .collect();
                let segment_count = active_series.len();
                for (segment_index, series_key) in active_series.iter().enumerate() {
                    let Some(entry) = cat_map.get(series_key.as_str()) else {
                        continue;
                    };
                    let series_index = *series_index_map
                        .get(series_key.as_str())
                        .unwrap_or(&segment_index);
                    let Some(segment) = stacked_segments
                        .as_ref()
                        .and_then(|s| s.get(category.as_str()))
                        .and_then(|m| m.get(series_key.as_str()))
                    else {
                        continue;
                    };
                    let start_x = layout.plot_area.x + x_scale.scale(segment.y0);
                    let end_x = layout.plot_area.x + x_scale.scale(segment.y1);
                    let bar_width = (end_x - start_x).abs();
                    let bar_x = js_min(start_x, end_x);
                    let bar_y = layout.plot_area.y + y_scale_cat.scale(category) + bar_offset;

                    let vis = get_series_visuals(
                        spec,
                        styles,
                        &palette,
                        bar_shadow,
                        series_key,
                        series_index,
                        data.category_label(category),
                    );
                    let scr = resolve_stacked_corner_radius(
                        vis.corner_radius.as_ref(),
                        segment_index,
                        segment_count,
                        true,
                    );
                    let mut d = Def::new("rectangle");
                    d.grows_from(
                        if start_x <= end_x {
                            BarBase::Left
                        } else {
                            BarBase::Right
                        },
                        segment.value,
                    );
                    d.set("x", json!(bar_x))
                        .set("y", json!(bar_y))
                        .set("width", json!(bar_width))
                        .set("height", json!(bar_height))
                        .set("cornerRadius", json!(scr.to_vec()))
                        .set("name", json!(format!("{chart_name} Bar {category} {series_key}")))
                        .set(
                            "metadata",
                            json!({"chartRole": "bar", "chartCategory": category, "chartSeries": series_key, "chartName": chart_name}),
                        );
                    for (k, v) in &vis.props {
                        d.set(k, v.clone());
                    }
                    nodes.push(d);

                    add_value_label(
                        &mut nodes,
                        ValueLabelArgs {
                            bar_x,
                            bar_y,
                            bar_width,
                            bar_height,
                            value: segment.value,
                            label: entry.label.as_deref(),
                            category,
                            series: Some(series_key),
                            label_color: vis.label_color.as_deref(),
                            label_color_explicit: vis.label_color_explicit,
                            bar_fill: Some(&vis.fill_value),
                        },
                        effective_generated_label_format,
                    );
                }
            }
        } else {
            let series_count = dims.series_keys.len() as f64;
            let sub_band = band_height / series_count;
            let bar_height = js_max(0.0, sub_band * (1.0 - clamped_bar_gap));
            let bar_offset = (sub_band - bar_height) / 2.0;
            effective_generated_label_format = resolve_generated_label_format(js_max(
                MIN_VALUE_LABEL_WIDTH,
                layout.plot_area.width,
            ));

            for category in &dims.ordered_categories {
                let Some(cat_map) = values_by_category.get(category) else {
                    continue;
                };
                for (series_index, series_key) in dims.series_keys.iter().enumerate() {
                    let Some(entry) = cat_map.get(series_key.as_str()) else {
                        continue;
                    };
                    let value = entry.value;
                    let scaled_value = x_scale.scale(value);
                    let scaled_floor = x_scale.scale(baseline_value_h);
                    let bar_width = (scaled_value - scaled_floor).abs();
                    let bar_x = if value >= baseline_value_h {
                        baseline_x
                    } else {
                        baseline_x - bar_width
                    };
                    let bar_y = layout.plot_area.y
                        + y_scale_cat.scale(category)
                        + series_index as f64 * sub_band
                        + bar_offset;

                    let vis = get_series_visuals(
                        spec,
                        styles,
                        &palette,
                        bar_shadow,
                        series_key,
                        series_index,
                        data.category_label(category),
                    );
                    let mut metadata = Map::new();
                    metadata.insert("chartRole".into(), json!("bar"));
                    metadata.insert("chartCategory".into(), json!(category));
                    metadata.insert("chartName".into(), json!(chart_name));
                    if dims.has_series_dimension {
                        metadata.insert("chartSeries".into(), json!(series_key));
                    }
                    let mut d = Def::new("rectangle");
                    d.grows_from(
                        if value >= baseline_value_h {
                            BarBase::Left
                        } else {
                            BarBase::Right
                        },
                        value,
                    );
                    d.set("x", json!(bar_x))
                        .set("y", json!(bar_y))
                        .set("width", json!(bar_width))
                        .set("height", json!(bar_height))
                        .set(
                            "cornerRadius",
                            corner_radius_value(vis.corner_radius.as_ref()),
                        )
                        .set(
                            "name",
                            json!(format!(
                                "{chart_name} Bar {category}{}",
                                if dims.has_series_dimension {
                                    format!(" {series_key}")
                                } else {
                                    String::new()
                                }
                            )),
                        )
                        .set("metadata", Value::Object(metadata));
                    for (k, v) in &vis.props {
                        d.set(k, v.clone());
                    }
                    nodes.push(d);

                    add_value_label(
                        &mut nodes,
                        ValueLabelArgs {
                            bar_x,
                            bar_y,
                            bar_width,
                            bar_height,
                            value,
                            label: entry.label.as_deref(),
                            category,
                            series: if dims.has_series_dimension {
                                Some(series_key)
                            } else {
                                None
                            },
                            label_color: vis.label_color.as_deref(),
                            label_color_explicit: vis.label_color_explicit,
                            bar_fill: Some(&vis.fill_value),
                        },
                        effective_generated_label_format,
                    );
                }
            }
        }

        nodes.extend(generate_numeric_axis_nodes(
            &x_scale,
            display_range,
            value_tick_count,
            axis_format,
            None,
            "bottom",
            layout.plot_area,
            layout.x_axis_area,
            &axis_config_x,
            &font_config,
            chart_name,
            None,
        )?);
        nodes.extend(generate_categorical_axis_nodes(
            &dims.ordered_categories,
            &y_scale_cat,
            "left",
            layout.plot_area,
            layout.y_axis_area,
            &axis_config_y,
            &font_config,
            chart_name,
            Some(baseline_x),
            data.category_labels(),
        ));
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
