//! Port of `utils/markup/elements/charts/generators/pieChart.ts`.

use super::ticks::TickError;
use crate::value::{json, Map, Value};
use std::collections::HashMap;

use super::cfg::{resolve_series_order, ConfigView, DataView, SeriesStyles, SpecView};
use super::colorutil::{resolve_palette_color, resolve_shadow_props};
use super::defs::Def;
use super::fills::parse_fill;
use super::format::{format_number_with_style, resolve_effective_value_label_format};
use super::html::{build_styled_html, escape_html, raw_size, TextStyleProps};
use super::jsmath::{js_max, js_min};
use super::layout::{
    calculate_chart_layout, estimate_value_label_width, CHARACTER_WIDTH_FONT_SIZE_RATIO,
    LABEL_WIDTH_PADDING,
};
use super::parts::{
    build_legend_items, generate_background_node, generate_legend_nodes,
    generate_source_footer_nodes, generate_title_nodes, resolve_outside_label_positions,
    LabelCandidate, DEFAULT_LABEL_MIN_GAP,
};

const OUTSIDE_LABEL_GAP: f64 = 16.0;
const CALLOUT_LABEL_HORIZONTAL_GAP: f64 = 4.0;
const CALLOUT_LABEL_RADIUS_BUDGET_MULTIPLIER: f64 = 1.3;
const CALLOUT_LABEL_BUDGET_PADDING: f64 = 16.0;
const CALLOUT_LABEL_CHART_WIDTH_BUDGET_RATIO: f64 = 0.35;

fn clamp(value: f64, min: f64, max: f64) -> f64 {
    js_max(min, js_min(max, value))
}

fn to_radians(degrees: f64) -> f64 {
    degrees * std::f64::consts::PI / 180.0
}

#[derive(Clone)]
struct SliceDatum {
    /// The slice's category key: unique per position (`positional_categories`).
    category: String,
    /// What the slice draws and is styled by: the category's label.
    name: String,
    value: f64,
    label: Option<String>,
}

/// `resolveSliceData` (pieChart.ts:227-248).
fn resolve_slice_data(data: &DataView, series_order: Option<&[String]>) -> Vec<SliceDatum> {
    let mut by_category: HashMap<String, SliceDatum> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for row in &data.rows {
        if row.missing() {
            continue;
        }
        let category = row.category_key();
        if category.is_empty() {
            continue;
        }
        let entry = by_category.entry(category.to_string()).or_insert_with(|| {
            order.push(category.to_string());
            SliceDatum {
                category: category.to_string(),
                name: data.category_label(category).to_string(),
                value: 0.0,
                label: None,
            }
        });
        entry.value += row.f("value");
        if let Some(l) = row.s_truthy("label") {
            entry.label = Some(l.to_string());
        }
    }
    let ordered = resolve_series_order(&order, series_order);
    ordered
        .into_iter()
        .map(|category| {
            let mut datum = by_category.remove(&category).unwrap_or(SliceDatum {
                name: data.category_label(&category).to_string(),
                category: category.clone(),
                value: 0.0,
                label: None,
            });
            datum.value = js_max(0.0, datum.value);
            datum
        })
        .collect()
}

/// `buildSliceLabelText` (pieChart.ts:52-73).
fn build_slice_label_text(
    category: &str,
    value: f64,
    percentage: f64,
    format: &str,
    value_format: &str,
) -> String {
    let formatted_value = format_number_with_style(value, value_format);
    let formatted_percent = format_number_with_style(percentage, "percent");
    match format {
        "category" => category.to_string(),
        "value" => formatted_value,
        "both" => {
            if !formatted_value.is_empty() {
                format!("{formatted_value} ({formatted_percent})")
            } else {
                formatted_percent
            }
        }
        _ => formatted_percent, // 'percent' | default
    }
}

fn pie_label_downgrade_order(format: &str) -> Vec<&str> {
    match format {
        "both" => vec!["both", "value", "percent"],
        "value" => vec!["value", "percent"],
        other => vec![other],
    }
}

fn resolve_callout_label_budget(radius: f64, chart_width: f64) -> f64 {
    js_max(
        radius * CALLOUT_LABEL_RADIUS_BUDGET_MULTIPLIER + CALLOUT_LABEL_BUDGET_PADDING,
        chart_width * CALLOUT_LABEL_CHART_WIDTH_BUDGET_RATIO,
    )
}

/// `resolveEffectiveCalloutLabelFormat` (pieChart.ts:123-152).
fn resolve_effective_callout_label_format<'a>(
    slices: &[SliceDatum],
    total: f64,
    preferred: &'a str,
    value_format: &str,
    available_width: f64,
    font_size: f64,
) -> &'a str {
    let candidates = pie_label_downgrade_order(preferred);
    for format in &candidates {
        let widest = slices.iter().fold(0.0f64, |m, slice| {
            let percentage = if total > 0.0 {
                slice.value / total
            } else {
                0.0
            };
            let text = build_slice_label_text(
                &slice.name,
                slice.value,
                percentage,
                format,
                value_format,
            );
            js_max(m, estimate_value_label_width(&text, font_size))
        });
        if widest <= available_width {
            return pin_format(format, preferred);
        }
    }
    pin_format(candidates[candidates.len() - 1], preferred)
}

fn pin_format<'a>(candidate: &str, preferred: &'a str) -> &'a str {
    match candidate {
        "both" => "both",
        "value" => "value",
        "percent" => "percent",
        "category" => "category",
        _ => preferred,
    }
}

/// `resolveEffectivePieLabel` — for inside/outside labels the TS uses
/// `config.pieLabelFormat` directly (pieChart.ts:438-459 builds the text
/// inline), so only the callout path needs the downgrade resolution.
struct CenterTextSizes {
    label: f64,
    sublabel: f64,
}

fn estimate_max_font_size_for_width(text: &str, available_width: f64) -> f64 {
    if text.trim().is_empty() {
        return f64::INFINITY;
    }
    let char_width = js_max(
        1.0,
        super::format::label_ems(text.trim(), CHARACTER_WIDTH_FONT_SIZE_RATIO),
    );
    let usable = js_max(10.0, available_width - LABEL_WIDTH_PADDING);
    usable / char_width
}

/// `resolveCenterTextSizes` (pieChart.ts:188-225).
fn resolve_center_text_sizes(
    inner_diameter: f64,
    center_label: &str,
    center_sublabel: &str,
    requested_center_size: f64,
) -> CenterTextSizes {
    let requested_sublabel_size =
        js_max(10.0, super::jsmath::js_round(requested_center_size * 0.5));
    let hole_padding = 12.0;
    let available_width = js_max(24.0, inner_diameter - hole_padding);
    let available_height = js_max(24.0, inner_diameter - hole_padding);

    let mut width_scale_candidates = vec![1.0f64];
    if !center_label.is_empty() {
        width_scale_candidates.push(
            estimate_max_font_size_for_width(center_label, available_width)
                / js_max(1.0, requested_center_size),
        );
    }
    if !center_sublabel.is_empty() {
        width_scale_candidates.push(
            estimate_max_font_size_for_width(center_sublabel, available_width)
                / js_max(1.0, requested_sublabel_size),
        );
    }
    let requested_label_height = if !center_label.is_empty() {
        requested_center_size + 4.0
    } else {
        0.0
    };
    let requested_sublabel_height = if !center_sublabel.is_empty() {
        requested_sublabel_size + 4.0
    } else {
        0.0
    };
    let gap = if !center_label.is_empty() && !center_sublabel.is_empty() {
        4.0
    } else {
        0.0
    };
    let requested_total = requested_label_height + requested_sublabel_height + gap;
    let height_scale = if requested_total > 0.0 {
        available_height / requested_total
    } else {
        1.0
    };
    let scale = width_scale_candidates
        .into_iter()
        .fold(js_min(1.0, height_scale), js_min);
    let center_label_size = js_max(10.0, (requested_center_size * scale).floor());
    let center_sublabel_size = if !center_sublabel.is_empty() {
        js_max(10.0, (requested_sublabel_size * scale).floor())
    } else {
        requested_sublabel_size
    };
    CenterTextSizes {
        label: center_label_size,
        sublabel: center_sublabel_size,
    }
}

struct PendingOutsideLabel {
    category: String,
    value: f64,
    percentage: f64,
    text: String,
    label_width: f64,
    label_height: f64,
    text_align: &'static str,
    side_right: bool,
    ideal_y: f64,
    ideal_x: f64,
    is_callout: bool,
    line_start_x: f64,
    line_start_y: f64,
}

/// Slice labels are the pie's value labels: an explicit `value-label-color` (or `label-color`) colours them, else
/// they take the axis label colour (as `pieChart.ts` `sliceLabelColor`).
fn slice_label_color<'a>(config: ConfigView<'a>) -> Option<&'a str> {
    if config.b("valueLabelColorExplicit") {
        config.s("valueLabelColor").or(config.s("labelColor"))
    } else {
        config.s("labelColor")
    }
}

/// Slice labels' size: an explicit `value-label-size` sets it, else `font-size` (as `pieChart.ts` `sliceLabelSize`).
fn slice_label_size(config: ConfigView) -> f64 {
    if config.b("valueLabelSizeExplicit") {
        config.f("valueLabelSize")
    } else {
        config.f("fontSize")
    }
}

fn slice_label_html(config: ConfigView, text_align: &str, text: &str) -> String {
    build_styled_html(
        &TextStyleProps {
            font_size: raw_size(slice_label_size(config)),
            font_family: config.s("fontFamily"),
            text_align: Some(text_align),
            color: slice_label_color(config),
            ..Default::default()
        },
        &escape_html(text),
    )
}

pub fn generate_pie_chart_nodes(
    spec: &SpecView,
    data: &DataView,
    styles: SeriesStyles,
    chart_name: &str,
) -> Result<Vec<Def>, TickError> {
    let config = spec.config;
    let mut nodes: Vec<Def> = Vec::new();

    let order = spec.series_order();
    let slices = resolve_slice_data(data, order.as_deref());
    let total: f64 = slices.iter().map(|s| s.value).sum();

    // legendData = { ...data, series: slice categories } (pieChart.ts:261)
    let mut legend_map = data.map.clone();
    legend_map.insert("series".into(),
        json!(slices
            .iter()
            .map(|s| s.name.clone())
            .collect::<Vec<_>>()),
    );
    let legend_data = DataView::new(&legend_map);
    let layout = calculate_chart_layout(spec, &legend_data)?;

    // resolveCalloutHorizontalBounds (pieChart.ts:160-176)
    let mut callout_min_x = 0.0f64;
    let mut callout_max_x = spec.width;
    if let Some(area) = layout.legend_area {
        match config.s("legendPosition") {
            Some("right") => callout_max_x = area.x,
            Some("left") => callout_min_x = area.x + area.width,
            _ => {}
        }
    }
    let clamp_callout_x = |label_x: f64, label_width: f64| -> f64 {
        clamp(
            label_x,
            callout_min_x,
            js_max(callout_min_x, callout_max_x - label_width),
        )
    };

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

    if layout.plot_area.width <= 0.0
        || layout.plot_area.height <= 0.0
        || slices.is_empty()
        || total <= 0.0
    {
        return Ok(nodes);
    }

    let center_x = layout.plot_area.x + layout.plot_area.width / 2.0;
    let center_y = layout.plot_area.y + layout.plot_area.height / 2.0;
    let inner_radius_ratio = clamp(config.f("innerRadius"), 0.0, 0.99);

    let start_angle = config.f("startAngle");
    let total_sweep = match config.get("endAngle") {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN) - start_angle,
        _ => {
            (if config.s("direction") == Some("cw") {
                1.0
            } else {
                -1.0
            }) * 360.0
        }
    };
    let pad_angle = if slices.len() > 1 {
        js_max(0.0, config.f("padAngle"))
    } else {
        0.0
    };

    let show_labels = config.b("showLabels");
    let pie_label_position = config.s("pieLabelPosition").unwrap_or("");
    let mut label_overflow_margin = 0.0;
    if show_labels && pie_label_position != "inside" && pie_label_position != "none" {
        let label_h = slice_label_size(config) + 4.0;
        label_overflow_margin = OUTSIDE_LABEL_GAP + label_h / 2.0;
        if config.b("explodeAll") || !config.explode_list().is_empty() {
            label_overflow_margin += config.f("explodeDistance");
        }
    }

    let radius = js_max(
        10.0,
        js_min(layout.plot_area.width, layout.plot_area.height) / 2.0 - label_overflow_margin,
    );
    let diameter = radius * 2.0;
    let values: Vec<f64> = slices.iter().map(|s| s.value).collect();
    let effective_value_format = resolve_effective_value_label_format(
        &values,
        config.s("valueFormat"),
        config.s("valueDensity").unwrap_or("auto"),
        resolve_callout_label_budget(radius, spec.width),
        slice_label_size(config),
    );
    let pie_label_format = config.s("pieLabelFormat").unwrap_or("");
    let effective_callout_label_format = if pie_label_position == "callout" {
        resolve_effective_callout_label_format(
            &slices,
            total,
            pie_label_format,
            effective_value_format,
            resolve_callout_label_budget(radius, spec.width),
            slice_label_size(config),
        )
    } else {
        pie_label_format
    };

    let mut pending: Vec<PendingOutsideLabel> = Vec::new();
    let mut current_angle = start_angle;

    for (index, slice) in slices.iter().enumerate() {
        let value = slice.value;
        let percentage = if total > 0.0 { value / total } else { 0.0 };
        let raw_sweep = total_sweep * percentage;
        let sweep_magnitude = js_max(0.0, raw_sweep.abs() - pad_angle);
        let sweep_angle = if raw_sweep == 0.0 {
            0.0
        } else {
            raw_sweep.signum() * sweep_magnitude
        };

        let slice_start = current_angle;
        current_angle += raw_sweep;
        if sweep_angle == 0.0 {
            continue;
        }

        let mid_angle = slice_start + sweep_angle / 2.0;
        let mid_radians = to_radians(mid_angle);
        let cos = mid_radians.cos();
        let sin = mid_radians.sin();

        let series_style = styles.get(&slice.name);
        let should_explode = config.b("explodeAll")
            || config.explode_list().contains(&slice.name.as_str())
            || series_style.and_then(|s| s.b_opt("explode")) == Some(true);
        let explode_distance = if should_explode {
            config.f("explodeDistance")
        } else {
            0.0
        };
        let offset_x = cos * explode_distance;
        let offset_y = sin * explode_distance;

        let slice_x = center_x - radius + offset_x;
        let slice_y = center_y - radius + offset_y;
        let fill_value = spec
            .point_colors()
            .and_then(|pc| pc.get(&slice.name))
            .and_then(Value::as_str)
            .or_else(|| series_style.and_then(|s| s.s_truthy("fill")))
            .map(str::to_string)
            .unwrap_or_else(|| resolve_palette_color(&config.palette(), index).to_string());
        let fill_props = parse_fill(&fill_value);
        let shadow_props = match series_style.and_then(|s| s.s_truthy("shadow")) {
            Some(sh) => resolve_shadow_props(Some(sh)),
            None => Map::new(),
        };
        // not-a-node-key: the chart spec's series style and config `stroke`
        let stroke_value = if series_style.map(|s| s.has("stroke")).unwrap_or(false) {
            series_style.and_then(|s| s.s("stroke"))
        } else {
            config.s("stroke")
        };
        let stroke_width = stroke_value.filter(|s| !s.is_empty()).map(|_| {
            series_style
                .and_then(|s| s.f_opt("strokeWidth"))
                .unwrap_or(config.f("strokeWidth"))
        });

        let mut d = Def::new("ellipse");
        d.set("x", json!(slice_x))
            .set("y", json!(slice_y))
            .set("width", json!(diameter))
            .set("height", json!(diameter))
            .set("startAngle", json!(slice_start))
            .set("sweepAngle", json!(sweep_angle))
            .set("innerRadiusRatio", json!(inner_radius_ratio))
            .set(
                "name",
                json!(format!("{chart_name} Slice {}", slice.category)),
            )
            .set(
                "metadata",
                json!({
                    "chartRole": "slice",
                    "chartCategory": slice.category,
                    "chartValue": value,
                    "chartPercentage": percentage,
                    "chartName": chart_name,
                }),
            );
        for (k, v) in &fill_props {
            d.set(k, v.clone());
        }
        for (k, v) in &shadow_props {
            d.set(k, v.clone());
        }
        if let Some(stroke) = stroke_value.filter(|s| !s.is_empty()) {
            d.stroke_color(json!(stroke));
        }
        if let Some(sw) = stroke_width {
            d.stroke_width(json!(sw));
        }
        if let Some(op) = series_style.and_then(|s| s.f_opt("opacity")) {
            d.set("opacity", json!(op));
        }
        nodes.push(d);

        if show_labels
            && pie_label_position != "none"
            && sweep_angle.abs() >= config.f("minLabelAngle")
        {
            let label_height = slice_label_size(config) + 4.0;
            let line_start_radius = radius;
            let line_end_radius = radius + OUTSIDE_LABEL_GAP;
            let outside_anchor_x = center_x + offset_x + cos * (radius + OUTSIDE_LABEL_GAP);
            let callout_end_x = center_x + offset_x + cos * line_end_radius;
            let effective_format = if pie_label_position == "callout" {
                effective_callout_label_format
            } else {
                pie_label_format
            };
            let text = build_slice_label_text(
                &slice.name,
                value,
                percentage,
                effective_format,
                effective_value_format,
            );

            if !text.is_empty() {
                let label_width = estimate_value_label_width(&text, slice_label_size(config));
                let inner_radius = radius * inner_radius_ratio;

                if pie_label_position == "inside" {
                    let label_radius = inner_radius + (radius - inner_radius) / 2.0;
                    let label_x = center_x + offset_x + cos * label_radius - label_width / 2.0;
                    let label_y = center_y + offset_y + sin * label_radius - label_height / 2.0;
                    let mut d = Def::new("richtext");
                    d.set("x", json!(label_x))
                        .set("y", json!(label_y))
                        .set("width", json!(label_width))
                        .set("height", json!(label_height))
                        .set(
                            "htmlContent",
                            json!(slice_label_html(config, "center", &text)),
                        )
                        .set(
                            "name",
                            json!(format!("{chart_name} Label {}", slice.category)),
                        )
                        .set(
                            "metadata",
                            json!({
                                "chartRole": "slice-label",
                                "chartCategory": slice.category,
                                "chartValue": value,
                                "chartPercentage": percentage,
                                "chartName": chart_name,
                            }),
                        );
                    nodes.push(d);
                } else if pie_label_position == "outside" {
                    let anchor_x = outside_anchor_x;
                    let anchor_y = center_y + offset_y + sin * (radius + OUTSIDE_LABEL_GAP);
                    let (text_align, label_x): (&'static str, f64) = if cos.abs() < 0.15 {
                        ("center", anchor_x - label_width / 2.0)
                    } else if cos >= 0.0 {
                        ("left", anchor_x)
                    } else {
                        ("right", anchor_x - label_width)
                    };
                    let label_y = anchor_y - label_height / 2.0;
                    pending.push(PendingOutsideLabel {
                        category: slice.category.clone(),
                        value,
                        percentage,
                        text,
                        label_width,
                        label_height,
                        text_align,
                        side_right: cos >= 0.0,
                        ideal_y: label_y,
                        ideal_x: label_x,
                        is_callout: false,
                        line_start_x: 0.0,
                        line_start_y: 0.0,
                    });
                } else {
                    // callout
                    let line_start_x = center_x + offset_x + cos * line_start_radius;
                    let line_start_y = center_y + offset_y + sin * line_start_radius;
                    let line_end_y = center_y + offset_y + sin * line_end_radius;
                    let text_align: &'static str = if cos >= 0.0 { "left" } else { "right" };
                    let raw_label_x = if cos >= 0.0 {
                        callout_end_x + CALLOUT_LABEL_HORIZONTAL_GAP
                    } else {
                        callout_end_x - label_width - CALLOUT_LABEL_HORIZONTAL_GAP
                    };
                    let label_x = clamp_callout_x(raw_label_x, label_width);
                    let label_y = line_end_y - label_height / 2.0;
                    pending.push(PendingOutsideLabel {
                        category: slice.category.clone(),
                        value,
                        percentage,
                        text,
                        label_width,
                        label_height,
                        text_align,
                        side_right: cos >= 0.0,
                        ideal_y: label_y,
                        ideal_x: label_x,
                        is_callout: true,
                        line_start_x,
                        line_start_y,
                    });
                }
            }
        }
    }

    if !pending.is_empty() {
        let candidates: Vec<LabelCandidate> = pending
            .iter()
            .map(|p| LabelCandidate {
                id: p.category.clone(),
                side_right: p.side_right,
                x: p.ideal_x,
                y: p.ideal_y,
                width: p.label_width,
                height: p.label_height,
            })
            .collect();
        let resolved = resolve_outside_label_positions(
            &candidates,
            DEFAULT_LABEL_MIN_GAP,
            Some(layout.plot_area.y),
            Some(layout.plot_area.y + layout.plot_area.height),
        );
        for (p, position) in pending.iter().zip(resolved.iter()) {
            if p.is_callout {
                let is_right = p.side_right;
                let line_end_x = if is_right {
                    position.x - CALLOUT_LABEL_HORIZONTAL_GAP
                } else {
                    position.x + p.label_width + CALLOUT_LABEL_HORIZONTAL_GAP
                };
                let line_end_y = position.y + p.label_height / 2.0;
                let line_would_point_inward = if is_right {
                    line_end_x < p.line_start_x
                } else {
                    line_end_x > p.line_start_x
                };
                if !line_would_point_inward {
                    let mut d = Def::new("line");
                    d.set(
                        "points",
                        json!([p.line_start_x, p.line_start_y, line_end_x, line_end_y]),
                    )
                    .stroke_color(config.s("labelColor").map(|c| json!(c)).unwrap_or(Value::Null),
                    )
                    .stroke_width(json!(1.0))
                    .set("name", json!(format!("{chart_name} Callout {}", p.category)))
                    .set(
                        "metadata",
                        json!({"chartRole": "slice-label-line", "chartCategory": p.category, "chartName": chart_name}),
                    );
                    nodes.push(d);
                }
            }
            let mut d = Def::new("richtext");
            d.set("x", json!(position.x))
                .set("y", json!(position.y))
                .set("width", json!(p.label_width))
                .set("height", json!(p.label_height))
                .set(
                    "htmlContent",
                    json!(slice_label_html(config, p.text_align, &p.text)),
                )
                .set("name", json!(format!("{chart_name} Label {}", p.category)))
                .set(
                    "metadata",
                    json!({
                        "chartRole": "slice-label",
                        "chartCategory": p.category,
                        "chartValue": p.value,
                        "chartPercentage": p.percentage,
                        "chartName": chart_name,
                    }),
                );
            nodes.push(d);
        }
    }

    let center_label = config.s("centerLabel").unwrap_or("");
    let center_sublabel = config.s("centerSublabel").unwrap_or("");
    if (!center_label.is_empty() || !center_sublabel.is_empty()) && inner_radius_ratio > 0.0 {
        let inner_radius = radius * inner_radius_ratio;
        let inner_diameter = inner_radius * 2.0;
        let label_width = js_max(40.0, inner_diameter);
        let sizes = resolve_center_text_sizes(
            inner_diameter,
            center_label,
            center_sublabel,
            config.f("centerSize"),
        );
        let label_height = if !center_label.is_empty() {
            sizes.label + 4.0
        } else {
            0.0
        };
        let sublabel_height = if !center_sublabel.is_empty() {
            sizes.sublabel + 4.0
        } else {
            0.0
        };
        let gap = if !center_label.is_empty() && !center_sublabel.is_empty() {
            4.0
        } else {
            0.0
        };
        let total_height = label_height + sublabel_height + gap;
        let mut cursor_y = center_y - total_height / 2.0;

        if !center_label.is_empty() {
            let mut d = Def::new("richtext");
            d.set("x", json!(center_x - label_width / 2.0))
                .set("y", json!(cursor_y))
                .set("width", json!(label_width))
                .set("height", json!(label_height))
                .set(
                    "htmlContent",
                    json!(build_styled_html(
                        &TextStyleProps {
                            font_size: Some(sizes.label),
                            font_family: config.s("fontFamily"),
                            text_align: Some("center"),
                            color: config.s("centerLabelColor"),
                            ..Default::default()
                        },
                        &escape_html(center_label),
                    )),
                )
                .set("name", json!(format!("{chart_name} Center Label")))
                .set(
                    "metadata",
                    json!({"chartRole": "center-label", "chartName": chart_name}),
                );
            nodes.push(d);
            cursor_y += label_height + gap;
        }
        if !center_sublabel.is_empty() {
            let mut d = Def::new("richtext");
            d.set("x", json!(center_x - label_width / 2.0))
                .set("y", json!(cursor_y))
                .set("width", json!(label_width))
                .set("height", json!(sublabel_height))
                .set(
                    "htmlContent",
                    json!(build_styled_html(
                        &TextStyleProps {
                            font_size: Some(sizes.sublabel),
                            font_family: config.s("fontFamily"),
                            text_align: Some("center"),
                            color: config.s("centerSublabelColor"),
                            ..Default::default()
                        },
                        &escape_html(center_sublabel),
                    )),
                )
                .set("name", json!(format!("{chart_name} Center Sublabel")))
                .set(
                    "metadata",
                    json!({"chartRole": "center-sublabel", "chartName": chart_name}),
                );
            nodes.push(d);
        }
    }

    let should_render_legend = (slices.len() > 1 || config.show_legend_true())
        && !config.show_legend_false()
        && layout.legend_area.is_some();
    if should_render_legend {
        if let Some(legend_area) = layout.legend_area {
            let names: Vec<String> = slices.iter().map(|s| s.name.clone()).collect();
            let items = build_legend_items(&names, styles, &config.palette());
            nodes.extend(generate_legend_nodes(
                &items,
                config,
                legend_area,
                chart_name,
                None,
            ));
        }
    }

    Ok(nodes)
}
