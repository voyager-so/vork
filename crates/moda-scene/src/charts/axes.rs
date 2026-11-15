//! Port of `utils/markup/elements/charts/axes.ts` (+ the dash presets from
//! `core/scene/strokeDashPresets.ts` the chart code resolves).

use super::ticks::TickError;
use crate::value::{json, Value};

use super::cfg::{category_label, ConfigView, JsonMap};
use super::defs::Def;
use super::format::{format_axis_tick, format_number};
use super::html::{build_styled_html, escape_html, raw_size, TextStyleProps};
use super::jsmath::js_max;
use super::layout::{
    Bounds, ChartLayout, AXIS_TITLE_GAP, AXIS_TITLE_TEXT_PADDING, CHARACTER_WIDTH_FONT_SIZE_RATIO,
    MIN_X_AXIS_LABEL_WIDTH,
};
use super::scales::{CategoricalScale, ValueScale};

/// `resolveDashPatternFromPreset` (strokeDashPresets.ts:40-48), default
/// solidMode 'undefined'.
pub fn resolve_dash_pattern(preset: Option<&str>) -> Option<Vec<f64>> {
    match preset {
        Some("dashed") => Some(vec![8.0, 8.0]),
        Some("dotted") => Some(vec![2.0, 4.0]),
        Some("dash-dot") => Some(vec![12.0, 6.0, 2.0, 6.0]),
        // 'solid', None and unknown presets → no dash. (The TS would throw
        // on an unknown preset id; markup parsing guarantees the enum.)
        _ => None,
    }
}

pub fn dash_value(dash: &Option<Vec<f64>>) -> Value {
    match dash {
        Some(d) => json!(d),
        None => Value::Null, // explicit-undefined convention (defs.rs)
    }
}

#[derive(Clone, Copy)]
pub struct AxisConfig<'a> {
    pub show: bool,
    pub show_labels: bool,
    pub show_ticks: bool,
    pub color: Option<&'a str>,
    pub width: f64,
    pub tick_length: f64,
    pub label_gap: f64,
}

#[derive(Clone, Copy)]
pub struct FontConfig<'a> {
    pub font_family: Option<&'a str>,
    pub font_size: f64,
    pub label_color: Option<&'a str>,
}

/// `createAxisAndFontConfig` (generators/shared.ts:6-30).
pub fn create_axis_and_font_config<'a>(
    config: ConfigView<'a>,
) -> (AxisConfig<'a>, AxisConfig<'a>, FontConfig<'a>) {
    let y = AxisConfig {
        show: config.b("showYAxis"),
        show_labels: config.b("showYLabels"),
        show_ticks: config.b("showYAxis") || config.b("showYLabels"),
        color: config.s("axisColor"),
        width: config.f("axisWidth"),
        tick_length: super::layout::AXIS_TICK_LENGTH,
        label_gap: super::layout::AXIS_LABEL_GAP,
    };
    let x = AxisConfig {
        show: config.b("showXAxis"),
        show_labels: config.b("showXLabels"),
        show_ticks: config.b("showXAxis") || config.b("showXLabels"),
        color: config.s("axisColor"),
        width: config.f("axisWidth"),
        tick_length: super::layout::AXIS_TICK_LENGTH,
        label_gap: super::layout::AXIS_LABEL_GAP,
    };
    let font = FontConfig {
        font_family: config.s("fontFamily"),
        font_size: config.f("fontSize"),
        label_color: config.s("labelColor"),
    };
    (y, x, font)
}

fn line_def(
    points: [f64; 4],
    color: Option<&str>,
    width: f64,
    name: String,
    metadata: Value,
) -> Def {
    let mut d = Def::new("line");
    d.set("points", json!(points));
    d.stroke_color(color.map(|c| json!(c)).unwrap_or(Value::Null));
    d.stroke_width(json!(width));
    d.set("name", json!(name));
    d.set("metadata", metadata);
    d
}

fn label_def(
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    html: String,
    name: String,
    metadata: Value,
) -> Def {
    let mut d = Def::new("richtext");
    d.set("x", json!(x));
    d.set("y", json!(y));
    d.set("width", json!(width));
    d.set("height", json!(height));
    d.set("htmlContent", json!(html));
    d.set("name", json!(name));
    d.set("metadata", metadata);
    d
}

fn label_html(font: &FontConfig, align: &str, text: &str) -> String {
    build_styled_html(
        &TextStyleProps {
            // Raw config read: a missing fontSize drops the declaration,
            // mirroring the TS undefined-drop (see html.rs raw_size).
            font_size: raw_size(font.font_size),
            font_family: font.font_family,
            text_align: Some(align),
            color: font.label_color,
            ..Default::default()
        },
        &escape_html(text),
    )
}

/// `generateCategoricalAxisNodes` (axes.ts:60-190).
#[allow(clippy::too_many_arguments)]
pub fn generate_categorical_axis_nodes(
    categories: &[String],
    scale: &CategoricalScale,
    position: &str, // 'bottom' | 'left'
    plot: Bounds,
    axis_area: Option<Bounds>,
    config: &AxisConfig,
    font: &FontConfig,
    chart_name: &str,
    baseline: Option<f64>,
    category_labels: Option<&JsonMap>,
) -> Vec<Def> {
    let mut nodes = Vec::new();
    if categories.is_empty() {
        return nodes;
    }
    let bottom = position == "bottom";
    let axis_role = if bottom { "axis-x" } else { "axis-y" };
    let tick_role = if bottom { "axis-x-tick" } else { "axis-y-tick" };
    let label_role = if bottom { "x-label" } else { "y-label" };
    let axis_name = if bottom { "X" } else { "Y" };
    let band_size = scale.bandwidth();

    if config.show {
        if bottom {
            let axis_y = baseline.unwrap_or(plot.y + plot.height);
            nodes.push(line_def(
                [plot.x, axis_y, plot.x + plot.width, axis_y],
                config.color,
                config.width,
                format!("{chart_name} {axis_name} Axis"),
                json!({"chartRole": axis_role, "chartName": chart_name}),
            ));
        } else {
            let axis_x = baseline.unwrap_or(plot.x);
            nodes.push(line_def(
                [axis_x, plot.y, axis_x, plot.y + plot.height],
                config.color,
                config.width,
                format!("{chart_name} {axis_name} Axis"),
                json!({"chartRole": axis_role, "chartName": chart_name}),
            ));
        }
    }

    if config.show_ticks || config.show_labels {
        if bottom {
            let axis_y = baseline.unwrap_or(plot.y + plot.height);
            let tick_y = axis_y + config.tick_length;
            for category in categories {
                let center_x = plot.x + scale.scale(category) + band_size / 2.0;
                nodes.push(line_def(
                    [center_x, axis_y, center_x, tick_y],
                    config.color,
                    config.width,
                    format!("{chart_name} X Tick {category}"),
                    json!({"chartRole": tick_role, "chartCategory": category, "chartName": chart_name}),
                ));
            }
        } else {
            let axis_x = baseline.unwrap_or(plot.x);
            for category in categories {
                let center_y = plot.y + scale.scale(category) + band_size / 2.0;
                nodes.push(line_def(
                    [axis_x - config.tick_length, center_y, axis_x, center_y],
                    config.color,
                    config.width,
                    format!("{chart_name} Y Tick {category}"),
                    json!({"chartRole": tick_role, "chartCategory": category, "chartName": chart_name}),
                ));
            }
        }
    }

    if config.show_labels {
        if let Some(area) = axis_area {
            if bottom {
                let label_y = area.y + (area.height - font.font_size) / 2.0;
                let step = if categories.len() > 1 {
                    scale.scale(&categories[1]) - scale.scale(&categories[0])
                } else {
                    plot.width
                };
                let labels: Vec<&str> = categories
                    .iter()
                    .map(|category| category_label(category_labels, category))
                    .collect();
                let (every, label_width) =
                    bottom_category_label_fit(&labels, step, band_size, font.font_size);
                for (index, category) in categories.iter().enumerate() {
                    if index % every != 0 {
                        continue;
                    }
                    nodes.push(label_def(
                        plot.x + scale.scale(category) + (band_size - label_width) / 2.0,
                        label_y,
                        label_width,
                        font.font_size + 2.0,
                        label_html(font, "center", category_label(category_labels, category)),
                        format!("{chart_name} X Label {category}"),
                        json!({"chartRole": label_role, "chartCategory": category, "chartName": chart_name}),
                    ));
                }
            } else {
                let label_width = js_max(0.0, area.width - config.tick_length - config.label_gap);
                for category in categories {
                    let center_y = plot.y + scale.scale(category) + band_size / 2.0;
                    nodes.push(label_def(
                        area.x,
                        center_y - font.font_size / 2.0,
                        label_width,
                        font.font_size + 2.0,
                        label_html(font, "right", category_label(category_labels, category)),
                        format!("{chart_name} Y Label {category}"),
                        json!({"chartRole": label_role, "chartCategory": category, "chartName": chart_name}),
                    ));
                }
            }
        }
    }

    nodes
}

/// `bottomCategoryLabelFit` (axes.ts): every how many categories a bottom
/// axis labels, and each label box's width (centred on its band). Labels that
/// fit their band (`label_ems` at `CHARACTER_WIDTH_FONT_SIZE_RATIO`) keep it;
/// a wider one broke inside a word, so every `every`-th category is labelled,
/// the fewest skipped for neighbours to clear each other a quarter em apart,
/// each box wide enough for the widest label.
pub fn bottom_category_label_fit(
    labels: &[&str],
    step: f64,
    band_size: f64,
    font_size: f64,
) -> (usize, f64) {
    let text = |label: &str| super::format::label_ems(label, CHARACTER_WIDTH_FONT_SIZE_RATIO) * font_size;
    // A size that isn't a number (a partial config) measures nothing: the labels keep their bands.
    if !font_size.is_finite() || labels.iter().all(|label| text(*label) <= band_size) {
        return (1, band_size);
    }
    let clears = |every: usize| {
        (every..labels.len()).step_by(every).all(|i| {
            every as f64 * step
                >= (text(labels[i - every]) + text(labels[i])) / 2.0 + font_size / 4.0
        })
    };
    let mut every = 1;
    while every < labels.len() && !clears(every) {
        every += 1;
    }
    // Boxes may overlap: each label is centred in its own, and the texts clear each other.
    let widest = labels
        .iter()
        .fold(band_size, |w, label| js_max(w, text(*label)));
    (every, js_max(widest, every as f64 * step - font_size / 4.0))
}

/// `fitBottomTickCount` (axes.ts): the tick count a value axis along the
/// bottom draws, lowered from `tick_count` until neighbouring labels (each
/// `CHARACTER_WIDTH_FONT_SIZE_RATIO` em a UTF-16 unit, a quarter em apart)
/// clear each other, down to 2. The grid takes the same count.
pub fn fit_bottom_tick_count(
    scale: &ValueScale,
    domain: (f64, f64),
    tick_count: f64,
    format: Option<&str>,
    font_size: f64,
) -> Result<f64, TickError> {
    let half = |tick: f64| {
        format_axis_tick(tick, format).encode_utf16().count() as f64
            * font_size
            * CHARACTER_WIDTH_FONT_SIZE_RATIO
            / 2.0
    };
    let mut remaining = super::ticks::MAX_TICKS;
    let mut fits = |count: f64| -> Result<bool, TickError> {
        let ticks = filter_ticks_to_domain(scale.ticks(count)?, domain);
        remaining = remaining.checked_sub(ticks.len()).ok_or(TickError)?;
        Ok(ticks.windows(2).all(|w| {
            (scale.scale(w[1]) - scale.scale(w[0])).abs()
                >= half(w[0]) + half(w[1]) + font_size / 4.0
        }))
    };
    super::ticks::check_count(tick_count)?;
    let mut count = tick_count;
    while count > 2.0 && !fits(count)? {
        count -= 1.0;
    }
    Ok(count)
}

/// `filterTicksToDomain` (axes.ts:463-469).
pub fn filter_ticks_to_domain(ticks: Vec<f64>, domain: (f64, f64)) -> Vec<f64> {
    let lo = super::jsmath::js_min(domain.0, domain.1);
    let hi = js_max(domain.0, domain.1);
    let epsilon = js_max(1e-6, (hi - lo) * 0.001);
    ticks
        .into_iter()
        .filter(|t| *t >= lo - epsilon && *t <= hi + epsilon)
        .collect()
}

/// `generateNumericAxisNodes` (axes.ts:229-388).
///
/// `formatter` overrides `format`/`formatAxisTick` when provided (axes.ts:207):
/// time axes pass d3-scaleUtc's multi-scale formatter (or `utcFormat`), which
/// picks a specifier per tick — something the numeric `format_axis_tick`
/// cannot express.
#[allow(clippy::too_many_arguments)]
pub fn generate_numeric_axis_nodes(
    scale: &ValueScale,
    domain: (f64, f64),
    tick_count: f64,
    format: Option<&str>,
    formatter: Option<&dyn Fn(f64) -> String>,
    position: &str, // 'bottom' | 'left' | 'right'
    plot: Bounds,
    axis_area: Option<Bounds>,
    config: &AxisConfig,
    font: &FontConfig,
    chart_name: &str,
    baseline: Option<f64>,
) -> Result<Vec<Def>, TickError> {
    let mut nodes = Vec::new();
    let ticks = filter_ticks_to_domain(scale.ticks(tick_count)?, domain);
    let format_tick = |v: f64| match formatter {
        Some(f) => f(v),
        None => format_axis_tick(v, format),
    };

    let bottom = position == "bottom";
    let right = position == "right";
    let axis_role = if bottom { "axis-x" } else { "axis-y" };
    let tick_role = if bottom { "axis-x-tick" } else { "axis-y-tick" };
    let label_role = if bottom { "x-label" } else { "y-label" };
    let axis_name = if bottom { "X" } else { "Y" };

    let bottom_anchor_y = baseline.unwrap_or(plot.y + plot.height);

    if config.show {
        if bottom {
            nodes.push(line_def(
                [
                    plot.x,
                    bottom_anchor_y,
                    plot.x + plot.width,
                    bottom_anchor_y,
                ],
                config.color,
                config.width,
                format!("{chart_name} {axis_name} Axis"),
                json!({"chartRole": axis_role, "chartName": chart_name}),
            ));
        } else {
            let axis_x = if right { plot.x + plot.width } else { plot.x };
            nodes.push(line_def(
                [axis_x, plot.y, axis_x, plot.y + plot.height],
                config.color,
                config.width,
                format!("{chart_name} {axis_name} Axis"),
                json!({"chartRole": axis_role, "chartName": chart_name}),
            ));
        }
    }

    if config.show_ticks || config.show_labels {
        if bottom {
            let tick_y = bottom_anchor_y + config.tick_length;
            for &tick in &ticks {
                let x = plot.x + scale.scale(tick);
                nodes.push(line_def(
                    [x, bottom_anchor_y, x, tick_y],
                    config.color,
                    config.width,
                    format!("{chart_name} X Tick {}", format_number(tick)),
                    json!({"chartRole": tick_role, "chartValue": tick, "chartName": chart_name}),
                ));
            }
        } else {
            let axis_x = if right { plot.x + plot.width } else { plot.x };
            let tick_offset = if right {
                config.tick_length
            } else {
                -config.tick_length
            };
            for &tick in &ticks {
                let y = plot.y + plot.height - scale.scale(tick);
                nodes.push(line_def(
                    [axis_x + tick_offset, y, axis_x, y],
                    config.color,
                    config.width,
                    format!("{chart_name} Y Tick {}", format_number(tick)),
                    json!({"chartRole": tick_role, "chartValue": tick, "chartName": chart_name}),
                ));
            }
        }
    }

    if config.show_labels {
        if let Some(area) = axis_area {
            if bottom {
                let label_y = area.y + (area.height - font.font_size) / 2.0;
                let max_label_length = ticks.iter().fold(0.0f64, |m, &t| {
                    js_max(m, format_tick(t).encode_utf16().count() as f64)
                });
                let label_width = js_max(
                    MIN_X_AXIS_LABEL_WIDTH,
                    (max_label_length * font.font_size * CHARACTER_WIDTH_FONT_SIZE_RATIO).ceil(),
                );
                for &tick in &ticks {
                    let label = format_tick(tick);
                    nodes.push(label_def(
                        plot.x + scale.scale(tick) - label_width / 2.0,
                        label_y,
                        label_width,
                        font.font_size + 2.0,
                        label_html(font, "center", &label),
                        format!("{chart_name} X Label {label}"),
                        json!({"chartRole": label_role, "chartValue": tick, "chartName": chart_name}),
                    ));
                }
            } else {
                let label_width = js_max(0.0, area.width - config.tick_length - config.label_gap);
                for &tick in &ticks {
                    let y = plot.y + plot.height - scale.scale(tick);
                    let label = format_tick(tick);
                    nodes.push(label_def(
                        area.x,
                        y - font.font_size / 2.0,
                        label_width,
                        font.font_size + 2.0,
                        label_html(font, "right", &label),
                        format!("{chart_name} Y Label {label}"),
                        json!({"chartRole": label_role, "chartValue": tick, "chartName": chart_name}),
                    ));
                }
            }
        }
    }

    Ok(nodes)
}

pub struct GridConfig<'a> {
    pub show: bool,
    pub color: Option<&'a str>,
    pub width: f64,
    pub style: Option<&'a str>,
}

/// `generateGridNodes` (axes.ts:407-447).
pub fn generate_grid_nodes(
    scale: &ValueScale,
    domain: (f64, f64),
    tick_count: f64,
    orientation: &str, // 'horizontal' | 'vertical'
    plot: Bounds,
    config: &GridConfig,
    chart_name: &str,
) -> Result<Vec<Def>, TickError> {
    if !config.show {
        return Ok(Vec::new());
    }
    let mut nodes = Vec::new();
    let ticks = filter_ticks_to_domain(scale.ticks(tick_count)?, domain);
    let grid_dash = resolve_dash_pattern(config.style);

    for &tick in &ticks {
        let mut d = if orientation == "horizontal" {
            let y = plot.y + plot.height - scale.scale(tick);
            line_def(
                [plot.x, y, plot.x + plot.width, y],
                config.color,
                config.width,
                format!("{chart_name} Grid {}", format_number(tick)),
                json!({"chartRole": "grid-line", "chartValue": tick, "chartName": chart_name}),
            )
        } else {
            let x = plot.x + scale.scale(tick);
            line_def(
                [x, plot.y, x, plot.y + plot.height],
                config.color,
                config.width,
                format!("{chart_name} Grid {}", format_number(tick)),
                json!({"chartRole": "grid-line", "chartValue": tick, "chartName": chart_name}),
            )
        };
        d.stroke_dash(dash_value(&grid_dash));
        nodes.push(d);
    }
    Ok(nodes)
}

/// `generateAxisTitleNodes` (axes.ts): `xLabel` centred under the plot in the band below the X tick labels,
/// `yLabel` turned -90° in the band left of the Y tick labels, each only where the layout reserved its band.
pub fn generate_axis_title_nodes(
    x_label: Option<&str>,
    y_label: Option<&str>,
    layout: &ChartLayout,
    font: &FontConfig,
    chart_name: &str,
) -> Vec<Def> {
    let plot = layout.plot_area;
    let height = font.font_size + AXIS_TITLE_TEXT_PADDING;
    let html = |title: &str| {
        build_styled_html(
            &TextStyleProps {
                font_size: raw_size(font.font_size),
                font_family: font.font_family,
                text_align: Some("center"),
                color: font.label_color,
                ..Default::default()
            },
            &escape_html(title),
        )
    };
    let mut nodes = Vec::new();
    if let (Some(title), Some(band)) = (x_label.filter(|s| !s.is_empty()), layout.x_axis_title_area)
    {
        let mut d = Def::new("richtext");
        d.set("x", json!(plot.x))
            .set("y", json!(band.y + AXIS_TITLE_GAP))
            .set("width", json!(plot.width))
            .set("height", json!(height))
            .set("htmlContent", json!(html(title)))
            .set("name", json!(format!("{chart_name} X Axis Title")))
            .set(
                "metadata",
                json!({"chartRole": "x-axis-title", "chartName": chart_name}),
            );
        nodes.push(d);
    }
    if let (Some(title), Some(band)) = (y_label.filter(|s| !s.is_empty()), layout.y_axis_title_area)
    {
        let mut d = Def::new("richtext");
        d.set("x", json!(band.x))
            .set("y", json!(plot.y + plot.height))
            .set("width", json!(plot.height))
            .set("height", json!(height))
            .set("htmlContent", json!(html(title)))
            .set("rotation", json!(-90.0))
            .set("name", json!(format!("{chart_name} Y Axis Title")))
            .set(
                "metadata",
                json!({"chartRole": "y-axis-title", "chartName": chart_name}),
            );
        nodes.push(d);
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bottom_category_labels_skip_until_they_fit() {
        // chartCategoryLabelFit.test.ts `bottomCategoryLabelFit`.
        assert_eq!(
            bottom_category_label_fit(&["Q1", "Q2"], 100.0, 80.0, 12.0),
            (1, 80.0)
        );
        assert_eq!(
            bottom_category_label_fit(&["Week 1", "Week 2"], 37.5, 30.0, f64::NAN),
            (1, 30.0)
        );
        let weeks = ["Week 1", "Week 2", "Week 3", "Week 4"];
        assert_eq!(
            bottom_category_label_fit(&weeks, 37.5, 30.0, 12.0),
            (2, 72.0)
        );
        assert_eq!(
            bottom_category_label_fit(&["A very long category name"], 400.0, 300.0, 12.0).0,
            1
        );
    }

    #[test]
    fn fit_bottom_tick_count_lowers_only_as_far_as_labels_need() {
        // chartBottomAxisFit.test.ts `fitBottomTickCount`.
        let scale = ValueScale::Linear {
            min: 0.0,
            max: 1000.0,
            range: 400.0,
        };
        assert_eq!(
            fit_bottom_tick_count(&scale, (0.0, 1000.0), 5.0, None, 12.0).unwrap(),
            5.0
        );
        assert_eq!(
            fit_bottom_tick_count(&scale, (0.0, 1000.0), 5.0, None, 40.0).unwrap(),
            3.0
        );
        assert_eq!(
            fit_bottom_tick_count(&scale, (0.0, 1000.0), 5.0, None, 400.0).unwrap(),
            2.0
        );
        assert_eq!(
            fit_bottom_tick_count(&scale, (0.0, 1000.0), 2.0, None, 400.0).unwrap(),
            2.0
        );
    }
}
