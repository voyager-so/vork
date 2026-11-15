//! Port of `utils/markup/elements/charts/{constants,layout}.ts`.
//! All arithmetic is NaN-propagating to mirror JS `undefined` math on
//! partially-populated configs (the chart-gated fixture relies on this).

use super::ticks::TickError;
use super::cfg::{
    aggregate_bar_values, bar_value_domain, num, resolve_bar_stacking, resolve_chart_dimensions,
    BarValueEntry, ConfigView, DataView, SpecView,
};
use super::format::{
    format_axis_tick, format_number_with_style, get_effective_axis_format,
    resolve_effective_value_label_format,
};
use super::jsmath::{js_max, js_min};
use super::scales::{get_display_range_from_ticks, is_log_domain_valid};
use super::value_axis_bounds::{apply_value_axis_bounds, resolve_value_axis_bounds};
use crate::value::Value;

// constants.ts
pub const TITLE_SUBTITLE_GAP: f64 = 4.0;
pub const TITLE_BOTTOM_GAP: f64 = 8.0;
pub const AXIS_TICK_LENGTH: f64 = 4.0;
pub const AXIS_LABEL_GAP: f64 = 4.0;
pub const MIN_Y_AXIS_LABEL_WIDTH: f64 = 40.0;
pub const MIN_X_AXIS_LABEL_WIDTH: f64 = 40.0;
pub const X_AXIS_LABEL_HEIGHT: f64 = 30.0;
pub const AXIS_TITLE_TEXT_PADDING: f64 = 4.0;
pub const AXIS_TITLE_GAP: f64 = 6.0;
pub const SOURCE_FOOTER_GAP: f64 = 6.0;
pub const VALUE_LABEL_GAP: f64 = 4.0;
pub const LABEL_WIDTH_PADDING: f64 = 10.0;
pub const MIN_VALUE_LABEL_WIDTH: f64 = 40.0;
pub const OUTSIDE_VALUE_LABEL_WIDTH: f64 = 60.0;
pub const MAX_VALUE_LABEL_BAND_RATIO: f64 = 0.5;
pub const CHARACTER_WIDTH_FONT_SIZE_RATIO: f64 = 0.65;
pub const LEGEND_MARKER_LABEL_GAP: f64 = 6.0;
pub const LEGEND_LINE_MARKER_WIDTH_MULTIPLIER: f64 = 1.5;
pub const LEGEND_LABEL_PADDING: f64 = 2.0;

#[derive(Clone, Copy, Debug, Default)]
pub struct Bounds {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Clone, Debug, Default)]
pub struct ChartLayout {
    pub bounds: Bounds,
    pub plot_area: Bounds,
    pub title_area: Option<Bounds>,
    pub legend_area: Option<Bounds>,
    pub x_axis_area: Option<Bounds>,
    pub x_axis_title_area: Option<Bounds>,
    pub y_axis_area: Option<Bounds>,
    /// `yAxisTitleArea`: the band left of the Y tick labels for `yLabel`, drawn turned -90°.
    pub y_axis_title_area: Option<Bounds>,
    pub source_footer_area: Option<Bounds>,
}

/// `computeXAxisTitleHeight` (layout.ts:50-52).
pub fn compute_x_axis_title_height(font_size: f64) -> f64 {
    font_size + AXIS_TITLE_TEXT_PADDING + AXIS_TITLE_GAP
}

/// `xAxisBandHeight` (constants.ts) — height of the X-axis tick-label band.
/// Tick labels render `fontSize + 2` tall, so the historical fixed 30px band
/// overflows once `fontSize > 20`; the `max()` keeps every `fontSize <= 20`
/// chart byte-identical while growing the band for larger type.
///
/// Finite-guard: malformed persisted configs read `fontSize` as NaN by design
/// (cfg.rs), and `max(30, NaN)` would poison the positive-height check at the
/// consumption sites and drop the band entirely — fall back to the constant.
/// Keep in lockstep with the TS mirror.
pub fn x_axis_band_height(font_size: f64) -> f64 {
    if !font_size.is_finite() {
        return X_AXIS_LABEL_HEIGHT;
    }
    js_max(X_AXIS_LABEL_HEIGHT, font_size + 2.0 + 2.0 * AXIS_LABEL_GAP)
}

/// `resolveBoxValue` (layout.ts:146-157) — number | [t,r,b,l] | undefined
/// (undefined → NaN sides, like JS).
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

/// `estimateLabelWidth` (layout.ts:180-182) — `label_ems` at 0.6.
fn estimate_label_width(label: &str, font_size: f64) -> f64 {
    (super::format::label_ems(label, 0.6) * font_size).ceil() + LABEL_WIDTH_PADDING
}

fn get_legend_marker_width(config: ConfigView) -> f64 {
    if config.s("legendMarker") == Some("line") {
        config.f("legendMarkerSize") * 1.5
    } else {
        config.f("legendMarkerSize")
    }
}

fn compute_legend_item_width(label: &str, config: ConfigView) -> f64 {
    get_legend_marker_width(config)
        + LEGEND_MARKER_LABEL_GAP
        + estimate_label_width(label, config.f("fontSize"))
}

fn compute_legend_thickness(config: ConfigView, data: &DataView, position: &str, available_width: f64) -> f64 {
    let row_height = js_max(config.f("legendMarkerSize"), config.f("fontSize"));
    if position == "left" || position == "right" {
        let max_item_width = data.series.iter().fold(row_height, |m, s| {
            js_max(m, compute_legend_item_width(s, config))
        });
        return max_item_width + config.f("legendGap");
    }
    // A top or bottom legend wraps into as many rows as its items need at this width (`generate_legend_nodes` lays
    // them out the same way); one row when there are no series yet.
    let widths: Vec<f64> = data
        .series
        .iter()
        .map(|s| {
            get_legend_marker_width(config)
                + LEGEND_MARKER_LABEL_GAP
                + estimate_legend_label_width(s, config.f("fontSize"))
        })
        .collect();
    let rows = legend_rows(&widths, available_width, config.f("legendGap")).len().max(1) as f64;
    rows * row_height + rows * config.f("legendGap")
}

/// `legendRows` (legend.ts): a top or bottom legend's rows, item indices in order, a row taking items while they fit
/// `available` (`gap` between). An item wider than `available` gets a row of its own.
pub fn legend_rows(widths: &[f64], available: f64, gap: f64) -> Vec<Vec<usize>> {
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut row: Vec<usize> = Vec::new();
    let mut used = 0.0;
    for (i, &width) in widths.iter().enumerate() {
        let next = if row.is_empty() { width } else { used + gap + width };
        if !row.is_empty() && next > available {
            rows.push(std::mem::take(&mut row));
            row.push(i);
            used = width;
        } else {
            row.push(i);
            used = next;
        }
    }
    if !row.is_empty() {
        rows.push(row);
    }
    rows
}

fn get_max_category_label_width(data: &DataView, font_size: f64) -> f64 {
    if data.categories.is_empty() {
        return 0.0;
    }
    data.categories
        .iter()
        .fold(0.0, |m, key| {
            js_max(m, estimate_label_width(data.category_label(key), font_size))
        })
}

/// `estimateYAxisLabelWidth` (layout.ts:189-231).
fn estimate_y_axis_label_width(data: &DataView, config: ConfigView, spec: &SpecView) -> Result<f64, TickError> {
    if data.rows.is_empty() {
        return Ok(MIN_Y_AXIS_LABEL_WIDTH);
    }
    // The value range from the data: a bar chart's is the generator's own (stacks reach their totals, so a stack
    // of 7.5 + 5.3 + 2.1 draws a "20" tick the rows alone would size the gutter for as "8"), any other chart's its
    // rows.
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut y_format = config.s("yFormat");
    let mut stacked = false;
    if spec.s("type") == Some("bar") {
        let dims = resolve_chart_dimensions(data, spec);
        let stacking = resolve_bar_stacking(spec, dims.has_series_dimension);
        (min, max) = bar_value_domain(
            &aggregate_bar_values(data, dims.has_series_dimension),
            &dims.ordered_categories,
            stacking,
        );
        stacked = stacking.0;
        if stacking.1 && y_format == Some("auto") {
            y_format = Some("percent");
        }
    } else {
        for row in &data.rows {
            min = js_min(min, row.f("value"));
            max = js_max(max, row.f("value"));
        }
    }
    if !min.is_finite() || !max.is_finite() {
        return Ok(MIN_Y_AXIS_LABEL_WIDTH);
    }
    // Mirror the generators' bound handling (value_axis_bounds.rs): a pinned
    // end replaces the data-derived one, is not widened back to zero, and
    // survives nicing. The reserved gutter has to be measured against the
    // ticks the drawn domain produces.
    let bounds = resolve_value_axis_bounds(spec.f_opt("yMin"), spec.f_opt("yMax"), (min, max));
    let (bounded_min, bounded_max) = apply_value_axis_bounds((min, max), &bounds);
    min = bounded_min;
    max = bounded_max;
    // Stacked bars are never log (the generator falls back to linear).
    let label_scale_kind =
        if config.s("yScale") == Some("log") && !stacked && is_log_domain_valid(min, max) {
            "log"
        } else {
            "linear"
        };
    if label_scale_kind != "log" {
        if bounds.min.is_none() {
            min = js_min(min, 0.0);
        }
        if bounds.max.is_none() {
            max = js_max(max, 0.0);
        }
    }
    let display_range = apply_value_axis_bounds(
        get_display_range_from_ticks((min, max), config.f("yTickCount"), label_scale_kind),
        &bounds,
    );
    let (ticks, effective_format) =
        get_effective_axis_format(display_range, config.f("yTickCount"), y_format)?;
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

/// `shouldShowLegend` (layout.ts:233-237).
fn should_show_legend(config: ConfigView, data: &DataView) -> bool {
    if config.show_legend_true() && data.rows.is_empty() {
        return true;
    }
    if data.series.len() <= 1 {
        return false;
    }
    config.show_legend_true() || config.show_legend_auto()
}

fn apply_legend_layout(
    position: &str,
    thickness: f64,
    top: f64,
    bottom: f64,
    left: f64,
    right: f64,
) -> Bounds {
    let width = right - left;
    let height = bottom - top;
    match position {
        "top" => Bounds {
            x: left,
            y: top,
            width,
            height: thickness,
        },
        "bottom" => Bounds {
            x: left,
            y: bottom - thickness,
            width,
            height: thickness,
        },
        "left" => Bounds {
            x: left,
            y: top,
            width: thickness,
            height,
        },
        _ => Bounds {
            x: right - thickness,
            y: top,
            width: thickness,
            height,
        },
    }
}

/// `estimateOutsideValueLabelBand` (layout.ts).
///
/// Width reserved on the right edge of a horizontal bar chart for value labels
/// that sit OUTSIDE their bar. Measures the AGGREGATED bar entries (rows
/// sharing a category/series pair are summed into one bar and the sum is what
/// gets formatted), and only the entries that land on the right side —
/// `add_value_label` sends negative values to the left of the bar.
fn estimate_outside_value_label_band(
    spec: &SpecView,
    data: &DataView,
    available_width: f64,
) -> f64 {
    let config = spec.config;
    let dims = resolve_chart_dimensions(data, spec);
    let entries: Vec<BarValueEntry> = aggregate_bar_values(data, dims.has_series_dimension)
        .into_values()
        .flat_map(|cat_map| cat_map.into_values())
        .filter(|entry| entry.value >= 0.0)
        .collect();
    if entries.is_empty() {
        return 0.0;
    }

    let preferred_format = if config.s("labelFormat") == Some("auto") {
        config.s("valueFormat")
    } else {
        config.s("labelFormat")
    };
    let generated_values: Vec<f64> = entries
        .iter()
        .filter(|entry| entry.label.is_none())
        .map(|entry| entry.value)
        .collect();
    let generated_format = resolve_effective_value_label_format(
        &generated_values,
        preferred_format,
        config.s("valueDensity").unwrap_or("auto"),
        available_width,
        config.f("valueLabelSize"),
    );

    let mut widest_label_width = 0.0_f64;
    for entry in &entries {
        let label_text = match &entry.label {
            Some(l) => l.clone(),
            None => format_number_with_style(entry.value, generated_format),
        };
        if label_text.is_empty() {
            continue;
        }
        widest_label_width = js_max(
            widest_label_width,
            estimate_value_label_width(&label_text, config.f("valueLabelSize")),
        );
    }
    if widest_label_width == 0.0 {
        return 0.0;
    }

    let band = js_max(OUTSIDE_VALUE_LABEL_WIDTH, widest_label_width) + VALUE_LABEL_GAP;
    js_min(band, available_width * MAX_VALUE_LABEL_BAND_RATIO)
}

/// `calculateChartLayout` (layout.ts:263-432). The generators always call
/// this with `{ ...spec, x: 0, y: 0 }`.
pub fn calculate_chart_layout(spec: &SpecView, data: &DataView) -> Result<ChartLayout, TickError> {
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
        let title_gap = if spec.ty() == "pie" {
            TITLE_BOTTOM_GAP
        } else {
            js_max(
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
            )
        };
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
        let position = config.s("legendPosition").unwrap_or("");
        let thickness = compute_legend_thickness(config, data, position, right - left);
        match position {
            "top" => {
                legend_area = Some(apply_legend_layout(
                    position, thickness, top, bottom, left, right,
                ));
                top += thickness;
            }
            "bottom" => {
                legend_area = Some(apply_legend_layout(
                    position, thickness, top, bottom, left, right,
                ));
                bottom -= thickness;
            }
            "left" => {
                legend_area = Some(apply_legend_layout(
                    position, thickness, top, bottom, left, right,
                ));
                left += thickness;
            }
            "right" => {
                legend_area = Some(apply_legend_layout(
                    position, thickness, top, bottom, left, right,
                ));
                right -= thickness;
            }
            // TS: unknown position → applyLegendLayout's exhaustive switch
            // yields undefined and no space is reserved.
            _ => {}
        }
    }

    let has_data = !data.rows.is_empty();

    // Y-axis title band (left of the Y tick-label band): every chart with axes (not pie), a non-empty `yLabel`,
    // and a Y axis drawn (its line or its tick labels) to name.
    let mut y_axis_title_area = None;
    let has_y_axis_title = has_data
        && spec.ty() != "pie"
        && spec
            .s("yLabel")
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false)
        && (config.b("showYAxis") || config.b("showYLabels"));
    if has_y_axis_title {
        let w = compute_x_axis_title_height(config.f("fontSize"));
        y_axis_title_area = Some(Bounds {
            x: left,
            y: top,
            width: w,
            height: js_max(0.0, bottom - top),
        });
        left += w;
    }

    // Y-axis area
    let mut y_axis_area = None;
    let should_reserve_y = has_data && (config.b("showYLabels") || config.b("showYAxis"));
    let orientation = spec.s("orientation").unwrap_or("vertical");
    let mut y_axis_width = 0.0;
    if should_reserve_y {
        if config.b("showYLabels") {
            if orientation == "horizontal" {
                let max_cat = get_max_category_label_width(data, config.f("fontSize"));
                y_axis_width = js_max(MIN_Y_AXIS_LABEL_WIDTH, max_cat);
            } else {
                y_axis_width = estimate_y_axis_label_width(data, config, spec)?;
            }
        } else {
            y_axis_width = config.f("axisWidth");
        }
    }
    if y_axis_width > 0.0 {
        let y_axis_height = js_max(0.0, bottom - top);
        y_axis_area = Some(Bounds {
            x: left,
            y: top,
            width: y_axis_width,
            height: y_axis_height,
        });
        left += y_axis_width;
    }

    // Outside value-label band (right side), horizontal bar charts only.
    if has_data && spec.ty() == "bar" && orientation == "horizontal" && config.b("showLabels") {
        right -= estimate_outside_value_label_band(spec, data, js_max(0.0, right - left));
    }

    // X-axis title area: every chart with axes (not pie)
    let mut x_axis_title_area = None;
    let has_x_axis_title = has_data
        && spec.ty() != "pie"
        && spec
            .s("xLabel")
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false)
        && (config.b("showXAxis") || config.b("showXLabels"));
    if has_x_axis_title {
        let h = compute_x_axis_title_height(config.f("fontSize"));
        let w = js_max(0.0, right - left);
        x_axis_title_area = Some(Bounds {
            x: left,
            y: bottom - h,
            width: w,
            height: h,
        });
        bottom -= h;
    }

    // X-axis area
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
        let w = js_max(0.0, right - left);
        x_axis_area = Some(Bounds {
            x: left,
            y: bottom - x_axis_height,
            width: w,
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

    Ok(ChartLayout {
        bounds,
        plot_area,
        title_area,
        legend_area,
        x_axis_area,
        x_axis_title_area,
        y_axis_area,
        y_axis_title_area,
        source_footer_area,
    })
}

/// `estimateLabelWidth` for value labels (barChart.ts:34-35) and pie labels
/// (pieChart.ts:45-46) — same formula, 0.65 ratio.
pub fn estimate_value_label_width(label: &str, font_size: f64) -> f64 {
    (super::format::label_ems(label, CHARACTER_WIDTH_FONT_SIZE_RATIO) * font_size).ceil() + LABEL_WIDTH_PADDING
}

/// `estimateLegendLabelWidth` (legend.ts:204-206) — 0.6 ratio + padding 2.
pub fn estimate_legend_label_width(label: &str, font_size: f64) -> f64 {
    (super::format::label_ems(label, 0.6) * font_size).ceil() + LEGEND_LABEL_PADDING
}

#[cfg(test)]
mod tests {
    use super::{x_axis_band_height, X_AXIS_LABEL_HEIGHT};

    /// The `max()` form is identity for every persisted `fontSize <= 20`
    /// (the byte-exactness contract the unchanged oracle fixtures pin).
    #[test]
    fn x_axis_band_is_identity_up_to_20() {
        for fs in [1.0, 12.0, 16.0, 20.0] {
            assert_eq!(x_axis_band_height(fs), X_AXIS_LABEL_HEIGHT);
        }
    }

    #[test]
    fn x_axis_band_grows_past_20() {
        assert_eq!(x_axis_band_height(21.0), 31.0);
        assert_eq!(x_axis_band_height(27.0), 37.0);
        assert_eq!(x_axis_band_height(30.0), 40.0);
    }

    /// Finite-guard: malformed persisted configs read `fontSize` as NaN
    /// (cfg.rs); the band must fall back to the constant instead of NaN —
    /// which would fail the `> 0` check and drop the band entirely.
    #[test]
    fn x_axis_band_non_finite_falls_back() {
        assert_eq!(x_axis_band_height(f64::NAN), X_AXIS_LABEL_HEIGHT);
        assert_eq!(x_axis_band_height(f64::INFINITY), X_AXIS_LABEL_HEIGHT);
        assert_eq!(x_axis_band_height(f64::NEG_INFINITY), X_AXIS_LABEL_HEIGHT);
    }
}
