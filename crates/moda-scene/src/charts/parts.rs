//! Ports of the smaller chart part generators:
//! `legend.ts`, `title.ts`, `footer.ts`, `background.ts`, `markers.ts`,
//! `labelLayout.ts`, `regression.ts`.

use crate::value::{json, Map, Value};

use super::area_path::{concat_area_top_edges, generate_area_regions, AreaRegion};
use super::axes::dash_value;
use super::cfg::{category_label, point_color, ConfigView, DataView, JsonMap, SeriesStyleView};
use super::colorutil::{extract_primary_color, resolve_palette_color};
use super::curves::{generate_curve_points, path_pts_to_value, PathPt};
use super::defs::Def;
use super::fills::parse_fill;
use super::html::{build_styled_html, escape_html, legend_label_style, raw_size, TextStyleProps};
use super::jsmath::js_max;
use super::layout::{
    estimate_legend_label_width, legend_rows, Bounds, LEGEND_LINE_MARKER_WIDTH_MULTIPLIER,
    LEGEND_MARKER_LABEL_GAP, TITLE_SUBTITLE_GAP,
};

// ---------------------------------------------------------------------------
// legend.ts
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LegendItem {
    pub name: String,
    pub fill: String,
    pub stroke: Option<String>,
    pub stroke_width: Option<f64>,
    pub opacity: Option<f64>,
    /// The series line's dash pattern (a dashed or dotted line series): a `line` marker draws it.
    pub dash: Option<Vec<f64>>,
}

/// `buildLegendItems` (legend.ts:183-199).
pub fn build_legend_items(
    series_names: &[String],
    series_styles: super::cfg::SeriesStyles,
    palette: &[String],
) -> Vec<LegendItem> {
    series_names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let style = series_styles.get(name);
            let fill = style
                .and_then(|s| s.s_truthy("fill"))
                .map(str::to_string)
                .unwrap_or_else(|| resolve_palette_color(palette, index).to_string());
            LegendItem {
                name: name.clone(),
                fill,
                stroke: style.and_then(|s| s.s("stroke")).map(str::to_string),
                stroke_width: style.and_then(|s| s.f_opt("strokeWidth")),
                opacity: style.and_then(|s| s.f_opt("opacity")),
                dash: None,
            }
        })
        .collect()
}

fn legend_marker_width(marker: &str, marker_size: f64) -> f64 {
    if marker == "line" {
        marker_size * LEGEND_LINE_MARKER_WIDTH_MULTIPLIER
    } else {
        marker_size
    }
}

/// `legendDash` (legend.ts): a series' dash for a line marker `width` long, as the series draws it, shrunk until a
/// dash, a gap and a dash fit.
fn legend_dash(dash: Option<&[f64]>, width: f64) -> Option<Vec<f64>> {
    let dash = dash.filter(|d| !d.is_empty())?;
    let period: f64 = dash.iter().sum();
    if !(period > 0.0) {
        return None;
    }
    let k = (width / (period + dash[0])).min(1.0);
    Some(dash.iter().map(|n| n * k).collect())
}

/// `generateLegendNodes` (legend.ts:31-178).
pub fn generate_legend_nodes(
    items: &[LegendItem],
    config: ConfigView,
    legend_area: Bounds,
    chart_name: &str,
    marker_override: Option<&str>,
) -> Vec<Def> {
    if items.is_empty() {
        return Vec::new();
    }
    let marker = marker_override.or(config.s("legendMarker")).unwrap_or("");
    let marker_size = config.f("legendMarkerSize");
    let marker_width = legend_marker_width(marker, marker_size);
    let item_height = js_max(marker_size, config.f("fontSize"));
    let position = config.s("legendPosition").unwrap_or("");
    let is_horizontal = position == "top" || position == "bottom";
    let legend_gap = config.f("legendGap");
    let align = config.s("legendAlign").unwrap_or("");

    struct Item<'a> {
        item: &'a LegendItem,
        label_width: f64,
        item_width: f64,
    }
    let legend_items: Vec<Item> = items
        .iter()
        .map(|item| {
            let label_width = estimate_legend_label_width(&item.name, config.f("fontSize"));
            Item {
                item,
                label_width,
                item_width: marker_width + LEGEND_MARKER_LABEL_GAP + label_width,
            }
        })
        .collect();

    // Where each item goes. A top or bottom legend wraps into rows that fit the area's width (each row aligned on its
    // own); a left or right one is a column.
    let mut positions: Vec<(f64, f64)> = vec![(0.0, 0.0); legend_items.len()];
    if is_horizontal {
        let widths: Vec<f64> = legend_items.iter().map(|i| i.item_width).collect();
        let rows = legend_rows(&widths, legend_area.width, legend_gap);
        let block_height = rows.len() as f64 * item_height + legend_gap * (rows.len() as f64 - 1.0);
        let mut row_y = legend_area.y + (legend_area.height - block_height) / 2.0;
        for row in &rows {
            let row_width: f64 = row.iter().map(|&i| legend_items[i].item_width).sum::<f64>()
                + legend_gap * (row.len() as f64 - 1.0);
            let mut x = legend_area.x;
            if align == "center" {
                x = legend_area.x + (legend_area.width - row_width) / 2.0;
            } else if align == "end" {
                x = legend_area.x + legend_area.width - row_width;
            }
            // An item wider than the area starts at its left edge rather than before it.
            x = js_max(legend_area.x, x);
            for &i in row {
                positions[i] = (x, row_y);
                x += legend_items[i].item_width + legend_gap;
            }
            row_y += item_height + legend_gap;
        }
    } else {
        let total_height = legend_items.len() as f64 * item_height
            + legend_gap * js_max(0.0, legend_items.len() as f64 - 1.0);
        let mut start_y = legend_area.y;
        if align == "center" {
            start_y = legend_area.y + (legend_area.height - total_height) / 2.0;
        } else if align == "end" {
            start_y = legend_area.y + legend_area.height - total_height;
        }
        let max_item_width = legend_items
            .iter()
            .fold(f64::NEG_INFINITY, |m, i| js_max(m, i.item_width));
        let start_x = legend_area.x + (legend_area.width - max_item_width) / 2.0;
        for (i, p) in positions.iter_mut().enumerate() {
            *p = (start_x, start_y + i as f64 * (item_height + legend_gap));
        }
    }

    let mut nodes = Vec::new();

    for (index, li) in legend_items.iter().enumerate() {
        let item = li.item;
        let (item_x, item_y) = positions[index];

        let fill_props = parse_fill(&item.fill);
        let primary_color =
            extract_primary_color(&item.fill).unwrap_or_else(|| "#1a1a1a".to_string());

        // hit-target rect
        let mut hit = Def::new("rectangle");
        hit.set("x", json!(item_x))
            .set("y", json!(item_y))
            .set("width", json!(li.item_width))
            .set("height", json!(item_height))
            .fill_color(json!("transparent"))
            .set("name", json!(format!("{chart_name} Legend Item {}", item.name)))
            .set(
                "metadata",
                json!({"chartRole": "legend-item", "chartSeries": item.name, "chartName": chart_name}),
            );
        nodes.push(hit);

        let marker_x = item_x;
        let marker_y = item_y + (item_height - marker_size) / 2.0;

        if marker == "line" {
            let line_y = item_y + item_height / 2.0;
            let mut d = Def::new("line");
            d.set("points", json!([marker_x, line_y, marker_x + marker_width, line_y]))
                .stroke_color(json!(primary_color))
                .stroke_width(json!(2.0));
            if let Some(dash) = legend_dash(item.dash.as_deref(), marker_width) {
                d.stroke_dash(json!(dash));
            }
            d.set("name", json!(format!("{chart_name} Legend Marker {}", item.name)))
                .set(
                    "metadata",
                    json!({"chartRole": "legend-marker", "chartSeries": item.name, "chartName": chart_name}),
                );
            nodes.push(d);
        } else {
            let mut d = Def::new("rectangle");
            d.set("x", json!(marker_x))
                .set("y", json!(marker_y))
                .set("width", json!(marker_size))
                .set("height", json!(marker_size))
                .set("name", json!(format!("{chart_name} Legend Marker {}", item.name)))
                .set(
                    "metadata",
                    json!({"chartRole": "legend-marker", "chartSeries": item.name, "chartName": chart_name}),
                );
            for (k, v) in &fill_props {
                d.set(k, v.clone());
            }
            if marker == "circle" {
                d.set("cornerRadius", json!(marker_size / 2.0));
            }
            if let Some(stroke) = &item.stroke {
                if !stroke.is_empty() {
                    d.stroke_color(json!(stroke));
                }
            }
            if let Some(sw) = item.stroke_width {
                d.stroke_width(json!(sw));
            }
            if let Some(op) = item.opacity {
                d.set("opacity", json!(op));
            }
            nodes.push(d);
        }

        let style = legend_label_style(
            config.s("fontFamily").unwrap_or("undefined"),
            config.f("fontSize"),
            config.s("legendColor").unwrap_or("undefined"),
        );
        let mut label = Def::new("richtext");
        label
            .set("x", json!(marker_x + marker_width + LEGEND_MARKER_LABEL_GAP))
            .set("y", json!(item_y + (item_height - (config.f("fontSize") + 2.0)) / 2.0))
            .set("width", json!(li.label_width))
            .set("height", json!(config.f("fontSize") + 2.0))
            .set(
                "htmlContent",
                json!(format!("<p style=\"{style}\">{}</p>", escape_html(&item.name))),
            )
            .set("verticalAlign", json!("middle"))
            // The width is an estimate (0.6 em a character); a label never wraps at it ("Team" broke as "Tea/m").
            .set("resizeMode", json!("auto-width"))
            .set("name", json!(format!("{chart_name} Legend Label {}", item.name)))
            .set(
                "metadata",
                json!({"chartRole": "legend-label", "chartSeries": item.name, "chartName": chart_name}),
            );
        nodes.push(label);
    }
    nodes
}

// ---------------------------------------------------------------------------
// title.ts / footer.ts / background.ts
// ---------------------------------------------------------------------------

/// `generateTitleNodes` (title.ts:19-76).
pub fn generate_title_nodes(config: ConfigView, title_area: Bounds, chart_name: &str) -> Vec<Def> {
    let mut nodes = Vec::new();
    let title = config.s_truthy("title");
    let subtitle = config.s_truthy("subtitle");
    if title.is_none() && subtitle.is_none() {
        return nodes;
    }
    let mut current_y = title_area.y;
    if let Some(title) = title {
        let mut d = Def::new("richtext");
        d.set("x", json!(title_area.x))
            .set("y", json!(current_y))
            .set("width", json!(title_area.width))
            .set("height", json!(config.f("titleSize")))
            .set(
                "htmlContent",
                json!(build_styled_html(
                    &TextStyleProps {
                        font_size: raw_size(config.f("titleSize")),
                        font_weight: raw_size(config.f("titleWeight")),
                        font_family: config.s("fontFamily"),
                        text_align: config.s("titleAlign"),
                        color: config.s("titleColor"),
                    },
                    &escape_html(title),
                )),
            )
            .set("name", json!(format!("{chart_name} Title")))
            .set(
                "metadata",
                json!({"chartRole": "title", "chartName": chart_name}),
            );
        nodes.push(d);
        current_y += config.f("titleSize") + TITLE_SUBTITLE_GAP;
    }
    if let Some(subtitle) = subtitle {
        let mut d = Def::new("richtext");
        d.set("x", json!(title_area.x))
            .set("y", json!(current_y))
            .set("width", json!(title_area.width))
            .set("height", json!(config.f("subtitleSize")))
            .set(
                "htmlContent",
                json!(build_styled_html(
                    &TextStyleProps {
                        font_size: raw_size(config.f("subtitleSize")),
                        font_weight: Some(400.0),
                        font_family: config.s("fontFamily"),
                        text_align: config.s("titleAlign"),
                        color: config.s("subtitleColor"),
                    },
                    &escape_html(subtitle),
                )),
            )
            .set("name", json!(format!("{chart_name} Subtitle")))
            .set(
                "metadata",
                json!({"chartRole": "subtitle", "chartName": chart_name}),
            );
        nodes.push(d);
    }
    nodes
}

/// `generateSourceFooterNodes` (footer.ts:28-70).
pub fn generate_source_footer_nodes(
    source: Option<(&str, Option<&str>)>,
    config: ConfigView,
    footer_area: Option<Bounds>,
    chart_name: &str,
) -> Vec<Def> {
    let (Some((label, url)), Some(area)) = (source, footer_area) else {
        return Vec::new();
    };
    let label = label.trim();
    if label.is_empty() {
        return Vec::new();
    }
    let escaped = escape_html(label);
    let url = url.map(str::trim).filter(|u| !u.is_empty());
    let is_safe = url
        .map(|u| {
            let l = u.to_lowercase();
            l.starts_with("http://") || l.starts_with("https://")
        })
        .unwrap_or(false);
    let inner = if is_safe {
        format!(
            "<a href=\"{}\" target=\"_blank\" rel=\"noopener noreferrer\">{escaped}</a>",
            escape_html(url.unwrap())
        )
    } else {
        escaped
    };
    let mut metadata = Map::new();
    metadata.insert("chartRole".into(), json!("source"));
    metadata.insert("chartName".into(), json!(chart_name));
    if is_safe {
        metadata.insert("sourceUrl".into(), json!(url.unwrap()));
    }
    let mut d = Def::new("richtext");
    d.set("x", json!(area.x))
        .set("y", json!(area.y))
        .set("width", json!(area.width))
        .set("height", json!(area.height))
        .set(
            "htmlContent",
            json!(build_styled_html(
                &TextStyleProps {
                    font_size: raw_size(config.f("subtitleSize")),
                    font_weight: Some(400.0),
                    font_family: config.s("fontFamily"),
                    text_align: config.s("titleAlign"),
                    color: config.s("subtitleColor"),
                },
                &inner,
            )),
        )
        .set("name", json!(format!("{chart_name} Source")))
        .set("metadata", Value::Object(metadata));
    vec![d]
}

/// `generateBackgroundNode` (background.ts:18-35).
pub fn generate_background_node(
    config: ConfigView,
    width: f64,
    height: f64,
    chart_name: &str,
) -> Option<Def> {
    let bg = config.s_truthy("background")?;
    if bg == "transparent" {
        return None;
    }
    let fill_props = parse_fill(bg);
    let mut d = Def::new("rectangle");
    d.set("x", json!(0.0))
        .set("y", json!(0.0))
        .set("width", json!(width))
        .set("height", json!(height))
        .set(
            "cornerRadius",
            config.get("cornerRadius").cloned().unwrap_or(Value::Null),
        )
        .set("name", json!(format!("{chart_name} Background")))
        .set(
            "metadata",
            json!({"chartRole": "background", "chartName": chart_name}),
        );
    for (k, v) in &fill_props {
        d.set(k, v.clone());
    }
    Some(d)
}

// ---------------------------------------------------------------------------
// markers.ts
// ---------------------------------------------------------------------------

pub fn resolve_point_fill(
    style: Option<SeriesStyleView>,
    config_fill: Option<&str>,
    fallback: &str,
) -> String {
    style
        .and_then(|s| s.s_truthy("pointFill"))
        .or(config_fill.filter(|s| !s.is_empty()))
        .unwrap_or(fallback)
        .to_string()
}

pub fn resolve_point_stroke(
    style: Option<SeriesStyleView>,
    config_stroke: Option<&str>,
    fallback: &str,
) -> String {
    style
        .and_then(|s| s.s_truthy("pointStroke"))
        .or(config_stroke.filter(|s| !s.is_empty()))
        .unwrap_or(fallback)
        .to_string()
}

pub fn resolve_point_shape<'a>(
    style: Option<SeriesStyleView<'a>>,
    config_shape: Option<&'a str>,
) -> &'a str {
    style
        .and_then(|s| s.s("pointShape"))
        .or(config_shape)
        .unwrap_or("")
}

pub fn resolve_point_size(style: Option<SeriesStyleView>, config_size: f64) -> f64 {
    style
        .and_then(|s| s.f_opt("pointSize"))
        .unwrap_or(config_size)
}

pub fn resolve_series_curve<'a>(
    style: Option<SeriesStyleView<'a>>,
    config_curve: Option<&'a str>,
) -> &'a str {
    style
        .and_then(|s| s.s("curve"))
        .or(config_curve)
        .unwrap_or("")
}

pub fn resolve_series_show_points(style: Option<SeriesStyleView>, config_show: bool) -> bool {
    style
        .and_then(|s| s.b_opt("showPoints"))
        .unwrap_or(config_show)
}

pub fn resolve_series_fill_opacity(style: Option<SeriesStyleView>, config_opacity: f64) -> f64 {
    style
        .and_then(|s| s.f_opt("fillOpacity"))
        .unwrap_or(config_opacity)
}

pub struct PointMarkerArgs<'a> {
    pub shape: &'a str,
    pub size: f64,
    pub x: f64,
    pub y: f64,
    pub fill: &'a str,
    pub stroke: &'a str,
    pub stroke_width: f64,
    pub name: String,
    pub metadata: Value,
}

/// `getPointMarkerNode` (markers.ts:67-172).
pub fn get_point_marker_node(args: PointMarkerArgs) -> Def {
    let base = |d: &mut Def| {
        d.set("x", json!(args.x))
            .set("y", json!(args.y))
            .set("width", json!(args.size))
            .set("height", json!(args.size))
            .set("name", json!(args.name))
            .set("metadata", args.metadata.clone());
        d.stroke_color(if args.stroke.is_empty() {
                Value::Null
            } else {
                json!(args.stroke)
            },
        );
        d.stroke_width(if args.stroke_width > 0.0 {
                json!(args.stroke_width)
            } else {
                Value::Null
            },
        );
    };

    match args.shape {
        "square" => {
            let mut d = Def::new("rectangle");
            base(&mut d);
            for (k, v) in &parse_fill(args.fill) {
                d.set(k, v.clone());
            }
            d
        }
        "diamond" => {
            let mut d = Def::new("rectangle");
            base(&mut d);
            d.set("rotation", json!(45.0));
            for (k, v) in &parse_fill(args.fill) {
                d.set(k, v.clone());
            }
            d
        }
        "triangle" | "triangle-down" => {
            let mut d = Def::new("polygon");
            base(&mut d);
            d.set(
                "rotation",
                json!(if args.shape == "triangle-down" {
                    180.0
                } else {
                    0.0
                }),
            );
            d.set("sides", json!(3.0));
            d.set("scale", json!(1.0));
            for (k, v) in &parse_fill(args.fill) {
                d.set(k, v.clone());
            }
            d
        }
        "star" => {
            let mut d = Def::new("star");
            base(&mut d);
            d.set("rotation", json!(0.0));
            d.set("scale", json!(1.0));
            d.set("numPoints", json!(5.0));
            d.set("innerRadiusRatio", json!(0.5));
            for (k, v) in &parse_fill(args.fill) {
                d.set(k, v.clone());
            }
            d
        }
        "cross" | "x" => {
            let half = args.size / 2.0;
            let cx = args.x + half;
            let cy = args.y + half;
            let offset = args.size * 0.35;
            let pts = if args.shape == "x" {
                vec![
                    PathPt::Move {
                        x: cx - offset,
                        y: cy - offset,
                    },
                    PathPt::Line {
                        x: cx + offset,
                        y: cy + offset,
                    },
                    PathPt::Move {
                        x: cx + offset,
                        y: cy - offset,
                    },
                    PathPt::Line {
                        x: cx - offset,
                        y: cy + offset,
                    },
                ]
            } else {
                vec![
                    PathPt::Move {
                        x: cx - offset,
                        y: cy,
                    },
                    PathPt::Line {
                        x: cx + offset,
                        y: cy,
                    },
                    PathPt::Move {
                        x: cx,
                        y: cy - offset,
                    },
                    PathPt::Line {
                        x: cx,
                        y: cy + offset,
                    },
                ]
            };
            let mut d = Def::new("path");
            d.set("x", json!(0.0))
                .set("y", json!(0.0))
                .set("points", path_pts_to_value(&pts))
                .stroke_color(json!(if args.stroke.is_empty() {
                        args.fill
                    } else {
                        args.stroke
                    }),
                )
                .stroke_width(json!(if args.stroke_width > 0.0 {
                        args.stroke_width
                    } else {
                        2.0
                    }),
                )
                .fill_color(json!("transparent"))
                .set("closed", json!(false))
                .set("name", json!(args.name))
                .set("metadata", args.metadata.clone());
            d
        }
        _ => {
            let mut d = Def::new("ellipse");
            base(&mut d);
            for (k, v) in &parse_fill(args.fill) {
                d.set(k, v.clone());
            }
            d
        }
    }
}

#[derive(Clone)]
pub struct SeriesPoint {
    pub category: String,
    pub value: f64,
    pub x: f64,
    pub y: f64,
    pub x_time: Option<f64>,
}

/// `chartPointGroups` (generators/shared.ts): explicit empty cells break a
/// line/area rather than connecting across the gap. Runs of point indices;
/// a new run starts after a gap row of this series that lies between two
/// neighbouring points.
pub fn chart_point_groups(
    points: &[SeriesPoint],
    data: &DataView,
    series_key: &str,
    has_series: bool,
    categories: &[String],
) -> Vec<Vec<usize>> {
    let gaps: Vec<_> = data
        .rows
        .iter()
        .filter(|row| row.missing() && (!has_series || row.s("series") == Some(series_key)))
        .collect();
    // JS `indexOf`: -1 when absent.
    let index_of = |key: &str| -> i64 {
        categories
            .iter()
            .position(|c| c == key)
            .map_or(-1, |i| i as i64)
    };
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, point) in points.iter().enumerate() {
        let broken = i > 0 && {
            let previous = &points[i - 1];
            gaps.iter().any(|gap| {
                if let (Some(pt), Some(pv), Some(gt)) =
                    (point.x_time, previous.x_time, gap.f_opt("xTime"))
                {
                    return gt > pt.min(pv) && gt < pt.max(pv);
                }
                let at = index_of(gap.category_key());
                let (a, b) = (index_of(&previous.category), index_of(&point.category));
                at > a.min(b) && at < a.max(b)
            })
        };
        if i == 0 || broken {
            groups.push(Vec::new());
        }
        groups.last_mut().unwrap().push(i);
    }
    groups
}

/// Where the parsed fill fields (`fill`/`fillGradient`/…) sit relative to the
/// tail block (`fillOpacity`, `closed`, `opacity`, `name`, `metadata`).
///
/// serde's preserve-order map makes property insertion order observable in the
/// emitted JSON. The two Rust callers already differ: the standalone area
/// generator inserts parsed fill fields after `metadata` (`AfterTail`), while
/// combo inserts them before `fillOpacity` (`BeforeTail`). This enum preserves
/// each caller's existing byte order rather than smoothing the difference — a
/// behaviour-preserving fold must carry forward exactly what each caller
/// emitted before, regardless of whether that matches the TS source (the
/// standalone Rust arm predates this fold and already diverges from
/// `areaChart.ts`, which spreads `parseFill` before `fillOpacity`).
#[derive(Clone, Copy)]
pub enum AreaFillOrder {
    BeforeTail,
    AfterTail,
}

/// Shared non-stacked area-series emission: sign-crossing regions plus the
/// optional top-edge stroke. Point collection, scale selection, time-axis
/// sorting, and stacked-area generation stay in the callers — only
/// already-resolved rendering values enter here. `metadata` is the caller-built
/// region metadata (in its exact key order, without `chartRegionIndex`); it is
/// cloned per region and `chartRegionIndex` is appended for multiple regions.
/// The top-edge line clones the same map and overrides `chartRole`. Shadows
/// apply to filled area regions only, never to the top-edge stroke.
#[allow(clippy::too_many_arguments)]
pub fn append_unstacked_area_series(
    nodes: &mut Vec<Def>,
    local_top_points: &[(f64, f64)],
    groups: &[Vec<usize>],
    baseline_y_pixel: f64,
    curve: &str,
    plot_area: Bounds,
    fill_color: &str,
    fill_opacity: f64,
    stroke: Option<&str>,
    stroke_width: Option<f64>,
    dash: Option<&[f64]>,
    stroke_cap: Option<&str>,
    stroke_join: Option<&str>,
    opacity: Option<f64>,
    base_name: &str,
    metadata: &Map,
    shadow_props: &Map,
    fill_order: AreaFillOrder,
) {
    let dash_vec = dash.map(|d| d.to_vec());
    // One run of regions per gap-free group (`chart_point_groups`); the stroke
    // joins regions within a group only, so it breaks at a gap like the fill.
    let region_groups: Vec<Vec<AreaRegion>> = groups
        .iter()
        .map(|g| {
            let pts: Vec<(f64, f64)> = g.iter().map(|&i| local_top_points[i]).collect();
            generate_area_regions(&pts, baseline_y_pixel, curve)
        })
        .collect();
    let regions: Vec<&AreaRegion> = region_groups.iter().flatten().collect();
    let multiple_regions = regions.len() > 1;
    let stroke_on_fill = stroke.is_some() && local_top_points.len() < 2;
    for (region_index, region) in regions.iter().enumerate() {
        let mut metadata = metadata.clone();
        if multiple_regions {
            metadata.insert("chartRegionIndex".into(), json!(region_index));
        }
        let mut d = Def::new("path");
        d.set("x", json!(plot_area.x))
            .set("y", json!(plot_area.y))
            .set("points", path_pts_to_value(&region.subpath))
            .stroke_color(if stroke_on_fill {
                    json!(stroke.unwrap())
                } else {
                    Value::Null
                },
            )
            .stroke_width(if stroke_on_fill {
                    stroke_width.map(|w| json!(w)).unwrap_or(Value::Null)
                } else {
                    Value::Null
                },
            )
            .stroke_dash(if stroke_on_fill {
                    dash_value(&dash_vec)
                } else {
                    Value::Null
                },
            )
            .stroke_cap(if stroke_on_fill {
                    stroke_cap.map(|v| json!(v)).unwrap_or(Value::Null)
                } else {
                    Value::Null
                },
            )
            .stroke_join(if stroke_on_fill {
                    stroke_join.map(|v| json!(v)).unwrap_or(Value::Null)
                } else {
                    Value::Null
                },
            );
        if matches!(fill_order, AreaFillOrder::BeforeTail) {
            for (k, v) in &parse_fill(fill_color) {
                d.set(k, v.clone());
            }
        }
        d.set("fillOpacity", json!(fill_opacity))
            .set("closed", json!(true))
            .set("opacity", opacity.map(|v| json!(v)).unwrap_or(Value::Null))
            .set(
                "name",
                json!(if multiple_regions {
                    format!("{base_name} {}", region_index + 1)
                } else {
                    base_name.to_string()
                }),
            )
            .set("metadata", Value::Object(metadata));
        if matches!(fill_order, AreaFillOrder::AfterTail) {
            for (k, v) in &parse_fill(fill_color) {
                d.set(k, v.clone());
            }
        }
        for (k, v) in shadow_props {
            d.set(k, v.clone());
        }
        nodes.push(d);
    }

    if let Some(stroke_val) = stroke {
        if local_top_points.len() >= 2 {
            let line_points: Vec<PathPt> = region_groups
                .iter()
                .flat_map(|r| concat_area_top_edges(r))
                .collect();
            if !line_points.is_empty() {
                let mut line_metadata = metadata.clone();
                line_metadata.insert("chartRole".into(), json!("area-series-line"));
                let mut d = Def::new("path");
                d.set("x", json!(plot_area.x))
                    .set("y", json!(plot_area.y))
                    .set("points", path_pts_to_value(&line_points))
                    .stroke_color(json!(stroke_val))
                    .stroke_width(stroke_width.map(|w| json!(w)).unwrap_or(Value::Null),
                    )
                    .stroke_dash(dash_value(&dash_vec))
                    .stroke_cap(stroke_cap.map(|v| json!(v)).unwrap_or(Value::Null),
                    )
                    .stroke_join(stroke_join.map(|v| json!(v)).unwrap_or(Value::Null),
                    )
                    .fill_color(json!("transparent"))
                    .set("closed", json!(false))
                    .set("opacity", opacity.map(|v| json!(v)).unwrap_or(Value::Null))
                    .set("name", json!(format!("{base_name} Line")))
                    .set("metadata", Value::Object(line_metadata));
                nodes.push(d);
            }
        }
    }
}

/// Shared line-series path emission. `local_points` are caller-resolved
/// plot-area-relative pixel coordinates; the `if … >= 2` guard lives here so
/// both callers can drop their own. `metadata` is caller-built in its exact key
/// order; `shadow_props` are applied after the tail block in both callers.
#[allow(clippy::too_many_arguments)]
pub fn append_line_series_path(
    nodes: &mut Vec<Def>,
    local_points: &[(f64, f64)],
    groups: &[Vec<usize>],
    curve: &str,
    plot_area: Bounds,
    stroke: &str,
    stroke_width: f64,
    dash: Option<&[f64]>,
    stroke_cap: Option<&str>,
    stroke_join: Option<&str>,
    opacity: Option<f64>,
    name: String,
    metadata: Map,
    shadow_props: &Map,
) {
    if local_points.len() >= 2 {
        let path_points: Vec<PathPt> = groups
            .iter()
            .flat_map(|g| {
                let pts: Vec<(f64, f64)> = g.iter().map(|&i| local_points[i]).collect();
                generate_curve_points(&pts, curve)
            })
            .collect();
        let mut d = Def::new("path");
        d.set("x", json!(plot_area.x))
            .set("y", json!(plot_area.y))
            .set("points", path_pts_to_value(&path_points))
            .stroke_color(json!(stroke))
            .stroke_width(json!(stroke_width))
            .stroke_dash(dash_value(&dash.map(|d| d.to_vec())))
            .stroke_cap(stroke_cap.map(|v| json!(v)).unwrap_or(Value::Null),
            )
            .stroke_join(stroke_join.map(|v| json!(v)).unwrap_or(Value::Null),
            )
            .fill_color(json!("transparent"))
            .set("closed", json!(false))
            .set("opacity", opacity.map(|v| json!(v)).unwrap_or(Value::Null))
            .set("name", json!(name))
            .set("metadata", Value::Object(metadata));
        for (k, v) in shadow_props {
            d.set(k, v.clone());
        }
        nodes.push(d);
    }
}

/// `generateSeriesPointMarkers` (markers.ts:192-237).
#[allow(clippy::too_many_arguments)]
pub fn generate_series_point_markers(
    points: &[SeriesPoint],
    series_style: Option<SeriesStyleView>,
    config: ConfigView,
    series_key: &str,
    has_series_dimension: bool,
    chart_name: &str,
    fill_fallback: &str,
    point_colors: Option<&JsonMap>,
    category_labels: Option<&JsonMap>,
) -> Vec<Def> {
    let mut nodes = Vec::new();
    let point_fill = resolve_point_fill(series_style, config.s("pointFill"), fill_fallback);
    let point_stroke = resolve_point_stroke(series_style, config.s("pointStroke"), "");
    let point_shape = resolve_point_shape(series_style, config.s("pointShape"));
    let point_size = resolve_point_size(series_style, config.f("pointSize"));
    let point_stroke_width = series_style
        .and_then(|s| s.f_opt("pointStrokeWidth"))
        .unwrap_or(config.f("pointStrokeWidth"));

    for point in points {
        let marker_x = point.x - point_size / 2.0;
        let marker_y = point.y - point_size / 2.0;
        let mut metadata = Map::new();
        metadata.insert("chartRole".into(), json!("point-marker"));
        metadata.insert("chartCategory".into(), json!(point.category));
        metadata.insert("chartValue".into(), json!(point.value));
        metadata.insert("chartName".into(), json!(chart_name));
        if has_series_dimension {
            metadata.insert("chartSeries".into(), json!(series_key));
        }
        if let Some(xt) = point.x_time {
            metadata.insert("chartX".into(), json!(xt));
        }
        let name = format!(
            "{chart_name} Point {}{}",
            point.category,
            if has_series_dimension {
                format!(" {series_key}")
            } else {
                String::new()
            }
        );
        nodes.push(get_point_marker_node(PointMarkerArgs {
            shape: point_shape,
            size: point_size,
            x: marker_x,
            y: marker_y,
            // Keyed by the label a repeated category draws, not its positional key.
            fill: point_color(
                point_colors,
                series_key,
                category_label(category_labels, &point.category),
            )
            .unwrap_or(&point_fill),
            stroke: &point_stroke,
            stroke_width: point_stroke_width,
            name,
            metadata: Value::Object(metadata),
        }));
    }
    nodes
}

// ---------------------------------------------------------------------------
// labelLayout.ts
// ---------------------------------------------------------------------------

pub const DEFAULT_LABEL_MIN_GAP: f64 = 4.0;

#[derive(Clone, Debug)]
pub struct LabelCandidate {
    pub id: String,
    pub side_right: bool,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Clone, Debug)]
pub struct ResolvedLabelPosition {
    pub x: f64,
    pub y: f64,
}

/// `resolveOutsideLabelPositions` (labelLayout.ts:89-170).
pub fn resolve_outside_label_positions(
    candidates: &[LabelCandidate],
    min_gap: f64,
    top_bound: Option<f64>,
    bottom_bound: Option<f64>,
) -> Vec<ResolvedLabelPosition> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let min_gap = js_max(0.0, min_gap);
    let mut adjusted: Vec<Option<ResolvedLabelPosition>> = vec![None; candidates.len()];

    for side_right in [false, true] {
        let mut side_items: Vec<(usize, &LabelCandidate)> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.side_right == side_right)
            .collect();
        if side_items.is_empty() {
            continue;
        }
        // stable sort by ideal y ascending (JS Array.sort is stable)
        side_items.sort_by(|a, b| {
            a.1.y
                .partial_cmp(&b.1.y)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut positions: Vec<f64> = side_items.iter().map(|(_, c)| c.y).collect();

        for i in 1..side_items.len() {
            let prev_bottom = positions[i - 1] + side_items[i - 1].1.height;
            let min_y = prev_bottom + min_gap;
            if positions[i] < min_y {
                positions[i] = min_y;
            }
        }
        if let Some(bottom_bound) = bottom_bound {
            let last = side_items.len() - 1;
            let last_bottom = positions[last] + side_items[last].1.height;
            if last_bottom > bottom_bound {
                positions[last] = bottom_bound - side_items[last].1.height;
                for i in (0..last).rev() {
                    let max_y = positions[i + 1] - min_gap - side_items[i].1.height;
                    if positions[i] > max_y {
                        positions[i] = max_y;
                    }
                }
            }
        }
        if let Some(top_bound) = top_bound {
            if positions[0] < top_bound {
                positions[0] = top_bound;
                for i in 1..side_items.len() {
                    let min_y = positions[i - 1] + side_items[i - 1].1.height + min_gap;
                    if positions[i] < min_y {
                        positions[i] = min_y;
                    }
                }
            }
        }
        for (i, (orig, c)) in side_items.iter().enumerate() {
            adjusted[*orig] = Some(ResolvedLabelPosition {
                x: c.x,
                y: positions[i],
            });
        }
    }
    adjusted.into_iter().map(|p| p.unwrap()).collect()
}

// ---------------------------------------------------------------------------
// regression.ts
// ---------------------------------------------------------------------------

pub struct Regression {
    pub slope: f64,
    pub intercept: f64,
}

/// `linearRegression` (regression.ts:22-80) — mean-centered two-pass.
pub fn linear_regression(points: &[(f64, f64)]) -> Option<Regression> {
    let n = points.len();
    if n < 2 {
        return None;
    }
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for &(x, y) in points {
        sum_x += x;
        sum_y += y;
    }
    let mean_x = sum_x / n as f64;
    let mean_y = sum_y / n as f64;
    let mut var_x = 0.0;
    let mut cov_xy = 0.0;
    for &(x, y) in points {
        let dx = x - mean_x;
        let dy = y - mean_y;
        var_x += dx * dx;
        cov_xy += dx * dy;
    }
    if var_x == 0.0 {
        return Some(Regression {
            slope: 0.0,
            intercept: mean_y,
        });
    }
    let slope = cov_xy / var_x;
    Some(Regression {
        slope,
        intercept: mean_y - slope * mean_x,
    })
}

#[cfg(test)]
mod tests {
    use super::legend_dash;

    /// chartLegendDash.test.ts: a line marker's dash shrinks until a dash, a gap and a dash fit.
    #[test]
    fn legend_dash_fits_a_dash_a_gap_and_a_dash() {
        assert_eq!(legend_dash(Some(&[8.0, 8.0]), 15.0), Some(vec![5.0, 5.0]));
        assert_eq!(legend_dash(Some(&[2.0, 4.0]), 15.0), Some(vec![2.0, 4.0]));
        assert_eq!(legend_dash(Some(&[8.0, 8.0]), 30.0), Some(vec![8.0, 8.0]));
        assert_eq!(legend_dash(Some(&[]), 15.0), None);
        assert_eq!(legend_dash(None, 15.0), None);
    }
}
