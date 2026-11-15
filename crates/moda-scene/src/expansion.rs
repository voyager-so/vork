//! Transient-node expansion — ports of:
//! - `expandTableNode`            (`core/scene/tableExpansion.ts:46-233`)
//! - `expandGeneratedGraphicNode` (`core/scene/generatedGraphicExpansion.ts:37-105`)
//! - chart expansion             (`core/scene/chartExpansion.ts:60-159`) —
//!   ported generators (bar/line/pie/scatter/area/combo) in `crate::charts`
//!   run here; unregistered types and empty/malformed data fall back to the
//!   placeholder rect, exactly what the TS does for an unregistered type
//! - `createRectangleFromPage`    (`core/scene/shapeHelpers.ts:803-842`)
//!
//! Expansion runs AFTER variable resolution (Scene.ts:3152-3221) and emits
//! plain `RawNode`s whose JSON shape mirrors the TS factory output (fields
//! with `undefined` values are simply absent).

use crate::charts::ticks::TickError;
use crate::paint;
use crate::types::{compute_page_bg_id, JsonMap, RawNode, RawPage};
use crate::value::{json, Map, Value};
use std::collections::HashSet;

/// `transientProxyId` (generatedGraphicExpansion.ts:124-126).
pub fn transient_proxy_id(parent_node_id: &str, index: usize) -> String {
    format!("{parent_node_id}_gen_proxy_{index}")
}

/// Give a transient child the host's visible strokes, bottom to top (none: no
/// stroke). Every layer draws (E3.1 P6).
fn copy_strokes(src: &RawNode, dst: &mut JsonMap) {
    let strokes: Vec<paint::Stroke> = paint::node_strokes(src)
        .into_iter()
        .filter(|s| paint::shown(s.visible, s.opacity))
        .collect();
    paint::set_strokes(dst, strokes);
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

/// `lineShouldRender` (tableExpansion.ts:241-265).
fn line_should_render(line_index: usize, axis_len: usize, config: &str) -> bool {
    match config {
        "all" => true,
        "none" => false,
        _ => {
            if line_index == 0 {
                config == "outside"
            } else if line_index == 1 {
                // NOTE quirk: index 1 wins over the `case axisLength` arm in
                // the TS switch even when axisLength == 1 (switch matches the
                // FIRST case, tableExpansion.ts:255-264).
                config == "inside" || config == "after-first"
            } else if line_index == axis_len {
                config == "outside"
            } else {
                config == "inside"
            }
        }
    }
}

/// `injectTextAlignIntoHtml` (utils/tableSelection.ts): the cell's `text-align` on EVERY `<p>` opening tag
/// (`/<p(\s[^>]*)?>/g`), replacing one the paragraph has, else leading its style. A cell with a line break is
/// two paragraphs, and only the first was aligned.
fn inject_text_align_into_html(html: &str, align: &str) -> String {
    let bytes = html.as_bytes();
    let mut out = String::with_capacity(html.len() + 32);
    let mut i = 0;
    while let Some(rel) = html[i..].find("<p") {
        let start = i + rel;
        let after = start + 2;
        let opens_p = matches!(bytes.get(after), Some(b'>')) || bytes.get(after).is_some_and(|c| c.is_ascii_whitespace());
        let end = if opens_p { html[after..].find('>').map(|e| after + e) } else { None };
        let Some(end) = end else {
            out.push_str(&html[i..after]);
            i = after;
            continue;
        };
        out.push_str(&html[i..start]);
        out.push_str(&align_p_tag(&html[start..=end], align));
        i = end + 1;
    }
    out.push_str(&html[i..]);
    out
}

/// One `<p …>` opening tag with `text-align: <align>` leading its style (an existing `text-align` removed).
fn align_p_tag(tag: &str, align: &str) -> String {
    // `/(\s)style="([^"]*)"/`: the first whitespace-preceded `style="`.
    let bytes = tag.as_bytes();
    let mut from = 0;
    while let Some(rel) = tag[from..].find("style=\"") {
        let at = from + rel;
        if at > 0 && bytes[at - 1].is_ascii_whitespace() {
            let value_start = at + "style=\"".len();
            if let Some(value_len) = tag[value_start..].find('"') {
                let cleaned = remove_text_align(&tag[value_start..value_start + value_len]);
                let mut new_style = format!("text-align: {align}; {}", cleaned.trim());
                // .replace(/;\s*$/, '').trim()
                new_style = new_style.trim_end().to_string();
                if new_style.ends_with(';') {
                    new_style.pop();
                }
                let new_style = new_style.trim();
                return format!("{}style=\"{new_style}\"{}", &tag[..at], &tag[value_start + value_len + 1..]);
            }
        }
        from = at + 1;
    }
    format!("<p style=\"text-align: {align}\"{}", &tag[2..])
}

/// Remove `/text-align:\s*[^;]+;?\s*/g` occurrences.
fn remove_text_align(style: &str) -> String {
    let mut out = String::with_capacity(style.len());
    let mut rest = style;
    while let Some(i) = rest.find("text-align:") {
        out.push_str(&rest[..i]);
        let mut j = i + "text-align:".len();
        let bytes = rest.as_bytes();
        // \s*
        while j < bytes.len() && (bytes[j] as char).is_whitespace() {
            j += 1;
        }
        // [^;]+
        while j < bytes.len() && bytes[j] != b';' {
            j += 1;
        }
        // ;?
        if j < bytes.len() && bytes[j] == b';' {
            j += 1;
        }
        // \s*
        while j < bytes.len() && (bytes[j] as char).is_whitespace() {
            j += 1;
        }
        rest = &rest[j..];
    }
    out.push_str(rest);
    out
}

/// `tableCellToRichTextCell` (utils/tableSelection.ts:112-129) — content node
/// BEFORE the expansion sets x/y/width/height/opacity. Text boxes have no
/// padding: the cell's `textPadding` insets the content box instead (see
/// `expand_table_node`).
///
/// `cell_id` is the id `resolveTableCellIds` RESOLVED for this cell
/// (utils/tableSelection.ts:46-73), not `cell.id`: an id-less or repeated cell
/// doesn't own its own id, and the TS threads the resolved one through the same
/// parameter (tableExpansion.ts:153).
fn table_cell_to_richtext(cell: &JsonMap, cell_id: &str) -> JsonMap {
    let mut html = cell
        .get("textContent")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let h_align = cell.get("textHorizontalAlign").and_then(Value::as_str);
    if let Some(a) = h_align {
        if a != "left" && !html.is_empty() {
            html = inject_text_align_into_html(&html, a);
        }
    }
    let mut m = Map::new();
    m.insert("id".into(), json!(format!("{cell_id}_content")));
    m.insert("type".into(), json!("richtext"));
    m.insert("htmlContent".into(), json!(html));
    m.insert("rotation".into(), json!(0.0));
    m.insert("scale".into(), json!(1.0));
    if let Some(v) = cell.get("textVerticalAlign") {
        if !v.is_null() {
            m.insert("verticalAlign".into(), v.clone());
        }
    }
    m
}

/// Text height seam installed by the engine, shared by native and wasm.
pub type TableTextMeasurer = fn(&RawNode) -> Option<f64>;
static TABLE_TEXT_MEASURER: std::sync::OnceLock<TableTextMeasurer> = std::sync::OnceLock::new();
pub fn install_table_text_measurer(measure: TableTextMeasurer) {
    let _ = TABLE_TEXT_MEASURER.set(measure);
}

/// Resolve auto rows without changing the authored record. Absent policy is auto;
/// explicit fixed rows retain their heights and their cell clips.
pub fn fit_table_rows(table: &RawNode) -> Option<RawNode> {
    fit_table_rows_with(table, *TABLE_TEXT_MEASURER.get()?)
}

pub fn fit_table_rows_with(table: &RawNode, measure: TableTextMeasurer) -> Option<RawNode> {
    if table.str("rowSizing") == Some("fixed") { return None; }
    let mut rows = table.array("rows")?.clone();
    let columns = table.array("columns")?;
    let cells = table.array("cells")?;
    let spans = crate::table_spans::resolve_table_spans(rows.len(), columns.len(), Some(cells));
    let stroke = paint::node_strokes(table).iter()
        .filter(|s| paint::shown(s.visible, s.opacity)).map(|s| s.width).fold(0.0, f64::max);
    // `autoRowMinHeight` (shapeDefaults.ts): the floor of a row with no text; a row with text is as
    // tall as its text.
    let mut required = vec![0.0_f64; rows.len()];
    let mut measured = vec![false; rows.len()];
    let mut holds_text = vec![false; rows.len()];
    let mut merged = Vec::new();
    for (r, row) in cells.iter().enumerate().take(rows.len()) {
        for (c, cell) in row.as_array()?.iter().enumerate().take(columns.len()) {
            if spans.covered.contains(&(r, c)) { continue; }
            let cell = cell.as_object()?;
            let region = spans.regions.get(&(r, c));
            let cs = region.map_or(1, |s| s.col_span);
            let rs = region.map_or(1, |s| s.row_span);
            let width: f64 = columns[c..c+cs].iter().map(|v| v.get("size").and_then(Value::as_f64).unwrap_or(0.0)).sum();
            let pad = cell.get("textPadding").and_then(Value::as_f64).filter(|p| p.is_finite() && *p > 0.0).unwrap_or(0.0);
            let mut content = table_cell_to_richtext(cell, "__table_measure");
            content.insert("width".into(), json!((width - stroke - 2.0 * pad).max(0.0)));
            // Unbounded, top-aligned measurement: a small/middle-aligned box must not hide lines.
            content.remove("height");
            content.insert("verticalAlign".into(), json!("top"));
            let height = measure(&RawNode::new(content))?;
            if !height.is_finite() { return None; }
            let need = (height + 2.0 * pad).ceil() + stroke + 3.0;
            if rs == 1 { required[r] = required[r].max(need); measured[r] = true; holds_text[r] |= height > 0.0; }
            else { merged.push((r, rs, need)); }
        }
    }
    for (r, row) in rows.iter_mut().enumerate() {
        if measured[r] {
            let size = if holds_text[r] { required[r] } else { required[r].max(32.0) };
            row.as_object_mut()?.insert("size".into(), json!(size));
        }
    }
    for (r, rs, need) in merged {
        let current: f64 = rows[r..r+rs].iter().map(|v| v.get("size").and_then(Value::as_f64).unwrap_or(0.0)).sum();
        if need > current {
            let last = &mut rows[r+rs-1];
            let size = last.get("size").and_then(Value::as_f64).unwrap_or(0.0);
            last.as_object_mut()?.insert("size".into(), json!(size + need - current));
        }
    }
    let height: f64 = rows.iter().map(|v| v.get("size").and_then(Value::as_f64).unwrap_or(0.0)).sum();
    let mut out = table.clone();
    out.set("rows", json!(rows));
    out.set("height", json!(height));
    Some(out)
}

/// `isOpaqueHexColor` (tableExpansion.ts): a literal opaque hex (`#rgb`,
/// `#rrggbb`, or with a full alpha). Deliberately narrow so both ports read every
/// value the same way.
pub fn is_opaque_hex_color(color: Option<&Value>) -> bool {
    let Some(hex) = color
        .and_then(Value::as_str)
        .and_then(|s| s.trim().strip_prefix('#'))
    else {
        return false;
    };
    if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return false;
    }
    match hex.len() {
        3 | 6 => true,
        4 => hex.ends_with(['f', 'F']),
        8 => hex[6..].eq_ignore_ascii_case("ff"),
        _ => false,
    }
}

/// `expandTableNode` (tableExpansion.ts:46-233). Coordinates are
/// table-PARENT-relative (cell positions include table x/y). Emission order:
/// `[...backgrounds, ...contents, ...borders]` (tableExpansion.ts:232).
///
/// The TS side carries an explicit guard-and-report for a missing/non-array
/// `cells` (WEB-NEXTJS-22R, tableExpansion.ts:62-71). This lane needs none: every
/// accessor below is already defensive (`array(...).unwrap_or_default()`,
/// `and_then(Value::as_array)`), so a malformed table degrades to borders-only
/// here for free. That was incidental rather than deliberate — keep it in mind
/// before "simplifying" any of those accessors into an unwrap.
pub fn expand_table_node(table: &RawNode) -> Vec<RawNode> {
    let table_x = table.f64("x").unwrap_or(0.0);
    let table_y = table.f64("y").unwrap_or(0.0);
    let rows: Vec<f64> = table
        .array("rows")
        .map(|a| {
            a.iter()
                .map(|r| r.get("size").and_then(Value::as_f64).unwrap_or(0.0))
                .collect()
        })
        .unwrap_or_default();
    // Explicit per-row background (`rows[i].backgroundColor`, tableExpansion.ts)
    // — wins over headerBackgroundColor and the zebra cycle for exactly that row.
    let row_fills: Vec<Option<Value>> = table
        .array("rows")
        .map(|a| {
            a.iter()
                .map(|r| r.get("backgroundColor").cloned().filter(|v| !v.is_null()))
                .collect()
        })
        .unwrap_or_default();
    let columns: Vec<f64> = table
        .array("columns")
        .map(|a| {
            a.iter()
                .map(|c| c.get("size").and_then(Value::as_f64).unwrap_or(0.0))
                .collect()
        })
        .unwrap_or_default();
    // The grid lines: every visible table stroke, bottom to top (none: no grid
    // lines). The line strokes carry each table stroke's paint, width, opacity
    // and blend mode (not its caps, joins or dashes). The layout reserves the
    // widest.
    let border: Vec<paint::Stroke> = paint::node_strokes(table)
        .into_iter()
        .filter(|s| paint::shown(s.visible, s.opacity))
        .map(|s| {
            let mut line = paint::Stroke::solid(&s.id, Value::Null, s.width);
            line.paint = s.paint;
            line.opacity = s.opacity;
            line.blend_mode = s.blend_mode;
            line
        })
        .collect();

    let stroke_width = border.iter().map(|s| s.width).fold(0.0, f64::max);
    let table_opacity = table.opacity();
    let opacity = |local: f64| table_opacity * local;

    let mut row_positions = vec![0.0_f64];
    for (i, r) in rows.iter().enumerate() {
        row_positions.push(row_positions[i] + r);
    }
    let mut column_positions = vec![0.0_f64];
    for (i, c) in columns.iter().enumerate() {
        column_positions.push(column_positions[i] + c);
    }
    let table_width = *column_positions.last().unwrap_or(&0.0);
    let table_height = *row_positions.last().unwrap_or(&0.0);

    let table_radii = table.corner_radii();

    let col_line_start = table_y - stroke_width / 2.0;
    let col_line_end = table_y + table_height + stroke_width / 2.0;
    let row_line_start = table_x - stroke_width / 2.0;
    let row_line_end = table_x + table_width + stroke_width / 2.0;

    let row_cfg = table.str("rowStrokeConfig").unwrap_or("all").to_string();
    let col_cfg = table.str("columnStrokeConfig").unwrap_or("all").to_string();

    let cells = table.array("cells").cloned().unwrap_or_default();
    let background_colors = table.array("backgroundColors").cloned();
    let header_bg = table
        .json_field("headerBackgroundColor")
        .cloned()
        .filter(|v| !v.is_null());

    let mut backgrounds: Vec<RawNode> = Vec::new();
    let mut contents: Vec<RawNode> = Vec::new();
    let mut borders: Vec<RawNode> = Vec::new();

    // Cell ids already claimed. The TS hoists this into `resolveTableCellIds`
    // (utils/tableSelection.ts:46-73) because the browser has four more consumers of the resolved
    // id (hit-test, toolbar proxy, edit commit); this lane has only expansion, so it stays inline —
    // but the RULE must match byte for byte. An id-less cell used to mint
    // `"_content"` / `"_background"` here — and a REPEATED id collides just the same — so both
    // overwrite each other in Scene's transient registries (`scene.rs:281-292`, last-cell-wins)
    // and only one of the cells renders. Same positional fallback as the TS, byte for byte.
    let mut seen_cell_ids: HashSet<String> = HashSet::new();

    let make_line = |id: String, points: [f64; 4], op: f64| -> RawNode {
        let mut m = Map::new();
        m.insert("id".into(), json!(id));
        m.insert("type".into(), json!("line"));
        m.insert("points".into(), json!(points));
        paint::set_strokes(&mut m, border.clone());
        m.insert("opacity".into(), json!(op));
        RawNode::new(m)
    };

    // Merged cells, byte-exact with `tableSpans.ts`. `is_plain_grid`
    // is the fast path every table without a merge takes, and on it every branch
    // below collapses to what it did before spans existed — same nodes, same ids,
    // same geometry, so `mutation_replay` cannot move on an existing document.
    let spans =
        crate::table_spans::resolve_table_spans(rows.len(), columns.len(), Some(&cells[..]));

    // The fill each slot's background paints. PRECEDENCE, most specific first:
    // the cell's own background, then an explicit per-row fill, then the header
    // colour, then the zebra cycle (tableExpansion.ts `slotFill`). Byte-exact
    // with the TS — a disagreement here is a table that looks one way in the
    // editor and another in every export, which is the class of bug this mirror
    // exists to prevent.
    //
    // `filter(!is_null)` matches the TS `??`: JSON null and an absent key both
    // mean "inherit", and only a real colour wins. The zebra is last, so a null
    // entry has nowhere to fall to; the write site drops it.
    let slot_fill = |row_idx: usize, col_idx: usize| -> Option<Value> {
        let cell = cells
            .get(row_idx)
            .and_then(Value::as_array)
            .and_then(|r| r.get(col_idx))
            .and_then(Value::as_object)?;
        if let Some(cell_bg) = cell
            .get("backgroundColor")
            .cloned()
            .filter(|v| !v.is_null())
        {
            Some(cell_bg)
        } else if let Some(row_fill) = row_fills.get(row_idx).and_then(|f| f.clone()) {
            Some(row_fill)
        } else if row_idx == 0 && header_bg.is_some() {
            header_bg.clone()
        } else {
            background_colors.as_ref().and_then(|colors| {
                if colors.is_empty() {
                    // `rowIdx % 0` is NaN in JS → undefined lookup
                    None
                } else {
                    colors.get(row_idx % colors.len()).cloned()
                }
            })
        }
    };
    // `backgroundReach` (tableExpansion.ts): an opaque background reaches under
    // the opaque neighbours drawn after it (right, below), so the shared edge is
    // anti-aliased once and the backdrop can't show through as a seam. Literal
    // opaque hex colours on a plain grid at full opacity, short of a rounded
    // table corner.
    let overlap_seams = spans.is_plain_grid() && table_opacity == 1.0;
    let opaque_at = |row_idx: usize, col_idx: usize| -> bool {
        row_idx < rows.len()
            && col_idx < columns.len()
            && is_opaque_hex_color(slot_fill(row_idx, col_idx).as_ref())
    };
    let last_row = rows.len().saturating_sub(1);
    let last_col = columns.len().saturating_sub(1);
    let background_reach = |row_idx: usize, col_idx: usize| -> (f64, f64) {
        if !overlap_seams || !opaque_at(row_idx, col_idx) {
            return (0.0, 0.0);
        }
        let right = if opaque_at(row_idx, col_idx + 1) {
            let corner = if col_idx + 1 == last_col {
                table_radii[1].max(table_radii[2])
            } else {
                0.0
            };
            (columns[col_idx + 1] - corner).max(0.0)
        } else {
            0.0
        };
        let down = if opaque_at(row_idx + 1, col_idx)
            && (right == 0.0 || opaque_at(row_idx + 1, col_idx + 1))
        {
            let corner = if row_idx + 1 == last_row {
                table_radii[2].max(table_radii[3])
            } else {
                0.0
            };
            (rows[row_idx + 1] - corner).max(0.0)
        } else {
            0.0
        };
        (right, down)
    };

    for (row_idx, _) in rows.iter().enumerate() {
        // row tops (tableExpansion.ts:114-125)
        if line_should_render(row_idx, rows.len(), &row_cfg) {
            // A rowSpan swallows the rule through it, so it is drawn in the runs
            // of columns the merge does NOT cover. With none, that is one whole
            // run and the original single node.
            for (from, to, whole) in crate::table_spans::line_runs(columns.len(), |c| {
                crate::table_spans::span_crosses_line(
                    &spans,
                    row_idx,
                    crate::table_spans::LineAxis::Row,
                    c,
                )
            }) {
                let id = if whole {
                    format!("{}_border_top_{}", table.id(), row_idx)
                } else {
                    format!("{}_border_top_{}_{}", table.id(), row_idx, from)
                };
                borders.push(make_line(
                    id,
                    [
                        if whole {
                            row_line_start
                        } else {
                            table_x + column_positions[from]
                        },
                        table_y + row_positions[row_idx],
                        if whole {
                            row_line_end
                        } else {
                            table_x + column_positions[to]
                        },
                        table_y + row_positions[row_idx],
                    ],
                    opacity(1.0),
                ));
            }
        }
        for (col_idx, _) in columns.iter().enumerate() {
            // column lefts only on first row pass (tableExpansion.ts:128-142)
            if row_idx == 0 && line_should_render(col_idx, columns.len(), &col_cfg) {
                for (from, to, whole) in crate::table_spans::line_runs(rows.len(), |r| {
                    crate::table_spans::span_crosses_line(
                        &spans,
                        col_idx,
                        crate::table_spans::LineAxis::Column,
                        r,
                    )
                }) {
                    let id = if whole {
                        format!("{}_border_left_{}_{}", table.id(), row_idx, col_idx)
                    } else {
                        format!(
                            "{}_border_left_{}_{}_{}",
                            table.id(),
                            row_idx,
                            col_idx,
                            from
                        )
                    };
                    borders.push(make_line(
                        id,
                        [
                            table_x + column_positions[col_idx],
                            if whole {
                                col_line_start
                            } else {
                                table_y + row_positions[from]
                            },
                            table_x + column_positions[col_idx],
                            if whole {
                                col_line_end
                            } else {
                                table_y + row_positions[to]
                            },
                        ],
                        opacity(1.0),
                    ));
                }
            }
            // A slot another cell's merge covers draws NOTHING — the anchor
            // already paints over it.
            if spans.covered.contains(&(row_idx, col_idx)) {
                continue;
            }
            let region = spans.regions.get(&(row_idx, col_idx));
            let cell_x = table_x + column_positions[col_idx];
            let cell_y = table_y + row_positions[row_idx];
            // A merged cell is the UNION of its slots, so its size comes from the
            // cumulative positions. With no region these are `columns[col_idx]` /
            // `rows[row_idx]` exactly as before.
            let cell_width = match region {
                Some(r) => column_positions[col_idx + r.col_span] - column_positions[col_idx],
                None => columns[col_idx],
            };
            let cell_height = match region {
                Some(r) => row_positions[row_idx + r.row_span] - row_positions[row_idx],
                None => rows[row_idx],
            };

            let cell = cells
                .get(row_idx)
                .and_then(Value::as_array)
                .and_then(|r| r.get(col_idx))
                .and_then(Value::as_object);
            let Some(cell) = cell else { continue };

            // `cell.id || `${table.id}_cell_${rowIdx}_${colIdx}`` with the
            // duplicate-id dedup (utils/tableSelection.ts:60-61). `filter` on
            // non-empty mirrors the JS falsy test: absent, null and `''` all
            // take the fallback. Walked over `rows` x `columns` — NOT the `cells`
            // array — so a ragged `cells` can't shift the dedup ordinals.
            let cell_id = match cell
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
            {
                Some(id) if !seen_cell_ids.contains(id) => id.to_string(),
                _ => format!("{}_cell_{}_{}", table.id(), row_idx, col_idx),
            };
            seen_cell_ids.insert(cell_id.clone());

            // content (tableExpansion.ts:153-167)
            let mut content = table_cell_to_richtext(cell, &cell_id);
            // `textPadding` insets the content box on every side.
            let pad = cell
                .get("textPadding")
                .and_then(Value::as_f64)
                .filter(|p| p.is_finite() && *p > 0.0)
                .unwrap_or(0.0);
            content.insert("x".into(), json!(cell_x + 0.5 * stroke_width + pad));
            content.insert("y".into(), json!(cell_y + 0.5 * stroke_width + pad));
            content.insert(
                "width".into(),
                json!((cell_width - stroke_width - 2.0 * pad).max(0.0)),
            );
            content.insert(
                "height".into(),
                json!((cell_height - stroke_width - 2.0 * pad).max(0.0)),
            );
            content.insert("opacity".into(), json!(opacity(1.0)));
            // Cell text is clipped to its own cell box so it can't bleed over the
            // neighbours (tableExpansion.ts:162-165). Unconditional on this
            // lane exactly as in the TS — every writer gets the clip, including
            // imports and agent patches that never measure the text.
            content.insert("clipTextToBounds".into(), json!(true));
            contents.push(RawNode::new(content));

            // background (tableExpansion.ts:169-197)
            let mut cell_radii = [0.0_f64; 4];
            if row_idx == 0 && col_idx == 0 {
                cell_radii[0] = table_radii[0];
            } else if row_idx == 0 && col_idx == columns.len() - 1 {
                cell_radii[1] = table_radii[1];
            } else if row_idx == rows.len() - 1 && col_idx == columns.len() - 1 {
                cell_radii[2] = table_radii[2];
            } else if row_idx == rows.len() - 1 && col_idx == 0 {
                cell_radii[3] = table_radii[3];
            }
            let fill = slot_fill(row_idx, col_idx);
            let (reach_right, reach_down) = background_reach(row_idx, col_idx);
            let mut bg = Map::new();
            bg.insert("id".into(), json!(format!("{cell_id}_background")));
            bg.insert("type".into(), json!("rectangle"));
            bg.insert("x".into(), json!(cell_x));
            bg.insert("y".into(), json!(cell_y));
            bg.insert("width".into(), json!(cell_width + reach_right));
            bg.insert("height".into(), json!(cell_height + reach_down));
            bg.insert("rotation".into(), json!(0.0));
            bg.insert("scale".into(), json!(1.0));
            if let Some(f) = fill {
                if !f.is_null() {
                    paint::set_solid_fill(&mut bg, f);
                }
            }
            bg.insert("opacity".into(), json!(opacity(1.0)));
            bg.insert("locked".into(), json!(true));
            bg.insert("cornerRadius".into(), json!(cell_radii.to_vec()));
            backgrounds.push(RawNode::new(bg));
        }
    }

    // bottom border (tableExpansion.ts:202-213)
    if line_should_render(rows.len(), rows.len(), &row_cfg) {
        borders.push(make_line(
            format!("{}_border_bottom_{}", table.id(), rows.len()),
            [
                row_line_start,
                table_y + row_positions[rows.len()],
                row_line_end,
                table_y + row_positions[rows.len()],
            ],
            opacity(1.0),
        ));
    }
    // right border (tableExpansion.ts:215-231)
    if line_should_render(columns.len(), columns.len(), &col_cfg) {
        borders.push(make_line(
            format!("{}_border_right_{}", table.id(), columns.len()),
            [
                table_x + column_positions[columns.len()],
                col_line_start,
                table_x + column_positions[columns.len()],
                col_line_end,
            ],
            opacity(1.0),
        ));
    }

    let mut out = backgrounds;
    out.extend(contents);
    out.extend(borders);
    out
}

// ---------------------------------------------------------------------------
// Generated graphics
// ---------------------------------------------------------------------------

/// `expandGeneratedGraphicNode` (generatedGraphicExpansion.ts:37-49).
pub fn expand_generated_graphic_node(node: &RawNode) -> Result<Vec<RawNode>, TickError> {
    Ok(match node.str("generatedKind") {
        // A QR is always a local vector; one without a stored vector draws the placeholder.
        Some("qr") if is_vector_qr(node) => expand_qr_vector(node),
        Some("qr") => vec![create_placeholder_rect(node)],
        Some("latex") => expand_single_asset(node),
        // Ported generators (bar/line/pie/scatter/area/combo) run here;
        // unregistered types and empty/malformed data fall back to the
        // placeholder inside the module (chartExpansion.ts semantics).
        Some("chart") => crate::charts::expand_chart_generated_node(node)?,
        _ => vec![create_placeholder_rect(node)],
    })
}

/// `isVectorQRNode` (generatedGraphicExpansion.ts): a QR generated locally
/// (utils/generatedGraphics/qrCode.ts) carries its outline path and expands into
/// native children; a QR without one draws the placeholder.
pub fn is_vector_qr(node: &RawNode) -> bool {
    let dims = node.object("cachedNaturalDimensions");
    let dim = |k: &str| {
        dims.and_then(|d| d.get(k))
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
    };
    node.str("generatedKind") == Some("qr")
        && node.str("renderMode") == Some("native-children")
        && node.str("cachedVectorPath").is_some()
        && dim("width") > 0.0
        && dim("height") > 0.0
}

/// `parseGeneratedVectorPath` (generatedGraphicExpansion.ts): absolute
/// `M x y` / `H x` / `V y` / `Z` path data → move/line points scaled by
/// (sx, sy); each `Z` adds an explicit line back to its outline's start.
pub fn parse_vector_path(d: &str, sx: f64, sy: f64) -> Vec<Value> {
    let mut points = Vec::new();
    let (mut x, mut y, mut start_x, mut start_y) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut rest = d;
    while let Some(cmd) = rest.chars().next() {
        // Char-boundary safe: a corrupted document must not panic the engine.
        let start = cmd.len_utf8();
        let body_end = rest[start..]
            .find(|c: char| matches!(c, 'M' | 'H' | 'V' | 'Z'))
            .map_or(rest.len(), |i| i + start);
        let nums: Vec<f64> = rest[start..body_end]
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();
        rest = &rest[body_end..];
        let point = |ty: &str, px: f64, py: f64| json!({"type": ty, "x": px * sx, "y": py * sy});
        match cmd {
            'M' => {
                x = nums.first().copied().unwrap_or(0.0);
                y = nums.get(1).copied().unwrap_or(0.0);
                start_x = x;
                start_y = y;
                points.push(point("move", x, y));
            }
            'H' => {
                x = nums.first().copied().unwrap_or(x);
                points.push(point("line", x, y));
            }
            'V' => {
                y = nums.first().copied().unwrap_or(y);
                points.push(point("line", x, y));
            }
            'Z' => {
                if x != start_x || y != start_y {
                    points.push(point("line", start_x, start_y));
                }
                x = start_x;
                y = start_y;
            }
            _ => {}
        }
    }
    points
}

/// `expandQRVectorNode` (generatedGraphicExpansion.ts): a background rect over
/// the box plus one module path, both in the host's LOCAL space (the
/// `native-children` host is their transform parent), scaled from module units
/// to width/height; host opacity composed into each child like chart children.
fn expand_qr_vector(node: &RawNode) -> Vec<RawNode> {
    let source = node.object("generatedSource");
    let color = |key: &str, default: &str| {
        source
            .and_then(|s| s.get(key))
            .and_then(Value::as_str)
            .unwrap_or(default)
            .to_string()
    };
    let dims = node.object("cachedNaturalDimensions");
    let dim = |k: &str| {
        dims.and_then(|d| d.get(k))
            .and_then(Value::as_f64)
            .unwrap_or(1.0)
    };
    let w = node.f64("width").unwrap_or(0.0).max(1.0);
    let h = node.f64("height").unwrap_or(0.0).max(1.0);
    let opacity = node.opacity();

    let mut bg = Map::new();
    bg.insert("id".into(), json!(transient_proxy_id(node.id(), 0)));
    bg.insert("type".into(), json!("rectangle"));
    bg.insert("x".into(), json!(0.0));
    bg.insert("y".into(), json!(0.0));
    bg.insert("width".into(), json!(w));
    bg.insert("height".into(), json!(h));
    bg.insert("rotation".into(), json!(0.0));
    bg.insert("scale".into(), json!(1.0));
    paint::set_solid_fill(
        &mut bg,
        json!(color(
            "backgroundColor",
            crate::defaults::generated::QR_BACKGROUND_COLOR
        )),
    );
    copy_strokes(node, &mut bg);
    bg.insert("opacity".into(), json!(opacity));

    let points = parse_vector_path(
        node.str("cachedVectorPath").unwrap_or(""),
        w / dim("width"),
        h / dim("height"),
    );
    if points.len() < 2 {
        return vec![RawNode::new(bg)];
    }
    let mut path = Map::new();
    path.insert("id".into(), json!(transient_proxy_id(node.id(), 1)));
    path.insert("type".into(), json!("path"));
    path.insert("x".into(), json!(0.0));
    path.insert("y".into(), json!(0.0));
    path.insert("points".into(), Value::Array(points));
    paint::set_solid_fill(
        &mut path,
        json!(color(
            "foregroundColor",
            crate::defaults::generated::QR_FOREGROUND_COLOR
        )),
    );
    path.insert("fillRule".into(), json!("nonzero"));
    path.insert("closed".into(), json!(true));
    path.insert("opacity".into(), json!(opacity));
    vec![RawNode::new(bg), RawNode::new(path)]
}

/// `getSingleAssetPatternProps` (generatedGraphicExpansion.ts:7-24).
fn single_asset_pattern_props(node: &RawNode) -> (f64, f64, f64, f64) {
    let nw = node
        .object("cachedNaturalDimensions")
        .and_then(|d| d.get("width"))
        .and_then(Value::as_f64)
        .unwrap_or_else(|| node.f64("width").unwrap_or(0.0).max(1.0));
    let nh = node
        .object("cachedNaturalDimensions")
        .and_then(|d| d.get("height"))
        .and_then(Value::as_f64)
        .unwrap_or_else(|| node.f64("height").unwrap_or(0.0).max(1.0));
    let w = node.f64("width").unwrap_or(0.0).max(1.0);
    let h = node.f64("height").unwrap_or(0.0).max(1.0);
    (nw, nh, w / nw.max(1.0), h / nh.max(1.0))
}

/// generatedGraphicExpansion.ts:51-84.
fn expand_single_asset(node: &RawNode) -> Vec<RawNode> {
    let cached = node.str("cachedAssetUrl").filter(|s| !s.is_empty());
    let Some(url) = cached else {
        return vec![create_placeholder_rect(node)];
    };
    let url = url.to_string();
    let (nw, nh, sx, sy) = single_asset_pattern_props(node);
    let mut m = Map::new();
    m.insert("id".into(), json!(transient_proxy_id(node.id(), 0)));
    m.insert("type".into(), json!("rectangle"));
    m.insert("x".into(), json!(node.f64("x").unwrap_or(0.0)));
    m.insert("y".into(), json!(node.f64("y").unwrap_or(0.0)));
    m.insert(
        "width".into(),
        json!(node.f64("width").unwrap_or(0.0).max(1.0)),
    );
    m.insert(
        "height".into(),
        json!(node.f64("height").unwrap_or(0.0).max(1.0)),
    );
    m.insert("rotation".into(), json!(node.rotation()));
    if node.skew_x() != 0.0 {
        m.insert("skewX".into(), json!(node.skew_x()));
    }
    m.insert("scale".into(), json!(node.f64("scale").unwrap_or(1.0)));
    // fill: undefined (omitted)
    paint::set_image_fill(&mut m, &url, (sx, sy), (nw, nh));
    let mut n = RawNode::new(m);
    copy_strokes(node, n.fields_mut());
    n.set("opacity", json!(node.opacity()));
    n.set("_transformSourceId", json!(node.id()));
    vec![n]
}

/// `createPlaceholderRect` (generatedGraphicExpansion.ts:86-105) +
/// `getGeneratedPlaceholderFill` (:107-117).
fn create_placeholder_rect(node: &RawNode) -> RawNode {
    let bg = if node.str("generatedKind") == Some("qr") {
        node.object("generatedSource")
            .and_then(|s| s.get("backgroundColor"))
            .and_then(Value::as_str)
            .map(str::to_string)
    } else {
        None
    };
    let mut m = Map::new();
    m.insert("id".into(), json!(transient_proxy_id(node.id(), 0)));
    m.insert("type".into(), json!("rectangle"));
    m.insert("x".into(), json!(node.f64("x").unwrap_or(0.0)));
    m.insert("y".into(), json!(node.f64("y").unwrap_or(0.0)));
    m.insert(
        "width".into(),
        json!(node.f64("width").unwrap_or(0.0).max(1.0)),
    );
    m.insert(
        "height".into(),
        json!(node.f64("height").unwrap_or(0.0).max(1.0)),
    );
    m.insert("rotation".into(), json!(node.rotation()));
    if node.skew_x() != 0.0 {
        m.insert("skewX".into(), json!(node.skew_x()));
    }
    m.insert("scale".into(), json!(node.f64("scale").unwrap_or(1.0)));
    paint::set_solid_fill(
        &mut m,
        json!(bg.unwrap_or_else(|| crate::defaults::generated::PLACEHOLDER_FILL.to_string())),
    );
    // The placeholder's outline: the node's own strokes, else the
    // placeholder look (#cccccc, 1 px).
    copy_strokes(node, &mut m);
    if !m.contains_key("strokes") {
        paint::set_strokes(
            &mut m,
            vec![paint::Stroke::solid(
                "stroke-0",
                json!(crate::defaults::generated::PLACEHOLDER_STROKE),
                1.0,
            )],
        );
    }
    m.insert("opacity".into(), json!(node.opacity()));
    m.insert("_transformSourceId".into(), json!(node.id()));
    RawNode::new(m)
}

// (M4c: the chart placeholder + ported generator dispatch live in
// `crate::charts` — see `charts::expand_chart_generated_node` /
// `charts::create_chart_placeholder`.)

// ---------------------------------------------------------------------------
// Page backgrounds
// ---------------------------------------------------------------------------

/// `createRectangleFromPage` (shapeHelpers.ts): the page's size and `fills`
/// (literal colours; the editor resolves bound ones before feeding the page).
pub fn create_rectangle_from_page(page: &RawPage) -> RawNode {
    let mut m = Map::new();
    m.insert("id".into(), json!(compute_page_bg_id(page.id())));
    m.insert("type".into(), json!("rectangle"));
    m.insert("x".into(), json!(0.0));
    m.insert("y".into(), json!(0.0));
    m.insert("width".into(), json!(page.width()));
    m.insert("height".into(), json!(page.height()));
    m.insert("rotation".into(), json!(0.0));
    m.insert("scale".into(), json!(1.0));

    m.extend(paint::page_background_fields(page));
    m.insert("locked".into(), json!(true));
    m.insert("opacity".into(), json!(1.0));
    RawNode::new(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    /// The top fill's solid colour.
    fn fill_of(n: &RawNode) -> Option<String> {
        paint::top_fill(n).and_then(|f| paint::solid_color(&f.paint).map(str::to_string))
    }

    /// The top stroke's solid colour.
    fn stroke_of(n: &RawNode) -> Option<String> {
        paint::top_stroke(n).and_then(|s| paint::solid_color(&s.paint).map(str::to_string))
    }

    fn page(v: Value) -> RawPage {
        RawPage(v.as_object().unwrap().clone())
    }

    fn table() -> RawNode {
        RawNode::from_value(json!({
            "id": "tbl", "type": "table",
            "x": 100.0, "y": 200.0, "width": 60.0, "height": 40.0,
            "rotation": 0, "scale": 1,
            "rows": [{"size": 20.0}, {"size": 20.0}],
            "columns": [{"size": 30.0}, {"size": 30.0}],
            "cells": [
                [{"id": "c00", "textContent": "<p>a</p>"}, {"id": "c01", "textContent": "<p>b</p>", "textHorizontalAlign": "center"}],
                [{"id": "c10", "textContent": "<p>c</p>"}, {"id": "c11", "textContent": "<p>d</p>"}]
            ],
            "backgroundColors": ["#eeeeee", "#dddddd"],
            "headerBackgroundColor": "#222222",
            "columnStrokeConfig": "all",
            "rowStrokeConfig": "all",
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": 2.0}],
            "cornerRadius": 8.0,
            "opacity": 0.5
        }))
        .unwrap()
    }

    /// A 3x3 opaque table at full opacity: (columns 30, 30.4, 29.6; rows 20, 24, 16).
    fn seam_table(radius: f64, zebra: Value) -> RawNode {
        let cell = |id: &str| json!({"id": id, "textContent": "<p>x</p>"});
        RawNode::from_value(json!({
            "id": "tbl", "type": "table", "x": 0.0, "y": 0.0, "width": 90.0, "height": 60.0,
            "rows": [{"size": 20.0}, {"size": 24.0}, {"size": 16.0}],
            "columns": [{"size": 30.0}, {"size": 30.4}, {"size": 29.6}],
            "cells": [
                [cell("a0"), cell("a1"), cell("a2")],
                [cell("b0"), cell("b1"), cell("b2")],
                [cell("c0"), cell("c1"), cell("c2")]
            ],
            "backgroundColors": zebra,
            "cornerRadius": radius,
        }))
        .unwrap()
    }

    fn box_of(out: &[RawNode], id: &str) -> (f64, f64) {
        let n = out.iter().find(|n| n.str("id") == Some(id)).expect(id);
        (n.f64("width").unwrap(), n.f64("height").unwrap())
    }

    #[test]
    fn opaque_cell_backgrounds_reach_under_the_neighbours_drawn_after_them() {
        // Two cells each anti-aliasing their shared edge leave a seam of the
        // backdrop where the edge falls between device pixels; reaching under
        // the next cell draws that edge once (tableExpansion.ts parity).
        let out = expand_table_node(&seam_table(0.0, json!(["#13264A", "#1a3360"])));
        assert_eq!(box_of(&out, "a0_background"), (30.0 + 30.4, 20.0 + 24.0));
        assert_eq!(box_of(&out, "a1_background"), (30.4 + 29.6, 20.0 + 24.0));
        assert_eq!(box_of(&out, "a2_background"), (29.6, 20.0 + 24.0), "last column");
        assert_eq!(box_of(&out, "c1_background"), (30.4 + 29.6, 16.0), "last row");
        assert_eq!(box_of(&out, "c2_background"), (29.6, 16.0));

        // A rounded table corner is never covered: the reach into the last
        // column and row stops the radius short of it.
        let out = expand_table_node(&seam_table(6.0, json!(["#13264A"])));
        assert_eq!(box_of(&out, "a1_background"), (30.4 + 29.6 - 6.0, 20.0 + 24.0));
        assert_eq!(box_of(&out, "b1_background"), (30.4 + 29.6 - 6.0, 24.0 + 16.0 - 6.0));

        // Translucent, unparsed or absent fills, merged cells and a faded table
        // keep their own boxes.
        for zebra in [json!(["#13264A80"]), json!(["rgb(1,2,3)"]), json!([])] {
            let out = expand_table_node(&seam_table(0.0, zebra.clone()));
            assert_eq!(box_of(&out, "a0_background"), (30.0, 20.0), "{zebra}");
        }
        let mut faded = seam_table(0.0, json!(["#13264A"]));
        faded.set("opacity", json!(0.9));
        assert_eq!(box_of(&expand_table_node(&faded), "a0_background"), (30.0, 20.0));
        let mut merged = seam_table(0.0, json!(["#13264A"]));
        let mut cells = merged.json_field("cells").cloned().unwrap();
        cells[2][1]["colSpan"] = json!(2);
        merged.set("cells", cells);
        assert_eq!(box_of(&expand_table_node(&merged), "a0_background"), (30.0, 20.0));
    }

    #[test]
    fn opaque_hex_colors() {
        for c in ["#fff", "#FFFFFF", "#ffff", "#12345678ff", "#123456FF"] {
            assert_eq!(is_opaque_hex_color(Some(&json!(c))), c != "#12345678ff", "{c}");
        }
        for c in ["#fff8", "#12345680", "red", "#ggg", "", "transparent"] {
            assert!(!is_opaque_hex_color(Some(&json!(c))), "{c}");
        }
        assert!(!is_opaque_hex_color(Some(&json!({"variableId": "v"}))));
        assert!(!is_opaque_hex_color(None));
    }

    #[test]
    fn table_row_fill_wins_over_header_and_zebra() {
        // rows[i].backgroundColor (from <tr fill>) beats headerBackgroundColor on row 0
        // and the zebra cycle on later rows; fill-less rows keep the cascade
        // (tableExpansion.ts parity).
        let mut t = table();
        t.set(
            "rows",
            json!([{"size": 20.0, "backgroundColor": "#123456"}, {"size": 20.0}]),
        );
        let out = expand_table_node(&t);
        // row 0: explicit row fill beats headerBackgroundColor (#222222)
        assert_eq!(fill_of(&out[0]).as_deref(), Some("#123456"));
        assert_eq!(fill_of(&out[1]).as_deref(), Some("#123456"));
        // row 1 keeps the zebra cycle: backgroundColors[1 % 2]
        assert_eq!(fill_of(&out[2]).as_deref(), Some("#dddddd"));
    }

    /// A merged region draws ONE cell and none of the slots it covers, sized to
    /// their union — the Rust half of `tableExpansion.spans.test.ts`.
    #[test]
    fn a_merged_region_draws_one_cell_sized_to_its_union() {
        let mut t = table();
        let mut cells = t.array("cells").unwrap().to_vec();
        cells[0][0]["colSpan"] = json!(2);
        t.set("cells", Value::Array(cells));

        let out = expand_table_node(&t);
        let ids: Vec<&str> = out.iter().filter_map(|n| n.str("id")).collect();

        assert!(ids.contains(&"c00_background"), "the anchor still draws");
        assert!(
            !ids.contains(&"c01_background"),
            "a covered slot must draw nothing — the anchor paints over it"
        );
        // 2 columns x 30 wide, so the union is 60 rather than one column's 30.
        let bg = out
            .iter()
            .find(|n| n.str("id") == Some("c00_background"))
            .unwrap();
        assert_eq!(bg.json_field("width").and_then(Value::as_f64), Some(60.0));
    }

    /// The byte-identity contract: with no merge, the expansion is unchanged —
    /// same node ids, in the same order, as before spans existed. This is what
    /// keeps `mutation_replay` from moving on any document already in the wild.
    #[test]
    fn a_span_less_table_expands_exactly_as_before() {
        let plain = expand_table_node(&table());

        let mut explicit = table();
        let mut cells = explicit.array("cells").unwrap().to_vec();
        cells[0][0]["colSpan"] = json!(1);
        cells[1][1]["rowSpan"] = json!(1);
        explicit.set("cells", Value::Array(cells));

        let ids = |nodes: &[RawNode]| -> Vec<String> {
            nodes
                .iter()
                .filter_map(|n| n.str("id").map(str::to_string))
                .collect()
        };
        assert_eq!(
            ids(&expand_table_node(&explicit)),
            ids(&plain),
            "colSpan:1 is not a merge"
        );
        assert!(
            !ids(&plain).iter().any(|id| id.matches('_').count() > 4),
            "no id should carry a segment suffix on a merge-less table"
        );
    }

    /// A merge swallows the gridlines inside it, and leaves every other line alone.
    #[test]
    fn a_merge_breaks_only_the_lines_it_crosses() {
        let mut t = table();
        let mut cells = t.array("cells").unwrap().to_vec();
        cells[0][0]["colSpan"] = json!(2);
        t.set("cells", Value::Array(cells));

        let out = expand_table_node(&t);
        let left_lines: Vec<&str> = out
            .iter()
            .filter_map(|n| n.str("id"))
            .filter(|id| id.contains("_border_left_"))
            .collect();

        // The boundary left of column 1 is crossed on row 0, so it is drawn as a
        // segment over row 1 only, under a segment id.
        assert!(
            left_lines
                .iter()
                .any(|id| id.ends_with("_border_left_0_1_1")),
            "expected a segmented line, got {left_lines:?}"
        );
        assert!(
            !left_lines.contains(&"tbl_border_left_0_1"),
            "the full-height line through the merge must be gone"
        );
    }

    #[test]
    fn table_cell_background_wins_over_every_other_source() {
        // The full cascade, most specific first: cell, then row,
        // then header, then zebra. Written as one table because the ORDER is
        // the contract — checking any single level in isolation would pass on
        // a cascade assembled in the wrong sequence.
        //
        // The `cells` fixture is 2x2; `table()` sets headerBackgroundColor
        // #222222 and backgroundColors [#eeeeee, #dddddd].
        let mut t = table();
        t.set(
            "rows",
            json!([{"size": 20.0, "backgroundColor": "#123456"}, {"size": 20.0}]),
        );
        let mut cells = t.json_field("cells").cloned().expect("fixture has cells");
        // (0,0) overrides an explicit ROW fill, which itself beats the header.
        cells[0][0]["backgroundColor"] = json!("#ff0000");
        // (1,1) overrides the zebra cycle on a row with no fill of its own.
        cells[1][1]["backgroundColor"] = json!("#00ff00");
        // (1,0) says null, which means INHERIT — not "no background".
        cells[1][0]["backgroundColor"] = Value::Null;
        t.set("cells", cells);

        let out = expand_table_node(&t);

        assert_eq!(
            fill_of(&out[0]).as_deref(),
            Some("#ff0000"),
            "cell must beat the row fill"
        );
        assert_eq!(
            fill_of(&out[1]).as_deref(),
            Some("#123456"),
            "its neighbour keeps the row fill"
        );
        assert_eq!(
            fill_of(&out[2]).as_deref(),
            Some("#dddddd"),
            "a null cell background inherits the zebra"
        );
        assert_eq!(
            fill_of(&out[3]).as_deref(),
            Some("#00ff00"),
            "cell must beat the zebra cycle"
        );
    }

    #[test]
    fn table_cell_text_padding_insets_the_content_box() {
        // Text boxes have no padding: `textPadding` insets the content node
        // (clamped at zero) and never reaches it as a `padding` key.
        let mut t = table();
        let mut cells = t.json_field("cells").cloned().expect("fixture has cells");
        cells[0][0]["textPadding"] = json!(4.0);
        cells[0][1]["textPadding"] = json!(40.0);
        t.set("cells", cells);
        let out = expand_table_node(&t);
        let c00 = &out[4];
        assert_eq!(c00.id(), "c00_content");
        assert_eq!(c00.f64("x"), Some(105.0));
        assert_eq!(c00.f64("y"), Some(205.0));
        assert_eq!(c00.f64("width"), Some(20.0));
        assert_eq!(c00.f64("height"), Some(10.0));
        assert!(c00.json_field("padding").is_none());
        let c01 = &out[5];
        assert_eq!(c01.f64("width"), Some(0.0));
        assert_eq!(c01.f64("height"), Some(0.0));
    }

    #[test]
    fn a_cells_alignment_reaches_every_paragraph() {
        // `A<br/>B` in a right-aligned cell is two paragraphs: both align, not the first alone.
        assert_eq!(
            inject_text_align_into_html(r#"<p style="font-size: 18px; text-align: left">A</p><p style="font-size: 18px">B</p><p>C</p>"#, "right"),
            r#"<p style="text-align: right; font-size: 18px">A</p><p style="text-align: right; font-size: 18px">B</p><p style="text-align: right">C</p>"#
        );
        // Other tags (`<pre>`, `<span>`) and a paragraph's other attributes are left as they are.
        assert_eq!(
            inject_text_align_into_html(r#"<p class="x"><span>a</span></p><pre>b</pre>"#, "center"),
            r#"<p style="text-align: center" class="x"><span>a</span></p><pre>b</pre>"#
        );
    }

    #[test]
    fn table_expansion_order_and_geometry() {
        // tableExpansion.ts:232 — [...backgrounds, ...contents, ...borders]
        let out = expand_table_node(&table());
        // 4 backgrounds, 4 contents, 3 row + 3 col borders
        assert_eq!(out.len(), 4 + 4 + 6);
        assert_eq!(out[0].id(), "c00_background");
        assert_eq!(out[4].id(), "c00_content");
        assert!(out[8].id().starts_with("tbl_border_"));

        // backgrounds carry table x/y (parent-relative, scene-schema.md §5.1)
        assert_eq!(out[0].f64("x"), Some(100.0));
        assert_eq!(out[1].f64("x"), Some(130.0)); // col 1
                                                  // header bg overrides zebra on row 0 (tableExpansion.ts:186-191)
        assert_eq!(fill_of(&out[0]).as_deref(), Some("#222222"));
        // row 1 zebra: backgroundColors[1 % 2]
        assert_eq!(fill_of(&out[2]).as_deref(), Some("#dddddd"));
        // corner radii only on corner cells (tableExpansion.ts:169-178)
        assert_eq!(
            out[0].json_field("cornerRadius"),
            Some(&json!([8.0, 0.0, 0.0, 0.0]))
        );
        assert_eq!(
            out[3].json_field("cornerRadius"),
            Some(&json!([0.0, 0.0, 8.0, 0.0]))
        );

        // contents inset by strokeWidth/2 and shrunk by strokeWidth
        // (tableExpansion.ts:157-160)
        assert_eq!(out[4].f64("x"), Some(101.0));
        assert_eq!(out[4].f64("width"), Some(28.0));
        // table opacity multiplied in (tableExpansion.ts:97)
        assert_eq!(out[4].f64("opacity"), Some(0.5));

        // text-align injected for non-left cells (tableSelection.ts:101-106)
        assert_eq!(
            out[5].str("htmlContent"),
            Some("<p style=\"text-align: center\">b</p>")
        );

        // borders extend strokeWidth/2 past edges (tableExpansion.ts:108-111)
        let b = &out[8]; // first border: row top 0
        assert_eq!(b.json_field("points"), Some(&json!([99.0, 200.0, 161.0, 200.0])));
    }

    #[test]
    fn table_cell_contents_clip_text_to_their_own_box() {
        // Overflowing cell text must not paint over the neighbouring cells,
        // so every CONTENT node carries `clipTextToBounds: true` — and only the
        // content nodes (a clip on the backgrounds/borders would be meaningless
        // schema spill). Mirrors the TS expansion (tableExpansion.ts:162-165).
        let out = expand_table_node(&table());
        let contents: Vec<&RawNode> = out.iter().filter(|n| n.node_type() == "richtext").collect();
        assert_eq!(contents.len(), 4);
        for c in &contents {
            assert_eq!(
                c.json_field("clipTextToBounds"),
                Some(&json!(true)),
                "cell content {} must clip text to its own box",
                c.id()
            );
        }
        assert!(out
            .iter()
            .filter(|n| n.node_type() != "richtext")
            .all(|n| !n.has("clipTextToBounds")));
    }

    /// Ids of the non-border nodes (cell contents + backgrounds), which are what the
    /// Scene transient registries are keyed by.
    fn cell_node_ids(out: &[RawNode]) -> Vec<String> {
        out.iter()
            .filter(|n| n.node_type() != "line")
            .map(|n| n.id().to_string())
            .collect()
    }

    // The next five tests pin degradation parity with the TS regression suite
    // (`tests/vitest/utils/markup/table.test.ts`, "malformed table resilience").
    // `rows`/`columns`/`cells` are typed required, but tables also arrive from persisted JSON,
    // agent patches and collab merges — where that is an assumption, not a guarantee
    // (WEB-NEXTJS-22R).

    #[test]
    fn malformed_cells_renders_borders_only() {
        // The TS guards explicitly and reports to Sentry; this lane degrades for free because
        // every accessor is defensive. Either way the table must not vanish — borders still
        // render off rows/columns, only cell content and backgrounds are lost.
        for cells in [json!(null), json!("not-an-array"), json!({})] {
            let mut t = table();
            t.set("cells", cells);
            let out = expand_table_node(&t);
            assert!(!out.is_empty(), "a malformed table still renders borders");
            assert!(out.iter().all(|n| n.node_type() == "line"));
        }

        let mut t = table();
        t.remove("cells");
        assert!(expand_table_node(&t)
            .iter()
            .all(|n| n.node_type() == "line"));
    }

    #[test]
    fn malformed_rows_degrades_instead_of_panicking() {
        // The TS reads `rows.length` before any guard, so a missing `rows` throws and
        // `Scene.computeResolvedNodes` contains the failure to this one node. Here the defensive
        // accessors degrade to a zero-height table instead — the invariant both lanes share is
        // that a corrupt table never takes the rest of the render down with it.
        let mut t = table();
        t.remove("rows");
        let out = expand_table_node(&t);
        assert!(cell_node_ids(&out).is_empty(), "no cells without rows");
        assert!(out.iter().all(|n| n.node_type() == "line"));

        // A malformed row ENTRY is not fatal either; it just sizes to 0.
        let mut t = table();
        t.set("rows", json!([{"size": null}, "nonsense"]));
        let out = expand_table_node(&t);
        assert_eq!(cell_node_ids(&out).len(), 8);
        assert!(out
            .iter()
            .all(|n| n.f64("height").unwrap_or(0.0).is_finite()));
    }

    #[test]
    fn idless_cells_get_distinct_deterministic_ids() {
        // Every id-less cell used to mint `"_content"` / `"_background"`, which collide in Scene's
        // transient registries (last-cell-wins) so only one of them rendered.
        let mut t = table();
        t.set(
            "cells",
            json!([
                [{"textContent": "<p>a</p>"}, {"id": "", "textContent": "<p>b</p>"}],
                [{"id": null, "textContent": "<p>c</p>"}, {"textContent": "<p>d</p>"}]
            ]),
        );
        let ids = cell_node_ids(&expand_table_node(&t));

        assert_eq!(ids.len(), 8, "4 cells × (content + background)");
        assert_eq!(
            ids.iter().collect::<HashSet<_>>().len(),
            ids.len(),
            "the positional fallback keeps every id distinct"
        );
        assert!(ids.iter().all(|id| !id.starts_with('_')));
        assert!(ids.contains(&"tbl_cell_0_0_content".to_string()));
        assert!(ids.contains(&"tbl_cell_1_1_background".to_string()));
    }

    #[test]
    fn a_real_cell_id_is_preserved() {
        let ids = cell_node_ids(&expand_table_node(&table()));
        assert!(ids.contains(&"c00_content".to_string()));
        assert!(ids.contains(&"c11_background".to_string()));
    }

    #[test]
    fn duplicate_real_cell_ids_are_deduped() {
        // A repeated NON-EMPTY id collides exactly like the id-less case, so it takes the same
        // positional fallback. The first claimant keeps the id.
        let mut t = table();
        t.set(
            "cells",
            json!([
                [{"id": "dup", "textContent": "<p>a</p>"}, {"id": "dup", "textContent": "<p>b</p>"}],
                [{"id": "dup", "textContent": "<p>c</p>"}, {"id": "other", "textContent": "<p>d</p>"}]
            ]),
        );
        let ids = cell_node_ids(&expand_table_node(&t));

        assert_eq!(ids.iter().collect::<HashSet<_>>().len(), ids.len());
        assert!(ids.contains(&"dup_content".to_string()));
        assert!(ids.contains(&"tbl_cell_0_1_content".to_string()));
        assert!(ids.contains(&"tbl_cell_1_0_background".to_string()));
        assert!(ids.contains(&"other_content".to_string()));
    }

    #[test]
    fn table_border_configs() {
        // lineShouldRender (tableExpansion.ts:241-265)
        let mut t = table();
        t.set("rowStrokeConfig", json!("outside"));
        t.set("columnStrokeConfig", json!("after-first"));
        let out = expand_table_node(&t);
        let borders: Vec<&RawNode> = out.iter().filter(|n| n.node_type() == "line").collect();
        // rows: index 0 and 2 (outside) → 2; columns: index 1 only → 1
        assert_eq!(borders.len(), 3);
    }

    #[test]
    fn vector_qr_expands_to_local_background_and_module_path() {
        // generatedGraphicExpansion.ts expandQRVectorNode: a 2x2-module natural box with one
        // dark module (top right), drawn at 100x50.
        let n = RawNode::from_value(json!({
            "id":"q","type":"generated","generatedKind":"qr",
            "x":10.0,"y":20.0,"width":100.0,"height":50.0,"rotation":0,"scale":1,"opacity":0.5,
            "renderMode":"native-children",
            "cachedVectorPath":"M1 0H2V1H1Z",
            "cachedNaturalDimensions":{"width":2.0,"height":2.0},
            "generatedSource":{"kind":"qr","data":"x","errorCorrection":"M","foregroundColor":"#112233"}
        }))
        .unwrap();
        let out = expand_generated_graphic_node(&n).unwrap();
        assert_eq!(out.len(), 2);
        let bg = &out[0];
        assert_eq!(bg.id(), "q_gen_proxy_0");
        assert_eq!(bg.node_type(), "rectangle");
        assert_eq!((bg.f64("x"), bg.f64("y")), (Some(0.0), Some(0.0)));
        assert_eq!(
            (bg.f64("width"), bg.f64("height")),
            (Some(100.0), Some(50.0))
        );
        assert_eq!(fill_of(bg).as_deref(), Some("#ffffff"));
        assert_eq!(bg.f64("opacity"), Some(0.5));
        let path = &out[1];
        assert_eq!(path.id(), "q_gen_proxy_1");
        assert_eq!(path.node_type(), "path");
        assert_eq!(fill_of(path).as_deref(), Some("#112233"));
        assert!(paint::top_stroke(path).is_none(), "the module path has no stroke");
        assert_eq!(
            path.json_field("points"),
            Some(&json!([
                {"type":"move","x":50.0,"y":0.0},
                {"type":"line","x":100.0,"y":0.0},
                {"type":"line","x":100.0,"y":25.0},
                {"type":"line","x":50.0,"y":25.0},
                {"type":"line","x":50.0,"y":0.0}
            ]))
        );
    }

    #[test]
    fn vector_path_parser_handles_multiple_outlines() {
        let pts = parse_vector_path("M0 0H1V1H0ZM2 2H3V3H2Z", 10.0, 10.0);
        let moves = pts.iter().filter(|p| p["type"] == "move").count();
        assert_eq!(moves, 2);
        assert_eq!(pts.len(), 10);
        assert_eq!(pts[5], json!({"type":"move","x":20.0,"y":20.0}));
    }

    #[test]
    fn vector_path_parser_tolerates_garbage() {
        assert!(parse_vector_path("é M1 2 ✓H3 Z", 1.0, 1.0).len() >= 2);
        assert!(parse_vector_path("", 1.0, 1.0).is_empty());
    }

    #[test]
    fn qr_without_vector_path_draws_the_placeholder() {
        let n = RawNode::from_value(json!({
            "id":"q","type":"generated","generatedKind":"qr",
            "x":0.0,"y":0.0,"width":10.0,"height":10.0,
            "renderMode":"native-children",
            "generatedSource":{"kind":"qr","data":"x","errorCorrection":"M"}
        }))
        .unwrap();
        assert!(!is_vector_qr(&n));
        let out = expand_generated_graphic_node(&n).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(stroke_of(&out[0]).as_deref(), Some("#cccccc"));
        assert!(
            matches!(paint::top_fill(&out[0]).map(|f| f.paint), Some(paint::Paint::Solid(_))),
            "the placeholder fills a solid, not an image"
        );
    }

    #[test]
    fn latex_with_cached_asset_expands_to_pattern_rect() {
        // generatedGraphicExpansion.ts:51-84
        let n = RawNode::from_value(json!({
            "id":"lx1","type":"generated","generatedKind":"latex",
            "x":10.0,"y":20.0,"width":100.0,"height":100.0,"rotation":0,"scale":1,
            "renderMode":"raster-asset",
            "cachedAssetUrl":"https://assets/latex.png",
            "cachedNaturalDimensions":{"width":200.0,"height":200.0},
            "generatedSource":{"kind":"latex","latex":"x","displayMode":"display","fontSize":32}
        }))
        .unwrap();
        let out = expand_generated_graphic_node(&n).unwrap();
        assert_eq!(out.len(), 1);
        let r = &out[0];
        assert_eq!(r.id(), "lx1_gen_proxy_0");
        let Some(paint::Paint::Image(img)) = paint::top_fill(r).map(|f| f.paint) else {
            panic!("the cached asset is an image fill");
        };
        assert_eq!(img.src.as_deref(), Some("https://assets/latex.png"));
        assert_eq!(img.placement.scale, Some(paint::PaintPoint { x: 0.5, y: 0.5 }));
        assert_eq!(img.placement.fit.as_deref(), Some("crop"));
        assert_eq!(img.placement.box_size, Some(paint::PaintSize { width: 100.0, height: 100.0 }));
        assert_eq!(r.str("_transformSourceId"), Some("lx1"));
        assert_eq!(r.f64("x"), Some(10.0)); // baked host x/y
    }

    #[test]
    fn qr_without_asset_gets_placeholder_with_qr_bg() {
        // generatedGraphicExpansion.ts:86-117
        let n = RawNode::from_value(json!({
            "id":"qr1","type":"generated","generatedKind":"qr",
            "x":0.0,"y":0.0,"width":50.0,"height":50.0,"rotation":0,"scale":1,
            "renderMode":"native-children",
            "generatedSource":{"kind":"qr","data":"x","errorCorrection":"M","backgroundColor":"#aabbcc"}
        }))
        .unwrap();
        let out = expand_generated_graphic_node(&n).unwrap();
        assert_eq!(fill_of(&out[0]).as_deref(), Some("#aabbcc"));
        assert_eq!(stroke_of(&out[0]).as_deref(), Some("#cccccc"));
        assert_eq!(paint::top_stroke(&out[0]).map(|s| s.width), Some(1.0));
    }

    #[test]
    fn unknown_generated_kind_expands_to_parity_placeholder_rect() {
        // The default arm (unrecognized / absent generatedKind) renders the SAME
        // placeholder rect the browser draws (generatedGraphicExpansion.ts:45-48
        // default branch → createPlaceholderRect :86-105): #f0f0f0 fill,
        // #cccccc stroke, strokeWidth??1, host geometry/opacity. This is the
        // pixel evidence behind assess.rs marking `unknown:generated-kind`
        // Degraded (renderable) rather than Unknown (browser-decline).
        for kind_field in [json!({"generatedKind":"weird"}), json!({})] {
            let mut base = json!({
                "id":"g1","type":"generated",
                "x":5.0,"y":6.0,"width":50.0,"height":40.0,"rotation":0,"scale":1,
                "opacity":0.5
            });
            base.as_object_mut()
                .unwrap()
                .extend(kind_field.as_object().unwrap().clone());
            let n = RawNode::from_value(base).unwrap();
            let out = expand_generated_graphic_node(&n).unwrap();
            assert_eq!(out.len(), 1);
            let r = &out[0];
            assert_eq!(r.id(), "g1_gen_proxy_0");
            assert_eq!(r.node_type(), "rectangle");
            assert_eq!(fill_of(r).as_deref(), Some("#f0f0f0"));
            assert_eq!(stroke_of(r).as_deref(), Some("#cccccc"));
            assert_eq!(paint::top_stroke(r).map(|s| s.width), Some(1.0));
            assert_eq!(r.f64("x"), Some(5.0));
            assert_eq!(r.f64("width"), Some(50.0));
            assert_eq!(r.f64("opacity"), Some(0.5));
            assert_eq!(r.str("_transformSourceId"), Some("g1"));
        }
    }

    #[test]
    fn chart_expands_to_placeholder_in_local_coords() {
        // chartExpansion.ts:131-159 (no generator registered in moda-scene)
        let n = RawNode::from_value(json!({
            "id":"ch1","type":"generated","generatedKind":"chart",
            "x":40.0,"y":40.0,"width":300.0,"height":200.0,"rotation":0,"scale":1,
            "renderMode":"native-children",
            "generatedSource":{"kind":"chart","chartVersion":1,"spec":{"type":"bar"},
                "data":{"columns":[],"rows":[],"series":[],"categories":[],"valueRange":[0,0]}}
        }))
        .unwrap();
        let out = expand_generated_graphic_node(&n).unwrap();
        assert_eq!(out.len(), 1);
        let p = &out[0];
        assert_eq!(p.id(), "ch1_gen_proxy_0_placeholder");
        assert_eq!(p.f64("x"), Some(0.0)); // LOCAL coords — host is transform parent
        assert_eq!(fill_of(p).as_deref(), Some("#f5f5f5"));
        assert_eq!(
            p.object("metadata").and_then(|m| m.get("chartPlaceholder")),
            Some(&json!(true))
        );
    }

    #[test]
    fn page_bg_rect() {
        // shapeHelpers.ts:803-842 + Scene.ts:3247-3264: the page's `fills`
        // become the background rectangle's.
        let page = page(json!({"id":"page-1","name":"P","width":1080.0,"height":1080.0,
            "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#fafafa"}}],"nodeIds":[]}));
        let bg = create_rectangle_from_page(&page);
        assert_eq!(bg.id(), "__page_bg_page-1");
        assert_eq!(bg.f64("x"), Some(0.0));
        assert_eq!(bg.f64("width"), Some(1080.0));
        assert_eq!(fill_of(&bg).as_deref(), Some("#fafafa"));
        assert_eq!(bg.bool("locked"), Some(true));
    }

    #[test]
    fn page_bg_with_empty_fills_paints_nothing() {
        // Pages always carry `fills` now; the white default for a colour page
        // lives in the document codemod (024-paint-stacks `transformPage`),
        // not in a render-time `?? '#ffffff'`. `[]` = no background [Q5].
        let empty = page(json!({"id":"p","name":"P","width":10.0,"height":10.0,"nodeIds":[],"fills":[]}));
        assert!(paint::top_fill(&create_rectangle_from_page(&empty)).is_none());
        let white = page(json!({"id":"p","name":"P","width":10.0,"height":10.0,"nodeIds":[],
            "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ffffff"}}]}));
        assert_eq!(
            fill_of(&create_rectangle_from_page(&white)).as_deref(),
            Some("#ffffff")
        );
    }
}

#[cfg(test)]
mod auto_height_tests {
    use super::*;
    fn height(node: &RawNode) -> Option<f64> {
        assert!(node.f64("width").unwrap() > 0.0);
        assert_eq!(node.str("verticalAlign"), Some("top"));
        assert!(node.json_field("height").is_none());
        Some(if node.str("htmlContent").unwrap_or("").contains("Presentations") { 60.0 } else { 24.0 })
    }
    #[test]
    fn table_fixture_auto_and_fixed_rows() {
        let value: Value = crate::value::from_str(include_str!("../tests/fixtures/table.json")).unwrap();
        let mut table = RawNode::new(value["nodes"].as_array().unwrap().iter().find(|n| n["type"] == "table").unwrap().as_object().unwrap().clone());
        let fitted = fit_table_rows_with(&table, height).unwrap();
        assert_ne!(fitted.json_field("rows"), table.json_field("rows"));
        let children = expand_table_node(&fitted);
        assert!(children.iter().any(|n| n.str("htmlContent").is_some_and(|s| s.contains("Presentations")) && n.f64("height").unwrap() >= 60.0));
        assert!(children.iter().filter(|n| n.node_type() == "richtext").all(|n| n.json_field("clipTextToBounds") == Some(&json!(true))));
        table.set("rowSizing", json!("fixed"));
        assert!(fit_table_rows_with(&table, height).is_none());
    }
    #[test]
    fn small_text_rows_fit_their_text_and_only_empty_rows_keep_the_floor() {
        fn small(node: &RawNode) -> Option<f64> {
            Some(if node.str("htmlContent").unwrap_or("").contains("8pt") { 10.0 } else { 0.0 })
        }
        let value: Value = crate::value::from_str(include_str!("../tests/fixtures/table.json")).unwrap();
        let mut table = RawNode::new(value["nodes"].as_array().unwrap().iter().find(|n| n["type"] == "table").unwrap().as_object().unwrap().clone());
        let cells: Vec<Value> = table.array("cells").unwrap().iter().enumerate().map(|(r, row)| {
            json!(row.as_array().unwrap().iter().map(|cell| {
                let mut cell = cell.as_object().unwrap().clone();
                cell.insert("textContent".into(), json!(if r == 1 { "" } else { "<p>8pt</p>" }));
                cell.insert("textPadding".into(), json!(3.0));
                Value::Object(cell)
            }).collect::<Vec<_>>())
        }).collect();
        table.set("cells", json!(cells));
        let fitted = fit_table_rows_with(&table, small).unwrap();
        let rows = fitted.array("rows").unwrap();
        let size = |r: usize| rows[r].get("size").and_then(Value::as_f64).unwrap();
        assert!(size(0) < 32.0 && size(0) >= 19.0, "a text row is as tall as its text: {}", size(0));
        assert_eq!(size(1), 32.0, "an empty row keeps the floor");
    }
}
