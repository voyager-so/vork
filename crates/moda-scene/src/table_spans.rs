//! Merged table cells: which grid slots an anchor covers.
//!
//! Byte-exact mirror of `core/scene/tableSpans.ts`, which is the oracle. The
//! model is dense-with-anchor-only spans: `cells` keeps one entry per grid
//! position, a merge is expressed ONLY by `colSpan`/`rowSpan` on its top-left
//! cell, and the covered slots carry no marker — coverage is derived here so
//! there is never a second copy of the truth to drift.
//!
//! The two repair rules are repairs rather than validation, because a table
//! arrives from persisted JSON, an agent patch and a collab merge, where a span
//! can be nonsense in ways no UI would produce:
//!
//! 1. **Clamp** a span reaching past the last row/column to the edge.
//! 2. **First anchor wins**, row-major. A later anchor whose region would
//!    overlap an already-covered slot is demoted to a plain 1x1 cell — not
//!    partially honoured, because a half-drawn merge is a rendering artefact.

use crate::value::Value;
use std::collections::{HashMap, HashSet};

/// A resolved merge: where it starts and how far it reaches, already clamped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanRegion {
    pub row_idx: usize,
    pub col_idx: usize,
    /// Always >= 1.
    pub row_span: usize,
    /// Always >= 1.
    pub col_span: usize,
}

#[derive(Debug, Default)]
pub struct ResolvedSpans {
    /// Anchors that cover more than their own slot.
    pub regions: HashMap<(usize, usize), SpanRegion>,
    /// Slots covered by some OTHER cell's region.
    pub covered: HashSet<(usize, usize)>,
}

impl ResolvedSpans {
    /// No merges at all — the byte-identical fast path.
    pub fn is_plain_grid(&self) -> bool {
        self.regions.is_empty()
    }
}

/// A span value as written, coerced to a usable count.
///
/// Anything that is not a finite number >= 1 means "no span". A fraction is
/// rounded DOWN (`floor`, matching the TS `Math.floor`) so a repair can only
/// ever shrink a region, never invent coverage the author's value did not reach.
fn span_count(raw: Option<&Value>) -> usize {
    let Some(n) = raw.and_then(Value::as_f64) else {
        return 1;
    };
    if !n.is_finite() || n < 1.0 {
        return 1;
    }
    n.floor() as usize
}

/// Resolve every merge on a table. Row-major, so "first anchor wins" is stable.
pub fn resolve_table_spans(
    rows_len: usize,
    cols_len: usize,
    cells: Option<&[Value]>,
) -> ResolvedSpans {
    let mut out = ResolvedSpans::default();
    let Some(cells) = cells else { return out };

    for row_idx in 0..rows_len {
        let Some(row) = cells.get(row_idx).and_then(Value::as_array) else {
            continue;
        };

        for col_idx in 0..cols_len {
            // A covered slot cannot itself anchor a region: it is already inside
            // one, and honouring a span on it would put two merges over a slot.
            if out.covered.contains(&(row_idx, col_idx)) {
                continue;
            }

            let Some(cell) = row.get(col_idx).and_then(Value::as_object) else {
                continue;
            };

            // Clamp before anything else, so every later step works on a region
            // that actually exists.
            let row_span = span_count(cell.get("rowSpan")).min(rows_len - row_idx);
            let col_span = span_count(cell.get("colSpan")).min(cols_len - col_idx);
            if row_span <= 1 && col_span <= 1 {
                continue;
            }

            // Checked BEFORE claiming any slot, so a rejected span leaves no
            // partial trace.
            let mut fits = true;
            'outer: for r in row_idx..row_idx + row_span {
                for c in col_idx..col_idx + col_span {
                    if r == row_idx && c == col_idx {
                        continue;
                    }
                    if out.covered.contains(&(r, c)) {
                        fits = false;
                        break 'outer;
                    }
                }
            }
            if !fits {
                continue;
            }

            for r in row_idx..row_idx + row_span {
                for c in col_idx..col_idx + col_span {
                    if r == row_idx && c == col_idx {
                        continue;
                    }
                    out.covered.insert((r, c));
                }
            }
            out.regions.insert(
                (row_idx, col_idx),
                SpanRegion {
                    row_idx,
                    col_idx,
                    row_span,
                    col_span,
                },
            );
        }
    }

    out
}

/// Which axis a grid line runs along, for [`span_crosses_line`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LineAxis {
    Column,
    Row,
}

/// Does a merge cross the grid line at `line_index`, so it must not be drawn there?
///
/// A `Column` line at index `c` is the boundary to the LEFT of column `c`, crossed
/// by any region starting before `c` and reaching past it. The outer edges are
/// never crossed: a region is clamped inside the grid.
pub fn span_crosses_line(
    spans: &ResolvedSpans,
    line_index: usize,
    axis: LineAxis,
    at_index: usize,
) -> bool {
    if spans.is_plain_grid() {
        return false;
    }
    spans.regions.values().any(|region| match axis {
        LineAxis::Column => {
            region.col_span > 1
                && region.col_idx < line_index
                && line_index < region.col_idx + region.col_span
                && at_index >= region.row_idx
                && at_index < region.row_idx + region.row_span
        }
        LineAxis::Row => {
            region.row_span > 1
                && region.row_idx < line_index
                && line_index < region.row_idx + region.row_span
                && at_index >= region.col_idx
                && at_index < region.col_idx + region.col_span
        }
    })
}

/// The unbroken runs a grid line is drawn in, given which slots a merge swallows.
///
/// The `whole` flag is the byte-identity contract: with nothing swallowed the
/// caller emits the ORIGINAL single node — same id, same full-bleed endpoints —
/// so a table with no merges expands exactly as it did before merged cells.
pub fn line_runs(
    slot_count: usize,
    is_swallowed: impl Fn(usize) -> bool,
) -> Vec<(usize, usize, bool)> {
    let mut runs = Vec::new();
    let mut from: Option<usize> = None;
    for slot in 0..slot_count {
        if is_swallowed(slot) {
            if let Some(start) = from.take() {
                runs.push((start, slot, false));
            }
            continue;
        }
        if from.is_none() {
            from = Some(slot);
        }
    }
    if let Some(start) = from {
        runs.push((start, slot_count, start == 0));
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    fn cells(grid: Vec<Vec<Value>>) -> Vec<Value> {
        grid.into_iter().map(Value::Array).collect()
    }
    fn cell(spec: Value) -> Value {
        spec
    }

    #[test]
    fn a_span_less_table_is_a_plain_grid() {
        let c = cells(vec![vec![cell(json!({"id":"a"})), cell(json!({"id":"b"}))]]);
        let spans = resolve_table_spans(1, 2, Some(&c));
        assert!(spans.is_plain_grid());
        assert!(spans.covered.is_empty());
    }

    #[test]
    fn a_colspan_covers_the_slots_to_its_right() {
        let c = cells(vec![vec![
            cell(json!({"id":"a","colSpan":3})),
            cell(json!({"id":"b"})),
            cell(json!({"id":"c"})),
            cell(json!({"id":"d"})),
        ]]);
        let spans = resolve_table_spans(1, 4, Some(&c));
        assert_eq!(spans.covered.len(), 2);
        assert!(spans.covered.contains(&(0, 1)));
        assert!(spans.covered.contains(&(0, 2)));
        assert!(
            !spans.covered.contains(&(0, 0)),
            "the anchor is not covered"
        );
    }

    #[test]
    fn a_span_off_the_grid_is_clamped() {
        // tableSpans.ts parity: a 9-wide span on a 3-wide table covers 2 slots,
        // never a key no consumer can index.
        let c = cells(vec![
            vec![
                cell(json!({"id":"a","colSpan":9,"rowSpan":9})),
                cell(json!({"id":"b"})),
                cell(json!({"id":"c"})),
            ],
            vec![
                cell(json!({"id":"d"})),
                cell(json!({"id":"e"})),
                cell(json!({"id":"f"})),
            ],
        ]);
        let spans = resolve_table_spans(2, 3, Some(&c));
        let region = spans.regions.get(&(0, 0)).expect("anchor");
        assert_eq!((region.col_span, region.row_span), (3, 2));
        assert_eq!(spans.covered.len(), 5);
    }

    #[test]
    fn an_uncovered_anchor_that_would_collide_is_demoted_whole() {
        // The case the covered-slot guard does not catch: (1,1) is free, but its
        // 2-wide region would reach (1,2), which (0,2)'s rowSpan already claimed.
        let c = cells(vec![
            vec![
                cell(json!({"id":"a","rowSpan":2})),
                cell(json!({"id":"b"})),
                cell(json!({"id":"c","rowSpan":2})),
            ],
            vec![
                cell(json!({"id":"d"})),
                cell(json!({"id":"e","colSpan":2})),
                cell(json!({"id":"f"})),
            ],
        ]);
        let spans = resolve_table_spans(2, 3, Some(&c));
        assert!(
            !spans.regions.contains_key(&(1, 1)),
            "the colliding anchor is demoted"
        );
        assert_eq!(spans.covered.len(), 2);
        assert!(spans.covered.contains(&(1, 0)) && spans.covered.contains(&(1, 2)));
    }

    #[test]
    fn a_non_count_span_is_no_span() {
        for bad in [json!(0), json!(-3), json!("two"), Value::Null] {
            let c = cells(vec![vec![
                cell(json!({"id":"a","colSpan":bad})),
                cell(json!({"id":"b"})),
            ]]);
            assert!(
                resolve_table_spans(1, 2, Some(&c)).is_plain_grid(),
                "colSpan={bad:?} must not merge"
            );
        }
    }

    #[test]
    fn a_fractional_span_rounds_down() {
        let c = cells(vec![vec![
            cell(json!({"id":"a","colSpan":2.9})),
            cell(json!({"id":"b"})),
            cell(json!({"id":"c"})),
        ]]);
        let spans = resolve_table_spans(1, 3, Some(&c));
        assert_eq!(spans.regions.get(&(0, 0)).expect("anchor").col_span, 2);
    }

    #[test]
    fn line_runs_reports_one_whole_run_when_nothing_is_swallowed() {
        assert_eq!(line_runs(4, |_| false), vec![(0, 4, true)]);
    }

    #[test]
    fn line_runs_splits_around_swallowed_slots_and_never_marks_them_whole() {
        assert_eq!(line_runs(4, |s| s == 0), vec![(1, 4, false)]);
        assert_eq!(line_runs(4, |s| s == 2), vec![(0, 2, false), (3, 4, false)]);
        assert!(line_runs(3, |_| true).is_empty());
    }
}
