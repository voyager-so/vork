//! Port of `utils/markup/elements/charts/stack.ts` — d3-shape `stack()` with
//! `stackOrderNone` and the two offsets the chart engine uses
//! (`d3-shape/src/offset/{none,diverging}.js`).

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StackSegment {
    pub y0: f64,
    pub y1: f64,
    pub value: f64,
}

pub type StackResult = HashMap<String, HashMap<String, StackSegment>>;

/// `computeStack` (stack.ts:84-141). `value_for(category, series)` mirrors the
/// TS callback; non-finite/missing → 0 (stack.ts:92-94).
pub fn compute_stack(
    categories: &[String],
    series: &[String],
    value_for: impl Fn(&str, &str) -> Option<f64>,
    mode: &str,   // 'stacked' | 'stacked-percent'
    offset: &str, // 'running-sum' | 'diverging'
) -> StackResult {
    let n = series.len();
    let m = categories.len();

    // rows[category][series]
    let mut rows: Vec<Vec<f64>> = categories
        .iter()
        .map(|c| {
            series
                .iter()
                .map(|s| match value_for(c, s) {
                    Some(v) if v.is_finite() => v,
                    _ => 0.0,
                })
                .collect()
        })
        .collect();

    if mode == "stacked-percent" {
        for row in rows.iter_mut() {
            let total: f64 = row.iter().sum();
            if total == 0.0 {
                continue;
            }
            for v in row.iter_mut() {
                *v = (*v / total) * 100.0;
            }
        }
    }

    // d3.stack(): series[s][c] = [0, value] then offset(series, order)
    let mut stacked: Vec<Vec<[f64; 2]>> = (0..n)
        .map(|s| (0..m).map(|c| [0.0, rows[c][s]]).collect())
        .collect();

    if offset == "diverging" {
        // offset/diverging.js
        if n > 0 {
            for j in 0..m {
                let mut yp = 0.0;
                let mut yn = 0.0;
                for i in 0..n {
                    let d = &mut stacked[i][j];
                    let dy = d[1] - d[0];
                    if dy > 0.0 {
                        d[0] = yp;
                        yp += dy;
                        d[1] = yp;
                    } else if dy < 0.0 {
                        d[1] = yn;
                        yn += dy;
                        d[0] = yn;
                    } else {
                        d[0] = 0.0;
                        d[1] = dy;
                    }
                }
            }
        }
    } else {
        // offset/none.js — running sum
        if n > 1 {
            for i in 1..n {
                for j in 0..m {
                    let prev = stacked[i - 1][j];
                    let base = if prev[1].is_nan() { prev[0] } else { prev[1] };
                    stacked[i][j][0] = base;
                    stacked[i][j][1] += base;
                }
            }
        }
    }

    let mut result: StackResult = HashMap::new();
    for (c, category) in categories.iter().enumerate() {
        let mut per_series = HashMap::new();
        for (s, series_key) in series.iter().enumerate() {
            per_series.insert(
                series_key.clone(),
                StackSegment {
                    y0: stacked[s][c][0],
                    y1: stacked[s][c][1],
                    value: rows[c][s],
                },
            );
        }
        result.insert(category.clone(), per_series);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diverging_splits_at_zero() {
        let cats = vec!["a".to_string()];
        let ser = vec!["s1".to_string(), "s2".to_string(), "s3".to_string()];
        let vals = |c: &str, s: &str| -> Option<f64> {
            let _ = c;
            Some(match s {
                "s1" => 3.0,
                "s2" => -2.0,
                _ => 4.0,
            })
        };
        let r = compute_stack(&cats, &ser, vals, "stacked", "diverging");
        let a = &r["a"];
        assert_eq!(
            a["s1"],
            StackSegment {
                y0: 0.0,
                y1: 3.0,
                value: 3.0
            }
        );
        assert_eq!(
            a["s2"],
            StackSegment {
                y0: -2.0,
                y1: 0.0,
                value: -2.0
            }
        );
        assert_eq!(
            a["s3"],
            StackSegment {
                y0: 3.0,
                y1: 7.0,
                value: 4.0
            }
        );
    }

    #[test]
    fn running_sum_layers() {
        let cats = vec!["a".to_string()];
        let ser = vec!["s1".to_string(), "s2".to_string()];
        let r = compute_stack(
            &cats,
            &ser,
            |_, s| Some(if s == "s1" { 2.0 } else { 5.0 }),
            "stacked",
            "running-sum",
        );
        let a = &r["a"];
        assert_eq!(
            a["s1"],
            StackSegment {
                y0: 0.0,
                y1: 2.0,
                value: 2.0
            }
        );
        assert_eq!(
            a["s2"],
            StackSegment {
                y0: 2.0,
                y1: 7.0,
                value: 5.0
            }
        );
    }

    #[test]
    fn percent_normalizes_rows() {
        let cats = vec!["a".to_string()];
        let ser = vec!["s1".to_string(), "s2".to_string()];
        let r = compute_stack(
            &cats,
            &ser,
            |_, s| Some(if s == "s1" { 1.0 } else { 3.0 }),
            "stacked-percent",
            "diverging",
        );
        let a = &r["a"];
        assert_eq!(a["s1"].y1, 25.0);
        assert_eq!(a["s2"].y1, 100.0);
    }
}
