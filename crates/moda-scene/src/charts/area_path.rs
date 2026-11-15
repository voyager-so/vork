//! Port of `utils/markup/elements/charts/generators/areaPath.ts` — baseline
//! clipping (De Casteljau cubic splits + bisection root finding) and region
//! segmentation for overlay area charts.

use super::curves::{generate_curve_points, PathPt};

const BASELINE_EPS: f64 = 1e-6;

fn is_on_baseline(y: f64, baseline_y: f64) -> bool {
    (y - baseline_y).abs() < BASELINE_EPS
}

fn lerp(a: (f64, f64), b: (f64, f64), t: f64) -> (f64, f64) {
    ((1.0 - t) * a.0 + t * b.0, (1.0 - t) * a.1 + t * b.1)
}

#[derive(Clone, Copy)]
struct Cubic {
    p0: (f64, f64),
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
}

fn split_cubic_at_t(c: Cubic, t: f64) -> (Cubic, Cubic) {
    let q0 = lerp(c.p0, c.p1, t);
    let q1 = lerp(c.p1, c.p2, t);
    let q2 = lerp(c.p2, c.p3, t);
    let r0 = lerp(q0, q1, t);
    let r1 = lerp(q1, q2, t);
    let s = lerp(r0, r1, t);
    (
        Cubic {
            p0: c.p0,
            p1: q0,
            p2: r0,
            p3: s,
        },
        Cubic {
            p0: s,
            p1: r1,
            p2: q2,
            p3: c.p3,
        },
    )
}

fn cubic_y(y0: f64, y1: f64, y2: f64, y3: f64, t: f64) -> f64 {
    let u = 1.0 - t;
    u * u * u * y0 + 3.0 * u * u * t * y1 + 3.0 * u * t * t * y2 + t * t * t * y3
}

/// `findCubicBaselineCrossings` (areaPath.ts:60-137).
pub fn find_cubic_baseline_crossings(
    y0: f64,
    y1: f64,
    y2: f64,
    y3: f64,
    baseline: f64,
) -> Vec<f64> {
    const BISECT_ITERS: usize = 60;
    const POLY_EPS: f64 = 1e-12;
    let delta = |t: f64| cubic_y(y0, y1, y2, y3, t) - baseline;

    let mut partition = vec![0.0f64];
    let a = y3 - 3.0 * y2 + 3.0 * y1 - y0;
    let b = 2.0 * y0 - 4.0 * y1 + 2.0 * y2;
    let c = y1 - y0;
    if a.abs() < POLY_EPS {
        if b.abs() > POLY_EPS {
            let t = -c / b;
            if t > 0.0 && t < 1.0 {
                partition.push(t);
            }
        }
    } else {
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            let inv2a = 1.0 / (2.0 * a);
            let ta = (-b - sq) * inv2a;
            let tb = (-b + sq) * inv2a;
            if ta > 0.0 && ta < 1.0 {
                partition.push(ta);
            }
            if tb > 0.0 && tb < 1.0 && (tb - ta).abs() > POLY_EPS {
                partition.push(tb);
            }
        }
    }
    partition.push(1.0);
    partition.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    let mut roots = Vec::new();
    for i in 1..partition.len() {
        let ta = partition[i - 1];
        let tb = partition[i];
        let ya = delta(ta);
        let yb = delta(tb);
        if ya != 0.0 && yb != 0.0 && ya * yb < 0.0 {
            let mut lo = ta;
            let mut hi = tb;
            let mut lo_val = ya;
            for _ in 0..BISECT_ITERS {
                let mid = (lo + hi) / 2.0;
                let mid_val = delta(mid);
                if mid_val == 0.0 {
                    lo = mid;
                    hi = mid;
                    break;
                }
                if lo_val * mid_val < 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    lo_val = mid_val;
                }
            }
            let root = (lo + hi) / 2.0;
            if root > BASELINE_EPS && root < 1.0 - BASELINE_EPS {
                roots.push(root);
            }
        } else if yb == 0.0 && tb > BASELINE_EPS && tb < 1.0 - BASELINE_EPS {
            roots.push(tb);
        }
    }
    roots
}

/// `clipCurveAtBaseline` (areaPath.ts:147-233).
fn clip_curve_at_baseline(points: &[PathPt], baseline_y: f64) -> Vec<PathPt> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut result: Vec<PathPt> = Vec::new();
    let mut cursor = (0.0f64, 0.0f64);
    for p in points {
        match *p {
            PathPt::Move { x, y } => {
                result.push(*p);
                cursor = (x, y);
            }
            PathPt::Line { x, y } => {
                let from_delta = cursor.1 - baseline_y;
                let to_delta = y - baseline_y;
                if from_delta * to_delta < 0.0 {
                    let t = from_delta / (from_delta - to_delta);
                    result.push(PathPt::Line {
                        x: cursor.0 + t * (x - cursor.0),
                        y: baseline_y,
                    });
                }
                result.push(*p);
                cursor = (x, y);
            }
            PathPt::Curve { x, y, cp1, cp2 } => {
                let crossings =
                    find_cubic_baseline_crossings(cursor.1, cp1.1, cp2.1, y, baseline_y);
                if crossings.is_empty() {
                    result.push(*p);
                    cursor = (x, y);
                    continue;
                }
                let mut remaining = Cubic {
                    p0: cursor,
                    p1: cp1,
                    p2: cp2,
                    p3: (x, y),
                };
                let mut consumed_t = 0.0;
                for &crossing_t in &crossings {
                    let t_prime = (crossing_t - consumed_t) / (1.0 - consumed_t);
                    let (left, right) = split_cubic_at_t(remaining, t_prime);
                    result.push(PathPt::Curve {
                        x: left.p3.0,
                        y: baseline_y, // snap
                        cp1: left.p1,
                        cp2: left.p2,
                    });
                    remaining = right;
                    consumed_t = crossing_t;
                }
                result.push(PathPt::Curve {
                    x: remaining.p3.0,
                    y: remaining.p3.1,
                    cp1: remaining.p1,
                    cp2: remaining.p2,
                });
                cursor = (x, y);
            }
        }
    }
    result
}

#[derive(Clone, Debug)]
pub struct AreaRegion {
    pub top_edge: Vec<PathPt>,
    pub subpath: Vec<PathPt>,
}

/// `generateAreaRegions` (areaPath.ts:276-296).
pub fn generate_area_regions(
    data_points: &[(f64, f64)],
    baseline_y: f64,
    curve: &str,
) -> Vec<AreaRegion> {
    if data_points.is_empty() {
        return Vec::new();
    }
    if data_points.len() == 1 {
        let top_edge = generate_curve_points(data_points, curve);
        if top_edge.is_empty() {
            return Vec::new();
        }
        let only = data_points[0];
        let mut subpath = top_edge.clone();
        subpath.push(PathPt::Line {
            x: only.0,
            y: baseline_y,
        });
        subpath.push(PathPt::Line {
            x: only.0,
            y: baseline_y,
        });
        return vec![AreaRegion { top_edge, subpath }];
    }
    let full_curve = generate_curve_points(data_points, curve);
    if full_curve.is_empty() {
        return Vec::new();
    }
    let clipped = clip_curve_at_baseline(&full_curve, baseline_y);
    segment_clipped_into_regions(&clipped, baseline_y)
}

/// `concatAreaTopEdges` (areaPath.ts:316-327).
pub fn concat_area_top_edges(regions: &[AreaRegion]) -> Vec<PathPt> {
    if regions.is_empty() {
        return Vec::new();
    }
    let mut result = regions[0].top_edge.clone();
    for region in &regions[1..] {
        let edge = &region.top_edge;
        let start = if matches!(edge.first(), Some(PathPt::Move { .. })) {
            1
        } else {
            0
        };
        result.extend(edge[start..].iter().cloned());
    }
    result
}

fn close_subpath(top_edge: &[PathPt], baseline_y: f64) -> Vec<PathPt> {
    let mut first_x = None;
    let mut last_x = None;
    for p in top_edge {
        let x = p.x();
        if first_x.is_none() {
            first_x = Some(x);
        }
        last_x = Some(x);
    }
    let (Some(first_x), Some(last_x)) = (first_x, last_x) else {
        return top_edge.to_vec();
    };
    let mut out = top_edge.to_vec();
    out.push(PathPt::Line {
        x: last_x,
        y: baseline_y,
    });
    out.push(PathPt::Line {
        x: first_x,
        y: baseline_y,
    });
    out
}

fn segment_midpoint_side(from: (f64, f64), to: &PathPt, baseline_y: f64) -> i8 {
    let mid_y = match *to {
        PathPt::Curve { y, cp1, cp2, .. } => cubic_y(from.1, cp1.1, cp2.1, y, 0.5),
        PathPt::Line { y, .. } | PathPt::Move { y, .. } => (from.1 + y) / 2.0,
    };
    if (mid_y - baseline_y).abs() < BASELINE_EPS {
        0
    } else if mid_y > baseline_y {
        1
    } else {
        -1
    }
}

fn is_on_baseline_anchor(p: &PathPt, baseline_y: f64) -> bool {
    is_on_baseline(p.y(), baseline_y)
}

/// `segmentClippedIntoRegions` (areaPath.ts:406-479).
fn segment_clipped_into_regions(clipped: &[PathPt], baseline_y: f64) -> Vec<AreaRegion> {
    if clipped.is_empty() {
        return Vec::new();
    }
    let mut regions: Vec<AreaRegion> = Vec::new();
    let mut current: Vec<PathPt> = vec![clipped[0].clone()];
    let mut current_side: i8 = 0;
    let mut last_baseline_idx: i64 = if is_on_baseline_anchor(&clipped[0], baseline_y) {
        0
    } else {
        -1
    };
    let mut prev_anchor = (clipped[0].x(), clipped[0].y());

    fn commit(current: &[PathPt], baseline_y: f64, regions: &mut Vec<AreaRegion>) {
        if current.len() < 2 {
            return;
        }
        let mut prev: Option<(f64, f64)> = None;
        let mut has_interior = false;
        for p in current {
            let a = (p.x(), p.y());
            if !is_on_baseline(a.1, baseline_y) {
                has_interior = true;
                break;
            }
            if let Some(prev) = prev {
                if segment_midpoint_side(prev, p, baseline_y) != 0 {
                    has_interior = true;
                    break;
                }
            }
            prev = Some(a);
        }
        if !has_interior {
            return;
        }
        regions.push(AreaRegion {
            top_edge: current.to_vec(),
            subpath: close_subpath(current, baseline_y),
        });
    }

    for pt in &clipped[1..] {
        let seg_side = segment_midpoint_side(prev_anchor, pt, baseline_y);
        if current_side == 0 && seg_side != 0 {
            current_side = seg_side;
        } else if current_side != 0 && seg_side != 0 && seg_side != current_side {
            if last_baseline_idx >= 0 && (last_baseline_idx as usize) < current.len() {
                let idx = last_baseline_idx as usize;
                let closing: Vec<PathPt> = current[..=idx].to_vec();
                let remaining: Vec<PathPt> = current[idx..].to_vec();
                commit(&closing, baseline_y, &mut regions);
                let seed = (remaining[0].x(), remaining[0].y());
                let mut next: Vec<PathPt> = vec![PathPt::Move {
                    x: seed.0,
                    y: baseline_y,
                }];
                next.extend(remaining[1..].iter().cloned());
                current = next;
                last_baseline_idx = -1;
            }
            current_side = seg_side;
        }
        current.push(pt.clone());
        if is_on_baseline_anchor(pt, baseline_y) {
            last_baseline_idx = current.len() as i64 - 1;
        }
        prev_anchor = (pt.x(), pt.y());
    }
    commit(&current, baseline_y, &mut regions);

    if regions.is_empty() && clipped.len() >= 2 {
        regions.push(AreaRegion {
            top_edge: clipped.to_vec(),
            subpath: close_subpath(clipped, baseline_y),
        });
    }
    regions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_region_above_baseline() {
        let regions =
            generate_area_regions(&[(0.0, 10.0), (10.0, 20.0), (20.0, 5.0)], 100.0, "linear");
        assert_eq!(regions.len(), 1);
        let sub = &regions[0].subpath;
        // top edge (3 pts) + 2 baseline closure lines
        assert_eq!(sub.len(), 5);
        assert_eq!(sub[3], PathPt::Line { x: 20.0, y: 100.0 });
        assert_eq!(sub[4], PathPt::Line { x: 0.0, y: 100.0 });
    }

    #[test]
    fn crossing_splits_regions() {
        // crosses baseline 50 between x=0 (y=0) and x=10 (y=100)
        let regions =
            generate_area_regions(&[(0.0, 0.0), (10.0, 100.0), (20.0, 0.0)], 50.0, "linear");
        assert_eq!(regions.len(), 3);
    }
}
