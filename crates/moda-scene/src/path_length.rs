//! Arc length of a path's anchors, for width markers (docs/roadmap/brushes.md
//! §4.2, "Arc length for markers"). The ONE implementation on the native side;
//! the TS mirror is `core/scene/pathArcLength.ts`, and the shared fixture
//! `fixtures/stroke-profile/remap.json` (`anchorFractions` cases) pins the two
//! bit-for-bit (tests/stroke_profile_parity.rs). The engine (P8) and the Width
//! tool (P11) both measure marker positions through it.
//!
//! Conventions: points are read like `cmds_from_node_points` reads them
//! (unknown types skipped, missing coordinates 0, a path without a leading `move` starts at its first drawing point,
//! `outgoingHandle` ignored); every subpath of a closed path is closed and the
//! closing segment counts; a `move` point's t is the start of its subpath;
//! zero-length segments contribute 0; quadratics are elevated to cubics, arcs
//! go through `arc_to_beziers` (degenerate → line), and cubics are measured by
//! adaptive subdivision until polygon − chord ≤ `ARC_LENGTH_TOLERANCE`. Only
//! `+ − × ÷` and `sqrt` in the TS order, so both lanes produce the same bits
//! (arcs also go through sin/cos/tan, which may differ by an ulp across libms).

use crate::path_geom::arc::{arc_to_beziers, ArcParams};
use crate::value::Value;

/// Control polygon minus chord, in path units (`ARC_LENGTH_TOLERANCE`).
pub const ARC_LENGTH_TOLERANCE: f64 = 1e-3;
/// Subdivision depth cap (`ARC_LENGTH_MAX_DEPTH`).
pub const ARC_LENGTH_MAX_DEPTH: u32 = 12;

type P = [f64; 2];

/// `SubpathExtent`: indices into `points` and fractions of the total length.
#[derive(Debug, Clone, PartialEq)]
pub struct SubpathExtent {
    pub start_index: usize,
    pub end_index: usize,
    pub start: f64,
    pub end: f64,
}

/// `AnchorFractionsResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorFractions {
    /// Arc-length fraction of each entry of `points`.
    pub anchor_t: Vec<f64>,
    pub subpaths: Vec<SubpathExtent>,
    pub total_length: f64,
}

impl AnchorFractions {
    /// `anchor_t` in the shape `resolve_width_markers` takes.
    pub fn as_options(&self) -> Vec<Option<f64>> {
        self.anchor_t.iter().copied().map(Some).collect()
    }

    /// `isPeriodicStroke`: a closed path with exactly one subpath, whose width
    /// profile wraps across the seam (`PreparedWidthProfile::new_with(.., periodic)`).
    /// Compound paths (markers over the total length) are never periodic.
    pub fn is_periodic(&self, closed: bool) -> bool {
        closed && self.subpaths.len() == 1
    }

    /// `subpathAt`: the extent of the subpath containing fraction `t`.
    pub fn subpath_at(&self, t: f64) -> Option<(f64, f64)> {
        self.subpaths
            .iter()
            .find(|s| t >= s.start && t <= s.end)
            .or(self.subpaths.last())
            .map(|s| (s.start, s.end))
    }
}

fn dist(a: P, b: P) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    (dx * dx + dy * dy).sqrt()
}

fn mid(a: P, b: P) -> P {
    [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
}

fn cubic_length(p0: P, p1: P, p2: P, p3: P, depth: u32) -> f64 {
    let chord = dist(p0, p3);
    let poly = dist(p0, p1) + dist(p1, p2) + dist(p2, p3);
    if poly - chord <= ARC_LENGTH_TOLERANCE || depth >= ARC_LENGTH_MAX_DEPTH {
        return (chord + poly) / 2.0;
    }
    let m01 = mid(p0, p1);
    let m12 = mid(p1, p2);
    let m23 = mid(p2, p3);
    let a = mid(m01, m12);
    let b = mid(m12, m23);
    let m = mid(a, b);
    cubic_length(p0, m01, a, m, depth + 1) + cubic_length(m, b, m23, p3, depth + 1)
}

/// `cubicArcLength`: adaptive arc length of one cubic Bézier.
pub fn cubic_arc_length(p0: P, p1: P, p2: P, p3: P) -> f64 {
    cubic_length(p0, p1, p2, p3, 0)
}

fn num(v: Option<&Value>) -> f64 {
    v.and_then(Value::as_f64).unwrap_or(0.0)
}

fn pt(v: Option<&Value>) -> P {
    match v.and_then(Value::as_object) {
        Some(o) => [num(o.get("x")), num(o.get("y"))],
        None => [0.0, 0.0],
    }
}

/// Length of the segment `point` draws from `from`, or None when skipped.
fn segment_length(from: P, o: &crate::value::Map) -> Option<(f64, P)> {
    let to = [num(o.get("x")), num(o.get("y"))];
    match o.get("type").and_then(Value::as_str)? {
        "line" => Some((dist(from, to), to)),
        "curve" => Some((
            cubic_arc_length(from, pt(o.get("cp1")), pt(o.get("cp2")), to),
            to,
        )),
        "quadratic" => {
            let cp = pt(o.get("cp"));
            let c1 = [
                from[0] + (2.0 / 3.0) * (cp[0] - from[0]),
                from[1] + (2.0 / 3.0) * (cp[1] - from[1]),
            ];
            let c2 = [
                to[0] + (2.0 / 3.0) * (cp[0] - to[0]),
                to[1] + (2.0 / 3.0) * (cp[1] - to[1]),
            ];
            Some((cubic_arc_length(from, c1, c2, to), to))
        }
        "arc" => {
            let arc = ArcParams {
                rx: num(o.get("rx")),
                ry: num(o.get("ry")),
                x_axis_rotation: num(o.get("xAxisRotation")),
                large_arc: o.get("largeArc").and_then(Value::as_bool) == Some(true),
                sweep: o.get("sweep").and_then(Value::as_bool) == Some(true),
                x: to[0],
                y: to[1],
            };
            let beziers = arc_to_beziers(from[0], from[1], &arc);
            if beziers.is_empty() {
                return Some((dist(from, to), to));
            }
            let mut length = 0.0;
            let mut start = from;
            for b in &beziers {
                length += cubic_arc_length(start, b.cp1, b.cp2, b.end);
                start = b.end;
            }
            Some((length, to))
        }
        _ => None,
    }
}

/// `anchorFractions(points, closed)`: arc-length fraction of every anchor
/// (index into `points`) and the extent of every subpath, for a path's LOCAL
/// points. `closed` closes every subpath, as the renderer does.
pub fn anchor_fractions(points: &[Value], closed: bool) -> AnchorFractions {
    let mut cumulative = vec![0.0f64; points.len()];
    let mut raw: Vec<(usize, usize, f64, f64)> = Vec::new();
    let mut total = 0.0f64;
    let mut pos: P = [0.0, 0.0];
    let mut sub_start: P = [0.0, 0.0];
    let mut current: Option<(usize, f64)> = None;

    let close = |current: &Option<(usize, f64)>,
                 end_index: usize,
                 total: &mut f64,
                 pos: P,
                 sub_start: P,
                 raw: &mut Vec<(usize, usize, f64, f64)>| {
        if let Some((start_index, start)) = *current {
            if closed {
                *total += dist(pos, sub_start);
            }
            raw.push((start_index, end_index, start, *total));
        }
    };

    for (i, point) in points.iter().enumerate() {
        let Some(o) = point.as_object() else {
            cumulative[i] = total;
            continue;
        };
        if o.get("type").and_then(Value::as_str) == Some("move") {
            close(
                &current,
                i.saturating_sub(1),
                &mut total,
                pos,
                sub_start,
                &mut raw,
            );
            pos = [num(o.get("x")), num(o.get("y"))];
            sub_start = pos;
            current = Some((i, total));
            cumulative[i] = total;
            continue;
        }
        if current.is_none() {
            current = Some((i, total));
            // A path that doesn't start with a move starts AT its first
            // drawing point (as the stroke outliner draws it).
            if matches!(
                o.get("type").and_then(Value::as_str),
                Some("line" | "curve" | "quadratic" | "arc")
            ) {
                pos = [num(o.get("x")), num(o.get("y"))];
                sub_start = pos;
                cumulative[i] = total;
                continue;
            }
        }
        if let Some((length, to)) = segment_length(pos, o) {
            total += length;
            pos = to;
        }
        cumulative[i] = total;
    }
    close(
        &current,
        points.len().saturating_sub(1),
        &mut total,
        pos,
        sub_start,
        &mut raw,
    );

    let frac = |v: f64| if total > 0.0 { v / total } else { 0.0 };
    AnchorFractions {
        anchor_t: cumulative.into_iter().map(frac).collect(),
        subpaths: raw
            .into_iter()
            .map(|(start_index, end_index, start, end)| SubpathExtent {
                start_index,
                end_index,
                start: frac(start),
                end: frac(end),
            })
            .collect(),
        total_length: total,
    }
}
