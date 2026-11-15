//! M2 — faithful port of the arc-length / dash infrastructure from
//! `utils/lineToPath.ts` (line citations are to that file):
//!   - `scaleDashPattern` (:166, BASE_DASH_STROKE_WIDTH = 2)
//!   - `PathSegment` + Gauss-Legendre arc lengths (:180-315)
//!   - `findParameterAtArcLength{Quadratic,Cubic}` Newton iterations (:350-464)
//!   - `walkPathByArcLength` (:623-699) + `getPointOnSegment` (:708-737)
//!   - `buildElbowedPathSegments` (:501-610)
//!
//! All math in f64 like the JS engine. COPY THE BUGS — thresholds (0.001),
//! iteration counts (8 / 10), and the 5-point Gauss-Legendre constants are
//! part of the pixel contract.

pub type Pt = [f64; 2];

#[derive(Debug, Clone)]
pub enum Seg {
    Line { p0: Pt, p1: Pt },
    Quadratic { p0: Pt, p1: Pt, cp: Pt },
    Cubic { p0: Pt, p1: Pt, cp1: Pt, cp2: Pt },
}

#[derive(Debug, Clone)]
pub struct PathSegment {
    pub seg: Seg,
    pub arc_length: f64,
}

/// `BASE_DASH_STROKE_WIDTH` (:160).
pub const BASE_DASH_STROKE_WIDTH: f64 = 2.0;

/// `scaleDashPattern` (:166-169).
pub fn scale_dash_pattern(dash_pattern: &[f64], stroke_width: f64) -> Vec<f64> {
    let scale = stroke_width / BASE_DASH_STROKE_WIDTH;
    dash_pattern.iter().map(|v| v * scale).collect()
}

/// Odd-length patterns are doubled (SVG semantics; tessellator :235 and
/// lineToPath :1044).
pub fn normalize_dash_pattern(scaled: Vec<f64>) -> Vec<f64> {
    if scaled.len() % 2 == 0 {
        scaled
    } else {
        let mut out = scaled.clone();
        out.extend(scaled);
        out
    }
}

/// Gauss-Legendre 5-point abscissae/weights mapped to [0,1] (:212-229),
/// written EXACTLY as the TS source expressions evaluate.
const GL5_T: [f64; 5] = [
    0.5 - 0.5 * 0.9061798459386639,
    0.5 - 0.5 * 0.5384693101056831,
    0.5,
    0.5 + 0.5 * 0.5384693101056831,
    0.5 + 0.5 * 0.9061798459386639,
];
const GL5_W: [f64; 5] = [
    0.5 * 0.2369268850561891,
    0.5 * 0.4786286704993665,
    0.5 * 0.5688888888888889,
    0.5 * 0.4786286704993665,
    0.5 * 0.2369268850561891,
];

/// `computeQuadraticBezierArcLength` (:243-272).
pub fn quadratic_arc_length(p0: Pt, cp: Pt, p1: Pt) -> f64 {
    let ax = cp[0] - p0[0];
    let ay = cp[1] - p0[1];
    let bx = p1[0] - 2.0 * cp[0] + p0[0];
    let by = p1[1] - 2.0 * cp[1] + p0[1];

    let mut length = 0.0;
    for i in 0..5 {
        let t = GL5_T[i];
        let w = GL5_W[i];
        let dx = 2.0 * (ax + t * bx);
        let dy = 2.0 * (ay + t * by);
        length += w * (dx * dx + dy * dy).sqrt();
    }
    length
}

/// `computeCubicBezierArcLength` (:286-315).
pub fn cubic_arc_length(p0: Pt, cp1: Pt, cp2: Pt, p1: Pt) -> f64 {
    let ax = 3.0 * (cp1[0] - p0[0]);
    let ay = 3.0 * (cp1[1] - p0[1]);
    let bx = 3.0 * (cp2[0] - cp1[0]);
    let by = 3.0 * (cp2[1] - cp1[1]);
    let cx = 3.0 * (p1[0] - cp2[0]);
    let cy = 3.0 * (p1[1] - cp2[1]);

    let mut length = 0.0;
    for i in 0..5 {
        let t = GL5_T[i];
        let w = GL5_W[i];
        let mt = 1.0 - t;
        let dx = mt * mt * ax + 2.0 * mt * t * bx + t * t * cx;
        let dy = mt * mt * ay + 2.0 * mt * t * by + t * t * cy;
        length += w * (dx * dx + dy * dy).sqrt();
    }
    length
}

/// `quadraticBezierAt` (:326-337).
pub fn quadratic_at(p0: Pt, cp: Pt, p1: Pt, t: f64) -> Pt {
    let mt = 1.0 - t;
    [
        mt * mt * p0[0] + 2.0 * mt * t * cp[0] + t * t * p1[0],
        mt * mt * p0[1] + 2.0 * mt * t * cp[1] + t * t * p1[1],
    ]
}

/// `cubicBezierAt` (:476-492).
pub fn cubic_at(p0: Pt, cp1: Pt, cp2: Pt, p1: Pt, t: f64) -> Pt {
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;
    let t2 = t * t;
    let t3 = t2 * t;
    [
        mt3 * p0[0] + 3.0 * mt2 * t * cp1[0] + 3.0 * mt * t2 * cp2[0] + t3 * p1[0],
        mt3 * p0[1] + 3.0 * mt2 * t * cp1[1] + 3.0 * mt * t2 * cp2[1] + t3 * p1[1],
    ]
}

/// `findParameterAtArcLengthQuadratic` (:350-397) — Newton-Raphson, 8 iters.
pub fn find_t_at_arc_length_quadratic(
    p0: Pt,
    cp: Pt,
    p1: Pt,
    target_length: f64,
    total_length: f64,
) -> f64 {
    if target_length <= 0.0 {
        return 0.0;
    }
    if target_length >= total_length {
        return 1.0;
    }

    let mut t = target_length / total_length;

    let ax = cp[0] - p0[0];
    let ay = cp[1] - p0[1];
    let bx = p1[0] - 2.0 * cp[0] + p0[0];
    let by = p1[1] - 2.0 * cp[1] + p0[1];

    for _ in 0..8 {
        let mut current_length = 0.0;
        for i in 0..5 {
            let ti = t * GL5_T[i];
            let wi = t * GL5_W[i];
            let dx = 2.0 * (ax + ti * bx);
            let dy = 2.0 * (ay + ti * by);
            current_length += wi * (dx * dx + dy * dy).sqrt();
        }

        let error = current_length - target_length;
        if error.abs() < 0.001 {
            break;
        }

        let dx = 2.0 * (ax + t * bx);
        let dy = 2.0 * (ay + t * by);
        let speed = (dx * dx + dy * dy).sqrt();
        if speed < 0.001 {
            break;
        }

        t -= error / speed;
        t = t.clamp(0.0, 1.0);
    }

    t
}

/// `findParameterAtArcLengthCubic` (:411-464) — Newton-Raphson, 10 iters.
pub fn find_t_at_arc_length_cubic(
    p0: Pt,
    cp1: Pt,
    cp2: Pt,
    p1: Pt,
    target_length: f64,
    total_length: f64,
) -> f64 {
    if target_length <= 0.0 {
        return 0.0;
    }
    if target_length >= total_length {
        return 1.0;
    }

    let mut t = target_length / total_length;

    let ax = 3.0 * (cp1[0] - p0[0]);
    let ay = 3.0 * (cp1[1] - p0[1]);
    let bx = 3.0 * (cp2[0] - cp1[0]);
    let by = 3.0 * (cp2[1] - cp1[1]);
    let cx = 3.0 * (p1[0] - cp2[0]);
    let cy = 3.0 * (p1[1] - cp2[1]);

    for _ in 0..10 {
        let mut current_length = 0.0;
        for i in 0..5 {
            let ti = t * GL5_T[i];
            let wi = t * GL5_W[i];
            let mti = 1.0 - ti;
            let dx = mti * mti * ax + 2.0 * mti * ti * bx + ti * ti * cx;
            let dy = mti * mti * ay + 2.0 * mti * ti * by + ti * ti * cy;
            current_length += wi * (dx * dx + dy * dy).sqrt();
        }

        let error = current_length - target_length;
        if error.abs() < 0.001 {
            break;
        }

        let mt = 1.0 - t;
        let dx = mt * mt * ax + 2.0 * mt * t * bx + t * t * cx;
        let dy = mt * mt * ay + 2.0 * mt * t * by + t * t * cy;
        let speed = (dx * dx + dy * dy).sqrt();
        if speed < 0.001 {
            break;
        }

        t -= error / speed;
        t = t.clamp(0.0, 1.0);
    }

    t
}

/// `getPointOnSegment` (:708-737).
pub fn point_on_segment(segment: &PathSegment, distance: f64) -> Pt {
    let segment_length = segment.arc_length;

    let (p0, p1) = match &segment.seg {
        Seg::Line { p0, p1 } => (*p0, *p1),
        Seg::Quadratic { p0, p1, .. } => (*p0, *p1),
        Seg::Cubic { p0, p1, .. } => (*p0, *p1),
    };

    if segment_length < 0.001 || distance <= 0.0 {
        return p0;
    }
    if distance >= segment_length {
        return p1;
    }

    match &segment.seg {
        Seg::Line { p0, p1 } => {
            let t = distance / segment_length;
            [p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])]
        }
        Seg::Quadratic { p0, p1, cp } => {
            let t = find_t_at_arc_length_quadratic(*p0, *cp, *p1, distance, segment_length);
            quadratic_at(*p0, *cp, *p1, t)
        }
        Seg::Cubic { p0, p1, cp1, cp2 } => {
            let t = find_t_at_arc_length_cubic(*p0, *cp1, *cp2, *p1, distance, segment_length);
            cubic_at(*p0, *cp1, *cp2, *p1, t)
        }
    }
}

/// Output command of the arc-length walk: `M` starts a dash, `L` continues it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WalkCmd {
    Move(Pt),
    Line(Pt),
}

/// `walkPathByArcLength` (:623-699). `dash_pattern` must already be scaled
/// and even-length-normalized.
pub fn walk_path_by_arc_length(segments: &[PathSegment], dash_pattern: &[f64]) -> Vec<WalkCmd> {
    let mut points: Vec<WalkCmd> = Vec::new();

    if segments.is_empty() || dash_pattern.is_empty() {
        return points;
    }

    let pattern_length: f64 = dash_pattern.iter().sum();
    if pattern_length == 0.0 {
        return points;
    }

    let total_length: f64 = segments.iter().map(|s| s.arc_length).sum();
    if total_length < 0.001 {
        return points;
    }

    let mut pattern_index = 0usize;
    let mut is_dash = true;
    let mut remaining_in_pattern_segment = dash_pattern[0];

    let mut current_segment_index = 0usize;
    let mut distance_into_current_segment = 0.0f64;

    let mut in_dash = false;

    while current_segment_index < segments.len() {
        let segment = &segments[current_segment_index];
        let segment_length = segment.arc_length;
        let remaining_in_segment = segment_length - distance_into_current_segment;

        if remaining_in_segment < 0.001 {
            current_segment_index += 1;
            distance_into_current_segment = 0.0;
            continue;
        }

        let step_length = remaining_in_pattern_segment.min(remaining_in_segment);

        if is_dash && step_length > 0.001 {
            let start_dist = distance_into_current_segment;
            let end_dist = distance_into_current_segment + step_length;

            let start_point = point_on_segment(segment, start_dist);
            let end_point = point_on_segment(segment, end_dist);

            if !in_dash {
                points.push(WalkCmd::Move(start_point));
                in_dash = true;
            }
            points.push(WalkCmd::Line(end_point));
        }

        distance_into_current_segment += step_length;
        remaining_in_pattern_segment -= step_length;

        if remaining_in_pattern_segment < 0.001 {
            pattern_index = (pattern_index + 1) % dash_pattern.len();
            remaining_in_pattern_segment = dash_pattern[pattern_index];
            if is_dash {
                in_dash = false;
            }
            is_dash = !is_dash;
        }
    }

    points
}

/// Seed the dash walk's phase from an offset DISTANCE into an
/// already-scaled+normalized dash pattern (marching-ants / `stroke-dashoffset`).
/// Returns `(pattern_index, is_dash, remaining_in_pattern_segment)`. Offset wraps
/// modulo the pattern length and may be negative (flow direction). At `offset == 0`
/// this returns `(0, true, pattern[0])` — the un-phased walk's init. (For a
/// degenerate leading `0.0` element the two could differ at the first step, but
/// `offset == 0` never routes through the phased walk — callers use the un-phased
/// walk there — so the phased path is only taken for `offset != 0`.)
pub(crate) fn seed_dash_phase(dash_pattern: &[f64], offset: f64) -> (usize, bool, f64) {
    let pattern_length: f64 = dash_pattern.iter().sum();
    if pattern_length <= 0.0 {
        return (0, true, dash_pattern.first().copied().unwrap_or(0.0));
    }
    let mut o = offset.rem_euclid(pattern_length); // normalized into [0, pattern_length)
    let mut index = 0usize;
    let mut is_dash = true;
    // The offset lands within one element after at most `len` steps (the elements
    // sum to pattern_length and o < pattern_length).
    for _ in 0..dash_pattern.len() {
        let seg = dash_pattern[index];
        if o < seg {
            return (index, is_dash, seg - o);
        }
        o -= seg;
        index = (index + 1) % dash_pattern.len();
        is_dash = !is_dash;
    }
    // Rounding put `o` exactly at the wrap — treat as phase 0.
    (0, true, dash_pattern[0])
}

/// Phased variant of [`walk_path_by_arc_length`] — starts the dash pattern at an
/// arc-length `offset` into the pattern (`dashOffset`, marching ants). Byte-identical
/// to the un-phased walk at `offset == 0`; callers use the un-phased walk for the
/// no-offset case so oracle-pinned output is untouched.
pub fn walk_path_by_arc_length_phased(
    segments: &[PathSegment],
    dash_pattern: &[f64],
    offset: f64,
) -> Vec<WalkCmd> {
    let mut points: Vec<WalkCmd> = Vec::new();

    if segments.is_empty() || dash_pattern.is_empty() {
        return points;
    }
    let pattern_length: f64 = dash_pattern.iter().sum();
    if pattern_length == 0.0 {
        return points;
    }
    let total_length: f64 = segments.iter().map(|s| s.arc_length).sum();
    if total_length < 0.001 {
        return points;
    }

    let (mut pattern_index, mut is_dash, mut remaining_in_pattern_segment) =
        seed_dash_phase(dash_pattern, offset);

    let mut current_segment_index = 0usize;
    let mut distance_into_current_segment = 0.0f64;
    let mut in_dash = false;

    while current_segment_index < segments.len() {
        let segment = &segments[current_segment_index];
        let segment_length = segment.arc_length;
        let remaining_in_segment = segment_length - distance_into_current_segment;

        if remaining_in_segment < 0.001 {
            current_segment_index += 1;
            distance_into_current_segment = 0.0;
            continue;
        }

        let step_length = remaining_in_pattern_segment.min(remaining_in_segment);

        if is_dash && step_length > 0.001 {
            let start_dist = distance_into_current_segment;
            let end_dist = distance_into_current_segment + step_length;
            let start_point = point_on_segment(segment, start_dist);
            let end_point = point_on_segment(segment, end_dist);
            if !in_dash {
                points.push(WalkCmd::Move(start_point));
                in_dash = true;
            }
            points.push(WalkCmd::Line(end_point));
        }

        distance_into_current_segment += step_length;
        remaining_in_pattern_segment -= step_length;

        if remaining_in_pattern_segment < 0.001 {
            pattern_index = (pattern_index + 1) % dash_pattern.len();
            remaining_in_pattern_segment = dash_pattern[pattern_index];
            if is_dash {
                in_dash = false;
            }
            is_dash = !is_dash;
        }
    }

    points
}

// ---------------------------------------------------------------------------
// Aligned dashes — Illustrator's "Align dashes to corners and path ends,
// adjusting lengths to fit" (a stroke's `dashAlignment: "corners"`).
// ---------------------------------------------------------------------------

/// Two tangents closer than this (the cosine of ~2°) meet smoothly: no corner.
const CORNER_COS: f64 = 0.9994;

fn unit(v: Pt) -> Option<Pt> {
    let l = v[0].hypot(v[1]);
    (l > 1e-9).then(|| [v[0] / l, v[1] / l])
}

fn sub(a: Pt, b: Pt) -> Pt {
    [a[0] - b[0], a[1] - b[1]]
}

/// A segment's direction leaving its start (the first control point off the
/// start, else the chord).
fn start_tangent(seg: &Seg) -> Option<Pt> {
    match seg {
        Seg::Line { p0, p1 } => unit(sub(*p1, *p0)),
        Seg::Quadratic { p0, p1, cp } => unit(sub(*cp, *p0)).or_else(|| unit(sub(*p1, *p0))),
        Seg::Cubic { p0, p1, cp1, cp2 } => unit(sub(*cp1, *p0))
            .or_else(|| unit(sub(*cp2, *p0)))
            .or_else(|| unit(sub(*p1, *p0))),
    }
}

/// A segment's direction arriving at its end.
fn end_tangent(seg: &Seg) -> Option<Pt> {
    match seg {
        Seg::Line { p0, p1 } => unit(sub(*p1, *p0)),
        Seg::Quadratic { p0, p1, cp } => unit(sub(*p1, *cp)).or_else(|| unit(sub(*p1, *p0))),
        Seg::Cubic { p0, p1, cp1, cp2 } => unit(sub(*p1, *cp2))
            .or_else(|| unit(sub(*p1, *cp1)))
            .or_else(|| unit(sub(*p1, *p0))),
    }
}

/// Arc-length positions of one sub-path's CORNERS: the vertices where the
/// direction turns (a smooth anchor is not one). `segments` are the sub-path's
/// own, unflattened (a flattened curve turns at every piece); a closed
/// sub-path's closing edge is one of them, and its start is a corner (at 0)
/// when the closing edge meets the first one at an angle. Degenerate
/// (zero-length) segments are skipped over.
pub fn segment_corner_positions(segments: &[PathSegment], closed: bool) -> Vec<f64> {
    let mut starts = Vec::with_capacity(segments.len());
    let mut acc = 0.0;
    for s in segments {
        starts.push(acc);
        acc += s.arc_length;
    }
    let live: Vec<usize> = (0..segments.len())
        .filter(|&i| segments[i].arc_length > 1e-6)
        .collect();
    let turns = |a: usize, b: usize| match (end_tangent(&segments[a].seg), start_tangent(&segments[b].seg)) {
        (Some(u), Some(v)) => u[0] * v[0] + u[1] * v[1] < CORNER_COS,
        _ => false,
    };
    let mut out = Vec::new();
    if closed && live.len() >= 2 && turns(live[live.len() - 1], live[0]) {
        out.push(starts[live[0]]);
    }
    for w in live.windows(2) {
        if turns(w[0], w[1]) {
            out.push(starts[w[1]]);
        }
    }
    out
}

/// Dash intervals (arc-length `[a, b]`, sorted, merged) for one sub-path of
/// length `total` whose corners sit at `corners` (from
/// [`segment_corner_positions`]), under Illustrator's aligned dashes: the
/// sub-path is cut into spans at its corners (and, open, at its two ends); on
/// each span the pattern (already scaled and even-length) is stretched or
/// squeezed to a whole number of repeats so that a dash is CENTRED on both of
/// the span's ends. So every corner and every path end is covered by a dash,
/// the halves either side of a corner making one dash that turns it. A closed
/// sub-path without corners is one span from its start round to its start.
pub fn aligned_dash_intervals(
    total: f64,
    corners: &[f64],
    closed: bool,
    pattern: &[f64],
) -> Vec<(f64, f64)> {
    let period: f64 = pattern.iter().sum();
    if pattern.is_empty() || !(period > 0.0) || !(total > 1e-3) {
        return Vec::new();
    }
    let mut cuts: Vec<f64> = corners
        .iter()
        .copied()
        .filter(|c| c.is_finite() && *c >= 0.0 && *c < total)
        .collect();
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    let spans: Vec<(f64, f64)> = if closed {
        if cuts.is_empty() {
            vec![(0.0, total)]
        } else {
            let mut v: Vec<(f64, f64)> = cuts.windows(2).map(|w| (w[0], w[1])).collect();
            v.push((cuts[cuts.len() - 1], cuts[0] + total));
            v
        }
    } else {
        let mut bounds = vec![0.0];
        bounds.extend(cuts.iter().copied().filter(|c| *c > 1e-6));
        bounds.push(total);
        bounds.windows(2).map(|w| (w[0], w[1])).collect()
    };
    let mut out: Vec<(f64, f64)> = Vec::new();
    // Emit on [0, total), splitting what wraps past the seam of a closed path.
    let mut push = |a: f64, b: f64| {
        if b - a <= 1e-9 {
            return;
        }
        if closed && b > total {
            if a < total {
                out.push((a, total));
            }
            out.push(((a - total).max(0.0), b - total));
        } else {
            out.push((a, b.min(total)));
        }
    };
    for (s, e) in spans {
        let len = e - s;
        if len <= 1e-6 {
            continue;
        }
        let repeats = (len / period).round().max(1.0);
        let k = len / (repeats * period);
        // The pattern starts half its first dash before the span, so that dash is centred on `s`; `repeats` whole
        // periods later its twin is centred on `e`.
        let mut pos = s - pattern[0] * k / 2.0;
        let mut j = 0usize;
        let mut guard = 0usize;
        while pos < e - 1e-9 && guard < 1_000_000 {
            guard += 1;
            let step = pattern[j % pattern.len()] * k;
            if j % 2 == 0 {
                push(pos.max(s), (pos + step).min(e));
            }
            pos += step;
            j += 1;
        }
    }
    out.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(out.len());
    for (a, b) in out {
        match merged.last_mut() {
            Some(last) if a <= last.1 + 1e-6 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    merged
}

/// Dash intervals as an on/off pattern for [`walk_path_by_arc_length`] over a
/// sub-path of length `total`: it starts "on" (a zero-length dash when the
/// first interval starts later) and ends with a gap longer than what is left,
/// so the walk never wraps round into a second repeat.
pub fn intervals_as_walk_pattern(intervals: &[(f64, f64)], total: f64) -> Vec<f64> {
    if intervals.is_empty() {
        return vec![0.0, total + 1.0];
    }
    let mut out = Vec::with_capacity(intervals.len() * 2 + 2);
    let mut cursor = 0.0;
    for (i, &(a, b)) in intervals.iter().enumerate() {
        if i == 0 && a > 0.0 {
            out.push(0.0);
            out.push(a);
        } else if i > 0 {
            out.push(a - cursor);
        }
        out.push(b - a);
        cursor = b;
    }
    out.push((total - cursor).max(0.0) + total + 1.0);
    out
}

#[cfg(test)]
#[path = "dash_aligned_tests.rs"]
mod aligned_dash_tests;

/// `buildElbowedPathSegments` (:501-610) — line segments + quadratic corner
/// arcs for an elbowed polyline. (`polylineSegmentIndex` is dropped: no M2
/// consumer needs it.)
pub fn build_elbowed_path_segments(all_points: &[Pt], corner_radius: f64) -> Vec<PathSegment> {
    let mut segments: Vec<PathSegment> = Vec::new();
    if all_points.len() < 2 {
        return segments;
    }

    for i in 0..all_points.len() - 1 {
        let curr = all_points[i];
        let next = all_points[i + 1];
        let after_next = if i < all_points.len() - 2 {
            Some(all_points[i + 2])
        } else {
            None
        };

        let seg_dx = next[0] - curr[0];
        let seg_dy = next[1] - curr[1];
        let seg_length = (seg_dx * seg_dx + seg_dy * seg_dy).sqrt();

        if seg_length < 0.001 {
            continue;
        }

        let unit_x = seg_dx / seg_length;
        let unit_y = seg_dy / seg_length;

        // Effective start (previous corner trim).
        let mut effective_start = curr;
        if i > 0 && corner_radius > 0.0 {
            let prev = all_points[i - 1];
            let prev_dx = curr[0] - prev[0];
            let prev_dy = curr[1] - prev[1];
            let prev_length = (prev_dx * prev_dx + prev_dy * prev_dy).sqrt();

            let max_radius = prev_length.min(seg_length) / 2.0;
            let radius_used = corner_radius.min(max_radius);

            if radius_used > 0.001 {
                effective_start = [
                    curr[0] + unit_x * radius_used,
                    curr[1] + unit_y * radius_used,
                ];
            }
        }

        // Effective end + corner radius used.
        let mut effective_end = next;
        let mut corner_radius_used = 0.0;

        if corner_radius > 0.0 {
            if let Some(after) = after_next {
                let next_dx = after[0] - next[0];
                let next_dy = after[1] - next[1];
                let next_length = (next_dx * next_dx + next_dy * next_dy).sqrt();

                let max_radius = seg_length.min(next_length) / 2.0;
                corner_radius_used = corner_radius.min(max_radius);

                if corner_radius_used > 0.001 {
                    effective_end = [
                        next[0] - unit_x * corner_radius_used,
                        next[1] - unit_y * corner_radius_used,
                    ];
                }
            }
        }

        let line_length = ((effective_end[0] - effective_start[0]).powi(2)
            + (effective_end[1] - effective_start[1]).powi(2))
        .sqrt();

        if line_length > 0.001 {
            segments.push(PathSegment {
                seg: Seg::Line {
                    p0: effective_start,
                    p1: effective_end,
                },
                arc_length: line_length,
            });
        }

        if corner_radius_used > 0.001 {
            if let Some(after) = after_next {
                let next_dx = after[0] - next[0];
                let next_dy = after[1] - next[1];
                let next_length = (next_dx * next_dx + next_dy * next_dy).sqrt();
                let next_unit_x = next_dx / next_length;
                let next_unit_y = next_dy / next_length;

                let arc_end = [
                    next[0] + next_unit_x * corner_radius_used,
                    next[1] + next_unit_y * corner_radius_used,
                ];

                let arc_length = quadratic_arc_length(effective_end, next, arc_end);

                segments.push(PathSegment {
                    seg: Seg::Quadratic {
                        p0: effective_end,
                        p1: arc_end,
                        cp: next,
                    },
                    arc_length,
                });
            }
        }
    }

    segments
}

// ---------------------------------------------------------------------------
// M4b — text-gap carving primitives (De Casteljau splits + arc-length
// sub-path extraction). Ports of `utils/lineToPath.ts`:
// `splitCubicBezier` (:744-775), `splitQuadraticBezier` (:782-799),
// `subdivideCubic` (:808-837), `subdivideQuadratic` (:842-864),
// `extractSubSegment` (:872-941), `extractPathSegmentsByArcLength`
// (:953-975).
// ---------------------------------------------------------------------------

/// A cubic Bezier as (p0, cp1, cp2, p1).
pub type Cubic = (Pt, Pt, Pt, Pt);
/// A quadratic Bezier as (p0, cp, p1).
pub type Quad = (Pt, Pt, Pt);

fn lerp_pt(a: Pt, b: Pt, t: f64) -> Pt {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

/// `splitCubicBezier` (:744-775): De Casteljau split at `t`;
/// first covers [0, t], second covers [t, 1].
pub fn split_cubic(p0: Pt, cp1: Pt, cp2: Pt, p1: Pt, t: f64) -> (Cubic, Cubic) {
    let q0 = lerp_pt(p0, cp1, t);
    let q1 = lerp_pt(cp1, cp2, t);
    let q2 = lerp_pt(cp2, p1, t);
    let r0 = lerp_pt(q0, q1, t);
    let r1 = lerp_pt(q1, q2, t);
    let s = lerp_pt(r0, r1, t);
    ((p0, q0, r0, s), (s, r1, q2, p1))
}

/// `splitQuadraticBezier` (:782-799).
pub fn split_quadratic(p0: Pt, cp: Pt, p1: Pt, t: f64) -> (Quad, Quad) {
    let q0 = lerp_pt(p0, cp, t);
    let q1 = lerp_pt(cp, p1, t);
    let r = lerp_pt(q0, q1, t);
    ((p0, q0, r), (r, q1, p1))
}

/// `subdivideCubic` (:808-837): sub-curve covering parameters [t_start, t_end].
pub fn subdivide_cubic(p0: Pt, cp1: Pt, cp2: Pt, p1: Pt, t_start: f64, t_end: f64) -> Cubic {
    let mut current = (p0, cp1, cp2, p1);
    let mut local_end = t_end;

    if t_start > 0.0 {
        let (_, second) = split_cubic(current.0, current.1, current.2, current.3, t_start);
        current = second;
        local_end = if t_start < 1.0 {
            (t_end - t_start) / (1.0 - t_start)
        } else {
            0.0
        };
    }
    if local_end < 1.0 {
        let (first, _) = split_cubic(current.0, current.1, current.2, current.3, local_end);
        current = first;
    }
    current
}

/// `subdivideQuadratic` (:842-864).
pub fn subdivide_quadratic(p0: Pt, cp: Pt, p1: Pt, t_start: f64, t_end: f64) -> Quad {
    let mut current = (p0, cp, p1);
    let mut local_end = t_end;

    if t_start > 0.0 {
        let (_, second) = split_quadratic(current.0, current.1, current.2, t_start);
        current = second;
        local_end = if t_start < 1.0 {
            (t_end - t_start) / (1.0 - t_start)
        } else {
            0.0
        };
    }
    if local_end < 1.0 {
        let (first, _) = split_quadratic(current.0, current.1, current.2, local_end);
        current = first;
    }
    current
}

/// `extractSubSegment` (:872-941): sub-segment covering the arc-length range
/// [local_start_dist, local_end_dist] measured from the segment's start.
pub fn extract_sub_segment(
    segment: &PathSegment,
    local_start_dist: f64,
    local_end_dist: f64,
) -> Option<PathSegment> {
    let seg_len = segment.arc_length;
    if seg_len <= 0.001 {
        return None;
    }

    let clamped_start = local_start_dist.clamp(0.0, seg_len);
    let clamped_end = local_end_dist.clamp(0.0, seg_len);
    if clamped_end - clamped_start <= 0.001 {
        return None;
    }

    // Fast path: fully covers the original segment.
    if clamped_start <= 0.001 && clamped_end >= seg_len - 0.001 {
        return Some(segment.clone());
    }

    match &segment.seg {
        Seg::Line { p0, p1 } => {
            let t_start = clamped_start / seg_len;
            let t_end = clamped_end / seg_len;
            let sp = lerp_pt(*p0, *p1, t_start);
            let ep = lerp_pt(*p0, *p1, t_end);
            Some(PathSegment {
                seg: Seg::Line { p0: sp, p1: ep },
                arc_length: clamped_end - clamped_start,
            })
        }
        Seg::Quadratic { p0, p1, cp } => {
            let t_start = find_t_at_arc_length_quadratic(*p0, *cp, *p1, clamped_start, seg_len);
            let t_end = find_t_at_arc_length_quadratic(*p0, *cp, *p1, clamped_end, seg_len);
            if t_end - t_start <= 1e-6 {
                return None;
            }
            let (sp0, scp, sp1) = subdivide_quadratic(*p0, *cp, *p1, t_start, t_end);
            Some(PathSegment {
                arc_length: quadratic_arc_length(sp0, scp, sp1),
                seg: Seg::Quadratic {
                    p0: sp0,
                    p1: sp1,
                    cp: scp,
                },
            })
        }
        Seg::Cubic { p0, p1, cp1, cp2 } => {
            let t_start = find_t_at_arc_length_cubic(*p0, *cp1, *cp2, *p1, clamped_start, seg_len);
            let t_end = find_t_at_arc_length_cubic(*p0, *cp1, *cp2, *p1, clamped_end, seg_len);
            if t_end - t_start <= 1e-6 {
                return None;
            }
            let (sp0, scp1, scp2, sp1) = subdivide_cubic(*p0, *cp1, *cp2, *p1, t_start, t_end);
            Some(PathSegment {
                arc_length: cubic_arc_length(sp0, scp1, scp2, sp1),
                seg: Seg::Cubic {
                    p0: sp0,
                    p1: sp1,
                    cp1: scp1,
                    cp2: scp2,
                },
            })
        }
    }
}

/// `extractPathSegmentsByArcLength` (:953-975): contiguous sub-path covering
/// [start_dist, end_dist] measured from the start of the whole path.
pub fn extract_path_segments_by_arc_length(
    segments: &[PathSegment],
    start_dist: f64,
    end_dist: f64,
) -> Vec<PathSegment> {
    let mut result: Vec<PathSegment> = Vec::new();
    if end_dist - start_dist <= 0.001 {
        return result;
    }

    let mut current_dist = 0.0f64;
    for segment in segments {
        let seg_len = segment.arc_length;
        let seg_start = current_dist;
        let seg_end = current_dist + seg_len;
        current_dist = seg_end;

        if seg_end <= start_dist + 0.001 {
            continue;
        }
        if seg_start >= end_dist - 0.001 {
            break;
        }

        let local_start = (start_dist - seg_start).max(0.0);
        let local_end = (end_dist - seg_start).min(seg_len);

        if let Some(sub) = extract_sub_segment(segment, local_start, local_end) {
            result.push(sub);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Dash pieces — one sub-path's walk output, grouped for the stroker.
// ---------------------------------------------------------------------------

/// One stroked run of a dashed sub-path. An open piece takes caps at its two ends and
/// the configured join at every interior vertex; a closed piece is a dash that never
/// turned off around a closed sub-path and is stroked as a loop (joins only, no caps).
#[derive(Debug, Clone, PartialEq)]
pub struct DashPiece {
    pub points: Vec<Pt>,
    pub closed: bool,
}

/// The walk reaches a segment end through `d + (len - d)`, which is not always exactly
/// `len`, so the seam vertex is matched within the walk's own 0.001 resolution.
const SEAM_EPS: f64 = 1e-3;

fn same_point(a: Pt, b: Pt) -> bool {
    (a[0] - b[0]).abs() < SEAM_EPS && (a[1] - b[1]).abs() < SEAM_EPS
}

/// Group one sub-path's walk output into [`DashPiece`]s. `start` / `end` are the
/// sub-path's first and last segment endpoints (equal on a closed sub-path, whose closing
/// edge is one of its segments). On a closed sub-path a dash that runs off the end and
/// one that starts at the beginning are the same dash, so they are stitched through the
/// seam vertex — a join there instead of two caps. A lone dash that wraps covers the
/// whole contour and comes back `closed`, without the walk's return to `start` (the
/// stroker closes a loop itself).
pub fn dash_pieces(cmds: &[WalkCmd], closed: bool, start: Pt, end: Pt) -> Vec<DashPiece> {
    let mut pieces: Vec<DashPiece> = Vec::new();
    for c in cmds {
        match *c {
            WalkCmd::Move(p) => pieces.push(DashPiece {
                points: vec![p],
                closed: false,
            }),
            WalkCmd::Line(p) => {
                if let Some(last) = pieces.last_mut() {
                    last.points.push(p);
                }
            }
        }
    }
    if !closed || pieces.is_empty() {
        return pieces;
    }
    let first_on = same_point(pieces[0].points[0], start);
    let last_on = pieces
        .last()
        .and_then(|p| p.points.last().copied())
        .map(|p| same_point(p, end))
        .unwrap_or(false);
    if !(first_on && last_on) {
        return pieces;
    }
    if pieces.len() == 1 {
        let only = &mut pieces[0];
        only.points.pop();
        only.closed = true;
        return pieces;
    }
    let mut tail = pieces.pop().expect("at least two pieces");
    // The seam vertex is kept from the head piece, which holds it exactly.
    tail.points.pop();
    tail.points.append(&mut pieces[0].points);
    pieces[0].points = tail.points;
    pieces
}

#[cfg(test)]
mod dash_piece_tests {
    use super::*;
    use crate::tess::moda_engine_path_types::{FillRule, LineCap, LineJoin, PathCmd, StrokeAlign};
    use crate::tess::tessellate_combined_oracle;

    fn seg_line(p0: Pt, p1: Pt) -> PathSegment {
        PathSegment {
            seg: Seg::Line { p0, p1 },
            arc_length: (p1[0] - p0[0]).hypot(p1[1] - p0[1]),
        }
    }

    /// Square of side 100 with the closing edge as its fourth segment.
    fn square_segments() -> Vec<PathSegment> {
        let c = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]];
        (0..4).map(|i| seg_line(c[i], c[(i + 1) % 4])).collect()
    }

    fn near(a: Pt, b: Pt) -> bool {
        (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6
    }

    #[test]
    fn dash_phase_continues_through_a_vertex() {
        // Two 50-long edges; the first 60-long dash runs 10 past the corner, then a gap
        // and a second dash (80-100) that never sees the corner.
        let segs = [
            seg_line([0.0, 0.0], [50.0, 0.0]),
            seg_line([50.0, 0.0], [50.0, 50.0]),
        ];
        let cmds = walk_path_by_arc_length(&segs, &[60.0, 20.0]);
        let pieces = dash_pieces(&cmds, false, [0.0, 0.0], [50.0, 50.0]);
        assert_eq!(pieces.len(), 2, "{pieces:?}");
        let pts = &pieces[0].points;
        assert_eq!(pts.len(), 3, "start, corner, end: {pts:?}");
        assert!(
            near(pts[1], [50.0, 0.0]),
            "the corner is an interior vertex: {pts:?}"
        );
        assert!(
            near(pts[2], [50.0, 10.0]),
            "the dash keeps its phase past the corner: {pts:?}"
        );
        assert!(!pieces[0].closed);
    }

    #[test]
    fn closed_square_stitches_the_wrapping_dash_through_the_seam() {
        // Perimeter 400, pattern 150/30: dashes 0-150, 180-330, 360-400, and the last one
        // is on at the seam, so 360-400 → 0-150 is ONE piece through the start corner.
        let cmds = walk_path_by_arc_length(&square_segments(), &[150.0, 30.0]);
        let pieces = dash_pieces(&cmds, true, [0.0, 0.0], [0.0, 0.0]);
        assert_eq!(pieces.len(), 2, "{pieces:?}");
        let head = &pieces[0];
        assert!(!head.closed);
        assert!(
            near(head.points[0], [0.0, 40.0]),
            "merged piece starts on the left edge: {head:?}"
        );
        assert!(
            near(head.points[1], [0.0, 0.0]),
            "the seam corner is interior: {head:?}"
        );
        assert!(
            near(head.points[2], [100.0, 0.0]) && near(head.points[3], [100.0, 50.0]),
            "{head:?}"
        );
        // The unstitched open walk is untouched.
        let open = dash_pieces(&cmds, false, [0.0, 0.0], [0.0, 0.0]);
        assert_eq!(open.len(), 3);
    }

    #[test]
    fn a_dash_longer_than_the_contour_is_one_closed_piece() {
        let cmds = walk_path_by_arc_length(&square_segments(), &[1000.0, 10.0]);
        let pieces = dash_pieces(&cmds, true, [0.0, 0.0], [0.0, 0.0]);
        assert_eq!(pieces.len(), 1);
        assert!(pieces[0].closed);
        assert_eq!(
            pieces[0].points.len(),
            4,
            "no return to start: {:?}",
            pieces[0].points
        );
    }

    #[test]
    fn a_gap_at_the_seam_leaves_the_pieces_open() {
        // Pattern 100/100 on perimeter 400: dashes 0-100, 200-300; off at the seam.
        let cmds = walk_path_by_arc_length(&square_segments(), &[100.0, 100.0]);
        let pieces = dash_pieces(&cmds, true, [0.0, 0.0], [0.0, 0.0]);
        assert_eq!(pieces.len(), 2);
        assert!(pieces.iter().all(|p| !p.closed && p.points.len() == 2));
    }

    fn square_cmds() -> Vec<PathCmd> {
        vec![
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Line { x: 100.0, y: 0.0 },
            PathCmd::Line { x: 100.0, y: 100.0 },
            PathCmd::Line { x: 0.0, y: 100.0 },
        ]
    }

    /// Stroke-only geometry (transparent fill contributes nothing on an open path; on a
    /// closed one the fill triangles are the same on both sides of each comparison).
    fn stroke(
        points: &[PathCmd],
        closed: bool,
        dash: Option<&[f64]>,
        cap: LineCap,
        join: LineJoin,
    ) -> Vec<f32> {
        // stroke_width 2 == BASE_DASH_STROKE_WIDTH keeps the pattern unscaled.
        tessellate_combined_oracle(
            points,
            2.0,
            closed,
            None,
            dash,
            0.0,
            cap,
            join,
            FillRule::NonZero,
            StrokeAlign::Center,
        )
        .positions
    }

    #[test]
    fn closed_square_fully_on_has_four_joins_and_no_caps() {
        // A dash that never turns off is the solid closed stroke: four miter corners,
        // and the round caps requested are never emitted.
        let pts = square_cmds();
        let solid = stroke(&pts, true, None, LineCap::Round, LineJoin::Miter);
        let dashed = stroke(
            &pts,
            true,
            Some(&[1000.0, 10.0]),
            LineCap::Round,
            LineJoin::Miter,
        );
        assert_eq!(dashed, solid);
    }

    #[test]
    fn a_join_is_emitted_at_a_vertex_under_dash() {
        // Open L; the first dash (60) runs through the corner at (50,0). With a miter join
        // the outer tip sits at (51,-1) — half a width past the corner on both axes.
        let pts = vec![
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Line { x: 50.0, y: 0.0 },
            PathCmd::Line { x: 50.0, y: 50.0 },
        ];
        let dashed = stroke(
            &pts,
            false,
            Some(&[60.0, 20.0]),
            LineCap::Butt,
            LineJoin::Miter,
        );
        let has_tip = dashed
            .chunks(2)
            .any(|v| (v[0] - 51.0).abs() < 1e-4 && (v[1] + 1.0).abs() < 1e-4);
        assert!(has_tip, "miter tip missing: {dashed:?}");
        // Covering the whole path reproduces the solid open stroke: joins and end caps.
        let solid = stroke(&pts, false, None, LineCap::Square, LineJoin::Round);
        let covered = stroke(
            &pts,
            false,
            Some(&[1000.0, 10.0]),
            LineCap::Square,
            LineJoin::Round,
        );
        assert_eq!(covered, solid);
    }

    #[test]
    fn open_dashed_path_caps_only_at_dash_ends() {
        // 100-long line, pattern 30/20 → dashes 0-30 and 50-80, each a 30-long line with
        // two round caps: exactly twice the geometry of one such line.
        let line = |x0: f64, x1: f64| {
            vec![
                PathCmd::Move { x: x0, y: 0.0 },
                PathCmd::Line { x: x1, y: 0.0 },
            ]
        };
        let dashed = stroke(
            &line(0.0, 100.0),
            false,
            Some(&[30.0, 20.0]),
            LineCap::Round,
            LineJoin::Miter,
        );
        let one = stroke(
            &line(0.0, 30.0),
            false,
            None,
            LineCap::Round,
            LineJoin::Miter,
        );
        assert_eq!(dashed.len(), 2 * one.len());
    }

    #[test]
    fn compound_path_dashes_each_subpath_from_its_own_start() {
        // Two squares 200 apart; a dash spanning the first's seam must not cross to the
        // second. Every emitted vertex lies on one square or the other.
        let mut pts = square_cmds();
        pts.extend(square_cmds().into_iter().map(|c| match c {
            PathCmd::Move { x, y } => PathCmd::Move { x: x + 300.0, y },
            PathCmd::Line { x, y } => PathCmd::Line { x: x + 300.0, y },
            other => other,
        }));
        let dashed = stroke(
            &pts,
            true,
            Some(&[150.0, 30.0]),
            LineCap::Butt,
            LineJoin::Miter,
        );
        for v in dashed.chunks(2) {
            assert!(
                v[0] <= 101.0 || v[0] >= 299.0,
                "vertex bridged the gap between sub-paths: {v:?}"
            );
        }
    }
}

#[cfg(test)]
mod dash_offset_tests {
    use super::*;

    fn seg_line(x0: f64, y0: f64, x1: f64, y1: f64) -> PathSegment {
        PathSegment {
            seg: Seg::Line {
                p0: [x0, y0],
                p1: [x1, y1],
            },
            arc_length: (x1 - x0).hypot(y1 - y0),
        }
    }

    #[test]
    fn seed_phase_at_zero_matches_unphased_init() {
        // Offset 0 → start of first dash, full remaining.
        assert_eq!(seed_dash_phase(&[10.0, 10.0], 0.0), (0, true, 10.0));
    }

    #[test]
    fn seed_phase_within_first_dash() {
        assert_eq!(seed_dash_phase(&[10.0, 10.0], 4.0), (0, true, 6.0));
    }

    #[test]
    fn seed_phase_into_gap() {
        // Offset 10 lands exactly at the gap start (element 1, not a dash).
        assert_eq!(seed_dash_phase(&[10.0, 10.0], 10.0), (1, false, 10.0));
        // Offset 15 → 5 into the gap.
        assert_eq!(seed_dash_phase(&[10.0, 10.0], 15.0), (1, false, 5.0));
    }

    #[test]
    fn seed_phase_wraps_modulo_pattern_length() {
        // 25 mod 20 == 5 (into first dash); -5 mod 20 == 15 (into gap).
        assert_eq!(seed_dash_phase(&[10.0, 10.0], 25.0), (0, true, 5.0));
        assert_eq!(seed_dash_phase(&[10.0, 10.0], -5.0), (1, false, 5.0));
    }

    #[test]
    fn phased_walk_at_zero_equals_unphased() {
        let segs = [seg_line(0.0, 0.0, 100.0, 0.0)];
        let pat = [10.0, 10.0];
        assert_eq!(
            walk_path_by_arc_length_phased(&segs, &pat, 0.0),
            walk_path_by_arc_length(&segs, &pat)
        );
    }

    #[test]
    fn phased_walk_full_period_offset_equals_unphased() {
        // Offsetting by a whole pattern length is a no-op (one full loop).
        let segs = [seg_line(0.0, 0.0, 100.0, 0.0)];
        let pat = [10.0, 10.0];
        assert_eq!(
            walk_path_by_arc_length_phased(&segs, &pat, 20.0),
            walk_path_by_arc_length(&segs, &pat)
        );
    }

    #[test]
    fn phased_walk_offset_shifts_first_dash() {
        // Offset 10 → pattern starts on the gap, so the first drawn dash begins at x=10.
        let segs = [seg_line(0.0, 0.0, 100.0, 0.0)];
        let pat = [10.0, 10.0];
        let cmds = walk_path_by_arc_length_phased(&segs, &pat, 10.0);
        match cmds.first() {
            Some(WalkCmd::Move(p)) => assert!(
                (p[0] - 10.0).abs() < 1e-6,
                "first move at x=10, got {}",
                p[0]
            ),
            other => panic!("expected leading Move, got {other:?}"),
        }
    }
}
