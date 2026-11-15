//! `BlendPathPoint` (the TS `PathPoint` mirror) ↔ `flo_curves::SimpleBezierPath`.
//!
//! Every segment becomes a cubic: flo_curves only knows cubics, and the TS
//! `PathPoint` shape carries lines / quadratics / SVG arcs as well. The inverse
//! direction re-detects lines (a cubic whose control points are collinear with
//! its chord) so that `rect ∪ rect` comes back as `line` points rather than as
//! degenerate curves.

use flo_curves::bezier::path::SimpleBezierPath;
use flo_curves::geo::Coord2;
use i_overlay::core::fill_rule::FillRule as OverlayFill;
use i_overlay::float::simplify::SimplifyShape;

use crate::blend::BlendPathPoint;
use crate::path_geom::arc::{arc_to_beziers, ArcParams};

/// Coincident-anchor epsilon. Smaller than any accuracy the solver accepts, so
/// dropping a segment here never changes a shape the caller could see.
const COINCIDENT_EPS: f64 = 1e-9;

/// A cubic segment in flo_curves' hull order: (cp1, cp2, end).
pub(crate) type Segment = (Coord2, Coord2, Coord2);

fn lerp(a: (f64, f64), b: (f64, f64), t: f64) -> Coord2 {
    Coord2(a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
}

fn near(a: Coord2, b: Coord2) -> bool {
    (a.0 - b.0).abs() <= COINCIDENT_EPS && (a.1 - b.1).abs() <= COINCIDENT_EPS
}

#[cfg(test)]
thread_local! {
    /// Bumped on entry to `points_to_subpaths`. Thread-local so the guard tests
    /// can assert a refused request never reached the conversion — the
    /// expansion the point budget exists to prevent — without racing the rest
    /// of the suite.
    static CONVERSIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn conversion_count() -> usize {
    CONVERSIONS.with(std::cell::Cell::get)
}

/// Split `points` at every `Move` and turn each subpath into a closed
/// `SimpleBezierPath`. Degenerate subpaths (fewer than two distinct anchors, or
/// |signed area| below `accuracy²`) are dropped — flo_curves' graph builder is
/// where zero-area spurs turn into panics, so this scrubbing is load-bearing.
pub(crate) fn points_to_subpaths(
    points: &[BlendPathPoint],
    accuracy: f64,
) -> Vec<SimpleBezierPath> {
    #[cfg(test)]
    CONVERSIONS.with(|c| c.set(c.get() + 1));
    let mut out: Vec<SimpleBezierPath> = Vec::new();
    let mut start: Option<Coord2> = None;
    let mut cursor = Coord2(0.0, 0.0);
    let mut segments: Vec<Segment> = Vec::new();

    let flush =
        |start: Option<Coord2>, segments: &mut Vec<Segment>, out: &mut Vec<SimpleBezierPath>| {
            let Some(start) = start else {
                segments.clear();
                return;
            };
            let mut segs = std::mem::take(segments);
            // Implicitly close: every boolean operand is a filled region.
            if let Some(last) = segs.last() {
                if !near(last.2, start) {
                    let from = (last.2 .0, last.2 .1);
                    let to = (start.0, start.1);
                    segs.push((lerp(from, to, 1.0 / 3.0), lerp(from, to, 2.0 / 3.0), start));
                }
            }
            if let Some(path) = finish_subpath(start, segs, accuracy) {
                out.push(path);
            }
        };

    for p in points {
        match p {
            BlendPathPoint::Move { x, y } => {
                flush(start, &mut segments, &mut out);
                cursor = Coord2(*x, *y);
                start = Some(cursor);
            }
            BlendPathPoint::Line { x, y } => {
                if start.is_none() {
                    cursor = Coord2(*x, *y);
                    start = Some(cursor);
                    continue;
                }
                let end = Coord2(*x, *y);
                let from = (cursor.0, cursor.1);
                let to = (end.0, end.1);
                segments.push((lerp(from, to, 1.0 / 3.0), lerp(from, to, 2.0 / 3.0), end));
                cursor = end;
            }
            BlendPathPoint::Curve { x, y, cp1, cp2 } => {
                if start.is_none() {
                    cursor = Coord2(*x, *y);
                    start = Some(cursor);
                    continue;
                }
                let end = Coord2(*x, *y);
                segments.push((Coord2(cp1.0, cp1.1), Coord2(cp2.0, cp2.1), end));
                cursor = end;
            }
            BlendPathPoint::Quadratic { x, y, cp } => {
                if start.is_none() {
                    cursor = Coord2(*x, *y);
                    start = Some(cursor);
                    continue;
                }
                let end = Coord2(*x, *y);
                // Exact degree elevation: c1 = p0 + 2/3·(q − p0), c2 = p1 + 2/3·(q − p1).
                let c1 = Coord2(
                    cursor.0 + 2.0 / 3.0 * (cp.0 - cursor.0),
                    cursor.1 + 2.0 / 3.0 * (cp.1 - cursor.1),
                );
                let c2 = Coord2(
                    end.0 + 2.0 / 3.0 * (cp.0 - end.0),
                    end.1 + 2.0 / 3.0 * (cp.1 - end.1),
                );
                segments.push((c1, c2, end));
                cursor = end;
            }
            BlendPathPoint::Arc {
                x,
                y,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            } => {
                if start.is_none() {
                    cursor = Coord2(*x, *y);
                    start = Some(cursor);
                    continue;
                }
                let end = Coord2(*x, *y);
                let params = ArcParams {
                    rx: *rx,
                    ry: *ry,
                    x_axis_rotation: *x_axis_rotation,
                    large_arc: *large_arc,
                    sweep: *sweep,
                    x: *x,
                    y: *y,
                };
                let beziers = arc_to_beziers(cursor.0, cursor.1, &params);
                if beziers.is_empty() {
                    // Degenerate arc → line to the endpoint (TS does the same).
                    let from = (cursor.0, cursor.1);
                    let to = (end.0, end.1);
                    segments.push((lerp(from, to, 1.0 / 3.0), lerp(from, to, 2.0 / 3.0), end));
                } else {
                    let last = beziers.len() - 1;
                    for (i, b) in beziers.iter().enumerate() {
                        // The arc parameterization reconstructs its endpoint
                        // through cos/sin, so the final cubic lands a few ulps
                        // off `end` while the cursor advances to `end` exactly.
                        // That hairline gap reads as a self-intersection to the
                        // solver, so pin the last end point to the real target.
                        let tip = if i == last {
                            end
                        } else {
                            Coord2(b.end[0], b.end[1])
                        };
                        segments.push((
                            Coord2(b.cp1[0], b.cp1[1]),
                            Coord2(b.cp2[0], b.cp2[1]),
                            tip,
                        ));
                    }
                }
                cursor = end;
            }
        }
    }
    flush(start, &mut segments, &mut out);
    out
}

/// Drop zero-length segments, then reject the subpath if it cannot bound area.
fn finish_subpath(
    start: Coord2,
    segments: Vec<Segment>,
    accuracy: f64,
) -> Option<SimpleBezierPath> {
    let mut cursor = start;
    let mut kept: Vec<Segment> = Vec::with_capacity(segments.len());
    for seg in segments {
        let flat = near(seg.2, cursor) && near(seg.0, cursor) && near(seg.1, cursor);
        if flat {
            continue;
        }
        cursor = seg.2;
        kept.push(seg);
    }
    // A single cubic can still bound area (a teardrop whose start and end
    // coincide), so only an empty segment list is rejected outright — the
    // area test below is what actually discards degenerate subpaths.
    if kept.is_empty() {
        return None;
    }
    let area = subpath_signed_area(&(start, kept.clone()));
    // A loop that crosses itself can enclose area whose signed parts cancel (a balanced figure-8 or bowtie: its
    // lobes wind opposite ways, and both are filled), so a zero net area only drops a subpath that encloses
    // nothing once its crossings are resolved (a spur, a doubled-back line).
    if area.abs() < accuracy * accuracy && enclosed_area(start, &kept) < accuracy * accuracy {
        return None;
    }
    Some((start, kept))
}

/// The area a closed subpath encloses with its self-crossings resolved (non-zero winding; a figure-8's two lobes
/// both count), from a flattening (16 chords per cubic). Only used to tell a self-crossing loop from a degenerate
/// one, so the flattening error doesn't matter.
fn enclosed_area(start: Coord2, segments: &[Segment]) -> f64 {
    let mut ring: Vec<[f64; 2]> = vec![[start.0, start.1]];
    let mut p0 = start;
    for &(c1, c2, p3) in segments {
        for i in 1..=16 {
            let t = i as f64 / 16.0;
            let mt = 1.0 - t;
            let (a, b, c, d) = (mt * mt * mt, 3.0 * mt * mt * t, 3.0 * mt * t * t, t * t * t);
            ring.push([
                a * p0.0 + b * c1.0 + c * c2.0 + d * p3.0,
                a * p0.1 + b * c1.1 + c * c2.1 + d * p3.1,
            ]);
        }
        p0 = p3;
    }
    if !ring.iter().all(|p| p[0].is_finite() && p[1].is_finite()) {
        return 0.0;
    }
    let ring_area = |c: &Vec<[f64; 2]>| {
        let n = c.len();
        ((0..n)
            .map(|i| c[i][0] * c[(i + 1) % n][1] - c[(i + 1) % n][0] * c[i][1])
            .sum::<f64>()
            / 2.0)
            .abs()
    };
    vec![ring]
        .simplify_shape(OverlayFill::NonZero)
        .iter()
        .map(|shape| {
            let mut rings = shape.iter();
            let outer = rings.next().map(ring_area).unwrap_or(0.0);
            outer - rings.map(ring_area).sum::<f64>()
        })
        .sum()
}

/// Exact signed area of a closed cubic subpath (Green's theorem, integrated
/// analytically over each Bezier — no flattening error). Positive for a
/// counter-clockwise loop in a y-down space.
pub(crate) fn subpath_signed_area(path: &SimpleBezierPath) -> f64 {
    let (start, segments) = path;
    let mut p0 = *start;
    let mut total = 0.0;
    for (c1, c2, p3) in segments {
        total += cubic_area_term(p0, *c1, *c2, *p3);
        p0 = *p3;
    }
    // Close with a straight chord if the subpath does not already return home.
    if !near(p0, *start) {
        total += (p0.0 * start.1 - start.0 * p0.1) / 2.0;
    }
    total
}

/// ½∮(x dy − y dx) over one cubic, integrated exactly in Bernstein form. The
/// bilinear coefficient matrix is (1/20)·[[0,6,3,1],[-6,0,3,3],[-3,-3,0,6],[-1,-3,-6,0]].
fn cubic_area_term(p0: Coord2, p1: Coord2, p2: Coord2, p3: Coord2) -> f64 {
    let (x0, y0) = (p0.0, p0.1);
    let (x1, y1) = (p1.0, p1.1);
    let (x2, y2) = (p2.0, p2.1);
    let (x3, y3) = (p3.0, p3.1);
    (x0 * (6.0 * y1 + 3.0 * y2 + y3)
        + x1 * (-6.0 * y0 + 3.0 * y2 + 3.0 * y3)
        + x2 * (-3.0 * y0 - 3.0 * y1 + 6.0 * y3)
        + x3 * (-y0 - 3.0 * y1 - 6.0 * y2))
        / 20.0
}

/// Inverse mapping: a solved `SimpleBezierPath` back to `PathPoint`s. Emits a
/// `Move` for the start anchor, then a `Line` for every collinear cubic and a
/// `Curve` otherwise. The closing segment is dropped when it is a straight line
/// back to the start — the TS node carries `closed: true` and supplies it —
/// unless that would leave the ring fewer than three points.
pub(crate) fn subpath_to_points(path: &SimpleBezierPath, out: &mut Vec<BlendPathPoint>) {
    let (start, segments) = path;
    out.push(BlendPathPoint::Move {
        x: start.0,
        y: start.1,
    });
    let mut cursor = *start;
    // A one-cubic loop (a teardrop) would be two points, which callers read as
    // "no area": split it in half (exactly, by de Casteljau) so it is three.
    let halves: Vec<Segment>;
    let segments = match segments.as_slice() {
        [(c1, c2, p3)] if !cubic_is_line(*start, *c1, *c2, *p3) => {
            let mid = |a: Coord2, b: Coord2| Coord2((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
            let (p01, p12, p23) = (mid(*start, *c1), mid(*c1, *c2), mid(*c2, *p3));
            let (p012, p123) = (mid(p01, p12), mid(p12, p23));
            let m = mid(p012, p123);
            halves = vec![(p01, p012, m), (p123, p23, *p3)];
            &halves
        }
        _ => segments,
    };
    let last_index = segments.len().saturating_sub(1);
    for (i, (c1, c2, p3)) in segments.iter().enumerate() {
        let is_line = cubic_is_line(cursor, *c1, *c2, *p3);
        // Not for a ring of two segments: dropping its closing line would leave
        // two points (a curve and its chord, like a "D"), which every caller
        // reads as "no area" and discards.
        if i == last_index && i >= 2 && is_line && near(*p3, *start) {
            break;
        }
        if is_line {
            out.push(BlendPathPoint::Line { x: p3.0, y: p3.1 });
        } else {
            out.push(BlendPathPoint::Curve {
                x: p3.0,
                y: p3.1,
                cp1: (c1.0, c1.1),
                cp2: (c2.0, c2.1),
            });
        }
        cursor = *p3;
    }
}

/// True when both control points lie on the chord (within a relative epsilon)
/// and between its endpoints, so the cubic traces the straight chord exactly.
fn cubic_is_line(p0: Coord2, c1: Coord2, c2: Coord2, p3: Coord2) -> bool {
    if (p3.0 - p0.0).hypot(p3.1 - p0.1) <= COINCIDENT_EPS {
        return near(c1, p0) && near(c2, p0);
    }
    use crate::path_geom::vec2::{v2, Cubic};
    let c = Cubic([
        v2(p0.0, p0.1),
        v2(c1.0, c1.1),
        v2(c2.0, c2.1),
        v2(p3.0, p3.1),
    ]);
    c.is_chord(1e-6 * (c.0[3] - c.0[0]).len().max(1.0), 1e-6)
}

/// Flatten a solved subpath to a polyline (uniform de Casteljau sampling, the
/// same 16 segments per curve `flattenPathToPolyline` uses). Used for area /
/// containment checks: by the tests, and by the pathfinder's face splitting,
/// which only needs to know which ring nests in which.
pub(crate) fn flatten_subpath(path: &SimpleBezierPath, per_curve: usize) -> Vec<(f64, f64)> {
    let (start, segments) = path;
    let mut out = vec![(start.0, start.1)];
    let mut p0 = *start;
    for (c1, c2, p3) in segments {
        for i in 1..=per_curve {
            let t = i as f64 / per_curve as f64;
            let mt = 1.0 - t;
            let a = mt * mt * mt;
            let b = 3.0 * mt * mt * t;
            let c = 3.0 * mt * t * t;
            let d = t * t * t;
            out.push((
                a * p0.0 + b * c1.0 + c * c2.0 + d * p3.0,
                a * p0.1 + b * c1.1 + c * c2.1 + d * p3.1,
            ));
        }
        p0 = *p3;
    }
    out
}
