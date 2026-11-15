//! shape_to_path.rs — export-mode shape → cubic-path conversion (M1: rect,
//! ellipse; M2's star/polygon converters landed in `tess/shape_paths.rs`, not here).
//!
//! Ports:
//! - `renderers/webgpu/utils/shapeToPathNodeUtils.ts`
//!   (`convertRectangleToPathNode` / `convertEllipseToPathNode`)
//! - `utils/rectangleToPath.ts` (`rectangleToPathPoints`, KAPPA corners)
//! - `utils/roundedRectRadii.ts` (`normalizeRoundedRectRadii` — CSS-like
//!   proportional scaling; note this is the EXPORT/tessellation behavior. The
//!   per-corner clamp WITHOUT pairwise scaling described in
//!   render-semantics §3.1 belongs to the SDF fast path, which is disabled
//!   for exports.)
//! - `utils/ellipseToPath.ts` (`ellipseDimensionsToPathPoints` — quadrant
//!   beziers, wedges, rings, annular sectors).
//!
//! All output is in node-local space (origin top-left); the model matrix
//! positions it.

use crate::tess::moda_engine_path_types::PathCmd;
use crate::tess::path_cmd_from_blend_point;
use moda_scene::value::Value;

// The converters live in `moda_scene::shape_outline` (the live geometry
// effects read shape outlines at scene resolution); these wrap them in the
// tessellator's command type.
pub use moda_scene::shape_outline::{expand_corner_radii, normalize_rounded_rect_radii, KAPPA};

/// `rectangleToPathPoints` (rectangleToPath.ts:32-...) with x=y=0, rotation 0
/// (the converter always calls it that way; transform comes from the model).
pub fn rectangle_to_path_cmds(
    width: f64,
    height: f64,
    corner_radius: Option<&Value>,
) -> Vec<PathCmd> {
    moda_scene::shape_outline::rectangle_outline(width, height, corner_radius)
        .iter()
        .map(path_cmd_from_blend_point)
        .collect()
}

/// `cornerSmoothing` as stored (0–1, clamped; absent or invalid → 0).
pub fn corner_smoothing(v: Option<&Value>) -> f64 {
    moda_scene::corner_smoothing::clamp_smoothing(v.and_then(Value::as_f64).unwrap_or(0.0))
}

/// Whether a rectangle / container outline needs the smoothed (squircle)
/// construction: a positive `cornerSmoothing` and at least one rounded corner.
pub fn rect_is_smoothed(corner_radius: Option<&Value>, corner_smoothing_v: Option<&Value>) -> bool {
    moda_scene::corner_smoothing::is_smoothed(expand_corner_radii(corner_radius), corner_smoothing(corner_smoothing_v))
}

/// Whether a rectangle's outline is a path no SDF draws: smoothed corners, or styled ones (`cornerStyle`
/// inverted / chamfer, `moda_scene::rect_corners`) with a rounded corner.
pub fn rect_has_path_corners(
    corner_radius: Option<&Value>,
    corner_smoothing_v: Option<&Value>,
    corner_style: Option<&Value>,
) -> bool {
    moda_scene::rect_corners::is_styled_rect(corner_radius, corner_style)
        || rect_is_smoothed(corner_radius, corner_smoothing_v)
}

/// The rectangle / container outline every consumer draws (body fill and
/// strokes of every alignment, clip masks, backdrop coverage, blends):
/// `rectangle_to_path_cmds` unless the corners are styled (`cornerStyle`,
/// `moda_scene::rect_corners`) or `cornerSmoothing` is on, then Figma's
/// smoothed corners (`moda_scene::corner_smoothing`).
pub fn rect_outline_cmds(
    width: f64,
    height: f64,
    corner_radius: Option<&Value>,
    corner_smoothing_v: Option<&Value>,
    corner_style: Option<&Value>,
) -> Vec<PathCmd> {
    use moda_scene::corner_smoothing::{smoothed_rect_path, Seg};
    let segs = if moda_scene::rect_corners::is_styled_rect(corner_radius, corner_style) {
        moda_scene::rect_corners::styled_rect_path(
            width,
            height,
            corner_radius,
            moda_scene::rect_corners::rect_corner_style(corner_style),
        )
    } else if rect_is_smoothed(corner_radius, corner_smoothing_v) {
        smoothed_rect_path(width, height, expand_corner_radii(corner_radius), corner_smoothing(corner_smoothing_v))
    } else {
        return rectangle_to_path_cmds(width, height, corner_radius);
    };
    segs.into_iter()
        .map(|s| match s {
            Seg::Move(x, y) => PathCmd::Move { x, y },
            Seg::Line(x, y) => PathCmd::Line { x, y },
            Seg::Cubic { cp1, cp2, to } => PathCmd::Curve { x: to.0, y: to.1, cp1, cp2 },
        })
        .collect()
}

/// `ellipseDimensionsToPathPoints` (ellipseToPath.ts:212-227) — node-local
/// (top-left origin), center (w/2, h/2), radii (w/2, h/2), rotation 0.
pub fn ellipse_to_path_cmds(
    width: f64,
    height: f64,
    start_angle: Option<f64>,
    sweep_angle: Option<f64>,
    inner_radius_ratio: Option<f64>,
) -> Vec<PathCmd> {
    moda_scene::shape_outline::ellipse_outline(
        width,
        height,
        start_angle,
        sweep_angle,
        inner_radius_ratio,
    )
    .iter()
    .map(path_cmd_from_blend_point)
    .collect()
}

// ---------------------------------------------------------------------------
// Per-side stroke weights (E6.2)
// ---------------------------------------------------------------------------

/// A box with elliptical corners `[tl, tr, br, bl]` (each `(rx, ry)`), as one closed sub-path: clockwise on screen
/// (y down), or counter-clockwise when `reverse`.
fn rounded_box_cmds(x0: f64, y0: f64, x1: f64, y1: f64, corners: [(f64, f64); 4], reverse: bool) -> Vec<PathCmd> {
    let (w, h) = (x1 - x0, y1 - y0);
    let c = corners.map(|(rx, ry)| (rx.clamp(0.0, w / 2.0), ry.clamp(0.0, h / 2.0)));
    let [tl, tr, br, bl] = c;
    // Clockwise: the top edge left to right, then each corner as a quarter ellipse.
    // (point, corner arc to the next point: (cp1, cp2)) pairs.
    let k = KAPPA;
    let mut pts: Vec<(f64, f64)> = Vec::new();
    let mut arcs: Vec<Option<((f64, f64), (f64, f64))>> = Vec::new();
    let mut push = |p: (f64, f64), arc: Option<((f64, f64), (f64, f64))>| {
        pts.push(p);
        arcs.push(arc);
    };
    // Each entry: a point, and how the path gets from the previous point to it (None = straight).
    push((x0 + tl.0, y0), None);
    push((x1 - tr.0, y0), None);
    push((x1, y0 + tr.1), Some(((x1 - tr.0 + tr.0 * k, y0), (x1, y0 + tr.1 - tr.1 * k))));
    push((x1, y1 - br.1), None);
    push((x1 - br.0, y1), Some(((x1, y1 - br.1 + br.1 * k), (x1 - br.0 + br.0 * k, y1))));
    push((x0 + bl.0, y1), None);
    push((x0, y1 - bl.1), Some(((x0 + bl.0 - bl.0 * k, y1), (x0, y1 - bl.1 + bl.1 * k))));
    push((x0, y0 + tl.1), None);
    push((x0 + tl.0, y0), Some(((x0, y0 + tl.1 - tl.1 * k), (x0 + tl.0 - tl.0 * k, y0))));
    let seg = |i: usize, rev: bool| -> PathCmd {
        // Segment arriving at pts[i] (forward) or, reversed, leaving pts[i] back to pts[i - 1].
        let (x, y) = if rev { pts[i - 1] } else { pts[i] };
        match arcs[i] {
            None => PathCmd::Line { x, y },
            Some((a, b)) => {
                let (cp1, cp2) = if rev { (b, a) } else { (a, b) };
                PathCmd::Curve { x, y, cp1, cp2 }
            }
        }
    };
    let n = pts.len();
    let mut out = Vec::with_capacity(n);
    if reverse {
        out.push(PathCmd::Move { x: pts[n - 1].0, y: pts[n - 1].1 });
        for i in (1..n).rev() {
            out.push(seg(i, true));
        }
    } else {
        out.push(PathCmd::Move { x: pts[0].0, y: pts[0].1 });
        for i in 1..n {
            out.push(seg(i, false));
        }
    }
    out
}

/// A rectangle's per-side stroke as the ring it paints (E6.2): the outer box, clockwise, then the inner box,
/// counter-clockwise (fill it nonzero). `sides` are `[top, right, bottom, left]` weights; `align` places each side's
/// band (`inside`, `outside`, else centred on the edge). Corner radii grow outward and shrink inward per axis by the
/// adjacent sides' offsets (elliptical where two sides differ), as Figma draws individual stroke weights. Empty when
/// every side is 0.
pub fn side_stroke_ring_cmds(
    width: f64,
    height: f64,
    corner_radius: Option<&Value>,
    sides: [f64; 4],
    align: Option<&str>,
) -> Vec<PathCmd> {
    if sides.iter().all(|s| *s <= 0.0) || width <= 0.0 || height <= 0.0 {
        return Vec::new();
    }
    let (outer, inner): (fn(f64) -> f64, fn(f64) -> f64) = match align {
        Some("inside") => (|_| 0.0, |s| s),
        Some("outside") => (|s| s, |_| 0.0),
        _ => (|s| s / 2.0, |s| s / 2.0),
    };
    let [t, r, b, l] = sides;
    let radii = normalize_rounded_rect_radii(width, height, corner_radius);
    // Each corner's two adjacent sides: (horizontal offset side, vertical offset side).
    let adj = [(l, t), (r, t), (r, b), (l, b)];
    let corners = |f: &dyn Fn(f64, f64) -> f64| -> [(f64, f64); 4] {
        std::array::from_fn(|i| {
            let rad = radii[i];
            if rad <= 0.0 {
                (0.0, 0.0)
            } else {
                (f(rad, adj[i].0), f(rad, adj[i].1))
            }
        })
    };
    let mut out = rounded_box_cmds(
        -outer(l),
        -outer(t),
        width + outer(r),
        height + outer(b),
        corners(&|rad, side| rad + outer(side)),
        false,
    );
    let (ix0, iy0, ix1, iy1) = (inner(l), inner(t), width - inner(r), height - inner(b));
    if ix1 - ix0 > 1e-9 && iy1 - iy0 > 1e-9 {
        out.extend(rounded_box_cmds(ix0, iy0, ix1, iy1, corners(&|rad, side| (rad - inner(side)).max(0.0)), true));
    }
    out
}

/// [`side_stroke_ring_cmds`] on a rectangle or frame that may have `cornerSmoothing` (E7.11): with smoothing on, the
/// ring follows the smoothed (squircle) outline the fill draws. Each corner of the outer and inner boxes is Figma's
/// smoothed corner at the radius grown (outer) or shrunk (inner) by the mean of its two sides' offsets, so it is
/// exact where the adjacent sides are equal (a uniform weight) and close where they differ (per-axis elliptical
/// corners have no smoothed counterpart). Without smoothing this is [`side_stroke_ring_cmds`].
pub fn side_stroke_ring_cmds_smoothed(
    width: f64,
    height: f64,
    corner_radius: Option<&Value>,
    corner_smoothing_v: Option<&Value>,
    sides: [f64; 4],
    align: Option<&str>,
) -> Vec<PathCmd> {
    if !rect_is_smoothed(corner_radius, corner_smoothing_v) {
        return side_stroke_ring_cmds(width, height, corner_radius, sides, align);
    }
    if sides.iter().all(|s| *s <= 0.0) || width <= 0.0 || height <= 0.0 {
        return Vec::new();
    }
    let (outer, inner): (fn(f64) -> f64, fn(f64) -> f64) = match align {
        Some("inside") => (|_| 0.0, |s| s),
        Some("outside") => (|s| s, |_| 0.0),
        _ => (|s| s / 2.0, |s| s / 2.0),
    };
    let smoothing = corner_smoothing(corner_smoothing_v);
    let [t, r, b, l] = sides;
    let radii = normalize_rounded_rect_radii(width, height, corner_radius);
    let adj = [(l, t), (r, t), (r, b), (l, b)];
    let corners = |f: &dyn Fn(f64, f64) -> f64| -> [f64; 4] {
        std::array::from_fn(|i| {
            let rad = radii[i];
            if rad <= 0.0 {
                0.0
            } else {
                f(rad, (adj[i].0 + adj[i].1) / 2.0)
            }
        })
    };
    let mut out = smoothed_box_cmds(
        -outer(l),
        -outer(t),
        width + outer(r),
        height + outer(b),
        corners(&|rad, side| rad + outer(side)),
        smoothing,
        false,
    );
    let (ix0, iy0, ix1, iy1) = (inner(l), inner(t), width - inner(r), height - inner(b));
    if ix1 - ix0 > 1e-9 && iy1 - iy0 > 1e-9 {
        out.extend(smoothed_box_cmds(
            ix0,
            iy0,
            ix1,
            iy1,
            corners(&|rad, side| (rad - inner(side)).max(0.0)),
            smoothing,
            true,
        ));
    }
    out
}

/// A box with smoothed corners `[tl, tr, br, bl]` as one closed sub-path: clockwise on screen, or
/// counter-clockwise when `reverse`.
fn smoothed_box_cmds(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    radii: [f64; 4],
    smoothing: f64,
    reverse: bool,
) -> Vec<PathCmd> {
    use moda_scene::corner_smoothing::{smoothed_rect_path, Seg};
    let segs = smoothed_rect_path(x1 - x0, y1 - y0, radii, smoothing);
    let at = |x: f64, y: f64| (x + x0, y + y0);
    if !reverse {
        return segs
            .into_iter()
            .map(|s| match s {
                Seg::Move(x, y) => {
                    let (x, y) = at(x, y);
                    PathCmd::Move { x, y }
                }
                Seg::Line(x, y) => {
                    let (x, y) = at(x, y);
                    PathCmd::Line { x, y }
                }
                Seg::Cubic { cp1, cp2, to } => {
                    let (x, y) = at(to.0, to.1);
                    PathCmd::Curve { x, y, cp1: at(cp1.0, cp1.1), cp2: at(cp2.0, cp2.1) }
                }
            })
            .collect();
    }
    // Reverse: walk the segments backwards, each from its end point to its start point.
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(segs.len());
    for s in &segs {
        pts.push(match *s {
            Seg::Move(x, y) | Seg::Line(x, y) => at(x, y),
            Seg::Cubic { to, .. } => at(to.0, to.1),
        });
    }
    let n = segs.len();
    let mut out = Vec::with_capacity(n);
    let (sx, sy) = pts[n - 1];
    out.push(PathCmd::Move { x: sx, y: sy });
    for i in (1..n).rev() {
        let (x, y) = pts[i - 1];
        out.push(match segs[i] {
            Seg::Cubic { cp1, cp2, .. } => PathCmd::Curve { x, y, cp1: at(cp2.0, cp2.1), cp2: at(cp1.0, cp1.1) },
            _ => PathCmd::Line { x, y },
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    /// Signed area (shoelace over the anchors; exact for straight rings) of each sub-path, summed.
    fn ring_area(cmds: &[PathCmd]) -> f64 {
        let mut area = 0.0;
        let mut sub: Vec<(f64, f64)> = Vec::new();
        let flush = |sub: &mut Vec<(f64, f64)>, area: &mut f64| {
            for i in 0..sub.len() {
                let (a, b) = (sub[i], sub[(i + 1) % sub.len()]);
                *area += a.0 * b.1 - b.0 * a.1;
            }
            sub.clear();
        };
        for c in cmds {
            match c {
                PathCmd::Move { x, y } => {
                    flush(&mut sub, &mut area);
                    sub.push((*x, *y));
                }
                PathCmd::Line { x, y } | PathCmd::Curve { x, y, .. } => sub.push((*x, *y)),
                _ => {}
            }
        }
        flush(&mut sub, &mut area);
        area / 2.0
    }

    #[test]
    fn side_ring_places_each_side_by_align() {
        // Inside: top 4, bottom 2 on 100 × 50 → the band is 100 × (4 + 2).
        let inside = side_stroke_ring_cmds(100.0, 50.0, None, [4.0, 0.0, 2.0, 0.0], Some("inside"));
        assert!((ring_area(&inside) - 600.0).abs() < 1e-9, "{}", ring_area(&inside));
        // Two sub-paths wound opposite ways (outer clockwise on screen, inner counter-clockwise).
        assert_eq!(inside.iter().filter(|c| matches!(c, PathCmd::Move { .. })).count(), 2);
        // Centre: each side straddles its edge; outside: all beyond it.
        let centre = side_stroke_ring_cmds(100.0, 50.0, None, [4.0, 0.0, 2.0, 0.0], None);
        assert!((ring_area(&centre) - 600.0).abs() < 1e-9);
        let outside = side_stroke_ring_cmds(100.0, 50.0, None, [0.0, 8.0, 0.0, 0.0], Some("outside"));
        assert!((ring_area(&outside) - 400.0).abs() < 1e-9);
        let xs: Vec<f64> = outside
            .iter()
            .filter_map(|c| match c {
                PathCmd::Move { x, .. } | PathCmd::Line { x, .. } => Some(*x),
                _ => None,
            })
            .collect();
        assert!(xs.iter().cloned().fold(f64::MIN, f64::max) == 108.0);
        // Nothing to draw.
        assert!(side_stroke_ring_cmds(100.0, 50.0, None, [0.0; 4], None).is_empty());
    }

    #[test]
    fn side_ring_corners_grow_out_and_shrink_in_per_axis() {
        // Radius 10, inside, left 4 / top 2: the inner top-left corner is 6 wide and 8 tall.
        let ring = side_stroke_ring_cmds(100.0, 50.0, Some(&json!(10)), [2.0, 0.0, 0.0, 4.0], Some("inside"));
        let moves: Vec<usize> = ring
            .iter()
            .enumerate()
            .filter_map(|(i, c)| matches!(c, PathCmd::Move { .. }).then_some(i))
            .collect();
        let inner = &ring[moves[1]..];
        // Reversed, the inner sub-path starts at the top-left corner's top point (x0 + rx, y0) = (4 + 6, 2).
        assert!(matches!(inner[0], PathCmd::Move { x, y } if (x - 10.0).abs() < 1e-9 && (y - 2.0).abs() < 1e-9), "{:?}", inner[0]);
        // Its first segment arrives at the corner's side point (x0, y0 + ry) = (4, 10).
        assert!(matches!(inner[1], PathCmd::Curve { x, y, .. } if (x - 4.0).abs() < 1e-9 && (y - 10.0).abs() < 1e-9), "{:?}", inner[1]);
    }

    #[test]
    fn side_ring_follows_the_smoothed_outline() {
        // E7.11 + E6.2: a smoothed rectangle's per-side stroke is the band between smoothed outlines. Uniform 4px
        // inside at radius 20 / 60% smoothing: the outer sub-path is the fill's own smoothed outline.
        let (r, s) = (json!(20), json!(0.6));
        let ring = side_stroke_ring_cmds_smoothed(100.0, 60.0, Some(&r), Some(&s), [4.0; 4], Some("inside"));
        let moves: Vec<usize> = ring
            .iter()
            .enumerate()
            .filter_map(|(i, c)| matches!(c, PathCmd::Move { .. }).then_some(i))
            .collect();
        assert_eq!(moves.len(), 2);
        let outline = rect_outline_cmds(100.0, 60.0, Some(&r), Some(&s), None);
        assert_eq!(&ring[..moves[1]], &outline[..]);
        // Opposite windings, so the nonzero fill leaves the inside empty; the band is about perimeter × 4.
        let (outer, inner) = (ring_area(&ring[..moves[1]]), ring_area(&ring[moves[1]..]));
        assert!(outer > 0.0 && inner < 0.0, "{outer} {inner}");
        // Without smoothing it is the plain ring.
        assert_eq!(
            side_stroke_ring_cmds_smoothed(100.0, 60.0, Some(&r), Some(&json!(0)), [4.0; 4], Some("inside")),
            side_stroke_ring_cmds(100.0, 60.0, Some(&r), [4.0; 4], Some("inside"))
        );
    }

    #[test]
    fn plain_rect_is_five_commands() {
        let cmds = rectangle_to_path_cmds(400.0, 300.0, None);
        assert_eq!(cmds.len(), 5);
        assert_eq!(cmds[0], PathCmd::Move { x: 400.0, y: 0.0 });
    }

    #[test]
    fn radii_shorthand_quirk() {
        // [tl, tr] → [tl, tr, tl, tr]; [tl, tr, br] → bl = tr.
        assert_eq!(
            normalize_rounded_rect_radii(200.0, 200.0, Some(&json!([32, 16]))),
            [32.0, 16.0, 32.0, 16.0]
        );
        assert_eq!(
            normalize_rounded_rect_radii(200.0, 200.0, Some(&json!([10, 20, 30]))),
            [10.0, 20.0, 30.0, 20.0]
        );
    }

    #[test]
    fn per_corner_clamp_then_proportional_scaling() {
        // Step 1: each corner clamps to min(w,h)/2 = 80 → [80, 80, 0, 0];
        // step 2: top edge sum 160 fits exactly (scale 1) → no rescale.
        let r = normalize_rounded_rect_radii(160.0, 400.0, Some(&json!([100, 100, 0, 0])));
        assert_eq!(r, [80.0, 80.0, 0.0, 0.0]);
        // Adjacent overlap on the *long* edge does scale: 300+300 on width 400
        // (clamped first to 100 = min(400,200)/2) → no overlap after clamp.
        // Overlap that survives the clamp: w=400, h=300, [200, 200, 0, 0] →
        // clamp to 150 each → top sum 300 < 400 → scale 1.
        let r2 = normalize_rounded_rect_radii(400.0, 300.0, Some(&json!([200, 200, 0, 0])));
        assert_eq!(r2, [150.0, 150.0, 0.0, 0.0]);
    }

    #[test]
    fn full_ellipse_is_four_beziers() {
        let cmds = ellipse_to_path_cmds(200.0, 100.0, None, None, None);
        assert_eq!(cmds.len(), 5);
        assert_eq!(cmds[0], PathCmd::Move { x: 200.0, y: 50.0 });
    }

    #[test]
    fn ring_has_two_contours() {
        let cmds = ellipse_to_path_cmds(200.0, 200.0, None, None, Some(0.5));
        let moves = cmds
            .iter()
            .filter(|c| matches!(c, PathCmd::Move { .. }))
            .count();
        assert_eq!(moves, 2);
    }

    #[test]
    fn rect_outline_switches_to_smoothed_corners() {
        // No smoothing → the plain KAPPA outline, bit for bit.
        let plain = rectangle_to_path_cmds(200.0, 100.0, Some(&json!(20)));
        assert_eq!(rect_outline_cmds(200.0, 100.0, Some(&json!(20)), None, None), plain);
        assert_eq!(rect_outline_cmds(200.0, 100.0, Some(&json!(20)), Some(&json!(0)), None), plain);
        // Smoothing on a sharp rect changes nothing either.
        assert_eq!(
            rect_outline_cmds(200.0, 100.0, None, Some(&json!(0.6)), None),
            rectangle_to_path_cmds(200.0, 100.0, None)
        );
        // 60%: the corner consumes 1.6 R of the top edge, per-corner radii honoured.
        let s = rect_outline_cmds(200.0, 100.0, Some(&json!([10, 20, 0, 0])), Some(&json!(0.6)), None);
        assert_eq!(s[0], PathCmd::Move { x: 200.0 - 32.0, y: 0.0 });
        assert!(s.contains(&PathCmd::Line { x: 200.0, y: 100.0 })); // br sharp
        assert!(s.contains(&PathCmd::Line { x: 0.0, y: 100.0 })); // bl sharp
        // Out-of-range values clamp.
        assert_eq!(corner_smoothing(Some(&json!(3))), 1.0);
        assert_eq!(corner_smoothing(Some(&json!(-1))), 0.0);
    }
}
