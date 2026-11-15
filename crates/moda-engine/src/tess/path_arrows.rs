//! E6.1: arrowheads on open paths (design-editor.md §7.19: arrowheads belong to the stroke and take its paint).
//!
//! A stroke's `startArrow` / `endArrow` draw on any OPEN path (`closed` absent or false): the start arrowhead at
//! the first sub-path's first anchor, the end arrowhead at the last sub-path's last anchor, each pointing along
//! the path's end tangent (a retracted handle falls back to the next control point, then the next anchor). The
//! shapes are the line's (`line_to_path`: triangle, open caret, reverse triangle, diamond, circle) and so is the
//! stroke's set-back under each filled one (`arrow_trim_distance`: the arrowhead's depth, plus half the weight under
//! a round or square cap), cut from the stroke by arc length so a curve keeps its shape. Under an open caret the
//! stroke runs on to the tip.
//!
//! Size: a path has no `pointerLength` / `pointerWidth` of its own; its arrowheads grow with the stroke, as
//! Figma's and Illustrator's do: `max(10, 4 × weight) × arrowScale` long and wide (10 px, a line's default,
//! up to a 2.5 px weight). A stroke with a brush (art, scatter, pattern or calligraphic) draws no arrowheads.
//!
//! The TypeScript mirror (exports, bounds, hit testing) is `packages/editor/src/core/scene/pathArrowheads.ts`.

use super::line_to_path::{
    arrow_trim_distance, filled_arrow_shape_points, open_arrow_stroke_points, ArrowShape,
};
use super::moda_engine_path_types::PathCmd;
use super::{cut_open_path_ends, open_path_ends, PathEnd};
use moda_scene::paint::Stroke;

/// The smallest arrowhead on a path (px): a line's default pointer size.
pub const PATH_ARROW_MIN_SIZE: f64 = 10.0;
/// Arrowhead length and width per unit of stroke weight past the minimum.
pub const PATH_ARROW_WEIGHT_FACTOR: f64 = 4.0;

/// A path's arrowhead length and width (they are equal) for a stroke of `width` and `arrow_scale`.
pub fn path_arrow_size(width: f64, arrow_scale: Option<f64>) -> f64 {
    let scale = arrow_scale
        .filter(|s| s.is_finite() && *s > 0.0)
        .unwrap_or(1.0);
    PATH_ARROW_MIN_SIZE.max(PATH_ARROW_WEIGHT_FACTOR * width.max(0.0)) * scale
}

/// A stroke's arrowhead shapes (start, end), `None` where it has none (or an unknown shape).
pub fn stroke_arrows(stroke: &Stroke) -> (Option<ArrowShape>, Option<ArrowShape>) {
    if stroke.brush.as_ref().is_some_and(|b| !b.is_null()) || stroke.width <= 0.0 {
        return (None, None);
    }
    (
        stroke.start_arrow.as_deref().and_then(ArrowShape::from_str),
        stroke.end_arrow.as_deref().and_then(ArrowShape::from_str),
    )
}

/// Whether `stroke` draws an arrowhead on an open path.
pub fn stroke_has_arrows(stroke: &Stroke) -> bool {
    let (s, e) = stroke_arrows(stroke);
    s.is_some() || e.is_some()
}

/// One stroke's arrowheads on an open path, in the path's own coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct PathArrows {
    /// The path the stroke draws: its ends cut back under the arrowheads.
    pub body: Vec<PathCmd>,
    /// Filled arrowheads (closed sub-paths), painted with the stroke's paint under the nonzero rule.
    pub filled: Vec<PathCmd>,
    /// Open carets (open sub-paths), stroked with the stroke's weight, caps and joins.
    pub open: Vec<PathCmd>,
    /// The arrowheads' length and width.
    pub size: f64,
}

fn head(
    shape: ArrowShape,
    end: &PathEnd,
    size: f64,
    filled: &mut Vec<PathCmd>,
    open: &mut Vec<PathCmd>,
) {
    let [ux, uy] = end.out;
    let (px, py) = (-uy, ux);
    let [tx, ty] = end.point;
    if shape == ArrowShape::Open {
        open.extend(open_arrow_stroke_points(tx, ty, ux, uy, px, py, size, size));
    } else if let Some(pts) = filled_arrow_shape_points(shape, tx, ty, ux, uy, px, py, size, size) {
        filled.extend(pts);
    }
}

/// `stroke`'s arrowheads on the path `cmds` (the drawn geometry, corner rounding applied), or `None` when the
/// path is closed, the stroke has no arrowhead, or the path has no length.
pub fn path_arrows(cmds: &[PathCmd], closed: bool, stroke: &Stroke) -> Option<PathArrows> {
    if closed {
        return None;
    }
    let (start, end) = stroke_arrows(stroke);
    run_arrows(
        cmds,
        start,
        end,
        path_arrow_size(stroke.width, stroke.arrow_scale),
        stroke.cap.as_deref(),
        stroke.width,
    )
}

/// Arrowheads `start` / `end` of `size` on the open run `cmds` stroked `width` wide with `cap` (the set-back under
/// a filled head grows with a round or square cap), or `None` when there is none or the run has no length. The
/// shared step of a path's stroke arrows and a vector network's arrow caps (`tess::network`).
pub fn run_arrows(
    cmds: &[PathCmd],
    start: Option<ArrowShape>,
    end: Option<ArrowShape>,
    size: f64,
    cap: Option<&str>,
    width: f64,
) -> Option<PathArrows> {
    if start.is_none() && end.is_none() {
        return None;
    }
    let (start_end, end_end) = open_path_ends(cmds)?;
    // An open caret is stroked like the path: the stroke runs on into its apex (as Figma's and Illustrator's line
    // arrows do), the caret's join covering the stroke's end.
    let trim = |shape: Option<ArrowShape>| match shape {
        None | Some(ArrowShape::Open) => 0.0,
        Some(s) => arrow_trim_distance(s, size, size, cap, width),
    };
    let body = cut_open_path_ends(cmds, trim(start), trim(end));
    let (mut filled, mut open) = (Vec::new(), Vec::new());
    if let Some(s) = start {
        head(s, &start_end, size, &mut filled, &mut open);
    }
    if let Some(s) = end {
        head(s, &end_end, size, &mut filled, &mut open);
    }
    Some(PathArrows {
        body,
        filled,
        open,
        size,
    })
}

/// How far a stroke's arrowheads reach past the path's anchors (0 without any): half their width across the
/// path, or the full size for a centred diamond or circle (they straddle the end). Bounds pad by it.
pub fn path_arrow_reach(stroke: &Stroke) -> f64 {
    let (s, e) = stroke_arrows(stroke);
    if s.is_none() && e.is_none() {
        return 0.0;
    }
    let size = path_arrow_size(stroke.width, stroke.arrow_scale);
    // An open caret is stroked: its outer corner reaches half a weight further.
    size / 2.0 + stroke.width / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    fn stroke(v: moda_scene::value::Value) -> Stroke {
        let mut base =
            json!({"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 2});
        for (k, val) in v.as_object().unwrap() {
            base[k] = val.clone();
        }
        moda_scene::value::from_value(base).unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    fn end_point(cmds: &[PathCmd]) -> (f64, f64) {
        match cmds.last().unwrap() {
            PathCmd::Move { x, y } | PathCmd::Line { x, y } | PathCmd::Curve { x, y, .. } => {
                (*x, *y)
            }
            PathCmd::Quadratic { x, y, .. } => (*x, *y),
            PathCmd::Arc { x, y, .. } => (*x, *y),
        }
    }

    #[test]
    fn size_grows_with_the_weight_past_the_minimum() {
        assert!(close(path_arrow_size(1.0, None), 10.0));
        assert!(close(path_arrow_size(2.5, None), 10.0));
        assert!(close(path_arrow_size(5.0, None), 20.0));
        assert!(close(path_arrow_size(5.0, Some(1.5)), 30.0));
        assert!(close(path_arrow_size(1.0, Some(-1.0)), 10.0));
    }

    #[test]
    fn a_straight_path_gets_a_triangle_at_its_end_and_is_cut_under_it() {
        let cmds = vec![
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Line { x: 100.0, y: 0.0 },
        ];
        let a = path_arrows(&cmds, false, &stroke(json!({"endArrow": "triangle"}))).unwrap();
        assert!(close(a.size, 10.0));
        // Butt cap: the body stops at the triangle's base.
        let (x, y) = end_point(&a.body);
        assert!(close(x, 90.0) && close(y, 0.0), "{x},{y}");
        // Tip on the anchor, base 10 back, 10 wide.
        let pts: Vec<(f64, f64)> = a
            .filled
            .iter()
            .map(|c| match c {
                PathCmd::Move { x, y } | PathCmd::Line { x, y } => (*x, *y),
                _ => panic!(),
            })
            .collect();
        assert_eq!(pts.len(), 3);
        assert!(close(pts[0].0, 100.0) && close(pts[0].1, 0.0));
        assert!(pts[1..]
            .iter()
            .all(|p| close(p.0, 90.0) && close(p.1.abs(), 5.0)));
        assert!(a.open.is_empty());
    }

    #[test]
    fn a_curve_points_along_its_end_tangent_and_keeps_its_curve() {
        // The end handle comes in from straight above the end anchor: the arrowhead points down.
        let cmds = vec![
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Curve {
                x: 100.0,
                y: 100.0,
                cp1: (60.0, 0.0),
                cp2: (100.0, 40.0),
            },
        ];
        let a = path_arrows(
            &cmds,
            false,
            &stroke(json!({"endArrow": "triangle", "startArrow": "circle", "cap": "round"})),
        )
        .unwrap();
        // Round cap: set back by the depth plus half the weight (10 + 1 at the end, 5 + 1 at the start).
        assert!(matches!(a.body.last(), Some(PathCmd::Curve { .. })));
        let (x, y) = end_point(&a.body);
        assert!(
            (x - 100.0).abs() < 1.0 && (100.0 - y - 11.0).abs() < 0.2,
            "{x},{y}"
        );
        match a.body[0] {
            PathCmd::Move { x, y } => assert!((x - 6.0).abs() < 0.2 && y.abs() < 0.2, "{x},{y}"),
            _ => panic!(),
        }
        // The triangle's base is 10 above its tip, level.
        let base: Vec<(f64, f64)> = a
            .filled
            .iter()
            .filter_map(|c| match c {
                PathCmd::Line { x, y } if close(*y, 90.0) => Some((*x, *y)),
                _ => None,
            })
            .collect();
        assert_eq!(base.len(), 2, "{:?}", a.filled);
    }

    #[test]
    fn closed_paths_unknown_shapes_and_brushes_draw_none() {
        let cmds = vec![
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Line { x: 100.0, y: 0.0 },
        ];
        assert!(path_arrows(&cmds, true, &stroke(json!({"endArrow": "triangle"}))).is_none());
        assert!(path_arrows(&cmds, false, &stroke(json!({"endArrow": "bogus"}))).is_none());
        assert!(path_arrows(
            &cmds,
            false,
            &stroke(json!({"endArrow": "triangle", "brush": {"brushId": "b"}}))
        )
        .is_none());
        assert!(path_arrows(
            &cmds,
            false,
            &stroke(json!({"endArrow": "triangle", "width": 0}))
        )
        .is_none());
    }

    #[test]
    fn multi_subpath_arrows_sit_at_the_first_start_and_the_last_end() {
        let cmds = vec![
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Line { x: 50.0, y: 0.0 },
            PathCmd::Move { x: 0.0, y: 50.0 },
            PathCmd::Line { x: 0.0, y: 150.0 },
        ];
        let a = path_arrows(
            &cmds,
            false,
            &stroke(json!({"startArrow": "open", "endArrow": "triangle"})),
        )
        .unwrap();
        assert_eq!(a.filled.len(), 3);
        // One caret, one triangle. The body: the first sub-path whole (the stroke runs on into an open caret),
        // the last cut back 10 under the triangle.
        assert_eq!(a.open.len(), 3);
        assert!(matches!(a.body[0], PathCmd::Move { x, y } if close(x, 0.0) && close(y, 0.0)));
        let (x, y) = end_point(&a.body);
        assert!(close(x, 0.0) && close(y, 140.0));
        // The start caret's tip is at the origin, pointing left (-x).
        assert!(matches!(a.open[1], PathCmd::Line { x, y } if close(x, 0.0) && close(y, 0.0)));
        assert!(matches!(a.open[0], PathCmd::Move { x, .. } if close(x, 10.0)));
        assert!(matches!(a.body[1], PathCmd::Line { x, .. } if close(x, 50.0)));
    }

    #[test]
    fn a_path_shorter_than_its_arrowheads_draws_only_the_heads() {
        let cmds = vec![
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Line { x: 12.0, y: 0.0 },
        ];
        let a = path_arrows(
            &cmds,
            false,
            &stroke(json!({"startArrow": "triangle", "endArrow": "triangle"})),
        )
        .unwrap();
        assert!(a.body.is_empty());
        assert_eq!(a.filled.len(), 6);
    }
}
