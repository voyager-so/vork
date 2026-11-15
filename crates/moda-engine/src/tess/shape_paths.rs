//! M2 — star / polygon inscribed paths + the perspective-pattern quad.
//!
//! Faithful ports of:
//! - `utils/starToPath.ts` (`starToPathPoints`) — vertices from
//!   `utils/geometryHelpers.ts` `getStretchedStarVertices`
//! - `utils/polygonToPath.ts` (`polygonToPathPoints`) —
//!   `getStretchedPolygonVertices`
//! - the converter call sites in
//!   `renderers/webgpu/utils/shapeToPathNodeUtils.ts` (`convertStarToPathNode`
//!   / `convertPolygonToPathNode` pass center (w/2, h/2), rotation 0)
//! - `getPerspectivePatternMetadata` + `quadToLinePathPoints`
//!   (shapeToPathNodeUtils.ts:218-261) with the validity predicates from
//!   `core/scene/projectiveTransform.ts` (`isValidConvexQuad`,
//!   `cornersToHomography`/`solveLinearSystem`).
//! - the `distort` CPU warp (`warp_path_cmds`): flatten a converted
//!   outline and project it through the node's local-px homography (from
//!   `crate::paint::homography`). Additive next to the legacy corner-pin quad
//!   above — the legacy path is untouched.
//!
//! All math in f64. Corner rounding is the shared
//! `moda_scene::path_geom::round_corners` rounder (the native half of
//! `core/geometry/roundPathCorners.ts`), NOT a local KAPPA construction.

use super::moda_engine_path_types::PathCmd;
use super::path_cmd_from_blend_point;
#[cfg(test)]
use moda_scene::shape_outline::ellipse_vertex_at_angle;
pub use moda_scene::shape_outline::{MAX_SHAPE_VERTICES, MIN_VISIBLE_CORNER_RADIUS};

type P = [f64; 2];

// The star / polygon converters live in `moda_scene::shape_outline` (the live
// geometry effects read shape outlines at scene resolution); these wrap them
// in the tessellator's command type.

fn to_cmds(points: Vec<moda_scene::blend::BlendPathPoint>) -> Vec<PathCmd> {
    points.iter().map(path_cmd_from_blend_point).collect()
}

/// `roundClosedVertexRing` (`utils/path/closedShapePath.ts`): round every corner
/// of a closed vertex ring through the shared rounder. Radii below
/// `MIN_VISIBLE_CORNER_RADIUS` stay sharp.
pub fn rounded_ring_path(vertices: &[P], corner_radius: f64) -> Vec<PathCmd> {
    to_cmds(moda_scene::shape_outline::rounded_ring(
        vertices,
        corner_radius,
    ))
}

/// `convertStarToPathNode` geometry (`starToPathPoints` with x = w/2,
/// y = h/2, rotation 0). `inner_radius_ratio` non-finite → 0.5 (the TS
/// `Number.isFinite` guard); clamped [0,1]. `corner_radius` is the TS
/// `star.cornerRadius || 0` value.
pub fn star_to_path_cmds(
    width: f64,
    height: f64,
    num_points: f64,
    inner_radius_ratio: Option<f64>,
    corner_radius: f64,
) -> Vec<PathCmd> {
    to_cmds(moda_scene::shape_outline::star_outline(
        width,
        height,
        num_points,
        inner_radius_ratio,
        corner_radius,
    ))
}

/// `convertPolygonToPathNode` geometry (`polygonToPathPoints` with center
/// (w/2, h/2), rotation 0). Empty vertices (non-finite sides) → empty path.
pub fn polygon_to_path_cmds(
    width: f64,
    height: f64,
    sides: f64,
    corner_radius: f64,
) -> Vec<PathCmd> {
    to_cmds(moda_scene::shape_outline::polygon_outline(
        width,
        height,
        sides,
        corner_radius,
    ))
}

// ---------------------------------------------------------------------------
// Distort warp: CPU pre-warp of a converted shape outline through
// the node's local-px homography. Nodes without a valid `distort` never reach
// these functions.
// ---------------------------------------------------------------------------

/// Segments per curve command when flattening a warped outline. Chosen by the
/// donor spike as visually smooth for the warp range; homographies
/// map straight segments to straight segments, so only curves need sampling.
const DISTORT_CURVE_SEGMENTS: usize = 32;

/// Flatten path commands into a polyline for the distort warp: cubic/quadratic
/// curves sample uniformly at t = s/N; arcs never occur in shape-converter
/// output and flatten to their endpoint defensively.
fn flatten_cmds_to_polyline(cmds: &[PathCmd], closed: bool, segments: usize) -> Vec<P> {
    let mut result: Vec<P> = Vec::new();
    let mut current: Option<P> = None;
    for cmd in cmds {
        match cmd {
            PathCmd::Move { x, y } => {
                current = Some([*x, *y]);
                if result.is_empty() {
                    result.push([*x, *y]);
                }
            }
            PathCmd::Line { x, y } | PathCmd::Arc { x, y, .. } => {
                if current.is_none() {
                    current = Some([*x, *y]);
                    result.push([*x, *y]);
                } else {
                    result.push([*x, *y]);
                    current = Some([*x, *y]);
                }
            }
            PathCmd::Curve { x, y, cp1, cp2 } => {
                if let Some(c) = current {
                    let (p1, p2, p3, p4) = (c, [cp1.0, cp1.1], [cp2.0, cp2.1], [*x, *y]);
                    for sidx in 1..=segments {
                        let t = sidx as f64 / segments as f64;
                        let it = 1.0 - t;
                        let px = it * it * it * p1[0]
                            + 3.0 * it * it * t * p2[0]
                            + 3.0 * it * t * t * p3[0]
                            + t * t * t * p4[0];
                        let py = it * it * it * p1[1]
                            + 3.0 * it * it * t * p2[1]
                            + 3.0 * it * t * t * p3[1]
                            + t * t * t * p4[1];
                        result.push([px, py]);
                    }
                    current = Some([*x, *y]);
                } else {
                    current = Some([*x, *y]);
                    result.push([*x, *y]);
                }
            }
            PathCmd::Quadratic { x, y, cp } => {
                if let Some(c) = current {
                    let (p1, p2, p3) = (c, [cp.0, cp.1], [*x, *y]);
                    for sidx in 1..=segments {
                        let t = sidx as f64 / segments as f64;
                        let it = 1.0 - t;
                        let px = it * it * p1[0] + 2.0 * it * t * p2[0] + t * t * p3[0];
                        let py = it * it * p1[1] + 2.0 * it * t * p2[1] + t * t * p3[1];
                        result.push([px, py]);
                    }
                    current = Some([*x, *y]);
                } else {
                    current = Some([*x, *y]);
                    result.push([*x, *y]);
                }
            }
        }
    }
    if closed && result.len() > 1 {
        // Ensure closure by appending the first point if needed — EXACT equality
        // (a final curve endpoint that misses the start by float noise gets a
        // closing point, which the caller's near-duplicate pop then removes).
        let first = result[0];
        let last = result[result.len() - 1];
        if first[0] != last[0] || first[1] != last[1] {
            result.push(first);
        }
    }
    result
}

/// Split a converted outline into its subpath contours (a new contour starts at
/// every `Move`). Ring/donut outlines are two contours whose winding produces
/// the hole — the warp must keep them separate, not bridge them.
fn split_cmds_into_contours(cmds: &[PathCmd]) -> Vec<Vec<PathCmd>> {
    let mut contours: Vec<Vec<PathCmd>> = Vec::new();
    let mut current: Vec<PathCmd> = Vec::new();
    for cmd in cmds {
        if matches!(cmd, PathCmd::Move { .. }) && !current.is_empty() {
            contours.push(std::mem::take(&mut current));
        }
        current.push(cmd.clone());
    }
    if !current.is_empty() {
        contours.push(current);
    }
    contours
}

/// CPU-warp a converted shape outline through the node's local-px `distort`
/// homography: flatten each contour's curves, drop the closing
/// duplicate point (the PathCmd list is implicitly closed), and project every
/// point. Homographies map straight segments to straight segments, so the
/// flattened outline warps exactly. Returns None when any contour fails to
/// warp (degenerate denominators are rejected at validation time — belt and
/// braces here); the caller keeps the unwarped geometry AND drops the warp
/// metadata so pattern UVs are never inverse-mapped over unwarped geometry.
pub fn warp_path_cmds(
    cmds: &[PathCmd],
    local: &crate::paint::homography::ProjectiveTransform,
) -> Option<Vec<PathCmd>> {
    use crate::paint::homography::{apply_projective_to_point, Point2D};

    let mut out: Vec<PathCmd> = Vec::new();
    for contour in split_cmds_into_contours(cmds) {
        let mut polyline = flatten_cmds_to_polyline(&contour, true, DISTORT_CURVE_SEGMENTS);
        // Drop trailing (near-)duplicates of the start point: the flattened
        // contour usually ends on an exact closing duplicate, and a final
        // curve endpoint that misses the start by float noise leaves TWO
        // trailing near-starts (the noisy endpoint + the appended exact
        // closure) — pop until the tail is a real segment, so the implicit
        // close never degrades into a zero-length stroke edge.
        while polyline.len() > 1 {
            let first = polyline[0];
            let last = polyline[polyline.len() - 1];
            if (first[0] - last[0]).abs() < 1e-9 && (first[1] - last[1]).abs() < 1e-9 {
                polyline.pop();
            } else {
                break;
            }
        }
        if polyline.len() < 3 {
            return None;
        }

        for (i, p) in polyline.iter().enumerate() {
            let projected = apply_projective_to_point(Point2D { x: p[0], y: p[1] }, local)?;
            out.push(if i == 0 {
                PathCmd::Move {
                    x: projected.x,
                    y: projected.y,
                }
            } else {
                PathCmd::Line {
                    x: projected.x,
                    y: projected.y,
                }
            });
        }
    }
    (out.len() >= 3).then_some(out)
}

#[cfg(test)]
mod warp_tests {
    use super::*;
    use crate::paint::homography::ProjectiveTransform;

    const IDENTITY: ProjectiveTransform = ProjectiveTransform {
        h11: 1.0,
        h12: 0.0,
        h13: 0.0,
        h21: 0.0,
        h22: 1.0,
        h23: 0.0,
        h31: 0.0,
        h32: 0.0,
        h33: 1.0,
    };

    fn rect_cmds(w: f64, h: f64) -> Vec<PathCmd> {
        vec![
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Line { x: w, y: 0.0 },
            PathCmd::Line { x: w, y: h },
            PathCmd::Line { x: 0.0, y: h },
            PathCmd::Line { x: 0.0, y: 0.0 },
        ]
    }

    #[test]
    fn warp_drops_closing_duplicate_and_keeps_corner_count() {
        let out = warp_path_cmds(&rect_cmds(100.0, 50.0), &IDENTITY).unwrap();
        // 5 input points with a closing duplicate → 4 output points, one Move.
        assert_eq!(out.len(), 4);
        assert!(matches!(out[0], PathCmd::Move { .. }));
        assert!(out[1..].iter().all(|c| matches!(c, PathCmd::Line { .. })));
    }

    #[test]
    fn warp_preserves_subpath_contours() {
        // Two concentric square contours (a ring): the warp must keep them as
        // two Move-led contours, not bridge them into one keyhole polyline.
        let mut cmds = rect_cmds(100.0, 100.0);
        cmds.extend(vec![
            PathCmd::Move { x: 25.0, y: 25.0 },
            PathCmd::Line { x: 75.0, y: 25.0 },
            PathCmd::Line { x: 75.0, y: 75.0 },
            PathCmd::Line { x: 25.0, y: 75.0 },
            PathCmd::Line { x: 25.0, y: 25.0 },
        ]);
        let lean = ProjectiveTransform {
            h31: 0.002,
            ..IDENTITY
        };
        let out = warp_path_cmds(&cmds, &lean).unwrap();
        let moves = out
            .iter()
            .filter(|c| matches!(c, PathCmd::Move { .. }))
            .count();
        assert_eq!(moves, 2);
    }

    #[test]
    fn warp_flattens_curves_and_projects_them() {
        // A single cubic (quarter-ish arc) flattens to DISTORT_CURVE_SEGMENTS
        // line segments, all projected. With the identity transform the
        // endpoints must be preserved exactly.
        let cmds = vec![
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Curve {
                x: 100.0,
                y: 100.0,
                cp1: (55.0, 0.0),
                cp2: (100.0, 45.0),
            },
            PathCmd::Line { x: 0.0, y: 100.0 },
            PathCmd::Line { x: 0.0, y: 0.0 },
        ];
        let out = warp_path_cmds(&cmds, &IDENTITY).unwrap();
        // 1 (Move) + 32 (flattened curve) + 1 (line) — closing duplicate dropped.
        assert_eq!(out.len(), 1 + DISTORT_CURVE_SEGMENTS + 1);
        let PathCmd::Move { x, y } = out[0] else {
            panic!("first cmd must be Move");
        };
        assert!(x.abs() < 1e-12 && y.abs() < 1e-12);
    }

    #[test]
    fn warp_falls_back_on_degenerate_contours() {
        // Fewer than 3 polyline points in any contour → None (caller keeps the
        // unwarped geometry).
        let degenerate = vec![
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Line { x: 10.0, y: 0.0 },
        ];
        assert!(warp_path_cmds(&degenerate, &IDENTITY).is_none());
    }
}

#[cfg(test)]
mod ring_tests {
    use super::*;

    /// Mirrors `star_five_points_starts_at_top` from the deleted
    /// `backdrop/star_polygon.rs`, whose coverage lane now calls this module.
    #[test]
    fn star_five_points_starts_at_top() {
        let cmds = star_to_path_cmds(200.0, 200.0, 5.0, Some(0.5), 0.0);
        // M(v0) + L(v1..v9) + closing L(v0) = 11 commands.
        assert_eq!(cmds.len(), 11);
        match cmds[0] {
            PathCmd::Move { x, y } => {
                assert!((x - 100.0).abs() < 1e-9);
                assert!(y.abs() < 1e-9); // top of the 200-box
            }
            _ => panic!("first cmd must be Move"),
        }
    }

    /// Mirrors `polygon_vertices_inscribed` from the deleted
    /// `backdrop/star_polygon.rs`. Radius 0 must stay the untouched sharp ring.
    #[test]
    fn polygon_radius_zero_is_the_sharp_inscribed_ring() {
        let cmds = polygon_to_path_cmds(100.0, 100.0, 6.0, 0.0);
        // M(v0) + L(v1..v5) + closing L(v0) = 7 commands, no curves.
        assert_eq!(cmds.len(), 7);
        assert!(!cmds.iter().any(|c| matches!(c, PathCmd::Curve { .. })));
        for c in &cmds {
            let (x, y) = match c {
                PathCmd::Move { x, y } | PathCmd::Line { x, y } => (*x, *y),
                _ => continue,
            };
            let d = ((x - 50.0f64).powi(2) + (y - 50.0f64).powi(2)).sqrt();
            assert!((d - 50.0).abs() < 1e-9, "vertex not on ellipse: {d}");
        }
    }

    /// A regular hexagon turns 60° at every vertex, so a radius-20 fillet is
    /// tangent `20·tan(30°) ≈ 11.547` from the vertex — NOT 20, which is what the
    /// old KAPPA rounder trimmed regardless of the turning angle.
    #[test]
    fn hexagon_fillet_trims_the_true_tangent_length() {
        let width = 400.0;
        let radius = 20.0;
        let cmds = polygon_to_path_cmds(width, width, 6.0, radius);
        assert!(cmds.iter().any(|c| matches!(c, PathCmd::Curve { .. })));

        // Vertex 1 of the ring, in the same translated space the converter emits.
        let rx = width / 2.0;
        let vertex = ellipse_vertex_at_angle(360.0 / 6.0 - 90.0, rx, rx);
        let vertex = [vertex[0] + rx, vertex[1] + rx];

        let expected = radius * (std::f64::consts::PI / 6.0).tan();
        // The fillet at vertex 1 starts at the end of the first emitted segment.
        let start = match cmds[1] {
            PathCmd::Line { x, y } => [x, y],
            ref other => panic!("expected a trimmed line before the first fillet, got {other:?}"),
        };
        let d = ((start[0] - vertex[0]).powi(2) + (start[1] - vertex[1]).powi(2)).sqrt();
        assert!(
            (d - expected).abs() < 1e-6,
            "fillet start is {d} from the vertex, expected {expected}"
        );
    }
}

#[cfg(test)]
mod vertex_count_tests {
    use super::*;

    #[test]
    fn absurd_counts_are_clamped_not_fatal() {
        for n in [1e20, 1e9, f64::MAX, -5.0] {
            let star = star_to_path_cmds(10.0, 10.0, n, None, 0.0);
            let poly = polygon_to_path_cmds(10.0, 10.0, n, 0.0);
            assert!(
                star.len() <= 2 * MAX_SHAPE_VERTICES + 2,
                "{n}: {}",
                star.len()
            );
            assert!(poly.len() <= MAX_SHAPE_VERTICES + 2, "{n}: {}", poly.len());
            assert!(!poly.is_empty() && !star.is_empty());
        }
        assert!(star_to_path_cmds(10.0, 10.0, f64::NAN, None, 0.0).is_empty());
        assert!(polygon_to_path_cmds(10.0, 10.0, f64::NAN, 0.0).is_empty());
        // Ordinary counts are untouched.
        assert_eq!(polygon_to_path_cmds(10.0, 10.0, 6.0, 0.0).len(), 7);
    }
}
