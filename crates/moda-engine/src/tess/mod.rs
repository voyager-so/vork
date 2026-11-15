//! tess/ — path tessellation. OWNED BY M2 (PLAN §0 rule 4).
//!
//! M2 replaced the M1 placeholder WHOLESALE with the faithful port of
//! `renderers/webgpu/geometry/WebGPUPathTessellator.ts`, gated by the vertex
//! oracle (`fixtures/oracle/*/tessellation.json`, tol 1e-4 via
//! `tools/oracle/diff_vertices.py` — stroke vertices; fills are lyon's, not
//! the TS ear clipper's, and are pinned by area/coverage tests + goldens):
//!   - `tessellator.rs` — De Casteljau flattening, lyon sweep-line fills
//!     (evenodd/nonzero on self-intersecting outlines, holes from winding),
//!     hard caps WITH truncation behavior, stroke bands
//!     (align/caps/joins/miter-4→bevel/round 22.5°), dash routing.
//!     Deliberate departures from the TS port (Illustrator comparison,
//!     2026-09-24; mirrored in `WebGPUPathTessellator.ts` except the bands):
//!       - Strokes and fills flatten to half a device pixel (F4.1): 0.5 units
//!         at 1× (as before), halved per zoom band above and doubled per band
//!         below (zoom × dpr on the canvas, the export scale for exports;
//!         bands clamped to 1/16×..64×), so curved fills — placed art-brush
//!         artwork included — keep their curves at 2× and zoomed-out curves
//!         cost fewer vertices. Tessellations are cached per node and band.
//!         A curved stroke or fill that would hit a cap at its band is redone
//!         coarser, band by band, and last with the pre-band flattening
//!         (`stroke_within_caps`, `fill_path`), so zoom never truncates or
//!         drops anything main drew.
//!       - Every stroke piece also obeys a turn limit derived from the stroke's
//!         outer offset (half the width centred, the whole width inside /
//!         outside), so wide curved strokes have no visible facets. This one
//!         changes 1× output for wide curved strokes (a few more edges; golden
//!         deltas in #87).
//!       - Dashes walk the flattened curve (each curve's pieces keep its arc
//!         length) instead of chords between dash ends.
//!       - Inner joins fill to the legs' offset crossing (the miter limit only
//!         clips the outer tip); past a short leg they pivot through the vertex.
//!     `fixtures/oracle/strokes-dashes.json/tessellation.json` was regenerated
//!     from `examples/dump_tess` for the dash change; the other oracle fixtures
//!     are unchanged.
//!   - `arc.rs` — arcs → cubics per `core/svg/arcUtils.ts`.
//!   - `dash.rs` — arc-length walk (`utils/lineToPath.ts`), strokeWidth/2
//!     pattern scaling, odd-pattern doubling; dash kills joins/align/closure.
//!   - `shape_paths.rs` — star/polygon inscribed paths + the distort outline warp.
//!   - `line_to_path.rs` — line bodies + arrowheads
//!     (`WebGPULineToPath.ts`, `core/scene/arrowheads.ts`).
//!
//! INTERFACE (frozen with seams.rs): `tessellate_combined` keeps the M1
//! signature and returns `seams::TessellatedGeometry` (fill vertices first,
//! then stroke; vertex_types 0.0 = fill / 1.0 = stroke).
//! `tessellate_combined_with_dash` is the M2 superset (additive — see the
//! SEAMS.md change log).

mod arc;
mod bounds;
pub mod connector_text;
mod dash;
pub mod line_to_path;
pub mod network;
pub mod outline;
pub mod path_arrows;
pub mod placed;
pub mod shape_paths;
pub mod stroke_param;
mod tessellator;
pub mod trim_axis;

pub use bounds::path_bounds;
pub use trim_axis::TrimBy;
pub use tessellator::{
    tessellate_combined_cached,
    cut_open_path_ends, is_trim_identity, last_fill_cell_count, last_fill_vertex_peak,
    open_path_ends, tessellate_combined, tessellate_combined_oracle, tessellate_combined_with_dash,
    tessellate_combined_with_dash_trim, tessellate_network, tessellate_path,
    dash_alignment_now, subpath_corners, trim_params_from_fields, with_dash_alignment,
    with_fill_sweep_budget, with_fill_vertex_cap, with_miter_limit, NetworkStrokeStyle,
    OracleGeometry, PathEnd, TrimParams, MAX_FILL_VERTICES, axis_trim_arc_windows,
};

/// The single derive point for a `path` node's drawn geometry — never read
/// `points` directly in a render lane.
///
/// `path` has an editable uniform `cornerRadius`; the drawn outline is
/// therefore no longer the authored `points` array. Every lane that draws, masks,
/// or measures a path's body (body tessellation, backdrop coverage, mask
/// geometry, the tess dump) must come through here so the rounding cannot be
/// applied in one place and skipped in another.
///
/// `radius <= 0` (the state of EVERY document authored before path rounding, and of
/// every path the user has not rounded) returns `cmds_from_node_points` on the
/// raw array — the same commands, produced by the same code, so existing render
/// plans stay byte-exact.
pub fn path_cmds_for_node(
    node: &moda_scene::types::RawNode,
) -> Vec<moda_engine_path_types::PathCmd> {
    // A vector network (E7.9) is authoritative: its region loops, then the
    // region-less runs — what the editor derives into `points`, recomputed
    // here so a scene that carries only the network still draws.
    if let Some(net) = network::node_network(node) {
        return network::derived_cmds(&net);
    }
    let points = node
        .points()
        .unwrap_or(moda_scene::path_points::PathPoints::Json(&[]));
    // `path.cornerRadius || 0` (JS falsy: 0/NaN/missing → 0), matching star/polygon; an anchor's own
    // `cornerRadius` / `cornerStyle` override it (live corners).
    let corners = moda_scene::path_geom::round_corners::PathCorners::read(node);
    if !corners.rounds_anything() {
        return moda_engine_path_types::cmds_from_path_points(points);
    }

    let parsed: Vec<moda_scene::blend::BlendPathPoint> = points
        .iter()
        .filter_map(moda_scene::blend::BlendPathPoint::from_ref)
        .collect();
    let rounded = moda_scene::path_geom::round_corners::round_path_corners_with(
        &parsed,
        node.bool("closed").unwrap_or(false),
        &corners,
    );
    rounded.iter().map(path_cmd_from_blend_point).collect()
}

/// `BlendPathPoint` → `PathCmd`, the one mapping shared by every consumer of
/// `moda_scene::path_geom::round_corners` (path bodies here, star/polygon rings
/// in `shape_paths.rs`).
pub fn path_cmd_from_blend_point(
    point: &moda_scene::blend::BlendPathPoint,
) -> moda_engine_path_types::PathCmd {
    use moda_engine_path_types::PathCmd;
    match point {
        moda_scene::blend::BlendPathPoint::Move { x, y } => PathCmd::Move { x: *x, y: *y },
        moda_scene::blend::BlendPathPoint::Line { x, y } => PathCmd::Line { x: *x, y: *y },
        moda_scene::blend::BlendPathPoint::Curve { x, y, cp1, cp2 } => PathCmd::Curve {
            x: *x,
            y: *y,
            cp1: *cp1,
            cp2: *cp2,
        },
        // The rounder normalizes quadratics and arcs to cubics, so these two
        // arms are unreachable for rounded output; keep them so a future variant
        // that does survive rounding is a compile error, not a silently dropped
        // command.
        moda_scene::blend::BlendPathPoint::Quadratic { x, y, cp } => PathCmd::Quadratic {
            x: *x,
            y: *y,
            cp: *cp,
        },
        moda_scene::blend::BlendPathPoint::Arc {
            x,
            y,
            rx,
            ry,
            x_axis_rotation,
            large_arc,
            sweep,
        } => PathCmd::Arc {
            x: *x,
            y: *y,
            rx: *rx,
            ry: *ry,
            x_axis_rotation: *x_axis_rotation,
            large_arc: *large_arc,
            sweep: *sweep,
        },
    }
}

/// Path-command model the tessellator consumes — mirrors
/// `core/scene/nodes.ts` PathPoint variants (move/line/curve/quadratic/arc).
pub mod moda_engine_path_types {
    #[derive(Debug, Clone, PartialEq)]
    pub enum PathCmd {
        Move {
            x: f64,
            y: f64,
        },
        Line {
            x: f64,
            y: f64,
        },
        Curve {
            x: f64,
            y: f64,
            cp1: (f64, f64),
            cp2: (f64, f64),
        },
        Quadratic {
            x: f64,
            y: f64,
            cp: (f64, f64),
        },
        Arc {
            x: f64,
            y: f64,
            rx: f64,
            ry: f64,
            x_axis_rotation: f64,
            large_arc: bool,
            sweep: bool,
        },
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FillRule {
        NonZero,
        EvenOdd,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum LineCap {
        Butt,
        Round,
        Square,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum LineJoin {
        Miter,
        Round,
        Bevel,
    }

    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub enum StrokeAlign {
        #[default]
        Center,
        Inside,
        Outside,
    }

    /// JSON `points` array of a path node → commands. Unknown types skipped.
    pub fn cmds_from_node_points(points: &[moda_scene::value::Value]) -> Vec<PathCmd> {
        cmds_from_path_points(moda_scene::path_points::PathPoints::Json(points))
    }

    /// A node's points ([`moda_scene::types::RawNode::points`]) → commands; exactly
    /// [`cmds_from_node_points`] of their JSON.
    pub fn cmds_from_path_points(points: moda_scene::path_points::PathPoints<'_>) -> Vec<PathCmd> {
        let pt = |p: &moda_scene::path_points::PathPointRef<'_>, k: &str| -> (f64, f64) {
            match p.xy(k) {
                Some((x, y)) => (x.unwrap_or(0.0), y.unwrap_or(0.0)),
                None => (0.0, 0.0),
            }
        };
        points
            .iter()
            .filter_map(|p| {
                if !p.is_object() {
                    return None;
                }
                let x = p.f64("x").unwrap_or(0.0);
                let y = p.f64("y").unwrap_or(0.0);
                match p.kind()? {
                    "move" => Some(PathCmd::Move { x, y }),
                    "line" => Some(PathCmd::Line { x, y }),
                    "curve" => Some(PathCmd::Curve {
                        x,
                        y,
                        cp1: pt(&p, "cp1"),
                        cp2: pt(&p, "cp2"),
                    }),
                    "quadratic" => Some(PathCmd::Quadratic {
                        x,
                        y,
                        cp: pt(&p, "cp"),
                    }),
                    "arc" => Some(PathCmd::Arc {
                        x,
                        y,
                        rx: p.f64("rx").unwrap_or(0.0),
                        ry: p.f64("ry").unwrap_or(0.0),
                        x_axis_rotation: p.f64("xAxisRotation").unwrap_or(0.0),
                        large_arc: p.bool("largeArc").unwrap_or(false),
                        sweep: p.bool("sweep").unwrap_or(false),
                    }),
                    _ => None,
                }
            })
            .collect()
    }

    /// `node.dash` → dash pattern. Non-array values (the WEB-NEXTJS-13T
    /// regression: a string at runtime) are silently skipped, exactly like
    /// the TS `Array.isArray(dash)` guard.
    pub fn dash_from_node_value(v: Option<&moda_scene::value::Value>) -> Option<Vec<f64>> {
        let arr = v?.as_array()?;
        Some(
            arr.iter()
                .filter_map(moda_scene::value::Value::as_f64)
                .collect::<Vec<f64>>(),
        )
    }

    pub fn fill_rule_from_str(s: Option<&str>) -> FillRule {
        match s {
            Some("evenodd") => FillRule::EvenOdd,
            _ => FillRule::NonZero,
        }
    }

    pub fn line_cap_from_str(s: Option<&str>) -> LineCap {
        match s {
            Some("round") => LineCap::Round,
            Some("square") => LineCap::Square,
            _ => LineCap::Butt,
        }
    }

    pub fn line_join_from_str(s: Option<&str>) -> LineJoin {
        match s {
            Some("round") => LineJoin::Round,
            Some("bevel") => LineJoin::Bevel,
            _ => LineJoin::Miter,
        }
    }

    pub fn stroke_align_from_str(s: Option<&str>) -> StrokeAlign {
        match s {
            Some("inside") => StrokeAlign::Inside,
            Some("outside") => StrokeAlign::Outside,
            _ => StrokeAlign::Center,
        }
    }
}
