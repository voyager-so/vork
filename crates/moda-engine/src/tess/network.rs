//! Vector networks (E7.9): how a `path` node with a `vectorNetwork` draws.
//!
//! Figma's model, which this follows:
//! - **Fills per region.** Each region is filled on its own, with its own
//!   winding rule (a region's loops are one compound path). Regions are
//!   disjoint faces in practice, so their triangles never overlap.
//! - **Strokes per segment.** Every segment is stroked exactly once, filled or
//!   not; an edge shared by two regions is not stroked twice.
//! - **Caps and joins at vertices.** Segments chain into pieces through
//!   vertices of degree 2, which take the tessellator's join. A piece stops at
//!   a vertex of degree 1 (an end: the vertex's `strokeCap`, else the
//!   stroke's cap), of degree 3 or more (a junction), or of degree 2 whose
//!   `strokeJoin` overrides the stroke's. Junctions are closed with the join
//!   geometry of every pair of neighbouring segments (round: a disc; bevel:
//!   the wedge between the two butt ends; miter: the wedge out to the miter
//!   point while within the miter limit, else the bevel), so a plus sign or
//!   a Y has no notch at its hub.
//!
//! Beyond Figma's own network drawing:
//! - **Corner radius.** A vertex's `cornerRadius` (else the path's) rounds it,
//!   junctions included: the network is drawn as
//!   [`VectorNetwork::rounded`] rewrites it, so fills, strokes, masks and
//!   bounds all see the rounded outline.
//! - **Arrow caps.** An open end (a vertex of degree 1) whose `strokeCap` is
//!   one of Figma's arrows draws the path arrowhead it maps to
//!   (`ARROW_LINES` → open caret, `ARROW_EQUILATERAL` → triangle,
//!   `TRIANGLE_FILLED` → reverse triangle, `DIAMOND_FILLED`, `CIRCLE_FILLED`),
//!   with the piece cut back under a filled one (`tess::path_arrows`). The
//!   stroke's own `startArrow` / `endArrow` do not apply: a network has no
//!   start or end.
//! - **Stroke alignment.** Inside / outside align the stroke of every region
//!   loop that closes, per region (an edge two regions share is stroked on
//!   both sides, so it draws centred); segments in no region stay centred. A
//!   dashed stroke is centred, as on a path.
//! - **Trim.** The pieces are the trimmed path's sub-paths, each drawn from
//!   the vertex it starts at: the window trims every piece at once, or (in
//!   `individual` mode) runs along them in order. A trimmed stroke draws its
//!   pieces open with the stroke's cap and no junction wedges, and keeps an
//!   arrowhead only while its end is in the window.
//!
//! A dash runs along each piece.

use moda_scene::types::RawNode;
use moda_scene::vector_network::{
    network_from_node, StrokeCap, StrokeJoin, Vector, VectorNetwork, WindingRule,
};

use super::line_to_path::ArrowShape;
use super::moda_engine_path_types::{FillRule, LineCap, LineJoin, PathCmd, StrokeAlign};

/// A chain of segments stroked with one cap and join.
#[derive(Debug, Clone, PartialEq)]
pub struct StrokePiece {
    pub cmds: Vec<PathCmd>,
    pub closed: bool,
    /// The cap the tessellator draws at both ends (`Butt` when the ends are
    /// junctions, or when the two ends' caps differ and [`NetworkStroke::caps`]
    /// draws them instead).
    pub cap: LineCap,
    /// Arrowheads at the piece's start and end (open ends with an arrow cap).
    pub arrows: [Option<ArrowShape>; 2],
}

/// A cap drawn separately from its piece: at `at`, facing `outward`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EndCap {
    pub at: Vector,
    pub outward: Vector,
    pub cap: LineCap,
}

/// A vertex where pieces meet: the unit directions every segment leaves it
/// in, and the join that closes the gaps between them.
#[derive(Debug, Clone, PartialEq)]
pub struct Junction {
    pub at: Vector,
    pub dirs: Vec<Vector>,
    pub join: LineJoin,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct NetworkStroke {
    pub pieces: Vec<StrokePiece>,
    pub caps: Vec<EndCap>,
    pub junctions: Vec<Junction>,
    /// Inside / outside strokes: each region's closed loops, one compound path
    /// per region, stroked with `align`. Their segments are in no piece.
    pub aligned: Vec<Vec<PathCmd>>,
    pub align: StrokeAlign,
}

/// The node's network as drawn (corner radius applied), if it has a valid one.
pub fn node_network(node: &RawNode) -> Option<VectorNetwork> {
    if node.node_type() != "path" {
        return None;
    }
    let radius = node.f64("cornerRadius").unwrap_or(0.0);
    network_from_node(node).map(|n| n.rounded(radius))
}

/// The path arrowhead a vertex's arrow cap draws (Figma's paste mapping, `arrowShapeForCap`).
pub fn arrow_from_vertex(c: &StrokeCap) -> Option<ArrowShape> {
    match c {
        StrokeCap::ArrowLines => Some(ArrowShape::Open),
        StrokeCap::ArrowEquilateral => Some(ArrowShape::Triangle),
        StrokeCap::TriangleFilled => Some(ArrowShape::ReverseTriangle),
        StrokeCap::DiamondFilled => Some(ArrowShape::Diamond),
        StrokeCap::CircleFilled => Some(ArrowShape::Circle),
        _ => None,
    }
}

/// Whether some open end of the network has an arrow cap.
pub fn has_arrow_caps(net: &VectorNetwork) -> bool {
    net.vertices.iter().enumerate().any(|(v, vx)| {
        vx.stroke_cap.as_ref().and_then(arrow_from_vertex).is_some() && net.degree(v) == 1
    })
}

pub fn cap_from_vertex(c: &StrokeCap) -> Option<LineCap> {
    match c {
        StrokeCap::None => Some(LineCap::Butt),
        StrokeCap::Round => Some(LineCap::Round),
        StrokeCap::Square => Some(LineCap::Square),
        // An arrow cap's end is butt: its arrowhead is drawn over it
        // (`arrow_from_vertex`).
        _ => Some(LineCap::Butt),
    }
}

pub fn join_from_vertex(j: &StrokeJoin) -> Option<LineJoin> {
    match j {
        StrokeJoin::Miter => Some(LineJoin::Miter),
        StrokeJoin::Bevel => Some(LineJoin::Bevel),
        StrokeJoin::Round => Some(LineJoin::Round),
        StrokeJoin::Other(_) => None,
    }
}

fn fill_rule(w: WindingRule) -> FillRule {
    match w {
        WindingRule::NonZero => FillRule::NonZero,
        WindingRule::EvenOdd => FillRule::EvenOdd,
    }
}

fn cmd_of(p: &moda_scene::blend::BlendPathPoint) -> PathCmd {
    super::path_cmd_from_blend_point(p)
}

/// Each region as a compound path and its fill rule.
pub fn region_fills(net: &VectorNetwork) -> Vec<(Vec<PathCmd>, FillRule)> {
    net.to_paths()
        .regions
        .iter()
        .map(|r| {
            (
                r.points.iter().map(cmd_of).collect(),
                fill_rule(net.regions[r.region].winding_rule),
            )
        })
        .collect()
}

/// The derived commands for lanes that read one outline (masks, backdrop
/// coverage, bounds): region loops, then the region-less runs, as the editor
/// stores them in `points`.
pub fn derived_cmds(net: &VectorNetwork) -> Vec<PathCmd> {
    moda_scene::vector_network::derived_points(net)
        .iter()
        .map(cmd_of)
        .collect()
}

fn unit(v: Vector) -> Vector {
    let l = v.len();
    if l > 1e-12 {
        v * (1.0 / l)
    } else {
        Vector::new(1.0, 0.0)
    }
}

/// Each region's loops that close, as one compound path per region, and the
/// segments they cover.
fn aligned_loops(net: &VectorNetwork) -> (Vec<Vec<PathCmd>>, Vec<bool>) {
    let mut covered = vec![false; net.segments.len()];
    let mut out = Vec::new();
    for region in &net.regions {
        let mut cmds = Vec::new();
        for lp in &region.loops {
            let Some((orient, true)) = net.chain_loop(lp) else {
                continue;
            };
            for (k, &(si, rev)) in orient.iter().enumerate() {
                covered[si] = true;
                let c = net.oriented_cubic(si, rev);
                if k == 0 {
                    cmds.push(PathCmd::Move {
                        x: c.p0.x,
                        y: c.p0.y,
                    });
                }
                cmds.push(seg_cmd(net, si, &c));
            }
        }
        if !cmds.is_empty() {
            out.push(cmds);
        }
    }
    (out, covered)
}

fn seg_cmd(net: &VectorNetwork, si: usize, c: &moda_scene::vector_network::Cubic) -> PathCmd {
    if net.segments[si].is_straight() {
        PathCmd::Line {
            x: c.p3.x,
            y: c.p3.y,
        }
    } else {
        PathCmd::Curve {
            x: c.p3.x,
            y: c.p3.y,
            cp1: (c.p1.x, c.p1.y),
            cp2: (c.p2.x, c.p2.y),
        }
    }
}

/// Plan the network's stroke: pieces, separately drawn caps, junctions, and
/// (aligned inside / outside) the region loops.
pub fn plan_stroke(
    net: &VectorNetwork,
    node_cap: LineCap,
    node_join: LineJoin,
    align: StrokeAlign,
) -> NetworkStroke {
    let nv = net.vertices.len();
    let (aligned, skip) = if align == StrokeAlign::Center {
        (Vec::new(), vec![false; net.segments.len()])
    } else {
        aligned_loops(net)
    };
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); nv];
    // Every segment end at a vertex, aligned loops included: an end of the
    // centred stroke that meets a loop is not an open end.
    let mut degree = vec![0usize; nv];
    for (i, s) in net.segments.iter().enumerate() {
        degree[s.start] += 1;
        degree[s.end] += 1;
        if skip[i] {
            continue;
        }
        incident[s.start].push(i);
        incident[s.end].push(i);
    }
    let join_at = |v: usize| {
        net.vertices[v]
            .stroke_join
            .as_ref()
            .and_then(join_from_vertex)
            .unwrap_or(node_join)
    };
    let cap_at = |v: usize| {
        net.vertices[v]
            .stroke_cap
            .as_ref()
            .and_then(cap_from_vertex)
            .unwrap_or(node_cap)
    };
    let arrow_at = |v: usize| {
        net.vertices[v]
            .stroke_cap
            .as_ref()
            .and_then(arrow_from_vertex)
    };
    let is_break = |v: usize| incident[v].len() != 2 || degree[v] != 2 || join_at(v) != node_join;

    let mut used = skip.clone();
    let mut out = NetworkStroke {
        aligned,
        align,
        ..NetworkStroke::default()
    };

    // Walk from vertex `from` along segment `first`, through non-break
    // vertices. Returns the oriented segments and the vertex it stops at.
    let walk = |from: usize, first: usize, used: &mut Vec<bool>| {
        let mut chain: Vec<(usize, bool)> = Vec::new();
        let mut cursor = from;
        let mut next = Some(first);
        while let Some(si) = next {
            used[si] = true;
            let s = &net.segments[si];
            let rev = s.start != cursor;
            cursor = if rev { s.start } else { s.end };
            chain.push((si, rev));
            next = if is_break(cursor) {
                None
            } else {
                incident[cursor].iter().copied().find(|&i| !used[i])
            };
        }
        (chain, cursor)
    };
    let emit = |chain: &[(usize, bool)], closed: bool, cap: LineCap, arrows| {
        let mut cmds = Vec::with_capacity(chain.len() + 1);
        for (k, &(si, rev)) in chain.iter().enumerate() {
            let c = net.oriented_cubic(si, rev);
            if k == 0 {
                cmds.push(PathCmd::Move {
                    x: c.p0.x,
                    y: c.p0.y,
                });
            }
            cmds.push(seg_cmd(net, si, &c));
        }
        StrokePiece {
            cmds,
            closed,
            cap,
            arrows,
        }
    };
    // The direction a segment leaves vertex `v` in (the start of a self-loop).
    let leave_dir = |si: usize, v: usize, at_start: bool| {
        let c = net.segment_cubic(si);
        let d = if net.segments[si].start == v && at_start {
            c.start_direction()
        } else {
            c.reversed().start_direction()
        };
        unit(d)
    };

    for v in 0..nv {
        if !is_break(v) {
            continue;
        }
        for k in 0..incident[v].len() {
            let si = incident[v][k];
            if used[si] {
                continue;
            }
            let (chain, end) = walk(v, si, &mut used);
            let (first, first_rev) = chain[0];
            let (last, last_rev) = chain[chain.len() - 1];
            let start_end = degree[v] == 1;
            let end_end = degree[end] == 1;
            let start_cap = start_end.then(|| cap_at(v));
            let end_cap = end_end.then(|| cap_at(end));
            // One cap for both ends goes to the tessellator (dashes keep it);
            // otherwise the piece is butt-ended and the caps drawn here.
            let shared = match (start_cap, end_cap) {
                (Some(a), Some(b)) if a == b => Some(a),
                _ => None,
            };
            let arrows = [
                start_end.then(|| arrow_at(v)).flatten(),
                end_end.then(|| arrow_at(end)).flatten(),
            ];
            out.pieces
                .push(emit(&chain, false, shared.unwrap_or(LineCap::Butt), arrows));
            if shared.is_none() {
                if let Some(cap) = start_cap.filter(|c| *c != LineCap::Butt) {
                    let d = leave_dir(first, v, !first_rev);
                    out.caps.push(EndCap {
                        at: net.vertex_pos(v),
                        outward: -d,
                        cap,
                    });
                }
                if let Some(cap) = end_cap.filter(|c| *c != LineCap::Butt) {
                    let d = leave_dir(last, end, last_rev);
                    out.caps.push(EndCap {
                        at: net.vertex_pos(end),
                        outward: -d,
                        cap,
                    });
                }
            }
        }
    }
    // What is left are cycles through degree-2 vertices with the stroke's join.
    for si in 0..net.segments.len() {
        if used[si] {
            continue;
        }
        let v = net.segments[si].start;
        let (chain, _) = walk(v, si, &mut used);
        out.pieces.push(emit(&chain, true, node_cap, [None, None]));
    }
    // Junctions: every break vertex with two or more segment ends.
    for (v, inc) in incident.iter().enumerate() {
        if inc.len() < 2 || !is_break(v) {
            continue;
        }
        let mut dirs = Vec::with_capacity(inc.len());
        let mut seen_loop = std::collections::HashSet::new();
        for &si in inc {
            let s = &net.segments[si];
            if s.start == s.end {
                // A self-loop leaves and returns: both of its ends.
                if seen_loop.insert(si) {
                    dirs.push(leave_dir(si, v, true));
                    dirs.push(leave_dir(si, v, false));
                }
            } else {
                dirs.push(leave_dir(si, v, s.start == v));
            }
        }
        out.junctions.push(Junction {
            at: net.vertex_pos(v),
            dirs,
            join: join_at(v),
        });
    }
    out
}

fn push_tri(tris: &mut Vec<f64>, a: Vector, b: Vector, c: Vector) {
    tris.extend_from_slice(&[a.x, a.y, b.x, b.y, c.x, c.y]);
}

/// A disc at `c` as a triangle fan, flattened to `tol`.
fn disc(tris: &mut Vec<f64>, c: Vector, r: f64, tol: f64) {
    if r <= 0.0 {
        return;
    }
    // Each chord sags at most a quarter of `tol` (the band's flatness): a quarter of a device pixel or less
    // anywhere in the zoom band.
    let tol = tol * 0.25;
    let n = if tol > 0.0 && tol < r {
        (std::f64::consts::PI / (1.0 - tol / r).acos()).ceil() as usize
    } else {
        8
    }
    .clamp(8, 4096);
    let at = |k: usize| {
        let a = std::f64::consts::TAU * (k as f64) / (n as f64);
        Vector::new(c.x + r * a.cos(), c.y + r * a.sin())
    };
    for k in 0..n {
        push_tri(tris, c, at(k), at(k + 1));
    }
}

fn perp_ccw(d: Vector) -> Vector {
    Vector::new(-d.y, d.x)
}

fn perp_cw(d: Vector) -> Vector {
    Vector::new(d.y, -d.x)
}

/// Triangles for the separately drawn caps and the junctions, for a stroke
/// of half-width `h`.
pub fn cap_and_junction_tris(
    stroke: &NetworkStroke,
    h: f64,
    miter_limit: f64,
    tol: f64,
) -> Vec<f64> {
    let mut tris = Vec::new();
    if h <= 0.0 {
        return tris;
    }
    for cap in &stroke.caps {
        match cap.cap {
            LineCap::Butt => {}
            LineCap::Round => disc(&mut tris, cap.at, h, tol),
            LineCap::Square => {
                let o = unit(cap.outward);
                let n = perp_ccw(o) * h;
                let a = cap.at + n;
                let b = cap.at - n;
                let c = b + o * h;
                let d = a + o * h;
                push_tri(&mut tris, a, b, c);
                push_tri(&mut tris, a, c, d);
            }
        }
    }
    for j in &stroke.junctions {
        if j.join == LineJoin::Round {
            disc(&mut tris, j.at, h, tol);
            continue;
        }
        let mut dirs: Vec<(f64, Vector)> = j.dirs.iter().map(|d| (d.y.atan2(d.x), *d)).collect();
        dirs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let n = dirs.len();
        for k in 0..n {
            let (a0, d0) = dirs[k];
            let (a1, d1) = dirs[(k + 1) % n];
            let mut gap = a1 - a0;
            if k + 1 == n {
                gap += std::f64::consts::TAU;
            }
            // Only the convex side between two neighbouring segments has a
            // notch to close; the reflex side is covered by the strokes.
            if n < 2 || gap <= 1e-9 || gap >= std::f64::consts::PI - 1e-9 {
                continue;
            }
            let p = j.at + perp_ccw(d0) * h;
            let q = j.at + perp_cw(d1) * h;
            if j.join == LineJoin::Miter {
                // The miter point on the bisector of the two outer edges.
                let half = (std::f64::consts::PI - gap) / 2.0;
                let cos = half.cos();
                if cos > 1e-9 && 1.0 / cos <= miter_limit {
                    let b = unit(perp_ccw(d0) + perp_cw(d1));
                    let m = j.at + b * (h / cos);
                    push_tri(&mut tris, j.at, p, m);
                    push_tri(&mut tris, j.at, m, q);
                    continue;
                }
            }
            push_tri(&mut tris, j.at, p, q);
        }
    }
    tris
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::vector_network::VectorNetwork;
    use moda_scene::value::json;

    fn net(v: moda_scene::value::Value) -> VectorNetwork {
        VectorNetwork::from_json_value(&v).unwrap()
    }

    /// A plus sign with a hub vertex: four spokes of degree 1, one junction.
    fn plus_hub() -> VectorNetwork {
        net(json!({
            "vertices": [{"x": 50, "y": 50}, {"x": 0, "y": 50}, {"x": 100, "y": 50}, {"x": 50, "y": 0}, {"x": 50, "y": 100}],
            "segments": [{"start": 0, "end": 1}, {"start": 0, "end": 2}, {"start": 0, "end": 3}, {"start": 0, "end": 4}],
        }))
    }

    #[test]
    fn plus_strokes_each_spoke_once_with_a_junction() {
        let s = plan_stroke(
            &plus_hub(),
            LineCap::Round,
            LineJoin::Miter,
            StrokeAlign::Center,
        );
        assert_eq!(s.pieces.len(), 4);
        assert!(s.pieces.iter().all(|p| !p.closed && p.cmds.len() == 2));
        assert_eq!(s.junctions.len(), 1);
        assert_eq!(s.junctions[0].dirs.len(), 4);
        // The hub end is a junction, the outer end a cap: drawn separately.
        assert!(s.pieces.iter().all(|p| p.cap == LineCap::Butt));
        assert_eq!(s.caps.len(), 4);
    }

    #[test]
    fn a_polyline_is_one_piece_with_its_caps() {
        let n = net(json!({
            "vertices": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}],
            "segments": [{"start": 0, "end": 1}, {"start": 1, "end": 2}],
        }));
        let s = plan_stroke(&n, LineCap::Round, LineJoin::Miter, StrokeAlign::Center);
        assert_eq!(s.pieces.len(), 1);
        assert_eq!(s.pieces[0].cap, LineCap::Round);
        assert!(s.caps.is_empty() && s.junctions.is_empty());
    }

    #[test]
    fn a_closed_loop_is_one_closed_piece() {
        let n = net(json!({
            "vertices": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}],
            "segments": [{"start": 0, "end": 1}, {"start": 1, "end": 2}, {"start": 2, "end": 0}],
        }));
        let s = plan_stroke(&n, LineCap::Butt, LineJoin::Miter, StrokeAlign::Center);
        assert_eq!(s.pieces.len(), 1);
        assert!(s.pieces[0].closed);
        assert_eq!(s.pieces[0].cmds.len(), 4);
    }

    #[test]
    fn a_vertex_join_override_breaks_the_run() {
        let n = net(json!({
            "vertices": [{"x": 0, "y": 0}, {"x": 10, "y": 0, "strokeJoin": "ROUND"}, {"x": 10, "y": 10}],
            "segments": [{"start": 0, "end": 1}, {"start": 1, "end": 2}],
        }));
        let s = plan_stroke(&n, LineCap::Butt, LineJoin::Miter, StrokeAlign::Center);
        assert_eq!(s.pieces.len(), 2);
        assert_eq!(s.junctions.len(), 1);
        assert_eq!(s.junctions[0].join, LineJoin::Round);
    }

    #[test]
    fn per_vertex_caps_differ_per_end() {
        let n = net(json!({
            "vertices": [{"x": 0, "y": 0, "strokeCap": "ROUND"}, {"x": 10, "y": 0, "strokeCap": "SQUARE"}],
            "segments": [{"start": 0, "end": 1}],
        }));
        let s = plan_stroke(&n, LineCap::Butt, LineJoin::Miter, StrokeAlign::Center);
        assert_eq!(s.pieces[0].cap, LineCap::Butt);
        assert_eq!(s.caps.len(), 2);
        // The start cap faces away from the segment (−x), the end cap +x.
        let start = s.caps.iter().find(|c| c.cap == LineCap::Round).unwrap();
        assert!(start.outward.x < -0.99);
        let end = s.caps.iter().find(|c| c.cap == LineCap::Square).unwrap();
        assert!(end.outward.x > 0.99);
    }

    fn tri_area(t: &[f64]) -> f64 {
        t.chunks(6)
            .map(|c| ((c[2] - c[0]) * (c[5] - c[1]) - (c[4] - c[0]) * (c[3] - c[1])).abs() / 2.0)
            .sum()
    }

    #[test]
    fn junction_wedges_close_a_right_angle_hub() {
        // Four spokes at right angles: miter wedges fill the hub's corners
        // out to the h×h square each (the miter of a 90° join), bevel to half.
        let s = plan_stroke(
            &plus_hub(),
            LineCap::Butt,
            LineJoin::Miter,
            StrokeAlign::Center,
        );
        let h = 5.0;
        let miter = cap_and_junction_tris(&s, h, 4.0, 0.1);
        assert!((tri_area(&miter) - 4.0 * h * h).abs() < 1e-9);
        let s = plan_stroke(
            &plus_hub(),
            LineCap::Butt,
            LineJoin::Bevel,
            StrokeAlign::Center,
        );
        let bevel = cap_and_junction_tris(&s, h, 4.0, 0.1);
        assert!((tri_area(&bevel) - 4.0 * h * h / 2.0).abs() < 1e-9);
    }

    #[test]
    fn region_fills_keep_each_regions_rule() {
        let n = net(json!({
            "vertices": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}],
            "segments": [{"start": 0, "end": 1}, {"start": 1, "end": 2}, {"start": 2, "end": 0}],
            "regions": [{"windingRule": "EVENODD", "loops": [[0, 1, 2]]}],
        }));
        let f = region_fills(&n);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].1, FillRule::EvenOdd);
    }

    #[test]
    fn arrow_caps_mark_their_open_ends_only() {
        // A V with an arrow at one open end, and an arrow cap on the junction vertex (not an end: ignored).
        let n = net(json!({
            "vertices": [{"x": 0, "y": 0, "strokeCap": "ARROW_EQUILATERAL"}, {"x": 50, "y": 50, "strokeCap": "ARROW_LINES"},
                         {"x": 100, "y": 0, "strokeCap": "ROUND"}],
            "segments": [{"start": 0, "end": 1}, {"start": 1, "end": 2}],
        }));
        assert!(has_arrow_caps(&n));
        let s = plan_stroke(&n, LineCap::Butt, LineJoin::Miter, StrokeAlign::Center);
        assert_eq!(s.pieces.len(), 1);
        let p = &s.pieces[0];
        let arrows: Vec<ArrowShape> = p.arrows.iter().flatten().copied().collect();
        assert_eq!(arrows, [ArrowShape::Triangle]);
        // The arrow end is butt; the round end drawn separately.
        assert_eq!(p.cap, LineCap::Butt);
        assert_eq!(s.caps.len(), 1);
        assert_eq!(s.caps[0].cap, LineCap::Round);
    }

    #[test]
    fn a_plus_with_arrow_ends_has_four_arrowed_spokes() {
        let mut n = plus_hub();
        for v in 1..5 {
            n.vertices[v].stroke_cap = Some(StrokeCap::ArrowEquilateral);
        }
        let s = plan_stroke(&n, LineCap::Butt, LineJoin::Miter, StrokeAlign::Center);
        assert_eq!(s.pieces.len(), 4);
        assert!(s
            .pieces
            .iter()
            .all(|p| p.arrows.iter().flatten().count() == 1));
        assert_eq!(s.junctions.len(), 1);
    }

    #[test]
    fn aligned_strokes_take_the_closed_region_loops_and_leave_the_rest_centred() {
        // A filled triangle with a tail off one corner.
        let n = net(json!({
            "vertices": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}, {"x": 30, "y": 10}],
            "segments": [{"start": 0, "end": 1}, {"start": 1, "end": 2}, {"start": 2, "end": 0}, {"start": 2, "end": 3}],
            "regions": [{"windingRule": "NONZERO", "loops": [[0, 1, 2]]}],
        }));
        let s = plan_stroke(&n, LineCap::Round, LineJoin::Miter, StrokeAlign::Inside);
        assert_eq!(s.aligned.len(), 1);
        assert_eq!(s.aligned[0].len(), 4);
        // The tail: one piece; its end on the loop is a butt end, not a cap, and no junction.
        assert_eq!(s.pieces.len(), 1);
        assert_eq!(s.caps.len(), 1);
        assert!(s.junctions.is_empty());
        // Centred: everything is a piece.
        let c = plan_stroke(&n, LineCap::Round, LineJoin::Miter, StrokeAlign::Center);
        assert!(c.aligned.is_empty());
        assert_eq!(c.junctions.len(), 1);
    }

    #[test]
    fn node_corner_radius_rounds_a_networks_vertices() {
        use moda_scene::types::RawNode;
        let node = |radius: moda_scene::value::Value| {
            let v = json!({"id": "n", "type": "path", "points": [], "cornerRadius": radius,
                "vectorNetwork": {
                    "vertices": [{"x": 50, "y": 50}, {"x": 0, "y": 50}, {"x": 100, "y": 50}, {"x": 50, "y": 0}],
                    "segments": [{"start": 0, "end": 1}, {"start": 0, "end": 2}, {"start": 0, "end": 3}]}});
            RawNode::new(v.as_object().unwrap().clone())
        };
        assert_eq!(node_network(&node(json!(0))).unwrap().segments.len(), 3);
        // A T: two fillets into the stem, the straight side kept through two stubs.
        assert_eq!(node_network(&node(json!(8))).unwrap().segments.len(), 7);
    }
}
