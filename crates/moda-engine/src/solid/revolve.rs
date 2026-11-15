//! Surfaces of revolution: a profile across the surface, `(r, y)` px (r from the axis, y down the layer), turned
//! about the vertical axis. Cylinders and cones are their outline's profile with its corners filleted
//! ([`fillet`]); a lathe (a layer's outline revolved) is the same turn of its contours.

use std::f64::consts::TAU;

use super::Part;

type V2 = [f64; 2];

/// One end of a band across the profile: where it is and the surface's outward normal there `(nr, ny)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub r: f64,
    pub y: f64,
    pub n: V2,
}

/// The profile as bands: each a strip of the surface between two samples, smooth across its width. Two bands that
/// meet with different normals leave a crease.
pub type Band = (Sample, Sample);

fn norm(v: V2) -> V2 {
    let l = v[0].hypot(v[1]);
    if l > 1e-12 { [v[0] / l, v[1] / l] } else { [0.0, 0.0] }
}

/// The outward normal of an edge going `a` → `b` (the solid on its right as y runs down: a profile traced from the
/// top of the axis, out and down, and back to the axis).
fn edge_normal(a: V2, b: V2) -> V2 {
    norm([b[1] - a[1], -(b[0] - a[0])])
}

/// A polyline as bands, the corners `round` picks filleted with `radius` (a circular arc tangent to both edges, in
/// `segments` steps; smaller where an edge is too short for it: at most half of each edge is used). Straight edges
/// keep their own normal, so an unrounded corner is a crease.
pub fn fillet(points: &[V2], radius: f64, segments: u32, round: impl Fn(V2) -> bool) -> Vec<Band> {
    let mut pts: Vec<V2> = Vec::with_capacity(points.len());
    for &p in points {
        if pts.last().is_none_or(|q: &V2| (q[0] - p[0]).hypot(q[1] - p[1]) > 1e-9) {
            pts.push(p);
        }
    }
    let n = pts.len();
    if n < 2 {
        return Vec::new();
    }
    let seg = segments.max(1);
    // Per corner: (where the incoming edge stops, where the outgoing one starts, the arc between them).
    let mut starts: Vec<V2> = pts.clone();
    let mut ends: Vec<V2> = pts.clone();
    let mut arcs: Vec<Vec<Sample>> = vec![Vec::new(); n];
    for i in 1..n - 1 {
        if radius <= 0.0 || !round(pts[i]) {
            continue;
        }
        let (a, p, b) = (pts[i - 1], pts[i], pts[i + 1]);
        let (la, lb) = ((p[0] - a[0]).hypot(p[1] - a[1]), (b[0] - p[0]).hypot(b[1] - p[1]));
        let d1 = norm([p[0] - a[0], p[1] - a[1]]);
        let d2 = norm([b[0] - p[0], b[1] - p[1]]);
        let cos = (-d1[0] * d2[0] - d1[1] * d2[1]).clamp(-1.0, 1.0);
        let theta = cos.acos();
        if theta < 1e-6 || (std::f64::consts::PI - theta) < 1e-6 {
            continue;
        }
        let tan_half = (theta / 2.0).tan();
        let t = (radius / tan_half).min(la / 2.0).min(lb / 2.0);
        let r = t * tan_half;
        let (n1, n2) = (edge_normal(a, p), edge_normal(p, b));
        let t1 = [p[0] - d1[0] * t, p[1] - d1[1] * t];
        let t2 = [p[0] + d2[0] * t, p[1] + d2[1] * t];
        // Convex (the next edge turns away from the outward side): the arc's centre is inside, else outside.
        let s = if n1[0] * d2[0] + n1[1] * d2[1] < 0.0 { 1.0 } else { -1.0 };
        let c = [t1[0] - s * n1[0] * r, t1[1] - s * n1[1] * r];
        let a1 = n1[1].atan2(n1[0]);
        let mut delta = n2[1].atan2(n2[0]) - a1;
        while delta > std::f64::consts::PI {
            delta -= TAU;
        }
        while delta <= -std::f64::consts::PI {
            delta += TAU;
        }
        arcs[i] = (0..=seg)
            .map(|k| {
                let a = a1 + delta * f64::from(k) / f64::from(seg);
                let nn = [a.cos(), a.sin()];
                Sample { r: c[0] + s * r * nn[0], y: c[1] + s * r * nn[1], n: nn }
            })
            .collect();
        ends[i] = t1;
        starts[i] = t2;
    }
    let mut bands = Vec::new();
    for i in 0..n - 1 {
        let (a, b) = (starts[i], ends[i + 1]);
        let nn = edge_normal(pts[i], pts[i + 1]);
        if (a[0] - b[0]).hypot(a[1] - b[1]) > 1e-9 {
            bands.push((Sample { r: a[0], y: a[1], n: nn }, Sample { r: b[0], y: b[1], n: nn }));
        }
        for w in arcs[i + 1].windows(2) {
            bands.push((w[0], w[1]));
        }
    }
    bands
}

/// How a profile turns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Turn {
    /// The axis's x (layer px).
    pub cx: f64,
    /// Depth over width: the turn's z is `r · depth_ratio` (an elliptical cross-section when ≠ 1).
    pub depth_ratio: f64,
    /// Steps around a full turn.
    pub segments: u32,
    /// Degrees turned, from the right (+x) toward the back (+z); 360 is a closed surface.
    pub sweep: f64,
}

/// Turn `bands` about the axis into `part`, with UVs (u around the turn, v along the profile).
pub fn revolve(bands: &[Band], turn: &Turn, part: &mut Part) {
    let sweep = turn.sweep.clamp(0.0, 360.0).to_radians();
    if bands.is_empty() || sweep <= 0.0 {
        return;
    }
    let steps = ((f64::from(turn.segments.clamp(3, 512)) * sweep / TAU).ceil() as usize).max(1);
    let k = if turn.depth_ratio.is_finite() && turn.depth_ratio > 1e-9 { turn.depth_ratio } else { 1.0 };
    let length: f64 = bands.iter().map(|(a, b)| (a.r - b.r).hypot(a.y - b.y)).sum::<f64>().max(1e-9);
    let mut along = 0.0;
    if part.uvs.len() != part.positions.len() {
        part.uvs.resize(part.positions.len(), [0.0, 0.0]);
    }
    for (a, b) in bands {
        let (va, vb) = (along / length, (along + (a.r - b.r).hypot(a.y - b.y)) / length);
        along += (a.r - b.r).hypot(a.y - b.y);
        let base = part.positions.len() as u32;
        for j in 0..=steps {
            let u = sweep * j as f64 / steps as f64;
            let (su, cu) = u.sin_cos();
            for (s, v) in [(a, va), (b, vb)] {
                let r = s.r.max(0.0);
                part.vertex([turn.cx + r * cu, s.y, k * r * su], [s.n[0] * cu, s.n[1], s.n[0] * su / k]);
                part.uvs.push([(j as f64 / steps as f64) as f32, v as f32]);
            }
        }
        for j in 0..steps as u32 {
            let (a0, b0, a1, b1) = (base + 2 * j, base + 2 * j + 1, base + 2 * j + 2, base + 2 * j + 3);
            // Wound so the outward side is front-facing like the extrusion's (the pass draws both sides).
            part.quad(a0, a1, b1, b0);
        }
    }
}

/// A cylinder's or cone's profile: the box's vertical axis, a top of `top_r` and a bottom of `bottom_r` (px), from
/// `y0` to `y1`, with or without its caps, its rim corners filleted by `bevel`.
pub fn tapered(top_r: f64, bottom_r: f64, y0: f64, y1: f64, caps: bool, bevel: f64, segments: u32) -> Vec<Band> {
    let (top_r, bottom_r) = (top_r.max(0.0), bottom_r.max(0.0));
    if !caps {
        return fillet(&[[top_r, y0], [bottom_r, y1]], 0.0, 1, |_| false);
    }
    let pts = [[0.0, y0], [top_r, y0], [bottom_r, y1], [0.0, y1]];
    fillet(&pts, bevel, segments, |p| p[0] > 1e-9)
}

/// A layer's outline turned about the vertical line `x = axis` (layer px): each contour's edges as bands (corners
/// sharper than 35° keep a crease, gentler ones are smooth), the parts left of the axis pressed onto it. Turned less
/// than a full circle, the outline closes each end.
pub fn lathe(contours: &[Vec<V2>], axis: f64, sweep: f64, segments: u32, part: &mut Part) {
    let contours: Vec<Vec<V2>> = super::geometry::orient(contours)
        .into_iter()
        .map(|c| c.into_iter().map(|p| [(p[0] - axis).max(0.0), p[1]]).collect::<Vec<V2>>())
        .filter(|c| c.len() >= 3)
        .collect();
    let turn = Turn { cx: axis, depth_ratio: 1.0, segments, sweep };
    for c in &contours {
        revolve(&contour_bands(c), &turn, part);
    }
    let sweep = sweep.clamp(0.0, 360.0);
    if sweep > 0.0 && sweep < 360.0 {
        for (angle, side) in [(0.0f64, -1.0f64), (sweep.to_radians(), 1.0)] {
            let (s, c) = angle.sin_cos();
            // The direction the turn moves at this angle is (−sin, cos) in (x, z): the start faces back along it.
            end_cap(&contours, part, |p| [axis + p[0] * c, p[1], p[0] * s], [-s * side, 0.0, c * side]);
        }
    }
}

/// A closed contour's edges as bands, each end's normal its edge's, or the two edges' mean at a gentle corner.
fn contour_bands(c: &[V2]) -> Vec<Band> {
    let n = c.len();
    let normals: Vec<V2> = (0..n).map(|i| edge_normal(c[i], c[(i + 1) % n])).collect();
    let at = |i: usize, edge: usize| -> V2 {
        // Vertex i's normal on edge `edge` (i is one of its ends): the edge's, or its mean with the vertex's other edge.
        let other = if edge == i { (i + n - 1) % n } else { i };
        let (a, b) = (normals[edge], normals[other]);
        if a[0] * b[0] + a[1] * b[1] >= super::geometry::CREASE_COS { norm([a[0] + b[0], a[1] + b[1]]) } else { a }
    };
    (0..n)
        .filter(|&i| (c[i][0] - c[(i + 1) % n][0]).hypot(c[i][1] - c[(i + 1) % n][1]) > 1e-9)
        .map(|i| {
            let j = (i + 1) % n;
            (Sample { r: c[i][0], y: c[i][1], n: at(i, i) }, Sample { r: c[j][0], y: c[j][1], n: at(j, i) })
        })
        .collect()
}

/// The outline's fill (in `(r, y)`) placed by `place` as a flat cap facing `normal`.
fn end_cap(contours: &[Vec<V2>], part: &mut Part, place: impl Fn(V2) -> [f64; 3], normal: [f64; 3]) {
    use lyon_tessellation::math::point;
    use lyon_tessellation::path::PathEvent;
    use lyon_tessellation::{BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, VertexBuffers};
    let events = contours.iter().flat_map(|c| {
        let pt = |p: &V2| point(p[0] as f32, p[1] as f32);
        let (first, last) = (pt(&c[0]), pt(&c[c.len() - 1]));
        std::iter::once(PathEvent::Begin { at: first })
            .chain(c.windows(2).map(move |w| PathEvent::Line { from: pt(&w[0]), to: pt(&w[1]) }))
            .chain(std::iter::once(PathEvent::End { last, first, close: true }))
    });
    let mut buffers: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    let tessellated = FillTessellator::new().tessellate(
        events,
        &FillOptions::DEFAULT.with_fill_rule(FillRule::NonZero),
        &mut BuffersBuilder::new(&mut buffers, |v: FillVertex| v.position().to_array()),
    );
    if tessellated.is_err() {
        return;
    }
    if part.uvs.len() != part.positions.len() {
        part.uvs.resize(part.positions.len(), [0.0, 0.0]);
    }
    let base = part.positions.len() as u32;
    for p in &buffers.vertices {
        part.vertex(place([f64::from(p[0]), f64::from(p[1])]), normal);
        part.uvs.push([0.0, 0.0]);
    }
    for t in buffers.indices.chunks_exact(3) {
        part.tri(base + t[0], base + t[1], base + t[2]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_profile_filleted_rounds_its_two_outer_corners() {
        let bands = tapered(50.0, 50.0, 0.0, 100.0, true, 10.0, 4);
        // Top cap, arc (4), wall, arc (4), bottom cap.
        assert_eq!(bands.len(), 11);
        let first = bands[0];
        assert_eq!((first.0.r, first.0.y, first.1.r), (0.0, 0.0, 40.0));
        assert_eq!(first.0.n, [0.0, -1.0], "the top cap faces up");
        let wall = bands[5];
        assert!((wall.0.r - 50.0).abs() < 1e-9 && (wall.0.y - 10.0).abs() < 1e-9 && (wall.1.y - 90.0).abs() < 1e-9);
        assert!((wall.0.n[0] - 1.0).abs() < 1e-9);
        // The arc's middle sample leans out and up at 45°, on a 10 px circle about (40, 10).
        let mid = bands[2].1;
        assert!((mid.n[0] - mid.n[1].abs()).abs() < 1e-9 && mid.n[1] < 0.0, "{mid:?}");
        assert!(((mid.r - 40.0).hypot(mid.y - 10.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn a_bevel_bigger_than_the_edges_is_held_to_half_of_each() {
        let bands = tapered(20.0, 20.0, 0.0, 10.0, true, 50.0, 2);
        let ys: Vec<f64> = bands.iter().flat_map(|(a, b)| [a.y, b.y]).collect();
        assert!(ys.iter().all(|y| (0.0..=10.0).contains(y)));
        assert!(bands.iter().all(|(a, b)| a.r >= 0.0 && b.r <= 20.0 + 1e-9));
    }

    #[test]
    fn an_open_tube_has_one_band_and_revolves_closed() {
        let bands = tapered(30.0, 30.0, 0.0, 60.0, false, 5.0, 4);
        assert_eq!(bands.len(), 1);
        let mut part = Part::new([1.0; 4]);
        revolve(&bands, &Turn { cx: 50.0, depth_ratio: 0.5, segments: 32, sweep: 360.0 }, &mut part);
        assert_eq!(part.uvs.len(), part.positions.len());
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for p in &part.positions {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        assert!((lo[0] - 20.0).abs() < 1e-4 && (hi[0] - 80.0).abs() < 1e-4);
        assert!((lo[2] + 15.0).abs() < 1e-3 && (hi[2] - 15.0).abs() < 1e-3, "the depth ratio squashes z: {lo:?} {hi:?}");
        // No cap: every normal is horizontal.
        assert!(part.normals.iter().all(|n| n[1].abs() < 1e-6));
    }

    #[test]
    fn a_cone_leans_its_wall_normals_up() {
        let bands = tapered(0.0, 40.0, 0.0, 80.0, true, 0.0, 1);
        assert_eq!(bands.len(), 2, "the wall and the bottom cap");
        let n = bands[0].0.n;
        assert!(n[0] > 0.0 && n[1] < 0.0, "{n:?}");
        // Perpendicular to the slant (40 out over 80 down).
        assert!((n[0] * 40.0 + n[1] * 80.0).abs() < 1e-9);
    }

    #[test]
    fn a_lathed_rectangle_off_the_axis_is_a_thick_ring() {
        // A 20 × 40 rectangle 30 px right of the axis: a tube 30–50 px around it, 40 tall.
        let rect = vec![[130.0, 0.0], [150.0, 0.0], [150.0, 40.0], [130.0, 40.0]];
        let mut part = Part::new([1.0; 4]);
        lathe(&[rect], 100.0, 360.0, 48, &mut part);
        assert_eq!(part.uvs.len(), part.positions.len());
        let r: Vec<f64> = part.positions.iter().map(|p| (f64::from(p[0]) - 100.0).hypot(f64::from(p[2]))).collect();
        assert!(r.iter().all(|r| *r > 29.9 && *r < 50.1), "{:?}", r.iter().copied().fold((f64::INFINITY, 0.0f64), |a, r| (a.0.min(r), a.1.max(r))));
        // The outer wall's normals point away from the axis, the inner's toward it.
        for (p, n) in part.positions.iter().zip(&part.normals) {
            let out = [f64::from(p[0]) - 100.0, f64::from(p[2])];
            let d = out[0] * f64::from(n[0]) + out[1] * f64::from(n[2]);
            let rr = out[0].hypot(out[1]);
            if (rr - 50.0).abs() < 0.01 && n[1].abs() < 1e-6 {
                assert!(d > 0.0, "outer wall {n:?}");
            }
            if (rr - 30.0).abs() < 0.01 && n[1].abs() < 1e-6 {
                assert!(d < 0.0, "inner wall {n:?}");
            }
        }
    }

    #[test]
    fn a_half_turn_closes_its_ends_and_a_part_past_the_axis_is_pressed_onto_it() {
        let rect = vec![[80.0, 0.0], [150.0, 0.0], [150.0, 40.0], [80.0, 40.0]];
        let mut part = Part::new([1.0; 4]);
        lathe(&[rect], 100.0, 180.0, 48, &mut part);
        // The ends: flat caps facing −z at the start and +z at 180°… both in the z = 0 plane.
        let caps = part.normals.iter().filter(|n| n[1].abs() < 1e-6 && (n[2].abs() - 1.0).abs() < 1e-6 && n[0].abs() < 1e-6).count();
        assert!(caps >= 6, "{caps} cap vertices");
        // Nothing left of the axis's mirror: x stays within 100 ± 50, z ≥ 0 (half a turn toward the back).
        assert!(part.positions.iter().all(|p| p[2] >= -1e-3 && (f64::from(p[0]) - 100.0).abs() <= 50.0 + 1e-3));
    }

    #[test]
    fn a_lathed_circle_is_smooth_a_torus_like_ring() {
        // A 64-gon circle of radius 10 centred 40 px right of the axis: every vertex smooth (one normal).
        let circle: Vec<V2> = (0..64).map(|i| {
            let a = f64::from(i) / 64.0 * TAU;
            [140.0 + 10.0 * a.cos(), 50.0 + 10.0 * a.sin()]
        }).collect();
        let bands = contour_bands(&super::super::geometry::orient(&[circle.iter().map(|p| [p[0] - 100.0, p[1]]).collect()])[0]);
        for w in bands.windows(2) {
            assert!((w[0].1.n[0] - w[1].0.n[0]).abs() < 1e-9 && (w[0].1.n[1] - w[1].0.n[1]).abs() < 1e-9, "a crease at {:?}", w[0].1);
        }
        // The normals point away from the circle's centre.
        for (a, _) in &bands {
            assert!((a.r - 40.0) * a.n[0] + (a.y - 50.0) * a.n[1] > 0.0);
        }
    }
}
