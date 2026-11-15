//! Solid geometry: extrusion of filled outlines with a rounded or chamfered bevel, and primitives fitted to a box.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use super::outline::Shape;
use super::{Mesh, Part};

/// How an outline becomes a solid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Extrude {
    /// Front (z = 0) to back (z = depth), px.
    pub depth: f64,
    /// How far the bevel cuts into the outline and into the depth, px (0: sharp edges).
    pub bevel: f64,
    /// Steps around a bevel: 1 is a chamfer, more are rounder.
    pub segments: u32,
}

type V2 = [f64; 2];

fn area(c: &[V2]) -> f64 {
    let n = c.len();
    (0..n).map(|i| c[i][0] * c[(i + 1) % n][1] - c[(i + 1) % n][0] * c[i][1]).sum::<f64>() / 2.0
}

fn inside(p: V2, c: &[V2]) -> bool {
    let mut odd = false;
    let n = c.len();
    for i in 0..n {
        let (a, b) = (c[i], c[(i + 1) % n]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) {
            odd = !odd;
        }
    }
    odd
}

/// Outlines oriented so solids have positive area and holes negative (by how many other contours hold each one), so
/// one normal rule gives every wall its outside.
pub(crate) fn orient(contours: &[Vec<V2>]) -> Vec<Vec<V2>> {
    contours
        .iter()
        .enumerate()
        .filter(|(_, c)| area(c).abs() > 1e-9)
        .map(|(i, c)| {
            // A point just inside the contour's first edge, so a shared vertex doesn't count.
            let probe = {
                let (a, b) = (c[0], c[1]);
                [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
            };
            let depth = contours.iter().enumerate().filter(|(j, o)| *j != i && area(o).abs() > 1e-9 && inside(probe, o)).count();
            let want_positive = depth % 2 == 0;
            let mut c = c.clone();
            if (area(&c) > 0.0) != want_positive {
                c.reverse();
            }
            c
        })
        .collect()
}

/// One place on a contour's rings: the vertex and the outward normal its walls use there (a sharp corner has two).
struct Entry {
    vertex: usize,
    normal: V2,
}

struct Ring {
    points: Vec<V2>,
    /// Per vertex: the offset direction (inward = −miter), scaled so a unit offset moves each edge by one.
    miters: Vec<V2>,
    entries: Vec<Entry>,
}

fn norm2(v: V2) -> V2 {
    let l = v[0].hypot(v[1]);
    if l > 1e-12 { [v[0] / l, v[1] / l] } else { [0.0, 0.0] }
}

/// A corner sharper than this (between the two edges' normals) keeps a crease.
pub(crate) const CREASE_COS: f64 = 0.819; // cos 35°

fn ring(c: &[V2]) -> Ring {
    let n = c.len();
    let edge_normal = |i: usize| {
        let (a, b) = (c[i], c[(i + 1) % n]);
        norm2([b[1] - a[1], -(b[0] - a[0])])
    };
    let normals: Vec<V2> = (0..n).map(edge_normal).collect();
    let mut miters = Vec::with_capacity(n);
    let mut entries = Vec::with_capacity(n + n / 2);
    for i in 0..n {
        let (np, nn) = (normals[(i + n - 1) % n], normals[i]);
        let d = np[0] * nn[0] + np[1] * nn[1];
        let m = [np[0] + nn[0], np[1] + nn[1]];
        // The miter, capped at twice the offset so a spike doesn't shoot off.
        let k = if 1.0 + d > 0.25 { 1.0 / (1.0 + d) } else { 4.0 };
        let mut mv = [m[0] * k, m[1] * k];
        let ml = mv[0].hypot(mv[1]);
        if ml > 2.0 {
            mv = [mv[0] * 2.0 / ml, mv[1] * 2.0 / ml];
        }
        miters.push(mv);
        if d < CREASE_COS {
            entries.push(Entry { vertex: i, normal: np });
            entries.push(Entry { vertex: i, normal: nn });
        } else {
            entries.push(Entry { vertex: i, normal: norm2(m) });
        }
    }
    Ring { points: c.to_vec(), miters, entries }
}

impl Ring {
    fn at(&self, i: usize, inset: f64) -> V2 {
        let (p, m) = (self.points[i], self.miters[i]);
        [p[0] - inset * m[0], p[1] - inset * m[1]]
    }
}

/// A profile sample across the edge: how far in, how deep, and the edge normal's (outward, z) components.
#[derive(Clone, Copy)]
struct Sample {
    inset: f64,
    z: f64,
    out: f64,
    nz: f64,
}

fn profile(depth: f64, bevel: f64, segments: u32) -> Vec<(Sample, Sample)> {
    let mut bands = Vec::new();
    if bevel <= 0.0 {
        let a = Sample { inset: 0.0, z: 0.0, out: 1.0, nz: 0.0 };
        bands.push((a, Sample { z: depth, ..a }));
        return bands;
    }
    let seg = segments.max(1);
    let front: Vec<Sample> = (0..=seg)
        .map(|k| {
            let t = f64::from(k) / f64::from(seg) * FRAC_PI_2;
            let (s, c) = t.sin_cos();
            let (out, nz) = if seg == 1 { (std::f64::consts::FRAC_1_SQRT_2, -std::f64::consts::FRAC_1_SQRT_2) } else { (s, -c) };
            Sample { inset: bevel * (1.0 - s), z: bevel * (1.0 - c), out, nz }
        })
        .collect();
    for w in front.windows(2) {
        bands.push((w[0], w[1]));
    }
    let wall = Sample { inset: 0.0, z: bevel, out: 1.0, nz: 0.0 };
    if depth - 2.0 * bevel > 1e-9 {
        bands.push((wall, Sample { z: depth - bevel, ..wall }));
    }
    for w in front.windows(2).rev() {
        let m = |s: Sample| Sample { z: depth - s.z, nz: -s.nz, ..s };
        bands.push((m(w[1]), m(w[0])));
    }
    bands
}

/// Tessellate a cap: the contours (at `inset`) as triangles in `part` at depth `z`, facing `nz`, its UVs the box
/// `(lo, hi)` spread over 0–1.
fn cap(part: &mut Part, rings: &[Ring], inset: f64, z: f64, nz: f64, (lo, hi): (V2, V2)) {
    use lyon_tessellation::math::point;
    use lyon_tessellation::path::PathEvent;
    use lyon_tessellation::{BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, VertexBuffers};
    let contours: Vec<Vec<V2>> = rings.iter().map(|r| (0..r.points.len()).map(|i| r.at(i, inset)).collect()).collect();
    let events = contours.iter().filter(|c| c.len() >= 3).flat_map(|c| {
        let pt = |p: &V2| point(p[0] as f32, p[1] as f32);
        let first = pt(&c[0]);
        let last = pt(&c[c.len() - 1]);
        std::iter::once(PathEvent::Begin { at: first })
            .chain(c.windows(2).map(move |w| PathEvent::Line { from: pt(&w[0]), to: pt(&w[1]) }))
            .chain(std::iter::once(PathEvent::End { last, first, close: true }))
    });
    let mut buffers: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    let ok = FillTessellator::new().tessellate(
        events,
        &FillOptions::DEFAULT.with_fill_rule(FillRule::NonZero),
        &mut BuffersBuilder::new(&mut buffers, |v: FillVertex| v.position().to_array()),
    );
    if ok.is_err() {
        return;
    }
    let base = part.positions.len() as u32;
    let (sx, sy) = (1.0 / (hi[0] - lo[0]).max(1e-9), 1.0 / (hi[1] - lo[1]).max(1e-9));
    for p in &buffers.vertices {
        let (x, y) = (f64::from(p[0]), f64::from(p[1]));
        part.vertex([x, y, z], [0.0, 0.0, nz]);
        part.uvs.push([((x - lo[0]) * sx) as f32, ((y - lo[1]) * sy) as f32]);
    }
    for t in buffers.indices.chunks_exact(3) {
        part.tri(base + t[0], base + t[1], base + t[2]);
    }
}

/// Extrude filled outlines (a part per colour) with an optional bevel (clamped to half the depth and to a quarter
/// of each shape's smaller side). UVs: the caps' are the box of every shape spread over 0–1 (a texture across a whole
/// word), the walls' u runs along each contour and v from front to back.
pub fn extrude(shapes: &[Shape], p: &Extrude) -> Mesh {
    let depth = p.depth.max(0.0);
    let mut mesh = Mesh::default();
    let (mut glo, mut ghi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for q in shapes.iter().flat_map(|s| s.contours.iter().flatten()) {
        glo = [glo[0].min(q[0]), glo[1].min(q[1])];
        ghi = [ghi[0].max(q[0]), ghi[1].max(q[1])];
    }
    for shape in shapes {
        let contours = orient(&shape.contours);
        if contours.is_empty() {
            continue;
        }
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for q in contours.iter().flatten() {
            lo = [lo[0].min(q[0]), lo[1].min(q[1])];
            hi = [hi[0].max(q[0]), hi[1].max(q[1])];
        }
        let bevel = p.bevel.max(0.0).min(depth / 2.0).min((hi[0] - lo[0]).min(hi[1] - lo[1]) / 4.0);
        let rings: Vec<Ring> = contours.iter().map(|c| ring(c)).collect();
        let mut part = Part::new(shape.color);
        if depth > 0.0 {
            for (a, b) in profile(depth, bevel, p.segments) {
                for r in &rings {
                    let n = r.entries.len();
                    // How far along the contour each vertex is, 0–1 (its first vertex is 0 and, closing it, 1).
                    let m = r.points.len();
                    let mut arc = vec![0.0; m + 1];
                    for i in 0..m {
                        let (p0, p1) = (r.points[i], r.points[(i + 1) % m]);
                        arc[i + 1] = arc[i] + (p1[0] - p0[0]).hypot(p1[1] - p0[1]);
                    }
                    let total = arc[m].max(1e-9);
                    for j in 0..n {
                        let (e0, e1) = (&r.entries[j], &r.entries[(j + 1) % n]);
                        if e0.vertex == e1.vertex {
                            continue;
                        }
                        let v = |e: &Entry, s: Sample, closing: bool, part: &mut Part| {
                            let q = r.at(e.vertex, s.inset);
                            let u = if closing { 1.0 } else { arc[e.vertex] / total };
                            part.uvs.push([u as f32, (s.z / depth) as f32]);
                            part.vertex([q[0], q[1], s.z], [e.normal[0] * s.out, e.normal[1] * s.out, s.nz])
                        };
                        let closing = e1.vertex < e0.vertex;
                        let (i0, i1) = (v(e0, a, false, &mut part), v(e1, a, closing, &mut part));
                        let (i2, i3) = (v(e1, b, closing, &mut part), v(e0, b, false, &mut part));
                        part.quad(i0, i1, i2, i3);
                    }
                }
            }
        }
        cap(&mut part, &rings, bevel, 0.0, -1.0, (glo, ghi));
        if depth > 0.0 {
            cap(&mut part, &rings, bevel, depth, 1.0, (glo, ghi));
        }
        if part.uvs.len() != part.positions.len() {
            part.uvs.clear();
        }
        mesh.parts.push(part);
    }
    mesh
}

/// A primitive fitted to a layer's box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Primitive {
    /// The box's width × height, `depth` deep, centred on the plane; its edges bevelled.
    Box { depth: f64, bevel: f64, segments: u32 },
    /// An ellipsoid filling the box, `depth` deep.
    Sphere { depth: f64 },
    /// A cylinder standing in the box (its axis vertical), `depth` across front to back; its top's radius `top` times
    /// its bottom's (0: a cone, its tip at the top); with or without its caps, their rims filleted by `bevel` px in
    /// `segments` steps.
    Cylinder { depth: f64, top: f64, caps: bool, bevel: f64, segments: u32 },
    /// A ring facing the viewer, filling the box; `tube` is its thickness (px).
    Torus { tube: f64 },
    /// A lit card: the box itself.
    Plane,
}

/// `prim` in the box `(x0, y0, x1, y1)`, `segments` around its curves, in `color`.
pub fn primitive(prim: Primitive, (x0, y0, x1, y1): (f64, f64, f64, f64), segments: u32, color: [f32; 4]) -> Mesh {
    let (w, h) = (x1 - x0, y1 - y0);
    let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let n = segments.clamp(8, 256) as usize;
    let mut part = Part::new(color);
    match prim {
        Primitive::Box { depth, bevel, segments } => {
            let rect = Shape { contours: vec![vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], color };
            let mut m = extrude(&[rect], &Extrude { depth, bevel, segments });
            for p in m.parts.iter_mut().flat_map(|p| p.positions.iter_mut()) {
                p[2] -= (depth / 2.0) as f32;
            }
            return m;
        }
        Primitive::Sphere { depth } => {
            let (rx, ry, rz) = (w / 2.0, h / 2.0, depth / 2.0);
            let rings = n / 2;
            for i in 0..=rings {
                let v = i as f64 / rings as f64 * PI;
                let (sv, cv) = v.sin_cos();
                for j in 0..=n {
                    let u = j as f64 / n as f64 * TAU;
                    let (su, cu) = u.sin_cos();
                    let d = [sv * cu, -cv, sv * su];
                    let idx = part.vertex([cx + rx * d[0], cy + ry * d[1], rz * d[2]], [d[0] / rx.max(1e-9), d[1] / ry.max(1e-9), d[2] / rz.max(1e-9)]);
                    part.uvs.push([j as f32 / n as f32, i as f32 / rings as f32]);
                    if i > 0 && j > 0 {
                        let s = (n + 1) as u32;
                        part.quad(idx - s - 1, idx - s, idx, idx - 1);
                    }
                }
            }
        }
        Primitive::Cylinder { depth, top, caps, bevel, segments: bevel_segments } => {
            let (rx, rz) = (w / 2.0, depth / 2.0);
            let bands = super::revolve::tapered(rx * top.clamp(0.0, 1.0), rx, y0, y1, caps, bevel.max(0.0), bevel_segments);
            let turn = super::revolve::Turn { cx, depth_ratio: rz / rx.max(1e-9), segments: n as u32, sweep: 360.0 };
            super::revolve::revolve(&bands, &turn, &mut part);
        }
        Primitive::Torus { tube } => {
            let r = (tube / 2.0).clamp(0.5, w.min(h) / 4.0);
            let (big_x, big_y) = (w / 2.0 - r, h / 2.0 - r);
            let m = (n / 2).max(8);
            for i in 0..=n {
                let u = i as f64 / n as f64 * TAU;
                let (su, cu) = u.sin_cos();
                for j in 0..=m {
                    let v = j as f64 / m as f64 * TAU;
                    let (sv, cv) = v.sin_cos();
                    // Around the ring in the plane; around the tube out of it.
                    let dir = [cu * cv, su * cv, sv];
                    let idx = part.vertex([cx + big_x * cu + r * dir[0], cy + big_y * su + r * dir[1], r * dir[2]], dir);
                    part.uvs.push([i as f32 / n as f32, j as f32 / m as f32]);
                    if i > 0 && j > 0 {
                        let s = (m + 1) as u32;
                        part.quad(idx - s - 1, idx - s, idx, idx - 1);
                    }
                }
            }
        }
        Primitive::Plane => {
            let a = part.vertex([x0, y0, 0.0], [0.0, 0.0, -1.0]);
            let b = part.vertex([x1, y0, 0.0], [0.0, 0.0, -1.0]);
            let c = part.vertex([x1, y1, 0.0], [0.0, 0.0, -1.0]);
            let d = part.vertex([x0, y1, 0.0], [0.0, 0.0, -1.0]);
            part.uvs.extend([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
            part.quad(a, b, c, d);
        }
    }
    if part.uvs.len() != part.positions.len() {
        part.uvs.clear();
    }
    Mesh { parts: vec![part] }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: f64, y: f64, s: f64) -> Vec<V2> {
        vec![[x, y], [x + s, y], [x + s, y + s], [x, y + s]]
    }

    #[test]
    fn a_square_with_a_hole_extrudes_to_a_closed_solid_with_outward_walls() {
        let shape = Shape { contours: vec![square(0.0, 0.0, 100.0), square(25.0, 25.0, 50.0)], color: [1.0, 0.0, 0.0, 1.0] };
        let m = extrude(&[shape], &Extrude { depth: 40.0, bevel: 0.0, segments: 1 });
        let (lo, hi) = m.bounds().unwrap();
        assert_eq!((lo, hi), ([0.0, 0.0, 0.0], [100.0, 100.0, 40.0]));
        let p = &m.parts[0];
        // Every wall normal points away from the solid: out of the outer square, into the hole.
        for (q, n) in p.positions.iter().zip(&p.normals) {
            if n[2].abs() > 0.5 {
                continue;
            }
            let (x, y) = (f64::from(q[0]), f64::from(q[1]));
            let outer = x <= 0.0 || x >= 100.0 || y <= 0.0 || y >= 100.0;
            let probe = [x + f64::from(n[0]), y + f64::from(n[1])];
            // Strictly inside the solid (half a pixel from any edge), so a corner's probe along one wall isn't counted.
            let in_solid = (0.5..=99.5).contains(&probe[0]) && (0.5..=99.5).contains(&probe[1]) && !((24.5..75.5).contains(&probe[0]) && (24.5..75.5).contains(&probe[1]));
            assert!(!in_solid, "wall at ({x}, {y}) outer {outer} faces into the solid: {n:?}");
        }
        // The front cap covers the ring: 100² − 50² of area.
        let front: f64 = p
            .indices
            .chunks_exact(3)
            .filter(|t| t.iter().all(|&i| p.normals[i as usize][2] < -0.5 && p.positions[i as usize][2] == 0.0))
            .map(|t| {
                let [a, b, c] = [t[0], t[1], t[2]].map(|i| p.positions[i as usize]);
                (f64::from((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0).abs()
            })
            .sum();
        assert!((front - 7500.0).abs() < 1.0, "{front}");
    }

    #[test]
    fn a_bevel_insets_the_caps_and_rounds_the_edge() {
        // A hole wound the same way as its outline (even-odd data): oriented by nesting, it still extrudes right.
        let shape = Shape { contours: vec![square(0.0, 0.0, 100.0)], color: [1.0; 4] };
        let m = extrude(&[shape], &Extrude { depth: 30.0, bevel: 5.0, segments: 4 });
        let p = &m.parts[0];
        let cap: Vec<_> = p.positions.iter().zip(&p.normals).filter(|(q, n)| q[2] == 0.0 && n[2] < -0.99).map(|(q, _)| *q).collect();
        let min_x = cap.iter().map(|q| q[0]).fold(f32::INFINITY, f32::min);
        assert!((min_x - 5.0).abs() < 1e-4, "the front cap is inset by the bevel: {min_x}");
        // The outline's full size is reached at the wall.
        let (lo, hi) = m.bounds().unwrap();
        assert!((lo[0] - 0.0).abs() < 1e-6 && (hi[0] - 100.0).abs() < 1e-6 && (hi[2] - 30.0).abs() < 1e-6);
        // Some normals lean between front and outward (the rounded edge).
        assert!(p.normals.iter().any(|n| n[2] < -0.3 && n[2] > -0.95));
    }

    #[test]
    fn primitives_fill_their_box() {
        for (prim, depth) in [
            (Primitive::Sphere { depth: 60.0 }, 60.0),
            (Primitive::Cylinder { depth: 60.0, top: 1.0, caps: true, bevel: 0.0, segments: 1 }, 60.0),
            (Primitive::Cylinder { depth: 60.0, top: 0.0, caps: true, bevel: 0.0, segments: 1 }, 60.0),
            (Primitive::Cylinder { depth: 60.0, top: 1.0, caps: true, bevel: 8.0, segments: 4 }, 60.0),
            (Primitive::Box { depth: 60.0, bevel: 4.0, segments: 3 }, 60.0),
        ] {
            let m = primitive(prim, (10.0, 20.0, 110.0, 220.0), 32, [1.0; 4]);
            let (lo, hi) = m.bounds().unwrap();
            assert!((lo[0] - 10.0).abs() < 1e-3 && (hi[0] - 110.0).abs() < 1e-3, "{prim:?} {lo:?} {hi:?}");
            assert!((lo[1] - 20.0).abs() < 1e-3 && (hi[1] - 220.0).abs() < 1e-3, "{prim:?} {lo:?} {hi:?}");
            assert!((hi[2] - lo[2] - depth).abs() < 1e-3 && (hi[2] + lo[2]).abs() < 1e-3, "{prim:?} centred: {lo:?} {hi:?}");
            assert!(m.triangles() > 10);
        }
        let t = primitive(Primitive::Torus { tube: 20.0 }, (0.0, 0.0, 100.0, 100.0), 32, [1.0; 4]);
        let (lo, hi) = t.bounds().unwrap();
        assert!((lo[0]).abs() < 1e-3 && (hi[0] - 100.0).abs() < 1e-3 && (hi[2] - 10.0).abs() < 1e-3);
    }
}
