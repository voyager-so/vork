//! Stroke gradient coordinates (E8.18): Illustrator's "Apply gradient along
//! stroke" and "Apply gradient across stroke".
//!
//! Every stroke vertex of a body's tessellation gets `(along, across)`:
//!
//! - `along` — the arc length of the vertex's closest point on the path's
//!   centreline, as a fraction of its sub-path's length (each sub-path runs
//!   0 → 1 from its start, as Illustrator's compound paths do). On a closed
//!   sub-path, a triangle straddling the start/end seam has its low values
//!   lifted by 1 so it interpolates through the seam instead of back across
//!   the whole ramp; the shader wraps values past 1 (`along − 1`), so the seam
//!   is the hard edge Illustrator draws at a closed path's start.
//! - `across` — 0 on the stroke's left edge (walking the path's direction),
//!   1 on its right, measured against the stroke's own extent at that arc
//!   length: the signed perpendicular offsets of the tessellation's vertices,
//!   binned along the path. So an inside / outside aligned stroke spans its
//!   one side, and a variable-width stroke (width profile, taper, brush
//!   outline) stretches the ramp over its local width, as Illustrator does.
//!
//! The coordinates ride a third vertex stream (`TessellatedGeometry::
//! stroke_uv`), which only the path-gradient pipeline reads; fill vertices
//! carry zeros.

use super::moda_engine_path_types::PathCmd;
use crate::seams::TessellatedGeometry;

/// One flattened sub-path of the centreline.
#[derive(Debug, Clone)]
struct Sub {
    pts: Vec<[f64; 2]>,
    /// Arc length at each point.
    cum: Vec<f64>,
    closed: bool,
}

impl Sub {
    fn len(&self) -> f64 {
        *self.cum.last().unwrap_or(&0.0)
    }
}

/// The path's centreline, flattened.
#[derive(Debug, Clone, Default)]
pub struct Centerline {
    subs: Vec<Sub>,
}

fn cubic(
    p0: [f64; 2],
    c1: [f64; 2],
    c2: [f64; 2],
    p1: [f64; 2],
    tol: f64,
    out: &mut Vec<[f64; 2]>,
) {
    // Enough pieces that the chord error (≈ control-polygon deviation / 8·n²) stays under `tol`.
    let dev = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        let x = a[0] - 2.0 * b[0] + c[0];
        let y = a[1] - 2.0 * b[1] + c[1];
        x.hypot(y)
    };
    let d = dev(p0, c1, c2).max(dev(c1, c2, p1));
    let n = ((d * 0.75 / tol.max(1e-6)).sqrt().ceil() as usize).clamp(1, 512);
    for i in 1..=n {
        let t = i as f64 / n as f64;
        let u = 1.0 - t;
        let (a, b, c, e) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
        out.push([
            a * p0[0] + b * c1[0] + c * c2[0] + e * p1[0],
            a * p0[1] + b * c1[1] + c * c2[1] + e * p1[1],
        ]);
    }
}

/// Flatten `cmds` (the drawn outline, node-local px) to within `tol`. A
/// sub-path closes when `closed` is set (every sub-path of a closed shape
/// closes, as the tessellator strokes it).
pub fn centerline(cmds: &[PathCmd], closed: bool, tol: f64) -> Centerline {
    let mut subs: Vec<Sub> = Vec::new();
    let mut pts: Vec<[f64; 2]> = Vec::new();
    let flush = |pts: &mut Vec<[f64; 2]>, subs: &mut Vec<Sub>| {
        if pts.len() >= 2 {
            let mut p = std::mem::take(pts);
            if closed && p.first() != p.last() {
                p.push(p[0]);
            }
            let mut cum = Vec::with_capacity(p.len());
            let mut acc = 0.0;
            for (i, q) in p.iter().enumerate() {
                if i > 0 {
                    acc += (q[0] - p[i - 1][0]).hypot(q[1] - p[i - 1][1]);
                }
                cum.push(acc);
            }
            subs.push(Sub {
                pts: p,
                cum,
                closed,
            });
        }
        pts.clear();
    };
    for cmd in cmds {
        let last = pts.last().copied();
        match *cmd {
            PathCmd::Move { x, y } => {
                flush(&mut pts, &mut subs);
                pts.push([x, y]);
            }
            PathCmd::Line { x, y } => pts.push([x, y]),
            PathCmd::Curve { x, y, cp1, cp2 } => match last {
                Some(p0) => cubic(p0, [cp1.0, cp1.1], [cp2.0, cp2.1], [x, y], tol, &mut pts),
                None => pts.push([x, y]),
            },
            PathCmd::Quadratic { x, y, cp } => match last {
                Some(p0) => {
                    let c1 = [
                        p0[0] + 2.0 / 3.0 * (cp.0 - p0[0]),
                        p0[1] + 2.0 / 3.0 * (cp.1 - p0[1]),
                    ];
                    let c2 = [x + 2.0 / 3.0 * (cp.0 - x), y + 2.0 / 3.0 * (cp.1 - y)];
                    cubic(p0, c1, c2, [x, y], tol, &mut pts);
                }
                None => pts.push([x, y]),
            },
            PathCmd::Arc {
                x,
                y,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            } => match last {
                Some(p0) => {
                    let arc = super::arc::ArcParams {
                        x,
                        y,
                        rx,
                        ry,
                        x_axis_rotation,
                        large_arc,
                        sweep,
                    };
                    let beziers = super::arc::arc_to_beziers(p0[0], p0[1], &arc);
                    if beziers.is_empty() {
                        pts.push([x, y]);
                    }
                    let mut from = p0;
                    for b in beziers {
                        cubic(from, b.cp1, b.cp2, b.end, tol, &mut pts);
                        from = b.end;
                    }
                }
                None => pts.push([x, y]),
            },
        }
    }
    flush(&mut pts, &mut subs);
    Centerline { subs }
}

/// One centreline segment, for the nearest-point search.
#[derive(Debug, Clone, Copy)]
struct Seg {
    a: [f64; 2],
    b: [f64; 2],
    sub: usize,
    /// Arc length at `a`.
    s0: f64,
}

/// A vertex's closest centreline point: sub-path, arc length, signed
/// perpendicular offset (> 0 on the right of the path's direction, y down).
#[derive(Debug, Clone, Copy)]
struct Foot {
    sub: usize,
    s: f64,
    d: f64,
}

/// A uniform grid over the segments' boxes: the nearest segment to a point is
/// found ring by ring outward from its cell.
struct Grid {
    segs: Vec<Seg>,
    origin: [f64; 2],
    cell: f64,
    cols: usize,
    rows: usize,
    cells: Vec<Vec<u32>>,
}

impl Grid {
    fn new(line: &Centerline) -> Option<Grid> {
        let mut segs = Vec::new();
        for (i, sub) in line.subs.iter().enumerate() {
            for k in 1..sub.pts.len() {
                segs.push(Seg {
                    a: sub.pts[k - 1],
                    b: sub.pts[k],
                    sub: i,
                    s0: sub.cum[k - 1],
                });
            }
        }
        if segs.is_empty() {
            return None;
        }
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for s in &segs {
            for p in [s.a, s.b] {
                lo = [lo[0].min(p[0]), lo[1].min(p[1])];
                hi = [hi[0].max(p[0]), hi[1].max(p[1])];
            }
        }
        let extent = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-6);
        // About as many cells as segments, at most 256 a side.
        let side = ((segs.len() as f64).sqrt().ceil() as usize).clamp(1, 256);
        let cell = extent / side as f64;
        let cols = (((hi[0] - lo[0]) / cell).floor() as usize + 1).min(257);
        let rows = (((hi[1] - lo[1]) / cell).floor() as usize + 1).min(257);
        let mut cells = vec![Vec::new(); cols * rows];
        let at = |v: f64, o: f64, n: usize| (((v - o) / cell).floor().max(0.0) as usize).min(n - 1);
        for (i, s) in segs.iter().enumerate() {
            let (c0, c1) = (
                at(s.a[0].min(s.b[0]), lo[0], cols),
                at(s.a[0].max(s.b[0]), lo[0], cols),
            );
            let (r0, r1) = (
                at(s.a[1].min(s.b[1]), lo[1], rows),
                at(s.a[1].max(s.b[1]), lo[1], rows),
            );
            for r in r0..=r1 {
                for c in c0..=c1 {
                    cells[r * cols + c].push(i as u32);
                }
            }
        }
        Some(Grid {
            segs,
            origin: lo,
            cell,
            cols,
            rows,
            cells,
        })
    }

    fn foot_on(&self, i: usize, p: [f64; 2]) -> (f64, Foot) {
        let s = &self.segs[i];
        let (dx, dy) = (s.b[0] - s.a[0], s.b[1] - s.a[1]);
        let len2 = dx * dx + dy * dy;
        let t = if len2 > 0.0 {
            (((p[0] - s.a[0]) * dx + (p[1] - s.a[1]) * dy) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (qx, qy) = (s.a[0] + dx * t, s.a[1] + dy * t);
        let dist = (p[0] - qx).hypot(p[1] - qy);
        let len = len2.sqrt();
        // Perpendicular offset against the segment's direction (cross product), so a cap vertex past an
        // end measures across the stroke, not along it.
        let d = if len > 0.0 {
            (dx * (p[1] - s.a[1]) - dy * (p[0] - s.a[0])) / len
        } else {
            dist
        };
        (
            dist,
            Foot {
                sub: s.sub,
                s: s.s0 + len * t,
                d,
            },
        )
    }

    fn nearest(&self, p: [f64; 2]) -> Foot {
        self.nearest_near(p, None)
    }

    /// The closest centreline point to `p`. Points within [`TIE`] of the
    /// closest (an inner join's vertex sits as near one leg as the other)
    /// resolve toward `near` — the arc position of the triangle the vertex
    /// belongs to — so a triangle measures against the leg it lies along.
    fn nearest_near(&self, p: [f64; 2], near: Option<Foot>) -> Foot {
        let cx = ((p[0] - self.origin[0]) / self.cell).floor() as i64;
        let cy = ((p[1] - self.origin[1]) / self.cell).floor() as i64;
        let mut found: Vec<(f64, Foot)> = Vec::new();
        let mut best = f64::INFINITY;
        let max_ring = self.cols.max(self.rows) as i64 + 1 + cx.abs().max(cy.abs());
        for ring in 0..=max_ring {
            // Every point in a cell `ring` away is at least (ring − 1) cells from p.
            if (ring - 1) as f64 * self.cell > best + TIE {
                break;
            }
            for r in cy - ring..=cy + ring {
                for c in cx - ring..=cx + ring {
                    if (r - cy).abs() != ring && (c - cx).abs() != ring {
                        continue;
                    }
                    if r < 0 || c < 0 || r >= self.rows as i64 || c >= self.cols as i64 {
                        continue;
                    }
                    for &i in &self.cells[r as usize * self.cols + c as usize] {
                        let (dist, foot) = self.foot_on(i as usize, p);
                        best = best.min(dist);
                        found.push((dist, foot));
                    }
                }
            }
        }
        let gap = |f: &Foot| match near {
            Some(n) if n.sub == f.sub => (f.s - n.s).abs(),
            Some(_) => f64::INFINITY,
            None => 0.0,
        };
        found
            .into_iter()
            .filter(|(d, _)| *d <= best + TIE)
            .min_by(|a, b| gap(&a.1).total_cmp(&gap(&b.1)).then(a.0.total_cmp(&b.0)))
            .map(|(_, f)| f)
            .unwrap_or(Foot {
                sub: 0,
                s: 0.0,
                d: 0.0,
            })
    }
}

/// Distances (px) closer than this to the nearest count as ties.
const TIE: f64 = 0.05;

/// Bins of arc length per sub-path holding the stroke's extent (min and max
/// perpendicular offset) there; empty (or degenerate) bins take their
/// populated neighbours' extents, interpolated (around the seam on a closed
/// sub-path).
struct Extents {
    bins: Vec<Vec<(f64, f64)>>,
    bin_len: Vec<f64>,
}

impl Extents {
    /// One extent everywhere: the stroke's least and greatest offset. A
    /// uniform-width band (centred, inside or outside, its joins and caps
    /// included) spans exactly that at every arc length.
    fn uniform(line: &Centerline, feet: &[Foot]) -> Extents {
        let lo = feet.iter().map(|f| f.d).fold(f64::INFINITY, f64::min);
        let hi = feet.iter().map(|f| f.d).fold(f64::NEG_INFINITY, f64::max);
        let e = if lo <= hi { (lo, hi) } else { (0.0, 0.0) };
        Extents {
            bins: line.subs.iter().map(|_| vec![e]).collect(),
            bin_len: line.subs.iter().map(|s| s.len().max(1e-9)).collect(),
        }
    }

    fn new(line: &Centerline, feet: &[Foot]) -> Extents {
        let mut bins: Vec<Vec<Option<(f64, f64)>>> = Vec::new();
        let mut bin_len = Vec::new();
        for sub in &line.subs {
            let n = ((sub.len() / 2.0).ceil() as usize).clamp(1, 4096);
            bins.push(vec![None; n]);
            bin_len.push(sub.len().max(1e-9) / n as f64);
        }
        for f in feet {
            let Some(b) = bins.get_mut(f.sub) else {
                continue;
            };
            let i = ((f.s / bin_len[f.sub]) as usize).min(b.len() - 1);
            b[i] = Some(match b[i] {
                Some((lo, hi)) => (lo.min(f.d), hi.max(f.d)),
                None => (f.d, f.d),
            });
        }
        let filled = bins
            .into_iter()
            .zip(&line.subs)
            .map(|(b, sub)| {
                let n = b.len();
                // A bin whose vertices all sit at one offset (a join's pivot on the path, a lone cap vertex)
                // says nothing about the stroke's width there: its neighbours decide.
                let b: Vec<Option<(f64, f64)>> = b
                    .into_iter()
                    .map(|v| v.filter(|(lo, hi)| hi - lo > 1e-3))
                    .collect();
                let known: Vec<usize> = (0..n).filter(|&i| b[i].is_some()).collect();
                if known.is_empty() {
                    return vec![(0.0, 0.0); n];
                }
                (0..n)
                    .map(|i| {
                        if let Some(v) = b[i] {
                            return v;
                        }
                        // The populated bins before and after (wrapping on a closed sub-path).
                        let after = known.iter().copied().find(|&k| k > i);
                        let before = known.iter().copied().rev().find(|&k| k < i);
                        let (a, z) = match (before, after, sub.closed) {
                            (Some(a), Some(z), _) => (a as f64, z as f64),
                            (Some(a), None, true) => (a as f64, (known[0] + n) as f64),
                            (None, Some(z), true) => {
                                (*known.last().unwrap() as f64 - n as f64, z as f64)
                            }
                            (Some(a), None, false) => return b[a].unwrap(),
                            (None, Some(z), false) => return b[z].unwrap(),
                            (None, None, _) => unreachable!(),
                        };
                        let va = b[(a.rem_euclid(n as f64)) as usize].unwrap();
                        let vz = b[(z.rem_euclid(n as f64)) as usize].unwrap();
                        let t = (i as f64 - a) / (z - a);
                        (va.0 + (vz.0 - va.0) * t, va.1 + (vz.1 - va.1) * t)
                    })
                    .collect()
            })
            .collect();
        Extents {
            bins: filled,
            bin_len,
        }
    }

    fn at(&self, f: &Foot) -> (f64, f64) {
        let b = &self.bins[f.sub];
        b[((f.s / self.bin_len[f.sub]) as usize).min(b.len() - 1)]
    }
}

/// Subdivision limits for triangles over which the stroke's width changes.
const MAX_DEPTH: u32 = 6;
const MIN_EDGE: f64 = 1.5;
const WIDTH_RATIO: f64 = 1.02;

struct Emit<'a> {
    grid: &'a Grid,
    line: &'a Centerline,
    extents: &'a Extents,
    positions: Vec<f32>,
    types: Vec<f32>,
    uv: Vec<f32>,
    /// Vertices left for subdivision before it stops (a memory bound).
    budget: usize,
}

impl Emit<'_> {
    fn width(&self, f: &Foot) -> f64 {
        let (lo, hi) = self.extents.at(f);
        hi - lo
    }

    /// One triangle, split along its longest edge while the stroke's width
    /// changes over it (across is not affine there).
    fn triangle(&mut self, v: [([f64; 2], Foot); 3], near: Foot, depth: u32) {
        let w = v.map(|(_, f)| self.width(&f));
        let (wmin, wmax) = (
            w.iter().copied().fold(f64::INFINITY, f64::min),
            w.iter().copied().fold(0.0, f64::max),
        );
        let len = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
        let edges = [
            len(v[0].0, v[1].0),
            len(v[1].0, v[2].0),
            len(v[2].0, v[0].0),
        ];
        let (k, longest) = edges
            .iter()
            .copied()
            .enumerate()
            .fold((0, 0.0), |b, (i, e)| if e > b.1 { (i, e) } else { b });
        if depth < MAX_DEPTH
            && self.budget >= 3
            && longest > MIN_EDGE
            && wmax > wmin * WIDTH_RATIO + 1e-6
        {
            self.budget -= 3;
            let (a, b, c) = (v[k], v[(k + 1) % 3], v[(k + 2) % 3]);
            let m = [(a.0[0] + b.0[0]) / 2.0, (a.0[1] + b.0[1]) / 2.0];
            let mf = self.grid.nearest_near(m, Some(near));
            self.triangle([a, (m, mf), c], near, depth + 1);
            self.triangle([(m, mf), b, c], near, depth + 1);
            return;
        }
        let start = self.uv.len();
        for (p, f) in v {
            let sub = &self.line.subs[f.sub];
            let along = if sub.len() > 0.0 {
                (f.s / sub.len()).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (lo, hi) = self.extents.at(&f);
            let across = if hi - lo > 1e-6 {
                ((f.d - lo) / (hi - lo)).clamp(0.0, 1.0)
            } else {
                0.5
            };
            self.positions.extend([p[0] as f32, p[1] as f32]);
            self.types.push(1.0);
            self.uv.extend([along as f32, across as f32]);
        }
        // A closed sub-path's seam: a triangle across it interpolates through
        // the start (values past 1 wrap in the shader) instead of back over
        // the whole ramp.
        if v.iter().all(|(_, f)| f.sub == v[0].1.sub) && self.line.subs[v[0].1.sub].closed {
            let a = [self.uv[start], self.uv[start + 2], self.uv[start + 4]];
            let (lo, hi) = (a[0].min(a[1]).min(a[2]), a[0].max(a[1]).max(a[2]));
            if hi - lo > 0.5 {
                for j in 0..3 {
                    if a[j] < 0.5 {
                        self.uv[start + 2 * j] += 1.0;
                    }
                }
            }
        }
    }
}

/// `geometry` with its stroke triangles carrying `(along, across)`
/// coordinates in `stroke_uv` (fill vertices `(0, 0)`), measured against
/// `line`. `uniform`: the stroke has one width along its length (no width
/// profile, taper, wave or brush outline), so across spans the band's whole
/// offset range; otherwise it spans the local extent. Each triangle resolves an inner join's tied vertices toward its
/// own arc position, and one over which the stroke's width changes (a width
/// profile, taper or brush outline) is split until across is close to
/// affine over each piece. `None` when there are no stroke vertices (or no
/// centreline).
pub fn stroke_uv(
    geometry: &TessellatedGeometry,
    line: &Centerline,
    uniform: bool,
) -> Option<TessellatedGeometry> {
    let n = geometry.vertex_types.len();
    let first = geometry.vertex_types.iter().position(|t| *t > 0.5)?;
    let grid = Grid::new(line)?;
    let at = |v: usize| {
        [
            geometry.positions[2 * v] as f64,
            geometry.positions[2 * v + 1] as f64,
        ]
    };
    let feet: Vec<Foot> = (first..n).map(|v| grid.nearest(at(v))).collect();
    let extents = if uniform {
        Extents::uniform(line, &feet)
    } else {
        Extents::new(line, &feet)
    };
    let mut emit = Emit {
        grid: &grid,
        line,
        extents: &extents,
        positions: geometry.positions[..2 * first].to_vec(),
        types: geometry.vertex_types[..first].to_vec(),
        uv: vec![0.0; 2 * first],
        budget: (n - first) * 8 + 30_000,
    };
    let mut v = first;
    while v + 3 <= n {
        let p = [at(v), at(v + 1), at(v + 2)];
        let centroid = [
            (p[0][0] + p[1][0] + p[2][0]) / 3.0,
            (p[0][1] + p[1][1] + p[2][1]) / 3.0,
        ];
        let near = grid.nearest(centroid);
        let tri = p.map(|q| (q, grid.nearest_near(q, Some(near))));
        emit.triangle(tri, near, 0);
        v += 3;
    }
    let vertex_count = emit.types.len() as u32;
    Some(TessellatedGeometry {
        positions: emit.positions,
        vertex_types: emit.types,
        vertex_count,
        bounds: geometry.bounds,
        fill_dropped: geometry.fill_dropped.clone(),
        stroke_dropped: geometry.stroke_dropped.clone(),
        stroke_uv: emit.uv,
    })
}

/// Flattening tolerance of the centreline (node-local px): well under a
/// device pixel at 1×, and the coordinates interpolate linearly between
/// vertices anyway.
pub const CENTERLINE_TOLERANCE: f64 = 0.1;

/// `geometry` re-emitted by [`stroke_uv`] against the centreline `line`
/// builds, when
/// `node`'s top stroke is a linear or radial gradient laid along or across it
/// (and the stream is not there yet); else `geometry` itself.
pub fn with_stroke_uv(
    geometry: std::sync::Arc<TessellatedGeometry>,
    node: &moda_scene::types::RawNode,
    uniform: bool,
    line: impl FnOnce() -> Centerline,
) -> std::sync::Arc<TessellatedGeometry> {
    if !needs_stroke_uv(node) || !geometry.stroke_uv.is_empty() {
        return geometry;
    }
    match stroke_uv(&geometry, &line(), uniform) {
        Some(g) => std::sync::Arc::new(g),
        None => geometry,
    }
}

/// Whether `node`'s top stroke is a visible linear or radial gradient laid
/// along or across it.
pub fn needs_stroke_uv(node: &moda_scene::types::RawNode) -> bool {
    crate::paint::has_visible_gradient_stroke(node)
        && crate::paint::gradient::top_stroke(node)
            .is_some_and(|s| s.gradient_mode() != moda_scene::paint::StrokeGradientMode::Within)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(stroke: &[[f32; 2]]) -> TessellatedGeometry {
        TessellatedGeometry {
            positions: stroke.iter().flat_map(|p| [p[0], p[1]]).collect(),
            vertex_types: vec![1.0; stroke.len()],
            vertex_count: stroke.len() as u32,
            ..Default::default()
        }
    }

    #[test]
    fn along_and_across_on_a_straight_band() {
        let line = centerline(
            &[
                PathCmd::Move { x: 0.0, y: 0.0 },
                PathCmd::Line { x: 100.0, y: 0.0 },
            ],
            false,
            CENTERLINE_TOLERANCE,
        );
        // A 10-wide band as two triangles: left edge y = -5, right edge y = +5 (y down).
        let g = geometry(&[
            [0.0, -5.0],
            [100.0, -5.0],
            [0.0, 5.0],
            [0.0, 5.0],
            [100.0, -5.0],
            [100.0, 5.0],
        ]);
        let uv = stroke_uv(&g, &line, true).unwrap().stroke_uv;
        let pairs: Vec<(f32, f32)> = uv.chunks(2).map(|c| (c[0], c[1])).collect();
        assert_eq!(
            pairs,
            vec![
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (1.0, 0.0),
                (1.0, 1.0)
            ]
        );
    }

    #[test]
    fn a_cap_past_the_end_measures_across_and_stays_at_the_end() {
        let line = centerline(
            &[
                PathCmd::Move { x: 0.0, y: 0.0 },
                PathCmd::Line { x: 100.0, y: 0.0 },
            ],
            false,
            CENTERLINE_TOLERANCE,
        );
        let g = geometry(&[
            [100.0, -5.0],
            [105.0, -5.0],
            [105.0, 5.0],
            [100.0, 5.0],
            [0.0, 0.0],
            [50.0, 0.0],
        ]);
        let uv = stroke_uv(&g, &line, true).unwrap().stroke_uv;
        assert_eq!((uv[2], uv[3]), (1.0, 0.0), "square cap corner, left side");
        assert_eq!((uv[4], uv[5]), (1.0, 1.0), "square cap corner, right side");
        assert_eq!((uv[10], uv[11]), (0.5, 0.5), "on the centreline, halfway");
    }

    #[test]
    fn a_closed_path_wraps_at_its_seam() {
        // A 100-square, closed; one triangle straddles the start (0, 0).
        let cmds = [
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Line { x: 100.0, y: 0.0 },
            PathCmd::Line { x: 100.0, y: 100.0 },
            PathCmd::Line { x: 0.0, y: 100.0 },
        ];
        let line = centerline(&cmds, true, CENTERLINE_TOLERANCE);
        let g = geometry(&[[2.0, -2.0], [-2.0, 2.0], [-2.0, 10.0]]);
        let uv = stroke_uv(&g, &line, true).unwrap().stroke_uv;
        // (2, -2) is 2 px along (0.005); (-2, 10) sits on the last side, 10 px before the end (0.975): lifted to 1.005.
        assert!((uv[0] - 1.005).abs() < 1e-4, "{uv:?}");
        assert!(uv[4] > 0.9 && uv[4] < 1.0, "{uv:?}");
    }

    #[test]
    fn an_inside_stroke_spans_its_one_side() {
        let line = centerline(
            &[
                PathCmd::Move { x: 0.0, y: 0.0 },
                PathCmd::Line { x: 100.0, y: 0.0 },
            ],
            false,
            CENTERLINE_TOLERANCE,
        );
        // Offsets 0 (on the path) and +8 (right) at both ends: across runs 0 → 1 over them. A vertex halfway,
        // alone at its arc length, measures against its neighbours' extent.
        let g = geometry(&[
            [0.0, 0.0],
            [0.0, 8.0],
            [100.0, 0.0],
            [100.0, 8.0],
            [50.0, 4.0],
            [50.0, 2.0],
        ]);
        let uv = stroke_uv(&g, &line, true).unwrap().stroke_uv;
        assert_eq!((uv[1], uv[3]), (0.0, 1.0));
        assert!((uv[9] - 0.5).abs() < 1e-6, "{uv:?}");
        assert!((uv[11] - 0.25).abs() < 1e-6, "{uv:?}");
    }
}
