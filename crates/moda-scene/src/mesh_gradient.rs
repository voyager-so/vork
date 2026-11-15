//! Gradient mesh (Illustrator's Object ▸ Create Gradient Mesh; audit §11): a grid of colour points over a shape,
//! each cell a Coons patch, the colour interpolated across it. The model and its maths; the TS twin is
//! `packages/editor/src/core/scene/meshGradient.ts` (the PDF export evaluates the same patches).
//!
//! - **Points.** `(rows + 1) × (columns + 1)` points, row by row from the top left, each at `x` / `y` % of the paint
//!   box (like the freeform gradient's) with a colour. A point may carry Bézier handles toward its left, right, up
//!   and down neighbours (offsets in the same % space). An absent handle is a third of the way to that neighbour
//!   (a straight edge), or on a `smooth` mesh the Catmull-Rom tangent through the point's neighbours along the line
//!   (straight on a regular grid; moved points then bend the lines smoothly instead of kinking them).
//! - **Geometry.** Cell `(r, c)` is the Coons patch bounded by the four edge curves between its corner points: `u`
//!   runs along the columns (left → right), `v` along the rows (top → bottom).
//! - **Colour.** `bilinear` (the default): the four corner colours blended bilinearly in `(u, v)`, straight RGBA —
//!   Illustrator's and PDF's patch model (type 6 shading), so the PDF export is exact. `smooth: true`: bicubic
//!   Hermite in `(u, v)` with each point's colour slopes estimated from its neighbours (Catmull-Rom; one-sided at the
//!   border; no twist), so colour flows across cell edges without creases; clamped to [0, 1]. With one cell, smooth
//!   and bilinear agree.
//! - **Drawing.** [`bake`] rasterizes the patches, in order (a later cell covers an earlier one where they fold),
//!   into a premultiplied RGBA8 image over the mesh's control bounds, each patch diced so no cell spans more than
//!   [`DICE_TEXELS`] texels and every triangle Gouraud-shaded; the engine draws the image as an image fill.

use serde::{Deserialize, Serialize};
use crate::value::Value;

use crate::paint::PaintPoint;

/// Most cells along either axis (Illustrator's dialog allows 50; this keeps the point list under ~2,600).
pub const MAX_MESH_CELLS: u32 = 50;

/// Longest a diced patch triangle may be, in texels: small enough that Gouraud shading between the dice matches the
/// patch's own interpolation to well under an 8-bit step.
pub const DICE_TEXELS: f64 = 2.0;

/// A point's Bézier handles toward its neighbours: offsets from the point, in the paint box's % space.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct MeshHandles {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub left: Option<PaintPoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub right: Option<PaintPoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub up: Option<PaintPoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub down: Option<PaintPoint>,
}

/// One mesh point: where it sits (% of the paint box) and its colour (a CSS string or a variable reference).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshGradientPoint {
    pub x: f64,
    pub y: f64,
    pub color: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handles: Option<MeshHandles>,
}

/// Illustrator's gradient mesh as a paint (fills only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshGradientPaint {
    /// Cells down (≥ 1).
    pub rows: u32,
    /// Cells across (≥ 1).
    pub columns: u32,
    /// `(rows + 1) × (columns + 1)` points, row by row from the top left.
    pub points: Vec<MeshGradientPoint>,
    /// Bicubic colour across cells instead of bilinear per cell. Absent = bilinear.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smooth: Option<bool>,
}

/// A 2D point in node-local px.
pub type P2 = [f64; 2];
/// Straight RGBA, 0..1.
pub type Rgba = [f64; 4];

/// A mesh placed over a box: positions in node-local px, parsed colours, every edge as a cubic.
#[derive(Debug, Clone)]
pub struct PlacedMesh {
    pub rows: usize,
    pub columns: usize,
    pub smooth: bool,
    /// `(rows + 1) × (columns + 1)` corner positions.
    pub corners: Vec<P2>,
    pub colors: Vec<Rgba>,
    /// Horizontal edges: `(rows + 1) × columns` cubics, row by row, left → right.
    pub h_edges: Vec<[P2; 4]>,
    /// Vertical edges: `rows × (columns + 1)` cubics, row by row, top → bottom.
    pub v_edges: Vec<[P2; 4]>,
    /// Colour slopes per point along the columns (`u`) and rows (`v`) for `smooth`.
    pub du: Vec<Rgba>,
    pub dv: Vec<Rgba>,
}

impl MeshGradientPaint {
    /// Whether the point list matches the grid (and the grid is within bounds): an invalid mesh draws nothing.
    pub fn is_valid(&self) -> bool {
        (1..=MAX_MESH_CELLS).contains(&self.rows)
            && (1..=MAX_MESH_CELLS).contains(&self.columns)
            && self.points.len() == ((self.rows + 1) * (self.columns + 1)) as usize
    }

    /// Place the mesh over `paint_box` `[x, y, w, h]` (node-local px), parsing colours with `parse` (unparseable:
    /// white, as a gradient stop). `None` when the mesh is invalid.
    pub fn place(&self, paint_box: [f64; 4], parse: impl Fn(&Value) -> Option<Rgba>) -> Option<PlacedMesh> {
        if !self.is_valid() {
            return None;
        }
        let (rows, columns) = (self.rows as usize, self.columns as usize);
        let [bx, by, bw, bh] = paint_box;
        let fin = |v: f64, d: f64| if v.is_finite() { v } else { d };
        let at = |p: &MeshGradientPoint| [bx + fin(p.x, 0.0) / 100.0 * bw, by + fin(p.y, 0.0) / 100.0 * bh];
        let off = |h: Option<PaintPoint>| h.map(|h| [fin(h.x, 0.0) / 100.0 * bw, fin(h.y, 0.0) / 100.0 * bh]);
        let corners: Vec<P2> = self.points.iter().map(at).collect();
        let colors: Vec<Rgba> = self
            .points
            .iter()
            .map(|p| parse(&p.color).unwrap_or([1.0, 1.0, 1.0, 1.0]).map(|c| c.clamp(0.0, 1.0)))
            .collect();
        let idx = |r: usize, c: usize| r * (columns + 1) + c;
        let handle = |i: usize, pick: fn(&MeshHandles) -> Option<PaintPoint>| {
            off(self.points[i].handles.as_ref().and_then(pick))
        };
        let edge = |a: usize, b: usize, ha: Option<P2>, hb: Option<P2>| {
            let (p0, p3) = (corners[a], corners[b]);
            let third = |from: P2, to: P2| [from[0] + (to[0] - from[0]) / 3.0, from[1] + (to[1] - from[1]) / 3.0];
            let p1 = ha.map(|h| [p0[0] + h[0], p0[1] + h[1]]).unwrap_or_else(|| third(p0, p3));
            let p2 = hb.map(|h| [p3[0] + h[0], p3[1] + h[1]]).unwrap_or_else(|| third(p3, p0));
            [p0, p1, p2, p3]
        };
        let smooth = self.smooth == Some(true);
        // A smooth mesh's absent handle is the Catmull-Rom tangent through the point's neighbours along that line
        // (a sixth of prev → next; a third of the edge at the border), so moved points bend the lines smoothly; on a
        // regular grid that is the straight edge. A bilinear mesh's is always the straight third.
        let tangent = |prev: Option<usize>, here: usize, next: usize| -> Option<P2> {
            if !smooth {
                return None;
            }
            let (p, n) = (corners[prev.unwrap_or(here)], corners[next]);
            let k = if prev.is_some() { 6.0 } else { 3.0 };
            Some([(n[0] - p[0]) / k, (n[1] - p[1]) / k])
        };
        let mut h_edges = Vec::with_capacity((rows + 1) * columns);
        for r in 0..=rows {
            for c in 0..columns {
                let (a, b) = (idx(r, c), idx(r, c + 1));
                let ha = handle(a, |h| h.right).or_else(|| tangent((c > 0).then(|| idx(r, c - 1)), a, b));
                let hb = handle(b, |h| h.left).or_else(|| tangent((c + 1 < columns).then(|| idx(r, c + 2)), b, a));
                h_edges.push(edge(a, b, ha, hb));
            }
        }
        let mut v_edges = Vec::with_capacity(rows * (columns + 1));
        for r in 0..rows {
            for c in 0..=columns {
                let (a, b) = (idx(r, c), idx(r + 1, c));
                let ha = handle(a, |h| h.down).or_else(|| tangent((r > 0).then(|| idx(r - 1, c)), a, b));
                let hb = handle(b, |h| h.up).or_else(|| tangent((r + 1 < rows).then(|| idx(r + 2, c)), b, a));
                v_edges.push(edge(a, b, ha, hb));
            }
        }
        let slope = |prev: Option<usize>, here: usize, next: Option<usize>| -> Rgba {
            let c = |i: usize| colors[i];
            match (prev, next) {
                (Some(p), Some(n)) => std::array::from_fn(|k| (c(n)[k] - c(p)[k]) / 2.0),
                (None, Some(n)) => std::array::from_fn(|k| c(n)[k] - c(here)[k]),
                (Some(p), None) => std::array::from_fn(|k| c(here)[k] - c(p)[k]),
                (None, None) => [0.0; 4],
            }
        };
        let (mut du, mut dv) = (Vec::new(), Vec::new());
        if smooth {
            for r in 0..=rows {
                for c in 0..=columns {
                    let i = idx(r, c);
                    du.push(slope((c > 0).then(|| idx(r, c - 1)), i, (c < columns).then(|| idx(r, c + 1))));
                    dv.push(slope((r > 0).then(|| idx(r - 1, c)), i, (r < rows).then(|| idx(r + 1, c))));
                }
            }
        }
        Some(PlacedMesh { rows, columns, smooth, corners, colors, h_edges, v_edges, du, dv })
    }
}

fn cubic(e: &[P2; 4], t: f64) -> P2 {
    let s = 1.0 - t;
    let (a, b, c, d) = (s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t);
    [
        a * e[0][0] + b * e[1][0] + c * e[2][0] + d * e[3][0],
        a * e[0][1] + b * e[1][1] + c * e[2][1] + d * e[3][1],
    ]
}

/// Hermite basis: values at 0 and 1, then slopes at 0 and 1.
fn hermite(t: f64) -> [f64; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [2.0 * t3 - 3.0 * t2 + 1.0, -2.0 * t3 + 3.0 * t2, t3 - 2.0 * t2 + t, t3 - t2]
}

impl PlacedMesh {
    fn idx(&self, r: usize, c: usize) -> usize {
        r * (self.columns + 1) + c
    }

    /// The four edges of cell `(r, c)`: top and bottom (left → right), left and right (top → bottom).
    pub fn cell_edges(&self, r: usize, c: usize) -> [[P2; 4]; 4] {
        let h = |rr: usize| self.h_edges[rr * self.columns + c];
        let v = |cc: usize| self.v_edges[r * (self.columns + 1) + cc];
        [h(r), h(r + 1), v(c), v(c + 1)]
    }

    /// Where cell `(r, c)` puts `(u, v)`: the Coons patch of its edges.
    pub fn position(&self, r: usize, c: usize, u: f64, v: f64) -> P2 {
        let [top, bottom, left, right] = self.cell_edges(r, c);
        let (t, b, l, rt) = (cubic(&top, u), cubic(&bottom, u), cubic(&left, v), cubic(&right, v));
        let (p00, p01, p10, p11) = (top[0], top[3], bottom[0], bottom[3]);
        std::array::from_fn(|k| {
            (1.0 - v) * t[k] + v * b[k] + (1.0 - u) * l[k] + u * rt[k]
                - ((1.0 - u) * (1.0 - v) * p00[k] + u * (1.0 - v) * p01[k] + (1.0 - u) * v * p10[k] + u * v * p11[k])
        })
    }

    /// The colour of cell `(r, c)` at `(u, v)` (straight RGBA).
    pub fn color(&self, r: usize, c: usize, u: f64, v: f64) -> Rgba {
        let i = [self.idx(r, c), self.idx(r, c + 1), self.idx(r + 1, c), self.idx(r + 1, c + 1)];
        let col = |j: usize| self.colors[i[j]];
        if !self.smooth {
            return std::array::from_fn(|k| {
                (1.0 - u) * (1.0 - v) * col(0)[k] + u * (1.0 - v) * col(1)[k] + (1.0 - u) * v * col(2)[k] + u * v * col(3)[k]
            });
        }
        let (hu, hv) = (hermite(u), hermite(v));
        // Corners (a, b) = (column side, row side): 0 = left / top, 1 = right / bottom.
        std::array::from_fn(|k| {
            let mut sum = 0.0;
            for (j, (a, b)) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                let p = i[j];
                sum += hu[a] * hv[b] * self.colors[p][k] + hu[2 + a] * hv[b] * self.du[p][k] + hu[a] * hv[2 + b] * self.dv[p][k];
            }
            sum.clamp(0.0, 1.0)
        })
    }

    /// The control bounds `[x0, y0, x1, y1]` (every edge's control points): the patches lie inside.
    pub fn bounds(&self) -> [f64; 4] {
        let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for e in self.h_edges.iter().chain(&self.v_edges) {
            for p in e {
                b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])];
            }
        }
        b
    }
}

/// A baked mesh: premultiplied RGBA8 over `origin` .. `origin + size` (node-local px), `width` × `height` texels.
#[derive(Debug, Clone)]
pub struct BakedMesh {
    pub width: u32,
    pub height: u32,
    pub rgba_premultiplied: Vec<u8>,
    pub origin: P2,
    pub size: P2,
}

fn control_length(e: &[P2; 4], sx: f64, sy: f64) -> f64 {
    (0..3).map(|i| ((e[i + 1][0] - e[i][0]) * sx).hypot((e[i + 1][1] - e[i][1]) * sy)).sum()
}

/// Rasterize `mesh` at `texels_per_px` (each axis capped so it fits `max_dim` texels), cells in order.
pub fn bake(mesh: &PlacedMesh, texels_per_px: f64, max_dim: u32) -> Option<BakedMesh> {
    let [x0, y0, x1, y1] = mesh.bounds();
    let (w, h) = (x1 - x0, y1 - y0);
    if !(w.is_finite() && h.is_finite()) || w <= 0.0 || h <= 0.0 {
        return None;
    }
    let dim = |extent: f64| ((extent * texels_per_px).ceil() as u32).clamp(2, max_dim.max(2));
    let (tw, th) = (dim(w), dim(h));
    let (sx, sy) = (tw as f64 / w, th as f64 / h);
    let mut out = vec![0u8; (tw * th * 4) as usize];
    // Straight RGBA at a texel-space vertex.
    type V = ([f64; 2], Rgba);
    let mut tri = |a: &V, b: &V, c: &V| {
        let area = (b.0[0] - a.0[0]) * (c.0[1] - a.0[1]) - (b.0[1] - a.0[1]) * (c.0[0] - a.0[0]);
        if area.abs() < 1e-12 {
            return;
        }
        let minx = a.0[0].min(b.0[0]).min(c.0[0]).floor().max(0.0) as i64;
        let maxx = (a.0[0].max(b.0[0]).max(c.0[0]).ceil() as i64).min(tw as i64 - 1);
        let miny = a.0[1].min(b.0[1]).min(c.0[1]).floor().max(0.0) as i64;
        let maxy = (a.0[1].max(b.0[1]).max(c.0[1]).ceil() as i64).min(th as i64 - 1);
        for ty in miny..=maxy {
            for tx in minx..=maxx {
                let p = [tx as f64 + 0.5, ty as f64 + 0.5];
                let wa = ((b.0[0] - p[0]) * (c.0[1] - p[1]) - (b.0[1] - p[1]) * (c.0[0] - p[0])) / area;
                let wb = ((c.0[0] - p[0]) * (a.0[1] - p[1]) - (c.0[1] - p[1]) * (a.0[0] - p[0])) / area;
                let wc = 1.0 - wa - wb;
                // Texel centres on a shared edge land in one of its triangles (a small tolerance closes seams).
                if wa < -1e-9 || wb < -1e-9 || wc < -1e-9 {
                    continue;
                }
                let col: Rgba = std::array::from_fn(|k| (wa * a.1[k] + wb * b.1[k] + wc * c.1[k]).clamp(0.0, 1.0));
                let i = ((ty as u32 * tw + tx as u32) * 4) as usize;
                let alpha = col[3];
                out[i] = (col[0] * alpha * 255.0).round() as u8;
                out[i + 1] = (col[1] * alpha * 255.0).round() as u8;
                out[i + 2] = (col[2] * alpha * 255.0).round() as u8;
                out[i + 3] = (alpha * 255.0).round() as u8;
            }
        }
    };
    for r in 0..mesh.rows {
        for c in 0..mesh.columns {
            let [top, bottom, left, right] = mesh.cell_edges(r, c);
            let dice = |len: f64| ((len / DICE_TEXELS).ceil() as usize).clamp(1, 512);
            let nu = dice(control_length(&top, sx, sy).max(control_length(&bottom, sx, sy)));
            let nv = dice(control_length(&left, sx, sy).max(control_length(&right, sx, sy)));
            let mut grid: Vec<V> = Vec::with_capacity((nu + 1) * (nv + 1));
            for j in 0..=nv {
                let v = j as f64 / nv as f64;
                for i in 0..=nu {
                    let u = i as f64 / nu as f64;
                    let p = mesh.position(r, c, u, v);
                    grid.push(([(p[0] - x0) * sx, (p[1] - y0) * sy], mesh.color(r, c, u, v)));
                }
            }
            for j in 0..nv {
                for i in 0..nu {
                    let k = j * (nu + 1) + i;
                    let (a, b, cc, d) = (&grid[k], &grid[k + 1], &grid[k + nu + 1], &grid[k + nu + 2]);
                    tri(a, b, cc);
                    tri(b, d, cc);
                }
            }
        }
    }
    Some(BakedMesh { width: tw, height: th, rgba_premultiplied: out, origin: [x0, y0], size: [w, h] })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    fn parse(v: &Value) -> Option<Rgba> {
        let s = v.as_str()?;
        let h = s.strip_prefix('#')?;
        let c = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).ok().map(|b| b as f64 / 255.0);
        Some([c(0)?, c(2)?, c(4)?, 1.0])
    }

    fn grid(rows: u32, columns: u32, colors: &[&str], smooth: bool) -> MeshGradientPaint {
        let mut points = Vec::new();
        for r in 0..=rows {
            for c in 0..=columns {
                points.push(MeshGradientPoint {
                    x: c as f64 * 100.0 / columns as f64,
                    y: r as f64 * 100.0 / rows as f64,
                    color: json!(colors[(r * (columns + 1) + c) as usize]),
                    handles: None,
                });
            }
        }
        MeshGradientPaint { rows, columns, points, smooth: smooth.then_some(true) }
    }

    #[test]
    fn straight_edges_make_a_bilinear_patch_and_corners_are_exact() {
        let m = grid(1, 1, &["#ff0000", "#00ff00", "#0000ff", "#ffffff"], false).place([10.0, 20.0, 100.0, 50.0], parse).unwrap();
        assert_eq!(m.position(0, 0, 0.0, 0.0), [10.0, 20.0]);
        assert_eq!(m.position(0, 0, 1.0, 1.0), [110.0, 70.0]);
        let p = m.position(0, 0, 0.25, 0.5);
        assert!((p[0] - 35.0).abs() < 1e-9 && (p[1] - 45.0).abs() < 1e-9, "{p:?}");
        let c = m.color(0, 0, 0.5, 0.5);
        assert!((c[0] - 0.5).abs() < 1e-9 && (c[1] - 0.5).abs() < 1e-9 && (c[2] - 0.5).abs() < 1e-9);
        assert_eq!(m.color(0, 0, 1.0, 0.0), [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn handles_bend_the_edges() {
        let mut paint = grid(1, 1, &["#000000", "#000000", "#000000", "#000000"], false);
        // The top-left point's right handle pulls the top edge up by 30 % of the box height.
        paint.points[0].handles = Some(MeshHandles { right: Some(PaintPoint { x: 33.0, y: -30.0 }), ..Default::default() });
        let m = paint.place([0.0, 0.0, 100.0, 100.0], parse).unwrap();
        assert!(m.position(0, 0, 0.3, 0.0)[1] < -5.0);
        assert!(m.bounds()[1] <= -30.0);
    }

    #[test]
    fn smooth_reproduces_a_linear_ramp_and_is_continuous_in_slope_across_cells() {
        // A ramp sampled on a 1 × 3 grid: smooth interpolation reproduces it exactly (Catmull-Rom is exact on lines).
        let cols = ["#000000", "#555555", "#aaaaaa", "#ffffff"];
        let colors: Vec<&str> = cols.iter().chain(cols.iter()).copied().collect();
        let m = grid(1, 3, &colors, true).place([0.0, 0.0, 300.0, 10.0], parse).unwrap();
        for (c, u) in [(0, 0.3), (1, 0.5), (2, 0.9)] {
            let want = (c as f64 + u) / 3.0;
            assert!((m.color(0, c, u, 0.5)[0] - want).abs() < 1e-3, "cell {c} u {u}");
        }
        // A bump: bilinear creases at the peak, smooth has matching slopes left and right of it.
        let bump = ["#000000", "#ffffff", "#000000", "#000000", "#ffffff", "#000000"];
        let s = grid(1, 2, &bump, true).place([0.0, 0.0, 200.0, 10.0], parse).unwrap();
        let slope = |c: usize, u0: f64, u1: f64| (s.color(0, c, u1, 0.5)[0] - s.color(0, c, u0, 0.5)[0]) / (u1 - u0);
        assert!((slope(0, 0.999, 1.0) - slope(1, 0.0, 0.001)).abs() < 0.05);
    }

    #[test]
    fn a_smooth_mesh_bends_its_lines_smoothly_through_a_moved_point() {
        let mut paint = grid(2, 2, &["#000000"; 9], true);
        // On the regular grid the smooth defaults are the straight edges.
        let regular = paint.place([0.0, 0.0, 100.0, 100.0], parse).unwrap();
        assert_eq!(regular.h_edges[2][1], [100.0 / 6.0, 50.0]);
        // Move the middle point: the row's two edges meet it with one tangent (no kink), unlike bilinear's.
        paint.points[4].x = 30.0;
        paint.points[4].y = 40.0;
        let dir = |m: &PlacedMesh| {
            let (l, r) = (m.h_edges[2], m.h_edges[3]);
            let a = [l[3][0] - l[2][0], l[3][1] - l[2][1]];
            let b = [r[1][0] - r[0][0], r[1][1] - r[0][1]];
            (a[0] * b[1] - a[1] * b[0]) / (a[0].hypot(a[1]) * b[0].hypot(b[1]))
        };
        assert!(dir(&paint.place([0.0, 0.0, 100.0, 100.0], parse).unwrap()).abs() < 1e-9);
        paint.smooth = None;
        assert!(dir(&paint.place([0.0, 0.0, 100.0, 100.0], parse).unwrap()).abs() > 0.1);
    }

    #[test]
    fn invalid_meshes_place_nothing() {
        let mut m = grid(2, 2, &["#000000"; 9], false);
        m.points.pop();
        assert!(m.place([0.0, 0.0, 10.0, 10.0], parse).is_none());
        let zero = MeshGradientPaint { rows: 0, columns: 1, points: vec![], smooth: None };
        assert!(zero.place([0.0, 0.0, 10.0, 10.0], parse).is_none());
    }

    #[test]
    fn bake_covers_the_mesh_with_its_colours() {
        let m = grid(2, 2, &["#ff0000", "#ff0000", "#ff0000", "#0000ff", "#ffffff", "#0000ff", "#00ff00", "#00ff00", "#00ff00"], false)
            .place([0.0, 0.0, 64.0, 64.0], parse)
            .unwrap();
        let b = bake(&m, 2.0, 4096).unwrap();
        assert_eq!((b.width, b.height), (128, 128));
        // Every texel is covered (opaque), and the centre is the white point's colour.
        assert!(b.rgba_premultiplied.chunks(4).all(|p| p[3] == 255));
        let at = |x: u32, y: u32| {
            let i = ((y * b.width + x) * 4) as usize;
            [b.rgba_premultiplied[i], b.rgba_premultiplied[i + 1], b.rgba_premultiplied[i + 2]]
        };
        let centre = at(64, 64);
        assert!(centre.iter().all(|&v| v >= 245), "{centre:?}");
        // Texels match the patch colour at their centre to within one 8-bit step (Gouraud over 2-texel dice).
        for (x, y) in [(10u32, 10u32), (100, 40), (30, 90), (120, 120)] {
            let (px, py) = ((x as f64 + 0.5) / 2.0, (y as f64 + 0.5) / 2.0);
            let (c, u) = (((px / 32.0) as usize).min(1), (px / 32.0).fract());
            let (r, v) = (((py / 32.0) as usize).min(1), (py / 32.0).fract());
            let want = m.color(r, c, u, v);
            let got = at(x, y);
            for k in 0..3 {
                assert!((got[k] as f64 - want[k] * 255.0).abs() <= 1.0, "({x},{y}) {got:?} vs {want:?}");
            }
        }
    }
}
