//! Faces of a planar network: a DCEL (half-edge) face walk on cubic segments.
//!
//! Segment `i` contributes half-edges `2i` (start → end) and `2i + 1`
//! (end → start). Around each vertex the outgoing half-edges are sorted
//! counter-clockwise by where they cross a small circle round the vertex (a
//! few tolerances out, not the raw derivative, which a near-zero handle makes
//! meaningless), and edges level there by a larger circle; that is the true
//! planar order when the graph is planar. The
//! successor of a half-edge `u → v` is the outgoing half-edge at `v` just
//! clockwise of `v → u`, so every walk keeps its face on the left: bounded
//! faces come out with positive signed area (in the frame where `atan2` angles
//! grow counter-clockwise) and the outer boundary of each connected component
//! comes out non-positive.
//!
//! A component's outer boundary is a hole in the smallest face of another
//! component that contains it. Dangling edges (spokes, bridges) are walked in
//! both directions within one face; they add no area and are dropped from the
//! face's loops.

use super::geom::{winding_number, Cubic, Rect, Vector};
use super::{OrientedLoop, VectorNetwork, VectorRegion, WindingRule};

/// A bounded face of the planar graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Face {
    /// Boundary loops as `(segment, reversed)` in walk order. The outer
    /// boundary's loops wind counter-clockwise (positive area) and holes wind
    /// the other way, so the face fills correctly under either winding rule.
    pub loops: Vec<OrientedLoop>,
    /// Area of the face (outer area minus holes), always positive.
    pub area: f64,
}

impl Face {
    /// The loops as bare segment indices: a Figma region's `loops`.
    pub fn segment_loops(&self) -> Vec<Vec<usize>> {
        self.loops
            .iter()
            .map(|lp| lp.iter().map(|&(s, _)| s).collect())
            .collect()
    }

    /// The face as a region. A region loop stores bare segment indices, so its
    /// direction is re-derived by chaining and a hole's opposite winding is
    /// lost; a face with holes is therefore stored even-odd, which fills
    /// outer-minus-holes whatever direction each loop chains in.
    pub fn region(&self) -> VectorRegion {
        let rule = if self.loops.len() > 1 {
            WindingRule::EvenOdd
        } else {
            WindingRule::NonZero
        };
        VectorRegion::new(rule, self.segment_loops())
    }

    fn polygons(&self, net: &VectorNetwork, tol: f64) -> Vec<Vec<Vector>> {
        self.loops
            .iter()
            .map(|lp| flatten_loop(net, lp, tol))
            .collect()
    }

    /// Is `p` inside the face (not in a hole, not outside)?
    pub fn contains(&self, net: &VectorNetwork, p: Vector, tol: f64) -> bool {
        let w: i32 = self
            .polygons(net, tol * 0.25)
            .iter()
            .map(|poly| winding_number(poly, p))
            .sum();
        w != 0
    }

    /// A point well inside the face (the midpoint of the widest interior span
    /// over a few horizontal scanlines).
    pub fn interior_point(&self, net: &VectorNetwork, tol: f64) -> Option<Vector> {
        interior_point(&self.polygons(net, tol * 0.25), |w| w != 0)
    }
}

/// Outcome of [`VectorNetwork::toggle_fill_at`].
#[derive(Debug, Clone, PartialEq)]
pub enum FillToggle {
    /// The face was empty and is now region `region`.
    Filled { region: usize },
    /// The face was filled and no longer is.
    Unfilled,
    /// The point is not inside any closed area.
    NoFace,
}

pub(crate) fn flatten_loop(net: &VectorNetwork, lp: &[(usize, bool)], tol: f64) -> Vec<Vector> {
    let mut poly = Vec::new();
    for (k, &(s, rev)) in lp.iter().enumerate() {
        let c = net.oriented_cubic(s, rev);
        if k == 0 {
            poly.push(c.p0);
        }
        c.flatten_into(tol, &mut poly);
    }
    if poly.len() > 1 && poly.first() == poly.last() {
        poly.pop();
    }
    poly
}

/// Midpoint of the widest span inside `polys` (by `inside(winding)`) across a
/// handful of scanlines.
pub(crate) fn interior_point(
    polys: &[Vec<Vector>],
    inside: impl Fn(i32) -> bool,
) -> Option<Vector> {
    let mut bounds: Option<Rect> = None;
    for poly in polys {
        if poly.is_empty() {
            continue;
        }
        let r = Rect::from_points(poly);
        bounds = Some(bounds.map_or(r, |b| b.union(r)));
    }
    let b = bounds?;
    let h = b.max.y - b.min.y;
    if h <= 0.0 {
        return None;
    }
    let mut best: Option<(f64, Vector)> = None;
    const LINES: usize = 11;
    for k in 0..LINES {
        // An irrational-ish offset keeps scanlines off vertices.
        let y = b.min.y + h * ((k as f64 + 0.5 + 0.0137) / LINES as f64);
        let mut xs: Vec<f64> = Vec::new();
        for poly in polys {
            let n = poly.len();
            for i in 0..n {
                let (a, c) = (poly[i], poly[(i + 1) % n]);
                if (a.y <= y) != (c.y <= y) {
                    xs.push(a.x + (y - a.y) / (c.y - a.y) * (c.x - a.x));
                }
            }
        }
        xs.sort_by(f64::total_cmp);
        for w in xs.windows(2) {
            let width = w[1] - w[0];
            if width <= 0.0 {
                continue;
            }
            let p = Vector::new(0.5 * (w[0] + w[1]), y);
            let wind: i32 = polys.iter().map(|poly| winding_number(poly, p)).sum();
            if inside(wind) && best.is_none_or(|(bw, _)| width > bw) {
                best = Some((width, p));
            }
        }
    }
    best.map(|(_, p)| p)
}

/// One walked cycle of half-edges.
struct Cycle {
    half_edges: Vec<usize>,
    area: f64,
}

impl VectorNetwork {
    /// Every bounded face of the network, which must be planar (call
    /// [`planarize`](Self::planarize) first; a self-loop segment, which
    /// planarising splits, is ignored here).
    pub fn faces(&self, tol: f64) -> Vec<Face> {
        let tol = if tol.is_finite() && tol > 0.0 {
            tol
        } else {
            super::DEFAULT_TOLERANCE
        };
        let n_he = self.segments.len() * 2;
        if n_he == 0 {
            return Vec::new();
        }
        let usable: Vec<bool> = self.segments.iter().map(|s| s.start != s.end).collect();
        let origin = |h: usize| {
            let s = &self.segments[h / 2];
            if h.is_multiple_of(2) {
                s.start
            } else {
                s.end
            }
        };
        let he_cubic = |h: usize| self.oriented_cubic(h / 2, h % 2 == 1);
        // Area is measured about the centre so large coordinates do not cost
        // precision.
        let centre = self.bounds().map(|b| b.center()).unwrap_or_default();

        // Rotation system: outgoing half-edges per vertex, counter-clockwise.
        let mut outgoing: Vec<Vec<usize>> = vec![Vec::new(); self.vertices.len()];
        for h in 0..n_he {
            if usable[h / 2] {
                outgoing[origin(h)].push(h);
            }
        }
        let mut pos_in_vertex = vec![usize::MAX; n_he];
        for list in outgoing.iter_mut() {
            if list.is_empty() {
                continue;
            }
            let cubics: Vec<Cubic> = list.iter().map(|&h| he_cubic(h)).collect();
            // Directions are read where each edge crosses a circle of radius
            // `r`, not from the derivative: planarising ignores crossings
            // within `tol` of a vertex, so a curve with a vanishing tangent
            // handle may "leave" in one direction and turn within `tol`; the
            // order at a few `tol` out is the one the rest of the graph obeys.
            // Edges still level there (leaving tangent to each other) are
            // ordered further out, where they have diverged.
            let min_chord = cubics
                .iter()
                .map(|c| c.p0.dist(c.p3))
                .fold(f64::INFINITY, f64::min);
            let r = (min_chord * 0.05).max(tol * 4.0).min(min_chord * 0.5);
            let keys: Vec<(f64, f64)> = cubics
                .iter()
                .map(|c| {
                    let near = (c.point_at_radius(r) - c.p0).angle();
                    let far = (c.point_at_radius(r * 4.0) - c.p0).angle();
                    (near, wrap_angle(far - near))
                })
                .collect();
            let mut idx: Vec<usize> = (0..list.len()).collect();
            idx.sort_by(|&x, &y| {
                let (ax, dx) = keys[x];
                let (ay, dy) = keys[y];
                if (ax - ay).abs() > 1e-9 {
                    ax.total_cmp(&ay)
                } else {
                    dx.total_cmp(&dy).then(x.cmp(&y))
                }
            });
            let sorted: Vec<usize> = idx.iter().map(|&k| list[k]).collect();
            *list = sorted;
            for (p, &h) in list.iter().enumerate() {
                pos_in_vertex[h] = p;
            }
        }
        let next = |h: usize| -> usize {
            let twin = h ^ 1;
            let v = origin(twin);
            let list = &outgoing[v];
            let p = pos_in_vertex[twin];
            list[(p + list.len() - 1) % list.len()]
        };

        // Walk every cycle.
        let mut seen = vec![false; n_he];
        let mut cycles: Vec<Cycle> = Vec::new();
        for h0 in 0..n_he {
            if seen[h0] || !usable[h0 / 2] {
                continue;
            }
            let mut hs = Vec::new();
            let mut h = h0;
            let mut area = 0.0;
            loop {
                seen[h] = true;
                hs.push(h);
                let c = he_cubic(h);
                let shifted =
                    Cubic::new(c.p0 - centre, c.p1 - centre, c.p2 - centre, c.p3 - centre);
                area += shifted.signed_area();
                h = next(h);
                if h == h0 || seen[h] || hs.len() > n_he {
                    break;
                }
            }
            cycles.push(Cycle {
                half_edges: hs,
                area,
            });
        }

        // Connected components (by vertex) to tell a component's own faces
        // from the faces that could contain it.
        let mut parent: Vec<usize> = (0..self.vertices.len()).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        for s in &self.segments {
            let (a, b) = (find(&mut parent, s.start), find(&mut parent, s.end));
            if a != b {
                parent[a] = b;
            }
        }
        let component = |c: &Cycle, parent: &mut Vec<usize>| find(parent, origin(c.half_edges[0]));

        let area_eps = tol * tol;
        let bounded: Vec<usize> = (0..cycles.len())
            .filter(|&i| cycles[i].area > area_eps)
            .collect();
        let bounded_polys: Vec<Vec<Vector>> = bounded
            .iter()
            .map(|&i| {
                let lp: OrientedLoop = cycles[i]
                    .half_edges
                    .iter()
                    .map(|&h| (h / 2, h % 2 == 1))
                    .collect();
                flatten_loop(self, &lp, tol * 0.25)
            })
            .collect();
        let bounded_comp: Vec<usize> = bounded
            .iter()
            .map(|&i| component(&cycles[i], &mut parent))
            .collect();

        let mut holes: Vec<Vec<usize>> = vec![Vec::new(); bounded.len()];
        for (ci, c) in cycles.iter().enumerate() {
            if c.area > area_eps {
                continue;
            }
            let comp = component(c, &mut parent);
            let probe = he_cubic(c.half_edges[0]).eval(0.5);
            let mut best: Option<(usize, f64)> = None;
            for (bi, &b) in bounded.iter().enumerate() {
                if bounded_comp[bi] == comp {
                    continue;
                }
                if winding_number(&bounded_polys[bi], probe) == 0 {
                    continue;
                }
                let a = cycles[b].area;
                if best.is_none_or(|(_, ba)| a < ba) {
                    best = Some((bi, a));
                }
            }
            if let Some((bi, _)) = best {
                holes[bi].push(ci);
            }
        }

        let mut faces = Vec::with_capacity(bounded.len());
        for (bi, &b) in bounded.iter().enumerate() {
            let mut area = cycles[b].area;
            let mut loops = cycle_loops(&cycles[b].half_edges, origin);
            for &hc in &holes[bi] {
                area += cycles[hc].area;
                loops.extend(cycle_loops(&cycles[hc].half_edges, origin));
            }
            let loops: Vec<OrientedLoop> = loops
                .into_iter()
                .map(|lp| lp.into_iter().map(|h| (h / 2, h % 2 == 1)).collect())
                .collect();
            faces.push(Face { loops, area });
        }
        faces
    }

    /// The bounded face containing `p`, if any. The network must be planar.
    pub fn face_at(&self, p: Vector, tol: f64) -> Option<Face> {
        self.faces(tol)
            .into_iter()
            .find(|f| f.contains(self, p, tol))
    }

    /// Is `p` inside region `r` under its winding rule? Loops that do not
    /// return to their start are closed implicitly, as Figma fills them.
    pub fn region_contains(&self, r: usize, p: Vector, tol: f64) -> bool {
        let Some(region) = self.regions.get(r) else {
            return false;
        };
        let mut w = 0;
        for lp in &region.loops {
            let Some((orient, _)) = self.chain_loop(lp) else {
                continue;
            };
            w += winding_number(&flatten_loop(self, &orient, tol * 0.25), p);
        }
        match region.winding_rule {
            WindingRule::NonZero => w != 0,
            WindingRule::EvenOdd => w % 2 != 0,
        }
    }

    /// Figma's "click an enclosed area to fill it": planarise, find the face
    /// under `p`, and fill it if no region covers it. If regions do cover it,
    /// unfill just that face: each covering region is replaced by one region
    /// per other face it covered (keeping its fills), so the rest of the area
    /// stays filled.
    pub fn toggle_fill_at(&mut self, p: Vector, tol: f64) -> FillToggle {
        self.planarize(tol);
        let faces = self.faces(tol);
        let Some(fi) = faces.iter().position(|f| f.contains(self, p, tol)) else {
            return FillToggle::NoFace;
        };
        let covering: Vec<usize> = (0..self.regions.len())
            .filter(|&r| self.region_contains(r, p, tol))
            .collect();
        if covering.is_empty() {
            self.regions.push(faces[fi].region());
            return FillToggle::Filled {
                region: self.regions.len() - 1,
            };
        }
        let samples: Vec<Option<Vector>> =
            faces.iter().map(|f| f.interior_point(self, tol)).collect();
        let mut taken = vec![false; faces.len()];
        taken[fi] = true;
        let mut replacements = Vec::new();
        for &r in &covering {
            for (k, face) in faces.iter().enumerate() {
                if taken[k] {
                    continue;
                }
                let Some(s) = samples[k] else { continue };
                if self.region_contains(r, s, tol) {
                    taken[k] = true;
                    let old = &self.regions[r];
                    replacements.push(VectorRegion {
                        fills: old.fills.clone(),
                        fill_style_id: old.fill_style_id.clone(),
                        ..face.region()
                    });
                }
            }
        }
        for &r in covering.iter().rev() {
            self.regions.remove(r);
        }
        self.regions.extend(replacements);
        FillToggle::Unfilled
    }

    /// Replace the regions with one per bounded face (every enclosed area
    /// filled): what Figma paints for a network that declares no regions.
    pub fn fill_all_faces(&mut self, tol: f64) {
        self.planarize(tol);
        let faces = self.faces(tol);
        self.regions = faces.iter().map(Face::region).collect();
    }
}

fn wrap_angle(a: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let mut a = a % tau;
    if a > std::f64::consts::PI {
        a -= tau;
    } else if a <= -std::f64::consts::PI {
        a += tau;
    }
    a
}

/// Split a walked cycle into closed loops, dropping half-edges whose twin is
/// in the same cycle (spokes and bridges, which enclose nothing).
fn cycle_loops(cycle: &[usize], origin: impl Fn(usize) -> usize) -> Vec<Vec<usize>> {
    let in_cycle: std::collections::HashSet<usize> = cycle.iter().copied().collect();
    let kept: Vec<usize> = cycle
        .iter()
        .copied()
        .filter(|&h| !in_cycle.contains(&(h ^ 1)))
        .collect();
    let dest = |h: usize| origin(h ^ 1);
    let mut used = vec![false; kept.len()];
    let mut loops = Vec::new();
    for s in 0..kept.len() {
        if used[s] {
            continue;
        }
        let start_v = origin(kept[s]);
        let mut lp = Vec::new();
        let mut i = s;
        loop {
            used[i] = true;
            lp.push(kept[i]);
            let end = dest(kept[i]);
            if end == start_v {
                break;
            }
            let found = (1..kept.len())
                .map(|k| (i + k) % kept.len())
                .find(|&j| !used[j] && origin(kept[j]) == end);
            match found {
                Some(j) => i = j,
                None => break,
            }
        }
        loops.push(lp);
    }
    loops
}
