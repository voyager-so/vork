//! The polygon lane: the boolean and Pathfinder ops on flattened geometry,
//! through `i_overlay` (MIT/Apache, integer arithmetic), refitted to curves.
//!
//! flo_curves (the exact lane, `solve.rs`) is curve-exact, but its ray casting
//! sorts collisions with a comparator that is not a total order. The vendored
//! sort no longer panics on that (`native/vendor/flo_curves/MODA-PATCH.md`),
//! but on some inputs (Blob Brush outlines with cusps, a shape against a near
//! copy of itself) the answer is still wrong: a lost or extra piece.
//! `i_overlay` snaps to an integer grid, so it has no such failure mode. Its
//! price is the flattening and refit: the answer is within `FLATTEN_TOLERANCE +
//! FIT_TOLERANCE` of the exact region, not made of the operands' own curve
//! pieces, and it has more anchors.
//!
//! So both lanes run on every request. The exact answer is kept when it
//! [`agrees`] with the polygon one (the same region within `AGREEMENT`), and
//! the polygon answer is used when it does not. The ignored
//! `quality_tests::measure_lane_quality` shows the numbers behind this.
//!
//! The polygon lane itself:
//!
//! 1. Every operand's cubics are flattened within `FLATTEN_TOLERANCE`.
//! 2. Each operand is resolved under its own fill rule (`simplify_shape`).
//! 3. The ops run on the rings (the booleans directly; Pathfinder through the
//!    [`Polygon`] kernel, which carries the polygons between its steps).
//! 4. Each output ring is refitted by the stroke outliner's fitter within
//!    `FIT_TOLERANCE`. It breaks at every vertex that is not an interior
//!    flattening sample, so the operands' own anchors and every new crossing
//!    stay sharp, curves come back as a few cubics, and straight edges come
//!    back as lines.

use std::collections::HashMap;

use flo_curves::bezier::path::{
    path_add, path_full_intersect, path_intersect, path_sub, SimpleBezierPath,
};
use flo_curves::geo::Coord2;
use i_overlay::core::fill_rule::FillRule as OverlayFill;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::simplify::SimplifyShape;
use i_overlay::float::single::SingleFloatOverlay;
use i_overlay::mesh::float::outline::offset::OutlineOffset;
use i_overlay::mesh::float::style::OutlineStyle;

use super::solve::{check_intermediate, check_step, err, Region};
use super::{BooleanError, BooleanOp, FillRule};
use crate::path_geom::fit::{fit_polyline, FitSeg};
use crate::path_geom::vec2::{v2, V2};

/// Largest distance between a flattened chord and its curve (path units).
pub(crate) const FLATTEN_TOLERANCE: f64 = 0.02;
/// Largest distance between a refitted ring and the polygon it came from.
pub(crate) const FIT_TOLERANCE: f64 = 0.05;
/// How wide (path units) the difference between the exact and polygon answers
/// may be and still count as the same region (`all_slivers`). Both are
/// flattened to `FLATTEN_TOLERANCE` for the comparison, so two correct answers
/// differ by slivers a few hundredths wide; a lost, extra or doubled piece is
/// wider than this unless it is too small to see.
pub(crate) const AGREEMENT: f64 = 0.1;
/// Chords per cubic are capped, so a huge cubic cannot allocate without bound.
const MAX_CHORDS_PER_CUBIC: usize = 512;
/// Budget on the flattened input: `MAX_TOTAL_CUBICS` (32_000) at ~30
/// chords per cubic. `i_overlay` is O(n log n), so this is a fraction of a second.
const MAX_FLAT_VERTICES: usize = 1_000_000;
/// Budget on the vertices the `Polygon` kernel carries between Pathfinder
/// steps (an arrangement's faces, a union): the flattened input, cut up.
const MAX_CARRIED_VERTICES: usize = 2 * MAX_FLAT_VERTICES;

type Contour = Vec<[f64; 2]>;
pub(crate) type Contours = Vec<Contour>;

// ---------------------------------------------------------------------------
// Flattening
// ---------------------------------------------------------------------------

/// Flattened operands, plus the positions of every *interior* flattening
/// sample: what tells a smooth vertex from a corner after the op.
pub(crate) struct Flat {
    pub(crate) operands: Vec<Contours>,
    smooth: SmoothSet,
}

/// Flatten every region (already translated to the origin by `convert`).
/// Refuses `too_complex` past `MAX_FLAT_VERTICES`.
pub(crate) fn flatten(regions: &[Region]) -> Result<Flat, BooleanError> {
    let mut smooth = SmoothSet::new(extent(regions.iter().flatten()));
    let operands: Vec<Contours> = regions
        .iter()
        .map(|region| flatten_region(region, Some(&mut smooth)))
        .collect();
    let vertices: usize = operands.iter().flatten().map(Vec::len).sum();
    if vertices > MAX_FLAT_VERTICES {
        return Err(err(
            "too_complex",
            format!("input flattens to {vertices} vertices (limit {MAX_FLAT_VERTICES})"),
        ));
    }
    Ok(Flat { operands, smooth })
}

impl Flat {
    /// Rings back to cubics (see the module doc for where they break).
    pub(crate) fn refit(&self, rings: &Contours) -> Region {
        refit(rings, &self.smooth)
    }

    /// A `Polygon`-kernel region (straight cubics, see [`lines`]) refitted:
    /// closed rings, or for an open run (outline's) one open polyline per
    /// subpath.
    pub(crate) fn refit_lines(&self, region: &Region, closed: bool) -> Region {
        if closed {
            return refit(&flatten_region(region, None), &self.smooth);
        }
        region
            .iter()
            .filter_map(|(start, segs)| {
                let mut run = vec![[start.0, start.1]];
                run.extend(segs.iter().map(|(_, _, p)| [p.0, p.1]));
                fit(&run, &self.smooth, false)
            })
            .collect()
    }
}

/// Largest coordinate magnitude (at least 1), for the smooth-set tolerance.
fn extent<'a>(subpaths: impl Iterator<Item = &'a SimpleBezierPath>) -> f64 {
    subpaths
        .flat_map(|(start, segs)| std::iter::once(*start).chain(segs.iter().map(|(_, _, p)| *p)))
        .map(|p| p.0.abs().max(p.1.abs()))
        .fold(1.0, f64::max)
}

fn flatten_region(region: &Region, mut smooth: Option<&mut SmoothSet>) -> Contours {
    region
        .iter()
        .map(|subpath| flatten_subpath(subpath, smooth.as_deref_mut()))
        .filter(|c| c.len() >= 3)
        .collect()
}

fn flatten_subpath(
    (start, segments): &SimpleBezierPath,
    mut smooth: Option<&mut SmoothSet>,
) -> Contour {
    let mut out = vec![[start.0, start.1]];
    let mut p0 = *start;
    for (c1, c2, p3) in segments {
        // A cubic cut into n equal-t chords is within (3/4)·L/n² of them, L the
        // larger second difference of its control polygon.
        let l = (p0.0 - 2.0 * c1.0 + c2.0)
            .hypot(p0.1 - 2.0 * c1.1 + c2.1)
            .max((c1.0 - 2.0 * c2.0 + p3.0).hypot(c1.1 - 2.0 * c2.1 + p3.1));
        let n =
            ((0.75 * l / FLATTEN_TOLERANCE).sqrt().ceil() as usize).clamp(1, MAX_CHORDS_PER_CUBIC);
        for i in 1..n {
            let t = i as f64 / n as f64;
            let mt = 1.0 - t;
            let (a, b, c, d) = (mt * mt * mt, 3.0 * mt * mt * t, 3.0 * mt * t * t, t * t * t);
            let p = [
                a * p0.0 + b * c1.0 + c * c2.0 + d * p3.0,
                a * p0.1 + b * c1.1 + c * c2.1 + d * p3.1,
            ];
            if let Some(s) = smooth.as_deref_mut() {
                s.insert(p);
            }
            out.push(p);
        }
        out.push([p3.0, p3.1]);
        p0 = *p3;
    }
    if out.len() > 1 && out[0] == out[out.len() - 1] {
        out.pop();
    }
    out
}

// ---------------------------------------------------------------------------
// Polygon ops
// ---------------------------------------------------------------------------

fn overlay_fill(rule: FillRule) -> OverlayFill {
    match rule {
        FillRule::NonZero => OverlayFill::NonZero,
        FillRule::EvenOdd => OverlayFill::EvenOdd,
    }
}

fn contours(shapes: Vec<Vec<Contour>>) -> Contours {
    shapes
        .into_iter()
        .flatten()
        .filter(|c| c.len() >= 3)
        .collect()
}

/// Every operand resolved under its own fill rule: non-overlapping rings.
pub(crate) fn resolve_all(flat: &Flat, fills: &[FillRule]) -> Vec<Contours> {
    flat.operands
        .iter()
        .zip(fills)
        .map(|(c, f)| {
            if c.is_empty() {
                Vec::new()
            } else {
                contours(c.simplify_shape(overlay_fill(*f)))
            }
        })
        .collect()
}

/// One overlay rule on two ring sets, read even-odd.
fn apply(a: &Contours, b: &Contours, rule: OverlayRule) -> Contours {
    contours(a.overlay(b, rule, OverlayFill::EvenOdd))
}

pub(crate) fn union_all<'a>(parts: impl IntoIterator<Item = &'a Contours>) -> Contours {
    fold_balanced(parts, OverlayRule::Union)
}

/// Fold an associative, commutative rule (union, xor) over ring sets as a balanced tree: pairs, then pairs of
/// pairs. A left fold overlays the whole growing accumulator against each next operand, so N operands cost
/// O(N²) in the accumulator's size; the tree keeps both sides of every overlay about the same size, O(N log N).
/// The region is the same either way; ring order and vertex placement may differ (every answer is compared
/// by area, see `agrees`).
fn fold_balanced<'a>(parts: impl IntoIterator<Item = &'a Contours>, rule: OverlayRule) -> Contours {
    let mut level: Vec<Contours> = parts
        .into_iter()
        .filter(|p| !p.is_empty())
        .cloned()
        .collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut it = level.into_iter();
        while let Some(a) = it.next() {
            next.push(match it.next() {
                Some(b) => apply(&a, &b, rule),
                None => a,
            });
        }
        level = next;
    }
    level.pop().unwrap_or_default()
}

pub(crate) fn intersect(a: &Contours, b: &Contours) -> Contours {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    apply(a, b, OverlayRule::Intersect)
}

pub(crate) fn difference(a: &Contours, b: &Contours) -> Contours {
    if a.is_empty() || b.is_empty() {
        return a.clone();
    }
    apply(a, b, OverlayRule::Difference)
}

/// The four boolean ops on resolved operands: the same regions `solve.rs` folds
/// (subtract is `a − ∪rest`), with union and exclude folded as a balanced tree.
pub(crate) fn boolean(op: BooleanOp, resolved: &[Contours]) -> Contours {
    match op {
        BooleanOp::Union => union_all(resolved),
        BooleanOp::Subtract => difference(&resolved[0], &union_all(&resolved[1..])),
        BooleanOp::Intersect => resolved[1..]
            .iter()
            .fold(resolved[0].clone(), |acc, next| intersect(&acc, next)),
        BooleanOp::Exclude => fold_balanced(resolved, OverlayRule::Xor),
    }
}

// ---------------------------------------------------------------------------
// Agreement
// ---------------------------------------------------------------------------

fn signed_area(c: &Contour) -> f64 {
    let n = c.len();
    (0..n)
        .map(|i| {
            let (a, b) = (c[i], c[(i + 1) % n]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f64>()
        / 2.0
}

/// Nothing in `shapes` is wider than `AGREEMENT`: shrunk by half of it on
/// every side, it vanishes. Two flattenings of the same boundary differ by
/// slivers a few hundredths wide; a lost, extra or doubled piece of any
/// visible size survives the shrink, even where it touches those slivers.
fn all_slivers(shapes: &Vec<Vec<Contour>>) -> bool {
    if shapes.is_empty() {
        return true;
    }
    shapes
        .outline(&OutlineStyle::new(-AGREEMENT / 2.0))
        .iter()
        .flatten()
        .all(|c| signed_area(c).abs() <= 1e-9)
}

/// True when the exact lane's `pieces` (each read even-odd) tile the polygon
/// lane's `expected` region: what they cover differs from it only by slivers,
/// and where pieces overlap is only slivers too.
pub(crate) fn agrees(pieces: &[&Region], expected: &Contours) -> bool {
    // Each piece resolved, so its outers run counter-clockwise and its holes
    // clockwise (`i_overlay`'s convention): over all of them together the
    // winding number is how many pieces cover a point.
    let all: Contours = pieces
        .iter()
        .flat_map(|r| {
            let flat = flatten_region(r, None);
            if flat.is_empty() {
                Vec::new()
            } else {
                contours(flat.simplify_shape(OverlayFill::EvenOdd))
            }
        })
        .collect();
    let covered = contours(all.simplify_shape(OverlayFill::NonZero));
    if !all_slivers(&covered.overlay(expected, OverlayRule::Xor, OverlayFill::EvenOdd)) {
        return false;
    }
    // Covered twice or more: a clockwise frame around everything takes one off
    // every winding number, and what is still positive is the overlap.
    let Some(frame) = clockwise_frame(&all) else {
        return true;
    };
    let mut framed = all;
    framed.push(frame);
    all_slivers(&framed.simplify_shape(OverlayFill::Positive))
}

/// A clockwise rectangle a unit outside every ring's bounds.
fn clockwise_frame(rings: &Contours) -> Option<Contour> {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for p in rings.iter().flatten() {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    if x0 > x1 {
        return None;
    }
    let (x0, y0, x1, y1) = (x0 - 1.0, y0 - 1.0, x1 + 1.0, y1 + 1.0);
    Some(vec![[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
}

// ---------------------------------------------------------------------------
// Refit
// ---------------------------------------------------------------------------

fn refit(rings: &Contours, smooth: &SmoothSet) -> Region {
    rings
        .iter()
        .filter_map(|ring| fit(ring, smooth, true))
        .collect()
}

/// One polyline (a ring when `closed`) fitted within `FIT_TOLERANCE`.
fn fit(line: &Contour, smooth: &SmoothSet, closed: bool) -> Option<SimpleBezierPath> {
    let points: Vec<V2> = line.iter().map(|p| v2(p[0], p[1])).collect();
    let breaks: Vec<bool> = line.iter().map(|p| !smooth.contains(*p)).collect();
    let segs = fit_polyline(&points, &breaks, closed, FIT_TOLERANCE);
    let start = match segs.first()? {
        FitSeg::Line(a, _) => *a,
        FitSeg::Cubic(c) => c.0[0],
    };
    let cubics = segs
        .iter()
        .map(|s| match s {
            FitSeg::Line(a, b) => line_cubic(*a, *b),
            FitSeg::Cubic(c) => (coord(c.0[1]), coord(c.0[2]), coord(c.0[3])),
        })
        .collect();
    Some((coord(start), cubics))
}

fn line_cubic(a: V2, b: V2) -> (Coord2, Coord2, Coord2) {
    (
        coord(a.lerp(b, 1.0 / 3.0)),
        coord(a.lerp(b, 2.0 / 3.0)),
        coord(b),
    )
}

/// Rings as straight cubics, one per edge: a lossless `Region` for a polygon.
/// [`flatten_region`] hands back exactly these vertices (a straight cubic
/// flattens to one chord), so the `Polygon` kernel can carry polygons between
/// steps without refitting each time.
pub(crate) fn lines(rings: &Contours) -> Region {
    rings
        .iter()
        .map(|ring| {
            let at = |i: usize| v2(ring[i % ring.len()][0], ring[i % ring.len()][1]);
            (
                coord(at(0)),
                (0..ring.len())
                    .map(|i| line_cubic(at(i), at(i + 1)))
                    .collect(),
            )
        })
        .collect()
}

fn coord(p: V2) -> Coord2 {
    Coord2(p.x, p.y)
}

/// Positions of interior flattening samples, looked up with a tolerance: the
/// integer engine hands vertices back rounded to its grid, never bit-exact.
struct SmoothSet {
    cell: f64,
    eps: f64,
    cells: HashMap<(i64, i64), Vec<[f64; 2]>>,
}

impl SmoothSet {
    fn new(extent: f64) -> SmoothSet {
        // i_overlay's i32 grid spans the operands' bounds in ~2^30 steps, so
        // 1e-7 of the extent is ~100 grid steps: far below any chord.
        let eps = extent * 1e-7;
        SmoothSet {
            cell: eps * 4.0,
            eps,
            cells: HashMap::new(),
        }
    }

    fn key(&self, p: [f64; 2]) -> (i64, i64) {
        (
            (p[0] / self.cell).floor() as i64,
            (p[1] / self.cell).floor() as i64,
        )
    }

    fn insert(&mut self, p: [f64; 2]) {
        let k = self.key(p);
        self.cells.entry(k).or_default().push(p);
    }

    fn contains(&self, p: [f64; 2]) -> bool {
        let (kx, ky) = self.key(p);
        (-1..=1).any(|dx| {
            (-1..=1).any(|dy| {
                self.cells.get(&(kx + dx, ky + dy)).is_some_and(|v| {
                    v.iter()
                        .any(|q| (q[0] - p[0]).abs() <= self.eps && (q[1] - p[1]).abs() <= self.eps)
                })
            })
        })
    }
}

// ---------------------------------------------------------------------------
// Kernels: what Pathfinder's ops run on
// ---------------------------------------------------------------------------

/// The region ops Pathfinder is built from, on either lane. Regions are sets
/// of closed rings read even-odd.
pub(crate) trait Kernel {
    /// The snapping accuracy the caller asked for (sliver and edge-matching
    /// tolerances in `pathfinder.rs` scale with it).
    fn accuracy(&self) -> f64;
    fn add(&self, a: &Region, b: &Region) -> Region;
    fn sub(&self, a: &Region, b: &Region) -> Region;
    fn intersect(&self, a: &Region, b: &Region) -> Region;
    /// `(a ∩ b, a − b, b − a)`.
    fn split(&self, a: &Region, b: &Region) -> (Region, Region, Region);
    /// The guard in front of every call on `a` and `b`.
    fn step(&self, a: &Region, b: &Region) -> Result<(), BooleanError>;
    /// The guard on what is carried between calls, in segments.
    fn carry(&self, segments: usize) -> Result<(), BooleanError>;
}

/// flo_curves' curve-exact arithmetic.
pub(crate) struct Exact(pub(crate) f64);

impl Kernel for Exact {
    fn accuracy(&self) -> f64 {
        self.0
    }
    fn add(&self, a: &Region, b: &Region) -> Region {
        path_add::<SimpleBezierPath>(a, b, self.0)
    }
    fn sub(&self, a: &Region, b: &Region) -> Region {
        path_sub::<SimpleBezierPath>(a, b, self.0)
    }
    fn intersect(&self, a: &Region, b: &Region) -> Region {
        path_intersect::<SimpleBezierPath>(a, b, self.0)
    }
    fn split(&self, a: &Region, b: &Region) -> (Region, Region, Region) {
        let split = path_full_intersect::<SimpleBezierPath>(a, b, self.0);
        let [a_out, b_out] = split.exterior_paths;
        (split.intersecting_path, a_out, b_out)
    }
    /// flo_curves' pairwise-work and carried-cubic budgets (`solve.rs`).
    fn step(&self, a: &Region, b: &Region) -> Result<(), BooleanError> {
        check_step(a, b)
    }
    fn carry(&self, segments: usize) -> Result<(), BooleanError> {
        check_intermediate(segments)
    }
}

/// `i_overlay` on polygons carried as straight-cubic regions ([`lines`]): no
/// flattening or refit between steps, so nothing accumulates. The caller
/// starts from [`lines`] of the resolved operands and refits the pieces once
/// at the end ([`Flat::refit_lines`]).
pub(crate) struct Polygon(pub(crate) f64);

impl Polygon {
    fn run(
        &self,
        a: &Region,
        b: &Region,
        f: impl FnOnce(&Contours, &Contours) -> Contours,
    ) -> Region {
        lines(&f(&flatten_region(a, None), &flatten_region(b, None)))
    }
}

impl Kernel for Polygon {
    fn accuracy(&self) -> f64 {
        self.0
    }
    fn add(&self, a: &Region, b: &Region) -> Region {
        self.run(a, b, |a, b| union_all([a, b]))
    }
    fn sub(&self, a: &Region, b: &Region) -> Region {
        self.run(a, b, difference)
    }
    fn intersect(&self, a: &Region, b: &Region) -> Region {
        self.run(a, b, intersect)
    }
    fn split(&self, a: &Region, b: &Region) -> (Region, Region, Region) {
        let (fa, fb) = (flatten_region(a, None), flatten_region(b, None));
        (
            lines(&intersect(&fa, &fb)),
            lines(&difference(&fa, &fb)),
            lines(&difference(&fb, &fa)),
        )
    }
    /// `i_overlay` is O(n log n) in the vertices, so there is no pairwise
    /// budget to keep; its regions are polygons, one segment per vertex.
    fn step(&self, _a: &Region, _b: &Region) -> Result<(), BooleanError> {
        Ok(())
    }
    fn carry(&self, segments: usize) -> Result<(), BooleanError> {
        if segments > MAX_CARRIED_VERTICES {
            return Err(err(
                "too_complex",
                format!("{segments} polygon vertices carried between steps (limit {MAX_CARRIED_VERTICES})"),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: f64, y: f64, s: f64) -> Region {
        let line = |a: Coord2, b: Coord2| {
            (
                Coord2(a.0 + (b.0 - a.0) / 3.0, a.1 + (b.1 - a.1) / 3.0),
                Coord2(a.0 + (b.0 - a.0) * 2.0 / 3.0, a.1 + (b.1 - a.1) * 2.0 / 3.0),
                b,
            )
        };
        let c = [
            Coord2(x, y),
            Coord2(x + s, y),
            Coord2(x + s, y + s),
            Coord2(x, y + s),
        ];
        vec![(c[0], (0..4).map(|i| line(c[i], c[(i + 1) % 4])).collect())]
    }

    fn rings(r: &Region) -> Contours {
        flatten_region(r, None)
    }

    #[test]
    fn outers_wind_positive() {
        // `agrees` counts coverage with `FillRule::Positive`, so a resolved
        // outer ring must be one `i_overlay` winds positively.
        let outer = contours(rings(&square(0.0, 0.0, 10.0)).simplify_shape(OverlayFill::EvenOdd));
        assert!(signed_area(&outer[0]) > 0.0);
        assert_eq!(
            contours(outer.simplify_shape(OverlayFill::Positive)).len(),
            1
        );
        assert!(contours(outer.simplify_shape(OverlayFill::Negative)).is_empty());
    }

    #[test]
    fn agreement_accepts_a_tiling_and_rejects_gaps_and_overlaps() {
        let (a, b, c) = (
            square(0.0, 0.0, 10.0),
            square(10.0, 0.0, 10.0),
            square(20.0, 0.0, 10.0),
        );
        let expected = union_all(&[rings(&a), rings(&b), rings(&c)]);
        assert!(agrees(&[&a, &b, &c], &expected));
        // A lost piece.
        assert!(!agrees(&[&a, &c], &expected));
        // An extra piece outside the region.
        let d = square(40.0, 0.0, 5.0);
        assert!(!agrees(&[&a, &b, &c, &d], &expected));
        // The same union, with a piece doubled, and tripled (odd coverage).
        assert!(!agrees(&[&a, &b, &b, &c], &expected));
        assert!(!agrees(&[&a, &b, &b, &b, &c], &expected));
        // A hair's difference is a sliver, not a disagreement.
        let nudged = square(0.0, 0.0, 10.01);
        assert!(agrees(&[&nudged, &b, &c], &expected));
    }
}
