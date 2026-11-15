//! Booleans and Pathfinder on hundreds of operands.
//!
//! The fixture is `scripts/perf/largeDoc.mjs`'s flat document (closed kinds 0 and 1: a jittered cubic blob and
//! a jittered star polygon, 20 anchors each, in 40-unit cells with 52-unit boxes so neighbours overlap), ported
//! here with the same PRNG so the timings line up with `scripts/perf/headless.mjs`.
//!
//! The tests check every answer against a point oracle: on a grid of sample points (each kept only where the
//! operands' own membership is the same a little way off in every direction, so no sample sits on a boundary),
//! the result must contain exactly the points the op's definition says it does. On subsets small enough for
//! the exact lane, they also check the exact answer against the polygon lane's.
//!
//! `bench_large_inputs` is `#[ignore]`d and prints timings for docs/qa/perf.md:
//!
//!   cargo test --release -p moda-scene bench_large_inputs -- --ignored --nocapture

use std::collections::HashMap;
use std::time::Instant;

use super::convert::{flatten_subpath, points_to_subpaths};
use super::solve::{solve_on, Lane};
use super::tests::{even_odd_area, rect};
use super::*;

const CELL: f64 = 40.0;
const SIZE: f64 = 52.0;

/// `largeDoc.mjs`'s LCG.
struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 as f64 / 4_294_967_296.0
    }
}

fn round(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// `pathPoints` for kinds 0 (cubic blob) and 1 (polygon star), offset to `(ox, oy)`.
fn path_points(kind: usize, anchors: usize, r: &mut Lcg, ox: f64, oy: f64) -> Vec<PathPoint> {
    let c = SIZE / 2.0;
    let radius = |k: usize, r: &mut Lcg| {
        if kind == 0 {
            14.0 + r.next() * 11.0
        } else if k % 2 == 1 {
            11.0 + r.next() * 5.0
        } else {
            20.0 + r.next() * 5.0
        }
    };
    let at = |k: usize, rad: f64| {
        let a = (k as f64 / anchors as f64) * std::f64::consts::PI * 2.0;
        (round(c + a.cos() * rad), round(c + a.sin() * rad))
    };
    let mut local: Vec<PathPoint> = Vec::with_capacity(anchors);
    let rad = radius(0, r);
    let (x, y) = at(0, rad);
    local.push(PathPoint::Move { x, y });
    let mut prev = (x, y);
    for k in 1..anchors {
        let rad = radius(k, r);
        let (x, y) = at(k, rad);
        if kind == 1 {
            local.push(PathPoint::Line { x, y });
        } else {
            let cp1 = (
                round(prev.0 + (x - prev.0) / 3.0 + (r.next() - 0.5) * 6.0),
                round(prev.1 + (y - prev.1) / 3.0 + (r.next() - 0.5) * 6.0),
            );
            let cp2 = (
                round(prev.0 + (x - prev.0) * 2.0 / 3.0 + (r.next() - 0.5) * 6.0),
                round(prev.1 + (y - prev.1) * 2.0 / 3.0 + (r.next() - 0.5) * 6.0),
            );
            local.push(PathPoint::Curve { x, y, cp1, cp2 });
        }
        prev = (x, y);
    }
    translate(local, ox, oy)
}

fn translate(points: Vec<PathPoint>, dx: f64, dy: f64) -> Vec<PathPoint> {
    points
        .into_iter()
        .map(|p| match p {
            PathPoint::Move { x, y } => PathPoint::Move {
                x: x + dx,
                y: y + dy,
            },
            PathPoint::Line { x, y } => PathPoint::Line {
                x: x + dx,
                y: y + dy,
            },
            PathPoint::Curve { x, y, cp1, cp2 } => PathPoint::Curve {
                x: x + dx,
                y: y + dy,
                cp1: (cp1.0 + dx, cp1.1 + dy),
                cp2: (cp2.0 + dx, cp2.1 + dy),
            },
            other => other,
        })
        .collect()
}

/// `buildLargeDoc({ paths: n, anchors, groups: false, kinds: [0, 1] })`'s paths, bottom-most first.
pub(super) fn grid_operands(n: usize, anchors: usize) -> Vec<BooleanOperand> {
    let mut r = Lcg(7);
    let cols = ((((n as f64).sqrt() / 10.0).ceil() * 10.0) as usize).max(10);
    (0..n)
        .map(|i| {
            let (col, row) = (i % cols, i / cols);
            BooleanOperand {
                points: path_points(
                    i % 2,
                    anchors,
                    &mut r,
                    40.0 + col as f64 * CELL,
                    40.0 + row as f64 * CELL,
                ),
                fill_rule: FillRule::NonZero,
            }
        })
        .collect()
}

/// Clusters of overlapping grid paths (`sizes[c]` paths in cluster `c`), the clusters well apart: what a
/// selection of many separate icons looks like.
fn clusters(sizes: &[usize], anchors: usize) -> Vec<Vec<BooleanOperand>> {
    sizes
        .iter()
        .enumerate()
        .map(|(c, &n)| {
            let (dx, dy) = ((c % 10) as f64 * 1_000.0, (c / 10) as f64 * 1_000.0);
            grid_operands(n, anchors)
                .into_iter()
                .map(|o| BooleanOperand {
                    points: translate(o.points, dx, dy),
                    fill_rule: o.fill_rule,
                })
                .collect()
        })
        .collect()
}

fn boolean(op: BooleanOp, operands: Vec<BooleanOperand>) -> Result<BooleanOutcome, BooleanError> {
    boolean_paths(&BooleanRequest {
        op,
        operands,
        accuracy: None,
    })
}

fn pathfinder(
    op: PathfinderOp,
    operands: Vec<BooleanOperand>,
) -> Result<PathfinderOutcome, BooleanError> {
    let paint = (0..operands.len())
        .map(|i| OperandPaint {
            painted: true,
            key: Some(format!("#{}", i % 8)),
        })
        .collect();
    pathfinder_paths(&PathfinderRequest {
        op,
        operands,
        paint,
        accuracy: None,
    })
}

fn describe_boolean(r: &Result<BooleanOutcome, BooleanError>) -> String {
    match r {
        Ok(BooleanOutcome::Shape {
            points,
            subpath_count,
            ..
        }) => format!("ok {} pts {} subpaths", points.len(), subpath_count),
        Ok(BooleanOutcome::Empty) => "empty".into(),
        Err(e) => format!("error {} ({})", e.code, e.message),
    }
}

fn describe_pathfinder(r: &Result<PathfinderOutcome, BooleanError>) -> String {
    match r {
        Ok(PathfinderOutcome::Pieces(p)) => format!(
            "ok {} pieces {} pts",
            p.len(),
            p.iter().map(|p| p.points.len()).sum::<usize>()
        ),
        Ok(PathfinderOutcome::Empty) => "empty".into(),
        Err(e) => format!("error {} ({})", e.code, e.message),
    }
}

#[test]
#[ignore]
fn bench_large_inputs() {
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;
    // `MODA_BENCH_MAX=<n>` skips the larger sizes (an uncapped run of the old solver takes minutes past 500).
    let max: usize = std::env::var("MODA_BENCH_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    for (anchors, sizes) in [
        (20, &[16, 64, 200, 500, 1000][..]),
        (8, &[500, 1000, 1500][..]),
    ] {
        for &n in sizes.iter().filter(|&&n| n <= max) {
            for op in [
                BooleanOp::Union,
                BooleanOp::Subtract,
                BooleanOp::Exclude,
                BooleanOp::Intersect,
            ] {
                let operands = grid_operands(n, anchors);
                let t = Instant::now();
                let r = boolean(op, operands);
                println!(
                    "grid {n:>5} x {anchors:>2} anchors {op:?}: {:.1} ms, {}",
                    ms(t),
                    describe_boolean(&r)
                );
            }
        }
    }
    for n in [64, 500, 1000].into_iter().filter(|&n| n <= max) {
        let operands: Vec<BooleanOperand> = clusters(&vec![5; n / 5], 20)
            .into_iter()
            .flatten()
            .collect();
        let t = Instant::now();
        let r = boolean(BooleanOp::Union, operands);
        println!(
            "clusters-of-5 {n:>5} Union: {:.1} ms, {}",
            ms(t),
            describe_boolean(&r)
        );
    }
    for n in [16, 64, 200, 300, 500].into_iter().filter(|&n| n <= max) {
        for op in [
            PathfinderOp::Divide,
            PathfinderOp::Trim,
            PathfinderOp::Merge,
            PathfinderOp::Crop,
            PathfinderOp::Outline,
            PathfinderOp::MinusBack,
        ] {
            let operands = grid_operands(n, 20);
            let t = Instant::now();
            let r = pathfinder(op, operands);
            println!(
                "grid {n:>5} x 20 anchors {op:?}: {:.1} ms, {}",
                ms(t),
                describe_pathfinder(&r)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Point oracle
// ---------------------------------------------------------------------------

type Pt = (f64, f64);

/// A path's rings, flattened (8 chords a cubic: within a few hundredths on these shapes).
fn rings(points: &[PathPoint]) -> Vec<Vec<Pt>> {
    points_to_subpaths(points, 1e-6)
        .iter()
        .map(|p| flatten_subpath(p, 8))
        .filter(|r| r.len() >= 3)
        .collect()
}

/// A path flattened to rings, with its box.
struct Shape {
    rings: Vec<Vec<Pt>>,
    min: Pt,
    max: Pt,
}

impl Shape {
    fn new(points: &[PathPoint]) -> Shape {
        let rings = rings(points);
        let (mut min, mut max) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
        for &(x, y) in rings.iter().flatten() {
            min = (min.0.min(x), min.1.min(y));
            max = (max.0.max(x), max.1.max(y));
        }
        Shape { rings, min, max }
    }

    fn in_box(&self, p: Pt) -> bool {
        p.0 >= self.min.0 && p.0 <= self.max.0 && p.1 >= self.min.1 && p.1 <= self.max.1
    }

    /// Signed winding number of every ring around `p`: non-zero is how an operand fills.
    fn nonzero(&self, p: Pt) -> bool {
        if !self.in_box(p) {
            return false;
        }
        let mut w = 0;
        for ring in &self.rings {
            for i in 0..ring.len() {
                let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
                let cross = (b.0 - a.0) * (p.1 - a.1) - (p.0 - a.0) * (b.1 - a.1);
                if a.1 <= p.1 && b.1 > p.1 && cross > 0.0 {
                    w += 1;
                } else if a.1 > p.1 && b.1 <= p.1 && cross < 0.0 {
                    w -= 1;
                }
            }
        }
        w != 0
    }

    /// Even-odd, how every result is emitted.
    fn even_odd(&self, p: Pt) -> bool {
        self.in_box(p)
            && self
                .rings
                .iter()
                .flat_map(|r| ring_edges(r))
                .filter(|e| crosses(*e, p))
                .count()
                % 2
                == 1
    }
}

fn ring_edges(ring: &[Pt]) -> impl Iterator<Item = (Pt, Pt)> + '_ {
    (0..ring.len()).map(move |i| (ring[i], ring[(i + 1) % ring.len()]))
}

/// The ray from `p` towards +x crosses the edge (half-open in y, so a vertex counts once).
fn crosses((a, b): (Pt, Pt), p: Pt) -> bool {
    (a.1 > p.1) != (b.1 > p.1) && p.0 < (b.0 - a.0) * (p.1 - a.1) / (b.1 - a.1) + a.0
}

/// Shapes bucketed into square cells by box, for point queries.
struct Index {
    shapes: Vec<Shape>,
    cell: f64,
    cells: HashMap<(i64, i64), Vec<usize>>,
}

impl Index {
    fn new(shapes: Vec<Shape>) -> Index {
        let cell = 50.0;
        let mut cells: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (i, s) in shapes.iter().enumerate() {
            if s.rings.is_empty() {
                continue;
            }
            let k0 = (
                (s.min.0 / cell).floor() as i64,
                (s.min.1 / cell).floor() as i64,
            );
            let k1 = (
                (s.max.0 / cell).floor() as i64,
                (s.max.1 / cell).floor() as i64,
            );
            for kx in k0.0..=k1.0 {
                for ky in k0.1..=k1.1 {
                    cells.entry((kx, ky)).or_default().push(i);
                }
            }
        }
        Index {
            shapes,
            cell,
            cells,
        }
    }

    fn key(&self, p: Pt) -> (i64, i64) {
        (
            (p.0 / self.cell).floor() as i64,
            (p.1 / self.cell).floor() as i64,
        )
    }

    /// Ascending indices of the shapes whose box holds `p`.
    fn near(&self, p: Pt) -> Vec<usize> {
        let mut out: Vec<usize> = self
            .cells
            .get(&self.key(p))
            .map(|v| {
                v.iter()
                    .copied()
                    .filter(|&i| self.shapes[i].in_box(p))
                    .collect()
            })
            .unwrap_or_default();
        out.sort_unstable();
        out
    }

    /// Ascending indices of the operands covering `p` (non-zero).
    fn covering(&self, p: Pt) -> Vec<usize> {
        self.near(p)
            .into_iter()
            .filter(|&i| self.shapes[i].nonzero(p))
            .collect()
    }

    fn bounds(&self) -> (Pt, Pt) {
        let (mut min, mut max) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
        for s in self.shapes.iter().filter(|s| !s.rings.is_empty()) {
            min = (min.0.min(s.min.0), min.1.min(s.min.1));
            max = (max.0.max(s.max.0), max.1.max(s.max.1));
        }
        (min, max)
    }
}

/// A result read even-odd, its edges bucketed by horizontal band: a +x ray from a point only meets edges in
/// the point's band, so a ring of thousands of edges costs a few dozen per query.
struct Answer {
    band: f64,
    bands: HashMap<i64, Vec<(Pt, Pt)>>,
}

impl Answer {
    fn new(points: &[PathPoint]) -> Answer {
        let band = 5.0;
        let mut bands: HashMap<i64, Vec<(Pt, Pt)>> = HashMap::new();
        for ring in rings(points) {
            for (a, b) in ring_edges(&ring) {
                let (lo, hi) = (a.1.min(b.1), a.1.max(b.1));
                for k in (lo / band).floor() as i64..=(hi / band).floor() as i64 {
                    bands.entry(k).or_default().push((a, b));
                }
            }
        }
        Answer { band, bands }
    }

    fn contains(&self, p: Pt) -> bool {
        self.bands
            .get(&((p.1 / self.band).floor() as i64))
            .is_some_and(|edges| edges.iter().filter(|e| crosses(**e, p)).count() % 2 == 1)
    }
}

/// Sample points `step` apart over `window` (default: the operands' bounds), each with the operands
/// covering it. A sample is dropped when the covering set changes within `MARGIN` of it: an answer may sit a
/// flattening and refit tolerance off the true boundary. Samples far from every operand are skipped.
fn samples(operands: &Index, step: f64, window: Option<(Pt, Pt)>) -> Vec<(Pt, Vec<usize>)> {
    const MARGIN: f64 = 0.3;
    let (min, max) = window.unwrap_or_else(|| operands.bounds());
    let mut out = Vec::new();
    let mut y = min.1 + step * 0.37;
    while y < max.1 {
        let mut x = min.0 + step * 0.61;
        while x < max.0 {
            let p = (x, y);
            x += step;
            if !operands.cells.contains_key(&operands.key(p)) {
                continue;
            }
            let here = operands.covering(p);
            let stable = [(MARGIN, 0.0), (-MARGIN, 0.0), (0.0, MARGIN), (0.0, -MARGIN)]
                .iter()
                .all(|(dx, dy)| operands.covering((p.0 + dx, p.1 + dy)) == here);
            if stable {
                out.push((p, here));
            }
        }
        y += step;
    }
    out
}

fn operand_index(operands: &[BooleanOperand]) -> Index {
    Index::new(operands.iter().map(|o| Shape::new(&o.points)).collect())
}

fn answer_of(r: Result<BooleanOutcome, BooleanError>, what: &str) -> Option<Answer> {
    match r {
        Ok(BooleanOutcome::Shape { points, .. }) => Some(Answer::new(&points)),
        Ok(BooleanOutcome::Empty) => None,
        Err(e) => panic!("{what}: {} ({})", e.code, e.message),
    }
}

/// The result contains exactly the samples `rule` says it should (one in a thousand may miss: a sample in a
/// sliver the margin did not catch).
fn check_boolean(
    what: &str,
    operands: &Index,
    result: Option<&Answer>,
    (step, window): (f64, Option<(Pt, Pt)>),
    rule: impl Fn(&[usize]) -> bool,
) {
    let samples = samples(operands, step, window);
    let wrong: Vec<Pt> = samples
        .iter()
        .filter(|(p, cover)| rule(cover) != result.is_some_and(|s| s.contains(*p)))
        .map(|(p, _)| *p)
        .collect();
    let inside = samples.iter().filter(|(_, c)| rule(c)).count();
    assert!(
        inside > 100,
        "{what}: only {inside} samples inside; the fixture is too small"
    );
    assert!(
        wrong.len() * 1_000 <= samples.len(),
        "{what}: {} of {} samples wrong, e.g. {:?}",
        wrong.len(),
        samples.len(),
        &wrong[..wrong.len().min(8)]
    );
}

fn request(op: BooleanOp, operands: &[BooleanOperand]) -> BooleanRequest {
    BooleanRequest {
        op,
        operands: operands.to_vec(),
        accuracy: None,
    }
}

// ---------------------------------------------------------------------------
// Booleans
// ---------------------------------------------------------------------------

#[test]
fn union_of_600_overlapping_paths() {
    let operands = grid_operands(600, 8);
    let index = operand_index(&operands);
    let result = answer_of(boolean(BooleanOp::Union, operands), "union of 600");
    check_boolean("union of 600", &index, result.as_ref(), (7.0, None), |c| {
        !c.is_empty()
    });
}

#[test]
fn exclude_of_500_overlapping_paths() {
    let operands = grid_operands(500, 8);
    let index = operand_index(&operands);
    let result = answer_of(boolean(BooleanOp::Exclude, operands), "exclude of 500");
    check_boolean(
        "exclude of 500",
        &index,
        result.as_ref(),
        (7.0, None),
        |c| c.len() % 2 == 1,
    );
}

#[test]
fn subtract_600_paths_from_a_rect_behind_them() {
    // The base covers a few dozen of the paths; the rest miss it and are dropped before the solve.
    let mut operands = vec![BooleanOperand {
        points: rect(150.0, 150.0, 260.0, 230.0),
        fill_rule: FillRule::NonZero,
    }];
    operands.extend(grid_operands(600, 8));
    let index = operand_index(&operands);
    let result = answer_of(boolean(BooleanOp::Subtract, operands), "subtract 600");
    let window = Some(((140.0, 140.0), (420.0, 390.0)));
    check_boolean(
        "subtract 600",
        &index,
        result.as_ref(),
        (3.0, window),
        |c| c == [0],
    );
}

#[test]
fn intersect_of_many_paths_that_share_a_region() {
    // 300 copies of a rect, each nudged: the answer is their common rect.
    let operands: Vec<BooleanOperand> = (0..300)
        .map(|i| BooleanOperand {
            points: rect(i as f64 * 0.1, (i % 7) as f64 * 0.2, 100.0, 80.0),
            fill_rule: FillRule::NonZero,
        })
        .collect();
    let index = operand_index(&operands);
    let n = operands.len();
    let result = answer_of(boolean(BooleanOp::Intersect, operands), "intersect of 300");
    check_boolean(
        "intersect of 300",
        &index,
        result.as_ref(),
        (4.0, None),
        |c| c.len() == n,
    );
}

#[test]
fn union_of_600_paths_in_separate_clusters_matches_each_cluster_on_its_own() {
    // Five clusters of 100 (past the exact lane's 64 operands: polygon lane) and twenty of 5 (exact lane):
    // bucketing solves each on its own, so the whole must be the sum of the clusters solved alone.
    let mut sizes = vec![100; 5];
    sizes.extend([5; 20]);
    let parts = clusters(&sizes, 8);
    let operands: Vec<BooleanOperand> = parts.iter().flatten().cloned().collect();
    assert_eq!(operands.len(), 600);
    let index = operand_index(&operands);
    let whole = match boolean(BooleanOp::Union, operands) {
        Ok(BooleanOutcome::Shape { points, .. }) => points,
        other => panic!("clusters: {}", describe_boolean(&other)),
    };
    check_boolean(
        "clusters",
        &index,
        Some(&Answer::new(&whole)),
        (7.0, None),
        |c| !c.is_empty(),
    );
    let alone: f64 = parts
        .iter()
        .map(|cluster| match boolean(BooleanOp::Union, cluster.clone()) {
            Ok(BooleanOutcome::Shape { points, .. }) => even_odd_area(&points),
            other => panic!("one cluster: {}", describe_boolean(&other)),
        })
        .sum();
    let area = even_odd_area(&whole);
    assert!(
        (area - alone).abs() <= alone * 1e-3,
        "the whole ({area}) is not the sum of its clusters ({alone})"
    );
}

#[test]
fn exact_answers_on_subsets_agree_with_the_polygon_lane() {
    // Subsets inside the exact lane's budget: what ships there is flo_curves' answer (checked), and it must
    // be the polygon lane's region, and the oracle's.
    let all = grid_operands(600, 8);
    for (from, n) in [(0, 40), (250, 60), (540, 60)] {
        let subset = &all[from..from + n];
        let index = operand_index(subset);
        for op in [BooleanOp::Union, BooleanOp::Exclude] {
            let what = format!("{op:?} of {from}..{}", from + n);
            let points = |r: Result<BooleanOutcome, BooleanError>, lane: &str| match r {
                Ok(BooleanOutcome::Shape { points, .. }) => points,
                other => panic!("{what} ({lane}): {}", describe_boolean(&other)),
            };
            let shipped = points(boolean_paths(&request(op, subset)), "shipped");
            let polygon = points(solve_on(&request(op, subset), Lane::Polygon), "polygon");
            // Exclude's exact pieces can share edges, which the nesting-depth area misreads; the oracle
            // below reads them even-odd, as the renderer does.
            if op == BooleanOp::Union {
                let (a, b) = (even_odd_area(&shipped), even_odd_area(&polygon));
                assert!(
                    (a - b).abs() <= b * 2e-3,
                    "{what}: shipped area {a}, polygon {b}"
                );
            }
            let rule = |c: &[usize]| match op {
                BooleanOp::Union => !c.is_empty(),
                _ => c.len() % 2 == 1,
            };
            check_boolean(
                &what,
                &index,
                Some(&Answer::new(&shipped)),
                (3.0, None),
                rule,
            );
            check_boolean(
                &what,
                &index,
                Some(&Answer::new(&polygon)),
                (3.0, None),
                rule,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Pathfinder
// ---------------------------------------------------------------------------

fn pieces_of(r: Result<PathfinderOutcome, BooleanError>, what: &str) -> Vec<PathfinderPiece> {
    match r {
        Ok(PathfinderOutcome::Pieces(p)) => p,
        Ok(PathfinderOutcome::Empty) => Vec::new(),
        Err(e) => panic!("{what}: {} ({})", e.code, e.message),
    }
}

/// Every sample `owner` gives an owner lies in exactly one piece, which wears paint `same_paint` accepts
/// for that owner; every other sample lies in none. Pieces must come back to front (sources ascending).
fn check_pieces(
    what: &str,
    operands: &Index,
    pieces: &[PathfinderPiece],
    window: Option<(Pt, Pt)>,
    owner: impl Fn(&[usize]) -> Option<usize>,
    same_paint: impl Fn(usize, usize) -> bool,
) {
    assert!(
        pieces.windows(2).all(|w| w[0].source <= w[1].source),
        "{what}: pieces are not back to front"
    );
    // Each piece is compound (a face and its holes), read even-odd.
    let shapes = Index::new(pieces.iter().map(|p| Shape::new(&p.points)).collect());
    let samples = samples(operands, 5.0, window);
    let mut wrong = Vec::new();
    for (p, cover) in &samples {
        let hit: Vec<usize> = shapes
            .near(*p)
            .into_iter()
            .filter(|&i| shapes.shapes[i].even_odd(*p))
            .collect();
        let ok = match owner(cover) {
            None => hit.is_empty(),
            Some(o) => hit.len() == 1 && same_paint(pieces[hit[0]].source, o),
        };
        if !ok {
            let sources: Vec<usize> = hit.iter().map(|&i| pieces[i].source).collect();
            wrong.push((*p, cover.clone(), sources));
        }
    }
    assert!(
        samples.len() > 1_000,
        "{what}: only {} samples",
        samples.len()
    );
    assert!(
        wrong.len() * 1_000 <= samples.len(),
        "{what}: {} of {} samples wrong, e.g. {:?}",
        wrong.len(),
        samples.len(),
        &wrong[..wrong.len().min(6)]
    );
}

#[test]
fn pathfinder_divide_trim_and_merge_on_210_paths() {
    let operands = grid_operands(210, 20);
    let index = operand_index(&operands);
    let top = |c: &[usize]| c.last().copied();
    let divide = pieces_of(pathfinder(PathfinderOp::Divide, operands.clone()), "divide");
    check_pieces("divide", &index, &divide, None, top, |a, b| a == b);
    let trim = pieces_of(pathfinder(PathfinderOp::Trim, operands.clone()), "trim");
    check_pieces("trim", &index, &trim, None, top, |a, b| a == b);
    // `pathfinder` keys operand i as `#{i % 8}`: a merged piece wears a same-keyed operand's paint.
    let merge = pieces_of(pathfinder(PathfinderOp::Merge, operands), "merge");
    check_pieces("merge", &index, &merge, None, top, |a, b| a % 8 == b % 8);
}

#[test]
fn pathfinder_crop_minus_back_and_outline_on_210_paths() {
    let mut operands = grid_operands(210, 20);
    // The frontmost: a rect over a few dozen of the paths.
    operands.push(BooleanOperand {
        points: rect(130.0, 170.0, 200.0, 150.0),
        fill_rule: FillRule::NonZero,
    });
    let clip = operands.len() - 1;
    let index = operand_index(&operands);
    let window = Some(((120.0, 160.0), (340.0, 330.0)));
    let crop = pieces_of(pathfinder(PathfinderOp::Crop, operands.clone()), "crop");
    // Inside the clip: the topmost operand under it, if any.
    let crop_owner = |c: &[usize]| {
        if c.last() == Some(&clip) {
            c.iter().rev().nth(1).copied()
        } else {
            None
        }
    };
    check_pieces("crop", &index, &crop, window, crop_owner, |a, b| a == b);

    let minus = pieces_of(
        pathfinder(PathfinderOp::MinusBack, operands.clone()),
        "minus back",
    );
    let minus_owner = |c: &[usize]| (c == [clip]).then_some(clip);
    check_pieces("minus back", &index, &minus, window, minus_owner, |a, b| {
        a == b
    });
    assert_eq!(minus.len(), 1);

    let outline = pieces_of(pathfinder(PathfinderOp::Outline, operands), "outline");
    assert!(outline.len() > 210, "outline: {} edge runs", outline.len());
    assert!(outline.windows(2).all(|w| w[0].source <= w[1].source));
}

/// Writes a scene for eyeballing a large union with the real renderer: page 0 is the operands
/// (`MODA_DUMP_N`, default 600, of `MODA_DUMP_ANCHORS`, default 8), page 1 both (the union in outline over
/// the operands), page 2 the union filled. The engine refuses to fill a path past its sweep budget.
///
///   MODA_DUMP_SCENE=/tmp/union.json cargo test -p moda-scene dump_large_union_scene -- --ignored
///   moda-render render --input /tmp/union.json --page all --out-dir /tmp/union
#[test]
#[ignore]
fn dump_large_union_scene() {
    use crate::value::{json, Value};
    let Ok(out) = std::env::var("MODA_DUMP_SCENE") else {
        eprintln!("MODA_DUMP_SCENE not set; skipping");
        return;
    };
    let env = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (n, anchors) = (env("MODA_DUMP_N", 600), env("MODA_DUMP_ANCHORS", 8));
    let operands = grid_operands(n, anchors);
    let Ok(BooleanOutcome::Shape { points, bounds, .. }) =
        boolean(BooleanOp::Union, operands.clone())
    else {
        panic!("union failed");
    };
    let size = (bounds.max_x + 40.0, bounds.max_y + 40.0);
    let solid =
        |color: &str| json!({ "id": "fill-0", "paint": { "type": "solid", "color": color } });
    let fills = [
        "#3a86ff", "#fb5607", "#8338ec", "#06d6a0", "#ef476f", "#118ab2",
    ];
    let path = |id: String, pts: &[PathPoint], extra: Value| {
        let mut node = json!({
            "id": id, "type": "path", "x": 0, "y": 0, "width": size.0, "height": size.1,
            "rotation": 0, "scale": 1, "closed": true,
            "points": pts.iter().map(PathPoint::to_json).collect::<Vec<_>>(),
        });
        for (k, v) in extra.as_object().expect("object") {
            node[k] = v.clone();
        }
        node
    };
    let mut nodes = Vec::new();
    let mut pages = Vec::new();
    for page in 0..3 {
        let mut ids = Vec::new();
        if page != 2 {
            for (i, o) in operands.iter().enumerate() {
                let id = format!("p{page}_{i}");
                let paint = json!({ "fills": [solid(fills[i % fills.len()])], "opacity": 0.6 });
                nodes.push(path(id.clone(), &o.points, paint));
                ids.push(id);
            }
        }
        if page != 0 {
            let paint = if page == 1 {
                let mut stroke = solid("#000000");
                stroke["width"] = json!(1.5);
                json!({ "fills": [], "strokes": [stroke], "fillRule": "evenodd" })
            } else {
                json!({ "fills": [solid("#1d2733")], "fillRule": "evenodd" })
            };
            let id = format!("union{page}");
            nodes.push(path(id.clone(), &points, paint));
            ids.push(id);
        }
        pages.push(json!({
            "id": format!("page-{page}"), "name": format!("page {page}"), "width": size.0,
            "height": size.1, "fills": [solid("#ffffff")], "nodeIds": ids,
        }));
    }
    let scene = json!({
        "nodes": nodes, "pages": pages, "canvasRootNodeIds": [], "layoutMode": "vertical",
        "variables": [], "config": { "renderPageClipping": true },
    });
    std::fs::write(&out, scene.to_string()).expect("write scene");
    eprintln!("wrote {out}: union of {n} has {} points", points.len());
}
