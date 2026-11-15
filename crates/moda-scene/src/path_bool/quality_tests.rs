//! Result quality of the solver's lanes against a high-resolution raster.
//!
//! For each case from the panic fuzz's shape families (`panic_fuzz_tests.rs`),
//! the truth is sampled at pixel centres on a grid over the operands' bounds
//! (800×800 for the measurement). Each operand's inside test uses its own fill
//! rule over a polyline flattened to 1e-4, and the op is applied per sample.
//! Each lane's pieces are sampled the same way, even-odd, as the envelope says.
//! A sample counts as wrong if it is outside the truth but covered, inside the
//! truth but uncovered, or covered by two pieces at once.
//!
//! The wrong area divided by the operands' total perimeter is the mean distance
//! between the answer's boundary and the exact one, in path units: the number
//! to compare with a tolerance. An answer is *wrong* (a defect, not an
//! approximation) when some wrong sample lies `GROSS` samples or more from
//! every operand's edge: a piece at least a few samples across is lost, extra
//! or doubled. Flattening and refit errors are a small fraction of a sample.
//!
//! `measure_lane_quality` is `#[ignore]`d and prints the table for the PR.
//! `checked_lane_fixes_the_exact_lanes_wrong_answers` pins the cases where the
//! exact lane alone was wrong.

use std::time::{Duration, Instant};

use super::convert::points_to_subpaths;
use super::panic_fuzz_tests::{case_operands, case_seed, Rng, BOOLEAN_OPS, FAMILIES};
use super::solve::{solve_on, Lane};
use super::*;

type Poly = Vec<(f64, f64)>;

/// A wrong sample this many samples (or more) from every operand edge makes
/// the answer wrong.
const GROSS: usize = 2;

const LANES: [Lane; 3] = [Lane::Exact, Lane::Polygon, Lane::Checked];

/// The Pathfinder ops whose pieces are closed and cover a region (outline's
/// open runs have no area; its faces are divide's).
const AREA_PATHFINDER_OPS: [PathfinderOp; 5] = [
    PathfinderOp::Divide,
    PathfinderOp::Trim,
    PathfinderOp::Merge,
    PathfinderOp::Crop,
    PathfinderOp::MinusBack,
];

fn flatten_fine(points: &[PathPoint]) -> Vec<Poly> {
    points_to_subpaths(points, 1e-9)
        .iter()
        .map(|(start, segs)| {
            let mut out = vec![(start.0, start.1)];
            let mut p0 = *start;
            for (c1, c2, p3) in segs {
                let l = (p0.0 - 2.0 * c1.0 + c2.0)
                    .hypot(p0.1 - 2.0 * c1.1 + c2.1)
                    .max((c1.0 - 2.0 * c2.0 + p3.0).hypot(c1.1 - 2.0 * c2.1 + p3.1));
                let n = ((0.75 * l / 1e-4).sqrt().ceil() as usize).clamp(1, 4096);
                for i in 1..=n {
                    let t = i as f64 / n as f64;
                    let mt = 1.0 - t;
                    let (a, b, c, d) =
                        (mt * mt * mt, 3.0 * mt * mt * t, 3.0 * mt * t * t, t * t * t);
                    out.push((
                        a * p0.0 + b * c1.0 + c * c2.0 + d * p3.0,
                        a * p0.1 + b * c1.1 + c * c2.1 + d * p3.1,
                    ));
                }
                p0 = *p3;
            }
            out
        })
        .collect()
}

fn perimeter(polys: &[Poly]) -> f64 {
    polys
        .iter()
        .map(|p| {
            (0..p.len())
                .map(|i| {
                    let (a, b) = (p[i], p[(i + 1) % p.len()]);
                    (b.0 - a.0).hypot(b.1 - a.1)
                })
                .sum::<f64>()
        })
        .sum()
}

fn inside(w: i32, rule: FillRule) -> bool {
    match rule {
        FillRule::NonZero => w != 0,
        FillRule::EvenOdd => w % 2 != 0,
    }
}

/// A sampling grid over some operands, with each operand's inside mask.
struct Case {
    x0: f64,
    y0: f64,
    px: f64,
    n: usize,
    masks: Vec<Vec<bool>>,
    /// Samples within `GROSS` samples of some operand's edge.
    near_edge: Vec<bool>,
    perimeter: f64,
}

impl Case {
    fn new(operands: &[BooleanOperand], n: usize) -> Case {
        let inputs: Vec<Vec<Poly>> = operands.iter().map(|o| flatten_fine(&o.points)).collect();
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for p in inputs.iter().flatten().flatten() {
            x0 = x0.min(p.0);
            y0 = y0.min(p.1);
            x1 = x1.max(p.0);
            y1 = y1.max(p.1);
        }
        let px = (x1 - x0).max(y1 - y0) / n as f64 * 1.01;
        let mut case = Case {
            x0: x0 - px,
            y0: y0 - px,
            px,
            n: n + 2,
            masks: Vec::new(),
            near_edge: Vec::new(),
            perimeter: inputs.iter().map(|p| perimeter(p)).sum(),
        };
        case.masks = inputs
            .iter()
            .zip(operands)
            .map(|(polys, o)| case.mask(polys, o.fill_rule))
            .collect();
        case.near_edge = case.near_edges(inputs.iter().flatten());
        case
    }

    /// Mark every sample within `GROSS` samples of an edge, walking each edge
    /// in half-sample steps. Exact, unlike reading it off the masks, which miss
    /// features thinner than a sample.
    fn near_edges<'a>(&self, polys: impl Iterator<Item = &'a Poly>) -> Vec<bool> {
        let n = self.n as i64;
        let reach = GROSS as i64 + 1;
        let mut out = vec![false; self.n * self.n];
        for poly in polys {
            for i in 0..poly.len() {
                let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
                let steps = ((b.0 - a.0).hypot(b.1 - a.1) / (self.px / 2.0))
                    .ceil()
                    .max(1.0) as usize;
                for k in 0..=steps {
                    let t = k as f64 / steps as f64;
                    let x = ((a.0 + (b.0 - a.0) * t - self.x0) / self.px).floor() as i64;
                    let y = ((a.1 + (b.1 - a.1) * t - self.y0) / self.px).floor() as i64;
                    for yy in (y - reach).max(0)..=(y + reach).min(n - 1) {
                        for xx in (x - reach).max(0)..=(x + reach).min(n - 1) {
                            out[(yy * n + xx) as usize] = true;
                        }
                    }
                }
            }
        }
        out
    }

    /// Inside mask of `polys` under `rule`, by scanline winding.
    fn mask(&self, polys: &[Poly], rule: FillRule) -> Vec<bool> {
        let mut out = vec![false; self.n * self.n];
        let mut crossings: Vec<(f64, i32)> = Vec::new();
        for j in 0..self.n {
            let y = self.y0 + (j as f64 + 0.5) * self.px;
            crossings.clear();
            for poly in polys {
                for i in 0..poly.len() {
                    let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
                    if (a.1 <= y) != (b.1 <= y) {
                        let x = a.0 + (y - a.1) / (b.1 - a.1) * (b.0 - a.0);
                        crossings.push((x, if b.1 > a.1 { 1 } else { -1 }));
                    }
                }
            }
            crossings.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (mut w, mut k) = (0, 0);
            for i in 0..self.n {
                let x = self.x0 + (i as f64 + 0.5) * self.px;
                while k < crossings.len() && crossings[k].0 < x {
                    w += crossings[k].1;
                    k += 1;
                }
                out[j * self.n + i] = inside(w, rule);
            }
        }
        out
    }

    fn truth(&self, f: impl Fn(&[bool]) -> bool) -> Vec<bool> {
        let mut at = vec![false; self.masks.len()];
        (0..self.n * self.n)
            .map(|i| {
                for (k, m) in self.masks.iter().enumerate() {
                    at[k] = m[i];
                }
                f(&at)
            })
            .collect()
    }

    /// `(mean boundary distance, gross samples)` of `pieces` (each even-odd)
    /// against `truth`: see the module doc.
    fn score(&self, truth: &[bool], pieces: &[Vec<PathPoint>]) -> (f64, usize) {
        let mut cover = vec![0u8; self.n * self.n];
        for piece in pieces {
            for (c, m) in cover
                .iter_mut()
                .zip(self.mask(&flatten_fine(piece), FillRule::EvenOdd))
            {
                *c = c.saturating_add(m as u8);
            }
        }
        let n = self.n;
        let wrong: Vec<usize> = (0..n * n)
            .filter(|&i| cover[i] > 1 || truth[i] != (cover[i] == 1))
            .collect();
        // Far from every operand's edge, not just the truth's: pieces meet
        // along operand edges inside the region, and a seam there is not gross.
        let gross = wrong.iter().filter(|&&i| !self.near_edge[i]).count();
        (
            wrong.len() as f64 * self.px * self.px / self.perimeter.max(1e-9),
            gross,
        )
    }
}

fn boolean_truth(op: BooleanOp) -> impl Fn(&[bool]) -> bool {
    move |m: &[bool]| match op {
        BooleanOp::Union => m.iter().any(|x| *x),
        BooleanOp::Intersect => m.iter().all(|x| *x),
        BooleanOp::Subtract => m[0] && !m[1..].iter().any(|x| *x),
        BooleanOp::Exclude => m.iter().filter(|x| **x).count() % 2 == 1,
    }
}

/// Every operand filled.
fn pathfinder_truth(op: PathfinderOp) -> impl Fn(&[bool]) -> bool {
    move |m: &[bool]| {
        let top = m.len() - 1;
        match op {
            PathfinderOp::Divide
            | PathfinderOp::Outline
            | PathfinderOp::Trim
            | PathfinderOp::Merge => m.iter().any(|x| *x),
            PathfinderOp::Crop => m[top] && m[..top].iter().any(|x| *x),
            PathfinderOp::MinusBack => m[top] && !m[..top].iter().any(|x| *x),
        }
    }
}

fn filled(operands: &[BooleanOperand]) -> PathfinderRequest {
    PathfinderRequest {
        op: PathfinderOp::Divide,
        operands: operands.to_vec(),
        paint: operands
            .iter()
            .map(|_| OperandPaint {
                painted: true,
                key: Some("a".into()),
            })
            .collect(),
        accuracy: None,
    }
}

/// `((distance, gross samples), points, time, the answer as JSON)` for one
/// lane, or `None` if it errored.
type Scored = Option<((f64, usize), usize, Duration, String)>;

fn scored(case: &Case, truth: &[bool], pieces: Vec<Vec<PathPoint>>, time: Duration) -> Scored {
    let points = pieces.iter().map(Vec::len).sum();
    let json = pieces
        .iter()
        .flatten()
        .map(|p| p.to_json().to_string())
        .collect();
    Some((case.score(truth, &pieces), points, time, json))
}

fn score_boolean(case: &Case, op: BooleanOp, operands: &[BooleanOperand], lane: Lane) -> Scored {
    let request = BooleanRequest {
        op,
        operands: operands.to_vec(),
        accuracy: None,
    };
    let t = Instant::now();
    let outcome = solve_on(&request, lane);
    let time = t.elapsed();
    let pieces = match outcome {
        Ok(BooleanOutcome::Shape { points, .. }) => vec![points],
        Ok(BooleanOutcome::Empty) => Vec::new(),
        Err(_) => return None,
    };
    scored(case, &case.truth(boolean_truth(op)), pieces, time)
}

fn score_pathfinder(
    case: &Case,
    op: PathfinderOp,
    operands: &[BooleanOperand],
    lane: Lane,
) -> Scored {
    let request = PathfinderRequest {
        op,
        ..filled(operands)
    };
    let t = Instant::now();
    let outcome = super::pathfinder::solve_on(&request, lane);
    let time = t.elapsed();
    let pieces: Vec<Vec<PathPoint>> = match outcome {
        Ok(PathfinderOutcome::Pieces(p)) => p.into_iter().map(|p| p.points).collect(),
        Ok(PathfinderOutcome::Empty) => Vec::new(),
        Err(_) => return None,
    };
    scored(case, &case.truth(pathfinder_truth(op)), pieces, time)
}

#[derive(Default)]
struct LaneStats {
    runs: usize,
    errors: usize,
    wrong: Vec<String>,
    /// Mean boundary distances of the answers that are not wrong.
    good: Vec<f64>,
    points: usize,
    time: Duration,
}

impl LaneStats {
    fn add(&mut self, scored: &Scored, what: impl FnOnce() -> String) {
        self.runs += 1;
        let Some(((dist, gross), points, time, _)) = scored.clone() else {
            self.errors += 1;
            return;
        };
        if gross > 0 {
            self.wrong.push(what());
        } else {
            self.good.push(dist);
        }
        self.points += points;
        self.time += time;
    }

    fn row(&mut self, fallbacks: Option<usize>) -> String {
        let n = (self.runs - self.errors).max(1) as f64;
        self.good.sort_by(f64::total_cmp);
        let at = |q: f64| {
            self.good
                .get(((self.good.len().max(1) - 1) as f64 * q).round() as usize)
                .copied()
                .unwrap_or(0.0)
        };
        format!(
            "{} | {} | {} | {} | {:.4} | {:.4} | {:.4} | {:.1} | {:.2}",
            self.runs,
            fallbacks.map_or("".to_string(), |f| f.to_string()),
            self.errors,
            self.wrong.len(),
            at(0.5),
            at(0.95),
            self.good.last().copied().unwrap_or(0.0),
            self.points as f64 / n,
            self.time.as_secs_f64() * 1000.0 / n
        )
    }
}

const QUALITY_SEED: u64 = 0x9a11_7e57_0000_0001;

/// `cargo test -p moda-scene --release --lib measure_lane_quality -- --ignored --nocapture`
#[test]
#[ignore = "measurement; prints the lane quality table"]
fn measure_lane_quality() {
    println!(
        "| family | ops | lane | runs | polygon answers | errors | wrong | median dist | \
         p95 dist | max dist (not wrong) | mean points | mean ms |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut wrong_seeds = Vec::new();
    for family in FAMILIES {
        let mut stats: Vec<[LaneStats; 3]> = vec![Default::default(), Default::default()];
        // Checked answers that are not the exact lane's (so the polygon lane's).
        let mut fallbacks = [0usize; 2];
        for case in 0..50u64 {
            let seed = case_seed(QUALITY_SEED, case);
            let mut rng = Rng::new(seed);
            let two = case_operands(&mut rng, family, 2);
            let three = case_operands(&mut rng, family, 3);
            let (c2, c3) = (Case::new(&two, 800), Case::new(&three, 800));
            let mut record = |kind: usize, op: String, by_lane: [Scored; 3]| {
                for (l, s) in by_lane.iter().enumerate() {
                    stats[kind][l].add(s, || format!("{family:?} {seed:#x} {op}"));
                }
                let json = |s: &Scored| s.as_ref().map(|s| s.3.clone());
                if json(&by_lane[2]) != json(&by_lane[0]) {
                    fallbacks[kind] += 1;
                }
            };
            for op in BOOLEAN_OPS {
                let by_lane = LANES.map(|lane| score_boolean(&c2, op, &two, lane));
                record(0, format!("{op:?}"), by_lane);
            }
            for op in AREA_PATHFINDER_OPS {
                let by_lane = LANES.map(|lane| score_pathfinder(&c3, op, &three, lane));
                record(1, format!("{op:?}"), by_lane);
            }
        }
        for (k, (kind, lanes)) in ["boolean", "pathfinder"]
            .iter()
            .zip(stats.iter_mut())
            .enumerate()
        {
            for (lane, s) in LANES.iter().zip(lanes.iter_mut()) {
                let fallbacks = (*lane == Lane::Checked).then_some(fallbacks[k]);
                println!("| {family:?} | {kind} | {lane:?} | {} |", s.row(fallbacks));
                wrong_seeds.extend(s.wrong.iter().map(|w| format!("{lane:?}: {w}")));
            }
        }
    }
    for w in &wrong_seeds {
        println!("wrong: {w}");
    }
    assert!(
        !wrong_seeds.iter().any(|w| w.starts_with("Checked")),
        "the shipped lane gave a wrong answer"
    );
}

/// Always on: cases from the sweep where the exact lane alone came back wrong
/// (a lost or extra piece), and the shipped lane must not. The exact lane is
/// asserted wrong too, so a fixture that stops exercising the check is noticed.
#[test]
fn checked_lane_fixes_the_exact_lanes_wrong_answers() {
    use super::panic_fuzz_tests::Family;
    enum Op {
        Boolean(BooleanOp),
        Pathfinder(PathfinderOp),
    }
    let cases = [
        (
            Family::RandomCurved,
            0x22ba_7aac_e3a3_6349,
            Op::Boolean(BooleanOp::Exclude),
        ),
        // Stroke outlines (the sweep's Blob Brush union seed stopped failing when the outliner's fit changed).
        (
            Family::BrushStroke,
            0xdf51_5f89_755d_453a,
            Op::Pathfinder(PathfinderOp::Merge),
        ),
        (
            Family::NearCopy,
            0xf7e6_a103_df62_4bc7,
            Op::Boolean(BooleanOp::Intersect),
        ),
        (
            Family::RandomCurved,
            0x8d04_1ec8_7c74_6c68,
            Op::Pathfinder(PathfinderOp::Trim),
        ),
    ];
    for (family, seed, op) in cases {
        let mut rng = Rng::new(seed);
        let two = case_operands(&mut rng, family, 2);
        let three = case_operands(&mut rng, family, 3);
        let gross = |lane: Lane| {
            match &op {
                Op::Boolean(op) => score_boolean(&Case::new(&two, 400), *op, &two, lane),
                Op::Pathfinder(op) => score_pathfinder(&Case::new(&three, 400), *op, &three, lane),
            }
            .map(|((_, gross), ..)| gross)
        };
        let what = format!("{family:?} {seed:#x}");
        assert!(
            gross(Lane::Exact).unwrap() > 0,
            "{what}: the exact lane is right now"
        );
        assert_eq!(
            gross(Lane::Checked),
            Some(0),
            "{what}: the shipped answer is wrong"
        );
    }
}
