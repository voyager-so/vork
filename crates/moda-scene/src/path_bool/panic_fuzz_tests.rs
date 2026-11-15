//! Panic fuzz over random *curved* shapes, for every boolean and pathfinder op.
//!
//! `prop_tests.rs` fuzzes the editor's primitive vocabulary (rects, ellipses,
//! rounded rects), which the solver handles. This module feeds it what people
//! actually draw with the Pen, Pencil and Blob Brush: closed paths of random
//! cubics (self-intersecting more often than not), smooth or spiky star-shaped
//! blobs, Blob Brush stroke outlines, and a shape against a near copy of itself.
//!
//! Seeded xorshift, so every run sees the same cases. A panic is observed as a
//! `solver_failed` envelope: on native the solver runs under `catch_unwind`,
//! and on wasm (`panic = "abort"`) the same panic would take down the engine.
//!
//! `measure_panic_rates` is `#[ignore]`d and prints the per-op table, with the
//! places the panics came from.

use std::f64::consts::PI;
use std::time::{Duration, Instant};

use super::*;

pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub(super) fn f(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }
    pub(super) fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// A closed path of 3–8 anchors in a `size` box around (cx, cy); most segments
/// are cubics with random handles, so the outline usually crosses itself.
pub(super) fn random_curved(rng: &mut Rng, cx: f64, cy: f64, size: f64) -> Vec<PathPoint> {
    let n = 3 + rng.below(6) as usize;
    let pt = |rng: &mut Rng| (cx + rng.f(-size, size) / 2.0, cy + rng.f(-size, size) / 2.0);
    let (x0, y0) = pt(rng);
    let mut out = vec![PathPoint::Move { x: x0, y: y0 }];
    for i in 0..n {
        let (x, y) = if i + 1 == n { (x0, y0) } else { pt(rng) };
        if rng.f(0.0, 1.0) < 0.7 {
            let cp1 = pt(rng);
            let cp2 = pt(rng);
            out.push(PathPoint::Curve { x, y, cp1, cp2 });
        } else {
            out.push(PathPoint::Line { x, y });
        }
    }
    out
}

/// A star-shaped blob of 5–16 lobes around (cx, cy): smooth Catmull-Rom-style
/// cubics, or a spiky polygon with alternating radii.
pub(super) fn blob(rng: &mut Rng, cx: f64, cy: f64, r: f64, spiky: bool) -> Vec<PathPoint> {
    let n = 5 + rng.below(12) as usize;
    let rs: Vec<f64> = (0..n)
        .map(|i| {
            if spiky && i % 2 == 1 {
                r * rng.f(0.3, 0.55)
            } else {
                r * rng.f(0.65, 1.0)
            }
        })
        .collect();
    let rot = rng.f(0.0, 2.0 * PI);
    let at = |i: usize| {
        let a = rot + (i % n) as f64 / n as f64 * 2.0 * PI;
        (cx + rs[i % n] * a.cos(), cy + rs[i % n] * a.sin())
    };
    let (x0, y0) = at(0);
    let mut out = vec![PathPoint::Move { x: x0, y: y0 }];
    for i in 1..=n {
        let p0 = at(i - 1);
        let p1 = at(i);
        if spiky {
            out.push(PathPoint::Line { x: p1.0, y: p1.1 });
        } else {
            let prev = at(i + n - 2);
            let next = at(i + 1);
            let t0 = ((p1.0 - prev.0) * 0.2, (p1.1 - prev.1) * 0.2);
            let t1 = ((next.0 - p0.0) * 0.2, (next.1 - p0.1) * 0.2);
            out.push(PathPoint::Curve {
                x: p1.0,
                y: p1.1,
                cp1: (p0.0 + t0.0, p0.1 + t0.1),
                cp2: (p1.0 - t1.0, p1.1 - t1.1),
            });
        }
    }
    out
}

/// Which family a case draws its operands from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Family {
    RandomCurved,
    Blob,
    /// Blob Brush strokes: a wobbly open curve outlined by the stroke
    /// outliner (round caps and joins), exactly the operand `blobBrush.ts`
    /// hands the solver. Wide strokes on tight curves give the outline cusps
    /// and self-overlapping loops.
    BrushStroke,
    /// A random curved shape against a near copy of itself (identical, nudged
    /// by a hair, or turned by a fraction of a degree): the duplicate-and-
    /// combine case, all coincident or near-coincident edges.
    NearCopy,
}

pub(super) const FAMILIES: [Family; 4] = [
    Family::RandomCurved,
    Family::Blob,
    Family::BrushStroke,
    Family::NearCopy,
];

/// A Pencil-like open stroke: a random walk of 4-10 cubics.
fn wobbly_centerline(rng: &mut Rng, cx: f64, cy: f64) -> Vec<PathPoint> {
    let n = 4 + rng.below(7) as usize;
    let (mut x, mut y) = (cx + rng.f(-30.0, 30.0), cy + rng.f(-30.0, 30.0));
    let mut heading = rng.f(0.0, 2.0 * PI);
    let mut out = vec![PathPoint::Move { x, y }];
    for _ in 0..n {
        heading += rng.f(-2.0, 2.0);
        let len = rng.f(8.0, 40.0);
        let (nx, ny) = (x + len * heading.cos(), y + len * heading.sin());
        let bend = rng.f(-1.2, 1.2);
        let cp1 = (
            x + len / 3.0 * (heading - bend).cos(),
            y + len / 3.0 * (heading - bend).sin(),
        );
        let cp2 = (
            nx - len / 3.0 * (heading + bend).cos(),
            ny - len / 3.0 * (heading + bend).sin(),
        );
        out.push(PathPoint::Curve {
            x: nx,
            y: ny,
            cp1,
            cp2,
        });
        (x, y) = (nx, ny);
    }
    out
}

/// The stroke outliner's closed outline of a wobbly stroke, as Blob Brush
/// requests it. `None` when the outliner declines (it has its own fuzz).
fn brush_stroke(rng: &mut Rng, cx: f64, cy: f64) -> Option<Vec<PathPoint>> {
    let centerline = wobbly_centerline(rng, cx, cy);
    let width = rng.f(4.0, 60.0);
    let request = crate::value::json!({
        "points": centerline.iter().map(PathPoint::to_json).collect::<Vec<_>>(),
        "closed": false,
        "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": width,
                     "cap": "round", "join": "round"}],
        "tolerance": 0.25,
    });
    let reply: crate::value::Value = crate::value::from_str(
        &crate::stroke_outline::stroke_outline_json(&request.to_string()),
    )
    .ok()?;
    if reply["status"] != "ok" {
        return None;
    }
    reply["points"]
        .as_array()?
        .iter()
        .map(PathPoint::from_json)
        .collect()
}

fn near_copy(rng: &mut Rng, points: &[PathPoint]) -> Vec<PathPoint> {
    let (e, f, r) = match rng.below(3) {
        0 => (0.0, 0.0, 0.0),
        1 => (rng.f(-1e-3, 1e-3), rng.f(-1e-3, 1e-3), 0.0),
        _ => (0.0, 0.0, rng.f(-0.5, 0.5).to_radians()),
    };
    let m = crate::types::Affine {
        a: r.cos(),
        b: r.sin(),
        c: -r.sin(),
        d: r.cos(),
        e,
        f,
    };
    points.iter().map(|p| p.apply_affine(m)).collect()
}

/// `count` overlapping operands of one family.
pub(super) fn case_operands(rng: &mut Rng, family: Family, count: usize) -> Vec<BooleanOperand> {
    if family == Family::NearCopy {
        let size = rng.f(40.0, 100.0);
        let base = random_curved(rng, 0.0, 0.0, size);
        return (0..count)
            .map(|i| BooleanOperand {
                points: if i == 0 {
                    base.clone()
                } else {
                    near_copy(rng, &base)
                },
                fill_rule: FillRule::NonZero,
            })
            .collect();
    }
    (0..count)
        .map(|_| {
            let cx = rng.f(-15.0, 15.0);
            let cy = rng.f(-15.0, 15.0);
            let points = match family {
                Family::BrushStroke => (0..100)
                    // The outliner answers every stroke it is given in practice;
                    // the retries only guard the fixture against an `empty` (and
                    // a request the outliner stops accepting fails loudly instead
                    // of spinning forever).
                    .find_map(|_| brush_stroke(rng, cx, cy))
                    .expect("the stroke outliner declined 100 Blob Brush strokes in a row"),
                Family::NearCopy => unreachable!("built above"),
                Family::RandomCurved => {
                    let size = rng.f(40.0, 100.0);
                    random_curved(rng, cx, cy, size)
                }
                Family::Blob => {
                    let spiky = rng.below(3) == 0;
                    let r = rng.f(25.0, 50.0);
                    blob(rng, cx, cy, r, spiky)
                }
            };
            let fill_rule = if rng.below(2) == 0 {
                FillRule::NonZero
            } else {
                FillRule::EvenOdd
            };
            BooleanOperand { points, fill_rule }
        })
        .collect()
}

pub(super) const BOOLEAN_OPS: [BooleanOp; 4] = [
    BooleanOp::Union,
    BooleanOp::Subtract,
    BooleanOp::Intersect,
    BooleanOp::Exclude,
];

pub(super) const PATHFINDER_OPS: [PathfinderOp; 6] = [
    PathfinderOp::Divide,
    PathfinderOp::Trim,
    PathfinderOp::Merge,
    PathfinderOp::Crop,
    PathfinderOp::Outline,
    PathfinderOp::MinusBack,
];

#[derive(Default, Debug)]
pub(super) struct Tally {
    pub(super) runs: usize,
    pub(super) panics: usize,
    /// Any other typed error (a budget refusal, say): not a crash.
    pub(super) other_errors: usize,
    pub(super) time: Duration,
    pub(super) first_panic: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Ops {
    All,
    Boolean,
    Pathfinder,
}

/// Every op (of `ops`) on one case: `(op label, error, time)` per op. Booleans run on
/// two operands, Pathfinder on three (divide / trim / merge see more
/// interesting arrangements with a third shape).
pub(super) fn run_case(
    family: Family,
    case_seed: u64,
    ops: Ops,
) -> Vec<(String, Option<BooleanError>, Duration)> {
    let mut rng = Rng::new(case_seed);
    let two = case_operands(&mut rng, family, 2);
    let three = case_operands(&mut rng, family, 3);
    let mut out = Vec::new();
    for op in BOOLEAN_OPS.into_iter().filter(|_| ops != Ops::Pathfinder) {
        let start = Instant::now();
        let outcome = boolean_paths(&BooleanRequest {
            op,
            operands: two.clone(),
            accuracy: None,
        });
        out.push((format!("{op:?}"), outcome.err(), start.elapsed()));
    }
    for op in PATHFINDER_OPS.into_iter().filter(|_| ops != Ops::Boolean) {
        let start = Instant::now();
        let outcome = pathfinder_paths(&PathfinderRequest {
            op,
            operands: three.clone(),
            paint: vec![
                OperandPaint {
                    painted: true,
                    key: Some("a".into()),
                };
                3
            ],
            accuracy: None,
        });
        out.push((format!("{op:?}"), outcome.err(), start.elapsed()));
    }
    out
}

pub(super) fn case_seed(seed: u64, case: u64) -> u64 {
    seed ^ case.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Run every op over `cases` cases per family; returns one tally per
/// (family, op label).
pub(super) fn sweep(seed: u64, cases: usize) -> Vec<(Family, String, Tally)> {
    let mut rows: Vec<(Family, String, Tally)> = Vec::new();
    for family in FAMILIES {
        for case in 0..cases as u64 {
            let case_seed = case_seed(seed, case);
            for (label, error, time) in run_case(family, case_seed, Ops::All) {
                let at = match rows
                    .iter()
                    .position(|(f, l, _)| *f == family && *l == label)
                {
                    Some(at) => at,
                    None => {
                        rows.push((family, label, Tally::default()));
                        rows.len() - 1
                    }
                };
                let row = &mut rows[at].2;
                row.time += time;
                record(row, error, case_seed);
            }
        }
    }
    rows
}

fn record(t: &mut Tally, error: Option<BooleanError>, case_seed: u64) {
    t.runs += 1;
    match error {
        Some(e) if e.code == "solver_failed" => {
            t.panics += 1;
            t.first_panic.get_or_insert(case_seed);
        }
        Some(_) => t.other_errors += 1,
        None => {}
    }
}

/// Replace the default hook while `f` runs: instead of printing every caught
/// panic, count them by site (the panicking location and the first flo_curves
/// frame above it) and print the counts at the end. Only used by the ignored
/// measurement test, which is run on its own (the hook is process-wide).
fn quietly<R>(f: impl FnOnce() -> R) -> R {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|info| {
        let at = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_default();
        let bt = std::backtrace::Backtrace::force_capture().to_string();
        // Frames read "  12: flo_curves::…"; drop the frame number.
        let caller = bt
            .lines()
            .find(|l| l.contains("flo_curves"))
            .map(|l| l.trim().split_once(": ").map_or(l, |(_, f)| f))
            .unwrap_or("(no flo_curves frame)");
        let at = format!("{at} <- {caller}");
        *PANIC_SITES
            .lock()
            .unwrap()
            .get_or_insert_with(Default::default)
            .entry(at)
            .or_insert(0) += 1;
    }));
    let r = f();
    std::panic::set_hook(hook);
    if let Some(sites) = PANIC_SITES.lock().unwrap().take() {
        for (at, n) in sites {
            println!("panic site {at}: {n}");
        }
    }
    r
}

static PANIC_SITES: std::sync::Mutex<Option<std::collections::HashMap<String, usize>>> =
    std::sync::Mutex::new(None);

pub(super) fn print_table(rows: &[(Family, String, Tally)]) {
    println!(
        "| family | op | runs | panics | rate | other errors | mean ms | first panicking seed |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    for (family, label, t) in rows {
        println!(
            "| {family:?} | {label} | {} | {} | {:.1}% | {} | {:.2} | {} |",
            t.runs,
            t.panics,
            100.0 * t.panics as f64 / t.runs.max(1) as f64,
            t.other_errors,
            t.time.as_secs_f64() * 1000.0 / t.runs.max(1) as f64,
            t.first_panic.map(|s| format!("{s:#x}")).unwrap_or_default(),
        );
    }
    let runs: usize = rows.iter().map(|r| r.2.runs).sum();
    let panics: usize = rows.iter().map(|r| r.2.panics).sum();
    println!(
        "total: {panics} / {runs} runs panicked ({:.1}%)",
        100.0 * panics as f64 / runs.max(1) as f64
    );
}

const SWEEP_SEED: u64 = 0x5eed_b001_0000_0001;

/// `cargo test -p moda-scene --release measure_panic_rates -- --ignored --nocapture`
#[test]
#[ignore = "measurement; prints the per-op panic table"]
fn measure_panic_rates() {
    let rows = quietly(|| sweep(SWEEP_SEED, 200));
    print_table(&rows);
    let panics: usize = rows.iter().map(|r| r.2.panics).sum();
    assert_eq!(panics, 0, "{panics} runs panicked");
}

/// Always on: the first case of the sweep that panicked for each op before
/// flo_curves' collision sort was patched (`native/vendor/flo_curves/
/// MODA-PATCH.md`). A panic here is an engine abort on wasm. The full sweep
/// (`measure_panic_rates`) is too slow for a debug test run.
#[test]
fn curved_shapes_never_panic() {
    let cases: [(Family, u64, Ops); 6] = [
        // Blob Brush outlines: every boolean op.
        (Family::BrushStroke, 0xc0da_c9b8_7f4a_7c14, Ops::Boolean),
        // Near copies: booleans; divide / outline; trim; merge; minus back.
        (Family::NearCopy, 0xca77_73a4_a267_fad5, Ops::Boolean),
        (Family::NearCopy, 0xb37f_ab81_ac55_31af, Ops::Pathfinder),
        (Family::NearCopy, 0x259d_39e1_7ef4_36ef, Ops::Pathfinder),
        (Family::NearCopy, 0x24ca_8862_3e0f_13a0, Ops::Pathfinder),
        (Family::NearCopy, 0x912d_d59a_6017_cfb0, Ops::Pathfinder),
    ];
    let mut failures = Vec::new();
    for (family, seed, ops) in cases {
        for (label, error, _) in run_case(family, seed, ops) {
            if let Some(e) = error.filter(|e| e.code == "solver_failed") {
                failures.push(format!("{family:?} {seed:#x} {label}: {}", e.message));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
