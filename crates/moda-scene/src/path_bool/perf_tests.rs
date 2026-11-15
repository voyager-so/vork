//! What a boolean on N operands would cost without the solver's caps (docs/roadmap/vector-app.md §7:
//! "Boolean on 500 paths < 1 s"): each lane on its own over the whole request, uncapped and unbucketed, where
//! `boolean_paths` buckets the operands and gives the exact lane only a budget (`bucket.rs`). This
//! reads a request the editor built (`node scripts/perf/headless.mjs
//! --dump-boolean <file>`), for docs/qa/perf.md. `#[ignore]`d: run it by hand.
//!
//!   MODA_BOOLEAN_REQUEST=<file> cargo test --release -p moda-scene boolean_lanes_uncapped -- --ignored --nocapture

use std::time::Instant;

use flo_curves::bezier::path::{path_add, SimpleBezierPath};

use super::convert::points_to_subpaths;
use super::overlay;
use super::solve::{normalize, Converted, Region};
use super::*;

#[test]
#[ignore]
fn boolean_lanes_uncapped() {
    let Ok(file) = std::env::var("MODA_BOOLEAN_REQUEST") else {
        eprintln!("MODA_BOOLEAN_REQUEST not set; skipping");
        return;
    };
    let json = std::fs::read_to_string(&file).expect("read request");
    let request = parse_request(&json).expect("parse request");
    let accuracy = request.accuracy.unwrap_or(0.01).clamp(1e-4, 1.0);
    let n = request.operands.len();
    let points: usize = request.operands.iter().map(|o| o.points.len()).sum();

    // Conversion (both lanes start here), translated to the origin as `convert` does.
    let t = Instant::now();
    let mut regions: Vec<Region> = request
        .operands
        .iter()
        .map(|o| points_to_subpaths(&o.points, accuracy))
        .collect();
    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    for (start, segs) in regions.iter().flatten() {
        for p in std::iter::once(start).chain(segs.iter().map(|(_, _, p)| p)) {
            min_x = min_x.min(p.0);
            min_y = min_y.min(p.1);
        }
    }
    for region in regions.iter_mut() {
        for (start, segs) in region.iter_mut() {
            start.0 -= min_x;
            start.1 -= min_y;
            for (a, b, c) in segs.iter_mut() {
                for p in [a, b, c] {
                    p.0 -= min_x;
                    p.1 -= min_y;
                }
            }
        }
    }
    let convert_ms = t.elapsed().as_secs_f64() * 1000.0;
    let cubics: usize = regions.iter().flatten().map(|(_, s)| s.len()).sum();

    // Polygon lane (i_overlay): flatten, resolve each operand under its fill rule, fold the op, refit.
    let fills: Vec<FillRule> = request.operands.iter().map(|o| o.fill_rule).collect();
    let t = Instant::now();
    let flat = overlay::flatten(&regions).expect("flatten");
    let expected = overlay::boolean(request.op, &overlay::resolve_all(&flat, &fills));
    let polygon_solve_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let refit = flat.refit(&expected);
    let refit_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Exact lane (flo_curves), union only: normalize each operand, then fold `path_add` without the
    // pairwise-work guard. Stops early past 30 s so a pathological run still reports.
    let exact = if request.op == BooleanOp::Union {
        let t = Instant::now();
        let prepared = normalize(
            Converted {
                regions: regions.clone(),
                offset: (0.0, 0.0),
                accuracy,
            },
            &fills,
        )
        .expect("normalize");
        let normalize_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let mut acc: Option<Region> = None;
        let mut folded = 0usize;
        for region in prepared.regions.iter().filter(|r| !r.is_empty()) {
            acc = Some(match acc {
                None => region.clone(),
                Some(acc) => path_add::<SimpleBezierPath>(&acc, region, accuracy),
            });
            folded += 1;
            if t.elapsed().as_secs() >= 30 {
                break;
            }
        }
        Some((normalize_ms, t.elapsed().as_secs_f64() * 1000.0, folded))
    } else {
        None
    };

    println!(
        "{}",
        crate::value::json!({
            "op": format!("{:?}", request.op),
            "operands": n,
            "points": points,
            "cubics": cubics,
            "convertMs": convert_ms,
            "polygon": { "solveMs": polygon_solve_ms, "refitMs": refit_ms, "rings": expected.len(), "resultSubpaths": refit.len() },
            "exactUnion": exact.map(|(norm, fold, folded)| crate::value::json!({ "normalizeMs": norm, "foldMs": fold, "operandsFolded": folded })),
        })
    );
}
