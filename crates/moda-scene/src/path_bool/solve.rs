//! Normalization + the flo_curves calls themselves.
//!
//! flo_curves has no winding rule: it treats every edge of an input path as an
//! exterior edge, which is *almost* even-odd. So each operand is pre-normalized
//! to a non-self-overlapping set of rings first, under its own fill rule, and
//! that set is what the ops below see.
//!
//! The two rules are the ones the renderers implement, not approximations of
//! them:
//!
//! * **non-zero** — a point is filled where the *signed* winding number is
//!   non-zero, exactly like lyon's `FillRule::NonZero` in
//!   `moda-engine/src/tess/tessellator.rs`, which fills one operand's whole
//!   point list as a single event stream. Direction therefore carries meaning:
//!   an inner loop wound against its outer is a hole. That is the repo's
//!   standard hole encoding — `lib/figmaClipboard/convert/vector.ts` relies on
//!   it for Figma paste, and `core/svg/importSVG.ts` carries SVG's default
//!   `nonzero` straight onto the node — so a solver that ignored direction
//!   would answer a question about a different shape than the one on screen.
//! * **even-odd** — parity, which is direction-independent by definition and is
//!   `path_remove_overlapped_points` unchanged.
//!
//! `path_remove_interior_points` cannot serve the non-zero lane: it is
//! documented as "a strict version of the non-zero rule" and keeps only the
//! outermost perimeter, and `GraphPath::from_path` reverses every
//! anti-clockwise subpath to clockwise on construction, so by the time its ray
//! caster runs there is no direction left to read. `normalize_nonzero` below
//! rebuilds the signs the graph discarded.

use flo_curves::bezier::path::{
    path_add, path_full_intersect, path_intersect, path_remove_overlapped_points, path_sub,
    GraphPath, PathLabel, SimpleBezierPath,
};
use flo_curves::geo::Coord2;

use super::bucket::{self, ExactBudget, Hull};
use super::convert::{points_to_subpaths, subpath_signed_area, subpath_to_points};
use super::overlay::{self, Contours};
use super::{
    BooleanError, BooleanOp, BooleanOperand, BooleanOutcome, BooleanRequest, Bounds, FillRule,
    PathPoint,
};

/// Hard caps on the request. Everything above them fails closed with a typed
/// error rather than handing either lane a workload that would wedge the shared
/// wasm instance. flo_curves (the exact lane) has its own, smaller allowance
/// per request (`bucket::ExactBudget`: 64 operands and 8,000 cubics, the caps
/// the whole request used to have); past it the polygon lane answers alone, so
/// these caps bound memory and the polygon lane's O(n log n) work.
pub(crate) const MAX_OPERANDS: usize = 2_000;
pub(crate) const MIN_OPERANDS: usize = 2;
/// Cap on the point list an `ok` envelope may carry — and, being one constant
/// with two uses, on the points a single operand may hand in.
///
/// MIRRORED as `MAX_PATH_SEGMENTS` in `moda-engine/src/tess/tessellator.rs`,
/// which bounds one subpath's flattening there (it used to slice the whole
/// path to its first `MAX_PATH_SEGMENTS` commands, silently, as the TS
/// tessellator did). A boolean result above this is refused here. moda-engine depends on moda-scene, never the reverse, so the constant
/// lives here and `tess::tessellator::cap_parity_tests` over there fails if the
/// two ever drift apart.
///
/// The exit side is not covered by the cubic budgets: `MAX_INTERMEDIATE_CUBICS`
/// is 50_000, so without this an `ok` envelope could carry ~50_000 points —
/// five times what the tessellator will draw — from operands well inside every
/// other guard (two perpendicular 60-bar grids are 240 cubics each and
/// intersect into 3_600 squares, ~18_000 points).
pub const MAX_RESULT_POINTS: usize = 10_000;
/// The input side of `MAX_RESULT_POINTS`, and the tessellator's per-path cap:
/// an operand the renderer would truncate is not an operand worth solving.
pub(crate) const MAX_POINTS_PER_OPERAND: usize = MAX_RESULT_POINTS;
/// 1,600 operands of 20 cubics each; the polygon lane flattens and folds them
/// in well under a second (`large_tests::bench_large_inputs`).
pub(crate) const MAX_TOTAL_CUBICS: usize = 32_000;
/// Cross-operand point budget, checked before any point is converted.
///
/// `MAX_POINTS_PER_OPERAND` alone admits 2_000 · 10_000 = 20M points, which
/// `points_to_subpaths` would expand to as many as 80M cubics *before*
/// `MAX_TOTAL_CUBICS` gets a chance to refuse them. Sizing this off the cubic
/// budget instead caps the expansion: a line / curve / quadratic point becomes
/// exactly one cubic and an arc point at most four, so 40_000 points peak at
/// 160_000 cubics (~8 MB) on the way to a cap that refuses at 32_000.
///
/// The 1.25× headroom over `MAX_TOTAL_CUBICS` is deliberate: subpaths that
/// scrub out as degenerate contribute points but no cubics, and a request that
/// the cubic budget would admit should be refused by the cubic budget — with
/// its more specific `too_complex` — rather than pre-empted here. It is not 2×
/// (as it was at 8_000 cubics) because the byte budget below scales with it.
pub(crate) const MAX_POINTS_TOTAL: usize = MAX_TOTAL_CUBICS + MAX_TOTAL_CUBICS / 4;
/// Per-point allowance for the byte budget below. The widest point kind is an
/// arc — seven keys, five numbers — which is 175 bytes at full f64 precision
/// compactly encoded and 190 with separator spaces; the editor sends compact
/// JSON (`JSON.stringify`), so no request it could emit is refused by the byte
/// gate before its point count is even looked at.
const MAX_BYTES_PER_POINT: usize = 200;
/// Room for everything that is not a point: `op`, `accuracy`, and up to
/// `MAX_OPERANDS` operand wrappers with their `fillRule` and paint fields.
const REQUEST_HEADER_SLACK: usize = MAX_OPERANDS * 128;
/// Byte budget for the raw request string, checked *before* `serde_json`
/// materializes a `Value` tree — the only cap that can be, since every other
/// one reads a parsed request. Without it a 10M-point operand allocates over a
/// gigabyte of `Value` (~500 B per point) and is only then refused
/// `too_many_points`, and on wasm an allocation failure aborts the shared
/// engine instance rather than this one call.
///
/// Derived from the caps it fronts: `MAX_POINTS_TOTAL · MAX_BYTES_PER_POINT`
/// plus header slack (~8.3 MB). It is a bound on the *parse*, not a second
/// point cap: points encoded at the 27-byte minimum still fit ~300_000 of them
/// into the budget (~150 MB of `Value`), which `MAX_POINTS_TOTAL` then refuses
/// immediately after the parse. Bounded and survivable, where the unguarded
/// path was neither.
pub(crate) const MAX_REQUEST_BYTES: usize =
    MAX_POINTS_TOTAL * MAX_BYTES_PER_POINT + REQUEST_HEADER_SLACK;
/// Largest coordinate magnitude we will hand the solver.
pub(crate) const MAX_COORD: f64 = 1e7;
/// Pairwise work bound, in curve-pair comparisons, applied at *every*
/// flo_curves entry — there is no lane that reaches the solver without passing
/// through `check_pairwise_work`. A refusal from it hands the group to the
/// polygon lane rather than failing the request:
///
/// * per-operand normalization (`normalize_nonzero` /
///   `path_remove_overlapped_points`) collides an operand's segments against
///   each other, so its work is `cubics(operand)²`;
/// * every fold step of union, intersect, subtract and exclude collides the
///   accumulator against the next operand: `cubics(acc) · cubics(next)`.
///
/// The exact lane's cubic allowance (`bucket::EXACT_MAX_CUBICS`) is not a
/// substitute: two operands splitting it evenly are 4_000 · 4_000 = 1.6e7 curve
/// pairs, 4× this ceiling, and a single 8_000-cubic operand normalizes at
/// 6.4e7, 16× it.
pub(crate) const MAX_PAIRWISE_WORK: f64 = 4e6;
/// Cap on the cubics carried between fold steps, and on the solved result.
pub(crate) const MAX_INTERMEDIATE_CUBICS: usize = 50_000;

pub(crate) const MIN_ACCURACY: f64 = 1e-4;
pub(crate) const MAX_ACCURACY: f64 = 1.0;
pub(crate) const DEFAULT_ACCURACY: f64 = 0.01;

pub(crate) type Region = Vec<SimpleBezierPath>;

pub(crate) fn cubic_count(region: &Region) -> usize {
    region.iter().map(|(_, segs)| segs.len()).sum()
}

pub(crate) fn err(code: &str, message: impl Into<String>) -> BooleanError {
    BooleanError {
        code: code.to_string(),
        message: message.into(),
        operand_index: None,
    }
}

fn err_at(code: &str, message: impl Into<String>, index: usize) -> BooleanError {
    BooleanError {
        code: code.to_string(),
        message: message.into(),
        operand_index: Some(index),
    }
}

/// Every anchor and control coordinate of a point, for validation.
fn point_coords(p: &PathPoint) -> Vec<f64> {
    match p {
        PathPoint::Move { x, y } | PathPoint::Line { x, y } => vec![*x, *y],
        PathPoint::Curve { x, y, cp1, cp2 } => vec![*x, *y, cp1.0, cp1.1, cp2.0, cp2.1],
        PathPoint::Quadratic { x, y, cp } => vec![*x, *y, cp.0, cp.1],
        PathPoint::Arc {
            x,
            y,
            rx,
            ry,
            x_axis_rotation,
            ..
        } => vec![*x, *y, *rx, *ry, *x_axis_rotation],
    }
}

pub(crate) fn validate(
    operands: &[BooleanOperand],
    accuracy: Option<f64>,
) -> Result<f64, BooleanError> {
    if operands.len() < MIN_OPERANDS {
        return Err(err(
            "too_few_operands",
            format!("boolean ops need at least {MIN_OPERANDS} operands"),
        ));
    }
    if operands.len() > MAX_OPERANDS {
        return Err(err(
            "too_many_operands",
            format!("at most {MAX_OPERANDS} operands are supported"),
        ));
    }
    let mut total_points = 0usize;
    for (i, operand) in operands.iter().enumerate() {
        if operand.points.len() > MAX_POINTS_PER_OPERAND {
            return Err(err_at(
                "too_many_points",
                format!("operand has more than {MAX_POINTS_PER_OPERAND} points"),
                i,
            ));
        }
        total_points += operand.points.len();
    }
    // Before the coordinate scan, and well before `points_to_subpaths`: the
    // per-operand cap says nothing about the sum across 64 of them.
    if total_points > MAX_POINTS_TOTAL {
        return Err(err(
            "too_many_points",
            format!("operands carry {total_points} points in total (limit {MAX_POINTS_TOTAL})"),
        ));
    }
    for (i, operand) in operands.iter().enumerate() {
        for p in &operand.points {
            for v in point_coords(p) {
                if !v.is_finite() || v.abs() > MAX_COORD {
                    return Err(err_at(
                        "non_finite_coordinate",
                        "operand has a non-finite or out-of-range coordinate",
                        i,
                    ));
                }
            }
        }
    }
    let accuracy = match accuracy {
        Some(a) if a.is_finite() => a.clamp(MIN_ACCURACY, MAX_ACCURACY),
        Some(_) => {
            return Err(err("non_finite_coordinate", "accuracy must be finite"));
        }
        None => DEFAULT_ACCURACY,
    };
    Ok(accuracy)
}

/// Run the request. Pure; every failure is a typed `BooleanError`.
///
/// Guard order — each stage refuses before the work the next one would pay for,
/// and the last one refuses on the way out: request bytes
/// (`check_request_bytes`, in `boolean_paths_json`, before the parse) → operand
/// and point counts (`validate`) → coordinates (`validate`) → total cubics
/// after conversion → flattened vertices (`overlay::flatten`) → result points
/// (`check_result_points`) → output finiteness (`check_output_finite`,
/// `check_bounds_finite`). Inside the exact lane, pairwise work at every
/// flo_curves entry (`check_pairwise_work`) and intermediate cubics
/// (`check_intermediate`) bound flo_curves; a refusal there hands the group to
/// the polygon lane.
///
/// The operands are bucketed first (`bucket.rs`): groups that cannot interact
/// are solved separately and their answers concatenated. Each group runs the
/// exact lane while the request's `ExactBudget` lasts, and its answer is kept
/// when it agrees with the polygon lane's; otherwise, and for every group past
/// the budget, the polygon answer, refitted, is used.
pub(crate) fn solve(request: &BooleanRequest) -> Result<BooleanOutcome, BooleanError> {
    solve_on(request, Lane::Checked)
}

/// Which lane answers. Only `Checked` ships; the other two exist so the
/// quality measurement (`quality_tests.rs`) can score each lane on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lane {
    /// Exact, checked against the polygon lane, which answers when they differ.
    Checked,
    Exact,
    Polygon,
}

pub(crate) fn solve_on(
    request: &BooleanRequest,
    lane: Lane,
) -> Result<BooleanOutcome, BooleanError> {
    let Some(converted) = convert(&request.operands, request.accuracy)? else {
        return Ok(BooleanOutcome::Empty);
    };
    let offset = converted.offset;
    let fills: Vec<FillRule> = request.operands.iter().map(|o| o.fill_rule).collect();
    if lane == Lane::Exact {
        return finish_region(solve_exact(request.op, converted, &fills)?, offset);
    }
    let flat = overlay::flatten(&converted.regions)?;
    let resolved = overlay::resolve_all(&flat, &fills);
    let hulls: Vec<Option<Hull>> = converted.regions.iter().map(bucket::hull).collect();
    let mut budget = ExactBudget::new();
    let mut result: Region = Vec::new();
    for group in groups(request.op, &hulls) {
        let picked: Vec<Contours> = group.iter().map(|&i| resolved[i].clone()).collect();
        let expected = overlay::boolean(request.op, &picked);
        let exact = (lane == Lane::Checked
            && budget.admit(group.iter().map(|&i| &converted.regions[i])))
        .then(|| {
            let sub = Converted {
                regions: group
                    .iter()
                    .map(|&i| converted.regions[i].clone())
                    .collect(),
                offset,
                accuracy: converted.accuracy,
            };
            let sub_fills: Vec<FillRule> = group.iter().map(|&i| fills[i]).collect();
            solve_exact(request.op, sub, &sub_fills)
        });
        match exact {
            Some(Ok(region))
                if region_is_finite(&region) && overlay::agrees(&[&region], &expected) =>
            {
                result.extend(region)
            }
            // Past the budget, a wrong answer, or a refusal from flo_curves' own
            // guards: the polygon lane answers this group.
            _ => result.extend(flat.refit(&expected)),
        }
    }
    finish_region(result, offset)
}

/// The groups `op` is solved over (see `bucket.rs`), each ascending. Union and
/// exclude: connected components, since disjoint regions union and xor to
/// their concatenation. Subtract: the base and the cutters that touch it.
/// Intersect: every operand, or nothing when two of them cannot meet.
fn groups(op: BooleanOp, hulls: &[Option<Hull>]) -> Vec<Vec<usize>> {
    let all: Vec<usize> = (0..hulls.len()).collect();
    match op {
        BooleanOp::Union | BooleanOp::Exclude => bucket::components(hulls),
        BooleanOp::Subtract => {
            let cut = bucket::touching(hulls, 0, &all[1..]);
            if cut.is_empty() {
                Vec::new()
            } else {
                vec![cut]
            }
        }
        BooleanOp::Intersect => {
            let meet = hulls.iter().try_fold(None::<Hull>, |acc, h| {
                let h = (*h)?;
                Some(Some(match acc {
                    None => h,
                    Some(a) => Hull {
                        min_x: a.min_x.max(h.min_x),
                        min_y: a.min_y.max(h.min_y),
                        max_x: a.max_x.min(h.max_x),
                        max_y: a.max_y.min(h.max_y),
                    },
                }))
            });
            match meet {
                Some(Some(m)) if m.min_x <= m.max_x && m.min_y <= m.max_y => vec![all],
                _ => Vec::new(),
            }
        }
    }
}

/// The exact lane: flo_curves on the operands' own cubics, in `convert`'s
/// translated space.
pub(crate) fn solve_exact(
    op: BooleanOp,
    converted: Converted,
    fills: &[FillRule],
) -> Result<Region, BooleanError> {
    let Prepared { regions, accuracy } = normalize(converted, fills)?;
    match op {
        BooleanOp::Union => solve_union(&regions, accuracy),
        BooleanOp::Intersect => solve_intersect(&regions, accuracy),
        BooleanOp::Subtract => solve_subtract(&regions, accuracy),
        BooleanOp::Exclude => solve_exclude(&regions, accuracy),
    }
}

/// Every coordinate finite. The agreement check flattens the region, which
/// must not see a NaN; `finish_region` refuses one on the way out anyway.
pub(crate) fn region_is_finite(region: &Region) -> bool {
    region.iter().all(|(start, segs)| {
        start.0.is_finite()
            && start.1.is_finite()
            && segs
                .iter()
                .all(|(a, b, c)| [a.0, a.1, b.0, b.1, c.0, c.1].iter().all(|v| v.is_finite()))
    })
}

/// The operands of a request, converted, budget-checked, translated to the
/// origin (`Converted::offset` undoes that) and normalized under their own
/// fill rules: what every exact-lane op (boolean or pathfinder) folds over.
pub(crate) struct Prepared {
    pub(crate) regions: Vec<Region>,
    pub(crate) accuracy: f64,
}

/// The operands of a request validated, converted to cubics, budget-checked
/// and translated to the origin, but not yet normalized: what both lanes
/// start from.
#[derive(Clone)]
pub(crate) struct Converted {
    pub(crate) regions: Vec<Region>,
    /// The translation applied to every region; `finish_region` undoes it.
    pub(crate) offset: (f64, f64),
    pub(crate) accuracy: f64,
}

/// Validation, conversion, the cubic budget and the translation to the origin,
/// shared by the boolean ops and the pathfinder ops (`pathfinder.rs`). `None`
/// means no operand carried any geometry.
pub(crate) fn convert(
    operands: &[BooleanOperand],
    accuracy: Option<f64>,
) -> Result<Option<Converted>, BooleanError> {
    let accuracy = validate(operands, accuracy)?;

    // Geometry: subpaths per operand, all cubics.
    let mut regions: Vec<Region> = Vec::with_capacity(operands.len());
    for operand in operands {
        regions.push(points_to_subpaths(&operand.points, accuracy));
    }

    let total: usize = regions.iter().map(cubic_count).sum();
    if total > MAX_TOTAL_CUBICS {
        return Err(err(
            "too_complex",
            format!("input expands to {total} cubic segments (limit {MAX_TOTAL_CUBICS})"),
        ));
    }

    // Translate everything so the union bbox min sits at the origin: flo_curves'
    // accuracy is absolute, so keeping coordinates small keeps the snapping
    // meaningful. Translation only — no scaling, so nothing else has to change.
    let Some((off_x, off_y)) = region_min(&regions) else {
        return Ok(None);
    };
    for region in regions.iter_mut() {
        translate_region(region, -off_x, -off_y);
    }
    Ok(Some(Converted {
        regions,
        offset: (off_x, off_y),
        accuracy,
    }))
}

/// The exact lane's per-operand normalization (flo_curves), under each
/// operand's own fill rule.
pub(crate) fn normalize(
    converted: Converted,
    fills: &[FillRule],
) -> Result<Prepared, BooleanError> {
    let Converted {
        mut regions,
        accuracy,
        ..
    } = converted;
    // Pre-normalize each operand to a non-self-overlapping set under its own
    // fill rule, so the ops below see the region the renderer would paint.
    for (i, (region, fill)) in regions.iter_mut().zip(fills.iter()).enumerate() {
        if region.is_empty() {
            continue;
        }
        // Normalization is itself a flo_curves entry: it collides the operand
        // against itself, so its work is quadratic in the operand's own cubics.
        let own = cubic_count(region) as f64;
        if let Err(e) = check_pairwise_work(own * own, "operand normalization") {
            return Err(BooleanError {
                operand_index: Some(i),
                ..e
            });
        }
        *region = match fill {
            FillRule::NonZero => normalize_nonzero(region, accuracy),
            FillRule::EvenOdd => {
                path_remove_overlapped_points::<SimpleBezierPath, SimpleBezierPath>(
                    region, accuracy,
                )
            }
        };
    }

    Ok(Prepared { regions, accuracy })
}

/// A solved region back to caller space and through every exit guard:
/// intermediate cubics, result points, output finiteness, bounds.
pub(crate) fn finish_region(
    mut result: Region,
    (off_x, off_y): (f64, f64),
) -> Result<BooleanOutcome, BooleanError> {
    // Every op gets the same exit budget. `check_step` bounds what goes *into*
    // each flo_curves call; nothing bounds what the last one hands back, so
    // without this a pathological result could still reach the JSON encoder
    // (and the caller's scene) unbounded.
    check_intermediate(cubic_count(&result))?;

    translate_region(&mut result, off_x, off_y);

    if result.is_empty() {
        return Ok(BooleanOutcome::Empty);
    }

    let mut points: Vec<PathPoint> = Vec::new();
    for subpath in &result {
        subpath_to_points(subpath, &mut points);
    }
    if points.len() < 3 {
        return Ok(BooleanOutcome::Empty);
    }
    // Last two guards, on the way out. Size first — it is a length comparison,
    // while the finiteness scan walks every coordinate of every point.
    check_result_points(points.len())?;
    // The inputs were finiteness-checked; the output never was.
    check_output_finite(&points)?;
    let Some(bounds) = region_bounds(&result) else {
        return Ok(BooleanOutcome::Empty);
    };
    check_bounds_finite(&bounds)?;
    Ok(BooleanOutcome::Shape {
        points,
        subpath_count: result.len(),
        bounds,
    })
}

/// The non-zero region of one operand: the set of points whose signed winding
/// number is non-zero, as a non-self-overlapping ring set the ops below can
/// treat even-odd.
///
/// flo_curves' own ray caster already accumulates *signed* per-label crossing
/// counts (`arithmetic/ray_cast.rs`: `+1` / `-1` by the sign of the ray's dot
/// product with the edge normal), which is precisely a winding number — but
/// `GraphPath::from_path` reverses anti-clockwise subpaths to clockwise before
/// they ever reach it, so every subpath contributes the same sign and the
/// counts say "inside this subpath", not "wound which way". That is why
/// `path_remove_interior_points` returns the outer perimeter of an operand
/// whose inner loop is wound against it, where the renderer paints a hole.
///
/// So: label every subpath separately, remember the orientation the graph is
/// about to discard (`subpath_signed_area`), and put the signs back in the
/// closure. `counts[i]` is then how deep the ray is inside subpath `i`, and
/// `Σ orientation[i] · counts[i]` is the true winding number. The global sign
/// convention does not matter, only that it is consistent, since the test is
/// `!= 0`.
///
/// Doing it this way rather than by nesting depth is not a shortcut: depth
/// arithmetic ("outermost minus the next ring in") gets alternating nesting
/// wrong — a clockwise loop inside an anti-clockwise loop inside a clockwise
/// loop has windings 1, 0, 1, so the innermost is filled *again*. The winding
/// sum gets that right by construction, and `exterior_paths` hands back the
/// rings for it.
fn normalize_nonzero(region: &Region, accuracy: f64) -> Region {
    let orientation: Vec<i32> = region
        .iter()
        .map(|subpath| {
            if subpath_signed_area(subpath) < 0.0 {
                -1
            } else {
                1
            }
        })
        .collect();

    let mut graph: GraphPath<Coord2, PathLabel> = GraphPath::new();
    graph = graph.merge(GraphPath::from_merged_paths(
        region
            .iter()
            .enumerate()
            .map(|(i, subpath)| (subpath, PathLabel(i as u32))),
    ));
    graph.self_collide(accuracy);
    graph.round(accuracy);
    graph.set_edge_kinds_by_ray_casting(move |counts| {
        // `counts` is padded to at least two entries and grows to the highest
        // label the ray met, so zip against the orientations rather than
        // indexing: anything past the end was never crossed and winds nothing.
        counts
            .iter()
            .zip(orientation.iter())
            .map(|(crossings, sign)| crossings * sign)
            .sum::<i32>()
            != 0
    });
    graph.heal_exterior_gaps();
    graph.exterior_paths()
}

/// Union and exclude fold as a balanced tree (`fold_balanced`) rather than
/// calling `path_add_chain`: the chain call is one unbounded flo_curves entry,
/// while an explicit fold puts every entry behind `check_step`, exactly like the
/// other two ops. Both are associative and commutative, so the region is the
/// same in any order.
pub(crate) fn solve_union(regions: &[Region], accuracy: f64) -> Result<Region, BooleanError> {
    fold_balanced(regions, |a, b| path_add::<SimpleBezierPath>(a, b, accuracy))
}

/// Fold `op` over the non-empty regions as a balanced tree: pairs, then pairs
/// of pairs. A left fold collides the whole growing accumulator with each next
/// operand, so N operands cost O(N²) in its size; the tree keeps both sides of
/// every step about the same size (a 200-operand grid's union: 1 s → 0.4 s
/// with the budgets lifted). Every step is behind `check_step`, and every
/// carried region behind `check_intermediate`.
fn fold_balanced(
    regions: &[Region],
    op: impl Fn(&Region, &Region) -> Region,
) -> Result<Region, BooleanError> {
    let mut level: Vec<Region> = regions.iter().filter(|r| !r.is_empty()).cloned().collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut it = level.into_iter();
        while let Some(a) = it.next() {
            next.push(match it.next() {
                Some(b) => {
                    check_step(&a, &b)?;
                    let folded = op(&a, &b);
                    check_intermediate(cubic_count(&folded))?;
                    folded
                }
                None => a,
            });
        }
        level = next;
    }
    Ok(level.pop().unwrap_or_default())
}

fn solve_intersect(regions: &[Region], accuracy: f64) -> Result<Region, BooleanError> {
    let mut acc = regions[0].clone();
    for next in &regions[1..] {
        if acc.is_empty() || next.is_empty() {
            return Ok(Vec::new());
        }
        check_step(&acc, next)?;
        acc = path_intersect::<SimpleBezierPath>(&acc, next, accuracy);
    }
    Ok(acc)
}

pub(crate) fn solve_subtract(regions: &[Region], accuracy: f64) -> Result<Region, BooleanError> {
    let base = regions[0].clone();
    if base.is_empty() {
        return Ok(Vec::new());
    }
    let cutter = solve_union(&regions[1..], accuracy)?;
    if cutter.is_empty() {
        return Ok(base);
    }
    check_step(&base, &cutter)?;
    Ok(path_sub::<SimpleBezierPath>(&base, &cutter, accuracy))
}

fn solve_exclude(regions: &[Region], accuracy: f64) -> Result<Region, BooleanError> {
    fold_balanced(regions, |a, b| {
        let split = path_full_intersect::<SimpleBezierPath>(a, b, accuracy);
        let [mut outside_a, outside_b] = split.exterior_paths;
        outside_a.extend(outside_b);
        outside_a
    })
}

/// The single gate for `MAX_RESULT_POINTS`. `too_complex` rather than
/// `solver_failed`: an oversized result is a legitimate answer to an
/// unreasonable question, the same class as the cubic budgets, not a solver
/// malfunction.
pub(crate) fn check_result_points(len: usize) -> Result<(), BooleanError> {
    if len > MAX_RESULT_POINTS {
        return Err(err(
            "too_complex",
            format!("result has {len} points (limit {MAX_RESULT_POINTS})"),
        ));
    }
    Ok(())
}

/// The output gate. Inputs are finiteness-checked in `validate`; what
/// flo_curves hands back was not checked anywhere, and `subpath_to_points`
/// copies coordinates straight through. That matters because a non-finite f64
/// does not survive as one: `json!` encodes it as `null`, so a single bad
/// coordinate ships as `{"type":"line","x":null,"y":null}` under
/// `status: "ok"` and is written into a saved path node, where every later
/// reader inherits it (guideline 6.3b — a derived number in the scene must be
/// finite, and `NaN` persists as `null` rather than failing loudly).
///
/// `solver_failed` is the right code: reaching here means the solver returned
/// geometry that is not geometry, which is the same class of event as a panic
/// caught by `catch_unwind`, not a refusal by one of the module's budgets.
///
/// Finiteness only. Magnitude is deliberately not re-checked on the way out:
/// `MAX_COORD` bounds what a caller may hand *in*, while a union of two legal
/// operands can legitimately span further than either, and every finite f64 is
/// representable in the envelope.
pub(crate) fn check_output_finite(points: &[PathPoint]) -> Result<(), BooleanError> {
    for p in points {
        if point_coords(p).iter().any(|v| !v.is_finite()) {
            return Err(err(
                "solver_failed",
                "solver produced a non-finite coordinate",
            ));
        }
    }
    Ok(())
}

/// Bounds ship in the envelope too, and `Bounds::include` reduces with
/// `f64::min` / `f64::max`, which *skip* NaN rather than propagating it — so a
/// partially non-finite region yields finite-looking bounds wrapped around it,
/// and nothing about the envelope would look wrong. `check_output_finite` runs
/// first and already rejects such a region; this states that dependency instead
/// of leaving the envelope's correctness resting on a NaN-skipping reduction.
pub(crate) fn check_bounds_finite(b: &Bounds) -> Result<(), BooleanError> {
    if [b.min_x, b.min_y, b.max_x, b.max_y]
        .iter()
        .all(|v| v.is_finite())
    {
        return Ok(());
    }
    Err(err("solver_failed", "solver produced non-finite bounds"))
}

/// The single gate for `MAX_REQUEST_BYTES`, and the first guard of all: it runs
/// on the raw request string, before `serde_json` has allocated anything.
///
/// It reuses `too_many_points` rather than introducing a code: the budget is
/// derived from the point caps and an over-budget request is over-budget *in
/// points* — `too_complex` is the module's token for geometry that expands too
/// far once parsed, which this deliberately never gets to find out.
pub(crate) fn check_request_bytes(len: usize) -> Result<(), BooleanError> {
    if len > MAX_REQUEST_BYTES {
        return Err(err(
            "too_many_points",
            format!("request is {len} bytes (limit {MAX_REQUEST_BYTES})"),
        ));
    }
    Ok(())
}

/// The single gate for `MAX_PAIRWISE_WORK`. Every flo_curves entry routes
/// through here, so removing the comparison removes the bound everywhere —
/// which is what the guard tests in `tests.rs` are pinned against.
pub(crate) fn check_pairwise_work(work: f64, what: &str) -> Result<(), BooleanError> {
    if work > MAX_PAIRWISE_WORK {
        return Err(err(
            "too_complex",
            format!("{what} needs {work:.0} curve-pair comparisons (limit {MAX_PAIRWISE_WORK:.0})"),
        ));
    }
    Ok(())
}

/// The single gate for `MAX_INTERMEDIATE_CUBICS`.
pub(crate) fn check_intermediate(count: usize) -> Result<(), BooleanError> {
    if count > MAX_INTERMEDIATE_CUBICS {
        return Err(err(
            "too_complex",
            format!(
                "{count} cubic segments carried between steps (limit {MAX_INTERMEDIATE_CUBICS})"
            ),
        ));
    }
    Ok(())
}

/// One fold step, for every op: bound the flo_curves entry it is about to make
/// and the accumulator it is about to carry in.
pub(crate) fn check_step(a: &Region, b: &Region) -> Result<(), BooleanError> {
    check_pairwise_work(
        cubic_count(a) as f64 * cubic_count(b) as f64,
        "boolean fold step",
    )?;
    check_intermediate(cubic_count(a))
}

pub(crate) fn translate_region(region: &mut Region, dx: f64, dy: f64) {
    for (start, segments) in region.iter_mut() {
        *start = Coord2(start.0 + dx, start.1 + dy);
        for seg in segments.iter_mut() {
            seg.0 = Coord2(seg.0 .0 + dx, seg.0 .1 + dy);
            seg.1 = Coord2(seg.1 .0 + dx, seg.1 .1 + dy);
            seg.2 = Coord2(seg.2 .0 + dx, seg.2 .1 + dy);
        }
    }
}

/// Control-hull minimum over every operand — only used as a translation, so the
/// hull's slack over the true curve extent does not matter.
fn region_min(regions: &[Region]) -> Option<(f64, f64)> {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    for region in regions {
        for (start, segments) in region {
            min_x = min_x.min(start.0);
            min_y = min_y.min(start.1);
            for (c1, c2, p) in segments {
                for c in [c1, c2, p] {
                    min_x = min_x.min(c.0);
                    min_y = min_y.min(c.1);
                }
            }
        }
    }
    if min_x.is_finite() && min_y.is_finite() {
        Some((min_x, min_y))
    } else {
        None
    }
}

/// Tight bounds of the solved region: per-cubic extrema, not the control hull.
pub(crate) fn region_bounds(region: &Region) -> Option<Bounds> {
    let mut b = Bounds {
        min_x: f64::INFINITY,
        min_y: f64::INFINITY,
        max_x: f64::NEG_INFINITY,
        max_y: f64::NEG_INFINITY,
    };
    for (start, segments) in region {
        let mut p0 = *start;
        b.include(p0.0, p0.1);
        for (c1, c2, p3) in segments {
            b.include(p3.0, p3.1);
            for t in cubic_extrema(p0.0, c1.0, c2.0, p3.0) {
                b.include(
                    cubic_at(p0.0, c1.0, c2.0, p3.0, t),
                    cubic_at(p0.1, c1.1, c2.1, p3.1, t),
                );
            }
            for t in cubic_extrema(p0.1, c1.1, c2.1, p3.1) {
                b.include(
                    cubic_at(p0.0, c1.0, c2.0, p3.0, t),
                    cubic_at(p0.1, c1.1, c2.1, p3.1, t),
                );
            }
            p0 = *p3;
        }
    }
    // Still at the `INFINITY` sentinels means the region held no segment at
    // all — `None` is "nothing to bound", not "the bounds came out bad". A
    // region whose *contents* are non-finite is `check_output_finite`'s job,
    // and `check_bounds_finite` pins what this returns either way.
    if b.min_x.is_finite() && b.min_y.is_finite() {
        Some(b)
    } else {
        None
    }
}

fn cubic_at(a: f64, b: f64, c: f64, d: f64, t: f64) -> f64 {
    let mt = 1.0 - t;
    mt * mt * mt * a + 3.0 * mt * mt * t * b + 3.0 * mt * t * t * c + t * t * t * d
}

/// Roots of the cubic's derivative inside (0, 1) on one axis.
fn cubic_extrema(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    // B'(t)/3 = (−a+3b−3c+d)t² + 2(a−2b+c)t + (b−a)
    let qa = -a + 3.0 * b - 3.0 * c + d;
    let qb = 2.0 * (a - 2.0 * b + c);
    let qc = b - a;
    let mut roots = Vec::new();
    let mut push = |t: f64| {
        if t.is_finite() && t > 0.0 && t < 1.0 {
            roots.push(t);
        }
    };
    if qa.abs() < 1e-12 {
        if qb.abs() >= 1e-12 {
            push(-qc / qb);
        }
        return roots;
    }
    let disc = qb * qb - 4.0 * qa * qc;
    if disc < 0.0 {
        return roots;
    }
    let sq = disc.sqrt();
    // Citardauq form: adding same-signed terms avoids the catastrophic
    // cancellation the textbook formula suffers when |qb| ≈ sq (a nearly
    // straight cubic, which is most of what a boolean result is made of).
    let sign = if qb < 0.0 { -1.0 } else { 1.0 };
    let q = -0.5 * (qb + sign * sq);
    push(q / qa);
    if q != 0.0 {
        push(qc / q);
    }
    roots
}
