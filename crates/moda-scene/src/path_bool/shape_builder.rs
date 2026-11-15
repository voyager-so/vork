//! The Shape Builder (Illustrator's Shape Builder, Figma Draw's shape builder):
//! the regions a set of overlapping shapes makes, and merging or removing some
//! of them — built on the pathfinder's Divide.
//!
//! **Regions** are Divide's faces over every operand, filled or not (an
//! unfilled shape's outline still bounds regions the user can pick, as in
//! Illustrator), one per connected face. Each region carries:
//! - `sample`: a point well inside it, the stable handle a caller (or an agent)
//!   picks it by — faces are recomputed per request, points survive that;
//! - `cover`: the operands (bottom-most first) that contain it;
//! - `source`: the operand whose style it wears — the topmost *filled* operand
//!   covering it, else the topmost covering one (Divide's rule).
//!
//! **Build** picks regions by points (each point picks the region containing
//! it; points outside every region are ignored, repeats count once) and then:
//! - `merge`: the picked regions become ONE shape (their union), styled from
//!   the FIRST picked region's `source` — the region the drag started on
//!   (Illustrator's "Pick Colour From: Artwork"; Figma's help does not state a
//!   rule). A single pick is Figma's "extract a region".
//! - `remove`: the picked regions are deleted.
//!
//! Either way every operand covering a picked region loses the picked area
//! (`operand − union(picked)`; an operand left with nothing is `removed`), and
//! the other operands are `unchanged`. So a merge leaves the untouched regions
//! with the shapes they were part of, as both apps do.
//!
//! Contract:
//!
//! ```jsonc
//! // request
//! { "op": "regions" | "merge" | "remove",
//!   "operands": [ { "points": PathPoint[], "fillRule": "nonzero"|"evenodd", "painted": true } ],
//!   "picks": [[x, y], ...],            // merge / remove only
//!   "accuracy": 0.01 }
//! // response, regions
//! { "status": "ok", "regions": [ { "points", "subpathCount", "bounds", "sample": {x, y},
//!     "cover": [0, 1], "source": 1 } ] }
//! // response, merge / remove
//! { "status": "ok", "picked": [2, 0],   // region indices, in pick order
//!   "merged": { "points", "subpathCount", "bounds", "source" } | null,
//!   "operands": [ { "status": "unchanged" } | { "status": "removed" }
//!               | { "status": "changed", "points", "subpathCount", "bounds" } ] }
//! { "status": "empty" }                 // regions: nothing to divide; build: no pick hit a region
//! { "status": "error", "code", "message", "operandIndex"? }
//! ```
//!
//! Every geometric step is the checked boolean / pathfinder solver (exact lane
//! cross-checked by the polygon lane), so this module adds no kernel of its own:
//! only picking (flattened polygons) and bookkeeping.

use crate::value::{json, Value};

use super::convert::{flatten_subpath, points_to_subpaths};
use super::pathfinder::{
    pathfinder_paths, OperandPaint, PathfinderOp, PathfinderOutcome, PathfinderRequest,
};
use super::solve::{err, MAX_OPERANDS};
use super::{
    boolean_paths, parse_accuracy, parse_operands, BooleanError, BooleanOp, BooleanOperand,
    BooleanOutcome, BooleanRequest, Bounds, FillRule, PathPoint,
};

/// Samples per curve when flattening for picking and cover tests.
const FLATTEN: usize = 16;
/// Cap on the pick points one request may carry.
pub const MAX_PICKS: usize = 10_000;

#[derive(Debug, Clone)]
pub struct ShapeBuilderRegion {
    pub points: Vec<PathPoint>,
    pub subpath_count: usize,
    pub bounds: Bounds,
    /// A point well inside the region.
    pub sample: (f64, f64),
    /// Operands containing the region, ascending.
    pub cover: Vec<usize>,
    /// The operand whose style the region wears.
    pub source: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildMode {
    Merge,
    Remove,
}

#[derive(Debug, Clone)]
pub struct Shape {
    pub points: Vec<PathPoint>,
    pub subpath_count: usize,
    pub bounds: Bounds,
}

#[derive(Debug, Clone)]
pub enum OperandResult {
    Unchanged,
    Removed,
    Changed(Shape),
}

#[derive(Debug, Clone)]
pub struct BuildResult {
    /// Indices into the regions, in pick order.
    pub picked: Vec<usize>,
    /// The merged shape and the operand it is styled from (`merge` only).
    pub merged: Option<(Shape, usize)>,
    /// Parallel to the operands.
    pub operands: Vec<OperandResult>,
}

/// The regions of `operands` (bottom-most first). `painted[i]`: operand `i`
/// has a visible fill (decides `source`). Empty when nothing encloses area.
pub fn shape_builder_regions(
    operands: &[BooleanOperand],
    painted: &[bool],
    accuracy: Option<f64>,
) -> Result<Vec<ShapeBuilderRegion>, BooleanError> {
    if painted.len() != operands.len() {
        return Err(err(
            "invalid_request",
            "painted must be parallel to operands",
        ));
    }
    // Divide with every operand painted keeps every face (Divide drops faces
    // only unfilled operands cover); the style owner is recomputed below from
    // the real `painted`.
    let request = PathfinderRequest {
        op: PathfinderOp::Divide,
        operands: operands.to_vec(),
        paint: vec![
            OperandPaint {
                painted: true,
                key: None
            };
            operands.len()
        ],
        accuracy,
    };
    let pieces = match pathfinder_paths(&request)? {
        PathfinderOutcome::Pieces(pieces) => pieces,
        PathfinderOutcome::Empty => return Ok(Vec::new()),
    };
    let shapes: Vec<Polygons> = operands
        .iter()
        .map(|o| Polygons::new(&o.points, o.fill_rule))
        .collect();
    let mut out = Vec::with_capacity(pieces.len());
    for piece in pieces {
        let polys = Polygons::new(&piece.points, FillRule::EvenOdd);
        let Some(sample) = polys.interior_sample(&piece.bounds) else {
            continue;
        };
        let mut cover: Vec<usize> = (0..operands.len())
            .filter(|&i| shapes[i].contains(sample))
            .collect();
        // A sample on an operand's edge (can't happen for a proper face, but
        // flattening is approximate): Divide's own answer is the fallback.
        if cover.is_empty() {
            cover.push(piece.source);
        }
        let source = cover
            .iter()
            .rev()
            .find(|&&i| painted[i])
            .or_else(|| cover.last())
            .copied()
            .unwrap_or(piece.source);
        out.push(ShapeBuilderRegion {
            points: piece.points,
            subpath_count: piece.subpath_count,
            bounds: piece.bounds,
            sample,
            cover,
            source,
        });
    }
    Ok(out)
}

/// Pick regions by `picks` and merge or remove them (module doc). `Ok(None)`
/// when no pick lands in a region.
pub fn shape_builder_build(
    operands: &[BooleanOperand],
    painted: &[bool],
    accuracy: Option<f64>,
    mode: BuildMode,
    picks: &[(f64, f64)],
) -> Result<Option<BuildResult>, BooleanError> {
    if picks.len() > MAX_PICKS {
        return Err(err(
            "too_complex",
            format!("at most {MAX_PICKS} pick points are supported"),
        ));
    }
    let regions = shape_builder_regions(operands, painted, accuracy)?;
    let polys: Vec<Polygons> = regions
        .iter()
        .map(|r| Polygons::new(&r.points, FillRule::EvenOdd))
        .collect();
    let mut picked: Vec<usize> = Vec::new();
    for &p in picks {
        if !p.0.is_finite() || !p.1.is_finite() {
            continue;
        }
        if let Some(i) = (0..regions.len())
            .find(|&i| inside_bounds(&regions[i].bounds, p) && polys[i].contains(p))
        {
            if !picked.contains(&i) {
                picked.push(i);
            }
        }
    }
    if picked.is_empty() {
        return Ok(None);
    }

    let area = union_regions(
        &picked.iter().map(|&i| &regions[i]).collect::<Vec<_>>(),
        accuracy,
    )?;
    let Some(area) = area else {
        return Ok(None);
    };

    let touched: Vec<bool> = (0..operands.len())
        .map(|i| picked.iter().any(|&r| regions[r].cover.contains(&i)))
        .collect();
    let mut results = Vec::with_capacity(operands.len());
    for (i, operand) in operands.iter().enumerate() {
        if !touched[i] {
            results.push(OperandResult::Unchanged);
            continue;
        }
        let outcome = boolean_paths(&BooleanRequest {
            op: BooleanOp::Subtract,
            operands: vec![
                operand.clone(),
                BooleanOperand {
                    points: area.points.clone(),
                    fill_rule: FillRule::EvenOdd,
                },
            ],
            accuracy,
        })
        .map_err(|e| BooleanError {
            operand_index: Some(i),
            ..e
        })?;
        results.push(match outcome {
            BooleanOutcome::Empty => OperandResult::Removed,
            BooleanOutcome::Shape {
                points,
                subpath_count,
                bounds,
            } => OperandResult::Changed(Shape {
                points,
                subpath_count,
                bounds,
            }),
        });
    }

    let merged = match mode {
        BuildMode::Merge => Some((area, regions[picked[0]].source)),
        BuildMode::Remove => None,
    };
    Ok(Some(BuildResult {
        picked,
        merged,
        operands: results,
    }))
}

/// The union of `regions`: a single region as it stands; more through the
/// boolean solver, folded `MAX_OPERANDS` at a time.
fn union_regions(
    regions: &[&ShapeBuilderRegion],
    accuracy: Option<f64>,
) -> Result<Option<Shape>, BooleanError> {
    if let [only] = regions {
        return Ok(Some(Shape {
            points: only.points.clone(),
            subpath_count: only.subpath_count,
            bounds: only.bounds,
        }));
    }
    let mut shape: Option<Shape> = None;
    let mut rest = regions.iter().map(|r| r.points.clone());
    loop {
        let mut batch: Vec<Vec<PathPoint>> = shape.take().map(|s| s.points).into_iter().collect();
        batch.extend(rest.by_ref().take(MAX_OPERANDS - batch.len()));
        if batch.len() < 2 {
            // Nothing new was added: the accumulated union is the answer.
            return Ok(batch.pop().map(|points| {
                let bounds = points_bounds(&points);
                let subpath_count = points
                    .iter()
                    .filter(|p| matches!(p, PathPoint::Move { .. }))
                    .count();
                Shape {
                    points,
                    subpath_count,
                    bounds,
                }
            }));
        }
        let more = rest.len() > 0;
        match boolean_paths(&BooleanRequest {
            op: BooleanOp::Union,
            operands: batch
                .into_iter()
                .map(|points| BooleanOperand {
                    points,
                    fill_rule: FillRule::EvenOdd,
                })
                .collect(),
            accuracy,
        })? {
            BooleanOutcome::Empty => return Ok(None),
            BooleanOutcome::Shape {
                points,
                subpath_count,
                bounds,
            } => {
                let s = Shape {
                    points,
                    subpath_count,
                    bounds,
                };
                if !more {
                    return Ok(Some(s));
                }
                shape = Some(s);
            }
        }
    }
}

fn inside_bounds(b: &Bounds, (x, y): (f64, f64)) -> bool {
    x >= b.min_x && x <= b.max_x && y >= b.min_y && y <= b.max_y
}

// ---------------------------------------------------------------------------
// Flattened polygons: containment and interior samples
// ---------------------------------------------------------------------------

struct Polygons {
    rings: Vec<Vec<(f64, f64)>>,
    fill_rule: FillRule,
}

impl Polygons {
    fn new(points: &[PathPoint], fill_rule: FillRule) -> Polygons {
        let rings = points_to_subpaths(points, 1e-9)
            .iter()
            .map(|sp| flatten_subpath(sp, FLATTEN))
            .filter(|r| r.len() >= 3)
            .collect();
        Polygons { rings, fill_rule }
    }

    /// Winding number of `p` over every ring.
    fn winding(&self, (px, py): (f64, f64)) -> i32 {
        let mut w = 0;
        for ring in &self.rings {
            let n = ring.len();
            for k in 0..n {
                let (x0, y0) = ring[k];
                let (x1, y1) = ring[(k + 1) % n];
                if y0 <= py {
                    if y1 > py && cross(x0, y0, x1, y1, px, py) > 0.0 {
                        w += 1;
                    }
                } else if y1 <= py && cross(x0, y0, x1, y1, px, py) < 0.0 {
                    w -= 1;
                }
            }
        }
        w
    }

    fn contains(&self, p: (f64, f64)) -> bool {
        let w = self.winding(p);
        match self.fill_rule {
            FillRule::NonZero => w != 0,
            FillRule::EvenOdd => w % 2 != 0,
        }
    }

    /// A point well inside the (even-odd) polygons: across several horizontal
    /// scanlines, the middle of the widest inside run. `None` for no area.
    fn interior_sample(&self, bounds: &Bounds) -> Option<(f64, f64)> {
        let h = bounds.max_y - bounds.min_y;
        if !(h > 0.0) {
            return None;
        }
        let mut best: Option<((f64, f64), f64)> = None;
        const LINES: usize = 15;
        for k in 0..LINES {
            // Offset from the exact fractions so a scanline rarely passes
            // through a vertex.
            let y = bounds.min_y + h * ((k as f64 + 0.5) / LINES as f64 + 0.000_137);
            let mut xs: Vec<f64> = Vec::new();
            for ring in &self.rings {
                let n = ring.len();
                for i in 0..n {
                    let (x0, y0) = ring[i];
                    let (x1, y1) = ring[(i + 1) % n];
                    if (y0 <= y) != (y1 <= y) {
                        xs.push(x0 + (y - y0) / (y1 - y0) * (x1 - x0));
                    }
                }
            }
            xs.sort_by(f64::total_cmp);
            for pair in xs.chunks_exact(2) {
                let w = pair[1] - pair[0];
                // Prefer width, with the vertical middle breaking near-ties.
                let score = w * (1.0 - 0.25 * ((k as f64 + 0.5) / LINES as f64 - 0.5).abs());
                if w > 0.0 && best.map_or(true, |(_, s)| score > s) {
                    best = Some((((pair[0] + pair[1]) / 2.0, y), score));
                }
            }
        }
        best.map(|(p, _)| p)
    }
}

fn cross(x0: f64, y0: f64, x1: f64, y1: f64, px: f64, py: f64) -> f64 {
    (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

/// JSON-in / JSON-out, the surface the wasm export wraps.
pub fn shape_builder_json(request_json: &str) -> String {
    match super::solve::check_request_bytes(request_json.len())
        .and_then(|()| run_json(request_json))
    {
        Ok(v) => v,
        Err(e) => {
            let mut obj = json!({ "status": "error", "code": e.code, "message": e.message });
            if let Some(i) = e.operand_index {
                obj["operandIndex"] = json!(i);
            }
            obj
        }
    }
    .to_string()
}

fn bounds_json(b: &Bounds) -> Value {
    json!({ "minX": b.min_x, "minY": b.min_y, "maxX": b.max_x, "maxY": b.max_y })
}

fn points_json(points: &[PathPoint]) -> Value {
    Value::Array(points.iter().map(PathPoint::to_json).collect())
}

fn run_json(request_json: &str) -> Result<Value, BooleanError> {
    let value: Value = crate::value::from_str(request_json).map_err(|e| BooleanError {
        code: "invalid_json".to_string(),
        message: format!("request is not valid JSON: {e}"),
        operand_index: None,
    })?;
    let obj = value
        .as_object()
        .ok_or_else(|| err("invalid_request", "request must be an object"))?;
    let op = obj.get("op").and_then(Value::as_str);
    let mode = match op {
        Some("regions") => None,
        Some("merge") => Some(BuildMode::Merge),
        Some("remove") => Some(BuildMode::Remove),
        _ => {
            return Err(err(
                "invalid_request",
                "op must be one of regions|merge|remove",
            ))
        }
    };
    let operands = parse_operands(obj)?;
    let accuracy = parse_accuracy(obj)?;
    let mut painted = Vec::with_capacity(operands.len());
    for (i, o) in obj
        .get("operands")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        painted.push(match o.get("painted") {
            None | Some(Value::Null) => true,
            Some(Value::Bool(b)) => *b,
            Some(_) => {
                return Err(BooleanError {
                    operand_index: Some(i),
                    ..err("invalid_request", "operand.painted must be a boolean")
                })
            }
        });
    }

    let Some(mode) = mode else {
        let regions = shape_builder_regions(&operands, &painted, accuracy)?;
        if regions.is_empty() {
            return Ok(json!({ "status": "empty" }));
        }
        return Ok(json!({
            "status": "ok",
            "regions": regions.iter().map(|r| json!({
                "points": points_json(&r.points),
                "subpathCount": r.subpath_count,
                "bounds": bounds_json(&r.bounds),
                "sample": { "x": r.sample.0, "y": r.sample.1 },
                "cover": r.cover,
                "source": r.source,
            })).collect::<Vec<_>>(),
        }));
    };

    let raw_picks = obj
        .get("picks")
        .and_then(Value::as_array)
        .ok_or_else(|| err("invalid_request", "picks must be an array of [x, y]"))?;
    let mut picks = Vec::with_capacity(raw_picks.len());
    for p in raw_picks {
        let xy = p
            .as_array()
            .filter(|a| a.len() == 2)
            .and_then(|a| Some((a[0].as_f64()?, a[1].as_f64()?)))
            .ok_or_else(|| err("invalid_request", "picks must be an array of [x, y]"))?;
        picks.push(xy);
    }
    let Some(result) = shape_builder_build(&operands, &painted, accuracy, mode, &picks)? else {
        return Ok(json!({ "status": "empty" }));
    };
    let shape_json = |s: &Shape| {
        json!({
            "points": points_json(&s.points),
            "subpathCount": s.subpath_count,
            "bounds": bounds_json(&s.bounds),
        })
    };
    Ok(json!({
        "status": "ok",
        "picked": result.picked,
        "merged": result.merged.as_ref().map(|(s, source)| {
            let mut v = shape_json(s);
            v["source"] = json!(source);
            v
        }),
        "operands": result.operands.iter().map(|o| match o {
            OperandResult::Unchanged => json!({ "status": "unchanged" }),
            OperandResult::Removed => json!({ "status": "removed" }),
            OperandResult::Changed(s) => {
                let mut v = shape_json(s);
                v["status"] = json!("changed");
                v
            }
        }).collect::<Vec<_>>(),
    }))
}

/// Bounds of a path's anchors and control points (a safe over-estimate).
fn points_bounds(points: &[PathPoint]) -> Bounds {
    let mut b = Bounds {
        min_x: f64::INFINITY,
        min_y: f64::INFINITY,
        max_x: f64::NEG_INFINITY,
        max_y: f64::NEG_INFINITY,
    };
    for p in points {
        match p {
            PathPoint::Move { x, y } | PathPoint::Line { x, y } | PathPoint::Arc { x, y, .. } => {
                b.include(*x, *y)
            }
            PathPoint::Curve { x, y, cp1, cp2 } => {
                b.include(*x, *y);
                b.include(cp1.0, cp1.1);
                b.include(cp2.0, cp2.1);
            }
            PathPoint::Quadratic { x, y, cp } => {
                b.include(*x, *y);
                b.include(cp.0, cp.1);
            }
        }
    }
    b
}
