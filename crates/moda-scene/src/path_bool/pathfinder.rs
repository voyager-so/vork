//! Illustrator's Pathfinder ops — Divide, Trim, Merge, Crop, Outline, Minus
//! Back — on top of the boolean solver's normalization and budgets.
//!
//! Where the four boolean ops answer with ONE region, every pathfinder op
//! answers with a list of **pieces**, each tagged with the operand (`source`,
//! an index into the request, bottom-most first) whose paint it should wear.
//! The caller turns every piece into its own node and styles it from that
//! operand; the op-specific paint rules below are the caller's half of the
//! contract (Adobe Illustrator Help, "Pathfinder effects"):
//!
//! | op | geometry | paint of each piece |
//! |---|---|---|
//! | divide | every *filled* face of the planar arrangement (covered by at least one filled operand), one piece per connected face | fill **and stroke** of the topmost *filled* operand covering the face |
//! | trim | each filled operand minus the filled operands in front of it; same-colour objects are **not** merged | the operand's fill, strokes removed |
//! | merge | as trim, then visible parts of operands with the same `paintKey` that touch or overlap are merged | the fill, strokes removed |
//! | crop | as trim over every operand but the frontmost, each clipped to the frontmost; the frontmost itself is consumed | the operand's fill, strokes removed |
//! | outline | the edges of divide's filled faces, as open paths; an edge two faces share is emitted once, for the frontmost | the face's **fill colour as a stroke**, no fill |
//! | minusBack | the frontmost operand minus the union of the rest, as one piece | the frontmost operand's fill and stroke |
//!
//! "Filled" is the caller's call (`painted` per operand, default `true`): an
//! unfilled operand hides nothing, so trim / merge / crop drop it and it cuts
//! nothing, and divide only uses it as a cutting line: a face no filled operand
//! covers is removed, as Illustrator's Divide removes unpainted artwork (its
//! default Pathfinder option; checked against Illustrator 30 by
//! scripts/compare-illustrator/run.mjs --parity, pf-divide-unfilled-middle).
//! That matches Adobe's wording — divide gives "component filled faces", trim
//! "removes the part of a filled object that is hidden".
//!
//! Every piece is split into connected faces (an outer ring plus the holes
//! directly inside it) except minus back, which Illustrator also answers with
//! one compound object.
//!
//! The arrangement is built incrementally with `Kernel::split`
//! (`path_full_intersect` on the exact lane): each operand is collided against
//! every existing face it overlaps, splitting the face into "inside" and
//! "outside" and carrying the rest of the operand on to the next face. Faces
//! therefore stay disjoint, each knows exactly which operands cover it, and
//! every kernel entry goes through the kernel's `step` guard (`check_step` on
//! the exact lane, like the boolean folds).
//!
//! Like the booleans, the operands are bucketed first (`bucket.rs`): divide,
//! trim, merge and outline solve each group of operands that touch (directly
//! or through others) on its own, and crop and minus back drop the operands
//! that miss the frontmost. Each group runs on the exact lane (flo_curves)
//! while the request's `ExactBudget` lasts, and is checked against the polygon
//! lane (`overlay.rs`): the pieces must cover the op's region
//! (`expected_region`) without overlapping. If they do not, or past the
//! budget, the same ops run on the `Polygon` kernel, over the operands as the
//! polygon lane resolved them.
//!
//! Hiding is local too: an operand's visible part (trim, merge, crop) is it
//! minus the union of the filled operands in front of it *that touch it*, and
//! every union of several regions is folded as a balanced tree, so no step
//! carries the whole stack.
//!
//! Contract:
//!
//! ```jsonc
//! // request
//! { "op": "divide"|"trim"|"merge"|"crop"|"outline"|"minusBack",
//!   "operands": [ { "points": PathPoint[], "fillRule": "nonzero"|"evenodd",
//!                   "painted": true, "paintKey": "#ff0000" } ],
//!   "accuracy": 0.01 }
//! // response
//! { "status": "ok", "pieces": [ { "points": [...], "closed": true,
//!     "fillRule": "evenodd", "source": 1, "subpathCount": 1,
//!     "bounds": { "minX", "minY", "maxX", "maxY" } } ] }
//! { "status": "empty" }
//! { "status": "error", "code": "...", "message": "...", "operandIndex": 1 }
//! ```

use std::collections::HashMap;

use flo_curves::bezier::path::SimpleBezierPath;
use flo_curves::geo::Coord2;
use crate::value::{json, Value};

use super::bucket::{self, ExactBudget, Hull};
use super::convert::{flatten_subpath, subpath_to_points};
use super::overlay::{self, Contours, Exact, Kernel, Polygon};
use super::solve::{
    check_output_finite, check_result_points, convert, cubic_count, err, finish_region, normalize,
    region_bounds, region_is_finite, translate_region, Converted, Lane, Region,
};
use super::{
    parse_accuracy, parse_operands, BooleanError, BooleanOperand, BooleanOutcome, Bounds, FillRule,
    PathPoint,
};

/// Cap on the pieces one request may answer with: each becomes a scene node.
pub const MAX_PIECES: usize = 5_000;
/// Cap on the points summed over every piece. Each piece is separately held to
/// `MAX_RESULT_POINTS` (the tessellator's per-path cap); this bounds the
/// envelope as a whole.
pub const MAX_TOTAL_PIECE_POINTS: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathfinderOp {
    Divide,
    Trim,
    Merge,
    Crop,
    Outline,
    MinusBack,
}

impl PathfinderOp {
    pub fn from_token(s: &str) -> Option<PathfinderOp> {
        Some(match s {
            "divide" => PathfinderOp::Divide,
            "trim" => PathfinderOp::Trim,
            "merge" => PathfinderOp::Merge,
            "crop" => PathfinderOp::Crop,
            "outline" => PathfinderOp::Outline,
            "minusBack" => PathfinderOp::MinusBack,
            _ => return None,
        })
    }
}

/// What the solver needs to know about an operand's paint. The paint itself
/// stays on the TS side.
#[derive(Debug, Clone, Default)]
pub struct OperandPaint {
    /// Has a visible fill. Unfilled operands hide nothing (see module doc).
    pub painted: bool,
    /// Operands with equal keys are "the same colour" for merge. `None` never
    /// equals anything, including another `None`.
    pub key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PathfinderRequest {
    pub op: PathfinderOp,
    pub operands: Vec<BooleanOperand>,
    /// Parallel to `operands`.
    pub paint: Vec<OperandPaint>,
    pub accuracy: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct PathfinderPiece {
    pub points: Vec<PathPoint>,
    /// `false` only for outline's edge runs.
    pub closed: bool,
    /// Index of the operand whose paint this piece wears.
    pub source: usize,
    pub subpath_count: usize,
    pub bounds: Bounds,
}

#[derive(Debug, Clone)]
pub enum PathfinderOutcome {
    Pieces(Vec<PathfinderPiece>),
    Empty,
}

/// Solve a pathfinder request. Never panics on native (same `catch_unwind`
/// net as `boolean_paths`).
pub fn pathfinder_paths(request: &PathfinderRequest) -> Result<PathfinderOutcome, BooleanError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| solve(request))) {
            Ok(result) => result,
            Err(_) => Err(err(
                "solver_failed",
                "the pathfinder solver panicked on this input",
            )),
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        solve(request)
    }
}

/// JSON-in / JSON-out wrapper, the surface the wasm export wraps. Same guard
/// order as `boolean_paths_json`: the byte budget runs before the parse.
pub fn pathfinder_paths_json(request_json: &str) -> String {
    let outcome = super::solve::check_request_bytes(request_json.len())
        .and_then(|()| parse_request(request_json))
        .and_then(|request| pathfinder_paths(&request));
    envelope(outcome).to_string()
}

fn envelope(outcome: Result<PathfinderOutcome, BooleanError>) -> Value {
    match outcome {
        Ok(PathfinderOutcome::Pieces(pieces)) => json!({
            "status": "ok",
            "pieces": pieces.iter().map(|p| json!({
                "points": p.points.iter().map(PathPoint::to_json).collect::<Vec<_>>(),
                "closed": p.closed,
                "fillRule": "evenodd",
                "source": p.source,
                "subpathCount": p.subpath_count,
                "bounds": {
                    "minX": p.bounds.min_x,
                    "minY": p.bounds.min_y,
                    "maxX": p.bounds.max_x,
                    "maxY": p.bounds.max_y,
                },
            })).collect::<Vec<_>>(),
        }),
        Ok(PathfinderOutcome::Empty) => json!({ "status": "empty" }),
        Err(e) => {
            let mut obj = json!({ "status": "error", "code": e.code, "message": e.message });
            if let Some(i) = e.operand_index {
                obj["operandIndex"] = json!(i);
            }
            obj
        }
    }
}

fn parse_request(request_json: &str) -> Result<PathfinderRequest, BooleanError> {
    let value: Value = crate::value::from_str(request_json).map_err(|e| BooleanError {
        code: "invalid_json".to_string(),
        message: format!("request is not valid JSON: {e}"),
        operand_index: None,
    })?;
    let obj = value
        .as_object()
        .ok_or_else(|| err("invalid_request", "request must be an object"))?;
    let op = obj
        .get("op")
        .and_then(Value::as_str)
        .and_then(PathfinderOp::from_token)
        .ok_or_else(|| {
            err(
                "invalid_request",
                "op must be one of divide|trim|merge|crop|outline|minusBack",
            )
        })?;
    let operands = parse_operands(obj)?;
    let accuracy = parse_accuracy(obj)?;
    // `parse_operands` already proved `operands` is an array of objects.
    let raw = obj
        .get("operands")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut paint = Vec::with_capacity(raw.len());
    for (i, o) in raw.iter().enumerate() {
        let painted = match o.get("painted") {
            None | Some(Value::Null) => true,
            Some(Value::Bool(b)) => *b,
            Some(_) => {
                return Err(BooleanError {
                    operand_index: Some(i),
                    ..err("invalid_request", "operand.painted must be a boolean")
                })
            }
        };
        let key = match o.get("paintKey") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(_) => {
                return Err(BooleanError {
                    operand_index: Some(i),
                    ..err("invalid_request", "operand.paintKey must be a string")
                })
            }
        };
        paint.push(OperandPaint { painted, key });
    }
    Ok(PathfinderRequest {
        op,
        operands,
        paint,
        accuracy,
    })
}

// ---------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------

/// A solved piece still in solver space, before the exit guards.
struct RawPiece {
    region: Region,
    closed: bool,
    source: usize,
}

pub(crate) fn solve(request: &PathfinderRequest) -> Result<PathfinderOutcome, BooleanError> {
    solve_on(request, Lane::Checked)
}

/// `solve` on one lane (`solve::Lane`; only `Checked` ships).
pub(crate) fn solve_on(
    request: &PathfinderRequest,
    lane: Lane,
) -> Result<PathfinderOutcome, BooleanError> {
    if request.paint.len() != request.operands.len() {
        return Err(err("invalid_request", "paint must be parallel to operands"));
    }
    let Some(converted) = convert(&request.operands, request.accuracy)? else {
        return Ok(PathfinderOutcome::Empty);
    };
    let offset = converted.offset;
    let accuracy = converted.accuracy;
    let paint = &request.paint;
    let fills: Vec<FillRule> = request.operands.iter().map(|o| o.fill_rule).collect();

    let raw = if lane == Lane::Exact {
        let prepared = normalize(converted, &fills)?;
        run(request.op, &Exact(accuracy), &prepared.regions, paint)?.pieces
    } else {
        let flat = overlay::flatten(&converted.regions)?;
        let resolved = overlay::resolve_all(&flat, &fills);
        let hulls: Vec<Option<Hull>> = converted.regions.iter().map(bucket::hull).collect();
        let mut budget = ExactBudget::new();
        let mut raw: Vec<RawPiece> = Vec::new();
        for group in groups(request.op, &hulls) {
            let sub_paint: Vec<OperandPaint> = group.iter().map(|&i| paint[i].clone()).collect();
            let sub_resolved: Vec<Contours> = group.iter().map(|&i| resolved[i].clone()).collect();
            let polygon = || {
                let regions: Vec<Region> = sub_resolved.iter().map(overlay::lines).collect();
                let mut pieces = run(request.op, &Polygon(accuracy), &regions, &sub_paint)?.pieces;
                for piece in &mut pieces {
                    piece.region = flat.refit_lines(&piece.region, piece.closed);
                }
                Ok::<_, BooleanError>(pieces)
            };
            let pieces = if lane == Lane::Checked
                && budget.admit(group.iter().map(|&i| &converted.regions[i]))
            {
                let expected = expected_region(request.op, &sub_resolved, &sub_paint);
                let sub = Converted {
                    regions: group
                        .iter()
                        .map(|&i| converted.regions[i].clone())
                        .collect(),
                    offset,
                    accuracy,
                };
                let sub_fills: Vec<FillRule> = group.iter().map(|&i| fills[i]).collect();
                match normalize(sub, &sub_fills)
                    .and_then(|p| run(request.op, &Exact(accuracy), &p.regions, &sub_paint))
                {
                    Ok(ran) if ran.checks_out(&expected) => ran.pieces,
                    // A wrong answer, or a refusal from flo_curves' own guards.
                    _ => polygon()?,
                }
            } else {
                polygon()?
            };
            raw.extend(pieces.into_iter().map(|mut piece| {
                piece.source = group[piece.source];
                piece
            }));
            if raw.len() > MAX_PIECES {
                break;
            }
        }
        if request.op == PathfinderOp::Merge {
            // Merge paints every piece of one colour from that colour's topmost
            // visible operand, across the whole request, not per group.
            let mut top: HashMap<&str, usize> = HashMap::new();
            for piece in &raw {
                if let Some(key) = paint[piece.source].key.as_deref() {
                    let t = top.entry(key).or_insert(piece.source);
                    *t = (*t).max(piece.source);
                }
            }
            for piece in &mut raw {
                if let Some(key) = paint[piece.source].key.as_deref() {
                    piece.source = top[key];
                }
            }
        }
        // Each group's pieces are already back to front; so is the whole, once
        // the groups are interleaved by source (stable, so ties keep their order).
        raw.sort_by_key(|piece| piece.source);
        raw
    };

    if raw.len() > MAX_PIECES {
        return Err(err(
            "too_complex",
            format!("result has {} pieces (limit {MAX_PIECES})", raw.len()),
        ));
    }

    let mut pieces = Vec::with_capacity(raw.len());
    let mut total_points = 0usize;
    for piece in raw {
        let finished = if piece.closed {
            finish_region(piece.region, offset)?
        } else {
            finish_open(piece.region, offset)?
        };
        if let BooleanOutcome::Shape {
            points,
            subpath_count,
            bounds,
        } = finished
        {
            total_points += points.len();
            if total_points > MAX_TOTAL_PIECE_POINTS {
                return Err(err(
                    "too_complex",
                    format!("result has over {MAX_TOTAL_PIECE_POINTS} points across its pieces"),
                ));
            }
            pieces.push(PathfinderPiece {
                points,
                closed: piece.closed,
                source: piece.source,
                subpath_count,
                bounds,
            });
        }
    }
    if pieces.is_empty() {
        return Ok(PathfinderOutcome::Empty);
    }
    Ok(PathfinderOutcome::Pieces(pieces))
}

/// One op's pieces, and the closed regions they were cut from: the pieces
/// themselves, except for outline, whose open runs are the edges of divide's
/// faces (so the faces are what gets checked).
struct Ran {
    pieces: Vec<RawPiece>,
    cover: Vec<Region>,
}

impl Ran {
    /// The exact lane's answer is trusted when its regions are finite and cover
    /// the polygon lane's region for the op, without overlapping each other.
    fn checks_out(&self, expected: &Contours) -> bool {
        self.cover.iter().all(region_is_finite)
            && overlay::agrees(&self.cover.iter().collect::<Vec<_>>(), expected)
    }
}

fn run(
    op: PathfinderOp,
    k: &dyn Kernel,
    regions: &[Region],
    paint: &[OperandPaint],
) -> Result<Ran, BooleanError> {
    let pieces = match op {
        PathfinderOp::Divide => divide(k, regions, paint)?,
        PathfinderOp::Trim => trim(k, regions, paint)?,
        PathfinderOp::Merge => merge(k, regions, paint)?,
        PathfinderOp::Crop => crop(k, regions, paint)?,
        PathfinderOp::MinusBack => minus_back(k, regions)?,
        PathfinderOp::Outline => {
            let faces = divided_faces(k, regions, paint)?;
            let cover = faces.iter().map(|(r, _)| r.clone()).collect();
            return Ok(Ran {
                pieces: outline(faces, paint, k.accuracy()),
                cover,
            });
        }
    };
    let cover = pieces.iter().map(|p| p.region.clone()).collect();
    Ok(Ran { pieces, cover })
}

/// The groups `op` is solved over (see `bucket.rs`), each ascending. Divide,
/// trim, merge and outline: connected components, since operands that cannot
/// touch neither cut nor hide each other. Crop and minus back: the frontmost
/// operand (last in its group, as the ops expect) and the operands that touch
/// it; nothing when the frontmost is empty.
fn groups(op: PathfinderOp, hulls: &[Option<Hull>]) -> Vec<Vec<usize>> {
    match op {
        PathfinderOp::Divide | PathfinderOp::Trim | PathfinderOp::Merge | PathfinderOp::Outline => {
            bucket::components(hulls)
        }
        PathfinderOp::Crop | PathfinderOp::MinusBack => {
            let top = hulls.len() - 1;
            let rest: Vec<usize> = (0..top).collect();
            let group = bucket::touching(hulls, top, &rest);
            if group.is_empty() {
                Vec::new()
            } else {
                vec![group]
            }
        }
    }
}

/// Union of `parts`, folded as a balanced tree: pairs, then pairs of pairs, so
/// every kernel call sees two operands of about the same size. Each call is
/// behind the kernel's `step` guard, and every carried region behind `carry`.
fn union_all(k: &dyn Kernel, parts: Vec<Region>) -> Result<Region, BooleanError> {
    let mut level: Vec<Region> = parts.into_iter().filter(|p| !p.is_empty()).collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut it = level.into_iter();
        while let Some(a) = it.next() {
            next.push(match it.next() {
                Some(b) => {
                    k.step(&a, &b)?;
                    let sum = k.add(&a, &b);
                    k.carry(cubic_count(&sum))?;
                    sum
                }
                None => a,
            });
        }
        level = next;
    }
    Ok(level.pop().unwrap_or_default())
}

/// What an op's pieces must cover between them, on the polygon lane: the union
/// of the operands for outline's faces; of the filled ones for divide, trim and
/// merge; the filled ones behind the frontmost, clipped to it, for
/// crop; the frontmost minus the rest for minus back.
fn expected_region(op: PathfinderOp, resolved: &[Contours], paint: &[OperandPaint]) -> Contours {
    let top = resolved.len() - 1;
    let filled = |end: usize| {
        overlay::union_all(
            resolved[..end]
                .iter()
                .zip(paint)
                .filter(|(_, p)| p.painted)
                .map(|(r, _)| r),
        )
    };
    match op {
        PathfinderOp::Outline => overlay::union_all(resolved),
        PathfinderOp::Divide | PathfinderOp::Trim | PathfinderOp::Merge => filled(resolved.len()),
        PathfinderOp::Crop => overlay::intersect(&filled(top), &resolved[top]),
        PathfinderOp::MinusBack => {
            overlay::difference(&resolved[top], &overlay::union_all(&resolved[..top]))
        }
    }
}

/// `finish_region` for an open run: same exit guards, but a two-point line is
/// a legitimate answer and the last segment is never an implied close.
fn finish_open(
    mut region: Region,
    (off_x, off_y): (f64, f64),
) -> Result<BooleanOutcome, BooleanError> {
    translate_region(&mut region, off_x, off_y);
    let mut points: Vec<PathPoint> = Vec::new();
    for run in &region {
        // `subpath_to_points` drops a final straight segment that returns to
        // the start; an open run never returns to its start (a run that would
        // is emitted closed), so the whole run survives.
        subpath_to_points(run, &mut points);
    }
    if points.len() < 2 {
        return Ok(BooleanOutcome::Empty);
    }
    check_result_points(points.len())?;
    check_output_finite(&points)?;
    let Some(bounds) = region_bounds(&region) else {
        return Ok(BooleanOutcome::Empty);
    };
    super::solve::check_bounds_finite(&bounds)?;
    Ok(BooleanOutcome::Shape {
        points,
        subpath_count: region.len(),
        bounds,
    })
}

/// The planar arrangement of every operand: disjoint faces, each with the
/// ascending list of operands that cover it. A face here may still be several
/// disconnected pieces; `split_faces` separates them.
pub(crate) fn arrangement(
    k: &dyn Kernel,
    regions: &[Region],
) -> Result<Vec<(Region, Vec<usize>)>, BooleanError> {
    let mut faces: Vec<(Region, Vec<usize>)> = Vec::new();
    for (i, operand) in regions.iter().enumerate() {
        if operand.is_empty() {
            continue;
        }
        let mut rest = operand.clone();
        let mut next: Vec<(Region, Vec<usize>)> = Vec::with_capacity(faces.len() * 2 + 1);
        for (face, cover) in faces {
            if rest.is_empty() || !hulls_overlap(&face, &rest) {
                next.push((face, cover));
                continue;
            }
            k.step(&face, &rest)?;
            let (inside, face_outside, rest_outside) = k.split(&face, &rest);
            if !inside.is_empty() {
                let mut inside_cover = cover.clone();
                inside_cover.push(i);
                next.push((inside, inside_cover));
            }
            if !face_outside.is_empty() {
                next.push((face_outside, cover));
            }
            rest = rest_outside;
        }
        if !rest.is_empty() {
            next.push((rest, vec![i]));
        }
        let carried: usize = next.iter().map(|(r, _)| cubic_count(r)).sum();
        k.carry(carried)?;
        if next.len() > MAX_PIECES {
            return Err(err(
                "too_complex",
                format!("arrangement has more than {MAX_PIECES} faces"),
            ));
        }
        faces = next;
    }
    Ok(faces)
}

/// The operand whose paint a face wears: the topmost *filled* operand covering
/// it, or the topmost covering operand when none is filled.
fn face_owner(cover: &[usize], paint: &[OperandPaint]) -> usize {
    cover
        .iter()
        .rev()
        .find(|&&i| paint[i].painted)
        .or_else(|| cover.last())
        .copied()
        .unwrap_or(0)
}

/// Faces of the arrangement, split into connected pieces, each with its owner.
/// Sorted by owner (stable), so a caller stacking them in order paints front
/// over back.
fn divided_faces(
    k: &dyn Kernel,
    regions: &[Region],
    paint: &[OperandPaint],
) -> Result<Vec<(Region, usize)>, BooleanError> {
    let mut out = Vec::new();
    for (region, cover) in arrangement(k, regions)? {
        let owner = face_owner(&cover, paint);
        for face in split_faces(region, k.accuracy()) {
            out.push((face, owner));
        }
    }
    out.sort_by_key(|(_, owner)| *owner);
    Ok(out)
}

fn divide(
    k: &dyn Kernel,
    regions: &[Region],
    paint: &[OperandPaint],
) -> Result<Vec<RawPiece>, BooleanError> {
    Ok(divided_faces(k, regions, paint)?
        .into_iter()
        // Unpainted artwork is removed: a face only unfilled operands cover.
        .filter(|(_, owner)| paint[*owner].painted)
        .map(|(region, source)| RawPiece {
            region,
            closed: true,
            source,
        })
        .collect())
}

/// For each operand in `0..end`, the part of it no *filled* operand in
/// `0..end` in front of it hides. `None` for unfilled or empty operands.
///
/// Only the filled operands in front that touch it can hide any of it, so each
/// operand is cut by the union of just those: work linear in the operand
/// count for artwork where each shape overlaps a few neighbours, where a
/// running union of everything in front grew with the whole stack.
fn visible_parts(
    k: &dyn Kernel,
    regions: &[Region],
    paint: &[OperandPaint],
    end: usize,
) -> Result<Vec<Option<Region>>, BooleanError> {
    let hulls: Vec<Option<Hull>> = regions[..end].iter().map(bucket::hull).collect();
    let mut out: Vec<Option<Region>> = vec![None; end];
    for i in 0..end {
        let own = &regions[i];
        let Some(hull) = hulls[i] else { continue };
        if !paint[i].painted {
            continue;
        }
        let front: Vec<Region> = (i + 1..end)
            .filter(|&j| paint[j].painted && hulls[j].is_some_and(|h| h.overlaps(&hull)))
            .map(|j| regions[j].clone())
            .collect();
        let cover = union_all(k, front)?;
        let visible = if cover.is_empty() {
            own.clone()
        } else {
            k.step(own, &cover)?;
            k.sub(own, &cover)
        };
        if !visible.is_empty() {
            out[i] = Some(visible);
        }
    }
    Ok(out)
}

fn split_into_pieces(parts: Vec<(Region, usize)>, accuracy: f64) -> Vec<RawPiece> {
    let mut out = Vec::new();
    for (region, source) in parts {
        for face in split_faces(region, accuracy) {
            out.push(RawPiece {
                region: face,
                closed: true,
                source,
            });
        }
    }
    out
}

fn trim(
    k: &dyn Kernel,
    regions: &[Region],
    paint: &[OperandPaint],
) -> Result<Vec<RawPiece>, BooleanError> {
    let parts = visible_parts(k, regions, paint, regions.len())?
        .into_iter()
        .enumerate()
        .filter_map(|(i, r)| r.map(|r| (r, i)))
        .collect();
    Ok(split_into_pieces(parts, k.accuracy()))
}

fn merge(
    k: &dyn Kernel,
    regions: &[Region],
    paint: &[OperandPaint],
) -> Result<Vec<RawPiece>, BooleanError> {
    let visible = visible_parts(k, regions, paint, regions.len())?;
    // Group by paint key; each group wears its topmost member's paint.
    let mut groups: Vec<(Vec<Region>, usize)> = Vec::new();
    let mut by_key: HashMap<&str, usize> = HashMap::new();
    for (i, part) in visible.into_iter().enumerate() {
        let Some(part) = part else { continue };
        match paint[i].key.as_deref() {
            Some(key) if by_key.contains_key(key) => {
                let g = &mut groups[by_key[key]];
                g.0.push(part);
                g.1 = i;
            }
            key => {
                if let Some(k) = key {
                    by_key.insert(k, groups.len());
                }
                groups.push((vec![part], i));
            }
        }
    }
    let mut parts: Vec<(Region, usize)> = Vec::with_capacity(groups.len());
    for (members, top) in groups {
        parts.push((union_all(k, members)?, top));
    }
    parts.sort_by_key(|(_, i)| *i);
    Ok(split_into_pieces(parts, k.accuracy()))
}

fn crop(
    k: &dyn Kernel,
    regions: &[Region],
    paint: &[OperandPaint],
) -> Result<Vec<RawPiece>, BooleanError> {
    let top = regions.len() - 1;
    let clip = &regions[top];
    if clip.is_empty() {
        return Ok(Vec::new());
    }
    let mut parts = Vec::new();
    for (i, part) in visible_parts(k, regions, paint, top)?
        .into_iter()
        .enumerate()
    {
        let Some(part) = part else { continue };
        if !hulls_overlap(&part, clip) {
            continue;
        }
        k.step(&part, clip)?;
        let kept = k.intersect(&part, clip);
        if !kept.is_empty() {
            parts.push((kept, i));
        }
    }
    Ok(split_into_pieces(parts, k.accuracy()))
}

/// The frontmost operand minus the union of the rest (`solve_subtract`'s
/// region, on either kernel).
fn minus_back(k: &dyn Kernel, regions: &[Region]) -> Result<Vec<RawPiece>, BooleanError> {
    let top = regions.len() - 1;
    let mut region = regions[top].clone();
    if region.is_empty() {
        return Ok(Vec::new());
    }
    let cutter = union_all(k, regions[..top].to_vec())?;
    if !cutter.is_empty() {
        k.step(&region, &cutter)?;
        region = k.sub(&region, &cutter);
    }
    if region.is_empty() {
        return Ok(Vec::new());
    }
    Ok(vec![RawPiece {
        region,
        closed: true,
        source: top,
    }])
}

// ---------------------------------------------------------------------------
// Outline
// ---------------------------------------------------------------------------

type Cubic = (Coord2, Coord2, Coord2, Coord2);

fn near(a: Coord2, b: Coord2, tol: f64) -> bool {
    (a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol
}

fn same_edge(a: &Cubic, b: &Cubic, tol: f64) -> bool {
    let forward =
        near(a.0, b.0, tol) && near(a.1, b.1, tol) && near(a.2, b.2, tol) && near(a.3, b.3, tol);
    let reverse =
        near(a.0, b.3, tol) && near(a.1, b.2, tol) && near(a.2, b.1, tol) && near(a.3, b.0, tol);
    forward || reverse
}

/// Divide's filled faces, then their edges: every ring is cut into maximal
/// runs of the segments this face owns. A segment two faces share is owned by
/// the face with the frontmost owner (the lower face index on a tie), so it is
/// emitted once, in the colour Illustrator's outline gives it ("the fill of the
/// frontmost object in that area").
///
/// Shared edges are matched segment-for-segment. Two faces built by different
/// `path_full_intersect` calls can split a shared edge at different points,
/// and such an edge then comes out twice; since pieces are sorted back to
/// front the frontmost copy still paints on top, so the result looks right.
fn outline(faces: Vec<(Region, usize)>, paint: &[OperandPaint], accuracy: f64) -> Vec<RawPiece> {
    let faces: Vec<(Region, usize)> = faces
        .into_iter()
        .filter(|(_, owner)| paint[*owner].painted)
        .collect();

    // Every segment of every ring, as (face, ring, index, cubic).
    let mut segments: Vec<(usize, usize, usize, Cubic)> = Vec::new();
    for (f, (region, _)) in faces.iter().enumerate() {
        for (r, (start, segs)) in region.iter().enumerate() {
            let mut p0 = *start;
            for (s, (c1, c2, p3)) in segs.iter().enumerate() {
                segments.push((f, r, s, (p0, *c1, *c2, *p3)));
                p0 = *p3;
            }
        }
    }

    let tol = (accuracy * 4.0).max(1e-6);
    let cell = tol * 4.0;
    let q = |c: Coord2| ((c.0 / cell).round() as i64, (c.1 / cell).round() as i64);
    let mut buckets: HashMap<((i64, i64), (i64, i64)), Vec<usize>> = HashMap::new();
    for (k, (_, _, _, cubic)) in segments.iter().enumerate() {
        let (a, b) = (q(cubic.0), q(cubic.3));
        let key = if a <= b { (a, b) } else { (b, a) };
        buckets.entry(key).or_default().push(k);
    }
    let mut dropped = vec![false; segments.len()];
    for ids in buckets.values() {
        for (x, &a) in ids.iter().enumerate() {
            for &b in &ids[x + 1..] {
                let (fa, fb) = (segments[a].0, segments[b].0);
                if fa == fb || !same_edge(&segments[a].3, &segments[b].3, tol) {
                    continue;
                }
                let (oa, ob) = (faces[fa].1, faces[fb].1);
                // Keep the frontmost owner's copy; on a tie, the earlier face's.
                let loser = if oa > ob || (oa == ob && fa < fb) {
                    b
                } else {
                    a
                };
                dropped[loser] = true;
            }
        }
    }

    let mut kept: HashMap<(usize, usize), Vec<bool>> = HashMap::new();
    for (k, (f, r, s, _)) in segments.iter().enumerate() {
        let entry = kept
            .entry((*f, *r))
            .or_insert_with(|| vec![true; faces[*f].0[*r].1.len()]);
        entry[*s] = !dropped[k];
    }

    let mut out = Vec::new();
    for (f, (region, owner)) in faces.iter().enumerate() {
        for (r, ring) in region.iter().enumerate() {
            let Some(flags) = kept.get(&(f, r)) else {
                continue;
            };
            for (run, closed) in ring_runs(ring, flags) {
                out.push(RawPiece {
                    region: vec![run],
                    closed,
                    source: *owner,
                });
            }
        }
    }
    out
}

/// Maximal cyclic runs of kept segments. A fully kept ring comes back whole
/// and closed.
fn ring_runs(ring: &SimpleBezierPath, kept: &[bool]) -> Vec<(SimpleBezierPath, bool)> {
    let (start, segs) = ring;
    let n = segs.len();
    if n == 0 || kept.iter().all(|k| !k) {
        return Vec::new();
    }
    if kept.iter().all(|k| *k) {
        return vec![(ring.clone(), true)];
    }
    let anchor = |i: usize| if i == 0 { *start } else { segs[i - 1].2 };
    // Begin just after a dropped segment so no run wraps the seam.
    let first = (0..n).find(|&i| !kept[i]).unwrap_or(0);
    let mut runs = Vec::new();
    let mut current: Option<SimpleBezierPath> = None;
    for step in 1..=n {
        let i = (first + step) % n;
        if kept[i] {
            let run = current.get_or_insert_with(|| (anchor(i), Vec::new()));
            run.1.push(segs[i]);
        } else if let Some(run) = current.take() {
            runs.push((run, false));
        }
    }
    if let Some(run) = current {
        runs.push((run, false));
    }
    runs
}

// ---------------------------------------------------------------------------
// Face splitting
// ---------------------------------------------------------------------------

fn hull_bounds(region: &Region) -> (f64, f64, f64, f64) {
    let mut b = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    let mut add = |c: &Coord2| {
        b.0 = b.0.min(c.0);
        b.1 = b.1.min(c.1);
        b.2 = b.2.max(c.0);
        b.3 = b.3.max(c.1);
    };
    for (start, segs) in region {
        add(start);
        for (c1, c2, p) in segs {
            add(c1);
            add(c2);
            add(p);
        }
    }
    b
}

/// Control-hull boxes overlap (touching counts): a cheap reject before a
/// flo_curves call that would only report "disjoint".
fn hulls_overlap(a: &Region, b: &Region) -> bool {
    let (a0, a1, a2, a3) = hull_bounds(a);
    let (b0, b1, b2, b3) = hull_bounds(b);
    a0 <= b2 && b0 <= a2 && a1 <= b3 && b1 <= a3
}

fn poly_area(c: &[(f64, f64)]) -> f64 {
    let mut total = 0.0;
    for i in 0..c.len() {
        let (x0, y0) = c[i];
        let (x1, y1) = c[(i + 1) % c.len()];
        total += x0 * y1 - x1 * y0;
    }
    total / 2.0
}

fn poly_perimeter(c: &[(f64, f64)]) -> f64 {
    let mut total = 0.0;
    for i in 0..c.len() {
        let (x0, y0) = c[i];
        let (x1, y1) = c[(i + 1) % c.len()];
        total += ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
    }
    total
}

fn point_in_poly(c: &[(f64, f64)], p: (f64, f64)) -> bool {
    let mut inside = false;
    let mut j = c.len() - 1;
    for i in 0..c.len() {
        let (xi, yi) = c[i];
        let (xj, yj) = c[j];
        if (yi > p.1) != (yj > p.1) && p.0 < (xj - xi) * (p.1 - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// A point just inside a simple ring, off its boundary: a step along the
/// inward bisector at the lowest-then-leftmost vertex, which is always convex.
/// Rings that meet at a vertex would make the vertex itself ambiguous.
fn interior_probe(c: &[(f64, f64)]) -> (f64, f64) {
    let n = c.len();
    let mut k = 0;
    for i in 1..n {
        if c[i].1 < c[k].1 || (c[i].1 == c[k].1 && c[i].0 < c[k].0) {
            k = i;
        }
    }
    let v = c[k];
    let unit = |p: (f64, f64)| {
        let (dx, dy) = (p.0 - v.0, p.1 - v.1);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-12 {
            (0.0, 0.0)
        } else {
            (dx / len, dy / len)
        }
    };
    let (ax, ay) = unit(c[(k + n - 1) % n]);
    let (bx, by) = unit(c[(k + 1) % n]);
    let (mut dx, mut dy) = (ax + bx, ay + by);
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-9 {
        // Straight through the vertex: everything else lies at larger y.
        (dx, dy) = (0.0, 1.0);
    } else {
        (dx, dy) = (dx / len, dy / len);
    }
    let step = 1e-4;
    (v.0 + dx * step, v.1 + dy * step)
}

/// Split a region (a set of even-odd rings) into its connected faces: each
/// outer ring with the holes directly inside it. Nesting is measured, not read
/// off the winding, because flo_curves does not promise a hole orientation.
///
/// Faces thinner on average than two `accuracy` widths are dropped: they are
/// the slivers snapping leaves along edges two operands share, not faces a
/// user drew.
pub(crate) fn split_faces(region: Region, accuracy: f64) -> Vec<Region> {
    let polys: Vec<Vec<(f64, f64)>> = region.iter().map(|r| flatten_subpath(r, 8)).collect();
    let n = region.len();
    let valid: Vec<bool> = polys.iter().map(|p| p.len() >= 3).collect();
    let probes: Vec<(f64, f64)> = polys
        .iter()
        .map(|p| {
            if p.len() >= 3 {
                interior_probe(p)
            } else {
                (0.0, 0.0)
            }
        })
        .collect();
    let areas: Vec<f64> = polys.iter().map(|p| poly_area(p).abs()).collect();
    let depth: Vec<usize> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i && valid[i] && valid[j] && point_in_poly(&polys[j], probes[i]))
                .count()
        })
        .collect();

    let outers: Vec<usize> = (0..n).filter(|&i| valid[i] && depth[i] % 2 == 0).collect();
    let mut holes_of: HashMap<usize, Vec<usize>> = HashMap::new();
    for h in (0..n).filter(|&i| valid[i] && depth[i] % 2 == 1) {
        let parent = outers
            .iter()
            .copied()
            .filter(|&o| depth[o] + 1 == depth[h] && point_in_poly(&polys[o], probes[h]))
            .min_by(|&a, &b| areas[a].total_cmp(&areas[b]));
        if let Some(o) = parent {
            holes_of.entry(o).or_default().push(h);
        }
    }

    let mut faces = Vec::new();
    for o in outers {
        let holes = holes_of.remove(&o).unwrap_or_default();
        let area = areas[o] - holes.iter().map(|&h| areas[h]).sum::<f64>();
        let perimeter = poly_perimeter(&polys[o])
            + holes
                .iter()
                .map(|&h| poly_perimeter(&polys[h]))
                .sum::<f64>();
        if area <= accuracy * perimeter {
            continue;
        }
        let mut face = vec![region[o].clone()];
        face.extend(holes.into_iter().map(|h| region[h].clone()));
        faces.push(face);
    }
    faces
}
