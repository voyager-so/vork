//! Boolean path operations (union / subtract / intersect / exclude), and
//! Illustrator's Pathfinder ops on top of them (`pathfinder.rs`), over the
//! TS `PathPoint` shape, backed by `flo_curves`' curve-exact bezier arithmetic.
//!
//! Every answer is checked against a second, polygon lane (`overlay.rs`:
//! `i_overlay`'s integer booleans on flattened operands). flo_curves loses or
//! doubles pieces on some inputs (Blob Brush outlines, near-duplicate shapes,
//! Pathfinder arrangements of self-intersecting paths), and when its answer and
//! the polygon lane's are not the same region, the polygon answer, refitted to
//! curves, is what ships. `quality_tests.rs` measures both lanes against a
//! raster.
//!
//! This is the single implementation: the editor reaches it through the
//! `moda-wasm` `boolean_paths` free function, and `moda-cli` / `moda-py` can
//! wrap the same `boolean_paths_json` entry point. There is no TS twin.
//!
//! Contract (shared with the TS client):
//!
//! ```jsonc
//! // request
//! { "op": "union"|"subtract"|"intersect"|"exclude",
//!   "operands": [ { "points": PathPoint[], "fillRule": "nonzero"|"evenodd" } ],
//!   "accuracy": 0.01 }
//! // response — always an envelope
//! { "status": "ok", "points": [...], "fillRule": "evenodd", "closed": true,
//!   "subpathCount": 2, "bounds": { "minX", "minY", "maxX", "maxY" } }
//! { "status": "empty" }
//! { "status": "error", "code": "...", "message": "...", "operandIndex": 1 }
//! ```
//!
//! Semantics: `subtract` is `operands[0] − union(rest)`; union / intersect /
//! exclude fold left→right. All operands must already share one coordinate
//! space — the caller rebases.
//!
//! Each operand is read under its own fill rule, matching what the renderers
//! paint: `nonzero` fills where the **signed winding number is non-zero** (so
//! an inner loop wound against its outer is a hole, as lyon's
//! `FillRule::NonZero` draws it, and as Figma paste and SVG import both
//! encode holes), and `evenodd` fills by **parity**. The result is always
//! emitted as `evenodd`, which both renderers draw correctly for compound
//! paths.
//!
//! SAFETY NOTE: the wasm release profile is `panic = "abort"`, so a panic inside
//! flo_curves would take down the shared engine instance, not just this call.
//! Degenerate-input scrubbing (`convert::points_to_subpaths`), the pre-solve
//! caps in `solve.rs` and the no-panic fuzz tests are therefore load-bearing, and
//! the native build additionally wraps the solve in `catch_unwind`. (It cannot
//! on wasm: `catch_unwind` needs `panic = "unwind"`, which wasm32 only has on
//! nightly with `-Zbuild-std` and the exception-handling proposal.) flo_curves
//! is vendored with its one known panic removed: its ray-collision sort used a
//! comparator that is not a total order, which Rust 1.81+ sort panics on
//! (`native/vendor/flo_curves/MODA-PATCH.md`, `panic_fuzz_tests.rs`).
//!
//! That `catch_unwind` is also why the fuzz cannot assert "the test did not
//! fail": on native it converts a panic into a well-formed
//! `{"status":"error","code":"solver_failed"}` envelope, so a run in which every
//! case panicked would look identical to a clean one. What the fuzz asserts is
//! that no case yields `solver_failed` — i.e. that no panic reached
//! `catch_unwind`, and so none would reach the abort on wasm.

mod bucket;
mod convert;
mod overlay;
mod pathfinder;
mod region_fill;
mod shape_builder;
mod solve;

#[cfg(test)]
mod large_tests;
#[cfg(test)]
mod panic_fuzz_tests;
#[cfg(test)]
mod pathfinder_tests;
#[cfg(test)]
mod perf_tests;
#[cfg(test)]
mod prop_tests;
#[cfg(test)]
mod quality_tests;
#[cfg(test)]
mod region_fill_tests;
#[cfg(test)]
mod shape_builder_tests;
#[cfg(test)]
mod tests;

use crate::value::{json, Value};

/// The TS `PathPoint` shape. Reuses the blend module's representation (same
/// JSON keys, same `apply_affine`) rather than defining a second one.
pub use crate::blend::BlendPathPoint as PathPoint;

/// Public so moda-engine can assert it still equals the tessellator's
/// `MAX_PATH_SEGMENTS`; see the constant's own doc comment.
pub use solve::MAX_RESULT_POINTS;

pub use pathfinder::{
    pathfinder_paths, pathfinder_paths_json, OperandPaint, PathfinderOp, PathfinderOutcome,
    PathfinderPiece, PathfinderRequest,
};

pub use region_fill::{
    region_fill_json, Arrangement, Drawing, FilledRegion, RegionAt, RegionSource, DEFAULT_OVERLAP,
    MAX_GAP,
};

pub use shape_builder::{
    shape_builder_build, shape_builder_json, shape_builder_regions, BuildMode, BuildResult,
    OperandResult, Shape, ShapeBuilderRegion,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    Union,
    Subtract,
    Intersect,
    Exclude,
}

impl BooleanOp {
    pub fn from_token(s: &str) -> Option<BooleanOp> {
        Some(match s {
            "union" => BooleanOp::Union,
            "subtract" => BooleanOp::Subtract,
            "intersect" => BooleanOp::Intersect,
            "exclude" => BooleanOp::Exclude,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillRule {
    NonZero,
    EvenOdd,
}

impl FillRule {
    pub fn from_token(s: &str) -> Option<FillRule> {
        Some(match s {
            "nonzero" => FillRule::NonZero,
            "evenodd" => FillRule::EvenOdd,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct BooleanOperand {
    pub points: Vec<PathPoint>,
    pub fill_rule: FillRule,
}

#[derive(Debug, Clone)]
pub struct BooleanRequest {
    pub op: BooleanOp,
    pub operands: Vec<BooleanOperand>,
    /// Absolute snapping tolerance in path units; clamped to [1e-4, 1].
    pub accuracy: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl Bounds {
    pub(crate) fn include(&mut self, x: f64, y: f64) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
    }
}

#[derive(Debug, Clone)]
pub enum BooleanOutcome {
    Shape {
        points: Vec<PathPoint>,
        subpath_count: usize,
        bounds: Bounds,
    },
    Empty,
}

/// A typed failure. `code` is the stable contract token the TS client switches
/// on; `message` is developer-facing only.
#[derive(Debug, Clone)]
pub struct BooleanError {
    pub code: String,
    pub message: String,
    pub operand_index: Option<usize>,
}

/// Solve a boolean request. Pure; never panics on native (the solver runs under
/// `catch_unwind`, surfacing a `solver_failed` error instead).
pub fn boolean_paths(request: &BooleanRequest) -> Result<BooleanOutcome, BooleanError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| solve::solve(request))) {
            Ok(result) => result,
            Err(_) => Err(BooleanError {
                code: "solver_failed".to_string(),
                message: "the boolean solver panicked on this input".to_string(),
                operand_index: None,
            }),
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        solve::solve(request)
    }
}

/// JSON-in / JSON-out wrapper. Always returns a valid envelope string — this is
/// the surface the wasm export and any future CLI binding wrap.
///
/// Guard order, cheapest first, each one refusing before the work the next
/// would pay for, and the last one refusing on the way out: request bytes
/// (here, before the parse) → operand and point counts → coordinates → total
/// cubics → pairwise work at every flo_curves entry → intermediate and result
/// cubics → result points → output finiteness. See `solve::solve`.
pub fn boolean_paths_json(request_json: &str) -> String {
    let outcome = solve::check_request_bytes(request_json.len())
        .and_then(|()| parse_request(request_json))
        .and_then(|request| boolean_paths(&request));
    envelope(outcome).to_string()
}

fn envelope(outcome: Result<BooleanOutcome, BooleanError>) -> Value {
    match outcome {
        Ok(BooleanOutcome::Shape {
            points,
            subpath_count,
            bounds,
        }) => json!({
            "status": "ok",
            "points": points.iter().map(PathPoint::to_json).collect::<Vec<_>>(),
            "fillRule": "evenodd",
            "closed": true,
            "subpathCount": subpath_count,
            "bounds": {
                "minX": bounds.min_x,
                "minY": bounds.min_y,
                "maxX": bounds.max_x,
                "maxY": bounds.max_y,
            },
        }),
        Ok(BooleanOutcome::Empty) => json!({ "status": "empty" }),
        Err(e) => {
            let mut obj = json!({
                "status": "error",
                "code": e.code,
                "message": e.message,
            });
            if let Some(i) = e.operand_index {
                obj["operandIndex"] = json!(i);
            }
            obj
        }
    }
}

fn invalid(message: &str) -> BooleanError {
    BooleanError {
        code: "invalid_request".to_string(),
        message: message.to_string(),
        operand_index: None,
    }
}

#[cfg(test)]
thread_local! {
    /// Bumped on entry to `parse_request`. Thread-local, so the guard tests can
    /// assert the byte gate refused *before* serde allocated a `Value` tree
    /// without racing the rest of the suite (10.1: pin the resource, not the
    /// error code — a post-parse refusal returns a code too).
    static PARSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn parse_count() -> usize {
    PARSES.with(std::cell::Cell::get)
}

fn parse_request(request_json: &str) -> Result<BooleanRequest, BooleanError> {
    #[cfg(test)]
    PARSES.with(|c| c.set(c.get() + 1));
    let value: Value = crate::value::from_str(request_json).map_err(|e| BooleanError {
        code: "invalid_json".to_string(),
        message: format!("request is not valid JSON: {e}"),
        operand_index: None,
    })?;
    let obj = value
        .as_object()
        .ok_or_else(|| invalid("request must be an object"))?;

    let op = obj
        .get("op")
        .and_then(Value::as_str)
        .and_then(BooleanOp::from_token)
        .ok_or_else(|| invalid("op must be one of union|subtract|intersect|exclude"))?;

    let operands = parse_operands(obj)?;
    let accuracy = parse_accuracy(obj)?;

    Ok(BooleanRequest {
        op,
        operands,
        accuracy,
    })
}

/// The `operands` array of a request, shared by the boolean and pathfinder
/// parsers. Fields other than `points` and `fillRule` are ignored here.
pub(crate) fn parse_operands(
    obj: &crate::value::Map,
) -> Result<Vec<BooleanOperand>, BooleanError> {
    let raw_operands = obj
        .get("operands")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("operands must be an array"))?;

    let mut operands = Vec::with_capacity(raw_operands.len());
    for (i, raw) in raw_operands.iter().enumerate() {
        let o = raw.as_object().ok_or_else(|| BooleanError {
            code: "invalid_request".to_string(),
            message: "operand must be an object".to_string(),
            operand_index: Some(i),
        })?;
        let raw_points = o
            .get("points")
            .and_then(Value::as_array)
            .ok_or_else(|| BooleanError {
                code: "invalid_request".to_string(),
                message: "operand.points must be an array".to_string(),
                operand_index: Some(i),
            })?;
        let mut points = Vec::with_capacity(raw_points.len());
        for p in raw_points {
            points.push(PathPoint::from_json(p).ok_or_else(|| BooleanError {
                code: "invalid_request".to_string(),
                message: "operand.points contains a malformed PathPoint".to_string(),
                operand_index: Some(i),
            })?);
        }
        let fill_rule = match o.get("fillRule") {
            None | Some(Value::Null) => FillRule::NonZero,
            Some(Value::String(s)) => FillRule::from_token(s).ok_or_else(|| BooleanError {
                code: "invalid_request".to_string(),
                message: "operand.fillRule must be nonzero or evenodd".to_string(),
                operand_index: Some(i),
            })?,
            Some(_) => {
                return Err(BooleanError {
                    code: "invalid_request".to_string(),
                    message: "operand.fillRule must be a string".to_string(),
                    operand_index: Some(i),
                })
            }
        };
        operands.push(BooleanOperand { points, fill_rule });
    }

    Ok(operands)
}

/// The optional `accuracy` field, shared by the boolean and pathfinder parsers.
pub(crate) fn parse_accuracy(
    obj: &crate::value::Map,
) -> Result<Option<f64>, BooleanError> {
    let accuracy = match obj.get("accuracy") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_f64()
                .ok_or_else(|| invalid("accuracy must be a number"))?,
        ),
    };
    Ok(accuracy)
}
