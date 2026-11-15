//! The JSON-in / JSON-out entry point the editor and the SDK reach through
//! `moda-wasm` (`vector_network`), in the shape of `boolean_paths_json`: the
//! reply is always an envelope, so a rejected request never throws.
//!
//! Request: `{ "op", "network"?, "tolerance"?, … }`:
//!
//! | op             | extra fields                     | reply (besides `status: "ok"`)                |
//! |----------------|----------------------------------|-----------------------------------------------|
//! | `normalize`    |                                  | `network` (validated)                         |
//! | `planarize`    |                                  | `network`, `changed`                          |
//! | `toggleFill`   | `point: {x, y}`                  | `network`, `result: filled / unfilled / none` |
//! | `fillAllFaces` |                                  | `network`                                     |
//! | `faces`        |                                  | `faces: [{ loops, area }]`                    |
//! | `toPaths`      |                                  | `regions: [{ region, points, fillRule }]`, `openRuns: [{ points, closed }]` |
//! | `fromPath`     | `points`, `closed`, `fillRule`   | `network`                                     |
//!
//! Errors: `{ "status": "error", "code", "message" }`.

use crate::value::{json, Value};

use super::{FillToggle, Vector, VectorNetwork, DEFAULT_TOLERANCE};
use crate::blend::BlendPathPoint;
use crate::path_bool::FillRule;
use crate::types::RawNode;

/// Requests past this size are refused before parsing (a hostile paste or a
/// runaway agent must not stall the editor).
const MAX_REQUEST_BYTES: usize = 8 << 20;

fn error(code: &str, message: impl Into<String>) -> String {
    json!({ "status": "error", "code": code, "message": message.into() }).to_string()
}

fn fill_rule_str(r: FillRule) -> &'static str {
    match r {
        FillRule::NonZero => "nonzero",
        FillRule::EvenOdd => "evenodd",
    }
}

/// The node's `vectorNetwork`, parsed and validated. `None` when absent or
/// invalid (an invalid network renders as the node's derived `points`).
pub fn network_from_node(node: &RawNode) -> Option<VectorNetwork> {
    let v = node.json_field("vectorNetwork")?;
    if v.is_null() {
        return None;
    }
    VectorNetwork::from_json_value(v).ok()
}

/// The derived point list for a network: every region's loops as closed
/// subpaths, then the segments in no region as runs. This is what the editor
/// stores in a network path's `points` (with `closed` = the network has a
/// region), so every lane that reads `points` sees the network's geometry.
pub fn derived_points(net: &VectorNetwork) -> Vec<BlendPathPoint> {
    let paths = net.to_paths();
    let mut out = Vec::new();
    // With no region the node is open, so a closed run spells out its closing side.
    let explicit_close = paths.regions.is_empty();
    for r in paths.regions {
        out.extend(r.points);
    }
    for run in paths.open_runs {
        let first = run.points.first().map(point_xy);
        let last = run.points.last().map(point_xy);
        out.extend(run.points);
        if let (true, true, Some(f), Some(l)) = (explicit_close, run.closed, first, last) {
            if (f.0 - l.0).hypot(f.1 - l.1) > 1e-9 {
                out.push(BlendPathPoint::Line { x: f.0, y: f.1 });
            }
        }
    }
    out
}

fn point_xy(p: &BlendPathPoint) -> (f64, f64) {
    match p {
        BlendPathPoint::Move { x, y }
        | BlendPathPoint::Line { x, y }
        | BlendPathPoint::Curve { x, y, .. }
        | BlendPathPoint::Quadratic { x, y, .. }
        | BlendPathPoint::Arc { x, y, .. } => (*x, *y),
    }
}

pub fn vector_network_json(request_json: &str) -> String {
    if request_json.len() > MAX_REQUEST_BYTES {
        return error("too_large", "vector network request is too large");
    }
    let req: Value = match crate::value::from_str(request_json) {
        Ok(v) => v,
        Err(e) => return error("bad_json", e.to_string()),
    };
    let tol = req
        .get("tolerance")
        .and_then(Value::as_f64)
        .filter(|t| t.is_finite() && *t > 0.0)
        .unwrap_or(DEFAULT_TOLERANCE);
    let op = req.get("op").and_then(Value::as_str).unwrap_or("");
    let network = || -> Result<VectorNetwork, String> {
        let v = req
            .get("network")
            .ok_or_else(|| "missing network".to_string())?;
        VectorNetwork::from_json_value(v).map_err(|e| e.to_string())
    };
    macro_rules! net {
        () => {
            match network() {
                Ok(n) => n,
                Err(m) => return error("bad_network", m),
            }
        };
    }
    match op {
        "normalize" => {
            let n = net!();
            json!({ "status": "ok", "network": n.to_json_value() }).to_string()
        }
        "planarize" => {
            let mut n = net!();
            let report = n.planarize(tol);
            json!({ "status": "ok", "network": n.to_json_value(), "changed": report.changed() })
                .to_string()
        }
        "toggleFill" => {
            let mut n = net!();
            let p = req.get("point");
            let x = p.and_then(|p| p.get("x")).and_then(Value::as_f64);
            let y = p.and_then(|p| p.get("y")).and_then(Value::as_f64);
            let (Some(x), Some(y)) = (x, y) else {
                return error("bad_point", "toggleFill needs point {x, y}");
            };
            let before = n.clone();
            let result = n.toggle_fill_at(Vector::new(x, y), tol);
            let (label, region) = match result {
                FillToggle::Filled { region } => ("filled", Some(region)),
                FillToggle::Unfilled => ("unfilled", None),
                FillToggle::NoFace => ("none", None),
            };
            // A click outside every closed area changes nothing, not even the
            // planarisation the toggle ran first.
            let out = if label == "none" { before } else { n };
            json!({
                "status": "ok",
                "network": out.to_json_value(),
                "result": label,
                "region": region,
            })
            .to_string()
        }
        "fillAllFaces" => {
            let mut n = net!();
            n.fill_all_faces(tol);
            json!({ "status": "ok", "network": n.to_json_value() }).to_string()
        }
        "faces" => {
            let mut n = net!();
            n.planarize(tol);
            let faces: Vec<Value> = n
                .faces(tol)
                .iter()
                .map(|f| json!({ "loops": f.segment_loops(), "area": f.area }))
                .collect();
            json!({ "status": "ok", "network": n.to_json_value(), "faces": faces }).to_string()
        }
        "toPaths" => {
            let n = net!();
            let paths = n.to_paths();
            let regions: Vec<Value> = paths
                .regions
                .iter()
                .map(|r| {
                    json!({
                        "region": r.region,
                        "points": r.points.iter().map(BlendPathPoint::to_json).collect::<Vec<_>>(),
                        "fillRule": fill_rule_str(r.fill_rule),
                    })
                })
                .collect();
            let runs: Vec<Value> = paths
                .open_runs
                .iter()
                .map(|r| {
                    json!({
                        "points": r.points.iter().map(BlendPathPoint::to_json).collect::<Vec<_>>(),
                        "closed": r.closed,
                    })
                })
                .collect();
            json!({ "status": "ok", "regions": regions, "openRuns": runs }).to_string()
        }
        "fromPath" => {
            let points: Vec<BlendPathPoint> = req
                .get("points")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(BlendPathPoint::from_json).collect())
                .unwrap_or_default();
            let closed = req.get("closed").and_then(Value::as_bool).unwrap_or(false);
            let rule = match req.get("fillRule").and_then(Value::as_str) {
                Some("evenodd") => FillRule::EvenOdd,
                _ => FillRule::NonZero,
            };
            let n = VectorNetwork::from_path(&points, closed, rule, tol);
            json!({ "status": "ok", "network": n.to_json_value() }).to_string()
        }
        other => error("bad_op", format!("unknown vector network op {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(v: Value) -> Value {
        crate::value::from_str(&vector_network_json(&v.to_string())).unwrap()
    }

    fn plus() -> Value {
        json!({
            "vertices": [{"x": 0, "y": 50}, {"x": 100, "y": 50}, {"x": 50, "y": 0}, {"x": 50, "y": 100}],
            "segments": [{"start": 0, "end": 1}, {"start": 2, "end": 3}],
        })
    }

    #[test]
    fn planarize_splits_the_plus_at_its_hub() {
        let r = call(json!({"op": "planarize", "network": plus()}));
        assert_eq!(r["status"], "ok");
        assert_eq!(r["changed"], true);
        assert_eq!(r["network"]["vertices"].as_array().unwrap().len(), 5);
        assert_eq!(r["network"]["segments"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn toggle_fill_fills_then_unfills_a_square() {
        let sq = json!({
            "vertices": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}, {"x": 0, "y": 10}],
            "segments": [{"start": 0, "end": 1}, {"start": 1, "end": 2}, {"start": 2, "end": 3}, {"start": 3, "end": 0}],
        });
        let r = call(json!({"op": "toggleFill", "network": sq, "point": {"x": 5, "y": 5}}));
        assert_eq!(r["result"], "filled");
        assert_eq!(r["network"]["regions"].as_array().unwrap().len(), 1);
        let r2 =
            call(json!({"op": "toggleFill", "network": r["network"], "point": {"x": 5, "y": 5}}));
        assert_eq!(r2["result"], "unfilled");
        assert!(r2["network"].get("regions").is_none());
        let miss = call(json!({"op": "toggleFill", "network": sq, "point": {"x": 50, "y": 5}}));
        assert_eq!(miss["result"], "none");
    }

    #[test]
    fn rejects_bad_input_with_an_envelope() {
        assert_eq!(call(json!({"op": "nope"}))["status"], "error");
        assert_eq!(call(json!({"op": "planarize"}))["code"], "bad_network");
        let dangling =
            json!({"vertices": [{"x": 0, "y": 0}], "segments": [{"start": 0, "end": 3}]});
        assert_eq!(
            call(json!({"op": "normalize", "network": dangling}))["code"],
            "bad_network"
        );
        let r: Value = crate::value::from_str(&vector_network_json("{not json")).unwrap();
        assert_eq!(r["code"], "bad_json");
    }

    #[test]
    fn from_path_and_to_paths_round_trip() {
        let pts = json!([
            {"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 10, "y": 0},
            {"type": "line", "x": 10, "y": 10}, {"type": "line", "x": 0, "y": 10}
        ]);
        let r =
            call(json!({"op": "fromPath", "points": pts, "closed": true, "fillRule": "evenodd"}));
        let net = &r["network"];
        assert_eq!(net["segments"].as_array().unwrap().len(), 4);
        assert_eq!(net["regions"][0]["windingRule"], "EVENODD");
        let p = call(json!({"op": "toPaths", "network": net}));
        assert_eq!(p["regions"][0]["fillRule"], "evenodd");
        assert_eq!(p["openRuns"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn a_holed_face_fills_even_odd() {
        // Square with a square hole: the ring face has two loops.
        let net = json!({
            "vertices": [
                {"x": 0, "y": 0}, {"x": 30, "y": 0}, {"x": 30, "y": 30}, {"x": 0, "y": 30},
                {"x": 10, "y": 10}, {"x": 20, "y": 10}, {"x": 20, "y": 20}, {"x": 10, "y": 20}
            ],
            "segments": [
                {"start": 0, "end": 1}, {"start": 1, "end": 2}, {"start": 2, "end": 3}, {"start": 3, "end": 0},
                {"start": 4, "end": 5}, {"start": 5, "end": 6}, {"start": 6, "end": 7}, {"start": 7, "end": 4}
            ],
        });
        let r = call(json!({"op": "toggleFill", "network": net, "point": {"x": 5, "y": 5}}));
        assert_eq!(r["result"], "filled");
        assert_eq!(r["network"]["regions"][0]["windingRule"], "EVENODD");
        let n = VectorNetwork::from_json_value(&r["network"]).unwrap();
        assert!(n.region_contains(0, Vector::new(5.0, 5.0), 1e-3));
        assert!(!n.region_contains(0, Vector::new(15.0, 15.0), 1e-3));
    }
}
