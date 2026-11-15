//! Art, scatter and pattern brushes (brushes.md §4.4, §4.6, §5; plan
//! P17–P19): brush artwork placed along a stroke's centerline. The geometry
//! half of these brushes — the engine draws the returned paths like a group
//! (plan P8 follow-up), the editor reaches them through the WASM lane.
//!
//! - [`art`]: the artwork warped along the path (stretch / proportional /
//!   guides, direction, flips, overlap).
//! - [`colorize`]: Illustrator's four colourisation methods — the one place
//!   the colour math lives.
//! - [`artwork`]: artwork nodes → shapes in artwork space.
//! - [`spine`] and [`warp`]: arc-length centerline and adaptive cubic warping,
//!   shared by the placers.
//!
//! Everything is deterministic (seeded draws only) and total: no panics on
//! any input, hard budgets on the work (the WASM build aborts on panic).
//! Brush definitions are parsed here, mirroring `normalizeBrushDefinition`
//! (`core/scene/strokeModel.ts`).

pub mod art;
pub mod artwork;
pub mod bristle;
pub mod clip;
pub mod colorize;
pub mod common;
pub mod dyn_value;
pub mod geom;
pub mod pattern;
pub mod scatter;
pub mod spine;
pub mod warp;

#[cfg(test)]
mod gallery;
#[cfg(test)]
mod raster;
#[cfg(test)]
mod tests_art;
#[cfg(test)]
mod tests_bristle;
#[cfg(test)]
mod tests_pattern;
#[cfg(test)]
mod tests_scatter;
#[cfg(test)]
mod tests_sizing;

use crate::value::{Map, Value};

use crate::blend::BlendPathPoint;

pub use art::{art_brush_for_node, art_brush_geometry, art_brush_json, parse_art_brush, ArtBrush};
pub use bristle::{
    bristle_brush_for_node, bristle_brush_geometry, bristle_brush_json, parse_bristle_brush,
    BristleBrush, BristleShape,
};
pub use pattern::{
    parse_pattern_brush, pattern_brush_for_node, pattern_brush_for_node_with,
    pattern_brush_geometry, pattern_brush_json, PatternBrush, PatternOptions,
};
pub use scatter::{
    parse_scatter_brush, scatter_brush_for_node, scatter_brush_geometry, scatter_brush_json,
    ScatterBrush,
};

/// One JSON entry for every placed brush type (the WASM `stroke_brush`
/// export): dispatches on `brush.type` to [`art_brush_json`],
/// [`scatter_brush_json`], [`pattern_brush_json`] or [`bristle_brush_json`]; see [`art::art_brush_json`] for the envelope.
pub fn stroke_brush_json(request_json: &str) -> String {
    stroke_brush_json_with(request_json, &artwork::default_geometry)
}

/// [`stroke_brush_json`] with a caller-supplied artwork geometry callback.
pub fn stroke_brush_json_with(request_json: &str, geometry: artwork::GeometryFn) -> String {
    let req = match common::parse_request(request_json) {
        Ok(r) => r,
        Err(e) => return e,
    };
    match req.brush.get("type").and_then(Value::as_str) {
        Some("art") => art::art_request(req, geometry),
        Some("scatter") => scatter::scatter_request(req, geometry),
        Some("pattern") => pattern::pattern_request(req, geometry),
        Some("bristle") => bristle::bristle_request(req),
        other => common::error(
            "invalid_brush",
            &format!(
                "brush.type must be \"art\", \"scatter\", \"pattern\" or \"bristle\" (got {})",
                other.map_or_else(|| "none".to_string(), |t| format!("\"{t}\""))
            ),
        ),
    }
}

/// One placed artwork shape: a path in the stroke node's local coordinates
/// with the source shape's paint.
#[derive(Debug, Clone)]
pub struct PlacedPath {
    /// `<artwork node id>#<stroke subpath>` for art brushes,
    /// `<artwork node id>#<stroke subpath>.<copy>` for scatter brushes;
    /// unique per placement.
    pub id: String,
    pub source_id: String,
    pub points: Vec<BlendPathPoint>,
    pub closed: bool,
    /// Paint fields of the artwork node (fill, stroke, strokeWidth, opacity,
    /// fillRule, …), colourised and scaled.
    pub paint: Map,
    /// Where it sits along the stroke (JSON: `subpath`, `s`, `sSpan`).
    pub at: common::PlacedAt,
}

impl PlacedPath {
    /// Append this as a `path` node JSON object (`x = y = 0`, points
    /// node-local): the paint fields, then id, type, sourceId, x, y, points,
    /// closed (those win over same-named paint keys).
    pub fn write_node_json(&self, s: &mut String) {
        const OWN: [&str; 10] = [
            "id", "type", "sourceId", "x", "y", "points", "closed", "subpath", "s", "sSpan",
        ];
        let key = |s: &mut String, k: &str| {
            s.push_str(&Value::String(k.to_string()).to_string());
            s.push(':');
        };
        s.push('{');
        for (k, v) in &self.paint {
            if OWN.contains(&k.as_str()) {
                continue;
            }
            key(s, k);
            s.push_str(&v.to_string());
            s.push(',');
        }
        key(s, "id");
        s.push_str(&Value::String(self.id.clone()).to_string());
        s.push_str(r#","type":"path","sourceId":"#);
        s.push_str(&Value::String(self.source_id.clone()).to_string());
        s.push_str(r#","x":0,"y":0,"points":["#);
        for (i, p) in self.points.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let xy = |s: &mut String, x: f64, y: f64| {
                s.push_str(r#""x":"#);
                common::num(s, x);
                s.push_str(r#","y":"#);
                common::num(s, y);
            };
            match p {
                BlendPathPoint::Move { x, y } => {
                    s.push_str(r#"{"type":"move","#);
                    xy(s, *x, *y);
                }
                BlendPathPoint::Line { x, y } => {
                    s.push_str(r#"{"type":"line","#);
                    xy(s, *x, *y);
                }
                BlendPathPoint::Curve { x, y, cp1, cp2 } => {
                    s.push_str(r#"{"type":"curve","#);
                    xy(s, *x, *y);
                    s.push_str(r#","cp1":{"#);
                    xy(s, cp1.0, cp1.1);
                    s.push_str(r#"},"cp2":{"#);
                    xy(s, cp2.0, cp2.1);
                    s.push('}');
                }
                // Placers emit only move / line / curve.
                other => {
                    let v = other.to_json();
                    let t = v.to_string();
                    s.push_str(&t[..t.len() - 1]);
                }
            }
            s.push('}');
        }
        s.push_str(r#"],"closed":"#);
        s.push_str(if self.closed { "true" } else { "false" });
        s.push_str(&format!(r#","subpath":{},"s":"#, self.at.subpath));
        common::num(s, self.at.s);
        s.push_str(r#","sSpan":["#);
        common::num(s, self.at.s_span.0);
        s.push(',');
        common::num(s, self.at.s_span.1);
        s.push(']');
        s.push('}');
    }

    /// As a `path` node with id `id`, its points held compactly: the node `to_node_json` parses to (with `id`),
    /// without writing and parsing the JSON text. A placement of thousands of strands made each one's text only to
    /// parse it back. Falls back to that JSON for a point compact storage can't hold (an arc, a whole coordinate
    /// past 2^53).
    pub fn to_node(&self, id: String) -> crate::types::RawNode {
        use crate::path_points::{CompactKind, CompactPoints};
        // `num`: a non-finite coordinate is written as 0.
        let fin = |v: f64| if v.is_finite() { v } else { 0.0 };
        let mut kinds = Vec::with_capacity(self.points.len());
        let mut coords = Vec::with_capacity(self.points.len() * 6);
        let mut canonical = true;
        for p in &self.points {
            match p {
                BlendPathPoint::Move { x, y } => {
                    kinds.push(CompactKind::Move);
                    coords.extend([fin(*x), fin(*y)]);
                }
                BlendPathPoint::Line { x, y } => {
                    kinds.push(CompactKind::Line);
                    coords.extend([fin(*x), fin(*y)]);
                }
                BlendPathPoint::Curve { x, y, cp1, cp2 } => {
                    kinds.push(CompactKind::Curve);
                    coords.extend([fin(*x), fin(*y), fin(cp1.0), fin(cp1.1), fin(cp2.0), fin(cp2.1)]);
                }
                _ => {
                    canonical = false;
                    break;
                }
            }
        }
        let compact = canonical.then(|| CompactPoints::from_js_coords(&kinds, &coords)).flatten();
        let Some(points) = compact else {
            let mut v = self.to_node_json();
            v["id"] = Value::String(id);
            return match v {
                Value::Object(m) => crate::types::RawNode::new(m),
                _ => crate::types::RawNode::new(Default::default()),
            };
        };
        const OWN: [&str; 10] = [
            "id", "type", "sourceId", "x", "y", "points", "closed", "subpath", "s", "sSpan",
        ];
        let num = |v: f64| crate::feed_points::js_number(fin(v));
        let mut m = Map::with_capacity(self.paint.len() + OWN.len());
        for (k, v) in &self.paint {
            if !OWN.contains(&k.as_str()) {
                m.insert(k.clone(), v.clone());
            }
        }
        m.insert("id".into(), Value::String(id));
        m.insert("type".into(), Value::String("path".into()));
        m.insert("sourceId".into(), Value::String(self.source_id.clone()));
        m.insert("x".into(), Value::from(0u64));
        m.insert("y".into(), Value::from(0u64));
        m.insert("points".into(), Value::Null);
        m.insert("closed".into(), Value::Bool(self.closed));
        m.insert("subpath".into(), Value::from(self.at.subpath as u64));
        m.insert("s".into(), num(self.at.s));
        m.insert(
            "sSpan".into(),
            Value::Array(vec![num(self.at.s_span.0), num(self.at.s_span.1)]),
        );
        crate::types::RawNode::with_compact_points(m, points)
    }

    /// As a `path` node JSON value (same content as `write_node_json`).
    pub fn to_node_json(&self) -> Value {
        let mut s = String::new();
        self.write_node_json(&mut s);
        crate::value::from_str(&s).unwrap_or(Value::Null)
    }
}
