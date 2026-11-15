//! A node's `points`, read through one accessor ([`RawNode::points`](crate::types::RawNode::points)), and their
//! compact storage.
//!
//! A path node's points are its bulk: a vector-heavy document carries millions of them, and held as JSON objects
//! (each point, and each of its control points, an ordered map of heap-allocated keys) they cost about 800 bytes each
//! in the resident scene (the 5k-layer vector battery document: 925 MB from 72 MB of JSON). A path whose points all
//! have one of the canonical shapes (`move` / `line` `{type, x, y}`, `curve` `{type, x, y, cp1, cp2}`, `quadratic`
//! `{type, x, y, cp}`, control points `{x, y}`, every coordinate a number) is stored as [`CompactPoints`]: its kinds
//! and coordinates in flat arrays, about 20–55 bytes a point. Anything else (an `arc`, an extra key, a non-numeric
//! field) keeps its JSON.
//!
//! Readers go through [`PathPoints`] / [`PathPointRef`], which answer exactly what reading the JSON did
//! (`as_object`, `get(key)`, `as_f64`, …) whichever way the points are stored; [`PathPoints::json`] gives them back as
//! JSON values, key for key and number for number what was stored (an integer stays an integer).

use crate::value::{Map, Number, Value};
use std::borrow::Cow;
use std::cell::Cell;

thread_local! {
    static COMPACT_ON_LOAD: Cell<bool> = const { Cell::new(true) };
}

/// Whether nodes entering a scene (load, edits) on this thread store their points compactly (default on). Off keeps
/// them as JSON: for A/B measurement and the equality tests. Per thread, so tests running in parallel (each on its
/// own thread) don't see each other's setting.
pub fn set_compact_on_load(on: bool) {
    COMPACT_ON_LOAD.with(|c| c.set(on));
}

pub fn compact_on_load() -> bool {
    COMPACT_ON_LOAD.with(Cell::get)
}

/// A node's `points` array.
#[derive(Debug, Clone, Copy)]
pub enum PathPoints<'a> {
    /// Stored as the JSON array it arrived as.
    Json(&'a [Value]),
    /// Stored compactly.
    Compact(&'a CompactPoints),
}

impl<'a> PathPoints<'a> {
    pub fn len(&self) -> usize {
        match self {
            PathPoints::Json(a) => a.len(),
            PathPoints::Compact(c) => c.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Point `i`.
    pub fn get(&self, i: usize) -> Option<PathPointRef<'a>> {
        match self {
            PathPoints::Json(a) => a.get(i).map(PathPointRef::Json),
            PathPoints::Compact(c) => (i < c.len()).then_some(PathPointRef::Compact(c, i)),
        }
    }

    pub fn first(&self) -> Option<PathPointRef<'a>> {
        self.get(0)
    }

    pub fn last(&self) -> Option<PathPointRef<'a>> {
        self.len().checked_sub(1).and_then(|i| self.get(i))
    }

    pub fn iter(&self) -> PathPointsIter<'a> {
        PathPointsIter { points: *self, at: 0 }
    }

    /// The points as JSON values, as the node's JSON has them (borrowed when stored as JSON).
    pub fn json(&self) -> Cow<'a, [Value]> {
        match self {
            PathPoints::Json(a) => Cow::Borrowed(a),
            PathPoints::Compact(c) => Cow::Owned(c.to_json()),
        }
    }
}

impl<'a> IntoIterator for PathPoints<'a> {
    type Item = PathPointRef<'a>;
    type IntoIter = PathPointsIter<'a>;
    fn into_iter(self) -> PathPointsIter<'a> {
        self.iter()
    }
}

pub struct PathPointsIter<'a> {
    points: PathPoints<'a>,
    at: usize,
}

impl<'a> Iterator for PathPointsIter<'a> {
    type Item = PathPointRef<'a>;
    fn next(&mut self) -> Option<PathPointRef<'a>> {
        let p = self.points.get(self.at)?;
        self.at += 1;
        Some(p)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.points.len().saturating_sub(self.at);
        (n, Some(n))
    }
}

impl ExactSizeIterator for PathPointsIter<'_> {}

/// One point of a node's `points`: the answers reading its JSON gives.
#[derive(Debug, Clone, Copy)]
pub enum PathPointRef<'a> {
    Json(&'a Value),
    /// Point `.1` of a compact store.
    Compact(&'a CompactPoints, usize),
}

impl<'a> PathPointRef<'a> {
    /// Whether the point is an object (`Value::is_object`); a number (a line's flat `points`) is not.
    pub fn is_object(&self) -> bool {
        match self {
            PathPointRef::Json(v) => v.is_object(),
            PathPointRef::Compact(..) => true,
        }
    }

    /// The point as a number (a line node's flat `[x0, y0, x1, y1, …]` points).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            PathPointRef::Json(v) => v.as_f64(),
            PathPointRef::Compact(..) => None,
        }
    }

    /// `point.type` (`move`, `line`, `curve`, `quadratic`, `arc`, …).
    pub fn kind(&self) -> Option<&'a str> {
        match self {
            PathPointRef::Json(v) => v.as_object()?.get("type").and_then(Value::as_str),
            PathPointRef::Compact(c, i) => Some(c.kind(*i).name()),
        }
    }

    /// A numeric field (`x`, `y`, `rx`, `ry`, `xAxisRotation`, …).
    pub fn f64(&self, key: &str) -> Option<f64> {
        match self {
            PathPointRef::Json(v) => v.as_object()?.get(key).and_then(Value::as_f64),
            PathPointRef::Compact(c, i) => match key {
                "x" => Some(c.coord(*i, 0)),
                "y" => Some(c.coord(*i, 1)),
                _ => None,
            },
        }
    }

    /// A boolean field (`largeArc`, `sweep`, …).
    pub fn bool(&self, key: &str) -> Option<bool> {
        match self {
            PathPointRef::Json(v) => v.as_object()?.get(key).and_then(Value::as_bool),
            PathPointRef::Compact(..) => None,
        }
    }

    /// A string field (`cornerStyle`, …). Compact points hold none.
    pub fn str(&self, key: &str) -> Option<&'a str> {
        match self {
            PathPointRef::Json(v) => v.as_object()?.get(key).and_then(Value::as_str),
            PathPointRef::Compact(..) => None,
        }
    }

    /// A nested `{x, y}` field (`cp1`, `cp2`, `cp`): `None` when it isn't an object, else its `x` and `y` as
    /// numbers when they are.
    pub fn xy(&self, key: &str) -> Option<(Option<f64>, Option<f64>)> {
        match self {
            PathPointRef::Json(v) => {
                let o = v.as_object()?.get(key)?.as_object()?;
                Some((o.get("x").and_then(Value::as_f64), o.get("y").and_then(Value::as_f64)))
            }
            PathPointRef::Compact(c, i) => {
                let at = c.kind(*i).control_slot(key)?;
                Some((Some(c.coord(*i, at)), Some(c.coord(*i, at + 1))))
            }
        }
    }

    /// Whether the point has `key` at all.
    pub fn has(&self, key: &str) -> bool {
        match self {
            PathPointRef::Json(v) => v.as_object().is_some_and(|o| o.contains_key(key)),
            PathPointRef::Compact(c, i) => c.kind(*i).keys().contains(&key),
        }
    }

    /// The point as JSON.
    pub fn json(&self) -> Cow<'a, Value> {
        match self {
            PathPointRef::Json(v) => Cow::Borrowed(v),
            PathPointRef::Compact(c, i) => Cow::Owned(c.point_json(*i)),
        }
    }
}

// ---------------------------------------------------------------------------
// Compact storage
// ---------------------------------------------------------------------------

/// The point shapes a compact store holds, each with its keys in the order its JSON has them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CompactKind {
    Move = 0,
    Line = 1,
    Curve = 2,
    Quadratic = 3,
}

impl CompactKind {
    pub fn name(self) -> &'static str {
        match self {
            CompactKind::Move => "move",
            CompactKind::Line => "line",
            CompactKind::Curve => "curve",
            CompactKind::Quadratic => "quadratic",
        }
    }

    pub fn from_name(s: &str) -> Option<CompactKind> {
        Some(match s {
            "move" => CompactKind::Move,
            "line" => CompactKind::Line,
            "curve" => CompactKind::Curve,
            "quadratic" => CompactKind::Quadratic,
            _ => return None,
        })
    }

    /// The kind stored as `b` (`kind as u8`), `None` for no kind.
    pub fn try_from_u8(b: u8) -> Option<CompactKind> {
        (b <= CompactKind::Quadratic as u8).then(|| CompactKind::from_u8(b))
    }

    fn from_u8(b: u8) -> CompactKind {
        match b {
            0 => CompactKind::Move,
            1 => CompactKind::Line,
            2 => CompactKind::Curve,
            _ => CompactKind::Quadratic,
        }
    }

    /// The point's keys, in order.
    pub fn keys(self) -> &'static [&'static str] {
        match self {
            CompactKind::Move | CompactKind::Line => &["type", "x", "y"],
            CompactKind::Curve => &["type", "x", "y", "cp1", "cp2"],
            CompactKind::Quadratic => &["type", "x", "y", "cp"],
        }
    }

    /// Its control points' keys, in order.
    pub fn controls(self) -> &'static [&'static str] {
        match self {
            CompactKind::Move | CompactKind::Line => &[],
            CompactKind::Curve => &["cp1", "cp2"],
            CompactKind::Quadratic => &["cp"],
        }
    }

    /// How many coordinates the point stores (anchor, then each control point's x and y).
    pub fn coords(self) -> usize {
        2 + 2 * self.controls().len()
    }

    /// Where control point `key`'s x sits among the point's coordinates.
    fn control_slot(self, key: &str) -> Option<usize> {
        self.controls().iter().position(|k| *k == key).map(|i| 2 + 2 * i)
    }
}

/// A path's points in flat arrays: per point its kind and where its coordinates start, and every coordinate with
/// whether its JSON was an integer (so it is written back as one).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CompactPoints {
    kinds: Vec<u8>,
    starts: Vec<u32>,
    coords: Vec<f64>,
    /// Bit `i`: coordinate `i` was a JSON integer.
    ints: Vec<u64>,
}

impl CompactPoints {
    /// The compact form of `points`, when every point has a canonical shape; `None` keeps the JSON.
    pub fn from_json(points: &[Value]) -> Option<CompactPoints> {
        let mut c = CompactPoints {
            kinds: Vec::with_capacity(points.len()),
            starts: Vec::with_capacity(points.len()),
            coords: Vec::new(),
            ints: Vec::new(),
        };
        for p in points {
            let o = p.as_object()?;
            let kind = CompactKind::from_name(o.get("type")?.as_str()?)?;
            let keys = kind.keys();
            if o.len() != keys.len() || !o.keys().zip(keys).all(|(a, b)| a == b) {
                return None;
            }
            c.kinds.push(kind as u8);
            c.starts.push(u32::try_from(c.coords.len()).ok()?);
            c.push_number(o.get("x")?)?;
            c.push_number(o.get("y")?)?;
            for k in kind.controls() {
                let cp = o.get(*k)?.as_object()?;
                if cp.len() != 2 || !cp.keys().zip(["x", "y"]).all(|(a, b)| a == b) {
                    return None;
                }
                c.push_number(cp.get("x")?)?;
                c.push_number(cp.get("y")?)?;
            }
        }
        c.kinds.shrink_to_fit();
        c.starts.shrink_to_fit();
        c.coords.shrink_to_fit();
        c.ints.shrink_to_fit();
        Some(c)
    }

    /// The compact form of points given as their kinds and their coordinates in order (anchor, then each control
    /// point's x and y), as JavaScript numbers: what [`CompactPoints::from_json`] stores for the JSON
    /// `JSON.stringify` writes for them, or `None` when that would keep the JSON (a non-finite coordinate, a whole
    /// number too large to be exact).
    pub fn from_js_coords(kinds: &[CompactKind], coords: &[f64]) -> Option<CompactPoints> {
        let mut c = CompactPoints {
            kinds: Vec::with_capacity(kinds.len()),
            starts: Vec::with_capacity(kinds.len()),
            coords: Vec::with_capacity(coords.len()),
            ints: Vec::with_capacity(coords.len().div_ceil(64)),
        };
        let mut at = 0;
        for &kind in kinds {
            c.kinds.push(kind as u8);
            c.starts.push(u32::try_from(c.coords.len()).ok()?);
            for _ in 0..kind.coords() {
                c.push_js(*coords.get(at)?)?;
                at += 1;
            }
        }
        (at == coords.len()).then_some(c)
    }

    /// The compact form of points given as their kinds and their coordinates in order (anchor, then each control
    /// point's x and y), each coordinate a float: what [`CompactPoints::from_json`] stores for point objects whose
    /// numbers are all written as floats. `None` for a non-finite coordinate (its JSON is `null`, not a canonical
    /// point) or a coordinate count that does not match the kinds.
    pub fn from_floats(kinds: &[CompactKind], coords: &[f64]) -> Option<CompactPoints> {
        if coords.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mut starts = Vec::with_capacity(kinds.len());
        let mut at = 0usize;
        for kind in kinds {
            starts.push(u32::try_from(at).ok()?);
            at += kind.coords();
        }
        (at == coords.len()).then(|| CompactPoints {
            kinds: kinds.iter().map(|kind| *kind as u8).collect(),
            starts,
            coords: coords.to_vec(),
            ints: vec![0; coords.len().div_ceil(64)],
        })
    }

    /// [`CompactPoints::push_number`] of `crate::feed_points::js_number(v)`, without the JSON value.
    fn push_js(&mut self, v: f64) -> Option<()> {
        if !v.is_finite() {
            return None; // JSON null: not a canonical point
        }
        const EXACT: f64 = 9007199254740992.0; // 2^53
        let (value, int) = if v.fract() == 0.0 && v.abs() < 1e21 {
            if (0.0..18446744073709551616.0).contains(&v) {
                // A u64 in the JSON: kept only when exact.
                if v >= EXACT {
                    return None;
                }
                ((v as u64) as f64, true)
            } else if (-9223372036854775808.0..0.0).contains(&v) {
                if v <= -EXACT {
                    return None;
                }
                (v, true)
            } else {
                (v, false)
            }
        } else {
            (v, false)
        };
        let i = self.coords.len();
        self.coords.push(value);
        if i / 64 >= self.ints.len() {
            self.ints.push(0);
        }
        if int {
            self.ints[i / 64] |= 1 << (i % 64);
        }
        Some(())
    }

    fn push_number(&mut self, v: &Value) -> Option<()> {
        let n = v.as_number()?;
        let i = self.coords.len();
        let (value, int) = if let Some(u) = n.as_u64() {
            // Written back from the f64: only when that is exact.
            (u as f64, u < (1 << 53))
        } else if let Some(s) = n.as_i64() {
            (s as f64, s > -(1 << 53))
        } else {
            (n.as_f64()?, false)
        };
        if (n.is_u64() || n.is_i64()) && !int {
            return None;
        }
        self.coords.push(value);
        if i / 64 >= self.ints.len() {
            self.ints.push(0);
        }
        if int {
            self.ints[i / 64] |= 1 << (i % 64);
        }
        Some(())
    }

    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    pub fn kind(&self, i: usize) -> CompactKind {
        CompactKind::from_u8(self.kinds[i])
    }

    fn coord(&self, i: usize, k: usize) -> f64 {
        self.coords[self.starts[i] as usize + k]
    }

    fn number(&self, at: usize) -> Value {
        let v = self.coords[at];
        let int = self.ints.get(at / 64).is_some_and(|w| w & (1 << (at % 64)) != 0);
        Value::Number(if int {
            if v >= 0.0 {
                Number::from(v as u64)
            } else {
                Number::from(v as i64)
            }
        } else {
            Number::from_f64(v).unwrap_or_else(|| Number::from(0))
        })
    }

    /// Semantic JSON equality without allocating point objects. Preserve JSON's
    /// integer/float distinction and reject extra fields, just like materialization.
    pub fn eq_json(&self, points: &[Value]) -> bool {
        points.len() == self.len() && points.iter().enumerate().all(|(i, point)| {
            let Some(o) = point.as_object() else { return false };
            let kind = self.kind(i);
            let start = self.starts[i] as usize;
            o.len() == kind.keys().len()
                && o.get("type").and_then(Value::as_str) == Some(kind.name())
                && o.get("x") == Some(&self.number(start))
                && o.get("y") == Some(&self.number(start + 1))
                && kind.controls().iter().enumerate().all(|(j, key)| {
                    let Some(cp) = o.get(*key).and_then(Value::as_object) else { return false };
                    cp.len() == 2 && cp.get("x") == Some(&self.number(start + 2 + 2 * j))
                        && cp.get("y") == Some(&self.number(start + 3 + 2 * j))
                })
        })
    }

    /// Point `i` as the JSON it was stored from.
    pub fn point_json(&self, i: usize) -> Value {
        let kind = self.kind(i);
        let s = self.starts[i] as usize;
        let mut m = Map::with_capacity(kind.keys().len());
        m.insert("type".into(), Value::String(kind.name().into()));
        m.insert("x".into(), self.number(s));
        m.insert("y".into(), self.number(s + 1));
        for (j, k) in kind.controls().iter().enumerate() {
            let at = s + 2 + 2 * j;
            let mut cp = Map::with_capacity(2);
            cp.insert("x".into(), self.number(at));
            cp.insert("y".into(), self.number(at + 1));
            m.insert((*k).into(), Value::Object(cp));
        }
        Value::Object(m)
    }

    /// The points as the JSON they were stored from.
    pub fn to_json(&self) -> Vec<Value> {
        (0..self.len()).map(|i| self.point_json(i)).collect()
    }

    /// The coordinates of point `i` (anchor, then control points), and the kind.
    pub fn point(&self, i: usize) -> (CompactKind, &[f64]) {
        let kind = self.kind(i);
        let s = self.starts[i] as usize;
        (kind, &self.coords[s..s + kind.coords()])
    }

    /// Heap bytes held.
    pub fn heap_bytes(&self) -> usize {
        self.kinds.capacity() + 4 * self.starts.capacity() + 8 * self.coords.capacity() + 8 * self.ints.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    #[test]
    fn a_point_answers_what_its_json_does() {
        let v = json!([
            {"type": "move", "x": 1.5, "y": -2},
            {"type": "curve", "x": 10, "y": 20, "cp1": {"x": 3, "y": 4}, "cp2": {"x": "5", "y": 6}},
            {"type": "arc", "x": 1, "y": 1, "rx": 2, "largeArc": true},
            7.25,
            {"type": "quadratic", "x": 0, "y": 0, "cp": 3}
        ]);
        let pts = PathPoints::Json(v.as_array().unwrap());
        assert_eq!(pts.len(), 5);
        let p: Vec<PathPointRef<'_>> = pts.iter().collect();
        assert_eq!(p[0].kind(), Some("move"));
        assert_eq!((p[0].f64("x"), p[0].f64("y")), (Some(1.5), Some(-2.0)));
        assert_eq!(p[1].xy("cp1"), Some((Some(3.0), Some(4.0))));
        assert_eq!(p[1].xy("cp2"), Some((None, Some(6.0))), "a non-numeric coordinate");
        assert_eq!(p[1].xy("cp"), None, "absent");
        assert_eq!(p[2].bool("largeArc"), Some(true));
        assert_eq!(p[2].f64("ry"), None);
        assert!(!p[3].is_object());
        assert_eq!(p[3].as_f64(), Some(7.25));
        assert_eq!(p[3].kind(), None);
        assert_eq!(p[4].xy("cp"), None, "not an object");
        assert!(p[4].has("cp") && !p[4].has("cp1"));
        assert_eq!(pts.last().unwrap().kind(), Some("quadratic"));
        assert_eq!(&*pts.json(), v.as_array().unwrap().as_slice());
        assert_eq!(&*p[1].json(), &v[1]);
    }

    fn canonical() -> Value {
        json!([
            {"type": "move", "x": 54.95, "y": 34},
            {"type": "curve", "x": 62.55, "y": -35.14, "cp1": {"x": 57, "y": 35.14}, "cp2": {"x": -59.9, "y": 0.1}},
            {"type": "line", "x": 1e-9, "y": 123456789012u64},
            {"type": "quadratic", "x": 3, "y": 4, "cp": {"x": 0.5, "y": -0.0}},
            {"type": "move", "x": -7, "y": 9007199254740992.0}
        ])
    }

    #[test]
    fn canonical_points_compact_and_come_back_byte_for_byte() {
        let v = canonical();
        let a = v.as_array().unwrap();
        let c = CompactPoints::from_json(a).expect("canonical");
        assert_eq!(c.len(), 5);
        let back = Value::Array(c.to_json());
        assert_eq!(crate::value::to_string(&back).unwrap(), crate::value::to_string(&v).unwrap());
        assert_eq!(back, v);
        // The compact view answers what the JSON view does, for every key a reader asks.
        let (js, cs) = (PathPoints::Json(a), PathPoints::Compact(&c));
        for (x, y) in js.iter().zip(cs.iter()) {
            assert_eq!(x.is_object(), y.is_object());
            assert_eq!(x.as_f64(), y.as_f64());
            assert_eq!(x.kind(), y.kind());
            for k in ["x", "y", "rx", "ry", "xAxisRotation", "type", "cp1"] {
                assert_eq!(x.f64(k).map(f64::to_bits), y.f64(k).map(f64::to_bits), "{k}");
            }
            for k in ["largeArc", "sweep", "x"] {
                assert_eq!(x.bool(k), y.bool(k));
            }
            for k in ["cp", "cp1", "cp2", "x"] {
                assert_eq!(x.xy(k), y.xy(k), "{k}");
            }
            for k in ["type", "x", "y", "cp", "cp1", "cp2", "rx"] {
                assert_eq!(x.has(k), y.has(k), "{k}");
            }
            assert_eq!(x.json(), y.json());
        }
    }

    #[test]
    fn anything_else_keeps_its_json() {
        for bad in [
            json!([{"type": "arc", "x": 1, "y": 1, "rx": 2, "ry": 2, "xAxisRotation": 0, "largeArc": false, "sweep": true}]),
            json!([{"type": "line", "x": 1, "y": 1, "smooth": true}]),
            json!([{"x": 1, "y": 1, "type": "line"}]),
            json!([{"type": "line", "x": "1", "y": 1}]),
            json!([{"type": "curve", "x": 1, "y": 1, "cp1": {"x": 1, "y": 2}, "cp2": {"y": 1, "x": 2}}]),
            json!([{"type": "curve", "x": 1, "y": 1, "cp1": {"x": 1, "y": 2}}]),
            json!([{"type": "move", "x": 1, "y": 9007199254740993u64}]),
            json!([3.0, 4.0]),
        ] {
            assert!(CompactPoints::from_json(bad.as_array().unwrap()).is_none(), "{bad}");
        }
        assert!(CompactPoints::from_json(&[]).is_some_and(|c| c.is_empty()));
    }
}
