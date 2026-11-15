//! Brushes and variable-width strokes — typed native reader for the stroke
//! model (docs/roadmap/brushes.md §4; TS types in `core/scene/nodes.ts`,
//! normalisation in `core/scene/strokeModel.ts`).
//!
//! Nodes stay untyped JSON (`RawNode`); these parsers turn the stroke-scoped
//! objects into typed values with the SAME repairs the TS boundary applies
//! (drop malformed entries, clamp, sort, pin ends, quantise), so a document
//! written by any build reads identically in both lanes. One exception:
//! `BrushApplication.overrides` is kept as raw JSON (TS repairs it); the brush
//! renderers (plan P16–P19) will parse it. Brush definitions are not parsed
//! here yet either. Accessors on `RawNode` live in `defaults.rs`. Nothing
//! renders these yet (brush plan P8).
//!
//! Float parsing: serde_json (without `float_roundtrip`) can differ from JS by
//! one ULP on some decimal inputs; widths and positions are not affected at
//! any visible scale, so the workspace default is kept.

use crate::value::{Map, Value};

/// Largest per-side width multiplier (`MAX_MARKER_WIDTH` in strokeModel.ts).
pub const MAX_MARKER_WIDTH: f64 = 100.0;
/// Anchor indices above this are dropped (`MAX_ANCHOR_INDEX` in strokeModel.ts).
pub const MAX_ANCHOR_INDEX: f64 = 4_294_967_295.0;
pub const DYNAMIC_FREQUENCY_MIN: f64 = 0.01;
pub const DYNAMIC_FREQUENCY_MAX: f64 = 20.0;

pub const WIDTH_PROFILE_PRESETS: &[&str] = &[
    "uniform",
    "wedge",
    "taper",
    "quarter-taper",
    "eye",
    "mirrored-taper",
];

/// `WidthMarker` (nodes.ts). Widths are per side, 1 = the plain stroke.
#[derive(Debug, Clone, PartialEq)]
pub struct WidthMarker {
    pub id: String,
    pub t: f64,
    pub left: f64,
    pub right: f64,
    /// Index into `PathNode.points` this marker is bound to.
    pub anchor: Option<usize>,
    pub corner: bool,
}

/// `WidthProfile` (nodes.ts).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct WidthProfile {
    pub preset: Option<String>,
    pub markers: Option<Vec<WidthMarker>>,
    pub flip_along: bool,
    pub flip_across: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaperUnits {
    Percent,
    Px,
}

/// `StrokeTaper` (nodes.ts).
#[derive(Debug, Clone, PartialEq)]
pub struct StrokeTaper {
    pub units: TaperUnits,
    pub start_length: f64,
    pub end_length: f64,
    pub start_width: f64,
    pub end_width: f64,
    pub start_ease: f64,
    pub end_ease: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveUnits {
    Px,
    Cycles,
}

/// `StrokeWave` (nodes.ts).
#[derive(Debug, Clone, PartialEq)]
pub struct StrokeWave {
    pub amount: f64,
    pub units: WaveUnits,
    pub wavelength: f64,
    pub phase: f64,
}

/// `StrokeDynamic` (nodes.ts).
#[derive(Debug, Clone, PartialEq)]
pub struct StrokeDynamic {
    pub frequency: f64,
    pub wiggle: f64,
    pub smoothen: f64,
    pub seed: Option<u32>,
}

/// `BrushApplication` (nodes.ts). `overrides` stays raw JSON until the brush
/// renderers (plan P16–P19) read it.
#[derive(Debug, Clone, PartialEq)]
pub struct BrushApplication {
    pub brush_id: String,
    pub seed: Option<u32>,
    pub overrides: Option<Map>,
}

/// `InkChannels` (nodes.ts): parallel arrays over arc length.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct InkChannels {
    pub t: Vec<f64>,
    pub pressure: Option<Vec<f64>>,
    pub tilt_x: Option<Vec<f64>>,
    pub tilt_y: Option<Vec<f64>>,
    pub twist: Option<Vec<f64>>,
    /// Stylus wheel (airbrush finger wheel, Pointer Events `tangentialPressure`), 0..1.
    pub wheel: Option<Vec<f64>>,
    /// Drawing speed, 0..1 (on-screen speed / 3 px per ms, recorded by the drawing tools).
    pub speed: Option<Vec<f64>>,
    pub time: Option<Vec<f64>>,
    /// The stroke's start on the document's drawing clock (whole ms, ≥ 0; not wall-clock time).
    pub started_at: Option<f64>,
}

fn finite(v: Option<&Value>) -> Option<f64> {
    v.and_then(Value::as_f64).filter(|x| x.is_finite())
}

fn num(v: Option<&Value>, fallback: f64) -> f64 {
    finite(v).unwrap_or(fallback)
}

fn seed(v: Option<&Value>) -> Option<u32> {
    finite(v).map(|x| (x.abs().trunc() % 4_294_967_296.0) as u32)
}

fn unique_id(base: &str, used: &mut Vec<String>) -> String {
    let mut id = base.to_string();
    let mut n = 2;
    while used.iter().any(|u| u == &id) {
        id = format!("{base}-{n}");
        n += 1;
    }
    used.push(id.clone());
    id
}

fn parse_marker(raw: &Value) -> Option<WidthMarker> {
    let o = raw.as_object()?;
    let t = finite(o.get("t"))?;
    let left = finite(o.get("left")).or_else(|| finite(o.get("right")))?;
    let right = finite(o.get("right")).unwrap_or(left);
    let anchor = o
        .get("anchor")
        .and_then(Value::as_f64)
        .filter(|a| a.is_finite() && *a >= 0.0 && a.fract() == 0.0 && *a <= MAX_ANCHOR_INDEX)
        .map(|a| a as usize);
    Some(WidthMarker {
        id: o
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        t: t.clamp(0.0, 1.0),
        left: left.clamp(0.0, MAX_MARKER_WIDTH),
        right: right.clamp(0.0, MAX_MARKER_WIDTH),
        anchor,
        corner: o.get("corner").and_then(Value::as_bool) == Some(true),
    })
}

/// `normalizeWidthMarkers` (strokeModel.ts): sorted, clamped, at most two per
/// `t`, ends pinned at 0 and 1, unique ids. `None` when nothing valid remains.
pub fn parse_width_markers(raw: &Value) -> Option<Vec<WidthMarker>> {
    let mut parsed: Vec<WidthMarker> = raw.as_array()?.iter().filter_map(parse_marker).collect();
    if parsed.is_empty() {
        return None;
    }
    // Stable sort by t (Vec::sort_by is stable, like Array.prototype.sort).
    parsed.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    let mut markers = Vec::with_capacity(parsed.len());
    for i in 0..parsed.len() {
        let same_prev = i > 0 && parsed[i - 1].t == parsed[i].t;
        let same_next = i + 1 < parsed.len() && parsed[i + 1].t == parsed[i].t;
        if same_prev && same_next {
            continue;
        }
        markers.push(parsed[i].clone());
    }
    let mut pinned = Vec::with_capacity(markers.len() + 2);
    let first = markers[0].clone();
    if first.t > 0.0 {
        pinned.push(WidthMarker {
            id: String::new(),
            t: 0.0,
            left: first.left,
            right: first.right,
            anchor: None,
            corner: false,
        });
    }
    let last = markers[markers.len() - 1].clone();
    pinned.extend(markers);
    if last.t < 1.0 {
        pinned.push(WidthMarker {
            id: String::new(),
            t: 1.0,
            left: last.left,
            right: last.right,
            anchor: None,
            corner: false,
        });
    }
    if pinned.len() > 1 && pinned[1].t == 0.0 {
        pinned.remove(0);
    }
    if pinned.len() > 1 && pinned[pinned.len() - 2].t == 1.0 {
        pinned.pop();
    }
    let mut used: Vec<String> = Vec::new();
    for m in &pinned {
        if !m.id.is_empty() && !used.contains(&m.id) {
            used.push(m.id.clone());
        }
    }
    let mut seen: Vec<String> = Vec::new();
    for (i, m) in pinned.iter_mut().enumerate() {
        if m.id.is_empty() || seen.contains(&m.id) {
            let base = if m.id.is_empty() {
                format!("w{i}")
            } else {
                m.id.clone()
            };
            m.id = unique_id(&base, &mut used);
        }
        seen.push(m.id.clone());
    }
    Some(pinned)
}

/// `normalizeWidthProfile` (strokeModel.ts).
pub fn parse_width_profile(raw: &Value) -> Option<WidthProfile> {
    let o = raw.as_object()?;
    let preset = o
        .get("preset")
        .and_then(Value::as_str)
        .filter(|p| WIDTH_PROFILE_PRESETS.contains(p))
        .map(str::to_string);
    let markers = o.get("markers").and_then(parse_width_markers);
    if preset.is_none() && markers.is_none() {
        return None;
    }
    Some(WidthProfile {
        preset,
        markers,
        flip_along: o.get("flipAlong").and_then(Value::as_bool) == Some(true),
        flip_across: o.get("flipAcross").and_then(Value::as_bool) == Some(true),
    })
}

/// `normalizeStrokeTaper` (strokeModel.ts).
pub fn parse_stroke_taper(raw: &Value) -> Option<StrokeTaper> {
    let o = raw.as_object()?;
    let units = if o.get("units").and_then(Value::as_str) == Some("px") {
        TaperUnits::Px
    } else {
        TaperUnits::Percent
    };
    let max_len = match units {
        TaperUnits::Percent => 100.0,
        TaperUnits::Px => f64::MAX,
    };
    Some(StrokeTaper {
        units,
        start_length: num(o.get("startLength"), 0.0).clamp(0.0, max_len),
        end_length: num(o.get("endLength"), 0.0).clamp(0.0, max_len),
        start_width: num(o.get("startWidth"), 0.0).clamp(0.0, 1.0),
        end_width: num(o.get("endWidth"), 0.0).clamp(0.0, 1.0),
        start_ease: num(o.get("startEase"), 0.0).clamp(-1.0, 1.0),
        end_ease: num(o.get("endEase"), 0.0).clamp(-1.0, 1.0),
    })
}

/// `normalizeStrokeWave` (strokeModel.ts).
pub fn parse_stroke_wave(raw: &Value) -> Option<StrokeWave> {
    let o = raw.as_object()?;
    let units = if o.get("units").and_then(Value::as_str) == Some("cycles") {
        WaveUnits::Cycles
    } else {
        WaveUnits::Px
    };
    let default_wavelength = match units {
        WaveUnits::Cycles => 1.0,
        WaveUnits::Px => 20.0,
    };
    Some(StrokeWave {
        amount: num(o.get("amount"), 0.0).max(0.0),
        units,
        wavelength: finite(o.get("wavelength"))
            .filter(|w| *w > 0.0)
            .unwrap_or(default_wavelength),
        phase: num(o.get("phase"), 0.0),
    })
}

/// `normalizeStrokeDynamic` (strokeModel.ts).
pub fn parse_stroke_dynamic(raw: &Value) -> Option<StrokeDynamic> {
    let o = raw.as_object()?;
    Some(StrokeDynamic {
        frequency: num(o.get("frequency"), 1.0).clamp(DYNAMIC_FREQUENCY_MIN, DYNAMIC_FREQUENCY_MAX),
        wiggle: num(o.get("wiggle"), 0.0).max(0.0),
        smoothen: num(o.get("smoothen"), 0.0).clamp(0.0, 1.0),
        seed: seed(o.get("seed")),
    })
}

/// `normalizeBrushApplication` (strokeModel.ts), minus the override repairs
/// (overrides are kept raw here).
pub fn parse_brush_application(raw: &Value) -> Option<BrushApplication> {
    let o = raw.as_object()?;
    let brush_id = o
        .get("brushId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())?;
    Some(BrushApplication {
        brush_id: brush_id.to_string(),
        seed: seed(o.get("seed")),
        overrides: o.get("overrides").and_then(Value::as_object).cloned(),
    })
}

fn quantize(channel: &str, v: f64) -> f64 {
    match channel {
        "t" => (v.clamp(0.0, 1.0) * 1000.0).round() / 1000.0,
        "pressure" | "wheel" | "speed" => (v.clamp(0.0, 1.0) * 256.0).round() / 256.0,
        "tiltX" | "tiltY" => js_round(v.clamp(-90.0, 90.0)),
        "twist" => ((js_round(v) % 360.0) + 360.0) % 360.0,
        "time" => js_round(v).max(0.0),
        _ => v,
    }
}

/// JS `Math.round` (half rounds toward +∞), which differs from Rust's
/// `f64::round` (half away from zero) for negative halves. Written without
/// `v + 0.5`, which rounds wrongly for 0.49999999999999994 and above 2^52.
fn js_round(v: f64) -> f64 {
    let f = v.floor();
    if v - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

const INK_VALUE_CHANNELS: &[&str] = &["pressure", "tiltX", "tiltY", "twist", "wheel", "speed", "time"];

/// Most ink samples a stroke keeps (`INK_MAX_SAMPLES`, strokeModel.ts).
pub const INK_MAX_SAMPLES: usize = 4096;

/// `normalizeInkChannels` (strokeModel.ts).
pub fn parse_ink_channels(raw: &Value) -> Option<InkChannels> {
    let o = raw.as_object()?;
    let ts = o.get("t")?.as_array()?;
    let n = ts.len();
    let channels: Vec<(&str, &Vec<Value>)> = INK_VALUE_CHANNELS
        .iter()
        .filter_map(|c| {
            let arr = o.get(*c)?.as_array()?;
            (arr.len() == n && arr.iter().all(|v| finite(Some(v)).is_some())).then_some((*c, arr))
        })
        .collect();
    let mut out_t = Vec::with_capacity(n);
    let mut out: Vec<Vec<f64>> = vec![Vec::with_capacity(n); channels.len()];
    let time_idx = channels.iter().position(|(c, _)| *c == "time");
    let mut last_t = f64::NEG_INFINITY;
    let mut last_time = f64::NEG_INFINITY;
    // `INK_MAX_SAMPLES` (strokeModel.ts): longer input is thinned evenly by index, ends kept.
    let count = n.min(INK_MAX_SAMPLES);
    for k in 0..count {
        let i = if n > INK_MAX_SAMPLES {
            ((k * (n - 1)) as f64 / (INK_MAX_SAMPLES - 1) as f64).round() as usize
        } else {
            k
        };
        let Some(raw_t) = finite(ts.get(i)) else {
            continue;
        };
        let t = quantize("t", raw_t);
        if t < last_t {
            continue;
        }
        if let Some(ti) = time_idx {
            let time = quantize("time", channels[ti].1[i].as_f64().unwrap_or(0.0));
            if time < last_time {
                continue;
            }
            last_time = time;
        }
        last_t = t;
        out_t.push(t);
        for (k, (c, arr)) in channels.iter().enumerate() {
            out[k].push(quantize(c, arr[i].as_f64().unwrap_or(0.0)));
        }
    }
    if out_t.is_empty() {
        return None;
    }
    let mut ink = InkChannels {
        t: out_t,
        ..Default::default()
    };
    for ((c, _), values) in channels.iter().zip(out) {
        match *c {
            "pressure" => ink.pressure = Some(values),
            "tiltX" => ink.tilt_x = Some(values),
            "tiltY" => ink.tilt_y = Some(values),
            "twist" => ink.twist = Some(values),
            "wheel" => ink.wheel = Some(values),
            "speed" => ink.speed = Some(values),
            "time" => ink.time = Some(values),
            _ => {}
        }
    }
    ink.started_at = finite(o.get("startedAt"))
        .filter(|v| *v >= 0.0)
        .map(f64::round);
    Some(ink)
}

// ---------------------------------------------------------------------------
// Width evaluation (mirror of core/scene/widthProfile.ts; pinned by the shared
// fixture fixtures/stroke-profile/remap.json via tests/stroke_profile_parity.rs)
// ---------------------------------------------------------------------------

/// Figma's presets as (t, width) pairs, width per side (`WIDTH_PROFILE_PRESET_POINTS`).
pub fn preset_points(preset: &str) -> &'static [(f64, f64)] {
    match preset {
        "uniform" => &[(0.0, 1.0), (1.0, 1.0)],
        "wedge" => &[(0.0, 1.0), (1.0, 0.0)],
        "taper" => &[(0.0, 1.0), (1.0, 0.25)],
        "quarter-taper" => &[(0.0, 0.25), (0.25, 1.0), (1.0, 0.25)],
        "eye" => &[(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)],
        "mirrored-taper" => &[(0.0, 0.25), (0.5, 1.0), (1.0, 0.25)],
        _ => &[(0.0, 1.0), (1.0, 1.0)],
    }
}

/// `widthProfileMarkers`: the stored markers, or the preset's table.
pub fn profile_markers(profile: &WidthProfile) -> Vec<WidthMarker> {
    if let Some(preset) = &profile.preset {
        return preset_points(preset)
            .iter()
            .enumerate()
            .map(|(i, &(t, w))| WidthMarker {
                id: format!("{preset}-{i}"),
                t,
                left: w,
                right: w,
                anchor: None,
                corner: false,
            })
            .collect();
    }
    profile.markers.clone().unwrap_or_default()
}

/// `resolveWidthMarkers`: anchored markers take their anchor's arc-length
/// fraction (`anchor_t[anchor]`, from `path_length::anchor_fractions`),
/// unanchored markers between two pivots keep their relative position,
/// first/last are pinned to 0/1, and an anchored marker that would resolve
/// before the previous pivot falls back to its stored t (as if unanchored).
/// The result is non-decreasing.
pub fn resolve_width_markers(
    markers: &[WidthMarker],
    anchor_t: Option<&[Option<f64>]>,
) -> Vec<WidthMarker> {
    let mut out = markers.to_vec();
    let ts = resolved_positions(markers, anchor_t);
    for (m, t) in out.iter_mut().zip(ts) {
        m.t = t;
    }
    out
}

/// The resolved positions alone (no marker clones).
fn resolved_positions(markers: &[WidthMarker], anchor_t: Option<&[Option<f64>]>) -> Vec<f64> {
    let n = markers.len();
    if n == 0 {
        return Vec::new();
    }
    let mut resolved: Vec<Option<f64>> = vec![None; n];
    resolved[0] = Some(0.0);
    resolved[n - 1] = Some(1.0);
    if let Some(anchor_t) = anchor_t {
        let mut last_pivot = 0.0f64;
        for i in 1..n.saturating_sub(1) {
            let Some(a) = markers[i].anchor else { continue };
            let Some(Some(t)) = anchor_t.get(a) else {
                continue;
            };
            if !t.is_finite() {
                continue;
            }
            let r = t.clamp(0.0, 1.0);
            if r < last_pivot {
                continue; // out of order: treat as unanchored
            }
            resolved[i] = Some(r);
            last_pivot = r;
        }
    }
    let mut prev = 0;
    for next in 1..n {
        let Some(rb) = resolved[next] else { continue };
        let ta = markers[prev].t;
        let tb = markers[next].t;
        let ra = resolved[prev].unwrap_or(0.0);
        for k in prev + 1..next {
            let u = if tb > ta {
                (markers[k].t - ta) / (tb - ta)
            } else {
                (k - prev) as f64 / (next - prev) as f64
            };
            resolved[k] = Some(ra + u.clamp(0.0, 1.0) * (rb - ra));
        }
        prev = next;
    }
    let mut floor = 0.0f64;
    resolved
        .into_iter()
        .map(|r| {
            floor = floor.max(r.unwrap_or(0.0));
            floor
        })
        .collect()
}

/// `effectiveWidthMarkers`: resolved, then flipped along (markers come back
/// unbound: the flip moves them off their anchors) / across.
pub fn effective_width_markers(
    profile: &WidthProfile,
    anchor_t: Option<&[Option<f64>]>,
) -> Vec<WidthMarker> {
    let mut markers = resolve_width_markers(&profile_markers(profile), anchor_t);
    if profile.flip_along {
        markers.reverse();
        for m in &mut markers {
            m.t = 1.0 - m.t;
            m.anchor = None;
        }
    }
    if profile.flip_across {
        for m in &mut markers {
            std::mem::swap(&mut m.left, &mut m.right);
        }
    }
    markers
}

/// `PreparedMarker`: what evaluation needs (no ids, no bindings).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreparedMarker {
    pub t: f64,
    pub left: f64,
    pub right: f64,
    pub corner: bool,
}

/// `PreparedWidthProfile`: a profile ready for per-sample evaluation —
/// resolved, flipped and stripped markers plus both sides' PCHIP tangents,
/// computed once. `width_at` is a binary search. Build one per stroke per
/// frame and evaluate every outline sample through it.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedWidthProfile {
    pub markers: Vec<PreparedMarker>,
    pub tangents_left: Vec<f64>,
    pub tangents_right: Vec<f64>,
    /// Presets apply per subpath.
    pub per_subpath: bool,
    /// One closed subpath: the end tangents wrap across a continuous seam
    /// (`periodic` in widthProfile.ts). Compound paths are never periodic.
    pub periodic: bool,
}

/// `seamWraps`: one side is continuous across the seam — equal widths at the
/// t = 0 and t = 1 markers, neither a corner.
fn seam_wraps(markers: &[PreparedMarker], s: Side) -> bool {
    let n = markers.len();
    if n < 2 {
        return false;
    }
    let (first, last) = (&markers[0], &markers[n - 1]);
    !first.corner
        && !last.corner
        && side(first, s) == side(last, s)
        && first.t == 0.0
        && last.t == 1.0
}

/// `seamTangent`: PCHIP tangent at the seam, the neighbour before t = 0 being
/// marker n-2 shifted back one period and the one after being marker 1.
fn seam_tangent(markers: &[PreparedMarker], s: Side) -> f64 {
    let n = markers.len();
    let m = &markers[0];
    let prev = &markers[n - 2];
    let next = &markers[1];
    let has_prev = n >= 3 && prev.t < 1.0;
    let has_next = n >= 2 && next.t > 0.0;
    let hl = if has_prev { 1.0 - prev.t } else { 0.0 };
    let hr = if has_next { next.t } else { 0.0 };
    let dl = if has_prev {
        (side(m, s) - side(prev, s)) / hl
    } else {
        0.0
    };
    let dr = if has_next {
        (side(next, s) - side(m, s)) / hr
    } else {
        0.0
    };
    if has_prev && has_next {
        if dl * dr <= 0.0 {
            return 0.0;
        }
        let w1 = 2.0 * hr + hl;
        let w2 = hr + 2.0 * hl;
        return (w1 + w2) / (w1 / dl + w2 / dr);
    }
    if has_prev {
        dl
    } else if has_next {
        dr
    } else {
        0.0
    }
}

#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
}

fn side(m: &PreparedMarker, s: Side) -> f64 {
    match s {
        Side::Left => m.left,
        Side::Right => m.right,
    }
}

/// PCHIP tangent at marker `k` (neighbours across a discontinuity excluded).
fn tangent(markers: &[PreparedMarker], k: usize, s: Side) -> f64 {
    let m = &markers[k];
    let has_prev = k > 0 && markers[k - 1].t < m.t;
    let has_next = k + 1 < markers.len() && markers[k + 1].t > m.t;
    let hl = if has_prev {
        m.t - markers[k - 1].t
    } else {
        0.0
    };
    let hr = if has_next {
        markers[k + 1].t - m.t
    } else {
        0.0
    };
    let dl = if has_prev {
        (side(m, s) - side(&markers[k - 1], s)) / hl
    } else {
        0.0
    };
    let dr = if has_next {
        (side(&markers[k + 1], s) - side(m, s)) / hr
    } else {
        0.0
    };
    if has_prev && has_next {
        if dl * dr <= 0.0 {
            return 0.0;
        }
        let w1 = 2.0 * hr + hl;
        let w2 = hr + 2.0 * hl;
        return (w1 + w2) / (w1 / dl + w2 / dr);
    }
    if has_prev {
        dl
    } else if has_next {
        dr
    } else {
        0.0
    }
}

impl PreparedWidthProfile {
    /// `prepareWidthMarkers`: from already-effective markers.
    pub fn from_markers(markers: &[WidthMarker], per_subpath: bool) -> Self {
        Self::from_markers_with(markers, per_subpath, false)
    }

    /// `prepareWidthMarkers(markers, perSubpath, periodic)`: `periodic` when the
    /// stroke is one closed subpath (`closed && subpaths.len() == 1`).
    pub fn from_markers_with(markers: &[WidthMarker], per_subpath: bool, periodic: bool) -> Self {
        let prepared: Vec<PreparedMarker> = markers
            .iter()
            .map(|m| PreparedMarker {
                t: m.t,
                left: m.left,
                right: m.right,
                corner: m.corner,
            })
            .collect();
        let tangents = |s: Side| -> Vec<f64> {
            let mut out: Vec<f64> = (0..prepared.len())
                .map(|k| tangent(&prepared, k, s))
                .collect();
            if periodic && seam_wraps(&prepared, s) {
                let seam = seam_tangent(&prepared, s);
                let last = out.len() - 1;
                out[0] = seam;
                out[last] = seam;
            }
            out
        };
        let tangents_left = tangents(Side::Left);
        let tangents_right = tangents(Side::Right);
        PreparedWidthProfile {
            markers: prepared,
            tangents_left,
            tangents_right,
            per_subpath,
            periodic,
        }
    }

    /// `prepareWidthProfile`: presets ignore `anchor_t` (they have no anchors).
    pub fn new(profile: &WidthProfile, anchor_t: Option<&[Option<f64>]>) -> Self {
        Self::new_with(profile, anchor_t, false)
    }

    /// `prepareWidthProfile(profile, anchorT, { periodic })`.
    pub fn new_with(
        profile: &WidthProfile,
        anchor_t: Option<&[Option<f64>]>,
        periodic: bool,
    ) -> Self {
        let per_subpath = profile.preset.is_some();
        let anchors = if per_subpath { None } else { anchor_t };
        Self::from_markers_with(
            &effective_width_markers(profile, anchors),
            per_subpath,
            periodic,
        )
    }

    fn segment_value(&self, a: usize, t: f64, s: Side) -> f64 {
        let ma = &self.markers[a];
        let mb = &self.markers[a + 1];
        let h = mb.t - ma.t;
        let u = (t - ma.t) / h;
        let y0 = side(ma, s);
        let y1 = side(mb, s);
        if ma.corner || mb.corner {
            return y0 + (y1 - y0) * u;
        }
        let tangents = match s {
            Side::Left => &self.tangents_left,
            Side::Right => &self.tangents_right,
        };
        let m0 = tangents[a];
        let m1 = tangents[a + 1];
        let u2 = u * u;
        let u3 = u2 * u;
        let v = (2.0 * u3 - 3.0 * u2 + 1.0) * y0
            + (u3 - 2.0 * u2 + u) * h * m0
            + (-2.0 * u3 + 3.0 * u2) * y1
            + (u3 - u2) * h * m1;
        if v < 0.0 {
            0.0
        } else {
            v
        }
    }

    /// `preparedWidthAtU`: (left, right) at `u` in the profile's own 0..1. At a
    /// discontinuous point `incoming` picks the incoming side, else outgoing.
    pub fn width_at_u(&self, u: f64, incoming: bool) -> (f64, f64) {
        let m = &self.markers;
        let n = m.len();
        if n == 0 {
            return (1.0, 1.0);
        }
        let first_at = |x: f64| m.iter().position(|k| k.t == x).unwrap_or(0);
        let last_at = |x: f64| m.iter().rposition(|k| k.t == x).unwrap_or(0);
        if u <= m[0].t {
            let k = &m[if incoming { 0 } else { last_at(m[0].t) }];
            return (k.left, k.right);
        }
        if u >= m[n - 1].t {
            let k = &m[if incoming {
                first_at(m[n - 1].t)
            } else {
                n - 1
            }];
            return (k.left, k.right);
        }
        // Largest a in [0, n-2] with t_a < u (or ≤ u outgoing): binary search.
        let (mut lo, mut hi) = (0usize, n - 2);
        while lo < hi {
            let mid = (lo + hi + 1) >> 1;
            let tm = m[mid].t;
            if tm < u || (!incoming && tm == u) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let a = lo;
        if m[a].t == u {
            return (m[a].left, m[a].right);
        }
        if m[a + 1].t == u && incoming {
            return (m[a + 1].left, m[a + 1].right);
        }
        (
            self.segment_value(a, u, Side::Left),
            self.segment_value(a, u, Side::Right),
        )
    }

    /// `preparedWidthAt`: (left, right) at `t`, a fraction of the total length
    /// of all subpaths; `subpath` is the containing subpath's (start, end)
    /// (presets apply per subpath).
    pub fn width_at(&self, t: f64, subpath: Option<(f64, f64)>) -> (f64, f64) {
        let mut u = t.clamp(0.0, 1.0);
        if self.per_subpath {
            if let Some((start, end)) = subpath {
                u = if end > start {
                    ((t - start) / (end - start)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
            }
        }
        self.width_at_u(u, false)
    }
}

/// `widthAtMarkers`: (left, right) at `t` for resolved, flipped markers.
pub fn width_at_markers(markers: &[WidthMarker], t: f64, incoming: bool) -> (f64, f64) {
    PreparedWidthProfile::from_markers(markers, false).width_at_u(t, incoming)
}

/// `widthAt`: one-off (left, right) width multipliers at `t` (fraction of the
/// total length of all subpaths; 1 = plain stroke). Per-sample callers build a
/// `PreparedWidthProfile` once instead.
pub fn width_at(
    profile: &WidthProfile,
    t: f64,
    anchor_t: Option<&[Option<f64>]>,
    subpath: Option<(f64, f64)>,
) -> (f64, f64) {
    PreparedWidthProfile::new(profile, anchor_t).width_at(t, subpath)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    #[test]
    fn markers_sort_clamp_and_pin_ends() {
        let m = parse_width_markers(&json!([
            {"id": "b", "t": 0.7, "left": 2, "right": 2},
            {"id": "a", "t": 0.2, "left": 0.5},
            {"id": "x", "t": "bad", "left": 1, "right": 1},
            {"id": "c", "t": 1.5, "left": -1, "right": 500}
        ]))
        .unwrap();
        let ts: Vec<f64> = m.iter().map(|m| m.t).collect();
        assert_eq!(ts, vec![0.0, 0.2, 0.7, 1.0]);
        assert_eq!(m[0].left, 0.5); // copied from the nearest marker
        assert_eq!(m[1].right, 0.5); // right defaults to left
        assert_eq!((m[3].left, m[3].right), (0.0, MAX_MARKER_WIDTH));
        assert_eq!(m[0].id, "w0");
    }

    #[test]
    fn discontinuous_points_keep_first_and_last() {
        let m = parse_width_markers(&json!([
            {"id": "a", "t": 0, "left": 1, "right": 1},
            {"id": "b", "t": 0.5, "left": 1, "right": 1},
            {"id": "c", "t": 0.5, "left": 2, "right": 2},
            {"id": "d", "t": 0.5, "left": 3, "right": 3},
            {"id": "e", "t": 1, "left": 1, "right": 1}
        ]))
        .unwrap();
        let ids: Vec<&str> = m.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "d", "e"]);
    }

    #[test]
    fn duplicate_ids_are_suffixed() {
        let m = parse_width_markers(&json!([
            {"id": "a", "t": 0, "left": 1},
            {"id": "a", "t": 1, "left": 1}
        ]))
        .unwrap();
        assert_eq!(m[0].id, "a");
        assert_eq!(m[1].id, "a-2");
    }

    #[test]
    fn taper_wave_dynamic_defaults_and_clamps() {
        let t = parse_stroke_taper(&json!({"startLength": 150, "endWidth": 2})).unwrap();
        assert_eq!(t.units, TaperUnits::Percent);
        assert_eq!(t.start_length, 100.0);
        assert_eq!(t.end_width, 1.0);
        let w = parse_stroke_wave(&json!({"units": "cycles", "wavelength": -3})).unwrap();
        assert_eq!(w.wavelength, 1.0);
        let d = parse_stroke_dynamic(&json!({"frequency": 99, "seed": -7.9})).unwrap();
        assert_eq!(d.frequency, DYNAMIC_FREQUENCY_MAX);
        assert_eq!(d.seed, Some(7));
        assert!(parse_brush_application(&json!({"brushId": ""})).is_none());
    }

    #[test]
    fn ink_channels_are_capped_evenly_by_index() {
        let n = 10_000;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let ink = parse_ink_channels(&json!({ "t": t })).unwrap();
        assert!(ink.t.len() <= INK_MAX_SAMPLES && ink.t.len() > INK_MAX_SAMPLES - 10);
        assert_eq!(ink.t[0], 0.0);
        assert_eq!(*ink.t.last().unwrap(), 1.0);
    }

    #[test]
    fn ink_channels_started_at_is_whole_non_negative_ms() {
        let ink =
            parse_ink_channels(&json!({ "t": [0, 1], "time": [0, 10], "startedAt": 12_345.6_f64 }))
                .unwrap();
        assert_eq!(ink.started_at, Some(12_346.0));
        for bad in [json!(-5), json!("x"), json!(null)] {
            let ink = parse_ink_channels(&json!({ "t": [0, 1], "startedAt": bad })).unwrap();
            assert_eq!(ink.started_at, None);
        }
    }

    #[test]
    fn ink_channels_quantise_and_drop_bad_samples() {
        let ink = parse_ink_channels(&json!({
            "t": [0, 0.33333, 0.2, 1],
            "pressure": [0.5, 0.7, 0.1, 2],
            "twist": [-1, 361, 0, 0],
            "tiltX": [1, 2]
        }))
        .unwrap();
        assert_eq!(ink.t, vec![0.0, 0.333, 1.0]);
        assert_eq!(ink.pressure.unwrap(), vec![0.5, 179.0 / 256.0, 1.0]);
        assert_eq!(ink.twist.unwrap(), vec![359.0, 1.0, 0.0]);
        assert!(ink.tilt_x.is_none()); // wrong length
        let wheel = parse_ink_channels(&json!({"t": [0, 0.5, 1], "wheel": [-0.5, 0.3, 1.5]})).unwrap();
        assert_eq!(wheel.wheel.unwrap(), vec![0.0, 77.0 / 256.0, 1.0]);
    }

    #[test]
    fn js_round_matches_math_round() {
        // Math.round results, including the cases `(v + 0.5).floor()` gets wrong.
        for (v, e) in [
            (0.49999999999999994, 0.0),
            (4503599627370497.0, 4503599627370497.0),
            (-0.5, 0.0),
            (-1.5, -1.0),
            (2.5, 3.0),
            (-2.4, -2.0),
        ] {
            assert_eq!(js_round(v), e, "js_round({v})");
        }
    }

    #[test]
    fn huge_anchor_is_dropped() {
        let m = parse_width_markers(&json!([
            {"id": "a", "t": 0, "left": 1, "anchor": 1e300},
            {"id": "b", "t": 1, "left": 1, "anchor": 4294967295u64}
        ]))
        .unwrap();
        assert_eq!(m[0].anchor, None);
        assert_eq!(m[1].anchor, Some(4_294_967_295));
    }
}
