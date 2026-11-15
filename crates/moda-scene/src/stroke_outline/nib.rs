//! Calligraphic nib (brushes.md §4.4, §4.6): the width at each point comes
//! from projecting the nib ellipse onto the path normal. Nib parameters are
//! `Dyn` values — fixed, random per stroke (seeded), or driven by an ink
//! channel (pressure, tilt, bearing, rotation, stylus wheel) read at the sample's `t`.
//!
//! Units (P4 review): angle in degrees (counter-clockwise on screen, 0 = the
//! nib's long axis along +x), roundness 0..1 (minor / major axis), size = nib
//! diameter along the major axis. Like Illustrator, the stroke weight scales
//! the brush: the nib is `size × strokeWidth` px (a 1 px stroke draws the nib
//! at its defined size).

use crate::value::Value;

use crate::path_geom::vec2::{v2, V2};
use crate::stroke_model::InkChannels;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vary {
    Random,
    Pressure,
    Tilt,
    Bearing,
    Rotation,
    /// The stylus wheel (airbrush finger wheel), 0..1.
    Wheel,
    /// Drawing speed (the `speed` channel): slow draws `max`, fast `min` (ink thins when flicked; an inverted
    /// range, `max < min`, makes a fast stroke thicker).
    Speed,
    /// Along the stroke: the start draws `max`, the end `min` (Krita's Fade; an inverted range swells toward the
    /// end). Needs no channel.
    Fade,
}

/// `Dyn` (nodes.ts): a value that can vary. `min` / `max` are kept as written: `max < min` is an inverted
/// mapping (the driver runs backwards), never swapped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dyn {
    pub value: f64,
    pub vary: Option<Vary>,
    pub min: f64,
    pub max: f64,
}

impl Dyn {
    pub fn fixed(value: f64) -> Dyn {
        Dyn {
            value,
            vary: None,
            min: value,
            max: value,
        }
    }

    /// Parse with a default and a clamp range (non-finite → default).
    pub fn parse(v: Option<&Value>, default: f64, lo: f64, hi: f64) -> Dyn {
        let clamp = |x: f64| x.clamp(lo, hi);
        let num = |k: &str, o: &crate::value::Map| {
            o.get(k).and_then(Value::as_f64).filter(|x| x.is_finite())
        };
        match v {
            Some(Value::Number(n)) => Dyn::fixed(clamp(
                n.as_f64().filter(|x| x.is_finite()).unwrap_or(default),
            )),
            Some(Value::Object(o)) => {
                let value = clamp(num("value", o).unwrap_or(default));
                let vary = match o.get("vary").and_then(Value::as_str) {
                    Some("random") => Some(Vary::Random),
                    Some("pressure") => Some(Vary::Pressure),
                    Some("tilt") => Some(Vary::Tilt),
                    Some("bearing") => Some(Vary::Bearing),
                    Some("rotation") => Some(Vary::Rotation),
                    Some("wheel") => Some(Vary::Wheel),
                    Some("speed") => Some(Vary::Speed),
                    Some("fade") => Some(Vary::Fade),
                    _ => None,
                };
                Dyn {
                    value,
                    vary,
                    min: clamp(num("min", o).unwrap_or(value)),
                    max: clamp(num("max", o).unwrap_or(value)),
                }
            }
            _ => Dyn::fixed(default),
        }
    }

    /// Evaluate at a point. `random01` is this stroke's seeded draw for this
    /// parameter.
    pub fn eval(&self, ink: &InkAt, random01: f64) -> f64 {
        let f = match self.vary {
            None => return self.value,
            Some(Vary::Random) => Some(random01),
            Some(Vary::Pressure) => ink.pressure,
            Some(Vary::Tilt) => ink.tilt.map(|t| t / 90.0),
            Some(Vary::Bearing) => ink.bearing.map(|b| b / 360.0),
            Some(Vary::Rotation) => ink.rotation.map(|r| r / 360.0),
            Some(Vary::Wheel) => ink.wheel,
            Some(Vary::Speed) => ink.speed.map(|s| 1.0 - s),
            Some(Vary::Fade) => ink.fade,
        };
        match f {
            Some(f) => self.min + (self.max - self.min) * f.clamp(0.0, 1.0),
            None => self.value,
        }
    }

    pub fn uses_ink(&self) -> bool {
        matches!(
            self.vary,
            Some(
                Vary::Pressure
                    | Vary::Tilt
                    | Vary::Bearing
                    | Vary::Rotation
                    | Vary::Wheel
                    | Vary::Speed
                    | Vary::Fade
            )
        )
    }

    fn largest(&self) -> f64 {
        match self.vary {
            None => self.value,
            Some(_) => self.value.max(self.min).max(self.max),
        }
    }
}

/// Ink channel values at one point.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct InkAt {
    /// 0..1
    pub pressure: Option<f64>,
    /// Pen tilt from vertical, 0..90°.
    pub tilt: Option<f64>,
    /// Direction the pen leans, 0..360°, counter-clockwise on screen from +x
    /// (converted from the clockwise Pointer Events frame).
    pub bearing: Option<f64>,
    /// Barrel rotation, 0..360°, counter-clockwise (converted from Pointer
    /// Events' clockwise `twist`).
    pub rotation: Option<f64>,
    /// Stylus wheel, 0..1.
    pub wheel: Option<f64>,
    /// Drawing speed, 0..1.
    pub speed: Option<f64>,
    /// 1 at the path's start … 0 at its end (`1 − t`): every stroke has it, with or without pen channels.
    pub fade: Option<f64>,
}

impl InkAt {
    /// No pen channels, only the position along the path (`t`, 0..1): what a stroke without ink reads.
    pub fn at_t(t: f64) -> InkAt {
        InkAt {
            fade: fade_at(t),
            ..InkAt::default()
        }
    }
}

fn fade_at(t: f64) -> Option<f64> {
    t.is_finite().then(|| 1.0 - t.clamp(0.0, 1.0))
}

fn lerp_channel(ts: &[f64], vs: &[f64], t: f64) -> Option<f64> {
    let n = ts.len().min(vs.len());
    if n == 0 {
        return None;
    }
    let i = ts[..n].partition_point(|x| *x < t);
    if i == 0 {
        return Some(vs[0]);
    }
    if i >= n {
        return Some(vs[n - 1]);
    }
    let (t0, t1) = (ts[i - 1], ts[i]);
    let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 1.0 };
    Some(vs[i - 1] + (vs[i] - vs[i - 1]) * f)
}

/// Like `lerp_channel`, for an angle in degrees: interpolates along the
/// shorter arc, so 350° → 10° passes through 0°, not 180°.
fn lerp_angle(ts: &[f64], vs: &[f64], t: f64) -> Option<f64> {
    let n = ts.len().min(vs.len());
    if n == 0 {
        return None;
    }
    let i = ts[..n].partition_point(|x| *x < t);
    if i == 0 {
        return Some(vs[0]);
    }
    if i >= n {
        return Some(vs[n - 1]);
    }
    let (t0, t1) = (ts[i - 1], ts[i]);
    let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 1.0 };
    let d = (vs[i] - vs[i - 1] + 540.0).rem_euclid(360.0) - 180.0;
    Some(vs[i - 1] + d * f)
}

/// Read the channels at `t` (fraction of the path's total length).
///
/// Direction convention (brushes.md §4.4): model angles are counter-clockwise
/// degrees on screen (Illustrator). Pointer Events report `twist` and the lean
/// direction implied by `tiltX` / `tiltY` in the y-down screen frame, i.e.
/// clockwise; both are converted here, so a nib angle driven by `bearing` or
/// `rotation` with `min 0` / `max 360` turns with the pen. A pen reporting
/// only one tilt axis has the other at 0.
pub fn ink_at(ink: &InkChannels, t: f64) -> InkAt {
    let ch = |c: &Option<Vec<f64>>| c.as_deref().and_then(|v| lerp_channel(&ink.t, v, t));
    let (tx, ty) = (ch(&ink.tilt_x), ch(&ink.tilt_y));
    let (tilt, bearing) = match (tx, ty) {
        (None, None) => (None, None),
        (x, y) => {
            let (a, b) = (
                x.unwrap_or(0.0).to_radians().tan(),
                y.unwrap_or(0.0).to_radians().tan(),
            );
            let tilt = a.hypot(b).atan().to_degrees();
            // atan2 on the y-down frame is clockwise; negate for CCW.
            let bearing = if a == 0.0 && b == 0.0 {
                0.0
            } else {
                (-b.atan2(a).to_degrees()).rem_euclid(360.0)
            };
            (Some(tilt), Some(bearing))
        }
    };
    InkAt {
        pressure: ch(&ink.pressure).map(|p| p.clamp(0.0, 1.0)),
        tilt: tilt.filter(|x| x.is_finite()),
        bearing: bearing.filter(|x| x.is_finite()),
        rotation: ink
            .twist
            .as_deref()
            .and_then(|v| lerp_angle(&ink.t, v, t))
            .map(|r| (-r).rem_euclid(360.0))
            .filter(|x| x.is_finite()),
        wheel: ch(&ink.wheel).map(|w| w.clamp(0.0, 1.0)),
        speed: ch(&ink.speed).map(|s| s.clamp(0.0, 1.0)),
        fade: fade_at(t),
    }
}

/// A calligraphic brush definition's nib.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nib {
    pub angle: Dyn,
    pub roundness: Dyn,
    pub size: Dyn,
    /// `angleRelativeTo: 'path'` (our extension; Illustrator's nib is fixed
    /// to the page): the angle is measured from the path's direction, so the
    /// nib turns with the path.
    pub angle_relative_to_path: bool,
}

/// Largest nib diameter accepted (px, before the stroke-weight scale).
const MAX_NIB: f64 = 1e5;

impl Nib {
    /// A `BrushDefinition` of type `calligraphic`; any other type → `None`.
    pub fn from_brush(v: &Value) -> Option<Nib> {
        let o = v.as_object()?;
        if o.get("type").and_then(Value::as_str) != Some("calligraphic") {
            return None;
        }
        Some(Nib {
            angle: Dyn::parse(o.get("angle"), 0.0, -1e6, 1e6),
            roundness: Dyn::parse(o.get("roundness"), 1.0, 0.0, 1.0),
            size: Dyn::parse(o.get("size"), 1.0, 0.0, MAX_NIB),
            angle_relative_to_path: o.get("angleRelativeTo").and_then(Value::as_str) == Some("path"),
        })
    }

    /// This stroke's `BrushApplication.overrides` ("Options of Selected
    /// Object"): `angle`, `roundness` and `size` replace the brush's.
    pub fn with_overrides(mut self, overrides: Option<&crate::value::Map>) -> Nib {
        if let Some(o) = overrides {
            if let Some(v) = o.get("angle") {
                self.angle = Dyn::parse(Some(v), self.angle.value, -1e6, 1e6);
            }
            if let Some(v) = o.get("roundness") {
                self.roundness = Dyn::parse(Some(v), self.roundness.value, 0.0, 1.0);
            }
            if let Some(v) = o.get("size") {
                self.size = Dyn::parse(Some(v), self.size.value, 0.0, MAX_NIB);
            }
        }
        self
    }

    pub fn uses_ink(&self) -> bool {
        self.angle.uses_ink() || self.roundness.uses_ink() || self.size.uses_ink()
    }

    /// Upper bound of the nib's half extent (for bounds).
    pub fn max_half(&self) -> f64 {
        self.size.largest() * 0.5
    }

    /// The nib ellipse at a point: `(a, b, u)` = semi-axis along the major
    /// axis, semi-axis along the minor axis (both in nib-size units: a 1-px
    /// stroke weight draws them as px), and the major axis direction (unit).
    /// Roundness is floored at 1e-3 so a flat nib stays a (thin) ellipse.
    /// `heading` is the path's direction there (unit, y-down canvas); a
    /// path-relative nib adds its angle (none: the page).
    pub fn ellipse(&self, ink: &InkAt, random: [f64; 3], heading: Option<V2>) -> (f64, f64, V2) {
        let mut angle = self.angle.eval(ink, random[0]).to_radians();
        if self.angle_relative_to_path {
            if let Some(h) = heading.filter(|h| h.x.is_finite() && h.y.is_finite() && (h.x != 0.0 || h.y != 0.0)) {
                // Screen-CCW direction of travel on a y-down canvas.
                angle += (-h.y).atan2(h.x);
            }
        }
        let round = self.roundness.eval(ink, random[1]).clamp(1e-3, 1.0);
        let size = self.size.eval(ink, random[2]).max(0.0);
        // Screen-CCW angle on a y-down canvas.
        let major = v2(angle.cos(), -angle.sin());
        (0.5 * size, 0.5 * size * round, major)
    }
}

/// The ellipse's minor-axis direction for major axis `u` (a right-handed frame
/// in path coordinates, so parameter angles turn the same way as directions).
pub fn minor_axis(u: V2) -> V2 {
    v2(-u.y, u.x)
}

/// Support point of the ellipse `(a, b, u)` centred at 0 in direction `n`: the
/// boundary point farthest along `n` (where the pen touches its envelope).
pub fn support(a: f64, b: f64, u: V2, n: V2) -> V2 {
    let v = minor_axis(u);
    let (du, dv) = (n.dot(u), n.dot(v));
    let den = (a * a * du * du + b * b * dv * dv).sqrt();
    if !(den > 0.0) {
        return V2::default();
    }
    (u * (a * a * du) + v * (b * b * dv)) * (1.0 / den)
}

/// Parameter angle of the support point in direction `n`.
pub fn support_param(a: f64, b: f64, u: V2, n: V2) -> f64 {
    (b * n.dot(minor_axis(u))).atan2(a * n.dot(u))
}

/// Ellipse boundary point at parameter `phi`.
pub fn ellipse_point(a: f64, b: f64, u: V2, phi: f64) -> V2 {
    u * (a * phi.cos()) + minor_axis(u) * (b * phi.sin())
}

/// splitmix64 → [0, 1).
pub fn hash01(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}
