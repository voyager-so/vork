//! Width along the stroke: the width profile (via `stroke_model`'s shared
//! evaluator), taper and wave (brushes.md §4.2–4.3, "Settled in P4 review").
//!
//! Units: a marker side of 1 is the plain stroke, so the half-width on a side is
//! `strokeWidth / 2 × side`. `t` for markers is a fraction of the total length
//! of all subpaths; presets, taper and wave apply per subpath.

use crate::stroke_model::{
    PreparedMarker, PreparedWidthProfile, StrokeTaper, StrokeWave, TaperUnits, WaveUnits,
    WidthProfile,
};

/// The width profile, prepared once per outline: `stroke_model`'s
/// `PreparedWidthProfile` (effective markers — preset table or stored markers,
/// anchors resolved, flips applied — with PCHIP tangents; binary search per
/// sample). That is the ONE width evaluator, shared with the Width tool and the
/// TS mirror; this wrapper only adds what the sampler needs: which knots to hit
/// exactly, and how dense to sample.
#[derive(Debug, Clone)]
pub struct ProfileCurve {
    prepared: Option<PreparedWidthProfile>,
    /// Presets apply per subpath (local t); markers over the whole path.
    pub per_subpath: bool,
}

impl ProfileCurve {
    /// No profile: the plain stroke (1 on both sides).
    pub fn uniform() -> ProfileCurve {
        ProfileCurve {
            prepared: None,
            per_subpath: true,
        }
    }

    /// `periodic`: a single closed subpath, whose profile wraps smoothly
    /// through the seam (`AnchorFractions::is_periodic`).
    pub fn from_model(p: &WidthProfile, anchor_t: &[Option<f64>], periodic: bool) -> ProfileCurve {
        let prepared = PreparedWidthProfile::new_with(p, Some(anchor_t), periodic);
        ProfileCurve {
            per_subpath: prepared.per_subpath,
            prepared: Some(prepared),
        }
    }

    fn markers(&self) -> &[PreparedMarker] {
        self.prepared
            .as_ref()
            .map(|p| p.markers.as_slice())
            .unwrap_or(&[])
    }

    /// `(left, right)` at `u` (the profile's own 0..1: path fraction for
    /// markers, subpath fraction for presets). `after` picks the outgoing side
    /// of a discontinuity exactly at `u`.
    pub fn eval(&self, u: f64, after: bool) -> (f64, f64) {
        let u = if u.is_finite() {
            u.clamp(0.0, 1.0)
        } else {
            0.0
        };
        match &self.prepared {
            Some(p) => p.width_at_u(u, !after),
            None => (1.0, 1.0),
        }
    }

    /// Largest side value anywhere (PCHIP never overshoots its knots).
    pub fn max_side(&self) -> f64 {
        match &self.prepared {
            Some(p) if !p.markers.is_empty() => p
                .markers
                .iter()
                .map(|k| k.left.max(k.right))
                .fold(0.0, f64::max),
            _ => 1.0,
        }
    }

    /// Knot positions a sampler must hit, as `(u, discontinuous)`.
    pub fn events(&self) -> Vec<(f64, bool)> {
        let m = self.markers();
        let mut out: Vec<(f64, bool)> = Vec::new();
        for (i, kn) in m.iter().enumerate() {
            if kn.t <= 0.0 || kn.t >= 1.0 {
                continue;
            }
            if i > 0 && m[i - 1].t == kn.t {
                if let Some(last) = out.last_mut() {
                    last.1 = true;
                }
                continue;
            }
            out.push((kn.t, false));
        }
        out
    }

    /// Smallest gap between distinct knots (drives sampling density).
    pub fn min_gap(&self) -> f64 {
        self.markers()
            .windows(2)
            .map(|w| w[1].t - w[0].t)
            .filter(|g| *g > 0.0)
            .fold(1.0, f64::min)
    }
}

/// Taper ease: `u` = 0 at the tip, 1 where the taper ends. `ease` −1 pointy
/// (u²) … 0 linear … 1 round (quarter circle).
fn taper_shape(u: f64, ease: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    if ease >= 0.0 {
        let round = (1.0 - (1.0 - u) * (1.0 - u)).max(0.0).sqrt();
        u + (round - u) * ease.min(1.0)
    } else {
        u + (u * u - u) * (-ease).min(1.0)
    }
}

/// Taper factor at `s` along a subpath of length `len`.
pub fn taper_factor(t: &StrokeTaper, s: f64, len: f64) -> f64 {
    let to_px = |v: f64| match t.units {
        TaperUnits::Percent => len * v / 100.0,
        TaperUnits::Px => v,
    };
    let (ls, le) = (to_px(t.start_length), to_px(t.end_length));
    let mut f: f64 = 1.0;
    if ls > 0.0 && s < ls {
        f = f.min(t.start_width + (1.0 - t.start_width) * taper_shape(s / ls, t.start_ease));
    }
    let from_end = len - s;
    if le > 0.0 && from_end < le {
        f = f.min(t.end_width + (1.0 - t.end_width) * taper_shape(from_end / le, t.end_ease));
    }
    f.max(0.0)
}

/// Taper boundaries within a subpath (arc length from its start).
pub fn taper_events(t: &StrokeTaper, len: f64) -> Vec<f64> {
    let to_px = |v: f64| match t.units {
        TaperUnits::Percent => len * v / 100.0,
        TaperUnits::Px => v,
    };
    [to_px(t.start_length), len - to_px(t.end_length)]
        .into_iter()
        .filter(|s| *s > 0.0 && *s < len)
        .collect()
}

pub fn taper_min_length(t: &StrokeTaper, len: f64) -> f64 {
    let to_px = |v: f64| match t.units {
        TaperUnits::Percent => len * v / 100.0,
        TaperUnits::Px => v,
    };
    [to_px(t.start_length), to_px(t.end_length)]
        .into_iter()
        .filter(|l| *l > 0.0)
        .fold(f64::INFINITY, f64::min)
}

/// Wavelength in px for a subpath of length `len`.
pub fn wave_length(w: &StrokeWave, len: f64) -> f64 {
    match w.units {
        WaveUnits::Px => w.wavelength,
        WaveUnits::Cycles => {
            if w.wavelength > 0.0 {
                len / w.wavelength
            } else {
                len
            }
        }
    }
}

/// Wave factor `1 + amount × sin(2π s / λ + phase)`, never below 0.
pub fn wave_factor(w: &StrokeWave, s: f64, len: f64) -> f64 {
    let lambda = wave_length(w, len);
    if !(lambda > 0.0) || w.amount == 0.0 {
        return 1.0;
    }
    let phase = w.phase.to_radians();
    (1.0 + w.amount * (std::f64::consts::TAU * s / lambda + phase).sin()).max(0.0)
}
