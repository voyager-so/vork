//! Dynamic stroke (Figma's DYNAMIC; brushes.md §4.3): a seeded, hand-drawn
//! wobble of the centerline, applied BEFORE widths so it combines with
//! profiles and brushes.
//!
//! `frequency` = wobbles per 100 px (one crest and one trough each — measured by
//! `wobble_frequency_is_wobbles_per_100px`), `wiggle` = amplitude in px,
//! `smoothen` 0..1 fades a finer, irregular octave on top. The main octave is
//! a lattice of `50 / frequency` px whose values alternate side with a seeded
//! random amplitude (so every two cells is exactly one wobble, of varying
//! size), quintic-interpolated; closed subpaths use a periodic lattice with an
//! even number of cells so the seam is seamless. Deterministic in (seed,
//! subpath index): the same document renders the same wobble everywhere.
//! Figma paste maps `dynamicStrokeSettings` 1:1 (frequency 0.01–20, wiggle px,
//! smoothen 0–1); Figma documents no unit for its frequency, so a pasted
//! stroke wobbles at our "per 100 px" rate.

use super::centerline::{Subpath, EPS_LEN};
use super::nib::hash01;
use crate::path_geom::vec2::V2;
use crate::stroke_model::StrokeDynamic;

/// Main lattice spacing in px (half a wobble).
pub fn wavelength(d: &StrokeDynamic) -> f64 {
    50.0 / d.frequency.max(1e-6)
}

fn lattice(seed: u64, octave: u64, i: i64, period: Option<i64>) -> f64 {
    let i = match period {
        Some(p) if p > 0 => i.rem_euclid(p),
        _ => i,
    };
    let r = hash01(seed ^ (octave << 56) ^ (i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
    if octave == 0 {
        // Alternating sides, random amplitude 0.35..1.
        let side = if i.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
        side * (0.35 + 0.65 * r)
    } else {
        r * 2.0 - 1.0
    }
}

fn value_noise(seed: u64, octave: u64, x: f64, period: Option<i64>) -> f64 {
    let i = x.floor();
    let f = x - i;
    let i = i as i64;
    let s = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = lattice(seed, octave, i, period);
    let b = lattice(seed, octave, i + 1, period);
    a + (b - a) * s
}

/// Displace a subpath's samples along their normals and recompute tangents.
pub(crate) fn displace(sub: &mut Subpath, d: &StrokeDynamic, seed: u32) {
    if !(d.wiggle > 0.0) || sub.samples.len() < 2 || !(sub.length > 0.0) {
        return;
    }
    let lambda = wavelength(d);
    let seed = (seed as u64) ^ ((sub.index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let (scale, period) = if sub.closed {
        // An even number of cells, so the alternation closes on itself.
        let cells = 2.0 * (sub.length / (2.0 * lambda)).round().max(1.0);
        (cells / sub.length, Some(cells as i64))
    } else {
        (1.0 / lambda, None)
    };
    let fine = (1.0 - d.smoothen.clamp(0.0, 1.0)) * 0.5;
    let norm = 1.0 / (1.0 + fine);
    for smp in sub.samples.iter_mut() {
        let x = smp.s_local * scale;
        let mut n = value_noise(seed, 0, x, period);
        if fine > 0.0 {
            n += fine * value_noise(seed, 1, x * 2.0, period.map(|p| p * 2));
        }
        let normal = (smp.t_in.left_normal() + smp.t_out.left_normal())
            .unit()
            .unwrap_or(smp.t_out.left_normal());
        smp.p = smp.p + normal * (d.wiggle * n * norm);
    }
    retangent(sub);
}

/// Tangents from the (displaced) sample polyline.
fn retangent(sub: &mut Subpath) {
    let n = sub.samples.len();
    let closed = sub.closed;
    let pts: Vec<V2> = sub.samples.iter().map(|s| s.p).collect();
    let dir = |from: usize, step: isize| -> Option<V2> {
        let mut j = from as isize;
        for _ in 0..n {
            j += step;
            if j < 0 || j >= n as isize {
                if !closed {
                    return None;
                }
                j = j.rem_euclid(n as isize);
            }
            let d = if step > 0 {
                pts[j as usize] - pts[from]
            } else {
                pts[from] - pts[j as usize]
            };
            if d.len() > EPS_LEN {
                return d.unit();
            }
        }
        None
    };
    for i in 0..n {
        let back = dir(i, -1);
        let fwd = dir(i, 1);
        let smp = &mut sub.samples[i];
        if smp.corner {
            if let Some(b) = back {
                smp.t_in = b;
            }
            if let Some(f) = fwd {
                smp.t_out = f;
            }
            continue;
        }
        let t = match (back, fwd) {
            (Some(b), Some(f)) => (b + f).unit().unwrap_or(f),
            (Some(b), None) => b,
            (None, Some(f)) => f,
            (None, None) => smp.t_out,
        };
        smp.t_in = t;
        smp.t_out = t;
    }
}
