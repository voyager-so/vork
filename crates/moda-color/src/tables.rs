//! The adjustments' 256-entry lookup tables, over encoded 8-bit values (gamma sRGB, as Photoshop applies them).
//! Built once per request (never per pixel) with deterministic maths only, so every target computes the same bytes.
//! Callers pass parameters already validated (clamped and quantised): moda-paint quantises to 1/65536 when it parses
//! a request, the engine when it reads a backdrop kind.

use crate::fmath;

/// One channel's table: `table[v]` for an encoded value `v`.
pub type Table = [u8; 256];

pub fn identity() -> Table {
    std::array::from_fn(|i| i as u8)
}

/// `f` over 0..1, sampled at the 256 encoded values and rounded.
pub fn table(f: impl Fn(f64) -> f64) -> Table {
    std::array::from_fn(|i| (f(i as f64 / 255.0) * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8)
}

/// Rec. 709 luma of encoded values, 0..=255.
#[inline(always)]
pub fn luma(r: u8, g: u8, b: u8) -> u32 {
    (2126 * r as u32 + 7152 * g as u32 + 722 * b as u32 + 5000) / 10000
}

/// Monotone cubic (Fritsch–Carlson) through `points` (0..255 each, at least one), sampled at 0..=255.
pub fn curve_table(points: &[(f64, f64)]) -> Result<Table, String> {
    let mut p: Vec<(f64, f64)> = points.to_vec();
    if p.is_empty() {
        return Err("curves: give at least one point".into());
    }
    p.sort_by(|a, b| a.0.total_cmp(&b.0));
    p.dedup_by(|b, a| {
        if a.0 == b.0 {
            a.1 = b.1;
            true
        } else {
            false
        }
    });
    let n = p.len();
    if n == 1 {
        return Ok([p[0].1.round() as u8; 256]);
    }
    let d: Vec<f64> = (0..n - 1).map(|k| (p[k + 1].1 - p[k].1) / (p[k + 1].0 - p[k].0)).collect();
    let mut m = vec![0.0; n];
    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for k in 1..n - 1 {
        m[k] = if d[k - 1] * d[k] <= 0.0 { 0.0 } else { (d[k - 1] + d[k]) / 2.0 };
    }
    for k in 0..n - 1 {
        if d[k] == 0.0 {
            m[k] = 0.0;
            m[k + 1] = 0.0;
        } else {
            let (a, b) = (m[k] / d[k], m[k + 1] / d[k]);
            let s = a * a + b * b;
            if s > 9.0 {
                let t = 3.0 / s.sqrt();
                m[k] = t * a * d[k];
                m[k + 1] = t * b * d[k];
            }
        }
    }
    Ok(std::array::from_fn(|i| {
        let x = i as f64;
        let y = if x <= p[0].0 {
            p[0].1
        } else if x >= p[n - 1].0 {
            p[n - 1].1
        } else {
            let k = (0..n - 1).find(|&k| x < p[k + 1].0).unwrap_or(n - 2);
            let h = p[k + 1].0 - p[k].0;
            let t = (x - p[k].0) / h;
            let (t2, t3) = (t * t, t * t * t);
            (2.0 * t3 - 3.0 * t2 + 1.0) * p[k].1 + (t3 - 2.0 * t2 + t) * h * m[k] + (-2.0 * t3 + 3.0 * t2) * p[k + 1].1 + (t3 - t2) * h * m[k + 1]
        };
        (y + 0.5).floor().clamp(0.0, 255.0) as u8
    }))
}

/// Levels: input black / white and output black / white 0..255, gamma 0.01..9.99.
pub fn levels_table(in_black: f64, in_white: f64, gamma: f64, out_black: f64, out_white: f64) -> Table {
    let (ib, iw) = (in_black / 255.0, in_white / 255.0);
    let (ob, ow) = (out_black / 255.0, out_white / 255.0);
    table(|v| {
        let t = if iw > ib {
            ((v - ib) / (iw - ib)).clamp(0.0, 1.0)
        } else if v >= ib {
            1.0
        } else {
            0.0
        };
        ob + fmath::pow(t, 1.0 / gamma) * (ow - ob)
    })
}

/// Brightness bends the tone curve with a power (`v^γ`, γ = 2^(−brightness / 100)), so black and white stay put;
/// contrast is a symmetric S-curve about mid grey (`0.5 (2v)^k` below it, mirrored above, k = 2^(contrast / 50)), so
/// nothing clips. Both −100..100.
pub fn brightness_contrast_table(brightness: f64, contrast: f64) -> Table {
    let gamma = fmath::exp2(-brightness / 100.0);
    let k = fmath::exp2(contrast / 50.0);
    table(|v| {
        let v = fmath::pow(v, gamma);
        if v < 0.5 {
            0.5 * fmath::pow(2.0 * v, k)
        } else {
            1.0 - 0.5 * fmath::pow(2.0 * (1.0 - v), k)
        }
    })
}

/// Exposure in linear light: stops (−20..20), an offset (−0.5..0.5) and a gamma (0.01..9.99).
pub fn exposure_table(exposure: f64, offset: f64, gamma: f64) -> Table {
    let k = fmath::exp2(exposure);
    table(|v| {
        let y = (fmath::srgb_to_linear(v) * k + offset).max(0.0);
        fmath::linear_to_srgb(fmath::pow(y, 1.0 / gamma).min(1.0))
    })
}

/// Posterize to `levels` (clamped to 2..255) evenly spaced values.
pub fn posterize_table(levels: u32) -> Table {
    let n = levels.clamp(2, 255);
    std::array::from_fn(|v| {
        let k = (v as u32 * (n - 1) + 127) / 255;
        ((k * 255 + (n - 1) / 2) / (n - 1)) as u8
    })
}

/// A gradient map's row: the colour for each luma 0..255, from stops (`at` 0..1, RGB), linear between them.
pub fn gradient_row(stops: &[(f64, [u8; 3])], reverse: bool) -> Result<[[u8; 3]; 256], String> {
    if stops.is_empty() {
        return Err("gradient-map: give at least one stop".into());
    }
    let mut s = stops.to_vec();
    s.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(std::array::from_fn(|i| {
        let t = i as f64 / 255.0;
        let t = if reverse { 1.0 - t } else { t };
        let last = s.len() - 1;
        if t <= s[0].0 {
            return s[0].1;
        }
        if t >= s[last].0 {
            return s[last].1;
        }
        let k = (0..last).find(|&k| t < s[k + 1].0).unwrap_or(last - 1);
        let (a, b) = (s[k], s[k + 1]);
        let u = if b.0 > a.0 { (t - a.0) / (b.0 - a.0) } else { 1.0 };
        std::array::from_fn(|c| (a.1[c] as f64 + (b.1[c] as f64 - a.1[c] as f64) * u + 0.5).floor() as u8)
    }))
}
