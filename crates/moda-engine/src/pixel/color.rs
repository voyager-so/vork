//! Colour conversions at typed edges (motion-model.md §10.1, point 3): transfers, primaries, white points, and the
//! BT.709 / BT.601 Y′CbCr matrices and ranges for decoded video.
//!
//! This is the CPU reference: the GPU `convert` kernel ([`super::kernels`]) takes its matrices from
//! [`conversion`] and implements the same transfers, and a test holds the two together.
//!
//! Conventions:
//! - graph images are premultiplied; a conversion un-premultiplies, converts colour, and premultiplies again;
//! - linear values are relative to diffuse (SDR reference) white = 1.0. For PQ and HLG that white is the BT.2408
//!   reference, 203 cd/m² (PQ) and a signal of 0.75 (HLG, scene light, no OOTF), so an SDR design dropped into an
//!   HDR image lands at reference white;
//! - transfers mirror negative values (extended sRGB), so out-of-gamut colours survive a round trip.

use super::image::ColorTag;

/// A transfer function (encoded ↔ linear).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transfer {
    Linear,
    Srgb,
    Pq,
    Hlg,
}

impl Transfer {
    /// The GPU kernel's id for this transfer.
    pub const fn id(self) -> u32 {
        match self {
            Transfer::Linear => 0,
            Transfer::Srgb => 1,
            Transfer::Pq => 2,
            Transfer::Hlg => 3,
        }
    }
}

/// Primaries and white.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gamut {
    Rec709,
    P3D65,
    Rec2020,
    /// ACES AP1 (ACEScg), D60 white.
    Ap1,
}

pub fn tag_parts(tag: ColorTag) -> Option<(Gamut, Transfer)> {
    Some(match tag {
        ColorTag::Srgb => (Gamut::Rec709, Transfer::Srgb),
        ColorTag::LinearSrgb => (Gamut::Rec709, Transfer::Linear),
        ColorTag::DisplayP3 => (Gamut::P3D65, Transfer::Srgb),
        ColorTag::LinearP3 => (Gamut::P3D65, Transfer::Linear),
        ColorTag::Rec2020Pq => (Gamut::Rec2020, Transfer::Pq),
        ColorTag::Rec2020Hlg => (Gamut::Rec2020, Transfer::Hlg),
        ColorTag::AcesCg => (Gamut::Ap1, Transfer::Linear),
        ColorTag::Data => return None,
    })
}

// ------------------------------------------------------------------------------------------------------------
// Transfers

/// PQ: 1.0 in the graph is 203 cd/m² (BT.2408 reference white); PQ's 1.0 signal is 10,000 cd/m².
pub const PQ_REFERENCE_WHITE_NITS: f64 = 203.0;
const PQ_M1: f64 = 2610.0 / 16384.0;
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f64 = 3424.0 / 4096.0;
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0;

const HLG_A: f64 = 0.178_832_77;
const HLG_B: f64 = 0.284_668_92;
const HLG_C: f64 = 0.559_910_73;

fn hlg_inverse_oetf(e: f64) -> f64 {
    if e <= 0.5 {
        e * e / 3.0
    } else {
        ((e - HLG_C) / HLG_A).exp() / 12.0 + HLG_B / 12.0
    }
}

fn hlg_oetf(l: f64) -> f64 {
    if l <= 1.0 / 12.0 {
        (3.0 * l).sqrt()
    } else {
        HLG_A * (12.0 * l - HLG_B).ln() + HLG_C
    }
}

/// HLG: 1.0 in the graph is the scene light of a 0.75 signal (BT.2408 reference white).
pub fn hlg_reference_white() -> f64 {
    hlg_inverse_oetf(0.75)
}

fn mirror(v: f64, f: impl Fn(f64) -> f64) -> f64 {
    if v < 0.0 {
        -f(-v)
    } else {
        f(v)
    }
}

/// Encoded → linear (graph-relative, reference white = 1.0).
pub fn decode(t: Transfer, v: f64) -> f64 {
    match t {
        Transfer::Linear => v,
        Transfer::Srgb => mirror(v, |c| {
            if c <= 0.040_45 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        }),
        // A PQ signal past 1 is outside the curve (10,000 cd/m² is the top); clamp it.
        Transfer::Pq => mirror(v, |e| {
            let p = e.min(1.0).powf(1.0 / PQ_M2);
            let y = ((p - PQ_C1).max(0.0) / (PQ_C2 - PQ_C3 * p)).powf(1.0 / PQ_M1);
            y * 10_000.0 / PQ_REFERENCE_WHITE_NITS
        }),
        Transfer::Hlg => mirror(v, |e| hlg_inverse_oetf(e) / hlg_reference_white()),
    }
}

/// Linear (graph-relative) → encoded.
pub fn encode(t: Transfer, v: f64) -> f64 {
    match t {
        Transfer::Linear => v,
        Transfer::Srgb => mirror(v, |l| {
            if l <= 0.003_130_8 {
                l * 12.92
            } else {
                1.055 * l.powf(1.0 / 2.4) - 0.055
            }
        }),
        Transfer::Pq => mirror(v, |l| {
            let y = l * PQ_REFERENCE_WHITE_NITS / 10_000.0;
            let p = y.powf(PQ_M1);
            ((PQ_C1 + PQ_C2 * p) / (1.0 + PQ_C3 * p)).powf(PQ_M2)
        }),
        Transfer::Hlg => mirror(v, |l| hlg_oetf(l * hlg_reference_white())),
    }
}

// ------------------------------------------------------------------------------------------------------------
// Primaries

pub type Mat3 = [[f64; 3]; 3];

pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

const D65: (f64, f64) = (0.3127, 0.3290);
const D60: (f64, f64) = (0.32168, 0.33767);

fn chromaticities(g: Gamut) -> ([(f64, f64); 3], (f64, f64)) {
    match g {
        Gamut::Rec709 => ([(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)], D65),
        Gamut::P3D65 => ([(0.680, 0.320), (0.265, 0.690), (0.150, 0.060)], D65),
        Gamut::Rec2020 => ([(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)], D65),
        Gamut::Ap1 => ([(0.713, 0.293), (0.165, 0.830), (0.128, 0.044)], D60),
    }
}

pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    m
}

pub fn apply(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

pub fn invert(m: &Mat3) -> Mat3 {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let inv = 1.0 / det;
    let c = |a: usize, b: usize, c: usize, d: usize| m[a][b] * m[c][d] - m[a][d] * m[c][b];
    [
        [c(1, 1, 2, 2) * inv, -c(0, 1, 2, 2) * inv, c(0, 1, 1, 2) * inv],
        [-c(1, 0, 2, 2) * inv, c(0, 0, 2, 2) * inv, -c(0, 0, 1, 2) * inv],
        [c(1, 0, 2, 1) * inv, -c(0, 0, 2, 1) * inv, c(0, 0, 1, 1) * inv],
    ]
}

fn xyz_of((x, y): (f64, f64)) -> [f64; 3] {
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// RGB → XYZ for `g`, in its own white.
pub fn rgb_to_xyz(g: Gamut) -> Mat3 {
    let (p, w) = chromaticities(g);
    let cols = p.map(xyz_of);
    let m = [
        [cols[0][0], cols[1][0], cols[2][0]],
        [cols[0][1], cols[1][1], cols[2][1]],
        [cols[0][2], cols[1][2], cols[2][2]],
    ];
    let s = apply(&invert(&m), xyz_of(w));
    [0, 1, 2].map(|i| [m[i][0] * s[0], m[i][1] * s[1], m[i][2] * s[2]])
}

/// Bradford chromatic adaptation from white `from` to `to`.
fn bradford(from: (f64, f64), to: (f64, f64)) -> Mat3 {
    const B: Mat3 = [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]];
    let (s, d) = (apply(&B, xyz_of(from)), apply(&B, xyz_of(to)));
    let scale = [[d[0] / s[0], 0.0, 0.0], [0.0, d[1] / s[1], 0.0], [0.0, 0.0, d[2] / s[2]]];
    mul(&invert(&B), &mul(&scale, &B))
}

/// Linear RGB in `from` → linear RGB in `to`, adapting the white.
pub fn gamut_matrix(from: Gamut, to: Gamut) -> Mat3 {
    if from == to {
        return IDENTITY;
    }
    let (_, wf) = chromaticities(from);
    let (_, wt) = chromaticities(to);
    let mut m = rgb_to_xyz(from);
    if wf != wt {
        m = mul(&bradford(wf, wt), &m);
    }
    mul(&invert(&rgb_to_xyz(to)), &m)
}

/// What the GPU `convert` kernel needs: decode `from`, the matrix, encode `to`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conversion {
    pub decode: Transfer,
    pub matrix: Mat3,
    pub encode: Transfer,
}

/// The conversion between two colour tags; `None` when either is [`ColorTag::Data`] (data is never converted).
pub fn conversion(from: ColorTag, to: ColorTag) -> Option<Conversion> {
    let (gf, tf) = tag_parts(from)?;
    let (gt, tt) = tag_parts(to)?;
    Some(Conversion {
        decode: tf,
        matrix: gamut_matrix(gf, gt),
        encode: tt,
    })
}

/// Convert one premultiplied RGBA pixel (the CPU reference for the GPU kernel).
pub fn convert_premultiplied(c: &Conversion, px: [f64; 4]) -> [f64; 4] {
    let a = px[3];
    if a <= 0.0 {
        return [0.0, 0.0, 0.0, a.max(0.0)];
    }
    let straight = [px[0] / a, px[1] / a, px[2] / a].map(|v| decode(c.decode, v));
    let rgb = apply(&c.matrix, straight).map(|v| encode(c.encode, v) * a);
    [rgb[0], rgb[1], rgb[2], a]
}

// ------------------------------------------------------------------------------------------------------------
// Y′CbCr for decoded video

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YcbcrMatrix {
    Bt709,
    Bt601,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YcbcrRange {
    /// Y′ 16–235, Cb/Cr 16–240 (8-bit codes).
    Limited,
    Full,
}

fn kr_kb(m: YcbcrMatrix) -> (f64, f64) {
    match m {
        YcbcrMatrix::Bt709 => (0.2126, 0.0722),
        YcbcrMatrix::Bt601 => (0.299, 0.114),
    }
}

/// Normalized Y′CbCr codes (0–1, as a video texture samples them) → encoded R′G′B′ (0–1).
pub fn ycbcr_to_rgb(y: f64, cb: f64, cr: f64, m: YcbcrMatrix, range: YcbcrRange) -> [f64; 3] {
    let (y, cb, cr) = match range {
        YcbcrRange::Limited => (
            (y * 255.0 - 16.0) / 219.0,
            (cb * 255.0 - 128.0) / 224.0,
            (cr * 255.0 - 128.0) / 224.0,
        ),
        YcbcrRange::Full => (y, cb - 128.0 / 255.0, cr - 128.0 / 255.0),
    };
    let (kr, kb) = kr_kb(m);
    let kg = 1.0 - kr - kb;
    let r = y + 2.0 * (1.0 - kr) * cr;
    let b = y + 2.0 * (1.0 - kb) * cb;
    let g = (y - kr * r - kb * b) / kg;
    [r, g, b]
}

/// Encoded R′G′B′ (0–1) → normalized Y′CbCr codes: the inverse of [`ycbcr_to_rgb`].
pub fn rgb_to_ycbcr(rgb: [f64; 3], m: YcbcrMatrix, range: YcbcrRange) -> [f64; 3] {
    let (kr, kb) = kr_kb(m);
    let kg = 1.0 - kr - kb;
    let y = kr * rgb[0] + kg * rgb[1] + kb * rgb[2];
    let cb = (rgb[2] - y) / (2.0 * (1.0 - kb));
    let cr = (rgb[0] - y) / (2.0 * (1.0 - kr));
    match range {
        YcbcrRange::Limited => [
            (16.0 + 219.0 * y) / 255.0,
            (128.0 + 224.0 * cb) / 255.0,
            (128.0 + 224.0 * cr) / 255.0,
        ],
        YcbcrRange::Full => [y, cb + 128.0 / 255.0, cr + 128.0 / 255.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn transfers_round_trip() {
        for t in [Transfer::Linear, Transfer::Srgb, Transfer::Pq, Transfer::Hlg] {
            // PQ signals stop at 1 (clamped past it).
            let top = if t == Transfer::Pq { 100 } else { 120 };
            for i in -20..=top {
                let v = i as f64 / 100.0;
                let back = encode(t, decode(t, v));
                // PQ's curve isn't exactly 0 at 0: encode(0) = c1^m2 ≈ 7e-7.
                let eps = if t == Transfer::Pq { 1e-6 } else { 1e-9 };
                assert!(close(back, v, eps), "{t:?} {v} → {back}");
            }
        }
    }

    #[test]
    fn transfer_reference_points() {
        // sRGB 0.5 ≈ 0.214 linear; 1 → 1.
        assert!(close(decode(Transfer::Srgb, 0.5), 0.214_041, 1e-6));
        assert!(close(decode(Transfer::Srgb, 1.0), 1.0, 1e-12));
        // PQ: 203 cd/m² (graph 1.0) encodes to ≈ 0.5807 (BT.2408); 10,000 cd/m² is 1.0.
        assert!(close(encode(Transfer::Pq, 1.0), 0.5807, 1e-4));
        assert!(close(decode(Transfer::Pq, 1.0), 10_000.0 / 203.0, 1e-6));
        // HLG: the 0.75 reference signal is graph white; the OETF is continuous at 1/12.
        assert!(close(decode(Transfer::Hlg, 0.75), 1.0, 1e-12));
        assert!(close(hlg_oetf(1.0 / 12.0), 0.5, 1e-7));
        assert!(close(hlg_oetf(1.0), 1.0, 1e-7));
    }

    #[test]
    fn primaries_match_published_matrices() {
        // sRGB → XYZ (IEC 61966-2-1, to 4 decimals).
        let m = rgb_to_xyz(Gamut::Rec709);
        let want = [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]];
        for i in 0..3 {
            for j in 0..3 {
                assert!(close(m[i][j], want[i][j], 1e-4), "{i}{j}: {}", m[i][j]);
            }
        }
        // Rec.709 → Rec.2020 (BT.2087): first row 0.6274 0.3293 0.0433.
        let m = gamut_matrix(Gamut::Rec709, Gamut::Rec2020);
        for (v, w) in m[0].iter().zip([0.6274, 0.3293, 0.0433]) {
            assert!(close(*v, w, 1e-4), "{v} vs {w}");
        }
        // White maps to white in every direction, through the D60 adaptation too.
        for (a, b) in [(Gamut::Rec709, Gamut::Ap1), (Gamut::Ap1, Gamut::P3D65), (Gamut::P3D65, Gamut::Rec2020)] {
            let w = apply(&gamut_matrix(a, b), [1.0, 1.0, 1.0]);
            assert!(w.iter().all(|v| close(*v, 1.0, 1e-9)), "{a:?}→{b:?}: {w:?}");
        }
        // sRGB → ACEScg (the widely published matrix, first row 0.6131 0.3395 0.0474).
        let m = gamut_matrix(Gamut::Rec709, Gamut::Ap1);
        for (v, w) in m[0].iter().zip([0.6131, 0.3395, 0.0474]) {
            assert!(close(*v, w, 1e-3), "{v} vs {w}");
        }
    }

    #[test]
    fn conversions_round_trip_premultiplied() {
        let tags = ColorTag::ALL.into_iter().filter(|t| t.is_color());
        for a in tags.clone() {
            for b in tags.clone() {
                let (ab, ba) = (conversion(a, b).unwrap(), conversion(b, a).unwrap());
                let px = [0.3, 0.12, 0.05, 0.6];
                let back = convert_premultiplied(&ba, convert_premultiplied(&ab, px));
                for i in 0..4 {
                    assert!(close(back[i], px[i], 1e-9), "{a:?}↔{b:?}: {back:?}");
                }
            }
        }
        assert!(conversion(ColorTag::Data, ColorTag::Srgb).is_none());
        let c = conversion(ColorTag::Srgb, ColorTag::LinearSrgb).unwrap();
        assert_eq!(convert_premultiplied(&c, [0.2, 0.2, 0.2, 0.0]), [0.0, 0.0, 0.0, 0.0]);
        // Premultiplied sRGB 50% grey at alpha 0.5 → linear 0.214 × 0.5.
        let out = convert_premultiplied(&c, [0.25, 0.25, 0.25, 0.5]);
        assert!(close(out[0], 0.214_041 * 0.5, 1e-6));
    }

    #[test]
    fn ycbcr_matrices_and_ranges() {
        for m in [YcbcrMatrix::Bt709, YcbcrMatrix::Bt601] {
            for r in [YcbcrRange::Limited, YcbcrRange::Full] {
                for rgb in [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.8, 0.2, 0.4], [0.1, 0.9, 0.3]] {
                    let back = ycbcr_to_rgb_arr(rgb_to_ycbcr(rgb, m, r), m, r);
                    for i in 0..3 {
                        assert!(close(back[i], rgb[i], 1e-12), "{m:?} {r:?} {rgb:?} → {back:?}");
                    }
                }
            }
        }
        // Limited-range black and white are codes 16 and 235; BT.709 red has Y′ = 0.2126.
        let w = rgb_to_ycbcr([1.0; 3], YcbcrMatrix::Bt709, YcbcrRange::Limited);
        assert!(close(w[0] * 255.0, 235.0, 1e-9) && close(w[1] * 255.0, 128.0, 1e-9));
        let b = rgb_to_ycbcr([0.0; 3], YcbcrMatrix::Bt601, YcbcrRange::Limited);
        assert!(close(b[0] * 255.0, 16.0, 1e-9));
        let red = rgb_to_ycbcr([1.0, 0.0, 0.0], YcbcrMatrix::Bt709, YcbcrRange::Full);
        assert!(close(red[0], 0.2126, 1e-12));
    }

    fn ycbcr_to_rgb_arr(v: [f64; 3], m: YcbcrMatrix, r: YcbcrRange) -> [f64; 3] {
        ycbcr_to_rgb(v[0], v[1], v[2], m, r)
    }
}
