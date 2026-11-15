//! Image-based lighting for solids: an environment around the scene, from a sky-to-ground gradient or an
//! equirectangular (latitude–longitude) image, that every solid reflects and is lit by.
//!
//! The environment is an equirectangular image in linear light (x: longitude, the image's centre straight ahead
//! (+z, away from the camera) and its left / right edges behind; y: latitude, its top straight up (−y)). Reflections
//! read it along the reflected ray from a chain of box-filtered levels, a rougher surface a blurrier level (level =
//! roughness × [`ROUGH_LEVELS`]); diffuse light is its irradiance, from nine spherical-harmonic coefficients
//! (Ramamoorthi and Hanrahan, "An Efficient Representation for Irradiance Environment Maps", 2001). Everything is
//! computed on the CPU in a fixed order, so it is the same on every machine.

use std::f64::consts::PI;

/// The environment's base level: this wide, half as high.
pub const WIDTH: u32 = 512;
/// How many levels the roughest surface reads down (of the chain's 10).
pub const ROUGH_LEVELS: f32 = 6.0;

/// An environment, as the solid pass takes it.
#[derive(Clone, Debug, PartialEq)]
pub struct Environment {
    /// Levels from `WIDTH × WIDTH / 2` down to 1 × 1 (well, 2 × 1), RGBA linear, alpha 1.
    pub levels: Vec<(u32, u32, Vec<[f32; 4]>)>,
    /// Irradiance: nine SH coefficients per channel, pre-multiplied by the cosine lobe's convolution (so the shader's
    /// sum is the irradiance over π: the light a white Lambert surface reflects).
    pub sh: [[f32; 3]; 9],
}

/// The unit direction of an equirectangular pixel centre `(u, v)` in 0–1.
pub fn direction(u: f64, v: f64) -> [f64; 3] {
    let lon = (u - 0.5) * 2.0 * PI;
    let lat = (0.5 - v) * PI;
    let (sl, cl) = lat.sin_cos();
    let (so, co) = lon.sin_cos();
    // Up is −y.
    [cl * so, -sl, cl * co]
}

/// A gradient environment: `sky` straight up, `horizon` at the horizon, `ground` straight down (linear RGB), blended by
/// the sine of the elevation (smooth through the horizon).
pub fn gradient(sky: [f64; 3], horizon: [f64; 3], ground: [f64; 3]) -> Vec<[f32; 4]> {
    let (w, h) = (WIDTH, WIDTH / 2);
    let mut px = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        let v = (f64::from(y) + 0.5) / f64::from(h);
        let up = -direction(0.5, v)[1];
        let (a, b, t) = if up >= 0.0 { (horizon, sky, up.sqrt()) } else { (horizon, ground, (-up).sqrt()) };
        let c = [0, 1, 2].map(|k| (a[k] + (b[k] - a[k]) * t) as f32);
        for _ in 0..w {
            px.push([c[0], c[1], c[2], 1.0]);
        }
    }
    px
}

/// An image (straight sRGB RGBA8, any size) resampled to the base level in linear light (bilinear, longitude
/// wrapping).
pub fn from_image(width: u32, height: u32, rgba: &[u8]) -> Vec<[f32; 4]> {
    let lin = |c: u8| {
        let c = f64::from(c) / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let (iw, ih) = (width.max(1) as usize, height.max(1) as usize);
    let at = |x: isize, y: isize| -> [f64; 3] {
        let x = x.rem_euclid(iw as isize) as usize;
        let y = y.clamp(0, ih as isize - 1) as usize;
        let i = (y * iw + x) * 4;
        if i + 2 < rgba.len() { [lin(rgba[i]), lin(rgba[i + 1]), lin(rgba[i + 2])] } else { [0.0; 3] }
    };
    let (w, h) = (WIDTH, WIDTH / 2);
    let mut px = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let sx = (f64::from(x) + 0.5) / f64::from(w) * iw as f64 - 0.5;
            let sy = (f64::from(y) + 0.5) / f64::from(h) * ih as f64 - 0.5;
            let (x0, y0) = (sx.floor(), sy.floor());
            let (fx, fy) = (sx - x0, sy - y0);
            let (x0, y0) = (x0 as isize, y0 as isize);
            let mut c = [0.0; 3];
            for (dx, dy, wgt) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
                let p = at(x0 + dx, y0 + dy);
                for k in 0..3 {
                    c[k] += p[k] * wgt;
                }
            }
            px.push([c[0] as f32, c[1] as f32, c[2] as f32, 1.0]);
        }
    }
    px
}

impl Environment {
    /// The environment of a base level (`WIDTH × WIDTH / 2`, linear).
    pub fn new(base: Vec<[f32; 4]>) -> Environment {
        let mut levels = vec![(WIDTH, WIDTH / 2, base)];
        while levels.last().is_some_and(|l| l.0 > 2) {
            let (w, h, px) = levels.last().expect("non-empty");
            let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
            let mut next = Vec::with_capacity((nw * nh) as usize);
            for y in 0..nh {
                for x in 0..nw {
                    let mut c = [0.0f32; 4];
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (sx, sy) = ((2 * x + dx).min(w - 1), (2 * y + dy).min(h - 1));
                        let p = px[(sy * w + sx) as usize];
                        for k in 0..4 {
                            c[k] += p[k] / 4.0;
                        }
                    }
                    next.push(c);
                }
            }
            levels.push((nw, nh, next));
        }
        let sh = irradiance_sh(&levels);
        Environment { levels, sh }
    }
}

/// The nine SH coefficients of the irradiance (over π), from the 64 × 32 level, each pixel weighted by its solid angle.
fn irradiance_sh(levels: &[(u32, u32, Vec<[f32; 4]>)]) -> [[f32; 3]; 9] {
    let (w, h, px) = levels.iter().find(|l| l.0 <= 64).unwrap_or(&levels[0]);
    let mut acc = [[0.0f64; 3]; 9];
    for y in 0..*h {
        let v = (f64::from(y) + 0.5) / f64::from(*h);
        // Solid angle of a pixel: (2π / w) × (π / h) × cos(latitude).
        let d_omega = (2.0 * PI / f64::from(*w)) * (PI / f64::from(*h)) * ((0.5 - v) * PI).cos();
        for x in 0..*w {
            let u = (f64::from(x) + 0.5) / f64::from(*w);
            let d = direction(u, v);
            let basis = sh_basis(d);
            let c = px[(y * w + x) as usize];
            for (i, b) in basis.iter().enumerate() {
                for k in 0..3 {
                    acc[i][k] += f64::from(c[k]) * b * d_omega;
                }
            }
        }
    }
    // The clamped cosine's convolution per band (π, 2π/3, π/4), over π.
    let band = [1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25];
    let mut out = [[0.0f32; 3]; 9];
    for i in 0..9 {
        for k in 0..3 {
            out[i][k] = (acc[i][k] * band[i]) as f32;
        }
    }
    out
}

/// The real SH basis up to band 2 at unit direction `d`.
pub fn sh_basis([x, y, z]: [f64; 3]) -> [f64; 9] {
    [
        0.282_095,
        0.488_603 * y,
        0.488_603 * z,
        0.488_603 * x,
        1.092_548 * x * y,
        1.092_548 * y * z,
        0.315_392 * (3.0 * z * z - 1.0),
        1.092_548 * x * z,
        0.546_274 * (x * x - y * y),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_uniform_environment_lights_every_normal_by_its_value() {
        let env = Environment::new(gradient([0.5; 3], [0.5; 3], [0.5; 3]));
        for d in [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]] {
            let b = sh_basis(d);
            let e: f64 = (0..9).map(|i| f64::from(env.sh[i][0]) * b[i]).sum();
            assert!((e - 0.5).abs() < 0.01, "{d:?}: {e}");
        }
    }

    #[test]
    fn a_bright_sky_lights_an_upward_normal_more_than_a_downward_one() {
        let env = Environment::new(gradient([1.0; 3], [0.2; 3], [0.0; 3]));
        let e = |d: [f64; 3]| -> f64 { (0..9).map(|i| f64::from(env.sh[i][1]) * sh_basis(d)[i]).sum() };
        assert!(e([0.0, -1.0, 0.0]) > 0.6 && e([0.0, 1.0, 0.0]) < 0.15, "{} {}", e([0.0, -1.0, 0.0]), e([0.0, 1.0, 0.0]));
        assert_eq!(env.levels.last().unwrap().0, 2);
        assert_eq!(env.levels[0].2.len(), (WIDTH * WIDTH / 2) as usize);
    }
}
