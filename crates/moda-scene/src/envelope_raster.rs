//! Expand of a bent image (E8.17, Illustrator's Object ▸ Envelope Distort ▸ Expand on an enveloped image): the
//! image's cells (the corner-pinned patches [`crate::envelope`] draws it with) resampled into ONE raster over the
//! bent art's box, so Expand gives one image layer, as Illustrator gives one embedded image, instead of thousands of
//! cell layers.
//!
//! Each output pixel is supersampled (`samples` × `samples`); a sample takes the topmost cell whose quad holds it
//! (the cell's projective map inverted exactly), and reads the image there with a bilinear filter, through the
//! cell's fill placement (`R(p − offset) / scale` natural px, as `path-pattern.wgsl`; rotation is not handled, the
//! caller keeps the cells then). Outside every cell the raster is transparent.
//!
//! Request: `{"cells": [{"quad": [x0, y0, … x3, y3] (TL TR BR BL, envelope units), "size": [w, h] (the cell's
//! source box), "offset": [x, y], "scale": [x, y], "repeat"?: "no-repeat" | "repeat" | "repeat-x" | "repeat-y"}],
//! "box": [x, y, w, h] (envelope units the raster covers), "pixels": [width, height], "samples"?: 1..4}`.

use crate::value::Value;

use crate::envelope::{apply_projective, unit_square_to_quad};
use crate::path_geom::vec2::{v2, V2};

/// Most pixels one bent-image raster may have (the caller lowers the density past it).
pub const MAX_RASTER_PIXELS: u64 = 40_000_000;
/// Longest side of one bent-image raster.
pub const MAX_RASTER_SIDE: u32 = 16_384;

/// One cell: its quad and how its local box reads the image.
#[derive(Debug, Clone)]
pub struct RasterCell {
    pub quad: [V2; 4],
    pub size: (f64, f64),
    pub offset: V2,
    pub scale: V2,
    /// Wrap across x / y (a tiled fill).
    pub wrap: (bool, bool),
}

/// A straight-alpha RGBA8 image.
#[derive(Debug, Clone)]
pub struct Rgba8 {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

fn invert3(m: &[f64; 9]) -> Option<[f64; 9]> {
    let [a, b, c, d, e, f, g, h, i] = *m;
    let co = [e * i - f * h, -(d * i - f * g), d * h - e * g];
    let det = a * co[0] + b * co[1] + c * co[2];
    if !det.is_finite() || det.abs() < 1e-18 {
        return None;
    }
    let inv = [
        co[0],
        -(b * i - c * h),
        b * f - c * e,
        co[1],
        a * i - c * g,
        -(a * f - c * d),
        co[2],
        -(a * h - b * g),
        a * e - b * d,
    ];
    let out = inv.map(|v| v / det);
    out.iter().all(|v| v.is_finite()).then_some(out)
}

struct Prepared {
    inv: [f64; 9],
    cell: RasterCell,
}

/// The image read at natural px `(x, y)` (bilinear, premultiplied), or `None` outside a no-repeat image.
fn sample(img: &Rgba8, x: f64, y: f64, wrap: (bool, bool)) -> Option<[f32; 4]> {
    let (w, h) = (img.width as f64, img.height as f64);
    let fold = |v: f64, n: f64, wraps: bool| -> Option<f64> {
        if wraps {
            Some(v.rem_euclid(n))
        } else if (0.0..=n).contains(&v) {
            Some(v)
        } else {
            None
        }
    };
    let x = fold(x, w, wrap.0)?;
    let y = fold(y, h, wrap.1)?;
    // Texel centres at +0.5; clamp to the edge texels (or wrap).
    let fx = x - 0.5;
    let fy = y - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = (fx - x0) as f32;
    let ty = (fy - y0) as f32;
    let idx = |v: f64, n: u32, wraps: bool| -> usize {
        let n = n as i64;
        let v = v as i64;
        (if wraps { v.rem_euclid(n) } else { v.clamp(0, n - 1) }) as usize
    };
    let xs = [idx(x0, img.width, wrap.0), idx(x0 + 1.0, img.width, wrap.0)];
    let ys = [idx(y0, img.height, wrap.1), idx(y0 + 1.0, img.height, wrap.1)];
    let mut acc = [0f32; 4];
    for (j, yy) in ys.iter().enumerate() {
        let wy = if j == 0 { 1.0 - ty } else { ty };
        for (i, xx) in xs.iter().enumerate() {
            let wx = if i == 0 { 1.0 - tx } else { tx };
            let k = (yy * img.width as usize + xx) * 4;
            let a = img.data[k + 3] as f32 / 255.0;
            let wgt = wx * wy;
            acc[0] += img.data[k] as f32 * a * wgt;
            acc[1] += img.data[k + 1] as f32 * a * wgt;
            acc[2] += img.data[k + 2] as f32 * a * wgt;
            acc[3] += a * wgt;
        }
    }
    Some(acc)
}

/// Resample `img` through `cells` into a `pixels` raster over `box_` (see the module docs).
pub fn warp_image(img: &Rgba8, cells: &[RasterCell], box_: [f64; 4], pixels: (u32, u32), samples: u32) -> Result<Rgba8, String> {
    let (ow, oh) = pixels;
    if ow == 0 || oh == 0 || ow > MAX_RASTER_SIDE || oh > MAX_RASTER_SIDE || ow as u64 * oh as u64 > MAX_RASTER_PIXELS {
        return Err(format!("the raster size {ow} × {oh} is out of range"));
    }
    if img.width == 0 || img.height == 0 || img.data.len() < img.width as usize * img.height as usize * 4 {
        return Err("the image has no pixels".into());
    }
    let [bx, by, bw, bh] = box_;
    if !(bw > 0.0 && bh > 0.0 && bx.is_finite() && by.is_finite()) {
        return Err("the raster box is empty".into());
    }
    let n = samples.clamp(1, 4);
    let (kx, ky) = (bw / ow as f64, bh / oh as f64);
    let prepared: Vec<Prepared> = cells
        .iter()
        .filter(|c| c.size.0 > 0.0 && c.size.1 > 0.0 && c.scale.x != 0.0 && c.scale.y != 0.0)
        .filter_map(|c| {
            // Pixel space: the quad in output px, so the inverse gives (u, v) straight from a sample position.
            let q = c.quad.map(|p| v2((p.x - bx) / kx, (p.y - by) / ky));
            let m = unit_square_to_quad(q)?;
            Some(Prepared { inv: invert3(&m)?, cell: c.clone() })
        })
        .collect();
    // A bin grid over the output: each bin lists the cells whose quad's box touches it.
    const BIN: u32 = 32;
    let (gw, gh) = (ow.div_ceil(BIN) as usize, oh.div_ceil(BIN) as usize);
    let mut bins: Vec<Vec<u32>> = vec![Vec::new(); gw * gh];
    for (k, p) in prepared.iter().enumerate() {
        let q = p.cell.quad.map(|pt| v2((pt.x - bx) / kx, (pt.y - by) / ky));
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for pt in q {
            x0 = x0.min(pt.x);
            y0 = y0.min(pt.y);
            x1 = x1.max(pt.x);
            y1 = y1.max(pt.y);
        }
        if x1 < 0.0 || y1 < 0.0 || x0 > ow as f64 || y0 > oh as f64 {
            continue;
        }
        let bin = |v: f64, max: usize| ((v.max(0.0) / BIN as f64) as usize).min(max - 1);
        for j in bin(y0, gh)..=bin(y1, gh) {
            for i in bin(x0, gw)..=bin(x1, gw) {
                bins[j * gw + i].push(k as u32);
            }
        }
    }
    let mut out = vec![0u8; ow as usize * oh as usize * 4];
    let inv_n = 1.0 / n as f64;
    let weight = 1.0 / (n * n) as f32;
    const EPS: f64 = 1e-9;
    for py in 0..oh {
        for px in 0..ow {
            let list = &bins[(py / BIN) as usize * gw + (px / BIN) as usize];
            if list.is_empty() {
                continue;
            }
            let mut acc = [0f32; 4];
            for sy in 0..n {
                for sx in 0..n {
                    let s = v2(px as f64 + (sx as f64 + 0.5) * inv_n, py as f64 + (sy as f64 + 0.5) * inv_n);
                    // The topmost cell holding the sample (cells draw in order).
                    for &k in list.iter().rev() {
                        let p = &prepared[k as usize];
                        let uv = apply_projective(&p.inv, s.x, s.y);
                        if !(uv.x >= -EPS && uv.x <= 1.0 + EPS && uv.y >= -EPS && uv.y <= 1.0 + EPS) {
                            continue;
                        }
                        let c = &p.cell;
                        let local = v2(uv.x.clamp(0.0, 1.0) * c.size.0, uv.y.clamp(0.0, 1.0) * c.size.1);
                        let ix = (local.x - c.offset.x) / c.scale.x;
                        let iy = (local.y - c.offset.y) / c.scale.y;
                        if let Some(t) = sample(img, ix, iy, c.wrap) {
                            for ch in 0..4 {
                                acc[ch] += t[ch] * weight;
                            }
                        }
                        break;
                    }
                }
            }
            let a = acc[3];
            if a <= 0.0 {
                continue;
            }
            let k = (py as usize * ow as usize + px as usize) * 4;
            out[k] = (acc[0] / a).round().clamp(0.0, 255.0) as u8;
            out[k + 1] = (acc[1] / a).round().clamp(0.0, 255.0) as u8;
            out[k + 2] = (acc[2] / a).round().clamp(0.0, 255.0) as u8;
            out[k + 3] = (a * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(Rgba8 { width: ow, height: oh, data: out })
}

fn pair(v: Option<&Value>) -> Option<V2> {
    let a = v?.as_array()?;
    let (x, y) = (a.first()?.as_f64()?, a.get(1)?.as_f64()?);
    (x.is_finite() && y.is_finite()).then(|| v2(x, y))
}

/// Parse a request (see the module docs) and warp `img` with it.
pub fn warp_image_request(img: &Rgba8, request_json: &str) -> Result<Rgba8, String> {
    let value: Value = crate::value::from_str(request_json).map_err(|e| format!("request is not valid JSON: {e}"))?;
    let nums = |k: &str, len: usize| -> Option<Vec<f64>> {
        let a = value.get(k)?.as_array()?;
        let v: Vec<f64> = a.iter().filter_map(Value::as_f64).filter(|x| x.is_finite()).collect();
        (v.len() == len).then_some(v)
    };
    let b = nums("box", 4).ok_or("request.box must be [x, y, w, h]")?;
    let px = nums("pixels", 2).ok_or("request.pixels must be [width, height]")?;
    let samples = value.get("samples").and_then(Value::as_u64).unwrap_or(2) as u32;
    let list = value.get("cells").and_then(Value::as_array).ok_or("request.cells must be an array")?;
    let mut cells = Vec::with_capacity(list.len());
    for c in list {
        let q: Vec<f64> = c.get("quad").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
        if q.len() != 8 || q.iter().any(|v| !v.is_finite()) {
            return Err("each cell needs a quad of 8 numbers".into());
        }
        let (Some(size), Some(offset), Some(scale)) = (pair(c.get("size")), pair(c.get("offset")), pair(c.get("scale"))) else {
            return Err("each cell needs size, offset and scale".into());
        };
        let wrap = match c.get("repeat").and_then(Value::as_str) {
            Some("repeat") => (true, true),
            Some("repeat-x") => (true, false),
            Some("repeat-y") => (false, true),
            _ => (false, false),
        };
        cells.push(RasterCell {
            quad: [v2(q[0], q[1]), v2(q[2], q[3]), v2(q[4], q[5]), v2(q[6], q[7])],
            size: (size.x, size.y),
            offset,
            scale,
            wrap,
        });
    }
    warp_image(img, &cells, [b[0], b[1], b[2], b[3]], (px[0].round().max(0.0) as u32, px[1].round().max(0.0) as u32), samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4 × 2 image: red left half, blue right half, opaque.
    fn two_tone() -> Rgba8 {
        let mut data = Vec::new();
        for _y in 0..2 {
            for x in 0..4 {
                data.extend_from_slice(if x < 2 { &[255, 0, 0, 255] } else { &[0, 0, 255, 255] });
            }
        }
        Rgba8 { width: 4, height: 2, data }
    }

    fn px(img: &Rgba8, x: u32, y: u32) -> [u8; 4] {
        let k = ((y * img.width + x) * 4) as usize;
        [img.data[k], img.data[k + 1], img.data[k + 2], img.data[k + 3]]
    }

    fn cell(quad: [f64; 8], size: (f64, f64), offset: (f64, f64), scale: (f64, f64)) -> RasterCell {
        RasterCell {
            quad: [v2(quad[0], quad[1]), v2(quad[2], quad[3]), v2(quad[4], quad[5]), v2(quad[6], quad[7])],
            size,
            offset: v2(offset.0, offset.1),
            scale: v2(scale.0, scale.1),
            wrap: (false, false),
        }
    }

    #[test]
    fn identity_cells_give_the_image_back() {
        // Two cells over a 40 × 20 box (the image at 10 units per px), split down the middle.
        let img = two_tone();
        let cells = [
            cell([0.0, 0.0, 20.0, 0.0, 20.0, 20.0, 0.0, 20.0], (20.0, 20.0), (0.0, 0.0), (10.0, 10.0)),
            cell([20.0, 0.0, 40.0, 0.0, 40.0, 20.0, 20.0, 20.0], (20.0, 20.0), (-20.0, 0.0), (10.0, 10.0)),
        ];
        let out = warp_image(&img, &cells, [0.0, 0.0, 40.0, 20.0], (40, 20), 2).unwrap();
        assert_eq!(px(&out, 2, 10), [255, 0, 0, 255]);
        assert_eq!(px(&out, 37, 10), [0, 0, 255, 255]);
        // Every pixel is covered: no seam between the cells.
        assert!(out.data.chunks(4).all(|p| p[3] == 255));
    }

    #[test]
    fn a_bent_cell_moves_the_pixels_and_leaves_the_rest_clear() {
        // One cell over the whole image, pinned onto a trapezoid narrower at the top.
        let img = two_tone();
        let cells = [cell([10.0, 0.0, 30.0, 0.0, 40.0, 20.0, 0.0, 20.0], (40.0, 20.0), (0.0, 0.0), (10.0, 10.0))];
        let out = warp_image(&img, &cells, [0.0, 0.0, 40.0, 20.0], (40, 20), 2).unwrap();
        // Outside the trapezoid (top corners): transparent.
        assert_eq!(px(&out, 1, 1)[3], 0);
        assert_eq!(px(&out, 38, 1)[3], 0);
        // Top row: the red/blue edge is at x 20 (the middle), the colours squeezed into 10..30.
        assert_eq!(px(&out, 12, 1), [255, 0, 0, 255]);
        assert_eq!(px(&out, 27, 1), [0, 0, 255, 255]);
        // Bottom row spans the whole box.
        assert_eq!(px(&out, 1, 19), [255, 0, 0, 255]);
        assert_eq!(px(&out, 38, 19), [0, 0, 255, 255]);
    }

    #[test]
    fn the_request_parses_and_bad_sizes_are_refused() {
        let img = two_tone();
        let req = r#"{"cells":[{"quad":[0,0,4,0,4,2,0,2],"size":[4,2],"offset":[0,0],"scale":[1,1]}],"box":[0,0,4,2],"pixels":[8,4]}"#;
        let out = warp_image_request(&img, req).unwrap();
        assert_eq!((out.width, out.height), (8, 4));
        assert_eq!(px(&out, 1, 1), [255, 0, 0, 255]);
        assert_eq!(px(&out, 7, 3), [0, 0, 255, 255]);
        let big = r#"{"cells":[],"box":[0,0,4,2],"pixels":[100000,4]}"#;
        assert!(warp_image_request(&img, big).is_err());
        assert!(warp_image_request(&img, "{").is_err());
    }
}
