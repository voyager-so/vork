//! Test-only nonzero scanline rasteriser: turns outlines into coverage masks
//! for the property tests and the visual gallery (no GPU, no dependencies).

use crate::blend::BlendPathPoint;
use crate::path_geom::vec2::{v2, Cubic, V2};

/// Flatten path points (move/line/curve) into closed polygons.
pub fn polygons(points: &[BlendPathPoint]) -> Vec<Vec<V2>> {
    let mut out: Vec<Vec<V2>> = Vec::new();
    let mut cur = V2::default();
    for p in points {
        match p {
            BlendPathPoint::Move { x, y } => {
                cur = v2(*x, *y);
                out.push(vec![cur]);
            }
            BlendPathPoint::Line { x, y } => {
                cur = v2(*x, *y);
                if let Some(poly) = out.last_mut() {
                    poly.push(cur);
                }
            }
            BlendPathPoint::Curve { x, y, cp1, cp2 } => {
                let c = Cubic([cur, v2(cp1.0, cp1.1), v2(cp2.0, cp2.1), v2(*x, *y)]);
                if let Some(poly) = out.last_mut() {
                    for i in 1..=16 {
                        poly.push(c.eval(i as f64 / 16.0));
                    }
                }
                cur = v2(*x, *y);
            }
            _ => panic!("outline emits only move/line/curve"),
        }
    }
    out
}

/// A coverage grid over `[x0, x0 + w·px) × [y0, y0 + h·px)`, sampled at pixel
/// centres.
#[derive(Clone)]
pub struct Grid {
    pub x0: f64,
    pub y0: f64,
    pub px: f64,
    pub w: usize,
    pub h: usize,
}

impl Grid {
    pub fn around(minx: f64, miny: f64, maxx: f64, maxy: f64, px: f64, margin: f64) -> Grid {
        let x0 = minx - margin;
        let y0 = miny - margin;
        Grid {
            x0,
            y0,
            px,
            w: (((maxx + margin - x0) / px).ceil() as usize).max(1),
            h: (((maxy + margin - y0) / px).ceil() as usize).max(1),
        }
    }

    /// Nonzero winding coverage of `polys`, each implicitly closed.
    pub fn fill_nonzero(&self, polys: &[Vec<V2>]) -> Vec<bool> {
        let mut out = vec![false; self.w * self.h];
        let mut xs: Vec<(f64, i32)> = Vec::new();
        for row in 0..self.h {
            let y = self.y0 + (row as f64 + 0.5) * self.px;
            xs.clear();
            for poly in polys {
                let n = poly.len();
                for i in 0..n {
                    let (a, b) = (poly[i], poly[(i + 1) % n]);
                    if (a.y <= y) != (b.y <= y) {
                        let x = a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x);
                        xs.push((x, if b.y > a.y { 1 } else { -1 }));
                    }
                }
            }
            xs.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut wind = 0;
            for k in 0..xs.len() {
                wind += xs[k].1;
                if wind != 0 && k + 1 < xs.len() {
                    let (xa, xb) = (xs[k].0, xs[k + 1].0);
                    let c0 = (((xa - self.x0) / self.px) - 0.5).ceil().max(0.0) as usize;
                    let c1 = (((xb - self.x0) / self.px) - 0.5).floor();
                    if c1 < 0.0 {
                        continue;
                    }
                    let c1 = (c1 as usize).min(self.w - 1);
                    for c in c0..=c1 {
                        out[row * self.w + c] = true;
                    }
                }
            }
        }
        out
    }

    /// OR of each polygon filled on its own (|winding| of each piece), i.e. the
    /// union regardless of orientation. Each piece only touches the rows of
    /// its own bounding box (the references are thousands of small quads).
    pub fn fill_union(&self, polys: &[Vec<V2>]) -> Vec<bool> {
        let mut out = vec![false; self.w * self.h];
        let mut xs: Vec<(f64, i32)> = Vec::new();
        for poly in polys {
            let n = poly.len();
            if n < 3 {
                continue;
            }
            let (mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY);
            for p in poly {
                y0 = y0.min(p.y);
                y1 = y1.max(p.y);
            }
            let r0 = (((y0 - self.y0) / self.px) - 0.5).ceil().max(0.0) as usize;
            let r1 = (((y1 - self.y0) / self.px) - 0.5).floor();
            if r1 < 0.0 {
                continue;
            }
            let r1 = (r1 as usize).min(self.h.saturating_sub(1));
            for row in r0..=r1 {
                let y = self.y0 + (row as f64 + 0.5) * self.px;
                xs.clear();
                for i in 0..n {
                    let (a, b) = (poly[i], poly[(i + 1) % n]);
                    if (a.y <= y) != (b.y <= y) {
                        let x = a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x);
                        xs.push((x, if b.y > a.y { 1 } else { -1 }));
                    }
                }
                xs.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
                let mut wind = 0;
                for k in 0..xs.len() {
                    wind += xs[k].1;
                    if wind != 0 && k + 1 < xs.len() {
                        let c0 = (((xs[k].0 - self.x0) / self.px) - 0.5).ceil().max(0.0) as usize;
                        let c1 = (((xs[k + 1].0 - self.x0) / self.px) - 0.5).floor();
                        if c1 < 0.0 {
                            continue;
                        }
                        let c1 = (c1 as usize).min(self.w - 1);
                        for c in c0..=c1 {
                            out[row * self.w + c] = true;
                        }
                    }
                }
            }
        }
        out
    }

    /// Cells set in `mask` whose 8-neighbourhood at distance `r` cells is set too.
    pub fn erode(&self, mask: &[bool], r: usize) -> Vec<bool> {
        let mut out = vec![false; mask.len()];
        let r = r as isize;
        for y in 0..self.h as isize {
            for x in 0..self.w as isize {
                let mut all = true;
                'n: for dy in [-r, 0, r] {
                    for dx in [-r, 0, r] {
                        let (xx, yy) = (x + dx, y + dy);
                        if xx < 0
                            || yy < 0
                            || xx >= self.w as isize
                            || yy >= self.h as isize
                            || !mask[(yy as usize) * self.w + xx as usize]
                        {
                            all = false;
                            break 'n;
                        }
                    }
                }
                out[(y as usize) * self.w + x as usize] = all;
            }
        }
        out
    }

    pub fn dilate(&self, mask: &[bool], r: usize) -> Vec<bool> {
        let inv: Vec<bool> = mask.iter().map(|b| !b).collect();
        self.erode(&inv, r).iter().map(|b| !b).collect()
    }

    /// Write an 8-bit grayscale PNG (ink = dark) with an optional overlay mask
    /// drawn mid-grey (e.g. the centerline).
    pub fn write_png(&self, path: &std::path::Path, mask: &[bool], overlay: Option<&[bool]>) {
        let mut data = vec![255u8; self.w * self.h];
        for i in 0..data.len() {
            if mask[i] {
                data[i] = 30;
            }
            if overlay.is_some_and(|o| o[i]) {
                data[i] = if mask[i] { 170 } else { 110 };
            }
        }
        let file = std::fs::File::create(path).expect("create png");
        let mut enc =
            png::Encoder::new(std::io::BufWriter::new(file), self.w as u32, self.h as u32);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().expect("png header");
        writer.write_image_data(&data).expect("png data");
    }
}

/// Anti-aliased coverage (0..1) at `ss × ss` supersampling.
pub fn coverage_aa(
    polys: &[Vec<V2>],
    x0: f64,
    y0: f64,
    px: f64,
    w: usize,
    h: usize,
    ss: usize,
) -> Vec<f32> {
    let g = Grid {
        x0,
        y0,
        px: px / ss as f64,
        w: w * ss,
        h: h * ss,
    };
    let m = g.fill_nonzero(polys);
    let mut out = vec![0f32; w * h];
    for y in 0..h * ss {
        for x in 0..w * ss {
            if m[y * w * ss + x] {
                out[(y / ss) * w + x / ss] += 1.0;
            }
        }
    }
    let k = (ss * ss) as f32;
    for v in out.iter_mut() {
        *v /= k;
    }
    out
}

/// Write an RGB PNG from layers of (coverage, colour), painted in order over white.
pub fn write_rgb_png(path: &std::path::Path, w: usize, h: usize, layers: &[(&[f32], [u8; 3])]) {
    let mut data = vec![255u8; w * h * 3];
    for (cov, col) in layers {
        for i in 0..w * h {
            let a = cov[i].clamp(0.0, 1.0);
            for c in 0..3 {
                let d = &mut data[i * 3 + c];
                *d = (*d as f32 * (1.0 - a) + col[c] as f32 * a).round() as u8;
            }
        }
    }
    let file = std::fs::File::create(path).expect("create png");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::High);
    let mut writer = enc.write_header().expect("png header");
    writer.write_image_data(&data).expect("png data");
}
