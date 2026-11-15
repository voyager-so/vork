//! Test-only rasteriser for the brush tests and gallery: flattens placed
//! paths, fills them (nonzero or evenodd) with supersampling and paints RGB
//! layers over white. Strokes are drawn as unions of per-segment quads plus
//! round dots at the vertices.

use crate::blend::BlendPathPoint;
use crate::path_geom::vec2::{v2, Cubic, V2};

use super::colorize::parse_color;
use super::PlacedPath;

/// Flatten move/line/curve points into polygons (closed implicitly).
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
                    for i in 1..=12 {
                        poly.push(c.eval(i as f64 / 12.0));
                    }
                }
                cur = v2(*x, *y);
            }
            _ => panic!("placed paths emit only move/line/curve"),
        }
    }
    out
}

pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pub x0: f64,
    pub y0: f64,
    pub px: f64,
    pub rgb: Vec<[f32; 3]>,
}

const SS: usize = 3;

impl Canvas {
    pub fn new(w: usize, h: usize, x0: f64, y0: f64, px: f64) -> Canvas {
        Canvas {
            w,
            h,
            x0,
            y0,
            px,
            rgb: vec![[255.0; 3]; w * h],
        }
    }

    /// Coverage 0..1 per pixel of `polys` under a fill rule.
    pub fn coverage(&self, polys: &[Vec<V2>], evenodd: bool) -> Vec<f32> {
        let (w, h) = (self.w * SS, self.h * SS);
        let px = self.px / SS as f64;
        let mut cov = vec![0f32; self.w * self.h];
        let mut xs: Vec<(f64, i32)> = Vec::new();
        for row in 0..h {
            let y = self.y0 + (row as f64 + 0.5) * px;
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
            xs.sort_by(|p, q| p.0.total_cmp(&q.0));
            let mut wind = 0;
            for k in 0..xs.len() {
                wind += xs[k].1;
                let inside = if evenodd { wind % 2 != 0 } else { wind != 0 };
                if inside && k + 1 < xs.len() {
                    let c0 = (((xs[k].0 - self.x0) / px) - 0.5).ceil().max(0.0) as usize;
                    let c1 = (((xs[k + 1].0 - self.x0) / px) - 0.5).floor();
                    if c1 < 0.0 {
                        continue;
                    }
                    let c1 = (c1 as usize).min(w - 1);
                    for c in c0..=c1 {
                        cov[(row / SS) * self.w + c / SS] += 1.0;
                    }
                }
            }
        }
        let k = (SS * SS) as f32;
        cov.iter_mut().for_each(|v| *v = (*v / k).min(1.0));
        cov
    }

    pub fn paint(&mut self, cov: &[f32], rgb: [f32; 3], alpha: f32) {
        for (i, c) in cov.iter().enumerate() {
            let a = c * alpha;
            for (dst, src) in self.rgb[i].iter_mut().zip(rgb) {
                *dst = *dst * (1.0 - a) + src * a;
            }
        }
    }

    /// Stroke polylines as a union of quads and vertex dots.
    pub fn stroke_polys(&self, polys: &[Vec<V2>], width: f64, closed: bool) -> Vec<f32> {
        let r = width / 2.0;
        let mut pieces: Vec<Vec<V2>> = Vec::new();
        for poly in polys {
            let n = poly.len();
            let segs = if closed { n } else { n.saturating_sub(1) };
            for i in 0..segs {
                let (a, b) = (poly[i], poly[(i + 1) % n]);
                let Some(d) = (b - a).unit() else { continue };
                let nn = v2(-d.y, d.x) * r;
                pieces.push(vec![a + nn, b + nn, b - nn, a - nn]);
            }
            for p in poly {
                pieces.push(
                    (0..12)
                        .map(|k| {
                            let t = k as f64 / 12.0 * std::f64::consts::TAU;
                            *p + v2(t.cos(), t.sin()) * r
                        })
                        .collect(),
                );
            }
        }
        let mut cov = vec![0f32; self.w * self.h];
        for piece in pieces {
            let c = self.coverage(std::slice::from_ref(&piece), false);
            for (o, v) in cov.iter_mut().zip(c) {
                *o = o.max(v);
            }
        }
        cov
    }

    /// Draw placed paths with their fill / stroke paints.
    pub fn draw(&mut self, paths: &[PlacedPath]) {
        for p in paths {
            let polys = polygons(&p.points);
            let opacity = p
                .paint
                .get("opacity")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            let evenodd = p.paint.get("fillRule").and_then(|v| v.as_str()) == Some("evenodd");
            let fill = p.paint.get("fill").and_then(|v| v.as_str()).or_else(|| {
                p.paint
                    .get("fills")
                    .and_then(|f| f.get(0))
                    .and_then(|f| f.get("paint"))
                    .and_then(|f| f.get("color"))
                    .and_then(|v| v.as_str())
            });
            if let Some(c) = fill.and_then(parse_color) {
                let cov = self.coverage(&polys, evenodd);
                self.paint(
                    &cov,
                    [c.r as f32, c.g as f32, c.b as f32],
                    c.a as f32 * opacity,
                );
            }
            let sw = p
                .paint
                .get("strokeWidth")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            if let Some(c) = p
                .paint
                .get("stroke")
                .and_then(|v| v.as_str())
                .and_then(parse_color)
            {
                if sw > 0.0 {
                    let cov = self.stroke_polys(&polys, sw, p.closed);
                    self.paint(
                        &cov,
                        [c.r as f32, c.g as f32, c.b as f32],
                        c.a as f32 * opacity,
                    );
                }
            }
        }
    }

    /// A thin guide line (e.g. the stroke's centerline).
    pub fn hairline(&mut self, polys: &[Vec<V2>], closed: bool, rgb: [f32; 3]) {
        let cov = self.stroke_polys(polys, self.px * 0.9, closed);
        self.paint(&cov, rgb, 0.8);
    }

    pub fn write_png(&self, path: &std::path::Path) {
        let data: Vec<u8> = self
            .rgb
            .iter()
            .flat_map(|p| p.map(|v| v.round().clamp(0.0, 255.0) as u8))
            .collect();
        let file = std::fs::File::create(path).expect("create png");
        let mut enc =
            png::Encoder::new(std::io::BufWriter::new(file), self.w as u32, self.h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_compression(png::Compression::High);
        let mut writer = enc.write_header().expect("png header");
        writer.write_image_data(&data).expect("png data");
    }
}
