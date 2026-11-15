//! The `sprites` pixel node: particles and flow-field lines (moda-motion `particles`) rasterized on the CPU into a
//! premultiplied linear-light image, then uploaded.
//!
//! The particles arrive flattened (moda-motion `particles::to_floats`): `[n, square, 0, 0]`, then per item `[x, y,
//! vx, vy, size, rotation, r, g, b, a, trail length, first trail point]` in frame pixels (straight colour), then the
//! trail points. Each item is drawn in order, antialiased by its coverage of each pixel centre:
//! - `circle`: a disc of diameter `size`; `square`: a square of side `size`, turned by `rotation`;
//! - `soft`: a dot whose opacity falls as `(1 − (d / r)²)²` to its radius;
//! - `streak`: a round-capped line from where it was `streak` seconds ago (its position minus velocity × streak) to
//!   where it is, `size` wide;
//! - with a sprite image: the image's drawn part (cropped to its alpha) scaled so its larger side is `size`, turned,
//!   and multiplied by the colour;
//! - a flow-field line: its trail as a polyline `size` wide, opacity falling toward the tail by `taper` (0: even,
//!   1: to nothing at the tail).
//!
//! `over` composites each item over the ones before; `add` sums them (light).

use super::exec::{Args, Executor, Param};
use super::image::{ColorTag, Domain, Image, Storage};
use super::PixelError;

type R<T> = Result<T, PixelError>;

const STRIDE: usize = 12;

#[derive(Clone, Copy, PartialEq)]
enum Look {
    Circle,
    Square,
    Soft,
    Streak,
}

/// A CPU canvas over `domain` (output pixels), premultiplied RGBA.
struct Canvas {
    d: Domain,
    px: Vec<f32>,
    add: bool,
}

impl Canvas {
    fn put(&mut self, x: i32, y: i32, c: [f32; 4], cov: f32) {
        if cov <= 0.0 {
            return;
        }
        let (lx, ly) = (x - self.d.x, y - self.d.y);
        if lx < 0 || ly < 0 || lx >= self.d.w as i32 || ly >= self.d.h as i32 {
            return;
        }
        let i = (ly as usize * self.d.w as usize + lx as usize) * 4;
        let a = (c[3] * cov).min(1.0);
        let src = [c[0] * a, c[1] * a, c[2] * a, a];
        let p = &mut self.px[i..i + 4];
        if self.add {
            for k in 0..3 {
                p[k] += src[k];
            }
            p[3] = (p[3] + src[3]).min(1.0);
        } else {
            let k = 1.0 - src[3];
            for j in 0..4 {
                p[j] = src[j] + p[j] * k;
            }
        }
    }

    /// Every pixel whose centre is within `reach` of the box `[x0, y0]–[x1, y1]`, clipped to the canvas.
    fn span(&self, x0: f32, y0: f32, x1: f32, y1: f32) -> (i32, i32, i32, i32) {
        let (dx0, dy0) = (self.d.x, self.d.y);
        let (dx1, dy1) = (self.d.x + self.d.w as i32, self.d.y + self.d.h as i32);
        (
            (x0.floor() as i32).max(dx0),
            (y0.floor() as i32).max(dy0),
            (x1.ceil() as i32).min(dx1),
            (y1.ceil() as i32).min(dy1),
        )
    }
}

/// The distance from `p` to the segment `a`–`b`, and where along it (0–1) the nearest point is.
fn seg_dist(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> (f32, f32) {
    let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
    let (wx, wy) = (p[0] - a[0], p[1] - a[1]);
    let ee = ex * ex + ey * ey;
    let u = if ee > 0.0 { ((wx * ex + wy * ey) / ee).clamp(0.0, 1.0) } else { 0.0 };
    let (dx, dy) = (wx - ex * u, wy - ey * u);
    ((dx * dx + dy * dy).sqrt(), u)
}

/// A capsule `a`–`b` of radius `r`, its opacity running from `ca` at `a` to `cb` at `b` (colour alpha).
fn capsule(cv: &mut Canvas, a: [f32; 2], b: [f32; 2], r: f32, color: [f32; 4], alpha: (f32, f32)) {
    let r = r.max(0.35);
    let (x0, y0, x1, y1) = cv.span(a[0].min(b[0]) - r - 1.0, a[1].min(b[1]) - r - 1.0, a[0].max(b[0]) + r + 1.0, a[1].max(b[1]) + r + 1.0);
    for y in y0..y1 {
        for x in x0..x1 {
            let (d, u) = seg_dist([x as f32 + 0.5, y as f32 + 0.5], a, b);
            let cov = (r - d + 0.5).clamp(0.0, 1.0);
            if cov > 0.0 {
                let k = alpha.0 + (alpha.1 - alpha.0) * u;
                cv.put(x, y, [color[0], color[1], color[2], color[3] * k], cov);
            }
        }
    }
}

/// A sprite image on the CPU: straight colour, bilinear, transparent outside.
struct Sprite {
    w: usize,
    h: usize,
    px: Vec<f32>,
}

impl Sprite {
    fn at(&self, u: f32, v: f32) -> [f32; 4] {
        // u, v in pixels of the sprite; premultiplied samples.
        let (x, y) = (u - 0.5, v - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let get = |i: i64, j: i64| -> [f32; 4] {
            if i < 0 || j < 0 || i >= self.w as i64 || j >= self.h as i64 {
                return [0.0; 4];
            }
            let k = (j as usize * self.w + i as usize) * 4;
            [self.px[k], self.px[k + 1], self.px[k + 2], self.px[k + 3]]
        };
        let (i, j) = (x0 as i64, y0 as i64);
        let (a, b, c, d) = (get(i, j), get(i + 1, j), get(i, j + 1), get(i + 1, j + 1));
        let mut out = [0.0; 4];
        for k in 0..4 {
            let top = a[k] + (b[k] - a[k]) * fx;
            let bot = c[k] + (d[k] - c[k]) * fx;
            out[k] = top + (bot - top) * fy;
        }
        out
    }
}

/// Run `sprites` (see the module docs).
pub fn run<'i>(
    exec: &Executor,
    args: &Args<'_>,
    input: &dyn Fn(&str) -> Result<&'i Image, PixelError>,
    scale: f64,
) -> R<Image> {
    let node = args.node;
    let data = match args.params.get("particles") {
        Some(Param::Floats(f)) => f.clone(),
        Some(p) => return Err(PixelError::Invalid(format!("`{node}.particles` needs a particles or flowfield node, not {p:?}"))),
        None => return Err(PixelError::Invalid(format!("`{node}` needs `particles`"))),
    };
    let n = data.first().copied().unwrap_or(0.0).max(0.0) as usize;
    let square = data.get(1).copied().unwrap_or(0.0) > 0.5;
    if data.len() < 4 + n * STRIDE {
        return Err(PixelError::Invalid(format!("`{node}.particles`: a short buffer")));
    }
    let look = match args.string("shape", if square { "square" } else { "circle" })? {
        "circle" => Look::Circle,
        "square" => Look::Square,
        "soft" => Look::Soft,
        "streak" => Look::Streak,
        other => return Err(PixelError::Invalid(format!("`{node}.shape`: `{other}` is not circle, square, soft or streak"))),
    };
    let add = match args.string("blend", "over")? {
        "over" => false,
        "add" => true,
        other => return Err(PixelError::Invalid(format!("`{node}.blend`: `{other}` is not over or add"))),
    };
    let streak = args.num("streak", 0.05)? as f32;
    let taper = args.num("taper", 1.0)?.clamp(0.0, 1.0) as f32;
    let sprite = match input("sprite") {
        Ok(img) => {
            let px = exec.read_f32(img, img.domain, None).map_err(|e| PixelError::Invalid(format!("`{node}.sprite`: {e}")))?;
            // read_f32 without a background un-premultiplies: premultiply again for filtering. The sprite is the
            // image's drawn part (a layer's render covers the frame): cropped to where it has alpha.
            let (w, h) = (img.domain.w as usize, img.domain.h as usize);
            let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
            for y in 0..h {
                for x in 0..w {
                    if px[(y * w + x) * 4 + 3] > 1e-3 {
                        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
                    }
                }
            }
            if x1 <= x0 || y1 <= y0 {
                None
            } else {
                let mut out = Vec::with_capacity((x1 - x0) * (y1 - y0) * 4);
                for y in y0..y1 {
                    for x in x0..x1 {
                        let c = &px[(y * w + x) * 4..(y * w + x) * 4 + 4];
                        out.extend([c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]]);
                    }
                }
                Some(Sprite { w: x1 - x0, h: y1 - y0, px: out })
            }
        }
        Err(_) => None,
    };
    let s = scale as f32;
    let (fw, fh) = exec.frame_size();
    let item = |i: usize| &data[4 + i * STRIDE..4 + (i + 1) * STRIDE];
    let trail_base = 4 + n * STRIDE;
    // The canvas: the items' reach, within the frame (grown a little for edges).
    let (mut bx0, mut by0, mut bx1, mut by1) = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut grow = |x: f32, y: f32, r: f32| {
        bx0 = bx0.min(x - r);
        by0 = by0.min(y - r);
        bx1 = bx1.max(x + r);
        by1 = by1.max(y + r);
    };
    for i in 0..n {
        let it = item(i);
        let r = it[4] * s * 0.75 + 2.0;
        grow(it[0] * s, it[1] * s, r);
        if look == Look::Streak {
            grow((it[0] - it[2] * streak) * s, (it[1] - it[3] * streak) * s, r);
        }
        let (len, first) = (it[10] as usize, it[11] as usize);
        for k in 0..len {
            let q = trail_base + (first + k) * 2;
            if let (Some(x), Some(y)) = (data.get(q), data.get(q + 1)) {
                grow(x * s, y * s, r);
            }
        }
    }
    let frame = Domain::frame(fw, fh).grow(2, 2);
    let d = if bx0.is_finite() { Domain::covering(bx0 as f64, by0 as f64, bx1 as f64, by1 as f64).intersect(&frame) } else { Domain::EMPTY };
    if d.is_empty() {
        return exec.solid(Domain::new(0, 0, 1, 1), [0.0; 4], ColorTag::LinearSrgb);
    }
    let items = Items { data: &data, n, trail_base, look, sprite: sprite.as_ref(), s, streak, taper };
    let texels = raster_f16(d, add, &items);
    // f16 bits in native byte order: little-endian on every target wgpu runs on, as the texture wants.
    let bytes: &[u8] = bytemuck::cast_slice(half::slice::HalfFloatSliceExt::reinterpret_cast(&texels[..]));
    let tex = exec.passes.upload_bytes(&exec.pool, d.w, d.h, Storage::F16, bytes)?;
    Ok(Image::rgba(d, ColorTag::LinearSrgb, tex))
}

/// The items to draw, as [`run`] parsed them.
struct Items<'a> {
    data: &'a [f32],
    n: usize,
    trail_base: usize,
    look: Look,
    sprite: Option<&'a Sprite>,
    s: f32,
    streak: f32,
    taper: f32,
}

/// Canvases smaller than this many pixels are drawn on one thread.
const BAND_MIN_PIXELS: usize = 256 * 256;

/// Draw `items` over `d` and return it as f16 RGBA texels. Large canvases are drawn in row bands on several threads:
/// each band draws every item clipped to its rows, so each pixel takes the same composites in the same order as on
/// one canvas, and the result is the same bits.
fn raster_f16(d: Domain, add: bool, items: &Items<'_>) -> Vec<half::f16> {
    let (w, h) = (d.w as usize, d.h as usize);
    let threads = if w * h >= BAND_MIN_PIXELS { std::thread::available_parallelism().map_or(1, |n| n.get()).min(8) } else { 1 };
    raster_f16_bands(d, add, items, threads)
}

fn raster_f16_bands(d: Domain, add: bool, items: &Items<'_>, threads: usize) -> Vec<half::f16> {
    use half::slice::HalfFloatSliceExt;
    let (w, h) = (d.w as usize, d.h as usize);
    let mut out = vec![half::f16::ZERO; w * h * 4];
    let band = h.div_ceil(threads.max(1)).max(1);
    let draw_band = |b: usize, dst: &mut [half::f16]| {
        let rows = dst.len() / (w * 4);
        let bd = Domain::new(d.x, d.y + (b * band) as i32, d.w, rows as u32);
        let mut cv = Canvas { d: bd, px: vec![0.0; w * rows * 4], add };
        draw(&mut cv, items);
        dst.convert_from_f32_slice(&cv.px);
    };
    if threads <= 1 || h <= band {
        draw_band(0, &mut out);
    } else {
        std::thread::scope(|sc| {
            for (b, dst) in out.chunks_mut(band * w * 4).enumerate() {
                let draw_band = &draw_band;
                sc.spawn(move || draw_band(b, dst));
            }
        });
    }
    out
}

/// Draw every item onto `cv`, in order, clipped to its domain.
fn draw(cv: &mut Canvas, items: &Items<'_>) {
    let Items { data, n, trail_base, look, sprite, s, streak, taper } = *items;
    let item = |i: usize| &data[4 + i * STRIDE..4 + (i + 1) * STRIDE];
    for i in 0..n {
        let it = item(i);
        let (x, y, vx, vy) = (it[0] * s, it[1] * s, it[2] * s, it[3] * s);
        let size = it[4] * s;
        let color = [it[6], it[7], it[8], it[9]];
        if color[3] <= 0.0 {
            continue;
        }
        let (len, first) = (it[10] as usize, it[11] as usize);
        if len >= 2 {
            let pt = |k: usize| {
                let q = trail_base + (first + k) * 2;
                [data[q] * s, data[q + 1] * s]
            };
            // The whole line as one shape: each pixel takes its largest coverage over the segments (opacity 1 at
            // the head, 1 − taper at the tail), then composites once, so joints don't double up.
            let r = (size * 0.5).max(0.35);
            let pts: Vec<[f32; 2]> = (0..len).map(pt).collect();
            let (mut lx0, mut ly0, mut lx1, mut ly1) = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
            for q in &pts {
                (lx0, ly0, lx1, ly1) = (lx0.min(q[0]), ly0.min(q[1]), lx1.max(q[0]), ly1.max(q[1]));
            }
            let (x0, y0, x1, y1) = cv.span(lx0 - r - 1.0, ly0 - r - 1.0, lx1 + r + 1.0, ly1 + r + 1.0);
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            let bw = (x1 - x0) as usize;
            let mut best = vec![0.0f32; bw * (y1 - y0) as usize];
            for k in 0..len - 1 {
                let (a, b) = (pts[k], pts[k + 1]);
                let fa = 1.0 - taper * (1.0 - k as f32 / (len - 1) as f32);
                let fb = 1.0 - taper * (1.0 - (k + 1) as f32 / (len - 1) as f32);
                let (sx0, sy0, sx1, sy1) = cv.span(a[0].min(b[0]) - r - 1.0, a[1].min(b[1]) - r - 1.0, a[0].max(b[0]) + r + 1.0, a[1].max(b[1]) + r + 1.0);
                for y in sy0.max(y0)..sy1.min(y1) {
                    for x in sx0.max(x0)..sx1.min(x1) {
                        let (d, u) = seg_dist([x as f32 + 0.5, y as f32 + 0.5], a, b);
                        let cov = (r - d + 0.5).clamp(0.0, 1.0) * (fa + (fb - fa) * u);
                        let slot = &mut best[(y - y0) as usize * bw + (x - x0) as usize];
                        *slot = slot.max(cov);
                    }
                }
            }
            for y in y0..y1 {
                for x in x0..x1 {
                    cv.put(x, y, color, best[(y - y0) as usize * bw + (x - x0) as usize]);
                }
            }
            continue;
        }
        let r = size * 0.5;
        if r <= 0.0 {
            continue;
        }
        if let Some(sp) = sprite {
            let side = sp.w.max(sp.h) as f32;
            let k = side / size.max(1e-3);
            let (sn, cs) = it[5].to_radians().sin_cos();
            let reach = r * std::f32::consts::SQRT_2 + 1.0;
            let (x0, y0, x1, y1) = cv.span(x - reach, y - reach, x + reach, y + reach);
            for py in y0..y1 {
                for px in x0..x1 {
                    let (dx, dy) = (px as f32 + 0.5 - x, py as f32 + 0.5 - y);
                    // Into the sprite's frame: turned back, scaled, centred.
                    let (u, v) = ((dx * cs + dy * sn) * k + sp.w as f32 * 0.5, (-dx * sn + dy * cs) * k + sp.h as f32 * 0.5);
                    let c = sp.at(u, v);
                    if c[3] > 0.0 {
                        let rgb = [c[0] / c[3] * color[0], c[1] / c[3] * color[1], c[2] / c[3] * color[2]];
                        cv.put(px, py, [rgb[0], rgb[1], rgb[2], color[3]], c[3]);
                    }
                }
            }
            continue;
        }
        match look {
            Look::Streak => capsule(cv, [x - vx * streak, y - vy * streak], [x, y], r, color, (1.0, 1.0)),
            Look::Circle | Look::Soft => {
                let (x0, y0, x1, y1) = cv.span(x - r - 1.0, y - r - 1.0, x + r + 1.0, y + r + 1.0);
                for py in y0..y1 {
                    for px in x0..x1 {
                        let (dx, dy) = (px as f32 + 0.5 - x, py as f32 + 0.5 - y);
                        let dd = (dx * dx + dy * dy).sqrt();
                        let cov = if look == Look::Soft {
                            let q = 1.0 - (dd / r).powi(2);
                            if q > 0.0 { q * q } else { 0.0 }
                        } else {
                            (r - dd + 0.5).clamp(0.0, 1.0)
                        };
                        cv.put(px, py, color, cov);
                    }
                }
            }
            Look::Square => {
                let (sn, cs) = it[5].to_radians().sin_cos();
                let reach = r * std::f32::consts::SQRT_2 + 1.0;
                let (x0, y0, x1, y1) = cv.span(x - reach, y - reach, x + reach, y + reach);
                for py in y0..y1 {
                    for px in x0..x1 {
                        let (dx, dy) = (px as f32 + 0.5 - x, py as f32 + 0.5 - y);
                        let (u, v) = (dx * cs + dy * sn, -dx * sn + dy * cs);
                        let cov = (r - u.abs() + 0.5).clamp(0.0, 1.0) * (r - v.abs() + 0.5).clamp(0.0, 1.0);
                        cv.put(px, py, color, cov);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A particles buffer (`to_floats` layout) of `n` items scattered over 1920×1080, every fifth with a trail.
    fn particles(n: usize, size: f32) -> Vec<f32> {
        let mut rng = 0x2545_f491_u32;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng >> 8) as f32 / (1u32 << 24) as f32
        };
        let mut items = Vec::new();
        let mut trails = Vec::new();
        for i in 0..n {
            let (x, y) = (next() * 2000.0 - 40.0, next() * 1160.0 - 40.0);
            let len = if i % 5 == 0 { 2 + (i % 4) } else { 0 };
            let first = trails.len() / 2;
            for k in 0..len {
                trails.extend([x - 9.0 * k as f32, y + 4.0 * k as f32 * next()]);
            }
            items.extend([x, y, next() * 400.0 - 200.0, next() * 400.0, size * (0.5 + next()), next() * 360.0, next(), next(), next(), 0.2 + 0.8 * next(), len as f32, first as f32]);
        }
        let mut v = vec![n as f32, 0.0, 0.0, 0.0];
        v.extend(items);
        v.extend(trails);
        v
    }

    #[test]
    fn bands_draw_the_same_bits_as_one_canvas() {
        let d = Domain::new(-2, -2, 1924, 1084);
        let sprite = Sprite { w: 5, h: 3, px: (0..60).map(|i| ((i * 7) % 11) as f32 / 10.0 * if i % 4 == 3 { 1.0 } else { 0.5 }).collect() };
        for look in [Look::Square, Look::Soft, Look::Circle, Look::Streak] {
            for (add, with_sprite) in [(false, false), (true, false), (false, true)] {
                let data = particles(1500, 14.0);
                let items = Items { data: &data, n: 1500, trail_base: 4 + 1500 * STRIDE, look, sprite: with_sprite.then_some(&sprite), s: 1.0, streak: 0.05, taper: 1.0 };
                let one = raster_f16_bands(d, add, &items, 1);
                assert!(one.iter().any(|v| v.to_bits() != 0));
                // The slice conversion rounds as `f16::from_f32` does (the old texel-by-texel upload).
                let mut cv = Canvas { d, px: vec![0.0; d.w as usize * d.h as usize * 4], add };
                draw(&mut cv, &items);
                assert!(cv.px.iter().zip(&one).all(|(f, h)| half::f16::from_f32(*f).to_bits() == h.to_bits()));
                for threads in [2, 3, 4, 7] {
                    assert!(raster_f16_bands(d, add, &items, threads).iter().zip(&one).all(|(a, b)| a.to_bits() == b.to_bits()), "{threads} bands differ from one canvas");
                }
            }
        }
    }

    /// `cargo test -p moda-engine --release sprites_raster_time -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn sprites_raster_time() {
        let d = Domain::new(-2, -2, 1924, 1084);
        let data = particles(2500, 14.0);
        let items = Items { data: &data, n: 2500, trail_base: 4 + 2500 * STRIDE, look: Look::Square, sprite: None, s: 1.0, streak: 0.05, taper: 1.0 };
        let time = |label: &str, f: &dyn Fn()| {
            let t = std::time::Instant::now();
            for _ in 0..20 {
                f();
            }
            eprintln!("{label}: {:.1} ms per canvas", t.elapsed().as_secs_f64() * 50.0);
        };
        // The texel-by-texel path this replaced: one canvas, then `Passes::upload`'s conversion.
        time("one canvas + per-texel bytes", &|| {
            let mut cv = Canvas { d, px: vec![0.0; d.w as usize * d.h as usize * 4], add: false };
            draw(&mut cv, &items);
            let bytes: Vec<u8> = cv.px.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
            std::hint::black_box(bytes);
        });
        for threads in [1, 2, 4] {
            time(&format!("{threads} band(s)"), &|| {
                std::hint::black_box(raster_f16_bands(d, false, &items, threads));
            });
        }
    }
}
