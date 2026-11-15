//! encode.rs — PNG / JPEG encoding of readback pixels.
//!
//! Lanes match the golden capture configurations (DECISION S6):
//! - PNG  = export config (lossless; straight alpha when transparent).
//! - JPEG = screenshot config (`screenshotHelper.ts:62-69`): quality 0.92,
//!   composited on white (the export render is already opaque-on-white when
//!   not transparent; JPEG has no alpha so any alpha is dropped).

use anyhow::{Context, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
    Jpeg,
}

impl ImageFormat {
    pub fn parse(s: &str) -> Result<ImageFormat> {
        match s.to_ascii_lowercase().as_str() {
            "png" => Ok(ImageFormat::Png),
            "jpeg" | "jpg" => Ok(ImageFormat::Jpeg),
            "pdf" => anyhow::bail!(
                "pdf is assembled by moda-pdf at the CLI layer (M5b), not an engine encode lane"
            ),
            other => anyhow::bail!("unknown format '{other}' (png|jpeg)"),
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            ImageFormat::Png => "png",
            ImageFormat::Jpeg => "jpg",
        }
    }
}

/// PNG compression level for the export lane. PNG is lossless, so the *decoded
/// pixels* are identical for every level — only encode speed and file size
/// differ. The renderer's correctness gates compare decoded pixels (SSIM /
/// `--require-exact` mean-abs-diff on decoded RGBA), never the PNG byte stream
/// (verified: `fixtures/thresholds.toml`, `moda-compare`), so the level is a
/// free speed/size knob.
///
/// Default `Fast` (fdeflate ultra-fast) over the `png` crate's `Balanced`
/// default: on the export-config corpus it cuts encode time ~3–10× on the
/// high-entropy gradient/shader pages (the warm-PNG bottleneck, e.g.
/// `mesh-gradient-advanced` 309 ms → tens of ms) with only a modest size
/// increase on those already-incompressible pages. Throughput is an explicit
/// goal for the headless-export + future-video surfaces. Override with
/// `MODA_PNG_COMPRESSION=fastest|fast|balanced|high`.
pub fn png_compression() -> png::Compression {
    // Read the env knob ONCE per process (it's a process-level setting) — the
    // per-call `std::env::var` would otherwise be syscall/lock overhead in a
    // high-throughput encode loop (batch export / future video).
    static LEVEL: std::sync::OnceLock<png::Compression> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("MODA_PNG_COMPRESSION").as_deref() {
        Ok("fastest") => png::Compression::Fastest,
        Ok("balanced") => png::Compression::Balanced,
        Ok("high") => png::Compression::High,
        // default + explicit "fast"
        _ => png::Compression::Fast,
    })
}

pub fn encode_png(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    encode_png8(rgba, width, height, png::ColorType::Rgba, png_compression())
}

/// An 8-bit PNG of `data` (gray, RGB or RGBA rows, as `color` says) at `compression`. The fdeflate levels go through
/// the `png` crate as they are (a 20 Mpx frame in about 0.1 s). A zlib level (`Balanced`, `High`: the PDF host's
/// passthrough rasters) is single-threaded in the `png` crate, about 1 s for 20 Mpx of soft gradients and 4 s for
/// 9 Mpx of photo, so its IDAT is deflated in row stripes side by side ([`zlib_png_rows`]).
fn encode_png8(
    data: &[u8],
    width: u32,
    height: u32,
    color: png::ColorType,
    compression: png::Compression,
) -> Result<Vec<u8>> {
    match zlib_level(compression) {
        Some(level) => {
            let row_len = width as usize * color.samples();
            let rows = zlib_png_rows(height as usize, row_len, color.samples(), level, &|y, out| {
                out.copy_from_slice(&data[y * row_len..(y + 1) * row_len])
            })?;
            png_with_idat(width, height, color, &rows)
        }
        None => {
            let mut out = Vec::new();
            {
                let mut enc = png::Encoder::new(&mut out, width, height);
                enc.set_color(color);
                enc.set_depth(png::BitDepth::Eight);
                enc.set_compression(compression);
                enc.write_header().context("png header")?.write_image_data(data).context("png data")?;
            }
            Ok(out)
        }
    }
}

/// An 8-bit PNG whose IDAT is the zlib stream `zlib`.
fn png_with_idat(width: u32, height: u32, color: png::ColorType, zlib: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, width, height);
        enc.set_color(color);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().context("png header")?;
        for chunk in zlib.chunks(1 << 30) {
            writer.write_chunk(png::chunk::IDAT, chunk).context("png data")?;
        }
    }
    Ok(out)
}

/// The zlib level the `png` crate gives `compression` (flate2's default 6 and best 9), `None` for the levels it
/// encodes with fdeflate or stores.
fn zlib_level(compression: png::Compression) -> Option<u32> {
    match compression {
        png::Compression::Balanced => Some(6),
        png::Compression::High => Some(9),
        _ => None,
    }
}

/// Rows a stripe holds at least: below this the threads cost more than they save.
const MIN_STRIPE_ROWS: usize = 64;

/// The zlib stream of a PNG's IDAT for `height` 8-bit rows (`row_len` bytes each, `bpp` bytes a pixel) at zlib
/// `level`, `row(y, out)` writing row `y` into `out`: rows filtered as the `png` crate filters them ([`filter_row`]),
/// deflated in horizontal stripes, one per core, side by side. Each stripe asks for its own rows, so a caller that
/// derives them (a plane of an RGBA frame) does that work on the stripes' threads, with no whole-image copy. Every
/// stripe but the last ends on a sync flush (an empty stored block: byte-aligned, not final), so the stripes' raw
/// deflate streams join into one; the Adler-32s of their filtered bytes combine into the stream's. A stripe starts
/// with an empty window, so it can't match back into the stripe before it: at one stripe per core of at least
/// [`MIN_STRIPE_ROWS`] rows that costs well under 1% of the bytes. The decoded pixels are the rows exactly.
fn zlib_png_rows(
    height: usize,
    row_len: usize,
    bpp: usize,
    level: u32,
    row: &(dyn Fn(usize, &mut [u8]) + Sync),
) -> Result<Vec<u8>> {
    use flate2::{Compress, Compression, FlushCompress, Status};
    let parts = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(height / MIN_STRIPE_ROWS)
        .max(1);
    let per = height.div_ceil(parts).max(1);
    let stripes: Vec<(usize, usize)> =
        (0..height.max(1)).step_by(per).map(|y| (y, (y + per).min(height))).collect();
    let last = stripes.len() - 1;
    let stripe = |(y0, y1): (usize, usize), finish: bool| -> Result<(Vec<u8>, u32, usize)> {
        let mut compress = Compress::new(Compression::new(level), false);
        let mut out = Vec::new();
        let mut adler = simd_adler32::Adler32::new();
        let mut line = vec![0u8; row_len + 1];
        let (mut prev, mut cur) = (vec![0u8; row_len], vec![0u8; row_len]);
        if y0 > 0 {
            row(y0 - 1, &mut prev);
        }
        // Deflates `input` whole; with a flush, until the flush is out too. Through a fixed buffer: `compress_vec`
        // zero-fills the Vec's whole spare capacity on every call, which per row of a growing stream is quadratic.
        let mut buf = vec![0u8; 128 * 1024];
        let mut deflate = |mut input: &[u8], flush: FlushCompress, out: &mut Vec<u8>| -> Result<()> {
            loop {
                let (before_in, before_out) = (compress.total_in(), compress.total_out());
                let status = compress.compress(input, &mut buf, flush).context("png deflate")?;
                let written = (compress.total_out() - before_out) as usize;
                out.extend_from_slice(&buf[..written]);
                input = &input[(compress.total_in() - before_in) as usize..];
                if input.is_empty() {
                    match flush {
                        FlushCompress::Finish if status != Status::StreamEnd => {}
                        FlushCompress::Sync if written == buf.len() => {}
                        _ => return Ok(()),
                    }
                }
            }
        };
        for y in y0..y1 {
            row(y, &mut cur);
            line[0] = filter_row(bpp, &prev, &cur, &mut line[1..]);
            adler.write(&line);
            deflate(&line, FlushCompress::None, &mut out)?;
            std::mem::swap(&mut prev, &mut cur);
        }
        deflate(&[], if finish { FlushCompress::Finish } else { FlushCompress::Sync }, &mut out)?;
        Ok((out, adler.finish(), (y1 - y0) * (row_len + 1)))
    };
    let parts: Vec<(Vec<u8>, u32, usize)> = if stripes.len() == 1 {
        vec![stripe(stripes[0], true)?]
    } else {
        std::thread::scope(|s| {
            let handles: Vec<_> = stripes
                .iter()
                .enumerate()
                .map(|(i, &rows)| {
                    let stripe = &stripe;
                    s.spawn(move || stripe(rows, i == last))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().map_err(|_| anyhow::anyhow!("png stripe encoder panicked"))?)
                .collect::<Result<Vec<_>>>()
        })?
    };
    // zlib header: deflate with a 32 KiB window, FLEVEL 2 (default) or 3 (maximum), no dictionary.
    let header: [u8; 2] = if level >= 7 { [0x78, 0xda] } else { [0x78, 0x9c] };
    let mut zlib = Vec::with_capacity(2 + parts.iter().map(|p| p.0.len()).sum::<usize>() + 4);
    zlib.extend_from_slice(&header);
    let mut adler = 1u32;
    for (bytes, part_adler, len) in &parts {
        zlib.extend_from_slice(bytes);
        adler = adler32_combine(adler, *part_adler, *len);
    }
    zlib.extend_from_slice(&adler.to_be_bytes());
    Ok(zlib)
}

/// The Adler-32 of `a`'s bytes followed by `len_b` bytes whose Adler-32 is `b` (zlib's `adler32_combine`).
fn adler32_combine(a: u32, b: u32, len_b: usize) -> u32 {
    const BASE: u64 = 65521;
    let rem = len_b as u64 % BASE;
    let (a1, a2) = (u64::from(a & 0xffff), u64::from(a >> 16));
    let (b1, b2) = (u64::from(b & 0xffff), u64::from(b >> 16));
    let s1 = (a1 + b1 + BASE - 1) % BASE;
    let s2 = (rem * a1 % BASE + a2 + b2 + BASE - rem) % BASE;
    (s1 | (s2 << 16)) as u32
}

/// Filters one row into `out` and returns its filter type, choosing as the `png` crate's `Filter::Adaptive` does: of
/// Up, Sub, Average and Paeth (in that order), the one whose bytes, read as signed, have the smallest sum of absolute
/// values, the later one on a tie, the first that sums to zero at once. The four sums come from one pass over the row
/// ([`filter_costs`]); only the chosen filter is written.
fn filter_row(bpp: usize, prev: &[u8], cur: &[u8], out: &mut [u8]) -> u8 {
    if cur == prev {
        // Up is all zeros.
        out.fill(0);
        return 2;
    }
    let n = cur.len();
    let head = bpp.min(n);
    let mut costs = filter_costs(&cur[head..], &cur[..n - head], &prev[head..], &prev[..n - head]);
    // The first pixel has no left neighbour: Sub predicts 0, Average half the byte above, Paeth the byte above.
    for (&x, &b) in cur[..head].iter().zip(&prev[..head]) {
        costs[0] += u64::from(residual(x, b));
        costs[1] += u64::from(residual(x, 0));
        costs[2] += u64::from(residual(x, b >> 1));
        costs[3] += u64::from(residual(x, b));
    }
    let mut best = (u64::MAX, 0u8);
    for (cost, filter) in costs.into_iter().zip([2u8, 1, 3, 4]) {
        if cost <= best.0 {
            best = (cost, filter);
            if cost == 0 {
                break;
            }
        }
    }
    apply_filter(best.1, bpp, prev, cur, out);
    best.1
}

/// Lanes [`filter_costs`] sums in: whole vectors of the Paeth predictor, where a byte at a time it doesn't vectorize.
const COST_LANES: usize = 16;

/// The Up, Sub, Average and Paeth costs of bytes `x`, given each one's left neighbour `a`, the byte above `b` and the
/// byte above-left `c`.
fn filter_costs(x: &[u8], a: &[u8], b: &[u8], c: &[u8]) -> [u64; 4] {
    // A residual is at most 128, so a u16 lane holds 511 of them.
    const BLOCK: usize = COST_LANES * 511;
    let mut costs = [0u64; 4];
    let whole = x.len() / COST_LANES * COST_LANES;
    for start in (0..whole).step_by(BLOCK) {
        let block = start..(start + BLOCK).min(whole);
        let mut lanes = [[0u16; COST_LANES]; 4];
        for (((x, a), b), c) in x[block.clone()]
            .chunks_exact(COST_LANES)
            .zip(a[block.clone()].chunks_exact(COST_LANES))
            .zip(b[block.clone()].chunks_exact(COST_LANES))
            .zip(c[block].chunks_exact(COST_LANES))
        {
            for j in 0..COST_LANES {
                lanes[0][j] += residual(x[j], b[j]);
                lanes[1][j] += residual(x[j], a[j]);
                lanes[2][j] += residual(x[j], average(a[j], b[j]));
                lanes[3][j] += residual(x[j], paeth(a[j], b[j], c[j]));
            }
        }
        for (cost, lanes) in costs.iter_mut().zip(&lanes) {
            *cost += lanes.iter().map(|&l| u64::from(l)).sum::<u64>();
        }
    }
    for (((&x, &a), &b), &c) in x[whole..].iter().zip(&a[whole..]).zip(&b[whole..]).zip(&c[whole..]) {
        costs[0] += u64::from(residual(x, b));
        costs[1] += u64::from(residual(x, a));
        costs[2] += u64::from(residual(x, average(a, b)));
        costs[3] += u64::from(residual(x, paeth(a, b, c)));
    }
    costs
}

/// `x` filtered against `pred`, read as a signed byte, made absolute.
#[inline(always)]
fn residual(x: u8, pred: u8) -> u16 {
    u16::from((x.wrapping_sub(pred) as i8).unsigned_abs())
}

#[inline(always)]
fn average(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b)) >> 1) as u8
}

/// The PNG Paeth predictor of left `a`, above `b` and above-left `c`: of the three, the one closest to `a + b - c`,
/// `a` then `b` on a tie. Written without branches so a block of it vectorizes.
#[inline(always)]
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (a, b, c) = (i16::from(a), i16::from(b), i16::from(c));
    let (pa, pb, pc) = ((b - c).abs(), (a - c).abs(), (a + b - c - c).abs());
    let bc = if pb <= pc { b } else { c };
    (if pa <= pb.min(pc) { a } else { bc }) as u8
}

/// PNG filter `filter` (1 Sub, 2 Up, 3 Average, 4 Paeth) of `cur` over `prev`, into `out`.
fn apply_filter(filter: u8, bpp: usize, prev: &[u8], cur: &[u8], out: &mut [u8]) {
    let n = cur.len();
    let head = bpp.min(n);
    match filter {
        1 => {
            out[..head].copy_from_slice(&cur[..head]);
            for i in head..n {
                out[i] = cur[i].wrapping_sub(cur[i - bpp]);
            }
        }
        2 => {
            for i in 0..n {
                out[i] = cur[i].wrapping_sub(prev[i]);
            }
        }
        3 => {
            for i in 0..head {
                out[i] = cur[i].wrapping_sub(prev[i] >> 1);
            }
            for i in head..n {
                out[i] = cur[i].wrapping_sub(average(cur[i - bpp], prev[i]));
            }
        }
        _ => {
            for i in 0..head {
                out[i] = cur[i].wrapping_sub(prev[i]);
            }
            for i in head..n {
                out[i] = cur[i].wrapping_sub(paeth(cur[i - bpp], prev[i], prev[i - bpp]));
            }
        }
    }
}

/// PDF-host lane (export-latency PR 10.11, `moda-render --png-opaque-rgb`):
/// encode a FULLY OPAQUE frame as color type 2 (RGB) so the PDF assembler can
/// pass the IDAT zlib stream straight into a `FlateDecode` image XObject with
/// PNG predictors (`/DecodeParms /Predictor 15` — img2pdf's technique) instead
/// of decode → channel-split → re-deflate in JS, which profiled as roughly
/// half that host's export residual. PDF images cannot carry interleaved
/// alpha, so the passthrough only exists for RGB — returns `None` when any
/// pixel is not fully opaque (e.g. a `--transparent` overlay slice) and the
/// caller falls back to [`encode_png`]. Same compression knob; pixel-identical
/// to the RGBA encode by construction (alpha is 0xff everywhere it drops).
pub fn encode_png_rgb_if_opaque(rgba: &[u8], width: u32, height: u32) -> Result<Option<Vec<u8>>> {
    if rgba.chunks_exact(4).any(|px| px[3] != 0xff) {
        return Ok(None);
    }
    encode_rgba_plane(rgba, width, height, png::ColorType::Rgb).map(Some)
}

/// An RGBA frame as the two images a PDF draws it with: an RGB PNG (color type 2) and its alpha as an 8-bit
/// grayscale PNG (color type 0), `None` when every pixel is opaque. PDF images carry no interleaved alpha, so a
/// transparent raster is an RGB image with a DeviceGray soft mask (PDF 1.7 §11.6.5.2); with the planes already
/// encoded, the PDF assembler passes both IDAT zlib streams through (`/Predictor 15`, as for
/// [`encode_png_rgb_if_opaque`]) instead of decoding the RGBA PNG, splitting it and deflating both planes again.
/// The planes are encoded side by side. Same compression knob; the alpha and every pixel that shows are the RGBA
/// frame's exactly. Under a fully clear pixel the colour plane holds a visible neighbour's colour
/// ([`bleed_clear_pixels`]), not the black a cleared frame holds.
pub fn encode_png_planes(rgba: &[u8], width: u32, height: u32) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    let opaque = rgba.chunks_exact(4).all(|px| px[3] == 0xff);
    let bled = (!opaque).then(|| bleed_clear_pixels(rgba, width as usize, height as usize));
    let color_src = bled.as_deref().unwrap_or(rgba);
    std::thread::scope(|s| {
        let mask = (!opaque).then(|| s.spawn(|| encode_rgba_plane(rgba, width, height, png::ColorType::Grayscale)));
        let color = encode_rgba_plane(color_src, width, height, png::ColorType::Rgb)?;
        let mask = match mask {
            Some(handle) => Some(handle.join().map_err(|_| anyhow::anyhow!("alpha plane encoder panicked"))??),
            None => None,
        };
        Ok((color, mask))
    })
}

/// The frame with a colour under each fully clear pixel: a visible 4-neighbour's where it has one (the ring around
/// every edge), else the colour carried along its row or column from the nearest filled pixel. A PDF reader or RIP
/// scales an image and its soft mask separately, so the black under clear pixels bleeds into the edge pixels it
/// interpolates with and draws a dark rim round every cut-out (dots of a pattern tile, a photo's cut-out). Alpha is
/// untouched, so the image composites exactly as before at its own size.
fn bleed_clear_pixels(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = rgba.to_vec();
    let mut filled: Vec<bool> = rgba.chunks_exact(4).map(|px| px[3] != 0).collect();
    let copy = |out: &mut [u8], to: usize, from: usize| out.copy_within(from * 4..from * 4 + 3, to * 4);
    // The ring: clear pixels next to a visible one take its colour (from the frame's own alpha, so the ring is one
    // pixel wide and every source is a pixel that shows).
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            if rgba[i * 4 + 3] != 0 {
                continue;
            }
            let neighbours = [
                (x > 0).then(|| i - 1),
                (x + 1 < width).then(|| i + 1),
                (y > 0).then(|| i - width),
                (y + 1 < height).then(|| i + width),
            ];
            if let Some(from) = neighbours.into_iter().flatten().find(|&n| rgba[n * 4 + 3] != 0) {
                copy(&mut out, i, from);
                filled[i] = true;
            }
        }
    }
    // The rest: carried forward from the left or above, then back from the right or below.
    for i in 0..width * height {
        if filled[i] {
            continue;
        }
        let (x, y) = (i % width, i / width);
        if let Some(from) = [(x > 0).then(|| i - 1), (y > 0).then(|| i - width)].into_iter().flatten().find(|&n| filled[n]) {
            copy(&mut out, i, from);
            filled[i] = true;
        }
    }
    for i in (0..width * height).rev() {
        if filled[i] {
            continue;
        }
        let (x, y) = (i % width, i / width);
        let next = [(x + 1 < width).then(|| i + 1), (y + 1 < height).then(|| i + width)];
        if let Some(from) = next.into_iter().flatten().find(|&n| filled[n]) {
            copy(&mut out, i, from);
            filled[i] = true;
        }
    }
    out
}

/// A PNG of one plane of an RGBA frame: its colour as `Rgb`, or its alpha as `Grayscale`. With a zlib level each
/// stripe splits its own rows out of the frame ([`zlib_png_rows`]); otherwise the plane is split first.
fn encode_rgba_plane(rgba: &[u8], width: u32, height: u32, color: png::ColorType) -> Result<Vec<u8>> {
    let (rows, row_len) = (height as usize, width as usize * color.samples());
    let frame_row = |y: usize| &rgba[y * width as usize * 4..(y + 1) * width as usize * 4];
    let row = |y: usize, out: &mut [u8]| match color {
        png::ColorType::Rgb => {
            for (rgb, px) in out.chunks_exact_mut(3).zip(frame_row(y).chunks_exact(4)) {
                rgb.copy_from_slice(&px[..3]);
            }
        }
        _ => {
            for (a, px) in out.iter_mut().zip(frame_row(y).chunks_exact(4)) {
                *a = px[3];
            }
        }
    };
    let compression = png_compression();
    match zlib_level(compression) {
        Some(level) => {
            let zlib = zlib_png_rows(rows, row_len, color.samples(), level, &row)?;
            png_with_idat(width, height, color, &zlib)
        }
        None => {
            let mut plane = vec![0u8; rows * row_len];
            for y in 0..rows {
                row(y, &mut plane[y * row_len..(y + 1) * row_len]);
            }
            encode_png8(&plane, width, height, color, compression)
        }
    }
}

/// JPEG q92 on white — canvas.toBlob('image/jpeg', 0.92) parity.
///
/// M5b (docs/jpeg-lane.md, closes SEAMS §9 KNOWN DRIFT): encoded with
/// mozjpeg under `set_fastest_defaults()` = the JCP_FASTEST compress
/// profile, which mozjpeg documents as producing files identical with
/// libjpeg-turbo — the library behind Chromium/Skia's canvas JPEG encode.
/// Settings mirror Skia's SkJpegEncoder at quality 0.92: q92 quant tables,
/// 4:2:0 chroma subsampling (verified on the captured goldens: SOF0 luma
/// h=2 v=2), baseline sequential, islow DCT. The previous `jpeg-encoder`
/// crate's DCT/downsample rounding cost up to ~0.05 SSIM vs the golden
/// bitstreams even when raw pixels matched; mozjpeg-fastest measures
/// 1.00000 (skimage SSIM) on every formerly-failing fixture.
/// NOTE: mozjpeg's tuned default profile (trellis quantization) does NOT
/// match libjpeg-turbo — `set_fastest_defaults()` is load-bearing.
/// Always present: with `jpeg-parity` (server/host lane) it routes to the
/// mozjpeg byte-parity helper; without it (mobile cross-compile lane, and the
/// wasm32 editor — mozjpeg is vendored C with no wasm32 build) it routes to
/// the image-crate q92 fallback. See the two `encode_jpeg_rgb` variants below.
pub fn encode_jpeg(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for px in rgba.chunks_exact(4) {
        let a = px[3] as u32;
        if a == 255 {
            rgb.extend_from_slice(&px[0..3]);
        } else {
            // Composite straight-alpha pixel over white.
            for c in &px[0..3] {
                rgb.push((((*c as u32) * a + 255 * (255 - a)) / 255) as u8);
            }
        }
    }
    encode_jpeg_rgb(rgb, width, height, 92, Chroma::Sub420)
}

/// Chroma sampling of a JPEG encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chroma {
    /// 4:2:0 (canvas.toBlob parity, [`encode_jpeg`]).
    Sub420,
    /// 4:4:4: full-resolution colour, as the export hosts' still JPEGs.
    Full444,
}

/// A still export's JPEG (`moda-render jpeg`): straight-alpha RGBA composited on white (rounded to nearest),
/// 4:4:4, at `quality` 1..=100 (the IJG scale). libjpeg-turbo, where the export host's pure-JS encoder took 0.3 s a
/// 1080² image and 2 s a 4000 px one.
pub fn encode_still_jpeg(rgba: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
    anyhow::ensure!((1..=100).contains(&quality), "jpeg quality {quality} is outside 1..=100");
    let mut rgb = Vec::with_capacity(rgba.len() / 4 * 3);
    for px in rgba.chunks_exact(4) {
        let a = px[3] as u32;
        if a == 255 {
            rgb.extend_from_slice(&px[0..3]);
        } else {
            for c in &px[0..3] {
                rgb.push((((*c as u32) * a + 255 * (255 - a) + 127) / 255) as u8);
            }
        }
    }
    encode_jpeg_rgb(rgb, width, height, quality, Chroma::Full444)
}

/// Byte-parity lane: mozjpeg under JCP_FASTEST (see [`encode_jpeg`] docs).
#[cfg(feature = "jpeg-parity")]
fn encode_jpeg_rgb(rgb: Vec<u8>, width: u32, height: u32, quality: u8, chroma: Chroma) -> Result<Vec<u8>> {
    // mozjpeg signals errors by panicking (libjpeg setjmp/longjmp translated
    // to unwinds); contain them to a typed render error.
    let out = std::panic::catch_unwind(move || -> Result<Vec<u8>> {
        let mut comp = mozjpeg::Compress::new(mozjpeg::ColorSpace::JCS_RGB);
        comp.set_size(width as usize, height as usize);
        // libjpeg-turbo-identical output (see doc comment). Order matters:
        // this calls jpeg_set_defaults, so quality/subsampling come after.
        comp.set_fastest_defaults();
        if quality >= 24 {
            comp.set_quality(quality as f32);
        } else {
            // Below 24 the IJG tables pass 255, which only extended (non-baseline) JPEG can carry: clamp them to
            // baseline instead, as libjpeg's force_baseline and every baseline encoder do.
            comp.set_luma_qtable(&mozjpeg::qtable::AnnexK_Luma.scaled(quality as f32, quality as f32));
            comp.set_chroma_qtable(&mozjpeg::qtable::AnnexK_Chroma.scaled(quality as f32, quality as f32));
        }
        let px = if chroma == Chroma::Sub420 { (2, 2) } else { (1, 1) };
        comp.set_chroma_sampling_pixel_sizes(px, px);
        if chroma == Chroma::Full444 {
            // Huffman tables fitted to the image: the same pixels in a smaller file, for one more pass.
            comp.set_optimize_coding(true);
        }
        let mut started = comp.start_compress(Vec::new()).context("jpeg start")?;
        started.write_scanlines(&rgb).context("jpeg scanlines")?;
        started.finish().context("jpeg finish")
    })
    .map_err(|p| {
        let msg = p
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| p.downcast_ref::<&str>().copied())
            .unwrap_or("mozjpeg panic");
        anyhow::anyhow!("jpeg encode: {msg}")
    })??;
    Ok(out)
}

/// Without `jpeg-parity` (mobile cross-compile lane: mozjpeg-sys is a C
/// build script needing the platform SDK), encode with the image crate's
/// JpegEncoder at the same q92. NOT byte-parity with the golden bitstreams
/// (different DCT/downsample rounding) — fine on device, never for server
/// export parity.
#[cfg(not(feature = "jpeg-parity"))]
fn encode_jpeg_rgb(rgb: Vec<u8>, width: u32, height: u32, quality: u8, _chroma: Chroma) -> Result<Vec<u8>> {
    use image::ImageEncoder as _;
    let mut out = Vec::new();
    let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality);
    enc.write_image(&rgb, width, height, image::ExtendedColorType::Rgb8)
        .context("jpeg encode (image crate)")?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(png: &[u8]) -> (png::OutputInfo, Vec<u8>) {
        let decoder = png::Decoder::new(std::io::Cursor::new(png));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        buf.truncate(info.buffer_size());
        (info, buf)
    }

    #[test]
    fn rgb_if_opaque_encodes_color_type_2_with_identical_pixels() {
        let (w, h) = (3u32, 2u32);
        let rgba: Vec<u8> = (0..w * h)
            .flat_map(|i| [(i * 37) as u8, (i * 101) as u8, (i * 197) as u8, 0xff])
            .collect();
        let png = encode_png_rgb_if_opaque(&rgba, w, h)
            .unwrap()
            .expect("opaque frame");
        let (info, pixels) = decode(&png);
        assert_eq!(info.color_type, png::ColorType::Rgb);
        let expected: Vec<u8> = rgba
            .chunks_exact(4)
            .flat_map(|px| px[..3].to_vec())
            .collect();
        assert_eq!(pixels, expected);
    }

    #[test]
    fn rgb_if_opaque_declines_any_translucent_pixel() {
        let mut rgba = vec![0xffu8; 4 * 4];
        rgba[7] = 0xfe; // one alpha byte off
        assert!(encode_png_rgb_if_opaque(&rgba, 2, 2).unwrap().is_none());
    }

    #[test]
    fn planes_are_the_frames_rgb_and_alpha_and_an_opaque_frame_has_no_alpha_image() {
        let (w, h) = (5u32, 3u32);
        let rgba: Vec<u8> = (0..w * h)
            .flat_map(|i| [(i * 37) as u8, (i * 101) as u8, (i * 197) as u8, (i * 53) as u8])
            .collect();
        let (color, alpha) = encode_png_planes(&rgba, w, h).unwrap();
        let (info, rgb) = decode(&color);
        assert_eq!(info.color_type, png::ColorType::Rgb);
        // Every pixel that shows keeps its colour; pixel 0 is clear and takes its neighbour's (pixel 1's).
        let mut want: Vec<u8> = rgba.chunks_exact(4).flat_map(|px| px[..3].to_vec()).collect();
        want.copy_within(3..6, 0);
        assert_eq!(rgb, want);
        let (info, a) = decode(&alpha.expect("a translucent frame has an alpha image"));
        assert_eq!(info.color_type, png::ColorType::Grayscale);
        assert_eq!(a, rgba.chunks_exact(4).map(|px| px[3]).collect::<Vec<_>>());

        let opaque: Vec<u8> = rgba.chunks_exact(4).flat_map(|px| [px[0], px[1], px[2], 0xff]).collect();
        let (color, alpha) = encode_png_planes(&opaque, w, h).unwrap();
        assert!(alpha.is_none());
        assert_eq!(decode(&color).1, opaque.chunks_exact(4).flat_map(|px| px[..3].to_vec()).collect::<Vec<_>>());
    }

    /// A clear pixel's colour comes from a visible pixel: the ring round a dot is the dot's colour, and pixels
    /// farther out take a filled neighbour's, never the black the frame holds there. Alpha is the frame's.
    #[test]
    fn clear_pixels_take_a_visible_neighbours_colour_in_the_colour_plane() {
        let (w, h) = (7usize, 5usize);
        let mut rgba = vec![0u8; w * h * 4];
        let mut set = |x: usize, y: usize, px: [u8; 4]| rgba[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&px);
        set(1, 1, [217, 222, 229, 255]); // a light dot
        set(5, 3, [200, 30, 40, 128]); // a translucent red one
        let (color, alpha) = encode_png_planes(&rgba, w as u32, h as u32).unwrap();
        let rgb = decode(&color).1;
        let at = |x: usize, y: usize| &rgb[(y * w + x) * 3..(y * w + x) * 3 + 3];
        assert_eq!(at(1, 1), [217, 222, 229]);
        assert_eq!(at(5, 3), [200, 30, 40]);
        for (x, y) in [(0, 1), (2, 1), (1, 0), (1, 2)] {
            assert_eq!(at(x, y), [217, 222, 229], "ring pixel {x},{y}");
        }
        for (x, y) in [(4, 3), (6, 3), (5, 2), (5, 4)] {
            assert_eq!(at(x, y), [200, 30, 40], "ring pixel {x},{y}");
        }
        assert!(rgb.chunks_exact(3).all(|px| px != [0, 0, 0]), "a clear pixel kept the frame's black: {rgb:?}");
        assert_eq!(decode(&alpha.unwrap()).1, rgba.chunks_exact(4).map(|px| px[3]).collect::<Vec<_>>());
        // A wholly clear frame has no colour to spread and stays as it is.
        let clear = vec![0u8; 3 * 2 * 4];
        assert_eq!(decode(&encode_png_planes(&clear, 3, 2).unwrap().0).1, vec![0u8; 3 * 2 * 3]);
    }

    /// Gradients, a hard edge and noise: rows that pick each filter.
    fn varied(w: usize, h: usize, bpp: usize) -> Vec<u8> {
        let mut s = 7u32;
        (0..w * h * bpp)
            .map(|i| {
                let (x, y, c) = (i / bpp % w, i / bpp / w, i % bpp);
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                match (y / 7) % 3 {
                    0 => (x * 3 + y + c * 50) as u8,
                    1 => if x < w / 2 { 200 } else { 30 + c as u8 },
                    _ => (s >> 24) as u8,
                }
            })
            .collect()
    }

    fn inflate(zlib: &[u8]) -> Vec<u8> {
        use std::io::Read as _;
        let mut out = Vec::new();
        flate2::read::ZlibDecoder::new(zlib).read_to_end(&mut out).unwrap();
        out
    }

    #[test]
    fn zlib_levels_decode_to_the_pixels_in_any_number_of_stripes() {
        for (color, bpp) in [(png::ColorType::Rgba, 4), (png::ColorType::Rgb, 3), (png::ColorType::Grayscale, 1)] {
            // One row, fewer rows than a stripe, several stripes, rows that don't divide evenly.
            for (w, h) in [(1usize, 1usize), (13, 5), (97, 64 * 5 + 3), (300, 1001)] {
                let data = varied(w, h, bpp);
                for compression in [png::Compression::Balanced, png::Compression::High] {
                    let png = encode_png8(&data, w as u32, h as u32, color, compression).unwrap();
                    let (info, pixels) = decode(&png);
                    assert_eq!((info.color_type, info.width, info.height), (color, w as u32, h as u32));
                    assert!(pixels == data, "{color:?} {w}x{h} {compression:?}");
                }
            }
        }
    }

    #[test]
    fn striped_rows_are_the_png_crates_filtered_rows() {
        // The stripes' joined stream inflates to exactly what the png crate's own encoder deflates: the same
        // adaptive filter choice per row, and no byte lost or doubled at a stripe boundary.
        let (w, h, bpp) = (211usize, 64 * 6 + 17, 4usize);
        let data = varied(w, h, bpp);
        let row_len = w * bpp;
        let ours =
            zlib_png_rows(h, row_len, bpp, 6, &|y, out| out.copy_from_slice(&data[y * row_len..(y + 1) * row_len]))
                .unwrap();
        let mut theirs = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut theirs, w as u32, h as u32);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            enc.set_compression(png::Compression::Balanced);
            enc.write_header().unwrap().write_image_data(&data).unwrap();
        }
        // The png crate's file: signature, IHDR (8 + 13 + 4), then one IDAT for an image this small.
        let idat_len = u32::from_be_bytes(theirs[33..37].try_into().unwrap()) as usize;
        assert_eq!(&theirs[37..41], b"IDAT");
        assert_eq!(inflate(&ours), inflate(&theirs[41..41 + idat_len]));
    }

    #[test]
    fn paeth_is_the_specs_predictor() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                for c in 0..=255u8 {
                    let (ai, bi, ci) = (i16::from(a), i16::from(b), i16::from(c));
                    let p = ai + bi - ci;
                    let (pa, pb, pc) = ((p - ai).abs(), (p - bi).abs(), (p - ci).abs());
                    let spec = if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c };
                    assert_eq!(paeth(a, b, c), spec, "a={a} b={b} c={c}");
                }
            }
        }
    }

    #[test]
    fn adler32_combine_matches_one_pass() {
        let data: Vec<u8> = (0..200_000u32).map(|i| (i * 31 % 251) as u8).collect();
        let whole = simd_adler32::adler32(&data.as_slice());
        for split in [0, 1, 65_521, 100_000, 200_000] {
            let (a, b) = data.split_at(split);
            let combined = adler32_combine(simd_adler32::adler32(&a), simd_adler32::adler32(&b), b.len());
            assert_eq!(combined, whole, "split at {split}");
        }
    }
}
