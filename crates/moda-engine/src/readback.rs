//! readback.rs — GPU → CPU pixel readback
//! (`WebGPURenderer.ts:483-500` + `renderers/webgpu/utils/pixelFormat.ts:9-74`).
//!
//! - `bytes_per_row` is 256-aligned (`ceil(w·4/256)·256`,
//!   `COPY_BYTES_PER_ROW_ALIGNMENT`).
//! - The returned bytes are ALWAYS canonical tightly-packed RGBA. The default
//!   frame format is rgba8unorm (never -srgb), read out verbatim; a
//!   `Bgra8Unorm` frame texture (a preferred-BGRA live engine) is
//!   canonicalized here — B↔R swizzled — BEFORE any un-premultiplication or
//!   caller exposure, exactly like the TS readback swizzle
//!   (Chromium-on-macOS prefers bgra8unorm; the CONTENT is identical after).
//! - Un-premultiply ONLY for transparent exports, with round-half-up integer
//!   math `min(255, floor((c·255 + (a>>1)) / a))`; a==0 → rgb 0. Opaque
//!   exports keep bytes as-is.

use anyhow::{Context, Result};

/// Reusable staging storage for at most two in-flight readbacks, each capped at 32 MiB.
#[derive(Default, Clone)]
pub struct ReadbackPool(std::rc::Rc<std::cell::RefCell<ReadbackBuffers>>);

#[derive(Default)]
struct ReadbackBuffers {
    free: Vec<wgpu::Buffer>,
    active: usize,
    spent: RgbaRecycler,
}

impl ReadbackPool {
    /// Where a finished readback's consumer hands its spent pixels back ([`RgbaRecycler`]).
    pub fn recycler(&self) -> RgbaRecycler {
        self.0.borrow().spent.clone()
    }
}

/// Spent RGBA frame buffers, handed back by the code that consumed a finished readback (an export's frame writer,
/// once the frame is in the pipe or the PNG), so the next readback of the same size writes into memory that is
/// already mapped instead of a fresh 8 MB (1080p) allocation's zeroed pages. It holds no pixels anyone reads: a
/// buffer is taken only at exactly the readback's size, and the readback overwrites every byte of it. At most
/// [`RgbaRecycler::KEEP`] buffers are held.
#[derive(Default, Clone)]
pub struct RgbaRecycler(std::sync::Arc<std::sync::Mutex<Vec<Vec<u8>>>>);

impl RgbaRecycler {
    /// The frames an export can have in flight at once (two queued for the writer, one being written, one being
    /// read back), so a steady export allocates no frame after its first few.
    const KEEP: usize = 4;

    /// Hand a spent buffer back. Its contents are never read again.
    pub fn give(&self, buffer: Vec<u8>) {
        let Ok(mut spent) = self.0.lock() else { return };
        if spent.len() < Self::KEEP && !buffer.is_empty() {
            spent.push(buffer);
        }
    }

    /// A buffer of exactly `len` bytes whose contents are unspecified (a spent frame's, or zeros): the caller
    /// overwrites all of it.
    fn take(&self, len: usize) -> Vec<u8> {
        let reused = self.0.lock().ok().and_then(|mut spent| {
            spent.retain(|buffer| buffer.len() == len);
            spent.pop()
        });
        reused.unwrap_or_else(|| vec![0; len])
    }
}

/// A submitted RGBA readback. Dropping it cancels mapping and returns its staging buffer.
pub struct PendingRgba {
    buffer: Option<wgpu::Buffer>,
    pool: ReadbackPool,
    completion: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    submission: Option<wgpu::SubmissionIndex>,
    ready: Option<Vec<u8>>,
    width: u32,
    height: u32,
    bgra: bool,
    transparent: bool,
    retries: u32,
}

impl PendingRgba {
    pub fn finish(mut self, device: &wgpu::Device) -> Result<Vec<u8>> {
        let _readback = crate::prof::span("gpu.readback");
        let mut out = if let Some(ready) = self.ready.take() {
            ready
        } else {
            let wait = crate::prof::span("gpu.readback.wait");
            poll_submission_with_budget(device, self.submission.take().expect("submitted"), self.retries)?;
            self.completion.take().expect("mapping").recv().context("map_async channel")??;
            drop(wait);
            let _copy = crate::prof::span("gpu.readback.copy");
            let buffer = self.buffer.as_ref().expect("staging buffer");
            let data = buffer.slice(..).get_mapped_range();
            // Every byte is written below: the tightly packed rows of the whole image.
            let mut out = self.pool.recycler().take(self.width as usize * self.height as usize * 4);
            assemble_band(&mut out, &data, padded_bytes_per_row(self.width) as usize, self.width as usize * 4, 0, self.height as usize);
            if self.bgra {
                bgra8_to_rgba8_in_place(&mut out);
            }
            out
        };
        if self.transparent {
            unpremultiply_in_place(&mut out);
        }
        Ok(out)
    }
}

impl Drop for PendingRgba {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            buffer.unmap();
            let mut buffers = self.pool.0.borrow_mut();
            buffers.active -= 1;
            buffers.free.push(buffer);
        }
    }
}

pub(crate) fn read_texture_rgba_start(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    pool: &ReadbackPool,
    transparent: bool,
    retries: u32,
) -> Result<PendingRgba> {
    let (width, height) = (texture.width(), texture.height());
    let stride = padded_bytes_per_row(width);
    let size = u64::from(stride) * u64::from(height);
    let mut pending = PendingRgba {
        buffer: None, pool: pool.clone(), completion: None, submission: None, ready: None,
        width, height, bgra: texture.format() == wgpu::TextureFormat::Bgra8Unorm, transparent, retries,
    };
    if size > device.limits().max_buffer_size.min(32 * 1024 * 1024) {
        pending.ready = Some(read_texture_rgba_with_budget(device, queue, texture, width, height, retries)?);
        return Ok(pending);
    }
    let buffer = {
        let mut buffers = pool.0.borrow_mut();
        anyhow::ensure!(buffers.active < 2, "export readback: at most two frames may be in flight");
        buffers.free.retain(|buffer| buffer.size() == size);
        buffers.active += 1;
        buffers.free.pop()
    }.unwrap_or_else(|| device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Pipelined Export Read Buffer"), size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    }));
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Pipelined Export Copy") });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(stride), rows_per_image: Some(height) } },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    crate::scratch::flush();
    pending.submission = Some(queue.submit([encoder.finish()]));
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| { let _ = tx.send(result); });
    pending.buffer = Some(buffer);
    pending.completion = Some(rx);
    Ok(pending)
}

pub fn padded_bytes_per_row(width: u32) -> u32 {
    padded_bytes_per_row_bpp(width, 4)
}

/// [`padded_bytes_per_row`] for `bytes_per_pixel`-byte texels (8 for `Rgba16Float`).
pub fn padded_bytes_per_row_bpp(width: u32, bytes_per_pixel: u32) -> u32 {
    (width * bytes_per_pixel).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
}

/// Read an `Rgba16Float` texture (sample_count 1) back as tightly packed RGBA half floats, as stored
/// (premultiplied; whatever encoding the texels hold). The same banded copy-out as the 8-bit readback.
pub fn read_texture_rgba16f(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Result<Vec<half::f16>> {
    anyhow::ensure!(
        texture.format() == wgpu::TextureFormat::Rgba16Float,
        "read_texture_rgba16f: texture is {:?}",
        texture.format()
    );
    let bytes = read_texture_rgba_staged(
        device,
        queue,
        texture,
        width,
        height,
        READBACK_POLL_TIMEOUT_RETRIES,
        device.limits().max_buffer_size,
    )?;
    Ok(bytes
        .chunks_exact(2)
        .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect())
}

/// How many texture rows fit in ONE mappable staging buffer of
/// `max_staging_bytes` — the band height of [`read_texture_rgba_staged`].
///
/// A row is the indivisible unit of a `copy_texture_to_buffer` here (the copy's
/// `bytes_per_row` is fixed by the texture width), so the result is at least 1;
/// it is capped at `height` so an image that already fit reads back in exactly
/// ONE band — the same single copy + single map this function's caller has
/// always issued.
///
/// Degenerate inputs (`bytes_per_row == 0`, i.e. a zero-width texture) return
/// the full height rather than dividing by zero. The caller rejects zero
/// dimensions before it gets here, so that branch is defence in depth.
pub(crate) fn readback_rows_per_band(
    bytes_per_row: u32,
    max_staging_bytes: u64,
    height: u32,
) -> u32 {
    if bytes_per_row == 0 {
        return height.max(1);
    }
    let fit = max_staging_bytes / bytes_per_row as u64;
    fit.max(1).min(height.max(1) as u64) as u32
}

/// Copy ONE readback band's rows out of its 256-aligned staging bytes into the
/// assembled tightly-packed image.
///
/// Split out of [`read_texture_rgba_staged`] so the band-assembly ARITHMETIC —
/// the destination offset, the short final band, the unpadded fast path — is
/// testable with no GPU adapter and therefore actually gated by CI's
/// `cargo test -p moda-engine` (rubric 6.2c: a test CI does not run is not a
/// test). The GPU test remains the end-to-end confirmation; this is what stops a
/// future refactor from silently shifting or duplicating a band, whose symptom is
/// not a crash but wrong pixels in every export large enough to need banding.
///
/// `band` must hold at least `rows * bytes_per_row` bytes and `out` must be the
/// full `height * row_bytes` image; `first_row` is the band's first row in it.
fn assemble_band(
    out: &mut [u8],
    band: &[u8],
    bytes_per_row: usize,
    row_bytes: usize,
    first_row: usize,
    rows: usize,
) {
    let band_start = first_row * row_bytes;
    if bytes_per_row == row_bytes {
        // No 256-byte padding — true whenever width % 64 == 0, which covers most
        // real export widths (7040, 3840, 1920, …). The band is then laid out
        // identically in the staging buffer and in `out`, so it moves in ONE
        // memcpy instead of `rows` of them (9922 of them for the incident
        // geometry).
        let len = rows * row_bytes;
        out[band_start..band_start + len].copy_from_slice(&band[..len]);
    } else {
        for y in 0..rows {
            let src = y * bytes_per_row;
            let dst = band_start + y * row_bytes;
            out[dst..dst + row_bytes].copy_from_slice(&band[src..src + row_bytes]);
        }
    }
}

/// How many extra `wait_indefinitely` polls to issue after a `PollError::Timeout`
/// before giving up. A timeout here is NOT device loss (`device::is_device_lost_error`
/// docs) — it is the platform GPU watchdog interrupting a long fence wait on a
/// slow frame (mobile Vulkan first-render shader compile / heavy draw). The
/// submitted work keeps running; re-polling waits for it again, so a handful of
/// retries lets a legitimately slow render settle instead of surfacing as
/// a user-facing "renderer failed during readback" load error. Bounded (never
/// infinite) so a genuinely wedged submission still fails eventually.
pub(crate) const READBACK_POLL_TIMEOUT_RETRIES: u32 = 8;

/// Budget for the FIRST readback after device creation. A cold first render on
/// real mobile hardware compiles every pipeline it touches inside the fence
/// wait, and on drivers whose watchdog interrupts that wait with `VK_TIMEOUT`
/// (rather than letting `wait_indefinitely` run to a `u64::MAX` fence wait), the
/// steady-state budget can be exhausted before the compile settles — surfacing
/// as a spurious "failed to load canvas" that succeeds on a (now-warm) retry.
/// The first readback gets a larger ceiling so the cold compile completes;
/// every subsequent readback drops back to the steady-state budget so a
/// genuinely wedged submission still surfaces promptly. Still bounded.
pub(crate) const READBACK_POLL_TIMEOUT_RETRIES_FIRST: u32 = 24;

/// Poll the device until the most recent submission completes, tolerating a
/// bounded number (`retries`) of `PollError::Timeout`s (the GPU watchdog
/// interrupting a slow frame). Any non-timeout poll error — or exhausting the
/// retry budget — propagates. `wait_indefinitely`'s `None` timeout maps to a
/// `u64::MAX` Vulkan fence wait, so each individual poll already blocks as long
/// as the driver allows; the retries exist only for drivers whose watchdog cuts
/// that wait short with `VK_TIMEOUT` rather than letting it run to completion.
/// The first cold readback passes [`READBACK_POLL_TIMEOUT_RETRIES_FIRST`];
/// steady state passes [`READBACK_POLL_TIMEOUT_RETRIES`].
pub(crate) fn poll_until_complete_with_budget(device: &wgpu::Device, retries: u32) -> Result<()> {
    let mut timeouts = 0u32;
    loop {
        match device.poll(wgpu::PollType::wait_indefinitely()) {
            Ok(_) => return Ok(()),
            Err(wgpu::PollError::Timeout) if timeouts < retries => {
                timeouts += 1;
                eprintln!(
                    "moda-engine: warn: readback poll timed out (slow frame); \
                     retry {timeouts}/{retries}"
                );
            }
            Err(e) => return Err(e.into()),
        }
    }
}

/// [`poll_until_complete_with_budget`] for one submission: waits until `index` has executed (and its map
/// callbacks ran), not for work submitted after it.
pub(crate) fn poll_submission_with_budget(
    device: &wgpu::Device,
    index: wgpu::SubmissionIndex,
    retries: u32,
) -> Result<()> {
    let mut timeouts = 0u32;
    loop {
        match device.poll(wgpu::PollType::Wait {
            submission_index: Some(index.clone()),
            timeout: None,
        }) {
            Ok(_) => return Ok(()),
            Err(wgpu::PollError::Timeout) if timeouts < retries => {
                timeouts += 1;
                eprintln!(
                    "moda-engine: warn: readback poll timed out (slow frame); \
                     retry {timeouts}/{retries}"
                );
            }
            Err(e) => return Err(e.into()),
        }
    }
}

/// Copy `texture` (rgba8unorm or bgra8unorm, sample_count 1) into a mappable
/// buffer and return tightly-packed CANONICAL RGBA bytes — a bgra8unorm
/// texture is swizzled on the way out. Uses the steady-state
/// readback poll budget.
pub fn read_texture_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    read_texture_rgba_with_budget(
        device,
        queue,
        texture,
        width,
        height,
        READBACK_POLL_TIMEOUT_RETRIES,
    )
}

/// As [`read_texture_rgba`] but with an explicit timeout-retry budget for the
/// map-completion poll (the cold first readback passes the larger first-render
/// budget; see [`READBACK_POLL_TIMEOUT_RETRIES_FIRST`]).
pub(crate) fn read_texture_rgba_with_budget(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    poll_retries: u32,
) -> Result<Vec<u8>> {
    read_texture_rgba_staged(
        device,
        queue,
        texture,
        width,
        height,
        poll_retries,
        device.limits().max_buffer_size,
    )
}

/// As [`read_texture_rgba_with_budget`] but with an explicit staging-buffer
/// budget — the seam that keeps a GIANT finished render inside
/// `max_buffer_size` by chunking the COPY-OUT rather than the render.
///
/// **Why this exists.** The render target is bounded by
/// `max_texture_dimension_2d` (`min(16384, adapter)` — see
/// `device::GpuContext::device_descriptor`), but the mappable readback buffer
/// is bounded by the much smaller `max_buffer_size` (wgpu's default 256 MB).
/// Those two limits disagree by 16x, so the engine would happily RENDER an
/// image it could not then hand back: a 7040x9922 page export (a 3520x4961
/// page at pixelRatio 2 — 69.9 MP) rendered fine and then panicked in
/// `Device::create_buffer, label = 'Export Read Buffer'` with
/// `Buffer size 279403520 is greater than the maximum buffer size (268435456)`
/// — over by 4.1%, on a texture whose pixels were already correct and resident
/// on the GPU. Because that is a size check against a fixed constant it is
/// deterministic: the affected page was permanently un-exportable at 2x, and
/// the browser fallback (same spec-default limit in Chromium) could not rescue
/// it either.
///
/// **What this does.** ONE render, ONE texture, unchanged — only the copy-out
/// is split, into consecutive full-width row bands each sized to fit
/// `max_staging_bytes` ([`readback_rows_per_band`]). Every band is a
/// `copy_texture_to_buffer` of a sub-rect of the SAME finished texture into a
/// single reused staging buffer, so the assembled result is the same bytes the
/// single-buffer readback produced, in the same order.
///
/// **Bit-identity contract.** When the whole image fits (`rows_per_band >=
/// height`) the loop runs exactly once, at `origin.y = 0` with the full extent
/// — byte for byte the commands this function issued before chunking existed.
/// Above that size the output is defined to be identical to what an
/// unconstrained single buffer would have returned; `chunked_readback_is_bit_identical`
/// pins both halves against a real GPU.
///
/// Peak staging memory DROPS (one band mapped at a time instead of the whole
/// image); the tightly-packed output `Vec` is unchanged at `w * h * 4`.
///
/// This does NOT lift the per-axis texture limit, nor the encode/RSS ceilings
/// above it — `export_scale::clamp_export_scale` remains the backstop for
/// those. It removes the readback buffer as the binding constraint, which at
/// the default 256 MB it always was (~67 MP vs the axis cap's ~268 MP).
pub(crate) fn read_texture_rgba_staged(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    poll_retries: u32,
    max_staging_bytes: u64,
) -> Result<Vec<u8>> {
    // A zero-dimension texture has no pixels to copy, and asking wgpu for the
    // zero-sized staging buffer it implies is a validation error. Banding made
    // the two degenerate axes disagree — a 0-height image now falls out of the
    // loop with an empty Vec while a 0-width one would still have allocated a
    // 0-byte buffer and panicked — so settle both here, the same way.
    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }
    // 4 bytes for the 8-bit frame formats, 8 for `Rgba16Float` (the pixel graph's captures).
    let bpp = texture.format().block_copy_size(None).unwrap_or(4);
    let bytes_per_row = padded_bytes_per_row_bpp(width, bpp);
    let rows_per_band = readback_rows_per_band(bytes_per_row, max_staging_bytes, height);
    let row_bytes = width as usize * bpp as usize;
    // usize (not u32) throughout: at the per-axis cap this is ~1.07 GB, and the
    // whole point of this function is that images that large now reach it.
    let mut out = vec![0u8; row_bytes * height as usize];

    // ONE staging buffer, sized for the largest band and reused across bands:
    // the alternative (a buffer per band) churns allocations for no benefit,
    // since only one band is in flight at a time.
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Export Read Buffer"),
        size: bytes_per_row as u64 * rows_per_band as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut first_row = 0u32;
    while first_row < height {
        let rows = rows_per_band.min(height - first_row);
        // Map only the prefix this band actually wrote — the final band is
        // short. `bytes_per_row` is 256-aligned, so the size is a multiple of
        // `wgpu::MAP_ALIGNMENT` for every band.
        let band_bytes = bytes_per_row as u64 * rows as u64;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Export Copy"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: first_row,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
        crate::scratch::flush();
        queue.submit([encoder.finish()]);

        let slice = buffer.slice(..band_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        poll_until_complete_with_budget(device, poll_retries)
            .context("device poll during readback")?;
        rx.recv().context("map_async channel")??;

        let data = slice.get_mapped_range();
        assemble_band(
            &mut out,
            &data[..],
            bytes_per_row as usize,
            row_bytes,
            first_row as usize,
            rows as usize,
        );
        drop(data);
        // Unmap before the next band re-maps the same buffer.
        buffer.unmap();

        first_row += rows;
    }

    // Canonicalize a BGRA frame texture's texels to RGBA HERE — the
    // single native readback seam — so every caller (export/PDF/goldens,
    // un-premultiplication, comparisons) always sees canonical RGBA bytes no
    // matter the engine's frame format. Applied once to the assembled image,
    // so band count cannot affect it.
    if texture.format() == wgpu::TextureFormat::Bgra8Unorm {
        bgra8_to_rgba8_in_place(&mut out);
    }
    Ok(out)
}

/// Swap the B and R channels of packed 4-byte pixels in place — canonicalize
/// physically-BGRA readback bytes to RGBA. Alpha-independent, so
/// ordering against un-premultiplication is a documentation contract
/// (canonicalize FIRST), not a numeric one.
pub fn bgra8_to_rgba8_in_place(px: &mut [u8]) {
    debug_assert_eq!(px.len() % 4, 0);
    for p in px.chunks_exact_mut(4) {
        p.swap(0, 2);
    }
}

/// `unpremultiplyChannel` (pixelFormat.ts:9-13): round-half-up integer math.
#[inline]
fn unpremultiply_channel(premultiplied: u8, alpha: u8) -> u8 {
    if alpha == 0 {
        return 0;
    }
    if alpha == 255 {
        return premultiplied;
    }
    let v = (premultiplied as u32 * 255 + (alpha as u32 >> 1)) / alpha as u32;
    v.min(255) as u8
}

/// In-place straight-alpha conversion for transparent exports
/// (pixelFormat.ts:50-62: a==0 → rgb zeroed).
pub fn unpremultiply_in_place(rgba: &mut [u8]) {
    for px in rgba.chunks_exact_mut(4) {
        let a = px[3];
        if a == 0 {
            px[0] = 0;
            px[1] = 0;
            px[2] = 0;
        } else if a < 255 {
            px[0] = unpremultiply_channel(px[0], a);
            px[1] = unpremultiply_channel(px[1], a);
            px[2] = unpremultiply_channel(px[2], a);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recycled_buffer_is_taken_only_at_its_exact_size_and_few_are_kept() {
        let spent = RgbaRecycler::default();
        // Nothing spent yet: a fresh buffer of the size asked for.
        assert_eq!(spent.take(16), vec![0; 16]);
        let a = vec![7u8; 16];
        let ptr = a.as_ptr();
        spent.give(a);
        // Same size: that very buffer (its old bytes are the caller's to overwrite).
        let again = spent.take(16);
        assert_eq!(again.as_ptr(), ptr);
        assert_eq!(again.len(), 16);
        // Another size: never handed out, and dropped rather than kept.
        spent.give(vec![1u8; 32]);
        assert_eq!(spent.take(16), vec![0; 16]);
        assert!(spent.0.lock().unwrap().is_empty());
        // At most KEEP buffers are held.
        for _ in 0..RgbaRecycler::KEEP + 3 {
            spent.give(vec![0u8; 8]);
        }
        assert_eq!(spent.0.lock().unwrap().len(), RgbaRecycler::KEEP);
        // An empty buffer is not kept.
        let empty = RgbaRecycler::default();
        empty.give(Vec::new());
        assert!(empty.0.lock().unwrap().is_empty());
    }

    #[test]
    fn round_half_up_matches_ts() {
        // floor((c*255 + (a>>1)) / a)
        assert_eq!(unpremultiply_channel(128, 128), 255);
        assert_eq!(unpremultiply_channel(64, 128), 128); // (64·255 + 64)/128 = 16384/128 = 128
        assert_eq!(unpremultiply_channel(0, 0), 0);
        assert_eq!(unpremultiply_channel(200, 255), 200);
    }

    #[test]
    fn alignment() {
        assert_eq!(padded_bytes_per_row(1000), 4096);
        assert_eq!(padded_bytes_per_row(64), 256);
        assert_eq!(padded_bytes_per_row(1280), 5120);
    }

    #[test]
    fn bgra8_to_rgba8_in_place_swaps_b_and_r() {
        // physically-BGRA readback bytes canonicalize to RGBA by
        // swapping bytes 0 and 2 per pixel; G and A are untouched.
        let mut px = [10u8, 20, 30, 40, 1, 2, 3, 4];
        bgra8_to_rgba8_in_place(&mut px);
        assert_eq!(px, [30, 20, 10, 40, 3, 2, 1, 4]);
        // Involution: applying twice restores the original.
        bgra8_to_rgba8_in_place(&mut px);
        assert_eq!(px, [10, 20, 30, 40, 1, 2, 3, 4]);
        // Empty slice is a no-op, not a panic.
        bgra8_to_rgba8_in_place(&mut []);
    }

    /// wgpu's default `max_buffer_size` — the limit the engine actually runs
    /// with (`device_descriptor` raises `max_texture_dimension_2d` and leaves
    /// this at the default).
    const WGPU_DEFAULT_MAX_BUFFER_SIZE: u64 = 268_435_456;

    #[test]
    fn rows_per_band_is_the_whole_image_when_it_fits() {
        // The overwhelmingly common case: one band == the pre-chunking
        // single-copy readback, which is what makes the change byte-inert for
        // every export that worked before.
        assert_eq!(
            readback_rows_per_band(
                padded_bytes_per_row(1920),
                WGPU_DEFAULT_MAX_BUFFER_SIZE,
                1080
            ),
            1080
        );
        // A 4K page at 2x (7680x4320, 132 MB) still fits in one band.
        assert_eq!(
            readback_rows_per_band(
                padded_bytes_per_row(7680),
                WGPU_DEFAULT_MAX_BUFFER_SIZE,
                4320
            ),
            4320
        );
    }

    #[test]
    fn rows_per_band_splits_the_incident_geometry() {
        // API-FASTAPI-1SN: a 3520x4961 page exported at pixelRatio 2 renders
        // 7040x9922. Its readback wanted 28160 * 9922 = 279,403,520 bytes
        // against a 268,435,456 limit — over by 4.1%, a hard
        // `Device::create_buffer` validation panic that no retry could fix.
        let bytes_per_row = padded_bytes_per_row(7040);
        assert_eq!(bytes_per_row, 28_160, "7040*4 is already 256-aligned");
        assert_eq!(bytes_per_row as u64 * 9922, 279_403_520);
        assert!(bytes_per_row as u64 * 9922 > WGPU_DEFAULT_MAX_BUFFER_SIZE);

        let rows = readback_rows_per_band(bytes_per_row, WGPU_DEFAULT_MAX_BUFFER_SIZE, 9922);
        // Two bands now, each comfortably inside the limit.
        assert_eq!(rows, 9532);
        assert!(bytes_per_row as u64 * rows as u64 <= WGPU_DEFAULT_MAX_BUFFER_SIZE);
        assert_eq!(9922u32.div_ceil(rows), 2);
    }

    #[test]
    fn rows_per_band_handles_degenerate_inputs() {
        // Never zero (a row is the indivisible copy unit) — a budget smaller
        // than one row still yields a one-row band rather than an infinite loop.
        assert_eq!(readback_rows_per_band(4096, 0, 100), 1);
        assert_eq!(readback_rows_per_band(4096, 1, 100), 1);
        // Never more than the image height.
        assert_eq!(readback_rows_per_band(4096, u64::MAX, 7), 7);
        // Zero-width (bytes_per_row 0) must not divide by zero.
        assert_eq!(
            readback_rows_per_band(0, WGPU_DEFAULT_MAX_BUFFER_SIZE, 12),
            12
        );
        // Zero-height must not panic in the clamp.
        assert_eq!(
            readback_rows_per_band(4096, WGPU_DEFAULT_MAX_BUFFER_SIZE, 0),
            1
        );
    }

    /// Reference image: tightly-packed RGBA whose every byte is a function of
    /// (x, y), so a band written at the wrong offset — or a row dropped or
    /// duplicated at a seam — cannot coincidentally compare equal.
    fn reference_image(w: usize, h: usize) -> Vec<u8> {
        let mut px = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                px[i] = (x % 251) as u8;
                px[i + 1] = (y % 241) as u8;
                px[i + 2] = ((x * 7 + y * 13) % 239) as u8;
                px[i + 3] = 255;
            }
        }
        px
    }

    /// Byte written into the row padding — it must never reach the output, so it
    /// has to be a value [`reference_image`] cannot itself produce. Its channels
    /// are `x % 251`, `y % 241`, `_ % 239` and a constant 255, so 251..=254 are
    /// all unreachable; 0xCD would NOT be (it appears in the blue channel).
    const PAD_SENTINEL: u8 = 0xFE;

    /// The 256-aligned staging bytes wgpu would hand back for `reference_image`
    /// — each row padded out to `bytes_per_row`, padding filled with
    /// [`PAD_SENTINEL`].
    fn staging_bytes(image: &[u8], w: usize, h: usize, bytes_per_row: usize) -> Vec<u8> {
        let row_bytes = w * 4;
        let mut staged = vec![PAD_SENTINEL; bytes_per_row * h];
        for y in 0..h {
            staged[y * bytes_per_row..y * bytes_per_row + row_bytes]
                .copy_from_slice(&image[y * row_bytes..(y + 1) * row_bytes]);
        }
        staged
    }

    /// CPU-side proof of the band-assembly arithmetic — the half of the fix that
    /// CI can actually gate (`cargo test -p moda-engine` runs on GPU-less Linux
    /// runners, where the GPU test below skips itself). Walks the same band loop
    /// `read_texture_rgba_staged` walks, for every band height from 1 to h, at a
    /// PADDED width (row loop) and an ALIGNED width (contiguous fast path), and
    /// requires the assembled result to equal the reference exactly.
    #[test]
    fn band_assembly_reassembles_the_image_at_every_band_height() {
        for (w, h) in [(100usize, 37usize), (64, 37), (64, 1), (1, 5)] {
            let row_bytes = w * 4;
            let bytes_per_row = padded_bytes_per_row(w as u32) as usize;
            let image = reference_image(w, h);
            let staged = staging_bytes(&image, w, h, bytes_per_row);

            for rows_per_band in 1..=h {
                let mut out = vec![0u8; row_bytes * h];
                let mut first_row = 0usize;
                while first_row < h {
                    let rows = rows_per_band.min(h - first_row);
                    // The band's staging bytes start at its first row — exactly
                    // what copy_texture_to_buffer writes at offset 0 for a copy
                    // whose origin.y is `first_row`.
                    let band = &staged[first_row * bytes_per_row..][..rows * bytes_per_row];
                    assemble_band(&mut out, band, bytes_per_row, row_bytes, first_row, rows);
                    first_row += rows;
                }
                assert_eq!(
                    out, image,
                    "{w}x{h} at {rows_per_band} rows/band did not reassemble the image"
                );
            }
        }
    }

    #[test]
    fn band_assembly_never_copies_row_padding_into_the_output() {
        // A padded width whose padding byte (0xCD) would be visible if the copy
        // used bytes_per_row as the output stride.
        let (w, h) = (100usize, 8usize);
        let row_bytes = w * 4;
        let bytes_per_row = padded_bytes_per_row(w as u32) as usize;
        assert!(bytes_per_row > row_bytes, "geometry must actually pad");
        let image = reference_image(w, h);
        let staged = staging_bytes(&image, w, h, bytes_per_row);

        let mut out = vec![0u8; row_bytes * h];
        assemble_band(&mut out, &staged, bytes_per_row, row_bytes, 0, h);
        assert_eq!(out, image);
        assert!(
            !out.contains(&PAD_SENTINEL),
            "row padding leaked into the image"
        );
    }

    /// Regression for the CAPTURE_TIMEOUT/oversized-export class: a readback
    /// forced into MANY bands must return exactly the bytes the single-band
    /// (pre-chunking) readback returns.
    ///
    /// Uses `write_texture` rather than a render pass on purpose — this pins the
    /// readback seam itself, independent of any plan/rasteriser behaviour, so it
    /// cannot be perturbed by unrelated renderer changes. The geometry is chosen
    /// to exercise every edge the real 7040x9922 case does, at a size that runs
    /// anywhere: a width whose row needs real 256-byte padding (100*4 = 400 ->
    /// 512) and a height that does NOT divide evenly by the band height, so the
    /// final band is short.
    ///
    /// Skips gracefully with no GPU adapter, like the other GPU tests in this
    /// crate. Run moda-engine's GPU tests with `--test-threads=1` (parallel
    /// device init deadlocks).
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn chunked_readback_is_bit_identical() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        // BOTH copy-out branches are exercised: width 100 pads (400 -> 512) and
        // takes the per-row loop; width 64 is exactly 256-aligned and takes the
        // contiguous whole-band memcpy. Testing only the padded width would
        // leave the fast path — the one most real export widths hit — unproven.
        for (w, expected_bpr) in [(100u32, 512u32), (64u32, 256u32)] {
            assert_eq!(padded_bytes_per_row(w), expected_bpr);
            check_bit_identity_at(&gpu, w, 37);
        }
    }

    /// The body of [`chunked_readback_is_bit_identical`], run per geometry.
    /// Takes the caller's `GpuContext` — a second device init would deadlock
    /// against this crate's other GPU tests.
    #[allow(non_snake_case)]
    fn check_bit_identity_at(gpu: &crate::device::GpuContext, w: u32, h: u32) {
        let (W, H) = (w, h);
        let bytes_per_row = padded_bytes_per_row(W);

        // Deterministic, position-dependent content: every byte differs from
        // its neighbours, so a band pasted at the wrong offset (or a dropped /
        // duplicated row at a seam) cannot coincidentally compare equal.
        let mut src = vec![0u8; (W * H * 4) as usize];
        for y in 0..H as usize {
            for x in 0..W as usize {
                let i = (y * W as usize + x) * 4;
                src[i] = (x % 251) as u8;
                src[i + 1] = (y % 241) as u8;
                src[i + 2] = ((x * 7 + y * 13) % 239) as u8;
                src[i + 3] = 255;
            }
        }

        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("chunked readback test"),
            size: wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &src,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(W * 4),
                rows_per_image: Some(H),
            },
            wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
        );
        crate::scratch::flush();
        gpu.queue.submit(std::iter::empty());

        // Reference: one band (what this function did before chunking existed).
        let one_band = read_texture_rgba_staged(
            &gpu.device,
            &gpu.queue,
            &texture,
            W,
            H,
            READBACK_POLL_TIMEOUT_RETRIES,
            u64::MAX,
        )
        .expect("single-band readback");
        assert_eq!(one_band, src, "single-band readback must round-trip");

        // 5 rows per band => 8 bands over 37 rows, the last one short (2 rows).
        let budget = bytes_per_row as u64 * 5;
        assert_eq!(readback_rows_per_band(bytes_per_row, budget, H), 5);
        let many_bands = read_texture_rgba_staged(
            &gpu.device,
            &gpu.queue,
            &texture,
            W,
            H,
            READBACK_POLL_TIMEOUT_RETRIES,
            budget,
        )
        .expect("multi-band readback");

        assert_eq!(
            many_bands, one_band,
            "8-band readback must be bit-identical to the single-band readback"
        );

        // And the pathological floor: one row per band, 37 bands.
        let per_row = read_texture_rgba_staged(
            &gpu.device,
            &gpu.queue,
            &texture,
            W,
            H,
            READBACK_POLL_TIMEOUT_RETRIES,
            1,
        )
        .expect("one-row-per-band readback");
        assert_eq!(
            per_row, one_band,
            "one-row-per-band readback must be bit-identical too"
        );

        // Degenerate extents short-circuit to an empty image on BOTH axes rather
        // than asking wgpu for a zero-sized staging buffer. Shares this test's
        // device on purpose — a second GPU test means a second device init, and
        // parallel device init in this crate deadlocks.
        for (w, h) in [(0u32, H), (W, 0u32), (0u32, 0u32)] {
            let empty = read_texture_rgba_staged(
                &gpu.device,
                &gpu.queue,
                &texture,
                w,
                h,
                READBACK_POLL_TIMEOUT_RETRIES,
                u64::MAX,
            )
            .expect("zero-dimension readback returns empty, not a validation error");
            assert!(empty.is_empty(), "{w}x{h} readback must be empty");
        }

        for format in [wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::Bgra8Unorm] {
            let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("pipelined readback parity"),
                size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                format, usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let pool = ReadbackPool::default();
            let mut pending = std::collections::VecDeque::new();
            for frame in 0..7u8 {
                let mut bytes = src.clone();
                for pixel in bytes.chunks_exact_mut(4) {
                    pixel[0] = pixel[0].wrapping_add(frame * 17);
                    pixel[3] = frame * 37;
                }
                gpu.queue.write_texture(texture.as_image_copy(), &bytes,
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(H) },
                    wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 });
                let straight = frame % 2 == 0;
                let read = read_texture_rgba_start(&gpu.device, &gpu.queue, &texture, &pool, straight, READBACK_POLL_TIMEOUT_RETRIES).unwrap();
                if format == wgpu::TextureFormat::Bgra8Unorm {
                    bgra8_to_rgba8_in_place(&mut bytes);
                }
                if straight {
                    unpremultiply_in_place(&mut bytes);
                }
                pending.push_back((read, bytes));
                if pending.len() == 2 {
                    assert!(read_texture_rgba_start(&gpu.device, &gpu.queue, &texture, &pool, false, READBACK_POLL_TIMEOUT_RETRIES).is_err(), "a third staging buffer must not be allocated");
                    let (read, expected) = pending.pop_front().unwrap();
                    assert_eq!(read.finish(&gpu.device).unwrap(), expected, "frame {frame}: copy must precede the next upload");
                }
            }
            for (read, expected) in pending {
                assert_eq!(read.finish(&gpu.device).unwrap(), expected);
            }
            assert_eq!(pool.0.borrow().free.len(), 2, "both staging buffers are reusable");
            let cancelled = read_texture_rgba_start(&gpu.device, &gpu.queue, &texture, &pool, false, READBACK_POLL_TIMEOUT_RETRIES).unwrap();
            drop(cancelled);
            let retried = read_texture_rgba_start(&gpu.device, &gpu.queue, &texture, &pool, false, READBACK_POLL_TIMEOUT_RETRIES).unwrap();
            assert_eq!(retried.finish(&gpu.device).unwrap(), read_texture_rgba(&gpu.device, &gpu.queue, &texture, W, H).unwrap());
        }
    }
}
