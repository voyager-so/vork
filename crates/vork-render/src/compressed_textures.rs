//! GPU-ready BC7/ASTC mip cache. The codec owns compression; installation owns
//! device selection, cache validation and wgpu uploads.
use super::*;
use crate::constants::{TEXTURE_BLOCK_BYTES as BLOCK, TEXTURE_BLOCK_EDGE as EDGE};
use basisu_c_sys::{BasisTextureFormat, extra::*};

struct Job {
    path: std::path::PathBuf,
    texture: wgpu::Texture,
    pixels: Vec<u8>,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
    bytes: usize,
    preserve_alpha: bool,
}

struct Worker {
    tx: Option<std::sync::mpsc::Sender<Job>>,
    join: Option<std::thread::JoinHandle<Result<()>>>,
}

/// Stream cold BC7 preparation through a fixed-size in-flight ring while
/// GPU work overlaps the rest of scene installation.
#[derive(Default)]
pub(crate) struct Preparation {
    worker: std::cell::RefCell<Option<Worker>>,
    cpu: std::cell::RefCell<Option<CpuWorkers>>,
}
impl Preparation {
    pub(crate) fn finish(&self, r: &Renderer) -> Result<()> {
        if let Some(mut worker) = self.worker.borrow_mut().take() {
            drop(worker.tx.take());
            if let Some(join) = worker.join.take() {
                join.join()
                    .map_err(|panic| crate::failure::panicked("BC7 preparation worker", panic))??;
            }
        }
        if let Some(workers) = self.cpu.borrow_mut().take() {
            workers.finish()?;
        }
        r.errors.check()
    }
    #[allow(clippy::too_many_arguments)]
    fn push(
        &self,
        r: &Renderer,
        path: std::path::PathBuf,
        pixels: &mut Vec<u8>,
        w: u32,
        h: u32,
        levels: u32,
        format: wgpu::TextureFormat,
        bytes: usize,
        preserve_alpha: bool,
    ) -> Result<wgpu::TextureView> {
        let mut slot = self.worker.borrow_mut();
        if slot.is_none() {
            let ctx = crate::gpu_bc7::Context::new(r)?;
            let (tx, rx) = std::sync::mpsc::channel::<Job>();
            let join = std::thread::spawn(move || -> Result<()> {
                let mut scratch = crate::gpu_bc7::Scratch::default();
                let mut in_flight = std::collections::VecDeque::<(
                    std::path::PathBuf,
                    crate::gpu_bc7::Pending,
                )>::with_capacity(
                    crate::constants::TEXTURE_PREPARATION_IN_FLIGHT
                );
                while let Ok(job) = rx.recv() {
                    if in_flight.len() >= crate::constants::TEXTURE_PREPARATION_IN_FLIGHT {
                        let (path, oldest) = in_flight.pop_front().unwrap();
                        let (_, data) = oldest.finish(&ctx, &mut scratch)?;
                        let _ = disk_cache::write(&path, &data);
                    }
                    let pending = scratch.submit(
                        &ctx,
                        job.texture,
                        job.pixels,
                        job.w,
                        job.h,
                        job.format,
                        job.bytes,
                        job.preserve_alpha,
                    )?;
                    in_flight.push_back((job.path, pending));
                }
                while let Some((path, pending)) = in_flight.pop_front() {
                    let (_, data) = pending.finish(&ctx, &mut scratch)?;
                    let _ = disk_cache::write(&path, &data);
                }
                drop(scratch);
                crate::gpu_memory::wait(&ctx.device, &ctx.queue, &ctx.errors, None)
            });
            *slot = Some(Worker {
                tx: Some(tx),
                join: Some(join),
            });
        }
        let texture = crate::gpu_bc7::texture(&r.device, w, h, levels, format);
        r.errors.check()?;
        let view = texture.create_view(&Default::default());
        let job = Job {
            path,
            texture,
            pixels: std::mem::take(pixels),
            w,
            h,
            format,
            bytes,
            preserve_alpha,
        };
        let send_failed = slot
            .as_ref()
            .and_then(|w| w.tx.as_ref())
            .is_none_or(|tx| tx.send(job).is_err());
        if send_failed {
            let mut worker = slot.take().unwrap();
            drop(worker.tx.take());
            if let Some(join) = worker.join.take() {
                join.join()
                    .map_err(|panic| crate::failure::panicked("BC7 preparation worker", panic))??;
            }
            bail!("BC7 preparation worker exited unexpectedly");
        }
        Ok(view)
    }
}
impl Drop for Preparation {
    fn drop(&mut self) {
        if let Some(workers) = self.cpu.get_mut().take() {
            let _ = workers.finish();
        }
        if let Some(mut worker) = self.worker.get_mut().take() {
            drop(worker.tx.take());
            if let Some(join) = worker.join.take() {
                let _ = join.join();
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn load_or_encode(
    r: &Renderer,
    preparation: &Preparation,
    hash: blake3::Hash,
    srgb: bool,
    preserve_alpha: bool,
    w: u32,
    h: u32,
    pixels: Option<&mut Vec<u8>>,
) -> Result<Option<wgpu::TextureView>> {
    // wgpu requires block-aligned base dimensions. Preserve authored UVs and
    // filtering by retaining the RGBA path for other sizes and fallback texels.
    if w % EDGE != 0 || h % EDGE != 0 {
        return Ok(None);
    }
    let astc = r
        .device
        .features()
        .contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC);
    let bc = r
        .device
        .features()
        .contains(wgpu::Features::TEXTURE_COMPRESSION_BC);
    // Prefer ASTC when the device can sample it; BC7 remains the fallback.
    let format = if astc {
        wgpu::TextureFormat::Astc {
            block: wgpu::AstcBlock::B4x4,
            channel: if srgb {
                wgpu::AstcChannel::UnormSrgb
            } else {
                wgpu::AstcChannel::Unorm
            },
        }
    } else if bc {
        if srgb {
            wgpu::TextureFormat::Bc7RgbaUnormSrgb
        } else {
            wgpu::TextureFormat::Bc7RgbaUnorm
        }
    } else {
        return Ok(None);
    };
    let levels = 32 - w.max(h).leading_zeros();
    let level_bytes = |mip: u32| {
        let mw = (w >> mip).max(1);
        let mh = (h >> mip).max(1);
        (mw.div_ceil(EDGE) * mh.div_ceil(EDGE) * BLOCK) as usize
    };
    let expected: usize = (0..levels).map(level_bytes).sum();
    // Hardware BC7 refines sampled alpha with SIMD; unsupported adapters keep
    // the native fallback. Encoder/channel requirements remain in the one key.
    let gpu_bc7 = !matches!(format, wgpu::TextureFormat::Astc { .. })
        && crate::gpu_bc7::supported(r, w, h, expected);
    let mut key = blake3::Hasher::new();
    key.update(&[u8::from(gpu_bc7)]);
    if !gpu_bc7 {
        let native_bc7 =
            cfg!(target_arch = "x86_64") && !matches!(format, wgpu::TextureFormat::Astc { .. });
        key.update(if native_bc7 {
            b"intel-bc7"
        } else {
            b"basis-astc-bc7"
        });
    }
    if gpu_bc7 {
        key.update(b"alpha-edge-v3");
        key.update(&[u8::from(preserve_alpha)]);
    }

    key.update(hash.as_bytes());
    key.update(format!("{format:?}").as_bytes());
    key.update(&w.to_le_bytes());
    key.update(&h.to_le_bytes());
    key.update(&crate::constants::TEXTURE_COMPRESSION_QUALITY.to_le_bytes());
    key.update(&crate::constants::TEXTURE_COMPRESSION_EFFORT.to_le_bytes());
    let path = r.disk_cache.path("compressed-texture", 2, key.finalize());
    let cached = disk_cache::read_mapped(&path).filter(|bytes| bytes.len() == expected);
    let fresh;
    let data: &[u8] = if let Some(ref bytes) = cached {
        bytes
    } else {
        let Some(pixels) = pixels else {
            return Ok(None);
        };
        if gpu_bc7 {
            return preparation
                .push(
                    r,
                    path,
                    pixels,
                    w,
                    h,
                    levels,
                    format,
                    expected,
                    preserve_alpha,
                )
                .map(Some);
        }
        #[cfg(target_arch = "x86_64")]
        let native = (!matches!(format, wgpu::TextureFormat::Astc { .. }))
            .then(|| native_bc7(pixels.as_slice(), w, h, srgb));
        #[cfg(not(target_arch = "x86_64"))]
        let native: Option<Vec<u8>> = None;
        let data = if let Some(data) = native {
            data
        } else {
            return preparation
                .push_cpu(
                    r,
                    path,
                    std::mem::take(pixels),
                    w,
                    h,
                    levels,
                    format,
                    expected,
                )
                .map(Some);
        };
        ensure!(
            data.len() == expected,
            "native compressed texture mip layout mismatch"
        );
        let _ = disk_cache::write(&path, &data);
        fresh = data;
        &fresh
    };
    let texture = crate::gpu_memory::texture(
        &r.device,
        &wgpu::TextureDescriptor {
            label: Some("cached compressed material mips"),
            size: extent(w, h),
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    );
    upload_mips(&r.queue, &texture, data, w, h, levels);
    Ok(Some(texture.create_view(&Default::default())))
}

/// The SIMD RGBA search may trade opacity for RGB through shared p-bits in
/// modes 6/7. Opaque blocks use RGB-only modes 1/3, whose alpha is exactly 255.
#[cfg(target_arch = "x86_64")]
fn bc7_settings(opaque: bool) -> intel_tex_2::bc7::EncodeSettings {
    if opaque {
        let mut settings = intel_tex_2::bc7::opaque_very_fast_settings();
        settings.mode_selection[3] = false; // mode 6 also loses alpha in RGB mode
        settings
    } else {
        intel_tex_2::bc7::alpha_very_fast_settings()
    }
}

#[cfg(target_arch = "x86_64")]
fn encode_bc7_level(input: &[u8], w: u32, h: u32) -> Vec<u8> {
    let opaque = input.chunks_exact(4).all(|p| p[3] == 255);
    let mut blocks = intel_tex_2::bc7::compress_blocks(
        &bc7_settings(opaque),
        &intel_tex_2::RgbaSurface {
            data: input,
            width: w,
            height: h,
            stride: w * 4,
        },
    );
    if !opaque {
        encode_bc7_selected_blocks(input, w, h, &mut blocks, true);
    }
    blocks
}

/// SIMD BC7 avoids the UASTC intermediate on NVIDIA hosts. Mips use linear
/// radiance for sRGB and bilinear sampling at the same texel centres as mip.slang.
#[cfg(target_arch = "x86_64")]
fn native_bc7(pixels: &[u8], mut w: u32, mut h: u32, srgb: bool) -> Vec<u8> {
    // Solid authored images still upload at full resolution. Every block and
    // every filtered mip has the same texels, so run the identical codec once.
    if pixels.chunks_exact(4).all(|pixel| pixel == &pixels[..4]) {
        let texels: [u8; (EDGE * EDGE * 4) as usize] = std::array::from_fn(|i| pixels[i % 4]);
        let block = encode_bc7_level(&texels, EDGE, EDGE);
        let mut result = Vec::new();
        loop {
            let blocks = (w.div_ceil(EDGE) * h.div_ceil(EDGE)) as usize;
            result.extend_from_slice(&block.repeat(blocks));
            if w == 1 && h == 1 {
                return result;
            }
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
    }
    let mut result = Vec::new();
    visit_native_mips(pixels, w, h, srgb, |blocks_input, pw, ph| {
        let workers = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min((ph / EDGE) as usize);
        let rows = ph.div_ceil(workers as u32 * EDGE) * EDGE;
        let blocks = std::thread::scope(|scope| {
            let handles: Vec<_> = blocks_input
                .chunks((pw * rows * 4) as usize)
                .map(|data| {
                    scope.spawn(move || encode_bc7_level(data, pw, data.len() as u32 / (pw * 4)))
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("BC7 encoder panicked"))
                .collect::<Vec<_>>()
        });
        result.extend_from_slice(&blocks);
    });
    result
}

/// Correct alpha edges with the default SIMD search. Fully transparent and
/// opaque blocks stay on the GPU; no second cache or texture representation.
#[cfg(target_arch = "x86_64")]
pub(crate) fn refine_bc7_alpha_level(input: &[u8], w: u32, h: u32, encoded: &mut [u8]) -> bool {
    encode_bc7_selected_blocks(input, w, h, encoded, false)
}

#[cfg(target_arch = "x86_64")]
fn encode_bc7_selected_blocks(
    input: &[u8],
    w: u32,
    h: u32,
    encoded: &mut [u8],
    opaque: bool,
) -> bool {
    if !opaque && input.chunks_exact(4).all(|p| p[3] == input[3]) {
        return false;
    }
    let (pw, ph) = (w.div_ceil(EDGE) * EDGE, h.div_ceil(EDGE) * EDGE);
    let mut changed = false;
    let mut texels = Vec::new();
    let mut indices = Vec::new();
    for by in 0..ph / EDGE {
        for bx in 0..pw / EDGE {
            let at = |x: u32, y: u32| ((y.min(h - 1) * w + x.min(w - 1)) * 4) as usize;
            let base = at(bx * EDGE, by * EDGE);
            let alpha = input[base + 3];
            let varying = (0..EDGE)
                .any(|y| (0..EDGE).any(|x| input[at(bx * EDGE + x, by * EDGE + y) + 3] != alpha));
            if if opaque {
                alpha == 255 && !varying
            } else {
                varying
            } {
                indices.push((by * (pw / EDGE) + bx) as usize);
                for y in 0..EDGE {
                    for x in 0..EDGE {
                        let source = at(bx * EDGE + x, by * EDGE + y);
                        texels.extend_from_slice(&input[source..source + 4]);
                    }
                }
            }
        }
    }
    // Lay the gathered blocks side by side so the SIMD lanes stay full.
    if !indices.is_empty() {
        changed = true;
        let workers = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(
                indices
                    .len()
                    .div_ceil(crate::constants::TEXTURE_BC7_ALPHA_BLOCKS_PER_WORKER),
            );
        let blocks_per_worker = indices.len().div_ceil(workers);
        let encode = |texels: &[u8]| {
            let count = texels.len() / (EDGE * EDGE * 4) as usize;
            let mut packed = Vec::with_capacity(texels.len());
            for y in 0..EDGE as usize {
                for block in texels.chunks_exact((EDGE * EDGE * 4) as usize) {
                    packed.extend_from_slice(
                        &block[y * EDGE as usize * 4..(y + 1) * EDGE as usize * 4],
                    );
                }
            }
            intel_tex_2::bc7::compress_blocks(
                &bc7_settings(opaque),
                &intel_tex_2::RgbaSurface {
                    data: &packed,
                    width: count as u32 * EDGE,
                    height: EDGE,
                    stride: count as u32 * EDGE * 4,
                },
            )
        };
        let blocks = if workers == 1 {
            encode(&texels)
        } else {
            std::thread::scope(|scope| {
                let handles: Vec<_> = texels
                    .chunks(blocks_per_worker * (EDGE * EDGE * 4) as usize)
                    .map(|chunk| scope.spawn(move || encode(chunk)))
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|handle| handle.join().expect("BC7 alpha encoder panicked"))
                    .collect::<Vec<_>>()
            })
        };
        for (i, block) in indices.into_iter().zip(blocks.chunks_exact(BLOCK as usize)) {
            let dst = i * BLOCK as usize;
            encoded[dst..dst + BLOCK as usize].copy_from_slice(block);
        }
    }
    changed
}

#[cfg(target_arch = "x86_64")]
fn visit_native_mips(
    pixels: &[u8],
    mut w: u32,
    mut h: u32,
    srgb: bool,
    mut visit: impl FnMut(&[u8], u32, u32),
) {
    let lut: [f32; 256] = std::array::from_fn(|i| {
        let v = i as f32 / 255.;
        if srgb {
            crate::texture_source::decode_srgb(v)
        } else {
            v
        }
    });
    let mut pixels = std::borrow::Cow::Borrowed(pixels);
    loop {
        let pw = w.div_ceil(EDGE) * EDGE;
        let ph = h.div_ceil(EDGE) * EDGE;
        let padded;
        let blocks_input = if pw == w && ph == h {
            pixels.as_ref()
        } else {
            padded = (0..ph)
                .flat_map(|y| {
                    let pixels = &pixels;
                    (0..pw).flat_map(move |x| {
                        let source = ((y.min(h - 1) * w + x.min(w - 1)) * 4) as usize;
                        pixels[source..source + 4].iter().copied()
                    })
                })
                .collect::<Vec<_>>();
            &padded
        };
        visit(blocks_input, pw, ph);
        if w == 1 && h == 1 {
            break;
        }
        let nw = (w / 2).max(1);
        let nh = (h / 2).max(1);
        let mut next = vec![0; (nw * nh * 4) as usize];
        let mip_workers = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(nh as usize)
            .min(
                ((nw as usize * nh as usize) / crate::constants::TEXTURE_MIP_TEXELS_PER_WORKER)
                    .max(1),
            );
        let rows_per_worker = nh.div_ceil(mip_workers as u32);
        std::thread::scope(|scope| {
            for (chunk, output) in next
                .chunks_mut((nw * rows_per_worker * 4) as usize)
                .enumerate()
            {
                let pixels = &pixels;
                let lut = &lut;
                let mut generate = move || {
                    for (local_y, row) in output.chunks_mut((nw * 4) as usize).enumerate() {
                        let y = chunk as u32 * rows_per_worker + local_y as u32;
                        for x in 0..nw {
                            let sx = ((x as f32 + 0.5) * w as f32 / nw as f32 - 0.5).max(0.);
                            let sy = ((y as f32 + 0.5) * h as f32 / nh as f32 - 0.5).max(0.);
                            let x0 = sx as u32;
                            let y0 = sy as u32;
                            let fx = sx.fract();
                            let fy = sy.fract();
                            for c in 0..4 {
                                let sample = |dx: u32, dy: u32| {
                                    let v = pixels[(((y0 + dy).min(h - 1) * w
                                        + (x0 + dx).min(w - 1))
                                        * 4)
                                        as usize
                                        + c];
                                    if c == 3 {
                                        v as f32 / 255.
                                    } else {
                                        lut[v as usize]
                                    }
                                };
                                let a = sample(0, 0) * (1. - fx) + sample(1, 0) * fx;
                                let b = sample(0, 1) * (1. - fx) + sample(1, 1) * fx;
                                let v = a * (1. - fy) + b * fy;
                                row[(x * 4) as usize + c] = if c == 3 || !srgb {
                                    (v * 255.).round() as u8
                                } else {
                                    let upper = lut.partition_point(|entry| *entry < v).min(255);
                                    let lower = upper.saturating_sub(1);
                                    if (v - lut[lower]).abs() < (lut[upper] - v).abs() {
                                        lower as u8
                                    } else {
                                        upper as u8
                                    }
                                };
                            }
                        }
                    }
                };
                if mip_workers == 1 {
                    generate();
                } else {
                    scope.spawn(generate);
                }
            }
        });
        pixels = std::borrow::Cow::Owned(next);
        w = nw;
        h = nh;
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn alpha_refinement_matches_default_blocks_and_preserves_others() {
        for (w, h) in [(4, 4), (12, 28), (64, 32), (512, 256)] {
            let pixels: Vec<u8> = (0..h)
                .flat_map(|y| {
                    (0..w).flat_map(move |x| {
                        [
                            (x * 17 + y * 13) as u8,
                            (x * 3 + y * 29) as u8,
                            (x * 7 + y * 11) as u8,
                            if x < 4 { 255 } else { ((x + y) * 23) as u8 },
                        ]
                    })
                })
                .collect();
            for srgb in [false, true] {
                let full = native_bc7(&pixels, w, h, srgb);
                let mut refined = vec![0; full.len()];

                let mut offset = 0;
                visit_native_mips(&pixels, w, h, srgb, |input, pw, ph| {
                    let length = (pw / EDGE * (ph / EDGE) * BLOCK) as usize;
                    refine_bc7_alpha_level(input, pw, ph, &mut refined[offset..offset + length]);
                    for by in 0..ph / EDGE {
                        for bx in 0..pw / EDGE {
                            let base = ((by * EDGE * pw + bx * EDGE) * 4) as usize;
                            let varying = (0..EDGE).any(|y| {
                                (0..EDGE).any(|x| {
                                    input[base + ((y * pw + x) * 4) as usize + 3] != input[base + 3]
                                })
                            });
                            let i = offset + ((by * (pw / EDGE) + bx) * BLOCK) as usize;
                            let actual = &refined[i..i + BLOCK as usize];
                            if varying {
                                assert_eq!(
                                    actual,
                                    &full[i..i + BLOCK as usize],
                                    "{w}x{h}, sRGB={srgb}"
                                );
                            } else {
                                assert_eq!(actual, &[0; BLOCK as usize]);
                            }
                        }
                    }
                    offset += (pw / EDGE * (ph / EDGE) * BLOCK) as usize;
                });
            }
        }
    }

    #[test]
    fn alpha_refinement_clamps_partial_blocks() {
        for (w, h) in [(3u32, 7u32), (7, 3), (1, 7), (7, 1)] {
            let pixels: Vec<_> = (0..w * h)
                .flat_map(|i| [i as u8, (i * 7) as u8, 123, (i * 19) as u8])
                .collect();
            let (pw, ph) = (w.div_ceil(EDGE) * EDGE, h.div_ceil(EDGE) * EDGE);
            let mut padded = Vec::new();
            for y in 0..ph {
                for x in 0..pw {
                    let at = ((y.min(h - 1) * w + x.min(w - 1)) * 4) as usize;
                    padded.extend_from_slice(&pixels[at..at + 4]);
                }
            }
            let expected = intel_tex_2::bc7::compress_blocks(
                &intel_tex_2::bc7::alpha_very_fast_settings(),
                &intel_tex_2::RgbaSurface {
                    data: &padded,
                    width: pw,
                    height: ph,
                    stride: pw * 4,
                },
            );
            let mut actual = vec![0; expected.len()];
            assert!(refine_bc7_alpha_level(&pixels, w, h, &mut actual));
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn solid_shortcut_matches_full_simd_encoding_for_every_mip() {
        for (width, height) in [(4u32, 4u32), (12, 28), (64, 32)] {
            for color in [[0, 0, 0, 255], [128, 128, 255, 255], [73, 19, 42, 91]] {
                let pixels = color.repeat((width * height) as usize);
                let mut expected = Vec::new();
                let (mut w, mut h) = (width, height);
                loop {
                    let pw = w.div_ceil(EDGE) * EDGE;
                    let ph = h.div_ceil(EDGE) * EDGE;
                    let full_surface = color.repeat((pw * ph) as usize);
                    expected.extend(intel_tex_2::bc7::compress_blocks(
                        &bc7_settings(color[3] == 255),
                        &intel_tex_2::RgbaSurface {
                            data: &full_surface,
                            width: pw,
                            height: ph,
                            stride: pw * 4,
                        },
                    ));
                    if w == 1 && h == 1 {
                        break;
                    }
                    w = (w / 2).max(1);
                    h = (h / 2).max(1);
                }
                for srgb in [false, true] {
                    assert_eq!(native_bc7(&pixels, width, height, srgb), expected);
                }
            }
        }
    }
}

fn upload_mips(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    data: &[u8],
    w: u32,
    h: u32,
    levels: u32,
) {
    let mut offset = 0;
    for mip in 0..levels {
        let mw = (w >> mip).max(1).div_ceil(EDGE) * EDGE;
        let mh = (h >> mip).max(1).div_ceil(EDGE) * EDGE;
        let length = (mw / EDGE * (mh / EDGE) * BLOCK) as usize;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data[offset..offset + length],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(mw / EDGE * BLOCK),
                rows_per_image: None,
            },
            extent(mw, mh),
        );
        offset += length;
    }
}

/// The baseline is retained for CPU-only byte-equivalence and corpus benchmarks.
fn encode_basis(
    pixels: &[u8],
    w: u32,
    h: u32,
    srgb: bool,
    astc: bool,
    optimized: bool,
) -> Result<Vec<u8>> {
    let encoded = basis_container(
        pixels,
        w,
        h,
        srgb,
        optimized,
        BasisTextureFormat::UastcLdr4x4,
    )?;
    let supported = if astc {
        SupportedTextureCompression::ASTC_LDR
    } else {
        SupportedTextureCompression::BC
    };
    let image = BasisuTranscoder::new(&encoded, supported, ChannelType::Rgba)?
        .transcode(None, Some(srgb))?;
    ensure!(
        image.mip_level_count == 32 - w.max(h).leading_zeros(),
        "compressed texture mip layout mismatch"
    );
    Ok(image.data)
}

struct CpuWorkers {
    tx: std::sync::mpsc::SyncSender<Job>,
    joins: Vec<std::thread::JoinHandle<Result<()>>>,
}
impl CpuWorkers {
    fn finish(self) -> Result<()> {
        drop(self.tx);
        let mut result = Ok(());
        for join in self.joins {
            let outcome = join
                .join()
                .map_err(|panic| crate::failure::panicked("CPU texture preparation worker", panic))
                .and_then(|r| r);
            if result.is_ok() {
                result = outcome;
            }
        }
        result
    }
}
impl Preparation {
    #[allow(clippy::too_many_arguments)]
    fn push_cpu(
        &self,
        r: &Renderer,
        path: std::path::PathBuf,
        pixels: Vec<u8>,
        w: u32,
        h: u32,
        levels: u32,
        format: wgpu::TextureFormat,
        bytes: usize,
    ) -> Result<wgpu::TextureView> {
        let mut slot = self.cpu.borrow_mut();
        let workers = slot.get_or_insert_with(|| {
            let count = std::thread::available_parallelism().map_or(1, usize::from);
            let (tx, rx) = std::sync::mpsc::sync_channel::<Job>(count);
            let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
            let joins = (0..count)
                .map(|_| {
                    let rx = rx.clone();
                    let queue = r.queue.clone();
                    std::thread::spawn(move || {
                        loop {
                            let job = { rx.lock().unwrap().recv() };
                            let Ok(job) = job else { break };
                            let data = encode_basis(
                                &job.pixels,
                                job.w,
                                job.h,
                                job.format.is_srgb(),
                                matches!(job.format, wgpu::TextureFormat::Astc { .. }),
                                true,
                            )?;
                            ensure!(
                                data.len() == job.bytes,
                                "compressed texture mip layout mismatch"
                            );
                            upload_mips(
                                &queue,
                                &job.texture,
                                &data,
                                job.w,
                                job.h,
                                32 - job.w.max(job.h).leading_zeros(),
                            );
                            let _ = disk_cache::write(&job.path, &data);
                        }
                        Ok(())
                    })
                })
                .collect();
            CpuWorkers { tx, joins }
        });
        let texture = crate::gpu_bc7::texture(&r.device, w, h, levels, format);
        let view = texture.create_view(&Default::default());
        workers
            .tx
            .send(Job {
                path,
                texture,
                pixels,
                w,
                h,
                format,
                bytes,
                preserve_alpha: true,
            })
            .map_err(|_| anyhow::anyhow!("CPU texture preparation workers exited"))?;
        Ok(view)
    }
}

#[cfg(test)]
#[path = "compressed_textures_tests.rs"]
mod astc_tests;

fn basis_container(
    pixels: &[u8],
    w: u32,
    h: u32,
    srgb: bool,
    optimized: bool,
    codec: BasisTextureFormat,
) -> Result<Vec<u8>> {
    basisu_encoder_init();
    basisu_transcoder_init();
    let mut encoder = BasisuEncoder::new();
    encoder.set_image(SourceImage {
        data: pixels,
        format: SourceImageFormat::Rgba8,
        size: types::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    })?;
    let mut params = if srgb {
        BasisuEncoderParams::new_with_srgb_defaults(codec)
    } else {
        BasisuEncoderParams::new_with_linear_defaults(codec)
    };
    params.quality_level = crate::constants::TEXTURE_COMPRESSION_QUALITY;
    params.effort_level = crate::constants::TEXTURE_COMPRESSION_EFFORT;
    if optimized {
        params = params
            .with_flags(
                basisu_c_sys::common::BU_COMP_FLAGS_NO_STATS
                    | basisu_c_sys::common::BU_COMP_FLAGS_NO_UASTC_HINTS,
            )
            .with_removed_flags(
                basisu_c_sys::common::BU_COMP_FLAGS_THREADED
                    | basisu_c_sys::common::BU_COMP_FLAGS_KTX2_UASTC_ZSTD,
            );
    }
    params = params.with_flags(basisu_c_sys::common::BU_COMP_FLAGS_GEN_MIPS_CLAMP as u64);
    Ok(encoder.compress(params)?)
}
