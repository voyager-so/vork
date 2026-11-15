//! nv12.rs — GPU RGBA→NV12 conversion at the export readback seam
//! (export-perf experiment: GPU color conversion for the raw-frame stream).
//!
//! Converts the finished frame texture to NV12 (BT.709 matrix, LIMITED/tv
//! range, 4:2:0 with box-averaged center-sited chroma) in a compute pass and
//! reads back 1.5 bytes/px instead of 4. The math mirrors what the media
//! executor's ffmpeg does on the RGBA stream (`mp4_video_filter`'s
//! white-composite + `out_color_matrix=bt709:out_range=tv`):
//!
//! - White composite first: the texture holds PREMULTIPLIED alpha, so
//!   `rgb + (1 - a)` is over-white compositing; opaque pixels (every mp4
//!   export frame) are the identity.
//! - Y' = 16 + 219·(0.2126 R' + 0.7152 G' + 0.0722 B'), gamma-encoded values,
//!   round-to-nearest — measured within ±1 LSB of swscale (which carries a
//!   ~+0.5 fixed-point bias).
//! - Cb/Cr per 2×2 block from the box-averaged composited RGB:
//!   Cb = 128 + 224·(B'−Y')/1.8556, Cr = 128 + 224·(R'−Y')/1.5748.
//!   Box average is swscale's `area` kernel with center siting — measured
//!   byte-exact (±1) against `flags=area+accurate_rnd`; the production
//!   `lanczos` flag differs only in the downsample kernel on high-frequency
//!   content (mean delta ~0, no siting or range shift).
//!
//! Dimensions must be EVEN on both axes (4:2:0 has no half-block); callers
//! route odd-sized frames to the RGBA lane instead.

use anyhow::{bail, Context, Result};

const SHADER: &str = r#"
struct Params {
    width: u32,
    height: u32,
    // ceil(width / 4): u32 words per output row, shared by the Y plane and
    // the interleaved UV plane (both have `width` BYTES per row).
    words_per_row: u32,
    // words_per_row * height: where the UV plane starts in `out`.
    uv_offset_words: u32,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

// Premultiplied-over-white composite (mp4_video_filter's _WHITE_COMPOSITE on
// the straight-alpha stream computes the same value); opaque = identity.
// i32 coords and a literal-0 mip level, like every other textureLoad in this
// repo: naga's GLSL backend cannot emit a uint mip level (its bounds-clamp
// textureSize call has no uint overload), and the CI runner's adapter is GL.
fn composited_rgb(x: u32, y: u32) -> vec3<f32> {
    let cx = min(x, params.width - 1u);
    let cy = min(y, params.height - 1u);
    let p = textureLoad(src, vec2<i32>(i32(cx), i32(cy)), 0);
    return p.rgb + vec3<f32>(1.0 - p.a);
}

fn luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn quant_y(ey: f32) -> u32 {
    return u32(clamp(round(16.0 + 219.0 * ey), 0.0, 255.0));
}

// One thread = one 4x2 pixel block: two Y words (4 bytes each) and one UV
// word (Cb0 Cr0 Cb1 Cr1 — the NV12 interleave for the block's two 2x2s).
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_x = gid.x;
    let row_pair = gid.y;
    if (word_x >= params.words_per_row || row_pair * 2u >= params.height) {
        return;
    }
    let x0 = word_x * 4u;
    let y0 = row_pair * 2u;

    var rgb: array<vec3<f32>, 8>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        rgb[i] = composited_rgb(x0 + i, y0);
        rgb[4u + i] = composited_rgb(x0 + i, y0 + 1u);
    }

    var yw0 = 0u;
    var yw1 = 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        yw0 = yw0 | (quant_y(luma(rgb[i])) << (8u * i));
        yw1 = yw1 | (quant_y(luma(rgb[4u + i])) << (8u * i));
    }
    out[y0 * params.words_per_row + word_x] = yw0;
    out[(y0 + 1u) * params.words_per_row + word_x] = yw1;

    var uvw = 0u;
    for (var b = 0u; b < 2u; b = b + 1u) {
        let avg = (rgb[2u * b] + rgb[2u * b + 1u] + rgb[4u + 2u * b] + rgb[4u + 2u * b + 1u]) * 0.25;
        let ey = luma(avg);
        let cb = u32(clamp(round(128.0 + 224.0 * (avg.b - ey) / 1.8556), 0.0, 255.0));
        let cr = u32(clamp(round(128.0 + 224.0 * (avg.r - ey) / 1.5748), 0.0, 255.0));
        uvw = uvw | (cb << (16u * b)) | (cr << (16u * b + 8u));
    }
    out[params.uv_offset_words + row_pair * params.words_per_row + word_x] = uvw;
}
"#;

/// Number of NV12 bytes for a `width`×`height` frame (Y plane + interleaved
/// UV plane). Both dimensions must be even.
pub fn nv12_frame_bytes(width: u32, height: u32) -> usize {
    width as usize * height as usize * 3 / 2
}

/// Size of the PADDED storage buffer the compute pass binds for a
/// `width`×`height` frame (rows are u32-word padded; see the shader's
/// `words_per_row`). This — not [`nv12_frame_bytes`] — is what must fit the
/// device's `max_storage_buffer_binding_size`: the export scale caps consider
/// `max_buffer_size` for the RGBA readback, but the storage-BINDING limit is
/// a separate, typically much smaller ceiling (128 MiB by default), so
/// eligibility must check it explicitly or a large admitted export dies of
/// GPU validation at bind time instead of falling back to rgba.
pub fn nv12_storage_bytes(width: u32, height: u32) -> u64 {
    let words_per_row = width.div_ceil(4) as u64;
    words_per_row * (height as u64 + (height / 2) as u64) * 4
}

/// Per-geometry GPU buffers, created on the first frame and reused for every
/// subsequent frame of the export (the dimensions are constant across a
/// stream, so a per-frame `create_buffer` triple would be pure driver churn —
/// the same reuse rationale as the readback path's single staging buffer).
/// The uniform's contents are a function of the geometry alone, so it is
/// written once here.
struct Nv12Buffers {
    width: u32,
    height: u32,
    total_bytes: u64,
    storage: wgpu::Buffer,
    staging: wgpu::Buffer,
    uniform: wgpu::Buffer,
}

/// The compute pipeline for RGBA→NV12 conversion, built once per device and
/// reused across every frame of an export. Not `Sync` (interior mutability
/// for the per-geometry buffer cache) — one converter per export loop, like
/// the engine itself.
pub struct Nv12Converter {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    buffers: std::cell::RefCell<Option<Nv12Buffers>>,
}

impl Nv12Converter {
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nv12 convert"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nv12 convert"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nv12 convert"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = moda_text_draw::device::create_compute_pipeline(device, &wgpu::ComputePipelineDescriptor {
            label: Some("nv12 convert"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            pipeline,
            layout,
            buffers: std::cell::RefCell::new(None),
        })
    }

    /// Convert a finished frame `texture` (rgba8unorm/bgra8unorm, sample
    /// count 1, TEXTURE_BINDING usage) to tightly-packed NV12 bytes.
    pub fn convert(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
        poll_retries: u32,
    ) -> Result<Vec<u8>> {
        if width == 0 || height == 0 {
            bail!("nv12 conversion: zero-sized frame ({width}x{height})");
        }
        if width % 2 != 0 || height % 2 != 0 {
            bail!("nv12 conversion requires even dimensions, got {width}x{height}");
        }
        let words_per_row = width.div_ceil(4);
        let y_words = words_per_row as u64 * height as u64;
        let uv_words = words_per_row as u64 * (height / 2) as u64;
        let total_bytes = (y_words + uv_words) * 4;
        // Belt to the caller's eligibility check (`nv12_ineligibility` routes
        // over-limit frames to the rgba lane): binding an oversized storage
        // buffer is a GPU validation PANIC, so it must never get that far.
        let binding_limit = u64::from(device.limits().max_storage_buffer_binding_size);
        if total_bytes > binding_limit {
            bail!(
                "nv12 conversion: {width}x{height} needs a {total_bytes}-byte storage binding, \
                 over this device's {binding_limit}-byte limit"
            );
        }

        // Reuse the buffer triple across the export's frames; (re)create only
        // when the geometry changes (in practice: once, on the first frame).
        let mut cache = self.buffers.borrow_mut();
        let cache_hit = matches!(cache.as_ref(), Some(b) if b.width == width && b.height == height);
        if !cache_hit {
            let storage = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nv12 plane buffer"),
                size: total_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nv12 read buffer"),
                size: total_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let params: [u32; 4] = [width, height, words_per_row, y_words as u32];
            let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nv12 params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // The params are a function of the geometry alone: written once
            // per (re)creation, never per frame.
            queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&params));
            *cache = Some(Nv12Buffers {
                width,
                height,
                total_bytes,
                storage,
                staging,
                uniform,
            });
        }
        let buffers = cache.as_ref().expect("ensured above");
        debug_assert_eq!(buffers.total_bytes, total_bytes);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nv12 convert"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.storage.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.uniform.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nv12 convert"),
        });
        {
            let profile_ts = crate::gpu_profile::pass_timestamps();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nv12 convert"),
                timestamp_writes: crate::gpu_profile::compute_writes(&profile_ts),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(words_per_row.div_ceil(8), (height / 2).div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(&buffers.storage, 0, &buffers.staging, 0, total_bytes);
        crate::scratch::flush();
        queue.submit([encoder.finish()]);

        let slice = buffers.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        crate::readback::poll_until_complete_with_budget(device, poll_retries)
            .context("device poll during nv12 readback")?;
        rx.recv().context("nv12 map_async channel")??;
        let data = slice.get_mapped_range();

        let out = compact_nv12(&data, width, height, words_per_row);
        drop(data);
        // Unmap before the next frame re-maps the same buffer.
        buffers.staging.unmap();
        Ok(out)
    }
}

/// Strip the per-row u32 padding (present only when `width % 4 != 0`) so the
/// output is the tightly-packed NV12 layout the rawvideo demuxer expects.
fn compact_nv12(padded: &[u8], width: u32, height: u32, words_per_row: u32) -> Vec<u8> {
    let stride = words_per_row as usize * 4;
    let row = width as usize;
    let uv_rows = (height / 2) as usize;
    if stride == row {
        return padded[..row * (height as usize + uv_rows)].to_vec();
    }
    let mut out = Vec::with_capacity(nv12_frame_bytes(width, height));
    for r in 0..height as usize + uv_rows {
        out.extend_from_slice(&padded[r * stride..r * stride + row]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_bytes_is_one_and_a_half_per_pixel() {
        assert_eq!(nv12_frame_bytes(1920, 1080), 1920 * 1080 * 3 / 2);
        assert_eq!(nv12_frame_bytes(2, 2), 6);
    }

    #[test]
    fn compact_strips_row_padding_only_when_present() {
        // width 4: no padding, pass-through of exactly the frame bytes.
        let padded: Vec<u8> = (0..12).collect();
        assert_eq!(compact_nv12(&padded, 4, 2, 1), padded);
        // width 6: words_per_row 2 (stride 8), 2 pad bytes per row stripped.
        // rows: 2 Y rows + 1 UV row.
        let mut padded = Vec::new();
        for r in 0..3u8 {
            padded.extend_from_slice(&[r; 6]);
            padded.extend_from_slice(&[0xFE, 0xFE]);
        }
        let out = compact_nv12(&padded, 6, 2, 2);
        assert_eq!(out, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]);
        assert!(!out.contains(&0xFE));
    }

    /// CPU reference of the shader's math — float BT.709/limited, white
    /// composite over premultiplied alpha, box-averaged chroma. The GPU test
    /// below holds the compute pass to this within ±1 LSB (GPU `round()` is
    /// round-to-even and FMA contraction may differ by an ulp; both can move
    /// a value across a rounding boundary, never further).
    fn cpu_reference_nv12(rgba_premul: &[u8], w: usize, h: usize) -> Vec<u8> {
        let px = |x: usize, y: usize| -> [f64; 3] {
            let i = (y * w + x) * 4;
            let a = rgba_premul[i + 3] as f64 / 255.0;
            [
                rgba_premul[i] as f64 / 255.0 + (1.0 - a),
                rgba_premul[i + 1] as f64 / 255.0 + (1.0 - a),
                rgba_premul[i + 2] as f64 / 255.0 + (1.0 - a),
            ]
        };
        let luma = |c: [f64; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let q = |v: f64| v.round().clamp(0.0, 255.0) as u8;
        let mut out = vec![0u8; w * h * 3 / 2];
        for y in 0..h {
            for x in 0..w {
                out[y * w + x] = q(16.0 + 219.0 * luma(px(x, y)));
            }
        }
        for by in 0..h / 2 {
            for bx in 0..w / 2 {
                let (x, y) = (bx * 2, by * 2);
                let mut avg = [0.0f64; 3];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let c = px(x + dx, y + dy);
                    for ch in 0..3 {
                        avg[ch] += c[ch] * 0.25;
                    }
                }
                let ey = luma(avg);
                let base = w * h + by * w + bx * 2;
                out[base] = q(128.0 + 224.0 * (avg[2] - ey) / 1.8556);
                out[base + 1] = q(128.0 + 224.0 * (avg[0] - ey) / 1.5748);
            }
        }
        out
    }

    /// GPU golden (skips without an adapter; run with --test-threads=1): a
    /// deterministic gradient WITH semi-transparent pixels through the compute
    /// pass matches [`cpu_reference_nv12`] within ±1 LSB on every byte of both
    /// planes — matrix, range, white-composite and chroma siting all pinned.
    /// The width is deliberately not a multiple of 4 so the row-padding
    /// compaction path is exercised on real GPU output.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn gpu_conversion_matches_cpu_reference_on_gradient_with_alpha() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let conv = Nv12Converter::new(&gpu.device).expect("converter");
        let (w, h) = (54usize, 32usize); // 54 % 4 != 0: padded rows compact
        let mut src = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                // PREMULTIPLIED texel: channels never exceed alpha.
                let a = ((x * 5 + y * 11) % 256) as u8;
                src[i] = ((x * 255 / (w - 1)) as u8).min(a);
                src[i + 1] = ((y * 255 / (h - 1)) as u8).min(a);
                src[i + 2] = (((x + y) * 7 % 251) as u8).min(a);
                src[i + 3] = a;
            }
        }
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("nv12 golden"),
            size: wgpu::Extent3d {
                width: w as u32,
                height: h as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
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
                bytes_per_row: Some((w * 4) as u32),
                rows_per_image: Some(h as u32),
            },
            wgpu::Extent3d {
                width: w as u32,
                height: h as u32,
                depth_or_array_layers: 1,
            },
        );
        let got = conv
            .convert(&gpu.device, &gpu.queue, &texture, w as u32, h as u32, 8)
            .expect("convert");
        let want = cpu_reference_nv12(&src, w, h);
        assert_eq!(got.len(), want.len());
        let mut worst = 0u8;
        for (i, (&g, &r)) in got.iter().zip(&want).enumerate() {
            let d = g.abs_diff(r);
            worst = worst.max(d);
            assert!(
                d <= 1,
                "byte {i} ({}): gpu {g} vs reference {r}",
                if i < w * h { "Y" } else { "UV" }
            );
        }
        // Not a bound — recorded so a future looser driver shows up in logs.
        eprintln!("nv12 golden: worst |delta| vs CPU reference = {worst}");
    }

    /// GPU proof of the conversion math (skips without an adapter; run with
    /// --test-threads=1): flat known colors through the compute pass land on
    /// the float BT.709/limited reference exactly, on both plane layouts.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn gpu_conversion_matches_bt709_reference_on_flat_colors() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let conv = Nv12Converter::new(&gpu.device).expect("converter");
        // (rgba, expected Y, expected Cb, expected Cr) — float reference:
        // Y = round(16 + 219*Ey), Cb = round(128 + 224*(B-Ey)/1.8556),
        // Cr = round(128 + 224*(R-Ey)/1.5748).
        for (rgba, ey, ecb, ecr) in [
            ([255u8, 255, 255, 255], 235u8, 128u8, 128u8), // white
            ([0, 0, 0, 255], 16, 128, 128),                // black
            ([255, 0, 0, 255], 63, 102, 240),              // red
            ([0, 255, 0, 255], 173, 42, 26),               // green
            ([0, 0, 255, 255], 32, 240, 118),              // blue
        ] {
            let (w, h) = (8u32, 4u32);
            let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("nv12 test"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let px: Vec<u8> = rgba
                .iter()
                .copied()
                .cycle()
                .take((w * h * 4) as usize)
                .collect();
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &px,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            let out = conv
                .convert(&gpu.device, &gpu.queue, &texture, w, h, 8)
                .expect("convert");
            assert_eq!(out.len(), nv12_frame_bytes(w, h));
            let y_plane = &out[..(w * h) as usize];
            let uv_plane = &out[(w * h) as usize..];
            assert!(
                y_plane.iter().all(|&v| v.abs_diff(ey) <= 1),
                "rgba {rgba:?}: Y {} != {ey}±1",
                y_plane[0]
            );
            for pair in uv_plane.chunks_exact(2) {
                assert!(
                    pair[0].abs_diff(ecb) <= 1 && pair[1].abs_diff(ecr) <= 1,
                    "rgba {rgba:?}: UV {pair:?} != ({ecb},{ecr})±1"
                );
            }
        }
    }
}
