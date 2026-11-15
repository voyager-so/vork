//! convert.rs — the output-format pass at the readback seam (render-serve.md §8.2, §8.3).
//!
//! The vector renderer composites premultiplied, in gamma sRGB, into an 8-bit target. One compute pass turns a
//! region of that target into the pixel format a client asked for, and reduces the tight bounding box of the
//! pixels with non-zero alpha on the way:
//!
//! | format | per pixel |
//! |---|---|
//! | `Rgba8Straight` | unpremultiplied with `readback::unpremultiply_in_place`'s integer math (round half up; `a = 0` → 0) |
//! | `Rgba8Premultiplied` | the target's bytes |
//! | `Rgba16fLinear` | `c = rgb / a`, `lin = srgb_eotf(c)` (the exact piecewise curve), `(lin · a, a)` as `f16`; `a = 0` → 0 |
//! | `Rgba16fSrgb` | the gamma-encoded premultiplied values widened to `f16`, no decode |
//!
//! A region may reach outside the target (a frame cut to an export box larger than its surface): those pixels are
//! transparent zeros, as `--crop-to-node` makes them. [`convert_cpu`] is the reference implementation, in `f64`
//! with round-to-nearest-even `f16`: the GPU matches it exactly for the 8-bit formats and within one `f16` ulp for
//! the float ones (`tests/convert_formats.rs`). It is also the fallback for a frame too large for one storage
//! binding.
//!
//! The WGSL is inline, like `nv12.rs`'s: moda-shaders holds only the WGSL extracted from the production shader
//! assemblers, and this pass has no production twin.

use anyhow::{bail, Context, Result};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub struct ConvertTimings {
    pub gpu_ms: f64,
    pub readback_ms: f64,
    pub wait_ms: f64,
    pub convert_ms: f64,
    pub gpu_timestamp: bool,
    pub readback_timestamp: bool,
}

#[derive(Clone)]
struct Timestamps {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    period: f64,
    copy: bool,
}

/// What the pass writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutFormat {
    Rgba8Straight,
    Rgba8Premultiplied,
    Rgba16fLinear,
    Rgba16fSrgb,
}

impl OutFormat {
    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            OutFormat::Rgba8Straight | OutFormat::Rgba8Premultiplied => 4,
            OutFormat::Rgba16fLinear | OutFormat::Rgba16fSrgb => 8,
        }
    }

    fn code(self) -> u32 {
        match self {
            OutFormat::Rgba8Straight => 0,
            OutFormat::Rgba8Premultiplied => 1,
            OutFormat::Rgba16fLinear => 2,
            OutFormat::Rgba16fSrgb => 3,
        }
    }
}

/// The part of the source target to convert, in target pixels; it may reach outside the target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Region {
    pub fn full(width: u32, height: u32) -> Region {
        Region {
            x: 0,
            y: 0,
            width,
            height,
        }
    }
}

/// A converted frame: tightly packed rows of `region.width × bytes_per_pixel`.
#[derive(Clone, Debug, PartialEq)]
pub struct Converted {
    pub bytes: Vec<u8>,
    /// `[x, y, width, height]` of the pixels with non-zero alpha, in region pixels; `None` for an empty frame.
    pub bbox: Option<[u32; 4]>,
}

const SHADER: &str = r#"
struct Params {
    src_w: u32,
    src_h: u32,
    x0: i32,
    y0: i32,
    w: u32,
    h: u32,
    format: u32,
    _pad: u32,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> bbox: array<atomic<u32>, 4>;

var<workgroup> wb: array<atomic<u32>, 4>;

// The sRGB transfer function's exact piecewise inverse (not a 2.2 power).
fn eotf(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if (li == 0u) {
        atomicStore(&wb[0], 0xffffffffu);
        atomicStore(&wb[1], 0xffffffffu);
        atomicStore(&wb[2], 0u);
        atomicStore(&wb[3], 0u);
    }
    workgroupBarrier();
    if (gid.x < params.w && gid.y < params.h) {
        let sx = i32(gid.x) + params.x0;
        let sy = i32(gid.y) + params.y0;
        var p = vec4<f32>(0.0);
        // i32 coords and a literal-0 mip level, as every textureLoad in this repo (naga's GLSL backend).
        if (sx >= 0 && sy >= 0 && sx < i32(params.src_w) && sy < i32(params.src_h)) {
            p = textureLoad(src, vec2<i32>(sx, sy), 0);
        }
        // The target's bytes (unorm8: exact through f32).
        let b = vec4<u32>(round(clamp(p, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0));
        let i = gid.y * params.w + gid.x;
        switch params.format {
            case 0u: {
                var c = b.rgb;
                if (b.a == 0u) {
                    c = vec3<u32>(0u);
                } else if (b.a < 255u) {
                    c = min((b.rgb * 255u + vec3<u32>(b.a >> 1u)) / vec3<u32>(b.a), vec3<u32>(255u));
                }
                out[i] = c.r | (c.g << 8u) | (c.b << 16u) | (b.a << 24u);
            }
            case 1u: {
                out[i] = b.r | (b.g << 8u) | (b.b << 16u) | (b.a << 24u);
            }
            case 2u: {
                var rgb = vec3<f32>(0.0);
                let a = f32(b.a) / 255.0;
                if (b.a > 0u) {
                    let c = min(vec3<f32>(b.rgb) / f32(b.a), vec3<f32>(1.0));
                    rgb = vec3<f32>(eotf(c.r), eotf(c.g), eotf(c.b)) * a;
                }
                out[2u * i] = pack2x16float(rgb.rg);
                out[2u * i + 1u] = pack2x16float(vec2<f32>(rgb.b, a));
            }
            default: {
                let f = vec4<f32>(b) / 255.0;
                out[2u * i] = pack2x16float(f.rg);
                out[2u * i + 1u] = pack2x16float(f.ba);
            }
        }
        if (b.a > 0u) {
            atomicMin(&wb[0], gid.x);
            atomicMin(&wb[1], gid.y);
            atomicMax(&wb[2], gid.x);
            atomicMax(&wb[3], gid.y);
        }
    }
    workgroupBarrier();
    if (li == 0u) {
        let min_x = atomicLoad(&wb[0]);
        if (min_x != 0xffffffffu) {
            atomicMin(&bbox[0], min_x);
            atomicMin(&bbox[1], atomicLoad(&wb[1]));
            atomicMax(&bbox[2], atomicLoad(&wb[2]));
            atomicMax(&bbox[3], atomicLoad(&wb[3]));
        }
    }
}
"#;

struct Buffers {
    bytes: u64,
    storage: wgpu::Buffer,
    staging: wgpu::Buffer,
    timestamps: Option<Timestamps>,
}

/// The conversion pipeline, built once per device.
///
/// A conversion is two halves (render-serve.md §8.3, S7): [`Converter::submit`] encodes the pass and the copy to a
/// staging buffer, submits them and asks for the staging buffer's map, without waiting; [`Converter::finish`] waits
/// for that submission alone (not for work submitted after it), and hands the mapped bytes to the caller, who copies
/// them straight where they go (a ring slot) with no intermediate buffer. Two staging sets alternate, so one frame's
/// readback can be in flight while the next frame is planned, drawn and submitted.
pub struct Converter {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    bbox: wgpu::Buffer,
    buffers: std::cell::RefCell<[Option<Buffers>; 2]>,
    /// The staging set the next submit uses.
    next: std::cell::Cell<usize>,
    /// Which sets have a conversion submitted and not yet finished.
    busy: std::cell::Cell<[bool; 2]>,
    first_done: std::cell::Cell<bool>,
    profile: bool,
}

/// A conversion submitted to the GPU, its staging buffer being mapped: [`Converter::finish`] reads it.
#[must_use = "a submitted conversion holds its staging set until finished"]
pub struct PendingConvert {
    set: usize,
    bytes: u64,
    index: wgpu::SubmissionIndex,
    mapped: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    done: std::sync::mpsc::Receiver<f64>,
    timestamps: Option<Timestamps>,
    work_done: Option<std::sync::mpsc::Receiver<f64>>,
}

impl PendingConvert {
    /// The converted frame's size in bytes.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl Converter {
    pub fn new(device: &wgpu::Device) -> Result<Converter> {
        Self::new_with_profile(device, false)
    }

    pub fn new_profiled(device: &wgpu::Device) -> Result<Converter> {
        Self::new_with_profile(device, true)
    }

    fn new_with_profile(device: &wgpu::Device, profile: bool) -> Result<Converter> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("format convert"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("format convert"),
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
                storage_entry(1),
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
                storage_entry(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("format convert"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = moda_text_draw::device::create_compute_pipeline(device, &wgpu::ComputePipelineDescriptor {
            label: Some("format convert"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("format convert params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bbox = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("format convert bbox"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Ok(Converter {
            pipeline,
            layout,
            uniform,
            bbox,
            buffers: std::cell::RefCell::new([None, None]),
            next: std::cell::Cell::new(0),
            busy: std::cell::Cell::new([false; 2]),
            first_done: std::cell::Cell::new(false),
            profile,
        })
    }

    /// The largest output (bytes) one pass writes: over it, use [`convert_cpu`].
    pub fn max_bytes(device: &wgpu::Device) -> u64 {
        let limits = device.limits();
        limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            - 48
    }

    /// Convert `region` of `texture` (rgba8unorm/bgra8unorm, premultiplied sRGB, sample count 1, TEXTURE_BINDING)
    /// to `format`: submit the pass and the readback, without waiting. At most two can be pending.
    pub fn submit(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        region: Region,
        format: OutFormat,
    ) -> Result<PendingConvert> {
        let (w, h) = (region.width, region.height);
        if w == 0 || h == 0 {
            bail!("format conversion: zero-sized region ({w}x{h})");
        }
        let bytes = u64::from(w) * u64::from(h) * u64::from(format.bytes_per_pixel());
        if bytes > Self::max_bytes(device) {
            bail!(
                "format conversion: {bytes} bytes is over one storage binding: convert on the CPU"
            );
        }
        let set = self.next.get();
        let mut busy = self.busy.get();
        if busy[set] {
            bail!("format conversion: both staging sets are in flight: finish one first");
        }
        let mut sets = self.buffers.borrow_mut();
        let cache = &mut sets[set];
        if cache.as_ref().is_none_or(|b| b.bytes < bytes) {
            // Grow to fit; a smaller frame reuses the larger buffers.
            let storage = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("format convert out"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("format convert read"),
                size: bytes + 48,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let timestamps = (self.profile
                && device.features().contains(wgpu::Features::TIMESTAMP_QUERY))
            .then(|| Timestamps {
                queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("format timings"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 4,
                }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("format timings resolve"),
                    size: 32,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                period: f64::from(queue.get_timestamp_period()),
                copy: device
                    .features()
                    .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
            });
            *cache = Some(Buffers {
                bytes,
                storage,
                staging,
                timestamps,
            });
        }
        let buffers = cache.as_ref().expect("ensured above");
        let size = texture.size();
        let params: [u32; 8] = [
            size.width,
            size.height,
            region.x as u32,
            region.y as u32,
            w,
            h,
            format.code(),
            0,
        ];
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&params));
        queue.write_buffer(
            &self.bbox,
            0,
            bytemuck::cast_slice(&[u32::MAX, u32::MAX, 0, 0]),
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("format convert"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers.storage,
                        offset: 0,
                        size: std::num::NonZeroU64::new(bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.bbox.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("format convert"),
        });
        let timestamps = buffers.timestamps.clone();
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("format convert"),
                timestamp_writes: timestamps.as_ref().map(|timing| {
                    wgpu::ComputePassTimestampWrites {
                        query_set: &timing.queries,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }
                }),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        }
        let submitted = Instant::now();
        let separate_copy = self.profile && timestamps.as_ref().is_none_or(|timing| !timing.copy);
        let work_done = if separate_copy {
            crate::scratch::flush();
            queue.submit([encoder.finish()]);
            let (work_tx, work_rx) = std::sync::mpsc::channel();
            queue.on_submitted_work_done(move || {
                let _ = work_tx.send(submitted.elapsed().as_secs_f64() * 1e3);
            });
            encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("format readback"),
            });
            Some(work_rx)
        } else {
            None
        };
        if let Some(timing) = timestamps.as_ref().filter(|timing| timing.copy) {
            encoder.write_timestamp(&timing.queries, 2);
        }
        encoder.copy_buffer_to_buffer(&buffers.storage, 0, &buffers.staging, 0, bytes);
        encoder.copy_buffer_to_buffer(&self.bbox, 0, &buffers.staging, bytes, 16);
        if let Some(timing) = &timestamps {
            if timing.copy {
                encoder.write_timestamp(&timing.queries, 3);
            }
            let count = if timing.copy { 4 } else { 2 };
            encoder.resolve_query_set(&timing.queries, 0..count, &timing.resolve, 0);
            encoder.copy_buffer_to_buffer(
                &timing.resolve,
                0,
                &buffers.staging,
                bytes + 16,
                u64::from(count) * 8,
            );
        }
        crate::scratch::flush();
        let index = queue.submit([encoder.finish()]);
        let (done_tx, done) = std::sync::mpsc::channel();
        queue.on_submitted_work_done(move || {
            let _ = done_tx.send(submitted.elapsed().as_secs_f64() * 1e3);
        });

        let (tx, rx) = std::sync::mpsc::channel();
        buffers
            .staging
            .slice(..bytes + 48)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        busy[set] = true;
        self.busy.set(busy);
        self.next.set(1 - set);
        Ok(PendingConvert {
            set,
            bytes,
            index,
            mapped: rx,
            done,
            timestamps,
            work_done,
        })
    }

    /// Wait for a submitted conversion (its own submission, not later ones), give its bytes to `sink` straight from
    /// the mapped staging buffer, and return the bbox (`[x, y, width, height]`, `None` for an empty frame).
    pub fn finish(
        &self,
        device: &wgpu::Device,
        pending: PendingConvert,
        sink: impl FnOnce(&[u8]),
    ) -> Result<Option<[u32; 4]>> {
        self.finish_profiled(device, pending, sink)
            .map(|(bbox, _)| bbox)
    }

    pub fn finish_profiled(
        &self,
        device: &wgpu::Device,
        pending: PendingConvert,
        sink: impl FnOnce(&[u8]),
    ) -> Result<(Option<[u32; 4]>, ConvertTimings)> {
        let wait = Instant::now();
        let PendingConvert {
            set,
            bytes,
            index,
            mapped,
            done,
            timestamps,
            work_done,
        } = pending;
        let mut busy = self.busy.get();
        busy[set] = false;
        self.busy.set(busy);
        // The first pass after device creation compiles its pipeline inside the wait.
        let retries = if self.first_done.get() {
            crate::readback::READBACK_POLL_TIMEOUT_RETRIES
        } else {
            crate::readback::READBACK_POLL_TIMEOUT_RETRIES_FIRST
        };
        crate::readback::poll_submission_with_budget(device, index, retries)
            .context("device poll during the format conversion")?;
        mapped
            .recv()
            .context("format convert map_async channel")??;
        let completed_ms = done.recv().context("format convert completion channel")?;
        let gpu_ms = match work_done {
            Some(work_done) => work_done.recv().context("format work completion channel")?,
            None => completed_ms,
        };
        let readback_ms = (completed_ms - gpu_ms).max(0.0);
        let wait_ms = wait.elapsed().as_secs_f64() * 1e3;
        let sets = self.buffers.borrow();
        let buffers = sets[set].as_ref().expect("a submitted set exists");
        let slice = buffers.staging.slice(..bytes + 48);
        let data = slice.get_mapped_range();
        let mut timings = ConvertTimings {
            gpu_ms,
            readback_ms,
            wait_ms,
            ..Default::default()
        };
        if let Some(timing) = &timestamps {
            let offset = bytes as usize + 16;
            let stamps: Vec<u64> = data[offset..offset + if timing.copy { 32 } else { 16 }]
                .chunks_exact(8)
                .map(|stamp| u64::from_ne_bytes(stamp.try_into().unwrap()))
                .collect();
            timings.gpu_ms = stamps[1].wrapping_sub(stamps[0]) as f64 * timing.period / 1e6;
            timings.gpu_timestamp = true;
            if timing.copy {
                timings.readback_ms =
                    stamps[3].wrapping_sub(stamps[2]) as f64 * timing.period / 1e6;
                timings.readback_timestamp = true;
            }
        }
        let copy = Instant::now();
        sink(&data[..bytes as usize]);
        timings.convert_ms = copy.elapsed().as_secs_f64() * 1e3;
        let b: [u32; 4] = bytemuck::pod_read_unaligned(&data[bytes as usize..bytes as usize + 16]);
        drop(data);
        buffers.staging.unmap();
        self.first_done.set(true);
        Ok((
            (b[0] != u32::MAX).then(|| [b[0], b[1], b[2] - b[0] + 1, b[3] - b[1] + 1]),
            timings,
        ))
    }

    /// [`Converter::submit`] then [`Converter::finish`] into a new buffer.
    pub fn convert(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        region: Region,
        format: OutFormat,
    ) -> Result<Converted> {
        let pending = self.submit(device, queue, texture, region, format)?;
        let mut out = Vec::new();
        let bbox = self.finish(device, pending, |b| out.extend_from_slice(b))?;
        Ok(Converted { bytes: out, bbox })
    }
}

/// A value → IEEE binary16 bits, round to nearest even (overflow to infinity, subnormals kept).
pub fn f16_bits(x: f64) -> u16 {
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a.is_nan() {
        return 0x7e00;
    }
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    // Subnormal step 2^-24; normals by exponent.
    if a < 6.103_515_625e-5 {
        let m = round_half_even(a / 5.960_464_477_539_063e-8);
        return sign | m as u16;
    }
    let e = a.log2().floor() as i32;
    let mut e = e;
    // Guard log2 rounding at powers of two.
    if 2f64.powi(e) > a {
        e -= 1;
    }
    if 2f64.powi(e + 1) <= a {
        e += 1;
    }
    let m = round_half_even((a / 2f64.powi(e) - 1.0) * 1024.0);
    let (mut e, mut m) = (e + 15, m as u32);
    if m == 1024 {
        m = 0;
        e += 1;
    }
    if e >= 31 {
        return sign | 0x7c00;
    }
    sign | ((e as u16) << 10) | m as u16
}

fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// The sRGB EOTF, exact, in `f64`.
pub fn srgb_eotf(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// The reference conversion (and the fallback for frames too large for one pass): `src` is the target's
/// premultiplied sRGB RGBA8, `src_w × src_h`.
pub fn convert_cpu(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    region: Region,
    format: OutFormat,
) -> Converted {
    let bpp = format.bytes_per_pixel() as usize;
    let (w, h) = (region.width as usize, region.height as usize);
    let mut out = vec![0u8; w * h * bpp];
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (
                x as i64 + i64::from(region.x),
                y as i64 + i64::from(region.y),
            );
            let p: [u8; 4] = if sx >= 0 && sy >= 0 && sx < i64::from(src_w) && sy < i64::from(src_h)
            {
                let i = (sy as usize * src_w as usize + sx as usize) * 4;
                src[i..i + 4].try_into().unwrap()
            } else {
                [0; 4]
            };
            let o = &mut out[(y * w + x) * bpp..(y * w + x + 1) * bpp];
            let a = p[3];
            match format {
                OutFormat::Rgba8Straight => {
                    let mut px = p;
                    crate::readback::unpremultiply_in_place(&mut px);
                    o.copy_from_slice(&px);
                }
                OutFormat::Rgba8Premultiplied => o.copy_from_slice(&p),
                OutFormat::Rgba16fLinear => {
                    let af = f64::from(a) / 255.0;
                    let mut v = [0.0f64; 4];
                    if a > 0 {
                        for c in 0..3 {
                            let straight = (f64::from(p[c]) / f64::from(a)).min(1.0);
                            v[c] = srgb_eotf(straight) * af;
                        }
                    }
                    v[3] = af;
                    for c in 0..4 {
                        o[c * 2..c * 2 + 2].copy_from_slice(&f16_bits(v[c]).to_le_bytes());
                    }
                }
                OutFormat::Rgba16fSrgb => {
                    for c in 0..4 {
                        o[c * 2..c * 2 + 2]
                            .copy_from_slice(&f16_bits(f64::from(p[c]) / 255.0).to_le_bytes());
                    }
                }
            }
            if a > 0 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    Converted {
        bytes: out,
        bbox: (x0 != usize::MAX).then(|| {
            [
                x0 as u32,
                y0 as u32,
                (x1 - x0 + 1) as u32,
                (y1 - y0 + 1) as u32,
            ]
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_reference_values() {
        assert_eq!(f16_bits(0.0), 0);
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(0.5), 0x3800);
        assert_eq!(f16_bits(65504.0), 0x7bff);
        assert_eq!(f16_bits(1e6), 0x7c00);
        assert_eq!(
            f16_bits(5.960_464_477_539_063e-8),
            1,
            "the smallest subnormal"
        );
        assert_eq!(f16_bits(6.103_515_625e-5), 0x0400, "the smallest normal");
        assert_eq!(f16_bits(1.0 / 3.0), 0x3555);
        // 1 + 2^-11 is halfway between 1 and the next f16: ties to even (1.0).
        assert_eq!(f16_bits(1.0 + 2f64.powi(-11)), 0x3c00);
        assert_eq!(f16_bits(1.0 + 3.0 * 2f64.powi(-11)), 0x3c02);
        assert_eq!(f16_bits(-2.0), 0xc000);
    }

    #[test]
    fn eotf_is_the_piecewise_curve() {
        assert_eq!(srgb_eotf(0.0), 0.0);
        assert!((srgb_eotf(1.0) - 1.0).abs() < 1e-15);
        assert!((srgb_eotf(0.04045) - 0.04045 / 12.92).abs() < 1e-15);
        assert!((srgb_eotf(0.5) - 0.214_041_140_482_232_5).abs() < 1e-12);
    }

    #[test]
    fn cpu_reference_formats() {
        // One opaque red, one half-transparent premultiplied grey, one clear.
        let src = [255u8, 0, 0, 255, 64, 64, 64, 128, 0, 0, 0, 0];
        let r = Region::full(3, 1);
        let s = convert_cpu(&src, 3, 1, r, OutFormat::Rgba8Straight);
        assert_eq!(s.bytes, [255, 0, 0, 255, 128, 128, 128, 128, 0, 0, 0, 0]);
        assert_eq!(s.bbox, Some([0, 0, 2, 1]));
        let l = convert_cpu(&src, 3, 1, r, OutFormat::Rgba16fLinear);
        let px = |i: usize, c: usize| {
            u16::from_le_bytes([l.bytes[i * 8 + c * 2], l.bytes[i * 8 + c * 2 + 1]])
        };
        assert_eq!((px(0, 0), px(0, 3)), (0x3c00, 0x3c00));
        // 64/128 = 0.5 straight → 0.2140 linear → × 128/255.
        assert_eq!(px(1, 0), f16_bits(0.214_041_140_482_232_5 * 128.0 / 255.0));
        assert!((0..4).all(|c| px(2, c) == 0));
        // A region past the edge is transparent there.
        let e = convert_cpu(
            &src,
            3,
            1,
            Region {
                x: 2,
                y: 0,
                width: 2,
                height: 1,
            },
            OutFormat::Rgba8Premultiplied,
        );
        assert_eq!(e.bytes, [0; 8]);
        assert_eq!(e.bbox, None);
    }
}
