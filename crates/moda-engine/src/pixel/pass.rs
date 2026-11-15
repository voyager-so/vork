//! One GPU pass of the pixel graph: a fullscreen fragment kernel over an output domain, reading up to four input
//! textures in frame space.
//!
//! Every kernel is WGSL `fn kernel(p: vec2<f32>) -> vec4<f32>`, where `p` is the output pixel's centre in frame
//! pixels (`(0.5, 0.5)` is the frame's top-left pixel). The prelude ([`PRELUDE`]) gives it the inputs as
//! `px(i, q)` (the texel at integer frame pixel `q`, transparent outside the input's domain) and `bilinear(i, p)`,
//! and the parameters as `P(k)` (`vec4` slot `k`, [`PARAM_SLOTS`] of them). Loads are exact `textureLoad`s, so `f32` data
//! groups (not filterable in WebGPU core) work the same as `f16` colour.
//!
//! Pipelines are cached by (kernel source hash, output format): the format is in the key for the same reason as the
//! engine's pipeline map.

use std::cell::RefCell;
use std::collections::HashMap;

use super::image::{Domain, Storage};
use super::pool::{PoolTexture, TexturePool};
use super::PixelError;

/// The largest tile of an agent shader's pass, in output px a side ([`Passes::run_agent`]).
pub const AGENT_TILE: u32 = 512;

/// What an agent shader's pass made: its texture, the first non-finite pixel it drew (output px), and how long it
/// took (ms).
pub struct AgentRun {
    pub texture: PoolTexture,
    pub nan: Option<(u32, u32)>,
    pub ms: f64,
}

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
/// The browser can't block on the GPU: a wasm build doesn't time agent passes (the tiles still bound each submission).
#[cfg(target_arch = "wasm32")]
struct Instant;
#[cfg(target_arch = "wasm32")]
impl Instant {
    fn now() -> Self {
        Instant
    }
    fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}

/// Wait for submitted work (native); a no-op in the browser.
fn wait(device: &wgpu::Device) -> Result<(), PixelError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::readback::poll_until_complete_with_budget(device, crate::readback::READBACK_POLL_TIMEOUT_RETRIES)
            .map_err(|e| PixelError::Invalid(format!("waiting for the GPU: {e:#}")))?;
    }
    #[cfg(target_arch = "wasm32")]
    let _ = device;
    Ok(())
}

/// Re-runs of a tile that took an agent shader's run past its budget ([`BudgetClock::charge`]).
pub const AGENT_TILE_RETRIES: u32 = 2;

/// An agent shader run's time against its budget, charged tile by tile. The time is wall-clock, so another
/// process's load can stretch one tile: a tile that takes the run past the budget is drawn again (at most
/// [`AGENT_TILE_RETRIES`] times, over the same pixels, so the output doesn't change) and its fastest time counts.
/// The run is over budget only when even that time overshoots, so a load spike doesn't fail an export.
struct BudgetClock {
    budget_ms: f64,
    spent_ms: f64,
}

impl BudgetClock {
    fn new(budget_ms: f64) -> Self {
        Self { budget_ms, spent_ms: 0.0 }
    }

    /// The time charged so far (ms).
    fn spent(&self) -> f64 {
        self.spent_ms
    }

    /// Run a tile (`draw` returns its time, ms) and charge its fastest time: `false` when the run is over budget.
    fn charge<E>(&mut self, mut draw: impl FnMut() -> Result<f64, E>) -> Result<bool, E> {
        let mut best = draw()?;
        let mut retries = 0;
        while self.spent_ms + best > self.budget_ms && retries < AGENT_TILE_RETRIES {
            best = best.min(draw()?);
            retries += 1;
        }
        self.spent_ms += best;
        Ok(self.spent_ms <= self.budget_ms)
    }
}

/// Inputs a pass can read.
pub const MAX_INPUTS: usize = moda_wgsl::MAX_INPUTS;
/// `vec4<f32>` parameter slots.
pub const PARAM_SLOTS: usize = moda_wgsl::PARAM_SLOTS;

/// The WGSL every kernel is compiled with (moda-wgsl owns it: agent shaders are checked against the same text).
pub const PRELUDE: &str = moda_wgsl::PASS_PRELUDE;

const FRAGMENT: &str = moda_wgsl::PASS_FRAGMENT;

/// A kernel: its WGSL body (with `fn kernel`) and a stable name for errors and stats.
#[derive(Clone, Copy, Debug)]
pub struct Kernel<'a> {
    pub name: &'a str,
    pub body: &'a str,
}

/// An input to a pass: a texture and the domain it covers.
#[derive(Clone, Copy)]
pub struct PassInput<'a> {
    pub texture: &'a PoolTexture,
    pub domain: Domain,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HeaderData {
    out_origin: [f32; 2],
    out_size: [f32; 2],
    rects: [[f32; 4]; MAX_INPUTS],
    params: [[f32; 4]; PARAM_SLOTS],
}

/// Runs kernels. One per executor; shares the device with the renderer.
pub struct Passes {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: RefCell<HashMap<(u64, wgpu::TextureFormat), wgpu::RenderPipeline>>,
    /// Agent shaders' pipelines: group 0 as every kernel, group 1 the non-finite report buffer.
    agent_layout: wgpu::BindGroupLayout,
    agent_pipeline_layout: wgpu::PipelineLayout,
    agent_pipelines: RefCell<HashMap<(u64, wgpu::TextureFormat), wgpu::RenderPipeline>>,
    /// Bound to unused input slots (and to `f32` ones: see [`Passes::run`]).
    dummy: wgpu::TextureView,
    passes: std::cell::Cell<u64>,
    /// Download staging buffers, kept between downloads (a 1080p `f32` read is 33 MB of fresh memory otherwise): one
    /// for each download in flight ([`Passes::download_start`]), at most [`KEPT_STAGING`] kept.
    staging: RefCell<Vec<wgpu::Buffer>>,
}

fn hash_source(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    s.hash(&mut h);
    h.finish()
}

impl Passes {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pixel pass"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex(1),
                tex(2),
                tex(3),
                tex(4),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pixel pass"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let agent_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("agent shader report"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let agent_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("agent shader"),
            bind_group_layouts: &[Some(&layout), Some(&agent_layout)],
            ..Default::default()
        });
        let dummy = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("pixel pass: empty input"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            device: device.clone(),
            queue: queue.clone(),
            layout,
            pipeline_layout,
            pipelines: RefCell::new(HashMap::new()),
            agent_layout,
            agent_pipeline_layout,
            agent_pipelines: RefCell::new(HashMap::new()),
            dummy,
            passes: std::cell::Cell::new(0),
            staging: RefCell::new(Vec::new()),
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Passes run so far (the executor's pass-count stats).
    pub fn count(&self) -> u64 {
        self.passes.get()
    }

    /// Compile a kernel (validating it) without running it; a WGSL error comes back as [`PixelError::Invalid`].
    pub fn compile(&self, kernel: Kernel<'_>, storage: Storage) -> Result<(), PixelError> {
        self.pipeline(kernel, storage.format()).map(|_| ())
    }

    fn pipeline(&self, kernel: Kernel<'_>, format: wgpu::TextureFormat) -> Result<wgpu::RenderPipeline, PixelError> {
        let key = (hash_source(kernel.body), format);
        if let Some(p) = self.pipelines.borrow().get(&key) {
            return Ok(p.clone());
        }
        let source = format!("{PRELUDE}\n{}\n{FRAGMENT}", kernel.body);
        let pipeline = self.build_pipeline(kernel.name, &source, "fs", &self.pipeline_layout, format)?;
        self.pipelines.borrow_mut().insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// Compile `source` (a whole module with `vs` and `fragment`) for `layout`, validating inside an error scope so a
    /// bad shader is a typed error, not an uncaptured device error.
    fn build_pipeline(
        &self,
        name: &str,
        source: &str,
        fragment: &str,
        layout: &wgpu::PipelineLayout,
        format: wgpu::TextureFormat,
    ) -> Result<wgpu::RenderPipeline, PixelError> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = moda_text_draw::device::create_render_pipeline(&self.device, &wgpu::RenderPipelineDescriptor {
            label: Some(name),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(fragment),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(PixelError::Invalid(format!("shader `{name}`: {e}")));
        }
        Ok(pipeline)
    }

    /// An agent shader's pipeline (a whole module from `moda_wgsl::module`), cached by source.
    fn agent_pipeline(&self, name: &str, wgsl: &str, format: wgpu::TextureFormat) -> Result<wgpu::RenderPipeline, PixelError> {
        let key = (hash_source(wgsl), format);
        if let Some(p) = self.agent_pipelines.borrow().get(&key) {
            return Ok(p.clone());
        }
        let pipeline = self
            .build_pipeline(name, wgsl, "fs", &self.agent_pipeline_layout, format)
            .map_err(|e| PixelError::Shader { code: "shader_invalid", message: e.to_string() })?;
        self.agent_pipelines.borrow_mut().insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// Run an agent shader module over `out` into a fresh pool texture: [`Passes::run_agent_on`].
    #[allow(clippy::too_many_arguments)]
    pub fn run_agent(
        &self,
        pool: &TexturePool,
        name: &str,
        wgsl: &str,
        inputs: &[PassInput<'_>],
        params: &[[f32; 4]],
        out: Domain,
        storage: Storage,
        budget_ms: f64,
    ) -> Result<AgentRun, PixelError> {
        let target = pool.acquire(out.w, out.h, storage)?;
        let views: Vec<(&wgpu::TextureView, Domain)> = inputs.iter().map(|i| (i.texture.view(), i.domain)).collect();
        let (nan, ms) = self.run_agent_on(name, wgsl, &views, params, target.view(), storage.format(), out, budget_ms)?;
        Ok(AgentRun { texture: target, nan, ms })
    }

    /// Run an agent shader module over `out` (output px) into `target` (`format`, `out`'s size), reading `inputs`
    /// (views and their domains), in tiles of at most [`AGENT_TILE`] px a side, each submitted and waited for, so
    /// no single submission can trip the OS GPU watchdog and the time is measured. Past `budget_ms` (each tile's
    /// fastest of its draws, [`BudgetClock`]) the run stops with `shader_budget_exceeded`. A non-finite result is
    /// drawn as a magenta checker by the module, and the first such pixel (output px) is returned, with the time
    /// taken (ms).
    #[allow(clippy::too_many_arguments)]
    pub fn run_agent_on(
        &self,
        name: &str,
        wgsl: &str,
        inputs: &[(&wgpu::TextureView, Domain)],
        params: &[[f32; 4]],
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        out: Domain,
        budget_ms: f64,
    ) -> Result<(Option<(u32, u32)>, f64), PixelError> {
        if inputs.len() > MAX_INPUTS || params.len() > PARAM_SLOTS {
            return Err(PixelError::Invalid(format!(
                "shader `{name}`: {} inputs and {} parameter slots (at most {MAX_INPUTS} and {PARAM_SLOTS})",
                inputs.len(),
                params.len()
            )));
        }
        let pipeline = self.agent_pipeline(name, wgsl, format)?;
        let bind = self.bind_views(inputs, params, out);
        let report = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agent shader report"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&report, 0, &[0u8; 16]);
        let report_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("agent shader report"),
            layout: &self.agent_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: report.as_entire_binding(),
            }],
        });
        // A one-pixel draw first, untimed: drivers that compile at first use (llvmpipe) would otherwise charge the
        // compile to the shader's budget.
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
            {
                let profile_ts = crate::gpu_profile::pass_timestamps();
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(name),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind.1, &[]);
                pass.set_bind_group(1, &report_bind, &[]);
                pass.set_scissor_rect(0, 0, 1.min(out.w), 1.min(out.h));
                pass.draw(0..3, 0..1);
            }
            crate::scratch::flush();
            self.queue.submit([encoder.finish()]);
            wait(&self.device)?;
        }
        let mut clock = BudgetClock::new(budget_ms);
        let mut first = true;
        let mut ty = 0;
        while ty < out.h {
            let th = AGENT_TILE.min(out.h - ty);
            let mut tx = 0;
            while tx < out.w {
                let tw = AGENT_TILE.min(out.w - tx);
                let clear = first;
                first = false;
                let within = clock.charge(|| -> Result<f64, PixelError> {
                    let started = Instant::now();
                    let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
                    {
                        let profile_ts = crate::gpu_profile::pass_timestamps();
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some(name),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: target,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: if clear { wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT) } else { wgpu::LoadOp::Load },
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        pass.set_pipeline(&pipeline);
                        pass.set_bind_group(0, &bind.1, &[]);
                        pass.set_bind_group(1, &report_bind, &[]);
                        pass.set_scissor_rect(tx, ty, tw, th);
                        pass.draw(0..3, 0..1);
                    }
                    crate::scratch::flush();
                    self.queue.submit([encoder.finish()]);
                    self.passes.set(self.passes.get() + 1);
                    wait(&self.device)?;
                    Ok(started.elapsed().as_secs_f64() * 1000.0)
                })?;
                if !within {
                    return Err(PixelError::Shader {
                        code: "shader_budget_exceeded",
                        message: format!(
                            "shader_budget_exceeded: shader `{name}` took over {budget_ms:.0} ms for {}×{} px (stopped after {:.0} ms): fewer octaves or loop iterations, or a cheaper formula",
                            out.w,
                            out.h,
                            clock.spent()
                        ),
                    });
                }
                tx += tw;
            }
            ty += th;
        }
        let ms = clock.spent();
        let nan = self.read_report(&report)?;
        Ok((nan, ms))
    }

    /// The report buffer: the first non-finite pixel (output px), if any. The browser can't block on a readback: a
    /// wasm build draws the magenta checker but doesn't report the pixel.
    #[cfg(target_arch = "wasm32")]
    fn read_report(&self, _report: &wgpu::Buffer) -> Result<Option<(u32, u32)>, PixelError> {
        Ok(None)
    }

    /// The report buffer: the first non-finite pixel (output px), if any.
    #[cfg(not(target_arch = "wasm32"))]
    fn read_report(&self, report: &wgpu::Buffer) -> Result<Option<(u32, u32)>, PixelError> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agent shader report readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(report, 0, &staging, 0, 16);
        crate::scratch::flush();
        self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        wait(&self.device)?;
        rx.recv()
            .map_err(|e| PixelError::Invalid(format!("agent shader report: {e}")))?
            .map_err(|e| PixelError::Invalid(format!("agent shader report: {e}")))?;
        let words: Vec<u32> = slice
            .get_mapped_range()
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        staging.unmap();
        Ok((words[0] > 0).then(|| (words[1], words[2])))
    }

    /// The header buffer and group-0 bind group for a pass.
    fn bind_inputs(&self, inputs: &[PassInput<'_>], params: &[[f32; 4]], out: Domain) -> (wgpu::Buffer, wgpu::BindGroup) {
        let views: Vec<(&wgpu::TextureView, Domain)> = inputs.iter().map(|i| (i.texture.view(), i.domain)).collect();
        self.bind_views(&views, params, out)
    }

    /// [`Passes::bind_inputs`] over plain texture views.
    fn bind_views(&self, inputs: &[(&wgpu::TextureView, Domain)], params: &[[f32; 4]], out: Domain) -> (wgpu::Buffer, wgpu::BindGroup) {
        let mut header = HeaderData {
            out_origin: [out.x as f32, out.y as f32],
            out_size: [out.w as f32, out.h as f32],
            rects: [[0.0; 4]; MAX_INPUTS],
            params: [[0.0; 4]; PARAM_SLOTS],
        };
        for (i, (_, d)) in inputs.iter().enumerate() {
            header.rects[i] = [d.x as f32, d.y as f32, d.w as f32, d.h as f32];
        }
        header.params[..params.len()].copy_from_slice(params);
        let buffer = wgpu::util::DeviceExt::create_buffer_init(
            &self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("pixel pass header"),
                contents: bytemuck::bytes_of(&header),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let view = |i: usize| inputs.get(i).map_or(&self.dummy, |(v, _)| *v);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pixel pass"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view(0)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(view(1)),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(view(2)),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(view(3)),
                },
            ],
        });
        (buffer, bind)
    }

    /// Run `kernel` over `out` into a fresh pool texture of `storage`, reading `inputs` (at most [`MAX_INPUTS`])
    /// with `params` (at most [`PARAM_SLOTS`] `vec4`s).
    pub fn run(
        &self,
        pool: &TexturePool,
        kernel: Kernel<'_>,
        inputs: &[PassInput<'_>],
        params: &[[f32; 4]],
        out: Domain,
        storage: Storage,
    ) -> Result<PoolTexture, PixelError> {
        if inputs.len() > MAX_INPUTS || params.len() > PARAM_SLOTS {
            return Err(PixelError::Invalid(format!(
                "kernel `{}`: {} inputs and {} parameter slots (at most {MAX_INPUTS} and {PARAM_SLOTS})",
                kernel.name,
                inputs.len(),
                params.len()
            )));
        }
        let pipeline = self.pipeline(kernel, storage.format())?;
        let target = pool.acquire(out.w, out.h, storage)?;
        let (_buffer, bind) = self.bind_inputs(inputs, params, out);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(kernel.name),
        });
        {
            let profile_ts = crate::gpu_profile::pass_timestamps();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(kernel.name),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        crate::scratch::flush();
        self.queue.submit([encoder.finish()]);
        self.passes.set(self.passes.get() + 1);
        Ok(target)
    }

    /// Upload tightly packed RGBA `f32` texels (a test fixture or a CPU-made image) into a pool texture.
    pub fn upload(
        &self,
        pool: &TexturePool,
        w: u32,
        h: u32,
        storage: Storage,
        rgba: &[f32],
    ) -> Result<PoolTexture, PixelError> {
        if rgba.len() != (w * h * 4) as usize {
            return Err(PixelError::Invalid("upload: texel count doesn't match the size".into()));
        }
        let bytes: Vec<u8> = match storage {
            Storage::U8 => rgba.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect(),
            Storage::F16 => rgba.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            Storage::F32 => rgba.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        self.upload_bytes(pool, w, h, storage, &bytes)
    }

    /// Upload tightly packed RGBA texels already in `storage`'s little-endian layout into a pool texture.
    pub fn upload_bytes(
        &self,
        pool: &TexturePool,
        w: u32,
        h: u32,
        storage: Storage,
        bytes: &[u8],
    ) -> Result<PoolTexture, PixelError> {
        if bytes.len() as u64 != w as u64 * h as u64 * storage.bytes_per_pixel() {
            return Err(PixelError::Invalid("upload: byte count doesn't match the size".into()));
        }
        let tex = pool.acquire(w, h, storage)?;
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * storage.bytes_per_pixel() as u32),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        Ok(tex)
    }

    /// Read a pool texture back as RGBA `f32` texels.
    pub fn download(&self, tex: &PoolTexture) -> anyhow::Result<Vec<f32>> {
        let (w, h) = tex.size();
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        let storage = tex.storage();
        self.download_rows(tex, |r| match storage {
            Storage::U8 => out.extend(r.iter().map(|v| *v as f32 / 255.0)),
            Storage::F16 => out.extend(r.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())),
            Storage::F32 => out.extend(r.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))),
        })?;
        Ok(out)
    }

    /// Read a pool texture back, handing `row` each row's tightly packed texel bytes (little-endian), top to bottom.
    pub fn download_rows(&self, tex: &PoolTexture, mut row_bytes: impl FnMut(&[u8])) -> anyhow::Result<()> {
        let (w, h) = tex.size();
        let used = (w as u64 * tex.storage().bytes_per_pixel()) as usize;
        self.download_mapped(tex, |data, row| {
            for y in 0..h as usize {
                row_bytes(&data[y * row..y * row + used]);
            }
        })
    }

    /// The texture's texels as mapped from the staging buffer: `mapped(bytes, row_stride)`, rows padded to the copy
    /// alignment.
    pub fn download_mapped(&self, tex: &PoolTexture, mapped: impl FnOnce(&[u8], usize)) -> anyhow::Result<()> {
        let download = self.download_start(tex)?;
        self.download_finish(download, mapped)
    }

    /// Start reading `tex` back: its copy into a staging buffer is submitted and the buffer's mapping requested, and
    /// the GPU goes on with whatever is submitted after it. [`Passes::download_finish`] waits for this copy alone.
    pub fn download_start(&self, tex: &PoolTexture) -> anyhow::Result<Download> {
        let (w, h) = tex.size();
        let bpp = tex.storage().bytes_per_pixel() as u32;
        let row = crate::readback::padded_bytes_per_row_bpp(w, bpp);
        let size = row as u64 * h as u64;
        let buffer = {
            let mut kept = self.staging.borrow_mut();
            match kept.iter().position(|b| b.size() >= size) {
                Some(i) => kept.swap_remove(i),
                None => self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pixel download"),
                    size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
            }
        };
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        crate::scratch::flush();
        let index = self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..size).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        Ok(Download { buffer, size, row, index, mapped: rx })
    }

    /// Wait for a [`Passes::download_start`] copy (not for the work submitted after it) and hand its texels to
    /// `mapped(bytes, row_stride)`.
    pub fn download_finish(&self, download: Download, mapped: impl FnOnce(&[u8], usize)) -> anyhow::Result<()> {
        let Download { buffer, size, row, index, mapped: rx } = download;
        let wait = crate::prof::span("pxr.download.wait");
        crate::readback::poll_submission_with_budget(&self.device, index, crate::readback::READBACK_POLL_TIMEOUT_RETRIES_FIRST)?;
        rx.recv()??;
        drop(wait);
        let _convert = crate::prof::span("pxr.download.convert");
        let slice = buffer.slice(..size);
        let data = slice.get_mapped_range();
        mapped(&data, row as usize);
        drop(data);
        buffer.unmap();
        let mut kept = self.staging.borrow_mut();
        if kept.len() < KEPT_STAGING {
            kept.push(buffer);
        }
        Ok(())
    }
}

/// Download staging buffers [`Passes`] keeps for reuse: a frame's readback and the one before it, in flight together.
const KEPT_STAGING: usize = 2;

/// A readback in flight ([`Passes::download_start`]).
pub struct Download {
    buffer: wgpu::Buffer,
    size: u64,
    row: u32,
    index: wgpu::SubmissionIndex,
    mapped: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}


#[cfg(test)]
mod budget_tests {
    use super::{BudgetClock, AGENT_TILE_RETRIES};

    /// A clock fed `times` (ms) for successive draws; returns whether each tile stayed in budget, and the draws made.
    fn run(budget: f64, tiles: usize, times: &[f64]) -> (Vec<bool>, usize) {
        let mut clock = BudgetClock::new(budget);
        let mut next = times.iter().copied();
        let mut draws = 0;
        let mut out = Vec::new();
        for _ in 0..tiles {
            let ok = clock
                .charge(|| -> Result<f64, ()> {
                    draws += 1;
                    Ok(next.next().expect("a time per draw"))
                })
                .unwrap();
            out.push(ok);
            if !ok {
                break;
            }
        }
        (out, draws)
    }

    #[test]
    fn a_load_spike_on_one_tile_is_redrawn_and_does_not_fail_the_run() {
        // 4 tiles of ~10 ms against 50 ms; the third is stretched to 40 ms once by another process's load (#2143).
        let (ok, draws) = run(50.0, 4, &[10.0, 10.0, 40.0, 10.0, 10.0]);
        assert_eq!(ok, vec![true; 4]);
        assert_eq!(draws, 5);
    }

    #[test]
    fn a_shader_that_is_really_over_budget_still_fails() {
        let (ok, draws) = run(50.0, 4, &[30.0, 30.0, 30.0, 30.0]);
        assert_eq!(ok, vec![true, false]);
        assert_eq!(draws, 2 + AGENT_TILE_RETRIES as usize);
    }

    #[test]
    fn tiles_within_budget_are_drawn_once() {
        let (ok, draws) = run(50.0, 3, &[10.0, 10.0, 10.0]);
        assert_eq!(ok, vec![true; 3]);
        assert_eq!(draws, 3);
    }
}
