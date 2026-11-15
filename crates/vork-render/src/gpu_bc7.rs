//! Install-time BC7 preparation. GPU mips and the BC7 modes 4/5/6 search feed the
//! existing compressed-texture entry; readback also supplies mip texels for
//! SIMD refinement of sampled alpha edges.
use super::*;
use crate::constants::{
    TEXTURE_BC7_GROUP_SIZE, TEXTURE_BLOCK_BYTES as BLOCK, TEXTURE_BLOCK_EDGE as EDGE,
};

pub(crate) const LAYOUT: [wgpu::BindGroupLayoutEntry; 4] = [
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
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(32),
        },
        count: None,
    },
    wgpu::BindGroupLayoutEntry {
        binding: 2,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(16),
        },
        count: None,
    },
    wgpu::BindGroupLayoutEntry {
        binding: 3,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(16),
        },
        count: None,
    },
];

#[derive(Clone)]
pub(crate) struct Codec {
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 3],
}
impl Codec {
    fn new(r: &Renderer) -> Self {
        let layout = r
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("BC7 preparation"),
                entries: &LAYOUT,
            });
        let pipeline_layout = r
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("BC7 preparation"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let shaders = [
            &native::BC7_TRYMODE456CS,
            &native::BC7_ENCODEBLOCKCS,
            &native::BC7_REFINEMODE56CS,
        ];
        let pipelines = std::array::from_fn(|i| {
            let compiled = shaders[i];
            let module = native::shader_module(&r.device, compiled);
            r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(compiled.name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(compiled.entry),
                compilation_options: Default::default(),
                cache: r.cache.as_ref(),
            })
        });
        Self { layout, pipelines }
    }
}

#[derive(Clone)]
pub(crate) struct Context {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) errors: crate::gpu_memory::Errors,
    codec: Codec,
    mips: ([wgpu::RenderPipeline; 3], wgpu::Sampler),
}
impl Context {
    pub(crate) fn new(r: &Renderer) -> Result<Self> {
        let codec = r.pipelines.bc7.get_or_init(|| Codec::new(r)).clone();
        let mips = materials::mip_resources(r).clone();
        r.errors.check()?;
        Ok(Self {
            device: r.device.clone(),
            queue: r.queue.clone(),
            errors: r.errors.clone(),
            codec,
            mips,
        })
    }
}

pub(crate) fn supported(r: &Renderer, w: u32, h: u32, bytes: usize) -> bool {
    let blocks = u64::from(w.div_ceil(EDGE)) * u64::from(h.div_ceil(EDGE));
    let blocks_per_group = u64::from(TEXTURE_BC7_GROUP_SIZE / (EDGE * EDGE));
    let workspace = blocks.div_ceil(blocks_per_group) * blocks_per_group * u64::from(BLOCK);
    let limits = r.device.limits();
    cfg!(target_arch = "x86_64")
        && r.adapter_info.device_type != wgpu::DeviceType::Cpu
        && workspace <= u64::from(limits.max_storage_buffer_binding_size)
        && workspace <= limits.max_buffer_size
        && bytes as u64 <= limits.max_buffer_size
}

pub(crate) fn texture(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    levels: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    crate::gpu_memory::texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("GPU prepared BC7 mips"),
            size: extent(w, h),
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    )
}

/// Shared GPU mip/BC7 workspace and reusable readback ring buffers.
#[derive(Default)]
pub(crate) struct Scratch {
    source: Option<(u32, u32, wgpu::Texture)>,
    workspace: Option<(u64, wgpu::Buffer, wgpu::Buffer)>,
    unused_input: Option<wgpu::Buffer>,
    readbacks: Vec<wgpu::Buffer>,
    source_readbacks: Vec<wgpu::Buffer>,
}
impl Scratch {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn submit(
        &mut self,
        ctx: &Context,
        texture: wgpu::Texture,
        pixels: Vec<u8>,
        w: u32,
        h: u32,
        format: wgpu::TextureFormat,
        bytes: usize,
        preserve_alpha: bool,
    ) -> Result<Pending> {
        let srgb = format.is_srgb();
        let levels = 32 - w.max(h).leading_zeros();
        let rgba_format = if srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        if self
            .source
            .as_ref()
            .is_none_or(|&(sw, sh, _)| (sw, sh) != (w, h))
        {
            let source = crate::gpu_memory::texture(
                &ctx.device,
                &wgpu::TextureDescriptor {
                    label: Some("BC7 source mips"),
                    size: extent(w, h),
                    mip_level_count: levels,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
                },
            );
            self.source = Some((w, h, source));
        }
        let source = &self.source.as_ref().unwrap().2;
        let blocks_per_group = TEXTURE_BC7_GROUP_SIZE / (EDGE * EDGE);
        let workspace = u64::from(w.div_ceil(EDGE)) * u64::from(h.div_ceil(EDGE));
        let workspace =
            workspace.div_ceil(u64::from(blocks_per_group)) * u64::from(blocks_per_group * BLOCK);
        if self
            .workspace
            .as_ref()
            .is_none_or(|&(cap, _, _)| cap < workspace)
        {
            let buffer = |label, usage| {
                crate::gpu_memory::buffer(
                    &ctx.device,
                    &wgpu::BufferDescriptor {
                        label: Some(label),
                        size: workspace,
                        usage,
                        mapped_at_creation: false,
                    },
                )
            };
            let candidate = buffer("BC7 candidate", wgpu::BufferUsages::STORAGE);
            let encoded = buffer(
                "BC7 encoded blocks",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            self.workspace = Some((workspace, candidate, encoded));
        }
        let (_, candidate, encoded) = self.workspace.as_ref().unwrap();
        // Metal retains the search kernel's unused input declaration.
        let unused_input = self.unused_input.get_or_insert_with(|| {
            crate::gpu_memory::buffer(
                &ctx.device,
                &wgpu::BufferDescriptor {
                    label: Some("BC7 unused search input"),
                    size: u64::from(BLOCK),
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                },
            )
        });
        let readback = self
            .readbacks
            .pop()
            .filter(|b| b.size() >= bytes as u64)
            .unwrap_or_else(|| {
                crate::gpu_memory::buffer(
                    &ctx.device,
                    &wgpu::BufferDescriptor {
                        label: Some("BC7 persistent payload"),
                        size: bytes as u64,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    },
                )
            });
        // Only alpha images need source mips on the CPU. Reuse the GPU mip chain
        // rather than filtering the same image again during SIMD refinement.
        let refine_alpha = preserve_alpha
            && cfg!(target_arch = "x86_64")
            && pixels.chunks_exact(4).any(|p| p[3] != pixels[3]);
        let mut mip_layout = Vec::new();
        let mut mip_bytes = 0;
        if refine_alpha {
            for mip in 1..levels {
                let mw = (w >> mip).max(1);
                let mh = (h >> mip).max(1);
                let pitch = (mw * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                    * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
                mip_layout.push((mip_bytes, pitch, mw, mh));
                mip_bytes += u64::from(pitch) * u64::from(mh);
            }
        }
        let source_readback = (mip_bytes != 0).then(|| {
            self.source_readbacks
                .pop()
                .filter(|b| b.size() >= mip_bytes)
                .unwrap_or_else(|| {
                    crate::gpu_memory::buffer(
                        &ctx.device,
                        &wgpu::BufferDescriptor {
                            label: Some("BC7 alpha refinement mips"),
                            size: mip_bytes,
                            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                            mapped_at_creation: false,
                        },
                    )
                })
        });
        ctx.errors.check()?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("prepare BC7 textures"),
            });
        let (mips, sampler) = &ctx.mips;
        materials::MipChain::new(&ctx.device,source,Some(rgba_format),&mips[usize::from(srgb)],sampler).encode(&mut encoder,None);
        if let Some(buffer) = &source_readback {
            for (i, &(offset, pitch, mw, mh)) in mip_layout.iter().enumerate() {
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: source,
                        mip_level: i as u32 + 1,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset,
                            bytes_per_row: Some(pitch),
                            rows_per_image: None,
                        },
                    },
                    extent(mw, mh),
                );
            }
        }
        let mut offset = 0;
        for mip in 0..levels {
            let mw = (w >> mip).max(1);
            let mh = (h >> mip).max(1);
            let columns = mw.div_ceil(EDGE);
            let rows = mh.div_ceil(EDGE);
            let count = (columns * rows).div_ceil(blocks_per_group) * blocks_per_group;
            let input = source.create_view(&wgpu::TextureViewDescriptor {
                format: Some(wgpu::TextureFormat::Rgba8Unorm),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            });
            // Modes 4/5/6 cover RGB and independent alpha without an exhaustive
            // partition search. Search/packing use four blocks/group; fitting uses
            // one block/thread without group barriers.
            for (step, pipeline) in ctx.codec.pipelines.iter().enumerate() {
                let (read, write) = if step == 0 {
                    (&*unused_input, candidate)
                } else {
                    (candidate, encoded)
                };
                let blocks_per_group = if step == 2 {
                    TEXTURE_BC7_GROUP_SIZE
                } else {
                    blocks_per_group
                };
                let total = count.div_ceil(blocks_per_group);
                let limit = ctx.device.limits().max_compute_workgroups_per_dimension;
                let mut group = 0;
                while group < total {
                    let groups = (total - group).min(limit);
                    let params = [
                        mw,
                        columns,
                        98,
                        0,
                        group * blocks_per_group,
                        count,
                        if preserve_alpha { 1.0f32.to_bits() } else { 0 },
                        u32::from(preserve_alpha),
                    ];
                    let uniform = init(
                        &ctx.device,
                        "BC7 range",
                        bytemuck::cast_slice(&params),
                        wgpu::BufferUsages::UNIFORM,
                    );
                    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("BC7 range"),
                        layout: &ctx.codec.layout,
                        entries: &[
                            entry(0, wgpu::BindingResource::TextureView(&input)),
                            entry(1, uniform.as_entire_binding()),
                            entry(2, read.as_entire_binding()),
                            entry(3, write.as_entire_binding()),
                        ],
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("BC7 mode search"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(groups, 1, 1);
                    group += groups;
                }
            }
            let row_bytes = columns * BLOCK;
            let length = u64::from(row_bytes) * u64::from(rows);
            encoder.copy_buffer_to_buffer(encoded, 0, &readback, offset, length);
            // Small mip rows need individual copies: packed cache bytes have no
            // artificial 256-byte row padding.
            if row_bytes % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT == 0 {
                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: encoded,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(row_bytes),
                            rows_per_image: None,
                        },
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    extent(columns * EDGE, rows * EDGE),
                );
            } else {
                for row in 0..rows {
                    encoder.copy_buffer_to_texture(
                        wgpu::TexelCopyBufferInfo {
                            buffer: encoded,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: u64::from(row * row_bytes),
                                bytes_per_row: None,
                                rows_per_image: None,
                            },
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture,
                            mip_level: mip,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: row * EDGE,
                                z: 0,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        extent(columns * EDGE, EDGE),
                    );
                }
            }
            offset += length;
        }
        ensure!(offset == bytes as u64, "GPU BC7 mip layout mismatch");
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            extent(w, h),
        );
        let submission = crate::command_finish::Segments::submit(&ctx.queue, encoder);
        ctx.errors.check()?;
        let (tx, rx) = std::sync::mpsc::channel();
        let encoded_tx = tx.clone();
        readback.map_async(wgpu::MapMode::Read, ..bytes as u64, move |result| {
            let _ = encoded_tx.send(result);
        });
        if let Some(buffer) = &source_readback {
            buffer.map_async(wgpu::MapMode::Read, ..mip_bytes, move |result| {
                let _ = tx.send(result);
            });
        }
        Ok(Pending {
            texture,
            readback,
            bytes,
            source_readback,
            mip_bytes,
            ready: rx,
            submission,
            w,
            h,
            levels,
            mip_layout,
            pixels: refine_alpha.then_some(pixels),
        })
    }
}

/// Submitted mip/BC7 work. The view is installed immediately; finish applies
/// alpha corrections and returns the persistent payload before any scene draw.
pub(crate) struct Pending {
    texture: wgpu::Texture,
    readback: wgpu::Buffer,
    bytes: usize,
    source_readback: Option<wgpu::Buffer>,
    mip_bytes: u64,
    ready: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    submission: wgpu::SubmissionIndex,
    w: u32,
    h: u32,
    levels: u32,
    mip_layout: Vec<(u64, u32, u32, u32)>,
    pixels: Option<Vec<u8>>,
}
impl Pending {
    pub(crate) fn finish(
        self,
        ctx: &Context,
        scratch: &mut Scratch,
    ) -> Result<(wgpu::TextureView, Vec<u8>)> {
        let Self {
            texture,
            readback,
            bytes: expected,
            source_readback,
            mip_bytes,
            ready: rx,
            submission,
            w,
            h,
            levels,
            mip_layout,
            pixels,
        } = self;
        crate::gpu_memory::wait(&ctx.device, &ctx.queue, &ctx.errors, Some(submission))?;
        for _ in 0..1 + usize::from(source_readback.is_some()) {
            rx.recv()??;
        }
        let bytes = readback.get_mapped_range(..expected as u64).to_vec();
        readback.unmap();
        scratch.readbacks.push(readback);
        #[cfg(target_arch = "x86_64")]
        let bytes = {
            let mut bytes = bytes;
            if let Some(pixels) = pixels.as_deref() {
                let mapped = source_readback
                    .as_ref()
                    .map(|buffer| buffer.get_mapped_range(..mip_bytes));
                let mut offset = 0;
                for mip in 0..levels {
                    let mw = (w >> mip).max(1);
                    let mh = (h >> mip).max(1);
                    let pw = mw.div_ceil(EDGE) * EDGE;
                    let ph = mh.div_ceil(EDGE) * EDGE;
                    let length = (pw / EDGE * (ph / EDGE) * BLOCK) as usize;
                    let packed;
                    let input = if mip == 0 {
                        pixels
                    } else {
                        let (at, pitch, _, _) = mip_layout[mip as usize - 1];
                        let mapped = mapped.as_ref().unwrap();
                        packed = mapped
                            [at as usize..(at + u64::from(pitch) * u64::from(mh)) as usize]
                            .chunks(pitch as usize)
                            .flat_map(|row| row[..(mw * 4) as usize].iter().copied())
                            .collect::<Vec<_>>();
                        &packed
                    };
                    if crate::compressed_textures::refine_bc7_alpha_level(
                        input,
                        mw,
                        mh,
                        &mut bytes[offset..offset + length],
                    ) {
                        ctx.queue.write_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture: &texture,
                                mip_level: mip,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            &bytes[offset..offset + length],
                            wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(pw / EDGE * BLOCK),
                                rows_per_image: None,
                            },
                            extent(pw, ph),
                        );
                    }
                    offset += length;
                }
                drop(mapped);
            }
            bytes
        };
        if let Some(buffer) = source_readback {
            buffer.unmap();
            scratch.source_readbacks.push(buffer);
        }
        Ok((texture.create_view(&Default::default()), bytes))
    }
}

#[cfg(test)]
fn encode(
    r: &Renderer,
    pixels: &[u8],
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
    bytes: usize,
    preserve_alpha: bool,
) -> Result<(wgpu::TextureView, Vec<u8>)> {
    let ctx = Context::new(r)?;
    let mut scratch = Scratch::default();
    let levels = 32 - w.max(h).leading_zeros();
    let tex = texture(&r.device, w, h, levels, format);
    scratch
        .submit(&ctx, tex, pixels.to_vec(), w, h, format, bytes, preserve_alpha)?
        .finish(&ctx, &mut scratch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn deferred_textures_match_serial_payloads() -> Result<()> {
        let r = crate::test_renderer()?;
        if !r
            .device
            .features()
            .contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
        {
            return Ok(());
        }
        let (w, h) = (24u32, 12u32);
        let levels = 32 - w.max(h).leading_zeros();
        let bytes = (0..levels)
            .map(|m| (w >> m).max(1).div_ceil(4) * (h >> m).max(1).div_ceil(4) * 16)
            .sum::<u32>() as usize;
        let inputs: Vec<(Vec<u8>, bool)> = (0..5u32)
            .map(|seed| {
                let pixels = (0..w * h)
                    .flat_map(|i| {
                        [
                            (i * 17 + seed * 9) as u8,
                            (i * 7) as u8,
                            127,
                            (i * 19 + seed) as u8,
                        ]
                    })
                    .collect();
                (pixels, seed % 2 == 0)
            })
            .collect();
        let format_of = |srgb| {
            if srgb {
                wgpu::TextureFormat::Bc7RgbaUnormSrgb
            } else {
                wgpu::TextureFormat::Bc7RgbaUnorm
            }
        };
        let serial = inputs
            .iter()
            .map(|(pixels, srgb)| {
                encode(&r, pixels, w, h, format_of(*srgb), bytes, true).map(|(_, data)| data)
            })
            .collect::<Result<Vec<_>>>()?;
        let ctx = Context::new(&r)?;
        let mut scratch = Scratch::default();
        let mut ring = Vec::new();
        for (i, (pixels, srgb)) in inputs.iter().enumerate() {
            if ring.len() >= crate::constants::TEXTURE_PREPARATION_IN_FLIGHT {
                let (oldest, expected): (Pending, Vec<u8>) = ring.remove(0);
                assert_eq!(oldest.finish(&ctx, &mut scratch)?.1, expected);
            }
            let format = format_of(*srgb);
            let tex = texture(&r.device, w, h, levels, format);
            let pending = scratch.submit(&ctx, tex, pixels.clone(), w, h, format, bytes, true)?;
            ring.push((pending, serial[i].clone()));
        }
        // Also verify reverse-order completion of the in-flight tail after dropping inputs.
        drop(inputs);
        for (pending, expected) in ring.into_iter().rev() {
            assert_eq!(pending.finish(&ctx, &mut scratch)?.1, expected);
        }
        assert!(scratch.readbacks.len() <= crate::constants::TEXTURE_PREPARATION_IN_FLIGHT);
        assert!(scratch.source_readbacks.len() <= crate::constants::TEXTURE_PREPARATION_IN_FLIGHT);
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn compute_encoder_handles_partial_mip_blocks() -> Result<()> {
        let r = crate::test_renderer()?;
        if !r
            .device
            .features()
            .contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
        {
            return Ok(());
        }
        for (w, h) in [(4u32, 4u32), (12, 28)] {
            let pixels: Vec<u8> = (0..w * h)
                .flat_map(|i| [(i * 17) as u8, (i * 7) as u8, 127, (i * 19) as u8])
                .collect();
            let levels = 32 - w.max(h).leading_zeros();
            let bytes = (0..levels)
                .map(|m| (w >> m).max(1).div_ceil(4) * (h >> m).max(1).div_ceil(4) * 16)
                .sum::<u32>() as usize;
            for srgb in [false, true] {
                let format = if srgb {
                    wgpu::TextureFormat::Bc7RgbaUnormSrgb
                } else {
                    wgpu::TextureFormat::Bc7RgbaUnorm
                };
                let (_, data) = encode(&r, &pixels, w, h, format, bytes, true)?;
                assert_eq!(data.len(), bytes);
                assert!(
                    data.chunks_exact(16).all(|block| block[0] != 0),
                    "BC7 block must select a mode"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn decoded_gradient_preserves_color_and_alpha() -> Result<()> {
        let r = crate::test_renderer()?;
        if !r
            .device
            .features()
            .contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
        {
            return Ok(());
        }
        let (w, h) = (64u32, 32u32);
        let gradient: Vec<u8> = (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    [
                        (x * 255 / (w - 1)) as u8,
                        (y * 255 / (h - 1)) as u8,
                        127,
                        ((x + y) * 255 / (w + h - 2)) as u8,
                    ]
                })
            })
            .collect();
        let ramp: Vec<u8> = (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    let ramp = ((x % 4 + (y % 4) * 4) * 17) as u8;
                    let v = if (x / 4) % 2 == 0 { ramp } else { 255 - ramp };
                    [v, 255 - v, v / 2, ((x + y) * 255 / (w + h - 2)) as u8]
                })
            })
            .collect();
        let plane: Vec<u8> = (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    let u = ((x % 4) * 85) as u8;
                    let v = ((y % 4) * 85) as u8;
                    let u = if (x / 4) % 2 == 0 { u } else { 255 - u };
                    let v = if (y / 4) % 2 == 0 { v } else { 255 - v };
                    [u, 255 - u, v, ((x + y) * 255 / (w + h - 2)) as u8]
                })
            })
            .collect();
        let bytes = (0..(32 - w.leading_zeros()))
            .map(|m| (w >> m).max(1).div_ceil(EDGE) * (h >> m).max(1).div_ceil(EDGE) * BLOCK)
            .sum::<u32>() as usize;
        for (pixels, require_mode6, require_mode5) in [
            (&gradient, false, false),
            (&ramp, true, false),
            (&plane, false, true),
        ] {
            for (srgb, preserve_alpha) in
                [(false, true), (true, true), (false, false), (true, false)]
            {
                if preserve_alpha
                    && (require_mode5 || (require_mode6 && !cfg!(target_arch = "x86_64")))
                {
                    continue;
                }
                let format = if srgb {
                    wgpu::TextureFormat::Bc7RgbaUnormSrgb
                } else {
                    wgpu::TextureFormat::Bc7RgbaUnorm
                };
                let (view, encoded) = encode(&r, pixels, w, h, format, bytes, preserve_alpha)?;
                if require_mode6 && !preserve_alpha {
                    let base_bytes = (w.div_ceil(EDGE) * h.div_ceil(EDGE) * BLOCK) as usize;
                    assert!(
                        encoded[..base_bytes]
                            .chunks_exact(BLOCK as usize)
                            .all(|b| b[0] & 0x7f == 0x40),
                        "PCA fit must select mode6 for the anticorrelated16-level ramp"
                    );
                }
                if require_mode5 {
                    let base_bytes = (w.div_ceil(EDGE) * h.div_ceil(EDGE) * BLOCK) as usize;
                    assert!(
                        encoded[..base_bytes]
                            .chunks_exact(BLOCK as usize)
                            .all(|b| b[0] & 0x3f == 0x20),
                        "PCA fit must select mode5 for the two-dimensional color plane"
                    );
                }
                let output = crate::gpu_memory::texture(
                    &r.device,
                    &wgpu::TextureDescriptor {
                        label: Some("decoded BC7 quality check"),
                        size: extent(w, h),
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    },
                );
                let readback = crate::gpu_memory::buffer(
                    &r.device,
                    &wgpu::BufferDescriptor {
                        label: Some("decoded BC7 pixels"),
                        size: pixels.len() as u64,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    },
                );
                let (pipelines, sampler) = materials::mip_resources(&r);
                let pipeline = &pipelines[0];
                let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sample encoded BC7"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        entry(0, wgpu::BindingResource::TextureView(&view)),
                        entry(1, wgpu::BindingResource::Sampler(sampler)),
                    ],
                });
                let target = output.create_view(&Default::default());
                let attachments = [Some(color_attachment(&target, wgpu::Color::BLACK))];
                let mut encoder = r.device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &attachments,
                        ..Default::default()
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.draw(0..3, 0..1);
                }
                encoder.copy_texture_to_buffer(
                    output.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: &readback,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(w * 4),
                            rows_per_image: None,
                        },
                    },
                    extent(w, h),
                );
                let submission = crate::command_finish::Segments::submit(&r.queue, encoder);
                let (tx, rx) = std::sync::mpsc::channel();
                readback.map_async(wgpu::MapMode::Read, .., move |result| {
                    let _ = tx.send(result);
                });
                r.wait(Some(submission))?;
                rx.recv()??;
                let mapped = readback.get_mapped_range(..);
                let error = mapped
                    .iter()
                    .zip(pixels)
                    .enumerate()
                    .map(|(i, (&got, &source))| {
                        let expected = if !preserve_alpha && i % 4 == 3 {
                            255.
                        } else if srgb && i % 4 != 3 {
                            texture_source::decode_srgb(source as f32 / 255.) * 255.
                        } else {
                            source as f32
                        };
                        (got as f32 - expected).abs()
                    })
                    .sum::<f32>()
                    / pixels.len() as f32;
                ensure!(
                    error < 4.,
                    "BC7 gradient mean channel error {error} exceeds 4/255 (sRGB={srgb})"
                );
                drop(mapped);
                readback.unmap();
            }
        }
        Ok(())
    }
}
