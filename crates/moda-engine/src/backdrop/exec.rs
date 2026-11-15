//! Backdrop effect-chain executor — the middle of the SEAMS §7 pass-split
//! contract: copy `region_px` from the resolved target, run
//! (coverage →) (refraction →) blur | pixelate | dither, composite back with
//! SDF/coverage alpha, sampling only already-drawn content.
//!
//! Faithful port of `WebGPUBackdropBlurRenderer.compositeShapeBackdropBlur`
//! + the `run*Pass` methods
//! (`renderers/webgpu/rendering/systems/effects/WebGPUBackdropBlurRenderer.ts`).
//! All WGSL comes from the extracted modules (`backdrop-*` in moda-shaders);
//! the composite blend state is consumed from the extraction manifest
//! (DECISION §1.3 — never hand-coded).

use anyhow::{anyhow, bail, Result};

use crate::seams::{BackdropOp, BackdropSplit, CoverageSource};

use super::kernel::{
    backdrop_blur_passes, generate_optimized_gaussian_kernel, BlurPass, MAX_BLUR_KERNEL_SIZE,
};

/// `MAX_DIFFUSION_DITHER_CELLS` (`WebGPUBackdropBlurRenderer.ts:212`).
pub const MAX_DIFFUSION_DITHER_CELLS: u64 = 1 << 18;

/// `resolveDitherRenderPattern` (`WebGPUBackdropBlurRenderer.ts:221-224`).
pub fn resolve_dither_render_pattern(pattern: u32, cell_count: u64) -> u32 {
    if pattern >= 2 && cell_count > MAX_DIFFUSION_DITHER_CELLS {
        1
    } else {
        pattern
    }
}

const SHAPE_TYPE_ROUNDED_RECT: u32 = 0;
const SHAPE_TYPE_ELLIPSE: u32 = 1;
const SHAPE_TYPE_COVERAGE: u32 = 2;

pub struct BackdropRenderer {
    blur: wgpu::RenderPipeline,
    progressive_blur: wgpu::RenderPipeline,
    refraction: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    /// The composite as a draw inside the frame's own single-sample pass (depth attached, never tested or written).
    composite_in_frame: wgpu::RenderPipeline,
    pixelate: wgpu::RenderPipeline,
    coverage: wgpu::RenderPipeline,
    dither_init: wgpu::ComputePipeline,
    dither_quantize: wgpu::ComputePipeline,
    dither_diffusion: wgpu::ComputePipeline,
    dither_render: wgpu::RenderPipeline,
    /// The backdrop filter kinds' pass (`backdrop-filter.wgsl`), compiled on first use: pages without a filter
    /// kind never build it.
    filter: std::cell::OnceCell<wgpu::RenderPipeline>,
    /// The adjustment kinds' pass (`backdrop-adjust.wgsl`) and its bind group layout, compiled on first use.
    adjust: std::cell::OnceCell<(wgpu::BindGroupLayout, wgpu::RenderPipeline)>,
    effect_bgl: wgpu::BindGroupLayout,
    shape_effect_bgl: wgpu::BindGroupLayout,
    coverage_bgl: wgpu::BindGroupLayout,
    dither_init_bgl: wgpu::BindGroupLayout,
    dither_buffer_bgl: wgpu::BindGroupLayout,
    dither_render_bgl: wgpu::BindGroupLayout,
    sampler_linear: wgpu::Sampler,
    sampler_nearest: wgpu::Sampler,
    /// 1×1 stand-in for the coverage slots on SDF shapes (never sampled).
    coverage_fallback_view: wgpu::TextureView,
    /// The owning engine's immutable frame color format. Every
    /// backdrop pipeline target and scratch texture uses it: the effect chain
    /// copies texels straight out of the frame/swapchain
    /// (`copy_texture_to_texture` requires compatible formats) and composites
    /// straight back into it.
    frame_format: wgpu::TextureFormat,
    /// Exact blend-mode splits (`blend.rs`).
    blend: super::blend::BlendExec,
}

fn shader(device: &wgpu::Device, id: &str) -> Result<wgpu::ShaderModule> {
    let src = moda_shaders::module_source(id)
        .ok_or_else(|| anyhow!("no extracted WGSL module '{id}'"))?;
    Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(id),
        source: wgpu::ShaderSource::Wgsl(src.source.into()),
    }))
}

/// Manifest blend → wgpu (only the composite pipeline carries one;
/// `one / one-minus-src-alpha` as extracted data).
fn manifest_blend(module_id: &str) -> Result<wgpu::BlendState> {
    let m = moda_shaders::manifest();
    let meta = m
        .modules
        .get(module_id)
        .ok_or_else(|| anyhow!("module '{module_id}' missing from manifest"))?;
    let blend = meta
        .pipelines
        .first()
        .and_then(|p| p.fragment.as_ref())
        .and_then(|f| f.targets.first())
        .and_then(|t| t.as_ref())
        .and_then(|t| t.blend.as_ref())
        .ok_or_else(|| anyhow!("module '{module_id}' has no extracted blend state"))?;
    let factor = |s: &str| -> Result<wgpu::BlendFactor> {
        Ok(match s {
            "one" => wgpu::BlendFactor::One,
            "zero" => wgpu::BlendFactor::Zero,
            "one-minus-src-alpha" => wgpu::BlendFactor::OneMinusSrcAlpha,
            "src-alpha" => wgpu::BlendFactor::SrcAlpha,
            other => bail!("unmapped blend factor '{other}'"),
        })
    };
    let op = |s: &str| -> Result<wgpu::BlendOperation> {
        Ok(match s {
            "add" => wgpu::BlendOperation::Add,
            other => bail!("unmapped blend operation '{other}'"),
        })
    };
    Ok(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: factor(&blend.color.src_factor)?,
            dst_factor: factor(&blend.color.dst_factor)?,
            operation: op(&blend.color.operation)?,
        },
        alpha: wgpu::BlendComponent {
            src_factor: factor(&blend.alpha.src_factor)?,
            dst_factor: factor(&blend.alpha.dst_factor)?,
            operation: op(&blend.alpha.operation)?,
        },
    })
}

impl BackdropRenderer {
    /// `frame_format` is the owning engine's immutable frame color format
    /// — see the field doc. `sample_count` is the frame's: blend splits
    /// composite into its multisample view (effects need 1).
    pub fn new(
        device: &wgpu::Device,
        frame_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Result<BackdropRenderer> {
        // Bind group layouts mirror the extraction manifest BGLs 338-343
        // ("Backdrop * BGL" — see manifest.json `bind_group_layouts`).
        let uniform_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let storage_entry =
            |binding: u32, vis: wgpu::ShaderStages, read_only: bool| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: vis,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };

        let frag = wgpu::ShaderStages::FRAGMENT;
        let comp = wgpu::ShaderStages::COMPUTE;

        let effect_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Backdrop Effect BGL"),
            entries: &[
                uniform_entry(0, frag),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: frag,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                texture_entry(2, frag),
            ],
        });
        let shape_effect_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Backdrop Shape Effect BGL"),
            entries: &[
                uniform_entry(0, frag),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: frag,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                texture_entry(2, frag),
                texture_entry(3, frag),
                texture_entry(4, frag),
            ],
        });
        let coverage_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Backdrop Coverage BGL"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX)],
        });
        let dither_init_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Backdrop Dither Init BGL"),
            entries: &[
                uniform_entry(0, comp),
                texture_entry(1, comp),
                storage_entry(2, comp, false),
            ],
        });
        let dither_buffer_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Backdrop Dither Buffer BGL"),
            entries: &[uniform_entry(0, comp), storage_entry(1, comp, false)],
        });
        let dither_render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Backdrop Dither Render BGL"),
            entries: &[
                uniform_entry(0, frag),
                storage_entry(1, frag, true),
                texture_entry(2, frag),
            ],
        });

        let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Backdrop Blur Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        // Nearest for pixelate — linear would smear adjacent grid blocks.
        let sampler_nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Backdrop Pixelate Nearest Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let layout = |label: &str, bgl: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                ..Default::default()
            })
        };

        let render_pipeline = |label: &str,
                               module: &wgpu::ShaderModule,
                               layout: &wgpu::PipelineLayout,
                               blend: Option<wgpu::BlendState>,
                               buffers: &[wgpu::VertexBufferLayout]|
         -> wgpu::RenderPipeline {
            // M5b (docs/productionization.md): env-gated compile stats.
            if std::env::var_os("MODA_PIPELINE_STATS").is_some() {
                eprintln!("moda-pipeline-stat: backdrop '{label}' (render)");
            }
            moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: frame_format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let blur_mod = shader(device, "backdrop-blur")?;
        let progressive_blur_mod = shader(device, "progressive-blur")?;
        let refraction_mod = shader(device, "backdrop-refraction")?;
        let composite_mod = shader(device, "backdrop-composite")?;
        let pixelate_mod = shader(device, "backdrop-pixelate")?;
        let coverage_mod = shader(device, "backdrop-coverage")?;
        let dither_init_mod = shader(device, "backdrop-dither-init")?;
        let dither_quantize_mod = shader(device, "backdrop-dither-quantize")?;
        let dither_diffusion_mod = shader(device, "backdrop-dither-diffusion")?;
        let dither_render_mod = shader(device, "backdrop-dither-render")?;

        let effect_layout = layout("Backdrop Blur Pipeline Layout", &effect_bgl);
        let shape_layout = layout("Backdrop Shape Pipeline Layout", &shape_effect_bgl);
        let coverage_layout = layout("Backdrop Coverage Pipeline Layout", &coverage_bgl);
        let dither_init_layout = layout("Backdrop Dither Init Pipeline Layout", &dither_init_bgl);
        let dither_buffer_layout =
            layout("Backdrop Dither Buffer Pipeline Layout", &dither_buffer_bgl);
        let dither_render_layout =
            layout("Backdrop Dither Render Pipeline Layout", &dither_render_bgl);

        let coverage_buffers = [wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            }],
        }];

        let compute_pipeline = |label: &str,
                                module: &wgpu::ShaderModule,
                                layout: &wgpu::PipelineLayout|
         -> wgpu::ComputePipeline {
            // M5b (docs/productionization.md): env-gated compile stats.
            if std::env::var_os("MODA_PIPELINE_STATS").is_some() {
                eprintln!("moda-pipeline-stat: backdrop '{label}' (compute)");
            }
            moda_text_draw::device::create_compute_pipeline(device, &wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module,
                entry_point: Some("cs"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let coverage_fallback = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Backdrop Coverage Fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: frame_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let coverage_fallback_view =
            coverage_fallback.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(BackdropRenderer {
            frame_format,
            blend: super::blend::BlendExec::new(device, frame_format, sample_count),
            blur: render_pipeline(
                "Backdrop Blur Pipeline",
                &blur_mod,
                &effect_layout,
                None,
                &[],
            ),
            progressive_blur: render_pipeline(
                "Backdrop Progressive Blur Pipeline",
                &progressive_blur_mod,
                &effect_layout,
                None,
                &[],
            ),
            refraction: render_pipeline(
                "Backdrop Refraction Pipeline",
                &refraction_mod,
                &shape_layout,
                None,
                &[],
            ),
            composite: render_pipeline(
                "Backdrop Composite Pipeline",
                &composite_mod,
                &shape_layout,
                Some(manifest_blend("backdrop-composite")?),
                &[],
            ),
            composite_in_frame: moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
                label: Some("Backdrop Composite In-Frame Pipeline"),
                layout: Some(&shape_layout),
                vertex: wgpu::VertexState {
                    module: &composite_mod,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &composite_mod,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: frame_format,
                        blend: Some(manifest_blend("backdrop-composite")?),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            }),
            pixelate: render_pipeline(
                "Backdrop Pixelate Pipeline",
                &pixelate_mod,
                &effect_layout,
                None,
                &[],
            ),
            coverage: render_pipeline(
                "Backdrop Coverage Pipeline",
                &coverage_mod,
                &coverage_layout,
                None,
                &coverage_buffers,
            ),
            dither_init: compute_pipeline(
                "Backdrop Dither Init Pipeline",
                &dither_init_mod,
                &dither_init_layout,
            ),
            dither_quantize: compute_pipeline(
                "Backdrop Dither Quantize Pipeline",
                &dither_quantize_mod,
                &dither_buffer_layout,
            ),
            dither_diffusion: compute_pipeline(
                "Backdrop Dither Diffusion Pipeline",
                &dither_diffusion_mod,
                &dither_buffer_layout,
            ),
            dither_render: render_pipeline(
                "Backdrop Dither Render Pipeline",
                &dither_render_mod,
                &dither_render_layout,
                None,
                &[],
            ),
            filter: std::cell::OnceCell::new(),
            adjust: std::cell::OnceCell::new(),
            effect_bgl,
            shape_effect_bgl,
            coverage_bgl,
            dither_init_bgl,
            dither_buffer_bgl,
            dither_render_bgl,
            sampler_linear,
            sampler_nearest,
            coverage_fallback_view,
        })
    }

    /// Forget blend-split state left by an earlier frame (see `BlendExec::reset`).
    pub fn begin_frame(&self) {
        self.blend.reset();
    }

    /// SEAMS §7 steps 2–3: copy `region_px` from the (resolved, sample_count-1)
    /// target, run the effect chain, composite back. The caller has already
    /// ended the main render pass and resumes it with LoadOp::Load after.
    /// `msaa_view` is the frame's multisample color view (`target_view` its
    /// resolve target), which only blend splits accept.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_split(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
        target_view: &wgpu::TextureView,
        msaa_view: Option<&wgpu::TextureView>,
        target_size: (u32, u32),
        split: &BackdropSplit,
    ) -> Result<()> {
        if split.op.is_blend_split() {
            return self.blend.execute(
                device,
                queue,
                encoder,
                target,
                target_view,
                msaa_view,
                target_size,
                split,
            );
        }
        if msaa_view.is_some() {
            bail!(
                "backdrop effect split for '{}' on a multisample target",
                split.node_id
            );
        }
        let composite = self.run_effect(device, queue, encoder, target, target_size, split)?;
        Self::draw_composites(encoder, target_view, std::slice::from_ref(&composite));
        Ok(())
    }

    /// An effect split's copy and effect chain, with its composite left to the caller: drawn as the first draw of
    /// the frame's next pass (`draw_composite_in_frame`) instead of a pass of its own. `None` for a blend split
    /// (the caller runs `execute_split`).
    pub fn execute_split_deferred(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
        target_size: (u32, u32),
        split: &BackdropSplit,
    ) -> Result<Option<PreparedComposite>> {
        if split.op.is_blend_split() {
            return Ok(None);
        }
        self.run_effect(device, queue, encoder, target, target_size, split).map(Some)
    }

    /// Draw a deferred composite inside the frame's pass (single-sample, with the frame's depth attachment): the
    /// same pipeline state as the composite pass but for a depth test that always passes and never writes. The
    /// scissor is back to the whole target after it.
    pub fn draw_composite_in_frame<'p>(
        &'p self,
        pass: &mut crate::gpu_profile::CountedPass<'p>,
        composite: &'p PreparedComposite,
        target_size: (u32, u32),
    ) {
        pass.set_pipeline(&self.composite_in_frame);
        let [x, y, w, h] = composite.scissor;
        pass.set_scissor_rect(x, y, w, h);
        pass.set_bind_group(0, &composite.bind_group, &[]);
        pass.draw(0..3, 0..1);
        pass.set_scissor_rect(0, 0, target_size.0, target_size.1);
    }

    /// Copy one effect split's backdrop out of the target and run its effect chain; the composite that draws the
    /// result back is returned (`draw_composites`).
    fn run_effect(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
        target_size: (u32, u32),
        split: &BackdropSplit,
    ) -> Result<PreparedComposite> {
        let [qx, qy, sample_w, sample_h] = split.region_px;
        if sample_w == 0 || sample_h == 0 {
            bail!("backdrop split for '{}' has empty region", split.node_id);
        }
        let max_dim = device.limits().max_texture_dimension_2d;
        if sample_w > max_dim || sample_h > max_dim {
            bail!("backdrop sample region exceeds maxTextureDimension2D");
        }

        let mk_tex = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: sample_w,
                    height: sample_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // The engine's frame format: the copy below reads
                // texels straight out of the frame/swapchain texture, and
                // `copy_texture_to_texture` requires compatible formats.
                format: self.frame_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };

        // ---- step 2: copy the already-drawn backdrop out of the target -----
        let source = mk_tex("Backdrop Blur Source");
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: target,
                mip_level: 0,
                origin: wgpu::Origin3d { x: qx, y: qy, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: sample_w,
                height: sample_h,
                depth_or_array_layers: 1,
            },
        );

        let node_rect_in_sample = [
            split.node_rect_px[0] - qx as f32,
            split.node_rect_px[1] - qy as f32,
            split.node_rect_px[2],
            split.node_rect_px[3],
        ];
        let (shape_type, corner_radii) = match &split.coverage {
            CoverageSource::Rect { radii, .. } => (SHAPE_TYPE_ROUNDED_RECT, *radii),
            CoverageSource::Ellipse { .. } => (SHAPE_TYPE_ELLIPSE, [0.0; 4]),
            CoverageSource::Geometry(_) => (SHAPE_TYPE_COVERAGE, [0.0; 4]),
        };

        // ---- coverage masks for arbitrary silhouettes -----------------------
        let mut sharp_view = self.coverage_fallback_view.clone();
        let mut field_view = self.coverage_fallback_view.clone();
        if let CoverageSource::Geometry(geo) = &split.coverage {
            if geo.positions.len() < 6 {
                bail!("coverage geometry for '{}' is degenerate", split.node_id);
            }
            let coverage_tex = mk_tex("Backdrop Coverage Mask");
            self.run_coverage_pass(
                device,
                queue,
                encoder,
                &coverage_tex,
                &geo.positions,
                (qx, qy),
                (sample_w, sample_h),
            );
            let coverage_view = coverage_tex.create_view(&Default::default());
            sharp_view = coverage_view.clone();
            field_view = coverage_view;

            // Glass with refraction / dispersion / rim needs the blurred
            // coverage field (`coverageGlassNeedsField`).
            if let BackdropOp::Glass {
                depth_px,
                refraction_px,
                dispersion_px,
                light_intensity,
                ..
            } = split.op
            {
                if refraction_px > 0.25 || dispersion_px > 1e-4 || light_intensity > 1e-4 {
                    // A shaping field for the refraction, not visible content: it
                    // keeps the single production kernel (its falloff is tuned
                    // to it), not the full-σ ladder the visible blur uses.
                    let field_kernel =
                        generate_optimized_gaussian_kernel((depth_px as f64).min(64.0).max(1.0));
                    let temp = mk_tex("Backdrop Coverage Field Temp");
                    let blurred = mk_tex("Backdrop Coverage Field");
                    self.run_blur_pass(
                        device,
                        queue,
                        encoder,
                        &sharp_view,
                        &temp,
                        sample_w,
                        sample_h,
                        &field_kernel,
                        0,
                    );
                    self.run_blur_pass(
                        device,
                        queue,
                        encoder,
                        &temp.create_view(&Default::default()),
                        &blurred,
                        sample_w,
                        sample_h,
                        &field_kernel,
                        1,
                    );
                    field_view = blurred.create_view(&Default::default());
                }
            }
        }

        // ---- step 3: effect chain ------------------------------------------
        // Each pass reads `working` and draws into a texture of the same size; the texture the pass before last drew
        // (`spare`) is free by then and takes the next pass's output, so a chain of any length makes at most two of
        // these textures, not one per pass: at 800 % a frosted pane's sample region is the whole view, and a blur at
        // its reach cap is three separable passes, seven view-sized textures a frame before. The passes, their order
        // and their inputs are unchanged, so are the pixels (passes encoded in order see the writes before them).
        let mut working = source;
        let mut spare: Option<wgpu::Texture> = None;
        let take = |spare: &mut Option<wgpu::Texture>, label: &str| {
            spare.take().unwrap_or_else(|| mk_tex(label))
        };
        if let BackdropOp::Glass {
            refraction_px,
            depth_px,
            refraction_style,
            ..
        } = split.op
        {
            if refraction_px > 0.25 {
                let refracted = take(&mut spare, "Backdrop Glass Refraction");
                self.run_refraction_pass(
                    device,
                    queue,
                    encoder,
                    &working.create_view(&Default::default()),
                    &refracted,
                    node_rect_in_sample,
                    corner_radii,
                    shape_type,
                    split,
                    refraction_px,
                    depth_px,
                    refraction_style,
                    &sharp_view,
                    &field_view,
                );
                spare = Some(std::mem::replace(&mut working, refracted));
            }
        }

        match &split.op {
            BackdropOp::Pixelate { cell_px } => {
                let pixelated = take(&mut spare, "Backdrop Pixelate Result");
                self.run_pixelate_pass(
                    device,
                    queue,
                    encoder,
                    &working.create_view(&Default::default()),
                    &pixelated,
                    node_rect_in_sample,
                    split,
                    *cell_px,
                );
                working = pixelated;
            }
            BackdropOp::Dither {
                cell_px,
                pattern,
                brightness,
                contrast,
            } => {
                let dithered = take(&mut spare, "Backdrop Dither Result");
                let did = self.run_dither_pass(
                    device,
                    queue,
                    encoder,
                    &working.create_view(&Default::default()),
                    &dithered,
                    sample_w,
                    sample_h,
                    *cell_px,
                    *pattern,
                    *brightness,
                    *contrast,
                )?;
                if did {
                    working = dithered;
                }
            }
            BackdropOp::Blur {
                radius_px,
                ramp: Some(ramp),
                ..
            } => {
                // The ramp is in target px; the sample texture starts at (qx, qy).
                let blur = crate::seams::BlurFilter::Progressive {
                    start_radius_px: ramp.start_radius_px,
                    end_radius_px: *radius_px,
                    start_px: [ramp.start_px[0] - qx as f32, ramp.start_px[1] - qy as f32],
                    end_px: [ramp.end_px[0] - qx as f32, ramp.end_px[1] - qy as f32],
                };
                for direction in 0..2 {
                    let out = take(
                        &mut spare,
                        if direction == 0 {
                            "Backdrop Progressive Blur Temp"
                        } else {
                            "Backdrop Progressive Blur Result"
                        },
                    );
                    let uniform = crate::layer_blur::exec::progressive_blur_uniform(
                        &blur,
                        direction,
                        [sample_w as f32, sample_h as f32],
                    );
                    let ubuf = self.uniform_buffer(
                        device,
                        queue,
                        "Backdrop Progressive Blur Uniform Buffer",
                        bytemuck::cast_slice(&uniform),
                    );
                    let bind_group = self.effect_bind_group(
                        device,
                        "Backdrop Progressive Blur Bind Group",
                        &ubuf,
                        &self.sampler_linear,
                        &working.create_view(&Default::default()),
                    );
                    Self::fullscreen_pass(
                        encoder,
                        "Backdrop Progressive Blur Pass",
                        &out.create_view(&Default::default()),
                        &self.progressive_blur,
                        &bind_group,
                    );
                    spare = Some(std::mem::replace(&mut working, out));
                }
            }
            BackdropOp::Blur {
                radius_px,
                ramp: None,
                ..
            }
            | BackdropOp::Glass {
                blur_px: radius_px, ..
            } => {
                if *radius_px > 0.5 {
                    // Separable Gaussian, horizontal then vertical, one or more
                    // passes each (`backdrop_blur_passes`: large radii no longer
                    // truncate at the 31-tap kernel).
                    for pass in backdrop_blur_passes(*radius_px as f64) {
                        for direction in [0, 1] {
                            let out = take(
                                &mut spare,
                                if direction == 0 {
                                    "Backdrop Blur Temp"
                                } else {
                                    "Backdrop Blur Result"
                                },
                            );
                            self.run_blur_pass(
                                device,
                                queue,
                                encoder,
                                &working.create_view(&Default::default()),
                                &out,
                                sample_w,
                                sample_h,
                                &pass,
                                direction,
                            );
                            spare = Some(std::mem::replace(&mut working, out));
                        }
                    }
                }
            }
            BackdropOp::Adjust(adjust) => {
                let adjusted = take(&mut spare, "Backdrop Adjust Result");
                self.run_adjust_pass(device, queue, encoder, &working.create_view(&Default::default()), &adjusted, adjust)?;
                working = adjusted;
            }
            BackdropOp::Filter(filter) => {
                let filtered = take(&mut spare, "Backdrop Filter Result");
                self.run_filter_pass(
                    device,
                    queue,
                    encoder,
                    &working.create_view(&Default::default()),
                    &filtered,
                    node_rect_in_sample,
                    split,
                    filter,
                )?;
                working = filtered;
            }
            // Dispatched to the blend executor at the top of this function.
            BackdropOp::BlendBegin
            | BackdropOp::GroupBegin
            | BackdropOp::BlendEnd { .. }
            | BackdropOp::StackEnd { .. } => {
                unreachable!()
            }
        }

        // ---- composite back over the target (the caller draws it) -----------
        Ok(self.prepare_composite(
            device,
            queue,
            &working.create_view(&Default::default()),
            corner_radii,
            shape_type,
            target_size,
            split,
            &sharp_view,
            &field_view,
        ))
    }

    fn uniform_buffer(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        bytes: &[u8],
    ) -> wgpu::Buffer {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytes);
        buf
    }

    fn effect_bind_group(
        &self,
        device: &wgpu::Device,
        label: &str,
        uniform: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
        texture: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.effect_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(texture),
                },
            ],
        })
    }

    fn shape_effect_bind_group(
        &self,
        device: &wgpu::Device,
        label: &str,
        uniform: &wgpu::Buffer,
        texture: &wgpu::TextureView,
        sharp: &wgpu::TextureView,
        field: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.shape_effect_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(texture),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(sharp),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(field),
                },
            ],
        })
    }

    /// Fullscreen pass into `out` with clear-to-transparent.
    fn fullscreen_pass(
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        out: &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
        bind_group: &wgpu::BindGroup,
    ) {
        let profile_ts = crate::gpu_profile::pass_timestamps();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: out,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// `renderCoveragePass` — rasterize target-px fill triangles into the
    /// sample-aligned coverage mask (white inside, 0 outside).
    #[allow(clippy::too_many_arguments)]
    fn run_coverage_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        coverage_tex: &wgpu::Texture,
        vertices_px: &[f32],
        sample_origin: (u32, u32),
        sample_size: (u32, u32),
    ) {
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Backdrop Coverage Vertex Buffer"),
            size: (vertices_px.len() * 4) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(vertices_px));
        let uniform_data: [f32; 4] = [
            sample_origin.0 as f32,
            sample_origin.1 as f32,
            1.0 / sample_size.0 as f32,
            1.0 / sample_size.1 as f32,
        ];
        let ubuf = self.uniform_buffer(
            device,
            queue,
            "Backdrop Coverage Uniform Buffer",
            bytemuck::cast_slice(&uniform_data),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Backdrop Coverage Bind Group"),
            layout: &self.coverage_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        let view = coverage_tex.create_view(&Default::default());
        let profile_ts = crate::gpu_profile::pass_timestamps();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Backdrop Coverage Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.coverage);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, vbuf.slice(..));
        pass.draw(0..(vertices_px.len() / 2) as u32, 0..1);
    }

    /// `runBlurPass` — one direction of the separable Gaussian.
    #[allow(clippy::too_many_arguments)]
    fn run_blur_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::Texture,
        width: u32,
        height: u32,
        kernel: &BlurPass,
        direction: u32,
    ) {
        let kernel_size = kernel.size.min(MAX_BLUR_KERNEL_SIZE);
        let mut uniform = vec![0f32; 4 + MAX_BLUR_KERNEL_SIZE * 4];
        uniform[0] = direction as f32;
        uniform[1] = kernel_size as f32;
        uniform[2] = width as f32;
        uniform[3] = height as f32;
        for i in 0..kernel_size {
            uniform[4 + i * 4] = kernel.weights.get(i).copied().unwrap_or(0.0);
            uniform[4 + i * 4 + 1] = kernel.offsets.get(i).copied().unwrap_or(0.0);
        }
        let ubuf = self.uniform_buffer(
            device,
            queue,
            "Backdrop Blur Uniform Buffer",
            bytemuck::cast_slice(&uniform),
        );
        let bind_group = self.effect_bind_group(
            device,
            "Backdrop Blur Bind Group",
            &ubuf,
            &self.sampler_linear,
            input,
        );
        let view = output.create_view(&Default::default());
        Self::fullscreen_pass(
            encoder,
            if direction == 0 {
                "Backdrop Blur Horizontal Pass"
            } else {
                "Backdrop Blur Vertical Pass"
            },
            &view,
            &self.blur,
            &bind_group,
        );
    }

    /// `runRefractionPass`.
    #[allow(clippy::too_many_arguments)]
    fn run_refraction_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::Texture,
        node_rect_in_sample: [f32; 4],
        corner_radii: [f32; 4],
        shape_type: u32,
        split: &BackdropSplit,
        refraction_px: f32,
        depth_px: f32,
        refraction_mode: u32,
        sharp: &wgpu::TextureView,
        field: &wgpu::TextureView,
    ) {
        // 20 × 4 B: nodeRect, cornerRadii, params, shapeParams(u32), shapeTransform.
        let mut bytes = [0u8; 80];
        let f = |bytes: &mut [u8], idx: usize, v: f32| {
            bytes[idx * 4..idx * 4 + 4].copy_from_slice(&v.to_le_bytes())
        };
        let u = |bytes: &mut [u8], idx: usize, v: u32| {
            bytes[idx * 4..idx * 4 + 4].copy_from_slice(&v.to_le_bytes())
        };
        for (i, v) in node_rect_in_sample.iter().enumerate() {
            f(&mut bytes, i, *v);
        }
        for (i, v) in corner_radii.iter().enumerate() {
            f(&mut bytes, 4 + i, *v);
        }
        f(&mut bytes, 8, refraction_px);
        f(&mut bytes, 9, depth_px);
        f(&mut bytes, 10, refraction_mode as f32);
        f(&mut bytes, 11, 0.0);
        u(&mut bytes, 12, shape_type);
        f(&mut bytes, 16, split.rotation_rad);
        f(&mut bytes, 17, split.local_half_size_px[0]);
        f(&mut bytes, 18, split.local_half_size_px[1]);
        f(&mut bytes, 19, split.local_frame_y_sign);
        let ubuf = self.uniform_buffer(device, queue, "Backdrop Refraction Uniform Buffer", &bytes);
        let bind_group = self.shape_effect_bind_group(
            device,
            "Backdrop Refraction Bind Group",
            &ubuf,
            input,
            sharp,
            field,
        );
        let view = output.create_view(&Default::default());
        Self::fullscreen_pass(
            encoder,
            "Backdrop Refraction Pass",
            &view,
            &self.refraction,
            &bind_group,
        );
    }

    /// `runPixelatePass` (nearest sampler — see production note).
    #[allow(clippy::too_many_arguments)]
    fn run_pixelate_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::Texture,
        node_rect_in_sample: [f32; 4],
        split: &BackdropSplit,
        pixel_size_px: f32,
    ) {
        let mut uniform = [0f32; 12];
        uniform[..4].copy_from_slice(&node_rect_in_sample);
        uniform[4] = split.rotation_rad;
        uniform[5] = split.local_half_size_px[0];
        uniform[6] = split.local_half_size_px[1];
        uniform[7] = split.local_frame_y_sign;
        uniform[8] = pixel_size_px;
        let ubuf = self.uniform_buffer(
            device,
            queue,
            "Backdrop Pixelate Uniform Buffer",
            bytemuck::cast_slice(&uniform),
        );
        let bind_group = self.effect_bind_group(
            device,
            "Backdrop Pixelate Bind Group",
            &ubuf,
            &self.sampler_nearest,
            input,
        );
        let view = output.create_view(&Default::default());
        Self::fullscreen_pass(
            encoder,
            "Backdrop Pixelate Pass",
            &view,
            &self.pixelate,
            &bind_group,
        );
    }

    /// One backdrop filter kind (`backdrop-filter.wgsl`): the node's rect and local frame from the split, the
    /// kind's uniforms from the planner.
    #[allow(clippy::too_many_arguments)]
    fn run_filter_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::Texture,
        node_rect_in_sample: [f32; 4],
        split: &BackdropSplit,
        filter: &crate::seams::BackdropFilter,
    ) -> Result<()> {
        let pipeline = match self.filter.get() {
            Some(p) => p,
            None => {
                let module = shader(device, "backdrop-filter")?;
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Backdrop Filter Pipeline Layout"),
                    bind_group_layouts: &[Some(&self.effect_bgl)],
                    ..Default::default()
                });
                let pipeline = moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
                    label: Some("Backdrop Filter Pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: self.frame_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });
                self.filter.get_or_init(|| pipeline)
            }
        };
        let mut uniform = [0f32; 8 + crate::seams::BACKDROP_FILTER_UNIFORM_FLOATS];
        uniform[..4].copy_from_slice(&node_rect_in_sample);
        uniform[4] = split.rotation_rad;
        uniform[5] = split.local_half_size_px[0];
        uniform[6] = split.local_half_size_px[1];
        uniform[7] = split.local_frame_y_sign;
        uniform[8..].copy_from_slice(&filter.uniforms);
        // The clip box (p2) and the rounded clip (p3, p4) in sample px.
        let (qx, qy) = (split.region_px[0] as f32, split.region_px[1] as f32);
        uniform[8 + 12] -= qx;
        uniform[8 + 13] -= qy;
        if let Some((rect, radii)) = split.clip_rrect_px {
            uniform[8 + 16..8 + 20].copy_from_slice(&[rect[0] - qx, rect[1] - qy, rect[2], rect[3]]);
            uniform[8 + 20..8 + 24].copy_from_slice(&radii);
        }
        let ubuf = self.uniform_buffer(
            device,
            queue,
            "Backdrop Filter Uniform Buffer",
            bytemuck::cast_slice(&uniform),
        );
        let bind_group = self.effect_bind_group(
            device,
            "Backdrop Filter Bind Group",
            &ubuf,
            &self.sampler_linear,
            input,
        );
        Self::fullscreen_pass(
            encoder,
            "Backdrop Filter Pass",
            &output.create_view(&Default::default()),
            pipeline,
            &bind_group,
        );
        Ok(())
    }

    /// One adjustment kind (`backdrop-adjust.wgsl`): the planner's uniforms and moda-color lookup texture. The
    /// kernel reads its input with `textureLoad` (texel for texel, no filtering), so a table lookup sees the exact
    /// 8-bit value.
    fn run_adjust_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::Texture,
        adjust: &crate::seams::BackdropAdjust,
    ) -> Result<()> {
        let (bgl, pipeline) = match self.adjust.get() {
            Some(p) => p,
            None => {
                let frag = wgpu::ShaderStages::FRAGMENT;
                let texture = |binding: u32, sample_type: wgpu::TextureSampleType| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: frag,
                    ty: wgpu::BindingType::Texture {
                        sample_type,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                };
                let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Backdrop Adjust BGL"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: frag,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        texture(1, wgpu::TextureSampleType::Float { filterable: false }),
                        texture(2, wgpu::TextureSampleType::Uint),
                    ],
                });
                let module = shader(device, "backdrop-adjust")?;
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Backdrop Adjust Pipeline Layout"),
                    bind_group_layouts: &[Some(&bgl)],
                    ..Default::default()
                });
                let pipeline = moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
                    label: Some("Backdrop Adjust Pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState { format: self.frame_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                    }),
                    primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });
                self.adjust.get_or_init(|| (bgl, pipeline))
            }
        };
        let lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Backdrop Adjust LUT"),
            size: wgpu::Extent3d { width: 256, height: 2, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &lut, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &adjust.lut,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256 * 4), rows_per_image: Some(2) },
            wgpu::Extent3d { width: 256, height: 2, depth_or_array_layers: 1 },
        );
        let ubuf = self.uniform_buffer(device, queue, "Backdrop Adjust Uniform Buffer", bytemuck::cast_slice(&adjust.uniforms));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Backdrop Adjust Bind Group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&lut.create_view(&Default::default())) },
            ],
        });
        Self::fullscreen_pass(encoder, "Backdrop Adjust Pass", &output.create_view(&Default::default()), pipeline, &bind_group);
        Ok(())
    }

    /// `runDitherPass` — init (parallel) → quantize | serial diffusion →
    /// render. The diffusion pass is `@workgroup_size(1)`, dispatched once:
    /// kept serial on purpose (error diffusion is order-dependent).
    #[allow(clippy::too_many_arguments)]
    fn run_dither_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::Texture,
        width: u32,
        height: u32,
        cell_px: f32,
        pattern: u32,
        brightness: f32,
        contrast: f32,
    ) -> Result<bool> {
        // Snap near-integer pixel sizes caused by f32 projection noise
        // (BackdropBlurRenderer:1914-1921).
        let rounded = (cell_px as f64).round();
        let pixel_size_px = if ((cell_px as f64) - rounded).abs() < 1e-3 {
            rounded
        } else {
            cell_px as f64
        }
        .max(1.0);
        let grid_w = ((width as f64 / pixel_size_px).ceil() as u64).max(1);
        let grid_h = ((height as f64 / pixel_size_px).ceil() as u64).max(1);
        let cell_count = grid_w * grid_h;
        let buffer_bytes = cell_count * 4;
        let max_storage = device.limits().max_storage_buffer_binding_size as u64;
        if buffer_bytes > max_storage {
            return Ok(false);
        }
        let render_pattern = resolve_dither_render_pattern(pattern, cell_count);
        let use_diffusion = render_pattern >= 2;

        let mut bytes = [0u8; 32];
        bytes[0..4].copy_from_slice(&(pixel_size_px as f32).to_le_bytes());
        bytes[4..8].copy_from_slice(&(brightness / 255.0).to_le_bytes());
        bytes[8..12].copy_from_slice(&(contrast / 100.0).to_le_bytes());
        bytes[12..16].copy_from_slice(&(render_pattern as f32).to_le_bytes());
        bytes[16..20].copy_from_slice(&width.to_le_bytes());
        bytes[20..24].copy_from_slice(&height.to_le_bytes());
        bytes[24..28].copy_from_slice(&(grid_w as u32).to_le_bytes());
        bytes[28..32].copy_from_slice(&(grid_h as u32).to_le_bytes());
        let ubuf = self.uniform_buffer(device, queue, "Backdrop Dither Uniform Buffer", &bytes);

        let gray = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Backdrop Dither Gray Buffer"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let init_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Backdrop Dither Init Bind Group"),
            layout: &self.dither_init_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gray.as_entire_binding(),
                },
            ],
        });
        let buffer_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Backdrop Dither Buffer Bind Group"),
            layout: &self.dither_buffer_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gray.as_entire_binding(),
                },
            ],
        });

        {
            let profile_ts = crate::gpu_profile::pass_timestamps();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Backdrop Dither Init Pass"),
                timestamp_writes: crate::gpu_profile::compute_writes(&profile_ts),
            });
            pass.set_pipeline(&self.dither_init);
            pass.set_bind_group(0, &init_bg, &[]);
            pass.dispatch_workgroups(grid_w.div_ceil(8) as u32, grid_h.div_ceil(8) as u32, 1);
        }
        {
            let profile_ts = crate::gpu_profile::pass_timestamps();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(if use_diffusion {
                    "Backdrop Dither Diffusion Pass"
                } else {
                    "Backdrop Dither Quantize Pass"
                }),
                timestamp_writes: crate::gpu_profile::compute_writes(&profile_ts),
            });
            pass.set_pipeline(if use_diffusion {
                &self.dither_diffusion
            } else {
                &self.dither_quantize
            });
            pass.set_bind_group(0, &buffer_bg, &[]);
            if use_diffusion {
                pass.dispatch_workgroups(1, 1, 1);
            } else {
                pass.dispatch_workgroups(grid_w.div_ceil(8) as u32, grid_h.div_ceil(8) as u32, 1);
            }
        }

        let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Backdrop Dither Render Bind Group"),
            layout: &self.dither_render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gray.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(input),
                },
            ],
        });
        let view = output.create_view(&Default::default());
        Self::fullscreen_pass(
            encoder,
            "Backdrop Dither Render Pass",
            &view,
            &self.dither_render,
            &render_bg,
        );
        Ok(true)
    }

    /// `runCompositePass` — SDF/coverage-masked blend of the effect texture
    /// back over the target, scissored to the visible node rect.
    #[allow(clippy::too_many_arguments)]
    fn prepare_composite(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        effect: &wgpu::TextureView,
        corner_radii: [f32; 4],
        shape_type: u32,
        target_size: (u32, u32),
        split: &BackdropSplit,
        sharp: &wgpu::TextureView,
        field: &wgpu::TextureView,
    ) -> PreparedComposite {
        let (is_glass, depth_px, light_intensity, light_angle, dispersion_px, splay, rim_width_px) =
            match split.op {
                BackdropOp::Glass {
                    depth_px,
                    light_intensity,
                    light_angle_rad,
                    dispersion_px,
                    light_splay,
                    rim_width_px,
                    ..
                } => (
                    1.0f32,
                    depth_px,
                    light_intensity,
                    light_angle_rad,
                    dispersion_px,
                    light_splay,
                    rim_width_px,
                ),
                _ => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            };

        // 36 × 4 B: nodeRect, sampleRect, cornerRadii, effectParams,
        // glassParams, shapeParams(u32), shapeTransform, clipRect, clipRadii.
        let mut bytes = [0u8; 144];
        let f = |bytes: &mut [u8], idx: usize, v: f32| {
            bytes[idx * 4..idx * 4 + 4].copy_from_slice(&v.to_le_bytes())
        };
        let u = |bytes: &mut [u8], idx: usize, v: u32| {
            bytes[idx * 4..idx * 4 + 4].copy_from_slice(&v.to_le_bytes())
        };
        for (i, v) in split.node_rect_px.iter().enumerate() {
            f(&mut bytes, i, *v);
        }
        // sampleRect = the QUANTIZED sample rect (composite uv space).
        f(&mut bytes, 4, split.region_px[0] as f32);
        f(&mut bytes, 5, split.region_px[1] as f32);
        f(&mut bytes, 6, split.region_px[2] as f32);
        f(&mut bytes, 7, split.region_px[3] as f32);
        for (i, v) in corner_radii.iter().enumerate() {
            f(&mut bytes, 8 + i, *v);
        }
        f(&mut bytes, 12, is_glass);
        f(&mut bytes, 13, depth_px);
        f(&mut bytes, 14, light_intensity);
        f(&mut bytes, 15, light_angle);
        f(&mut bytes, 16, dispersion_px);
        f(&mut bytes, 17, splay);
        f(&mut bytes, 18, rim_width_px);
        f(&mut bytes, 19, 0.0);
        u(&mut bytes, 20, shape_type);
        f(&mut bytes, 24, split.rotation_rad);
        f(&mut bytes, 25, split.local_half_size_px[0]);
        f(&mut bytes, 26, split.local_half_size_px[1]);
        f(&mut bytes, 27, split.local_frame_y_sign);
        if let Some((rect, radii)) = split.clip_rrect_px {
            for (i, v) in rect.iter().chain(radii.iter()).enumerate() {
                f(&mut bytes, 28 + i, *v);
            }
        }
        let ubuf = self.uniform_buffer(device, queue, "Backdrop Composite Uniform Buffer", &bytes);
        let bind_group = self.shape_effect_bind_group(
            device,
            "Backdrop Composite Bind Group",
            &ubuf,
            effect,
            sharp,
            field,
        );

        PreparedComposite {
            bind_group,
            scissor: composite_scissor(split, target_size),
            pipeline: self.composite.clone(),
        }
    }

    /// Draw prepared composites over the target in one pass, in order.
    fn draw_composites(
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        composites: &[PreparedComposite],
    ) {
        if composites.is_empty() {
            return;
        }
        let profile_ts = crate::gpu_profile::pass_timestamps();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Backdrop Composite Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        for c in composites {
            pass.set_pipeline(&c.pipeline);
            let [x, y, w, h] = c.scissor;
            pass.set_scissor_rect(x, y, w, h);
            pass.set_bind_group(0, &c.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// An effect split's result, ready to draw back over the target (`BackdropRenderer::draw_composites`).
pub struct PreparedComposite {
    bind_group: wgpu::BindGroup,
    scissor: [u32; 4],
    pipeline: wgpu::RenderPipeline,
}

/// The composite's scissor rect in target pixels (BackdropBlurRenderer:2109-2114): what an effect split's
/// composite can change.
pub fn composite_scissor(split: &BackdropSplit, target_size: (u32, u32)) -> [u32; 4] {
    let v = split.visible_rect_px;
    let sx = (v[0].floor() as f64).max(0.0);
    let sy = (v[1].floor() as f64).max(0.0);
    let sw = ((target_size.0 as f64 - sx).min(v[2].ceil() as f64)).max(1.0);
    let sh = ((target_size.1 as f64 - sy).min(v[3].ceil() as f64)).max(1.0);
    [sx as u32, sy as u32, sw as u32, sh as u32]
}
