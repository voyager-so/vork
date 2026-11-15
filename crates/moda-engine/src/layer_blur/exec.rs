//! Exec side of the layer filters (layer blur, noise, texture): the
//! isolation-capture mini plan, the filter passes over the capture, and the
//! engine-owned composite pipeline the main pass draws at the node's z.
//!
//! GPU flow per `LayerBlurGroup` (driven from `render_plan_to_texture_impl`'s
//! pre-pass, BEFORE the main scene pass — the mask-atlas slot):
//!   1. `build_capture_plan` → a mini `CoveragePlan` holding just the node's
//!      diverted items; the caller renders it through the ordinary plan
//!      executor (own encoder + submit), which handles pattern/text/shader
//!      resources exactly like any page render.
//!   2. `filter_capture` → the group's `LayerFilters` in order, each pass
//!      reading the previous one's output: texture (`layer-filter-texture`),
//!      blur (`backdrop-blur` passes — backdrop's ladder past σ = 5 — or two
//!      `progressive-blur` passes, H then V),
//!      then every noise (`layer-filter-noise`), then the feather
//!      (`layer-filter-shadow` `fs_feather`). Own encoder + submit; queue
//!      submission order makes the capture-read safe even though the capture
//!      target is a size-keyed shared texture.
//!   3. The main pass binds the result through `composite_bind_group` and
//!      draws one quad via `composite_pipeline` in the node's z bucket.

use anyhow::{anyhow, Result};

use crate::backdrop::kernel::{backdrop_blur_passes, BlurPass, MAX_BLUR_KERNEL_SIZE};
use crate::scratch::Bound;
use crate::seams::{
    BlurFilter, CaptureBlend, ClipRRect, CoveragePlan, LayerBlurGroup, NoiseFilter,
    ShadowPass, TextureFilter,
};

/// Composite uniform: rect(4) + clip(4) + params(4) + rounded clip rect(4) +
/// its corner radii(4) floats — see `layer-blur-composite.wgsl`.
pub const COMPOSITE_UNIFORM_FLOATS: usize = 20;

/// The uniform blur's pass targets: filterable, renderable everywhere, and
/// free of 8-bit steps (the result may be magnified several times).
const BLUR_FLOAT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// `MAX_TAPS` of `progressive-blur.wgsl`: taps per side, so a reach (3σ) up to
/// this many px samples every texel.
pub(crate) const PROGRESSIVE_BLUR_MAX_TAPS: f32 = 32.0;

fn shader(device: &wgpu::Device, id: &str) -> Result<wgpu::ShaderModule> {
    let src = moda_shaders::module_source(id)
        .ok_or_else(|| anyhow!("missing extracted WGSL module '{id}'"))?;
    Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(id),
        source: wgpu::ShaderSource::Wgsl(src.source.into()),
    }))
}

thread_local! {
    static SUPERSAMPLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Restores the previous [`enter_supersampled_capture`] state on drop.
pub struct SupersampledGuard(bool);

impl Drop for SupersampledGuard {
    fn drop(&mut self) {
        SUPERSAMPLED.with(|c| c.set(self.0));
    }
}

/// Renders on this thread until the guard drops are supersampled captures: a target drawn at least 2× sharper than
/// it is shown and filtered down (moda-cli's 3D units), single-sampled. Their capture shadows take the body's one
/// sample too, rather than MSAA ×4: the downsampling already antialiases the knockout at the body's edge.
pub fn enter_supersampled_capture() -> SupersampledGuard {
    SupersampledGuard(SUPERSAMPLED.with(|c| c.replace(true)))
}

/// Whether [`enter_supersampled_capture`] is in force on this thread.
pub fn supersampled_capture() -> bool {
    SUPERSAMPLED.with(std::cell::Cell::get)
}

/// Build the isolation-capture mini plan for one group. The plan maps the
/// padded canvas region 1:1 onto a `width_px × height_px` transparent target
/// via the ordinary page-local projection (camera = region origin, page_size
/// = region canvas size — the physical scale comes solely from the target
/// dimensions, exactly like an export plan).
pub fn build_capture_plan(node_id: &str, group: &LayerBlurGroup) -> CoveragePlan {
    // Nested filtered descendants are rendered by the same executor before this
    // group's filter. The owned capture tree follows the scene hierarchy.
    let [rx, ry, rw, rh] = group.region_canvas;
    CoveragePlan {
        page_id: format!("layer-blur:{node_id}"),
        page_index: 0,
        width_px: group.width_px,
        height_px: group.height_px,
        page_size: [rw, rh],
        // The capture's canvas→px scale (informational; the projection derives
        // solely from page_size × target dims).
        scale: group.width_px as f64 / rw.max(1e-6),
        transparent: true,
        // A shadow capture's coverage also knocks out / clips the shadows at
        // the body's edge, so it is antialiased like the body (MSAA 4; the
        // executor clamps to what the device takes). Single-sample coverage
        // also let the vector text lane's ray-cast leak a hairline where a
        // pixel centre sits exactly on a glyph's horizontal extremum.
        // A motion capture stands in for the layer's own antialiased draw.
        // A filter shadow's capture is coverage too: the shadows' edges are the layer's antialiased edges, and
        // under another filter the capture is drawn back as the layer itself.
        sample_count: if group.filters.shadow.is_some() || !group.filters.drop_shadows.is_empty() {
            4
        } else if group.filters.motion {
            crate::motion_capture::capture_samples()
        } else {
            1
        },
        camera: [rx, ry],
        items: group.items.clone(),
        masks: group.masks.clone(),
        camera_frame: None,
        clear_rgba: Some([0.0, 0.0, 0.0, 0.0]),
        region_viewport: None,
    }
}

/// Per-render layer-blur GPU resources (blur + composite pipelines). Built
/// once per `render_plan_to_texture_impl` call that has layer-blur items —
/// the same per-call posture as `BackdropRenderer`.
pub struct LayerBlurRenderer {
    /// The uniform blur's passes, into [`BLUR_FLOAT_FORMAT`] targets.
    blur: wgpu::RenderPipeline,
    progressive_blur: wgpu::RenderPipeline,
    texture: wgpu::RenderPipeline,
    noise: wgpu::RenderPipeline,
    shadow_prepare: wgpu::RenderPipeline,
    shadow_composite: wgpu::RenderPipeline,
    /// The capture drawn source-over on its filter shadows (`layer-filter-shadow` `fs_source`, the filter
    /// layout).
    shadow_source: wgpu::RenderPipeline,
    /// Feather: the layer's hard alpha (`layer-filter-shadow`
    /// `fs_feather_source`, the filter layout), then the layer scaled by its
    /// blur (`fs_feather`, the shadow layout: the blur at binding 2, the layer
    /// at 3).
    feather_source: wgpu::RenderPipeline,
    feather: wgpu::RenderPipeline,
    shadow_bgl: wgpu::BindGroupLayout,
    effect_bgl: wgpu::BindGroupLayout,
    composite: wgpu::RenderPipeline,
    /// The glow composites: the same quad with the mode's colour blend
    /// (`CaptureBlend`; normal is `composite`).
    screen_composite: wgpu::RenderPipeline,
    multiply_composite: wgpu::RenderPipeline,
    additive_composite: wgpu::RenderPipeline,
    composite_bgl: wgpu::BindGroupLayout,
    sampler_linear: wgpu::Sampler,
    /// The owning engine's immutable frame color format — the filter passes'
    /// textures and color targets use it, except the uniform blur's (float,
    /// [`BLUR_FLOAT_FORMAT`]); the capture source is the shared frame target,
    /// and the composite draws into the main-pass frame attachment.
    frame_format: wgpu::TextureFormat,
    /// Agent shader layer effects (`LayerFilters::shaders`) run as pixel passes, made on first use.
    agent: std::sync::OnceLock<std::sync::Mutex<crate::pixel::Passes>>,
}

impl LayerBlurRenderer {
    /// `frame_bgl` is the engine's shared group-0 frame-uniform layout;
    /// `mask_bgl` the group-2 clip-mask layout every path pipeline binds (the
    /// composite takes the node's clip mask there, or the inactive group);
    /// `sample_count` / `clear_is_opaque` describe the MAIN pass the composite
    /// quad draws into (the blur passes themselves are always single-sample).
    /// `frame_format` is the owning engine's immutable frame color format
    /// — see the field doc.
    pub fn new(
        device: &wgpu::Device,
        frame_bgl: &wgpu::BindGroupLayout,
        mask_bgl: &wgpu::BindGroupLayout,
        sample_count: u32,
        frame_format: wgpu::TextureFormat,
        working: Option<&moda_shaders::WorkingColor>,
    ) -> Result<LayerBlurRenderer> {
        let frag = wgpu::ShaderStages::FRAGMENT;
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
        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: frag,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: frag,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };

        // Blur pass resources — same shape as backdrop's effect BGL
        // (uniform / sampler / texture) driving the shared `backdrop-blur`
        // module at sample_count 1.
        let effect_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Layer Blur Effect BGL"),
            entries: &[uniform_entry(0, frag), sampler_entry(1), texture_entry(2)],
        });
        let blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Layer Blur Pipeline Layout"),
            bind_group_layouts: &[Some(&effect_bgl)],
            ..Default::default()
        });
        let blur_mod = shader(device, "backdrop-blur")?;
        let blur = moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
            label: Some("Layer Blur Gaussian Pipeline"),
            layout: Some(&blur_layout),
            vertex: wgpu::VertexState {
                module: &blur_mod,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_mod,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: BLUR_FLOAT_FORMAT,
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

        // The other filter passes: same bind-group shape, same single-sample
        // target, a fullscreen triangle each.
        let filter_pipeline = |id: &str,
                               layout: &wgpu::PipelineLayout|
         -> Result<wgpu::RenderPipeline> {
            let module = shader(device, id)?;
            Ok(
                moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
                    label: Some(id),
                    layout: Some(layout),
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
                            format: frame_format,
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
                }),
            )
        };
        let progressive_blur = filter_pipeline("progressive-blur", &blur_layout)?;
        let texture = filter_pipeline("layer-filter-texture", &blur_layout)?;

        // Capture shadows (`layer-filter-shadow`): `fs_prepare` on the filter
        // layout; `fs_composite` reads the blurred source AND the capture
        // (binding 3) and draws source-over into the result, one shadow after
        // another.
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Layer Shadow BGL"),
            entries: &[
                uniform_entry(0, frag),
                sampler_entry(1),
                texture_entry(2),
                texture_entry(3),
            ],
        });
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Layer Shadow Pipeline Layout"),
            bind_group_layouts: &[Some(&shadow_bgl)],
            ..Default::default()
        });
        // Noise reads the layer's shape coverage too (binding 3).
        let noise = filter_pipeline("layer-filter-noise", &shadow_layout)?;
        // A filter shadow's tint is an authored colour: the module converts it to the engine's working colour.
        let shadow_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("layer-filter-shadow"),
            source: wgpu::ShaderSource::Wgsl(moda_shaders::working_color::specialize(
                working,
                moda_shaders::module_source("layer-filter-shadow")
                    .ok_or_else(|| anyhow!("missing extracted WGSL module 'layer-filter-shadow'"))?
                    .source
                    .into(),
            )),
        });
        let shadow_pipeline = |label: &str,
                               layout: &wgpu::PipelineLayout,
                               entry: &str,
                               blend: Option<wgpu::BlendState>| {
            moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shadow_mod,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shadow_mod,
                    entry_point: Some(entry),
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
        let shadow_prepare =
            shadow_pipeline("Layer Shadow Prepare", &blur_layout, "fs_prepare", None);
        let feather_source =
            shadow_pipeline("Layer Feather Source", &blur_layout, "fs_feather_source", None);
        let feather = shadow_pipeline("Layer Feather", &shadow_layout, "fs_feather", None);
        let shadow_over = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let shadow_composite = shadow_pipeline(
            "Layer Shadow Composite",
            &shadow_layout,
            "fs_composite",
            Some(wgpu::BlendState {
                color: shadow_over,
                alpha: shadow_over,
            }),
        );
        let shadow_source = shadow_pipeline(
            "Layer Shadow Source",
            &blur_layout,
            "fs_source",
            Some(wgpu::BlendState {
                color: shadow_over,
                alpha: shadow_over,
            }),
        );

        // Composite pipeline — engine-owned (manifest entry
        // `layer-blur-composite` carries `pipelines: []` like `shape-solid`):
        // premultiplied blend, transparent-phase depth (write OFF, compare
        // Less), MSAA count matching the main pass.
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Layer Blur Composite BGL"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                sampler_entry(1),
                texture_entry(2),
            ],
        });
        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Layer Blur Composite Pipeline Layout"),
            bind_group_layouts: &[Some(frame_bgl), Some(&composite_bgl), Some(mask_bgl)],
            ..Default::default()
        });
        let composite_mod = shader(device, "layer-blur-composite")?;
        let premul = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let composite_pipeline = |label: &str, blend: wgpu::BlendState| {
            moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{label} s{sample_count}")),
                layout: Some(&composite_layout),
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
                        blend: Some(blend),
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
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let composite = composite_pipeline("Layer Blur Composite Pipeline", premul);
        // Glow modes (`CaptureBlend`) other than normal: only the colour's
        // blend changes; coverage unions like source-over, so a transparent
        // export keeps the halo's alpha. The halo is premultiplied (S).
        let glow_composite = |label: &str, src: wgpu::BlendFactor, dst: wgpu::BlendFactor| {
            composite_pipeline(
                label,
                wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: src,
                        dst_factor: dst,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: premul.alpha,
                },
            )
        };
        use wgpu::BlendFactor as F;
        // Screen: S + D (1 - S).
        let screen_composite = glow_composite(
            "Layer Glow Screen Composite Pipeline",
            F::One,
            F::OneMinusSrc,
        );
        // Multiply: S D + D (1 - αS).
        let multiply_composite = glow_composite(
            "Layer Glow Multiply Composite Pipeline",
            F::Dst,
            F::OneMinusSrcAlpha,
        );
        // Linear dodge: S + D (emitted light).
        let additive_composite =
            glow_composite("Layer Glow Add Composite Pipeline", F::One, F::One);

        let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Layer Blur Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Ok(LayerBlurRenderer {
            frame_format,
            agent: std::sync::OnceLock::new(),
            blur,
            progressive_blur,
            texture,
            noise,
            shadow_prepare,
            shadow_composite,
            shadow_source,
            feather_source,
            feather,
            shadow_bgl,
            effect_bgl,
            composite,
            screen_composite,
            multiply_composite,
            additive_composite,
            composite_bgl,
            sampler_linear,
        })
    }

    pub fn composite_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.composite
    }

    /// The composite of a capture group with this blend (`ShadowFilter::blend`;
    /// every non-shadow group composites source-over).
    pub fn blend_composite_pipeline(&self, blend: CaptureBlend) -> &wgpu::RenderPipeline {
        match blend {
            CaptureBlend::Normal => &self.composite,
            CaptureBlend::Screen => &self.screen_composite,
            CaptureBlend::Multiply => &self.multiply_composite,
            CaptureBlend::Add => &self.additive_composite,
        }
    }

    /// A copy of a rendered capture (`group`'s size) in a fresh texture: the
    /// capture target is shared, and the next capture overwrites it.
    pub fn copy_capture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scratch: &crate::scratch::ScratchPool,
        capture: &wgpu::Texture,
        group: &LayerBlurGroup,
    ) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width: group.width_px,
            height: group.height_px,
            depth_or_array_layers: 1,
        };
        let copy = scratch.texture(device, &wgpu::TextureDescriptor {
            label: Some("Layer Filter Coverage"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.frame_format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Layer Filter Coverage Copy"),
        });
        encoder.copy_texture_to_texture(
            capture.as_image_copy(),
            copy.as_image_copy(),
            size,
        );
        crate::scratch::submit(queue, [encoder.finish()]);
        scratch.view(&copy)
    }

    /// Run the group's filters over an already-rendered capture into a fresh
    /// per-node result texture; submits its own encoder. The capture texture
    /// may be the shared size-keyed frame target — the result is what
    /// survives into the main pass. `coverage`: the node's shape coverage
    /// (a render of `group.coverage`) the noise is clipped to; `None` clips
    /// it to the capture's own alpha.
    pub fn filter_capture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scratch: &crate::scratch::ScratchPool,
        scissor: Option<[u32; 4]>,
        capture: &wgpu::Texture,
        group: &LayerBlurGroup,
        coverage: Option<&wgpu::TextureView>,
    ) -> Result<wgpu::TextureView> {
        let (width, height) = (group.width_px, group.height_px);
        if let Some(shadow) = &group.filters.shadow {
            let capture = scratch.view(capture);
            return Ok(self.shadows_of(device, queue, scratch, scissor, &capture, [width, height], &shadow.shadows, shadow.inner, false));
        }
        let filters = &group.filters;
        // Filter shadows alone: the result is the shadows, and the layer's own items draw over the composite.
        if filters.shadows_alone {
            let capture = scratch.view(capture);
            return Ok(self.shadows_of(device, queue, scratch, scissor, &capture, [width, height], &filters.drop_shadows, false, false));
        }
        // A motion capture runs no filter: its result is a copy of the capture (the target is shared).
        if filters.drop_shadows.is_empty()
            && filters.texture.is_none()
            && filters.blur.is_none()
            && filters.noise.is_empty()
            && filters.feather.is_none()
            && filters.shaders.is_empty()
        {
            return Ok(self.copy_capture(device, queue, scratch, capture, group));
        }
        let mk_tex_as = |label: &str, format: wgpu::TextureFormat| {
            scratch.texture(device, &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let mk_tex = |label: &str| mk_tex_as(label, self.frame_format);
        // The capture may come from the shared frame-target cache, whose
        // texture matches the engine frame format; a plain default view is
        // correct.
        let mut current = scratch.view(capture);
        // Filter shadows first, under another filter (their own encoder, submitted before the passes below):
        // the capture's alpha casts them and the capture is drawn over them, so every later filter runs over
        // the picture with its shadows.
        if !filters.drop_shadows.is_empty() {
            current = self.shadows_of(device, queue, scratch, scissor, &current, [width, height], &filters.drop_shadows, false, true);
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Layer Filter Encoder"),
        });
        // Each pass writes a fresh texture: the final one outlives this call
        // as the composite's source, and no pass reads what it writes.
        let mut step_as = |label: &str,
                           format: wgpu::TextureFormat,
                           pipeline: &wgpu::RenderPipeline,
                           uniform: &[f32],
                           input: &wgpu::TextureView| {
            let out = mk_tex_as(label, format);
            self.run_pass(
                device,
                queue, scratch, scissor,
                &mut encoder,
                label,
                pipeline,
                uniform,
                input,
                &out,
            );
            scratch.view(&out)
        };
        let frame_format = self.frame_format;
        let mut step = |label: &str,
                        pipeline: &wgpu::RenderPipeline,
                        uniform: &[f32],
                        input: &wgpu::TextureView| {
            step_as(label, frame_format, pipeline, uniform, input)
        };
        let size = [width as f32, height as f32];

        if let Some(t) = &filters.texture {
            current = step(
                "Layer Filter Texture",
                &self.texture,
                &texture_uniform(t, size, filters.origin_px),
                &current,
            );
        }
        match filters.blur {
            Some(BlurFilter::Uniform { radius_px }) => {
                // Past σ = 5 capture px the blur runs as backdrop's ladder of
                // separable passes (`backdrop_blur_passes`), so the plan can
                // keep the capture close to full resolution (plan.rs
                // `SIGMA_CAPTURE_MAX`). The passes write float: the result is
                // magnified by the composite's bilinear read, and 8-bit steps
                // would come back as blocks of capture texels in the faint
                // tail (a boxy glow).
                let passes = backdrop_blur_passes((radius_px as f64).max(0.5));
                for direction in 0..2 {
                    for pass in &passes {
                        current = step_as(
                            if direction == 0 {
                                "Layer Blur Horizontal Pass"
                            } else {
                                "Layer Blur Vertical Pass"
                            },
                            BLUR_FLOAT_FORMAT,
                            &self.blur,
                            &blur_pass_uniform(pass, direction, size),
                            &current,
                        );
                    }
                }
            }
            Some(blur @ BlurFilter::Progressive { .. }) => {
                for direction in 0..2 {
                    current = step(
                        if direction == 0 {
                            "Layer Progressive Blur Horizontal Pass"
                        } else {
                            "Layer Progressive Blur Vertical Pass"
                        },
                        &self.progressive_blur,
                        &progressive_blur_uniform(&blur, direction, size),
                        &current,
                    );
                }
            }
            None => {}
        }
        for n in &filters.noise {
            let out = mk_tex("Layer Filter Noise");
            self.run_pass_with(
                device,
                queue, scratch, scissor,
                &mut encoder,
                "Layer Filter Noise",
                &self.noise,
                &noise_uniform(n, size, filters.origin_px),
                &current,
                Some(coverage.unwrap_or(&current)),
                &out,
            );
            current = scratch.view(&out);
        }
        if let Some(f) = &filters.feather {
            // The layer's hard alpha, blurred, then the layer scaled by the
            // remapped blur.
            let source = mk_tex("Layer Feather Source");
            self.run_pass(
                device,
                queue, scratch, scissor,
                &mut encoder,
                "Layer Feather Source",
                &self.feather_source,
                &feather_uniform(size),
                &current,
                &source,
            );
            let blurred = self.blur_alpha(
                device,
                queue, scratch, scissor,
                &mut encoder,
                scratch.view(&source),
                [width, height],
                f.radius_px,
            );
            let out = mk_tex("Layer Filter Feather");
            self.run_pass_with(
                device,
                queue, scratch, scissor,
                &mut encoder,
                "Layer Filter Feather",
                &self.feather,
                &feather_uniform(size),
                &blurred,
                Some(&current),
                &out,
            );
            current = scratch.view(&out);
        }
        crate::scratch::submit(queue, [encoder.finish()]);
        for shader in &filters.shaders {
            current = self.agent_shader(device, queue, scratch, shader, &current, [width, height])?;
        }
        Ok(current)
    }

    /// One agent shader layer effect over `input` (the capture so far): a pixel pass of its moda-wgsl effect module,
    /// tiled and timed. Past the time budget an export fails (`shader_budget_exceeded`) and a preview draws a
    /// magenta checker and warns; a NaN or infinity is a magenta checker and a `shader_nan` warning.
    fn agent_shader(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _scratch: &crate::scratch::ScratchPool,
        shader: &crate::seams::AgentShaderFilter,
        input: &wgpu::TextureView,
        [width, height]: [u32; 2],
    ) -> Result<wgpu::TextureView> {
        // A timed pass (its time budget): what this render holds goes first.
        crate::scratch::flush();
        let passes = self
            .agent
            .get_or_init(|| std::sync::Mutex::new(crate::pixel::Passes::new(device, queue)))
            .lock()
            .map_err(|_| anyhow::anyhow!("agent shader passes poisoned"))?;
        let out = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Layer Agent Shader"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.frame_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = out.create_view(&Default::default());
        let domain = crate::pixel::Domain::new(0, 0, width, height);
        let budget = crate::pixel::agent_budget_per_1080p() * (shader.area_px / (1920.0 * 1080.0)).max(1.0);
        let warn = |code: &str, message: String| {
            eprintln!("moda-engine: warning: {code}: {message}");
            moda_assets::report::record(moda_assets::report::RuntimeEvent::RenderQuality {
                code: code.to_string(),
                node_id: shader.node_id.clone(),
                message,
            });
        };
        match passes.run_agent_on(&shader.code, &shader.wgsl, &[(input, domain)], &shader.slots, &view, self.frame_format, domain, budget) {
            Ok((nan, _ms)) => {
                if let Some((x, y)) = nan {
                    // The report is in capture px: back to the layer's px (header slots 3–4: capture → p).
                    let (m, t) = (shader.slots[3], shader.slots[4]);
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let (lx, ly) = (m[0] * px + m[2] * py + t[0], m[1] * px + m[3] * py + t[1]);
                    warn(
                        "shader_nan",
                        format!(
                            "layer effect `{}` on `{}` returned NaN or infinity at layer px ({lx:.0}, {ly:.0}): drawn as a magenta checker",
                            shader.code, shader.node_id
                        ),
                    );
                }
                Ok(view)
            }
            Err(e @ crate::pixel::PixelError::Shader { code: "shader_budget_exceeded", .. }) if crate::pixel::is_preview() => {
                warn("shader_budget_exceeded", format!("layer effect on `{}`: {e} (drawn as a magenta checker; an export fails)", shader.node_id));
                let magenta = moda_wgsl::module(
                    "fn shade(p: vec2<f32>) -> vec4<f32> {\n  let k = (i32(floor(p.x / 8.0)) + i32(floor(p.y / 8.0))) & 1;\n  return select(vec4<f32>(1.0, 0.0, 1.0, 1.0), vec4<f32>(0.1, 0.0, 0.1, 1.0), k == 1);\n}",
                    &moda_wgsl::Schema::default(),
                    moda_wgsl::Placement::Effect,
                );
                passes
                    .run_agent_on("magenta", &magenta.wgsl, &[], &shader.slots[..moda_wgsl::RESERVED_SLOTS], &view, self.frame_format, domain, f64::INFINITY)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                Ok(view)
            }
            Err(e) => Err(anyhow::anyhow!(crate::pixel::PixelError::Shader {
                code: match &e {
                    crate::pixel::PixelError::Shader { code, .. } => code,
                    _ => "shader_invalid",
                },
                message: format!("layer effect `{}` on `{}`: {e}", shader.code, shader.node_id),
            })),
        }
    }

    /// The shadows of a capture, drawn back to front into a fresh texture.
    /// Per shadow: the capture's coverage (drop) or outside (`inner`) moved
    /// by the offset (`fs_prepare`), blurred with two `progressive-blur`
    /// passes at a constant radius (any size at full resolution), then
    /// tinted and knocked out by / clipped to the coverage (`fs_composite`,
    /// source-over into the result). A shadow group's result is its shadows
    /// alone; `under_source` (filter shadows) draws the capture itself over
    /// them (`fs_source`), the picture with its shadows.
    #[allow(clippy::too_many_arguments)]
    fn shadows_of(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scratch: &crate::scratch::ScratchPool,
        scissor: Option<[u32; 4]>,
        capture_view: &wgpu::TextureView,
        [width, height]: [u32; 2],
        shadows: &[ShadowPass],
        inner: bool,
        under_source: bool,
    ) -> wgpu::TextureView {
        let size = [width as f32, height as f32];
        let mk_tex = |label: &str| {
            scratch.texture(device, &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.frame_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let result = mk_tex("Layer Shadow Result");
        let result_view = scratch.view(&result);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Layer Shadow Encoder"),
        });
        for (i, pass) in shadows.iter().enumerate() {
            let uniform = shadow_uniform(pass, inner, size);
            let prepared = mk_tex("Layer Shadow Source");
            self.run_pass(
                device,
                queue, scratch, scissor,
                &mut encoder,
                "Layer Shadow Prepare",
                &self.shadow_prepare,
                &uniform,
                capture_view,
                &prepared,
            );
            let source = self.blur_alpha(
                device,
                queue, scratch, scissor,
                &mut encoder,
                scratch.view(&prepared),
                [width, height],
                pass.radius_px,
            );
            let ubuf = scratch.buffer(device, &wgpu::BufferDescriptor {
                label: Some("Layer Shadow Composite Uniform"),
                size: (uniform.len() * 4) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            crate::scratch::write_buffer(device, queue, &ubuf, 0, bytemuck::cast_slice(&uniform));
            let bind_group = scratch.bind(device, "Layer Shadow Composite", &self.shadow_bgl, &[
                Bound::Buffer(ubuf),
                Bound::Sampler(self.sampler_linear.clone()),
                Bound::View(source),
                Bound::View(capture_view.clone()),
            ]);
            let profile_ts = crate::gpu_profile::pass_timestamps();
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Layer Shadow Composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &result_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if i == 0 {
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some([x, y, w, h]) = scissor.and_then(|r| fit(r, width, height)) {
                rp.set_scissor_rect(x, y, w, h);
            }
            rp.set_pipeline(&self.shadow_composite);
            rp.set_bind_group(0, &bind_group, &[]);
            rp.draw(0..3, 0..1);
        }
        if under_source {
            let uniform = feather_uniform(size);
            let ubuf = scratch.buffer(device, &wgpu::BufferDescriptor {
                label: Some("Layer Shadow Source Uniform"),
                size: (uniform.len() * 4) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            crate::scratch::write_buffer(device, queue, &ubuf, 0, bytemuck::cast_slice(&uniform));
            let bind_group = scratch.bind(device, "Layer Shadow Source", &self.effect_bgl, &[
                Bound::Buffer(ubuf),
                Bound::Sampler(self.sampler_linear.clone()),
                Bound::View(capture_view.clone()),
            ]);
            let profile_ts = crate::gpu_profile::pass_timestamps();
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Layer Shadow Source"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &result_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if shadows.is_empty() {
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some([x, y, w, h]) = scissor.and_then(|r| fit(r, width, height)) {
                rp.set_scissor_rect(x, y, w, h);
            }
            rp.set_pipeline(&self.shadow_source);
            rp.set_bind_group(0, &bind_group, &[]);
            rp.draw(0..3, 0..1);
        }
        crate::scratch::submit(queue, [encoder.finish()]);
        result_view
    }

    /// `source` blurred at a constant `radius_px` reach (3σ) with two
    /// `progressive-blur` passes (any size at full resolution); under 0.6 px
    /// it is returned as is (σ below 0.2 px is a pass-through). The result
    /// may be smaller than `size` (see below): read it with normalised uv.
    #[allow(clippy::too_many_arguments)]
    fn blur_alpha(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scratch: &crate::scratch::ScratchPool,
        scissor: Option<[u32; 4]>,
        encoder: &mut wgpu::CommandEncoder,
        mut source: wgpu::TextureView,
        size: [u32; 2],
        radius_px: f32,
    ) -> wgpu::TextureView {
        if radius_px < 0.6 {
            return source;
        }
        let [width, height] = size;
        // Past `PROGRESSIVE_BLUR_MAX_TAPS` px of reach the blur's taps
        // spread wider than a texel, and a hard edge (every shadow's
        // source) blurs into visible steps. Blur at 1/2^n resolution
        // instead, so the taps stay a texel apart: the source is first
        // halved n times (a pass-through `progressive-blur` read at the
        // corner of each 2×2 block: their exact average, so the halved
        // source keeps the edge's sub-texel position), and the
        // composite's bilinear read (normalised uv) brings the smooth
        // result back up.
        let (mut bw, mut bh) = (width, height);
        let mut radius_px = radius_px;
        let mk_blur_tex = |label: &str, w: u32, h: u32| {
            scratch.texture(device, &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.frame_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        while radius_px > PROGRESSIVE_BLUR_MAX_TAPS && bw > 2 && bh > 2 {
            let (hw, hh) = (bw.div_ceil(2), bh.div_ceil(2));
            radius_px *= hw as f32 / bw as f32;
            (bw, bh) = (hw, hh);
            let out = mk_blur_tex("Layer Shadow Downsample", bw, bh);
            let copy = BlurFilter::Uniform { radius_px: 0.0 };
            self.run_pass(
                device,
                queue, scratch, scaled(scissor, [width, height], [bw, bh]),
                encoder,
                "Layer Shadow Downsample",
                &self.progressive_blur,
                &progressive_blur_uniform(&copy, 0, [bw as f32, bh as f32]),
                &source,
                &out,
            );
            source = scratch.view(&out);
        }
        let blur = BlurFilter::Progressive {
            start_radius_px: radius_px,
            end_radius_px: radius_px,
            start_px: [0.0; 2],
            end_px: [0.0; 2],
        };
        for direction in 0..2 {
            let out = mk_blur_tex("Layer Shadow Blur", bw, bh);
            self.run_pass(
                device,
                queue, scratch, scaled(scissor, [width, height], [bw, bh]),
                encoder,
                "Layer Shadow Blur",
                &self.progressive_blur,
                &progressive_blur_uniform(&blur, direction, [bw as f32, bh as f32]),
                &source,
                &out,
            );
            source = scratch.view(&out);
        }
        source
    }

    /// One fullscreen filter pass: `uniform` at binding 0, the linear sampler
    /// at 1, `input` at 2, drawn into `output` (cleared).
    #[allow(clippy::too_many_arguments)]
    fn run_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scratch: &crate::scratch::ScratchPool,
        scissor: Option<[u32; 4]>,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::RenderPipeline,
        uniform: &[f32],
        input: &wgpu::TextureView,
        output: &wgpu::Texture,
    ) {
        self.run_pass_with(
            device, queue, scratch, scissor, encoder, label, pipeline, uniform, input, None, output,
        );
    }

    /// [`Self::run_pass`] with a second input at binding 3 (the shadow layout,
    /// which noise shares): `second` = Some.
    #[allow(clippy::too_many_arguments)]
    fn run_pass_with(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scratch: &crate::scratch::ScratchPool,
        scissor: Option<[u32; 4]>,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::RenderPipeline,
        uniform: &[f32],
        input: &wgpu::TextureView,
        second: Option<&wgpu::TextureView>,
        output: &wgpu::Texture,
    ) {
        let ubuf = scratch.buffer(device, &wgpu::BufferDescriptor {
            label: Some(label),
            size: (uniform.len() * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        crate::scratch::write_buffer(device, queue, &ubuf, 0, bytemuck::cast_slice(uniform));
        let mut bound = vec![
            Bound::Buffer(ubuf),
            Bound::Sampler(self.sampler_linear.clone()),
            Bound::View(input.clone()),
        ];
        if let Some(second) = second {
            bound.push(Bound::View(second.clone()));
        }
        let layout = if second.is_some() { &self.shadow_bgl } else { &self.effect_bgl };
        let bind_group = scratch.bind(device, label, layout, &bound);
        let out_view = scratch.view(output);
        let profile_ts = crate::gpu_profile::pass_timestamps();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &out_view,
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
        if let Some([x, y, w, h]) = scissor.and_then(|r| fit(r, output.width(), output.height())) {
            pass.set_scissor_rect(x, y, w, h);
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Uniform data for one composite quad — must match
    /// `LayerBlurCompositeUniforms` in `layer-blur-composite.wgsl`.
    pub fn composite_uniform_data(
        group: &LayerBlurGroup,
        page_clip: [f32; 4],
        clip_rrect: Option<ClipRRect>,
        depth: f32,
        opacity: f32,
    ) -> [f32; COMPOSITE_UNIFORM_FLOATS] {
        let [rx, ry, rw, rh] = group.region_canvas;
        // No rounded clip: a zero-size rect, which the shader skips.
        let rr = clip_rrect.unwrap_or(ClipRRect {
            rect: [0.0; 4],
            radii: [0.0; 4],
        });
        [
            rx as f32,
            ry as f32,
            rw as f32,
            rh as f32,
            page_clip[0],
            page_clip[1],
            page_clip[2],
            page_clip[3],
            depth,
            opacity,
            0.0,
            0.0,
            rr.rect[0],
            rr.rect[1],
            rr.rect[2],
            rr.rect[3],
            rr.radii[0],
            rr.radii[1],
            rr.radii[2],
            rr.radii[3],
        ]
    }

    /// The bind group of one composite draw: its uniform and the capture result it reads. `lasting`: the frame
    /// may keep the group past its render (a prepared frame's recorded pass binds it on later frames), so its
    /// uniform buffer is its own and not the pool's.
    pub fn composite_bind_group(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scratch: &crate::scratch::ScratchPool,
        lasting: bool,
        uniform_data: &[f32; COMPOSITE_UNIFORM_FLOATS],
        capture: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let desc = wgpu::BufferDescriptor {
            label: Some("Layer Blur Composite Uniform"),
            size: (COMPOSITE_UNIFORM_FLOATS * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        let ubuf = if lasting { device.create_buffer(&desc) } else { scratch.buffer(device, &desc) };
        crate::scratch::write_buffer(device, queue, &ubuf, 0, bytemuck::cast_slice(uniform_data));
        scratch.bind(device, "Layer Blur Composite Bind Group", &self.composite_bgl, &[
            Bound::Buffer(ubuf),
            Bound::Sampler(self.sampler_linear.clone()),
            Bound::View(capture.clone()),
        ])
    }
}

/// `BlurUniforms` of `backdrop-blur.wgsl` for one pass in one direction — the
/// same packing as backdrop's `run_blur_pass`.
fn blur_pass_uniform(kernel: &BlurPass, direction: u32, size: [f32; 2]) -> Vec<f32> {
    let kernel_size = kernel.size.min(MAX_BLUR_KERNEL_SIZE);
    let mut uniform = vec![0f32; 4 + MAX_BLUR_KERNEL_SIZE * 4];
    uniform[0] = direction as f32;
    uniform[1] = kernel_size as f32;
    uniform[2] = size[0];
    uniform[3] = size[1];
    for i in 0..kernel_size {
        uniform[4 + i * 4] = kernel.weights.get(i).copied().unwrap_or(0.0);
        uniform[4 + i * 4 + 1] = kernel.offsets.get(i).copied().unwrap_or(0.0);
    }
    uniform
}

/// `ProgressiveBlurUniforms` of `progressive-blur.wgsl` for one direction.
pub(crate) fn progressive_blur_uniform(
    blur: &BlurFilter,
    direction: u32,
    size: [f32; 2],
) -> [f32; 12] {
    let (start_r, end_r, start, end) = match *blur {
        BlurFilter::Progressive {
            start_radius_px,
            end_radius_px,
            start_px,
            end_px,
        } => (start_radius_px, end_radius_px, start_px, end_px),
        BlurFilter::Uniform { radius_px } => (radius_px, radius_px, [0.0; 2], [0.0; 2]),
    };
    [
        direction as f32,
        0.0,
        size[0],
        size[1],
        start[0],
        start[1],
        end[0],
        end[1],
        start_r,
        end_r,
        0.0,
        0.0,
    ]
}

/// `ShadowUniforms` of `layer-filter-shadow.wgsl`.
fn shadow_uniform(pass: &ShadowPass, inner: bool, size: [f32; 2]) -> [f32; 12] {
    [
        size[0],
        size[1],
        pass.offset_px[0],
        pass.offset_px[1],
        pass.color[0],
        pass.color[1],
        pass.color[2],
        pass.color[3],
        if inner { 1.0 } else { 0.0 },
        if pass.show_behind { 1.0 } else { 0.0 },
        if pass.center { 1.0 } else { 0.0 },
        0.0,
    ]
}

/// `ShadowUniforms` of `layer-filter-shadow.wgsl` for `fs_feather_source` /
/// `fs_feather` (they read only the texture size).
fn feather_uniform(size: [f32; 2]) -> [f32; 12] {
    let mut u = [0.0; 12];
    u[0] = size[0];
    u[1] = size[1];
    u
}

/// `TextureUniforms` of `layer-filter-texture.wgsl`.
fn texture_uniform(t: &TextureFilter, size: [f32; 2], origin: [f32; 2]) -> [f32; 8] {
    [
        size[0],
        size[1],
        origin[0],
        origin[1],
        t.size_px,
        t.radius_px,
        if t.clip_to_shape { 1.0 } else { 0.0 },
        0.0,
    ]
}

/// `NoiseUniforms` of `layer-filter-noise.wgsl`.
fn noise_uniform(n: &NoiseFilter, size: [f32; 2], origin: [f32; 2]) -> [f32; 20] {
    let secondary = n.secondary.unwrap_or([0.0; 3]);
    [
        size[0],
        size[1],
        origin[0],
        origin[1],
        n.size_px,
        n.density,
        n.opacity,
        n.mode as f32,
        n.color[0],
        n.color[1],
        n.color[2],
        1.0,
        secondary[0],
        secondary[1],
        secondary[2],
        if n.secondary.is_some() { 1.0 } else { 0.0 },
        n.seed as f32,
        noise_threshold(n.density),
        0.0,
        0.0,
    ]
}

/// The share of cells Figma's noise paints at `density` (0..1), each at a
/// ramp of its random value (`layer-filter-noise.wgsl`). Fitted to Figma's
/// own export (mono, density 0.6: 28 % of cells paint, the card darkens 5.5 %
/// on average; docs/qa/paste-fidelity.md §2.1): none at 0, every cell at 1.
pub fn noise_on_share(density: f32) -> f32 {
    density.clamp(0.0, 1.0).powf(2.5)
}

/// The cell-value threshold of [`noise_on_share`]: cells above it paint.
pub fn noise_threshold(density: f32) -> f32 {
    1.0 - noise_on_share(density)
}

/// CPU twin of `cellHash` in `layer-filter-noise.wgsl` /
/// `layer-filter-texture.wgsl`: bit-for-bit the same u32 arithmetic, so tests
/// can predict exactly which noise cells are on.
pub fn noise_cell_hash(x: i32, y: i32, salt: u32) -> f32 {
    fn hash_u32(x: u32) -> u32 {
        let mut h = x;
        h ^= h >> 16;
        h = h.wrapping_mul(0x7feb_352d);
        h ^= h >> 15;
        h = h.wrapping_mul(0x846c_a68b);
        h ^= h >> 16;
        h
    }
    let h = hash_u32(x as u32 ^ hash_u32(y as u32 ^ hash_u32(salt)));
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// `r` (texels of a `w × h` texture) cut to the texture; `None` when nothing is left.
fn fit(r: [u32; 4], w: u32, h: u32) -> Option<[u32; 4]> {
    let (x0, y0) = (r[0].min(w), r[1].min(h));
    let (x1, y1) = (r[0].saturating_add(r[2]).min(w), r[1].saturating_add(r[3]).min(h));
    (x1 > x0 && y1 > y0).then_some([x0, y0, x1 - x0, y1 - y0])
}

/// `r` in the texels of a `from`-sized texture, as the texels of a `to`-sized one covering the same area (rounded
/// out).
fn scaled(r: Option<[u32; 4]>, from: [u32; 2], to: [u32; 2]) -> Option<[u32; 4]> {
    let r = r?;
    let (fx, fy) = (f64::from(to[0]) / f64::from(from[0].max(1)), f64::from(to[1]) / f64::from(from[1].max(1)));
    let x0 = (f64::from(r[0]) * fx).floor() as u32;
    let y0 = (f64::from(r[1]) * fy).floor() as u32;
    let x1 = (f64::from(r[0] + r[2]) * fx).ceil() as u32;
    let y1 = (f64::from(r[1] + r[3]) * fy).ceil() as u32;
    Some([x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)])
}
