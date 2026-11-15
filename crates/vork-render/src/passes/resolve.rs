//! Post resources and the compose and resolve steps: MSAA resolve, shutter
//! accumulation, display conversion and packed controls; optional pixel readback.
use crate::*;

pub(crate) struct Resources {
    pub pixels: wgpu::Buffer,
    pub(crate) invalid_bitmap: std::ops::Range<u64>,
    pub(crate) output_layout: wgpu::BindGroupLayout,
    pub(crate) display_inputs: Option<wgpu::Buffer>,
    pub(crate) compositor_inputs: crate::compositor::FrameInputs,
    pub nonfinite: wgpu::Buffer,
    pub(crate) display_base: Option<u32>,
    pub(crate) denoiser: Option<crate::denoise::Stage>,
    pub(crate) reconstruction: Option<crate::denoise::spatial::Stage>,
}
impl Resources {
    pub(crate) fn new(
        s: &InstalledScene,
        r: &Renderer,
        uniform: &wgpu::Buffer,
        ao_guides: Option<(&wgpu::TextureView, &wgpu::TextureView)>,
        lights: &wgpu::Buffer,
        analytic_light_tree: &wgpu::Buffer,
        cluster_config: &wgpu::Buffer,
        cluster_lights: &wgpu::Buffer,
        material_records: &wgpu::Buffer,
        objects: &wgpu::Buffer,
        targets: &crate::view::Targets,
        reflections: &passes::reflections::Resources,
        ao: &passes::ao::Resources,
        output: &wgpu::Texture,
        planes: u32,
        exposure_samples: u32,
        rays: Option<&crate::rt::FrameRays>,
        material: &wgpu::BindGroup,
        shader_errors: Option<&wgpu::Buffer>,
        recorder: &mut crate::graph::Recorder,
        view_desc: &crate::views::ViewDesc,
        compositor: &[vork_scene::v2::CompositeEntry],
        denoiser: Option<&vork_scene::v2::Denoiser>,
        reconstruct: bool,
        graph: &crate::graph::Graph,
        dependencies: crate::compositor::Dependencies,
    ) -> Result<(Self, wgpu::BindGroup)> {
        let needs_source=dependencies.beauty;
        let denoiser=denoiser.filter(|_|needs_source);
        let reconstruct=reconstruct && needs_source;
        // Delivery of a composed image has no scene/medium bindings of its own.
        let (pipelines,effects)=if needs_source {(s.pipelines.clone(),s.plan.effects)} else {
            let effects=crate::plan::EffectPlan {compositor:!s.plan.effects.fuse_controls(),
                outputs:Some(crate::compositor::Dependencies {color:dependencies.color,..Default::default()}),..Default::default()};
            let pipelines=r.cached_pipelines(s.pipelines.raster_samples);
            (pipelines,effects)
        };
        pipelines.ensure_output(r,effects)?;
        let resolve_pipeline = pipelines.output_resolve(effects)
            .get()
            .context("base pipelines installed")?;
        let d = &r.device;
        let [w,h] = view_desc.extent;
        let crate::view::Targets {
            post_depth,
            post_ids,
            post_normals,
            post_ambient,
            post_motion,
            resolved,
            ..
        } = targets;
        let ao_view = &ao.output;
        let capture = view_desc.passes == crate::views::PassSet::Image;
        // Captures share the camera output texture binding, but their planar
        // pixels and dispatches belong to their own logical output extent.
        let [output_width, output_height] = view_desc.extent;
        let size = if graph.runs(crate::graph::Pass::Resolve) || graph.runs(crate::graph::Pass::AovResolve) || graph.runs(crate::graph::Pass::Path) {u64::from(output_width) * u64::from(output_height) * u64::from(planes) * 4} else {16};
        let bitmap_bytes = crate::frame::invalid_pixel_bitmap_bytes(u64::from(output_width) * u64::from(output_height));
        let invalid_bitmap = size..size + bitmap_bytes;
        // Captures can rerun between camera exposure samples.
        // Keep each capture's resolve separate from the retained camera sum.
        let buffer = |label: &str, size: u64, usage| {
            let owned;
            let label = if let crate::views::ViewKind::Image { name, .. } = &view_desc.kind {
                owned = format!("image {name}: {label}");
                owned.as_str()
            } else { label };
            s.transients.buffer(d, label, size, usage)
        };
        // Resolve stages unassociated scene-linear colour for the display program.
        let pixels = buffer("packed PNG pixels", invalid_bitmap.end,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        let catcher_base=planes.saturating_sub(4 + if !capture {4} else {0});
        if graph.runs(crate::graph::Pass::Path) {passes::path::record(s,r,uniform,lights,analytic_light_tree,cluster_config,cluster_lights,objects,rays,material,resolved,&targets.aov_values,targets,&targets.catcher,&pixels,catcher_base,(invalid_bitmap.start / 4) as u32,recorder,view_desc)?;}
        let accumulation = buffer(
            "linear sample accumulation",
            if needs_source && (exposure_samples > 1 || !compositor.is_empty() || s.has_catchers || denoiser.is_some() || reconstruct) {
                u64::from(w) * u64::from(h) * constants::ACCUMULATION_PIXEL_BYTES
            } else {
                constants::ACCUMULATION_PIXEL_BYTES
            },
            wgpu::BufferUsages::STORAGE
                | if denoiser.is_some() {
                    wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
                } else {
                    wgpu::BufferUsages::empty()
                },
        );
        let output_view = output.create_view(&Default::default());
        let view_render = match &view_desc.kind {
            crate::views::ViewKind::Image {render,..} => &render.settings,
            _ => &s.render,
        };
        let camera = match &view_desc.kind {
            crate::views::ViewKind::Camera(camera) => *camera,
            crate::views::ViewKind::Image {render,..} => render.camera,
        };
        // Ray generation alone does not change the guide meaning. Only the
        // active view's detector/throughput can invalidate surface RGB guides;
        // an unrelated camera program must not disable them for this view.
        let camera_program=s.camera_programs.iter().find(|p|p.node==camera);
        let use_guides = !s.has_media && camera_program
            .is_none_or(|p|p.unit_weight && !p.spectral && p.program.sensor_entry.is_none());
        let spectral=s.spectral || camera_program.is_some_and(|p|p.spectral);
        let guides = (denoiser.is_some() || reconstruct).then(|| targets.denoising_guides.as_ref()
            .context("OIDN guide buffer missing")).transpose()?;
        let denoised = (denoiser.is_some() || reconstruct).then(|| {
            buffer(
                "denoised colour",
                u64::from(w) * u64::from(h) * 16,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )
        });
        let reconstruction = reconstruct.then(|| crate::denoise::spatial::Stage::new(r, &accumulation, denoised.as_ref().unwrap(), [w,h], exposure_samples)).transpose()?;
        let denoiser = denoiser.filter(|_|s.plan.effects.denoise)
            .map(|settings| {
                let mut stage=crate::denoise::Stage::new(
                    r,
                    settings,
                    &accumulation,
                    denoised.as_ref().unwrap(),
                    guides.expect("validated guide buffer"),
                    view_render.background.transparent(),
                    use_guides,
                    [w, h],
                )?;
                stage.spectral=spectral;
                Ok::<_,anyhow::Error>(stage)
            })
            .transpose()?;

        let compositor_inputs = crate::compositor::FrameInputs::new(&s.compositor, r, compositor);
        let mut post_color = if let Some(denoised) = &denoised {
            denoised.clone()
        } else {
            buffer("unused post colour",constants::ACCUMULATION_PIXEL_BYTES,wgpu::BufferUsages::STORAGE)
        };
        if dependencies.color && (!compositor.is_empty() || s.has_catchers) {
            post_color = crate::compositor::record(
                s,
                r,
                uniform,
                targets,
                &accumulation,
                compositor,
                &compositor_inputs,
                [w, h],
                crate::compositor::Normalization::Frame,
                recorder,
                crate::graph::Pass::Compositor,
                dependencies.beauty,
            )?;
        }
        let resolve_entries = vec![
                entry(47, targets.nonfinite.as_entire_binding()),
                entry(70, shader_errors.context("radiance diagnostics")?.as_entire_binding()),
                entry(51, wgpu::BindingResource::TextureView(&targets.depth)),
                entry(52, wgpu::BindingResource::TextureView(&targets.ids)),
                entry(53, wgpu::BindingResource::TextureView(&targets.normals)),
                entry(54, wgpu::BindingResource::TextureView(&targets.ambient)),
                entry(55, wgpu::BindingResource::TextureView(&targets.motion)),
                entry(45, post_color.as_entire_binding()),
                entry(24, wgpu::BindingResource::TextureView(&reflections.output)),
                entry(41, wgpu::BindingResource::TextureView(&ao.incident)),
                entry(58, wgpu::BindingResource::TextureView(&ao.receivers)),
                entry(0, wgpu::BindingResource::TextureView(&resolved)),
                entry(1, wgpu::BindingResource::TextureView(&post_depth)),
                entry(2, wgpu::BindingResource::TextureView(&post_ids)),
                entry(4, pixels.as_entire_binding()),
                entry(67, s.sampling.buffer.as_entire_binding()),
                entry(11, targets.diffuse_response.as_entire_binding()),
                // Disabled AO keeps ordinary view placeholders; enabled AO requires guides.
                entry(68, wgpu::BindingResource::TextureView(ao_guides.map_or(post_depth, |g| g.0))),
                entry(69, wgpu::BindingResource::TextureView(ao_guides.map_or(post_normals, |g| g.1))),
                entry(5, uniform.as_entire_binding()),
                entry(6, wgpu::BindingResource::TextureView(&post_normals)),
                entry(7, wgpu::BindingResource::TextureView(post_ambient)),
                entry(8, wgpu::BindingResource::TextureView(&ao_view)),
                entry(9, wgpu::BindingResource::TextureView(&post_motion)),
                entry(10, accumulation.as_entire_binding()),
                entry(12, wgpu::BindingResource::Sampler(&s.environment.sampler)),
                entry(43, material_records.as_entire_binding()),
                entry(44, lights.as_entire_binding()),
                entry(46, objects.as_entire_binding()),
                entry(
                    16,
                    wgpu::BindingResource::TextureView(&s.environment.specular),
                ),
            ];
        let resolve_bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resolve plan"),
            layout: &resolve_pipeline.get_bind_group_layout(0),
            entries: &resolve_entries,
        });
        let output_bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("encoder surface"),
            layout: &resolve_pipeline.get_bind_group_layout(1),
            entries: &[entry(0, wgpu::BindingResource::TextureView(&output_view))],
        });
        let pipeline = |p: &OnceLock<wgpu::ComputePipeline>| {
            p.get().cloned().context("incomplete installed post plan")
        };
        let full = [w.div_ceil(8), h.div_ceil(8), 1];
        let mut dispatches = Vec::new();
        if needs_source && (!compositor.is_empty() || s.has_catchers || denoiser.is_some() || reconstruct) {
            dispatches.push(crate::graph::Dispatch::new(
                &pipeline(&pipelines.post_source_pipeline)?,
                resolve_bind.clone(),
                full,
            ));
            recorder.compute(
                crate::graph::Pass::PostSource,
                "resolve scene-linear post source",
                "post_source",
                true,
                std::mem::take(&mut dispatches),
            );
        }
        dispatches.push(
            crate::graph::Dispatch::new(resolve_pipeline, resolve_bind.clone(), full).all_targets(),
        );
        let display_inputs = if capture { None } else { s.display_program.frame_inputs(r) };
        let display_base = (!capture).then_some(planes - 4);
        if !capture {
            let stage = &s.display_program;
            dispatches.push(stage.dispatch(r, uniform, display_inputs.as_ref(), pixels.as_entire_binding(), shader_errors.context("display diagnostics")?, [output_width,output_height]));
        }
        recorder.compute(
            crate::graph::Pass::Resolve,
            "display conversion + exact controls",
            "resolve",
            true,
            dispatches,
        );
        Ok((Self { pixels, invalid_bitmap, output_layout:resolve_pipeline.get_bind_group_layout(1), display_inputs, compositor_inputs, denoiser, reconstruction, nonfinite:targets.nonfinite.clone(), display_base }, output_bind))
    }
}

/// Original API bindings; native sources map these to Vulkan's dense indices.
pub(crate) fn layout_entries(entry: &str) -> Vec<wgpu::BindGroupLayoutEntry> {
    use wgpu::{BindingType as T, BufferBindingType as B};
    use crate::resolve_native::Kind;
    let binding_type = |binding| match crate::resolve_native::kind(binding) {
        Kind::Uniform => T::Buffer {
            ty: B::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        Kind::Read | Kind::Write => T::Buffer {
            ty: B::Storage {
                read_only: matches!(crate::resolve_native::kind(binding), Kind::Read),
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        Kind::Sampler => T::Sampler(wgpu::SamplerBindingType::Filtering),
        Kind::Texture => T::Texture {
            sample_type: if binding == 51 {wgpu::TextureSampleType::Depth} else if matches!(binding, 2 | 52) {
                wgpu::TextureSampleType::Uint
            } else {
                wgpu::TextureSampleType::Float {
                    filterable: matches!(binding, 13 | 16),
                }
            },
            view_dimension: if binding == 24 {wgpu::TextureViewDimension::D2Array} else {
                wgpu::TextureViewDimension::D2
            },
            multisampled: (51..=55).contains(&binding) && entry != "resolve_fused_single",
        },
    };
    crate::resolve_native::bindings(entry)
        .iter()
        .map(|&binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: binding_type(binding),
            count: None,
        })
        .collect()
}
/// Lookup tables used by the shared straight-segment closure evaluator.
pub(crate) fn shaders() -> [&'static crate::native::Compiled; 4] {
    use crate::native::*;
    [
        &RESOLVE_RESOLVE_COLOR,
        &RESOLVE_POST_SOURCE,
        &RESOLVE_RESOLVE_FUSED,
        &RESOLVE_RESOLVE_FUSED_SINGLE,
    ]
}
pub(crate) fn pipeline(
    r: &Renderer,
    _source: &str,
    entry: &str,
    installed_layout: Option<&wgpu::PipelineLayout>,
) -> wgpu::ComputePipeline {
    let entry = if entry == "main" {
        "resolve_color"
    } else {
        entry
    };
    let d = &r.device;
    let local_layout = installed_layout.is_none().then(|| {
        let bind = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(entry),
            entries: &layout_entries(entry),
        });
        d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry),
            bind_group_layouts: &[Some(&bind)],
            immediate_size: 0,
        })
    });
    let shader = shaders().into_iter().find(|s| s.entry == entry).expect("registered resolve entry");
    let module = crate::native::shader_module(d, shader);
    r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: installed_layout.or(local_layout.as_ref()),
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: r.cache.as_ref(),
    })
}
