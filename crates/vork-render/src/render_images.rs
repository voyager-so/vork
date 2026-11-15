//! Render-level views over one installed scene and one evaluated frame.
use crate::*;
use vork_scene::v2;

/// Raster applies view membership while recording draws. Path camera rays still
/// use the whole-scene TLAS, so accepting a subtree can render excluded objects
/// (including a monitor reading its own initially empty capture).
pub(crate) fn require_supported_roots(input: &v2::Scene, order: &[String]) -> Result<()> {
    if !input.render.settings.integrator.is_path() { return Ok(()); }
    ensure!(input.render.root.is_none(),
        "render.root: path transport does not support subtree renders; use --profile std/final with depth of field and path-only effects disabled, or put the selected content in a separate scene and remove root");
    for name in order {
        ensure!(input.images[name].render.as_ref().unwrap().root.is_none(),
            "images[{name:?}].render.root: path transport does not support subtree renders; use --profile std/final with depth of field and path-only effects disabled, or render the capture from a separate scene without root");
    }
    Ok(())
}

#[cfg(test)]
mod subtree_tests {
    use super::*;

    #[test]
    fn path_subtrees_refuse_without_a_rendering_device() -> Result<()> {
        let mut value = serde_json::json!({"version":2,
            "render":{"settings":{"integrator":"path"}},
            "images":{"feed":{"render":{"root":"content"}}},
            "nodes":[]});
        let scene = |value| vork_scene::v2::from_test_value::<v2::Scene>(value);
        let live = vec!["feed".to_owned()];
        let input = scene(value.clone())?;
        // Unused captures must not restrict the main view.
        require_supported_roots(&input, &[])?;
        let error = require_supported_roots(&input, &live).unwrap_err().to_string();
        assert!(error.contains("images[\"feed\"].render.root"), "{error}");
        assert!(error.contains("path transport does not support subtree renders"), "{error}");
        value["render"]["root"] = "monitor".into();
        let error = require_supported_roots(&scene(value.clone())?, &[]).unwrap_err().to_string();
        assert_eq!(error,
            "render.root: path transport does not support subtree renders; use --profile std/final with depth of field and path-only effects disabled, or put the selected content in a separate scene and remove root");
        vork_scene::v2::replace_integrator(&mut value, serde_json::json!({"integrator":"raster"}));
        require_supported_roots(&scene(value.clone())?, &live)?;
        vork_scene::v2::replace_integrator(&mut value, serde_json::json!({"integrator":"path"}));
        value["render"]["root"] = serde_json::Value::Null;
        value["images"]["feed"]["render"]["root"] = serde_json::Value::Null;
        require_supported_roots(&scene(value)?, &live)?;
        Ok(())
    }
}

pub(crate) const PUBLICATION_LAYOUT: [wgpu::BindGroupLayoutEntry; 3] = [
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: wgpu::TextureViewDimension::D2,
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
];

pub(crate) struct InstalledImageRender {
    pub name: String,
    pub camera: usize,
    pub settings: v2::Render,
    pub exposure_samples: u32,
    pub visibility_samples: u32,
    pub draws: Vec<bool>,
    pub objects: Vec<bool>,
    pub lights: Vec<bool>,
    pub pipeline: wgpu::ComputePipeline,
    pub output: wgpu::TextureView,
    texture: wgpu::Texture,
    pub level_zero: wgpu::TextureView,
    mips: materials::MipChain,
    time_dependent: bool,
}

impl InstalledImageRender {
    pub(crate) fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        for mip in 0..self.texture.mip_level_count() {
            let view = self.texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: mip, mip_level_count: Some(1), ..Default::default()
            });
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("reset render image"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None, depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
        }
    }

    fn publish(&self,encoder:&mut wgpu::CommandEncoder,bind:&wgpu::BindGroup,[width,height]:[u32;2],end_query:Option<(&wgpu::QuerySet,u32)>) {
        {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:Some("publish render image"),
                timestamp_writes: end_query.filter(|_| self.mips.is_empty()).map(|(query_set,index)|wgpu::ComputePassTimestampWrites {query_set,beginning_of_pass_write_index:None,end_of_pass_write_index:Some(index)}),
            });
            pass.set_pipeline(&self.pipeline);pass.set_bind_group(0,bind,&[]);
            let group=constants::IMAGE_WORKGROUP_SIZE;
            pass.dispatch_workgroups(width.div_ceil(group),height.div_ceil(group),1);
        }
        self.mips.encode(encoder,end_query.map(|(query_set,index)| wgpu::RenderPassTimestampWrites {query_set,beginning_of_pass_write_index:None,end_of_pass_write_index:Some(index)}));
    }
}

pub(crate) fn plan(
    input: &v2::Scene,
    s: &InstalledScene,
    r: &Renderer,
    order: &[String],
) -> Result<Vec<InstalledImageRender>> {
    if order.is_empty() {
        return Ok(Vec::new());
    }
    let pipeline = publication_pipeline(r);
    let (mip_pipelines,mip_sampler)=materials::mip_resources(r);
    order
        .iter()
        .map(|name| {
            let source = input.images[name].render.as_ref().unwrap();
            let settings = input.images[name].resolved_render(&input.render).expect("a rendered image");
            ensure!(
                settings
                    .resolution
                    .iter()
                    .all(|n| *n <= r.device.limits().max_texture_dimension_2d),
                crate::gpu_memory::device_limit(format_args!("render image {name}"), format_args!("{}x{} pixels",settings.resolution[0],settings.resolution[1]), format_args!("max_texture_dimension_2d={} pixels per axis",r.device.limits().max_texture_dimension_2d), "reduce image.render.resolution")
            );
            let (camera, exposure_samples, visibility_samples) = camera_sampling(&s.nodes, &settings, s.camera)?;
            let Subset {
                draws,
                objects,
                lights,
            } = subset(s, source.root.as_deref())?;
            // Rays may reach surfaces outside the capture subtree. Only absent consumers prove
            // that an excluded draw's clock cannot affect this capture.
            let integrator=&settings.settings.integrator;
            let ray_surfaces=s.plan.rays && (integrator.is_path()
                || matches!(integrator.reflections(),Some(vork_scene::Reflections::Auto|vork_scene::Reflections::Ray))
                || integrator.indirect().is_some_and(|gi|gi.method==vork_scene::FastGiMethod::Ray)
                || integrator.probe_volume().is_some()
                || integrator.raster().is_some_and(|r|matches!(r.refraction,vork_scene::Refraction::Auto|vork_scene::Refraction::Ray))
                || s.plan.effects.fibre);
            let shadows=lights.iter().any(|&included|included);
            let surface_time = s.draws.iter().enumerate().any(|(index,draw)| {
                let consumed=draws[index] || ray_surfaces || (shadows && s.plan.draws[index].casts_shadows);
                let id=s.materials[draw.material].program;
                consumed && id!=0 && s.surface_dependencies[id as usize-1].time
            });
            let time_dependent = surface_time || s.has_media
                || s.camera_programs.iter().any(|p|p.node==camera && p.uses_time)
                || s.environment.primary.as_ref().is_some_and(|p|p.uses_time)
                || s.scene.propagation.is_some() || settings.settings.integrator.is_path()
                || !settings.compositor.is_empty()
               ;
            let [width,height]=settings.resolution;
            let texture=gpu_memory::texture(&r.device,&wgpu::TextureDescriptor {
                label:Some("render image output"),size:extent(width,height),
                mip_level_count:32-width.max(height).leading_zeros(),sample_count:1,
                dimension:wgpu::TextureDimension::D2,format:wgpu::TextureFormat::Rgba16Float,
                usage:wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats:&[],
            });
            let mips=materials::MipChain::new(&r.device,&texture,None,&mip_pipelines[2],mip_sampler);
            let level_zero=texture.create_view(&wgpu::TextureViewDescriptor {mip_level_count:Some(1),..Default::default()});
            Ok(InstalledImageRender {
                name: name.clone(),
                time_dependent,
                camera,
                settings, exposure_samples, visibility_samples,
                draws,
                objects,
                lights,
                pipeline: pipeline.clone(),
                output: texture.create_view(&Default::default()),
                level_zero,mips,texture,
            })
        })
        .collect()
}

pub(crate) fn camera_sampling(nodes: &[EvalNode], settings: &v2::Render, default: usize) -> Result<(usize, u32, u32)> {
    let camera = settings.camera.as_deref().map(|name| named_node(nodes, name)).transpose()?.unwrap_or(default);
    let (exposure, visibility) = crate::plan::sampling_counts(settings, false);
    Ok((camera, exposure, visibility))
}

fn named_node(nodes: &[EvalNode], name: &str) -> Result<usize> {
    let mut matches = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.node.name == name);
    let first = matches
        .next()
        .with_context(|| format!("render node {name} is not installed"))?
        .0;
    ensure!(
        matches.next().is_none(),
        "render node {name} is ambiguous after asset expansion"
    );
    Ok(first)
}

/// Shared subtree membership for view planning and resource admission.
pub(crate) fn included_nodes(nodes: &[EvalNode], root: Option<&str>) -> Result<Vec<bool>> {
    let root = root.map(|name| named_node(nodes, name)).transpose()?;
    let mut included = vec![root.is_none(); nodes.len()];
    for (id, node) in nodes.iter().enumerate() {
        included[id] |= Some(id) == root || node.parent.is_some_and(|p| included[p]);
    }
    Ok(included)
}

pub(crate) struct Subset {
    pub draws: Vec<bool>,
    pub objects: Vec<bool>,
    pub lights: Vec<bool>,
}
pub(crate) fn subset(s: &InstalledScene, root: Option<&str>) -> Result<Subset> {
    let included = included_nodes(&s.nodes, root)?;
    let objects: Vec<_> = s.roots.iter().map(|root| included[root.node]).collect();
    let draws = s
        .draws
        .iter()
        .map(|draw| {
            draw.instances
                .clone()
                .any(|object| objects[object as usize])
        })
        .collect();
    let lights = s
        .light_plan
        .nodes
        .iter()
        .enumerate()
        .map(|(_, node)| included[*node])
        .collect();
    Ok(Subset {
        draws,
        objects,
        lights,
    })
}

pub(crate) struct CaptureView {
    pub(crate) view: crate::view::ViewState,
    uniforms: wgpu::Buffer,
    sampled_uniforms: Vec<Uniform>,
    camera_selections: Option<wgpu::Buffer>,
}
impl CaptureView {
    pub(crate) fn new(
        s: &InstalledScene,
        r: &Renderer,
        shared: &mut FrameSlot,
        desc: &crate::views::ViewDesc,
        image: &InstalledImageRender,
    ) -> Result<Self> {
        let subset = Subset {
            draws: image.draws.clone(),
            objects: image.objects.clone(),
            lights: image.lights.clone(),
        };
        let view = crate::view::ViewState::new(
            s,
            r,
            &image.settings,
            ReadbackPrecision::Float,
            desc,
            &shared.queries,
            crate::view::SharedBindings {
                points: &shared.point_publication.buffer,
                rays: shared.rays.as_ref(),
                compute: shared.compute_commands.as_ref(),
                objects: &shared.objects,
                deforms: &shared.deforms,
                materials: &shared.materials,
                shadow_bind: &shared.shadow_bind,
                output: Some(&shared.output),
            },
            Some(&subset),
        )?;
        let uniforms = crate::gpu_memory::buffer(&r.device, &wgpu::BufferDescriptor {
            label: Some("capture sample uniforms"),
            size: view.sample_count as u64 * std::mem::size_of::<Uniform>() as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_selections = view.camera_cull.sample_values(&r.device, view.sample_count as u32);
        Ok(Self {
            sampled_uniforms: Vec::with_capacity(view.sample_count),
            view,
            uniforms,
            camera_selections,
        })
    }

    pub(crate) fn prepare(
        &mut self,
        s: &InstalledScene,
        frame: &mut FrameSlot,
        image: &InstalledImageRender,
        base: Uniform,
        changes: &mut values::ChangeTracker,
        id: usize,
        dependency_changed: bool,
        lighting_revisions: [u64; 2],
    ) -> Result<bool> {
        self.sampled_uniforms.clear();
        for sample in 0..self.view.sample_count as u32 {
            self.sampled_uniforms.push(values::capture_uniform(s, frame, image, base, sample)?);
        }
        // GPU-authored scene data has no CPU input identity.
        // Their consumers replay even at the same time; compare observable inputs
        // only for captures whose complete state can be retained here.
        if s.compute.is_none() && !s.has_media {
            let mut inputs = Vec::new();
            // The shared stateless volume has its own exact input tracker. Retain
            // its revision, including updates made while this capture did not run.
            inputs.extend_from_slice(bytemuck::cast_slice(&lighting_revisions));
            let media=s.environment.media_request(&s.animation,&frame.sample_request)?.unwrap_or_else(||frame.sample_request.clone());
            inputs.extend_from_slice(&s.environment.sample_key(&frame.sample_request,&media)?);
            // Captures share the current parent sample's pose (independent capture
            // shutters are rejected). Include every immutable footage selection:
            // secondary hits, shadows and upstream captures can observe it too.
            // Use selected source frames, not publication revisions, so a seek or
            // another slot's overwrite cannot force replay of identical inputs.
            for source in s.footage.values().filter(|source|source.per_sample) {
                let request=frame.sample_request.exact()?;
                let clock=if source.canvas.is_some() {
                    source.canvas_instant(s,&frame.animation_values,&request)?
                } else {
                    let base=source.time.as_ref().map(|t|vork_time::exact::Germ::point(t.exact())).unwrap_or_else(||request.clone());
                    Some(s.animation.image_time_exact(&frame.animation_values,&request,&source.name,base)?)
                };
                inputs.extend(source.sample_key(clock.as_ref(),s.render.fps)?);
            }
            // Native raster shading has no time input. Programs, media, compositors
            // and animated path seeds do; keep the actual uniforms unchanged.
            for uniform in &self.sampled_uniforms {
                let mut input = *uniform;
                if !image.time_dependent { input.scene[..3].fill(0.); input.composite_shutter.fill(0.); }
                inputs.extend_from_slice(bytemuck::bytes_of(&input));
            }
            // Ray hits can see casters outside the image
            // subtree. These are evaluated values, not time/animation heuristics.
            for &object in &s.dynamic_objects {
                inputs.extend_from_slice(bytemuck::bytes_of(&frame.object_values[object].gpu()));
            }
            inputs.extend_from_slice(bytemuck::cast_slice(&frame.evaluated_lights));
            if let Some(materials) = &frame.materials {
                frame.changes.append_buffer_inputs(&materials.records, &mut inputs);
            }
            frame.geometry_chain.append_inputs(&frame.changes,&mut inputs);
            let changed = changes.changed(values::ChangeKey::Capture(id), &inputs);
            if !(changed || dependency_changed) {
                return Ok(false);
            }
        }
        if let Some(values) = &self.camera_selections {
            for (sample, uniform) in self.sampled_uniforms.iter().enumerate() {
                let projection = if uniform.sampling[3] > 1 {
                    let radius=crate::sampling::lens_selection_radius(uniform.post[3], uniform.lens[1], uniform.params[0], uniform.params[1]);
                    crate::sampling::selection_projection(uniform.vp, self.view.internal, radius)
                } else { uniform.vp };
                self.view.camera_cull.prepare_sample(projection, &frame.object_values, &frame.geometry_bounds,
                    &mut frame.changes, values, sample as u32);
            }
        }
        self.view.post.compositor_inputs.upload(s, base.scene[0], &mut frame.changes)?;
        frame.changes.upload(
            &self.uniforms,
            0,
            bytemuck::cast_slice(&self.sampled_uniforms),
        );
        let lights: Vec<_> = frame
            .evaluated_lights
            .iter()
            .zip(&image.lights)
            .map(|(light, included)| {
                let mut light = *light;
                if !included {
                    light.radiance[..3].fill(0.);
                }
                light
            })
            .collect();
        self.view.lighting.upload_lights(&mut frame.changes, &lights)?;
        let view = self.sampled_uniforms[0].view;
        if !self.view.transparent_order.is_empty() {
            let indirect = self.view.transparent_indirect.is_some();
            let sorted = passes::transparent::sort(
                s,
                &frame.object_values,
                &mut self.view.transparent_order,
                view,
                indirect,
            );
            if let Some((objects, args, runs)) = sorted {
                for (index, value) in objects {
                    frame.changes.upload(
                        self.view.transparent_instances.as_ref().unwrap(),
                        (index * std::mem::size_of::<u32>()) as u64,
                        bytemuck::bytes_of(&value),
                    );
                }
                frame.changes.upload(
                    self.view.transparent_indirect.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&args),
                );
                self.view.transparent_runs = runs;
            }
        }
        Ok(true)
    }

    pub(crate) fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &Renderer,
        environment:&crate::environment::Environment,
        queries: &wgpu::QuerySet,
        image: &InstalledImageRender,
        last: bool,
        denoise_timing: &mut denoise::Timing,
        segments: &mut crate::command_finish::Segments,
        changes: &mut crate::values::ChangeTracker,
    ) -> Result<()> {
        let samples = self.view.sample_count as u32;
        let frame = &mut self.view;
        for sample in 0..samples {
            frame.sample_index = sample as usize;
            encoder.copy_buffer_to_buffer(
                &self.uniforms,
                u64::from(sample) * std::mem::size_of::<Uniform>() as u64,
                &frame.uniform,
                0,
                std::mem::size_of::<Uniform>() as u64,
            );
            environment.publish_metadata(encoder,&frame.uniform);
            if let Some(values) = &self.camera_selections {
                frame.camera_cull.copy_sample(encoder, changes, values, sample);
            }
            let bits = crate::graph::RunBits {
                color: frame.color_enabled, final_sample: sample + 1 == samples,
                probes: false, fibre: false, images: false, pixels: false,
            };
            let view = &*frame;
            for step in &view.steps {
                if bits.runs(step.run) {
                    if step.pass == crate::graph::Pass::Denoise {
                        if let Some(stage) = &view.post.reconstruction { stage.encode(encoder,None); }
                        if let Some(stage) = &view.post.denoiser {
                            stage.run(renderer, encoder, view.sample_count, 1., denoise_timing, segments)?;
                        }
                    } else {
                        crate::graph::encode_view(encoder, view, step, renderer, segments)?;
                    }
                }
            }
        }
        image.publish(encoder,frame.publication.as_ref().expect("image publication installed"),
            [frame.width,frame.height],last.then_some((queries,timing::query("images")+1)));
        Ok(())
    }
}

#[cfg(all(test, vork_private_tests))]
include!("render_images_reuse_tests_private.rs");

pub(crate) fn publication_pipeline(r: &Renderer) -> wgpu::ComputePipeline {
    let shader = native::shader_module(&r.device, &native::RENDER_IMAGES_PUBLISH);
    let bind_layout = r.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("render image publication"),
        entries: &PUBLICATION_LAYOUT,
    });
    let layout = r.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render image publication"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });
    r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("render image publication"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some(native::RENDER_IMAGES_PUBLISH.entry),
        compilation_options: Default::default(),
        cache: r.cache.as_ref(),
    })
}
