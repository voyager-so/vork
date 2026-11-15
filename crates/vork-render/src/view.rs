//! Per-camera resources over one shared installed and evaluated scene.
use crate::*;
/// Immutable resources and recorded bindings for one installed view/ring variant.
pub struct View {
    pub(crate) diagnostic_name: String,
    pub(crate) decoupled: Option<passes::decoupled::View>,
    pub(crate) aov_resolve: Option<crate::aov::Resolve>,
    pub(crate) aovs: Vec<vork_scene::v2::Aov>,
    pub(crate) has_catchers: bool,
    variants: std::sync::OnceLock<ShadingBundles>,
    pending_bundles: std::sync::Mutex<Option<(ShadingBundles, Rerecord)>>,
    pub(crate) steps: Vec<crate::graph::Step>,
    graph: crate::graph::Graph,
    pub(crate) uniform: wgpu::Buffer,
    pub(crate) depth: wgpu::TextureView,
    pub(crate) ids: wgpu::TextureView,
    pub(crate) normals: wgpu::TextureView,
    pub(crate) motion: wgpu::TextureView,
    pub(crate) ambient: wgpu::TextureView,
    pub(crate) denoising_albedo: Option<wgpu::TextureView>,
    pub(crate) subsurface: Option<passes::subsurface::Resources>,
    pub(crate) depth_bundle: wgpu::RenderBundle,
    pub(crate) camera_cull: passes::camera_cull::Resources,
    pub(crate) transparent_instances: Option<wgpu::Buffer>,
    pub(crate) transparent_indirect: Option<wgpu::Buffer>,
    pub(crate) post: passes::resolve::Resources,
    pub(crate) publication: Option<wgpu::BindGroup>,
    pub(crate) float_output: bool,
    /// Output resolution: colour output, packed planes and readback.
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Raster and screen-effect resolution.
    pub(crate) internal: [u32; 2],
    pub(crate) lighting: ViewLighting,
}
/// Values and upload handles for one ring entry over its immutable recorded work.
pub struct ViewState {
    recorded: std::sync::Arc<View>,
    /// Optional graph choice, captured once per exposure independently of classic variants.
    pub(crate) fused_shading: bool,
    pub(crate) queries: wgpu::QuerySet,
    pub(crate) output_bind: wgpu::BindGroup,
    /// Tonemapped GPU output for downstream encoding.
    pub output: wgpu::Texture,
    /// Produce GPU colour; CPU readback is independent.
    pub write_color: bool,
    pub(crate) color_enabled: bool,
    pub(crate) transparent_order: Vec<usize>,
    pub(crate) transparent_runs: Vec<usize>,
    pub(crate) readback_outputs: ReadbackOutputs,
    pub(crate) sample_index: usize,
    pub(crate) sample_count: usize,
    pub(crate) visibility_samples: u32,
    pub(crate) nominal_sample: bool,
    pub(crate) composite_shutter: [f32; 4],
    pub(crate) composite_steps: u32,
}
impl std::ops::Deref for ViewState {
    type Target = View;
    fn deref(&self) -> &View {
        &self.recorded
    }
}

pub(crate) struct ViewLighting {
    pub(crate) active: bool,
    pub(crate) uniform: wgpu::Buffer,
    pub(crate) lights: wgpu::Buffer,
    pub(crate) analytic_light_tree: wgpu::Buffer,
    pub(crate) cluster_config: wgpu::Buffer,
    pub(crate) cluster_lights: wgpu::Buffer,
    pub(crate) clusters: Option<(wgpu::ComputePipeline, wgpu::BindGroup, [u32; 3])>,
}

impl ViewLighting {
    pub(crate) fn upload_lights(&self, changes: &mut crate::values::ChangeTracker, lights: &[Light]) -> Result<()> {
        if !self.active || lights.is_empty() { return Ok(()); }
        let bytes = bytemuck::cast_slice(lights);
        changes.upload(&self.lights, 0, bytes);
        let tree = &self.analytic_light_tree;
        changes.derive_upload(crate::values::ChangeKey::LightTree(tree.clone()), bytes, tree, 0, || {
            Ok((0u32, bytemuck::cast_slice(&crate::lights::distribution::nodes(lights)?).to_vec()))
        })?;
        Ok(())
    }
}

pub(crate) struct SharedBindings<'a> {
    pub compute: Option<&'a wgpu::Buffer>,
    pub points: &'a wgpu::Buffer,
    pub objects: &'a wgpu::Buffer,
    pub deforms: &'a [Option<EvaluatedGeometry>],
    pub materials: &'a Option<materials::FrameMaterials>,
    pub shadow_bind: &'a wgpu::BindGroup,
    pub output: Option<&'a wgpu::Texture>,
    pub rays: Option<&'a crate::rt::FrameRays>,
}
impl ViewState {
    pub(crate) fn refresh_variants(&mut self,s:&InstalledScene,r:&Renderer)->Result<()> {
        self.recorded.refresh_variants(s,r)?;
        self.fused_shading=self.decoupled.as_ref().is_some_and(passes::decoupled::fused_ready);
        Ok(())
    }
    /// Keep the installed recording borrowed independently while frame passes
    /// update ring state (ray acceleration structures, for example).
    pub(crate) fn recording(&self) -> std::sync::Arc<View> {
        self.recorded.clone()
    }

    pub(crate) fn new(
        s: &InstalledScene,
        r: &Renderer,
        settings: &vork_scene::v2::Render,
        precision: ReadbackPrecision,
        desc: &crate::views::ViewDesc,
        queries: &wgpu::QuerySet,
        shared: SharedBindings<'_>,
        subset: Option<&crate::render_images::Subset>,
    ) -> Result<Self> {
        let capture = desc.passes == crate::views::PassSet::Image;
        let (mut dependencies,_)=crate::compositor::dependencies(&s.compositor,&settings.compositor);
        if !capture {dependencies=s.main_outputs;}
        dependencies.images &= s.graph.views.image_renders().next().is_some();
        let view_graph=crate::graph::Graph::for_outputs(&s.plan,!capture && s.readback.any(),Some(dependencies))?;
        use crate::graph::Pass;
        let raster=view_graph.runs(Pass::Depth) || view_graph.runs(Pass::Visibility) || view_graph.runs(Pass::Forward);
        let scene_work=raster || view_graph.runs(Pass::Path) || view_graph.runs(Pass::Shade);
        let unused=(!scene_work).then(||crate::render_images::Subset {draws:vec![false;s.draws.len()],objects:vec![false;s.objects.len()],lights:vec![false;s.light_plan.nodes.len()]});
        let subset=unused.as_ref().or(subset);
        let decoupled_enabled=s.plan.decoupled && (view_graph.runs(Pass::Visibility) || (s.pipelines.field_controls && view_graph.runs(Pass::Forward)));
        let label = match &desc.kind {
            crate::views::ViewKind::Camera(node) => format!("camera view {node}"),
            crate::views::ViewKind::Image { name, .. } => format!("image view {name}"),
        };
        ensure!(
            desc.extent == settings.resolution,
            "view extent differs from installed render settings"
        );
        let (exposure_samples, visibility_samples) = match &desc.kind {
            crate::views::ViewKind::Image { render, .. } => (render.exposure_samples, render.visibility_samples),
            _ => crate::plan::sampling_counts(settings, true),
        };
        let d = &r.device;
        let [w, h] = desc.extent;
        let internal = settings.resolution;
        let [iw, ih] = internal;
        ensure!(
            w <= d.limits().max_texture_dimension_2d && h <= d.limits().max_texture_dimension_2d,
            crate::gpu_memory::device_limit("render resolution", format_args!("{w}x{h} pixels"), format_args!("max_texture_dimension_2d={} pixels per axis",d.limits().max_texture_dimension_2d), "reduce render.resolution")
        );
        let float_output = precision == ReadbackPrecision::Float;
        let aov_base = if float_output {
            if capture {
                (constants::FLOAT_COLOR_PLANE + 4) as u32
            } else {
                constants::FLOAT_OUTPUT_PLANES
            }
        } else {
            constants::PACKED_OUTPUT_PLANES
        };
        // Packed colour needs one delivery plane and four float display-input
        // planes. Keep fixed control/AOV offsets only when a consumer uses them.
        let output_planes=if !capture && !float_output && !s.main_outputs.controls()
            && !dependencies.aov_output && !s.has_catchers {5} else {
            aov_base+if dependencies.aov_output {s.render.aovs.len() as u32*4} else {0} + if s.has_catchers {4} else {0} + if !capture {4} else {0}
        };
        let output_pixels = u64::from(w) * u64::from(h);
        let pixels = u64::from(iw) * u64::from(ih);
        let aov_bytes=if view_graph.uses("AOV resolved") {(pixels*s.render.aovs.len() as u64*16).max(16)} else {16};
        for (name, bytes) in [
            ("AOV resolved",aov_bytes),
            ("catcher radiance sums",if s.has_catchers && view_graph.runs(Pass::Path) {pixels*48} else {48}),
            (
                "packed output",
                output_pixels * 4 * u64::from(output_planes) + crate::frame::invalid_pixel_bitmap_bytes(output_pixels),
            ),
            (
                "linear sample accumulation",
                if exposure_samples > 1 || s.has_catchers || !settings.compositor.is_empty() || settings.settings.integrator.denoiser().is_some() || settings.settings.reconstruct {
                    pixels * 16
                } else {
                    16
                },
            ),

        ] {
            ensure!(
                bytes <= d.limits().max_storage_buffer_binding_size
                    && bytes <= d.limits().max_buffer_size,
                crate::gpu_memory::device_limit(format_args!("render.aovs: {} declared; {name}",s.render.aovs.len()), format_args!("{bytes} bytes at {w}x{h}"), format_args!("max_storage_buffer_binding_size={} bytes, max_buffer_size={} bytes",d.limits().max_storage_buffer_binding_size,d.limits().max_buffer_size), "reduce resolution or the number of render.aovs")
            );
        }
        if s.pipelines.denoise && view_graph.runs(Pass::Denoise) {
            let guide_bytes = pixels * 32;
            ensure!(guide_bytes <= d.limits().max_storage_buffer_binding_size && guide_bytes <= d.limits().max_buffer_size,
                crate::gpu_memory::device_limit("OIDN guides", format_args!("{guide_bytes} bytes"), format_args!("storage binding={} bytes, buffer={} bytes",d.limits().max_storage_buffer_binding_size,d.limits().max_buffer_size), "reduce resolution or disable denoising with --set denoise=false"));
            ensure!(pixels * 48 <= d.limits().max_buffer_size,
                crate::gpu_memory::device_limit("OIDN staging", format_args!("{} bytes",pixels*48), format_args!("max_buffer_size={} bytes",d.limits().max_buffer_size), "reduce resolution or disable denoising with --set denoise=false"));
        }
        let uniform = crate::gpu_memory::buffer(&d, &wgpu::BufferDescriptor {
            label: Some(&label),
            size: std::mem::size_of::<Uniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let has_transparent=s.plan.draws.iter().enumerate().any(|(i,f)|f.transparent && subset.is_none_or(|s|s.draws[i]));
        let objects = shared.objects;
        let transparent_instancing = d
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
            && has_transparent;
        // Sort instance indices, never object records: ray hits and raster fragments
        // must keep the same installed object identity through every view.
        let transparent_instances = has_transparent.then(|| {
            init(
                d,
                "sorted transparent instances",
                bytemuck::cast_slice(&(1..=s.objects.len() as u32).collect::<Vec<_>>()),
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            )
        });
        let transparent_indirect = transparent_instancing.then(|| {
            crate::gpu_memory::buffer(&d, &wgpu::BufferDescriptor {
                label: Some("transparent instance runs"),
                size: (s.draws.len().max(1)
                    * std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>())
                    as u64,
                usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let no_clusters = init(
            d,
            "unclustered config",
            &[0; 32],
            wgpu::BufferUsages::UNIFORM,
        );
        let empty_mask = init(d, "unclustered mask", &[0; 4], wgpu::BufferUsages::STORAGE);
        let camera_cull = if view_graph.runs(Pass::Depth) {passes::camera_cull::Resources::new(s,r,&uniform,&objects,
            shared.materials.as_ref().map_or_else(||s.material_table.records(),|m|&m.records),shared.compute,subset)} else {passes::camera_cull::Resources::unused(d)};
        let make_global = |uniform: &wgpu::Buffer,
                           lights: &wgpu::Buffer,
                           config: &wgpu::Buffer,
                           mask: &wgpu::Buffer,
                           objects: &wgpu::Buffer, tree: Option<&wgpu::Buffer>| {
            global(s, d, uniform, lights, config, mask, objects, shared.points, tree)
        };
        let dummy_lights = init(
            d,
            "empty light",
            bytemuck::bytes_of(&Light::zeroed()),
            wgpu::BufferUsages::STORAGE,
        );
        let global = make_global(&uniform, &dummy_lights, &no_clusters, &empty_mask, &objects, None);
        let forward_global;
        // Compute work the pass modules record for this view's graph steps.
        let mut work = crate::graph::Recorder::default();
        let capture_owner = capture.then_some(label.as_str());
        let targets = Targets::new(s, r, internal, aov_bytes, &uniform, &view_graph, &mut work, capture_owner)?;
        // Float-only capture resolve never writes the display output. Reuse
        // the existing binding rather than allocating a throwaway texture.
        let output = shared.output.cloned().unwrap_or_else(|| {
            texture(
                d,
                "colour output",
                w,
                h,
                1,
                r.output_format(),
                wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            )
        });
        let shadow_bind = shared.shadow_bind;
        let deforms = shared.deforms;
        let frame_materials = shared.materials;
        let software_resources=s.pipelines.software_raster && raster;
        let visibility = (decoupled_enabled || software_resources).then(|| passes::decoupled::Sampling::new(d, [iw, ih], visibility_samples));
        let fragments = (decoupled_enabled || software_resources).then(|| passes::decoupled::Fragments::new(s, r, [iw, ih], visibility_samples, visibility.as_ref().unwrap(), &deforms)).transpose()?;
        let fragments_bind = fragments.as_ref().and_then(|f| f.bind.clone());
        let fragments_geometry = fragments.as_ref().and_then(|f| f.geometry.clone());
        let recorder = crate::bundles::Recorder {
            shadow_bind,
            graph: &view_graph,
            camera_instances: Some(&camera_cull.instances),
            transparent_instances: transparent_instances.as_ref(),
            compute: shared.compute,
            included: subset.map(|s| s.draws.as_slice()),
            included_objects: subset.map(|s| s.objects.as_slice()),
            fibre_capture: None,
            visibility: visibility.as_ref(),
            fragments: fragments_bind.as_ref(),
            fragments_geometry: fragments_geometry.as_ref(),
            material_bind: frame_materials
                .as_ref()
                .map_or_else(|| s.material_table.bind(), |m| &m.bind),
            s,
            deforms: &deforms,
            r,
        };
        use crate::bundles::Kind;
        let depth_bundle = recorder.record("depth plan", Kind::Depth, &global, camera_cull.commands(), None)?;
        let visibility_passes = decoupled_enabled.then(|| -> Result<_> { Ok([
            (0..visibility_samples)
                .map(|sample| recorder.record("decoupled visibility", Kind::Visibility(sample), &global, camera_cull.commands(), None))
                .collect::<Result<Vec<_>>>()?,
            if s.plan.transparent { (0..visibility_samples)
                .map(|sample| recorder.record("decoupled transparent fragments", Kind::TransparentFragments(sample), &global, camera_cull.commands(), None))
                .collect::<Result<Vec<_>>>()? } else { Vec::new() },
        ]) }).transpose()?;
        let lighting;
        let controls_bundle;
        {
            let batch = &s.light_plan;
            let batch_uniform = uniform.clone();
            let lights = init(
                d,
                "batch lights",
                &vec![0; if scene_work {batch.nodes.len().max(1)} else {1} * std::mem::size_of::<Light>()],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let analytic_light_tree = init(
                d, "analytic spatial light tree",
                &vec![0; crate::lights::distribution::node_count(if scene_work {batch.nodes.len()} else {0}) * std::mem::size_of::<crate::lights::tree::Node>()],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let list_stride = batch.nodes.len() as u32 + constants::CLUSTER_LIST_HEADER;
            let max_bytes = u64::from(d.limits().max_storage_buffer_binding_size)
                .min(d.limits().max_buffer_size);
            let (grid, tile, bytes) = light_grid(iw, ih, list_stride, max_bytes)?;
            let slices = grid[2];
            let enabled = s.plan.cluster && view_graph.runs(Pass::Clusters);
            let config_data = [
                grid[0],
                grid[1],
                list_stride,
                u32::from(enabled),
                iw,
                ih,
                slices,
                tile,
            ];
            let config = init(
                d,
                "cluster config",
                bytemuck::cast_slice(&config_data),
                wgpu::BufferUsages::UNIFORM,
            );
            let mask = s.transients.buffer(
                d,
                "cluster light lists",
                if enabled { bytes } else { 4 },
                wgpu::BufferUsages::STORAGE,
            );
            let clusters = if enabled {
                let pipeline = s
                    .pipelines
                    .cluster_pipeline
                    .get()
                    .filter(|_| s.plan.cluster)
                    .context("incomplete installed cluster plan")?;
                Some((
                    pipeline.clone(),
                    d.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("cluster culling"),
                        layout: &pipeline.get_bind_group_layout(0),
                        entries: &[
                            entry(0, batch_uniform.as_entire_binding()),
                            entry(1, lights.as_entire_binding()),
                            entry(2, config.as_entire_binding()),
                            entry(3, mask.as_entire_binding()),
                        ],
                    }),
                    grid,
                ))
            } else {
                None
            };
            let global = make_global(&batch_uniform, &lights, &config, &mask, &objects, Some(&analytic_light_tree));
            controls_bundle = recorder.record("opaque controls plan", Kind::Controls, &global, camera_cull.commands(), None)?;
            forward_global = global;
            lighting = ViewLighting {
                active: scene_work,
                uniform: batch_uniform,
                lights,
                analytic_light_tree,
                cluster_config: config,
                cluster_lights: mask,
                clusters,
            };
        }
        let decoupled = visibility_passes.zip(fragments).map(|(passes, fragments)| passes::decoupled::View::new(s, r, [iw, ih], visibility.as_ref().unwrap(), &targets, passes, fragments, deforms,
            [forward_global.clone(), recorder.material_bind.clone(), shadow_bind.clone()],frame_materials.as_ref().map(|m|&m.shader_errors),shared.rays.and_then(|r|r.emitter_weights.as_ref().map(|w|&w.buffer)))).transpose()?;
        let subsurface = if view_graph.runs(Pass::SubsurfaceDiffuse) {
            let visibility=decoupled.as_ref().context("subsurface visibility")?;
            Some(passes::subsurface::Resources::new(s, r, internal, settings.settings.samples, &uniform, &targets.resolved, &targets.post_ambient, &visibility.slots, &visibility.subsurface_records, visibility.refraction_source.as_ref(), &mut work)?)
        } else {
            None
        };
        // Ray reflections shade unseen hits with this view's lights.
        let reflections = passes::reflections::Resources::new(
            d,
            s,
            iw,
            ih,
            decoupled.as_ref(),
            &targets.post_ambient,
            &targets.resolved,
            &lighting.uniform,
            &lighting.lights,
            &lighting.analytic_light_tree,
            objects,
            &lighting.cluster_config,
            &lighting.cluster_lights,
            shared.rays,
            shared
                .materials
                .as_ref()
                .map_or_else(|| s.material_table.bind(), |m| &m.bind),
            view_graph.runs(Pass::Reflections),
            &mut work,
        )?;
        let transparent_bundles = transparent_bundles(&recorder, &forward_global, transparent_indirect.as_ref())?;
        // While feature variants compile in the background, keep what re-recording
        // the shading bundles needs; the first frame after they land switches.
        let recording = Rerecord {
            fragments: fragments_bind.clone(),
            fragments_geometry: fragments_geometry.clone(),
            shadow_bind: shadow_bind.clone(),
            compute: shared.compute.cloned(),
            subset: subset.map(|s| (s.draws.clone(), s.objects.clone())),
            deforms: deforms.to_vec(),
            material_bind: recorder.material_bind.clone(),
            forward: forward_global,
        };
        let rerecord=(!s.variants_ready()).then_some(recording);
        let ao_guides = decoupled.as_ref().map(|v| (&v.receivers, &v.normals));
        let ao = passes::ao::Resources::new(
            s, d, &lighting.uniform,
            ao_guides, [targets.resolved.texture().width(), targets.resolved.texture().height()], &targets.post_ambient, &targets.diffuse_response,
            shared.rays, &lighting.lights, Some(&lighting.analytic_light_tree), objects, &lighting.cluster_config, &lighting.cluster_lights,
            frame_materials.as_ref().map_or_else(|| s.material_table.bind(), |m| &m.bind), visibility_samples, view_graph.runs(Pass::Ao), &mut work,
        )?;
        let (mut post, output_bind) = passes::resolve::Resources::new(
            s,
            r,
            &lighting.uniform,
            ao_guides,
            &lighting.lights,
            &lighting.analytic_light_tree,
            &lighting.cluster_config,
            &lighting.cluster_lights,
            frame_materials
                .as_ref()
                .map_or_else(|| s.material_table.records(), |m| &m.records),
            objects,
            &targets,
            &reflections,
            &ao,
            &output,
            output_planes,
            exposure_samples,
            shared.rays,
            frame_materials
                .as_ref()
                .map_or_else(|| s.material_table.bind(), |m| &m.bind),
            frame_materials.as_ref().map(|m| &m.shader_errors),
            &mut work,
            desc,
            &settings.compositor,
            settings.settings.integrator.denoiser(),
            settings.settings.reconstruct,
            &view_graph,
            dependencies,
        )?;
        if let Some(stage) = &mut post.denoiser {
            let mut feedback = Vec::new();
            if let Some(materials) = frame_materials {
                // Fatal diagnostics and evaluation demand; the intervening
                // detail and nonfatal continuation warning are not rejections.
                feedback.push((materials.shader_errors.clone(), 0..4));
                feedback.push((materials.shader_errors.clone(), 12..16));
            }
            if let Some(view) = &decoupled {
                feedback.push((view.overflow.clone(), 4..8));
                feedback.push((view.fragments.buffer.clone(), 4..8));
            }
            stage.set_rejection_feedback(r, feedback);
        }
        let aov_resolve=if view_graph.runs(Pass::AovResolve) {crate::aov::Resolve::new(s,r,&uniform,&targets.aov_values,&post.pixels,[w,h],[iw,ih],aov_base)?} else {None};
        let publication = if let crate::views::ViewKind::Image { render, .. } = &desc.kind {
            // Native Metal does not expose StructuredBuffer.GetDimensions.
            let settings = init(d, "render image publication offset",
                bytemuck::cast_slice(&[(post.invalid_bitmap.start / 4) as u32, 0, 0, 0]),
                wgpu::BufferUsages::UNIFORM);
            Some(d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("render image publication"),
                layout: &render.pipeline.get_bind_group_layout(0),
                entries: &[
                    entry(0, post.pixels.as_entire_binding()),
                    entry(1, wgpu::BindingResource::TextureView(&render.level_zero)),
                    entry(2, settings.as_entire_binding()),
                ],
            }))
        } else {
            None
        };
        let Targets {
            depth,
            ids,
            normals,
            motion,
            ambient,
            denoising_albedo,
            ..
        } = targets;
        let bundles = ShadingBundles {
            controls: controls_bundle,
            transparent: transparent_bundles,
        };
        let (variants, pending_bundles) = match rerecord {
            Some(inputs) => (Default::default(), Some((bundles, inputs))),
            None => (std::sync::OnceLock::from(bundles), None),
        };
        let result = View {
            diagnostic_name: match &desc.kind {
                crate::views::ViewKind::Camera(_) => "camera".into(),
                crate::views::ViewKind::Image { name, .. } => format!("image:{name}"),
            },
            decoupled,
            camera_cull,
            aov_resolve,
            has_catchers: s.has_catchers,
            aovs: s.render.aovs.iter().cloned().chain(s.has_catchers.then(||vork_scene::v2::Aov {name:"shadow_catcher".into(),kind:vork_scene::v2::AovType::Color})).collect(),
            variants,
            subsurface,
            pending_bundles: std::sync::Mutex::new(pending_bundles),
            float_output,
            uniform: uniform,
            depth: depth,
            ids: ids,
            normals: normals,
            motion: motion,
            ambient: ambient,
            denoising_albedo,
            depth_bundle: depth_bundle,
            transparent_instances: transparent_instances,
            transparent_indirect: transparent_indirect,
            post: post,
            publication,
            steps: view_graph.record(!capture, work)?,
            graph: view_graph,
            width: w,
            height: h,
            internal,
            lighting,
        };
        let recorded = std::sync::Arc::new(result);
        Ok(Self {
            queries: queries.clone(),
            recorded,
            fused_shading: false,
            output_bind,
            output,
            write_color: !capture,
            color_enabled: true,
            readback_outputs: if capture {
                ReadbackOutputs {
                    float: FloatOutputs {
                        color: true,
                        ..FloatOutputs::NONE
                    },
                    ..Default::default()
                }
            } else {
                ReadbackOutputs::ALL
            },
            sample_index: 0,
            sample_count: exposure_samples as usize,
            visibility_samples,
            nominal_sample: false,
            composite_shutter: [0.; 4],
            composite_steps: 1,
            transparent_order: {
                let order:Vec<_>=s.plan.draws.iter().enumerate()
                    .filter_map(|(i,f)|(subset.is_none_or(|s|s.draws[i]) && f.transparent).then_some(i)).collect();
                order
            },
            transparent_runs: Vec::new(),
        })
    }
}

/// Owned recorder inputs for the one-time switch from superset to variant pipelines.
#[derive(Clone)]
pub(crate) struct Rerecord {
    fragments: Option<wgpu::BindGroup>,
    fragments_geometry: Option<wgpu::BindGroup>,
    compute: Option<wgpu::Buffer>,
    subset: Option<(Vec<bool>, Vec<bool>)>,
    deforms: Vec<Option<EvaluatedGeometry>>,
    material_bind: wgpu::BindGroup,
    shadow_bind: wgpu::BindGroup,
    forward: wgpu::BindGroup,
}
pub(crate) struct ShadingBundles {
    pub controls: wgpu::RenderBundle,
    /// Each transparent draw's controls bundle (the shade blends its colour).
    pub transparent: Vec<Option<wgpu::RenderBundle>>,
}
fn transparent_bundles(
    recorder: &crate::bundles::Recorder,
    global: &wgpu::BindGroup,
    indirect: Option<&wgpu::Buffer>,
) -> Result<Vec<Option<wgpu::RenderBundle>>> {
    let s = recorder.s;
    (0..s.draws.len())
        .map(|i| s.plan.draws[i].transparent.then(|| recorder.record("transparent controls", crate::bundles::Kind::TransparentControls, global, indirect, Some(i))).transpose())
        .collect()
}
/// Warm views borrow the published bundles without a lock. Only cold fallback
/// draws hold a guard, until the one-time switch releases their old bundles.
pub(crate) enum BundleRef<'a> {
    Ready(&'a ShadingBundles),
    Cold(std::sync::MutexGuard<'a, Option<(ShadingBundles, Rerecord)>>),
}
impl std::ops::Deref for BundleRef<'_> {
    type Target = ShadingBundles;
    fn deref(&self) -> &ShadingBundles {
        match self {
            Self::Ready(bundles) => bundles,
            Self::Cold(pending) => &pending.as_ref().expect("cold view bundles").0,
        }
    }
}
impl View {
    pub(crate) fn bundles(&self) -> BundleRef<'_> {
        if let Some(ready) = self.variants.get() {
            return BundleRef::Ready(ready);
        }
        let pending = self.pending_bundles.lock().expect("view variants poisoned");
        if pending.is_none() {
            return BundleRef::Ready(self.variants.get().expect("published view bundles"));
        }
        BundleRef::Cold(pending)
    }
    pub(crate) fn variants_pending(&self) -> bool {
        self.variants.get().is_none()
    }
    pub(crate) fn refresh_variants(&self, s: &InstalledScene, r: &Renderer) -> Result<()> {
        if !self.variants_pending() || !s.variants_ready() {
            return Ok(());
        }
        let mut pending = self.pending_bundles.lock().expect("view variants poisoned");
        let Some((_, p)) = pending.as_ref() else {
            return Ok(());
        };
        let recorder = crate::bundles::Recorder {
            shadow_bind: &p.shadow_bind,
            graph: &self.graph,
            camera_instances: Some(&self.camera_cull.instances),
            transparent_instances: self.transparent_instances.as_ref(),
            compute: p.compute.as_ref(),
            included: p.subset.as_ref().map(|s| s.0.as_slice()),
            included_objects: p.subset.as_ref().map(|s| s.1.as_slice()),
            fibre_capture: None,
            visibility: None,
            fragments: p.fragments.as_ref(),
            fragments_geometry: p.fragments_geometry.as_ref(),
            material_bind: &p.material_bind,
            s,
            deforms: &p.deforms,
            r,
        };
        use crate::bundles::Kind;
        let controls = recorder.record("opaque controls plan", Kind::Controls, &p.forward, self.camera_cull.commands(), None)?;
        let transparent = transparent_bundles(&recorder, &p.forward, self.transparent_indirect.as_ref())?;
        let _ = self.variants.set(ShadingBundles {
            controls,
            transparent,
        });
        // The completed set replaces the fallback and its recorder inputs.
        pending.take();
        Ok(())
    }
}

pub(crate) fn global(
    s: &InstalledScene,
    d: &wgpu::Device,
    uniform: &wgpu::Buffer,
    lights: &wgpu::Buffer,
    config: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    object_buffer: &wgpu::Buffer,
    points: &wgpu::Buffer,
    tree: Option<&wgpu::Buffer>,
) -> wgpu::BindGroup {
    let empty_tree = tree.is_none().then(|| init(d, "empty analytic tree", bytemuck::bytes_of(&crate::lights::tree::Node::zeroed()), wgpu::BufferUsages::STORAGE));
    let tree = tree.or(empty_tree.as_ref()).expect("analytic tree binding");
    d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("frame bindings"),
        layout: &s.pipelines.global_layout,
        entries: &[
            entry(0, uniform.as_entire_binding()),
            entry(1, object_buffer.as_entire_binding()),
            entry(11, points.as_entire_binding()),
            entry(10, tree.as_entire_binding()),
            entry(2, lights.as_entire_binding()),
            entry(7, config.as_entire_binding()),
            entry(8, mask.as_entire_binding()),
            entry(
                3,
                wgpu::BindingResource::TextureView(&s.environment.specular),
            ),
            entry(4, wgpu::BindingResource::Sampler(&s.environment.sampler)),
            entry(
                5,
                wgpu::BindingResource::TextureView(&s.environment.diffuse),
            ),
            entry(6, wgpu::BindingResource::TextureView(&s.environment.brdf)),
            entry(
                9,
                wgpu::BindingResource::TextureView(&s.environment.ltc),
            ),
        ],
    })
}

/// Raster attachments and resolved controls of one view, backed by the installed
/// scene's shared transients.
pub(crate) struct Targets {
    pub depth: wgpu::TextureView,
    pub resolved: wgpu::TextureView,
    pub aov_values: wgpu::Buffer,
    pub catcher: wgpu::Buffer,
    pub nonfinite: wgpu::Buffer,
    pub ids: wgpu::TextureView,
    pub motion: wgpu::TextureView,
    pub normals: wgpu::TextureView,
    pub ambient: wgpu::TextureView,
    pub denoising_albedo: Option<wgpu::TextureView>,
    pub denoising_guides: Option<wgpu::Buffer>,
    pub post_depth: wgpu::TextureView,
    pub post_ids: wgpu::TextureView,
    pub diffuse_response: wgpu::Buffer,
    pub post_normals: wgpu::TextureView,
    pub post_ambient: wgpu::TextureView,
    pub post_motion: wgpu::TextureView,
}
impl Targets {
    fn new(
        s: &InstalledScene,
        r: &Renderer,
        resolution: [u32; 2],
        aov_bytes: u64,
        uniform: &wgpu::Buffer,
        graph: &crate::graph::Graph,
        recorder: &mut crate::graph::Recorder,
        capture_owner: Option<&str>,
    ) -> Result<Self> {
        use crate::graph::Pass;
        let d = &r.device;
        let [w, h] = resolution;
        let buffer = |label: &str, size, usage| {
            let owned;
            let label = if let Some(owner) = capture_owner {
                owned = format!("{owner}: {label}"); owned.as_str()
            } else { label };
            s.transients.buffer(d,label,size,usage)
        };
        let nonfinite=s.transients.buffer(d,"nonfinite output diagnostics",(vork_slang::registry::NONFINITE_WORDS*4) as u64,wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST);
        let catcher=buffer("catcher radiance sums",
            if s.has_catchers && graph.runs(Pass::Path) {u64::from(w)*u64::from(h)*48} else {48},wgpu::BufferUsages::STORAGE);
        let aov_values=buffer("AOV resolved",if graph.uses("AOV resolved") {aov_bytes} else {16},wgpu::BufferUsages::STORAGE);
        let samples = s.pipelines.raster_samples;
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let view = |label, format, samples, usage| {
            let (tw,th)=if graph.uses(label) || label=="denoising albedo 4x" {(w,h)} else {(1,1)};
            s.transients
                .target(d, label, tw, th, samples, format, usage)
                .create_view(&Default::default())
        };
        let depth = view(
            "depth 4x",
            wgpu::TextureFormat::Depth32Float,
            samples,
            usage,
        );
        let resolved = view(
            "HDR resolved",
            wgpu::TextureFormat::Rgba32Float,
            1,
            usage | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
        );
        let ids = view("ID 4x", wgpu::TextureFormat::Rgba32Uint, samples, usage);
        let motion = view(
            "screen motion 4x",
            wgpu::TextureFormat::Rgba16Float,
            samples,
            usage,
        );
        let normals = view(
            "world normals 4x",
            wgpu::TextureFormat::Rgba16Float,
            samples,
            usage,
        );
        let ambient = view(
            "ambient 4x",
            wgpu::TextureFormat::Rgba16Float,
            samples,
            usage,
        );
        let denoising_albedo = (s.pipelines.denoise && graph.runs(Pass::Forward)).then(|| {
            view(
                "denoising albedo 4x",
                wgpu::TextureFormat::Rgba32Float,
                samples,
                usage,
            )
        });
        let control_view = |label, format| {
            view(
                label,
                format,
                1,
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            )
        };
        let post_depth = control_view("resolved depth", wgpu::TextureFormat::Rg32Float);
        let post_ids = control_view("resolved IDs", wgpu::TextureFormat::R32Uint);
        // Only raster indirect replacement consumes the exact three-channel
        // response. All other plans bind one unused RGB record.
        let diffuse_response = buffer( "indirect diffuse response",
            if graph.runs(Pass::Ao) {u64::from(w)*u64::from(h)*12} else {12},
            wgpu::BufferUsages::STORAGE);
        let post_normals = control_view("resolved normals", wgpu::TextureFormat::Rgba16Float);
        // The subsurface correction adds to the resolved ambient as an attachment.
        let post_ambient = view("resolved ambient", pipelines::resolved_ambient_format(samples), 1,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT);
        let denoising_guides = (s.pipelines.denoise && (graph.runs(Pass::Path) || graph.runs(Pass::Controls))).then(|| buffer( "denoising guides", u64::from(w)*u64::from(h)*32, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC));
        let post_motion = control_view("resolved motion", wgpu::TextureFormat::Rgba16Float);
        if graph.runs(crate::graph::Pass::Controls) {
        let control_resolve = s.pipelines.control_resolve_pipeline.get()
            .context("scheduled controls pass has no installed control-resolve pipeline")?;
        let mut control_entries = vec![
                entry(0, wgpu::BindingResource::TextureView(&depth)),
                entry(1, wgpu::BindingResource::TextureView(&ids)),
                entry(2, wgpu::BindingResource::TextureView(&normals)),
                entry(3, wgpu::BindingResource::TextureView(&ambient)),
                entry(4, wgpu::BindingResource::TextureView(&motion)),
                entry(5, wgpu::BindingResource::TextureView(&post_depth)),
                entry(6, wgpu::BindingResource::TextureView(&post_ids)),
                entry(7, wgpu::BindingResource::TextureView(&post_normals)),
                entry(8, wgpu::BindingResource::TextureView(&post_ambient)),
                entry(9, wgpu::BindingResource::TextureView(&post_motion)),
                entry(10, uniform.as_entire_binding()),
            ];
        if let (Some(albedo), Some(guides)) = (&denoising_albedo, &denoising_guides) {
            control_entries.push(entry(11, wgpu::BindingResource::TextureView(albedo)));
            control_entries.push(entry(12, guides.as_entire_binding()));
        }
        let control_resolve_bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("control resolve"),
            layout: &control_resolve.get_bind_group_layout(0),
            entries: &control_entries,
        });
        recorder.compute(
            crate::graph::Pass::Controls,
            "resolve controls once",
            "controls",
            false,
            vec![
                crate::graph::Dispatch::new(
                    control_resolve,
                    control_resolve_bind,
                    [w.div_ceil(8), h.div_ceil(8), 1],
                )
                .all_targets(),
            ],
        );
        }
        Ok(Self {
            depth,
            resolved,
            aov_values, catcher, nonfinite,
            ids,
            motion,
            normals,
            ambient,
            denoising_albedo,
            denoising_guides,
            post_depth,
            post_ids, diffuse_response,
            post_normals,
            post_ambient,
            post_motion,
        })
    }
}

// Coarsen the installed grid to fit the device, retaining every light in every
// list. Small batches and large images use the same culling path.
fn light_grid(
    width: u32,
    height: u32,
    stride: u32,
    max_bytes: u64,
) -> Result<([u32; 3], u32, u64)> {
    ensure!(
        u64::from(stride) * 4 <= max_bytes,
        crate::gpu_memory::device_limit("one cluster light list", format_args!("{} bytes",u64::from(stride)*4), format_args!("{max_bytes} bytes"), "reduce light count")
    );
    let mut tile = constants::CLUSTER_TILE_PIXELS;
    let mut slices = constants::CLUSTER_DEPTH_SLICES;
    loop {
        let grid = [width.div_ceil(tile), height.div_ceil(tile), slices];
        let bytes =
            u64::from(grid[0]) * u64::from(grid[1]) * u64::from(slices) * u64::from(stride) * 4;
        if bytes <= max_bytes {
            return Ok((grid, tile, bytes));
        }
        if grid[0] > 1 || grid[1] > 1 {
            tile *= 2;
        } else {
            slices = (slices / 2).max(1);
        }
    }
}

#[cfg(test)]
mod light_grid_tests {
    use super::*;
    #[test]
    fn large_light_lists_coarsen_without_dropping_entries() {
        for (width, height, stride, limit) in [
            (1920, 1080, 2, 128 << 20),
            (3840, 2160, 4097, 128 << 20),
            (8192, 8192, 65537, 1 << 20),
        ] {
            let (grid, tile, bytes) = light_grid(width, height, stride, limit).unwrap();
            assert_eq!(grid[0], width.div_ceil(tile));
            assert_eq!(grid[1], height.div_ceil(tile));
            assert!(grid[2] >= 1 && bytes <= limit);
            assert_eq!(
                bytes,
                u64::from(grid[0])
                    * u64::from(grid[1])
                    * u64::from(grid[2])
                    * u64::from(stride)
                    * 4
            );
        }
    }
}
