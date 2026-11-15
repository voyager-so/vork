//! Retained AO visibility and tile-local separable reconstruction.
use crate::*;
use wgpu::util::DeviceExt;
/// Retain one GI hit-shading specialization over every installed material.
pub(crate) fn ray_pipeline(r: &Renderer, cached: &std::sync::Arc<crate::pipelines::CachedPipelines>, features: crate::forward_variant::FeatureMask, probes: bool, mode: &crate::pipelines::VariantCompile) -> Result<crate::pipelines::Drawn<wgpu::ComputePipeline>> {
    let layout=crate::rt::gi_layout(&r.device,&cached.mat_layout,probes);
    cached.install_hit(r, crate::pipelines::ProgramKernel::Gi { features }, mode, layout, "ray-traced diffuse indirect")
}

#[cfg(test)]
thread_local! {
    static TEST_GI_BUDGET: std::cell::Cell<Option<u32>> = const {std::cell::Cell::new(None)};
    static TEST_GI_LIMIT: std::cell::Cell<Option<u64>> = const {std::cell::Cell::new(None)};
    static TEST_GI_BYTES: std::cell::Cell<u64> = const {std::cell::Cell::new(0)};
}

// Retain complete rows, reusing their records only after every ray batch finishes.
fn accumulation_band(limits: &wgpu::Limits, width: u32, height: u32, budget: u32) -> Result<(u32, u64)> {
    let row = u64::from(width) * constants::GI_QUERY_BYTES;
    gpu_memory::storage_binding(limits, "GI prepared queries: one row", row, "reduce resolution or choose --profile std/draft for a lower sample budget")?;
    let rows = (limits.max_storage_buffer_binding_size.min(limits.max_buffer_size) / row)
        .min(u64::from(height)).min(u64::from((budget / width).max(1))) as u32;
    let bytes = row * u64::from(rows);
    gpu_memory::storage_binding(limits, "GI prepared queries and accumulation", bytes, "reduce resolution or choose --profile std/draft for a lower sample budget")?;
    Ok((rows, bytes))
}

// Internal execution control: never changes scene sampling or estimator budgets.
// Metal needs smaller command buffers; retain ordinary grouping elsewhere
// unless the operator explicitly enables the same work limit.
fn default_query_budget(backend: wgpu::Backend) -> Option<u32> {
    (backend == wgpu::Backend::Metal).then_some(8192)
}

fn query_budget(backend: wgpu::Backend) -> Result<Option<u32>> {
    #[cfg(test)]
    if let Some(budget) = TEST_GI_BUDGET.with(|cap|cap.get()) { return Ok(Some(budget)); }
    let budget = match std::env::var("VORK_GI_QUERIES_PER_SUBMISSION") {
        Ok(value) => value.parse::<u32>().context("VORK_GI_QUERIES_PER_SUBMISSION must be a positive integer")?,
        Err(std::env::VarError::NotPresent) => return Ok(default_query_budget(backend)),
        Err(error) => return Err(error.into()),
    };
    anyhow::ensure!(budget > 0, "VORK_GI_QUERIES_PER_SUBMISSION must be positive");
    Ok(Some(budget))
}

fn query_tiles(width: u32, height: u32, budget: u32) -> Vec<[u32; 4]> {
    let tile_width = width.min(budget);
    let tile_height = (budget / tile_width).min(height);
    let mut tiles = Vec::new();
    for y in (0..height).step_by(tile_height as usize) {
        for x in (0..width).step_by(tile_width as usize) {
            tiles.push([x, y, tile_width.min(width-x), tile_height.min(height-y)]);
        }
    }
    tiles
}

// Each tile owns a disjoint output rectangle. Stage inputs remain immutable
// until every tile finishes; ray accumulation keeps its original band indices.
fn tiled_dispatches(
    d: &wgpu::Device, pipeline: &wgpu::ComputePipeline, label: &'static str,
    entries: &[wgpu::BindGroupEntry<'_>], binding: u32, prefix: &[u32],
    extent: [u32; 2], budget: u32,
) -> Vec<crate::graph::Dispatch> {
    let tiles = query_tiles(extent[0], extent[1], budget);
    let mut words = vec![0u32; tiles.len()*64];
    for (i,tile) in tiles.iter().enumerate() {
        words[i*64..i*64+prefix.len()].copy_from_slice(prefix);
        words[i*64+prefix.len()..i*64+prefix.len()+4].copy_from_slice(tile);
    }
    let parameters = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label), contents: bytemuck::cast_slice(&words), usage: wgpu::BufferUsages::UNIFORM,
    });
    tiles.iter().enumerate().map(|(i,tile)| {
        let mut entries = entries.to_vec();
        entries.iter_mut().find(|entry|entry.binding==binding).unwrap().resource =
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &parameters, offset: i as u64*256,
                size: std::num::NonZeroU64::new((prefix.len() as u64+4)*4),
            });
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label), layout: &pipeline.get_bind_group_layout(0), entries: &entries,
        });
        crate::graph::Dispatch::new(pipeline, bind, [tile[2].div_ceil(8),tile[3].div_ceil(8),1])
    }).collect()
}

pub(crate) struct Resources {
    pub output: wgpu::TextureView,
    pub incident: wgpu::TextureView,
    pub receivers: wgpu::TextureView,
}
impl Resources {
    pub(crate) fn new(
        s: &InstalledScene,
        d: &wgpu::Device,
        uniform: &wgpu::Buffer,
        guides: Option<(&wgpu::TextureView, &wgpu::TextureView)>,
        extent: [u32; 2],
        ambient: &wgpu::TextureView,
        diffuse_response: &wgpu::Buffer,
        rays: Option<&crate::rt::FrameRays>,
        lights: &wgpu::Buffer,
        light_tree: Option<&wgpu::Buffer>,
        objects: &wgpu::Buffer,
        cluster_config: &wgpu::Buffer,
        cluster_lights: &wgpu::Buffer,
        material: &wgpu::BindGroup,
        visibility_samples: u32,
        enabled: bool,
        recorder: &mut crate::graph::Recorder,
    ) -> Result<Self> {
        let [w, h] = extent;
        let configured_budget = query_budget(d.adapter_info().backend)?;
        let budget = configured_budget.unwrap_or(u32::MAX);
        let divisor = constants::GI_RESOLUTION_DIVISOR;
        let (iw, ih) = if enabled && s.plan.effects.indirect {
            (w.div_ceil(divisor), h.div_ceil(divisor))
        } else {
            (1, 1)
        };
        let image = |label, width, height, format| {
            s.transients
                .target(
                    d,
                    label,
                    width,
                    height,
                    1,
                    format,
                    wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                )
                .create_view(&Default::default())
        };
        let (aw, ah) = if enabled && s.plan.effects.indirect { (w.div_ceil(divisor), h.div_ceil(divisor)) } else { (1, 1) };
        let raw_ao = image("AO per-sample visibility depth normal", aw, ah, wgpu::TextureFormat::Rgba32Float);
        let incident_image = |label| image(label, iw, ih, wgpu::TextureFormat::Rgba32Float);
        let incident = incident_image("hemisphere per-sample incident radiance");
        // Query selection is shared by both blur axes and every upsampling tap.
        let cell_receivers = image("AO selected receivers", aw, ah, wgpu::TextureFormat::Rgba32Float);
        let mut dispatches = Vec::new();
        if enabled && s.plan.effects.indirect {
            let ao_view = image("AO filtered visibility depth normal", aw, ah, wgpu::TextureFormat::Rgba32Float);
            let incident_horizontal = incident_image("hemisphere incident radiance");
            let (receivers, post_normals) = guides.context("AO needs retained slot receivers")?;
            let demand = [
                image("GI demand a", aw, ah, wgpu::TextureFormat::R32Uint),
                image("GI demand b", aw, ah, wgpu::TextureFormat::R32Uint),
            ];
            let (demand_work, demand_ping) = demand_dispatches(
                d, s.pipelines.gi_demand_pipeline.get().context("GI demand pipeline")?,
                uniform, receivers, ambient, diffuse_response, &demand, extent, budget,
            );
            recorder.compute(crate::graph::Pass::Ao,"GI exact consumer demand","gi demand",false,demand_work);
            if configured_budget.is_some() {recorder.separate_last_every(1);}
            let frame_rays = rays.context("diffuse GI needs the scene TLAS")?;
            let batch=vork_slang::registry::GI_RAY_BATCH;
            let batches=(visibility_samples.max(1)*constants::GI_RAYS_PER_SAMPLE).div_ceil(batch);
            let limits = d.limits();
            #[cfg(test)]
            let limits = TEST_GI_LIMIT.with(|cap| {
                let mut limits=limits;
                if let Some(bytes)=cap.get() {limits.max_storage_buffer_binding_size=bytes;}
                limits
            });
            let (band_rows, accumulation_bytes) = accumulation_band(&limits, aw, ah, budget)?;
            let bands = ah.div_ceil(band_rows);
            let guide_bytes=u64::from(aw)*u64::from(ah)*32;
            gpu_memory::storage_binding(&d.limits(),"GI independent pilot guide",guide_bytes, "reduce resolution or choose --profile std/draft for a lower sample budget")?;
            let guide=s.transients.buffer(d,"GI independent pilot guide",guide_bytes,wgpu::BufferUsages::STORAGE);
            let hits={
                let accumulation=s.transients.buffer(d,"GI prepared queries and accumulation",accumulation_bytes,wgpu::BufferUsages::STORAGE);
                #[cfg(test)]
                TEST_GI_BYTES.with(|bytes|bytes.set(accumulation.size()));
                let geometry=frame_rays.hit_bindings();
                d.create_bind_group(&wgpu::BindGroupDescriptor {
                    label:Some("GI hit and accumulation buffers"),layout:&crate::rt::gi_read_layout(d),
                    entries:&[entry(0,wgpu::BindingResource::BufferArray(&geometry)),entry(1,accumulation.as_entire_binding()),entry(2,guide.as_entire_binding()),entry(3,light_tree.context("GI light tree")?.as_entire_binding())],
                })
            };
            let source_records=u64::from(bands)*(u64::from(batches)+1);
            anyhow::ensure!(source_records<=d.limits().max_buffer_size/256,
                crate::gpu_memory::device_limit("GI dispatch uniforms", format_args!("{} bytes",source_records*256), format_args!("max_buffer_size={} bytes",d.limits().max_buffer_size), "reduce resolution or samples with --set samples=N"));
            let mut source_words=vec![0u32;source_records as usize*64];
            let topology=s.rays.as_ref().unwrap();
            for band in 0..bands {
                for batch_index in 0..batches {
                    let at=((band*batches+batch_index)*64) as usize;
                    source_words[at..at+4].copy_from_slice(&[topology.emitters,topology.implicit[2],topology.implicit[3],0]);
                    source_words[at+4..at+8].copy_from_slice(&[batch_index*batch,band*band_rows,(ah-band*band_rows).min(band_rows),0]);
                }
            }
            for band in 0..bands {
                let at=((bands*batches+band)*64) as usize;
                source_words[at..at+4].copy_from_slice(&[topology.emitters,topology.implicit[2],topology.implicit[3],0]);
                source_words[at+4..at+8].copy_from_slice(&[0,band*band_rows,(ah-band*band_rows).min(band_rows),1]);
            }
            let sources=d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label:Some("diffuse source topology"),
                    contents:bytemuck::cast_slice(&source_words),
                    usage:wgpu::BufferUsages::UNIFORM,
            });
            let visibility = {
                let frame_rays = rays.context("ray GI needs the scene TLAS")?;
                let tlas = &frame_rays.tlas;
                let table = &s
                    .rays
                    .as_ref()
                    .context("ray GI needs the scene TLAS")?
                    .alpha;
                (
                    s.gi.as_ref().map(|pipeline|pipeline.current(false)),
                    "ray GI plan",
                    // rt::gi_entries: the scene globals, then GI's images and rays.
                    { let mut entries=pipelines::ray_globals(s,uniform,objects,lights,cluster_config,cluster_lights,light_tree.context("analytic light tree")?);
                      entries.extend([
                        entry(11, wgpu::BindingResource::TextureView(receivers)),
                        entry(12, wgpu::BindingResource::TextureView(post_normals)),
                        entry(13, wgpu::BindingResource::TextureView(&raw_ao)),
                        entry(14, wgpu::BindingResource::TextureView(&incident)),
                        entry(15, tlas.as_binding()),
                        entry(16, table.as_entire_binding()),
                        entry(17, wgpu::BindingResource::TextureView(&cell_receivers)),
                        entry(18,sources.as_entire_binding()),
                        entry(19,frame_rays.emitter_weights.as_ref().map_or(table,|weights|&weights.buffer).as_entire_binding()),
                        entry(20,wgpu::BindingResource::TextureView(ambient)),
                        entry(21,diffuse_response.as_entire_binding()),
                        entry(22,wgpu::BindingResource::TextureView(&demand[demand_ping])),
                    ]);
                    entries },
                    Some(material),
                )
            };
            let max_step=(w.min(h)/vork_slang::registry::GI_FILTER_EXTENT_PER_STEP).clamp(1,vork_slang::registry::GI_FILTER_MAX_STEP);
            let mut steps=Vec::new();let mut step=1;
            while step<=max_step {steps.push(step);step*=2;}
            let mut scale_words=vec![0u32;steps.len()*64];
            for (level,step) in steps.iter().enumerate() {scale_words[level*64]=*step;}
            let scales=d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:Some("GI reconstruction scales"),contents:bytemuck::cast_slice(&scale_words),usage:wgpu::BufferUsages::UNIFORM,
            });
            let reconstruction = |pipeline, label, vertical, level: u64| {
                let (output, input, out_incident, in_incident) = if vertical {
                    (&raw_ao, &ao_view, &incident, &incident_horizontal)
                } else {
                    (&ao_view, &raw_ao, &incident_horizontal, &incident)
                };
                (pipeline, label, vec![
                    entry(0, wgpu::BindingResource::TextureView(&cell_receivers)),
                    entry(1, uniform.as_entire_binding()),
                    entry(2, wgpu::BindingResource::TextureView(output)),
                    entry(3, wgpu::BindingResource::TextureView(input)),
                    entry(4, wgpu::BindingResource::TextureView(out_incident)),
                    entry(5, wgpu::BindingResource::TextureView(in_incident)),
                    entry(6,guide.as_entire_binding()),
                    entry(7,wgpu::BindingResource::Buffer(wgpu::BufferBinding {buffer:&scales,offset:level*256,size:std::num::NonZeroU64::new(16)})),
                ], None)
            };
            let mut pilot=visibility.clone();pilot.1="GI independent proposal pilots";
            let mut stages=vec![(pilot,true),(visibility,false)];
            for level in 0..steps.len() {
                stages.push((reconstruction(s.pipelines.ao_blur_horizontal_pipeline.get(),"GI horizontal reconstruction",false,level as u64),false));
                stages.push((reconstruction(s.pipelines.ao_blur_vertical_pipeline.get(),"GI vertical reconstruction",true,level as u64),false));
            }
            for ((pipeline, label, entries, material), pilot) in stages {
              for band in 0..if material.is_some() {bands} else {1} {
              for batch in 0..if material.is_some() && !pilot {batches} else {1} {
                let mut entries=entries.clone();
                if material.is_some() {
                    entries.iter_mut().find(|e|e.binding==18).unwrap().resource=wgpu::BindingResource::Buffer(wgpu::BufferBinding {buffer:&sources,offset:u64::from(if pilot {bands*batches+band} else {band*batches+batch})*256,size:std::num::NonZeroU64::new(32)});
                }
                let pipeline = pipeline.context("incomplete installed AO plan")?;
                let (binding, prefix, tile_extent) = if material.is_some() {
                    let at = (if pilot {bands*batches+band} else {band*batches+batch}) as usize*64;
                    (18, &source_words[at..at+8], [aw,(ah-band*band_rows).min(band_rows)])
                } else {
                    let offset = match entries.iter().find(|e|e.binding==7).unwrap().resource {
                        wgpu::BindingResource::Buffer(ref b) => b.offset as usize / 4,
                        _ => unreachable!(),
                    };
                    (7, &scale_words[offset..offset+4], [aw,ah])
                };
                for mut dispatch in tiled_dispatches(d,pipeline,label,&entries,binding,prefix,tile_extent,budget) {
                if material.is_some() {
                    // Ray GI hits read the irradiance volume for further bounces.
                    let probes = (s.plan.effects.ray_indirect).then(|| s.probes.as_ref().map(|p| p.read_bind(d))).flatten();
                    dispatch = dispatch.with_pipeline(s.gi.as_ref().unwrap().clone()).with_hits(hits.clone()).with_read_table(probes);
                }
                dispatches.push(match material {
                    Some(material) => dispatch.with_material(material),
                    None => dispatch,
                });
                }
              }
              }
              let span=if pilot {"gi pilots"} else if material.is_some() {"gi beauty"} else {"gi reconstruction"};
              recorder.compute(crate::graph::Pass::Ao,label,span,false,std::mem::take(&mut dispatches));
              if configured_budget.is_some() {recorder.separate_last_every(1);}
            }
        }
        Ok(Self {
            output: raw_ao,
            incident,
            receivers: cell_receivers,
        })
    }
}

// Three translated copies extend [-r,r] to [-r-s,r+s] without gaps
// whenever s <= 2r+1. This interval schedule reaches the exact requested
// radius, including non-power-of-two tails; it never caps reconstruction.
fn demand_steps([w,h]: [u32;2]) -> Vec<[u32;4]> {
    let max_step=(w.min(h)/vork_slang::registry::GI_FILTER_EXTENT_PER_STEP)
        .clamp(1,vork_slang::registry::GI_FILTER_MAX_STEP);
    let mut step=1;let mut radius=vork_slang::renderer_registry::REG_AO_UPSAMPLE_RADIUS as u32;
    while step<=max_step {radius+=vork_slang::renderer_registry::REG_GI_FILTER_RADIUS as u32*step;step*=2;}
    let mut steps=vec![[0,0,1,0]];
    for axis in 0..2 {
        let mut covered=0;
        while covered<radius {
            let stride=(2*covered+1).min(radius-covered);
            let mut value=[0;4];value[axis]=stride;steps.push(value);
            covered+=stride;
        }
    }
    steps
}

fn demand_dispatches(
    d: &wgpu::Device, pipeline: &wgpu::ComputePipeline, uniform: &wgpu::Buffer,
    receivers: &wgpu::TextureView, ambient: &wgpu::TextureView, response: &wgpu::Buffer,
    masks: &[wgpu::TextureView;2], extent: [u32;2], budget: u32,
) -> (Vec<crate::graph::Dispatch>, usize) {
    let steps=demand_steps(extent);
    let mut words=vec![0u32;steps.len()*64];
    for (i,step) in steps.iter().enumerate() {words[i*64..i*64+4].copy_from_slice(step);}
    let parameters=d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:Some("GI exact demand intervals"),contents:bytemuck::cast_slice(&words),usage:wgpu::BufferUsages::UNIFORM,
    });
    let mut dispatches=Vec::new();let mut ping=0;
    for (i,_) in steps.iter().enumerate() {
        let next=1-ping;
        let entries = [
                entry(0,uniform.as_entire_binding()),
                entry(1,wgpu::BindingResource::TextureView(receivers)),
                entry(2,wgpu::BindingResource::TextureView(ambient)),
                entry(3,response.as_entire_binding()),
                entry(4,wgpu::BindingResource::TextureView(&masks[ping])),
                entry(5,wgpu::BindingResource::TextureView(&masks[next])),
                entry(6,wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer:&parameters,offset:i as u64*256,size:std::num::NonZeroU64::new(16),
                })),
            ];
        dispatches.extend(tiled_dispatches(d,pipeline,"GI exact consumer demand",&entries,6,&steps[i],
            [extent[0].div_ceil(constants::GI_RESOLUTION_DIVISOR),extent[1].div_ceil(constants::GI_RESOLUTION_DIVISOR)],budget));
        ping=next;
    }
    (dispatches,ping)
}

pub(crate) fn demand_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let compute=wgpu::ShaderStages::COMPUTE;
    vec![
        pipelines::buffer_entry(0,compute,wgpu::BufferBindingType::Uniform),
        pipelines::texture_entry(1,wgpu::TextureSampleType::Float {filterable:false},false),
        pipelines::texture_entry(2,wgpu::TextureSampleType::Float {filterable:false},false),
        pipelines::buffer_entry(3,compute,wgpu::BufferBindingType::Storage {read_only:true}),
        pipelines::texture_entry(4,wgpu::TextureSampleType::Uint,false),
        wgpu::BindGroupLayoutEntry {binding:5,visibility:compute,ty:wgpu::BindingType::StorageTexture {
            access:wgpu::StorageTextureAccess::WriteOnly,format:wgpu::TextureFormat::R32Uint,view_dimension:wgpu::TextureViewDimension::D2,
        },count:None},
        pipelines::buffer_entry(6,compute,wgpu::BufferBindingType::Uniform),
    ].into_iter().map(|mut entry| {entry.visibility=compute;entry}).collect()
}

pub(crate) fn demand_layout(d: &wgpu::Device) -> wgpu::PipelineLayout {
    let group=d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label:Some("GI consumer demand"),entries:&demand_entries(),
    });
    d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:Some("GI consumer demand"),bind_group_layouts:&[Some(&group)],immediate_size:0,
    })
}

/// Native shaders need explicit layouts; Metal indices follow these bindings.
pub(crate) fn slang_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..8).map(|binding| wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: match binding {
            6 => wgpu::BindingType::Buffer {ty:wgpu::BufferBindingType::Storage {read_only:true},has_dynamic_offset:false,min_binding_size:None},
            1 | 7 => wgpu::BindingType::Buffer {ty:wgpu::BufferBindingType::Uniform,has_dynamic_offset:false,min_binding_size:None},
            2 | 4 => wgpu::BindingType::StorageTexture {access:wgpu::StorageTextureAccess::WriteOnly,format:wgpu::TextureFormat::Rgba32Float,view_dimension:wgpu::TextureViewDimension::D2},
            _ => wgpu::BindingType::Texture {sample_type:wgpu::TextureSampleType::Float {filterable:false},view_dimension:wgpu::TextureViewDimension::D2,multisampled:false},
        },count:None,
    }).collect()
}

pub(crate) fn slang_layout(d: &wgpu::Device) -> wgpu::PipelineLayout {
    let entries = slang_entries();
    let group = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Slang AO resources"),
        entries: &entries,
    });
    d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Slang AO"),
        bind_group_layouts: &[Some(&group)],
        immediate_size: 0,
    })
}

#[cfg(test)]
mod accumulation_tests {
    use super::*;
    #[test]
    fn gi_work_budget_defaults_to_metal_only() {
        assert_eq!(default_query_budget(wgpu::Backend::Metal), Some(8192));
        assert_eq!(default_query_budget(wgpu::Backend::Vulkan), None);
        assert_eq!(default_query_budget(wgpu::Backend::Noop), None);
    }

    #[test]
    fn gi_submission_tiles_cover_odd_and_wide_images_once() {
        for (width,height,budget) in [(1,1,1),(33,25,7),(33,25,37),(960,540,8192),(16384,3,8192)] {
            let mut visits=vec![0u8;(width*height) as usize];
            for [x,y,w,h] in query_tiles(width,height,budget) {
                assert!(w*h<=budget && w>0 && h>0);
                for row in y..y+h { for col in x..x+w {visits[(row*width+col) as usize]+=1;} }
            }
            assert!(visits.iter().all(|count|*count==1));
        }
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn gi_bands_preserve_float_output_with_independent_storage_limit() -> Result<()> {
        // Explicit GPU runs require renderer creation and all assertions to succeed.
        let r = crate::test_renderer()?;
        let mut scene=crate::test_scene(serde_json::json!({"version":2,
            "render":{"resolution":[66,50]},
            "nodes":[
                {"type":"mesh","geometry":crate::test_geometry::latitude_longitude(1.,32,16),"materials":[{"inputs":{"base_color":[0.2,0.4,0.7]}}]},
                {"type":"mesh","translation":[0,-1,0],"geometry":crate::test_geometry::plane([8.,8.],[1,1])},
                {"type":"camera","translation":[3,2,5],"target":[0,0,0]},
                {"type":"light","shape":"environment","color":[0.4,0.6,0.8]},
                {"type":"light","shape":"rectangle","translation":[-3,5,4],"target":[0,0,0],"energy":500,"size":3}
            ]}),Path::new("."))?;
        scene.render.settings=vork_scene::v2::Profile::load(vork_scene::v2::DEFAULT_PROFILE)?.resolve(
            &serde_json::from_value(serde_json::json!({"denoise":false}))?)?;
        let draw=|limit, budget| -> Result<Vec<[f32;4]>> {
            TEST_GI_BUDGET.with(|cap|cap.set(budget));
            // Only GI sees this limit: reflection and visibility admission use
            // the actual device limits. Seven-row bands include partial groups
            // and a partial final band; every band retains all sample batches.
            TEST_GI_LIMIT.with(|cap|cap.set(limit));
            TEST_GI_BYTES.with(|bytes|bytes.set(0));
            struct Reset;
            impl Drop for Reset {fn drop(&mut self) {TEST_GI_LIMIT.with(|cap|cap.set(None));TEST_GI_BUDGET.with(|cap|cap.set(None));}}
            let _reset=Reset;
            let mut installed=r.install_with_options(scene.clone(),Path::new("."),InstallOptions {
                variants:crate::pipelines::VariantCompile::Inline,..Default::default()})?;
            assert!(installed.plan.effects.ray_indirect);
            let mut slot=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
            let image=installed.render_frame(&r,&mut slot,0.)?;
            if let Some(limit)=limit {
                TEST_GI_BYTES.with(|bytes|assert!(bytes.get()>0 && bytes.get()<=limit,
                    "GI allocation {} exceeds independent limit {limit}",bytes.get()));
            }
            image.linear_rgba.context("float readback")
        };
        let whole=draw(None,None)?;
        let banded=draw(Some(33*7*constants::GI_QUERY_BYTES),None)?;
        for budget in [7,37,8192] {
            let tiled=draw(None,Some(budget))?;
            assert_eq!(bytemuck::cast_slice::<_,u32>(&whole),bytemuck::cast_slice::<_,u32>(&tiled),"budget {budget}");
        }
        assert!(whole.iter().any(|p|p[0]>0.01 && p[1]>0.01));
        crate::test_linear::assert_linear_eq(&whole,&banded);
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn gi_demand_matches_full_pixel_support_without_false_negatives() -> Result<()> {
        let r = crate::test_renderer()?;
        let d=&r.device;
        let layout=demand_layout(d);
        let module=crate::native::shader_module(d,&crate::native::GI_DEMAND_BUILD);
        let pipeline=r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:Some("exact demand qualification"),layout:Some(&layout),module:&module,
            entry_point:Some("build"),compilation_options:Default::default(),cache:r.cache.as_ref(),
        });
        for (w,h) in [(1u32,1u32),(3,5),(131,129),(257,255),(1920,1080)] {
          for diffuse in [false,true] {
            let (cw,ch)=(w.div_ceil(2),h.div_ceil(2));
            let texture=|width,height,format| d.create_texture(&wgpu::TextureDescriptor {
                label:Some("exact demand fixture"),size:extent(width,height),mip_level_count:1,sample_count:1,
                dimension:wgpu::TextureDimension::D2,format,
                usage:wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::STORAGE_BINDING|wgpu::TextureUsages::COPY_DST|wgpu::TextureUsages::COPY_SRC,view_formats:&[],
            });
            let receivers=texture(w,h,wgpu::TextureFormat::Rgba32Float);
            let ambient=texture(w,h,wgpu::TextureFormat::Rgba32Float);
            let masks=[texture(cw,ch,wgpu::TextureFormat::R32Uint),texture(cw,ch,wgpu::TextureFormat::R32Uint)];
            let mut geometry=vec![[0f32;4];(w*h) as usize];
            let mut baseline=geometry.clone();let mut response=vec![0f32;(w*h*3) as usize];
            // Sparse donors exercise every support edge, all channels, signed
            // and tiny FP32 payloads, and the clipped odd-size corner.
            let mut consumers=Vec::new();
            for (i,(x,y)) in [(0,0),(w-1,h-1),(w/2,h/2),(w/3,h/3),(w-1,0),(0,h-1)].into_iter().enumerate() {
                let at=(y*w+x) as usize;
                if i%2==0 {baseline[at][i%3]=if i==0 {-1.0e-25} else {0.001};}
                else {response[at*3+i%3]=1.0e-20;}
            }
            // A background pixel with nonzero payload is never a consumer.
            geometry[(h/3*w+w/3) as usize][3]=1.;
            for y in 0..h {for x in 0..w {
                let at=(y*w+x) as usize;
                if geometry[at][3]<1. && (baseline[at][..3].iter().any(|v|*v!=0.) ||
                    (diffuse && response[at*3..at*3+3].iter().any(|v|*v!=0.))) {consumers.push((x,y));}
            }}
            for (image,data) in [(&receivers,&geometry),(&ambient,&baseline)] {
                r.queue.write_texture(image.as_image_copy(),bytemuck::cast_slice(data),wgpu::TexelCopyBufferLayout {
                    offset:0,bytes_per_row:Some(w*16),rows_per_image:Some(h),
                },extent(w,h));
            }
            let response=init(d,"exact diffuse response",bytemuck::cast_slice(&response),wgpu::BufferUsages::STORAGE);
            let mut frame=Uniform::zeroed();frame.ao[3]=2.;frame.flags[3]=if diffuse {1.} else {0.};
            let uniform=init(d,"exact demand frame",bytemuck::bytes_of(&frame),wgpu::BufferUsages::UNIFORM);
            let rv=receivers.create_view(&Default::default());let av=ambient.create_view(&Default::default());
            let mv=masks.each_ref().map(|t|t.create_view(&Default::default()));
            let mut encoder=d.create_command_encoder(&Default::default());let mut ping=0;
            for step in demand_steps([w,h]) {
                let next=1-ping;
                let parameter=init(d,"exact demand interval",bytemuck::cast_slice(&[step[0],step[1],step[2],step[3],0,0,cw,ch]),wgpu::BufferUsages::UNIFORM);
                let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {
                    label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[
                        entry(0,uniform.as_entire_binding()),entry(1,wgpu::BindingResource::TextureView(&rv)),
                        entry(2,wgpu::BindingResource::TextureView(&av)),entry(3,response.as_entire_binding()),
                        entry(4,wgpu::BindingResource::TextureView(&mv[ping])),entry(5,wgpu::BindingResource::TextureView(&mv[next])),
                        entry(6,parameter.as_entire_binding()),
                    ],
                });
                {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(cw.div_ceil(8),ch.div_ceil(8),1);}
                ping=next;
            }
            let row=(cw*4).div_ceil(256)*256;
            let readback=d.create_buffer(&wgpu::BufferDescriptor {
                label:Some("exact demand readback"),size:u64::from(row*ch),usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false,
            });
            encoder.copy_texture_to_buffer(masks[ping].as_image_copy(),wgpu::TexelCopyBufferInfo {
                buffer:&readback,layout:wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(row),rows_per_image:Some(ch)},
            },extent(cw,ch));
            let submission=r.queue.submit([encoder.finish()]);let(tx,rx)=std::sync::mpsc::channel();
            readback.map_async(wgpu::MapMode::Read,..,move|v|{let _=tx.send(v);});r.wait(Some(submission))?;rx.recv()??;
            let mapped=readback.get_mapped_range(..);
            // Independent oracle: original clipped full-resolution square.
            let max_step=(w.min(h)/64).clamp(1,16);let mut stride=1;let mut radius=2i32;
            while stride<=max_step {radius+=2*stride as i32;stride*=2;}
            for y in 0..ch {for x in 0..cw {
                let lo=[((x as i32-radius)*2).max(0),((y as i32-radius)*2).max(0)];
                let hi=[((x as i32+radius+1)*2).min(w as i32),((y as i32+radius+1)*2).min(h as i32)];
                let expected=consumers.iter().any(|&(px,py)|(px as i32)>=lo[0] && (px as i32)<hi[0] && (py as i32)>=lo[1] && (py as i32)<hi[1]);
                let at=(y*row+x*4) as usize;
                let actual=u32::from_ne_bytes(mapped[at..at+4].try_into().unwrap());
                assert_eq!(actual,u32::from(expected),"{w}x{h}, diffuse={diffuse}, cell ({x},{y})");
            }}
          }
        }
        r.errors.check()?;Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn gi_proposal_keeps_sampled_cap_density_at_float_boundaries() -> Result<()> {
        let r = crate::test_renderer()?;
        let source=r#"
import gi_proposal;
import vork_sampling;
import vork_constants;
export float3 ray_eye() {return float3(0.);}
[[vk::binding(0,0)]] RWStructuredBuffer<float4> result;
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    let a=pixel_hash(id.x+1u);let b=pixel_hash(a);let c=pixel_hash(b);
    var axis=float3(float(a&65535u),float(b&65535u),float(c&65535u))/65536.-.5;
    if(dot(axis,axis)<1.e-6) axis=float3(0.,0.,1.);
    axis=normalize(axis);
    GiProposal guide={};guide.axis[0]=axis;guide.weight[0]=1.;guide.total=1.;
    let edge=(id.x&1u)==0u ? 1.-REG_FP32_EPSILON*.5 : 1.-1./65536.;
    uint sampled;
    let direction=gi_proposal_direction(guide,axis,float2(edge,float(c>>16u)/65536.),float2(0.),sampled);
    let minimum=REG_GI_GUIDE_MIX/(2.*REG_PI*(1.-REG_GI_GUIDE_COSINE));
    result[id.x]=float4(gi_proposal_pdf(guide,axis,direction,sampled)/minimum,
        gi_proposal_pdf(guide,axis,direction)/minimum,length(direction),float(sampled));
}
"#;
        let entries=[pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false})];
        let pipeline=crate::medium::test_pipeline(&r,source,&crate::medium::Native::default(),&entries,(1,1,1))?;
        let bytes=4096*16;
        let output=r.device.create_buffer(&wgpu::BufferDescriptor {label:None,size:bytes,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
        let read=r.device.create_buffer(&wgpu::BufferDescriptor {label:None,size:bytes,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,output.as_entire_binding())]});
        let mut encoder=r.device.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(4096,1,1);}
        encoder.copy_buffer_to_buffer(&output,0,&read,0,bytes);
        let submission=r.queue.submit([encoder.finish()]);let(tx,rx)=std::sync::mpsc::channel();
        read.map_async(wgpu::MapMode::Read,..,move|v|{let _=tx.send(v);});r.wait(Some(submission))?;rx.recv()??;
        let mapped=read.get_mapped_range(..);let values:&[[f32;4]]=bytemuck::cast_slice(&mapped);
        for (i,value) in values.iter().enumerate() {
            assert!(value[0].is_finite() && value[0]>=1.,"sampled cap omitted at {i}: {value:?}");
            assert!((value[2]-1.).abs()<1e-6 && value[3]==0.,"invalid cap sample at {i}: {value:?}");
        }
        eprintln!("unguarded boundary gaps: {} / {}",values.iter().filter(|v|v[1]<1.).count(),values.len());
        r.errors.check()?;Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn gi_reconstruction_preserves_incident_and_visibility_sums() -> Result<()> {
        let r = crate::test_renderer()?;
        let d=&r.device;let (w,h)=(17u32,9u32);let count=(w*h) as usize;
        let texture=|label,format| d.create_texture(&wgpu::TextureDescriptor {
            label:Some(label),size:extent(w,h),mip_level_count:1,sample_count:1,
            dimension:wgpu::TextureDimension::D2,format,
            usage:wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::STORAGE_BINDING|wgpu::TextureUsages::COPY_DST|wgpu::TextureUsages::COPY_SRC,view_formats:&[],
        });
        let receivers=texture("conservation planes",wgpu::TextureFormat::Rgba32Float);
        let ao=[texture("visibility a",wgpu::TextureFormat::Rgba32Float),texture("visibility b",wgpu::TextureFormat::Rgba32Float)];
        let incident=[texture("incident a",wgpu::TextureFormat::Rgba32Float),texture("incident b",wgpu::TextureFormat::Rgba32Float)];
        let mut positions=vec![[0f32;4];count];let mut visibility=vec![[0f32;4];count];let mut lighting=vec![[0f32;4];count];let mut guides=vec![[0f32;4];count*2];
        let mut expected=[0f64;4];
        for y in 0..h {for x in 0..w {
            let at=(y*w+x) as usize;let valid=x!=8;
            positions[at]=[x as f32*0.002,y as f32*0.002,((x*x+y)%5) as f32*0.0002,0.];
            let v=if x==4 && y==2 {-0.5} else if x==0 && y==0 {0.25} else {1.};
            let rgb=[if x==0 && y==0 {64.} else {0.},if x==7 && y==h-1 {32.} else {0.},if x==10 && y==3 {-1.} else if x>8 {0.2} else {0.}];
            let a=[v,if valid {1.+x as f32*0.05} else {0.},0.,0.];
            let b=if valid {[rgb[0],rgb[1],rgb[2],8.]} else {[0.;4]};
            visibility[at]=a;lighting[at]=b;
            for c in 0..3 {expected[c]+=f64::from(lighting[at][c]);}
            expected[3]+=f64::from(visibility[at][0]);
            guides[at*2+1]=[0.,0.03,16.,0.];
        }}
        r.queue.write_texture(receivers.as_image_copy(),bytemuck::cast_slice(&positions),wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(w*16),rows_per_image:Some(h)},extent(w,h));
        for (image,data) in [(&ao[0],&visibility),(&incident[0],&lighting)] {
            r.queue.write_texture(image.as_image_copy(),bytemuck::cast_slice(data),wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(w*16),rows_per_image:Some(h)},extent(w,h));
        }
        let guide=init(d,"raw visibility moments",bytemuck::cast_slice(&guides),wgpu::BufferUsages::STORAGE);
        let mut frame=Uniform::zeroed();frame.flags[3]=1.;
        let uniform=init(d,"conservation frame",bytemuck::bytes_of(&frame),wgpu::BufferUsages::UNIFORM);
        let layout=slang_layout(d);
        let pipelines=[(&crate::native::AO_BLUR_HORIZONTAL,"horizontal"),(&crate::native::AO_BLUR_VERTICAL,"vertical")].map(|(shader,entry)| {
            let module=crate::native::shader_module(d,shader);
            r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {label:Some("conservative reconstruction"),layout:Some(&layout),module:&module,entry_point:Some(entry),compilation_options:Default::default(),cache:r.cache.as_ref()})
        });
        let av=ao.each_ref().map(|t|t.create_view(&Default::default()));let iv=incident.each_ref().map(|t|t.create_view(&Default::default()));
        let rv=receivers.create_view(&Default::default());let mut encoder=d.create_command_encoder(&Default::default());let mut ping=0;
        for step in [1u32,2,4,8,16] {
            let scale=init(d,"conservation scale",bytemuck::cast_slice(&[step,0,0,0,0,0,w,h]),wgpu::BufferUsages::UNIFORM);
            for pipeline in &pipelines {
                let next=1-ping;
                let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[
                    entry(0,wgpu::BindingResource::TextureView(&rv)),entry(1,uniform.as_entire_binding()),
                    entry(2,wgpu::BindingResource::TextureView(&av[next])),entry(3,wgpu::BindingResource::TextureView(&av[ping])),
                    entry(4,wgpu::BindingResource::TextureView(&iv[next])),entry(5,wgpu::BindingResource::TextureView(&iv[ping])),
                    entry(6,guide.as_entire_binding()),entry(7,scale.as_entire_binding()),
                ]});
                {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(w.div_ceil(8),h.div_ceil(8),1);}
                ping=next;
            }
        }
        let row=512u32;let plane=u64::from(row*h);
        let readback=d.create_buffer(&wgpu::BufferDescriptor {label:Some("conservation readback"),size:plane*2,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
        for (i,image) in [&incident[ping],&ao[ping]].into_iter().enumerate() {
            encoder.copy_texture_to_buffer(image.as_image_copy(),wgpu::TexelCopyBufferInfo {buffer:&readback,layout:wgpu::TexelCopyBufferLayout {offset:plane*i as u64,bytes_per_row:Some(row),rows_per_image:Some(h)}},extent(w,h));
        }
        let submission=r.queue.submit([encoder.finish()]);let(tx,rx)=std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read,..,move|v|{let _=tx.send(v);});r.wait(Some(submission))?;rx.recv()??;
        let mapped=readback.get_mapped_range(..);let mut actual=[0f64;4];let mut negative_visibility=false;
        for y in 0..h {for x in 0..w {
            let at=(y*row+x*16) as usize;
            let get=|offset| f64::from(f32::from_ne_bytes([mapped[offset],mapped[offset+1],mapped[offset+2],mapped[offset+3]]));
            for c in 0..3 {actual[c]+=get(at+c*4);}
            let v=get(plane as usize+at);actual[3]+=v;negative_visibility|=v<0.;
        }}
        for c in 0..4 {assert!((actual[c]-expected[c]).abs()<0.00001*expected[c].abs().max(1.),"channel {c}: {actual:?} != {expected:?}");}
        assert!(negative_visibility,"signed ambient replacement was clipped");
        r.errors.check()?;Ok(())
    }

    #[test]
    fn gi_accumulation_bands_follow_both_device_limits() -> Result<()> {
        let mut limits=wgpu::Limits::default();
        for (binding, allocation) in [(128<<20, 1<<30), (1<<30,128<<20), (1<<30,1<<30)] {
            limits.max_storage_buffer_binding_size=binding;
            limits.max_buffer_size=allocation;
            let (rows,bytes)=accumulation_band(&limits,1920,1080,u32::MAX)?;
            assert_eq!(bytes,u64::from(rows)*1920*constants::GI_QUERY_BYTES);
            assert!(bytes<=binding.min(allocation));
            assert_eq!(rows,(binding.min(allocation)/(1920*constants::GI_QUERY_BYTES)).min(1080) as u32);
            // Every full-resolution GI cell is processed exactly once, including the tail.
            assert_eq!((0..1080u32.div_ceil(rows)).map(|b|(1080-b*rows).min(rows)).sum::<u32>(),1080);
        }
        limits.max_buffer_size=1920*constants::GI_QUERY_BYTES-1;
        assert!(accumulation_band(&limits,1920,1080,u32::MAX).unwrap_err().to_string().contains("GI prepared queries"));
        Ok(())
    }
}
