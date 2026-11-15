//! Stable per-view instance compaction. Mesh order and instance order survive
//! conservative rejection; indirect commands and vertex fetch share one map.
use crate::*;

pub(crate) struct Resources {
    plan: Option<Selection>,
    pub(crate) instances: wgpu::Buffer,
    run: std::sync::atomic::AtomicBool,
}
struct Selection {
    commands: wgpu::Buffer,
    cpu: Vec<Receiver>,
    objects: wgpu::Buffer,
    evaluated_bounds: Vec<usize>,
    scratch: std::sync::Mutex<CpuScratch>,
    #[cfg(test)]
    visits: std::sync::atomic::AtomicUsize,
    gpu: Option<Compaction>,
}
#[derive(Default)]
struct CpuScratch {
    instances: Vec<u32>,
    commands: Vec<wgpu::util::DrawIndexedIndirectArgs>,
}
struct Receiver {
    draw: usize,
    index_count: u32,
    objects: std::ops::Range<u32>,
    bounds: Option<(Vec3, Vec3)>,
    included: Vec<bool>,
}
fn outside(vp: Mat4, object: &Object, bounds: Option<(Vec3, Vec3)>) -> bool {
    if object.metadata[1] & constants::RAY_CAMERA == 0 {
        return true;
    }
    let Some((lo, hi)) = bounds else {
        return false;
    };
    let error = vork_slang::registry::CAMERA_CULL_ERROR;
    let center = (lo + hi) * 0.5;
    let extent = (hi - lo) * 0.5 + error * lo.abs().max(hi.abs());
    let absolute = |m: Mat4| {
        Mat4::from_cols(
            m.x_axis.abs(),
            m.y_axis.abs(),
            m.z_axis.abs(),
            m.w_axis.abs(),
        )
    };
    let model = object.model;
    let world = model * center.extend(1.);
    let radius = absolute(model) * extent.extend(0.);
    let clip = vp * world;
    let spread = absolute(vp) * radius;
    let magnitude = absolute(vp) * (absolute(model) * (center.abs() + extent).extend(1.));
    let margin = spread + error * magnitude;
    if !clip.is_finite() || !margin.is_finite() {
        return false;
    }
    clip.x + clip.w < -margin.x - margin.w
        || clip.w - clip.x < -margin.x - margin.w
        || clip.y + clip.w < -margin.y - margin.w
        || clip.w - clip.y < -margin.y - margin.w
        || clip.z < -margin.z
        || clip.w - clip.z < -margin.w - margin.z
}
struct Compaction {
    pipelines: [wgpu::ComputePipeline; 3],
    bind: wgpu::BindGroup,
    blocks: [u32; 2],
    draws: [u32; 2],
    config: wgpu::Buffer,
}
pub(crate) fn entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    use crate::pipelines::buffer_entry;
    use wgpu::BufferBindingType::{Storage, Uniform};
    [
        Uniform,
        Storage { read_only: true },
        Storage { read_only: true },
        Storage { read_only: true },
        Storage { read_only: false },
        Storage { read_only: false },
        Storage { read_only: false },
        Storage { read_only: true },
        Uniform,
    ]
    .into_iter()
    .enumerate()
    .map(|(binding, ty)| buffer_entry(binding as u32, wgpu::ShaderStages::COMPUTE, ty))
    .collect()
}
impl Resources {
    pub(crate) fn unused(d:&wgpu::Device)->Self {
        Self {plan:None,instances:init(d,"unread camera instances",&[0;4],wgpu::BufferUsages::VERTEX|wgpu::BufferUsages::STORAGE),run:std::sync::atomic::AtomicBool::new(false)}
    }

    pub(crate) fn new(
        s: &InstalledScene,
        r: &Renderer,
        uniform: &wgpu::Buffer,
        objects: &wgpu::Buffer,
        words: &wgpu::Buffer,
        live: Option<&wgpu::Buffer>,
        subset: Option<&crate::render_images::Subset>,
    ) -> Self {
        Self::with_ownership(s, r, uniform, objects, words, live, subset, |i| {
            s.compute.as_ref().is_some_and(|c| c.computed_draw(i))
        })
    }
    fn with_ownership(
        s: &InstalledScene,
        r: &Renderer,
        uniform: &wgpu::Buffer,
        objects: &wgpu::Buffer,
        words: &wgpu::Buffer,
        live: Option<&wgpu::Buffer>,
        subset: Option<&crate::render_images::Subset>,
        gpu_owned: impl Fn(usize) -> bool,
    ) -> Self {
        let d = &r.device;
        let count = (objects.size() / std::mem::size_of::<GpuObject>() as u64) as u32;
        let instances = init(
            d,
            "visible camera instances",
            bytemuck::cast_slice(&vec![0u32; count as usize]),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        // Path frames draw transparent draws' controls through their own instance
        // map; decoupled visibility passes draw them with the opaque ones.
        let culled = |i: usize| !s.plan.draws[i].transparent || s.plan.decoupled;
        let enabled = s.graph.runs(crate::graph::Pass::Depth) && d
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
            && s.draws.iter().enumerate().any(|(i, draw)| {
                culled(i)
                    && draw.deform.is_none()
                    && !draw.instances.is_empty()
                    && subset.is_none_or(|subset| subset.draws[i])
            });
        if !enabled {
            return Self {
                plan: None,
                instances,
                run: std::sync::atomic::AtomicBool::new(true),
            };
        }

        // Ownership decides where selection runs, independently for each range.
        // Both producers write the same stable object-ID map and indirect count.
        let selected = |i: usize| culled(i) && subset.is_none_or(|subset| subset.draws[i]);
        let cpu: Vec<_> = s
            .draws
            .iter()
            .enumerate()
            .filter(|(i, draw)| selected(*i) && !gpu_owned(*i) && !draw.instances.is_empty())
            .map(|(i, draw)| Receiver {
                draw: i,
                index_count: draw.count,
                objects: draw.instances.clone(),
                bounds: (!draw.evaluated()
                    && draw.bounds.0.is_finite()
                    && draw.bounds.1.is_finite())
                .then_some(draw.bounds),
                included: draw
                    .instances
                    .clone()
                    .map(|object| subset.is_none_or(|subset| subset.objects[object as usize]))
                    .collect(),
            })
            .collect();
        let evaluated_bounds = cpu.iter().filter_map(|range| s.draws[range.draw].evaluated().then_some(range.draw)).collect();
        let args: Vec<_> = s
            .draws
            .iter()
            .map(|draw| wgpu::util::DrawIndexedIndirectArgs {
                index_count: draw.count,
                instance_count: 0,
                first_index: 0,
                base_vertex: 0,
                first_instance: draw.instances.start,
            })
            .collect();
        let commands = init(
            d,
            "camera selection commands",
            bytemuck::cast_slice(&args),
            wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        if !s
            .draws
            .iter()
            .enumerate()
            .any(|(i, draw)| selected(i) && gpu_owned(i) && !draw.instances.is_empty())
        {
            return Self {
                instances,
                run: std::sync::atomic::AtomicBool::new(true),
                plan: Some(Selection {
                    commands,
                    cpu,
                    objects: objects.clone(),
                    evaluated_bounds,
                    scratch: Default::default(),
                    #[cfg(test)] visits: Default::default(),
                    gpu: None,
                }),
            };
        }

        let pipelines = pipelines(r, &s.pipelines).clone();
        let group = vork_slang::registry::CAMERA_CULL_GROUP;
        let mut records: Vec<[u32; 4]> = Vec::new();
        let mut draw_records = Vec::new();
        for (i, draw) in s.draws.iter().enumerate() {
            let first =
                records.len() as u32 / vork_slang::registry::CAMERA_CULL_BLOCK_VEC4S as u32;
            if selected(i) && gpu_owned(i) {
                for start in (draw.instances.start..draw.instances.end).step_by(group as usize) {
                    let count = group.min(draw.instances.end - start);
                    let mut mask = [0u32; 8];
                    for lane in 0..count {
                        if subset.is_none_or(|s| s.objects[(start + lane) as usize]) {
                            mask[lane as usize / 32] |= 1u32 << (lane % 32);
                        }
                    }
                    records.push([start, count, i as u32, 0]);
                    records.push(mask[..4].try_into().unwrap());
                    records.push(mask[4..].try_into().unwrap());
                }
            }
            draw_records.push([
                first,
                records.len() as u32 / vork_slang::registry::CAMERA_CULL_BLOCK_VEC4S as u32
                    - first,
                draw.instances.start,
                draw.count,
            ]);
            draw_records.push([
                draw.instances.len() as u32,
                u32::from(s.compute.as_ref().is_some_and(|c| c.computed_draw(i))),
                u32::from(!gpu_owned(i) || !selected(i)),
                0,
            ]);
        }
        let block_count =
            records.len() as u32 / vork_slang::registry::CAMERA_CULL_BLOCK_VEC4S as u32;
        let draw_base = records.len() as u32;
        records.extend(draw_records);
        if records.is_empty() {
            records.push([0; 4]);
        }
        let records = init(
            d,
            "camera instance ranges",
            bytemuck::cast_slice(&records),
            wgpu::BufferUsages::STORAGE,
        );
        let allocate = |label, size, usage| {
            crate::gpu_memory::buffer(
                d,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                },
            )
        };
        let scratch = allocate(
            "camera prefix workspace",
            u64::from(count + block_count * 2) * 4,
            wgpu::BufferUsages::STORAGE,
        );
        let width = block_count
            .max(1)
            .min(d.limits().max_compute_workgroups_per_dimension);
        let draw_count = s.draws.len() as u32;
        let draw_width = draw_count
            .max(1)
            .min(d.limits().max_compute_workgroups_per_dimension);
        let mut config_words = vec![block_count, width, count, draw_base,
            draw_count, draw_width, 0, 0];
        config_words.extend([0;16]);
        let config = init(d, "camera compaction dispatch",
            bytemuck::cast_slice(&config_words),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera instance compaction"),
            layout: &pipelines[0].get_bind_group_layout(0),
            entries: &[
                entry(0, uniform.as_entire_binding()),
                entry(1, objects.as_entire_binding()),
                entry(2, words.as_entire_binding()),
                entry(3, records.as_entire_binding()),
                entry(4, scratch.as_entire_binding()),
                entry(5, commands.as_entire_binding()),
                entry(6, instances.as_entire_binding()),
                entry(7, live.unwrap_or(words).as_entire_binding()),
                entry(8, config.as_entire_binding()),
            ],
        });
        Self {
            instances,
            run: std::sync::atomic::AtomicBool::new(true),
            plan: Some(Selection {
                commands,
                cpu,
                objects: objects.clone(),
                evaluated_bounds,
                scratch: Default::default(),
                #[cfg(test)] visits: Default::default(),
                gpu: Some(Compaction {
                    pipelines,
                    bind,
                    blocks: [width, block_count.div_ceil(width).max(1)],
                    draws: [draw_width, draw_count.div_ceil(draw_width).max(1)],
                    config,
                }),
            }),
        }
    }
    #[cfg(test)]
    pub(crate) fn test_visits(&self) -> usize {
        self.plan.as_ref().map_or(0, |plan| plan.visits.load(std::sync::atomic::Ordering::Relaxed))
    }
    pub(crate) fn commands(&self) -> Option<&wgpu::Buffer> {
        self.plan.as_ref().map(|plan| &plan.commands)
    }
    /// Main exposures share a conservative selection across spatial AA.
    /// Captures retain their existing per-sample camera and staging path.
    pub(crate) fn prepare_exposure(
        &self, vp: Mat4, objects: &[Object],
        geometry_bounds: &[Option<(Vec3,Vec3)>],
        changes: &mut crate::values::ChangeTracker, reuse: bool,
    ) {
        self.run.store(!reuse, std::sync::atomic::Ordering::Relaxed);
        if reuse { return; }
        if let Some(gpu) = self.plan.as_ref().and_then(|plan| plan.gpu.as_ref()) {
            changes.upload(&gpu.config, 28, bytemuck::bytes_of(&1u32));
            changes.upload(&gpu.config, 32, bytemuck::bytes_of(&vp));
        }
        let Some(plan) = &self.plan else { return; };
        if plan.cpu.is_empty() { return; }
        // Static local bounds and membership belong to the installed ranges.
        // Only evaluated bounds can change. Never scan the object population to
        // discover that its published buffer and camera stayed unchanged.
        let mut inputs = Vec::with_capacity(72 + 28 * plan.evaluated_bounds.len());
        inputs.extend_from_slice(bytemuck::bytes_of(&vp));
        inputs.extend_from_slice(&changes.buffer_revision(&plan.objects).to_le_bytes());
        for &draw in &plan.evaluated_bounds {
            if let Some((lo, hi)) = geometry_bounds.get(draw).copied().flatten() {
                inputs.extend_from_slice(bytemuck::cast_slice(&[1.,lo.x,lo.y,lo.z,hi.x,hi.y,hi.z]));
            } else {
                inputs.extend_from_slice(&[0;28]);
            }
        }
        let changed = changes.changed(crate::values::ChangeKey::CameraSelection(plan.commands.clone()), &inputs);
        // Two publications can share a sample stamp. Reprepare on that sample
        // even if an earlier call already consumed its first publication.
        if !changed && !changes.buffer_dirty(&plan.objects) { return; }
        self.prepare_into(vp, objects, geometry_bounds, changes,
            [(&plan.commands, 0), (&self.instances, 0)]);
    }
    fn prepare_into(
        &self,
        vp: Mat4,
        objects: &[Object],
        geometry_bounds: &[Option<(Vec3,Vec3)>],
        changes: &mut crate::values::ChangeTracker,
        targets: [(&wgpu::Buffer, u64); 2],
    ) {
        let Some(plan) = &self.plan else {
            return;
        };
        // Adjacent CPU ranges share uploads. GPU-owned ranges split the spans,
        // so dirty CPU values can never overwrite GPU-produced counts or IDs.
        let mut scratch = plan.scratch.lock().unwrap();
        let CpuScratch { instances, commands } = &mut *scratch;
        for ranges in plan.cpu.chunk_by(|a, b| a.draw + 1 == b.draw) {
            let first = ranges[0].objects.start;
            let end = ranges.last().unwrap().objects.end;
            instances.resize((end - first) as usize, 0);
            commands.clear();
            #[cfg(test)]
            plan.visits.fetch_add(ranges.iter().map(|range| range.objects.len()).sum::<usize>(), std::sync::atomic::Ordering::Relaxed);
            for range in ranges {
                let bounds = geometry_bounds.get(range.draw).copied().flatten().or(range.bounds);
                let mut count = 0;
                for (object, included) in range.objects.clone().zip(&range.included) {
                    if *included && !outside(vp, &objects[object as usize], bounds) {
                        instances[(range.objects.start - first + count) as usize] = object + 1;
                        count += 1;
                    }
                }
                commands.push(wgpu::util::DrawIndexedIndirectArgs {
                    index_count: range.index_count,
                    instance_count: count,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: range.objects.start,
                });
            }
            changes.upload(
                targets[0].0,
                targets[0].1
                    + ranges[0].draw as u64
                        * std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>() as u64,
                bytemuck::cast_slice(commands),
            );
            changes.upload(
                targets[1].0,
                targets[1].1 + u64::from(first) * 4,
                bytemuck::cast_slice(instances),
            );
            // Keep scratch zeroed between uses. Only selected prefixes were
            // written; clearing the whole population penalizes camera misses
            // even when almost every object remains outside the view.
            for (range, command) in ranges.iter().zip(commands.iter()) {
                let start = (range.objects.start - first) as usize;
                instances[start..start + command.instance_count as usize].fill(0);
            }
        }
    }

    // Captures prepare all sample values before encoding. Retain each CPU
    // selection alongside its camera uniform, then restore just CPU-owned spans
    // before each sample. GPU producers continue to own their output bytes.
    pub(crate) fn sample_values(
        &self,
        device: &wgpu::Device,
        samples: u32,
    ) -> Option<wgpu::Buffer> {
        let plan = self.plan.as_ref().filter(|plan| !plan.cpu.is_empty())?;
        Some(crate::gpu_memory::buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("capture camera selections"),
                size: (plan.commands.size() + self.instances.size()) * u64::from(samples),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        ))
    }
    pub(crate) fn prepare_sample(
        &self,
        vp: Mat4,
        objects: &[Object],
        geometry_bounds: &[Option<(Vec3,Vec3)>],
        changes: &mut crate::values::ChangeTracker,
        values: &wgpu::Buffer,
        sample: u32,
    ) {
        let plan = self.plan.as_ref().unwrap();
        let base = u64::from(sample) * (plan.commands.size() + self.instances.size());
        self.prepare_into(
            vp,
            objects,
            geometry_bounds,
            changes,
            [(values, base), (values, base + plan.commands.size())],
        );
    }
    pub(crate) fn copy_sample(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        changes: &mut crate::values::ChangeTracker,
        values: &wgpu::Buffer,
        sample: u32,
    ) {
        let plan = self.plan.as_ref().unwrap();
        changes.replace_gpu(&[&plan.commands, &self.instances],
            crate::values::ChangeKey::CameraSelection(plan.commands.clone()));
        let base = u64::from(sample) * (plan.commands.size() + self.instances.size());
        let command_size = std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>() as u64;
        for ranges in plan.cpu.chunk_by(|a, b| a.draw + 1 == b.draw) {
            let offset = ranges[0].draw as u64 * command_size;
            encoder.copy_buffer_to_buffer(
                values,
                base + offset,
                &plan.commands,
                offset,
                ranges.len() as u64 * command_size,
            );
            let first = u64::from(ranges[0].objects.start) * 4;
            let end = u64::from(ranges.last().unwrap().objects.end) * 4;
            encoder.copy_buffer_to_buffer(
                values,
                base + plan.commands.size() + first,
                &self.instances,
                first,
                end - first,
            );
        }
    }

    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queries: Option<&wgpu::QuerySet>,
    ) -> bool {
        if !self.run.load(std::sync::atomic::Ordering::Relaxed) { return false; }
        let Some(plan) = self.plan.as_ref().and_then(|plan| plan.gpu.as_ref()) else {
            return false;
        };
        for (i, pipeline) in plan.pipelines.iter().enumerate() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("camera instance compaction"),
                timestamp_writes: queries.filter(|_| i == 0).map(|queries| wgpu::ComputePassTimestampWrites {
                    query_set: queries,
                    beginning_of_pass_write_index: Some(timing::query("depth")),
                    end_of_pass_write_index: None,
                }),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &plan.bind, &[]);
            let grid = if i == 1 { plan.draws } else { plan.blocks };
            pass.dispatch_workgroups(grid[0], grid[1], 1);
        }
        true
    }
}

#[cfg(test)]
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn single_instance_draws_follow_each_camera_view() -> Result<()> {
    let r = crate::test_renderer()?;
    // Distinct geometry keeps these as single-instance draws. Neither draw
    // may depend on an unrelated instanced group to enable camera rejection.
    let scene: Scene = serde_json::from_value(serde_json::json!({"version":2,
        "nodes":[{"type":"camera","name":"camera"},
            {"type":"mesh","geometry":"a","pass_index":7,"translation":[20,0,-5]},
            {"type":"mesh","geometry":"b","pass_index":9,"translation":[0,0,-5]}],
        "buffers":{"a":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[0,1,0]]},
            "b":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[0,0.9,0]]}},
        "geometry":{"a":{"attributes":{"POSITION":"a"}},"b":{"attributes":{"POSITION":"b"}}},
        "animation":[{"tracks":[{"path":["nodes","camera","translation"],"keys":[[0,[0,0,0]],[1,[20,0,0]]]}]}],
        "render":{"resolution":[16,16],"settings":{"integrator":"raster","samples":1}}}))?;
    let mut s = r.install(scene, std::path::Path::new("."))?;
    assert_eq!(s.draws.len(), 2);
    assert!(s.draws.iter().all(|draw| draw.instances.len() == 1));
    let mut frame = s.create_frame_slot(&r, Default::default())?;
    assert!(frame.view.camera_cull.plan.as_ref().unwrap().gpu.is_none());
    for (time, visible, hidden) in [(0., 9, 7), (1., 7, 9), (0., 9, 7)] {
        let pixels = s.render_frame(&r, &mut frame, time)?;
        assert!(pixels.ids.contains(&visible));
        assert!(!pixels.ids.contains(&hidden));
    }
    Ok(())
}

#[cfg(test)]
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn compaction_preserves_order_across_blocks_views_and_subsets() -> Result<()> {
    let r = crate::test_renderer()?;
    let count = vork_slang::registry::CAMERA_CULL_GROUP.pow(2) + 1;
    let matrices: Vec<_> = (0..count)
        .map(|i| {
            Mat4::from_translation(Vec3::new(if i % 10 == 0 { 0. } else { 20. }, 0., -5.))
                .to_cols_array()
        })
        .collect();
    let scene: Scene = serde_json::from_value(serde_json::json!({"version":2,
        "nodes":[{"type":"camera"},{"type":"mesh","geometry":"g","instances":{"attributes":{"TRANSLATION":"translations"}}},{"type":"mesh","geometry":"g","materials":["a"],"translation":[0,0,-5]},{"type":"mesh","geometry":"g","materials":["b"],"translation":[0,0,-5]},{"type":"mesh","geometry":"g","materials":["b"],"translation":[20,0,-5]}],
        "buffers":{"p":{"type":"vec3","data":[[-0.25,-0.25,0],[0.25,-0.25,0],[0,0.25,0]]},"translations":{"type":"vec3","data":matrices.iter().map(|m|[m[12],m[13],m[14]]).collect::<Vec<_>>()}},
        "materials":{"a":{"inputs":{"base_color":[1,0,0]}},"b":{"inputs":{"base_color":[0,1,0]}}},
        "geometry":{"g":{"attributes":{"POSITION":"p"}}},
        "render":{"resolution":[16,16],"settings":{"integrator":"raster","samples":1}}}))?;
    let mut s = r.install(scene, std::path::Path::new("."))?;
    assert_eq!(s.draws.len(), 3);
    assert_eq!(s.draws[1].instances.len(), 1);
    assert_eq!(s.draws[2].instances.len(), 2);
    let mut frame = s.create_frame_slot(&r, Default::default())?;
    let pixels = s.render_frame(&r, &mut frame, 0.)?;
    assert!(
        pixels.ids.iter().any(|&id| id != 0),
        "compacted instances did not rasterize"
    );
    let objects = init(
        &r.device,
        "compaction test objects",
        bytemuck::cast_slice(&gpu_objects(&s.objects)),
        wgpu::BufferUsages::STORAGE,
    );
    let mut uniform = Uniform::zeroed();
    let projection = Mat4::perspective_rh(1., 1., 0.1, 100.);
    uniform.vp = projection;
    let uniform_buffer = init(
        &r.device,
        "compaction test camera",
        bytemuck::bytes_of(&uniform),
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let subset = crate::render_images::Subset {
        draws: vec![true, true, true],
        objects: (0..s.objects.len()).map(|i| i % 3 == 0).collect(),
        lights: vec![],
    };
    let run = |vp: Mat4,
               selection: Option<&crate::render_images::Subset>,
               backend: u32,
               bounded: bool|
     -> Result<Vec<u32>> {
        let mut value = uniform;
        value.vp = vp;
        r.queue
            .write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&value));
        let cull = Resources::with_ownership(
            &s,
            &r,
            &uniform_buffer,
            &objects,
            s.material_table.records(),
            None,
            selection,
            |i| backend == 1 || ((backend == 2 || backend == 4) && i == 2),
        );
        let mut changes = crate::values::ChangeTracker::default();
        let samples = (backend >= 3).then(|| cull.sample_values(&r.device, 2).unwrap());
        if let Some(samples) = &samples {
            cull.prepare_sample(vp, &s.objects, &[], &mut changes, samples, 0);
            cull.prepare_sample(
                vp * Mat4::from_translation(Vec3::new(-20., 0., 0.)),
                &s.objects,
                &[],
                &mut changes,
                samples,
                1,
            );
        } else {
            cull.prepare_into(vp, &s.objects, &[], &mut changes,
                [(cull.commands().unwrap(), 0), (&cull.instances, 0)]);
        }
        for (buffer, offset, source, size) in &changes.uploads {
            r.queue.write_buffer(
                buffer,
                *offset,
                &changes.bytes[*source as usize..(*source + *size) as usize],
            );
        }
        let queries = r.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: timing::count(),
        });
        let read = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: cull.commands().unwrap().size() + cull.instances.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = r.device.create_command_encoder(&Default::default());
        if let Some(samples) = &samples {
            // Both selections were uploaded before encoding: restoring sample 0
            // must not use the last sample's camera, including in a mixed view.
            cull.copy_sample(&mut encoder, &mut changes, samples, 0);
        }
        cull.encode(&mut encoder, Some(&queries));
        encoder.copy_buffer_to_buffer(
            cull.commands().unwrap(),
            0,
            &read,
            0,
            cull.commands().unwrap().size(),
        );
        encoder.copy_buffer_to_buffer(
            &cull.instances,
            0,
            &read,
            cull.commands().unwrap().size(),
            cull.instances.size(),
        );
        let submission = r.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        read.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = tx.send(result);
        });
        r.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })?;
        rx.recv()??;
        let mapped = read.get_mapped_range(..);
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        assert_eq!(words[0], s.draws[0].count);
        let first = words[4] as usize;
        let command_words = s.draws.len() * 5;
        // Check the singleton and two-instance ranges too: in mixed mode the
        // GPU must leave CPU commands/maps intact and produce its own range.
        for (i, draw) in s.draws.iter().enumerate().skip(1) {
            let expected: Vec<_> = draw
                .instances
                .clone()
                .filter(|&object| {
                    selection.is_none_or(|subset| subset.objects[object as usize])
                        && !outside(
                            vp,
                            &s.objects[object as usize],
                            bounded.then_some(draw.bounds),
                        )
                })
                .map(|object| object + 1)
                .collect();
            assert_eq!(words[i * 5 + 1] as usize, expected.len());
            let start = command_words + draw.instances.start as usize;
            assert_eq!(&words[start..start + expected.len()], expected);
        }
        let result = words[command_words + first..command_words + first + words[1] as usize]
            .iter()
            .map(|v| v - 1)
            .collect();
        drop(mapped);
        read.unmap();
        r.errors.check()?;
        Ok(result)
    };
    let first = s.draws[0].instances.start;
    let expected = |select: fn(u32) -> bool| {
        (0..count)
            .filter(|&i| select(i))
            .map(|i| first + i)
            .collect::<Vec<_>>()
    };
    for gpu in [0, 1, 2, 3, 4] {
        assert_eq!(run(projection, None, gpu, true)?, expected(|i| i % 10 == 0));
        assert_eq!(
            run(
                projection * Mat4::from_translation(Vec3::new(-20., 0., 0.)),
                None,
                gpu,
                true
            )?,
            expected(|i| i % 10 != 0)
        );
        assert_eq!(
            run(projection, Some(&subset), gpu, true)?,
            expected(|i| i % 10 == 0 && i % 3 == 0)
        );
    }
    // Invalidated bounds retain all instances, even when a prior view rejected them.
    let offset = (s.objects[first as usize].metadata[2] as u64 + 12) * 4;
    r.queue.write_buffer(
        s.material_table.records(),
        offset,
        bytemuck::bytes_of(&0u32),
    );
    assert_eq!(run(projection, None, 1, false)?, expected(|_| true));
    Ok(())
}

#[cfg(test)]
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn remapped_first_visible_instance_rasterizes() -> Result<()> {
    let r = crate::test_renderer()?;
    let scene: Scene = serde_json::from_value(
        serde_json::json!({"version":2,
        "nodes":[{"type":"camera"},
        {"type":"mesh","geometry":"g","pass_index":7,"translation":[20,0,-5]},
        {"type":"mesh","geometry":"g","pass_index":9,"translation":[0,0,-5]}],
        "buffers":{"p":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[0,1,0]]}},
        "geometry":{"g":{"attributes":{"POSITION":"p"}}},
        "render":{"resolution":[16,16],"settings":{"integrator":"raster","samples":1}}}),
    )?;
    let mut s = r.install(scene, std::path::Path::new("."))?;
    let mut frame = s.create_frame_slot(&r, Default::default())?;
    let pixels = s.render_frame(&r, &mut frame, 0.)?;
    let cull = &frame.view.camera_cull;
    let read = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: cull.commands().unwrap().size() + cull.instances.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = r.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(
        cull.commands().unwrap(),
        0,
        &read,
        0,
        cull.commands().unwrap().size(),
    );
    encoder.copy_buffer_to_buffer(
        &cull.instances,
        0,
        &read,
        cull.commands().unwrap().size(),
        cull.instances.size(),
    );
    let submission = r.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    read.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = tx.send(result);
    });
    r.device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    })?;
    rx.recv()??;
    let mapped = read.get_mapped_range(..);
    let words: &[u32] = bytemuck::cast_slice(&mapped);
    assert_eq!(words[1], 1);
    assert_eq!(words[5], 2);
    assert!(
        pixels.ids.iter().any(|&id| id == 9),
        "the surviving original instance must rasterize"
    );
    drop(mapped);
    read.unmap();
    Ok(())
}

#[cfg(test)]
#[test]
fn cpu_selection_keeps_touching_and_unbounded_receivers() {
    let bounds = Some((Vec3::splat(-0.25), Vec3::splat(0.25)));
    for scale in [Vec3::ONE, Vec3::new(-1., 2., 0.5)] {
        let mut object =
            Object::new(Mat4::from_translation(Vec3::new(1.25, 0., 0.5)) * Mat4::from_scale(scale));
        object.metadata[1] = constants::RAY_CAMERA;
        assert!(
            !outside(Mat4::IDENTITY, &object, bounds),
            "touching the clip plane stays eligible"
        );
        object.model = Mat4::from_translation(Vec3::new(1.251, 0., 0.5)) * Mat4::from_scale(scale);
        assert!(outside(Mat4::IDENTITY, &object, bounds));
        assert!(
            !outside(Mat4::IDENTITY, &object, None),
            "unknown bounds stay eligible"
        );
        assert!(!outside(
            Mat4::from_cols_array(&[f32::NAN; 16]),
            &object,
            bounds
        ));
        object.metadata[1] = 0;
        assert!(
            outside(Mat4::IDENTITY, &object, None),
            "camera visibility remains independent of bounds"
        );
    }
}

pub(crate) fn pipelines<'a>(r: &Renderer, p: &'a crate::pipelines::CachedPipelines) -> &'a [wgpu::ComputePipeline; 3] {
    let d = &r.device;
    p
        .camera_cull_pipeline
        .get_or_init(|| {
            let group = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera instance compaction"),
                entries: &entries(),
            });
            let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("camera instance compaction"),
                bind_group_layouts: &[Some(&group)],
                immediate_size: 0,
            });
            [
                &native::CAMERA_CULL_CLASSIFY,
                &native::CAMERA_CULL_OFFSETS,
                &native::CAMERA_CULL_COMPACT,
            ]
            .map(|code| {
                let module = native::shader_module(d, code);
                r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("camera instance compaction"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(code.entry),
                    compilation_options: Default::default(),
                    cache: r.cache.as_ref(),
                })
            })
        })

}
