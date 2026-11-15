use super::*;

// Shutter samples are recorded ahead of submission. Every UPDATE must consume
// the compatible build immediately preceding it in queue order, including when
// several recorded buffers originally observed the same submitted build.
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn refits_recorded_ahead_of_submission_preserve_query_results() -> anyhow::Result<()> {
    let r = crate::test_renderer()?;
    let d = &r.device;
    for (deform, native) in [(false, false), (true, false), (false, true), (true, true)] {
        let vertices = init(
            d,
            "refit vertices",
            bytemuck::cast_slice(&[[-1f32, -1., 0.], [1., -1., 0.], [0., 1., 0.]]),
            wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::COPY_DST,
        );
        let indices = init(
            d,
            "immutable refit topology",
            bytemuck::cast_slice(&[0u32, 1, 2]),
            wgpu::BufferUsages::BLAS_INPUT,
        );
        let size = triangles(false, 3, 3);
        let bottom = d.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("deferred refit BLAS"),
                flags: BUILD_FLAGS,
                update_mode: wgpu::AccelerationStructureUpdateMode::PreferUpdate,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size.clone()],
            },
        );
        let mut top = d.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("deferred refit TLAS"),
            max_instances: 1,
            flags: BUILD_FLAGS,
            update_mode: wgpu::AccelerationStructureUpdateMode::PreferUpdate,
        });
        let transform = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.];
        top[0] = Some(wgpu::TlasInstance::new(&bottom, transform, 0, 255));
        let entry = wgpu::BlasBuildEntry {
            blas: &bottom,
            geometry: geometry(&size, &vertices, 0, 12, &indices, 0),
        };
        let mut initial = d.create_command_encoder(&Default::default());
        initial.build_acceleration_structures([&entry], [&top]);
        r.queue.submit([initial.finish()]);

        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("deferred refit query"), source: wgpu::ShaderSource::Wgsl(r#"
                enable wgpu_ray_query;
                @group(0) @binding(0) var scene: acceleration_structure;
                @group(0) @binding(1) var<storage, read_write> output: u32;
                @compute @workgroup_size(1) fn main() {
                    var query: ray_query;
                    rayQueryInitialize(&query, scene, RayDesc(0u, 255u, 0., 10., vec3(0.,0.,1.), vec3(0.,0.,-1.)));
                    rayQueryProceed(&query);
                    let hit = rayQueryGetCommittedIntersection(&query);
                    output = select(hit.instance_custom_data + 1u, 0u, hit.kind == RAY_QUERY_INTERSECTION_NONE);
                }
            "#.into()),
        });
        let group_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                layout_entries(0, wgpu::ShaderStages::COMPUTE)[0],
                pipelines::buffer_entry(
                    1,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BufferBindingType::Storage { read_only: false },
                ),
            ],
        });
        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&group_layout)],
            immediate_size: 0,
        });
        let pipeline = r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 48,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &group_layout,
            entries: &[
                crate::entry(0, top.as_binding()),
                crate::entry(1, output.as_entire_binding()),
            ],
        });
        let mut expected = Vec::new();
        for batch in 0..3 {
            let mut pending = Vec::new();
            for sample in 0..4 {
                let i = batch * 4 + sample;
                let x = if deform && i % 2 == 1 { 4. } else { 0. };
                let source = init(
                    d,
                    "sample vertices",
                    bytemuck::cast_slice(&[[-1f32 + x, -1., 0.], [1. + x, -1., 0.], [x, 1., 0.]]),
                    wgpu::BufferUsages::COPY_SRC,
                );
                // Fixed slots may disappear and become visible again between shutter
                // samples. A zero ray mask retains the active native reference.
                let mask = if i % 3 == 1 { 0 } else { 255 };
                top[0] = Some(wgpu::TlasInstance::new(&bottom, transform, i, mask));
                let mut encoder = d.create_command_encoder(&Default::default());
                if deform {
                    encoder.copy_buffer_to_buffer(&source, 0, &vertices, 0, 36);
                    encoder.build_acceleration_structures([&entry], std::iter::empty());
                    // Also cover a compatible rebuild within one recording.
                    if sample == 1 {
                        encoder.build_acceleration_structures([&entry], std::iter::empty());
                    }
                }
                if native {
                    let (_, bytes) = bottom.native_tlas_instance(i, mask).expect("native TLAS template");
                    let descriptors = init(d, "native refit descriptors", &bytes, wgpu::BufferUsages::TLAS_INPUT);
                    // Alternate forced full builds with refits, including buffers
                    // recorded before earlier builds have reached submission.
                    unsafe { encoder.build_tlas_from_buffer(&top, &descriptors, 0, 1, &[&bottom], sample == 1); }
                } else {
                    encoder.build_acceleration_structures(std::iter::empty(), [&top]);
                }
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&output, 0, &readback, u64::from(i) * 4, 4);
                pending.push(encoder.finish());
                expected.push(if x == 0. && mask != 0 { i + 1 } else { 0 });
            }
            if batch == 1 {
                for command in pending {
                    r.queue.submit([command]);
                }
            } else {
                r.queue.submit(pending);
            }
        }
        let (tx, rx) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            tx.send(result).unwrap();
        });
        r.wait(None)?;
        rx.recv()??;
        let mapped = readback.get_mapped_range(..);
        assert_eq!(
            bytemuck::cast_slice::<u8, u32>(&mapped),
            expected,
            "deform={deform}, native={native}"
        );
        drop(mapped);
        readback.unmap();
        r.errors.check()?;

        // Compatibility still matters: changing topology or instance count
        // after recording an UPDATE must reject it before native submission.
        let mut stale = d.create_command_encoder(&Default::default());
        let mut replacement = d.create_command_encoder(&Default::default());
        if deform {
            stale.build_acceleration_structures([&entry], std::iter::empty());
            let other_indices = init(
                d,
                "replacement topology",
                bytemuck::cast_slice(&[0u32, 2, 1]),
                wgpu::BufferUsages::BLAS_INPUT,
            );
            let other = wgpu::BlasBuildEntry {
                blas: &bottom,
                geometry: geometry(&size, &vertices, 0, 12, &other_indices, 0),
            };
            replacement.build_acceleration_structures([&other], std::iter::empty());
        } else {
            stale.build_acceleration_structures(std::iter::empty(), [&top]);
            top[0] = None;
            replacement.build_acceleration_structures(std::iter::empty(), [&top]);
        }
        let stale = stale.finish();
        r.queue.submit([replacement.finish()]);

        // A later failed submission must not replace the actual source's
        // compatibility metadata with its merely validated build of A.
        let mut restore = d.create_command_encoder(&Default::default());
        if deform {
            restore.build_acceleration_structures([&entry], std::iter::empty());
        } else {
            top[0] = Some(wgpu::TlasInstance::new(&bottom, transform, 0, 255));
            restore.build_acceleration_structures(std::iter::empty(), [&top]);
        }
        let mapped_source = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("force submission failure after AS validation"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        let destination = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut invalid = d.create_command_encoder(&Default::default());
        invalid.copy_buffer_to_buffer(&mapped_source, 0, &destination, 0, 4);
        let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
        r.queue.submit([restore.finish(), invalid.finish()]);
        assert!(pollster::block_on(scope.pop()).is_some());
        mapped_source.unmap();

        let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
        r.queue.submit([stale]);
        let error =
            pollster::block_on(scope.pop()).expect("incompatible update source must be rejected");
        assert!(
            error.to_string().contains("update source changed"),
            "{error}"
        );
    }
    Ok(())
}

#[test]
fn tlas_motion_budget_rebuilds_and_resets() {
    let mut motion = 0;
    assert!(!rebuild_tlas(&mut motion, 0, 100, false, true));
    assert!(!rebuild_tlas(&mut motion, 10, 100, false, true));
    assert!(!rebuild_tlas(&mut motion, 10, 100, false, true));
    assert!(rebuild_tlas(&mut motion, 5, 100, false, true));
    assert_eq!(motion, 0);
    assert!(!rebuild_tlas(&mut motion, 1, 100, false, true));
    assert!(rebuild_tlas(&mut motion, 0, 100, true, true));
    assert_eq!(motion, 0);
    assert!(rebuild_tlas(&mut motion, 100, 100, false, true));
}

#[test]
fn shutter_refits_retain_rebuild_demand_for_the_next_exposure() {
    for total_bytes in [0, 1, 100, u64::MAX] {
        let mut motion = 0;
        assert!(rebuild_tlas(&mut motion, 0, total_bytes, true, true));
        // GPU publications and complete CPU uploads both refit after the
        // exposure's first build; their deferred demand survives the boundary.
        for changed in [0, total_bytes, u64::MAX] {
            assert!(!rebuild_tlas(&mut motion, changed, total_bytes, true, false));
        }
        assert!(rebuild_tlas(&mut motion, 0, total_bytes, false, true));
        assert_eq!(motion, 0);
        assert!(!rebuild_tlas(&mut motion, 0, total_bytes, false, true));
    }
}

// Query hits cannot distinguish a valid refit from a rebuild. Inspect the
// native descriptors actually recorded through wgpu-core on the no-op backend.
struct ModeOracle {
    d: wgpu::Device,
    queue: wgpu::Queue,
    size: wgpu::BlasTriangleGeometrySizeDescriptor,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    bottom: wgpu::Blas,
}

impl ModeOracle {
    fn new(max_instances: u32) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::NOOP,
            backend_options: wgpu::BackendOptions {
                noop: wgpu::NoopBackendOptions { enable: true }, ..Default::default()
            }, ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
        let (d, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            required_limits: wgpu::Limits {
                max_blas_geometry_count: 1, max_blas_primitive_count: 3,
                max_tlas_instance_count: max_instances, ..Default::default()
            },
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        }))?;
        let vertices = init(&d, "mode oracle vertices", bytemuck::cast_slice(&[0f32; 9]), wgpu::BufferUsages::BLAS_INPUT);
        let indices = init(&d, "mode oracle indices", bytemuck::cast_slice(&[0u32, 1, 2]), wgpu::BufferUsages::BLAS_INPUT);
        let size = triangles(false, 3, 3);
        let bottom = d.create_blas(&wgpu::CreateBlasDescriptor {
            label: None, flags: BUILD_FLAGS,
            update_mode: wgpu::AccelerationStructureUpdateMode::PreferUpdate,
        }, wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] });
        Ok(Self { d, queue, size, vertices, indices, bottom })
    }
    fn entry(&self) -> wgpu::BlasBuildEntry<'_> {
        wgpu::BlasBuildEntry { blas: &self.bottom, geometry: geometry(&self.size, &self.vertices, 0, 12, &self.indices, 0) }
    }
    fn tlas(&self, max_instances: u32) -> wgpu::Tlas {
        self.d.create_tlas(&wgpu::CreateTlasDescriptor {
            label: None, max_instances, flags: BUILD_FLAGS,
            update_mode: wgpu::AccelerationStructureUpdateMode::PreferUpdate,
        })
    }
    /// Submit one recording and return the queue's cumulative native
    /// (full build, update) counts. No-op submission is synchronous; the
    /// counters observe descriptors from the real core path.
    fn submit(&self, encoder: wgpu::CommandEncoder) -> anyhow::Result<(u64, u64)> {
        self.queue.submit([encoder.finish()]);
        self.d.poll(wgpu::PollType::wait_indefinitely())?;
        let native = unsafe { self.queue.as_hal::<wgpu::hal::api::Noop>() }.expect("no-op queue");
        Ok(native.acceleration_structure_build_counts())
    }
}

#[test]
fn native_force_rebuild_records_build_then_resumes_update() -> anyhow::Result<()> {
    let oracle = ModeOracle::new(1)?;
    let top = oracle.tlas(1);
    let (_, bytes) = oracle.bottom.native_tlas_instance(0, 255).expect("no-op native descriptor");
    let descriptors = init(&oracle.d, "mode oracle descriptors", &bytes, wgpu::BufferUsages::TLAS_INPUT);
    for (step, (force_rebuild, expected)) in [(false, (2, 0)), (false, (2, 1)),
        (true, (3, 1)), (false, (3, 2))].into_iter().enumerate() {
        let mut encoder = oracle.d.create_command_encoder(&Default::default());
        if step == 0 { encoder.build_acceleration_structures([&oracle.entry()], std::iter::empty()); }
        unsafe { encoder.build_tlas_from_buffer(&top, &descriptors, 0, 1, &[&oracle.bottom], force_rebuild); }
        assert_eq!(oracle.submit(encoder)?, expected, "step={step}, force_rebuild={force_rebuild}");
    }
    Ok(())
}

// Grouped shutter samples share one recording. Each dirty sample must refit
// the build recorded before it, whether that build was submitted earlier or
// is the first one in the same, never-submitted recording.
#[test]
fn dirty_samples_in_one_recording_refit_the_preceding_build() -> anyhow::Result<()> {
    let oracle = ModeOracle::new(2)?;
    let top = oracle.tlas(2);
    let (_, bytes) = oracle.bottom.native_tlas_instance(0, 255).expect("no-op native descriptor");
    let descriptors = init(&oracle.d, "mode oracle descriptors", &bytes.repeat(2), wgpu::BufferUsages::TLAS_INPUT);
    // Each sample deforms the BLAS and publishes (instance count, force rebuild).
    let exposures: [(&str, [(u32, bool); 4], (u64, u64)); 5] = [
        // Nothing submitted yet: one BLAS and one TLAS build, then refits.
        ("cold", [(1, true), (1, false), (1, false), (1, false)], (2, 6)),
        // A submitted source and a new exposure's single requested rebuild.
        ("warm rebuild", [(1, true), (1, false), (1, false), (1, false)], (3, 13)),
        ("warm refit", [(1, false); 4], (3, 21)),
        // A rebuild requested mid-recording becomes the next sample's source.
        ("late rebuild", [(1, false), (1, false), (1, true), (1, false)], (4, 28)),
        // An UPDATE cannot change the instance count: those samples build.
        ("resized", [(1, false), (2, false), (2, false), (1, false)], (6, 34)),
    ];
    for (name, samples, expected) in exposures {
        let mut encoder = oracle.d.create_command_encoder(&Default::default());
        for (count, force_rebuild) in samples {
            encoder.build_acceleration_structures([&oracle.entry()], std::iter::empty());
            unsafe { encoder.build_tlas_from_buffer(&top, &descriptors, 0, count, &[&oracle.bottom], force_rebuild); }
        }
        assert_eq!(oracle.submit(encoder)?, expected, "{name}");
    }

    // Tracked instance lists take the same decision as native descriptors.
    let mut tracked = oracle.tlas(2);
    let transform = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.];
    for (name, counts, expected) in [("tracked cold", [1, 1, 1], (7, 36)),
        ("tracked warm", [1, 1, 1], (7, 39)), ("tracked resized", [1, 2, 2], (8, 41))] {
        let mut encoder = oracle.d.create_command_encoder(&Default::default());
        for (sample, count) in counts.into_iter().enumerate() {
            for slot in 0..2 {
                tracked[slot] = (slot < count).then(|| wgpu::TlasInstance::new(&oracle.bottom, transform, sample as u32, 255));
            }
            encoder.build_acceleration_structures(std::iter::empty(), [&tracked]);
        }
        assert_eq!(oracle.submit(encoder)?, expected, "{name}");
    }
    Ok(())
}

// Hardware facing must survive both automatic opaque traversal and the explicit
// candidate traversal used by GI. Test installed shared single/double bindings.
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn opaque_material_sides_use_hardware_traversal() -> anyhow::Result<()> {
    use super::*;
    let r = crate::test_renderer()?;
    let scene=serde_json::json!({"version":2,
        "render":{"resolution":[8,8],"settings":{"integrator":"path","samples":1,"denoiser":null}},
        "nodes":[{"type":"mesh","geometry":"triangle","materials":["single"],"translation":[-2,0,0]},
            {"type":"mesh","geometry":"triangle","materials":["double"],"translation":[2,0,0]},
            {"type":"camera","translation":[0,0,4]}],
        "buffers":{"p":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[0,1,0]]},"i":{"type":"u32","data":[0,1,2]}},
        "geometry":{"triangle":{"attributes":{"POSITION":"p"},"indices":"i"}},
        "materials":{"single":{"double_sided":false},"double":{"double_sided":true}}});
    let mut installed=r.install(crate::test_scene(scene,Path::new("."))?,Path::new("."))?;
    let mut slot=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
    installed.render_frame(&r,&mut slot,0.)?;
    let source=r#"
[[vk::binding(0,0)]] RaytracingAccelerationStructure scene;
[[vk::binding(1,0)]] RWStructuredBuffer<uint> result;
[shader("compute")][numthreads(1,1,1)] void numerical_test() {
for(uint mode=0u;mode<2u;mode++) {
for(uint i=0u;i<4u;i++) {
    bool front=(i&1u)==0u;
    RayDesc ray;ray.Origin=float3(i<2u ? -2. : 2.,0.,front ? 1. : -1.);
    ray.Direction=float3(0.,0.,front ? -1. : 1.);ray.TMin=0.;ray.TMax=2.;
    RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES> query;
    query.TraceRayInline(scene,mode==0u ? RAY_FLAG_NONE : RAY_FLAG_FORCE_NON_OPAQUE,255u,ray);
    uint candidates=0u;
    while(query.Proceed()) {candidates++;query.CommitNonOpaqueTriangleHit();}
    result[8u*mode+2u*i]=query.CommittedStatus()==COMMITTED_TRIANGLE_HIT ? 1u : 0u;
    result[8u*mode+2u*i+1u]=candidates;
}
}
}"#;
    let entries=[layout_entries(0,wgpu::ShaderStages::COMPUTE)[0],pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false})];
    let pipeline=crate::medium::test_pipeline_defines(&r,source,&Default::default(),&entries,(1,1,1),&[])?;
    let output=init(&r.device,"hardware side results",bytemuck::cast_slice(&[0u32;16]),wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC);
    let readback=r.device.create_buffer(&wgpu::BufferDescriptor {label:None,size:64,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
    let group=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,slot.rays.as_ref().unwrap().tlas.as_binding()),entry(1,output.as_entire_binding())]});
    let mut encoder=r.device.create_command_encoder(&Default::default());
    {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&group,&[]);pass.dispatch_workgroups(1,1,1);}
    encoder.copy_buffer_to_buffer(&output,0,&readback,0,64);r.queue.submit([encoder.finish()]);
    let (tx,rx)=std::sync::mpsc::channel();readback.map_async(wgpu::MapMode::Read,..,move |value| {tx.send(value).unwrap();});r.wait(None)?;rx.recv()??;
    let data=readback.get_mapped_range(..);
    assert_eq!(bytemuck::cast_slice::<u8,u32>(&data),[1,0,0,0,1,0,1,0, 1,1,0,0,1,1,1,1],"single front/back and double front/back must preserve facing in both traversal modes; forced nonopaque must yield every accepted triangle");
    Ok(())
}
