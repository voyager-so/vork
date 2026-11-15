//! The material geometry stage. Topology is installed once; GPU evaluation owns
//! both motion poses and produces the bounds consumed by visibility planning.
use crate::*;
use std::sync::Arc;

pub(crate) struct Mesh {
    pub(crate) source: wgpu::Buffer,
    attributes: wgpu::Buffer,
    topology: wgpu::Buffer,
    count: u32,
    timed: bool,
    point_vertices: u32,
    point_template: Vec<u32>,
    point_attributes: Option<wgpu::Buffer>,
}
impl Mesh {
    fn output_size(&self)->u64 {if self.point_vertices==0 {u64::from(self.count)*40} else {self.point_template.len() as u64*4}}
    fn history_size(&self)->u64 {u64::from(self.count/self.point_vertices.max(1))*80}

    pub(crate) fn new(
        r: &Renderer,
        geometry: &geometry::Geometry,
        attributes: &[String],
        timed: bool,
    ) -> Result<Arc<Self>> {
        ensure!(
            geometry.indices.len().is_multiple_of(3),
            "material displacement requires triangle geometry"
        );
        let count = u32::try_from(geometry.vertices.len())?;
        let source: Vec<_> = geometry
            .vertices
            .iter()
            .map(|v| DeformedVertex {
                POSITION: v.POSITION,
                NORMAL: v.NORMAL,
                TANGENT: v.TANGENT,
            })
            .collect();
        let point_vertices = if geometry.point_centers.is_empty() { 0 } else { (geometry.vertices.len() / geometry.point_centers.len()) as u32 };
        let mut point_template=Vec::new();
        let point_attributes=if point_vertices!=0 {
            let capacity=geometry.point_centers.len();
            point_template.extend([capacity as u32,point_vertices]);
            point_template.resize(2+capacity*10,0);
            for v in &geometry.vertices[..point_vertices as usize] {point_template.extend(v.NORMAL.into_iter().chain(v.TEXCOORD_0).map(f32::to_bits));}
            let mut rows=Vec::new();
            for (i,center) in geometry.point_centers.iter().enumerate() {
                let v=&geometry.vertices[i*point_vertices as usize];
                rows.extend(center.iter().copied());rows.extend([geometry.point_widths[i],0.,0.,0.]);
                rows.extend(v.COLOR_0);
                for name in attributes {rows.extend(geometry.attributes.get(name).map_or([0.;4],|a|a[i*point_vertices as usize]));}
            }
            Some(init(&r.device,"authored point attributes",bytemuck::cast_slice(&rows),wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::VERTEX|wgpu::BufferUsages::COPY_SRC))
        } else {None};
        let mut values = Vec::new();
        for (i, v) in geometry.vertices.iter().enumerate().filter(|(i, _)| point_vertices == 0 || i % point_vertices as usize == 0) {
            values.push([v.TEXCOORD_0[0], v.TEXCOORD_0[1], 0., 0.]);
            for name in attributes {
                values.push(
                    geometry
                        .attributes
                        .get(name)
                        .map_or([0.; 4], |values| values[i]),
                );
            }
        }
        let topology = if point_vertices != 0 {
            geometry.point_centers.iter().zip(&geometry.point_widths).flat_map(|(p,w)|p.iter().chain(std::iter::once(w))).map(|v|v.to_bits()).collect()
        } else {
            let mut incidents = vec![Vec::new(); count as usize];
            for (face, triangle) in geometry.indices.chunks_exact(3).enumerate() {
                for &vertex in triangle {
                    incidents
                        .get_mut(vertex as usize)
                        .context("displacement triangle index outside geometry")?
                        .push(face as u32);
                }
            }
            let offset_base = u32::try_from(geometry.indices.len() + 1)?;
            let mut topology = vec![offset_base];
            topology.extend_from_slice(&geometry.indices);
            let mut offset = 0u32;
            for faces in &incidents {
                topology.push(offset);
                offset += faces.len() as u32;
            }
            topology.push(offset);
            for faces in incidents {
                topology.extend(faces);
            }
            topology
        };
        let storage =
            |label, bytes: &[u8]| init(&r.device, label, bytes, wgpu::BufferUsages::STORAGE);
        Ok(Arc::new(Self {
            source: storage("undisplaced geometry", bytemuck::cast_slice(&source)),
            attributes: storage("displacement attributes", bytemuck::cast_slice(&values)),
            topology: storage("displacement topology", bytemuck::cast_slice(&topology)),
            count,
            timed,
            point_vertices, point_template, point_attributes,
        }))
    }
}

fn batch_capacity(limits: &wgpu::Limits) -> usize {
    ((u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size)
        / (3 * std::mem::size_of::<Parameters>()) as u64) as usize)
        .min(limits.max_compute_workgroups_per_dimension as usize / 3)
}

pub(crate) struct Pipeline {
    initialize: wgpu::ComputePipeline,
    evaluate: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    capacity: usize,
    attribute_poses: bool,
}
impl Pipeline {
    pub(crate) fn new(
        r: &Renderer,
        native_programs: &programs::Native,
        materials: &wgpu::BindGroupLayout,
        attributes: usize,
        attribute_poses: bool,
    ) -> Result<Self> {
        ensure!(
            r.device.limits().max_storage_buffers_per_shader_stage >= 11,
            crate::gpu_memory::device_limit("material displacement", "11 storage buffer bindings", format_args!("{} storage buffer bindings",r.device.limits().max_storage_buffers_per_shader_stage), "remove material displacement or use a GPU supporting at least eleven storage buffer bindings")
        );
        let entries: Vec<_> = (0..8)
            .map(|binding| {
                pipelines::buffer_entry(
                    binding,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BufferBindingType::Storage { read_only: binding < 5 },
                )
            })
            .collect();
        let layout = r
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("material displacement"),
                entries: &entries,
            });
        let pipeline_layout = r
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("material displacement"),
                bind_group_layouts: &[None, Some(materials), Some(&layout)],
                immediate_size: 0,
            });
        let source = include_str!("../shaders/displacement.slangh");
        let metal = r.adapter_info.backend == wgpu::Backend::Metal;
        let (_, target, flags) = if metal {
            forward_variant::METAL_TARGET
        } else {
            forward_variant::SPIRV_TARGET
        };
        let capacity = batch_capacity(&r.device.limits());
        let define = format!("-DVORK_ATTRIBUTES={attributes}");
        let includes: Vec<_> = native_programs
            .include
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        let args: Vec<_> = ["-target", target]
            .into_iter()
            .chain(["-fp-mode", "precise"])
            .chain(forward_variant::SLANG_FLAGS.iter().copied())
            .chain(flags.iter().copied())
            .chain(attribute_poses.then_some("-DVORK_ATTRIBUTE_POSES=1"))
            .chain([
                define.as_str(),
                "-DVORK_MATERIAL_RECORDS_REGISTER=t0",
                "-DVORK_MATERIAL_WORDS_REGISTER=t3",
                "-DVORK_MATERIAL_TEXTURES_REGISTER=b1",
                "-DVORK_MATERIAL_SAMPLERS_REGISTER=b2",
                "-DVORK_SHADER_ERRORS_REGISTER=u4",
            ])
            .chain(includes.iter().flat_map(|p| ["-I", p.as_str()]))
            .collect();
        let mut modules = native::SOURCES.to_vec();
        modules.extend(
            native_programs
                .modules
                .iter()
                .map(|(n, s)| (n.as_str(), s.as_str())),
        );
        modules.push(("displacement", source));
        let compile = |entry: &str| -> Result<wgpu::ComputePipeline> {
            let compiler=vork_slang::Compiler::discover();
            let (identity,prepared)=crate::programs::native_input_identity(
                &|key|r.disk_cache.path("material-displacement",1,key),&compiler,&modules,"displacement",&[(entry,"compute")],&args,&native_programs.identities,&native_programs.inputs)?;
            let mut key = blake3::Hasher::new();
            for part in [target, entry, define.as_str()] {
                key.update(part.as_bytes());
            }
            for argument in &args {
                key.update(argument.as_bytes());
            }
            let prefix = key.clone();
            let receipt_path = crate::programs::native_input_receipt_path(
                &|key| r.disk_cache.path("material-displacement", 1, key), &compiler,
                &modules, "displacement", &[(entry, "compute")], &args, &native_programs.identities, &native_programs.inputs)?;
            let code = crate::programs::native_code(&identity, prepared, &native_programs.inputs, &receipt_path, |identity| {
                let mut key = prefix.clone();
                key.update(identity);
                r.disk_cache.path("material-displacement", 1, key.finalize())
            }, || compiler.prepare_identified_modules(&modules, "displacement", &[(entry, "compute")], &args, &native_programs.identities), |prepared| {
                let start = std::time::Instant::now();
                let code = prepared.compile()?;
                crate::timing::creation(&r.device,entry,"Slang",start.elapsed().as_secs_f64()*1000.,0.);
                Ok(code)
            })?;
            let threads = if entry == "initialize" {
                (1, 1, 1)
            } else {
                (constants::DEFORM_GROUP, 1, 1)
            };
            let shader = if metal {
                native::module(
                    &r.device,
                    "material displacement",
                    None,
                    Some(&String::from_utf8(code)?),
                    threads,
                )
            } else {
                native::module(
                    &r.device,
                    "material displacement",
                    Some(&code),
                    None,
                    threads,
                )
            };
            Ok(r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("material displacement"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: r.cache.as_ref(),
            }))
        };
        Ok(Self {
            attribute_poses,
            initialize: compile("initialize")?,
            evaluate: compile("evaluate")?,
            finish: compile("finish")?,
            layout,
            capacity,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Parameters {
    count: u32,
    material: u32,
    time: f32,
    previous_time: f32,
    point_vertices: u32,
    pose: u32,
    dispatch_width: u32,
    reserved: u32,
    object_header: [u32; 4],
    to_stream: Mat4,
    from_stream: Mat4,
    /// Source words, output words, scratch words and bounds words within pages.
    addresses: [u32; 4],
}

/// Deterministic frame-private page addresses, also used by installed hit records.
/// Endpoint pages contain two interleaved vertices per current vertex.
pub(crate) fn page_layout(
    draws: &[install::Draw],
    limits: &wgpu::Limits,
) -> Result<(Vec<Option<(usize, u64)>>, Vec<(usize, u64)>)> {
    let alignment = u64::from(limits.min_storage_buffer_offset_alignment) * 5;
    let capacity =
        (u64::from(limits.max_storage_buffer_binding_size) / 2).min(limits.max_buffer_size / 2);
    let mut ranges = Vec::with_capacity(draws.len());
    let mut pages: Vec<(usize, u64)> = Vec::new();
    for (i, draw) in draws.iter().enumerate() {
        let Some(mesh) = &draw.displacement else {
            ranges.push(None);
            continue;
        };
        if let Some(leader) = draw.displacement_leader.filter(|&leader| leader < i) {
            ranges.push(ranges[leader]);
            continue;
        }
        let bytes = mesh.output_size();
        ensure!(
            bytes <= capacity,
            crate::gpu_memory::device_limit("displaced geometry", format_args!("{} endpoint bytes",bytes*2), format_args!("{} endpoint bytes",capacity*2), "reduce displacement subdivision or split geometry into smaller objects")
        );
        if pages
            .last()
            .is_none_or(|(_, size)| size.next_multiple_of(alignment) + bytes > capacity)
        {
            pages.push((i, 0));
        }
        let (leader, size) = pages.last_mut().unwrap();
        let offset = size.next_multiple_of(alignment);
        ranges.push(Some((*leader, offset)));
        *size = offset + bytes;
    }
    Ok((ranges, pages))
}

pub(crate) fn resources(
    r: &Renderer,
    s: &InstalledScene,
    inputs: &[Option<EvaluatedGeometry>],
) -> Result<Vec<Option<Frame>>> {
    let Some(pipeline) = &s.displacement else {
        return Ok(Vec::new());
    };
    let (ranges, pages) = page_layout(&s.draws, &r.device.limits())?;
    let mut streams = std::collections::HashMap::new();
    for (leader, size) in pages {
        let make = |label, size, rays| {
            gpu_memory::buffer(
                &r.device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST
                        | rt::geometry_usage(rays),
                    mapped_at_creation: false,
                },
            )
        };
        streams.insert(
            leader,
            (
                make("evaluated geometry page", size, s.rays.is_some()),
                make("evaluated endpoint page", size * 2, false),
            ),
        );
    }
    // A batch owns its control resources. Jobs retain only their logical identity
    // and output range. Equal source/topology and destination pages share bindings.
    let mut groups: std::collections::HashMap<_, Vec<usize>> = std::collections::HashMap::new();
    let mut converters=std::collections::HashMap::new();
    for (i, draw) in s.draws.iter().enumerate() {
        let Some(mesh) = &draw.displacement else { continue };
        if draw.displacement_leader.is_some_and(|leader| leader != i) { continue; }
        let (leader, _) = ranges[i].unwrap();
        let source = inputs[i].as_ref().map_or(&mesh.source, |s| &s.output.buffer);
        let previous = inputs[i].as_ref().map_or(&mesh.source, |s| &s.previous_output.buffer);
        let (source_offset, source_size) = inputs[i].as_ref().map_or((0, source.size()),
            |s| (s.output.offset, s.output.size));
        let (previous_offset,previous_size)=inputs[i].as_ref().map_or((0,previous.size()),|s|(s.previous_output.offset,s.previous_output.size));
        let mut attributes=mesh.attributes.clone();
        if let Some(input)=inputs[i].as_ref().filter(|_|draw.layout.custom!=0) {
            if let Some(current)=&input.attributes {
                ensure!(pipeline.attribute_poses,"displacement pipeline omitted historical attribute inputs");
                ensure!(mesh.point_vertices==0,"computed point displacement needs a point-domain producer; use triangle geometry or compute the offsets in the producer");
                let converter=Arc::new(crate::geometry_stage_source::AttributeRows::new(r,current,input.previous_attributes.as_ref(),&mesh.attributes,mesh.count,draw.layout)?);
                attributes=converter.buffer.clone();converters.insert(attributes.clone(),converter);
            }
        }
        groups.entry((source.clone(), source_offset, source_size, previous.clone(),previous_offset,previous_size,attributes,
            mesh.topology.clone(), leader,inputs[i].is_some())).or_default().push(i);
    }
    let mut frames: Vec<Option<Frame>> = (0..s.draws.len()).map(|_| None).collect();
    for ((source, source_offset, source_size, previous,previous_offset,previous_size,attributes, topology, leader,evaluated), jobs) in groups {
        let (current, endpoints) = &streams[&leader];
        for jobs in jobs.chunks(pipeline.capacity) {
            let mesh = s.draws[jobs[0]].displacement.as_ref().unwrap();
            let batch = Arc::new(Batch::new(r, pipeline, mesh, jobs, &source, source_offset, source_size, &previous,
                previous_offset,previous_size,&attributes, &topology, current, endpoints,evaluated,converters.get(&attributes).cloned()));
            for (slot, &i) in jobs.iter().enumerate() {
                let (_, offset) = ranges[i].unwrap();
                let size = mesh.output_size();
                if mesh.point_vertices!=0 {r.queue.write_buffer(current,offset,bytemuck::cast_slice(&mesh.point_template));}
                let gpu_bounds=(mesh.point_vertices!=0).then(||gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {
                    label:Some("point displacement bounds"),size:32,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false}));
                frames[i] = Some(Frame {
                    geometry: EvaluatedGeometry {
                        point_poses: None, point_domain: (mesh.point_vertices!=0).then_some((mesh.count/mesh.point_vertices.max(1),mesh.point_vertices)), gpu_bounds,
                        attributes: mesh.point_attributes.clone(),
                        previous_attributes: None,
                        output: geometry::VertexStream { buffer: current.clone(), offset, size },
                        previous_output: geometry::VertexStream {
                            buffer: endpoints.clone(), offset: offset * 2, size: mesh.history_size(),
                        },
                    },
                    batch: batch.clone(), slot, job: i, published:std::sync::atomic::AtomicBool::new(false),
                });
            }
        }
    }
    Ok(frames)
}

/// Retained control and scratch storage for evaluations with compatible bindings.
struct Batch {
    evaluated: bool,
    attributes: Option<Arc<crate::geometry_stage_source::AttributeRows>>,
    bind: wgpu::BindGroup,
    params: wgpu::Buffer,
    scratch: wgpu::Buffer,
    bounds_offset: u64,
    readback: wgpu::Buffer,
    jobs: Vec<usize>,
    count: u32,
    groups: [u32; 2],
    timed: bool,
    point_vertices: u32,
}
pub(crate) struct Frame {
    pub(crate) geometry: EvaluatedGeometry,
    batch: Arc<Batch>,
    slot: usize,
    job: usize,
    published: std::sync::atomic::AtomicBool,
}
impl Frame {
    pub(crate) fn append_inputs(&self, changes: &values::ChangeTracker, inputs: &mut Vec<u8>) {
        changes.append_inputs(values::ChangeKey::Displacement(self.job), inputs);
    }
    pub(crate) fn dirty(&self, changes: &values::ChangeTracker) -> bool {
        changes.dirty(values::ChangeKey::DisplacedPositions(self.job))
    }
}
impl Batch {
    fn new(r: &Renderer, pipeline: &Pipeline, mesh: &Mesh, jobs: &[usize],
        source: &wgpu::Buffer, source_offset: u64, source_size: u64,
        previous: &wgpu::Buffer, previous_offset:u64, previous_size:u64, attributes: &wgpu::Buffer,
        topology: &wgpu::Buffer, current: &wgpu::Buffer, endpoints: &wgpu::Buffer,
        evaluated:bool, attribute_input:Option<Arc<crate::geometry_stage_source::AttributeRows>>,
    ) -> Self {
        let buffer = |label, size, usage| gpu_memory::buffer(&r.device, &wgpu::BufferDescriptor {
            label: Some(label), size, usage, mapped_at_creation: false,
        });
        let params = buffer("geometry evaluation table",
            (jobs.len() * 3 * std::mem::size_of::<Parameters>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let bounds_offset = jobs.len() as u64 * u64::from(mesh.count / mesh.point_vertices.max(1)) * 36;
        let scratch = buffer("geometry scratch and bounds", bounds_offset + jobs.len() as u64 * 32,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let readback = buffer("evaluated bounds page readback", jobs.len() as u64 * 32,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);
        let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material geometry batch"), layout: &pipeline.layout,
            entries: &[
                entry(0, params.as_entire_binding()),
                // Skin pools can exceed a storage binding limit. Retain the
                // source range that the deformation owner admitted, not its pool.
                entry(1, wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: source, offset: source_offset,
                    size: std::num::NonZeroU64::new(source_size),
                })),
                entry(2, wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer:previous,offset:previous_offset,size:std::num::NonZeroU64::new(previous_size),
                })), entry(3, attributes.as_entire_binding()),
                entry(4, topology.as_entire_binding()), entry(5, scratch.as_entire_binding()),
                entry(6, current.as_entire_binding()), entry(7, endpoints.as_entire_binding()),
            ],
        });
        let count = (mesh.count/mesh.point_vertices.max(1)).div_ceil(constants::DEFORM_GROUP);
        let width = count.min(r.device.limits().max_compute_workgroups_per_dimension);
        Self { evaluated,attributes:attribute_input,bind, params, scratch, bounds_offset, readback, jobs: jobs.to_vec(), count: mesh.count,
            groups: [width, count.div_ceil(width)], timed: mesh.timed,
            point_vertices: mesh.point_vertices }
    }
}

/// A stateless prefix evaluated by the existing compute clock. Only pose zero
/// executes; presentation history belongs to the downstream frame chain.
pub(crate) struct Canonical {
    diagnostics:wgpu::ComputePipeline,
    pipeline: [wgpu::ComputePipeline;3],
    batch: Batch,
    pub geometry: EvaluatedGeometry,
    material: u32,
    object_header: u32,
}
impl Canonical {
    pub(crate) fn new(r:&Renderer,pipeline:&Pipeline,mesh:&Mesh,source:Option<&wgpu::Buffer>,material:u32,object_header:u32,object:u32)->Self {
        let size=u64::from(mesh.count)*40;
        let allocate=|bytes|gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {
            label:Some("canonical displacement stream"),size:bytes,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false,
        });
        let output=allocate(size);
        let previous=allocate(size*2);
        let unused_history=allocate(size*2);
        let source=source.unwrap_or(&mesh.source);
        let batch=Batch::new(r,pipeline,mesh,&[0],source,0,size,&unused_history,0,size*2,
            &mesh.attributes,&mesh.topology,&output,&previous,false,None);
        let code=format!("@group(0) @binding(0) var<storage,read> bounds:array<u32>; @group(0) @binding(1) var<storage,read_write> errors:array<atomic<u32>>; @compute @workgroup_size(1) fn main() {{if(bounds[{}u]!=0u) {{atomicMax(&errors[0],{}u);atomicMin(&errors[1],{object}u);}}}}",batch.bounds_offset/4+6,vork_slang::registry::GEOMETRY_NONFINITE);
        let diagnostics=compute::pipeline_layout(r,&code,"canonical geometry diagnostics",&[Some(true),Some(false)],"main");
        Self {diagnostics,pipeline:[pipeline.initialize.clone(),pipeline.evaluate.clone(),pipeline.finish.clone()],batch,
            geometry:EvaluatedGeometry {point_poses:None,point_domain:None,gpu_bounds:None,output:geometry::VertexStream {buffer:output,offset:0,size},
                previous_output:geometry::VertexStream {buffer:previous,offset:0,size:size*2},attributes:None,previous_attributes:None},
            material,object_header}
    }
    pub(crate) fn encode(&self,device:&wgpu::Device,encoder:&mut wgpu::CommandEncoder,materials:&materials::FrameMaterials,time:f32,to_stream:Mat4)->Result<()> {
        let from_stream=crate::inverse_or_identity(to_stream);
        ensure!(from_stream.is_finite(),"canonical displacement requires an invertible mesh frame");
        let mut poses=[Parameters::zeroed();3];
        poses[0]=Parameters {count:self.batch.count,material:self.material,time,previous_time:time,
            dispatch_width:self.batch.groups[0]*constants::DEFORM_GROUP,object_header:[self.object_header,1,0,0],
            to_stream,from_stream,addresses:[0,0,0,(self.batch.bounds_offset/4) as u32],..Parameters::zeroed()};
        let upload=init(device,"canonical displacement parameters",bytemuck::cast_slice(&poses),wgpu::BufferUsages::COPY_SRC);
        encoder.copy_buffer_to_buffer(&upload,0,&self.batch.params,0,upload.size());
        for (i,pipeline) in self.pipeline.iter().enumerate() {
            let mut pass=encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1,&materials.bind,&[]);
            pass.set_bind_group(2,&self.batch.bind,&[]);
            let groups=if i==0 {[1,1]} else {self.batch.groups};
            pass.dispatch_workgroups(groups[0],groups[1],1);
        }
        let bind=device.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("canonical geometry diagnostics"),layout:&self.diagnostics.get_bind_group_layout(0),entries:&[
            entry(0,self.batch.scratch.as_entire_binding()),entry(1,materials.shader_errors.as_entire_binding()),
        ]});
        let mut pass=encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.diagnostics);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(1,1,1);
        Ok(())
    }
}

/// GPU geometry precedes CPU visibility planning. Only the reduced bounds cross
/// this boundary; every triangle consumer retains the GPU position/frame stream.
pub(crate) fn evaluate(
    r: &Renderer,
    s: &InstalledScene,
    skin: &[Option<FrameDeform>],
    frames: &[Option<Frame>],
    materials: Option<&materials::FrameMaterials>,
    objects: &[Object],
    spaces: &[[Mat4; 3]],
    cached_bounds: &[Option<(Vec3, Vec3)>],
    queries: &wgpu::QuerySet,
    changes: &mut values::ChangeTracker,
    time: f32,
    previous_time: f32,
    next_time: f32,
    predecessor_inputs: Option<&[Vec<u8>]>,
    encode_predecessors: impl FnOnce(&mut wgpu::CommandEncoder),
) -> Result<Vec<Option<(Vec3, Vec3)>>> {
    let Some(pipeline) = &s.displacement else {
        return Ok(Vec::new());
    };
    let mut bounds = vec![None; s.draws.len()];
    let mut parameters = vec![[Parameters::zeroed(); 3]; frames.len()];
    let mut active = vec![false; frames.len()];
    for (i, frame) in frames.iter().enumerate() {
        let Some(frame) = frame else { continue };
        let batch = &frame.batch;
        let offset = s.roots[s.draws[i].object as usize].time_offset;
        for pose in 0..3 {
            let to_stream = s.draws[i]
                .deform
                .as_ref()
                .map_or(Mat4::IDENTITY, |d| spaces[d.job][pose]);
            let from_stream = crate::inverse_or_identity(to_stream);
            ensure!(
                from_stream.is_finite(),
                "material displacement requires an invertible mesh-local frame for draw {i}"
            );
            let params = Parameters {
                count: batch.count,
                material: objects[s.draws[i].object as usize].material_index[0],
                time: if batch.timed { time + offset } else { 0. },
                previous_time: if batch.timed {
                    if pose==2 {next_time+offset} else {previous_time + offset}
                } else {
                    0.
                },
                point_vertices: batch.point_vertices,
                pose: pose as u32,
                dispatch_width: batch.groups[0] * constants::DEFORM_GROUP,
                reserved: u32::from(batch.evaluated),
                object_header: [objects[s.draws[i].object as usize].metadata[2], 0, batch.attributes.as_ref().map_or(0,|a|a.pose_stride), 0],
                to_stream,
                from_stream,
                addresses: [
                    0, // Source addresses are relative to the bound deformation range.
                    (frame.geometry.output.offset / 4) as u32,
                    frame.slot as u32 * (batch.count / batch.point_vertices.max(1)) * 9,
                    (batch.bounds_offset / 4) as u32 + frame.slot as u32 * 8,
                ],
            };
            parameters[i][pose] = params;
        }
        // Logical jobs own dirty state, not the shared parameter table. A skin
        // update invalidates only its consumers, even when descriptors share a page.
        let changed = if let Some(inputs)=predecessor_inputs {
            let mut key=bytemuck::cast_slice(&parameters[i]).to_vec();
            key.extend_from_slice(&inputs[i]);
            if let Some(materials)=materials {changes.append_buffer_inputs(&materials.records,&mut key);}
            changes.changed(values::ChangeKey::Displacement(i),&key)
        } else {changes.changed(values::ChangeKey::Displacement(i),bytemuck::cast_slice(&parameters[i]))};
        active[i] = (if batch.point_vertices!=0 {!frame.published.load(std::sync::atomic::Ordering::Relaxed)} else {cached_bounds[i].is_none()}) || changed
            || skin.get(i).and_then(Option::as_ref).is_some_and(|skin|
                changes.buffer_dirty(&skin.params) || changes.buffer_dirty(&skin.previous_params));
        if active[i] { changes.publish(values::ChangeKey::DisplacedPositions(i)); }
    }
    let batches: Vec<_> = frames.iter().flatten().filter(|f| f.slot == 0)
        .map(|f| &*f.batch).filter(|b| b.jobs.iter().any(|&i| active[i])).collect();
    for batch in &batches {
        for (slot, &i) in batch.jobs.iter().enumerate() {
            let mut poses = parameters[i];
            for params in &mut poses {
                params.object_header[1] = u32::from(active[i]);
            }
            // Keep logical dirty subranges even though jobs share one binding.
            // The upload owner merges adjacent changed rows into one copy.
            changes.upload(&batch.params,
                (slot * 3 * std::mem::size_of::<Parameters>()) as u64,
                bytemuck::cast_slice(&poses));
        }
    }
    for (i, frame) in frames.iter().enumerate() {
        if frame.is_some() && !active[i] {
            bounds[i] = cached_bounds[i];
        }
    }
    let mut encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("evaluated geometry before visibility"),
        });
    if !changes.bytes.is_empty() {
        let upload = init(
            &r.device,
            "geometry value uploads",
            &changes.bytes,
            wgpu::BufferUsages::COPY_SRC,
        );
        for (destination, offset, source, size) in &changes.uploads {
            encoder.copy_buffer_to_buffer(&upload, *source, destination, *offset, *size);
        }
    }
    drop(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("begin geometry evaluation"),
        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
            query_set: queries,
            beginning_of_pass_write_index: Some(timing::query("deform")),
            end_of_pass_write_index: None,
        }),
    }));
    encode_predecessors(&mut encoder);
    for batch in &batches {if let Some(attributes)=&batch.attributes {attributes.encode(&mut encoder);}}
    for stage in [&pipeline.initialize, &pipeline.evaluate, &pipeline.finish] {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("material geometry stage"),
            timestamp_writes: std::ptr::eq(stage, &pipeline.finish).then_some(
                wgpu::ComputePassTimestampWrites {
                    query_set: queries,
                    beginning_of_pass_write_index: None,
                    end_of_pass_write_index: Some(timing::query("deform") + 1),
                },
            ),
        });
        pass.set_pipeline(stage);
        pass.set_bind_group(1, materials.map_or_else(|| s.material_table.bind(), |m| &m.bind), &[]);
        for batch in &batches {
            pass.set_bind_group(2, &batch.bind, &[]);
            let initialize = std::ptr::eq(stage, &pipeline.initialize);
            let groups = if initialize { [1, 1] }
                else { batch.groups };
            pass.dispatch_workgroups(groups[0], groups[1], batch.jobs.len() as u32 * if initialize { 1 } else { 3 });
        }
    }
    let mut receipts=Vec::new();
    for batch in &batches {
        if batch.point_vertices==0 {encoder.copy_buffer_to_buffer(&batch.scratch, batch.bounds_offset, &batch.readback, 0, batch.jobs.len() as u64 * 32);}
        else {for (slot,&i) in batch.jobs.iter().enumerate().filter(|(_,i)|active[**i]) {
            let frame=frames[i].as_ref().unwrap();
            encoder.copy_buffer_to_buffer(&batch.scratch,batch.bounds_offset+slot as u64*32,frame.geometry.gpu_bounds.as_ref().unwrap(),0,32);
            let receipt=gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {label:Some("point displacement validation"),size:4,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
            encoder.copy_buffer_to_buffer(&batch.scratch,batch.bounds_offset+slot as u64*32+24,&receipt,0,4);
            receipts.push((crate::graph::PublicationDiagnostic::PointDisplacement,receipt));frame.published.store(true,std::sync::atomic::Ordering::Relaxed);
        }}
    }
    let submission = crate::command_finish::Segments::submit(&r.queue, encoder);
    r.errors.check()?;
    if !receipts.is_empty() {s.transients.publication_checks.lock().unwrap().push((submission.clone(),receipts));}
    let batches:Vec<_>=batches.into_iter().filter(|b|b.point_vertices==0).collect();
    if batches.is_empty() {return Ok(bounds);}
    let (send, receive) = std::sync::mpsc::channel();
    for batch in &batches {
        let send = send.clone();
        batch.readback.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = send.send(result);
        });
    }
    if !active.iter().any(|active| *active) {
        for (i, draw) in s.draws.iter().enumerate() {
            if let Some(leader) = draw.displacement_leader { bounds[i] = bounds[leader]; }
        }
        return Ok(bounds);
    }
    r.wait(Some(submission))?;
    let decode = |key: u32| {
        f32::from_bits(if key & 0x80000000 != 0 {
            key ^ 0x80000000
        } else {
            !key
        })
    };
    let mut invalid = None;
    for batch in &batches {
        receive.recv()??;
        let bytes = batch.readback.get_mapped_range(..);
        let words: &[u32] = bytemuck::cast_slice(&bytes);
        for (slot, &i) in batch.jobs.iter().enumerate().filter(|(_, i)| active[**i]) {
            let words = &words[slot * 8..][..8];
            let valid = words[6] == 0;
            let minimum = Vec3::new(decode(words[0]), decode(words[1]), decode(words[2]));
            let maximum = Vec3::new(decode(words[3]), decode(words[4]), decode(words[5]));
            if valid && minimum.is_finite() && maximum.is_finite() {
                bounds[i] = Some((minimum, maximum));
            } else { invalid.get_or_insert(i); }
        }
        drop(bytes);
        batch.readback.unmap();
    }
    if let Some(i) = invalid {
        anyhow::bail!("material displacement produced nonfinite geometry for draw {i}");
    }
    for (i, draw) in s.draws.iter().enumerate() {
        if let Some(leader) = draw.displacement_leader { bounds[i] = bounds[leader]; }
    }
    Ok(bounds)
}

#[cfg(all(test, vork_private_tests))]
include!("displacement_tests_private.rs");
