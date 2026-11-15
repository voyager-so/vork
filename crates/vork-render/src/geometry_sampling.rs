//! Canonical, read-only geometry inputs. The compute cursor chooses every time;
//! this owner only samples a pose and records GPU work into that cursor's encoder.
use crate::buffers::ProgramBuffers;
use crate::*;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

struct Job {
    root: install::MeshRoot,
    native: Option<Native>,
    normals: Option<crate::geometry_stage_source::Normals>,
    displacement: Option<displacement::Canonical>,
    displacement_first: bool,
    source: Option<crate::geometry_stage_source::Source>,
    count: u32,
    reference_copy: Option<(wgpu::Buffer, u64, u64)>,
    scatter: wgpu::ComputePipeline,
    scatter_bind: wgpu::BindGroup,
}

struct Native {
    deform: Deform,
    params: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

pub(crate) struct Inputs {
    pub reference: Option<crate::geometry_reference::Table>,
    materials: Option<materials::FrameMaterials>,
    publish: Option<wgpu::ComputePipeline>,
    device: wgpu::Device,
    pipeline: wgpu::ComputePipeline,
    nodes: Vec<EvalNode>,
    assets: Vec<Asset>,
    animation: std::sync::Arc<animation::Plan>,
    targets: crate::targets::Plan,
    pages: BTreeMap<(u32, u32), wgpu::Buffer>,
    jobs: Vec<Job>,
}

impl Inputs {
    pub(crate) fn page(&self, object: u32, page: u32) -> Option<&wgpu::Buffer> {
        self.pages.get(&(object, page))
    }

    pub(crate) fn new(
        r: &Renderer,
        scene: &Scene,
        decoded: &buffers::SceneBuffers,
        named: &ProgramBuffers,
        required: &BTreeSet<String>,
        references: &BTreeSet<String>,
        nodes: &[EvalNode],
        assets: &[Asset],
        animation: &std::sync::Arc<animation::Plan>,
        targets: &crate::targets::Plan,
        roots: &[install::MeshRoot],
        draws: &[install::Draw],
        order: &[crate::geometry_chain::Kind],
        displacement: Option<&displacement::Pipeline>,
        material_table: &materials::MaterialTable,
        objects: &[Object],
        attribute_names: &[String],
    ) -> Result<Option<Self>> {
        use crate::geometry_chain::Kind;
        let deform_before_compute = Kind::before(order, Kind::Deform, Kind::Compute);
        let displace_before_compute = Kind::before(order, Kind::Displace, Kind::Compute);
        let reference =
            crate::geometry_reference::Table::new(r, scene, decoded, nodes, roots, draws, references)?;
        let referenced: BTreeSet<_> = reference
            .iter()
            .flat_map(|table| table.references.iter().map(|reference| reference.object))
            .collect();
        let selected: Vec<_> = draws
            .iter()
            .filter(|draw| draw.computed || referenced.contains(&draw.object))
            .filter(|draw| {
                matches!(&nodes[roots[draw.object as usize].node].node.kind,
                NodeKind::Mesh {geometry, ..} if referenced.contains(&draw.object) || required.contains(geometry)
                    && (compute::geometry_has_deformation(scene,&scene.geometry[geometry]) || draw.displacement.is_some()))
            })
            .collect();
        if selected.is_empty() && reference.is_none() {
            return Ok(None);
        }
        let pipeline = passes::deform::pipeline(r, &std::sync::OnceLock::new());
        let mut result = Self {
            reference,
            materials: if selected.iter().any(|draw| {
                draw.displacement.is_some()
                    && (displace_before_compute || referenced.contains(&draw.object))
            }) {
                Some(material_table.canonical(r)?)
            } else {
                None
            },
            publish: None,
            device: r.device.clone(),
            pipeline,
            nodes: nodes.to_vec(),
            assets: assets.to_vec(),
            animation: animation.clone(),
            targets: targets.clone(),
            pages: BTreeMap::new(),
            jobs: Vec::new(),
        };
        if result.materials.is_some() {
            result.publish = Some(compute::pipeline_layout(
                r,
                "@group(0) @binding(0) var<storage,read_write> source:array<atomic<u32>>; @group(0) @binding(1) var<storage,read_write> destination:array<atomic<u32>>; @compute @workgroup_size(1) fn main() {atomicMax(&destination[0],atomicExchange(&source[0],0u));atomicMin(&destination[1],atomicExchange(&source[1],0xffffffffu));}",
                "publish canonical geometry diagnostics",
                &[Some(false), Some(false)],
                "main",
            ));
        }
        let mut encoder = r.device.create_command_encoder(&Default::default());
        let mut installed = BTreeSet::new();
        for draw in selected {
            if !installed.insert(draw.object) {
                continue;
            }
            let root = &roots[draw.object as usize];
            let NodeKind::Mesh { geometry, skin, .. } = &nodes[root.node].node.kind else {
                unreachable!()
            };
            let shape = &scene.geometry[geometry];
            let packed = (draw.computed && required.contains(geometry))
                .then(|| compute::predecessor_plan(named, shape))
                .transpose()?;
            for (page, size) in packed
                .iter()
                .flat_map(|plan| plan.sizes.iter())
                .copied()
                .enumerate()
            {
                let target = crate::gpu_memory::buffer(
                    &r.device,
                    &wgpu::BufferDescriptor {
                        label: Some("read-only evaluated predecessor columns"),
                        size,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                );
                result.pages.insert((draw.object, page as u32), target);
            }
            for &(source_page, source_offset, page, offset, bytes) in
                packed.iter().flat_map(|plan| plan.copies.iter())
            {
                encoder.copy_buffer_to_buffer(
                    &named.storage[source_page as usize],
                    source_offset,
                    &result.pages[&(draw.object, page)],
                    offset,
                    bytes,
                );
            }
            let deform_before_compute = deform_before_compute || referenced.contains(&draw.object);
            let displace_before_compute =
                displace_before_compute || referenced.contains(&draw.object);
            let native = deform_before_compute && draw.deform.is_some();
            let displaced = displace_before_compute && draw.displacement.is_some();
            if !native && !displaced {
                continue;
            }
            ensure!(
                shape.subdivision.is_none(),
                "sampled compute predecessor requires a control cage; place subdivision after compute"
            );
            let get = |semantic: &str| {
                shape
                    .attributes
                    .get(semantic)
                    .map(|name| &decoded.data[name])
            };
            let positions = get("POSITION")
                .context("sampled predecessor POSITION")?
                .floats::<3>()?;
            let count = positions.len();
            let normals = get("NORMAL")
                .map(|data| data.floats::<3>())
                .transpose()?
                .unwrap_or_else(|| vec![[0.; 3]; count]);
            // Preserve all source rows, including rows omitted by render-side
            // flat-normal corner expansion. Input identity is authored identity.
            let mut canonical = geometry::Geometry::channels(
                positions,
                normals,
                vec![[[0.; 2]; 2]; count],
                Vec::new(),
                0,
            );
            canonical.indices = shape.indices.as_ref().map_or_else(
                || (0..count as u32).collect(),
                |name| decoded.data[name].words.clone(),
            );
            for (set, name) in ["TEXCOORD_0", "TEXCOORD_1"].iter().enumerate() {
                if let Some(data) = get(name) {
                    for (vertex, uv) in canonical.vertices.iter_mut().zip(data.floats::<2>()?) {
                        if set == 0 {
                            vertex.TEXCOORD_0 = uv;
                        } else {
                            vertex.TEXCOORD_1 = uv;
                        }
                    }
                }
            }
            buffers::apply_point_attributes(
                decoded,
                shape,
                attribute_names,
                count,
                1,
                &mut canonical,
            )?;
            if let Some(tangents) = get("TANGENT") {
                for (vertex, value) in canonical.vertices.iter_mut().zip(tangents.floats::<4>()?) {
                    vertex.TANGENT = value;
                }
            }
            // The complete authored row domain is retained, even for unused rows.
            let deform = if native {
                let binding = draw
                    .deform
                    .as_ref()
                    .unwrap()
                    .native
                    .clone()
                    .context("native compute predecessor binding")?;
                buffers::deformation(r, decoded, shape, skin.as_ref(), &canonical, binding)?
            } else {
                None
            };
            let geometry_bytes = count as u64 * DeformedVertex::BYTES;
            let limit = u64::from(r.device.limits().max_storage_buffer_binding_size)
                .min(r.device.limits().max_buffer_size);
            ensure!(
                geometry_bytes * 2 <= limit,
                crate::gpu_memory::device_limit("canonical predecessor", format_args!("{} bytes",geometry_bytes*2), format_args!("{limit} bytes"), "split geometry into smaller objects")
            );
            let buffer = |label, size, usage| {
                crate::gpu_memory::buffer(
                    &r.device,
                    &wgpu::BufferDescriptor {
                        label: Some(label),
                        size,
                        usage,
                        mapped_at_creation: false,
                    },
                )
            };
            let output = buffer(
                "canonical native geometry",
                geometry_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let displacement_first = Kind::before(order, Kind::Displace, Kind::Deform);
            let mut normals = None;
            let displaced = if displaced {
                let materials: BTreeSet<_> = draws
                    .iter()
                    .filter(|other| other.object == draw.object)
                    .map(|other| other.material)
                    .collect();
                ensure!(
                    materials.len() == 1,
                    "canonical displacement on node {:?} requires one material; split the node into separately named geometry objects",
                    nodes[root.node].node.name
                );
                let mesh = displacement::Mesh::new(r, &canonical, attribute_names, true)?;
                let input = if native && !displacement_first {
                    &output
                } else {
                    &mesh.source
                };
                if !shape.attributes.contains_key("NORMAL") {
                    normals = Some(crate::geometry_stage_source::Normals::new(
                        r,
                        input,
                        &canonical.indices,
                        count as u32,
                    )?);
                }
                Some(displacement::Canonical::new(
                    r,
                    displacement.context("canonical displacement pipeline")?,
                    &mesh,
                    Some(normals.as_ref().map_or(input, |normals| &normals.output)),
                    objects[draw.object as usize].material_index[0],
                    objects[draw.object as usize].metadata[2],
                    draw.object,
                ))
            } else {
                None
            };
            let source = if displacement_first {
                deform
                    .as_ref()
                    .zip(displaced.as_ref())
                    .map(|(native, displaced)| {
                        crate::geometry_stage_source::Source::new(
                            r,
                            &native.source,
                            &displaced.geometry,
                            count as u32,
                        )
                    })
                    .transpose()?
            } else {
                None
            };
            let native = if let Some(deform) = deform {
                let param_bytes =
                    ((4 + deform.joints * 16 + deform.targets).div_ceil(4) * 16) as u64;
                ensure!(
                    param_bytes <= limit,
                    crate::gpu_memory::device_limit("canonical skin palette", format_args!("{param_bytes} bytes"), format_args!("{limit} bytes"), "split the skin into smaller objects")
                );
                let params = buffer(
                    "canonical predecessor palette",
                    param_bytes,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                );
                let unused = buffer(
                    "unused predecessor history",
                    16,
                    wgpu::BufferUsages::STORAGE,
                );
                let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("canonical predecessor deformation"),
                    layout: &result.pipeline.get_bind_group_layout(0),
                    entries: &[
                        entry(
                            0,
                            source
                                .as_ref()
                                .map_or(&deform.source, |source| &source.buffer)
                                .as_entire_binding(),
                        ),
                        entry(1, deform.influences.as_entire_binding()),
                        entry(2, deform.morphs.as_entire_binding()),
                        entry(3, params.as_entire_binding()),
                        entry(4, output.as_entire_binding()),
                        entry(5, params.as_entire_binding()),
                        entry(6, unused.as_entire_binding()),
                    ],
                });
                Some(Native {
                    deform,
                    params,
                    bind,
                })
            } else {
                None
            };
            let final_output = if displacement_first && native.is_some() || displaced.is_none() {
                &output
            } else {
                &displaced.as_ref().unwrap().geometry.output.buffer
            };
            let semantics: Vec<_> = [
                ("POSITION", 0u32, 3u32),
                ("NORMAL", 3, 3),
                ("TANGENT", 6, 4),
            ]
            .into_iter()
            .filter_map(|(semantic, offset, width)| {
                shape
                    .attributes
                    .get(semantic)
                    .and_then(|name| {
                        packed
                            .as_ref()
                            .and_then(|plan| plan.buffers.bindings.get(name))
                    })
                    .map(|binding| (binding, offset, width))
            })
            .collect();
            let pages: BTreeSet<_> = semantics
                .iter()
                .flat_map(|(binding, _, _)| binding.spans.iter().map(|span| span.page))
                .collect();
            let storage_limit=crate::pipelines::scalar_storage_limit(&r.device.limits(),r.adapter_info.backend==wgpu::Backend::Metal,0);
            ensure!(
                pages.len() + 1 <= storage_limit as usize,
                crate::gpu_memory::device_limit("deformed predecessor pages", format_args!("{} storage bindings",pages.len()+1), format_args!("{} storage bindings",storage_limit), "split geometry into smaller objects")
            );
            let slots: BTreeMap<_, _> = pages
                .iter()
                .enumerate()
                .map(|(slot, &page)| (page, slot as u32 + 1))
                .collect();
            let mut code =
                String::from("@group(0) @binding(0) var<storage,read> source:array<u32>;\n");
            for (&page, &slot) in &slots {
                writeln!(
                    code,
                    "@group(0) @binding({slot}) var<storage,read_write> page_{page}:array<u32>;"
                )?;
            }
            writeln!(
                code,
                "@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>) {{let i=id.x;if(i>={count}u) {{return;}}"
            )?;
            for (binding, offset, width) in semantics {
                ensure!(
                    binding.count as usize == count && binding.width == width,
                    "deformed predecessor channel has incompatible rows"
                );
                for span in &binding.spans {
                    let end = span.first + span.count;
                    writeln!(code, "if(i>={}u && i<{end}u) {{", span.first)?;
                    for component in 0..width {
                        writeln!(
                            code,
                            "page_{}[{}u+(i-{}u)*{width}u+{component}u]=source[i*{}u+{}u];",
                            span.page,
                            span.offset,
                            span.first,
                            std::mem::size_of::<DeformedVertex>() / 4,
                            offset + component
                        )?;
                    }
                    code.push_str("}\n");
                }
            }
            code.push_str("}\n");
            let mut types = vec![Some(true)];
            types.extend(pages.iter().map(|_| Some(false)));
            let scatter =
                compute::pipeline_layout(r, &code, "publish evaluated predecessor", &types, "main");
            let mut entries = vec![entry(0, final_output.as_entire_binding())];
            entries.extend(slots.iter().map(|(&page, &slot)| {
                entry(slot, result.pages[&(draw.object, page)].as_entire_binding())
            }));
            let scatter_bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("publish evaluated predecessor"),
                layout: &scatter.get_bind_group_layout(0),
                entries: &entries,
            });
            result.jobs.push(Job {
                root: root.clone(),
                reference_copy: result
                    .reference
                    .as_ref()
                    .and_then(|table| {
                        table
                            .references
                            .iter()
                            .find(|reference| reference.object == draw.object)
                    })
                    .map(|reference| {
                        (
                            final_output.clone(),
                            u64::from(reference.vertex_word) * 4,
                            u64::from(reference.count) * DeformedVertex::BYTES,
                        )
                    }),
                native,
                normals,
                displacement: displaced,
                displacement_first,
                source,
                count: count as u32,
                scatter,
                scatter_bind,
            });
        }
        crate::command_finish::Segments::submit(&r.queue, encoder);
        Ok(Some(result))
    }

    pub(crate) fn publish_errors(
        &self,
        r: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
        destination: &wgpu::Buffer,
    ) {
        let (Some(pipeline), Some(materials)) = (&self.publish, &self.materials) else {
            return;
        };
        let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("canonical geometry diagnostics"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, materials.shader_errors.as_entire_binding()),
                entry(1, destination.as_entire_binding()),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    pub(crate) fn sample(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &vork_time::Request,
    ) -> Result<()> {
        for job in &self.jobs {
            let mut samples = self.animation.samples();
            let mut values = values::sample_storage(&self.nodes);
            let mut worlds: Vec<_> = self.nodes.iter().map(|node| node.world).collect();
            let shifted = if job.root.time_offset == 0. {
                request.clone()
            } else {
                vork_time::Request::Exact(
                    request
                        .exact()?
                        .add(&vork_time::Request::Numeric(job.root.time_offset).exact()?),
                )
            };
            values::sample_pose_inputs(
                &self.nodes,
                &self.assets,
                &self.animation,
                &self.targets,
                &shifted,
                &mut samples,
                &mut values,
                &mut worlds,
            )?;
            if let Some(materials) = &self.materials {
                for (binding, value) in self
                    .animation
                    .material_values(&samples, job.root.time_offset)
                {
                    let upload = if binding.storage.is_some() {
                        let words = programs::sample_words(value, binding);
                        Some((
                            binding.slot as u64 * 4,
                            bytemuck::cast_slice(&words).to_vec(),
                        ))
                    } else if let (animation::Sample::Numeric(value), Some((at, width))) = (
                        value,
                        materials::value_location(binding.slot, binding.parameter),
                    ) {
                        Some((at, bytemuck::cast_slice(&value[..width]).to_vec()))
                    } else {
                        None
                    };
                    if let Some((at, bytes)) = upload {
                        let source = init(
                            &self.device,
                            "canonical material input",
                            &bytes,
                            wgpu::BufferUsages::COPY_SRC,
                        );
                        encoder.copy_buffer_to_buffer(
                            &source,
                            0,
                            &materials.records,
                            at,
                            source.size(),
                        );
                    }
                }
            }
            let displace = |encoder: &mut wgpu::CommandEncoder| -> Result<()> {
                if let Some(displaced) = &job.displacement {
                    if let Some(normals) = &job.normals {
                        normals.encode(encoder);
                    }
                    displaced.encode(
                        &self.device,
                        encoder,
                        self.materials.as_ref().unwrap(),
                        shifted.numeric()?,
                        Mat4::IDENTITY,
                    )?;
                }
                Ok(())
            };
            if job.displacement_first {
                displace(encoder)?;
            }
            if let Some(data) = &job.native {
                if let Some(source) = &job.source {
                    source.encode(encoder);
                }
                let native = data.deform.native.as_ref().unwrap();
                let node = values[native.mesh]
                    .as_ref()
                    .unwrap_or(&self.nodes[native.mesh]);
                let NodeKind::Mesh { morph_weights, .. } = &node.node.kind else {
                    unreachable!()
                };
                let (matrices, weights) = native.palette(&worlds, morph_weights);
                let mut words = passes::deform::header(&data.deform).to_vec();
                for matrix in matrices {
                    words.extend(matrix.to_cols_array());
                }
                words.extend(
                    (0..data.deform.targets).map(|index| weights.get(index).copied().unwrap_or(0.)),
                );
                words.resize(data.params.size() as usize / 4, 0.);
                let upload = init(
                    &self.device,
                    "canonical predecessor palette upload",
                    bytemuck::cast_slice(&words),
                    wgpu::BufferUsages::COPY_SRC,
                );
                encoder.copy_buffer_to_buffer(&upload, 0, &data.params, 0, data.params.size());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &data.bind, &[]);
                    pass.dispatch_workgroups(
                        data.deform.count.div_ceil(constants::DEFORM_GROUP),
                        1,
                        1,
                    );
                }
            }
            if !job.displacement_first {
                displace(encoder)?;
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&job.scatter);
                pass.set_bind_group(0, &job.scatter_bind, &[]);
                pass.dispatch_workgroups(job.count.div_ceil(64), 1, 1);
            }
            if let (Some(table), Some((source, offset, bytes))) =
                (&self.reference, &job.reference_copy)
            {
                encoder.copy_buffer_to_buffer(source, 0, &table.buffer, *offset, *bytes);
            }
        }
        if let Some(table) = &self.reference {
            for reference in &table.references {
                let request = vork_time::Request::Exact(
                    request
                        .exact()?
                        .add(&vork_time::Request::Numeric(reference.root.time_offset).exact()?),
                );
                let mut samples = self.animation.samples();
                let mut values = values::sample_storage(&self.nodes);
                let mut worlds: Vec<_> = self.nodes.iter().map(|node| node.world).collect();
                values::sample_pose_inputs(
                    &self.nodes,
                    &self.assets,
                    &self.animation,
                    &self.targets,
                    &request,
                    &mut samples,
                    &mut values,
                    &mut worlds,
                )?;
                let matrix = reference.root.world(&worlds).to_cols_array();
                let upload = init(
                    &self.device,
                    "canonical collider frame",
                    bytemuck::cast_slice(&matrix),
                    wgpu::BufferUsages::COPY_SRC,
                );
                encoder.copy_buffer_to_buffer(&upload, 0, &table.buffer, reference.row + 16, 64);
            }
        }
        Ok(())
    }
}
