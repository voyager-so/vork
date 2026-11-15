//! Pack native TLAS descriptors from the exact object snapshot consumed by shading.
//! Native references are immutable templates supplied by the backend. Installed
//! material policy sets backend-described traversal bits; the GPU replaces affine
//! coefficients and visibility. No address arithmetic.
use crate::*;
use std::fmt::Write as _;

pub(crate) struct Placement {
    pub buffer: wgpu::Buffer,
    pub references: Vec<wgpu::Blas>,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    count: u32,
    width: u32,
    points: Vec<(wgpu::ComputePipeline,wgpu::BindGroup,u32,u32)>,
    software_masked: Option<SoftwareMasked>,
}

/// Mesa 25.2 lavapipe encodes only the active leaf count, but its BVH retains
/// sparse input leaf offsets. A zero-mask slot before a live slot leaves that
/// later leaf uninitialized. Keep every software TLAS slot active using a finite
/// zero-area triangle that cannot intersect any ray. Hardware keeps zero masks.
struct SoftwareMasked {
    templates: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
}
impl SoftwareMasked {
    fn new(r: &Renderer, buffer: &wgpu::Buffer, bytes: &[u8], references: &mut Vec<wgpu::Blas>, layout: wgpu::hal::TlasInstanceLayout, count: u32, width: u32) -> Result<Self> {
        let d = &r.device;
        let vertices = init(d, "masked software triangle", bytemuck::cast_slice(&[[0f32; 3]; 3]), wgpu::BufferUsages::BLAS_INPUT);
        let size = wgpu::BlasTriangleGeometrySizeDescriptor { vertex_format: wgpu::VertexFormat::Float32x3, vertex_count: 3, index_format: None, index_count: None, flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE };
        let blas = d.create_blas(&wgpu::CreateBlasDescriptor { label: Some("masked software BLAS"), flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE, update_mode: wgpu::AccelerationStructureUpdateMode::Build }, wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] });
        let mut encoder = d.create_command_encoder(&Default::default());
        encoder.build_acceleration_structures([wgpu::BlasBuildEntry { blas: &blas, geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry { size: &size, vertex_buffer: &vertices, first_vertex: 0, vertex_stride: 12, index_buffer: None, first_index: None, transform_buffer: None, transform_buffer_offset: None }]) }].iter(), std::iter::empty());
        r.queue.submit([encoder.finish()]);
        let (sentinel_layout, sentinel) = blas.native_tlas_instance(0, 255).context("masked software TLAS template")?;
        ensure!(sentinel_layout == layout, "software TLAS template layout differs");
        references.push(blas);
        let mut templates = bytes.to_vec();
        templates.extend(sentinel);
        let templates = init(d, "software TLAS templates", &templates, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let words = layout.stride / 4;
        let code = format!(r#"
@group(0) @binding(0) var<storage,read> templates:array<u32>;
@group(0) @binding(1) var<storage,read_write> descriptors:array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {{
    let i=gid.x+gid.y*{width}u;if(i>={count}u) {{return;}}
    let base=i*{words}u;
    if(((descriptors[base+{mask}u]>>{shift}u)&255u)==0u) {{
        for(var word=0u;word<{words}u;word++) {{descriptors[base+word]=templates[{count}u*{words}u+word];}}
    }}
}}
"#, mask=layout.mask, shift=layout.mask_shift);
        let pipeline = crate::compute::pipeline_layout(r, &code, "software masked TLAS slots", &[Some(true), Some(false)], "main");
        let group = d.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("software masked TLAS slots"), layout: &pipeline.get_bind_group_layout(0), entries: &[entry(0, templates.as_entire_binding()), entry(1, buffer.as_entire_binding())] });
        Ok(Self { templates, pipeline, group })
    }
}

/// Generate from backend layout facts, shared by production and packing tests. The text
/// holds no scene size: the instance count is the mapping's length and the dispatch row
/// width follows from it and the device's workgroup limit (`row_groups`), as `dispatch`
/// computes them, so one pipeline places any number of instances.
fn source(
    stride: u32,
    transform: [u32; 12],
    mask_word: u32,
    mask_shift: u32,
    row_groups: u32,
) -> String {
    let mut code = format!(
        r#"
{finite_checks}
{object_abi}
@group(0) @binding(0) var<storage,read> objects:array<Object>;
@group(0) @binding(1) var<storage,read> mapping:array<vec4<u32>>;
@group(0) @binding(2) var<storage,read_write> descriptors:array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {{
    let count=arrayLength(&mapping);
    let i=gid.x+gid.y*min((count+63u)/64u,{row_groups}u)*64u; if(i>=count) {{return;}}
    let object=objects[mapping[i].x];
    var rows=object.model;
    let a=rows[0].xyz; let b=rows[1].xyz; let c=rows[2].xyz;
    let determinant=dot(a,cross(b,c));
    var mask=select(0u,object.metadata.y,mapping[i].z!=0u);
    // A masked descriptor must still carry a finite nonsingular placeholder.
    if(mask==0u || determinant==0. || !finite_f32(determinant) ||
       !finite4(rows[0]) ||
       !finite4(rows[1]) ||
       !finite4(rows[2])) {{
        mask=0u; rows=mat3x4<f32>(vec4(1.,0.,0.,0.),vec4(0.,1.,0.,0.),vec4(0.,0.,1.,0.));
    }}
    if(mapping[i].y!=0u && (mask & {diffuse}u)!=0u) {{ mask=mask | {occluder}u; }}
    if(mapping[i].w!=0u && (mask & {shadow}u)!=0u) {{mask=mask | {solid}u;}}
    let base=i*{words}u;
"#,
        object_abi = crate::object_abi::wgsl("Object"),
        finite_checks = crate::compute::FINITE_WGSL,
        shadow=constants::RAY_SHADOW,
        solid=constants::RAY_SHADOW_SOLID,
        diffuse = constants::RAY_DIFFUSE,
        occluder = constants::RAY_GI_OCCLUDER,
        words = stride / 4
    );
    for (coefficient, offset) in transform.into_iter().enumerate() {
        writeln!(
            code,
            "descriptors[base+{offset}u]=bitcast<u32>(rows[{}][{}]);",
            coefficient / 4,
            coefficient % 4
        )
        .unwrap();
    }
    let bits = 0xffu32 << mask_shift;
    writeln!(code,"descriptors[base+{mask_word}u]=(descriptors[base+{mask_word}u] & {}u) | ((mask & 255u)<<{mask_shift}u);\n}}",!bits).unwrap();
    code
}

impl Placement {
    pub(crate) fn new(
        r: &Renderer,
        objects: &wgpu::Buffer,
        instances: &[(u32, bool, bool, bool, wgpu::Blas)],
        retained: &std::sync::OnceLock<wgpu::ComputePipeline>,
    ) -> Result<Self> {
        Self::with_points(r,objects,instances,retained,&vec![None;instances.len()],&vec![(false,false);instances.len()])
    }
    pub(crate) fn with_points(
        r: &Renderer,
        objects: &wgpu::Buffer,
        instances: &[(u32,bool,bool,bool,wgpu::Blas)],
        retained: &std::sync::OnceLock<wgpu::ComputePipeline>,
        point_poses: &[Option<(wgpu::Buffer,u32)>],
        triangle_sides: &[(bool,bool)],
    ) -> Result<Self> {
        ensure!(triangle_sides.len()==instances.len(),"triangle sidedness mapping length differs from TLAS");
        ensure!(point_poses.len()==instances.len(),"point placement mapping length differs from TLAS");
        let mut bytes = Vec::new();
        let mut mapping = Vec::new();
        let mut references = Vec::new();
        let mut abi = None;
        let mut templates = std::collections::HashMap::new();
        for ((object, opaque, hardware_visible, shadow_solid, blas), &(double_sided, solid_exit)) in instances.iter().zip(triangle_sides) {
            let (layout, template) = match templates.entry(blas.clone()) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    references.push(blas.clone());
                    entry.insert(
                        blas.native_tlas_instance(0, 0)
                            .context("native TLAS reference template")?,
                    )
                }
            };
            if let Some(previous) = abi {
                assert_eq!(previous, *layout);
            } else {
                abi = Some(*layout);
            }
            let custom_index = bytes.len() + layout.custom_index as usize * 4;
            let flags = bytes.len() + layout.triangle_flags as usize * 4;
            bytes.extend_from_slice(template);
            let mut options = u32::from_ne_bytes(bytes[flags..flags+4].try_into().unwrap());
            options &= !(layout.disable_triangle_culling | layout.force_non_opaque);
            if double_sided || solid_exit { options |= layout.disable_triangle_culling; }
            // Path transport accepts solid exits; diffuse/reflection queries must
            // still reject a single-sided rear surface through their candidate rule.
            if solid_exit && !double_sided { options |= layout.force_non_opaque; }
            bytes[flags..flags+4].copy_from_slice(&options.to_ne_bytes());
            // The backend owns this field's location. Its template has mask zero;
            // the installed object index is already admitted to the portable 24 bits.
            bytes[custom_index..custom_index + 4].copy_from_slice(&object.to_ne_bytes());
            // Software field bounds keep an identity but never become triangle hits.
            mapping.push([*object, u32::from(*opaque), u32::from(*hardware_visible), u32::from(*shadow_solid)]);
        }
        let layout = abi.context("empty native TLAS placement")?;
        let buffer = init(
            &r.device,
            "native TLAS descriptors",
            &bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::TLAS_INPUT | wgpu::BufferUsages::COPY_DST,
        );
        let mapping = init(
            &r.device,
            "TLAS object mapping",
            bytemuck::cast_slice(&mapping),
            wgpu::BufferUsages::STORAGE,
        );
        let width = (instances.len() as u32)
            .div_ceil(64)
            .min(r.device.limits().max_compute_workgroups_per_dimension);
        let pipeline = retained.get_or_init(|| crate::compute::pipeline_layout(
            r,
            &source(
                layout.stride,
                layout.transform,
                layout.mask,
                layout.mask_shift,
                r.device.limits().max_compute_workgroups_per_dimension,
            ),
            "native TLAS placement",
            &[Some(true), Some(true), Some(false)],
            "main",
        )).clone();
        let group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("native TLAS placement"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, objects.as_entire_binding()),
                entry(1, mapping.as_entire_binding()),
                entry(2, buffer.as_entire_binding()),
            ],
        });
        let mut point_groups:Vec<(wgpu::Buffer,Vec<[u32;4]>)>=Vec::new();
        for (slot,pose) in point_poses.iter().enumerate() {
            let Some((poses,row))=pose else {continue};
            let index=point_groups.iter().position(|(b,_)|b==poses).unwrap_or_else(||{point_groups.push((poses.clone(),Vec::new()));point_groups.len()-1});
            point_groups[index].1.push([slot as u32,instances[slot].0,*row,0]);
        }
        let mut points=Vec::new();
        for (poses,rows) in point_groups {
            let count=rows.len() as u32;
            let width=count.div_ceil(64).min(r.device.limits().max_compute_workgroups_per_dimension);
            let mut code=format!(r#"
{finite_checks}
{object_abi}
@group(0) @binding(0) var<storage,read> objects:array<Object>;
@group(0) @binding(1) var<storage,read> mapping:array<vec4<u32>>;
@group(0) @binding(2) var<storage,read_write> descriptors:array<u32>;
@group(0) @binding(3) var<storage,read> poses:array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {{
    let i=gid.x+gid.y*{dispatch_width}u;if(i>={count}u) {{return;}}
    let map=mapping[i];let pose=bitcast<vec4<f32>>(vec4<u32>(poses[map.z],poses[map.z+1u],poses[map.z+2u],poses[map.z+3u]));
    let object=objects[map.y];var rows=object.model;
    let base=map.x*{words}u;
    var mask=(descriptors[base+{mask_word}u]>>{mask_shift}u)&255u;
    let radius=pose.w*0.5;
    for(var row=0u;row<3u;row++) {{let original=rows[row];rows[row]=vec4<f32>(original.xyz*radius,dot(original.xyz,pose.xyz)+original.w);}}
    let determinant=dot(rows[0].xyz,cross(rows[1].xyz,rows[2].xyz));
    if(!(radius>0.) || determinant==0. || !finite_f32(determinant) || !finite4(pose) || !finite4(rows[0]) || !finite4(rows[1]) || !finite4(rows[2])) {{mask=0u;}}
    if(mask==0u) {{rows=mat3x4<f32>(vec4(1.,0.,0.,0.),vec4(0.,1.,0.,0.),vec4(0.,0.,1.,0.));}}
"#,finite_checks=crate::compute::FINITE_WGSL,object_abi=crate::object_abi::wgsl("Object"),dispatch_width=width*64,words=layout.stride/4,mask_word=layout.mask,mask_shift=layout.mask_shift);
            for (coefficient,offset) in layout.transform.into_iter().enumerate() {
                writeln!(code,"descriptors[base+{offset}u]=bitcast<u32>(rows[{}][{}]);",coefficient/4,coefficient%4)?;
            }
            writeln!(code,"descriptors[base+{}u]=(descriptors[base+{}u]&{}u)|(mask<<{}u);}}",layout.mask,layout.mask,!(255u32<<layout.mask_shift),layout.mask_shift)?;
            let pipeline=crate::compute::pipeline_layout(r,&code,"compact point TLAS placement",&[Some(true),Some(true),Some(false),Some(true)],"main");
            let mapping=init(&r.device,"point placement mapping",bytemuck::cast_slice(&rows),wgpu::BufferUsages::STORAGE);
            let group=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("compact point TLAS placement"),layout:&pipeline.get_bind_group_layout(0),entries:&[
                entry(0,objects.as_entire_binding()),entry(1,mapping.as_entire_binding()),entry(2,buffer.as_entire_binding()),entry(3,poses.as_entire_binding())]});
            points.push((pipeline,group,count,width));
        }
        let software_masked = if r.adapter_info.device_type == wgpu::DeviceType::Cpu {
            Some(SoftwareMasked::new(r, &buffer, &bytes, &mut references, layout, instances.len() as u32, width * 64)?)
        } else { None };
        Ok(Self {
            software_masked,
            points,
            buffer,
            references,
            pipeline,
            group,
            count: instances.len() as u32,
            width,
        })
    }
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder, tlas: &wgpu::Tlas, force_rebuild: bool) {
        if let Some(masked) = &self.software_masked {
            // Restore original references before evaluating this sample: a hidden
            // object or point can become visible again, including backward seeks.
            encoder.copy_buffer_to_buffer(&masked.templates, 0, &self.buffer, 0, self.buffer.size());
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("native TLAS placement"),
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(self.width, self.count.div_ceil(64).div_ceil(self.width), 1);
        }
        for (pipeline,group,count,width) in &self.points {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("compact point TLAS placement"),..Default::default()});
            pass.set_pipeline(pipeline);pass.set_bind_group(0,group,&[]);
            pass.dispatch_workgroups(*width,count.div_ceil(64).div_ceil(*width),1);
        }
        if let Some(masked) = &self.software_masked {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("software masked TLAS slots"), ..Default::default() });
            pass.set_pipeline(&masked.pipeline);
            pass.set_bind_group(0, &masked.group, &[]);
            pass.dispatch_workgroups(self.width, self.count.div_ceil(64).div_ceil(self.width), 1);
        }
        let references: Vec<_> = self.references.iter().collect();
        // Templates retain their backend-owned native reference and object ID;
        // software placeholders also copy complete backend-owned templates above.
        unsafe {
            encoder.build_tlas_from_buffer(tlas, &self.buffer, 0, self.count, &references, force_rebuild);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn native_descriptor_packing_preserves_references_and_masks_tails() -> Result<()> {
        let r = crate::test_renderer()?;
        let mut visible = Object::new(Mat4::from_scale_rotation_translation(
            Vec3::new(-2., 3., 0.5),
            glam::Quat::from_rotation_y(0.7),
            Vec3::new(11., -7., 4.),
        ));
        visible.metadata[1] = constants::RAY_DIFFUSE | constants::RAY_CAMERA;
        let mut tail = visible;
        tail.metadata[1] = 0;
        let collapsed = Object::new(Mat4::from_scale(Vec3::ZERO));
        let objects = init(
            &r.device,
            "packing objects",
            bytemuck::cast_slice(&gpu_objects(&[visible, tail, collapsed])),
            wgpu::BufferUsages::STORAGE,
        );
        let mapping = init(
            &r.device,
            "packing map",
            bytemuck::cast_slice(&[[0u32, 1, 1, 0], [1, 1, 1, 0], [2, 1, 1, 0], [0, 1, 0, 0]]),
            wgpu::BufferUsages::STORAGE,
        );
        // Exercise both native ABIs even on a device without ray-query support.
        for (stride, transform, mask, shift) in [
            (64, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 12, 24),
            (72, [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], 13, 0),
        ] {
            let words = stride / 4;
            let initial: Vec<u32> = (0..4 * words).map(|i| 0x12345600 + i).collect();
            let output = init(
                &r.device,
                "packing descriptors",
                bytemuck::cast_slice(&initial),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let pipeline = crate::compute::pipeline_layout(
                &r,
                &source(stride, transform, mask, shift, r.device.limits().max_compute_workgroups_per_dimension),
                "test native packing",
                &[Some(true), Some(true), Some(false)],
                "main",
            );
            let group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    entry(0, objects.as_entire_binding()),
                    entry(1, mapping.as_entire_binding()),
                    entry(2, output.as_entire_binding()),
                ],
            });
            let readback = r.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: output.size(),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = r.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
            r.queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback.map_async(wgpu::MapMode::Read, .., move |result| {
                tx.send(result).unwrap();
            });
            r.wait(None)?;
            rx.recv()??;
            let bytes = readback.get_mapped_range(..);
            let actual: &[u32] = bytemuck::cast_slice(&bytes);
            let mut expected = initial;
            for i in 0..4usize {
                let rows = if i == 0 {
                    visible.gpu().model
                } else {
                    Object::new(Mat4::IDENTITY).gpu().model
                };
                for (coefficient, offset) in transform.into_iter().enumerate() {
                    expected[i * words as usize + offset as usize] =
                        rows[coefficient / 4][coefficient % 4].to_bits();
                }
                let index = i * words as usize + mask as usize;
                let visibility = if i == 0 {
                    visible.metadata[1] | constants::RAY_GI_OCCLUDER
                } else {
                    0
                };
                expected[index] = (expected[index] & !(255 << shift)) | (visibility << shift);
            }
            assert_eq!(
                actual, expected,
                "native transform order, tail masks and immutable reference/options words"
            );
            drop(bytes);
            readback.unmap();
        }
        r.errors.check()?;
        Ok(())
    }
}
