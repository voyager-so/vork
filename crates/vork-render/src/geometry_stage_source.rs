//! Adapt an evaluated stream to a builtin deformation source. Geometry stays on
//! the GPU; current and both historical endpoints remain distinct.
use crate::*;

#[derive(Clone)]
pub(crate) struct Source {
    pub buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    groups: [u32; 2],
}

impl Source {
    pub(crate) fn new(
        r: &Renderer,
        rest: &wgpu::Buffer,
        input: &EvaluatedGeometry,
        count: u32,
    ) -> Result<Self> {
        let stride = std::mem::size_of::<Vertex>() as u64;
        let evaluated_stride = std::mem::size_of::<DeformedVertex>() as u64;
        ensure!(
            rest.size() == u64::from(count) * stride
                && input.output.size == u64::from(count) * evaluated_stride
                && input.previous_output.size == u64::from(count) * evaluated_stride * 2,
            "deformation stage predecessor has incompatible vertex correspondence"
        );
        let size = rest
            .size()
            .checked_mul(3)
            .context("deformation input size overflow")?;
        let limit = u64::from(r.device.limits().max_storage_buffer_binding_size)
            .min(r.device.limits().max_buffer_size);
        ensure!(
            size <= limit,
            crate::gpu_memory::device_limit("deformation input", format_args!("{size} bytes"), format_args!("{limit} bytes"), "split the mesh into smaller objects")
        );
        let buffer = gpu_memory::buffer(
            &r.device,
            &wgpu::BufferDescriptor {
                label: Some("geometry stage deformation source"),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        );
        let groups = count.div_ceil(64);
        let width = groups
            .min(r.device.limits().max_compute_workgroups_per_dimension)
            .max(1);
        let words = stride / 4;
        let changing = evaluated_stride / 4;
        let code = format!(
            r#"
@group(0) @binding(0) var<storage,read> rest:array<u32>;
@group(0) @binding(1) var<storage,read> current:array<u32>;
@group(0) @binding(2) var<storage,read> history:array<u32>;
@group(0) @binding(3) var<storage,read_write> destination:array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32) {{
    let i=(group.y*{width}u+group.x)*64u+lane;
    if(i>={count}u) {{return;}}
    for(var pose=0u;pose<3u;pose++) {{
        let dst=(pose*{count}u+i)*{words}u;
        for(var c=0u;c<{words}u;c++) {{destination[dst+c]=rest[i*{words}u+c];}}
        for(var c=0u;c<{changing}u;c++) {{
            var value=0u;
            if(pose==0u) {{value=current[i*{changing}u+c];}}
            else {{value=history[i*{changing}u*2u+(pose-1u)*{changing}u+c];}}
            if(c<6u) {{destination[dst+c]=value;}}
            else {{destination[dst+{words}u-4u+c-6u]=value;}}
        }}
    }}
}}
"#
        );
        let pipeline = compute::pipeline_layout(
            r,
            &code,
            "geometry stage source",
            &[Some(true), Some(true), Some(true), Some(false)],
            "main",
        );
        let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("geometry stage source"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, rest.as_entire_binding()),
                entry(1, input.output.binding()),
                entry(2, input.previous_output.binding()),
                entry(3, buffer.as_entire_binding()),
            ],
        });
        Ok(Self {
            buffer,
            pipeline,
            bind,
            groups: [width, groups.div_ceil(width)],
        })
    }

    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.dispatch_workgroups(self.groups[0], self.groups[1], 1);
    }
}

/// Repack a predecessor's typed attributes into the material reader's rows.
/// Each pose has its own rows; unchanged attributes may share current storage.
pub(crate) struct AttributeRows {
    pub buffer: wgpu::Buffer,
    pub pose_stride: u32,
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    groups: [u32; 2],
}
impl AttributeRows {
    pub(crate) fn new(
        r: &Renderer,
        current: &wgpu::Buffer,
        historical: Option<&wgpu::Buffer>,
        rest: &wgpu::Buffer,
        count: u32,
        layout: geometry::VertexLayout,
    ) -> Result<Self> {
        let rows = u64::from(layout.custom) + 1;
        let size = u64::from(count) * rows * 16 * 3;
        let limit = u64::from(r.device.limits().max_storage_buffer_binding_size)
            .min(r.device.limits().max_buffer_size);
        ensure!(
            size <= limit,
            crate::gpu_memory::device_limit("geometry stage attribute inputs", format_args!("{size} bytes"), format_args!("{limit} bytes"), "split geometry into smaller objects")
        );
        ensure!(
            rest.size() == u64::from(count) * rows * 16,
            "geometry stage material attribute rows do not match predecessor topology"
        );
        let buffer = gpu_memory::buffer(
            &r.device,
            &wgpu::BufferDescriptor {
                label: Some("geometry stage material attribute poses"),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        );
        let stride = layout.attribute_stride() / 4;
        let custom_start = stride - u64::from(layout.custom) * 4;
        let target_stride = rows * 4;
        let groups = count.div_ceil(64);
        let width = groups
            .min(r.device.limits().max_compute_workgroups_per_dimension)
            .max(1);
        let historical_load = if historical.is_some() {
            format!("if(pose!=0u) {{return history[i*{stride}u*2u+(pose-1u)*{stride}u+c];}}")
        } else {
            String::new()
        };
        let uv = if layout.uv[0] != constants::VERTEX_DEFAULT_UV {
            let at = 3 + u64::from(layout.uv[0]) * 2;
            format!(
                "destination[dst]=load(i,{at}u,pose);destination[dst+1u]=load(i,{}u,pose);",
                at + 1
            )
        } else {
            String::new()
        };
        let code = format!(
            r#"
@group(0) @binding(0) var<storage,read> current:array<u32>;
@group(0) @binding(1) var<storage,read> history:array<u32>;
@group(0) @binding(2) var<storage,read> rest:array<u32>;
@group(0) @binding(3) var<storage,read_write> destination:array<u32>;
fn load(i:u32,c:u32,pose:u32)->u32 {{{historical_load} return current[i*{stride}u+c];}}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {{
 let i=(group.y*{width}u+group.x)*64u+lane;if(i>={count}u) {{return;}}
 for(var pose=0u;pose<3u;pose++) {{
  let dst=(pose*{count}u+i)*{target_stride}u;
  for(var c=0u;c<4u;c++) {{destination[dst+c]=rest[i*{target_stride}u+c];}}
  {uv}
  for(var c=0u;c<{}u;c++) {{destination[dst+4u+c]=load(i,{custom_start}u+c,pose);}}
 }}
}}
"#,
            u32::from(layout.custom) * 4
        );
        let pipeline = compute::pipeline_layout(
            r,
            &code,
            "geometry stage attribute poses",
            &[Some(true), Some(true), Some(true), Some(false)],
            "main",
        );
        let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("geometry stage attribute poses"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, current.as_entire_binding()),
                entry(1, historical.unwrap_or(current).as_entire_binding()),
                entry(2, rest.as_entire_binding()),
                entry(3, buffer.as_entire_binding()),
            ],
        });
        Ok(Self {
            buffer,
            pose_stride: (u64::from(count) * rows)
                .try_into()
                .context("material attribute pose addressing overflow")?,
            pipeline,
            bind,
            groups: [width, groups.div_ceil(width)],
        })
    }
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.dispatch_workgroups(self.groups[0], self.groups[1], 1);
    }
}

/// Generic final-stream bounds for a stage that does not already fuse a bound
/// reduction into its numerical kernel. Only this eight-word receipt is mapped.
#[derive(Clone)]
pub(crate) struct Bounds {
    initial: wgpu::Buffer,
    values: wgpu::Buffer,
    readback: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    groups: [u32; 2],
}
impl Bounds {
    pub(crate) fn new(r: &Renderer, input: &geometry::VertexStream, count: u32) -> Self {
        let initial = init(
            &r.device,
            "geometry stage bounds reset",
            bytemuck::cast_slice(&[u32::MAX, u32::MAX, u32::MAX, 0, 0, 0, 0, 0]),
            wgpu::BufferUsages::COPY_SRC,
        );
        let values = gpu_memory::buffer(
            &r.device,
            &wgpu::BufferDescriptor {
                label: Some("geometry stage bounds"),
                size: 32,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        );
        let readback = gpu_memory::buffer(
            &r.device,
            &wgpu::BufferDescriptor {
                label: Some("geometry stage bounds receipt"),
                size: 32,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        );
        let groups = count.div_ceil(64);
        let width = groups
            .min(r.device.limits().max_compute_workgroups_per_dimension)
            .max(1);
        let stride = std::mem::size_of::<DeformedVertex>() / 4;
        let code = format!(
            r#"
@group(0) @binding(0) var<storage,read> source:array<u32>;
@group(0) @binding(1) var<storage,read_write> bounds:array<atomic<u32>>;
fn ordered(v:u32)->u32 {{return select(v^0x80000000u,~v,(v&0x80000000u)!=0u);}}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {{
    let i=(group.y*{width}u+group.x)*64u+lane;
    if(i>={count}u) {{return;}}
    for(var c=0u;c<3u;c++) {{
        let value=source[i*{stride}u+c];
        if((value&0x7f800000u)==0x7f800000u) {{atomicOr(&bounds[6],1u);}}
        else {{atomicMin(&bounds[c],ordered(value));atomicMax(&bounds[c+3u],ordered(value));}}
    }}
}}
"#
        );
        let pipeline = compute::pipeline_layout(
            r,
            &code,
            "geometry stage bounds",
            &[Some(true), Some(false)],
            "main",
        );
        let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("geometry stage bounds"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, input.binding()),
                entry(1, values.as_entire_binding()),
            ],
        });
        Self {
            initial,
            values,
            readback,
            pipeline,
            bind,
            groups: [width, groups.div_ceil(width)],
        }
    }
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.initial, 0, &self.values, 0, 32);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(self.groups[0], self.groups[1], 1);
        }
        encoder.copy_buffer_to_buffer(&self.values, 0, &self.readback, 0, 32);
    }
    pub(crate) fn map(
        &self,
        callback: impl FnOnce(std::result::Result<(), wgpu::BufferAsyncError>) + Send + 'static,
    ) {
        self.readback.map_async(wgpu::MapMode::Read, .., callback);
    }
    pub(crate) fn collect(&self) -> Result<(Vec3, Vec3)> {
        let mapped = self.readback.get_mapped_range(..);
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        let decode = |v: u32| {
            f32::from_bits(if v & 0x80000000 != 0 {
                v ^ 0x80000000
            } else {
                !v
            })
        };
        let result = if words[6] == 0 {
            Ok((
                Vec3::new(decode(words[0]), decode(words[1]), decode(words[2])),
                Vec3::new(decode(words[3]), decode(words[4]), decode(words[5])),
            ))
        } else {
            Err(anyhow::anyhow!(
                "geometry stage produced nonfinite positions; reduce the deformation or producer values"
            ))
        };
        drop(mapped);
        self.readback.unmap();
        result
    }
}

/// Restore omitted face normals on the canonical authored row domain. Seam
/// admission has already rejected ambiguous normal-dependent displacement.
pub(crate) struct Normals {
    pub output:wgpu::Buffer,
    pipeline:wgpu::ComputePipeline,
    bind:wgpu::BindGroup,
    groups:[u32;2],
}
impl Normals {
    pub(crate) fn new(r:&Renderer,input:&wgpu::Buffer,indices:&[u32],count:u32)->Result<Self> {
        let mut first=vec![[u32::MAX;4];count as usize];
        for triangle in indices.chunks_exact(3) {
            for &vertex in triangle {
                let row=first.get_mut(vertex as usize).context("canonical normal index outside geometry")?;
                if row[0]==u32::MAX {*row=[triangle[0],triangle[1],triangle[2],0];}
            }
        }
        let faces=init(&r.device,"canonical normal correspondence",bytemuck::cast_slice(&first),wgpu::BufferUsages::STORAGE);
        let output=gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {label:Some("canonical geometric normals"),size:u64::from(count)*DeformedVertex::BYTES,usage:wgpu::BufferUsages::STORAGE,mapped_at_creation:false});
        let groups=count.div_ceil(64);let width=groups.min(r.device.limits().max_compute_workgroups_per_dimension).max(1);
        let stride=DeformedVertex::WORDS;
        let normal=DeformedVertex::NORMAL_WORD;
        let code=format!(r#"
@group(0) @binding(0) var<storage,read> source:array<u32>;
@group(0) @binding(1) var<storage,read> faces:array<vec4<u32>>;
@group(0) @binding(2) var<storage,read_write> normal_output:array<u32>;
fn position(i:u32)->vec3<f32>{{let at=i*{stride}u;return bitcast<vec3<f32>>(vec3(source[at],source[at+1u],source[at+2u]));}}
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {{
    let i=(group.y*{width}u+group.x)*64u+lane;if(i>={count}u){{return;}}
    for(var c=0u;c<{stride}u;c++){{normal_output[i*{stride}u+c]=source[i*{stride}u+c];}}
    let face=faces[i];var normal=vec3(0.);
    if(face.x!=0xffffffffu){{
        let a=position(face.x);let b=position(face.y);let c=position(face.z);
        let scale=max(max(max(abs(a.x),abs(a.y)),abs(a.z)),max(max(max(abs(b.x),abs(b.y)),abs(b.z)),max(max(abs(c.x),abs(c.y)),abs(c.z))));
        if(scale>0.){{let area=cross(b/scale-a/scale,c/scale-a/scale);let m=max(max(abs(area.x),abs(area.y)),abs(area.z));if(m>0.){{normal=normalize(area/m);}}}}
    }}
    normal_output[i*{stride}u+{normal}u]=bitcast<u32>(normal.x);normal_output[i*{stride}u+{normal}u+1u]=bitcast<u32>(normal.y);normal_output[i*{stride}u+{normal}u+2u]=bitcast<u32>(normal.z);
}}
"#);
        let pipeline=compute::pipeline_layout(r,&code,"canonical geometric normals",&[Some(true),Some(true),Some(false)],"main");
        let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("canonical geometric normals"),layout:&pipeline.get_bind_group_layout(0),entries:&[
            entry(0,input.as_entire_binding()),entry(1,faces.as_entire_binding()),entry(2,output.as_entire_binding()),
        ]});
        Ok(Self {output,pipeline,bind,groups:[width,groups.div_ceil(width)]})
    }
    pub(crate) fn encode(&self,encoder:&mut wgpu::CommandEncoder){
        let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&self.pipeline);pass.set_bind_group(0,&self.bind,&[]);pass.dispatch_workgroups(self.groups[0],self.groups[1],1);
    }
}

/// Preserve packed predecessor columns that an installed stage does not own.
/// This splice runs only for an evaluated predecessor and a nonempty carry mask.
pub(crate) struct Carried {
    pipeline:wgpu::ComputePipeline,
    bind:wgpu::BindGroup,
    groups:[u32;2],
    pub bounds:Option<Bounds>,
}
impl Carried {
    pub(crate) fn new(r:&Renderer,input:&EvaluatedGeometry,output:&EvaluatedGeometry,mask:[bool;3])->Result<Self> {
        ensure!(input.output.size==output.output.size && input.previous_output.size==output.previous_output.size,
            "carried geometry columns require matching predecessor rows; declare a topology correspondence before compute");
        let count=u32::try_from(output.output.size/DeformedVertex::BYTES)?;
        let groups=count.div_ceil(64);let width=groups.min(r.device.limits().max_compute_workgroups_per_dimension).max(1);
        let stride=DeformedVertex::WORDS;
        let history_stride=2*stride;
        let components:Vec<_>=(0..stride).filter(|&word|mask[if word<DeformedVertex::NORMAL_WORD {0} else if word<DeformedVertex::TANGENT_WORD {1} else {2}]).collect();
        let mut copies=String::new();
        for word in components {
            use std::fmt::Write;
            writeln!(copies,"destination[i*{stride}u+{word}u]=source[i*{stride}u+{word}u];for(var pose=0u;pose<2u;pose++){{destination_history[i*{history_stride}u+pose*{stride}u+{word}u]=source_history[i*{history_stride}u+pose*{stride}u+{word}u];}}")?;
        }
        let code=format!(r#"
@group(0) @binding(0) var<storage,read> source:array<u32>;
@group(0) @binding(1) var<storage,read> source_history:array<u32>;
@group(0) @binding(2) var<storage,read_write> destination:array<u32>;
@group(0) @binding(3) var<storage,read_write> destination_history:array<u32>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {{
    let i=(group.y*{width}u+group.x)*64u+lane;if(i>={count}u){{return;}}{copies}
}}
"#);
        let pipeline=compute::pipeline_layout(r,&code,"carry unowned geometry columns",&[Some(true),Some(true),Some(false),Some(false)],"main");
        let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("carry unowned geometry columns"),layout:&pipeline.get_bind_group_layout(0),entries:&[
            entry(0,input.output.binding()),entry(1,input.previous_output.binding()),entry(2,output.output.binding()),entry(3,output.previous_output.binding()),
        ]});
        let bounds=mask[0].then(||Bounds::new(r,&output.output,count));
        Ok(Self {pipeline,bind,groups:[width,groups.div_ceil(width)],bounds})
    }
    pub(crate) fn encode(&self,encoder:&mut wgpu::CommandEncoder) {
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&self.pipeline);pass.set_bind_group(0,&self.bind,&[]);pass.dispatch_workgroups(self.groups[0],self.groups[1],1);}
        if let Some(bounds)=&self.bounds {bounds.encode(encoder);}
    }
}
