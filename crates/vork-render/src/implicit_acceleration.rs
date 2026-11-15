//! Conservative occupied blocks over the published scalar field. Tri-linear
//! values stay between their eight corners, so a block with one strict sign
//! cannot contain the zero set. Publication and queries remain entirely on GPU.
use crate::*;
use std::fmt::Write;

const BLOCK: u32 = vork_slang::registry::IMPLICIT_BLOCK;
pub(crate) struct Acceleration {
    pub buffer: wgpu::Buffer,
    #[cfg(test)] pub builds: std::sync::atomic::AtomicUsize,
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    blocks: [u32;3],
    diagnostics_pipeline: wgpu::ComputePipeline,
    diagnostics_layout: wgpu::BindGroupLayout,
}
impl Acceleration {
    pub fn new(r:&Renderer,dims:[u32;3],spans:&[(wgpu::Buffer,buffers::AttributeSpan)],object:u32) -> Result<Self> {
        let blocks=dims.map(|d|(d-1).div_ceil(BLOCK));
        let bytes=blocks.into_iter().map(u64::from).product::<u64>()*4;
        let buffer=gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {
            label:Some("implicit occupied blocks"),size:bytes,
            usage:wgpu::BufferUsages::STORAGE,mapped_at_creation:false,
        });
        let mut code=String::new();
        for i in 0..spans.len() {writeln!(code,"@group(0) @binding({i}) var<storage,read> page_{i}:array<u32>;")?;}
        writeln!(code,"@group(0) @binding({}) var<storage,read_write> blocks:array<u32>;",spans.len())?;
        code.push_str("fn value(i:u32)->f32 {\n");
        for (i,(_,span)) in spans.iter().enumerate() {
            writeln!(code,"if(i>={}u && i-{}u<{}u) {{return bitcast<f32>(page_{i}[{}u+i-{}u]);}}",span.first,span.first,span.count,span.offset,span.first)?;
        }
        code.push_str("return 0.;}\nvar<workgroup> signs:atomic<u32>;\n");
        writeln!(code,"const dims=vec3<u32>({}u,{}u,{}u);const counts=vec3<u32>({}u,{}u,{}u);",dims[0],dims[1],dims[2],blocks[0],blocks[1],blocks[2])?;
        writeln!(code,"const edge={}u;const step={}u;",BLOCK+1,BLOCK)?;
        code.push_str(r#"
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    if(lane==0u) {atomicStore(&signs,0u);} workgroupBarrier();
    var bits=0u;
    for(var i=lane;i<edge*edge*edge;i+=64u) {
        let c=group*step+vec3<u32>(i%edge,(i/edge)%edge,i/(edge*edge));
        if(all(c<dims)) {
            let v=value(c.x+dims.x*(c.y+dims.y*c.z));
            bits|=select(0u,1u,v<=0.)|select(0u,2u,v>=0.);
            var invalid=(bitcast<u32>(v)&0x7f800000u)==0x7f800000u;
            if(v==0. && all(c<dims-1u)) {
                var zero=true;
                for(var corner=1u;corner<8u;corner++) {
                    let q=c+vec3<u32>(corner&1u,(corner>>1u)&1u,(corner>>2u)&1u);
                    zero=zero && value(q.x+dims.x*(q.y+dims.y*q.z))==0.;
                }
                invalid=invalid || zero;
            }
            if(invalid) {bits|=7u;}
        }
    }
    atomicOr(&signs,bits);workgroupBarrier();
    if(lane==0u) {blocks[group.x+counts.x*(group.y+counts.y*group.z)]=atomicLoad(&signs);}
}
"#);
        let module=crate::timing::shader_module(&r.device,wgpu::ShaderModuleDescriptor {label:Some("implicit occupied blocks"),source:wgpu::ShaderSource::Wgsl(code.into())});
        let entries:Vec<_>=(0..=spans.len()).map(|i|pipelines::buffer_entry(i as u32,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:i<spans.len()})).collect();
        let layout=r.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("implicit occupied blocks"),entries:&entries});
        let pl=r.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {label:Some("implicit occupied blocks"),bind_group_layouts:&[Some(&layout)],immediate_size:0});
        let pipeline=r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {label:Some("implicit occupied blocks"),layout:Some(&pl),module:&module,entry_point:Some("main"),compilation_options:Default::default(),cache:r.cache.as_ref()});
        let entries:Vec<_>=spans.iter().map(|(b,_)|b).chain(std::iter::once(&buffer)).enumerate().map(|(i,b)|wgpu::BindGroupEntry {binding:i as u32,resource:b.as_entire_binding()}).collect();
        let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("implicit occupied blocks"),layout:&layout,entries:&entries});
        // Retain invalidity in the index. Republishing diagnostics after the frame
        // error buffer is reset must not require rereading unchanged scalar data.
        let source=format!(r#"
@group(0) @binding(0) var<storage,read> blocks:array<u32>;
@group(0) @binding(1) var<storage,read_write> errors:array<atomic<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>) {{
    if(id.x<arrayLength(&blocks) && (blocks[id.x]&4u)!=0u) {{
        atomicMax(&errors[0],{invalid}u);atomicMin(&errors[1],{object}u);
    }}
}}
"#,invalid=vork_slang::registry::IMPLICIT_INVALID);
        let module=crate::timing::shader_module(&r.device,wgpu::ShaderModuleDescriptor {label:Some("implicit validation publication"),source:wgpu::ShaderSource::Wgsl(source.into())});
        let diagnostics_layout=r.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("implicit validation publication"),entries:&[
            pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:true}),
            pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false}),
        ]});
        let layout=r.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {label:Some("implicit validation publication"),bind_group_layouts:&[Some(&diagnostics_layout)],immediate_size:0});
        let diagnostics_pipeline=r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {label:Some("implicit validation publication"),layout:Some(&layout),module:&module,entry_point:Some("main"),compilation_options:Default::default(),cache:r.cache.as_ref()});
        Ok(Self {buffer,pipeline,bind,blocks,diagnostics_pipeline,diagnostics_layout, #[cfg(test)] builds:std::sync::atomic::AtomicUsize::new(0)})
    }
    pub fn publish_diagnostics(&self,r:&Renderer,encoder:&mut wgpu::CommandEncoder,errors:&wgpu::Buffer) {
        let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("implicit validation publication"),layout:&self.diagnostics_layout,entries:&[
            wgpu::BindGroupEntry {binding:0,resource:self.buffer.as_entire_binding()},
            wgpu::BindGroupEntry {binding:1,resource:errors.as_entire_binding()},
        ]});
        let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("implicit validation publication"),timestamp_writes:None});
        pass.set_pipeline(&self.diagnostics_pipeline);pass.set_bind_group(0,&bind,&[]);
        pass.dispatch_workgroups(self.blocks.into_iter().product::<u32>().div_ceil(64),1,1);
    }
    // Called under the installed submission lock. The shared change tracker,
    // rather than sample-value reuse or a frame slot, owns publication history.
    pub fn publish(&self,encoder:&mut wgpu::CommandEncoder,changes:&mut values::ChangeTracker,revision:u64)->bool {
        if !changes.changed(values::ChangeKey::FieldIndex(self.buffer.clone()),&revision.to_le_bytes()) {return false;}
        self.encode(encoder);
        true
    }
    pub fn encode(&self,encoder:&mut wgpu::CommandEncoder) {
        #[cfg(test)] self.builds.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
        let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("implicit occupied blocks"),timestamp_writes:None});
        pass.set_pipeline(&self.pipeline);pass.set_bind_group(0,&self.bind,&[]);
        pass.dispatch_workgroups(self.blocks[0],self.blocks[1],self.blocks[2]);
    }
}
