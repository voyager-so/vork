//! GPU-owned radiance, proposal and moments. Installation reads metadata once; sampled
//! filters and frame uniforms consume the GPU statistics through ordered copies.
use super::*;
// Packed vector layout shared with environment_evaluated.wgsl.
#[repr(C)]
#[derive(Clone,Copy,Pod,Zeroable)]
struct Summary {scale:f32,total:f32,invalid:f32,inverse:f32}
#[repr(C)]
#[derive(Clone,Copy,Pod,Zeroable)]
struct Statistics {
    sh:[[f32;4];4],summary:Summary,mean:[f32;4],
    // Exact lighting RGB bits, plus 1.0f32 bits when every texel matches.
    uniform:[u32;4],
}
pub(super) struct Evaluation {
    source:wgpu::Texture,
    uniform:std::sync::OnceLock<Option<[u32;3]>>,
    _proposal:Option<wgpu::Texture>,
    reduce:Option<wgpu::BindGroup>,
    bind:wgpu::BindGroup,
    pipelines:Vec<wgpu::ComputePipeline>,
    statistics:wgpu::Buffer,
    read:wgpu::Buffer,
    validation:std::sync::OnceLock<wgpu::ComputePipeline>,
    pub distribution:Option<wgpu::Buffer>,
    pub size:[u32;2],
}
impl Evaluation {
    pub fn new(r:&Renderer,source:&wgpu::Texture)->Self {
        Self::build(r,source,true)
    }
    /// Warm static raster filters need source validation and scale, not a CDF.
    pub fn validation(r:&Renderer,source:&wgpu::Texture)->Self {Self::build(r,source,false)}
    fn build(r:&Renderer,source:&wgpu::Texture,proposal_needed:bool)->Self {
        let d=&r.device;
        let scale=source.width().max(source.height()).div_ceil(constants::ENVIRONMENT_PROPOSAL_MAX);
        let size=if proposal_needed {[source.width().div_ceil(scale),source.height().div_ceil(scale)]} else {[1,1]};
        let proposal=proposal_needed.then(||crate::gpu_memory::texture(d,&wgpu::TextureDescriptor {label:Some("GPU environment proposal"),size:wgpu::Extent3d {width:size[0],height:size[1],depth_or_array_layers:1},mip_level_count:1,sample_count:1,dimension:wgpu::TextureDimension::D2,format:wgpu::TextureFormat::Rgba32Float,usage:wgpu::TextureUsages::STORAGE_BINDING|wgpu::TextureUsages::TEXTURE_BINDING,view_formats:&[]}));
        let buffer=|label,size,usage|crate::gpu_memory::buffer(d,&wgpu::BufferDescriptor {label:Some(label),size,usage,mapped_at_creation:false});
        let distribution=proposal_needed.then(||buffer("GPU environment CDF",size[0] as u64*size[1] as u64*8,wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC));
        let rows=buffer("GPU environment row reductions",source.height().max(size[1]) as u64*80,wgpu::BufferUsages::STORAGE);
        let metadata_size=std::mem::size_of::<Statistics>() as u64;
        let statistics=buffer("GPU environment moments",metadata_size,wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC);
        let read=buffer("environment metadata readback",metadata_size,wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST);
        let source_view=source.create_view(&Default::default());
        let proposal_view=proposal.as_ref().map(|p|p.create_view(&Default::default()));
        let reduce=proposal_view.as_ref().map(|proposal_view|d.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("GPU environment area reduction"),layout:&reduce_pipeline(r).get_bind_group_layout(0),entries:&[entry(0,wgpu::BindingResource::TextureView(&source_view)),entry(1,wgpu::BindingResource::TextureView(proposal_view))]}));
        let mut entries=vec![REDUCE_LAYOUT[0]];
        if proposal_needed {entries.push(wgpu::BindGroupLayoutEntry {binding:1,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Texture {sample_type:wgpu::TextureSampleType::Float {filterable:false},view_dimension:wgpu::TextureViewDimension::D2,multisampled:false},count:None});}
        for binding in [2,3,4] {if binding!=3 || proposal_needed {entries.push(pipelines::buffer_entry(binding,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false}));}}
        let layout=d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("GPU environment statistics"),entries:&entries});
        let mut entries=vec![entry(0,wgpu::BindingResource::TextureView(&source_view)),entry(2,rows.as_entire_binding()),entry(4,statistics.as_entire_binding())];
        if let (Some(proposal),Some(distribution))=(&proposal_view,&distribution) {entries.push(entry(1,wgpu::BindingResource::TextureView(proposal)));entries.push(entry(3,distribution.as_entire_binding()));}
        let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&layout,entries:&entries});
        let shader=include_str!("../../shaders/environment_evaluated.wgsl")
            .replace("$STATISTICS_VECTORS",&(std::mem::size_of::<Statistics>()/16).to_string())
            .replace("$PI",&format!("{:.9e}",vork_slang::registry::PI))
            .replace("$HALF_SCALE",&format!("{:.9e}",constants::HALF_FLOAT_MAX*0.5))
            .replace("$LUMINANCE",&constants::LUMINANCE_REC709.map(|v|format!("{v:.9e}")).join(","))
            .replace("$DERING",&format!("{:.9e}",1.+0.05/(constants::SH_FIRST_ORDER*(2./3.))))
            .replace("$EPSILON",&format!("{:.9e}",constants::POSITIVE_EPSILON_F64));
        let module=crate::timing::shader_module(d,wgpu::ShaderModuleDescriptor {label:Some("GPU environment evaluated state"),source:wgpu::ShaderSource::Wgsl(shader.into())});
        let layout=d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {label:None,bind_group_layouts:&[Some(&layout)],immediate_size:0});
        let entries=if proposal_needed {&["peaks","peak_summary","moments_and_rows","summarize","normalize_rows"][..]} else {&["peaks","peak_summary"][..]};
        let pipelines=entries.iter().map(|name|r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {label:Some(name),layout:Some(&layout),module:&module,entry_point:Some(*name),compilation_options:Default::default(),cache:r.cache.as_ref()})).collect();
        Self {uniform:std::sync::OnceLock::new(),source:source.clone(),_proposal:proposal,reduce,bind,pipelines,statistics,read,distribution,size,validation:Default::default()}
    }
    #[cfg(test)]
    pub fn work(&self)->(bool,u64,usize) {
        (self._proposal.is_some(),self.distribution.as_ref().map_or(0,wgpu::Buffer::size),self.pipelines.len()+usize::from(self.reduce.is_some()))
    }
    pub fn encode(&self,r:&Renderer,encoder:&mut wgpu::CommandEncoder) {
        if let Some(reduce)=&self.reduce {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(reduce_pipeline(r));pass.set_bind_group(0,reduce,&[]);pass.dispatch_workgroups(self.size[0].div_ceil(8),self.size[1].div_ceil(8),1);}
        for (index,pipeline) in self.pipelines.iter().enumerate() {
            let count=if index==1 || index==3 {1} else {self.source.height().max(self.size[1]).div_ceil(64)};
            let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(pipeline);pass.set_bind_group(0,&self.bind,&[]);pass.dispatch_workgroups(count,1,1);
        }
    }
    pub fn publish_filter(&self,encoder:&mut wgpu::CommandEncoder,params:&wgpu::Buffer) {
        // The filter ABI is [roughness,mode,inverse_scale,_], four normalized
        // SH vectors, then proposal dimensions. Only scale and moments vary.
        encoder.copy_buffer_to_buffer(&self.statistics,(std::mem::offset_of!(Statistics,summary)+std::mem::offset_of!(Summary,inverse)) as u64,params,2*std::mem::size_of::<f32>() as u64,std::mem::size_of::<f32>() as u64);
        encoder.copy_buffer_to_buffer(&self.statistics,std::mem::offset_of!(Statistics,sh) as u64,params,std::mem::size_of::<[f32;4]>() as u64,std::mem::size_of::<[[f32;4];4]>() as u64);
    }
    pub fn publish_frame(&self,encoder:&mut wgpu::CommandEncoder,uniform:&wgpu::Buffer) {
        encoder.copy_buffer_to_buffer(&self.statistics,std::mem::offset_of!(Statistics,summary) as u64,uniform,(std::mem::offset_of!(Uniform,environment)+std::mem::size_of::<f32>()) as u64,std::mem::size_of::<f32>() as u64);
        encoder.copy_buffer_to_buffer(&self.statistics,std::mem::offset_of!(Statistics,mean) as u64,uniform,std::mem::offset_of!(Uniform,environment_mean) as u64,std::mem::size_of::<[f32;4]>() as u64);
    }
    /// Exact source bits, proved equal at every lighting texel by the existing
    /// statistics pass. Compaction never changes the proposal or its dimensions.
    pub fn uniform(&self)->Option<[u32;3]> {self.uniform.get().copied().flatten()}
    pub fn diagnostics(&self,r:&Renderer)->Validation {Validation::new(r,&self.statistics,self.validation.get_or_init(||Validation::pipeline(r)).clone())}
    pub fn update(&self,r:&Renderer)->Result<([glam::DVec3;4],f32)> {
        let mut encoder=r.device.create_command_encoder(&Default::default());
        self.encode(r,&mut encoder);
        self.read_metadata(r,encoder)
    }
    pub fn read_metadata(&self,r:&Renderer,mut encoder:wgpu::CommandEncoder)->Result<([glam::DVec3;4],f32)> {
        encoder.copy_buffer_to_buffer(&self.statistics,0,&self.read,0,std::mem::size_of::<Statistics>() as u64);r.queue.submit([encoder.finish()]);
        let(tx,rx)=std::sync::mpsc::channel();self.read.map_async(wgpu::MapMode::Read,..,move |v|{let _=tx.send(v);});r.wait(None)?;rx.recv()??;
        let mapped=self.read.get_mapped_range(..);let data:&Statistics=bytemuck::from_bytes(&mapped);
        let scale=data.summary.scale;let sh=std::array::from_fn(|i|glam::DVec3::new(data.sh[i][0] as f64,data.sh[i][1] as f64,data.sh[i][2] as f64)*scale as f64);
        let valid=data.summary.invalid==0. && scale.is_finite() && sh.iter().all(|v|v.is_finite());
        let uniform=(data.uniform[3]==1f32.to_bits()).then(||data.uniform[..3].try_into().unwrap());
        drop(mapped);self.read.unmap();
        // What the device recorded outranks what its readback says about the scene.
        r.errors.check()?;
        ensure!(valid,"environment radiance must be finite and nonnegative");
        let _=self.uniform.set(uniform);
        Ok((sh,scale))
    }
}

/// Each exposure accumulates its own failures even when a later sample is valid.
pub(crate) struct Validation {
    failed:wgpu::Buffer,
    pub readback:wgpu::Buffer,
    bind:wgpu::BindGroup,
    pipeline:wgpu::ComputePipeline,
}
impl Validation {
    fn new(r:&Renderer,statistics:&wgpu::Buffer,pipeline:wgpu::ComputePipeline)->Self {
        let d=&r.device;
        let buffer=|label,usage|crate::gpu_memory::buffer(d,&wgpu::BufferDescriptor {label:Some(label),size:8,usage,mapped_at_creation:false});
        let failed=buffer("environment validation",wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::COPY_SRC);
        let readback=buffer("environment validation readback",wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST);
        let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,statistics.as_entire_binding()),entry(1,failed.as_entire_binding())]});
        Self {failed,readback,bind,pipeline}
    }
    fn pipeline(r:&Renderer)->wgpu::ComputePipeline {
        let d=&r.device;
        let module=crate::timing::shader_module(d,wgpu::ShaderModuleDescriptor {label:Some("environment acceptance diagnostics"),source:wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage,read> statistics:array<vec4<f32>>; @group(0) @binding(1) var<storage,read_write> failed:array<atomic<u32>>; @compute @workgroup_size(1) fn main(){if statistics[4].z!=0. {atomicOr(&failed[0],1u);}}".into())});
        let layout=d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:None,entries:&[pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:true}),pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false})]});
        let layout=d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {label:None,bind_group_layouts:&[Some(&layout)],immediate_size:0});
        r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {label:Some("environment validation"),layout:Some(&layout),module:&module,entry_point:Some("main"),compilation_options:Default::default(),cache:r.cache.as_ref()})
    }
    pub fn clear(&self,queue:&wgpu::Queue) {queue.write_buffer(&self.failed,0,&[0;8]);}
    pub fn encode(&self,encoder:&mut wgpu::CommandEncoder) {
        let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&self.pipeline);pass.set_bind_group(0,&self.bind,&[]);pass.dispatch_workgroups(1,1,1);
    }
    pub fn copy(&self,encoder:&mut wgpu::CommandEncoder) {encoder.copy_buffer_to_buffer(&self.failed,0,&self.readback,0,8);}
}
