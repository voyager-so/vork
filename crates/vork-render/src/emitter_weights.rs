//! Frame-private current-area/emission proposal; geometry stays in the shared hit table.
use crate::*;

pub(crate) fn entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let stages=wgpu::ShaderStages::COMPUTE;
    let mut entries=pipelines::global_entries();
    entries.extend([
        pipelines::buffer_entry(11,stages,wgpu::BufferBindingType::Storage {read_only:true}),
        pipelines::buffer_entry(12,stages,wgpu::BufferBindingType::Storage {read_only:false}),
        pipelines::buffer_entry(13,stages,wgpu::BufferBindingType::Uniform),
        pipelines::buffer_entry(14,stages,wgpu::BufferBindingType::Storage {read_only:true}),
    ]);
    for e in &mut entries {e.visibility=stages;}
    entries
}
pub(crate) struct Plan {
    layout: wgpu::BindGroupLayout,
    areas: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    base: u32,
    leaves: u32,
    bytes: u64,
}
impl Plan {
    pub(crate) fn new(r:&Renderer,base:u32,count:u32)->Result<Option<Self>> {
        if count==0 {return Ok(None);}
        let leaves=count.checked_next_power_of_two().context("too many mesh-source triangles")?;
        let bytes=u64::from(leaves)*8;
        let limits=r.device.limits();
        ensure!(bytes<=limits.max_buffer_size && bytes<=u64::from(limits.max_storage_buffer_binding_size),
            crate::gpu_memory::device_limit("mesh-source area tree", format_args!("{bytes} bytes"), format_args!("storage binding={} bytes, buffer={} bytes",limits.max_storage_buffer_binding_size,limits.max_buffer_size), "reduce emissive mesh geometry"));
        let d=&r.device;
        let layout=d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("mesh-source areas"),entries:&entries()});
        let hits=rt::hits_layout(d);
        let pipeline_layout=d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {label:Some("mesh-source areas"),
            bind_group_layouts:&[Some(&layout),None,None,Some(&hits)],immediate_size:0});
        let make=|compiled:&native::Compiled,entry| {
            let module=native::shader_module(d,compiled);
            r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {label:Some("mesh-source areas"),layout:Some(&pipeline_layout),
                module:&module,entry_point:Some(entry),compilation_options:Default::default(),cache:r.cache.as_ref()})
        };
        Ok(Some(Self {layout,areas:make(&native::EMITTER_WEIGHTS_AREAS,"areas"),reduce:make(&native::EMITTER_WEIGHTS_REDUCE,"reduce"),base,leaves,bytes}))
    }
}
struct Step {bind:wgpu::BindGroup,dispatch:[u32;3]}
pub(crate) struct Slot {
    pub(crate) buffer:wgpu::Buffer,
    steps:Vec<Step>,
    hits:Option<wgpu::BindGroup>,
}
impl Slot {
    pub(crate) fn new(plan:&Plan,d:&wgpu::Device)->Self {
        Self {buffer:crate::gpu_memory::buffer(d, &wgpu::BufferDescriptor {label:Some("current mesh-source areas"),size:plan.bytes,
            usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false}),steps:Vec::new(),hits:None}
    }
    pub(crate) fn prepare(&mut self,plan:&Plan,s:&InstalledScene,d:&wgpu::Device,lighting:&crate::view::ViewLighting,
        objects:&wgpu::Buffer,materials:&wgpu::Buffer,hits:wgpu::BindGroup) {
        self.steps.clear();
        let max=d.limits().max_compute_workgroups_per_dimension;
        let mut first=plan.leaves;
        loop {
            let groups=first.div_ceil(vork_slang::registry::EMITTER_WEIGHT_GROUP);let width=groups.min(max);let height=groups.div_ceil(width);
            // The storage admission bounds the tree well below a two-dimensional dispatch's capacity.
            assert!(height<=max,"mesh-source tree dispatch exceeds device dimensions");
            let config=init(d,"mesh-source area level",bytemuck::cast_slice(&[plan.base,first,first,width,crate::materials::value_word(vork_scene::v2::OPENPBR.iter().position(|p|p.name=="emission_color").unwrap()) as u32,crate::materials::value_word(vork_scene::v2::OPENPBR.iter().position(|p|p.name=="emission_luminance").unwrap()) as u32,0,0]),wgpu::BufferUsages::UNIFORM);
            let mut entries=pipelines::ray_globals(s,&lighting.uniform,objects,&lighting.lights,&lighting.cluster_config,&lighting.cluster_lights, &lighting.analytic_light_tree);
            entries.extend([entry(11,s.rays.as_ref().unwrap().alpha.as_entire_binding()),entry(12,self.buffer.as_entire_binding()),entry(13,config.as_entire_binding()),entry(14,materials.as_entire_binding())]);
            self.steps.push(Step {bind:d.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("mesh-source area level"),layout:&plan.layout,entries:&entries}),dispatch:[width,height,1]});
            if first==1 {break;} first/=2;
        }
        self.hits=Some(hits);
    }
    pub(crate) fn encode(&self,plan:&Plan,encoder:&mut wgpu::CommandEncoder) {
        for (i,step) in self.steps.iter().enumerate() {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("current mesh-source areas"),..Default::default()});
            pass.set_pipeline(if i==0 {&plan.areas} else {&plan.reduce});
            pass.set_bind_group(0,&step.bind,&[]);pass.set_bind_group(3,self.hits.as_ref().unwrap(),&[]);
            pass.dispatch_workgroups(step.dispatch[0],step.dispatch[1],1);
        }
    }
}
#[cfg(test)]
#[test]
fn native_layout() {
    let hits=[rt::hit_buffers_entry(0,wgpu::ShaderStages::COMPUTE)];
    for shader in [&native::EMITTER_WEIGHTS_AREAS,&native::EMITTER_WEIGHTS_REDUCE] {native::tests::check(shader,&[&entries(),&[],&[],&hits]);}
}

#[cfg(test)]
#[path="emitter_weights_tests.rs"]
mod tests;
