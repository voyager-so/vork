//! AOV resolve uses the same sample weights and retained planar output ring as colour.
use crate::*;

pub(crate) struct Resolve {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    size: [u32;2],
}
impl Resolve {
    pub(crate) fn new(s:&InstalledScene,r:&Renderer,uniform:&wgpu::Buffer,values:&wgpu::Buffer,pixels:&wgpu::Buffer,size:[u32;2],internal:[u32;2],base:u32)->Result<Option<Self>> {
        if s.render.aovs.is_empty() {return Ok(None);}
        let metal=r.adapter_info.backend==wgpu::Backend::Metal;
        // Slang renames `main` for Metal; retain an explicit native entry name.
        let entry_point="aov_accumulate";
        // One shader per backend target: the output and internal sizes, the AOV count and the
        // first AOV plane are data (`resolve`), so no render size or readback layout asks Slang
        // for another.
        let source=format!("import vork_prelude;\n\
[[vk::binding(0,0)]] ConstantBuffer<Frame> frame : register(b0);\n\
[[vk::binding(1,0)]] RWStructuredBuffer<uint> pixels : register(u1);\n\
[[vk::binding(2,0)]] StructuredBuffer<float4> aovs : register(t2);\n\
struct AovResolve {{ uint2 size; uint2 inner; uint count; uint plane; uint2 reserved; }};\n\
[[vk::binding(3,0)]] ConstantBuffer<AovResolve> resolve : register(b3);\n\
[shader(\"compute\")][numthreads(8,8,1)] void {entry_point}(uint3 id:SV_DispatchThreadID) {{\n\
uint2 size=resolve.size;uint2 inner=resolve.inner;uint count=resolve.count;\n\
if(id.x>=size.x || id.y>=size.y) return;uint pixel=id.y*size.x+id.x;float weight=1./frame.temporal.y;\n\
uint2 p=min(uint2((float2(id.xy)+.5)*float2(inner)/float2(size)),inner-1u);\n\
for(uint i=0u;i<count;i++){{float4 v=aovs[(p.y*inner.x+p.x)*count+i];for(uint c=0u;c<4u;c++){{uint at=(resolve.plane+i*4u+c)*(size.x*size.y)+pixel;float old=frame.temporal.x==0. ? 0. : asfloat(pixels[at]);pixels[at]=asuint(old+v[c]*weight);}}}}\n\
}}\n");
        let [iw,ih]=internal;
        let [w,h]=size;
        // AovResolve: size, inner, count, plane, reserved.
        let values_of=[w,h,iw,ih,s.render.aovs.len() as u32,base,0,0];
        let parameters=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor{label:Some("AOV resolve parameters"),size:std::mem::size_of_val(&values_of) as u64,usage:wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        r.queue.write_buffer(&parameters,0,bytemuck::cast_slice(&values_of));
        let (_,target,flags)=if metal {crate::forward_variant::METAL_TARGET} else {crate::forward_variant::SPIRV_TARGET};
        // The stored resolve shader is the selected compiler's output: admit it before reading one.
        let compiler=vork_slang::Compiler::discover();
        crate::programs::admit(&|key|r.disk_cache.path("aov-resolve",1,key),&compiler)?;
        let path=r.disk_cache.path("aov-resolve",1,blake3::hash(format!("{target}{source}").as_bytes()));
        let code=if let Some(code)=crate::disk_cache::read(&path) {code} else {
            let mut modules=crate::native::SOURCES.to_vec();modules.push(("aov_resolve",&source));
            let args:Vec<_>=["-target",target].into_iter().chain(crate::forward_variant::SLANG_FLAGS.iter().copied()).chain(flags.iter().copied()).collect();
            let code=compiler.compile_modules(&modules,"aov_resolve",&[(entry_point,"compute")],&args)?;
            let _=crate::disk_cache::write(&path,&code);code
        };
        let shader=if metal {crate::native::module(&r.device,"AOV resolve",None,Some(&String::from_utf8(code)?),(8,8,1))} else {crate::native::module(&r.device,"AOV resolve",Some(&code),None,(8,8,1))};
        let entries=[pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Uniform),pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage{read_only:false}),pipelines::buffer_entry(2,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage{read_only:true}),pipelines::buffer_entry(3,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Uniform)];
        let group=r.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{label:Some("AOV resolve"),entries:&entries});
        let layout=r.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{label:Some("AOV resolve"),bind_group_layouts:&[Some(&group)],immediate_size:0});
        let pipeline=r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:Some("AOV resolve"),layout:Some(&layout),module:&shader,entry_point:Some(entry_point),compilation_options:Default::default(),cache:r.cache.as_ref()});
        let entries=[entry(0,uniform.as_entire_binding()),entry(1,pixels.as_entire_binding()),entry(2,values.as_entire_binding()),entry(3,parameters.as_entire_binding())];
        let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor{label:Some("AOV resolve"),layout:&group,entries:&entries});
        Ok(Some(Self{pipeline,bind,size}))
    }
    pub(crate) fn encode(&self,encoder:&mut wgpu::CommandEncoder) {
        let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{label:Some("AOV sample accumulation"),timestamp_writes:None});
        pass.set_pipeline(&self.pipeline);pass.set_bind_group(0,&self.bind,&[]);pass.dispatch_workgroups(self.size[0].div_ceil(8),self.size[1].div_ceil(8),1);
    }
}

/// Admit the scene's interleaved float4 AOV storage before compiling its shaders.
/// Resolved values and packed output planes both use the authored resolution.
pub(crate) fn check_storage(limits: &wgpu::Limits, render: &vork_scene::v2::Render) -> Result<()> {
    let count = render.aovs.len() as u64;
    if count == 0 { return Ok(()); }
    for (name, [w, h]) in [
        ("AOV resolved", render.resolution),
        ("AOV output planes", render.resolution),
    ] {
        // The packed frame also retains its built-in planes. Float readback
        // checks its larger prefix when the caller selects that view precision.
        let prefix = if name == "AOV output planes" {u64::from(crate::constants::PACKED_OUTPUT_PLANES)*4} else {0};
        let bytes = count.checked_mul(16).and_then(|n| n.checked_add(prefix)).and_then(|n| n.checked_mul(u64::from(w))).and_then(|n| n.checked_mul(u64::from(h)));
        anyhow::ensure!(bytes.is_some_and(|bytes| bytes <= limits.max_storage_buffer_binding_size && bytes <= limits.max_buffer_size),
            crate::gpu_memory::device_limit(format_args!("render.aovs: {count} declared; {name}"), format_args!("{} bytes at {w}x{h}",bytes.map_or_else(|| "more than u64::MAX".to_owned(), |n| n.to_string())), format_args!("max_storage_buffer_binding_size={} bytes, max_buffer_size is {} bytes",limits.max_storage_buffer_binding_size,limits.max_buffer_size), "reduce resolution or the number of render.aovs"));
    }
    Ok(())
}

#[cfg(all(test, vork_private_tests))]
include!("aov_tests_private.rs");
