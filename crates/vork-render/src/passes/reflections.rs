//! Full-resolution reflections through the installed TLAS.
use crate::*;
#[cfg(test)]
use crate::constants::{VISIBILITY_KEY_WORDS,VISIBILITY_SLOTS,VISIBILITY_REFLECTION_WORDS,VISIBILITY_REFLECTION_GUIDE_WORDS};
/// One binding per packed word: every field retains the original slot index.
pub(crate) fn record_fields_entry(binding:u32, read_only:bool)->wgpu::BindGroupLayoutEntry {
    let mut entry=pipelines::buffer_entry(binding,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only});
    entry.count=std::num::NonZeroU32::new(constants::VISIBILITY_REFLECTION_WORDS);
    entry
}
/// One binding per record field. Fields are packed `per_buffer` at a time, so the
/// complete array may exceed the device's single-allocation limit.
pub(crate) fn record_fields(buffers:&[wgpu::Buffer], bytes:u64, per_buffer:u32)->Vec<wgpu::BufferBinding<'_>> {
    let per_buffer=per_buffer.max(1);
    (0..constants::VISIBILITY_REFLECTION_WORDS).map(|field|{
        let chunk=(field/per_buffer) as usize;
        let offset=u64::from(field%per_buffer)*bytes;
        wgpu::BufferBinding {buffer:&buffers[chunk],offset,size:std::num::NonZeroU64::new(bytes)}
    }).collect()
}
/// One retained hit-shading specialization for the complete installed scene.
pub(crate) fn pipeline(r: &Renderer, cached: &std::sync::Arc<crate::pipelines::CachedPipelines>, features: crate::forward_variant::FeatureMask, probes: bool, mode: &crate::pipelines::VariantCompile) -> Result<crate::pipelines::Drawn<wgpu::ComputePipeline>> {
    let layout=crate::rt::reflections_hits_layout(&r.device,&cached.mat_layout,probes);
    cached.install_hit(r, crate::pipelines::ProgramKernel::Reflections { features }, mode, layout, "hardware reflections")
}

pub(crate) struct Resources {
    pub(crate) output: wgpu::TextureView,
}
impl Resources {
    pub(crate) fn new(
        d: &wgpu::Device,
        s: &InstalledScene,
        w: u32,
        h: u32,
        decoupled: Option<&crate::passes::decoupled::View>,
        ambient: &wgpu::TextureView,
        color: &wgpu::TextureView,
        uniform: &wgpu::Buffer,
        lights: &wgpu::Buffer,
        objects: &wgpu::Buffer,
        cluster_config: &wgpu::Buffer,
        cluster_lights: &wgpu::Buffer,
        rays: Option<&crate::rt::FrameRays>,
        material: &wgpu::BindGroup,
        enabled: bool,
        recorder: &mut crate::graph::Recorder,
    ) -> Result<Self> {
        let enabled = enabled && s.plan.effects.reflections;
        let (rw,rh)=if enabled {(w,h)} else {(1,1)};
        let output=s.transients.texture(d,&wgpu::TextureDescriptor {
            label:Some("full-resolution reflection estimates"),size:wgpu::Extent3d {width:rw,height:rh,depth_or_array_layers:constants::REFLECTION_CLASSES},
            mip_level_count:1,sample_count:1,dimension:wgpu::TextureDimension::D2,format:wgpu::TextureFormat::Rgba16Float,
            usage:wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::STORAGE_BINDING,view_formats:&[],
        }).create_view(&wgpu::TextureViewDescriptor {dimension:Some(wgpu::TextureViewDimension::D2Array),..Default::default()});
        if !enabled {
            return Ok(Self { output });
        }
        let v=decoupled.context("reflections need decoupled receiver guides")?;
        let (slots,reflection_records,reflection_guides)=(&v.slots,&v.reflection_records,&v.reflection_guides);
        let record_fields=record_fields(reflection_records,v.reflection_field_bytes,v.reflection_fields_per_buffer);
        let batch=constants::REFLECTION_BATCH_SAMPLES;
        let samples=v.samples();
        let chunks=samples.div_ceil(batch);
        let packet_pixels=if v.packets {w*v.band_rows} else {0};
        let packet_count=if v.packets {v.bands} else {1};
        let mut owners=Vec::new();
        let mut ranges=Vec::<u32>::new();
        for packet in 0..packet_count {
            let first=if v.packets {packet*packet_pixels} else {0};
            let end=if v.packets {((packet+1)*packet_pixels).min(w*h)} else {w*h};
            let spans=(end-first).div_ceil(constants::REFLECTION_DISPATCH_PIXELS);
            for chunk in 0..chunks {
                for span in 0..spans {
                    let at=ranges.len();ranges.resize(at+64,0);
                    ranges[at]=chunk*batch;
                    ranges[at+1]=((chunk+1)*batch).min(samples);
                    ranges[at+2]=s.rays.as_ref().map_or(0,|r|r.emitters);
                    ranges[at+3]=first+span*constants::REFLECTION_DISPATCH_PIXELS | if crate::passes::decoupled::direct_visibility(samples,s.plan.transparent,s.plan.effects.subsurface) {0} else {constants::REFLECTION_MERGED_SLOTS};
                    ranges[at+4]=packet_pixels;ranges[at+5]=end;ranges[at+6]=first;
                    owners.push(packet as usize);
                }
            }
        }
        let ranges=crate::init(d,"reflection sample ranges",bytemuck::cast_slice(&ranges),wgpu::BufferUsages::UNIFORM);
        let range=|item|wgpu::BindingResource::Buffer(wgpu::BufferBinding {buffer:&ranges,offset:item as u64*256,size:std::num::NonZeroU64::new(32)});
        let accumulation=s.transients.buffer(d,"reflection accumulation",u64::from(w)*u64::from(h)*u64::from(constants::REFLECTION_CLASSES)*16,wgpu::BufferUsages::STORAGE);

        let mut dispatches=Vec::new();
        let (receivers,normals)=(&v.receivers,&v.normals);
        let frame_rays = rays.context("ray reflections need the scene TLAS")?;
        let tlas = &frame_rays.tlas;
        let alpha = &s
            .rays
            .as_ref()
            .context("ray reflections need the scene TLAS")?
            .alpha;
        let pipeline = s.reflections.as_ref().map(|pipeline|pipeline.current(false))
            .context("incomplete installed ray reflection plan")?;
        let geometry=frame_rays.hit_bindings();
        let field_entries=vec![entry(0,wgpu::BindingResource::BufferArray(&geometry)),entry(1,wgpu::BindingResource::BufferArray(&record_fields))];
        let field_bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("reflection fields, geometry and medium"),layout:&pipeline.get_bind_group_layout(3),entries:&field_entries});
        // rt::reflections_hits_entries: the scene globals, then the trace's images and rays.
        for item in 0..owners.len() {
        let entries = vec![
            entry(0, uniform.as_entire_binding()),
            entry(1, objects.as_entire_binding()),
            entry(2, lights.as_entire_binding()),
            entry(3, wgpu::BindingResource::TextureView(&s.environment.specular)),
            entry(4, wgpu::BindingResource::Sampler(&s.environment.sampler)),
            entry(5, wgpu::BindingResource::TextureView(&s.environment.diffuse)),
            entry(6, wgpu::BindingResource::TextureView(&s.environment.brdf)),
            entry(7, cluster_config.as_entire_binding()),
            entry(8, cluster_lights.as_entire_binding()),
            entry(9, wgpu::BindingResource::TextureView(&s.environment.ltc)),
            entry(10, wgpu::BindingResource::TextureView(color)),
            entry(11, wgpu::BindingResource::TextureView(receivers)),
            entry(12, wgpu::BindingResource::TextureView(normals)),
            entry(13, wgpu::BindingResource::TextureView(ambient)),
            entry(14, wgpu::BindingResource::TextureView(&output)),
            entry(15, tlas.as_binding()),
            entry(16, alpha.as_entire_binding()),
            entry(17, s.sampling.buffer.as_entire_binding()),
            entry(18, slots.as_entire_binding()),
            entry(19,reflection_guides.as_entire_binding()),
            entry(20,range(item)),entry(21,accumulation.as_entire_binding()),
            entry(22,frame_rays.emitter_weights.as_ref().map_or(alpha,|w|&w.buffer).as_entire_binding()),
        ];
        let trace = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ray reflection resources"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let dispatch = crate::graph::Dispatch::new(pipeline, trace, [constants::REFLECTION_DISPATCH_PIXELS.div_ceil(crate::native::REFLECTIONS_HITS_COMPUTE.threads.0), 1, constants::REFLECTION_CLASSES])
            .with_pipeline(s.reflections.as_ref().unwrap().clone()).with_material(material);
        // Unseen hits read the irradiance volume instead of the world.
        let dispatch = dispatch.with_hits(field_bind.clone()).with_read_table(s.probes.as_ref().map(|p| p.read_bind(d)));
        // The ray-traced trace binds the material table for its alpha tests and hit surfaces.
        dispatches.push(dispatch);
        }
        if v.packets {
            let mut packets=vec![Vec::new();packet_count as usize];
            for (owner,dispatch) in owners.into_iter().zip(dispatches.drain(..)) {packets[owner].push(dispatch);}
            ensure!(v.packet_reflections.set(packets).is_ok(),"camera packet reflections installed twice");
        }
        recorder.compute(
            crate::graph::Pass::Reflections,
            "full-resolution ray-traced reflections",
            "reflections",
            false,
            dispatches,
        );
        recorder.separate_last_every(constants::REFLECTION_SEGMENT_DISPATCHES);
        Ok(Self { output })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compute_vec4(r: &Renderer, pipeline: &wgpu::ComputePipeline, bind: &wgpu::BindGroup,
        output: &wgpu::Buffer, groups: u32) -> Result<Vec<[f32;4]>> {
        compute_vec4_count(r,pipeline,bind,output,groups,groups)
    }

    fn compute_vec4_count(r: &Renderer, pipeline: &wgpu::ComputePipeline, bind: &wgpu::BindGroup,
        output: &wgpu::Buffer, groups: u32, values: u32) -> Result<Vec<[f32;4]>> {
        let size=u64::from(values)*std::mem::size_of::<[f32;4]>() as u64;
        let readback=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {label:None,size,
            usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let mut encoder=r.device.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(pipeline);pass.set_bind_group(0,bind,&[]);pass.dispatch_workgroups(groups,1,1);}
        encoder.copy_buffer_to_buffer(output,0,&readback,0,size);
        let submission=r.queue.submit([encoder.finish()]);
        let(tx,rx)=std::sync::mpsc::channel();readback.map_async(wgpu::MapMode::Read,..,move|v|{let _=tx.send(v);});
        r.device.poll(wgpu::PollType::Wait{submission_index:Some(submission),timeout:None})?;rx.recv()??;
        let values=bytemuck::cast_slice(&readback.get_mapped_range(..)).to_vec();
        readback.unmap();r.errors.check()?;Ok(values)
    }

    #[test]
    fn reflection_mixture_keeps_energy_with_one_query_per_sample() -> Result<()> {
        let Some(r)=crate::test_renderer()? else {return Ok(());};
        let source=r#"
import vork_prelude;
import reflection_receiver;
import reflection_bsdf;
import closure;
import vork_constants;
[[vk::binding(0,0)]] RWStructuredBuffer<float4> result;
[[vk::binding(1,0)]] RWStructuredBuffer<uint> calls;
struct MockTrace : IReflectionTrace {
    static float connection_probability() {return .5;}
    static float3 direct(ReflectionSite site,uint2 cell,uint index,float3 incident) {calls[0]++;return float3(.3);}
    static float3 trace(ReflectionSite site,uint2 cell,uint index,float3 incident,float3 ray,float pdf) {calls[0]++;return site.receiver.reflected;}
};
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 gid:SV_DispatchThreadID) {
    Closure c={};c.base_color=float3(1.);c.base_weight=1.;c.base_metalness=1.;c.specular_weight=1.;c.specular_color=float3(1.);c.specular_ior=1.5;
    c.geometry_normal=float3(0.,0.,1.);c.geometry_tangent=float3(1.,0.,0.);c.specular_roughness=.5;
    ReflectionSite site={};site.bsdf=reflection_bsdf(c,c,float3(1.),1.,1.,float3(0.));site.geometric=c.geometry_normal;
    const uint count=32768u;var reference=float3(0.);var actual=float3(0.);
    for(uint i=0u;i<count;i++) {
        let u=float2((float(i)+.5)/float(count),float(reversebits(i))*REG_RADICAL_INVERSE_SCALE);
        reference+=reflection_sample<MockTrace>(site,uint2(0u),i,float3(0.,0.,-1.),u,0.);
    }
    calls[0]=0u;
    for(uint i=0u;i<count;i++) {
        let u=float2((float(i)+.5)/float(count),float(reversebits(i))*REG_RADICAL_INVERSE_SCALE);
        actual+=reflection_sample<MockTrace>(site,uint2(0u),i,float3(0.,0.,-1.),u,.5);
    }
    result[0]=float4(actual.r/float(count),reference.r/float(count)+.3,float(calls[0]),float(count));
}
"#;
        let entries=[pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false}),
            pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false})];
        let pipeline=crate::medium::test_pipeline(&r,source,&crate::medium::Native::default(),&entries,(1,1,1))?;
        let calls=crate::init(&r.device,"reflection query count",bytemuck::cast_slice(&[0u32]),wgpu::BufferUsages::STORAGE);
        let output=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {label:None,size:16,mapped_at_creation:false,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC});
        let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[
            entry(0,output.as_entire_binding()),entry(1,calls.as_entire_binding())]});
        let value=compute_vec4(&r,&pipeline,&bind,&output,1)?[0];
        assert!(value[0]>0.5 && (value[0]-value[1]).abs()<0.01,"mixture changed reflected/source energy: {value:?}");
        assert!(value[2]<=value[3],"one camera sample spent multiple reflection queries: {value:?}");
        r.errors.check()?;Ok(())
    }

    #[test]
    fn decoupled_reflections_keep_lobes_with_covered_samples() -> Result<()> {
        let Some(r)=crate::test_renderer()? else {return Ok(());};
        let d=&r.device;
        let source=r#"
import vork_prelude;
import vork_sampling;
import reflection_receiver;
import vork_visibility;
import vork_constants;
[[vk::binding(0,0)]] StructuredBuffer<uint> slots : register(t0);
[[vk::binding(1,0)]] RWStructuredBuffer<float4> result : register(u1);
[[vk::binding(2,0)]] StructuredBuffer<uint> guides : register(t2);
#ifdef VORK_METAL
struct TestReflectionFields {StructuredBuffer<uint> buffers[REG_VISIBILITY_REFLECTION_WORDS];};
[[vk::binding(3,0)]] ParameterBlock<TestReflectionFields> record_fields : register(b3);
#define records record_fields.buffers
#else
[[vk::binding(3,0)]] StructuredBuffer<uint> record_fields[REG_VISIBILITY_REFLECTION_WORDS] : register(t3);
#define records record_fields
#endif
[[vk::binding(4,0)]] StructuredBuffer<uint> film : register(t4);
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    Frame f={};f.flags.x=1.;f.sampling.w=7u;f.post.z=1.;f.projection.xy=1.;
    f.vp=float4x4(1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.);f.inverse_vp=f.vp;f.view=f.vp;
    if (id.x==16u) {let selected=reflection_receiver(guides,uint2(2u),uint2(0u),0u);result[id.x]=float4(float2(selected.pixel),selected.source_tag,float(selected.eligible));return;}
    if (id.x>=14u) {
        let c=id.x-14u;var samples=ReflectionSlots(slots,uint2(2u),int2(0),f,c);
        ReflectionSite site;uint index;var correction=0.;
        while(samples.next(slots,records,film,f,site,index)) {correction+=select(site.world_specular,-1.,0.);}
        let coverage=reflection_guide(guides,uint2(2u),int2(0),c).coverage;
        result[id.x]=float4(correction/(float(f.sampling.w)*coverage),0.,0.,0.);return;
    }
    if (id.x==13u) {let g=reflection_guide(guides,uint2(2u),int2(0),1u);result[id.x]=float4(g.normal,g.coverage);return;}
    if (id.x>=10u) {
        let payload=float2(select(id.x==11u,0.,31.),select(id.x==12u,16.,15.)*32.);
        result[id.x]=float4(float(surface_reflection_traceable(payload)),0.,0.,0.);return;
    }
    if (id.x==9u) {
        let selected=reflection_receiver(guides,uint2(2u),uint2(0u),1u);
        result[id.x]=float4(float2(selected.pixel),selected.source_tag,float(selected.eligible));return;
    }
    if (id.x>=7u) {
        f.flags.x=0.;f.vp=float4x4(1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,-1.,-1.,0.,0.,-.1,0.);
        let pixel=select(id.x==7u,float2(.75,.25),float2(.25,.75));
        let camera=ReflectionCamera(f);let point=camera.point(pixel,uint2(1u),float3(1499.,0.,-750.5),1500.);
        result[id.x]=float4(0.,0.,-point.z,1.);return;
    }
    var iterator=ReflectionSlots(slots,uint2(2u),int2(0),f,select(id.x==5u,1u,0u));
    ReflectionSite receiver;uint index;ReflectionSite site={};
    while (iterator.next(slots,records,film,f,receiver,index)) {if (index==id.x) {site=receiver;}}
    result[id.x]=float4(site.receiver.roughness,site.receiver.reflected.r,site.position.z,float(any(site.receiver.reflected>float3(0.))));
}
"#;
        let entries=[pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage{read_only:true}),
            pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage{read_only:false}),
            pipelines::buffer_entry(2,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage{read_only:true}),
            record_fields_entry(3,true),
            pipelines::buffer_entry(4,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage{read_only:true})];
        let pipeline=crate::medium::test_pipeline(&r,source,&crate::medium::Native::default(),&entries,(1,1,1))?;
        let stride=(VISIBILITY_KEY_WORDS+1) as usize;
        let pixels=4usize;
        let keys=VISIBILITY_KEY_WORDS as usize;
        let mut words=vec![0u32;pixels*((VISIBILITY_SLOTS as usize+1)*stride+2)];
        let mut records=vec![0u32;pixels*(VISIBILITY_SLOTS as usize+1)*VISIBILITY_REFLECTION_WORDS as usize];
        let half2=|a,b|u32::from(half::f16::from_f32(a).to_bits())|(u32::from(half::f16::from_f32(b).to_bits())<<16);
        // Rough and smooth lobes, a neighboring pixel filling only this
        // pixel's uncovered sample, and one World-excluded overflow entry.
        let pool=pixels*(VISIBILITY_SLOTS as usize*stride+1);
        words[pixels*VISIBILITY_SLOTS as usize*stride]=1;
        for (record,at,mask,z,roughness) in [(0,0,7,0.5f32,992f32),(1,stride,8,0.75f32,0f32),
            (VISIBILITY_SLOTS as usize,VISIBILITY_SLOTS as usize*stride,127,0.5f32,0f32),
            (pixels*VISIBILITY_SLOTS as usize,pool+1,32,0.875f32,0f32)] {
            words[at]=1;words[at+keys]=mask;
            let base=record*VISIBILITY_REFLECTION_WORDS as usize;
            records[base+2]=1f32.to_bits();records[base+3]=z.to_bits();records[base+5]=half2(if at==pool+1 {-33.} else {31.},roughness);
        }
        // A near-tangent geometric normal must retain its nonzero Z. Half
        // octahedral packing rounded this plane to (1,0,0) and produced NaNs.
        words[2*stride]=1;words[2*stride+keys]=64;
        let grazing=2*VISIBILITY_REFLECTION_WORDS as usize;
        records[grazing]=1f32.to_bits();records[grazing+2]=1e-5f32.to_bits();
        records[grazing+3]=(-0.5f32-crate::sampling::offset(6,7)[0]+1e-5*0.625).to_bits();records[grazing+5]=half2(31.,0.);
        let classes=constants::REFLECTION_CLASSES as usize;
        let mut guide_words=vec![0u32;pixels*classes*VISIBILITY_REFLECTION_GUIDE_WORDS as usize];
        let guide_at=|pixel,c|(pixel*classes+c)*VISIBILITY_REFLECTION_GUIDE_WORDS as usize;
        for pixel in 0..pixels {for c in 0..classes {
            let at=guide_at(pixel,c);
            guide_words[at+3]=1f32.to_bits();
        }}
        for (pixel,c,depth,coverage) in [(0,0,0.25f32,5f32/7.),(0,1,0.875,1./7.),(1,0,0.5,1.),(1,1,0.1,0.)] {
            let at=guide_at(pixel,c);
            guide_words[at+3]=depth.to_bits();guide_words[at+5]=coverage.to_bits();
        }
        let guides=crate::init(d,"reflection guide fixture",bytemuck::cast_slice(&guide_words),wgpu::BufferUsages::STORAGE);
        let input=crate::init(d,"reflection slot fixture",bytemuck::cast_slice(&words),wgpu::BufferUsages::STORAGE);
        let count=records.len()/VISIBILITY_REFLECTION_WORDS as usize;
        let field_bytes=(count as u64*4).div_ceil(u64::from(d.limits().min_storage_buffer_offset_alignment))*u64::from(d.limits().min_storage_buffer_offset_alignment);
        let mut fields=vec![0u32;field_bytes as usize/4*VISIBILITY_REFLECTION_WORDS as usize];
        for record in 0..count {for word in 0..VISIBILITY_REFLECTION_WORDS as usize {fields[word*field_bytes as usize/4+record]=records[record*VISIBILITY_REFLECTION_WORDS as usize+word];}}
        let records=crate::init(d,"reflection record fixture",bytemuck::cast_slice(&fields),wgpu::BufferUsages::STORAGE);
        let record_fields=record_fields(std::slice::from_ref(&records),field_bytes,VISIBILITY_REFLECTION_WORDS);
        let film:Vec<u32>=(0..7).flat_map(|i|crate::sampling::offset(i,7).map(f32::to_bits)).collect();
        let film=crate::init(d,"reflection film fixture",bytemuck::cast_slice(&film),wgpu::BufferUsages::STORAGE);
        let groups=17u32;
        let output=crate::gpu_memory::buffer(d,&wgpu::BufferDescriptor {label:None,size:u64::from(groups)*16,
            usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
        let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[
            entry(0,input.as_entire_binding()),entry(1,output.as_entire_binding()),entry(2,guides.as_entire_binding()),entry(3,wgpu::BindingResource::BufferArray(&record_fields)),entry(4,film.as_entire_binding())]});
        let values=compute_vec4(&r,&pipeline,&bind,&output,groups)?;
        for value in &values[..3] {
            // Packed response reconstruction can round one ULP below unity on NVIDIA.
            assert!((value[0]-1.).abs()<1e-6);assert!((value[1]-1.).abs()<1e-6);assert_eq!(value[3],1.);
        }
        assert_eq!(values[3][0],0.);assert!((values[3][1]-1.).abs()<1e-6);assert_eq!(values[3][2],0.75);assert_eq!(values[3][3],1.);
        assert_eq!(values[4][3],0.,"uncovered sample traced an extended receiver plane");
        assert_eq!(values[5][0],0.);assert!((values[5][1]-1.).abs()<1e-6);
        assert_eq!([values[5][2],values[5][3]],[0.875,1.],"overflow reflection record was lost");
        assert!(values[6].iter().all(|x|x.is_finite()),"grazing receiver produced nonfinite data: {:?}",values[6]);
        assert_eq!(values[6][3],1.);assert!((0.5..0.75).contains(&values[6][2]));
        assert_eq!(values[7][2],1.);assert_eq!(values[8][2],1500.,"valid far sample was clamped to its near anchor");
        assert_eq!(values[9],[0.,0.,1.,1.],"zero-coverage foreground canceled the reflective pixel");
        assert_eq!(values[10][0],1.,"nonzero packed response must be traced and counted");
        assert_eq!(values[11][0],0.,"zero packed response must not contribute reflective coverage");
        assert_eq!(values[12][0],1.,"rough response remains traceable without a roughness cutoff");
        assert_eq!(values[13],[0.,0.,1.,1./7.],"World-excluded coverage was lost");
        assert_eq!(values[14][0],-1.,"World-enabled correction was diluted by an excluded slot");
        assert_eq!(values[15][0],0.,"World-excluded black hit subtracted World radiance");
        assert_eq!(values[16],[0.,0.,1.,1.],"reflection receiver borrowed a neighbouring pixel instead of retaining its own coverage");
        Ok(())
    }

    #[test]
    fn ggx_reflection_estimator_matches_hemisphere_integral() -> Result<()> {
        let Some(r)=crate::test_renderer()? else {return Ok(());};
        let source=r#"
import reflection_bsdf;
import reflection_receiver;
import bsdf_microfacet;
import bsdf_film;
import closure;
import openpbr_core;
import vork_constants;
import vork_sampling;
[[vk::binding(0,0)]] RWStructuredBuffer<float4> result;
float3 evaluate(Closure c,float3 view,float3 light) {
    let h=normalize(view+light);let vh=max(dot(view,h),0.);
    let fresnel_=reflection_fresnel(c.base_color,c.base_weight,c.base_metalness,c.specular_weight,c.specular_color,vh,c.specular_ior);
    return film_reflection(c,vh,fresnel_)*microfacet_distribution(c,h)*microfacet_visibility(c,view,light)*max(dot(c.geometry_normal,light),0.);
}
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    var c=(Closure)0;c.base_color=float3(.7,.4,.2);c.base_weight=1.;c.base_metalness=float(id.x%2u);
    c.specular_weight=1.;c.specular_color=float3(1.);c.specular_ior=1.5;
    c.geometry_normal=float3(0.,0.,1.);c.geometry_tangent=float3(1.,0.,0.);
    c.specular_roughness=id.x/2u==2u ? .5 : .2+.1*float(id.x/2u);
    c.geometry_coat_normal=c.geometry_normal;c.coat_roughness=.25;c.coat_ior=1.4;
    if(id.x==6u) {c.specular_roughness=.3;c.coat_weight=.7;c.specular_roughness_anisotropy=.5;}
    let view=normalize(float3(.95,0.,.3));let mirror=reflect(-view,c.geometry_normal);
    let state=reflection_bsdf(c,c,float3(.8),.9,.7,float3(0.));
    var estimate=float3(0.);var tail=0.;
    for(uint i=0u;i<65536u;i++) {
        float3 direction;
        direction=reflection_bsdf_sample(state,view,sample_2d(uint2(id.x,0u),i,65536u,REG_SAMPLING_STREAM_REFLECTION));
        float pdf;let response=reflection_bsdf_evaluate(state,view,direction,pdf);
        let weight=pdf>0. ? response/pdf : float3(0.);
        estimate+=weight/65536.;
        if(dot(direction,mirror)<rsqrt(1.+pow(c.specular_roughness,4.))) {tail+=1./65536.;}
    }
    var reference=float3(0.);
    for(uint z=0u;z<512u;z++) {for(uint p=0u;p<1024u;p++) {
        let cosine=(float(z)+.5)/512.;let phi=REG_TAU*(float(p)+.5)/1024.;
        let light=float3(sqrt(1.-cosine*cosine)*float2(cos(phi),sin(phi)),cosine);
        var value=evaluate(c,view,light)*.8*.9*.7;
        if(c.coat_weight>0.) {
            value*=coat_transmission(c.coat_weight,dot(c.geometry_coat_normal,light),c.coat_ior);
            var coat=(Closure)0;coat.geometry_normal=c.geometry_coat_normal;coat.specular_roughness=c.coat_roughness;
            coat.specular_ior=c.coat_ior;coat.specular_color=float3(1.);coat.specular_weight=c.coat_weight;
            value+=evaluate(coat,view,light)*.9*.7;
        }
        reference+=value*(REG_TAU/(512.*1024.));
    }}
    result[id.x]=float4(estimate.r,reference.r,tail,estimate.g-reference.g);
}
"#;
        let entries=[pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage{read_only:false})];
        let pipeline=crate::medium::test_pipeline(&r,source,&crate::medium::Native::default(),&entries,(1,1,1))?;
        let output=crate::init(&r.device,"GGX integral results",&[0u8;7*16],wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC);
        let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,output.as_entire_binding())]});
        let values=compute_vec4(&r,&pipeline,&bind,&output,7)?;
        for (case,v) in values.iter().enumerate() {
            assert!((v[0]-v[1]).abs()<0.002+v[1]*0.02,"GGX BSDF/PDF integral case {case}: {v:?}");
            assert!(v[3].abs()<0.004,"GGX green integral case {case}: {v:?}");
            assert!(v[2]>0.1,"bounded mirror disk lost GGX tail in case {case}: {v:?}");
        }
        Ok(())
    }



    /// A tiny bright source must be integrated by both proposals, without a
    /// radiance clamp or a persistent bias when its guide is much broader.
    #[test]
    fn sampled_hit_emitters_preserve_energy_and_partial_visibility() -> Result<()> {
        let Some(r)=crate::test_renderer()? else {return Ok(());};
        let source=r#"
import surface_lighting;
import subsurface_transport;
import closure;
import vork_prelude;
import reflection_bsdf;
import bsdf;
import emitter;
import variant;
import vork_sampling;
import vork_constants;
[[vk::binding(0,0)]] RWStructuredBuffer<float4> result;
// The path direct estimator integrates these same physical responses with
// emitter NEE. Independent area quadrature supplies its converged energy.
float response(uint test,Closure c,ReflectionBsdf state,float3 view,float3 direction) {
    if(test==5u) {return direct_brdf(c,view,direction,REG_RAY_DIFFUSE|REG_RAY_SPECULAR).r;}
    if(test==6u) {return subsurface_transmission(c,c.geometry_normal,view,direction,.2).r*max(-direction.z,0.)/REG_PI;}
    float pdf;return reflection_bsdf_evaluate(state,view,direction,pdf).r;
}
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    Closure c={};c.base_color=float3(.7);c.base_weight=1.;c.base_metalness=1.;
    c.specular_weight=1.;c.specular_color=float3(1.);c.specular_ior=1.5;
    c.geometry_normal=float3(0.,0.,1.);c.geometry_tangent=float3(1.,0.,0.);
    c.specular_roughness=id.x==0u ? .2 : .6;
    c.geometry_coat_normal=c.geometry_normal;c.coat_roughness=.2;c.coat_ior=1.4;
    if(id.x==2u) {c.coat_weight=.7;}
    if(id.x==5u || id.x==6u) {
        c.base_metalness=0.;c.specular_roughness=1.;c.subsurface_weight=1.;
        c.geometry_thin_walled=id.x==5u;c.subsurface_color=float3(.8);
        c.subsurface_radius=1.;c.subsurface_radius_scale=float3(1.);
    }
    // Each lobe sees directions below the other lobe's horizon.
    if(id.x==7u || id.x==8u) {
        c.coat_weight=.7;c.specular_roughness=.9;c.coat_roughness=.9;
        c.geometry_coat_normal=normalize(float3(.9,0.,.3));
        if(id.x==8u) {c.geometry_normal=normalize(float3(-.9,0.,.3));}
    }
    Layers l={};l.base=c;l.scale=float3(1.);l.throughput=1.;
    let state=reflection_bsdf(c,c,float3(1.),1.,1.,float3(0.));let view=float3(0.,0.,1.);
    LightValues light={};light.position=float4(0.,0.,2.,3.);
    light.direction=float4(0.,0.,-1.,0.);light.radiance=float4(1.);
    light.emitter_u=float4(2.,0.,0.,0.);light.emitter_v=float4(0.,2.,0.,0.);
    if(id.x==3u) {light.position.w=0.;light.params.y=.3;}
    if(id.x==4u) {light.position.w=1.;light.params.y=.5;}
    if(id.x==5u || id.x==6u) {light.position.z=-2.;light.direction.z=1.;}
    if(id.x==7u || id.x==8u) {light.position.xyz=float3(2.,0.,.2);light.direction.xyz=normalize(-light.position.xyz);
        light.emitter_u.xyz=float3(0.,2.,0.);light.emitter_v.xyz=cross(light.direction.xyz,float3(0.,2.,0.));}
    var reference=0.;
    for(uint y=0u;y<512u;y++) {for(uint x=0u;x<512u;x++) {
        let source=emitter_sample_li<Key>(light,0u,float3(0.),(float2(x,y)+.5)/512.);
        let f=response(id.x,c,state,view,source.direction);
        reference+=f*source.radiance.r/source.pdf/(512.*512.);
    }}
    var estimate=0.;var visible=0.;
    for(uint i=0u;i<65536u;i++) {
        let u=sample_2d(uint2(id.x,0u),i,65536u,REG_SAMPLING_STREAM_SECONDARY_EMITTER);
        let source=surface_light_sample<Key>(light,0u,float3(0.),l,c,view,false,REG_RAY_DIFFUSE|REG_RAY_SPECULAR,u,i);
        if(source.pdf<=0.) {continue;}
        let f=response(id.x,c,state,view,source.direction);
        let value=f*source.radiance.r/source.pdf/65536.;
        estimate+=value;
        // A straight blocker hides half of this symmetric finite emitter.
        // Visibility belongs to the sampled endpoint, never its centre.
        if(source.direction.x>0.) {visible+=value;}
    }
    result[id.x]=float4(estimate,reference,visible,0.);
}
"#;
        // direct_brdf's diffuse response reads the production dielectric-albedo
        // LUT. Native passthrough requires every reachable resource explicitly.
        let scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,"nodes":[]}))?;
        let environment=crate::environment::Environment::new(&r,
            crate::environment_node::Settings::from_scene(&scene)?,&scene,Path::new("."))?;
        let compute=wgpu::ShaderStages::COMPUTE;
        let mut entries=vec![pipelines::buffer_entry(0,compute,wgpu::BufferBindingType::Storage {read_only:false})];
        for binding in [1,2] {entries.push(pipelines::buffer_entry(binding,compute,wgpu::BufferBindingType::Storage {read_only:true}));}
        // Keep Vulkan bindings dense and Metal's three frame texture slots in
        // their production order: specular, diffuse, BRDF; sampler is slot zero.
        for binding in [3,5,6] {
            let mut entry=pipelines::texture_entry(binding,wgpu::TextureSampleType::Float {filterable:true},false);
            entry.visibility=compute;entries.push(entry);
        }
        let mut sampler=pipelines::sampler_entry(4,false);sampler.visibility=compute;entries.push(sampler);
        // No photometric profile is installed in these geometric emitter oracles.
        let source=crate::forward_variant::variant_module(0,false,false,false,None,source);
        let pipeline=crate::medium::test_pipeline(&r,&source,&crate::medium::Native::default(),&entries,(1,1,1))?;
        let output=crate::init(&r.device,"sampled emitter integrals",&[0u8;9*16],wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC);
        let unused=crate::init(&r.device,"unused emitter frame slots",&[0u8;16],wgpu::BufferUsages::STORAGE);
        let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[
            entry(0,output.as_entire_binding()),entry(1,unused.as_entire_binding()),entry(2,unused.as_entire_binding()),
            entry(3,wgpu::BindingResource::TextureView(&environment.specular)),
            entry(4,wgpu::BindingResource::Sampler(&environment.sampler)),
            entry(5,wgpu::BindingResource::TextureView(&environment.diffuse)),
            entry(6,wgpu::BindingResource::TextureView(&environment.brdf))]});
        for (case,value) in compute_vec4(&r,&pipeline,&bind,&output,9)?.into_iter().enumerate() {
            eprintln!("finite-emitter case {case}: {value:?}");
            assert!(value[1]>0. && (value[0]/value[1]-1.).abs()<0.03,"emitter/BSDF mixture changed energy: {value:?}");
            if case<7 {assert!((value[2]/value[1]-0.5).abs()<0.02,"partial emitter visibility became a centre test: {value:?}");}
        }
        Ok(())
    }

    /// Compare finite-source energy to the independent path direct estimator,
    /// including the transmission domain and non-coincident lobe horizons.
    #[test]
    fn sampled_hit_backlighting_and_normals_match_path() -> Result<()> {
        let Some(r)=crate::test_renderer()? else {return Ok(());};
        if !r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | crate::rt::HIT_BUFFER_FEATURES) {return Ok(());}
        for backlight in [true,false] {
            let mut document:serde_json::Value=serde_json::from_str(include_str!("../../tests/fixtures/sampled-hit-energy.json"))?;
            if !backlight {
                document["programs"]["receiver"]["source"]="struct Inputs {}; Closure surface(ShadingContext ctx,Inputs i) {Closure c={}; c.base_color=float3(0.7); c.specular_roughness=0.9; c.coat_weight=0.7; c.coat_ior=1.; c.coat_roughness=0.9; c.geometry_coat_normal=normalize(float3(0.8660254,0.,-0.5)); return c;}".into();
                document["nodes"][3]["translation"]=serde_json::json!([-2.5,0,2]);
                // An index-matched coat has zero response but still supplies a
                // proposal. This isolates PDF support from path-only conductor
                // multiscatter compensation and rough-coat attenuation policies.
                document["nodes"][3]["size"]=4.into();document["nodes"][3]["size_y"]=4.into();
                // The source faces the receiver and spans both lobe horizons.
                document["nodes"][3]["rotation"]=serde_json::json!([0,1,0,0]);
            }
            let mut energies=Vec::new();
            for profile in ["std/final","std/path"] {
                let mut scene=crate::test_scene(document.clone(),Path::new("."))?;
                let settings=vork_scene::Profile::load(profile)?.resolve(&Default::default())?;
                let mut overrides=serde_json::json!({"samples":4096,"denoiser":null});
                if profile=="std/final" {
                    overrides["shadow_method"]="ray".into();overrides["fast_gi"]=serde_json::Value::Null;
                    overrides["reflections"]="ray".into();
                }
                scene.render.settings=vork_scene::test_settings(&settings,overrides.as_object().unwrap())?;
                let mut installed=r.install(scene,Path::new("."))?;
                let mut slot=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
                let frame=installed.render_frame(&r,&mut slot,0.)?;
                let pixels=frame.linear_rgba.as_ref().context("float camera energy")?;
                energies.push(pixels.iter().map(|p|p[0] as f64).sum::<f64>()/(16.*16.));
            }
            eprintln!("backlight={backlight}: raster/path energy {energies:?}");
            assert!(energies[1]>0. && (energies[0]/energies[1]-1.).abs()<0.03,"backlight={backlight}: raster/path energy {energies:?}");
        }
        Ok(())
    }

    // Equal flux from sources differing by 100x in area must remain stable
    // even when the tiny source is outside the camera-colour guide domain.
    #[test]
    fn hardware_reflections_discover_subpixel_offscreen_emitters() -> Result<()> {
        let Some(r)=crate::test_renderer()? else {return Ok(());};
        if !r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | crate::rt::HIT_BUFFER_FEATURES) {return Ok(());}
        let path=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-reflection-emitter.json");
        let tiny=crate::load_test_fixture(&path, include_str!("../../tests/presets/tiny-reflection-emitter-25918a49.json"))?;
        let mut value:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        for position in value["nodes"][1]["geometry"]["attributes"]["POSITION"]["data"].as_array_mut().unwrap() {
            for axis in [0,2] { position[axis]=serde_json::json!(position[axis].as_f64().unwrap().signum()*0.1); }
        }
        value["materials"]["source"]["inputs"]["emission_luminance"]=serde_json::json!(30);
        let mut broad=crate::test_scene(value.clone(), std::path::Path::new("."))?;
        broad.apply_settings(tiny.render.settings.clone(),"test")?;
        let render=|scene:Scene| -> Result<Vec<[f32;4]>> {
            let mut installed=r.install(scene,path.parent().unwrap())?;
            let mut frame=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
            Ok(installed.render_frame(&r, &mut frame, 0.)?.linear_rgba.context("linear reflection regression")?)
        };
        let reference=render(broad)?;
        let actual=render(tiny)?;
        let mut error=0.;let mut energy=0.;let mut peak=0f32;
        for y in 10..22 {for x in 4..28 {
            let at=y*32+x;let a=actual[at][0];let b=reference[at][0];
            assert!(a.is_finite() && b.is_finite());
            error+=(a-b).abs();energy+=b;peak=peak.max((a-b).abs());
        }}
        assert!(energy>0.1,"source failed to illuminate the reflection: {energy}");
        assert!(error/energy<0.06 && peak<0.01,"subpixel emission remains noisy: relative error {}, peak {peak}, energy {energy}",error/energy);
        Ok(())
    }

    #[test]
    fn installed_reflection_specialization_matches_the_full_evaluator() -> Result<()> {
        let Some(r) = crate::test_renderer()? else { return Ok(()); };
        let path=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hit-layers.json");
        let mut scene=crate::load_test_fixture(&path, include_str!("../../tests/presets/hit-layers-3bd1f7c2.json"))?;
        scene.render.resolution=[96,64];
        scene.render.settings.samples=8;
        let mut installed=r.install(scene,path.parent().unwrap())?;
        let mut specialized=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
        let specialized=installed.render_frame(&r, &mut specialized, 0.)?;
        // No specialized bit: retain every texture, lobe and light capability.
        let full=crate::forward_variant::FeatureMask::from(crate::materials::SHADING_LAYERS | crate::materials::SHADING_METAL);
        installed.reflections=Some(pipeline(&r,&installed.pipelines,full,installed.plan.effects.probes,&crate::pipelines::VariantCompile::Inline)?);
        let mut generic=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
        let generic=installed.render_frame(&r, &mut generic, 0.)?;
        crate::test_linear::assert_linear_eq(specialized.linear_rgba.as_ref().expect("linear readback"), generic.linear_rgba.as_ref().expect("linear readback"));
        Ok(())
    }

    #[test]
    fn transport_controls_keep_nearest_sample_and_exported_depth() -> Result<()> {
        let Some(r)=crate::test_renderer()? else {return Ok(());};
        let d=&r.device;
        // Shared pass IDs cannot join disconnected primitives at different depths.
        let module=d.create_shader_module(wgpu::ShaderModuleDescriptor {label:None,source:wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#"
struct Out { @builtin(position) p:vec4f };
@vertex fn vs(@builtin(vertex_index) i:u32)->Out {
    var p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));
    var o:Out;o.p=vec4f(p[i],0.,1.);return o;
}
struct Pixel { @location(0) ids:vec4u,@location(1) normal:vec4f,@builtin(frag_depth) depth:f32 };
@fragment fn fs(@builtin(sample_index) sample:u32,@builtin(position) p:vec4f)->Pixel {
    let near=sample<2u;var o:Pixel;
    o.ids=vec4u(7u,0u,select(1u,0u,near),bitcast<u32>(.02));
    o.normal=select(vec4f(0.,1.,0.,-2.),vec4f(1.,0.,0.,0.),near);
    o.depth=select(.8,.2+.02*f32(sample),near);
    if (p.x>1. && p.x<2.) {
        // Continuous surface: the depth span fits inside its one-pixel slope.
        o.ids=vec4u(7u,0u,sample,bitcast<u32>(.6));o.depth=.2+.2*f32(sample);
    }
    if (p.x>2.) {o.ids.y=select(1u,0u,near);o.ids.w=bitcast<u32>(.6);}
    return o;
}
"#))});
        let targets=[wgpu::TextureFormat::Rgba32Uint,wgpu::TextureFormat::Rgba16Float].map(|format|Some(wgpu::ColorTargetState {format,blend:None,write_mask:wgpu::ColorWrites::ALL}));
        let pipeline=d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {label:None,layout:None,
            vertex:wgpu::VertexState {module:&module,entry_point:Some("vs"),compilation_options:Default::default(),buffers:&[]},
            fragment:Some(wgpu::FragmentState {module:&module,entry_point:Some("fs"),compilation_options:Default::default(),targets:&targets}),
            primitive:Default::default(),depth_stencil:Some(wgpu::DepthStencilState {format:wgpu::TextureFormat::Depth32Float,depth_write_enabled:Some(true),depth_compare:Some(wgpu::CompareFunction::Always),stencil:Default::default(),bias:Default::default()}),
            multisample:wgpu::MultisampleState {count:4,..Default::default()},multiview_mask:None,cache:None});
        let images=[wgpu::TextureFormat::Rgba32Uint,wgpu::TextureFormat::Rgba16Float,wgpu::TextureFormat::Depth32Float].map(|format|texture(d,"shared-pass MSAA edge",3,1,4,format,wgpu::TextureUsages::RENDER_ATTACHMENT|wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default()));
        let attachments=[0,1].map(|i|Some(wgpu::RenderPassColorAttachment {view:&images[i],depth_slice:None,resolve_target:None,ops:wgpu::Operations {load:wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),store:wgpu::StoreOp::Store}}));
        let mut encoder=d.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_render_pass(&wgpu::RenderPassDescriptor {label:None,color_attachments:&attachments,multiview_mask:None,
            depth_stencil_attachment:Some(wgpu::RenderPassDepthStencilAttachment {view:&images[2],depth_ops:Some(wgpu::Operations {load:wgpu::LoadOp::Clear(1.),store:wgpu::StoreOp::Store}),stencil_ops:None}),timestamp_writes:None,occlusion_query_set:None});
            pass.set_pipeline(&pipeline);pass.draw(0..3,0..1);}
        r.queue.submit([encoder.finish()]);
        let source=r#"
import vork_control_sample;
[[vk::binding(0,0)]] Texture2DMS<float> depth : register(t0);
[[vk::binding(1,0)]] Texture2DMS<uint4> ids : register(t1);
[[vk::binding(2,0)]] Texture2DMS<float4> normals : register(t2);
[[vk::binding(3,0)]] RWStructuredBuffer<float4> result : register(u3);
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    let c=sample_controls(int2(id.x,0),depth,ids,normals,normals,normals,4u);
    result[id.x*2]=c.depth;result[id.x*2+1]=c.normals;
}
"#;
        let mut entries=vec![pipelines::texture_entry(0,wgpu::TextureSampleType::Depth,true),pipelines::texture_entry(1,wgpu::TextureSampleType::Uint,true),pipelines::texture_entry(2,wgpu::TextureSampleType::Float {filterable:false},true),pipelines::buffer_entry(3,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false})];
        for e in &mut entries {e.visibility=wgpu::ShaderStages::COMPUTE;}
        let compute=crate::medium::test_pipeline(&r,source,&crate::medium::Native::default(),&entries,(1,1,1))?;
        let output=crate::init(d,"transport controls result",&[0u8;96],wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC);
        let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&compute.get_bind_group_layout(0),entries:&[
            entry(0,wgpu::BindingResource::TextureView(&images[2])),entry(1,wgpu::BindingResource::TextureView(&images[0])),entry(2,wgpu::BindingResource::TextureView(&images[1])),entry(3,output.as_entire_binding())]});
        let actual=compute_vec4_count(&r,&compute,&bind,&output,3,6)?;
        assert!((actual[0][0]-0.21).abs()<1e-6,"exported depth changed: {:?}",actual[0]);
        assert!((actual[0][1]-0.2).abs()<1e-6,"invented transport surface: {:?}",actual[0]);
        assert_eq!(actual[1],[1.,0.,0.,1.],"nearest normal or sample index was lost");
        assert!((actual[2][0]-0.5).abs()<1e-6,"continuous triangle boundary moved the centroid: {:?}",actual[2]);
        assert!((actual[4][0]-0.21).abs()<1e-6,"separate objects joined: {:?}",actual[4]);
        r.errors.check()?;Ok(())
    }


    // This reduction retains the camera, vertex positions and pixel coordinates
    // from BB109. Bounds come from projecting the original triangle vertices
    // onto the camera's view axis, independently of raster depth reconstruction.
    #[test]
    fn folded_mesh_depth_stays_on_near_surface() -> Result<()> {
        let Some(renderer) = crate::test_renderer()? else { return Ok(()); };
        let mut json: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/depth-folds.json"))?;
        json["render"]["settings"] = serde_json::json!({"integrator": "path", "samples": 4, "denoiser": null});
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        // A shared object, separate objects sharing a public pass index, and distinct
        // public indices must all select the same near fold.
        for split in 0..3 {
            let mut scene = json.clone();
            if split != 0 {
                let mut rear = scene["nodes"][0].clone();
                scene["nodes"][0]["geometry"]["indices"]["data"] = serde_json::json!([3,5,4]);
                rear["geometry"]["indices"]["data"] = serde_json::json!([0,2,1]);
                rear["name"] = serde_json::json!("rear");
                if split == 2 { rear["pass_index"] = serde_json::json!(81); }
                scene["nodes"].as_array_mut().unwrap().push(rear);
            }
            let mut installed = renderer.install(vork_scene::v2::from_test_value(scene)?, &base)?;
            let mut slot = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
            let output = installed.render_frame(&renderer, &mut slot, 2./7.)?;
            let at = 108 * 320 + 167;
            let z = output.depth_f32.as_ref().unwrap()[at];
            assert_eq!(output.ids[at], 80);
            assert!((4.56..=4.5966).contains(&z), "split {split}: depth {z} lies off the near fold");
        }
        Ok(())
    }

}
