use crate::*;

fn read(r:&Renderer,buffer:&wgpu::Buffer)->Result<Vec<f32>> {
    let output=r.device.create_buffer(&wgpu::BufferDescriptor {label:None,size:buffer.size(),usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
    let mut encoder=r.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer,0,&output,0,buffer.size());let submission=r.queue.submit([encoder.finish()]);
    let (tx,rx)=std::sync::mpsc::channel();output.map_async(wgpu::MapMode::Read,..,move |v|{let _=tx.send(v);});
    r.device.poll(wgpu::PollType::Wait {submission_index:Some(submission),timeout:None})?;rx.recv()??;
    Ok(bytemuck::cast_slice(&output.get_mapped_range(..)).to_vec())
}
fn document()->serde_json::Value {
    serde_json::json!({"version":2,
        "composition":{"space":"3d","view":{"program":"std/standard"}},"render":{"resolution":[16,16],"fps":1,"duration":2,"settings":{"integrator":"path","denoiser":null,"samples":2}},
        "buffers":{
            "p":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0],[2,0,0],[4,0,0],[2,2,0],[5,0,0],[5,0,0],[5,0,0]]},
            "n":{"type":"vec3","data":[[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]},
            "delta":{"type":"vec3","data":[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,1,0]]},
            "i":{"type":"u32","data":[0,1,2,3,4,5,6,7,8]}},
        "geometry":{"source":{"topology":"triangles","attributes":{"POSITION":"p","NORMAL":"n"},"indices":"i","morphs":{"grow":{"POSITION":"delta"}}}},
        "materials":{"source":{"inputs":{"base_weight":0,"specular_weight":0,"emission_luminance":3}}},
        "nodes":[{"type":"camera","translation":[2,1,10],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":12}}},
            {"type":"mesh","name":"source","geometry":"source","materials":["source"],"morph_weights":{"grow":0}}],
        "animation":[{"tracks":[
            {"path":["nodes","source","scale"],"keys":[[0,[1,1,1]],[1,[2,3,1]]]},
            {"path":["nodes","source","morph_weights","grow"],"keys":[[0,0],[1,1]]}]}]})
}
fn check(values:&[f32],expected:[f32;3]) {
    assert_eq!(values.len(),8);
    for (a,b) in values[4..7].iter().zip(expected.map(|x|x*3.)) {assert!((*a-b).abs()<1e-5,"current area: {values:?}, expected {expected:?}");}
    assert_eq!(values[7],0.,"padded triangles have no probability");
    assert!((values[1]*4.-expected.iter().sum::<f32>()*3.).abs()<1e-5,"root area {values:?}");
}
#[test]
#[ignore="requires pinned Slang and Vulkan ray queries (lavapipe works)"]
fn current_areas_follow_deformation_transforms_and_frame_slots()->Result<()> {
    let r=Renderer::new()?;let mut s=r.install(vork_scene::v2::from_test_value(document())?,Path::new("."))?;
    let mut a=s.create_frame_slot(&r,Default::default())?;let mut b=s.create_frame_slot(&r,Default::default())?;
    let mut render=|frame:&mut FrameSlot,time|->Result<Vec<f32>> {
        s.render_frame(&r, &mut *frame, time)?;
        read(&r,&frame.rays.as_ref().unwrap().emitter_weights.as_ref().unwrap().buffer)
    };
    check(&render(&mut a,0.)?,[0.5,2.,0.]);
    check(&render(&mut b,1.)?,[3.,12.,3.]);
    // Another frame slot cannot overwrite the first slot's retained proposal.
    check(&read(&r,&a.rays.as_ref().unwrap().emitter_weights.as_ref().unwrap().buffer)?,[0.5,2.,0.]);
    check(&render(&mut a,0.5)?,[1.5,6.,0.375]);
    check(&render(&mut a,1.)?,[3.,12.,3.]);
    check(&render(&mut a,0.)?,[0.5,2.,0.]);
    r.errors.check()?;Ok(())
}

#[test]
#[ignore="requires pinned Slang and Vulkan (lavapipe works)"]
fn proposal_histogram_matches_defensive_mixture_and_reports_zero_mass()->Result<()> {
    let r=Renderer::new()?;
    // Two sources, three actual triangles and one padded leaf. The inverse
    // instance map deliberately includes a non-emitting instance between them.
    let support=init(&r.device,"mesh proposal support",bytemuck::cast_slice(&[2u32,3,3,0,u32::MAX,1,0,0,1,2,1,3]),wgpu::BufferUsages::STORAGE);
    let source=r#"
import emitter_mesh_proposal;
import hit_geometry;
import vork_constants;
import vork_sampling;
[[vk::binding(0,0)]] RWStructuredBuffer<float4> result;
[[vk::binding(1,0)]] StructuredBuffer<float> distribution;
[[vk::binding(2,0)]] StructuredBuffer<uint> support;
struct Geometry : IHitGeometry {
    StructuredBuffer<uint> support;
    __init(StructuredBuffer<uint> support) {this.support=support;}
    uint word(uint i) {return support[i];}
    uint load(uint buffer,uint i) {return 0u;}
}
[shader("compute")][numthreads(64,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    let i=id.x;let x=pixel_hash(i);let y=pixel_hash(x);let z=pixel_hash(y);
    // Exhaust the centered 16-bit choice lattice in each mixture component.
    // Barycentric coordinates stay independently hashed.
    let u=float3((float(i)+.5)/65536.,(float(y>>9u)+.5)/8388608.,(float(z>>9u)+.5)/8388608.);
    let large=Geometry(support).word(1u)>65536u;
    let choice=large ? (float(i)+.5)/float(Geometry(support).word(1u)) : (i==0u ? 1. : (i==1u ? 0. : u.x));
    for(uint component=0u;component<2u;component++) {
        let sample=emitter_mesh_sample<Geometry>(Geometry(support),0u,float3(choice,u.y,(u.z+float(component))*.5),distribution);
        let triangle=sample.primitive+(sample.instance==0u ? 0u : 1u);
        let pdf=emitter_mesh_probability<Geometry>(Geometry(support),0u,sample.instance,sample.primitive,distribution);
        result[2u*i+component]=float4(float(triangle),abs(sample.probability-pdf)<1e-6 ? pdf : -1.,sample.barycentrics);
    }
}
"#;
    let stages=wgpu::ShaderStages::COMPUTE;
    let entries=[pipelines::buffer_entry(0,stages,wgpu::BufferBindingType::Storage {read_only:false}),
        pipelines::buffer_entry(1,stages,wgpu::BufferBindingType::Storage {read_only:true}),
        pipelines::buffer_entry(2,stages,wgpu::BufferBindingType::Storage {read_only:true})];
    let pipeline=medium::test_pipeline(&r,source,&medium::Native::default(),&entries,(64,1,1))?;
    let output=r.device.create_buffer(&wgpu::BufferDescriptor {label:None,size:131072*16,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
    for areas in [[0.5f32,2.,0.],[0.,2.,0.5],[0.,0.,0.],[1e38,1e38,1e38],[1e-20,2e-20,0.],[67108864.,1.,0.],[1.,67108864.,0.],[1e38,1e-20,1.]] {
        let total=areas.iter().map(|x|f64::from(*x)).sum::<f64>();
        let weights=[0.,(total/4.) as f32,0.5*areas[0]+0.5*areas[1],0.5*areas[2],areas[0],areas[1],areas[2],0.];
        let distribution=init(&r.device,"test area tree",bytemuck::cast_slice(&weights),wgpu::BufferUsages::STORAGE);
        let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[
            entry(0,output.as_entire_binding()),entry(1,distribution.as_entire_binding()),entry(2,support.as_entire_binding())]});
        let mut encoder=r.device.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(1024,1,1);}
        r.queue.submit([encoder.finish()]);let samples=read(&r,&output)?;
        let mut histogram=[0usize;3];let mut barycentric=[0f64;2];
        for sample in samples.chunks_exact(4) {
            assert!(sample.iter().all(|x|x.is_finite()),"nonfinite area sample: {sample:?}");
            let triangle=sample[0] as usize;assert!(triangle<3);
            let expected=if total>0. {(0.5*f64::from(areas[triangle])/total+0.5/3.) as f32} else {0.};
            assert!((sample[1]-expected).abs()<1e-6,"PDF mismatch {areas:?}: {sample:?}");
            histogram[triangle]+=1;barycentric[0]+=f64::from(sample[2]);barycentric[1]+=f64::from(sample[3]);
        }
        if total>0. {
            eprintln!("mesh mixture masses={areas:?}, histogram={histogram:?}");
            assert!(histogram.iter().all(|n|*n>0),"uniform proposal lost source support");
            for i in 0..3 {assert!((histogram[i] as f32/131072.-(0.5*f64::from(areas[i])/total+0.5/3.) as f32).abs()<3./131072.,"finite-lattice mixture histogram {areas:?}: {histogram:?}");}
            for mean in barycentric {assert!((mean/131072.-1./3.).abs()<0.005,"triangle barycentrics are not area-uniform");}
        }
    }
    // At 2^23 equal leaves, the path sampler's centered FP32 choice still
    // addresses every leaf. Splitting that coordinate for a mixture used to
    // select only odd leaves; verify both components retain even and odd IDs.
    let count=1u32<<23;
    let large_support=init(&r.device,"large proposal support",bytemuck::cast_slice(&[1u32,count,1,0,0,0,count]),wgpu::BufferUsages::STORAGE);
    let large_weights=init(&r.device,"large balanced tree",bytemuck::cast_slice(&vec![1f32;2*count as usize]),wgpu::BufferUsages::STORAGE);
    let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[
        entry(0,output.as_entire_binding()),entry(1,large_weights.as_entire_binding()),entry(2,large_support.as_entire_binding())]});
    let mut encoder=r.device.create_command_encoder(&Default::default());
    {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(1,1,1);}
    r.queue.submit([encoder.finish()]);let samples=read(&r,&output)?;
    for (i,sample) in samples.chunks_exact(4).take(128).enumerate() {
        assert_eq!(sample[0] as usize,i/2,"mixture lost a representable leaf");
        assert_eq!(sample[1],1./count as f32,"balanced mixture PDF");
    }
    r.errors.check()?;Ok(())
}

#[test]
#[ignore="requires pinned Slang and Vulkan ray queries (lavapipe works)"]
fn area_reduction_crosses_workgroups_and_zeroes_padding()->Result<()> {
    let mut input=document();let mut positions=Vec::new();
    for i in 0..257 {
        let x=(i%17)*2;let y=(i/17)*2;
        positions.extend([[x,y,0],[x+1,y,0],[x,y+1,0]]);
    }
    input["buffers"]["p"]["data"]=serde_json::json!(positions);
    input["buffers"]["n"]["data"]=serde_json::json!(vec![[0,0,1];771]);
    input["buffers"]["i"]["data"]=serde_json::json!((0..771).collect::<Vec<_>>());
    input["geometry"]["source"].as_object_mut().unwrap().remove("morphs");
    input["nodes"][1].as_object_mut().unwrap().remove("morph_weights");
    input.as_object_mut().unwrap().remove("animation");
    let r=Renderer::new()?;let mut s=r.install(vork_scene::v2::from_test_value(input)?,Path::new("."))?;
    let mut frame=s.create_frame_slot(&r,Default::default())?;s.render_frame(&r, &mut frame, 0.)?;
    let values=read(&r,&frame.rays.as_ref().unwrap().emitter_weights.as_ref().unwrap().buffer)?;
    assert_eq!(values.len(),1024);assert_eq!(values[1]*512.,385.5);
    assert!(values[512..769].iter().all(|x|*x==1.5));
    assert!(values[769..].iter().all(|x|*x==0.));
    r.errors.check()?;Ok(())
}

/// A glTF material without emission (black emissiveFactor; strength 1 by
/// default, or 5 here) contributes no path mesh sources; a coloured factor does.
#[test]
#[ignore="requires pinned Slang and Vulkan ray queries (lavapipe works)"]
fn gltf_black_emission_is_not_a_mesh_source()->Result<()> {
    let dir=std::env::temp_dir().join(format!("vork-gltf-dark-emission-{}",std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let positions=[[-1_f32,-1.,0.],[1.,-1.,0.],[0.,1.,0.]];
    std::fs::write(dir.join("mesh.bin"),positions.iter().flatten().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    let r=Renderer::new()?;
    let mut asset=serde_json::json!({"asset":{"version":"2.0"},"extensionsUsed":["KHR_materials_emissive_strength"],
        "buffers":[{"uri":"mesh.bin","byteLength":36}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36}],
        "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[-1,-1,0],"max":[1,1,0]}],
        "meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]}],
        "nodes":[{"mesh":0}],"scenes":[{"nodes":[0]}],"scene":0});
    for (material,sources) in [
        (serde_json::json!({}),false),
        (serde_json::json!({"emissiveFactor":[0,0,0],"extensions":{"KHR_materials_emissive_strength":{"emissiveStrength":5}}}),false),
        (serde_json::json!({"emissiveFactor":[1,0.5,0.25]}),true)] {
        asset["materials"]=serde_json::json!([material]);
        std::fs::write(dir.join("mesh.gltf"),serde_json::to_vec(&asset)?)?;
        let scene=serde_json::json!({"version":2,
            "render":{"resolution":[8,8],"settings":{"integrator":"path","denoiser":null,"samples":1}},
            "nodes":[{"type":"asset","name":"mesh","file":"mesh.gltf"},{"type":"camera","name":"camera","translation":[0,0,3],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":3}}}]});
        let s=r.install(vork_scene::v2::from_test_value(scene)?,&dir)?;
        let weights=s.rays.as_ref().context("path installs rays")?.emitter_weights.as_ref();
        assert_eq!(weights.map(|plan|plan.leaves),sources.then_some(1),"{material}");
    }
    std::fs::remove_dir_all(dir)?;r.errors.check()?;Ok(())
}

/// Subdivision changes the number of leaves but not an emitter's integrated
/// proposal mass. Animated emission is read from the same slot as its geometry.
#[test]
#[ignore="requires pinned Slang and Vulkan ray queries"]
fn emitting_mass_tracks_material_animation_and_tessellation()->Result<()> {
    let r=Renderer::new()?;
    for segments in [1,4,16] {
        let mut input=document();let mut positions=Vec::new();
        for y in 0..segments {for x in 0..segments {
            let a=2.*x as f32/segments as f32;let b=2.*y as f32/segments as f32;let step=2./segments as f32;
            positions.extend([[a,b,0.],[a+step,b,0.],[a,b+step,0.],
                [a+step,b,0.],[a+step,b+step,0.],[a,b+step,0.]]);
        }}
        let vertices=positions.len();
        input["buffers"]["p"]["data"]=serde_json::json!(positions);
        input["buffers"]["n"]["data"]=serde_json::json!(vec![[0.,0.,1.];vertices]);
        input["buffers"]["i"]["data"]=serde_json::json!((0..vertices).collect::<Vec<_>>());
        input["geometry"]["source"].as_object_mut().unwrap().remove("morphs");
        input["nodes"][1].as_object_mut().unwrap().remove("morph_weights");
        input["materials"]["source"]["inputs"]["emission_luminance"]=serde_json::json!(0);
        input["animation"]=serde_json::json!([{"tracks":[{"path":["materials","source","inputs","emission_luminance"],"keys":[[0,0],[1,6]]}]}]);
        let mut s=r.install(vork_scene::v2::from_test_value(input)?,Path::new("."))?;
        let mut frame=s.create_frame_slot(&r,Default::default())?;
        for (time,expected) in [(0.,0.),(1.,24.),(0.5,12.),(0.,0.)] {
            s.render_frame(&r,&mut frame,time)?;
            let weights=read(&r,&frame.rays.as_ref().unwrap().emitter_weights.as_ref().unwrap().buffer)?;
            assert!((weights[1]*(weights.len()/2) as f32-expected).abs()<1e-4,"segments {segments}, time {time}: {} != {expected}",weights[1]);
        }
    }
    r.errors.check()?;Ok(())
}

#[test]
#[ignore="requires pinned Slang and Vulkan ray queries"]
fn procedural_emission_retains_area_support_when_initially_dark()->Result<()> {
    let mut input=document();
    input["programs"]=serde_json::json!({"source":{"source":"import vork; struct Inputs {}; Closure surface(ShadingContext ctx, Inputs i) { Closure c={}; c.base_weight=0.; c.specular_weight=0.; c.emission_color=float3(1.); c.emission_luminance=ctx.time*ctx.time; return c; }"}});
    input["materials"]["source"]=serde_json::json!({"program":"source"});
    let r=Renderer::new()?;let mut s=r.install(vork_scene::v2::from_test_value(input)?,Path::new("."))?;
    let mut frame=s.create_frame_slot(&r,Default::default())?;
    for (time,expected) in [(0.,2.5),(1.,18.),(0.,2.5)] {
        s.render_frame(&r,&mut frame,time)?;
        let weights=read(&r,&frame.rays.as_ref().unwrap().emitter_weights.as_ref().unwrap().buffer)?;
        assert!((weights[1]*(weights.len()/2) as f32-expected).abs()<1e-5,"program area support at {time}: {} != {expected}",weights[1]);
    }
    r.errors.check()?;Ok(())
}
