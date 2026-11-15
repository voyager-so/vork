//! Install the authored segment hook in the existing native program workspace.
use crate::*;
#[derive(Clone)]
pub(crate) struct Native {
    pub modules: Vec<(String,String)>,
    pub include: Vec<PathBuf>,
    pub words: Vec<u32>,
}
pub(crate) fn prepare(r:&Renderer,scene:&vork_scene::v2::Scene,base:&Path)->Result<Option<Native>> {
    let Some(p)=&scene.propagation else {return Ok(None);};
    ensure!(p.bounds.iter().flatten().all(|v|v.is_finite()) && (0..3).all(|i|p.bounds[0][i]<p.bounds[1][i]), "propagation.bounds must have finite ordered endpoints");
    let (source,origin)=programs::program_source(base,&p.program,scene.programs.get(&p.program).context("unknown propagation program")?)?;
    let abi=programs::abi();let mut hash=blake3::Hasher::new();
    hash.update(b"authored-segments-v1");hash.update(include_str!("../../vork-slang/src/propagation.rs").as_bytes());hash.update(abi.cache_identity()?.as_bytes());hash.update(source.as_bytes());hash.update(origin.to_string_lossy().as_bytes());
    let key=hash.finalize();let tag=key.to_hex();
    let program=programs::cached(r,key,&vork_slang::Compiler::discover(),
        &abi, &source, &origin,
        |compiler|compiler.compile_propagation(&abi,&source,&origin,&tag[..12]))?;
    if !program.warnings.trim().is_empty() {eprintln!("{}",program.warnings);}
    let words=program.inputs.pack(&std::collections::BTreeMap::new(),|_,_|unreachable!())?;
    let mut text=program.native.clone();
    for dependency in &program.dependencies {text.push_str(&format!("\n// dependency {}\n",blake3::hash(&std::fs::read(dependency)?)));}
    for (name,v) in [("propagation_lower",p.bounds[0]),("propagation_upper",p.bounds[1])] {
        text.push_str(&format!("\npublic float3 {name}() {{return float3({:.9e},{:.9e},{:.9e});}}\n",v[0],v[1],v[2]));
    }
    let mut modules=abi.native_modules();modules.push(("propagation_program".into(),text));
    Ok(Some(Native {modules,include:program.include,words}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value,json};
    fn scene(step:Option<&str>,medium:bool)->Value {
        let mut s=json!({"version":2,"render":{"resolution":[8,8]},
            "programs":{"surface":{"source":"struct Inputs {}; Closure surface(ShadingContext c,Inputs i) {return unlit(float3(.4,.6,.8));}"}},
            "materials":{"surface":{"program":"surface"}},
            "buffers":{"p":{"type":"vec3","data":[[-10,-10,-2],[10,-10,-2],[0,10,-2]]}},
            "geometry":{"g":{"attributes":{"POSITION":"p"}}},
            "nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":1,"clip_start":0.01}}},
                {"type":"mesh","geometry":"g","materials":["surface"]}]});
        if let Some(body)=step {
            s["propagation"]=json!({"program":"step","bounds":[[-3,-3,-3],[3,3,-0.5]]});
            s["programs"]["step"]=json!({"source":format!("struct Inputs {{}}; PropagationStep propagation(PropagationContext c,Inputs i) {{PropagationStep s;s.direction=c.direction;s.distance=.125;{body}return s;}}")});
        }
        if medium {
            s["programs"]["fog"]=json!({"source":"struct Inputs {}; MediumCoefficients medium(MediumContext c,Inputs i) {MediumCoefficients m;m.absorption=float3(.2);return m;}"});
            s["materials"]["fog"]=json!({"program":"fog"});
            s["nodes"].as_array_mut().unwrap().push(json!({"type":"light","shape":"environment","energy":0,"medium":"fog"}));
        }
        s
    }
    fn render(r:&Renderer,s:Value)->Result<Vec<[f32;4]>> {
        let mut scene:vork_scene::v2::Scene=vork_scene::v2::from_test_value(s)?;
        scene.render.settings=vork_scene::v2::from_test_value(json!({"integrator":"path","samples":4,"denoiser":null}))?;
        let installed=r.install(scene,Path::new("."))?;
        let mut slot=installed.create_frame_slot(r,ReadbackPrecision::Float)?;
        slot.submit_outputs(&installed,r,0.,ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
        slot.collect(r)?.linear_rgba.context("linear pixels")
    }
    #[test]
    #[ignore="real NVIDIA L4 and pinned Slang"]
    fn schwarzschild_actual_solver_matches_photon_sphere_deflection_and_capture()->Result<()> {
        let r=Renderer::new()?;
        // Execute the example's actual step function, not a Rust port of it.
        let source=format!(r#"
import vork;
{}
[[vk::binding(0,0)]] RWStructuredBuffer<float4> results;
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {{
    let critical=3.*sqrt(3.);
    let impact=id.x==0u ? 50. : id.x==1u ? 100. : critical*(id.x==2u ? .99 : 1.01);
    float3 p=float3(impact,0.,1000.);float3 d=float3(0.,0.,-1.);float previous=0.;
    bool absorbed=false;uint steps=0u;
    for(;steps<200000u;steps++) {{
        let s=propagation(PropagationContext(p,d,0.,previous),Inputs());
        if(s.absorbed) {{absorbed=true;break;}}
        p+=s.direction*s.distance;d=s.direction;previous=s.distance;
        if(length(p)>1000. && dot(p,d)>0.) break;
    }}
    results[id.x]=float4(d.x,d.z,float(absorbed),float(steps));
    if(id.x==0u) {{let radius=1.+sqrt(3.)*.5;results[4]=float4(radius,-gradient(float3(radius,0.,0.)).x*radius,0.,0.);}}
}}
"#,include_str!("../../../examples/black-hole/propagation.slang"));
        let output=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {label:Some("black hole oracle"),size:80,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
        let read=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {label:Some("black hole oracle read"),size:80,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let entries=[pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false})];
        let pipeline=crate::medium::test_pipeline(&r,&source,&crate::medium::Native::default(),&entries,(1,1,1))?;
        let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,output.as_entire_binding())]});
        let mut encoder=r.device.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(4,1,1);}
        encoder.copy_buffer_to_buffer(&output,0,&read,0,80);r.queue.submit([encoder.finish()]);
        let(tx,rx)=std::sync::mpsc::channel();read.map_async(wgpu::MapMode::Read,..,move |v|{let _=tx.send(v);});r.wait(None)?;rx.recv()??;
        let mapped=read.get_mapped_range(..);let rows:&[[f32;4]]=bytemuck::cast_slice(&mapped);
        // Areal R=r(1+M/(2r))^2; the photon orbit R=3M is at
        // isotropic r=1+sqrt(3)/2, where d log(n)/d log(r)=-1.
        let radius=rows[4][0] as f64;
        assert!((radius*(1.+0.5/radius).powi(2)-3.).abs()<1e-6);
        assert!((rows[4][1]-1.).abs()<2e-6);
        for (index,b) in [50f64,100.].into_iter().enumerate() {
            let angle=(-(rows[index][0] as f64)).atan2(-(rows[index][1] as f64));
            let reference=4./b+15.*std::f64::consts::PI/(4.*b*b)+128./(3.*b.powi(3));
            // The example uses 2%-radius polygonal steps and tapers the metric
            // at 500..1000M. Budget includes the finite-bound tail O(b/1000^2).
            assert!((angle-reference).abs()<0.001,"b={b}: {angle} vs {reference}; {rows:?}");
            assert_eq!(rows[index][2],0.);
        }
        assert_eq!(rows[2][2],1.,"below 3sqrt(3)M must capture: {rows:?}");
        assert_eq!(rows[3][2],0.,"above 3sqrt(3)M must escape: {rows:?}");
        assert!(rows[..4].iter().all(|row|row[3]<200000.),"oracle failed to finish: {rows:?}");
        Ok(())
    }

    #[test]
    #[ignore="real GPU and pinned Slang"]
    fn segments_identity_absorption_and_media()->Result<()> {
        let r=crate::test_renderer()?;
        for medium in [false,true] {
            let straight=render(&r,scene(None,medium))?;
            let segmented=render(&r,scene(Some(""),medium))?;
            for (a,b) in straight.iter().zip(&segmented) {for c in 0..3 {assert!((a[c]-b[c]).abs()<1e-4,"identity medium={medium}: {a:?} versus {b:?}");}}
            let absorbed=render(&r,scene(Some("s.absorbed=true;"),medium))?;
            assert!(absorbed.iter().all(|p|p[..3].iter().all(|v|v.abs()<1e-6)));
        }
        Ok(())
    }

    #[test]
    #[ignore="real GPU and pinned Slang"]
    fn segments_reject_invalid_steps_with_their_diagnostic()->Result<()> {
        let r=crate::test_renderer()?;
        let error=render(&r,scene(Some("s.distance=0.;"),false)).unwrap_err().to_string();
        assert!(error.contains("authored propagation failed"),"invalid steps need their own diagnostic: {error}");
        Ok(())
    }

    #[test]
    #[ignore="real GPU and pinned Slang"]
    fn segments_turn_camera_bounces_and_escape_direction()->Result<()> {
        let r=crate::test_renderer()?;
        let mut turned=scene(Some("s.direction=float3(1.,0.,0.);s.distance=4.;s.radiance=float3(.5,.25,1.);"),false);
        turned["buffers"]["p"]["data"]=json!([[2,-10,-10],[2,10,-10],[2,0,10]]);
        let pixels=render(&r,turned)?;
        for p in pixels {for (c,v) in [0.2,0.15,0.8].iter().enumerate() {assert!((p[c]-v).abs()<0.002,"turned segment transfer: {p:?}");}}
        let mut sky=scene(Some("s.direction=float3(1.,0.,0.);s.distance=4.;"),false);
        sky["nodes"].as_array_mut().unwrap().truncate(1);
        sky["render"]["background"]=json!("environment");
        sky["programs"]["sky"]=json!({"source":"struct Inputs {}; float3 environment(EnvironmentContext c,Inputs i) {return c.direction.x>.9 ? float3(.2,.4,.8) : float3(0.);}"});
        sky["materials"]["sky"]=json!({"program":"sky"});
        sky["nodes"].as_array_mut().unwrap().push(json!({"type":"light","shape":"environment","material":"sky","energy":1}));
        let pixels=render(&r,sky)?;
        assert!(pixels.iter().all(|p|p[2]>0.7 && p[0]>0.1),"sky must use the escaping +X direction");
        let mut mirror=scene(Some("if(c.direction.z>.5) s.direction=float3(1.,0.,0.);"),false);
        mirror["render"]["background"]=json!([0,0,0]);
        mirror["materials"]["mirror"]=json!({"inputs":{"base_color":[1,1,1],"base_metalness":1,"specular_roughness":0}});
        mirror["nodes"][1]["materials"]=json!(["mirror"]);
        mirror["buffers"]["q"]=json!({"type":"vec3","data":[[2,-10,-2.5],[2,10,-2.5],[2,0,-1.5]]});
        mirror["geometry"]["lamp"]=json!({"attributes":{"POSITION":"q"}});
        mirror["nodes"].as_array_mut().unwrap().push(json!({"type":"mesh","geometry":"lamp","materials":["surface"]}));
        mirror["nodes"].as_array_mut().unwrap().push(json!({"type":"light","shape":"environment","energy":0}));
        let bent=render(&r,mirror.clone())?;
        mirror.as_object_mut().unwrap().remove("propagation");
        let straight=render(&r,mirror)?;
        assert!(bent.iter().all(|p|p[2]>0.5),"bounce rays must invoke the authored step");
        assert!(straight.iter().all(|p|p[2]<0.01),"the emitter is outside the straight reflection: {straight:?}");
        Ok(())
    }

    #[test]
    #[ignore="real GPU and pinned Slang"]
    fn segments_apply_to_delta_connections_even_without_occlusion()->Result<()> {
        let r=crate::test_renderer()?;
        for shadows in [true,false] {
            let mut input=scene(Some("if(c.direction.z>.1) s.direction=float3(1.,0.,0.);"),false);
            input["render"]["background"]=json!([0,0,0]);
            input["materials"]["surface"]=json!({"inputs":{"base_color":[0.8,0.8,0.8],"specular_weight":0}});
            input["nodes"].as_array_mut().unwrap().push(json!({"type":"light","shape":"point","radius":0,"energy":160,"translation":[0,0,0]}));
            input["nodes"].as_array_mut().unwrap().push(json!({"type":"light","shape":"environment","energy":0}));
            if !shadows {input["nodes"][2]["shadowed_by"]=json!([]);}
            let bent=render(&r,input.clone())?;
            input.as_object_mut().unwrap().remove("propagation");
            let straight=render(&r,input)?;
            assert!(bent.iter().all(|p|p[0]<0.001),"a turned connection cannot reach a point source: shadows={shadows}");
            assert!(straight.iter().all(|p|p[0]>0.1),"straight reference must reach the source");
        }
        let mut mirror=scene(Some(""),false);
        mirror["materials"]["surface"]=json!({"inputs":{"base_color":[1,1,1],"base_metalness":1,"specular_roughness":0}});
        mirror["nodes"].as_array_mut().unwrap().extend([
            json!({"type":"light","shape":"environment","energy":0}),
            json!({"type":"light","shape":"point","radius":1,"energy":40,"translation":[0,0,1]})]);
        let segmented=render(&r,mirror.clone())?;
        mirror.as_object_mut().unwrap().remove("propagation");
        let straight=render(&r,mirror)?;
        assert!(straight.iter().all(|p|p[0]>0.1),"the finite analytic light must appear in the mirror");
        let errors:Vec<f32>=straight.iter().zip(&segmented).flat_map(|(a,b)|(0..3).map(move |c|(a[c]-b[c]).abs())).collect();
        let mean=errors.iter().sum::<f32>()/errors.len() as f32;let maximum=errors.iter().copied().fold(0.,f32::max);
        // Four-sample estimates use different launch proposals: test brightness,
        // allowing their finite-sample variance instead of demanding pixel identity.
        assert!(mean<0.005 && maximum<0.02,"finite analytic mirror: mean error {mean}, maximum {maximum}");
        eprintln!("finite analytic mirror: mean error {mean}, maximum {maximum}");
        let mut sun=scene(Some(""),false);
        sun["materials"]["surface"]=json!({"inputs":{"base_color":[0.8,0.8,0.8],"specular_weight":0}});
        sun["nodes"].as_array_mut().unwrap().extend([
            json!({"type":"light","shape":"environment","energy":0}),
            json!({"type":"light","shape":"sun","angle":0.05,"energy":3})]);
        let segmented=render(&r,sun.clone())?;
        sun.as_object_mut().unwrap().remove("propagation");
        let straight=render(&r,sun)?;
        let average=|pixels:&Vec<[f32;4]>|pixels.iter().map(|p|p[0]).sum::<f32>()/pixels.len() as f32;
        let reference=average(&straight);let measured=average(&segmented);
        eprintln!("narrow sun: segmented {measured}, straight {reference}");
        assert!(reference>0.1 && (measured/reference-1.).abs()<0.25,"launch-space light proposal: {measured} versus {reference}");
        Ok(())
    }
}
