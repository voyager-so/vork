//! Catcher transport storage is independent of output selection. Both radiance
//! sums use the colour ray's film, lens and shutter samples; division is last.
use crate::*;
pub(crate) fn has_mattes(scene:&Scene)->bool {
    fn any(nodes:&[vork_scene::Node])->bool {nodes.iter().any(|n|n.is_holdout || n.is_shadow_catcher || any(&n.children))}
    any(&scene.nodes)
}
pub(crate) fn has_catchers(scene:&Scene)->bool {
    fn any(nodes:&[vork_scene::Node])->bool {nodes.iter().any(|n|n.is_shadow_catcher || any(&n.children))}
    any(&scene.nodes)
}
#[cfg(test)]
fn promote(scene:&mut Scene) {
    if has_mattes(scene) && !scene.render.settings.integrator.is_path() {
        crate::rt::promote_to_path(scene);
    }
}

/// Restore opaque-film catcher background after exposure integration, before
/// authored compositors. Transparent film keeps the catcher at zero RGBA.
pub(crate) fn finish(s:&InstalledScene,r:&Renderer,uniform:&wgpu::Buffer,targets:&crate::view::Targets,source:&wgpu::Buffer,size:[u32;2],normalization:crate::compositor::Normalization)->Result<(wgpu::Buffer,crate::graph::Dispatch)> {
    let [w,h]=size;
    let words=std::mem::size_of::<Uniform>()/16;
    let temporal=std::mem::offset_of!(Uniform,temporal)/16;
    let flags=std::mem::offset_of!(Uniform,flags)/16;
    let group_width=constants::IMAGE_WORKGROUP_SIZE;
    let divisor=match normalization {crate::compositor::Normalization::Frame=>format!("frame.words[{temporal}].y"),_=>"1.".into()};
    let source_text=format!("struct Frame {{words:array<vec4<f32>,{words}>}};\n@group(0) @binding(0) var<uniform> frame:Frame;\n@group(0) @binding(1) var<storage,read> input:array<vec4<f32>>;\n@group(0) @binding(2) var<storage,read> sums:array<vec4<f32>>;\n@group(0) @binding(3) var<storage,read_write> output:array<vec4<f32>>;\n@compute @workgroup_size({group_width},{group_width}) fn main(@builtin(global_invocation_id) id:vec3<u32>) {{if id.x>={w}u || id.y>={h}u {{return;}}let p=id.y*{w}u+id.x;var value=input[p]/{divisor};let a=sums[p*3u];let b=sums[p*3u+1u];let bg=sums[p*3u+2u];if frame.words[{flags}].z==0. {{let valid=b.xyz!=vec3(0.);let ratio=select(vec3(1.),a.xyz/select(vec3(1.),b.xyz,valid),valid);value+=vec4(bg.xyz*ratio,0.)/max(bg.w,1.);}}output[p]=value;}}");
    let shader=crate::timing::shader_module(&r.device,wgpu::ShaderModuleDescriptor {label:Some("catcher film composition"),source:wgpu::ShaderSource::Wgsl(source_text.into())});
    let entries=[pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Uniform),pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:true}),pipelines::buffer_entry(2,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:true}),pipelines::buffer_entry(3,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false})];
    let group=r.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("catcher film composition"),entries:&entries});
    let layout=r.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {label:Some("catcher film composition"),bind_group_layouts:&[Some(&group)],immediate_size:0});
    let pipeline=r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {label:Some("catcher film composition"),layout:Some(&layout),module:&shader,entry_point:Some("main"),compilation_options:Default::default(),cache:r.cache.as_ref()});
    let output=s.transients.buffer(&r.device,"catcher film colour",u64::from(w)*u64::from(h)*16,wgpu::BufferUsages::STORAGE);
    let entries=[entry(0,uniform.as_entire_binding()),entry(1,source.as_entire_binding()),entry(2,targets.catcher.as_entire_binding()),entry(3,output.as_entire_binding())];
    let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("catcher film composition"),layout:&group,entries:&entries});
    Ok((output,crate::graph::Dispatch::new(&pipeline,bind,[w.div_ceil(group_width),h.div_ceil(group_width),1])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json,Value};
    fn scene()->Value {
        json!({"version":2,"composition":{"space":"3d","view":{"program":"std/standard","inputs":{}}},"render":{"dither":false,"resolution":[16,16],"background":"transparent",
            "settings":{"integrator":"path","samples":64,"denoiser":null}},
            "buffers":{"p":{"type":"vec3","data":[[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]]},"n":{"type":"vec3","data":[[0,0,1],[0,0,1],[0,0,1],[0,0,1]]},"i":{"type":"u32","data":[0,1,2,0,2,3]}},
            "geometry":{"quad":{"attributes":{"POSITION":"p","NORMAL":"n"},"indices":"i"}},
            "materials":{"matte":{"inputs":{"base_color":[0.5,0.5,0.5],"specular_weight":0}}},
            "nodes":[{"type":"camera","name":"camera","translation":[0,0,3],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":2}}},
                {"type":"mesh","geometry":"quad","materials":["matte"],"is_shadow_catcher":true},
                {"type":"light","shape":"environment","energy":0},
                {"type":"light","shape":"sun","energy":3}]})
    }
    fn render(r:&Renderer,input:Value,deliver_ratio:bool)->Result<FrameOutput> {
        let scene=r.install(vork_scene::v2::from_test_value(input)?,Path::new("."))?;
        if scene.has_catchers {assert!(!scene.path.as_ref().unwrap().adaptive(),"catchers must accumulate every sample");}
        let mut slot=scene.create_frame_slot(r,ReadbackPrecision::Float)?;
        slot.submit_outputs(&scene,r,0.,ReadbackOutputs {aovs:deliver_ratio,float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
        let output=slot.collect(r)?;
        Ok(output)
    }
    #[test]
    #[ignore="requires real GPU with ray queries and pinned Slang"]
    fn neutral_catcher_and_unlit_baseline_are_finite_and_channel_selection_preserves_colour()->Result<()> {
        let r = crate::test_renderer()?;
        let input=scene();
        let with=render(&r,input.clone(),true)?;
        let without=render(&r,input.clone(),false)?;
        crate::test_linear::assert_linear_eq(with.linear_rgba.as_ref().unwrap(),without.linear_rgba.as_ref().unwrap());
        assert!(with.linear_rgba.unwrap().iter().all(|p|*p==[0.;4]));
        for p in &with.aovs["shadow_catcher"].pixels {for c in &p[..3] {assert!((*c-1.).abs()<1e-5,"neutral catcher: {p:?}");}}
        let mut dark=input;dark["nodes"].as_array_mut().unwrap().pop();
        let dark=render(&r,dark,true)?;
        assert!(dark.aovs["shadow_catcher"].pixels.iter().all(|p|p[..3]==[1.,1.,1.]));
        Ok(())
    }
    #[test]
    #[ignore="requires real GPU with ray queries and pinned Slang"]
    fn holdout_and_opaque_catcher_match_film_contract()->Result<()> {
        let r = crate::test_renderer()?;
        let mut input=scene();
        input["render"]["background"]=json!([0.2,0.3,0.4]);
        let catcher=render(&r,input.clone(),true)?.linear_rgba.unwrap();
        for pixel in catcher {for (actual,expected) in pixel.into_iter().zip([0.2,0.3,0.4,1.]) {assert!((actual-expected).abs()<0.002,"{pixel:?}");}}
        input["nodes"][1]["is_shadow_catcher"]=false.into();input["nodes"][1]["is_holdout"]=true.into();
        let opaque=render(&r,input.clone(),true)?.linear_rgba.unwrap();
        assert!(opaque.iter().all(|p|*p==[0.,0.,0.,1.]));
        input["render"]["background"]=json!("transparent");
        let transparent=render(&r,input,true)?.linear_rgba.unwrap();
        assert!(transparent.iter().all(|p|*p==[0.;4]));
        Ok(())
    }
    #[test]
    #[ignore="requires real GPU with ray queries and pinned Slang"]
    fn shadow_occlusion_and_added_cg_light_are_unclamped()->Result<()> {
        let r = crate::test_renderer()?;
        let mut input=scene();
        input["materials"]["black"]=json!({"inputs":{"base_color":[0,0,0],"specular_weight":0}});
        input["nodes"].as_array_mut().unwrap().push(json!({"type":"mesh","geometry":"quad","materials":["black"],"translation":[0,0,1],"scale":[0.15,0.15,1],"visible_camera":false}));
        let shadow=render(&r,input.clone(),true)?;
        let ratio=&shadow.aovs["shadow_catcher"].pixels;
        assert!(ratio[8*16+8][0]<0.02,"CG casts shadow: {:?}",ratio[8*16+8]);
        assert!((ratio[1*16+1][0]-1.).abs()<0.02,"outside shadow is neutral");
        input["materials"]["black"]=json!({"inputs":{"base_weight":0,"specular_weight":0,"emission_color":[1,0,0],"emission_luminance":40}});
        let last=input["nodes"].as_array().unwrap().len()-1;
        input["nodes"][last]["visible_shadow"]=false.into();
        input["nodes"][last]["rotation"]=json!({"euler":[std::f32::consts::PI,0,0]});
        let lit=render(&r,input,true)?;
        let ratio=lit.aovs["shadow_catcher"].pixels[8*16+8];
        assert!(ratio[0]>1.2,"added CG light exceeds one: {ratio:?}");
        assert!(ratio[..3].iter().all(|c|c.is_finite()));
        let mut unlit=scene();
        unlit["nodes"].as_array_mut().unwrap().pop();
        unlit["materials"]["emitter"]=json!({"inputs":{"base_weight":0,"specular_weight":0,"emission_color":[1,0,0],"emission_luminance":40}});
        unlit["nodes"].as_array_mut().unwrap().push(json!({"type":"mesh","geometry":"quad","materials":["emitter"],"translation":[0,0,1],"scale":[0.15,0.15,1],"rotation":{"euler":[std::f32::consts::PI,0,0]},"visible_camera":false}));
        let unlit=render(&r,unlit,true)?;
        assert!(unlit.aovs["shadow_catcher"].pixels.iter().all(|p|p[..3]==[1.,1.,1.]));
        // Film opacity must not change the filtered RGB pass at catcher edges.
        let mut edge=scene();edge["nodes"][1]["scale"]=json!([0.25,0.25,1]);
        edge["materials"]["black"]=json!({"inputs":{"base_color":[0,0,0],"specular_weight":0}});
        edge["nodes"].as_array_mut().unwrap().push(json!({"type":"mesh","geometry":"quad","materials":["black"],"translation":[0,0,1],"scale":[0.4,0.4,1],"visible_camera":false}));
        let transparent=render(&r,edge.clone(),true)?;
        edge["render"]["background"]=json!([0.2,0.3,0.4]);
        let opaque=render(&r,edge,true)?;
        for (a,b) in transparent.aovs["shadow_catcher"].pixels.iter().zip(&opaque.aovs["shadow_catcher"].pixels) {
            assert_eq!(&a[..3],&b[..3],"film-independent catcher edges");
        }
        Ok(())
    }
    #[test]
    #[ignore="requires real GPU with ray queries and pinned Slang"]
    fn implicit_geometry_casts_catcher_shadows() -> Result<()> {
        let r = crate::test_renderer()?;
        let mut input=scene();
        input["buffers"]["field"]=json!({"type":"f32","dims":[24,24,24]});
        input["geometry"]["sphere"]=json!({"topology":"implicit","attributes":{"phi":"field"}});
        input["programs"]=json!({"seed":{"schedule":"install","dispatch":13824,"bindings":{"field":"field"},
            "source":"struct Inputs {}; void compute(ComputeContext ctx,Inputs i) { uint n=ctx.id.x; uint3 c=uint3(n%24,(n/24)%24,n/576); float3 p=(float3(c)+.5)/12.-1.; field_store(n,length(p)-.65); }"}});
        input["materials"]["black"]=json!({"inputs":{"base_color":[0,0,0],"specular_weight":0}});
        input["nodes"].as_array_mut().unwrap().push(json!({"type":"mesh","geometry":"sphere","materials":["black"],"translation":[0,0,1],"scale":[0.4,0.4,0.4],"visible_camera":false}));
        let output=render(&r,input,true)?;
        let ratio=&output.aovs["shadow_catcher"].pixels;
        assert!(ratio[8*16+8][0]<0.02,"implicit CG must be absent from the reference transport: {:?}",ratio[8*16+8]);
        assert!((ratio[1*16+1][0]-1.).abs()<0.02,"outside the implicit shadow is neutral");
        assert!(output.linear_rgba.unwrap().iter().all(|p|*p==[0.;4]),"catcher stays transparent");
        Ok(())
    }
    #[test]
    fn matte_profiles_select_full_transport_and_preserve_sample_budget()->Result<()> {
        let mut input:Scene=vork_scene::v2::from_test_value(scene())?;
        input.render.settings.integrator=Default::default();
        let samples=input.render.settings.samples;
        promote(&mut input);
        assert!(input.render.settings.integrator.is_path());
        assert_eq!(input.render.settings.samples,samples);
        Ok(())
    }
}
