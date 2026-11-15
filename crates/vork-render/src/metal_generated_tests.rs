//! GPU-free compilation of scene-installed kernels omitted by native::ALL.
use super::*;
use serde_json::json;

#[test]
#[ignore = "compiles installed MSL; run scripts/check-metal-generated"]
fn installed_scene_modules() -> Result<()> {
    let destination = std::env::var_os("VORK_INSTALLED_METAL_DIR")
        .expect("scripts/check-metal-generated supplies an output directory");
    std::fs::create_dir_all(&destination)?;
    let mut renderer = pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
    renderer.adapter_info.backend = wgpu::Backend::Metal;
    let mut field = Vec::new();
    for z in 0..8 { for y in 0..8 { for x in 0..8 {
        let p = [x,y,z].map(|v| 2. * v as f32 / 7. - 1.);
        field.push(p.iter().map(|v|v*v).sum::<f32>().sqrt() - 0.7);
    }}}
    let document = json!({"version":2,
        "buffers":{"field":{"type":"f32","dims":[8,8,8],"data":field}},
        "geometry":{"field":{"topology":"implicit","attributes":{"field":"field"}}},
        "materials":{
            "opaque":{"inputs":{"base_color":[0.8,0.2,0.1]}},
            "glass":{"inputs":{"base_weight":0,"transmission_weight":1,"specular_ior":1.333}}},
        "nodes":[
            {"type":"mesh","geometry":"field","materials":["opaque"],"translation":[-1,0,0]},
            {"type":"mesh","geometry":"field","materials":["glass"],"translation":[1,0,0]},
            {"type":"light","shape":"sun","energy":2},
            {"type":"camera","name":"camera","translation":[0,0,6],"target":[0,0,0]}],
        "render":{"camera":"camera","resolution":[16,16],"background":[0.1,0.1,0.1]}});
    // Also retain a portable fixture for real-device validation.
    std::fs::write(PathBuf::from(&destination).join("implicit.json"), serde_json::to_vec_pretty(&document)?)?;
    let trace_document = json!({"version":2,
        "buffers":{
            "p":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]},
            "n":{"type":"vec3","data":[[0,0,1],[0,0,1],[0,0,1],[0,0,1]]},
            "i":{"type":"u32","data":[0,1,2,0,2,3]}},
        "geometry":{"quad":{"attributes":{"POSITION":"p","NORMAL":"n"},"indices":"i"}},
        "programs":{"nested":{"source":r#"
            import vork;
            struct Inputs {};
            TraceHit inner_trace(float3 p,float3 n) {return trace(p,n,0.001,10.);}
            float outer_trace(ShadingContext ctx) {return inner_trace(ctx.POSITION,ctx.NORMAL).hit ? 0.25 : 0.75;}
            Closure surface(ShadingContext ctx,Inputs inputs) {
                let direct=trace(ctx.POSITION,-ctx.NORMAL,0.001,10.);
                Closure c={};c.base_color=float3(outer_trace(ctx),direct.hit ? 0.2 : 0.8,0.3);return c;
            }
        "#}},
        "materials":{"nested":{"program":"nested"},"alpha":{"inputs":{"geometry_opacity":0.5}}},
        "nodes":[
            {"type":"mesh","geometry":"quad","materials":["nested"]},
            {"type":"mesh","geometry":"quad","materials":["alpha"],"translation":[0,0,-1]},
            {"type":"light","shape":"sun","energy":2},
            {"type":"camera","name":"camera","translation":[0,0,4],"target":[0,0,0]}],
        "render":{"camera":"camera","resolution":[16,16],"background":[0.1,0.1,0.1]}});
    std::fs::write(PathBuf::from(&destination).join("nested-trace.json"), serde_json::to_vec_pretty(&trace_document)?)?;
    for (name, document) in [("implicit", document), ("nested trace", trace_document)] {
    for profile in ["std/final", "std/path"] {
        let mut scene: vork_scene::v2::Scene = vork_scene::v2::from_test_value(document.clone())?;
        scene.render.settings = vork_scene::Profile::load(profile)?.resolve(&Default::default())?;
        let installed = renderer.install_with_options(scene, Path::new("."),
            InstallOptions { variants: pipelines::VariantCompile::Setup, ..Default::default() })
            .with_context(|| format!("installed Metal scene: {name} / {profile}"))?;
        let _slot = installed.create_frame_slot(&renderer, Default::default())?;
    }
    }
    renderer.finish_shader_warmup()
}
