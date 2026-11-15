use crate::*;

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn surface_queries_match_analytic_fields_in_path_and_raster() -> Result<()> {
    let r = crate::test_renderer()?;
    let mut original:serde_json::Value=serde_json::from_str(include_str!("../tests/fixtures/compute-trace.json"))?;
    let source=original["programs"]["bake"]["source"].as_str().unwrap()
        .replace("void compute(ComputeContext ctx,Inputs i)","Closure surface(ShadingContext ctx,Inputs i)")
        .replace("color_store(ctx.id.x,good ? float3(0.,1.,0.) : float3(1.,0.,0.));","return unlit(good ? float3(0.,1.,0.) : float3(1.,0.,0.));");
    original["programs"].as_object_mut().unwrap().remove("bake");
    original["programs"]["show"]["source"]=serde_json::json!(source);
    original["geometry"]["screen"]["attributes"].as_object_mut().unwrap().remove("COLOR_0");
    original["render"]["resolution"]=serde_json::json!([32,32]);
    for coverage in [false,true] {
    for kind in ["path","raster"] {
        let mut scene=original.clone();
        if coverage {scene["programs"]["show"]["source"]=serde_json::json!(source.replace("return unlit(good ? float3(0.,1.,0.) : float3(1.,0.,0.));","Closure c=unlit(good ? float3(0.,1.,0.) : float3(1.,0.,0.));c.geometry_opacity=good ? 1. : .5;return c;"));}
        vork_scene::v2::replace_integrator(&mut scene,serde_json::json!({"integrator":kind,"samples":1,"denoiser":null}));
        let render=|scene:serde_json::Value|->Result<crate::test_linear::LinearImage> {
            let installed=r.install(vork_scene::v2::from_test_value(scene)?,std::path::Path::new("."))?;
            let mut frame=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
            frame.submit(&installed,&r,0.)?;
            {let output=frame.collect(&r)?;Ok(crate::test_linear::LinearImage {rgba:output.rgba,linear:output.linear_rgba.expect("linear readback")})}
        };
        let queried=render(scene.clone())?;
        scene["programs"]["show"]["source"]=serde_json::json!("struct Inputs {};Closure surface(ShadingContext ctx,Inputs i){return unlit(float3(0.,1.,0.));}");
        let reference=render(scene)?;
        assert_eq!(queried,reference,"{kind}: the analytic boundary fields must match");
        assert!(queried.chunks_exact(4).any(|p|p[1]>100 && u16::from(p[1])*2>u16::from(p[0])*3),"{kind}: the witness must be visible");
    }
    }
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn hair_surface_queries_preserve_complete_camera_routing()->Result<()> {
    let r = crate::test_renderer()?;
    let mut scene:serde_json::Value=serde_json::from_str(include_str!("../tests/fixtures/compute-trace.json"))?;
    let source=scene["programs"]["bake"]["source"].as_str().unwrap();
    let body=source.split_once("bool good=true;").unwrap().1.split_once("#endif").unwrap().0;
    let helper=format!("bool query_contract(){{bool good=true;{body}\n#endif\nreturn good;}}\n");
    let hair=include_str!("../../../libraries/std/hair.slang").replace("Hair surface",&format!("{helper}Hair surface"));
    let queried=hair.replace("h.scattering = true;","h.scattering = query_contract();");
    let floor=scene["nodes"][0].clone();
    scene["nodes"]=serde_json::json!([floor,
        {"type":"mesh","geometry":"strand","materials":["hair"],"pass_index":2},
        {"type":"mesh","geometry":"strand","materials":["hair"],"pass_index":3,"visible_camera":false,"translation":[0,0,-0.6]},
        {"type":"camera","translation":[0,0,5],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":3.}}},
        {"type":"light","shape":"sun","energy":2.,"angle":0.}]);
    scene["buffers"]["strand_p"]=serde_json::json!({"type":"vec3","data":[[0,-1,0],[0,1,0]]});
    scene["buffers"]["strand_w"]=serde_json::json!({"type":"f32","data":[0.4,0.4]});
    scene["buffers"]["strand_c"]=serde_json::json!({"type":"u32","data":[2]});
    scene["geometry"]["strand"]=serde_json::json!({"topology":"curves","attributes":{"POSITION":"strand_p"},"widths":"strand_w","curve_vertex_counts":"strand_c"});
    scene["materials"]=serde_json::json!({"invisible":{"program":"invisible"},"hair":{"program":"query_hair","inputs":{"color":[0.5,0.3,0.1],"offset":0.}}});
    let invisible=scene["programs"]["invisible"].clone();scene["programs"]=serde_json::json!({"invisible":invisible,"query_hair":{"source":queried}});
    scene["render"]["resolution"]=serde_json::json!([32,32]);
    vork_scene::v2::replace_integrator(&mut scene,serde_json::json!({"integrator":"raster","samples":1,"denoiser":null}));
    let render=|scene:serde_json::Value|->Result<crate::test_linear::LinearImage> {
        let installed=r.install(vork_scene::v2::from_test_value(scene)?,std::path::Path::new("."))?;
        assert!(installed.plan.effects.path,"physical hair must retain complete-camera transport");
        let mut frame=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
        frame.submit(&installed,&r,0.)?;{let output=frame.collect(&r)?;Ok(crate::test_linear::LinearImage {rgba:output.rgba,linear:output.linear_rgba.expect("linear readback")})}
    };
    let actual=render(scene.clone())?;
    scene["programs"]["query_hair"]["source"]=serde_json::json!(hair);
    assert_eq!(actual,render(scene)?,"query-dependent hair scattering must match its analytic reference");
    assert!(actual.chunks_exact(4).any(|p|p[..3].iter().any(|&v|v>20)),"the hair witness must be visible");
    Ok(())
}
