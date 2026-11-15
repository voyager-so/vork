use super::*;
use serde_json::{Value, json};

fn scene() -> Value {
    json!({"version":2,"composition":{"space":"3d","view":{"program":"std/standard","inputs":{}}},"render":{"dither":false,"resolution":[32,32],"background":[0,0,0],"settings":{"integrator":"raster","samples":1}},
        "buffers":{"p":{"type":"vec3","data":[[-0.4,-0.6,0],[0.4,-0.6,0],[0,0.6,0]]},
            "translation":{"type":"vec3","data":[[-0.5,0,0],[0.5,0,0]]},
            "paint":{"type":"vec4","data":[[0.8,0.2,0.1,1],[0.1,0.6,0.9,1]]},
            "offset":{"type":"f32","data":[0.1,-0.1]}},
        "geometry":{"g":{"attributes":{"POSITION":"p"}}},
        "materials":{"m":{"inputs":{"base_weight":0,"specular_weight":0,"emission_luminance":1,"emission_color":{"attribute":"_paint","factor":[0.5,1,0.5]}}}},
        "nodes":[{"type":"mesh","name":"instances","geometry":"g","materials":["m"],"instances":{"attributes":{"TRANSLATION":"translation","_paint":"paint","_offset":"offset"}}},
            {"type":"camera","translation":[0,0,3],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":2}}}]})
}

#[test]
fn instance_attribute_validation() -> Result<()> {
    let parse = |v: Value| -> Result<()> { let s: Scene=vork_scene::v2::from_test_value(v)?; crate::validate(&s) };
    parse(scene())?;
    let mut v=scene(); v["nodes"][0]["instances"]["attributes"]["_paint"]=json!("missing");
    let error = format!("{:#}", parse(v).unwrap_err());
    assert!(error.contains("unknown attribute buffer \"missing\""), "{error}");
    let mut v=scene(); v["geometry"]["g"]["attributes"]["_paint"]=json!("p");
    assert!(format!("{:#}",parse(v).unwrap_err()).contains("both geometry and instances"));
    let mut v=scene(); v["nodes"][0]["instances"]["attributes"].as_object_mut().unwrap().remove("_paint");
    assert!(format!("{:#}",parse(v).unwrap_err()).contains("missing attribute"));
    let mut v=scene(); v["buffers"]["paint"]=json!({"type":"u32","data":[1,2]});
    assert!(parse(v).is_err());
    let mut v=scene(); v["programs"]=json!({"update":{"source":"struct Inputs {}; void compute(ComputeContext ctx, Inputs i) {}","bindings":{"paint":"paint"},"dispatch":1,"schedule":"frame"}});
    parse(v)?;
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn instance_attributes_match_split_meshes_in_surface_and_displacement() -> Result<()> {
    let r = crate::test_renderer()?;
    let render=|value:Value| -> Result<_> {
        let mut installed=r.install(vork_scene::v2::from_test_value(value)?,Path::new("."))?;
        let mut slot=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
        installed.render_frame(&r, &mut slot, 0.)
    };
    for stage in ["table", "computed", "typed", "displacement", "path"] {
        let mut value=scene();
        if stage=="path" {
            if !r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | crate::rt::HIT_BUFFER_FEATURES) {
                continue;
            }
            value["render"]["settings"]=json!({"integrator":"path","denoiser":null,"samples":1});
        }
        if stage=="computed" {
            value["programs"]=json!({"move":{"source":"struct Inputs {}; void compute(ComputeContext ctx,Inputs i) { translation_store(ctx.id.x, float3(ctx.id.x==0u ? -.5 : .5,0.,0.)); }", "bindings":{"translation":"translation"}, "dispatch":2, "schedule":"frame"}});
        }
        if stage=="typed" || stage=="displacement" {
            let displacement=if stage=="displacement" { "float3 displace(DisplacementContext ctx, Inputs i) { return float3(read_attribute(i.offset,ctx),0,0); }" } else { "" };
            value["programs"]=json!({"paint":{"source":format!("struct Inputs {{Attribute<float3> tint; Attribute<float> offset;}}; Closure surface(ShadingContext ctx,Inputs i) {{let a=read_attribute(i.tint,ctx); return unlit(a.value+abs(a.dx)+abs(a.dy)+float3(ctx.primitive_id));}} {displacement}")}});
            value["materials"]["m"]=json!({"program":"paint","inputs":{"tint":{"attribute":"_paint","factor":[0.5,1,0.5]},"offset":{"attribute":"_offset"}}});
        }
        let actual=render(value.clone())?;
        // Independent reference: two separately placed meshes with constant vertex
        // attributes. Displacement receives a distinct offset for each mesh too.
        let mut reference=value;
        if stage=="computed" { reference.as_object_mut().unwrap().remove("programs"); }
        let mut nodes=vec![];
        for (index, (translation, paint, offset)) in [(-0.5,[0.8,0.2,0.1,1.],0.1),(0.5,[0.1,0.6,0.9,1.],-0.1)].into_iter().enumerate() {
            let name=format!("g{index}");let color=format!("paint{index}");let height=format!("offset{index}");
            reference["buffers"][&color]=json!({"type":"vec4","data":[paint,paint,paint]});
            reference["buffers"][&height]=json!({"type":"f32","data":[offset,offset,offset]});
            reference["geometry"][&name]=json!({"attributes":{"POSITION":"p","_paint":color,"_offset":height}});
            nodes.push(json!({"type":"mesh","name":name,"geometry":name,"materials":["m"],"translation":[translation,0,0]}));
        }
        nodes.push(reference["nodes"][1].clone()); reference["nodes"]=json!(nodes);
        let expected=render(reference)?;
        crate::test_linear::assert_linear_eq(actual.linear_rgba.as_ref().expect("linear readback"), expected.linear_rgba.as_ref().expect("linear readback"));
        crate::test_linear::assert_depth_eq(actual.depth_f32.as_ref().expect("float depth readback"), expected.depth_f32.as_ref().expect("float depth readback"));
        assert!(actual.rgba.chunks_exact(4).any(|p|p[0]>50 && p[0]>p[1]),"left instance visible");
        assert!(actual.rgba.chunks_exact(4).any(|p|p[1]>50 && p[1]>p[0]),"right instance visible");
    }
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn zero_scale_geometry_is_absent_and_animation_stays_finite() -> Result<()> {
    let r = crate::test_renderer()?;
    let render=|value:Value,time:f32| -> Result<FrameOutput> {
        let installed=r.install(vork_scene::v2::from_test_value(value)?,Path::new("."))?;
        let mut slot=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
        slot.submit_outputs(&installed,&r,time,ReadbackOutputs {color:true,depth_ids:true,normals:true,motion:true,aovs:true,float:FloatOutputs::ALL,..Default::default()})?;
        slot.collect(&r)
    };
    let mut empty=scene();empty["nodes"].as_array_mut().unwrap().remove(0);
    let absent=render(empty,0.)?;
    let visible=render(scene(),0.)?;
    assert_ne!(visible.rgba,absent.rgba);
    // Z alone collapses an XY triangle without reducing its raster coverage.
    for scale in [[0.,1.,1.],[1.,0.,1.],[1.,1.,0.],[0.,0.,0.]] {
        for parent in [false,true] {
            let mut value=scene();
            if parent {
                let mut mesh=value["nodes"][0].clone();
                mesh["rotation"]=json!({"euler":[0.3,0.4,0.7]});
                value["nodes"][0]=json!({"type":"node","scale":scale,"children":[mesh]});
            } else {value["nodes"][0]["scale"]=json!(scale);}
            let actual=render(value,0.)?;
            crate::test_linear::assert_linear_eq(actual.linear_rgba.as_ref().expect("linear readback"), absent.linear_rgba.as_ref().expect("linear readback"));
            assert_eq!(actual.ids,absent.ids);
            crate::test_linear::assert_depth_eq(actual.depth_f32.as_ref().expect("float readback"), absent.depth_f32.as_ref().expect("float readback"));
            assert!(actual.linear_rgba.unwrap().iter().flatten().all(|x|x.is_finite()));
        }
    }
    let mut value=scene();
    value["buffers"]["scale"]=json!({"type":"vec3","data":[[1,1,0],[0,1,1]]});
    value["nodes"][0]["instances"]["attributes"]["SCALE"]=json!("scale");
    crate::test_linear::assert_linear_eq((render(value,0.)?).linear_rgba.as_ref().expect("linear readback"), (absent).linear_rgba.as_ref().expect("linear readback"));
    let mut mixed=scene();
    mixed["buffers"]["scale"]=json!({"type":"vec3","data":[[1,1,0],[1,1,1]]});
    mixed["nodes"][0]["instances"]["attributes"]["SCALE"]=json!("scale");
    let mut reference=scene();reference["buffers"]["translation"]["data"][0]=json!([5,0,0]);
    crate::test_linear::assert_linear_eq((render(mixed,0.)?).linear_rgba.as_ref().expect("linear readback"), (render(reference,0.)?).linear_rgba.as_ref().expect("linear readback"));
    let mut aov=scene();
    aov["render"]["aovs"]=json!([{"name":"heat","type":"value"}]);
    aov["programs"]=json!({"heat":{"source":"import vork; struct Inputs {Aov heat;}; Closure surface(ShadingContext ctx,Inputs i) {return unlit(float3(1.));} void aov(ShadingContext ctx,Inputs i,inout AovOutputs o) {write_aov(o,i.heat,1.);}"}});
    aov["materials"]["m"]=json!({"program":"heat","inputs":{"heat":{"aov":"heat"}}});
    assert!(render(aov.clone(),0.)?.aovs["heat"].pixels.iter().any(|p|p[0]>0.));
    aov["nodes"][0]["scale"]=json!([1,1,0]);
    assert!(render(aov,0.)?.aovs["heat"].pixels.iter().flatten().all(|x|*x==0.));
    let mut value=scene();
    value["animation"]=json!([{"tracks":[{"path":["nodes","instances","scale"],"keys":[[0,[0,0,0]],[1,[1,1,1]]]}]}]);
    value["render"]["fps"]=json!(1);
    crate::test_linear::assert_linear_eq((render(value.clone(),0.)?).linear_rgba.as_ref().expect("linear readback"), (absent).linear_rgba.as_ref().expect("linear readback"));
    let grown=render(value.clone(),1.)?;
    crate::test_linear::assert_linear_eq(grown.linear_rgba.as_ref().expect("linear readback"), visible.linear_rgba.as_ref().expect("linear readback"));
    assert!(grown.motion_f32.unwrap().iter().flatten().all(|x|x.abs()<1e-5),"absent previous endpoint uses visible pose");
    value["render"]["motion_blur"]=json!({"shutter":1,"position":"center"});
    value["render"]["settings"]["samples"]=json!(4);

    let blurred=render(value,0.5)?;
    assert!(blurred.linear_rgba.unwrap().iter().flatten().all(|x|x.is_finite()));
    assert!(blurred.motion_f32.unwrap().iter().flatten().all(|x|x.is_finite()));
    Ok(())
}

#[test]
fn zero_scale_validation_keeps_nonfinite_and_zero_rotation_errors() -> Result<()> {
    let mut value:Scene=vork_scene::v2::from_test_value(scene())?;
    value.nodes[0].transform.scale=[0.,1.,1.];
    crate::validate(&value)?;
    for invalid in [f32::NAN,f32::INFINITY,f32::NEG_INFINITY] {
        value.nodes[0].transform.scale=[invalid,1.,1.];
        assert!(crate::validate(&value).is_err());
    }
    value.nodes[0].transform.scale=[0.,1.,1.];
    // Rotations are normalized: rounded quaternions stay valid, zero length does not.
    value.nodes[0].transform.rotation=Some([0.,0.,0.,2.]);
    crate::validate(&value)?;
    value.nodes[0].transform.rotation=Some([0.,0.,0.,0.]);
    assert!(crate::validate(&value).is_err());
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn zero_scale_casts_no_shadow_and_lights_ignore_scale() -> Result<()> {
    let r = crate::test_renderer()?;
    let render=|value:Value| -> Result<_> {
        let mut installed=r.install(vork_scene::v2::from_test_value(value)?,Path::new("."))?;
        let mut slot=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
        installed.render_frame(&r, &mut slot, 0.)
    };
    let mut value=scene();

    value["materials"]["floor"]=json!({"inputs":{"base_color":[0.8,0.8,0.8],"specular_weight":0}});
    value["buffers"]["floor"]=json!({"type":"vec3","data":[[-2,-2,-0.5],[2,-2,-0.5],[2,2,-0.5],[-2,-2,-0.5],[2,2,-0.5],[-2,2,-0.5]]});
    value["geometry"]["floor"]=json!({"attributes":{"POSITION":"floor"}});
    value["nodes"].as_array_mut().unwrap().extend([
        json!({"type":"mesh","geometry":"floor","materials":["floor"]}),
        json!({"type":"light","shape":"sun","energy":3,"angle":0})]);
    let shadowed=render(value.clone())?;
    let mut empty=value.clone();empty["nodes"].as_array_mut().unwrap().remove(0);
    let expected=render(empty)?;
    assert_ne!(shadowed.rgba,expected.rgba);
    value["nodes"][0]["visible_camera"]=json!(false);
    assert_ne!(render(value.clone())?.rgba,expected.rgba,"caster really shadows receiver");
    value["nodes"][0]["scale"]=json!([1,1,0]);
    crate::test_linear::assert_linear_eq((render(value.clone())?).linear_rgba.as_ref().expect("linear readback"), (expected).linear_rgba.as_ref().expect("linear readback"));
    value["nodes"][3]["scale"]=json!([0,0,0]);
    crate::test_linear::assert_linear_eq((render(value)?).linear_rgba.as_ref().expect("linear readback"), (expected).linear_rgba.as_ref().expect("linear readback"));
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn zero_scale_path_geometry_and_emission_are_absent() -> Result<()> {
    let r = crate::test_renderer()?;
    if !r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | crate::rt::HIT_BUFFER_FEATURES) {return Ok(())}
    let render=|value:Value| -> Result<_> {
        let mut installed=r.install(vork_scene::v2::from_test_value(value)?,Path::new("."))?;
        let mut slot=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
        installed.render_frame(&r, &mut slot, 0.)
    };
    let mut value=scene();
    value["render"]["settings"]=json!({"integrator":"path","denoiser":null,"samples":4});
    value["nodes"][0]["visible_camera"]=json!(false);
    value["materials"]["floor"]=json!({"inputs":{"base_color":[0.8,0.8,0.8],"specular_weight":0}});
    value["buffers"]["floor"]=json!({"type":"vec3","data":[[-2,-2,-0.5],[2,-2,-0.5],[2,2,-0.5],[-2,-2,-0.5],[2,2,-0.5],[-2,2,-0.5]]});
    value["geometry"]["floor"]=json!({"attributes":{"POSITION":"floor"}});
    value["nodes"].as_array_mut().unwrap().push(json!({"type":"mesh","geometry":"floor","materials":["floor"]}));
    let mut empty=value.clone();empty["nodes"].as_array_mut().unwrap().remove(0);
    let absent=render(empty)?;
    assert_ne!(render(value.clone())?.rgba,absent.rgba,"emitter illuminates receiver");
    let mut instances=value.clone();
    instances["buffers"]["scale"]=json!({"type":"vec3","data":[[1,1,0],[1,1,0]]});
    instances["nodes"][0]["instances"]["attributes"]["SCALE"]=json!("scale");
    crate::test_linear::assert_linear_eq((render(instances)?).linear_rgba.as_ref().expect("linear readback"), (absent).linear_rgba.as_ref().expect("linear readback"));
    value["nodes"][0]["scale"]=json!([1,1,0]);
    crate::test_linear::assert_linear_eq((render(value)?).linear_rgba.as_ref().expect("linear readback"), (absent).linear_rgba.as_ref().expect("linear readback"));
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn zero_scale_skinned_displacement_stays_finite() -> Result<()> {
    let r = crate::test_renderer()?;
    let mut value=scene();
    value["buffers"]["joints"]=json!({"type":"uvec4","data":[[0,0,0,0],[0,0,0,0],[0,0,0,0]]});
    value["buffers"]["weights"]=json!({"type":"vec4","data":[[1,0,0,0],[1,0,0,0],[1,0,0,0]]});
    value["buffers"]["bind"]=json!({"type":"mat4","data":[Mat4::IDENTITY.to_cols_array()]});
    value["geometry"]["g"]["attributes"]["JOINTS_0"]=json!("joints");
    value["geometry"]["g"]["attributes"]["WEIGHTS_0"]=json!("weights");
    value["nodes"][0]["skin"]=json!({"joints":["joint"],"inverse_bind_matrices":"bind"});
    let mesh=value["nodes"][0].clone();
    value["nodes"][0]=json!({"type":"node","name":"rig","children":[mesh,{"type":"node","name":"joint"}]});
    value["programs"]=json!({"skin":{"source":"import vork; struct Inputs {}; float3 displace(DisplacementContext ctx,Inputs i) {return ctx.NORMAL*.1;} Closure surface(ShadingContext ctx,Inputs i) {return unlit(float3(1.));}"}});
    value["materials"]["m"]=json!({"program":"skin"});
    value["animation"]=json!([{"tracks":[{"path":["nodes","rig","scale"],"keys":[[0,[0,0,0]],[1,[1,1,1]]]}]}]);
    let installed=r.install(vork_scene::v2::from_test_value(value)?,Path::new("."))?;
    let mut slot=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
    for time in [0.,1.,0.] {
        slot.submit_outputs(&installed,&r,time,ReadbackOutputs {color:true,float:FloatOutputs::ALL,..Default::default()})?;
        let frame=slot.collect(&r)?;
        let pixels=frame.linear_rgba.unwrap();
        assert!(pixels.iter().flatten().all(|x|x.is_finite()));
        assert_eq!(pixels.iter().any(|p|p[0]>0.1),time==1.);
    }
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn zero_scale_imported_parent_hides_rotated_static_and_animated_meshes() -> Result<()> {
    let r = crate::test_renderer()?;
    let root=std::env::temp_dir().join(format!("vork-zero-scale-import-{}",std::process::id()));
    std::fs::create_dir_all(&root)?;
    let positions=[[-0.6f32,-0.6,0.],[0.6,-0.6,0.],[0.,0.6,0.]];
    std::fs::write(root.join("mesh.bin"),bytemuck::cast_slice(&positions))?;
    let document=json!({"asset":{"version":"2.0"},"buffers":[{"uri":"mesh.bin","byteLength":36}],
        "bufferViews":[{"buffer":0,"byteLength":36}],
        "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[-0.6,-0.6,0],"max":[0.6,0.6,0]}],
        "materials":[{"name":"m"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]}],
        "nodes":[{"name":"parent","scale":[1,1,0],"children":[1]},
            {"name":"triangle","mesh":0,"rotation":glam::Quat::from_euler(glam::EulerRot::XYZ,0.3,0.4,0.7).to_array()}],
        "scenes":[{"nodes":[0]}],"scene":0});
    std::fs::write(root.join("mesh.gltf"),serde_json::to_vec(&document)?)?;
    let mut value=scene();
    value["nodes"][0]=json!({"type":"asset","name":"asset","file":"mesh.gltf","materials":{"*":"m"}});
    value["materials"]["m"]=json!({"inputs":{"base_weight":0,"specular_weight":0,"emission_luminance":1,"emission_color":[1,1,1]}});
    for animated in [false,true] {
        if animated {
            value["animation"]=json!([{"tracks":[{"path":["nodes","asset","nodes","parent","scale"],"keys":[[0,[1,1,0]],[1,[1,1,1]]]}]}]);
        }
        let installed=r.install(vork_scene::v2::from_test_value(value.clone())?,&root)?;
        let mut slot=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
        for time in if animated {vec![0.,1.,0.]} else {vec![0.]} {
            slot.submit_outputs(&installed,&r,time,ReadbackOutputs {color:true,float:FloatOutputs::ALL,..Default::default()})?;
            let pixels=slot.collect(&r)?.linear_rgba.unwrap();
            assert!(pixels.iter().flatten().all(|x|x.is_finite()));
            assert_eq!(pixels.iter().any(|p|p[0]>0.1),animated && time==1.);
        }
    }
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn simulation_pixels_are_independent_of_seek_slots_and_fps() -> Result<()> {
    let r = crate::test_renderer()?;
    let mut value=scene();
    value["simulation"]=json!({"t0":0,"dt":0.015625,"seed":7,"state":["translation"],"init":["initialize"],"steps":["step"]});
    value["programs"]=json!({
        "initialize":{"source":"struct Inputs {}; void compute(ComputeContext c,Inputs i){translation_store(c.id.x,float3(c.id.x==0 ? -.5 : .5,0,0));}","bindings":{"translation":"translation"},"dispatch":2,"schedule":"simulation"},
        "step":{"source":"struct Inputs {}; void compute(ComputeContext c,Inputs i){translation_store(c.id.x,translation_load(c.id.x)+float3(0,c.dt*.25,0));}","bindings":{"translation":"translation"},"dispatch":2,"schedule":"simulation"}});
    let render_fresh=|mut value:Value,fps:f32| -> Result<FrameOutput> {
        value["render"]["fps"]=json!(fps);
        let installed=r.install(vork_scene::v2::from_test_value(value)?,Path::new("."))?;
        let mut slot=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
        slot.submit(&installed,&r,1.)?; slot.collect(&r)
    };
    let mut shutter_value=value.clone();
    shutter_value["render"]["fps"]=json!(1);
    shutter_value["render"]["motion_blur"]=json!({"shutter":0.5,"position":"center"});

    shutter_value["render"]["settings"]["samples"]=json!(8);
    let expected=render_fresh(value.clone(),24.)?;
    crate::test_linear::assert_linear_eq((expected).linear_rgba.as_ref().expect("linear readback"), (render_fresh(value.clone(),60.)?).linear_rgba.as_ref().expect("linear readback"));
    let installed=r.install(vork_scene::v2::from_test_value(value)?,Path::new("."))?;
    let mut slots=(0..3).map(|_|installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)).collect::<Result<Vec<_>>>()?;
    // Queue several output frames before collecting; each bridge must snapshot its own state.
    for times in [[1.,0.,1.], [0.5,0.25,1.], [1.01,0.99,1.]] {
        for (slot,time) in slots.iter_mut().zip(times) {slot.submit(&installed,&r,time)?;}
        for (slot,time) in slots.iter_mut().zip(times) {
            let output=slot.collect(&r)?;
            if time==1. || time==1.01 {crate::test_linear::assert_linear_eq(output.linear_rgba.as_ref().expect("linear readback"), expected.linear_rgba.as_ref().expect("linear readback"));crate::test_linear::assert_depth_eq(output.depth_f32.as_ref().expect("float depth readback"), expected.depth_f32.as_ref().expect("float depth readback"));}
        }
    }
    for workers in [1,3] {
        let outputs=std::thread::scope(|scope| {
            let jobs:Vec<_>=(0..workers).map(|worker| {
                let installed=&installed; let r=&r;
                scope.spawn(move || -> Result<FrameOutput> {
                    let mut slot=installed.create_frame_slot(r,crate::ReadbackPrecision::Float)?;
                    for time in [worker as f32*0.125,0.75,0.,1.] {
                        slot.submit(installed,r,time)?;
                        let output=slot.collect(r)?;
                        if time==1. {return Ok(output);}
                    }
                    unreachable!()
                })
            }).collect();
            jobs.into_iter().map(|job|job.join().unwrap()).collect::<Result<Vec<_>>>()
        })?;
        for output in outputs {crate::test_linear::assert_linear_eq(output.linear_rgba.as_ref().expect("linear readback"), expected.linear_rgba.as_ref().expect("linear readback"));}
    }
    let reference=render_fresh(shutter_value.clone(),1.)?;
    let blurred=r.install(vork_scene::v2::from_test_value(shutter_value)?,Path::new("."))?;
    let mut slot=blurred.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
    for time in [1.,0.,1.75,0.5,1.] {
        slot.submit(&blurred,&r,time)?;let output=slot.collect(&r)?;
        if time==1. {crate::test_linear::assert_linear_eq(output.linear_rgba.as_ref().expect("linear readback"), reference.linear_rgba.as_ref().expect("linear readback"));}
    }
    Ok(())
}
