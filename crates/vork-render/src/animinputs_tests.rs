//! Numeric inputs share transforms' evaluator, shutter sampling and seek contract.
use super::*;
use serde_json::json;

fn scene(program: bool, blur: bool) -> Result<Scene> {
    let material = if program {
        json!({"program":"paint","inputs":{"color":[1,0,0]}})
    } else {
        json!({"inputs":{"base_weight":0,"specular_weight":0,"emission_luminance":1,"emission_color":[1,0,0]}})
    };
    Ok(vork_scene::v2::from_test_value(json!({"version":2,
        "composition":{"space":"3d","view":{"program":"std/agx","inputs":{}}},"render":{"dither":false,"resolution":[4,4],"fps":1,"duration":2,
            "motion_blur":if blur {json!({"shutter":1,"position":"center"})}else{json!(null)},
            "settings":{"integrator":"raster","samples":16,"reflections":"environment"}},
        "nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":1}}},
            {"type":"mesh","geometry":"quad","materials":["m"]}],
        "buffers":{"p":{"type":"vec3","data":[[-20,-20,-2],[20,-20,-2],[20,20,-2],[-20,20,-2]]},
            "n":{"type":"vec3","data":[[0,0,1],[0,0,1],[0,0,1],[0,0,1]]},"i":{"type":"u32","data":[0,1,2,0,2,3]}},
        "geometry":{"quad":{"attributes":{"POSITION":"p","NORMAL":"n"},"indices":"i"}},
        "materials":{"m":material},
        "programs":{"paint":{"source":"struct Inputs {float3 color=float3(1,0,0);}; Closure surface(ShadingContext ctx,Inputs i){Closure c={};c.base_weight=0.;c.specular_weight=0.;c.emission_color=i.color;c.emission_luminance=1.;return c;}"}},
        "animation":[{"tracks":[{"path":["materials","m","inputs",if program {"color"} else {"emission_color"}],
            "interpolation":"STEP","keys":[[0,[1,0,0]],[0.5,[0,0,1]]]}]}]
    }))?)
}
fn pixels(
    r: &Renderer,
    s: &InstalledScene,
    slot: &mut FrameSlot,
    time: f32,
) -> Result<Vec<[f32; 4]>> {
    slot.submit_outputs(
        s,
        r,
        time,
        ReadbackOutputs {
            float: FloatOutputs {
                color: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    slot.collect(r)?.linear_rgba.context("float colour")
}
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn animated_emission_program_shutter_and_seek() -> Result<()> {
    let r = crate::test_renderer()?;
    for program in [false, true] {
        let installed = r.install(scene(program, true)?, Path::new("."))?;
        assert!(installed.animation.has_material_bindings());
        let mut slot = installed.create_frame_slot(&r, ReadbackPrecision::Float)?;
        let mixed = pixels(&r, &installed, &mut slot, 0.5)?;
        let red = pixels(&r, &installed, &mut slot, 0.)?;
        let blue = pixels(&r, &installed, &mut slot, 1.)?;
        assert!(red[0][0] > 0.9 && red[0][2] < 0.01, "{red:?}");
        assert!(blue[0][2] > 0.9 && blue[0][0] < 0.01, "{blue:?}");
        assert!(
            (mixed[0][0] - 0.5).abs() < 0.08 && (mixed[0][2] - 0.5).abs() < 0.08,
            "{mixed:?}"
        );
        let sought = pixels(&r, &installed, &mut slot, 0.5)?;
        let mut alone = installed.create_frame_slot(&r, ReadbackPrecision::Float)?;
        crate::test_linear::assert_linear_eq(&sought, &pixels(&r, &installed, &mut alone, 0.5)?);
        crate::test_linear::assert_linear_eq(&mixed, &sought);
    }
    Ok(())
}
#[test]
fn constant_input_tracks_keep_static_install_path_and_loop_span() -> Result<()> {
    let mut source = scene(true, false)?;
    let track = &mut source.animation[0].tracks.as_mut().unwrap()[0];
    if let vork_scene::Values::Inline(values) = &mut track.values {
        values[1] = values[0].clone();
    }
    if let vork_scene::Keys::Inline(keys) = &mut track.keys {
        keys[1][1] = keys[0][1].clone();
    }
    source.animation[0].looping = true;
    let mut looped = source.clone();
    looped.nodes[0].name = "camera".into();
    looped.animation[0]
        .tracks
        .as_mut()
        .unwrap()
        .push(vork_scene::v2::from_test_value(json!({
            "path":["nodes","camera","translation"],"keys":[[0,[0,0,0]],[0.25,[1,0,0]]]
        }))?);
    animation::fold_material_inputs(&mut looped, Path::new("."), &Default::default())?;
    assert_eq!(looped.animation[0].source_span, Some((0., 0.5)));
    animation::fold_material_inputs(&mut source, Path::new("."), &Default::default())?;
    assert!(materials::animated_inputs(&source)?.is_empty());
    assert!(source.animation.is_empty());
    let plan = animation::Plan::compile(&source, Path::new("."))?;
    assert!(!plan.has_material_bindings());
    assert!(plan.material_inputs().is_empty());
    Ok(())
}
#[test]
fn reflected_program_inputs_use_ordered_layers_and_seek() -> Result<()> {
    let s = scene(true, false)?;
    let layout = vork_slang::InputLayout {
        fields: vec![vork_slang::InputField {
            name: "color".into(),
            ty: vork_slang::Type::Vector(3),
            offset: 1,
            enumeration: None,
            range: None,
        }],
        words: 4,
    };
    let mut words = vec![1, 1f32.to_bits(), 0, 0];
    let animation =
        programs::InputAnimation::new(&s, Path::new("."), "m", &layout, &mut words)?.unwrap();
    for time in [1., 0., 0.5, 1.] {
        animation.words(time, &mut words);
        let expected = if time < 0.5 {
            [1f32.to_bits(), 0, 0]
        } else {
            [0, 0, 1f32.to_bits()]
        };
        assert_eq!(&words[1..], expected);
    }
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn animated_medium_parameter_matches_static_and_seek() -> Result<()> {
    let r = crate::test_renderer()?;
    let mut source = scene(false, false)?;
    source.animation.clear();
    source.programs.insert("fog".into(),vork_scene::v2::from_test_value(json!({"source":"struct Inputs {float density=.1;}; MediumCoefficients medium(MediumContext ctx,Inputs i){MediumCoefficients m={};m.absorption=float3(i.density);return m;}"}))?);
    source.materials.insert(
        "fog".into(),
        vork_scene::v2::from_test_value(json!({"program":"fog","inputs":{"density":0.1}}))?,
    );
    source.nodes.push(vork_scene::v2::from_test_value(
        json!({"name":"world","type":"light","shape":"environment","energy":0,"medium":"fog"}),
    )?);
    source.animation = vork_scene::v2::from_test_value(
        json!([{"tracks":[{"path":["materials","fog","inputs","density"],"keys":[[0,0.1],[1,0.8]]}]}]),
    )?;
    let installed = r.install(source.clone(), Path::new("."))?;
    let mut slot = installed.create_frame_slot(&r, ReadbackPrecision::Float)?;
    let first = pixels(&r, &installed, &mut slot, 0.)?;
    let later = pixels(&r, &installed, &mut slot, 1.)?;
    assert!(first[0][0] > later[0][0] + 0.1, "{first:?} {later:?}");
    crate::test_linear::assert_linear_eq(&first, &pixels(&r, &installed, &mut slot, 0.)?);
    source.animation.clear();
    source
        .materials
        .get_mut("fog")
        .unwrap()
        .inputs
        .insert("density".into(), vork_scene::Input::Number(0.8));
    let static_scene = r.install(source, Path::new("."))?;
    let mut reference = static_scene.create_frame_slot(&r, ReadbackPrecision::Float)?;
    let expected = pixels(&r, &static_scene, &mut reference, 1.)?;
    for (a, b) in later.iter().zip(expected) {
        for c in 0..3 {
            assert!((a[c] - b[c]).abs() < 0.003, "{a:?} {b:?}");
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn animated_sky_parameter_updates_atmosphere_and_seeks() -> Result<()> {
    let r = crate::test_renderer()?;
    let source: Scene = vork_scene::v2::from_test_value(json!({"version":2,"nodes":[],
        "materials":{"sky":{"program":"std/sky","inputs":{"sun_elevation":0.2}}},
        "animation":[{"tracks":[{"path":["materials","sky","inputs","sun_elevation"],"keys":[[0,0.2],[1,0.8]]}]}]}))?;
    let (first, program) = environment_program::install(&r, &source, "sky", Path::new("."))?;
    let program = program.context("input track retains environment evaluation")?;
    let later = program.evaluate(&r, 1.)?;
    assert_ne!(first.image, later.image);
    assert_eq!(later.image, later.camera);
    assert_eq!(first.image, program.evaluate(&r, 0.)?.image);
    Ok(())
}

#[test]
fn buffer_backed_inputs_have_content_cache_identity() -> Result<()> {
    let mut a = scene(true, false)?;
    a.buffers.insert(
        "times".into(),
        vork_scene::v2::from_test_value(json!({"type":"f32","data":[0,1]}))?,
    );
    a.buffers.insert(
        "colors".into(),
        vork_scene::v2::from_test_value(json!({"type":"vec3","data":[[1,0,0],[0,0,1]]}))?,
    );
    a.animation[0].tracks = Some(vec![vork_scene::v2::from_test_value(
        json!({"path":["materials","m","inputs","color"],"keys":{"times":"times","values":"colors"}}),
    )?]);
    let mut b = a.clone();
    b.buffers.insert(
        "colors".into(),
        vork_scene::v2::from_test_value(json!({"type":"vec3","data":[[1,0,0],[0,1,0]]}))?,
    );
    let make = |s: &Scene| -> Result<programs::InputAnimation> {
        let layout = vork_slang::InputLayout {
            fields: vec![vork_slang::InputField {
                name: "color".into(),
                ty: vork_slang::Type::Vector(3),
                offset: 1,
                enumeration: None,
                range: None,
            }],
            words: 4,
        };
        let mut words = vec![1, 1f32.to_bits(), 0, 0];
        Ok(programs::InputAnimation::new(s, Path::new("."), "m", &layout, &mut words)?.unwrap())
    };
    assert_ne!(make(&a)?.identity(), make(&b)?.identity());
    Ok(())
}

#[test]
fn weighted_constant_program_inputs_fold_to_the_evaluated_baseline() -> Result<()> {
    let mut source = scene(true, false)?;
    source
        .materials
        .get_mut("m")
        .unwrap()
        .inputs
        .remove("color");
    source.animation[0].weight = 0.5;
    source.animation[0].tracks = Some(vec![vork_scene::v2::from_test_value(json!({
        "path":["materials","m","inputs","color"],"keys":[[0,[1,0,0]],[1,[1,0,0]]]
    }))?]);
    let layout = vork_slang::InputLayout {
        fields: vec![vork_slang::InputField {
            name: "color".into(),
            ty: vork_slang::Type::Vector(3),
            offset: 1,
            enumeration: None,
            range: None,
        }],
        words: 4,
    };
    let mut words = vec![0; 4];
    let animation =
        programs::InputAnimation::new(&source, Path::new("."), "m", &layout, &mut words)?.unwrap();
    animation.words(0.5, &mut words);
    let evaluated: Vec<_> = words[1..].iter().map(|w| f32::from_bits(*w)).collect();
    animation::fold_material_inputs(&mut source, Path::new("."), &Default::default())?;
    assert_eq!(
        serde_json::to_value(&source.materials["m"].inputs["color"])?,
        json!(evaluated)
    );
    assert_eq!(evaluated, vec![0.5, 0., 0.]);
    assert!(source.animation.is_empty());
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn animated_rgb_background_uses_the_same_shutter_rule() -> Result<()> {
    let r = crate::test_renderer()?;
    let mut source = scene(false, true)?;
    source.nodes.truncate(1);
    source.render.background = vork_scene::Background::Color([1., 0., 0.]);
    source.animation = vork_scene::v2::from_test_value(json!([{"tracks":[{
        "path":["render","background"],"interpolation":"STEP","keys":[[0,[1,0,0]],[0.5,[0,0,1]]]
    }]}]))?;
    let installed = r.install(source, Path::new("."))?;
    let mut slot = installed.create_frame_slot(&r, ReadbackPrecision::Float)?;
    let mixed = pixels(&r, &installed, &mut slot, 0.5)?;
    assert!(
        (mixed[0][0] - 0.5).abs() < 0.08 && (mixed[0][2] - 0.5).abs() < 0.08,
        "{mixed:?}"
    );
    Ok(())
}

#[test]
fn normal_map_factors_are_ordinary_numeric_program_inputs() -> Result<()> {
    let mut source: Scene = vork_scene::v2::from_test_value(json!({"version":2,"nodes":[{"type":"camera"}],
        "images":{"normal":{"file":"normal.png","color_space":"Non-Color"}},
        "programs":{"mapped":{"source":"import vork; struct Inputs {Image normal;float3 strength;}; Closure surface(ShadingContext ctx,Inputs p) {Closure c={};return c;}"}},
        "materials":{"m":{"program":"mapped","inputs":{"normal":{"image":"normal"},"strength":[0.5,0.5,1.]}}},
        "animation":[{"weight":0.5,"tracks":[{"path":["materials","m","inputs","strength"],"keys":[[0,[1,2,3]],[1,[1,2,3]]]}]}]}))?;
    animation::fold_material_inputs(&mut source, Path::new("."), &Default::default())?;
    assert_eq!(serde_json::to_value(&source.materials["m"].inputs["strength"])?,json!([0.75,1.25,2.]));
    assert_eq!(serde_json::to_value(&source.materials["m"].inputs["normal"])?,json!({"image":"normal"}));
    source.validate()?;
    assert!(source.animation.is_empty());
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn animated_fog_preserves_proofs_of_static_inputs() -> Result<()> {
    let r = crate::test_renderer()?;
    let mut source: Scene = vork_scene::v2::from_test_value(json!({"version":2,"nodes":[],
        "materials":{"fog":{"program":{"source":include_str!("../../vork-scene/tests/fixtures/medium.slang")},"inputs":{"density":0.1}}},
        "animation":[{"tracks":[{"path":["materials","fog","inputs","density"],"keys":[[0,0.1],[1,0.8]]}]}]}))?;
    let medium = medium::install(
        &r,
        &source,
        Some("fog"),
        Path::new("."),
        &Default::default(),
    )?
    .unwrap();
    assert!(
        medium.native.homogeneous,
        "static zero height falloff remains position-independent with unknown density"
    );
    assert!(medium.native.sampled_error.is_none());
    source.animation[0].tracks.as_mut().unwrap()[0].path[3] = "height_falloff".into();
    let varying = medium::install(
        &r,
        &source,
        Some("fog"),
        Path::new("."),
        &Default::default(),
    )?
    .unwrap();
    assert!(
        !varying.native.homogeneous,
        "animated height falloff must remain unknown to the proof"
    );
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn medium_input_tracks_follow_instance_time_offsets() -> Result<()> {
    let r = crate::test_renderer()?;
    let mut source: Scene = vork_scene::v2::from_test_value(vork_scene::v2::with_test_settings(
        include_str!("../tests/fixtures/analytic.json"),
        include_str!("../tests/presets/analytic-7fefd52a.json"),
    )?)?;
    source.nodes.truncate(2);
    source.render.resolution = [4, 4];
    source.programs.insert("test-medium".into(), serde_json::from_value(json!({"source":include_str!("../../vork-scene/tests/fixtures/medium.slang")}))?);
    source.materials.insert(
        "fog".into(),
        vork_scene::v2::from_test_value(json!({"program":"test-medium","inputs":{"density":0.1}}))?,
    );
    source.buffers.insert(
        "translations".into(),
        vork_scene::v2::from_test_value(json!({"type":"vec3","data":[[0,0,0],[3,0,0]]}))?,
    );
    source.buffers.insert(
        "phases".into(),
        vork_scene::v2::from_test_value(json!({"type":"f32","data":[0,0.5]}))?,
    );
    if let vork_scene::NodeKind::Mesh {
        materials,
        instances,
        ..
    } = &mut source.nodes[1].kind
    {
        *materials = vec!["fog".into()];
        *instances = Some(vork_scene::v2::from_test_value(
            json!({"attributes":{"TRANSLATION":"translations","time_offset":"phases"}}),
        )?);
    }
    source.animation = vork_scene::v2::from_test_value(
        json!([{"tracks":[{"path":["materials","fog","inputs","density"],"keys":[[0,0.1],[1,0.8]]}]}]),
    )?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let installed = r.install(source, &fixture)?;
    assert!(installed.animation.material_offsets().contains(&0.5));
    let mut samples = installed.animation.samples();
    installed
        .animation
        .sample_material_offset(0., 0.5, &mut samples);
    let values: Vec<_> = installed.animation.material_values(&samples, 0.5).collect();
    assert_eq!(values.len(), 1);
    let animation::Sample::Numeric(value) = values[0].1 else {
        panic!("numeric density")
    };
    assert!((value[0] - 0.45).abs() < 1e-6, "{value:?}");
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn animated_bounded_path_extinction_requires_an_input_independent_certificate() -> Result<()> {
    let r = crate::test_renderer()?;
    let mut value = json!({"version":2,"render":{"resolution":[4,4],"duration":2},
        "nodes":[{"type":"camera"}],"materials":{},"programs":{}});
    value["render"]["settings"] = json!({"integrator":"path","samples":1,"denoiser":null});
    value["programs"]["fog"] = json!({"source":"struct Inputs {float density=.1;}; MediumCoefficients medium(MediumContext c,Inputs i){MediumCoefficients m;m.absorption=float3(i.density*(1.+c.position.x*c.position.x));return m;}"});
    value["materials"]["fog"] = json!({"program":"fog","inputs":{"density":0.1}});
    value["nodes"].as_array_mut().unwrap().push(json!({"type":"mesh","geometry":{"std":"rounded_box","radius":0,"size":[2,2,2]},"materials":["fog"]}));
    value["animation"] = json!([{"tracks":[{"path":["materials","fog","inputs","density"],"keys":[[0,0.1],[1,10.]]}]}]);
    let parse = |mut value: serde_json::Value| -> Result<Scene> {
        let settings = value["render"].as_object_mut().unwrap().remove("settings").unwrap();
        let mut scene = crate::test_scene(value.clone(), std::path::Path::new("."))?;
        scene.render.settings = serde_json::from_value(settings)?;
        Ok(scene)
    };
    let error = r.install(parse(value.clone())?, Path::new(".")).err().context("animated extinction must not use an initial-value certificate")?;
    assert!(format!("{error:#}").contains("cannot certify extinction majorant"), "{error:#}");
    // An animated source term does not invalidate a static extinction proof.
    value["programs"]["fog"]["source"] = json!("struct Inputs {float density=.1;}; MediumCoefficients medium(MediumContext c,Inputs i){MediumCoefficients m;m.absorption=float3(.1*(1.+c.position.x*c.position.x));m.emission=float3(i.density);return m;}");
    r.install(parse(value)?, Path::new("."))?;
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn animated_compositor_inputs_match_static_and_seek() -> Result<()> {
    let r = crate::test_renderer()?;
    let mut source = scene(false, false)?;
    source.animation.clear();
    source.render.resolution = [16, 16];
    *source.compositor_mut() = vec!["glow".into(), "glow".into()];
    source.programs.insert("gain".into(), vork_scene::v2::from_test_value(json!({"source":"import vork; struct Inputs { float strength=0.; }; float4 composite(CompositeContext c,Inputs i) { float4 v=compositor_sample(c.uv);return float4(v.rgb*(1.+i.strength),v.a); }"}))?);
    source.materials.insert("glow".into(), vork_scene::v2::from_test_value(json!({
        "program":"gain", "inputs":{"strength":0}
    }))?);
    source.animation = vork_scene::v2::from_test_value(json!([{"tracks":[{
        "path":["materials","glow","inputs","strength"],"keys":[[0,0],[1,2]]
    }]}]))?;
    let installed = r.install(source.clone(), Path::new("."))?;
    let mut slot = installed.create_frame_slot(&r, ReadbackPrecision::Float)?;
    let first = pixels(&r, &installed, &mut slot, 0.)?;
    let later = pixels(&r, &installed, &mut slot, 1.)?;
    assert!(later[0][0] > first[0][0] + 0.1);
    crate::test_linear::assert_linear_eq(&first, &pixels(&r, &installed, &mut slot, 0.)?);
    source.animation.clear();
    source.materials.get_mut("glow").unwrap().inputs.insert("strength".into(), vork_scene::Input::Number(2.));
    let static_scene = r.install(source, Path::new("."))?;
    let mut static_slot = static_scene.create_frame_slot(&r, ReadbackPrecision::Float)?;
    crate::test_linear::assert_linear_eq(&later, &pixels(&r, &static_scene, &mut static_slot, 1.)?);
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn camera_cut_left_limit_keeps_step_emission_and_visibility() -> Result<()> {
    let r = crate::test_renderer()?;
    for (program,visibility) in [(false,false),(true,false),(false,true)] {
        let mut input=scene(program,true)?;
        input.render.fps=vork_time::Time::integer(24);
        input.render.motion_blur.as_mut().unwrap().shutter=4.;
        input.render.camera=Some("a".into());
        input.nodes[0].name="a".into();
        input.nodes[1].name="subject".into();
        let mut camera=input.nodes[0].clone();camera.name="b".into();input.nodes.push(camera);
        if visibility {
            input.animation=vork_scene::v2::from_test_value(json!([{"tracks":[{
                "path":["nodes","subject","visible"],"keys":[[0,true],[0.5,false]]
            }]}]))?;
        }
        input.animation.push(vork_scene::v2::from_test_value(json!({"tracks":[{
            "path":["render","camera"],"keys":[[0,"a"],[0.5,"b"]]
        }]}))?);
        let installed=r.install(input,Path::new("."))?;
        let mut slot=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
        slot.submit_outputs(&installed,&r,vork_time::Request::Frame(11,vork_time::Time::integer(24)),ReadbackOutputs {
            float:FloatOutputs {color:true,..Default::default()},..Default::default()
        })?;
        let pixels=slot.collect(&r)?.linear_rgba.context("float colour")?;
        assert!(pixels.iter().all(|p|p[0]>0.9 && p[1]<0.01 && p[2]<0.01),
            "next-shot discrete state leaked across the left limit: program={program}, visibility={visibility}, {pixels:?}");
    }
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn discrete_pose_reuse_distinguishes_left_limit_from_equal_numeric_point()->Result<()> {
    let r = crate::test_renderer()?;
    let mut input=scene(true,false)?;input.render.fps=vork_time::Time::integer(24);
    let installed=r.install(input,Path::new("."))?;
    let mut slot=installed.create_frame_slot(&r,Default::default())?;
    slot.view.sample_index=1;
    let mut lighting=values::LightingTracker::default();
    let cut=vork_time::exact::decimal("1/2")?;
    let before=vork_time::Request::Exact(vork_time::exact::Germ::left(cut.clone()));
    let after=vork_time::Request::Exact(vork_time::exact::Germ::point(cut));
    assert_eq!(before.numeric()?,after.numeric()?);
    for request in [&before,&after,&before] {
        slot.prepare_values(&installed,&r,request,0.5,&mut lighting)?;
        assert!(!slot.reuse_values,"discrete ownership change reused the numeric pose");
        slot.prepare_values(&installed,&r,request,0.5,&mut lighting)?;
        assert!(slot.reuse_values,"identical discrete and numeric inputs should reuse");
    }
    Ok(())
}
