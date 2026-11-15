use super::*;
#[path = "test_capture.rs"]
mod capture_test;

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn hidden_world_medium_has_no_absorption_or_emission() -> Result<()> {
    let renderer = crate::test_renderer()?;
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    // Point into the dense lower exterior: constant-emission height fog diverges
    // upward. Keep the bounded volume and capture screen centered on this view.
    for (falloff,capture,bounded) in [(0.,false,false),(0.2,false,false),(0.,true,false),(0.,false,true),(0.2,false,true)] {
        eprintln!("world visibility: falloff={falloff}, capture={capture}, bounded={bounded}");
        let mut input = serde_json::json!({"version":2,
            "render":{"resolution":[8,8],"background":[0.4,0.4,0.4],"settings":{"samples":1}},
            "materials":{"fog":{"program":{"source":include_str!("../../vork-scene/tests/fixtures/medium.slang")},"inputs":{"density":0.5,"scattering":[0,0,0],"absorption":[1,1,1],"emission":[0.3,0,0],"height_falloff":falloff}},"camera_settings":{"program":"std/camera","inputs":{"clip_end":10}}},
            "nodes":[{"type":"camera","name":"cam","target":[0,-0.6,-0.8],"material":"camera_settings"},
                {"type":"node","name":"parent","children":[{"type":"light","shape":"environment","medium":"fog","color":[0,0,0]}]}],
            "animation":[{"tracks":[{"path":["nodes","parent","visible"],"keys":[[0,false],[1,true]]}]}]
        });
        if bounded {
            let fixture: serde_json::Value = serde_json::from_str(include_str!("../tests/fixtures/analytic.json"))?;
            input["buffers"] = fixture["buffers"].clone();
            input["geometry"] = fixture["geometry"].clone();
            // A varying medium requires a finite boundary. Keep that boundary
            // under the animated parent so this still tests inherited visibility.
            input["nodes"][1]["children"][0].as_object_mut().unwrap().remove("medium");
            input["nodes"][1]["children"].as_array_mut().unwrap().push(serde_json::json!({"type":"mesh","geometry":"shape_box","materials":["fog"],"translation":[0,-1.8,-2.4]}));
        }
        if capture {
            // Consume a whole-scene capture without a monitor surface inside
            // the medium or unsupported path subtree membership.
            input["images"] = serde_json::json!({"feed":{"render":{"camera":"cam","resolution":[8,8],"background":[0.4,0.4,0.4]}}});
            input["composition"]["space"] = "3d".into(); input["composition"]["compositor"] = capture_test::compositor(&mut input, "feed");
        }
        if falloff > 0. && !bounded {
            // Certified exponential-height world fog installs and renders below.
            // A varying world profile without that certificate (Gaussian in
            // height here) still requires a finite domain.
            let certified = include_str!("../../vork-scene/tests/fixtures/medium.slang");
            let uncertified = certified.replace("exp(-inputs.height_falloff*ctx.position.y)", "exp(-inputs.height_falloff*ctx.position.y*ctx.position.y)");
            assert_ne!(uncertified, certified, "fixture height profile changed");
            let mut refused = input.clone();
            refused["materials"]["fog"]["program"]["source"] = serde_json::Value::String(uncertified);
            let error = renderer.install(vork_scene::v2::from_test_value(refused)?, &base)
                .err().expect("uncertified varying world fog must require a finite domain");
            assert!(format!("{error:#}").contains("sampled heterogeneous world medium fog needs a finite domain"), "{error:#}");
        }
        let mut installed = renderer.install(vork_scene::v2::from_test_value(input.clone())?,&base)?;
        let mut frame = installed.create_frame_slot(&renderer,Default::default())?;
        let mut images = Vec::new();
        for time in [0.,1.,0.] {
            let output = installed.render_frame(&renderer, &mut frame, time)?;
            if let Some(out) = std::env::var_os("VORK_TEST_OUTPUT") {
                std::fs::create_dir_all(&out)?;
                image::save_buffer(Path::new(&out).join(format!("medium-{falloff}-{capture}-{bounded}-{time}.png")), &output.rgba, 8, 8, image::ColorType::Rgba8)?;
            }
            images.push(output.rgba);
        }
        let mut clear = input;
        clear["nodes"][1]["children"][0].as_object_mut().unwrap().remove("medium");
        if bounded {
            clear["nodes"][1]["children"].as_array_mut().unwrap().pop();
        }
        let mut installed = renderer.install(vork_scene::v2::from_test_value(clear)?,&base)?;
        let mut frame = installed.create_frame_slot(&renderer,Default::default())?;
        let clear = installed.render_frame(&renderer, &mut frame, 0.)?.rgba;
        assert_eq!(images[0],clear,"hidden world medium at falloff {falloff}");
        assert_eq!(images[2],clear,"backward sample at falloff {falloff}");
        assert_ne!(images[1],clear,"visible world medium must affect pixels");
    }
    Ok(())
}

/// BB105: a hidden middle TLAS slot must not corrupt the later live slot on CPU.
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn hidden_geometry_matches_deleted_geometry() -> Result<()> {
    let r = crate::test_renderer()?;
    let input: serde_json::Value = serde_json::from_str(include_str!("../tests/fixtures/hidden-geometry-bb105.json"))?;
    let render = |input: serde_json::Value, times: &[f32]| -> Result<Vec<Vec<[f32; 4]>>> {
        let mut scene = crate::test_scene(input, Path::new("."))?;
        scene.render.settings = vork_scene::Profile::load("std/path")?.resolve(
            serde_json::json!({"samples": 1, "denoise": false, "seed": 0}).as_object().unwrap())?;
        let installed = r.install(scene, Path::new("."))?;
        let mut slot = installed.create_frame_slot(&r, ReadbackPrecision::Float)?;
        let mut frames = Vec::new();
        for &time in times {
            slot.submit_outputs(&installed, &r, time, ReadbackOutputs { float: FloatOutputs { color: true, ..Default::default() }, ..Default::default() })?;
            let pixels = slot.collect(&r)?.linear_rgba.context("BB105 linear colour")?;
            assert!(pixels.iter().flatten().all(|c| c.is_finite()));
            frames.push(pixels);
        }
        Ok(frames)
    };
    let hidden = render(input.clone(), &[0.])?;
    let mut visible = input.clone();
    visible["nodes"][1]["visible"] = true.into();
    let visible = render(visible, &[1.])?;
    assert_eq!(visible[0].len(), hidden[0].len());
    let mut deleted = input.clone();
    deleted["nodes"].as_array_mut().unwrap().remove(1);
    let deleted = render(deleted, &[0.])?;
    assert!(hidden == deleted, "hidden and deleted geometry must produce identical linear pixels");
    // Reuse a frame slot through visibility changes and a backward seek. The
    // software placeholder must restore the real BLAS when visibility returns.
    let mut animated = input;
    // The repro's fox is behind the screen. Bring it in front for an observable
    // reactivation check, while retaining the unmodified three controls above.
    animated["nodes"][1]["translation"][2] = 1.into();
    animated["nodes"][1]["visible"] = true.into();
    let visible_front = render(animated.clone(), &[1.])?;
    assert!(visible_front[0] != hidden[0], "reactivated geometry must affect pixels");
    animated["animation"] = serde_json::json!([{"tracks":[{"path":["nodes","fox","visible"],"keys":[[0,false],[1,true]]}]}]);
    let animated = render(animated, &[0., 1., 0.])?;
    assert!(animated[0] == hidden[0], "initial hidden animation frame");
    assert!(animated[1] == visible_front[0], "reactivated geometry");
    assert!(animated[2] == hidden[0], "backward seek to hidden geometry");
    r.errors.check()?;
    Ok(())
}
