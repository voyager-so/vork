//! CPU read-back must also work when no rendering device is available.
use serde_json::{Value, json};
use std::{
    io::Write,
    path::PathBuf,
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new(scene: Value) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "vork-ask-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("scene.json"), serde_json::to_vec(&scene).unwrap()).unwrap();
        Self(root)
    }
    fn run(&self, questions: &Value, stdin: bool) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vork"));
        command
            .env("VORK_GPU", "no-such-device")
            .env("VK_ICD_FILENAMES", self.0.join("no-vulkan.json"))
            .env("VORK_NO_UPDATE_NOTIFIER", "1")
            .arg("ask")
            .arg(self.0.join("scene.json"));
        if stdin {
            command.arg("-");
        } else {
            command.arg(questions.to_string());
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        if stdin {
            child
                .stdin
                .take()
                .unwrap()
                .write_all(questions.to_string().as_bytes())
                .unwrap();
        }
        child.wait_with_output().unwrap()
    }
    fn ask(&self, questions: Value) -> Value {
        let result = self.run(&questions, true);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            result.stderr.is_empty(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        serde_json::from_slice(&result.stdout).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn triangle() -> Value {
    json!({"version":2,"render":{"resolution":[320,240]},
      "buffers":{"p":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[0,1,0]]}},
      "geometry":{"tri":{"attributes":{"POSITION":"p"}}},
      "nodes":[{"name":"parent","translation":[2,0,0],"children":[{"type":"mesh","name":"hero","geometry":"tri"}]},
        {"type":"camera","name":"camera","translation":[0,0,5]}]})
}

fn near(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                near(a, b);
            }
        }
        _ => assert!(
            (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5,
            "{actual} != {expected}"
        ),
    }
}

fn pixel_enclosure(actual: &Value, geometric: [f64; 4]) {
    for (i, expected) in geometric.into_iter().enumerate() {
        let edge = actual[i].as_i64().expect("integer pixel edge") as f64;
        let outward = if i < 2 {
            expected - edge
        } else {
            edge - expected
        };
        // The shared camera evaluates in f32; a value on an integer boundary
        // can conservatively cover the adjacent pixel after outward rounding.
        assert!(
            (-1e-5..=1.00001).contains(&outward),
            "{actual} does not enclose {geometric:?}"
        );
    }
}

#[test]
fn gaussian_assets_share_cpu_transforms_and_support_bounds() {
    let fixture=Fixture::new(json!({"version":2,"nodes":[
        {"type":"asset","name":"capture","file":"point.ply","translation":[2,-1,4],"scale":[2,3,4]}]}));
    let mut ply=String::from("ply\nformat ascii 1.0\nelement vertex 1\n");
    for name in ["x","y","z","scale_0","scale_1","scale_2","opacity","rot_0","rot_1","rot_2","rot_3","f_dc_0","f_dc_1","f_dc_2"] {
        ply.push_str(&format!("property float {name}\n"));
    }
    ply.push_str("end_header\n1 2 3 0 -1 -2 0 1 0 0 0 0.1 0.2 0.3\n");
    std::fs::write(fixture.0.join("point.ply"),ply).unwrap();
    let result=fixture.ask(json!([{"node":"capture","read":"transform"},{"node":"capture","read":"bounds"}]));
    near(&result[0],&json!([2,0,0,0,0,3,0,0,0,0,4,0,2,-1,4,1]));
    let y=9.0*(-1.0f64).exp();let z=12.0*(-2.0f64).exp();
    near(&result[1],&json!([[-2,5.-y,16.-z],[10,5.+y,16.+z]]));
}

#[test]
fn ordered_batches_stdin_world_subtree_and_empty() {
    let fixture = Fixture::new(triangle());
    let questions = json!([{"node":"hero","read":"transform"},{"node":"parent","read":"bounds"},{"node":"camera","read":"bounds"},{"field":"nodes.hero.translation"}]);
    let result = fixture.ask(questions.clone());
    near(
        &result[0],
        &json!([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1]),
    );
    near(&result[1], &json!([[1, -1, 0], [3, 1, 0]]));
    assert_eq!(result[2], Value::Null);
    near(&result[3], &json!([0, 0, 0]));
    let direct = fixture.run(&questions, false);
    assert!(direct.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&direct.stdout).unwrap(),
        result
    );
    assert_eq!(fixture.ask(json!([])), json!([]));
}

#[test]
fn fields_and_worlds_share_step_linear_and_cubic_values() {
    for (interpolation, keys, expected) in [
        (
            "STEP",
            json!([[0, [0, 0, 0]], [2, [4, 0, 0]]]),
            json!([0, 0, 4]),
        ),
        (
            "LINEAR",
            json!([[0, [0, 0, 0]], [2, [4, 0, 0]]]),
            json!([0, 1, 4]),
        ),
        // Hermite at u=.25: h10 * duration * outgoing derivative = .5625.
        (
            "CUBICSPLINE",
            json!([
                [0, [0, 0, 0], [0, 0, 0], [2, 0, 0]],
                [2, [0, 0, 0], [0, 0, 0], [0, 0, 0]]
            ]),
            json!([0, 0.5625, 0]),
        ),
    ] {
        let mut scene = triangle();
        scene["animation"] = json!([{"tracks":[{"path":["nodes","hero","translation"],"keys":keys,"interpolation":interpolation}]}]);
        let fixture = Fixture::new(scene);
        let result = fixture.ask(json!([{"field":"nodes.hero.translation","at":[0,"1/2",2]},
          {"node":"hero","read":"transform","at":[0,"1/2",2]},
          {"node":"hero","read":"bounds","at":[0,"1/2",2]}]));
        for i in 0..3 {
            near(&result[0][i][0], &expected[i]);
            near(
                &result[1][i][12],
                &json!(2. + expected[i].as_f64().unwrap()),
            );
            near(
                &result[2][i][0][0],
                &json!(1. + expected[i].as_f64().unwrap()),
            );
        }
    }
}

#[test]
fn unnamed_node_cannot_be_selected_by_an_empty_field_segment() {
    let fixture = Fixture::new(json!({"version":2,"nodes":[{"translation":[1,2,3]}]}));
    let output = fixture.run(&json!([{"field":"nodes..translation"}]), true);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let error = String::from_utf8_lossy(&output.stderr);
    assert_eq!(error.lines().count(), 1, "{error}");
    assert!(error.contains("named node"), "{error}");
}

#[test]
fn material_defaults_keep_declared_types_and_empty_times_refuse() {
    let mut scene = triangle();
    scene["materials"] = json!({"paint": {},"optics":{"program":"std/camera"}});
    let fixture = Fixture::new(scene);
    let defaults = fixture.ask(json!([
        {"field":"materials.paint.inputs.geometry_thin_walled"},
        {"field":"materials.optics.inputs.lens"},
        {"field":"materials.optics.inputs.projection"},
        {"field":"materials.optics.inputs.use_dof"},
        {"field":"materials.optics.inputs.aperture_blades"}
    ]));
    assert_eq!(defaults[0], json!(false));
    near(&defaults[1], &json!(50));
    assert_eq!(defaults[2], json!("persp"));
    assert_eq!(defaults[3], json!(false));
    assert_eq!(defaults[4].as_u64(), Some(0));
    for node in ["hero", "missing"] {
        let output = fixture.run(&json!([{"node":node,"read":"bounds","at":[]}]), true);
        assert!(!output.status.success() && output.stdout.is_empty());
        let message = String::from_utf8(output.stderr).unwrap();
        assert_eq!(message.lines().count(), 1);
        assert!(
            message.contains("at must contain a scene time"),
            "{message}"
        );
    }
}

#[test]
fn imported_camera_render_notices_do_not_escape_question_diagnostics() {
    for projection in [
        json!({"type":"perspective","perspective":{"yfov":0.6,"znear":0.01}}),
        json!({"type":"orthographic","orthographic":{"xmag":2,"ymag":2,"znear":0,"zfar":100}}),
    ] {
        let fixture = Fixture::new(
            json!({"version":2,"nodes":[{"type":"asset","name":"rig","file":"camera.gltf"}]}),
        );
        std::fs::write(
            fixture.0.join("camera.gltf"),
            serde_json::to_vec(&json!({
                "asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],
                "nodes":[{"camera":0,"name":"view","translation":[0,0,8]}],"cameras":[projection]
            }))
            .unwrap(),
        )
        .unwrap();
        let output = fixture.run(&json!([{"buffer":"missing"}]), true);
        assert!(!output.status.success() && output.stdout.is_empty());
        let message = String::from_utf8(output.stderr).unwrap();
        assert_eq!(message.lines().count(), 1, "{message}");
        assert!(message.contains("unknown buffer"), "{message}");
        let values = fixture.ask(json!([{"node":"rig/view","read":"transform"}]));
        near(&values[0][14], &json!(8));
    }
}

#[test]
fn typed_buffer_ranges_counts_and_file_offsets() {
    let fixture = Fixture::new(json!({"version":2,"nodes":[],"buffers":{
      "unsigned":{"type":"u32","data":[0,4294967295_u64]},"signed":{"type":"i32","data":[-2147483648,7]},
      "vectors":{"type":"vec3","data":[[1,2,3],[4,5,6]]},
      "file":{"type":"f32","file":"values.bin","offset":4,"count":2}}}));
    std::fs::write(
        fixture.0.join("values.bin"),
        [99_f32, 2.5, -3.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let result = fixture.ask(
        json!([{"buffer":"unsigned"},{"buffer":"signed"},{"buffer":"vectors","range":[1,1]},
      {"buffer":"vectors","range":[2,0]},{"buffer":"vectors","read":"count"},{"buffer":"file"}]),
    );
    assert_eq!(
        result,
        json!([[0,4294967295_u64],[-2147483648_i64,7],[[4.,5.,6.]],[],{"count":2,"capacity":2},[2.5,-3.]])
    );
}

#[test]
fn instance_selection_live_count_and_time_offset() {
    let mut scene = triangle();
    scene["buffers"]["placements"] = json!({"type":"vec3","data":[[0,0,0],[5,0,0]]});
    scene["buffers"]["offsets"] = json!({"type":"f32","data":[0,1]});

    scene["nodes"][0]["children"][0]["instances"] =
        json!({"attributes":{"TRANSLATION":"placements","time_offset":"offsets"}});
    scene["animation"] = json!([{"tracks":[{"path":["nodes","hero","translation"],"keys":[[0,[0,0,0]],[2,[2,0,0]]]}]}]);
    let fixture = Fixture::new(scene);
    let result = fixture.ask(
        json!([{"node":"hero","read":"transform"},{"node":"hero","read":"transform","instance":1},
      {"node":"hero","read":"bounds"},{"node":"hero","read":"bounds","instance":1}]),
    );
    near(&result[0][12], &json!(2));
    near(&result[1][12], &json!(8));
    near(&result[2], &json!([[1, -1, 0], [9, 1, 0]]));
    near(&result[3], &json!([[7, -1, 0], [9, 1, 0]]));
}

#[test]
fn image_headers_are_read_without_pixels_or_device() {
    let fixture =
        Fixture::new(json!({"version":2,"nodes":[],"images":{"plate":{"file":"plate.png"}}}));
    let bytes = [
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 3, 0, 0, 0, 2, 8, 2,
        0, 0, 0, 18, 22, 241, 77, 0, 0, 0, 14, 73, 68, 65, 84, 120, 156, 99, 168, 135, 1, 6, 56,
        11, 0, 89, 96, 8, 239, 238, 131, 155, 210, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];
    std::fs::write(fixture.0.join("plate.png"), bytes).unwrap();
    assert_eq!(
        fixture.ask(json!([{"image":"plate","read":"info"}])),
        json!([{"width":3,"height":2}])
    );
}

#[test]
fn camera_fit_shift_orthographic_and_cut_use_output_pixels() {
    let mut scene = triangle();
    scene["nodes"][0]["translation"] = json!([0, 0, 0]);
    scene["nodes"][1]["material"] = json!("optics");
    scene["materials"] = json!({"optics":{"program":"std/camera","inputs":{"projection":"ortho","fit":"vertical","ortho_scale":4,"shift":[0.1,0.2]}}});
    scene["nodes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"type":"camera","name":"behind","translation":[0,0,-5]}));
    scene["render"]["camera"] = json!("camera");
    scene["animation"] = json!([{"tracks":[{"path":["render","camera"],"interpolation":"STEP","keys":[[0,"camera"],[1,"behind"]]}]}]);
    let fixture = Fixture::new(scene);
    let result=fixture.ask(json!([{"node":"hero","read":"screen","at":[0,1]},
      {"node":"hero","read":"screen","at":1,"camera":"camera"},{"field":"render.camera","at":[0,1]}]));
    // Vertical extent = four metres; one metre = sixty pixels. Shift offsets
    // the optical centre by (-24,+48) pixels at this output resolution.
    pixel_enclosure(&result[0][0]["box"], [76., 108., 196., 228.]);
    near(&result[0][0]["depth"], &json!([5, 5]));
    assert_eq!(result[0][1], Value::Null);
    assert_eq!(result[1], result[0][0]);
    assert_eq!(result[2], json!(["camera", "behind"]));
}

#[test]
fn native_skin_and_morph_are_evaluated_before_world_bounds() {
    let mut scene = triangle();
    scene["buffers"]["j"] = json!({"type":"uvec4","data":[[0,0,0,0],[0,0,0,0],[1,0,0,0]]});
    scene["buffers"]["w"] = json!({"type":"vec4","data":[[2,0,0,0],[1,0,0,0],[3,0,0,0]]});
    scene["buffers"]["bind"] = json!({"type":"mat4","data":[[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]]});
    scene["buffers"]["delta"] = json!({"type":"vec3","data":[[0,0,0],[0,0,0],[0,1,0]]});
    scene["geometry"]["tri"]["attributes"]["JOINTS_0"] = json!("j");
    scene["geometry"]["tri"]["attributes"]["WEIGHTS_0"] = json!("w");
    scene["geometry"]["tri"]["morphs"] = json!({"grow":{"POSITION":"delta"}});
    scene["nodes"][0]["children"][0]["skin"] =
        json!({"joints":["root","bend"],"inverse_bind_matrices":"bind"});
    scene["nodes"][0]["children"][0]["morph_weights"] = json!({"grow":0.5});
    scene["nodes"]
        .as_array_mut()
        .unwrap()
        .extend([json!({"name":"root"}), json!({"name":"bend"})]);
    scene["animation"] = json!([{"tracks":[{"path":["nodes","bend","translation"],"keys":[[0,[0,0,0]],[2,[0,0,2]]]}]}]);
    let fixture = Fixture::new(scene);
    let result = fixture.ask(json!([{"node":"hero","read":"bounds","at":[0,1]}]));
    // Skin palette is inverse(mesh world) * joint world * inverse bind.
    near(&result[0][0], &json!([[-1, -1, 0], [1, 1.5, 0]]));
    near(&result[0][1], &json!([[-1, -1, 0], [1, 1.5, 1]]));
}

#[test]
fn aperture_envelope_keeps_focus_plane_and_expands_defocus() {
    let mut scene = triangle();
    scene["nodes"][0]["translation"] = json!([0, 0, 0]);
    scene["nodes"][1]["material"] = json!("optics");
    scene["materials"] = json!({"optics":{"program":"std/camera","inputs":{"use_dof":true,"focus_distance":5,"aperture_fstop":1,"aperture_ratio":0.5}}});
    scene["animation"] = json!([{"tracks":[{"path":["materials","optics","inputs","focus_distance"],"keys":[[0,5],[1,10]]}]}]);
    let fixture = Fixture::new(scene);
    let result = fixture.ask(json!([{"node":"hero","read":"screen","at":[0,1]},
      {"field":"materials.optics.inputs.focus_distance","at":[0,1]}]));
    near(&result[1], &json!([5, 10]));
    let half = 800. / 9.;
    pixel_enclosure(
        &result[0][0]["box"],
        [160. - half, 120. - half, 160. + half, 120. + half],
    );
    pixel_enclosure(
        &result[0][1]["box"],
        [
            160. - half - 10. / 9.,
            120. - half - 5. / 9.,
            160. + half + 10. / 9.,
            120. + half + 5. / 9.,
        ],
    );
}

#[test]
fn screen_is_geometric_beyond_clip_planes_and_supports_orthographic_crossing() {
    let mut scene = triangle();
    scene["nodes"][0]["translation"] = json!([0, 0, 0]);
    scene["nodes"][1]["material"] = json!("optics");
    scene["materials"] =
        json!({"optics":{"program":"std/camera","inputs":{"clip_start":1,"clip_end":4}}});
    scene["animation"] = json!([{"tracks":[{"path":["nodes","hero","translation"],"keys":[[0,[0,0,0]],[1,[0,0,4.9]],[2,[0,0,10]]]}]}]);
    let fixture = Fixture::new(scene.clone());
    let values = fixture.ask(json!([{"node":"hero","read":"screen","at":[0,1,2]}]));
    assert!(values[0][0].is_object() && values[0][1].is_object());
    near(&values[0][0]["depth"], &json!([5, 5]));
    near(&values[0][1]["depth"], &json!([0.1, 0.1]));
    assert!(values[0][2].is_null());
    scene["animation"] = json!([]);
    scene["nodes"][1]["translation"] = json!([0, 0, 0]);
    scene["buffers"]["p"]["data"] = json!([[-1, -1, -1], [1, -1, 0], [0, 1, 1]]);
    scene["materials"]["optics"]["inputs"] =
        json!({"projection":"ortho","fit":"vertical","ortho_scale":4});
    let crossing = Fixture::new(scene);
    let values = crossing.ask(json!([{"node":"hero","read":"screen"}]));
    pixel_enclosure(&values[0]["box"], [100., 60., 220., 180.]);
    near(&values[0]["depth"], &json!([0, 1]));
}

#[test]
fn perspective_crossing_uses_the_authored_near_plane() {
    let mut scene = triangle();
    scene["nodes"][0]["translation"] = json!([0, 0, 0]);
    scene["nodes"][1]["translation"] = json!([0, 0, 0]);
    scene["nodes"][1]["material"] = json!("optics");
    scene["buffers"]["p"]["data"] = json!([[-1, -1, -1], [1, -1, 0], [0, 1, 1]]);
    scene["materials"] = json!({"optics":{"program":"std/camera","inputs":{"clip_start":0.25}}});
    let fixture = Fixture::new(scene);
    let result = fixture.ask(json!([{"node":"hero","read":"screen"}]));
    let half = 16000. / 9.;
    pixel_enclosure(
        &result[0]["box"],
        [160. - half, 120. - half, 160. + half, 120. + half],
    );
    near(&result[0]["depth"], &json!([0.25, 1]));
}

#[test]
fn forward_perspective_bounds_crossing_near_plane_are_clipped() {
    for plane in [4.25, 5., 5.75] {
        let mut scene = triangle();
        scene["nodes"][0]["translation"] = json!([0, 0, 0]);
        scene["nodes"][1]["material"] = json!("optics");
        scene["buffers"]["p"]["data"] = json!([[-1, -1, -1], [1, -1, 1], [0, 1, 0]]);
        scene["materials"] =
            json!({"optics":{"program":"std/camera","inputs":{"clip_start":plane}}});
        let fixture = Fixture::new(scene);
        let result = fixture.ask(json!([{"node":"hero","read":"screen"}]));
        let half = 4000. / (9. * plane);
        pixel_enclosure(
            &result[0]["box"],
            [160. - half, 120. - half, 160. + half, 120. + half],
        );
        near(&result[0]["depth"], &json!([plane, 6]));
    }
}

#[test]
fn imported_clip_skin_and_local_fields_share_the_sampled_pose() {
    let fixture = Fixture::new(json!({"version":2,"nodes":[
      {"type":"asset","name":"rig","file":"rig.gltf","translation":[2,0,0],"scale":[2,2,2]}],
      "animation":[{"clip":{"target":"rig","index":0}}]}));
    std::fs::write(
        fixture.0.join("rig.gltf"),
        include_str!("../../vork-render/tests/goldens/skinned-box.gltf"),
    )
    .unwrap();
    let result = fixture.ask(
        json!([{"field":"nodes.rig/joint.translation","at":[0,0.5,1]},
      {"node":"rig","read":"bounds","at":[0,1]},{"node":"rig/joint","read":"transform","at":0.5}]),
    );
    near(
        &result[0],
        &json!([[-0.7, 0, 0], [0, 0.3, 0], [0.7, 0.6, 0]]),
    );
    near(
        &result[1],
        &json!([
            [[-0.4, -1, -1], [1.6, 1, 1]],
            [[2.4, 0.2, -1], [4.4, 2.2, 1]]
        ]),
    );
    near(&result[2][12], &json!(2));
    near(&result[2][13], &json!(0.6));
}

#[test]
fn imported_morph_defaults_and_partial_named_tracks_match_explicit_zeroes() {
    for skinned in [false, true] {
        for animated in [false, true] {
            let mut answers = Vec::new();
            for explicit in [false, true] {
                let mut source: Value = serde_json::from_str(include_str!(
                    "../../vork-render/tests/goldens/skinned-box.gltf"
                ))
                .unwrap();
                source["nodes"][0]["name"] = json!("morphed");
                if !skinned {
                    source["nodes"][0].as_object_mut().unwrap().remove("skin");
                }
                let mesh = &mut source["meshes"][0];
                let position = mesh["primitives"][0]["attributes"]["POSITION"].clone();
                mesh["primitives"][0]["targets"] =
                    json!([{"POSITION":position},{"POSITION":position}]);
                mesh["extras"] = json!({"targetNames":["first","second"]});
                if explicit {
                    mesh["weights"] = json!([0, 0]);
                } else {
                    mesh.as_object_mut().unwrap().remove("weights");
                }
                source["nodes"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("weights");
                let mut scene = json!({"version":2,"nodes":[
                    {"type":"asset","name":"rig","file":"rig.gltf"}]});
                if animated {
                    scene["animation"] = json!([{"tracks":[{
                        "path":["nodes","rig","nodes","morphed","morph_weights","first"],
                        "keys":[[0,0],[1,0.5]]}]}]);
                }
                let fixture = Fixture::new(scene);
                std::fs::write(fixture.0.join("rig.gltf"), source.to_string()).unwrap();
                answers.push(fixture.ask(json!([{"node":"rig","read":"bounds","at":[0,1]}])));
            }
            near(&answers[0], &answers[1]);
            assert_eq!(answers[0][0][0] != answers[0][0][1], animated);
        }
    }
}

#[test]
fn gpu_written_geometry_refuses_only_dependent_answers() {
    let mut scene = triangle();
    scene["programs"] = json!({"move":{"source":"import vork; struct Inputs {}; void compute(ComputeContext ctx, Inputs i) { p_store(ctx.id.x,float3(0.)); }",
      "bindings":{"p":"p"},"dispatch":3,"schedule":"frame"}});
    let fixture = Fixture::new(scene);
    near(
        &fixture.ask(json!([{"field":"nodes.hero.translation"}])),
        &json!([[0, 0, 0]]),
    );
    for question in [
        json!({"buffer":"p"}),
        json!({"buffer":"p","read":"count"}),
        json!({"node":"hero","read":"bounds"}),
    ] {
        let output = fixture.run(&json!([question]), true);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(stderr.lines().count(), 1, "{stderr}");
        assert!(
            stderr.contains("not built yet: this answer needs the GPU program"),
            "{stderr}"
        );
    }
}

#[test]
fn refusals_are_one_line_and_never_emit_partial_answers() {
    let fixture = Fixture::new(triangle());
    for (question, needle) in [
        (json!({"node":"","read":"bounds"}), "give the node a name"),
        (json!({"node":"absent","read":"transform"}), "node"),
        (json!({"node":"hero","read":"coverage"}), "not built yet"),
        (json!({"node":"hero","read":"luminance"}), "not built yet"),
        (json!({"node":"hero","read":"info"}), "property"),
        (json!({"field":"nodes.hero.no_such_field"}), "field"),
        (json!({"buffer":"p","range":[2,2]}), "range"),
        (json!({"buffer":"p","file":"never.bin"}), "not built yet"),
        (
            json!({"node":"hero","read":"transform","instance":0}),
            "instance",
        ),
        (json!({"geometry":"tri"}), "not built yet"),
        (json!({"pixel":[1,2]}), "not built yet"),
        (json!({"ray":{"origin":[0,0,0]}}), "not built yet"),
        (json!({"text":"title"}), "not built yet"),
        (json!({"path":"curve"}), "not built yet"),
        (json!({"audio":"voice"}), "window"),
        (json!({"audio":"voice","window":[0,0.01]}), "unknown node"),
        (json!({"image":"plate"}), "not built yet"),
    ] {
        let result = fixture.run(&json!([{"field":"version"},question]), true);
        assert!(!result.status.success(), "{question}");
        assert!(result.stdout.is_empty(), "{question}");
        let error = String::from_utf8_lossy(&result.stderr);
        assert_eq!(error.lines().count(), 1, "{question}: {error}");
        assert!(error.contains(needle), "{question}: {error}");
    }
    assert!(!fixture.0.join("never.bin").exists());
}

#[test]
#[cfg(not(target_os = "macos"))]
fn cached_update_notices_do_not_add_diagnostic_lines() {
    let fixture = Fixture::new(triangle());
    std::fs::create_dir_all(fixture.0.join("vork")).unwrap();
    let checked = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    std::fs::write(fixture.0.join("vork/update-notice.json"),
        json!({"checked":checked,"release":{"version":"999.0.0","vorkUpdate":{"reason":"cached test notice"}}}).to_string()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_vork"))
        .env_remove("CI")
        .env_remove("VORK_NO_UPDATE_NOTIFIER")
        .env("XDG_CACHE_HOME", &fixture.0)
        .env("LOCALAPPDATA", &fixture.0)
        .arg("ask")
        .arg(fixture.0.join("scene.json"))
        .arg(r#"[{"node":"hero","read":"coverage"}]"#)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(stderr.lines().count(), 1, "{stderr}");
    assert!(stderr.contains("not built yet"), "{stderr}");
}

#[test]
fn compositor_fields_preserve_nullable_values_in_output_and_captures() {
    for fields in [
        json!({}),
        json!({"name":null,"resolution":null}),
        json!({"name":"glow","resolution":0.5}),
        json!({"name":"glow","resolution":[80,60]}),
    ] {
        let mut scene = triangle();
        let mut entry = fields.clone();
        entry["material"] = json!("effect");
        scene["materials"] = json!({"effect":{"program":"std/bloom","inputs":{
            "source":{"channel":"color"},"original":{"channel":"color"}}}});
        scene["composition"]["space"] = "3d".into(); scene["composition"]["compositor"] = json!([entry]);
        scene["images"] = json!({"capture":{"composition":{"compositor":[entry]},"render":{
            "camera":"camera"}}});
        let fixture = Fixture::new(scene);
        for prefix in ["composition", "images.capture.composition"] {
            let value = fixture.ask(json!([
                {"field":format!("{prefix}.compositor.0.name")},
                {"field":format!("{prefix}.compositor.0.resolution")},
                {"field":format!("{prefix}.compositor")}
            ]));
            assert_eq!(value[0], fields["name"]);
            assert_eq!(value[1], fields["resolution"]);
            assert_eq!(value[2][0]["name"], fields["name"]);
            assert_eq!(value[2][0]["resolution"], fields["resolution"]);
            assert!(value[2][0].as_object().unwrap().contains_key("name"));
            assert!(value[2][0].as_object().unwrap().contains_key("resolution"));
        }
    }
}
