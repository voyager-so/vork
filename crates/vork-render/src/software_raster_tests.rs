//! Cross-consumer invariants for quantised raster coverage, independent of the
//! driver's interpolants. Run with VORK_GPU_TESTS=1 VORK_GPU=cpu for lavapipe.
use crate::*;

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn bounded_triangle_depth_matches_shading_and_preserves_nearer_surfaces() -> Result<()> {
    let r = crate::test_renderer()?;
    let software = r.adapter_info.device_type == wgpu::DeviceType::Cpu;
    let mut covered = 0;
    for rotation in 0..4 {
        for slope in [-1., 1.] {
            // Finite, almost collinear viewport corners. Rotate and reverse the
            // depth gradient so neither a particular edge nor slope is special.
            let xy = [
                [-7.039084434509277, 15.120407104492188],
                [-9.2980318069458, -12.10944938659668],
                [-9.188946723937988, -10.793975830078125],
            ];
            let positions: Vec<_> = xy
                .into_iter()
                .enumerate()
                .map(|(i, [mut x, mut y])| {
                    for _ in 0..rotation {
                        (x, y) = (-y, x);
                    }
                    [x, y, -(0.601 + slope * (i as f64 - 1.) * 0.001)]
                })
                .collect();
            let mesh = |name: &str,
                        material: &str,
                        points: serde_json::Value,
                        indices: serde_json::Value| {
                serde_json::json!({
                "type":"mesh","name":name,"materials":[material],"geometry":{"topology":"triangles",
                "attributes":{"POSITION":{"type":"vec3","data":points},"NORMAL":{"type":"vec3","data":vec![[0.,0.,1.];points.as_array().unwrap().len()]}},
                "indices":{"type":"u32","data":indices}}})
            };
            for occluded in [false, true] {
                let mut nodes = vec![mesh(
                    "sliver",
                    "probe",
                    serde_json::json!(positions),
                    serde_json::json!([0, 1, 2]),
                )];
                if occluded {
                    nodes.push(mesh(
                        "nearer",
                        "blue",
                        serde_json::json!([
                            [-16., -16., -0.59],
                            [16., -16., -0.59],
                            [16., 16., -0.59],
                            [-16., 16., -0.59]
                        ]),
                        serde_json::json!([0, 1, 2, 0, 2, 3]),
                    ));
                }
                nodes.push(serde_json::json!({"type":"camera","name":"camera","material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":32,"fit":"horizontal","clip_start":0.1,"clip_end":1.1}}}));
                let doc = serde_json::json!({"version":2,"programs":{
                    "point":{"source":"import vork; struct Inputs {}; Closure surface(ShadingContext ctx, Inputs i) { return unlit(float3(-ctx.POSITION.z,1.,0.)); }"},
                    "blue":{"source":"import vork; struct Inputs {}; Closure surface(ShadingContext ctx, Inputs i) { return unlit(float3(0.,0.,1.)); }"}},
                    "materials":{"probe":{"program":"point"},"blue":{"program":"blue"}},"nodes":nodes,
                    "render":{"camera":"camera","resolution":[32,32],"background":"transparent","settings":{"integrator":"raster","samples":1,"denoiser":null,"fast_gi":null}}});
                let mut installed = r.install(test_scene(doc, Path::new("."))?, Path::new("."))?;
                let mut slot = installed.create_frame_slot(&r, ReadbackPrecision::Float)?;
                let result = installed.render_frame(&r, &mut slot, 0.)?;
                let colours = result.linear_rgba.as_ref().unwrap();
                let depths = result.depth_f32.as_ref().unwrap();
                for (pixel, (&id, (&z, c))) in result
                    .ids
                    .iter()
                    .zip(depths.iter().zip(colours))
                    .enumerate()
                {
                    if occluded {
                        assert_eq!(
                            id, 2,
                            "nearer plane lost at {pixel}, rotation {rotation}, slope {slope}"
                        );
                        assert!(
                            (z - 0.59).abs() < 2e-6
                                && c[0].abs() < 2e-6
                                && (c[2] - 1.).abs() < 2e-6,
                            "wrong nearer point {pixel}: depth {z}, colour {c:?}"
                        );
                    } else if id != 0 {
                        covered += 1;
                        assert!(
                            (0.6 - 2e-6..=0.602 + 2e-6).contains(&z),
                            "out-of-range depth {z}"
                        );
                        // Hardware retains its native raster interpolation;
                        // software additionally shares the retained shade point.
                        if software {
                            assert!(
                                (z - c[0]).abs() < 2e-6,
                                "depth/colour disagree at {pixel}: {z} / {c:?}"
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(covered > 0, "test must exercise quantised sliver coverage");
    Ok(())
}
