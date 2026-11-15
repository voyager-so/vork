use super::*;
use serde_json::json;

fn render(r: &Renderer, source: &str, color: f32) -> Result<FrameOutput> {
    render_samples(r, source, color, 1)
}
fn render_samples(r: &Renderer, source: &str, color: f32, samples: u32) -> Result<FrameOutput> {
    let scene = serde_json::from_value(
        json!({"version":2,"composition":{"space":"3d","view":{"program":"display"}},"render":{"dither":false,"resolution":[8,8],"background":[color,color,color],"settings":{"integrator":"raster","samples":samples,"reflections":"environment"}},
        "programs":{"display":{"source":source}},"nodes":[{"type":"camera"}]}),
    )?;
    let scene = r.install(scene, Path::new("."))?;
    let mut frame = scene.create_frame_slot(r, ReadbackPrecision::Float)?;
    frame.submit_outputs(
        &scene,
        r,
        0.,
        ReadbackOutputs {
            color: true,
            float: FloatOutputs {
                color: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    frame.collect(r)
}
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn display_program_preserves_sharp_thresholds_and_unclamped_hdr_inputs() -> Result<()> {
    let r = crate::test_renderer()?;
    let source = "struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs i) {return finish_view(float3(ctx.color.x > .181 ? 1. : 0.),ctx,1.);}";
    for (color, expected) in [(0.18, 0), (0.19, 255)] {
        let result = render(&r, source, color)?;
        assert_eq!(&result.rgba[..3], &[expected; 3]);
        assert!((result.linear_rgba.as_ref().unwrap()[0][0] - color).abs() < 0.001);
    }
    let result = render(
        &r,
        "struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs i) {return finish_view(ctx.color/100000.,ctx,1.);}",
        50000.,
    )?;
    assert!(
        (i32::from(result.rgba[0]) - 128).abs() <= 1,
        "{:?}",
        &result.rgba[..4]
    );
    assert!(result.linear_rgba.as_ref().unwrap()[0][0] > 49000.);
    Ok(())
}
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn display_program_nonfinite_result_is_a_frame_error() -> Result<()> {
    let r = crate::test_renderer()?;
    let result = render(
        &r,
        "struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs i) {return finish_view(float3(1./(ctx.color.x-ctx.color.x)),ctx,1.);}",
        0.18,
    );
    let error = format!(
        "{:#}",
        result
            .err()
            .context("nonfinite display program succeeded")?
    );
    assert!(
        error.contains("view returned a non-finite colour"),
        "{error}"
    );
    Ok(())
}
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn view_result_forms_reach_the_destination_one_way() -> Result<()> {
    let r = crate::test_renderer()?;
    // The helper sets no dither (render.dither false), so every code below is exact.
    // Code values for the destination pass as they are.
    let codes = render(&r, "struct Inputs {}; ViewResult view(ViewContext ctx, Inputs i) {return view_from_codes(float3(64.,128.,191.)/255.,ctx.display);}", 0.18)?;
    assert_eq!(&codes.rgba[..3], &[64, 128, 191]);
    // Absolute XYZ is converted by the engine: D65 white at the reference white is sRGB white, and half its
    // luminance is the sRGB code for linear one half.
    for (scale, expected) in [(1., 255), (0.5, 188)] {
        let source = format!("struct Inputs {{}}; ViewResult view(ViewContext ctx, Inputs i) {{return view_from_xyz(float3(.95045593,1.,1.08905775)*ctx.reference_white*{scale:?});}}");
        let xyz = render(&r, &source, 0.18)?;
        for channel in &xyz.rgba[..3] { assert!((i32::from(*channel) - expected).abs() <= 1, "{:?}", &xyz.rgba[..4]); }
    }
    // std.view's helper returns the destination's codes: the same picture as returning them directly.
    let helper = render(&r, "struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs i) {return finish_view(float3(64.,128.,191.)/255.,ctx,1.);}", 0.18)?;
    assert_eq!(helper.rgba, codes.rgba);
    Ok(())
}
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn view_codes_for_another_display_are_refused() -> Result<()> {
    let r = crate::test_renderer()?;
    let result = render(&r, "struct Inputs {}; ViewResult view(ViewContext ctx, Inputs i) {return view_from_codes(float3(.5),DisplayTarget::DisplayP3);}", 0.18);
    let error = format!("{:#}", result.err().context("codes for another display were accepted")?);
    assert!(error.contains("code values for another display"), "{error}");
    Ok(())
}

/// One undithered pixel as 16-bit code values over one: `view` at the scene-linear `color`, encoded by the
/// display stage for the target `display`.
fn encoded(r: &Renderer, view: serde_json::Value, programs: serde_json::Value, color: [f32; 3], display: crate::output_device::Target) -> Result<[f32; 3]> {
    let readback = ReadbackOutputs { color: true, color16: true, ..Default::default() };
    let scene = serde_json::from_value(
        json!({"version":2,"composition":{"space":"3d","view":view},"programs":programs,
        "render":{"dither":false,"resolution":[8,8],"background":color,"settings":{"integrator":"raster","samples":1,"reflections":"environment"}},
        "nodes":[{"type":"camera"}]}),
    )?;
    let scene = r.install_with_options(scene, Path::new("."), InstallOptions {
        readback, gpu_color: false, display, ..Default::default()
    })?;
    let mut frame = scene.create_frame_slot(r, ReadbackPrecision::Packed)?;
    frame.submit_outputs(&scene, r, 0., readback)?;
    let pixel = frame.collect(r)?.rgba16[0];
    Ok([0, 1, 2].map(|c| f32::from(pixel[c]) / 65535.))
}
fn srgb_encode(v: f32) -> f32 { if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1. / 2.4) - 0.055 } }
fn srgb_decode(v: f32) -> f32 { if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) } }
/// Linear Rec.709 as Display P3 code values, the reference for what the stage must produce.
fn display_p3(c: [f32; 3]) -> [f32; 3] {
    [[0.82246197, 0.17753803, 0.], [0.03319420, 0.96680580, 0.], [0.01708263, 0.07239744, 0.91051993]]
        .map(|row: [f32; 3]| srgb_encode(row[0] * c[0] + row[1] * c[1] + row[2] * c[2]))
}
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn bundled_views_match_direct_ocio_for_display_p3() -> Result<()> {
    let to_p3 = crate::output_device::Target::for_display(vork_scene::v2::OutputDisplay::DisplayP3);
    let r = crate::test_renderer()?;
    // Direct OpenColorIO results for Blender 5.2.1's Standard and AgX views (scripts/color/bake-device-views.py).
    let reference: serde_json::Value = serde_json::from_str(include_str!("../data/device-reference.json"))?;
    for family in ["standard", "agx"] {
        let row = reference[family].as_array().unwrap().iter().find(|row| row["display"] == "Display P3").context("Display P3 reference")?;
        // A scene background is nonnegative; the grey ramp and the random colours are.
        let patches: Vec<_> = row["samples"].as_array().unwrap().iter()
            .filter(|patch| patch["input"].as_array().unwrap().iter().all(|v| v.as_f64().unwrap() >= 0.)).collect();
        let mut worst = 0f32;
        for patch in patches.iter().step_by(16) {
            let input = [0, 1, 2].map(|c| patch["input"][c].as_f64().unwrap() as f32);
            let actual = encoded(&r, json!({"program": format!("std/{family}")}), json!({}), input, to_p3)?;
            for c in 0..3 {
                let expected = (patch["output"][c].as_f64().unwrap() as f32).clamp(0., 1.);
                worst = worst.max((actual[c] - expected).abs());
            }
        }
        eprintln!("std/{family} Display P3: worst encoded error {worst}");
        assert!(worst < 0.0005, "std/{family} Display P3: {worst}");
    }
    Ok(())
}
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn display_p3_is_encoded_by_the_display_stage_from_either_result_form() -> Result<()> {
    use vork_scene::v2::OutputDisplay;
    let (to_p3, to_srgb) = (crate::output_device::Target::for_display(OutputDisplay::DisplayP3), crate::output_device::Target::default());
    let r = crate::test_renderer()?;
    let close = |actual: [f32; 3], expected: [f32; 3], tolerance: f32, what: &str| {
        for c in 0..3 { assert!((actual[c] - expected[c]).abs() <= tolerance, "{what}: {actual:?} vs {expected:?}"); }
    };
    let authored = |source: &str| (json!({"program":"authored"}), json!({"authored":{"source":source}}));
    // The view is told its destination, and code values for it pass as they are.
    let (view, programs) = authored("struct Inputs {}; ViewResult view(ViewContext ctx, Inputs i) {return view_from_codes(ctx.display == DisplayTarget::DisplayP3 ? float3(.25,.5,.75) : float3(0.),ctx.display);}");
    close(encoded(&r, view, programs, [0.18; 3], to_p3)?, [0.25, 0.5, 0.75], 1. / 65535., "codes");
    // XYZ is converted by the engine: sRGB red lies inside Display P3.
    let red = display_p3([1., 0., 0.]);
    let (view, programs) = authored("struct Inputs {}; ViewResult view(ViewContext ctx, Inputs i) {return view_from_xyz(float3(.4123908,.21263901,.01933082)*ctx.reference_white);}");
    close(encoded(&r, view, programs, [0.18; 3], to_p3)?, red, 2e-4, "XYZ");
    close(red, [0.9175, 0.2003, 0.1386], 2e-4, "reference");
    // std/standard encodes for the display itself; its creative gamma leaves unit red alone, as it does in sRGB.
    for gamma in [1., 2.] {
        close(encoded(&r, json!({"program":"std/standard","inputs":{"gamma":gamma}}), json!({}), [1., 0., 0.], to_p3)?, red, 2e-4, "std/standard");
    }
    // An sRGB rendering returned as XYZ is converted by the engine. A look is such a rendering; a render call
    // refuses it for Display P3 (output_device::admit), and the stage itself stays general.
    let look = json!({"program":"std/agx","inputs":{"look":"punchy"}});
    let srgb = encoded(&r, look.clone(), json!({}), [0.5, 0.2, 0.1], to_srgb)?;
    close(encoded(&r, look, json!({}), [0.5, 0.2, 0.1], to_p3)?, display_p3(srgb.map(srgb_decode)), 1e-3, "AgX look");
    // The default look takes AgX's own Display P3 rendering, which is not that conversion. The two agree
    // wherever AgX's colour lies inside sRGB (a bright colour it desaturates, such as [4, 0.2, 0.1], is the same
    // colour for both displays), so this uses a saturated orange-red whose Display P3 rendering lies outside
    // sRGB: a patch of the bake's direct OpenColorIO rows (device-reference.json, family agx).
    let plain = json!({"program":"std/agx"});
    let saturated = [0.4134502410888672, 0.03218769654631615, 0.002290958072990179];
    let wide = encoded(&r, plain.clone(), json!({}), saturated, to_p3)?;
    close(wide, [0.6068548560142517, 0.22532910108566284, 0.029246654361486435], 5e-4, "AgX Display P3 against OpenColorIO");
    // Its blue in linear Rec.709 is negative: no sRGB picture, converted, can be this colour.
    let light = wide.map(srgb_decode);
    let blue = -0.01964 * light[0] - 0.07864 * light[1] + 1.09827 * light[2];
    assert!(blue < -0.003, "AgX Display P3 is inside sRGB: {wide:?} (linear Rec.709 blue {blue})");
    let srgb = encoded(&r, plain, json!({}), saturated, to_srgb)?;
    let converted = display_p3(srgb.map(srgb_decode));
    assert!((wide[2] - converted[2]).abs() > 0.02, "AgX Display P3 is the sRGB picture: {wide:?} against {converted:?}");
    // Code values for another display than the destination still refuse.
    let (view, programs) = authored("struct Inputs {}; ViewResult view(ViewContext ctx, Inputs i) {return view_from_codes(float3(.5),DisplayTarget::Srgb);}");
    let error = format!("{:#}", encoded(&r, view, programs, [0.18; 3], to_p3).err().context("sRGB codes were accepted for Display P3")?);
    assert!(error.contains("code values for another display"), "{error}");
    Ok(())
}

/// ST 2084 code value to cd/m².
fn pq_nits(code: f32) -> f32 {
    let v = f64::from(code).powf(32. / 2523.);
    (10000. * ((v - 3424. / 4096.).max(0.) / (2413. / 128. - 2392. / 128. * v)).powf(16384. / 2610.)) as f32
}
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn bundled_views_match_direct_ocio_for_rec2100() -> Result<()> {
    use vork_scene::v2::OutputDisplay;
    let r = crate::test_renderer()?;
    let reference: serde_json::Value = serde_json::from_str(include_str!("../data/device-reference.json"))?;
    // OpenColorIO's Standard and the published AgX HDR view put 100 cd/m² at one display unit; `agx_203` is
    // the bake's 203 cd/m² recipe. All are for a 1000 cd/m² display.
    for (family, program, white) in [("standard", "std/standard", 100.), ("agx", "std/agx", 100.), ("agx_203", "std/agx", 203.)] {
        for display in [OutputDisplay::Rec2100Pq, OutputDisplay::Rec2100Hlg] {
            let target = crate::output_device::Target { display, peak: 1000., white };
            let row = reference[family].as_array().unwrap().iter().find(|row| row["display"] == display.name()).context("Rec.2100 reference")?;
            let patches: Vec<_> = row["samples"].as_array().unwrap().iter()
                .filter(|patch| patch["input"].as_array().unwrap().iter().all(|v| v.as_f64().unwrap() >= 0.)).collect();
            // OpenColorIO's PQ has no display peak; the stage limits to it, so compare below it.
            let ceiling = if display == OutputDisplay::Rec2100Pq { 0.7518271 } else { 1. };
            let mut worst = 0f32;
            for patch in patches.iter().step_by(16) {
                let input = [0, 1, 2].map(|c| patch["input"][c].as_f64().unwrap() as f32);
                let actual = encoded(&r, json!({"program": program}), json!({}), input, target)?;
                for c in 0..3 {
                    let expected = (patch["output"][c].as_f64().unwrap() as f32).clamp(0., ceiling);
                    worst = worst.max((actual[c].min(ceiling) - expected).abs());
                }
            }
            eprintln!("{family} {}: worst encoded error {worst}", display.name());
            assert!(worst < 0.0005, "{family} {}: {worst}", display.name());
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn rec2100_luminances_reach_the_view_and_bound_the_signal() -> Result<()> {
    use vork_scene::v2::OutputDisplay::{Rec2100Hlg, Rec2100Pq};
    use crate::output_device::Target;
    let r = crate::test_renderer()?;
    let authored = |source: &str| (json!({"program":"authored"}), json!({"authored":{"source":source}}));
    let (pq, hlg) = (Target::for_display(Rec2100Pq), Target::for_display(Rec2100Hlg));
    // The view is told the destination's luminances.
    let (view, programs) = authored("struct Inputs {}; ViewResult view(ViewContext ctx, Inputs i) {return view_from_codes(float3(ctx.peak_luminance/10000.,ctx.reference_white/1000.,0.),ctx.display);}");
    let told = encoded(&r, view, programs, [0.18; 3], Target { peak: 4000., white: 300., ..pq })?;
    assert!((told[0] - 0.4).abs() < 1e-4 && (told[1] - 0.3).abs() < 1e-4, "{told:?}");
    // XYZ is absolute: D65 white at the reference white is 203 cd/m² in PQ, and the engine limits to the peak.
    let white = "struct Inputs {}; ViewResult view(ViewContext ctx, Inputs i) {return view_from_xyz(float3(.95045593,1.,1.08905775)*ctx.reference_white*SCALE);}";
    for (scale, nits) in [("1.", 203.), ("100.", 1000.)] {
        let (view, programs) = authored(&white.replace("SCALE", scale));
        let code = encoded(&r, view, programs, [0.18; 3], pq)?;
        for c in 0..3 { assert!((pq_nits(code[c]) - nits).abs() < nits * 0.005, "{scale}: {code:?}"); }
    }
    // The same XYZ as HLG for a 1000 cd/m² display, and std/standard's own encoding of unit white.
    let (view, programs) = authored(&white.replace("SCALE", "1."));
    let xyz = encoded(&r, view, programs, [0.18; 3], hlg)?;
    let standard = encoded(&r, json!({"program":"std/standard"}), json!({}), [1.; 3], hlg)?;
    for c in 0..3 { assert!((xyz[c] - 0.74988).abs() < 3e-4 && (standard[c] - 0.74988).abs() < 3e-4, "{xyz:?} {standard:?}"); }
    let peak = encoded(&r, json!({"program":"std/standard"}), json!({}), [100.; 3], hlg)?;
    assert!((peak[1] - 1.).abs() < 1e-4, "{peak:?}");
    let standard = encoded(&r, json!({"program":"std/standard"}), json!({}), [1.; 3], pq)?;
    assert!((pq_nits(standard[1]) - 203.).abs() < 1., "{standard:?}");
    // Code values above the display's peak do not leave the stage.
    let (view, programs) = authored("struct Inputs {}; ViewResult view(ViewContext ctx, Inputs i) {return view_from_codes(float3(1.),ctx.display);}");
    let limited = encoded(&r, view, programs, [0.18; 3], pq)?;
    assert!((limited[0] - 0.7518271).abs() < 1e-4, "{limited:?}");
    // std/agx: unit scene white lands on the default reference white; the published view (reference white
    // 100) puts it near 148 cd/m²; both reach the same 1000 cd/m² ceiling, in PQ and through HLG alike.
    let agx = json!({"program":"std/agx"});
    let calibrated = encoded(&r, agx.clone(), json!({}), [1.; 3], pq)?;
    assert!((pq_nits(calibrated[1]) - 203.).abs() < 1., "{calibrated:?}");
    let published = encoded(&r, agx.clone(), json!({}), [1.; 3], Target { white: 100., ..pq })?;
    assert!((pq_nits(published[1]) - 148.3).abs() < 1., "{published:?}");
    for target in [pq, Target { white: 100., ..pq }] {
        let bright = encoded(&r, agx.clone(), json!({}), [256.; 3], target)?;
        assert!((pq_nits(bright[1]) - 1000.).abs() < 5., "{bright:?}");
    }
    let bright = encoded(&r, agx.clone(), json!({}), [256.; 3], hlg)?;
    assert!(bright[1] > 0.99, "{bright:?}");
    // A creative gamma grades the HLG signal, so it cannot exceed the peak in either encoding.
    let graded = encoded(&r, json!({"program":"std/agx","inputs":{"gamma":2.}}), json!({}), [256.; 3], pq)?;
    assert!(pq_nits(graded[1]) < 1005., "{graded:?}");
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn display_program_runs_only_after_final_sample() -> Result<()> {
    let r = crate::test_renderer()?;
    // Scratch is initially zero. Evaluating before resolve's final sample would
    // report a non-finite value, even though the actual scene color is positive.
    let source =
        "struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs i) {return finish_view(.09/ctx.color,ctx,1.);}";
    let single = render_samples(&r, source, 0.18, 1)?;
    let accumulated = render_samples(&r, source, 0.18, 4)?;
    crate::test_display::assert_display_eq(&single.rgba, &accumulated.rgba);
    let wrong = render_samples(&r, "struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs i) {return finish_view(ctx.color,ctx,1.);}", 0.18, 4)?;
    // Bypassing the transform preserves linear readback but must fail here.
    crate::test_linear::assert_linear_eq(wrong.linear_rgba.as_ref().unwrap(), accumulated.linear_rgba.as_ref().unwrap());
    crate::test_display::assert_display_ne(&wrong.rgba, &accumulated.rgba);
    assert!((i32::from(accumulated.rgba[0]) - 128).abs() <= 1);
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn cloth_rest_edge_filter_keeps_small_crossing_edges_out_of_contacts() -> Result<()> {
    let r = crate::test_renderer()?;
    // Execute the shipped Slang functions, rather than a Rust copy of their
    // arithmetic. Display output is only a compact GPU numeric receipt.
    let library = include_str!("../../../libraries/cloth/self.slang");
    let segments = library.split_once("void cloth_segments(").unwrap().1
        .split_once("void cloth_register(").unwrap().0;
    let rest_edges = library.split_once("bool cloth_rest_edges(").unwrap().1
        .split_once("bool cloth_self_boxes(").unwrap().0;
    let source = format!(r#"
        static const float SELF_MARGIN=3e-3;
        void cloth_segments({segments}
        float3 rest_position_load(uint i) {{
            if(i==0) return float3(-4e-5,0,0);
            if(i==1) return float3(4e-5,0,0);
            if(i==2) return float3(0,-4e-5,0);
            return float3(0,4e-5,0);
        }}
        bool cloth_rest_edges({rest_edges}
        struct Inputs {{}};
        import std.view; ViewResult view(ViewContext ctx, Inputs inputs) {{
            bool ok=cloth_rest_edges(uint4(0,1,2,3),6e-6);
            ok=ok&&!cloth_rest_edges(uint4(0,1,2,3),0);
            for(uint i=0;i<3;i++) {{
                float scale=i==0?8e-5:(i==1?1.:100.);
                float3 p=float3(-.5,0,0)*scale,q=-p;
                float3 a=float3(0,-.5,0)*scale,b=-a;
                float s,t;cloth_segments(p,q,a,b,s,t);
                ok=ok&&abs(s-.5)<1e-5&&abs(t-.5)<1e-5;
                // Parallel, separated perpendicular and zero-length edges.
                a=float3(-.5,1,0)*scale;b=float3(.5,1,0)*scale;
                cloth_segments(p,q,a,b,s,t);
                ok=ok&&abs(length(lerp(p,q,s)-lerp(a,b,t))/scale-1)<1e-5;
                a=float3(0,-.5,1)*scale;b=float3(0,.5,1)*scale;
                cloth_segments(p,q,a,b,s,t);
                ok=ok&&abs(length(lerp(p,q,s)-lerp(a,b,t))/scale-1)<1e-5;
                cloth_segments(float3(0),float3(0),a,b,s,t);
                ok=ok&&s==0&&abs(t-.5)<1e-5;
                cloth_segments(a,b,float3(0),float3(0),s,t);
                ok=ok&&t==0&&abs(s-.5)<1e-5;
                cloth_segments(float3(0),float3(0),float3(0),float3(0),s,t);
                ok=ok&&s==0&&t==0;
                cloth_segments(float3(0),float3(1e-10,0,0)*scale,a,b,s,t);
                ok=ok&&s==0&&abs(t-.5)<1e-5;
            }}
            return finish_view(ok?float3(1):float3(0),ctx,1.);
        }}
    "#);
    let result = render(&r, &source, 0.)?;
    assert!(result.rgba.chunks_exact(4).all(|pixel| pixel[..3] == [255; 3]),
        "cloth closest-point/rest-edge numeric receipt failed: {:?}", &result.rgba[..4]);
    Ok(())
}
