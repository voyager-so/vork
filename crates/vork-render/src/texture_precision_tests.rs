//! Material images keep their file precision: 16-bit integer and float channels
//! reach shading as half float, 8-bit channels as 8-bit.
use super::*;
use serde_json::{Value, json};

fn directory(name: &str) -> Result<PathBuf> {
    let base = std::env::temp_dir().join(format!("vork-texture-precision-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&base)?;
    Ok(base)
}

/// An emission-only quad filling a `width`x`height` orthographic frame with
/// `image` (nearest texels unless `filter` names a min filter).
fn emission_scene(width: u32, height: u32, image: Value) -> Value {
    let aspect = height as f32 / width as f32;
    json!({"version":2,
        "composition":{"space":"3d","view":{"program":"std/standard","inputs":{}}},"render":{"dither":false,"resolution":[width,height],"background":[0,0,0],"settings":{"integrator":"raster","samples":1,"reflections":"environment"}},
        "images":{"source":image},
        "materials":{"m":{"inputs":{"base_weight":0,"specular_weight":0,"emission_luminance":1,
            "emission_color":{"image":"source"}}}},
        "buffers":{"p":{"type":"vec3","data":[[-1,-aspect,0],[1,-aspect,0],[1,aspect,0],[-1,aspect,0]]},
            "uv":{"type":"vec2","data":[[0,1],[1,1],[1,0],[0,0]]},
            "n":{"type":"vec3","data":[[0,0,1],[0,0,1],[0,0,1],[0,0,1]]},
            "i":{"type":"u32","data":[0,1,2,0,2,3]}},
        "geometry":{"quad":{"attributes":{"POSITION":"p","NORMAL":"n","TEXCOORD_0":"uv"},"indices":"i"}},
        "nodes":[{"type":"mesh","geometry":"quad","materials":["m"]},
            {"type":"camera","translation":[0,0,3],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":2}}}]})
}

/// Scene-linear RGBA before the view transform.
fn render_linear(r: &Renderer, scene: Value, base: &Path) -> Result<Vec<[f32; 4]>> {
    let scene = r.install(vork_scene::v2::from_test_value(scene)?, base)?;
    let mut frame = scene.create_frame_slot(r, ReadbackPrecision::Float)?;
    frame.submit_outputs(&scene, r, 0., ReadbackOutputs {
        float: FloatOutputs { color: true, ..Default::default() },
        ..Default::default()
    })?;
    frame.collect(r)?.linear_rgba.context("linear colour readback")
}

fn nearest(file: &str, color_space: &str) -> Value {
    json!({"file":file,"color_space":color_space,"mag_filter":"nearest","min_filter":"nearest"})
}

fn distinct(row: &[[f32; 4]]) -> usize {
    let mut values: Vec<u32> = row.iter().map(|p| p[0].to_bits()).collect();
    values.sort_unstable();
    values.dedup();
    values.len()
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn sixteen_bit_png_gradient_keeps_more_than_256_levels() -> Result<()> {
    let r = crate::test_renderer()?;
    let base = directory("png16")?;
    let width = 1024u32;
    let codes: Vec<u16> = (0..width).map(|x| ((x as f64 * 65535. / (width - 1) as f64).round()) as u16).collect();
    image::ImageBuffer::<image::Luma<u16>, _>::from_raw(width, 1, codes.clone()).unwrap().save(base.join("ramp16.png"))?;
    let bytes: Vec<u8> = codes.iter().map(|c| ((u32::from(*c) + 128) / 257) as u8).collect();
    image::GrayImage::from_raw(width, 1, bytes).unwrap().save(base.join("ramp8.png"))?;
    let row = |pixels: Vec<[f32; 4]>| pixels[..width as usize].to_vec();
    for (color_space, decode) in [("Non-Color", false), ("sRGB", true)] {
        let sixteen = row(render_linear(&r, emission_scene(width, 4, nearest("ramp16.png", color_space)), &base)?);
        let eight = row(render_linear(&r, emission_scene(width, 4, nearest("ramp8.png", color_space)), &base)?);
        eprintln!("{color_space}: 16-bit ramp {} levels, 8-bit ramp {} levels", distinct(&sixteen), distinct(&eight));
        assert!(distinct(&eight) <= 256, "{color_space}: 8-bit control has {} levels", distinct(&eight));
        assert!(distinct(&sixteen) > 900, "{color_space}: 16-bit ramp kept only {} levels", distinct(&sixteen));
        let mut error = 0f32;
        for (x, p) in sixteen.iter().enumerate() {
            let v = f32::from(codes[x]) / 65535.;
            let expected = if decode { texture_source::decode_srgb(v) } else { v };
            // Half-float texture and colour target: 11 significant bits.
            error = error.max((p[0] - expected).abs() / expected.max(1. / 64.));
            assert!(p[0] == p[1] && p[1] == p[2], "grey stays grey: {p:?}");
        }
        assert!(error < 2e-3, "{color_space}: 16-bit ramp relative error {error}");
    }
    std::fs::remove_dir_all(base)?;
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn float_exr_and_hdr_values_above_one_render_above_one() -> Result<()> {
    let r = crate::test_renderer()?;
    let base = directory("float")?;
    // Exact in RGBE as well as binary32; 1e6 clamps to the half-float maximum.
    let values = [0.25f32, 1.5, 4., 16., 300., 1000., 60000., 1e6];
    let width = values.len();
    exr::prelude::write_rgb_file(base.join("ramp.exr"), width, 1, |x, _| (values[x], values[x] * 0.5, values[x] * 2.))?;
    image::Rgb32FImage::from_fn(width as u32, 1, |x, _| image::Rgb([values[x as usize]; 3])).save(base.join("ramp.hdr"))?;
    for (file, tolerance) in [("ramp.exr", 2e-3f32), ("ramp.hdr", 1e-2)] {
        // Float files default to linear (Blender's default colour space for .exr/.hdr).
        let image = json!({"file":file,"mag_filter":"nearest","min_filter":"nearest"});
        let pixels = render_linear(&r, emission_scene(width as u32, 2, image), &base)?;
        for (x, value) in values.iter().enumerate() {
            let expected = if file.ends_with("exr") { [*value, value * 0.5, value * 2.] } else { [*value; 3] };
            for (channel, expected) in expected.into_iter().enumerate() {
                let expected = expected.min(constants::HALF_FLOAT_MAX);
                let got = pixels[x][channel];
                assert!((got - expected).abs() <= tolerance * expected, "{file} texel {x} channel {channel}: {got}, expected {expected}");
            }
        }
    }
    std::fs::remove_dir_all(base)?;
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn float_mips_average_values_above_one() -> Result<()> {
    let r = crate::test_renderer()?;
    let base = directory("mips")?;
    // A one-texel 0/8 checker minified 16x: only the filtered mip chain gives 4.
    exr::prelude::write_rgb_file(base.join("checker.exr"), 128, 128, |x, y| {
        let v = if (x + y) % 2 == 0 { 8f32 } else { 0. };
        (v, v, v)
    })?;
    image::ImageBuffer::<image::Luma<u16>, _>::from_fn(128, 128, |x, y| image::Luma([if (x + y) % 2 == 0 { 65535 } else { 0 }]))
        .save(base.join("checker.png"))?;
    for (file, expected) in [("checker.exr", 4f32), ("checker.png", 0.5)] {
        let image = json!({"file":file,"color_space":"Non-Color"});
        let pixels = render_linear(&r, emission_scene(8, 8, image), &base)?;
        for p in &pixels {
            assert!((p[0] - expected).abs() < 0.02 * expected, "{file}: minified checker {p:?}, expected {expected}");
        }
    }
    std::fs::remove_dir_all(base)?;
    Ok(())
}

#[test]
fn half_texels_expand_decode_and_clamp() -> Result<()> {
    use gltf::image::{Data, Format};
    let f16 = |bits: u16| half::f16::from_bits(bits).to_f32();
    let la16 = Data { width: 1, height: 1, format: Format::R16G16, pixels: [32768u16, 65535].iter().flat_map(|v| v.to_ne_bytes()).collect() };
    let texels = materials::half_texels(&la16, true)?;
    let grey = texture_source::decode_srgb(32768. / 65535.);
    assert!((f16(texels[0]) - grey).abs() < 1e-3 && texels[0] == texels[1] && texels[1] == texels[2]);
    assert_eq!(f16(texels[3]), 1., "alpha is never sRGB-decoded");
    let rgb = Data { width: 1, height: 1, format: Format::R32G32B32FLOAT, pixels: [-2f32, 3.5, 1e9].iter().flat_map(|v| v.to_ne_bytes()).collect() };
    let texels = materials::half_texels(&rgb, false)?;
    assert_eq!(texels.iter().map(|t| f16(*t)).collect::<Vec<_>>(), [-2., 3.5, constants::HALF_FLOAT_MAX, 1.]);
    let nan = Data { width: 1, height: 1, format: Format::R32G32B32FLOAT, pixels: [f32::NAN, 0., 0.].iter().flat_map(|v| v.to_ne_bytes()).collect() };
    assert!(materials::half_texels(&nan, false).is_err());
    Ok(())
}
