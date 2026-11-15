use super::*;

fn read_lookup(r: &Renderer, texture: &wgpu::Texture) -> Result<Vec<[f32; 4]>> {
    let stride = (texture.width() * 8).div_ceil(256) * 256;
    let read = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BRDF lookup regression readback"),
        size: u64::from(stride) * u64::from(texture.height()),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = r.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(texture.as_image_copy(), wgpu::TexelCopyBufferInfo {
        buffer: &read,
        layout: wgpu::TexelCopyBufferLayout {
            offset: 0, bytes_per_row: Some(stride), rows_per_image: Some(texture.height()),
        },
    }, texture.size());
    let submission = r.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    read.map_async(wgpu::MapMode::Read, .., move |result| { let _ = tx.send(result); });
    r.wait(Some(submission))?;
    rx.recv()??;
    let mapped = read.get_mapped_range(..);
    Ok(mapped.chunks_exact(stride as usize).flat_map(|row| {
        row[..texture.width() as usize * 8].chunks_exact(8).map(|pixel| {
            std::array::from_fn(|channel| {
                half::f16::from_bits(u16::from_le_bytes(pixel[channel * 2..channel * 2 + 2].try_into().unwrap())).to_f32()
            })
        })
    }).collect())
}

#[test]
#[ignore = "requires real GPU and pinned Slang"]
fn startup_brdf_lookup_preserves_moments_and_dielectric_atlas() -> Result<()> {
    let r = Renderer::new()?;
    let scene = vork_scene::v2::from_test_value(serde_json::json!({
        "version": 2, "nodes": [{"type": "camera"}], "render": {"resolution": [16, 16]},
    }))?;
    let installed = r.install(scene, Path::new("."))?;
    let lookup = installed.environment.brdf.texture();
    let actual = read_lookup(&r, lookup)?;
    let width = lookup.width() as usize;

    // An untiled prefix spans two eight-row boundaries and a partial final
    // workgroup. It uses zero offset, independently of the startup row schedule.
    let reference = r.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("untiled BRDF prefix"),
        size: wgpu::Extent3d { width: lookup.width(), height: 17, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let parameters = filter_parameters(0, 2, 1., [glam::DVec3::ZERO; 4], [1, 1]);
    let params = init(&r.device, "untiled BRDF parameters", bytemuck::cast_slice(&parameters), wgpu::BufferUsages::UNIFORM);
    let cdf = init(&r.device, "unused BRDF proposal", bytemuck::cast_slice(&[1f32, 1.]), wgpu::BufferUsages::STORAGE);
    let view = reference.create_view(&Default::default());
    let pipeline = Environment::pipeline(&r);
    let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("untiled BRDF reference"), layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            entry(0, wgpu::BindingResource::TextureView(&installed.environment.source)),
            entry(1, wgpu::BindingResource::TextureView(&view)),
            entry(2, params.as_entire_binding()), entry(3, cdf.as_entire_binding()),
        ],
    });
    let mut encoder = r.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(lookup.width().div_ceil(8), 3, 1);
    }
    r.queue.submit([encoder.finish()]);
    let expected = read_lookup(&r, &reference)?;
    assert_eq!(&actual[..expected.len()], expected.as_slice(), "bounded startup changed BRDF moments across row boundaries");

    // Every valid dielectric texel writes alpha=1, including the zero-energy
    // eta=1 slice. Check every row to catch skipped or overwritten tiles.
    let n = vork_slang::registry::BRDF_ALBEDO_SIZE as usize;
    let first = vork_slang::registry::BRDF_SIZE as usize;
    for y in first..lookup.height() as usize {
        for x in 0..width {
            let pixel = actual[y * width + x];
            let valid = ((y - first) / n) * 2 + x / n <= vork_slang::registry::BRDF_IOR_STEPS as usize;
            assert!(pixel.iter().all(|v| v.is_finite()), "nonfinite BRDF texel ({x}, {y}): {pixel:?}");
            assert_eq!(pixel[3], if valid { 1. } else { 0. }, "unwritten dielectric texel ({x}, {y})");
        }
    }
    // Independent physical references: smooth normal-incidence Fresnel, equal
    // IOR (no interface), and unit-Fresnel GGX at alpha=1 (1 - ln(2)).
    for (x, y, expected) in [
        (63, 128, 1.), (127, 128, 0.9384765625), (63, 192, 0.87890625),
        (127, 1088, 0.0009765625), (63, 1152, 0.), (127, 2112, 0.9384765625),
        (63, 2176, 1.), (63, 2239, 1. - std::f32::consts::LN_2),
    ] {
        let pixel = actual[y * width + x];
        assert!((pixel[0] - expected).abs() < 0.001, "dielectric albedo ({x}, {y}): {pixel:?}, expected {expected}");
        assert_eq!([pixel[1], pixel[2]], [0., 0.]);
    }
    Ok(())
}
