use vork_encode::{Backend, DeviceInfo, EncodeSettings, Encoder};

#[test]
fn decimal_movie_rates_round_trip() -> anyhow::Result<()> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default()))?;
    let root = std::env::var_os("VORK_DECIMAL_RATE_OUTPUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(format!("vork-decimal-rates-{}", std::process::id()));
    std::fs::create_dir_all(&root)?;
    for (rate, expected) in [
        ("29.97", "2997/100"),
        ("23.976", "2997/125"),
        ("24000/1001", "24000/1001"),
        ("30000/1001", "30000/1001"),
        ("96/17", "96/17"),
        ("1/1001000", "1/1001000"),
        ("1001000", "1001000/1"),
    ] {
        for backend in [Backend::Readback, Backend::Auto] {
            for extension in ["mp4", "mov"] {
                let path = root.join(format!(
                    "{}-{backend:?}.{extension}",
                    rate.replace('/', "_")
                ));
                let mut encoder = Encoder::open(
                    DeviceInfo {
                        device: device.clone(),
                        queue: queue.clone(),
                        backend,
                    },
                    64,
                    32,
                    rate.parse()?,
                    EncodeSettings::default(),
                    &path,
                    32,
                    1,
                )?;
                let mut surface = encoder.create_surface()?;
                for index in 0..32 {
                    if index > 0 {
                        encoder.release(index - 1, &mut surface)?;
                    }
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: None,
                        size: wgpu::Extent3d {
                            width: 64,
                            height: 32,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Bgra8Unorm,
                        usage: wgpu::TextureUsages::COPY_SRC
                            | wgpu::TextureUsages::COPY_DST
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    queue.write_texture(
                        texture.as_image_copy(),
                        &[40, 80, 200, 255].repeat(64 * 32),
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(256),
                            rows_per_image: None,
                        },
                        texture.size(),
                    );
                    let mut commands = device.create_command_encoder(&Default::default());
                    surface.record_copy(&mut commands, &texture);
                    let submission = queue.submit([commands.finish()]);
                    encoder.push(index, texture, &mut surface, &submission)?;
                }
                encoder.finish()?;
                let probe = std::process::Command::new("ffprobe")
                .args([
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=avg_frame_rate,r_frame_rate,time_base,nb_frames:packet=pts,dts,duration",
                    "-of",
                    "default=noprint_wrappers=1",
                ])
                .arg(&path)
                .output()?;
                anyhow::ensure!(probe.status.success(), "ffprobe: {:?}", probe);
                let output = String::from_utf8(probe.stdout)?;
                println!("{path:?}: {output}");
                assert!(
                    output
                        .lines()
                        .any(|v| v == format!("avg_frame_rate={expected}"))
                );
                assert!(
                    output
                        .lines()
                        .any(|v| v == format!("r_frame_rate={expected}"))
                );
                assert!(output.lines().any(|v| v == "nb_frames=32"));
                let fps: vork_encode::Time = rate.parse()?;
                assert!(
                    output
                        .lines()
                        .any(|v| v == format!("time_base=1/{}", fps.numerator()))
                );
                for (field, offset) in [("pts", 0), ("dts", 0), ("duration", 1)] {
                    let values: Vec<i64> = output
                        .lines()
                        .filter_map(|line| line.strip_prefix(&format!("{field}=")))
                        .map(|value| value.parse().unwrap())
                        .collect();
                    assert_eq!(values.len(), 32);
                    for (index, value) in values.into_iter().enumerate() {
                        let ticks = if offset == 1 { 1 } else { index as i64 };
                        assert_eq!(
                            value,
                            ticks * fps.denominator(),
                            "{rate} {extension} {field} frame {index}"
                        );
                    }
                }
            }
        }
    }
    // BB034's original decimal is exact but exceeds the native signed-32-bit clock.
    // Refuse before creating or truncating an output instead of silently rounding it.
    let path = root.join("unrepresentable.mp4");
    std::fs::write(&path, b"preserve existing output")?;
    let error = Encoder::prepare(
        Backend::Readback,
        64,
        32,
        "5.64705898192515".parse()?,
        EncodeSettings::default(),
        &path,
    )
    .err()
    .expect("unrepresentable clock must refuse");
    println!("BB034 original value refuses: {error:#}");
    assert_eq!(
        error.to_string(),
        "MP4 cannot represent fps 112941179638503/20000000000000; use fps \"1113587715/197197819\""
    );
    assert_eq!(std::fs::read(&path)?, b"preserve existing output");
    if std::env::var_os("VORK_DECIMAL_RATE_OUTPUT").is_none() {
        std::fs::remove_dir_all(root)?;
    }
    Ok(())
}
