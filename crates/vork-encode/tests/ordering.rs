use vork_encode::{Backend, DeviceInfo, EncodeSettings, Encoder};
#[test]
fn rejects_bad_sequences_and_finalizes_ordered_ring_entries() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let root = std::env::temp_dir().join(format!("vork-encode-test-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let texture = || {
        let t = device.create_texture(&wgpu::TextureDescriptor {
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
            t.as_image_copy(),
            &[40u8, 40, 200, 128].repeat(64 * 32),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
            t.size(),
        );
        t
    };
    let copy = |surface: &vork_encode::Surface, texture: &wgpu::Texture| {
        let mut commands = device.create_command_encoder(&Default::default());
        surface.record_copy(&mut commands, texture);
        queue.submit([commands.finish()])
    };
    let idle = queue.submit([]);
    for (backend, extension) in [
        (Backend::Readback, "mp4"),
        (Backend::Auto, "mp4"),
        (Backend::Readback, "mov"),
        (Backend::Auto, "mov"),
    ] {
        let gpu = || DeviceInfo {
            device: device.clone(),
            queue: queue.clone(),
            backend,
        };
        assert!(
            Encoder::open(
                gpu(),
                63,
                32,
                vork_encode::Time::integer(24),
                EncodeSettings::default(),
                root.join("odd.mp4"),
                1,
                3
            )
            .is_err()
        );
        let limits_path = root.join("limits.mp4");
        std::fs::write(&limits_path, b"keep existing output").unwrap();
        let prepared = Encoder::prepare(
            backend,
            device.limits().max_texture_dimension_2d + 2,
            32,
            vork_encode::Time::integer(24),
            EncodeSettings::default(),
            &limits_path,
        )
        .unwrap();
        assert_eq!(
            std::fs::read(&limits_path).unwrap(),
            b"keep existing output"
        );
        assert!(prepared.open(device.clone(), queue.clone(), 1, 3).is_err());
        assert_eq!(
            std::fs::read(&limits_path).unwrap(),
            b"keep existing output"
        );
        let mut empty = Encoder::open(
            gpu(),
            64,
            32,
            vork_encode::Time::integer(24),
            EncodeSettings::default(),
            root.join("empty.mp4"),
            1,
            3,
        )
        .unwrap();
        assert!(empty.finish().is_err());
        let mut missing = Encoder::open(
            gpu(),
            64,
            32,
            vork_encode::Time::integer(24),
            EncodeSettings::default(),
            root.join("missing.mp4"),
            1,
            3,
        )
        .unwrap();
        let mut missing_surface = missing.create_surface().unwrap();
        assert!(missing.push(1, texture(), &mut missing_surface, &idle).is_err());
        assert!(missing.finish().is_err());
        let source = texture();
        let submission = copy(&missing_surface, &source);
        missing.push(0, source, &mut missing_surface, &submission).unwrap();
        missing.finish().unwrap();
        let full_path = root.join(format!("full.{extension}"));
        let prepared = std::thread::spawn(move || {
            Encoder::prepare(backend, 64, 32, vork_encode::Time::integer(24), EncodeSettings::default(), full_path)
        })
        .join()
        .unwrap()
        .unwrap();
        let mut full = prepared.open(device.clone(), queue.clone(), 33, 3).unwrap();
        #[cfg(target_os = "macos")]
        if extension == "mov"
            && matches!(backend, Backend::Auto)
            && std::env::var_os("VORK_REQUIRE_VT_PRORES").is_some()
        {
            assert!(
                !full.uses_pixel_readback(),
                "Auto MOV must use VideoToolbox"
            );
        }
        let mut surfaces: Vec<_> = (0..3).map(|_| full.create_surface().unwrap()).collect();
        assert!(full.push(1, texture(), &mut surfaces[1], &idle).is_err());
        for i in 0..33 {
            let surface = &mut surfaces[i % 3];
            if i >= 3 {
                full.release(i as u64 - 3, surface).unwrap();
            }
            let source = texture();
            // Distinct frames expose stale or prematurely reused pipe payloads
            // when the three-entry staging ring wraps.
            queue.write_texture(
                source.as_image_copy(),
                &[40u8, 40, 200 - i as u8 * 4, 128].repeat(64 * 32),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
                source.size(),
            );
            let submission = copy(surface, &source);
            full.push(i as u64, source, surface, &submission).unwrap();
        }
        assert!(full.push(0, texture(), &mut surfaces[0], &idle).is_err());
        let path = full.finish().unwrap();
        #[cfg(unix)]
        if matches!(backend, Backend::Readback) {
            use std::os::unix::fs::PermissionsExt;
            let ordinary = root.join("ordinary-output");
            std::fs::write(&ordinary, b"normal file creation under this umask").unwrap();
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                std::fs::metadata(ordinary).unwrap().permissions().mode() & 0o777,
                "published movies must retain ordinary output permissions");
        }
        assert!(full.finish().is_err());
        assert!(full.push(33, texture(), &mut surfaces[0], &idle).is_err());
        assert!(full.create_surface().is_err());
        let output = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "33");
        let tags = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "stream=color_space,color_range,color_primaries,color_transfer",
                "-of",
                "default=noprint_wrappers=1",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(tags.status.success());
        let tags = String::from_utf8(tags.stdout).unwrap();
        for tag in [
            "color_space=bt709",
            "color_primaries=bt709",
            "color_transfer=bt709",
        ] {
            assert!(
                tags.lines().any(|line| line == tag),
                "missing {tag}: {tags}"
            );
        }
        // QuickTime nclc has no range bit. Validate MOV's video-range payload
        // by forcing BT.709/TV on decode instead of trusting decoder defaults.
        if extension == "mp4" {
            assert!(tags.lines().any(|line| line == "color_range=tv"), "{tags}");
        }
        let pixels = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(root.join(format!("full.{extension}")))
            .args([
                "-frames:v",
                "33",
                "-vf",
                "scale=in_color_matrix=bt709:in_range=tv:out_range=pc",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "-",
            ])
            .output()
            .unwrap();
        assert!(pixels.status.success());
        assert_eq!(pixels.stdout.len(), 33 * 64 * 32 * 4);
        for (i, frame) in pixels.stdout.chunks_exact(64 * 32 * 4).enumerate() {
            for (actual, expected) in frame[..3].iter().zip([200i16 - i as i16 * 4, 40, 40]) {
                assert!(
                    (*actual as i16 - expected).abs() < 6,
                    "frame {i}: ordered BGRA channel conversion"
                );
            }
        }
        if extension == "mov" {
            assert!(
                (pixels.stdout[3] as i16 - 128).abs() <= 1,
                "ProRes preserves alpha"
            );
        }
    }
    let gpu = || DeviceInfo {
        device: device.clone(),
        queue: queue.clone(),
        backend: Backend::Readback,
    };
    let bad_path = root.join("missing-parent/failed.mov");
    // Staging validates the destination directory before streaming any frames.
    let error = match Encoder::open(gpu(), 64, 32, vork_encode::Time::integer(24), EncodeSettings::default(), &bad_path, 1, 3) {
        Err(error) => error,
        Ok(_) => panic!("missing output directory must refuse before streaming"),
    };
    let diagnosis = format!("{error:#}");
    assert!(diagnosis.contains("prepare video output"), "{diagnosis}");
    assert!(diagnosis.contains("failed.mov"), "{diagnosis}");
    assert!(!bad_path.exists());
    let hardware = DeviceInfo {
        device: device.clone(),
        queue: queue.clone(),
        backend: Backend::HardwareReadback,
    };
    assert!(
        Encoder::open(
            hardware,
            64,
            32,
            vork_encode::Time::integer(24),
            EncodeSettings::default(),
            root.join("hardware.mov"),
            1,
            3
        )
        .is_err()
    );
    // Linux hardware H.264 follows the rendering adapter, and only NVIDIA has one: any other
    // adapter's MP4 is software at its quality target, so explicit hardware readback refuses.
    #[cfg(target_os = "linux")]
    if adapter.get_info().vendor != 0x10de {
        let error = Encoder::open(
            DeviceInfo { device: device.clone(), queue: queue.clone(), backend: Backend::HardwareReadback },
            64,
            32,
            vork_encode::Time::integer(24),
            EncodeSettings::default(),
            root.join("hardware.mp4"),
            1,
            3,
        )
        .err()
        .expect("hardware readback needs an NVIDIA rendering adapter");
        assert!(format!("{error:#}").contains("needs an NVIDIA rendering adapter"), "{error:#}");
        assert!(!root.join("hardware.mp4").exists());
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn dropping_preparation_preserves_existing_output() {
    let path =
        std::env::temp_dir().join(format!("vork-prepare-cancel-{}.mp4", std::process::id()));
    std::fs::write(&path, b"keep existing output").unwrap();
    let worker_path = path.clone();
    let prepared = std::thread::spawn(move || {
        Encoder::prepare(
            Backend::Auto,
            64,
            32,
            vork_encode::Time::integer(24),
            EncodeSettings::default(),
            worker_path,
        )
    })
    .join()
    .unwrap()
    .unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"keep existing output");
    drop(prepared);
    assert_eq!(std::fs::read(&path).unwrap(), b"keep existing output");
    std::fs::remove_file(path).unwrap();
}
