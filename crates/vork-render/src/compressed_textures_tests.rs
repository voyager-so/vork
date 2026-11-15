use super::*;

#[test]
fn skipped_work_preserves_gpu_bytes() -> Result<()> {
    for (w, h) in [(4, 4), (16, 8), (32, 64)] {
        for alpha in [false, true] {
            let pixels: Vec<u8> = (0..w * h)
                .flat_map(|i| {
                    let x = i % w;
                    let y = i / w;
                    [
                        (x * 79 + y * 11) as u8,
                        (x * y * 17) as u8,
                        (i * 137) as u8,
                        if alpha { (i * 53) as u8 } else { 255 },
                    ]
                })
                .collect();
            for srgb in [false, true] {
                for astc in [false, true] {
                    assert_eq!(
                        encode_basis(&pixels, w, h, srgb, astc, false)?,
                        encode_basis(&pixels, w, h, srgb, astc, true)?,
                        "{w}x{h}, alpha={alpha}, srgb={srgb}, astc={astc}"
                    );
                }
            }
        }
    }
    for color in [
        [0, 0, 0, 0],
        [255, 255, 255, 255],
        [19, 83, 207, 0],
        [19, 83, 207, 143],
    ] {
        let pixels = color.repeat(16);
        for srgb in [false, true] {
            for astc in [false, true] {
                assert_eq!(
                    encode_basis(&pixels, 4, 4, srgb, astc, false)?,
                    encode_basis(&pixels, 4, 4, srgb, astc, true)?
                );
            }
        }
    }
    Ok(())
}

fn direct_quality(im: &image::RgbaImage, srgb: bool) -> Result<serde_json::Value> {
    let start = Instant::now();
    let direct = basis_container(
        &im,
        im.width(),
        im.height(),
        srgb,
        true,
        BasisTextureFormat::AstcLdr4x4,
    )?;
    let direct_s = start.elapsed().as_secs_f64();
    let baseline = basis_container(
        &im,
        im.width(),
        im.height(),
        srgb,
        true,
        BasisTextureFormat::UastcLdr4x4,
    )?;
    let decode = |container: &[u8]| -> Result<Vec<u8>> {
        Ok(BasisuTranscoder::new(
            container,
            SupportedTextureCompression::empty(),
            ChannelType::Rgba,
        )?
        .transcode(None, Some(srgb))?
        .data)
    };
    let psnr = |decoded: &[u8], channels: usize| {
        let sum: f64 = im
            .as_raw()
            .chunks_exact(4)
            .zip(decoded.chunks_exact(4))
            .map(|(a, b)| {
                (0..channels)
                    .map(|c| (f64::from(a[c]) - f64::from(b[c])).powi(2))
                    .sum::<f64>()
            })
            .sum();
        10. * (255.0_f64.powi(2) * f64::from(im.width()) * f64::from(im.height()) * channels as f64
            / sum)
            .log10()
    };
    let baseline = decode(&baseline)?;
    let direct = decode(&direct)?;
    Ok(
        serde_json::json!({"direct_s":direct_s,"baseline_rgb_psnr":psnr(&baseline,3),"direct_rgb_psnr":psnr(&direct,3),"baseline_rgba_psnr":psnr(&baseline,4),"direct_rgba_psnr":psnr(&direct,4)}),
    )
}

#[test]
#[ignore = "parallel direct ASTC quality evaluation; set VORK_ASTC_CORPUS"]
fn direct_corpus() -> Result<()> {
    let paths: Vec<std::path::PathBuf> =
        serde_json::from_slice(&std::fs::read(std::env::var("VORK_ASTC_CORPUS")?)?)?;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let rows = std::sync::Mutex::new(Vec::new());
    let count = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(paths.len());
    std::thread::scope(|scope| -> Result<()> {
        let joins: Vec<_> = (0..count).map(|_| scope.spawn(|| -> Result<()> {
            loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(path) = paths.get(i) else { break };
                let im = image::open(path)?.to_rgba8();
                for srgb in [false, true] {
                    let row = serde_json::json!({"path":path,"srgb":srgb,"quality":direct_quality(&im, srgb)?});
                    eprintln!("{row}");
                    rows.lock().unwrap().push(row);
                }
            }
            Ok(())
        })).collect();
        for join in joins {
            join.join().unwrap()?;
        }
        Ok(())
    })?;
    if let Ok(output) = std::env::var("VORK_ASTC_RESULTS") {
        std::fs::write(
            output,
            serde_json::to_vec_pretty(&rows.into_inner().unwrap())?,
        )?;
    }
    Ok(())
}

/// VORK_ASTC_CORPUS is a JSON array of image paths. No GPU required.
#[test]
#[ignore = "corpus encode timing; set VORK_ASTC_CORPUS"]
fn corpus() -> Result<()> {
    let paths: Vec<std::path::PathBuf> =
        serde_json::from_slice(&std::fs::read(std::env::var("VORK_ASTC_CORPUS")?)?)?;
    let mut rows = Vec::new();
    for path in paths {
        let im = image::open(&path)?.to_rgba8();
        for srgb in [false, true] {
            let start = Instant::now();
            let baseline = encode_basis(&im, im.width(), im.height(), srgb, true, false)?;
            let before = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let candidate = encode_basis(&im, im.width(), im.height(), srgb, true, true)?;
            let after = start.elapsed().as_secs_f64();
            ensure!(
                baseline == candidate,
                "ASTC bytes differ for {} srgb={srgb}",
                path.display()
            );
            let quality = if std::env::var_os("VORK_ASTC_DIRECT").is_some() {
                direct_quality(&im, srgb)?
            } else {
                serde_json::Value::Null
            };
            let row = serde_json::json!({"path":path,"width":im.width(),"height":im.height(),"srgb":srgb,"baseline_s":before,"candidate_s":after,"quality":quality,"bytes":baseline.len(),"hash":blake3::hash(&candidate).to_hex().to_string()});
            eprintln!("{row}");
            rows.push(row);
        }
    }
    // Exercise the same across-texture scheduling as installation, without a GPU.
    let images: Vec<_> = rows
        .iter()
        .map(|row| {
            let path = row["path"].as_str().unwrap();
            Ok((
                image::open(path)?.to_rgba8(),
                row["srgb"].as_bool().unwrap(),
                row["hash"].as_str().unwrap().to_owned(),
            ))
        })
        .collect::<Result<_>>()?;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let count = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(images.len());
    let start = Instant::now();
    std::thread::scope(|scope| -> Result<()> {
        let joins: Vec<_> = (0..count)
            .map(|_| {
                scope.spawn(|| -> Result<()> {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((im, srgb, hash)) = images.get(i) else {
                            break;
                        };
                        let data = encode_basis(im, im.width(), im.height(), *srgb, true, true)?;
                        ensure!(
                            blake3::hash(&data).to_hex().as_str() == hash,
                            "parallel encoding changed bytes"
                        );
                    }
                    Ok(())
                })
            })
            .collect();
        for join in joins {
            join.join().unwrap()?;
        }
        Ok(())
    })?;
    eprintln!(
        "batch: {} textures, {count} workers, {:.6}s",
        images.len(),
        start.elapsed().as_secs_f64()
    );
    if let Ok(output) = std::env::var("VORK_ASTC_RESULTS") {
        std::fs::write(output, serde_json::to_vec_pretty(&rows)?)?;
    }
    Ok(())
}

/// Match actual glTF tap/color-space jobs to the serial corpus measurements.
/// VORK_ASTC_JOBS is an array of {path,srgb}; duplicate decoded inputs encode once.
#[test]
#[ignore = "scene encode timing; requires corpus results and VORK_ASTC_JOBS"]
fn scene_batch() -> Result<()> {
    #[derive(serde::Deserialize)]
    struct Input {
        path: std::path::PathBuf,
        srgb: bool,
    }
    let jobs: Vec<Input> =
        serde_json::from_slice(&std::fs::read(std::env::var("VORK_ASTC_JOBS")?)?)?;
    let rows: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(std::env::var("VORK_ASTC_CORPUS_RESULTS")?)?)?;
    let mut measured = std::collections::HashMap::new();
    for row in rows {
        let im = image::open(row["path"].as_str().unwrap())?.to_rgba8();
        measured.insert(
            (
                blake3::hash(&im),
                im.width(),
                im.height(),
                row["srgb"].as_bool().unwrap(),
            ),
            row,
        );
    }
    let mut seen = std::collections::HashSet::new();
    let mut images = Vec::new();
    let mut baseline = 0.;
    let mut serial_candidate = 0.;
    for job in jobs {
        let im = image::open(job.path)?.to_rgba8();
        let key = (blake3::hash(&im), im.width(), im.height(), job.srgb);
        if !seen.insert(key) {
            continue;
        }
        let row = measured
            .get(&key)
            .ok_or_else(|| anyhow::anyhow!("scene texture absent from corpus"))?;
        baseline += row["baseline_s"].as_f64().unwrap();
        serial_candidate += row["candidate_s"].as_f64().unwrap();
        images.push((im, job.srgb, row["hash"].as_str().unwrap().to_owned()));
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let count = std::env::var("VORK_ASTC_WORKERS")
        .ok()
        .map(|n| n.parse())
        .transpose()?
        .unwrap_or(std::thread::available_parallelism().map_or(1, usize::from))
        .min(images.len());
    ensure!(
        count > 0,
        "scene timing needs at least one worker and image"
    );
    let start = Instant::now();
    std::thread::scope(|scope| -> Result<()> {
        let joins: Vec<_> = (0..count)
            .map(|_| {
                scope.spawn(|| -> Result<()> {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((im, srgb, hash)) = images.get(i) else {
                            break;
                        };
                        let data = encode_basis(im, im.width(), im.height(), *srgb, true, true)?;
                        ensure!(
                            blake3::hash(&data).to_hex().as_str() == hash,
                            "scene bytes changed"
                        );
                    }
                    Ok(())
                })
            })
            .collect();
        for join in joins {
            join.join().unwrap()?;
        }
        Ok(())
    })?;
    let row = serde_json::json!({"textures":images.len(),"workers":count,"baseline_serial_sum_s":baseline,"candidate_serial_sum_s":serial_candidate,"candidate_parallel_s":start.elapsed().as_secs_f64()});
    eprintln!("{row}");
    if let Ok(output) = std::env::var("VORK_ASTC_RESULTS") {
        std::fs::write(output, serde_json::to_vec_pretty(&row)?)?;
    }
    Ok(())
}

/// Exercise endpoint parity with solid, correlated, and unrelated RGB values.
/// A mixed image keeps opaque blocks next to nonconstant-alpha blocks.
pub(crate) fn opacity_pixels(w: u32, h: u32, seed: u32, mixed: bool) -> Vec<u8> {
    let mut state = seed + 1;
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .flat_map(|(x, y)| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = ((x + y * 4) * 13 + seed * 29) as u8;
            let rgb = match (x / 4 + y / 4 + seed) % 3 {
                0 => [v, v.wrapping_add(16), v.wrapping_add(32)],
                1 => [state as u8, (state >> 8) as u8, (state >> 16) as u8],
                _ => [seed as u8 * 2, 180, 156],
            };
            [
                rgb[0],
                rgb[1],
                rgb[2],
                if mixed && x >= w / 2 { v } else { 255 },
            ]
        })
        .collect()
}

pub(crate) fn assert_opaque_blocks(input: &[u8], w: u32, h: u32, data: &[u8], astc: bool) -> usize {
    let mut checked = 0;
    for by in 0..h.div_ceil(4) {
        for bx in 0..w.div_ceil(4) {
            let opaque = (0..4).all(|y| {
                (0..4).all(|x| {
                    input
                        [(((by * 4 + y).min(h - 1) * w + (bx * 4 + x).min(w - 1)) * 4 + 3) as usize]
                        == 255
                })
            });
            if !opaque {
                continue;
            }
            let at = ((by * w.div_ceil(4) + bx) * 16) as usize;
            let mut decoded = [0; 16];
            if astc {
                texture2ddecoder::decode_astc_block(&data[at..at + 16], 4, 4, &mut decoded);
            } else {
                texture2ddecoder::decode_bc7_block(&data[at..at + 16], &mut decoded);
            }
            for (i, pixel) in decoded.into_iter().enumerate() {
                assert_eq!(
                    pixel >> 24,
                    255,
                    "astc={astc}, {w}x{h}, block=({bx},{by}), texel={i}, bytes={:02x?}",
                    &data[at..at + 16]
                );
            }
            checked += 1;
        }
    }
    checked
}

#[test]
fn opaque_alpha_native_codecs_and_mips() -> Result<()> {
    let mut checked = 0;
    for (w, h) in [(4u32, 4u32), (12, 28), (64, 32)] {
        for seed in 0..8 {
            for mixed in [false, true] {
                let pixels = opacity_pixels(w, h, seed, mixed);
                for srgb in [false, true] {
                    // Basis generates its own filtered mip chain. Opaque images
                    // must stay opaque throughout; mixed images check base blocks.
                    for astc in [false, true] {
                        let encoded = encode_basis(&pixels, w, h, srgb, astc, true)?;
                        checked += assert_opaque_blocks(&pixels, w, h, &encoded, astc);
                        if !mixed {
                            let (mut mw, mut mh, mut offset) = (w, h, 0);
                            loop {
                                let length = (mw.div_ceil(4) * mh.div_ceil(4) * 16) as usize;
                                checked += assert_opaque_blocks(
                                    &[0, 0, 0, 255].repeat((mw * mh) as usize),
                                    mw,
                                    mh,
                                    &encoded[offset..offset + length],
                                    astc,
                                );
                                offset += length;
                                if mw == 1 && mh == 1 {
                                    break;
                                }
                                mw = (mw / 2).max(1);
                                mh = (mh / 2).max(1);
                            }
                            assert_eq!(offset, encoded.len());
                        }
                    }
                    #[cfg(target_arch = "x86_64")]
                    {
                        let encoded = native_bc7(&pixels, w, h, srgb);
                        let mut offset = 0;
                        visit_native_mips(&pixels, w, h, srgb, |input, pw, ph| {
                            let length = (pw / 4 * (ph / 4) * 16) as usize;
                            checked += assert_opaque_blocks(
                                input,
                                pw,
                                ph,
                                &encoded[offset..offset + length],
                                false,
                            );
                            offset += length;
                        });
                        assert_eq!(offset, encoded.len());
                    }
                }
            }
        }
    }
    assert!(checked > 1000);
    eprintln!(
        "exact alpha 255: {checked} opaque blocks, native BC7 and Basis BC7/ASTC, linear/sRGB and mips"
    );
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn opaque_alpha_gpu_codecs_and_mips() -> Result<()> {
    let r = crate::test_renderer()?;
    let bc = r
        .device
        .features()
        .contains(wgpu::Features::TEXTURE_COMPRESSION_BC);
    if !bc {
        return Ok(());
    }
    let ctx = crate::gpu_bc7::Context::new(&r)?;
    let mut scratch = crate::gpu_bc7::Scratch::default();
    let mut checked = 0;
    for (w, h) in [(4u32, 4u32), (12, 28), (64, 32)] {
        for seed in 0..8 {
            for mixed in [false, true] {
                let pixels = opacity_pixels(w, h, seed, mixed);
                for srgb in [false, true] {
                    let levels = 32 - w.max(h).leading_zeros();
                    let bytes = (0..levels)
                        .map(|m| (w >> m).max(1).div_ceil(4) * (h >> m).max(1).div_ceil(4) * 16)
                        .sum::<u32>() as usize;
                    let format = if srgb {
                        wgpu::TextureFormat::Bc7RgbaUnormSrgb
                    } else {
                        wgpu::TextureFormat::Bc7RgbaUnorm
                    };
                    let tex = crate::gpu_bc7::texture(&r.device, w, h, levels, format);
                    let (_, encoded) = scratch
                        .submit(&ctx, tex, pixels.clone(), w, h, format, bytes, true)?
                        .finish(&ctx, &mut scratch)?;
                    checked += assert_opaque_blocks(&pixels, w, h, &encoded, false);
                    // Power-of-two mips use the same texel-centre filter on CPU/GPU.
                    #[cfg(target_arch = "x86_64")]
                    if !mixed || w.is_power_of_two() && h.is_power_of_two() {
                        let mut offset = 0;
                        visit_native_mips(&pixels, w, h, srgb, |input, pw, ph| {
                            let length = (pw / 4 * (ph / 4) * 16) as usize;
                            checked += assert_opaque_blocks(
                                input,
                                pw,
                                ph,
                                &encoded[offset..offset + length],
                                false,
                            );
                            offset += length;
                        });
                        assert_eq!(offset, encoded.len());
                    }
                }
            }
        }
    }
    assert!(checked > 1000);
    eprintln!(
        "exact alpha 255: {checked} opaque blocks, GPU BC7, linear/sRGB and mips; adapter={:?}",
        r.adapter_info
    );
    Ok(())
}

#[test]
#[ignore = "file reproduction; set VORK_TEXTURE_OPACITY_IMAGE and VORK_TEXTURE_OPACITY_OUT"]
fn opaque_file_encoders() -> Result<()> {
    let image = image::open(std::env::var("VORK_TEXTURE_OPACITY_IMAGE")?)?.to_rgba8();
    ensure!(image.pixels().all(|p| p[3] == 255), "source must be opaque");
    let out = std::path::PathBuf::from(std::env::var("VORK_TEXTURE_OPACITY_OUT")?);
    std::fs::create_dir_all(&out)?;
    let (w, h) = image.dimensions();
    for srgb in [false, true] {
        let mut encodings = vec![
            (
                "basis-bc7",
                false,
                encode_basis(&image, w, h, srgb, false, true)?,
            ),
            (
                "basis-astc",
                true,
                encode_basis(&image, w, h, srgb, true, true)?,
            ),
        ];
        #[cfg(target_arch = "x86_64")]
        encodings.push(("native-bc7", false, native_bc7(&image, w, h, srgb)));
        for (name, astc, encoded) in encodings {
            let (mut mw, mut mh, mut offset) = (w, h, 0);
            loop {
                let length = (mw.div_ceil(4) * mh.div_ceil(4) * 16) as usize;
                assert_opaque_blocks(
                    &[0, 0, 0, 255].repeat((mw * mh) as usize),
                    mw,
                    mh,
                    &encoded[offset..offset + length],
                    astc,
                );
                offset += length;
                if mw == 1 && mh == 1 {
                    break;
                }
                mw = (mw / 2).max(1);
                mh = (mh / 2).max(1);
            }
            assert_eq!(offset, encoded.len());
            std::fs::write(out.join(format!("{name}-srgb-{srgb}.blocks")), &encoded)?;
            eprintln!(
                "{name} srgb={srgb}: exact alpha 255 in every mip; {} bytes",
                encoded.len()
            );
        }
    }
    Ok(())
}
