//! HDR preprocessing belongs to authoring, not scene installation.
use anyhow::{Context, Result, ensure};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::path::Path;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sun {
    pub rotation: [f32; 4],
    pub color: [f32; 3],
    pub intensity: f32,
    pub angle: f32,
}
const ENV_CORE_BACKGROUND_MIN_ANGLE: f64 = 5.0f64.to_radians();
const ENV_CORE_BACKGROUND_MIN_TEXELS: f64 = 4.;
const ENV_CORE_MAX_ANGLE: f64 = 40.0f64.to_radians();
const ENV_CORE_BACKGROUND_SAMPLES: u32 = 32;
const ENV_CORE_BACKGROUND_CONTRAST: f32 = 4.;
const ENV_CORE_PEAK_FRACTION: f32 = 0.01;

pub fn split(image: &image::Rgb32FImage, threshold: f32) -> (image::Rgb32FImage, Option<Sun>) {
    let residual = core_residual(image, threshold);
    let sun = measure(image, &residual);
    if sun.is_some() {
        (residual, sun)
    } else {
        (image.clone(), None)
    }
}
fn measure(image: &image::Rgb32FImage, residual_image: &image::Rgb32FImage) -> Option<Sun> {
    let mut sun = None;
    let mut energy = glam::DVec3::ZERO;
    let mut moment = glam::DVec3::ZERO;
    let mut extent = 0f64;
    let width = image.width();
    let height = image.height();

    let longitude: Vec<_> = (0..width)
        .map(|x| {
            let phi = std::f64::consts::TAU * ((x as f64 + 0.5) / width as f64 - 0.5);
            (phi.cos(), phi.sin())
        })
        .collect();
    for y in 0..height {
        let omega = std::f64::consts::TAU / width as f64
            * ((std::f64::consts::PI * y as f64 / height as f64).cos()
                - (std::f64::consts::PI * (y + 1) as f64 / height as f64).cos());
        let theta = std::f64::consts::PI * (y as f64 + 0.5) / height as f64;
        let (sin_theta, cos_theta) = (theta.sin(), theta.cos());
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            let radiance = glam::DVec3::new(pixel[0] as f64, pixel[1] as f64, pixel[2] as f64);
            let p = residual_image.get_pixel(x, y);
            let excess = radiance - glam::DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64);
            let (cos_phi, sin_phi) = longitude[x as usize];
            let direction = glam::DVec3::new(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
            if excess.max_element() > 0. {
                extent += omega;
            }
            energy += excess * omega;
            moment += direction * excess.length() * omega;
        }
    }
    if energy.max_element() > 0. && moment.length_squared() > 1e-20 {
        let angle = (2. * (1. - extent / std::f64::consts::TAU).clamp(-1., 1.).acos()) as f32;
        let power = energy.max_element();
        // Preserve the measured RGB integral when represented by a uniform cap.
        // The sun evaluator expects normal irradiance, not cap-integrated radiance.
        let intensity = power * (angle as f64 * 0.25).cos().powi(2);
        sun = Some(Sun {
            rotation: glam::Quat::from_rotation_arc(Vec3::NEG_Z, (-moment.normalize()).as_vec3())
                .to_array(),
            color: (energy / power).as_vec3().to_array(),
            intensity: intensity as f32,
            // Match the measured support solid angle with a circular emitter:
            // omega = 2*pi*(1-cos(diameter/2)). This measures source extent,
            // independently of its radiance or extracted energy.
            angle,
        });
    }
    sun
}

fn core_residual(image: &image::Rgb32FImage, threshold: f32) -> image::Rgb32FImage {
    let mut residual = image.clone();
    let (w, h) = image.dimensions();
    let peak = |x, y| image.get_pixel(x, y).0.into_iter().fold(0f32, f32::max);
    let Some((x, y, p)) = image.enumerate_pixels().max_by(|a, b| {
        a.2.0
            .into_iter()
            .fold(0f32, f32::max)
            .total_cmp(&b.2.0.into_iter().fold(0f32, f32::max))
    }) else {
        return residual;
    };
    let maximum = p.0.into_iter().fold(0f32, f32::max);
    if maximum <= threshold {
        return residual;
    }
    let theta = std::f64::consts::PI * (y as f64 + 0.5) / h as f64;
    let phi = std::f64::consts::TAU * ((x as f64 + 0.5) / w as f64 - 0.5);
    let axis = glam::DVec3::new(
        theta.sin() * phi.cos(),
        theta.cos(),
        theta.sin() * phi.sin(),
    );
    let tangent = axis.any_orthonormal_vector();
    let bitangent = axis.cross(tangent);
    let mut radius = ENV_CORE_BACKGROUND_MIN_ANGLE
        .max(ENV_CORE_BACKGROUND_MIN_TEXELS * std::f64::consts::PI / h as f64);
    let mut ring = Vec::new();
    let background = loop {
        ring.clear();
        for k in 0..ENV_CORE_BACKGROUND_SAMPLES {
            let a = std::f64::consts::TAU * k as f64 / ENV_CORE_BACKGROUND_SAMPLES as f64;
            let d = axis * radius.cos() + (tangent * a.cos() + bitangent * a.sin()) * radius.sin();
            let u = (d.z.atan2(d.x) / std::f64::consts::TAU + 0.5).rem_euclid(1.);
            let v = d.y.clamp(-1., 1.).acos() / std::f64::consts::PI;
            ring.push(peak(
                (u * w as f64) as u32,
                ((v * h as f64) as u32).min(h - 1),
            ));
        }
        ring.sort_by(f32::total_cmp);
        let background = ring[ring.len() / 2];
        if maximum > background * ENV_CORE_BACKGROUND_CONTRAST {
            break background;
        }
        if radius >= ENV_CORE_MAX_ANGLE {
            return residual;
        }
        radius = (radius * 2.).min(ENV_CORE_MAX_ANGLE);
    };
    let floor = threshold.max(background);
    let cutoff = floor.max(maximum * ENV_CORE_PEAK_FRACTION);
    let mut seen = vec![false; (w * h) as usize];
    let mut stack = vec![(x, y)];
    while let Some((x, y)) = stack.pop() {
        let at = (y * w + x) as usize;
        if seen[at] {
            continue;
        }
        seen[at] = true;
        let value = peak(x, y);
        if value <= cutoff {
            continue;
        }
        // Reject broad components instead of clipping their support to the
        // background-estimation annulus. A flat core may select an edge texel.
        let t = std::f64::consts::PI * (y as f64 + 0.5) / h as f64;
        let p = std::f64::consts::TAU * ((x as f64 + 0.5) / w as f64 - 0.5);
        if glam::DVec3::new(t.sin() * p.cos(), t.cos(), t.sin() * p.sin()).dot(axis)
            < ENV_CORE_MAX_ANGLE.cos()
        {
            return image.clone();
        }
        residual.get_pixel_mut(x, y).0 =
            image.get_pixel(x, y).0.map(|v| v * (floor / value).min(1.));
        stack.push(((x + w - 1) % w, y));
        stack.push(((x + 1) % w, y));
        if y > 0 {
            stack.push((x, y - 1));
        }
        if y + 1 < h {
            stack.push((x, y + 1));
        }
    }
    residual
}

/// An authoring operation: writes a residual EXR and returns ordinary scene
/// declarations. The caller supplies the application's single persistent store.
pub fn prepare(
    argument: &str,
    mut cached: impl FnMut(blake3::Hash) -> Option<Vec<u8>>,
    mut publish: impl FnMut(blake3::Hash, &[u8]),
) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Options {
        file: String,
        output: String,
        #[serde(default = "name")]
        name: String,
        #[serde(default)]
        color_space: Option<String>,
        #[serde(default)]
        fps: Option<vork_time::Time>,
        #[serde(default = "threshold")]
        threshold: f32,
    }
    fn name() -> String {
        "environment".into()
    }
    fn threshold() -> f32 {
        10.
    }
    let o: Options = serde_json::from_str(argument)?;
    ensure!(
        o.fps.is_none_or(|fps| fps > vork_time::Time::ZERO),
        "environment fps must be positive"
    );
    ensure!(
        !o.name.is_empty() && o.threshold.is_finite() && o.threshold > 0.,
        "environment requires a name and positive finite threshold"
    );
    ensure!(
        Path::new(&o.output)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("exr")),
        "environment output must be an EXR path"
    );
    ensure!(
        std::path::absolute(&o.file)? != std::path::absolute(&o.output)?,
        "environment output must differ from source"
    );
    let bytes = std::fs::read(&o.file)?;
    let srgb = match o.color_space.as_deref() {
        Some("sRGB") => true,
        Some("Linear Rec.709" | "Non-Color") => false,
        None => !Path::new(&o.file)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("hdr") || e.eq_ignore_ascii_case("exr")),
        _ => anyhow::bail!("unknown color_space"),
    };
    let decode_image = || -> Result<image::Rgb32FImage> {
        let mut image =
            image::load_from_memory_with_format(&bytes, image::ImageFormat::from_path(&o.file)?)?
                .into_rgb32f();
        if srgb {
            for p in image.pixels_mut() {
                for v in &mut p.0 {
                    *v = if *v <= 0.04045 {
                        *v / 12.92
                    } else {
                        ((*v + 0.055) / 1.055).powf(2.4)
                    };
                }
            }
        }
        Ok(image)
    };
    let mut h = blake3::Hasher::new();
    h.update(&bytes);
    h.update(&o.threshold.to_le_bytes());
    h.update(&[srgb as u8]);
    h.update(include_bytes!("environment.rs"));
    let key = h.finalize();
    let decode = |data: &[u8]| -> Option<(image::Rgb32FImage, Option<Sun>)> {
        let w = u32::from_le_bytes(data.get(..4)?.try_into().ok()?);
        let height = u32::from_le_bytes(data.get(4..8)?.try_into().ok()?);
        let n = u32::from_le_bytes(data.get(8..12)?.try_into().ok()?) as usize;
        let sun = serde_json::from_slice(data.get(12..12 + n)?).ok()?;
        let pixels = data.get(12 + n..)?;
        if pixels.len() != w as usize * height as usize * 12 {
            return None;
        }
        let pixels = pixels
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        Some((image::Rgb32FImage::from_raw(w, height, pixels)?, sun))
    };
    let (residual, sun) = if let Some(value) = cached(key).and_then(|d| decode(&d)) {
        value
    } else {
        let image = decode_image()?;
        ensure!(
            image
                .pixels()
                .all(|p| p.0.iter().all(|v| v.is_finite() && *v >= 0.)),
            "environment radiance must be finite and nonnegative"
        );
        let value = split(&image, o.threshold);
        let metadata = serde_json::to_vec(&value.1)?;
        let mut data = Vec::new();
        data.extend(value.0.width().to_le_bytes());
        data.extend(value.0.height().to_le_bytes());
        data.extend((metadata.len() as u32).to_le_bytes());
        data.extend(metadata);
        for v in value.0.as_raw() {
            data.extend(v.to_le_bytes());
        }
        publish(key, &data);
        value
    };
    let full = decode_image()?;
    let mut weighted = image::Rgba32FImage::new(full.width(), full.height());
    for ((dst, full), residual) in weighted
        .pixels_mut()
        .zip(full.pixels())
        .zip(residual.pixels())
    {
        let peak = full.0.into_iter().fold(0f32, f32::max);
        let left = residual.0.into_iter().fold(0f32, f32::max);
        let weight = if peak > 0. { (left / peak).min(1.) } else { 1. };
        *dst = image::Rgba([weight, weight, weight, 1.]);
    }
    {
        use exr::prelude::{Encoding, Image, Layer, SpecificChannels, Vec2, WritableImage};
        let channels = SpecificChannels::rgba(|Vec2(x, y): Vec2<usize>| {
            let p = weighted.get_pixel(x as u32, y as u32);
            (p[0], p[1], p[2], p[3])
        });
        let attributes = sequence_attributes();
        Image::from_layer(Layer::new(
            (weighted.width() as usize, weighted.height() as usize),
            attributes,
            Encoding::SMALL_LOSSLESS,
            channels,
        ))
        .write()
        .to_file(&o.output)
        .context("write weighted environment EXR")?;
    }
    let file = std::path::absolute(&o.output)?;
    let image_name = format!("{}_image", o.name);
    let weight_name = format!("{}_lighting_weight", o.name);
    let material = format!("{}_radiance", o.name);
    let mut nodes = vec![
        serde_json::json!({"type":"light","name":o.name,"shape":"environment","material":material}),
    ];
    if let Some(s) = sun {
        nodes.push(serde_json::json!({"type":"light","name":format!("{}_sun",o.name),"shape":"sun","visible_camera":false,"rotation":s.rotation,"color":s.color,"energy":s.intensity,"angle":s.angle}));
    }
    Ok(serde_json::to_string(&serde_json::json!({"images":{image_name.clone():{"file":std::path::absolute(&o.file)?,"color_space":if srgb {"sRGB"} else {"Linear Rec.709"},"wrap_t":"clamp_to_edge"},weight_name.clone():{"file":file,"color_space":"Non-Color","wrap_t":"clamp_to_edge"}},"materials":{material.clone():{"program":"std/environment_image","inputs":{"image":{"image":image_name},"lighting_weight":{"image":weight_name}}}},"nodes":nodes}))?)
}

// Generated EXR sequences use an index clock (one frame per second), recorded
// in the standard EXR header. STEP tracks map source presentation times to it;
// playback therefore does not depend on the destination scene's frame rate.
fn sequence_attributes() -> exr::prelude::LayerAttributes {
    exr::prelude::LayerAttributes {
        frames_per_second: Some((1, 1)),
        ..Default::default()
    }
}

// Follow symlinks and identify hard links without retaining one open file per
// frame. Only missing paths lack an identity; other errors must stop the write.
fn file_identity(path: &Path) -> std::io::Result<Option<(u64, u128)>> {
    let identity: std::io::Result<_> = (|| {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let metadata = std::fs::metadata(path)?;
            Ok((metadata.dev(), metadata.ino() as u128))
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::Storage::FileSystem::{
                FILE_ID_INFO, FileIdInfo, GetFileInformationByHandleEx,
            };
            let file = std::fs::File::open(path)?;
            let mut info = FILE_ID_INFO::default();
            // SAFETY: the live file owns the handle and info is a writable,
            // correctly sized FILE_ID_INFO for the requested information class.
            let ok = unsafe {
                GetFileInformationByHandleEx(
                    file.as_raw_handle(),
                    FileIdInfo,
                    (&mut info as *mut FILE_ID_INFO).cast(),
                    std::mem::size_of::<FILE_ID_INFO>() as u32,
                )
            };
            if ok == 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok((
                info.VolumeSerialNumber,
                u128::from_le_bytes(info.FileId.Identifier),
            ))
        }
    })();
    match identity {
        Ok(identity) => Ok(Some(identity)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// The host supplies the opened source files and streams decoded frames; std owns splitting.
pub fn prepare_frames(
    argument: &str,
    inputs: &[std::path::PathBuf],
    times: &[vork_time::Clock],
    mut frame: impl FnMut(&vork_time::Clock) -> Result<image::DynamicImage>,
    mut cached: impl FnMut(blake3::Hash) -> Option<Vec<u8>>,
    mut publish: impl FnMut(blake3::Hash, &[u8]),
) -> Result<String> {
    let mut options: serde_json::Value = serde_json::from_str(argument)?;
    let fps: vork_time::Time =
        serde_json::from_value(options.get("fps").cloned().unwrap_or(24.into()))?;
    ensure!(
        fps > vork_time::Time::ZERO && !times.is_empty(),
        "environment footage needs positive fps and displayed frames"
    );
    let output = std::path::absolute(
        options["output"]
            .as_str()
            .context("environment output path")?,
    )?;
    ensure!(
        output
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("exr")),
        "environment footage output must be an EXR path"
    );

    // Refuse every destination before decoding or writing any frame. Retain
    // the original input identity: prepare() only sees our scratch EXR.
    let mut input_identities = std::collections::HashSet::new();
    for input in inputs {
        if let Some(identity) = file_identity(&input)
            .with_context(|| format!("identify environment source {}", input.display()))?
        {
            input_identities.insert(identity);
        }
    }
    let stem = output.with_extension("").to_string_lossy().into_owned();
    for index in 0..times.len() {
        for destination in [
            format!("{stem}.{index:06}.exr"),
            format!("{stem}.source.{index:06}.exr"),
        ] {
            if let Some(identity) = file_identity(Path::new(&destination))
                .with_context(|| format!("identify environment output {destination}"))?
            {
                ensure!(
                    !input_identities.contains(&identity),
                    "environment output must differ from source: {destination}"
                );
            }
        }
    }
    let name = options
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("environment")
        .to_owned();
    let image_name = format!("{name}_image");
    let weight_name = format!("{name}_lighting_weight");
    let sun_name = format!("{name}_sun");
    let scratch = std::env::temp_dir().join(format!(
        "vork-environment-source-{}-{}.exr",
        std::process::id(),
        blake3::hash(argument.as_bytes())
    ));
    struct Scratch(std::path::PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let scratch = Scratch(scratch);
    if options.get("color_space").is_none() {
        let source = Path::new(
            options["file"]
                .as_str()
                .context("environment source path")?,
        );
        options["color_space"] = if source
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("exr") || e.eq_ignore_ascii_case("hdr"))
        {
            "Linear Rec.709"
        } else {
            "sRGB"
        }
        .into();
    }
    let mut result = None;
    let mut fields = ["rotation", "color", "energy", "angle"].map(|field| (field, Vec::new()));
    let mut clocks = Vec::new();
    for (index, time) in times.iter().enumerate() {
        let file = format!("{stem}.{index:06}.exr");
        let source = frame(time)?.into_rgba32f();
        source.save(&scratch.0)?;
        {
            use exr::prelude::*;
            let channels = SpecificChannels::rgba(|Vec2(x, y): Vec2<usize>| {
                let p = source.get_pixel(x as u32, y as u32);
                (p[0], p[1], p[2], p[3])
            });
            Image::from_layer(Layer::new(
                (source.width() as usize, source.height() as usize),
                sequence_attributes(),
                Encoding::SMALL_LOSSLESS,
                channels,
            ))
            .write()
            .to_file(format!("{stem}.source.{index:06}.exr"))?;
        }
        options["file"] = serde_json::json!(scratch.0);
        options["output"] = file.into();
        let fragment: serde_json::Value =
            serde_json::from_str(&prepare(&serde_json::to_string(&options)?, &mut cached, &mut publish)?)?;
        let sun=fragment["nodes"].get(1).cloned().unwrap_or_else(||serde_json::json!({"type":"light","shape":"sun","name":sun_name,"visible_camera":false,"rotation":[0,0,0,1],"color":[0,0,0],"energy":0,"angle":0}));
        for (field, keys) in &mut fields {
            keys.push(serde_json::json!([time, sun[*field]]));
        }
        clocks.push(serde_json::json!([time, index as f64 + 0.5]));
        if result.is_none() {
            let mut first = fragment;
            if first["nodes"].as_array().unwrap().len() == 1 {
                first["nodes"].as_array_mut().unwrap().push(sun);
            }
            result = Some(first);
        }
    }
    let mut result = result.unwrap();
    result["images"][&image_name]["file"] = format!("{stem}.source.%06d.exr").into();
    result["images"][&weight_name]["file"] = format!("{stem}.%06d.exr").into();
    let mut tracks=fields.into_iter().map(|(field,keys)|serde_json::json!({"path":["nodes",sun_name,field],"interpolation":"STEP","keys":keys})).collect::<Vec<_>>();
    tracks.push(serde_json::json!({"path":["images",image_name,"time"],"interpolation":"STEP","keys":clocks}));
    tracks.push(serde_json::json!({"path":["images",weight_name,"time"],"interpolation":"STEP","keys":clocks}));
    result["animation"] = serde_json::json!([{"tracks":tracks}]);
    Ok(serde_json::to_string(&result)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frame_rates_reject_invalid_values_before_decoding() {
        for fps in [
            serde_json::json!("0/1"),
            serde_json::json!("-30000/1001"),
            serde_json::json!("1/0"),
            serde_json::json!("bad"),
        ] {
            let argument =
                serde_json::json!({"file":"missing.mov","output":"weighted.exr","fps":fps})
                    .to_string();
            assert!(
                prepare_frames(
                    &argument,
                    &[],
                    &[0_f32.into()],
                    |_| panic!("invalid rate must be rejected first"),
                    |_| None,
                    |_, _| {}
                )
                .is_err()
            );
            assert!(prepare(&argument, |_| None, |_, _| {}).is_err());
        }
    }
    #[test]
    fn percent_video_name_reaches_decoder() {
        let argument =
            serde_json::json!({"file":"100%clip.mov","output":"weighted.exr"}).to_string();
        let error = prepare_frames(
            &argument,
            &["100%clip.mov".into()],
            &[0_f32.into()],
            |_| anyhow::bail!("decoder reached"),
            |_| None,
            |_, _| {},
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "decoder reached");
    }
    #[test]
    fn frame_outputs_cannot_overwrite_input_sequence() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("vork-environment-collision-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        for suffix in ["", ".source"] {
            let input = root.join(format!("weighted{suffix}.000000.exr"));
            let bytes = b"source must survive";
            std::fs::write(&input, bytes)?;
            let argument=serde_json::json!({"file":root.join(format!("weighted{suffix}.%06d.exr")),"output":root.join("weighted.exr")}).to_string();
            let error = prepare_frames(
                &argument,
                &[input.clone()],
                &[0_f32.into()],
                |_| panic!("must reject before decoding"),
                |_| None,
                |_, _| {},
            )
            .unwrap_err();
            assert!(
                error.to_string().contains("output must differ from source"),
                "{error}"
            );
            assert_eq!(std::fs::read(input)?, bytes);
        }
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn frame_outputs_cannot_overwrite_hard_linked_input() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("vork-environment-hard-link-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let input = root.join("input.000000.exr");
        let bytes = b"source must survive";
        std::fs::write(&input, bytes)?;
        for suffix in ["", ".source"] {
            let destination = root.join(format!("weighted{suffix}.000000.exr"));
            std::fs::hard_link(&input, &destination)?;
            let argument=serde_json::json!({"file":root.join("input.%06d.exr"),"output":root.join("weighted.exr")}).to_string();
            let error = prepare_frames(
                &argument,
                &[input.clone()],
                &[0_f32.into()],
                |_| panic!("must reject before decoding"),
                |_| None,
                |_, _| {},
            )
            .unwrap_err();
            assert!(
                error.to_string().contains("output must differ from source"),
                "{error}"
            );
            assert_eq!(std::fs::read(&input)?, bytes);
            assert_eq!(std::fs::read(&destination)?, bytes);
            std::fs::remove_file(destination)?;
        }
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn core_split_preserves_camera_rgb_and_measured_rgb_energy() -> Result<()> {
        let mut image = image::Rgb32FImage::from_pixel(512, 256, image::Rgb([2., 3., 4.]));
        for y in 120..124 {
            for x in 250..254 {
                image.put_pixel(x, y, image::Rgb([2e6, 1e6, 5e5]));
            }
        }
        let (residual, sun) = split(&image, 10.);
        let sun = sun.context("isolated core")?;
        let mut removed = glam::DVec3::ZERO;
        for y in 0..image.height() {
            let omega = std::f64::consts::TAU / image.width() as f64
                * ((std::f64::consts::PI * y as f64 / image.height() as f64).cos()
                    - (std::f64::consts::PI * (y + 1) as f64 / image.height() as f64).cos());
            for x in 0..image.width() {
                removed += (Vec3::from_array(image.get_pixel(x, y).0)
                    - Vec3::from_array(residual.get_pixel(x, y).0))
                .as_dvec3()
                    * omega;
            }
        }
        let represented = Vec3::from_array(sun.color).as_dvec3() * sun.intensity as f64
            / (sun.angle as f64 * 0.25).cos().powi(2);
        assert!((represented - removed).abs().max_element() < removed.max_element() * 1e-6);
        assert_eq!(residual.get_pixel(0, 0), image.get_pixel(0, 0));
        let root =
            std::env::temp_dir().join(format!("vork-std-environment-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let input = root.join("source.exr");
        let output = root.join("weighted.exr");
        image.save(&input)?;
        let argument = serde_json::json!({"file":input,"output":output}).to_string();
        let mut cache = std::collections::HashMap::new();
        let first = prepare(
            &argument,
            |_| None,
            |key, data| {
                cache.insert(key, data.to_vec());
            },
        )?;
        let second = prepare(
            &argument,
            |key| cache.get(&key).cloned(),
            |_, _| panic!("extraction should hit content cache"),
        )?;
        assert_eq!(first, second);
        let weighted = image::open(&output)?.into_rgba32f();
        for ((full, weight), residual) in
            image.pixels().zip(weighted.pixels()).zip(residual.pixels())
        {
            for k in 0..3 {
                assert!((full[k] * weight[0] - residual[k]).abs() < 0.001);
            }
        }
        let fragment: serde_json::Value = serde_json::from_str(&first)?;
        assert_eq!(fragment["nodes"][1]["shape"], "sun");
        assert_eq!(fragment["nodes"][1]["visible_camera"], false);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn broad_and_zero_energy_fields_do_not_create_lights() {
        for rgb in [[0.; 3], [1000.; 3]] {
            let image = image::Rgb32FImage::from_pixel(64, 32, image::Rgb(rgb));
            let (residual, sun) = split(&image, 10.);
            assert_eq!(residual, image);
            assert!(sun.is_none());
        }
    }
    #[test]
    fn frame_preprocessing_preserves_vfr_clock_and_step_lights() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("vork-std-environment-vfr-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let argument=serde_json::json!({"file":"source.mov","output":root.join("weighted.exr"),"fps":"30000/1001"}).to_string();
        let result = prepare_frames(
            &argument,
            &[],
            &[0_f32.into(), 0.04_f32.into(), 0.15_f32.into()],
            |time| {
                Ok(image::DynamicImage::ImageRgb32F(
                    image::Rgb32FImage::from_pixel(8, 4, image::Rgb([time.as_f32(), 0., 0.])),
                ))
            },
            |_| None,
            |_, _| {},
        )?;
        let fragment: serde_json::Value = serde_json::from_str(&result)?;
        let tracks = fragment["animation"][0]["tracks"].as_array().unwrap();
        assert_eq!(tracks.len(), 6);
        let clocks = tracks.last().unwrap();
        assert_eq!(clocks["interpolation"], "STEP");
        assert_eq!(clocks["keys"][2][0], serde_json::json!(0.15f32));
        assert!((clocks["keys"][2][1].as_f64().unwrap() - 2.5).abs() < 1e-6);
        assert!(root.join("weighted.000002.exr").exists());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
