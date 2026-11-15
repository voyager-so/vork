//! Developer-only retained-device worker for scripts/profile-conformance.py.
//! One request owns a fresh install and frame; only the existing Renderer is retained.
use anyhow::{Context, Result, ensure};
use exr::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
    time::Instant,
};
use vork_render::{FloatOutputs, ReadbackOutputs, ReadbackPrecision, Renderer};
use vork_scene::{DEFAULT_PROFILE, Profile, Scene};

#[derive(Deserialize)]
struct Request {
    scene: PathBuf,
    #[serde(default)]
    internal_media: bool,
    resolution: Option<[u32; 2]>,
    preview: Option<PathBuf>,
    linear_rgba_output: Option<PathBuf>,
    #[serde(default = "default_at")]
    at: f32,
    reference_bounces: Option<u32>,
    profile: Option<String>,
    overrides: serde_json::Map<String, Value>,
    output: PathBuf,
    reference_key: Option<String>,
    #[serde(default)]
    refresh_reference: bool,
    #[serde(default)]
    measure_warm: bool,
    #[serde(default)]
    measure_spans: bool,
}

#[derive(serde::Serialize, Deserialize)]
struct Reference {
    exr: Vec<u8>,
    metadata: Value,
    timing: Value,
}

fn default_at() -> f32 { 0.5 }

fn publish(request: &Request, reference: &Reference) -> Result<()> {
    std::fs::write(&request.output, &reference.exr)?;
    std::fs::write(
        request.output.with_file_name("frame.exr.meta.json"),
        serde_json::to_vec_pretty(&reference.metadata)?,
    )?;
    std::fs::write(
        request.output.with_file_name("frame.exr.timings.json"),
        serde_json::to_vec_pretty(&reference.timing)?,
    )?;
    Ok(())
}

fn render(
    renderer: &mut Renderer,
    store: &vork_render::disk_cache::Store,
    request: Request,
) -> Result<Value> {
    renderer.reference_transport_bounces = request.reference_bounces;
    let start = Instant::now();
    let cache_path = request
        .reference_key
        .as_ref()
        .map(|key| store.path("profile-reference", 1, blake3::hash(key.as_bytes())));
    if !request.refresh_reference && request.linear_rgba_output.is_none()
        && let Some(reference) = cache_path
            .as_ref()
            .and_then(|path| vork_render::disk_cache::read(path))
            .and_then(|bytes| serde_json::from_slice::<Reference>(&bytes).ok())
    {
        publish(&request, &reference)?;
        let mut timing = reference.timing;
        timing["reference_cache_hit"] = true.into();
        return Ok(timing);
    }
    let profile = Profile::load(request.profile.as_deref().unwrap_or(DEFAULT_PROFILE))?;
    let mut scene = Scene::from_json(&vork_std::load_scene(&request.scene)?)?;
    if let Some(resolution) = request.resolution {
        scene.render.resolution = resolution;
    }
    let (intent, techniques): (serde_json::Map<_,_>, serde_json::Map<_,_>) = request.overrides.iter()
        .map(|(key,value)| (key.clone(),value.clone()))
        .partition(|(key,_)| matches!(key.as_str(), "seed" | "denoise"));
    let base = profile.resolve(&intent)?;
    scene.apply_settings(vork_scene::test_settings(&base, &techniques)?, &profile.name)?;
    let [width, height] = scene.render.resolution;
    let install_start = Instant::now();
    let base = request.scene.parent().context("scene directory")?;
    let mut installed = if request.internal_media {
        renderer.install_media_for_test(scene, base)?
    } else {
        renderer.install(scene, base)?
    };
    let install_seconds = install_start.elapsed().as_secs_f64();
    let mut frame = installed.create_frame_slot(renderer, ReadbackPrecision::Float)?;
    frame.profile=request.measure_spans;
    let frame_redos=std::cell::Cell::new(0u32);
    let mut render_once = || -> Result<_> { loop {
        frame.submit_outputs(
            &installed,
            renderer,
            request.at,
            ReadbackOutputs {
                color: request.preview.is_some(),
                float: FloatOutputs {
                    color: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;
        match frame.collect(renderer) {
            Ok(output) => break Ok(output),
            Err(error) => {
                let Some(redo) = error.downcast_ref::<vork_render::FrameRedo>() else {
                    return Err(error);
                };
                frame_redos.set(frame_redos.get()+1);
                installed.redo_frame(renderer, std::slice::from_mut(&mut frame), redo.0, 0)?;
            }
        }
    }};
    let cold_start=Instant::now();
    let output=render_once()?;
    let cold_frame_seconds=cold_start.elapsed().as_secs_f64();
    let cold_scene_seconds=start.elapsed().as_secs_f64();
    let (warm_frame_seconds,warm_linear_hash,warm_timing)=if request.measure_warm {
        let warm_start=Instant::now();
        let warm=render_once()?;
        ensure!(warm.linear_rgba.as_ref().is_some_and(|pixels|pixels.iter().flatten().all(|v|v.is_finite())),"nonfinite warm frame");
        let seconds=warm_start.elapsed().as_secs_f64();
        // Numeric image equality treats both signed zero encodings alike.
        let pixels:Vec<f32>=warm.linear_rgba.as_deref().context("warm pixels")?.iter().flatten().map(|&v|if v==0. {0.} else {v}).collect();
        let hash=blake3::hash(bytemuck::cast_slice(&pixels)).to_hex().to_string();
        let timing=serde_json::json!({"gpu_elapsed_ms":warm.timing.gpu_elapsed_ms,
            "cpu_prepare_ms":warm.timing.cpu_prepare_ms,"cpu_submit_ms":warm.timing.cpu_submit_ms,
            "cpu_finish_ms":warm.timing.cpu_finish_ms,"gpu_spans_ms":warm.timing.gpu_spans_ms});
        (Some(seconds),Some(hash),Some(timing))
    } else {(None,None,None)};
    if let Some(path) = &request.preview {
        let mut encoder = png::Encoder::new(std::fs::File::create(path)?, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&output.rgba)?;
        writer.finish()?;
    }
    let pixels = output
        .linear_rgba
        .context("missing scene-linear readback")?;
    ensure!(
        pixels.iter().flatten().all(|v| v.is_finite()),
        "nonfinite frame"
    );
    if let Some(path) = &request.linear_rgba_output {
        std::fs::write(path, bytemuck::cast_slice(&pixels))?;
    }
    let layer = Layer::new(
        (width as usize, height as usize),
        LayerAttributes::default(),
        Encoding::UNCOMPRESSED,
        SpecificChannels::rgb(|position: Vec2<usize>| {
            let p = pixels[position.y() * width as usize + position.x()];
            (p[0], p[1], p[2])
        }),
    );
    Image::from_layer(layer)
        .write()
        .non_parallel()
        .to_file(&request.output)?;
    let metadata = json!({"profile":{"name":profile.name,"version":profile.version},
        "settings":installed.render().settings,"camera":output.camera});
    let timing = json!({"device":renderer.adapter_info.name,
        "backend":format!("{:?}",renderer.adapter_info.backend),
        "driver":renderer.adapter_info.driver_info, "install_seconds":install_seconds,
        "reference_bounces":request.reference_bounces, "at":request.at,
        "internal_media":request.internal_media,
        "wall_seconds":start.elapsed().as_secs_f64(), "retained_device":true,
        "cold_scene_seconds":cold_scene_seconds,"cold_frame_seconds":cold_frame_seconds,"warm_frame_seconds":warm_frame_seconds,"frame_redos":frame_redos.get(),"warm_linear_hash":warm_linear_hash,"warm_timing":warm_timing});
    let reference = Reference {
        exr: std::fs::read(&request.output)?,
        metadata,
        timing: timing.clone(),
    };
    publish(&request, &reference)?;
    if let Some(path) = cache_path {
        vork_render::disk_cache::write(&path, &serde_json::to_vec(&reference)?)?;
    }
    Ok(timing)
}

fn main() -> Result<()> {
    let mut renderer = Renderer::new()?;
    let store = vork_render::disk_cache::Store::new(&renderer.adapter_info);
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let result = serde_json::from_str::<Request>(&line?)
            .map_err(anyhow::Error::from)
            .and_then(|request| render(&mut renderer, &store, request));
        let reply = match result {
            Ok(timing) => json!({"ok":true,"timing":timing}),
            Err(error) => json!({"ok":false,"error":format!("{error:#}")}),
        };
        println!("{reply}");
        io::stdout().flush()?;
    }
    Ok(())
}
