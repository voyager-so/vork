//! Complete ordered shot rendering, encoding and CPU output delivery.
pub mod worker;
pub use crate::png_output::Output;
use crate::png_output::{FrameOutputWriter, Outputs};
use crate::{FRAMES_IN_FLIGHT, FrameSlot, InstalledScene, Renderer, constants};
use anyhow::{Context, Result};
use crate::failure::{FailureKind, device, panicked, scene as scene_error, setup};

macro_rules! ensure_usage {
    ($condition:expr, $($message:tt)*) => {
        if !$condition { return Err(anyhow::anyhow!($($message)*).context(FailureKind::Usage)); }
    };
}
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
use vork_encode::{Backend, EncodeSettings, Encoder};
use vork_scene::Scene;
/// Output and scheduling values for a shot; structural GPU resources stay installed.
#[derive(Clone)]
pub struct Shot {
    /// Finish submitted frames and close output files when requested by the caller.
    pub stop: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub scene: PathBuf,
    /// Empty selects the built-in control set and every scene-declared AOV.
    pub outputs: Vec<Output>,
    pub out: Vec<vork_scene::v2::OutputDestination>,
    pub at: Option<vork_time::Clock>,
    /// First and last frame index to render (inclusive, absolute; Blender's
    /// frame start/end). None selects the clip's first or last frame.
    pub frame_start: Option<usize>,
    pub frame_end: Option<usize>,
    pub resolution: Option<[u32; 2]>,
    /// Render settings: the profile (None = `std/final`) and explicit values above
    /// it (`settings`), replacing the profile's seed, samples and denoise intent.
    pub profile: Option<vork_scene::Profile>,
    /// Call fields: resolution, fps, seed, samples and denoise.
    pub settings: serde_json::Map<String, serde_json::Value>,
    #[cfg(any(test, feature = "test-support"))]
    pub test_techniques: serde_json::Map<String, serde_json::Value>,
    pub no_write: bool,
    pub gpu_only: bool,
    pub diagnostics: bool,
    /// Blender ImageFormatSettings.exr_codec; defaults to lossless ZIP (16 scanlines).
    pub exr_codec: crate::ExrCodec,
    /// Bits per component of PNG colour (still and directory frames); defaults to 8.
    pub png_depth: crate::PngDepth,
    /// A kept-alive session's first picture: one sample, no denoise, labelled in the
    /// receipt and the sidecar; never a final output. A one-shot render never sets it.
    pub provisional: bool,
    /// Rewrites loaded content with resolved settings (and the scene file's directory)
    /// before renderer-support validation and the resolution override.
    pub edit: Option<SceneEdit>,
    /// The authored scene document, when a caller holds it in memory (a kept-alive
    /// worker's patched document); `scene` then only names the file for paths and
    /// diagnostics and is not read.
    pub document: Option<Vec<u8>>,
}
/// A caller's scene rewrite, such as a caller-provided content transform.
pub type SceneEdit = fn(&mut vork_scene::Scene, &Path) -> Result<()>;

/// Finished files and their timings, retaining GPU ownership until dropped.
/// Library callers reclaim resources normally. A one-shot process can exit with
/// this value alive after flushing its output, avoiding redundant driver teardown.
/// Input decoder children have already been cancelled and reaped.
/// With hardware video, drop this before std::process::exit/libc exit; only _exit
/// may skip the drop without running driver exit hooks.
pub struct RenderedShot {
    pub report: serde_json::Value,
    // Drop dependent frame and scene resources before the device for library use.
    _encoders: Vec<Encoder>,
    _slots: Vec<FrameSlot>,
    /// None for a 2D canvas: it installs no 3D scene.
    _scene: Option<InstalledScene>,
    /// A 2D canvas's document and GPU side.
    _canvas: Option<crate::canvas::Kept>,
    _renderer: Option<Renderer>,
    /// The install identity this shot rendered under (delivery shape and resolved settings).
    delivery: String,
}
/// GPU owners a kept-alive worker lends to a shot and takes back afterwards
/// (`RenderedShot::into_retained`). The worker proves the document current; the
/// shot checks the delivery shape and resolved settings, and reinstalls on the
/// retained device when they differ.
pub struct Retained {
    pub renderer: Renderer,
    /// An installation with the delivery identity it was installed under.
    pub installed: Option<(InstalledScene, String)>,
    /// A 2D canvas's document and GPU side, reused when the next render resolves to the same scene.
    pub(crate) canvas: Option<crate::canvas::Kept>,
}
impl RenderedShot {
    /// Keep the device and installation for another shot. Output delivery has
    /// completed, so frame rings and finished encoders are released here.
    pub fn into_retained(self) -> Option<Retained> {
        let Self { _encoders, _slots, _scene, _canvas, _renderer, delivery, .. } = self;
        drop(_encoders);
        drop(_slots);
        _renderer.map(|renderer| Retained { renderer, installed: _scene.map(|scene| (scene, delivery)), canvas: _canvas })
    }
    /// A 2D canvas's result: no 3D installation.
    pub(crate) fn canvas(report: serde_json::Value, encoders: Vec<Encoder>, renderer: Renderer, canvas: crate::canvas::Kept) -> Self {
        Self { report, _encoders: encoders, _slots: Vec::new(), _scene: None, _canvas: Some(canvas), _renderer: Some(renderer), delivery: String::new() }
    }
    /// End a successful one-shot process after its caller has flushed reports.
    /// Files, encoders and decoder children are already finished. Library
    /// callers instead drop this value to reclaim GPU owners normally.
    #[doc(hidden)]
    pub fn exit_process(self) -> ! {
        let code = if self.report["interrupted"] == true { 130 } else { 0 };
        // The output is delivered; what this process compiled reaches the disk before it ends.
        if let Some(renderer) = &self._renderer {
            renderer.finish_pipeline_save();
        }
        exit_process(code)
    }
}
/// End a process that keeps GPU owners alive, after every output is delivered
/// and every Rust stream is flushed (one-shot renders and kept-alive workers).
/// The caller has finished its pipeline-cache save (`Renderer::finish_pipeline_save`);
/// a save still running here is abandoned, and the store keeps its previous entries.
#[doc(hidden)]
pub fn exit_process(code: i32) -> ! {
    // _exit skips the pipeline cache's destructor. Its Metal binary archive
    // backs onto temporary files (read lazily while the process runs);
    // remove them here, after every output is delivered. Live owners and
    // any unfinished background compiles keep the in-memory archive. A
    // serialization in flight finishes first, so its file is removed too;
    // nothing is serialized or published afterwards.
    #[cfg(target_os = "macos")]
    wgpu::hal::metal::release_archive_files_for_exit();
    // libc exit can run CUDA/NVENC atexit hooks against intentionally live
    // owners. On Unix, the OS reclaims completed resources and discards
    // unfinished optional compiler work without driver teardown. Windows
    // process exit still runs main-thread TLS destructors; their GPU owners
    // must tolerate other thread-local registries already being destroyed.
    #[cfg(unix)]
    // SAFETY: Shot::render joined file writers, reaped muxers and finished
    // outputs; the caller flushed Rust streams before consuming this value.
    unsafe { libc::_exit(code) }
    #[cfg(not(unix))]
    std::process::exit(code)
}
impl Default for Shot {
    fn default() -> Self {
        Self {
            stop: None,
            scene: PathBuf::new(),
            outputs: vec![Output::Color],
            out: Vec::new(),
            at: None,
            frame_start: None,
            frame_end: None,
            resolution: None,
            profile: None,
            settings: Default::default(),
            #[cfg(any(test, feature = "test-support"))]
            test_techniques: Default::default(),
            no_write: false,
            gpu_only: false,
            diagnostics: false,
            exr_codec: Default::default(),
            png_depth: Default::default(),
            provisional: false,
            edit: None,
            document: None,
        }
    }
}
// Thread CPU time excludes time blocked on GPU completion or encoder backpressure.
#[cfg(unix)]
fn thread_cpu_ms() -> Option<f64> {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a valid writable timespec and the current thread's clock ID.
    let result = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) };
    (result == 0).then_some(t.tv_sec as f64 * 1000. + t.tv_nsec as f64 / 1e6)
}
#[cfg(not(unix))]
fn thread_cpu_ms() -> Option<f64> {
    None
}
fn cpu_elapsed(start: Option<f64>) -> Option<f64> {
    start.zip(thread_cpu_ms()).map(|(a, b)| (b - a).max(0.))
}
// Deliver only accepted frames; native surfaces and portable copies belong to
// the output ring until their consumers release them.
/// The frames a sequence of per-frame files refused and skipped, for the 3D and the canvas
/// frame loops alike. A refused frame (`FrameRefused`: its values were rejected, non-finite
/// radiance or a view's non-finite colour, say) writes nothing; later frames do not depend on
/// it, so the rest of the range renders, and the render then fails with the refusals,
/// leaving its output without meta.json.
#[derive(Default)]
pub(crate) struct Refusals {
    frames: Vec<(u32, String)>,
    /// Whether any refusal was not the scene's (an authored program's): then the summary is not either.
    device: bool,
}
impl Refusals {
    /// Record `error` as output frame `index`'s refusal and warn; false when it is no refusal,
    /// and the caller's error stands.
    pub(crate) fn skip(&mut self, index: u32, error: &anyhow::Error) -> bool {
        let Some(refusal) = crate::failure::cause::<crate::FrameRefused>(error) else { return false };
        eprintln!("warning: frame {index} refused and not written: {refusal}");
        self.frames.push((index, refusal.0.clone()));
        self.device |= crate::failure::kind(error) != Some(FailureKind::Scene);
        true
    }
    pub(crate) fn count(&self) -> usize { self.frames.len() }
    /// The render's result once the other frames are written: the refusals, when there were any.
    /// Withholding meta.json leaves the output marked incomplete, and the exit status reports them.
    pub(crate) fn finish(&self, frames: usize) -> Result<()> {
        if self.frames.is_empty() { return Ok(()); }
        let mut reasons: Vec<&str> = self.frames.iter().map(|(_, reason)| reason.as_str()).collect();
        reasons.sort_unstable();
        reasons.dedup();
        let list = self.frames.iter().map(|(index, _)| index.to_string()).collect::<Vec<_>>().join(", ");
        let summary = anyhow::anyhow!("{} of {frames} frames refused and not written ({}): {list}; the other frames were written, without meta.json",
            self.frames.len(), reasons.join("; "));
        Err(if self.device { device(summary) } else { scene_error(summary) })
    }
}
fn push_encoders(encoders: &mut [Encoder], slot: &mut FrameSlot, frame: usize, submission: &wgpu::SubmissionIndex) -> Result<f64> {
    if encoders.is_empty() {
        slot.ring.frame = Some(frame);
        return Ok(0.);
    }
    let start = Instant::now();
    let output = slot.view.output.clone();
    for (encoder, surface) in encoders.iter_mut().zip(&mut slot.ring.encoder_surfaces) {
        encoder.push(frame as u64, output.clone(), surface, submission).map_err(device)?;
    }
    slot.ring.frame = Some(frame);
    crate::timing::encoded(start);
    Ok(start.elapsed().as_secs_f64() * 1000.)
}
/// File sidecars retain the destination name, including its format, without an
/// EXR frame placeholder: `x.%06d.exr` → `x.exr.meta.json`, `exr/%06d.exr` →
/// `exr/meta.json` (as a directory destination).
pub(crate) fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let name = path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let name = match name.find("%06d") {
        Some(i) => {
            let before = name[..i].trim_end_matches(['.', '_', '-']);
            format!("{before}{}", &name[i + 4..])
        }
        None => name,
    };
    let name = if name.starts_with('.') && name[1..].find('.').is_none() {
        suffix.to_owned()
    } else {
        format!("{name}.{suffix}")
    };
    path.with_file_name(name)
}

/// A trailing separator (`frames.v2/`) marks a directory destination.
pub(crate) fn directory_marked(path: &Path) -> bool {
    path.as_os_str().to_string_lossy().ends_with(std::path::is_separator)
}
pub(crate) fn video_destination(path: &Path) -> bool {
    !directory_marked(path)
        && path.extension().is_some_and(|s| s.eq_ignore_ascii_case("mp4") || s.eq_ignore_ascii_case("mov"))
}
/// A known file extension, or a directory: no extension or a trailing separator
/// (`frames.v2/`). Any other extension is a mistake, never a silent PNG folder.
pub(crate) fn recognized_destination(path: &Path) -> bool {
    directory_marked(path)
        || path.extension().is_none_or(|e| ["png", "exr", "mp4", "mov", "wav"].iter().any(|k| e.eq_ignore_ascii_case(k)))
}

/// Frame placeholders belong to the basename; parent directories are literal.
pub(crate) fn has_frame_pattern(path: &Path) -> bool {
    let Some(name) = path.file_name() else { return false; };
    let name = name.to_string_lossy();
    if name.contains("##") {
        return true;
    }
    let bytes = name.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'%' {
            i += 1;
            continue;
        }
        i += 1;
        // A doubled percent is a literal percent in printf syntax.
        if bytes.get(i) == Some(&b'%') {
            i += 1;
            continue;
        }
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        if bytes.get(i) == Some(&b'd') {
            return true;
        }
    }
    false
}

fn resolve_outputs(outputs: Vec<Output>, aovs: &[vork_scene::v2::Aov]) -> Vec<Output> {
    if !outputs.is_empty() {
        return outputs;
    }
    Output::builtins()
        .chain(aovs.iter().map(|aov| Output::Aov(aov.name.clone())))
        .collect()
}

impl Shot {
    pub fn render(self) -> Result<RenderedShot> {
        // Device startup follows source admission and must finish before any error can
        // return to the caller and tear down the process or GPU runtime.
        std::thread::scope(|scope| self.render_scoped(scope))
    }

    /// Render with a kept-alive worker's device and, when its delivery identity
    /// matches, its installation. Both come back in the result; on an error they
    /// are dropped with it, so a failed request cannot leave a half-updated install.
    /// A device that lacks what the scene needs is replaced by one that provides it
    /// (`Renderer::providing`), and the installation is built again on that one.
    pub fn render_retained(self, retained: Retained) -> Result<RenderedShot> {
        std::thread::scope(|scope| self.render_scoped_with(scope, Some(retained), None))
    }

    /// Render the exact expansion against which the worker established identity,
    /// with what the worker holds: nothing before its first device.
    fn render_retained_decoded(self, retained: Option<Retained>, decoded: Scene) -> Result<RenderedShot> {
        std::thread::scope(|scope| self.render_scoped_with(scope, retained, Some(decoded)))
    }

    pub(crate) fn render_scoped<'scope>(self, scope: &'scope std::thread::Scope<'scope, '_>) -> Result<RenderedShot> {
        self.render_scoped_with(scope, None, None)
    }

    fn render_scoped_with<'scope>(self, scope: &'scope std::thread::Scope<'scope, '_>, retained: Option<Retained>, decoded: Option<Scene>) -> Result<RenderedShot> {
        let warnings = crate::warnings::Collector::default();
        let _warning_scope = warnings.enter();
        let wall = Instant::now();
        let (previous_renderer, mut previous_install, mut previous_canvas) = match retained {
            Some(Retained { renderer, installed, canvas }) => (Some(renderer), installed, canvas),
            None => (None, None, None),
        };
        let Self {
            scene: scene_path,
            outputs,
            out,
            at,
            frame_start,
            frame_end,
            resolution,
            profile,
            settings,
            #[cfg(any(test, feature = "test-support"))]
            test_techniques,
            no_write,
            gpu_only,
            diagnostics,
            exr_codec,
            png_depth,
            provisional,
            edit,
            stop,
            document,
        } = self;
        let destinations = out;
        let out: Vec<PathBuf> = destinations.iter().map(|d| d.file.clone()).collect();
        ensure_usage!(
            !out.is_empty(),
            "at least one output destination is required"
        );
        // What a destination asks for is refused before any encoder is probed.
        let target = crate::output_device::target(&destinations)?;

        ensure_usage!(png_depth != crate::PngDepth::Sixteen
            || (!no_write && !gpu_only && out.iter().any(|p| !video_destination(p) && (directory_marked(p) || !p.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("exr"))))
                && (outputs.is_empty() || outputs.contains(&Output::Color))),
            "--png-depth 16 requires PNG colour output");
        if !no_write && !gpu_only {
            for path in &out {
                if video_destination(path) {
                    let dependency = vork_encode::validate_video_output(path).map_err(device)?;
                    vork_encode::validate_video_clock_capabilities(|| {
                        crate::ffmpeg_cache::clock_help(&dependency)
                    }).map_err(device)?;
                }
            }
        }
        let audio_only = out.iter().all(|p| !directory_marked(p) && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav")));
        let parse_start = Instant::now();
        let profile = match profile { Some(profile) => profile, None => vork_scene::Profile::load(vork_scene::DEFAULT_PROFILE)? };
        if profile.level == 3 {
            eprintln!("Hint: std/previs omits indirect light; interiors lit by bounced light can look black. Use --profile std/path for full bounce lighting (or std/draft for a quick approximation).");
        }
        let profile_record = serde_json::json!({"name": profile.name, "version": profile.version, "level": profile.level});
        let overrides = vork_scene::CallOverrides::resolve(settings)
            .map_err(|error| error.context(FailureKind::Usage))?;
        let settings = profile.resolve(&overrides.settings).map_err(|error| error.context(FailureKind::Usage))?;
        // A library draft or final profile denoises only where that is fast; an
        // explicit denoise value or profile file asks for it regardless of device.
        let default_denoise = (vork_scene::libraries::profile(&profile.name).is_some() && profile.level <= 1 && !overrides.settings.contains_key("denoise"))
            .then(|| (profile.name.clone(), profile.level, overrides.settings.contains_key("samples")));
        #[cfg(any(test, feature = "test-support"))]
        let settings = vork_scene::test_settings(&settings, &test_techniques)?;
        // Until the device is asked for, everything that fails is the document, its files or
        // the call: each refusal below is the scene's unless it says otherwise.
        let mut scene = if audio_only {
            let mut scene = match (decoded, &document) {
                (Some(scene), _) => scene,
                (None, Some(bytes)) => crate::decode_scene_document(bytes, &scene_path)?,
                (None, None) => crate::load_scene_document(&scene_path)?,
            };
            scene.render.settings = settings;
            scene.validate_content(false).map_err(scene_error)?;
            if let Some(edit) = edit { edit(&mut scene, scene_path.parent().unwrap_or(Path::new("."))).map_err(scene_error)?; }
            scene
        } else {
            match &document {
                Some(bytes) => load_document_with_settings(bytes, &scene_path, settings, &profile.name, edit, decoded).map_err(scene_error)?,
                None => crate::load_with_settings(&scene_path, settings, &profile.name, edit)?,
            }
        };
        if let Some(resolution)=overrides.resolution {scene.render.resolution=resolution;}
        if let Some(fps)=overrides.fps {scene.render.fps=fps;}
        scene.validate_content(!audio_only).map_err(scene_error)?;
        // What the scene can encode for the display is refused with the scene's other errors, before device work.
        crate::output_device::admit(&scene, &destinations, target).map_err(scene_error)?;
        let raw_colors: std::collections::BTreeMap<_, _> = destinations.iter()
            .map(|destination| (destination.file.clone(), crate::output_device::RawColor::of(&scene, destination))).collect();
        let clip_frames = clip_frame_count(&scene.render).map_err(scene_error)?;
        let first_frame = frame_start.unwrap_or(0);
        let last_frame = frame_end.unwrap_or(clip_frames.saturating_sub(1));
        ensure_usage!(first_frame <= last_frame && last_frame < clip_frames,
            "frame range {first_frame}..{last_frame} must lie within the clip's frames 0..{} (render.fps x render.duration) with start <= end",
            clip_frames.saturating_sub(1));
        if !no_write && !gpu_only {
            for path in out.iter().filter(|path| video_destination(path)) {
                vork_encode::validate_video_rate(scene.render.fps, path).context(FailureKind::Usage)?;
                vork_encode::validate_video_clock(scene.render.fps, (last_frame + 1 - first_frame) as u64, path).map_err(device)?;
            }
        }
        let wav = |p: &PathBuf| {
            !directory_marked(p) && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav"))
        };
        if out.iter().any(wav) {
            ensure_usage!(
                out.iter().all(wav),
                "mixed WAV and picture destinations are not built yet; render them in separate calls"
            );
            ensure_usage!(
                at.is_none() && !gpu_only,
                "WAV renders a frame range; omit --at and --gpu-only"
            );
            ensure_usage!(outputs.is_empty() && !provisional && png_depth != crate::PngDepth::Sixteen,
                "WAV does not accept picture outputs, provisional output or PNG depth");
            let base = scene_path.parent().unwrap_or(Path::new("."));
            let mix = crate::audio::Mix::new(&scene, base).map_err(scene_error)?;
            let root = scene
                .render
                .root
                .as_deref()
                .map(|n| mix.node(n))
                .transpose().map_err(scene_error)?;
            let frames = mix.render_range(first_frame, last_frame, scene.render.fps, root).map_err(scene_error)?;
            let mut sidecars = Vec::new();
            if !no_write {
                for path in &out {
                    std::fs::create_dir_all(
                        path.parent()
                            .filter(|p| !p.as_os_str().is_empty())
                            .unwrap_or(Path::new(".")),
                    )?;
                }
                // The owner picture delivery uses: lock the destinations, withdraw the previous
                // completion record before replacing any data, publish the new one last.
                let run = crate::output_run::OutputRun::begin(&out, None)?;
                for path in &out {
                    vork_audio::write_wav(path, &frames)?;
                }
                let mut metadata = Vec::new();
                for path in &out {
                    let meta = sidecar(path, "meta.json");
                    let timings = sidecar(path, "timings.json");
                    metadata.push((
                        meta.clone(),
                        serde_json::to_vec(
                            &serde_json::json!({"version":1,"kind":"audio","fps":scene.render.fps,"frame_start":first_frame,"frame_end":last_frame,"warnings":[],"audio":{"sample_rate":vork_audio::SAMPLE_RATE,"channels":["L","R"],"samples":frames.len()}}),
                        )?,
                    ));
                    std::fs::write(
                        &timings,
                        serde_json::to_vec(
                            &serde_json::json!({"version":1,"audio":true,"wall_ms":wall.elapsed().as_secs_f64()*1000.}),
                        )?,
                    )?;
                    sidecars.extend([meta, timings]);
                }
                run.complete(&metadata)?;
            }
            return Ok(RenderedShot {
                report: serde_json::json!({"audio":true,"sample_rate":vork_audio::SAMPLE_RATE,"samples":frames.len(),"frames":last_frame+1-first_frame,"wall_ms":wall.elapsed().as_secs_f64()*1000.,"warnings":[],"sidecars":sidecars,"interrupted":false}),
                _encoders: Vec::new(),
                _slots: Vec::new(),
                _scene: None,
                _canvas: None,
                _renderer: previous_renderer,
                delivery: String::new(),
            });
        }
        let sound = if !no_write && !gpu_only && out.iter().any(|p| video_destination(p)) {
            crate::audio::MovieSound::prepare(
                &scene,
                scene_path.parent().unwrap_or(Path::new(".")),
                first_frame,
                last_frame,
                &out,
            ).map_err(scene_error)?
        } else {
            None
        };
        // Reject scene/file problems before native adapter, device or encoder work.
        // The device is opened for what this scene needs. A retained device that provides
        // it is used as it is. One that does not is replaced, and what was installed on it
        // goes first: an installation never meets a device other than the one that made it.
        let needs = crate::rt::device_needs(&scene).map_err(scene_error)?;
        let device_reused = previous_renderer.as_ref().is_some_and(|renderer| renderer.provides(needs));
        if !device_reused {
            previous_install = None;
            previous_canvas = None;
        }
        // Receipts describe this request: creations and timings from earlier shots
        // on a retained device are excluded.
        let pipeline_timings_before: std::collections::BTreeMap<_, _> = previous_renderer.as_ref().filter(|_| device_reused)
            .map(|r| r.pipeline_timings().into_iter().collect()).unwrap_or_default();
        let shader_creation_start = previous_renderer.as_ref().filter(|_| device_reused).map_or(0, |r| crate::timing::creations(&r.device).len());
        let device_worker = scope.spawn(move || Renderer::providing(previous_renderer, needs).map_err(setup));

        if let Some(resolution) = resolution {
            ensure_usage!(
                resolution.iter().all(|&v| v > 0),
                "resolution must be positive"
            );
            scene.render.resolution = resolution;
        }
        // A 2D canvas (composition: {space: "2d"}) renders through the canvas stage.
        if scene.is_canvas() {
            return crate::canvas::render(crate::canvas::Request {
                base: scene_path.parent().unwrap_or(Path::new(".")).to_path_buf(),
                scene, out, outputs, at, frame_start, frame_end, png_depth, exr_codec, raw_colors, display: target, no_write, gpu_only, diagnostics, stop, profile_record, device_reused,
                sound, kept: previous_canvas,
            }, device_worker, &warnings, wall);
        }
        // HDR code values are written in 16 bits, whatever depth was asked for.
        let png_depth = target.png_depth(png_depth);
        for output in &outputs {
            if let Output::Aov(name)=output {ensure_usage!(name=="shadow_catcher" || scene.render.aovs.iter().any(|a| &a.name==name),"unknown AOV output {name}");}
        }
        let outputs = resolve_outputs(outputs, &scene.render.aovs);
        let scene_parse_ms = parse_start.elapsed().as_secs_f64() * 1000.;
        let authored_at=at;
        let at=authored_at.as_ref().map(|time|vork_time::Request::from(time).numeric()).transpose().map_err(|error|error.context(FailureKind::Usage))?;
        if let Some(t) = at {
            ensure_usage!(
                t.is_finite() && authored_at.as_ref().is_none_or(|t|t.exact()>=Default::default()),
                "--at must be nonnegative and finite"
            );
        }
        let [w, h] = scene.render.resolution;
        let fps = scene.render.fps;
        // A trailing separator always names a directory, whatever its extension.
        let has_format = |path: &PathBuf, extension: &str| {
            !directory_marked(path) && path.extension().is_some_and(|s| s.eq_ignore_ascii_case(extension))
        };
        let is_video = |path: &PathBuf| video_destination(path);
        let is_exr = |path: &PathBuf| has_format(path, "exr");
        // One still: colour at the path, other channels beside it as NAME.<channel>.png.
        let is_png = |path: &PathBuf| has_format(path, "png");
        let is_file = |path: &PathBuf| is_exr(path) || is_png(path);
        for path in &out {
            ensure_usage!(
                recognized_destination(path),
                "--out {}: unrecognized extension; use .png (one still), .exr (with %06d), .mp4, .mov, or a directory (no extension, or end the path with /)",
                path.display()
            );
        }
        let video_paths: Vec<_> = out.iter().filter(|p| is_video(p)).collect();
        let directory_paths: Vec<_> = out.iter().filter(|p| !is_video(p)).collect();
        let video = !video_paths.is_empty();
        if directory_paths.iter().any(|p|is_exr(p)) {
            i32::try_from(fps.numerator()).context("EXR rate numerator exceeds i32").map_err(scene_error)?;
            u32::try_from(fps.denominator()).context("EXR rate denominator exceeds u32").map_err(scene_error)?;
        }
        for path in &directory_paths {
            ensure_usage!(
                !is_png(path) || !has_frame_pattern(path),
                "--out {}: PNG frame patterns are unsupported; use a directory (every frame as NNNNNN.png) or %06d.exr",
                path.display()
            );
        }
        // A .png is one still; with no time or range given it renders t=0, like Blender's
        // current frame. Mixed with frame directories or EXR sequences it stays an error.
        let implicit_still = at.is_none() && frame_start.is_none() && frame_end.is_none() && clip_frames > 1
            && !directory_paths.is_empty() && directory_paths.iter().all(|p| is_png(p));
        let at = if implicit_still { Some(0.) } else { at };
        ensure_usage!(at.is_none() || (frame_start.is_none() && frame_end.is_none()),
            "--at selects one still; it cannot be combined with --frame-start or --frame-end");
        let frames = if at.is_some() && !video {
            1
        } else {
            last_frame + 1 - first_frame
        };
        let in_flight = FRAMES_IN_FLIGHT.min(frames);
        let ring_depth = constants::OUTPUT_RING_DEPTH;
        let ring_entries = (in_flight * ring_depth).min(frames);
        let still_pass = video && at.is_some() && !directory_paths.is_empty();
        // Loop positions count from the range's first frame; files, sidecars and
        // times use absolute frame indices.
        let index = |frame: usize| first_frame + frame;
        // Frame identity stays integer; evaluation receives numeric seconds.
        let frame_request=|frame:usize| -> vork_time::Request {
            if video || at.is_none() {vork_time::Request::Frame(index(frame) as u32,fps)}
            else {authored_at.as_ref().map(vork_time::Request::from).unwrap_or(vork_time::Request::Numeric(0.))}
        };
        let frame_time = |frame: usize| frame_request(frame).numeric().expect("validated output time");
        let still_request=|| authored_at.as_ref().map(vork_time::Request::from).unwrap_or(vork_time::Request::Numeric(0.));
        let still_index=at.map(|_| -> Result<u32> {
            let frame=still_request().exact()?.scale(&vork_time::exact::decimal(&fps.to_string())?)
                .add(&vork_time::exact::Germ::point(vork_time::exact::decimal("1/2")?)).floor();
            u32::try_from(frame).context("still frame index exceeds u32 identity capacity").context(FailureKind::Usage)
        }).transpose().map_err(scene_error)?;
        let output_index = |frame: usize| if video || at.is_none() { index(frame) as u32 } else { still_index.unwrap() };
        for (i, path) in out.iter().enumerate() {
            ensure_usage!(
                !out[..i].contains(path),
                "duplicate output path: {}",
                path.display()
            );
        }
        if video {
            ensure_usage!(
                at.is_none() || !directory_paths.is_empty(),
                "--at with video requires a .png, directory or EXR still destination"
            );
            ensure_usage!(
                !no_write && !gpu_only,
                "video output cannot be combined with --no-write or --gpu-only"
            );
            ensure_usage!(w % 2 == 0 && h % 2 == 0, "video output requires even dimensions");
        }
        if video {
            i64::try_from(last_frame - first_frame).ok().and_then(|frames| frames.checked_mul(fps.denominator())).context("video timestamp overflow").map_err(scene_error)?;
        }
        for path in &video_paths {
            ensure_usage!(!path.is_dir(), "video output path is a directory: {}", path.display());
            std::fs::create_dir_all(
                path.parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new(".")),
            )?;
        }
        for path in &directory_paths {
            ensure_usage!(
                !is_png(path) || frames == 1 || still_pass,
                "--out {}: a PNG file is one still; add --at SECONDS, or use a directory for every frame",
                path.display()
            );
            if is_png(path) {
                std::fs::create_dir_all(
                    path.parent()
                        .filter(|p| !p.as_os_str().is_empty())
                        .unwrap_or(Path::new(".")),
                )?;
            } else if is_exr(path) {
                ensure_usage!(
                    path.file_name().unwrap().to_string_lossy().contains("%06d"),
                    "EXR destination requires a %06d frame placeholder"
                );
                std::fs::create_dir_all(
                    path.parent()
                        .filter(|p| !p.as_os_str().is_empty())
                        .unwrap_or(Path::new(".")),
                )?;
            } else {
                std::fs::create_dir_all(path)?;
            }
        }
        // Explicit selections replace only their numbered files. A still
        // alongside video uses its own absolute identity, not the video range.
        // The run begins (and removes old outputs) only after encoder
        // preparation has proved every output can be written.
        let selected = if let Some(index) = still_index {
            Some(index..=index)
        } else if frame_start.is_some() || frame_end.is_some() {
            Some(first_frame as u32..=last_frame as u32)
        } else {
            None
        };
        let outputs = Outputs::new(outputs);
        let readback = outputs.readback(
            directory_paths.iter().any(|p| !is_exr(p)),
            directory_paths.iter().any(|p| is_exr(p)),
            png_depth,
        );
        // GPU-only frames and directory-less video read nothing back, except a still pass.
        let install_readback = if still_pass || !(gpu_only || (video && directory_paths.is_empty()))
        {
            readback
        } else {
            crate::ReadbackOutputs::default()
        };
        let (renderer, installed, mut encoders, encoder_prepare_ms, encoder_attach_ms, foreground, output_run) =
            std::thread::scope(|prepare_scope| -> Result<_> {
                // Codec/driver preparation needs no render device. Begin it
                // before joining device creation; stills spawn no worker.
                let prepare = video.then(|| {
                    prepare_scope.spawn(|| -> Result<_> {
                        let start = Instant::now();
                        let prepared = video_paths
                            .iter()
                            .map(|path| {
                                Encoder::prepare(
                                    Backend::Auto,
                                    w,
                                    h,
                                    fps,
                                    EncodeSettings { diagnostics, primaries: target.video_primaries(), ..EncodeSettings::default() },
                                    path,
                                ).map_err(device)
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Ok((prepared, start.elapsed().as_secs_f64() * 1000.))
                    })
                });
                vork_slang::progress::phase("Render: preparing scene, shaders and encoder", 0);
                let renderer = match device_worker
                    .join()
                    .map_err(|panic| panicked("device creation worker", panic))
                    .and_then(|result| result)
                {
                    Ok(renderer) => renderer,
                    Err(error) => {
                        // Consume the scoped handle even on device failure so a
                        // preparation panic cannot replace the original error.
                        if let Some(worker) = prepare {
                            let _ = worker.join();
                        }
                        return Err(error);
                    }
                };
                if diagnostics {
                    eprintln!(
                        "device: {:?}; pipeline cache supported: {}",
                        renderer.adapter_info,
                        renderer.pipeline_cache_supported()
                    );
                }
                // Keep optional variants behind install AND frame-slot pipelines.
                let foreground = renderer.compiler.foreground_preparation();
                let (installed, encoders, encoder_prepare_ms, encoder_attach_ms, output_run) =
                    std::thread::scope(|scope| -> Result<_> {
                        let install = scope.spawn(|| {
                            let _warning_scope = warnings.enter();
                            let mut scene = scene;
                            let mut previous_install = previous_install;
                            let plan = install_plan(&renderer, &mut scene, default_denoise.as_ref(), install_readback, frames == 1, video || gpu_only, &outputs.0, frame_request(0), target)?;
                            if let Some((installed, key)) = previous_install.take() {
                                if key == plan.delivery { return Ok((installed, plan.delivery, true)); }
                                // Release the stale installation before the new one allocates.
                                drop(installed);
                            }
                            renderer.install_with_options(scene, scene_path.parent().unwrap_or(Path::new(".")), plan.options)
                            .with_context(|| format!("install scene {}", scene_path.display()))
                            .map(|installed| (installed, plan.delivery, false))
                        });
                        let (prepared, preparation_ms) = match prepare {
                            Some(worker) => worker.join().map_err(|panic| panicked("encoder preparation worker", panic))??,
                            None => (Vec::new(), 0.),
                        };
                        for prepared in &prepared {
                            prepared.validate_fallback(frames as u64, Some(&renderer.device)).map_err(device)?;
                        }
                        let output_run = if no_write || gpu_only { None } else {
                            Some(crate::output_run::OutputRun::begin(&out, selected.clone())?)
                        };
                        let attach_start = Instant::now();
                        // Attach is the longer branch; keep pipeline creation from contending.
                        renderer.compiler.set_critical(!prepared.is_empty());
                        let encoders = prepared
                            .into_iter()
                            .map(|prepared| {
                                prepared.open(
                                    renderer.device.clone(),
                                    renderer.queue.clone(),
                                    frames as u64,
                                    ring_entries,
                                ).map_err(device)
                            })
                            .collect::<Result<Vec<_>>>();
                        renderer.compiler.set_critical(false);
                        let encoders = encoders?;
                        let encoder_attach_ms = attach_start.elapsed().as_secs_f64() * 1000.;
                        let installed = install
                            .join()
                            .map_err(|panic| panicked("scene install worker", panic))??;
                        validate_clock_requests(&installed.0,(0..frames).map(&frame_request)
                            .chain(still_pass.then(&still_request))).map_err(scene_error)?;
                        Ok((installed, encoders, preparation_ms, encoder_attach_ms, output_run))
                    })?;
                Ok((
                    renderer,
                    installed,
                    encoders,
                    encoder_prepare_ms,
                    encoder_attach_ms,
                    foreground,
                    output_run,
                ))
            })?;
        let (mut installed, delivery, install_reused) = installed;
        let effective = &installed.render().settings;
        let adaptive = installed.path.as_ref().is_some_and(|path| path.adaptive());
        let denoise = effective.integrator.denoiser().is_some();
        let render_settings = serde_json::json!({
            "profile": profile_record, "samples": effective.samples,
            "adaptive": adaptive, "denoise": denoise,
            "resolution": [w, h], "frames": frames,
        });
        eprintln!("render: {}; {}{} samples ({}); denoise {}; {}x{}; {} frames",
            profile.name, if adaptive { "up to " } else { "" }, effective.samples,
            if adaptive { "adaptive" } else { "fixed" },
            if denoise { "on" } else { "off" }, w, h, frames);
        let encoder_readback = video.then(|| encoders.iter().any(|e| e.uses_pixel_readback()));
        let mut slots = (0..in_flight)
            .map(|_| {
                installed.create_frame_slot(
                    &renderer,
                    if directory_paths.iter().any(|p| is_exr(p)) {
                        crate::ReadbackPrecision::Float
                    } else {
                        crate::ReadbackPrecision::Packed
                    },
                ).map_err(device)
            })
            .collect::<Result<Vec<_>>>()?;
        // Install in frame order so native entry indices match the output ring.
        for phase in 0..ring_depth {
            for (index, slot) in slots.iter_mut().enumerate() {
                if phase * in_flight + index >= ring_entries {
                    continue;
                }
                if phase != 0 {
                    let spare = crate::ring::Entry::new(
                        &renderer.device,
                        &slot.view.post.pixels,
                        u64::from(w) * u64::from(h) * 4,
                        slot.view.float_output,
                        installed.readback,
                        slot.view.aovs.len(),
                        slot.view.post.display_base,
                    );
                    let previous = std::mem::replace(&mut slot.ring, spare);
                    slot.spare_ring.push_back(previous);
                }
                for encoder in &mut encoders {
                    slot.ring.encoder_surfaces.push(encoder.create_surface().map_err(setup)?);
                }
                if let Some(output) = slot
                    .ring
                    .encoder_surfaces
                    .first()
                    .and_then(|s| s.texture.as_ref())
                {
                    if output.format() == renderer.output_format() {
                        slot.set_output(&installed, &renderer, output.clone()).map_err(device)?;
                        slot.ring.output =
                            Some((slot.view.output.clone(), slot.view.output_bind.clone()));
                    }
                }
            }
        }
        for slot in &mut slots {
            slot.advance_ring();
        }
        let png_outputs: Vec<_> = if no_write || gpu_only {
            Vec::new()
        } else {
            directory_paths
                .iter()
                .map(|path| {
                    FrameOutputWriter::new(
                        (*path).clone(),
                        w,
                        h,
                        outputs.clone(),
                        is_exr(path),
                        exr_codec,
                        png_depth,
                        ring_entries,
                        if still_pass { 0 } else { first_frame },
                        if is_exr(path) {(fps.numerator() as i32,fps.denominator() as u32)} else {(1,1)},
                        target.display,
                        raw_colors[*path],
                    )
                })
                .collect()
        };
        let png_workers: usize = png_outputs.iter().map(|p| p.worker_count).sum();
        // Frames whose display image is delivered report its exposure.
        if video || gpu_only || readback.color || readback.color16 {
            crate::display_stats::prepare(&renderer);
        }
        // Preparation is complete: install's and the frame slots' pipelines all exist, and a cold 3D
        // install has spent seconds in the driver for them. Save them now, in the background, so a
        // render that is killed before it ends does not cost the next process that compile again.
        renderer.save_pipelines_soon();
        let mut foreground = Some(foreground);
        let ready_ms = wall.elapsed().as_secs_f64() * 1000.;
        vork_slang::progress::phase("Rendering frames", frames);
        let loop_start = Instant::now();
        let mut samples = Vec::new();
        let mut metadata_frames = Vec::new();
        let mut camera_frames = Vec::new();
        let mut still_camera = None;
        let mut still_exposure = None;
        let mut still_dropped_light_samples = 0;
        let mut still_dropped_path_samples = 0;
        let mut still_invalid_pixels = std::collections::BTreeMap::<String, crate::frame::InvalidPixels>::new();
        let mut still_invalid_count = 0u64;
        let mut invalid_pixel_count = 0u64;
        let mut first_invalid_pixel: Option<(u32, String, [u32; 2])> = None;
        let mut still_terminated_medium_paths = 0;
        let mut still_discarded_medium_radiance = 0.;
        let mut frame_exposures = Vec::new();
        let mut first_frame_ms = 0.;
        // Stage totals over every collected exposure, including a selected still.
        let mut gpu_spans_total: Vec<(&'static str, f64)> = constants::GPU_SPANS.iter().map(|&name| (name, 0.)).collect();
        let (mut denoise_busy_ms, mut denoise_wait_ms) = (0., 0.);
        let mut stage_totals = |timing: &crate::frame::FrameTiming| {
            for ((_, total), (_, ms)) in gpu_spans_total.iter_mut().zip(&timing.gpu_spans_ms) { *total += ms; }
            denoise_busy_ms += timing.denoise_pack_ms + timing.denoise_filter_ms + timing.denoise_upload_ms;
            denoise_wait_ms += timing.denoise_wait_ms;
        };
        let mut submit_cpu = vec![None; slots.len()];
        let mut output_release_ms = vec![0.; slots.len()];
        // A video cannot skip a frame; per-frame files can, so the rest of the range renders.
        let mut refused = Refusals::default();
        let mut redos = 0u32;
        let mut redo_ms = 0.;
        for slot in &mut slots {
            slot.profile = diagnostics;
            slot.write_color = video || gpu_only;
        }
        let rendered = (|| -> Result<_> {
        let mut memory_frames = Vec::new();
        let mut frame = 0;
        let mut next_submit = 0;
        let mut redo_started = None;
        while frame < frames {
            let stopping = stop.as_ref().is_some_and(|stop| stop.load(std::sync::atomic::Ordering::Relaxed));
            if stopping && frame > 0 && frame == next_submit { break; }
            let attempt = (|| -> Result<_> {
                // Deliver the first accepted frame before recording successors:
                // advancing their simulations can take minutes on the host.
                // Once output has started, restore the normal overlap window.
                let submit_limit = frame + if samples.is_empty() { 1 } else { in_flight };
                while next_submit < frames && next_submit < submit_limit {
                    if stopping && next_submit > 0 { break; }
                    let i = next_submit % slots.len();
                    let slot = &mut slots[i];
                    let cpu_start = thread_cpu_ms();
                    // `ring.frame` is a delivered frame the entry still holds. Advance
                    // past it once, and free the next entry's frame before reuse; a
                    // redone frame's entry was never delivered, so resubmission keeps
                    // it and frame f stays on encoder entry f % ring_entries.
                    let release_start = Instant::now();
                    if slot.ring.frame.is_some() {
                        slot.advance_ring();
                        if let Some(previous) = slot.ring.frame.take() {
                            for (encoder, surface) in encoders.iter_mut().zip(&mut slot.ring.encoder_surfaces) {
                                encoder.release(previous as u64, surface).map_err(device)?;
                            }
                        }
                    }
                    output_release_ms[i] = release_start.elapsed().as_secs_f64() * 1000.;
                    if gpu_only || (video && (directory_paths.is_empty() || still_pass)) {
                        slot.submit_indexed_outputs(&installed, &renderer, output_index(next_submit), frame_request(next_submit), crate::ReadbackOutputs::default()).map_err(device)?;
                    } else {
                        slot.submit_indexed_outputs(&installed, &renderer, output_index(next_submit), frame_request(next_submit), readback).map_err(device)?;
                    }
                    submit_cpu[i] = cpu_elapsed(cpu_start);
                    next_submit += 1;
                }
                let slot = &mut slots[frame % in_flight];
                let submission = slot.submission.as_ref().expect("submitted frame").clone();
                let cpu_start = thread_cpu_ms();
                let collect_start = Instant::now();
                let p = slot.collect_readback(&renderer).map_err(device)?;
                Ok((p, submission, cpu_start, collect_start))
            })();
            let (p, submission, cpu_start, collect_start) = match attempt {
                Ok(result) => result,
                Err(error) => {
                    // Only an installation whose publications remain valid can
                    // skip an isolated refusal. Failed retained producers require
                    // reinstallation, so their queued successors cannot be delivered.
                    if !video && installed.transients.check_recording().is_ok() && refused.skip(output_index(frame), &error) {
                        if !still_pass {
                            for writer in &png_outputs { writer.skip(index(frame))?; }
                        }
                        frame += 1;
                        vork_slang::progress::completed(frame);
                        continue;
                    }
                    let Some(redo) = error.downcast_ref::<crate::FrameRedo>() else { return Err(error); };
                    redo_started.get_or_insert_with(Instant::now);
                    // The exposure's own frame keys its shadow requests: a
                    // still at `at` is not loop frame `index(frame)`.
                    let exposure = output_index(frame) as usize;
                    redos += installed.redo_frame(&renderer, &mut slots, redo.0, exposure).map_err(device)?;
                    next_submit = frame;
                    continue;
                }
            };
            if let Some(started) = redo_started.take() { redo_ms += started.elapsed().as_secs_f64() * 1000.; }
            let invalid_count = p.timing.invalid_pixels.values().map(|pixels| pixels.count).sum::<u64>();
            invalid_pixel_count += invalid_count;
            stage_totals(&p.timing);
            if first_invalid_pixel.is_none() {
                first_invalid_pixel = p.timing.invalid_pixels.iter().find_map(|(view, pixels)|
                    pixels.coordinates.first().map(|&xy| (output_index(frame), view.clone(), xy)));
            }
            let i = frame % slots.len();
            let slot = &mut slots[i];
            let collect_wall_ms = collect_start.elapsed().as_secs_f64() * 1000.;
            // Retain main's CPU/GPU overlap decision; measured sample-group
            // limits still apply independently to every queue submission.
            slot.stream_samples = frame >= in_flight && collect_wall_ms < p.timing.cpu_submit_ms;
            // Portable encoders map the copy already recorded in this ring entry.
            // Deliver after collection, with later frames already queued; wait only
            // for this frame rather than draining unrelated work.
            // The existing final display reduction observes the exact RGBA8
            // image delivered to the encoder, after every compositor entry.
            // Check each accepted frame, including animated alpha, before push.
            if video_paths.iter().any(|path|has_format(path,"mp4")) {
                ensure_usage!(p.exposure.is_some_and(|e|e.coverage==1.),
                    "MP4 requires an opaque final image; transparency requires .mov (ProRes 4444)");
            }
            let encode_push_ms = push_encoders(&mut encoders, slot, frame, &submission)?;
            if diagnostics && (frame == 0 || frame + 1 == frames) {
                if let Some(memory) = renderer.device.generate_allocator_report() {
                    memory_frames.push(serde_json::json!({"frame": index(frame),
                        "allocated_bytes": memory.total_allocated_bytes,
                        "reserved_bytes": memory.total_reserved_bytes}));
                }
            }
            if video || outputs.has(Output::Camera) {
                camera_frames.push(p.camera);
            }
            let cpu_thread_ms = submit_cpu[i]
                .zip(cpu_elapsed(cpu_start))
                .map(|(a, b)| a + b);
            if samples.is_empty() {
                first_frame_ms = loop_start.elapsed().as_secs_f64() * 1000.;
                drop(foreground.take());
            }
            metadata_frames.push(serde_json::json!({"index":output_index(frame),"time":frame_time(frame),"camera":installed.camera_name_at_request(&frame_request(frame))?,"depth_unit_m":p.camera.depth_unit_m,"terminated_medium_paths":p.timing.terminated_medium_paths,"discarded_medium_radiance":p.timing.discarded_medium_radiance,"dropped_path_samples":p.timing.dropped_path_samples,"dropped_light_samples":p.timing.dropped_light_samples,"invalid_pixel_count":invalid_count,"invalid_pixels":&p.timing.invalid_pixels,"exposure":p.exposure.map(|e|e.json())}));
            if let Some(e) = p.exposure { frame_exposures.push((index(frame), e)); }
            samples.push(serde_json::json!({"frame":index(frame),"time":frame_time(frame),"cpu_thread_ms":cpu_thread_ms,"encode_push_wall_ms":encode_push_ms,"output_release_wall_ms":output_release_ms[i],"encode_pipe_wait_wall_ms":slot.ring.encoder_surfaces.iter().map(|s|s.pipe_wait_ms()).sum::<f64>(),"collect_wall_ms":collect_wall_ms,"cpu_prepare_ms":p.timing.cpu_prepare_ms,"cpu_submit_ms":p.timing.cpu_submit_ms,"cpu_finish_ms":p.timing.cpu_finish_ms,"denoise_wait_ms":p.timing.denoise_wait_ms,"denoise_pack_ms":p.timing.denoise_pack_ms,"denoise_filter_ms":p.timing.denoise_filter_ms,"denoise_upload_ms":p.timing.denoise_upload_ms,"timing_source":p.timing.timing_source,"gpu_ms":p.timing.gpu_ms,"gpu_elapsed_ms":p.timing.gpu_elapsed_ms,"gpu_spans_ms":&p.timing.gpu_spans_ms,"nonfinite":p.timing.nonfinite,"terminated_medium_paths":p.timing.terminated_medium_paths,"discarded_medium_radiance":p.timing.discarded_medium_radiance,"dropped_path_samples":p.timing.dropped_path_samples,"dropped_light_samples":p.timing.dropped_light_samples,"invalid_pixel_count":invalid_count,"invalid_pixels":&p.timing.invalid_pixels,"sample_count":p.timing.sample_count,"variants_pending":p.timing.variants_pending,"path_kernel":p.timing.path_kernel}));
            if !still_pass {
                for writer in &png_outputs { writer.push(index(frame), p.clone())?; }
            }
            frame += 1;
            vork_slang::progress::completed(frame);
        }
        vork_slang::progress::phase("Render: finishing encoders and writing files", 0);
        let finish_start = Instant::now();
        for encoder in &mut encoders {
            encoder.finish().map_err(device)?;
        }
        if let Some(sound) = &sound {
            sound.mux(&out, metadata_frames.len() as u64)?;
        }
        if !encoders.is_empty() { crate::timing::encoded(finish_start); }
        let encoder_finish_ms = if video {
            finish_start.elapsed().as_secs_f64() * 1000.
        } else {
            0.
        };
        if still_pass {
            vork_slang::progress::phase("Rendering selected still", 1);
            let still_frame = still_index.unwrap() as usize;
            let pixels = loop {
                let slot = &mut slots[0];
                slot.write_color = false;
                let attempt = slot.submit_indexed_outputs(&installed, &renderer, still_index.unwrap(), still_request(), readback)
                    .and_then(|()| slot.collect_readback(&renderer)).map_err(device);
                match attempt {
                    Ok(pixels) => break pixels,
                    Err(error) => {
                        let Some(redo) = error.downcast_ref::<crate::FrameRedo>() else { return Err(error); };
                        let started = Instant::now();
                        redos += installed.redo_frame(&renderer, &mut slots, redo.0, still_frame).map_err(device)?;
                        redo_ms += started.elapsed().as_secs_f64() * 1000.;
                    }
                }
            };
            vork_slang::progress::completed(1);
            stage_totals(&pixels.timing);
            still_dropped_light_samples = pixels.timing.dropped_light_samples;
            still_dropped_path_samples = pixels.timing.dropped_path_samples;
            still_invalid_pixels = pixels.timing.invalid_pixels.clone();
            still_invalid_count = still_invalid_pixels.values().map(|pixels| pixels.count).sum::<u64>();
            invalid_pixel_count += still_invalid_count;
            if first_invalid_pixel.is_none() {
                first_invalid_pixel = still_invalid_pixels.iter().find_map(|(view, pixels)|
                    pixels.coordinates.first().map(|&xy| (still_index.unwrap(), view.clone(), xy)));
            }
            still_terminated_medium_paths = pixels.timing.terminated_medium_paths;
            still_discarded_medium_radiance = pixels.timing.discarded_medium_radiance;
            still_exposure = pixels.exposure;
            still_camera = Some(pixels.camera);
            for writer in &png_outputs {
                writer.push(0, pixels.clone())?;
            }
        }
        Ok((encoder_finish_ms, memory_frames))
        })();
        let mut output_result = Ok(());
        vork_slang::progress::phase("Render: writing files and reports", 0);
        for writer in png_outputs {
            let result = writer.finish();
            if output_result.is_ok() { output_result = result; }
        }
        // Input decoders can still be blocked writing their next frame. Finish
        // them at the shot boundary, before a one-shot caller skips GPU teardown.
        for footage in installed.footage.values_mut() { footage.finish(); }
        let (encoder_finish_ms, memory_frames) = rendered?;
        output_result?;
        renderer.errors.check().map_err(device)?;
        // Save completed pipelines without waiting for optional variants. The
        // one-shot exit finishes this save after all outputs have been delivered.
        drop(foreground.take());
        renderer.save_pipelines_soon();
        // The other frames are written; withholding meta.json leaves the output
        // marked incomplete, and the exit status reports the refusal.
        refused.finish(frames)?;
        let interrupted = stop.as_ref().is_some_and(|stop| stop.load(std::sync::atomic::Ordering::Relaxed));
        let frames = if interrupted { metadata_frames.len() } else { frames };
        let mut sidecars = Vec::new();
        for path in &video_paths {
            let camera_path = sidecar(path, "camera.json");
            std::fs::write(
                &camera_path,
                serde_json::to_vec_pretty(&crate::camera::clip(&camera_frames, fps, Some(first_frame))?)?,
            )?;
            sidecars.push(camera_path);
        }
        if outputs.has(Output::Camera) && !no_write && !gpu_only {
            let frames = still_camera
                .as_ref()
                .map(std::slice::from_ref)
                .unwrap_or(&camera_frames);
            let bytes = serde_json::to_vec_pretty(&crate::camera::clip(frames, fps, at.is_none().then_some(first_frame))?)?;
            for path in &directory_paths {
                let camera_path = if is_file(path) {
                    sidecar(path, "camera.json")
                } else {
                    path.join("camera.json")
                };
                if let Some(parent) = camera_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&camera_path, &bytes)?;
                sidecars.push(camera_path);
            }
        }
        if implicit_still && !no_write && !gpu_only {
            eprintln!("rendered t=0; use --at SECONDS for another time, or a directory for every frame");
        }
        let pipeline_creation_ms: Vec<_> = renderer.pipeline_timings().into_iter().filter_map(|(name, ms)| {
            let delta = ms - pipeline_timings_before.get(&name).copied().unwrap_or(0.);
            (delta > 0.).then_some((name, delta))
        }).collect();
        let stages_end = Instant::now();
        let mut report = serde_json::json!({"version":4,"device":renderer.adapter_info.name,"backend":format!("{:?}",renderer.adapter_info.backend),"pipeline_cache_supported":renderer.pipeline_cache_supported(),"pipeline_cache_status":renderer.pipeline_cache_status,"pipeline_cache_input_bytes":renderer.pipeline_cache_input_bytes,"pipeline_cache_rejection":renderer.pipeline_cache_rejection,"frames":frames,"frame_start":at.is_none().then_some(first_frame),"still_time":at,"resolution":[w,h],"png_write":!no_write && !gpu_only && !directory_paths.is_empty(),"video_write":video,"output":out,"renderer_pixel_readback":readback.any() && (!gpu_only && !directory_paths.is_empty()),"encoder_pixel_readback":encoder_readback,"encoder_open_ms":encoder_prepare_ms+encoder_attach_ms,"encoder_prepare_ms":encoder_prepare_ms,"encoder_attach_ms":encoder_attach_ms,"encoder_finish_ms":encoder_finish_ms,"scene_parse_ms":scene_parse_ms,"pipeline_ms":installed.pipeline_ms,"pipeline_creation_ms":pipeline_creation_ms,"environment_install_ms":installed.environment_install_ms,"material_bind_ms":installed.material_bind_ms,"asset_decode_ms":installed.asset_decode_ms,"texture_upload_ms":installed.texture_upload_ms,"texture_cache_hits":installed.texture_cache_hits,"first_frame_ms":first_frame_ms,"device_ms":if device_reused { 0. } else { renderer.startup.as_secs_f64()*1000. },"device_stages_ms":if device_reused { serde_json::json!([]) } else { serde_json::json!(renderer.startup_stages_ms) },"geometry_vertex_bytes":installed.geometry_vertex_bytes,"install_ms":installed.install_time.as_secs_f64()*1000.,"ready_ms":ready_ms,"loop_wall_ms":(stages_end-loop_start).as_secs_f64()*1000.,"wall_ms":(stages_end-wall).as_secs_f64()*1000.,"frames_in_flight":in_flight,"transient_bytes":installed.transient_bytes(),"gpu_timing_readback":true,"png_workers":png_workers,"samples":samples});
        let (stages, pipeline_cache_saves) = crate::timing::stages(&renderer.device, wall, stages_end, crate::timing::StageTotals {
            setup_ms: ready_ms, first_frame_ms, frames_ms: (stages_end - loop_start).as_secs_f64() * 1000.,
            gpu_spans_ms: gpu_spans_total, denoise_busy_ms, denoise_wait_ms,
        });
        report["stages"] = stages;
        report["pipeline_cache_saves"] = serde_json::to_value(pipeline_cache_saves)?;
        report["pipeline_cache_save_pending"] = renderer.compiler.save_pending().into();
        // Background feature variants that initially drew with a superset fallback.
        report["render_settings"] = render_settings.clone();
        report["shader_creation"] = serde_json::to_value(crate::timing::creations(&renderer.device).into_iter().skip(shader_creation_start).collect::<Vec<_>>())?;
        report["device_reused"] = device_reused.into();
        report["install_reused"] = install_reused.into();
        if install_reused {
            for key in ["install_ms", "pipeline_ms", "environment_install_ms", "material_bind_ms", "asset_decode_ms", "texture_upload_ms"] { report[key] = 0.into(); }
        }
        report["frame_redos"] = redos.into();
        report["frame_redo_ms"] = redo_ms.into();
        report["deferred_variants"] = installed.deferred_variants().into();
        // Path transport's first picture: the prebuilt engine, or the scene's own
        // kernel compiled first, with what in the scene the engine cannot draw.
        if let Some(path) = &installed.path {
            report["path_engine"] = match path.engine() {
                Ok(()) => serde_json::json!({"draws_scene": true}),
                Err(reason) => serde_json::json!({"draws_scene": false, "reason": reason}),
            };
        }
        if diagnostics {
            report["memory_frames"] = memory_frames.into();
            report["retained_gpu_bytes"] = renderer.residency.retained_bytes().into();
            report["camera_memory_bytes"] = renderer.camera_memory.into();
            report["camera_record_budget_bytes"] =
                (renderer.camera_memory / constants::CAMERA_RECORD_MEMORY_SHARE).into();
            report["camera_memory_override_lowered"] = renderer.camera_memory_override_lowered.into();
            report["probe_field"] = installed.probes.as_ref().map(|p| p.diagnostics()).into();
            report["compiled_pipeline_count"] = renderer.pipeline_timings().len().into();
            report["program_pipeline_count"] = renderer.program_pipelines.lock().unwrap().len().into();
        }
        report["invalid_pixel_count"] = invalid_pixel_count.into();
        if let Some((frame, view, pixel)) = first_invalid_pixel {
            let diagnostic = out.first().filter(|_| output_run.is_some()).map(|path| {
                let path = if is_video(path) || is_file(path) { sidecar(path, "meta.json") } else { path.join("meta.json") };
                format!("{} frames[].invalid_pixels", path.to_string_lossy().escape_debug())
            }).unwrap_or_else(|| "render report samples[].invalid_pixels".into());
            let witness = format!("first frame {frame} {} ({},{}); see {diagnostic}", view.escape_debug(), pixel[0], pixel[1]);
            if let Some(warning) = invalid_samples_warning(invalid_pixel_count, &witness) {
                crate::warnings::dropped("nonfinite_samples", serde_json::json!({
                    "invalid_pixel_count":invalid_pixel_count,"first":{"frame":frame,"view":view,"pixel":pixel}
                }), warning);
            }
        }
        // The profile asked for intent; `settings` records the techniques this
        // device actually ran (automatic choices resolved at install).
        let metadata = |frames: &[serde_json::Value]| serde_json::json!({
            "version": 3, "fps": fps, "frames": frames,
            "object_ids": &*installed.object_ids,
            "warnings": warnings.snapshot(),
            "profile": profile_record, "settings": installed.render().settings,
            "render_settings": render_settings,
            "denoiser": installed.render().settings.integrator.denoiser().and_then(|_| crate::denoise::device_kind(&renderer)),
        });
        let mut exposures = Vec::new();
        let mut pending_metadata = Vec::new();
        for path in out.iter().filter(|_| output_run.is_some()) {
            let file_path = |suffix: &str| {
                if is_video(path) || is_file(path) { sidecar(path, suffix) }
                else { path.join(suffix) }
            };
            let mut frames = if !is_video(path) && still_pass {
                vec![serde_json::json!({"index":still_index.unwrap(),"time":at.unwrap(),"camera":installed.camera_name_at_request(&still_request())?,
                    "depth_unit_m":still_camera.as_ref().unwrap().depth_unit_m,"terminated_medium_paths":still_terminated_medium_paths,"discarded_medium_radiance":still_discarded_medium_radiance,"dropped_path_samples":still_dropped_path_samples,"dropped_light_samples":still_dropped_light_samples,"invalid_pixel_count":still_invalid_count,"invalid_pixels":&still_invalid_pixels,"exposure":still_exposure.map(|e: crate::Exposure|e.json())})]
            } else { metadata_frames.clone() };
            let still_rows: Vec<_> = still_exposure.map(|e| (0, e)).into_iter().collect();
            let mut rows = if !is_video(path) && still_pass { &still_rows[..] } else { &frame_exposures[..] };
            // Only a destination that writes the display image gets its report: video, or PNG
            // colour. EXR holds scene-linear values; control-only directories hold no colour.
            // The report reads SDR code values (clipping at one, luma); PQ and HLG signals are not those.
            if target.display.is_hdr() || !(is_video(path) || (!is_exr(path) && outputs.has(Output::Color))) {
                rows = &[];
                for frame in &mut frames { frame["exposure"] = serde_json::Value::Null; }
            }
            let meta_path = file_path("meta.json");
            if let Some(parent) = meta_path.parent() { std::fs::create_dir_all(parent)?; }
            let mut meta = metadata(&frames);
            // Every viewed picture's sidecar names the display it is encoded for, sRGB included, as its PNG's
            // colour chunk or its video stream does.
            if is_video(path) || (!is_exr(path) && outputs.has(Output::Color)) {
                meta["output_display"] = serde_json::to_value(target.display)?;
                if !is_video(path) { meta["output_png_depth"] = if png_depth == crate::PngDepth::Sixteen { 16 } else { 8 }.into(); }
                if target.display.is_hdr() {
                    meta["output_peak_luminance"] = target.peak.into();
                    meta["output_reference_white"] = target.white.into();
                }
            }
            if provisional { meta["provisional"] = true.into(); }
            if let Some(summary) = exposure_summary(rows) {
                meta["exposure"] = summary.clone();
                exposures.push(serde_json::json!({"output": path, "exposure": summary}));
            }
            pending_metadata.push((meta_path.clone(), serde_json::to_vec_pretty(&meta)?));
            sidecars.push(meta_path);
            // Every written render reports where its time went.
            let timing_path = file_path("timings.json");
            std::fs::write(&timing_path, serde_json::to_vec_pretty(&report)?)?;
            sidecars.push(timing_path);
        }
        report["interrupted"] = interrupted.into();
        report["warnings"] = serde_json::to_value(warnings.snapshot())?;
        report["sidecars"] = serde_json::to_value(sidecars)?;
        report["exposure"] = exposures.into();
        if let Some(run) = &output_run { run.complete(&pending_metadata)?; }
        Ok(RenderedShot {
            report,
            _encoders: encoders,
            _slots: slots,
            _scene: Some(installed),
            _canvas: None,
            _renderer: Some(renderer),
            delivery,
        })
    }
}

/// Everything below the document that an installation depends on, and how to
/// install it: the delivery identity (readback shape, single frame, GPU colour,
/// output set, the display encoded for, the resolved render block) and the install options. One place for
/// the shot and for the kept-alive worker preparing a candidate, so the two
/// cannot drift.
pub(crate) struct InstallPlan {
    pub(crate) delivery: String,
    pub(crate) options: crate::InstallOptions,
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn install_plan(renderer: &Renderer, scene: &mut vork_scene::Scene, default_denoise: Option<&(String, u8, bool)>,
    install_readback: crate::ReadbackOutputs, single_frame: bool, gpu_color: bool, outputs: &[Output], initial_time: vork_time::Request,
    display: crate::output_device::Target) -> Result<InstallPlan> {
    if let Some((profile, level, keep_samples)) = default_denoise {
        crate::denoise::resolve_default(renderer, scene, profile, *level, *keep_samples);
    }
    let delivery = format!("{install_readback:?}|{single_frame}|{gpu_color}|{outputs:?}|{display:?}|{}|{}", serde_json::to_string(&scene.render)?,
        serde_json::to_string(scene.compositor())?);
    Ok(InstallPlan { delivery, options: crate::InstallOptions {
        initial_time,
        readback: install_readback,
        gpu_color,
        display,
        variants: if single_frame { crate::pipelines::VariantCompile::Fallback } else { Default::default() },
        ..Default::default()
    } })
}

/// Load an in-memory authored document the way `load_with_settings` loads a file:
/// std expansion, authored locations for diagnostics, settings, the caller's
/// rewrite, then renderer-support validation.
fn load_document_with_settings(bytes: &[u8], path: &Path, settings: vork_scene::Settings, profile: &str, edit: Option<SceneEdit>, decoded: Option<Scene>) -> Result<vork_scene::Scene> {
    let base = path.parent().unwrap_or(Path::new("."));
    let mut scene = (|| -> Result<vork_scene::Scene> {
        let scene = match decoded { Some(scene) => scene, None => crate::decode_scene_document(bytes, path)? };
        scene.validate().map_err(|error| scene.authored_error(error))?;
        Ok(scene)
    })().with_context(|| format!("load scene {}", path.display()))?;
    scene.apply_settings(settings, profile)
        .with_context(|| format!("load scene {} with profile {profile}", path.display()))?;
    // The same source admission as `load_with_settings`: typed scene and source
    // problems (missing or irregular files included) before any device work.
    crate::validate_inputs(&scene, base)
        .with_context(|| format!("load scene {} with profile {profile}", path.display()))?;
    if let Some(edit) = edit {
        edit(&mut scene, base).with_context(|| format!("edit scene {}", path.display()))?;
        crate::validate_inputs(&scene, base)?;
    }
    Ok(scene)
}

fn invalid_samples_warning(count: u64, diagnostic: &str) -> Option<String> {
    (count != 0).then(|| format!("{count} pixel{} had non-finite camera/light samples; marked magenta; {diagnostic}", if count == 1 { "" } else { "s" }))
}

/// An output's exposure over its frames: mean shares and luma, and the worst frame for
/// clipping and near-black. None when no frame wrote colour. Clipping is null for a picture
/// no view made (`Exposure::clipped`); one output's frames are all of one kind.
pub(crate) fn exposure_summary(rows: &[(usize, crate::Exposure)]) -> Option<serde_json::Value> {
    if rows.is_empty() { return None; }
    let coverage = rows.iter().map(|(_, e)| e.coverage).sum::<f64>() / rows.len() as f64;
    let frames = rows.len();
    let rows: Vec<_> = rows.iter().filter(|(_, e)| e.coverage > 0.).copied().collect();
    if rows.is_empty() {
        return Some(serde_json::json!({"frames": frames, "coverage": 0., "clipped": null, "near_black": null, "mean": null, "worst_clipped": null, "worst_near_black": null}));
    }
    let n = rows.len() as f64;
    let mean = |f: fn(&crate::Exposure) -> f64| rows.iter().map(|(_, e)| f(e)).sum::<f64>() / n;
    let worst = |f: fn(&crate::Exposure) -> f64| {
        let (frame, e) = rows.iter().max_by(|a, b| f(&a.1).total_cmp(&f(&b.1))).unwrap();
        serde_json::json!({"share": f(e), "frame": frame})
    };
    let viewed = rows.iter().all(|(_, e)| e.clipped.is_some());
    let clipped = viewed.then(|| mean(|e| e.clipped.unwrap_or(f64::NAN)));
    let worst_clipped = viewed.then(|| worst(|e| e.clipped.unwrap_or(f64::NAN)));
    Some(serde_json::json!({
        "frames": frames, "coverage": coverage,
        "clipped": clipped, "near_black": mean(|e| e.near_black), "mean": mean(|e| e.mean),
        "worst_clipped": worst_clipped, "worst_near_black": worst(|e| e.near_black),
    }))
}

#[cfg(test)]
mod refusal_tests {
    use super::*;
    use crate::failure::exit_code;
    #[test]
    fn a_sequence_skips_refused_frames_and_reports_them_with_their_category() {
        let scene = |message: &str| scene_error(anyhow::Error::new(crate::FrameRefused(message.into())));
        let mut refused = Refusals::default();
        refused.finish(8).unwrap();
        // Only a refusal is skipped: any other error stands.
        assert!(!refused.skip(0, &device(anyhow::anyhow!("GPU device lost"))));
        assert!(!refused.skip(0, &scene_error(anyhow::anyhow!("view returned code values for another display"))));
        refused.finish(8).unwrap();
        // The scene's refusals, through a frame boundary as the loops see them: the summary is the scene's.
        assert!(refused.skip(2, &device(scene("view returned a non-finite colour"))));
        assert!(refused.skip(5, &scene("view returned a non-finite colour").context("frame 5")));
        assert_eq!(refused.count(), 2);
        let error = refused.finish(8).unwrap_err();
        assert_eq!(exit_code(&error), 3);
        assert_eq!(error.to_string(), "2 of 8 frames refused and not written (view returned a non-finite colour): 2, 5; the other frames were written, without meta.json");
        // One refusal that is not the scene's makes the summary a device failure.
        assert!(refused.skip(6, &device(anyhow::Error::new(crate::FrameRefused("non-finite scene-linear radiance; frame refused".into())))));
        assert_eq!(exit_code(&refused.finish(8).unwrap_err()), 4);
    }
}
#[cfg(test)]
mod sidecar_tests {
    use super::*;

    #[test]
    #[ignore = "requires a supported GPU, pinned Slang and ffmpeg; scheduled explicitly in main CI"]
    fn cancellation_closes_a_partial_sequence_and_records_only_saved_frames() -> Result<()> {
        crate::test_renderer()?;
        let root = std::env::temp_dir().join(format!("vork-cancel-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let scene = root.join("scene.json");
        std::fs::write(&scene, vork_scene::STARTER_SCENE)?;
        let destination = root.join("frames");
        let rendered = Shot {
            stop: Some(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true))),
            scene, out: vec![destination.clone().into(), root.join("clip.mov").into()], resolution: Some([32, 24]),
            frame_start: Some(0), frame_end: Some(3),
            settings: [("samples".into(), serde_json::json!(1)), ("denoise".into(), serde_json::json!(false))].into_iter().collect(),
            ..Default::default()
        }.render()?;
        assert_eq!(rendered.report["interrupted"], true);
        assert_eq!(rendered.report["frames"], 1);
        let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(destination.join("meta.json"))?)?;
        assert_eq!(metadata["frames"].as_array().unwrap().len(), 1);
        assert_eq!(metadata["render_settings"], rendered.report["render_settings"]);
        let settings = &rendered.report["render_settings"];
        assert_eq!(settings["samples"], 1);
        assert_eq!(settings["denoise"], false);
        assert_eq!(settings["resolution"], serde_json::json!([32,24]));
        assert_eq!(settings["frames"], 4); // Requested budget, including on interruption.
        assert_eq!(settings["profile"], metadata["profile"]);

        let probe = std::process::Command::new("ffprobe").args([
            "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
            "-of", "default=noprint_wrappers=1:nokey=1",
        ]).arg(root.join("clip.mov")).output()?;
        assert!(probe.status.success(), "{}", String::from_utf8_lossy(&probe.stderr));
        assert_eq!(String::from_utf8_lossy(&probe.stdout).trim(), "1");
        // Match the CLI ownership contract: files are finished before GPU owners
        // are discarded by process exit. Dropping here would also wait for optional
        // pipeline preparation, which is separate from cancellation/file closure.
        std::mem::forget(rendered);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn invalid_sample_warning_is_one_line_with_count_and_pixel_diagnostic() {
        let diagnostic = "frame.png.meta.json frames[].invalid_pixels";
        assert_eq!(invalid_samples_warning(0, diagnostic), None);
        let warning = invalid_samples_warning(4294967299, diagnostic).unwrap();
        assert_eq!(warning.lines().count(), 1);
        assert!(warning.contains("4294967299 pixels"));
        assert!(warning.contains(diagnostic));
    }

    #[test]
    fn exposure_summary_excludes_empty_frames() {
        let empty = crate::Exposure { coverage: 0., clipped: Some(f64::NAN), near_black: f64::NAN, mean: f64::NAN };
        let covered = crate::Exposure { coverage: 0.5, clipped: Some(0.2), near_black: 0.1, mean: 0.6 };
        let e = exposure_summary(&[(1, empty), (2, covered)]).unwrap();
        assert_eq!(e["frames"], 2);
        assert_eq!(e["coverage"], 0.25);
        assert_eq!(e["mean"], 0.6);
        assert_eq!(e["worst_clipped"]["frame"], 2);
        assert!(exposure_summary(&[(1, empty)]).unwrap()["mean"].is_null());
        assert!(exposure_summary(&[]).is_none());
        // A picture no view made reports no clipping; its other statistics stand.
        let authored = exposure_summary(&[(3, crate::Exposure { clipped: None, ..covered })]).unwrap();
        assert!(authored["clipped"].is_null() && authored["worst_clipped"].is_null());
        assert_eq!((&authored["near_black"], &authored["worst_near_black"]["frame"]), (&serde_json::json!(0.1), &serde_json::json!(3)));
    }

    #[test]
    fn frame_patterns_only_match_filename_placeholders() {
        for path in ["renders/review#2/still.png", "renders/100%/still.png",
            "renders/####/still.png", "renders/%04d/still.png", "part#1.png",
            "100%.png", "frame_%%04d.png"] {
            assert!(!has_frame_pattern(Path::new(path)), "{path}");
        }
        for path in ["frame_####.png", "frame_%04d.png", "frame_##.png", "frame_%d.png"] {
            assert!(has_frame_pattern(Path::new(path)), "{path}");
        }
    }

    #[test]
    fn output_formats_have_distinct_sidecars() {
        for suffix in ["camera.json", "meta.json", "timings.json"] {
            let mp4 = sidecar(Path::new("out/x.mp4"), suffix);
            let mov = sidecar(Path::new("out/x.mov"), suffix);
            let exr = sidecar(Path::new("out/x.%06d.exr"), suffix);
            assert_eq!(mp4, PathBuf::from(format!("out/x.mp4.{suffix}")));
            assert_eq!(mov, PathBuf::from(format!("out/x.mov.{suffix}")));
            assert_eq!(exr, PathBuf::from(format!("out/x.exr.{suffix}")));
            assert_eq!(sidecar(Path::new("exr/%06d.exr"), suffix), PathBuf::from(format!("exr/{suffix}")));
            assert_eq!(sidecar(Path::new("shot_%06d.exr"), suffix), PathBuf::from(format!("shot.exr.{suffix}")));
            assert_eq!(sidecar(Path::new("out/still.png"), suffix), PathBuf::from(format!("out/still.png.{suffix}")));
            assert_ne!(mp4, mov);
        }
    }

    #[test]
    fn unknown_extensions_are_not_directories() {
        for ok in ["out", "out/frames", "out/still.PNG", "x.%06d.exr", "a.mp4", "a.MOV", "frames.v2/", ".", "..", "out/.hidden"] {
            assert!(recognized_destination(Path::new(ok)), "{ok}");
        }
        for bad in ["still.jpg", "frames.v2", "clip.mp5", "out/x.webp", "shot.json"] {
            assert!(!recognized_destination(Path::new(bad)), "{bad}");
        }
        assert!(directory_marked(Path::new("frames.png/")) && !directory_marked(Path::new("frames.png")));
        for directory in ["frames.mp4/", "frames.MOV/", "frames"] {
            assert!(!video_destination(Path::new(directory)), "{directory}");
        }
        assert!(video_destination(Path::new("clip.MP4")) && video_destination(Path::new("clip.mov")));
    }
}

#[cfg(test)]
mod channel_tests {
    use super::*;
    #[test]
    fn default_is_color_and_all_includes_declared_aovs() {
        let aovs = vec![
            vork_scene::v2::Aov { name: "heat".into(), kind: vork_scene::v2::AovType::Value },
            vork_scene::v2::Aov { name: "tint".into(), kind: vork_scene::v2::AovType::Color },
        ];
        let mut expected: Vec<Output> = Output::builtins().collect();
        assert_eq!(resolve_outputs(Shot::default().outputs, &aovs), vec![Output::Color]);
        expected.extend([Output::Aov("heat".into()), Output::Aov("tint".into())]);
        assert_eq!(resolve_outputs(Vec::new(), &aovs), expected);
        for explicit in [vec![Output::Color], vec![Output::Aov("heat".into())]] {
            assert_eq!(resolve_outputs(explicit.clone(), &aovs), explicit);
        }
    }
}

/// Admission of the finite output request set, before any replay or exposure.
/// Simulation and shutter consumers need this traversal; ordinary animation
/// does not get an extra evaluation. Each shutter envelope is derived independently from its
/// nominal integer identity and existing numeric shutter controls.
fn validate_clock_requests(scene:&InstalledScene,requests:impl Iterator<Item=vork_time::Request>)->Result<()> {
    let compute=scene.compute.as_ref().filter(|compute|compute.has_simulation());
    let animated_shutter=scene.animation.has_shutter_animation();
    if compute.is_none() && !animated_shutter && scene.render.motion_blur.as_ref().is_none_or(|m| m.shutter == 0.) {return Ok(())}
    use vork_time::exact::{Germ,decimal,ieee};
    let mut samples=scene.animation.samples();
    let rate=decimal(&scene.render.fps.to_string())?;
    for request in requests {
        request.numeric()?;
        if let Some(compute)=compute {compute.validate_request(&request)?;}
        let nominal=request.exact()?;
        let previous=request.previous(scene.render.fps)?.exact()?;
        let cut=scene.animation.clamp_camera_request(&nominal,&previous,scene.camera)?!=previous;
        if animated_shutter {scene.animation.sample_shutter(request.numeric()?,&mut samples,&scene.render);}
        let shutter=if animated_shutter {samples.render.as_ref().unwrap().motion_blur.as_ref()}else{scene.render.motion_blur.as_ref()};
        let span=ieee(if cut {0.}else{shutter.map_or(0.,|s|s.shutter)})?/&rate;
        anyhow::ensure!(span>=Default::default(),"animated shutter must be nonnegative");
        let offset=ieee(shutter.map_or(0.,|s|s.position.offset()))?;
        for position in [decimal("-1/2")?,decimal("1/2")?] {
            let endpoint=nominal.add(&Germ::point((&position+&offset)*&span));
            let endpoint=if endpoint.compare_point(&Default::default()).is_lt() {Germ::point(Default::default())}else{endpoint};
            let endpoint=scene.animation.clamp_camera_request(&nominal,&endpoint,scene.camera)?;
            let endpoint=vork_time::Request::Exact(endpoint);
            endpoint.numeric()?;
            if let Some(compute)=compute {compute.validate_request(&endpoint)?;}
        }
    }
    Ok(())
}

/// Check every nominal frame and its shutter endpoints using render's clock admission.
/// The installed scene's clocks over its whole clip: CPU arithmetic on authored times.
pub(crate) fn check_timeline(scene: &InstalledScene) -> Result<()> {
    let frames = clip_frame_count(&scene.render)?;
    validate_clock_requests(scene, (0..frames).map(|index| vork_time::Request::Frame(index as u32, scene.render.fps))).map_err(scene_error)
}

pub(crate) fn clip_frame_count(render: &vork_scene::Render) -> Result<usize> {
    let count = render.duration.exact() * vork_time::exact::decimal(&render.fps.to_string()).map_err(scene_error)?;
    let frames: usize = count.ceil().to_integer().try_into().context("output frame count exceeds usize").map_err(scene_error)?;
    ensure_usage!(frames > 0 && frames <= u32::MAX as usize, "output frame count exceeds u32");
    Ok(frames)
}
