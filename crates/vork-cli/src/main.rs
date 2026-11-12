//! Unstable, internal: all Rust APIs in this crate are free to change.

// All command output is fallible; the process boundary treats a closed consumer as success.
macro_rules! output {
    ($($arg:tt)*) => { crate::output::line(format_args!($($arg)*))? };
}
mod output;
mod update_notice;

mod constants;
mod init;
mod gpu_diagnostics;
mod progress;
mod interrupt;
mod setup;
mod help;
mod std_command;
use anyhow::Result;
use clap::{Parser, Subcommand};
use std::{io::Write, path::PathBuf};
use vork_render::shot::{Output, Shot};
#[derive(Parser)]
#[command(
    name = "vork",
    bin_name = "vork",
    about = "Retained GPU renderer for declarative 3D shots",
    version = env!("VORK_VERSION"),
    disable_help_subcommand = true,
    help_template = "{after-help}",
    after_help = root_help()
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Create a small offline project ready to render.
    Init(init::Args),
    /// Warm the init starter at the default profile in this machine's driver cache.
    Setup {
        /// Also warm every bundled program/profile and built-in descriptor.
        #[arg(long)]
        all: bool,
        /// Check tools without warming shaders; renders compile on demand.
        #[arg(long, conflicts_with = "all")]
        skip_warmup: bool,
        /// Include raw encoder probe diagnostics.
        #[arg(long)]
        verbose: bool,
        #[arg(long, hide = true)]
        warmup_worker: bool,
        /// Hardware GPU index in PCI bus order, or cpu for software rendering; default the high-performance GPU. Also VORK_GPU.
        #[arg(long, value_name = "N|cpu")]
        gpu: Option<vork_render::GpuSelection>,
    },
    /// Compile portable release shaders without opening a GPU driver.
    #[command(hide = true)]
    PrepareReleaseShaders {
        scenes: Vec<PathBuf>,
    },
    /// Standard-library JSON fragments.
    Std(std_command::Args),
    /// Topic reference.
    Help {
        topic: Option<String>,
        /// Exact generated field path, when topic is field.
        field: Option<String>,
        /// Include generated fields and complete stage declarations.
        #[arg(long)]
        all: bool,
    },
    /// Validate scene installation without opening a GPU or rendering.
    Check {
        /// Scene JSON path, or - to read stdin (paths relative to the current directory).
        scene: PathBuf,
        /// Also generate native portable shaders; CPU-only, but can take minutes.
        #[arg(long)]
        shaders: bool,
        /// Render settings profile; default std/final.
        #[arg(long)]
        profile: Option<String>,
    },
    /// Export evaluated triangles and optional native animation to glTF 2.0.
    Export {
        scene: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        root: Option<String>,
        #[arg(long)]
        at: Option<vork_time::Clock>,
        /// Inclusive source frame range; omitted endpoint uses the scene boundary.
        #[arg(long, conflicts_with = "at")]
        frame_start: Option<u32>,
        #[arg(long, conflicts_with = "at")]
        frame_end: Option<u32>,
        /// Acknowledge one named export loss; repeat for multiple issue codes.
        #[arg(long = "allow-loss")]
        allow_losses: Vec<String>,
    },
    /// Render scene JSON to MP4, MOV, PNG/control directories, or multilayer EXR frames.
    Render {
        /// Scene JSON path.
        scene: PathBuf,
        /// Comma-separated channels; default color; all selects controls and every declared AOV.
        #[arg(
            long = "channels",
            value_name = "CHANNELS",
            value_delimiter = ',',
            default_value = "color",
            value_parser = channel
        )]
        outputs: Vec<String>,
        /// MP4 (H.264), MOV (ProRes 4444), one still .png, PNG/control directory, or %06d.exr multilayer frames; repeat for simultaneous outputs.
        #[arg(long, required = true)]
        out: Vec<PathBuf>,
        /// Blender ImageFormatSettings.exr_codec; default zip.
        #[arg(long, default_value = "zip")]
        exr_codec: vork_render::ExrCodec,
        /// PNG colour bits per component; default 8.
        #[arg(long, default_value = "8", value_name = "8|16")]
        png_depth: vork_render::PngDepth,
        /// Select the still time in seconds; with video, directory/EXR outputs receive one still.
        #[arg(long, conflicts_with_all = ["frame_start", "frame_end"])]
        at: Option<vork_time::Clock>,
        /// First frame to render (absolute index, inclusive; default the clip's first frame, 0).
        #[arg(long)]
        frame_start: Option<usize>,
        /// Last frame to render (absolute index, inclusive; default the clip's last frame). Output frame numbers stay absolute; video contains just the range.
        #[arg(long)]
        frame_end: Option<usize>,
        /// Render settings profile: a library profile (std/previs, std/draft, std/final, std/path) or a profile file; default std/final.
        #[arg(long, value_name = "LIB/NAME|FILE")]
        profile: Option<String>,
        /// Render-call fields: resolution (WxH), fps, seed, samples, denoise. Repeatable KEY=VALUE.
        #[arg(long = "set", value_name = "KEY=VALUE", value_parser = setting, help = format!("Render-call fields: {}. Repeatable KEY=VALUE; values are JSON, resolution also accepts WxH or W,H.", vork_scene::CallOverrides::KEYS.join(", ")))]
        set: Vec<(String, serde_json::Value)>,
        /// Skip PNG compression/writes; retain GPU rendering and mapped readback.
        #[arg(long, hide = true, requires = "diagnostics")]
        no_write: bool,
        /// Keep pixels on GPU; no PNG output or pixel readback.
        #[arg(long, hide = true, requires = "diagnostics")]
        gpu_only: bool,
        /// Adapter/cache/GPU timings; default false.
        #[arg(long)]
        diagnostics: bool,
        /// Hardware GPU index in PCI bus order, or cpu for software rendering; default the high-performance GPU. Also VORK_GPU. On a multi-GPU machine,
        /// run one render per GPU on separate --frame-start/--frame-end ranges.
        #[arg(long, value_name = "N|cpu")]
        gpu: Option<vork_render::GpuSelection>,
    },
}
fn channel(value: &str) -> Result<String, String> {
    if value != "all" { value.parse::<Output>().map_err(|e| e.to_string())?; }
    Ok(value.to_owned())
}
fn main() -> std::process::ExitCode {
    if update_notice::worker_entry() { return std::process::ExitCode::SUCCESS; }
    let result = run();
    update_notice::after_command();
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) if output::broken_pipe(&error) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error:#}");
            if vork_render::failure::suggests_setup(&error) { eprintln!("Run `vork setup` for this machine’s GPU and tools report."); }
            std::process::ExitCode::from(vork_render::failure::exit_code(&error))
        }
    }
}
fn run() -> Result<()> {
    let entry = std::time::SystemTime::now();
    let start = std::time::Instant::now();
    let Args { command } = match Args::try_parse() {
        Ok(args) => args,
        Err(error) => {
            let code = error.exit_code();
            if error.use_stderr() { error.print()?; }
            else { error.print().map_err(output::error)?; }
            std::process::exit(code);
        }
    };
    if matches!(command, Command::Render { diagnostics: true, .. }) {
        eprintln!(
            "cli_entry_unix_ns: {}; cli_parse_ms: {:.3}",
            entry.duration_since(std::time::UNIX_EPOCH)?.as_nanos(),
            start.elapsed().as_secs_f64() * 1000.
        );
    }
    if let Command::Setup { gpu: Some(index), .. } | Command::Render { gpu: Some(index), .. } = &command {
        vork_render::select_gpu(*index);
    }
    if let Command::PrepareReleaseShaders { scenes } = &command {
        return vork_render::Renderer::prepare_release_shaders(scenes);
    }
    if let Command::Setup { all, skip_warmup, verbose, warmup_worker, gpu } = &command {
        // Supervise required checks even when warming is skipped. Only time
        // spent warming shaders may expire successfully.
        if !warmup_worker { return setup::bounded(*all, *skip_warmup, *verbose, *gpu); }
        output!("vork {}", env!("VORK_VERSION"));
        output!("OS: {} ({})", std::env::consts::OS, std::env::consts::ARCH);
        setup::stage("checking FFmpeg")?;
        match std::process::Command::new("ffmpeg").arg("-version").output() {
            Ok(output) if output.status.success() => {
                if let Some(version) = String::from_utf8_lossy(&output.stdout).lines().next() {
                    output!("FFmpeg version: {version}");
                }
            }
            _ => output!("FFmpeg version: unavailable"),
        }
        let progress = progress::Monitor::start("Setup: opening GPU and checking tools");
        setup::stage("opening GPU")?;
        let renderer=vork_render::Renderer::new().map_err(|error| {
            // Prefer concrete loader failures to the renderer's generic container hint.
            let error = match gpu_diagnostics::report() {
                Some(problems) => anyhow::anyhow!("GPU initialization failed. {problems}"),
                None => error,
            };
            eprintln!("GPU: {error:#}");
            eprintln!("Rendering requires ray queries and storage buffer arrays; see the platform-specific device diagnostic above. A container needs host graphics-driver libraries and device access; Docker does not provide a GPU on its own. On your own NVIDIA or AMD Linux machine, use the Docker runtime: docs/REMOTE-GPU.md. Scene authoring and `vork help` work without a GPU.");
            error.context(vork_render::failure::FailureKind::Device)
        })?;
        let device_s = start.elapsed().as_secs_f64();
        let tools_start = std::time::Instant::now();
        setup::stage("checking tools")?;
        renderer.setup_report_verbose(&mut std::io::stdout().lock(), *verbose).map_err(|error| {
            if error.downcast_ref::<std::io::Error>().is_some() { output::context(error) }
            else { error.context(vork_render::failure::FailureKind::Setup) }
        })?;
        let tools_s = tools_start.elapsed().as_secs_f64();
        if *skip_warmup {
            progress.finish();
            output!("Shader warmup skipped; rendering will compile pipelines on demand.");
            return Ok(());
        }
        setup::stage("warming shaders")?;
        let warm_start = std::time::Instant::now();
        vork_slang::progress::phase("Setup: warming shaders and driver pipelines", 0);
        if *all { renderer.warm_shaders()?; } else { renderer.warm_startup_shaders()?; }
        progress.finish();
        output!("Setup phases: device {:.3} s; tools {:.3} s; shaders/cache {:.3} s", device_s, tools_s, warm_start.elapsed().as_secs_f64());
        output!("Shader setup completed in {:.3} s",start.elapsed().as_secs_f64());
        return Ok(());
    }
    if let Command::Help { topic, field, all } = &command {
        return help::run(topic.as_deref(), field.as_deref(), *all);
    }
    if let Command::Init(args) = &command {
        return args.run();
    }
    if let Command::Std(args) = &command {
        return args.run();
    }
    if let Command::Check { scene, profile, shaders } = &command {
        let profile = vork_scene::Profile::load(profile.as_deref().unwrap_or(vork_scene::DEFAULT_PROFILE))
            .map_err(|error| error.context(vork_render::failure::FailureKind::Usage))?;
        vork_render::Renderer::check(scene, &profile, *shaders)?;
        output!("Scene checked successfully");
        return Ok(());
    }
    if let Command::Export { scene, out, root, at, frame_start, frame_end, allow_losses } = &command {
        let report = vork_render::export::Export {scene:scene.clone(),out:out.clone(),root:root.clone(),at:at.clone(),frame_start:*frame_start,frame_end:*frame_end,allow_losses:allow_losses.clone(),vork_version:env!("VORK_VERSION").to_owned()}.run()?;
        output!("Exported {} meshes to {}; report: {}", report["counts"]["meshes"], out.display(), out.with_extension("export.json").display());
        for issue in report["issues"].as_array().into_iter().flatten() {eprintln!("WARNING {} at {}: {}; {}",issue["code"],issue["source"],issue["dropped"],issue["remediation"]);}
        return Ok(());
    }
    let progress = progress::Monitor::start("Render: loading scene and opening GPU");
    let diagnostics = matches!(command, Command::Render { diagnostics: true, .. });
    let (rendered, out) = match command {
        Command::Render {
            scene: scene_path, outputs, out, exr_codec, png_depth, at, frame_start, frame_end,
            profile, set, no_write, gpu_only, diagnostics, ..
        } => {
            let stop = interrupt::install()?;
            let rendered = Shot {
                stop,
                scene: scene_path,
                outputs: if outputs == ["all"] { Vec::new() } else { outputs.iter().map(|s| s.parse()).collect::<Result<Vec<Output>, _>>().map_err(|e| anyhow::anyhow!(e).context(vork_render::failure::FailureKind::Usage))? },
                out: out.clone(),
                resolution: None,
                profile: Some(vork_scene::Profile::load(profile.as_deref().unwrap_or(vork_scene::DEFAULT_PROFILE))
                    .map_err(|error| error.context(vork_render::failure::FailureKind::Usage))?),
                settings: set.into_iter().collect(),
                at, frame_start, frame_end, no_write, gpu_only, diagnostics, exr_codec, png_depth,
                edit: None,
                ..Default::default()
            }.render()?;
            (rendered, out)
        }
        _ => unreachable!(),
    };
    if diagnostics {
        eprintln!(
            "cli_render_return_ms: {:.3}",
            start.elapsed().as_secs_f64() * 1000.
        );
    }
    progress.finish();
    let report = &rendered.report;
    output!(
        "Rendered {} frames in {:.3}s; outputs: {}",
        report["frames"],
        report["wall_ms"].as_f64().unwrap() / 1000.,
        out.iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if let Some(sidecars) = report["sidecars"].as_array().filter(|paths| !paths.is_empty()) {
        output!("Sidecars: {}", sidecars.iter().filter_map(|p| p.as_str()).collect::<Vec<_>>().join(", "));
    }
    let exposures = report["exposure"].as_array().map_or(&[][..], Vec::as_slice);
    for row in exposures {
        let line = exposure_line(&row["exposure"]);
        if exposures.len() == 1 { output!("{line}"); } else { output!("{}: {line}", row["output"].as_str().unwrap_or_default()); }
    }
    if let Some(warnings) = report["warnings"].as_array().filter(|warnings| !warnings.is_empty()) {
        eprintln!("\nWARNING: {} RENDER WARNING(S): {}. Review the detailed warnings above and in render metadata before accepting this render.",
            warnings.len(), warnings.iter().filter_map(|warning| warning["code"].as_str()).collect::<Vec<_>>().join(", "));
    }
    if let Some(line) = interruption_line(report, &out) { eprintln!("{line}"); }
    // Shot::render has joined file writers, reaped input decoders, finished muxing, saved idle caches and
    // written reports. Keep GPU owners alive: process exit discards unused variants
    // and reclaims resources without driver teardown. Errors still unwind.
    std::io::stdout().flush().map_err(output::error)?;
    std::io::stderr().flush()?;
    rendered.exit_process()
}

fn interruption_line(report: &serde_json::Value, paths: &[PathBuf]) -> Option<String> {
    if report["interrupted"] != true { return None; }
    Some(if report["sidecars"].as_array().is_some_and(|paths| !paths.is_empty()) {
        format!("Interrupted: saved {} frame(s) to {}; output files closed cleanly. Exiting 130.", report["frames"], paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", "))
    } else {
        format!("Interrupted after {} rendered frame(s); no output files were written. Exiting 130.", report["frames"])
    })
}

/// Parse values only after resolving their writable schema field.
fn setting(text: &str) -> Result<(String, serde_json::Value), String> {
    vork_scene::CallOverrides::parse(text).map_err(|error| error.to_string())
}

/// `exposure: 12% clipped, 3% near-black, mean 0.41`; a clip adds each worst frame.
fn exposure_line(e: &serde_json::Value) -> String {
    if e["coverage"].as_f64() == Some(0.) {
        return "exposure: undefined (no covered pixels)".into();
    }
    fn percent(share: f64) -> String {
        match share * 100. {
            0. => "0%".into(),
            p if p < 0.1 => "<0.1%".into(),
            p if p < 10. => format!("{p:.1}%"),
            p => format!("{p:.0}%"),
        }
    }
    let share = |key: &str| e[key].as_f64().unwrap_or(0.);
    let clip = e["frames"].as_u64().unwrap_or(1) > 1;
    let part = |key: &str, label: &str| {
        let worst = &e[format!("worst_{key}")];
        let mut text = format!("{} {label}", percent(share(key)));
        if clip {
            text += &format!(" (worst {} at frame {})", percent(worst["share"].as_f64().unwrap_or(0.)), worst["frame"]);
        }
        text
    };
    let covered = if e["coverage"].as_f64().is_some_and(|a| a < 1.) { " of covered pixels" } else { "" };
    format!("exposure: {}, {}, mean {:.2}{covered}", part("clipped", "clipped"), part("near_black", "near-black"), share("mean"))
}

fn help_text(document: &str) -> String {
    document
        .lines()
        .filter(|line| !line.starts_with("<!--"))
        .map(|line| format!("{line}\n"))
        .collect()
}
fn root_help() -> String {
    help_text(include_str!("../../../docs/help/index.md"))
        .lines()
        .map(|line| format!("{line}\n"))
        .collect()
}
fn topic_for_line(line: &str) -> Option<&'static str> {
    // Tagged camera/light fields live directly on the node. Their discriminator,
    // rather than a fictitious object prefix, owns the focused help topic.
    let name = line.split_once(':').map_or(line, |(name, _)| name);
    if name.ends_with(" (camera)") || line.starts_with("node [type=\"camera\"]:") {
        return Some("cameras");
    }
    if name.ends_with(" (light)") || line.starts_with("node [type=\"light\"]:") {
        return Some("lights");
    }
    if let Some((library, _)) = name.split_once('/') {
        if let Some(library) = vork_scene::libraries::library(library) { return Some(library.name); }
    }
    let area = line.split(['.', ':', ' ']).next()?;
    Some(match area {
        "scene" | "names" | "paths" | "structure" | "coordinates" => "contract",
        "node" => "nodes",
        "camera" | "asset" => "cameras",
        "light" | "lights" => "lights",
        "material" | "normal" => "materials",
        "geometry" => "geometry",
        "buffer" => "buffers",
        "image" => "images",
        "program" | "simulation" => "programs",
        "animation" => "animation",
        "settings" | "profile" => "profiles",
        "render" | "colour" => "render",
        "compositor" => "compositor",
        _ => return None,
    })
}
#[cfg(test)]
mod help_tests {
    use super::*;

    #[test]
    fn benchmark_flags_require_diagnostics() {
        for flag in ["--no-write", "--gpu-only"] {
            let args = ["vork", "render", "scene.json", "--out", "still", flag];
            let error = Args::try_parse_from(args).err().expect("diagnostics is required");
            assert_eq!(error.kind(), clap::error::ErrorKind::MissingRequiredArgument);
            assert!(Args::try_parse_from(args.into_iter().chain(["--diagnostics"])).is_ok());
        }
    }

    #[test]
    fn channels_default_to_color_and_preserve_explicit_selections() {
        for (flags, expected) in [
            (vec![], vec!["color".to_string()]),
            (vec!["--channels", "all"], vec!["all".to_string()]),
            (vec!["--channels", "color,aov:heat"], vec!["color".to_string(), "aov:heat".to_string()]),
        ] {
            let mut args = vec!["vork", "render", "scene.json", "--out", "still"];
            args.extend(flags);
            let Command::Render { outputs, .. } = Args::try_parse_from(args).unwrap().command else { panic!() };
            assert_eq!(outputs, expected);
        }
    }

    #[test]
    fn setup_and_render_take_a_gpu_index() {
        for args in [vec!["vork", "setup", "--gpu", "cpu"], vec!["vork", "render", "s.json", "--out", "a.png", "--gpu", "cpu"]] {
            let command = Args::try_parse_from(args).unwrap().command;
            assert!(matches!(command, Command::Setup { gpu: Some(vork_render::GpuSelection::Cpu), .. } | Command::Render { gpu: Some(vork_render::GpuSelection::Cpu), .. }));
        }
        let Command::Render { gpu, .. } = Args::try_parse_from(["vork", "render", "s.json", "--out", "a.mp4", "--gpu", "1"]).unwrap().command else { panic!() };
        assert_eq!(gpu, Some(vork_render::GpuSelection::Hardware(1)));
        let Command::Setup { gpu, .. } = Args::try_parse_from(["vork", "setup", "--gpu", "3"]).unwrap().command else { panic!() };
        assert_eq!(gpu, Some(vork_render::GpuSelection::Hardware(3)));
        assert!(Args::try_parse_from(["vork", "setup", "--gpu", "-1"]).is_err());
    }

    #[test]
    fn export_animation_range_is_separate_from_snapshot_time() {
        let parse = |flags: &[&str]| {
            let mut args = vec!["vork", "export", "scene.json", "--out", "walk.glb"];
            args.extend_from_slice(flags); Args::try_parse_from(args)
        };
        let Command::Export { frame_start, frame_end, .. } = parse(&["--frame-start", "12", "--frame-end", "48"]).unwrap().command else { panic!() };
        assert_eq!((frame_start, frame_end), (Some(12), Some(48)));
        for flags in [["--at", "1/24", "--frame-start", "0"], ["--at", "0", "--frame-end", "24"]] {
            assert!(parse(&flags).is_err());
        }
    }
    #[test]
    fn removed_render_aliases_fail() {
        for flag in ["--resolution", "--samples"] {
            assert!(Args::try_parse_from(["vork", "render", "scene.json", "--out", "still", flag, "1"]).is_err());
        }
    }
    #[test]
    fn profiles_and_explicit_settings_parse() {
        let parse = |flags: &[&str]| {
            let mut args = vec!["vork", "render", "scene.json", "--out", "still"];
            args.extend_from_slice(flags);
            Args::try_parse_from(args)
        };
        let Command::Render { profile, set, .. } = parse(&["--profile", "std/draft", "--set", "seed=7", "--set", "samples=8", "--set", "denoise=true"]).unwrap().command
        else { panic!() };
        assert_eq!(profile.as_deref(), Some("std/draft"));
        assert_eq!(set, [("seed".to_string(), serde_json::json!(7)), ("samples".to_string(), serde_json::json!(8)), ("denoise".to_string(), serde_json::json!(true))]);
        assert!(Args::try_parse_from(["vork", "previs", "in.json", "out.json"]).is_err());
        assert!(parse(&["--set", "fast_gi"]).is_err());
        for value in ["0", "-1", "4097", "1.5", "null", "true"] {
            let error = parse(&["--set", &format!("samples={value}")]).err().unwrap().to_string();
            assert!(error.contains("samples"), "{error}");
        }
    }
    #[test]
    fn removed_technique_keys_name_profile_levels() {
        for key in ["reflections", "settings.samples", "shadow_method", "fast_gi", "denoiser"] {
            let error = Args::try_parse_from(["vork", "render", "scene.json", "--out", "still", "--set", &format!("{key}=1")]).err().unwrap().to_string();
            for name in ["std/draft", "std/final", "std/path"] { assert!(error.contains(name), "{error}"); }
        }
    }
    #[test]
    fn frame_range_is_blender_start_end_and_excludes_at() {
        let parse = |flags: &[&str]| {
            let mut args = vec!["vork", "render", "scene.json", "--out", "frames/"];
            args.extend_from_slice(flags);
            Args::try_parse_from(args)
        };
        let Command::Render { frame_start, frame_end, .. } = parse(&["--frame-start", "3", "--frame-end", "7"]).unwrap().command
        else { panic!() };
        assert_eq!((frame_start, frame_end), (Some(3), Some(7)));
        assert!(parse(&["--set", "frame_start=3"]).is_err());
        assert!(parse(&["--set", "frame_end=7"]).is_err());
        assert!(parse(&["--at", "1", "--frame-start", "3"]).is_err());
        assert!(parse(&["--at", "1", "--frame-end", "3"]).is_err());
        assert!(parse(&["--frame-start", "-1"]).is_err());
    }
    #[test]
    fn root_lists_the_help_surface() {
        use clap::CommandFactory;
        let root = root_help();
        assert_eq!(
            Args::command().render_long_help().to_string().trim(),
            root.trim()
        );
        let root_lines = root
            .lines()
            .filter(|line| !line.is_empty())
            .collect::<std::collections::BTreeSet<_>>();
        for document in [
            help::REFERENCE,
            include_str!("../../../docs/help/outputs.md"),
            include_str!("../../../docs/help/errors.md"),
            include_str!("../../../docs/help/camera-json.md"),
            include_str!("../../vork-std/src/help.txt"),
        ] {
            for line in help_text(document).lines().filter(|line| !line.is_empty()) {
                assert!(
                    !root_lines.contains(line),
                    "duplicate root/topic line: {line}"
                );
            }
        }
        for library in vork_scene::libraries::LIBRARIES {
            assert!(constants::HELP_TOPICS.contains(&library.name), "library {} has no help topic", library.name);
        }
        for topic in constants::HELP_TOPICS {
            if *topic == "blender" {
                assert!(include_str!("../../../docs/help/programs.md").contains("vork help blender --all"));
                continue;
            }
            assert!(
                root.split(|c: char| !(c.is_alphanumeric() || c == '-'))
                    .any(|token| token == *topic),
                "unlisted topic {topic}"
            );
        }
    }
    #[test]
    fn camera_json_topic_covers_emitted_fields() {
        let mut camera = vork_render::CameraRecord::default();
        camera.k = Some([[0.; 3]; 3]);
        camera.dist = Some([0.;5]);
        camera.orthographic = Some(vork_render::Orthographic { height: 1. });
        let object = serde_json::to_value(camera).unwrap();
        let documented = include_str!("../../../docs/help/camera-json.md")
            .lines()
            .filter_map(|line| line.split_once(": ").map(|(name, _)| name))
            .filter(|name| !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.'))
            .collect::<std::collections::BTreeSet<_>>();
        let emitted = object
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .chain(["frames", "frame_start", "version", "fps"])
            .collect::<std::collections::BTreeSet<_>>();
        let roots = documented
            .into_iter()
            .filter(|name| !name.contains('.'))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(emitted, roots);
    }
    #[test]
    fn flat_tagged_fields_retain_their_focused_topics() {
        assert_eq!(topic_for_line("node.material (camera): material-name"), Some("cameras"));
        assert_eq!(topic_for_line("node.energy (light): number"), Some("lights"));
        assert_eq!(topic_for_line("node.geometry (mesh): geometry-name"), Some("nodes"));
        assert_eq!(topic_for_line("node [type=\"camera\"]: flat sibling fields"), Some("cameras"));
    }

    #[test]
    fn hair_inputs_are_documented_by_the_std_program() {
        assert!(!help::REFERENCE.lines().any(|line| line.starts_with("hair.")));
        let help = include_str!("../../../libraries/std/HELP.txt");
        assert!(help.contains("std/hair:"));
        assert!(help.contains("vork help field std/hair"));
    }

    #[test]
    fn every_reference_line_has_a_focused_topic() {
        for line in help::REFERENCE
            .lines()
            .filter(|line| line.contains(": "))
        {
            let topic =
                topic_for_line(line).unwrap_or_else(|| panic!("unassigned help line: {line}"));
            assert!(constants::HELP_TOPICS.contains(&topic), "{line}");
        }
    }
}

#[cfg(test)]
mod exposure_tests {
    use super::*;
    #[test]
    fn coverage_labels_and_empty_exposure() {
        let mut e = serde_json::json!({"coverage": 1., "clipped": 0., "near_black": 0., "mean": 0.5});
        assert_eq!(exposure_line(&e), "exposure: 0% clipped, 0% near-black, mean 0.50");
        e["coverage"] = serde_json::json!(0.1);
        assert_eq!(exposure_line(&e), "exposure: 0% clipped, 0% near-black, mean 0.50 of covered pixels");
        e["coverage"] = serde_json::json!(0.);
        assert_eq!(exposure_line(&e), "exposure: undefined (no covered pixels)");
    }
}

#[cfg(test)]
mod interruption_tests {
    use super::*;
    #[test]
    fn interruption_reports_saved_files_without_claiming_no_write_outputs() {
        let paths = [PathBuf::from("my shot/frames")];
        let mut report = serde_json::json!({"interrupted":true,"frames":3,"sidecars":[]});
        let text = interruption_line(&report, &paths).unwrap();
        assert!(text.contains("no output files were written"));
        assert!(!text.contains("saved"));
        report["sidecars"] = serde_json::json!(["my shot/frames/meta.json"]);
        assert!(interruption_line(&report, &paths).unwrap().contains("saved 3 frame(s) to my shot/frames"));
        report["interrupted"] = false.into();
        assert!(interruption_line(&report, &paths).is_none());
    }
}
