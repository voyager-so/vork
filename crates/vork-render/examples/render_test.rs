//! Test-only shot driver. Provider overrides never enter production profile parsing.
use anyhow::{Context, Result, bail};
use std::io::Write;
use vork_render::shot::Shot;
fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error:#}");
        std::process::exit(i32::from(vork_render::failure::exit_code(&error)));
    }
}
fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("render") {
        let executable = std::env::current_exe()?;
        let directory = executable.parent().context("driver directory")?;
        let sibling = directory.join("vork");
        let production = if sibling.is_file() && sibling != executable { sibling } else { directory.parent().context("target directory")?.join("vork") };
        let status = std::process::Command::new(production).args(std::env::args().skip(1)).status()?;
        std::process::exit(status.code().unwrap_or(1));
    }
    let mut shot = Shot::default();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--out" => shot.out.push(args.next().context("output")?.parse().map_err(anyhow::Error::msg)?),
            "--profile" => shot.profile = Some(vork_scene::Profile::load(&args.next().context("profile")?)?),
            "--techniques" => {
                let mut techniques = serde_json::from_slice::<serde_json::Map<String,serde_json::Value>>(&std::fs::read(args.next().context("techniques")?)?)?;
                techniques.entry("integrator").or_insert_with(|| "raster".into());
                shot.test_techniques.extend(techniques);
            }
            "--force" => {
                let value = args.next().context("KEY=JSON")?;
                let (key,value) = value.split_once('=').context("KEY=JSON")?;
                shot.test_techniques.insert(key.into(),serde_json::from_str(value)?);
            }
            "--set" => { let (key,value) = vork_scene::CallOverrides::parse(&args.next().context("override")?)?; shot.settings.insert(key,value); }
            "--at" => shot.at = Some(args.next().context("time")?.parse()?),
            "--frame-start" => shot.frame_start = Some(args.next().context("frame")?.parse()?),
            "--frame-end" => shot.frame_end = Some(args.next().context("frame")?.parse()?),
            "--channels" => {
                let channels = args.next().context("channels")?;
                shot.outputs = if channels == "all" { Vec::new() } else {
                    channels.split(',').map(|value| value.parse().map_err(anyhow::Error::msg)).collect::<Result<_>>()?
                };
            }
            "--diagnostics" => shot.diagnostics = true,
            "--no-write" => shot.no_write = true,
            "--gpu-only" => shot.gpu_only = true,
            "--exr-codec" => shot.exr_codec = args.next().context("codec")?.parse().map_err(anyhow::Error::msg)?,
            "--png-depth" => shot.png_depth = args.next().context("depth")?.parse().map_err(anyhow::Error::msg)?,
            _ if !flag.starts_with('-') && shot.scene.as_os_str().is_empty() => shot.scene = flag.into(),
            _ => bail!("unknown test driver argument {flag}"),
        }
    }
    anyhow::ensure!(!shot.scene.as_os_str().is_empty(), "scene is required");
    let rendered = shot.render()?;
    writeln!(std::io::stdout().lock(), "{}", rendered.report)?;
    std::io::stdout().flush()?;
    std::io::stderr().flush()?;
    rendered.exit_process()
}
