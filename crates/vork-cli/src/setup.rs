//! Native driver calls cannot be cancelled in-process. Supervise required checks
//! separately from optional shader warming; cache entries publish atomically.
use anyhow::{Context, Result};
use std::{path::Path, process::{Child, Command, ExitStatus}, thread, time::{Duration, Instant}};

const STAGE_FILE: &str = "VORK_SETUP_STAGE_FILE";
const WARMING: &str = "warming shaders";

pub fn stage(name: &str) -> Result<()> {
    if let Some(path) = std::env::var_os(STAGE_FILE) {
        std::fs::write(path, name).context("report setup stage")?;
    }
    Ok(())
}

pub fn bounded(all: bool, skip_warmup: bool, verbose: bool, gpu: Option<vork_render::GpuSelection>) -> Result<()> {
    let unbounded = all || vork_render::selected_gpu().context(vork_render::failure::FailureKind::Device)? == Some(vork_render::GpuSelection::Cpu);
    let path = std::env::temp_dir().join(format!("vork-setup-{}-{}", std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()));
    let file = std::fs::OpenOptions::new().write(true).create_new(true).open(&path)?;
    let result: Result<Option<ExitStatus>> = (|| {
        let mut command = Command::new(std::env::current_exe()?);
        command.args(["setup", "--warmup-worker"]).env(STAGE_FILE, &path);
        if let Some(index) = gpu { command.args(["--gpu", &index.to_string()]); }
        if all { command.arg("--all"); }
        if verbose { command.arg("--verbose"); }
        if skip_warmup { command.arg("--skip-warmup"); }
        let mut child = command.spawn().context("start setup checks")?;
        match wait_bounded(&mut child, &path, Duration::from_secs(600), unbounded)? {
            Some(status) if !status.success() => return Ok(Some(status)),
            None => output!("Shader warmup stopped at the 600 s limit; completed cache entries are retained. Rendering will compile remaining pipelines on demand. Run `vork setup` again to continue."),
            _ => {}
        }
        Ok(None)
    })();
    drop(file);
    // The worker records its stage before each phase. A non-zero exit while the
    // stage is WARMING means the required checks finished and only the optional
    // shader warmup failed. Warmup is cache preparation, not a rendering
    // prerequisite ("Without setup, nothing breaks"), so a driver crash there
    // must not turn a working install into a failed `setup` and a failing
    // installer. Report it, keep the completed cache entries, and let rendering
    // compile the remaining pipelines on demand.
    let stage = std::fs::read_to_string(&path).unwrap_or_default();
    let _ = std::fs::remove_file(path);
    if let Some(status) = result? {
        if stage.trim() == WARMING {
            output!("Shader warmup did not complete ({status}); completed cache entries are retained. Rendering will compile pipelines on demand. Run `vork setup` again to continue.");
            return Ok(());
        }
        // Preserve the worker’s diagnostic and typed CLI failure code.
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

fn wait_bounded(child: &mut Child, path: &Path, budget: Duration, all: bool) -> Result<Option<ExitStatus>> {
    let mut start = Instant::now();
    let mut stage = "starting setup checks".to_owned();
    loop {
        if let Some(status) = child.try_wait()? { return Ok(Some(status)); }
        // A stage is tiny; tolerate an empty read while the worker updates it.
        if let Ok(next) = std::fs::read_to_string(path) {
            if !next.is_empty() && next != stage {
                stage = next;
                start = Instant::now();
            }
        }
        if !(all && stage == WARMING) && start.elapsed() >= budget {
            child.kill().context("stop setup at its time limit")?;
            child.wait()?;
            if stage != WARMING { anyhow::bail!("setup timed out while {stage} ({} s limit)", budget.as_secs_f64()); }
            return Ok(None);
        }
        thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(all(test,unix))]
mod tests {
    use super::*;

    #[test]
    fn required_checks_fail_but_shader_timeout_is_optional() -> Result<()> {
        let path = std::env::temp_dir().join(format!("vork-setup-test-{}",std::process::id()));
        for stage in ["checking FFmpeg", "opening GPU", "checking tools", WARMING] {
            std::fs::write(&path,stage)?;
            let mut slow = Command::new("sh").args(["-c","exec sleep 30"]).spawn()?;
            let result = wait_bounded(&mut slow,&path,Duration::from_millis(5),false);
            if stage == WARMING { assert!(result?.is_none()); }
            else { assert!(result.unwrap_err().to_string().contains(stage)); }
            assert!(slow.try_wait()?.is_some());
        }
        let mut failed = Command::new("sh").args(["-c","exit 7"]).spawn()?;
        assert_eq!(wait_bounded(&mut failed,&path,Duration::from_secs(5),false)?.unwrap().code(),Some(7));
        std::fs::remove_file(path)?;
        Ok(())
    }
}
