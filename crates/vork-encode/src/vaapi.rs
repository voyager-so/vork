//! FFmpeg VAAPI output for Linux AMD GPUs. FFmpeg owns the encoder; this module
//! only finds a render node and proves the encoder initialises on it before the
//! render pipeline commits to hardware encoding. `VORK_VAAPI_DEVICE` overrides the node.
use anyhow::{Context, Result, ensure};
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

/// Keep the probe close to the NVENC one: a real frame through the real encoder.
const PROBE_EXTENT: u32 = 256;

/// Every DRM render node to try, in order; an explicit override is used alone.
fn render_nodes() -> Vec<PathBuf> {
    render_nodes_from(std::env::var_os("VORK_VAAPI_DEVICE"))
}

fn render_nodes_from(device: Option<OsString>) -> Vec<PathBuf> {
    match device {
        Some(path) if !path.is_empty() => vec![PathBuf::from(path)],
        _ => (128..=135)
            .map(|index| PathBuf::from(format!("/dev/dri/renderD{index}")))
            .filter(|path| path.exists())
            .collect(),
    }
}

/// Return the first render node where FFmpeg can initialise h264_vaapi.
pub(super) fn probe(width: u32, height: u32) -> Result<PathBuf> {
    probe_nodes(render_nodes(), width, height)
}

fn probe_nodes(nodes: Vec<PathBuf>, width: u32, height: u32) -> Result<PathBuf> {
    let mut nodes = nodes.into_iter();
    let first = nodes.next().context(
        "no /dev/dri/renderD* node; install the GPU driver or pass it into the container",
    )?;
    let error = match probe_node(&first, width, height) {
        Ok(()) => return Ok(first),
        Err(error) => error,
    };
    for node in nodes {
        if probe_node(&node, width, height).is_ok() {
            return Ok(node);
        }
    }
    Err(error)
}

fn probe_node(device: &Path, width: u32, height: u32) -> Result<()> {
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-init_hw_device"])
        .arg(format!("vaapi=va:{}", device.display()))
        .args(["-f", "lavfi", "-i"])
        .arg(format!("color=size={}x{}", probe_extent(width), probe_extent(height)))
        .args(["-frames:v", "1", "-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-f", "null", "-"])
        .stdin(Stdio::null())
        .output()
        .context("start FFmpeg VAAPI probe")?;
    ensure!(
        output.status.success(),
        "FFmpeg h264_vaapi initialization failed ({}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

/// Clamp to the probe range and round up so NV12 gets even dimensions.
fn probe_extent(value: u32) -> u32 {
    (value.clamp(2, PROBE_EXTENT) + 1) & !1
}

/// Hardware-device arguments placed before the rawvideo input.
pub(super) fn device_arguments(device: &Path) -> [String; 4] {
    [
        "-init_hw_device".into(),
        format!("vaapi=va:{}", device.display()),
        "-filter_hw_device".into(),
        "va".into(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_reports_a_missing_render_node() {
        let error = format!("{:#}", probe_nodes(Vec::new(), 64, 64).unwrap_err());
        assert!(error.contains("/dev/dri/renderD"), "{error}");
    }

    #[test]
    fn device_arguments_name_the_render_node() {
        let arguments = device_arguments(Path::new("/dev/dri/renderD128"));
        assert_eq!(
            arguments.iter().map(String::as_str).collect::<Vec<_>>(),
            ["-init_hw_device", "vaapi=va:/dev/dri/renderD128", "-filter_hw_device", "va"]
        );
    }

    #[test]
    fn override_device_is_returned_alone() {
        let device = OsString::from("/dev/dri/renderD999");
        assert_eq!(
            render_nodes_from(Some(device.clone())),
            vec![PathBuf::from(&device)]
        );
    }

    #[test]
    fn probe_extent_clamps_and_rounds_up_to_even() {
        assert_eq!(probe_extent(0), 2);
        assert_eq!(probe_extent(1), 2);
        assert_eq!(probe_extent(63), 64);
        assert_eq!(probe_extent(64), 64);
        assert_eq!(probe_extent(257), 256);
        assert_eq!(probe_extent(4000), 256);
    }
}
