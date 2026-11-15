//! A compiled-in encoder can still be incompatible with the installed driver.
#![cfg(unix)]
#[cfg(unix)]
#[path = "../../test_support.rs"]
mod test_support;

use std::process::Command;

#[test]
fn nvenc_probe_preserves_driver_diagnosis() {
    const CHILD: &str = "VORK_TEST_NVENC_PROBE";
    if std::env::var_os(CHILD).is_some() {
        let error = vork_encode::probe_ffmpeg_nvenc(1920, 1080).unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("Driver does not support the required nvenc API version"), "{message}");
        assert!(message.contains("Required: 13.0 Found: 12.2"), "{message}");
        return;
    }
    let root = std::env::temp_dir().join(format!("vork-probe-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let ffmpeg = root.join("ffmpeg");
    test_support::write_executable(&ffmpeg, b"#!/bin/sh\nprintf '%s\\n' 'Driver does not support the required nvenc API version. Required: 13.0 Found: 12.2' >&2\nexit 1\n").unwrap();
    let result = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "nvenc_probe_preserves_driver_diagnosis", "--nocapture"])
        .env(CHILD, "1").env("PATH", &root).output().unwrap();
    std::fs::remove_dir_all(root).unwrap();
    assert!(result.status.success(), "{}\n{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
}
