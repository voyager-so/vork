//! Helpers shared by Unix executable-fixture tests.
#![cfg(unix)]

use std::{
    io::{self, Write},
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Command, Stdio},
};

/// Write in a separate process, then wait until all its write descriptors close.
/// CLOEXEC alone is insufficient: another test can fork while fs::write has the
/// executable open and keep the inode busy until that child execs. Renaming the
/// inode does not help. The test process must never open it for writing.
pub fn write_executable(path: &Path, contents: impl AsRef<[u8]>) -> io::Result<()> {
    let mut child = Command::new("/bin/sh")
        .args(["-c", "exec /bin/cat > \"$1\"", "write-executable"])
        .arg(path)
        .stdin(Stdio::piped())
        .spawn()?;
    let written = child.stdin.take().unwrap().write_all(contents.as_ref());
    // Reap even on a broken pipe; taking stdin closes it before waiting.
    let status = child.wait()?;
    written?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "write executable {}: {status}",
            path.display()
        )));
    }
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
}
