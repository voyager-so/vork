#![cfg(unix)]
#[path = "../../test_support.rs"]
mod test_support;

use std::{fs, process::Command};

#[test]
fn executable_fixture_preserves_bytes_and_handles_shell_metacharacters() {
    let root = std::env::temp_dir().join(format!("vork-script-fixture-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("a 'quoted' $script");
    let script = b"#!/bin/sh\nprintf '%s' 'fixture $value'\n";
    test_support::write_executable(&path, script).unwrap();
    assert_eq!(fs::read(&path).unwrap(), script);
    assert_eq!(
        Command::new(&path).output().unwrap().stdout,
        b"fixture $value"
    );
    test_support::write_executable(&path, b"#!/bin/sh\nexit 7\n").unwrap();
    assert_eq!(Command::new(&path).status().unwrap().code(), Some(7));
    assert!(test_support::write_executable(&root, script).is_err());
    fs::remove_dir_all(root).unwrap();
}

// Force fork rather than posix_spawn, and hold inherited CLOEXEC descriptors
// before exec. This recreates the window other tests' subprocesses can expose.
#[cfg(target_os = "linux")]
#[test]
fn executable_fixtures_survive_concurrent_forks() {
    use std::os::unix::process::CommandExt;
    let root = std::env::temp_dir().join(format!("vork-script-forks-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("script");
    let script = format!("#!/bin/sh\nexit 0\n#{}\n", "x".repeat(1024 * 1024));
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..128 {
                    let mut child = Command::new("/bin/true");
                    // SAFETY: usleep is async-signal-safe on Linux; the callback
                    // neither allocates nor takes a lock after fork.
                    unsafe {
                        child.pre_exec(|| {
                            libc::usleep(5_000);
                            Ok(())
                        });
                    }
                    assert!(child.status().unwrap().success());
                }
            });
        }
        for _ in 0..128 {
            test_support::write_executable(&path, &script).unwrap();
            assert!(Command::new(&path).status().unwrap().success());
        }
    });
    fs::remove_dir_all(root).unwrap();
}
