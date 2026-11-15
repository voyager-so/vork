use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

struct Project(PathBuf);
impl Project {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("vork-skills-{}-{name}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn run(&self) -> Output {
        Command::new(env!("CARGO_BIN_EXE_vork"))
            .current_dir(&self.0)
            .args(["skills", "add"])
            .env("VORK_SLANGC", self.0.join("missing-slangc"))
            .env("VK_DRIVER_FILES", self.0.join("missing-driver"))
            .output()
            .unwrap()
    }
}
impl Drop for Project {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const FILES: &[&str] = &["SKILL.md", "references/shot.md", "references/assets.md"];
const DESTINATIONS: &[&str] = &[".agents/skills/vork", ".claude/skills/vork"];

#[test]
fn help_uses_the_approved_sentence_and_pointer() {
    for args in [
        ["skills", "--help"].as_slice(),
        &["skills", "add", "--help"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_vork"))
            .args(args)
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        let help = String::from_utf8(output.stdout).unwrap();
        assert!(help.contains("Install the bundled vork skill for Claude Code and Codex."));
    }
    for arg in ["help", "--help"] {
        let output = Command::new(env!("CARGO_BIN_EXE_vork"))
            .arg(arg)
            .env("VORK_NO_UPDATE_NOTIFIER", "1")
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        let help = String::from_utf8(output.stdout).unwrap();
        assert_eq!(
            help.lines()
                .filter(|line| *line == "Authoring skill: vork skills add")
                .count(),
            1
        );
    }
}

#[cfg(any(target_os = "linux", windows))]
#[test]
fn success_and_refusal_skip_cached_update_notices() {
    let project = Project::new("no-update");
    let cache = project.0.join("cache/vork");
    fs::create_dir_all(&cache).unwrap();
    fs::write(cache.join("update-notice.json"), serde_json::to_vec(&serde_json::json!({
        "checked": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        "release": {"version": "999.0.0"}
    })).unwrap()).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_vork"));
    command
        .current_dir(&project.0)
        .args(["skills", "add"])
        .env_remove("CI")
        .env_remove("VORK_NO_UPDATE_NOTIFIER")
        .env("XDG_CACHE_HOME", cache.parent().unwrap())
        .env("LOCALAPPDATA", cache.parent().unwrap());
    let output = command.output().unwrap();
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty(), "{output:?}");
    let cache_path = cache.join("update-notice.json");
    let cached = fs::read(&cache_path).unwrap();
    let modified = fs::metadata(&cache_path).unwrap().modified().unwrap();
    fs::remove_file(project.0.join(".claude/skills/vork/SKILL.md")).unwrap();
    fs::create_dir(project.0.join(".claude/skills/vork/SKILL.md")).unwrap();
    let output = command.output().unwrap();
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(!String::from_utf8_lossy(&output.stderr).contains("Update available"));
    assert_eq!(fs::read(&cache_path).unwrap(), cached);
    assert_eq!(
        fs::metadata(&cache_path).unwrap().modified().unwrap(),
        modified
    );
}

#[test]
fn installs_exact_bytes_is_idempotent_and_refreshes_only_owned_files() {
    let project = Project::new("space in path");
    let installed = project.run();
    assert!(installed.status.success(), "{installed:?}");
    assert_eq!(
        String::from_utf8(installed.stdout).unwrap(),
        ".agents/skills/vork/\n.claude/skills/vork/\n"
    );
    let mut mtimes = Vec::new();
    for destination in DESTINATIONS {
        for name in FILES {
            let path = project.0.join(destination).join(name);
            let source = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../skills/vork")
                .join(name);
            assert_eq!(fs::read(&path).unwrap(), fs::read(source).unwrap());
            mtimes.push((
                path.clone(),
                fs::metadata(path).unwrap().modified().unwrap(),
            ));
        }
        fs::write(project.0.join(destination).join("personal.md"), "keep").unwrap();
    }
    for agent in [".agents", ".claude"] {
        fs::create_dir_all(project.0.join(agent).join("skills/other")).unwrap();
        fs::write(
            project.0.join(agent).join("skills/other/SKILL.md"),
            "other skill",
        )
        .unwrap();
        fs::write(
            project.0.join(agent).join("skills/personal.txt"),
            "other file",
        )
        .unwrap();
    }
    // Force a distinct old mtime so an unnecessary rewrite is observable on every filesystem.
    for (path, time) in &mut mtimes {
        *time = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_600_000_000);
        fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(*time)
            .unwrap();
    }
    assert!(project.run().status.success());
    for (path, time) in mtimes {
        assert_eq!(fs::metadata(path).unwrap().modified().unwrap(), time);
    }
    for destination in DESTINATIONS {
        for name in FILES {
            fs::write(project.0.join(destination).join(name), "old version").unwrap();
        }
    }
    assert!(project.run().status.success());
    for destination in DESTINATIONS {
        for name in FILES {
            let source = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../skills/vork")
                .join(name);
            assert_eq!(
                fs::read(project.0.join(destination).join(name)).unwrap(),
                fs::read(source).unwrap()
            );
        }
        assert_eq!(
            fs::read_to_string(project.0.join(destination).join("personal.md")).unwrap(),
            "keep"
        );
    }
    for agent in [".agents", ".claude"] {
        assert_eq!(
            fs::read_to_string(project.0.join(agent).join("skills/other/SKILL.md")).unwrap(),
            "other skill"
        );
        assert_eq!(
            fs::read_to_string(project.0.join(agent).join("skills/personal.txt")).unwrap(),
            "other file"
        );
    }
}

#[test]
fn all_conflicts_are_refused_before_either_destination_changes() {
    for (i, conflict) in [
        ".claude",
        ".claude/skills",
        ".claude/skills/vork",
        ".claude/skills/vork/references",
        ".claude/skills/vork/SKILL.md",
        ".claude/skills/vork/references/shot.md",
        ".claude/skills/vork/references/assets.md",
    ]
    .iter()
    .enumerate()
    {
        let project = Project::new(&format!("conflict-{i}"));
        let path = project.0.join(conflict);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        if conflict.ends_with(".md") {
            fs::create_dir(&path).unwrap();
        } else {
            fs::write(&path, "preserve").unwrap();
        }
        let output = project.run();
        assert_eq!(output.status.code(), Some(2), "{output:?}");
        assert!(String::from_utf8_lossy(&output.stderr)
            .replace('\\', "/")
            .contains(conflict));
        assert!(String::from_utf8_lossy(&output.stderr).contains("move it aside"));
        assert_eq!(String::from_utf8_lossy(&output.stderr).lines().count(), 1);
        assert!(output.stdout.is_empty());
        assert!(!project.0.join(".agents").exists());
        if !conflict.ends_with(".md") {
            assert_eq!(fs::read_to_string(path).unwrap(), "preserve");
        }
    }
}

#[test]
fn a_later_conflict_preserves_existing_files_that_would_be_refreshed() {
    let project = Project::new("preflight-refresh");
    assert!(project.run().status.success());
    let earlier = project.0.join(".agents/skills/vork/SKILL.md");
    fs::write(&earlier, "local change").unwrap();
    let conflict = project.0.join(".claude/skills/vork/references/assets.md");
    fs::remove_file(&conflict).unwrap();
    fs::create_dir(&conflict).unwrap();
    let output = project.run();
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert_eq!(fs::read_to_string(earlier).unwrap(), "local change");
    assert_eq!(fs::read_dir(conflict).unwrap().count(), 0);
}

#[cfg(unix)]
#[test]
fn symlinks_including_dangling_links_are_refused_before_writing() {
    for (i, conflict) in [
        ".claude",
        ".claude/skills",
        ".claude/skills/vork",
        ".claude/skills/vork/references",
        ".claude/skills/vork/SKILL.md",
        ".claude/skills/vork/references/shot.md",
        ".claude/skills/vork/references/assets.md",
    ]
    .iter()
    .enumerate()
    {
        for dangling in [false, true] {
            let project = Project::new(&format!("symlink-{i}-{dangling}"));
            let target = project.0.join("external");
            if !dangling {
                fs::create_dir(&target).unwrap();
            }
            let path = project.0.join(conflict);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::os::unix::fs::symlink(&target, &path).unwrap();
            let output = project.run();
            assert_eq!(output.status.code(), Some(2), "{output:?}");
            assert!(String::from_utf8_lossy(&output.stderr).contains(conflict));
            assert!(String::from_utf8_lossy(&output.stderr).contains("move it aside"));
            assert_eq!(String::from_utf8_lossy(&output.stderr).lines().count(), 1);
            assert!(output.stdout.is_empty());
            assert!(path.symlink_metadata().unwrap().is_symlink());
            assert!(!project.0.join(".agents").exists());
            assert_eq!(target.exists(), !dangling);
            if !dangling {
                assert_eq!(fs::read_dir(target).unwrap().count(), 0);
            }
        }
    }
}
