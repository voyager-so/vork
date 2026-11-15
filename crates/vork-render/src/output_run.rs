//! Destination ownership and the incomplete -> complete output transition.
use anyhow::{Context, Result};
use std::{
    fs,
    ops::RangeInclusive,
    path::{Path, PathBuf},
};

pub(crate) fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    if super::shot::directory_marked(path) || path.extension().is_none() {
        path.join(suffix)
    } else {
        super::shot::sidecar(path, suffix)
    }
}

fn remove(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("remove previous output {}", path.display())),
    }
}

fn numbered(value: &str) -> bool {
    value.len() >= 6 && value.bytes().all(|b| b.is_ascii_digit())
}

fn png_suffix(value: &str) -> bool {
    matches!(
        value,
        ".png"
            | ".depth.png"
            | ".object_index.png"
            | ".view_normal.png"
            | ".motion.png"
            | ".shadow_catcher.png"
    ) || value
        .strip_prefix(".aov.")
        .and_then(|v| v.strip_suffix(".png"))
        .is_some_and(|v| {
            !v.is_empty()
                && v.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
        })
}

fn owns(path: &Path, name: &std::ffi::OsStr, frames: Option<&RangeInclusive<u32>>) -> bool {
    if super::shot::directory_marked(path) || path.extension().is_none() {
        let Some(name) = name.to_str() else {
            return false;
        };
        let i = name.find('.').unwrap_or(name.len());
        numbered(&name[..i]) && png_suffix(&name[i..])
            && frames.is_none_or(|frames| name[..i].parse::<u32>().is_ok_and(|n| frames.contains(&n)))
    } else if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("exr"))
    {
        let Some(name) = name.to_str() else {
            return false;
        };
        let stem = path.file_stem().unwrap().to_string_lossy();
        let Some((prefix, _)) = stem.split_once("%06d") else {
            return false;
        };
        let Some(name) = name.strip_suffix(".exr") else {
            return false;
        };
        let Some(rest) = name.strip_prefix(prefix) else {
            return false;
        };
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        (6..=digits).any(|n| stem.replace("%06d", &rest[..n]) == name
            && frames.is_none_or(|frames| rest[..n].parse::<u32>().is_ok_and(|n| frames.contains(&n))))
    } else if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("png"))
    {
        name == path.file_name().unwrap()
            || name
                .to_str()
                .and_then(|name| {
                    name.strip_prefix(path.file_stem().unwrap().to_string_lossy().as_ref())
                })
                .is_some_and(png_suffix)
    } else {
        name == path.file_name().unwrap()
    }
}

// A namespace is a small regular filename language. EXR placeholders may repeat;
// treating their digits independently is conservative for overlap admission.
#[derive(Clone, Copy)]
enum Token { Byte(u8), Digits, Aov, AovTail }
fn accepts(token: Token, byte: u8) -> bool {
    match token {
        // Reserve ASCII case variants too, so admission is safe on case-insensitive volumes.
        Token::Byte(value) => byte.eq_ignore_ascii_case(&value),
        Token::Digits => byte.is_ascii_digit(),
        Token::Aov | Token::AovTail => byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'),
    }
}
fn literal(bytes: &[u8]) -> Vec<Token> { bytes.iter().copied().map(Token::Byte).collect() }
fn pattern(bytes: &[u8]) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut rest = bytes;
    while let Some(index) = rest.windows(4).position(|part| part == b"%06d") {
        tokens.extend(literal(&rest[..index]));
        tokens.extend(std::iter::repeat_n(Token::Digits, 6));
        // A repeated token is represented separately below by a self-loop.
        tokens.push(Token::Digits);
        rest = &rest[index + 4..];
    }
    tokens.extend(literal(rest));
    tokens
}
// Only the seventh consecutive digit token and AOV tokens have a self-loop
// and an epsilon exit. The first six digit tokens are mandatory.
fn repeating(tokens: &[Token], index: usize) -> bool {
    matches!(tokens.get(index), Some(Token::AovTail)) ||
        (index >= 6 && tokens[index-6..=index].iter().all(|t|matches!(t, Token::Digits)))
}
fn intersects(a: &[Token], b: &[Token]) -> bool {
    let mut pending = vec![(0, 0)];
    let mut seen = std::collections::BTreeSet::new();
    while let Some((i, j)) = pending.pop() {
        if !seen.insert((i, j)) { continue; }
        if i == a.len() && j == b.len() { return true; }
        if i < a.len() && repeating(a, i) { pending.push((i+1, j)); }
        if j < b.len() && repeating(b, j) { pending.push((i, j+1)); }
        if i < a.len() && j < b.len() && (0..=255).any(|c|accepts(a[i], c) && accepts(b[j], c)) {
            pending.push((if repeating(a, i) {i}else{i+1}, if repeating(b, j) {j}else{j+1}));
        }
    }
    false
}
fn conflicts(a_dir: &Path, a: &[Vec<Token>], b_dir: &Path, b: &[Vec<Token>]) -> bool {
    if a_dir == b_dir { return a.iter().any(|a|b.iter().any(|b|intersects(a,b))); }
    // A generated file cannot also be a containing directory of another output.
    let contains = |parent: &Path, names: &[Vec<Token>], child: &Path| {
        child.strip_prefix(parent).ok().and_then(|rest|rest.components().next())
            .is_some_and(|part|names.iter().any(|name|intersects(name,&literal(part.as_os_str().as_encoded_bytes()))))
    };
    contains(a_dir,a,b_dir) || contains(b_dir,b,a_dir)
}
fn directory(path: &Path) -> &Path {
    if super::shot::directory_marked(path) || path.extension().is_none() { path }
    else { path.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or(Path::new(".")) }
}
fn namespace(path: &Path) -> Vec<Vec<Token>> {
    let is_directory = super::shot::directory_marked(path) || path.extension().is_none();
    let mut names: Vec<Vec<Token>> = ["meta.json", "meta.json.pending", "camera.json", "timings.json"]
        .iter().map(|suffix|literal(sidecar(path, suffix).file_name().unwrap().as_encoded_bytes())).collect();
    if is_directory || path.extension().is_some_and(|e|e.eq_ignore_ascii_case("png")) {
        let stem = if is_directory { pattern(b"%06d") } else { literal(path.file_stem().unwrap().as_encoded_bytes()) };
        for suffix in [".png", ".depth.png", ".object_index.png", ".view_normal.png", ".motion.png", ".shadow_catcher.png"] {
            let mut name = stem.clone(); name.extend(literal(suffix.as_bytes())); names.push(name);
        }
        let mut aov = stem; aov.extend(literal(b".aov."));
        // At least one AOV character, then zero or more.
        aov.push(Token::Aov); aov.push(Token::AovTail); aov.extend(literal(b".png")); names.push(aov);
    } else if path.extension().is_some_and(|e|e.eq_ignore_ascii_case("exr")) {
        names.push(pattern(path.file_name().unwrap().as_encoded_bytes()));
    } else { names.push(literal(path.file_name().unwrap().as_encoded_bytes())); }
    names
}

struct DirectoryLock(fs::File);
impl Drop for DirectoryLock {
    fn drop(&mut self) {
        // Release ownership explicitly: a concurrently spawned child can hold
        // an inherited descriptor briefly before close-on-exec runs.
        let _ = self.0.unlock();
    }
}

pub(crate) struct OutputRun {
    // Keep lock files in place: unlinking a locked inode admits a second writer.
    _locks: Vec<DirectoryLock>,
    manifests: Vec<PathBuf>,
}

impl OutputRun {
    /// Called before any encoder attaches or frame writer opens a final path.
    /// Invalidation is eager, so process termination needs no destructor/handler.
    /// `frames` limits numbered PNG/EXR cleanup; None replaces the whole clip.
    /// Completion, camera and timing sidecars always belong to this run alone.
    pub(crate) fn begin(paths: &[PathBuf], frames: Option<RangeInclusive<u32>>) -> Result<Self> {
        let directories: Vec<_> = paths.iter().map(|path|fs::canonicalize(directory(path)))
            .collect::<std::io::Result<_>>()?;
        let namespaces: Vec<_> = paths.iter().map(|path|namespace(path)).collect();
        for i in 0..paths.len() {
            for j in 0..i {
                if conflicts(&directories[i], &namespaces[i], &directories[j], &namespaces[j]) {
                    return Err(anyhow::anyhow!("overlapping output namespaces: {} and {}", paths[j].display(), paths[i].display())
                        .context(crate::failure::FailureKind::Usage));
                }
            }
        }
        let mut locks = Vec::new();
        // One lock per canonical directory protects every possible generated
        // filename there, including names not present when the run begins.
        let unique: std::collections::BTreeSet<_> = directories.into_iter().collect();
        for directory in unique {
            let lock = fs::OpenOptions::new().create(true).truncate(false).write(true)
                .open(directory.join(".vork-output.lock"))?;
            lock.try_lock().map_err(std::io::Error::from)
                .with_context(||format!("output already in use: {}", directory.display()))?;
            locks.push(DirectoryLock(lock));
        }
        let manifests: Vec<_> = paths.iter().map(|path|sidecar(path, "meta.json")).collect();
        // Invalidate every destination before deleting any data.
        for manifest in &manifests {
            remove(manifest)?;
        }
        for path in paths {
            for suffix in ["camera.json", "timings.json", "meta.json.pending"] {
                remove(&sidecar(path, suffix))?;
            }
            let directory = if super::shot::directory_marked(path) || path.extension().is_none() {
                path.as_path()
            } else {
                path.parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."))
            };
            for entry in fs::read_dir(directory)? {
                let entry = entry?;
                if owns(path, &entry.file_name(), frames.as_ref()) {
                    remove(&entry.path())?;
                }
            }
        }
        Ok(Self {
            _locks: locks,
            manifests,
        })
    }

    /// All data, camera and diagnostic writes must succeed before this call.
    pub(crate) fn complete(&self, metadata: &[(PathBuf, Vec<u8>)]) -> Result<()> {
        for (path, bytes) in metadata {
            fs::write(path.with_extension("json.pending"), bytes)?;
        }
        for (path, _) in metadata {
            if let Err(error) = fs::rename(path.with_extension("json.pending"), path) {
                // A later publication failure must not leave an earlier manifest.
                let cleanup: Vec<_> = self
                    .manifests
                    .iter()
                    .filter_map(|manifest| remove(manifest).err())
                    .map(|error| format!("{error:#}"))
                    .collect();
                return Err(anyhow::Error::from(error).context(format!(
                    "publish output metadata{}",
                    if cleanup.is_empty() {
                        String::new()
                    } else {
                        format!("; cleanup also failed: {}", cleanup.join("; "))
                    }
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_frames_preserve_other_invocations_even_on_failure() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-output-range-{}", std::process::id()));
        for destination in ["frames", "frames.v2/", "clip-%06d.exr", "repeat-%06d2-%06d.exr"] {
            let path = root.join(destination);
            let dir = directory(&path);
            fs::create_dir_all(dir)?;
            let names = |frame: u32| -> Vec<String> {
                if destination.ends_with(".exr") {
                    vec![destination.replace("%06d", &format!("{frame:06}"))]
                } else {
                    [".png", ".depth.png", ".aov.old.png"].iter()
                        .map(|suffix| format!("{frame:06}{suffix}")).collect()
                }
            };
            // First invocation saves frame 216 and publishes completion metadata.
            let run = OutputRun::begin(std::slice::from_ref(&path), Some(216..=216))?;
            for name in names(216) { fs::write(dir.join(name), "first success")?; }
            run.complete(&[(sidecar(&path, "meta.json"), b"first metadata".to_vec())])?;
            drop(run);
            for frame in [217, 218, 219, 1_000_000] {
                for name in names(frame) { fs::write(dir.join(name), "old")?; }
            }
            fs::write(dir.join("notes.png"), "unrelated")?;
            for suffix in ["camera.json", "timings.json", "meta.json.pending"] {
                fs::write(sidecar(&path, suffix), "old")?;
            }
            // A second invocation fails before saving anything. Only its inclusive
            // interval and per-run sidecars may have been invalidated.
            let run = OutputRun::begin(std::slice::from_ref(&path), Some(217..=218))?;
            drop(run);
            for frame in [217, 218] {
                for name in names(frame) { assert!(!dir.join(name).exists()); }
            }
            for frame in [216, 219, 1_000_000] {
                for name in names(frame) { assert!(dir.join(name).exists(), "preserve {frame}"); }
            }
            assert_eq!(fs::read(dir.join(&names(216)[0]))?, b"first success");
            assert!(dir.join("notes.png").exists());
            for suffix in ["meta.json", "camera.json", "timings.json", "meta.json.pending"] {
                assert!(!sidecar(&path, suffix).exists());
            }
            // Retrying can complete, and publishes only this invocation's record.
            let run = OutputRun::begin(std::slice::from_ref(&path), Some(217..=218))?;
            for frame in [217, 218] {
                for name in names(frame) { fs::write(dir.join(name), "retry success")?; }
            }
            run.complete(&[(sidecar(&path, "meta.json"), b"second metadata".to_vec())])?;
            drop(run);
            assert_eq!(fs::read(sidecar(&path, "meta.json"))?, b"second metadata");
            for name in names(216) { assert_eq!(fs::read(dir.join(name))?, b"first success"); }
        }
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn overwrite_cleans_only_owned_outputs_and_locks_until_finished() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-output-run-{}", std::process::id()));
        fs::create_dir_all(&root)?;
        let cases = [
            (
                root.join("frames"),
                vec!["000000.png", "9999999.depth.png", "000002.aov.old.png"],
                vec!["notes.png", "000003.unknown.png"],
            ),
            (
                root.join("still.png"),
                vec!["still.png", "still.depth.png", "still.aov.old.png"],
                vec!["other.png"],
            ),
            (
                root.join("clip-%06d.exr"),
                vec!["clip-000000.exr", "clip-1000000.exr"],
                vec!["other-000000.exr"],
            ),
            (
                root.join("repeat-%06d2-%06d.exr"),
                vec!["repeat-0000002-000000.exr"],
                vec!["repeat-0000002-000001.exr"],
            ),
            (root.join("video.mp4"), vec!["video.mp4"], vec!["other.mp4"]),
        ];
        for (path, owned, unrelated) in cases {
            let directory = if path.extension().is_none() {
                &path
            } else {
                &root
            };
            fs::create_dir_all(directory)?;
            for name in owned.iter().chain(&unrelated) {
                fs::write(directory.join(name), "old")?;
            }
            for suffix in ["meta.json", "camera.json", "timings.json"] {
                fs::write(sidecar(&path, suffix), "old")?;
            }
            let run = OutputRun::begin(std::slice::from_ref(&path), None)?;
            let conflict = OutputRun::begin(std::slice::from_ref(&path), None)
                .err()
                .expect("locked output");
            assert!(conflict.chain().any(|error| error.is::<std::io::Error>()));
            for suffix in ["meta.json", "camera.json", "timings.json"] {
                assert!(!sidecar(&path, suffix).exists());
            }
            for name in owned {
                assert!(!directory.join(name).exists(), "{name}");
            }
            for name in unrelated {
                assert!(directory.join(name).exists(), "{name}");
            }
            // Abandoning a run, including abrupt termination, cannot expose old success.
            drop(run);
            assert!(!sidecar(&path, "meta.json").exists());
            let run = OutputRun::begin(std::slice::from_ref(&path), None)?;
            run.complete(&[(sidecar(&path, "meta.json"), b"{}".to_vec())])?;
            assert_eq!(fs::read(sidecar(&path, "meta.json"))?, b"{}");
        }
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn overlapping_namespaces_are_rejected_before_cleanup() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-output-overlap-{}", std::process::id()));
        fs::create_dir_all(&root)?;
        for paths in [
            vec![root.clone(), root.join("000000.png")],
            vec![root.join("still.png"), root.join("still.depth.png")],
            vec![root.join("Still.png"), root.join("still.depth.png")],
            vec![root.join("still.png"), root.join("still.aov.mask.png")],
            vec![root.join("clip-%06d.exr"), root.join("clip-%06d.exr")],
        ] {
            let sentinel = root.join("000000.png"); fs::write(&sentinel, "live")?;
            assert!(format!("{:#}", OutputRun::begin(&paths, None).err().expect("overlap")).contains("overlapping"));
            assert_eq!(fs::read(&sentinel)?, b"live");
        }
        let nested = root.join("container.png"); fs::create_dir_all(&nested)?;
        fs::write(nested.join("000000.png"), "live")?;
        assert!(OutputRun::begin(&[nested.clone(), PathBuf::from(format!("{}/", nested.display()))], None).is_err());
        assert_eq!(fs::read(nested.join("000000.png"))?, b"live");
        let distinct = OutputRun::begin(&[root.join("a.png"),root.join("b.png"),root.join("video.mp4")], None)?;
        drop(distinct);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn overlapping_process_cannot_delete_live_frames() -> Result<()> {
        const CHILD: &str = "VORK_OUTPUT_OVERLAP_CHILD";
        if let Some(root) = std::env::var_os(CHILD) {
            assert!(OutputRun::begin(&[PathBuf::from(root).join("000000.png")], None).is_err());
            return Ok(());
        }
        let root = std::env::temp_dir().join(format!("vork-output-process-{}", std::process::id()));
        fs::create_dir_all(&root)?;
        let run = OutputRun::begin(std::slice::from_ref(&root), None)?;
        fs::write(root.join("000000.png"), "live")?;
        let status = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", "output_run::tests::overlapping_process_cannot_delete_live_frames"])
            .env(CHILD, &root).status()?;
        assert!(status.success());
        assert_eq!(fs::read(root.join("000000.png"))?, b"live");
        drop(run); fs::remove_dir_all(root)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn native_still_filenames_are_cleaned_when_color_is_deselected() -> Result<()> {
        use std::os::unix::ffi::OsStringExt;
        let root = std::env::temp_dir().join(format!("vork-output-native-{}", std::process::id()));
        fs::create_dir_all(&root)?;
        let path = root.join(std::ffi::OsString::from_vec(b"still-\xff.png".to_vec()));
        fs::write(&path, "previous color")?;
        let run = OutputRun::begin(std::slice::from_ref(&path), None)?;
        assert!(
            !path.exists(),
            "deselected color must not survive a new run"
        );
        drop(run);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn failed_invalidation_never_modifies_frames() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("vork-output-invalidate-{}", std::process::id()));
        fs::create_dir_all(root.join("meta.json"))?;
        fs::write(root.join("000000.png"), "previous")?;
        let error = OutputRun::begin(std::slice::from_ref(&root), None)
            .err()
            .expect("directory cannot be removed as a manifest");
        assert!(format!("{error:#}").contains("meta.json"));
        assert_eq!(fs::read(root.join("000000.png"))?, b"previous");
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn failed_publication_invalidates_all_destinations() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-output-publish-{}", std::process::id()));
        let paths = [root.join("a"), root.join("b")];
        for path in &paths {
            fs::create_dir_all(path)?;
        }
        let run = OutputRun::begin(&paths, None)?;
        let metadata: Vec<_> = paths
            .iter()
            .map(|p| (sidecar(p, "meta.json"), b"{}".to_vec()))
            .collect();
        // Fail the second rename after the first has succeeded.
        fs::create_dir(&metadata[1].0)?;
        assert!(run.complete(&metadata).is_err());
        assert!(!metadata[0].0.exists());
        assert!(metadata[1].0.is_dir());
        drop(run);
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
