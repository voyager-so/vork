//! Slang owns resolution. This filesystem records exactly what it asks and can
//! replay those observations without consulting mutable external inputs.
use super::*;
use std::io::{Read, Write};
use std::process::Stdio;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Query {
    pub op: u8,
    pub path: PathBuf,
}
pub(super) type Observations = BTreeMap<Query, Option<Vec<u8>>>;

pub(super) mod observations_serde {
    use super::*;
    pub fn serialize<S: serde::Serializer>(
        value: &Observations,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        value.iter().collect::<Vec<_>>().serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Observations, D::Error> {
        Ok(Vec::<(Query, Option<Vec<u8>>)>::deserialize(deserializer)?
            .into_iter()
            .collect())
    }
}

fn path(text: &str) -> PathBuf {
    // Slang's OS filesystem accepts both separators on every host.
    PathBuf::from(text.replace('\\', "/"))
}
pub(super) fn compiler_path(p: &Path) -> String {
    let text = p.to_string_lossy().replace('\\', "/");
    if let Some(rest) = text.strip_prefix("//?/UNC/") {
        format!("//{rest}")
    } else {
        text.strip_prefix("//?/").unwrap_or(&text).to_owned()
    }
}

pub(super) fn fingerprint(query: &Query, value: &Option<Vec<u8>>) -> Option<Vec<u8>> {
    value.as_ref().map(|bytes| {
        if query.op == 0 {
            blake3::hash(bytes).as_bytes().to_vec()
        } else {
            bytes.clone()
        }
    })
}

pub(super) fn observe(query: &Query) -> Option<Vec<u8>> {
    match query.op {
        0 => fs::read(&query.path).ok(),
        1 => query
            .path
            .canonicalize()
            .ok()
            .map(|p| compiler_path(&p).into_bytes()),
        2 => fs::metadata(&query.path).ok().and_then(|m| {
            if m.is_file() {
                Some(vec![1])
            } else if m.is_dir() {
                Some(vec![0])
            } else {
                None
            }
        }),
        _ => None,
    }
}
struct State {
    root: PathBuf,
    observations: Observations,
    replay: bool,
    unobserved: Vec<Query>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    dependencies: BTreeMap<PathBuf, Vec<PathBuf>>,
}
impl State {
    fn absolute(&self, p: PathBuf) -> PathBuf {
        if p.is_absolute() {
            p
        } else {
            self.root.join(p)
        }
    }
    fn read(&mut self, op: i32, name: &str, extra: Option<&str>) -> Option<Vec<u8>> {
        let p = path(name);
        if op == 4 || op == 5 {
            let from = if op == 4 {
                p.parent().unwrap_or(Path::new(".")).to_path_buf()
            } else {
                p
            };
            return Some(
                from.join(path(extra?))
                    .to_string_lossy()
                    .as_bytes()
                    .to_vec(),
            );
        }
        let p = self.absolute(p);
        if op == 3 {
            return Some(p.to_string_lossy().as_bytes().to_vec());
        }
        let query = Query {
            op: op as u8,
            path: p,
        };
        if let Some(value) = self.observations.get(&query) {
            return value.clone();
        }
        if self.replay {
            self.unobserved.push(query);
            return None;
        }
        let value = observe(&query);
        self.observations.insert(query, value.clone());
        value
    }
}
fn library(executable: &Path) -> Result<PathBuf> {
    let executable = executable.canonicalize()?;
    let root = executable
        .parent()
        .context("Slang executable directory")?
        .parent()
        .context("Slang toolchain root")?;
    let names = if cfg!(target_os = "windows") {
        vec!["slang.dll".into(), "slang-compiler.dll".into()]
    } else if cfg!(target_os = "macos") {
        vec![
            "libslang.dylib".into(),
            format!("libslang-compiler.0.{VERSION}.dylib"),
        ]
    } else {
        vec![
            "libslang.so".into(),
            format!("libslang-compiler.so.0.{VERSION}"),
        ]
    };
    for dir in [root.join("lib"), root.join("bin")] {
        for name in &names {
            let p = dir.join(name);
            if p.is_file() {
                return Ok(p);
            }
        }
    }
    bail!(
        "pinned Slang shared library missing beside {}",
        executable.display()
    )
}
fn adapter() -> Result<PathBuf> {
    let executable = std::env::current_exe()?;
    // Resolve launch symlinks when possible. Replacing a running executable
    // (including Cargo rebuilding a test binary) can remove its old directory
    // entry; its original parent still locates the packaged companion.
    let executable = executable.canonicalize().unwrap_or(executable);
    let name = if cfg!(windows) {
        "vork-slang-filesystem.exe"
    } else {
        "vork-slang-filesystem"
    };
    let directory = executable.parent().context("executable directory")?;
    let mut packaged = directory.join(name);
    // Cargo test executables and packaged profile workers live one directory
    // below the CLI. They share that package's compiler companion.
    if !packaged.is_file()
        && matches!(
            directory.file_name().and_then(|n| n.to_str()),
            Some("examples" | "deps")
        )
    {
        packaged = directory
            .parent()
            .context("compiler companion directory")?
            .join(name);
    }
    // Installed release packages never execute a build-machine OUT_DIR path.
    #[cfg(any(debug_assertions, test))]
    if !packaged.is_file() {
        packaged = PathBuf::from(env!("VORK_SLANG_FILESYSTEM_ADAPTER"));
    }
    let adapter = packaged;
    ensure!(
        adapter.is_file(),
        "Slang filesystem adapter missing; reinstall the complete vork package"
    );
    Ok(adapter)
}

pub(super) fn compile(
    command: &Command,
    observations: Option<&Observations>,
) -> Result<(std::process::Output, Observations)> {
    let root = command
        .get_current_dir()
        .context("Slang request must have an owned working directory")?
        .to_path_buf();
    let mut state = State {
        root,
        observations: observations.cloned().unwrap_or_default(),
        replay: observations.is_some(),
        unobserved: Vec::new(),
        stdout: Vec::new(),
        stderr: Vec::new(),
        dependencies: BTreeMap::new(),
    };
    if state.replay {
        // Code generation may ask for the type/bytes of a canonical dependency
        // which capture opened through an alias. These answers follow directly
        // from the recorded identity and bytes; no new filesystem lookup occurs.
        for (query, bytes) in observations.unwrap() {
            if query.op != 0 {
                continue;
            }
            let Some(bytes) = bytes else {
                continue;
            };
            let identity = Query {
                op: 1,
                path: query.path.clone(),
            };
            let Some(Some(canonical)) = observations.unwrap().get(&identity) else {
                continue;
            };
            let canonical_path = path(std::str::from_utf8(canonical)?);
            for (op, value) in [(0, bytes.clone()), (1, canonical.clone()), (2, vec![1])] {
                state
                    .observations
                    .entry(Query {
                        op,
                        path: canonical_path.clone(),
                    })
                    .or_insert(Some(value));
            }
        }
    }
    let adapter = adapter()?;
    let mut child_command = Command::new(&adapter);
    child_command
        .current_dir(&state.root)
        .arg(compiler_path(&library(Path::new(command.get_program()))?))
        .arg(compiler_path(
            Path::new(command.get_program())
                .canonicalize()?
                .parent()
                .context("Slang compiler directory")?,
        ))
        .args(command.get_args())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = spawn_compiler(&mut child_command)?;
    let mut input = std::io::BufWriter::new(child.stdin.take().unwrap());
    let mut output = std::io::BufReader::new(child.stdout.take().unwrap());
    let mut stderr = child.stderr.take().unwrap();
    let diagnostics = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stderr.read_to_end(&mut bytes);
        bytes
    });
    fn integer(reader: &mut impl Read) -> Result<u64> {
        let mut bytes = [0; 8];
        reader.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }
    fn bytes(reader: &mut impl Read) -> Result<Vec<u8>> {
        let n = integer(reader)?;
        ensure!(n <= 512 * 1024 * 1024, "oversized Slang filesystem message");
        let mut bytes = vec![0; n as usize];
        reader.read_exact(&mut bytes)?;
        Ok(bytes)
    }
    let exchange = (|| -> Result<i32> {
        ensure!(
            integer(&mut output)? == 1,
            "unsupported Slang filesystem protocol"
        );
        loop {
            let mut tag = [0];
            output.read_exact(&mut tag)?;
            match tag[0] {
                b'R' => {
                    let op = integer(&mut output)? as i32;
                    let name = String::from_utf8(bytes(&mut output)?)?;
                    let extra = String::from_utf8(bytes(&mut output)?)?;
                    let result = state.read(op, &name, Some(&extra));
                    input.write_all(
                        &result
                            .as_ref()
                            .map_or(u64::MAX, |b| b.len() as u64)
                            .to_le_bytes(),
                    )?;
                    if let Some(bytes) = result {
                        input.write_all(&bytes)?;
                    }
                    input.flush()?;
                }
                b'E' => {
                    let channel = integer(&mut output)?;
                    let name = String::from_utf8(bytes(&mut output)?)?;
                    let bytes = bytes(&mut output)?;
                    match channel {
                        1 => state.stdout.extend(bytes),
                        2 => state.stderr.extend(bytes),
                        3 => {
                            let name = state.absolute(path(&name));
                            state
                                .dependencies
                                .entry(name)
                                .or_default()
                                .push(path(&String::from_utf8(bytes)?));
                        }
                        _ => bail!("unknown Slang output channel"),
                    }
                }
                b'D' => return Ok(integer(&mut output)? as u32 as i32),
                _ => bail!("invalid Slang filesystem protocol"),
            }
        }
    })();
    drop(input);
    if exchange.is_err() {
        let _ = child.kill();
    }
    let exit = child.wait()?;
    state.stderr.extend(diagnostics.join().unwrap_or_default());
    let status = exchange.with_context(|| {
        format!(
            "Slang filesystem adapter {exit}: {}",
            String::from_utf8_lossy(&state.stderr)
        )
    })?;
    ensure!(
        exit.success(),
        "Slang filesystem adapter {exit}: {}",
        String::from_utf8_lossy(&state.stderr)
    );
    if status < 0 && state.stderr.is_empty() {
        state
            .stderr
            .extend_from_slice(format!("Slang API returned {status:#x}").as_bytes());
    }
    ensure!(
        state.unobserved.is_empty(),
        "Slang replay requested uncaptured filesystem inputs: {:?}",
        state.unobserved
    );
    if status >= 0 {
        for (path, deps) in &state.dependencies {
            let mut text = String::from("output:");
            for dep in deps {
                text.push(' ');
                text.push_str(
                    &dep.to_string_lossy()
                        .replace('\\', "\\\\")
                        .replace(' ', "\\ ")
                        .replace('#', "\\#"),
                );
            }
            text.push('\n');
            fs::write(path, text)?;
        }
    }
    #[cfg(unix)]
    let status = {
        use std::os::unix::process::ExitStatusExt;
        std::process::ExitStatus::from_raw(if status >= 0 { 0 } else { 1 << 8 })
    };
    #[cfg(windows)]
    let status = {
        use std::os::windows::process::ExitStatusExt;
        std::process::ExitStatus::from_raw(if status >= 0 { 0 } else { 1 })
    };
    Ok((
        std::process::Output {
            status,
            stdout: state.stdout,
            stderr: state.stderr,
        },
        state.observations,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replay_rejects_unrecorded_queries() -> Result<()> {
        let compiler = Compiler::discover();
        let work = Workspace::new()?;
        let source = work.0.join("main.slang");
        fs::write(&source, "[numthreads(1,1,1)] void main(){}")?;
        let mut command = Command::new(&compiler.executable);
        command.current_dir(&work.0).arg(&source).args([
            "-target",
            "spirv",
            "-entry",
            "main",
            "-no-codegen",
        ]);
        let (_, mut observations) = compile(&command, None)?;
        observations.remove(&Query {
            op: 0,
            path: source,
        });
        let error = compile(&command, Some(&observations)).unwrap_err();
        assert!(
            error.to_string().contains("uncaptured filesystem inputs"),
            "{error:#}"
        );
        Ok(())
    }
    #[cfg(unix)]
    #[test]
    fn replaced_process_executable_keeps_companion_discovery() -> Result<()> {
        const CHILD: &str = "VORK_TEST_REPLACED_EXECUTABLE";
        if std::env::var_os(CHILD).is_some() {
            fs::remove_file(std::env::current_exe()?)?;
            let compiler = Compiler::discover();
            let modules = [("main","RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){result[id.x]=0.25;}")];
            let prepared = compiler.prepare_modules(
                &modules,
                "main",
                &[("main", "compute")],
                &["-target", "spirv"],
            )?;
            assert!(!prepared.compile()?.is_empty());
            return Ok(());
        }
        let work = Workspace::new()?;
        let executable = work.0.join("test-runner");
        // Keep writable executable descriptors out of this multithreaded
        // process: another test's fork can inherit one until exec and make
        // our immediate launch fail with ETXTBSY, even with close-on-exec.
        // Waiting for a separate copier closes its descriptors before launch.
        for (source, destination) in [
            (std::env::current_exe()?, executable.clone()),
            (adapter()?, work.0.join("vork-slang-filesystem")),
        ] {
            let copied = Command::new("cp").arg(source).arg(destination).status()?;
            ensure!(copied.success(), "copy test executable: {copied}");
        }
        let result = Command::new(executable)
            .args([
                "--exact",
                "filesystem::tests::replaced_process_executable_keeps_companion_discovery",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()?;
        ensure!(
            result.status.success(),
            "unlinked executable: {}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        Ok(())
    }

    #[test]
    fn windows_canonical_path_representation() {
        assert_eq!(
            compiler_path(Path::new(r"\\?\C:\shot café\palette.slang")),
            "C:/shot café/palette.slang"
        );
        assert_eq!(
            compiler_path(Path::new(r"\\?\UNC\server\shot\palette.slang")),
            "//server/shot/palette.slang"
        );
    }
}
