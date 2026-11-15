//! Conservative compiler-input identity. Slang owns lexing, preprocessing and
//! import resolution. Filesystem callbacks identify the reads and failed probes;
//! vork retains those observations for immutable subsequent compilation.
use super::*;

pub struct PreparedModules {
    work: Workspace,
    executable: PathBuf,
    main: PathBuf,
    args: Vec<String>,
    entries: Vec<(String, String)>,
    snapshot: Snapshot,
    identity: Vec<u8>,
}

/// A cheap exact-input validation receipt, stored by the renderer in its
/// ordinary content-addressed store. Failed resolver probes are dependencies
/// too: a new higher-priority file must force compiler resolution again.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputIdentity {
    pub identity: Vec<u8>,
    supplied_only: bool,
    // File loads are digests; path types, absence and canonical identities retain
    // their exact values. Raw source bytes live only in the in-flight snapshot.
    #[serde(with = "filesystem::observations_serde")]
    observations: filesystem::Observations,
}

/// A captured closure made entirely from supplied modules. The only external
/// observations it may carry are missing path-type probes in declared search
/// roots. Relocation rechecks those exact probes, including binary-module
/// precedence, before returning an ordinary host-local receipt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClosedInputIdentity {
    identity: Vec<u8>,
    roots: usize,
    missing: Vec<(usize, Vec<String>)>,
}
impl ClosedInputIdentity {
    pub fn relocate(&self, roots: &[PathBuf]) -> Option<InputIdentity> {
        if roots.len() != self.roots { return None; }
        let mut observations = filesystem::Observations::new();
        for (index, relative) in &self.missing {
            if relative.is_empty() || relative.iter().any(|c| c.is_empty() || matches!(c.as_str(), "." | "..") || c.contains(['/', '\\'])) { return None; }
            let path=relative.iter().fold(roots.get(*index)?.clone(),|path,part|path.join(part));
            let query = filesystem::Query { op: 2, path };
            if filesystem::observe(&query).is_some() { return None; }
            observations.insert(query, None);
        }
        Some(InputIdentity { identity: self.identity.clone(), supplied_only: true, observations })
    }
}
impl InputIdentity {
    pub fn closed(&self, roots: &[PathBuf]) -> Option<ClosedInputIdentity> {
        if !self.supplied_only { return None; }
        let mut missing = Vec::new();
        for (query, value) in &self.observations {
            if query.op != 2 || value.is_some() { return None; }
            let before = missing.len();
            // A query can belong to several overlapping/aliased roots. Retain
            // every mapping: after relocation they may name different paths.
            missing.extend(roots.iter().enumerate().filter_map(|(i, root)| {
                let relative = query.path.strip_prefix(root).ok()?;
                let parts:Option<Vec<String>>=relative.components().map(|c| match c {
                    std::path::Component::Normal(part)=>part.to_str().map(str::to_owned),_=>None,
                }).collect();
                let parts=parts?;
                (!parts.is_empty()).then_some((i,parts))
            }));
            if missing.len() == before { return None; }
        }
        Some(ClosedInputIdentity { identity: self.identity.clone(), roots: roots.len(), missing })
    }
    pub fn valid(&self) -> bool { self.current() }
    fn current(&self) -> bool {
        // Failed searches commonly probe hundreds of module names below one
        // missing include directory. Resolve that parent once for this check;
        // existing directories still use the exact original filesystem query.
        let mut parents = BTreeMap::new();
        for (query, value) in &self.observations {
            if query.op == 2 && value.is_none() && query.path.is_absolute() && let Some(parent) = query.path.parent() {
                *parents.entry(parent).or_insert(0usize) += 1;
            }
        }
        let missing = |parent: &Path| fs::metadata(parent)
            .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound);
        let absent: std::collections::BTreeSet<_> = parents.into_iter()
            .filter_map(|(parent, count)| (count >= 8 && missing(parent)).then_some(parent)).collect();
        self.observations.iter().all(|(query, value)| {
            (query.op == 2 && value.is_none() && query.path.parent().is_some_and(|parent| absent.contains(parent)))
                || filesystem::fingerprint(query, &filesystem::observe(query)) == *value
        }) && absent.iter().all(|parent| missing(parent))
        // No absence survives this validation. A new include directory, even
        // one appearing during the check, must force compiler resolution.
    }
    /// A portable proof key must cover every resolver input of code generation.
    /// Native composition separately permits target-only inputs through `verify`.
    pub fn verify_proof(&self, expected: &Self) -> Result<()> {
        self.verify(expected)?;
        ensure!(self.observations.iter().all(|(query,value)|expected.observations.get(query)==Some(value)),
            "Slang proof observed resolver inputs absent from its compiler identity; cannot cache the result");
        Ok(())
    }
    /// Compare shared imported inputs, failed probes and canonical identities.
    pub fn verify(&self, expected: &Self) -> Result<()> {
        for (query,value) in &self.observations {
            if let Some(previous) = expected.observations.get(query) {
                ensure!(value == previous, "Slang filesystem observation {} changed during compilation; retry the command", query.path.display());
            }
        }
        Ok(())
    }
}

pub(super) struct Snapshot {
    files: BTreeMap<PathBuf, Vec<u8>>,
    resolved: BTreeMap<PathBuf, PathBuf>,
    observations: filesystem::Observations,
    supplied_only: bool,
}

/// Verify the retained inputs used to produce each actual proof pass.
/// Shared imports must agree between passes, including failed search probes.
pub(super) fn check_imports(
    compiler: &Compiler,
    work: &Workspace,
    snapshot: &Snapshot,
) -> Result<()> {
    let receipt = snapshot.receipt(work, Vec::new());
    if let Some(expected) = &compiler.expected {
        receipt.verify(expected)?;
    }
    if let Some(captured) = &compiler.captured {
        let mut captured = captured.lock().unwrap();
        if let Some(previous) = captured.as_mut() {
            receipt.verify(previous)?;
            previous.supplied_only &= receipt.supplied_only;
            previous.observations.extend(receipt.observations);
        } else {
            *captured = Some(receipt);
        }
    }
    Ok(())
}

fn part(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// The compiler discovers dependencies, but never supplies serialized filesystem
/// state. Only code produced from our retained files may be published.
impl Snapshot {
    fn has_supplied_files(&self, supplied: &BTreeSet<PathBuf>) -> bool {
        self.files.keys().all(|p|supplied.contains(p))
            && self.observations.iter().filter(|(q,value)|q.op==0 && value.is_some()).all(|(q,_)|supplied.contains(&q.path))
    }
    fn compile(&self, command: &Command) -> Result<std::process::Output> {
        Ok(filesystem::compile(command, Some(&self.observations))?.0)
    }

    fn receipt(&self, work: &Workspace, identity: Vec<u8>) -> InputIdentity {
        InputIdentity {
            identity,
            supplied_only: self.supplied_only,
            observations: self.observations.iter().filter(|(q,_)| !q.path.starts_with(&work.0))
                .map(|(q,v)|(q.clone(),filesystem::fingerprint(q,v))).collect(),
        }
    }

    pub(super) fn dependencies(&self) -> Vec<PathBuf> {
        self.files.keys().cloned().collect()
    }


}

/// Compile with Slang's filesystem interface, retaining the bytes actually
/// returned and every lookup result. The same callbacks serve repeat reads from
/// memory during this pass, so its output is already tied to retained inputs.
pub(super) fn capture_command(
    work: &Workspace,
    command: &mut Command,
    depfile: &Path,
) -> Result<(std::process::Output, Snapshot)> {
    let (output, observations) = filesystem::compile(command, None)?;
    if !output.status.success() {
        let diagnostics = String::from_utf8_lossy(&output.stderr);
        // Slang owns import resolution, including nested imports and preprocessing.
        // Condense its missing-file diagnostic only for the retired bundled names.
        for line in diagnostics.lines() {
            if let Some(name) = line.strip_prefix("error[E00001]: cannot open file '")
                .and_then(|name| name.strip_suffix(".slang'"))
                && let Some(message) = vork_scene::libraries::removed_module(name) {
                bail!("{message}");
            }
        }
        if let Some(message) = enums::missing_members(&work.0, &diagnostics)? {
            bail!("Slang identity: {message}");
        }
        bail!("Slang identity: {diagnostics}");
    }
    let mut snapshot = Snapshot {
        files: BTreeMap::new(),
        resolved: BTreeMap::new(), observations, supplied_only: false,
    };
    for (query, value) in &snapshot.observations {
        if query.op == 1 && !query.path.starts_with(&work.0) {
            if let Some(bytes) = value {
                let target = PathBuf::from(std::str::from_utf8(bytes)?);
                snapshot.resolved.insert(query.path.clone(), target);
            }
        }
    }
    for path in depfile_paths(&read_compiler_text(depfile)?) {
        let bytes = snapshot.observations.get(&filesystem::Query {op:0,path:path.clone()}).and_then(Option::as_ref).or_else(|| {
            snapshot.observations.iter().find_map(|(requested, bytes)| {
                if requested.op != 0 {return None;}
                let query = filesystem::Query {op:1,path:requested.path.clone()};
                let target = snapshot.observations.get(&query)?.as_ref()?;
                (Path::new(std::str::from_utf8(target).ok()?) == path).then_some(bytes.as_ref()).flatten()
            })
        }).with_context(|| format!("Slang dependency {} was not read through its filesystem", path.display()))?;
        snapshot.files.insert(path,bytes.clone());
    }
    ensure!(snapshot.receipt(work, Vec::new()).current(), "Slang inputs changed during compilation; retry the command");
    Ok((output, snapshot))
}

/// The compiler's part of every receipt key. Only an admitted compiler has one, so no
/// receipt is looked up, reused or written for an executable that has not been checked.
impl Admitted<'_> {
    pub fn portable_input_policy(&self) -> String {
        format!("closed-filesystem-inputs-v1:{}:{INITIALIZATION}:{}",self.version(),std::env::consts::OS)
    }
    pub fn input_policy(&self) -> String {
        format!("filesystem-callbacks-v2:{}:{}", self.version(), self.compiler.executable.display())
    }
}

impl Compiler {
    /// Search roots used by the filesystem adapter, in request order. Unknown
    /// external observations cannot be exported as a closed-input receipt.
    pub fn input_roots(&self, includes: &[PathBuf]) -> Result<Vec<PathBuf>> {
        let cwd = std::env::current_dir()?;
        let mut roots: Vec<_> = includes.iter().map(|p| cwd.join(p)).collect();
        roots.push(self.executable.canonicalize()?.parent().context("Slang compiler directory")?.to_path_buf());
        // Windows canonicalize adds a verbatim prefix that the adapter removes
        // before handing paths to Slang. Compare roots in that same spelling.
        #[cfg(windows)]
        let roots = roots.iter().map(|p| PathBuf::from(filesystem::compiler_path(p))).collect();
        Ok(roots)
    }
    pub fn program_input_roots(&self, origin: &Path) -> Result<Vec<PathBuf>> {
        self.input_roots(&include_paths(origin))
    }
    pub fn with_identity(&self, identity: &InputIdentity) -> Self {
        Self {
            executable: self.executable.clone(),
            expected: Some(identity.clone()),
            captured: self.captured.clone(),
        }
    }
    /// The current exact receipt associated with a caller's normalized proof key.
    pub fn input_identity(&self) -> Option<&InputIdentity> {
        self.expected.as_ref()
    }

    /// Run a proof against checked compiler input receipts. Changes between
    /// passes or before publication fail rather than caching mismatched output.
    pub fn capture_inputs<T>(&self, build: impl FnOnce(&Compiler) -> Result<T>) -> Result<(T, InputIdentity)> {
        let captured = std::sync::Arc::new(Mutex::new(None));
        let compiler = Self {
            executable: self.executable.clone(), expected: self.expected.clone(),
            captured: Some(captured.clone()),
        };
        let result = build(&compiler)?;
        let receipt = captured.lock().unwrap().take().context("proof did not capture Slang inputs")?;
        ensure!(receipt.current(), "Slang inputs changed during compilation; retry the command");
        Ok((result, receipt))
    }

    /// Resolve and type-check a native entry without downstream optimization.
    /// Supplied modules retain conservative text identity; only reachable ones
    /// participate. External imports use compiler tokens where safe.
    pub fn prepare_modules(
        &self,
        modules: &[(&str, &str)],
        main: &str,
        entries: &[(&str, &str)],
        args: &[&str],
    ) -> Result<PreparedModules> {
        self.prepare_identity(modules, main, entries, args, None, &BTreeMap::new())
    }

    /// Generated material modules have compiler-proven input identities. Use
    /// those rather than allowing original comments in their retained source
    /// to re-enter native keys. Their generator/compiler policy is covered by
    /// the renderer's shader namespace; native-only dependencies are still
    /// resolved and keyed by this request.
    pub fn prepare_identified_modules(
        &self,
        modules: &[(&str, &str)],
        main: &str,
        entries: &[(&str, &str)],
        args: &[&str],
        identities: &BTreeMap<String, Vec<u8>>,
    ) -> Result<PreparedModules> {
        self.prepare_identity(modules, main, entries, args, None, identities)
    }

    fn prepare_identity(
        &self,
        modules: &[(&str, &str)],
        main: &str,
        entries: &[(&str, &str)],
        args: &[&str],
        normalize_root: Option<bool>,
        identities: &BTreeMap<String, Vec<u8>>,
    ) -> Result<PreparedModules> {
        let _progress = crate::progress::Job::front_end().named(main);
        self.version()?;
        let cwd = std::env::current_dir()?;
        let mut include = false;
        let absolute_args: Vec<String> = args.iter().map(|arg| {
            let value = if include { cwd.join(arg).components().collect::<PathBuf>().to_string_lossy().into_owned() } else { (*arg).to_owned() };
            include = *arg == "-I";
            value
        }).collect();
        let args: Vec<&str> = absolute_args.iter().map(String::as_str).collect();
        let args = args.as_slice();
        let work = Workspace::new()?;
        for (name, source) in modules {
            let path = work.0.join(format!("{name}.slang"));
            fs::create_dir_all(path.parent().unwrap())?;
            fs::write(path, source)?;
        }
        let main = work.0.join(format!("{main}.slang"));
        let depfile = work.0.join("capture.d");
        let mut command = Command::new(&self.executable);
        command
            .arg(&main)
            .current_dir(&work.0)
            .arg(INITIALIZATION)
            .arg("-I")
            .arg(&work.0);
        // The API reports dependencies without emitting code or a serialized
        // module. Resolve/type-check once before looking up the native artifact.
        command.args(args).args(["-no-codegen", "-dump-ast"]);
        for (entry, stage) in entries {
            command.args(["-entry", entry, "-stage", stage]);
        }
        command
                .args(["-notes-disable", "88"])
                .arg("-depfile")
                .arg(&depfile);
        let (output, mut snapshot) = capture_command(&work, &mut command, &depfile)?;
        // Exact supplied filenames also exclude case-folded or parent-path
        // aliases whose resolution could differ on the installed filesystem.
        let supplied:BTreeSet<_>=modules.iter().map(|(name,_)|work.0.join(format!("{name}.slang"))).collect();
        snapshot.supplied_only=snapshot.has_supplied_files(&supplied);
        ensure!(
            output.status.success(),
            "Slang identity: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let mut dependencies: Vec<_> = snapshot.files.keys().cloned().collect();
        dependencies.sort_by_key(|path| path.strip_prefix(&work.0).unwrap_or(path).to_path_buf());
        let mut identity = Vec::new();
        part(&mut identity, if normalize_root.is_some() { b"slang-input-filesystem-v1" } else { b"slang-input-filesystem-v1-dependencies" });
        part(&mut identity, VERSION.as_bytes());
        part(&mut identity, INITIALIZATION.as_bytes());
        let external = snapshot.files.keys().any(|path| !path.starts_with(&work.0));
        let mut options = args.iter();
        while let Some(arg) = options.next() {
            // A wholly embedded closure does not depend on unused external
            // search directories. This keeps bundled programs portable across
            // working directories. Exact-input receipts still validate lookups.
            if !external && *arg == "-I" {
                options.next();
                continue;
            }
            part(&mut identity, arg.as_bytes());
        }
        for (entry, stage) in entries {
            part(&mut identity, entry.as_bytes());
            part(&mut identity, stage.as_bytes());
        }
        for path in dependencies {
            let bytes = snapshot
                .files
                .get(&path)
                .with_context(|| format!("Slang snapshot missing {}", path.display()))?;
            let supplied = path.strip_prefix(&work.0).ok();
            part(
                &mut identity,
                supplied.unwrap_or(&path).to_string_lossy().as_bytes(),
            );
            if let Some(relative) = supplied {
                let name = relative
                    .with_extension("")
                    .to_string_lossy()
                    .replace('\\', "/");
                if let Some(proven) = identities.get(&name) {
                    part(&mut identity, b"compiler-proven-material-module");
                    part(&mut identity, proven);
                    continue;
                }
            }
            let root = path == main;
            // -E is a token stream, not a serialization of preprocessor state.
            // Directives may affect that state without emitting tokens. Keep
            // their raw identity, and do the same for any diagnostic. This is
            // deliberately conservative (even a '#' in a string falls back).
            // Proof diagnostics are part of the cached Program. Native cache
            // payloads contain code only: an unrelated renderer warning must
            // not make a comment in an imported helper invalidate that code.
            let eligible = (normalize_root.is_none() || output.stderr.is_empty())
                && if root && normalize_root.is_some() {
                    normalize_root.unwrap()
                } else {
                    supplied.is_none() && !bytes.contains(&b'#')
                };
            if supplied.is_none() {
                reject_reserved(std::str::from_utf8(bytes)?, &path.display().to_string())?;
            }
            if eligible {
                let mut command = Command::new(&self.executable);
                command
                    // Dependencies include headers whose extensions do not
                    // identify a language when preprocessed on their own.
                    .args(["-lang", "slang"])
                    .arg(&path)
                    .current_dir(&work.0)
                    .arg(INITIALIZATION)
                    .arg("-I")
                    .arg(&work.0)
                    .args(args)
                    .args(["-E", "-line-directive-mode", "none"]);
                let tokens = snapshot.compile(&command)?;
                ensure!(
                    tokens.status.success(),
                    "preprocess Slang identity: {}",
                    String::from_utf8_lossy(&tokens.stderr)
                );
                if tokens.stderr.is_empty() {
                    part(&mut identity, &tokens.stdout);
                } else {
                    part(&mut identity, bytes);
                }
            } else {
                part(&mut identity, bytes);
            }
        }
        let mut bindings: Vec<_> = snapshot
            .resolved
            .iter()
            .map(|(path, target)| {
                (
                    path.strip_prefix(&work.0).unwrap_or(path).to_path_buf(),
                    target.strip_prefix(&work.0).unwrap_or(target).to_path_buf(),
                )
            })
            .collect();
        bindings.sort();
        for (path, target) in bindings {
            part(&mut identity, path.to_string_lossy().as_bytes());
            part(&mut identity, target.to_string_lossy().as_bytes());
        }
        Ok(PreparedModules {
            work,
            executable: self.executable.clone(),
            main,
            args: args.iter().map(|s| s.to_string()).collect(),
            entries: entries
                .iter()
                .map(|(e, s)| (e.to_string(), s.to_string()))
                .collect(),
            snapshot,
            identity: blake3::hash(&identity).as_bytes().to_vec(),
        })
    }

    /// Surface proofs share the same normalized root and resolved imports.
    /// The returned compiler checks every proof subprocess's captured imports
    /// against this snapshot before returning a publishable result.
    pub fn prepare_program(
        &self,
        abi: &Abi,
        source: &str,
        origin: &Path,
    ) -> Result<PreparedProgram> {
        self.prepare_bound_program(abi,source,origin,"")
    }

    /// Generated compute bindings are trusted compiler input, separate from
    /// authored source so its reserved-name checks stay identical to compilation.
    pub fn prepare_bound_program(
        &self, abi: &Abi, source: &str, origin: &Path, bindings: &str,
    ) -> Result<PreparedProgram> {
        let work = Workspace::new()?;
        write_modules(&work, abi, source)?;
        let mut modules = Vec::new();
        fn visit(root: &Path, path: &Path, modules: &mut Vec<(String, String)>) -> Result<()> {
            for entry in fs::read_dir(path)? {
                let path = entry?.path();
                if path.is_dir() {
                    visit(root, &path, modules)?;
                } else {
                    modules.push((
                        path.strip_prefix(root)?
                            .with_extension("")
                            .to_string_lossy()
                            .replace('\\', "/"),
                        fs::read_to_string(path)?,
                    ));
                }
            }
            Ok(())
        }
        visit(&work.0, &work.0, &mut modules)?;
        let program=program_text(source,origin);
        let program=if bindings.is_empty() {program} else {format!("import vork;\n{bindings}\n{program}")};
        modules.push(("program".into(), program));
        let includes = include_paths(origin);
        let mut args = vec!["-target", "wgsl"];
        let include_strings: Vec<_> = includes
            .iter()
            .map(|p| std::env::current_dir().map(|cwd| cwd.join(p).display().to_string()))
            .collect::<std::io::Result<_>>()?;
        for include in &include_strings {
            args.extend(["-I", include]);
        }
        let refs: Vec<_> = modules
            .iter()
            .map(|(n, s)| (n.as_str(), s.as_str()))
            .collect();
        let prepared = self.prepare_identity(
            &refs,
            "program",
            &[],
            &args,
            Some(!source.contains('#') && !bindings.contains('#')),
            &BTreeMap::new(),
        )?;
        let expected = prepared.receipt();
        Ok(PreparedProgram {
            compiler: Compiler {
                executable: self.executable.clone(),
                expected: Some(expected),
                captured: self.captured.clone(),
            },
            prepared,
        })
    }
}

pub struct PreparedProgram {
    compiler: Compiler,
    prepared: PreparedModules,
}
impl PreparedProgram {
    pub fn identity(&self) -> &[u8] {
        self.prepared.identity()
    }
    pub fn compiler(&self) -> &Compiler {
        &self.compiler
    }
    pub fn receipt(&self) -> InputIdentity {
        self.prepared.receipt()
    }
}

impl PreparedModules {
    pub fn identity(&self) -> &[u8] {
        &self.identity
    }
    pub fn receipt(&self) -> InputIdentity {
        self.snapshot.receipt(&self.work, self.identity.clone())
    }
    pub fn compile(&self) -> Result<Vec<u8>> {
        let _progress = crate::progress::Job::start(false).named(&self.main.file_stem().unwrap_or_default().to_string_lossy());
        let output = self.work.0.join("out");
        let mut command = Command::new(&self.executable);
        // Replay source with the immutable filesystem observations. Slang's
        // serialized modules rewrite import spellings and can misbind distinct
        // imports sharing a leaf name; they are not used as filesystem receipts.
        command
            .current_dir(&self.work.0)
            .arg(INITIALIZATION)
            .arg("-I")
            .arg(&self.work.0)
            .args(&self.args)
            .arg(&self.main);
        for (entry, stage) in &self.entries {
            command.args(["-entry", entry, "-stage", stage]);
        }
        let metal = self.args.windows(2).any(|a| a == ["-target", "metal"]);
        if metal && !self.args.iter().any(|a| a == "-line-directive-mode") {
            command.args(["-line-directive-mode", "none"]);
        }
        let depfile = self.work.0.join("native.d");
        command.arg("-depfile").arg(&depfile).arg("-o").arg(&output);
        let result = self.snapshot.compile(&command)?;
        ensure!(
            result.status.success(),
            "slangc {}: {}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        );
        let allowed: BTreeSet<_> = self.snapshot.files.keys().cloned().collect();
        ensure!(depfile_paths(&read_compiler_text(&depfile)?).iter().all(|p| allowed.contains(p)),
            "Slang code generation escaped its captured dependency closure; cannot cache the result");
        let mut code = fs::read(output)?;
        if metal {crate::metal_source::validate(std::str::from_utf8(&code)?)?;}
        if metal
            && self.args.windows(2).any(|a| {
                matches!(a[0].as_str(), "-fp-mode" | "-floating-point-mode") && a[1] == "precise"
            })
        {
            let mut prefix =
                b"#pragma METAL fp math_mode(safe)\n#pragma METAL fp contract(off)\n".to_vec();
            prefix.append(&mut code);
            code = prefix;
        }
        Ok(code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identified_modules_sweep_workspace_path_lengths() -> Result<()> {
        const CHILD: &str = "VORK_TEST_SNAPSHOT_PATH_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let compiler = Compiler::discover();
            let material = "public float value(){return 0.25;}";
            let main = "import material; RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){result[id.x]=value();}";
            for target in ["spirv", "metal"] {
                let ids = BTreeMap::from([("material".into(), b"proven-material".to_vec())]);
                let prepared = compiler.prepare_identified_modules(
                    &[("main", main), ("material", material)], "main",
                    &[("main", "compute")], &["-target", target], &ids)?;
                assert!(!prepared.compile()?.is_empty());
                assert!(prepared.receipt().valid());
            }
            return Ok(());
        }
        // Each subprocess gets its own TMPDIR (and Windows TEMP/TMP); never
        // mutate the multithreaded test runner's process environment.
        let root = Workspace::new()?;
        for length in 1..=40 {
            let directory = root.0.join("x".repeat(length));
            fs::create_dir(&directory)?;
            let output = Command::new(std::env::current_exe()?)
                .args(["--exact", "identity::tests::identified_modules_sweep_workspace_path_lengths", "--nocapture"])
                .env(CHILD, "1").env("TMPDIR", &directory)
                .env("TEMP", &directory).env("TMP", &directory).output()?;
            ensure!(output.status.success(), "workspace path length {length}: {}{}",
                String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        }
        Ok(())
    }

    #[test]
    fn nested_import_and_textual_include_probes_track_new_files() -> Result<()> {
        let compiler = Compiler::discover();
        let external = Workspace::new()?;
        let first = external.0.join("first");
        let second = external.0.join("second");
        fs::create_dir_all(first.join("nested"))?;
        fs::create_dir_all(second.join("nested"))?;
        fs::write(second.join("nested/helper.slang"), "#include \"value.h\"\npublic float value(){return VALUE;}")?;
        fs::write(second.join("nested/value.h"), "#define VALUE 0.25\n")?;
        let first_text = first.display().to_string();
        let second_text = second.display().to_string();
        let modules = [("main", "import nested.helper; RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){result[id.x]=value();}")];
        for target in ["spirv", "metal"] {
            let args = ["-target", target, "-I", &first_text, "-I", &second_text];
            let before = compiler.prepare_modules(&modules, "main", &[("main", "compute")], &args)?;
            let code = before.compile()?;
            assert!(before.receipt().valid());
            fs::write(first.join("nested/helper.slang"), "public float value(){return 0.75;}")?;
            assert!(!before.receipt().valid());
            let after = compiler.prepare_modules(&modules, "main", &[("main", "compute")], &args)?;
            assert_ne!(before.identity(), after.identity());
            assert_ne!(code, after.compile()?);
            assert_eq!(code, before.compile()?);
            fs::remove_file(first.join("nested/helper.slang"))?;
        }
        Ok(())
    }

    #[test]
    fn proof_capture_rejects_import_changes_between_passes() -> Result<()> {
        let work=Workspace::new()?;
        let helper=work.0.join("helper.slang");
        fs::write(&helper,"public float value(){return 0.25;}")?;
        let origin=work.0.join("main.slang");
        let source="import helper; struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs inputs){return finish_view(float3(value()),ctx,1.);}";
        let compiler=Compiler::discover();
        let result=compiler.capture_inputs(|compiler| {
            compiler.compile_display(&Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() },source,&origin,"first")?;
            fs::write(&helper,"public float value(){return 0.75;}")?;
            compiler.compile_display(&Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() },source,&origin,"second")
        });
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("changed during compilation"), "{error}");
        Ok(())
    }

    const SOURCE: &str = "struct Inputs {}; Closure surface(ShadingContext ctx, Inputs inputs) { Closure c; c.base_color=float3(value()); return c; }";

    #[test]
    fn same_leaf_library_imports_replay_captured_source() -> Result<()> {
        let compiler = Compiler::discover();
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        let vork = abi.module();
        let modules = [
            ("vork", vork.as_str()),
            ("main", "import std.color; import blender.color; RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){result[id.x]=mix(blackbody(6500.),float3(.2)).x;}"),
            ("std/color", module_source("std/color").unwrap()),
            ("blender/color", module_source("blender/color").unwrap()),
        ];
        for target in ["spirv", "metal"] {
            let args = ["-target", target, "-O2", "-fp-mode", "precise"];
            let entries = [("main", "compute")];
            let prepared = compiler.prepare_modules(&modules, "main", &entries, &args)?;
            assert_eq!(prepared.compile()?, compiler.compile_modules(&modules, "main", &entries, &args)?, "{target}: captured replay must preserve imported functions");
        }
        Ok(())
    }

    #[test]
    fn missing_search_parents_are_rechecked() -> Result<()> {
        let work = Workspace::new()?;
        let parent = work.0.join("new-include");
        let receipt = InputIdentity {
            identity: Vec::new(), supplied_only: true,
            observations: (0..16).map(|i| (filesystem::Query {op: 2, path: parent.join(format!("module{i}.slang"))}, None)).collect(),
        };
        assert!(receipt.valid());
        fs::create_dir(&parent)?;
        fs::write(parent.join("module7.slang"), "public int value=7;")?;
        assert!(!receipt.valid(), "a newly populated include directory must invalidate every subsequent check");
        fs::remove_file(parent.join("module7.slang"))?;
        assert!(receipt.valid(), "existing directories retain exact failed probes");
        #[cfg(unix)] {
            fs::remove_dir(&parent)?;
            let target = work.0.join("target");
            fs::create_dir(&target)?;
            std::os::unix::fs::symlink(&target, &parent)?;
            assert!(receipt.valid());
            fs::write(target.join("module7.slang"), "public int value=7;")?;
            assert!(!receipt.valid(), "include-directory symlinks retain ordinary filesystem resolution");
        }
        Ok(())
    }

    #[test]
    fn checked_native_module_matches_source_codegen() -> Result<()> {
        let compiler = Compiler::discover();
        let modules = [
            ("main", "import helper; RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){float x; result[id.x]=value()+x;}"),
            ("helper", "public float value(){return 0.25;}"),
        ];
        for target in ["spirv", "metal"] {
            let args = ["-target", target, "-O2", "-fp-mode", "precise"];
            let entries = [("main", "compute")];
            let prepared = compiler.prepare_modules(&modules, "main", &entries, &args)?;
            assert_eq!(prepared.compile()?, compiler.compile_modules(&modules, "main", &entries, &args)?, "{target}: checked module must preserve codegen, including initialization and precise math");
        }
        Ok(())
    }

    #[test]
    fn program_tokens_reuse_comments_but_invalidate_real_and_imported_changes() -> Result<()> {
        let compiler = Compiler::discover();
        let work = Workspace::new()?;
        let origin = work.0.join("material.slang");
        let helper = work.0.join("helper.slang");
        fs::write(&helper, "public float value(){return 0.25;}\n")?;
        let source = format!("import helper; {SOURCE}");
        let abi = Abi {
            aov_count: 0,
            context: Abi::spec_context(),
            closure: openpbr_fields(),
        };
        let base = compiler.prepare_program(&abi, &source, &origin)?;
        let comment = compiler.prepare_program(
            &abi,
            &format!("// changed description\n{source}\n/* more */"),
            &origin,
        )?;
        assert!(
            base.identity() == comment.identity(),
            "root comment must hit"
        );
        fs::write(
            &helper,
            "// imported description\npublic float value() { return 0.25; }\n",
        )?;
        let comment = compiler.prepare_program(&abi, &source, &origin)?;
        assert!(
            base.identity() == comment.identity(),
            "imported comment must hit"
        );
        fs::write(&helper, "public float value(){return 0.75;}\n")?;
        let edited = compiler.prepare_program(&abi, &source, &origin)?;
        assert!(
            base.identity() != edited.identity(),
            "imported colour edit must miss"
        );
        let edited = compiler.prepare_program(
            &abi,
            &source.replace("float3(value())", "float3(0.5)"),
            &origin,
        )?;
        assert!(
            comment.identity() != edited.identity(),
            "root colour edit must miss"
        );
        fs::remove_file(&helper)?;
        assert!(
            compiler.prepare_program(&abi, &source, &origin).is_err(),
            "deleted import cannot hit"
        );
        fs::write(&helper, "public float value(){return 0.25;}\n")?;
        assert_eq!(
            base.identity(),
            compiler.prepare_program(&abi, &source, &origin)?.identity()
        );
        assert_ne!(
            base.identity(),
            compiler
                .prepare_program(&abi, &format!("#warning check me\n{source}"), &origin)?
                .identity(),
            "diagnostic cannot disappear behind a hit"
        );
        Ok(())
    }

    #[test]
    fn native_snapshot_tracks_resolution_and_codegen_uses_captured_bytes() -> Result<()> {
        let compiler = Compiler::discover();
        for target in ["spirv", "metal"] {
        let work = Workspace::new()?;
        let first = work.0.join("first");
        let second = work.0.join("second");
        fs::create_dir(&first)?;
        fs::create_dir(&second)?;
        let helper = second.join("helper.slang");
        fs::write(&helper, "public float value(){return 0.25;}")?;
        let main = "import helper; RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){result[id.x]=value();}";
        let modules = [
            ("main", main),
            ("unused", "public float unrelated(){return 1.;}"),
        ];
        let first_text = first.display().to_string();
        let second_text = second.display().to_string();
        let args = ["-target", target, "-I", &first_text, "-I", &second_text];
        let base = compiler.prepare_modules(&modules, "main", &[("main", "compute")], &args)?;
        let binary = base.compile()?;
        let mut artifacts = BTreeMap::new();
        artifacts.insert(base.identity().to_vec(), binary.clone());
        assert!(base.receipt().valid());
        fs::write(&helper, "// comment\npublic float value(){return 0.25;}")?;
        assert!(
            !base.receipt().valid(),
            "changed imported bytes must revalidate with the compiler"
        );
        let comment = compiler.prepare_modules(
            &[
                ("main", main),
                ("unused", "public float unrelated(){return 9.;}"),
            ],
            "main",
            &[("main", "compute")],
            &args,
        )?;
        assert!(
            base.identity() == comment.identity(),
            "unused modules and imported comments must hit"
        );
        assert_eq!(
            artifacts.get(comment.identity()),
            Some(&binary),
            "comment edit reuses the stored artifact"
        );
        fs::write(&helper, "public float value(){return 0.75;}")?;
        let edited = compiler.prepare_modules(&modules, "main", &[("main", "compute")], &args)?;
        assert_ne!(base.identity(), edited.identity());
        assert!(
            !artifacts.contains_key(edited.identity()),
            "real imported edit must miss"
        );
        assert_ne!(
            binary,
            edited.compile()?,
            "semantic import edit must produce a new artifact"
        );
        assert_eq!(
            binary,
            base.compile()?,
            "concurrent edit cannot change the keyed snapshot's code"
        );
        fs::write(
            first.join("helper.slang"),
            "public float value(){return 0.5;}",
        )?;
        assert!(
            !edited.receipt().valid(),
            "a previously failed resolution probe is now a file"
        );
        let shadowed = compiler.prepare_modules(&modules, "main", &[("main", "compute")], &args)?;
        assert_ne!(
            edited.identity(),
            shadowed.identity(),
            "new higher-priority file must invalidate"
        );
        assert_ne!(edited.compile()?, shadowed.compile()?);
        }
        Ok(())
    }

    #[test]
    fn proven_generated_modules_do_not_reintroduce_original_comments() -> Result<()> {
        let compiler = Compiler::discover();
        for target in ["spirv", "metal"] {
        let work = Workspace::new()?;
        let abi = Abi {
            aov_count: 0,
            context: Abi::spec_context(),
            closure: openpbr_fields(),
        };
        let origin = work.0.join("material.slang");
        let material = "public float value(){return 0.25;}";
        let main = "import material; RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){result[id.x]=value();}";
        let proof = compiler.prepare_program(&abi, material, &origin)?;
        let ids = BTreeMap::from([("material".into(), proof.identity().to_vec())]);
        let args = ["-target", target];
        let base = compiler.prepare_identified_modules(
            &[("main", main), ("material", material)],
            "main",
            &[("main", "compute")],
            &args,
            &ids,
        )?;
        let binary = base.compile()?;
        let comment = format!("// retained source after a proof-cache rebuild\n{material}");
        let comment_proof = compiler.prepare_program(&abi, &comment, &origin)?;
        assert_eq!(proof.identity(), comment_proof.identity());
        let comment = compiler.prepare_identified_modules(
            &[("main", main), ("material", &comment)],
            "main",
            &[("main", "compute")],
            &args,
            &ids,
        )?;
        assert_eq!(base.identity(), comment.identity());
        let edit = material.replace("0.25", "0.75");
        let edited_proof = compiler.prepare_program(&abi, &edit, &origin)?;
        let ids = BTreeMap::from([("material".into(), edited_proof.identity().to_vec())]);
        let edited = compiler.prepare_identified_modules(
            &[("main", main), ("material", &edit)],
            "main",
            &[("main", "compute")],
            &args,
            &ids,
        )?;
        assert_ne!(base.identity(), edited.identity());
        assert_ne!(binary, edited.compile()?);
        }
        Ok(())
    }

    #[test]
    fn unrelated_native_diagnostics_do_not_invalidate_imported_comments() -> Result<()> {
        let compiler = Compiler::discover();
        let work = Workspace::new()?;
        let helper = work.0.join("helper.slang");
        fs::write(&helper, "public float value(){return 0.25;}")?;
        let include = work.0.display().to_string();
        let args = ["-target", "spirv", "-I", &include];
        let main = "#warning renderer diagnostic\nimport helper; RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){result[id.x]=value();}";
        let base =
            compiler.prepare_modules(&[("main", main)], "main", &[("main", "compute")], &args)?;
        fs::write(
            &helper,
            "// imported description\npublic float value(){return 0.25;}",
        )?;
        let comment =
            compiler.prepare_modules(&[("main", main)], "main", &[("main", "compute")], &args)?;
        assert_eq!(base.identity(), comment.identity());
        Ok(())
    }

    #[test]
    fn target_only_imports_invalidate_native_artifacts_on_both_targets() -> Result<()> {
        let compiler = Compiler::discover();
        let work = Workspace::new()?;
        fs::write(
            work.0.join("helper.slang"),
            "#if defined(NATIVE_LINK)\nimport leaf;\npublic float value(){return leaf_value();}\n#else\npublic float value(){return 0.25;}\n#endif\n",
        )?;
        let leaf = work.0.join("leaf.slang");
        let origin = work.0.join("material.slang");
        let source = format!("import helper; {SOURCE}");
        let abi = Abi {
            aov_count: 0,
            context: Abi::spec_context(),
            closure: openpbr_fields(),
        };
        let proof = compiler.prepare_program(&abi, &source, &origin)?;
        let main = "import helper; RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){result[id.x]=value();}";
        let include = work.0.display().to_string();
        for target in ["spirv", "metal"] {
            fs::write(&leaf, "public float leaf_value(){return 0.25;}")?;
            let args = ["-target", target, "-DNATIVE_LINK=1", "-I", &include];
            let before = compiler.prepare_modules(
                &[("main", main)],
                "main",
                &[("main", "compute")],
                &args,
            )?;
            let binary = before.compile()?;
            fs::write(&leaf, "public float leaf_value(){return 0.75;}")?;
            assert!(!before.receipt().valid());
            let after = compiler.prepare_modules(
                &[("main", main)],
                "main",
                &[("main", "compute")],
                &args,
            )?;
            assert_ne!(before.identity(), after.identity());
            assert_ne!(
                binary,
                after.compile()?,
                "{target}: native-only import must change code"
            );
            assert_eq!(
                proof.identity(),
                compiler.prepare_program(&abi, &source, &origin)?.identity(),
                "the WGSL proof alone does not see this dependency"
            );
        }
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn imported_program_ignores_unreadable_descendants() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let compiler = Compiler::discover();
        let external = Workspace::new()?;
        fs::write(external.0.join("helper.slang"), "public float value(){return 0.25;}")?;
        std::os::unix::fs::symlink("loop", external.0.join("loop"))?;
        let include = external.0.to_string_lossy();
        let modules = [("main", "import helper; RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){result[id.x]=value();}")];
        for layout in ["unreadable", "one/two/unreadable", "home/Library"] {
            let blocked = external.0.join(layout);
            fs::create_dir_all(&blocked)?;
            fs::set_permissions(&blocked, fs::Permissions::from_mode(0))?;
            let result = (|| -> Result<()> {
                for target in ["spirv", "metal"] {
                    let args = ["-target", target, "-I", &include];
                    let prepared = compiler.prepare_modules(&modules, "main", &[("main", "compute")], &args)?;
                    assert!(prepared.receipt().valid());
                    assert!(!prepared.compile()?.is_empty());
                }
                Ok(())
            })();
            fs::set_permissions(&blocked, fs::Permissions::from_mode(0o700))?;
            result?;
        }
        Ok(())
    }

    #[test]
    fn relative_program_origins_and_native_search_paths_use_callers_directory() -> Result<()> {
        let compiler = Compiler::discover();
        let unique = Workspace::new()?;
        let local = Workspace(std::env::current_dir()?.join(unique.0.file_name().unwrap()));
        fs::create_dir(&local.0)?;
        fs::write(local.0.join("helper.slang"), "public float value(){return 0.25;}")?;
        let relative = local.0.file_name().unwrap().to_str().unwrap();
        let modules = [("main", "import helper; RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){result[id.x]=value();}")];
        for target in ["spirv", "metal"] {
            let prepared = compiler.prepare_modules(&modules, "main", &[("main", "compute")], &["-target", target, "-I", relative])?;
            assert!(prepared.receipt().valid());
            assert!(!prepared.compile()?.is_empty());
        }
        assert_eq!(include_paths(Path::new("material.slang")), [PathBuf::from(".")]);
        let source = format!("import \"{relative}/helper.slang\"; {SOURCE}");
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        compiler.prepare_program(&abi, &source, Path::new("material.slang"))?;
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn file_symlink_names_are_captured_and_probed() -> Result<()> {
        let compiler = Compiler::discover();
        let external = Workspace::new()?;
        let first = external.0.join("first");
        let second = external.0.join("second");
        fs::create_dir(&first)?;
        fs::create_dir(&second)?;
        let implementation = external.0.join("implementation.slang");
        fs::write(&implementation, "public float value(){return 0.25;}")?;
        std::os::unix::fs::symlink(&implementation, second.join("helper.slang"))?;
        let first_text = first.display().to_string();
        let second_text = second.display().to_string();
        let modules = [("main", "import helper; RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){result[id.x]=value();}")];
        for target in ["spirv", "metal"] {
            let args = ["-target", target, "-I", &first_text, "-I", &second_text];
            let before = compiler.prepare_modules(&modules, "main", &[("main", "compute")], &args)?;
            let code = before.compile()?;
            assert!(before.receipt().valid());
            fs::write(first.join("helper.slang"), "public float value(){return 0.75;}")?;
            assert!(!before.receipt().valid());
            let after = compiler.prepare_modules(&modules, "main", &[("main", "compute")], &args)?;
            assert_ne!(before.identity(), after.identity());
            assert_ne!(code, after.compile()?);
            assert_eq!(code, before.compile()?);
            fs::remove_file(first.join("helper.slang"))?;
        }
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn changing_a_symlink_revalidates_even_when_its_bytes_match() -> Result<()> {
        let compiler = Compiler::discover();
        let work = Workspace::new()?;
        let first = work.0.join("one");
        let second = work.0.join("two");
        fs::create_dir(&first)?;
        fs::create_dir(&second)?;
        for directory in [&first, &second] {
            fs::write(
                directory.join("helper.slang"),
                "public float value(){return 0.25;}",
            )?;
        }
        let link = work.0.join("selected");
        std::os::unix::fs::symlink(&first, &link)?;
        let include = link.display().to_string();
        let args = ["-target", "spirv", "-I", &include];
        let main = "import helper; RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){result[id.x]=value();}";
        let base =
            compiler.prepare_modules(&[("main", main)], "main", &[("main", "compute")], &args)?;
        assert!(base.receipt().valid());
        let binary = base.compile()?;
        fs::remove_file(&link)?;
        std::os::unix::fs::symlink(&second, &link)?;
        assert!(
            !base.receipt().valid(),
            "matching file bytes do not prove the same resolution"
        );
        fs::write(
            second.join("helper.slang"),
            "public float value(){return 0.75;}",
        )?;
        let edited =
            compiler.prepare_modules(&[("main", main)], "main", &[("main", "compute")], &args)?;
        assert_ne!(base.identity(), edited.identity());
        assert_ne!(binary, edited.compile()?);
        Ok(())
    }

    #[test]
    fn proof_cannot_publish_against_an_old_dependency_snapshot() -> Result<()> {
        let compiler = Compiler::discover();
        let work = Workspace::new()?;
        fs::write(
            work.0.join("helper.slang"),
            "public float value(){return 0.25;}",
        )?;
        let source = format!("import helper; {SOURCE}");
        let origin = work.0.join("material.slang");
        let abi = Abi {
            aov_count: 0,
            context: Abi::spec_context(),
            closure: openpbr_fields(),
        };
        let prepared = compiler.prepare_program(&abi, &source, &origin)?;
        fs::write(
            work.0.join("helper.slang"),
            "public float value(){return 0.75;}",
        )?;
        let error = prepared
            .compiler()
            .stages(&abi, &source, &origin)
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("changed during compilation"),
            "{error:#}"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "identity_matrix.rs"]
mod matrix;

#[cfg(test)]
mod closed_input_tests {
    use super::*;

    #[test]
    fn closed_input_rejects_filesystem_dependent_supplied_aliases() {
        let path=PathBuf::from("/workspace/helper.slang");
        let supplied=[path.clone()].into();
        let mut snapshot=Snapshot {files:[(path.clone(),vec![])].into(),resolved:Default::default(),
            observations:[(filesystem::Query {op:0,path},Some(vec![]))].into(),supplied_only:false};
        assert!(snapshot.has_supplied_files(&supplied));
        for alias in ["/workspace/HELPER.slang","/workspace/nested/../helper.slang","/external/helper.slang"] {
            snapshot.observations.clear();
            snapshot.observations.insert(filesystem::Query {op:0,path:PathBuf::from(alias)},Some(vec![]));
            assert!(!snapshot.has_supplied_files(&supplied),"captured alias {alias} must be resolved again");
        }
    }

    #[test]
    fn closed_input_requires_only_mapped_absent_type_probes() {
        let roots=vec![PathBuf::from("/build/includes")];
        let query=filesystem::Query {op:2,path:roots[0].join("helper.slang-module")};
        let mut input=InputIdentity {identity:vec![1,2,3],supplied_only:true,observations:[(query.clone(),None)].into()};
        assert!(input.closed(&roots).is_some());
        input.supplied_only=false;
        assert!(input.closed(&roots).is_none(),"every loaded dependency must be supplied");
        input.supplied_only=true;
        input.observations.insert(query.clone(),Some(vec![1]));
        assert!(input.closed(&roots).is_none(),"existing external files are not closed");
        input.observations.clear();
        input.observations.insert(filesystem::Query {op:0,..query.clone()},None);
        assert!(input.closed(&roots).is_none(),"even absent file loads require ordinary capture");
        input.observations.clear();input.observations.insert(query,None);
        assert!(input.closed(&[PathBuf::from("/elsewhere")]).is_none());
    }

    #[test]
    fn closed_input_rechecks_each_relocated_root_and_rejects_escape() -> Result<()> {
        let root=std::env::temp_dir().join(format!("vork-closed-input-{}",std::process::id()));
        let roots=vec![root.join("first"),root.join("second")];
        for path in &roots {fs::create_dir_all(path)?;}
        // Identical source roots can become distinct after relocation. Neither
        // occurrence may lose its absent-file proof.
        let old=PathBuf::from("/closed-input-build");
        let input=InputIdentity {identity:vec![4,5],supplied_only:true,observations:[(filesystem::Query {op:2,path:old.join("helper.slang-module")},None)].into()};
        let closed=input.closed(&[old.clone(),old]).unwrap();
        assert_eq!(closed.missing.len(),2);
        assert_eq!(closed.relocate(&roots).unwrap().identity,input.identity);
        for path in &roots {
            let file=path.join("helper.slang-module");fs::write(&file,b"new module")?;
            assert!(closed.relocate(&roots).is_none(),"new import precedence must miss");
            fs::remove_file(file)?;
        }
        assert!(closed.relocate(&roots[..1]).is_none());
        let mut invalid=closed.clone();invalid.missing[0].1=vec!["..".into(),"escape".into()];
        assert!(invalid.relocate(&roots).is_none());
        invalid.missing[0].1=vec!["/absolute".into()];assert!(invalid.relocate(&roots).is_none());
        invalid.missing[0].1=vec!["nested\\helper".into()];assert!(invalid.relocate(&roots).is_none());
        invalid.missing[0]=(9,vec!["helper".into()]);assert!(invalid.relocate(&roots).is_none());
        fs::remove_dir_all(root)?;Ok(())
    }
}
