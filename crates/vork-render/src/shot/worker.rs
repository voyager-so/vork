//! A kept-alive worker: one device, one open document, and an installation that
//! survives edits proven not to change installed facts.
//!
//! The worker owns what a fresh process would redo: the `Renderer` (device,
//! pipeline caches, compiler) and the current document's `InstalledScene`. Each
//! request renders through the ordinary shot pipeline, which checks the delivery
//! shape and resolved settings itself (`Shot::render_retained`). Editing is
//! resend-and-compare: `open` again with the whole document, and the worker
//! compares it with the open one. A changed light energy/colour or static leaf
//! camera placement is written into the installed records; anything else drops
//! the installation so the next render reinstalls on the retained device. Never
//! patch a shader specialization fact: when in doubt, reinstall.
//!
//! A 2D canvas's installation is its document and GPU side (`canvas::Kept`): the
//! same document renders again without device, pipeline or document setup; any
//! change to it or to its files builds both again on the retained device.
use super::{Retained, Shot};
use crate::{InstalledScene, Renderer};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use vork_scene::{Node, NodeKind, Scene};

/// What an `open` kept, or how a render obtained its installation. The receipt
/// key is `REUSE_KEY`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reuse {
    /// The retained installation is kept as it is: the document is identical.
    Reused,
    /// The retained installation, with changed values written into it.
    Updated,
    /// The installation was dropped (or none was retained); the next render installs on the retained device.
    Reinstalled,
    /// Reinstalled, and a program body changed, so the install compiles.
    Recompiled,
}
impl Reuse {
    pub fn name(self) -> &'static str {
        match self {
            Self::Reused => "reused",
            Self::Updated => "updated",
            Self::Reinstalled => "reinstalled",
            Self::Recompiled => "recompiled",
        }
    }
}
/// The receipt field carrying a `Reuse` name, in `open` and `render` replies.
/// Named here once (Choice 9: `reuse`).
pub const REUSE_KEY: &str = "reuse";

/// The whole document an `open` installs: a file, or an in-memory document with
/// the directory its relative files resolve against.
pub enum Source<'a> {
    File(&'a Path),
    Document { document: serde_json::Value, base: &'a Path },
}

/// What an `open` did.
#[derive(Clone, Debug, PartialEq)]
pub struct Opened {
    pub revision: u64,
    pub reuse: Reuse,
    /// Authored paths that differ from the previous document, in the
    /// animation-track grammar (`["nodes","key","energy"]`), with a number for an
    /// unnamed list position and `["file", relative]` for a changed external file.
    pub changed: Vec<serde_json::Value>,
}

pub struct Worker {
    renderer: Option<Renderer>,
    open: Option<Open>,
    /// Installations renders have drawn from.
    installs: u64,
    /// The last render's options: a candidate is prepared under their delivery
    /// identity so the next render draws from it without installing again, and
    /// source admission at open reports in the one-shot loader's words, which
    /// name that render's profile.
    template: Option<Shot>,
}
struct Open {
    reader: std::cell::OnceCell<crate::ask::Reader>,
    path: PathBuf,
    base: PathBuf,
    /// The authored document as last opened.
    document: serde_json::Value,
    /// The document decoded (std-expanded) for comparison and dependency tracking.
    scene: Scene,
    /// Content hashes of the tracked external files at open, when enumerable.
    files: Option<Vec<(PathBuf, blake3::Hash)>>,
    revision: u64,
    installed: Option<Installed>,
}
/// What an installation holds on the device.
enum Kept {
    /// A 3D scene, with the delivery identity it was installed under.
    World(InstalledScene, String),
    /// A 2D canvas's document and GPU side; it carries its own identity.
    Canvas(crate::canvas::Kept),
}
struct Installed {
    kept: Kept,
    /// The decoded document this installation was built from.
    snapshot: Scene,
    files: Vec<(PathBuf, blake3::Hash)>,
    pending: Reuse,
    /// Whether `installs` has counted it (a render drew from it).
    counted: bool,
}

impl Default for Worker {
    fn default() -> Self { Self::new() }
}
impl Worker {
    pub fn new() -> Self { Self { renderer: None, open: None, installs: 0, template: None } }
    /// Installations renders have drawn from.
    pub fn installs(&self) -> u64 { self.installs }
    pub fn revision(&self) -> Option<u64> { self.open.as_ref().map(|open| open.revision) }

    /// Read CPU values from the last successful open without rendering or changing its revision.
    pub fn ask(&self, questions: &serde_json::Value) -> Result<serde_json::Value> {
        let open = self.open.as_ref().context("ask requires an open scene; open a scene first").context(crate::failure::FailureKind::Usage)?;
        if open.reader.get().is_none() {
            let reader = crate::ask::Reader::new(open.scene.clone(), &open.base).map_err(crate::failure::scene)?;
            let _ = open.reader.set(reader);
        }
        open.reader.get().unwrap().ask(questions).map_err(crate::failure::scene)
    }

    /// Install a whole document, comparing it with the open one: the candidate is
    /// decoded, its files hashed, the comparison classified and, when it changes
    /// installed facts, the candidate installed on the renderer's device before anything
    /// is published, so a failure keeps the previous revision and installation.
    pub fn open(&mut self, source: Source) -> Result<Opened> {
        let (path, base, document) = match source {
            Source::File(path) => {
                let bytes = std::fs::read(path).with_context(|| format!("load scene {}", path.display()))?;
                let document = vork_scene::v2::diagnostics::read_document(&bytes)
                    .with_context(|| format!("load scene {}", path.display()))?;
                let base = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new(".")).to_path_buf();
                (path.to_path_buf(), base, document)
            }
            Source::Document { document, base } => (base.join("(document)"), base.to_path_buf(), document),
        };
        let scene = decode(&document, &path)?;
        // The one-shot loader's source admission, before any file is read for
        // hashing or any device work: the same errors, in the same words.
        let profile = self.template.as_ref().and_then(|template| template.profile.as_ref())
            .map_or(vork_scene::DEFAULT_PROFILE, |profile| profile.name.as_str()).to_owned();
        let paths = crate::validate_inputs(&scene, &base)
            .with_context(|| format!("load scene {} with profile {profile}", path.display()))?;
        let files = tracked_files(paths);
        let mut changed = Vec::new();
        let mut reuse = Reuse::Reinstalled;
        let mut updates = None;
        if let Some(open) = &self.open {
            diff(&open.document, &document, &mut Vec::new(), &mut changed);
            // Dependencies that cannot be enumerated are never known to be the same.
            let same_files = matches!((&open.files, &files), (Some(before), Some(after)) if before == after);
            if let (Some(before), Some(after)) = (&open.files, &files) {
                for (file, hash) in after {
                    if before.iter().any(|(f, h)| f == file && h != hash) {
                        let relative = file.strip_prefix(&base).unwrap_or(file);
                        changed.push(serde_json::json!(["file", relative.to_string_lossy()]));
                    }
                }
            }
            let identical = same_files && same_scene(&open.scene, &scene)?;
            // An explicit default is not a change.
            if identical { changed.clear(); }
            let recompile = changed.iter().any(|path| touches_program(path));
            // `reused` means a retained installation is kept as it is; with none
            // retained an identical document is still prepared and reinstalled.
            let same_installed_files = matches!((&open.installed, &files),
                (Some(installed), Some(files)) if installed.files == *files);
            reuse = match &open.installed {
                Some(installed) if same_installed_files && same_scene(&installed.snapshot, &scene)? => Reuse::Reused,
                Some(installed) if same_installed_files && !recompile => match installed.plan(&scene)? {
                    Some(plan) => { updates = Some(plan); Reuse::Updated }
                    None => Reuse::Reinstalled,
                },
                _ if recompile => Reuse::Recompiled,
                _ => Reuse::Reinstalled,
            };
        }
        // A candidate that changes installed facts is installed on the device first,
        // under the last render's delivery identity, so a failure here (a program
        // that does not compile, a missing asset, a geometry stage that fails) keeps
        // the previous revision and installation, and a success is the installation
        // the next render draws from. Two installations are alive until publication.
        let prepared = if matches!(reuse, Reuse::Reused | Reuse::Updated) { None }
            else { self.prepare(&scene, &document, &base, &path, files.as_deref(), reuse)? };
        // Publish. The previous installation is dropped here, after the candidate's.
        let (revision, installed) = match self.open.take() {
            Some(previous) => (previous.revision + 1, match (reuse, previous.installed) {
                (Reuse::Reused, installed) => installed,
                (Reuse::Updated, Some(mut installed)) => {
                    installed.write(updates.take().unwrap_or_default(), &scene);
                    installed.pending = Reuse::Updated;
                    Some(installed)
                }
                _ => prepared,
            }),
            None => (1, prepared),
        };
        self.open = Some(Open { path, base, document, scene, files, revision, installed, reader: Default::default() });
        Ok(Opened { revision, reuse, changed })
    }

    /// Install the candidate on the retained renderer's own device under the last
    /// render's delivery identity and hand it back to be published (before any
    /// render: validate and drop, as a one-shot install would): the ordinary install errors, with their authored
    /// paths, before publication, and the next render draws from it without
    /// installing again when its options match. The device is created here when no
    /// render has opened it yet. None when the install emitted warnings: a render
    /// that reused it could not reproduce them, so the render installs instead.
    fn prepare(&mut self, decoded: &Scene, document: &serde_json::Value, base: &Path, path: &Path,
        files: Option<&[(PathBuf, blake3::Hash)]>, pending: Reuse) -> Result<Option<Installed>> {
        // A candidate reads its canvases' image files again, as a fresh process would: Moda's image store is
        // the process's and is keyed by path, and a file may have changed since it was read. Any installation
        // can hold a canvas (the document itself, or a canvas picture in a 3D world), so every candidate
        // starts from an empty store. The previous installation stays, and when the candidate fails the
        // store it drew from is put back, so the previous revision renders on from what it had.
        let images = crate::canvas_draw::forget_images();
        let prepared = self.prepare_candidate(decoded, document, base, path, files, pending);
        if prepared.is_err() { crate::canvas_draw::restore_images(images); }
        prepared
    }

    fn prepare_candidate(&mut self, decoded: &Scene, document: &serde_json::Value, base: &Path, path: &Path,
        files: Option<&[(PathBuf, blake3::Hash)]>, pending: Reuse) -> Result<Option<Installed>> {
        let usage = |error: anyhow::Error| error.context(crate::failure::FailureKind::Usage);
        // Before any render there is no identity to prepare under: validate as a
        // one-shot install would, under the scene's own defaults, and keep nothing.
        // Preparing under a guessed profile would pay that profile's first install
        // (std/final on a CPU device is minutes) for a render that may never match.
        let Some(template) = self.template.clone() else {
            let warnings = crate::warnings::Collector::default();
            let promoted = self.candidate_device(decoded)?;
            let renderer = promoted.as_ref().or(self.renderer.as_ref()).expect("a candidate has a device");
            let result = (|| -> Result<()> {
                let _scope = warnings.enter();
                // A canvas: its fonts load, its images decode, its first frame plans and its worlds install.
                if decoded.is_canvas() {
                    crate::canvas::check(decoded, base, renderer).with_context(|| format!("install scene {}", path.display()))?;
                    return renderer.wait(None);
                }
                let installed = renderer.install_with_options(decoded.clone(), base,
                        crate::InstallOptions { variants: crate::pipelines::VariantCompile::Fallback, ..Default::default() })
                    .with_context(|| format!("install scene {}", path.display()))?;
                super::check_timeline(&installed)?;
                drop(installed);
                renderer.wait(None)
            })();
            // A failure on the device opened beside the retained one takes only that device.
            if promoted.is_none() && result.as_ref().is_err_and(|error| crate::failure::exit_code(error) == 4) { self.discard_device(); }
            result?;
            self.promote(promoted);
            let _ = (document, files, pending);
            return Ok(None);
        };
        // The same derivation the shot performs before it installs.
        let profile = match &template.profile { Some(profile) => profile.clone(), None => vork_scene::Profile::load(vork_scene::DEFAULT_PROFILE)? };
        let overrides = vork_scene::CallOverrides::resolve(template.settings.clone()).map_err(usage)?;
        let settings = profile.resolve(&overrides.settings).map_err(usage)?;
        let default_denoise = (vork_scene::libraries::profile(&profile.name).is_some() && profile.level <= 1 && !overrides.settings.contains_key("denoise"))
            .then(|| (profile.name.clone(), profile.level, overrides.settings.contains_key("samples")));
        let bytes = serde_json::to_vec(document)?;
        let mut scene = super::load_document_with_settings(&bytes, path, settings, &profile.name, template.edit, Some(decoded.clone())).map_err(crate::failure::scene)?;
        if let Some(resolution) = overrides.resolution { scene.render.resolution = resolution; }
        if let Some(fps) = overrides.fps { scene.render.fps = fps; }
        scene.validate().map_err(crate::failure::scene)?;
        let target = crate::output_device::target(&template.out)?;
        crate::output_device::admit(&scene, &template.out, target).map_err(crate::failure::scene)?;
        if let Some(resolution) = template.resolution { scene.render.resolution = resolution; }
        let outputs = crate::png_output::Outputs::new(super::resolve_outputs(template.outputs.clone(), &scene.render.aovs));
        let has_format = |p: &PathBuf, ext: &str| !super::directory_marked(p) && p.extension().is_some_and(|s| s.eq_ignore_ascii_case(ext));
        let video = template.out.iter().any(|p| super::video_destination(p));
        let directory_paths: Vec<&PathBuf> = template.out.iter().filter(|p| !super::video_destination(p)).map(|p| &p.file).collect();
        let clip_frames = super::clip_frame_count(&scene.render)?;
        let implicit_still = template.at.is_none() && template.frame_start.is_none() && template.frame_end.is_none() && clip_frames > 1
            && !directory_paths.is_empty() && directory_paths.iter().all(|p| has_format(p, "png"));
        let still = template.at.is_some() || implicit_still;
        let first_frame = template.frame_start.unwrap_or(0);
        let frames = if still && !video { 1 } else { template.frame_end.unwrap_or(clip_frames.saturating_sub(1)) + 1 - first_frame.min(clip_frames) };
        let still_pass = video && template.at.is_some() && !directory_paths.is_empty();
        let readback = outputs.readback(directory_paths.iter().any(|p| !has_format(p, "exr")), directory_paths.iter().any(|p| has_format(p, "exr")), target.png_depth(template.png_depth));
        let install_readback = if still_pass || !(template.gpu_only || (video && directory_paths.is_empty())) { readback } else { Default::default() };
        let initial_time = if video || !still { vork_time::Request::Frame(first_frame as u32, scene.render.fps) }
            else { template.at.as_ref().map(vork_time::Request::from).unwrap_or(vork_time::Request::Numeric(0.)) };
        let warnings = crate::warnings::Collector::default();
        let promoted = self.candidate_device(&scene)?;
        let renderer = promoted.as_ref().or(self.renderer.as_ref()).expect("a candidate has a device");
        let result = (|| -> Result<Option<Installed>> {
            let _scope = warnings.enter();
            // A canvas: the document and GPU side a render builds before its first frame.
            if scene.is_canvas() {
                let kept = crate::canvas::Kept::prepare(renderer, &scene, base, target)
                    .with_context(|| format!("install scene {}", path.display()))?;
                let Some(files) = files.filter(|_| template.edit.is_none()) else {
                    drop(kept);
                    renderer.wait(None)?;
                    return Ok(None);
                };
                return Ok(Some(Installed { kept: Kept::Canvas(kept), snapshot: decoded.clone(), files: files.to_vec(), pending, counted: false }));
            }
            let plan = super::install_plan(renderer, &mut scene, default_denoise.as_ref(), install_readback, frames == 1, video || template.gpu_only, &outputs.0, initial_time, target)?;
            let installed = renderer.install_with_options(scene, base, plan.options)
                .with_context(|| format!("install scene {}", path.display()))?;
            super::check_timeline(&installed)?;
            // Dependencies that cannot be enumerated (programs, media, simulation)
            // are read again by every render, as a one-shot would: the candidate
            // proved the document installs and is dropped, kept by nothing.
            let Some(files) = files.filter(|_| template.edit.is_none()) else {
                drop(installed);
                renderer.wait(None)?;
                return Ok(None);
            };
            Ok(Some(Installed { kept: Kept::World(installed, plan.delivery), snapshot: decoded.clone(), files: files.to_vec(), pending, counted: false }))
        })();
        // A device failure (lost device, out of memory), in the install or the
        // wait that drops a candidate, leaves no renderer to trust, as after a
        // failed render; ordinary install errors do. On the device opened beside
        // the retained one, it takes only that device.
        if promoted.is_none() && result.as_ref().is_err_and(|error| crate::failure::exit_code(error) == 4) { self.discard_device(); }
        let installed = result?;
        self.promote(promoted);
        Ok(installed.filter(|_| warnings.snapshot().is_empty()))
    }

    /// Make sure a candidate `scene` has a device to be prepared on, by the rule every device is opened by
    /// (`rt::device_needs`, `Renderer::providing`). With no device retained, one is opened and retained. When
    /// the retained device lacks what the scene needs, the device that would replace it is opened beside it and
    /// returned: the retained device and its installation stay until the candidate is proven (`promote`), so
    /// a refused candidate costs the open revision nothing.
    fn candidate_device(&mut self, scene: &Scene) -> Result<Option<Renderer>> {
        let needs = crate::rt::device_needs(scene).map_err(crate::failure::scene)?;
        if let Some(renderer) = &self.renderer {
            return if renderer.provides(needs) { Ok(None) } else { renderer.beside(needs).map(Some).map_err(crate::failure::setup) };
        }
        // An installation only exists together with the renderer that made it.
        if let Some(open) = &mut self.open { open.installed = None; }
        self.renderer = Some(Renderer::providing(None, needs).map_err(crate::failure::setup)?);
        Ok(None)
    }

    /// A candidate was proven on `promoted`, the device opened beside the retained one: it becomes the
    /// retained device. The installation on the old device goes first, then the old device.
    fn promote(&mut self, promoted: Option<Renderer>) {
        let Some(promoted) = promoted else { return };
        if let Some(open) = &mut self.open { open.installed = None; }
        self.renderer = Some(promoted);
    }

    /// Drop the device and, with it, the installation; the document and revision stay.
    fn discard_device(&mut self) {
        if let Some(open) = &mut self.open { open.installed = None; }
        self.renderer = None;
    }

    /// Render the open document with the shot's options. The shot's `scene` and
    /// `document` are set here. Returns the shot report and how the installation was obtained.
    pub fn render(&mut self, mut shot: Shot) -> Result<(serde_json::Value, Reuse)> {
        let open = self.open.as_mut().context("no scene is open; send open first").context(crate::failure::FailureKind::Usage)?;
        shot.scene = open.path.clone();
        shot.document = Some(serde_json::to_vec(&open.document)?);
        // Ordinary source admission can refuse before touching the device. Cache
        // identity cannot: untracked or unreadable hashes just force installation.
        // An unsupported destination refuses before this request takes or creates the device, and so does
        // a display the open scene cannot encode for: a refusal leaves the device and the installation kept.
        // A caller's rewrite runs later and may change the scene, as it may change its sources below: the
        // shot then admits the rewritten scene itself, where its errors are every rewritten scene's.
        let target = crate::output_device::target(&shot.out)?;
        let decoded = decode(&open.document, &open.path)?;
        if shot.edit.is_none() { crate::output_device::admit(&decoded, &shot.out, target).map_err(crate::failure::scene)?; }
        let paths = crate::validate_files(&decoded, &open.base)?;
        // A caller's rewrite runs later and may introduce different sources.
        let files = tracked_files(paths).filter(|_| shot.edit.is_none());
        let same_install = match (&open.installed, &files) {
            (Some(installed), Some(files)) => installed.files == *files && same_scene(&installed.snapshot, &decoded)?,
            _ => false,
        };
        // From here on the renderer and the installation leave together: a
        // failure drops both, and an installation never meets a device other
        // than the one that made it. The shot opens the device when none is
        // retained, or when the retained one lacks what the resolved scene
        // needs; its receipt then reports that device as this request's.
        let renderer = self.renderer.take();
        let installed = open.installed.take().filter(|_| renderer.is_some());
        let previous = installed.filter(|_| same_install);
        let (pending, counted) = previous.as_ref().map_or((None, true), |installed| (Some(installed.pending), installed.counted));
        // An installation built by this render reads its canvases' image files again, as `prepare` does for a
        // candidate. No installation is alive here to need the earlier contents: a failed render drops device
        // and installation.
        if previous.is_none() { crate::canvas_draw::forget_images(); }
        let (installed, canvas) = match previous.map(|installed| installed.kept) {
            Some(Kept::World(scene, delivery)) => (Some((scene, delivery)), None),
            Some(Kept::Canvas(kept)) => (None, Some(kept)),
            None => (None, None),
        };
        let retained = renderer.map(|renderer| Retained { renderer, installed, canvas });
        // On an error the device and installation went down with the shot; the
        // next request opens a fresh device rather than trusting partial state.
        // Only options that rendered become the identity the next open prepares
        // under: refused options would make every later open fail on them.
        let mut template = shot.clone();
        let rendered = shot.render_retained_decoded(retained, decoded.clone())?;
        template.document = None;
        self.template = Some(template);
        let report = rendered.report.clone();
        let reuse = if report["install_reused"] == true {
            if !counted { self.installs += 1; }
            pending.unwrap_or(Reuse::Reused)
        } else {
            self.installs += 1;
            Reuse::Reinstalled
        };
        let retained = rendered.into_retained();
        if let Some(retained) = retained {
            self.renderer = Some(retained.renderer);
            // A shot with warnings must reproduce them next time: reinstall then.
            let clean = report["warnings"].as_array().is_some_and(Vec::is_empty);
            let kept = match (retained.installed, retained.canvas) {
                (Some((scene, delivery)), _) => Some(Kept::World(scene, delivery)),
                (None, Some(canvas)) => Some(Kept::Canvas(canvas)),
                (None, None) => None,
            };
            if let (Some(kept), Some(files), true) = (kept, files, clean) {
                open.installed = Some(Installed { kept, snapshot: decoded, files, pending: Reuse::Reused, counted: true });
            }
        }
        Ok((report, reuse))
    }

    /// Release the document and its installation; the device stays for a later open.
    pub fn close(&mut self) { self.open = None; }

    /// Leave on disk whatever pipelines the worker's renderer has finished, before a process that skips
    /// teardown exits (`Renderer::finish_pipeline_save`).
    pub fn finish_pipeline_save(&self) {
        if let Some(renderer) = &self.renderer { renderer.finish_pipeline_save(); }
    }
}

/// Decode the authored document as the renderer will (std expansion, authored locations).
fn decode(document: &serde_json::Value, path: &Path) -> Result<Scene> {
    let authored = serde_json::to_vec(document)?;
    (|| -> Result<Scene> {
        let scene = crate::decode_scene_document(&authored, path)?;
        scene.validate().map_err(|error| scene.authored_error(error))?;
        Ok(scene)
    })().with_context(|| format!("load scene {}", path.display())).map_err(crate::failure::scene)
}

/// Whether two decoded scenes have the same content: what each serializes, hashed as the typed scene is walked
/// (`crate::identity`). An explicit default, a field's position in its object and an authored location are not
/// content.
fn same_scene(before: &Scene, after: &Scene) -> Result<bool> {
    Ok(crate::identity::digest(before)? == crate::identity::digest(after)?)
}

/// A changed path that makes the next install compile: a `programs` entry, or
/// an inline `program` body (`materials.x.program.source`, a compositor entry's).
fn touches_program(path: &serde_json::Value) -> bool {
    let Some(segments) = path.as_array() else { return false };
    if segments.first().and_then(|s| s.as_str()) == Some("programs") { return true; }
    segments.iter().enumerate().any(|(index, segment)| segment == "program"
        && segments.get(index + 1).is_none_or(|next| next == "source"))
}

/// The authored paths at which two documents differ. Objects compare by key;
/// list items pair by non-empty `name` where both sides have one, the remaining
/// items by position in order, and an item left over is reported at the list
/// with its name or new index; a list whose named items changed order, and a
/// list of scalars that differs, is reported as the list itself. Numbers compare
/// numerically (1 == 1.0).
fn diff(before: &serde_json::Value, after: &serde_json::Value, path: &mut Vec<serde_json::Value>, out: &mut Vec<serde_json::Value>) {
    use serde_json::Value;
    match (before, after) {
        (Value::Object(a), Value::Object(b)) => {
            let keys: std::collections::BTreeSet<&String> = a.keys().chain(b.keys()).collect();
            for key in keys {
                path.push(key.as_str().into());
                match (a.get(key), b.get(key)) {
                    (Some(x), Some(y)) => diff(x, y, path, out),
                    _ => out.push(path.clone().into()),
                }
                path.pop();
            }
        }
        // A list of scalars (a colour, a transform, a name list) is one value.
        (Value::Array(a), Value::Array(b)) if !a.iter().chain(b).any(Value::is_object) => {
            if a.len() != b.len() || a.iter().zip(b).any(|(x, y)| match (x, y) {
                (Value::Number(x), Value::Number(y)) => x.as_f64() != y.as_f64(),
                _ => x != y,
            }) {
                out.push(path.clone().into());
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            let name = |item: &Value| item.get("name").and_then(|n| n.as_str()).filter(|n| !n.is_empty()).map(str::to_owned);
            let mut taken = vec![false; b.len()];
            let mut pairs = Vec::new();
            let mut unnamed = Vec::new();
            for (i, item) in a.iter().enumerate() {
                match name(item) {
                    Some(n) => match b.iter().enumerate().position(|(j, other)| !taken[j] && name(other).as_deref() == Some(&n)) {
                        Some(j) => { taken[j] = true; pairs.push((i, j)); }
                        None => { path.push(n.into()); out.push(path.clone().into()); path.pop(); }
                    },
                    None => unnamed.push(i),
                }
            }
            let mut rest = (0..b.len()).filter(|j| !taken[*j] && name(&b[*j]).is_none()).collect::<Vec<_>>().into_iter();
            for i in unnamed {
                match rest.next() {
                    Some(j) => { taken[j] = true; pairs.push((i, j)); }
                    None => { path.push(i.into()); out.push(path.clone().into()); path.pop(); }
                }
            }
            for (j, item) in b.iter().enumerate().filter(|(j, _)| !taken[*j]) {
                path.push(name(item).map_or(j.into(), Value::from));
                out.push(path.clone().into());
                path.pop();
            }
            pairs.sort();
            if pairs.windows(2).any(|w| w[0].1 > w[1].1) { out.push(path.clone().into()); }
            for (i, j) in pairs {
                path.push(name(&b[j]).map_or(j.into(), Value::from));
                diff(&a[i], &b[j], path, out);
                path.pop();
            }
        }
        (Value::Number(x), Value::Number(y)) if x.as_f64() == y.as_f64() => {}
        _ if before == after => {}
        _ => out.push(path.clone().into()),
    }
}

/// Hash only the sources enumerated by ordinary admission. Failure to identify
/// an installation disables retention; only the ordinary loader may refuse it.
fn tracked_files(paths: Option<std::collections::BTreeSet<PathBuf>>) -> Option<Vec<(PathBuf, blake3::Hash)>> {
    paths?.into_iter().map(|path| {
        let mut hash = blake3::Hasher::new();
        hash.update_reader(crate::source_file(&path).ok()?).ok()?;
        Some((path, hash.finalize()))
    }).collect()
}

impl Installed {
    /// The value-only changes between the installed document and `next`, or None
    /// when the change touches anything the installation proved or specialized.
    fn plan(&self, next: &Scene) -> Result<Option<Vec<(usize, serde_json::Value)>>> {
        // A changed canvas builds its document again: no value is written into a kept one.
        let Kept::World(installed, _) = &self.kept else { return Ok(None) };
        if !installed.footage.is_empty() || installed.compute.is_some() { return Ok(None); }
        let camera_updates = installed.displacement.is_none()
            && installed.probes.is_none()
            && installed.fibre.is_none()
            && self.snapshot.geometry.values().all(|g| matches!(g.topology, vork_scene::v2::Topology::Triangles));
        let cameras: Vec<bool> = (0..installed.nodes.len())
            .map(|id| camera_updates && static_camera_placement(&installed.nodes, &installed.animation, id))
            .collect();
        changes(&self.snapshot, next, &cameras, &installed.animation)
    }

    /// Write planned value changes into the installation (`plan` proved them).
    fn write(&mut self, updates: Vec<(usize, serde_json::Value)>, next: &Scene) {
        let Kept::World(installed, _) = &mut self.kept else { unreachable!("plan offers updates for 3D installations only") };
        fn field<T: serde::de::DeserializeOwned>(fields: &serde_json::Value, name: &str) -> T {
            serde_json::from_value(fields[name].clone()).expect("planned update values come from a decoded scene")
        }
        for (index, fields) in updates {
            if let NodeKind::Light { energy, color, .. } = &mut installed.nodes[index].node.kind {
                *energy = field(&fields, "energy");
                *color = field(&fields, "color");
            } else {
                let parent = installed.nodes[index].parent.map_or(glam::Mat4::IDENTITY, |p| installed.nodes[p].world);
                let node = &mut installed.nodes[index];
                node.node.transform.translation = field(&fields, "translation");
                node.node.transform.rotation = field(&fields, "rotation");
                node.node.transform.scale = field(&fields, "scale");
                node.world = parent * node.node.transform.matrix();
            }
        }
        // The evaluated light plan reads the flattened nodes; diagnostics and
        // the next comparison read the authored tree.
        fn copy(nodes: &mut [Node], next: &[Node]) {
            for (old, new) in nodes.iter_mut().zip(next) {
                if let (NodeKind::Light { energy, color, .. }, NodeKind::Light { energy: e, color: c, .. }) = (&mut old.kind, &new.kind) {
                    *energy = *e;
                    *color = *c;
                }
                if matches!(old.kind, NodeKind::Camera { .. }) { old.transform = new.transform.clone(); }
                copy(&mut old.children, &new.children);
            }
        }
        copy(&mut installed.scene.nodes, &next.nodes);
        self.snapshot = next.clone();
    }
}

// Cameras are always evaluated for program uniforms, even with a static pose.
// Test transform/aim dependencies instead of the general evaluation flag.
fn static_camera_placement(nodes: &[crate::install::EvalNode], animation: &crate::animation::Plan, camera: usize) -> bool {
    if !matches!(nodes[camera].node.kind, NodeKind::Camera { focus_object: None, .. }) { return false; }
    let mut current = Some(camera);
    while let Some(index) = current {
        let node = &nodes[index];
        if node.embedded.is_some()
            || node.node.target.is_some()
            || matches!(node.node.kind, NodeKind::Asset { .. })
            || animation.transforms_node(index)
        {
            return false;
        }
        current = node.parent;
    }
    true
}

/// The value changes between two decoded documents, or None when anything else
/// differs. Every unmasked byte is part of the installation identity, so a new
/// scene field fails closed without extending this classifier.
fn changes(before_scene: &Scene, after_scene: &Scene, cameras: &[bool], animation: &crate::animation::Plan) -> Result<Option<Vec<(usize, serde_json::Value)>>> {
    let mut before = serde_json::to_value(before_scene)?;
    let mut after = serde_json::to_value(after_scene)?;
    let mut updates = Vec::new();
    // Only radiometric local-light values are independent of install-time
    // topology, acceleration, shader proofs and projection specialization.
    // Animated values have defaults captured by the plan and must be rebound.
    let animated_lights: std::collections::BTreeSet<_> = animation.targets().filter_map(|target| {
        use vork_scene::v2::{TrackTarget, NodeProperty};
        match target {
            TrackTarget::Node {node, property: NodeProperty::Energy | NodeProperty::Color} => Some(node.as_str()),
            _ => None,
        }
    }).collect();
    fn strip(value: &mut serde_json::Value, animated_lights: &std::collections::BTreeSet<&str>, cameras: &[bool], updates: &mut Vec<(usize, serde_json::Value)>, index: &mut usize) {
        for node in value.as_array_mut().into_iter().flatten() {
            let id = *index;
            *index += 1;
            let animated_light = node["name"].as_str().is_some_and(|name| animated_lights.contains(name));
            if node["type"] == "light" && node["shape"] != "environment" && !animated_light {
                let mut fields = serde_json::Map::new();
                for field in ["energy", "color"] {
                    if let Some(v) = node.as_object_mut().unwrap().remove(field) { fields.insert(field.into(), v); }
                }
                updates.push((id, fields.into()));
            }
            if node["type"] == "camera"
                && cameras.get(id).copied().unwrap_or(false)
                && node["children"].as_array().is_none_or(|c| c.is_empty())
            {
                let mut fields = serde_json::Map::new();
                for field in ["translation", "rotation", "scale"] {
                    if let Some(v) = node.as_object_mut().unwrap().remove(field) { fields.insert(field.into(), v); }
                }
                updates.push((id, fields.into()));
            }
            strip(&mut node["children"], animated_lights, cameras, updates, index);
        }
    }
    strip(&mut before["nodes"], &animated_lights, cameras, &mut Vec::new(), &mut 0);
    strip(&mut after["nodes"], &animated_lights, cameras, &mut updates, &mut 0);
    if before != after { return Ok(None); }
    Ok(Some(updates))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn changes(before: &Scene, after: &Scene, cameras: &[bool]) -> Result<Option<Vec<(usize, serde_json::Value)>>> {
        let animation = crate::animation::Plan::compile(before, Path::new("."))?;
        super::changes(before, after, cameras, &animation)
    }
    #[test]
    fn ask_reads_open_document_without_device_or_revision_changes() -> Result<()> {
        let mut worker = Worker::new();
        let questions = json!([{"field":"nodes.key.energy"}]);
        assert!(worker.ask(&questions).is_err());
        let source = scene();
        worker.open = Some(Open {
            reader: Default::default(),
            path: PathBuf::from("does-not-exist.json"), base: PathBuf::from("."),
            document: serde_json::to_value(&source)?, scene: source, files: None,
            revision: 7, installed: None,
        });
        let expected = serde_json::to_value(&worker.open.as_ref().unwrap().scene.nodes[0])?["energy"].clone();
        assert_eq!(worker.ask(&questions)?, json!([expected]));
        assert!(worker.ask(&json!([{"node":"absent","read":"bounds"}])).is_err());
        assert_eq!(worker.revision(), Some(7));
        assert_eq!(worker.installs(), 0);
        assert!(worker.renderer.is_none());
        assert!(worker.open.as_ref().unwrap().installed.is_none());
        worker.close();
        assert!(worker.ask(&questions).is_err());
        Ok(())
    }
    fn scene() -> Scene {
        Scene::from_json(br#"{"version":2,"nodes":[{"name":"key","type":"light","shape":"point","energy":10},{"name":"camera","type":"camera"}]}"#).unwrap()
    }
    fn changed(before: serde_json::Value, after: serde_json::Value) -> Vec<serde_json::Value> {
        let mut out = Vec::new();
        diff(&before, &after, &mut Vec::new(), &mut out);
        out
    }
    #[test]
    fn diff_pairs_list_items_by_name_then_position() {
        let before = json!({"version":2,"composition":{"space":"3d","compositor":["offset","gain"]},"render":{"resolution":[64,48]},
            "nodes":[{"name":"key","type":"light","energy":10,"color":[1,1,1]},{"name":"cam","type":"camera","children":[{"name":"rig","type":"node"}]},{"type":"node"}],
            "materials":{"a":{"inputs":{"base_color":[1,0,0]}},"b":{}}});
        // Equal, including 1 == 1.0 and key order.
        let same = json!({"nodes":[{"type":"light","name":"key","color":[1.0,1,1],"energy":10.0},{"name":"cam","type":"camera","children":[{"name":"rig","type":"node"}]},{"type":"node"}],
            "materials":{"b":{},"a":{"inputs":{"base_color":[1,0,0]}}},"composition":{"space":"3d","compositor":["offset","gain"]},"render":{"resolution":[64,48]},"version":2});
        assert_eq!(changed(before.clone(), same), Vec::<serde_json::Value>::new());
        // Named items match by name wherever they sit; a pure reorder names the list.
        let mut reordered = before.clone();
        let nodes = reordered["nodes"].as_array_mut().unwrap();
        nodes.swap(0, 1);
        assert_eq!(changed(before.clone(), reordered), vec![json!(["nodes"])]);
        // A value, a nested child, a table entry, an unnamed compositor position.
        let mut edited = before.clone();
        edited["nodes"][0]["energy"] = 20.into();
        edited["nodes"][1]["children"][0]["translation"] = json!([1,0,0]);
        edited["materials"]["a"]["inputs"]["base_color"] = json!([0,1,0]);
        edited["composition"]["compositor"] = json!(["gain","offset"]);
        assert_eq!(changed(before.clone(), edited), vec![
            json!(["composition","compositor"]), json!(["materials","a","inputs","base_color"]), json!(["nodes","key","energy"]),
            json!(["nodes","cam","children","rig","translation"])]);
        // Added, removed and renamed items are reported at the list; an unnamed
        // item pairs with the next unnamed item on the other side.
        let mut shaped = before.clone();
        shaped["nodes"].as_array_mut().unwrap().remove(2);
        shaped["nodes"].as_array_mut().unwrap().push(json!({"name":"fill","type":"light"}));
        shaped["nodes"][0]["name"] = "sun".into();
        shaped["materials"].as_object_mut().unwrap().remove("b");
        assert_eq!(changed(before.clone(), shaped), vec![
            json!(["materials","b"]), json!(["nodes","key"]), json!(["nodes",2]), json!(["nodes","sun"]), json!(["nodes","fill"])]);
        assert!(touches_program(&json!(["programs","custom","source"])));
        assert!(touches_program(&json!(["materials","a","program","source"])));
        assert!(touches_program(&json!(["composition","compositor",0,"material","program"])));
        assert!(touches_program(&json!(["materials","a","program"])));
        assert!(!touches_program(&json!(["materials","a","inputs","gain"])));
    }
    #[test]
    fn camera_program_evaluation_is_not_placement_animation() -> Result<()> {
        let scene = scene();
        let mut nodes: Vec<_> = scene.nodes.iter().map(|node| crate::install::EvalNode {
            node: node.clone(), camera: None, parent: None, world: node.transform.matrix(), dynamic: true, embedded: None,
        }).collect();
        let plan = crate::animation::Plan::compile(&scene, Path::new("."))?;
        assert!(static_camera_placement(&nodes, &plan, 1));
        nodes[1].node.target = Some(vork_scene::v2::AimTarget::Point([0., 0., 0.]));
        assert!(!static_camera_placement(&nodes, &plan, 1));
        Ok(())
    }
    #[test]
    fn values_update_but_new_install_facts_do_not() -> Result<()> {
        let before = scene();
        let mut after = before.clone();
        if let NodeKind::Light { energy, color, .. } = &mut after.nodes[0].kind {
            *energy = Some(20.);
            *color = Some([0.2, 0.4, 0.8]);
        }
        assert!(changes(&before, &after, &[false, true])?.is_some());
        after.nodes[1].transform.translation[0] = 2.;
        assert!(changes(&before, &after, &[false, true])?.is_some());
        assert!(changes(&before, &after, &[false, false])?.is_none());
        after = before.clone();
        if let NodeKind::Light { radius, .. } = &mut after.nodes[0].kind { *radius = 0.5; }
        assert!(changes(&before, &after, &[false, true])?.is_none());
        after = before.clone();
        after.nodes.push(before.nodes[1].clone());
        assert!(changes(&before, &after, &[false, true])?.is_none());
        after = before.clone();
        after.render.resolution = [32, 32];
        assert!(changes(&before, &after, &[false, true])?.is_none());
        Ok(())
    }
    #[test]
    fn environment_and_animated_defaults_require_reinstall() -> Result<()> {
        let before = Scene::from_json(br#"{"version":2,"nodes":[{"name":"world","type":"light","shape":"environment","energy":1},{"name":"camera","type":"camera"}]}"#)?;
        let mut after = before.clone();
        if let NodeKind::Light { energy, .. } = &mut after.nodes[0].kind { *energy = Some(2.); }
        assert!(changes(&before, &after, &[])?.is_none());
        let before = Scene::from_json(br#"{"version":2,"nodes":[{"name":"key","type":"light","shape":"point","energy":10},{"name":"camera","type":"camera"}],"animation":[{"tracks":[{"path":["nodes","key","energy"],"keys":[[0,10],[1,20]]}]}]}"#)?;
        let mut after = before.clone();
        if let NodeKind::Light { energy, .. } = &mut after.nodes[0].kind { *energy = Some(20.); }
        assert!(changes(&before, &after, &[])?.is_none());
        Ok(())
    }
    #[test]
    fn node_owned_animated_light_defaults_require_reinstall() -> Result<()> {
        for (ancestor, property) in [false, true].into_iter().flat_map(|ancestor| ["energy", "color"].map(|property| (ancestor, property))) {
            let light = json!({"name":"key","type":"light","shape":"point","energy":10,"color":[1,1,1]});
            let mut owner = if ancestor { json!({"name":"rig","type":"node","children":[light]}) } else {light};
            let path = if ancestor {json!(["children","key",property])} else {json!([property])};
            let value = if property == "energy" {json!(20)} else {json!([0.2,0.4,0.8])};
            owner["animation"] = json!([{"weight":0.5,"tracks":[{"path":path,"keys":[[0,value],[1,value]]}]}]);
            let before = Scene::from_json(&serde_json::to_vec(&json!({"version":2,"nodes":[owner,{"type":"camera"}]}))?)?;
            let mut after = before.clone();
            let light = if ancestor {&mut after.nodes[0].children[0]} else {&mut after.nodes[0]};
            if let NodeKind::Light {energy,color,..} = &mut light.kind {
                if property == "energy" { *energy = Some(30.); } else { *color = Some([0.6,0.5,0.4]); }
            }
            assert!(changes(&before, &after, &[])?.is_none(), "ancestor {ancestor}, property {property}");
        }
        Ok(())
    }

    #[test]
    fn animation_names_are_not_light_value_dependencies() -> Result<()> {
        let before = Scene::from_json(br#"{"version":2,"nodes":[{"name":"key","type":"light","shape":"point","energy":10},{"name":"camera","type":"camera"}],"animation":[{"name":"key","tracks":[{"path":["nodes","camera","translation"],"keys":[[0,[0,0,0]],[1,[1,0,0]]]}]}]}"#)?;
        let mut after = before.clone();
        if let NodeKind::Light { energy, .. } = &mut after.nodes[0].kind { *energy = Some(30.); }
        assert!(changes(&before, &after, &[])?.is_some());
        Ok(())
    }

    #[test]
    fn node_owned_clips_participate_in_installation_identity() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-worker-clips-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let document = json!({"asset":{"version":"2.0"},"buffers":[{"uri":"motion%20data.bin","byteLength":4}]});
        std::fs::write(base.join("motion.gltf"), serde_json::to_vec(&document)?)?;
        std::fs::write(base.join("asset.gltf"), br#"{"asset":{"version":"2.0"}}"#)?;
        std::fs::write(base.join("motion data.bin"), [0u8;4])?;
        for ancestor in [false, true] {
            let asset = json!({"name":"model","type":"asset","file":"asset.gltf"});
            let mut owner = if ancestor { json!({"name":"rig","type":"node","children":[asset]}) } else {asset};
            owner["animation"] = json!([{"clip":{"target":"model","file":"motion.gltf","index":0}}]);
            let scene = Scene::from_json(&serde_json::to_vec(&json!({"version":2,"nodes":[owner,{"type":"camera"}]}))?)?;
            let before = tracked_files(crate::validate_files(&scene, &base)?).unwrap();
            assert!(before.iter().any(|(path,_)|path == &base.join("motion.gltf")));
            assert!(before.iter().any(|(path,_)|path == &base.join("motion data.bin")));
            std::fs::write(base.join("motion data.bin"), [u8::from(!ancestor)+1;4])?;
            assert_ne!(before, tracked_files(crate::validate_files(&scene, &base)?).unwrap());
            std::fs::remove_file(base.join("motion data.bin"))?;
            assert!(crate::validate_files(&scene, &base).is_err());
            std::fs::write(base.join("motion data.bin"), [0u8;4])?;
            std::fs::remove_file(base.join("motion.gltf"))?;
            assert!(crate::validate_files(&scene, &base).is_err());
            std::fs::write(base.join("motion.gltf"), serde_json::to_vec(&document)?)?;
        }
        std::fs::remove_dir_all(base)?;
        Ok(())
    }

    #[test]
    fn untracked_dependencies_and_changed_files_fail_closed() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-worker-files-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let mut scene = scene();
        scene.buffers.insert("data".into(), serde_json::from_value(json!({"type":"f32","file":"values.bin","count":1}))?);
        std::fs::write(base.join("values.bin"), 1f32.to_le_bytes())?;
        let before = tracked_files(crate::validate_files(&scene, &base)?).unwrap();
        std::fs::write(base.join("values.bin"), 2f32.to_le_bytes())?;
        assert_ne!(before, tracked_files(crate::validate_files(&scene, &base)?).unwrap());
        scene.images.insert("frames".into(), serde_json::from_value(json!({"file":"frame.%04d.png"}))?);
        assert!(tracked_files(crate::validate_files(&scene, &base)?).is_some());
        // Identity failure after admission must only disable retention.
        let paths = crate::validate_files(&scene, &base)?;
        std::fs::remove_file(base.join("values.bin"))?;
        assert!(tracked_files(paths).is_none());
        std::fs::write(base.join("values.bin"), 2f32.to_le_bytes())?;
        scene.images.clear();
        scene.programs.insert("custom".into(), serde_json::from_value(json!({"source":"import sibling;"}))?);
        assert!(tracked_files(crate::validate_files(&scene, &base)?).is_none());
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
    #[test]
    fn retention_uses_admitted_sources_without_an_extension_policy() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-worker-sources-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let image_scene = |file: &str| -> Result<Scene> { Ok(Scene::from_json(&serde_json::to_vec(&json!({
            "version":2, "images":{"plate":{"file":file}},
            "materials":{"world":{"program":"std/environment_image","inputs":{"image":{"image":"plate"}}}},
            "nodes":[{"type":"light","shape":"environment","material":"world"},{"type":"camera"}]
        }))?)?) };
        // Admission owns format support. Retention hashes the file it supplies,
        // including formats the former worker extension list left untracked.
        std::fs::write(base.join("plate.tif"), b"source bytes")?;
        let scene = image_scene("plate.tif")?;
        let files = tracked_files(crate::validate_files(&scene, &base)?).unwrap();
        assert_eq!(files[0].0, base.join("plate.tif"));
        // Every composition's compositor supplies its sources, a world's inside a canvas included (source
        // admission reads bindings; it compiles no program).
        let nested = Scene::from_json(&serde_json::to_vec(&json!({
            "version":2, "composition":{"space":"2d"}, "images":{"plate":{"file":"plate.tif"}},
            "materials":{"grade":{"program":"std/environment_image","inputs":{"image":{"image":"plate"}}}},
            "nodes":[{"name":"world","composition":{"space":"3d","compositor":["grade"]},"children":[{"type":"camera","name":"camera"}]}]
        }))?)?;
        assert_eq!(tracked_files(crate::validate_files(&nested, &base)?).unwrap()[0].0, base.join("plate.tif"));
        let missing = image_scene("missing.png")?;
        assert!(tracked_files(crate::validate_files(&missing, &base)?).is_none());
        // Enumeration uses the same sequence reader as ordinary admission.
        for name in ["0001.png", "0002.png", "1.png", "2.png", "movie.MOV"] {
            std::fs::write(base.join(name), b"source bytes")?;
        }
        for file in ["%04d.png", "%d.png", "movie.MOV"] {
            assert!(tracked_files(crate::validate_files(&image_scene(file)?, &base)?).is_none(), "{file}");
        }
        // glTF dependencies use ordinary URI decoding, including BOM admission.
        std::fs::write(base.join("mesh data.payload"), [0u8; 4])?;
        let mut gltf = b"\xef\xbb\xbf".to_vec();
        gltf.extend(serde_json::to_vec(&json!({"asset":{"version":"2.0"},
            "buffers":[{"uri":"mesh%20data.payload","byteLength":4}]}))?);
        std::fs::write(base.join("asset.gltf"), gltf)?;
        let scene = Scene::from_json(&serde_json::to_vec(&json!({"version":2,
            "nodes":[{"type":"asset","file":"asset.gltf"},{"type":"camera"}]}))?)?;
        let before = tracked_files(crate::validate_files(&scene, &base)?).unwrap();
        assert!(before.iter().any(|(path,_)| path == &base.join("mesh data.payload")));
        std::fs::write(base.join("mesh data.payload"), [1u8; 4])?;
        assert_ne!(before, tracked_files(crate::validate_files(&scene, &base)?).unwrap());
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn reopen_compares_and_prepares_on_the_device() -> Result<()> {
        // Opt-in GPU test; without an adapter (CI runners) it skips like the others.
        let probe = crate::test_renderer()?;
        drop(probe);
        let base = std::env::temp_dir().join(format!("vork-worker-open-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let path = base.join("scene.json");
        let document = json!({"version":2,"nodes":[{"name":"key","type":"light","shape":"point","energy":10},{"name":"cam","type":"camera"}]});
        std::fs::write(&path, serde_json::to_vec(&document)?)?;
        let mut worker = Worker::new();
        assert!(format!("{:#}", worker.render(Shot::default()).unwrap_err()).contains("no scene is open"));
        // First open: nothing to compare with.
        assert_eq!(worker.open(Source::File(&path))?, Opened { revision: 1, reuse: Reuse::Reinstalled, changed: vec![] });
        // Before any render nothing is kept, so identical up to explicit defaults
        // lists no change but is prepared and reinstalled. The document form
        // resolves against base.
        let mut explicit = document.clone();
        explicit["nodes"][0]["radius"] = 0.into();
        explicit["nodes"][1]["scale"] = json!([1,1,1]);
        assert_eq!(worker.open(Source::Document { document: explicit, base: &base })?, Opened { revision: 2, reuse: Reuse::Reinstalled, changed: vec![] });
        // A value change with nothing retained reinstalls too. The list is authored
        // text against authored text, so the explicit defaults the previous
        // document wrote are listed as dropped.
        let mut brighter = document.clone();
        brighter["nodes"][0]["energy"] = 20.into();
        assert_eq!(worker.open(Source::Document { document: brighter.clone(), base: &base })?,
            Opened { revision: 3, reuse: Reuse::Reinstalled, changed: vec![json!(["nodes","key","energy"]), json!(["nodes","key","radius"]), json!(["nodes","cam","scale"])] });
        // A geometry change reinstalls; the candidate was prepared on the check renderer.
        let mut reshaped = brighter.clone();
        reshaped["nodes"].as_array_mut().unwrap().push(json!({"name":"ball","type":"mesh","geometry":{"std":"sphere","radius":0.5}}));
        let opened = worker.open(Source::Document { document: reshaped, base: &base })?;
        assert_eq!(opened.reuse, Reuse::Reinstalled);
        assert!(opened.changed.contains(&json!(["nodes","ball"])), "{opened:?}");
        // A value the schema refuses fails at its authored path and keeps the revision and
        // document; so does a file the install needs (programs are covered by the CLI test).
        let mut broken = brighter.clone();
        broken["nodes"][0]["energy"] = "bright".into();
        let error = worker.open(Source::Document { document: broken, base: &base }).unwrap_err();
        assert!(format!("{error:#}").contains("energy"), "{error:#}");
        let mut unreadable = brighter.clone();
        unreadable["buffers"] = json!({"positions":{"type":"vec3","file":"missing.bin","count":3}});
        unreadable["nodes"].as_array_mut().unwrap().push(json!({"name":"tri","type":"mesh","geometry":{"attributes":{"POSITION":"positions"}}}));
        let error = worker.open(Source::Document { document: unreadable, base: &base }).unwrap_err();
        assert!(format!("{error:#}").contains("missing.bin"), "{error:#}");
        assert_eq!(worker.revision(), Some(4));
        assert_eq!(worker.open.as_ref().unwrap().document["nodes"].as_array().unwrap().len(), 3);
        assert_eq!(worker.installs(), 0);
        worker.close();
        assert_eq!(worker.revision(), None);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn the_device_gains_what_a_later_scene_needs_and_keeps_it() -> Result<()> {
        let probe = crate::test_renderer()?;
        drop(probe);
        let base = std::env::temp_dir().join(format!("vork-worker-needs-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let canvas = json!({"version":2,"composition":{"space":"2d"},"render":{"resolution":[64,64]},
            "nodes":[{"name":"shape","type":"path","d":"M0 0H32V32H0Z","fill":"red"}]});
        let world = json!({"version":2,"render":{"resolution":[64,64]},
            "nodes":[{"name":"cam","type":"camera","translation":[0,0,5]},{"name":"ball","type":"mesh","geometry":{"std":"sphere","radius":0.5}}]});
        let shot = |name: &str| -> Result<Shot> {
            Ok(Shot { out: vec![base.join(name).into()], profile: Some(vork_scene::Profile::load("std/path")?), ..Default::default() })
        };
        let rays = |worker: &Worker| worker.renderer.as_ref().is_some_and(|renderer| renderer.provides(crate::DeviceNeeds::ALL));
        let mut worker = Worker::new();
        // A canvas opens and renders on a device without ray queries.
        worker.open(Source::Document { document: canvas.clone(), base: &base })?;
        assert!(worker.renderer.is_some() && !rays(&worker));
        let (report, _) = worker.render(shot("canvas.png")?)?;
        assert_eq!(report["device_reused"], true);
        assert!(!rays(&worker));
        // A world that does not install is refused on the device opened beside the canvas's: the canvas
        // keeps its device and its installation.
        let mut broken = world.clone();
        broken["nodes"][1]["materials"] = json!([{"program":{"source":"THIS IS NOT A VALID SLANG PROGRAM"}}]);
        assert!(worker.open(Source::Document { document: broken, base: &base }).is_err());
        assert!(!rays(&worker) && worker.open.as_ref().is_some_and(|open| open.installed.is_some()));
        let (report, _) = worker.render(shot("canvas.png")?)?;
        assert_eq!((&report["device_reused"], &report["install_reused"]), (&json!(true), &json!(true)));
        // A path-traced world needs them: its candidate is proven on a device that provides them, which
        // then replaces the canvas's.
        worker.open(Source::Document { document: world, base: &base })?;
        assert!(rays(&worker));
        let (report, _) = worker.render(shot("world.png")?)?;
        assert_eq!(report["device_reused"], true);
        // The canvas again: the device stays, and draws the picture the first device drew.
        worker.open(Source::Document { document: canvas, base: &base })?;
        let (report, _) = worker.render(shot("canvas-again.png")?)?;
        assert_eq!(report["device_reused"], true);
        assert!(rays(&worker));
        assert_eq!(image::open(base.join("canvas.png"))?.to_rgba8(), image::open(base.join("canvas-again.png"))?.to_rgba8());
        worker.close();
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
}
